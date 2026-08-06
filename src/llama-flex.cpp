#include "llama-flex.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <cinttypes>
#include <mutex>
#include <thread>
#include <unordered_map>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <cstdlib>
#endif

namespace {

uint64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

enum class layer_state { not_resident, loading, resident };

struct flex_layer {
    std::vector<llama_flex_tensor> tensors;
    size_t bytes        = 0;          // total tensor bytes for this layer
    size_t stream_bytes = 0;          // unlocked bytes streamed into a slot
    int    slot  = -1;                // ring slot currently holding it, or -1
    layer_state state = layer_state::not_resident;
    bool   released   = true;         // compute has consumed it (slot reusable)
    bool   always_resident = false;   // fully locked: no streaming needed
    uint64_t last_use = 0;
};

} // namespace

struct llama_flex_context {
    llama_flex_params params;
    llama_flex_stats  stats;
    int n_layers = 0;

    std::vector<int>         fds;          // one fd per file
    std::vector<size_t>      file_sizes;   // size of each file (for EOF clamping)
    std::vector<flex_layer>  layers;
    std::unordered_map<std::string, int> name_layer;  // tensor name -> layer id

    bool   direct_io_active = false;       // O_DIRECT actually in effect
    size_t align            = 4096;        // direct-IO alignment (offset/buf/len)
    size_t max_tensor       = 0;           // largest single tensor (bounce sizing)

    // per-graph compute state (written only by ith==0 in the stream callback)
    int cur_compute_layer = -1;
    uint64_t graph_id = 0;
    uint64_t last_layer_enter_us = 0;

    // ring of slots
    size_t                   slot_bytes = 0;
    std::vector<void *>      slots;
    std::vector<int>         slot_layer;   // which layer occupies slot, or -1

    // persistent balanced-lock buffer (resident for the whole run)
    void *                   lock_buf  = nullptr;
    size_t                   lock_size = 0;

    std::vector<std::thread> workers;
    std::deque<int>          queue;        // layer ids to stream
    std::mutex               mutex;
    std::condition_variable  cv_work;      // wakes IO threads
    std::condition_variable  cv_ready;     // wakes waiters on layer-ready
    bool                     shutdown = false;
    FILE *                   trace = nullptr;

    // Adaptive prefetch-depth controller. IO EWMA is updated by workers under
    // mutex; compute/wait EWMAs are updated by ith==0 on layer transitions.
    int      adaptive_ahead = 0;
    uint64_t adaptive_transitions = 0;
    uint64_t adaptive_quiet = 0;
    uint64_t adaptive_last_requeues = 0;
    double   ewma_io_us = 0.0;
    double   ewma_compute_us = 0.0;
    double   ewma_wait_us = 0.0;

    bool     prefetch_budget_enabled = false;
    uint64_t prefetch_budget_bytes = 0;
    uint64_t prefetch_budget_available_bytes = 0;

    ~llama_flex_context() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        cv_work.notify_all();
        for (auto & w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
        if (params.debug_log && stats.read_ops > 0) {
            const double phys_mib = stats.bytes_read_phys / 1048576.0;
            const double log_mib  = stats.bytes_streamed  / 1048576.0;
            const double io_s     = stats.total_io_us / 1e6;
            const double bw       = io_s > 0 ? phys_mib / io_s : 0.0;          // per-thread achieved MiB/s
            const double redun    = log_mib > 0 ? (phys_mib / log_mib - 1.0) * 100.0 : 0.0;
            const double avg_read = stats.read_ops > 0 ? phys_mib * 1024.0 / stats.read_ops : 0.0; // KiB/read
            const double wait_ms  = stats.total_wait_us / 1000.0;
            const double wait_avg = stats.wait_events > 0 ? wait_ms / (double) stats.wait_events : 0.0;
            std::fprintf(stderr,
                "llama_flex IO: loads=%llu reads=%llu avg_read=%.1f KiB  logical=%.0f MiB phys=%.0f MiB "
                "align_redundancy=%.2f%% achieved_bw=%.0f MiB/s waits=%llu wait=%.0f ms avg_wait=%.2f ms "
                "demand=%llu prefetch=%llu requeue=%llu evict=%llu release=%llu graphs=%llu ahead=%d "
                "ahead_adj=%llu ahead_range=[%d,%d] io_ewma=%.2f ms compute_ewma=%.2f ms\n",
                (unsigned long long) stats.layer_loads, (unsigned long long) stats.read_ops,
                avg_read, log_mib, phys_mib, redun, bw,
                (unsigned long long) stats.wait_events, wait_ms, wait_avg,
                (unsigned long long) stats.demand_loads,
                (unsigned long long) stats.prefetch_queued,
                (unsigned long long) stats.queue_requeues,
                (unsigned long long) stats.evictions,
                (unsigned long long) stats.releases,
                (unsigned long long) stats.graphs,
                stats.effective_ahead,
                (unsigned long long) stats.ahead_adjustments,
                stats.min_effective_ahead, stats.max_effective_ahead,
                ewma_io_us / 1000.0, ewma_compute_us / 1000.0);
        }
        if (trace) {
            std::fclose(trace);
        }
        for (void * p : slots) {
            free(p);
        }
        free(lock_buf);
        for (int fd : fds) {
            if (fd >= 0) {
                close(fd);
            }
        }
    }
};

static void flex_ewma(double & dst, double sample, double alpha = 0.2) {
    dst = dst == 0.0 ? sample : dst * (1.0 - alpha) + sample * alpha;
}

static int flex_ring_room_ahead(const llama_flex_context & ctx) {
    if (ctx.n_layers <= 1) {
        return 0;
    }
    int room = ctx.slots.empty() ? ctx.params.ring_layers : (int) ctx.slots.size();
    // Keep at least current + previous/released slot space when the ring is tiny.
    room = std::max(1, room - 2);
    return std::max(1, std::min(ctx.n_layers - 1, room));
}

static int flex_effective_ahead(const llama_flex_context & ctx) {
    if (ctx.n_layers <= 1) {
        return 0;
    }
    const int ring_room = flex_ring_room_ahead(ctx);
    const int requested = ctx.params.adaptive_ahead
            ? (ctx.adaptive_ahead > 0 ? ctx.adaptive_ahead : ctx.params.prefetch_ahead)
            : ctx.params.prefetch_ahead;
    const int max_ahead = ctx.params.adaptive_ahead
            ? std::max(ctx.params.prefetch_ahead, ctx.params.prefetch_ahead_max)
            : ctx.params.prefetch_ahead;
    return std::max(1, std::min({ requested, max_ahead, ring_room }));
}

static void flex_trace_locked(
        llama_flex_context & ctx,
        const char * event,
        int layer,
        int slot,
        size_t logical,
        size_t phys,
        uint64_t us);

static int flex_ring_occupancy_locked(const llama_flex_context & ctx) {
    int n = 0;
    for (int layer : ctx.slot_layer) {
        if (layer >= 0) {
            ++n;
        }
    }
    return n;
}

static void flex_note_ahead_locked(llama_flex_context & ctx) {
    const int ahead = flex_effective_ahead(ctx);
    ctx.stats.effective_ahead = ahead;
    if (ctx.stats.min_effective_ahead == 0 || ahead < ctx.stats.min_effective_ahead) {
        ctx.stats.min_effective_ahead = ahead;
    }
    if (ahead > ctx.stats.max_effective_ahead) {
        ctx.stats.max_effective_ahead = ahead;
    }
}

static void flex_adapt_after_layer(llama_flex_context & ctx, uint64_t wait_us) {
    if (!ctx.params.adaptive_ahead || ctx.n_layers <= 1) {
        return;
    }

    std::lock_guard<std::mutex> lock(ctx.mutex);
    const uint64_t now = now_us();
    if (ctx.last_layer_enter_us != 0) {
        const uint64_t interval = now - ctx.last_layer_enter_us;
        const uint64_t compute = interval > wait_us ? interval - wait_us : interval;
        flex_ewma(ctx.ewma_compute_us, (double) std::max<uint64_t>(compute, 1));
    }
    flex_ewma(ctx.ewma_wait_us, (double) wait_us);
    ctx.last_layer_enter_us = now;

    ctx.adaptive_transitions++;
    if ((ctx.adaptive_transitions % 4) != 0) {
        flex_note_ahead_locked(ctx);
        return;
    }

    const int old_ahead = flex_effective_ahead(ctx);
    int next_ahead = old_ahead;
    const int max_ahead = std::min(
            std::max(ctx.params.prefetch_ahead, ctx.params.prefetch_ahead_max),
            flex_ring_room_ahead(ctx));
    const int occupancy = flex_ring_occupancy_locked(ctx);
    const int slots = (int) ctx.slots.size();
    const uint64_t requeues = ctx.stats.queue_requeues - ctx.adaptive_last_requeues;
    ctx.adaptive_last_requeues = ctx.stats.queue_requeues;

    // A full ring is the normal steady state for dense sequential prefetching:
    // all slots should hold current/future layers. Treat it as backpressure only
    // when workers actually fail to acquire a reusable slot and requeue work.
    const bool ring_backpressure = requeues > 0;
    const bool stalled = wait_us > 1000 || ctx.ewma_wait_us > 1000.0;
    const bool io_lagging = ctx.ewma_compute_us > 0.0 && ctx.ewma_io_us > 0.0 &&
            ctx.ewma_io_us > ctx.ewma_compute_us * (double) std::max(1, old_ahead) * 1.10;

    if (ring_backpressure && old_ahead > 1) {
        next_ahead = old_ahead - 1;
        ctx.adaptive_quiet = 0;
    } else if ((stalled || io_lagging) && old_ahead < max_ahead) {
        next_ahead = old_ahead + 1;
        ctx.adaptive_quiet = 0;
    } else {
        const bool has_slack = ctx.ewma_compute_us > 0.0 && ctx.ewma_io_us > 0.0 &&
                old_ahead > 1 &&
                ctx.ewma_io_us * 1.40 < ctx.ewma_compute_us * (double) (old_ahead - 1);
        if (wait_us == 0 && has_slack) {
            ctx.adaptive_quiet++;
        } else {
            ctx.adaptive_quiet = 0;
        }
        if (ctx.adaptive_quiet >= 8) {
            next_ahead = old_ahead - 1;
            ctx.adaptive_quiet = 0;
        }
    }

    next_ahead = std::max(1, std::min(next_ahead, max_ahead));
    if (next_ahead != old_ahead) {
        ctx.adaptive_ahead = next_ahead;
        ctx.stats.ahead_adjustments++;
        flex_trace_locked(ctx, "adapt_ahead", ctx.cur_compute_layer, -1,
                (size_t) old_ahead, (size_t) next_ahead, wait_us);
        if (ctx.params.debug_log) {
            std::fprintf(stderr,
                    "llama_flex: adapt ahead %d -> %d (wait=%.2f ms io=%.2f ms compute=%.2f ms occ=%d/%d requeues=%llu)\n",
                    old_ahead, next_ahead, wait_us / 1000.0,
                    ctx.ewma_io_us / 1000.0, ctx.ewma_compute_us / 1000.0,
                    occupancy, slots, (unsigned long long) requeues);
        }
    }
    flex_note_ahead_locked(ctx);
}

static void flex_trace_locked(
        llama_flex_context & ctx,
        const char * event,
        int layer,
        int slot,
        size_t logical,
        size_t phys,
        uint64_t us) {
    if (ctx.trace == nullptr) {
        return;
    }
    std::fprintf(ctx.trace,
            "%" PRIu64 "\t%s\t%" PRIu64 "\t%d\t%d\t%zu\t%zu\t%" PRIu64 "\t%zu\t%llu\t%llu\t%llu\n",
            now_us(), event, ctx.graph_id, layer, slot, logical, phys, us,
            ctx.queue.size(),
            (unsigned long long) ctx.stats.layer_loads,
            (unsigned long long) ctx.stats.wait_events,
            (unsigned long long) ctx.stats.evictions);
}

static bool flex_name_has(const llama_flex_tensor & t, const char * needle) {
    return t.name.find(needle) != std::string::npos;
}

static int flex_pin_group(const llama_flex_tensor & t, const std::string & policy) {
    if (policy == "attn-first") {
        return flex_name_has(t, ".attn_") || flex_name_has(t, "attn_") ? 0 : 1;
    }
    if (policy == "ffn-first") {
        return flex_name_has(t, ".ffn_") || flex_name_has(t, "ffn_") ? 0 : 1;
    }
    return 0;
}

static size_t flex_aligned_read_bytes(const llama_flex_context & ctx, const llama_flex_tensor & t) {
    if (!ctx.direct_io_active) {
        return t.size;
    }
    const size_t A    = ctx.align;
    const size_t aoff = t.file_offset & ~(A - 1);
    const size_t head = t.file_offset - aoff;
    size_t bytes = head + t.size;
    bytes = (bytes + A - 1) & ~(A - 1);
    if (t.file_idx < ctx.file_sizes.size()) {
        const size_t fsz = ctx.file_sizes[t.file_idx];
        if (aoff < fsz && aoff + bytes > fsz) {
            bytes = fsz - aoff;
        }
    }
    return bytes;
}

static bool flex_policy_cost_aware(const std::string & policy) {
    return policy == "cost-aware" || policy == "cost-aware-balanced";
}

static size_t flex_pin_value_bytes(const llama_flex_context & ctx, const llama_flex_tensor & t) {
    return flex_aligned_read_bytes(ctx, t) + ctx.params.read_cost_bytes;
}

static void flex_sort_for_pin(std::vector<llama_flex_tensor> & tensors, const std::string & policy) {
    std::sort(tensors.begin(), tensors.end(),
            [&](const llama_flex_tensor & a, const llama_flex_tensor & b) {
                const int ga = flex_pin_group(a, policy);
                const int gb = flex_pin_group(b, policy);
                if (ga != gb) {
                    return ga < gb;
                }
                if (policy == "large-first" || policy == "ffn-first") {
                    if (a.size != b.size) {
                        return a.size > b.size;
                    }
                } else {
                    if (a.size != b.size) {
                        return a.size < b.size;
                    }
                }
                return a.name < b.name;
            });
}

static void flex_sort_for_pin(
        const llama_flex_context & ctx,
        std::vector<llama_flex_tensor> & tensors,
        const std::string & policy) {
    if (!flex_policy_cost_aware(policy)) {
        flex_sort_for_pin(tensors, policy);
        return;
    }

    std::sort(tensors.begin(), tensors.end(),
            [&](const llama_flex_tensor & a, const llama_flex_tensor & b) {
                const double score_a = a.size > 0
                        ? (double) flex_pin_value_bytes(ctx, a) / (double) a.size : 0.0;
                const double score_b = b.size > 0
                        ? (double) flex_pin_value_bytes(ctx, b) / (double) b.size : 0.0;
                if (score_a != score_b) {
                    return score_a > score_b;
                }
                const size_t benefit_a = flex_pin_value_bytes(ctx, a);
                const size_t benefit_b = flex_pin_value_bytes(ctx, b);
                if (benefit_a != benefit_b) {
                    return benefit_a > benefit_b;
                }
                if (a.size != b.size) {
                    return a.size < b.size;
                }
                return a.name < b.name;
            });
}

static std::vector<bool> flex_choose_cost_aware_pins(
        const llama_flex_context & ctx,
        std::vector<llama_flex_tensor> & tensors,
        size_t budget) {
    std::vector<bool> selected(tensors.size(), false);
    if (budget == 0 || tensors.empty()) {
        return selected;
    }

    // Dense layers normally have only a small handful of tensors. Exhaustive
    // per-layer 0/1 knapsack gives the best budget fill while preserving the
    // balanced per-layer budget. Fall back to the cost-aware greedy order for
    // unusual architectures with many tensors per layer.
    constexpr size_t max_exact_items = 22;
    if (tensors.size() > max_exact_items) {
        flex_sort_for_pin(ctx, tensors, "cost-aware");
        size_t used = 0;
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (used + tensors[i].size <= budget) {
                selected[i] = true;
                used += tensors[i].size;
            }
        }
        return selected;
    }

    const uint64_t n_mask = 1ull << tensors.size();
    uint64_t best_mask = 0;
    size_t best_value = 0;
    size_t best_weight = 0;
    int best_count = 0;

    for (uint64_t mask = 1; mask < n_mask; ++mask) {
        size_t value = 0;
        size_t weight = 0;
        int count = 0;
        bool ok = true;

        for (size_t i = 0; i < tensors.size(); ++i) {
            if ((mask & (1ull << i)) == 0) {
                continue;
            }
            weight += tensors[i].size;
            if (weight > budget) {
                ok = false;
                break;
            }
            value += flex_pin_value_bytes(ctx, tensors[i]);
            count++;
        }
        if (!ok) {
            continue;
        }

        if (value > best_value ||
                (value == best_value && weight > best_weight) ||
                (value == best_value && weight == best_weight && count > best_count)) {
            best_mask = mask;
            best_value = value;
            best_weight = weight;
            best_count = count;
        }
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
        selected[i] = (best_mask & (1ull << i)) != 0;
    }
    return selected;
}

struct flex_pin_state {
    uint64_t mask = 0;
    size_t used = 0;
    size_t value = 0;
    int count = 0;
};

static flex_pin_state flex_solve_layer_knapsack_state(
        const llama_flex_context & ctx,
        const std::vector<llama_flex_tensor> & tensors,
        size_t budget) {
    flex_pin_state best;
    if (budget == 0 || ctx.layers.empty()) {
        return best;
    }
    constexpr size_t max_exact_items = 22;
    if (tensors.size() > max_exact_items) {
        return best;
    }

    const uint64_t n_mask = 1ull << tensors.size();
    for (uint64_t mask = 1; mask < n_mask; ++mask) {
        flex_pin_state cur;
        cur.mask = mask;
        bool ok = true;
        for (size_t i = 0; i < tensors.size(); ++i) {
            if ((mask & (1ull << i)) == 0) {
                continue;
            }
            cur.used += tensors[i].size;
            if (cur.used > budget) {
                ok = false;
                break;
            }
            cur.value += flex_pin_value_bytes(ctx, tensors[i]);
            cur.count++;
        }
        if (!ok) {
            continue;
        }
        if (cur.value > best.value ||
                (cur.value == best.value && cur.used > best.used) ||
                (cur.value == best.value && cur.used == best.used && cur.count > best.count)) {
            best = cur;
        }
    }
    return best;
}

static void flex_apply_state_to_layer(flex_layer & L, const flex_pin_state & state) {
    for (size_t i = 0; i < L.tensors.size(); ++i) {
        L.tensors[i].locked = (state.mask & (1ull << i)) != 0;
    }
}

static flex_pin_state flex_current_layer_state(
        const llama_flex_context & ctx,
        const std::vector<llama_flex_tensor> & tensors) {
    flex_pin_state state;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (!tensors[i].locked) {
            continue;
        }
        state.mask |= 1ull << i;
        state.used += tensors[i].size;
        state.value += flex_pin_value_bytes(ctx, tensors[i]);
        state.count++;
    }
    return state;
}

static size_t flex_layer_total_value(
        const llama_flex_context & ctx,
        const std::vector<llama_flex_tensor> & tensors) {
    size_t value = 0;
    for (const auto & t : tensors) {
        value += flex_pin_value_bytes(ctx, t);
    }
    return value;
}

static void flex_apply_global_rebalance(llama_flex_context & ctx, size_t budget) {
    if (budget == 0 || ctx.layers.empty()) {
        return;
    }

    std::vector<flex_pin_state> states(ctx.layers.size());
    std::vector<size_t> layer_total_value(ctx.layers.size(), 0);
    size_t base_locked = 0;
    int base_locked_count = 0;
    for (size_t il = 0; il < ctx.layers.size(); ++il) {
        states[il] = flex_current_layer_state(ctx, ctx.layers[il].tensors);
        layer_total_value[il] = flex_layer_total_value(ctx, ctx.layers[il].tensors);
        base_locked += states[il].used;
        base_locked_count += states[il].count;
    }

    struct proposal {
        int layer = -1;
        flex_pin_state state;
        size_t extra = 0;
        size_t gain = 0;
        double score = 0.0;
    };

    while (budget > 0) {
        std::vector<size_t> layer_stream_cost(ctx.layers.size(), 0);
        size_t total_stream_cost = 0;
        for (size_t il = 0; il < ctx.layers.size(); ++il) {
            layer_stream_cost[il] = layer_total_value[il] > states[il].value
                    ? layer_total_value[il] - states[il].value : 0;
            total_stream_cost += layer_stream_cost[il];
        }
        if (total_stream_cost == 0) {
            break;
        }
        const double avg_stream_cost = (double) total_stream_cost / (double) ctx.layers.size();

        proposal best;
        for (size_t il = 0; il < ctx.layers.size(); ++il) {
            if ((double) layer_stream_cost[il] < avg_stream_cost) {
                continue;
            }
            const auto & tensors = ctx.layers[il].tensors;
            if (tensors.size() > 22) {
                continue;
            }
            const double pressure = avg_stream_cost > 0.0
                    ? (double) layer_stream_cost[il] / avg_stream_cost : 1.0;
            const uint64_t n_mask = 1ull << tensors.size();
            for (uint64_t mask = 1; mask < n_mask; ++mask) {
                flex_pin_state cand;
                cand.mask = mask;
                bool ok = true;
                for (size_t i = 0; i < tensors.size(); ++i) {
                    if ((mask & (1ull << i)) == 0) {
                        continue;
                    }
                    cand.used += tensors[i].size;
                    if (cand.used > states[il].used + budget) {
                        ok = false;
                        break;
                    }
                    cand.value += flex_pin_value_bytes(ctx, tensors[i]);
                    cand.count++;
                }
                if (!ok || cand.used <= states[il].used || cand.value <= states[il].value) {
                    continue;
                }
                const size_t extra = cand.used - states[il].used;
                const size_t gain = cand.value - states[il].value;
                const double score = extra > 0 ? pressure * (double) gain / (double) extra : 0.0;
                if (best.layer < 0 ||
                        score > best.score ||
                        (score == best.score && gain > best.gain) ||
                        (score == best.score && gain == best.gain && extra < best.extra) ||
                        (score == best.score && gain == best.gain && extra == best.extra && cand.used > best.state.used) ||
                        (score == best.score && gain == best.gain && extra == best.extra && cand.used == best.state.used && (int) il < best.layer)) {
                    best = { (int) il, cand, extra, gain, score };
                }
            }
        }
        if (best.layer < 0) {
            break;
        }
        states[best.layer] = best.state;
        budget -= best.extra;
    }

    size_t final_locked = 0;
    int final_locked_count = 0;
    for (size_t il = 0; il < ctx.layers.size(); ++il) {
        flex_apply_state_to_layer(ctx.layers[il], states[il]);
        final_locked += states[il].used;
        final_locked_count += states[il].count;
    }
    ctx.stats.global_rebalance_bytes = final_locked > base_locked ? final_locked - base_locked : 0;
    ctx.stats.global_rebalance_tensors = final_locked_count > base_locked_count
            ? (uint64_t) (final_locked_count - base_locked_count) : 0;
    if (ctx.params.debug_log && ctx.stats.global_rebalance_bytes > 0) {
        std::fprintf(stderr,
                "llama_flex: global water-fill pinned %llu tensors, %.2f MiB\n",
                (unsigned long long) ctx.stats.global_rebalance_tensors,
                ctx.stats.global_rebalance_bytes / 1048576.0);
    }
}

static bool flex_pin_policy_valid(const std::string & policy) {
    return policy == "small-first" ||
           policy == "large-first" ||
           policy == "attn-first"  ||
           policy == "ffn-first"   ||
           policy == "cost-aware"  ||
           policy == "cost-aware-balanced" ||
           policy == "none";
}

// Pick a slot for `layer`: a free slot, else evict the least-recently-used
// resident-and-released layer. Returns slot index or -1 if none available.
// Must hold ctx.mutex.
static int flex_acquire_slot(llama_flex_context & ctx, int layer) {
    for (size_t s = 0; s < ctx.slots.size(); ++s) {
        if (ctx.slot_layer[s] < 0) {
            ctx.slot_layer[s] = layer;
            return (int) s;
        }
    }
    // No free slot: evict the LRU released layer.
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int l = 0; l < ctx.n_layers; ++l) {
        auto & L = ctx.layers[l];
        if (L.slot >= 0 && L.released && L.state == layer_state::resident &&
                L.last_use < oldest) {
            oldest = L.last_use;
            victim = l;
        }
    }
    if (victim < 0) {
        return -1;
    }
    int slot = ctx.layers[victim].slot;
    ctx.layers[victim].slot  = -1;
    ctx.layers[victim].state = layer_state::not_resident;
    ctx.slot_layer[slot]     = layer;
    ctx.stats.evictions++;
    flex_trace_locked(ctx, "evict", victim, slot, 0, 0, 0);
    return slot;
}

// Read `size` bytes at `foff` from file `file_idx` into `dst`. When O_DIRECT is
// active, the read is issued on a block-aligned superset into the per-thread
// `bounce` buffer and the exact bytes are copied out; otherwise a plain pread
// loop is used. `bcap` is the bounce capacity. Returns true on success.
static bool flex_read(llama_flex_context * ctx,
                      uint8_t * dst, uint16_t file_idx, size_t foff, size_t size,
                      uint8_t * bounce, size_t bcap, size_t * phys_out = nullptr) {
    const int fd = ctx->fds[file_idx];
    if (!ctx->direct_io_active) {
        size_t left = size; off_t off = (off_t) foff; uint8_t * d = dst;
        while (left > 0) {
            ssize_t r = pread(fd, d, left, off);
            if (r <= 0) return false;
            d += r; off += r; left -= (size_t) r;
        }
        if (phys_out) *phys_out = size;
        return true;
    }
    const size_t A    = ctx->align;
    const size_t aoff = foff & ~(A - 1);
    const size_t head = foff - aoff;
    size_t alen = head + size;
    alen = (alen + A - 1) & ~(A - 1);
    const size_t fsz  = ctx->file_sizes[file_idx];
    size_t want = alen;
    if (aoff + want > fsz) {
        want = fsz - aoff;            // final read may be a short EOF block
    }
    if (head + size > bcap || want > bcap) {
        return false;
    }
    ssize_t r = pread(fd, bounce, want, (off_t) aoff);
    if (r < 0 || (size_t) r < head + size) {
        return false;
    }
    std::memcpy(dst, bounce + head, size);
    if (phys_out) *phys_out = want;
    return true;
}

static double flex_calibration_avg_pread_us(int fd, void * buf, size_t size, off_t offset, int repeats) {
    if (repeats <= 0 || size == 0) {
        return 0.0;
    }
    uint64_t total_us = 0;
    for (int i = 0; i < repeats; ++i) {
        const uint64_t t0 = now_us();
        ssize_t r = pread(fd, buf, size, offset);
        const uint64_t dt = now_us() - t0;
        if (r < 0 || (size_t) r < size) {
            return 0.0;
        }
        total_us += std::max<uint64_t>(dt, 1);
    }
    return (double) total_us / (double) repeats;
}

static size_t flex_calibrate_read_cost_bytes(llama_flex_context & ctx) {
    if (!ctx.direct_io_active || ctx.fds.empty() || ctx.file_sizes.empty()) {
        return 0;
    }

    const size_t A = ctx.align;
    const size_t fsz = ctx.file_sizes[0];
    if (fsz < 2 * A) {
        return 0;
    }

    const size_t small = A;
    size_t large = 16ull * 1024 * 1024;
    large = std::min(large, fsz & ~(A - 1));
    if (large <= small) {
        return 0;
    }

    void * small_buf = nullptr;
    void * large_buf = nullptr;
    if (posix_memalign(&small_buf, A, small) != 0 || posix_memalign(&large_buf, A, large) != 0) {
        free(small_buf);
        free(large_buf);
        return 0;
    }

    const int fd = ctx.fds[0];
    const double small_us = flex_calibration_avg_pread_us(fd, small_buf, small, 0, 32);
    const double large_us = flex_calibration_avg_pread_us(fd, large_buf, large, 0, 4);
    free(small_buf);
    free(large_buf);

    if (small_us <= 0.0 || large_us <= small_us) {
        return 0;
    }

    const double bw_bytes_per_us = ((double) large - (double) small) / (large_us - small_us);
    if (bw_bytes_per_us <= 0.0) {
        return 0;
    }
    const double fixed_us = small_us - (double) small / bw_bytes_per_us;
    if (fixed_us <= 0.0) {
        return 0;
    }

    const double fixed_bytes = fixed_us * bw_bytes_per_us;
    const size_t cap = 4ull * 1024 * 1024;
    const size_t result = (size_t) std::min<double>(fixed_bytes, (double) cap);
    if (ctx.params.debug_log) {
        std::fprintf(stderr,
                "llama_flex: read-cost calibration small=%.2f us large=%.2f us bw=%.0f MiB/s fixed=%.2f us read_cost=%.1f KiB\n",
                small_us, large_us, bw_bytes_per_us * 1000000.0 / 1048576.0,
                fixed_us, result / 1024.0);
    }
    return result;
}

static void flex_worker(llama_flex_context * ctx) {
    // Per-thread aligned bounce buffer for O_DIRECT reads.
    uint8_t * bounce = nullptr;
    size_t    bcap   = 0;
    if (ctx->direct_io_active) {
        bcap = ctx->max_tensor + 2 * ctx->align;
        if (posix_memalign((void **) &bounce, ctx->align, bcap) != 0) {
            bounce = nullptr; bcap = 0;
        }
    }

    for (;;) {
        int layer = -1;
        {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            ctx->cv_work.wait(lock, [&] {
                return ctx->shutdown || !ctx->queue.empty();
            });
            if (ctx->shutdown && ctx->queue.empty()) {
                free(bounce);
                return;
            }
            layer = ctx->queue.front();
            ctx->queue.pop_front();

            auto & L = ctx->layers[layer];
            if (L.state == layer_state::resident || L.state == layer_state::loading) {
                continue; // already handled
            }
            int slot = flex_acquire_slot(*ctx, layer);
            if (slot < 0) {
                // No slot available right now; requeue and back off.
                ctx->stats.queue_requeues++;
                flex_trace_locked(*ctx, "requeue", layer, -1, 0, 0, 0);
                ctx->queue.push_back(layer);
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            L.slot     = slot;
            L.state    = layer_state::loading;
            L.released = false;
        }

        // Stream tensors into the slot (outside the lock).
        auto & L = ctx->layers[layer];
        uint8_t * base = (uint8_t *) ctx->slots[L.slot];
        const uint64_t t0 = now_us();
        bool ok = true;
        size_t streamed = 0;
        size_t phys     = 0;
        size_t ops      = 0;
        for (const auto & t : L.tensors) {
            if (t.locked) {
                continue; // locked tensors live permanently in the lock buffer
            }
            size_t p = 0;
            if (!flex_read(ctx, base + t.buf_offset, t.file_idx, t.file_offset, t.size, bounce, bcap, &p)) {
                ok = false;
                break;
            }
            streamed += t.size;
            phys     += p;
            ops      += 1;
        }
        const uint64_t dt = now_us() - t0;

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            if (ok) {
                L.state    = layer_state::resident;
                L.last_use = now_us();
                ctx->stats.layer_loads++;
                ctx->stats.bytes_streamed  += streamed;
                ctx->stats.bytes_read_phys += phys;
                ctx->stats.read_ops        += ops;
                ctx->stats.total_io_us     += dt;
                flex_ewma(ctx->ewma_io_us, (double) std::max<uint64_t>(dt, 1));
                flex_trace_locked(*ctx, "load", layer, L.slot, streamed, phys, dt);
            } else {
                // Failed: drop the slot back.
                ctx->slot_layer[L.slot] = -1;
                L.slot  = -1;
                L.state = layer_state::not_resident;
                flex_trace_locked(*ctx, "load_fail", layer, -1, streamed, phys, dt);
                if (ctx->params.debug_log) {
                    std::fprintf(stderr, "llama_flex: stream failed for layer %d\n", layer);
                }
            }
        }
        ctx->cv_ready.notify_all();
    }
}

std::shared_ptr<llama_flex_context> llama_flex_create(
        const std::vector<int> &  fds,
        int                       n_layers,
        const llama_flex_params & params) {
    auto ctx = std::make_shared<llama_flex_context>();
    ctx->params   = params;
    ctx->n_layers = n_layers;
    ctx->layers.resize(std::max(0, n_layers));
    if (const char * v = std::getenv("LLAMA_FLEX_PIN_POLICY")) {
        ctx->params.pin_policy = v;
    }
    if (const char * v = std::getenv("LLAMA_FLEX_READ_COST_KB")) {
        if (std::strcmp(v, "auto") == 0 || std::strcmp(v, "AUTO") == 0) {
            ctx->params.read_cost_auto = true;
        } else {
            const double kb = std::max(0.0, std::atof(v));
            ctx->params.read_cost_bytes = (size_t) (kb * 1024.0);
        }
    }
    if (const char * v = std::getenv("LLAMA_FLEX_READ_COST_AUTO")) {
        ctx->params.read_cost_auto = std::atoi(v) > 0;
    }
    if (const char * v = std::getenv("LLAMA_FLEX_GLOBAL_REBALANCE")) {
        ctx->params.global_rebalance = std::atoi(v) > 0;
    }
    if (!flex_pin_policy_valid(ctx->params.pin_policy)) {
        if (ctx->params.debug_log) {
            std::fprintf(stderr, "llama_flex: invalid pin policy '%s', using small-first\n",
                    ctx->params.pin_policy.c_str());
        }
        ctx->params.pin_policy = "small-first";
    }

    if (!params.enabled || n_layers <= 0) {
        return ctx;
    }

    bool all_direct = params.direct_io;
    for (int src_fd : fds) {
        // Reopen via /proc/self/fd to get an independent file position and,
        // optionally, an O_DIRECT description. Fall back to a plain dup().
        int fd = -1;
        bool got_direct = false;
#if defined(__linux__)
        char proc[64];
        std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", src_fd);
        int flags = O_RDONLY;
#if defined(O_DIRECT)
        if (params.direct_io) {
            fd = open(proc, flags | O_DIRECT);
            if (fd >= 0) {
                got_direct = true;
            }
        }
#endif
        if (fd < 0) {
            fd = open(proc, flags);
        }
#endif
        if (fd < 0) {
            fd = dup(src_fd);
        }
        if (fd < 0) {
            if (params.debug_log) {
                std::fprintf(stderr, "llama_flex: failed to reopen fd %d\n", src_fd);
            }
            ctx->params.enabled = false;
            return ctx;
        }
        if (!got_direct) {
            all_direct = false;
        }
        struct stat st;
        ctx->file_sizes.push_back(fstat(fd, &st) == 0 ? (size_t) st.st_size : SIZE_MAX);
        ctx->fds.push_back(fd);
    }
    ctx->direct_io_active = all_direct;
    if (const char * path = std::getenv("LLAMA_FLEX_TRACE")) {
        ctx->trace = std::fopen(path, "w");
        if (ctx->trace != nullptr) {
            std::fprintf(ctx->trace,
                    "time_us\tevent\tgraph\tlayer\tslot\tlogical_bytes\tphys_bytes\telapsed_us\tqueue_depth\tloads\twaits\tevictions\n");
        } else if (params.debug_log) {
            std::fprintf(stderr, "llama_flex: failed to open trace %s\n", path);
        }
    }

    return ctx;
}

bool llama_flex_enabled(const llama_flex_context * ctx) {
    return ctx != nullptr && ctx->params.enabled && ctx->n_layers > 0;
}

void llama_flex_register_tensor(
        llama_flex_context & ctx,
        int                  layer_id,
        const llama_flex_tensor & tensor) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return;
    }
    auto & L = ctx.layers[layer_id];
    llama_flex_tensor t = tensor;
    t.locked     = false;
    t.buf_offset = 0; // assigned in finalize once locking is decided
    L.bytes += t.size;
    ctx.max_tensor = std::max(ctx.max_tensor, t.size);
    ctx.name_layer[t.name] = layer_id;
    L.tensors.push_back(std::move(t));
}

void llama_flex_finalize(llama_flex_context & ctx) {
    if (!llama_flex_enabled(&ctx)) {
        return;
    }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    if (ctx.params.read_cost_auto) {
        ctx.params.read_cost_bytes = flex_calibrate_read_cost_bytes(ctx);
    }

    // Balanced memory locking: give every layer the same lock budget so the
    // per-layer streamed IO stays uniform (avoids pipeline stalls). Within a
    // layer, lock tensors in registration order until the budget is hit; since
    // all layers share the same tensor structure this locks the same set per
    // layer. The remaining (unlocked) tensors are streamed through the ring.
    const bool pin_disabled = ctx.params.pin_policy == "none";
    const size_t per_layer_lock = !pin_disabled && ctx.n_layers > 0
            ? ctx.params.lock_bytes / (size_t) ctx.n_layers : 0;

    size_t balanced_locked_total = 0;
    for (auto & L : ctx.layers) {
        // Balanced pinning keeps the same byte budget per layer so streamed IO
        // remains uniform. Policies only change tensor order within that budget.
        std::vector<bool> cost_aware_pins;
        if (flex_policy_cost_aware(ctx.params.pin_policy)) {
            cost_aware_pins = flex_choose_cost_aware_pins(ctx, L.tensors, per_layer_lock);
        } else {
            flex_sort_for_pin(ctx, L.tensors, ctx.params.pin_policy);
        }
        size_t locked_here = 0;
        for (size_t i = 0; i < L.tensors.size(); ++i) {
            auto & t = L.tensors[i];
            const bool should_lock = flex_policy_cost_aware(ctx.params.pin_policy)
                    ? cost_aware_pins[i] : locked_here + t.size <= per_layer_lock;
            if (should_lock) {
                t.locked     = true;
                locked_here += t.size;
            } else {
                t.locked     = false;
            }
        }
        if (per_layer_lock > locked_here) {
            ctx.stats.lock_budget_unused += per_layer_lock - locked_here;
        }
        balanced_locked_total += locked_here;
    }

    if (ctx.params.global_rebalance && flex_policy_cost_aware(ctx.params.pin_policy) &&
            ctx.params.lock_bytes > balanced_locked_total) {
        flex_apply_global_rebalance(ctx, ctx.params.lock_bytes - balanced_locked_total);
    }

    size_t lock_total = 0;
    ctx.stats.locked_tensors = 0;
    ctx.stats.streamed_tensors = 0;
    ctx.stats.lock_budget_unused = 0;
    ctx.stats.stream_per_token = 0;
    for (auto & L : ctx.layers) {
        size_t stream_off = 0;
        size_t locked_here = 0;
        for (auto & t : L.tensors) {
            if (t.locked) {
                t.buf_offset = lock_total;   // offset into the global lock buffer
                lock_total  += t.size;
                locked_here += t.size;
                ctx.stats.locked_tensors++;
            } else {
                t.buf_offset = stream_off;   // offset within this layer's slot
                stream_off  += t.size;
                ctx.stats.streamed_tensors++;
            }
        }
        L.stream_bytes    = stream_off;
        L.always_resident = (stream_off == 0);
    }
    if (ctx.params.lock_bytes > lock_total) {
        ctx.stats.lock_budget_unused = ctx.params.lock_bytes - lock_total;
    }

    size_t max_bytes = 0;
    for (const auto & L : ctx.layers) {
        max_bytes = std::max(max_bytes, L.stream_bytes);
    }
    // Round the slot up to a page so it can back O_DIRECT later.
    ctx.slot_bytes = max_bytes > 0 ? ((max_bytes + page - 1) / page) * page : page;

    // Allocate and fill the persistent lock buffer.
    if (lock_total > 0) {
        if (posix_memalign(&ctx.lock_buf, page, lock_total) != 0 || ctx.lock_buf == nullptr) {
            ctx.params.enabled = false;
            if (ctx.params.debug_log) {
                std::fprintf(stderr, "llama_flex: lock buffer alloc failed (%zu bytes)\n", lock_total);
            }
            return;
        }
        ctx.lock_size = lock_total;
        // Temp aligned bounce for direct-IO lock fill (workers not started yet).
        uint8_t * bounce = nullptr;
        size_t    bcap   = 0;
        if (ctx.direct_io_active) {
            bcap = ctx.max_tensor + 2 * ctx.align;
            if (posix_memalign((void **) &bounce, ctx.align, bcap) != 0) {
                bounce = nullptr; bcap = 0;
            }
        }
        for (auto & L : ctx.layers) {
            for (const auto & t : L.tensors) {
                if (!t.locked) {
                    continue;
                }
                if (!flex_read(&ctx, (uint8_t *) ctx.lock_buf + t.buf_offset,
                               t.file_idx, t.file_offset, t.size, bounce, bcap)) {
                    ctx.params.enabled = false;
                }
            }
        }
        free(bounce);
        ctx.stats.locked_bytes = lock_total;
    }
    for (const auto & L : ctx.layers) {
        ctx.stats.stream_per_token += L.stream_bytes;
    }

    if (ctx.params.sched_auto && ctx.params.memory_budget_bytes > 0 && ctx.slot_bytes > 0) {
        const size_t used_fixed = ctx.params.fixed_bytes + ctx.stats.locked_bytes;
        const size_t room = ctx.params.memory_budget_bytes > used_fixed
                ? ctx.params.memory_budget_bytes - used_fixed : 0;
        int auto_k = (int) std::min<size_t>(ctx.n_layers,
                std::max<size_t>(1, room / ctx.slot_bytes));
        const int min_k = std::min(ctx.n_layers, std::max(2, ctx.params.prefetch_ahead + 2));
        auto_k = std::max(auto_k, min_k);
        ctx.params.ring_layers = auto_k;
        ctx.stats.sched_budget_bytes = ctx.params.memory_budget_bytes;
        ctx.stats.sched_fixed_bytes  = ctx.params.fixed_bytes;
        ctx.stats.sched_ring_room    = room;
    }
    const int k = std::max(1, std::min(ctx.params.ring_layers, ctx.n_layers));
    ctx.slots.resize(k, nullptr);
    ctx.slot_layer.assign(k, -1);
    for (int s = 0; s < k; ++s) {
        void * p = nullptr;
        if (posix_memalign(&p, page, ctx.slot_bytes) != 0 || p == nullptr) {
            ctx.params.enabled = false;
            if (ctx.params.debug_log) {
                std::fprintf(stderr, "llama_flex: slot alloc failed (%zu bytes)\n", ctx.slot_bytes);
            }
            return;
        }
        ctx.slots[s] = p;
    }
    ctx.stats.ring_bytes = ctx.slot_bytes * (size_t) k;
    ctx.stats.slot_bytes = ctx.slot_bytes;
    ctx.stats.read_cost_bytes = ctx.params.read_cost_bytes;
    ctx.adaptive_ahead = std::max(1, std::min(ctx.params.prefetch_ahead, flex_ring_room_ahead(ctx)));
    ctx.stats.effective_ahead = flex_effective_ahead(ctx);
    ctx.stats.min_effective_ahead = ctx.stats.effective_ahead;
    ctx.stats.max_effective_ahead = ctx.stats.effective_ahead;

    // Fully-locked layers never need streaming: mark them permanently resident.
    for (auto & L : ctx.layers) {
        if (L.always_resident) {
            L.state    = layer_state::resident;
            L.released = false;
        }
    }

    const int nthreads = std::max(1, std::min(ctx.params.io_threads, 8));
    for (int i = 0; i < nthreads; ++i) {
        ctx.workers.emplace_back(flex_worker, &ctx);
    }

    if (ctx.params.debug_log) {
        std::fprintf(stderr,
                "llama_flex: layers=%d ring=%d slot=%.2f MiB ring_total=%.2f MiB "
                "locked=%.2f MiB stream/token=%.2f MiB io_threads=%d direct_io=%d ahead=%d requested_ahead=%d "
                "adaptive_ahead=%d max_ahead=%d pin_policy=%s read_cost=%.1f KiB locked_tensors=%llu streamed_tensors=%llu lock_unused=%.2f MiB "
                "global_rebalance=%d global_locked=%.2f MiB global_tensors=%llu "
                "sched=%d budget=%.0f MiB fixed=%.0f MiB ring_room=%.0f MiB\n",
                ctx.n_layers, k, ctx.slot_bytes / 1048576.0,
                ctx.stats.ring_bytes / 1048576.0,
                ctx.stats.locked_bytes / 1048576.0,
                ctx.stats.stream_per_token / 1048576.0,
                nthreads, ctx.direct_io_active ? 1 : 0,
                ctx.stats.effective_ahead, ctx.params.prefetch_ahead,
                ctx.params.adaptive_ahead ? 1 : 0, ctx.params.prefetch_ahead_max,
                ctx.params.pin_policy.c_str(),
                ctx.params.read_cost_bytes / 1024.0,
                (unsigned long long) ctx.stats.locked_tensors,
                (unsigned long long) ctx.stats.streamed_tensors,
                ctx.stats.lock_budget_unused / 1048576.0,
                ctx.params.global_rebalance ? 1 : 0,
                ctx.stats.global_rebalance_bytes / 1048576.0,
                (unsigned long long) ctx.stats.global_rebalance_tensors,
                ctx.params.sched_auto ? 1 : 0,
                ctx.stats.sched_budget_bytes / 1048576.0,
                ctx.stats.sched_fixed_bytes / 1048576.0,
                ctx.stats.sched_ring_room / 1048576.0);
    }
}

static void flex_request_layer(llama_flex_context & ctx, int layer_id, bool prefetch) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    if (L.state == layer_state::resident) {
        ctx.stats.layer_hits++;
        return;
    }
    if (L.state == layer_state::loading) {
        return;
    }
    if (prefetch && ctx.prefetch_budget_enabled) {
        const uint64_t charge = (uint64_t) ctx.slot_bytes;
        if (charge > ctx.prefetch_budget_available_bytes) {
            ctx.stats.prefetch_budget_dropped++;
            flex_trace_locked(ctx, "prefetch_budget_drop", layer_id, -1,
                    charge, ctx.prefetch_budget_available_bytes, 0);
            return;
        }
        ctx.prefetch_budget_available_bytes -= charge;
        ctx.stats.prefetch_budget_available_bytes =
            (size_t) ctx.prefetch_budget_available_bytes;
    }
    if (prefetch) {
        ctx.stats.prefetch_queued++;
    } else {
        ctx.stats.demand_loads++;
    }
    flex_trace_locked(ctx, prefetch ? "prefetch" : "demand", layer_id, -1, 0, 0, 0);
    ctx.queue.push_back(layer_id);
    ctx.cv_work.notify_one();
}

void llama_flex_request_layer(llama_flex_context & ctx, int layer_id) {
    flex_request_layer(ctx, layer_id, true);
}

static uint64_t flex_wait_layer_us(llama_flex_context & ctx, int layer_id) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return 0;
    }
    std::unique_lock<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    if (L.state == layer_state::resident) {
        return 0;
    }
    // Make sure it is at least queued.
    if (L.state == layer_state::not_resident) {
        ctx.stats.demand_loads++;
        flex_trace_locked(ctx, "demand_front", layer_id, -1, 0, 0, 0);
        ctx.queue.push_front(layer_id);
        ctx.cv_work.notify_one();
    }
    const uint64_t t0 = now_us();
    ctx.stats.wait_events++;
    ctx.cv_ready.wait(lock, [&] {
        return L.state == layer_state::resident || ctx.shutdown;
    });
    const uint64_t wait_us = now_us() - t0;
    ctx.stats.total_wait_us += wait_us;
    flex_trace_locked(ctx, "wait", layer_id, L.slot, 0, 0, wait_us);
    return wait_us;
}

void llama_flex_wait_layer(llama_flex_context & ctx, int layer_id) {
    (void) flex_wait_layer_us(ctx, layer_id);
}

void * llama_flex_get_tensor(llama_flex_context & ctx, int layer_id, const std::string & name) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    for (const auto & t : L.tensors) {
        if (t.name != name) {
            continue;
        }
        if (t.locked) {
            return (uint8_t *) ctx.lock_buf + t.buf_offset; // always resident
        }
        if (L.state != layer_state::resident || L.slot < 0) {
            return nullptr; // streamed tensor, layer not in a slot yet
        }
        return (uint8_t *) ctx.slots[L.slot] + t.buf_offset;
    }
    return nullptr;
}

void llama_flex_release_layer(llama_flex_context & ctx, int layer_id) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    L.released = true;
    L.last_use = now_us();
    ctx.stats.releases++;
    flex_trace_locked(ctx, "release", layer_id, L.slot, 0, 0, 0);
}

const llama_flex_stats & llama_flex_get_stats(const llama_flex_context & ctx) {
    return ctx.stats;
}

void llama_flex_set_prefetch_budget(
        llama_flex_context & ctx,
        uint64_t             budget_bytes) {
    if (!llama_flex_enabled(&ctx)) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.prefetch_budget_enabled = budget_bytes > 0;
    ctx.prefetch_budget_bytes = budget_bytes;
    ctx.prefetch_budget_available_bytes = budget_bytes;
    ctx.stats.prefetch_budget_bytes = (size_t) budget_bytes;
    ctx.stats.prefetch_budget_available_bytes = (size_t) budget_bytes;
}

llama_flex_reclaim_result llama_flex_reclaim_released(
        llama_flex_context & ctx,
        uint64_t             target_bytes,
        uint32_t             max_layers) {
    llama_flex_reclaim_result result;
    if (!llama_flex_enabled(&ctx) || target_bytes == 0 || max_layers == 0) {
        return result;
    }

    std::lock_guard<std::mutex> lock(ctx.mutex);
    while (result.released_bytes < target_bytes && result.released_layers < max_layers) {
        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int l = 0; l < ctx.n_layers; ++l) {
            const auto & L = ctx.layers[l];
            if (l == ctx.cur_compute_layer || L.always_resident ||
                    L.slot < 0 || !L.released || L.state != layer_state::resident) {
                continue;
            }
            if (L.last_use < oldest) {
                oldest = L.last_use;
                victim = l;
            }
        }
        if (victim < 0) {
            break;
        }

        auto & L = ctx.layers[victim];
        const int slot = L.slot;
        if (slot >= 0 && slot < (int) ctx.slots.size() && ctx.slots[slot] != nullptr) {
#if defined(MADV_DONTNEED)
            madvise(ctx.slots[slot], ctx.slot_bytes, MADV_DONTNEED);
#endif
            ctx.slot_layer[slot] = -1;
        }
        L.slot = -1;
        L.state = layer_state::not_resident;
        L.released = true;
        L.last_use = now_us();
        ctx.stats.evictions++;
        result.released_bytes += (uint64_t) ctx.slot_bytes;
        result.released_layers++;
        flex_trace_locked(ctx, "governor_reclaim", victim, slot, ctx.slot_bytes, 0, 0);
    }
    result.target_satisfied = result.released_bytes >= target_bytes;
    return result;
}

void llama_flex_graph_begin(llama_flex_context & ctx) {
    if (!llama_flex_enabled(&ctx)) {
        return;
    }
    if (ctx.cur_compute_layer >= 0) {
        llama_flex_release_layer(ctx, ctx.cur_compute_layer);
    }
    ctx.cur_compute_layer = -1;
    ctx.last_layer_enter_us = 0;
    ctx.graph_id++;
    ctx.stats.graphs++;
    const int ahead = flex_effective_ahead(ctx);
    ctx.stats.effective_ahead = ahead;
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        flex_trace_locked(ctx, "graph_begin", -1, -1, 0, 0, 0);
    }
    for (int l = 0; l <= ahead; ++l) {
        flex_request_layer(ctx, l % ctx.n_layers, true);
    }
}

bool llama_flex_stream_callback(struct ggml_tensor * op, int ith, void * user_data) {
    auto * ctx = static_cast<llama_flex_context *>(user_data);
    if (ctx == nullptr || op == nullptr) {
        return false;
    }

    int op_layer = -1;
    uint64_t wait_us_total = 0;
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        ggml_tensor * s = op->src[i];
        if (s == nullptr || s->name[0] == '\0') {
            continue;
        }
        auto it = ctx->name_layer.find(s->name);
        if (it == ctx->name_layer.end()) {
            continue;
        }
        op_layer = it->second;
        if (ith == 0) {
            const uint64_t wait_us = flex_wait_layer_us(*ctx, op_layer);
            wait_us_total += wait_us;
            void * p = llama_flex_get_tensor(*ctx, op_layer, s->name);
            if (p != nullptr) {
                s->data = p;
            }
        }
    }

    if (op_layer < 0) {
        return false;
    }

    // Drive prefetch/release when compute advances to a new layer (ith==0 only).
    if (ith == 0 && op_layer != ctx->cur_compute_layer) {
        const int prev = ctx->cur_compute_layer;
        flex_adapt_after_layer(*ctx, wait_us_total);
        ctx->cur_compute_layer = op_layer;
        const int ahead = flex_effective_ahead(*ctx);
        ctx->stats.effective_ahead = ahead;
        for (int a = 1; a <= ahead; ++a) {
            flex_request_layer(*ctx, (op_layer + a) % ctx->n_layers, true);
        }
        if (prev >= 0) {
            llama_flex_release_layer(*ctx, prev);
        }
    }

    return true;
}
