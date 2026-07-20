#include "llama-window.h"
#include "llama-moe-buffer.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <numeric>
#include <set>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>

enum class llama_window_state {
    cold,
    queued,
    advised,
    resident,
};

enum class llama_window_task_type {
    prefetch,
    reclaim,
    expert_prefetch,
    expert_evict,
};

struct llama_window_range {
    uint8_t * addr = nullptr;
    size_t size = 0;
};

struct llama_window_layer {
    std::vector<llama_window_range> ranges;
    size_t bytes = 0;
    llama_window_state state = llama_window_state::cold;
    bool prefetch_queued = false;
    bool reclaim_queued = false;
};

struct llama_window_task {
    llama_window_task_type type = llama_window_task_type::prefetch;
    int layer = -1;
    int expert_id = -1;       // -1 for layer tasks; expert index for expert_* tasks
    int priority = 0;
    uint64_t graph_seq = 0;
    uint64_t task_token_step = 0; // expert_token_step when this task was enqueued (stale detection)
};

// Per-expert mmap slot for the MoE expert sliding window.
// Each slot aggregates ranges from all expert weight tensors (gate, up, down)
// that correspond to the same (layer, expert_id) pair.
struct llama_window_expert_slot {
    std::vector<llama_window_range> ranges;
    size_t bytes = 0;
    llama_window_state state = llama_window_state::cold;
    uint64_t last_used_step   = 0;  // expert_token_step when last routed to (stale detection)
    uint32_t activation_count = 0;  // decode-only activation count (LRU frequency weighting)
    uint32_t miss_count       = 0;  // CLG misprediction count for this slot
    bool prefetch_queued = false;
    bool evict_queued    = false;
};

// Per-layer data for CLG prediction. Populated at init time from gate_inputs.
// gate_w stores the gate weight matrix in [n_expert × n_embd] FP32 row-major
// format: expert e's embedding vector is at gate_w.data() + e * n_embd.
// This layout makes scores[e] = dot(hs_norm, gate_w + e*n_embd) with contiguous
// memory access for both operands.
struct llama_window_clg_layer {
    std::vector<float> norm_w;     // [n_embd] FP32 — ffn_norm scale weights
    float              norm_eps = 1e-6f;
    std::vector<float> gate_w;     // [n_expert × n_embd] FP32 row-major
    int  n_embd        = 0;
    int  n_expert      = 0;
    int  n_expert_used = 0;        // K (actual top-K from model hparams)
    bool valid         = false;    // true when norm_w and gate_w are populated
};

struct llama_window_context {
    llama_window_params params;
    llama_window_stats stats;
    std::vector<llama_window_layer> layers;
    std::deque<llama_window_task> tasks;
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable cv;
    bool shutdown = false;
    int n_layers = 0;
    int active_layer = -1;
    std::atomic<int> observed_layer {-1};
    bool graph_active = false;
    bool rss_updated = false;
    size_t prefetch_remaining = 0;
    size_t reclaim_remaining = 0;
    size_t indexed_bytes = 0;
    std::chrono::steady_clock::time_point last_layer_time;
    double layer_interval_us = 0.0;
    double prefetch_latency_us = 0.0;
    // MoE expert window state.
    // expert_slots[layer][expert_id] — only populated when params.expert_window is true
    // and the model has expert weight tensors.
    std::vector<std::vector<llama_window_expert_slot>> expert_slots;
    uint64_t expert_token_step  = 0;  // incremented once per graph_begin call (all tokens)
    uint64_t expert_decode_step = 0;  // incremented only for non-prefill tokens
    bool     expert_in_prefill  = false;  // current graph is prefill
    // CLG prediction state
    std::vector<llama_window_clg_layer> clg_layers;    // [n_layers], indexed 0..n_layers-1
    std::vector<uint64_t> clg_predicted_masks;         // [n_layers]: bitmask of experts predicted in current decode token
    // When set, CLG routes predicted experts to the explicit-buffer streamer's
    // async prefetch instead of the mmap window (buffer mode).
    llama_moe_buffer_context * moe_buffer = nullptr;

    ~llama_window_context() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            shutdown = true;
        }
        cv.notify_all();
        for (auto & worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
};

struct llama_window_candidate {
    int layer = -1;
    llama_window_range range;
    bool safe = true;
};

// MADV_POPULATE_READ (Linux 5.14+) may be missing from older <sys/mman.h>.
#if defined(__linux__) && !defined(MADV_POPULATE_READ)
#define MADV_POPULATE_READ 22
#endif

// Force the given range to become resident, on the calling (worker) thread, so
// that the compute thread does not take a synchronous major fault when it later
// touches these pages. MADV_POPULATE_READ faults the whole range in (honouring
// readahead) without raising SIGBUS; on kernels that lack it we fall back to a
// manual one-byte-per-page touch. Returns true if the range was populated.
static bool llama_window_populate(uint8_t * addr, size_t size) {
    if (addr == nullptr || size == 0) {
        return false;
    }
#if defined(__linux__) && defined(MADV_POPULATE_READ)
    if (madvise(addr, size, MADV_POPULATE_READ) == 0) {
        return true;
    }
    // EINVAL/ENOSYS on kernels < 5.14: fall through to the manual touch path.
#endif
#if defined(MADV_WILLNEED)
    madvise(addr, size, MADV_WILLNEED);
#endif
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    volatile uint8_t sink = 0;
    for (size_t off = 0; off < size; off += page) {
        sink ^= addr[off];
    }
    (void) sink;
    return true;
}

static size_t llama_window_current_rss() {
#if defined(__linux__)
    long resident_pages = 0;
    FILE * file = std::fopen("/proc/self/statm", "r");
    if (file != nullptr) {
        long total_pages = 0;
        if (std::fscanf(file, "%ld %ld", &total_pages, &resident_pages) != 2) {
            resident_pages = 0;
        }
        std::fclose(file);
    }
    return resident_pages > 0 ? (size_t) resident_pages * (size_t) sysconf(_SC_PAGESIZE) : 0;
#else
    return 0;
#endif
}

static int llama_window_parse_layer(const char * name) {
    if (name == nullptr || name[0] == '\0') {
        return -1;
    }

    const char * end = name + std::strlen(name);
    const char * p = end;
    while (p > name && p[-1] >= '0' && p[-1] <= '9') {
        --p;
    }
    if (p == end || p == name || p[-1] != '-') {
        return -1;
    }

    char * parsed_end = nullptr;
    const long value = std::strtol(p, &parsed_end, 10);
    if (parsed_end != end || value < 0 || value > std::numeric_limits<int>::max()) {
        return -1;
    }
    return (int) value;
}

static int llama_window_distance_forward(int from, int to, int n_layers) {
    return (to - from + n_layers) % n_layers;
}

static bool llama_window_is_protected(
        int layer,
        int current,
        int n_layers,
        int keep_behind,
        int prefetch_ahead) {
    if (layer < 0 || current < 0 || n_layers <= 0) {
        return true;
    }

    for (int delta = -keep_behind; delta <= prefetch_ahead; ++delta) {
        int protected_layer = (current + delta) % n_layers;
        if (protected_layer < 0) {
            protected_layer += n_layers;
        }
        if (layer == protected_layer) {
            return true;
        }
    }
    return false;
}

static void llama_window_insert_task(
        llama_window_context & ctx,
        llama_window_task task) {
    auto it = std::find_if(ctx.tasks.begin(), ctx.tasks.end(),
            [&task](const llama_window_task & queued) {
                return task.priority < queued.priority;
            });
    ctx.tasks.insert(it, task);
    ctx.cv.notify_one();
}

static void llama_window_request_prefetch_locked(
        llama_window_context & ctx,
        int layer,
        int priority) {
    if (ctx.prefetch_remaining == 0) {
        return;
    }
    auto & entry = ctx.layers[layer];
    if (entry.ranges.empty() || entry.prefetch_queued ||
            entry.state == llama_window_state::resident ||
            entry.state == llama_window_state::advised) {
        return;
    }

    if (entry.bytes > ctx.prefetch_remaining &&
            ctx.prefetch_remaining != ctx.params.prefetch_budget) {
        return;
    }
    ctx.prefetch_remaining =
            entry.bytes >= ctx.prefetch_remaining ? 0 : ctx.prefetch_remaining - entry.bytes;
    entry.prefetch_queued = true;
    entry.state = llama_window_state::queued;
    ctx.stats.prefetch_tasks++;
    llama_window_insert_task(ctx, {
            llama_window_task_type::prefetch,
            layer,
            /*expert_id=*/ -1,
            priority,
            ctx.stats.graph_seq,
    });
}

static void llama_window_request_reclaim_locked(
        llama_window_context & ctx,
        int layer,
        int priority) {
    if (ctx.reclaim_remaining == 0) {
        return;
    }
    auto & entry = ctx.layers[layer];
    if (entry.ranges.empty() || entry.reclaim_queued ||
            entry.state == llama_window_state::cold ||
            llama_window_is_protected(
                layer,
                ctx.active_layer,
                ctx.n_layers,
                ctx.params.keep_behind,
                ctx.params.prefetch_ahead)) {
        return;
    }

    if (entry.bytes > ctx.reclaim_remaining &&
            ctx.reclaim_remaining != ctx.params.reclaim_budget) {
        return;
    }
    ctx.reclaim_remaining =
            entry.bytes >= ctx.reclaim_remaining ? 0 : ctx.reclaim_remaining - entry.bytes;
    entry.reclaim_queued = true;
    ctx.stats.reclaim_tasks++;
    llama_window_insert_task(ctx, {
            llama_window_task_type::reclaim,
            layer,
            /*expert_id=*/ -1,
            priority,
            ctx.stats.graph_seq,
    });
}

// Apply madvise(MADV_DONTNEED) to the given range (returns true on success).
static bool llama_window_dontneed(uint8_t * addr, size_t size) {
    if (addr == nullptr || size == 0) {
        return false;
    }
    int advice = MADV_NORMAL;
#if defined(MADV_DONTNEED)
    advice = MADV_DONTNEED;
#endif
    return madvise(addr, size, advice) == 0;
}

static void llama_window_worker(llama_window_context * ctx) {
    for (;;) {
        llama_window_task task;
        {
            std::unique_lock<std::mutex> lock(ctx->mutex);
            ctx->cv.wait(lock, [&] {
                return ctx->shutdown || !ctx->tasks.empty();
            });
            if (ctx->shutdown && ctx->tasks.empty()) {
                return;
            }
            task = ctx->tasks.front();
            ctx->tasks.pop_front();

            // --- Layer task: check stale reclaim ---
            if (task.expert_id < 0) {
                auto & entry = ctx->layers[task.layer];
                if (task.type == llama_window_task_type::reclaim &&
                        llama_window_is_protected(
                            task.layer,
                            ctx->active_layer,
                            ctx->n_layers,
                            ctx->params.keep_behind,
                            ctx->params.prefetch_ahead)) {
                    entry.reclaim_queued = false;
                    ctx->stats.stale_tasks++;
                    continue;
                }
            }
        }

        // --- Expert task ---
        if (task.expert_id >= 0) {
            const int l = task.layer;
            const int e = task.expert_id;
            if (l < 0 || l >= (int) ctx->expert_slots.size() ||
                    e < 0 || e >= (int) ctx->expert_slots[l].size()) {
                continue;
            }

            // Stale-evict protection: if a newer token has already re-activated
            // this expert, the evict is stale and must be dropped to avoid a
            // DONTNEED that would immediately undo a just-completed prefetch.
            if (task.type == llama_window_task_type::expert_evict) {
                std::lock_guard<std::mutex> lock(ctx->mutex);
                auto & s = ctx->expert_slots[l][e];
                s.evict_queued = false;
                // Re-check: if expert was used after this evict was enqueued, skip.
                if (s.last_used_step >= task.task_token_step) {
                    ctx->stats.stale_tasks++;
                    continue;
                }
                // Still cold-eligible; fall through to perform DONTNEED below.
                // We keep evict_queued = false so the outer lock path sees it.
            }

            const auto & slot = ctx->expert_slots[l][e];
            size_t bytes = 0;
            bool success = true;
            if (task.type == llama_window_task_type::expert_prefetch) {
                for (const auto & range : slot.ranges) {
                    if (llama_window_populate(range.addr, range.size)) {
                        bytes += range.size;
                    } else {
                        success = false;
                    }
                }
            } else {
                // expert_evict: already validated above (stale check)
                for (const auto & range : slot.ranges) {
                    if (llama_window_dontneed(range.addr, range.size)) {
                        bytes += range.size;
                    } else {
                        success = false;
                    }
                }
            }
            {
                std::lock_guard<std::mutex> lock(ctx->mutex);
                auto & s = ctx->expert_slots[l][e];
                if (task.type == llama_window_task_type::expert_prefetch) {
                    s.prefetch_queued = false;
                    if (success) {
                        s.state = llama_window_state::advised;
                        ctx->stats.expert_prefetch_calls++;
                        ctx->stats.expert_bytes_prefetched += bytes;
                    } else {
                        s.state = llama_window_state::cold;
                    }
                } else {
                    // evict_queued was already cleared in the stale-check block above.
                    // Double-check: between our stale check and now, the main thread
                    // may have re-activated this expert (set last_used_step >= our step).
                    // The DONTNEED is already done (pages dropped), which is safe for
                    // read-only mmap (re-faulted on next access). But don't mark cold
                    // if the expert is live again — it will be prefetched immediately.
                    if (success && s.last_used_step < task.task_token_step) {
                        s.state = llama_window_state::cold;
                        ctx->stats.expert_evict_calls++;
                        ctx->stats.expert_bytes_evicted += bytes;
                    }
                }
            }
            continue;
        }

        // --- Layer task ---
        const auto begin = std::chrono::steady_clock::now();
        bool success = true;
        size_t bytes = 0;
        for (const auto & range : ctx->layers[task.layer].ranges) {
            if (task.type == llama_window_task_type::prefetch) {
                // Actually fault the pages in here, on the worker thread, instead
                // of only hinting with MADV_WILLNEED. This converts the major
                // fault that the compute thread would otherwise take into a
                // background populate that overlaps computation.
                if (llama_window_populate(range.addr, range.size)) {
                    bytes += range.size;
                } else {
                    success = false;
                }
            } else {
                if (llama_window_dontneed(range.addr, range.size)) {
                    bytes += range.size;
                } else {
                    success = false;
                }
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_us =
                std::chrono::duration<double, std::micro>(end - begin).count();

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            auto & entry = ctx->layers[task.layer];
            if (task.type == llama_window_task_type::prefetch) {
                entry.prefetch_queued = false;
                if (success) {
                    entry.state = llama_window_state::advised;
                    ctx->stats.prefetch_calls++;
                    ctx->stats.bytes_prefetched += bytes;
                    ctx->prefetch_latency_us = ctx->prefetch_latency_us == 0.0
                            ? elapsed_us
                            : ctx->prefetch_latency_us * 0.8 + elapsed_us * 0.2;
                } else {
                    entry.state = llama_window_state::cold;
                    ctx->stats.prefetch_failures++;
                }
            } else {
                entry.reclaim_queued = false;
                if (success) {
                    entry.state = llama_window_state::cold;
                    ctx->stats.reclaim_calls++;
                    ctx->stats.bytes_reclaimed += bytes;
                } else {
                    ctx->stats.reclaim_failures++;
                }
            }
        }
    }
}

static void llama_window_auto_tune_locked(llama_window_context & ctx) {
    if (!ctx.params.auto_tune) {
        return;
    }

    const int max_ahead = std::max(1, std::min(8, ctx.params.window_size - 1));
    const bool memory_pressure =
            ctx.params.memory_limit > 0 &&
            ctx.stats.current_rss > ctx.params.memory_limit;
    if (memory_pressure && ctx.params.prefetch_ahead > 1) {
        ctx.params.prefetch_ahead--;
    } else if (ctx.layer_interval_us > 0.0 &&
            ctx.prefetch_latency_us > ctx.layer_interval_us * ctx.params.prefetch_ahead &&
            ctx.params.prefetch_ahead < max_ahead) {
        ctx.params.prefetch_ahead++;
    }
    ctx.params.keep_behind = std::max(
            2,
            ctx.params.window_size - ctx.params.prefetch_ahead - 1);
}

static void llama_window_advance_locked(
        llama_window_context & ctx,
        int layer,
        bool accessed) {
    if (layer < 0 || layer >= ctx.n_layers) {
        return;
    }
    if (layer == ctx.active_layer) {
        if (accessed) {
            ctx.layers[layer].state = llama_window_state::resident;
        }
        return;
    }

    const auto now = std::chrono::steady_clock::now();
    if (ctx.active_layer >= 0) {
        const double interval_us =
                std::chrono::duration<double, std::micro>(now - ctx.last_layer_time).count();
        ctx.layer_interval_us = ctx.layer_interval_us == 0.0
                ? interval_us
                : ctx.layer_interval_us * 0.9 + interval_us * 0.1;
    }
    ctx.last_layer_time = now;
    ctx.active_layer = layer;
    ctx.stats.layer_events++;
    if (accessed) {
        ctx.layers[layer].state = llama_window_state::resident;
    }

    if (layer == 0 && ctx.rss_updated) {
        llama_window_auto_tune_locked(ctx);
        ctx.rss_updated = false;
    }

    for (int delta = 1;
            delta <= ctx.params.prefetch_ahead && ctx.prefetch_remaining > 0;
            ++delta) {
        const int future = (layer + delta) % ctx.n_layers;
        llama_window_request_prefetch_locked(ctx, future, delta);
    }

    const bool memory_pressure =
            ctx.params.memory_limit > 0 &&
            ctx.stats.current_rss > ctx.params.memory_limit;
    if (ctx.params.use_dontneed && (ctx.params.memory_limit == 0 || memory_pressure)) {
        std::vector<int> candidates;
        candidates.reserve(ctx.n_layers);
        for (int candidate = 0; candidate < ctx.n_layers; ++candidate) {
            if (!llama_window_is_protected(
                    candidate,
                    layer,
                    ctx.n_layers,
                    ctx.params.keep_behind,
                    ctx.params.prefetch_ahead)) {
                candidates.push_back(candidate);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
            const size_t score_a =
                    (size_t) llama_window_distance_forward(layer, a, ctx.n_layers) * 1024ull * 1024ull +
                    ctx.layers[a].bytes / 2;
            const size_t score_b =
                    (size_t) llama_window_distance_forward(layer, b, ctx.n_layers) * 1024ull * 1024ull +
                    ctx.layers[b].bytes / 2;
            return score_a > score_b;
        });
        for (int candidate : candidates) {
            if (ctx.reclaim_remaining == 0) {
                break;
            }
            llama_window_request_reclaim_locked(ctx, candidate, 100);
        }
    }

    if (ctx.params.debug_log &&
            (ctx.stats.layer_events <= 8 ||
             (layer == 0 && ctx.stats.graph_seq % 32 == 0))) {
        std::fprintf(
                stderr,
                "llama_window_v2: graph=%llu layer=%d ahead=%d behind=%d "
                "queued=%zu prefetch=%llu reclaim=%llu rss=%.2f MiB failures=%llu/%llu\n",
                (unsigned long long) ctx.stats.graph_seq,
                layer,
                ctx.params.prefetch_ahead,
                ctx.params.keep_behind,
                ctx.tasks.size(),
                (unsigned long long) ctx.stats.prefetch_calls,
                (unsigned long long) ctx.stats.reclaim_calls,
                ctx.stats.current_rss / 1024.0 / 1024.0,
                (unsigned long long) ctx.stats.prefetch_failures,
                (unsigned long long) ctx.stats.reclaim_failures);
    }
}

std::shared_ptr<llama_window_context> llama_window_create(
        const std::vector<llama_window_region_input> & inputs,
        int n_layers,
        const llama_window_params & params,
        const std::vector<llama_window_gate_input> & gate_inputs) {
    auto ctx = std::make_shared<llama_window_context>();
    ctx->params = params;
    ctx->n_layers = n_layers;
    ctx->layers.resize(std::max(0, n_layers));

    if (!params.enabled || n_layers <= 0) {
        return ctx;
    }

    const size_t page_size = (size_t) sysconf(_SC_PAGESIZE);
    std::vector<llama_window_candidate> candidates;
    candidates.reserve(inputs.size());

    // ---- Expert tensor indexing (MoE sliding window) ----
    // Expert weight tensors (e.g. ffn_gate_exps.weight, shape {n_ff, n_embd, n_expert})
    // are sliced per-expert and tracked separately from the per-layer layer window.
    // They are excluded from the layer-level candidates below.
    std::set<const void *> expert_tensor_addrs; // used to skip them in layer phase

    if (params.expert_window) {
        // Collect all expert tensors and determine the maximum expert count per layer.
        std::vector<int> expert_count_per_layer(n_layers, 0);
        for (const auto & input : inputs) {
            if (input.n_expert <= 0 || input.addr == nullptr || input.expert_stride == 0) {
                continue;
            }
            int layer = -1;
            if (std::sscanf(input.name.c_str(), "blk.%d.", &layer) != 1 ||
                    layer < 0 || layer >= n_layers) {
                continue;
            }
            expert_count_per_layer[layer] = std::max(
                    expert_count_per_layer[layer], input.n_expert);
            expert_tensor_addrs.insert(input.addr);
        }

        // Allocate expert_slots[layer][expert_id]
        ctx->expert_slots.resize(n_layers);
        for (int l = 0; l < n_layers; ++l) {
            if (expert_count_per_layer[l] > 0) {
                ctx->expert_slots[l].resize(expert_count_per_layer[l]);
            }
        }

        // Populate expert_slots with page-aligned ranges from each expert tensor.
        for (const auto & input : inputs) {
            if (input.n_expert <= 0 || input.addr == nullptr || input.expert_stride == 0) {
                continue;
            }
            int layer = -1;
            if (std::sscanf(input.name.c_str(), "blk.%d.", &layer) != 1 ||
                    layer < 0 || layer >= n_layers) {
                continue;
            }
            if (layer >= (int) ctx->expert_slots.size()) {
                continue;
            }
            for (int e = 0; e < input.n_expert && e < (int) ctx->expert_slots[layer].size(); ++e) {
                const uintptr_t begin = (uintptr_t) input.addr + (uintptr_t) e * input.expert_stride;
                const uintptr_t end   = begin + input.expert_stride;
                const uintptr_t page_begin = (begin + page_size - 1) & ~(uintptr_t) (page_size - 1);
                const uintptr_t page_end   = end & ~(uintptr_t) (page_size - 1);
                if (page_begin >= page_end) {
                    continue;
                }
                ctx->expert_slots[layer][e].ranges.push_back(
                        {(uint8_t *) page_begin, page_end - page_begin});
            }
        }

        // Merge overlapping ranges and compute byte sizes per slot.
        for (auto & layer_slots : ctx->expert_slots) {
            for (auto & slot : layer_slots) {
                std::sort(slot.ranges.begin(), slot.ranges.end(),
                        [](const llama_window_range & a, const llama_window_range & b) {
                            return a.addr < b.addr;
                        });
                std::vector<llama_window_range> merged;
                for (const auto & r : slot.ranges) {
                    if (merged.empty() || merged.back().addr + merged.back().size < r.addr) {
                        merged.push_back(r);
                    } else {
                        uint8_t * rend = std::max(
                                merged.back().addr + merged.back().size, r.addr + r.size);
                        merged.back().size = rend - merged.back().addr;
                    }
                }
                slot.ranges = std::move(merged);
                for (const auto & r : slot.ranges) {
                    slot.bytes += r.size;
                }
            }
        }

        if (params.debug_log) {
            int total_experts = 0;
            size_t total_expert_bytes = 0;
            for (const auto & layer_slots : ctx->expert_slots) {
                total_experts += (int) layer_slots.size();
                for (const auto & s : layer_slots) {
                    total_expert_bytes += s.bytes;
                }
            }
            std::fprintf(stderr,
                    "llama_window_moe: indexed %d expert slots, %.2f MiB, "
                    "window_tokens=%d dontneed=%d clg=%d\n",
                    total_experts,
                    total_expert_bytes / 1024.0 / 1024.0,
                    params.expert_window_tokens,
                    params.expert_dontneed ? 1 : 0,
                    params.clg_predict ? 1 : 0);
        }
    }
    // ---- End expert indexing ----

    // ---- CLG (Cross-Layer Gate) predictor initialisation ----
    // For each layer supplied in gate_inputs, dequantize the ffn_norm and
    // ffn_gate_inp weight tensors to FP32 and store them in ctx->clg_layers.
    // The gate matrix is transposed from GGML's [n_embd, n_expert] layout to
    // [n_expert × n_embd] row-major so that each expert's embedding vector is
    // a contiguous row, enabling efficient dot-product scoring.
    if (params.clg_predict && !gate_inputs.empty()) {
        ctx->clg_layers.resize(n_layers);
        ctx->clg_predicted_masks.assign(n_layers, 0ull);

        for (const auto & gi : gate_inputs) {
            if (gi.layer < 0 || gi.layer >= n_layers) {
                continue;
            }
            if (gi.norm_tensor == nullptr || gi.gate_tensor == nullptr) {
                continue;
            }

            auto & clg = ctx->clg_layers[gi.layer];
            const int n_embd   = (int) gi.gate_tensor->ne[0];
            const int n_expert = (int) gi.gate_tensor->ne[1];
            if (n_embd <= 0 || n_expert <= 0) {
                continue;
            }
            clg.n_embd        = n_embd;
            clg.n_expert      = n_expert;
            clg.n_expert_used = gi.n_expert_used;
            clg.norm_eps      = gi.norm_eps;

            // Helper: dequantize any GGML tensor to FP32.
            // For F32 tensors, to_float is NULL (no conversion needed) —
            // handle via memcpy instead of the generic fallback.
            auto dequant_to_fp32 = [](const ggml_tensor * t,
                                      float * dst, int64_t n_elems) -> bool {
                if (t->type == GGML_TYPE_F32) {
                    std::memcpy(dst, t->data, n_elems * sizeof(float));
                    return true;
                }
                if (t->type == GGML_TYPE_F16) {
                    const ggml_fp16_t * src = static_cast<const ggml_fp16_t *>(t->data);
                    for (int64_t i = 0; i < n_elems; ++i) {
                        dst[i] = ggml_fp16_to_fp32(src[i]);
                    }
                    return true;
                }
                const ggml_type_traits * traits = ggml_get_type_traits(t->type);
                if (traits && traits->to_float) {
                    traits->to_float(t->data, dst, n_elems);
                    return true;
                }
                return false;
            };

            // Dequantize ffn_norm weights → FP32 [n_embd]
            clg.norm_w.resize(n_embd);
            if (!dequant_to_fp32(gi.norm_tensor, clg.norm_w.data(), n_embd)) {
                std::fill(clg.norm_w.begin(), clg.norm_w.end(), 1.0f);
            }

            // Dequantize ffn_gate_inp weights → FP32.
            // GGML layout: element (d, e) at linear index d + e*n_embd.
            // After dequantization: gate_w[e*n_embd + d] = gate_weight(d,e).
            // (Identical to [d + e*n_embd] — GGML's fastest-dim is d=ne[0]=n_embd,
            //  so column e is contiguous at offset e*n_embd. This matches the
            //  dot-product loop: scores[e] = dot(hs_norm, gate_w + e*n_embd).)
            const int64_t n_total = (int64_t) n_embd * n_expert;
            clg.gate_w.resize(n_total);
            if (!dequant_to_fp32(gi.gate_tensor, clg.gate_w.data(), n_total)) {
                std::fill(clg.gate_w.begin(), clg.gate_w.end(), 0.0f);
            }

            clg.valid = true;
        }

        if (params.debug_log) {
            int n_valid = 0;
            size_t gate_bytes = 0;
            for (const auto & clg : ctx->clg_layers) {
                if (clg.valid) {
                    ++n_valid;
                    gate_bytes += clg.gate_w.size() * sizeof(float)
                                + clg.norm_w.size() * sizeof(float);
                }
            }
            std::fprintf(stderr,
                    "llama_window_clg: loaded %d/%d gate layers, "
                    "%.2f MiB FP32, delta=%d prefill_thr=%d\n",
                    n_valid, n_layers,
                    gate_bytes / 1024.0 / 1024.0,
                    params.clg_delta,
                    params.clg_prefill_threshold);
        }
    }
    // ---- End CLG initialisation ----

    for (const auto & input : inputs) {
        int layer = -1;
        if (std::sscanf(input.name.c_str(), "blk.%d.", &layer) != 1 ||
                layer < 0 || layer >= n_layers ||
                input.addr == nullptr || input.size == 0) {
            continue;
        }
        // Expert tensors are managed at expert granularity; skip them here.
        if (params.expert_window && expert_tensor_addrs.count(input.addr)) {
            continue;
        }

        const uintptr_t begin = (uintptr_t) input.addr;
        const uintptr_t end = begin + input.size;
        const uintptr_t page_begin = (begin + page_size - 1) & ~(uintptr_t) (page_size - 1);
        const uintptr_t page_end = end & ~(uintptr_t) (page_size - 1);
        if (page_begin >= page_end) {
            continue;
        }
        candidates.push_back({
                layer,
                {(uint8_t *) page_begin, page_end - page_begin},
                true,
        });
    }

    std::sort(candidates.begin(), candidates.end(),
            [](const llama_window_candidate & a, const llama_window_candidate & b) {
                return a.range.addr < b.range.addr;
            });
    for (size_t i = 0; i < candidates.size(); ++i) {
        const uintptr_t end_i =
                (uintptr_t) candidates[i].range.addr + candidates[i].range.size;
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if ((uintptr_t) candidates[j].range.addr >= end_i) {
                break;
            }
            if (candidates[i].layer != candidates[j].layer) {
                candidates[i].safe = false;
                candidates[j].safe = false;
            }
        }
    }

    size_t indexed = 0;
    for (const auto & candidate : candidates) {
        if (candidate.safe) {
            ctx->layers[candidate.layer].ranges.push_back(candidate.range);
        }
    }
    for (auto & layer : ctx->layers) {
        std::sort(layer.ranges.begin(), layer.ranges.end(),
                [](const llama_window_range & a, const llama_window_range & b) {
                    return a.addr < b.addr;
                });
        std::vector<llama_window_range> merged;
        for (const auto & range : layer.ranges) {
            if (merged.empty() || merged.back().addr + merged.back().size < range.addr) {
                merged.push_back(range);
            } else {
                uint8_t * end = std::max(
                        merged.back().addr + merged.back().size,
                        range.addr + range.size);
                merged.back().size = end - merged.back().addr;
            }
        }
        layer.ranges = std::move(merged);
        for (const auto & range : layer.ranges) {
            layer.bytes += range.size;
            indexed += range.size;
        }
    }
    ctx->indexed_bytes = indexed;

    ctx->params.window_size = std::max(4, std::min(ctx->params.window_size, n_layers));
    ctx->params.prefetch_ahead = std::max(
            1,
            std::min(ctx->params.prefetch_ahead, ctx->params.window_size - 1));
    ctx->params.keep_behind = std::max(
            2,
            ctx->params.window_size - ctx->params.prefetch_ahead - 1);

    const int n_workers = std::max(1, std::min(ctx->params.worker_threads, 4));
    ctx->workers.reserve(n_workers);
    for (int i = 0; i < n_workers; ++i) {
        ctx->workers.emplace_back(llama_window_worker, ctx.get());
    }

    if (ctx->params.debug_log) {
        std::fprintf(
                stderr,
                "llama_window_v2: indexed=%0.2f MiB layers=%d window=%d ahead=%d "
                "behind=%d workers=%d memory_limit=%0.2f MiB dontneed=%d auto_tune=%d\n",
                indexed / 1024.0 / 1024.0,
                n_layers,
                ctx->params.window_size,
                ctx->params.prefetch_ahead,
                ctx->params.keep_behind,
                n_workers,
                ctx->params.memory_limit / 1024.0 / 1024.0,
                ctx->params.use_dontneed ? 1 : 0,
                ctx->params.auto_tune ? 1 : 0);
    }

    return ctx;
}

bool llama_window_enabled(const llama_window_context * ctx) {
    return ctx != nullptr && ctx->params.enabled && ctx->n_layers > 0;
}

void llama_window_set_moe_buffer(llama_window_context & ctx, llama_moe_buffer_context * moe) {
    ctx.moe_buffer = moe;
}

// Schedule an expert prefetch task (must be called with ctx.mutex held).
static void llama_window_request_expert_prefetch_locked(
        llama_window_context & ctx, int layer, int expert_id) {
    if (layer < 0 || layer >= (int) ctx.expert_slots.size()) {
        return;
    }
    if (expert_id < 0 || expert_id >= (int) ctx.expert_slots[layer].size()) {
        return;
    }
    auto & slot = ctx.expert_slots[layer][expert_id];
    if (slot.ranges.empty() || slot.prefetch_queued ||
            slot.state == llama_window_state::resident ||
            slot.state == llama_window_state::advised) {
        return;
    }
    slot.prefetch_queued = true;
    slot.state = llama_window_state::queued;
    llama_window_insert_task(ctx, {
            llama_window_task_type::expert_prefetch,
            layer,
            expert_id,
            0,    // priority 0 = high (same as immediate layer prefetch)
            ctx.stats.graph_seq,
            ctx.expert_token_step,
    });
}

// Schedule an expert eviction task (must be called with ctx.mutex held).
static void llama_window_request_expert_evict_locked(
        llama_window_context & ctx, int layer, int expert_id) {
    if (layer < 0 || layer >= (int) ctx.expert_slots.size()) {
        return;
    }
    if (expert_id < 0 || expert_id >= (int) ctx.expert_slots[layer].size()) {
        return;
    }
    auto & slot = ctx.expert_slots[layer][expert_id];
    if (slot.ranges.empty() || slot.evict_queued || slot.prefetch_queued ||
            slot.state == llama_window_state::cold) {
        return;
    }
    slot.evict_queued = true;
    llama_window_insert_task(ctx, {
            llama_window_task_type::expert_evict,
            layer,
            expert_id,
            200,  // low priority — run after any pending prefetches
            ctx.stats.graph_seq,
            ctx.expert_token_step,
    });
}

// Thread-local scratch buffers used by the CLG prediction function.
// Avoids heap allocation in the decode hot path (called every token per layer).
static thread_local std::vector<float> tl_hs_norm;
static thread_local std::vector<float> tl_scores;
static thread_local std::vector<float> tl_best_scores;

// Run the CLG predictor for expert_slots[next_layer], using the hidden state
// carried by l_out-{next_layer-1}.  Schedules prefetch for the top-(K+delta)
// predicted experts and evict for the remainder.  Must be called WITHOUT the
// ctx.mutex held (computation happens before the lock is taken).
static void llama_window_clg_predict_and_schedule(
        llama_window_context & ctx,
        int                    next_layer,
        const ggml_tensor    * node) {
    if (next_layer < 0 || next_layer >= (int) ctx.clg_layers.size()) {
        return;
    }
    const auto & clg = ctx.clg_layers[next_layer];
    if (!clg.valid) {
        return;
    }
    // In buffer mode the exps tensors are owned by the moe-buffer (not indexed as
    // expert_slots), so the slot requirement is relaxed: prediction still runs and
    // its result is routed to the async prefetch below.
    const bool has_buffer = ctx.moe_buffer != nullptr;
    if (!has_buffer && (next_layer >= (int) ctx.expert_slots.size() ||
            ctx.expert_slots[next_layer].empty())) {
        return;
    }
    if (node->data == nullptr || node->type != GGML_TYPE_F32) {
        return; // only handle F32 activations (CPU default)
    }

    const int n_embd   = clg.n_embd;
    const int n_expert = clg.n_expert;
    const int n_select = std::min(clg.n_expert_used + ctx.params.clg_delta, n_expert);
    const int n_tokens = (int) (node->ne[1] > 0 ? node->ne[1] : 1);

    // Prefill guard: when many tokens are batched, the union of predicted experts
    // covers almost the full set → skip eviction to avoid thrashing.
    const bool skip_evict = (n_tokens > ctx.params.clg_prefill_threshold);

    tl_hs_norm.resize(n_embd);
    tl_scores.resize(n_expert);
    tl_best_scores.assign(n_expert, -1e38f);

    // predicted_mask: bit e set means expert e is in the predicted set.
    // Use uint64_t; supports up to 64 experts (covers all known MoE models).
    uint64_t predicted_mask = 0;

    const float * raw = static_cast<const float *>(node->data);

    for (int t = 0; t < n_tokens; ++t) {
        const float * hs = raw + (int64_t) t * n_embd;

        // 1. RMS norm: hs_norm[d] = hs[d] / rms * norm_w[d]
        float sum_sq = 0.0f;
        for (int d = 0; d < n_embd; ++d) {
            sum_sq += hs[d] * hs[d];
        }
        const float inv_rms = 1.0f / std::sqrt(sum_sq / (float) n_embd + clg.norm_eps);
        for (int d = 0; d < n_embd; ++d) {
            tl_hs_norm[d] = hs[d] * inv_rms * clg.norm_w[d];
        }

        // 2. Gate scores: scores[e] = dot(hs_norm, gate_w[e])
        //    gate_w[e] is contiguous at gate_w.data() + e * n_embd.
        for (int e = 0; e < n_expert; ++e) {
            const float * gw = clg.gate_w.data() + (int64_t) e * n_embd;
            float score = 0.0f;
            for (int d = 0; d < n_embd; ++d) {
                score += tl_hs_norm[d] * gw[d];
            }
            tl_scores[e] = score;
        }

        // 3. Top-(K+delta) selection via linear scan (O(n_expert * n_select),
        //    negligible for n_expert <= 64).
        uint64_t local_mask = 0;
        for (int k = 0; k < n_select; ++k) {
            int   best_e     = -1;
            float best_score = -1e38f;
            for (int e = 0; e < n_expert; ++e) {
                if (!(local_mask & (1ull << e)) && tl_scores[e] > best_score) {
                    best_score = tl_scores[e];
                    best_e     = e;
                }
            }
            if (best_e >= 0) {
                local_mask |= (1ull << best_e);
                tl_best_scores[best_e] = std::max(tl_best_scores[best_e], best_score);
            }
        }
        predicted_mask |= local_mask;

        // If all experts predicted and eviction skipped anyway, no need to
        // process remaining tokens.
        if (skip_evict && predicted_mask == ((1ull << n_expert) - 1ull)) {
            break;
        }
    }

    // 4a. Buffer mode: route the predicted experts to the moe-buffer's async
    // prefetch worker so layer next_layer's slices stream in while the current
    // layer computes. The weight-stream callback guarantees correctness; this
    // only hides the read latency. No expert_slots / mmap madvise involved.
    if (has_buffer) {
        std::vector<std::pair<float, int>> ranked;
        ranked.reserve(n_expert);
        for (int e = 0; e < n_expert && e < 64; ++e) {
            if (predicted_mask & (1ull << e)) {
                ranked.push_back({tl_best_scores[e], e});
            }
        }
        std::sort(ranked.begin(), ranked.end(),
                [](const auto & a, const auto & b) {
                    if (a.first != b.first) {
                        return a.first > b.first;
                    }
                    return a.second < b.second;
                });
        int ids[64];
        float scores[64];
        int n = 0;
        for (const auto & p : ranked) {
            ids[n] = p.second;
            scores[n] = p.first;
            ++n;
        }
        llama_moe_buffer_prefetch_ranked(ctx.moe_buffer, next_layer, ids, scores, n);
        ctx.stats.clg_predict_calls++;
        return;
    }

    // 4b. mmap window mode: schedule prefetch / evict tasks and record prediction
    // under the mutex.
    {
        std::lock_guard<std::mutex> lock(ctx.mutex);
        if (!ctx.graph_active) {
            return;
        }

        // Build hot-expert mask: experts selected more frequently than
        // clg_hot_thr_pct% of decode tokens are never evicted, regardless of
        // CLG prediction.  This handles "always-on" experts that appear in nearly
        // every token (their repeated eviction + re-prefetch would just thrash).
        // Requires clg_hot_warmup decode steps of data before activating.
        uint64_t hot_mask = 0;
        const int hot_thr = ctx.params.clg_hot_thr_pct;
        if (hot_thr > 0 &&
                ctx.expert_decode_step >= (uint64_t) ctx.params.clg_hot_warmup) {
            const auto & lslots = ctx.expert_slots[next_layer];
            for (int e = 0; e < n_expert && e < (int) lslots.size(); ++e) {
                // activation_count incremented at ffn_moe_topk in CLG mode.
                if (lslots[e].activation_count * 100 >
                        hot_thr * ctx.expert_decode_step) {
                    hot_mask |= (1ull << e);
                }
            }
        }

        // The effective "keep" set is CLG prediction UNION hot experts.
        const uint64_t keep_mask = predicted_mask | hot_mask;

        // Store combined mask for accuracy tracking at ffn_moe_topk.
        if (next_layer < (int) ctx.clg_predicted_masks.size()) {
            ctx.clg_predicted_masks[next_layer] = keep_mask;
        }

        const auto & layer_slots = ctx.expert_slots[next_layer];
        for (int e = 0; e < (int) layer_slots.size() && e < n_expert; ++e) {
            if (keep_mask & (1ull << e)) {
                llama_window_request_expert_prefetch_locked(ctx, next_layer, e);
            } else if (!skip_evict && ctx.params.expert_dontneed) {
                // Only evict if LLAMA_LAZY_MOE_DONTNEED=1 is explicitly set.
                const auto & slot = ctx.expert_slots[next_layer][e];
                if (!slot.evict_queued && !slot.prefetch_queued &&
                        slot.state != llama_window_state::cold) {
                    llama_window_request_expert_evict_locked(ctx, next_layer, e);
                    ctx.stats.clg_evict_calls++;
                }
            }
        }

        ctx.stats.clg_predict_calls++;
        // Hot stats: count the number of hot-protected eviction skips (for logging).
        if (ctx.params.expert_dontneed && !skip_evict) {
            ctx.stats.clg_hot_protected +=
                    (uint64_t) __builtin_popcountll(hot_mask);
        }
    }
}

void llama_window_graph_begin(llama_window_context & ctx, bool prefill) {
    if (!llama_window_enabled(&ctx)) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.graph_active = true;
    ctx.active_layer = -1;
    ctx.observed_layer.store(-1, std::memory_order_relaxed);
    const size_t average_layer_bytes =
            ctx.n_layers > 0 ? ctx.indexed_bytes / (size_t) ctx.n_layers : 0;
    const bool aggressive_prefill = prefill && ctx.params.prefill_aggressive;
    ctx.prefetch_remaining = aggressive_prefill
            ? std::max(
                ctx.params.prefetch_budget,
                average_layer_bytes * (size_t) ctx.params.window_size)
            : ctx.params.prefetch_budget;
    ctx.reclaim_remaining = ctx.params.reclaim_budget;
    ctx.stats.graph_seq++;
    if (ctx.stats.graph_seq == 1 || ctx.stats.graph_seq % 8 == 0) {
        ctx.stats.current_rss = llama_window_current_rss();
        ctx.stats.peak_rss = std::max(ctx.stats.peak_rss, ctx.stats.current_rss);
        ctx.rss_updated = true;
    }

    llama_window_request_prefetch_locked(ctx, 0, 0);
    const int initial_ahead = aggressive_prefill
            ? std::min(ctx.params.window_size - 1, ctx.n_layers - 1)
            : ctx.params.prefetch_ahead;
    for (int delta = 1;
            delta <= initial_ahead && ctx.prefetch_remaining > 0;
            ++delta) {
        llama_window_request_prefetch_locked(
                ctx,
                delta % ctx.n_layers,
                delta);
    }

    // ---- MoE expert window: step counter + optional LRU pre-warm/evict ----
    if (ctx.params.expert_window && !ctx.expert_slots.empty()) {
        ctx.expert_token_step++;
        ctx.expert_in_prefill = prefill;
        if (!prefill) {
            ctx.expert_decode_step++;
        }

        if (!ctx.params.clg_predict) {
            // LRU mode: pre-warm experts used in the previous token, and evict
            // experts that have been cold for longer than expert_window_tokens.
            const uint64_t step = ctx.expert_token_step;
            const uint64_t w    = (uint64_t) std::max(1, ctx.params.expert_window_tokens);
            const uint64_t prev = step - 1;

            for (int l = 0; l < (int) ctx.expert_slots.size(); ++l) {
                for (int e = 0; e < (int) ctx.expert_slots[l].size(); ++e) {
                    auto & slot = ctx.expert_slots[l][e];

                    if (step > 1 && slot.last_used_step == prev) {
                        llama_window_request_expert_prefetch_locked(ctx, l, e);
                    }

                    if (!ctx.params.expert_dontneed) continue;
                    if (slot.evict_queued || slot.prefetch_queued)  continue;

                    const uint64_t d_steps = ctx.expert_decode_step > 0
                            ? ctx.expert_decode_step : 1;
                    const bool high_freq = d_steps > 8 &&
                            slot.activation_count > (uint32_t)(d_steps / 3);
                    const uint64_t eff_w = high_freq ? w * 2 : w;
                    if (step > eff_w && slot.last_used_step + eff_w < step) {
                        llama_window_request_expert_evict_locked(ctx, l, e);
                    }
                }
            }

            if (ctx.params.debug_log && !prefill && ctx.expert_decode_step % 32 == 0) {
                std::fprintf(stderr,
                        "llama_window_moe: step=%llu decode=%llu "
                        "prefetch=%llu evict=%llu stale=%llu "
                        "prefetched=%.1f MiB evicted=%.1f MiB\n",
                        (unsigned long long) step,
                        (unsigned long long) ctx.expert_decode_step,
                        (unsigned long long) ctx.stats.expert_prefetch_calls,
                        (unsigned long long) ctx.stats.expert_evict_calls,
                        (unsigned long long) ctx.stats.stale_tasks,
                        ctx.stats.expert_bytes_prefetched / 1024.0 / 1024.0,
                        ctx.stats.expert_bytes_evicted / 1024.0 / 1024.0);
            }
        }
        // CLG mode: prediction happens in node_done(l_out-{L}), not here.
    }
}

void llama_window_node_done(llama_window_context & ctx, const ggml_tensor * node) {
    if (node == nullptr) {
        return;
    }
    const char * name = ggml_get_name(node);

    // ---- CLG prediction: intercept ffn_inp-{L} to predict experts for layer L+1 ----
    // ffn_inp[L] = l_out[L-1] + attn_out[L] is the hidden state AFTER attention
    // but BEFORE the FFN.  The Fate paper (arXiv:2502.12224) shows that adjacent
    // layers' ffn_inp tensors have >83% cosine similarity, making ffn_inp[L] the
    // best available proxy for ffn_inp[L+1] (and therefore for layer L+1's gate
    // scores).  Triggering here gives the entire layer L FFN+expert computation
    // plus layer L+1's attention as an overlap window for prefetch/evict tasks.
    if (ctx.params.clg_predict && !ctx.clg_layers.empty()) {
        int inp_layer = -1;
        if (std::sscanf(name, "ffn_inp-%d", &inp_layer) == 1 &&
                inp_layer >= 0 &&
                inp_layer + 1 < (int) ctx.clg_layers.size()) {
            // CLG: use ffn_inp[L] to predict layer L+1's expert routing.
            // ffn_inp[L] ≈ ffn_inp[L+1] (adjacent layers, high cosine similarity),
            // so applying gate_weight[L+1] to norm(ffn_inp[L]) approximates the
            // actual routing decision before L+1's attention even runs.
            llama_window_clg_predict_and_schedule(ctx, inp_layer + 1, node);
        }
    }

    // ---- MoE expert window: intercept routing decision (ffn_moe_topk-{L}) ----
    // Serves two purposes regardless of LRU vs CLG mode:
    //   1. Update last_used_step for stale-evict protection (both modes).
    //   2. JIT fallback prefetch for cold experts (mispredictions in CLG mode,
    //      or first-token cold-start in LRU mode).
    //   3. Accuracy tracking for CLG mode.
    if (ctx.params.expert_window && !ctx.expert_slots.empty()) {
        int moe_layer = -1;
        if (std::sscanf(name, "ffn_moe_topk-%d", &moe_layer) == 1 &&
                moe_layer >= 0 && moe_layer < (int) ctx.expert_slots.size() &&
                !ctx.expert_slots[moe_layer].empty()) {
            const int n_total = (int) (node->ne[0] * node->ne[1]);
            if (n_total > 0 && node->data != nullptr) {
                const int32_t * ids    = static_cast<const int32_t *>(node->data);
                const int       n_slots = (int) ctx.expert_slots[moe_layer].size();

                // For CLG accuracy tracking: build the set of actually selected experts.
                // We compare against what CLG predicted (predicted_mask is not stored,
                // so we track hits by checking which selected experts are already warm).

                std::lock_guard<std::mutex> lock(ctx.mutex);
                if (ctx.graph_active) {
                    // Deduplicate expert IDs across the batch for accurate stats.
                    uint64_t seen_mask = 0;
                    for (int i = 0; i < n_total; ++i) {
                        const int32_t eid = ids[i];
                        if (eid < 0 || eid >= n_slots || eid >= 64) {
                            continue;
                        }
                        if (seen_mask & (1ull << eid)) continue;
                        seen_mask |= (1ull << eid);

                        auto & slot = ctx.expert_slots[moe_layer][eid];

                        // Stale-evict protection: update last_used_step so any
                        // in-flight evict task will be detected as stale.
                        slot.last_used_step = ctx.expert_token_step;

                        if (ctx.params.clg_predict) {
                            // Track activation frequency for hot-expert detection.
                            if (!ctx.expert_in_prefill) {
                                slot.activation_count++;
                            }
                            // CLG mode: accuracy via predicted_mask (not state).
                            ctx.stats.clg_check_total++;
                            const bool was_predicted =
                                    (moe_layer < (int) ctx.clg_predicted_masks.size()) &&
                                    (ctx.clg_predicted_masks[moe_layer] & (1ull << (uint32_t)eid));
                            if (was_predicted) {
                                ctx.stats.clg_hit_total++;
                            }
                            // JIT fallback for cold/mispredicted experts.
                            if (slot.state == llama_window_state::cold) {
                                llama_window_request_expert_prefetch_locked(
                                        ctx, moe_layer, eid);
                                if (!was_predicted) {
                                    ctx.stats.clg_miss_calls++;
                                    slot.miss_count++;
                                }
                            }
                        } else {
                            // LRU mode: JIT prefetch for cold experts + cross-layer hint.
                            if (slot.state == llama_window_state::cold) {
                                llama_window_request_expert_prefetch_locked(
                                        ctx, moe_layer, eid);
                            }
                            if (!ctx.expert_in_prefill) {
                                slot.activation_count++;
                            }
                            // Cross-layer prefetch hint: same expert at L+1.
                            const bool has_next =
                                    (moe_layer + 1 < (int) ctx.expert_slots.size()) &&
                                    !ctx.expert_slots[moe_layer + 1].empty();
                            if (has_next && eid < (int) ctx.expert_slots[moe_layer+1].size()) {
                                auto & ns = ctx.expert_slots[moe_layer + 1][eid];
                                if (ns.activation_count > 0) {
                                    llama_window_request_expert_prefetch_locked(
                                            ctx, moe_layer + 1, eid);
                                }
                            }
                        }
                        slot.state = llama_window_state::resident;
                    }

                    // Emit CLG accuracy log periodically (decode only)
                    if (ctx.params.clg_predict && ctx.params.debug_log &&
                            !ctx.expert_in_prefill &&
                            ctx.expert_decode_step % 32 == 0 &&
                            moe_layer == 0) {
                        const float acc = ctx.stats.clg_check_total > 0
                                ? 100.f * (float) ctx.stats.clg_hit_total /
                                  (float) ctx.stats.clg_check_total
                                : 0.f;
                        // Count hot experts (per-layer average across layers 0..n)
                        int n_hot_total = 0;
                        const int hot_thr = ctx.params.clg_hot_thr_pct;
                        if (hot_thr > 0 &&
                                ctx.expert_decode_step >= (uint64_t)ctx.params.clg_hot_warmup) {
                            for (int il = 0; il < (int)ctx.expert_slots.size(); ++il) {
                                for (int e = 0; e < (int)ctx.expert_slots[il].size(); ++e) {
                                    if (ctx.expert_slots[il][e].activation_count * 100 >
                                            hot_thr * ctx.expert_decode_step) {
                                        n_hot_total++;
                                    }
                                }
                            }
                        }
                        const int n_layers_with_experts = (int)ctx.expert_slots.size();
                        const float hot_avg = n_layers_with_experts > 0
                                ? (float)n_hot_total / n_layers_with_experts : 0.f;
                        std::fprintf(stderr,
                                "llama_window_clg: step=%llu decode=%llu "
                                "predict=%llu evict=%llu miss=%llu acc=%.1f%% "
                                "hot=%.1f/layer(>%d%%)\n",
                                (unsigned long long) ctx.expert_token_step,
                                (unsigned long long) ctx.expert_decode_step,
                                (unsigned long long) ctx.stats.clg_predict_calls,
                                (unsigned long long) ctx.stats.clg_evict_calls,
                                (unsigned long long) ctx.stats.clg_miss_calls,
                                acc,
                                hot_avg,
                                hot_thr);
                    }

                }
            }
        }
    }

    // ---- Layer window: advance per-layer prefetch ----
    const int layer = llama_window_parse_layer(name);
    if (layer < 0 || layer >= ctx.n_layers) {
        return;
    }
    if (ctx.observed_layer.exchange(layer, std::memory_order_relaxed) == layer) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    if (!ctx.graph_active) {
        return;
    }
    llama_window_advance_locked(ctx, layer, true);
}

void llama_window_graph_end(llama_window_context & ctx) {
    if (!llama_window_enabled(&ctx)) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.graph_active = false;
}

const llama_window_stats & llama_window_get_stats(const llama_window_context & ctx) {
    return ctx.stats;
}
