#include "llama-window.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
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
    int priority = 0;
    uint64_t graph_seq = 0;
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
            priority,
            ctx.stats.graph_seq,
    });
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
                int advice = MADV_NORMAL;
#if defined(MADV_DONTNEED)
                advice = MADV_DONTNEED;
#endif
                if (madvise(range.addr, range.size, advice) != 0) {
                    success = false;
                } else {
                    bytes += range.size;
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
        const llama_window_params & params) {
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

    for (const auto & input : inputs) {
        int layer = -1;
        if (std::sscanf(input.name.c_str(), "blk.%d.", &layer) != 1 ||
                layer < 0 || layer >= n_layers ||
                input.addr == nullptr || input.size == 0) {
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
}

void llama_window_node_done(llama_window_context & ctx, const ggml_tensor * node) {
    if (node == nullptr) {
        return;
    }
    const int layer = llama_window_parse_layer(ggml_get_name(node));
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
