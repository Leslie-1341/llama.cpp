#include "llama-flex.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
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
    return slot;
}

// Read `size` bytes at `foff` from file `file_idx` into `dst`. When O_DIRECT is
// active, the read is issued on a block-aligned superset into the per-thread
// `bounce` buffer and the exact bytes are copied out; otherwise a plain pread
// loop is used. `bcap` is the bounce capacity. Returns true on success.
static bool flex_read(llama_flex_context * ctx,
                      uint8_t * dst, uint16_t file_idx, size_t foff, size_t size,
                      uint8_t * bounce, size_t bcap) {
    const int fd = ctx->fds[file_idx];
    if (!ctx->direct_io_active) {
        size_t left = size; off_t off = (off_t) foff; uint8_t * d = dst;
        while (left > 0) {
            ssize_t r = pread(fd, d, left, off);
            if (r <= 0) return false;
            d += r; off += r; left -= (size_t) r;
        }
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
    return true;
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
        for (const auto & t : L.tensors) {
            if (t.locked) {
                continue; // locked tensors live permanently in the lock buffer
            }
            if (!flex_read(ctx, base + t.buf_offset, t.file_idx, t.file_offset, t.size, bounce, bcap)) {
                ok = false;
                break;
            }
            streamed += t.size;
        }
        const uint64_t dt = now_us() - t0;

        {
            std::lock_guard<std::mutex> lock(ctx->mutex);
            if (ok) {
                L.state    = layer_state::resident;
                L.last_use = now_us();
                ctx->stats.layer_loads++;
                ctx->stats.bytes_streamed += streamed;
                ctx->stats.total_io_us    += dt;
            } else {
                // Failed: drop the slot back.
                ctx->slot_layer[L.slot] = -1;
                L.slot  = -1;
                L.state = layer_state::not_resident;
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

    // Balanced memory locking: give every layer the same lock budget so the
    // per-layer streamed IO stays uniform (avoids pipeline stalls). Within a
    // layer, lock tensors in registration order until the budget is hit; since
    // all layers share the same tensor structure this locks the same set per
    // layer. The remaining (unlocked) tensors are streamed through the ring.
    const size_t per_layer_lock = ctx.n_layers > 0
            ? ctx.params.lock_bytes / (size_t) ctx.n_layers : 0;

    size_t lock_total = 0;
    for (auto & L : ctx.layers) {
        // Flexible tensor preservation: lock smallest tensors first. In a
        // transformer layer the attention projections (and, under GQA, K/V in
        // particular) are smaller than the FFN tensors, so smallest-first locks
        // as many tensors as the budget allows -- eliminating the most per-token
        // IO operations -- while leaving the large FFN tensors to efficient
        // large streaming reads. Identical layer structure => the same set is
        // locked in every layer, keeping per-layer streamed IO uniform.
        std::sort(L.tensors.begin(), L.tensors.end(),
                [](const llama_flex_tensor & a, const llama_flex_tensor & b) {
                    return a.size < b.size;
                });
        size_t locked_here = 0;
        size_t stream_off  = 0;
        for (auto & t : L.tensors) {
            if (locked_here + t.size <= per_layer_lock) {
                t.locked     = true;
                t.buf_offset = lock_total;   // offset into the global lock buffer
                lock_total  += t.size;
                locked_here += t.size;
            } else {
                t.locked     = false;
                t.buf_offset = stream_off;   // offset within this layer's slot
                stream_off  += t.size;
            }
        }
        L.stream_bytes    = stream_off;
        L.always_resident = (stream_off == 0);
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
                "locked=%.2f MiB stream/token=%.2f MiB io_threads=%d direct_io=%d\n",
                ctx.n_layers, k, ctx.slot_bytes / 1048576.0,
                ctx.stats.ring_bytes / 1048576.0,
                ctx.stats.locked_bytes / 1048576.0,
                ctx.stats.stream_per_token / 1048576.0,
                nthreads, ctx.direct_io_active ? 1 : 0);
    }
}

void llama_flex_request_layer(llama_flex_context & ctx, int layer_id) {
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
    ctx.queue.push_back(layer_id);
    ctx.cv_work.notify_one();
}

void llama_flex_wait_layer(llama_flex_context & ctx, int layer_id) {
    if (layer_id < 0 || layer_id >= ctx.n_layers) {
        return;
    }
    std::unique_lock<std::mutex> lock(ctx.mutex);
    auto & L = ctx.layers[layer_id];
    if (L.state == layer_state::resident) {
        return;
    }
    // Make sure it is at least queued.
    if (L.state == layer_state::not_resident) {
        ctx.queue.push_front(layer_id);
        ctx.cv_work.notify_one();
    }
    const uint64_t t0 = now_us();
    ctx.stats.wait_events++;
    ctx.cv_ready.wait(lock, [&] {
        return L.state == layer_state::resident || ctx.shutdown;
    });
    ctx.stats.total_wait_us += now_us() - t0;
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
}

const llama_flex_stats & llama_flex_get_stats(const llama_flex_context & ctx) {
    return ctx.stats;
}

void llama_flex_graph_begin(llama_flex_context & ctx) {
    if (!llama_flex_enabled(&ctx)) {
        return;
    }
    ctx.cur_compute_layer = -1;
    const int ahead = std::max(1, std::min(ctx.params.prefetch_ahead, ctx.n_layers - 1));
    for (int l = 0; l <= ahead; ++l) {
        llama_flex_request_layer(ctx, l % ctx.n_layers);
    }
}

bool llama_flex_stream_callback(struct ggml_tensor * op, int ith, void * user_data) {
    auto * ctx = static_cast<llama_flex_context *>(user_data);
    if (ctx == nullptr || op == nullptr) {
        return false;
    }

    int op_layer = -1;
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
            llama_flex_wait_layer(*ctx, op_layer);
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
        ctx->cur_compute_layer = op_layer;
        const int ahead = std::max(1, std::min(ctx->params.prefetch_ahead, ctx->n_layers - 1));
        for (int a = 1; a <= ahead; ++a) {
            llama_flex_request_layer(*ctx, (op_layer + a) % ctx->n_layers);
        }
        if (prev >= 0) {
            llama_flex_release_layer(*ctx, prev);
        }
    }

    return true;
}
