#include "llama-moe-buffer.h"

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdlib>
#endif

namespace {
size_t g_page = 4096;

// Per-expert residency state.
enum moe_slot_state : char {
    ST_COLD     = 0,   // not in buffer
    ST_RESIDENT = 1,   // streamed in, occupies a slot, lives in the LRU list
    ST_INFLIGHT = 2,   // a thread (sync callback or worker) is currently pread'ing it
};

// O_DIRECT bounce buffer, one per thread so the synchronous weight-stream
// callback and the async prefetch worker never share it (their pread()s run
// outside the residency mutex and may overlap). Leaked at thread exit.
thread_local uint8_t * tls_bounce = nullptr;
thread_local size_t    tls_bcap   = 0;
}

// Metadata for one *_exps tensor (keyed by tensor name). The anonymous buffer and
// the ->data repoint are done lazily on first compute, because the tensor object
// seen at load (weights_map) differs from the one used in the compute graph; the
// callback receives the real compute tensor and repoints THAT.
struct moe_managed {
    std::string   name;
    int           layer        = -1;    // parsed from "blk.N." for by-layer prefetch
    int           fd           = -1;    // dup'd model fd (O_DIRECT when available)
    bool          direct       = false;
    size_t        fsize        = 0;     // file size (EOF clamp for O_DIRECT)
    size_t        file_offset  = 0;     // base file offset of the exps tensor
    size_t        stride       = 0;     // per-expert byte stride (nb[2])
    int           n_expert     = 0;
    size_t        nbytes       = 0;     // full tensor size

    ggml_tensor * tensor = nullptr;     // bound on first compute (the repointed one)
    uint8_t *     buf    = nullptr;     // anon, nbytes; allocated on first use
    std::vector<char> resident;         // moe_slot_state per expert
    std::vector<uint32_t> activation;   // per-expert touch count (for hot pinning)
    uint64_t      total_act = 0;        // sum of activation[] (denominator for hot ratio)
    std::vector<std::list<std::pair<moe_managed *, int>>::iterator> lru_pos;
};

struct llama_moe_buffer_context {
    llama_moe_buffer_params params;

    std::unordered_map<std::string, moe_managed>      by_name;
    std::unordered_map<int, std::vector<moe_managed*>> by_layer;
    std::vector<int> fds;

    // Residency state (resident[], lru, resident_bytes, buf allocation).
    std::mutex              mtx;
    std::condition_variable cv_done;   // signalled when an in-flight slice completes
    std::list<std::pair<moe_managed *, int>> lru;  // front = MRU
    size_t resident_bytes = 0;
    size_t expert_total   = 0;   // sum of all registered exps tensor bytes
    size_t align          = 4096;

    // Async prefetch worker pool + its own queue lock (kept separate from mtx so
    // enqueueing never blocks on an in-progress stream). Multiple workers issue
    // pread() concurrently (each with its own thread_local bounce), which lifts
    // effective read bandwidth on NVMe where a single O_DIRECT random-read stream
    // under-utilises the device.
    std::vector<std::thread> workers;
    std::mutex              qmtx;
    std::condition_variable cv_q;
    std::deque<std::pair<moe_managed *, int>> queue;
    bool                    stop = false;

    std::atomic<uint64_t> streams{0}, hits{0}, evictions{0}, bytes_read{0};
    std::atomic<uint64_t> enqueued{0}, worker_streams{0};

    ~llama_moe_buffer_context() {
        {
            std::lock_guard<std::mutex> lock(qmtx);
            stop = true;
        }
        cv_q.notify_all();
        for (auto & w : workers) {
            if (w.joinable()) {
                w.join();
            }
        }
        if (params.debug_log) {
            std::fprintf(stderr,
                "llama_moe_buffer[teardown]: tensors=%zu resident=%.1f MiB budget=%.1f MiB "
                "streams=%llu (worker=%llu) hits=%llu evictions=%llu enqueued=%llu read=%.1f MiB\n",
                by_name.size(), resident_bytes/1048576.0, params.budget_bytes/1048576.0,
                (unsigned long long)streams.load(), (unsigned long long)worker_streams.load(),
                (unsigned long long)hits.load(), (unsigned long long)evictions.load(),
                (unsigned long long)enqueued.load(), bytes_read.load()/1048576.0);
        }
        for (auto & kv : by_name) {
            free(kv.second.buf);
        }
        for (int fd : fds) {
            if (fd >= 0) close(fd);
        }
    }
};

// ---- forward decls of internals ----
static void moe_stream_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e);

static void moe_worker_loop(llama_moe_buffer_context * ctx) {
    for (;;) {
        std::pair<moe_managed *, int> task;
        {
            std::unique_lock<std::mutex> lk(ctx->qmtx);
            ctx->cv_q.wait(lk, [ctx] { return ctx->stop || !ctx->queue.empty(); });
            if (ctx->stop && ctx->queue.empty()) {
                return;
            }
            task = ctx->queue.front();
            ctx->queue.pop_front();
        }
        moe_stream_slice(*ctx, *task.first, task.second);
        ctx->worker_streams.fetch_add(1, std::memory_order_relaxed);
    }
}

std::shared_ptr<llama_moe_buffer_context> llama_moe_buffer_create(const llama_moe_buffer_params & params) {
    auto ctx = std::make_shared<llama_moe_buffer_context>();
    ctx->params = params;
#if defined(_SC_PAGESIZE)
    g_page = (size_t) sysconf(_SC_PAGESIZE);
#endif
    ctx->align = g_page;
    if (params.enabled) {
        const int n = std::max(1, params.n_workers);
        ctx->workers.reserve(n);
        for (int i = 0; i < n; ++i) {
            ctx->workers.emplace_back(moe_worker_loop, ctx.get());
        }
    }
    return ctx;
}

bool llama_moe_buffer_enabled(const llama_moe_buffer_context * ctx) {
    return ctx != nullptr && ctx->params.enabled && !ctx->by_name.empty();
}

bool llama_moe_buffer_register(
        llama_moe_buffer_context & ctx,
        ggml_tensor *              exps,
        int                        fd,
        size_t                     file_offset,
        size_t                     expert_stride,
        int                        n_expert) {
    if (!ctx.params.enabled || exps == nullptr || n_expert <= 0 || expert_stride == 0) {
        return false;
    }
    // Reopen the fd, preferring O_DIRECT so streamed expert pages bypass the page
    // cache (the exps region is still mmap-mapped, which would otherwise pin any
    // buffered-read cache pages and defeat the footprint reduction).
    int  dfd    = -1;
    bool direct = false;
#if defined(__linux__)
    char proc[64];
    std::snprintf(proc, sizeof(proc), "/proc/self/fd/%d", fd);
#if defined(O_DIRECT)
    dfd = open(proc, O_RDONLY | O_DIRECT);
    if (dfd >= 0) direct = true;
#endif
    if (dfd < 0) dfd = open(proc, O_RDONLY);
#endif
    if (dfd < 0) dfd = dup(fd);
    if (dfd < 0) {
        return false;
    }
    ctx.fds.push_back(dfd);

    struct stat st;
    const size_t fsize = (fstat(dfd, &st) == 0) ? (size_t) st.st_size : SIZE_MAX;

    moe_managed m;
    m.name        = ggml_get_name(exps);
    m.fd          = dfd;
    m.direct      = direct;
    m.fsize       = fsize;
    m.file_offset = file_offset;
    m.stride      = expert_stride;
    m.n_expert    = n_expert;
    m.nbytes      = ggml_nbytes(exps);
    m.resident.assign(n_expert, ST_COLD);
    m.activation.assign(n_expert, 0);
    m.lru_pos.resize(n_expert);
    std::sscanf(m.name.c_str(), "blk.%d.", &m.layer);

    auto res = ctx.by_name.emplace(m.name, std::move(m));
    if (res.second) {
        ctx.expert_total += res.first->second.nbytes;
        if (res.first->second.layer >= 0) {
            ctx.by_layer[res.first->second.layer].push_back(&res.first->second);
        }
    }
    return true;
}

size_t llama_moe_buffer_expert_bytes(const llama_moe_buffer_context * ctx) {
    return ctx != nullptr ? ctx->expert_total : 0;
}

void llama_moe_buffer_set_budget(llama_moe_buffer_context * ctx, size_t budget_bytes) {
    if (ctx == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(ctx->mtx);
    ctx->params.budget_bytes = budget_bytes;
}

// Allocate the anonymous buffer for a managed tensor on first use. Cold expert
// slots stay zero-fill / unbacked. Caller must hold ctx.mtx.
static bool moe_ensure_buf(moe_managed & m) {
    if (m.buf != nullptr) {
        return true;
    }
    uint8_t * buf = nullptr;
    if (posix_memalign((void **) &buf, g_page, m.nbytes) != 0 || buf == nullptr) {
        return false;
    }
    m.buf = buf;
    return true;
}

// Is expert e of m "hot"? Relative criterion: its activation count exceeds
// hot_ratio * the tensor's mean activation (total_act / n_expert). Bounded — a
// pinned set defined by "above hot_ratio× average" cannot grow to all experts as
// the sequence lengthens (numerator and denominator scale together). A short
// warmup (every expert seen ~twice on average) gates it so early noise doesn't pin.
static bool moe_is_hot(const llama_moe_buffer_context & ctx, const moe_managed & m, int e) {
    if (ctx.params.hot_ratio <= 0.0f) return false;
    if (m.total_act < (uint64_t) m.n_expert * 2) return false;   // warmup
    return (double) m.activation[e] * m.n_expert >
           (double) ctx.params.hot_ratio * (double) m.total_act;
}

// Evict the coldest non-hot slot. Caller must hold ctx.mtx. Scans the LRU list
// from the back (coldest) and skips experts that are "hot" (pinned — never
// evicted). Returns false if nothing is evictable (list empty or all hot).
static bool moe_evict_lru(llama_moe_buffer_context & ctx) {
    auto victim = ctx.lru.end();
    if (ctx.params.hot_ratio <= 0.0f) {
        if (ctx.lru.empty()) return false;
        victim = std::prev(ctx.lru.end());           // pure LRU: coldest
    } else {
        for (auto it = ctx.lru.end(); it != ctx.lru.begin(); ) {
            --it;
            if (!moe_is_hot(ctx, *it->first, it->second)) {
                victim = it;                            // coldest non-hot
                break;
            }
        }
        if (victim == ctx.lru.end()) return false;     // all resident slots are hot
    }
    moe_managed * m = victim->first;
    const int     e = victim->second;
    ctx.lru.erase(victim);
    const uintptr_t base = (uintptr_t) m->buf + (uintptr_t) e * m->stride;
    const uintptr_t pb   = (base + g_page - 1) & ~(uintptr_t) (g_page - 1);
    const uintptr_t pe   = (base + m->stride) & ~(uintptr_t) (g_page - 1);
#if defined(MADV_DONTNEED)
    if (pe > pb) madvise((void *) pb, pe - pb, MADV_DONTNEED);  // anon → freed
#endif
    m->resident[e] = ST_COLD;
    ctx.resident_bytes -= std::min(ctx.resident_bytes, m->stride);
    ctx.evictions.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Read expert slice e of m into m.buf + e*stride. No shared state: uses a
// thread-local bounce, so it is safe to run outside ctx.mtx. Returns true on
// success.
static bool moe_pread_slice(moe_managed & m, int e, size_t align) {
    const size_t foff = m.file_offset + (size_t) e * m.stride;
    uint8_t *    dst  = m.buf + (size_t) e * m.stride;
    if (m.direct) {
        const size_t A    = align;
        const size_t aoff = foff & ~(A - 1);
        const size_t head = foff - aoff;
        size_t want = (head + m.stride + A - 1) & ~(A - 1);
        if (aoff + want > m.fsize) want = m.fsize - aoff;
        if (head + m.stride > tls_bcap) {
            free(tls_bounce);
            tls_bcap = ((head + m.stride + 2 * A) + A - 1) & ~(A - 1);
            if (posix_memalign((void **) &tls_bounce, A, tls_bcap) != 0) {
                tls_bounce = nullptr;
                tls_bcap   = 0;
            }
        }
        if (tls_bounce == nullptr || want > tls_bcap) {
            return false;
        }
        ssize_t r = pread(m.fd, tls_bounce, want, (off_t) aoff);
        if (r < 0 || (size_t) r < head + m.stride) {
            return false;
        }
        std::memcpy(dst, tls_bounce + head, m.stride);
        return true;
    }
    // Buffered fallback (non-block device etc.).
    size_t left = m.stride; off_t off = (off_t) foff; uint8_t * d = dst;
    while (left > 0) {
        ssize_t r = pread(m.fd, d, left, off);
        if (r <= 0) return false;
        d += r; off += r; left -= (size_t) r;
    }
#if defined(POSIX_FADV_DONTNEED)
    posix_fadvise(m.fd, (off_t) foff, (off_t) m.stride, POSIX_FADV_DONTNEED);
#endif
    return true;
}

// Ensure expert slice e of m is resident. Coordinates the synchronous callback
// and the async worker via a per-slot state machine: the pread happens outside
// the residency mutex; a second thread requesting the same slice waits for the
// in-flight read to complete instead of issuing a duplicate read.
static void moe_stream_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e) {
    if (e < 0 || e >= m.n_expert) return;

    std::unique_lock<std::mutex> lk(ctx.mtx);
    for (;;) {
        const char s = m.resident[e];
        if (s == ST_RESIDENT) {
            ctx.lru.splice(ctx.lru.begin(), ctx.lru, m.lru_pos[e]);  // touch (MRU)
            if (m.activation[e] != UINT32_MAX) { m.activation[e]++; m.total_act++; }  // hot tracking
            ctx.hits.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        if (s == ST_INFLIGHT) {
            ctx.cv_done.wait(lk);   // another thread is loading it; wait for completion
            continue;
        }
        break;  // ST_COLD → we load it
    }

    if (!moe_ensure_buf(m)) {
        return;
    }
    m.resident[e] = ST_INFLIGHT;
    if (ctx.params.budget_bytes > 0) {
        while (ctx.resident_bytes + m.stride > ctx.params.budget_bytes && moe_evict_lru(ctx)) {}
    }
    ctx.resident_bytes += m.stride;   // reserve before the (unlocked) read

    lk.unlock();
    const bool ok = moe_pread_slice(m, e, ctx.align);
    lk.lock();

    if (ok) {
        m.resident[e] = ST_RESIDENT;
        ctx.lru.push_front({&m, e});
        m.lru_pos[e] = ctx.lru.begin();
        if (m.activation[e] != UINT32_MAX) { m.activation[e]++; m.total_act++; }  // hot tracking
        ctx.streams.fetch_add(1, std::memory_order_relaxed);
        ctx.bytes_read.fetch_add(m.stride, std::memory_order_relaxed);
    } else {
        m.resident[e] = ST_COLD;
        ctx.resident_bytes -= std::min(ctx.resident_bytes, m.stride);
        if (ctx.params.debug_log) {
            std::fprintf(stderr, "llama_moe_buffer: stream failed %s expert %d\n", m.name.c_str(), e);
        }
    }
    ctx.cv_done.notify_all();
}

void llama_moe_buffer_prefetch(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        int                        n_experts) {
    if (ctx == nullptr || !ctx->params.enabled || experts == nullptr || n_experts <= 0) {
        return;
    }
    auto it = ctx->by_layer.find(layer);
    if (it == ctx->by_layer.end()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(ctx->qmtx);
        for (moe_managed * m : it->second) {
            for (int i = 0; i < n_experts; ++i) {
                const int e = experts[i];
                if (e < 0 || e >= m->n_expert) continue;
                // Cheap pre-filter: skip slices already resident. The state read is
                // benign w.r.t. mtx (a stale COLD just enqueues a no-op hit later).
                if (m->resident[e] == ST_RESIDENT) continue;
                ctx->queue.push_back({m, e});
                ctx->enqueued.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    ctx->cv_q.notify_one();
}

bool llama_moe_buffer_stream_callback(ggml_tensor * op, int ith, void * user_data) {
    auto * ctx = static_cast<llama_moe_buffer_context *>(user_data);
    if (ctx == nullptr || op == nullptr || op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }
    ggml_tensor * exps = op->src[0];
    if (exps == nullptr) return false;
    auto it = ctx->by_name.find(ggml_get_name(exps));
    if (it == ctx->by_name.end()) {
        return false;  // not a managed expert GEMM
    }
    moe_managed & m = it->second;

    if (ith == 0) {
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            if (!moe_ensure_buf(m)) {
                return false;
            }
            m.tensor = exps;
        }
        // Repoint the live compute tensor at our anonymous buffer (graph
        // reallocation could in principle reset ->data between graphs).
        if (exps->data != m.buf) {
            exps->data = m.buf;
        }
        ggml_tensor * ids = op->src[2];  // selected experts, I32
        if (ids != nullptr && ids->data != nullptr && ids->type == GGML_TYPE_I32) {
            const int32_t * idp = static_cast<const int32_t *>(ids->data);
            const int64_t   n   = ggml_nelements(ids);
            for (int64_t i = 0; i < n; ++i) {
                moe_stream_slice(*ctx, m, (int) idp[i]);  // resident or wait-for-inflight
            }
        }
    }
    return true;  // managed → caller issues a barrier
}

void llama_moe_buffer_print_stats(const llama_moe_buffer_context & ctx) {
    std::fprintf(stderr,
            "llama_moe_buffer: tensors=%zu resident=%.1f MiB streams=%llu hits=%llu evictions=%llu enqueued=%llu\n",
            ctx.by_name.size(), ctx.resident_bytes / 1048576.0,
            (unsigned long long) ctx.streams.load(), (unsigned long long) ctx.hits.load(),
            (unsigned long long) ctx.evictions.load(), (unsigned long long) ctx.enqueued.load());
}
