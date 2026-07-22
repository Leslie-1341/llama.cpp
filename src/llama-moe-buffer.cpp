#include "llama-moe-buffer.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <cstring>
#include <deque>
#include <cmath>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define LLAMA_MOE_CAN_COMPILE_AVX2 1
#define LLAMA_MOE_CAN_COMPILE_AVX512 1
#endif

#if defined(__unix__) || defined(__APPLE__)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdlib>
#endif

struct moe_managed;

namespace {
size_t g_page = 4096;
static constexpr uint16_t MOE_SIDECAR_CODEC_RAW = 0;
static constexpr uint16_t MOE_SIDECAR_CODEC_GGML_QUANT = 1;
static constexpr uint16_t MOE_SIDECAR_CODEC_MWQ = 2;
static constexpr uint16_t MOE_SIDECAR_CODEC_MWQ_HIER = 3;
static constexpr int      MOE_MWQ_BLOCK = 32;
static constexpr int      MOE_MWQ_BLOCK_DEFAULT = 32;
static constexpr int      MOE_MWQ_SCALE_GROUP_DEFAULT = 16;
static constexpr uint64_t MOE_STREAM_FAIL_LOG_EVERY = 128; // throttle "MWQ stream failed" to 1-in-N occurrences

enum moe_kernel_backend : uint8_t {
    MOE_KERNEL_MWQ = 0,
    MOE_KERNEL_Q4K_NATIVE = 1,
};

enum class moe_tensor_kind : uint8_t {
    other = 0,
    gate,
    up,
    down,
};

// Why moe_op_all_mwq_resident()/moe_op_all_mwq_available() returned false, for
// direct-SWIGLU miss attribution. RANGE/UNAVAILABLE only — "entry missing" vs
// "read failed" is captured separately via moe_stream_mwq_slice()'s out-params,
// since by the time these residency checks run, a load attempt already happened.
enum class moe_resident_fail_reason : uint8_t {
    NONE = 0,
    RANGE = 1,
    UNAVAILABLE = 2,
};

static constexpr std::array<uint32_t, 4096> mwq_make_q3_lut4() {
    std::array<uint32_t, 4096> lut = {};
    for (uint32_t p = 0; p < lut.size(); ++p) {
        lut[p] =
            (((p >> 0) & 7u) <<  0) |
            (((p >> 3) & 7u) <<  8) |
            (((p >> 6) & 7u) << 16) |
            (((p >> 9) & 7u) << 24);
    }
    return lut;
}

static constexpr std::array<uint64_t, 65536> mwq_make_q2_lut8() {
    std::array<uint64_t, 65536> lut = {};
    for (uint32_t p = 0; p < lut.size(); ++p) {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) {
            v |= (uint64_t) ((p >> (2 * i)) & 3u) << (8 * i);
        }
        lut[p] = v;
    }
    return lut;
}

static constexpr auto MWQ_Q2_LUT8 = mwq_make_q2_lut8();
static constexpr auto MWQ_Q3_LUT4 = mwq_make_q3_lut4();

struct mwq_kernel_pair {
    moe_kernel_backend backend = MOE_KERNEL_MWQ;
    int expert = -1;
    int bits = 0;
    int block_size = MOE_MWQ_BLOCK_DEFAULT;
    int codec = MOE_SIDECAR_CODEC_MWQ;
    int scale_group = MOE_MWQ_SCALE_GROUP_DEFAULT;
    int outlier_max = 0;
    int64_t pair = 0;
    int64_t token = 0;
    int64_t rank = 0;
    int64_t n_elem = 0;
    const uint8_t * mwq = nullptr;
    size_t native_row_size = 0;
    ggml_type native_type = GGML_TYPE_COUNT;
    const char * x = nullptr;
    float * dst = nullptr;
};

struct mwq_sum_cache_entry {
    const char * x = nullptr;
    int block_size = 0;
    int64_t n_col = 0;
    std::vector<float> sums;
};

struct mwq_qx_int8_cache_entry {
    const char * x = nullptr;
    int qblock = 0;
    int64_t n_col = 0;
    std::vector<int8_t> qx;
    std::vector<float> scales;
    std::vector<int32_t> sums;
};

struct mwq_transient_slice {
    ::moe_managed * m = nullptr;
    int expert = -1;
    int bits = 0;
};

struct mwq_swiglu_pair {
    mwq_kernel_pair gate;
    mwq_kernel_pair up;
};

struct mwq_ffn_triplet {
    mwq_kernel_pair gate;
    mwq_kernel_pair up;
    mwq_kernel_pair down;
};

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
thread_local std::vector<uint8_t> tls_encoded;
thread_local std::vector<float>   tls_f32;
thread_local std::vector<float>   tls_mwq_sum_x;
thread_local std::vector<float>   tls_mwq_sum_x_batch;
thread_local std::vector<mwq_kernel_pair> tls_mwq_pairs;
thread_local std::vector<mwq_kernel_pair> tls_mwq_pairs_sorted;
thread_local std::vector<mwq_swiglu_pair> tls_mwq_swiglu_pairs;
thread_local std::vector<mwq_swiglu_pair> tls_mwq_swiglu_pairs_sorted;
thread_local std::vector<mwq_ffn_triplet> tls_mwq_ffn_triplets;
thread_local std::vector<mwq_ffn_triplet> tls_mwq_ffn_triplets_sorted;
thread_local std::vector<float> tls_mwq_ffn_swiglu_tile;
thread_local std::vector<int> tls_mwq_counts;
thread_local std::vector<int> tls_mwq_offsets;
thread_local std::vector<int> tls_mwq_cursor;
thread_local std::vector<mwq_sum_cache_entry> tls_mwq_sum_cache;
thread_local std::vector<mwq_qx_int8_cache_entry> tls_mwq_qx_int8_cache;
thread_local std::vector<const char *> tls_mwq_qx_unique_xs;
thread_local const std::vector<mwq_qx_int8_cache_entry> * tls_mwq_qx_shared_cache = nullptr;
thread_local bool tls_mwq_keep_sum_cache = false;
thread_local std::vector<uint8_t> tls_native_x;
thread_local uint64_t tls_mwq_avx512_q2_swiglu_calls = 0;
thread_local uint64_t tls_mwq_avx512_q2_swiglu_pairs = 0;
thread_local uint64_t tls_mwq_avx512_q2_swiglu_rows = 0;
thread_local uint64_t tls_mwq_avx512_q2_dot_calls = 0;
thread_local uint64_t tls_mwq_avx512_q2_dot_pairs = 0;
thread_local uint64_t tls_mwq_avx512_q2_dot_rows = 0;
thread_local uint64_t tls_mwq_vnni_q2_swiglu_calls = 0;
thread_local uint64_t tls_mwq_vnni_q2_swiglu_pairs = 0;
thread_local uint64_t tls_mwq_vnni_q2_swiglu_rows = 0;
thread_local uint64_t tls_mwq_vnni_q2_dot_calls = 0;
thread_local uint64_t tls_mwq_vnni_q2_dot_pairs = 0;
thread_local uint64_t tls_mwq_vnni_q2_dot_rows = 0;
}

static uint64_t moe_profile_now_ns() {
#if defined(CLOCK_MONOTONIC_RAW)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
#else
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

struct moe_mwq_store {
    uint8_t * buf = nullptr;
    size_t stride = 0;
    int block_size = MOE_MWQ_BLOCK_DEFAULT;
    int codec = MOE_SIDECAR_CODEC_MWQ;
    int scale_group = MOE_MWQ_SCALE_GROUP_DEFAULT;
    int outlier_max = 0;
    std::vector<char> loaded;
};

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
    ggml_type     type         = GGML_TYPE_COUNT;
    int64_t       elems        = 0;     // per-expert element count

    ggml_tensor * tensor = nullptr;     // bound on first compute (the repointed one)
    uint8_t *     buf    = nullptr;     // anon, nbytes; allocated on first use
    std::unordered_map<int, moe_mwq_store> mwq;
    std::vector<char> resident;         // moe_slot_state per expert
    std::vector<char> resident_mwq;      // resident slot stores MWQ encoded data, not original tensor layout
    std::vector<int>  resident_bits;     // precision currently materialized in buf
    std::vector<size_t> resident_size;   // bytes charged to resident budget
    std::vector<char> resident_touched;  // touched by a real routed op since load
    std::vector<char> prefetched;        // loaded by async prefetch and not yet used
    std::vector<uint64_t> last_evicted_token;
    std::vector<char> queued;           // async prefetch already queued
    std::vector<int>  queued_bits;       // highest precision still pending for this expert
    std::vector<uint32_t> activation;   // per-expert touch count (for hot pinning)
    uint64_t      total_act = 0;        // sum of activation[] (denominator for hot ratio)
    std::vector<std::list<std::pair<moe_managed *, int>>::iterator> lru_pos;
};

struct moe_prefetch_task {
    moe_managed * m = nullptr;
    int e = -1;
    int rank = 0;
    int priority = 0;
    int target_bits = 0;
    float score = 0.0f;
    uint64_t seq = 0;
};

struct moe_prefetch_task_less {
    bool operator()(const moe_prefetch_task & a, const moe_prefetch_task & b) const {
        if (a.priority != b.priority) {
            return a.priority < b.priority;
        }
        return a.seq > b.seq;
    }
};

struct moe_sidecar_entry {
    uint64_t offset       = 0;
    uint64_t encoded_size = 0;
    uint64_t decoded_size = 0;
    uint64_t cache_offset = 0;
    uint16_t codec        = 0;
    int      expert       = -1;
    int      bits         = 0;
    int      block_size   = MOE_MWQ_BLOCK_DEFAULT;
    int      scale_group  = MOE_MWQ_SCALE_GROUP_DEFAULT;
    int      outlier_max  = 0;
};

struct moe_group_state {
    int layer = -1;
    int expert = -1;
    uint64_t access = 0;
    uint64_t rank0_access = 0;
    uint64_t last_used_epoch = 0;
    uint64_t last_used_token_epoch = 0;
    uint64_t next_use_epoch = UINT64_MAX;
    uint64_t pin_epoch = 0;
    double hot_score = 0.0;
    bool pinned = false;

    // Sequence-level Expert Activation Matrix (EAM). The first step used these
    // fields only for observability; the activation-aware cache policy now also
    // uses them to score pinned groups and eviction victims.
    uint64_t seq_access = 0;
    uint64_t seq_rank0_access = 0;
    uint64_t seq_first_token_epoch = 0;
    uint64_t seq_last_token_epoch = 0;
    uint64_t seq_future_hints = 0;
    uint64_t seq_future_rank0_hints = 0;
    uint64_t seq_prefetch_queued = 0;
    uint64_t seq_prefetch_hits = 0;
    uint64_t seq_prefetch_late = 0;
    uint64_t seq_prefetch_unused = 0;
    uint64_t seq_cache_hits = 0;
    uint64_t seq_cache_misses = 0;
    uint64_t seq_predicted = 0;
    uint64_t seq_pred_enqueued = 0;
    double   seq_future_score_sum = 0.0;
    double   seq_predict_score_sum = 0.0;
    std::array<uint64_t, 4> seq_tensor_access = {0, 0, 0, 0}; // other, gate, up, down
};

struct moe_eamc_snapshot {
    uint64_t id = 0;
    uint64_t start_token = 0;
    uint64_t end_token = 0;
    uint64_t total = 0;
    double norm = 0.0;
    std::unordered_map<uint64_t, uint32_t> counts;
};

struct llama_moe_buffer_context;
static void moe_print_stats_impl(const llama_moe_buffer_context & ctx, const char * prefix);
static void moe_eamc_snapshot_current_locked(llama_moe_buffer_context & ctx, bool force);
static void moe_eam_trace_append_locked(const llama_moe_buffer_context & ctx);
static void moe_eamc_load_sidecar(llama_moe_buffer_context & ctx);

struct llama_moe_buffer_context {
    llama_moe_buffer_params params;

    std::unordered_map<std::string, moe_managed>      by_name;
    std::unordered_map<int, std::vector<moe_managed*>> by_layer;
    std::unordered_map<uint64_t, moe_group_state> groups;
    std::vector<int> fds;
    int sidecar_fd = -1;
    uint32_t sidecar_bits_mask = 0;
    int cache_fd = -1;
    uint64_t cache_next = 0;
    std::unordered_map<std::string, moe_sidecar_entry> sidecar;
    std::unordered_map<std::string, char> cache_state; // 0 missing, 1 building, 2 ready
    std::mutex cache_mtx;
    std::condition_variable cache_cv;
    std::vector<mwq_transient_slice> mwq_transients;
    ggml_tensor * pending_up_op = nullptr;
    ggml_tensor * pending_up_src1 = nullptr;
    ggml_tensor * pending_up_ids = nullptr;
    int pending_up_layer = -1;
    ggml_tensor * pending_gate_op = nullptr;
    ggml_tensor * pending_gate_src1 = nullptr;
    ggml_tensor * pending_gate_ids = nullptr;
    int pending_gate_layer = -1;
    ggml_tensor * fused_up_op = nullptr;
    ggml_tensor * fused_gate_op = nullptr;
    int fused_layer = -1;
    ggml_tensor * last_fused_up_op = nullptr;
    ggml_tensor * last_fused_gate_op = nullptr;

    // Residency state (resident[], lru, resident_bytes, buf allocation).
    std::mutex              mtx;
    std::condition_variable cv_done;   // signalled when an in-flight slice completes
    std::condition_variable cv_fuse;   // signalled when a deferred gate/up pair is matched
    std::list<std::pair<moe_managed *, int>> lru;  // front = MRU
    size_t resident_bytes = 0;
    size_t expert_total   = 0;   // sum of all registered exps tensor bytes
    size_t align          = 4096;

    // Single-flight coordination for llama_moe_buffer_swiglu_direct_compute's
    // residency-ensure phase: that function is invoked as a ggml op_override
    // callback (no barrier separates the nth worker threads, unlike the
    // weight-stream+barrier path which only fires for GGML_OP_MUL_MAT_ID), so
    // without this every thread would redundantly redo the ensure-residency
    // loop. Keyed by the GLU op tensor pointer, which is stable across the nth
    // per-thread invocations of one "wave" but distinct across graph nodes.
    struct moe_direct_ensure_entry {
        bool ok        = false;   // residency-ensure result, valid once done == true
        bool done      = false;
        int  remaining = 0;       // threads of this wave still to consume the result
    };
    std::condition_variable cv_direct_ensure;
    std::unordered_map<const ggml_tensor *, moe_direct_ensure_entry> direct_ensure_map; // guarded by mtx

    struct moe_qx_cache_entry {
        bool initialized = false;
        bool done        = false;
        int  remaining   = 0;
        int  build_remaining = 0;
        std::vector<mwq_qx_int8_cache_entry> entries;
    };
    std::condition_variable cv_qx_cache;
    std::unordered_map<const ggml_tensor *, moe_qx_cache_entry> qx_cache_map; // guarded by mtx

    struct moe_thread_barrier_entry {
        int arrived = 0;
        int leaving = 0;
        bool done = false;
    };
    std::condition_variable cv_ffn_barrier;
    std::unordered_map<const ggml_tensor *, moe_thread_barrier_entry> ffn_barrier_map; // guarded by mtx

    struct moe_ffn_pipe_slot {
        std::atomic<int> state{0};          // 0=empty, 1=filling, 2=ready
        std::atomic<int> block{-1};
        std::atomic<int> consumers_left{0};
    };
    struct moe_ffn_pipe_entry {
        int batch = 0;
        int block_size = 0;
        int n_blocks = 0;
        int ring = 0;
        int consumer_threads = 0;
        std::atomic<int> remaining{0};
        std::atomic<int> done_count{0};
        std::atomic<int> can_leave{0};
        std::array<moe_ffn_pipe_slot, 4> slots;
        std::vector<float> data;
    };
    std::unordered_map<const ggml_tensor *, std::unordered_map<size_t, std::shared_ptr<moe_ffn_pipe_entry>>> ffn_pipe_map; // guarded by mtx

    // Async prefetch worker pool + its own queue lock (kept separate from mtx so
    // enqueueing never blocks on an in-progress stream). Multiple workers issue
    // pread() concurrently (each with its own thread_local bounce), which lifts
    // effective read bandwidth on NVMe where a single O_DIRECT random-read stream
    // under-utilises the device.
    std::vector<std::thread> workers;
    std::mutex              qmtx;
    std::condition_variable cv_q;
    std::priority_queue<moe_prefetch_task, std::vector<moe_prefetch_task>, moe_prefetch_task_less> queue;
    bool                    stop = false;
    uint64_t                next_task_seq = 0;

    // EAM predictor state. Guarded by mtx. The transition table learns
    // P(next-layer expert | previous-layer expert) from real routed touches
    // within the current sequence; generated predictions still go through the
    // normal admission gate and never count as real activations.
    std::unordered_map<uint64_t, std::vector<uint32_t>> eam_layer_transition;
    std::vector<int>       eam_last_layer_experts;
    int                    eam_last_actual_layer = -1;
    uint64_t               eam_last_actual_token_epoch = 0;

    // Request-level EAM collection (EAMC). The current request/window is kept as
    // a sparse (layer, expert) activation vector. Completed vectors are stored in
    // a bounded in-process collection and the closest historical vector provides
    // a prior for prediction, cache pinning, and eviction.
    std::unordered_map<uint64_t, uint32_t> eamc_current_counts;
    uint64_t               eamc_current_total = 0;
    double                 eamc_current_norm = 0.0;
    uint64_t               eamc_current_start_token = 0;
    uint64_t               eamc_last_snapshot_token = 0;
    uint64_t               eamc_last_match_token = 0;
    uint64_t               eamc_next_snapshot_id = 1;
    std::vector<moe_eamc_snapshot> eamc_snapshots;
    std::vector<std::pair<size_t, double>> eamc_active_matches;

    std::atomic<uint64_t> streams{0}, hits{0}, evictions{0}, bytes_read{0};
    std::atomic<uint64_t> enqueued{0}, worker_streams{0};
    std::atomic<uint64_t> queue_dups{0}, hebf_reordered{0};
    std::atomic<uint64_t> dyn_bit2{0}, dyn_bit3{0}, dyn_bit4{0}, dyn_bit_other{0};
    std::atomic<uint64_t> dyn_effective_bytes{0}, dyn_saved_bytes{0};
    std::atomic<uint64_t> sidecar_hits{0}, sidecar_misses{0}, sidecar_bytes{0};
    std::atomic<uint64_t> cache_hits{0}, cache_misses{0}, cache_writes{0}, cache_bytes{0};
    std::atomic<uint64_t> mwq_kernel_ops{0}, mwq_kernel_slices{0}, mwq_bytes_read{0};
    std::atomic<uint64_t> mwq_group_batches{0}, mwq_group_pairs{0};
    std::atomic<uint64_t> mwq_op_tokens{0}, mwq_op_multi_token{0};
    std::atomic<uint64_t> mwq_compat_hits{0};
    std::atomic<uint64_t> mwq_sum_cache_hits{0}, mwq_sum_cache_builds{0};
    std::atomic<uint64_t> mwq_countsort_ops{0};
    std::atomic<uint64_t> mwq_avx512_q2_swiglu_calls{0}, mwq_avx512_q2_swiglu_pairs{0}, mwq_avx512_q2_swiglu_rows{0};
    std::atomic<uint64_t> mwq_avx512_q2_dot_calls{0}, mwq_avx512_q2_dot_pairs{0}, mwq_avx512_q2_dot_rows{0};
    std::atomic<uint64_t> mwq_vnni_q2_swiglu_calls{0}, mwq_vnni_q2_swiglu_pairs{0}, mwq_vnni_q2_swiglu_rows{0};
    std::atomic<uint64_t> mwq_vnni_q2_dot_calls{0}, mwq_vnni_q2_dot_pairs{0}, mwq_vnni_q2_dot_rows{0};
    std::atomic<uint64_t> mwq_vnni_qx_hits{0}, mwq_vnni_qx_builds{0};
    std::atomic<uint64_t> fused_gate_up_hits{0}, fused_gate_up_deferred{0}, fused_gate_up_misses{0};
    std::atomic<uint64_t> fused_swiglu_hits{0}, fused_swiglu_misses{0}, fused_swiglu_rows{0};
    std::atomic<uint64_t> fused_direct_swiglu_hits{0}, fused_direct_swiglu_misses{0}, fused_direct_swiglu_rows{0};
    std::atomic<uint64_t> fused_direct_down_prefetch{0};
    std::atomic<uint64_t> fused_ffn_hits{0}, fused_ffn_misses{0}, fused_ffn_runs{0}, fused_ffn_pairs{0}, fused_ffn_tiles{0}, fused_ffn_fallbacks{0};
    std::atomic<uint64_t> group_touches{0}, group_evictions{0}, group_evicted_slices{0};
    std::atomic<uint64_t> pinned_hits{0}, active_hits{0}, belady_evictions{0}, lru_fallback_evictions{0};
    std::atomic<uint64_t> eam_access{0}, eam_rank0_access{0}, eam_future_hints{0}, eam_prefetch_queued{0};
    std::atomic<uint64_t> eam_prefetch_hits{0}, eam_prefetch_late{0}, eam_prefetch_unused{0};
    std::atomic<uint64_t> eam_cache_hits{0}, eam_cache_misses{0};
    std::atomic<uint64_t> eam_prefetch_admit{0}, eam_prefetch_drop{0};
    std::atomic<uint64_t> eam_prefetch_drop_budget{0}, eam_prefetch_drop_distance{0}, eam_prefetch_drop_score{0};
    std::atomic<uint64_t> eam_prefetch_drop_active{0}, eam_prefetch_drop_pinned{0}, eam_prefetch_drop_recent{0};
    std::atomic<uint64_t> eam_prefetch_admit_score_x1000{0};
    std::atomic<uint64_t> eam_prefetch_drop_score_x1000{0};
    std::atomic<uint64_t> eam_speculative_bytes_peak{0};
    std::atomic<uint64_t> eam_predict_runs{0}, eam_predict_candidates{0}, eam_predict_enqueued{0}, eam_predict_dropped{0};
    std::atomic<uint64_t> eam_predict_score_x1000{0};
    std::atomic<uint64_t> eamc_snapshot_count{0}, eamc_match_runs{0}, eamc_match_hits{0}, eamc_match_misses{0};
    std::atomic<uint64_t> eamc_match_score_x1000{0}, eamc_prior_hits{0};
    std::atomic<uint64_t> eam_evict_spec_unused{0}, eam_evict_low_score{0}, eam_evict_far_future{0}, eam_evict_lru_relaxed{0};
    std::atomic<uint64_t> eam_evict_protect_active{0}, eam_evict_protect_pinned{0}, eam_evict_protect_recent{0};
    std::atomic<uint64_t> eam_evict_protect_high{0}, eam_evict_protect_early{0};
    uint64_t              exec_epoch = 0;
    uint64_t              future_epoch = 0;
    uint64_t              pin_refresh_at = 0;
    std::atomic<bool>     dyn_real_warned{false};
    mutable std::atomic<bool> sidecar_bit_clamp_warned{false};

    // Fine-grained reasons llama_moe_buffer_swiglu_direct_compute() bails to the
    // non-direct fallback path (see moe_stream_mwq_slice()'s out-params and
    // moe_op_all_mwq_resident()'s reason out-param for how these are set).
    std::atomic<uint64_t> direct_swiglu_fail_precheck{0};        // shape/type/src precondition failed
    std::atomic<uint64_t> direct_swiglu_fail_byname{0};          // by_name lookup miss for gate/up tensor
    std::atomic<uint64_t> direct_swiglu_fail_entry_missing{0};   // no sidecar entry for (expert,bits)
    std::atomic<uint64_t> direct_swiglu_fail_read{0};            // entry existed but pread failed
    std::atomic<uint64_t> direct_swiglu_fail_range{0};           // expert index out of range
    std::atomic<uint64_t> direct_swiglu_fail_kernelpair{0};      // moe_make_kernel_pair construction failed
    std::atomic<uint64_t> direct_swiglu_fail_native_prepare{0};  // moe_dot_kernel_pair_prepare_native failed
    std::atomic<uint64_t> direct_swiglu_fail_other{0};           // resident check false, none of the above fired
    std::atomic<uint64_t> mwq_stream_fail_total{0};               // every moe_stream_mwq_slice read_full() failure

    // Cache/observability closure (pinned vs LRU split, stream wait time, actual
    // MWQ bytes saved vs full-precision read).
    std::atomic<uint64_t> pinned_resident_bytes{0};
    std::atomic<uint64_t> stream_wait_ns{0};
    std::atomic<uint64_t> stream_wait_count{0};
    std::atomic<uint64_t> mwq_actual_full_bytes{0};
    std::atomic<uint64_t> mwq_actual_saved_bytes{0};

    bool profile = false;
    uint64_t profile_token_epoch = 0;
    int profile_last_layer = -1;
    std::atomic<uint64_t> prof_token_total_us{0};
    std::atomic<uint64_t> prof_attention_us{0};
    std::atomic<uint64_t> prof_moe_total_us{0};
    std::atomic<uint64_t> prof_route_us{0};
    std::atomic<uint64_t> prof_cache_lookup_us{0};
    std::atomic<uint64_t> prof_victim_select_us{0};
    std::atomic<uint64_t> prof_sidecar_submit_us{0};
    std::atomic<uint64_t> prof_sidecar_wait_us{0};
    std::atomic<uint64_t> prof_sidecar_read_us{0};
    std::atomic<uint64_t> prof_sidecar_bytes{0};
    std::atomic<uint64_t> prof_sidecar_read_count{0};
    std::atomic<uint64_t> prof_q2_unpack_us{0};
    std::atomic<uint64_t> prof_gate_dot_us{0};
    std::atomic<uint64_t> prof_up_dot_us{0};
    std::atomic<uint64_t> prof_swiglu_us{0};
    std::atomic<uint64_t> prof_down_dot_us{0};
    std::atomic<uint64_t> prof_graph_dispatch_us{0};
    std::atomic<uint64_t> prof_worker_wait_us{0};
    std::atomic<uint64_t> prof_barrier_us{0};
    std::atomic<uint64_t> prof_cache_hit{0};
    std::atomic<uint64_t> prof_cache_miss{0};
    std::atomic<uint64_t> prof_prefetch_hit{0};
    std::atomic<uint64_t> prof_prefetch_late{0};
    std::atomic<uint64_t> prof_prefetch_unused{0};
    std::atomic<uint64_t> prof_evict_clean{0};
    std::atomic<uint64_t> prof_evict_active_window{0};
    std::atomic<uint64_t> prof_evict_reloaded_within_1_token{0};
    std::atomic<uint64_t> prof_evict_reloaded_within_4_tokens{0};

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
        {
            std::lock_guard<std::mutex> lock(mtx);
            moe_eamc_snapshot_current_locked(*this, true);
            moe_eam_trace_append_locked(*this);
        }
        if (params.debug_log) {
            moe_print_stats_impl(*this, "llama_moe_buffer[teardown]");
        }
        for (auto & kv : by_name) {
            free(kv.second.buf);
            for (auto & mkv : kv.second.mwq) {
                free(mkv.second.buf);
            }
        }
        for (int fd : fds) {
            if (fd >= 0) close(fd);
        }
        if (sidecar_fd >= 0) {
            close(sidecar_fd);
        }
        if (cache_fd >= 0) {
            close(cache_fd);
        }
    }
};

// ---- forward decls of internals ----
static void moe_stream_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank);
static void moe_stream_mwq_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank,
        bool * out_entry_missing = nullptr, bool * out_read_failed = nullptr);
static const moe_sidecar_entry * moe_mwq_entry(const llama_moe_buffer_context & ctx, const moe_managed & m, int e, int bits);
static moe_group_state & moe_group_get(llama_moe_buffer_context & ctx, int layer, int expert);

static bool read_full(int fd, void * dst, size_t size, uint64_t off) {
    uint8_t * p = (uint8_t *) dst;
    size_t left = size;
    while (left > 0) {
        ssize_t r = pread(fd, p, left, (off_t) off);
        if (r <= 0) {
            return false;
        }
        p += r;
        off += (uint64_t) r;
        left -= (size_t) r;
    }
    return true;
}

static bool write_full(int fd, const void * src, size_t size, uint64_t off) {
    const uint8_t * p = (const uint8_t *) src;
    size_t left = size;
    while (left > 0) {
        ssize_t r = pwrite(fd, p, left, (off_t) off);
        if (r <= 0) {
            return false;
        }
        p += r;
        off += (uint64_t) r;
        left -= (size_t) r;
    }
    return true;
}

static uint16_t rd_u16(const uint8_t * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static uint32_t rd_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

static uint64_t rd_u64(const uint8_t * p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | p[i];
    }
    return v;
}

static uint8_t rd_u8(const uint8_t * p) {
    return p[0];
}

static std::string sidecar_key(const std::string & name, int expert, int bits) {
    return name + "#" + std::to_string(expert) + "#" + std::to_string(bits);
}

static uint64_t moe_group_key(int layer, int expert) {
    return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert;
}

static void moe_cache_create(llama_moe_buffer_context & ctx) {
    if (ctx.cache_fd >= 0 || ctx.sidecar.empty()) {
        return;
    }
    char path[] = "/tmp/llama-moe-transcode-cache-XXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) {
        if (ctx.params.debug_log) {
            std::fprintf(stderr, "llama_moe_buffer: failed to create transcode cache\n");
        }
        return;
    }
    unlink(path);
    ctx.cache_fd = fd;
    if (ctx.params.debug_log) {
        std::fprintf(stderr, "llama_moe_buffer: transcode cache ready, virtual size %.1f MiB\n",
                ctx.cache_next / 1048576.0);
    }
}

static bool moe_cache_try_read_or_begin(
        llama_moe_buffer_context & ctx,
        const std::string &        key,
        const moe_sidecar_entry &  ent,
        uint8_t *                  dst,
        size_t &                   bytes_read) {
    if (ctx.cache_fd < 0 || ent.decoded_size == 0) {
        return false;
    }

    {
        std::unique_lock<std::mutex> lk(ctx.cache_mtx);
        for (;;) {
            char & state = ctx.cache_state[key];
            if (state == 2) {
                break;
            }
            if (state == 1) {
                ctx.cache_cv.wait(lk);
                continue;
            }
            state = 1;
            ctx.cache_misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
    }

    if (!read_full(ctx.cache_fd, dst, (size_t) ent.decoded_size, ent.cache_offset)) {
        return false;
    }
    bytes_read = (size_t) ent.decoded_size;
    ctx.cache_hits.fetch_add(1, std::memory_order_relaxed);
    ctx.cache_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
    return true;
}

static void moe_cache_finish(
        llama_moe_buffer_context & ctx,
        const std::string &        key,
        const moe_sidecar_entry &  ent,
        const uint8_t *            decoded,
        bool                       ok) {
    if (ctx.cache_fd < 0 || ent.decoded_size == 0) {
        return;
    }
    bool ready = false;
    if (ok) {
        ready = write_full(ctx.cache_fd, decoded, (size_t) ent.decoded_size, ent.cache_offset);
    }
    {
        std::lock_guard<std::mutex> lk(ctx.cache_mtx);
        ctx.cache_state[key] = ready ? 2 : 0;
    }
    if (ready) {
        ctx.cache_writes.fetch_add(1, std::memory_order_relaxed);
    }
    ctx.cache_cv.notify_all();
}

static ggml_type sidecar_type_for_bits(int bits) {
    switch (bits) {
        case 2: return GGML_TYPE_Q2_K;
        case 3: return GGML_TYPE_Q3_K;
        case 4: return GGML_TYPE_Q4_K;
        case 5: return GGML_TYPE_Q5_K;
        case 6: return GGML_TYPE_Q6_K;
        default: return GGML_TYPE_COUNT;
    }
}

static bool mwq_decode_to_f32(const uint8_t * src, size_t src_size, int bits, int block_size, float * dst, int64_t n) {
    if (bits < 1 || bits > 8 || block_size <= 0 || n < 0) {
        return false;
    }

    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    if (src_size != n_blocks * block_bytes) {
        return false;
    }

    const uint8_t mask = (uint8_t) ((1u << bits) - 1u);
    int64_t out = 0;
    for (size_t ib = 0; ib < n_blocks; ++ib) {
        const ggml_fp16_t min_h = (ggml_fp16_t) rd_u16(src + ib * block_bytes + 0);
        const ggml_fp16_t scl_h = (ggml_fp16_t) rd_u16(src + ib * block_bytes + 2);
        const float min = ggml_fp16_to_fp32(min_h);
        const float scl = ggml_fp16_to_fp32(scl_h);
        const uint8_t * q = src + ib * block_bytes + 4;
        const int n_this = (int) std::min<int64_t>(block_size, n - out);
        for (int i = 0; i < n_this; ++i) {
            const int bit = i * bits;
            uint32_t acc = q[bit >> 3];
            if ((bit & 7) + bits > 8) {
                acc |= (uint32_t) q[(bit >> 3) + 1] << 8;
            }
            const uint8_t v = (acc >> (bit & 7)) & mask;
            dst[out + i] = min + scl * (float) v;
        }
        out += n_this;
    }
    return true;
}

static size_t mwq_hier_weights_bytes(int64_t n, int bits, int block_size) {
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    return n_blocks * qbytes;
}

static size_t mwq_hier_mins_bytes(int64_t n, int block_size) {
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    return n_blocks * sizeof(ggml_fp16_t);
}

static size_t mwq_hier_gscales_bytes(int64_t n, int block_size, int scale_group) {
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    const size_t n_groups = (n_blocks + (size_t) scale_group - 1) / (size_t) scale_group;
    return n_groups * sizeof(ggml_fp16_t);
}

static size_t mwq_hier_scodes_bytes(int64_t n, int block_size) {
    return (size_t) ((n + block_size - 1) / block_size);
}

static size_t mwq_hier_outlier_stride(int outlier_max) {
    return 1 + (size_t) std::max(0, outlier_max) * 3;
}

static size_t mwq_hier_outliers_bytes(int64_t n, int block_size, int outlier_max) {
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    return n_blocks * mwq_hier_outlier_stride(outlier_max);
}

static bool mwq_hier_layout(
        int64_t n,
        int bits,
        int block_size,
        int scale_group,
        int outlier_max,
        size_t src_size,
        size_t & off_weights,
        size_t & off_mins,
        size_t & off_gscales,
        size_t & off_scodes,
        size_t & off_outliers,
        size_t & total) {
    if (bits < 1 || bits > 8 || block_size <= 0 || scale_group <= 0 || outlier_max < 0 || n < 0) {
        return false;
    }
    off_weights = 0;
    off_mins = off_weights + mwq_hier_weights_bytes(n, bits, block_size);
    off_gscales = off_mins + mwq_hier_mins_bytes(n, block_size);
    off_scodes = off_gscales + mwq_hier_gscales_bytes(n, block_size, scale_group);
    off_outliers = off_scodes + mwq_hier_scodes_bytes(n, block_size);
    total = off_outliers + mwq_hier_outliers_bytes(n, block_size, outlier_max);
    return total == src_size;
}

static bool mwq_decode_hier_to_f32(
        const uint8_t * src,
        size_t src_size,
        int bits,
        int block_size,
        int scale_group,
        int outlier_max,
        float * dst,
        int64_t n) {
    size_t off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0, total = 0;
    if (!mwq_hier_layout(n, bits, block_size, scale_group, outlier_max, src_size,
                off_weights, off_mins, off_gscales, off_scodes, off_outliers, total)) {
        return false;
    }
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    const uint8_t mask = (uint8_t) ((1u << bits) - 1u);
    const uint8_t * weights = src + off_weights;
    const uint8_t * mins = src + off_mins;
    const uint8_t * gscales = src + off_gscales;
    const uint8_t * scodes = src + off_scodes;
    const uint8_t * outliers = src + off_outliers;
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);

    int64_t out = 0;
    for (size_t ib = 0; ib < n_blocks; ++ib) {
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mins + ib * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gscales + (ib / (size_t) scale_group) * 2));
        const float sc = gs * (float) scodes[ib] * (1.0f / 255.0f);
        const uint8_t * q = weights + ib * qbytes;
        const int n_this = (int) std::min<int64_t>(block_size, n - out);
        for (int i = 0; i < n_this; ++i) {
            const int bit = i * bits;
            uint32_t acc = q[bit >> 3];
            if ((bit & 7) + bits > 8) {
                acc |= (uint32_t) q[(bit >> 3) + 1] << 8;
            }
            const uint8_t v = (acc >> (bit & 7)) & mask;
            dst[out + i] = mn + sc * (float) v;
        }
        const uint8_t * ob = outliers + ib * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        for (int j = 0; j < count; ++j) {
            const int idx = ob[1 + j * 3];
            if (idx < n_this) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + j * 3 + 1));
                dst[out + idx] += residual;
            }
        }
        out += n_this;
    }
    return true;
}

static bool moe_sidecar_index(llama_moe_buffer_context & ctx) {
    if (ctx.params.sidecar_path.empty()) {
        return false;
    }

    int fd = open(ctx.params.sidecar_path.c_str(), O_RDONLY);
    if (fd < 0) {
        if (ctx.params.debug_log) {
            std::fprintf(stderr, "llama_moe_buffer: failed to open sidecar %s\n", ctx.params.sidecar_path.c_str());
        }
        return false;
    }

    uint8_t hdr[32];
    if (!read_full(fd, hdr, 16, 0)) {
        close(fd);
        return false;
    }
    const bool is_v1 = std::memcmp(hdr, "LMOEBC1\0", 8) == 0;
    const bool is_v2 = std::memcmp(hdr, "LMOEBV2\0", 8) == 0;
    if (!is_v1 && !is_v2) {
        if (ctx.params.debug_log) {
            std::fprintf(stderr, "llama_moe_buffer: invalid sidecar header %s\n", ctx.params.sidecar_path.c_str());
        }
        close(fd);
        return false;
    }

    const uint32_t n_entries = rd_u32(hdr + 8);
    uint64_t file_size = 0;
#if defined(__unix__) || defined(__APPLE__)
    struct stat st;
    if (fstat(fd, &st) == 0 && st.st_size > 0) {
        file_size = (uint64_t) st.st_size;
    }
#endif
    uint64_t off = 16;
    if (is_v2) {
        if (!read_full(fd, hdr + 16, 16, 16)) {
            close(fd);
            return false;
        }
        off = rd_u64(hdr + 16);
    }
    uint64_t truncated_entries = 0;
    for (uint32_t i = 0; i < n_entries; ++i) {
        moe_sidecar_entry ent;
        uint16_t name_len = 0;
        if (is_v1) {
            uint8_t eh[34];
            if (!read_full(fd, eh, sizeof(eh), off)) {
                break;
            }
            off += sizeof(eh);

            name_len = rd_u16(eh + 0);
            ent.expert       = (int) rd_u32(eh + 2);
            const uint16_t bits_field = rd_u16(eh + 6);
            ent.bits         = (int) (bits_field & 0x00ffu);
            const int block_log2 = (int) ((bits_field >> 8) & 0x000fu);
            ent.block_size   = block_log2 > 0 ? (1 << block_log2) : MOE_MWQ_BLOCK_DEFAULT;
            ent.codec        = rd_u16(eh + 8);
            ent.decoded_size = rd_u64(eh + 10);
            ent.encoded_size = rd_u64(eh + 18);
            ent.offset       = rd_u64(eh + 26);
        } else {
            uint8_t eh[58];
            if (!read_full(fd, eh, sizeof(eh), off)) {
                break;
            }
            off += sizeof(eh);

            name_len = rd_u16(eh + 0);
            ent.expert       = (int) rd_u32(eh + 2);
            ent.bits         = (int) rd_u8(eh + 6);
            const int block_log2 = (int) rd_u8(eh + 7);
            ent.block_size   = block_log2 > 0 ? (1 << block_log2) : MOE_MWQ_BLOCK_DEFAULT;
            ent.codec        = rd_u16(eh + 8);
            ent.scale_group  = (int) rd_u16(eh + 10);
            ent.outlier_max  = (int) rd_u16(eh + 12);
            ent.decoded_size = rd_u64(eh + 18);
            ent.encoded_size = rd_u64(eh + 26);
            ent.offset       = rd_u64(eh + 34);
        }
        ent.cache_offset = ctx.cache_next;
        ctx.cache_next += ent.decoded_size;

        std::string name(name_len, '\0');
        if (!read_full(fd, name.data(), name_len, off)) {
            break;
        }
        off += name_len;

        if (file_size > 0 && (ent.offset > file_size || ent.encoded_size > file_size - ent.offset)) {
            ++truncated_entries;
            continue;
        }

        if (ent.bits > 0 && ent.bits < 32) {
            ctx.sidecar_bits_mask |= 1u << ent.bits;
        }
        ctx.sidecar.emplace(sidecar_key(name, ent.expert, ent.bits), ent);
    }

    ctx.sidecar_fd = fd;
    if (ctx.params.debug_log) {
        std::fprintf(stderr, "llama_moe_buffer: indexed %zu sidecar expert slices from %s bits_mask=0x%x",
                ctx.sidecar.size(), ctx.params.sidecar_path.c_str(), ctx.sidecar_bits_mask);
        if (truncated_entries > 0) {
            std::fprintf(stderr, " (ignored %llu truncated entries; file_size=%llu)\n",
                    (unsigned long long) truncated_entries,
                    (unsigned long long) file_size);
        } else {
            std::fprintf(stderr, "\n");
        }
    }
    return !ctx.sidecar.empty();
}

static size_t moe_effective_stream_bytes_for_target(
        const llama_moe_buffer_context & ctx,
        const moe_managed              & m,
        int                              e,
        int                              target_bits) {
    const size_t full = m.stride;
    const int base_bits = std::max(1, ctx.params.base_bits);
    const int bits = std::max(1, target_bits);

    if (ctx.params.dynamic_bits_real) {
        auto it = ctx.sidecar.find(sidecar_key(m.name, e, bits));
        if (it != ctx.sidecar.end() && it->second.decoded_size == full) {
            return (size_t) it->second.encoded_size;
        }
        return full;
    }

    if (ctx.params.dynamic_bits) {
        return (full * (uint64_t) std::min(bits, base_bits) + base_bits - 1) /
            (uint64_t) base_bits;
    }

    return full;
}

static void moe_worker_loop(llama_moe_buffer_context * ctx) {
    for (;;) {
        moe_prefetch_task task;
        {
            std::unique_lock<std::mutex> lk(ctx->qmtx);
            ctx->cv_q.wait(lk, [ctx] { return ctx->stop || !ctx->queue.empty(); });
            if (ctx->stop && ctx->queue.empty()) {
                return;
            }
            task = ctx->queue.top();
            ctx->queue.pop();
            if (task.m != nullptr && task.e >= 0 && task.e < (int) task.m->queued.size()) {
                if (task.e < (int) task.m->queued_bits.size() && task.m->queued_bits[task.e] > task.target_bits) {
                    ctx->queue_dups.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                task.m->queued[task.e] = false;
                if (task.e < (int) task.m->queued_bits.size()) {
                    task.m->queued_bits[task.e] = 0;
                }
            }
        }
        if (ctx->params.dynamic_bits && task.m != nullptr) {
            const uint64_t full = task.m->stride;
            const uint64_t eff = moe_effective_stream_bytes_for_target(
                    *ctx, *task.m, task.e, task.target_bits);
            ctx->dyn_effective_bytes.fetch_add(eff, std::memory_order_relaxed);
            ctx->dyn_saved_bytes.fetch_add(full > eff ? full - eff : 0, std::memory_order_relaxed);
        }
        if (task.m != nullptr) {
            if (ctx->params.dynamic_bits_real && moe_mwq_entry(*ctx, *task.m, task.e, task.target_bits) != nullptr) {
                moe_stream_mwq_slice(*ctx, *task.m, task.e, false, task.target_bits, task.rank);
            } else {
                moe_stream_slice(*ctx, *task.m, task.e, false, task.target_bits, task.rank);
            }
        }
        ctx->worker_streams.fetch_add(1, std::memory_order_relaxed);
    }
}

std::shared_ptr<llama_moe_buffer_context> llama_moe_buffer_create(const llama_moe_buffer_params & params) {
    auto ctx = std::make_shared<llama_moe_buffer_context>();
    ctx->params = params;
    ctx->profile = std::getenv("LLAMA_MOE_PROFILE") != nullptr &&
            std::atoi(std::getenv("LLAMA_MOE_PROFILE")) > 0;
#if defined(_SC_PAGESIZE)
    g_page = (size_t) sysconf(_SC_PAGESIZE);
#endif
    ctx->align = g_page;
    moe_sidecar_index(*ctx);
    moe_cache_create(*ctx);
    moe_eamc_load_sidecar(*ctx);
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
    m.type        = exps->type;
    m.elems       = ggml_nelements(exps) / n_expert;
    m.resident.assign(n_expert, ST_COLD);
    m.resident_mwq.assign(n_expert, false);
    m.resident_bits.assign(n_expert, 0);
    m.resident_size.assign(n_expert, 0);
    m.resident_touched.assign(n_expert, false);
    m.prefetched.assign(n_expert, false);
    m.last_evicted_token.assign(n_expert, 0);
    m.queued.assign(n_expert, false);
    m.queued_bits.assign(n_expert, 0);
    m.activation.assign(n_expert, 0);
    m.lru_pos.resize(n_expert);
    std::sscanf(m.name.c_str(), "blk.%d.", &m.layer);

    auto res = ctx.by_name.emplace(m.name, std::move(m));
    if (res.second) {
        ctx.expert_total += res.first->second.nbytes;
        if (res.first->second.layer >= 0) {
            ctx.by_layer[res.first->second.layer].push_back(&res.first->second);
            for (int e = 0; e < n_expert; ++e) {
                moe_group_state & g = moe_group_get(ctx, res.first->second.layer, e);
                g.layer = res.first->second.layer;
                g.expert = e;
            }
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

static bool moe_ensure_mwq_buf(moe_managed & m, int bits, size_t stride, int block_size, int codec, int scale_group, int outlier_max) {
    moe_mwq_store & store = m.mwq[bits];
    if (store.buf != nullptr) {
        if (store.stride != stride ||
                store.block_size != block_size ||
                store.codec != codec ||
                store.scale_group != scale_group ||
                store.outlier_max != outlier_max) {
            return false;
        }
        if ((int) store.loaded.size() != m.n_expert) {
            store.loaded.assign(m.n_expert, false);
        }
        return true;
    }
    uint8_t * buf = nullptr;
    const size_t nbytes = stride * (size_t) m.n_expert;
    if (posix_memalign((void **) &buf, g_page, nbytes) != 0 || buf == nullptr) {
        return false;
    }
    store.buf = buf;
    store.stride = stride;
    store.block_size = block_size;
    store.codec = codec;
    store.scale_group = scale_group;
    store.outlier_max = outlier_max;
    store.loaded.assign(m.n_expert, false);
    return true;
}

static bool moe_mwq_loaded(const moe_managed & m, int e, int bits) {
    auto it = m.mwq.find(bits);
    return it != m.mwq.end() &&
        it->second.buf != nullptr &&
        e >= 0 &&
        e < (int) it->second.loaded.size() &&
        it->second.loaded[e];
}

static int moe_mwq_resolve_loaded_bits(const moe_managed & m, int e, int requested_bits) {
    int resolved = 0;
    if (e < 0 || e >= m.n_expert) {
        return 0;
    }
    for (const auto & kv : m.mwq) {
        const int bits = kv.first;
        const moe_mwq_store & store = kv.second;
        if (bits < requested_bits || store.buf == nullptr ||
                e >= (int) store.loaded.size() ||
                !store.loaded[e]) {
            continue;
        }
        resolved = std::max(resolved, bits);
    }
    return resolved;
}

static void moe_mark_mwq_transient(llama_moe_buffer_context & ctx, moe_managed & m, int e, int bits) {
    if (!moe_mwq_loaded(m, e, bits)) {
        return;
    }
    for (const mwq_transient_slice & cur : ctx.mwq_transients) {
        if (cur.m == &m && cur.expert == e && cur.bits == bits) {
            return;
        }
    }
    ctx.mwq_transients.push_back({&m, e, bits});
}

static void moe_cleanup_mwq_transients(llama_moe_buffer_context & ctx) {
    std::lock_guard<std::mutex> lk(ctx.mtx);
    for (const mwq_transient_slice & cur : ctx.mwq_transients) {
        moe_managed * m = cur.m;
        if (m == nullptr || cur.expert < 0 || cur.expert >= m->n_expert) {
            continue;
        }
        if (m->resident[cur.expert] == ST_INFLIGHT) {
            continue;
        }
        if (m->resident[cur.expert] == ST_RESIDENT &&
                m->resident_mwq[cur.expert] &&
                m->resident_bits[cur.expert] == cur.bits) {
            continue;
        }

        auto it = m->mwq.find(cur.bits);
        if (it == m->mwq.end() || it->second.buf == nullptr ||
                cur.expert >= (int) it->second.loaded.size() ||
                !it->second.loaded[cur.expert]) {
            continue;
        }

        uint8_t * ptr = it->second.buf + (size_t) cur.expert * it->second.stride;
        const uintptr_t base = (uintptr_t) ptr;
        const uintptr_t pb   = (base + g_page - 1) & ~(uintptr_t) (g_page - 1);
        const uintptr_t pe   = (base + it->second.stride) & ~(uintptr_t) (g_page - 1);
#if defined(MADV_DONTNEED)
        if (pe > pb) {
            madvise((void *) pb, pe - pb, MADV_DONTNEED);
        }
#endif
        it->second.loaded[cur.expert] = false;
    }
    ctx.mwq_transients.clear();
}

static const moe_sidecar_entry * moe_mwq_entry(const llama_moe_buffer_context & ctx, const moe_managed & m, int e, int bits) {
    auto it = ctx.sidecar.find(sidecar_key(m.name, e, bits));
    if (it == ctx.sidecar.end() ||
            (it->second.codec != MOE_SIDECAR_CODEC_MWQ && it->second.codec != MOE_SIDECAR_CODEC_MWQ_HIER) ||
            it->second.decoded_size != m.stride) {
        return nullptr;
    }
    return &it->second;
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

static moe_tensor_kind moe_kind(const moe_managed & m) {
    if (m.name.find(".ffn_gate_exps.weight") != std::string::npos) {
        return moe_tensor_kind::gate;
    }
    if (m.name.find(".ffn_up_exps.weight") != std::string::npos) {
        return moe_tensor_kind::up;
    }
    if (m.name.find(".ffn_down_exps.weight") != std::string::npos) {
        return moe_tensor_kind::down;
    }
    return moe_tensor_kind::other;
}

static int moe_clamp_bits(const llama_moe_buffer_context & ctx, int bits) {
    const int hi = std::max(1, ctx.params.base_bits);
    return std::max(1, std::min(bits, hi));
}

static int moe_clamp_sidecar_bits(const llama_moe_buffer_context & ctx, int bits) {
    bits = moe_clamp_bits(ctx, bits);
    if (!ctx.params.dynamic_bits_real || ctx.sidecar_bits_mask == 0 || ctx.sidecar.empty()) {
        return bits;
    }
    const uint32_t requested = bits < 32 ? (1u << bits) : 0u;
    if ((ctx.sidecar_bits_mask & requested) != 0) {
        return bits;
    }

    int clamped = 0;
    for (int b = bits; b >= 1; --b) {
        if ((ctx.sidecar_bits_mask & (1u << b)) != 0) {
            clamped = b;
            break;
        }
    }
    if (clamped == 0) {
        for (int b = 1; b < 32; ++b) {
            if ((ctx.sidecar_bits_mask & (1u << b)) != 0) {
                clamped = b;
                break;
            }
        }
    }

    if (ctx.params.strict_sidecar || clamped == 0) {
        std::fprintf(stderr,
                "llama_moe_buffer: requested sidecar bits=%d unavailable (available_mask=0x%x)\n",
                bits, ctx.sidecar_bits_mask);
        std::abort();
    }
    if (!ctx.sidecar_bit_clamp_warned.exchange(true, std::memory_order_relaxed) && ctx.params.debug_log) {
        std::fprintf(stderr,
                "llama_moe_buffer: requested sidecar bits=%d unavailable, clamped_to=%d available_mask=0x%x\n",
                bits, clamped, ctx.sidecar_bits_mask);
    }
    return clamped;
}

static int moe_target_bits_for_rank(const llama_moe_buffer_context & ctx, const moe_managed & m, int e, int rank) {
    if (ctx.params.fixed_bits > 0) {
        return moe_clamp_sidecar_bits(ctx, ctx.params.fixed_bits);
    }
    if (!ctx.params.dynamic_bits) {
        return moe_clamp_sidecar_bits(ctx, ctx.params.base_bits);
    }

    const moe_tensor_kind kind = moe_kind(m);
    if (ctx.params.gate_bits > 0 && kind == moe_tensor_kind::gate) {
        return moe_clamp_sidecar_bits(ctx, ctx.params.gate_bits);
    }
    if (ctx.params.up_bits > 0 && kind == moe_tensor_kind::up) {
        return moe_clamp_sidecar_bits(ctx, ctx.params.up_bits);
    }
    if (ctx.params.down_bits > 0 && kind == moe_tensor_kind::down) {
        return moe_clamp_sidecar_bits(ctx, ctx.params.down_bits);
    }

    int bits = ctx.params.cold_bits;
    if (moe_is_hot(ctx, m, e) || rank <= 0) {
        bits = ctx.params.hot_bits;
    } else if (rank <= 2) {
        bits = ctx.params.warm_bits;
    }

    switch (kind) {
        case moe_tensor_kind::gate:
            bits = std::max(bits, ctx.params.gate_min_bits);
            break;
        case moe_tensor_kind::up:
            bits = std::max(bits, ctx.params.up_min_bits);
            break;
        case moe_tensor_kind::down:
            bits = std::max(bits, ctx.params.down_min_bits);
            break;
        case moe_tensor_kind::other:
            break;
    }

    return moe_clamp_sidecar_bits(ctx, bits);
}

static int moe_prefetch_priority(const llama_moe_buffer_context & ctx, const moe_managed & m, int e, int rank, float score, int target_bits) {
    if (!ctx.params.hebf_schedule) {
        return -rank;
    }

    int priority = 10000;
    priority += std::max(0, 512 - rank * 64);
    priority += std::max(0, target_bits) * 32;
    switch (moe_kind(m)) {
        case moe_tensor_kind::down:
            priority += 256;
            break;
        case moe_tensor_kind::gate:
            priority += 128;
            break;
        case moe_tensor_kind::up:
        case moe_tensor_kind::other:
            break;
    }
    if (moe_is_hot(ctx, m, e)) {
        priority += 1024;
    }

    // The score is only a hint. Clamp its influence so an outlier gate logit does
    // not starve rank-0/near-term tasks.
    if (score > 0.0f) {
        priority += std::min(384, (int) (score * 16.0f));
    }
    return priority;
}

static int moe_target_bits_for_id_index(const llama_moe_buffer_context & ctx, const moe_managed & m, int expert, int64_t id_index) {
    if (!ctx.params.dynamic_bits_real) {
        return ctx.params.base_bits;
    }
    const int rank = ctx.params.sync_top_k > 0 ? (int) (id_index % ctx.params.sync_top_k) : 0;
    return moe_target_bits_for_rank(ctx, m, expert, rank);
}

static bool moe_native_entry_available(const llama_moe_buffer_context & ctx, const moe_managed & m, int bits) {
    if (!ctx.params.native_hot || !ctx.params.dynamic_bits_real || bits < ctx.params.base_bits) {
        return false;
    }
    const ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(m.type);
    if (traits == nullptr || traits->vec_dot == nullptr || traits->vec_dot_type == GGML_TYPE_COUNT) {
        return false;
    }
    const ggml_type_traits_cpu * xtraits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    return xtraits != nullptr && xtraits->from_float != nullptr;
}

static bool moe_native_resident_compatible(const llama_moe_buffer_context & ctx, const moe_managed & m, int expert, int bits) {
    if (!ctx.params.dynamic_bits_real ||
            expert < 0 || expert >= m.n_expert ||
            m.resident[expert] != ST_RESIDENT ||
            m.resident_mwq[expert] ||
            m.resident_bits[expert] < bits) {
        return false;
    }
    const ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(m.type);
    if (traits == nullptr || traits->vec_dot == nullptr || traits->vec_dot_type == GGML_TYPE_COUNT) {
        return false;
    }
    const ggml_type_traits_cpu * xtraits = ggml_get_type_traits_cpu(traits->vec_dot_type);
    return xtraits != nullptr && xtraits->from_float != nullptr;
}

static bool moe_op_all_mwq_available(const llama_moe_buffer_context & ctx, const moe_managed & m, const ggml_tensor * ids,
        moe_resident_fail_reason * reason = nullptr) {
    if (!ctx.params.dynamic_bits_real || ids == nullptr || ids->data == nullptr || ids->type != GGML_TYPE_I32) {
        if (reason) *reason = moe_resident_fail_reason::UNAVAILABLE;
        return false;
    }
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
            const int32_t expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
            const int64_t flat = token * ids->ne[0] + rank;
            const int bits = moe_target_bits_for_id_index(ctx, m, expert, flat);
            if (moe_mwq_entry(ctx, m, expert, bits) == nullptr && !moe_native_entry_available(ctx, m, bits)) {
                if (reason) *reason = moe_resident_fail_reason::UNAVAILABLE;
                return false;
            }
        }
    }
    return true;
}

static moe_group_state & moe_group_get(llama_moe_buffer_context & ctx, int layer, int expert) {
    const uint64_t key = moe_group_key(layer, expert);
    auto it = ctx.groups.find(key);
    if (it != ctx.groups.end()) {
        return it->second;
    }
    moe_group_state g;
    g.layer = layer;
    g.expert = expert;
    return ctx.groups.emplace(key, g).first->second;
}

static size_t moe_slice_resident_bytes(const moe_managed & m, int e) {
    if (e < 0 || e >= m.n_expert || m.resident[e] != ST_RESIDENT) {
        return 0;
    }
    return m.resident_size[e];
}

static size_t moe_group_resident_bytes(const llama_moe_buffer_context & ctx, int layer, int expert) {
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return 0;
    }
    size_t total = 0;
    for (const moe_managed * m : it->second) {
        if (m != nullptr && expert >= 0 && expert < m->n_expert) {
            total += moe_slice_resident_bytes(*m, expert);
        }
    }
    return total;
}

static double moe_group_resident_bit_score(const llama_moe_buffer_context & ctx, int layer, int expert) {
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return 0.0;
    }

    double total = 0.0;
    double weight = 0.0;
    for (const moe_managed * m : it->second) {
        if (m == nullptr || expert < 0 || expert >= m->n_expert ||
                m->resident[expert] != ST_RESIDENT) {
            continue;
        }
        const size_t bytes = moe_slice_resident_bytes(*m, expert);
        const int bits = std::max(1, m->resident_bits[expert]);
        total += (double) bytes * (double) bits;
        weight += (double) bytes;
    }
    return weight > 0.0 ? total / weight : 0.0;
}

static bool moe_group_has_inflight_or_queued(const llama_moe_buffer_context & ctx, int layer, int expert) {
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return false;
    }
    for (const moe_managed * m : it->second) {
        if (m == nullptr || expert < 0 || expert >= m->n_expert) {
            continue;
        }
        if (m->resident[expert] == ST_INFLIGHT || m->queued[expert]) {
            return true;
        }
    }
    return false;
}

static bool moe_group_is_active(const llama_moe_buffer_context & ctx, const moe_group_state & g) {
    if (g.next_use_epoch == UINT64_MAX || ctx.params.active_window <= 0) {
        return false;
    }
    if (g.next_use_epoch <= ctx.exec_epoch) {
        return true;
    }
    return g.next_use_epoch - ctx.exec_epoch <= (uint64_t) ctx.params.active_window;
}

static bool moe_group_is_recently_used(const llama_moe_buffer_context & ctx, const moe_group_state & g) {
    if (g.last_used_epoch == 0) {
        return false;
    }
    const uint64_t guard = (uint64_t) std::max(2, ctx.params.active_window);
    return ctx.exec_epoch <= g.last_used_epoch || ctx.exec_epoch - g.last_used_epoch <= guard;
}

static bool moe_group_in_cooldown(const llama_moe_buffer_context & ctx, const moe_group_state & g) {
    if (ctx.params.group_cooldown_tokens <= 0 || g.last_used_token_epoch == 0) {
        return false;
    }
    return ctx.profile_token_epoch <= g.last_used_token_epoch ||
            ctx.profile_token_epoch - g.last_used_token_epoch <= (uint64_t) ctx.params.group_cooldown_tokens;
}

static bool moe_group_used_within_tokens(const llama_moe_buffer_context & ctx, const moe_group_state & g, int n_tokens) {
    if (n_tokens <= 0 || g.last_used_token_epoch == 0) {
        return false;
    }
    return ctx.profile_token_epoch <= g.last_used_token_epoch ||
            ctx.profile_token_epoch - g.last_used_token_epoch <= (uint64_t) n_tokens;
}

static int moe_kind_index(const moe_managed & m) {
    switch (moe_kind(m)) {
        case moe_tensor_kind::gate: return 1;
        case moe_tensor_kind::up:   return 2;
        case moe_tensor_kind::down: return 3;
        case moe_tensor_kind::other:
        default: return 0;
    }
}

static double moe_env_f64(const char * name, double def) {
    const char * v = std::getenv(name);
    return v == nullptr || v[0] == '\0' ? def : std::atof(v);
}

static int moe_env_i32(const char * name, int def) {
    const char * v = std::getenv(name);
    return v == nullptr || v[0] == '\0' ? def : std::atoi(v);
}

static bool moe_env_flag(const char * name, int def) {
    return moe_env_i32(name, def) > 0;
}

static void moe_atomic_max_u64(std::atomic<uint64_t> & dst, uint64_t value) {
    uint64_t old = dst.load(std::memory_order_relaxed);
    while (old < value && !dst.compare_exchange_weak(old, value, std::memory_order_relaxed)) {}
}

static bool moe_eamc_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_EAMC", 0);
}

static int moe_eamc_max_snapshots() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_EAMC_MAX", 16));
}

static int moe_eamc_snapshot_tokens() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_EAMC_SNAPSHOT_TOKENS", 64));
}

static int moe_eamc_min_access() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_EAMC_MIN_ACCESS", 256));
}

static int moe_eamc_match_every_tokens() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_EAMC_MATCH_EVERY", 4));
}

static int moe_eamc_match_topk() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_EAMC_MATCH_TOPK", 2));
}

static double moe_eamc_min_similarity() {
    return moe_env_f64("LLAMA_LAZY_MOE_EAMC_MIN_SIM", 0.10);
}

static int moe_max_layer_index(const llama_moe_buffer_context & ctx) {
    int max_layer = 0;
    for (const auto & kv : ctx.by_layer) {
        max_layer = std::max(max_layer, kv.first);
    }
    return max_layer;
}

static double moe_eamc_compute_norm(const std::unordered_map<uint64_t, uint32_t> & counts) {
    double sum = 0.0;
    for (const auto & kv : counts) {
        sum += (double) kv.second * (double) kv.second;
    }
    return std::sqrt(sum);
}

static void moe_eamc_clear_current_locked(llama_moe_buffer_context & ctx) {
    ctx.eamc_current_counts.clear();
    ctx.eamc_current_total = 0;
    ctx.eamc_current_norm = 0.0;
    ctx.eamc_current_start_token = ctx.profile_token_epoch;
    ctx.eamc_last_snapshot_token = ctx.profile_token_epoch;
    ctx.eamc_last_match_token = 0;
    ctx.eamc_active_matches.clear();
}

static void moe_eamc_note_access_locked(
        llama_moe_buffer_context & ctx,
        int                        layer,
        int                        expert,
        uint64_t                   count) {
    if (!moe_eamc_enabled() || layer < 0 || expert < 0 || count == 0) {
        return;
    }
    if (ctx.eamc_current_start_token == 0) {
        ctx.eamc_current_start_token = ctx.profile_token_epoch;
        ctx.eamc_last_snapshot_token = ctx.profile_token_epoch;
    }
    const uint64_t key = moe_group_key(layer, expert);
    uint32_t & v = ctx.eamc_current_counts[key];
    const uint64_t nv = std::min<uint64_t>(UINT32_MAX, (uint64_t) v + count);
    v = (uint32_t) nv;
    ctx.eamc_current_total += count;
}

static double moe_eamc_similarity_locked(
        const llama_moe_buffer_context & ctx,
        const moe_eamc_snapshot &        snap) {
    if (ctx.eamc_current_counts.empty() || snap.counts.empty()) {
        return 0.0;
    }
    const double current_norm = ctx.eamc_current_norm > 0.0 ?
        ctx.eamc_current_norm : moe_eamc_compute_norm(ctx.eamc_current_counts);
    if (current_norm <= 0.0 || snap.norm <= 0.0) {
        return 0.0;
    }
    const auto * small = &ctx.eamc_current_counts;
    const auto * large = &snap.counts;
    if (small->size() > large->size()) {
        small = &snap.counts;
        large = &ctx.eamc_current_counts;
    }
    double dot = 0.0;
    for (const auto & kv : *small) {
        auto it = large->find(kv.first);
        if (it != large->end()) {
            dot += (double) kv.second * (double) it->second;
        }
    }
    return dot / (current_norm * snap.norm);
}

static void moe_eamc_match_locked(llama_moe_buffer_context & ctx, bool force) {
    if (!moe_eamc_enabled()) {
        return;
    }
    if (ctx.eamc_snapshots.empty() || ctx.eamc_current_total < (uint64_t) moe_eamc_min_access()) {
        if (force) {
            ctx.eamc_match_misses.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    if (!force && ctx.eamc_last_match_token != 0 &&
            ctx.profile_token_epoch <= ctx.eamc_last_match_token + (uint64_t) moe_eamc_match_every_tokens()) {
        return;
    }
    ctx.eamc_current_norm = moe_eamc_compute_norm(ctx.eamc_current_counts);
    struct match_candidate {
        size_t idx = 0;
        double sim = 0.0;
    };
    std::vector<match_candidate> matches;
    matches.reserve(ctx.eamc_snapshots.size());
    const double min_sim = moe_eamc_min_similarity();
    for (size_t i = 0; i < ctx.eamc_snapshots.size(); ++i) {
        const double sim = moe_eamc_similarity_locked(ctx, ctx.eamc_snapshots[i]);
        if (sim >= min_sim) {
            matches.push_back({i, sim});
        }
    }
    std::sort(matches.begin(), matches.end(), [](const match_candidate & a, const match_candidate & b) {
        if (a.sim != b.sim) {
            return a.sim > b.sim;
        }
        return a.idx > b.idx;
    });
    ctx.eamc_active_matches.clear();
    const int topk = std::min<int>(moe_eamc_match_topk(), (int) matches.size());
    double score_sum = 0.0;
    for (int i = 0; i < topk; ++i) {
        ctx.eamc_active_matches.push_back({matches[(size_t) i].idx, matches[(size_t) i].sim});
        score_sum += matches[(size_t) i].sim;
    }
    ctx.eamc_last_match_token = ctx.profile_token_epoch;
    ctx.eamc_match_runs.fetch_add(1, std::memory_order_relaxed);
    if (topk > 0) {
        const double avg_sim = score_sum / (double) topk;
        ctx.eamc_match_hits.fetch_add(1, std::memory_order_relaxed);
        ctx.eamc_match_score_x1000.fetch_add((uint64_t) std::max(0.0, avg_sim * 1000.0), std::memory_order_relaxed);
    } else {
        ctx.eamc_match_misses.fetch_add(1, std::memory_order_relaxed);
    }
}

static void moe_eamc_snapshot_current_locked(llama_moe_buffer_context & ctx, bool force) {
    if (!moe_eamc_enabled()) {
        return;
    }
    const uint64_t age = ctx.profile_token_epoch >= ctx.eamc_current_start_token ?
        ctx.profile_token_epoch - ctx.eamc_current_start_token : 0;
    if (!force && age < (uint64_t) moe_eamc_snapshot_tokens()) {
        return;
    }
    if (ctx.eamc_current_total < (uint64_t) moe_eamc_min_access() || ctx.eamc_current_counts.empty()) {
        if (force) {
            moe_eamc_clear_current_locked(ctx);
        }
        return;
    }
    moe_eamc_snapshot snap;
    snap.id = ctx.eamc_next_snapshot_id++;
    snap.start_token = ctx.eamc_current_start_token;
    snap.end_token = ctx.profile_token_epoch;
    snap.total = ctx.eamc_current_total;
    snap.counts = ctx.eamc_current_counts;
    snap.norm = moe_eamc_compute_norm(snap.counts);
    if (snap.norm > 0.0) {
        const int max_snapshots = moe_eamc_max_snapshots();
        while ((int) ctx.eamc_snapshots.size() >= max_snapshots) {
            ctx.eamc_snapshots.erase(ctx.eamc_snapshots.begin());
        }
        ctx.eamc_snapshots.push_back(std::move(snap));
        ctx.eamc_snapshot_count.fetch_add(1, std::memory_order_relaxed);
    }
    moe_eamc_clear_current_locked(ctx);
}

static void moe_eamc_tick_locked(llama_moe_buffer_context & ctx) {
    if (!moe_eamc_enabled()) {
        return;
    }
    moe_eamc_match_locked(ctx, false);
    moe_eamc_snapshot_current_locked(ctx, false);
}

static double moe_eamc_prior_locked(
        const llama_moe_buffer_context & ctx,
        int                              layer,
        int                              expert) {
    if (!moe_eamc_enabled() || ctx.eamc_active_matches.empty() || layer < 0 || expert < 0) {
        return 0.0;
    }
    const uint64_t key = moe_group_key(layer, expert);
    double weighted = 0.0;
    double weights = 0.0;
    for (const auto & match : ctx.eamc_active_matches) {
        if (match.first >= ctx.eamc_snapshots.size()) {
            continue;
        }
        const moe_eamc_snapshot & snap = ctx.eamc_snapshots[match.first];
        auto it = snap.counts.find(key);
        if (it == snap.counts.end() || snap.total == 0) {
            continue;
        }
        weighted += match.second * ((double) it->second / (double) snap.total);
        weights += match.second;
    }
    return weights > 0.0 ? weighted / weights : 0.0;
}

static double moe_eamc_prior(
        const llama_moe_buffer_context & ctx,
        int                              layer,
        int                              expert) {
    const double prior = moe_eamc_prior_locked(ctx, layer, expert);
    if (prior > 0.0) {
        const_cast<llama_moe_buffer_context &>(ctx).eamc_prior_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return prior;
}

static void moe_eamc_load_sidecar(llama_moe_buffer_context & ctx) {
    const char * path = std::getenv("LLAMA_LAZY_MOE_EAMC_SIDECAR");
    if (path == nullptr || path[0] == '\0') {
        return;
    }

    FILE * f = std::fopen(path, "rb");
    if (f == nullptr) {
        if (ctx.params.debug_log) {
            std::fprintf(stderr, "llama_moe_buffer: failed to open EAMC sidecar %s\n", path);
        }
        return;
    }

    char magic[8] = {};
    uint32_t version = 0;
    uint32_t n_layer = 0;
    uint32_t n_expert = 0;
    uint32_t n_snapshot = 0;
    bool ok = std::fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
        std::fread(&version, sizeof(version), 1, f) == 1 &&
        std::fread(&n_layer, sizeof(n_layer), 1, f) == 1 &&
        std::fread(&n_expert, sizeof(n_expert), 1, f) == 1 &&
        std::fread(&n_snapshot, sizeof(n_snapshot), 1, f) == 1;
    if (!ok || std::memcmp(magic, "EAMCV1\0", 7) != 0 || version != 1) {
        if (ctx.params.debug_log) {
            std::fprintf(stderr, "llama_moe_buffer: invalid EAMC sidecar %s\n", path);
        }
        std::fclose(f);
        return;
    }

    const int max_snapshots = moe_eamc_max_snapshots();
    uint32_t loaded = 0;
    for (uint32_t i = 0; i < n_snapshot; ++i) {
        uint64_t id = 0;
        uint64_t weight = 0;
        uint64_t total = 0;
        double norm = 0.0;
        uint32_t item_count = 0;
        ok = std::fread(&id, sizeof(id), 1, f) == 1 &&
            std::fread(&weight, sizeof(weight), 1, f) == 1 &&
            std::fread(&total, sizeof(total), 1, f) == 1 &&
            std::fread(&norm, sizeof(norm), 1, f) == 1 &&
            std::fread(&item_count, sizeof(item_count), 1, f) == 1;
        if (!ok) {
            break;
        }

        moe_eamc_snapshot snap;
        snap.id = id;
        snap.start_token = 0;
        snap.end_token = weight;
        snap.total = total;
        snap.norm = norm;
        snap.counts.reserve((size_t) item_count);
        for (uint32_t j = 0; j < item_count; ++j) {
            uint16_t layer = 0;
            uint16_t expert = 0;
            uint32_t count = 0;
            uint32_t rank0 = 0;
            ok = std::fread(&layer, sizeof(layer), 1, f) == 1 &&
                std::fread(&expert, sizeof(expert), 1, f) == 1 &&
                std::fread(&count, sizeof(count), 1, f) == 1 &&
                std::fread(&rank0, sizeof(rank0), 1, f) == 1;
            GGML_UNUSED(rank0);
            if (!ok) {
                break;
            }
            if (layer < n_layer && expert < n_expert && count > 0) {
                snap.counts[moe_group_key((int) layer, (int) expert)] = count;
            }
        }
        if (!ok) {
            break;
        }
        if (snap.total == 0) {
            uint64_t total_from_items = 0;
            for (const auto & kv : snap.counts) {
                total_from_items += kv.second;
            }
            snap.total = total_from_items;
        }
        if (snap.norm <= 0.0) {
            snap.norm = moe_eamc_compute_norm(snap.counts);
        }
        if (!snap.counts.empty() && snap.total > 0 && snap.norm > 0.0 &&
                (int) ctx.eamc_snapshots.size() < max_snapshots) {
            ctx.eamc_snapshots.push_back(std::move(snap));
            ++loaded;
        }
    }
    std::fclose(f);
    ctx.eamc_snapshot_count.fetch_add(loaded, std::memory_order_relaxed);
    if (ctx.params.debug_log) {
        std::fprintf(stderr,
                "llama_moe_buffer: loaded %u/%u EAMC snapshots from %s (layers=%u experts=%u)\n",
                loaded, n_snapshot, path, n_layer, n_expert);
    }
}

static double moe_group_cache_score(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        size_t                           resident_bytes) {
    const double total_access = (double) std::max<uint64_t>(1, ctx.eam_access.load(std::memory_order_relaxed));
    const double seq_activation_rate = (double) g.seq_access / total_access;
    const double rank0_rate = g.seq_access > 0 ? (double) g.seq_rank0_access / (double) g.seq_access : 0.0;

    const uint64_t hit = g.seq_cache_hits + g.seq_prefetch_hits;
    const uint64_t miss = g.seq_cache_misses + g.seq_prefetch_late + g.seq_prefetch_unused;
    const double hit_rate = (hit + miss) > 0 ? (double) hit / (double) (hit + miss) : 0.0;
    const uint64_t age = g.seq_last_token_epoch == 0 || ctx.profile_token_epoch < g.seq_last_token_epoch ?
        0 : ctx.profile_token_epoch - g.seq_last_token_epoch;
    const double recency = g.seq_last_token_epoch == 0 ? 0.0 : 1.0 / (1.0 + (double) age);
    const double recent_hit = hit_rate * recency;

    const int max_layer = std::max(1, moe_max_layer_index(ctx));
    const double early_layer_bonus = g.layer >= 0 ?
        1.0 - std::min(1.0, (double) g.layer / (double) max_layer) : 0.0;
    const double size_cost = (double) resident_bytes / 1048576.0;

    const double a = moe_env_f64("LLAMA_LAZY_MOE_EAM_CACHE_A", 10000.0);
    const double b = moe_env_f64("LLAMA_LAZY_MOE_EAM_CACHE_B", 40.0);
    const double c = moe_env_f64("LLAMA_LAZY_MOE_EAM_CACHE_C", 25.0);
    const double d = moe_env_f64("LLAMA_LAZY_MOE_EAM_CACHE_D", 20.0);
    const double e = moe_env_f64("LLAMA_LAZY_MOE_EAM_CACHE_E", 0.50);
    const double eamc = moe_env_f64("LLAMA_LAZY_MOE_EAMC_CACHE_W", 9000.0);

    return a * seq_activation_rate +
           b * rank0_rate +
           c * recent_hit +
           d * early_layer_bonus -
           e * size_cost +
           eamc * moe_eamc_prior(ctx, g.layer, g.expert);
}

static double moe_group_cache_score(const llama_moe_buffer_context & ctx, const moe_group_state & g) {
    return moe_group_cache_score(ctx, g, moe_group_resident_bytes(ctx, g.layer, g.expert));
}

static int moe_forward_layer_distance(const llama_moe_buffer_context & ctx, int target_layer) {
    if (target_layer < 0) {
        return INT_MAX / 2;
    }
    const int current = ctx.profile_last_layer >= 0 ? ctx.profile_last_layer : 0;
    const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
    int dist = target_layer - current;
    if (dist < 0) {
        dist += n_layer;
    }
    return std::max(0, dist);
}

static size_t moe_eam_prefetch_budget_bytes(const llama_moe_buffer_context & ctx) {
    const int env_mb = moe_env_i32("LLAMA_LAZY_MOE_EAM_PREFETCH_MB", -1);
    if (env_mb >= 0) {
        return (size_t) env_mb * 1048576ull;
    }
    if (ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 1536ull * 1048576ull) {
        return 128ull * 1048576ull;
    }
    if (ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 3072ull * 1048576ull) {
        return 192ull * 1048576ull;
    }
    return 256ull * 1048576ull;
}

static int moe_eam_prefetch_max_distance(const llama_moe_buffer_context & ctx) {
    const int env_dist = moe_env_i32("LLAMA_LAZY_MOE_EAM_PREFETCH_DISTANCE", -1);
    if (env_dist >= 0) {
        return env_dist;
    }
    return ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 1536ull * 1048576ull ? 2 : 4;
}

static size_t moe_speculative_bytes_locked(const llama_moe_buffer_context & ctx) {
    size_t total = 0;
    for (const auto & kv : ctx.by_name) {
        const moe_managed & m = kv.second;
        for (int e = 0; e < m.n_expert; ++e) {
            if (e < (int) m.queued.size() && m.queued[e]) {
                const int bits = e < (int) m.queued_bits.size() && m.queued_bits[e] > 0 ?
                    m.queued_bits[e] : ctx.params.base_bits;
                total += moe_effective_stream_bytes_for_target(ctx, m, e, bits);
            }
            if (e < (int) m.prefetched.size() && m.prefetched[e] &&
                    e < (int) m.resident_touched.size() && !m.resident_touched[e]) {
                total += moe_slice_resident_bytes(m, e);
            }
        }
    }
    return total;
}

static bool moe_group_has_speculative_unused(const llama_moe_buffer_context & ctx, int layer, int expert) {
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return false;
    }
    for (const moe_managed * m : it->second) {
        if (m == nullptr || expert < 0 || expert >= m->n_expert) {
            continue;
        }
        if (expert < (int) m->prefetched.size() && m->prefetched[expert] &&
                expert < (int) m->resident_touched.size() && !m->resident_touched[expert]) {
            return true;
        }
    }
    return false;
}

static double moe_group_seq_rate(const llama_moe_buffer_context & ctx, const moe_group_state & g) {
    const double total_access = (double) std::max<uint64_t>(1, ctx.eam_access.load(std::memory_order_relaxed));
    return (double) g.seq_access / total_access;
}

static bool moe_group_is_high_sequence(const llama_moe_buffer_context & ctx, const moe_group_state & g, double cache_score) {
    const double keep_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_SCORE", 125.0);
    const double keep_rate = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_RATE", 0.0080);
    return cache_score >= keep_score || moe_group_seq_rate(ctx, g) >= keep_rate;
}

static bool moe_group_is_early_high_reuse(const moe_group_state & g, double cache_score) {
    const int early_layers = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_LAYERS", 4);
    if (g.layer < 0 || g.layer > early_layers) {
        return false;
    }
    const double keep_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_KEEP_SCORE", 95.0);
    return cache_score >= keep_score || g.seq_rank0_access >= std::max<uint64_t>(4, g.seq_access / 2);
}

static bool moe_group_speculative_unused_evictable(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        uint64_t                         future_distance,
        int                              layer_distance) {
    const double low_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_SPEC_LOW_SCORE", 80.0);
    if (g.hot_score <= low_score) {
        return true;
    }

    const int layer_guard = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_SPEC_LAYER_GUARD", 6);
    if (layer_distance <= layer_guard) {
        return false;
    }

    const uint64_t reuse_guard = (uint64_t) moe_env_i32(
            "LLAMA_LAZY_MOE_EAM_EVICT_SPEC_REUSE_GUARD",
            std::max(2 * std::max(1, ctx.params.active_window), 24));
    if (future_distance != UINT64_MAX && future_distance <= reuse_guard) {
        return false;
    }

    const uint64_t hits = g.seq_prefetch_hits;
    const uint64_t unused = g.seq_prefetch_unused;
    const bool bad_history = unused >= std::max<uint64_t>(2, hits * 2 + 1);
    return future_distance == UINT64_MAX || bad_history;
}

static bool moe_eam_can_reclaim_without_protected_locked(
        llama_moe_buffer_context & ctx,
        size_t                     needed_bytes,
        double                     recent_score_threshold) {
    if (ctx.params.budget_bytes == 0 || ctx.resident_bytes + needed_bytes <= ctx.params.budget_bytes) {
        return true;
    }

    size_t reclaimable = 0;
    for (auto it = ctx.lru.rbegin(); it != ctx.lru.rend(); ++it) {
        moe_managed * m = it->first;
        const int e = it->second;
        if (m == nullptr || e < 0 || e >= m->n_expert || m->layer < 0) {
            continue;
        }
        moe_group_state & g = moe_group_get(ctx, m->layer, e);
        const bool recent_high =
            moe_group_is_recently_used(ctx, g) &&
            moe_group_cache_score(ctx, g) >= recent_score_threshold;
        if (g.pinned || moe_group_is_active(ctx, g) || recent_high ||
                moe_group_in_cooldown(ctx, g) || moe_group_has_inflight_or_queued(ctx, m->layer, e)) {
            continue;
        }
        reclaimable += moe_group_resident_bytes(ctx, m->layer, e);
        if (ctx.resident_bytes + needed_bytes <= ctx.params.budget_bytes + reclaimable) {
            return true;
        }
    }
    return false;
}

static double moe_eam_prefetch_score_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        int                              target_layer,
        int                              rank,
        float                            clg_score,
        size_t                           speculative_bytes) {
    const double total_access = (double) std::max<uint64_t>(1, ctx.eam_access.load(std::memory_order_relaxed));
    const double seq_activation_rate = (double) g.seq_access / total_access;
    const int dist = moe_forward_layer_distance(ctx, target_layer);
    const double proximity = 1.0 / (1.0 + (double) std::max(0, dist));
    const double miss_cost = g.seq_access > 0 ?
        (double) (g.seq_cache_misses + g.seq_prefetch_late) / (double) g.seq_access : 0.25;
    const double rank_conf = 1.0 / (1.0 + (double) std::max(0, rank));
    const double score_conf = clg_score > 0.0f ? std::min(1.0, (double) clg_score) : 0.0;
    const double confidence = 0.75 * rank_conf + 0.25 * score_conf;
    const double unused_rate = g.seq_prefetch_queued > 0 ?
        (double) g.seq_prefetch_unused / (double) g.seq_prefetch_queued : 0.0;
    const double pressure = (double) speculative_bytes /
        (double) std::max<size_t>(1, moe_eam_prefetch_budget_bytes(ctx));
    const double pollution_penalty = unused_rate + std::max(0.0, pressure - 0.75);

    const double a = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREFETCH_A", 6000.0);
    const double b = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREFETCH_B", 2.0);
    const double c = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREFETCH_C", 1.5);
    const double d = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREFETCH_D", 1.5);
    const double e = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREFETCH_E", 2.0);
    const double hist = moe_env_f64("LLAMA_LAZY_MOE_EAMC_PREFETCH_W", 5000.0);
    return a * seq_activation_rate +
           b * proximity +
           c * miss_cost +
           d * confidence -
           e * pollution_penalty +
           hist * moe_eamc_prior(ctx, target_layer, g.expert);
}

enum class moe_prefetch_admit_result : uint8_t {
    ADMIT = 0,
    BUDGET,
    DISTANCE,
    SCORE,
    ACTIVE,
    PINNED,
    RECENT,
};

static void moe_eam_note_prefetch_admission(
        llama_moe_buffer_context & ctx,
        moe_prefetch_admit_result  result,
        double                     score,
        size_t                     speculative_bytes) {
    const uint64_t score_x1000 = (uint64_t) std::max(0.0, score * 1000.0);
    moe_atomic_max_u64(ctx.eam_speculative_bytes_peak, (uint64_t) speculative_bytes);
    if (result == moe_prefetch_admit_result::ADMIT) {
        ctx.eam_prefetch_admit.fetch_add(1, std::memory_order_relaxed);
        ctx.eam_prefetch_admit_score_x1000.fetch_add(score_x1000, std::memory_order_relaxed);
        return;
    }
    ctx.eam_prefetch_drop.fetch_add(1, std::memory_order_relaxed);
    ctx.eam_prefetch_drop_score_x1000.fetch_add(score_x1000, std::memory_order_relaxed);
    switch (result) {
        case moe_prefetch_admit_result::BUDGET:
            ctx.eam_prefetch_drop_budget.fetch_add(1, std::memory_order_relaxed);
            break;
        case moe_prefetch_admit_result::DISTANCE:
            ctx.eam_prefetch_drop_distance.fetch_add(1, std::memory_order_relaxed);
            break;
        case moe_prefetch_admit_result::SCORE:
            ctx.eam_prefetch_drop_score.fetch_add(1, std::memory_order_relaxed);
            break;
        case moe_prefetch_admit_result::ACTIVE:
            ctx.eam_prefetch_drop_active.fetch_add(1, std::memory_order_relaxed);
            break;
        case moe_prefetch_admit_result::PINNED:
            ctx.eam_prefetch_drop_pinned.fetch_add(1, std::memory_order_relaxed);
            break;
        case moe_prefetch_admit_result::RECENT:
            ctx.eam_prefetch_drop_recent.fetch_add(1, std::memory_order_relaxed);
            break;
        case moe_prefetch_admit_result::ADMIT:
            break;
    }
}

static moe_prefetch_admit_result moe_eam_prefetch_blocked_reason_locked(
        llama_moe_buffer_context & ctx,
        double                     recent_score_threshold) {
    for (auto it = ctx.lru.rbegin(); it != ctx.lru.rend(); ++it) {
        moe_managed * m = it->first;
        const int e = it->second;
        if (m == nullptr || e < 0 || e >= m->n_expert || m->layer < 0) {
            continue;
        }
        moe_group_state & g = moe_group_get(ctx, m->layer, e);
        if (g.pinned) {
            return moe_prefetch_admit_result::PINNED;
        }
        if (moe_group_is_active(ctx, g)) {
            return moe_prefetch_admit_result::ACTIVE;
        }
        if (moe_group_is_recently_used(ctx, g) &&
                moe_group_cache_score(ctx, g) >= recent_score_threshold) {
            return moe_prefetch_admit_result::RECENT;
        }
    }
    return moe_prefetch_admit_result::RECENT;
}

static moe_prefetch_admit_result moe_eam_prefetch_admit_locked(
        llama_moe_buffer_context & ctx,
        moe_managed &              m,
        int                        e,
        int                        rank,
        float                      clg_score,
        int                        target_bits,
        double *                   out_score,
        size_t *                   out_speculative_bytes) {
    const size_t speculative_bytes = moe_speculative_bytes_locked(ctx);
    const size_t target_bytes = moe_effective_stream_bytes_for_target(ctx, m, e, target_bits);
    const size_t budget = moe_eam_prefetch_budget_bytes(ctx);
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    const double score = moe_eam_prefetch_score_locked(ctx, g, m.layer, rank, clg_score, speculative_bytes);
    if (out_score != nullptr) {
        *out_score = score;
    }
    if (out_speculative_bytes != nullptr) {
        *out_speculative_bytes = speculative_bytes + target_bytes;
    }

    const int dist = moe_forward_layer_distance(ctx, m.layer);
    if (dist > moe_eam_prefetch_max_distance(ctx)) {
        return moe_prefetch_admit_result::DISTANCE;
    }
    if (budget > 0 && speculative_bytes + target_bytes > budget) {
        return moe_prefetch_admit_result::BUDGET;
    }
    const double recent_score_threshold = moe_env_f64("LLAMA_LAZY_MOE_EAM_RECENT_SCORE", 80.0);
    if (!moe_eam_can_reclaim_without_protected_locked(ctx, target_bytes, recent_score_threshold)) {
        return moe_eam_prefetch_blocked_reason_locked(ctx, recent_score_threshold);
    }
    const double min_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREFETCH_MIN_SCORE", 1.0);
    if (score < min_score) {
        return moe_prefetch_admit_result::SCORE;
    }
    return moe_prefetch_admit_result::ADMIT;
}

static void moe_eam_note_token_locked(llama_moe_buffer_context & ctx, moe_group_state & g, uint64_t count, int rank) {
    g.seq_access += count;
    if (rank == 0) {
        g.seq_rank0_access += count;
        ctx.eam_rank0_access.fetch_add(count, std::memory_order_relaxed);
    }
    if (g.seq_first_token_epoch == 0) {
        g.seq_first_token_epoch = ctx.profile_token_epoch;
    }
    g.seq_last_token_epoch = ctx.profile_token_epoch;
    ctx.eam_access.fetch_add(count, std::memory_order_relaxed);
}

static void moe_group_touch_locked(llama_moe_buffer_context & ctx, moe_managed & m, int e, int rank, uint64_t count) {
    if (m.layer < 0 || e < 0 || e >= m.n_expert || count == 0) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    g.access += count;
    moe_eam_note_token_locked(ctx, g, count, rank);
    moe_eamc_note_access_locked(ctx, m.layer, e, count);
    g.seq_tensor_access[(size_t) moe_kind_index(m)] += count;
    if (rank == 0) {
        g.rank0_access += count;
    }
    g.last_used_epoch = ctx.exec_epoch;
    g.last_used_token_epoch = ctx.profile_token_epoch;
    if (g.next_use_epoch <= g.last_used_epoch) {
        g.next_use_epoch = UINT64_MAX;
    }
    g.hot_score = moe_group_cache_score(ctx, g);
    ctx.group_touches.fetch_add(count, std::memory_order_relaxed);
    if (g.pinned) {
        ctx.pinned_hits.fetch_add(count, std::memory_order_relaxed);
    } else if (moe_group_is_active(ctx, g)) {
        ctx.active_hits.fetch_add(count, std::memory_order_relaxed);
    }
    moe_eamc_tick_locked(ctx);
}

static void moe_group_note_cache_touch_locked(
        llama_moe_buffer_context & ctx,
        moe_managed &              m,
        int                        e,
        bool                       prefetch_hit,
        bool                       cache_miss) {
    if (m.layer < 0 || e < 0 || e >= m.n_expert) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    if (prefetch_hit) {
        g.seq_prefetch_hits++;
        ctx.eam_prefetch_hits.fetch_add(1, std::memory_order_relaxed);
    } else if (cache_miss) {
        g.seq_cache_misses++;
        ctx.eam_cache_misses.fetch_add(1, std::memory_order_relaxed);
    } else {
        g.seq_cache_hits++;
        ctx.eam_cache_hits.fetch_add(1, std::memory_order_relaxed);
    }
    g.hot_score = moe_group_cache_score(ctx, g);
}

static void moe_group_note_prefetch_late_locked(llama_moe_buffer_context & ctx, moe_managed & m, int e) {
    if (m.layer < 0 || e < 0 || e >= m.n_expert) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    g.seq_prefetch_late++;
    g.hot_score = moe_group_cache_score(ctx, g);
    ctx.eam_prefetch_late.fetch_add(1, std::memory_order_relaxed);
}

static void moe_group_note_prefetch_unused_locked(llama_moe_buffer_context & ctx, moe_managed & m, int e) {
    if (m.layer < 0 || e < 0 || e >= m.n_expert) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    g.seq_prefetch_unused++;
    g.hot_score = moe_group_cache_score(ctx, g);
    ctx.eam_prefetch_unused.fetch_add(1, std::memory_order_relaxed);
}

static void moe_group_note_prefetch_queued_locked(llama_moe_buffer_context & ctx, moe_managed & m, int e) {
    if (m.layer < 0 || e < 0 || e >= m.n_expert) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    g.seq_prefetch_queued++;
    g.hot_score = moe_group_cache_score(ctx, g);
    ctx.eam_prefetch_queued.fetch_add(1, std::memory_order_relaxed);
}

static void moe_group_future_hint_locked(llama_moe_buffer_context & ctx, int layer, int expert, int rank, float score) {
    if (layer < 0 || expert < 0) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, layer, expert);
    ctx.future_epoch = std::max(ctx.future_epoch + 1, ctx.exec_epoch + 1);
    const uint64_t epoch = ctx.future_epoch;
    g.next_use_epoch = std::min(g.next_use_epoch, epoch);
    const double rank_bonus = rank == 0 ? 4.0 : 1.0 / (double) std::max(1, rank + 1);
    const double score_bonus = score > 0.0f ? std::min(8.0, (double) score) : 0.0;
    g.seq_future_hints++;
    if (rank == 0) {
        g.seq_future_rank0_hints++;
    }
    if (score > 0.0f) {
        g.seq_future_score_sum += (double) score;
    }
    g.hot_score = std::max(g.hot_score, moe_group_cache_score(ctx, g) + rank_bonus + score_bonus);
    ctx.eam_future_hints.fetch_add(1, std::memory_order_relaxed);
}

static void moe_refresh_pins_locked(llama_moe_buffer_context & ctx) {
    if (ctx.params.budget_bytes == 0 || ctx.params.pinned_fraction <= 0.0f) {
        for (auto & kv : ctx.groups) {
            kv.second.pinned = false;
        }
        ctx.pinned_resident_bytes.store(0, std::memory_order_relaxed);
        return;
    }
    if (ctx.pin_refresh_at != 0 && ctx.group_touches.load(std::memory_order_relaxed) < ctx.pin_refresh_at) {
        return;
    }
    ctx.pin_refresh_at = ctx.group_touches.load(std::memory_order_relaxed) +
            (uint64_t) std::max(1, ctx.params.pin_refresh_interval);

    struct pin_candidate {
        uint64_t key;
        int layer;
        size_t bytes;
        double score;
    };
    std::vector<pin_candidate> candidates;
    candidates.reserve(ctx.groups.size());
    for (auto & kv : ctx.groups) {
        moe_group_state & g = kv.second;
        const size_t bytes = moe_group_resident_bytes(ctx, g.layer, g.expert);
        g.hot_score = moe_group_cache_score(ctx, g, bytes);
        if (bytes == 0 || g.hot_score <= 0.0 || (g.seq_access == 0 && g.seq_future_hints == 0)) {
            g.pinned = false;
            continue;
        }
        candidates.push_back({kv.first, g.layer, bytes, g.hot_score});
        g.pinned = false;
    }
    std::sort(candidates.begin(), candidates.end(), [](const pin_candidate & a, const pin_candidate & b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.bytes > b.bytes;
    });

    const size_t pin_budget = (size_t) ((double) ctx.params.budget_bytes * (double) ctx.params.pinned_fraction);
    const size_t layer_budget = (size_t) ((double) pin_budget * (double) std::max(0.0f, ctx.params.pinned_layer_fraction));
    std::unordered_map<int, size_t> by_layer_bytes;
    size_t used = 0;
    for (const pin_candidate & c : candidates) {
        if (used + c.bytes > pin_budget) {
            continue;
        }
        if (layer_budget > 0 && by_layer_bytes[c.layer] + c.bytes > layer_budget) {
            continue;
        }
        auto it = ctx.groups.find(c.key);
        if (it == ctx.groups.end()) {
            continue;
        }
        it->second.pinned = true;
        it->second.pin_epoch = ctx.exec_epoch;
        used += c.bytes;
        by_layer_bytes[c.layer] += c.bytes;
    }
    ctx.pinned_resident_bytes.store(used, std::memory_order_relaxed);
}

static void moe_release_slice_locked(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool erase_lru) {
    if (e < 0 || e >= m.n_expert || m.resident[e] != ST_RESIDENT) {
        return;
    }
    if (erase_lru) {
        ctx.lru.erase(m.lru_pos[e]);
    }
    uint8_t * resident_ptr = nullptr;
    if (m.resident_mwq[e]) {
        auto mit = m.mwq.find(m.resident_bits[e]);
        if (mit != m.mwq.end() && mit->second.buf != nullptr) {
            resident_ptr = mit->second.buf + (size_t) e * mit->second.stride;
            if (e < (int) mit->second.loaded.size()) {
                mit->second.loaded[e] = false;
            }
        }
    } else {
        resident_ptr = m.buf != nullptr ? m.buf + (size_t) e * m.stride : nullptr;
    }
    const size_t resident_size = m.resident_size[e];
    if (e < (int) m.prefetched.size() && m.prefetched[e] &&
            e < (int) m.resident_touched.size() && !m.resident_touched[e]) {
        moe_group_note_prefetch_unused_locked(ctx, m, e);
        if (ctx.profile) {
            ctx.prof_prefetch_unused.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (ctx.profile) {
        if (e < (int) m.last_evicted_token.size()) {
            m.last_evicted_token[e] = ctx.profile_token_epoch;
        }
        ctx.prof_evict_clean.fetch_add(1, std::memory_order_relaxed);
    }
    const uintptr_t base = (uintptr_t) resident_ptr;
    const uintptr_t pb   = (base + g_page - 1) & ~(uintptr_t) (g_page - 1);
    const uintptr_t pe   = (base + resident_size) & ~(uintptr_t) (g_page - 1);
#if defined(MADV_DONTNEED)
    if (resident_ptr != nullptr && pe > pb) {
        madvise((void *) pb, pe - pb, MADV_DONTNEED);
    }
#endif
    m.resident[e] = ST_COLD;
    m.resident_mwq[e] = false;
    m.resident_bits[e] = 0;
    m.resident_size[e] = 0;
    if (e < (int) m.resident_touched.size()) {
        m.resident_touched[e] = false;
    }
    if (e < (int) m.prefetched.size()) {
        m.prefetched[e] = false;
    }
    ctx.resident_bytes -= std::min(ctx.resident_bytes, resident_size);
    ctx.evictions.fetch_add(1, std::memory_order_relaxed);
    ctx.group_evicted_slices.fetch_add(1, std::memory_order_relaxed);
}

static size_t moe_release_group_locked(llama_moe_buffer_context & ctx, int layer, int expert) {
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return 0;
    }
    size_t released = 0;
    for (moe_managed * m : it->second) {
        if (m == nullptr || expert < 0 || expert >= m->n_expert || m->resident[expert] != ST_RESIDENT) {
            continue;
        }
        released += m->resident_size[expert];
        moe_release_slice_locked(ctx, *m, expert, true);
    }
    if (released > 0) {
        ctx.group_evictions.fetch_add(1, std::memory_order_relaxed);
    }
    return released;
}

static bool moe_enqueue_prefetch(
        llama_moe_buffer_context & ctx,
        moe_managed &              m,
        int                        e,
        int                        rank,
        float                      score) {
    if (e < 0 || e >= m.n_expert) {
        return false;
    }

    const int target_bits = moe_target_bits_for_rank(ctx, m, e, rank);
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        moe_group_future_hint_locked(ctx, m.layer, e, rank, score);
    }

    // Cheap pre-filter: a stale read only creates a no-op hit in the worker.
    if ((m.resident[e] == ST_RESIDENT && (!ctx.params.dynamic_bits_real || m.resident_bits[e] >= target_bits)) ||
            (ctx.params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(m, e, target_bits) > 0) ||
            m.resident[e] == ST_INFLIGHT) {
        ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (m.queued[e]) {
        const int pending_bits = e < (int) m.queued_bits.size() ? m.queued_bits[e] : 0;
        if (pending_bits >= target_bits) {
            ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (e < (int) m.queued_bits.size()) {
            m.queued_bits[e] = target_bits;
        }
    }

    if (ctx.params.dynamic_bits_real &&
            moe_mwq_entry(ctx, m, e, target_bits) == nullptr &&
            !moe_native_entry_available(ctx, m, target_bits)) {
        ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        double prefetch_score = 0.0;
        size_t speculative_bytes = 0;
        const moe_prefetch_admit_result admit = moe_eam_prefetch_admit_locked(
                ctx, m, e, rank, score, target_bits, &prefetch_score, &speculative_bytes);
        moe_eam_note_prefetch_admission(ctx, admit, prefetch_score, speculative_bytes);
        if (admit != moe_prefetch_admit_result::ADMIT) {
            return false;
        }
    }

    if (ctx.params.dynamic_bits && ctx.params.fixed_bits <= 0) {
        switch (target_bits) {
            case 2: ctx.dyn_bit2.fetch_add(1, std::memory_order_relaxed); break;
            case 3: ctx.dyn_bit3.fetch_add(1, std::memory_order_relaxed); break;
            case 4: ctx.dyn_bit4.fetch_add(1, std::memory_order_relaxed); break;
            default: ctx.dyn_bit_other.fetch_add(1, std::memory_order_relaxed); break;
        }
    }

    moe_prefetch_task task;
    task.m = &m;
    task.e = e;
    task.rank = rank;
    task.score = score;
    task.target_bits = target_bits;
    task.priority = moe_prefetch_priority(ctx, m, e, rank, score, target_bits);
    task.seq = ctx.next_task_seq++;

    if (ctx.params.hebf_schedule && task.seq > 0 && rank == 0) {
        ctx.hebf_reordered.fetch_add(1, std::memory_order_relaxed);
    }

    m.queued[e] = true;
    if (e < (int) m.queued_bits.size()) {
        m.queued_bits[e] = std::max(m.queued_bits[e], target_bits);
    }
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        moe_group_note_prefetch_queued_locked(ctx, m, e);
    }
    ctx.queue.push(task);
    ctx.enqueued.fetch_add(1, std::memory_order_relaxed);
    return true;
}

static bool moe_eam_predict_enabled() {
    return moe_env_i32("LLAMA_LAZY_MOE_EAM_PREDICT", 0) > 0;
}

static int moe_eam_predict_depth(const llama_moe_buffer_context & ctx) {
    const int env_depth = moe_env_i32("LLAMA_LAZY_MOE_EAM_PREDICT_DEPTH", -1);
    if (env_depth >= 0) {
        return env_depth;
    }
    if (ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 1536ull * 1048576ull) {
        return 2;
    }
    if (ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 3072ull * 1048576ull) {
        return 4;
    }
    return 6;
}

static int moe_eam_predict_max_tasks_per_layer() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_EAM_MAX_TASKS_PER_LAYER", 1));
}

static int moe_eam_predict_max_tasks_per_token() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_EAM_MAX_TASKS_PER_TOKEN", 8));
}

static double moe_eam_transition_score_locked(
        const llama_moe_buffer_context & ctx,
        int                              src_layer,
        const std::vector<int> &         src_experts,
        int                              target_expert) {
    double acc = 0.0;
    int n = 0;
    for (int src_expert : src_experts) {
        if (src_expert < 0) {
            continue;
        }
        auto it = ctx.eam_layer_transition.find(moe_group_key(src_layer, src_expert));
        if (it == ctx.eam_layer_transition.end() || target_expert < 0 ||
                target_expert >= (int) it->second.size()) {
            continue;
        }
        uint64_t total = 0;
        for (uint32_t v : it->second) {
            total += v;
        }
        if (total == 0) {
            continue;
        }
        acc += (double) it->second[(size_t) target_expert] / (double) total;
        ++n;
    }
    return n > 0 ? acc / (double) n : 0.0;
}

static double moe_eam_predict_score_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        int                              src_layer,
        const std::vector<int> &         src_experts,
        int                              target_layer,
        int                              target_expert) {
    const double total_access = (double) std::max<uint64_t>(1, ctx.eam_access.load(std::memory_order_relaxed));
    const double seq_activation_rate = (double) g.seq_access / total_access;
    const double rank0_rate = g.seq_access > 0 ? (double) g.seq_rank0_access / (double) g.seq_access : 0.0;
    const uint64_t age = g.seq_last_token_epoch == 0 || ctx.profile_token_epoch < g.seq_last_token_epoch ?
        UINT64_MAX : ctx.profile_token_epoch - g.seq_last_token_epoch;
    const double recency = age == UINT64_MAX ? 0.0 : 1.0 / (1.0 + (double) age);
    const double transition = moe_eam_transition_score_locked(ctx, src_layer, src_experts, target_expert);
    const int max_layer = std::max(1, moe_max_layer_index(ctx));
    const double early_urgency = target_layer >= 0 ?
        1.0 - std::min(1.0, (double) target_layer / (double) max_layer) : 0.0;
    const double miss_cost = g.seq_access > 0 ?
        (double) (g.seq_cache_misses + g.seq_prefetch_late) / (double) g.seq_access : 0.15;
    const double unused_rate = g.seq_prefetch_queued > 0 ?
        (double) g.seq_prefetch_unused / (double) g.seq_prefetch_queued : 0.0;
    const double pressure = (double) moe_speculative_bytes_locked(ctx) /
        (double) std::max<size_t>(1, moe_eam_prefetch_budget_bytes(ctx));

    const double a = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_A", 9000.0);
    const double b = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_B", 35.0);
    const double c = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_C", 20.0);
    const double d = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_D", 45.0);
    const double e = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_E", 12.0);
    const double f = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_F", 8.0);
    const double gpen = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_G", 18.0);
    const double h = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_H", 5.0);
    const double hist = moe_env_f64("LLAMA_LAZY_MOE_EAMC_PREDICT_W", 12000.0);

    return a * seq_activation_rate +
           b * rank0_rate +
           c * recency +
           d * transition +
           e * early_urgency +
           f * miss_cost -
           gpen * unused_rate -
           h * std::max(0.0, pressure - 0.75) +
           hist * moe_eamc_prior(ctx, target_layer, target_expert);
}

static bool moe_prefetch_needed_unlocked(
        const llama_moe_buffer_context & ctx,
        const moe_managed &              m,
        int                              e,
        int                              target_bits) {
    if (e < 0 || e >= m.n_expert) {
        return false;
    }
    if (e < (int) m.queued.size() && m.queued[e]) {
        const int pending_bits = e < (int) m.queued_bits.size() ? m.queued_bits[e] : 0;
        if (pending_bits >= target_bits) {
            return false;
        }
    }
    if (m.resident[e] == ST_INFLIGHT) {
        return false;
    }
    if (m.resident[e] == ST_RESIDENT &&
            (!ctx.params.dynamic_bits_real || m.resident_bits[e] >= target_bits)) {
        return false;
    }
    if (ctx.params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(m, e, target_bits) > 0) {
        return false;
    }
    return true;
}

static void moe_eam_update_transition_locked(
        llama_moe_buffer_context & ctx,
        int                        layer,
        const std::vector<int> &   actual_experts,
        int                        n_expert) {
    if (layer < 0 || actual_experts.empty()) {
        return;
    }
    const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
    const bool adjacent =
        ctx.eam_last_actual_layer >= 0 &&
        ctx.eam_last_actual_layer != layer &&
        ((ctx.eam_last_actual_layer + 1) % n_layer) == layer;
    if (adjacent && !ctx.eam_last_layer_experts.empty()) {
        for (int prev_expert : ctx.eam_last_layer_experts) {
            if (prev_expert < 0) {
                continue;
            }
            std::vector<uint32_t> & row =
                ctx.eam_layer_transition[moe_group_key(ctx.eam_last_actual_layer, prev_expert)];
            if ((int) row.size() < n_expert) {
                row.resize((size_t) n_expert, 0);
            }
            for (int expert : actual_experts) {
                if (expert >= 0 && expert < n_expert && row[(size_t) expert] != UINT32_MAX) {
                    row[(size_t) expert]++;
                }
            }
        }
    }
    ctx.eam_last_actual_layer = layer;
    ctx.eam_last_actual_token_epoch = ctx.profile_token_epoch;
    ctx.eam_last_layer_experts = actual_experts;
}

static void moe_eam_predict_after_route(
        llama_moe_buffer_context & ctx,
        int                        layer,
        const std::vector<int> &   actual_experts,
        int                        n_expert) {
    if (!moe_eam_predict_enabled() || layer < 0 || actual_experts.empty()) {
        return;
    }

    struct predict_candidate {
        int layer = -1;
        int expert = -1;
        int rank = 0;
        double score = 0.0;
    };

    std::vector<predict_candidate> candidates;
    const int depth = moe_eam_predict_depth(ctx);
    const int per_layer = moe_eam_predict_max_tasks_per_layer();
    const int per_token = moe_eam_predict_max_tasks_per_token();
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        moe_eam_update_transition_locked(ctx, layer, actual_experts, n_expert);
        const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
        for (int d = 1; d <= depth; ++d) {
            const int target_layer = (layer + d) % n_layer;
            auto lit = ctx.by_layer.find(target_layer);
            if (lit == ctx.by_layer.end() || lit->second.empty() || lit->second[0] == nullptr) {
                continue;
            }
            const int target_n_expert = lit->second[0]->n_expert;
            std::vector<predict_candidate> layer_candidates;
            layer_candidates.reserve((size_t) target_n_expert);
            for (int expert = 0; expert < target_n_expert; ++expert) {
                bool needs_prefetch = false;
                for (const moe_managed * m : lit->second) {
                    if (m == nullptr) {
                        continue;
                    }
                    const int target_bits = moe_target_bits_for_rank(ctx, *m, expert, 0);
                    if (moe_prefetch_needed_unlocked(ctx, *m, expert, target_bits)) {
                        needs_prefetch = true;
                        break;
                    }
                }
                if (!needs_prefetch) {
                    continue;
                }
                moe_group_state & g = moe_group_get(ctx, target_layer, expert);
                const double score = moe_eam_predict_score_locked(
                        ctx, g, layer, actual_experts, target_layer, expert);
                if (score < moe_env_f64("LLAMA_LAZY_MOE_EAM_PREDICT_MIN_SCORE", 75.0)) {
                    continue;
                }
                layer_candidates.push_back({target_layer, expert, 0, score});
            }
            std::sort(layer_candidates.begin(), layer_candidates.end(),
                    [](const predict_candidate & a, const predict_candidate & b) {
                if (a.score != b.score) {
                    return a.score > b.score;
                }
                return a.expert < b.expert;
            });
            const int n_take = std::min<int>(per_layer, (int) layer_candidates.size());
            for (int i = 0; i < n_take && (int) candidates.size() < per_token; ++i) {
                layer_candidates[(size_t) i].rank = i;
                candidates.push_back(layer_candidates[(size_t) i]);
                moe_group_state & g = moe_group_get(ctx, layer_candidates[(size_t) i].layer,
                        layer_candidates[(size_t) i].expert);
                g.seq_predicted++;
                g.seq_predict_score_sum += layer_candidates[(size_t) i].score;
            }
        }
    }

    if (candidates.empty()) {
        return;
    }

    ctx.eam_predict_runs.fetch_add(1, std::memory_order_relaxed);
    ctx.eam_predict_candidates.fetch_add((uint64_t) candidates.size(), std::memory_order_relaxed);

    int enqueued = 0;
    {
        std::lock_guard<std::mutex> lk(ctx.qmtx);
        for (const predict_candidate & c : candidates) {
            auto lit = ctx.by_layer.find(c.layer);
            if (lit == ctx.by_layer.end()) {
                continue;
            }
            bool any = false;
            for (moe_managed * m : lit->second) {
                if (m == nullptr) {
                    continue;
                }
                const int target_bits = moe_target_bits_for_rank(ctx, *m, c.expert, c.rank);
                if (!moe_prefetch_needed_unlocked(ctx, *m, c.expert, target_bits)) {
                    continue;
                }
                any = moe_enqueue_prefetch(ctx, *m, c.expert, c.rank, (float) c.score) || any;
            }
            if (any) {
                ++enqueued;
                std::lock_guard<std::mutex> mlk(ctx.mtx);
                moe_group_state & g = moe_group_get(ctx, c.layer, c.expert);
                g.seq_pred_enqueued++;
            }
        }
    }

    if (enqueued > 1) {
        ctx.cv_q.notify_all();
    } else if (enqueued == 1) {
        ctx.cv_q.notify_one();
    }
    ctx.eam_predict_enqueued.fetch_add((uint64_t) enqueued, std::memory_order_relaxed);
    ctx.eam_predict_dropped.fetch_add((uint64_t) (candidates.size() - (size_t) enqueued), std::memory_order_relaxed);
    uint64_t score_sum = 0;
    for (const predict_candidate & c : candidates) {
        score_sum += (uint64_t) std::max(0.0, c.score * 1000.0);
    }
    ctx.eam_predict_score_x1000.fetch_add(score_sum, std::memory_order_relaxed);
}

// Evict one ExpertGroup. Caller must hold ctx.mtx. Victim selection is a
// Belady-style approximation: skip pinned/currently protected groups, then
// prefer groups with no near-future use, lower heat, and older LRU position.
static bool moe_evict_lru(llama_moe_buffer_context & ctx) {
    if (ctx.lru.empty()) {
        return false;
    }
    const uint64_t prof_t0 = ctx.profile ? moe_profile_now_ns() : 0;

    moe_refresh_pins_locked(ctx);

    struct victim_candidate {
        int layer = -1;
        int expert = -1;
        double score = -1.0e300;
        bool belady = false;
        enum reason_t {
            SPEC_UNUSED,
            LOW_SCORE,
            FAR_FUTURE,
            RELAXED,
        } reason = LOW_SCORE;
    };

    victim_candidate best;
    int lru_depth = 0;
    const int recent_tokens = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_RECENT_TOKENS", 1);
    for (auto it = ctx.lru.rbegin(); it != ctx.lru.rend(); ++it, ++lru_depth) {
        moe_managed * m = it->first;
        const int e = it->second;
        if (m == nullptr || e < 0 || e >= m->n_expert || m->layer < 0) {
            continue;
        }

        const uint64_t key = moe_group_key(m->layer, e);
        auto git = ctx.groups.find(key);
        if (git == ctx.groups.end()) {
            moe_group_get(ctx, m->layer, e);
            git = ctx.groups.find(key);
        }
        moe_group_state & g = git->second;
        if (moe_group_has_inflight_or_queued(ctx, m->layer, e)) {
            continue;
        }

        const size_t bytes = moe_group_resident_bytes(ctx, m->layer, e);
        if (bytes == 0) {
            continue;
        }

        g.hot_score = moe_group_cache_score(ctx, g, bytes);
        const bool active = moe_group_is_active(ctx, g);
        const bool current_layer = ctx.profile_last_layer >= 0 && m->layer == ctx.profile_last_layer;
        const bool speculative_unused = moe_group_has_speculative_unused(ctx, m->layer, e);
        const bool recent = moe_group_used_within_tokens(ctx, g, recent_tokens) ||
            moe_group_is_recently_used(ctx, g) || moe_group_in_cooldown(ctx, g);
        const bool high_sequence = moe_group_is_high_sequence(ctx, g, g.hot_score);
        const bool early_high = moe_group_is_early_high_reuse(g, g.hot_score);

        if (g.pinned) {
            ctx.eam_evict_protect_pinned.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (active || current_layer) {
            ctx.eam_evict_protect_active.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        if (!speculative_unused) {
            if (recent) {
                ctx.eam_evict_protect_recent.fetch_add(1, std::memory_order_relaxed);
            }
            if (high_sequence || moe_is_hot(ctx, *m, e)) {
                ctx.eam_evict_protect_high.fetch_add(1, std::memory_order_relaxed);
            }
            if (early_high) {
                ctx.eam_evict_protect_early.fetch_add(1, std::memory_order_relaxed);
            }
        }

        uint64_t dist = UINT64_MAX;
        if (g.next_use_epoch != UINT64_MAX && g.next_use_epoch > ctx.exec_epoch) {
            dist = g.next_use_epoch - ctx.exec_epoch;
        }
        const int layer_dist = moe_forward_layer_distance(ctx, m->layer);
        const bool spec_evictable = speculative_unused &&
            moe_group_speculative_unused_evictable(ctx, g, dist, layer_dist);
        const double speculative_bonus = spec_evictable ?
            moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_SPEC_UNUSED_BONUS", 3.0e8) : 0.0;
        const double low_score_bonus = std::max(0.0,
                moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_LOW_SCORE_CEIL", 140.0) - g.hot_score) * 16384.0;
        const double dist_score = dist == UINT64_MAX ? 1.0e9 : (double) dist * 8192.0;
        const double recent_penalty = recent ? 1.0e8 : 0.0;
        const double spec_guard_penalty = speculative_unused && !spec_evictable ? 4.0e8 : 0.0;
        const double size_bonus = (double) bytes / 1048576.0;
        const double bit_bonus = moe_group_resident_bit_score(ctx, m->layer, e) * 8.0;
        const double high_penalty = high_sequence || moe_is_hot(ctx, *m, e) ? 5.0e8 : 0.0;
        const double early_penalty = early_high ? 2.0e8 : 0.0;
        const double score = speculative_bonus + low_score_bonus + dist_score +
            (double) lru_depth * 1024.0 + size_bonus + bit_bonus -
            recent_penalty - spec_guard_penalty - high_penalty - early_penalty;
        if (score > best.score) {
            best.layer = m->layer;
            best.expert = e;
            best.score = score;
            best.belady = dist != UINT64_MAX;
            best.reason = spec_evictable ? victim_candidate::SPEC_UNUSED :
                (dist == UINT64_MAX || dist > (uint64_t) std::max(1, ctx.params.active_window) ?
                    victim_candidate::FAR_FUTURE : victim_candidate::LOW_SCORE);
        }
    }

    if (best.layer < 0) {
        for (auto it = ctx.lru.end(); it != ctx.lru.begin(); ) {
            --it;
            moe_managed * m = it->first;
            const int e = it->second;
            if (m == nullptr || e < 0 || e >= m->n_expert || m->layer < 0) {
                continue;
            }
            const uint64_t key = moe_group_key(m->layer, e);
            auto git = ctx.groups.find(key);
            if (git != ctx.groups.end() && git->second.pinned) {
                continue;
            }
            if (git != ctx.groups.end() && moe_group_is_active(ctx, git->second)) {
                continue;
            }
            if (ctx.profile_last_layer >= 0 && m->layer == ctx.profile_last_layer) {
                continue;
            }
            if (moe_group_has_inflight_or_queued(ctx, m->layer, e)) {
                continue;
            }
            best.layer = m->layer;
            best.expert = e;
            best.belady = false;
            best.reason = victim_candidate::RELAXED;
            break;
        }
    }

    if (best.layer < 0) {
        if (ctx.profile) {
            ctx.prof_victim_select_us.fetch_add((moe_profile_now_ns() - prof_t0) / 1000, std::memory_order_relaxed);
        }
        return false;
    }

    const size_t released = moe_release_group_locked(ctx, best.layer, best.expert);
    if (released == 0) {
        if (ctx.profile) {
            ctx.prof_victim_select_us.fetch_add((moe_profile_now_ns() - prof_t0) / 1000, std::memory_order_relaxed);
        }
        return false;
    }
    if (best.belady) {
        ctx.belady_evictions.fetch_add(1, std::memory_order_relaxed);
    } else {
        ctx.lru_fallback_evictions.fetch_add(1, std::memory_order_relaxed);
    }
    switch (best.reason) {
        case victim_candidate::SPEC_UNUSED:
            ctx.eam_evict_spec_unused.fetch_add(1, std::memory_order_relaxed);
            break;
        case victim_candidate::LOW_SCORE:
            ctx.eam_evict_low_score.fetch_add(1, std::memory_order_relaxed);
            break;
        case victim_candidate::FAR_FUTURE:
            ctx.eam_evict_far_future.fetch_add(1, std::memory_order_relaxed);
            break;
        case victim_candidate::RELAXED:
            ctx.eam_evict_lru_relaxed.fetch_add(1, std::memory_order_relaxed);
            break;
    }
    if (ctx.profile) {
        ctx.prof_victim_select_us.fetch_add((moe_profile_now_ns() - prof_t0) / 1000, std::memory_order_relaxed);
    }
    return true;
}

// Read expert slice e of m into m.buf + e*stride. No shared state: uses a
// thread-local bounce, so it is safe to run outside ctx.mtx. Returns true on
// success.
static size_t moe_read_size(llama_moe_buffer_context & ctx, const moe_managed & m, int target_bits) {
    if (!ctx.params.dynamic_bits_real) {
        return m.stride;
    }
    if (!moe_native_entry_available(ctx, m, target_bits) &&
            !ctx.dyn_real_warned.exchange(true, std::memory_order_relaxed) &&
            ctx.params.debug_log) {
        std::fprintf(stderr,
                "llama_moe_buffer: LLAMA_LAZY_MOE_DYNBITS_REAL requires exact low-bit/MWQ expert data; "
                "falling back to full expert-slice reads\n");
    }
    return m.stride;
}

static bool moe_sidecar_read_slice(llama_moe_buffer_context & ctx, const moe_managed & m, int e, int target_bits, uint8_t * dst, size_t & bytes_read) {
    if (!ctx.params.dynamic_bits_real || ctx.sidecar_fd < 0) {
        return false;
    }

    const std::string key = sidecar_key(m.name, e, target_bits);
    auto it = ctx.sidecar.find(key);
    if (it == ctx.sidecar.end()) {
        ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const auto & ent = it->second;
    if (ent.decoded_size != m.stride) {
        if (ctx.params.debug_log) {
            std::fprintf(stderr,
                    "llama_moe_buffer: sidecar decoded size mismatch %s expert %d bits %d (%llu != %zu)\n",
                    m.name.c_str(), e, target_bits,
                    (unsigned long long) ent.decoded_size, m.stride);
        }
        return false;
    }

    if (ent.codec == MOE_SIDECAR_CODEC_RAW) {
        if (ent.encoded_size != m.stride) {
            if (ctx.params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: raw sidecar size mismatch %s expert %d bits %d (%llu != %zu)\n",
                        m.name.c_str(), e, target_bits,
                        (unsigned long long) ent.encoded_size, m.stride);
            }
            return false;
        }
        if (!read_full(ctx.sidecar_fd, dst, m.stride, ent.offset)) {
            ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        bytes_read = (size_t) ent.encoded_size;
        ctx.sidecar_hits.fetch_add(1, std::memory_order_relaxed);
        ctx.sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
        return true;
    }

    if (ent.codec == MOE_SIDECAR_CODEC_GGML_QUANT) {
        const ggml_type src_type = sidecar_type_for_bits(ent.bits);
        if (src_type == GGML_TYPE_COUNT || m.type == GGML_TYPE_COUNT || m.elems <= 0) {
            return false;
        }

        const size_t expected_src = ggml_row_size(src_type, m.elems);
        if (ent.encoded_size != expected_src) {
            if (ctx.params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: quant sidecar size mismatch %s expert %d bits %d (%llu != %zu)\n",
                        m.name.c_str(), e, target_bits,
                        (unsigned long long) ent.encoded_size, expected_src);
            }
            return false;
        }

        if (src_type == m.type && ent.encoded_size == m.stride) {
            if (!read_full(ctx.sidecar_fd, dst, m.stride, ent.offset)) {
                ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
                return false;
            }
            bytes_read = (size_t) ent.encoded_size;
            ctx.sidecar_hits.fetch_add(1, std::memory_order_relaxed);
            ctx.sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
            return true;
        }

        const ggml_type_traits * src_traits = ggml_get_type_traits(src_type);
        const ggml_type_traits * dst_traits = ggml_get_type_traits(m.type);
        if (src_traits == nullptr || src_traits->to_float == nullptr ||
                (m.type != GGML_TYPE_F32 && (dst_traits == nullptr || dst_traits->from_float_ref == nullptr))) {
            if (ctx.params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: cannot transcode sidecar %s expert %d bits %d (%s -> %s)\n",
                        m.name.c_str(), e, target_bits,
                        ggml_type_name(src_type), ggml_type_name(m.type));
            }
            return false;
        }

        if (moe_cache_try_read_or_begin(ctx, key, ent, dst, bytes_read)) {
            return true;
        }

        bool ok = false;
        tls_encoded.resize((size_t) ent.encoded_size);
        tls_f32.resize((size_t) m.elems);
        if (!read_full(ctx.sidecar_fd, tls_encoded.data(), tls_encoded.size(), ent.offset)) {
            ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
            moe_cache_finish(ctx, key, ent, dst, false);
            return false;
        }

        src_traits->to_float(tls_encoded.data(), tls_f32.data(), m.elems);
        if (m.type == GGML_TYPE_F32) {
            if (m.stride != tls_f32.size() * sizeof(float)) {
                moe_cache_finish(ctx, key, ent, dst, false);
                return false;
            }
            std::memcpy(dst, tls_f32.data(), m.stride);
            ok = true;
        } else {
            const size_t expected_dst = ggml_row_size(m.type, m.elems);
            if (expected_dst != m.stride) {
                moe_cache_finish(ctx, key, ent, dst, false);
                return false;
            }
            dst_traits->from_float_ref(tls_f32.data(), dst, m.elems);
            ok = true;
        }
        moe_cache_finish(ctx, key, ent, dst, ok);

        bytes_read = (size_t) ent.encoded_size;
        ctx.sidecar_hits.fetch_add(1, std::memory_order_relaxed);
        ctx.sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
        return true;
    }

    if (ent.codec == MOE_SIDECAR_CODEC_MWQ) {
        const ggml_type_traits * dst_traits = ggml_get_type_traits(m.type);
        if (m.type != GGML_TYPE_F32 && (dst_traits == nullptr || dst_traits->from_float_ref == nullptr)) {
            return false;
        }

        if (moe_cache_try_read_or_begin(ctx, key, ent, dst, bytes_read)) {
            return true;
        }

        bool ok = false;
        tls_encoded.resize((size_t) ent.encoded_size);
        tls_f32.resize((size_t) m.elems);
        if (!read_full(ctx.sidecar_fd, tls_encoded.data(), tls_encoded.size(), ent.offset)) {
            ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
            moe_cache_finish(ctx, key, ent, dst, false);
            return false;
        }
        const uint64_t unpack_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        if (!mwq_decode_to_f32(tls_encoded.data(), tls_encoded.size(), ent.bits, ent.block_size, tls_f32.data(), m.elems)) {
            if (ctx.params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: invalid MWQ sidecar %s expert %d bits %d\n",
                        m.name.c_str(), e, target_bits);
            }
            moe_cache_finish(ctx, key, ent, dst, false);
            return false;
        }
        if (ctx.profile) {
            ctx.prof_q2_unpack_us.fetch_add((moe_profile_now_ns() - unpack_t0) / 1000, std::memory_order_relaxed);
        }

        if (m.type == GGML_TYPE_F32) {
            if (m.stride != tls_f32.size() * sizeof(float)) {
                moe_cache_finish(ctx, key, ent, dst, false);
                return false;
            }
            std::memcpy(dst, tls_f32.data(), m.stride);
            ok = true;
        } else {
            const size_t expected_dst = ggml_row_size(m.type, m.elems);
            if (expected_dst != m.stride) {
                moe_cache_finish(ctx, key, ent, dst, false);
                return false;
            }
            dst_traits->from_float_ref(tls_f32.data(), dst, m.elems);
            ok = true;
        }
        moe_cache_finish(ctx, key, ent, dst, ok);

        bytes_read = (size_t) ent.encoded_size;
        ctx.sidecar_hits.fetch_add(1, std::memory_order_relaxed);
        ctx.sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
        return true;
    }

    if (ent.codec == MOE_SIDECAR_CODEC_MWQ_HIER) {
        const ggml_type_traits * dst_traits = ggml_get_type_traits(m.type);
        if (m.type != GGML_TYPE_F32 && (dst_traits == nullptr || dst_traits->from_float_ref == nullptr)) {
            return false;
        }

        if (moe_cache_try_read_or_begin(ctx, key, ent, dst, bytes_read)) {
            return true;
        }

        bool ok = false;
        tls_encoded.resize((size_t) ent.encoded_size);
        tls_f32.resize((size_t) m.elems);
        if (!read_full(ctx.sidecar_fd, tls_encoded.data(), tls_encoded.size(), ent.offset)) {
            ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
            moe_cache_finish(ctx, key, ent, dst, false);
            return false;
        }
        const uint64_t unpack_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        if (!mwq_decode_hier_to_f32(tls_encoded.data(), tls_encoded.size(), ent.bits, ent.block_size,
                    ent.scale_group, ent.outlier_max, tls_f32.data(), m.elems)) {
            if (ctx.params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: invalid MWQ-HIER sidecar %s expert %d bits %d\n",
                        m.name.c_str(), e, target_bits);
            }
            moe_cache_finish(ctx, key, ent, dst, false);
            return false;
        }
        if (ctx.profile) {
            ctx.prof_q2_unpack_us.fetch_add((moe_profile_now_ns() - unpack_t0) / 1000, std::memory_order_relaxed);
        }

        if (m.type == GGML_TYPE_F32) {
            if (m.stride != tls_f32.size() * sizeof(float)) {
                moe_cache_finish(ctx, key, ent, dst, false);
                return false;
            }
            std::memcpy(dst, tls_f32.data(), m.stride);
            ok = true;
        } else {
            const size_t expected_dst = ggml_row_size(m.type, m.elems);
            if (expected_dst != m.stride) {
                moe_cache_finish(ctx, key, ent, dst, false);
                return false;
            }
            dst_traits->from_float_ref(tls_f32.data(), dst, m.elems);
            ok = true;
        }
        moe_cache_finish(ctx, key, ent, dst, ok);

        bytes_read = (size_t) ent.encoded_size;
        ctx.sidecar_hits.fetch_add(1, std::memory_order_relaxed);
        ctx.sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
        return true;
    }

    if (ctx.params.debug_log) {
        std::fprintf(stderr,
                "llama_moe_buffer: sidecar codec %u not implemented for %s expert %d bits %d; fallback\n",
                ent.codec, m.name.c_str(), e, target_bits);
    }
    return false;
}

static bool moe_pread_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, int target_bits, size_t & bytes_read) {
    const size_t foff = m.file_offset + (size_t) e * m.stride;
    uint8_t *    dst  = m.buf + (size_t) e * m.stride;
    bytes_read = 0;

    uint64_t read_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    if (moe_sidecar_read_slice(ctx, m, e, target_bits, dst, bytes_read)) {
        if (ctx.profile) {
            ctx.prof_sidecar_read_us.fetch_add((moe_profile_now_ns() - read_t0) / 1000, std::memory_order_relaxed);
            ctx.prof_sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
            ctx.prof_sidecar_read_count.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }

    const size_t read_size = moe_read_size(ctx, m, target_bits);
    read_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    if (m.direct) {
        const size_t A    = ctx.align;
        const size_t aoff = foff & ~(A - 1);
        const size_t head = foff - aoff;
        size_t want = (head + read_size + A - 1) & ~(A - 1);
        if (aoff + want > m.fsize) want = m.fsize - aoff;
        if (head + read_size > tls_bcap) {
            free(tls_bounce);
            tls_bcap = ((head + read_size + 2 * A) + A - 1) & ~(A - 1);
            if (posix_memalign((void **) &tls_bounce, A, tls_bcap) != 0) {
                tls_bounce = nullptr;
                tls_bcap   = 0;
            }
        }
        if (tls_bounce == nullptr || want > tls_bcap) {
            return false;
        }
        ssize_t r = pread(m.fd, tls_bounce, want, (off_t) aoff);
        if (r < 0 || (size_t) r < head + read_size) {
            return false;
        }
        std::memcpy(dst, tls_bounce + head, read_size);
        bytes_read = read_size;
        if (ctx.profile) {
            ctx.prof_sidecar_read_us.fetch_add((moe_profile_now_ns() - read_t0) / 1000, std::memory_order_relaxed);
            ctx.prof_sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
            ctx.prof_sidecar_read_count.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }
    // Buffered fallback (non-block device etc.).
    size_t left = read_size; off_t off = (off_t) foff; uint8_t * d = dst;
    while (left > 0) {
        ssize_t r = pread(m.fd, d, left, off);
        if (r <= 0) return false;
        d += r; off += r; left -= (size_t) r;
    }
#if defined(POSIX_FADV_DONTNEED)
    posix_fadvise(m.fd, (off_t) foff, (off_t) read_size, POSIX_FADV_DONTNEED);
#endif
    bytes_read = read_size;
    if (ctx.profile) {
        ctx.prof_sidecar_read_us.fetch_add((moe_profile_now_ns() - read_t0) / 1000, std::memory_order_relaxed);
        ctx.prof_sidecar_bytes.fetch_add(bytes_read, std::memory_order_relaxed);
        ctx.prof_sidecar_read_count.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

static void moe_stream_mwq_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank,
        bool * out_entry_missing, bool * out_read_failed) {
    if (e < 0 || e >= m.n_expert) return;

    if (moe_native_entry_available(ctx, m, target_bits)) {
        moe_stream_slice(ctx, m, e, touch, target_bits, rank);
        return;
    }

    std::unique_lock<std::mutex> lk(ctx.mtx);
    uint64_t lookup_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    for (;;) {
        const int loaded_bits = moe_mwq_resolve_loaded_bits(m, e, target_bits);
        if (loaded_bits > 0) {
            if (ctx.profile) {
                ctx.prof_cache_lookup_us.fetch_add((moe_profile_now_ns() - lookup_t0) / 1000, std::memory_order_relaxed);
            }
            if (touch) {
                if (m.activation[e] != UINT32_MAX) { m.activation[e]++; m.total_act++; }
                moe_group_touch_locked(ctx, m, e, rank, 1);
                ctx.hits.fetch_add(1, std::memory_order_relaxed);
                const bool was_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
                moe_group_note_cache_touch_locked(ctx, m, e, was_prefetched, false);
                if (was_prefetched) {
                    ctx.prof_prefetch_hit.fetch_add(1, std::memory_order_relaxed);
                    m.prefetched[e] = false;
                } else {
                    ctx.prof_cache_hit.fetch_add(1, std::memory_order_relaxed);
                }
                if (e < (int) m.resident_touched.size()) {
                    m.resident_touched[e] = true;
                }
                if (loaded_bits != target_bits) {
                    ctx.mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
                }
            }
            return;
        }

        const char s = m.resident[e];
        if (s == ST_RESIDENT) {
            if (moe_native_resident_compatible(ctx, m, e, target_bits)) {
                ctx.lru.splice(ctx.lru.begin(), ctx.lru, m.lru_pos[e]);
                if (ctx.profile) {
                    ctx.prof_cache_lookup_us.fetch_add((moe_profile_now_ns() - lookup_t0) / 1000, std::memory_order_relaxed);
                }
                if (touch) {
                    if (m.activation[e] != UINT32_MAX) { m.activation[e]++; m.total_act++; }
                    moe_group_touch_locked(ctx, m, e, rank, 1);
                    ctx.hits.fetch_add(1, std::memory_order_relaxed);
                    const bool was_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
                    moe_group_note_cache_touch_locked(ctx, m, e, was_prefetched, false);
                    if (was_prefetched) {
                        ctx.prof_prefetch_hit.fetch_add(1, std::memory_order_relaxed);
                        m.prefetched[e] = false;
                    } else {
                        ctx.prof_cache_hit.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (e < (int) m.resident_touched.size()) {
                        m.resident_touched[e] = true;
                    }
                    if (m.resident_bits[e] != target_bits) {
                        ctx.mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
                    }
                }
                return;
            }
            if (m.resident_mwq[e] && m.resident_bits[e] == target_bits) {
                ctx.lru.splice(ctx.lru.begin(), ctx.lru, m.lru_pos[e]);
                if (ctx.profile) {
                    ctx.prof_cache_lookup_us.fetch_add((moe_profile_now_ns() - lookup_t0) / 1000, std::memory_order_relaxed);
                }
                if (touch) {
                    if (m.activation[e] != UINT32_MAX) { m.activation[e]++; m.total_act++; }
                    moe_group_touch_locked(ctx, m, e, rank, 1);
                    ctx.hits.fetch_add(1, std::memory_order_relaxed);
                    const bool was_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
                    moe_group_note_cache_touch_locked(ctx, m, e, was_prefetched, false);
                    if (was_prefetched) {
                        ctx.prof_prefetch_hit.fetch_add(1, std::memory_order_relaxed);
                        m.prefetched[e] = false;
                    } else {
                        ctx.prof_cache_hit.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (e < (int) m.resident_touched.size()) {
                        m.resident_touched[e] = true;
                    }
                }
                return;
            }
            if (m.resident_mwq[e]) {
                moe_mark_mwq_transient(ctx, m, e, m.resident_bits[e]);
            }
            ctx.lru.erase(m.lru_pos[e]);
            m.resident[e] = ST_COLD;
            m.resident_mwq[e] = false;
            m.resident_bits[e] = 0;
            ctx.resident_bytes -= std::min(ctx.resident_bytes, m.resident_size[e]);
            m.resident_size[e] = 0;
            break;
        }
        if (s == ST_INFLIGHT) {
            if (touch) {
                moe_group_note_prefetch_late_locked(ctx, m, e);
                if (ctx.profile) {
                    ctx.prof_prefetch_late.fetch_add(1, std::memory_order_relaxed);
                }
            }
            const auto wait_t0 = std::chrono::steady_clock::now();
            ctx.cv_done.wait(lk);
            const auto wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_t0).count();
            ctx.stream_wait_ns.fetch_add((uint64_t) wait_ns, std::memory_order_relaxed);
            ctx.stream_wait_count.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        break;
    }
    if (ctx.profile) {
        ctx.prof_cache_lookup_us.fetch_add((moe_profile_now_ns() - lookup_t0) / 1000, std::memory_order_relaxed);
        if (touch) {
            ctx.prof_cache_miss.fetch_add(1, std::memory_order_relaxed);
        }
    }

    const moe_sidecar_entry * ent = moe_mwq_entry(ctx, m, e, target_bits);
    if (ent == nullptr) {
        if (out_entry_missing) *out_entry_missing = true;
        lk.unlock();
        moe_stream_slice(ctx, m, e, touch, target_bits, rank);
        return;
    }

    if (!moe_ensure_mwq_buf(m, target_bits, (size_t) ent->encoded_size, ent->block_size, ent->codec, ent->scale_group, ent->outlier_max)) {
        return;
    }
    moe_mwq_store & store = m.mwq[target_bits];
    uint8_t * dst = store.buf + (size_t) e * store.stride;

    m.resident[e] = ST_INFLIGHT;
    m.resident_mwq[e] = true;
    m.resident_bits[e] = target_bits;
    m.resident_size[e] = (size_t) ent->encoded_size;
    if (ctx.params.budget_bytes > 0) {
        while (ctx.resident_bytes + (size_t) ent->encoded_size > ctx.params.budget_bytes && moe_evict_lru(ctx)) {}
    }
    if (ctx.profile && e < (int) m.last_evicted_token.size() && m.last_evicted_token[e] > 0) {
        const uint64_t dt = ctx.profile_token_epoch - m.last_evicted_token[e];
        if (dt <= 1) {
            ctx.prof_evict_reloaded_within_1_token.fetch_add(1, std::memory_order_relaxed);
        }
        if (dt <= 4) {
            ctx.prof_evict_reloaded_within_4_tokens.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ctx.resident_bytes += (size_t) ent->encoded_size;

    lk.unlock();
    const uint64_t read_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    const bool ok = read_full(ctx.sidecar_fd, dst, (size_t) ent->encoded_size, ent->offset);
    if (ctx.profile) {
        ctx.prof_sidecar_read_us.fetch_add((moe_profile_now_ns() - read_t0) / 1000, std::memory_order_relaxed);
        ctx.prof_sidecar_bytes.fetch_add((size_t) ent->encoded_size, std::memory_order_relaxed);
        ctx.prof_sidecar_read_count.fetch_add(1, std::memory_order_relaxed);
    }
    lk.lock();

    if (ok) {
        m.resident[e] = ST_RESIDENT;
        if (e >= 0 && e < (int) store.loaded.size()) {
            store.loaded[e] = true;
        }
        ctx.lru.push_front({&m, e});
        m.lru_pos[e] = ctx.lru.begin();
        if (e < (int) m.resident_touched.size()) {
            m.resident_touched[e] = touch;
        }
        if (e < (int) m.prefetched.size()) {
            m.prefetched[e] = !touch;
        }
        if (touch && m.activation[e] != UINT32_MAX) {
            m.activation[e]++;
            m.total_act++;
            moe_group_touch_locked(ctx, m, e, rank, 1);
            moe_group_note_cache_touch_locked(ctx, m, e, false, true);
        }
        ctx.streams.fetch_add(1, std::memory_order_relaxed);
        ctx.bytes_read.fetch_add((size_t) ent->encoded_size, std::memory_order_relaxed);
        ctx.sidecar_hits.fetch_add(1, std::memory_order_relaxed);
        ctx.sidecar_bytes.fetch_add((size_t) ent->encoded_size, std::memory_order_relaxed);
        ctx.mwq_bytes_read.fetch_add((size_t) ent->encoded_size, std::memory_order_relaxed);
        ctx.mwq_actual_full_bytes.fetch_add(m.stride, std::memory_order_relaxed);
        ctx.mwq_actual_saved_bytes.fetch_add(
                m.stride > (size_t) ent->encoded_size ? m.stride - (size_t) ent->encoded_size : 0,
                std::memory_order_relaxed);
    } else {
        m.resident[e] = ST_COLD;
        m.resident_mwq[e] = false;
        m.resident_bits[e] = 0;
        if (e >= 0 && e < (int) store.loaded.size()) {
            store.loaded[e] = false;
        }
        ctx.resident_bytes -= std::min(ctx.resident_bytes, m.resident_size[e]);
        m.resident_size[e] = 0;
        ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
        if (out_read_failed) *out_read_failed = true;
        const uint64_t fail_no = ctx.mwq_stream_fail_total.fetch_add(1, std::memory_order_relaxed);
        if (ctx.params.debug_log && (fail_no % MOE_STREAM_FAIL_LOG_EVERY) == 0) {
            std::fprintf(stderr, "llama_moe_buffer: MWQ stream failed %s expert %d bits %d (occurrence #%llu)\n",
                    m.name.c_str(), e, target_bits, (unsigned long long) fail_no + 1);
        }
    }
    ctx.cv_done.notify_all();
}

// Ensure expert slice e of m is resident. Coordinates the synchronous callback
// and the async worker via a per-slot state machine: the pread happens outside
// the residency mutex; a second thread requesting the same slice waits for the
// in-flight read to complete instead of issuing a duplicate read.
static void moe_stream_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank) {
    if (e < 0 || e >= m.n_expert) return;

    std::unique_lock<std::mutex> lk(ctx.mtx);
    uint64_t lookup_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    for (;;) {
        const char s = m.resident[e];
        if (s == ST_RESIDENT) {
            const int need_bits = std::max(1, target_bits);
            if (!m.resident_mwq[e] && (!ctx.params.dynamic_bits_real || m.resident_bits[e] >= need_bits)) {
                ctx.lru.splice(ctx.lru.begin(), ctx.lru, m.lru_pos[e]);  // touch (MRU)
                if (ctx.profile) {
                    ctx.prof_cache_lookup_us.fetch_add((moe_profile_now_ns() - lookup_t0) / 1000, std::memory_order_relaxed);
                }
                if (touch) {
                    if (m.activation[e] != UINT32_MAX) { m.activation[e]++; m.total_act++; }  // hot tracking
                    moe_group_touch_locked(ctx, m, e, rank, 1);
                    ctx.hits.fetch_add(1, std::memory_order_relaxed);
                    const bool was_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
                    moe_group_note_cache_touch_locked(ctx, m, e, was_prefetched, false);
                    if (was_prefetched) {
                        ctx.prof_prefetch_hit.fetch_add(1, std::memory_order_relaxed);
                        m.prefetched[e] = false;
                    } else {
                        ctx.prof_cache_hit.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (e < (int) m.resident_touched.size()) {
                        m.resident_touched[e] = true;
                    }
                }
                return;
            }
            ctx.lru.erase(m.lru_pos[e]);
            m.resident[e] = ST_COLD;
            m.resident_mwq[e] = false;
            m.resident_bits[e] = 0;
            ctx.resident_bytes -= std::min(ctx.resident_bytes, m.resident_size[e]);
            m.resident_size[e] = 0;
            break;
        }
        if (s == ST_INFLIGHT) {
            if (touch) {
                moe_group_note_prefetch_late_locked(ctx, m, e);
                if (ctx.profile) {
                    ctx.prof_prefetch_late.fetch_add(1, std::memory_order_relaxed);
                }
            }
            const auto wait_t0 = std::chrono::steady_clock::now();
            ctx.cv_done.wait(lk);   // another thread is loading it; wait for completion
            const auto wait_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait_t0).count();
            ctx.stream_wait_ns.fetch_add((uint64_t) wait_ns, std::memory_order_relaxed);
            ctx.stream_wait_count.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        break;  // ST_COLD → we load it
    }
    if (ctx.profile) {
        ctx.prof_cache_lookup_us.fetch_add((moe_profile_now_ns() - lookup_t0) / 1000, std::memory_order_relaxed);
        if (touch) {
            ctx.prof_cache_miss.fetch_add(1, std::memory_order_relaxed);
        }
    }

    if (!moe_ensure_buf(m)) {
        return;
    }
    m.resident[e] = ST_INFLIGHT;
    m.resident_mwq[e] = false;
    m.resident_size[e] = m.stride;
    if (ctx.params.budget_bytes > 0) {
        while (ctx.resident_bytes + m.stride > ctx.params.budget_bytes && moe_evict_lru(ctx)) {}
    }
    if (ctx.profile && e < (int) m.last_evicted_token.size() && m.last_evicted_token[e] > 0) {
        const uint64_t dt = ctx.profile_token_epoch - m.last_evicted_token[e];
        if (dt <= 1) {
            ctx.prof_evict_reloaded_within_1_token.fetch_add(1, std::memory_order_relaxed);
        }
        if (dt <= 4) {
            ctx.prof_evict_reloaded_within_4_tokens.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ctx.resident_bytes += m.stride;   // reserve before the (unlocked) read

    lk.unlock();
    size_t read_size = 0;
    const bool ok = moe_pread_slice(ctx, m, e, target_bits, read_size);
    lk.lock();

    if (ok) {
        m.resident[e] = ST_RESIDENT;
        m.resident_mwq[e] = false;
        m.resident_bits[e] = ctx.params.dynamic_bits_real ? std::max(1, target_bits) : ctx.params.base_bits;
        m.resident_size[e] = m.stride;
        ctx.lru.push_front({&m, e});
        m.lru_pos[e] = ctx.lru.begin();
        if (e < (int) m.resident_touched.size()) {
            m.resident_touched[e] = touch;
        }
        if (e < (int) m.prefetched.size()) {
            m.prefetched[e] = !touch;
        }
        if (touch && m.activation[e] != UINT32_MAX) {
            m.activation[e]++;
            m.total_act++;
            moe_group_touch_locked(ctx, m, e, rank, 1);
            moe_group_note_cache_touch_locked(ctx, m, e, false, true);
        }
        ctx.streams.fetch_add(1, std::memory_order_relaxed);
        ctx.bytes_read.fetch_add(read_size, std::memory_order_relaxed);
    } else {
        m.resident[e] = ST_COLD;
        m.resident_mwq[e] = false;
        m.resident_bits[e] = 0;
        ctx.resident_bytes -= std::min(ctx.resident_bytes, m.resident_size[e]);
        m.resident_size[e] = 0;
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
    llama_moe_buffer_prefetch_ranked(ctx, layer, experts, nullptr, n_experts);
}

void llama_moe_buffer_prefetch_ranked(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        const float *              scores,
        int                        n_experts) {
    if (ctx == nullptr || !ctx->params.enabled || experts == nullptr || n_experts <= 0) {
        return;
    }
    const uint64_t submit_t0 = ctx->profile ? moe_profile_now_ns() : 0;
    auto it = ctx->by_layer.find(layer);
    if (it == ctx->by_layer.end()) {
        return;
    }
    int n_enqueued = 0;
    {
        std::lock_guard<std::mutex> lk(ctx->qmtx);
        for (moe_managed * m : it->second) {
            for (int i = 0; i < n_experts; ++i) {
                const int e = experts[i];
                const float score = scores != nullptr ? scores[i] : 0.0f;
                n_enqueued += moe_enqueue_prefetch(*ctx, *m, e, i, score) ? 1 : 0;
            }
        }
    }
    if (n_enqueued > 1) {
        ctx->cv_q.notify_all();
    } else if (n_enqueued == 1) {
        ctx->cv_q.notify_one();
    }
    if (ctx->profile) {
        ctx->prof_sidecar_submit_us.fetch_add((moe_profile_now_ns() - submit_t0) / 1000, std::memory_order_relaxed);
    }
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
    const uint64_t callback_t0 = (ith == 0 && ctx->profile) ? moe_profile_now_ns() : 0;

    if (ith == 0) {
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            if (m.layer >= 0) {
                if (ctx->profile_last_layer >= 0 && m.layer < ctx->profile_last_layer) {
                    ++ctx->profile_token_epoch;
                }
                ctx->profile_last_layer = m.layer;
            }
            ++ctx->exec_epoch;
        }
        moe_cleanup_mwq_transients(*ctx);

        ggml_tensor * ids = op->src[2];  // selected experts, I32
        const bool use_mwq_kernel = moe_op_all_mwq_available(*ctx, m, ids);
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            if (!use_mwq_kernel && !moe_ensure_buf(m)) {
                return false;
            }
            m.tensor = exps;
        }
        // Repoint only for the fallback original-layout kernel. The MWQ override
        // reads the managed MWQ buffers directly.
        if (!use_mwq_kernel && exps->data != m.buf) {
            exps->data = m.buf;
        }
        if (ids != nullptr && ids->data != nullptr && ids->type == GGML_TYPE_I32) {
            std::vector<int> op_target_bits;
            std::vector<int> op_counts;
            std::vector<int> op_min_rank;
            if (ctx->params.dynamic_bits_real) {
                const uint64_t route_t0 = ctx->profile ? moe_profile_now_ns() : 0;
                op_target_bits.assign(m.n_expert, 0);
                op_counts.assign(m.n_expert, 0);
                op_min_rank.assign(m.n_expert, 999999);
                for (int64_t token = 0; token < ids->ne[1]; ++token) {
                    for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                        const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                        if (expert < 0 || expert >= m.n_expert) {
                            continue;
                        }
                        const int64_t flat = token * ids->ne[0] + rank;
                        const int target_bits = moe_target_bits_for_id_index(*ctx, m, expert, flat);
                        op_target_bits[expert] = std::max(op_target_bits[expert], target_bits);
                        op_counts[expert]++;
                        op_min_rank[expert] = std::min(op_min_rank[expert], (int) rank);
                    }
                }
                if (ctx->profile) {
                    ctx->prof_route_us.fetch_add((moe_profile_now_ns() - route_t0) / 1000, std::memory_order_relaxed);
                }

                for (int expert = 0; expert < m.n_expert; ++expert) {
                    const int target_bits = op_target_bits[expert];
                    if (target_bits <= 0) {
                        continue;
                    }
                    const int best_rank = op_min_rank[expert] == 999999 ? 0 : op_min_rank[expert];
                    if (use_mwq_kernel) {
                        moe_stream_mwq_slice(*ctx, m, expert, true, target_bits, best_rank);
                    } else {
                        moe_stream_slice(*ctx, m, expert, true, target_bits, best_rank);
                    }
                    const int extra = op_counts[expert] - 1;
                    if (extra > 0) {
                        std::lock_guard<std::mutex> lk(ctx->mtx);
                        const uint64_t add = (uint64_t) extra;
                        const uint64_t room = UINT32_MAX - m.activation[expert];
                        const uint64_t inc = std::min(add, room);
                        m.activation[expert] += (uint32_t) inc;
                        m.total_act += inc;
                        moe_group_touch_locked(*ctx, m, expert, best_rank, inc);
                        ctx->hits.fetch_add(add, std::memory_order_relaxed);
                    }
                }
                if (moe_kind(m) == moe_tensor_kind::down && moe_eam_predict_enabled()) {
                    std::vector<int> actual_experts;
                    actual_experts.reserve((size_t) m.n_expert);
                    for (int expert = 0; expert < m.n_expert; ++expert) {
                        if (op_counts[expert] > 0) {
                            actual_experts.push_back(expert);
                        }
                    }
                    moe_eam_predict_after_route(*ctx, m.layer, actual_experts, m.n_expert);
                }
                if (ctx->profile) {
                    ctx->prof_moe_total_us.fetch_add((moe_profile_now_ns() - callback_t0) / 1000, std::memory_order_relaxed);
                }
                return true;
            }

            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                    const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                    const int64_t flat = token * ids->ne[0] + rank;
                    int target_bits = moe_target_bits_for_id_index(*ctx, m, expert, flat);
                    if (expert >= 0 && expert < m.n_expert && !op_target_bits.empty()) {
                        target_bits = std::max(target_bits, op_target_bits[expert]);
                    }
                    if (use_mwq_kernel) {
                        moe_stream_mwq_slice(*ctx, m, expert, true, target_bits, (int) rank);
                    } else {
                        moe_stream_slice(*ctx, m, expert, true, target_bits, (int) rank);  // resident or wait-for-inflight
                    }
                }
            }
        }
        if (ctx->profile) {
            ctx->prof_moe_total_us.fetch_add((moe_profile_now_ns() - callback_t0) / 1000, std::memory_order_relaxed);
        }
    }
    return true;  // managed → caller issues a barrier
}

static inline float moe_silu_f32(float x);

#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
static bool mwq_cpu_has_avx2_fma() {
    static const bool has = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    }();
    return has;
}

static void mwq_sum_x_blocks_avx2(const char * x, int64_t n_col, int block_size, std::vector<float> & out);
static float mwq_dot_row_avx2(const uint8_t * mwq, int bits, int block_size, int64_t row, int64_t n_col, const char * x, const float * sum_x_blocks);
static void mwq_dot_rows_avx2(
        const uint8_t * mwq,
        int             bits,
        int             block_size,
        int64_t         row,
        int             n_rows,
        int64_t         n_col,
        const char *    x,
        const float *   sum_x_blocks,
        float *         dst);
static void mwq_dot_row_batch_avx2(
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float * const *       dsts);
static void mwq_dot_row_batch_values_avx2(
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float *               out);
static void mwq_swiglu_row_batch_values_avx2(
        const uint8_t *       gate_mwq,
        const uint8_t *       up_mwq,
        int                   bits,
        int                   block_size,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float *               out);
#endif

#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
static bool mwq_cpu_has_avx512_q2() {
    static const bool has = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") &&
               __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512dq") &&
               __builtin_cpu_supports("fma");
    }();
    return has;
}

static bool mwq_cpu_has_avx512_vnni_q2() {
    static const bool has = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") &&
               __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512dq") &&
               __builtin_cpu_supports("avx512vnni");
    }();
    return has;
}

static void mwq_swiglu_row_batch_values_hier_q2_avx512(
        const uint8_t *       gate_mwq,
        const uint8_t *       up_mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        int                   prefetch_distance,
        float *               out);
static void mwq_dot_rows_hier_q2_avx512(
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   n_rows,
        int64_t               n_col,
        const char *          x,
        int                   prefetch_distance,
        float *               dst);
static void mwq_dot_row_batch_hier_q2_avx512(
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        int                   prefetch_distance,
        float * const *       dsts);
static void mwq_dot_rows_hier_q2_vnni(
        llama_moe_buffer_context * ctx,
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   n_rows,
        int64_t               n_col,
        const char *          x,
        float *               dst);
static void mwq_dot_row_batch_hier_q2_vnni(
        llama_moe_buffer_context * ctx,
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        float * const *       dsts);
static void mwq_swiglu_row_batch_values_hier_q2_vnni(
        llama_moe_buffer_context * ctx,
        const uint8_t *       gate_mwq,
        const uint8_t *       up_mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        float *               out);
static float mwq_dot_row_range_hier_q2_avx512(
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int64_t               n_col,
        int64_t               col0,
        int                   col_count,
        const float *         x);
static void mwq_qx_int8_build_avx512(
        const char * x,
        int64_t      n_col,
        int          qblock,
        int8_t *     qx,
        float *      scales,
        int32_t *    sums);
#endif

static void mwq_sum_x_blocks(const char * x, size_t x_stride, int64_t n_col, int block_size, std::vector<float> & out) {
    const int64_t n_blocks = n_col / block_size;
    out.resize((size_t) n_blocks);

#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
    if (block_size > 0 && (block_size % 8) == 0 && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
        mwq_sum_x_blocks_avx2(x, n_col, block_size, out);
        return;
    }
#endif

    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        float sum = 0.0f;
        const int64_t base = ib * block_size;
        for (int i = 0; i < block_size; ++i) {
            sum += *(const float *) (x + (base + i) * x_stride);
        }
        out[(size_t) ib] = sum;
    }
}

static int moe_microbatch_size() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_MICROBATCH");
    if (v == nullptr || v[0] == '\0') {
        return 16;
    }
    return std::max(1, std::min(16, std::atoi(v)));
}

static const float * mwq_sum_x_blocks_cached(
        llama_moe_buffer_context & ctx,
        const char * x,
        size_t x_stride,
        int64_t n_col,
        int block_size) {
    if (x_stride != sizeof(float)) {
        mwq_sum_x_blocks(x, x_stride, n_col, block_size, tls_mwq_sum_x);
        ctx.mwq_sum_cache_builds.fetch_add(1, std::memory_order_relaxed);
        return tls_mwq_sum_x.data();
    }
    for (mwq_sum_cache_entry & ent : tls_mwq_sum_cache) {
        if (ent.x == x && ent.block_size == block_size && ent.n_col == n_col) {
            ctx.mwq_sum_cache_hits.fetch_add(1, std::memory_order_relaxed);
            return ent.sums.data();
        }
    }
    mwq_sum_cache_entry ent;
    ent.x = x;
    ent.block_size = block_size;
    ent.n_col = n_col;
    mwq_sum_x_blocks(x, x_stride, n_col, block_size, ent.sums);
    tls_mwq_sum_cache.push_back(std::move(ent));
    ctx.mwq_sum_cache_builds.fetch_add(1, std::memory_order_relaxed);
    return tls_mwq_sum_cache.back().sums.data();
}

static int mwq_vnni_qblock(const llama_moe_buffer_context & ctx, int block_size) {
    int qblock = ctx.params.vnni_block;
    if (qblock != 64 && qblock != block_size) {
        qblock = 64;
    }
    if (qblock <= 0 || block_size % qblock != 0 || qblock % 64 != 0) {
        return 0;
    }
    return qblock;
}

static void mwq_qx_int8_build_scalar(
        const char * x,
        int64_t n_col,
        int qblock,
        int8_t * qx,
        float * scales,
        int32_t * sums) {
    const int64_t n_blocks = n_col / qblock;
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const float * xp = (const float *) (x + (size_t) ib * (size_t) qblock * sizeof(float));
        float amax = 0.0f;
        for (int i = 0; i < qblock; ++i) {
            const float ax = std::fabs(xp[i]);
            if (std::isfinite(ax)) {
                amax = std::max(amax, ax);
            }
        }
        const float scale = amax > 0.0f ? amax * (1.0f / 127.0f) : 0.0f;
        int32_t sum = 0;
        int8_t * qp = qx + (size_t) ib * (size_t) qblock;
        if (scale == 0.0f) {
            std::memset(qp, 0, (size_t) qblock);
        } else {
            const float inv = 1.0f / scale;
            for (int i = 0; i < qblock; ++i) {
                int qi = (int) std::lrintf(xp[i] * inv);
                qi = std::max(-127, std::min(127, qi));
                qp[i] = (int8_t) qi;
                sum += qi;
            }
        }
        scales[(size_t) ib] = scale;
        sums[(size_t) ib] = sum;
    }
}

static void mwq_qx_int8_build_entry(
        const char * x,
        int64_t n_col,
        int qblock,
        mwq_qx_int8_cache_entry & ent) {
    ent.x = x;
    ent.qblock = qblock;
    ent.n_col = n_col;
    const int64_t n_blocks = n_col / qblock;
    ent.qx.resize((size_t) n_blocks * (size_t) qblock);
    ent.scales.resize((size_t) n_blocks);
    ent.sums.resize((size_t) n_blocks);

#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
    if ((qblock % 16) == 0 && mwq_cpu_has_avx512_q2()) {
        mwq_qx_int8_build_avx512(x, n_col, qblock, ent.qx.data(), ent.scales.data(), ent.sums.data());
        return;
    }
#endif

    mwq_qx_int8_build_scalar(x, n_col, qblock, ent.qx.data(), ent.scales.data(), ent.sums.data());
}

static const mwq_qx_int8_cache_entry * mwq_qx_int8_cached(
        llama_moe_buffer_context & ctx,
        const char * x,
        int64_t n_col,
        int qblock) {
    if (qblock <= 0 || n_col <= 0 || (n_col % qblock) != 0) {
        return nullptr;
    }
    if (tls_mwq_qx_shared_cache != nullptr) {
        for (const mwq_qx_int8_cache_entry & ent : *tls_mwq_qx_shared_cache) {
            if (ent.x == x && ent.qblock == qblock && ent.n_col == n_col) {
                ctx.mwq_vnni_qx_hits.fetch_add(1, std::memory_order_relaxed);
                return &ent;
            }
        }
    }
    for (mwq_qx_int8_cache_entry & ent : tls_mwq_qx_int8_cache) {
        if (ent.x == x && ent.qblock == qblock && ent.n_col == n_col) {
            ctx.mwq_vnni_qx_hits.fetch_add(1, std::memory_order_relaxed);
            return &ent;
        }
    }

    mwq_qx_int8_cache_entry ent;
    mwq_qx_int8_build_entry(x, n_col, qblock, ent);
    tls_mwq_qx_int8_cache.push_back(std::move(ent));
    ctx.mwq_vnni_qx_builds.fetch_add(1, std::memory_order_relaxed);
    return &tls_mwq_qx_int8_cache.back();
}

static bool moe_vnni_check_enabled() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_VNNI_CHECK");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static void mwq_qx_unique_push(std::vector<const char *> & xs, const char * x) {
    if (x == nullptr) {
        return;
    }
    for (const char * seen : xs) {
        if (seen == x) {
            return;
        }
    }
    xs.push_back(x);
}

static bool mwq_qx_shared_begin(
        llama_moe_buffer_context & ctx,
        const ggml_tensor * op,
        int ith,
        int nth,
        int64_t n_col,
        int qblock,
        const std::vector<const char *> & xs) {
    if (op == nullptr || nth <= 0 || qblock <= 0 || n_col <= 0 || xs.empty()) {
        tls_mwq_qx_shared_cache = nullptr;
        return false;
    }

    std::unique_lock<std::mutex> lk(ctx.mtx);
    auto it = ctx.qx_cache_map.find(op);
    if (it == ctx.qx_cache_map.end()) {
        it = ctx.qx_cache_map.emplace(op, llama_moe_buffer_context::moe_qx_cache_entry{}).first;
        it->second.remaining = nth;
        it->second.build_remaining = nth;
        it->second.entries.resize(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) {
            mwq_qx_int8_cache_entry & ent = it->second.entries[i];
            ent.x = xs[i];
            ent.qblock = qblock;
            ent.n_col = n_col;
        }
        it->second.initialized = true;
        ctx.cv_qx_cache.notify_all();
    } else {
        while (!it->second.initialized) {
            ctx.cv_qx_cache.wait(lk);
            it = ctx.qx_cache_map.find(op);
            if (it == ctx.qx_cache_map.end()) {
                tls_mwq_qx_shared_cache = nullptr;
                return false;
            }
        }
    }

    std::vector<mwq_qx_int8_cache_entry> * entries = &it->second.entries;
    lk.unlock();

    uint64_t builds = 0;
    for (size_t i = (size_t) std::max(0, ith); i < entries->size(); i += (size_t) nth) {
        mwq_qx_int8_build_entry((*entries)[i].x, n_col, qblock, (*entries)[i]);
        ++builds;
    }
    if (builds != 0) {
        ctx.mwq_vnni_qx_builds.fetch_add(builds, std::memory_order_relaxed);
    }

    lk.lock();
    it = ctx.qx_cache_map.find(op);
    if (it == ctx.qx_cache_map.end()) {
        tls_mwq_qx_shared_cache = nullptr;
        return false;
    }
    if (--it->second.build_remaining == 0) {
        it->second.done = true;
        tls_mwq_qx_shared_cache = &it->second.entries;
        lk.unlock();
        ctx.cv_qx_cache.notify_all();
        return true;
    }
    while (!it->second.done) {
        ctx.cv_qx_cache.wait(lk);
        it = ctx.qx_cache_map.find(op);
        if (it == ctx.qx_cache_map.end()) {
            tls_mwq_qx_shared_cache = nullptr;
            return false;
        }
    }
    tls_mwq_qx_shared_cache = &it->second.entries;
    return true;
}

static void mwq_qx_shared_end(llama_moe_buffer_context & ctx, const ggml_tensor * op) {
    tls_mwq_qx_shared_cache = nullptr;
    if (op == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lk(ctx.mtx);
    auto it = ctx.qx_cache_map.find(op);
    if (it == ctx.qx_cache_map.end()) {
        return;
    }
    it->second.remaining--;
    if (it->second.remaining <= 0) {
        ctx.qx_cache_map.erase(it);
    }
}

static float mwq_dot_row_scalar(const uint8_t * mwq, int bits, int block_size, int64_t row, int64_t n_col, const char * x, size_t x_stride, const float * sum_x_blocks) {
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const uint8_t mask = (uint8_t) ((1u << bits) - 1u);
    const int64_t block0 = (row * n_col) / block_size;
    const int64_t n_blocks = n_col / block_size;

    float acc = 0.0f;
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const uint8_t * block = mwq + (size_t) (block0 + ib) * block_bytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 0));
        const float sc = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 2));
        const uint8_t * q = block + 4;

        float sum_x = sum_x_blocks != nullptr ? sum_x_blocks[ib] : 0.0f;
        float sum_qx = 0.0f;
        const int64_t base = ib * block_size;
        for (int i = 0; i < block_size; ++i) {
            const int bit = i * bits;
            uint32_t packed = q[bit >> 3];
            if ((bit & 7) + bits > 8) {
                packed |= (uint32_t) q[(bit >> 3) + 1] << 8;
            }
            const float xv = *(const float *) (x + (base + i) * x_stride);
            const uint8_t qv = (packed >> (bit & 7)) & mask;
            if (sum_x_blocks == nullptr) {
                sum_x += xv;
            }
            sum_qx += (float) qv * xv;
        }
        acc += mn * sum_x + sc * sum_qx;
    }
    return acc;
}

static float mwq_dot_row_hier_scalar(
        const uint8_t * mwq,
        int bits,
        int block_size,
        int scale_group,
        int outlier_max,
        int64_t n_elem,
        int64_t row,
        int64_t n_col,
        const char * x,
        size_t x_stride,
        const float * sum_x_blocks) {
    const int64_t n_blocks_total = n_elem / block_size;
    const int64_t n_blocks_row = n_col / block_size;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t off_weights = 0;
    const size_t off_mins = off_weights + (size_t) n_blocks_total * qbytes;
    const size_t off_gscales = off_mins + (size_t) n_blocks_total * 2;
    const size_t n_scale_groups = ((size_t) n_blocks_total + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t off_scodes = off_gscales + n_scale_groups * 2;
    const size_t off_outliers = off_scodes + (size_t) n_blocks_total;
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const uint8_t mask = (uint8_t) ((1u << bits) - 1u);

    float acc = 0.0f;
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);

        float sum_x = sum_x_blocks != nullptr ? sum_x_blocks[ib] : 0.0f;
        float sum_qx = 0.0f;
        const int64_t base = ib * block_size;
        for (int i = 0; i < block_size; ++i) {
            const int bit = i * bits;
            uint32_t packed = q[bit >> 3];
            if ((bit & 7) + bits > 8) {
                packed |= (uint32_t) q[(bit >> 3) + 1] << 8;
            }
            const float xv = *(const float *) (x + (base + i) * x_stride);
            const uint8_t qv = (packed >> (bit & 7)) & mask;
            if (sum_x_blocks == nullptr) {
                sum_x += xv;
            }
            sum_qx += (float) qv * xv;
        }
        acc += mn * sum_x + sc * sum_qx;

        const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        for (int j = 0; j < count; ++j) {
            const int idx = ob[1 + j * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + j * 3 + 1));
                acc += residual * *(const float *) (x + (base + idx) * x_stride);
            }
        }
    }
    return acc;
}

#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
#pragma GCC push_options
#pragma GCC target("avx2,fma")

static inline float mwq_hsum256_ps(__m256 v) {
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static void mwq_sum_x_blocks_avx2(const char * x, int64_t n_col, int block_size, std::vector<float> & out) {
    const int64_t n_blocks = n_col / block_size;
    const int groups = block_size / 8;
    out.resize((size_t) n_blocks);

    const float * xf = (const float *) x;
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        __m256 sum = _mm256_setzero_ps();
        for (int g = 0; g < groups; ++g) {
            sum = _mm256_add_ps(sum, _mm256_loadu_ps(xf + ib * block_size + g * 8));
        }
        out[(size_t) ib] = mwq_hsum256_ps(sum);
    }
}

static inline __m256 mwq_q2_f32x8(const uint8_t * q, int group) {
    const uint16_t p = rd_u16(q + group * 2);
    const __m256i qi = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) MWQ_Q2_LUT8[p]));
    return _mm256_cvtepi32_ps(qi);
}

static inline __m256 mwq_q3_f32x8(const uint8_t * q, int group) {
    const int byte0 = group * 3;
    const uint32_t p =
        ((uint32_t) q[byte0 + 0] <<  0) |
        ((uint32_t) q[byte0 + 1] <<  8) |
        ((uint32_t) q[byte0 + 2] << 16);
    const uint64_t qbytes =
        (uint64_t) MWQ_Q3_LUT4[p & 0x0fffu] |
        ((uint64_t) MWQ_Q3_LUT4[(p >> 12) & 0x0fffu] << 32);
    const __m256i qi = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) qbytes));
    return _mm256_cvtepi32_ps(qi);
}

static float mwq_dot_row_avx2(const uint8_t * mwq, int bits, int block_size, int64_t row, int64_t n_col, const char * x, const float * sum_x_blocks) {
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const int64_t block0 = (row * n_col) / block_size;
    const int64_t n_blocks = n_col / block_size;
    const int groups = block_size / 8;

    float acc = 0.0f;
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const uint8_t * block = mwq + (size_t) (block0 + ib) * block_bytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 0));
        const float sc = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 2));
        const uint8_t * q = block + 4;
        const float * xf = (const float *) (x + ib * block_size * sizeof(float));

        __m256 sum_x = _mm256_setzero_ps();
        __m256 sum_qx = _mm256_setzero_ps();
        for (int g = 0; g < groups; ++g) {
            const __m256 xv = _mm256_loadu_ps(xf + g * 8);
            const __m256 qv = bits == 2 ? mwq_q2_f32x8(q, g) : mwq_q3_f32x8(q, g);
            if (sum_x_blocks == nullptr) {
                sum_x = _mm256_add_ps(sum_x, xv);
            }
            sum_qx = _mm256_fmadd_ps(qv, xv, sum_qx);
        }

        const float sx = sum_x_blocks != nullptr ? sum_x_blocks[ib] : mwq_hsum256_ps(sum_x);
        acc += mn * sx + sc * mwq_hsum256_ps(sum_qx);
    }
    return acc;
}

static float mwq_dot_row_hier_avx2(
        const uint8_t * mwq,
        int bits,
        int block_size,
        int scale_group,
        int outlier_max,
        int64_t n_elem,
        int64_t row,
        int64_t n_col,
        const char * x,
        const float * sum_x_blocks) {
    const int64_t n_blocks_total = n_elem / block_size;
    const int64_t n_blocks_row = n_col / block_size;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t off_weights = 0;
    const size_t off_mins = off_weights + (size_t) n_blocks_total * qbytes;
    const size_t off_gscales = off_mins + (size_t) n_blocks_total * 2;
    const size_t n_scale_groups = ((size_t) n_blocks_total + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t off_scodes = off_gscales + n_scale_groups * 2;
    const size_t off_outliers = off_scodes + (size_t) n_blocks_total;
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 8;

    float acc = 0.0f;
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
        const float * xf = (const float *) (x + ib * block_size * sizeof(float));

        __m256 sum_x = _mm256_setzero_ps();
        __m256 sum_qx = _mm256_setzero_ps();
        for (int g = 0; g < groups; ++g) {
            const __m256 xv = _mm256_loadu_ps(xf + g * 8);
            const __m256 qv = bits == 2 ? mwq_q2_f32x8(q, g) : mwq_q3_f32x8(q, g);
            if (sum_x_blocks == nullptr) {
                sum_x = _mm256_add_ps(sum_x, xv);
            }
            sum_qx = _mm256_fmadd_ps(qv, xv, sum_qx);
        }

        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);
        const float sx = sum_x_blocks != nullptr ? sum_x_blocks[ib] : mwq_hsum256_ps(sum_x);
        acc += mn * sx + sc * mwq_hsum256_ps(sum_qx);

        const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        for (int j = 0; j < count; ++j) {
            const int idx = ob[1 + j * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + j * 3 + 1));
                acc += residual * *(const float *) (x + (ib * block_size + idx) * sizeof(float));
            }
        }
    }
    return acc;
}

static void mwq_dot_rows_avx2(
        const uint8_t * mwq,
        int             bits,
        int             block_size,
        int64_t         row,
        int             n_rows,
        int64_t         n_col,
        const char *    x,
        const float *   sum_x_blocks,
        float *         dst) {
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const int64_t n_blocks = n_col / block_size;
    const int groups = block_size / 8;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const float * xf = (const float *) (x + ib * block_size * sizeof(float));
        const float sx = sum_x_blocks != nullptr ? sum_x_blocks[ib] : 0.0f;

        __m256 sum_x = _mm256_setzero_ps();
        __m256 sum_qx0 = _mm256_setzero_ps();
        __m256 sum_qx1 = _mm256_setzero_ps();
        __m256 sum_qx2 = _mm256_setzero_ps();
        __m256 sum_qx3 = _mm256_setzero_ps();

        const uint8_t * block0 = mwq + (size_t) ((row + 0) * n_blocks + ib) * block_bytes;
        const uint8_t * block1 = n_rows > 1 ? mwq + (size_t) ((row + 1) * n_blocks + ib) * block_bytes : nullptr;
        const uint8_t * block2 = n_rows > 2 ? mwq + (size_t) ((row + 2) * n_blocks + ib) * block_bytes : nullptr;
        const uint8_t * block3 = n_rows > 3 ? mwq + (size_t) ((row + 3) * n_blocks + ib) * block_bytes : nullptr;

        const uint8_t * q0 = block0 + 4;
        const uint8_t * q1 = block1 != nullptr ? block1 + 4 : nullptr;
        const uint8_t * q2 = block2 != nullptr ? block2 + 4 : nullptr;
        const uint8_t * q3 = block3 != nullptr ? block3 + 4 : nullptr;

        for (int g = 0; g < groups; ++g) {
            const __m256 xv = _mm256_loadu_ps(xf + g * 8);
            if (sum_x_blocks == nullptr) {
                sum_x = _mm256_add_ps(sum_x, xv);
            }

            const __m256 qv0 = bits == 2 ? mwq_q2_f32x8(q0, g) : mwq_q3_f32x8(q0, g);
            sum_qx0 = _mm256_fmadd_ps(qv0, xv, sum_qx0);
            if (n_rows > 1) {
                const __m256 qv1 = bits == 2 ? mwq_q2_f32x8(q1, g) : mwq_q3_f32x8(q1, g);
                sum_qx1 = _mm256_fmadd_ps(qv1, xv, sum_qx1);
            }
            if (n_rows > 2) {
                const __m256 qv2 = bits == 2 ? mwq_q2_f32x8(q2, g) : mwq_q3_f32x8(q2, g);
                sum_qx2 = _mm256_fmadd_ps(qv2, xv, sum_qx2);
            }
            if (n_rows > 3) {
                const __m256 qv3 = bits == 2 ? mwq_q2_f32x8(q3, g) : mwq_q3_f32x8(q3, g);
                sum_qx3 = _mm256_fmadd_ps(qv3, xv, sum_qx3);
            }
        }

        const float sxv = sum_x_blocks != nullptr ? sx : mwq_hsum256_ps(sum_x);
        const float mn0 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block0 + 0));
        const float sc0 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block0 + 2));
        acc[0] += mn0 * sxv + sc0 * mwq_hsum256_ps(sum_qx0);
        if (n_rows > 1) {
            const float mn1 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block1 + 0));
            const float sc1 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block1 + 2));
            acc[1] += mn1 * sxv + sc1 * mwq_hsum256_ps(sum_qx1);
        }
        if (n_rows > 2) {
            const float mn2 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block2 + 0));
            const float sc2 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block2 + 2));
            acc[2] += mn2 * sxv + sc2 * mwq_hsum256_ps(sum_qx2);
        }
        if (n_rows > 3) {
            const float mn3 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block3 + 0));
            const float sc3 = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block3 + 2));
            acc[3] += mn3 * sxv + sc3 * mwq_hsum256_ps(sum_qx3);
        }
    }

    for (int i = 0; i < n_rows; ++i) {
        dst[row + i] = acc[i];
    }
}

static void mwq_dot_row_batch_avx2(
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float * const *       dsts) {
    static constexpr int MWQ_BATCH_MAX = 16;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const int64_t n_blocks = n_col / block_size;
    const int groups = block_size / 8;
    const int nb = std::min(batch, MWQ_BATCH_MAX);

    float acc[MWQ_BATCH_MAX] = {};
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const uint8_t * block = mwq + (size_t) (row * n_blocks + ib) * block_bytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 0));
        const float sc = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 2));
        const uint8_t * q = block + 4;

        __m256 sum_qx[MWQ_BATCH_MAX];
        for (int j = 0; j < nb; ++j) {
            sum_qx[j] = _mm256_setzero_ps();
        }
        for (int g = 0; g < groups; ++g) {
            const __m256 qv = bits == 2 ? mwq_q2_f32x8(q, g) : mwq_q3_f32x8(q, g);
            const size_t xoff = (size_t) (ib * block_size + g * 8) * sizeof(float);
            for (int j = 0; j < nb; ++j) {
                const __m256 xv = _mm256_loadu_ps((const float *) (xs[j] + xoff));
                sum_qx[j] = _mm256_fmadd_ps(qv, xv, sum_qx[j]);
            }
        }

        for (int j = 0; j < nb; ++j) {
            acc[j] += mn * sum_x_blocks[j][ib] + sc * mwq_hsum256_ps(sum_qx[j]);
        }
    }

    for (int i = 0; i < nb; ++i) {
        dsts[i][row] = acc[i];
    }
}

static void mwq_dot_row_batch_values_avx2(
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float *               out) {
    static constexpr int MWQ_BATCH_MAX = 16;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const int64_t n_blocks = n_col / block_size;
    const int groups = block_size / 8;
    const int nb = std::min(batch, MWQ_BATCH_MAX);

    float acc[MWQ_BATCH_MAX] = {};
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const uint8_t * block = mwq + (size_t) (row * n_blocks + ib) * block_bytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 0));
        const float sc = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(block + 2));
        const uint8_t * q = block + 4;

        __m256 sum_qx[MWQ_BATCH_MAX];
        for (int j = 0; j < nb; ++j) {
            sum_qx[j] = _mm256_setzero_ps();
        }
        for (int g = 0; g < groups; ++g) {
            const __m256 qv = bits == 2 ? mwq_q2_f32x8(q, g) : mwq_q3_f32x8(q, g);
            const size_t xoff = (size_t) (ib * block_size + g * 8) * sizeof(float);
            for (int j = 0; j < nb; ++j) {
                const __m256 xv = _mm256_loadu_ps((const float *) (xs[j] + xoff));
                sum_qx[j] = _mm256_fmadd_ps(qv, xv, sum_qx[j]);
            }
        }

        for (int j = 0; j < nb; ++j) {
            acc[j] += mn * sum_x_blocks[j][ib] + sc * mwq_hsum256_ps(sum_qx[j]);
        }
    }

    for (int i = 0; i < nb; ++i) {
        out[i] = acc[i];
    }
}

static void mwq_swiglu_row_batch_values_avx2(
        const uint8_t *       gate_mwq,
        const uint8_t *       up_mwq,
        int                   bits,
        int                   block_size,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float *               out) {
    static constexpr int MWQ_BATCH_MAX = 16;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const int64_t n_blocks = n_col / block_size;
    const int groups = block_size / 8;
    const int nb = std::min(batch, MWQ_BATCH_MAX);

    float gate_acc[MWQ_BATCH_MAX] = {};
    float up_acc[MWQ_BATCH_MAX] = {};
    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const uint8_t * gate_block = gate_mwq + (size_t) (row * n_blocks + ib) * block_bytes;
        const uint8_t * up_block   = up_mwq   + (size_t) (row * n_blocks + ib) * block_bytes;
        const float gate_mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_block + 0));
        const float gate_sc = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_block + 2));
        const float up_mn   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_block + 0));
        const float up_sc   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_block + 2));
        const uint8_t * gate_q = gate_block + 4;
        const uint8_t * up_q   = up_block + 4;

        __m256 gate_sum_qx[MWQ_BATCH_MAX];
        __m256 up_sum_qx[MWQ_BATCH_MAX];
        for (int j = 0; j < nb; ++j) {
            gate_sum_qx[j] = _mm256_setzero_ps();
            up_sum_qx[j] = _mm256_setzero_ps();
        }
        for (int g = 0; g < groups; ++g) {
            const __m256 gate_qv = bits == 2 ? mwq_q2_f32x8(gate_q, g) : mwq_q3_f32x8(gate_q, g);
            const __m256 up_qv   = bits == 2 ? mwq_q2_f32x8(up_q,   g) : mwq_q3_f32x8(up_q,   g);
            const size_t xoff = (size_t) (ib * block_size + g * 8) * sizeof(float);
            for (int j = 0; j < nb; ++j) {
                const __m256 xv = _mm256_loadu_ps((const float *) (xs[j] + xoff));
                gate_sum_qx[j] = _mm256_fmadd_ps(gate_qv, xv, gate_sum_qx[j]);
                up_sum_qx[j]   = _mm256_fmadd_ps(up_qv,   xv, up_sum_qx[j]);
            }
        }

        for (int j = 0; j < nb; ++j) {
            const float sx = sum_x_blocks[j][ib];
            gate_acc[j] += gate_mn * sx + gate_sc * mwq_hsum256_ps(gate_sum_qx[j]);
            up_acc[j]   += up_mn   * sx + up_sc   * mwq_hsum256_ps(up_sum_qx[j]);
        }
    }

    for (int i = 0; i < nb; ++i) {
        out[i] = moe_silu_f32(gate_acc[i]) * up_acc[i];
    }
}

static void mwq_dot_row_batch_hier_avx2(
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float * const *       dsts) {
    static constexpr int MWQ_BATCH_MAX = 16;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int64_t n_blocks_total = n_elem / block_size;
    const int64_t n_blocks_row = n_col / block_size;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t off_weights = 0;
    const size_t off_mins = off_weights + (size_t) n_blocks_total * qbytes;
    const size_t off_gscales = off_mins + (size_t) n_blocks_total * 2;
    const size_t n_scale_groups = ((size_t) n_blocks_total + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t off_scodes = off_gscales + n_scale_groups * 2;
    const size_t off_outliers = off_scodes + (size_t) n_blocks_total;
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 8;

    float acc[MWQ_BATCH_MAX] = {};
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
        __m256 sum_qx[MWQ_BATCH_MAX];
        for (int j = 0; j < nb; ++j) {
            sum_qx[j] = _mm256_setzero_ps();
        }

        for (int g = 0; g < groups; ++g) {
            const __m256 qv = bits == 2 ? mwq_q2_f32x8(q, g) : mwq_q3_f32x8(q, g);
            const size_t xoff = (size_t) (ib * block_size + g * 8) * sizeof(float);
            for (int j = 0; j < nb; ++j) {
                const __m256 xv = _mm256_loadu_ps((const float *) (xs[j] + xoff));
                sum_qx[j] = _mm256_fmadd_ps(qv, xv, sum_qx[j]);
            }
        }

        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);
        for (int j = 0; j < nb; ++j) {
            acc[j] += mn * sum_x_blocks[j][ib] + sc * mwq_hsum256_ps(sum_qx[j]);
        }

        const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        for (int oi = 0; oi < count; ++oi) {
            const int idx = ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                const size_t xoff = (size_t) (ib * block_size + idx) * sizeof(float);
                for (int j = 0; j < nb; ++j) {
                    acc[j] += residual * *(const float *) (xs[j] + xoff);
                }
            }
        }
    }

    for (int i = 0; i < nb; ++i) {
        dsts[i][row] = acc[i];
    }
}

static void mwq_dot_row_batch_values_hier_avx2(
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float *               out) {
    static constexpr int MWQ_BATCH_MAX = 16;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int64_t n_blocks_total = n_elem / block_size;
    const int64_t n_blocks_row = n_col / block_size;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t off_weights = 0;
    const size_t off_mins = off_weights + (size_t) n_blocks_total * qbytes;
    const size_t off_gscales = off_mins + (size_t) n_blocks_total * 2;
    const size_t n_scale_groups = ((size_t) n_blocks_total + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t off_scodes = off_gscales + n_scale_groups * 2;
    const size_t off_outliers = off_scodes + (size_t) n_blocks_total;
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 8;

    float acc[MWQ_BATCH_MAX] = {};
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
        __m256 sum_qx[MWQ_BATCH_MAX];
        for (int j = 0; j < nb; ++j) {
            sum_qx[j] = _mm256_setzero_ps();
        }

        for (int g = 0; g < groups; ++g) {
            const __m256 qv = bits == 2 ? mwq_q2_f32x8(q, g) : mwq_q3_f32x8(q, g);
            const size_t xoff = (size_t) (ib * block_size + g * 8) * sizeof(float);
            for (int j = 0; j < nb; ++j) {
                const __m256 xv = _mm256_loadu_ps((const float *) (xs[j] + xoff));
                sum_qx[j] = _mm256_fmadd_ps(qv, xv, sum_qx[j]);
            }
        }

        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);
        for (int j = 0; j < nb; ++j) {
            acc[j] += mn * sum_x_blocks[j][ib] + sc * mwq_hsum256_ps(sum_qx[j]);
        }

        const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        for (int oi = 0; oi < count; ++oi) {
            const int idx = ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                const size_t xoff = (size_t) (ib * block_size + idx) * sizeof(float);
                for (int j = 0; j < nb; ++j) {
                    acc[j] += residual * *(const float *) (xs[j] + xoff);
                }
            }
        }
    }

    for (int i = 0; i < nb; ++i) {
        out[i] = acc[i];
    }
}

static void mwq_swiglu_row_batch_values_hier_avx2(
        const uint8_t *       gate_mwq,
        const uint8_t *       up_mwq,
        int                   bits,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        float *               out) {
    static constexpr int MWQ_BATCH_MAX = 16;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int64_t n_blocks_total = n_elem / block_size;
    const int64_t n_blocks_row = n_col / block_size;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t off_weights = 0;
    const size_t off_mins = off_weights + (size_t) n_blocks_total * qbytes;
    const size_t off_gscales = off_mins + (size_t) n_blocks_total * 2;
    const size_t n_scale_groups = ((size_t) n_blocks_total + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t off_scodes = off_gscales + n_scale_groups * 2;
    const size_t off_outliers = off_scodes + (size_t) n_blocks_total;
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 8;

    float gate_acc[MWQ_BATCH_MAX] = {};
    float up_acc[MWQ_BATCH_MAX] = {};
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * gate_q = gate_mwq + off_weights + (size_t) gb * qbytes;
        const uint8_t * up_q   = up_mwq   + off_weights + (size_t) gb * qbytes;
        __m256 gate_sum_qx[MWQ_BATCH_MAX];
        __m256 up_sum_qx[MWQ_BATCH_MAX];
        for (int j = 0; j < nb; ++j) {
            gate_sum_qx[j] = _mm256_setzero_ps();
            up_sum_qx[j] = _mm256_setzero_ps();
        }

        for (int g = 0; g < groups; ++g) {
            const __m256 gate_qv = bits == 2 ? mwq_q2_f32x8(gate_q, g) : mwq_q3_f32x8(gate_q, g);
            const __m256 up_qv   = bits == 2 ? mwq_q2_f32x8(up_q,   g) : mwq_q3_f32x8(up_q,   g);
            const size_t xoff = (size_t) (ib * block_size + g * 8) * sizeof(float);
            for (int j = 0; j < nb; ++j) {
                const __m256 xv = _mm256_loadu_ps((const float *) (xs[j] + xoff));
                gate_sum_qx[j] = _mm256_fmadd_ps(gate_qv, xv, gate_sum_qx[j]);
                up_sum_qx[j]   = _mm256_fmadd_ps(up_qv,   xv, up_sum_qx[j]);
            }
        }

        const float gate_mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_mwq + off_mins + (size_t) gb * 2));
        const float up_mn   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_mwq   + off_mins + (size_t) gb * 2));
        const float gate_gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float up_gs   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_mwq   + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float gate_sc = gate_gs * (float) *(gate_mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);
        const float up_sc   = up_gs   * (float) *(up_mwq   + off_scodes + (size_t) gb) * (1.0f / 255.0f);
        for (int j = 0; j < nb; ++j) {
            const float sx = sum_x_blocks[j][ib];
            gate_acc[j] += gate_mn * sx + gate_sc * mwq_hsum256_ps(gate_sum_qx[j]);
            up_acc[j]   += up_mn   * sx + up_sc   * mwq_hsum256_ps(up_sum_qx[j]);
        }

        const uint8_t * gate_ob = gate_mwq + off_outliers + (size_t) gb * ostride;
        const int gate_count = std::min<int>((int) gate_ob[0], outlier_max);
        for (int oi = 0; oi < gate_count; ++oi) {
            const int idx = gate_ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_ob + 1 + oi * 3 + 1));
                const size_t xoff = (size_t) (ib * block_size + idx) * sizeof(float);
                for (int j = 0; j < nb; ++j) {
                    gate_acc[j] += residual * *(const float *) (xs[j] + xoff);
                }
            }
        }

        const uint8_t * up_ob = up_mwq + off_outliers + (size_t) gb * ostride;
        const int up_count = std::min<int>((int) up_ob[0], outlier_max);
        for (int oi = 0; oi < up_count; ++oi) {
            const int idx = up_ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_ob + 1 + oi * 3 + 1));
                const size_t xoff = (size_t) (ib * block_size + idx) * sizeof(float);
                for (int j = 0; j < nb; ++j) {
                    up_acc[j] += residual * *(const float *) (xs[j] + xoff);
                }
            }
        }
    }

    for (int i = 0; i < nb; ++i) {
        out[i] = moe_silu_f32(gate_acc[i]) * up_acc[i];
    }
}

#pragma GCC pop_options
#endif

#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx512vnni,avx2,fma")

static inline float mwq_hsum512_ps(__m512 v) {
    const __m256 lo = _mm512_castps512_ps256(v);
    const __m256 hi = _mm512_extractf32x8_ps(v, 1);
    return mwq_hsum256_ps(_mm256_add_ps(lo, hi));
}

static inline __m512 mwq_q2_f32x16_avx512(const uint8_t * q, int group) {
    const uint32_t p = rd_u32(q + group * 4);
    const uint64_t lo = MWQ_Q2_LUT8[p & 0xffffu];
    const uint64_t hi = MWQ_Q2_LUT8[(p >> 16) & 0xffffu];
    const __m128i qi8 = _mm_set_epi64x((long long) hi, (long long) lo);
    return _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(qi8));
}

static inline void mwq_prefetch_t0(const void * p) {
    _mm_prefetch((const char *) p, _MM_HINT_T0);
}

static inline void mwq_hier_q2_layout_unchecked(
        int64_t n_elem,
        int block_size,
        int scale_group,
        size_t & qbytes,
        size_t & off_weights,
        size_t & off_mins,
        size_t & off_gscales,
        size_t & off_scodes,
        size_t & off_outliers) {
    const int64_t n_blocks_total = n_elem / block_size;
    qbytes = (size_t) block_size / 4;
    off_weights = 0;
    off_mins = off_weights + (size_t) n_blocks_total * qbytes;
    off_gscales = off_mins + (size_t) n_blocks_total * 2;
    const size_t n_scale_groups = ((size_t) n_blocks_total + (size_t) scale_group - 1) / (size_t) scale_group;
    off_scodes = off_gscales + n_scale_groups * 2;
    off_outliers = off_scodes + (size_t) n_blocks_total;
}

static inline __m512 mwq_hier_q2_weight_f32x16(
        const uint8_t * q,
        int             group,
        float           mn,
        float           sc) {
    return _mm512_fmadd_ps(mwq_q2_f32x16_avx512(q, group), _mm512_set1_ps(sc), _mm512_set1_ps(mn));
}

static void mwq_qx_int8_build_avx512(
        const char * x,
        int64_t      n_col,
        int          qblock,
        int8_t *     qx,
        float *      scales,
        int32_t *    sums) {
    const int64_t n_blocks = n_col / qblock;
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512i lo = _mm512_set1_epi32(-127);
    const __m512i hi = _mm512_set1_epi32(127);
    alignas(64) int32_t tmp[16];

    for (int64_t ib = 0; ib < n_blocks; ++ib) {
        const float * xp = (const float *) (x + (size_t) ib * (size_t) qblock * sizeof(float));
        __m512 maxv = _mm512_setzero_ps();
        for (int i = 0; i < qblock; i += 16) {
            const __m512 xv = _mm512_loadu_ps(xp + i);
            maxv = _mm512_max_ps(maxv, _mm512_and_ps(xv, abs_mask));
        }
        const float amax = _mm512_reduce_max_ps(maxv);
        const float scale = std::isfinite(amax) && amax > 0.0f ? amax * (1.0f / 127.0f) : 0.0f;
        scales[(size_t) ib] = scale;

        int32_t sum = 0;
        int8_t * qp = qx + (size_t) ib * (size_t) qblock;
        if (scale == 0.0f) {
            std::memset(qp, 0, (size_t) qblock);
            sums[(size_t) ib] = 0;
            continue;
        }
        const __m512 inv = _mm512_set1_ps(1.0f / scale);
        for (int i = 0; i < qblock; i += 16) {
            const __m512 xv = _mm512_mul_ps(_mm512_loadu_ps(xp + i), inv);
            __m512i qi = _mm512_cvtps_epi32(xv);
            qi = _mm512_max_epi32(lo, _mm512_min_epi32(hi, qi));
            sum += _mm512_reduce_add_epi32(qi);
            _mm512_store_si512((__m512i *) tmp, qi);
            for (int k = 0; k < 16; ++k) {
                qp[i + k] = (int8_t) tmp[k];
            }
        }
        sums[(size_t) ib] = sum;
    }
}

static inline int32_t mwq_hsum512_epi32(__m512i v) {
    return _mm512_reduce_add_epi32(v);
}

static inline __m512i mwq_q2_u8x64_avx512(const uint8_t * q, int chunk64) {
    alignas(64) uint64_t expanded[8];
    const uint8_t * qc = q + (size_t) chunk64 * 16;
    for (int i = 0; i < 8; ++i) {
        expanded[i] = MWQ_Q2_LUT8[rd_u16(qc + i * 2)];
    }
    return _mm512_load_si512((const __m512i *) expanded);
}

static inline int32_t mwq_q2_vnni_dot64(const uint8_t * q, int chunk64, const int8_t * xq) {
    const __m512i wv = mwq_q2_u8x64_avx512(q, chunk64);
    const __m512i xv = _mm512_loadu_si512((const void *) xq);
    const __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv);
    return mwq_hsum512_epi32(acc);
}

static float mwq_dot_row_hier_q2_vnni_one(
        const uint8_t * mwq,
        int             block_size,
        int             scale_group,
        int             outlier_max,
        int64_t         n_elem,
        int64_t         row,
        int64_t         n_col,
        const char *    x,
        const mwq_qx_int8_cache_entry & qxent) {
    const int qblock = qxent.qblock;
    const int chunks_per_block = block_size / 64;
    const int64_t n_blocks_row = n_col / block_size;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(n_elem, block_size, scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);

    __m512 accv = _mm512_setzero_ps();
    float scalar_acc = 0.0f;
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);

        if (qblock == block_size) {
            const __m512 scale_v = _mm512_set1_ps(qxent.scales[(size_t) ib] * sc);
            for (int c = 0; c < chunks_per_block; ++c) {
                const int8_t * xq = qxent.qx.data() + (size_t) ib * (size_t) qblock + (size_t) c * 64;
                const __m512i dot = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), mwq_q2_u8x64_avx512(q, c), _mm512_loadu_si512((const void *) xq));
                accv = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), scale_v, accv);
            }
            const float sx = qxent.scales[(size_t) ib];
            const int32_t sum_xq = qxent.sums[(size_t) ib];
            scalar_acc += sx * mn * (float) sum_xq;
        } else {
            for (int c = 0; c < chunks_per_block; ++c) {
                const int64_t qb = (ib * (int64_t) block_size + (int64_t) c * 64) / qblock;
                const int off = (int) ((ib * (int64_t) block_size + (int64_t) c * 64) % qblock);
                const int8_t * xq = qxent.qx.data() + (size_t) qb * (size_t) qblock + (size_t) off;
                const float sx = qxent.scales[(size_t) qb];
                const __m512i dot = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), mwq_q2_u8x64_avx512(q, c), _mm512_loadu_si512((const void *) xq));
                accv = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), _mm512_set1_ps(sx * sc), accv);
                scalar_acc += sx * mn * (float) qxent.sums[(size_t) qb];
            }
        }

        const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        const size_t xbase = (size_t) ib * (size_t) block_size * sizeof(float);
        for (int oi = 0; oi < count; ++oi) {
            const int idx = ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                scalar_acc += residual * *(const float *) (x + xbase + (size_t) idx * sizeof(float));
            }
        }
    }
    return mwq_hsum512_ps(accv) + scalar_acc;
}

static void mwq_dot_rows_hier_q2_vnni(
        llama_moe_buffer_context * ctx,
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   n_rows,
        int64_t               n_col,
        const char *          x,
        float *               dst) {
    if (ctx == nullptr) {
        return;
    }
    const int qblock = mwq_vnni_qblock(*ctx, block_size);
    const mwq_qx_int8_cache_entry * qxent = mwq_qx_int8_cached(*ctx, x, n_col, qblock);
    if (qxent == nullptr) {
        return;
    }
    for (int r = 0; r < n_rows; ++r) {
        dst[row + r] = mwq_dot_row_hier_q2_vnni_one(mwq, block_size, scale_group, outlier_max,
                n_elem, row + r, n_col, x, *qxent);
    }
    static std::atomic<bool> checked{false};
    if (moe_vnni_check_enabled() && n_rows > 0 && !checked.exchange(true, std::memory_order_relaxed)) {
        const float ref = mwq_dot_row_hier_scalar(mwq, 2, block_size, scale_group, outlier_max, n_elem,
                row, n_col, x, sizeof(float), nullptr);
        const float got = dst[row];
        const double abs = std::fabs((double) ref - (double) got);
        const double rel = abs / std::max(1.0, std::fabs((double) ref));
        std::fprintf(stderr,
                "llama_moe_buffer[vnni-check]: kind=dot_rows row=%lld ref=% .9e vnni=% .9e abs=%.9e rel=%.9e qblock=%d\n",
                (long long) row, (double) ref, (double) got, abs, rel, qxent->qblock);
    }
}

static void mwq_dot_row_batch_hier_q2_vnni(
        llama_moe_buffer_context * ctx,
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        float * const *       dsts) {
    if (ctx == nullptr) {
        return;
    }
    static constexpr int MWQ_BATCH_MAX = 16;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int qblock = mwq_vnni_qblock(*ctx, block_size);
    if (qblock <= 0) {
        return;
    }
    tls_mwq_qx_int8_cache.reserve(tls_mwq_qx_int8_cache.size() + (size_t) nb);
    const mwq_qx_int8_cache_entry * qx[MWQ_BATCH_MAX] = {};
    for (int j = 0; j < nb; ++j) {
        qx[j] = mwq_qx_int8_cached(*ctx, xs[j], n_col, qblock);
        if (qx[j] == nullptr) {
            return;
        }
    }

    const int chunks_per_block = block_size / 64;
    const int64_t n_blocks_row = n_col / block_size;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(n_elem, block_size, scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);

    __m512 accv[MWQ_BATCH_MAX];
    float scalar_acc[MWQ_BATCH_MAX] = {};
    for (int j = 0; j < nb; ++j) {
        accv[j] = _mm512_setzero_ps();
    }
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);

        for (int c = 0; c < chunks_per_block; ++c) {
            const __m512i wv = mwq_q2_u8x64_avx512(q, c);
            for (int j = 0; j < nb; ++j) {
                const int64_t qb = qblock == block_size ?
                    ib : (ib * (int64_t) block_size + (int64_t) c * 64) / qblock;
                const int off = qblock == block_size ?
                    c * 64 : (int) ((ib * (int64_t) block_size + (int64_t) c * 64) % qblock);
                const int8_t * xq = qx[j]->qx.data() + (size_t) qb * (size_t) qblock + (size_t) off;
                const __m512i dot = _mm512_dpbusd_epi32(
                        _mm512_setzero_si512(), wv, _mm512_loadu_si512((const void *) xq));
                const float sx = qx[j]->scales[(size_t) qb];
                const int32_t sum_xq = qx[j]->sums[(size_t) qb];
                accv[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot), _mm512_set1_ps(sx * sc), accv[j]);
                if (qblock == block_size) {
                    if (c == chunks_per_block - 1) {
                        scalar_acc[j] += sx * mn * (float) sum_xq;
                    }
                } else {
                    scalar_acc[j] += sx * mn * (float) sum_xq;
                }
            }
        }

        const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        const size_t xbase = (size_t) ib * (size_t) block_size * sizeof(float);
        for (int oi = 0; oi < count; ++oi) {
            const int idx = ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                for (int j = 0; j < nb; ++j) {
                    scalar_acc[j] += residual * *(const float *) (xs[j] + xbase + (size_t) idx * sizeof(float));
                }
            }
        }
    }

    for (int j = 0; j < nb; ++j) {
        dsts[j][row] = mwq_hsum512_ps(accv[j]) + scalar_acc[j];
    }
    static std::atomic<bool> checked{false};
    if (moe_vnni_check_enabled() && nb > 0 && !checked.exchange(true, std::memory_order_relaxed)) {
        const float ref = mwq_dot_row_hier_scalar(mwq, 2, block_size, scale_group, outlier_max, n_elem,
                row, n_col, xs[0], sizeof(float), nullptr);
        const float got = dsts[0][row];
        const double abs = std::fabs((double) ref - (double) got);
        const double rel = abs / std::max(1.0, std::fabs((double) ref));
        std::fprintf(stderr,
                "llama_moe_buffer[vnni-check]: kind=dot_batch row=%lld ref=% .9e vnni=% .9e abs=%.9e rel=%.9e qblock=%d batch=%d\n",
                (long long) row, (double) ref, (double) got, abs, rel, qx[0]->qblock, nb);
    }
}

static void mwq_swiglu_row_batch_values_hier_q2_vnni(
        llama_moe_buffer_context * ctx,
        const uint8_t *       gate_mwq,
        const uint8_t *       up_mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        float *               out) {
    if (ctx == nullptr) {
        return;
    }
    static constexpr int MWQ_BATCH_MAX = 16;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int qblock = mwq_vnni_qblock(*ctx, block_size);
    if (qblock <= 0) {
        return;
    }
    tls_mwq_qx_int8_cache.reserve(tls_mwq_qx_int8_cache.size() + (size_t) nb);
    const mwq_qx_int8_cache_entry * qx[MWQ_BATCH_MAX] = {};
    for (int j = 0; j < nb; ++j) {
        qx[j] = mwq_qx_int8_cached(*ctx, xs[j], n_col, qblock);
        if (qx[j] == nullptr) {
            return;
        }
    }

    const int chunks_per_block = block_size / 64;
    const int64_t n_blocks_row = n_col / block_size;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(n_elem, block_size, scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);

    __m512 gate_accv[MWQ_BATCH_MAX];
    __m512 up_accv[MWQ_BATCH_MAX];
    float gate_scalar_acc[MWQ_BATCH_MAX] = {};
    float up_scalar_acc[MWQ_BATCH_MAX] = {};
    for (int j = 0; j < nb; ++j) {
        gate_accv[j] = _mm512_setzero_ps();
        up_accv[j] = _mm512_setzero_ps();
    }
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * gate_q = gate_mwq + off_weights + (size_t) gb * qbytes;
        const uint8_t * up_q   = up_mwq   + off_weights + (size_t) gb * qbytes;
        const float gate_mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_mwq + off_mins + (size_t) gb * 2));
        const float up_mn   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_mwq   + off_mins + (size_t) gb * 2));
        const float gate_gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float up_gs   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_mwq   + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float gate_sc = gate_gs * (float) *(gate_mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);
        const float up_sc   = up_gs   * (float) *(up_mwq   + off_scodes + (size_t) gb) * (1.0f / 255.0f);

        for (int c = 0; c < chunks_per_block; ++c) {
            const __m512i gate_wv = mwq_q2_u8x64_avx512(gate_q, c);
            const __m512i up_wv   = mwq_q2_u8x64_avx512(up_q, c);
            for (int j = 0; j < nb; ++j) {
                const int64_t qb = qblock == block_size ?
                    ib : (ib * (int64_t) block_size + (int64_t) c * 64) / qblock;
                const int off = qblock == block_size ?
                    c * 64 : (int) ((ib * (int64_t) block_size + (int64_t) c * 64) % qblock);
                const int8_t * xq = qx[j]->qx.data() + (size_t) qb * (size_t) qblock + (size_t) off;
                const __m512i xv = _mm512_loadu_si512((const void *) xq);
                const __m512i gate_dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), gate_wv, xv);
                const __m512i up_dot   = _mm512_dpbusd_epi32(_mm512_setzero_si512(), up_wv,   xv);
                const float sx = qx[j]->scales[(size_t) qb];
                const int32_t sum_xq = qblock == block_size && c != chunks_per_block - 1 ? 0 : qx[j]->sums[(size_t) qb];
                gate_accv[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(gate_dot), _mm512_set1_ps(sx * gate_sc), gate_accv[j]);
                up_accv[j]   = _mm512_fmadd_ps(_mm512_cvtepi32_ps(up_dot),   _mm512_set1_ps(sx * up_sc),   up_accv[j]);
                if (qblock == block_size) {
                    if (c == chunks_per_block - 1) {
                        gate_scalar_acc[j] += sx * gate_mn * (float) sum_xq;
                        up_scalar_acc[j]   += sx * up_mn   * (float) sum_xq;
                    }
                } else {
                    gate_scalar_acc[j] += sx * gate_mn * (float) sum_xq;
                    up_scalar_acc[j]   += sx * up_mn   * (float) sum_xq;
                }
            }
        }

        const size_t xbase = (size_t) ib * (size_t) block_size * sizeof(float);
        const uint8_t * gate_ob = gate_mwq + off_outliers + (size_t) gb * ostride;
        const int gate_count = std::min<int>((int) gate_ob[0], outlier_max);
        for (int oi = 0; oi < gate_count; ++oi) {
            const int idx = gate_ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_ob + 1 + oi * 3 + 1));
                for (int j = 0; j < nb; ++j) {
                    gate_scalar_acc[j] += residual * *(const float *) (xs[j] + xbase + (size_t) idx * sizeof(float));
                }
            }
        }

        const uint8_t * up_ob = up_mwq + off_outliers + (size_t) gb * ostride;
        const int up_count = std::min<int>((int) up_ob[0], outlier_max);
        for (int oi = 0; oi < up_count; ++oi) {
            const int idx = up_ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_ob + 1 + oi * 3 + 1));
                for (int j = 0; j < nb; ++j) {
                    up_scalar_acc[j] += residual * *(const float *) (xs[j] + xbase + (size_t) idx * sizeof(float));
                }
            }
        }
    }

    for (int j = 0; j < nb; ++j) {
        const float gate_v = mwq_hsum512_ps(gate_accv[j]) + gate_scalar_acc[j];
        const float up_v = mwq_hsum512_ps(up_accv[j]) + up_scalar_acc[j];
        out[j] = moe_silu_f32(gate_v) * up_v;
    }
    static std::atomic<bool> checked{false};
    if (moe_vnni_check_enabled() && nb > 0 && !checked.exchange(true, std::memory_order_relaxed)) {
        const float gate_ref = mwq_dot_row_hier_scalar(gate_mwq, 2, block_size, scale_group, outlier_max, n_elem,
                row, n_col, xs[0], sizeof(float), nullptr);
        const float up_ref = mwq_dot_row_hier_scalar(up_mwq, 2, block_size, scale_group, outlier_max, n_elem,
                row, n_col, xs[0], sizeof(float), nullptr);
        const float ref = moe_silu_f32(gate_ref) * up_ref;
        const double abs = std::fabs((double) ref - (double) out[0]);
        const double rel = abs / std::max(1.0, std::fabs((double) ref));
        std::fprintf(stderr,
                "llama_moe_buffer[vnni-check]: kind=swiglu row=%lld ref=% .9e vnni=% .9e abs=%.9e rel=%.9e qblock=%d batch=%d\n",
                (long long) row, (double) ref, (double) out[0], abs, rel, qx[0]->qblock, nb);
    }
}

static void mwq_dot_rows_hier_q2_avx512(
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   n_rows,
        int64_t               n_col,
        const char *          x,
        int                   prefetch_distance,
        float *               dst) {
    static constexpr int MWQ_ROW_TILE = 4;
    const int nr = std::min(n_rows, MWQ_ROW_TILE);
    const int64_t n_blocks_row = n_col / block_size;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(n_elem, block_size, scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 16;
    const int pf = std::max(0, prefetch_distance);

    __m512 accv[MWQ_ROW_TILE];
    float outlier_acc[MWQ_ROW_TILE] = {};
    for (int r = 0; r < nr; ++r) {
        accv[r] = _mm512_setzero_ps();
    }

    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const size_t xbase = (size_t) ib * (size_t) block_size * sizeof(float);
        if (pf > 0 && ib + pf < n_blocks_row) {
            mwq_prefetch_t0(x + ((size_t) (ib + pf) * (size_t) block_size) * sizeof(float));
        }

        float mn[MWQ_ROW_TILE] = {};
        float sc[MWQ_ROW_TILE] = {};
        for (int r = 0; r < nr; ++r) {
            const int64_t gb = (row + r) * n_blocks_row + ib;
            mn[r] = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
            const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
            sc[r] = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);
            if (pf > 0 && ib + pf < n_blocks_row) {
                const int64_t pgb = (row + r) * n_blocks_row + ib + pf;
                mwq_prefetch_t0(mwq + off_weights  + (size_t) pgb * qbytes);
                mwq_prefetch_t0(mwq + off_mins     + (size_t) pgb * 2);
                mwq_prefetch_t0(mwq + off_scodes   + (size_t) pgb);
                mwq_prefetch_t0(mwq + off_outliers + (size_t) pgb * ostride);
            }
        }

        for (int g = 0; g < groups; ++g) {
            const __m512 xv = _mm512_loadu_ps((const float *) (x + xbase + (size_t) g * 16 * sizeof(float)));
            for (int r = 0; r < nr; ++r) {
                const int64_t gb = (row + r) * n_blocks_row + ib;
                const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
                const __m512 wv = mwq_hier_q2_weight_f32x16(q, g, mn[r], sc[r]);
                accv[r] = _mm512_fmadd_ps(wv, xv, accv[r]);
            }
        }

        for (int r = 0; r < nr; ++r) {
            const int64_t gb = (row + r) * n_blocks_row + ib;
            const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
            const int count = std::min<int>((int) ob[0], outlier_max);
            for (int oi = 0; oi < count; ++oi) {
                const int idx = ob[1 + oi * 3];
                if (idx >= 0 && idx < block_size) {
                    const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                    outlier_acc[r] += residual * *(const float *) (x + xbase + (size_t) idx * sizeof(float));
                }
            }
        }
    }

    for (int r = 0; r < nr; ++r) {
        dst[row + r] = mwq_hsum512_ps(accv[r]) + outlier_acc[r];
    }
}

static void mwq_dot_row_batch_hier_q2_avx512(
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        int                   prefetch_distance,
        float * const *       dsts) {
    static constexpr int MWQ_BATCH_MAX = 16;
    static constexpr int MWQ_BATCH_TILE = 8;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int64_t n_blocks_row = n_col / block_size;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(n_elem, block_size, scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 16;
    const int pf = std::max(0, prefetch_distance);

    for (int jb = 0; jb < nb; jb += MWQ_BATCH_TILE) {
        const int tile = std::min(MWQ_BATCH_TILE, nb - jb);
        __m512 accv[MWQ_BATCH_TILE];
        float outlier_acc[MWQ_BATCH_TILE] = {};
        for (int j = 0; j < tile; ++j) {
            accv[j] = _mm512_setzero_ps();
        }

        for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
            const int64_t gb = row * n_blocks_row + ib;
            const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
            const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
            const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
            const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);

            if (pf > 0 && ib + pf < n_blocks_row) {
                const int64_t pgb = row * n_blocks_row + ib + pf;
                mwq_prefetch_t0(mwq + off_weights  + (size_t) pgb * qbytes);
                mwq_prefetch_t0(mwq + off_mins     + (size_t) pgb * 2);
                mwq_prefetch_t0(mwq + off_scodes   + (size_t) pgb);
                mwq_prefetch_t0(mwq + off_outliers + (size_t) pgb * ostride);
            }

            for (int g = 0; g < groups; ++g) {
                const __m512 wv = mwq_hier_q2_weight_f32x16(q, g, mn, sc);
                const size_t xoff = ((size_t) ib * (size_t) block_size + (size_t) g * 16) * sizeof(float);
                for (int j = 0; j < tile; ++j) {
                    const __m512 xv = _mm512_loadu_ps((const float *) (xs[jb + j] + xoff));
                    accv[j] = _mm512_fmadd_ps(wv, xv, accv[j]);
                }
            }

            const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
            const int count = std::min<int>((int) ob[0], outlier_max);
            for (int oi = 0; oi < count; ++oi) {
                const int idx = ob[1 + oi * 3];
                if (idx >= 0 && idx < block_size) {
                    const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                    const size_t xoff = ((size_t) ib * (size_t) block_size + (size_t) idx) * sizeof(float);
                    for (int j = 0; j < tile; ++j) {
                        outlier_acc[j] += residual * *(const float *) (xs[jb + j] + xoff);
                    }
                }
            }
        }

        for (int j = 0; j < tile; ++j) {
            dsts[jb + j][row] = mwq_hsum512_ps(accv[j]) + outlier_acc[j];
        }
    }
}

static float mwq_dot_row_range_hier_q2_avx512(
        const uint8_t *       mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int64_t               n_col,
        int64_t               col0,
        int                   col_count,
        const float *         x) {
    if (mwq == nullptr || x == nullptr || block_size <= 0 || scale_group <= 0 ||
            col_count <= 0 || (block_size % 16) != 0 ||
            (col0 % block_size) != 0 || (col_count % block_size) != 0 ||
            col0 < 0 || col0 + col_count > n_col) {
        return 0.0f;
    }

    const int64_t n_blocks_row = n_col / block_size;
    const int64_t ib0 = col0 / block_size;
    const int64_t nb = col_count / block_size;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(n_elem, block_size, scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 16;

    __m512 accv = _mm512_setzero_ps();
    float outlier_acc = 0.0f;
    for (int64_t jb = 0; jb < nb; ++jb) {
        const int64_t ib = ib0 + jb;
        const int64_t gb = row * n_blocks_row + ib;
        const uint8_t * q = mwq + off_weights + (size_t) gb * qbytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
        const float sc = gs * (float) *(mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);

        for (int g = 0; g < groups; ++g) {
            const __m512 wv = mwq_hier_q2_weight_f32x16(q, g, mn, sc);
            const __m512 xv = _mm512_loadu_ps(x + (size_t) jb * (size_t) block_size + (size_t) g * 16);
            accv = _mm512_fmadd_ps(wv, xv, accv);
        }

        const uint8_t * ob = mwq + off_outliers + (size_t) gb * ostride;
        const int count = std::min<int>((int) ob[0], outlier_max);
        for (int oi = 0; oi < count; ++oi) {
            const int idx = ob[1 + oi * 3];
            if (idx >= 0 && idx < block_size) {
                const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                outlier_acc += residual * x[(size_t) jb * (size_t) block_size + (size_t) idx];
            }
        }
    }
    return mwq_hsum512_ps(accv) + outlier_acc;
}

static void mwq_swiglu_row_batch_values_hier_q2_avx512(
        const uint8_t *       gate_mwq,
        const uint8_t *       up_mwq,
        int                   block_size,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        const float * const * sum_x_blocks,
        int                   prefetch_distance,
        float *               out) {
    static constexpr int MWQ_BATCH_MAX = 16;
    static constexpr int MWQ_BATCH_TILE = 8;
    (void) sum_x_blocks;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int64_t n_blocks_total = n_elem / block_size;
    const int64_t n_blocks_row = n_col / block_size;
    const size_t qbytes = (size_t) block_size / 4;
    const size_t off_weights = 0;
    const size_t off_mins = off_weights + (size_t) n_blocks_total * qbytes;
    const size_t off_gscales = off_mins + (size_t) n_blocks_total * 2;
    const size_t n_scale_groups = ((size_t) n_blocks_total + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t off_scodes = off_gscales + n_scale_groups * 2;
    const size_t off_outliers = off_scodes + (size_t) n_blocks_total;
    const size_t ostride = mwq_hier_outlier_stride(outlier_max);
    const int groups = block_size / 16;
    const int pf = std::max(0, prefetch_distance);

    for (int jb = 0; jb < nb; jb += MWQ_BATCH_TILE) {
        const int tile = std::min(MWQ_BATCH_TILE, nb - jb);
        __m512 gate_accv[MWQ_BATCH_TILE];
        __m512 up_accv[MWQ_BATCH_TILE];
        float gate_outlier_acc[MWQ_BATCH_TILE] = {};
        float up_outlier_acc[MWQ_BATCH_TILE] = {};
        for (int j = 0; j < tile; ++j) {
            gate_accv[j] = _mm512_setzero_ps();
            up_accv[j] = _mm512_setzero_ps();
        }

        for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
            const int64_t gb = row * n_blocks_row + ib;
            const uint8_t * gate_q = gate_mwq + off_weights + (size_t) gb * qbytes;
            const uint8_t * up_q   = up_mwq   + off_weights + (size_t) gb * qbytes;

            if (pf > 0 && ib + pf < n_blocks_row) {
                const int64_t pgb = row * n_blocks_row + ib + pf;
                mwq_prefetch_t0(gate_mwq + off_weights  + (size_t) pgb * qbytes);
                mwq_prefetch_t0(up_mwq   + off_weights  + (size_t) pgb * qbytes);
                mwq_prefetch_t0(gate_mwq + off_mins     + (size_t) pgb * 2);
                mwq_prefetch_t0(up_mwq   + off_mins     + (size_t) pgb * 2);
                mwq_prefetch_t0(gate_mwq + off_scodes   + (size_t) pgb);
                mwq_prefetch_t0(up_mwq   + off_scodes   + (size_t) pgb);
                mwq_prefetch_t0(gate_mwq + off_outliers + (size_t) pgb * ostride);
                mwq_prefetch_t0(up_mwq   + off_outliers + (size_t) pgb * ostride);
                if ((pgb % scale_group) == 0) {
                    mwq_prefetch_t0(gate_mwq + off_gscales + ((size_t) pgb / (size_t) scale_group) * 2);
                    mwq_prefetch_t0(up_mwq   + off_gscales + ((size_t) pgb / (size_t) scale_group) * 2);
                }
            }

            const float gate_mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_mwq + off_mins + (size_t) gb * 2));
            const float up_mn   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_mwq   + off_mins + (size_t) gb * 2));
            const float gate_gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_mwq + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
            const float up_gs   = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_mwq   + off_gscales + ((size_t) gb / (size_t) scale_group) * 2));
            const float gate_sc = gate_gs * (float) *(gate_mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);
            const float up_sc   = up_gs   * (float) *(up_mwq   + off_scodes + (size_t) gb) * (1.0f / 255.0f);
            const __m512 gate_mn_v = _mm512_set1_ps(gate_mn);
            const __m512 up_mn_v   = _mm512_set1_ps(up_mn);
            const __m512 gate_sc_v = _mm512_set1_ps(gate_sc);
            const __m512 up_sc_v   = _mm512_set1_ps(up_sc);

            for (int g = 0; g < groups; ++g) {
                const __m512 gate_wv = _mm512_fmadd_ps(mwq_q2_f32x16_avx512(gate_q, g), gate_sc_v, gate_mn_v);
                const __m512 up_wv   = _mm512_fmadd_ps(mwq_q2_f32x16_avx512(up_q,   g), up_sc_v,   up_mn_v);
                const size_t xoff = (size_t) (ib * block_size + g * 16) * sizeof(float);
                for (int j = 0; j < tile; ++j) {
                    const char * xp = xs[jb + j] + xoff;
                    if (pf > 0 && g + 2 < groups) {
                        mwq_prefetch_t0(xp + (size_t) 2 * 16 * sizeof(float));
                    }
                    const __m512 xv = _mm512_loadu_ps((const float *) xp);
                    gate_accv[j] = _mm512_fmadd_ps(gate_wv, xv, gate_accv[j]);
                    up_accv[j]   = _mm512_fmadd_ps(up_wv,   xv, up_accv[j]);
                }
            }

            const uint8_t * gate_ob = gate_mwq + off_outliers + (size_t) gb * ostride;
            const int gate_count = std::min<int>((int) gate_ob[0], outlier_max);
            for (int oi = 0; oi < gate_count; ++oi) {
                const int idx = gate_ob[1 + oi * 3];
                if (idx >= 0 && idx < block_size) {
                    const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(gate_ob + 1 + oi * 3 + 1));
                    const size_t xoff = (size_t) (ib * block_size + idx) * sizeof(float);
                    for (int j = 0; j < tile; ++j) {
                        gate_outlier_acc[j] += residual * *(const float *) (xs[jb + j] + xoff);
                    }
                }
            }

            const uint8_t * up_ob = up_mwq + off_outliers + (size_t) gb * ostride;
            const int up_count = std::min<int>((int) up_ob[0], outlier_max);
            for (int oi = 0; oi < up_count; ++oi) {
                const int idx = up_ob[1 + oi * 3];
                if (idx >= 0 && idx < block_size) {
                    const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(up_ob + 1 + oi * 3 + 1));
                    const size_t xoff = (size_t) (ib * block_size + idx) * sizeof(float);
                    for (int j = 0; j < tile; ++j) {
                        up_outlier_acc[j] += residual * *(const float *) (xs[jb + j] + xoff);
                    }
                }
            }
        }

        for (int j = 0; j < tile; ++j) {
            const float gate_v = mwq_hsum512_ps(gate_accv[j]) + gate_outlier_acc[j];
            const float up_v = mwq_hsum512_ps(up_accv[j]) + up_outlier_acc[j];
            out[jb + j] = moe_silu_f32(gate_v) * up_v;
        }
    }
}

#pragma GCC pop_options
#endif

static float mwq_dot_row(
        const uint8_t * mwq,
        int bits,
        int block_size,
        int codec,
        int scale_group,
        int outlier_max,
        int64_t n_elem,
        int64_t row,
        int64_t n_col,
        const char * x,
        size_t x_stride,
        const float * sum_x_blocks) {
    if (codec == MOE_SIDECAR_CODEC_MWQ_HIER) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
        if (block_size > 0 && (block_size % 8) == 0 && (bits == 2 || bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
            return mwq_dot_row_hier_avx2(mwq, bits, block_size, scale_group, outlier_max, n_elem,
                    row, n_col, x, sum_x_blocks);
        }
#endif
        return mwq_dot_row_hier_scalar(mwq, bits, block_size, scale_group, outlier_max, n_elem, row, n_col, x, x_stride, sum_x_blocks);
    }
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
    if (block_size > 0 && (block_size % 8) == 0 && (bits == 2 || bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
        return mwq_dot_row_avx2(mwq, bits, block_size, row, n_col, x, sum_x_blocks);
    }
#endif
    return mwq_dot_row_scalar(mwq, bits, block_size, row, n_col, x, x_stride, sum_x_blocks);
}

static void mwq_dot_rows(
        llama_moe_buffer_context * ctx,
        const uint8_t * mwq,
        int             bits,
        int             block_size,
        int             codec,
        int             scale_group,
        int             outlier_max,
        int64_t         n_elem,
        int64_t         row,
        int             n_rows,
        int64_t         n_col,
        const char *    x,
        size_t          x_stride,
        const float *   sum_x_blocks,
        float *         dst) {
    if (codec == MOE_SIDECAR_CODEC_MWQ_HIER) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
        if (ctx != nullptr && ctx->params.vnni_q2 && ctx->params.vnni_q2_down && bits == 2 &&
                block_size > 0 && (block_size % 64) == 0 && mwq_vnni_qblock(*ctx, block_size) > 0 &&
                x_stride == sizeof(float) && mwq_cpu_has_avx512_vnni_q2()) {
            mwq_dot_rows_hier_q2_vnni(ctx, mwq, block_size, scale_group, outlier_max, n_elem,
                    row, n_rows, n_col, x, dst);
            tls_mwq_vnni_q2_dot_calls++;
            tls_mwq_vnni_q2_dot_pairs += 1;
            tls_mwq_vnni_q2_dot_rows += (uint64_t) n_rows;
            return;
        }
        if (ctx != nullptr && ctx->params.avx512_q2 && ctx->params.avx512_q2_dot && bits == 2 &&
                block_size > 0 && (block_size % 16) == 0 && x_stride == sizeof(float) &&
                mwq_cpu_has_avx512_q2()) {
            mwq_dot_rows_hier_q2_avx512(mwq, block_size, scale_group, outlier_max, n_elem,
                    row, n_rows, n_col, x, ctx->params.avx512_prefetch, dst);
            tls_mwq_avx512_q2_dot_calls++;
            tls_mwq_avx512_q2_dot_pairs += 1;
            tls_mwq_avx512_q2_dot_rows += (uint64_t) n_rows;
            return;
        }
#endif
        for (int i = 0; i < n_rows; ++i) {
            dst[row + i] = mwq_dot_row(mwq, bits, block_size, codec, scale_group, outlier_max, n_elem,
                    row + i, n_col, x, x_stride, sum_x_blocks);
        }
        return;
    }
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
    if (block_size > 0 && (block_size % 8) == 0 && n_rows > 1 && (bits == 2 || bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
        mwq_dot_rows_avx2(mwq, bits, block_size, row, n_rows, n_col, x, sum_x_blocks, dst);
        return;
    }
#endif
    for (int i = 0; i < n_rows; ++i) {
        dst[row + i] = mwq_dot_row(mwq, bits, block_size, codec, scale_group, outlier_max, n_elem,
                row + i, n_col, x, x_stride, sum_x_blocks);
    }
}

static void mwq_dot_row_batch(
        llama_moe_buffer_context * ctx,
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int                   codec,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        size_t                x_stride,
        const float * const * sum_x_blocks,
        float * const *       dsts) {
    if (codec == MOE_SIDECAR_CODEC_MWQ_HIER) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
        if (ctx != nullptr && ctx->params.vnni_q2 && ctx->params.vnni_q2_down && bits == 2 &&
                block_size > 0 && (block_size % 64) == 0 && mwq_vnni_qblock(*ctx, block_size) > 0 &&
                batch > 1 && x_stride == sizeof(float) && mwq_cpu_has_avx512_vnni_q2()) {
            mwq_dot_row_batch_hier_q2_vnni(ctx, mwq, block_size, scale_group, outlier_max, n_elem,
                    row, batch, n_col, xs, dsts);
            tls_mwq_vnni_q2_dot_calls++;
            tls_mwq_vnni_q2_dot_pairs += (uint64_t) std::min(batch, 16);
            tls_mwq_vnni_q2_dot_rows += (uint64_t) std::min(batch, 16);
            return;
        }
        if (ctx != nullptr && ctx->params.avx512_q2 && ctx->params.avx512_q2_dot && bits == 2 &&
                block_size > 0 && (block_size % 16) == 0 && batch > 1 &&
                x_stride == sizeof(float) && mwq_cpu_has_avx512_q2()) {
            mwq_dot_row_batch_hier_q2_avx512(mwq, block_size, scale_group, outlier_max, n_elem,
                    row, batch, n_col, xs, ctx->params.avx512_prefetch, dsts);
            tls_mwq_avx512_q2_dot_calls++;
            tls_mwq_avx512_q2_dot_pairs += (uint64_t) std::min(batch, 16);
            tls_mwq_avx512_q2_dot_rows += (uint64_t) std::min(batch, 16);
            return;
        }
#endif
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
        if (block_size > 0 && (block_size % 8) == 0 && batch > 1 && (bits == 2 || bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
            mwq_dot_row_batch_hier_avx2(mwq, bits, block_size, scale_group, outlier_max, n_elem,
                    row, batch, n_col, xs, sum_x_blocks, dsts);
            return;
        }
#endif
        for (int i = 0; i < batch; ++i) {
            dsts[i][row] = mwq_dot_row(mwq, bits, block_size, codec, scale_group, outlier_max, n_elem,
                    row, n_col, xs[i], x_stride, sum_x_blocks[i]);
        }
        return;
    }
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
    if (block_size > 0 && (block_size % 8) == 0 && batch > 1 && (bits == 2 || bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
        mwq_dot_row_batch_avx2(mwq, bits, block_size, row, batch, n_col, xs, sum_x_blocks, dsts);
        return;
    }
#endif
    for (int i = 0; i < batch; ++i) {
        dsts[i][row] = mwq_dot_row(mwq, bits, block_size, codec, scale_group, outlier_max, n_elem,
                row, n_col, xs[i], x_stride, sum_x_blocks[i]);
    }
}

static void mwq_dot_row_batch_values(
        const uint8_t *       mwq,
        int                   bits,
        int                   block_size,
        int                   codec,
        int                   scale_group,
        int                   outlier_max,
        int64_t               n_elem,
        int64_t               row,
        int                   batch,
        int64_t               n_col,
        const char * const *  xs,
        size_t                x_stride,
        const float * const * sum_x_blocks,
        float *               out) {
    if (codec == MOE_SIDECAR_CODEC_MWQ_HIER) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
        if (block_size > 0 && (block_size % 8) == 0 && batch > 1 && (bits == 2 || bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
            mwq_dot_row_batch_values_hier_avx2(mwq, bits, block_size, scale_group, outlier_max, n_elem,
                    row, batch, n_col, xs, sum_x_blocks, out);
            return;
        }
#endif
        for (int i = 0; i < batch; ++i) {
            out[i] = mwq_dot_row(mwq, bits, block_size, codec, scale_group, outlier_max, n_elem,
                    row, n_col, xs[i], x_stride, sum_x_blocks[i]);
        }
        return;
    }
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
    if (block_size > 0 && (block_size % 8) == 0 && batch > 1 && (bits == 2 || bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
        mwq_dot_row_batch_values_avx2(mwq, bits, block_size, row, batch, n_col, xs, sum_x_blocks, out);
        return;
    }
#endif
    for (int i = 0; i < batch; ++i) {
        out[i] = mwq_dot_row(mwq, bits, block_size, codec, scale_group, outlier_max, n_elem,
                row, n_col, xs[i], x_stride, sum_x_blocks[i]);
    }
}

static bool mwq_swiglu_can_avx512_q2(
        llama_moe_buffer_context * ctx,
        const mwq_kernel_pair & gate,
        const mwq_kernel_pair & up,
        int                     batch,
        size_t                  x_stride) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
    return ctx != nullptr && ctx->params.avx512_q2 &&
        gate.backend == MOE_KERNEL_MWQ && up.backend == MOE_KERNEL_MWQ &&
        gate.bits == 2 && up.bits == 2 &&
        gate.block_size == up.block_size &&
        gate.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        up.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        gate.scale_group == up.scale_group &&
        gate.outlier_max == up.outlier_max &&
        gate.n_elem == up.n_elem &&
        gate.block_size > 0 && (gate.block_size % 16) == 0 &&
        batch > 1 && x_stride == sizeof(float) &&
        mwq_cpu_has_avx512_q2();
#else
    (void) ctx; (void) gate; (void) up; (void) batch; (void) x_stride;
    return false;
#endif
}

static bool mwq_swiglu_can_vnni_q2(
        llama_moe_buffer_context * ctx,
        const mwq_kernel_pair & gate,
        const mwq_kernel_pair & up,
        int                     batch,
        size_t                  x_stride) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
    return ctx != nullptr && ctx->params.vnni_q2 && ctx->params.vnni_q2_swiglu &&
        gate.backend == MOE_KERNEL_MWQ && up.backend == MOE_KERNEL_MWQ &&
        gate.bits == 2 && up.bits == 2 &&
        gate.block_size == up.block_size &&
        gate.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        up.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        gate.scale_group == up.scale_group &&
        gate.outlier_max == up.outlier_max &&
        gate.n_elem == up.n_elem &&
        gate.block_size > 0 && (gate.block_size % 64) == 0 &&
        mwq_vnni_qblock(*ctx, gate.block_size) > 0 &&
        batch > 1 && x_stride == sizeof(float) &&
        mwq_cpu_has_avx512_vnni_q2();
#else
    (void) ctx; (void) gate; (void) up; (void) batch; (void) x_stride;
    return false;
#endif
}

static bool mwq_dot_can_avx512_q2(
        llama_moe_buffer_context * ctx,
        const mwq_kernel_pair & item,
        int                     batch,
        size_t                  x_stride) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
    return ctx != nullptr && ctx->params.avx512_q2 && ctx->params.avx512_q2_dot &&
        item.backend == MOE_KERNEL_MWQ &&
        item.bits == 2 &&
        item.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        item.block_size > 0 && (item.block_size % 16) == 0 &&
        batch >= 1 && x_stride == sizeof(float) &&
        mwq_cpu_has_avx512_q2();
#else
    (void) ctx; (void) item; (void) batch; (void) x_stride;
    return false;
#endif
}

static bool mwq_dot_can_vnni_q2(
        llama_moe_buffer_context * ctx,
        const mwq_kernel_pair & item,
        int                     batch,
        size_t                  x_stride) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
    return ctx != nullptr && ctx->params.vnni_q2 && ctx->params.vnni_q2_down &&
        item.backend == MOE_KERNEL_MWQ &&
        item.bits == 2 &&
        item.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        item.block_size > 0 && (item.block_size % 64) == 0 &&
        mwq_vnni_qblock(*ctx, item.block_size) > 0 &&
        batch >= 1 && x_stride == sizeof(float) &&
        mwq_cpu_has_avx512_vnni_q2();
#else
    (void) ctx; (void) item; (void) batch; (void) x_stride;
    return false;
#endif
}

static void mwq_swiglu_row_batch_values(
        llama_moe_buffer_context * ctx,
        const mwq_kernel_pair & gate,
        const mwq_kernel_pair & up,
        int64_t                 row,
        int                     batch,
        int64_t                 n_col,
        const char * const *    xs,
        size_t                  x_stride,
        const float * const *   sum_x_blocks,
        float *                 out) {
    if (gate.backend == MOE_KERNEL_MWQ && up.backend == MOE_KERNEL_MWQ &&
            gate.bits == up.bits &&
            gate.block_size == up.block_size &&
            gate.codec == up.codec &&
            gate.scale_group == up.scale_group &&
            gate.outlier_max == up.outlier_max &&
            gate.n_elem == up.n_elem) {
        if (gate.codec == MOE_SIDECAR_CODEC_MWQ_HIER) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
            if (mwq_swiglu_can_vnni_q2(ctx, gate, up, batch, x_stride)) {
                mwq_swiglu_row_batch_values_hier_q2_vnni(ctx, gate.mwq, up.mwq, gate.block_size,
                        gate.scale_group, gate.outlier_max, gate.n_elem, row, batch, n_col, xs, out);
                tls_mwq_vnni_q2_swiglu_calls++;
                tls_mwq_vnni_q2_swiglu_pairs += (uint64_t) std::min(batch, 16);
                tls_mwq_vnni_q2_swiglu_rows += (uint64_t) std::min(batch, 16);
                return;
            }
            if (mwq_swiglu_can_avx512_q2(ctx, gate, up, batch, x_stride)) {
                mwq_swiglu_row_batch_values_hier_q2_avx512(gate.mwq, up.mwq, gate.block_size,
                        gate.scale_group, gate.outlier_max, gate.n_elem, row, batch, n_col, xs,
                        sum_x_blocks, ctx->params.avx512_prefetch, out);
                tls_mwq_avx512_q2_swiglu_calls++;
                tls_mwq_avx512_q2_swiglu_pairs += (uint64_t) std::min(batch, 16);
                tls_mwq_avx512_q2_swiglu_rows += (uint64_t) std::min(batch, 16);
                return;
            }
#endif
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
            if (gate.block_size > 0 && (gate.block_size % 8) == 0 && batch > 1 &&
                    (gate.bits == 2 || gate.bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
                mwq_swiglu_row_batch_values_hier_avx2(gate.mwq, up.mwq, gate.bits, gate.block_size,
                        gate.scale_group, gate.outlier_max, gate.n_elem, row, batch, n_col, xs, sum_x_blocks, out);
                return;
            }
#endif
        } else {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX2)
            if (gate.block_size > 0 && (gate.block_size % 8) == 0 && batch > 1 &&
                    (gate.bits == 2 || gate.bits == 3) && x_stride == sizeof(float) && mwq_cpu_has_avx2_fma()) {
                mwq_swiglu_row_batch_values_avx2(gate.mwq, up.mwq, gate.bits, gate.block_size,
                        row, batch, n_col, xs, sum_x_blocks, out);
                return;
            }
#endif
        }
    }

    float gate_vals[16] = {};
    float up_vals[16] = {};
    mwq_dot_row_batch_values(gate.mwq, gate.bits, gate.block_size, gate.codec, gate.scale_group,
            gate.outlier_max, gate.n_elem, row, batch, n_col, xs, x_stride, sum_x_blocks, gate_vals);
    mwq_dot_row_batch_values(up.mwq, up.bits, up.block_size, up.codec, up.scale_group,
            up.outlier_max, up.n_elem, row, batch, n_col, xs, x_stride, sum_x_blocks, up_vals);
    for (int i = 0; i < batch; ++i) {
        out[i] = moe_silu_f32(gate_vals[i]) * up_vals[i];
    }
}

#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
#pragma GCC push_options
#pragma GCC target("avx512f,avx512bw,avx512dq,avx2,fma")

static int moe_fused_ffn_row_tile(bool pipeline) {
    int tile = pipeline ? 16 : 128;
    if (const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_ROW_TILE")) {
        tile = std::atoi(v);
    }
    return std::max(1, std::min(pipeline ? 64 : 128, tile));
}

static bool moe_fused_ffn_fast_enabled() {
    const char * fast = std::getenv("LLAMA_LAZY_MOE_FFN_FAST");
    if (fast != nullptr && std::atoi(fast) > 0) {
        return true;
    }
    const char * pipe = std::getenv("LLAMA_LAZY_MOE_FFN_PIPELINE");
    return pipe != nullptr && std::atoi(pipe) > 0;
}

static bool moe_fused_ffn_serial_direct_enabled() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_SERIAL_DIRECT");
    return v != nullptr && std::atoi(v) > 0;
}

static int moe_fused_ffn_ring_slots() {
    int slots = 3;
    if (const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_BLOCK_RING")) {
        slots = std::atoi(v);
    }
    return std::max(1, std::min(4, slots));
}

static int moe_fused_ffn_producer_threads(int nth) {
    (void) nth;
    // Multiple producers need a per-slot ticket to preserve block order when
    // different future blocks alias the same ring slot. Keep the first pipeline
    // version single-producer so consumers can wait on block i in strict order.
    return 1;
}

static inline void moe_spin_pause() {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    _mm_pause();
#else
    std::this_thread::yield();
#endif
}

static std::shared_ptr<llama_moe_buffer_context::moe_ffn_pipe_entry> moe_ffn_pipe_acquire(
        llama_moe_buffer_context & ctx,
        const ggml_tensor *        op,
        size_t                     run_index,
        int                        nth,
        int                        batch,
        int                        block_size,
        int                        n_blocks,
        int                        ring,
        int                        consumer_threads) {
    std::lock_guard<std::mutex> lk(ctx.mtx);
    auto & per_op = ctx.ffn_pipe_map[op];
    auto it = per_op.find(run_index);
    if (it == per_op.end()) {
        auto ent = std::make_shared<llama_moe_buffer_context::moe_ffn_pipe_entry>();
        ent->batch = batch;
        ent->block_size = block_size;
        ent->n_blocks = n_blocks;
        ent->ring = ring;
        ent->consumer_threads = consumer_threads;
        ent->remaining.store(nth, std::memory_order_relaxed);
        ent->done_count.store(0, std::memory_order_relaxed);
        ent->can_leave.store(0, std::memory_order_relaxed);
        ent->data.resize((size_t) ring * (size_t) batch * (size_t) block_size);
        for (int s = 0; s < 4; ++s) {
            ent->slots[s].state.store(0, std::memory_order_relaxed);
            ent->slots[s].block.store(-1, std::memory_order_relaxed);
            ent->slots[s].consumers_left.store(0, std::memory_order_relaxed);
        }
        it = per_op.emplace(run_index, ent).first;
    }
    return it->second;
}

static void moe_ffn_pipe_release(
        llama_moe_buffer_context & ctx,
        const ggml_tensor *        op,
        size_t                     run_index,
        const std::shared_ptr<llama_moe_buffer_context::moe_ffn_pipe_entry> & ent) {
    if (!ent || ent->remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    std::lock_guard<std::mutex> lk(ctx.mtx);
    auto op_it = ctx.ffn_pipe_map.find(op);
    if (op_it == ctx.ffn_pipe_map.end()) {
        return;
    }
    op_it->second.erase(run_index);
    if (op_it->second.empty()) {
        ctx.ffn_pipe_map.erase(op_it);
    }
}

static void moe_ffn_op_barrier(llama_moe_buffer_context & ctx, const ggml_tensor * op, int nth) {
    if (op == nullptr || nth <= 1) {
        return;
    }
    std::unique_lock<std::mutex> lk(ctx.mtx);
    auto it = ctx.ffn_barrier_map.find(op);
    if (it == ctx.ffn_barrier_map.end()) {
        it = ctx.ffn_barrier_map.emplace(op, llama_moe_buffer_context::moe_thread_barrier_entry{}).first;
        it->second.leaving = nth;
    }
    llama_moe_buffer_context::moe_thread_barrier_entry & ent = it->second;
    ent.arrived++;
    if (ent.arrived >= nth) {
        ent.done = true;
        ctx.cv_ffn_barrier.notify_all();
    } else {
        while (!ent.done) {
            ctx.cv_ffn_barrier.wait(lk);
            it = ctx.ffn_barrier_map.find(op);
            if (it == ctx.ffn_barrier_map.end()) {
                return;
            }
        }
    }
    if (--ent.leaving == 0) {
        ctx.ffn_barrier_map.erase(op);
    }
}

static void mwq_ffn_build_swiglu_block_hier_q2_avx512(
        llama_moe_buffer_context * ctx,
        const mwq_ffn_triplet &    first,
        int                        batch,
        int64_t                    n_hidden,
        const char * const *       xs,
        const float * const *      sums,
        size_t                     x_stride,
        int64_t                    k0,
        float *                    tile) {
    static constexpr int MWQ_BATCH_MAX = 16;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int block_size = first.down.block_size;
    float vals[MWQ_BATCH_MAX] = {};
    for (int kk = 0; kk < block_size; ++kk) {
        mwq_swiglu_row_batch_values(ctx, first.gate, first.up, k0 + kk, nb,
                n_hidden, xs, x_stride, sums, vals);
        for (int j = 0; j < nb; ++j) {
            tile[(size_t) j * (size_t) block_size + (size_t) kk] = vals[j];
        }
    }
}

static void mwq_down_accum_swiglu_block_hier_q2_avx512(
        const mwq_ffn_triplet & first,
        int                     batch,
        int64_t                 n_ff,
        int64_t                 block_index,
        const float *           tile,
        float * const *         dsts,
        int64_t                 row0,
        int                     n_rows) {
    static constexpr int MWQ_BATCH_MAX = 16;
    static constexpr int MWQ_BATCH_TILE = 8;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int block_size = first.down.block_size;
    const int64_t n_blocks_row = n_ff / block_size;
    const int groups = block_size / 16;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(first.down.n_elem, block_size, first.down.scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(first.down.outlier_max);

    for (int r = 0; r < n_rows; ++r) {
        const int64_t row = row0 + r;
        const int64_t gb = row * n_blocks_row + block_index;
        const uint8_t * q = first.down.mwq + off_weights + (size_t) gb * qbytes;
        const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(first.down.mwq + off_mins + (size_t) gb * 2));
        const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(first.down.mwq + off_gscales + ((size_t) gb / (size_t) first.down.scale_group) * 2));
        const float sc = gs * (float) *(first.down.mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);

        for (int jb = 0; jb < nb; jb += MWQ_BATCH_TILE) {
            const int bt = std::min(MWQ_BATCH_TILE, nb - jb);
            __m512 accv[MWQ_BATCH_TILE];
            float scalar_acc[MWQ_BATCH_TILE] = {};
            for (int j = 0; j < bt; ++j) {
                accv[j] = _mm512_setzero_ps();
            }

            for (int g = 0; g < groups; ++g) {
                const __m512 wv = mwq_hier_q2_weight_f32x16(q, g, mn, sc);
                const size_t xoff = (size_t) g * 16;
                for (int j = 0; j < bt; ++j) {
                    const __m512 xv = _mm512_loadu_ps(tile + (size_t) (jb + j) * (size_t) block_size + xoff);
                    accv[j] = _mm512_fmadd_ps(wv, xv, accv[j]);
                }
            }

            const uint8_t * ob = first.down.mwq + off_outliers + (size_t) gb * ostride;
            const int count = std::min<int>((int) ob[0], first.down.outlier_max);
            for (int oi = 0; oi < count; ++oi) {
                const int idx = ob[1 + oi * 3];
                if (idx >= 0 && idx < block_size) {
                    const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                    for (int j = 0; j < bt; ++j) {
                        scalar_acc[j] += residual * tile[(size_t) (jb + j) * (size_t) block_size + (size_t) idx];
                    }
                }
            }

            for (int j = 0; j < bt; ++j) {
                dsts[jb + j][row] += mwq_hsum512_ps(accv[j]) + scalar_acc[j];
            }
        }
    }
}

static void mwq_ffn_run_rows_hier_q2_avx512(
        llama_moe_buffer_context * ctx,
        const mwq_ffn_triplet &    first,
        int                        batch,
        int64_t                    n_hidden,
        int64_t                    n_ff,
        const char * const *       xs,
        const float * const *      sums,
        float * const *            dsts,
        size_t                     x_stride,
        int64_t                    row0,
        int                        n_rows) {
    static constexpr int MWQ_BATCH_MAX = 16;
    static constexpr int MWQ_ROW_TILE_MAX = 128;
    const int nb = std::min(batch, MWQ_BATCH_MAX);
    const int nr = std::min(n_rows, MWQ_ROW_TILE_MAX);
    const int block_size = first.down.block_size;
    const int64_t n_blocks_row = n_ff / block_size;
    const int groups = block_size / 16;
    size_t qbytes = 0, off_weights = 0, off_mins = 0, off_gscales = 0, off_scodes = 0, off_outliers = 0;
    mwq_hier_q2_layout_unchecked(first.down.n_elem, block_size, first.down.scale_group, qbytes,
            off_weights, off_mins, off_gscales, off_scodes, off_outliers);
    const size_t ostride = mwq_hier_outlier_stride(first.down.outlier_max);

    tls_mwq_ffn_swiglu_tile.resize((size_t) nb * (size_t) block_size);
    __m512 accv[MWQ_ROW_TILE_MAX][MWQ_BATCH_MAX];
    float scalar_acc[MWQ_ROW_TILE_MAX][MWQ_BATCH_MAX] = {};
    for (int r = 0; r < nr; ++r) {
        for (int j = 0; j < nb; ++j) {
            accv[r][j] = _mm512_setzero_ps();
        }
    }

    float vals[MWQ_BATCH_MAX] = {};
    for (int64_t ib = 0; ib < n_blocks_row; ++ib) {
        const int64_t k0 = ib * block_size;
        for (int kk = 0; kk < block_size; ++kk) {
            mwq_swiglu_row_batch_values(ctx, first.gate, first.up, k0 + kk, nb,
                    n_hidden, xs, x_stride, sums, vals);
            for (int j = 0; j < nb; ++j) {
                tls_mwq_ffn_swiglu_tile[(size_t) j * (size_t) block_size + (size_t) kk] = vals[j];
            }
        }

        for (int r = 0; r < nr; ++r) {
            const int64_t gb = (row0 + r) * n_blocks_row + ib;
            const uint8_t * q = first.down.mwq + off_weights + (size_t) gb * qbytes;
            const float mn = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(first.down.mwq + off_mins + (size_t) gb * 2));
            const float gs = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(first.down.mwq + off_gscales + ((size_t) gb / (size_t) first.down.scale_group) * 2));
            const float sc = gs * (float) *(first.down.mwq + off_scodes + (size_t) gb) * (1.0f / 255.0f);

            for (int g = 0; g < groups; ++g) {
                const __m512 wv = mwq_hier_q2_weight_f32x16(q, g, mn, sc);
                const size_t xoff = (size_t) g * 16;
                for (int j = 0; j < nb; ++j) {
                    const __m512 xv = _mm512_loadu_ps(tls_mwq_ffn_swiglu_tile.data() + (size_t) j * (size_t) block_size + xoff);
                    accv[r][j] = _mm512_fmadd_ps(wv, xv, accv[r][j]);
                }
            }

            const uint8_t * ob = first.down.mwq + off_outliers + (size_t) gb * ostride;
            const int count = std::min<int>((int) ob[0], first.down.outlier_max);
            for (int oi = 0; oi < count; ++oi) {
                const int idx = ob[1 + oi * 3];
                if (idx >= 0 && idx < block_size) {
                    const float residual = ggml_fp16_to_fp32((ggml_fp16_t) rd_u16(ob + 1 + oi * 3 + 1));
                    for (int j = 0; j < nb; ++j) {
                        scalar_acc[r][j] += residual *
                            tls_mwq_ffn_swiglu_tile[(size_t) j * (size_t) block_size + (size_t) idx];
                    }
                }
            }
        }
    }

    for (int j = 0; j < nb; ++j) {
        for (int r = 0; r < nr; ++r) {
            dsts[j][row0 + r] = mwq_hsum512_ps(accv[r][j]) + scalar_acc[r][j];
        }
    }
}

static void moe_ffn_pipeline_check_once(
        llama_moe_buffer_context * ctx,
        const mwq_ffn_triplet &    first,
        int                        batch,
        int64_t                    n_hidden,
        int64_t                    n_ff,
        const char * const *       xs,
        const float * const *      sums,
        float * const *            dsts,
        size_t                     x_stride,
        int64_t                    row0,
        int                        n_rows) {
    static std::atomic<bool> checked{false};
    if (std::getenv("LLAMA_LAZY_MOE_FFN_CHECK") == nullptr ||
            checked.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    std::vector<float> ref((size_t) batch * (size_t) n_hidden, 0.0f);
    float * ref_dsts[16] = {};
    for (int j = 0; j < batch; ++j) {
        ref_dsts[j] = ref.data() + (size_t) j * (size_t) n_hidden;
    }
    mwq_ffn_run_rows_hier_q2_avx512(ctx, first, batch, n_hidden, n_ff,
            xs, sums, ref_dsts, x_stride, row0, n_rows);
    double max_abs = 0.0;
    double max_rel = 0.0;
    int max_j = -1;
    int64_t max_row = -1;
    for (int j = 0; j < batch; ++j) {
        for (int r = 0; r < n_rows; ++r) {
            const int64_t row = row0 + r;
            const double a = (double) dsts[j][row];
            const double b = (double) ref_dsts[j][row];
            const double abs = std::fabs(a - b);
            const double rel = abs / std::max(1.0, std::fabs(b));
            if (abs > max_abs) {
                max_abs = abs;
                max_rel = rel;
                max_j = j;
                max_row = row;
            }
        }
    }
    std::fprintf(stderr,
            "llama_moe_buffer[ffn-check]: row0=%lld n_rows=%d batch=%d max_abs=%.9e max_rel=%.9e max_row=%lld max_batch=%d pipe=% .9e ref=% .9e\n",
            (long long) row0, n_rows, batch, max_abs, max_rel, (long long) max_row, max_j,
            max_j >= 0 && max_row >= 0 ? (double) dsts[max_j][max_row] : 0.0,
            max_j >= 0 && max_row >= 0 ? (double) ref_dsts[max_j][max_row] : 0.0);
}

#pragma GCC pop_options
#endif

static bool moe_native_quantize_x(ggml_type type, const char * x, size_t x_stride, int64_t n_col, std::vector<uint8_t> & qx, ggml_type & qtype) {
    const ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(type);
    if (traits == nullptr || traits->vec_dot == nullptr || traits->vec_dot_type == GGML_TYPE_COUNT) {
        return false;
    }
    qtype = traits->vec_dot_type;
    const ggml_type_traits_cpu * qtraits = ggml_get_type_traits_cpu(qtype);
    if (qtraits == nullptr || qtraits->from_float == nullptr) {
        return false;
    }
    if (x_stride != sizeof(float)) {
        return false;
    }
    qx.resize(ggml_row_size(qtype, n_col));
    qtraits->from_float((const float *) x, qx.data(), n_col);
    return true;
}

static float moe_native_dot_row(
        ggml_type type,
        const uint8_t * native,
        size_t native_row_size,
        int64_t row,
        int64_t n_col,
        const uint8_t * qx) {
    const ggml_type_traits_cpu * traits = ggml_get_type_traits_cpu(type);
    float out = 0.0f;
    traits->vec_dot((int) n_col, &out, 0, native + (size_t) row * native_row_size, 0, qx, 0, 1);
    return out;
}

static void moe_native_dot_rows(
        ggml_type type,
        const uint8_t * native,
        size_t native_row_size,
        int64_t row,
        int n_rows,
        int64_t n_col,
        const uint8_t * qx,
        float * dst) {
    for (int i = 0; i < n_rows; ++i) {
        dst[row + i] = moe_native_dot_row(type, native, native_row_size, row + i, n_col, qx);
    }
}

static bool moe_mwq_check_enabled() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_MWQ_CHECK");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static int moe_mwq_check_rows() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_MWQ_CHECK_ROWS");
    if (v == nullptr || v[0] == '\0') {
        return 8;
    }
    return std::max(1, std::atoi(v));
}

static bool moe_swiglu_check_enabled() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_SWIGLU_CHECK");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static bool moe_ffn_oracle_enabled() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_ORACLE");
    return v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0;
}

static int moe_ffn_oracle_ops() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_ORACLE_OPS");
    return v == nullptr || v[0] == '\0' ? 1 : std::max(1, std::atoi(v));
}

static int moe_ffn_oracle_pairs() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_ORACLE_PAIRS");
    return v == nullptr || v[0] == '\0' ? 1 : std::max(1, std::atoi(v));
}

static int moe_ffn_oracle_ff_rows() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_ORACLE_FF_ROWS");
    return v == nullptr || v[0] == '\0' ? 8 : std::max(1, std::atoi(v));
}

static int moe_ffn_oracle_out_rows() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_ORACLE_OUT_ROWS");
    return v == nullptr || v[0] == '\0' ? 8 : std::max(1, std::atoi(v));
}

static double moe_ffn_oracle_tol() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_FFN_ORACLE_TOL");
    return v == nullptr || v[0] == '\0' ? 1.0e-4 : std::max(0.0, std::atof(v));
}

static float moe_f32_dot_row(const float * w, int64_t row, int64_t n_col, const char * x, size_t x_stride) {
    float acc = 0.0f;
    const float * wr = w + row * n_col;
    for (int64_t col = 0; col < n_col; ++col) {
        acc += wr[col] * *(const float *) (x + col * x_stride);
    }
    return acc;
}

static void moe_mwq_check_once(
        const llama_moe_buffer_context & ctx,
        const moe_managed &              m,
        const mwq_kernel_pair &          item,
        int                              requested_bits,
        int64_t                          n_col,
        int64_t                          n_row,
        size_t                           x_stride) {
    static std::atomic<bool> checked{false};
    if (!moe_mwq_check_enabled() || checked.exchange(true, std::memory_order_relaxed)) {
        return;
    }

    const moe_sidecar_entry * ent = moe_mwq_entry(ctx, m, item.expert, item.bits);
    const ggml_type_traits * traits = ggml_get_type_traits(m.type);
    if (ent == nullptr || traits == nullptr || traits->to_float == nullptr) {
        std::fprintf(stderr,
                "llama_moe_buffer[mwq-check]: unavailable metadata tensor=%s expert=%d bits=%d type=%s\n",
                m.name.c_str(), item.expert, item.bits, ggml_type_name(m.type));
        return;
    }

    std::vector<uint8_t> original_bytes(m.stride);
    int src_fd = -1;
#if defined(__linux__)
    char fd_path[64];
    std::snprintf(fd_path, sizeof(fd_path), "/proc/self/fd/%d", m.fd);
    src_fd = open(fd_path, O_RDONLY);
#endif
    const int read_fd = src_fd >= 0 ? src_fd : m.fd;
    const bool original_ok = read_full(read_fd, original_bytes.data(), original_bytes.size(),
            m.file_offset + (size_t) item.expert * m.stride);
    if (src_fd >= 0) {
        close(src_fd);
    }

    std::vector<float> original_f32((size_t) m.elems);
    std::vector<float> mwq_f32((size_t) m.elems);
    bool mwq_ok = false;
    if (original_ok) {
        traits->to_float(original_bytes.data(), original_f32.data(), m.elems);
    }
    if (ent->encoded_size == m.mwq.at(item.bits).stride && ent->codec == MOE_SIDECAR_CODEC_MWQ) {
        mwq_ok = mwq_decode_to_f32(item.mwq, (size_t) ent->encoded_size, item.bits, ent->block_size, mwq_f32.data(), m.elems);
    } else if (ent->encoded_size == m.mwq.at(item.bits).stride && ent->codec == MOE_SIDECAR_CODEC_MWQ_HIER) {
        mwq_ok = mwq_decode_hier_to_f32(item.mwq, (size_t) ent->encoded_size, item.bits, ent->block_size,
                ent->scale_group, ent->outlier_max, mwq_f32.data(), m.elems);
    }

    std::vector<float> sum_x;
    mwq_sum_x_blocks(item.x, x_stride, n_col, item.block_size, sum_x);

    const int rows = std::min<int64_t>(moe_mwq_check_rows(), n_row);
    std::fprintf(stderr,
            "llama_moe_buffer[mwq-check]: tensor=%s type=%s expert=%d requested_bits=%d loaded_bits=%d "
            "n_col=%lld n_row=%lld elems=%lld stride=%zu encoded=%llu x_stride=%zu original_ok=%d mwq_ok=%d\n",
            m.name.c_str(), ggml_type_name(m.type), item.expert, requested_bits, item.bits,
            (long long) n_col, (long long) n_row, (long long) m.elems, m.stride,
            (unsigned long long) ent->encoded_size, x_stride, original_ok ? 1 : 0, mwq_ok ? 1 : 0);

    double max_abs_orig_mwq = 0.0;
    double max_abs_mwq_direct = 0.0;
    double max_rel_mwq_direct = 0.0;
    for (int row = 0; row < rows; ++row) {
        const float direct = mwq_dot_row(item.mwq, item.bits, item.block_size, item.codec, item.scale_group,
                item.outlier_max, item.n_elem, row, n_col, item.x, x_stride, sum_x.data());
        const float mwq_dec = mwq_ok ? moe_f32_dot_row(mwq_f32.data(), row, n_col, item.x, x_stride) : 0.0f;
        const float orig = original_ok ? moe_f32_dot_row(original_f32.data(), row, n_col, item.x, x_stride) : 0.0f;
        const double abs_orig_mwq = std::fabs((double) orig - (double) mwq_dec);
        const double abs_mwq_direct = std::fabs((double) mwq_dec - (double) direct);
        const double rel_mwq_direct = abs_mwq_direct / std::max(1.0, std::fabs((double) mwq_dec));
        max_abs_orig_mwq = std::max(max_abs_orig_mwq, abs_orig_mwq);
        max_abs_mwq_direct = std::max(max_abs_mwq_direct, abs_mwq_direct);
        max_rel_mwq_direct = std::max(max_rel_mwq_direct, rel_mwq_direct);
        std::fprintf(stderr,
                "llama_moe_buffer[mwq-check]: row=%d orig=% .9e mwq_decode=% .9e direct=% .9e "
                "abs(orig-mwq)=%.9e abs(mwq-direct)=%.9e rel(mwq-direct)=%.9e\n",
                row, (double) orig, (double) mwq_dec, (double) direct,
                abs_orig_mwq, abs_mwq_direct, rel_mwq_direct);
    }
    std::fprintf(stderr,
            "llama_moe_buffer[mwq-check]: max_abs_orig_mwq=%.9e max_abs_mwq_direct=%.9e max_rel_mwq_direct=%.9e\n",
            max_abs_orig_mwq, max_abs_mwq_direct, max_rel_mwq_direct);
}

static bool moe_op_all_mwq_resident(const llama_moe_buffer_context & ctx, const moe_managed & m, const ggml_tensor * ids,
        moe_resident_fail_reason * reason = nullptr) {
    if (!moe_op_all_mwq_available(ctx, m, ids, reason)) {
        return false;
    }
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
            const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
            const int64_t flat = token * ids->ne[0] + rank;
            const int bits = moe_target_bits_for_id_index(ctx, m, expert, flat);
            if (expert < 0 || expert >= m.n_expert) {
                if (reason) *reason = moe_resident_fail_reason::RANGE;
                return false;
            }
            if (moe_mwq_resolve_loaded_bits(m, expert, bits) > 0) {
                continue;
            }
            if (moe_native_resident_compatible(ctx, m, expert, bits)) {
                continue;
            }
            if (reason) *reason = moe_resident_fail_reason::UNAVAILABLE;
            return false;
        }
    }
    return true;
}

static bool moe_name_has(const ggml_tensor * t, const char * needle);
static int moe_layer_from_name(const ggml_tensor * t);
static bool moe_dot_kernel_pair_prepare_native(
        const mwq_kernel_pair & item,
        size_t                  x_stride,
        int64_t                 n_col,
        std::vector<uint8_t> &  qx);
static float moe_dot_kernel_pair_row(
        const mwq_kernel_pair & item,
        int64_t                 row,
        int64_t                 n_col,
        size_t                  x_stride,
        const float *           sum_x,
        const std::vector<uint8_t> & qx);

static bool moe_make_kernel_pair(
        llama_moe_buffer_context & ctx,
        moe_managed &              m,
        ggml_tensor *              op,
        int64_t                    pair,
        int64_t                    n_col,
        mwq_kernel_pair &          item,
        int *                      requested_bits_out) {
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * ids  = op->src[2];
    const int64_t n_rank = ids->ne[0];
    const int64_t rank = pair % n_rank;
    const int64_t token = pair / n_rank;
    const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
    const int requested_bits = moe_target_bits_for_id_index(ctx, m, expert, pair);
    const int bits = moe_mwq_resolve_loaded_bits(m, expert, requested_bits);
    const bool use_native =
        expert >= 0 && expert < m.n_expert &&
        bits == 0 &&
        m.buf != nullptr &&
        moe_native_resident_compatible(ctx, m, expert, requested_bits);
    auto mit = m.mwq.find(bits);
    if (!use_native && (mit == m.mwq.end() || mit->second.buf == nullptr ||
            expert < 0 || expert >= m.n_expert ||
            expert >= (int) mit->second.loaded.size() ||
            !mit->second.loaded[expert])) {
        return false;
    }

    item = {};
    item.backend = use_native ? MOE_KERNEL_Q4K_NATIVE : MOE_KERNEL_MWQ;
    item.expert = expert;
    item.bits = use_native ? requested_bits : bits;
    item.block_size = use_native ? 0 : mit->second.block_size;
    item.codec = use_native ? 0 : mit->second.codec;
    item.scale_group = use_native ? 0 : mit->second.scale_group;
    item.outlier_max = use_native ? 0 : mit->second.outlier_max;
    item.pair = pair;
    item.token = token;
    item.rank = rank;
    item.n_elem = m.elems;
    item.mwq = use_native ? m.buf + (size_t) expert * m.stride : mit->second.buf + (size_t) expert * mit->second.stride;
    item.native_type = use_native ? m.type : GGML_TYPE_COUNT;
    item.native_row_size = use_native ? ggml_row_size(m.type, n_col) : 0;
    item.x = (const char *) src1->data + token * src1->nb[2] + (rank % src1->ne[1]) * src1->nb[1];
    item.dst = (float *) ((char *) op->data + rank * op->nb[1] + token * op->nb[2]);
    if (!use_native && (item.block_size <= 0 || n_col % item.block_size != 0)) {
        return false;
    }
    if (requested_bits_out != nullptr) {
        *requested_bits_out = requested_bits;
    }
    return true;
}

static float moe_tensor_f32_read_3d(const ggml_tensor * t, int64_t row, int64_t rank, int64_t token) {
    return *(const float *) ((const char *) t->data + row * t->nb[0] + rank * t->nb[1] + token * t->nb[2]);
}

static size_t moe_tensor_offset_3d(const ggml_tensor * t, int64_t row, int64_t rank, int64_t token) {
    return (size_t) (row * t->nb[0] + rank * t->nb[1] + token * t->nb[2]);
}

static void llama_moe_buffer_fused_ffn_oracle_check_swiglu(
        llama_moe_buffer_context * ctx,
        ggml_tensor *              swiglu_op,
        ggml_tensor *              gate_op,
        ggml_tensor *              up_op) {
    static std::atomic<int> checked_ops{0};
    if (!moe_ffn_oracle_enabled() || ctx == nullptr || swiglu_op == nullptr || gate_op == nullptr || up_op == nullptr ||
            checked_ops.load(std::memory_order_relaxed) >= moe_ffn_oracle_ops()) {
        return;
    }

    ggml_tensor * gate_w = gate_op->src[0];
    ggml_tensor * up_w   = up_op->src[0];
    ggml_tensor * src1   = gate_op->src[1];
    ggml_tensor * ids    = gate_op->src[2];
    if (gate_w == nullptr || up_w == nullptr || src1 == nullptr || ids == nullptr ||
            swiglu_op->op != GGML_OP_GLU ||
            gate_op->op != GGML_OP_MUL_MAT_ID ||
            up_op->op != GGML_OP_MUL_MAT_ID ||
            ggml_get_glu_op(swiglu_op) != GGML_GLU_OP_SWIGLU ||
            !moe_name_has(gate_w, ".ffn_gate_exps.weight") ||
            !moe_name_has(up_w, ".ffn_up_exps.weight") ||
            src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            swiglu_op->type != GGML_TYPE_F32 ||
            gate_w->ne[0] != up_w->ne[0] ||
            gate_w->ne[1] != up_w->ne[1]) {
        return;
    }

    const int op_index = checked_ops.fetch_add(1, std::memory_order_relaxed);
    if (op_index >= moe_ffn_oracle_ops()) {
        return;
    }

    auto git = ctx->by_name.find(ggml_get_name(gate_w));
    auto uit = ctx->by_name.find(ggml_get_name(up_w));
    if (git == ctx->by_name.end() || uit == ctx->by_name.end()) {
        std::fprintf(stderr, "llama_moe_buffer[ffn-oracle]: swiglu_op=%d missing gate/up metadata\n", op_index);
        return;
    }
    moe_managed & gate_m = git->second;
    moe_managed & up_m   = uit->second;

    const int layer = moe_layer_from_name(gate_w);
    const int64_t n_hidden = gate_w->ne[0];
    const int64_t n_ff = gate_w->ne[1];
    const int64_t n_rank = ids->ne[0];
    const int64_t n_token = ids->ne[1];
    const int64_t n_pair = n_rank * n_token;
    const size_t x_stride = (size_t) src1->nb[0];
    const int pair_limit = std::min<int64_t>(moe_ffn_oracle_pairs(), n_pair);
    const int ff_rows = std::min<int64_t>(moe_ffn_oracle_ff_rows(), n_ff);
    const double tol = moe_ffn_oracle_tol();
    const bool gu_ref_available = !(ctx->params.fuse_gate_up && ctx->params.fuse_direct_swiglu);

    std::fprintf(stderr,
            "llama_moe_buffer[ffn-oracle]: begin_swiglu op=%d layer=%d pairs=%lld/%lld ff_rows=%d/%lld "
            "gu_ref_available=%d tol=%.3e swiglu=%s\n",
            op_index, layer, (long long) pair_limit, (long long) n_pair,
            ff_rows, (long long) n_ff, gu_ref_available ? 1 : 0, tol, ggml_get_name(swiglu_op));

    double max_gate_abs = 0.0, max_up_abs = 0.0, max_swiglu_abs = 0.0;
    bool first_mismatch_printed = false;
    for (int64_t pair = 0; pair < pair_limit; ++pair) {
        const int64_t rank = pair % n_rank;
        const int64_t token = pair / n_rank;
        const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
        const int gate_bits = moe_target_bits_for_id_index(*ctx, gate_m, expert, pair);
        const int up_bits   = moe_target_bits_for_id_index(*ctx, up_m,   expert, pair);
        bool em = false, rf = false;
        moe_stream_mwq_slice(*ctx, gate_m, expert, true, gate_bits, (int) rank, &em, &rf);
        moe_stream_mwq_slice(*ctx, up_m,   expert, true, up_bits,   (int) rank, &em, &rf);

        mwq_kernel_pair gate_item;
        mwq_kernel_pair up_item;
        int gate_requested_bits = 0;
        int up_requested_bits = 0;
        if (!moe_make_kernel_pair(*ctx, gate_m, gate_op, pair, n_hidden, gate_item, &gate_requested_bits) ||
                !moe_make_kernel_pair(*ctx, up_m, up_op, pair, n_hidden, up_item, &up_requested_bits)) {
            std::fprintf(stderr,
                    "llama_moe_buffer[ffn-oracle]: stage=swiglu pair=%lld layer=%d expert=%d token=%lld rank=%lld "
                    "make_pair_failed requested_bits=(g%d,u%d) stream_missing=%d stream_read_fail=%d\n",
                    (long long) pair, layer, expert, (long long) token, (long long) rank,
                    gate_requested_bits, up_requested_bits, em ? 1 : 0, rf ? 1 : 0);
            continue;
        }

        const float * gate_sum_x = gate_item.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
            mwq_sum_x_blocks_cached(*ctx, gate_item.x, x_stride, n_hidden, gate_item.block_size);
        const float * up_sum_x = up_item.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
            mwq_sum_x_blocks_cached(*ctx, up_item.x, x_stride, n_hidden, up_item.block_size);
        std::vector<uint8_t> gate_qx;
        std::vector<uint8_t> up_qx;
        if (!moe_dot_kernel_pair_prepare_native(gate_item, x_stride, n_hidden, gate_qx) ||
                !moe_dot_kernel_pair_prepare_native(up_item, x_stride, n_hidden, up_qx)) {
            std::fprintf(stderr,
                    "llama_moe_buffer[ffn-oracle]: stage=swiglu pair=%lld layer=%d expert=%d token=%lld rank=%lld native_prepare_failed\n",
                    (long long) pair, layer, expert, (long long) token, (long long) rank);
            continue;
        }

        for (int64_t row = 0; row < ff_rows; ++row) {
            const float gate_v = moe_dot_kernel_pair_row(gate_item, row, n_hidden, x_stride, gate_sum_x, gate_qx);
            const float up_v   = moe_dot_kernel_pair_row(up_item,   row, n_hidden, x_stride, up_sum_x,   up_qx);
            const float fused_swiglu = moe_silu_f32(gate_v) * up_v;
            const float swiglu_ref = moe_tensor_f32_read_3d(swiglu_op, row, rank, token);
            const double swiglu_abs = std::fabs((double) fused_swiglu - (double) swiglu_ref);
            const double swiglu_rel = swiglu_abs / std::max(1.0, std::fabs((double) swiglu_ref));
            max_swiglu_abs = std::max(max_swiglu_abs, swiglu_abs);
            if (gu_ref_available) {
                const float gate_ref = moe_tensor_f32_read_3d(gate_op, row, rank, token);
                const float up_ref   = moe_tensor_f32_read_3d(up_op,   row, rank, token);
                const double gate_abs = std::fabs((double) gate_v - (double) gate_ref);
                const double up_abs   = std::fabs((double) up_v   - (double) up_ref);
                max_gate_abs = std::max(max_gate_abs, gate_abs);
                max_up_abs   = std::max(max_up_abs, up_abs);
                std::fprintf(stderr,
                        "llama_moe_buffer[ffn-oracle]: stage=gate_up layer=%d expert=%d token=%lld rank=%lld ff_row=%lld "
                        "gate_ref=% .9e gate_fused=% .9e gate_abs=%.9e up_ref=% .9e up_fused=% .9e up_abs=%.9e "
                        "gate_offset=%zu up_offset=%zu\n",
                        layer, expert, (long long) token, (long long) rank, (long long) row,
                        (double) gate_ref, (double) gate_v, gate_abs,
                        (double) up_ref, (double) up_v, up_abs,
                        moe_tensor_offset_3d(gate_op, row, rank, token),
                        moe_tensor_offset_3d(up_op, row, rank, token));
                if (!first_mismatch_printed && (gate_abs > tol || up_abs > tol)) {
                    first_mismatch_printed = true;
                    std::fprintf(stderr,
                            "llama_moe_buffer[ffn-oracle]: first_mismatch stage=gate_up layer=%d expert=%d token=%lld rank=%lld ff_row=%lld\n",
                            layer, expert, (long long) token, (long long) rank, (long long) row);
                }
            }
            std::fprintf(stderr,
                    "llama_moe_buffer[ffn-oracle]: stage=swiglu layer=%d expert=%d token=%lld rank=%lld ff_row=%lld "
                    "swiglu_ref=% .9e swiglu_fused=% .9e abs=%.9e rel=%.9e offset=%zu\n",
                    layer, expert, (long long) token, (long long) rank, (long long) row,
                    (double) swiglu_ref, (double) fused_swiglu, swiglu_abs, swiglu_rel,
                    moe_tensor_offset_3d(swiglu_op, row, rank, token));
            if (!first_mismatch_printed && swiglu_abs > tol) {
                first_mismatch_printed = true;
                std::fprintf(stderr,
                        "llama_moe_buffer[ffn-oracle]: first_mismatch stage=swiglu layer=%d expert=%d token=%lld rank=%lld ff_row=%lld\n",
                        layer, expert, (long long) token, (long long) rank, (long long) row);
            }
        }
    }

    std::fprintf(stderr,
            "llama_moe_buffer[ffn-oracle]: summary_swiglu op=%d layer=%d checked_pairs=%d max_abs={gate:%.9e,up:%.9e,swiglu:%.9e}\n",
            op_index, layer, pair_limit, max_gate_abs, max_up_abs, max_swiglu_abs);
}

static void llama_moe_buffer_fused_ffn_oracle_after_swiglu(
        llama_moe_buffer_context * ctx,
        ggml_tensor *              op,
        ggml_tensor *              gate_op,
        ggml_tensor *              up_op,
        int                        ith,
        int                        nth) {
    if (!moe_ffn_oracle_enabled() || ctx == nullptr || op == nullptr ||
            op->op != GGML_OP_GLU || !moe_name_has(op, "ffn_moe_swiglu")) {
        return;
    }
    moe_ffn_op_barrier(*ctx, op, nth);
    if (ith == 0) {
        llama_moe_buffer_fused_ffn_oracle_check_swiglu(ctx, op, gate_op, up_op);
    }
}

static void llama_moe_buffer_fused_ffn_oracle_check(
        llama_moe_buffer_context * ctx,
        ggml_tensor *              down_op,
        int                        nth) {
    static std::atomic<int> checked_ops{0};
    if (!moe_ffn_oracle_enabled() || ctx == nullptr || down_op == nullptr ||
            checked_ops.load(std::memory_order_relaxed) >= moe_ffn_oracle_ops()) {
        return;
    }

    ggml_tensor * swiglu_op = down_op->src[1];
    ggml_tensor * down_w = down_op->src[0];
    ggml_tensor * ids = down_op->src[2];
    ggml_tensor * gate_op = swiglu_op != nullptr ? swiglu_op->src[0] : nullptr;
    ggml_tensor * up_op   = swiglu_op != nullptr ? swiglu_op->src[1] : nullptr;
    ggml_tensor * gate_w  = gate_op != nullptr ? gate_op->src[0] : nullptr;
    ggml_tensor * up_w    = up_op   != nullptr ? up_op->src[0]   : nullptr;
    ggml_tensor * src1    = gate_op != nullptr ? gate_op->src[1] : nullptr;

    if (swiglu_op == nullptr || gate_op == nullptr || up_op == nullptr ||
            down_w == nullptr || gate_w == nullptr || up_w == nullptr ||
            src1 == nullptr || ids == nullptr ||
            down_op->op != GGML_OP_MUL_MAT_ID ||
            swiglu_op->op != GGML_OP_GLU ||
            gate_op->op != GGML_OP_MUL_MAT_ID ||
            up_op->op != GGML_OP_MUL_MAT_ID ||
            ggml_get_glu_op(swiglu_op) != GGML_GLU_OP_SWIGLU ||
            !moe_name_has(down_w, ".ffn_down_exps.weight") ||
            !moe_name_has(gate_w, ".ffn_gate_exps.weight") ||
            !moe_name_has(up_w, ".ffn_up_exps.weight") ||
            src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            down_op->type != GGML_TYPE_F32 || swiglu_op->type != GGML_TYPE_F32 ||
            down_op->nb[0] != (int64_t) sizeof(float) ||
            src1->ne[0] != gate_w->ne[0] ||
            gate_w->ne[0] != up_w->ne[0] ||
            gate_w->ne[1] != up_w->ne[1] ||
            gate_w->ne[1] != down_w->ne[0] ||
            down_w->ne[1] != gate_w->ne[0]) {
        return;
    }

    const int op_index = checked_ops.fetch_add(1, std::memory_order_relaxed);
    if (op_index >= moe_ffn_oracle_ops()) {
        return;
    }

    auto dit = ctx->by_name.find(ggml_get_name(down_w));
    auto git = ctx->by_name.find(ggml_get_name(gate_w));
    auto uit = ctx->by_name.find(ggml_get_name(up_w));
    if (dit == ctx->by_name.end() || git == ctx->by_name.end() || uit == ctx->by_name.end()) {
        std::fprintf(stderr, "llama_moe_buffer[ffn-oracle]: op=%d missing managed tensor metadata\n", op_index);
        return;
    }
    moe_managed & down_m = dit->second;
    moe_managed & gate_m = git->second;
    moe_managed & up_m   = uit->second;

    const int layer = moe_layer_from_name(down_w);
    const int64_t n_hidden = gate_w->ne[0];
    const int64_t n_ff = down_w->ne[0];
    const int64_t n_rank = ids->ne[0];
    const int64_t n_token = ids->ne[1];
    const int64_t n_pair = n_rank * n_token;
    const size_t x_stride = (size_t) src1->nb[0];
    const int pair_limit = std::min<int64_t>(moe_ffn_oracle_pairs(), n_pair);
    const int ff_rows = std::min<int64_t>(moe_ffn_oracle_ff_rows(), n_ff);
    const int out_rows = std::min<int64_t>(moe_ffn_oracle_out_rows(), down_op->ne[0]);
    const double tol = moe_ffn_oracle_tol();
    // At the down op, graph-planner memory reuse may already have recycled the
    // gate/up buffers. Gate/up are checked at the SWIGLU op boundary instead.
    const bool gu_ref_available = false;

    std::fprintf(stderr,
            "llama_moe_buffer[ffn-oracle]: begin op=%d layer=%d pairs=%lld/%lld ff_rows=%d/%lld out_rows=%d/%lld "
            "gu_ref_available=%d tol=%.3e down=%s swiglu=%s\n",
            op_index, layer, (long long) pair_limit, (long long) n_pair,
            ff_rows, (long long) n_ff, out_rows, (long long) down_op->ne[0],
            gu_ref_available ? 1 : 0, tol, ggml_get_name(down_op), ggml_get_name(swiglu_op));

    double max_gate_abs = 0.0, max_up_abs = 0.0, max_swiglu_abs = 0.0, max_down_abs = 0.0;
    bool first_mismatch_printed = false;

    for (int64_t pair = 0; pair < pair_limit; ++pair) {
        const int64_t rank = pair % n_rank;
        const int64_t token = pair / n_rank;
        const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
        const int gate_bits = moe_target_bits_for_id_index(*ctx, gate_m, expert, pair);
        const int up_bits   = moe_target_bits_for_id_index(*ctx, up_m,   expert, pair);
        const int down_bits = moe_target_bits_for_id_index(*ctx, down_m, expert, pair);
        bool em = false, rf = false;
        moe_stream_mwq_slice(*ctx, gate_m, expert, true, gate_bits, (int) rank, &em, &rf);
        moe_stream_mwq_slice(*ctx, up_m,   expert, true, up_bits,   (int) rank, &em, &rf);
        moe_stream_mwq_slice(*ctx, down_m, expert, true, down_bits, (int) rank, &em, &rf);

        mwq_ffn_triplet item;
        int gate_requested_bits = 0;
        int up_requested_bits = 0;
        int down_requested_bits = 0;
        if (!moe_make_kernel_pair(*ctx, gate_m, gate_op, pair, n_hidden, item.gate, &gate_requested_bits) ||
                !moe_make_kernel_pair(*ctx, up_m, up_op, pair, n_hidden, item.up, &up_requested_bits) ||
                !moe_make_kernel_pair(*ctx, down_m, down_op, pair, n_ff, item.down, &down_requested_bits)) {
            std::fprintf(stderr,
                    "llama_moe_buffer[ffn-oracle]: pair=%lld layer=%d expert=%d token=%lld rank=%lld make_pair_failed "
                    "requested_bits=(g%d,u%d,d%d) stream_missing=%d stream_read_fail=%d\n",
                    (long long) pair, layer, expert, (long long) token, (long long) rank,
                    gate_requested_bits, up_requested_bits, down_requested_bits, em ? 1 : 0, rf ? 1 : 0);
            continue;
        }

        const float * gate_sum_x = item.gate.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
            mwq_sum_x_blocks_cached(*ctx, item.gate.x, x_stride, n_hidden, item.gate.block_size);
        const float * up_sum_x = item.up.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
            mwq_sum_x_blocks_cached(*ctx, item.up.x, x_stride, n_hidden, item.up.block_size);

        std::vector<uint8_t> gate_qx;
        std::vector<uint8_t> up_qx;
        if (!moe_dot_kernel_pair_prepare_native(item.gate, x_stride, n_hidden, gate_qx) ||
                !moe_dot_kernel_pair_prepare_native(item.up, x_stride, n_hidden, up_qx)) {
            std::fprintf(stderr,
                    "llama_moe_buffer[ffn-oracle]: pair=%lld layer=%d expert=%d token=%lld rank=%lld native_prepare_failed\n",
                    (long long) pair, layer, expert, (long long) token, (long long) rank);
            continue;
        }

        std::vector<float> swiglu((size_t) n_ff);
        for (int64_t row = 0; row < n_ff; ++row) {
            const float gate_v = moe_dot_kernel_pair_row(item.gate, row, n_hidden, x_stride, gate_sum_x, gate_qx);
            const float up_v   = moe_dot_kernel_pair_row(item.up,   row, n_hidden, x_stride, up_sum_x,   up_qx);
            swiglu[(size_t) row] = moe_silu_f32(gate_v) * up_v;

            if (row < ff_rows) {
                const float swiglu_ref = moe_tensor_f32_read_3d(swiglu_op, row, rank, token);
                const double swiglu_abs = std::fabs((double) swiglu[(size_t) row] - (double) swiglu_ref);
                max_swiglu_abs = std::max(max_swiglu_abs, swiglu_abs);
                double gate_abs = 0.0;
                double up_abs = 0.0;
                if (gu_ref_available) {
                    const float gate_ref = moe_tensor_f32_read_3d(gate_op, row, rank, token);
                    const float up_ref   = moe_tensor_f32_read_3d(up_op,   row, rank, token);
                    gate_abs = std::fabs((double) gate_v - (double) gate_ref);
                    up_abs   = std::fabs((double) up_v   - (double) up_ref);
                    max_gate_abs = std::max(max_gate_abs, gate_abs);
                    max_up_abs   = std::max(max_up_abs, up_abs);
                    std::fprintf(stderr,
                            "llama_moe_buffer[ffn-oracle]: stage=gu layer=%d expert=%d token=%lld rank=%lld ff_row=%lld "
                            "gate_ref=% .9e gate_fused=% .9e gate_abs=%.9e up_ref=% .9e up_fused=% .9e up_abs=%.9e\n",
                            layer, expert, (long long) token, (long long) rank, (long long) row,
                            (double) gate_ref, (double) gate_v, gate_abs,
                            (double) up_ref, (double) up_v, up_abs);
                }
                std::fprintf(stderr,
                        "llama_moe_buffer[ffn-oracle]: stage=swiglu layer=%d expert=%d token=%lld rank=%lld ff_row=%lld "
                        "swiglu_ref=% .9e swiglu_fused=% .9e abs=%.9e rel=%.9e offset=%zu\n",
                        layer, expert, (long long) token, (long long) rank, (long long) row,
                        (double) swiglu_ref, (double) swiglu[(size_t) row], swiglu_abs,
                        swiglu_abs / std::max(1.0, std::fabs((double) swiglu_ref)),
                        moe_tensor_offset_3d(swiglu_op, row, rank, token));
                if (!first_mismatch_printed && (swiglu_abs > tol || gate_abs > tol || up_abs > tol)) {
                    first_mismatch_printed = true;
                    std::fprintf(stderr,
                            "llama_moe_buffer[ffn-oracle]: first_mismatch stage=swiglu layer=%d expert=%d token=%lld rank=%lld ff_row=%lld\n",
                            layer, expert, (long long) token, (long long) rank, (long long) row);
                }
            }
        }

        std::vector<float> down_sum_x;
        const float * down_sum = nullptr;
        if (item.down.backend != MOE_KERNEL_Q4K_NATIVE) {
            mwq_sum_x_blocks((const char *) swiglu.data(), sizeof(float), n_ff, item.down.block_size, down_sum_x);
            down_sum = down_sum_x.data();
        }
        std::vector<uint8_t> down_qx;
        mwq_kernel_pair down_item = item.down;
        down_item.x = (const char *) swiglu.data();
        if (!moe_dot_kernel_pair_prepare_native(down_item, sizeof(float), n_ff, down_qx)) {
            std::fprintf(stderr,
                    "llama_moe_buffer[ffn-oracle]: pair=%lld layer=%d expert=%d token=%lld rank=%lld down_native_prepare_failed\n",
                    (long long) pair, layer, expert, (long long) token, (long long) rank);
            continue;
        }
        for (int64_t irow = 0; irow < out_rows; ++irow) {
            const int64_t row = irow * std::max(1, nth);
            if (row >= down_op->ne[0]) {
                break;
            }
            const float fused = moe_dot_kernel_pair_row(down_item, row, n_ff, sizeof(float), down_sum, down_qx);
            const float ref = moe_tensor_f32_read_3d(down_op, row, rank, token);
            const double abs = std::fabs((double) fused - (double) ref);
            const double rel = abs / std::max(1.0, std::fabs((double) ref));
            max_down_abs = std::max(max_down_abs, abs);
            std::fprintf(stderr,
                    "llama_moe_buffer[ffn-oracle]: stage=down layer=%d expert=%d token=%lld rank=%lld out_row=%lld "
                    "down_ref=% .9e down_fused=% .9e abs=%.9e rel=%.9e dst_offset=%zu\n",
                    layer, expert, (long long) token, (long long) rank, (long long) row,
                    (double) ref, (double) fused, abs, rel,
                    moe_tensor_offset_3d(down_op, row, rank, token));
            if (!first_mismatch_printed && abs > tol) {
                first_mismatch_printed = true;
                std::fprintf(stderr,
                        "llama_moe_buffer[ffn-oracle]: first_mismatch stage=down layer=%d expert=%d token=%lld rank=%lld out_row=%lld\n",
                        layer, expert, (long long) token, (long long) rank, (long long) row);
            }
        }
    }

    std::fprintf(stderr,
            "llama_moe_buffer[ffn-oracle]: summary op=%d layer=%d checked_pairs=%d max_abs={gate:%.9e,up:%.9e,swiglu:%.9e,down:%.9e}\n",
            op_index, layer, pair_limit, max_gate_abs, max_up_abs, max_swiglu_abs, max_down_abs);
}

static void llama_moe_buffer_fused_ffn_oracle_after_down(
        llama_moe_buffer_context * ctx,
        ggml_tensor *              op,
        int                        ith,
        int                        nth) {
    if (!moe_ffn_oracle_enabled() || ctx == nullptr || op == nullptr ||
            op->op != GGML_OP_MUL_MAT_ID || op->src[0] == nullptr ||
            !moe_name_has(op->src[0], ".ffn_down_exps.weight")) {
        return;
    }
    if (ith == 0) {
        llama_moe_buffer_fused_ffn_oracle_check(ctx, op, nth);
    }
}

static bool llama_moe_buffer_mul_mat_id_compute(ggml_tensor * op, int ith, int nth, void * user_data) {
    auto * ctx = static_cast<llama_moe_buffer_context *>(user_data);
    if (ctx == nullptr || op == nullptr || op->op != GGML_OP_MUL_MAT_ID) {
        return false;
    }

    ggml_tensor * src0 = op->src[0];
    ggml_tensor * src1 = op->src[1];
    ggml_tensor * ids  = op->src[2];
    if (src0 == nullptr || src1 == nullptr || ids == nullptr || src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32) {
        return false;
    }

    auto it = ctx->by_name.find(ggml_get_name(src0));
    if (it == ctx->by_name.end()) {
        return false;
    }
    moe_managed & m = it->second;
    if (!moe_op_all_mwq_resident(*ctx, m, ids)) {
        return false;
    }

    const int64_t n_col = src0->ne[0];
    const int64_t n_row = src0->ne[1];
    const int64_t n_rank = ids->ne[0];
    const int64_t n_token = ids->ne[1];
    if (src1->ne[0] != n_col || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (ith == 0) {
        ctx->mwq_op_tokens.fetch_add((uint64_t) n_token, std::memory_order_relaxed);
        if (n_token > 1) {
            ctx->mwq_op_multi_token.fetch_add(1, std::memory_order_relaxed);
        }
    }

    const int64_t n_pair = n_token * n_rank;
    tls_mwq_pairs.clear();
    if (!tls_mwq_keep_sum_cache) {
        tls_mwq_sum_cache.clear();
        tls_mwq_qx_int8_cache.clear();
    }
    tls_mwq_pairs.reserve((size_t) n_pair);
    for (int64_t pair = 0; pair < n_pair; ++pair) {
        int requested_bits = 0;
        mwq_kernel_pair item;
        if (!moe_make_kernel_pair(*ctx, m, op, pair, n_col, item, &requested_bits)) {
            return false;
        }
        if (ith == 0 && item.backend != MOE_KERNEL_Q4K_NATIVE && item.bits != requested_bits) {
            ctx->mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
        }

        tls_mwq_pairs.push_back(item);

        if (ith == 0 && tls_mwq_pairs.size() == 1) {
            moe_mwq_check_once(*ctx, m, item, requested_bits, n_col, n_row, (size_t) src1->nb[0]);
        }
    }

    const size_t x_stride = (size_t) src1->nb[0];
    bool qx_shared = false;
    auto qx_shared_end = [&]() {
        if (qx_shared) {
            mwq_qx_shared_end(*ctx, op);
            qx_shared = false;
        }
    };
    if (ctx->params.vnni_q2 && ctx->params.vnni_q2_down && x_stride == sizeof(float)) {
        int qblock = 0;
        tls_mwq_qx_unique_xs.clear();
        for (const mwq_kernel_pair & item : tls_mwq_pairs) {
            if (mwq_dot_can_vnni_q2(ctx, item, 1, x_stride)) {
                const int qb = mwq_vnni_qblock(*ctx, item.block_size);
                if (qb > 0 && (qblock == 0 || qblock == qb)) {
                    qblock = qb;
                    mwq_qx_unique_push(tls_mwq_qx_unique_xs, item.x);
                }
            }
        }
        qx_shared = qblock > 0 && mwq_qx_shared_begin(*ctx, op, ith, nth, n_col, qblock, tls_mwq_qx_unique_xs);
    }
    const uint64_t avx512_q2_dot_calls0 = tls_mwq_avx512_q2_dot_calls;
    const uint64_t avx512_q2_dot_pairs0 = tls_mwq_avx512_q2_dot_pairs;
    const uint64_t avx512_q2_dot_rows0  = tls_mwq_avx512_q2_dot_rows;
    const uint64_t vnni_q2_dot_calls0 = tls_mwq_vnni_q2_dot_calls;
    const uint64_t vnni_q2_dot_pairs0 = tls_mwq_vnni_q2_dot_pairs;
    const uint64_t vnni_q2_dot_rows0  = tls_mwq_vnni_q2_dot_rows;
    auto flush_avx512_q2_dot_stats = [&]() {
        const uint64_t calls = tls_mwq_avx512_q2_dot_calls - avx512_q2_dot_calls0;
        const uint64_t pairs = tls_mwq_avx512_q2_dot_pairs - avx512_q2_dot_pairs0;
        const uint64_t rows  = tls_mwq_avx512_q2_dot_rows  - avx512_q2_dot_rows0;
        if (calls != 0) {
            ctx->mwq_avx512_q2_dot_calls.fetch_add(calls, std::memory_order_relaxed);
            ctx->mwq_avx512_q2_dot_pairs.fetch_add(pairs, std::memory_order_relaxed);
            ctx->mwq_avx512_q2_dot_rows.fetch_add(rows, std::memory_order_relaxed);
        }
    };
    auto flush_vnni_q2_dot_stats = [&]() {
        const uint64_t calls = tls_mwq_vnni_q2_dot_calls - vnni_q2_dot_calls0;
        const uint64_t pairs = tls_mwq_vnni_q2_dot_pairs - vnni_q2_dot_pairs0;
        const uint64_t rows  = tls_mwq_vnni_q2_dot_rows  - vnni_q2_dot_rows0;
        if (calls != 0) {
            ctx->mwq_vnni_q2_dot_calls.fetch_add(calls, std::memory_order_relaxed);
            ctx->mwq_vnni_q2_dot_pairs.fetch_add(pairs, std::memory_order_relaxed);
            ctx->mwq_vnni_q2_dot_rows.fetch_add(rows, std::memory_order_relaxed);
        }
    };
    if (n_token <= 1 || x_stride != sizeof(float)) {
        for (const mwq_kernel_pair & item : tls_mwq_pairs) {
            const int64_t n_tiles = (n_row + 3) / 4;
            if (item.backend == MOE_KERNEL_Q4K_NATIVE) {
                ggml_type qtype = GGML_TYPE_COUNT;
                if (!moe_native_quantize_x(item.native_type, item.x, x_stride, n_col, tls_native_x, qtype)) {
                    qx_shared_end();
                    return false;
                }
                for (int64_t tile = ith; tile < n_tiles; tile += nth) {
                    const int64_t row = tile * 4;
                    const int n_rows = (int) std::min<int64_t>(4, n_row - row);
                    moe_native_dot_rows(item.native_type, item.mwq, item.native_row_size, row, n_rows, n_col,
                            tls_native_x.data(), item.dst);
                }
                continue;
            }
            const bool fast_q2_item = mwq_dot_can_vnni_q2(ctx, item, 1, x_stride) ||
                mwq_dot_can_avx512_q2(ctx, item, 1, x_stride);
            const float * sum_x = (!fast_q2_item && ith < n_tiles) ?
                mwq_sum_x_blocks_cached(*ctx, item.x, x_stride, n_col, item.block_size) : nullptr;

            for (int64_t tile = ith; tile < n_tiles; tile += nth) {
                const int64_t row = tile * 4;
                const int n_rows = (int) std::min<int64_t>(4, n_row - row);
                mwq_dot_rows(ctx, item.mwq, item.bits, item.block_size, item.codec, item.scale_group, item.outlier_max,
                        item.n_elem, row, n_rows, n_col, item.x, x_stride,
                        sum_x, item.dst);
            }
        }
        if (ith == 0) {
            ctx->mwq_kernel_ops.fetch_add(1, std::memory_order_relaxed);
            ctx->mwq_kernel_slices.fetch_add((uint64_t) (n_token * n_rank), std::memory_order_relaxed);
        }
        flush_avx512_q2_dot_stats();
        flush_vnni_q2_dot_stats();
        qx_shared_end();
        return true;
    }

    tls_mwq_counts.assign((size_t) m.n_expert, 0);
    for (const mwq_kernel_pair & item : tls_mwq_pairs) {
        if (item.expert >= 0 && item.expert < m.n_expert) {
            tls_mwq_counts[(size_t) item.expert]++;
        }
    }
    tls_mwq_offsets.assign((size_t) m.n_expert + 1, 0);
    for (int e = 0; e < m.n_expert; ++e) {
        tls_mwq_offsets[(size_t) e + 1] = tls_mwq_offsets[(size_t) e] + tls_mwq_counts[(size_t) e];
    }
    tls_mwq_cursor = tls_mwq_offsets;
    tls_mwq_pairs_sorted.resize(tls_mwq_pairs.size());
    for (const mwq_kernel_pair & item : tls_mwq_pairs) {
        const int pos = tls_mwq_cursor[(size_t) item.expert]++;
        tls_mwq_pairs_sorted[(size_t) pos] = item;
    }
    for (int e = 0; e < m.n_expert; ++e) {
        const int b0 = tls_mwq_offsets[(size_t) e];
        const int b1 = tls_mwq_offsets[(size_t) e + 1];
        if (b1 - b0 > 1) {
            std::stable_sort(tls_mwq_pairs_sorted.begin() + b0, tls_mwq_pairs_sorted.begin() + b1,
                    [](const mwq_kernel_pair & a, const mwq_kernel_pair & b) {
                if (a.backend != b.backend) return a.backend < b.backend;
                if (a.bits != b.bits) return a.bits < b.bits;
                if (a.block_size != b.block_size) return a.block_size < b.block_size;
                if (a.codec != b.codec) return a.codec < b.codec;
                if (a.scale_group != b.scale_group) return a.scale_group < b.scale_group;
                if (a.outlier_max != b.outlier_max) return a.outlier_max < b.outlier_max;
                return a.pair < b.pair;
            });
        }
    }
    tls_mwq_pairs.swap(tls_mwq_pairs_sorted);
    if (ith == 0) {
        ctx->mwq_countsort_ops.fetch_add(1, std::memory_order_relaxed);
    }

    for (size_t g0 = 0; g0 < tls_mwq_pairs.size(); ) {
        size_t g1 = g0 + 1;
        while (g1 < tls_mwq_pairs.size() &&
                tls_mwq_pairs[g1].backend == tls_mwq_pairs[g0].backend &&
                tls_mwq_pairs[g1].expert == tls_mwq_pairs[g0].expert &&
                tls_mwq_pairs[g1].bits == tls_mwq_pairs[g0].bits &&
                tls_mwq_pairs[g1].block_size == tls_mwq_pairs[g0].block_size &&
                tls_mwq_pairs[g1].codec == tls_mwq_pairs[g0].codec &&
                tls_mwq_pairs[g1].scale_group == tls_mwq_pairs[g0].scale_group &&
                tls_mwq_pairs[g1].outlier_max == tls_mwq_pairs[g0].outlier_max) {
            ++g1;
        }

        const size_t group = g1 - g0;
        if (tls_mwq_pairs[g0].backend == MOE_KERNEL_Q4K_NATIVE) {
            for (size_t b = g0; b < g1; ++b) {
                const mwq_kernel_pair & item = tls_mwq_pairs[b];
                ggml_type qtype = GGML_TYPE_COUNT;
                if (!moe_native_quantize_x(item.native_type, item.x, x_stride, n_col, tls_native_x, qtype)) {
                    qx_shared_end();
                    return false;
                }
                const int64_t n_tiles = (n_row + 3) / 4;
                for (int64_t tile = ith; tile < n_tiles; tile += nth) {
                    const int64_t row = tile * 4;
                    const int n_rows = (int) std::min<int64_t>(4, n_row - row);
                    moe_native_dot_rows(item.native_type, item.mwq, item.native_row_size, row, n_rows, n_col,
                            tls_native_x.data(), item.dst);
                }
            }
        } else if (group >= 2 && x_stride == sizeof(float)) {
            const int micro_batch = moe_microbatch_size();
            if (ith == 0) {
                ctx->mwq_group_batches.fetch_add((uint64_t) ((group + (size_t) micro_batch - 1) / (size_t) micro_batch), std::memory_order_relaxed);
                ctx->mwq_group_pairs.fetch_add((uint64_t) group, std::memory_order_relaxed);
            }
            for (size_t mb = g0; mb < g1; mb += (size_t) micro_batch) {
                const size_t mb_end = std::min(g1, mb + (size_t) micro_batch);
                const int batch = (int) (mb_end - mb);
                const char * xs[16] = {};
                const float * sums[16] = {};
                float * dsts[16] = {};
                const bool fast_q2_batch = mwq_dot_can_vnni_q2(ctx, tls_mwq_pairs[mb], batch, x_stride) ||
                    mwq_dot_can_avx512_q2(ctx, tls_mwq_pairs[mb], batch, x_stride);

                for (int j = 0; j < batch; ++j) {
                    const mwq_kernel_pair & item = tls_mwq_pairs[mb + (size_t) j];
                    xs[j] = item.x;
                    dsts[j] = item.dst;
                    sums[j] = (!fast_q2_batch && ith < n_row) ?
                        mwq_sum_x_blocks_cached(*ctx, item.x, x_stride, n_col, item.block_size) : nullptr;
                }

                for (int64_t row = ith; row < n_row; row += nth) {
                    mwq_dot_row_batch(ctx, tls_mwq_pairs[mb].mwq, tls_mwq_pairs[mb].bits, tls_mwq_pairs[mb].block_size,
                            tls_mwq_pairs[mb].codec, tls_mwq_pairs[mb].scale_group, tls_mwq_pairs[mb].outlier_max,
                            tls_mwq_pairs[mb].n_elem, row, batch,
                            n_col, xs, x_stride, sums, dsts);
                }
            }
        } else {
            for (size_t b = g0; b < g1; ++b) {
                const mwq_kernel_pair & item = tls_mwq_pairs[b];
                const int64_t n_tiles = (n_row + 3) / 4;
                const bool fast_q2_item = mwq_dot_can_vnni_q2(ctx, item, 1, x_stride) ||
                    mwq_dot_can_avx512_q2(ctx, item, 1, x_stride);
                const float * sum_x = (!fast_q2_item && ith < n_tiles) ?
                    mwq_sum_x_blocks_cached(*ctx, item.x, x_stride, n_col, item.block_size) : nullptr;

                for (int64_t tile = ith; tile < n_tiles; tile += nth) {
                    const int64_t row = tile * 4;
                    const int n_rows = (int) std::min<int64_t>(4, n_row - row);
                    mwq_dot_rows(ctx, item.mwq, item.bits, item.block_size, item.codec, item.scale_group, item.outlier_max,
                            item.n_elem, row, n_rows, n_col, item.x, x_stride,
                            sum_x, item.dst);
                }
            }
        }
        g0 = g1;
    }

    if (ith == 0) {
        ctx->mwq_kernel_ops.fetch_add(1, std::memory_order_relaxed);
        ctx->mwq_kernel_slices.fetch_add((uint64_t) (n_token * n_rank), std::memory_order_relaxed);
    }
    flush_avx512_q2_dot_stats();
    flush_vnni_q2_dot_stats();
    qx_shared_end();
    return true;
}

static bool moe_name_has(const ggml_tensor * t, const char * needle) {
    return t != nullptr && std::strstr(ggml_get_name(t), needle) != nullptr;
}

static int moe_layer_from_name(const ggml_tensor * t) {
    int layer = -1;
    if (t != nullptr) {
        std::sscanf(ggml_get_name(t), "blk.%d.", &layer);
    }
    return layer;
}

static std::string moe_sibling_exps_name(const char * name, bool is_up) {
    std::string out = name != nullptr ? name : "";
    const char * from = is_up ? ".ffn_up_exps.weight" : ".ffn_gate_exps.weight";
    const char * to   = is_up ? ".ffn_gate_exps.weight" : ".ffn_up_exps.weight";
    const size_t pos = out.find(from);
    if (pos != std::string::npos) {
        out.replace(pos, std::strlen(from), to);
    }
    return out;
}

static std::string moe_down_exps_name(const char * name) {
    std::string out = name != nullptr ? name : "";
    const char * gate = ".ffn_gate_exps.weight";
    const char * up   = ".ffn_up_exps.weight";
    const char * down = ".ffn_down_exps.weight";
    size_t pos = out.find(gate);
    if (pos != std::string::npos) {
        out.replace(pos, std::strlen(gate), down);
        return out;
    }
    pos = out.find(up);
    if (pos != std::string::npos) {
        out.replace(pos, std::strlen(up), down);
    }
    return out;
}

static inline float moe_silu_f32(float x) {
    return x / (1.0f + std::exp(-x));
}

static void moe_vec_swiglu_f32(int n, float * dst, const float * gate, const float * up) {
    for (int i = 0; i < n; ++i) {
        dst[i] = moe_silu_f32(gate[i]) * up[i];
    }
}

static bool moe_dot_kernel_pair_prepare_native(
        const mwq_kernel_pair & item,
        size_t                  x_stride,
        int64_t                 n_col,
        std::vector<uint8_t> &  qx) {
    if (item.backend != MOE_KERNEL_Q4K_NATIVE) {
        return true;
    }
    ggml_type qtype = GGML_TYPE_COUNT;
    return moe_native_quantize_x(item.native_type, item.x, x_stride, n_col, qx, qtype);
}

static float moe_dot_kernel_pair_row(
        const mwq_kernel_pair & item,
        int64_t                 row,
        int64_t                 n_col,
        size_t                  x_stride,
        const float *           sum_x,
        const std::vector<uint8_t> & qx) {
    if (item.backend == MOE_KERNEL_Q4K_NATIVE) {
        return moe_native_dot_row(item.native_type, item.mwq, item.native_row_size, row, n_col, qx.data());
    }
    return mwq_dot_row(item.mwq, item.bits, item.block_size, item.codec, item.scale_group,
            item.outlier_max, item.n_elem, row, n_col, item.x, x_stride, sum_x);
}

static void moe_swiglu_check_once(
        const mwq_swiglu_pair * items,
        const float *           fused,
        int                     batch,
        int64_t                 row,
        int64_t                 n_col,
        size_t                  x_stride,
        const float * const *   sums) {
    static std::atomic<bool> checked{false};
    if (!moe_swiglu_check_enabled() || checked.exchange(true, std::memory_order_relaxed)) {
        return;
    }

    std::vector<uint8_t> empty_qx;
    double max_abs = 0.0;
    double max_rel = 0.0;
    int max_j = -1;
    for (int j = 0; j < batch; ++j) {
        const float gate_v = moe_dot_kernel_pair_row(items[j].gate, row, n_col, x_stride, sums[j], empty_qx);
        const float up_v   = moe_dot_kernel_pair_row(items[j].up,   row, n_col, x_stride, sums[j], empty_qx);
        const float ref = moe_silu_f32(gate_v) * up_v;
        const double abs = std::fabs((double) ref - (double) fused[j]);
        const double rel = abs / std::max(1.0, std::fabs((double) ref));
        if (abs > max_abs) {
            max_abs = abs;
            max_rel = rel;
            max_j = j;
        }
        std::fprintf(stderr,
                "llama_moe_buffer[swiglu-check]: row=%lld batch=%d pair=%lld expert=%d ref=% .9e fused=% .9e abs=%.9e rel=%.9e\n",
                (long long) row, j, (long long) items[j].gate.pair, items[j].gate.expert,
                (double) ref, (double) fused[j], abs, rel);
    }
    std::fprintf(stderr,
            "llama_moe_buffer[swiglu-check]: max_abs=%.9e max_rel=%.9e max_batch=%d\n",
            max_abs, max_rel, max_j);
}

static int moe_kernel_pair_group_cmp(const mwq_kernel_pair & a, const mwq_kernel_pair & b) {
    if (a.backend != b.backend) return a.backend < b.backend ? -1 : 1;
    if (a.expert != b.expert) return a.expert < b.expert ? -1 : 1;
    if (a.bits != b.bits) return a.bits < b.bits ? -1 : 1;
    if (a.block_size != b.block_size) return a.block_size < b.block_size ? -1 : 1;
    if (a.codec != b.codec) return a.codec < b.codec ? -1 : 1;
    if (a.scale_group != b.scale_group) return a.scale_group < b.scale_group ? -1 : 1;
    if (a.outlier_max != b.outlier_max) return a.outlier_max < b.outlier_max ? -1 : 1;
    if (a.n_elem != b.n_elem) return a.n_elem < b.n_elem ? -1 : 1;
    if (a.mwq != b.mwq) return a.mwq < b.mwq ? -1 : 1;
    if (a.native_type != b.native_type) return a.native_type < b.native_type ? -1 : 1;
    if (a.native_row_size != b.native_row_size) return a.native_row_size < b.native_row_size ? -1 : 1;
    return 0;
}

static bool moe_kernel_pair_same_group(const mwq_kernel_pair & a, const mwq_kernel_pair & b) {
    return moe_kernel_pair_group_cmp(a, b) == 0;
}

static bool moe_swiglu_pair_less(const mwq_swiglu_pair & a, const mwq_swiglu_pair & b) {
    const int gc = moe_kernel_pair_group_cmp(a.gate, b.gate);
    if (gc != 0) return gc < 0;
    const int uc = moe_kernel_pair_group_cmp(a.up, b.up);
    if (uc != 0) return uc < 0;
    return a.gate.pair < b.gate.pair;
}

static bool moe_swiglu_pair_same_group(const mwq_swiglu_pair & a, const mwq_swiglu_pair & b) {
    return moe_kernel_pair_same_group(a.gate, b.gate) && moe_kernel_pair_same_group(a.up, b.up);
}

static bool moe_ffn_triplet_less(const mwq_ffn_triplet & a, const mwq_ffn_triplet & b) {
    const int gc = moe_kernel_pair_group_cmp(a.gate, b.gate);
    if (gc != 0) return gc < 0;
    const int uc = moe_kernel_pair_group_cmp(a.up, b.up);
    if (uc != 0) return uc < 0;
    const int dc = moe_kernel_pair_group_cmp(a.down, b.down);
    if (dc != 0) return dc < 0;
    return a.gate.pair < b.gate.pair;
}

static bool moe_ffn_triplet_same_group(const mwq_ffn_triplet & a, const mwq_ffn_triplet & b) {
    return moe_kernel_pair_same_group(a.gate, b.gate) &&
        moe_kernel_pair_same_group(a.up, b.up) &&
        moe_kernel_pair_same_group(a.down, b.down);
}

static bool mwq_ffn_triplet_can_fuse_avx512_q2(
        llama_moe_buffer_context * ctx,
        const mwq_ffn_triplet &    item,
        size_t                     x_stride,
        int64_t                    n_ff) {
#if defined(LLAMA_MOE_CAN_COMPILE_AVX512)
    return ctx != nullptr &&
        item.gate.backend == MOE_KERNEL_MWQ &&
        item.up.backend == MOE_KERNEL_MWQ &&
        item.down.backend == MOE_KERNEL_MWQ &&
        item.gate.bits == 2 &&
        item.up.bits == 2 &&
        item.down.bits == 2 &&
        item.gate.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        item.up.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        item.down.codec == MOE_SIDECAR_CODEC_MWQ_HIER &&
        item.gate.block_size > 0 &&
        item.down.block_size > 0 &&
        (item.gate.block_size % 16) == 0 &&
        (item.down.block_size % 16) == 0 &&
        (n_ff % item.down.block_size) == 0 &&
        x_stride == sizeof(float) &&
        mwq_cpu_has_avx512_q2();
#else
    (void) ctx; (void) item; (void) x_stride; (void) n_ff;
    return false;
#endif
}

static bool llama_moe_buffer_swiglu_direct_compute(
        llama_moe_buffer_context * ctx,
        ggml_tensor *              op,
        ggml_tensor *              gate_op,
        ggml_tensor *              up_op,
        int                        ith,
        int                        nth) {
    if (ctx == nullptr || op == nullptr || gate_op == nullptr || up_op == nullptr ||
            gate_op->op != GGML_OP_MUL_MAT_ID || up_op->op != GGML_OP_MUL_MAT_ID ||
            gate_op->src[0] == nullptr || gate_op->src[1] == nullptr || gate_op->src[2] == nullptr ||
            up_op->src[0] == nullptr || up_op->src[1] == nullptr || up_op->src[2] == nullptr) {
        if (ith == 0 && ctx != nullptr) ctx->direct_swiglu_fail_precheck.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    ggml_tensor * gate_w = gate_op->src[0];
    ggml_tensor * up_w   = up_op->src[0];
    ggml_tensor * src1   = gate_op->src[1];
    ggml_tensor * ids    = gate_op->src[2];
    ggml_tensor * up_src1 = up_op->src[1];
    ggml_tensor * up_ids  = up_op->src[2];
    if (src1->data != up_src1->data || ids->data != up_ids->data ||
            src1->ne[0] != up_src1->ne[0] || src1->ne[1] != up_src1->ne[1] || src1->ne[2] != up_src1->ne[2] ||
            ids->ne[0] != up_ids->ne[0] || ids->ne[1] != up_ids->ne[1]) {
        if (ith == 0) ctx->direct_swiglu_fail_precheck.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32 ||
            gate_w->ne[0] != up_w->ne[0] || gate_w->ne[1] != up_w->ne[1] ||
            src1->ne[0] != gate_w->ne[0]) {
        if (ith == 0) ctx->direct_swiglu_fail_precheck.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    auto git = ctx->by_name.find(ggml_get_name(gate_w));
    auto uit = ctx->by_name.find(ggml_get_name(up_w));
    if (git == ctx->by_name.end() || uit == ctx->by_name.end()) {
        if (ith == 0) ctx->direct_swiglu_fail_byname.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    moe_managed & gate_m = git->second;
    moe_managed & up_m   = uit->second;
    moe_managed * down_m = nullptr;
    if (ctx->params.prefetch_down_with_swiglu) {
        auto dit = ctx->by_name.find(moe_down_exps_name(ggml_get_name(gate_w)));
        if (dit != ctx->by_name.end()) {
            down_m = &dit->second;
        }
    }

    const int64_t n_col = gate_w->ne[0];
    const int64_t n_row = gate_w->ne[1];
    const int64_t n_rank = ids->ne[0];
    const int64_t n_token = ids->ne[1];
    const int64_t n_pair = n_rank * n_token;
    if (op->ne[0] != n_row || op->ne[1] != n_rank || op->ne[2] != n_token) {
        if (ith == 0) ctx->direct_swiglu_fail_precheck.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    // Single-flight residency-ensure: this function is a ggml op_override
    // callback with no barrier between the nth worker threads (unlike
    // MUL_MAT_ID's weight-stream+barrier path), so without this every thread
    // would redundantly redo the streaming loop below. The first thread to
    // reach this op becomes the "leader" (not necessarily ith==0, since there
    // is no barrier guaranteeing ith==0 arrives first) and does the ensure
    // work alone; the rest wait on cv_direct_ensure for its result.
    bool ensure_ok;
    {
        std::unique_lock<std::mutex> lk(ctx->mtx);
        auto ent_it = ctx->direct_ensure_map.find(op);
        if (ent_it == ctx->direct_ensure_map.end()) {
            ent_it = ctx->direct_ensure_map.emplace(op, llama_moe_buffer_context::moe_direct_ensure_entry{}).first;
            ent_it->second.remaining = nth;
            lk.unlock();

            bool any_entry_missing = false;
            bool any_read_failed = false;
            for (int64_t pair = 0; pair < n_pair; ++pair) {
                const int64_t rank = pair % n_rank;
                const int64_t token = pair / n_rank;
                const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                const int gate_bits = moe_target_bits_for_id_index(*ctx, gate_m, expert, pair);
                const int up_bits   = moe_target_bits_for_id_index(*ctx, up_m,   expert, pair);
                bool em = false, rf = false;
                moe_stream_mwq_slice(*ctx, gate_m, expert, true, gate_bits, (int) rank, &em, &rf);
                any_entry_missing |= em; any_read_failed |= rf;
                em = false; rf = false;
                moe_stream_mwq_slice(*ctx, up_m, expert, true, up_bits, (int) rank, &em, &rf);
                any_entry_missing |= em; any_read_failed |= rf;
                if (down_m != nullptr) {
                    const int down_bits = moe_target_bits_for_id_index(*ctx, *down_m, expert, pair);
                    em = false; rf = false;
                    moe_stream_mwq_slice(*ctx, *down_m, expert, true, down_bits, (int) rank, &em, &rf);
                    if (!em && !rf) {
                        ctx->fused_direct_down_prefetch.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }

            moe_resident_fail_reason reason = moe_resident_fail_reason::NONE;
            const bool ok = moe_op_all_mwq_resident(*ctx, gate_m, ids, &reason) &&
                            moe_op_all_mwq_resident(*ctx, up_m, ids, &reason);
            if (!ok) {
                if (any_entry_missing) {
                    ctx->direct_swiglu_fail_entry_missing.fetch_add(1, std::memory_order_relaxed);
                } else if (any_read_failed) {
                    ctx->direct_swiglu_fail_read.fetch_add(1, std::memory_order_relaxed);
                } else if (reason == moe_resident_fail_reason::RANGE) {
                    ctx->direct_swiglu_fail_range.fetch_add(1, std::memory_order_relaxed);
                } else {
                    ctx->direct_swiglu_fail_other.fetch_add(1, std::memory_order_relaxed);
                }
            }

            lk.lock();
            ent_it = ctx->direct_ensure_map.find(op); // re-find: map may have rehashed while unlocked
            ent_it->second.ok = ok;
            ent_it->second.done = true;
            ctx->cv_direct_ensure.notify_all();
        } else {
            while (!ent_it->second.done) {
                ctx->cv_direct_ensure.wait(lk);
                ent_it = ctx->direct_ensure_map.find(op);
            }
        }
        ensure_ok = ent_it->second.ok;
        if (--ent_it->second.remaining == 0) {
            ctx->direct_ensure_map.erase(ent_it);
        }
    }
    if (!ensure_ok) {
        return false;
    }

    const size_t x_stride = (size_t) src1->nb[0];
    tls_mwq_swiglu_pairs.clear();
    tls_mwq_sum_cache.clear();
    tls_mwq_qx_int8_cache.clear();
    tls_mwq_swiglu_pairs.reserve((size_t) n_pair);
    for (int64_t pair = 0; pair < n_pair; ++pair) {
        mwq_kernel_pair gate_item;
        mwq_kernel_pair up_item;
        int gate_requested_bits = 0;
        int up_requested_bits = 0;
        if (!moe_make_kernel_pair(*ctx, gate_m, gate_op, pair, n_col, gate_item, &gate_requested_bits) ||
                !moe_make_kernel_pair(*ctx, up_m, up_op, pair, n_col, up_item, &up_requested_bits)) {
            if (ith == 0) ctx->direct_swiglu_fail_kernelpair.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (ith == 0) {
            if (gate_item.backend != MOE_KERNEL_Q4K_NATIVE && gate_item.bits != gate_requested_bits) {
                ctx->mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
            }
            if (up_item.backend != MOE_KERNEL_Q4K_NATIVE && up_item.bits != up_requested_bits) {
                ctx->mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
            }
        }
        tls_mwq_swiglu_pairs.push_back({gate_item, up_item});
    }

    tls_mwq_swiglu_pairs_sorted = tls_mwq_swiglu_pairs;
    std::stable_sort(tls_mwq_swiglu_pairs_sorted.begin(), tls_mwq_swiglu_pairs_sorted.end(), moe_swiglu_pair_less);
    tls_mwq_swiglu_pairs.swap(tls_mwq_swiglu_pairs_sorted);
    if (ith == 0 && n_pair > 1) {
        ctx->mwq_countsort_ops.fetch_add(1, std::memory_order_relaxed);
    }

    const uint64_t avx512_q2_calls0 = tls_mwq_avx512_q2_swiglu_calls;
    const uint64_t avx512_q2_pairs0 = tls_mwq_avx512_q2_swiglu_pairs;
    const uint64_t avx512_q2_rows0  = tls_mwq_avx512_q2_swiglu_rows;
    const uint64_t vnni_q2_calls0 = tls_mwq_vnni_q2_swiglu_calls;
    const uint64_t vnni_q2_pairs0 = tls_mwq_vnni_q2_swiglu_pairs;
    const uint64_t vnni_q2_rows0  = tls_mwq_vnni_q2_swiglu_rows;

    bool qx_shared = false;
    auto qx_shared_end = [&]() {
        if (qx_shared) {
            mwq_qx_shared_end(*ctx, op);
            qx_shared = false;
        }
    };
    if (ctx->params.vnni_q2 && ctx->params.vnni_q2_swiglu && x_stride == sizeof(float)) {
        int qblock = 0;
        tls_mwq_qx_unique_xs.clear();
        for (size_t g0 = 0; g0 < tls_mwq_swiglu_pairs.size(); ) {
            size_t g1 = g0 + 1;
            while (g1 < tls_mwq_swiglu_pairs.size() &&
                    moe_swiglu_pair_same_group(tls_mwq_swiglu_pairs[g0], tls_mwq_swiglu_pairs[g1])) {
                ++g1;
            }
            const int batch = (int) std::min(g1 - g0, (size_t) moe_microbatch_size());
            const mwq_swiglu_pair & first = tls_mwq_swiglu_pairs[g0];
            if (batch >= 2 && mwq_swiglu_can_vnni_q2(ctx, first.gate, first.up, batch, x_stride)) {
                const int qb = mwq_vnni_qblock(*ctx, first.gate.block_size);
                if (qb > 0 && (qblock == 0 || qblock == qb)) {
                    qblock = qb;
                    for (size_t i = g0; i < g1; ++i) {
                        mwq_qx_unique_push(tls_mwq_qx_unique_xs, tls_mwq_swiglu_pairs[i].gate.x);
                    }
                }
            }
            g0 = g1;
        }
        qx_shared = qblock > 0 && mwq_qx_shared_begin(*ctx, op, ith, nth, n_col, qblock, tls_mwq_qx_unique_xs);
    }

    for (size_t g0 = 0; g0 < tls_mwq_swiglu_pairs.size(); ) {
        size_t g1 = g0 + 1;
        while (g1 < tls_mwq_swiglu_pairs.size() &&
                moe_swiglu_pair_same_group(tls_mwq_swiglu_pairs[g0], tls_mwq_swiglu_pairs[g1])) {
            ++g1;
        }

        const size_t group = g1 - g0;
        const mwq_swiglu_pair & first = tls_mwq_swiglu_pairs[g0];
        const bool batchable =
            group >= 2 &&
            x_stride == sizeof(float) &&
            first.gate.backend == MOE_KERNEL_MWQ &&
            first.up.backend == MOE_KERNEL_MWQ;

        if (batchable) {
            const int micro_batch = moe_microbatch_size();
            if (ith == 0) {
                ctx->mwq_group_batches.fetch_add((uint64_t) ((group + (size_t) micro_batch - 1) / (size_t) micro_batch), std::memory_order_relaxed);
                ctx->mwq_group_pairs.fetch_add((uint64_t) group, std::memory_order_relaxed);
            }

            for (size_t mb = g0; mb < g1; mb += (size_t) micro_batch) {
                const size_t mb_end = std::min(g1, mb + (size_t) micro_batch);
                const int batch = (int) (mb_end - mb);
                const char * xs[16] = {};
                const float * sums[16] = {};
                float swiglu_vals[16] = {};
                const bool fast_q2_batch =
                    mwq_swiglu_can_vnni_q2(ctx, first.gate, first.up, batch, x_stride) ||
                    mwq_swiglu_can_avx512_q2(ctx, first.gate, first.up, batch, x_stride);

                for (int j = 0; j < batch; ++j) {
                    const mwq_swiglu_pair & item = tls_mwq_swiglu_pairs[mb + (size_t) j];
                    xs[j] = item.gate.x;
                    sums[j] = fast_q2_batch ? nullptr :
                        mwq_sum_x_blocks_cached(*ctx, item.gate.x, x_stride, n_col, item.gate.block_size);
                }

                for (int64_t row = ith; row < n_row; row += nth) {
                    mwq_swiglu_row_batch_values(ctx, first.gate, first.up, row, batch, n_col, xs, x_stride, sums, swiglu_vals);
                    moe_swiglu_check_once(&tls_mwq_swiglu_pairs[mb], swiglu_vals, batch, row, n_col, x_stride, sums);

                    for (int j = 0; j < batch; ++j) {
                        const mwq_swiglu_pair & item = tls_mwq_swiglu_pairs[mb + (size_t) j];
                        float * dst = (float *) ((char *) op->data +
                                row * op->nb[0] + item.gate.rank * op->nb[1] + item.gate.token * op->nb[2]);
                        *dst = swiglu_vals[j];
                    }
                }
            }
        } else {
            for (size_t i = g0; i < g1; ++i) {
                const mwq_swiglu_pair & item = tls_mwq_swiglu_pairs[i];
                std::vector<uint8_t> gate_qx;
                std::vector<uint8_t> up_qx;
                if (!moe_dot_kernel_pair_prepare_native(item.gate, x_stride, n_col, gate_qx) ||
                        !moe_dot_kernel_pair_prepare_native(item.up, x_stride, n_col, up_qx)) {
                    if (ith == 0) ctx->direct_swiglu_fail_native_prepare.fetch_add(1, std::memory_order_relaxed);
                    qx_shared_end();
                    return false;
                }

                const float * gate_sum_x = item.gate.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
                    mwq_sum_x_blocks_cached(*ctx, item.gate.x, x_stride, n_col, item.gate.block_size);
                const float * up_sum_x = item.up.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
                    mwq_sum_x_blocks_cached(*ctx, item.up.x, x_stride, n_col, item.up.block_size);

                for (int64_t row = ith; row < n_row; row += nth) {
                    const float gate_v = moe_dot_kernel_pair_row(item.gate, row, n_col, x_stride, gate_sum_x, gate_qx);
                    const float up_v   = moe_dot_kernel_pair_row(item.up,   row, n_col, x_stride, up_sum_x,   up_qx);
                    float * dst = (float *) ((char *) op->data +
                            row * op->nb[0] + item.gate.rank * op->nb[1] + item.gate.token * op->nb[2]);
                    *dst = moe_silu_f32(gate_v) * up_v;
                }
            }
        }
        g0 = g1;
    }

    const uint64_t avx512_q2_calls = tls_mwq_avx512_q2_swiglu_calls - avx512_q2_calls0;
    const uint64_t avx512_q2_pairs = tls_mwq_avx512_q2_swiglu_pairs - avx512_q2_pairs0;
    const uint64_t avx512_q2_rows  = tls_mwq_avx512_q2_swiglu_rows  - avx512_q2_rows0;
    if (avx512_q2_calls != 0) {
        ctx->mwq_avx512_q2_swiglu_calls.fetch_add(avx512_q2_calls, std::memory_order_relaxed);
        ctx->mwq_avx512_q2_swiglu_pairs.fetch_add(avx512_q2_pairs, std::memory_order_relaxed);
        ctx->mwq_avx512_q2_swiglu_rows.fetch_add(avx512_q2_rows, std::memory_order_relaxed);
    }
    const uint64_t vnni_q2_calls = tls_mwq_vnni_q2_swiglu_calls - vnni_q2_calls0;
    const uint64_t vnni_q2_pairs = tls_mwq_vnni_q2_swiglu_pairs - vnni_q2_pairs0;
    const uint64_t vnni_q2_rows  = tls_mwq_vnni_q2_swiglu_rows  - vnni_q2_rows0;
    if (vnni_q2_calls != 0) {
        ctx->mwq_vnni_q2_swiglu_calls.fetch_add(vnni_q2_calls, std::memory_order_relaxed);
        ctx->mwq_vnni_q2_swiglu_pairs.fetch_add(vnni_q2_pairs, std::memory_order_relaxed);
        ctx->mwq_vnni_q2_swiglu_rows.fetch_add(vnni_q2_rows, std::memory_order_relaxed);
    }

    if (ith == 0) {
        ctx->fused_direct_swiglu_hits.fetch_add(1, std::memory_order_relaxed);
        ctx->fused_direct_swiglu_rows.fetch_add((uint64_t) (n_pair * n_row), std::memory_order_relaxed);
        ctx->mwq_kernel_ops.fetch_add(2, std::memory_order_relaxed);
        ctx->mwq_kernel_slices.fetch_add((uint64_t) (2 * n_pair), std::memory_order_relaxed);
    }
    qx_shared_end();
    return true;
}

static bool llama_moe_buffer_swiglu_materialize(
        llama_moe_buffer_context * ctx,
        ggml_tensor *              op,
        ggml_tensor *              gate_op,
        ggml_tensor *              up_op,
        int                        ith,
        int                        nth) {
    if (ctx == nullptr || op == nullptr || gate_op == nullptr || up_op == nullptr) {
        return false;
    }
    if (llama_moe_buffer_swiglu_direct_compute(ctx, op, gate_op, up_op, ith, nth)) {
        return true;
    }

    tls_mwq_keep_sum_cache = false;
    const bool ok_up = llama_moe_buffer_mul_mat_id_compute(up_op, ith, nth, ctx);
    tls_mwq_keep_sum_cache = true;
    const bool ok_gate = llama_moe_buffer_mul_mat_id_compute(gate_op, ith, nth, ctx);
    tls_mwq_keep_sum_cache = false;
    if (!ok_up || !ok_gate) {
        return false;
    }

    ggml_tensor * gate = op->src[0];
    ggml_tensor * up   = op->src[1];
    if (gate == nullptr || up == nullptr ||
            gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(gate, up) ||
            !ggml_is_contiguous_1(gate) || !ggml_is_contiguous_1(up) || !ggml_is_contiguous_1(op)) {
        return false;
    }

    const int64_t nc = gate->ne[0];
    const int64_t nr = ggml_nrows(gate);
    if (op->ne[0] != nc || ggml_nrows(op) != nr || nc > INT_MAX) {
        return false;
    }

    const int64_t dr = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = std::min<int64_t>(ir0 + dr, nr);
    for (int64_t row = ir0; row < ir1; ++row) {
        const float * gate_p = (const float *) ((const char *) gate->data + row * gate->nb[1]);
        const float * up_p   = (const float *) ((const char *) up->data   + row * up->nb[1]);
        float * dst_p        = (float *)       ((char *) op->data         + row * op->nb[1]);
        moe_vec_swiglu_f32((int) nc, dst_p, gate_p, up_p);
    }
    return true;
}

static bool llama_moe_buffer_fused_ffn_serial_run(
        llama_moe_buffer_context *          ctx,
        ggml_tensor *                       down_op,
        ggml_tensor *                       swiglu_op,
        const std::vector<mwq_ffn_triplet> & items,
        int64_t                             n_hidden,
        int64_t                             n_ff,
        size_t                              x_stride,
        int                                 ith,
        int                                 nth) {
    if (ctx == nullptr || down_op == nullptr) {
        return false;
    }

    bool ok = true;
    if (ith == 0) {
        std::vector<float> swiglu((size_t) n_ff);
        std::vector<float> down_sum_x;
        std::vector<uint8_t> gate_qx;
        std::vector<uint8_t> up_qx;
        std::vector<uint8_t> down_qx;

        for (const mwq_ffn_triplet & item : items) {
            const bool use_materialized_swiglu =
                !ctx->params.fuse_direct_swiglu &&
                swiglu_op != nullptr &&
                swiglu_op->data != nullptr &&
                swiglu_op->type == GGML_TYPE_F32 &&
                swiglu_op->nb[0] == (int64_t) sizeof(float) &&
                swiglu_op->ne[0] == n_ff;
            const char * swiglu_x = nullptr;
            size_t swiglu_stride = sizeof(float);

            gate_qx.clear();
            up_qx.clear();
            down_qx.clear();
            if (use_materialized_swiglu) {
                swiglu_x = item.down.x;
                swiglu_stride = (size_t) swiglu_op->nb[0];
                for (int64_t row = 0; row < n_ff; ++row) {
                    swiglu[(size_t) row] = *(const float *) (swiglu_x + row * swiglu_stride);
                }
                swiglu_x = (const char *) swiglu.data();
                swiglu_stride = sizeof(float);
            } else {
                if (!moe_dot_kernel_pair_prepare_native(item.gate, x_stride, n_hidden, gate_qx) ||
                        !moe_dot_kernel_pair_prepare_native(item.up, x_stride, n_hidden, up_qx)) {
                    ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
                    ok = false;
                    break;
                }

                const float * gate_sum_x = item.gate.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
                    mwq_sum_x_blocks_cached(*ctx, item.gate.x, x_stride, n_hidden, item.gate.block_size);
                const float * up_sum_x = item.up.backend == MOE_KERNEL_Q4K_NATIVE ? nullptr :
                    mwq_sum_x_blocks_cached(*ctx, item.up.x, x_stride, n_hidden, item.up.block_size);

                for (int64_t row = 0; row < n_ff; ++row) {
                    const float gate_v = moe_dot_kernel_pair_row(item.gate, row, n_hidden, x_stride, gate_sum_x, gate_qx);
                    const float up_v   = moe_dot_kernel_pair_row(item.up,   row, n_hidden, x_stride, up_sum_x,   up_qx);
                    swiglu[(size_t) row] = moe_silu_f32(gate_v) * up_v;
                }
                swiglu_x = (const char *) swiglu.data();
            }

            mwq_ffn_triplet serial_item = item;
            serial_item.down.x = swiglu_x;
            down_sum_x.clear();
            const float * down_sum = nullptr;
            if (serial_item.down.backend != MOE_KERNEL_Q4K_NATIVE) {
                mwq_sum_x_blocks(swiglu_x, swiglu_stride, n_ff,
                        serial_item.down.block_size, down_sum_x);
                down_sum = down_sum_x.data();
            }
            if (!moe_dot_kernel_pair_prepare_native(serial_item.down, swiglu_stride, n_ff, down_qx)) {
                ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
                ok = false;
                break;
            }

            for (int64_t row = 0; row < n_hidden; ++row) {
                item.down.dst[row] = moe_dot_kernel_pair_row(serial_item.down, row, n_ff,
                        swiglu_stride, down_sum, down_qx);
            }
        }

        if (ok) {
            ctx->fused_ffn_hits.fetch_add(1, std::memory_order_relaxed);
            ctx->fused_ffn_runs.fetch_add((uint64_t) items.size(), std::memory_order_relaxed);
            ctx->fused_ffn_pairs.fetch_add((uint64_t) items.size(), std::memory_order_relaxed);
            ctx->fused_ffn_tiles.fetch_add((uint64_t) items.size(), std::memory_order_relaxed);
            ctx->mwq_kernel_ops.fetch_add(3, std::memory_order_relaxed);
            ctx->mwq_kernel_slices.fetch_add((uint64_t) (3 * items.size()), std::memory_order_relaxed);
        }
    }

    moe_ffn_op_barrier(*ctx, down_op, nth);
    return ok;
}

static bool llama_moe_buffer_fused_ffn_compute(
        llama_moe_buffer_context * ctx,
        ggml_tensor *              down_op,
        ggml_tensor *              swiglu_op,
        ggml_tensor *              gate_op,
        ggml_tensor *              up_op,
        int                        ith,
        int                        nth) {
    if (ctx == nullptr || down_op == nullptr || swiglu_op == nullptr || gate_op == nullptr || up_op == nullptr ||
            down_op->op != GGML_OP_MUL_MAT_ID ||
            swiglu_op->op != GGML_OP_GLU ||
            gate_op->op != GGML_OP_MUL_MAT_ID ||
            up_op->op != GGML_OP_MUL_MAT_ID ||
            ggml_get_glu_op(swiglu_op) != GGML_GLU_OP_SWIGLU ||
            swiglu_op->op_params[1] != 0) {
        return false;
    }

    ggml_tensor * down_w = down_op->src[0];
    ggml_tensor * gate_w = gate_op->src[0];
    ggml_tensor * up_w   = up_op->src[0];
    ggml_tensor * src1   = gate_op->src[1];
    ggml_tensor * ids    = down_op->src[2];
    if (down_w == nullptr || gate_w == nullptr || up_w == nullptr ||
            src1 == nullptr || ids == nullptr ||
            gate_op->src[2] == nullptr || up_op->src[2] == nullptr ||
            gate_op->src[1] == nullptr || up_op->src[1] == nullptr ||
            !moe_name_has(down_w, ".ffn_down_exps.weight") ||
            !moe_name_has(gate_w, ".ffn_gate_exps.weight") ||
            !moe_name_has(up_w, ".ffn_up_exps.weight")) {
        if (ith == 0) ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    if (src1->data != up_op->src[1]->data ||
            ids->data != gate_op->src[2]->data || ids->data != up_op->src[2]->data ||
            src1->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            down_op->type != GGML_TYPE_F32 || swiglu_op->type != GGML_TYPE_F32 ||
            down_op->nb[0] != (int64_t) sizeof(float) ||
            swiglu_op->ne[0] != down_w->ne[0] ||
            gate_w->ne[0] != up_w->ne[0] ||
            gate_w->ne[1] != up_w->ne[1] ||
            gate_w->ne[1] != down_w->ne[0] ||
            down_w->ne[1] != gate_w->ne[0] ||
            src1->ne[0] != gate_w->ne[0] ||
            down_op->ne[0] != down_w->ne[1]) {
        if (ith == 0) ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    auto dit = ctx->by_name.find(ggml_get_name(down_w));
    auto git = ctx->by_name.find(ggml_get_name(gate_w));
    auto uit = ctx->by_name.find(ggml_get_name(up_w));
    if (dit == ctx->by_name.end() || git == ctx->by_name.end() || uit == ctx->by_name.end()) {
        if (ith == 0) ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    moe_managed & down_m = dit->second;
    moe_managed & gate_m = git->second;
    moe_managed & up_m   = uit->second;

    const int64_t n_hidden = gate_w->ne[0];
    const int64_t n_ff = down_w->ne[0];
    const int64_t n_rank = ids->ne[0];
    const int64_t n_token = ids->ne[1];
    const int64_t n_pair = n_rank * n_token;
    const size_t x_stride = (size_t) src1->nb[0];
    const bool fast_ffn = moe_fused_ffn_fast_enabled();

    if (!fast_ffn && !moe_fused_ffn_serial_direct_enabled()) {
        if (!llama_moe_buffer_swiglu_materialize(ctx, swiglu_op, gate_op, up_op, ith, nth)) {
            if (ith == 0) ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        const bool ok = llama_moe_buffer_mul_mat_id_compute(down_op, ith, nth, ctx);
        if (ok && ith == 0) {
            ctx->fused_ffn_hits.fetch_add(1, std::memory_order_relaxed);
            ctx->fused_ffn_runs.fetch_add(1, std::memory_order_relaxed);
            ctx->fused_ffn_pairs.fetch_add((uint64_t) n_pair, std::memory_order_relaxed);
        }
        return ok;
    }

    bool ensure_ok;
    {
        std::unique_lock<std::mutex> lk(ctx->mtx);
        auto ent_it = ctx->direct_ensure_map.find(down_op);
        if (ent_it == ctx->direct_ensure_map.end()) {
            ent_it = ctx->direct_ensure_map.emplace(down_op, llama_moe_buffer_context::moe_direct_ensure_entry{}).first;
            ent_it->second.remaining = nth;
            lk.unlock();

            bool any_entry_missing = false;
            bool any_read_failed = false;
            for (int64_t pair = 0; pair < n_pair; ++pair) {
                const int64_t rank = pair % n_rank;
                const int64_t token = pair / n_rank;
                const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                const int gate_bits = moe_target_bits_for_id_index(*ctx, gate_m, expert, pair);
                const int up_bits   = moe_target_bits_for_id_index(*ctx, up_m,   expert, pair);
                const int down_bits = moe_target_bits_for_id_index(*ctx, down_m, expert, pair);
                bool em = false, rf = false;
                moe_stream_mwq_slice(*ctx, gate_m, expert, true, gate_bits, (int) rank, &em, &rf);
                any_entry_missing |= em; any_read_failed |= rf;
                em = false; rf = false;
                moe_stream_mwq_slice(*ctx, up_m, expert, true, up_bits, (int) rank, &em, &rf);
                any_entry_missing |= em; any_read_failed |= rf;
                em = false; rf = false;
                moe_stream_mwq_slice(*ctx, down_m, expert, true, down_bits, (int) rank, &em, &rf);
                any_entry_missing |= em; any_read_failed |= rf;
            }

            moe_resident_fail_reason reason = moe_resident_fail_reason::NONE;
            const bool ok =
                moe_op_all_mwq_resident(*ctx, gate_m, ids, &reason) &&
                moe_op_all_mwq_resident(*ctx, up_m, ids, &reason) &&
                moe_op_all_mwq_resident(*ctx, down_m, ids, &reason);
            (void) any_entry_missing;
            (void) any_read_failed;

            lk.lock();
            ent_it = ctx->direct_ensure_map.find(down_op);
            ent_it->second.ok = ok;
            ent_it->second.done = true;
            ctx->cv_direct_ensure.notify_all();
        } else {
            while (!ent_it->second.done) {
                ctx->cv_direct_ensure.wait(lk);
                ent_it = ctx->direct_ensure_map.find(down_op);
            }
        }
        ensure_ok = ent_it->second.ok;
        if (--ent_it->second.remaining == 0) {
            ctx->direct_ensure_map.erase(ent_it);
        }
    }
    if (!ensure_ok) {
        if (ith == 0) ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    tls_mwq_ffn_triplets.clear();
    tls_mwq_ffn_triplets.reserve((size_t) n_pair);
    tls_mwq_sum_cache.clear();
    tls_mwq_qx_int8_cache.clear();
    for (int64_t pair = 0; pair < n_pair; ++pair) {
        mwq_ffn_triplet item;
        int gate_requested_bits = 0;
        int up_requested_bits = 0;
        int down_requested_bits = 0;
        if (!moe_make_kernel_pair(*ctx, gate_m, gate_op, pair, n_hidden, item.gate, &gate_requested_bits) ||
                !moe_make_kernel_pair(*ctx, up_m, up_op, pair, n_hidden, item.up, &up_requested_bits) ||
                !moe_make_kernel_pair(*ctx, down_m, down_op, pair, n_ff, item.down, &down_requested_bits)) {
            if (ith == 0) ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (fast_ffn && !mwq_ffn_triplet_can_fuse_avx512_q2(ctx, item, x_stride, n_ff)) {
            if (ith == 0) ctx->fused_ffn_misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (ith == 0) {
            if (item.gate.bits != gate_requested_bits) ctx->mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
            if (item.up.bits != up_requested_bits) ctx->mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
            if (item.down.bits != down_requested_bits) ctx->mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
        }
        tls_mwq_ffn_triplets.push_back(item);
    }

    if (!fast_ffn) {
        return llama_moe_buffer_fused_ffn_serial_run(ctx, down_op, swiglu_op, tls_mwq_ffn_triplets,
                n_hidden, n_ff, x_stride, ith, nth);
    }

    tls_mwq_ffn_triplets_sorted = tls_mwq_ffn_triplets;
    std::stable_sort(tls_mwq_ffn_triplets_sorted.begin(), tls_mwq_ffn_triplets_sorted.end(), moe_ffn_triplet_less);
    tls_mwq_ffn_triplets.swap(tls_mwq_ffn_triplets_sorted);

    size_t run_index = 0;
    uint64_t local_runs = 0;
    uint64_t local_pairs = 0;
    uint64_t local_tiles = 0;
    const int micro_batch = moe_microbatch_size();
    const int producer_threads = moe_fused_ffn_producer_threads(nth);
    const int consumer_threads = nth > 1 ? nth - producer_threads : 0;
    const bool pipeline_enabled =
        std::getenv("LLAMA_LAZY_MOE_FFN_PIPELINE") != nullptr &&
        std::atoi(std::getenv("LLAMA_LAZY_MOE_FFN_PIPELINE")) > 0;
    const bool use_pipeline = pipeline_enabled && consumer_threads > 0;
    const bool is_producer = !use_pipeline || ith < producer_threads;
    const bool is_consumer = !use_pipeline || ith >= producer_threads;
    const int consumer_index = use_pipeline ? ith - producer_threads : ith;
    for (size_t g0 = 0; g0 < tls_mwq_ffn_triplets.size(); ) {
        size_t g1 = g0 + 1;
        while (g1 < tls_mwq_ffn_triplets.size() &&
                moe_ffn_triplet_same_group(tls_mwq_ffn_triplets[g0], tls_mwq_ffn_triplets[g1])) {
            ++g1;
        }

        for (size_t mb = g0; mb < g1; mb += (size_t) micro_batch, ++run_index) {
            const size_t mb_end = std::min(g1, mb + (size_t) micro_batch);
            const int batch = (int) (mb_end - mb);
            const mwq_ffn_triplet & first = tls_mwq_ffn_triplets[mb];
            const int n_blocks = (int) (n_ff / first.down.block_size);

            const char * xs[16] = {};
            const float * sums[16] = {};
            float * dsts[16] = {};
            const bool fast_swiglu =
                batch > 1 && mwq_swiglu_can_avx512_q2(ctx, first.gate, first.up, batch, x_stride);
            for (int j = 0; j < batch; ++j) {
                const mwq_ffn_triplet & item = tls_mwq_ffn_triplets[mb + (size_t) j];
                xs[j] = item.gate.x;
                dsts[j] = item.down.dst;
                sums[j] = (fast_swiglu || (use_pipeline && !is_producer)) ? nullptr :
                    mwq_sum_x_blocks_cached(*ctx, item.gate.x, x_stride, n_hidden, item.gate.block_size);
            }

            const int row_tile = moe_fused_ffn_row_tile(use_pipeline);
            const int64_t n_row_tiles = (n_hidden + row_tile - 1) / row_tile;
            if (!use_pipeline) {
                for (int64_t tile = ith; tile < n_row_tiles; tile += nth) {
                    const int64_t row0 = tile * row_tile;
                    const int n_rows = (int) std::min<int64_t>(row_tile, n_hidden - row0);
                    mwq_ffn_run_rows_hier_q2_avx512(ctx, first, batch, n_hidden, n_ff,
                            xs, sums, dsts, x_stride, row0, n_rows);
                }
            } else {
                const int ring = moe_fused_ffn_ring_slots();
                auto pipe = moe_ffn_pipe_acquire(*ctx, down_op, run_index, nth, batch,
                        first.down.block_size, n_blocks, ring, consumer_threads);

                if (is_consumer) {
                    for (int64_t tile = consumer_index; tile < n_row_tiles; tile += consumer_threads) {
                        const int64_t row0 = tile * row_tile;
                        const int n_rows = (int) std::min<int64_t>(row_tile, n_hidden - row0);
                        for (int j = 0; j < batch; ++j) {
                            std::fill(dsts[j] + row0, dsts[j] + row0 + n_rows, 0.0f);
                        }
                    }
                }

                if (is_producer) {
                    for (int ib = ith; ib < n_blocks; ib += producer_threads) {
                        const int slot = ib % ring;
                        auto & ps = pipe->slots[(size_t) slot];
                        while (ps.state.load(std::memory_order_acquire) != 0) {
                            moe_spin_pause();
                        }
                        ps.state.store(1, std::memory_order_release);
                        float * tile_data = pipe->data.data() +
                            (size_t) slot * (size_t) batch * (size_t) first.down.block_size;
                        mwq_ffn_build_swiglu_block_hier_q2_avx512(ctx, first, batch, n_hidden,
                                xs, sums, x_stride, (int64_t) ib * first.down.block_size, tile_data);
                        ps.block.store(ib, std::memory_order_release);
                        ps.consumers_left.store(consumer_threads, std::memory_order_release);
                        ps.state.store(2, std::memory_order_release);
                    }
                    for (int s = 0; s < ring; ++s) {
                        while (pipe->slots[(size_t) s].state.load(std::memory_order_acquire) != 0) {
                            moe_spin_pause();
                        }
                    }
                } else {
                    for (int ib = 0; ib < n_blocks; ++ib) {
                        const int slot = ib % ring;
                        auto & ps = pipe->slots[(size_t) slot];
                        for (;;) {
                            const int state = ps.state.load(std::memory_order_acquire);
                            const int block = ps.block.load(std::memory_order_acquire);
                            if (state == 2 && block == ib) {
                                break;
                            }
                            moe_spin_pause();
                        }
                        const float * tile_data = pipe->data.data() +
                            (size_t) slot * (size_t) batch * (size_t) first.down.block_size;
                        for (int64_t tile = consumer_index; tile < n_row_tiles; tile += consumer_threads) {
                            const int64_t row0 = tile * row_tile;
                            const int n_rows = (int) std::min<int64_t>(row_tile, n_hidden - row0);
                            mwq_down_accum_swiglu_block_hier_q2_avx512(first, batch, n_ff, ib,
                                    tile_data, dsts, row0, n_rows);
                        }
                        if (ps.consumers_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            ps.block.store(-1, std::memory_order_release);
                            ps.state.store(0, std::memory_order_release);
                        }
                    }
                    for (int64_t tile = consumer_index; tile < n_row_tiles; tile += consumer_threads) {
                        const int64_t row0 = tile * row_tile;
                        const int n_rows = (int) std::min<int64_t>(row_tile, n_hidden - row0);
                        moe_ffn_pipeline_check_once(ctx, first, batch, n_hidden, n_ff,
                                xs, sums, dsts, x_stride, row0, n_rows);
                    }
                }
                if (pipe->done_count.fetch_add(1, std::memory_order_acq_rel) + 1 == nth) {
                    pipe->can_leave.store(1, std::memory_order_release);
                } else {
                    while (pipe->can_leave.load(std::memory_order_acquire) == 0) {
                        moe_spin_pause();
                    }
                }
                moe_ffn_pipe_release(*ctx, down_op, run_index, pipe);
            }

            if (ith == 0) {
                ++local_runs;
                local_pairs += (uint64_t) batch;
                local_tiles += (uint64_t) n_blocks;
            }
        }
        g0 = g1;
    }

    moe_ffn_op_barrier(*ctx, down_op, nth);

    if (local_runs != 0) {
        ctx->fused_ffn_runs.fetch_add(local_runs, std::memory_order_relaxed);
        ctx->fused_ffn_pairs.fetch_add(local_pairs, std::memory_order_relaxed);
        ctx->fused_ffn_tiles.fetch_add(local_tiles, std::memory_order_relaxed);
    }
    if (ith == 0) {
        ctx->fused_ffn_hits.fetch_add(1, std::memory_order_relaxed);
        ctx->mwq_kernel_ops.fetch_add(3, std::memory_order_relaxed);
        ctx->mwq_kernel_slices.fetch_add((uint64_t) (3 * n_pair), std::memory_order_relaxed);
        ctx->mwq_countsort_ops.fetch_add(1, std::memory_order_relaxed);
    }
    return true;
}

static bool llama_moe_buffer_swiglu_callback(llama_moe_buffer_context * ctx, ggml_tensor * op, int ith, int nth) {
    if (ctx == nullptr || op == nullptr || !ctx->params.fuse_swiglu || op->op != GGML_OP_GLU) {
        return false;
    }
    if (ggml_get_glu_op(op) != GGML_GLU_OP_SWIGLU || op->op_params[1] != 0) {
        return false;
    }

    ggml_tensor * gate = op->src[0];
    ggml_tensor * up   = op->src[1];
    if (gate == nullptr || up == nullptr ||
            gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(gate, up) ||
            !ggml_is_contiguous_1(gate) || !ggml_is_contiguous_1(up) || !ggml_is_contiguous_1(op)) {
        return false;
    }

    bool match = false;
    {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        match = gate == ctx->last_fused_gate_op && up == ctx->last_fused_up_op;
    }
    if (!match) {
        if (ith == 0 && moe_name_has(op, "ffn_moe_swiglu")) {
            ctx->fused_swiglu_misses.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }

    if (ctx->params.fuse_expert_ffn && ctx->params.fuse_direct_swiglu && moe_name_has(op, "ffn_moe_swiglu")) {
        return true;
    }

    if (ctx->params.fuse_direct_swiglu) {
        if (llama_moe_buffer_swiglu_direct_compute(ctx, op, gate, up, ith, nth)) {
            llama_moe_buffer_fused_ffn_oracle_after_swiglu(ctx, op, gate, up, ith, nth);
            if (ith == 0) {
                ctx->fused_swiglu_hits.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
        if (ith == 0) {
            ctx->fused_direct_swiglu_misses.fetch_add(1, std::memory_order_relaxed);
        }
        tls_mwq_keep_sum_cache = false;
        const bool ok_up = llama_moe_buffer_mul_mat_id_compute(up, ith, nth, ctx);
        tls_mwq_keep_sum_cache = true;
        const bool ok_gate = llama_moe_buffer_mul_mat_id_compute(gate, ith, nth, ctx);
        tls_mwq_keep_sum_cache = false;
        if (!ok_up || !ok_gate) {
            return false;
        }
    }

    const int64_t nc = gate->ne[0];
    const int64_t nr = ggml_nrows(gate);
    if (op->ne[0] != nc || ggml_nrows(op) != nr || nc > INT_MAX) {
        return false;
    }

    const int64_t dr = (nr + nth - 1) / nth;
    const int64_t ir0 = dr * ith;
    const int64_t ir1 = std::min<int64_t>(ir0 + dr, nr);
    for (int64_t row = ir0; row < ir1; ++row) {
        const float * gate_p = (const float *) ((const char *) gate->data + row * gate->nb[1]);
        const float * up_p   = (const float *) ((const char *) up->data   + row * up->nb[1]);
        float * dst_p        = (float *)       ((char *) op->data         + row * op->nb[1]);
        moe_vec_swiglu_f32((int) nc, dst_p, gate_p, up_p);
    }

    llama_moe_buffer_fused_ffn_oracle_after_swiglu(ctx, op, gate, up, ith, nth);

    if (ith == 0) {
        ctx->fused_swiglu_hits.fetch_add(1, std::memory_order_relaxed);
        ctx->fused_swiglu_rows.fetch_add((uint64_t) nr, std::memory_order_relaxed);
    }
    return true;
}

bool llama_moe_buffer_mul_mat_id_callback(ggml_tensor * op, int ith, int nth, void * user_data) {
    auto * ctx = static_cast<llama_moe_buffer_context *>(user_data);
    if (ctx == nullptr || op == nullptr) {
        return false;
    }
    if (op->op == GGML_OP_GLU) {
        return llama_moe_buffer_swiglu_callback(ctx, op, ith, nth);
    }
    if (op->op != GGML_OP_MUL_MAT_ID || op->src[0] == nullptr) {
        return false;
    }

    if (ctx->params.fuse_expert_ffn && moe_name_has(op->src[0], ".ffn_down_exps.weight")) {
        ggml_tensor * swiglu_op = op->src[1];
        ggml_tensor * gate_op = swiglu_op != nullptr ? swiglu_op->src[0] : nullptr;
        ggml_tensor * up_op   = swiglu_op != nullptr ? swiglu_op->src[1] : nullptr;
        if (llama_moe_buffer_fused_ffn_compute(ctx, op, swiglu_op, gate_op, up_op, ith, nth)) {
            return true;
        }
        if (ith == 0) {
            ctx->fused_ffn_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
        if (swiglu_op != nullptr && swiglu_op->op == GGML_OP_GLU && gate_op != nullptr && up_op != nullptr &&
                !llama_moe_buffer_swiglu_materialize(ctx, swiglu_op, gate_op, up_op, ith, nth)) {
            return false;
        }
        const bool ok = llama_moe_buffer_mul_mat_id_compute(op, ith, nth, user_data);
        if (ok) {
            llama_moe_buffer_fused_ffn_oracle_after_down(ctx, op, ith, nth);
        }
        return ok;
    }

    if (!ctx->params.fuse_gate_up) {
        const bool ok = llama_moe_buffer_mul_mat_id_compute(op, ith, nth, user_data);
        if (ok) {
            llama_moe_buffer_fused_ffn_oracle_after_down(ctx, op, ith, nth);
        }
        return ok;
    }

    ggml_tensor * src0 = op->src[0];
    auto it = ctx->by_name.find(ggml_get_name(src0));
    if (it == ctx->by_name.end()) {
        return false;
    }
    moe_managed & m = it->second;

    const bool is_up = moe_name_has(src0, ".ffn_up_exps.weight");
    const bool is_gate = moe_name_has(src0, ".ffn_gate_exps.weight");
    if (!is_up && !is_gate) {
        const bool ok = llama_moe_buffer_mul_mat_id_compute(op, ith, nth, user_data);
        if (ok) {
            llama_moe_buffer_fused_ffn_oracle_after_down(ctx, op, ith, nth);
        }
        return ok;
    }

    ggml_tensor * up_op = nullptr;
    ggml_tensor * gate_op = nullptr;
    const int layer = moe_layer_from_name(src0);

    if (ctx->params.fuse_direct_swiglu) {
        ggml_tensor * ids = op->src[2];
        const std::string sibling_name = moe_sibling_exps_name(ggml_get_name(src0), is_up);
        auto sibling_it = ctx->by_name.find(sibling_name);
        if (sibling_it == ctx->by_name.end() ||
                !moe_op_all_mwq_available(*ctx, m, ids) ||
                !moe_op_all_mwq_available(*ctx, sibling_it->second, ids)) {
            if (ith == 0) {
                ctx->fused_gate_up_misses.fetch_add(1, std::memory_order_relaxed);
            }
            return false;
        }
    }

    auto active_matches = [&]() {
        return ctx->fused_layer == layer &&
            (ctx->fused_up_op == op || ctx->fused_gate_op == op);
    };

    {
        std::unique_lock<std::mutex> lk(ctx->mtx);
        if (ith == 0) {
            if (is_up) {
                if (ctx->pending_gate_op != nullptr && ctx->pending_gate_layer == layer) {
                    ctx->fused_up_op = op;
                    ctx->fused_gate_op = ctx->pending_gate_op;
                    ctx->fused_layer = layer;
                    ctx->pending_gate_op = nullptr;
                    ctx->pending_gate_src1 = nullptr;
                    ctx->pending_gate_ids = nullptr;
                    ctx->pending_gate_layer = -1;
                    up_op = ctx->fused_up_op;
                    gate_op = ctx->fused_gate_op;
                    ctx->cv_fuse.notify_all();
                } else {
                    ctx->pending_up_op = op;
                    ctx->pending_up_src1 = op->src[1];
                    ctx->pending_up_ids = op->src[2];
                    ctx->pending_up_layer = layer;
                    ctx->fused_gate_up_deferred.fetch_add(1, std::memory_order_relaxed);
                    ctx->cv_fuse.notify_all();
                    return true;
                }
            } else {
                if (ctx->pending_up_op != nullptr && ctx->pending_up_layer == layer) {
                    ctx->fused_up_op = ctx->pending_up_op;
                    ctx->fused_gate_op = op;
                    ctx->fused_layer = layer;
                    ctx->pending_up_op = nullptr;
                    ctx->pending_up_src1 = nullptr;
                    ctx->pending_up_ids = nullptr;
                    ctx->pending_up_layer = -1;
                    up_op = ctx->fused_up_op;
                    gate_op = ctx->fused_gate_op;
                    ctx->cv_fuse.notify_all();
                } else {
                    ctx->pending_gate_op = op;
                    ctx->pending_gate_src1 = op->src[1];
                    ctx->pending_gate_ids = op->src[2];
                    ctx->pending_gate_layer = layer;
                    ctx->fused_gate_up_deferred.fetch_add(1, std::memory_order_relaxed);
                    ctx->cv_fuse.notify_all();
                    return true;
                }
            }
        } else {
            const bool second_half_wait =
                (is_up   && ctx->pending_gate_op != nullptr && ctx->pending_gate_layer == layer) ||
                (is_gate && ctx->pending_up_op   != nullptr && ctx->pending_up_layer   == layer);
            if (second_half_wait) {
                ctx->cv_fuse.wait_for(lk, std::chrono::milliseconds(100), active_matches);
            }
            if (active_matches()) {
                up_op = ctx->fused_up_op;
                gate_op = ctx->fused_gate_op;
            } else {
                return true;
            }
        }
    }

    if (ctx->params.fuse_direct_swiglu) {
        if (ith == 0) {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            ctx->last_fused_up_op = up_op;
            ctx->last_fused_gate_op = gate_op;
            ctx->fused_gate_up_hits.fetch_add(1, std::memory_order_relaxed);
        }
        return true;
    }

    tls_mwq_keep_sum_cache = false;
    const bool ok_up = llama_moe_buffer_mul_mat_id_compute(up_op, ith, nth, user_data);
    tls_mwq_keep_sum_cache = true;
    const bool ok_gate = llama_moe_buffer_mul_mat_id_compute(gate_op, ith, nth, user_data);
    tls_mwq_keep_sum_cache = false;
    if (ith == 0 && ok_up && ok_gate) {
        std::lock_guard<std::mutex> lk(ctx->mtx);
        ctx->last_fused_up_op = up_op;
        ctx->last_fused_gate_op = gate_op;
        ctx->fused_gate_up_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return ok_up && ok_gate;
}

static int moe_eam_topn() {
    const char * v = std::getenv("LLAMA_LAZY_MOE_EAM_TOPN");
    return v == nullptr || v[0] == '\0' ? 12 : std::max(0, std::atoi(v));
}

static void moe_print_eam_top_impl(const llama_moe_buffer_context & ctx, const char * prefix) {
    const int topn = moe_eam_topn();
    if (topn <= 0 || (ctx.eam_access.load(std::memory_order_relaxed) == 0 &&
                ctx.eam_future_hints.load(std::memory_order_relaxed) == 0)) {
        return;
    }

    struct eam_candidate {
        const moe_group_state * g = nullptr;
        size_t resident_bytes = 0;
        double bit_score = 0.0;
    };
    std::vector<eam_candidate> candidates;
    candidates.reserve(ctx.groups.size());
    for (const auto & kv : ctx.groups) {
        const moe_group_state & g = kv.second;
        if (g.seq_access == 0 && g.seq_future_hints == 0 && g.seq_prefetch_queued == 0) {
            continue;
        }
        candidates.push_back({&g, moe_group_resident_bytes(ctx, g.layer, g.expert),
                moe_group_resident_bit_score(ctx, g.layer, g.expert)});
    }
    std::sort(candidates.begin(), candidates.end(), [](const eam_candidate & a, const eam_candidate & b) {
        const moe_group_state & ga = *a.g;
        const moe_group_state & gb = *b.g;
        if (ga.seq_access != gb.seq_access) {
            return ga.seq_access > gb.seq_access;
        }
        if (ga.seq_rank0_access != gb.seq_rank0_access) {
            return ga.seq_rank0_access > gb.seq_rank0_access;
        }
        if (ga.seq_prefetch_hits != gb.seq_prefetch_hits) {
            return ga.seq_prefetch_hits > gb.seq_prefetch_hits;
        }
        if (ga.seq_future_hints != gb.seq_future_hints) {
            return ga.seq_future_hints > gb.seq_future_hints;
        }
        if (ga.layer != gb.layer) {
            return ga.layer < gb.layer;
        }
        return ga.expert < gb.expert;
    });

    const uint64_t total = std::max<uint64_t>(1, ctx.eam_access.load(std::memory_order_relaxed));
    const int limit = std::min<int>(topn, (int) candidates.size());
    for (int i = 0; i < limit; ++i) {
        const moe_group_state & g = *candidates[i].g;
        const uint64_t age = g.seq_last_token_epoch == 0 || ctx.profile_token_epoch < g.seq_last_token_epoch ?
            0 : ctx.profile_token_epoch - g.seq_last_token_epoch;
        const double rate = (double) g.seq_access / (double) total;
        const double avg_hint_score = g.seq_future_hints > 0 ?
            g.seq_future_score_sum / (double) g.seq_future_hints : 0.0;
        const double avg_pred_score = g.seq_predicted > 0 ?
            g.seq_predict_score_sum / (double) g.seq_predicted : 0.0;
        std::fprintf(stderr,
                "%s[eam_top]: #%d layer=%d expert=%d access=%llu rank0=%llu rate=%.6f "
                "token={first:%llu,last:%llu,age:%llu} future={hints:%llu,rank0:%llu,avg_score:%.4f} "
                "predict={picked:%llu,enq:%llu,avg_score:%.4f} "
                "prefetch={queued:%llu,hit:%llu,late:%llu,unused:%llu} cache={hit:%llu,miss:%llu} "
                "tensor={other:%llu,gate:%llu,up:%llu,down:%llu} resident=%.1f MiB bit=%.2f pinned=%d active=%d cache_score=%.3f\n",
                prefix, i + 1, g.layer, g.expert,
                (unsigned long long) g.seq_access,
                (unsigned long long) g.seq_rank0_access,
                rate,
                (unsigned long long) g.seq_first_token_epoch,
                (unsigned long long) g.seq_last_token_epoch,
                (unsigned long long) age,
                (unsigned long long) g.seq_future_hints,
                (unsigned long long) g.seq_future_rank0_hints,
                avg_hint_score,
                (unsigned long long) g.seq_predicted,
                (unsigned long long) g.seq_pred_enqueued,
                avg_pred_score,
                (unsigned long long) g.seq_prefetch_queued,
                (unsigned long long) g.seq_prefetch_hits,
                (unsigned long long) g.seq_prefetch_late,
                (unsigned long long) g.seq_prefetch_unused,
                (unsigned long long) g.seq_cache_hits,
                (unsigned long long) g.seq_cache_misses,
                (unsigned long long) g.seq_tensor_access[0],
                (unsigned long long) g.seq_tensor_access[1],
                (unsigned long long) g.seq_tensor_access[2],
                (unsigned long long) g.seq_tensor_access[3],
                candidates[i].resident_bytes / 1048576.0,
                candidates[i].bit_score,
                g.pinned ? 1 : 0,
                moe_group_is_active(ctx, g) ? 1 : 0,
                g.hot_score);
    }
}

static void moe_json_write_escaped(FILE * f, const char * s) {
    std::fputc('"', f);
    if (s != nullptr) {
        for (const unsigned char * p = (const unsigned char *) s; *p != '\0'; ++p) {
            switch (*p) {
                case '\\': std::fputs("\\\\", f); break;
                case '"':  std::fputs("\\\"", f); break;
                case '\b': std::fputs("\\b", f); break;
                case '\f': std::fputs("\\f", f); break;
                case '\n': std::fputs("\\n", f); break;
                case '\r': std::fputs("\\r", f); break;
                case '\t': std::fputs("\\t", f); break;
                default:
                    if (*p < 0x20) {
                        std::fprintf(f, "\\u%04x", (unsigned) *p);
                    } else {
                        std::fputc(*p, f);
                    }
                    break;
            }
        }
    }
    std::fputc('"', f);
}

static const char * moe_env_str(const char * name, const char * def = "") {
    const char * v = std::getenv(name);
    return v == nullptr ? def : v;
}

static void moe_eam_trace_append_locked(const llama_moe_buffer_context & ctx) {
    const char * path = std::getenv("LLAMA_LAZY_MOE_EAM_TRACE");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    FILE * f = std::fopen(path, "ab");
    if (f == nullptr) {
        if (ctx.params.debug_log) {
            std::fprintf(stderr, "llama_moe_buffer: failed to append EAM trace %s\n", path);
        }
        return;
    }

    struct trace_item {
        const moe_group_state * g = nullptr;
    };
    std::vector<trace_item> items;
    items.reserve(ctx.groups.size());
    for (const auto & kv : ctx.groups) {
        const moe_group_state & g = kv.second;
        if (g.seq_access == 0) {
            continue;
        }
        items.push_back({&g});
    }
    std::sort(items.begin(), items.end(), [](const trace_item & a, const trace_item & b) {
        if (a.g->layer != b.g->layer) {
            return a.g->layer < b.g->layer;
        }
        return a.g->expert < b.g->expert;
    });

    std::fputc('{', f);
    std::fputs("\"request_id\":", f);
    moe_json_write_escaped(f, moe_env_str("LLAMA_LAZY_MOE_EAM_TRACE_REQUEST_ID", "unknown"));
    std::fputs(",\"workload_index\":", f);
    std::fputs(moe_env_str("LLAMA_LAZY_MOE_EAM_TRACE_WORKLOAD_INDEX", "-1"), f);
    std::fputs(",\"prompt_file\":", f);
    moe_json_write_escaped(f, moe_env_str("LLAMA_LAZY_MOE_EAM_TRACE_PROMPT_FILE", ""));
    std::fputs(",\"prompt_length_bucket\":", f);
    moe_json_write_escaped(f, moe_env_str("LLAMA_LAZY_MOE_EAM_TRACE_BUCKET", "unknown"));
    std::fputs(",\"prompt_chars\":", f);
    std::fputs(moe_env_str("LLAMA_LAZY_MOE_EAM_TRACE_CHARS", "0"), f);
    std::fputs(",\"decode_tokens\":", f);
    std::fputs(moe_env_str("LLAMA_LAZY_MOE_EAM_TRACE_DECODE_TOKENS", "0"), f);
    std::fprintf(f,
            ",\"token_epoch\":%llu,\"total_access\":%llu,\"rank0_access\":%llu,"
            "\"prefetch\":{\"queued\":%llu,\"hit\":%llu,\"late\":%llu,\"unused\":%llu},"
            "\"cache\":{\"hit\":%llu,\"miss\":%llu},\"experts\":[",
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.eam_access.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_rank0_access.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_prefetch_queued.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_prefetch_hits.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_prefetch_late.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_prefetch_unused.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_cache_hits.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_cache_misses.load(std::memory_order_relaxed));
    for (size_t i = 0; i < items.size(); ++i) {
        const moe_group_state & g = *items[i].g;
        if (i > 0) {
            std::fputc(',', f);
        }
        std::fprintf(f,
                "{\"layer\":%d,\"expert\":%d,\"count\":%llu,\"rank0\":%llu,"
                "\"first_token\":%llu,\"last_token\":%llu,"
                "\"prefetch_queued\":%llu,\"prefetch_hit\":%llu,\"prefetch_unused\":%llu,"
                "\"cache_hit\":%llu,\"cache_miss\":%llu,"
                "\"tensor\":[%llu,%llu,%llu,%llu]}",
                g.layer, g.expert,
                (unsigned long long) g.seq_access,
                (unsigned long long) g.seq_rank0_access,
                (unsigned long long) g.seq_first_token_epoch,
                (unsigned long long) g.seq_last_token_epoch,
                (unsigned long long) g.seq_prefetch_queued,
                (unsigned long long) g.seq_prefetch_hits,
                (unsigned long long) g.seq_prefetch_unused,
                (unsigned long long) g.seq_cache_hits,
                (unsigned long long) g.seq_cache_misses,
                (unsigned long long) g.seq_tensor_access[0],
                (unsigned long long) g.seq_tensor_access[1],
                (unsigned long long) g.seq_tensor_access[2],
                (unsigned long long) g.seq_tensor_access[3]);
    }
    std::fputs("]}\n", f);
    std::fclose(f);
}

static void moe_print_stats_impl(const llama_moe_buffer_context & ctx, const char * prefix) {
    const double lru_bytes = (double) (ctx.resident_bytes > ctx.pinned_resident_bytes.load(std::memory_order_relaxed) ?
            ctx.resident_bytes - ctx.pinned_resident_bytes.load(std::memory_order_relaxed) : 0);
    std::fprintf(stderr,
            "%s: tensors=%zu resident=%.1f MiB pinned=%.1f MiB lru=%.1f MiB streams=%llu hits=%llu evictions=%llu "
            "enqueued=%llu dup=%llu dyn_bits={2:%llu,3:%llu,4:%llu,other:%llu} "
            "sidecar=%llu/%llu/%.1f MiB cache=%llu/%llu/%llu/%.1f MiB mwq_kernel=%llu/%llu/%.1f MiB group=%llu/%llu "
            "optok=%llu/%llu compat=%llu sumcache=%llu/%llu countsort=%llu avx512_q2=%llu/%llu/%llu dot=%llu/%llu/%llu "
            "vnni_q2=%llu/%llu/%llu dot=%llu/%llu/%llu qx=%llu/%llu "
            "fuse_gu=%llu/%llu/%llu fuse_swiglu=%llu/%llu/%llu direct_swiglu=%llu/%llu/%llu down_prefetch=%llu "
            "fuse_ffn=%llu/%llu runs=%llu pairs=%llu tiles=%llu fallback=%llu "
            "egroup_touch=%llu egroup_evict=%llu/%llu pinned_hit=%llu active_hit=%llu belady=%llu lru_fb=%llu "
            "eam={access:%llu,rank0:%llu,future:%llu,queued:%llu,prefetch_hit:%llu,late:%llu,unused:%llu,cache_hit:%llu,cache_miss:%llu} "
            "eam_admit={ok:%llu,drop:%llu,budget:%llu,distance:%llu,score:%llu,active:%llu,pinned:%llu,recent:%llu,avg_ok:%.3f,avg_drop:%.3f,spec_peak:%.1f MiB} "
            "eam_predict={runs:%llu,cand:%llu,enq:%llu,drop:%llu,avg_score:%.3f} "
            "eamc={snap:%llu,stored:%zu,match:%llu,hit:%llu,miss:%llu,active:%zu,avg_sim:%.3f,prior:%llu} "
            "eam_evict={spec_unused:%llu,low:%llu,far:%llu,relaxed:%llu,protect_active:%llu,pinned:%llu,recent:%llu,high:%llu,early:%llu} "
            "dyn_eff=%.1f MiB dyn_saved=%.1f MiB wait=%llu/%.1f ms mwq_actual_saved=%.1f/%.1f MiB "
            "stream_fail_total=%llu direct_swiglu_fail={precheck:%llu,byname:%llu,entry_missing:%llu,read:%llu,range:%llu,kernelpair:%llu,native_prep:%llu,other:%llu}\n",
            prefix,
            ctx.by_name.size(), ctx.resident_bytes / 1048576.0,
            ctx.pinned_resident_bytes.load(std::memory_order_relaxed) / 1048576.0,
            lru_bytes / 1048576.0,
            (unsigned long long) ctx.streams.load(), (unsigned long long) ctx.hits.load(),
            (unsigned long long) ctx.evictions.load(), (unsigned long long) ctx.enqueued.load(),
            (unsigned long long) ctx.queue_dups.load(),
            (unsigned long long) ctx.dyn_bit2.load(),
            (unsigned long long) ctx.dyn_bit3.load(),
            (unsigned long long) ctx.dyn_bit4.load(),
            (unsigned long long) ctx.dyn_bit_other.load(),
            (unsigned long long) ctx.sidecar_hits.load(),
            (unsigned long long) ctx.sidecar_misses.load(),
            ctx.sidecar_bytes.load() / 1048576.0,
            (unsigned long long) ctx.cache_hits.load(),
            (unsigned long long) ctx.cache_misses.load(),
            (unsigned long long) ctx.cache_writes.load(),
            ctx.cache_bytes.load() / 1048576.0,
            (unsigned long long) ctx.mwq_kernel_ops.load(),
            (unsigned long long) ctx.mwq_kernel_slices.load(),
            ctx.mwq_bytes_read.load() / 1048576.0,
            (unsigned long long) ctx.mwq_group_batches.load(),
            (unsigned long long) ctx.mwq_group_pairs.load(),
            (unsigned long long) ctx.mwq_op_multi_token.load(),
            (unsigned long long) ctx.mwq_op_tokens.load(),
            (unsigned long long) ctx.mwq_compat_hits.load(),
            (unsigned long long) ctx.mwq_sum_cache_hits.load(),
            (unsigned long long) ctx.mwq_sum_cache_builds.load(),
            (unsigned long long) ctx.mwq_countsort_ops.load(),
            (unsigned long long) ctx.mwq_avx512_q2_swiglu_calls.load(),
            (unsigned long long) ctx.mwq_avx512_q2_swiglu_pairs.load(),
            (unsigned long long) ctx.mwq_avx512_q2_swiglu_rows.load(),
            (unsigned long long) ctx.mwq_avx512_q2_dot_calls.load(),
            (unsigned long long) ctx.mwq_avx512_q2_dot_pairs.load(),
            (unsigned long long) ctx.mwq_avx512_q2_dot_rows.load(),
            (unsigned long long) ctx.mwq_vnni_q2_swiglu_calls.load(),
            (unsigned long long) ctx.mwq_vnni_q2_swiglu_pairs.load(),
            (unsigned long long) ctx.mwq_vnni_q2_swiglu_rows.load(),
            (unsigned long long) ctx.mwq_vnni_q2_dot_calls.load(),
            (unsigned long long) ctx.mwq_vnni_q2_dot_pairs.load(),
            (unsigned long long) ctx.mwq_vnni_q2_dot_rows.load(),
            (unsigned long long) ctx.mwq_vnni_qx_hits.load(),
            (unsigned long long) ctx.mwq_vnni_qx_builds.load(),
            (unsigned long long) ctx.fused_gate_up_hits.load(),
            (unsigned long long) ctx.fused_gate_up_deferred.load(),
            (unsigned long long) ctx.fused_gate_up_misses.load(),
            (unsigned long long) ctx.fused_swiglu_hits.load(),
            (unsigned long long) ctx.fused_swiglu_misses.load(),
            (unsigned long long) ctx.fused_swiglu_rows.load(),
            (unsigned long long) ctx.fused_direct_swiglu_hits.load(),
            (unsigned long long) ctx.fused_direct_swiglu_misses.load(),
            (unsigned long long) ctx.fused_direct_swiglu_rows.load(),
            (unsigned long long) ctx.fused_direct_down_prefetch.load(),
            (unsigned long long) ctx.fused_ffn_hits.load(),
            (unsigned long long) ctx.fused_ffn_misses.load(),
            (unsigned long long) ctx.fused_ffn_runs.load(),
            (unsigned long long) ctx.fused_ffn_pairs.load(),
            (unsigned long long) ctx.fused_ffn_tiles.load(),
            (unsigned long long) ctx.fused_ffn_fallbacks.load(),
            (unsigned long long) ctx.group_touches.load(),
            (unsigned long long) ctx.group_evictions.load(),
            (unsigned long long) ctx.group_evicted_slices.load(),
            (unsigned long long) ctx.pinned_hits.load(),
            (unsigned long long) ctx.active_hits.load(),
            (unsigned long long) ctx.belady_evictions.load(),
            (unsigned long long) ctx.lru_fallback_evictions.load(),
            (unsigned long long) ctx.eam_access.load(),
            (unsigned long long) ctx.eam_rank0_access.load(),
            (unsigned long long) ctx.eam_future_hints.load(),
            (unsigned long long) ctx.eam_prefetch_queued.load(),
            (unsigned long long) ctx.eam_prefetch_hits.load(),
            (unsigned long long) ctx.eam_prefetch_late.load(),
            (unsigned long long) ctx.eam_prefetch_unused.load(),
            (unsigned long long) ctx.eam_cache_hits.load(),
            (unsigned long long) ctx.eam_cache_misses.load(),
            (unsigned long long) ctx.eam_prefetch_admit.load(),
            (unsigned long long) ctx.eam_prefetch_drop.load(),
            (unsigned long long) ctx.eam_prefetch_drop_budget.load(),
            (unsigned long long) ctx.eam_prefetch_drop_distance.load(),
            (unsigned long long) ctx.eam_prefetch_drop_score.load(),
            (unsigned long long) ctx.eam_prefetch_drop_active.load(),
            (unsigned long long) ctx.eam_prefetch_drop_pinned.load(),
            (unsigned long long) ctx.eam_prefetch_drop_recent.load(),
            ctx.eam_prefetch_admit.load() > 0 ?
                (double) ctx.eam_prefetch_admit_score_x1000.load() / 1000.0 / (double) ctx.eam_prefetch_admit.load() : 0.0,
            ctx.eam_prefetch_drop.load() > 0 ?
                (double) ctx.eam_prefetch_drop_score_x1000.load() / 1000.0 / (double) ctx.eam_prefetch_drop.load() : 0.0,
            ctx.eam_speculative_bytes_peak.load() / 1048576.0,
            (unsigned long long) ctx.eam_predict_runs.load(),
            (unsigned long long) ctx.eam_predict_candidates.load(),
            (unsigned long long) ctx.eam_predict_enqueued.load(),
            (unsigned long long) ctx.eam_predict_dropped.load(),
            ctx.eam_predict_candidates.load() > 0 ?
                (double) ctx.eam_predict_score_x1000.load() / 1000.0 / (double) ctx.eam_predict_candidates.load() : 0.0,
            (unsigned long long) ctx.eamc_snapshot_count.load(),
            ctx.eamc_snapshots.size(),
            (unsigned long long) ctx.eamc_match_runs.load(),
            (unsigned long long) ctx.eamc_match_hits.load(),
            (unsigned long long) ctx.eamc_match_misses.load(),
            ctx.eamc_active_matches.size(),
            ctx.eamc_match_hits.load() > 0 ?
                (double) ctx.eamc_match_score_x1000.load() / 1000.0 / (double) ctx.eamc_match_hits.load() : 0.0,
            (unsigned long long) ctx.eamc_prior_hits.load(),
            (unsigned long long) ctx.eam_evict_spec_unused.load(),
            (unsigned long long) ctx.eam_evict_low_score.load(),
            (unsigned long long) ctx.eam_evict_far_future.load(),
            (unsigned long long) ctx.eam_evict_lru_relaxed.load(),
            (unsigned long long) ctx.eam_evict_protect_active.load(),
            (unsigned long long) ctx.eam_evict_protect_pinned.load(),
            (unsigned long long) ctx.eam_evict_protect_recent.load(),
            (unsigned long long) ctx.eam_evict_protect_high.load(),
            (unsigned long long) ctx.eam_evict_protect_early.load(),
            ctx.dyn_effective_bytes.load() / 1048576.0,
            ctx.dyn_saved_bytes.load() / 1048576.0,
            (unsigned long long) ctx.stream_wait_count.load(),
            (double) ctx.stream_wait_ns.load() / 1.0e6,
            ctx.mwq_actual_saved_bytes.load() / 1048576.0,
            ctx.mwq_actual_full_bytes.load() / 1048576.0,
            (unsigned long long) ctx.mwq_stream_fail_total.load(),
            (unsigned long long) ctx.direct_swiglu_fail_precheck.load(),
            (unsigned long long) ctx.direct_swiglu_fail_byname.load(),
            (unsigned long long) ctx.direct_swiglu_fail_entry_missing.load(),
            (unsigned long long) ctx.direct_swiglu_fail_read.load(),
            (unsigned long long) ctx.direct_swiglu_fail_range.load(),
            (unsigned long long) ctx.direct_swiglu_fail_kernelpair.load(),
            (unsigned long long) ctx.direct_swiglu_fail_native_prepare.load(),
            (unsigned long long) ctx.direct_swiglu_fail_other.load());
    moe_print_eam_top_impl(ctx, prefix);
    if (ctx.profile) {
        const uint64_t wait_us = ctx.stream_wait_ns.load(std::memory_order_relaxed) / 1000;
        std::fprintf(stderr,
                "llama_moe_profile: tokens=%llu token_total_us=%llu attention_us=%llu moe_total_us=%llu "
                "route_us=%llu cache_lookup_us=%llu victim_select_us=%llu "
                "sidecar_submit_us=%llu sidecar_wait_us=%llu sidecar_read_us=%llu "
                "sidecar_bytes=%llu sidecar_read_count=%llu "
                "q2_unpack_us=%llu gate_dot_us=%llu up_dot_us=%llu swiglu_us=%llu down_dot_us=%llu "
                "graph_dispatch_us=%llu worker_wait_us=%llu barrier_us=%llu "
                "cache_hit=%llu cache_miss=%llu prefetch_hit=%llu prefetch_late=%llu prefetch_unused=%llu "
                "evict_clean=%llu evict_active_window=%llu evict_reloaded_within_1_token=%llu "
                "evict_reloaded_within_4_tokens=%llu\n",
                (unsigned long long) ctx.profile_token_epoch,
                (unsigned long long) ctx.prof_token_total_us.load(),
                (unsigned long long) ctx.prof_attention_us.load(),
                (unsigned long long) ctx.prof_moe_total_us.load(),
                (unsigned long long) ctx.prof_route_us.load(),
                (unsigned long long) ctx.prof_cache_lookup_us.load(),
                (unsigned long long) ctx.prof_victim_select_us.load(),
                (unsigned long long) ctx.prof_sidecar_submit_us.load(),
                (unsigned long long) wait_us,
                (unsigned long long) ctx.prof_sidecar_read_us.load(),
                (unsigned long long) ctx.prof_sidecar_bytes.load(),
                (unsigned long long) ctx.prof_sidecar_read_count.load(),
                (unsigned long long) ctx.prof_q2_unpack_us.load(),
                (unsigned long long) ctx.prof_gate_dot_us.load(),
                (unsigned long long) ctx.prof_up_dot_us.load(),
                (unsigned long long) ctx.prof_swiglu_us.load(),
                (unsigned long long) ctx.prof_down_dot_us.load(),
                (unsigned long long) ctx.prof_graph_dispatch_us.load(),
                (unsigned long long) ctx.prof_worker_wait_us.load(),
                (unsigned long long) ctx.prof_barrier_us.load(),
                (unsigned long long) ctx.prof_cache_hit.load(),
                (unsigned long long) ctx.prof_cache_miss.load(),
                (unsigned long long) ctx.prof_prefetch_hit.load(),
                (unsigned long long) ctx.prof_prefetch_late.load(),
                (unsigned long long) ctx.prof_prefetch_unused.load(),
                (unsigned long long) ctx.prof_evict_clean.load(),
                (unsigned long long) ctx.prof_evict_active_window.load(),
                (unsigned long long) ctx.prof_evict_reloaded_within_1_token.load(),
                (unsigned long long) ctx.prof_evict_reloaded_within_4_tokens.load());
    }
}

void llama_moe_buffer_print_stats(const llama_moe_buffer_context & ctx) {
    moe_print_stats_impl(ctx, "llama_moe_buffer");
}
