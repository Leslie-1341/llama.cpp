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
#include <unordered_set>
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

enum moe_evict_reason_code : uint8_t {
    MOE_EVICT_REASON_UNKNOWN      = 0,
    MOE_EVICT_REASON_SPEC_UNUSED  = 1,
    MOE_EVICT_REASON_LOW_SCORE    = 2,
    MOE_EVICT_REASON_FAR_FUTURE   = 3,
    MOE_EVICT_REASON_LAYER_WINDOW = 4,
    MOE_EVICT_REASON_REUSE_LOW    = 5,
    MOE_EVICT_REASON_RELAXED      = 6,
    MOE_EVICT_REASON_EAM_REPLACE  = 7,
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
    std::vector<char> demand_async_loaded; // demand load completed; first touch remains a cache miss
    std::vector<uint64_t> last_evicted_token;
    std::vector<char> queued;           // async prefetch already queued
    std::vector<int>  queued_bits;       // highest precision still pending for this expert
    std::vector<uint32_t> activation;   // per-expert touch count (for hot pinning)
    uint64_t      total_act = 0;        // sum of activation[] (denominator for hot ratio)
    std::vector<std::list<std::pair<moe_managed *, int>>::iterator> lru_pos;
};

struct moe_prefetch_task {
    moe_managed * m = nullptr;
    bool is_group = false;
    bool is_demand = false;
    int layer = -1;
    int skip_kind = -1;
    int primary_kind = -1;
    int e = -1;
    int rank = 0;
    int priority = 0;
    int target_bits = 0;
    float score = 0.0f;
    uint64_t seq = 0;
    uint64_t generation = 0;
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
    bool group_lru_linked = false;
    std::list<uint64_t>::iterator group_lru_pos;
    size_t resident_bytes_cached = 0;
    uint64_t resident_bit_weight_cached = 0;
    uint8_t resident_slices_cached = 0;
    uint64_t evict_scan_generation = 0;
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

    // Online short-term reuse estimator. This is learned from the current
    // request as it runs; trace files are only used offline to validate it.
    double   reuse_ema = 0.0;
    double   inter_token_gap_ema = 16.0;
    uint64_t reuse_observed = 0;
    uint64_t reuse_within_1 = 0;
    uint64_t reuse_within_4 = 0;
    uint64_t reuse_within_16 = 0;
    uint64_t last_evicted_token_epoch = 0;
    bool     last_evicted_valid = false;
    uint8_t  last_evicted_reason = MOE_EVICT_REASON_UNKNOWN;
    uint64_t layer_window_bad_reload = 0;
    uint64_t layer_window_last_bad_reload_token = 0;
    double   bad_reload_score = 0.0;
    uint64_t bad_reload_last_token = 0;
    uint64_t bad_reload_1 = 0;
    uint64_t bad_reload_4 = 0;
    uint64_t bad_reload_16 = 0;
    uint64_t bad_reload_protect_hits = 0;
    uint64_t bad_reload_soft_hits = 0;
    double   ghost_reload_risk_ema = 0.0;
    uint64_t ghost_generation = 0;
    uint64_t ghost_outcomes = 0;
    uint64_t ghost_bad_1 = 0;
    uint64_t ghost_bad_4 = 0;
    uint64_t ghost_bad_16 = 0;
    uint64_t ghost_success = 0;
    uint64_t ghost_last_outcome_token = 0;
    bool     ghost_pending = false;
    uint64_t demand_async_generation = 0;
    bool     demand_async_pending = false;
    uint64_t demand_admission_token = UINT64_MAX;
    bool     demand_admission_decided = false;
    bool     demand_admission_bypass = false;
    uint64_t admission_outcome_generation = 0;
    uint64_t admission_outcome_token = 0;
    uint64_t admission_outcome_deadline = 0;
    uint8_t  admission_outcome_action = 0; // 1=admit, 2=bypass
    bool     admission_outcome_pending = false;
    size_t   admission_staging_reserved_bytes = 0;
    double   admission_outcome_candidate = 0.0;
    double   admission_outcome_victim = 0.0;
    int      admission_outcome_victim_layer = -1;
    int      admission_outcome_victim_expert = -1;
    int      admission_outcome_victim_groups = 0;
    size_t   admission_outcome_victim_bytes = 0;
    double   admission_feedback_ema = 0.0;
    uint64_t admission_feedback_samples = 0;
    double   admission_regret_ema = 0.0;
    uint64_t admission_regret_samples = 0;
    double   eam_replace_pred = 0.0;
    double   eam_replace_mass = 0.0;
    double   eam_replace_keep = 0.0;
    uint8_t  cct_conf = 0;
    uint64_t cct_protect_hits = 0;

    // Same-layer next-token resident predictor. This is intentionally a
    // retention-only signal: it protects resident groups from layer-done
    // eviction, but it does not enqueue speculative reads.
    uint8_t  next_token_conf = 0;
    uint64_t next_token_protect_until = 0;
    uint64_t next_token_predicted = 0;
    uint64_t next_token_hit = 0;
    uint64_t next_token_miss = 0;
    uint64_t next_token_protect_hits = 0;
};

struct moe_cct_entry {
    uint8_t  conf = 0; // 0..3 saturating counter, branch-predictor style
    uint64_t hit = 0;
    uint64_t miss = 0;
    uint64_t unused = 0;
    uint64_t replace = 0;
    uint64_t last_update_token = 0;
};

struct moe_ghost_record {
    uint64_t deadline = 0;
    uint64_t key = 0;
    uint64_t generation = 0;
};

struct moe_olecar_record {
    uint64_t id = 0;
    uint64_t decision_id = 0;
    uint64_t decision_token = 0;
    uint64_t deadline = 0;
    uint64_t key = 0;
    int layer = -1;
    int expert = -1;
    int policy_id = -1;
    int bucket_id = 0;
    bool selected = false;
    bool exp4_update = false;
    const char * policy = "";
    const char * kind = "counterfactual";
    double weight_before = 0.0;
    double weight_after = 0.0;
    double policy_prob = 0.0;
    double support_prob = 0.0;
    double estimated_cost = 0.0;
    uint64_t update_count = 0;
    double final_score = 0.0;
    double lru_score = 0.0;
    double recency_score = 0.0;
    double cache_score = 0.0;
    double bad_reload_score = 0.0;
    double next_use_score = 0.0;
    double layer_score = 0.0;
    double reuse_score = 0.0;
    double cct_score = 0.0;
};

struct moe_olecar_deadline {
    uint64_t deadline = 0;
    uint64_t id = 0;
};

static constexpr int MOE_OLECAR_POLICY_COUNT = 9;
static constexpr int MOE_OLECAR_BUCKET_COUNT = 3;
static constexpr int MOE_OLECAR_FAMILY_COUNT = 5;

struct moe_lrb_group_features {
    int layer = -1;
    int expert = -1;
    uint64_t last_touch_gap = UINT64_MAX;
    uint64_t last_evict_gap = UINT64_MAX;
    uint64_t next_use_dist = UINT64_MAX;
    int layer_dist = INT_MAX / 2;
    size_t resident_bytes = 0;
    double cache_score = 0.0;
    double reuse_ema = 0.0;
    double inter_token_gap_ema = 0.0;
    uint64_t reuse_observed = 0;
    uint64_t reuse_within_1 = 0;
    uint64_t reuse_within_4 = 0;
    uint64_t reuse_within_16 = 0;
    double bad_reload_score = 0.0;
    double bad_reload_effective = 0.0;
    uint64_t bad_reload_age = UINT64_MAX;
    uint64_t bad_reload_1 = 0;
    uint64_t bad_reload_4 = 0;
    uint64_t bad_reload_16 = 0;
    double ghost_reload_risk_ema = 0.0;
    uint64_t ghost_outcomes = 0;
    double admission_feedback_ema = 0.0;
    uint64_t admission_feedback_samples = 0;
    double admission_regret_ema = 0.0;
    uint64_t admission_regret_samples = 0;
    uint64_t seq_access = 0;
    uint64_t seq_rank0_access = 0;
    uint64_t seq_cache_hits = 0;
    uint64_t seq_cache_misses = 0;
    uint64_t seq_prefetch_hits = 0;
    uint64_t seq_prefetch_late = 0;
    uint64_t seq_prefetch_unused = 0;
    uint64_t seq_future_hints = 0;
    uint64_t seq_future_rank0_hints = 0;
    uint64_t seq_predicted = 0;
    uint64_t seq_pred_enqueued = 0;
    double eamc_prior = 0.0;
    uint8_t cct_conf = 0;
    uint8_t next_token_conf = 0;
    bool pinned = false;
    bool active = false;
    bool demand_pending = false;
};

struct moe_admission_pair_record {
    uint64_t id = 0;
    uint64_t incoming_key = 0;
    uint64_t victim_key = 0;
    uint64_t decision_token = 0;
    uint64_t deadline = 0;
    uint64_t incoming_gap = UINT64_MAX;
    uint64_t victim_gap = UINT64_MAX;
    bool bypass = false;
    moe_lrb_group_features incoming_features;
    moe_lrb_group_features victim_features;
};

struct moe_admission_pair_deadline {
    uint64_t deadline = 0;
    uint64_t id = 0;
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
static void moe_admission_outcome_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason);
static void moe_admission_regret_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason);
static void moe_olecar_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason);
static void moe_cct_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        int                        src_layer,
        int                        src_expert,
        int                        target_layer,
        int                        target_expert,
        int                        conf_before,
        int                        conf_after,
        const char *               reason);
static bool moe_next_token_group_protected_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g);

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
    std::list<uint64_t> group_lru;  // front = MRU, one node per resident ExpertGroup
    size_t resident_bytes = 0;
    size_t demand_admission_staging_bytes = 0; // guarded by mtx; included in resident_bytes
    size_t demand_admission_staging_reserved_bytes = 0; // guarded by mtx
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
    bool                    prefetch_budget_enabled = false; // guarded by qmtx
    uint64_t                prefetch_budget_bytes = 0; // guarded by qmtx
    uint64_t                prefetch_budget_available_bytes = 0; // guarded by qmtx
    std::unordered_map<uint64_t, uint64_t> group_prefetch_last_enqueue_token; // guarded by qmtx
    std::unordered_map<uint64_t, moe_admission_pair_record> admission_pairs; // guarded by mtx
    std::unordered_map<uint64_t, std::vector<uint64_t>> admission_pair_watchers; // guarded by mtx
    std::deque<moe_admission_pair_deadline> admission_pair_deadlines; // guarded by mtx
    uint64_t next_admission_pair_id = 1;
    std::unordered_map<uint64_t, moe_olecar_record> olecar_records; // guarded by mtx
    std::unordered_map<uint64_t, std::vector<uint64_t>> olecar_watchers; // guarded by mtx
    std::deque<moe_olecar_deadline> olecar_deadlines; // guarded by mtx
    uint64_t next_olecar_id = 1;
    uint64_t next_olecar_decision_id = 1;
    std::array<std::array<double, MOE_OLECAR_POLICY_COUNT>, MOE_OLECAR_BUCKET_COUNT> olecar_weights = {};
    std::array<uint64_t, MOE_OLECAR_BUCKET_COUNT> olecar_updates = {0, 0, 0};

    // EAM predictor state. Guarded by mtx. The transition table learns
    // P(next-layer expert | previous-layer expert) from real routed touches
    // within the current sequence; generated predictions still go through the
    // normal admission gate and never count as real activations.
    std::unordered_map<uint64_t, std::vector<uint32_t>> eam_layer_transition;
    std::vector<int>       eam_last_layer_experts;
    int                    eam_last_actual_layer = -1;
    uint64_t               eam_last_actual_token_epoch = 0;
    std::unordered_map<uint64_t, std::vector<moe_cct_entry>> cct_transition;
    std::vector<std::vector<int>> cct_recent_layer_experts;
    std::vector<std::vector<int>> next_token_last_layer_experts;
    int                    evict_target_layer = -1;
    int                    evict_target_ahead = 0;
    size_t                 evict_layer_reserve_bytes = 0;
    size_t                 evict_persistent_bytes = 0;

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
    std::deque<moe_ghost_record> ghost_records;

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
    std::atomic<uint64_t> group_lru_rebuilds{0}, group_lru_unlinks{0};
    std::atomic<uint64_t> evict_sample_calls{0}, evict_sample_k{0};
    std::atomic<uint64_t> evict_sample_source{0}, evict_sample_fallback{0};
    std::atomic<uint64_t> evict_cold_window_calls{0}, evict_cold_window_items{0};
    std::atomic<uint64_t> evict_cold_window_source{0}, evict_cold_window_fallback{0};
    mutable std::atomic<uint64_t> resident_bytes_cached_queries{0};
    mutable std::atomic<uint64_t> resident_bytes_slow_queries{0};
    std::atomic<uint64_t> pinned_hits{0}, active_hits{0}, belady_evictions{0}, lru_fallback_evictions{0};
    std::atomic<uint64_t> evict_scan_calls{0}, evict_scan_entries{0}, evict_scan_unique{0}, evict_scan_duplicates{0};
    std::atomic<uint64_t> evict_scan_absolute{0}, evict_scan_temporary{0}, evict_scan_normal{0};
    std::atomic<uint64_t> evict_scan_selected_temporary{0}, evict_scan_no_victim{0};
    std::atomic<uint64_t> eam_access{0}, eam_rank0_access{0}, eam_future_hints{0}, eam_prefetch_queued{0};
    std::atomic<uint64_t> eam_prefetch_hits{0}, eam_prefetch_late{0}, eam_prefetch_unused{0};
    std::atomic<uint64_t> eam_cache_hits{0}, eam_cache_misses{0};
    std::atomic<uint64_t> eam_prefetch_admit{0}, eam_prefetch_drop{0};
    std::atomic<uint64_t> eam_prefetch_drop_budget{0}, eam_prefetch_drop_distance{0}, eam_prefetch_drop_score{0};
    std::atomic<uint64_t> eam_prefetch_drop_active{0}, eam_prefetch_drop_pinned{0}, eam_prefetch_drop_recent{0};
    std::atomic<uint64_t> eam_prefetch_admit_score_x1000{0};
    std::atomic<uint64_t> eam_prefetch_drop_score_x1000{0};
    std::atomic<uint64_t> eam_speculative_bytes_peak{0};
    std::atomic<uint64_t> admission_compare_runs{0}, admission_compare_admit{0}, admission_compare_drop{0};
    std::atomic<uint64_t> admission_compare_no_victim{0};
    std::atomic<uint64_t> admission_floor_refresh{0}, admission_floor_stale{0};
    std::atomic<uint64_t> admission_candidate_value_x1000{0}, admission_victim_value_x1000{0};
    std::atomic<uint64_t> demand_admission_runs{0}, demand_admission_admit{0};
    std::atomic<uint64_t> demand_admission_bypass{0}, demand_admission_no_victim{0};
    std::atomic<uint64_t> demand_admission_release_groups{0}, demand_admission_release_slices{0};
    std::atomic<uint64_t> demand_admission_release_bytes{0}, demand_admission_staging_peak{0};
    std::atomic<uint64_t> demand_admission_candidate_x1000{0}, demand_admission_victim_x1000{0};
    std::atomic<uint64_t> admission_outcome_admit_reuse_1{0}, admission_outcome_admit_reuse_4{0};
    std::atomic<uint64_t> admission_outcome_admit_reuse_16{0}, admission_outcome_admit_unused{0};
    std::atomic<uint64_t> admission_outcome_bypass_reload_1{0}, admission_outcome_bypass_reload_4{0};
    std::atomic<uint64_t> admission_outcome_bypass_reload_16{0}, admission_outcome_bypass_success{0};
    std::atomic<uint64_t> admission_feedback_updates{0}, admission_feedback_positive{0};
    std::atomic<uint64_t> admission_feedback_negative{0}, admission_feedback_score_candidates{0};
    std::atomic<uint64_t> admission_regret_pairs{0}, admission_regret_resolved{0};
    std::atomic<uint64_t> admission_regret_expired{0}, admission_regret_prefer_admit{0};
    std::atomic<uint64_t> admission_regret_prefer_bypass{0}, admission_regret_tie{0};
    std::atomic<uint64_t> admission_regret_score_candidates{0};
    std::atomic<uint64_t> eam_predict_runs{0}, eam_predict_candidates{0}, eam_predict_enqueued{0}, eam_predict_dropped{0};
    std::atomic<uint64_t> eam_predict_score_x1000{0};
    std::atomic<uint64_t> eamc_snapshot_count{0}, eamc_match_runs{0}, eamc_match_hits{0}, eamc_match_misses{0};
    std::atomic<uint64_t> eamc_match_score_x1000{0}, eamc_prior_hits{0};
    std::atomic<uint64_t> eam_evict_spec_unused{0}, eam_evict_low_score{0}, eam_evict_far_future{0}, eam_evict_lru_relaxed{0};
    std::atomic<uint64_t> eam_evict_protect_active{0}, eam_evict_protect_pinned{0}, eam_evict_protect_recent{0};
    std::atomic<uint64_t> eam_evict_protect_high{0}, eam_evict_protect_early{0};
    std::atomic<uint64_t> layer_reserve_evictions{0}, layer_reserve_protected{0}, layer_reserve_relaxed{0};
    std::atomic<uint64_t> layer_reload_bad{0}, layer_reload_protect{0};
    std::atomic<uint64_t> reuse_evict_low{0}, reuse_protect{0};
    std::atomic<uint64_t> reuse_predict_protect{0}, reuse_low_candidates{0};
    std::atomic<uint64_t> reuse_reload_1{0}, reuse_reload_4{0}, reuse_reload_16{0};
    std::atomic<uint64_t> bad_reload_protect{0}, bad_reload_soft_keep{0};
    std::atomic<uint64_t> bad_reload_1{0}, bad_reload_4{0}, bad_reload_16{0};
    std::atomic<uint64_t> ghost_evictions{0}, ghost_success{0};
    std::atomic<uint64_t> ghost_bad_1{0}, ghost_bad_4{0}, ghost_bad_16{0};
    uint64_t ghost_score_candidates = 0; // guarded by mtx
    std::atomic<uint64_t> cct_updates{0}, cct_hits{0}, cct_misses{0}, cct_unused{0}, cct_replaced{0};
    std::atomic<uint64_t> cct_prefetch_admit{0}, cct_prefetch_drop{0};
    std::atomic<uint64_t> cct_evict_protect{0}, cct_evict_keep{0};
    std::atomic<uint64_t> next_token_updates{0}, next_token_predicted{0};
    std::atomic<uint64_t> next_token_hits{0}, next_token_misses{0}, next_token_admit{0};
    std::atomic<uint64_t> next_token_layer_done_protect{0}, next_token_evict_protect{0};
    std::atomic<uint64_t> runq_ops{0}, runq_ready{0}, runq_loading{0}, runq_cold{0};
    std::atomic<uint64_t> group_fill_sibling_loads{0};
    std::atomic<uint64_t> demand_async_plans{0}, demand_async_groups{0}, demand_async_submitted{0};
    std::atomic<uint64_t> demand_async_joined{0}, demand_async_dedup{0}, demand_async_slices{0};
    std::atomic<uint64_t> demand_async_failed{0}, demand_async_wait_us{0}, demand_async_wall_us{0};
    std::atomic<uint64_t> demand_async_batch_evict_runs{0}, demand_async_batch_evict_groups{0};
    std::atomic<uint64_t> demand_async_batch_reserved_bytes{0}, demand_async_batch_reserved_peak{0};
    std::atomic<uint64_t> demand_async_batch_shortfall{0}, demand_async_worker_evictions{0};
    std::atomic<uint64_t> group_prefetch_enqueued{0}, group_prefetch_runs{0};
    std::atomic<uint64_t> group_prefetch_slices{0}, group_prefetch_hit_existing{0};
    std::atomic<uint64_t> group_prefetch_fail{0}, group_prefetch_bytes{0};
    std::atomic<uint64_t> group_prefetch_same_layer_ops{0}, group_prefetch_dups{0};
    std::atomic<uint64_t> layer_done_evict_groups{0}, layer_done_evict_slices{0};
    std::atomic<uint64_t> layer_done_evict_bytes{0};
    std::atomic<uint64_t> layer_done_protect_pinned{0}, layer_done_protect_active{0};
    std::atomic<uint64_t> layer_done_protect_future{0}, layer_done_protect_recent{0};
    std::atomic<uint64_t> layer_done_protect_bad_reload{0}, layer_done_protect_inflight{0};
    std::atomic<uint64_t> layer_done_protect_next_token{0};
    uint64_t              exec_epoch = 0;
    uint64_t              future_epoch = 0;
    uint64_t              evict_scan_generation = 0;
    uint64_t              pin_refresh_at = 0;
    bool                  admission_floor_valid = false;
    double                admission_floor_value = 0.0;
    int                   admission_floor_layer = -1;
    int                   admission_floor_expert = -1;
    uint64_t              admission_floor_token = UINT64_MAX;
    uint64_t              admission_floor_epoch = 0;
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
    std::atomic<uint64_t> prof_evict_scan_us{0};
    std::atomic<uint64_t> prof_evict_sort_us{0};
    std::atomic<uint64_t> prof_evict_online_us{0};
    std::atomic<uint64_t> prof_evict_admission_us{0};
    std::atomic<uint64_t> prof_evict_trace_us{0};
    std::atomic<uint64_t> prof_evict_release_us{0};
    std::atomic<uint64_t> prof_evict_candidate_us{0};
    std::atomic<uint64_t> prof_evict_resident_queries{0};
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
    FILE * cache_trace = nullptr;
    FILE * prefetch_trace = nullptr;
    FILE * resident_trace = nullptr;
    FILE * evict_trace = nullptr;
    FILE * cct_trace = nullptr;
    FILE * next_token_trace = nullptr;
    FILE * run_queue_trace = nullptr;
    FILE * group_prefetch_trace = nullptr;
    FILE * admission_trace = nullptr;
    FILE * admission_regret_trace = nullptr;
    FILE * lrb_trace = nullptr;
    FILE * olecar_trace = nullptr;
    uint64_t resident_trace_last_token = UINT64_MAX;
    int resident_trace_last_layer = -1;
    bool group_prefetch_selftest_done = false;

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
            moe_admission_outcome_finish_locked(*this, "teardown");
            moe_admission_regret_finish_locked(*this, "teardown");
            moe_olecar_finish_locked(*this, "teardown");
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
        if (cache_trace != nullptr) {
            std::fclose(cache_trace);
            cache_trace = nullptr;
        }
        if (prefetch_trace != nullptr) {
            std::fclose(prefetch_trace);
            prefetch_trace = nullptr;
        }
        if (resident_trace != nullptr) {
            std::fclose(resident_trace);
            resident_trace = nullptr;
        }
        if (evict_trace != nullptr) {
            std::fclose(evict_trace);
            evict_trace = nullptr;
        }
        if (cct_trace != nullptr) {
            std::fclose(cct_trace);
            cct_trace = nullptr;
        }
        if (next_token_trace != nullptr) {
            std::fclose(next_token_trace);
            next_token_trace = nullptr;
        }
        if (run_queue_trace != nullptr) {
            std::fclose(run_queue_trace);
            run_queue_trace = nullptr;
        }
        if (group_prefetch_trace != nullptr) {
            std::fclose(group_prefetch_trace);
            group_prefetch_trace = nullptr;
        }
        if (admission_trace != nullptr) {
            std::fclose(admission_trace);
            admission_trace = nullptr;
        }
        if (admission_regret_trace != nullptr) {
            std::fclose(admission_regret_trace);
            admission_regret_trace = nullptr;
        }
        if (lrb_trace != nullptr) {
            std::fclose(lrb_trace);
            lrb_trace = nullptr;
        }
        if (olecar_trace != nullptr) {
            std::fclose(olecar_trace);
            olecar_trace = nullptr;
        }
    }
};

// ---- forward decls of internals ----
static void moe_stream_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank,
        bool group_fill = true, bool demand_async = false);
static void moe_stream_mwq_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank,
        bool * out_entry_missing = nullptr, bool * out_read_failed = nullptr, bool group_fill = true,
        bool demand_async = false);
static const moe_sidecar_entry * moe_mwq_entry(const llama_moe_buffer_context & ctx, const moe_managed & m, int e, int bits);
static moe_group_state & moe_group_get(llama_moe_buffer_context & ctx, int layer, int expert);
static void moe_prefetch_group_worker(llama_moe_buffer_context & ctx, int layer, int expert, int rank, float score,
        int skip_kind, bool is_demand = false, int primary_kind = -1);

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
            if (!task.is_group && task.m != nullptr && task.e >= 0 && task.e < (int) task.m->queued.size()) {
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
        if (task.is_group) {
            moe_prefetch_group_worker(*ctx, task.layer, task.e, task.rank, task.score, task.skip_kind,
                    task.is_demand, task.primary_kind);
            if (task.is_demand) {
                std::lock_guard<std::mutex> lk(ctx->mtx);
                auto git = ctx->groups.find(moe_group_key(task.layer, task.e));
                if (git != ctx->groups.end() &&
                        git->second.demand_async_generation == task.generation) {
                    ctx->demand_admission_staging_reserved_bytes -= std::min(
                            ctx->demand_admission_staging_reserved_bytes,
                            git->second.admission_staging_reserved_bytes);
                    git->second.admission_staging_reserved_bytes = 0;
                    git->second.demand_async_pending = false;
                }
                ctx->cv_done.notify_all();
            }
            ctx->worker_streams.fetch_add(1, std::memory_order_relaxed);
            continue;
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
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_CACHE_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->cache_trace = std::fopen(trace_path, "wb");
            if (ctx->cache_trace != nullptr) {
                std::fprintf(ctx->cache_trace,
                        "token\texec\tlayer\texpert\tkind\trank\ttarget_bits\tarrival_state\tarrival_bits\tarrival_mwq\tarrival_queued\tarrival_queued_bits\tarrival_prefetched\taction\twait_ns\tresident_mib\tbudget_mib\tlru_entries\tresident_groups_total\tresident_groups_current_layer\tresident_groups_other_layer\tresident_full_groups_total\tresident_full_groups_current_layer\texact_group_resident_slices\texact_group_resident_bytes\texact_group_resident_full\tresident_current_layer_mib\tresident_other_layer_mib\tstreams\tevictions\tcache_hit_total\tcache_miss_total\tgroup_access\tgroup_rank0\tgroup_cache_hit\tgroup_cache_miss\tgroup_prefetch_hit\tgroup_prefetch_unused\tgroup_last_token\tevicted_age_tokens\tgroup_score\tpinned\tactive\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open cache trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_PREFETCH_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->prefetch_trace = std::fopen(trace_path, "wb");
            if (ctx->prefetch_trace != nullptr) {
                std::fprintf(ctx->prefetch_trace,
                        "token\texec\tevent\treason\tlayer\texpert\tkind\trank\ttarget_bits\tclg_score\tadmit_score\tcandidate_value\tvictim_value\tvictim_layer\tvictim_expert\tspeculative_mib\tresident_mib\tbudget_mib\tstate\tresident_bits\tresident_mwq\tqueued\tqueued_bits\tprefetched\tresident_groups_total\tresident_groups_current_layer\tresident_groups_other_layer\texact_group_resident_slices\texact_group_resident_bytes\texact_group_full\tgroup_access\tgroup_rank0\tgroup_cache_hit\tgroup_cache_miss\tgroup_prefetch_hit\tgroup_prefetch_unused\tgroup_score\tpinned\tactive\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open prefetch trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_RESIDENT_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->resident_trace = std::fopen(trace_path, "wb");
            if (ctx->resident_trace != nullptr) {
                std::fprintf(ctx->resident_trace,
                        "token\texec\tcurrent_layer\tresident_layer\texpert\tslices\tbytes\tfull\tbits_min\tbits_max\tmwq_slices\tprefetched_slices\ttouched_slices\tqueued_slices\tqueued_bits_max\tresident_mib\tbudget_mib\tgroup_access\tgroup_rank0\tgroup_cache_hit\tgroup_cache_miss\tgroup_prefetch_hit\tgroup_prefetch_unused\tgroup_last_token\tgroup_score\treuse_keep\treuse_ema\treuse_gap_ema\treuse_observed\treuse_within_1\treuse_within_4\treuse_within_16\tnext_use_dist\tlayer_dist\treuse_predicted_soon\th0_current_layer\th1_next_token_keep\th4_short_reuse\tnext_token_conf\tnext_token_protect_until\tpinned\tactive\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open resident trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_EVICT_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->evict_trace = std::fopen(trace_path, "wb");
            if (ctx->evict_trace != nullptr) {
                std::fprintf(ctx->evict_trace,
                        "event\ttoken\texec\tlayer\texpert\treason\treload_gap\treleased_bytes\tresident_mib\tbudget_mib\tscore\treuse_keep\treuse_ema\treuse_gap_ema\treuse_observed\treuse_within_1\treuse_within_4\treuse_within_16\tseq_access\tseq_rank0\tseq_rate\tcache_hit\tcache_miss\tprefetch_hit\tprefetch_unused\thot_score\teamc_prior\team_replace_pred\team_replace_mass\team_replace_keep\tcct_conf\tcct_protect\tbad_reload_score\tbad_reload_effective\tbad_reload_age\tbad_reload_1\tbad_reload_4\tbad_reload_16\tbad_reload_protect\tnext_use_dist\tlayer_dist\tpinned\tactive\trecent\thigh_sequence\tearly_high\tspeculative_unused\tlayer_window\tpredicted_soon\tlast_evict_reason\tlayer_bad_reload\tlayer_last_bad_reload_token\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open evict trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_CCT_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->cct_trace = std::fopen(trace_path, "wb");
            if (ctx->cct_trace != nullptr) {
                std::fprintf(ctx->cct_trace,
                        "event\ttoken\texec\tsrc_layer\tsrc_expert\ttarget_layer\ttarget_expert\tconf_before\tconf_after\treason\tresident_mib\tbudget_mib\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open CCT trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_NEXT_TOKEN_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->next_token_trace = std::fopen(trace_path, "wb");
            if (ctx->next_token_trace != nullptr) {
                std::fprintf(ctx->next_token_trace,
                        "event\ttoken\texec\tlayer\texpert\tconf_before\tconf_after\tprotect_until\tresident_mib\tbudget_mib\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open next-token trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_RUN_QUEUE_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->run_queue_trace = std::fopen(trace_path, "wb");
            if (ctx->run_queue_trace != nullptr) {
                std::fprintf(ctx->run_queue_trace,
                        "token\texec\tlayer\texpert\tkind\ttarget_bits\trank\tcount\tstate\torder\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open run queue trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_GROUP_PREFETCH_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->group_prefetch_trace = std::fopen(trace_path, "wb");
            if (ctx->group_prefetch_trace != nullptr) {
                std::fprintf(ctx->group_prefetch_trace,
                        "token\texec\tevent\tlayer\texpert\tkind\ttarget_bits\tstate\tresident_mib\tbudget_mib\n");
            } else if (params.debug_log) {
                std::fprintf(stderr, "llama_moe_buffer: failed to open group prefetch trace %s\n", trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_ADMISSION_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->admission_trace = std::fopen(trace_path, "wb");
            if (ctx->admission_trace != nullptr) {
                std::fprintf(ctx->admission_trace,
                        "event\ttoken\texec\tlayer\texpert\tgeneration\taction\tgap\tcandidate_value\tvictim_value\tvictim_layer\tvictim_expert\tvictim_groups\tvictim_bytes\tresident_mib\tbudget_mib\tfeedback_ema\tfeedback_samples\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open admission trace %s\n",
                        trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_ADMISSION_REGRET_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->admission_regret_trace = std::fopen(trace_path, "wb");
            if (ctx->admission_regret_trace != nullptr) {
                std::fprintf(ctx->admission_regret_trace,
                        "event\ttoken\texec\tpair_id\taction\tincoming_layer\tincoming_expert\tvictim_layer\tvictim_expert\tincoming_gap\tvictim_gap\tregret_target\tregret_ema\tregret_samples\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open admission regret trace %s\n",
                        trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_LRB_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->lrb_trace = std::fopen(trace_path, "wb");
            if (ctx->lrb_trace != nullptr) {
                std::fprintf(ctx->lrb_trace,
                        "event\ttoken\texec\tpair_id\taction\tdecision_token\tdeadline\tincoming_gap\tvictim_gap\tregret_target\tpreference\tcorrect\tresident_mib\tbudget_mib\tstaging_mib\tresident_groups");
                const char * cols[] = {
                    "layer", "expert", "last_touch_gap", "last_evict_gap",
                    "next_use_dist", "layer_dist", "resident_mib", "cache_score",
                    "reuse_ema", "inter_token_gap_ema", "reuse_observed",
                    "reuse_within_1", "reuse_within_4", "reuse_within_16",
                    "bad_reload_score", "bad_reload_effective", "bad_reload_age",
                    "bad_reload_1", "bad_reload_4", "bad_reload_16",
                    "ghost_reload_risk_ema", "ghost_outcomes",
                    "admission_feedback_ema", "admission_feedback_samples",
                    "admission_regret_ema", "admission_regret_samples",
                    "seq_access", "seq_rank0_access", "seq_cache_hits",
                    "seq_cache_misses", "seq_prefetch_hits", "seq_prefetch_late",
                    "seq_prefetch_unused", "seq_future_hints",
                    "seq_future_rank0_hints", "seq_predicted", "seq_pred_enqueued",
                    "eamc_prior", "cct_conf", "next_token_conf", "pinned",
                    "active", "demand_pending",
                };
                for (const char * prefix : {"incoming", "victim"}) {
                    for (const char * col : cols) {
                        std::fprintf(ctx->lrb_trace, "\t%s_%s", prefix, col);
                    }
                }
                std::fprintf(ctx->lrb_trace, "\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open LRB trace %s\n",
                        trace_path);
            }
        }
    }
    if (const char * trace_path = std::getenv("LLAMA_LAZY_MOE_OLECAR_TRACE")) {
        if (trace_path[0] != '\0') {
            ctx->olecar_trace = std::fopen(trace_path, "wb");
            if (ctx->olecar_trace != nullptr) {
                std::fprintf(ctx->olecar_trace,
                        "event\tkind\ttoken\texec\tid\tdecision_id\tbucket_id\tpolicy_id\tpolicy\tlayer\texpert\tselected\tgap\tcost\testimated_cost\tdecision_token\tdeadline\tresident_mib\tbudget_mib\tweight_before\tweight_after\tpolicy_prob\tsupport_prob\tupdate_count\tfinal_score\tlru_score\trecency_score\tcache_score\tbad_reload_score\tnext_use_score\tlayer_score\treuse_score\tcct_score\treason\n");
            } else if (params.debug_log) {
                std::fprintf(stderr,
                        "llama_moe_buffer: failed to open OLECAR trace %s\n",
                        trace_path);
            }
        }
    }
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
    m.demand_async_loaded.assign(n_expert, false);
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

static bool moe_group_is_hot(const llama_moe_buffer_context & ctx, int layer, int expert) {
    const auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return false;
    }
    for (const moe_managed * m : it->second) {
        if (m != nullptr && expert >= 0 && expert < m->n_expert && moe_is_hot(ctx, *m, expert)) {
            return true;
        }
    }
    return false;
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

static void moe_group_resident_add_locked(
        llama_moe_buffer_context & ctx,
        const moe_managed &        m,
        int                        e,
        size_t                     bytes,
        int                        bits) {
    if (m.layer < 0 || e < 0 || e >= m.n_expert || bytes == 0) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    g.resident_bytes_cached += bytes;
    g.resident_bit_weight_cached += (uint64_t) bytes * (uint64_t) std::max(1, bits);
    if (g.resident_slices_cached != UINT8_MAX) {
        g.resident_slices_cached++;
    }
}

static void moe_group_resident_remove_locked(
        llama_moe_buffer_context & ctx,
        const moe_managed &        m,
        int                        e,
        size_t                     bytes,
        int                        bits) {
    if (m.layer < 0 || e < 0 || e >= m.n_expert || bytes == 0) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, m.layer, e);
    g.resident_bytes_cached -= std::min(g.resident_bytes_cached, bytes);
    const uint64_t bit_weight = (uint64_t) bytes * (uint64_t) std::max(1, bits);
    g.resident_bit_weight_cached -= std::min(g.resident_bit_weight_cached, bit_weight);
    if (g.resident_slices_cached > 0) {
        g.resident_slices_cached--;
    }
}

static size_t moe_group_resident_bytes(const llama_moe_buffer_context & ctx, int layer, int expert) {
    const uint64_t key = moe_group_key(layer, expert);
    auto git = ctx.groups.find(key);
    if (git != ctx.groups.end()) {
        ctx.resident_bytes_cached_queries.fetch_add(1, std::memory_order_relaxed);
        return git->second.resident_bytes_cached;
    }
    ctx.resident_bytes_slow_queries.fetch_add(1, std::memory_order_relaxed);
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
    const uint64_t key = moe_group_key(layer, expert);
    auto git = ctx.groups.find(key);
    if (git != ctx.groups.end()) {
        const moe_group_state & g = git->second;
        return g.resident_bytes_cached > 0 ?
            (double) g.resident_bit_weight_cached / (double) g.resident_bytes_cached : 0.0;
    }
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

static double moe_env_f64(const char * name, double def);
static int moe_env_i32(const char * name, int def);
static size_t moe_env_mib_bytes(const char * name, int def_mib);
static int moe_max_layer_index(const llama_moe_buffer_context & ctx);
static void moe_resident_trace_snapshot_locked(llama_moe_buffer_context & ctx, int current_layer);
static double moe_group_short_reuse_keep_score_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        int                              current_layer);
static bool moe_eam_replace_enabled();
static double moe_group_seq_rate(const llama_moe_buffer_context & ctx, const moe_group_state & g);
static double moe_eamc_prior(const llama_moe_buffer_context & ctx, int layer, int expert);
static int moe_forward_layer_distance(const llama_moe_buffer_context & ctx, int target_layer);
static bool moe_group_is_high_sequence(const llama_moe_buffer_context & ctx, const moe_group_state & g, double cache_score);
static bool moe_group_is_early_high_reuse(const moe_group_state & g, double cache_score);
static bool moe_group_has_speculative_unused(const llama_moe_buffer_context & ctx, int layer, int expert);
static uint8_t moe_cct_predict_conf_locked(
        const llama_moe_buffer_context & ctx,
        int                              target_layer,
        int                              target_expert);
static double moe_group_bad_reload_effective_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        uint64_t *                       out_age = nullptr);
static bool moe_group_predicted_soon_for_evict(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        bool                             layer_window,
        uint64_t                         next_use_dist);
static double moe_group_eam_replace_pred_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g);
static double moe_layer_eam_replace_mass_locked(
        const llama_moe_buffer_context & ctx,
        int                              layer);
static double moe_group_eam_replace_keep_score_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        double                           layer_mass);

static bool moe_layer_in_evict_window(const llama_moe_buffer_context & ctx, int layer) {
    if (ctx.evict_target_layer < 0 || layer < 0) {
        return false;
    }
    const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
    for (int d = 0; d <= std::max(0, ctx.evict_target_ahead); ++d) {
        if (((ctx.evict_target_layer + d) % n_layer) == layer) {
            return true;
        }
    }
    return false;
}

static size_t moe_layer_resident_bytes_locked(const llama_moe_buffer_context & ctx, int layer) {
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return 0;
    }
    size_t total = 0;
    int n_expert = 0;
    for (const moe_managed * m : it->second) {
        if (m != nullptr) {
            n_expert = std::max(n_expert, m->n_expert);
        }
    }
    for (int e = 0; e < n_expert; ++e) {
        total += moe_group_resident_bytes(ctx, layer, e);
    }
    return total;
}

struct moe_evict_context_guard {
    llama_moe_buffer_context & ctx;
    int old_layer;
    int old_ahead;
    size_t old_reserve;
    size_t old_persistent;

    moe_evict_context_guard(llama_moe_buffer_context & c, const moe_managed & m, bool touch)
        : ctx(c),
          old_layer(c.evict_target_layer),
          old_ahead(c.evict_target_ahead),
          old_reserve(c.evict_layer_reserve_bytes),
          old_persistent(c.evict_persistent_bytes) {
        const size_t default_reserve = c.params.budget_bytes > 0 && c.params.budget_bytes <= 1280ull * 1048576ull ?
            64ull * 1048576ull : 0;
        const size_t reserve = moe_env_mib_bytes("LLAMA_LAZY_MOE_LAYER_RESERVE_MB", (int) (default_reserve / 1048576ull));
        if (touch && reserve > 0 && m.layer >= 0) {
            c.evict_target_layer = m.layer;
            c.evict_target_ahead = std::max(0, moe_env_i32("LLAMA_LAZY_MOE_LAYER_RESERVE_AHEAD", 0));
            c.evict_layer_reserve_bytes = reserve;
            c.evict_persistent_bytes = moe_env_mib_bytes("LLAMA_LAZY_MOE_LAYER_RESERVE_PERSISTENT_MB", 640);
        }
    }

    ~moe_evict_context_guard() {
        ctx.evict_target_layer = old_layer;
        ctx.evict_target_ahead = old_ahead;
        ctx.evict_layer_reserve_bytes = old_reserve;
        ctx.evict_persistent_bytes = old_persistent;
    }
};

static int moe_kind_index(const moe_managed & m) {
    switch (moe_kind(m)) {
        case moe_tensor_kind::gate: return 1;
        case moe_tensor_kind::up:   return 2;
        case moe_tensor_kind::down: return 3;
        case moe_tensor_kind::other:
        default: return 0;
    }
}

static const char * moe_kind_label(const moe_managed & m) {
    switch (moe_kind(m)) {
        case moe_tensor_kind::gate: return "gate";
        case moe_tensor_kind::up:   return "up";
        case moe_tensor_kind::down: return "down";
        case moe_tensor_kind::other:
        default: return "other";
    }
}

static const char * moe_resident_state_label(char s) {
    switch (s) {
        case ST_RESIDENT: return "resident";
        case ST_INFLIGHT: return "inflight";
        case ST_COLD:
        default: return "cold";
    }
}

static void moe_cache_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const moe_managed &        m,
        int                        e,
        int                        rank,
        int                        target_bits,
        const char *               action,
        char                       arrival_state,
        int                        arrival_bits,
        bool                       arrival_mwq,
        bool                       arrival_queued,
        int                        arrival_queued_bits,
        bool                       arrival_prefetched,
        uint64_t                   wait_ns) {
    moe_resident_trace_snapshot_locked(ctx, m.layer);
    if (ctx.cache_trace == nullptr || e < 0 || e >= m.n_expert) {
        return;
    }

    const uint64_t key = moe_group_key(m.layer, e);
    const auto git = ctx.groups.find(key);
    const moe_group_state * g = git == ctx.groups.end() ? nullptr : &git->second;
    const uint64_t group_access = g == nullptr ? 0 : g->seq_access;
    const uint64_t group_rank0 = g == nullptr ? 0 : g->seq_rank0_access;
    const uint64_t group_hit = g == nullptr ? 0 : g->seq_cache_hits;
    const uint64_t group_miss = g == nullptr ? 0 : g->seq_cache_misses;
    const uint64_t group_prefetch_hit = g == nullptr ? 0 : g->seq_prefetch_hits;
    const uint64_t group_prefetch_unused = g == nullptr ? 0 : g->seq_prefetch_unused;
    const uint64_t group_last_token = g == nullptr ? 0 : g->seq_last_token_epoch;
    const double group_score = g == nullptr ? 0.0 : g->hot_score;
    const int pinned = g != nullptr && g->pinned ? 1 : 0;
    const int active = g != nullptr && moe_group_is_active(ctx, *g) ? 1 : 0;
    const uint64_t last_evicted = e < (int) m.last_evicted_token.size() ? m.last_evicted_token[e] : 0;
    const uint64_t evicted_age = last_evicted > 0 && ctx.profile_token_epoch >= last_evicted ?
        ctx.profile_token_epoch - last_evicted : UINT64_MAX;

    size_t resident_groups_total = 0;
    size_t resident_groups_current_layer = 0;
    size_t resident_groups_other_layer = 0;
    size_t resident_full_groups_total = 0;
    size_t resident_full_groups_current_layer = 0;
    size_t exact_group_resident_slices = 0;
    size_t exact_group_resident_bytes = 0;
    double resident_current_layer_mib = 0.0;
    double resident_other_layer_mib = 0.0;

    for (const auto & lkv : ctx.by_layer) {
        const int layer = lkv.first;
        const std::vector<moe_managed *> & tensors = lkv.second;
        int n_expert = 0;
        for (const moe_managed * tm : tensors) {
            if (tm != nullptr) {
                n_expert = std::max(n_expert, tm->n_expert);
            }
        }
        for (int expert = 0; expert < n_expert; ++expert) {
            size_t slices = 0;
            size_t bytes = 0;
            for (const moe_managed * tm : tensors) {
                if (tm == nullptr || expert < 0 || expert >= tm->n_expert) {
                    continue;
                }
                if (tm->resident[expert] == ST_RESIDENT) {
                    ++slices;
                    bytes += tm->resident_size[expert];
                }
            }
            if (slices == 0) {
                continue;
            }
            ++resident_groups_total;
            if (slices >= 3) {
                ++resident_full_groups_total;
            }
            if (layer == m.layer) {
                ++resident_groups_current_layer;
                resident_current_layer_mib += bytes / 1048576.0;
                if (slices >= 3) {
                    ++resident_full_groups_current_layer;
                }
            } else {
                ++resident_groups_other_layer;
                resident_other_layer_mib += bytes / 1048576.0;
            }
            if (layer == m.layer && expert == e) {
                exact_group_resident_slices = slices;
                exact_group_resident_bytes = bytes;
            }
        }
    }

    std::fprintf(ctx.cache_trace,
            "%llu\t%llu\t%d\t%d\t%s\t%d\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t%s\t%llu\t%.3f\t%.3f\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%zu\t%.3f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%.3f\t%d\t%d\n",
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            m.layer,
            e,
            moe_kind_label(m),
            rank,
            target_bits,
            moe_resident_state_label(arrival_state),
            arrival_bits,
            arrival_mwq ? 1 : 0,
            arrival_queued ? 1 : 0,
            arrival_queued_bits,
            arrival_prefetched ? 1 : 0,
            action,
            (unsigned long long) wait_ns,
            ctx.resident_bytes / 1048576.0,
            ctx.params.budget_bytes / 1048576.0,
            ctx.lru.size(),
            resident_groups_total,
            resident_groups_current_layer,
            resident_groups_other_layer,
            resident_full_groups_total,
            resident_full_groups_current_layer,
            exact_group_resident_slices,
            exact_group_resident_bytes,
            (size_t) (exact_group_resident_slices >= 3 ? 1 : 0),
            resident_current_layer_mib,
            resident_other_layer_mib,
            (unsigned long long) ctx.streams.load(std::memory_order_relaxed),
            (unsigned long long) ctx.evictions.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_cache_hits.load(std::memory_order_relaxed),
            (unsigned long long) ctx.eam_cache_misses.load(std::memory_order_relaxed),
            (unsigned long long) group_access,
            (unsigned long long) group_rank0,
            (unsigned long long) group_hit,
            (unsigned long long) group_miss,
            (unsigned long long) group_prefetch_hit,
            (unsigned long long) group_prefetch_unused,
            (unsigned long long) group_last_token,
            last_evicted == 0 ? 0ull : (unsigned long long) evicted_age,
            group_score,
            pinned,
            active);
}

static void moe_prefetch_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const moe_managed &        m,
        int                        e,
        int                        rank,
        int                        target_bits,
        const char *               event,
        const char *               reason,
        float                      clg_score,
        double                     admit_score,
        size_t                     speculative_bytes,
        double                     candidate_value = 0.0,
        double                     victim_value = 0.0,
        int                        victim_layer = -1,
        int                        victim_expert = -1) {
    if (ctx.prefetch_trace == nullptr || e < 0 || e >= m.n_expert) {
        return;
    }

    const uint64_t key = moe_group_key(m.layer, e);
    const auto git = ctx.groups.find(key);
    const moe_group_state * g = git == ctx.groups.end() ? nullptr : &git->second;

    size_t resident_groups_total = 0;
    size_t resident_groups_current_layer = 0;
    size_t resident_groups_other_layer = 0;
    size_t exact_group_resident_slices = 0;
    size_t exact_group_resident_bytes = 0;

    for (const auto & lkv : ctx.by_layer) {
        const int layer = lkv.first;
        const std::vector<moe_managed *> & tensors = lkv.second;
        int n_expert = 0;
        for (const moe_managed * tm : tensors) {
            if (tm != nullptr) {
                n_expert = std::max(n_expert, tm->n_expert);
            }
        }
        for (int expert = 0; expert < n_expert; ++expert) {
            size_t slices = 0;
            size_t bytes = 0;
            for (const moe_managed * tm : tensors) {
                if (tm == nullptr || expert >= tm->n_expert) {
                    continue;
                }
                if (tm->resident[expert] == ST_RESIDENT) {
                    ++slices;
                    bytes += tm->resident_size[expert];
                }
            }
            if (slices == 0) {
                continue;
            }
            ++resident_groups_total;
            if (layer == m.layer) {
                ++resident_groups_current_layer;
            } else {
                ++resident_groups_other_layer;
            }
            if (layer == m.layer && expert == e) {
                exact_group_resident_slices = slices;
                exact_group_resident_bytes = bytes;
            }
        }
    }

    std::fprintf(ctx.prefetch_trace,
            "%llu\t%llu\t%s\t%s\t%d\t%d\t%s\t%d\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%d\t%d\t%.3f\t%.3f\t%.3f\t%s\t%d\t%d\t%d\t%d\t%d\t%zu\t%zu\t%zu\t%zu\t%zu\t%d\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%.3f\t%d\t%d\n",
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            event,
            reason,
            m.layer,
            e,
            moe_kind_label(m),
            rank,
            target_bits,
            (double) clg_score,
            admit_score,
            candidate_value,
            victim_value,
            victim_layer,
            victim_expert,
            speculative_bytes / 1048576.0,
            ctx.resident_bytes / 1048576.0,
            ctx.params.budget_bytes / 1048576.0,
            moe_resident_state_label(m.resident[e]),
            e < (int) m.resident_bits.size() ? m.resident_bits[e] : 0,
            e < (int) m.resident_mwq.size() && m.resident_mwq[e] ? 1 : 0,
            e < (int) m.queued.size() && m.queued[e] ? 1 : 0,
            e < (int) m.queued_bits.size() ? m.queued_bits[e] : 0,
            e < (int) m.prefetched.size() && m.prefetched[e] ? 1 : 0,
            resident_groups_total,
            resident_groups_current_layer,
            resident_groups_other_layer,
            exact_group_resident_slices,
            exact_group_resident_bytes,
            exact_group_resident_slices >= 3 ? 1 : 0,
            (unsigned long long) (g == nullptr ? 0 : g->seq_access),
            (unsigned long long) (g == nullptr ? 0 : g->seq_rank0_access),
            (unsigned long long) (g == nullptr ? 0 : g->seq_cache_hits),
            (unsigned long long) (g == nullptr ? 0 : g->seq_cache_misses),
            (unsigned long long) (g == nullptr ? 0 : g->seq_prefetch_hits),
            (unsigned long long) (g == nullptr ? 0 : g->seq_prefetch_unused),
            g == nullptr ? 0.0 : g->hot_score,
            g != nullptr && g->pinned ? 1 : 0,
            g != nullptr && moe_group_is_active(ctx, *g) ? 1 : 0);
}

static void moe_resident_trace_snapshot_locked(
        llama_moe_buffer_context & ctx,
        int                        current_layer) {
    if (ctx.resident_trace == nullptr || current_layer < 0) {
        return;
    }
    if (ctx.resident_trace_last_token == ctx.profile_token_epoch &&
            ctx.resident_trace_last_layer == current_layer) {
        return;
    }
    ctx.resident_trace_last_token = ctx.profile_token_epoch;
    ctx.resident_trace_last_layer = current_layer;

    for (const auto & lkv : ctx.by_layer) {
        const int resident_layer = lkv.first;
        const std::vector<moe_managed *> & tensors = lkv.second;
        int n_expert = 0;
        for (const moe_managed * tm : tensors) {
            if (tm != nullptr) {
                n_expert = std::max(n_expert, tm->n_expert);
            }
        }
        for (int expert = 0; expert < n_expert; ++expert) {
            size_t slices = 0;
            size_t bytes = 0;
            int bits_min = INT_MAX;
            int bits_max = 0;
            size_t mwq_slices = 0;
            size_t prefetched_slices = 0;
            size_t touched_slices = 0;
            size_t queued_slices = 0;
            int queued_bits_max = 0;
            for (const moe_managed * tm : tensors) {
                if (tm == nullptr || expert < 0 || expert >= tm->n_expert) {
                    continue;
                }
                if (tm->resident[expert] == ST_RESIDENT) {
                    ++slices;
                    bytes += tm->resident_size[expert];
                    const int bits = expert < (int) tm->resident_bits.size() ? tm->resident_bits[expert] : 0;
                    if (bits > 0) {
                        bits_min = std::min(bits_min, bits);
                        bits_max = std::max(bits_max, bits);
                    }
                    if (expert < (int) tm->resident_mwq.size() && tm->resident_mwq[expert]) {
                        ++mwq_slices;
                    }
                    if (expert < (int) tm->prefetched.size() && tm->prefetched[expert]) {
                        ++prefetched_slices;
                    }
                    if (expert < (int) tm->resident_touched.size() && tm->resident_touched[expert]) {
                        ++touched_slices;
                    }
                }
                if (expert < (int) tm->queued.size() && tm->queued[expert]) {
                    ++queued_slices;
                    if (expert < (int) tm->queued_bits.size()) {
                        queued_bits_max = std::max(queued_bits_max, tm->queued_bits[expert]);
                    }
                }
            }
            if (slices == 0) {
                continue;
            }

            const uint64_t key = moe_group_key(resident_layer, expert);
            const auto git = ctx.groups.find(key);
            const moe_group_state * g = git == ctx.groups.end() ? nullptr : &git->second;
            const double reuse_keep = g == nullptr ? 0.0 :
                moe_group_short_reuse_keep_score_locked(ctx, *g, current_layer);
            uint64_t next_use_dist = UINT64_MAX;
            if (g != nullptr && g->next_use_epoch != UINT64_MAX && g->next_use_epoch > ctx.exec_epoch) {
                next_use_dist = g->next_use_epoch - ctx.exec_epoch;
            }
            const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
            const int layer_dist = resident_layer >= current_layer ?
                resident_layer - current_layer : n_layer - current_layer + resident_layer;
            const bool predicted_soon = g != nullptr &&
                moe_group_predicted_soon_for_evict(ctx, *g,
                        layer_dist <= moe_env_i32("LLAMA_LAZY_MOE_RESIDENT_TRACE_NEAR_AHEAD", 1),
                        next_use_dist);
            const bool h0_current_layer = resident_layer == current_layer;
            const bool h1_next_token_keep = g != nullptr && moe_next_token_group_protected_locked(ctx, *g);
            const bool h4_short_reuse = h1_next_token_keep ||
                (g != nullptr && (g->reuse_within_4 > 0 || g->inter_token_gap_ema <= 4.0));
            std::fprintf(ctx.resident_trace,
                    "%llu\t%llu\t%d\t%d\t%d\t%zu\t%zu\t%d\t%d\t%d\t%zu\t%zu\t%zu\t%zu\t%d\t%.3f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%.3f\t%.3f\t%.6f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%u\t%llu\t%d\t%d\n",
                    (unsigned long long) ctx.profile_token_epoch,
                    (unsigned long long) ctx.exec_epoch,
                    current_layer,
                    resident_layer,
                    expert,
                    slices,
                    bytes,
                    slices >= 3 ? 1 : 0,
                    bits_min == INT_MAX ? 0 : bits_min,
                    bits_max,
                    mwq_slices,
                    prefetched_slices,
                    touched_slices,
                    queued_slices,
                    queued_bits_max,
                    ctx.resident_bytes / 1048576.0,
                    ctx.params.budget_bytes / 1048576.0,
                    (unsigned long long) (g == nullptr ? 0 : g->seq_access),
                    (unsigned long long) (g == nullptr ? 0 : g->seq_rank0_access),
                    (unsigned long long) (g == nullptr ? 0 : g->seq_cache_hits),
                    (unsigned long long) (g == nullptr ? 0 : g->seq_cache_misses),
                    (unsigned long long) (g == nullptr ? 0 : g->seq_prefetch_hits),
                    (unsigned long long) (g == nullptr ? 0 : g->seq_prefetch_unused),
                    (unsigned long long) (g == nullptr ? 0 : g->seq_last_token_epoch),
                    g == nullptr ? 0.0 : g->hot_score,
                    reuse_keep,
                    g == nullptr ? 0.0 : g->reuse_ema,
                    g == nullptr ? 0.0 : g->inter_token_gap_ema,
                    (unsigned long long) (g == nullptr ? 0 : g->reuse_observed),
                    (unsigned long long) (g == nullptr ? 0 : g->reuse_within_1),
                    (unsigned long long) (g == nullptr ? 0 : g->reuse_within_4),
                    (unsigned long long) (g == nullptr ? 0 : g->reuse_within_16),
                    (unsigned long long) next_use_dist,
                    layer_dist,
                    predicted_soon ? 1 : 0,
                    h0_current_layer ? 1 : 0,
                    h1_next_token_keep ? 1 : 0,
                    h4_short_reuse ? 1 : 0,
                    (unsigned) (g == nullptr ? 0 : g->next_token_conf),
                    (unsigned long long) (g == nullptr ? 0 : g->next_token_protect_until),
                    g != nullptr && g->pinned ? 1 : 0,
                    g != nullptr && moe_group_is_active(ctx, *g) ? 1 : 0);
        }
    }
}

static void moe_evict_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        int                        layer,
        int                        expert,
        const char *               reason,
        uint64_t                   reload_gap,
        size_t                     released_bytes,
        double                     score) {
    if (ctx.evict_trace == nullptr || layer < 0 || expert < 0) {
        return;
    }

    moe_group_state & g = moe_group_get(ctx, layer, expert);
    const double reuse_keep = moe_group_short_reuse_keep_score_locked(ctx, g, ctx.evict_target_layer);
    const double seq_rate = moe_group_seq_rate(ctx, g);
    uint64_t next_use_dist = UINT64_MAX;
    if (g.next_use_epoch != UINT64_MAX && g.next_use_epoch > ctx.exec_epoch) {
        next_use_dist = g.next_use_epoch - ctx.exec_epoch;
    }
    const int layer_dist = moe_forward_layer_distance(ctx, layer);
    const bool active = moe_group_is_active(ctx, g);
    const bool recent = moe_group_is_recently_used(ctx, g) || moe_group_in_cooldown(ctx, g) ||
        moe_group_used_within_tokens(ctx, g, moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_RECENT_TOKENS", 3));
    const bool high_sequence = moe_group_is_high_sequence(ctx, g, g.hot_score);
    const bool early_high = moe_group_is_early_high_reuse(g, g.hot_score);
    const bool speculative_unused = moe_group_has_speculative_unused(ctx, layer, expert);
    const bool layer_window = moe_layer_in_evict_window(ctx, layer);
    const bool predicted_soon = moe_group_predicted_soon_for_evict(ctx, g, layer_window, next_use_dist);
    double eam_replace_pred = 0.0;
    double eam_replace_mass = 0.0;
    double eam_replace_keep = 0.0;
    if (moe_eam_replace_enabled()) {
        eam_replace_pred = g.eam_replace_pred;
        eam_replace_mass = g.eam_replace_mass;
        eam_replace_keep = g.eam_replace_keep;
    }
    const uint8_t cct_conf = moe_cct_predict_conf_locked(ctx, layer, expert);
    uint64_t bad_reload_age = UINT64_MAX;
    const double bad_reload_effective =
        moe_group_bad_reload_effective_locked(ctx, g, &bad_reload_age);
    const bool bad_reload_protected = bad_reload_effective >=
        moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_PROTECT_SCORE", 6.0);

    std::fprintf(ctx.evict_trace,
            "%s\t%llu\t%llu\t%d\t%d\t%s\t%llu\t%zu\t%.3f\t%.3f\t%.6f\t%.3f\t%.6f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%.8f\t%llu\t%llu\t%llu\t%llu\t%.3f\t%.8f\t%.8f\t%.8f\t%.8f\t%u\t%llu\t%.3f\t%.3f\t%llu\t%llu\t%llu\t%llu\t%d\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%u\t%llu\t%llu\n",
            event,
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            layer,
            expert,
            reason,
            (unsigned long long) reload_gap,
            released_bytes,
            ctx.resident_bytes / 1048576.0,
            ctx.params.budget_bytes / 1048576.0,
            score,
            reuse_keep,
            g.reuse_ema,
            g.inter_token_gap_ema,
            (unsigned long long) g.reuse_observed,
            (unsigned long long) g.reuse_within_1,
            (unsigned long long) g.reuse_within_4,
            (unsigned long long) g.reuse_within_16,
            (unsigned long long) g.seq_access,
            (unsigned long long) g.seq_rank0_access,
            seq_rate,
            (unsigned long long) g.seq_cache_hits,
            (unsigned long long) g.seq_cache_misses,
            (unsigned long long) g.seq_prefetch_hits,
            (unsigned long long) g.seq_prefetch_unused,
            g.hot_score,
            moe_eamc_prior(ctx, layer, expert),
            eam_replace_pred,
            eam_replace_mass,
            eam_replace_keep,
            (unsigned) cct_conf,
            (unsigned long long) g.cct_protect_hits,
            g.bad_reload_score,
            bad_reload_effective,
            (unsigned long long) bad_reload_age,
            (unsigned long long) g.bad_reload_1,
            (unsigned long long) g.bad_reload_4,
            (unsigned long long) g.bad_reload_16,
            bad_reload_protected ? 1 : 0,
            (unsigned long long) next_use_dist,
            layer_dist,
            g.pinned ? 1 : 0,
            active ? 1 : 0,
            recent ? 1 : 0,
            high_sequence ? 1 : 0,
            early_high ? 1 : 0,
            speculative_unused ? 1 : 0,
            layer_window ? 1 : 0,
            predicted_soon ? 1 : 0,
            (unsigned) g.last_evicted_reason,
            (unsigned long long) g.layer_window_bad_reload,
            (unsigned long long) g.layer_window_last_bad_reload_token);
}

static const char * moe_admission_action_label(uint8_t action) {
    return action == 1 ? "admit" : (action == 2 ? "bypass" : "none");
}

static void moe_admission_outcome_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        const moe_group_state &    g,
        uint64_t                   gap,
        const char *               reason) {
    if (ctx.admission_trace == nullptr || g.layer < 0 || g.expert < 0) {
        return;
    }
    std::fprintf(ctx.admission_trace,
            "%s\t%llu\t%llu\t%d\t%d\t%llu\t%s\t%llu\t%.6f\t%.6f\t%d\t%d\t%d\t%zu\t%.3f\t%.3f\t%.6f\t%llu\t%s\n",
            event,
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            g.layer,
            g.expert,
            (unsigned long long) g.admission_outcome_generation,
            moe_admission_action_label(g.admission_outcome_action),
            (unsigned long long) gap,
            g.admission_outcome_candidate,
            g.admission_outcome_victim,
            g.admission_outcome_victim_layer,
            g.admission_outcome_victim_expert,
            g.admission_outcome_victim_groups,
            g.admission_outcome_victim_bytes,
            ctx.resident_bytes / 1048576.0,
            ctx.params.budget_bytes / 1048576.0,
            g.admission_feedback_ema,
            (unsigned long long) g.admission_feedback_samples,
            reason);
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

static bool moe_group_prefetch_same_layer_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_GROUP_PREFETCH_SAME_LAYER", 0);
}

static int moe_group_prefetch_same_layer_max_groups() {
    return std::max(0, moe_env_i32("LLAMA_LAZY_MOE_GROUP_PREFETCH_MAX_GROUPS", 4));
}

static int moe_group_prefetch_same_layer_min_count() {
    return std::max(1, moe_env_i32("LLAMA_LAZY_MOE_GROUP_PREFETCH_MIN_COUNT", 1));
}

static bool moe_next_token_keep_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_NEXT_TOKEN_KEEP", 0);
}

static int moe_next_token_keep_min_conf() {
    return std::max(0, std::min(3, moe_env_i32("LLAMA_LAZY_MOE_NEXT_TOKEN_KEEP_MIN_CONF", 2)));
}

static uint64_t moe_next_token_keep_horizon() {
    return (uint64_t) std::max(1, moe_env_i32("LLAMA_LAZY_MOE_NEXT_TOKEN_KEEP_HORIZON", 1));
}

static uint8_t moe_sat_inc2(uint8_t v) {
    return (uint8_t) std::min(3, (int) v + 1);
}

static uint8_t moe_sat_dec2(uint8_t v) {
    return (uint8_t) std::max(0, (int) v - 1);
}

static void moe_next_token_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        int                        layer,
        int                        expert,
        int                        conf_before,
        int                        conf_after,
        uint64_t                   protect_until,
        const char *               reason) {
    if (ctx.next_token_trace == nullptr || layer < 0 || expert < 0) {
        return;
    }
    std::fprintf(ctx.next_token_trace,
            "%s\t%llu\t%llu\t%d\t%d\t%d\t%d\t%llu\t%.3f\t%.3f\t%s\n",
            event,
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            layer,
            expert,
            conf_before,
            conf_after,
            (unsigned long long) protect_until,
            ctx.resident_bytes / 1048576.0,
            ctx.params.budget_bytes / 1048576.0,
            reason == nullptr ? "" : reason);
}

static bool moe_next_token_group_protected_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g) {
    return moe_next_token_keep_enabled() &&
        g.next_token_conf >= (uint8_t) moe_next_token_keep_min_conf() &&
        g.next_token_protect_until >= ctx.profile_token_epoch;
}

static void moe_next_token_update_after_route_locked(
        llama_moe_buffer_context & ctx,
        int                        layer,
        const std::vector<int> &   last_token_experts,
        int                        n_expert) {
    if (!moe_next_token_keep_enabled() || layer < 0 || n_expert <= 0) {
        return;
    }

    if ((int) ctx.next_token_last_layer_experts.size() <= layer) {
        ctx.next_token_last_layer_experts.resize((size_t) layer + 1);
    }

    std::vector<char> actual((size_t) n_expert, 0);
    for (int expert : last_token_experts) {
        if (expert >= 0 && expert < n_expert) {
            actual[(size_t) expert] = 1;
        }
    }

    std::vector<int> & previous = ctx.next_token_last_layer_experts[(size_t) layer];
    for (int expert : previous) {
        if (expert < 0 || expert >= n_expert) {
            continue;
        }
        moe_group_state & g = moe_group_get(ctx, layer, expert);
        const uint8_t before = g.next_token_conf;
        if (actual[(size_t) expert]) {
            g.next_token_conf = moe_sat_inc2(g.next_token_conf);
            g.next_token_hit++;
            ctx.next_token_hits.fetch_add(1, std::memory_order_relaxed);
            moe_next_token_trace_write_locked(ctx, "hit", layer, expert, before, g.next_token_conf,
                    g.next_token_protect_until, "predicted_actual");
        } else {
            g.next_token_conf = moe_sat_dec2(g.next_token_conf);
            g.next_token_miss++;
            ctx.next_token_misses.fetch_add(1, std::memory_order_relaxed);
            moe_next_token_trace_write_locked(ctx, "miss", layer, expert, before, g.next_token_conf,
                    g.next_token_protect_until, "predicted_unused");
        }
    }

    std::vector<int> deduped;
    deduped.reserve(last_token_experts.size());
    for (int expert : last_token_experts) {
        if (expert < 0 || expert >= n_expert || !actual[(size_t) expert]) {
            continue;
        }
        actual[(size_t) expert] = 0;
        deduped.push_back(expert);
        moe_group_state & g = moe_group_get(ctx, layer, expert);
        const uint8_t before = g.next_token_conf;
        if (g.next_token_conf < 2) {
            g.next_token_conf = 2;
            ctx.next_token_admit.fetch_add(1, std::memory_order_relaxed);
            moe_next_token_trace_write_locked(ctx, "admit", layer, expert, before, g.next_token_conf,
                    g.next_token_protect_until, "actual_unseen");
        }
        if (g.next_token_conf >= (uint8_t) moe_next_token_keep_min_conf()) {
            g.next_token_protect_until = ctx.profile_token_epoch + moe_next_token_keep_horizon();
            g.next_token_predicted++;
            ctx.next_token_predicted.fetch_add(1, std::memory_order_relaxed);
            moe_next_token_trace_write_locked(ctx, "protect", layer, expert, g.next_token_conf, g.next_token_conf,
                    g.next_token_protect_until, "same_layer_next_token");
        }
    }

    previous.swap(deduped);
    ctx.next_token_updates.fetch_add(1, std::memory_order_relaxed);
}

static size_t moe_env_mib_bytes(const char * name, int def_mib) {
    return (size_t) std::max(0, moe_env_i32(name, def_mib)) * 1048576ull;
}

static void moe_atomic_max_u64(std::atomic<uint64_t> & dst, uint64_t value) {
    uint64_t old = dst.load(std::memory_order_relaxed);
    while (old < value && !dst.compare_exchange_weak(old, value, std::memory_order_relaxed)) {}
}

static bool moe_cct_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_CCT", 0);
}

static bool moe_cct_prefetch_enabled() {
    return moe_cct_enabled() && moe_env_flag("LLAMA_LAZY_MOE_CCT_PREFETCH", 0);
}

static bool moe_cct_evict_enabled() {
    return moe_cct_enabled() && moe_env_flag("LLAMA_LAZY_MOE_CCT_EVICT", 0);
}

static bool moe_cct_lowmem_active(const llama_moe_buffer_context & ctx) {
    const int threshold_mb = moe_env_i32("LLAMA_LAZY_MOE_CCT_LOW_BUDGET_MB", 640);
    return ctx.params.budget_bytes > 0 &&
        ctx.params.budget_bytes <= (size_t) std::max(0, threshold_mb) * 1048576ull;
}

static void moe_cct_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        int                        src_layer,
        int                        src_expert,
        int                        target_layer,
        int                        target_expert,
        int                        conf_before,
        int                        conf_after,
        const char *               reason) {
    if (ctx.cct_trace == nullptr) {
        return;
    }
    std::fprintf(ctx.cct_trace,
            "%s\t%llu\t%llu\t%d\t%d\t%d\t%d\t%d\t%d\t%s\t%.3f\t%.3f\n",
            event,
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            src_layer,
            src_expert,
            target_layer,
            target_expert,
            conf_before,
            conf_after,
            reason == nullptr ? "" : reason,
            ctx.resident_bytes / 1048576.0,
            ctx.params.budget_bytes / 1048576.0);
}

static uint8_t moe_cct_conf_from_source_locked(
        const llama_moe_buffer_context & ctx,
        int                              src_layer,
        int                              src_expert,
        int                              target_expert) {
    if (!moe_cct_enabled() || src_layer < 0 || src_expert < 0 || target_expert < 0) {
        return 0;
    }
    auto it = ctx.cct_transition.find(moe_group_key(src_layer, src_expert));
    if (it == ctx.cct_transition.end() || target_expert >= (int) it->second.size()) {
        return 0;
    }
    return it->second[(size_t) target_expert].conf;
}

static uint8_t moe_cct_predict_conf_locked(
        const llama_moe_buffer_context & ctx,
        int                              target_layer,
        int                              target_expert) {
    if (!moe_cct_enabled() || target_layer < 0 || target_expert < 0) {
        return 0;
    }
    const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
    const int src_layer = (target_layer + n_layer - 1) % n_layer;

    const std::vector<int> * src_experts = nullptr;
    if (ctx.eam_last_actual_layer == src_layer && !ctx.eam_last_layer_experts.empty()) {
        src_experts = &ctx.eam_last_layer_experts;
    } else if (src_layer >= 0 && src_layer < (int) ctx.cct_recent_layer_experts.size() &&
            !ctx.cct_recent_layer_experts[(size_t) src_layer].empty()) {
        src_experts = &ctx.cct_recent_layer_experts[(size_t) src_layer];
    }
    if (src_experts == nullptr) {
        return 0;
    }
    uint8_t best = 0;
    for (int src_expert : *src_experts) {
        best = std::max(best, moe_cct_conf_from_source_locked(ctx, src_layer, src_expert, target_expert));
    }
    return best;
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
        return 16ull * 1048576ull;
    }
    if (ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 3072ull * 1048576ull) {
        return 192ull * 1048576ull;
    }
    return 256ull * 1048576ull;
}

static bool moe_force_next_layer_prefetch_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_FORCE_NEXT_LAYER_PREFETCH", 0);
}

static void moe_layer_speculative_pending_locked(
        const llama_moe_buffer_context & ctx,
        int                              layer,
        int                              candidate_expert,
        size_t *                         out_groups,
        size_t *                         out_bytes,
        bool *                           out_candidate_pending) {
    size_t groups = 0;
    size_t bytes = 0;
    bool candidate_pending = false;
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end() || it->second.empty() || it->second[0] == nullptr) {
        if (out_groups != nullptr) {
            *out_groups = 0;
        }
        if (out_bytes != nullptr) {
            *out_bytes = 0;
        }
        if (out_candidate_pending != nullptr) {
            *out_candidate_pending = false;
        }
        return;
    }

    const int n_expert = it->second[0]->n_expert;
    std::vector<char> seen((size_t) std::max(0, n_expert), false);
    for (const moe_managed * m : it->second) {
        if (m == nullptr) {
            continue;
        }
        for (int e = 0; e < m->n_expert; ++e) {
            bool pending = false;
            if (e < (int) m->queued.size() && m->queued[e]) {
                pending = true;
                const int bits = e < (int) m->queued_bits.size() && m->queued_bits[e] > 0 ?
                    m->queued_bits[e] : ctx.params.base_bits;
                bytes += moe_effective_stream_bytes_for_target(ctx, *m, e, bits);
            }
            if (e < (int) m->prefetched.size() && m->prefetched[e] &&
                    e < (int) m->resident_touched.size() && !m->resident_touched[e]) {
                pending = true;
                bytes += moe_slice_resident_bytes(*m, e);
            }
            if (pending && e == candidate_expert) {
                candidate_pending = true;
            }
            if (pending && e >= 0 && e < (int) seen.size() && !seen[(size_t) e]) {
                seen[(size_t) e] = true;
                ++groups;
            }
        }
    }

    if (out_groups != nullptr) {
        *out_groups = groups;
    }
    if (out_bytes != nullptr) {
        *out_bytes = bytes;
    }
    if (out_candidate_pending != nullptr) {
        *out_candidate_pending = candidate_pending;
    }
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

static bool moe_reuse_evict_enabled() {
    return moe_env_i32("LLAMA_LAZY_MOE_REUSE_EVICT", 0) > 0;
}

static bool moe_bad_reload_evict_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_BAD_RELOAD_EVICT", 1);
}

static bool moe_ghost_feedback_enabled() {
    static const bool enabled = moe_env_flag("LLAMA_LAZY_MOE_GHOST_FEEDBACK", 0);
    return enabled;
}

static uint64_t moe_ghost_feedback_horizon() {
    static const uint64_t horizon =
        (uint64_t) std::max(1, moe_env_i32("LLAMA_LAZY_MOE_GHOST_HORIZON_TOKENS", 16));
    return horizon;
}

static void moe_group_ghost_update_locked(
        llama_moe_buffer_context & ctx,
        moe_group_state &          g,
        double                     target,
        bool                       success,
        uint64_t                   reload_gap) {
    static const double alpha = std::max(0.0, std::min(1.0,
            moe_env_f64("LLAMA_LAZY_MOE_GHOST_ALPHA", 0.25)));
    g.ghost_reload_risk_ema =
        (1.0 - alpha) * g.ghost_reload_risk_ema + alpha * std::max(0.0, std::min(1.0, target));
    g.ghost_outcomes++;
    g.ghost_last_outcome_token = ctx.profile_token_epoch;
    if (success) {
        g.ghost_success++;
        ctx.ghost_success.fetch_add(1, std::memory_order_relaxed);
    } else if (reload_gap <= 1) {
        g.ghost_bad_1++;
        ctx.ghost_bad_1.fetch_add(1, std::memory_order_relaxed);
    } else if (reload_gap <= 4) {
        g.ghost_bad_4++;
        ctx.ghost_bad_4.fetch_add(1, std::memory_order_relaxed);
    } else {
        g.ghost_bad_16++;
        ctx.ghost_bad_16.fetch_add(1, std::memory_order_relaxed);
    }
}

static void moe_group_ghost_note_reload_locked(
        llama_moe_buffer_context & ctx,
        moe_group_state &          g,
        uint64_t                   reload_gap) {
    if (!moe_ghost_feedback_enabled() || !g.ghost_pending) {
        return;
    }
    g.ghost_pending = false;
    const uint64_t horizon = moe_ghost_feedback_horizon();
    if (reload_gap > horizon) {
        moe_group_ghost_update_locked(ctx, g, 0.0, true, reload_gap);
        return;
    }
    static const double target_1 = moe_env_f64("LLAMA_LAZY_MOE_GHOST_TARGET_1", 1.0);
    static const double target_4 = moe_env_f64("LLAMA_LAZY_MOE_GHOST_TARGET_4", 0.65);
    static const double target_16 = moe_env_f64("LLAMA_LAZY_MOE_GHOST_TARGET_16", 0.25);
    const double target = reload_gap <= 1 ? target_1 : (reload_gap <= 4 ? target_4 : target_16);
    moe_group_ghost_update_locked(ctx, g, target, false, reload_gap);
}

static void moe_ghost_expire_locked(llama_moe_buffer_context & ctx) {
    if (!moe_ghost_feedback_enabled()) {
        return;
    }
    const uint64_t token = ctx.profile_token_epoch;
    while (!ctx.ghost_records.empty() && ctx.ghost_records.front().deadline < token) {
        const moe_ghost_record record = ctx.ghost_records.front();
        ctx.ghost_records.pop_front();
        const auto it = ctx.groups.find(record.key);
        if (it == ctx.groups.end()) {
            continue;
        }
        moe_group_state & g = it->second;
        if (!g.ghost_pending || g.ghost_generation != record.generation) {
            continue;
        }
        g.ghost_pending = false;
        moe_group_ghost_update_locked(ctx, g, 0.0, true, token - record.deadline);
    }
}

static void moe_admission_outcome_note_reuse_locked(
        llama_moe_buffer_context & ctx,
        moe_group_state &          g) {
    if (!g.admission_outcome_pending ||
            ctx.profile_token_epoch <= g.admission_outcome_token) {
        return;
    }
    const uint64_t gap = ctx.profile_token_epoch - g.admission_outcome_token;
    double feedback_target = 0.0;
    if (g.admission_outcome_action == 1) {
        if (gap <= 1) {
            ctx.admission_outcome_admit_reuse_1.fetch_add(1, std::memory_order_relaxed);
            feedback_target = 4.0;
        } else if (gap <= 4) {
            ctx.admission_outcome_admit_reuse_4.fetch_add(1, std::memory_order_relaxed);
            feedback_target = 2.0;
        } else {
            ctx.admission_outcome_admit_reuse_16.fetch_add(1, std::memory_order_relaxed);
            feedback_target = 0.5;
        }
    } else if (g.admission_outcome_action == 2) {
        if (gap <= 1) {
            ctx.admission_outcome_bypass_reload_1.fetch_add(1, std::memory_order_relaxed);
            feedback_target = 8.0;
        } else if (gap <= 4) {
            ctx.admission_outcome_bypass_reload_4.fetch_add(1, std::memory_order_relaxed);
            feedback_target = 4.0;
        } else {
            ctx.admission_outcome_bypass_reload_16.fetch_add(1, std::memory_order_relaxed);
            feedback_target = 1.0;
        }
    }
    const double alpha = std::max(0.0, std::min(1.0,
            moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_FEEDBACK_ALPHA", 0.125)));
    g.admission_feedback_ema =
        (1.0 - alpha) * g.admission_feedback_ema + alpha * feedback_target;
    g.admission_feedback_samples++;
    ctx.admission_feedback_updates.fetch_add(1, std::memory_order_relaxed);
    ctx.admission_feedback_positive.fetch_add(1, std::memory_order_relaxed);
    moe_admission_outcome_trace_write_locked(
            ctx,
            g.admission_outcome_action == 1 ? "reuse" : "reload",
            g,
            gap,
            g.admission_outcome_action == 1 ? "admit_reused" : "bypass_reloaded");
    g.admission_outcome_pending = false;
}

static void moe_admission_outcome_note_evicted_locked(
        llama_moe_buffer_context & ctx,
        moe_group_state &          g) {
    if (!g.admission_outcome_pending || g.admission_outcome_action != 1) {
        return;
    }
    const uint64_t gap = ctx.profile_token_epoch >= g.admission_outcome_token ?
        ctx.profile_token_epoch - g.admission_outcome_token : 0;
    const double alpha = std::max(0.0, std::min(1.0,
            moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_FEEDBACK_ALPHA", 0.125)));
    g.admission_feedback_ema =
        (1.0 - alpha) * g.admission_feedback_ema + alpha * -4.0;
    g.admission_feedback_samples++;
    ctx.admission_feedback_updates.fetch_add(1, std::memory_order_relaxed);
    ctx.admission_feedback_negative.fetch_add(1, std::memory_order_relaxed);
    ctx.admission_outcome_admit_unused.fetch_add(1, std::memory_order_relaxed);
    moe_admission_outcome_trace_write_locked(
            ctx, "evict_unused", g, gap, "admit_evicted_before_reuse");
    g.admission_outcome_pending = false;
}

static void moe_admission_outcome_expire_locked(llama_moe_buffer_context & ctx) {
    for (auto & kv : ctx.groups) {
        moe_group_state & g = kv.second;
        if (!g.admission_outcome_pending ||
                ctx.profile_token_epoch <= g.admission_outcome_deadline) {
            continue;
        }
        const uint64_t gap = ctx.profile_token_epoch - g.admission_outcome_token;
        const double target = g.admission_outcome_action == 1 ? -2.0 : -1.0;
        const double alpha = std::max(0.0, std::min(1.0,
                moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_FEEDBACK_ALPHA", 0.125)));
        g.admission_feedback_ema =
            (1.0 - alpha) * g.admission_feedback_ema + alpha * target;
        g.admission_feedback_samples++;
        ctx.admission_feedback_updates.fetch_add(1, std::memory_order_relaxed);
        ctx.admission_feedback_negative.fetch_add(1, std::memory_order_relaxed);
        if (g.admission_outcome_action == 1) {
            ctx.admission_outcome_admit_unused.fetch_add(1, std::memory_order_relaxed);
            moe_admission_outcome_trace_write_locked(
                    ctx, "expire", g, gap, "admit_not_reused");
        } else if (g.admission_outcome_action == 2) {
            ctx.admission_outcome_bypass_success.fetch_add(1, std::memory_order_relaxed);
            moe_admission_outcome_trace_write_locked(
                    ctx, "expire", g, gap, "bypass_not_reloaded");
        }
        g.admission_outcome_pending = false;
    }
}

static void moe_admission_outcome_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason) {
    for (auto & kv : ctx.groups) {
        moe_group_state & g = kv.second;
        if (!g.admission_outcome_pending) {
            continue;
        }
        const uint64_t gap = ctx.profile_token_epoch >= g.admission_outcome_token ?
            ctx.profile_token_epoch - g.admission_outcome_token : 0;
        moe_admission_outcome_trace_write_locked(ctx, "unfinished", g, gap, reason);
        g.admission_outcome_pending = false;
    }
}

static double moe_admission_regret_cost(uint64_t gap) {
    if (gap <= 1) {
        return 8.0;
    }
    if (gap <= 4) {
        return 4.0;
    }
    if (gap <= 16) {
        return 1.0;
    }
    return 0.0;
}

static uint64_t moe_token_gap_or_max(uint64_t now, uint64_t then) {
    if (then == 0 || now < then) {
        return UINT64_MAX;
    }
    return now - then;
}

static uint64_t moe_lrb_resident_group_count_locked(const llama_moe_buffer_context & ctx) {
    uint64_t n = 0;
    for (const auto & kv : ctx.groups) {
        if (moe_group_resident_bytes(ctx, kv.second.layer, kv.second.expert) > 0) {
            n++;
        }
    }
    return n;
}

static moe_lrb_group_features moe_lrb_features_snapshot_locked(
        llama_moe_buffer_context & ctx,
        const moe_group_state &    g) {
    moe_lrb_group_features f;
    f.layer = g.layer;
    f.expert = g.expert;
    f.last_touch_gap = moe_token_gap_or_max(ctx.profile_token_epoch, g.last_used_token_epoch);
    f.last_evict_gap = g.last_evicted_valid ?
        moe_token_gap_or_max(ctx.profile_token_epoch, g.last_evicted_token_epoch) : UINT64_MAX;
    if (g.next_use_epoch != UINT64_MAX && g.next_use_epoch > ctx.exec_epoch) {
        f.next_use_dist = g.next_use_epoch - ctx.exec_epoch;
    }
    f.layer_dist = moe_forward_layer_distance(ctx, g.layer);
    f.resident_bytes = moe_group_resident_bytes(ctx, g.layer, g.expert);
    f.cache_score = moe_group_cache_score(ctx, g, f.resident_bytes);
    f.reuse_ema = g.reuse_ema;
    f.inter_token_gap_ema = g.inter_token_gap_ema;
    f.reuse_observed = g.reuse_observed;
    f.reuse_within_1 = g.reuse_within_1;
    f.reuse_within_4 = g.reuse_within_4;
    f.reuse_within_16 = g.reuse_within_16;
    f.bad_reload_score = g.bad_reload_score;
    f.bad_reload_effective = moe_group_bad_reload_effective_locked(ctx, g, &f.bad_reload_age);
    f.bad_reload_1 = g.bad_reload_1;
    f.bad_reload_4 = g.bad_reload_4;
    f.bad_reload_16 = g.bad_reload_16;
    f.ghost_reload_risk_ema = g.ghost_reload_risk_ema;
    f.ghost_outcomes = g.ghost_outcomes;
    f.admission_feedback_ema = g.admission_feedback_ema;
    f.admission_feedback_samples = g.admission_feedback_samples;
    f.admission_regret_ema = g.admission_regret_ema;
    f.admission_regret_samples = g.admission_regret_samples;
    f.seq_access = g.seq_access;
    f.seq_rank0_access = g.seq_rank0_access;
    f.seq_cache_hits = g.seq_cache_hits;
    f.seq_cache_misses = g.seq_cache_misses;
    f.seq_prefetch_hits = g.seq_prefetch_hits;
    f.seq_prefetch_late = g.seq_prefetch_late;
    f.seq_prefetch_unused = g.seq_prefetch_unused;
    f.seq_future_hints = g.seq_future_hints;
    f.seq_future_rank0_hints = g.seq_future_rank0_hints;
    f.seq_predicted = g.seq_predicted;
    f.seq_pred_enqueued = g.seq_pred_enqueued;
    f.eamc_prior = moe_eamc_prior(ctx, g.layer, g.expert);
    f.cct_conf = moe_cct_predict_conf_locked(ctx, g.layer, g.expert);
    f.next_token_conf = g.next_token_conf;
    f.pinned = g.pinned;
    f.active = moe_group_is_active(ctx, g);
    f.demand_pending = g.demand_async_pending || g.demand_admission_bypass;
    return f;
}

static void moe_lrb_trace_write_features(FILE * trace, const moe_lrb_group_features & f) {
    std::fprintf(trace,
            "\t%d\t%d\t%llu\t%llu\t%llu\t%d\t%.3f\t%.6f\t%.6f\t%.6f\t%llu\t%llu\t%llu\t%llu\t%.6f\t%.6f\t%llu\t%llu\t%llu\t%llu\t%.6f\t%llu\t%.6f\t%llu\t%.6f\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%llu\t%.9f\t%u\t%u\t%d\t%d\t%d",
            f.layer,
            f.expert,
            (unsigned long long) f.last_touch_gap,
            (unsigned long long) f.last_evict_gap,
            (unsigned long long) f.next_use_dist,
            f.layer_dist,
            (double) f.resident_bytes / 1048576.0,
            f.cache_score,
            f.reuse_ema,
            f.inter_token_gap_ema,
            (unsigned long long) f.reuse_observed,
            (unsigned long long) f.reuse_within_1,
            (unsigned long long) f.reuse_within_4,
            (unsigned long long) f.reuse_within_16,
            f.bad_reload_score,
            f.bad_reload_effective,
            (unsigned long long) f.bad_reload_age,
            (unsigned long long) f.bad_reload_1,
            (unsigned long long) f.bad_reload_4,
            (unsigned long long) f.bad_reload_16,
            f.ghost_reload_risk_ema,
            (unsigned long long) f.ghost_outcomes,
            f.admission_feedback_ema,
            (unsigned long long) f.admission_feedback_samples,
            f.admission_regret_ema,
            (unsigned long long) f.admission_regret_samples,
            (unsigned long long) f.seq_access,
            (unsigned long long) f.seq_rank0_access,
            (unsigned long long) f.seq_cache_hits,
            (unsigned long long) f.seq_cache_misses,
            (unsigned long long) f.seq_prefetch_hits,
            (unsigned long long) f.seq_prefetch_late,
            (unsigned long long) f.seq_prefetch_unused,
            (unsigned long long) f.seq_future_hints,
            (unsigned long long) f.seq_future_rank0_hints,
            (unsigned long long) f.seq_predicted,
            (unsigned long long) f.seq_pred_enqueued,
            f.eamc_prior,
            (unsigned) f.cct_conf,
            (unsigned) f.next_token_conf,
            f.pinned ? 1 : 0,
            f.active ? 1 : 0,
            f.demand_pending ? 1 : 0);
}

static void moe_lrb_trace_write_locked(
        llama_moe_buffer_context &        ctx,
        const char *                      event,
        const moe_admission_pair_record & record,
        double                            target,
        const char *                      reason) {
    if (ctx.lrb_trace == nullptr) {
        return;
    }
    const char * preference =
        target > 0.0 ? "admit" : (target < 0.0 ? "bypass" : "tie");
    const char * action = record.bypass ? "bypass" : "admit";
    int correct = -1;
    if (target != 0.0) {
        correct = (record.bypass == (target < 0.0)) ? 1 : 0;
    }
    std::fprintf(ctx.lrb_trace,
            "%s\t%llu\t%llu\t%llu\t%s\t%llu\t%llu\t%llu\t%llu\t%.6f\t%s\t%d\t%.3f\t%.3f\t%.3f\t%llu",
            event,
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            (unsigned long long) record.id,
            action,
            (unsigned long long) record.decision_token,
            (unsigned long long) record.deadline,
            (unsigned long long) record.incoming_gap,
            (unsigned long long) record.victim_gap,
            target,
            preference,
            correct,
            (double) ctx.resident_bytes / 1048576.0,
            (double) ctx.params.budget_bytes / 1048576.0,
            (double) ctx.demand_admission_staging_bytes / 1048576.0,
            (unsigned long long) moe_lrb_resident_group_count_locked(ctx));
    moe_lrb_trace_write_features(ctx.lrb_trace, record.incoming_features);
    moe_lrb_trace_write_features(ctx.lrb_trace, record.victim_features);
    std::fprintf(ctx.lrb_trace, "\t%s\n", reason);
}

static void moe_admission_regret_trace_write_locked(
        llama_moe_buffer_context &       ctx,
        const char *                     event,
        const moe_admission_pair_record & record,
        double                           target,
        const char *                     reason) {
    if (ctx.admission_regret_trace == nullptr) {
        return;
    }
    const auto incoming_it = ctx.groups.find(record.incoming_key);
    const auto victim_it = ctx.groups.find(record.victim_key);
    if (incoming_it == ctx.groups.end() || victim_it == ctx.groups.end()) {
        return;
    }
    const moe_group_state & incoming = incoming_it->second;
    const moe_group_state & victim = victim_it->second;
    std::fprintf(ctx.admission_regret_trace,
            "%s\t%llu\t%llu\t%llu\t%s\t%d\t%d\t%d\t%d\t%llu\t%llu\t%.6f\t%.6f\t%llu\t%s\n",
            event,
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            (unsigned long long) record.id,
            record.bypass ? "bypass" : "admit",
            incoming.layer,
            incoming.expert,
            victim.layer,
            victim.expert,
            (unsigned long long) record.incoming_gap,
            (unsigned long long) record.victim_gap,
            target,
            incoming.admission_regret_ema,
            (unsigned long long) incoming.admission_regret_samples,
            reason);
}

static void moe_admission_regret_resolve_locked(
        llama_moe_buffer_context & ctx,
        moe_admission_pair_record & record,
        bool                       expired) {
    const double incoming_cost = moe_admission_regret_cost(record.incoming_gap);
    const double victim_cost = moe_admission_regret_cost(record.victim_gap);
    const double target = incoming_cost - victim_cost;
    auto incoming_it = ctx.groups.find(record.incoming_key);
    if (incoming_it == ctx.groups.end()) {
        return;
    }
    moe_group_state & incoming = incoming_it->second;
    if (target != 0.0) {
        const double alpha = std::max(0.0, std::min(1.0,
                moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_REGRET_ALPHA", 0.125)));
        incoming.admission_regret_ema =
            (1.0 - alpha) * incoming.admission_regret_ema + alpha * target;
        incoming.admission_regret_samples++;
        if (target > 0.0) {
            ctx.admission_regret_prefer_admit.fetch_add(1, std::memory_order_relaxed);
        } else {
            ctx.admission_regret_prefer_bypass.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        ctx.admission_regret_tie.fetch_add(1, std::memory_order_relaxed);
    }
    if (expired) {
        ctx.admission_regret_expired.fetch_add(1, std::memory_order_relaxed);
    } else {
        ctx.admission_regret_resolved.fetch_add(1, std::memory_order_relaxed);
    }
    moe_lrb_trace_write_locked(
            ctx,
            expired ? "expire" : "resolve",
            record,
            target,
            target > 0.0 ? "incoming_sooner" :
                (target < 0.0 ? "victim_sooner" : "equal_or_unseen"));
    moe_admission_regret_trace_write_locked(
            ctx,
            expired ? "expire" : "resolve",
            record,
            target,
            target > 0.0 ? "incoming_sooner" :
                (target < 0.0 ? "victim_sooner" : "equal_or_unseen"));
}

static void moe_admission_regret_compact_watchers_locked(
        llama_moe_buffer_context & ctx,
        uint64_t                   key) {
    auto it = ctx.admission_pair_watchers.find(key);
    if (it == ctx.admission_pair_watchers.end()) {
        return;
    }
    std::vector<uint64_t> & ids = it->second;
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](uint64_t id) {
        return ctx.admission_pairs.find(id) == ctx.admission_pairs.end();
    }), ids.end());
    if (ids.empty()) {
        ctx.admission_pair_watchers.erase(it);
    }
}

static void moe_admission_regret_begin_locked(
        llama_moe_buffer_context & ctx,
        const moe_group_state &    incoming,
        bool                       bypass,
        const std::vector<uint64_t> & victim_keys) {
    const uint64_t incoming_key = moe_group_key(incoming.layer, incoming.expert);
    const uint64_t horizon = (uint64_t) std::max(1,
            moe_env_i32("LLAMA_LAZY_MOE_ADMISSION_REGRET_TOKENS", 16));
    for (uint64_t victim_key : victim_keys) {
        if (victim_key == incoming_key || ctx.groups.find(victim_key) == ctx.groups.end()) {
            continue;
        }
        if (ctx.admission_pair_watchers[incoming_key].size() >= 64) {
            moe_admission_regret_compact_watchers_locked(ctx, incoming_key);
        }
        if (ctx.admission_pair_watchers[victim_key].size() >= 64) {
            moe_admission_regret_compact_watchers_locked(ctx, victim_key);
        }

        moe_admission_pair_record record;
        record.id = ctx.next_admission_pair_id++;
        record.incoming_key = incoming_key;
        record.victim_key = victim_key;
        record.decision_token = ctx.profile_token_epoch;
        record.deadline = ctx.profile_token_epoch > UINT64_MAX - horizon ?
            UINT64_MAX : ctx.profile_token_epoch + horizon;
        record.bypass = bypass;
        record.incoming_features = moe_lrb_features_snapshot_locked(ctx, incoming);
        record.victim_features = moe_lrb_features_snapshot_locked(
                ctx, ctx.groups.find(victim_key)->second);
        ctx.admission_pairs.emplace(record.id, record);
        ctx.admission_pair_watchers[incoming_key].push_back(record.id);
        ctx.admission_pair_watchers[victim_key].push_back(record.id);
        ctx.admission_pair_deadlines.push_back({record.deadline, record.id});
        ctx.admission_regret_pairs.fetch_add(1, std::memory_order_relaxed);
        moe_lrb_trace_write_locked(ctx, "decision", record, 0.0, "paired");
        moe_admission_regret_trace_write_locked(ctx, "decision", record, 0.0, "paired");
    }
}

static void moe_admission_regret_note_touch_locked(
        llama_moe_buffer_context & ctx,
        uint64_t                   key) {
    auto watcher_it = ctx.admission_pair_watchers.find(key);
    if (watcher_it == ctx.admission_pair_watchers.end()) {
        return;
    }
    std::vector<uint64_t> ids = watcher_it->second;
    for (uint64_t id : ids) {
        auto pair_it = ctx.admission_pairs.find(id);
        if (pair_it == ctx.admission_pairs.end()) {
            continue;
        }
        moe_admission_pair_record & record = pair_it->second;
        if (ctx.profile_token_epoch <= record.decision_token) {
            continue;
        }
        const uint64_t gap = ctx.profile_token_epoch - record.decision_token;
        if (record.incoming_key == key && record.incoming_gap == UINT64_MAX) {
            record.incoming_gap = gap;
        }
        if (record.victim_key == key && record.victim_gap == UINT64_MAX) {
            record.victim_gap = gap;
        }
        if (record.incoming_gap != UINT64_MAX && record.victim_gap != UINT64_MAX) {
            moe_admission_pair_record completed = record;
            moe_admission_regret_resolve_locked(ctx, completed, false);
            ctx.admission_pairs.erase(pair_it);
        }
    }
    moe_admission_regret_compact_watchers_locked(ctx, key);
}

static void moe_admission_regret_expire_locked(llama_moe_buffer_context & ctx) {
    while (!ctx.admission_pair_deadlines.empty() &&
            ctx.admission_pair_deadlines.front().deadline < ctx.profile_token_epoch) {
        const moe_admission_pair_deadline deadline =
            ctx.admission_pair_deadlines.front();
        ctx.admission_pair_deadlines.pop_front();
        auto it = ctx.admission_pairs.find(deadline.id);
        if (it == ctx.admission_pairs.end()) {
            continue;
        }
        moe_admission_pair_record expired = it->second;
        moe_admission_regret_resolve_locked(ctx, expired, true);
        ctx.admission_pairs.erase(it);
    }
}

static void moe_admission_regret_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason) {
    for (const auto & kv : ctx.admission_pairs) {
        moe_lrb_trace_write_locked(
                ctx, "unfinished", kv.second, 0.0, reason);
        moe_admission_regret_trace_write_locked(
                ctx, "unfinished", kv.second, 0.0, reason);
    }
    ctx.admission_pairs.clear();
    ctx.admission_pair_watchers.clear();
    ctx.admission_pair_deadlines.clear();
}

static double moe_olecar_delay_cost(uint64_t gap) {
    if (gap <= 1) {
        return 1.0;
    }
    if (gap <= 4) {
        return 0.5;
    }
    if (gap <= 16) {
        return 0.125;
    }
    return 0.0;
}

static int moe_olecar_budget_bucket(const llama_moe_buffer_context & ctx) {
    if (ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 640ull * 1048576ull) {
        return 0;
    }
    if (ctx.params.budget_bytes > 0 && ctx.params.budget_bytes <= 768ull * 1048576ull) {
        return 1;
    }
    return 2;
}

static void moe_olecar_init_weights_locked(llama_moe_buffer_context & ctx, int bucket) {
    if (bucket < 0 || bucket >= MOE_OLECAR_BUCKET_COUNT) {
        return;
    }
    double sum = 0.0;
    for (double w : ctx.olecar_weights[(size_t) bucket]) {
        sum += w;
    }
    if (sum > 0.0) {
        return;
    }
    const double uniform = 1.0 / (double) MOE_OLECAR_POLICY_COUNT;
    for (double & w : ctx.olecar_weights[(size_t) bucket]) {
        w = uniform;
    }
}

static double moe_olecar_weight_sum_locked(llama_moe_buffer_context & ctx, int bucket) {
    moe_olecar_init_weights_locked(ctx, bucket);
    double sum = 0.0;
    if (bucket < 0 || bucket >= MOE_OLECAR_BUCKET_COUNT) {
        return 0.0;
    }
    for (double w : ctx.olecar_weights[(size_t) bucket]) {
        sum += w;
    }
    return sum;
}

static double moe_olecar_policy_prob_locked(
        llama_moe_buffer_context & ctx,
        int                        bucket,
        int                        policy_id) {
    const double sum = moe_olecar_weight_sum_locked(ctx, bucket);
    if (sum <= 0.0 || policy_id < 0 || policy_id >= MOE_OLECAR_POLICY_COUNT ||
            bucket < 0 || bucket >= MOE_OLECAR_BUCKET_COUNT) {
        return 0.0;
    }
    return ctx.olecar_weights[(size_t) bucket][(size_t) policy_id] / sum;
}

static int moe_olecar_policy_family(int policy_id) {
    switch (policy_id) {
        case 0: // final
        case 3: // cache
            return 0;
        case 1: // lru
        case 2: // recency
        case 7: // reuse
        case 8: // cct
            return 1;
        case 5: // next_use
            return 2;
        case 4: // bad_reload
            return 3;
        case 6: // layer
            return 4;
        default:
            return -1;
    }
}

static void moe_olecar_shadow_update_locked(
        llama_moe_buffer_context & ctx,
        moe_olecar_record &        record,
        uint64_t                   gap,
        double                     cost) {
    if (!moe_env_flag("LLAMA_LAZY_MOE_OLECAR_SHADOW_UPDATE", 1) ||
            cost <= 0.0 || record.policy_id < 0 ||
            record.policy_id >= MOE_OLECAR_POLICY_COUNT ||
            record.bucket_id < 0 || record.bucket_id >= MOE_OLECAR_BUCKET_COUNT) {
        record.weight_after = record.weight_before;
        return;
    }

    moe_olecar_init_weights_locked(ctx, record.bucket_id);
    auto & weights = ctx.olecar_weights[(size_t) record.bucket_id];
    double & weight = weights[(size_t) record.policy_id];
    record.weight_before = weight;

    const double min_prob = moe_env_f64("LLAMA_LAZY_MOE_OLECAR_MIN_PROB", 0.10);
    const double denom = std::max(min_prob, record.support_prob);
    const double delay = (double) std::max<uint64_t>(1, gap);
    const double max_est = moe_env_f64("LLAMA_LAZY_MOE_OLECAR_MAX_EST_COST", 2.0);
    record.estimated_cost = std::min(max_est, cost / (delay * denom));

    const double default_eta = std::min(1.0,
            std::sqrt((double) MOE_OLECAR_POLICY_COUNT * std::log((double) MOE_OLECAR_POLICY_COUNT) / 2.0));
    const double eta = moe_env_f64("LLAMA_LAZY_MOE_OLECAR_ETA", default_eta);
    const double actions = std::max(1.0,
            moe_env_f64("LLAMA_LAZY_MOE_OLECAR_ACTIONS", (double) MOE_OLECAR_POLICY_COUNT));
    weight *= std::exp(-(eta * record.estimated_cost) / actions);

    const double min_weight = moe_env_f64("LLAMA_LAZY_MOE_OLECAR_MIN_WEIGHT", 1.0e-6);
    double sum = 0.0;
    for (double & w : weights) {
        if (!std::isfinite(w) || w < min_weight) {
            w = min_weight;
        }
        sum += w;
    }
    if (sum > 0.0) {
        for (double & w : weights) {
            w /= sum;
        }
    } else {
        const double uniform = 1.0 / (double) MOE_OLECAR_POLICY_COUNT;
        for (double & w : weights) {
            w = uniform;
        }
    }

    const double gamma = std::max(0.0, std::min(1.0,
            moe_env_f64("LLAMA_LAZY_MOE_OLECAR_MIX_GAMMA", 0.01)));
    const double uniform = 1.0 / (double) MOE_OLECAR_POLICY_COUNT;
    if (gamma > 0.0) {
        for (double & w : weights) {
            w = (1.0 - gamma) * w + gamma * uniform;
        }
    }

    ctx.olecar_updates[(size_t) record.bucket_id]++;
    record.update_count = ctx.olecar_updates[(size_t) record.bucket_id];
    record.weight_after = weight;
}

static void moe_olecar_trace_write_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        const moe_olecar_record &  record,
        uint64_t                   gap,
        double                     cost,
        const char *               reason) {
    if (ctx.olecar_trace == nullptr) {
        return;
    }
    std::fprintf(ctx.olecar_trace,
            "%s\t%s\t%llu\t%llu\t%llu\t%llu\t%d\t%d\t%s\t%d\t%d\t%d\t%llu\t%.6f\t%.6f\t%llu\t%llu\t%.3f\t%.3f\t%.6f\t%.6f\t%.6f\t%.6f\t%llu\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%s\n",
            event,
            record.kind,
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            (unsigned long long) record.id,
            (unsigned long long) record.decision_id,
            record.bucket_id,
            record.policy_id,
            record.policy,
            record.layer,
            record.expert,
            record.selected ? 1 : 0,
            (unsigned long long) gap,
            cost,
            record.estimated_cost,
            (unsigned long long) record.decision_token,
            (unsigned long long) record.deadline,
            (double) ctx.resident_bytes / 1048576.0,
            (double) ctx.params.budget_bytes / 1048576.0,
            record.weight_before,
            record.weight_after,
            record.policy_prob,
            record.support_prob,
            (unsigned long long) record.update_count,
            record.final_score,
            record.lru_score,
            record.recency_score,
            record.cache_score,
            record.bad_reload_score,
            record.next_use_score,
            record.layer_score,
            record.reuse_score,
            record.cct_score,
            reason);
}

static void moe_olecar_compact_watchers_locked(
        llama_moe_buffer_context & ctx,
        uint64_t                   key) {
    auto it = ctx.olecar_watchers.find(key);
    if (it == ctx.olecar_watchers.end()) {
        return;
    }
    std::vector<uint64_t> & ids = it->second;
    ids.erase(std::remove_if(ids.begin(), ids.end(), [&](uint64_t id) {
        return ctx.olecar_records.find(id) == ctx.olecar_records.end();
    }), ids.end());
    if (ids.empty()) {
        ctx.olecar_watchers.erase(it);
    }
}

static void moe_olecar_begin_locked(
        llama_moe_buffer_context & ctx,
        moe_olecar_record          record) {
    if (ctx.olecar_trace == nullptr || record.layer < 0 || record.expert < 0) {
        return;
    }
    const uint64_t horizon = (uint64_t) std::max(1,
            moe_env_i32("LLAMA_LAZY_MOE_OLECAR_TOKENS", 16));
    record.id = ctx.next_olecar_id++;
    record.decision_token = ctx.profile_token_epoch;
    record.deadline = ctx.profile_token_epoch > UINT64_MAX - horizon ?
        UINT64_MAX : ctx.profile_token_epoch + horizon;
    record.key = moe_group_key(record.layer, record.expert);
    ctx.olecar_records.emplace(record.id, record);
    if (ctx.olecar_watchers[record.key].size() >= 128) {
        moe_olecar_compact_watchers_locked(ctx, record.key);
    }
    ctx.olecar_watchers[record.key].push_back(record.id);
    ctx.olecar_deadlines.push_back({record.deadline, record.id});
    moe_olecar_trace_write_locked(ctx, "decision", record, UINT64_MAX, 0.0, "recommend");
}

static void moe_olecar_note_touch_locked(
        llama_moe_buffer_context & ctx,
        uint64_t                   key) {
    auto watcher_it = ctx.olecar_watchers.find(key);
    if (watcher_it == ctx.olecar_watchers.end()) {
        return;
    }
    std::vector<uint64_t> ids = watcher_it->second;
    for (uint64_t id : ids) {
        auto it = ctx.olecar_records.find(id);
        if (it == ctx.olecar_records.end()) {
            continue;
        }
        const moe_olecar_record record = it->second;
        if (ctx.profile_token_epoch <= record.decision_token) {
            continue;
        }
        const uint64_t gap = ctx.profile_token_epoch - record.decision_token;
        moe_olecar_record resolved = record;
        const double cost = moe_olecar_delay_cost(gap);
        if (resolved.exp4_update) {
            moe_olecar_shadow_update_locked(ctx, resolved, gap, cost);
        }
        moe_olecar_trace_write_locked(ctx, "resolve", resolved, gap, cost, "touch");
        ctx.olecar_records.erase(it);
    }
    moe_olecar_compact_watchers_locked(ctx, key);
}

static void moe_olecar_expire_locked(llama_moe_buffer_context & ctx) {
    while (!ctx.olecar_deadlines.empty() &&
            ctx.olecar_deadlines.front().deadline < ctx.profile_token_epoch) {
        const moe_olecar_deadline deadline = ctx.olecar_deadlines.front();
        ctx.olecar_deadlines.pop_front();
        auto it = ctx.olecar_records.find(deadline.id);
        if (it == ctx.olecar_records.end()) {
            continue;
        }
        moe_olecar_record record = it->second;
        record.weight_after = record.weight_before;
        moe_olecar_trace_write_locked(ctx, "expire", record, UINT64_MAX, 0.0, "timeout");
        ctx.olecar_records.erase(it);
    }
}

static void moe_olecar_finish_locked(
        llama_moe_buffer_context & ctx,
        const char *               reason) {
    for (const auto & kv : ctx.olecar_records) {
        moe_olecar_record record = kv.second;
        record.weight_after = record.weight_before;
        moe_olecar_trace_write_locked(ctx, "unfinished", record, UINT64_MAX, 0.0, reason);
    }
    ctx.olecar_records.clear();
    ctx.olecar_watchers.clear();
    ctx.olecar_deadlines.clear();
}

static void moe_group_note_evicted_locked(
        llama_moe_buffer_context & ctx,
        moe_group_state &          g) {
    moe_admission_outcome_note_evicted_locked(ctx, g);
    g.last_evicted_token_epoch = ctx.profile_token_epoch;
    g.last_evicted_valid = true;
    if (!moe_ghost_feedback_enabled()) {
        g.ghost_pending = false;
        return;
    }
    g.ghost_generation++;
    g.ghost_pending = true;
    const uint64_t horizon = moe_ghost_feedback_horizon();
    const uint64_t deadline = ctx.profile_token_epoch > UINT64_MAX - horizon ?
        UINT64_MAX : ctx.profile_token_epoch + horizon;
    ctx.ghost_records.push_back({deadline, moe_group_key(g.layer, g.expert), g.ghost_generation});
    ctx.ghost_evictions.fetch_add(1, std::memory_order_relaxed);
}

static void moe_group_lru_unlink_locked(llama_moe_buffer_context & ctx, moe_group_state & g) {
    if (!g.group_lru_linked) {
        return;
    }
    ctx.group_lru.erase(g.group_lru_pos);
    g.group_lru_linked = false;
    ctx.group_lru_unlinks.fetch_add(1, std::memory_order_relaxed);
}

static void moe_group_lru_touch_locked(llama_moe_buffer_context & ctx, int layer, int expert) {
    if (layer < 0 || expert < 0) {
        return;
    }
    moe_group_state & g = moe_group_get(ctx, layer, expert);
    const uint64_t key = moe_group_key(layer, expert);
    if (g.group_lru_linked) {
        ctx.group_lru.splice(ctx.group_lru.begin(), ctx.group_lru, g.group_lru_pos);
        g.group_lru_pos = ctx.group_lru.begin();
        return;
    }
    ctx.group_lru.push_front(key);
    g.group_lru_pos = ctx.group_lru.begin();
    g.group_lru_linked = true;
}

static void moe_group_lru_rebuild_locked(llama_moe_buffer_context & ctx) {
    ctx.group_lru.clear();
    for (auto & kv : ctx.groups) {
        kv.second.group_lru_linked = false;
    }
    std::unordered_set<uint64_t> seen;
    for (auto it = ctx.lru.begin(); it != ctx.lru.end(); ++it) {
        moe_managed * m = it->first;
        const int e = it->second;
        if (m == nullptr || m->layer < 0 || e < 0 || e >= m->n_expert ||
                m->resident[e] != ST_RESIDENT) {
            continue;
        }
        const uint64_t key = moe_group_key(m->layer, e);
        if (!seen.insert(key).second) {
            continue;
        }
        ctx.group_lru.push_back(key);
        moe_group_state & g = moe_group_get(ctx, m->layer, e);
        g.group_lru_pos = std::prev(ctx.group_lru.end());
        g.group_lru_linked = true;
    }
    ctx.group_lru_rebuilds.fetch_add(1, std::memory_order_relaxed);
}

static double moe_group_ghost_keep_score_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g) {
    if (!moe_ghost_feedback_enabled() || g.ghost_outcomes == 0 || g.ghost_reload_risk_ema <= 0.0) {
        return 0.0;
    }
    static const double min_samples = std::max(1.0,
            moe_env_f64("LLAMA_LAZY_MOE_GHOST_MIN_SAMPLES", 4.0));
    const double confidence = std::min(1.0, (double) g.ghost_outcomes / min_samples);
    static const double half_life = std::max(1.0,
            moe_env_f64("LLAMA_LAZY_MOE_GHOST_HALF_LIFE_TOKENS", 128.0));
    static const double weight =
        moe_env_f64("LLAMA_LAZY_MOE_GHOST_KEEP_WEIGHT", 5.0e8);
    const uint64_t age =
        ctx.profile_token_epoch >= g.ghost_last_outcome_token ?
        ctx.profile_token_epoch - g.ghost_last_outcome_token : 0;
    const double risk = g.ghost_reload_risk_ema * half_life / (half_life + (double) age);
    return risk * confidence * weight;
}

static double moe_group_bad_reload_effective_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        uint64_t *                       out_age) {
    if (!moe_bad_reload_evict_enabled() || g.bad_reload_last_token == 0 ||
            ctx.profile_token_epoch < g.bad_reload_last_token) {
        if (out_age != nullptr) {
            *out_age = UINT64_MAX;
        }
        return 0.0;
    }
    const uint64_t age = ctx.profile_token_epoch - g.bad_reload_last_token;
    if (out_age != nullptr) {
        *out_age = age;
    }
    double scale = 0.0;
    if (age <= 4) {
        scale = 1.0;
    } else if (age <= 8) {
        scale = 0.75;
    } else if (age <= 16) {
        scale = 0.50;
    } else if (age <= 32) {
        scale = 0.25;
    }
    return g.bad_reload_score * scale;
}

static void moe_group_note_bad_reload_locked(
        llama_moe_buffer_context & ctx,
        moe_group_state &          g,
        uint64_t                   reload_gap) {
    if (!moe_bad_reload_evict_enabled()) {
        return;
    }
    double add = 0.0;
    if (reload_gap <= 1) {
        add = moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_ADD_1", 4.0);
        g.bad_reload_1++;
        ctx.bad_reload_1.fetch_add(1, std::memory_order_relaxed);
    } else if (reload_gap <= 4) {
        add = moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_ADD_4", 2.0);
        g.bad_reload_4++;
        ctx.bad_reload_4.fetch_add(1, std::memory_order_relaxed);
    } else if (reload_gap <= 16) {
        add = moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_ADD_16", 1.0);
        g.bad_reload_16++;
        ctx.bad_reload_16.fetch_add(1, std::memory_order_relaxed);
    }
    if (add <= 0.0) {
        return;
    }
    uint64_t age = UINT64_MAX;
    const double old_effective = moe_group_bad_reload_effective_locked(ctx, g, &age);
    const double cap = moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_SCORE_CAP", 24.0);
    g.bad_reload_score = std::min(cap, old_effective + add);
    g.bad_reload_last_token = ctx.profile_token_epoch;
}

static void moe_group_update_reuse_on_touch_locked(llama_moe_buffer_context & ctx, moe_group_state & g) {
    moe_olecar_note_touch_locked(ctx, moe_group_key(g.layer, g.expert));
    const uint64_t token = ctx.profile_token_epoch;
    if (g.last_evicted_valid && token >= g.last_evicted_token_epoch) {
        const uint64_t reload_gap = token - g.last_evicted_token_epoch;
        moe_group_ghost_note_reload_locked(ctx, g, reload_gap);
        moe_group_note_bad_reload_locked(ctx, g, reload_gap);
        moe_evict_trace_write_locked(ctx, "reload", g.layer, g.expert, "reload_after_evict",
                reload_gap, 0, 0.0);
        const uint64_t layer_reload_guard = (uint64_t) std::max(0,
                moe_env_i32("LLAMA_LAZY_MOE_LAYER_RELOAD_GUARD_TOKENS", 4));
        if (g.last_evicted_reason == MOE_EVICT_REASON_LAYER_WINDOW &&
                layer_reload_guard > 0 && reload_gap <= layer_reload_guard) {
            g.layer_window_bad_reload++;
            g.layer_window_last_bad_reload_token = token;
            ctx.layer_reload_bad.fetch_add(1, std::memory_order_relaxed);
        }
        if (moe_reuse_evict_enabled()) {
            if (reload_gap <= 1) {
                ctx.reuse_reload_1.fetch_add(1, std::memory_order_relaxed);
            }
            if (reload_gap <= 4) {
                ctx.reuse_reload_4.fetch_add(1, std::memory_order_relaxed);
            }
            if (reload_gap <= 16) {
                ctx.reuse_reload_16.fetch_add(1, std::memory_order_relaxed);
            }
        }
        g.last_evicted_valid = false;
        g.last_evicted_reason = MOE_EVICT_REASON_UNKNOWN;
    }
    if (token == 0) {
        return;
    }
    if (!moe_reuse_evict_enabled()) {
        return;
    }
    if (g.last_used_token_epoch == 0 || token <= g.last_used_token_epoch) {
        return;
    }

    const uint64_t gap = token - g.last_used_token_epoch;
    const double alpha = moe_env_f64("LLAMA_LAZY_MOE_REUSE_ALPHA", 0.15);
    const double hit = gap <= 1 ? 1.0 : (gap <= 4 ? 0.75 : (gap <= 16 ? 0.35 : 0.05));
    g.reuse_ema = (1.0 - alpha) * g.reuse_ema + alpha * hit;
    g.inter_token_gap_ema = (1.0 - alpha) * g.inter_token_gap_ema + alpha * (double) gap;
    g.reuse_observed++;
    if (gap <= 1) {
        g.reuse_within_1++;
    }
    if (gap <= 4) {
        g.reuse_within_4++;
    }
    if (gap <= 16) {
        g.reuse_within_16++;
    }
}

static double moe_group_short_reuse_keep_score_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        int                              current_layer) {
    if (!moe_reuse_evict_enabled()) {
        return 0.0;
    }

    const double obs = (double) g.reuse_observed;
    const double p1 = ((double) g.reuse_within_1 + 0.25) / (obs + 1.0);
    const double p4 = ((double) g.reuse_within_4 + 0.50) / (obs + 1.0);
    const double p16 = ((double) g.reuse_within_16 + 0.75) / (obs + 1.0);
    const double seq_rate = moe_group_seq_rate(ctx, g);
    const double rank0_rate = g.seq_access > 0 ? (double) g.seq_rank0_access / (double) g.seq_access : 0.0;
    const uint64_t age = g.seq_last_token_epoch == 0 || ctx.profile_token_epoch < g.seq_last_token_epoch ?
        UINT64_MAX : ctx.profile_token_epoch - g.seq_last_token_epoch;
    const double recency = age == UINT64_MAX ? 0.0 : 1.0 / (1.0 + (double) age);
    const double stale_penalty = age == UINT64_MAX ? 1.0 :
        std::max(0.0, (double) age - 16.0) / 64.0;
    const double gap = std::max(1.0, g.inter_token_gap_ema);
    const double gap_score = 1.0 / gap;
    const int layer_dist = moe_forward_layer_distance(ctx, g.layer);
    const double near_layer = current_layer >= 0 && g.layer >= 0 ?
        1.0 / (1.0 + (double) std::max(0, layer_dist)) : 0.0;
    const double unused_rate = g.seq_prefetch_queued > 0 ?
        (double) g.seq_prefetch_unused / (double) g.seq_prefetch_queued : 0.0;

    const double a = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_A", 45.0);
    const double b = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_B", 110.0);
    const double c = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_C", 85.0);
    const double d = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_D", 9000.0);
    const double e = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_E", 35.0);
    const double f = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_F", 70.0);
    const double h = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_H", 12000.0);
    const double i = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_I", 45.0);
    const double j = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_J", 40.0);

    return a * p1 +
           b * p4 +
           c * p16 +
           d * seq_rate +
           e * rank0_rate +
           f * recency +
           40.0 * gap_score +
           30.0 * near_layer +
           h * moe_eamc_prior(ctx, g.layer, g.expert) -
           i * unused_rate -
           j * stale_penalty;
}

static bool moe_group_predicted_soon_for_evict(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        bool                             layer_window,
        uint64_t                         next_use_dist) {
    if (!moe_reuse_evict_enabled()) {
        return false;
    }
    const uint64_t epoch_guard = (uint64_t) moe_env_i32("LLAMA_LAZY_MOE_REUSE_PREDICT_EPOCHS",
            std::max(4, ctx.params.active_window * 4));
    if (next_use_dist != UINT64_MAX && next_use_dist <= epoch_guard) {
        return true;
    }
    if (layer_window && g.next_use_epoch != UINT64_MAX) {
        return true;
    }
    return false;
}

static bool moe_eam_replace_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_EAM_REPLACE", 1);
}

static double moe_group_eam_replace_pred_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g) {
    if (g.layer < 0 || g.expert < 0) {
        return 0.0;
    }

    uint64_t layer_access = 0;
    uint64_t layer_hints = 0;
    uint64_t layer_predicted = 0;
    for (const auto & kv : ctx.groups) {
        const moe_group_state & other = kv.second;
        if (other.layer != g.layer) {
            continue;
        }
        layer_access += other.seq_access;
        layer_hints += other.seq_future_hints;
        layer_predicted += other.seq_predicted;
    }

    const double seq_prob = layer_access > 0 ?
        (double) g.seq_access / (double) layer_access : 0.0;
    const double hint_prob = layer_hints > 0 ?
        (double) g.seq_future_hints / (double) layer_hints : 0.0;
    const double pred_prob = layer_predicted > 0 ?
        (double) g.seq_predicted / (double) layer_predicted : 0.0;
    const double eamc_prob = moe_eamc_prior(ctx, g.layer, g.expert);

    const double w_eamc = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_EAMC_W", 1.0);
    const double w_seq  = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_SEQ_W", 1.0);
    const double w_hint = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_HINT_W", 0.5);
    const double w_pred = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_PRED_W", 0.5);
    return w_eamc * eamc_prob + w_seq * seq_prob + w_hint * hint_prob + w_pred * pred_prob;
}

static double moe_layer_eam_replace_mass_locked(
        const llama_moe_buffer_context & ctx,
        int                              layer) {
    if (layer < 0) {
        return 0.0;
    }
    double mass = 0.0;
    for (const auto & kv : ctx.groups) {
        const moe_group_state & g = kv.second;
        if (g.layer == layer) {
            mass += moe_group_eam_replace_pred_locked(ctx, g);
        }
    }
    return mass;
}

static double moe_group_eam_replace_keep_score_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        double                           layer_mass) {
    const int n_layers = std::max(1, moe_max_layer_index(ctx) + 1);
    const double min_layer_weight = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_LAYER_MIN", 0.40);
    const double raw_layer_weight = g.layer >= 0 ?
        std::max(0.0, 1.0 - (double) g.layer / (double) n_layers) : 0.0;
    const double layer_weight = g.layer >= 0 ? std::max(min_layer_weight, raw_layer_weight) : 0.0;
    const double smooth = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_C", 1.0e-4);
    const double eps = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_EPS", 1.0e-9);
    const double p = moe_group_eam_replace_pred_locked(ctx, g);
    return (p + smooth) * layer_weight / std::max(layer_mass, eps);
}

struct moe_eam_replace_stats {
    int n_layers = 0;
    std::vector<uint64_t> layer_access;
    std::vector<uint64_t> layer_hints;
    std::vector<uint64_t> layer_predicted;
    std::vector<double> layer_mass;
};

static moe_eam_replace_stats moe_eam_replace_build_stats_locked(
        const llama_moe_buffer_context & ctx) {
    moe_eam_replace_stats st;
    st.n_layers = std::max(1, moe_max_layer_index(ctx) + 1);
    st.layer_access.assign((size_t) st.n_layers, 0);
    st.layer_hints.assign((size_t) st.n_layers, 0);
    st.layer_predicted.assign((size_t) st.n_layers, 0);
    st.layer_mass.assign((size_t) st.n_layers, 0.0);

    for (const auto & kv : ctx.groups) {
        const moe_group_state & g = kv.second;
        if (g.layer < 0 || g.layer >= st.n_layers) {
            continue;
        }
        const size_t l = (size_t) g.layer;
        st.layer_access[l] += g.seq_access;
        st.layer_hints[l] += g.seq_future_hints;
        st.layer_predicted[l] += g.seq_predicted;
    }

    const double w_eamc = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_EAMC_W", 1.0);
    const double w_seq  = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_SEQ_W", 1.0);
    const double w_hint = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_HINT_W", 0.5);
    const double w_pred = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_PRED_W", 0.5);
    for (const auto & kv : ctx.groups) {
        const moe_group_state & g = kv.second;
        if (g.layer < 0 || g.layer >= st.n_layers || g.expert < 0) {
            continue;
        }
        const size_t l = (size_t) g.layer;
        const double seq_prob = st.layer_access[l] > 0 ?
            (double) g.seq_access / (double) st.layer_access[l] : 0.0;
        const double hint_prob = st.layer_hints[l] > 0 ?
            (double) g.seq_future_hints / (double) st.layer_hints[l] : 0.0;
        const double pred_prob = st.layer_predicted[l] > 0 ?
            (double) g.seq_predicted / (double) st.layer_predicted[l] : 0.0;
        const double eamc_prob = moe_eamc_prior(ctx, g.layer, g.expert);
        st.layer_mass[l] += w_eamc * eamc_prob + w_seq * seq_prob + w_hint * hint_prob + w_pred * pred_prob;
    }

    return st;
}

static double moe_group_eam_replace_pred_from_stats_locked(
        const llama_moe_buffer_context & ctx,
        const moe_eam_replace_stats &    st,
        const moe_group_state &          g) {
    if (g.layer < 0 || g.layer >= st.n_layers || g.expert < 0) {
        return 0.0;
    }
    const size_t l = (size_t) g.layer;
    const double seq_prob = st.layer_access[l] > 0 ?
        (double) g.seq_access / (double) st.layer_access[l] : 0.0;
    const double hint_prob = st.layer_hints[l] > 0 ?
        (double) g.seq_future_hints / (double) st.layer_hints[l] : 0.0;
    const double pred_prob = st.layer_predicted[l] > 0 ?
        (double) g.seq_predicted / (double) st.layer_predicted[l] : 0.0;
    const double eamc_prob = moe_eamc_prior(ctx, g.layer, g.expert);

    const double w_eamc = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_EAMC_W", 1.0);
    const double w_seq  = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_SEQ_W", 1.0);
    const double w_hint = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_HINT_W", 0.5);
    const double w_pred = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_PRED_W", 0.5);
    return w_eamc * eamc_prob + w_seq * seq_prob + w_hint * hint_prob + w_pred * pred_prob;
}

static double moe_group_eam_replace_keep_from_stats_locked(
        const llama_moe_buffer_context & ctx,
        const moe_eam_replace_stats &    st,
        const moe_group_state &          g) {
    if (g.layer < 0 || g.layer >= st.n_layers) {
        return 0.0;
    }
    const double min_layer_weight = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_LAYER_MIN", 0.40);
    const double raw_layer_weight = std::max(0.0, 1.0 - (double) g.layer / (double) st.n_layers);
    const double layer_weight = std::max(min_layer_weight, raw_layer_weight);
    const double smooth = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_C", 1.0e-4);
    const double eps = moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_EPS", 1.0e-9);
    const double p = moe_group_eam_replace_pred_from_stats_locked(ctx, st, g);
    return (p + smooth) * layer_weight / std::max(st.layer_mass[(size_t) g.layer], eps);
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

enum class moe_evict_protect_reason : uint8_t {
    NONE = 0,
    PINNED,
    ACTIVE,
    CURRENT_LAYER,
    DEMAND,
    INFLIGHT,
    LAYER_WINDOW,
    RECENT,
    BAD_RELOAD,
    NEXT_TOKEN,
    CCT,
    PREDICTED_SOON,
    REUSE,
};

enum class moe_evict_protect_class : uint8_t {
    NONE = 0,
    TEMPORARY,
    ABSOLUTE,
};

struct moe_evict_protect_options {
    bool speculative_unused = false;
    bool current_layer = false;
    bool layer_window = false;
    bool protect_layer_window = false;
    bool protect_recent = true;
    bool protect_high_recent = false;
    bool protect_cct = true;
    bool protect_predicted_soon = true;
    bool protect_reuse = true;
    bool protect_next_token = true;
    int aggressive_lazy_mode = 0;
    int recent_tokens = 3;
    double recent_score = 0.0;
    uint64_t bad_reload_guard_tokens = 16;
    double bad_reload_score = 6.0;
    bool cct_evict_active = false;
    bool need_cct_score = false;
    int cct_protect_conf = 3;
    int cct_protect_distance = 1;
    bool reuse_evict_active = false;
    double reuse_protect_score = 260.0;
    int reuse_protect_tokens = 4;
    double high_keep_score = 125.0;
    double high_keep_rate = 0.0080;
    int early_high_layers = 4;
    double early_high_keep_score = 95.0;
    double relaxed_next_use_weight = 0.0;
    double relaxed_layer_distance_weight = 2.5e8;
};

struct moe_evict_protect_result {
    moe_evict_protect_reason reason = moe_evict_protect_reason::NONE;
    moe_evict_protect_class protect_class = moe_evict_protect_class::NONE;
    bool recent = false;
    bool high_sequence = false;
    bool early_high = false;
    bool predicted_soon = false;
    bool reuse_protected = false;
    bool cct_protected = false;
    bool bad_reload_protected = false;
    bool next_token_protected = false;
    bool future_ranked = false;
    uint64_t next_use_dist = UINT64_MAX;
    int layer_dist = INT_MAX;
    uint64_t recent_age = UINT64_MAX;
    uint64_t recent_exec_age = UINT64_MAX;
    uint64_t bad_reload_age = UINT64_MAX;
    double bad_reload_effective = 0.0;
    double reuse_keep = 0.0;
    double ghost_keep = 0.0;
    double temporary_keep_score = 0.0;
    uint8_t cct_conf = 0;
};

static moe_evict_protect_result moe_group_evict_protect_locked(
        llama_moe_buffer_context &       ctx,
        moe_group_state &                g,
        int                              layer,
        int                              expert,
        double                           cache_score,
        const moe_evict_protect_options & opt) {
    moe_evict_protect_result r;
    auto absolute = [&](moe_evict_protect_reason reason) {
        r.reason = reason;
        r.protect_class = moe_evict_protect_class::ABSOLUTE;
        return r;
    };
    auto temporary = [&](moe_evict_protect_reason reason, double keep_score) {
        r.reason = reason;
        r.protect_class = moe_evict_protect_class::TEMPORARY;
        r.temporary_keep_score = keep_score;
        r.future_ranked = r.next_use_dist != UINT64_MAX && opt.relaxed_next_use_weight > 0.0;
        return r;
    };
    if (g.pinned) {
        return absolute(moe_evict_protect_reason::PINNED);
    }
    if (moe_group_is_active(ctx, g)) {
        return absolute(moe_evict_protect_reason::ACTIVE);
    }
    if (opt.current_layer) {
        return absolute(moe_evict_protect_reason::CURRENT_LAYER);
    }
    if (g.demand_async_pending || g.demand_admission_bypass) {
        return absolute(moe_evict_protect_reason::DEMAND);
    }
    if (moe_group_has_inflight_or_queued(ctx, layer, expert)) {
        return absolute(moe_evict_protect_reason::INFLIGHT);
    }

    if (opt.aggressive_lazy_mode >= 2) {
        if (g.next_use_epoch != UINT64_MAX && g.next_use_epoch > ctx.exec_epoch) {
            r.next_use_dist = g.next_use_epoch - ctx.exec_epoch;
        }
        if (g.last_used_token_epoch != 0 && ctx.profile_token_epoch >= g.last_used_token_epoch) {
            r.recent_age = ctx.profile_token_epoch - g.last_used_token_epoch;
        }
        if (g.last_used_epoch != 0 && ctx.exec_epoch >= g.last_used_epoch) {
            r.recent_exec_age = ctx.exec_epoch - g.last_used_epoch;
        }
        r.recent = moe_group_used_within_tokens(ctx, g, opt.recent_tokens) ||
            moe_group_is_recently_used(ctx, g) || moe_group_in_cooldown(ctx, g);
        const double cache_keep = std::max(0.0, cache_score) * 1.0e3;
        if (opt.protect_layer_window && opt.layer_window) {
            return temporary(moe_evict_protect_reason::LAYER_WINDOW, 2.5e8 + cache_keep);
        }
        if (!opt.speculative_unused && opt.protect_recent && r.recent &&
                (!opt.protect_high_recent || cache_score >= opt.recent_score)) {
            const double recent_strength =
                r.recent_age == UINT64_MAX || opt.recent_tokens <= 0 ? 0.0 :
                (double) std::max<int64_t>(0,
                        (int64_t) opt.recent_tokens + 1 - (int64_t) r.recent_age);
            const int exec_guard = std::max(2, ctx.params.active_window);
            const double recent_exec_strength =
                r.recent_exec_age == UINT64_MAX ? 0.0 :
                (double) std::max<int64_t>(0,
                        (int64_t) exec_guard + 1 - (int64_t) r.recent_exec_age);
            return temporary(moe_evict_protect_reason::RECENT,
                    recent_strength * 1.0e9 + recent_exec_strength * 1.0e7 + cache_keep);
        }
        r.bad_reload_effective = moe_group_bad_reload_effective_locked(ctx, g, &r.bad_reload_age);
        r.bad_reload_protected = !opt.speculative_unused && opt.bad_reload_guard_tokens > 0 &&
            r.bad_reload_age != UINT64_MAX &&
            r.bad_reload_age <= opt.bad_reload_guard_tokens &&
            r.bad_reload_effective >= opt.bad_reload_score;
        if (r.bad_reload_protected) {
            return temporary(moe_evict_protect_reason::BAD_RELOAD,
                    r.bad_reload_effective * 1.0e8 + cache_keep);
        }
        r.next_token_protected = !opt.speculative_unused && opt.protect_next_token &&
            moe_next_token_group_protected_locked(ctx, g);
        if (r.next_token_protected) {
            return temporary(moe_evict_protect_reason::NEXT_TOKEN,
                    (double) (1 + g.next_token_conf) * 1.0e9 + cache_keep);
        }
        r.predicted_soon = !opt.speculative_unused &&
            moe_group_predicted_soon_for_evict(ctx, g, opt.layer_window, r.next_use_dist);
        if (opt.protect_predicted_soon && r.predicted_soon) {
            return temporary(moe_evict_protect_reason::PREDICTED_SOON, 5.0e8 + cache_keep);
        }
        r.layer_dist = moe_forward_layer_distance(ctx, layer);
        if (opt.cct_evict_active || opt.need_cct_score) {
            r.cct_conf = moe_cct_predict_conf_locked(ctx, layer, expert);
            g.cct_conf = r.cct_conf;
        }
        r.cct_protected = !opt.speculative_unused && opt.protect_cct &&
            opt.cct_evict_active &&
            (int) r.cct_conf >= opt.cct_protect_conf &&
            r.layer_dist <= std::max(0, opt.cct_protect_distance);
        if (r.cct_protected) {
            return temporary(moe_evict_protect_reason::CCT,
                    (double) (1 + r.cct_conf) * 5.0e8 + cache_keep);
        }
        if (opt.reuse_evict_active) {
            r.reuse_keep = moe_group_short_reuse_keep_score_locked(ctx, g, ctx.evict_target_layer);
        }
        r.reuse_protected = !opt.speculative_unused && opt.protect_reuse &&
            opt.reuse_evict_active &&
            r.reuse_keep >= opt.reuse_protect_score &&
            moe_group_used_within_tokens(ctx, g, opt.reuse_protect_tokens);
        if (r.reuse_protected) {
            return temporary(moe_evict_protect_reason::REUSE,
                    r.reuse_keep * 1.0e5 + cache_keep);
        }
        r.high_sequence = cache_score >= opt.high_keep_score ||
            moe_group_seq_rate(ctx, g) >= opt.high_keep_rate;
        r.early_high = g.layer >= 0 && g.layer <= opt.early_high_layers &&
            (cache_score >= opt.early_high_keep_score ||
                g.seq_rank0_access >= std::max<uint64_t>(4, g.seq_access / 2));
        return r;
    }

    r.layer_dist = moe_forward_layer_distance(ctx, layer);
    if (g.next_use_epoch != UINT64_MAX && g.next_use_epoch > ctx.exec_epoch) {
        r.next_use_dist = g.next_use_epoch - ctx.exec_epoch;
    }
    if (g.last_used_token_epoch != 0 && ctx.profile_token_epoch >= g.last_used_token_epoch) {
        r.recent_age = ctx.profile_token_epoch - g.last_used_token_epoch;
    }
    if (g.last_used_epoch != 0 && ctx.exec_epoch >= g.last_used_epoch) {
        r.recent_exec_age = ctx.exec_epoch - g.last_used_epoch;
    }
    r.recent = moe_group_used_within_tokens(ctx, g, opt.recent_tokens) ||
        moe_group_is_recently_used(ctx, g) || moe_group_in_cooldown(ctx, g);
    r.high_sequence = cache_score >= opt.high_keep_score ||
        moe_group_seq_rate(ctx, g) >= opt.high_keep_rate;
    r.early_high = g.layer >= 0 && g.layer <= opt.early_high_layers &&
        (cache_score >= opt.early_high_keep_score ||
            g.seq_rank0_access >= std::max<uint64_t>(4, g.seq_access / 2));

    if (opt.reuse_evict_active) {
        r.reuse_keep = moe_group_short_reuse_keep_score_locked(ctx, g, ctx.evict_target_layer);
    }
    r.bad_reload_effective = moe_group_bad_reload_effective_locked(ctx, g, &r.bad_reload_age);
    r.bad_reload_protected = !opt.speculative_unused && opt.bad_reload_guard_tokens > 0 &&
        r.bad_reload_age != UINT64_MAX &&
        r.bad_reload_age <= opt.bad_reload_guard_tokens &&
        r.bad_reload_effective >= opt.bad_reload_score;
    r.next_token_protected = !opt.speculative_unused && opt.protect_next_token &&
        moe_next_token_group_protected_locked(ctx, g);
    r.predicted_soon = !opt.speculative_unused &&
        moe_group_predicted_soon_for_evict(ctx, g, opt.layer_window, r.next_use_dist);
    if (opt.cct_evict_active || opt.need_cct_score) {
        r.cct_conf = moe_cct_predict_conf_locked(ctx, layer, expert);
        g.cct_conf = r.cct_conf;
    }
    r.cct_protected = !opt.speculative_unused && opt.protect_cct &&
        opt.cct_evict_active &&
        (int) r.cct_conf >= opt.cct_protect_conf &&
        r.layer_dist <= std::max(0, opt.cct_protect_distance);
    r.reuse_protected = !opt.speculative_unused && opt.protect_reuse &&
        opt.reuse_evict_active &&
        r.reuse_keep >= opt.reuse_protect_score &&
        moe_group_used_within_tokens(ctx, g, opt.reuse_protect_tokens);

    if (opt.protect_layer_window && opt.layer_window) {
        r.reason = moe_evict_protect_reason::LAYER_WINDOW;
    } else if (!opt.speculative_unused && opt.protect_recent && r.recent &&
            (!opt.protect_high_recent || cache_score >= opt.recent_score)) {
        r.reason = moe_evict_protect_reason::RECENT;
    } else if (r.bad_reload_protected) {
        r.reason = moe_evict_protect_reason::BAD_RELOAD;
    } else if (r.next_token_protected) {
        r.reason = moe_evict_protect_reason::NEXT_TOKEN;
    } else if (r.cct_protected) {
        r.reason = moe_evict_protect_reason::CCT;
    } else if (opt.protect_predicted_soon && r.predicted_soon) {
        r.reason = moe_evict_protect_reason::PREDICTED_SOON;
    } else if (r.reuse_protected) {
        r.reason = moe_evict_protect_reason::REUSE;
    }

    switch (r.reason) {
        case moe_evict_protect_reason::PINNED:
        case moe_evict_protect_reason::ACTIVE:
        case moe_evict_protect_reason::CURRENT_LAYER:
        case moe_evict_protect_reason::DEMAND:
        case moe_evict_protect_reason::INFLIGHT:
            r.protect_class = moe_evict_protect_class::ABSOLUTE;
            break;
        case moe_evict_protect_reason::NONE:
            break;
        default:
            r.protect_class = moe_evict_protect_class::TEMPORARY;
            break;
    }

    if (r.protect_class == moe_evict_protect_class::TEMPORARY) {
        r.ghost_keep = moe_group_ghost_keep_score_locked(ctx, g);
        const double recent_strength =
            r.recent_age == UINT64_MAX || opt.recent_tokens <= 0 ? 0.0 :
            (double) std::max<int64_t>(0,
                    (int64_t) opt.recent_tokens + 1 - (int64_t) r.recent_age);
        const int exec_guard = std::max(2, ctx.params.active_window);
        const double recent_exec_strength =
            r.recent_exec_age == UINT64_MAX ? 0.0 :
            (double) std::max<int64_t>(0,
                    (int64_t) exec_guard + 1 - (int64_t) r.recent_exec_age);
        const double next_use_strength = r.next_use_dist == UINT64_MAX ? 0.0 :
            1.0 / (1.0 + (double) r.next_use_dist);
        const double layer_distance_strength =
            r.layer_dist < 0 || r.layer_dist >= INT_MAX / 2 ? 0.0 :
            1.0 / (1.0 + (double) r.layer_dist);
        r.future_ranked = r.next_use_dist != UINT64_MAX && opt.relaxed_next_use_weight > 0.0;
        r.temporary_keep_score =
            recent_strength * 1.0e9 +
            recent_exec_strength * 1.0e7 +
            next_use_strength * opt.relaxed_next_use_weight +
            layer_distance_strength * opt.relaxed_layer_distance_weight +
            (r.next_token_protected ? (double) (1 + g.next_token_conf) * 1.0e9 : 0.0) +
            (r.cct_protected ? (double) (1 + r.cct_conf) * 5.0e8 : 0.0) +
            (opt.protect_predicted_soon && r.predicted_soon ? (1.0 + next_use_strength) * 5.0e8 : 0.0) +
            (opt.protect_layer_window && opt.layer_window ? 2.5e8 : 0.0) +
            r.bad_reload_effective * 1.0e8 +
            r.reuse_keep * 1.0e5 +
            r.ghost_keep +
            std::max(0.0, cache_score) * 1.0e3;
        if (r.ghost_keep > 0.0) {
            ctx.ghost_score_candidates++;
        }
    }
    return r;
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
                moe_group_in_cooldown(ctx, g) ||
                moe_next_token_group_protected_locked(ctx, g) ||
                moe_group_has_inflight_or_queued(ctx, m->layer, e)) {
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

struct moe_prefetch_admission_compare {
    double candidate_value = 0.0;
    double victim_value = 0.0;
    int victim_layer = -1;
    int victim_expert = -1;
    bool pressure = false;
    bool has_victim = false;
};

static double moe_prefetch_candidate_value_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        double                           prefetch_score,
        size_t                           target_bytes) {
    uint64_t bad_age = UINT64_MAX;
    const double bad = moe_group_bad_reload_effective_locked(ctx, g, &bad_age);
    const double miss_rate = g.seq_access > 0 ?
        (double) (g.seq_cache_misses + g.seq_prefetch_late) / (double) g.seq_access : 0.15;
    const double unused_rate = g.seq_prefetch_queued > 0 ?
        (double) g.seq_prefetch_unused / (double) g.seq_prefetch_queued : 0.0;
    const double hit_rate = g.seq_prefetch_queued > 0 ?
        (double) g.seq_prefetch_hits / (double) g.seq_prefetch_queued : 0.0;
    const double size_mib = (double) target_bytes / 1048576.0;
    return prefetch_score +
           bad * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_CAND_BAD_RELOAD_W", 60.0) +
           miss_rate * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_CAND_MISS_W", 60.0) +
           hit_rate * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_CAND_HIT_W", 35.0) -
           unused_rate * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_CAND_UNUSED_W", 80.0) -
           size_mib * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_CAND_SIZE_W", 1.5);
}

static double moe_prefetch_victim_value_locked(
        const llama_moe_buffer_context & ctx,
        const moe_group_state &          g,
        size_t                           resident_bytes,
        double                           cache_score) {
    uint64_t bad_age = UINT64_MAX;
    const double bad = moe_group_bad_reload_effective_locked(ctx, g, &bad_age);
    const bool recent = moe_group_used_within_tokens(ctx, g,
            moe_env_i32("LLAMA_LAZY_MOE_ADMIT_RECENT_TOKENS", 4)) ||
        moe_group_is_recently_used(ctx, g) || moe_group_in_cooldown(ctx, g);
    const bool high_sequence = moe_group_is_high_sequence(ctx, g, cache_score);
    const bool early_high = moe_group_is_early_high_reuse(g, cache_score);
    uint64_t next_use_dist = UINT64_MAX;
    if (g.next_use_epoch != UINT64_MAX && g.next_use_epoch > ctx.exec_epoch) {
        next_use_dist = g.next_use_epoch - ctx.exec_epoch;
    }
    const double next_use = next_use_dist == UINT64_MAX ? 0.0 :
        1.0 / (1.0 + (double) next_use_dist);
    const double size_mib = (double) resident_bytes / 1048576.0;
    return bad * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_VICTIM_BAD_RELOAD_W", 120.0) +
           (recent ? moe_env_f64("LLAMA_LAZY_MOE_ADMIT_VICTIM_RECENT_W", 180.0) : 0.0) +
           (high_sequence ? moe_env_f64("LLAMA_LAZY_MOE_ADMIT_VICTIM_HIGH_W", 120.0) : 0.0) +
           (early_high ? moe_env_f64("LLAMA_LAZY_MOE_ADMIT_VICTIM_EARLY_W", 80.0) : 0.0) +
           next_use * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_VICTIM_NEXT_USE_W", 200.0) +
           cache_score * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_VICTIM_CACHE_W", 1.0) -
           size_mib * moe_env_f64("LLAMA_LAZY_MOE_ADMIT_VICTIM_SIZE_W", 1.0);
}

static void moe_admission_floor_refresh_locked(llama_moe_buffer_context & ctx) {
    ctx.admission_floor_valid = false;
    ctx.admission_floor_value = 0.0;
    ctx.admission_floor_layer = -1;
    ctx.admission_floor_expert = -1;
    ctx.admission_floor_token = ctx.profile_token_epoch;
    ctx.admission_floor_epoch = ctx.exec_epoch;

    double best_value = 1.0e300;
    for (auto it = ctx.lru.rbegin(); it != ctx.lru.rend(); ++it) {
        moe_managed * m = it->first;
        const int e = it->second;
        if (m == nullptr || e < 0 || e >= m->n_expert || m->layer < 0) {
            continue;
        }
        moe_group_state & g = moe_group_get(ctx, m->layer, e);
        if (g.pinned || moe_group_is_active(ctx, g) ||
                moe_next_token_group_protected_locked(ctx, g) ||
                moe_group_has_inflight_or_queued(ctx, m->layer, e)) {
            continue;
        }
        const size_t resident = moe_group_resident_bytes(ctx, m->layer, e);
        if (resident == 0) {
            continue;
        }
        const double cache_score = moe_group_cache_score(ctx, g, resident);
        const double value = moe_prefetch_victim_value_locked(ctx, g, resident, cache_score);
        if (value < best_value) {
            best_value = value;
            ctx.admission_floor_value = value;
            ctx.admission_floor_layer = m->layer;
            ctx.admission_floor_expert = e;
            ctx.admission_floor_valid = true;
        }
    }
    ctx.admission_floor_refresh.fetch_add(1, std::memory_order_relaxed);
}

static bool moe_admission_floor_stale_locked(const llama_moe_buffer_context & ctx) {
    if (!ctx.admission_floor_valid) {
        return true;
    }
    const int refresh_tokens = std::max(1, moe_env_i32("LLAMA_LAZY_MOE_ADMISSION_REFRESH_TOKENS", 1));
    if (ctx.profile_token_epoch >= ctx.admission_floor_token + (uint64_t) refresh_tokens) {
        return true;
    }
    if (ctx.admission_floor_layer < 0 || ctx.admission_floor_expert < 0 ||
            moe_group_resident_bytes(ctx, ctx.admission_floor_layer, ctx.admission_floor_expert) == 0) {
        return true;
    }
    return false;
}

static moe_prefetch_admission_compare moe_prefetch_compare_locked(
        llama_moe_buffer_context & ctx,
        const moe_managed &        cand_m,
        int                        cand_expert,
        double                     prefetch_score,
        size_t                     target_bytes,
        size_t                     speculative_bytes) {
    moe_prefetch_admission_compare cmp;
    if (ctx.params.budget_bytes == 0 || !moe_env_flag("LLAMA_LAZY_MOE_ADMISSION_COMPARE", 1)) {
        return cmp;
    }
    const int low_budget_mb = moe_env_i32("LLAMA_LAZY_MOE_ADMISSION_LOW_BUDGET_MB", 768);
    if (ctx.params.budget_bytes > (size_t) std::max(0, low_budget_mb) * 1048576ull) {
        return cmp;
    }
    const double pressure_threshold = moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_PRESSURE", 0.88);
    const size_t projected = ctx.resident_bytes + speculative_bytes + target_bytes;
    cmp.pressure = (double) projected > (double) ctx.params.budget_bytes * pressure_threshold;
    if (!cmp.pressure) {
        return cmp;
    }

    const moe_group_state & cand_g = moe_group_get(ctx, cand_m.layer, cand_expert);
    cmp.candidate_value = moe_prefetch_candidate_value_locked(ctx, cand_g, prefetch_score, target_bytes);

    if (moe_admission_floor_stale_locked(ctx)) {
        ctx.admission_floor_stale.fetch_add(1, std::memory_order_relaxed);
        moe_admission_floor_refresh_locked(ctx);
    }
    if (ctx.admission_floor_valid &&
            !(ctx.admission_floor_layer == cand_m.layer && ctx.admission_floor_expert == cand_expert)) {
        cmp.victim_layer = ctx.admission_floor_layer;
        cmp.victim_expert = ctx.admission_floor_expert;
        cmp.victim_value = ctx.admission_floor_value;
        cmp.has_victim = true;
    }
    return cmp;
}

static bool moe_demand_admission_enabled() {
    static const bool enabled = moe_env_flag("LLAMA_LAZY_MOE_DEMAND_ADMISSION", 0);
    return enabled;
}

struct moe_demand_admission_request {
    int layer = -1;
    int expert = -1;
    size_t target_bytes = 0;
    double incoming_value = 0.0;
    double margin = 0.0;
};

struct moe_demand_admission_result {
    bool evaluated = false;
    bool bypass = false;
    bool has_victim = false;
    int victim_layer = -1;
    int victim_expert = -1;
    int victim_groups = 0;
    size_t victim_bytes = 0;
    double victim_value = 0.0;
    std::vector<uint64_t> victim_keys;
};

static void moe_admission_outcome_begin_locked(
        llama_moe_buffer_context &             ctx,
        moe_group_state &                      g,
        bool                                   bypass,
        double                                 candidate_value,
        const moe_demand_admission_result &    result) {
    ++g.admission_outcome_generation;
    g.admission_outcome_token = ctx.profile_token_epoch;
    const uint64_t horizon = (uint64_t) std::max(1,
            moe_env_i32("LLAMA_LAZY_MOE_ADMISSION_OUTCOME_TOKENS", 16));
    g.admission_outcome_deadline =
        ctx.profile_token_epoch > UINT64_MAX - horizon ?
        UINT64_MAX : ctx.profile_token_epoch + horizon;
    g.admission_outcome_action = bypass ? 2 : 1;
    g.admission_outcome_pending = true;
    g.admission_outcome_candidate = candidate_value;
    g.admission_outcome_victim = result.victim_value;
    g.admission_outcome_victim_layer = result.victim_layer;
    g.admission_outcome_victim_expert = result.victim_expert;
    g.admission_outcome_victim_groups = result.victim_groups;
    g.admission_outcome_victim_bytes = result.victim_bytes;
    moe_admission_outcome_trace_write_locked(
            ctx, "decision", g, 0, result.has_victim ? "real_victim" : "no_victim");
    moe_admission_regret_begin_locked(ctx, g, bypass, result.victim_keys);
}

static size_t moe_demand_group_missing_bytes_locked(
        const llama_moe_buffer_context & ctx,
        int                              layer,
        int                              expert,
        int                              rank) {
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return 0;
    }
    size_t bytes = 0;
    for (const moe_managed * m : it->second) {
        if (m == nullptr || expert < 0 || expert >= m->n_expert) {
            continue;
        }
        const int bits = moe_target_bits_for_rank(ctx, *m, expert, rank);
        if (moe_mwq_resolve_loaded_bits(*m, expert, bits) > 0 ||
                moe_native_resident_compatible(ctx, *m, expert, bits)) {
            continue;
        }
        bytes += moe_effective_stream_bytes_for_target(ctx, *m, expert, bits);
    }
    return bytes;
}

enum class moe_prefetch_admit_result : uint8_t {
    ADMIT = 0,
    BUDGET,
    DISTANCE,
    SCORE,
    ACTIVE,
    PINNED,
    RECENT,
    CCT,
    ADMISSION,
};

static const char * moe_prefetch_admit_label(moe_prefetch_admit_result r) {
    switch (r) {
        case moe_prefetch_admit_result::ADMIT: return "admit";
        case moe_prefetch_admit_result::BUDGET: return "drop_budget";
        case moe_prefetch_admit_result::DISTANCE: return "drop_distance";
        case moe_prefetch_admit_result::SCORE: return "drop_score";
        case moe_prefetch_admit_result::ACTIVE: return "drop_active";
        case moe_prefetch_admit_result::PINNED: return "drop_pinned";
        case moe_prefetch_admit_result::RECENT: return "drop_recent";
        case moe_prefetch_admit_result::CCT: return "drop_cct";
        case moe_prefetch_admit_result::ADMISSION: return "drop_admission_value";
    }
    return "drop_unknown";
}

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
        case moe_prefetch_admit_result::CCT:
            ctx.cct_prefetch_drop.fetch_add(1, std::memory_order_relaxed);
            break;
        case moe_prefetch_admit_result::ADMISSION:
            ctx.admission_compare_drop.fetch_add(1, std::memory_order_relaxed);
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
        bool                       force_next_layer,
        double *                   out_score,
        size_t *                   out_speculative_bytes,
        bool *                     out_forced,
        double *                   out_candidate_value = nullptr,
        double *                   out_victim_value = nullptr,
        int *                      out_victim_layer = nullptr,
        int *                      out_victim_expert = nullptr) {
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
    if (out_forced != nullptr) {
        *out_forced = false;
    }
    auto set_compare_out = [&](const moe_prefetch_admission_compare & cmp) {
        if (out_candidate_value != nullptr) {
            *out_candidate_value = cmp.candidate_value;
        }
        if (out_victim_value != nullptr) {
            *out_victim_value = cmp.victim_value;
        }
        if (out_victim_layer != nullptr) {
            *out_victim_layer = cmp.victim_layer;
        }
        if (out_victim_expert != nullptr) {
            *out_victim_expert = cmp.victim_expert;
        }
    };

    const int dist = moe_forward_layer_distance(ctx, m.layer);
    if (dist > moe_eam_prefetch_max_distance(ctx)) {
        return moe_prefetch_admit_result::DISTANCE;
    }
    const bool force_candidate = force_next_layer && moe_force_next_layer_prefetch_enabled() && dist == 1;
    if (budget == 0 && !force_candidate) {
        return moe_prefetch_admit_result::BUDGET;
    }
    if (force_candidate) {
        const size_t force_budget = (size_t) std::max(0,
                moe_env_i32("LLAMA_LAZY_MOE_FORCE_NEXT_LAYER_MAX_MB", 16)) * 1048576ull;
        const size_t max_groups = (size_t) std::max(0,
                moe_env_i32("LLAMA_LAZY_MOE_FORCE_NEXT_LAYER_MAX_GROUPS", 4));
        size_t layer_pending_groups = 0;
        size_t layer_pending_bytes = 0;
        bool candidate_pending = false;
        moe_layer_speculative_pending_locked(ctx, m.layer, e,
                &layer_pending_groups, &layer_pending_bytes, &candidate_pending);
        if (force_budget == 0 || layer_pending_bytes + target_bytes > force_budget) {
            return moe_prefetch_admit_result::BUDGET;
        }
        if (!candidate_pending && max_groups > 0 && layer_pending_groups >= max_groups) {
            return moe_prefetch_admit_result::BUDGET;
        }
        const double recent_score_threshold = moe_env_f64("LLAMA_LAZY_MOE_EAM_RECENT_SCORE", 80.0);
        if (!moe_eam_can_reclaim_without_protected_locked(ctx, target_bytes, recent_score_threshold)) {
            return moe_eam_prefetch_blocked_reason_locked(ctx, recent_score_threshold);
        }
        if (out_forced != nullptr) {
            *out_forced = true;
        }
        return moe_prefetch_admit_result::ADMIT;
    }
    if (moe_cct_prefetch_enabled() && moe_cct_lowmem_active(ctx)) {
        const uint8_t cct_conf = moe_cct_predict_conf_locked(ctx, m.layer, e);
        const int min_conf = moe_env_i32("LLAMA_LAZY_MOE_CCT_PREFETCH_MIN_CONF", 3);
        if ((int) cct_conf < min_conf) {
            moe_cct_trace_write_locked(ctx, "prefetch_drop", ctx.eam_last_actual_layer, -1,
                    m.layer, e, cct_conf, cct_conf, "low_conf");
            return moe_prefetch_admit_result::CCT;
        }
        ctx.cct_prefetch_admit.fetch_add(1, std::memory_order_relaxed);
        moe_cct_trace_write_locked(ctx, "prefetch_admit", ctx.eam_last_actual_layer, -1,
                m.layer, e, cct_conf, cct_conf, "high_conf");
    }
    if (budget > 0 && speculative_bytes + target_bytes > budget) {
        return moe_prefetch_admit_result::BUDGET;
    }
    moe_prefetch_admission_compare cmp =
        moe_prefetch_compare_locked(ctx, m, e, score, target_bytes, speculative_bytes);
    set_compare_out(cmp);
    if (cmp.pressure) {
        ctx.admission_compare_runs.fetch_add(1, std::memory_order_relaxed);
        ctx.admission_candidate_value_x1000.fetch_add(
                (uint64_t) std::max(0.0, cmp.candidate_value * 1000.0), std::memory_order_relaxed);
        ctx.admission_victim_value_x1000.fetch_add(
                (uint64_t) std::max(0.0, cmp.victim_value * 1000.0), std::memory_order_relaxed);
        const double margin = moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_MARGIN", 25.0);
        if (!cmp.has_victim) {
            ctx.admission_compare_no_victim.fetch_add(1, std::memory_order_relaxed);
            return moe_prefetch_admit_result::ADMISSION;
        }
        if (cmp.candidate_value <= cmp.victim_value + margin) {
            return moe_prefetch_admit_result::ADMISSION;
        }
        ctx.admission_compare_admit.fetch_add(1, std::memory_order_relaxed);
    }
    const double recent_score_threshold = moe_env_f64("LLAMA_LAZY_MOE_EAM_RECENT_SCORE", 80.0);
    if (!moe_eam_can_reclaim_without_protected_locked(ctx, target_bytes, recent_score_threshold)) {
        return moe_eam_prefetch_blocked_reason_locked(ctx, recent_score_threshold);
    }
    const double min_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_PREFETCH_MIN_SCORE", 30.0);
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
    moe_admission_regret_note_touch_locked(ctx, moe_group_key(m.layer, e));
    moe_admission_outcome_note_reuse_locked(ctx, g);
    g.access += count;
    moe_group_update_reuse_on_touch_locked(ctx, g);
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
    moe_group_lru_touch_locked(ctx, m.layer, e);
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

static void moe_release_slice_locked(
        llama_moe_buffer_context & ctx,
        moe_managed &              m,
        int                        e,
        bool                       erase_lru,
        bool                       keep_mwq_transient = false,
        bool                       count_eviction = true) {
    if (e < 0 || e >= m.n_expert || m.resident[e] != ST_RESIDENT) {
        return;
    }
    const bool keep_loaded = keep_mwq_transient && m.resident_mwq[e];
    if (keep_loaded) {
        moe_mark_mwq_transient(ctx, m, e, m.resident_bits[e]);
    }
    if (erase_lru) {
        ctx.lru.erase(m.lru_pos[e]);
    }
    uint8_t * resident_ptr = nullptr;
    if (m.resident_mwq[e]) {
        auto mit = m.mwq.find(m.resident_bits[e]);
        if (mit != m.mwq.end() && mit->second.buf != nullptr) {
            resident_ptr = mit->second.buf + (size_t) e * mit->second.stride;
            if (!keep_loaded && e < (int) mit->second.loaded.size()) {
                mit->second.loaded[e] = false;
            }
        }
    } else {
        resident_ptr = m.buf != nullptr ? m.buf + (size_t) e * m.stride : nullptr;
    }
    const size_t resident_size = m.resident_size[e];
    moe_group_resident_remove_locked(ctx, m, e, resident_size, m.resident_bits[e]);
    if (e < (int) m.prefetched.size() && m.prefetched[e] &&
            e < (int) m.resident_touched.size() && !m.resident_touched[e]) {
        moe_group_note_prefetch_unused_locked(ctx, m, e);
        if (ctx.profile) {
            ctx.prof_prefetch_unused.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (ctx.profile && count_eviction) {
        if (e < (int) m.last_evicted_token.size()) {
            m.last_evicted_token[e] = ctx.profile_token_epoch;
        }
        ctx.prof_evict_clean.fetch_add(1, std::memory_order_relaxed);
    }
    const uintptr_t base = (uintptr_t) resident_ptr;
    const uintptr_t pb   = (base + g_page - 1) & ~(uintptr_t) (g_page - 1);
    const uintptr_t pe   = (base + resident_size) & ~(uintptr_t) (g_page - 1);
#if defined(MADV_DONTNEED)
    if (!keep_loaded && resident_ptr != nullptr && pe > pb) {
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
    if (count_eviction) {
        ctx.evictions.fetch_add(1, std::memory_order_relaxed);
        ctx.group_evicted_slices.fetch_add(1, std::memory_order_relaxed);
    }
}

static void moe_release_demand_bypass_layer_locked(
        llama_moe_buffer_context & ctx,
        int                        layer) {
    if (!moe_demand_admission_enabled() || layer < 0) {
        return;
    }
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end() || it->second.empty() || it->second.front() == nullptr) {
        return;
    }
    const int n_expert = it->second.front()->n_expert;
    for (int expert = 0; expert < n_expert; ++expert) {
        moe_group_state & g = moe_group_get(ctx, layer, expert);
        if (!g.demand_admission_bypass) {
            continue;
        }
        size_t released = 0;
        uint64_t slices = 0;
        for (moe_managed * m : it->second) {
            if (m == nullptr || expert >= m->n_expert ||
                    m->resident[expert] != ST_RESIDENT) {
                continue;
            }
            released += m->resident_size[expert];
            ++slices;
            moe_release_slice_locked(ctx, *m, expert, true, true, false);
        }
        g.demand_admission_bypass = false;
        g.demand_admission_decided = false;
        if (released == 0) {
            continue;
        }
        moe_group_lru_unlink_locked(ctx, g);
        ctx.demand_admission_staging_bytes -= std::min(
                ctx.demand_admission_staging_bytes, released);
        ctx.demand_admission_release_groups.fetch_add(1, std::memory_order_relaxed);
        ctx.demand_admission_release_slices.fetch_add(slices, std::memory_order_relaxed);
        ctx.demand_admission_release_bytes.fetch_add(released, std::memory_order_relaxed);
    }
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
        if (layer >= 0 && expert >= 0) {
            moe_group_state & g = moe_group_get(ctx, layer, expert);
            moe_group_note_evicted_locked(ctx, g);
            moe_group_lru_unlink_locked(ctx, g);
        }
        ctx.group_evictions.fetch_add(1, std::memory_order_relaxed);
    }
    return released;
}

static size_t moe_drop_group_incompatible_locked(
        llama_moe_buffer_context & ctx,
        int                        layer,
        int                        expert) {
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
        moe_release_slice_locked(ctx, *m, expert, true, true);
    }
    if (released > 0 && layer >= 0 && expert >= 0) {
        moe_group_state & g = moe_group_get(ctx, layer, expert);
        moe_group_note_evicted_locked(ctx, g);
        moe_group_lru_unlink_locked(ctx, g);
        ctx.group_evictions.fetch_add(1, std::memory_order_relaxed);
    }
    return released;
}

static bool moe_layer_done_evict_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_EVICT_LAYER_DONE", 0);
}

static void moe_evict_layer_done_locked(llama_moe_buffer_context & ctx, int layer) {
    if (!moe_layer_done_evict_enabled() || layer < 0) {
        return;
    }
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end() || it->second.empty() || it->second[0] == nullptr) {
        return;
    }

    const int n_expert = it->second[0]->n_expert;
    const int recent_tokens = std::max(0, moe_env_i32("LLAMA_LAZY_MOE_EVICT_LAYER_DONE_RECENT_TOKENS", 1));
    const double recent_score = moe_env_f64("LLAMA_LAZY_MOE_EVICT_LAYER_DONE_RECENT_SCORE", 80.0);
    const uint64_t future_guard = (uint64_t) std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_EVICT_LAYER_DONE_FUTURE_EPOCHS",
                std::max(4, ctx.params.active_window)));
    const uint64_t bad_guard = (uint64_t) std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_EVICT_LAYER_DONE_BAD_RELOAD_TOKENS", 16));
    const double bad_score = moe_env_f64("LLAMA_LAZY_MOE_EVICT_LAYER_DONE_BAD_RELOAD_SCORE", 6.0);
    const bool cct_evict_active = moe_cct_evict_enabled() && moe_cct_lowmem_active(ctx);
    const int cct_protect_conf = moe_env_i32("LLAMA_LAZY_MOE_CCT_PROTECT_CONF", 3);
    const int cct_protect_distance = moe_env_i32("LLAMA_LAZY_MOE_CCT_PROTECT_DISTANCE", 1);
    const bool reuse_evict_active = moe_reuse_evict_enabled();
    const double reuse_protect_score = moe_env_f64("LLAMA_LAZY_MOE_REUSE_PROTECT_SCORE", 260.0);
    const int reuse_protect_tokens = moe_env_i32("LLAMA_LAZY_MOE_REUSE_PROTECT_TOKENS", 4);
    const double high_keep_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_SCORE", 125.0);
    const double high_keep_rate = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_RATE", 0.0080);
    const int early_high_layers = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_LAYERS", 4);
    const double early_high_keep_score =
        moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_KEEP_SCORE", 95.0);
    const double relaxed_next_use_weight = moe_env_f64("LLAMA_LAZY_MOE_RELAXED_NEXT_USE_WEIGHT", 0.0);
    const double relaxed_layer_distance_weight =
        moe_env_f64("LLAMA_LAZY_MOE_RELAXED_LAYER_DISTANCE_WEIGHT", 2.5e8);
    size_t released_total = 0;
    const size_t max_release = moe_env_mib_bytes("LLAMA_LAZY_MOE_EVICT_LAYER_DONE_MAX_MB", 0);

    for (int e = 0; e < n_expert; ++e) {
        const size_t resident = moe_group_resident_bytes(ctx, layer, e);
        if (resident == 0) {
            continue;
        }
        moe_group_state & g = moe_group_get(ctx, layer, e);
        g.hot_score = moe_group_cache_score(ctx, g, resident);

        moe_evict_protect_options opt;
        opt.protect_recent = true;
        opt.protect_high_recent = true;
        opt.protect_cct = false;
        opt.protect_predicted_soon = false;
        opt.aggressive_lazy_mode = 0;
        opt.recent_tokens = recent_tokens;
        opt.recent_score = recent_score;
        opt.bad_reload_guard_tokens = bad_guard;
        opt.bad_reload_score = bad_score;
        opt.cct_evict_active = cct_evict_active;
        opt.need_cct_score = false;
        opt.cct_protect_conf = cct_protect_conf;
        opt.cct_protect_distance = cct_protect_distance;
        opt.reuse_evict_active = reuse_evict_active;
        opt.reuse_protect_score = reuse_protect_score;
        opt.reuse_protect_tokens = reuse_protect_tokens;
        opt.high_keep_score = high_keep_score;
        opt.high_keep_rate = high_keep_rate;
        opt.early_high_layers = early_high_layers;
        opt.early_high_keep_score = early_high_keep_score;
        opt.relaxed_next_use_weight = relaxed_next_use_weight;
        opt.relaxed_layer_distance_weight = relaxed_layer_distance_weight;
        const moe_evict_protect_result protect =
            moe_group_evict_protect_locked(ctx, g, layer, e, g.hot_score, opt);
        if (protect.reason != moe_evict_protect_reason::NONE) {
            switch (protect.reason) {
                case moe_evict_protect_reason::PINNED:
                    ctx.layer_done_protect_pinned.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::ACTIVE:
                case moe_evict_protect_reason::CURRENT_LAYER:
                    ctx.layer_done_protect_active.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::INFLIGHT:
                    ctx.layer_done_protect_inflight.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::RECENT:
                    ctx.layer_done_protect_recent.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::BAD_RELOAD:
                    ctx.layer_done_protect_bad_reload.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::NEXT_TOKEN:
                    g.next_token_protect_hits++;
                    ctx.next_token_layer_done_protect.fetch_add(1, std::memory_order_relaxed);
                    ctx.layer_done_protect_next_token.fetch_add(1, std::memory_order_relaxed);
                    moe_next_token_trace_write_locked(ctx, "layer_done_keep", layer, e,
                            g.next_token_conf, g.next_token_conf, g.next_token_protect_until,
                            "hard_protect");
                    break;
                case moe_evict_protect_reason::PREDICTED_SOON:
                    if (future_guard > 0 && protect.next_use_dist != UINT64_MAX &&
                            protect.next_use_dist <= future_guard) {
                        ctx.layer_done_protect_future.fetch_add(1, std::memory_order_relaxed);
                    }
                    break;
                default:
                    break;
            }
            continue;
        }

        if (future_guard > 0 && protect.next_use_dist != UINT64_MAX && protect.next_use_dist <= future_guard) {
            ctx.layer_done_protect_future.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        const size_t released = moe_release_group_locked(ctx, layer, e);
        if (released == 0) {
            continue;
        }
        released_total += released;
        ctx.layer_done_evict_groups.fetch_add(1, std::memory_order_relaxed);
        ctx.layer_done_evict_slices.fetch_add(3, std::memory_order_relaxed);
        ctx.layer_done_evict_bytes.fetch_add((uint64_t) released, std::memory_order_relaxed);
        moe_group_get(ctx, layer, e).last_evicted_reason = MOE_EVICT_REASON_UNKNOWN;
        moe_evict_trace_write_locked(ctx, "evict", layer, e, "layer_done", 0, released, g.hot_score);

        if (ctx.admission_floor_layer == layer && ctx.admission_floor_expert == e) {
            ctx.admission_floor_valid = false;
        }
        if (max_release > 0 && released_total >= max_release) {
            break;
        }
    }
}

static bool moe_enqueue_prefetch(
        llama_moe_buffer_context & ctx,
        moe_managed &              m,
        int                        e,
        int                        rank,
        float                      score,
        bool                       force_next_layer = false) {
    if (e < 0 || e >= m.n_expert) {
        return false;
    }

    const int target_bits = moe_target_bits_for_rank(ctx, m, e, rank);
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        moe_group_future_hint_locked(ctx, m.layer, e, rank, score);
    }

    auto trace_prefetch = [&](const char * event, const char * reason, double admit_score) {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        const size_t speculative_bytes = moe_speculative_bytes_locked(ctx);
        moe_prefetch_trace_write_locked(ctx, m, e, rank, target_bits, event, reason,
                score, admit_score, speculative_bytes);
    };

    // Cheap pre-filter: a stale read only creates a no-op hit in the worker.
    if (m.resident[e] == ST_RESIDENT && (!ctx.params.dynamic_bits_real || m.resident_bits[e] >= target_bits)) {
        trace_prefetch("skip", "resident_compatible", 0.0);
        ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (ctx.params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(m, e, target_bits) > 0) {
        trace_prefetch("skip", "mwq_compatible", 0.0);
        ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (m.resident[e] == ST_INFLIGHT) {
        trace_prefetch("skip", "inflight", 0.0);
        ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    if (m.queued[e]) {
        const int pending_bits = e < (int) m.queued_bits.size() ? m.queued_bits[e] : 0;
        if (pending_bits >= target_bits) {
            trace_prefetch("skip", "queued_compatible", 0.0);
            ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        if (e < (int) m.queued_bits.size()) {
            m.queued_bits[e] = target_bits;
        }
        trace_prefetch("update", "queued_upgrade", 0.0);
    }

    if (ctx.params.dynamic_bits_real &&
            moe_mwq_entry(ctx, m, e, target_bits) == nullptr &&
            !moe_native_entry_available(ctx, m, target_bits)) {
        trace_prefetch("skip", "entry_missing", 0.0);
        ctx.queue_dups.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    double prefetch_score = 0.0;
    size_t speculative_bytes = 0;
    double candidate_value = 0.0;
    double victim_value = 0.0;
    int victim_layer = -1;
    int victim_expert = -1;
    bool forced = false;
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        const moe_prefetch_admit_result admit = moe_eam_prefetch_admit_locked(
                ctx, m, e, rank, score, target_bits, force_next_layer,
                &prefetch_score, &speculative_bytes, &forced,
                &candidate_value, &victim_value, &victim_layer, &victim_expert);
        moe_eam_note_prefetch_admission(ctx, admit, prefetch_score, speculative_bytes);
        if (admit != moe_prefetch_admit_result::ADMIT) {
            moe_prefetch_trace_write_locked(ctx, m, e, rank, target_bits, "drop",
                    moe_prefetch_admit_label(admit), score, prefetch_score, speculative_bytes,
                    candidate_value, victim_value, victim_layer, victim_expert);
            return false;
        }
        moe_prefetch_trace_write_locked(ctx, m, e, rank, target_bits, "admit",
                forced ? "force_next_layer" : "admit",
                score, prefetch_score, speculative_bytes,
                candidate_value, victim_value, victim_layer, victim_expert);
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
        moe_prefetch_trace_write_locked(ctx, m, e, rank, target_bits, "enqueue",
                forced ? "force_next_layer" : "queued",
                score, prefetch_score, speculative_bytes,
                candidate_value, victim_value, victim_layer, victim_expert);
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
    if (moe_cct_enabled() && (int) ctx.cct_recent_layer_experts.size() < n_layer) {
        ctx.cct_recent_layer_experts.resize((size_t) n_layer);
    }
    const bool adjacent =
        ctx.eam_last_actual_layer >= 0 &&
        ctx.eam_last_actual_layer != layer &&
        ((ctx.eam_last_actual_layer + 1) % n_layer) == layer;
    if (adjacent && !ctx.eam_last_layer_experts.empty()) {
        std::vector<uint8_t> predicted_before((size_t) n_expert, 0);
        if (moe_cct_enabled()) {
            for (int expert : actual_experts) {
                if (expert < 0 || expert >= n_expert) {
                    continue;
                }
                for (int prev_expert : ctx.eam_last_layer_experts) {
                    if (moe_cct_conf_from_source_locked(ctx, ctx.eam_last_actual_layer,
                                prev_expert, expert) > 0) {
                        predicted_before[(size_t) expert] = 1;
                        break;
                    }
                }
            }
        }
        for (int prev_expert : ctx.eam_last_layer_experts) {
            if (prev_expert < 0) {
                continue;
            }
            std::vector<uint32_t> & row =
                ctx.eam_layer_transition[moe_group_key(ctx.eam_last_actual_layer, prev_expert)];
            if ((int) row.size() < n_expert) {
                row.resize((size_t) n_expert, 0);
            }
            std::vector<moe_cct_entry> * cct_row = nullptr;
            if (moe_cct_enabled()) {
                cct_row = &ctx.cct_transition[moe_group_key(ctx.eam_last_actual_layer, prev_expert)];
                if ((int) cct_row->size() < n_expert) {
                    cct_row->resize((size_t) n_expert);
                }
            }
            std::vector<uint8_t> actual_mask((size_t) n_expert, 0);
            for (int expert : actual_experts) {
                if (expert >= 0 && expert < n_expert && row[(size_t) expert] != UINT32_MAX) {
                    row[(size_t) expert]++;
                    actual_mask[(size_t) expert] = 1;
                }
            }
            if (cct_row != nullptr) {
                for (int expert = 0; expert < n_expert; ++expert) {
                    moe_cct_entry & ent = (*cct_row)[(size_t) expert];
                    const uint8_t before = ent.conf;
                    if (actual_mask[(size_t) expert]) {
                        ent.conf = before == 0 ? 2 : (uint8_t) std::min<int>(3, before + 1);
                        if (before == 0) {
                            ent.replace++;
                            ctx.cct_replaced.fetch_add(1, std::memory_order_relaxed);
                        } else {
                            ent.hit++;
                            ctx.cct_hits.fetch_add(1, std::memory_order_relaxed);
                        }
                        moe_cct_trace_write_locked(ctx, "update", ctx.eam_last_actual_layer,
                                prev_expert, layer, expert, before, ent.conf,
                                before == 0 ? "replace_actual" : "actual_hit");
                    } else if (before > 0) {
                        ent.conf = (uint8_t) (before - 1);
                        ent.unused++;
                        ctx.cct_unused.fetch_add(1, std::memory_order_relaxed);
                        moe_cct_trace_write_locked(ctx, "update", ctx.eam_last_actual_layer,
                                prev_expert, layer, expert, before, ent.conf, "not_used");
                    }
                    ent.last_update_token = ctx.profile_token_epoch;
                }
            }
        }
        if (moe_cct_enabled()) {
            for (int expert : actual_experts) {
                if (expert >= 0 && expert < n_expert && !predicted_before[(size_t) expert]) {
                    ctx.cct_misses.fetch_add(1, std::memory_order_relaxed);
                    moe_cct_trace_write_locked(ctx, "miss", ctx.eam_last_actual_layer,
                            -1, layer, expert, 0, 2, "actual_not_predicted");
                }
            }
            ctx.cct_updates.fetch_add(1, std::memory_order_relaxed);
        }
    }
    ctx.eam_last_actual_layer = layer;
    ctx.eam_last_actual_token_epoch = ctx.profile_token_epoch;
    ctx.eam_last_layer_experts = actual_experts;
    if (moe_cct_enabled() && layer >= 0 && layer < (int) ctx.cct_recent_layer_experts.size()) {
        ctx.cct_recent_layer_experts[(size_t) layer] = actual_experts;
    }
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
                any = moe_enqueue_prefetch(ctx, *m, c.expert, c.rank, (float) c.score, true) || any;
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
static bool moe_evict_lru(
        llama_moe_buffer_context &              ctx,
        size_t                                  target_release_bytes = 0,
        const moe_demand_admission_request *    admission_request = nullptr,
        moe_demand_admission_result *           admission_result = nullptr,
        const std::vector<moe_demand_admission_request> * admission_batch = nullptr,
        std::vector<moe_demand_admission_result> * batch_results = nullptr,
        size_t                                  batch_free_bytes = 0,
        size_t                                  batch_staging_bytes = 0) {
    if (ctx.lru.empty()) {
        return false;
    }
    const uint64_t prof_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    const uint64_t prof_scan_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    uint64_t prof_sort_us = 0;
    uint64_t prof_online_us = 0;
    uint64_t prof_admission_us = 0;
    uint64_t prof_trace_us = 0;
    uint64_t prof_release_us = 0;
    uint64_t prof_candidate_us = 0;
    uint64_t prof_resident_queries = 0;

    moe_ghost_expire_locked(ctx);
    moe_olecar_expire_locked(ctx);
    moe_refresh_pins_locked(ctx);

    struct victim_candidate {
        int layer = -1;
        int expert = -1;
        double score = -1.0e300;
        double hot_score = 0.0;
        double lru_score = 0.0;
        double recency_score = 0.0;
        double cache_policy_score = 0.0;
        double bad_reload_policy_score = 0.0;
        double next_use_score = 0.0;
        double layer_score = 0.0;
        double reuse_score = 0.0;
        double cct_score = 0.0;
        uint64_t age = 0;
        bool belady = false;
        enum reason_t {
            SPEC_UNUSED,
            LOW_SCORE,
            FAR_FUTURE,
            LAYER_WINDOW,
            REUSE_LOW,
            EAM_REPLACE,
            RELAXED,
        } reason = LOW_SCORE;
    };

    struct olecar_policy_best {
        const char * name = "";
        int id = -1;
        bool has = false;
        double score = -1.0e300;
        victim_candidate candidate;
    };

    victim_candidate best;
    victim_candidate temporary_best;
    std::vector<olecar_policy_best> olecar_best = {
        {"final", 0, false, -1.0e300, {}},
        {"lru", 1, false, -1.0e300, {}},
        {"recency", 2, false, -1.0e300, {}},
        {"cache", 3, false, -1.0e300, {}},
        {"bad_reload", 4, false, -1.0e300, {}},
        {"next_use", 5, false, -1.0e300, {}},
        {"layer", 6, false, -1.0e300, {}},
        {"reuse", 7, false, -1.0e300, {}},
        {"cct", 8, false, -1.0e300, {}},
    };
    auto olecar_consider = [&](victim_candidate candidate) {
        if (candidate.layer < 0 || candidate.expert < 0) {
            return;
        }
        const double scores[] = {
            candidate.score,
            candidate.lru_score,
            candidate.recency_score,
            candidate.cache_policy_score,
            candidate.bad_reload_policy_score,
            candidate.next_use_score,
            candidate.layer_score,
            candidate.reuse_score,
            candidate.cct_score,
        };
        for (size_t i = 0; i < olecar_best.size(); ++i) {
            const double s = scores[i];
            olecar_policy_best & best_policy = olecar_best[i];
            const bool better =
                !best_policy.has || s > best_policy.score ||
                (s == best_policy.score &&
                    (candidate.age > best_policy.candidate.age ||
                        (candidate.age == best_policy.candidate.age &&
                            (candidate.hot_score < best_policy.candidate.hot_score ||
                                (candidate.hot_score == best_policy.candidate.hot_score &&
                                    (candidate.layer < best_policy.candidate.layer ||
                                        (candidate.layer == best_policy.candidate.layer &&
                                            candidate.expert < best_policy.candidate.expert)))))));
            if (better) {
                best_policy.has = true;
                best_policy.score = s;
                best_policy.candidate = candidate;
            }
        }
    };
    const bool olecar_online = moe_env_flag("LLAMA_LAZY_MOE_OLECAR_ONLINE", 0);
    const char * force_policy_env = std::getenv("LLAMA_LAZY_MOE_OLECAR_FORCE_POLICY");
    std::string force_policy = force_policy_env == nullptr ? "" : std::string(force_policy_env);
    const bool explicit_force_policy = !force_policy.empty();
    if (force_policy.empty() && ctx.params.budget_bytes > 0 &&
            ctx.params.budget_bytes <= 640ull * 1048576ull) {
        force_policy = "cache";
    }
    auto forced_policy_score = [&](const victim_candidate & candidate, double * out_score) {
        if (force_policy.empty() || force_policy == "final") {
            return false;
        }
        if (force_policy == "lru") {
            *out_score = candidate.lru_score;
        } else if (force_policy == "recency") {
            *out_score = candidate.recency_score;
        } else if (force_policy == "cache") {
            *out_score = candidate.cache_policy_score;
        } else if (force_policy == "bad_reload") {
            *out_score = candidate.bad_reload_policy_score;
        } else if (force_policy == "next_use") {
            *out_score = candidate.next_use_score;
        } else if (force_policy == "layer") {
            *out_score = candidate.layer_score;
        } else if (force_policy == "reuse") {
            *out_score = candidate.reuse_score;
        } else if (force_policy == "cct") {
            *out_score = candidate.cct_score;
        } else {
            return false;
        }
        return true;
    };
    auto better_candidate = [](const victim_candidate & a, const victim_candidate & b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        if (a.age != b.age) {
            return a.age > b.age;
        }
        if (a.hot_score != b.hot_score) {
            return a.hot_score < b.hot_score;
        }
        if (a.layer != b.layer) {
            return a.layer < b.layer;
        }
        return a.expert < b.expert;
    };
    std::vector<victim_candidate> normal_candidates;
    std::vector<victim_candidate> temporary_candidates;
    if (target_release_bytes > 0) {
        normal_candidates.reserve(ctx.groups.size());
        temporary_candidates.reserve(ctx.groups.size());
    }
    double temporary_keep_score = 1.0e300;
    const int recent_tokens = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_RECENT_TOKENS", 3);
    const uint64_t bad_reload_guard_tokens = (uint64_t) std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_BAD_RELOAD_GUARD_TOKENS", 16));
    const double bad_reload_protect_score =
        moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_PROTECT_SCORE", 6.0);
    const bool cct_evict_active = moe_cct_evict_enabled() && moe_cct_lowmem_active(ctx);
    const bool need_cct_score = cct_evict_active || olecar_online ||
        force_policy == "cct" || ctx.olecar_trace != nullptr;
    const int cct_protect_conf = moe_env_i32("LLAMA_LAZY_MOE_CCT_PROTECT_CONF", 3);
    const int cct_protect_distance = moe_env_i32("LLAMA_LAZY_MOE_CCT_PROTECT_DISTANCE", 1);
    const bool reuse_evict_active = moe_reuse_evict_enabled();
    const int aggressive_lazy_mode = std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_EVICT_AGGRESSIVE_LAZY", 2));
    const double reuse_protect_score = moe_env_f64("LLAMA_LAZY_MOE_REUSE_PROTECT_SCORE", 260.0);
    const int reuse_protect_tokens = moe_env_i32("LLAMA_LAZY_MOE_REUSE_PROTECT_TOKENS", 4);
    const double high_keep_score = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_SCORE", 125.0);
    const double high_keep_rate = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_KEEP_RATE", 0.0080);
    const int early_high_layers = moe_env_i32("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_LAYERS", 4);
    const double early_high_keep_score =
        moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_EARLY_KEEP_SCORE", 95.0);
    const double relaxed_next_use_weight = moe_env_f64("LLAMA_LAZY_MOE_RELAXED_NEXT_USE_WEIGHT", 0.0);
    const double relaxed_layer_distance_weight =
        moe_env_f64("LLAMA_LAZY_MOE_RELAXED_LAYER_DISTANCE_WEIGHT", 2.5e8);
    const uint64_t eam_replace_next_guard = (uint64_t) std::max(0,
            moe_env_i32("LLAMA_LAZY_MOE_EAM_REPLACE_NEXT_USE_GUARD",
                std::max(4, ctx.params.active_window)));
    const double layer_reload_protect_penalty =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT_PENALTY", 4.0e8);
    const double spec_unused_bonus_env =
        moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_SPEC_UNUSED_BONUS", 3.0e8);
    const double low_score_ceil = moe_env_f64("LLAMA_LAZY_MOE_EAM_EVICT_LOW_SCORE_CEIL", 140.0);
    const double reuse_evict_ceil = moe_env_f64("LLAMA_LAZY_MOE_REUSE_EVICT_CEIL", 170.0);
    const double reuse_evict_weight = moe_env_f64("LLAMA_LAZY_MOE_REUSE_EVICT_WEIGHT", 65536.0);
    const double reuse_keep_weight = moe_env_f64("LLAMA_LAZY_MOE_REUSE_KEEP_WEIGHT", 0.0);
    const double reuse_protect_penalty_env =
        moe_env_f64("LLAMA_LAZY_MOE_REUSE_PROTECT_PENALTY", 3.0e8);
    const double layer_reserve_evict_bonus =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RESERVE_EVICT_BONUS", 2.5e8);
    const double layer_reserve_protect_penalty =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RESERVE_PROTECT_PENALTY", 3.0e8);
    const double layer_reserve_persistent_score =
        moe_env_f64("LLAMA_LAZY_MOE_LAYER_RESERVE_PERSISTENT_SCORE", 120.0);
    const double bad_reload_evict_weight =
        moe_env_f64("LLAMA_LAZY_MOE_BAD_RELOAD_EVICT_WEIGHT", 2.0e8);
    const double cct_evict_keep_weight =
        moe_env_f64("LLAMA_LAZY_MOE_CCT_EVICT_KEEP_WEIGHT", 2.5e8);
    const double eam_replace_score_scale =
        moe_env_f64("LLAMA_LAZY_MOE_EAM_REPLACE_SCORE_SCALE", 1.0e12);
    const bool eam_replace = moe_eam_replace_enabled();
    bool eam_replace_stats_ready = false;
    moe_eam_replace_stats eam_replace_stats;
    uint64_t scan_entries = 0;
    uint64_t scan_unique = 0;
    uint64_t scan_duplicates = 0;
    uint64_t scan_absolute = 0;
    uint64_t scan_temporary = 0;
    uint64_t scan_normal = 0;
    size_t scan_candidate_bytes = 0;
    if (++ctx.evict_scan_generation == 0) {
        for (auto & kv : ctx.groups) {
            kv.second.evict_scan_generation = 0;
        }
        ctx.evict_scan_generation = 1;
    }
    const uint64_t scan_generation = ctx.evict_scan_generation;
    size_t target_window_bytes = 0;
    bool layer_reserve_needed = false;
    const bool layer_reload_protect_enabled = moe_env_flag("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT", 0);
    const bool layer_reload_hard_protect = moe_env_flag("LLAMA_LAZY_MOE_LAYER_RELOAD_HARD_PROTECT", 1);
    const int layer_feedback_window = moe_env_i32("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT_WINDOW", 16);
    const uint64_t layer_feedback_min = (uint64_t) std::max(1,
            moe_env_i32("LLAMA_LAZY_MOE_LAYER_RELOAD_PROTECT_MIN", 2));
    auto layer_feedback_protected = [&](const moe_group_state & g, bool layer_window) {
        if (!layer_reload_protect_enabled || !layer_reserve_needed || layer_window ||
                g.layer_window_bad_reload < layer_feedback_min) {
            return false;
        }
        const uint64_t age =
            g.layer_window_last_bad_reload_token == 0 || ctx.profile_token_epoch < g.layer_window_last_bad_reload_token ?
            UINT64_MAX : ctx.profile_token_epoch - g.layer_window_last_bad_reload_token;
        return layer_feedback_window <= 0 ||
            (age != UINT64_MAX && age <= (uint64_t) layer_feedback_window);
    };
    if (ctx.evict_target_layer >= 0 && ctx.evict_layer_reserve_bytes > 0) {
        const int n_layer = std::max(1, moe_max_layer_index(ctx) + 1);
        for (int d = 0; d <= std::max(0, ctx.evict_target_ahead); ++d) {
            target_window_bytes += moe_layer_resident_bytes_locked(ctx, (ctx.evict_target_layer + d) % n_layer);
        }
        layer_reserve_needed = target_window_bytes < ctx.evict_layer_reserve_bytes;
    }
    std::vector<std::pair<moe_managed *, int>> evict_scan_items;
    const bool group_lru_scan = moe_env_flag("LLAMA_LAZY_MOE_GROUP_LRU_SCAN", 1);
    const int sample_k_env = moe_env_i32("LLAMA_LAZY_MOE_EVICT_SAMPLE_K", 0);
    const size_t sample_k = sample_k_env <= 0 ? 0 : (size_t) sample_k_env;
    const bool sample_fallback_bytes =
        moe_env_flag("LLAMA_LAZY_MOE_EVICT_SAMPLE_FALLBACK_BYTES", 0);
    const int cold_window_env = moe_env_i32("LLAMA_LAZY_MOE_EVICT_COLD_WINDOW", 0);
    const size_t cold_window = cold_window_env <= 0 ? 0 : (size_t) cold_window_env;
    std::vector<std::pair<moe_managed *, int>> all_scan_items;
    std::vector<std::pair<moe_managed *, int>> candidate_pool;
    bool sampled_scan = false;
    bool cold_window_scan = false;
    if (group_lru_scan) {
        if (ctx.group_lru.empty() && !ctx.lru.empty()) {
            moe_group_lru_rebuild_locked(ctx);
        }
        all_scan_items.reserve(ctx.group_lru.size());
        for (auto it = ctx.group_lru.rbegin(); it != ctx.group_lru.rend(); ++it) {
            const int layer = (int) (uint32_t) (*it >> 32);
            const int e = (int) (uint32_t) *it;
            auto lit = ctx.by_layer.find(layer);
            if (lit == ctx.by_layer.end() || lit->second.empty() || lit->second.front() == nullptr) {
                continue;
            }
            all_scan_items.push_back({lit->second.front(), e});
        }
        if (cold_window > 0 && cold_window < all_scan_items.size()) {
            cold_window_scan = true;
            candidate_pool.assign(all_scan_items.begin(), all_scan_items.begin() + cold_window);
            ctx.evict_cold_window_calls.fetch_add(1, std::memory_order_relaxed);
            ctx.evict_cold_window_items.fetch_add(candidate_pool.size(), std::memory_order_relaxed);
            ctx.evict_cold_window_source.fetch_add(all_scan_items.size(), std::memory_order_relaxed);
        } else {
            candidate_pool = all_scan_items;
        }
        size_t effective_sample_k = sample_k;
        if (effective_sample_k > 0 && target_release_bytes > 0 && ctx.resident_bytes > 0 &&
                !ctx.group_lru.empty()) {
            const size_t avg_group_bytes =
                std::max<size_t>(1, ctx.resident_bytes / ctx.group_lru.size());
            const size_t release_groups =
                (target_release_bytes + avg_group_bytes - 1) / avg_group_bytes;
            effective_sample_k = std::max(effective_sample_k, release_groups + sample_k);
        }
        if (effective_sample_k > 0 && effective_sample_k < candidate_pool.size()) {
            sampled_scan = true;
            evict_scan_items.reserve(effective_sample_k);
            std::unordered_set<size_t> selected_indexes;
            uint64_t x = scan_generation * 0x9e3779b97f4a7c15ull;
            x ^= ctx.profile_token_epoch + 0xbf58476d1ce4e5b9ull + (x << 6) + (x >> 2);
            x ^= ctx.exec_epoch + 0x94d049bb133111ebull + (x << 6) + (x >> 2);
            size_t attempts = 0;
            const size_t max_attempts = candidate_pool.size() * 4;
            while (evict_scan_items.size() < effective_sample_k && attempts++ < max_attempts) {
                x ^= x >> 12;
                x ^= x << 25;
                x ^= x >> 27;
                const size_t idx =
                    (size_t) ((x * 0x2545f4914f6cdd1dull) % candidate_pool.size());
                if (!selected_indexes.insert(idx).second) {
                    continue;
                }
                evict_scan_items.push_back(candidate_pool[idx]);
            }
            for (size_t i = 0; evict_scan_items.size() < effective_sample_k && i < candidate_pool.size(); ++i) {
                if (selected_indexes.insert(i).second) {
                    evict_scan_items.push_back(candidate_pool[i]);
                }
            }
            ctx.evict_sample_calls.fetch_add(1, std::memory_order_relaxed);
            ctx.evict_sample_k.fetch_add(evict_scan_items.size(), std::memory_order_relaxed);
            ctx.evict_sample_source.fetch_add(candidate_pool.size(), std::memory_order_relaxed);
        } else {
            evict_scan_items = candidate_pool;
        }
    } else {
        evict_scan_items.reserve(ctx.lru.size());
        for (auto it = ctx.lru.rbegin(); it != ctx.lru.rend(); ++it) {
            evict_scan_items.push_back(*it);
        }
        all_scan_items = evict_scan_items;
    }
    auto scan_pass = [&](const std::vector<std::pair<moe_managed *, int>> & scan_items) {
    for (const auto & scan_item : scan_items) {
        ++scan_entries;
        moe_managed * m = scan_item.first;
        const int e = scan_item.second;
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
        if (g.evict_scan_generation == scan_generation) {
            ++scan_duplicates;
            continue;
        }
        g.evict_scan_generation = scan_generation;

        ++prof_resident_queries;
        const uint64_t prof_candidate_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        const size_t bytes = moe_group_resident_bytes(ctx, m->layer, e);
        if (bytes == 0) {
            continue;
        }
        ++scan_unique;
        const uint64_t group_age =
            g.last_used_epoch == 0 || ctx.exec_epoch < g.last_used_epoch ?
            UINT64_MAX : ctx.exec_epoch - g.last_used_epoch;
        const uint64_t token_age =
            g.last_used_token_epoch == 0 || ctx.profile_token_epoch < g.last_used_token_epoch ?
            UINT64_MAX : ctx.profile_token_epoch - g.last_used_token_epoch;

        g.hot_score = moe_group_cache_score(ctx, g, bytes);
        const bool current_layer = ctx.profile_last_layer >= 0 && m->layer == ctx.profile_last_layer;
        const bool layer_window = moe_layer_in_evict_window(ctx, m->layer);
        const bool speculative_unused = moe_group_has_speculative_unused(ctx, m->layer, e);
        const bool layer_feedback_guard = !speculative_unused &&
            layer_feedback_protected(g, layer_window);
        if (layer_feedback_guard) {
            ctx.layer_reload_protect.fetch_add(1, std::memory_order_relaxed);
        }

        moe_evict_protect_options opt;
        opt.speculative_unused = speculative_unused;
        opt.current_layer = current_layer;
        opt.layer_window = layer_window;
        opt.protect_layer_window = layer_reserve_needed;
        opt.protect_recent = true;
        opt.aggressive_lazy_mode = aggressive_lazy_mode;
        opt.recent_tokens = recent_tokens;
        opt.bad_reload_guard_tokens = bad_reload_guard_tokens;
        opt.bad_reload_score = bad_reload_protect_score;
        opt.cct_evict_active = cct_evict_active;
        opt.need_cct_score = need_cct_score;
        opt.cct_protect_conf = cct_protect_conf;
        opt.cct_protect_distance = cct_protect_distance;
        opt.reuse_evict_active = reuse_evict_active;
        opt.reuse_protect_score = reuse_protect_score;
        opt.reuse_protect_tokens = reuse_protect_tokens;
        opt.high_keep_score = high_keep_score;
        opt.high_keep_rate = high_keep_rate;
        opt.early_high_layers = early_high_layers;
        opt.early_high_keep_score = early_high_keep_score;
        opt.relaxed_next_use_weight = relaxed_next_use_weight;
        opt.relaxed_layer_distance_weight = relaxed_layer_distance_weight;
        const moe_evict_protect_result protect =
            moe_group_evict_protect_locked(ctx, g, m->layer, e, g.hot_score, opt);
        if (!speculative_unused) {
            if (protect.recent) {
                ctx.eam_evict_protect_recent.fetch_add(1, std::memory_order_relaxed);
            }
            if (protect.high_sequence || moe_group_is_hot(ctx, m->layer, e)) {
                ctx.eam_evict_protect_high.fetch_add(1, std::memory_order_relaxed);
            }
            if (protect.early_high) {
                ctx.eam_evict_protect_early.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (protect.reason != moe_evict_protect_reason::NONE) {
            switch (protect.reason) {
                case moe_evict_protect_reason::PINNED:
                    ctx.eam_evict_protect_pinned.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::ACTIVE:
                case moe_evict_protect_reason::CURRENT_LAYER:
                    ctx.eam_evict_protect_active.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::BAD_RELOAD:
                    g.bad_reload_protect_hits++;
                    ctx.bad_reload_protect.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::NEXT_TOKEN:
                    g.next_token_protect_hits++;
                    ctx.next_token_evict_protect.fetch_add(1, std::memory_order_relaxed);
                    moe_next_token_trace_write_locked(ctx, "evict_keep", m->layer, e,
                            g.next_token_conf, g.next_token_conf, g.next_token_protect_until,
                            "global_victim_protect");
                    break;
                case moe_evict_protect_reason::CCT:
                    g.cct_protect_hits++;
                    ctx.cct_evict_protect.fetch_add(1, std::memory_order_relaxed);
                    moe_cct_trace_write_locked(ctx, "evict_protect", ctx.eam_last_actual_layer, -1,
                            m->layer, e, protect.cct_conf, protect.cct_conf, "near_high_conf");
                    break;
                case moe_evict_protect_reason::PREDICTED_SOON:
                    ctx.reuse_predict_protect.fetch_add(1, std::memory_order_relaxed);
                    break;
                case moe_evict_protect_reason::REUSE:
                    ctx.reuse_protect.fetch_add(1, std::memory_order_relaxed);
                    break;
                default:
                    break;
            }
        }

        const uint64_t dist = protect.next_use_dist;
        const int layer_dist = protect.layer_dist;
        const bool recent = protect.recent;
        const bool high_sequence = protect.high_sequence;
        const bool early_high = protect.early_high;
        const double reuse_keep = protect.reuse_keep;
        const double bad_reload_effective = protect.bad_reload_effective;
        const uint8_t cct_conf = protect.cct_conf;
        const bool cct_lowmem = cct_evict_active;
        const bool eam_replace_next_protected =
            eam_replace && !speculative_unused && eam_replace_next_guard > 0 &&
            dist != UINT64_MAX && dist <= eam_replace_next_guard;
        if (eam_replace_next_protected) {
            ctx.reuse_predict_protect.fetch_add(1, std::memory_order_relaxed);
        }
        if (protect.protect_class == moe_evict_protect_class::ABSOLUTE) {
            ++scan_absolute;
            continue;
        }
        const bool external_temporary =
            (layer_feedback_guard && layer_reload_hard_protect) || eam_replace_next_protected;
        if (protect.protect_class == moe_evict_protect_class::TEMPORARY || external_temporary) {
            ++scan_temporary;
            double keep_score = protect.temporary_keep_score;
            if (layer_feedback_guard && layer_reload_hard_protect) {
                keep_score += layer_reload_protect_penalty *
                    (double) std::min<uint64_t>(4, g.layer_window_bad_reload);
            }
            if (eam_replace_next_protected) {
                keep_score += (double) (eam_replace_next_guard + 1 - dist) * 1.0e8;
            }
            victim_candidate candidate;
            candidate.layer = m->layer;
            candidate.expert = e;
            candidate.score = -keep_score;
            candidate.hot_score = g.hot_score;
            candidate.lru_score = group_age == UINT64_MAX ? 1.0e9 : (double) group_age;
            candidate.recency_score = token_age == UINT64_MAX ? 1.0e9 : (double) token_age;
            candidate.cache_policy_score = -g.hot_score;
            candidate.bad_reload_policy_score = -bad_reload_effective;
            candidate.next_use_score = dist == UINT64_MAX ? 1.0e9 : (double) dist;
            candidate.layer_score = (double) layer_dist;
            candidate.reuse_score = -reuse_keep;
            candidate.cct_score = -(double) cct_conf;
            candidate.age = group_age;
            candidate.belady = protect.future_ranked;
            candidate.reason = victim_candidate::RELAXED;
            double forced_score = 0.0;
            const bool forced = forced_policy_score(candidate, &forced_score);
            if (forced) {
                candidate.score = forced_score;
            }
            olecar_consider(candidate);
            if (forced) {
                if (target_release_bytes > 0) {
                    normal_candidates.push_back(candidate);
                }
                if (better_candidate(candidate, best)) {
                    best = candidate;
                }
                continue;
            }
            if (target_release_bytes > 0) {
                temporary_candidates.push_back(candidate);
                scan_candidate_bytes += bytes;
            }
            const bool better_temporary =
                keep_score < temporary_keep_score ||
                (keep_score == temporary_keep_score &&
                    (group_age > temporary_best.age ||
                        (group_age == temporary_best.age &&
                            (g.hot_score < temporary_best.hot_score ||
                                (g.hot_score == temporary_best.hot_score &&
                                    (m->layer < temporary_best.layer ||
                                        (m->layer == temporary_best.layer && e < temporary_best.expert)))))));
            if (better_temporary) {
                temporary_keep_score = keep_score;
                temporary_best = candidate;
            }
            continue;
        }
        ++scan_normal;
        const bool predicted_soon = protect.predicted_soon;
        const bool reuse_protected = protect.reuse_protected;
        const bool spec_evictable = speculative_unused &&
            moe_group_speculative_unused_evictable(ctx, g, dist, layer_dist);
        const double speculative_bonus = spec_evictable ?
            spec_unused_bonus_env : 0.0;
        const double low_score_bonus = std::max(0.0,
                low_score_ceil - g.hot_score) * 16384.0;
        const double dist_score = dist == UINT64_MAX ? 1.0e9 : (double) dist * 8192.0;
        const double recent_penalty = recent ? 1.0e8 : 0.0;
        const double spec_guard_penalty = speculative_unused && !spec_evictable ? 4.0e8 : 0.0;
        const double size_bonus = (double) bytes / 1048576.0;
        const double bit_bonus = moe_group_resident_bit_score(ctx, m->layer, e) * 8.0;
        const double high_penalty = high_sequence || moe_group_is_hot(ctx, m->layer, e) ? 5.0e8 : 0.0;
        const double early_penalty = early_high ? 2.0e8 : 0.0;
        const bool reuse_low_candidate = reuse_evict_active && g.reuse_observed > 0 &&
            !predicted_soon && !layer_window &&
            reuse_keep < reuse_evict_ceil;
        if (reuse_low_candidate) {
            ctx.reuse_low_candidates.fetch_add(1, std::memory_order_relaxed);
        }
        const double reuse_low_bonus = reuse_low_candidate ?
            std::max(0.0, reuse_evict_ceil - reuse_keep) *
                reuse_evict_weight : 0.0;
        const double reuse_keep_penalty = reuse_evict_active ?
            reuse_keep * reuse_keep_weight : 0.0;
        const double reuse_protect_penalty = reuse_protected ?
            reuse_protect_penalty_env : 0.0;
        const double layer_window_bonus = layer_reserve_needed && !layer_window ?
            layer_reserve_evict_bonus : 0.0;
        const double layer_window_penalty = layer_reserve_needed && layer_window ?
            layer_reserve_protect_penalty : 0.0;
        const double layer_feedback_penalty = layer_feedback_guard ?
            layer_reload_protect_penalty *
                (double) std::min<uint64_t>(4, g.layer_window_bad_reload) : 0.0;
        const double persistent_penalty = !layer_window && ctx.evict_persistent_bytes > 0 &&
            ctx.resident_bytes <= ctx.evict_persistent_bytes && g.hot_score >=
                layer_reserve_persistent_score ? 2.0e8 : 0.0;
        const double bad_reload_penalty = !speculative_unused ?
            bad_reload_effective * bad_reload_evict_weight : 0.0;
        if (bad_reload_penalty > 0.0) {
            g.bad_reload_soft_hits++;
            ctx.bad_reload_soft_keep.fetch_add(1, std::memory_order_relaxed);
        }
        const double cct_keep_penalty = cct_lowmem ?
            (double) cct_conf * cct_evict_keep_weight : 0.0;
        if (cct_keep_penalty > 0.0) {
            ctx.cct_evict_keep.fetch_add(1, std::memory_order_relaxed);
        }
        if (eam_replace && !eam_replace_stats_ready) {
            eam_replace_stats = moe_eam_replace_build_stats_locked(ctx);
            eam_replace_stats_ready = true;
        }
        const double layer_replace_mass = eam_replace && m->layer >= 0 && m->layer < eam_replace_stats.n_layers ?
            eam_replace_stats.layer_mass[(size_t) m->layer] : 0.0;
        const double eam_replace_pred = eam_replace ?
            moe_group_eam_replace_pred_from_stats_locked(ctx, eam_replace_stats, g) : 0.0;
        const double eam_replace_keep = eam_replace ?
            moe_group_eam_replace_keep_from_stats_locked(ctx, eam_replace_stats, g) : 0.0;
        if (eam_replace) {
            g.eam_replace_pred = eam_replace_pred;
            g.eam_replace_mass = layer_replace_mass;
            g.eam_replace_keep = eam_replace_keep;
        }
        const double score = eam_replace ?
            (-eam_replace_keep * eam_replace_score_scale +
                size_bonus - cct_keep_penalty - bad_reload_penalty) :
            (speculative_bonus + low_score_bonus + reuse_low_bonus + dist_score +
                size_bonus + bit_bonus + layer_window_bonus -
                recent_penalty - spec_guard_penalty - high_penalty - early_penalty -
                reuse_keep_penalty - reuse_protect_penalty -
                layer_window_penalty - layer_feedback_penalty - persistent_penalty -
                cct_keep_penalty - bad_reload_penalty);
        victim_candidate candidate;
        candidate.layer = m->layer;
        candidate.expert = e;
        candidate.score = score;
        candidate.hot_score = g.hot_score;
        candidate.lru_score = group_age == UINT64_MAX ? 1.0e9 : (double) group_age;
        candidate.recency_score = token_age == UINT64_MAX ? 1.0e9 : (double) token_age;
        candidate.cache_policy_score = -g.hot_score;
        candidate.bad_reload_policy_score = -bad_reload_effective;
        candidate.next_use_score = dist == UINT64_MAX ? 1.0e9 : (double) dist;
        candidate.layer_score = (double) layer_dist;
        candidate.reuse_score = -reuse_keep;
        candidate.cct_score = -(double) cct_conf;
        candidate.age = group_age;
        candidate.belady = dist != UINT64_MAX;
        candidate.reason = eam_replace ? victim_candidate::EAM_REPLACE :
            layer_reserve_needed && !layer_window ? victim_candidate::LAYER_WINDOW :
            spec_evictable ? victim_candidate::SPEC_UNUSED :
            (reuse_low_bonus > 0.0 && reuse_keep < reuse_evict_ceil ?
                victim_candidate::REUSE_LOW :
            (dist == UINT64_MAX || dist > (uint64_t) std::max(1, ctx.params.active_window) ?
                victim_candidate::FAR_FUTURE : victim_candidate::LOW_SCORE));
        double forced_score = 0.0;
        if (forced_policy_score(candidate, &forced_score)) {
            candidate.score = forced_score;
        }
        olecar_consider(candidate);
        if (target_release_bytes > 0) {
            normal_candidates.push_back(candidate);
            scan_candidate_bytes += bytes;
        }
        if (better_candidate(candidate, best)) {
            best = candidate;
        }
        if (ctx.profile) {
            prof_candidate_us += (moe_profile_now_ns() - prof_candidate_t0) / 1000;
        }
    }
    };
    scan_pass(evict_scan_items);
    const bool limited_no_candidate =
        (sampled_scan || cold_window_scan) &&
        ((target_release_bytes > 0 && normal_candidates.empty() && temporary_candidates.empty()) ||
         (sample_fallback_bytes && target_release_bytes > 0 &&
             scan_candidate_bytes < target_release_bytes) ||
         (target_release_bytes == 0 && best.layer < 0 && temporary_best.layer < 0));
    if (limited_no_candidate) {
        if (sampled_scan) {
            ctx.evict_sample_fallback.fetch_add(1, std::memory_order_relaxed);
        }
        if (cold_window_scan) {
            ctx.evict_cold_window_fallback.fetch_add(1, std::memory_order_relaxed);
        }
        scan_pass(all_scan_items);
    }
    if (ctx.profile) {
        ctx.prof_evict_scan_us.fetch_add((moe_profile_now_ns() - prof_scan_t0) / 1000,
                std::memory_order_relaxed);
        ctx.prof_evict_candidate_us.fetch_add(prof_candidate_us, std::memory_order_relaxed);
    }

    ctx.evict_scan_calls.fetch_add(1, std::memory_order_relaxed);
    ctx.evict_scan_entries.fetch_add(scan_entries, std::memory_order_relaxed);
    ctx.evict_scan_unique.fetch_add(scan_unique, std::memory_order_relaxed);
    ctx.evict_scan_duplicates.fetch_add(scan_duplicates, std::memory_order_relaxed);
    ctx.evict_scan_absolute.fetch_add(scan_absolute, std::memory_order_relaxed);
    ctx.evict_scan_temporary.fetch_add(scan_temporary, std::memory_order_relaxed);
    ctx.evict_scan_normal.fetch_add(scan_normal, std::memory_order_relaxed);

    auto release_candidate = [&](const victim_candidate & candidate) {
        const uint64_t release_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        const size_t released = moe_release_group_locked(ctx, candidate.layer, candidate.expert);
        if (released == 0) {
            return (size_t) 0;
        }
        if (ctx.admission_floor_layer == candidate.layer &&
                ctx.admission_floor_expert == candidate.expert) {
            ctx.admission_floor_valid = false;
        }
        if (candidate.belady) {
            ctx.belady_evictions.fetch_add(1, std::memory_order_relaxed);
        } else {
            ctx.lru_fallback_evictions.fetch_add(1, std::memory_order_relaxed);
        }
        const char * evict_reason = "unknown";
        uint8_t evict_reason_code = MOE_EVICT_REASON_UNKNOWN;
        switch (candidate.reason) {
            case victim_candidate::SPEC_UNUSED:
                evict_reason = "spec_unused";
                evict_reason_code = MOE_EVICT_REASON_SPEC_UNUSED;
                ctx.eam_evict_spec_unused.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::LOW_SCORE:
                evict_reason = "low_score";
                evict_reason_code = MOE_EVICT_REASON_LOW_SCORE;
                ctx.eam_evict_low_score.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::FAR_FUTURE:
                evict_reason = "far_future";
                evict_reason_code = MOE_EVICT_REASON_FAR_FUTURE;
                ctx.eam_evict_far_future.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::LAYER_WINDOW:
                evict_reason = "layer_window";
                evict_reason_code = MOE_EVICT_REASON_LAYER_WINDOW;
                ctx.layer_reserve_evictions.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::REUSE_LOW:
                evict_reason = "reuse_low";
                evict_reason_code = MOE_EVICT_REASON_REUSE_LOW;
                ctx.reuse_evict_low.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::EAM_REPLACE:
                evict_reason = "eam_replace";
                evict_reason_code = MOE_EVICT_REASON_EAM_REPLACE;
                ctx.eam_evict_low_score.fetch_add(1, std::memory_order_relaxed);
                break;
            case victim_candidate::RELAXED:
                evict_reason = "relaxed";
                evict_reason_code = MOE_EVICT_REASON_RELAXED;
                if (layer_reserve_needed) {
                    ctx.layer_reserve_relaxed.fetch_add(1, std::memory_order_relaxed);
                }
                ctx.eam_evict_lru_relaxed.fetch_add(1, std::memory_order_relaxed);
                break;
        }
        moe_group_get(ctx, candidate.layer, candidate.expert).last_evicted_reason = evict_reason_code;
        moe_evict_trace_write_locked(ctx, "evict", candidate.layer, candidate.expert,
                evict_reason, 0, released, candidate.score);
        if (ctx.profile) {
            prof_release_us += (moe_profile_now_ns() - release_t0) / 1000;
        }
        return released;
    };

    auto write_olecar_recommendations = [&](const std::vector<victim_candidate> & selected) {
        if (ctx.olecar_trace == nullptr) {
            return;
        }
        const uint64_t trace_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        const uint64_t decision_id = ctx.next_olecar_decision_id++;
        const int bucket = moe_olecar_budget_bucket(ctx);
        const double weight_sum = moe_olecar_weight_sum_locked(ctx, bucket);
        std::unordered_map<uint64_t, std::array<double, MOE_OLECAR_FAMILY_COUNT>> family_support;
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has || policy.id < 0 || policy.id >= MOE_OLECAR_POLICY_COUNT) {
                continue;
            }
            const int family = moe_olecar_policy_family(policy.id);
            if (family < 0 || family >= MOE_OLECAR_FAMILY_COUNT) {
                continue;
            }
            const uint64_t key = moe_group_key(policy.candidate.layer, policy.candidate.expert);
            const double p = weight_sum > 0.0 ?
                ctx.olecar_weights[(size_t) bucket][(size_t) policy.id] / weight_sum : 0.0;
            auto & per_family = family_support[key];
            per_family[(size_t) family] = std::max(per_family[(size_t) family], p);
        }
        std::unordered_map<uint64_t, double> support_prob;
        for (const auto & kv : family_support) {
            double s = 0.0;
            for (double p : kv.second) {
                s += p;
            }
            support_prob[kv.first] = s;
        }
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has) {
                continue;
            }
            const victim_candidate & c = policy.candidate;
            bool selected_by_current = false;
            for (const victim_candidate & s : selected) {
                if (s.layer == c.layer && s.expert == c.expert) {
                    selected_by_current = true;
                    break;
                }
            }
            moe_olecar_record record;
            record.decision_id = decision_id;
            record.policy_id = policy.id;
            record.bucket_id = bucket;
            record.policy = policy.name;
            record.layer = c.layer;
            record.expert = c.expert;
            record.selected = selected_by_current;
            if (policy.id >= 0 && policy.id < MOE_OLECAR_POLICY_COUNT &&
                    bucket >= 0 && bucket < MOE_OLECAR_BUCKET_COUNT) {
                record.weight_before =
                    ctx.olecar_weights[(size_t) bucket][(size_t) policy.id];
                record.weight_after = record.weight_before;
                record.policy_prob = weight_sum > 0.0 ? record.weight_before / weight_sum : 0.0;
            }
            record.support_prob = support_prob[moe_group_key(c.layer, c.expert)];
            record.update_count = ctx.olecar_updates[(size_t) bucket];
            record.final_score = c.score;
            record.lru_score = c.lru_score;
            record.recency_score = c.recency_score;
            record.cache_score = c.cache_policy_score;
            record.bad_reload_score = c.bad_reload_policy_score;
            record.next_use_score = c.next_use_score;
            record.layer_score = c.layer_score;
            record.reuse_score = c.reuse_score;
            record.cct_score = c.cct_score;
            moe_olecar_begin_locked(ctx, record);
            if (selected_by_current) {
                record.id = 0;
                record.kind = "exp4";
                record.exp4_update = true;
                moe_olecar_begin_locked(ctx, record);
            }
        }
        if (ctx.profile) {
            prof_trace_us += (moe_profile_now_ns() - trace_t0) / 1000;
        }
    };

    auto finish_profile = [&]() {
        if (!ctx.profile) {
            return;
        }
        ctx.prof_evict_sort_us.fetch_add(prof_sort_us, std::memory_order_relaxed);
        ctx.prof_evict_online_us.fetch_add(prof_online_us, std::memory_order_relaxed);
        ctx.prof_evict_admission_us.fetch_add(prof_admission_us, std::memory_order_relaxed);
        ctx.prof_evict_trace_us.fetch_add(prof_trace_us, std::memory_order_relaxed);
        ctx.prof_evict_release_us.fetch_add(prof_release_us, std::memory_order_relaxed);
        ctx.prof_evict_resident_queries.fetch_add(prof_resident_queries, std::memory_order_relaxed);
        ctx.prof_victim_select_us.fetch_add(
                (moe_profile_now_ns() - prof_t0) / 1000,
                std::memory_order_relaxed);
    };

    auto olecar_online_explore = [&]() {
        const double epsilon = std::max(0.0, std::min(1.0,
                moe_env_f64("LLAMA_LAZY_MOE_OLECAR_EPSILON", 0.0)));
        if (epsilon <= 0.0) {
            return false;
        }
        uint64_t x = ctx.next_olecar_decision_id * 0x9e3779b97f4a7c15ull;
        x ^= ctx.profile_token_epoch + 0xbf58476d1ce4e5b9ull + (x << 6) + (x >> 2);
        x ^= ctx.exec_epoch + 0x94d049bb133111ebull + (x << 6) + (x >> 2);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        x ^= x >> 33;
        const double u = (double) (x >> 11) * (1.0 / 9007199254740992.0);
        return u < epsilon;
    };

    auto olecar_online_support = [&]() {
        std::unordered_map<uint64_t, double> support;
        if (!olecar_online || explicit_force_policy || olecar_online_explore()) {
            return support;
        }
        const int bucket = moe_olecar_budget_bucket(ctx);
        const double weight_sum = moe_olecar_weight_sum_locked(ctx, bucket);
        if (weight_sum <= 0.0) {
            return support;
        }
        std::unordered_map<uint64_t, std::array<double, MOE_OLECAR_FAMILY_COUNT>> family_support;
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has || policy.id < 0 || policy.id >= MOE_OLECAR_POLICY_COUNT ||
                    policy.candidate.layer < 0 || policy.candidate.expert < 0) {
                continue;
            }
            const int family = moe_olecar_policy_family(policy.id);
            if (family < 0 || family >= MOE_OLECAR_FAMILY_COUNT) {
                continue;
            }
            const double p =
                ctx.olecar_weights[(size_t) bucket][(size_t) policy.id] / weight_sum;
            auto & per_family =
                family_support[moe_group_key(policy.candidate.layer, policy.candidate.expert)];
            per_family[(size_t) family] = std::max(per_family[(size_t) family], p);
        }
        for (const auto & kv : family_support) {
            double s = 0.0;
            for (double p : kv.second) {
                s += p;
            }
            support[kv.first] = s;
        }
        return support;
    };

    auto reorder_olecar_online = [&](std::vector<victim_candidate> & candidates) {
        if (candidates.size() < 2) {
            return;
        }
        const std::unordered_map<uint64_t, double> support = olecar_online_support();
        if (support.empty()) {
            return;
        }
        auto score_of = [&](const victim_candidate & c) {
            auto it = support.find(moe_group_key(c.layer, c.expert));
            return it == support.end() ? 0.0 : it->second;
        };
        const int topk_env = moe_env_i32("LLAMA_LAZY_MOE_OLECAR_ONLINE_TOPK", 32);
        const size_t topk = topk_env <= 0 ? candidates.size() :
            std::min(candidates.size(), (size_t) topk_env);
        std::stable_sort(candidates.begin(), candidates.begin() + topk,
                [&](const victim_candidate & a, const victim_candidate & b) {
                    const double sa = score_of(a);
                    const double sb = score_of(b);
                    if (sa != sb) {
                        return sa > sb;
                    }
                    return better_candidate(a, b);
                });
    };

    auto choose_olecar_online_best = [&](const victim_candidate & fallback) {
        const std::unordered_map<uint64_t, double> support = olecar_online_support();
        if (support.empty()) {
            return fallback;
        }
        victim_candidate best_online = fallback;
        double best_support = -1.0;
        for (const olecar_policy_best & policy : olecar_best) {
            if (!policy.has || policy.candidate.layer < 0 || policy.candidate.expert < 0) {
                continue;
            }
            const victim_candidate & c = policy.candidate;
            if (moe_group_resident_bytes(ctx, c.layer, c.expert) == 0) {
                continue;
            }
            auto it = support.find(moe_group_key(c.layer, c.expert));
            const double s = it == support.end() ? 0.0 : it->second;
            if (s <= 0.0) {
                continue;
            }
            const double max_score_drop =
                moe_env_f64("LLAMA_LAZY_MOE_OLECAR_ONLINE_MAX_SCORE_DROP", 1.0e300);
            if (c.score < fallback.score - max_score_drop) {
                continue;
            }
            if (s > best_support ||
                    (s == best_support && better_candidate(c, best_online))) {
                best_support = s;
                best_online = c;
            }
        }
        return best_support > 0.0 ? best_online : fallback;
    };

    if (target_release_bytes > 0) {
        uint64_t phase_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        std::sort(normal_candidates.begin(), normal_candidates.end(), better_candidate);
        std::sort(temporary_candidates.begin(), temporary_candidates.end(), better_candidate);
        if (ctx.profile) {
            prof_sort_us += (moe_profile_now_ns() - phase_t0) / 1000;
            phase_t0 = moe_profile_now_ns();
        }
        reorder_olecar_online(normal_candidates);
        reorder_olecar_online(temporary_candidates);
        if (ctx.profile) {
            prof_online_us += (moe_profile_now_ns() - phase_t0) / 1000;
        }

        if (admission_batch != nullptr && batch_results != nullptr) {
            const uint64_t admission_t0 = ctx.profile ? moe_profile_now_ns() : 0;
            batch_results->assign(admission_batch->size(), {});
            std::vector<size_t> order(admission_batch->size());
            for (size_t i = 0; i < order.size(); ++i) {
                order[i] = i;
            }
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                const moe_demand_admission_request & ra = (*admission_batch)[a];
                const moe_demand_admission_request & rb = (*admission_batch)[b];
                if (ra.incoming_value != rb.incoming_value) {
                    return ra.incoming_value > rb.incoming_value;
                }
                if (ra.layer != rb.layer) {
                    return ra.layer < rb.layer;
                }
                return ra.expert < rb.expert;
            });

            std::vector<victim_candidate> victim_pool;
            victim_pool.reserve(normal_candidates.size() + temporary_candidates.size());
            victim_pool.insert(victim_pool.end(), normal_candidates.begin(), normal_candidates.end());
            victim_pool.insert(victim_pool.end(), temporary_candidates.begin(), temporary_candidates.end());

            std::vector<victim_candidate> selected;
            selected.reserve(victim_pool.size());
            size_t victim_cursor = 0;
            size_t cache_credit = batch_free_bytes;
            size_t staging_credit = batch_staging_bytes;

            for (size_t request_index : order) {
                const moe_demand_admission_request & request =
                    (*admission_batch)[request_index];
                moe_demand_admission_result & result =
                    (*batch_results)[request_index];
                if (request.target_bytes == 0) {
                    continue;
                }
                if (cache_credit >= request.target_bytes) {
                    cache_credit -= request.target_bytes;
                    continue;
                }

                const size_t required = request.target_bytes - cache_credit;
                size_t peek_cursor = victim_cursor;
                size_t valued_bytes = 0;
                size_t full_victim_bytes = 0;
                double weighted_value = 0.0;
                while (peek_cursor < victim_pool.size() && valued_bytes < required) {
                    const victim_candidate & candidate = victim_pool[peek_cursor++];
                    ++prof_resident_queries;
                    const size_t resident =
                        moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                    if (resident == 0) {
                        continue;
                    }
                    moe_group_state & victim =
                        moe_group_get(ctx, candidate.layer, candidate.expert);
                    const double cache_score =
                        moe_group_cache_score(ctx, victim, resident);
                    const double value = moe_prefetch_victim_value_locked(
                            ctx, victim, resident, cache_score);
                    const size_t take = std::min(resident, required - valued_bytes);
                    weighted_value += value * (double) take / (double) required;
                    valued_bytes += take;
                    full_victim_bytes += resident;
                    if (!result.has_victim) {
                        result.has_victim = true;
                        result.victim_layer = candidate.layer;
                        result.victim_expert = candidate.expert;
                    }
                    result.victim_groups++;
                    result.victim_keys.push_back(
                            moe_group_key(candidate.layer, candidate.expert));
                }
                result.evaluated = true;
                result.victim_bytes = full_victim_bytes;
                result.victim_value = weighted_value;

                const bool staging_available =
                    staging_credit >= request.target_bytes;
                result.bypass = staging_available &&
                    (!result.has_victim ||
                     request.incoming_value <= result.victim_value + request.margin);
                if (result.bypass) {
                    staging_credit -= request.target_bytes;
                    continue;
                }

                cache_credit = 0;
                while (victim_cursor < peek_cursor) {
                    const victim_candidate & candidate = victim_pool[victim_cursor++];
                    ++prof_resident_queries;
                    const size_t resident =
                        moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                    if (resident == 0) {
                        continue;
                    }
                    selected.push_back(candidate);
                    cache_credit += resident;
                }
                cache_credit = cache_credit > required ?
                    cache_credit - required : 0;
            }
            if (ctx.profile) {
                prof_admission_us += (moe_profile_now_ns() - admission_t0) / 1000;
            }

            size_t released_total = 0;
            write_olecar_recommendations(selected);
            for (const victim_candidate & candidate : selected) {
                const size_t released = release_candidate(candidate);
                if (released > 0) {
                    released_total += released;
                    if (candidate.reason == victim_candidate::RELAXED) {
                        ctx.evict_scan_selected_temporary.fetch_add(
                                1, std::memory_order_relaxed);
                    }
                }
            }
            finish_profile();
            return released_total > 0;
        }

        std::vector<victim_candidate> selected;
        const uint64_t admission_t0 = ctx.profile ? moe_profile_now_ns() : 0;
        selected.reserve(normal_candidates.size() + temporary_candidates.size());
        size_t selected_bytes = 0;
        auto select_candidates = [&](const std::vector<victim_candidate> & candidates) {
            for (const victim_candidate & candidate : candidates) {
                ++prof_resident_queries;
                const size_t bytes =
                    moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                if (bytes == 0) {
                    continue;
                }
                selected.push_back(candidate);
                selected_bytes += bytes;
                if (selected_bytes >= target_release_bytes) {
                    break;
                }
            }
        };
        select_candidates(normal_candidates);
        if (selected_bytes < target_release_bytes) {
            select_candidates(temporary_candidates);
        }

        if (admission_request != nullptr && admission_result != nullptr) {
            admission_result->evaluated = true;
            admission_result->victim_groups = (int) selected.size();
            admission_result->victim_bytes = selected_bytes;
            for (const victim_candidate & candidate : selected) {
                admission_result->victim_keys.push_back(
                        moe_group_key(candidate.layer, candidate.expert));
            }
            if (!selected.empty()) {
                admission_result->has_victim = true;
                admission_result->victim_layer = selected.front().layer;
                admission_result->victim_expert = selected.front().expert;
                size_t valued_bytes = 0;
                double weighted_value = 0.0;
                for (const victim_candidate & candidate : selected) {
                    ++prof_resident_queries;
                    const size_t resident =
                        moe_group_resident_bytes(ctx, candidate.layer, candidate.expert);
                    if (resident == 0 || valued_bytes >= target_release_bytes) {
                        continue;
                    }
                    moe_group_state & victim =
                        moe_group_get(ctx, candidate.layer, candidate.expert);
                    const double cache_score =
                        moe_group_cache_score(ctx, victim, resident);
                    const double value = moe_prefetch_victim_value_locked(
                            ctx, victim, resident, cache_score);
                    const size_t take =
                        std::min(resident, target_release_bytes - valued_bytes);
                    weighted_value += value * (double) take /
                        (double) target_release_bytes;
                    valued_bytes += take;
                }
                admission_result->victim_value = weighted_value;
            }
            admission_result->bypass =
                !admission_result->has_victim ||
                admission_request->incoming_value <=
                    admission_result->victim_value + admission_request->margin;
            if (admission_result->bypass) {
                if (!admission_result->has_victim) {
                    ctx.evict_scan_no_victim.fetch_add(1, std::memory_order_relaxed);
                }
                finish_profile();
                return false;
            }
        }
        if (ctx.profile) {
            prof_admission_us += (moe_profile_now_ns() - admission_t0) / 1000;
        }

        size_t released_total = 0;
        write_olecar_recommendations(selected);
        for (const victim_candidate & candidate : selected) {
            const size_t released = release_candidate(candidate);
            if (released > 0) {
                released_total += released;
                if (candidate.reason == victim_candidate::RELAXED) {
                    ctx.evict_scan_selected_temporary.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
        if (released_total == 0) {
            ctx.evict_scan_no_victim.fetch_add(1, std::memory_order_relaxed);
        }
        finish_profile();
        return released_total > 0;
    }

    if (best.layer < 0 && temporary_best.layer >= 0) {
        best = temporary_best;
        ctx.evict_scan_selected_temporary.fetch_add(1, std::memory_order_relaxed);
    }
    if (best.layer < 0) {
        ctx.evict_scan_no_victim.fetch_add(1, std::memory_order_relaxed);
        finish_profile();
        return false;
    }

    best = choose_olecar_online_best(best);
    write_olecar_recommendations(std::vector<victim_candidate>{best});
    const bool released = release_candidate(best) > 0;
    finish_profile();
    return released;
}

// Demand admission controls cache replacement, not whether the requested group
// can execute. It consumes the exact victim set selected by moe_evict_lru()'s
// single scan. A rejected group temporarily occupies staging and is detached
// after its layer completes.
static double moe_demand_incoming_value_locked(
        llama_moe_buffer_context & ctx,
        moe_group_state &          g,
        size_t                     target_bytes,
        int                        rank) {
    const double cache_score = moe_group_cache_score(ctx, g, target_bytes);
    double value =
        moe_prefetch_victim_value_locked(ctx, g, target_bytes, cache_score) +
        moe_env_f64("LLAMA_LAZY_MOE_DEMAND_ADMIT_CURRENT_W", 160.0);
    if (rank == 0) {
        value += moe_env_f64("LLAMA_LAZY_MOE_DEMAND_ADMIT_RANK0_W", 40.0);
    }
    if (moe_env_flag("LLAMA_LAZY_MOE_ADMISSION_FEEDBACK", 0) &&
            g.admission_feedback_samples > 0) {
        value += g.admission_feedback_ema *
            moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_FEEDBACK_WEIGHT", 20.0);
        ctx.admission_feedback_score_candidates.fetch_add(1, std::memory_order_relaxed);
    }
    if (moe_env_flag("LLAMA_LAZY_MOE_ADMISSION_REGRET_FEEDBACK", 0) &&
            g.admission_regret_samples > 0) {
        value += g.admission_regret_ema *
            moe_env_f64("LLAMA_LAZY_MOE_ADMISSION_REGRET_WEIGHT", 30.0);
        ctx.admission_regret_score_candidates.fetch_add(1, std::memory_order_relaxed);
    }
    return value;
}

static bool moe_demand_admission_bypass_locked(
        llama_moe_buffer_context & ctx,
        moe_managed &              m,
        int                        expert,
        int                        rank,
        bool                       real_touch) {
    if (!moe_demand_admission_enabled() || !ctx.params.dynamic_bits_real ||
            ctx.params.budget_bytes == 0 ||
            m.layer < 0 || expert < 0 || expert >= m.n_expert) {
        return false;
    }

    moe_group_state & g = moe_group_get(ctx, m.layer, expert);
    moe_admission_outcome_note_reuse_locked(ctx, g);
    if (g.demand_admission_decided &&
            g.demand_admission_token == ctx.profile_token_epoch) {
        return g.demand_admission_bypass;
    }
    if (!real_touch) {
        return false;
    }

    g.demand_admission_token = ctx.profile_token_epoch;
    g.demand_admission_decided = true;
    g.demand_admission_bypass = false;

    const size_t target_bytes =
        moe_demand_group_missing_bytes_locked(ctx, m.layer, expert, rank);
    const size_t cache_bytes = ctx.resident_bytes -
        std::min(ctx.resident_bytes, ctx.demand_admission_staging_bytes);
    if (target_bytes == 0 ||
            cache_bytes + target_bytes <= ctx.params.budget_bytes) {
        return false;
    }

    ctx.demand_admission_runs.fetch_add(1, std::memory_order_relaxed);
    const size_t staging_limit = moe_env_mib_bytes(
            "LLAMA_LAZY_MOE_DEMAND_ADMISSION_STAGING_MB", 64);
    if (staging_limit == 0 ||
            ctx.demand_admission_staging_bytes +
                ctx.demand_admission_staging_reserved_bytes +
                target_bytes > staging_limit) {
        ctx.demand_admission_admit.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    const double candidate_value =
        moe_demand_incoming_value_locked(ctx, g, target_bytes, rank);

    moe_demand_admission_request request;
    request.layer = m.layer;
    request.expert = expert;
    request.target_bytes = target_bytes;
    request.incoming_value = candidate_value;
    request.margin = moe_env_f64("LLAMA_LAZY_MOE_DEMAND_ADMIT_MARGIN", 0.0);
    moe_demand_admission_result result;
    const size_t release_needed =
        cache_bytes + target_bytes - ctx.params.budget_bytes;
    {
        moe_evict_context_guard evict_ctx(ctx, m, true);
        moe_evict_lru(ctx, release_needed, &request, &result);
    }

    ctx.demand_admission_candidate_x1000.fetch_add(
            (uint64_t) std::max(0.0, candidate_value * 1000.0),
            std::memory_order_relaxed);
    ctx.demand_admission_victim_x1000.fetch_add(
            (uint64_t) std::max(0.0, result.victim_value * 1000.0),
            std::memory_order_relaxed);
    if (result.bypass) {
        g.demand_admission_bypass = true;
        ctx.demand_admission_bypass.fetch_add(1, std::memory_order_relaxed);
        if (!result.has_victim) {
            ctx.demand_admission_no_victim.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        ctx.demand_admission_admit.fetch_add(1, std::memory_order_relaxed);
    }
    moe_admission_outcome_begin_locked(
            ctx, g, result.bypass, candidate_value, result);
    return result.bypass;
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

static void moe_group_fill_mwq_siblings(
        llama_moe_buffer_context & ctx,
        moe_managed &              src,
        int                        e,
        int                        rank);
static void moe_group_fill_native_siblings(
        llama_moe_buffer_context & ctx,
        moe_managed &              src,
        int                        e,
        int                        rank);

static void moe_stream_mwq_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank,
        bool * out_entry_missing, bool * out_read_failed, bool group_fill, bool demand_async) {
    if (e < 0 || e >= m.n_expert) return;

    if (moe_native_entry_available(ctx, m, target_bits)) {
        moe_stream_slice(ctx, m, e, touch, target_bits, rank);
        return;
    }

    std::unique_lock<std::mutex> lk(ctx.mtx);
    const char arrival_state = m.resident[e];
    const int arrival_loaded_bits = moe_mwq_resolve_loaded_bits(m, e, target_bits);
    const int arrival_bits = arrival_loaded_bits > 0 ? arrival_loaded_bits : m.resident_bits[e];
    const bool arrival_mwq = e < (int) m.resident_mwq.size() && m.resident_mwq[e];
    const bool arrival_queued = e < (int) m.queued.size() && m.queued[e];
    const int arrival_queued_bits = e < (int) m.queued_bits.size() ? m.queued_bits[e] : 0;
    const bool arrival_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
    auto trace_compute = [&](const char * action, uint64_t wait_ns = 0) {
        if (touch) {
            moe_cache_trace_write_locked(ctx, m, e, rank, target_bits, action,
                    arrival_state, arrival_bits, arrival_mwq, arrival_queued,
                    arrival_queued_bits, arrival_prefetched, wait_ns);
        }
    };
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
                const bool was_demand_async =
                    e < (int) m.demand_async_loaded.size() && m.demand_async_loaded[e];
                const bool was_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
                if (was_demand_async) {
                    m.demand_async_loaded[e] = false;
                    if (e < (int) m.prefetched.size()) {
                        m.prefetched[e] = false;
                    }
                    moe_group_note_cache_touch_locked(ctx, m, e, false, true);
                    ctx.prof_cache_miss.fetch_add(1, std::memory_order_relaxed);
                } else {
                    ctx.hits.fetch_add(1, std::memory_order_relaxed);
                    moe_group_note_cache_touch_locked(ctx, m, e, was_prefetched, false);
                }
                if (!was_demand_async && was_prefetched) {
                    ctx.prof_prefetch_hit.fetch_add(1, std::memory_order_relaxed);
                    m.prefetched[e] = false;
                } else if (!was_demand_async) {
                    ctx.prof_cache_hit.fetch_add(1, std::memory_order_relaxed);
                }
                if (e < (int) m.resident_touched.size()) {
                    m.resident_touched[e] = true;
                }
                if (loaded_bits != target_bits) {
                    ctx.mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
                }
                trace_compute(was_demand_async ? "demand_async_miss_mwq" :
                        (was_prefetched ? "prefetch_hit_mwq" :
                         (loaded_bits != target_bits ? "compat_hit_mwq" : "cache_hit_mwq")));
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
                    const bool was_demand_async =
                        e < (int) m.demand_async_loaded.size() && m.demand_async_loaded[e];
                    const bool was_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
                    if (was_demand_async) {
                        m.demand_async_loaded[e] = false;
                        if (e < (int) m.prefetched.size()) {
                            m.prefetched[e] = false;
                        }
                        moe_group_note_cache_touch_locked(ctx, m, e, false, true);
                        ctx.prof_cache_miss.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        ctx.hits.fetch_add(1, std::memory_order_relaxed);
                        moe_group_note_cache_touch_locked(ctx, m, e, was_prefetched, false);
                    }
                    if (!was_demand_async && was_prefetched) {
                        ctx.prof_prefetch_hit.fetch_add(1, std::memory_order_relaxed);
                        m.prefetched[e] = false;
                    } else if (!was_demand_async) {
                        ctx.prof_cache_hit.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (e < (int) m.resident_touched.size()) {
                        m.resident_touched[e] = true;
                    }
                    if (m.resident_bits[e] != target_bits) {
                        ctx.mwq_compat_hits.fetch_add(1, std::memory_order_relaxed);
                    }
                    trace_compute(was_demand_async ? "demand_async_miss_native" :
                            (was_prefetched ? "prefetch_hit_native" :
                             (m.resident_bits[e] != target_bits ? "compat_hit_native" : "cache_hit_native")));
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
                    const bool was_demand_async =
                        e < (int) m.demand_async_loaded.size() && m.demand_async_loaded[e];
                    const bool was_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
                    if (was_demand_async) {
                        m.demand_async_loaded[e] = false;
                        if (e < (int) m.prefetched.size()) {
                            m.prefetched[e] = false;
                        }
                        moe_group_note_cache_touch_locked(ctx, m, e, false, true);
                        ctx.prof_cache_miss.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        ctx.hits.fetch_add(1, std::memory_order_relaxed);
                        moe_group_note_cache_touch_locked(ctx, m, e, was_prefetched, false);
                    }
                    if (!was_demand_async && was_prefetched) {
                        ctx.prof_prefetch_hit.fetch_add(1, std::memory_order_relaxed);
                        m.prefetched[e] = false;
                    } else if (!was_demand_async) {
                        ctx.prof_cache_hit.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (e < (int) m.resident_touched.size()) {
                        m.resident_touched[e] = true;
                    }
                    trace_compute(was_demand_async ? "demand_async_miss_mwq_exact" :
                            (was_prefetched ? "prefetch_hit_mwq_exact" : "cache_hit_mwq_exact"));
                }
                return;
            }
            trace_compute("resident_incompatible_drop");
            moe_drop_group_incompatible_locked(ctx, m.layer, e);
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
            trace_compute("wait_inflight", (uint64_t) wait_ns);
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
        trace_compute("sidecar_entry_missing_fallback");
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
    const bool admission_bypass =
        moe_demand_admission_bypass_locked(ctx, m, e, rank, touch);
    if (ctx.params.budget_bytes > 0 && !admission_bypass) {
        const uint64_t evictions_before = ctx.group_evictions.load(std::memory_order_relaxed);
        moe_evict_context_guard evict_ctx(ctx, m, touch);
        while (ctx.resident_bytes - std::min(ctx.resident_bytes, ctx.demand_admission_staging_bytes) +
                        (size_t) ent->encoded_size > ctx.params.budget_bytes &&
                moe_evict_lru(ctx)) {}
        if (demand_async) {
            const uint64_t evictions_after = ctx.group_evictions.load(std::memory_order_relaxed);
            if (evictions_after > evictions_before) {
                ctx.demand_async_worker_evictions.fetch_add(
                        evictions_after - evictions_before, std::memory_order_relaxed);
            }
        }
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
    if (admission_bypass) {
        if (demand_async) {
            moe_group_state & g = moe_group_get(ctx, m.layer, e);
            const size_t transferred = std::min(
                    g.admission_staging_reserved_bytes,
                    (size_t) ent->encoded_size);
            g.admission_staging_reserved_bytes -= transferred;
            ctx.demand_admission_staging_reserved_bytes -= std::min(
                    ctx.demand_admission_staging_reserved_bytes, transferred);
        }
        ctx.demand_admission_staging_bytes += (size_t) ent->encoded_size;
        const uint64_t staging = ctx.demand_admission_staging_bytes;
        uint64_t peak = ctx.demand_admission_staging_peak.load(std::memory_order_relaxed);
        while (peak < staging &&
                !ctx.demand_admission_staging_peak.compare_exchange_weak(
                        peak, staging, std::memory_order_relaxed)) {
        }
    }

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
        moe_group_resident_add_locked(ctx, m, e, (size_t) ent->encoded_size, m.resident_bits[e]);
        ctx.lru.push_front({&m, e});
        m.lru_pos[e] = ctx.lru.begin();
        moe_group_lru_touch_locked(ctx, m.layer, e);
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
        trace_compute(touch ? "miss_load_mwq" : "prefetch_load_mwq");
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
        if (admission_bypass) {
            ctx.demand_admission_staging_bytes -= std::min(
                    ctx.demand_admission_staging_bytes, m.resident_size[e]);
        }
        m.resident_size[e] = 0;
        ctx.sidecar_misses.fetch_add(1, std::memory_order_relaxed);
        if (out_read_failed) *out_read_failed = true;
        trace_compute("miss_read_failed_mwq");
        const uint64_t fail_no = ctx.mwq_stream_fail_total.fetch_add(1, std::memory_order_relaxed);
        if (ctx.params.debug_log && (fail_no % MOE_STREAM_FAIL_LOG_EVERY) == 0) {
            std::fprintf(stderr, "llama_moe_buffer: MWQ stream failed %s expert %d bits %d (occurrence #%llu)\n",
                    m.name.c_str(), e, target_bits, (unsigned long long) fail_no + 1);
        }
    }
    ctx.cv_done.notify_all();
    const bool fill_siblings = ok && group_fill && m.layer >= 0 && e >= 0;
    lk.unlock();
    if (fill_siblings) {
        moe_group_fill_mwq_siblings(ctx, m, e, rank);
    }
}

// Ensure expert slice e of m is resident. Coordinates the synchronous callback
// and the async worker via a per-slot state machine: the pread happens outside
// the residency mutex; a second thread requesting the same slice waits for the
// in-flight read to complete instead of issuing a duplicate read.
static void moe_stream_slice(llama_moe_buffer_context & ctx, moe_managed & m, int e, bool touch, int target_bits, int rank,
        bool group_fill, bool demand_async) {
    if (e < 0 || e >= m.n_expert) return;

    std::unique_lock<std::mutex> lk(ctx.mtx);
    const char arrival_state = m.resident[e];
    const int arrival_bits = m.resident_bits[e];
    const bool arrival_mwq = e < (int) m.resident_mwq.size() && m.resident_mwq[e];
    const bool arrival_queued = e < (int) m.queued.size() && m.queued[e];
    const int arrival_queued_bits = e < (int) m.queued_bits.size() ? m.queued_bits[e] : 0;
    const bool arrival_prefetched = e < (int) m.prefetched.size() && m.prefetched[e];
    auto trace_compute = [&](const char * action, uint64_t wait_ns = 0) {
        if (touch) {
            moe_cache_trace_write_locked(ctx, m, e, rank, target_bits, action,
                    arrival_state, arrival_bits, arrival_mwq, arrival_queued,
                    arrival_queued_bits, arrival_prefetched, wait_ns);
        }
    };
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
                    trace_compute(was_prefetched ? "prefetch_hit_native_full" : "cache_hit_native_full");
                }
                return;
            }
            trace_compute("resident_incompatible_drop_native");
            moe_drop_group_incompatible_locked(ctx, m.layer, e);
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
            trace_compute("wait_inflight_native", (uint64_t) wait_ns);
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
        const uint64_t evictions_before = ctx.group_evictions.load(std::memory_order_relaxed);
        moe_evict_context_guard evict_ctx(ctx, m, touch);
        while (ctx.resident_bytes + m.stride > ctx.params.budget_bytes && moe_evict_lru(ctx)) {}
        if (demand_async) {
            const uint64_t evictions_after = ctx.group_evictions.load(std::memory_order_relaxed);
            if (evictions_after > evictions_before) {
                ctx.demand_async_worker_evictions.fetch_add(
                        evictions_after - evictions_before, std::memory_order_relaxed);
            }
        }
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
        moe_group_resident_add_locked(ctx, m, e, m.stride, m.resident_bits[e]);
        ctx.lru.push_front({&m, e});
        m.lru_pos[e] = ctx.lru.begin();
        moe_group_lru_touch_locked(ctx, m.layer, e);
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
        trace_compute(touch ? "miss_load_native_full" : "prefetch_load_native_full");
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
        trace_compute("miss_read_failed_native");
    }
    ctx.cv_done.notify_all();
    const bool fill_siblings = ok && group_fill && m.layer >= 0 && e >= 0;
    lk.unlock();
    if (fill_siblings) {
        moe_group_fill_native_siblings(ctx, m, e, rank);
    }
}

static bool moe_group_fill_enabled() {
    return moe_env_flag("LLAMA_LAZY_MOE_GROUP_FILL", 1);
}

static void moe_group_fill_mwq_siblings(
        llama_moe_buffer_context & ctx,
        moe_managed &              src,
        int                        e,
        int                        rank) {
    if (!moe_group_fill_enabled() || src.layer < 0 || e < 0) {
        return;
    }
    auto it = ctx.by_layer.find(src.layer);
    if (it == ctx.by_layer.end()) {
        return;
    }
    for (moe_managed * m : it->second) {
        if (m == nullptr || m == &src || e >= m->n_expert) {
            continue;
        }
        const int bits = moe_target_bits_for_rank(ctx, *m, e, rank);
        bool need = false;
        {
            std::lock_guard<std::mutex> lk(ctx.mtx);
            need = moe_mwq_resolve_loaded_bits(*m, e, bits) <= 0 &&
                !moe_native_resident_compatible(ctx, *m, e, bits);
        }
        if (!need) {
            continue;
        }
        ctx.group_fill_sibling_loads.fetch_add(1, std::memory_order_relaxed);
        moe_stream_mwq_slice(ctx, *m, e, false, bits, rank, nullptr, nullptr, false);
    }
}

static void moe_group_fill_native_siblings(
        llama_moe_buffer_context & ctx,
        moe_managed &              src,
        int                        e,
        int                        rank) {
    if (!moe_group_fill_enabled() || src.layer < 0 || e < 0) {
        return;
    }
    auto it = ctx.by_layer.find(src.layer);
    if (it == ctx.by_layer.end()) {
        return;
    }
    for (moe_managed * m : it->second) {
        if (m == nullptr || m == &src || e >= m->n_expert) {
            continue;
        }
        const int bits = moe_target_bits_for_rank(ctx, *m, e, rank);
        bool need = false;
        {
            std::lock_guard<std::mutex> lk(ctx.mtx);
            need = !moe_native_resident_compatible(ctx, *m, e, bits);
        }
        if (!need) {
            continue;
        }
        ctx.group_fill_sibling_loads.fetch_add(1, std::memory_order_relaxed);
        moe_stream_slice(ctx, *m, e, false, bits, rank, false);
    }
}

static const char * moe_tensor_kind_cstr(moe_tensor_kind kind) {
    switch (kind) {
        case moe_tensor_kind::gate: return "gate";
        case moe_tensor_kind::up:   return "up";
        case moe_tensor_kind::down: return "down";
        default: return "other";
    }
}

static void moe_group_prefetch_trace_locked(
        llama_moe_buffer_context & ctx,
        const char *               event,
        int                        layer,
        int                        expert,
        const moe_managed &        m,
        int                        target_bits,
        const char *               state) {
    if (ctx.group_prefetch_trace == nullptr) {
        return;
    }
    std::fprintf(ctx.group_prefetch_trace,
            "%llu\t%llu\t%s\t%d\t%d\t%s\t%d\t%s\t%.3f\t%.3f\n",
            (unsigned long long) ctx.profile_token_epoch,
            (unsigned long long) ctx.exec_epoch,
            event,
            layer,
            expert,
            moe_tensor_kind_cstr(moe_kind(m)),
            target_bits,
            state,
            ctx.resident_bytes / 1048576.0,
            ctx.params.budget_bytes / 1048576.0);
}

static int moe_group_missing_sibling_slices_locked(
        llama_moe_buffer_context & ctx,
        int                        layer,
        int                        expert,
        int                        rank,
        int                        skip_kind) {
    if (layer < 0 || expert < 0) {
        return 0;
    }
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return 0;
    }
    int missing = 0;
    for (moe_managed * m : it->second) {
        if (m == nullptr || expert >= m->n_expert) {
            continue;
        }
        if (skip_kind >= 0 && (int) moe_kind(*m) == skip_kind) {
            continue;
        }
        const int bits = moe_target_bits_for_rank(ctx, *m, expert, rank);
        const bool ready = (ctx.params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(*m, expert, bits) > 0) ||
            moe_native_resident_compatible(ctx, *m, expert, bits);
        if (!ready) {
            ++missing;
        }
    }
    return missing;
}

static void moe_prefetch_group_worker(
        llama_moe_buffer_context & ctx,
        int                        layer,
        int                        expert,
        int                        rank,
        float                      score,
        int                        skip_kind,
        bool                       is_demand,
        int                        primary_kind) {
    (void) score;
    if (layer < 0 || expert < 0) {
        return;
    }
    auto it = ctx.by_layer.find(layer);
    if (it == ctx.by_layer.end()) {
        return;
    }
    if (!is_demand) {
        ctx.group_prefetch_runs.fetch_add(1, std::memory_order_relaxed);
    }
    for (moe_managed * m : it->second) {
        if (m == nullptr || expert >= m->n_expert) {
            continue;
        }
        if (skip_kind >= 0 && (int) moe_kind(*m) == skip_kind) {
            continue;
        }
        const int bits = moe_target_bits_for_rank(ctx, *m, expert, rank);
        bool ready = false;
        {
            std::lock_guard<std::mutex> lk(ctx.mtx);
            ready = (ctx.params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(*m, expert, bits) > 0) ||
                moe_native_resident_compatible(ctx, *m, expert, bits);
            moe_group_prefetch_trace_locked(ctx, "check", layer, expert, *m, bits,
                    ready ? "ready" : "need_load");
        }
        if (ready) {
            if (!is_demand) {
                ctx.group_prefetch_hit_existing.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }

        const uint64_t before_streams = ctx.streams.load(std::memory_order_relaxed);
        const uint64_t before_bytes = ctx.bytes_read.load(std::memory_order_relaxed);
        bool entry_missing = false;
        bool read_failed = false;
        if (ctx.params.dynamic_bits_real && moe_mwq_entry(ctx, *m, expert, bits) != nullptr) {
            moe_stream_mwq_slice(ctx, *m, expert, false, bits, rank, &entry_missing, &read_failed, false, is_demand);
        } else {
            moe_stream_slice(ctx, *m, expert, false, bits, rank, false, is_demand);
        }
        const uint64_t after_streams = ctx.streams.load(std::memory_order_relaxed);
        const uint64_t after_bytes = ctx.bytes_read.load(std::memory_order_relaxed);
        if (is_demand && !entry_missing && !read_failed) {
            ctx.demand_async_slices.fetch_add(1, std::memory_order_relaxed);
        } else if (!is_demand && after_streams > before_streams) {
            ctx.group_prefetch_slices.fetch_add(after_streams - before_streams, std::memory_order_relaxed);
        }
        if (!is_demand && after_bytes > before_bytes) {
            ctx.group_prefetch_bytes.fetch_add(after_bytes - before_bytes, std::memory_order_relaxed);
        }
        if (entry_missing || read_failed) {
            if (is_demand) {
                ctx.demand_async_failed.fetch_add(1, std::memory_order_relaxed);
            } else {
                ctx.group_prefetch_fail.fetch_add(1, std::memory_order_relaxed);
            }
        }
        {
            std::lock_guard<std::mutex> lk(ctx.mtx);
            if (is_demand && !entry_missing && !read_failed &&
                    (int) moe_kind(*m) == primary_kind &&
                    expert < (int) m->demand_async_loaded.size() &&
                    ((ctx.params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(*m, expert, bits) > 0) ||
                     moe_native_resident_compatible(ctx, *m, expert, bits))) {
                m->demand_async_loaded[expert] = true;
                if (expert < (int) m->prefetched.size()) {
                    m->prefetched[expert] = false;
                }
            }
            if (!is_demand) {
                moe_group_prefetch_trace_locked(ctx,
                        entry_missing ? "entry_missing" : (read_failed ? "read_failed" : "load"),
                        layer, expert, *m, bits, after_streams > before_streams ? "loaded" : "no_stream");
            }
        }
    }
}

static bool moe_enqueue_group_prefetch_task(
        llama_moe_buffer_context & ctx,
        int                        layer,
        int                        expert,
        int                        rank,
        float                      score,
        int                        priority,
        uint64_t                   token_epoch,
        int                        skip_kind) {
    if (layer < 0 || expert < 0) {
        return false;
    }
    moe_prefetch_task task;
    task.is_group = true;
    task.layer = layer;
    task.skip_kind = skip_kind;
    task.e = expert;
    task.rank = rank;
    task.score = score;
    task.target_bits = ctx.params.base_bits;
    task.priority = priority;
    {
        std::lock_guard<std::mutex> lk(ctx.qmtx);
        const uint64_t key = moe_group_key(layer, expert);
        auto last_it = ctx.group_prefetch_last_enqueue_token.find(key);
        if (last_it != ctx.group_prefetch_last_enqueue_token.end() && last_it->second == token_epoch) {
            ctx.group_prefetch_dups.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        ctx.group_prefetch_last_enqueue_token[key] = token_epoch;
        task.seq = ctx.next_task_seq++;
        ctx.queue.push(task);
    }
    ctx.enqueued.fetch_add(1, std::memory_order_relaxed);
    ctx.group_prefetch_enqueued.fetch_add(1, std::memory_order_relaxed);
    ctx.cv_q.notify_one();
    return true;
}

static bool moe_demand_async_enabled() {
    static const bool enabled = moe_env_flag("LLAMA_LAZY_MOE_DEMAND_ASYNC", 0);
    return enabled;
}

static int moe_demand_kind_order(moe_tensor_kind kind) {
    switch (kind) {
        case moe_tensor_kind::gate: return 0;
        case moe_tensor_kind::up:   return 1;
        case moe_tensor_kind::down: return 2;
        default:                    return 3;
    }
}

static void moe_prepare_demand_groups(
        llama_moe_buffer_context & ctx,
        int                        layer,
        const ggml_tensor *        ids) {
    if (!moe_demand_async_enabled() || !ctx.params.dynamic_bits_real ||
            layer < 0 || ids == nullptr || ids->data == nullptr || ids->type != GGML_TYPE_I32) {
        return;
    }
    auto layer_it = ctx.by_layer.find(layer);
    if (layer_it == ctx.by_layer.end() || layer_it->second.empty()) {
        return;
    }

    const int n_expert = layer_it->second.front()->n_expert;
    std::vector<int> min_rank((size_t) n_expert, INT_MAX);
    for (int64_t token = 0; token < ids->ne[1]; ++token) {
        for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
            const int expert = *(const int32_t *) ((const char *) ids->data +
                    token * ids->nb[1] + rank * ids->nb[0]);
            if (expert >= 0 && expert < n_expert) {
                min_rank[(size_t) expert] = std::min(min_rank[(size_t) expert], (int) rank);
            }
        }
    }

    struct demand_submit {
        int expert = -1;
        int rank = 0;
        int primary_kind = -1;
        uint64_t generation = 0;
        size_t needed_bytes = 0;
    };
    struct demand_join {
        uint64_t key = 0;
        uint64_t generation = 0;
    };
    std::vector<demand_submit> submits;
    std::vector<demand_join> joins;
    size_t batch_needed_bytes = 0;
    const uint64_t wall_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    ctx.demand_async_plans.fetch_add(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        for (int expert = 0; expert < n_expert; ++expert) {
            if (min_rank[(size_t) expert] == INT_MAX) {
                continue;
            }
            int primary_kind = -1;
            int primary_order = INT_MAX;
            size_t expert_needed_bytes = 0;
            for (moe_managed * m : layer_it->second) {
                if (m == nullptr || expert >= m->n_expert) {
                    continue;
                }
                const int bits = moe_target_bits_for_rank(ctx, *m, expert, min_rank[(size_t) expert]);
                const bool ready =
                    moe_mwq_resolve_loaded_bits(*m, expert, bits) > 0 ||
                    moe_native_resident_compatible(ctx, *m, expert, bits);
                if (ready) {
                    continue;
                }
                expert_needed_bytes += moe_effective_stream_bytes_for_target(
                        ctx, *m, expert, bits);
                const int order = moe_demand_kind_order(moe_kind(*m));
                if (order < primary_order) {
                    primary_order = order;
                    primary_kind = (int) moe_kind(*m);
                }
            }
            if (primary_kind < 0) {
                continue;
            }

            moe_group_state & g = moe_group_get(ctx, layer, expert);
            if (g.demand_async_pending) {
                ctx.demand_async_dedup.fetch_add(1, std::memory_order_relaxed);
            } else {
                g.demand_async_pending = true;
                ++g.demand_async_generation;
                submits.push_back({
                    expert,
                    min_rank[(size_t) expert],
                    primary_kind,
                    g.demand_async_generation,
                    expert_needed_bytes,
                });
            }
            joins.push_back({moe_group_key(layer, expert), g.demand_async_generation});
        }

        bool batch_admission_planned = false;
        uint64_t batch_groups_before =
            ctx.group_evictions.load(std::memory_order_relaxed);
        if (moe_demand_admission_enabled() && ctx.params.budget_bytes > 0 &&
                !submits.empty() && layer_it->second.front() != nullptr) {
            batch_admission_planned = true;
            const size_t cache_bytes = ctx.resident_bytes -
                std::min(ctx.resident_bytes, ctx.demand_admission_staging_bytes);
            const size_t cache_free = ctx.params.budget_bytes -
                std::min(cache_bytes, ctx.params.budget_bytes);
            const size_t staging_limit = moe_env_mib_bytes(
                    "LLAMA_LAZY_MOE_DEMAND_ADMISSION_STAGING_MB", 64);
            const size_t staging_used =
                ctx.demand_admission_staging_bytes +
                ctx.demand_admission_staging_reserved_bytes;
            const size_t staging_free =
                staging_limit > staging_used ? staging_limit - staging_used : 0;

            std::vector<moe_demand_admission_request> requests;
            requests.reserve(submits.size());
            size_t total_needed_bytes = 0;
            for (const demand_submit & submit : submits) {
                moe_group_state & g = moe_group_get(ctx, layer, submit.expert);
                moe_admission_outcome_note_reuse_locked(ctx, g);
                g.demand_admission_token = ctx.profile_token_epoch;
                g.demand_admission_decided = true;
                g.demand_admission_bypass = false;

                moe_demand_admission_request request;
                request.layer = layer;
                request.expert = submit.expert;
                request.target_bytes = submit.needed_bytes;
                request.incoming_value = moe_demand_incoming_value_locked(
                        ctx, g, submit.needed_bytes, submit.rank);
                request.margin =
                    moe_env_f64("LLAMA_LAZY_MOE_DEMAND_ADMIT_MARGIN", 0.0);
                requests.push_back(request);
                total_needed_bytes += submit.needed_bytes;
            }

            std::vector<moe_demand_admission_result> results(requests.size());
            const size_t release_needed =
                total_needed_bytes > cache_free ? total_needed_bytes - cache_free : 0;
            if (release_needed > 0) {
                moe_managed * target = layer_it->second.front();
                moe_evict_context_guard evict_ctx(ctx, *target, true);
                moe_evict_lru(
                        ctx,
                        release_needed,
                        nullptr,
                        nullptr,
                        &requests,
                        &results,
                        cache_free,
                        staging_free);
            }

            for (size_t i = 0; i < submits.size(); ++i) {
                const demand_submit & submit = submits[i];
                const moe_demand_admission_request & request = requests[i];
                const moe_demand_admission_result & result = results[i];
                moe_group_state & g = moe_group_get(ctx, layer, submit.expert);
                if (result.evaluated) {
                    ctx.demand_admission_runs.fetch_add(1, std::memory_order_relaxed);
                    ctx.demand_admission_candidate_x1000.fetch_add(
                            (uint64_t) std::max(0.0, request.incoming_value * 1000.0),
                            std::memory_order_relaxed);
                    ctx.demand_admission_victim_x1000.fetch_add(
                            (uint64_t) std::max(0.0, result.victim_value * 1000.0),
                            std::memory_order_relaxed);
                    if (result.bypass) {
                        g.demand_admission_bypass = true;
                        g.admission_staging_reserved_bytes += submit.needed_bytes;
                        ctx.demand_admission_staging_reserved_bytes += submit.needed_bytes;
                        ctx.demand_admission_bypass.fetch_add(1, std::memory_order_relaxed);
                        if (!result.has_victim) {
                            ctx.demand_admission_no_victim.fetch_add(
                                    1, std::memory_order_relaxed);
                        }
                    } else {
                        ctx.demand_admission_admit.fetch_add(1, std::memory_order_relaxed);
                    }
                    moe_admission_outcome_begin_locked(
                            ctx, g, result.bypass, request.incoming_value, result);
                }
                if (!result.bypass) {
                    batch_needed_bytes += submit.needed_bytes;
                }
            }
        } else {
            for (const demand_submit & submit : submits) {
                batch_needed_bytes += submit.needed_bytes;
            }
        }

        if (batch_needed_bytes > 0 && ctx.params.budget_bytes > 0) {
            ctx.demand_async_batch_evict_runs.fetch_add(1, std::memory_order_relaxed);
            ctx.demand_async_batch_reserved_bytes.fetch_add(
                    batch_needed_bytes, std::memory_order_relaxed);
            uint64_t peak = ctx.demand_async_batch_reserved_peak.load(std::memory_order_relaxed);
            while (peak < batch_needed_bytes &&
                    !ctx.demand_async_batch_reserved_peak.compare_exchange_weak(
                            peak, batch_needed_bytes, std::memory_order_relaxed)) {
            }

            moe_managed * target = layer_it->second.front();
            if (!batch_admission_planned && target != nullptr) {
                moe_evict_context_guard evict_ctx(ctx, *target, true);
                const size_t cache_bytes = ctx.resident_bytes -
                    std::min(ctx.resident_bytes, ctx.demand_admission_staging_bytes);
                const size_t available = ctx.params.budget_bytes -
                    std::min(cache_bytes, ctx.params.budget_bytes);
                const size_t release_needed =
                    batch_needed_bytes > available ? batch_needed_bytes - available : 0;
                if (release_needed > 0) {
                    moe_evict_lru(ctx, release_needed);
                }
            }
            const uint64_t groups_after = ctx.group_evictions.load(std::memory_order_relaxed);
            if (groups_after > batch_groups_before) {
                ctx.demand_async_batch_evict_groups.fetch_add(
                        groups_after - batch_groups_before, std::memory_order_relaxed);
            }
            const size_t cache_bytes = ctx.resident_bytes -
                std::min(ctx.resident_bytes, ctx.demand_admission_staging_bytes);
            if (batch_needed_bytes > ctx.params.budget_bytes -
                    std::min(cache_bytes, ctx.params.budget_bytes)) {
                ctx.demand_async_batch_shortfall.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    if (joins.empty()) {
        return;
    }
    ctx.demand_async_groups.fetch_add(joins.size(), std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lk(ctx.qmtx);
        for (const demand_submit & submit : submits) {
            moe_prefetch_task task;
            task.is_group = true;
            task.is_demand = true;
            task.layer = layer;
            task.primary_kind = submit.primary_kind;
            task.e = submit.expert;
            task.rank = submit.rank;
            task.target_bits = ctx.params.base_bits;
            task.priority = INT_MAX;
            task.generation = submit.generation;
            task.seq = ctx.next_task_seq++;
            ctx.queue.push(task);
        }
    }
    ctx.demand_async_submitted.fetch_add(submits.size(), std::memory_order_relaxed);
    if (submits.size() > 1) {
        ctx.cv_q.notify_all();
    } else if (!submits.empty()) {
        ctx.cv_q.notify_one();
    }

    const uint64_t wait_t0 = ctx.profile ? moe_profile_now_ns() : 0;
    {
        std::unique_lock<std::mutex> lk(ctx.mtx);
        ctx.cv_done.wait(lk, [&] {
            for (const demand_join & join : joins) {
                auto it = ctx.groups.find(join.key);
                if (it != ctx.groups.end() &&
                        it->second.demand_async_pending &&
                        it->second.demand_async_generation == join.generation) {
                    return false;
                }
            }
            return true;
        });
    }
    ctx.demand_async_joined.fetch_add(joins.size(), std::memory_order_relaxed);
    if (ctx.profile) {
        const uint64_t now = moe_profile_now_ns();
        ctx.demand_async_wait_us.fetch_add((now - wait_t0) / 1000, std::memory_order_relaxed);
        ctx.demand_async_wall_us.fetch_add((now - wall_t0) / 1000, std::memory_order_relaxed);
    }
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
                if (ctx->prefetch_budget_enabled) {
                    const uint64_t charge = (uint64_t) m->stride;
                    if (charge > ctx->prefetch_budget_available_bytes) {
                        ctx->eam_prefetch_drop_budget.fetch_add(1, std::memory_order_relaxed);
                        continue;
                    }
                    ctx->prefetch_budget_available_bytes -= charge;
                }
                n_enqueued += moe_enqueue_prefetch(*ctx, *m, e, i, score, true) ? 1 : 0;
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

void llama_moe_buffer_set_prefetch_budget(
        llama_moe_buffer_context & ctx,
        uint64_t                  budget_bytes) {
    if (!llama_moe_buffer_enabled(&ctx)) {
        return;
    }
    std::lock_guard<std::mutex> lk(ctx.qmtx);
    ctx.prefetch_budget_enabled = budget_bytes > 0;
    ctx.prefetch_budget_bytes = budget_bytes;
    ctx.prefetch_budget_available_bytes = budget_bytes;
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
        uint64_t token_epoch_snapshot = 0;
        {
            std::lock_guard<std::mutex> lk(ctx->mtx);
            if (m.layer >= 0) {
                const int completed_layer = ctx->profile_last_layer;
                if (completed_layer >= 0 && completed_layer != m.layer) {
                    moe_release_demand_bypass_layer_locked(*ctx, completed_layer);
                    moe_evict_layer_done_locked(*ctx, completed_layer);
                }
                if (ctx->profile_last_layer >= 0 && m.layer < ctx->profile_last_layer) {
                    ++ctx->profile_token_epoch;
                    moe_admission_outcome_expire_locked(*ctx);
                    moe_admission_regret_expire_locked(*ctx);
                }
                ctx->profile_last_layer = m.layer;
            }
            ++ctx->exec_epoch;
            token_epoch_snapshot = ctx->profile_token_epoch;
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

                if (moe_env_flag("LLAMA_LAZY_MOE_GROUP_PREFETCH_SELFTEST", 0) &&
                        !ctx->group_prefetch_selftest_done && m.layer >= 0) {
                    for (int expert = 0; expert < m.n_expert; ++expert) {
                        if (op_counts[expert] > 0) {
                            ctx->group_prefetch_selftest_done = true;
                            const int rank = op_min_rank[expert] == 999999 ? 0 : op_min_rank[expert];
                            moe_enqueue_group_prefetch_task(*ctx, m.layer, expert, rank, 1.0f, INT_MAX / 2,
                                    token_epoch_snapshot, -1);
                            break;
                        }
                    }
                }

                struct route_run {
                    int expert = -1;
                    int target_bits = 0;
                    int best_rank = 0;
                    int count = 0;
                    int state = 2; // 0=ready, 1=loading/queued, 2=cold
                };
                auto classify_run_state = [&](int expert, int target_bits) {
                    std::lock_guard<std::mutex> lk(ctx->mtx);
                    if (ctx->params.dynamic_bits_real && moe_mwq_resolve_loaded_bits(m, expert, target_bits) > 0) {
                        return 0;
                    }
                    if (moe_native_resident_compatible(*ctx, m, expert, target_bits)) {
                        return 0;
                    }
                    if (expert >= 0 && expert < m.n_expert &&
                            (m.resident[expert] == ST_INFLIGHT ||
                             (expert < (int) m.queued.size() && m.queued[expert]))) {
                        return 1;
                    }
                    return 2;
                };

                std::vector<route_run> runs;
                runs.reserve((size_t) m.n_expert);
                for (int expert = 0; expert < m.n_expert; ++expert) {
                    if (op_target_bits[expert] <= 0) {
                        continue;
                    }
                    route_run run;
                    run.expert = expert;
                    run.target_bits = op_target_bits[expert];
                    run.best_rank = op_min_rank[expert] == 999999 ? 0 : op_min_rank[expert];
                    run.count = op_counts[expert];
                    run.state = classify_run_state(expert, run.target_bits);
                    runs.push_back(run);
                }
                if (moe_env_flag("LLAMA_LAZY_MOE_RUN_QUEUE", 1)) {
                    std::stable_sort(runs.begin(), runs.end(), [](const route_run & a, const route_run & b) {
                        if (a.state != b.state) {
                            return a.state < b.state;
                        }
                        if (a.count != b.count) {
                            return a.count > b.count;
                        }
                        return a.expert < b.expert;
                    });
                }
                if (!runs.empty()) {
                    ctx->runq_ops.fetch_add(1, std::memory_order_relaxed);
                }
                if (moe_kind(m) == moe_tensor_kind::gate) {
                    moe_prepare_demand_groups(*ctx, m.layer, ids);
                }
                if (moe_kind(m) == moe_tensor_kind::gate && moe_group_prefetch_same_layer_enabled()) {
                    const int max_groups = moe_group_prefetch_same_layer_max_groups();
                    const int min_count = moe_group_prefetch_same_layer_min_count();
                    struct group_prefetch_candidate {
                        route_run run;
                        int missing_siblings = 0;
                    };
                    std::vector<group_prefetch_candidate> candidates;
                    candidates.reserve(runs.size());
                    {
                        std::lock_guard<std::mutex> lk(ctx->mtx);
                        for (const route_run & run : runs) {
                            if (run.count < min_count) {
                                continue;
                            }
                            const int missing = moe_group_missing_sibling_slices_locked(
                                    *ctx, m.layer, run.expert, run.best_rank, (int) moe_tensor_kind::gate);
                            if (missing <= 0) {
                                continue;
                            }
                            candidates.push_back({run, missing});
                        }
                    }
                    std::stable_sort(candidates.begin(), candidates.end(),
                            [](const group_prefetch_candidate & a, const group_prefetch_candidate & b) {
                        if (a.missing_siblings != b.missing_siblings) {
                            return a.missing_siblings > b.missing_siblings;
                        }
                        if (a.run.count != b.run.count) {
                            return a.run.count > b.run.count;
                        }
                        return a.run.expert < b.run.expert;
                    });
                    int submitted = 0;
                    for (const group_prefetch_candidate & cand : candidates) {
                        if (max_groups > 0 && submitted >= max_groups) {
                            break;
                        }
                        const route_run & run = cand.run;
                        const float score = (float) run.count;
                        const int priority = INT_MAX / 4 + cand.missing_siblings * 2048 + std::min(run.count, 1024);
                        if (moe_enqueue_group_prefetch_task(*ctx, m.layer, run.expert, run.best_rank, score, priority,
                                    token_epoch_snapshot, (int) moe_tensor_kind::gate)) {
                            ++submitted;
                        }
                    }
                    if (submitted > 0) {
                        ctx->group_prefetch_same_layer_ops.fetch_add(1, std::memory_order_relaxed);
                    }
                }

                int run_order = 0;
                for (const route_run & run : runs) {
                    const int expert = run.expert;
                    const int target_bits = run.target_bits;
                    const int best_rank = run.best_rank;
                    if (run.state == 0) {
                        ctx->runq_ready.fetch_add(1, std::memory_order_relaxed);
                    } else if (run.state == 1) {
                        ctx->runq_loading.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        ctx->runq_cold.fetch_add(1, std::memory_order_relaxed);
                    }
                    if (ctx->run_queue_trace != nullptr) {
                        const moe_tensor_kind kind = moe_kind(m);
                        const char * kind_name = kind == moe_tensor_kind::gate ? "gate" :
                            (kind == moe_tensor_kind::up ? "up" :
                             (kind == moe_tensor_kind::down ? "down" : "other"));
                        std::fprintf(ctx->run_queue_trace,
                                "%llu\t%llu\t%d\t%d\t%s\t%d\t%d\t%d\t%s\t%d\n",
                                (unsigned long long) ctx->profile_token_epoch,
                                (unsigned long long) ctx->exec_epoch,
                                m.layer,
                                expert,
                                kind_name,
                                target_bits,
                                best_rank,
                                run.count,
                                run.state == 0 ? "ready" : (run.state == 1 ? "loading" : "cold"),
                                run_order);
                    }
                    ++run_order;
                    if (use_mwq_kernel) {
                        moe_stream_mwq_slice(*ctx, m, expert, true, target_bits, best_rank);
                    } else {
                        moe_stream_slice(*ctx, m, expert, true, target_bits, best_rank);
                    }
                    const int extra = run.count - 1;
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
                if (moe_kind(m) == moe_tensor_kind::down) {
                    if (moe_next_token_keep_enabled()) {
                        std::vector<int> last_token_experts;
                        if (ids->ne[1] > 0) {
                            const int64_t token = ids->ne[1] - 1;
                            last_token_experts.reserve((size_t) ids->ne[0]);
                            for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                                const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                                if (expert >= 0 && expert < m.n_expert) {
                                    last_token_experts.push_back(expert);
                                }
                            }
                            std::sort(last_token_experts.begin(), last_token_experts.end());
                            last_token_experts.erase(std::unique(last_token_experts.begin(), last_token_experts.end()),
                                    last_token_experts.end());
                        }
                        std::lock_guard<std::mutex> lk(ctx->mtx);
                        moe_next_token_update_after_route_locked(*ctx, m.layer, last_token_experts, m.n_expert);
                    }
                    if (moe_eam_predict_enabled()) {
                        std::vector<int> actual_experts;
                        actual_experts.reserve((size_t) m.n_expert);
                        for (int expert = 0; expert < m.n_expert; ++expert) {
                            if (op_counts[expert] > 0) {
                                actual_experts.push_back(expert);
                            }
                        }
                        moe_eam_predict_after_route(*ctx, m.layer, actual_experts, m.n_expert);
                    }
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
            if (moe_kind(m) == moe_tensor_kind::down && moe_next_token_keep_enabled()) {
                std::vector<int> last_token_experts;
                if (ids->ne[1] > 0) {
                    const int64_t token = ids->ne[1] - 1;
                    last_token_experts.reserve((size_t) ids->ne[0]);
                    for (int64_t rank = 0; rank < ids->ne[0]; ++rank) {
                        const int expert = *(const int32_t *) ((const char *) ids->data + token * ids->nb[1] + rank * ids->nb[0]);
                        if (expert >= 0 && expert < m.n_expert) {
                            last_token_experts.push_back(expert);
                        }
                    }
                    std::sort(last_token_experts.begin(), last_token_experts.end());
                    last_token_experts.erase(std::unique(last_token_experts.begin(), last_token_experts.end()),
                            last_token_experts.end());
                }
                std::lock_guard<std::mutex> lk(ctx->mtx);
                moe_next_token_update_after_route_locked(*ctx, m.layer, last_token_experts, m.n_expert);
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
            "admission={runs:%llu,ok:%llu,drop:%llu,no_victim:%llu,refresh:%llu,stale:%llu,avg_candidate:%.3f,avg_victim:%.3f} "
            "eam_predict={runs:%llu,cand:%llu,enq:%llu,drop:%llu,avg_score:%.3f} "
            "eamc={snap:%llu,stored:%zu,match:%llu,hit:%llu,miss:%llu,active:%zu,avg_sim:%.3f,prior:%llu} "
            "eam_evict={spec_unused:%llu,low:%llu,far:%llu,relaxed:%llu,protect_active:%llu,pinned:%llu,recent:%llu,high:%llu,early:%llu} "
            "layer_reserve={evict:%llu,protect:%llu,relaxed:%llu,reload_bad:%llu,reload_protect:%llu} "
            "reuse_evict={low:%llu,protect:%llu,predict_protect:%llu,low_cand:%llu,reload1:%llu,reload4:%llu,reload16:%llu} "
            "bad_reload={protect:%llu,soft:%llu,reload1:%llu,reload4:%llu,reload16:%llu} "
            "cct={updates:%llu,hit:%llu,miss:%llu,unused:%llu,replaced:%llu,prefetch_admit:%llu,prefetch_drop:%llu,evict_protect:%llu,evict_keep:%llu} "
            "next_token={updates:%llu,pred:%llu,hit:%llu,miss:%llu,admit:%llu,layer_done_keep:%llu,evict_keep:%llu} "
            "runq={ops:%llu,ready:%llu,loading:%llu,cold:%llu} group_fill={sibling_loads:%llu} "
            "group_prefetch={enq:%llu,run:%llu,slices:%llu,hit_existing:%llu,fail:%llu,dups:%llu,same_layer_ops:%llu,bytes:%.1f MiB} "
            "layer_done={evict:%llu,slices:%llu,bytes:%.1f MiB,protect_pinned:%llu,active:%llu,future:%llu,recent:%llu,bad_reload:%llu,inflight:%llu,next_token:%llu} "
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
            (unsigned long long) ctx.admission_compare_runs.load(),
            (unsigned long long) ctx.admission_compare_admit.load(),
            (unsigned long long) ctx.admission_compare_drop.load(),
            (unsigned long long) ctx.admission_compare_no_victim.load(),
            (unsigned long long) ctx.admission_floor_refresh.load(),
            (unsigned long long) ctx.admission_floor_stale.load(),
            ctx.admission_compare_runs.load() > 0 ?
                (double) ctx.admission_candidate_value_x1000.load() / 1000.0 / (double) ctx.admission_compare_runs.load() : 0.0,
            ctx.admission_compare_runs.load() > 0 ?
                (double) ctx.admission_victim_value_x1000.load() / 1000.0 / (double) ctx.admission_compare_runs.load() : 0.0,
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
            (unsigned long long) ctx.layer_reserve_evictions.load(),
            (unsigned long long) ctx.layer_reserve_protected.load(),
            (unsigned long long) ctx.layer_reserve_relaxed.load(),
            (unsigned long long) ctx.layer_reload_bad.load(),
            (unsigned long long) ctx.layer_reload_protect.load(),
            (unsigned long long) ctx.reuse_evict_low.load(),
            (unsigned long long) ctx.reuse_protect.load(),
            (unsigned long long) ctx.reuse_predict_protect.load(),
            (unsigned long long) ctx.reuse_low_candidates.load(),
            (unsigned long long) ctx.reuse_reload_1.load(),
            (unsigned long long) ctx.reuse_reload_4.load(),
            (unsigned long long) ctx.reuse_reload_16.load(),
            (unsigned long long) ctx.bad_reload_protect.load(),
            (unsigned long long) ctx.bad_reload_soft_keep.load(),
            (unsigned long long) ctx.bad_reload_1.load(),
            (unsigned long long) ctx.bad_reload_4.load(),
            (unsigned long long) ctx.bad_reload_16.load(),
            (unsigned long long) ctx.cct_updates.load(),
            (unsigned long long) ctx.cct_hits.load(),
            (unsigned long long) ctx.cct_misses.load(),
            (unsigned long long) ctx.cct_unused.load(),
            (unsigned long long) ctx.cct_replaced.load(),
            (unsigned long long) ctx.cct_prefetch_admit.load(),
            (unsigned long long) ctx.cct_prefetch_drop.load(),
            (unsigned long long) ctx.cct_evict_protect.load(),
            (unsigned long long) ctx.cct_evict_keep.load(),
            (unsigned long long) ctx.next_token_updates.load(),
            (unsigned long long) ctx.next_token_predicted.load(),
            (unsigned long long) ctx.next_token_hits.load(),
            (unsigned long long) ctx.next_token_misses.load(),
            (unsigned long long) ctx.next_token_admit.load(),
            (unsigned long long) ctx.next_token_layer_done_protect.load(),
            (unsigned long long) ctx.next_token_evict_protect.load(),
            (unsigned long long) ctx.runq_ops.load(),
            (unsigned long long) ctx.runq_ready.load(),
            (unsigned long long) ctx.runq_loading.load(),
            (unsigned long long) ctx.runq_cold.load(),
            (unsigned long long) ctx.group_fill_sibling_loads.load(),
            (unsigned long long) ctx.group_prefetch_enqueued.load(),
            (unsigned long long) ctx.group_prefetch_runs.load(),
            (unsigned long long) ctx.group_prefetch_slices.load(),
            (unsigned long long) ctx.group_prefetch_hit_existing.load(),
            (unsigned long long) ctx.group_prefetch_fail.load(),
            (unsigned long long) ctx.group_prefetch_dups.load(),
            (unsigned long long) ctx.group_prefetch_same_layer_ops.load(),
            ctx.group_prefetch_bytes.load() / 1048576.0,
            (unsigned long long) ctx.layer_done_evict_groups.load(),
            (unsigned long long) ctx.layer_done_evict_slices.load(),
            ctx.layer_done_evict_bytes.load() / 1048576.0,
            (unsigned long long) ctx.layer_done_protect_pinned.load(),
            (unsigned long long) ctx.layer_done_protect_active.load(),
            (unsigned long long) ctx.layer_done_protect_future.load(),
            (unsigned long long) ctx.layer_done_protect_recent.load(),
            (unsigned long long) ctx.layer_done_protect_bad_reload.load(),
            (unsigned long long) ctx.layer_done_protect_inflight.load(),
            (unsigned long long) ctx.layer_done_protect_next_token.load(),
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
    std::fprintf(stderr,
            "%s: demand_async={plans:%llu,groups:%llu,submitted:%llu,joined:%llu,dedup:%llu,"
            "slices:%llu,failed:%llu,wait_us:%llu,wall_us:%llu,batch_runs:%llu,"
            "batch_evict:%llu,reserved_mib:%.1f,reserved_peak_mib:%.1f,shortfall:%llu,"
            "worker_evict:%llu}\n",
            prefix,
            (unsigned long long) ctx.demand_async_plans.load(),
            (unsigned long long) ctx.demand_async_groups.load(),
            (unsigned long long) ctx.demand_async_submitted.load(),
            (unsigned long long) ctx.demand_async_joined.load(),
            (unsigned long long) ctx.demand_async_dedup.load(),
            (unsigned long long) ctx.demand_async_slices.load(),
            (unsigned long long) ctx.demand_async_failed.load(),
            (unsigned long long) ctx.demand_async_wait_us.load(),
            (unsigned long long) ctx.demand_async_wall_us.load(),
            (unsigned long long) ctx.demand_async_batch_evict_runs.load(),
            (unsigned long long) ctx.demand_async_batch_evict_groups.load(),
            ctx.demand_async_batch_reserved_bytes.load() / 1048576.0,
            ctx.demand_async_batch_reserved_peak.load() / 1048576.0,
            (unsigned long long) ctx.demand_async_batch_shortfall.load(),
            (unsigned long long) ctx.demand_async_worker_evictions.load());
    std::fprintf(stderr,
            "%s: demand_admission={runs:%llu,admit:%llu,bypass:%llu,no_victim:%llu,"
            "release_groups:%llu,release_slices:%llu,release_mib:%.1f,staging_peak_mib:%.1f,"
            "staging_reserved_mib:%.1f,avg_candidate:%.3f,avg_victim:%.3f}\n",
            prefix,
            (unsigned long long) ctx.demand_admission_runs.load(),
            (unsigned long long) ctx.demand_admission_admit.load(),
            (unsigned long long) ctx.demand_admission_bypass.load(),
            (unsigned long long) ctx.demand_admission_no_victim.load(),
            (unsigned long long) ctx.demand_admission_release_groups.load(),
            (unsigned long long) ctx.demand_admission_release_slices.load(),
            ctx.demand_admission_release_bytes.load() / 1048576.0,
            ctx.demand_admission_staging_peak.load() / 1048576.0,
            ctx.demand_admission_staging_reserved_bytes / 1048576.0,
            ctx.demand_admission_runs.load() > 0 ?
                (double) ctx.demand_admission_candidate_x1000.load() /
                    1000.0 / (double) ctx.demand_admission_runs.load() : 0.0,
            ctx.demand_admission_runs.load() > 0 ?
                (double) ctx.demand_admission_victim_x1000.load() /
                    1000.0 / (double) ctx.demand_admission_runs.load() : 0.0);
    std::fprintf(stderr,
            "%s: admission_outcome={admit_reuse1:%llu,admit_reuse4:%llu,admit_reuse16:%llu,"
            "admit_unused:%llu,bypass_reload1:%llu,bypass_reload4:%llu,bypass_reload16:%llu,"
            "bypass_success:%llu} admission_feedback={updates:%llu,positive:%llu,negative:%llu,"
            "score_candidates:%llu}\n",
            prefix,
            (unsigned long long) ctx.admission_outcome_admit_reuse_1.load(),
            (unsigned long long) ctx.admission_outcome_admit_reuse_4.load(),
            (unsigned long long) ctx.admission_outcome_admit_reuse_16.load(),
            (unsigned long long) ctx.admission_outcome_admit_unused.load(),
            (unsigned long long) ctx.admission_outcome_bypass_reload_1.load(),
            (unsigned long long) ctx.admission_outcome_bypass_reload_4.load(),
            (unsigned long long) ctx.admission_outcome_bypass_reload_16.load(),
            (unsigned long long) ctx.admission_outcome_bypass_success.load(),
            (unsigned long long) ctx.admission_feedback_updates.load(),
            (unsigned long long) ctx.admission_feedback_positive.load(),
            (unsigned long long) ctx.admission_feedback_negative.load(),
            (unsigned long long) ctx.admission_feedback_score_candidates.load());
    std::fprintf(stderr,
            "%s: admission_regret={pairs:%llu,resolved:%llu,expired:%llu,prefer_admit:%llu,"
            "prefer_bypass:%llu,tie:%llu,score_candidates:%llu,pending:%zu}\n",
            prefix,
            (unsigned long long) ctx.admission_regret_pairs.load(),
            (unsigned long long) ctx.admission_regret_resolved.load(),
            (unsigned long long) ctx.admission_regret_expired.load(),
            (unsigned long long) ctx.admission_regret_prefer_admit.load(),
            (unsigned long long) ctx.admission_regret_prefer_bypass.load(),
            (unsigned long long) ctx.admission_regret_tie.load(),
            (unsigned long long) ctx.admission_regret_score_candidates.load(),
            ctx.admission_pairs.size());
    std::fprintf(stderr,
            "%s: evict_scan={calls:%llu,entries:%llu,unique:%llu,dup:%llu,absolute:%llu,"
            "temporary:%llu,normal:%llu,selected_temporary:%llu,no_victim:%llu}\n",
            prefix,
            (unsigned long long) ctx.evict_scan_calls.load(),
            (unsigned long long) ctx.evict_scan_entries.load(),
            (unsigned long long) ctx.evict_scan_unique.load(),
            (unsigned long long) ctx.evict_scan_duplicates.load(),
            (unsigned long long) ctx.evict_scan_absolute.load(),
            (unsigned long long) ctx.evict_scan_temporary.load(),
            (unsigned long long) ctx.evict_scan_normal.load(),
            (unsigned long long) ctx.evict_scan_selected_temporary.load(),
            (unsigned long long) ctx.evict_scan_no_victim.load());
    std::fprintf(stderr,
            "%s: group_lru={size:%zu,rebuilds:%llu,unlinks:%llu,slice_lru_size:%zu}\n",
            prefix,
            ctx.group_lru.size(),
            (unsigned long long) ctx.group_lru_rebuilds.load(),
            (unsigned long long) ctx.group_lru_unlinks.load(),
            ctx.lru.size());
    std::fprintf(stderr,
            "%s: evict_sample={calls:%llu,k_sum:%llu,source:%llu,fallback:%llu}\n",
            prefix,
            (unsigned long long) ctx.evict_sample_calls.load(),
            (unsigned long long) ctx.evict_sample_k.load(),
            (unsigned long long) ctx.evict_sample_source.load(),
            (unsigned long long) ctx.evict_sample_fallback.load());
    std::fprintf(stderr,
            "%s: evict_cold_window={calls:%llu,items:%llu,source:%llu,fallback:%llu}\n",
            prefix,
            (unsigned long long) ctx.evict_cold_window_calls.load(),
            (unsigned long long) ctx.evict_cold_window_items.load(),
            (unsigned long long) ctx.evict_cold_window_source.load(),
            (unsigned long long) ctx.evict_cold_window_fallback.load());
    std::fprintf(stderr,
            "%s: resident_cache={cached_queries:%llu,slow_queries:%llu}\n",
            prefix,
            (unsigned long long) ctx.resident_bytes_cached_queries.load(),
            (unsigned long long) ctx.resident_bytes_slow_queries.load());
    std::fprintf(stderr,
            "%s: evict_profile={scan_us:%llu,sort_us:%llu,online_us:%llu,admission_us:%llu,"
            "trace_us:%llu,release_us:%llu,candidate_us:%llu,resident_queries:%llu}\n",
            prefix,
            (unsigned long long) ctx.prof_evict_scan_us.load(),
            (unsigned long long) ctx.prof_evict_sort_us.load(),
            (unsigned long long) ctx.prof_evict_online_us.load(),
            (unsigned long long) ctx.prof_evict_admission_us.load(),
            (unsigned long long) ctx.prof_evict_trace_us.load(),
            (unsigned long long) ctx.prof_evict_release_us.load(),
            (unsigned long long) ctx.prof_evict_candidate_us.load(),
            (unsigned long long) ctx.prof_evict_resident_queries.load());
    std::fprintf(stderr,
            "%s: ghost_feedback={evict:%llu,success:%llu,bad1:%llu,bad4:%llu,bad16:%llu,"
            "score_candidates:%llu}\n",
            prefix,
            (unsigned long long) ctx.ghost_evictions.load(),
            (unsigned long long) ctx.ghost_success.load(),
            (unsigned long long) ctx.ghost_bad_1.load(),
            (unsigned long long) ctx.ghost_bad_4.load(),
            (unsigned long long) ctx.ghost_bad_16.load(),
            (unsigned long long) ctx.ghost_score_candidates);
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

llama_moe_buffer_stats llama_moe_buffer_get_stats(llama_moe_buffer_context & ctx) {
    llama_moe_buffer_stats stats;
    {
        std::lock_guard<std::mutex> lk(ctx.mtx);
        stats.resident_bytes = ctx.resident_bytes;
        stats.budget_bytes   = ctx.params.budget_bytes;
        stats.expert_bytes   = ctx.expert_total;
    }
    stats.streams         = ctx.streams.load(std::memory_order_relaxed);
    stats.hits            = ctx.hits.load(std::memory_order_relaxed);
    stats.evictions       = ctx.evictions.load(std::memory_order_relaxed);
    stats.bytes_read      = ctx.bytes_read.load(std::memory_order_relaxed);
    stats.cache_hits      = ctx.cache_hits.load(std::memory_order_relaxed);
    stats.cache_misses    = ctx.cache_misses.load(std::memory_order_relaxed);
    stats.prefetch_hits   = ctx.eam_prefetch_hits.load(std::memory_order_relaxed);
    stats.prefetch_late   = ctx.eam_prefetch_late.load(std::memory_order_relaxed);
    stats.prefetch_unused = ctx.eam_prefetch_unused.load(std::memory_order_relaxed);
    stats.prefetch_budget_dropped =
        ctx.eam_prefetch_drop_budget.load(std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(ctx.qmtx);
        stats.prefetch_budget_bytes =
            (size_t) ctx.prefetch_budget_bytes;
        stats.prefetch_budget_available_bytes =
            (size_t) ctx.prefetch_budget_available_bytes;
    }
    return stats;
}

llama_moe_buffer_reclaim_result llama_moe_buffer_reclaim_clean(
        llama_moe_buffer_context & ctx,
        uint64_t                   target_bytes,
        uint32_t                   max_groups) {
    llama_moe_buffer_reclaim_result result;
    if (!llama_moe_buffer_enabled(&ctx) || target_bytes == 0 || max_groups == 0) {
        return result;
    }

    std::lock_guard<std::mutex> lk(ctx.mtx);
    while (result.released_bytes < target_bytes && result.released_groups < max_groups) {
        const size_t before = ctx.resident_bytes;
        if (before == 0 || !moe_evict_lru(ctx, target_bytes - result.released_bytes)) {
            break;
        }
        const size_t after = ctx.resident_bytes;
        if (before <= after) {
            break;
        }
        result.released_bytes += before - after;
        result.released_groups++;
    }
    result.target_satisfied = result.released_bytes >= target_bytes;
    return result;
}
