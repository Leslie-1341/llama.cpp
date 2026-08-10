#pragma once

// MoE expert streaming via explicit anonymous buffers (per-expert-slice repoint).
//
// Unlike the mmap+madvise expert window (whose pages the kernel keeps re-caching,
// so the real physical footprint stays pinned at the memory cap), this path never
// keeps the expert weights mmap-backed. For every `ffn_*_exps` weight tensor it
// allocates a full-size ANONYMOUS buffer and repoints tensor->data to it. Only the
// expert slices actually selected by the router are pread() into their id-indexed
// slot; cold experts are never read, so their slots stay zero-fill (unbacked) and
// occupy no physical memory. Resident slots are bounded by an LRU byte budget and
// evicted with madvise(MADV_DONTNEED) (anonymous => pages are truly freed).
//
// Because mul_mat_id reads `tensor->data + expert_id * stride`, the full-size
// buffer means no id remapping and no kernel change: the GEMM just reads the slot
// we streamed. Residency is guaranteed before the kernel runs by the CPU
// weight-stream callback (ith==0 streams the op's selected experts, then a barrier
// publishes them to all worker threads).

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct ggml_tensor;
struct llama_moe_buffer_context;

struct llama_moe_buffer_params {
    bool   enabled      = false;
    bool   debug_log    = false;
    bool   direct_io    = false;     // O_DIRECT streaming reads (needs alignment)
    bool   hebf_schedule = true;     // priority scheduling for async expert prefetch
    bool   dynamic_bits  = true;     // policy-only bit-width selection (no data-format change)
    bool   dynamic_bits_real = true; // require exact low-bit/MWQ data; fallback to full reads if unavailable
    bool   strict_sidecar = false; // abort if the selected sidecar bit width is unavailable
    bool   native_hot = false;       // allow target bits >= base_bits to use original GGUF quant slices in the op override
    bool   fuse_gate_up = true;      // defer ffn_gate/up_exps MUL_MAT_ID until the sibling op is seen, then compute both together
    bool   fuse_swiglu = true;       // override SWIGLU when its inputs are the just-fused gate/up expert outputs
    bool   fuse_direct_swiglu = true; // compute dot(gate), dot(up), and silu*up inside the SWIGLU override without writing gate/up tensors
    bool   fuse_expert_ffn = false;  // skip full SWIGLU tensor and fuse gate/up SWIGLU directly into down projection
    bool   prefetch_down_with_swiglu = false; // when direct-SWIGLU ensures gate/up, also warm the same expert's down slice
    bool   avx512_q2 = true;         // use the AVX512 q2 hierarchical direct-SWIGLU hot path when available
    bool   avx512_q2_dot = true;     // use the AVX512 q2 hierarchical MUL_MAT_ID/down-projection dot path when available
    bool   vnni_q2 = false;          // quantize F32 activations to int8 and use AVX512-VNNI for q2 hierarchical expert dots
    bool   vnni_q2_down = true;      // allow the VNNI q2 path for MUL_MAT_ID/down-projection dots
    bool   vnni_q2_swiglu = true;    // allow the VNNI q2 path for fused gate/up SWIGLU
    int    vnni_block = 64;          // activation int8 quantization block; q2 VNNI currently uses 64-column chunks
    int    avx512_prefetch = 0;      // q2-hier kernel prefetch distance in MWQ blocks; 0 disables explicit prefetch
    size_t budget_bytes = 1024ull * 1024ull * 1024ull; // resident-expert byte budget when bounded
    bool   budget_unbounded = false; // explicit unbounded mode; separates unlimited from a zero-byte bounded budget
    size_t planner_safe_budget_bytes = 0; // load-time safe budget derived from memory.max
    size_t planner_floor_bytes = 0;       // non-reclaimable load-time floor
    int    n_workers    = 2;         // parallel prefetch workers (raise to lift effective
                                     // read bandwidth on NVMe: single-thread O_DIRECT
                                     // random reads under-utilise the device)
    int    base_bits    = 2;         // current on-disk expert precision for dyn-bit accounting
    int    hot_bits     = 2;         // target bits for rank-0/hot predicted experts
    int    warm_bits    = 2;         // target bits for mid-rank predicted experts
    int    cold_bits    = 2;         // target bits for low-rank predicted experts
    int    gate_bits    = 2;         // optional exact target bits for ffn_gate_exps; 0 = use hot/warm/cold policy
    int    up_bits      = 2;         // optional exact target bits for ffn_up_exps; 0 = use hot/warm/cold policy
    int    down_bits    = 2;         // optional exact target bits for ffn_down_exps; 0 = use hot/warm/cold policy
    int    fixed_bits   = 0;         // force every expert tensor/rank to this sidecar bit width; 0 = dynamic policy
    int    gate_min_bits = 2;        // default sensitivity floor: gate logits should not use q2 unless explicitly requested
    int    up_min_bits   = 0;        // up is the least sensitive expert projection; 0 = no floor
    int    down_min_bits = 2;        // down projection feeds the residual path, keep at least q3 by default
    int    sync_top_k   = 4;         // routed expert ids are assumed grouped by top-k rank
    float  hot_ratio    = 0.0f;      // relative hotness: an expert is pinned (never
                                     // LRU-evicted) when its activation count exceeds
                                     // hot_ratio * (tensor mean activation). 0 = pure
                                     // LRU. Bounded by construction: only experts
                                     // routed more than hot_ratio× the per-tensor
                                     // average stay pinned, so the pin set cannot grow
                                     // to "all experts" as the sequence lengthens.
    float  pinned_fraction = 0.35f;  // fraction of resident budget reserved for dynamic
                                     // (layer, expert) hot groups. Pinned groups are
                                     // excluded from eviction until demoted by the
                                     // periodic top-score refresh.
    float  pinned_layer_fraction = 0.18f; // max fraction of pinned budget one layer may use
    int    active_window = 4;        // future-use distance protected by Belady-style eviction
    int    group_cooldown_tokens = 0; // protect recently used (layer, expert) groups for N decode-token epochs
    int    pin_refresh_interval = 128; // group touches between top-score pin refreshes
    double warm_coverage = 0.95;    // target per-layer expert probability mass for warm-start sizing
    std::string sidecar_path;        // optional exact low-bit/MWQ sidecar data source
};

struct llama_moe_buffer_stats {
    size_t   resident_bytes = 0;
    size_t   budget_bytes = 0;
    bool     budget_unbounded = false;
    size_t   planner_safe_budget_bytes = 0;
    size_t   planner_floor_bytes = 0;
    size_t   expert_bytes = 0;
    uint64_t streams = 0;
    uint64_t hits = 0;
    uint64_t evictions = 0;
    uint64_t bytes_read = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_late = 0;
    uint64_t prefetch_unused = 0;
    uint64_t prefetch_budget_dropped = 0;
    size_t   prefetch_budget_bytes = 0;
    size_t   prefetch_budget_available_bytes = 0;
    size_t   warm_working_set_bytes = 0;
    uint64_t warm_working_set_groups = 0;
    double   warm_working_set_coverage = 0.0;
};

struct llama_moe_buffer_reclaim_result {
    uint64_t released_bytes = 0;
    uint32_t released_groups = 0;
    bool target_satisfied = false;
};

std::shared_ptr<llama_moe_buffer_context> llama_moe_buffer_create(const llama_moe_buffer_params & params);

bool llama_moe_buffer_enabled(const llama_moe_buffer_context * ctx);

// Total bytes of all registered expert tensors. Used by the adaptive-budget
// decision (the streamable working set upper bound).
size_t llama_moe_buffer_expert_bytes(const llama_moe_buffer_context * ctx);

// Override the resident-expert byte budget after registration (for adaptive
// budgeting computed once the expert total and available memory are known).
void llama_moe_buffer_set_budget(llama_moe_buffer_context * ctx, size_t budget_bytes);

// Install the load-time memory.max plan. This makes the bounded/unbounded
// state explicit so a bounded zero-byte budget is not confused with unlimited.
void llama_moe_buffer_set_budget_plan(
        llama_moe_buffer_context * ctx,
        size_t                    budget_bytes,
        bool                      budget_unbounded,
        size_t                    planner_safe_budget_bytes,
        size_t                    planner_floor_bytes);

// Estimate the warm-start resident working set from the registered
// (layer, expert) groups. Uses observed per-request/historical group scores when
// present, and the structural top-k/uniform prior before any routing history
// exists.
size_t llama_moe_buffer_warm_working_set_bytes(
        llama_moe_buffer_context * ctx,
        double                    coverage);

// Register one `*_exps` weight tensor: allocate a full-size anonymous buffer,
// repoint exps->data to it, and record per-expert file metadata. `fd` is the
// model file descriptor (duplicated internally). Returns true on success; on
// failure the tensor is left untouched (still mmap-backed).
bool llama_moe_buffer_register(
        llama_moe_buffer_context & ctx,
        ggml_tensor *              exps,
        int                        fd,
        size_t                     file_offset,
        size_t                     expert_stride,
        int                        n_expert);

// ggml_cpu_weight_stream_callback. user_data must be a llama_moe_buffer_context*.
// For a managed GGML_OP_MUL_MAT_ID op, ith==0 ensures every expert in op->src[2]
// is resident in the buffer (pread on miss, LRU-evict to stay within budget).
// Returns true iff the op is managed (so the CPU backend issues a barrier).
bool llama_moe_buffer_stream_callback(ggml_tensor * op, int ith, void * user_data);

// CPU override for managed GGML_OP_MUL_MAT_ID ops whose selected experts are
// resident as MWQ sidecar slices. Returns true when it completed the op.
bool llama_moe_buffer_mul_mat_id_callback(ggml_tensor * op, int ith, int nth, void * user_data);

// Asynchronous prefetch hint, issued by the CLG predictor while layer L computes
// to give layer L+1's experts lead time. For every `*_exps` tensor of `layer`,
// the listed experts are enqueued to a background worker that streams their
// slices into the anonymous buffer (no-op for already-resident/in-flight slots).
// When the layer's mul_mat_id later runs, the weight-stream callback finds the
// slices already resident (hit) instead of taking a synchronous read. Safe to
// call concurrently with the weight-stream callback. No-op if ctx is null/disabled.
void llama_moe_buffer_prefetch(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        int                        n_experts);

// Ranked prefetch hint. `scores` may be null; when present, higher scores are
// scheduled first and used by the dynamic-bit policy for accounting. `experts`
// should be ordered by predicted utility (rank 0 = hottest/most likely).
void llama_moe_buffer_prefetch_ranked(
        llama_moe_buffer_context * ctx,
        int                        layer,
        const int *                experts,
        const float *              scores,
        int                        n_experts);

llama_moe_buffer_stats llama_moe_buffer_get_stats(llama_moe_buffer_context & ctx);

// Reclaim cold resident expert groups using the buffer's existing victim
// selection. This releases clean anonymous expert pages; the model file/sidecar
// remains the authoritative source for future reloads.
llama_moe_buffer_reclaim_result llama_moe_buffer_reclaim_clean(
        llama_moe_buffer_context & ctx,
        uint64_t                   target_bytes,
        uint32_t                   max_groups);

// Set a per-tick speculative prefetch budget. A zero budget disables the gate.
// Demand loads are never gated by this API.
void llama_moe_buffer_set_prefetch_budget(
        llama_moe_buffer_context & ctx,
        uint64_t                  budget_bytes);

void llama_moe_buffer_print_stats(const llama_moe_buffer_context & ctx);
