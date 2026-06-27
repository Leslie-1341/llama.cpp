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

struct ggml_tensor;
struct llama_moe_buffer_context;

struct llama_moe_buffer_params {
    bool   enabled      = false;
    bool   debug_log    = false;
    bool   direct_io    = false;     // O_DIRECT streaming reads (needs alignment)
    size_t budget_bytes = 0;         // resident-expert byte budget; 0 = unbounded
    int    n_workers    = 1;         // parallel prefetch workers (raise to lift effective
                                     // read bandwidth on NVMe: single-thread O_DIRECT
                                     // random reads under-utilise the device)
    float  hot_ratio    = 0.0f;      // relative hotness: an expert is pinned (never
                                     // LRU-evicted) when its activation count exceeds
                                     // hot_ratio * (tensor mean activation). 0 = pure
                                     // LRU. Bounded by construction: only experts
                                     // routed more than hot_ratio× the per-tensor
                                     // average stay pinned, so the pin set cannot grow
                                     // to "all experts" as the sequence lengthens.
};

std::shared_ptr<llama_moe_buffer_context> llama_moe_buffer_create(const llama_moe_buffer_params & params);

bool llama_moe_buffer_enabled(const llama_moe_buffer_context * ctx);

// Total bytes of all registered expert tensors. Used by the adaptive-budget
// decision (the streamable working set upper bound).
size_t llama_moe_buffer_expert_bytes(const llama_moe_buffer_context * ctx);

// Override the resident-expert byte budget after registration (for adaptive
// budgeting computed once the expert total and available memory are known).
// 0 = unbounded.
void llama_moe_buffer_set_budget(llama_moe_buffer_context * ctx, size_t budget_bytes);

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

void llama_moe_buffer_print_stats(const llama_moe_buffer_context & ctx);
