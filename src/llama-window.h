#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
struct llama_moe_buffer_context;

// Per-layer gate data for CLG (Cross-Layer Gate) prediction mode.
// Passed to llama_window_create alongside region inputs.
// llama_window_create dequantizes the tensors to FP32 at init time.
struct llama_window_gate_input {
    int                 layer         = -1;
    const ggml_tensor * norm_tensor   = nullptr; // ffn_norm[L].weight, shape [n_embd]
    float               norm_eps      = 1e-6f;
    const ggml_tensor * gate_tensor   = nullptr; // ffn_gate_inp[L].weight, [n_embd, n_expert]
    int                 n_expert_used = 0;        // K (actual top-K used by this model)
};

struct llama_window_region_input {
    std::string name;
    void * addr = nullptr;
    size_t size = 0;
    uint16_t file_idx = 0;
    size_t file_offset = 0;
    // For MoE expert tensors: if n_expert > 0, this tensor is indexed at per-expert
    // granularity instead of per-layer. expert_stride is nb[2] of the 3-D weight
    // tensor (bytes per expert slice, i.e. n_ff * n_embd * element_size).
    int n_expert = 0;
    size_t expert_stride = 0;
};

struct llama_window_params {
    bool enabled = false;
    bool use_dontneed = false;
    bool auto_tune = true;
    bool prefill_aggressive = false;
    bool debug_log = false;
    int window_size = 12;
    int prefetch_ahead = 2;
    int keep_behind = 3;
    int worker_threads = 1;
    size_t prefetch_budget = 256ull * 1024ull * 1024ull;
    size_t reclaim_budget = 128ull * 1024ull * 1024ull;
    size_t memory_limit = 0;
    // MoE expert sliding-window (Section 3.1 of "LLM in a Flash").
    // When enabled, expert weight tensors are indexed at per-expert granularity.
    // The window keeps every expert that was selected in any of the last
    // expert_window_tokens tokens resident; colder experts are evicted via
    // MADV_DONTNEED between tokens, and the newly-selected ones are prefetched
    // just-in-time after the routing decision (ffn_moe_topk node).
    bool expert_window = false;
    int  expert_window_tokens = 4;   // sliding-window history depth (LRU mode)
    bool expert_dontneed = false;    // evict cold experts via MADV_DONTNEED (LRU mode)

    // CLG (Cross-Layer Gate) prediction mode.
    // Replaces LRU-based eviction with input-driven deterministic prediction.
    // At l_out-{L}, applies the gate weights of layer L+1 to the current hidden
    // state to predict which top-(K+clg_delta) experts will be routed to.
    // Predicted experts are prefetched; the rest are evicted immediately.
    // This eliminates "sparsity evaporation" (the fundamental failure of LRU on
    // high-density MoE routing like Top-4/20) by acting on future routing rather
    // than past history.
    bool clg_predict           = false; // enable CLG prediction (requires gate_inputs)
    int  clg_delta             = 2;     // over-predict top-(K+delta) for recall margin
    int  clg_prefill_threshold = 4;     // skip eviction when n_tokens > this (prefill)
    // Hot-expert stratification: experts selected more than clg_hot_thr_pct% of decode
    // tokens are treated as "always resident" — they are never evicted by CLG, regardless
    // of whether CLG predicted them.  0 disables hot protection.
    int  clg_hot_thr_pct       = 20;   // activation-rate threshold to become "hot" (%)
    int  clg_hot_warmup        = 16;   // min decode tokens before hot detection activates
};

struct llama_window_stats {
    uint64_t graph_seq = 0;
    uint64_t layer_events = 0;
    uint64_t prefetch_calls = 0;
    uint64_t reclaim_calls = 0;
    uint64_t prefetch_failures = 0;
    uint64_t reclaim_failures = 0;
    uint64_t prefetch_tasks = 0;
    uint64_t reclaim_tasks = 0;
    uint64_t stale_tasks = 0;
    size_t bytes_prefetched = 0;
    size_t bytes_reclaimed = 0;
    size_t current_rss = 0;
    size_t peak_rss = 0;
    // Expert window stats (LRU mode)
    uint64_t expert_prefetch_calls   = 0;
    uint64_t expert_evict_calls      = 0;
    size_t   expert_bytes_prefetched = 0;
    size_t   expert_bytes_evicted    = 0;
    // CLG prediction stats
    uint64_t clg_predict_calls  = 0; // total prediction rounds
    uint64_t clg_evict_calls    = 0; // expert evictions issued by CLG
    uint64_t clg_miss_calls     = 0; // mispredictions caught at ffn_moe_topk (JIT fallback)
    uint64_t clg_hit_total      = 0; // sum of correctly predicted expert slots
    uint64_t clg_check_total    = 0; // sum of actual expert selections (denominator)
    uint64_t clg_hot_protected  = 0; // evictions skipped because expert was "hot"
};

struct llama_window_context;

std::shared_ptr<llama_window_context> llama_window_create(
        const std::vector<llama_window_region_input> & inputs,
        int n_layers,
        const llama_window_params & params,
        const std::vector<llama_window_gate_input> & gate_inputs = {});

bool llama_window_enabled(const llama_window_context * ctx);

// Connect the CLG predictor to an explicit-buffer expert streamer. When set, CLG
// routes its predicted experts to the moe-buffer's async prefetch (instead of the
// mmap window), giving layer L+1's slices lead time while layer L computes.
void llama_window_set_moe_buffer(llama_window_context & ctx, llama_moe_buffer_context * moe);

void llama_window_graph_begin(llama_window_context & ctx, bool prefill);
void llama_window_node_done(llama_window_context & ctx, const ggml_tensor * node);
void llama_window_graph_end(llama_window_context & ctx);

const llama_window_stats & llama_window_get_stats(const llama_window_context & ctx);
