#pragma once

// FlexInfer-style streaming weight loader.
//
// Unlike the mmap-based llama-window (which only advises the kernel and leaves
// residency to the page cache), llama-flex never mmaps weights. It keeps a small
// ring of per-layer host buffers and streams each layer's tensors in from the
// file with direct, large, sequential reads on background IO threads, releasing
// a slot as soon as the compute has consumed the layer. The resident weight
// footprint is therefore bounded by the ring size (k layers), independent of the
// page cache:  RSS_weights ~= (k / n_layers) * model_size.
//
// This module (stage 2a) is self-contained and testable without ggml: it owns
// the file descriptors, the layer/tensor metadata and the ring buffer, and
// exposes a "request layer -> wait ready -> get tensor ptr -> release layer"
// API. Integration with the compute path is layered on top separately.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_flex_tensor {
    std::string name;
    uint16_t    file_idx    = 0;
    size_t      file_offset = 0;  // byte offset of tensor data in the file
    size_t      size        = 0;  // tensor data size in bytes
    size_t      buf_offset  = 0;  // offset within the ring slot (streamed) or lock buffer (locked)
    bool        locked      = false; // balanced-locked: resident permanently, never streamed
    bool        delta_locked = false; // runtime-pinned into a delta lock buffer
    size_t      delta_buf_index = 0;
    size_t      delta_buf_offset = 0;
};

struct llama_flex_params {
    bool   enabled        = false;
    bool   direct_io      = true; // use O_DIRECT for streaming reads when possible
    bool   debug_log      = false;
    bool   sched_auto     = false; // cap-aware scheduler: tune ring after pinning
    bool   adaptive_ahead = true;  // runtime prefetch-depth controller
    int    ring_layers    = 4;    // k: number of layer slots kept resident
    int    prefetch_ahead = 2;    // initial/fixed layers ahead to stream
    int    prefetch_ahead_max = 8; // max runtime ahead when adaptive_ahead is enabled
    int    io_threads     = 4;    // background streaming threads
    size_t lock_bytes     = 0;    // balanced-locking budget (stage 2c); 0 = off
    size_t memory_budget_bytes = 0; // cgroup/MemAvailable budget for sched_auto
    size_t fixed_bytes    = 0;    // non-flex memory reserve for sched_auto
    size_t read_cost_bytes = 0;   // per-stream-read fixed cost in equivalent bytes for cost-aware pinning
    bool   read_cost_auto = false; // estimate read_cost_bytes from startup pread latency/bandwidth calibration
    bool   global_rebalance = false; // spend leftover per-layer lock budget on high stream-cost layers
    std::string pin_policy = "small-first"; // small-first, large-first, attn-first, ffn-first, cost-aware, none
};

struct llama_flex_stats {
    uint64_t layer_loads     = 0;
    uint64_t layer_hits      = 0;  // requested while already resident
    uint64_t wait_events     = 0;
    uint64_t bytes_streamed  = 0;  // logical tensor bytes delivered to slots
    uint64_t bytes_read_phys = 0;  // physical bytes pread() from disk (incl O_DIRECT alignment padding)
    uint64_t read_ops        = 0;  // number of pread() calls issued for streaming
    uint64_t total_io_us     = 0;
    uint64_t total_wait_us   = 0;
    double   ewma_io_us      = 0.0;
    double   ewma_compute_us = 0.0;
    double   ewma_wait_us    = 0.0;
    uint64_t demand_loads    = 0;  // layer was not already queued/loading when compute needed it
    uint64_t prefetch_queued = 0;
    uint64_t prefetch_budget_dropped = 0;
    uint64_t queue_requeues  = 0;  // IO worker could not acquire a slot
    uint64_t evictions       = 0;
    uint64_t releases        = 0;
    uint64_t graphs          = 0;
    uint64_t ahead_adjustments = 0;
    uint64_t locked_tensors  = 0;
    uint64_t streamed_tensors = 0;
    uint64_t delta_locked_tensors = 0;
    uint64_t delta_pin_attempts = 0;
    uint64_t delta_pin_failures = 0;
    size_t   ring_bytes      = 0;  // total bytes held by the ring
    size_t   slot_bytes      = 0;  // bytes charged for one streamed layer slot
    size_t   locked_bytes    = 0;  // bytes pinned by balanced locking
    size_t   delta_locked_bytes = 0;
    size_t   delta_pin_saved_per_token = 0;
    size_t   lock_budget_unused = 0;
    size_t   stream_per_token = 0; // unlocked bytes that must be read each token
    int      effective_ahead = 0;
    int      min_effective_ahead = 0;
    int      max_effective_ahead = 0;
    size_t   sched_budget_bytes = 0;
    size_t   sched_fixed_bytes = 0;
    size_t   sched_ring_room = 0;
    size_t   read_cost_bytes = 0;
    size_t   global_rebalance_bytes = 0;
    uint64_t global_rebalance_tensors = 0;
    size_t   prefetch_budget_bytes = 0;
    size_t   prefetch_budget_available_bytes = 0;
};

struct llama_flex_reclaim_result {
    uint64_t released_bytes = 0;
    uint32_t released_layers = 0;
    bool target_satisfied = false;
};

struct llama_flex_resize_result {
    bool attempted = false;
    bool changed = false;
    int old_slots = 0;
    int new_slots = 0;
    uint64_t old_bytes = 0;
    uint64_t new_bytes = 0;
    const char * reason = "none";
};

struct llama_flex_delta_pin_result {
    bool attempted = false;
    bool changed = false;
    uint64_t requested_bytes = 0;
    uint64_t pinned_bytes = 0;
    uint64_t saved_per_token_bytes = 0;
    uint64_t candidates = 0;
    uint64_t pinned_tensors = 0;
    double roi = 0.0;
    const char * reason = "none";
};

struct ggml_tensor;
struct llama_flex_context;

// Create a streaming context. Each fd is dup()'d and reopened for streaming
// reads (with O_DIRECT when params.direct_io and the fs supports it). n_layers
// is the number of decoder layers. Returns a disabled context
// (llama_flex_enabled()==false) when params.enabled is false.
std::shared_ptr<llama_flex_context> llama_flex_create(
        const std::vector<int> &  fds,
        int                       n_layers,
        const llama_flex_params & params);

bool llama_flex_enabled(const llama_flex_context * ctx);

// Register one tensor as belonging to a decoder layer. Must be called for every
// streamed tensor before any layer is requested. layer_id in [0, n_layers).
void llama_flex_register_tensor(
        llama_flex_context & ctx,
        int                  layer_id,
        const llama_flex_tensor & tensor);

// Finalize registration: sizes each layer slot to the largest layer and
// allocates the ring. Must be called once after all tensors are registered.
void llama_flex_finalize(llama_flex_context & ctx);

// Ask the IO threads to stream a layer into the ring (non-blocking).
void llama_flex_request_layer(llama_flex_context & ctx, int layer_id);

// Block until a layer is resident in the ring.
void llama_flex_wait_layer(llama_flex_context & ctx, int layer_id);

// Return a pointer to a tensor's data inside the ring. The layer must be
// resident (call llama_flex_wait_layer first). Returns nullptr on miss.
void * llama_flex_get_tensor(llama_flex_context & ctx, int layer_id, const std::string & name);

// Mark a layer as consumed; its slot may be reused for future prefetches.
void llama_flex_release_layer(llama_flex_context & ctx, int layer_id);

const llama_flex_stats & llama_flex_get_stats(const llama_flex_context & ctx);

// Set a per-tick speculative prefetch budget. A zero budget disables the gate.
// Demand loads are never gated by this API.
void llama_flex_set_prefetch_budget(
        llama_flex_context & ctx,
        uint64_t             budget_bytes);

// Release already-consumed resident layer slots back to the OS with
// MADV_DONTNEED. This never touches the current compute layer; it only reclaims
// layers that were previously marked released by llama_flex_release_layer().
llama_flex_reclaim_result llama_flex_reclaim_released(
        llama_flex_context & ctx,
        uint64_t             target_bytes,
        uint32_t             max_layers);

// Resize the streaming ring. Growing is immediate. Shrinking only removes free
// slots so active/resident layers are never invalidated.
llama_flex_resize_result llama_flex_resize_ring(
        llama_flex_context & ctx,
        int                 target_slots);

llama_flex_delta_pin_result llama_flex_delta_pin(
        llama_flex_context & ctx,
        uint64_t             budget_bytes,
        double               min_roi);

// --- compute-path integration ---------------------------------------------

// Reset per-graph state and kick off the initial prefetch (layers [0, ahead]).
// Call once before each decode graph.
void llama_flex_graph_begin(llama_flex_context & ctx);

// ggml_cpu_weight_stream_callback. user_data must be a llama_flex_context*.
// On ith==0 it streams the op's flex-managed weight(s) into the ring (waiting
// if necessary), repoints their ->data, and drives prefetch/release at layer
// boundaries. Returns true iff the op owns flex-managed weights (so the CPU
// backend issues a barrier before running the kernel).
bool llama_flex_stream_callback(struct ggml_tensor * op, int ith, void * user_data);
