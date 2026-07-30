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
};

struct llama_flex_params {
    bool   enabled        = false;
    bool   direct_io      = true; // use O_DIRECT for streaming reads when possible
    bool   debug_log      = false;
    int    ring_layers    = 4;    // k: number of layer slots kept resident
    int    prefetch_ahead = 2;    // how many layers ahead to stream
    int    io_threads     = 4;    // background streaming threads
    size_t lock_bytes     = 0;    // balanced-locking budget (stage 2c); 0 = off
    std::string pin_policy = "small-first";
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
    uint64_t demand_loads    = 0;  // layer was not already queued/loading when compute needed it
    uint64_t prefetch_queued = 0;
    uint64_t queue_requeues  = 0;  // IO worker could not acquire a slot
    uint64_t evictions       = 0;
    uint64_t releases        = 0;
    uint64_t graphs          = 0;
    uint64_t locked_tensors  = 0;
    uint64_t streamed_tensors = 0;
    size_t   ring_bytes      = 0;  // total bytes held by the ring
    size_t   locked_bytes    = 0;  // bytes pinned by balanced locking
    size_t   lock_budget_unused = 0;
    size_t   stream_per_token = 0; // unlocked bytes that must be read each token
    int      effective_ahead = 0;
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
