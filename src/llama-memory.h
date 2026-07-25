#pragma once

#include "llama.h"
#include "llama-graph.h"
#include "llama-kv-cache-release.h"

#include <cstdint>
#include <map>
#include <memory>
#include <functional>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    llama_context_type ctx_type;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

enum class llama_paged_swap_error_reason : uint8_t {
    NONE = 0,
    SWAP_IN_IO_FAILURE,
    PAGED_SWAP_IN_IO_ERROR,
    ACTIVE_VISIBLE_RESTORE_FAILURE,
    NO_DUMMY_RESTORE_FAILURE,
    ACTIVE_READ_RELEASED_BLOCK,
    PAGED_ROW_MAPPING_INVALID,
    PAGED_WRITE_MAPPING_INVALID,
    ACTIVE_ROW_NOT_RESIDENT,
    INPUT_SETUP_FAILURE,
    PAGED_WRITE_ROLLBACK_DISCARD_FAILURE,
    PAGED_WRITE_GRAPH_ALLOC_FAILURE,
    PAGED_WRITE_COMPUTE_FAILURE,
    PAGED_WRITE_CONTEXT_INVALID,
};

enum class llama_paged_kv_write_action : uint8_t {
    COMMIT,
    ROLLBACK_PRE_COMPUTE,
    INVALIDATE_COMPUTE_STARTED,
};

struct llama_paged_swap_error {
    bool pending = false;
    llama_paged_swap_error_reason reason = llama_paged_swap_error_reason::NONE;
    uint32_t physical_block = UINT32_MAX;
    uint32_t physical_cell  = UINT32_MAX;
    int backend_status = 0;
    int backend_errno  = 0;
};

const char * llama_paged_swap_error_reason_name(llama_paged_swap_error_reason reason);

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;

    // Internal paged-KV synchronous decode error state. Default memory contexts do not
    // participate. Implementations that do use it must clear it once at ubatch start and leave
    // first-error-wins state pending until llama_context::process_ubatch checks it before compute.
    virtual void clear_paged_swap_error() {}
    virtual bool has_paged_swap_error() const { return false; }
    virtual llama_paged_swap_error get_paged_swap_error() const { return {}; }
    // Complete the metadata/KV write transaction opened while applying this ubatch.
    // Pre-compute failures roll back metadata and fresh allocations. Once compute may
    // have written K/V bytes, failure must invalidate the cache until an explicit reset.
    virtual void mark_paged_kv_compute_started() {}
    virtual bool finish_paged_kv_write(llama_paged_kv_write_action action) {
        GGML_UNUSED(action);
        return true;
    }
    virtual bool needs_paged_kv_post_graph_sync() const { return false; }
    virtual bool paged_kv_failure_handled() const { return false; }

    // Deterministic test-only failure injection at the real process_ubatch lifecycle
    // boundaries. Default memory contexts never inject.
    virtual bool test_paged_kv_fail_graph_alloc() { return false; }
    virtual bool test_paged_kv_fail_after_compute() { return false; }
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual int32_t prefetch_seq(llama_seq_id seq_id) { GGML_UNUSED(seq_id); return 0; }
    virtual int32_t prefetch_seq_step(llama_seq_id seq_id, uint32_t max_blocks) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(max_blocks);
        return 0;
    }
    virtual void set_seq_prefetch_protected(llama_seq_id seq_id, bool enabled) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(enabled);
    }

    // Dry-run bounded release: read-only evaluation of would-be release candidates
    // under given budget constraints.  Default no-op returns empty result — only
    // paged-KV memory implementations provide a real scanner.
    virtual llama_kv_bounded_release_result bounded_release_dry_run(
            uint64_t target_bytes, uint32_t max_scan_blocks) {
        GGML_UNUSED(target_bytes);
        GGML_UNUSED(max_scan_blocks);
        return {};
    }

    // Whether paged destructive release is currently active, and if not,
    // the precise reason.  Used by server pressure policy to differentiate
    // skip reasons (swap vs non-paged vs layout vs disabled).
    virtual llama_kv_release_status paged_release_status() const {
        return llama_kv_release_status::not_paged;
    }

    // Bounded destructive release: per-call budget-controlled MADV_DONTNEED
    // of eligible dead/unused KV blocks.  Default no-op returns empty result.
    // Only paged-KV memory implementations provide a real release path.
    // This is gated by LLAMA_KV_PRESSURE_BOUNDED_RELEASE (server policy),
    // NOT LLAMA_KV_PAGED_RELEASE (legacy apply-path gate).
    virtual llama_kv_bounded_release_result bounded_release(
            uint64_t target_bytes, uint32_t max_scan_blocks) {
        GGML_UNUSED(target_bytes);
        GGML_UNUSED(max_scan_blocks);
        return {};
    }

    // Read-only KV resident-page sampling via mincore(2).  Walks every KV
    // layer's K/V tensor data range, page-aligns, and counts resident pages.
    // Returns the total resident byte count across all KV tensors, or 0 if
    // mincore is not available on this platform / backend.
    // This is a diagnostic operation, NOT a hot-path call — it is O(n_layers ×
    // kv_size) in page-table walks.  Callers must rate-limit accordingly.
    virtual uint64_t sample_kv_resident_bytes() const {
        return 0;
    }

    virtual llama_kv_release_budget_snapshot sample_kv_release_budget() const {
        return {};
    }

    // Structural capability query for bounded destructive release,
    // independent of LLAMA_KV_PAGED_RELEASE or any legacy policy state.
    // Returns true when paged KV is active with a valid layout, in-graph
    // row-index gather, and swap is disabled — the mechanical preconditions
    // for safe MADV_DONTNEED of KV blocks.
    virtual bool bounded_release_can_enable() const {
        return false;
    }

    // Per-condition decomposition of bounded_release_can_enable() for precise
    // diagnostic skip-reason attribution.  Each field reports whether the
    // corresponding structural precondition is met.
    // Default (no-paged-KV) returns all-false.
    virtual llama_kv_bounded_release_capability bounded_release_can_enable_diagnose() const {
        return {};
    }

    // Independent bounded-release cumulative counters.  These are separate
    // from the legacy paged_block_release_* counters and are only incremented
    // by the bounded_release() path (LLAMA_KV_PRESSURE_BOUNDED_RELEASE).
    // Used by server pressure policy to cross-check per-call result attribution.
    virtual uint64_t bounded_release_counter_bytes() const  { return 0; }
    virtual uint64_t bounded_release_counter_blocks() const { return 0; }

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
