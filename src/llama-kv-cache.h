#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;
class llama_kv_cache_context;

//
// llama_kv_cache
//

// Stage2-backend0: backing-store abstraction shell for runtime KV swap.
//
// This interface is intentionally not instantiated or called in backend0. It only fixes the
// seam for future exact offload work, where a file-backed/tmpfile implementation can persist
// cell or block bytes outside the anonymous KV tensor allocation. No file I/O, swap-out,
// swap-in, ensure_resident, prefetch, or madvise behavior is implemented here.
enum class llama_kv_backing_store_status : uint8_t {
    ok = 0,
    disabled,
    io_error,
    bad_slot,
};

struct llama_kv_backing_store_stats {
    uint64_t bytes_written  = 0;
    uint64_t bytes_read     = 0;
    uint64_t bytes_released = 0;
    uint64_t write_calls    = 0;
    uint64_t read_calls     = 0;
    uint64_t release_calls  = 0;
    int      last_errno     = 0;
};

class llama_kv_backing_store_i {
public:
    virtual ~llama_kv_backing_store_i() = default;

    virtual llama_kv_backing_store_status write_cell(
            uint32_t   strm,
            uint32_t   cell,
            const void * data,
            size_t     size,
            uint64_t & offset_out) {
        (void) strm;
        (void) cell;
        (void) data;
        (void) size;
        offset_out = 0;
        return llama_kv_backing_store_status::disabled;
    }

    virtual llama_kv_backing_store_status read_cell(
            uint32_t strm,
            uint32_t cell,
            uint64_t offset,
            void *   data,
            size_t   size) {
        (void) strm;
        (void) cell;
        (void) offset;
        (void) data;
        (void) size;
        return llama_kv_backing_store_status::disabled;
    }

    virtual llama_kv_backing_store_status release(uint64_t offset, size_t size) {
        (void) offset;
        (void) size;
        return llama_kv_backing_store_status::disabled;
    }

    virtual llama_kv_backing_store_status reset() {
        return llama_kv_backing_store_status::disabled;
    }

    virtual const llama_kv_backing_store_stats & get_stats() const {
        static const llama_kv_backing_store_stats empty;
        return empty;
    }
};

class llama_kv_backing_store_file : public llama_kv_backing_store_i {
public:
    llama_kv_backing_store_file();
    ~llama_kv_backing_store_file() override;

    llama_kv_backing_store_status write_cell(
            uint32_t   strm,
            uint32_t   cell,
            const void * data,
            size_t     size,
            uint64_t & offset_out) override;

    llama_kv_backing_store_status read_cell(
            uint32_t strm,
            uint32_t cell,
            uint64_t offset,
            void *   data,
            size_t   size) override;

    llama_kv_backing_store_status release(uint64_t offset, size_t size) override;
    llama_kv_backing_store_status reset() override;

    bool is_enabled() const {
        return fd >= 0;
    }

    uint64_t get_file_len() const {
        return file_len;
    }

    const llama_kv_backing_store_stats & get_stats() const override {
        return stats;
    }

private:
    int fd = -1;
    std::FILE * file = nullptr;
    uint64_t file_len = 0;
    llama_kv_backing_store_stats stats;
};

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    llama_kv_cache(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse);

    ~llama_kv_cache();

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;
    uint32_t get_visible_lo(const slot_info & sinfo) const;
    uint32_t get_reserve_n_kv() const;
    bool uses_approx_dynamic_view() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t visible_lo, const slot_info & sinfo, bool causal_attn, ggml_tensor * row_idx = nullptr) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, uint32_t visible_lo, const slot_info & sinfo, bool causal_attn, ggml_tensor * row_idx = nullptr) const;

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    // Stage2-exact-swapout0: no-op scaffold for future exact swap-in before KV reads.
    // Not called from apply() in this stage.
    void ensure_resident(uint32_t n_kv);
    void swap_out_window(uint32_t n_kv);
    void sample_swap_rss();

    // stage F1 / P1: advise the unused tail capacity [GGML_PAD(n_kv, 256), kv_size) away via
    // MADV_DONTNEED to lower current RSS. No-op unless LLAMA_KV_LAZY_TAIL=1 (and !v_trans &&
    // n_stream==1). See docs/kv_lazy_block_stage_f1_design.md.
    void madvise_tail(uint32_t n_kv);

    // stage P2: clear-frontier. When LLAMA_KV_LAZY_CLEAR=1 (and !v_trans && n_stream==1),
    // the construction-time full buffer clear is replaced by clearing only the [0, clear_frontier)
    // prefix; the tail [clear_frontier, kv_size) is left untouched so it is never committed,
    // lowering peak RSS. As n_kv grows past clear_frontier this advances the frontier, zeroing
    // newly-readable rows before the graph reads K/V. No-op unless kv_lazy_clear.
    // See docs/kv_lazy_block_stage_p2_read.md.
    void clear_frontier_advance(uint32_t n_kv);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_paged_row_idx(ggml_context * ctx, uint32_t n_kv) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_paged_row_idx(ggml_tensor * dst) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, uint32_t visible_lo, const slot_info & sinfo) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    friend class llama_kv_cache_context;

    const llama_model & model;
    const llama_hparams & hparams;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // Stage2-exact-swapout0 scaffold. This only parses env, owns the file-backed backend
    // when explicitly enabled, and defines counters/no-op hooks. It does not touch KV tensors,
    // cell state, apply(), attention, madvise, or prefetch.
    enum class kv_swap_mode {
        off,
        exact,
        approx,
    };

    std::unique_ptr<llama_kv_backing_store_i> kv_swap_store;
    bool kv_swap_enabled = false;
    kv_swap_mode kv_swap_mode_ = kv_swap_mode::off;
    uint64_t kv_swap_out_calls = 0;
    uint64_t kv_swap_in_calls = 0;
    uint64_t kv_swap_ensure_calls = 0;
    uint32_t kv_swap_window = 0;
    uint32_t kv_swap_sink = 0;
    uint64_t kv_swap_window_calls = 0;
    uint64_t kv_swap_window_skipped = 0;
    uint64_t kv_swap_backend_failures = 0;
    mutable uint64_t kv_approx_calls = 0;
    uint64_t kv_approx_window = 0;
    mutable uint64_t kv_approx_masked = 0;
    mutable uint64_t kv_approx_debug_get_k_visible_gt0_calls = 0;
    mutable uint64_t kv_approx_debug_get_v_visible_gt0_calls = 0;
    mutable bool     kv_approx_dynamic_warned = false;
    bool     kv_swap_rss_sample = false;
    uint64_t kv_swap_rss_samples = 0;
    uint64_t kv_swap_rss_min_kb = 0;
    uint64_t kv_swap_rss_max_kb = 0;
    uint64_t kv_swap_rss_last_kb = 0;
    bool     kv_swap_madvise = false;
    uint64_t kv_swap_madvise_calls = 0;
    uint64_t kv_swap_madvise_candidate_runs = 0;
    uint64_t kv_swap_madvise_advised_runs = 0;
    uint64_t kv_swap_madvise_advised_bytes = 0;
    uint64_t kv_swap_madvise_failures = 0;
    uint64_t kv_swap_madvise_skipped_bytes = 0;

    void swap_out_cell(uint32_t cell);
    void swap_in_cell(uint32_t cell);
    void madvise_swapped_runs(uint32_t n_kv);
    void kv_swap_roundtrip_selftest();

    // Stage 1 paged KV metadata scaffold. Off unless LLAMA_KV_PAGED=1 and only maintains an
    // identity block table for internal accounting; it is not consumed by KV read/write paths.
    void paged_init(uint32_t kv_size);
    void paged_reset();
    void paged_build_block_table();
    void paged_note_cells(const slot_info & sinfo);
    uint32_t paged_resolve(uint32_t cell) const;
    uint32_t paged_write_resolve(uint32_t cell) const;
    void paged_assert_identity(const slot_info & sinfo);
    void paged_shadow_validate(const slot_info & sinfo, uint32_t n_kv) const;
    bool paged_ingraph_gather_supported(int32_t il) const;
    void paged_log_stats() const;

    static constexpr uint32_t PAGED_BLOCK_INVALID = UINT32_MAX;

    bool     kv_paged_enabled  = false;
    bool     kv_paged_warned   = false;
    uint32_t paged_block_size  = 16;
    uint32_t paged_n_blocks    = 0;
    uint32_t paged_kv_size     = 0;
    uint32_t paged_shift       = 0;
    bool     paged_non_identity_enabled = false;
    std::vector<uint32_t> paged_block_table;
    std::vector<uint8_t>  paged_block_used;
    std::vector<uint32_t> paged_free_list;
    uint64_t paged_alloc_calls     = 0;
    uint64_t paged_blocks_in_use   = 0;
    uint64_t paged_identity_checks = 0;
    uint64_t paged_identity_fail   = 0;
    mutable uint64_t paged_write_resolve_checks  = 0;
    mutable uint64_t paged_write_resolve_fail    = 0;
    mutable uint64_t paged_write_resolve_changed = 0;
    mutable uint64_t paged_shadow_gather_calls    = 0;
    mutable uint64_t paged_shadow_gather_changed  = 0;
    mutable uint64_t paged_shadow_gather_mismatch = 0;
    mutable uint64_t paged_shadow_gather_fail     = 0;
    mutable uint64_t paged_shadow_skipped_non_identity = 0;
    mutable uint64_t paged_ingraph_gather_layers  = 0;
    mutable uint64_t paged_row_idx_changed        = 0;
    mutable uint64_t paged_row_idx_fail           = 0;
    mutable bool     paged_ingraph_warned         = false;
    uint64_t paged_block_mapping_changed = 0;
    uint64_t paged_mapping_oob_fail = 0;
    mutable uint64_t paged_logical_to_physical_checks = 0;
    mutable uint64_t paged_logical_to_physical_fail = 0;

    // stage F1 / P1: KV Lazy-Block tail madvise. When LLAMA_KV_LAZY_TAIL=1, after n_kv is
    // known each step we advise the page-aligned interior of the *unused tail* capacity
    // [GGML_PAD(n_kv, 256), kv_size) of every layer's K/V tensor away via MADV_DONTNEED.
    // This targets capacity that is never inside the [0, n_kv) read window -> aims to lower
    // *current* RSS (not peak; peak is pinned by the construction-time buffer clear). Off by
    // default; requires !v_trans && n_stream==1. See docs/kv_lazy_block_stage_f1_design.md.
    bool     kv_lazy_tail              = false;
    bool     kv_lazy_tail_warned       = false; // unsupported-layout warning emitted once
    uint64_t lazy_tail_madvise_calls    = 0; // madvise() invocations issued
    uint64_t lazy_tail_madvise_bytes    = 0; // total page-aligned tail bytes advised away
    uint64_t lazy_tail_madvise_failures = 0; // madvise() calls that returned non-zero
    uint64_t lazy_tail_madvise_us       = 0; // cumulative time spent in the tail probe
    uint64_t lazy_tail_rss_before_kb    = 0; // /proc/self/statm RSS before first tail advise
    uint64_t lazy_tail_rss_after_kb     = 0; // /proc/self/statm RSS after most recent advise

    // stage P2: clear-frontier state. When kv_lazy_clear, only [0, clear_frontier) is ever
    // zeroed; the tail is left uncommitted to lower peak RSS. See docs/kv_lazy_block_stage_p2_read.md.
    bool     kv_lazy_clear         = false;
    bool     kv_lazy_clear_warned  = false; // unsupported-layout warning emitted once
    uint32_t clear_frontier        = 0;     // cells in [0, clear_frontier) have been zeroed
    uint64_t lazy_clear_init_bytes = 0;     // bytes zeroed at construction (prefix)
    uint64_t lazy_clear_grow_bytes = 0;     // bytes zeroed by frontier advances
    uint64_t lazy_clear_skipped_bytes = 0;  // tail bytes left uncleared at construction
    uint64_t lazy_clear_calls      = 0;     // frontier-advance invocations that zeroed rows
    uint64_t lazy_clear_us         = 0;     // cumulative time spent zeroing

    // current process RSS in KiB from /proc/self/statm (0 if unavailable).
    uint64_t get_current_rss_kb() const;
    // peak process RSS in KiB from /proc/self/status VmHWM (0 if unavailable).
    uint64_t get_peak_rss_kb() const;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    std::vector<llama_kv_cells> v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;
    uint32_t get_visible_lo() const;
    bool uses_approx_dynamic_view() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, bool causal_attn, ggml_tensor * row_idx = nullptr) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, bool causal_attn, ggml_tensor * row_idx = nullptr) const;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_paged_row_idx(ggml_context * ctx) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_paged_row_idx(ggml_tensor * dst) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
    uint32_t visible_lo = 0;

    bool     paged_shadow_pending = false;
    uint32_t paged_shadow_n_kv    = 0;
};
