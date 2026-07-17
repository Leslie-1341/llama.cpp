#pragma once

#include "llama-kv-cache-identity.h"

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-kv-cache-stability.h"
#include "llama-memory.h"

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <array>
#include <bitset>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <sys/types.h>
#endif

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
    // cumulative I/O volume across the lifetime of the store (grows with every swap-out/in,
    // independent of the fixed on-disk capacity below).
    uint64_t bytes_written  = 0;
    uint64_t bytes_read     = 0;
    uint64_t bytes_released = 0;
    uint64_t write_calls    = 0;
    uint64_t read_calls     = 0;
    uint64_t release_calls  = 0;
    int      last_errno     = 0;
    llama_kv_backing_store_status last_status = llama_kv_backing_store_status::ok;
    uint64_t syscall_attempts  = 0;
    uint64_t read_syscalls     = 0;
    uint64_t write_syscalls    = 0;
    uint64_t eintr_retries     = 0;
    uint64_t short_io_events   = 0;
    uint64_t terminal_failures = 0;
    // fixed logical capacity (n_slots * cell_stride) of the fixed-slot backing store. Set once
    // at construction; unlike the counters above this never grows and is preserved by reset().
    uint64_t bytes_capacity = 0;
};

struct llama_kv_backing_store_faults {
    bool read_eintr_once    = false;
    bool write_eintr_once   = false;
    bool read_short_once    = false;
    bool write_short_once   = false;
    bool read_eof_once      = false;
    bool write_enospc_once  = false;
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

    // Writes adjacent fixed physical-cell slots as one logical range. Implementations must
    // preserve the fixed-slot address mapping and may retry short/EINTR I/O internally.
    virtual llama_kv_backing_store_status write_cells(
            uint32_t     strm,
            uint32_t     begin_cell,
            uint32_t     cell_count,
            const void * data,
            size_t       total_size,
            uint64_t &   offset_out) {
        (void) strm; (void) begin_cell; (void) cell_count; (void) data; (void) total_size;
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

    virtual llama_kv_backing_store_status read_cells(
            uint32_t strm,
            uint32_t begin_cell,
            uint32_t cell_count,
            uint64_t offset,
            void *   data,
            size_t   total_size) {
        (void) strm; (void) begin_cell; (void) cell_count; (void) offset; (void) data; (void) total_size;
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

// KV-P0-B1: fixed physical-cell slot backing store.
//
// The file is a fixed-capacity (n_slots * cell_stride) sparse file. offset(cell_id) =
// cell_id * cell_stride is a pure function of the constructor arguments - there is no
// append-at-file_len allocation, so the file never grows past its initial capacity no matter
// how many times a given cell is swapped out. A repeat swap-out of the same physical cell
// overwrites its own slot in place.
class llama_kv_backing_store_file : public llama_kv_backing_store_i {
public:
    // dir:         directory the backing file is created in ("" => "/tmp").
    // n_slots:     number of fixed physical-cell slots (== KV physical cell count).
    // cell_stride: bytes per slot (== full K/V byte width of one physical cell across all
    //              layers). Must be > 0 for the store to become enabled.
    llama_kv_backing_store_file(const std::string & dir, uint32_t n_slots, size_t cell_stride);
    ~llama_kv_backing_store_file() override;

    llama_kv_backing_store_status write_cell(
            uint32_t   strm,
            uint32_t   cell,
            const void * data,
            size_t     size,
            uint64_t & offset_out) override;

    llama_kv_backing_store_status write_cells(
            uint32_t     strm,
            uint32_t     begin_cell,
            uint32_t     cell_count,
            const void * data,
            size_t       total_size,
            uint64_t &   offset_out) override;

    llama_kv_backing_store_status read_cell(
            uint32_t strm,
            uint32_t cell,
            uint64_t offset,
            void *   data,
            size_t   size) override;

    llama_kv_backing_store_status read_cells(
            uint32_t strm,
            uint32_t begin_cell,
            uint32_t cell_count,
            uint64_t offset,
            void *   data,
            size_t   total_size) override;

    llama_kv_backing_store_status release(uint64_t offset, size_t size) override;
    llama_kv_backing_store_status reset() override;

    bool is_enabled() const {
        return fd >= 0;
    }

    uint32_t get_n_slots() const {
        return n_slots;
    }

    size_t get_cell_stride() const {
        return cell_stride;
    }

    uint64_t get_capacity() const {
        return capacity;
    }

    bool used_o_tmpfile() const {
        return used_o_tmpfile_;
    }

    const std::string & get_dir() const {
        return dir;
    }

    const llama_kv_backing_store_stats & get_stats() const override {
        return stats;
    }

    void set_test_faults(const llama_kv_backing_store_faults & faults);
    void clear_test_faults();
    const llama_kv_backing_store_faults & get_test_faults() const {
        return faults;
    }

    // actual on-disk size/allocation of the backing file (fstat-based). Used to verify the
    // fixed-capacity invariant (actual_file_size() == get_capacity() at all times) and that
    // repeated in-place overwrites of the same slots do not grow disk usage.
    bool stat_actual(uint64_t & actual_file_size, uint64_t & actual_blocks_512) const;
    uint64_t get_actual_file_size() const;
    uint64_t get_actual_blocks_512() const;

private:
    int fd = -1;
    std::string dir;
    uint32_t n_slots     = 0;
    size_t   cell_stride = 0;
    uint64_t capacity    = 0;
    bool     used_o_tmpfile_ = false;
    llama_kv_backing_store_stats stats;
    llama_kv_backing_store_faults faults;

    int64_t pwrite_once(const char * data, size_t size, uint64_t offset);
    int64_t pread_once(char * data, size_t size, uint64_t offset);
    llama_kv_backing_store_status finish_status(llama_kv_backing_store_status status, int err);
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

    int32_t prefetch_seq(llama_seq_id seq_id) override;
    int32_t prefetch_seq_step(llama_seq_id seq_id, uint32_t max_blocks) override;
    void set_seq_prefetch_protected(llama_seq_id seq_id, bool enabled) override;
    void defer_idle_swapout(int32_t n_steps);
    void prefetch_seq_last_stats(
            uint64_t & owned_blocks,
            uint64_t & swapped_blocks,
            uint64_t & resident_blocks,
            uint64_t & released_blocks,
            uint64_t & invalid_cells,
            uint64_t & failures) const;
    bool paged_stability_cycle(llama_seq_id seq_id, llama_kv_stability_stats & after_swap_out, llama_kv_stability_stats & after_swap_in);
    void paged_stability_stats_for_seq(llama_seq_id seq_id, llama_kv_stability_stats & stats) const;

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
    void paged_release_blocks(uint32_t n_kv);
    void paged_swap_out_window(uint32_t n_kv);

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
    bool uses_paged_row_idx() const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    bool set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    bool set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    bool set_input_paged_row_idx(ggml_tensor * dst, const llama_ubatch * ubatch) const;

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
    uint32_t paged_write_resolve_input(uint32_t cell, const llama_ubatch * ubatch, uint32_t token) const;
    bool paged_validate_write_mapping(uint32_t phys_cell) const;
    bool paged_test_mapping_fault_token_matches(const llama_ubatch * ubatch, uint32_t token) const;
    bool paged_test_mapping_fault_should_inject_read(uint32_t logical_cell) const;
    bool paged_test_mapping_fault_should_inject_write(const llama_ubatch * ubatch, uint32_t token) const;
    void paged_test_mapping_fault_log_and_consume(
            bool read_scope,
            uint32_t logical_cell,
            llama_paged_swap_error_reason reason,
            uint64_t fatal_counter_next) const;
    bool paged_ensure_write_resident(uint32_t phys_cell) const;
    bool paged_check_read_resident(uint32_t phys_cell, bool required_by_active) const;
    bool paged_check_read_resident_impl(uint32_t phys_cell, bool required_by_active) const;
    void paged_swap_out_block(uint32_t physical_block, bool do_madvise = true) const;
    void paged_swap_out_block_impl(uint32_t physical_block, bool do_madvise) const;
    bool paged_swap_in_block(
            uint32_t physical_block,
            bool fatal_on_failure,
            llama_paged_swap_error_reason failure_reason) const;
    struct paged_release_range {
        void * addr = nullptr;
        size_t len = 0;
        uint64_t before_resident = 0;
        uint32_t block = UINT32_MAX;
    };
    void paged_finish_write_transaction(bool success, const slot_info & sinfo);
    uint64_t paged_sample_release_ranges(
            const std::vector<paged_release_range> & ranges,
            uint64_t * total_bytes = nullptr) const;
    void paged_verify_fresh_writes(const slot_info & sinfo) const;
    void clear_paged_swap_error();
    void set_paged_swap_error(
            llama_paged_swap_error_reason reason,
            uint32_t physical_block,
            uint32_t physical_cell,
            int backend_status = 0,
            int backend_errno = 0) const;
    bool has_paged_swap_error() const;
    llama_paged_swap_error get_paged_swap_error() const;
    uint64_t paged_madvise_block(
            uint32_t physical_block,
            const std::vector<uint8_t> * active,
            uint64_t & failures,
            uint64_t & skipped,
            uint64_t & skip_live,
            std::vector<paged_release_range> * advised_ranges = nullptr) const;
    // Stage 5E-1: read-only KV resident page sampling via mincore(2). Walks every KV layer's
    // K/V tensor, page-aligns each tensor's [data, data+nbytes) interval (same align rule as
    // paged_madvise_block), and counts resident pages. Updates the kv_mincore_* counters and
    // returns total resident bytes across all KV tensors. No-op (returns 0) unless
    // paged_mincore_enabled. Does not touch tensor contents or block state.
    uint64_t paged_sample_mincore() const;

    // Stage 7D-A: debug-only SWAPPED-page refault tracing. When LLAMA_KV_PAGED_REFAULT_TRACE=1,
    // a block that has been swapped out + madvise'd has its K/V page ranges mprotect(PROT_NONE)'d
    // (same page-aligned interior as paged_madvise_block, so only pages fully owned by the block
    // are touched). Any later read/write to those pages -- e.g. the decode graph's
    // ggml_get_rows(k2d/v2d, row_idx) -- traps into a SIGSEGV handler that records the fault site
    // (K/V, layer, block, step), restores the page to PROT_READ|PROT_WRITE, and returns so the
    // faulting instruction retries. This is a diagnostic, NOT a memory-optimization mechanism: it
    // does not change swap/madvise/row_idx semantics and is a no-op unless explicitly enabled.
    void paged_refault_init();
    void paged_refault_protect_block(uint32_t physical_block) const;
    void paged_refault_unprotect_block(uint32_t physical_block) const;
    void paged_refault_unprotect_all() const;
    void paged_refault_drain() const;

    void paged_assert_identity(const slot_info & sinfo);
    void paged_shadow_validate(const slot_info & sinfo, uint32_t n_kv) const;
    bool paged_ingraph_gather_supported(int32_t il) const;
    void paged_log_base_timing() const;
    void paged_log_timing() const;
    void paged_log_stats() const;

    static constexpr uint32_t PAGED_BLOCK_INVALID = UINT32_MAX;

    // RELEASED is destructive: its tensor pages have no authoritative backing and old KV can
    // never be recovered. Only a block with no live/owned cell may enter it; later reuse is a
    // fresh allocation populated by new K/V writes. Recoverable idle/shared history is SWAPPED.
    enum class paged_block_state : uint8_t {
        UNUSED   = 0,
        RESIDENT = 1,
        RELEASED = 2,
        SWAPPED  = 3,
        PENDING_WRITE = 4,
    };

    bool     kv_paged_enabled  = false;
    bool     kv_paged_warned   = false;
    bool     paged_ingraph_enabled = true;
    bool     paged_row_idx_enabled = false;
    bool     paged_nonidentity_probe_requested = false;
    bool     paged_identity_fast_path_enabled = false;
    uint32_t paged_identity_fast_path_layers = 0;
    llama_kv_paged_identity_fast_path_reject paged_identity_fast_path_reject =
        llama_kv_paged_identity_fast_path_reject::NOT_REQUESTED;
    uint32_t paged_block_size  = 16;
    uint32_t paged_n_blocks    = 0;
    uint32_t paged_kv_size     = 0;
    uint32_t paged_shift       = 0;
    bool     paged_non_identity_enabled = false;
    std::vector<uint32_t> paged_block_table;
    std::vector<uint8_t>  paged_block_used;
    mutable std::vector<paged_block_state> paged_block_states;
    mutable std::vector<uint64_t> paged_swap_offsets;
    mutable std::vector<size_t>   paged_swap_sizes;
    mutable std::vector<uint8_t>  paged_pending_write_cells;
    mutable std::vector<uint32_t> paged_pending_write_blocks;
    mutable std::vector<paged_release_range> paged_release_post_ranges;
    mutable std::vector<std::vector<paged_release_range>> paged_released_ranges_by_block;
    std::vector<uint32_t> paged_free_list;
    uint64_t paged_alloc_calls     = 0;
    uint64_t paged_blocks_in_use   = 0;
    uint64_t paged_identity_checks = 0;
    uint64_t paged_identity_fail   = 0;
    mutable uint64_t paged_write_resolve_checks  = 0;
    mutable uint64_t paged_write_resolve_fail    = 0;
    mutable uint64_t paged_write_resolve_changed = 0;
    mutable uint64_t paged_test_mapping_fault_triggers       = 0;
    mutable uint64_t paged_test_mapping_fault_read_triggers  = 0;
    mutable uint64_t paged_test_mapping_fault_write_triggers = 0;
    mutable uint64_t paged_shadow_gather_calls    = 0;
    mutable uint64_t paged_shadow_gather_changed  = 0;
    mutable uint64_t paged_shadow_gather_mismatch = 0;
    mutable uint64_t paged_shadow_gather_fail     = 0;
    mutable uint64_t paged_shadow_skipped_non_identity = 0;
    // Stage 7D-B: state-aware shadow validation. paged_shadow_validate() used to read every
    // row's raw K/V tensor memory unconditionally, refaulting SWAPPED pages back to resident
    // (and polluting Stage 7C-G residency conclusions). These count the SWAPPED-aware skips.
    bool     paged_shadow_validate_enabled = false;
    mutable uint64_t paged_shadow_validate_calls          = 0;
    mutable uint64_t paged_shadow_validate_blocks_checked = 0;
    mutable uint64_t paged_shadow_validate_swapped_blocks_skipped = 0;
    mutable uint64_t paged_shadow_validate_fault_risk_skipped    = 0;
    mutable uint64_t paged_shadow_validate_bytes_skipped         = 0;
    mutable uint64_t paged_ingraph_gather_layers  = 0;
    mutable uint64_t paged_row_idx_inputs_created = 0;
    mutable uint64_t paged_row_idx_set_calls      = 0;
    mutable uint64_t paged_row_idx_changed        = 0;
    mutable uint64_t paged_row_idx_fail           = 0;
    mutable bool     paged_ingraph_warned         = false;
    uint64_t paged_block_mapping_changed = 0;
    uint64_t paged_mapping_oob_fail = 0;
    mutable uint64_t paged_logical_to_physical_checks = 0;
    mutable uint64_t paged_logical_to_physical_fail = 0;
    bool     paged_block_release_enabled = false;
    bool     paged_block_release_requested = false;
    uint64_t paged_block_release_calls = 0;
    uint64_t paged_blocks_released = 0;
    uint64_t paged_blocks_released_unused = 0;
    uint64_t paged_block_release_bytes = 0;
    uint64_t paged_block_release_blocks_last = 0;
    uint64_t paged_block_release_bytes_last = 0;
    uint64_t paged_block_release_skip_live = 0;
    uint64_t paged_block_release_skip_owned = 0;
    uint64_t paged_block_release_skip_shared = 0;
    uint64_t paged_block_release_ownership_invalid = 0;
    uint64_t paged_block_release_metadata_cleared = 0;
    uint64_t paged_block_release_metadata_stale = 0;
    uint64_t paged_block_release_idempotent = 0;
    uint64_t paged_blocks_released_dead = 0;
    uint64_t paged_block_release_skip_unaligned = 0;
    uint64_t paged_block_release_fail = 0;
    uint64_t paged_block_release_rss_samples = 0;
    uint64_t paged_block_release_rss_before_last_kb = 0;
    uint64_t paged_block_release_rss_after_last_kb = 0;
    uint64_t paged_block_release_rss_before_max_kb = 0;
    uint64_t paged_block_release_rss_after_min_kb = 0;
    uint64_t paged_block_release_rss_drop_last_kb = 0;
    uint64_t paged_block_release_rss_drop_max_kb = 0;
    mutable uint64_t paged_block_ensure_calls = 0;
    mutable uint64_t paged_block_ensure_released = 0;
    uint64_t paged_block_release_reuse_allocations = 0;
    mutable uint64_t paged_release_violation = 0;
    mutable uint64_t paged_active_release_violation = 0;
    mutable uint64_t paged_padded_release_violation = 0;
    mutable uint64_t paged_released_redirect_rows = 0;
    mutable uint64_t paged_released_redirect_blocks = 0;
    mutable uint64_t paged_released_redirect_no_dummy = 0;
    mutable uint64_t paged_release_mincore_samples = 0;
    mutable uint64_t paged_release_mincore_before_last = 0;
    mutable uint64_t paged_release_mincore_after_last = 0;
    mutable uint64_t paged_release_mincore_post_graph_last = 0;
    mutable uint64_t paged_release_mincore_total_last = 0;
    mutable uint64_t paged_release_mincore_drop_max = 0;
    mutable uint64_t paged_release_mincore_reaccess_last = 0;
    mutable uint64_t paged_release_mincore_reaccess_total_last = 0;
    mutable uint64_t paged_release_write_commits = 0;
    mutable uint64_t paged_release_write_rollbacks = 0;
    mutable uint64_t paged_release_fresh_verify_bytes = 0;
    mutable uint64_t paged_release_fresh_verify_hash = 1469598103934665603ULL;
    bool paged_release_fresh_verify_enabled = false;
    bool paged_release_test_repeat = false;
    bool paged_release_test_repeat_active = false;
    bool paged_release_test_reuse = false;
    bool paged_release_test_repeat_marker_emitted = false;
    bool paged_release_test_reuse_marker_emitted = false;
    bool paged_release_test_r5_marker_emitted = false;
    uint32_t paged_release_test_reuse_block = PAGED_BLOCK_INVALID;
    bool paged_release_test_reuse_pending_seen = false;
    bool paged_release_test_reuse_metadata_absent = false;
    uint64_t paged_release_test_reuse_fresh_bytes_before = 0;
    mutable uint64_t paged_release_test_repeat_passes = 0;
    mutable uint64_t paged_release_test_reuse_commits = 0;
    bool paged_test_force_active_release = false;
    llama_seq_id paged_test_force_active_release_seq = -1;
    mutable bool paged_test_force_active_release_consumed = false;
    mutable uint64_t paged_test_force_active_release_triggers = 0;
    bool     paged_swap_enabled = false;
    mutable uint64_t paged_swap_out_calls = 0;
    mutable uint64_t paged_swap_in_calls = 0;
    mutable uint64_t paged_blocks_swapped_out = 0;
    mutable uint64_t paged_blocks_swapped_in = 0;
    mutable uint64_t paged_swap_bytes_out = 0;
    mutable uint64_t paged_swap_bytes_in = 0;
    mutable uint32_t paged_swap_in_last_block = PAGED_BLOCK_INVALID;
    mutable uint64_t paged_swap_backend_failures = 0;
    mutable uint64_t paged_swap_window_skipped = 0;
    mutable uint64_t paged_swap_read_swapped_hits = 0;
    mutable uint64_t paged_swap_read_swap_in_calls = 0;
    mutable uint64_t paged_swap_read_swap_in_failures = 0;
    mutable uint64_t paged_swap_write_swapped_hits = 0;
    mutable uint64_t paged_swap_write_swap_in_calls = 0;
    mutable uint64_t paged_swap_write_swap_in_failures = 0;
    mutable uint64_t paged_swap_in_fail_no_offset = 0;
    mutable uint64_t paged_swap_in_fail_bad_size = 0;
    mutable uint64_t paged_swap_in_fail_read_cell = 0;
    mutable uint64_t paged_swap_in_fail_tensor_set = 0;
    mutable uint64_t paged_swap_madvise_calls = 0;
    mutable uint64_t paged_swap_madvise_bytes = 0;
    mutable uint64_t paged_swap_madvise_failures = 0;
    mutable uint64_t paged_swap_madvise_skipped = 0;
    mutable uint64_t paged_swap_madvise_skip_no_full_page = 0;
    mutable uint64_t paged_swap_madvise_skip_neighbor = 0;
    mutable uint64_t paged_swap_rss_samples = 0;
    mutable uint64_t paged_swap_rss_before_last_kb = 0;
    mutable uint64_t paged_swap_rss_after_last_kb = 0;
    mutable uint64_t paged_swap_rss_drop_last_kb = 0;
    mutable uint64_t paged_swap_rss_drop_max_kb = 0;
    // cumulative RSS telemetry across the whole idle-swap madvise window (Stage 5C-scale-B).
    // before_first: RSS before the FIRST madvise sample (recorded once, never overwritten).
    // total_drop:   max(0, before_first - after_last) — net RSS change over the window.
    // drop_sum:     sum of per-call max(0, before - after) — accumulated local positive drops.
    mutable uint64_t paged_swap_rss_before_first_kb = 0;
    mutable bool     paged_swap_rss_before_first_set = false;
    mutable uint64_t paged_swap_rss_total_drop_kb = 0;
    mutable uint64_t paged_swap_rss_drop_sum_kb = 0;
    mutable uint64_t paged_prefetch_seq_calls = 0;
    mutable uint64_t paged_prefetch_seq_blocks = 0;
    mutable uint64_t paged_prefetch_seq_bytes = 0;
    mutable uint64_t paged_prefetch_seq_skip_resident = 0;
    mutable uint64_t paged_prefetch_seq_skip_released = 0;
    mutable uint64_t paged_prefetch_seq_failures = 0;
    mutable uint64_t paged_prefetch_seq_last_owned_blocks = 0;
    mutable uint64_t paged_prefetch_seq_last_swapped_blocks = 0;
    mutable uint64_t paged_prefetch_seq_last_resident_blocks = 0;
    mutable uint64_t paged_prefetch_seq_last_released_blocks = 0;
    mutable uint64_t paged_prefetch_seq_last_invalid_cells = 0;
    mutable uint64_t paged_prefetch_seq_last_failures = 0;
    bool     paged_io_stats_enabled = false;
    // Diagnostic-only per-prefetch-call/block phase events. Off unless
    // LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=1; the default path emits nothing.
    bool     paged_prefetch_phase_trace_enabled = false;
    mutable uint64_t paged_prefetch_phase_trace_calls = 0;
    mutable uint64_t paged_io_swap_out_latency_us = 0;
    mutable uint64_t paged_io_swap_in_latency_us = 0;
    mutable uint64_t paged_io_swap_out_latency_max_us = 0;
    mutable uint64_t paged_io_swap_in_latency_max_us = 0;
    mutable uint64_t paged_io_swap_out_timed_calls = 0;
    mutable uint64_t paged_io_swap_in_timed_calls = 0;
    mutable uint64_t paged_io_staging_buffer_bytes = 0;
    mutable uint64_t paged_io_block_out_validate_us = 0;
    mutable uint64_t paged_io_block_out_validate_calls = 0;
    mutable uint64_t paged_io_block_out_pack_us = 0;
    mutable uint64_t paged_io_block_out_pack_calls = 0;
    mutable uint64_t paged_io_block_out_write_us = 0;
    mutable uint64_t paged_io_block_out_write_calls = 0;
    mutable uint64_t paged_io_block_out_metadata_us = 0;
    mutable uint64_t paged_io_block_out_metadata_calls = 0;
    mutable uint64_t paged_io_block_out_madvise_us = 0;
    mutable uint64_t paged_io_block_out_madvise_calls = 0;
    mutable uint64_t paged_io_block_in_validate_us = 0;
    mutable uint64_t paged_io_block_in_validate_calls = 0;
    mutable uint64_t paged_io_block_in_read_us = 0;
    mutable uint64_t paged_io_block_in_read_calls = 0;
    mutable uint64_t paged_io_block_in_unpack_us = 0;
    mutable uint64_t paged_io_block_in_unpack_calls = 0;
    mutable uint64_t paged_io_block_in_commit_us = 0;
    mutable uint64_t paged_io_block_in_commit_calls = 0;
    // The paged paths execute synchronously. Keep grow-only cell-major staging so a normal
    // 16-cell swap does not allocate/free 4 MiB per operation.
    mutable std::vector<uint8_t> paged_io_staging;

    enum class paged_test_swapin_fail_scope : uint8_t {
        OFF,
        PREFETCH,
        ACTIVE,
    };

    struct paged_test_swapin_fault {
        paged_test_swapin_fail_scope scope = paged_test_swapin_fail_scope::OFF;
        uint64_t fail_after_cells = 0;
        bool fail_once = true;
        bool consumed = false;
        uint64_t matching_attempts = 0;
        uint64_t trigger_count = 0;
        uint64_t prefetch_trigger_count = 0;
        uint64_t active_trigger_count = 0;
    };

    // KV-P0-B2B-1: context-local, environment-configured test fault for paged swap-in reads.
    // Parsed once during construction and off by default. This is deliberately separate from
    // the backing store so exact swap and backing-store telemetry are unaffected.
    mutable paged_test_swapin_fault paged_test_swapin_fault_;

    enum class paged_test_mapping_fail_scope : uint8_t {
        OFF,
        READ,
        WRITE,
    };

    struct paged_test_mapping_fault {
        paged_test_mapping_fail_scope scope = paged_test_mapping_fail_scope::OFF;
        llama_seq_id target_seq = -1;
        bool fail_once = true;
        bool consumed = false;
    };

    // Context-local deterministic test fault for paged logical->physical mapping. Off by
    // default and parsed once at construction; production mapping paths only read this state.
    mutable paged_test_mapping_fault paged_test_mapping_fault_;

    enum class paged_test_io_fail_scope : uint8_t {
        OFF,
        SWAP_OUT,
        ACTIVE_SWAP_IN,
    };

    enum class paged_test_io_fail_kind : uint8_t {
        NONE,
        WRITE_ENOSPC_ONCE,
        READ_EOF_ONCE,
    };

    struct paged_test_io_fault {
        paged_test_io_fail_scope scope = paged_test_io_fail_scope::OFF;
        paged_test_io_fail_kind kind = paged_test_io_fail_kind::NONE;
        llama_seq_id target_seq = -1;
        bool fail_once = true;
        bool consumed = false;
        uint64_t matching_attempts = 0;
        uint64_t trigger_count = 0;
        uint32_t failed_block = PAGED_BLOCK_INVALID;
        uint64_t failed_attempt_id = 0;
        uint64_t retry_success_count = 0;
    };

    // KV-P0-B4B: context-local model-level backing-store I/O fault. The cache layer decides
    // when the target seq/block is on a real paged swap path, then arms the store-local B4A
    // one-shot fault immediately before calling write_cell/read_cell.
    mutable paged_test_io_fault paged_test_io_fault_;

    // Synchronous decode-path error latch for paged KV swap-in. This is diagnostic/control
    // state only, not a second block-state machine; RESIDENT/SWAPPED/RELEASED/UNUSED remain
    // authoritative. There is no async worker in this path, so no mutex/atomic is used.
    // process_ubatch clears this exactly once at ubatch start and checks it before graph_compute.
    mutable llama_paged_swap_error paged_swap_error;

    // Stage 5E-1: read-only KV resident page telemetry via mincore(2). Off unless
    // LLAMA_KV_PAGED_MINCORE=1 (Linux + CPU + kv_paged_enabled && !v_trans && n_stream==1).
    // These counters never feed back into swap/madvise/state-machine decisions.
    bool     paged_mincore_requested = false;
    mutable bool     paged_mincore_enabled = false;
    mutable bool     paged_mincore_warned  = false;
    mutable uint64_t paged_mincore_sample_calls = 0;
    mutable uint64_t paged_mincore_failures = 0;
    // last-sample aggregate (overwritten each sample)
    mutable uint64_t paged_mincore_total_bytes = 0;
    mutable uint64_t paged_mincore_resident_bytes = 0;
    mutable uint64_t paged_mincore_total_pages = 0;
    mutable uint64_t paged_mincore_resident_pages = 0;
    mutable uint64_t paged_mincore_k_total_bytes = 0;
    mutable uint64_t paged_mincore_k_resident_bytes = 0;
    mutable uint64_t paged_mincore_v_total_bytes = 0;
    mutable uint64_t paged_mincore_v_resident_bytes = 0;
    // snapshots at the four sample points (0 if that point never fired)
    mutable uint64_t paged_mincore_prefill_resident_bytes = 0;
    mutable bool     paged_mincore_prefill_set = false;
    mutable uint64_t paged_mincore_before_madvise_resident_bytes = 0;
    mutable bool     paged_mincore_before_madvise_set = false;
    mutable uint64_t paged_mincore_after_madvise_resident_bytes = 0;
    mutable uint64_t paged_mincore_after_resume_resident_bytes = 0;
    // Stage 7C-C: per-block residency of currently-SWAPPED blocks (overwritten each sample).
    // Detects SWAPPED blocks whose K/V pages were re-touched back to resident after swap-out.
    mutable uint64_t paged_mincore_swapped_block_count = 0;
    mutable uint64_t paged_mincore_swapped_total_bytes = 0;
    mutable uint64_t paged_mincore_swapped_resident_bytes = 0;
    mutable uint64_t paged_mincore_swapped_nonresident_bytes = 0;
    mutable uint64_t paged_mincore_swapped_resident_blocks = 0;
    mutable uint64_t paged_mincore_swapped_nonresident_blocks = 0;
    bool     paged_swap_pending = false;
    uint32_t paged_swap_pending_n_kv = 0;

    // Stage 7D-A: debug-only refault tracing state. All off unless LLAMA_KV_PAGED_REFAULT_TRACE=1.
    // A trap range maps a contiguous page-aligned [lo, hi) host interval back to (block, kind,
    // layer) so the (async-signal-safe) handler can identify the fault and restore the page.
    struct paged_refault_range {
        uintptr_t lo;          // page-aligned start of the protected interval
        uintptr_t hi;          // page-aligned end (exclusive)
        uint32_t  block;       // owning physical block
        uint32_t  layer_il;    // KV layer id
        uint8_t   is_v;        // 0 = K tensor, 1 = V tensor
    };
    bool     paged_refault_trace_requested = false;
    mutable bool     paged_refault_trace_enabled = false;
    bool     paged_refault_trace_backtrace = false;
    bool     paged_refault_trace_once = true;   // unprotect a page on first fault (default on)
    uint64_t paged_refault_trace_max = 64;      // max faults logged before tracing self-disables
    // Per-block protection bookkeeping. paged_refault_protected[block] != 0 means the block's
    // pages are currently PROT_NONE. Indexed by physical block id, sized paged_n_blocks.
    mutable std::vector<uint8_t> paged_refault_protected;
    mutable uint64_t paged_refault_trace_enabled_flag = 0; // mirrors enabled for trace line
    mutable uint64_t paged_refault_fault_count = 0;
    mutable uint64_t paged_refault_fault_k_count = 0;
    mutable uint64_t paged_refault_fault_v_count = 0;
    mutable uint64_t paged_refault_fault_blocks = 0;       // distinct blocks that faulted
    mutable uint64_t paged_refault_unmapped_fault_count = 0; // faults not in any KV trap range
    mutable uint64_t paged_refault_protect_calls = 0;
    mutable uint64_t paged_refault_unprotect_calls = 0;
    mutable uint64_t paged_refault_protected_pages = 0;
    mutable uint64_t paged_refault_unprotected_pages = 0;
    mutable uint64_t paged_refault_protect_failures = 0;
    mutable uint64_t paged_refault_unprotect_failures = 0;

    // Stage 4C-3: idle-seq and block-ownership telemetry only.
    bool     paged_idle_trace_enabled = false;
    mutable std::array<uint64_t, LLAMA_MAX_SEQ> paged_idle_seq_last_active_step = {};
    mutable std::bitset<LLAMA_MAX_SEQ> paged_idle_seq_seen;
    // Stage 8D-3: gated resume timing telemetry. Off unless
    // LLAMA_KV_PAGED_RESUME_TIMING=1; counters are emitted once from the dtor path.
    bool     paged_resume_timing_enabled = false;
    // Stage 8D-4: gated per-step resume timing telemetry. Off unless
    // LLAMA_KV_PAGED_RESUME_TIMING_STEP=1; emitted once per paged row_idx fill.
    bool     paged_resume_timing_step_enabled = false;
    // Stage 11-B-D: low-frequency paged base-path timing telemetry. Off unless
    // LLAMA_KV_PAGED_TIMING=1; counters are emitted once from the dtor path.
    bool     paged_base_timing_enabled = false;
    mutable uint64_t paged_base_timing_getenv_calls = 0;
    mutable uint64_t paged_base_timing_apply_calls = 0;
    mutable uint64_t paged_base_timing_apply_paged_total_us = 0;
    mutable uint64_t paged_base_timing_apply_ubatch_us = 0;
    mutable uint64_t paged_base_timing_note_cells_us = 0;
    mutable uint64_t paged_base_timing_assert_identity_us = 0;
    mutable uint64_t paged_base_timing_swap_out_window_us = 0;
    mutable uint64_t paged_base_timing_ensure_resident_us = 0;
    mutable uint64_t paged_base_timing_clear_frontier_us = 0;
    mutable uint64_t paged_base_timing_madvise_tail_us = 0;
    mutable uint64_t paged_base_timing_paged_release_blocks_us = 0;
    mutable uint64_t paged_base_timing_set_row_idx_calls = 0;
    mutable uint64_t paged_base_timing_set_row_idx_total_us = 0;
    mutable uint64_t paged_base_timing_active_visible_us = 0;
    mutable uint64_t paged_base_timing_nonidentity_probe_us = 0;
    mutable uint64_t paged_base_timing_swapped_blocks_scan_us = 0;
    mutable uint64_t paged_base_timing_row_idx_fill_us = 0;
    mutable uint64_t paged_base_timing_check_read_resident_us = 0;
    mutable uint64_t paged_base_timing_check_read_resident_calls = 0;
    mutable uint64_t paged_base_timing_paged_resolve_calls = 0;
    mutable uint64_t paged_base_timing_cells_scanned = 0;
    mutable uint64_t paged_base_timing_blocks_scanned = 0;
    mutable uint64_t paged_base_timing_row_idx_entries = 0;
    mutable uint64_t paged_timing_set_input_us = 0;
    mutable uint64_t paged_timing_set_input_calls = 0;
    mutable uint64_t paged_timing_idle_maintenance_us = 0;
    mutable uint64_t paged_timing_idle_maintenance_calls = 0;
    mutable uint64_t paged_timing_swap_out_us = 0;
    mutable uint64_t paged_timing_swap_out_calls = 0;
    mutable uint64_t paged_timing_check_read_us = 0;
    mutable uint64_t paged_timing_check_read_calls = 0;
    // Stage 6C-1A: seqs marked prefetch-protected (resume-pending) are excluded from idle
    // swap-out victim selection so interleaved prefetch is not undone by the same-step idle
    // gate. Does not change read-window / nonidentity / state-machine semantics.
    std::bitset<LLAMA_MAX_SEQ> paged_prefetch_protected_seq;
    mutable uint64_t paged_idle_swap_skip_protected = 0;
    mutable uint64_t paged_idle_active_seq_steps = 0;
    mutable uint64_t paged_idle_active_seq_empty = 0;
    mutable uint64_t paged_idle_seq_seen_count = 0;
    mutable uint64_t paged_idle_active_seq_count_last = 0;
    mutable uint64_t paged_idle_idle_seq_count_last = 0;
    mutable uint64_t paged_idle_active_seq_count_max = 0;
    mutable uint64_t paged_idle_non_empty_blocks = 0;
    mutable uint64_t paged_idle_single_seq_blocks = 0;
    mutable uint64_t paged_idle_multi_seq_blocks = 0;
    mutable uint64_t paged_idle_blocks_with_active_seq = 0;
    mutable uint64_t paged_idle_blocks_without_active_seq = 0;
    mutable uint64_t paged_idle_cold_candidates = 0;
    mutable uint64_t paged_idle_read_window_blocks = 0;
    mutable uint64_t paged_idle_cold_in_read_window = 0;
    mutable uint64_t paged_idle_cold_not_in_read_window = 0;
    mutable uint64_t paged_idle_skip_mixed_active = 0;
    mutable uint64_t paged_idle_safe_swap_candidates = 0;
    mutable uint64_t paged_cov_idle_owned_blocks = 0;
    mutable uint64_t paged_cov_in_read_window_blocks = 0;
    mutable uint64_t paged_cov_not_in_read_window_blocks = 0;
    mutable uint64_t paged_cov_resident_safe_blocks = 0;
    mutable uint64_t paged_cov_nonidentity_remapped_blocks = 0;
    mutable uint64_t paged_cov_idle_owned_bytes = 0;
    mutable uint64_t paged_cov_in_read_window_bytes = 0;
    mutable uint64_t paged_cov_resident_safe_bytes = 0;
    mutable uint64_t paged_cov_nonidentity_remapped_bytes = 0;
    bool     paged_idle_swap_requested = false;
    bool     paged_idle_swap_madvise_requested = false;
    mutable bool     paged_idle_swap_enabled = false;
    mutable bool     paged_idle_swap_madvise_enabled = false;
    mutable bool     paged_idle_swap_warned = false;
    mutable bool     paged_idle_swap_madvise_warned = false;
    uint64_t paged_idle_swap_every_tokens = 1;
    uint64_t paged_idle_swap_max_blocks_per_step = 0;
    uint64_t paged_idle_swap_min_idle_steps = 0;
    bool     paged_idle_swap_debug_probes = true;
    mutable uint64_t paged_idle_swap_candidates = 0;
    mutable uint64_t paged_idle_swap_out_calls = 0;
    mutable uint64_t paged_idle_swap_skip_not_remapped = 0;
    mutable uint64_t paged_idle_swap_skip_not_resident = 0;
    mutable uint64_t paged_idle_swap_skip_deferred = 0;
    mutable uint64_t paged_idle_swap_skip_min_idle = 0;
    mutable int32_t  paged_defer_idle_swapout_steps = 0;
    mutable bool     paged_nonidentity_probe_enabled = false;
    mutable uint64_t paged_nonidentity_remap_rows = 0;
    mutable uint64_t paged_nonidentity_remap_blocks = 0;
    mutable uint64_t paged_nonidentity_skip_no_dummy = 0;
    mutable uint64_t paged_nonidentity_skip_not_masked = 0;
    mutable uint64_t paged_nonidentity_skip_not_resident = 0;
    mutable uint64_t paged_nonidentity_cold_in_read_window_before = 0;
    mutable uint64_t paged_nonidentity_cold_in_read_window_after = 0;
    mutable uint64_t paged_nonidentity_safe_candidates_after = 0;
    // Stage 7C-E: SWAPPED-block row_idx redirect. When a row maps to a physical block whose
    // state is SWAPPED and that row is not visible to / needed by the active seq, the row_idx
    // entry is redirected to a resident dummy physical row so the decode graph's ggml_get_rows
    // never touches the madvise'd SWAPPED pages (which would refault them resident). These
    // counters are telemetry only and never feed scheduling decisions.
    //   redirect_rows            : rows redirected from a real SWAPPED phys row to the dummy row.
    //   redirect_blocks          : distinct SWAPPED blocks that had >=1 row redirected this call.
    //   redirect_skip_no_dummy   : rows that should have been redirected but had no resident dummy.
    //   active_visible_violation : SWAPPED rows still visible to / needed by the active seq; this
    //                              is a swap-out / visibility bug, surfaced rather than masked.
    mutable uint64_t paged_swapped_redirect_rows = 0;
    mutable uint64_t paged_swapped_redirect_blocks = 0;
    mutable uint64_t paged_swapped_redirect_skip_no_dummy = 0;
    mutable uint64_t paged_swapped_active_visible_violation = 0;
    mutable uint64_t paged_swapped_redirect_probe_rows = 0;
    mutable uint64_t paged_swapped_redirect_probe_swapped_rows = 0;
    mutable uint64_t paged_swapped_redirect_probe_resident_rows = 0;
    mutable uint64_t paged_swapped_redirect_probe_invalid_rows = 0;
    mutable uint64_t paged_swapped_redirect_probe_state_mismatch = 0;
    mutable uint64_t paged_swapped_redirect_probe_disabled = 0;
    mutable uint64_t paged_swapped_active_visible_violation_rows = 0;
    mutable uint64_t paged_swapped_active_visible_violation_blocks = 0;
    mutable uint64_t paged_swapped_active_visible_logical_seq_has = 0;
    mutable uint64_t paged_swapped_active_visible_phys_seq_has = 0;
    mutable uint64_t paged_swapped_active_visible_in_read_window = 0;
    mutable uint64_t paged_swapped_active_visible_not_in_read_window = 0;
    mutable uint64_t paged_swapped_active_visible_masked = 0;
    mutable uint64_t paged_swapped_active_visible_unmasked = 0;
    mutable uint64_t paged_row_mapping_invalid_fatal = 0;
    mutable uint64_t paged_write_mapping_invalid_fatal = 0;
    mutable uint64_t paged_active_row_nonresident_fatal = 0;
    mutable uint64_t paged_input_setup_fatal = 0;
    mutable uint64_t paged_no_dummy_restore_sync = 0;
    mutable uint64_t paged_active_row_dummy_redirect_blocked = 0;
    // Stage 7C-F: invariant telemetry. A block that is SWAPPED must not contain rows that
    // are currently active-visible and unmasked. These counters distinguish stale active
    // ownership, later active writes, and logical resolution to an already-swapped block.
    mutable uint64_t paged_swapped_active_violation_rows = 0;
    mutable uint64_t paged_swapped_active_violation_blocks = 0;
    mutable uint64_t paged_swapped_active_violation_block_had_active_owner_at_swapout = 0;
    mutable uint64_t paged_swapped_active_violation_after_swapout_write = 0;
    mutable uint64_t paged_swapped_active_violation_resolve_to_swapped = 0;
    mutable uint64_t paged_swapped_active_visible_restore_rows = 0;
    mutable uint64_t paged_swapped_active_visible_restore_blocks = 0;
    mutable uint64_t paged_swap_out_skip_active_visible_block = 0;
    mutable uint64_t paged_swap_out_skip_active_owned_block = 0;
    // Stage 7C-G: narrowed swap-out gate telemetry. 7C-F blocked every idle swap-out by
    // treating "block appears in the full-prefix read window" as "active-visible". These
    // counters distinguish the real reasons a candidate is skipped from the over-broad
    // read-window membership, so the trace can answer: was this idle block skipped because
    // it is genuinely active-needed, or only because it fell inside the full-prefix gather?
    //   candidate_blocks                 : idle-only RESIDENT blocks that reached the gate.
    //   allowed_blocks                   : candidates that were actually swapped out.
    //   skip_fullprefix_read_window_only : candidates that were in the full-prefix read
    //                                      window yet had NO true active-needed unmasked cell,
    //                                      so 7C-G still swaps them out (the recovered class).
    //   skip_true_active_owned           : candidates whose physical block cells seq_has an
    //                                      active seq (should be 0 for idle-only candidates).
    //   skip_true_active_unmasked        : candidates with a physical cell that is active-
    //                                      visible AND unmasked vs active_seq_pos_max; the only
    //                                      correctness-mandated hard skip.
    //   active_restore_from_swapped      : SWAPPED blocks swapped back in by the graph-pre
    //                                      restore because they hold a true active-needed row.
    //   idle_only_swapped_blocks         : blocks currently SWAPPED whose owners are idle-only.
    mutable uint64_t paged_swap_out_candidate_blocks = 0;
    mutable uint64_t paged_swap_out_allowed_blocks = 0;
    mutable uint64_t paged_swap_out_skip_fullprefix_read_window_only = 0;
    mutable uint64_t paged_swap_out_skip_true_active_owned = 0;
    mutable uint64_t paged_swap_out_skip_true_active_unmasked = 0;
    mutable uint64_t paged_active_restore_from_swapped_blocks = 0;
    mutable uint64_t paged_idle_only_swapped_blocks = 0;
    mutable uint64_t paged_write_to_swapped_block = 0;
    mutable uint64_t paged_write_to_swapped_block_seq = 0;

    // Stage 4C-0: KV block access trace. When LLAMA_KV_PAGED_TRACE=1, emit one line per
    // decode step (per set_input_paged_row_idx call) to stderr describing the physical
    // blocks read/written this step plus current block-state population counts. Telemetry
    // only: it does not change paged_resolve / write paths / swap / release behavior and the
    // collection is gated behind paged_trace_enabled so the default path is untouched.
    bool     paged_trace_enabled = false;
    mutable uint64_t paged_trace_step = 0;
    // physical blocks written during the current step, collected by set_input_k/v_idxs and
    // consumed (and cleared) by the trace emit in set_input_paged_row_idx.
    mutable std::set<uint32_t> paged_trace_write_blocks;
    void paged_trace_note_write_block(uint32_t physical_block) const;
    void paged_trace_emit_step(
            uint64_t step,
            const std::set<uint32_t> & read_blocks,
            const std::set<uint32_t> & active_read_blocks,
            uint32_t n_kv,
            uint32_t active_n_kv) const;

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
    void clear_paged_swap_error() override;
    bool has_paged_swap_error() const override;
    llama_paged_swap_error get_paged_swap_error() const override;
    void finish_paged_kv_write(bool success) override;
    bool needs_paged_kv_post_graph_sync() const override;

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
    bool uses_paged_row_idx() const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    bool set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    bool set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    bool set_input_paged_row_idx(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_paged_input_setup_error() const;

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
    bool paged_write_transaction_open = false;
};
