#pragma once

#include <cstdint>
#include <vector>

// Precise release capability — why paged destructive release can or cannot run.
// Used by server pressure policy to differentiate skip reasons instead of
// overloading a single boolean.
enum class llama_kv_release_status {
    available,           // Release can run: paged + ingraph + layers + row_idx + !swap
    not_paged,           // KV paging not enabled
    layout_unsupported,  // v_trans or n_stream != 1 or block_size == 0
    swap_enabled,        // Swap is active; release + swap are mutually exclusive
    disabled,            // paged_block_release_enabled is false
};

// Bounded release result — reported by both destructive and dry-run release paths.
// released_bytes may overshoot target_bytes by at most one block.
// In dry-run mode, these fields carry would-release semantics (no actual release occurred).
struct llama_kv_bounded_release_result {
    uint64_t released_bytes     = 0;
    uint64_t shortfall_bytes    = 0;
    uint64_t overshoot_bytes    = 0;
    uint32_t released_blocks    = 0;
    uint32_t blocks_scanned     = 0;
    uint32_t blocks_skipped_owned = 0;
    uint32_t blocks_skipped_state = 0;
    uint32_t madvise_failures   = 0;
    bool     block_scan_exhausted = false;
    bool     ownership_aborted    = false;
};

// Per-condition decomposition of bounded_release_can_enable() for diagnostic use.
// Each field reports whether the corresponding structural precondition is met.
// can_enable is true iff all fields are true.
struct llama_kv_bounded_release_capability {
    bool can_enable       = false;
    bool paged            = false;  // kv_paged_enabled
    bool ingraph          = false;  // paged_ingraph_enabled
    bool layers_supported = false;  // paged_layers_supported: !layers.empty() && all layers have K and V F32 tensors
    bool row_idx          = false;  // paged_row_idx_enabled: paged && ingraph && layers_supported && !identity_fast_path
    bool swap_disabled    = false;  // !paged_swap_enabled
    bool layout_supported = false;  // !v_trans && n_stream==1 && block_size>0 && n_blocks>0
};

inline bool llama_kv_destructive_release_can_enable(
        bool paged_enabled,
        bool ingraph_enabled,
        bool layers_supported,
        bool row_idx_enabled,
        bool swap_enabled) {
    return paged_enabled && ingraph_enabled && layers_supported && row_idx_enabled && !swap_enabled;
}

struct llama_kv_release_block_ownership {
    std::vector<uint8_t> owned;
    std::vector<uint8_t> shared;
    bool valid = true;
    uint64_t invalid_mappings = 0;
};

template<class CellStreams, class Resolve>
llama_kv_release_block_ownership llama_kv_release_collect_ownership(
        const CellStreams & streams,
        uint32_t n_blocks,
        uint32_t block_size,
        uint32_t invalid_cell,
        uint32_t max_sequences,
        Resolve resolve) {
    llama_kv_release_block_ownership result {
        std::vector<uint8_t>(n_blocks, 0),
        std::vector<uint8_t>(n_blocks, 0),
        true,
        0,
    };
    if (block_size == 0) {
        return result;
    }
    std::vector<int32_t> first_owner(n_blocks, -1);
    for (const auto & cells : streams) {
        for (uint32_t logical_cell = 0; logical_cell < cells.used_max_p1(); ++logical_cell) {
            if (cells.is_empty(logical_cell)) {
                continue;
            }
            const uint32_t physical_cell = resolve(logical_cell);
            if (physical_cell == invalid_cell) {
                result.valid = false;
                result.invalid_mappings += 1;
                continue;
            }
            const uint32_t block = physical_cell / block_size;
            if (block >= n_blocks) {
                result.valid = false;
                result.invalid_mappings += 1;
                continue;
            }
            result.owned[block] = 1;
            for (uint32_t seq = 0; seq < max_sequences; ++seq) {
                if (!cells.seq_has(logical_cell, (int32_t) seq)) {
                    continue;
                }
                if (first_owner[block] < 0) {
                    first_owner[block] = (int32_t) seq;
                } else if (first_owner[block] != (int32_t) seq) {
                    result.shared[block] = 1;
                }
            }
        }
    }
    return result;
}
