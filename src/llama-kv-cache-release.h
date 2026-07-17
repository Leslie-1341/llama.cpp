#pragma once

#include <cstdint>
#include <vector>

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
