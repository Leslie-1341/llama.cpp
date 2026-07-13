#pragma once

#include "llama.h"

#include <cstdint>

struct llama_kv_stability_stats {
    uint64_t swap_out_calls = 0;
    uint64_t swap_in_calls = 0;
    uint64_t swapped_blocks = 0;
    uint64_t resident_blocks = 0;
    uint64_t target_blocks = 0;
    uint64_t backing_stat_valid = 0;
    uint64_t backing_capacity = 0;
    uint64_t backing_size = 0;
    uint64_t backing_blocks_512 = 0;
    uint64_t fatal_counters = 0;
    uint64_t pending_error = 0;
};

extern "C" bool llama_kv_cache_paged_stability_cycle(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        llama_kv_stability_stats * after_swap_out,
        llama_kv_stability_stats * after_swap_in);

extern "C" bool llama_kv_cache_paged_stability_stats(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        llama_kv_stability_stats * stats);
