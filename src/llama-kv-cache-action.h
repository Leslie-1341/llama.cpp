#pragma once

#include "llama.h"

#include <cstdint>

enum class llama_kv_action : uint8_t {
    noop,
    evaluate,
    prefetch,
    release,
    offload,
};

enum class llama_kv_action_outcome : uint8_t {
    completed,
    no_op,
    unsupported,
    rejected,
    failed,
    partial_failure,
};

enum class llama_kv_action_reason : uint8_t {
    none,
    zero_budget,
    invalid_sequence,
    context_invalid,
    write_transaction_open,
    unsupported,
    protected_sequence,
    shared_block,
    no_eligible_block,
    state_rejected,
    ownership_invalid,
    io_failure,
    prefetch_failed,
};

struct llama_kv_action_request {
    llama_kv_action action = llama_kv_action::noop;
    uint64_t decision_id = 0;
    llama_seq_id seq_id = -1;
    uint64_t target_bytes = 0;
    uint32_t max_blocks = 0;
    bool correctness_required = false;
    // Restore every currently SWAPPED block owned by seq_id; max_blocks is
    // intentionally ignored for this explicit correctness path.
    bool all_required = false;
};

struct llama_kv_action_capability {
    bool context_invalid = false;
    bool write_transaction_open = false;
    bool can_prefetch = false;
    bool can_release = false;
    bool can_offload = false;
};

struct llama_kv_action_result {
    llama_kv_action action = llama_kv_action::noop;
    uint64_t decision_id = 0;
    uint64_t core_transaction_id = 0;
    llama_kv_action_outcome outcome = llama_kv_action_outcome::no_op;
    llama_kv_action_reason reason = llama_kv_action_reason::none;
    bool state_changed = false;
    bool fail_stop = false;
    bool io_failure = false;
    uint32_t blocks = 0;
    uint64_t bytes = 0;
    uint64_t shortfall_bytes = 0;
    llama_kv_action_capability capability;
};
