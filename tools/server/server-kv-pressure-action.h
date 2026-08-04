#pragma once

#include "llama-kv-cache-action.h"
#include "llama-kv-pressure.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

struct server_kv_pressure_unified_action_config {
    static constexpr uint32_t DEFAULT_MAX_BLOCKS = 64;

    bool enabled = false;
    uint64_t target_bytes = 0;
    uint32_t max_blocks = DEFAULT_MAX_BLOCKS;
    uint32_t cooldown_samples = 1;
    uint32_t io_failure_backoff_samples = 4;
    uint32_t max_failure_penalty = 8;
};

struct server_kv_pressure_unified_action_enablement {
    bool requested = false;
    bool legacy_requested = false;
    bool dry_run_requested = false;
    bool bounded_requested = false;
};

enum class server_kv_pressure_unified_action_startup_status {
    disabled,
    enabled,
    invalid,
    conflict,
};

struct server_kv_pressure_unified_action_startup_decision {
    server_kv_pressure_unified_action_enablement enablement;
    server_kv_pressure_unified_action_config config;
    server_kv_pressure_unified_action_startup_status status =
            server_kv_pressure_unified_action_startup_status::disabled;
    std::string error;
};

server_kv_pressure_unified_action_startup_decision
server_kv_pressure_unified_action_startup_decide_from_env();

struct server_kv_pressure_action_ops {
    std::function<llama_kv_action_result(const llama_kv_action_request &)> execute;
};

struct server_kv_pressure_action_observation {
    kv_pressure_state pressure_state = kv_pressure_state::NORMAL;
    kv_pressure_source pressure_source = kv_pressure_source::NONE;
    bool stale = false;
    bool idle = false;
    uint64_t sample_count = 0;
    uint64_t decision_id = 0;
    uint64_t target_bytes = 0;
    uint32_t max_blocks = 0;
    bool evaluate_attempted = false;
    bool release_attempted = false;
    bool offload_attempted = false;
    const char * reason = "disabled";
};

enum class server_kv_claimant_exclusion : uint8_t {
    none,
    active,
    protected_sequence,
    shared,
    write_transaction_open,
    fail_stop,
    exhausted,
    epoch_mismatch,
    empty,
};

struct server_kv_pressure_snapshot {
    kv_pressure_state state = kv_pressure_state::NORMAL;
    kv_pressure_source source = kv_pressure_source::NONE;
    bool sample_valid = false;
    bool stale = false;
    bool pressure_basis_valid = false;
    uint64_t pressure_current_bytes = 0;
    uint64_t pressure_low_water_bytes = 0;
    uint64_t pressure_basis_generation = 0;
    uint64_t sample_count = 0;
    uint64_t decision_id = 0;
};

struct server_kv_claimant_snapshot {
    llama_kv_memory_claimant claimant = llama_kv_memory_claimant::kv;
    llama_seq_id seq_id = -1;
    uint64_t epoch = 1;
    bool active = false;
    bool protected_sequence = false;
    bool shared = false;
    uint64_t idle_age_us = 0;
    uint64_t logical_kv_tokens = 0;
    uint64_t reclaimable_bytes = 0;
    uint64_t lcp_n_past_hint_tokens = 0;
    uint64_t io_cost_bytes = 0;
};

struct server_kv_claimant_score {
    llama_seq_id seq_id = -1;
    bool eligible = false;
    server_kv_claimant_exclusion exclusion = server_kv_claimant_exclusion::none;
    int64_t idle_age_score = 0;
    int64_t logical_kv_score = 0;
    int64_t reclaimable_score = 0;
    int64_t lcp_n_past_penalty = 0;
    int64_t io_cost_penalty = 0;
    int64_t failure_penalty = 0;
    int64_t total = 0;
};

struct server_kv_claimant_runtime_observation {
    llama_seq_id seq_id = -1;
    uint64_t epoch = 1;
    bool active = false;
    bool exhausted = false;
    llama_kv_runtime_claimant runtime;
};

struct server_kv_pressure_action_result;

class server_kv_governor_state {
public:
    void reset();
    uint64_t claimant_epoch(llama_seq_id seq_id) const;
    bool claimant_exhausted(llama_seq_id seq_id, uint64_t epoch) const;
    void invalidate_claimant(llama_seq_id seq_id);
    void invalidate_all_claimants();

    uint64_t pressure_debt_bytes() const { return pressure_debt_bytes_; }
    uint64_t episode() const { return episode_; }
    bool offload_armed() const { return offload_armed_; }
    uint64_t next_action_sample() const { return next_action_sample_; }

private:
    friend server_kv_pressure_action_result server_kv_pressure_execute_governor(
            const server_kv_pressure_unified_action_config &,
            server_kv_governor_state &,
            const server_kv_pressure_action_ops &,
            const server_kv_pressure_snapshot &,
            const std::vector<server_kv_claimant_snapshot> &);

    bool episode_active_ = false;
    uint64_t episode_ = 0;
    uint64_t pressure_basis_generation_ = 0;
    uint64_t pressure_debt_bytes_ = 0;
    bool offload_armed_ = false;
    uint64_t next_action_sample_ = 0;
    std::map<llama_seq_id, uint64_t> claimant_epochs_;
    std::map<llama_seq_id, uint64_t> exhausted_claimants_;
    std::map<llama_seq_id, uint32_t> failure_counts_;
};

struct server_kv_pressure_action_result {
    server_kv_pressure_action_observation observation;
    llama_kv_action_result evaluation;
    llama_kv_action_result release;
    llama_kv_action_result offload;
    uint64_t episode = 0;
    uint64_t observed_excess_bytes = 0;
    uint64_t debt_before_bytes = 0;
    uint64_t debt_after_bytes = 0;
    bool offload_armed_before = false;
    bool offload_armed_after = false;
    uint64_t next_action_sample = 0;
    llama_seq_id selected_seq_id = -1;
    uint64_t selected_claimant_epoch = 0;
    std::vector<server_kv_claimant_runtime_observation> runtime_claimants;
    std::vector<server_kv_claimant_score> scores;
};

server_kv_pressure_action_result server_kv_pressure_execute_unified_action(
        const server_kv_pressure_unified_action_config & config,
        const server_kv_pressure_action_ops & ops,
        const kv_pressure_telemetry & telemetry,
        bool idle,
        uint64_t sample_count,
        uint64_t decision_id);

server_kv_pressure_action_result server_kv_pressure_execute_governor(
        const server_kv_pressure_unified_action_config & config,
        server_kv_governor_state & state,
        const server_kv_pressure_action_ops & ops,
        const server_kv_pressure_snapshot & pressure,
        const std::vector<server_kv_claimant_snapshot> & claimants);

std::string server_kv_pressure_unified_action_format_marker(
        const server_kv_pressure_action_result & result);
