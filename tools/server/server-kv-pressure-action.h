#pragma once

#include "llama-kv-cache-action.h"
#include "llama-kv-pressure.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

// KV-Budget-V2 step1: soft steady-resident budget input.
//
// target_kv_resident_bytes is a high-watermark target the Governor uses to
// derive a soft budget debt.  It is INDEPENDENT from pressure debt, never
// overrides PRESSURE/CRITICAL safety semantics, and only triggers RELEASE/OFFLOAD
// when pressure is not active.  target=0 (the default) disables the soft path.
//
// source is a static string label (env_static today; HTTP/RPC future) so the
// marker can attribute which authority issued the target.
struct server_kv_resident_target_state {
    bool        enabled = false;
    uint64_t    target_bytes = 0;
    uint64_t    basis_generation = 0;       // bumped on every setter write
    const char *source = "none";            // "env_static" / "none" / future
    bool valid() const { return enabled && target_bytes > 0 && source != nullptr &&
                              std::string(source) != "none"; }
};

// Parse the step1 static target source. Invalid or incomplete input returns
// the disabled state and writes a diagnostic; callers fail closed.
server_kv_resident_target_state server_kv_resident_target_from_env(std::string & error);

// Read-only whole-KV budget view fed to the Governor's soft budget chain.
// Mirrors llama_kv_physical_budget_view fields that the Governor consumes —
// does NOT introduce new shadow counters or shadow policy state.  Reuses the
// existing core sampler; the server side only re-projects the relevant subset.
struct server_kv_budget_view {
    bool     valid = false;
    bool     resident_available = false;
    bool     reclaimable_available = false;
    bool     swapped_metadata_consistent = false;
    uint64_t object_id = 0;
    uint64_t generation = 0;
    uint64_t resident_bytes = 0;
    uint64_t dead_resident_reclaimable_bytes = 0;
    uint64_t transient_staging_bound_bytes = 0;  // advisory; not subtracted from target
};

struct server_kv_pressure_unified_action_config {
    static constexpr uint32_t DEFAULT_MAX_BLOCKS = 64;
    static constexpr uint32_t DEFAULT_BUDGET_UNMET_BACKOFF_SAMPLES = 16;

    bool enabled = false;
    uint64_t target_bytes = 0;
    uint32_t max_blocks = DEFAULT_MAX_BLOCKS;
    uint32_t cooldown_samples = 1;
    uint32_t io_failure_backoff_samples = 4;
    uint32_t max_failure_penalty = 8;

    // V2 step1: soft budget target (independent from pressure).  When
    // budget_target_enabled && budget_target_bytes > 0, the Governor derives
    // a soft debt from sample_kv_physical_budget_view().resident_bytes - target
    // and runs an additional RELEASE→armed OFFLOAD chain ONLY when pressure
    // is not in PRESSURE/CRITICAL.
    bool        budget_target_enabled = false;
    uint64_t    budget_target_bytes = 0;
    uint64_t    budget_basis_generation = 0;
    const char *budget_source = "none";
    uint32_t    budget_unmet_backoff_samples = DEFAULT_BUDGET_UNMET_BACKOFF_SAMPLES;
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
    bool idle_follow_up_pending() const { return idle_follow_up_pending_; }
    uint64_t next_action_sample() const { return next_action_sample_; }

    // V2 step1: soft budget debt (independent from pressure debt).
    uint64_t budget_debt_bytes() const { return budget_debt_bytes_; }
    uint64_t budget_basis_generation() const { return budget_basis_generation_; }
    bool soft_offload_armed() const { return soft_offload_armed_; }
    uint64_t unmet_budget_bytes() const { return unmet_budget_bytes_; }
    uint64_t budget_next_action_sample() const { return budget_next_action_sample_; }

private:
    friend server_kv_pressure_action_result server_kv_pressure_execute_governor(
            const server_kv_pressure_unified_action_config &,
            server_kv_governor_state &,
            const server_kv_pressure_action_ops &,
            const server_kv_pressure_snapshot &,
            const std::vector<server_kv_claimant_snapshot> &,
            const server_kv_budget_view &);

    bool episode_active_ = false;
    uint64_t episode_ = 0;
    uint64_t pressure_basis_generation_ = 0;
    uint64_t pressure_debt_bytes_ = 0;
    bool offload_armed_ = false;
    bool idle_follow_up_pending_ = false;
    uint64_t next_action_sample_ = 0;
    std::map<llama_seq_id, uint64_t> claimant_epochs_;
    std::map<llama_seq_id, uint64_t> exhausted_claimants_;
    std::map<llama_seq_id, uint32_t> failure_counts_;

    // V2 step1: soft budget debt state — independent of pressure_* and never
    // shares backoff/cooldown with the pressure chain.  Reset on basis
    // generation change OR when target is disabled (target_bytes==0).
    uint64_t budget_debt_bytes_ = 0;
    uint64_t budget_basis_generation_ = 0;
    bool soft_offload_armed_ = false;
    uint64_t unmet_budget_bytes_ = 0;
    uint64_t budget_next_action_sample_ = 0;
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

    // V2 step1: soft budget chain telemetry (independent from pressure debt).
    // These are populated by the V2 entry point and left at defaults by the
    // V1-only entry point — callers must not assume they are zero/disabled
    // unless they routed through the V2 path.
    bool     budget_active = false;                  // true iff soft chain entered this decision
    bool     budget_target_enabled = false;
    uint64_t budget_target_bytes = 0;
    uint64_t budget_basis_generation = 0;
    const char *budget_source = "none";
    bool     budget_view_valid = false;
    bool     budget_resident_available = false;
    bool     budget_reclaimable_available = false;
    uint64_t budget_resident_bytes = 0;             // last view.resident_bytes sampled
    uint64_t budget_dead_resident_reclaimable_bytes = 0;  // last view.dead_resident_reclaimable_bytes
    uint64_t budget_transient_staging_bound_bytes = 0;    // advisory; never subtracted
    uint64_t budget_observed_excess_bytes = 0;      // saturating_sub(resident, target)
    uint64_t budget_debt_before_bytes = 0;
    uint64_t budget_debt_after_bytes = 0;
    bool     soft_offload_armed_before = false;
    bool     soft_offload_armed_after = false;
    uint64_t budget_next_action_sample = 0;
    uint64_t unmet_budget_bytes_after = 0;
};

server_kv_pressure_action_result server_kv_pressure_execute_unified_action(
        const server_kv_pressure_unified_action_config & config,
        const server_kv_pressure_action_ops & ops,
        const kv_pressure_telemetry & telemetry,
        bool idle,
        uint64_t sample_count,
        uint64_t decision_id);

// Compatibility entry point for existing hard-pressure callers. It supplies
// an unavailable view, so only the V1 pressure chain runs.
server_kv_pressure_action_result server_kv_pressure_execute_governor(
        const server_kv_pressure_unified_action_config & config,
        server_kv_governor_state & state,
        const server_kv_pressure_action_ops & ops,
        const server_kv_pressure_snapshot & pressure,
        const std::vector<server_kv_claimant_snapshot> & claimants);

// V2 entry point. The view is sampled by the single server scheduler before
// this call. Hard PRESSURE/CRITICAL always takes precedence.
server_kv_pressure_action_result server_kv_pressure_execute_governor(
        const server_kv_pressure_unified_action_config & config,
        server_kv_governor_state & state,
        const server_kv_pressure_action_ops & ops,
        const server_kv_pressure_snapshot & pressure,
        const std::vector<server_kv_claimant_snapshot> & claimants,
        const server_kv_budget_view & budget_view);

std::string server_kv_pressure_unified_action_format_marker(
        const server_kv_pressure_action_result & result);
