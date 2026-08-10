#include "server-kv-pressure-action.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>

namespace {

bool parse_bool_env(const char * name) {
    const char * value = std::getenv(name);
    return value && std::string(value) == "1";
}

bool parse_required_uint64_env(const char * name, uint64_t & value, std::string & error) {
    const char * input = std::getenv(name);
    if (!input) {
        error = std::string("missing ") + name;
        return false;
    }
    if (input[0] == '\0') {
        error = std::string("empty ") + name;
        return false;
    }
    for (const char * cursor = input; *cursor; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            error = std::string("invalid ") + name;
            return false;
        }
    }

    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(input, &end, 10);
    if (errno == ERANGE || end == input || *end != '\0' ||
            parsed > std::numeric_limits<uint64_t>::max()) {
        error = std::string("invalid ") + name;
        return false;
    }
    value = (uint64_t) parsed;
    return true;
}

bool parse_required_uint32_env(const char * name, uint32_t & value, std::string & error) {
    uint64_t parsed = 0;
    if (!parse_required_uint64_env(name, parsed, error)) {
        return false;
    }
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        error = std::string("out of range ") + name;
        return false;
    }
    value = (uint32_t) parsed;
    return true;
}

bool parse_optional_uint32_env(
        const char * name, uint32_t & value, std::string & error) {
    if (!std::getenv(name)) {
        return true;
    }
    return parse_required_uint32_env(name, value, error);
}

bool parse_optional_uint64_env(
        const char * name, uint64_t & value, std::string & error) {
    if (!std::getenv(name)) {
        return true;
    }
    return parse_required_uint64_env(name, value, error);
}

const char * outcome_name(llama_kv_action_outcome outcome) {
    switch (outcome) {
    case llama_kv_action_outcome::completed:       return "completed";
    case llama_kv_action_outcome::no_op:           return "no_op";
    case llama_kv_action_outcome::unsupported:     return "unsupported";
    case llama_kv_action_outcome::rejected:        return "rejected";
    case llama_kv_action_outcome::failed:          return "failed";
    case llama_kv_action_outcome::partial_failure: return "partial_failure";
    }
    return "unknown";
}

const char * reason_name(llama_kv_action_reason reason) {
    switch (reason) {
    case llama_kv_action_reason::none:                   return "none";
    case llama_kv_action_reason::zero_budget:            return "zero_budget";
    case llama_kv_action_reason::invalid_sequence:       return "invalid_sequence";
    case llama_kv_action_reason::context_invalid:        return "context_invalid";
    case llama_kv_action_reason::write_transaction_open: return "write_transaction_open";
    case llama_kv_action_reason::unsupported:            return "unsupported";
    case llama_kv_action_reason::protected_sequence:     return "protected_sequence";
    case llama_kv_action_reason::shared_block:           return "shared_block";
    case llama_kv_action_reason::no_eligible_block:      return "no_eligible_block";
    case llama_kv_action_reason::state_rejected:         return "state_rejected";
    case llama_kv_action_reason::ownership_invalid:      return "ownership_invalid";
    case llama_kv_action_reason::io_failure:             return "io_failure";
    case llama_kv_action_reason::prefetch_failed:        return "prefetch_failed";
    case llama_kv_action_reason::no_candidate:           return "no_candidate";
    case llama_kv_action_reason::scan_budget_exhausted:  return "scan_budget_exhausted";
    case llama_kv_action_reason::target_satisfied:       return "target_satisfied";
    case llama_kv_action_reason::target_shortfall:       return "target_shortfall";
    case llama_kv_action_reason::blocked:                return "blocked";
    case llama_kv_action_reason::failed:                 return "failed";
    }
    return "unknown";
}

bool evaluate_allows_release(const llama_kv_action_result & evaluation) {
    return evaluation.outcome == llama_kv_action_outcome::completed &&
        !evaluation.io_failure &&
        !evaluation.fail_stop &&
        !evaluation.capability.context_invalid &&
        !evaluation.capability.write_transaction_open &&
        evaluation.capability.can_release;
}

bool evaluate_allows_offload(const llama_kv_action_result & evaluation) {
    return evaluation.outcome == llama_kv_action_outcome::completed &&
        !evaluation.io_failure &&
        !evaluation.fail_stop &&
        !evaluation.capability.context_invalid &&
        !evaluation.capability.write_transaction_open &&
        evaluation.capability.can_offload;
}

uint64_t saturating_add(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs
        ? std::numeric_limits<uint64_t>::max()
        : lhs + rhs;
}

uint64_t saturating_sub(uint64_t lhs, uint64_t rhs) {
    return rhs >= lhs ? 0 : lhs - rhs;
}

int64_t saturating_score_add(int64_t lhs, int64_t rhs) {
    if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) {
        return std::numeric_limits<int64_t>::max();
    }
    if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs) {
        return std::numeric_limits<int64_t>::min();
    }
    return lhs + rhs;
}

int64_t bounded_score(uint64_t value, uint64_t divisor, int64_t multiplier) {
    const uint64_t normalized = divisor == 0 ? value : value / divisor;
    const uint64_t limit = (uint64_t) std::numeric_limits<int64_t>::max() /
        (uint64_t) std::max<int64_t>(multiplier, 1);
    return (int64_t) std::min(normalized, limit) * multiplier;
}

const char * exclusion_name(server_kv_claimant_exclusion exclusion) {
    switch (exclusion) {
    case server_kv_claimant_exclusion::none:                   return "none";
    case server_kv_claimant_exclusion::active:                 return "active";
    case server_kv_claimant_exclusion::protected_sequence:     return "protected";
    case server_kv_claimant_exclusion::shared:                 return "shared";
    case server_kv_claimant_exclusion::write_transaction_open: return "write_open";
    case server_kv_claimant_exclusion::fail_stop:              return "fail_stop";
    case server_kv_claimant_exclusion::exhausted:              return "exhausted";
    case server_kv_claimant_exclusion::epoch_mismatch:          return "epoch_mismatch";
    case server_kv_claimant_exclusion::stale_generation:        return "stale_generation";
    case server_kv_claimant_exclusion::resident_lease:          return "resident_lease";
    case server_kv_claimant_exclusion::no_physical_relief:      return "no_physical_relief";
    case server_kv_claimant_exclusion::empty:                  return "empty";
    }
    return "unknown";
}

server_kv_claimant_score score_claimant(
        const server_kv_claimant_snapshot & claimant,
        const llama_kv_action_result & evaluation,
        uint32_t failure_count,
        uint64_t current_epoch,
        bool exhausted,
        bool require_legacy_bytes = true) {
    server_kv_claimant_score score;
    score.seq_id = claimant.seq_id;
    if (evaluation.capability.write_transaction_open) {
        score.exclusion = server_kv_claimant_exclusion::write_transaction_open;
        return score;
    }
    if (evaluation.fail_stop || evaluation.capability.context_invalid) {
        score.exclusion = server_kv_claimant_exclusion::fail_stop;
        return score;
    }
    if (claimant.epoch != current_epoch) {
        score.exclusion = server_kv_claimant_exclusion::epoch_mismatch;
        return score;
    }
    if (exhausted) {
        score.exclusion = server_kv_claimant_exclusion::exhausted;
        return score;
    }
    if (claimant.active) {
        score.exclusion = server_kv_claimant_exclusion::active;
        return score;
    }
    if (claimant.protected_sequence) {
        score.exclusion = server_kv_claimant_exclusion::protected_sequence;
        return score;
    }
    if (claimant.shared) {
        score.exclusion = server_kv_claimant_exclusion::shared;
        return score;
    }
    if (claimant.seq_id < 0 || claimant.logical_kv_tokens == 0 ||
            (require_legacy_bytes && claimant.reclaimable_bytes == 0)) {
        score.exclusion = server_kv_claimant_exclusion::empty;
        return score;
    }

    score.eligible = true;
    score.idle_age_score = bounded_score(claimant.idle_age_us, 1000, 4);
    score.logical_kv_score = bounded_score(claimant.logical_kv_tokens, 1, 256);
    score.reclaimable_score = bounded_score(claimant.reclaimable_bytes, 4096, 512);
    score.lcp_n_past_penalty = bounded_score(claimant.lcp_n_past_hint_tokens, 1, 1024);
    score.io_cost_penalty = bounded_score(claimant.io_cost_bytes, 4096, 64);
    score.failure_penalty = bounded_score(failure_count, 1, 1000000);
    score.total = 0;
    score.total = saturating_score_add(score.total, score.idle_age_score);
    score.total = saturating_score_add(score.total, score.logical_kv_score);
    score.total = saturating_score_add(score.total, score.reclaimable_score);
    score.total = saturating_score_add(score.total, -score.lcp_n_past_penalty);
    score.total = saturating_score_add(score.total, -score.io_cost_penalty);
    score.total = saturating_score_add(score.total, -score.failure_penalty);
    return score;
}

uint64_t saturating_cost_add(uint64_t lhs, uint64_t rhs) {
    return rhs > std::numeric_limits<uint64_t>::max() - lhs
        ? std::numeric_limits<uint64_t>::max()
        : lhs + rhs;
}

// V3 decision authority: returns the first non-empty fallback_reason seen on
// any eligible claimant, or nullptr when every eligible claimant is cost-aware.
// A mixed decision (some cost-aware, some fallback) must NOT be ranked with a
// single comparator that mixes µs-per-byte cost against idle-age scores, so the
// caller falls back to a uniform idle-age / LRU authority when this is non-null.
const char * v3_decision_fallback_reason(const std::vector<server_kv_claimant_score> & scores) {
    for (const auto & score : scores) {
        if (!score.eligible) {
            continue;
        }
        if (!score.cost_aware && score.fallback_reason &&
                std::string(score.fallback_reason) != "none") {
            return score.fallback_reason;
        }
    }
    return nullptr;
}

bool v3_score_less(const server_kv_claimant_score & lhs, const server_kv_claimant_score & rhs) {
    // cost-aware ranking: lower expected future cost per physical byte first.
    return lhs.total < rhs.total;
}

bool idle_age_score_greater(
        const server_kv_claimant_score & lhs, const server_kv_claimant_score & rhs) {
    // idle-age / LRU authority used by both idle_age policy and V3 decision
    // fallback: higher idle_age_score (older / colder) first.
    return lhs.idle_age_score > rhs.idle_age_score;
}

uint64_t estimate_reuse_probability_ppm(
        const server_kv_claimant_snapshot & claimant,
        const server_kv_claimant_history & history) {
    if (history.last_restore_bytes > 0) {
        return history.last_restore_gate_us > 0 ? 1000000 : 0;
    }
    if (claimant.logical_kv_tokens == 0 || claimant.lcp_n_past_hint_tokens == 0) {
        return 0;
    }
    const uint64_t numerator = std::min<uint64_t>(
            claimant.lcp_n_past_hint_tokens, claimant.logical_kv_tokens);
    return std::min<uint64_t>(1000000, (numerator * 1000000) /
            claimant.logical_kv_tokens);
}

server_kv_claimant_score score_claimant_v3(
        const server_kv_claimant_snapshot & claimant,
        const llama_kv_action_result & evaluation,
        uint32_t failure_count,
        uint64_t current_epoch,
        bool exhausted,
        const server_kv_pressure_snapshot & pressure,
        uint64_t sample_count,
        uint64_t churn_penalty_us) {
    server_kv_claimant_score score = score_claimant(
            claimant, evaluation, failure_count, current_epoch, exhausted, false);
    score.physical_estimate_available = claimant.physical_estimate_available;
    score.physical_estimate_authoritative = claimant.physical_estimate_authoritative;
    score.estimated_physical_bytes = claimant.estimated_exclusive_resident_bytes;
    score.raw_idle_age_us = claimant.idle_age_us;
    score.raw_answer_tokens = claimant.answer_tokens;
    score.raw_lcp_hint_tokens = claimant.lcp_n_past_hint_tokens;
    score.physical_object_id = claimant.physical_object_id;
    score.physical_generation = claimant.physical_generation;
    score.actual_relief_bytes = claimant.history.last_actual_relief_bytes;
    score.resident_lease_until_sample = claimant.history.resident_lease_until_sample;
    score.round_trip_count = claimant.history.round_trip_count;
    score.last_offload_bytes = claimant.history.last_offload_bytes;
    score.last_restore_bytes = claimant.history.last_restore_bytes;
    if (!score.eligible) {
        return score;
    }
    score.reuse_probability_ppm = estimate_reuse_probability_ppm(claimant, claimant.history);
    if (claimant.history.resident_lease_until_sample > sample_count) {
        score.eligible = false;
        score.exclusion = server_kv_claimant_exclusion::resident_lease;
        return score;
    }
    if (claimant.physical_estimate_available && pressure.kv_physical_view_available &&
            (claimant.physical_object_id != pressure.kv_object_id ||
             claimant.physical_generation != pressure.kv_generation)) {
        score.eligible = false;
        score.exclusion = server_kv_claimant_exclusion::stale_generation;
        return score;
    }
    if (!claimant.physical_estimate_available) {
        score.fallback_reason = "physical_unavailable";
        score.total = score.idle_age_score;
        return score;
    }
    if (!claimant.physical_estimate_authoritative) {
        score.fallback_reason = "physical_not_authoritative";
        score.total = score.idle_age_score;
        return score;
    }
    if (claimant.estimated_exclusive_resident_bytes == 0) {
        score.eligible = false;
        score.exclusion = server_kv_claimant_exclusion::no_physical_relief;
        return score;
    }
    if (claimant.history.last_offload_time_us == 0) {
        score.fallback_reason = "write_cost_unavailable";
        score.total = score.idle_age_score;
        return score;
    }
    if (score.reuse_probability_ppm > 0 && claimant.history.last_restore_gate_us == 0) {
        score.fallback_reason = "restore_cost_unavailable";
        score.total = score.idle_age_score;
        return score;
    }
    if (score.reuse_probability_ppm == 0 && claimant.history.last_restore_bytes == 0 &&
            claimant.lcp_n_past_hint_tokens == 0) {
        score.fallback_reason = "no_reuse_evidence";
        score.total = score.idle_age_score;
        return score;
    }

    score.cost_aware = true;
    score.expected_offload_write_cost_us = claimant.history.last_offload_time_us;
    score.expected_restore_gate_cost_us = claimant.history.last_restore_gate_us;
    score.churn_penalty_us = churn_penalty_us >
            std::numeric_limits<uint64_t>::max() /
                std::max<uint32_t>(claimant.history.round_trip_count, 1)
        ? std::numeric_limits<uint64_t>::max()
        : churn_penalty_us * claimant.history.round_trip_count;
    const uint64_t restore_cost = score.reuse_probability_ppm > 0 &&
            score.expected_restore_gate_cost_us > 0
        ? (score.expected_restore_gate_cost_us * score.reuse_probability_ppm) / 1000000
        : 0;
    score.expected_cost_us = saturating_cost_add(
            saturating_cost_add(score.expected_offload_write_cost_us, restore_cost),
            score.churn_penalty_us);
    const uint64_t scaled_cost = score.expected_cost_us >
            std::numeric_limits<uint64_t>::max() / 1000000
        ? std::numeric_limits<uint64_t>::max()
        : score.expected_cost_us * 1000000;
    score.total = scaled_cost > (uint64_t) std::numeric_limits<int64_t>::max()
        ? std::numeric_limits<int64_t>::max()
        : (int64_t) (scaled_cost / claimant.estimated_exclusive_resident_bytes);
    score.fallback_reason = "none";
    return score;
}

} // namespace

const char * server_kv_pressure_policy_name(server_kv_pressure_policy policy) {
    switch (policy) {
    case server_kv_pressure_policy::v2:       return "v2";
    case server_kv_pressure_policy::idle_age: return "idle_age";
    case server_kv_pressure_policy::v3:       return "v3";
    }
    return "unknown";
}

server_kv_pressure_unified_action_startup_decision
server_kv_pressure_unified_action_startup_decide_from_env() {
    server_kv_pressure_unified_action_startup_decision decision;
    auto & enablement = decision.enablement;
    enablement.requested = parse_bool_env("LLAMA_KV_PRESSURE_UNIFIED_ACTION");
    enablement.legacy_requested = parse_bool_env("LLAMA_KV_PAGED_RELEASE");
    enablement.dry_run_requested = parse_bool_env("LLAMA_KV_PRESSURE_DRY_RUN");
    enablement.bounded_requested = parse_bool_env("LLAMA_KV_PRESSURE_BOUNDED_RELEASE");

    if (!enablement.requested) {
        return decision;
    }

    const char * conflicting_env = nullptr;
    if (enablement.legacy_requested) {
        conflicting_env = "LLAMA_KV_PAGED_RELEASE";
    } else if (enablement.dry_run_requested) {
        conflicting_env = "LLAMA_KV_PRESSURE_DRY_RUN";
    } else if (enablement.bounded_requested) {
        conflicting_env = "LLAMA_KV_PRESSURE_BOUNDED_RELEASE";
    }
    if (conflicting_env) {
        decision.status = server_kv_pressure_unified_action_startup_status::conflict;
        decision.error = std::string("LLAMA_KV_PRESSURE_UNIFIED_ACTION=1 conflicts with ") +
                conflicting_env + "=1";
        return decision;
    }

    auto & config = decision.config;
    if (!parse_required_uint64_env("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES",
                                   config.target_bytes, decision.error) ||
            !parse_required_uint32_env("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS",
                                       config.max_blocks, decision.error)) {
        decision.status = server_kv_pressure_unified_action_startup_status::invalid;
        return decision;
    }
    if (config.target_bytes == 0) {
        decision.status = server_kv_pressure_unified_action_startup_status::invalid;
        decision.error = "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES must be greater than zero";
        return decision;
    }
    if (config.max_blocks == 0) {
        decision.status = server_kv_pressure_unified_action_startup_status::invalid;
        decision.error = "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS must be greater than zero";
        return decision;
    }

    const char * policy = std::getenv("LLAMA_KV_PRESSURE_POLICY");
    if (policy) {
        if (std::strcmp(policy, "v2") == 0) {
            config.policy = server_kv_pressure_policy::v2;
        } else if (std::strcmp(policy, "idle_age") == 0 || std::strcmp(policy, "lru") == 0) {
            config.policy = server_kv_pressure_policy::idle_age;
        } else if (std::strcmp(policy, "v3") == 0) {
            config.policy = server_kv_pressure_policy::v3;
        } else {
            decision.status = server_kv_pressure_unified_action_startup_status::invalid;
            decision.error = "LLAMA_KV_PRESSURE_POLICY must be v2, idle_age, lru, or v3";
            return decision;
        }
    }
    if (!parse_optional_uint32_env(
                "LLAMA_KV_PRESSURE_RESIDENT_LEASE_SAMPLES",
                config.resident_lease_samples, decision.error) ||
            !parse_optional_uint64_env(
                "LLAMA_KV_PRESSURE_CHURN_PENALTY_US",
                config.churn_penalty_us, decision.error)) {
        decision.status = server_kv_pressure_unified_action_startup_status::invalid;
        return decision;
    }

    config.enabled = true;
    decision.status = server_kv_pressure_unified_action_startup_status::enabled;
    return decision;
}

server_kv_resident_target_state server_kv_resident_target_from_env(std::string & error) {
    server_kv_resident_target_state result;
    const char * bytes = std::getenv("LLAMA_KV_RESIDENT_TARGET_BYTES");
    const char * source = std::getenv("LLAMA_KV_RESIDENT_TARGET_SOURCE");
    if (!bytes && !source) {
        return result;
    }
    if (!source || std::strcmp(source, "env_static") != 0) {
        error = "LLAMA_KV_RESIDENT_TARGET_SOURCE must be env_static";
        return result;
    }
    uint64_t target = 0;
    if (!parse_required_uint64_env("LLAMA_KV_RESIDENT_TARGET_BYTES", target, error) ||
            target == 0) {
        if (error.empty()) {
            error = "LLAMA_KV_RESIDENT_TARGET_BYTES must be greater than zero";
        }
        return result;
    }
    result.enabled = true;
    result.target_bytes = target;
    result.basis_generation = 1;
    result.source = "env_static";
    return result;
}

server_kv_pressure_action_result server_kv_pressure_execute_unified_action(
        const server_kv_pressure_unified_action_config & config,
        const server_kv_pressure_action_ops & ops,
        const kv_pressure_telemetry & telemetry,
        bool idle,
        uint64_t sample_count,
        uint64_t decision_id) {
    server_kv_pressure_action_result result;
    auto & observation = result.observation;
    observation.pressure_state = telemetry.state;
    observation.pressure_source = telemetry.source;
    observation.sample_valid = telemetry.sample_valid;
    observation.stale = telemetry.stale;
    observation.pressure_basis_valid = telemetry.pressure_basis_valid;
    observation.idle = idle;
    observation.sample_count = sample_count;
    observation.decision_id = decision_id;
    observation.target_bytes = config.target_bytes;
    observation.max_blocks = config.max_blocks;

    if (!config.enabled) {
        return result;
    }
    if (!ops.execute) {
        observation.reason = "no_memory";
        return result;
    }
    if (telemetry.stale || !telemetry.sample_valid) {
        observation.reason = "stale";
        return result;
    }
    if (telemetry.state != kv_pressure_state::PRESSURE &&
            telemetry.state != kv_pressure_state::CRITICAL) {
        observation.reason = "not_pressure";
        return result;
    }

    observation.evaluate_attempted = true;
    result.evaluation = ops.execute({
            llama_kv_action::evaluate, decision_id, -1, 0, 0, false, false,
    });
    if (result.evaluation.decision_id != decision_id) {
        observation.reason = "decision_mismatch";
        return result;
    }
    if (result.evaluation.capability.context_invalid) {
        observation.reason = "context_invalid";
        return result;
    }
    if (result.evaluation.capability.write_transaction_open) {
        observation.reason = "write_transaction_open";
        return result;
    }
    if (result.evaluation.fail_stop) {
        observation.reason = "fail_stop";
        return result;
    }
    if (!result.evaluation.capability.can_release) {
        observation.reason = "release_unsupported";
        return result;
    }
    if (!evaluate_allows_release(result.evaluation)) {
        observation.reason = "evaluate_rejected";
        return result;
    }

    observation.release_attempted = true;
    result.release = ops.execute({
            llama_kv_action::release, decision_id, -1,
            config.target_bytes, config.max_blocks, false, false,
    });
    observation.reason = "release_submitted";
    return result;
}

void server_kv_governor_state::reset() {
    episode_active_ = false;
    episode_ = 0;
    pressure_basis_generation_ = 0;
    pressure_debt_bytes_ = 0;
    offload_armed_ = false;
    idle_follow_up_pending_ = false;
    next_action_sample_ = 0;
    claimant_epochs_.clear();
    exhausted_claimants_.clear();
    failure_counts_.clear();
    claimant_history_.clear();
    // V2 step1: clear soft budget state too — handles sleeping/reload boundaries.
    budget_debt_bytes_ = 0;
    budget_basis_generation_ = 0;
    soft_offload_armed_ = false;
    unmet_budget_bytes_ = 0;
    budget_next_action_sample_ = 0;
}

uint64_t server_kv_governor_state::claimant_epoch(llama_seq_id seq_id) const {
    const auto it = claimant_epochs_.find(seq_id);
    return it == claimant_epochs_.end() ? 1 : it->second;
}

bool server_kv_governor_state::claimant_exhausted(llama_seq_id seq_id, uint64_t epoch) const {
    const auto it = exhausted_claimants_.find(seq_id);
    return it != exhausted_claimants_.end() && it->second == epoch;
}

void server_kv_governor_state::invalidate_claimant(llama_seq_id seq_id) {
    if (seq_id < 0) {
        return;
    }
    const uint64_t next_epoch = saturating_add(claimant_epoch(seq_id), 1);
    claimant_epochs_[seq_id] = next_epoch == 0 ? 1 : next_epoch;
    exhausted_claimants_.erase(seq_id);
    failure_counts_.erase(seq_id);
    // Ordinary turn rollover must not discard cost evidence that still
    // describes the live KV lineage.  Migrate the history epoch so
    // claimant_history(seq, next_epoch) resolves; object_id / generation still
    // guard the cost evidence against a real lineage change.
    auto it = claimant_history_.find(seq_id);
    if (it != claimant_history_.end() && it->second.epoch != 0) {
        it->second.epoch = next_epoch;
    }
}

void server_kv_governor_state::clear_claimant_lineage(llama_seq_id seq_id) {
    if (seq_id < 0) {
        return;
    }
    const uint64_t next_epoch = saturating_add(claimant_epoch(seq_id), 1);
    claimant_epochs_[seq_id] = next_epoch == 0 ? 1 : next_epoch;
    exhausted_claimants_.erase(seq_id);
    failure_counts_.erase(seq_id);
    claimant_history_.erase(seq_id);
}

server_kv_claimant_history server_kv_governor_state::claimant_history(
        llama_seq_id seq_id, uint64_t epoch) const {
    const auto it = claimant_history_.find(seq_id);
    if (it == claimant_history_.end() ||
            (it->second.epoch != 0 && it->second.epoch != epoch)) {
        return {};
    }
    return it->second;
}

void server_kv_governor_state::record_offload_feedback(
        const server_kv_claimant_snapshot & claimant,
        const llama_kv_action_result & action,
        uint64_t sample_count) {
    if (claimant.seq_id < 0 || claimant.epoch == 0 ||
            claimant.epoch != claimant_epoch(claimant.seq_id) ||
            action.action != llama_kv_action::offload ||
            !action.state_changed || action.blocks == 0 || action.bytes == 0) {
        return;
    }
    auto & history = claimant_history_[claimant.seq_id];
    if (history.epoch != 0 && history.epoch != claimant.epoch) {
        history = {};
    }
    if (history.object_id != 0 && claimant.physical_object_id != 0 &&
            (history.object_id != claimant.physical_object_id ||
             history.generation != claimant.physical_generation)) {
        history = {};
    }
    if (claimant.physical_object_id != 0 &&
            (action.physical_object_id != claimant.physical_object_id ||
             action.physical_generation != claimant.physical_generation)) {
        return;
    }
    if (history.object_id != 0 &&
            (action.physical_object_id == 0 || action.physical_generation == 0 ||
             history.object_id != action.physical_object_id ||
             history.generation != action.physical_generation)) {
        history = {};
    }
    history.epoch = claimant.epoch;
    history.object_id = claimant.physical_object_id != 0
        ? claimant.physical_object_id : action.physical_object_id;
    history.generation = claimant.physical_generation != 0
        ? claimant.physical_generation : action.physical_generation;
    history.last_offload_sample = sample_count;
    history.last_offload_bytes = action.bytes;
    history.last_offload_time_us = action.action_elapsed_us;
    history.last_transition_sample = sample_count;
    if (action.physical_relief_available) {
        history.last_actual_relief_bytes = action.physical_relief_bytes;
    }
}

void server_kv_governor_state::record_restore_feedback(
        llama_seq_id seq_id,
        uint64_t epoch,
        const llama_kv_action_result & action,
        uint64_t gate_us,
        uint64_t sample_count,
        uint32_t resident_lease_samples) {
    if (seq_id < 0 || epoch == 0 || epoch != claimant_epoch(seq_id) ||
            action.action != llama_kv_action::prefetch ||
            !action.state_changed || action.blocks == 0 || action.bytes == 0) {
        return;
    }
    auto & history = claimant_history_[seq_id];
    if (history.epoch != 0 && history.epoch != epoch) {
        return;
    }
    if (history.epoch != 0 &&
            (action.physical_object_id == 0 || action.physical_generation == 0 ||
             history.object_id != action.physical_object_id ||
             history.generation != action.physical_generation)) {
        return;
    }
    history.epoch = epoch;
    if (history.object_id == 0 && action.physical_object_id != 0 &&
            action.physical_generation != 0) {
        history.object_id = action.physical_object_id;
        history.generation = action.physical_generation;
    }
    const bool round_trip = history.last_offload_sample > history.last_restore_sample;
    history.last_restore_sample = sample_count;
    history.last_restore_bytes = action.bytes;
    history.last_restore_gate_us = gate_us;
    if (round_trip) {
        history.round_trip_count = std::min<uint32_t>(
                history.round_trip_count + 1, std::numeric_limits<uint32_t>::max());
    }
    history.last_transition_sample = sample_count;
    history.resident_lease_until_sample = saturating_add(
            sample_count, resident_lease_samples);
}

void server_kv_governor_state::record_resume_feedback(
        llama_seq_id seq_id,
        const llama_kv_action_result & action,
        uint64_t gate_us,
        uint64_t sample_count,
        uint32_t resident_lease_samples) {
    if (seq_id < 0) {
        return;
    }
    const uint64_t epoch = claimant_epoch(seq_id);
    const auto history = claimant_history(seq_id, epoch);
    if (history.epoch != 0 &&
            (action.physical_object_id == 0 || action.physical_generation == 0 ||
             history.object_id != action.physical_object_id ||
             history.generation != action.physical_generation)) {
        // A physical generation mismatch means the previous cost evidence no
        // longer describes the live KV object: hard-clear the lineage instead
        // of migrating it across the epoch rollover.
        clear_claimant_lineage(seq_id);
        return;
    }
    exhausted_claimants_.erase(seq_id);
    failure_counts_.erase(seq_id);
    record_restore_feedback(
            seq_id, epoch, action, gate_us, sample_count, resident_lease_samples);
}

void server_kv_governor_state::invalidate_all_claimants() {
    exhausted_claimants_.clear();
    failure_counts_.clear();
    claimant_history_.clear();
    for (auto & entry : claimant_epochs_) {
        const uint64_t next_epoch = saturating_add(entry.second, 1);
        entry.second = next_epoch == 0 ? 1 : next_epoch;
    }
}

// KV-Budget-V2 step1: soft budget debt derivation from the read-only view.
// Saturating-subtract resident_bytes by target_bytes; the result is the
// steady-resident excess.  Zero (or any non-negative value) means the soft
// target is satisfied and no soft action runs.
//
// invalid view (layout corruption, write-context invalid, swap-metadata
// drift) MUST fail-closed — the debt is left at 0, but `budget_active` stays
// false and the soft chain does NOT enter the decision.  This prevents
// stale-view soft actions from racing with the hard pressure sampler.
static uint64_t derive_budget_excess(
        const server_kv_pressure_unified_action_config & config,
        const server_kv_budget_view & view) {
    if (!config.budget_target_enabled || config.budget_target_bytes == 0) {
        return 0;
    }
    if (!view.valid || !view.resident_available) {
        return 0;
    }
    return view.resident_bytes > config.budget_target_bytes
        ? view.resident_bytes - config.budget_target_bytes
        : 0;
}

server_kv_pressure_action_result server_kv_pressure_execute_governor(
        const server_kv_pressure_unified_action_config & config,
        server_kv_governor_state & state,
        const server_kv_pressure_action_ops & ops,
        const server_kv_pressure_snapshot & pressure,
        const std::vector<server_kv_claimant_snapshot> & claimants,
        const server_kv_budget_view & budget_view) {
    server_kv_pressure_action_result result;
    result.policy = config.policy;
    auto & observation = result.observation;
    observation.pressure_state = pressure.state;
    observation.pressure_source = pressure.source;
    observation.sample_valid = pressure.sample_valid;
    observation.stale = pressure.stale;
    observation.pressure_basis_valid = pressure.pressure_basis_valid;
    observation.sample_count = pressure.sample_count;
    observation.decision_id = pressure.decision_id;
    observation.target_bytes = config.target_bytes;
    observation.max_blocks = config.max_blocks;
    result.episode = state.episode_;
    result.debt_before_bytes = state.pressure_debt_bytes_;
    result.debt_after_bytes = state.pressure_debt_bytes_;
    result.offload_armed_before = state.offload_armed_;
    result.offload_armed_after = state.offload_armed_;
    result.next_action_sample = state.next_action_sample_;

    // V2 step1: populate soft budget side-channel fields with last-known
    // view snapshot regardless of which chain runs.  These are advisory and
    // do not affect any decision state when `budget_target_enabled` is false.
    result.budget_target_enabled  = config.budget_target_enabled;
    result.budget_target_bytes    = config.budget_target_bytes;
    result.budget_basis_generation = config.budget_basis_generation;
    result.budget_source          = config.budget_source ? config.budget_source : "none";
    result.budget_view_valid      = budget_view.valid;
    result.budget_resident_available = budget_view.resident_available;
    result.budget_reclaimable_available = budget_view.reclaimable_available;
    result.budget_resident_bytes  = budget_view.resident_bytes;
    result.budget_dead_resident_reclaimable_bytes = budget_view.dead_resident_reclaimable_bytes;
    result.budget_transient_staging_bound_bytes  = budget_view.transient_staging_bound_bytes;
    result.budget_debt_before_bytes = state.budget_debt_bytes_;
    result.budget_debt_after_bytes  = state.budget_debt_bytes_;
    result.soft_offload_armed_before = state.soft_offload_armed_;
    result.soft_offload_armed_after  = state.soft_offload_armed_;
    result.budget_next_action_sample = state.budget_next_action_sample_;
    result.unmet_budget_bytes_after  = state.unmet_budget_bytes_;

    if (!config.enabled) {
        state.idle_follow_up_pending_ = false;
        // V2 step1: when unified action is globally disabled we still keep
        // the budget side-channel populated but never enter the soft chain.
        result.budget_active = false;
        return result;
    }
    if (!ops.execute) {
        state.idle_follow_up_pending_ = false;
        observation.reason = "no_memory";
        return result;
    }
    if (!pressure.sample_valid || pressure.stale) {
        state.idle_follow_up_pending_ = false;
        observation.reason = "stale_hold";
        return result;
    }
    // Pressure basis validity gates both hard and soft arbitration.  An
    // invalid basis is not a zero-pressure observation and must preserve all
    // existing soft debt, armed state, and unmet-budget backoff.
    if (!pressure.pressure_basis_valid) {
        observation.reason = "invalid_basis_hold";
        return result;
    }
    if (pressure.state != kv_pressure_state::PRESSURE &&
            pressure.state != kv_pressure_state::CRITICAL) {
        // Hard-pressure episode ends, but claimant epochs/failure history and
        // the independent soft budget state must not be reset together.
        state.episode_active_ = false;
        state.episode_ = 0;
        state.pressure_basis_generation_ = 0;
        state.pressure_debt_bytes_ = 0;
        state.offload_armed_ = false;
        state.idle_follow_up_pending_ = false;
        state.next_action_sample_ = 0;
        result.episode = 0;
        result.debt_before_bytes = 0;
        result.debt_after_bytes = 0;
        result.offload_armed_before = false;
        result.offload_armed_after = false;
        result.next_action_sample = 0;

        if (!config.budget_target_enabled || config.budget_target_bytes == 0) {
            state.budget_debt_bytes_ = 0;
            state.budget_basis_generation_ = 0;
            state.soft_offload_armed_ = false;
            state.unmet_budget_bytes_ = 0;
            state.budget_next_action_sample_ = 0;
            result.budget_debt_before_bytes = 0;
            result.budget_debt_after_bytes = 0;
            result.soft_offload_armed_before = false;
            result.soft_offload_armed_after = false;
            result.unmet_budget_bytes_after = 0;
            observation.reason = "not_pressure_reset";
            return result;
        }

        // A missing or invalid physical view is not a zero debt observation.
        // Hold the soft chain and let the next scheduler sample retry.
        if (!budget_view.valid || !budget_view.resident_available) {
            result.budget_active = false;
            observation.reason = "budget_view_unavailable";
            return result;
        }

        result.budget_active = true;
        result.budget_observed_excess_bytes = derive_budget_excess(config, budget_view);
        if (state.budget_basis_generation_ != config.budget_basis_generation) {
            state.budget_basis_generation_ = config.budget_basis_generation;
            state.budget_debt_bytes_ = 0;
            state.soft_offload_armed_ = false;
            state.unmet_budget_bytes_ = 0;
            state.budget_next_action_sample_ = 0;
        }
        state.budget_debt_bytes_ = result.budget_observed_excess_bytes;
        result.budget_debt_before_bytes = state.budget_debt_bytes_;
        result.budget_debt_after_bytes = state.budget_debt_bytes_;
        result.soft_offload_armed_before = state.soft_offload_armed_;
        result.soft_offload_armed_after = state.soft_offload_armed_;

        if (state.budget_debt_bytes_ == 0) {
            state.soft_offload_armed_ = false;
            state.unmet_budget_bytes_ = 0;
            state.budget_next_action_sample_ = 0;
            observation.reason = "budget_target_satisfied";
            result.budget_debt_after_bytes = 0;
            result.soft_offload_armed_after = false;
            result.budget_next_action_sample = 0;
            result.unmet_budget_bytes_after = 0;
            return result;
        }

        // A terminal unmet-budget result is held until the view improves, the
        // target basis changes, or the explicit backoff expires. This prevents
        // repeated claimant scans when all remaining KV is correctness-live.
        if (state.unmet_budget_bytes_ > 0 &&
                result.budget_observed_excess_bytes >= state.unmet_budget_bytes_) {
            if (pressure.sample_count < state.budget_next_action_sample_) {
                observation.reason = "budget_unmet_hold";
                result.budget_next_action_sample = state.budget_next_action_sample_;
                result.unmet_budget_bytes_after = state.unmet_budget_bytes_;
                return result;
            }
            state.unmet_budget_bytes_ = 0;
        }
        if (pressure.sample_count < state.budget_next_action_sample_) {
            observation.reason = "budget_backoff";
            result.budget_next_action_sample = state.budget_next_action_sample_;
            return result;
        }

        state.idle_follow_up_pending_ = false;
        observation.evaluate_attempted = true;
        result.evaluation = ops.execute({
                llama_kv_action::evaluate,
                pressure.decision_id,
                -1,
                0,
                0,
                false,
                false,
                llama_kv_memory_claimant::kv,
                state.soft_offload_armed_ ? llama_kv_io_class::capacity_write
                                          : llama_kv_io_class::background_write,
                0,
                0,
        });
        if (result.evaluation.decision_id != pressure.decision_id) {
            observation.reason = "decision_mismatch";
            return result;
        }
        if (result.evaluation.capability.context_invalid) {
            observation.reason = "context_invalid";
            return result;
        }
        if (result.evaluation.capability.write_transaction_open) {
            observation.reason = "write_transaction_open";
            return result;
        }
        if (result.evaluation.fail_stop) {
            observation.reason = "fail_stop";
            return result;
        }

        const uint64_t action_bytes = config.target_bytes > 0
            ? std::min(config.target_bytes, state.budget_debt_bytes_)
            : state.budget_debt_bytes_;
        const uint64_t next_cooldown = saturating_add(
                pressure.sample_count, std::max<uint32_t>(config.cooldown_samples, 1));

        if (!state.soft_offload_armed_) {
            if (!evaluate_allows_release(result.evaluation)) {
                observation.reason = result.evaluation.capability.can_release
                    ? "evaluate_rejected"
                    : "release_unsupported";
                return result;
            }
            observation.release_attempted = true;
            result.release = ops.execute({
                    llama_kv_action::release,
                    pressure.decision_id,
                    -1,
                    action_bytes,
                    config.max_blocks,
                    false,
                    false,
                    llama_kv_memory_claimant::kv,
                    llama_kv_io_class::background_write,
                    0,
                    action_bytes,
            });
            const bool matches = result.release.action == llama_kv_action::release &&
                result.release.decision_id == pressure.decision_id;
            const bool progressed = matches && !result.release.io_failure &&
                result.release.state_changed && result.release.relieved_bytes > 0;
            const bool scan_advanced = matches && !result.release.io_failure &&
                !result.release.fail_stop &&
                result.release.reason == llama_kv_action_reason::scan_budget_exhausted &&
                (result.release.outcome == llama_kv_action_outcome::completed ||
                 result.release.outcome == llama_kv_action_outcome::no_op);
            const bool retry = matches && result.release.io_failure && !result.release.fail_stop;
            if (matches) {
                state.budget_debt_bytes_ = saturating_sub(
                        state.budget_debt_bytes_, result.release.relieved_bytes);
            }
            const bool arm = matches &&
                result.release.reason == llama_kv_action_reason::no_candidate &&
                state.budget_debt_bytes_ > 0;
            if (arm) {
                state.soft_offload_armed_ = true;
            }
            state.budget_next_action_sample_ = result.release.io_failure
                ? saturating_add(pressure.sample_count,
                        std::max<uint32_t>(config.io_failure_backoff_samples, 1))
                : next_cooldown;
            state.idle_follow_up_pending_ = state.budget_debt_bytes_ > 0 &&
                (progressed || scan_advanced || arm || retry);
            observation.reason = "budget_release_submitted";
        } else {
            if (!evaluate_allows_offload(result.evaluation)) {
                observation.reason = result.evaluation.capability.can_offload
                    ? "evaluate_rejected"
                    : "offload_unsupported";
                return result;
            }
            result.scores.reserve(claimants.size());
            for (const auto & claimant : claimants) {
                const auto it = state.failure_counts_.find(claimant.seq_id);
                const uint32_t failure_count = it == state.failure_counts_.end()
                    ? 0
                    : std::min(it->second, config.max_failure_penalty);
                const uint64_t current_epoch = state.claimant_epoch(claimant.seq_id);
                const auto exhausted_it = state.exhausted_claimants_.find(claimant.seq_id);
                const bool exhausted = exhausted_it != state.exhausted_claimants_.end() &&
                    exhausted_it->second == current_epoch;
                auto score = config.policy == server_kv_pressure_policy::v3
                    ? score_claimant_v3(
                            claimant, result.evaluation, failure_count, current_epoch, exhausted,
                            pressure, pressure.sample_count, config.churn_penalty_us)
                    : score_claimant(
                            claimant, result.evaluation, failure_count, current_epoch, exhausted);
                if (config.policy == server_kv_pressure_policy::idle_age && score.eligible) {
                    score.total = score.idle_age_score;
                }
                result.scores.push_back(score);
            }
            const bool v3_policy = config.policy == server_kv_pressure_policy::v3;
            const char * decision_fallback = v3_policy
                ? v3_decision_fallback_reason(result.scores) : nullptr;
            if (v3_policy) {
                result.decision_fallback_reason = decision_fallback ? decision_fallback : "";
            }
            std::sort(result.scores.begin(), result.scores.end(),
                    [v3_policy, decision_fallback](const server_kv_claimant_score & lhs,
                                                    const server_kv_claimant_score & rhs) {
                        if (lhs.eligible != rhs.eligible) return lhs.eligible > rhs.eligible;
                        if (!v3_policy) {
                            // v2 / idle_age: legacy descending total, identical authority.
                            if (lhs.total != rhs.total) return lhs.total > rhs.total;
                            return lhs.seq_id < rhs.seq_id;
                        }
                        // V3: a mixed-evidence decision falls back to a single
                        // idle-age / LRU authority so cost-aware µs-per-byte and
                        // idle_age_score are never compared directly.
                        if (decision_fallback) {
                            if (lhs.idle_age_score != rhs.idle_age_score) {
                                return idle_age_score_greater(lhs, rhs);
                            }
                            return lhs.seq_id < rhs.seq_id;
                        }
                        if (lhs.total != rhs.total) return v3_score_less(lhs, rhs);
                        return lhs.seq_id < rhs.seq_id;
                    });
            for (size_t rank = 0; rank < result.scores.size(); ++rank) {
                result.scores[rank].rank = (uint32_t) rank;
            }
            const auto selected = std::find_if(
                    result.scores.begin(), result.scores.end(),
                    [](const server_kv_claimant_score & score) { return score.eligible; });
            if (selected == result.scores.end()) {
                state.unmet_budget_bytes_ = state.budget_debt_bytes_;
                state.soft_offload_armed_ = false;
                state.idle_follow_up_pending_ = false;
                state.budget_next_action_sample_ = saturating_add(
                        pressure.sample_count,
                        std::max<uint32_t>(config.budget_unmet_backoff_samples, 1));
                observation.reason = "budget_unmet_terminal";
            } else {
                result.selected_seq_id = selected->seq_id;
                result.selected_claimant_epoch = state.claimant_epoch(selected->seq_id);
                const int64_t bounded_priority = std::max<int64_t>(
                        std::numeric_limits<int32_t>::min(),
                        std::min<int64_t>(std::numeric_limits<int32_t>::max(), selected->total));
                observation.offload_attempted = true;
                result.offload = ops.execute({
                        llama_kv_action::offload,
                        pressure.decision_id,
                        selected->seq_id,
                        action_bytes,
                        config.max_blocks,
                        false,
                        false,
                        llama_kv_memory_claimant::kv,
                        llama_kv_io_class::capacity_write,
                        (int32_t) bounded_priority,
                        action_bytes,
                });
                const bool matches = result.offload.action == llama_kv_action::offload &&
                    result.offload.decision_id == pressure.decision_id;
                if (matches && result.offload.state_changed) {
                    const auto claimant_it = std::find_if(
                            claimants.begin(), claimants.end(),
                            [selected](const server_kv_claimant_snapshot & item) {
                                return item.seq_id == selected->seq_id;
                            });
                    if (claimant_it != claimants.end()) {
                        state.record_offload_feedback(
                                *claimant_it, result.offload, pressure.sample_count);
                    }
                }
                const bool protected_or_shared = matches &&
                    (result.offload.reason == llama_kv_action_reason::protected_sequence ||
                     result.offload.reason == llama_kv_action_reason::shared_block);
                const bool progressed = matches && !protected_or_shared &&
                    !result.offload.io_failure && result.offload.state_changed &&
                    result.offload.relieved_bytes > 0;
                const bool retry = matches && !protected_or_shared &&
                    result.offload.io_failure && !result.offload.fail_stop;
                if (matches && !protected_or_shared) {
                    state.budget_debt_bytes_ = saturating_sub(
                            state.budget_debt_bytes_, result.offload.relieved_bytes);
                }
                const bool exhausted = matches && !protected_or_shared &&
                    result.offload.outcome == llama_kv_action_outcome::no_op &&
                    result.offload.reason == llama_kv_action_reason::no_candidate &&
                    !result.offload.state_changed && result.offload.relieved_bytes == 0;
                if (protected_or_shared) {
                    // Core ownership/state protection is terminal for this
                    // soft-budget pursuit; do not penalize or retry the
                    // protected claimant as if it were an I/O failure.
                    state.unmet_budget_bytes_ = state.budget_debt_bytes_;
                    state.soft_offload_armed_ = false;
                    state.idle_follow_up_pending_ = false;
                    state.budget_next_action_sample_ = saturating_add(
                            pressure.sample_count,
                            std::max<uint32_t>(config.budget_unmet_backoff_samples, 1));
                    observation.reason = "budget_unmet_terminal";
                } else {
                    if (matches && (result.offload.io_failure || exhausted)) {
                        if (exhausted) {
                            state.exhausted_claimants_[selected->seq_id] = result.selected_claimant_epoch;
                        }
                        auto & failures = state.failure_counts_[selected->seq_id];
                        if (failures < config.max_failure_penalty) ++failures;
                        state.budget_next_action_sample_ = result.offload.io_failure
                            ? saturating_add(pressure.sample_count,
                                    std::max<uint32_t>(config.io_failure_backoff_samples, 1))
                            : next_cooldown;
                    } else if (matches) {
                        state.failure_counts_.erase(selected->seq_id);
                        state.budget_next_action_sample_ = next_cooldown;
                    } else {
                        state.budget_next_action_sample_ = next_cooldown;
                    }
                    state.idle_follow_up_pending_ = state.budget_debt_bytes_ > 0 &&
                        (progressed || exhausted || retry);
                    observation.reason = "budget_offload_submitted";
                }
            }
        }

        if (state.budget_debt_bytes_ == 0) {
            state.soft_offload_armed_ = false;
            state.unmet_budget_bytes_ = 0;
            state.idle_follow_up_pending_ = false;
        }
        result.budget_debt_after_bytes = state.budget_debt_bytes_;
        result.soft_offload_armed_after = state.soft_offload_armed_;
        result.budget_next_action_sample = state.budget_next_action_sample_;
        result.unmet_budget_bytes_after = state.unmet_budget_bytes_;
        return result;
    }
    if (state.episode_active_ &&
            state.pressure_basis_generation_ != pressure.pressure_basis_generation) {
        state.reset();
    }
    if (!state.episode_active_) {
        state.episode_active_ = true;
        state.episode_ = saturating_add(state.episode_, 1);
        state.pressure_basis_generation_ = pressure.pressure_basis_generation;
    }

    result.observed_excess_bytes =
        pressure.pressure_current_bytes > pressure.pressure_low_water_bytes
            ? pressure.pressure_current_bytes - pressure.pressure_low_water_bytes
            : 0;
    state.pressure_debt_bytes_ = std::max(
            result.observed_excess_bytes, state.pressure_debt_bytes_);
    result.episode = state.episode_;
    result.debt_before_bytes = state.pressure_debt_bytes_;
    result.debt_after_bytes = state.pressure_debt_bytes_;
    result.offload_armed_before = state.offload_armed_;
    result.offload_armed_after = state.offload_armed_;

    if (state.pressure_debt_bytes_ == 0) {
        state.offload_armed_ = false;
        state.idle_follow_up_pending_ = false;
        observation.reason = "zero_debt";
        result.debt_after_bytes = 0;
        result.offload_armed_after = false;
        return result;
    }
    if (pressure.sample_count < state.next_action_sample_) {
        observation.reason = "backoff";
        result.next_action_sample = state.next_action_sample_;
        return result;
    }

    state.idle_follow_up_pending_ = false;
    observation.evaluate_attempted = true;
    result.evaluation = ops.execute({
            llama_kv_action::evaluate,
            pressure.decision_id,
            -1,
            0,
            0,
            false,
            false,
            llama_kv_memory_claimant::kv,
            state.offload_armed_ ? llama_kv_io_class::capacity_write : llama_kv_io_class::background_write,
            0,
            0,
    });
    if (result.evaluation.decision_id != pressure.decision_id) {
        observation.reason = "decision_mismatch";
        return result;
    }
    if (result.evaluation.capability.context_invalid) {
        observation.reason = "context_invalid";
        return result;
    }
    if (result.evaluation.capability.write_transaction_open) {
        observation.reason = "write_transaction_open";
        return result;
    }
    if (result.evaluation.fail_stop) {
        observation.reason = "fail_stop";
        return result;
    }

    const uint64_t target_bytes = std::min(config.target_bytes, state.pressure_debt_bytes_);
    const uint64_t next_cooldown = saturating_add(
            pressure.sample_count, std::max<uint32_t>(config.cooldown_samples, 1));

    if (!state.offload_armed_) {
        if (!evaluate_allows_release(result.evaluation)) {
            observation.reason = result.evaluation.capability.can_release
                ? "evaluate_rejected"
                : "release_unsupported";
            return result;
        }

        observation.release_attempted = true;
        result.release = ops.execute({
                llama_kv_action::release,
                pressure.decision_id,
                -1,
                target_bytes,
                config.max_blocks,
                false,
                false,
                llama_kv_memory_claimant::kv,
                llama_kv_io_class::background_write,
                0,
                target_bytes,
        });
        const bool release_matches_decision =
            result.release.action == llama_kv_action::release &&
            result.release.decision_id == pressure.decision_id;
        const bool release_progressed = release_matches_decision &&
            !result.release.io_failure && result.release.state_changed &&
            result.release.relieved_bytes > 0;
        const bool release_scan_advanced = release_matches_decision &&
            !result.release.io_failure && !result.release.fail_stop &&
            result.release.reason == llama_kv_action_reason::scan_budget_exhausted &&
            (result.release.outcome == llama_kv_action_outcome::completed ||
             result.release.outcome == llama_kv_action_outcome::no_op);
        const bool release_retry_scheduled = release_matches_decision &&
            result.release.io_failure && !result.release.fail_stop;
        if (release_matches_decision) {
            state.pressure_debt_bytes_ = saturating_sub(
                    state.pressure_debt_bytes_, result.release.relieved_bytes);
        }
        const bool armed_offload = release_matches_decision &&
            result.release.reason == llama_kv_action_reason::no_candidate &&
            state.pressure_debt_bytes_ > 0;
        if (armed_offload) {
            state.offload_armed_ = true;
        }
        if (result.release.io_failure) {
            state.next_action_sample_ = saturating_add(
                    pressure.sample_count,
                    std::max<uint32_t>(config.io_failure_backoff_samples, 1));
        } else {
            state.next_action_sample_ = next_cooldown;
        }
        state.idle_follow_up_pending_ = state.pressure_debt_bytes_ > 0 &&
            (release_progressed || release_scan_advanced || armed_offload ||
             release_retry_scheduled);
        observation.reason = "release_submitted";
    } else {
        if (!evaluate_allows_offload(result.evaluation)) {
            observation.reason = result.evaluation.capability.can_offload
                ? "evaluate_rejected"
                : "offload_unsupported";
            return result;
        }

        result.scores.reserve(claimants.size());
        for (const auto & claimant : claimants) {
            const auto it = state.failure_counts_.find(claimant.seq_id);
            const uint32_t failure_count = it == state.failure_counts_.end()
                ? 0
                : std::min(it->second, config.max_failure_penalty);
            const uint64_t current_epoch = state.claimant_epoch(claimant.seq_id);
            const auto exhausted_it = state.exhausted_claimants_.find(claimant.seq_id);
            const bool exhausted = exhausted_it != state.exhausted_claimants_.end() &&
                exhausted_it->second == current_epoch;
            auto score = config.policy == server_kv_pressure_policy::v3
                ? score_claimant_v3(
                        claimant, result.evaluation, failure_count, current_epoch, exhausted,
                        pressure, pressure.sample_count, config.churn_penalty_us)
                : score_claimant(
                        claimant, result.evaluation, failure_count, current_epoch, exhausted);
            if (config.policy == server_kv_pressure_policy::idle_age && score.eligible) {
                score.total = score.idle_age_score;
            }
            result.scores.push_back(score);
        }
        const bool v3_policy = config.policy == server_kv_pressure_policy::v3;
        const char * decision_fallback = v3_policy
            ? v3_decision_fallback_reason(result.scores) : nullptr;
        if (v3_policy) {
            result.decision_fallback_reason = decision_fallback ? decision_fallback : "";
        }
        std::sort(result.scores.begin(), result.scores.end(),
                [v3_policy, decision_fallback](const server_kv_claimant_score & lhs,
                                                const server_kv_claimant_score & rhs) {
                    if (lhs.eligible != rhs.eligible) return lhs.eligible > rhs.eligible;
                    if (!v3_policy) {
                        if (lhs.total != rhs.total) return lhs.total > rhs.total;
                        return lhs.seq_id < rhs.seq_id;
                    }
                    // V3 mixed-evidence decision falls back to idle-age / LRU.
                    if (decision_fallback) {
                        if (lhs.idle_age_score != rhs.idle_age_score) {
                            return idle_age_score_greater(lhs, rhs);
                        }
                        return lhs.seq_id < rhs.seq_id;
                    }
                    if (lhs.total != rhs.total) return v3_score_less(lhs, rhs);
                    return lhs.seq_id < rhs.seq_id;
                });
        for (size_t rank = 0; rank < result.scores.size(); ++rank) {
            result.scores[rank].rank = (uint32_t) rank;
        }

        const auto selected = std::find_if(
                result.scores.begin(), result.scores.end(),
                [](const server_kv_claimant_score & score) { return score.eligible; });
        if (selected == result.scores.end()) {
            state.next_action_sample_ = next_cooldown;
            observation.reason = "claimant_no_candidate";
        } else {
            result.selected_seq_id = selected->seq_id;
            result.selected_claimant_epoch = state.claimant_epoch(selected->seq_id);
            const int64_t bounded_priority = std::max<int64_t>(
                    std::numeric_limits<int32_t>::min(),
                    std::min<int64_t>(std::numeric_limits<int32_t>::max(), selected->total));
            observation.offload_attempted = true;
            result.offload = ops.execute({
                    llama_kv_action::offload,
                    pressure.decision_id,
                    selected->seq_id,
                    target_bytes,
                    config.max_blocks,
                    false,
                    false,
                    llama_kv_memory_claimant::kv,
                    llama_kv_io_class::capacity_write,
                    (int32_t) bounded_priority,
                    target_bytes,
            });
            const bool offload_matches_decision =
                result.offload.action == llama_kv_action::offload &&
                result.offload.decision_id == pressure.decision_id;
            if (offload_matches_decision && result.offload.state_changed) {
                const auto claimant_it = std::find_if(
                        claimants.begin(), claimants.end(),
                        [selected](const server_kv_claimant_snapshot & item) {
                            return item.seq_id == selected->seq_id;
                        });
                if (claimant_it != claimants.end()) {
                    state.record_offload_feedback(
                            *claimant_it, result.offload, pressure.sample_count);
                }
            }
            const bool offload_progressed = offload_matches_decision &&
                !result.offload.io_failure && result.offload.state_changed &&
                result.offload.relieved_bytes > 0;
            const bool offload_retry_scheduled = offload_matches_decision &&
                result.offload.io_failure && !result.offload.fail_stop;
            if (offload_matches_decision) {
                state.pressure_debt_bytes_ = saturating_sub(
                        state.pressure_debt_bytes_, result.offload.relieved_bytes);
            }
            const bool claimant_exhausted = offload_matches_decision &&
                result.offload.outcome == llama_kv_action_outcome::no_op &&
                result.offload.reason == llama_kv_action_reason::no_candidate &&
                !result.offload.state_changed &&
                result.offload.relieved_bytes == 0;
            if (offload_matches_decision && (result.offload.io_failure || claimant_exhausted)) {
                if (claimant_exhausted) {
                    state.exhausted_claimants_[selected->seq_id] = result.selected_claimant_epoch;
                }
                auto & failures = state.failure_counts_[selected->seq_id];
                if (failures < config.max_failure_penalty) ++failures;
                state.next_action_sample_ = result.offload.io_failure
                    ? saturating_add(
                            pressure.sample_count,
                            std::max<uint32_t>(config.io_failure_backoff_samples, 1))
                    : next_cooldown;
            } else if (offload_matches_decision) {
                state.failure_counts_.erase(selected->seq_id);
                state.next_action_sample_ = next_cooldown;
            } else {
                state.next_action_sample_ = next_cooldown;
            }
            state.idle_follow_up_pending_ = state.pressure_debt_bytes_ > 0 &&
                (offload_progressed || claimant_exhausted || offload_retry_scheduled);
            observation.reason = "offload_submitted";
        }
    }

    if (state.pressure_debt_bytes_ == 0) {
        state.offload_armed_ = false;
        state.idle_follow_up_pending_ = false;
    }
    result.debt_after_bytes = state.pressure_debt_bytes_;
    result.offload_armed_after = state.offload_armed_;
    result.next_action_sample = state.next_action_sample_;
    return result;
}

server_kv_pressure_action_result server_kv_pressure_execute_governor(
        const server_kv_pressure_unified_action_config & config,
        server_kv_governor_state & state,
        const server_kv_pressure_action_ops & ops,
        const server_kv_pressure_snapshot & pressure,
        const std::vector<server_kv_claimant_snapshot> & claimants) {
    return server_kv_pressure_execute_governor(
            config, state, ops, pressure, claimants, server_kv_budget_view {});
}

std::string server_kv_pressure_unified_action_format_marker(
        const server_kv_pressure_action_result & result) {
    const auto & observation = result.observation;
    const auto & evaluation = result.evaluation;
    const auto & action = observation.offload_attempted ? result.offload : result.release;
    std::ostringstream out;
    out << "kv_pressure_unified_action"
        << " policy=" << server_kv_pressure_policy_name(result.policy)
        << " decision_fallback="
        << (result.decision_fallback_reason && result.decision_fallback_reason[0]
                ? result.decision_fallback_reason : "none")
        << " state=" << kv_pressure_state_name(observation.pressure_state)
        << " source=" << kv_pressure_source_name(observation.pressure_source)
        << " sample_valid=" << (observation.sample_valid ? 1 : 0)
        << " stale=" << (observation.stale ? 1 : 0)
        << " pressure_basis_valid=" << (observation.pressure_basis_valid ? 1 : 0)
        << " decision_id=" << observation.decision_id
        << " episode=" << result.episode
        << " target_bytes=" << observation.target_bytes
        << " max_blocks=" << observation.max_blocks
        << " observed_excess_bytes=" << result.observed_excess_bytes
        << " debt_before_bytes=" << result.debt_before_bytes
        << " debt_after_bytes=" << result.debt_after_bytes
        << " budget_active=" << (result.budget_active ? 1 : 0)
        << " budget_target_enabled=" << (result.budget_target_enabled ? 1 : 0)
        << " budget_source=" << (result.budget_source ? result.budget_source : "none")
        << " budget_target_bytes=" << result.budget_target_bytes
        << " budget_basis_generation=" << result.budget_basis_generation
        << " budget_view_valid=" << (result.budget_view_valid ? 1 : 0)
        << " budget_resident_available=" << (result.budget_resident_available ? 1 : 0)
        << " budget_reclaimable_available=" << (result.budget_reclaimable_available ? 1 : 0)
        << " budget_resident_bytes=" << result.budget_resident_bytes
        << " budget_dead_resident_reclaimable_bytes="
        << result.budget_dead_resident_reclaimable_bytes
        << " budget_transient_staging_bound_bytes="
        << result.budget_transient_staging_bound_bytes
        << " budget_observed_excess_bytes=" << result.budget_observed_excess_bytes
        << " budget_debt_before_bytes=" << result.budget_debt_before_bytes
        << " budget_debt_after_bytes=" << result.budget_debt_after_bytes
        << " soft_offload_armed_before=" << (result.soft_offload_armed_before ? 1 : 0)
        << " soft_offload_armed_after=" << (result.soft_offload_armed_after ? 1 : 0)
        << " budget_next_action_sample=" << result.budget_next_action_sample
        << " unmet_budget_bytes_after=" << result.unmet_budget_bytes_after
        << " offload_armed_before=" << (result.offload_armed_before ? 1 : 0)
        << " offload_armed_after=" << (result.offload_armed_after ? 1 : 0)
        << " next_action_sample=" << result.next_action_sample
        << " evaluate_attempted=" << (observation.evaluate_attempted ? 1 : 0)
        << " evaluate_outcome=" << outcome_name(evaluation.outcome)
        << " evaluate_reason=" << reason_name(evaluation.reason)
        << " release_attempted=" << (observation.release_attempted ? 1 : 0)
        << " offload_attempted=" << (observation.offload_attempted ? 1 : 0)
        << " selected_seq_id=" << result.selected_seq_id
        << " selected_claimant_epoch=" << result.selected_claimant_epoch
        << " transaction_id=" << action.core_transaction_id
        << " outcome=" << outcome_name(action.outcome)
        << " reason=" << reason_name(action.reason)
        << " blocks=" << action.blocks
        << " bytes=" << action.bytes
        << " relieved_bytes=" << action.relieved_bytes
        << " shortfall_bytes=" << action.shortfall_bytes
        << " action_elapsed_us=" << action.action_elapsed_us
        << " physical_relief_available=" << (action.physical_relief_available ? 1 : 0)
        << " physical_relief_bytes=" << action.physical_relief_bytes
        << " physical_object_id=" << action.physical_object_id
        << " physical_generation=" << action.physical_generation
        << " io_failure=" << (action.io_failure ? 1 : 0)
        << " io_errno=" << action.io_errno
        << " state_changed=" << (action.state_changed ? 1 : 0)
        << " decision_reason=" << observation.reason
        << " sample_count=" << observation.sample_count
        << " idle=" << (observation.idle ? 1 : 0)
        << " claimants=";
    if (result.runtime_claimants.empty()) {
        out << "none";
    } else {
        for (size_t i = 0; i < result.runtime_claimants.size(); ++i) {
            if (i != 0) out << ';';
            const auto & claimant = result.runtime_claimants[i];
            out << claimant.seq_id << ':'
                << claimant.epoch << ':'
                << (claimant.active ? 1 : 0) << ':'
                << (claimant.exhausted ? 1 : 0) << ':'
                << (claimant.runtime.valid ? 1 : 0) << ':'
                << claimant.runtime.target_blocks << ':'
                << claimant.runtime.eligible_resident_blocks << ':'
                << claimant.runtime.swapped_blocks << ':'
                << claimant.runtime.shared_blocks << ':'
                << claimant.runtime.blocked_blocks;
        }
    }
    out << " scores=";
    if (result.scores.empty()) {
        out << "none";
    } else {
        for (size_t i = 0; i < result.scores.size(); ++i) {
            if (i != 0) out << ';';
            const auto & score = result.scores[i];
            out << score.seq_id << ':'
                << (score.eligible ? 1 : 0) << ':'
                << exclusion_name(score.exclusion) << ':'
                << score.total << ':'
                << score.idle_age_score << ':'
                << score.logical_kv_score << ':'
                << score.reclaimable_score << ':'
                << score.lcp_n_past_penalty << ':'
                << score.io_cost_penalty << ':'
                << score.failure_penalty << ':'
                << score.rank << ':'
                << (score.cost_aware ? 1 : 0) << ':'
                << (score.physical_estimate_available ? 1 : 0) << ':'
                << (score.physical_estimate_authoritative ? 1 : 0) << ':'
                << score.estimated_physical_bytes << ':'
                << score.reuse_probability_ppm << ':'
                << score.expected_offload_write_cost_us << ':'
                << score.expected_restore_gate_cost_us << ':'
                << score.expected_cost_us << ':'
                << score.churn_penalty_us << ':'
                << score.raw_idle_age_us << ':'
                << score.raw_answer_tokens << ':'
                << score.raw_lcp_hint_tokens << ':'
                << score.physical_object_id << ':'
                << score.physical_generation << ':'
                << score.actual_relief_bytes << ':'
                << score.resident_lease_until_sample << ':'
                << score.round_trip_count << ':'
                << score.last_offload_bytes << ':'
                << score.last_restore_bytes << ':'
                << (score.fallback_reason ? score.fallback_reason : "none");
        }
    }
    return out.str();
}
