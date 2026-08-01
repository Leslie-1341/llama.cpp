#include "server-kv-pressure-action.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
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
    case server_kv_claimant_exclusion::empty:                  return "empty";
    }
    return "unknown";
}

server_kv_claimant_score score_claimant(
        const server_kv_claimant_snapshot & claimant,
        const llama_kv_action_result & evaluation,
        uint32_t failure_count,
        uint64_t current_epoch,
        bool exhausted) {
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
    if (claimant.seq_id < 0 || claimant.reclaimable_bytes == 0 || claimant.logical_kv_tokens == 0) {
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

} // namespace

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

    config.enabled = true;
    decision.status = server_kv_pressure_unified_action_startup_status::enabled;
    return decision;
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
    observation.stale = telemetry.stale;
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
    next_action_sample_ = 0;
    claimant_epochs_.clear();
    exhausted_claimants_.clear();
    failure_counts_.clear();
}

uint64_t server_kv_governor_state::claimant_epoch(llama_seq_id seq_id) const {
    const auto it = claimant_epochs_.find(seq_id);
    return it == claimant_epochs_.end() ? 1 : it->second;
}

void server_kv_governor_state::invalidate_claimant(llama_seq_id seq_id) {
    if (seq_id < 0) {
        return;
    }
    const uint64_t next_epoch = saturating_add(claimant_epoch(seq_id), 1);
    claimant_epochs_[seq_id] = next_epoch == 0 ? 1 : next_epoch;
    exhausted_claimants_.erase(seq_id);
    failure_counts_.erase(seq_id);
}

void server_kv_governor_state::invalidate_all_claimants() {
    exhausted_claimants_.clear();
    failure_counts_.clear();
    for (auto & entry : claimant_epochs_) {
        const uint64_t next_epoch = saturating_add(entry.second, 1);
        entry.second = next_epoch == 0 ? 1 : next_epoch;
    }
}

server_kv_pressure_action_result server_kv_pressure_execute_governor(
        const server_kv_pressure_unified_action_config & config,
        server_kv_governor_state & state,
        const server_kv_pressure_action_ops & ops,
        const server_kv_pressure_snapshot & pressure,
        const std::vector<server_kv_claimant_snapshot> & claimants) {
    server_kv_pressure_action_result result;
    auto & observation = result.observation;
    observation.pressure_state = pressure.state;
    observation.pressure_source = pressure.source;
    observation.stale = pressure.stale;
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

    if (!config.enabled) {
        return result;
    }
    if (!ops.execute) {
        observation.reason = "no_memory";
        return result;
    }
    if (!pressure.sample_valid || pressure.stale) {
        observation.reason = "stale_hold";
        return result;
    }
    if (pressure.state != kv_pressure_state::PRESSURE &&
            pressure.state != kv_pressure_state::CRITICAL) {
        state.reset();
        result.episode = 0;
        result.debt_after_bytes = 0;
        result.offload_armed_after = false;
        result.next_action_sample = 0;
        observation.reason = "not_pressure_reset";
        return result;
    }
    if (!pressure.pressure_basis_valid) {
        observation.reason = "invalid_basis_hold";
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
        if (release_matches_decision) {
            state.pressure_debt_bytes_ = saturating_sub(
                    state.pressure_debt_bytes_, result.release.relieved_bytes);
        }
        if (release_matches_decision &&
                result.release.reason == llama_kv_action_reason::no_candidate &&
                state.pressure_debt_bytes_ > 0) {
            state.offload_armed_ = true;
        }
        if (result.release.io_failure) {
            state.next_action_sample_ = saturating_add(
                    pressure.sample_count,
                    std::max<uint32_t>(config.io_failure_backoff_samples, 1));
        } else {
            state.next_action_sample_ = next_cooldown;
        }
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
            result.scores.push_back(score_claimant(
                    claimant, result.evaluation, failure_count, current_epoch, exhausted));
        }
        std::sort(result.scores.begin(), result.scores.end(),
                [](const server_kv_claimant_score & lhs, const server_kv_claimant_score & rhs) {
                    if (lhs.eligible != rhs.eligible) return lhs.eligible > rhs.eligible;
                    if (lhs.total != rhs.total) return lhs.total > rhs.total;
                    return lhs.seq_id < rhs.seq_id;
                });

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
            observation.reason = "offload_submitted";
        }
    }

    if (state.pressure_debt_bytes_ == 0) {
        state.offload_armed_ = false;
    }
    result.debt_after_bytes = state.pressure_debt_bytes_;
    result.offload_armed_after = state.offload_armed_;
    result.next_action_sample = state.next_action_sample_;
    return result;
}

std::string server_kv_pressure_unified_action_format_marker(
        const server_kv_pressure_action_result & result) {
    const auto & observation = result.observation;
    const auto & evaluation = result.evaluation;
    const auto & action = observation.offload_attempted ? result.offload : result.release;
    std::ostringstream out;
    out << "kv_pressure_unified_action"
        << " state=" << kv_pressure_state_name(observation.pressure_state)
        << " source=" << kv_pressure_source_name(observation.pressure_source)
        << " stale=" << (observation.stale ? 1 : 0)
        << " decision_id=" << observation.decision_id
        << " episode=" << result.episode
        << " target_bytes=" << observation.target_bytes
        << " max_blocks=" << observation.max_blocks
        << " observed_excess_bytes=" << result.observed_excess_bytes
        << " debt_before_bytes=" << result.debt_before_bytes
        << " debt_after_bytes=" << result.debt_after_bytes
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
        << " io_failure=" << (action.io_failure ? 1 : 0)
        << " io_errno=" << action.io_errno
        << " state_changed=" << (action.state_changed ? 1 : 0)
        << " decision_reason=" << observation.reason
        << " sample_count=" << observation.sample_count
        << " idle=" << (observation.idle ? 1 : 0)
        << " scores=";
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
                << score.failure_penalty;
        }
    }
    return out.str();
}
