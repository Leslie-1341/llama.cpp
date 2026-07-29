#include "server-kv-pressure-action.h"

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

std::string server_kv_pressure_unified_action_format_marker(
        const server_kv_pressure_action_result & result) {
    const auto & observation = result.observation;
    const auto & evaluation = result.evaluation;
    const auto & release = result.release;
    std::ostringstream out;
    out << "kv_pressure_unified_action"
        << " state=" << kv_pressure_state_name(observation.pressure_state)
        << " source=" << kv_pressure_source_name(observation.pressure_source)
        << " stale=" << (observation.stale ? 1 : 0)
        << " decision_id=" << observation.decision_id
        << " target_bytes=" << observation.target_bytes
        << " max_blocks=" << observation.max_blocks
        << " evaluate_attempted=" << (observation.evaluate_attempted ? 1 : 0)
        << " evaluate_outcome=" << outcome_name(evaluation.outcome)
        << " evaluate_reason=" << reason_name(evaluation.reason)
        << " release_attempted=" << (observation.release_attempted ? 1 : 0)
        << " transaction_id=" << release.core_transaction_id
        << " outcome=" << outcome_name(release.outcome)
        << " reason=" << reason_name(release.reason)
        << " blocks=" << release.blocks
        << " bytes=" << release.bytes
        << " shortfall_bytes=" << release.shortfall_bytes
        << " io_failure=" << (release.io_failure ? 1 : 0)
        << " state_changed=" << (release.state_changed ? 1 : 0)
        << " decision_reason=" << observation.reason
        << " sample_count=" << observation.sample_count
        << " idle=" << (observation.idle ? 1 : 0);
    return out.str();
}
