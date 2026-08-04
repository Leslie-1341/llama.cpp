#include "server-kv-resume.h"

#include <cstdio>
#include <sstream>

namespace {

const char * action_outcome_name(llama_kv_action_outcome outcome) {
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

const char * action_reason_name(llama_kv_action_reason reason) {
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

} // namespace

server_kv_resume_gate_result server_kv_resume_gate(
        const server_kv_resume_ops & ops,
        server_kv_resume_trigger trigger,
        llama_seq_id seq_id,
        uint64_t decision_id) {
    server_kv_resume_gate_result result;
    result.decision_id = decision_id;

    if (!ops.set_protected || !ops.execute) {
        return result;
    }

    ops.set_protected(seq_id, true);
    result.action = ops.execute({
            llama_kv_action::prefetch,
            decision_id,
            seq_id,
            0,
            0,
            true,
            true,
    });

    const bool outcome_ok =
        result.action.outcome == llama_kv_action_outcome::completed ||
        result.action.outcome == llama_kv_action_outcome::no_op;
    result.graph_allowed =
        result.action.decision_id == decision_id &&
        outcome_ok &&
        !result.action.io_failure &&
        !result.action.fail_stop &&
        !result.action.capability.context_invalid &&
        result.action.shortfall_bytes == 0;

    (void) trigger;
    return result;
}

std::string server_kv_resume_failure_message(const server_kv_resume_gate_result & result) {
    char buffer[384];
    const auto & action = result.action;
    std::snprintf(buffer, sizeof(buffer),
            "KV resume prefetch failed: decision_id=%llu transaction_id=%llu "
            "completed_blocks=%u completed_bytes=%llu shortfall_bytes=%llu "
            "outcome=%s reason=%s io_failure=%d fail_stop=%d context_invalid=%d",
            (unsigned long long) result.decision_id,
            (unsigned long long) action.core_transaction_id,
            action.blocks,
            (unsigned long long) action.bytes,
            (unsigned long long) action.shortfall_bytes,
            action_outcome_name(action.outcome),
            action_reason_name(action.reason),
            action.io_failure ? 1 : 0,
            action.fail_stop ? 1 : 0,
            action.capability.context_invalid ? 1 : 0);
    return buffer;
}


std::string server_kv_resume_format_event(
        const server_kv_resume_gate_result & result,
        llama_seq_id seq_id,
        uint64_t claimant_epoch,
        bool graph_gate) {
    const auto & action = result.action;
    std::ostringstream out;
    out << "kv_resume_order_event"
        << " phase=" << (graph_gate ? "graph_gate" : "prefetch")
        << " decision_id=" << result.decision_id
        << " seq_id=" << seq_id
        << " claimant_epoch=" << claimant_epoch
        << " transaction_id=" << action.core_transaction_id
        << " action=prefetch"
        << " outcome=" << action_outcome_name(action.outcome)
        << " reason=" << action_reason_name(action.reason)
        << " graph_allowed=" << (result.graph_allowed ? 1 : 0);
    return out.str();
}
