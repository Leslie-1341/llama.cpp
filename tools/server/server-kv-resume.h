#pragma once

#include "llama-kv-cache-action.h"

#include <cstdint>
#include <functional>
#include <string>

enum class server_kv_resume_trigger : uint8_t {
    resume,
    active_access,
};

struct server_kv_resume_ops {
    std::function<void(llama_seq_id, bool)> set_protected;
    std::function<llama_kv_action_result(const llama_kv_action_request &)> execute;
};

struct server_kv_resume_gate_result {
    uint64_t decision_id = 0;
    llama_kv_action_result action;
    bool graph_allowed = true;
};

server_kv_resume_gate_result server_kv_resume_gate(
        const server_kv_resume_ops & ops,
        server_kv_resume_trigger trigger,
        llama_seq_id seq_id,
        uint64_t decision_id);

std::string server_kv_resume_failure_message(const server_kv_resume_gate_result & result);

std::string server_kv_resume_format_event(
        const server_kv_resume_gate_result & result,
        llama_seq_id seq_id,
        uint64_t claimant_epoch,
        bool graph_gate);
