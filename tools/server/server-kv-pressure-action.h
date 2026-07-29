#pragma once

#include "llama-kv-cache-action.h"
#include "llama-kv-pressure.h"

#include <cstdint>
#include <functional>
#include <string>

struct server_kv_pressure_unified_action_config {
    static constexpr uint32_t DEFAULT_MAX_BLOCKS = 64;

    bool enabled = false;
    uint64_t target_bytes = 0;
    uint32_t max_blocks = DEFAULT_MAX_BLOCKS;
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
    const char * reason = "disabled";
};

struct server_kv_pressure_action_result {
    server_kv_pressure_action_observation observation;
    llama_kv_action_result evaluation;
    llama_kv_action_result release;
};

server_kv_pressure_action_result server_kv_pressure_execute_unified_action(
        const server_kv_pressure_unified_action_config & config,
        const server_kv_pressure_action_ops & ops,
        const kv_pressure_telemetry & telemetry,
        bool idle,
        uint64_t sample_count,
        uint64_t decision_id);

std::string server_kv_pressure_unified_action_format_marker(
        const server_kv_pressure_action_result & result);
