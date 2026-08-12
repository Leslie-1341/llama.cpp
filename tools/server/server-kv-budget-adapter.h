#pragma once

#include "server-kv-pressure-action.h"

#include <cstdint>
#include <string>

// A scalar crossing the KV -> Global boundary. Every value is bound to the
// physical KV object and generation that produced it. Unavailable values are
// never represented as zero-valued evidence.
struct server_kv_budget_scalar {
    uint64_t value = 0;
    bool available = false;
    const char * authority = "UNAVAILABLE";
    uint64_t object_id = 0;
    uint64_t generation = 0;
};

struct server_kv_global_budget_view {
    bool valid = false;
    bool global_scheduler_available = false;
    const char * global_authority = "UNAVAILABLE";

    server_kv_budget_scalar resident_bytes;
    server_kv_budget_scalar reclaimable_bytes;
    server_kv_budget_scalar protected_resident_bytes;
    server_kv_budget_scalar non_reclaimable_resident_bytes;

    // No calibrated restore-cost curve is part of the V1A runtime contract.
    server_kv_budget_scalar reclaim_cost_us_per_mib;
    server_kv_budget_scalar grow_value;
    server_kv_budget_scalar recommended_target_bytes;
    server_kv_budget_scalar knee_bytes;

    // Confirmed only by an authoritative physical before/after drop with the
    // same object and generation.
    server_kv_budget_scalar confirmed_credit_bytes;
};

struct server_kv_global_target_update {
    bool available = false;
    uint64_t target_kv_resident_bytes = 0;
    uint64_t object_id = 0;
    uint64_t generation = 0;
};

class server_kv_budget_adapter {
public:
    void reset();

    // The static env source is only a fallback. It has no physical source
    // identity; a valid dynamic target must carry object/generation identity.
    server_kv_resident_target_state initialize_static_target(std::string & error);
    bool set_global_target(const server_kv_global_target_update & update);
    void clear_global_target();

    server_kv_resident_target_state effective_target() const;
    void apply_effective_target(
            server_kv_pressure_unified_action_config & config) const;

    server_kv_global_budget_view project(
            const llama_kv_physical_budget_view & view) const;
    server_kv_budget_view project_for_governor(
            const llama_kv_physical_budget_view & view) const;

    static server_kv_budget_scalar confirm_physical_credit(
            const llama_kv_resident_sample & before,
            const llama_kv_resident_sample & after);

private:
    void refresh_effective_target();
    static uint64_t next_generation(uint64_t generation);

    server_kv_resident_target_state static_target_;
    server_kv_resident_target_state global_target_;
    bool global_target_available_ = false;
    uint64_t effective_generation_ = 0;
    server_kv_resident_target_state effective_target_;
};

std::string server_kv_global_budget_view_format_marker(
        const server_kv_global_budget_view & view);
