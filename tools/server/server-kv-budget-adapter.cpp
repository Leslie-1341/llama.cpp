#include "server-kv-budget-adapter.h"

#include <limits>
#include <sstream>
#include <string>

namespace {

constexpr const char * AUTH_UNAVAILABLE = "UNAVAILABLE";
constexpr const char * AUTH_RESIDENT = "kv_physical_mincore";
constexpr const char * AUTH_RECLAIMABLE = "kv_release_eligibility";
constexpr const char * AUTH_DERIVED_NON_RECLAIMABLE =
        "kv_physical_resident_minus_release_eligibility";
constexpr const char * AUTH_CREDIT = "kv_physical_mincore_delta";
constexpr const char * SOURCE_GLOBAL_DYNAMIC = "global_dynamic";

server_kv_budget_scalar unavailable_scalar(uint64_t object_id = 0, uint64_t generation = 0) {
    server_kv_budget_scalar result;
    result.authority = AUTH_UNAVAILABLE;
    result.object_id = object_id;
    result.generation = generation;
    return result;
}

server_kv_budget_scalar scalar(
        uint64_t value,
        bool available,
        const char * authority,
        uint64_t object_id,
        uint64_t generation) {
    server_kv_budget_scalar result;
    result.value = value;
    result.available = available;
    result.authority = available && authority ? authority : AUTH_UNAVAILABLE;
    result.object_id = object_id;
    result.generation = generation;
    return result;
}

bool same_target(
        const server_kv_resident_target_state & lhs,
        const server_kv_resident_target_state & rhs) {
    const char * lhs_source = lhs.source ? lhs.source : AUTH_UNAVAILABLE;
    const char * rhs_source = rhs.source ? rhs.source : AUTH_UNAVAILABLE;
    return lhs.enabled == rhs.enabled &&
        lhs.target_bytes == rhs.target_bytes &&
        std::string(lhs_source) == rhs_source &&
        lhs.source_object_id == rhs.source_object_id &&
        lhs.source_generation == rhs.source_generation;
}

} // namespace

void server_kv_budget_adapter::reset() {
    static_target_ = {};
    global_target_ = {};
    global_target_available_ = false;
    effective_generation_ = 0;
    effective_target_ = {};
}

server_kv_resident_target_state server_kv_budget_adapter::initialize_static_target(
        std::string & error) {
    static_target_ = server_kv_resident_target_from_env(error);
    refresh_effective_target();
    return effective_target_;
}

bool server_kv_budget_adapter::set_global_target(
        const server_kv_global_target_update & update) {
    if (!update.available || update.target_kv_resident_bytes == 0 ||
            update.object_id == 0 || update.generation == 0) {
        return false;
    }

    server_kv_resident_target_state next;
    next.enabled = true;
    next.target_bytes = update.target_kv_resident_bytes;
    next.source = SOURCE_GLOBAL_DYNAMIC;
    next.source_object_id = update.object_id;
    next.source_generation = update.generation;

    if (!global_target_available_ || !same_target(global_target_, next)) {
        global_target_ = next;
        global_target_available_ = true;
        refresh_effective_target();
    }
    return true;
}

void server_kv_budget_adapter::clear_global_target() {
    if (!global_target_available_) {
        return;
    }
    global_target_ = {};
    global_target_available_ = false;
    refresh_effective_target();
}

server_kv_resident_target_state server_kv_budget_adapter::effective_target() const {
    return effective_target_;
}

void server_kv_budget_adapter::apply_effective_target(
        server_kv_pressure_unified_action_config & config) const {
    const auto & target = effective_target_;
    config.budget_target_enabled = target.valid();
    config.budget_target_bytes = target.valid() ? target.target_bytes : 0;
    config.budget_basis_generation = target.valid() ? target.basis_generation : 0;
    config.budget_source = target.valid() ? target.source : AUTH_UNAVAILABLE;
    config.budget_source_object_id = target.valid() ? target.source_object_id : 0;
    config.budget_source_generation = target.valid() ? target.source_generation : 0;
}

server_kv_global_budget_view server_kv_budget_adapter::project(
        const llama_kv_physical_budget_view & view) const {
    server_kv_global_budget_view result;
    const bool identity_available = view.object_id != 0 && view.generation != 0;
    const bool base_valid = view.valid && identity_available;
    const uint64_t object_id = identity_available ? view.object_id : 0;
    const uint64_t generation = identity_available ? view.generation : 0;

    result.valid = base_valid;
    result.global_scheduler_available = base_valid;
    result.global_authority = base_valid ? "kv_physical_budget_view" : AUTH_UNAVAILABLE;
    result.resident_bytes = scalar(
            view.resident_bytes,
            base_valid && view.resident_available,
            AUTH_RESIDENT,
            object_id,
            generation);
    result.reclaimable_bytes = scalar(
            view.dead_resident_reclaimable_bytes,
            base_valid && view.reclaimable_available,
            AUTH_RECLAIMABLE,
            object_id,
            generation);

    // A whole-KV release-reclaimable scalar is not a per-claimant protected
    // aggregate. Without authoritative claimant decomposition these fields
    // stay unavailable rather than treating resident - reclaimable as protected.
    result.protected_resident_bytes = unavailable_scalar(object_id, generation);
    result.non_reclaimable_resident_bytes = unavailable_scalar(object_id, generation);

    result.reclaim_cost_us_per_mib = unavailable_scalar(object_id, generation);
    result.grow_value = unavailable_scalar(object_id, generation);
    result.recommended_target_bytes = unavailable_scalar(object_id, generation);
    result.knee_bytes = unavailable_scalar(object_id, generation);
    result.confirmed_credit_bytes = unavailable_scalar(object_id, generation);
    return result;
}

server_kv_budget_view server_kv_budget_adapter::project_for_governor(
        const llama_kv_physical_budget_view & view) const {
    server_kv_budget_view result;
    const bool identity_available = view.object_id != 0 && view.generation != 0;
    result.valid = view.valid && identity_available;
    result.resident_available = result.valid && view.resident_available;
    result.reclaimable_available = result.valid && view.reclaimable_available;
    result.swapped_metadata_consistent = result.valid && view.swapped_metadata_consistent;
    result.object_id = identity_available ? view.object_id : 0;
    result.generation = identity_available ? view.generation : 0;
    result.authority = result.valid ? "kv_physical_budget_view" : AUTH_UNAVAILABLE;
    result.resident_bytes = view.resident_bytes;
    result.dead_resident_reclaimable_bytes = view.dead_resident_reclaimable_bytes;
    result.transient_staging_bound_bytes = view.transient_staging_bound_bytes;
    return result;
}

server_kv_budget_scalar server_kv_budget_adapter::confirm_physical_credit(
        const llama_kv_resident_sample & before,
        const llama_kv_resident_sample & after) {
    if (!before.available || !after.available || before.object_id == 0 ||
            before.generation == 0 || before.object_id != after.object_id ||
            before.generation != after.generation ||
            before.resident_bytes <= after.resident_bytes) {
        return unavailable_scalar();
    }
    return scalar(
            before.resident_bytes - after.resident_bytes,
            true,
            AUTH_CREDIT,
            before.object_id,
            before.generation);
}

void server_kv_budget_adapter::refresh_effective_target() {
    const server_kv_resident_target_state next =
            global_target_available_ ? global_target_ : static_target_;
    if (same_target(effective_target_, next)) {
        return;
    }

    effective_generation_ = effective_generation_ == 0
        ? 1
        : next_generation(effective_generation_);
    effective_target_ = next;
    effective_target_.basis_generation = effective_generation_;
}

uint64_t server_kv_budget_adapter::next_generation(uint64_t generation) {
    return generation == std::numeric_limits<uint64_t>::max()
        ? generation
        : generation + 1;
}

std::string server_kv_global_budget_view_format_marker(
        const server_kv_global_budget_view & view) {
    auto append = [](std::ostringstream & out, const char * name,
                     const server_kv_budget_scalar & value) {
        out << ' ' << name << "_available=" << (value.available ? 1 : 0)
            << ' ' << name << "_authority="
            << (value.authority ? value.authority : AUTH_UNAVAILABLE)
            << ' ' << name << "_object_id=" << value.object_id
            << ' ' << name << "_generation=" << value.generation
            << ' ' << name << "_bytes=" << value.value;
    };

    std::ostringstream out;
    out << "kv_global_budget_view"
        << " valid=" << (view.valid ? 1 : 0)
        << " global_scheduler_available=" << (view.global_scheduler_available ? 1 : 0)
        << " global_authority="
        << (view.global_authority ? view.global_authority : AUTH_UNAVAILABLE);
    append(out, "resident", view.resident_bytes);
    append(out, "reclaimable", view.reclaimable_bytes);
    append(out, "protected_resident", view.protected_resident_bytes);
    append(out, "non_reclaimable_resident", view.non_reclaimable_resident_bytes);
    append(out, "reclaim_cost_us_per_mib", view.reclaim_cost_us_per_mib);
    append(out, "grow_value", view.grow_value);
    append(out, "recommended_target", view.recommended_target_bytes);
    append(out, "knee", view.knee_bytes);
    append(out, "confirmed_credit", view.confirmed_credit_bytes);
    return out.str();
}
