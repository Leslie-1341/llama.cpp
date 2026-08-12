#include "server-kv-budget-adapter.h"
#include "server-kv-pressure-action.h"
#include "server-kv-resume.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int tests_total = 0;
static int tests_failed = 0;

static bool check(bool condition, const char * message) {
    ++tests_total;
    if (!condition) {
        ++tests_failed;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

#define CHECK(condition) check((condition), #condition)

struct fake_core {
    std::vector<llama_kv_action_result> responses;
    std::vector<llama_kv_action_request> requests;

    server_kv_pressure_action_ops ops() {
        return { [this](const llama_kv_action_request & request) {
            requests.push_back(request);
            if (responses.empty()) {
                return llama_kv_action_result {};
            }
            const auto response = responses.front();
            responses.erase(responses.begin());
            return response;
        } };
    }
};

static kv_pressure_telemetry pressure(kv_pressure_state state = kv_pressure_state::PRESSURE) {
    kv_pressure_telemetry telemetry;
    telemetry.sample_valid = true;
    telemetry.state = state;
    return telemetry;
}

static llama_kv_action_result evaluation(uint64_t decision_id) {
    llama_kv_action_result result;
    result.action = llama_kv_action::evaluate;
    result.decision_id = decision_id;
    result.outcome = llama_kv_action_outcome::completed;
    result.capability.can_release = true;
    result.capability.can_offload = true;
    return result;
}

static llama_kv_action_result release(uint64_t decision_id) {
    llama_kv_action_result result;
    result.action = llama_kv_action::release;
    result.decision_id = decision_id;
    result.outcome = llama_kv_action_outcome::completed;
    result.state_changed = true;
    result.core_transaction_id = 99;
    result.blocks = 2;
    result.bytes = 4096;
    result.relieved_bytes = 4096;
    return result;
}

static server_kv_pressure_unified_action_config enabled_config() {
    server_kv_pressure_unified_action_config config;
    config.enabled = true;
    config.target_bytes = 4096;
    config.max_blocks = 3;
    return config;
}

static server_kv_pressure_unified_action_config budget_config() {
    auto config = enabled_config();
    config.budget_target_enabled = true;
    config.budget_target_bytes = 4096;
    config.budget_basis_generation = 7;
    config.budget_source = "env_static";
    return config;
}

static server_kv_budget_view budget_view(uint64_t resident_bytes, bool valid = true) {
    server_kv_budget_view view;
    view.valid = valid;
    view.resident_available = valid;
    view.reclaimable_available = valid;
    view.swapped_metadata_consistent = valid;
    view.object_id = 11;
    view.generation = 3;
    view.resident_bytes = resident_bytes;
    view.dead_resident_reclaimable_bytes = resident_bytes > 4096
        ? resident_bytes - 4096
        : 0;
    view.transient_staging_bound_bytes = 1234;
    return view;
}

static void clear_pressure_action_env() {
    unsetenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION");
    unsetenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES");
    unsetenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS");
    unsetenv("LLAMA_KV_PAGED_RELEASE");
    unsetenv("LLAMA_KV_PRESSURE_DRY_RUN");
    unsetenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE");
    unsetenv("LLAMA_KV_RESIDENT_TARGET_BYTES");
    unsetenv("LLAMA_KV_RESIDENT_TARGET_SOURCE");
    unsetenv("LLAMA_KV_PRESSURE_POLICY");
    unsetenv("LLAMA_KV_PRESSURE_RESIDENT_LEASE_SAMPLES");
    unsetenv("LLAMA_KV_PRESSURE_CHURN_PENALTY_US");
}

static uint32_t release_request_count(const fake_core & core, uint64_t decision_id) {
    uint32_t count = 0;
    for (const auto & request : core.requests) {
        if (request.action == llama_kv_action::release && request.decision_id == decision_id) {
            ++count;
        }
    }
    return count;
}

static server_kv_pressure_unified_action_startup_decision startup_decision(
        const char * target, const char * max_blocks) {
    clear_pressure_action_env();
    setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION", "1", 1);
    if (target) {
        setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", target, 1);
    }
    if (max_blocks) {
        setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS", max_blocks, 1);
    }
    return server_kv_pressure_unified_action_startup_decide_from_env();
}

static void test_startup_rejects_global_legacy_mode_conflict() {
    std::string error;
    CHECK(!server_kv_pressure_global_kv_legacy_modes_conflict(false, true, error));
    CHECK(error.empty());

    error.clear();
    CHECK(!server_kv_pressure_global_kv_legacy_modes_conflict(true, false, error));
    CHECK(error.empty());

    error.clear();
    CHECK(server_kv_pressure_global_kv_legacy_modes_conflict(true, true, error));
    CHECK(error.find("LLAMA_MEMORY_GOVERNOR_LEGACY_KV_ACTIONS=1") != std::string::npos);
}

static void test_startup_decision_distinguishes_disabled_and_enabled() {
    clear_pressure_action_env();
    setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", "invalid", 1);
    setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS", "0", 1);
    const auto disabled = server_kv_pressure_unified_action_startup_decide_from_env();
    CHECK(disabled.status == server_kv_pressure_unified_action_startup_status::disabled);
    CHECK(!disabled.enablement.requested && !disabled.config.enabled);

    const auto enabled = startup_decision("4096", "3");
    CHECK(enabled.status == server_kv_pressure_unified_action_startup_status::enabled);
    CHECK(enabled.enablement.requested && enabled.config.enabled);
    CHECK(enabled.config.target_bytes == 4096 && enabled.config.max_blocks == 3);
    clear_pressure_action_env();
}

static void test_startup_decision_rejects_invalid_unified_configuration() {
    struct invalid_case {
        const char * target;
        const char * max_blocks;
        const char * invalid_env;
    };
    const invalid_case cases[] = {
        { "invalid", "3", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES" },
        { "4096x", "3", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES" },
        { "0", "3", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES" },
        { "18446744073709551616", "3", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES" },
        { "4096", "invalid", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS" },
        { "4096", "3x", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS" },
        { "4096", "0", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS" },
        { "4096", "4294967296", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS" },
        { nullptr, "3", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES" },
        { "4096", nullptr, "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS" },
    };
    const char * conflicting_envs[] = {
        "LLAMA_KV_PAGED_RELEASE",
        "LLAMA_KV_PRESSURE_DRY_RUN",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE",
    };

    for (const auto & invalid : cases) {
        const auto decision = startup_decision(invalid.target, invalid.max_blocks);
        CHECK(decision.status == server_kv_pressure_unified_action_startup_status::invalid);
        CHECK(decision.enablement.requested && !decision.config.enabled);
        CHECK(decision.error.find(invalid.invalid_env) != std::string::npos);

        for (const char * conflicting_env : conflicting_envs) {
            startup_decision(invalid.target, invalid.max_blocks);
            setenv(conflicting_env, "1", 1);
            const auto conflict_decision =
                    server_kv_pressure_unified_action_startup_decide_from_env();
            CHECK(conflict_decision.status ==
                    server_kv_pressure_unified_action_startup_status::conflict);
            CHECK(conflict_decision.enablement.requested && !conflict_decision.config.enabled);
            CHECK(conflict_decision.error.find(conflicting_env) != std::string::npos);
        }
    }
    clear_pressure_action_env();
}

static void test_startup_decision_preserves_legacy_modes_when_unified_is_not_requested() {
    const char * legacy_envs[] = {
        "LLAMA_KV_PAGED_RELEASE",
        "LLAMA_KV_PRESSURE_DRY_RUN",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE",
    };
    for (const char * legacy_env : legacy_envs) {
        clear_pressure_action_env();
        setenv(legacy_env, "1", 1);
        setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", "invalid", 1);
        setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS", "0", 1);
        const auto decision = server_kv_pressure_unified_action_startup_decide_from_env();
        CHECK(decision.status == server_kv_pressure_unified_action_startup_status::disabled);
        CHECK(!decision.enablement.requested && !decision.config.enabled);
    }
    clear_pressure_action_env();
}

static void test_default_disabled_and_non_pressure_noop() {
    fake_core core;
    const auto disabled = server_kv_pressure_execute_unified_action(
            {}, core.ops(), pressure(), false, 1, 1001);
    CHECK(core.requests.empty());
    CHECK(std::string(disabled.observation.reason) == "disabled");

    const auto normal = server_kv_pressure_execute_unified_action(
            enabled_config(), core.ops(), pressure(kv_pressure_state::NORMAL), false, 2, 1002);
    CHECK(core.requests.empty());
    CHECK(std::string(normal.observation.reason) == "not_pressure");

    const auto recovery = server_kv_pressure_execute_unified_action(
            enabled_config(), core.ops(), pressure(kv_pressure_state::RECOVERY), false, 3, 1003);
    CHECK(core.requests.empty());
    CHECK(std::string(recovery.observation.reason) == "not_pressure");
}

static void test_stale_and_no_memory_are_blocked() {
    fake_core core;
    auto stale = pressure();
    stale.stale = true;
    const auto stale_result = server_kv_pressure_execute_unified_action(
            enabled_config(), core.ops(), stale, false, 1, 1010);
    CHECK(core.requests.empty());
    CHECK(std::string(stale_result.observation.reason) == "stale");

    const auto no_memory = server_kv_pressure_execute_unified_action(
            enabled_config(), {}, pressure(), false, 1, 1011);
    CHECK(std::string(no_memory.observation.reason) == "no_memory");
}

static void test_evaluate_blocks_invalid_open_or_unsupported() {
    for (int scenario = 0; scenario < 4; ++scenario) {
        fake_core core;
        auto e = evaluation(1020 + scenario);
        if (scenario == 0) e.capability.context_invalid = true;
        if (scenario == 1) e.capability.write_transaction_open = true;
        if (scenario == 2) e.fail_stop = true;
        if (scenario == 3) e.capability.can_release = false;
        core.responses = { e };
        const auto result = server_kv_pressure_execute_unified_action(
                enabled_config(), core.ops(), pressure(), false, 1, 1020 + scenario);
        CHECK(core.requests.size() == 1);
        CHECK(core.requests[0].action == llama_kv_action::evaluate);
        CHECK(!result.observation.release_attempted);
    }
}


static void test_release_unsupported_keeps_governor_debt_and_marker_stable() {
    auto config = enabled_config();
    server_kv_governor_state state;
    fake_core core;
    auto unsupported = evaluation(1030);
    unsupported.capability.can_release = false;
    core.responses = { unsupported };
    server_kv_pressure_snapshot snapshot;
    snapshot.state = kv_pressure_state::PRESSURE;
    snapshot.sample_valid = true;
    snapshot.pressure_basis_valid = true;
    snapshot.pressure_current_bytes = 12288;
    snapshot.pressure_low_water_bytes = 4096;
    snapshot.pressure_basis_generation = 1;
    snapshot.sample_count = 1;
    snapshot.decision_id = 1030;

    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), snapshot, {});
    CHECK(result.debt_before_bytes == 8192);
    CHECK(result.debt_after_bytes == result.debt_before_bytes);
    CHECK(!result.offload_armed_before && !result.offload_armed_after);
    CHECK(!result.observation.release_attempted && !result.observation.offload_attempted);
    const auto marker = server_kv_pressure_unified_action_format_marker(result);
    CHECK(marker.find("debt_before_bytes=8192") != std::string::npos);
    CHECK(marker.find("debt_after_bytes=8192") != std::string::npos);
    CHECK(marker.find("offload_armed_before=0") != std::string::npos);
    CHECK(marker.find("offload_armed_after=0") != std::string::npos);
}

static void test_evaluate_decision_mismatch_never_releases() {
    fake_core core;
    core.responses = { evaluation(1029) };
    const auto result = server_kv_pressure_execute_unified_action(
            enabled_config(), core.ops(), pressure(), false, 1, 1030);
    CHECK(core.requests.size() == 1);
    CHECK(core.requests[0].action == llama_kv_action::evaluate);
    CHECK(release_request_count(core, 1030) == 0);
    CHECK(!result.observation.release_attempted);
    CHECK(std::string(result.observation.reason) == "decision_mismatch");
}

static void test_release_reuses_decision_and_stops_after_one_action() {
    fake_core core;
    core.responses = { evaluation(1030), release(1030) };
    const auto result = server_kv_pressure_execute_unified_action(
            enabled_config(), core.ops(), pressure(kv_pressure_state::CRITICAL), false, 7, 1030);
    CHECK(core.requests.size() == 2);
    CHECK(core.requests[0].action == llama_kv_action::evaluate);
    CHECK(core.requests[1].action == llama_kv_action::release);
    CHECK(core.requests[0].decision_id == 1030 && core.requests[1].decision_id == 1030);
    CHECK(release_request_count(core, 1030) == 1);
    CHECK(core.requests[1].target_bytes == 4096 && core.requests[1].max_blocks == 3);
    CHECK(result.release.core_transaction_id == 99 && result.release.state_changed);
}

static void test_release_terminal_results_do_not_chain() {
    const llama_kv_action_outcome outcomes[] = {
        llama_kv_action_outcome::no_op,
        llama_kv_action_outcome::completed,
        llama_kv_action_outcome::partial_failure,
        llama_kv_action_outcome::failed,
    };
    for (const auto outcome : outcomes) {
        fake_core core;
        auto r = release(1040);
        r.outcome = outcome;
        if (outcome == llama_kv_action_outcome::partial_failure) r.shortfall_bytes = 1;
        if (outcome == llama_kv_action_outcome::failed) r.io_failure = true;
        core.responses = { evaluation(1040), r };
        const auto result = server_kv_pressure_execute_unified_action(
                enabled_config(), core.ops(), pressure(), false, 1, 1040);
        CHECK(core.requests.size() == 2);
        CHECK(release_request_count(core, 1040) == 1);
        CHECK(result.observation.release_attempted);
        CHECK(result.release.outcome == outcome);
    }
}

static void test_marker_keeps_observation_and_core_result_separate() {
    fake_core core;
    core.responses = { evaluation(1050), release(1050) };
    auto result = server_kv_pressure_execute_unified_action(
            enabled_config(), core.ops(), pressure(), true, 9, 1050);
    result.runtime_claimants.push_back({
            0, 1, false, false,
            { true, 3, 2, 1, 0, 0 },
    });
    const auto marker = server_kv_pressure_unified_action_format_marker(result);
    CHECK(marker.find("decision_id=1050") != std::string::npos);
    CHECK(marker.find("transaction_id=99") != std::string::npos);
    CHECK(marker.find("state_changed=1") != std::string::npos);
    CHECK(marker.find("claimants=0:1:0:0:1:3:2:1:0:0") != std::string::npos);
    CHECK(marker.find("rss_") == std::string::npos);

    const llama_kv_action_reason reasons[] = {
        llama_kv_action_reason::zero_budget,
        llama_kv_action_reason::no_candidate,
        llama_kv_action_reason::scan_budget_exhausted,
        llama_kv_action_reason::target_satisfied,
        llama_kv_action_reason::target_shortfall,
        llama_kv_action_reason::unsupported,
        llama_kv_action_reason::blocked,
        llama_kv_action_reason::failed,
    };
    const char * names[] = {
        "zero_budget", "no_candidate", "scan_budget_exhausted", "target_satisfied",
        "target_shortfall", "unsupported", "blocked", "failed",
    };
    for (size_t i = 0; i < sizeof(reasons) / sizeof(reasons[0]); ++i) {
        auto result_with_reason = result;
        result_with_reason.release.reason = reasons[i];
        const auto reason_marker = server_kv_pressure_unified_action_format_marker(result_with_reason);
        CHECK(reason_marker.find(std::string("reason=") + names[i]) != std::string::npos);
    }
}

static server_kv_pressure_snapshot governor_pressure(
        uint64_t sample_count,
        uint64_t decision_id,
        kv_pressure_state state = kv_pressure_state::PRESSURE,
        uint64_t basis_generation = 1,
        bool stale = false) {
    server_kv_pressure_snapshot snapshot;
    snapshot.state = state;
    snapshot.sample_valid = true;
    snapshot.stale = stale;
    snapshot.pressure_basis_valid = true;
    snapshot.pressure_current_bytes = 12288;
    snapshot.pressure_low_water_bytes = 4096;
    snapshot.pressure_basis_generation = basis_generation;
    snapshot.sample_count = sample_count;
    snapshot.decision_id = decision_id;
    return snapshot;
}

static llama_kv_action_result no_candidate_release(uint64_t decision_id) {
    llama_kv_action_result result;
    result.action = llama_kv_action::release;
    result.decision_id = decision_id;
    result.outcome = llama_kv_action_outcome::no_op;
    result.reason = llama_kv_action_reason::no_candidate;
    result.shortfall_bytes = 4096;
    return result;
}

static llama_kv_action_result no_candidate_offload(uint64_t decision_id) {
    llama_kv_action_result result;
    result.action = llama_kv_action::offload;
    result.decision_id = decision_id;
    result.outcome = llama_kv_action_outcome::no_op;
    result.reason = llama_kv_action_reason::no_candidate;
    result.shortfall_bytes = 4096;
    return result;
}

static llama_kv_action_result protected_sequence_offload(uint64_t decision_id) {
    llama_kv_action_result result;
    result.action = llama_kv_action::offload;
    result.decision_id = decision_id;
    result.outcome = llama_kv_action_outcome::rejected;
    result.reason = llama_kv_action_reason::protected_sequence;
    result.shortfall_bytes = 4096;
    return result;
}

static llama_kv_action_result shared_block_offload(uint64_t decision_id) {
    llama_kv_action_result result;
    result.action = llama_kv_action::offload;
    result.decision_id = decision_id;
    result.outcome = llama_kv_action_outcome::rejected;
    result.reason = llama_kv_action_reason::shared_block;
    result.shortfall_bytes = 4096;
    return result;
}

static llama_kv_action_result offload_result(
        uint64_t decision_id, uint64_t bytes, bool io_failure = false) {
    llama_kv_action_result result;
    result.action = llama_kv_action::offload;
    result.decision_id = decision_id;
    result.outcome = io_failure
        ? llama_kv_action_outcome::failed
        : llama_kv_action_outcome::completed;
    result.reason = io_failure
        ? llama_kv_action_reason::io_failure
        : llama_kv_action_reason::target_satisfied;
    result.io_failure = io_failure;
    result.io_errno = io_failure ? ENOSPC : 0;
    result.state_changed = !io_failure;
    result.core_transaction_id = io_failure ? 0 : decision_id + 100;
    result.blocks = io_failure ? 0 : 1;
    result.bytes = io_failure ? 0 : bytes;
    result.relieved_bytes = io_failure ? 0 : bytes;
    result.shortfall_bytes = io_failure ? 4096 : 0;
    return result;
}

static server_kv_claimant_snapshot claimant(
        llama_seq_id seq_id,
        bool active = false,
        bool protected_sequence = false,
        bool shared = false) {
    server_kv_claimant_snapshot snapshot;
    snapshot.seq_id = seq_id;
    snapshot.active = active;
    snapshot.protected_sequence = protected_sequence;
    snapshot.shared = shared;
    snapshot.idle_age_us = 100000;
    snapshot.logical_kv_tokens = 64;
    snapshot.reclaimable_bytes = 8192;
    snapshot.lcp_n_past_hint_tokens = 8;
    snapshot.io_cost_bytes = 8192;
    return snapshot;
}

static server_kv_claimant_snapshot active_producer() {
    return claimant(999, true);
}

static server_kv_pressure_unified_action_config v3_config() {
    auto config = enabled_config();
    config.policy = server_kv_pressure_policy::v3;
    config.resident_lease_samples = 2;
    config.churn_penalty_us = 1000;
    return config;
}

static server_kv_claimant_snapshot v3_claimant(
        llama_seq_id seq_id,
        uint64_t physical_bytes,
        uint64_t write_us,
        uint64_t restore_us,
        uint64_t lcp_tokens = 64) {
    auto snapshot = claimant(seq_id);
    snapshot.reclaimable_bytes = 0;
    snapshot.io_cost_bytes = 0;
    snapshot.lcp_n_past_hint_tokens = lcp_tokens;
    snapshot.physical_object_id = 11;
    snapshot.physical_generation = 3;
    snapshot.physical_estimate_available = true;
    snapshot.physical_estimate_authoritative = true;
    snapshot.estimated_exclusive_resident_bytes = physical_bytes;
    snapshot.exclusive_resident_blocks = 1;
    snapshot.history.epoch = snapshot.epoch;
    snapshot.history.object_id = 11;
    snapshot.history.generation = 3;
    snapshot.history.last_offload_time_us = write_us;
    snapshot.history.last_restore_gate_us = restore_us;
    snapshot.history.last_restore_bytes = restore_us > 0 ? physical_bytes : 0;
    return snapshot;
}

static void arm_governor_offload(
        server_kv_governor_state & state,
        const server_kv_pressure_unified_action_config & config,
        uint64_t sample_count,
        uint64_t decision_id) {
    fake_core core;
    core.responses = { evaluation(decision_id), no_candidate_release(decision_id) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(),
            governor_pressure(sample_count, decision_id), {});
    CHECK(core.requests.size() == 2);
    CHECK(core.requests[0].action == llama_kv_action::evaluate);
    CHECK(core.requests[1].action == llama_kv_action::release);
    CHECK(result.offload_armed_after && state.offload_armed());
    CHECK(!result.observation.offload_attempted);
}

static void arm_soft_budget_offload(
        server_kv_governor_state & state,
        const server_kv_pressure_unified_action_config & config,
        uint64_t sample_count,
        uint64_t decision_id) {
    fake_core core;
    core.responses = { evaluation(decision_id), no_candidate_release(decision_id) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(sample_count, decision_id,
                kv_pressure_state::NORMAL), {}, budget_view(12288));
    CHECK(core.requests.size() == 2);
    CHECK(core.requests[0].action == llama_kv_action::evaluate);
    CHECK(core.requests[1].action == llama_kv_action::release);
    CHECK(result.soft_offload_armed_after && state.soft_offload_armed());
    CHECK(result.budget_debt_after_bytes == 8192);
}

static void test_governor_scores_idle_claimant_without_active_producer() {
    const auto config = enabled_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 1991);

    auto offload = release(1992);
    offload.action = llama_kv_action::offload;
    fake_core core;
    core.responses = { evaluation(1992), offload };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 1992), { claimant(2) });
    CHECK(core.requests.size() == 2);
    CHECK(core.requests[0].action == llama_kv_action::evaluate);
    CHECK(core.requests[1].action == llama_kv_action::offload);
    CHECK(result.observation.offload_attempted && result.selected_seq_id == 2);
    CHECK(std::string(result.observation.reason) == "offload_submitted");
    CHECK(result.scores.size() == 1 && result.scores[0].eligible);
}

static void test_v3_cost_aware_ranking_uses_physical_relief_and_feedback() {
    const auto config = v3_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 1981);

    fake_core core;
    core.responses = { evaluation(1982), offload_result(1982, 4096) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 1982),
            { v3_claimant(1, 1000, 100, 100), v3_claimant(2, 3000, 100, 50) });
    CHECK(result.selected_seq_id == 2);
    CHECK(result.scores.size() == 2);
    CHECK(result.scores[0].seq_id == 2 && result.scores[1].seq_id == 1);
    CHECK(result.scores[0].cost_aware && result.scores[1].cost_aware);
    CHECK(result.scores[0].estimated_physical_bytes == 3000);
    CHECK(result.scores[0].expected_cost_us < result.scores[1].expected_cost_us);
    CHECK(result.scores[0].fallback_reason &&
            std::string(result.scores[0].fallback_reason) == "none");
}

static void test_v3_missing_physical_estimate_falls_back_to_idle_age() {
    const auto config = v3_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 1985);

    auto older = claimant(1);
    auto newer = claimant(2);
    older.reclaimable_bytes = 0;
    newer.reclaimable_bytes = 0;
    older.idle_age_us = 300000;
    newer.idle_age_us = 100000;
    fake_core core;
    core.responses = { evaluation(1986), offload_result(1986, 4096) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 1986), { newer, older });
    CHECK(result.selected_seq_id == 1);
    CHECK(result.scores[0].fallback_reason &&
            std::string(result.scores[0].fallback_reason) == "physical_unavailable");
    CHECK(!result.scores[0].cost_aware);
}

static void test_v3_stale_physical_generation_is_excluded() {
    const auto config = v3_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 1987);

    auto pressure_snapshot = governor_pressure(2, 1988);
    pressure_snapshot.kv_physical_view_available = true;
    pressure_snapshot.kv_object_id = 11;
    pressure_snapshot.kv_generation = 3;
    auto stale = v3_claimant(1, 1000, 100, 100);
    stale.physical_generation = 2;
    auto current = v3_claimant(2, 1000, 100, 100);
    fake_core core;
    core.responses = { evaluation(1988), offload_result(1988, 4096) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), pressure_snapshot, { stale, current });
    CHECK(result.selected_seq_id == 2);
    const auto stale_score = std::find_if(
            result.scores.begin(), result.scores.end(),
            [](const server_kv_claimant_score & score) { return score.seq_id == 1; });
    CHECK(stale_score != result.scores.end());
    CHECK(stale_score->exclusion == server_kv_claimant_exclusion::stale_generation);
}

static void test_v3_restore_lease_blocks_immediate_offload() {
    const auto config = v3_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 1989);

    llama_kv_action_result restore;
    restore.action = llama_kv_action::prefetch;
    restore.state_changed = true;
    restore.outcome = llama_kv_action_outcome::completed;
    restore.blocks = 1;
    restore.bytes = 1024;
    state.record_resume_feedback(1, restore, 50, 2, 4);

    auto leased = v3_claimant(1, 1000, 100, 100);
    leased.epoch = state.claimant_epoch(1);
    leased.history = state.claimant_history(1, leased.epoch);
    auto alternative = v3_claimant(2, 1000, 100, 100);
    fake_core core;
    core.responses = { evaluation(1990), offload_result(1990, 4096) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(3, 1990),
            { leased, alternative });
    CHECK(result.selected_seq_id == 2);
    const auto leased_score = std::find_if(
            result.scores.begin(), result.scores.end(),
            [](const server_kv_claimant_score & score) { return score.seq_id == 1; });
    CHECK(leased_score != result.scores.end());
    CHECK(leased_score->exclusion == server_kv_claimant_exclusion::resident_lease);
}

static void test_v3_stale_feedback_is_discarded_after_lineage_loss() {
    const auto config = v3_config();
    server_kv_governor_state state;
    auto snapshot = v3_claimant(1, 1000, 100, 100);
    auto action = offload_result(1991, 1000);
    action.action_elapsed_us = 17;
    action.physical_object_id = 11;
    action.physical_generation = 3;
    state.record_offload_feedback(snapshot, action, 1);
    CHECK(state.claimant_history(1, 1).last_offload_time_us == 17);
    // Real prompt / object / generation loss hard-clears the lineage: cost
    // evidence describing the dead KV object must not survive.
    state.clear_claimant_lineage(1);
    CHECK(state.claimant_history(1, state.claimant_epoch(1)).epoch == 0);
    // Stale feedback bound to the old epoch is rejected under the new epoch.
    state.record_offload_feedback(snapshot, action, 2);
    CHECK(state.claimant_history(1, state.claimant_epoch(1)).epoch == 0);
    (void) config;
}

// Ordinary turn rollover (reset) advances the epoch but migrates the cost
// history so offload cost, actual relief, restore gate and round_trip persist
// across turns and feed the next ranking — as long as the KV lineage is intact.
static void test_v3_history_survives_ordinary_epoch_rollover() {
    const auto config = v3_config();
    server_kv_governor_state state;
    const uint64_t epoch0 = state.claimant_epoch(1);
    auto snapshot = v3_claimant(1, 1000, 100, 100);
    snapshot.epoch = epoch0;
    auto action = offload_result(1991, 1000);
    action.action_elapsed_us = 17;
    action.physical_object_id = 11;
    action.physical_generation = 3;
    state.record_offload_feedback(snapshot, action, 1);
    CHECK(state.claimant_history(1, epoch0).last_offload_time_us == 17);
    CHECK(state.claimant_history(1, epoch0).last_offload_bytes == 1000);

    // Turn boundary: ordinary rollover (reset) — lineage unchanged.
    state.invalidate_claimant(1);
    const uint64_t epoch1 = state.claimant_epoch(1);
    CHECK(epoch1 == epoch0 + 1);
    // History migrated to the new epoch; cost evidence survives.
    const auto migrated = state.claimant_history(1, epoch1);
    CHECK(migrated.epoch == epoch1);
    CHECK(migrated.last_offload_time_us == 17);
    CHECK(migrated.last_offload_bytes == 1000);

    // Record an offload restore (round trip) under the new epoch.
    llama_kv_action_result restore;
    restore.action = llama_kv_action::prefetch;
    restore.state_changed = true;
    restore.outcome = llama_kv_action_outcome::completed;
    restore.blocks = 1;
    restore.bytes = 1000;
    restore.physical_object_id = 11;
    restore.physical_generation = 3;
    state.record_restore_feedback(1, epoch1, restore, 50, 2, 4);
    const auto after_restore = state.claimant_history(1, epoch1);
    CHECK(after_restore.last_restore_gate_us == 50);
    CHECK(after_restore.round_trip_count == 1);
    CHECK(after_restore.resident_lease_until_sample == 6);

    // Next turn's rollover again preserves the accumulated churn evidence.
    state.invalidate_claimant(1);
    const uint64_t epoch2 = state.claimant_epoch(1);
    const auto next_turn = state.claimant_history(1, epoch2);
    CHECK(next_turn.epoch == epoch2);
    CHECK(next_turn.last_offload_time_us == 17);
    CHECK(next_turn.last_restore_gate_us == 50);
    CHECK(next_turn.round_trip_count == 1);
    (void) config;
}

// Mixed-evidence decision: when some eligible claimants are cost-aware and
// others fall back, the whole decision uses a single idle-age / LRU authority
// — a claimant with history must NOT be permanently preferred just because
// it carries cost evidence.
static void test_v3_mixed_evidence_decision_falls_back_to_idle_age() {
    const auto config = v3_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 2001);

    // historical claimant: full cost evidence, would rank first under cost-aware.
    auto historical = v3_claimant(1, 1000, 100, 100, 64);
    const uint64_t epoch = state.claimant_epoch(1);
    historical.epoch = epoch;
    historical.history.epoch = epoch;
    historical.history.object_id = 11;
    historical.history.generation = 3;
    historical.history.last_offload_time_us = 100;
    historical.history.last_restore_gate_us = 100;
    historical.history.last_restore_bytes = 1000;
    historical.idle_age_us = 100000;

    // fallback claimant: physical estimate missing, but far colder (older).
    auto cold = claimant(2);
    cold.reclaimable_bytes = 0;
    cold.physical_estimate_available = false;
    cold.idle_age_us = 900000;
    cold.epoch = state.claimant_epoch(2);

    fake_core core;
    core.responses = { evaluation(2002), offload_result(2002, 4096) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 2002),
            { historical, cold });
    // Decision fell back to a uniform idle-age / LRU authority.
    CHECK(result.decision_fallback_reason &&
            std::string(result.decision_fallback_reason) == "physical_unavailable");
    // The colder (fallback) claimant wins under the single authority — history
    // does not permanently pin the historical claimant as the victim.
    CHECK(result.selected_seq_id == 2);
    const auto historical_score = std::find_if(
            result.scores.begin(), result.scores.end(),
            [](const server_kv_claimant_score & s) { return s.seq_id == 1; });
    CHECK(historical_score != result.scores.end());
    // historical claimant is itself cost-aware, but the mixed decision still
    // falls back to idle-age authority and ranks by idle_age_score.
    CHECK(historical_score->cost_aware);
    const auto cold_score = std::find_if(
            result.scores.begin(), result.scores.end(),
            [](const server_kv_claimant_score & s) { return s.seq_id == 2; });
    CHECK(cold_score != result.scores.end());
    CHECK(!cold_score->cost_aware);
    CHECK(std::string(cold_score->fallback_reason) == "physical_unavailable");
    CHECK(result.scores[0].seq_id == 2);
}

// V3 cost-aware ranking must never override the legacy safety exclusions.
// An active / protected / shared claimant carries a large physical estimate
// (cheap to offload) yet must still be excluded — safety precedes ranking.
static void test_v3_safety_exclusions_precede_cost_ranking() {
    const auto config = v3_config();
    struct case_t {
        server_kv_claimant_snapshot claimant;
        server_kv_claimant_exclusion expected;
        const char * name;
    };
    auto active = v3_claimant(1, 9999, 10, 0, 64);
    active.active = true;
    auto protected_seq = v3_claimant(2, 9999, 10, 0, 64);
    protected_seq.protected_sequence = true;
    auto shared = v3_claimant(3, 9999, 10, 0, 64);
    shared.shared = true;
    auto safe = v3_claimant(4, 1000, 100, 100, 64);
    const case_t cases[] = {
        { active,        server_kv_claimant_exclusion::active,            "active" },
        { protected_seq, server_kv_claimant_exclusion::protected_sequence, "protected" },
        { shared,        server_kv_claimant_exclusion::shared,            "shared" },
    };
    for (const auto & c : cases) {
        server_kv_governor_state state;
        arm_governor_offload(state, config, 1, 1992);
        fake_core core;
        core.responses = { evaluation(1993), offload_result(1993, 4096) };
        const auto result = server_kv_pressure_execute_governor(
                config, state, core.ops(), governor_pressure(2, 1993), { c.claimant, safe });
        const auto victim = std::find_if(
                result.scores.begin(), result.scores.end(),
                [&c](const server_kv_claimant_score & s) { return s.seq_id == c.claimant.seq_id; });
        CHECK(victim != result.scores.end());
        CHECK(victim->exclusion == c.expected);
        CHECK(!victim->eligible);
        CHECK(!victim->cost_aware);
        CHECK(result.selected_seq_id == 4);
    }
}

// Legacy V2 is the default policy when LLAMA_KV_PRESSURE_POLICY is unset and
// must preserve the existing score (legacy bytes) and descending sort.
static void test_v3_policy_defaults_to_v2_when_env_unset() {
    clear_pressure_action_env();
    setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION", "1", 1);
    setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", "4096", 1);
    setenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS", "3", 1);
    const auto decision = server_kv_pressure_unified_action_startup_decide_from_env();
    CHECK(decision.status == server_kv_pressure_unified_action_startup_status::enabled);
    CHECK(decision.config.policy == server_kv_pressure_policy::v2);
    clear_pressure_action_env();

    // V2 selection uses logical reclaimable bytes (descending), not physical.
    // Hold idle_age constant so the reclaimable_score term alone decides order.
    const auto config = enabled_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 1994);
    auto large = claimant(1);
    large.reclaimable_bytes = 16384;
    large.idle_age_us = 100000;
    auto small = claimant(2);
    small.reclaimable_bytes = 4096;
    small.idle_age_us = 100000;
    fake_core core;
    core.responses = { evaluation(1995), offload_result(1995, 4096) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 1995), { small, large });
    CHECK(result.policy == server_kv_pressure_policy::v2);
    CHECK(result.selected_seq_id == 1);
    CHECK(result.scores.size() == 2);
    CHECK(!result.scores[0].cost_aware);
    CHECK(result.scores[0].estimated_physical_bytes == 0);
}

// The Exact request-resume gate is policy-independent: correctness_required
// restore calls execute_action(prefetch) directly and never consults the
// V3 lease.  A leased claimant still restores through the resume gate.
static void test_v3_resume_gate_path_is_policy_independent() {
    const auto config = v3_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 1996);

    // Establish a resident lease on seq 1 so the ordinary OFFLOAD path blocks.
    llama_kv_action_result restore;
    restore.action = llama_kv_action::prefetch;
    restore.state_changed = true;
    restore.outcome = llama_kv_action_outcome::completed;
    restore.decision_id = 1997;
    restore.blocks = 1;
    restore.bytes = 1024;
    restore.physical_object_id = 11;
    restore.physical_generation = 3;
    state.record_resume_feedback(1, restore, 50, 2, 4);

    fake_core resume_core;
    resume_core.responses = { restore };
    llama_seq_id protected_seq = -1;
    bool protected_enabled = false;
    server_kv_resume_ops resume_ops;
    resume_ops.set_protected = [&](llama_seq_id seq_id, bool enabled) {
        protected_seq = seq_id;
        protected_enabled = enabled;
    };
    resume_ops.execute = [&](const llama_kv_action_request & request) {
        return resume_core.ops().execute(request);
    };
    const auto resume_result = server_kv_resume_gate(
            resume_ops,
            server_kv_resume_trigger::active_access,
            1, 1997);
    // The resume gate executed a correctness_required prefetch and succeeded,
    // unaffected by the resident lease that blocks ordinary OFFLOAD.
    CHECK(protected_seq == 1 && protected_enabled);
    CHECK(resume_core.requests.size() == 1);
    CHECK(resume_core.requests[0].action == llama_kv_action::prefetch);
    CHECK(resume_core.requests[0].correctness_required);
    CHECK(resume_core.requests[0].all_required);
    CHECK(resume_core.requests[0].seq_id == 1);
    CHECK(resume_result.action.outcome == llama_kv_action_outcome::completed);
    CHECK(resume_result.graph_allowed);
    CHECK(resume_result.action.bytes == 1024);
}

static void test_governor_debt_release_then_later_offload() {
    auto config = enabled_config();
    server_kv_governor_state state;

    fake_core first;
    auto partial_release = release(2001);
    partial_release.bytes = 4096;
    partial_release.shortfall_bytes = 0;
    first.responses = { evaluation(2001), partial_release };
    const auto first_result = server_kv_pressure_execute_governor(
            config, state, first.ops(), governor_pressure(1, 2001), {});
    CHECK(first_result.debt_before_bytes == 8192);
    CHECK(first_result.debt_after_bytes == 4096);
    CHECK(!first_result.offload_armed_after);
    CHECK(first.requests.size() == 2 && first.requests[1].action == llama_kv_action::release);

    fake_core second;
    second.responses = { evaluation(2002), no_candidate_release(2002) };
    const auto second_result = server_kv_pressure_execute_governor(
            config, state, second.ops(), governor_pressure(2, 2002), {});
    CHECK(second_result.debt_before_bytes == 8192);
    CHECK(second_result.offload_armed_after);
    CHECK(second.requests.size() == 2 && second.requests[1].action == llama_kv_action::release);
    CHECK(!second_result.observation.offload_attempted);

    fake_core third;
    third.responses = { evaluation(2003), offload_result(2003, 4096) };
    const std::vector<server_kv_claimant_snapshot> claimants = {
        claimant(9, true), claimant(7, false, true), claimant(6, false, false, true),
        claimant(5), claimant(2),
    };
    const auto third_result = server_kv_pressure_execute_governor(
            config, state, third.ops(), governor_pressure(3, 2003), claimants);
    CHECK(third.requests.size() == 2);
    CHECK(third.requests[0].action == llama_kv_action::evaluate);
    CHECK(third.requests[1].action == llama_kv_action::offload);
    CHECK(third.requests[1].decision_id == 2003 && third.requests[1].seq_id == 2);
    CHECK(third.requests[1].target_bytes == 4096 && third.requests[1].max_blocks == 3);
    CHECK(third.requests[1].claimant == llama_kv_memory_claimant::kv);
    CHECK(third.requests[1].io_class == llama_kv_io_class::capacity_write);
    CHECK(third.requests[1].io_byte_budget == 4096);
    CHECK(third_result.selected_seq_id == 2);
    CHECK(third_result.scores.size() == claimants.size());
    CHECK(third_result.scores[0].seq_id == 2 && third_result.scores[1].seq_id == 5);
    CHECK(third_result.scores[0].idle_age_score > 0 &&
            third_result.scores[0].logical_kv_score > 0 &&
            third_result.scores[0].reclaimable_score > 0 &&
            third_result.scores[0].lcp_n_past_penalty > 0 &&
            third_result.scores[0].io_cost_penalty > 0);
    CHECK(third_result.scores[2].exclusion == server_kv_claimant_exclusion::shared);
    CHECK(third_result.scores[3].exclusion == server_kv_claimant_exclusion::protected_sequence);
    CHECK(third_result.scores[4].exclusion == server_kv_claimant_exclusion::active);
    const auto marker = server_kv_pressure_unified_action_format_marker(third_result);
    CHECK(marker.find("selected_seq_id=2") != std::string::npos);
    CHECK(marker.find("relieved_bytes=4096") != std::string::npos);
    CHECK(marker.find("scores=2:1:none:") != std::string::npos);

    server_kv_governor_state no_relief_state;
    arm_governor_offload(no_relief_state, config, 1, 2011);
    fake_core no_relief_core;
    auto logical_only = offload_result(2012, 4096);
    logical_only.relieved_bytes = 0;
    no_relief_core.responses = { evaluation(2012), logical_only };
    const auto no_relief = server_kv_pressure_execute_governor(
            config, no_relief_state, no_relief_core.ops(),
            governor_pressure(2, 2012), { claimant(2), active_producer() });
    CHECK(no_relief.offload.bytes == 4096 && no_relief.offload.relieved_bytes == 0);
    CHECK(no_relief.debt_after_bytes == no_relief.debt_before_bytes);
    CHECK(no_relief.offload_armed_after);

    fake_core exhausted_core;
    exhausted_core.responses = { evaluation(2013), no_candidate_offload(2013) };
    const auto exhausted = server_kv_pressure_execute_governor(
            config, no_relief_state, exhausted_core.ops(),
            governor_pressure(3, 2013), { claimant(2), claimant(5), active_producer() });
    CHECK(exhausted.selected_seq_id == 2);
    CHECK(exhausted.offload.reason == llama_kv_action_reason::no_candidate);

    fake_core advance_core;
    advance_core.responses = { evaluation(2014), offload_result(2014, 4096) };
    const auto advance = server_kv_pressure_execute_governor(
            config, no_relief_state, advance_core.ops(),
            governor_pressure(4, 2014), { claimant(2), claimant(5), active_producer() });
    CHECK(advance.selected_seq_id == 5);
    CHECK(advance.scores[0].seq_id == 5);
    CHECK(advance.scores[1].seq_id == 2 &&
            advance.scores[1].exclusion == server_kv_claimant_exclusion::exhausted);
}

static void test_governor_idle_follow_up_completes_release_windows_and_arms_offload() {
    const auto config = enabled_config();
    server_kv_governor_state state;

    const llama_kv_action_reason release_reasons[] = {
        llama_kv_action_reason::target_satisfied,
        llama_kv_action_reason::target_satisfied,
        llama_kv_action_reason::scan_budget_exhausted,
    };
    for (size_t index = 0; index < sizeof(release_reasons) / sizeof(release_reasons[0]); ++index) {
        const uint64_t sample_count = index + 1;
        const uint64_t decision_id = 2051 + index;
        auto window = release(decision_id);
        window.reason = release_reasons[index];
        if (window.reason == llama_kv_action_reason::scan_budget_exhausted) {
            window.outcome = llama_kv_action_outcome::no_op;
            window.state_changed = false;
            window.core_transaction_id = 0;
            window.blocks = 0;
            window.bytes = 0;
            window.relieved_bytes = 0;
            window.shortfall_bytes = 4096;
        }
        fake_core core;
        core.responses = { evaluation(decision_id), window };
        const auto result = server_kv_pressure_execute_governor(
                config, state, core.ops(), governor_pressure(sample_count, decision_id), {});
        CHECK(core.requests.size() == 2);
        CHECK(core.requests[0].action == llama_kv_action::evaluate);
        CHECK(core.requests[1].action == llama_kv_action::release);
        CHECK(core.requests[1].decision_id == decision_id);
        CHECK(result.observation.release_attempted && !result.observation.offload_attempted);
        if (window.reason == llama_kv_action_reason::scan_budget_exhausted) {
            CHECK(result.release.outcome == llama_kv_action_outcome::no_op);
            CHECK(!result.release.state_changed && result.release.relieved_bytes == 0);
        }
        CHECK(state.idle_follow_up_pending());
        CHECK(!state.offload_armed());
    }

    fake_core arm_core;
    arm_core.responses = { evaluation(2054), no_candidate_release(2054) };
    const auto armed = server_kv_pressure_execute_governor(
            config, state, arm_core.ops(), governor_pressure(4, 2054), {});
    CHECK(arm_core.requests.size() == 2);
    CHECK(arm_core.requests[1].action == llama_kv_action::release);
    CHECK(armed.offload_armed_after && state.offload_armed());
    CHECK(!armed.observation.offload_attempted);
    CHECK(state.idle_follow_up_pending());

    fake_core stable_core;
    stable_core.responses = { evaluation(2055) };
    const auto stable = server_kv_pressure_execute_governor(
            config, state, stable_core.ops(), governor_pressure(5, 2055), {});
    CHECK(stable_core.requests.size() == 1);
    CHECK(std::string(stable.observation.reason) == "claimant_no_candidate");
    CHECK(state.offload_armed());
    CHECK(!state.idle_follow_up_pending());
}

static void test_governor_sort_is_input_order_independent() {
    auto config = enabled_config();
    const std::vector<server_kv_claimant_snapshot> forward = {
        claimant(8), claimant(3), claimant(5), active_producer(),
    };
    const std::vector<server_kv_claimant_snapshot> reverse = {
        claimant(5), claimant(3), claimant(8), active_producer(),
    };

    for (const auto * order : { &forward, &reverse }) {
        server_kv_governor_state state;
        arm_governor_offload(state, config, 1, 2101);
        fake_core core;
        core.responses = { evaluation(2102), offload_result(2102, 4096) };
        const auto result = server_kv_pressure_execute_governor(
                config, state, core.ops(), governor_pressure(2, 2102), *order);
        CHECK(result.selected_seq_id == 3);
        CHECK(result.scores.size() == 4);
        CHECK(result.scores[0].seq_id == 3 &&
                result.scores[1].seq_id == 5 && result.scores[2].seq_id == 8 &&
                result.scores[3].seq_id == 999);
    }
}

static void test_governor_stale_basis_normal_and_reset() {
    auto config = enabled_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 2201);
    const uint64_t held_debt = state.pressure_debt_bytes();

    fake_core stale_core;
    const auto stale = server_kv_pressure_execute_governor(
            config, state, stale_core.ops(), governor_pressure(2, 2202,
                kv_pressure_state::PRESSURE, 1, true), {});
    CHECK(stale_core.requests.empty());
    CHECK(stale.debt_after_bytes == held_debt && state.offload_armed());
    CHECK(std::string(stale.observation.reason) == "stale_hold");

    fake_core basis_core;
    basis_core.responses = { evaluation(2203), no_candidate_release(2203) };
    const auto basis = server_kv_pressure_execute_governor(
            config, state, basis_core.ops(), governor_pressure(3, 2203,
                kv_pressure_state::PRESSURE, 2), {});
    CHECK(!basis.offload_armed_before);
    CHECK(basis_core.requests.size() == 2 && basis_core.requests[1].action == llama_kv_action::release);

    fake_core normal_core;
    const auto normal = server_kv_pressure_execute_governor(
            config, state, normal_core.ops(), governor_pressure(4, 2204,
                kv_pressure_state::NORMAL, 2), {});
    CHECK(normal_core.requests.empty());
    CHECK(normal.debt_after_bytes == 0 && !state.offload_armed());

    arm_governor_offload(state, config, 5, 2205);
    state.reset();
    CHECK(state.pressure_debt_bytes() == 0);
    CHECK(state.episode() == 0);
    CHECK(!state.offload_armed() && !state.idle_follow_up_pending());
    CHECK(state.next_action_sample() == 0);
}

static void test_governor_global_exclusions_and_debt_saturation() {
    auto config = enabled_config();
    server_kv_governor_state state;

    fake_core saturated_core;
    auto saturated_pressure = governor_pressure(1, 2251);
    saturated_pressure.pressure_current_bytes = UINT64_MAX;
    saturated_pressure.pressure_low_water_bytes = 0;
    auto one_byte_release = release(2251);
    one_byte_release.bytes = 1;
    one_byte_release.relieved_bytes = 1;
    saturated_core.responses = { evaluation(2251), one_byte_release };
    const auto saturated = server_kv_pressure_execute_governor(
            config, state, saturated_core.ops(), saturated_pressure, {});
    CHECK(saturated.debt_before_bytes == UINT64_MAX);
    CHECK(saturated.debt_after_bytes == UINT64_MAX - 1);

    state.reset();
    arm_governor_offload(state, config, 2, 2252);
    fake_core open_core;
    auto open_evaluation = evaluation(2253);
    open_evaluation.capability.write_transaction_open = true;
    open_core.responses = { open_evaluation };
    const auto open = server_kv_pressure_execute_governor(
            config, state, open_core.ops(), governor_pressure(3, 2253), { claimant(2), active_producer() });
    CHECK(open_core.requests.size() == 1);
    CHECK(!open.observation.offload_attempted);
    CHECK(std::string(open.observation.reason) == "write_transaction_open");

    fake_core fail_stop_core;
    auto fail_stop_evaluation = evaluation(2254);
    fail_stop_evaluation.fail_stop = true;
    fail_stop_core.responses = { fail_stop_evaluation };
    const auto fail_stop = server_kv_pressure_execute_governor(
            config, state, fail_stop_core.ops(), governor_pressure(4, 2254), { claimant(2), active_producer() });
    CHECK(fail_stop_core.requests.size() == 1);
    CHECK(!fail_stop.observation.offload_attempted);
    CHECK(std::string(fail_stop.observation.reason) == "fail_stop");
}

static void test_governor_io_failure_backoff_and_penalty() {
    auto config = enabled_config();
    config.io_failure_backoff_samples = 4;

    server_kv_governor_state release_state;
    auto release_failure = release(2291);
    release_failure.outcome = llama_kv_action_outcome::failed;
    release_failure.reason = llama_kv_action_reason::failed;
    release_failure.io_failure = true;
    release_failure.io_errno = EIO;
    release_failure.state_changed = false;
    release_failure.core_transaction_id = 0;
    release_failure.blocks = 0;
    release_failure.bytes = 0;
    release_failure.relieved_bytes = 0;
    fake_core release_failed_core;
    release_failed_core.responses = { evaluation(2291), release_failure };
    server_kv_pressure_execute_governor(
            config, release_state, release_failed_core.ops(), governor_pressure(1, 2291), {});
    CHECK(release_state.next_action_sample() == 5);
    CHECK(release_state.idle_follow_up_pending());

    fake_core release_held_core;
    const auto release_held = server_kv_pressure_execute_governor(
            config, release_state, release_held_core.ops(), governor_pressure(2, 2292), {});
    CHECK(release_held_core.requests.empty());
    CHECK(std::string(release_held.observation.reason) == "backoff");
    CHECK(release_state.idle_follow_up_pending());

    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 2301);

    fake_core failed_core;
    failed_core.responses = { evaluation(2302), offload_result(2302, 0, true) };
    const auto failed = server_kv_pressure_execute_governor(
            config, state, failed_core.ops(), governor_pressure(2, 2302),
            { claimant(2), claimant(5), active_producer() });
    CHECK(failed.selected_seq_id == 2);
    CHECK(failed.offload.io_failure && failed.offload.io_errno == ENOSPC);
    CHECK(state.next_action_sample() == 6);
    CHECK(state.idle_follow_up_pending());

    fake_core held_core;
    const auto held = server_kv_pressure_execute_governor(
            config, state, held_core.ops(), governor_pressure(3, 2303),
            { claimant(2), claimant(5), active_producer() });
    CHECK(held_core.requests.empty());
    CHECK(std::string(held.observation.reason) == "backoff");
    CHECK(state.idle_follow_up_pending());

    fake_core retry_core;
    retry_core.responses = { evaluation(2306), offload_result(2306, 4096) };
    const auto retry = server_kv_pressure_execute_governor(
            config, state, retry_core.ops(), governor_pressure(6, 2306),
            { claimant(2), claimant(5), active_producer() });
    CHECK(retry.selected_seq_id == 5);
    CHECK(retry.scores[0].seq_id == 5);
    CHECK(retry.scores[1].seq_id == 2 && retry.scores[1].failure_penalty > 0);
    CHECK(state.idle_follow_up_pending());
}

static void test_governor_exhaustion_epoch_and_stable_noop() {
    const auto config = enabled_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 2401);
    const uint64_t old_epoch = state.claimant_epoch(2);

    fake_core exhausted_core;
    exhausted_core.responses = { evaluation(2402), no_candidate_offload(2402) };
    const auto exhausted = server_kv_pressure_execute_governor(
            config, state, exhausted_core.ops(), governor_pressure(2, 2402),
            { claimant(2), claimant(5), active_producer() });
    CHECK(exhausted.selected_seq_id == 2);
    CHECK(exhausted.scores[0].exclusion == server_kv_claimant_exclusion::none);
    CHECK(state.claimant_exhausted(2, old_epoch));

    fake_core advance_core;
    advance_core.responses = { evaluation(2403), offload_result(2403, 4096) };
    const auto advance = server_kv_pressure_execute_governor(
            config, state, advance_core.ops(), governor_pressure(3, 2403),
            { claimant(2), claimant(5), active_producer() });
    CHECK(advance.selected_seq_id == 5);
    CHECK(advance.scores[1].exclusion == server_kv_claimant_exclusion::exhausted);
    CHECK(advance.scores[1].seq_id == 2);

    server_kv_governor_state all_exhausted;
    arm_governor_offload(all_exhausted, config, 1, 2404);
    fake_core first;
    first.responses = { evaluation(2405), no_candidate_offload(2405) };
    server_kv_pressure_execute_governor(
            config, all_exhausted, first.ops(), governor_pressure(2, 2405), { claimant(2), active_producer() });
    fake_core stable;
    stable.responses = { evaluation(2406) };
    const auto noop = server_kv_pressure_execute_governor(
            config, all_exhausted, stable.ops(), governor_pressure(3, 2406), { claimant(2), active_producer() });
    CHECK(stable.requests.size() == 1 && stable.requests[0].action == llama_kv_action::evaluate);
    CHECK(!noop.observation.offload_attempted && noop.selected_seq_id == -1);
    CHECK(std::string(noop.observation.reason) == "claimant_no_candidate");
    CHECK(noop.debt_after_bytes == noop.debt_before_bytes);
    CHECK(all_exhausted.offload_armed());

    state.invalidate_claimant(2);
    CHECK(state.claimant_epoch(2) != old_epoch);
    CHECK(!state.claimant_exhausted(2, state.claimant_epoch(2)));
    fake_core invalidated;
    invalidated.responses = { evaluation(2407), offload_result(2407, 4096) };
    auto reused_claimant = claimant(2);
    reused_claimant.epoch = state.claimant_epoch(2);
    const auto reused = server_kv_pressure_execute_governor(
            config, state, invalidated.ops(), governor_pressure(4, 2407), { reused_claimant, active_producer() });
    CHECK(reused.selected_seq_id == 2);
    CHECK(reused.scores[0].seq_id == 2 && reused.scores[0].eligible);

    const auto marker = server_kv_pressure_unified_action_format_marker(advance);
    CHECK(marker.find("selected_claimant_epoch=") != std::string::npos);
    state.reset();
    CHECK(state.claimant_epoch(2) == 1 && state.pressure_debt_bytes() == 0);
}

static void test_governor_exhaustion_result_pollution_is_ignored() {
    const auto config = enabled_config();
    server_kv_governor_state state;
    arm_governor_offload(state, config, 1, 2411);

    auto mismatched = no_candidate_offload(9999);
    mismatched.state_changed = false;
    fake_core stale;
    stale.responses = { evaluation(2412), mismatched };
    const auto stale_result = server_kv_pressure_execute_governor(
            config, state, stale.ops(), governor_pressure(2, 2412), { claimant(2), claimant(5), active_producer() });
    CHECK(stale_result.selected_seq_id == 2);

    auto failed = no_candidate_offload(2413);
    failed.outcome = llama_kv_action_outcome::failed;
    fake_core failed_core;
    failed_core.responses = { evaluation(2413), failed };
    server_kv_pressure_execute_governor(
            config, state, failed_core.ops(), governor_pressure(3, 2413), { claimant(2), claimant(5), active_producer() });

    auto changed = no_candidate_offload(2414);
    changed.state_changed = true;
    fake_core changed_core;
    changed_core.responses = { evaluation(2414), changed };
    server_kv_pressure_execute_governor(
            config, state, changed_core.ops(), governor_pressure(4, 2414), { claimant(2), claimant(5), active_producer() });
    fake_core retry;
    retry.responses = { evaluation(2415), offload_result(2415, 4096) };
    const auto retried = server_kv_pressure_execute_governor(
            config, state, retry.ops(), governor_pressure(5, 2415), { claimant(2), claimant(5), active_producer() });
    CHECK(retried.selected_seq_id == 2);
}

static void test_v2_target_env_source_and_generation() {
    clear_pressure_action_env();
    std::string error;
    const auto disabled = server_kv_resident_target_from_env(error);
    CHECK(!disabled.valid() && error.empty());

    setenv("LLAMA_KV_RESIDENT_TARGET_BYTES", "8192", 1);
    setenv("LLAMA_KV_RESIDENT_TARGET_SOURCE", "env_static", 1);
    error.clear();
    const auto enabled = server_kv_resident_target_from_env(error);
    CHECK(error.empty() && enabled.valid());
    CHECK(enabled.target_bytes == 8192 && enabled.basis_generation == 1);
    CHECK(std::string(enabled.source) == "env_static");

    setenv("LLAMA_KV_RESIDENT_TARGET_SOURCE", "unknown", 1);
    error.clear();
    const auto invalid = server_kv_resident_target_from_env(error);
    CHECK(!invalid.valid() && !error.empty());
    clear_pressure_action_env();
}

static void test_v2_hard_pressure_wins_over_soft_budget() {
    const auto config = budget_config();
    server_kv_governor_state state;
    fake_core core;
    core.responses = { evaluation(3001), release(3001) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(1, 3001,
                kv_pressure_state::PRESSURE), {}, budget_view(16384));
    CHECK(result.budget_target_enabled && !result.budget_active);
    CHECK(result.debt_before_bytes == 8192);
    CHECK(result.budget_debt_before_bytes == 0);
    CHECK(core.requests.size() == 2);
    CHECK(core.requests[0].action == llama_kv_action::evaluate);
    CHECK(core.requests[1].action == llama_kv_action::release);
    CHECK(!state.soft_offload_armed() && state.budget_debt_bytes() == 0);
    for (const auto & request : core.requests) {
        CHECK(request.action != llama_kv_action::prefetch);
        CHECK(!request.correctness_required);
    }
}

static void test_v2_soft_release_then_later_offload() {
    const auto config = budget_config();
    server_kv_governor_state state;

    fake_core first;
    first.responses = { evaluation(3011), release(3011) };
    const auto first_result = server_kv_pressure_execute_governor(
            config, state, first.ops(), governor_pressure(1, 3011,
                kv_pressure_state::NORMAL), {}, budget_view(12288));
    CHECK(first.requests.size() == 2 && first.requests[1].action == llama_kv_action::release);
    CHECK(first_result.budget_active);
    CHECK(first_result.budget_observed_excess_bytes == 8192);
    CHECK(first_result.budget_debt_after_bytes == 4096);
    CHECK(!first_result.soft_offload_armed_after);

    fake_core second;
    second.responses = { evaluation(3012), no_candidate_release(3012) };
    const auto second_result = server_kv_pressure_execute_governor(
            config, state, second.ops(), governor_pressure(2, 3012,
                kv_pressure_state::NORMAL), {}, budget_view(8192));
    CHECK(second.requests.size() == 2 && second.requests[1].action == llama_kv_action::release);
    CHECK(second_result.soft_offload_armed_after && state.soft_offload_armed());
    CHECK(second_result.budget_debt_after_bytes == 4096);

    fake_core third;
    third.responses = { evaluation(3013), offload_result(3013, 4096) };
    const auto third_result = server_kv_pressure_execute_governor(
            config, state, third.ops(), governor_pressure(3, 3013,
                kv_pressure_state::NORMAL), { claimant(2) }, budget_view(8192));
    CHECK(third.requests.size() == 2);
    CHECK(third.requests[0].action == llama_kv_action::evaluate);
    CHECK(third.requests[1].action == llama_kv_action::offload);
    CHECK(third.requests[1].seq_id == 2 && third.requests[1].target_bytes == 4096);
    CHECK(third_result.budget_debt_after_bytes == 0);
    CHECK(!third_result.soft_offload_armed_after && !state.soft_offload_armed());
    for (const auto & request : third.requests) {
        CHECK(request.action != llama_kv_action::prefetch);
        CHECK(!request.correctness_required);
    }
    const auto marker = server_kv_pressure_unified_action_format_marker(third_result);
    CHECK(marker.find("budget_target_enabled=1") != std::string::npos);
    CHECK(marker.find("budget_source=env_static") != std::string::npos);
    CHECK(marker.find("budget_debt_after_bytes=0") != std::string::npos);
    CHECK(marker.find("budget_transient_staging_bound_bytes=1234") != std::string::npos);
}

static void test_v2_unmet_budget_terminates_correctness_protected_chasing() {
    const auto config = budget_config();
    server_kv_governor_state state;

    fake_core arm;
    arm.responses = { evaluation(3021), no_candidate_release(3021) };
    server_kv_pressure_execute_governor(
            config, state, arm.ops(), governor_pressure(1, 3021,
                kv_pressure_state::NORMAL), {}, budget_view(12288));
    CHECK(state.soft_offload_armed());

    fake_core terminal;
    terminal.responses = { evaluation(3022) };
    const std::vector<server_kv_claimant_snapshot> protected_claimants = {
        claimant(1, true), claimant(2, false, true), claimant(3, false, false, true),
    };
    const auto result = server_kv_pressure_execute_governor(
            config, state, terminal.ops(), governor_pressure(2, 3022,
                kv_pressure_state::NORMAL), protected_claimants, budget_view(12288));
    CHECK(terminal.requests.size() == 1);
    CHECK(!result.observation.offload_attempted);
    CHECK(std::string(result.observation.reason) == "budget_unmet_terminal");
    CHECK(result.unmet_budget_bytes_after == 8192);
    CHECK(state.unmet_budget_bytes() == 8192);
    CHECK(!state.idle_follow_up_pending());
    for (const auto & request : terminal.requests) {
        CHECK(request.action != llama_kv_action::prefetch);
        CHECK(!request.correctness_required);
    }

    fake_core held;
    const auto held_result = server_kv_pressure_execute_governor(
            config, state, held.ops(), governor_pressure(3, 3023,
                kv_pressure_state::NORMAL), protected_claimants, budget_view(12288));
    CHECK(held.requests.empty());
    CHECK(std::string(held_result.observation.reason) == "budget_unmet_hold");
}

static void test_v2_invalid_budget_view_fails_closed() {
    const auto config = budget_config();
    server_kv_governor_state state;
    fake_core core;
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(1, 3031,
                kv_pressure_state::NORMAL), { claimant(1) }, budget_view(16384, false));
    CHECK(core.requests.empty());
    CHECK(!result.budget_active);
    CHECK(std::string(result.observation.reason) == "budget_view_unavailable");
    CHECK(state.budget_debt_bytes() == 0);
}

static void test_v2_invalid_pressure_basis_holds_soft_state() {
    const auto config = budget_config();
    server_kv_governor_state state;
    arm_soft_budget_offload(state, config, 1, 3041);
    const uint64_t debt = state.budget_debt_bytes();
    const uint64_t next_sample = state.budget_next_action_sample();

    auto invalid_basis = governor_pressure(2, 3042, kv_pressure_state::NORMAL);
    invalid_basis.pressure_basis_valid = false;
    fake_core core;
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), invalid_basis, { claimant(2) }, budget_view(12288));
    CHECK(core.requests.empty());
    CHECK(!result.budget_active);
    CHECK(std::string(result.observation.reason) == "invalid_basis_hold");
    CHECK(result.budget_view_valid && result.budget_resident_available);
    CHECK(result.budget_debt_before_bytes == debt);
    CHECK(result.budget_debt_after_bytes == debt);
    CHECK(state.budget_debt_bytes() == debt);
    CHECK(state.soft_offload_armed());
    CHECK(state.budget_next_action_sample() == next_sample);
}

static void test_v2_protected_soft_offload_becomes_unmet_terminal() {
    const auto config = budget_config();
    server_kv_governor_state state;
    arm_soft_budget_offload(state, config, 1, 3051);

    fake_core core;
    core.responses = { evaluation(3052), protected_sequence_offload(3052) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 3052,
                kv_pressure_state::NORMAL), { claimant(2) }, budget_view(12288));
    CHECK(core.requests.size() == 2);
    CHECK(core.requests[1].action == llama_kv_action::offload);
    CHECK(result.offload.reason == llama_kv_action_reason::protected_sequence);
    CHECK(std::string(result.observation.reason) == "budget_unmet_terminal");
    CHECK(result.unmet_budget_bytes_after == 8192);
    CHECK(state.unmet_budget_bytes() == 8192);
    CHECK(!state.soft_offload_armed() && !state.idle_follow_up_pending());
    CHECK(state.budget_next_action_sample() == 2 + config.budget_unmet_backoff_samples);

    fake_core held;
    const auto held_result = server_kv_pressure_execute_governor(
            config, state, held.ops(), governor_pressure(3, 3053,
                kv_pressure_state::NORMAL), { claimant(2) }, budget_view(12288));
    CHECK(held.requests.empty());
    CHECK(std::string(held_result.observation.reason) == "budget_unmet_hold");
}

static void test_v2_shared_soft_offload_becomes_unmet_terminal() {
    const auto config = budget_config();
    server_kv_governor_state state;
    arm_soft_budget_offload(state, config, 1, 3061);

    fake_core core;
    core.responses = { evaluation(3062), shared_block_offload(3062) };
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 3062,
                kv_pressure_state::NORMAL), { claimant(2) }, budget_view(12288));
    CHECK(core.requests.size() == 2);
    CHECK(core.requests[1].action == llama_kv_action::offload);
    CHECK(result.offload.reason == llama_kv_action_reason::shared_block);
    CHECK(std::string(result.observation.reason) == "budget_unmet_terminal");
    CHECK(result.unmet_budget_bytes_after == 8192);
    CHECK(state.unmet_budget_bytes() == 8192);
    CHECK(!state.soft_offload_armed() && !state.idle_follow_up_pending());
    CHECK(state.budget_next_action_sample() == 2 + config.budget_unmet_backoff_samples);
}

static void test_global_kv_budget_adapter_authority_and_physical_credit() {
    server_kv_budget_adapter adapter;
    std::string error;
    unsetenv("LLAMA_KV_RESIDENT_TARGET_BYTES");
    unsetenv("LLAMA_KV_RESIDENT_TARGET_SOURCE");
    CHECK(!adapter.initialize_static_target(error).valid());

    llama_kv_physical_budget_view physical;
    physical.valid = true;
    physical.resident_available = true;
    physical.reclaimable_available = true;
    physical.swapped_metadata_consistent = true;
    physical.object_id = 11;
    physical.generation = 3;
    physical.resident_bytes = 12288;
    physical.dead_resident_reclaimable_bytes = 4096;
    CHECK(adapter.set_global_target({ true, 8192, 11, 3 }));
    auto target = adapter.effective_target();
    CHECK(target.valid() && std::string(target.source) == "global_dynamic");
    CHECK(target.source_object_id == 11 && target.source_generation == 3);
    CHECK(target.basis_generation != target.source_generation);

    server_kv_pressure_unified_action_config config;
    adapter.apply_effective_target(config);
    CHECK(config.budget_target_enabled && config.budget_target_bytes == 8192);
    CHECK(config.budget_source_object_id == 11 && config.budget_source_generation == 3);

    const auto projected = adapter.project_for_governor(physical);
    CHECK(projected.valid && projected.authority != nullptr);
    CHECK(projected.object_id == 11 && projected.generation == 3);

    llama_kv_resident_sample before;
    before.available = true;
    before.object_id = 11;
    before.generation = 3;
    before.resident_bytes = 10000;
    auto after = before;
    after.resident_bytes = 7000;
    const auto credit = server_kv_budget_adapter::confirm_physical_credit(before, after);
    CHECK(credit.available && credit.value == 3000);
    auto stale = after;
    stale.generation = 4;
    CHECK(!server_kv_budget_adapter::confirm_physical_credit(before, stale).available);
    auto equal = before;
    CHECK(!server_kv_budget_adapter::confirm_physical_credit(before, equal).available);
    auto growth = after;
    growth.resident_bytes = 11000;
    CHECK(!server_kv_budget_adapter::confirm_physical_credit(before, growth).available);
}

static void test_global_target_relaxation_clears_soft_debt_without_prefetch() {
    auto config = budget_config();
    server_kv_governor_state state;
    fake_core core;
    core.responses = { evaluation(3001), no_candidate_release(3001) };
    const auto first = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(1, 3001, kv_pressure_state::NORMAL), {},
            budget_view(12288));
    CHECK(first.budget_debt_after_bytes == 8192);
    CHECK(first.observation.release_attempted);

    config.budget_target_bytes = 12288;
    config.budget_basis_generation += 1;
    core.requests.clear();
    core.responses = { evaluation(3002) };
    const auto relaxed = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(2, 3002, kv_pressure_state::NORMAL), {},
            budget_view(12288));
    CHECK(std::string(relaxed.observation.reason) == "budget_target_relaxed");
    CHECK(core.requests.empty());
    CHECK(!relaxed.observation.release_attempted && !relaxed.observation.offload_attempted);
    CHECK(state.budget_debt_bytes() == 0 && !state.soft_offload_armed());
}

static void test_global_dynamic_target_requires_matching_physical_authority() {
    auto config = budget_config();
    config.budget_source = "global_dynamic";
    config.budget_source_object_id = 99;
    config.budget_source_generation = 3;
    server_kv_governor_state state;
    fake_core core;
    const auto result = server_kv_pressure_execute_governor(
            config, state, core.ops(), governor_pressure(1, 3011, kv_pressure_state::NORMAL), {},
            budget_view(12288));
    CHECK(std::string(result.observation.reason) == "budget_authority_mismatch");
    CHECK(core.requests.empty());
}

int main() {
    test_startup_rejects_global_legacy_mode_conflict();
    test_startup_decision_distinguishes_disabled_and_enabled();
    test_startup_decision_rejects_invalid_unified_configuration();
    test_startup_decision_preserves_legacy_modes_when_unified_is_not_requested();
    test_default_disabled_and_non_pressure_noop();
    test_stale_and_no_memory_are_blocked();
    test_evaluate_blocks_invalid_open_or_unsupported();
    test_release_unsupported_keeps_governor_debt_and_marker_stable();
    test_evaluate_decision_mismatch_never_releases();
    test_release_reuses_decision_and_stops_after_one_action();
    test_release_terminal_results_do_not_chain();
    test_marker_keeps_observation_and_core_result_separate();
    test_governor_scores_idle_claimant_without_active_producer();
    test_v3_cost_aware_ranking_uses_physical_relief_and_feedback();
    test_v3_missing_physical_estimate_falls_back_to_idle_age();
    test_v3_stale_physical_generation_is_excluded();
    test_v3_restore_lease_blocks_immediate_offload();
    test_v3_stale_feedback_is_discarded_after_lineage_loss();
    test_v3_history_survives_ordinary_epoch_rollover();
    test_v3_mixed_evidence_decision_falls_back_to_idle_age();
    test_v3_safety_exclusions_precede_cost_ranking();
    test_v3_policy_defaults_to_v2_when_env_unset();
    test_v3_resume_gate_path_is_policy_independent();
    test_governor_debt_release_then_later_offload();
    test_governor_idle_follow_up_completes_release_windows_and_arms_offload();
    test_governor_sort_is_input_order_independent();
    test_governor_stale_basis_normal_and_reset();
    test_governor_global_exclusions_and_debt_saturation();
    test_governor_io_failure_backoff_and_penalty();
    test_governor_exhaustion_epoch_and_stable_noop();
    test_governor_exhaustion_result_pollution_is_ignored();
    test_v2_target_env_source_and_generation();
    test_v2_hard_pressure_wins_over_soft_budget();
    test_v2_soft_release_then_later_offload();
    test_v2_unmet_budget_terminates_correctness_protected_chasing();
    test_v2_invalid_budget_view_fails_closed();
    test_v2_invalid_pressure_basis_holds_soft_state();
    test_v2_protected_soft_offload_becomes_unmet_terminal();
    test_v2_shared_soft_offload_becomes_unmet_terminal();
    test_global_kv_budget_adapter_authority_and_physical_credit();
    test_global_target_relaxation_clears_soft_debt_without_prefetch();
    test_global_dynamic_target_requires_matching_physical_authority();

    std::printf("server KV pressure unified action tests: %d/%d passed\n",
            tests_total - tests_failed, tests_total);
    return tests_failed == 0 ? 0 : 1;
}
