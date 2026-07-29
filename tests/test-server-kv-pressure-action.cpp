#include "server-kv-pressure-action.h"

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
    return result;
}

static server_kv_pressure_unified_action_config enabled_config() {
    server_kv_pressure_unified_action_config config;
    config.enabled = true;
    config.target_bytes = 4096;
    config.max_blocks = 3;
    return config;
}

static void clear_pressure_action_env() {
    unsetenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION");
    unsetenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES");
    unsetenv("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS");
    unsetenv("LLAMA_KV_PAGED_RELEASE");
    unsetenv("LLAMA_KV_PRESSURE_DRY_RUN");
    unsetenv("LLAMA_KV_PRESSURE_BOUNDED_RELEASE");
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
    const auto result = server_kv_pressure_execute_unified_action(
            enabled_config(), core.ops(), pressure(), true, 9, 1050);
    const auto marker = server_kv_pressure_unified_action_format_marker(result);
    CHECK(marker.find("decision_id=1050") != std::string::npos);
    CHECK(marker.find("transaction_id=99") != std::string::npos);
    CHECK(marker.find("state_changed=1") != std::string::npos);
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

int main() {
    test_startup_decision_distinguishes_disabled_and_enabled();
    test_startup_decision_rejects_invalid_unified_configuration();
    test_startup_decision_preserves_legacy_modes_when_unified_is_not_requested();
    test_default_disabled_and_non_pressure_noop();
    test_stale_and_no_memory_are_blocked();
    test_evaluate_blocks_invalid_open_or_unsupported();
    test_evaluate_decision_mismatch_never_releases();
    test_release_reuses_decision_and_stops_after_one_action();
    test_release_terminal_results_do_not_chain();
    test_marker_keeps_observation_and_core_result_separate();

    std::printf("server KV pressure unified action tests: %d/%d passed\n",
            tests_total - tests_failed, tests_total);
    return tests_failed == 0 ? 0 : 1;
}
