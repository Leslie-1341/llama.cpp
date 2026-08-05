#include "server-kv-resume.h"

#include <cstdio>
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
    llama_kv_action_result next;
    llama_kv_action_request request;
    std::vector<bool> protections;
    uint32_t execute_calls = 0;

    server_kv_resume_ops ops() {
        return {
            [this](llama_seq_id seq_id, bool enabled) {
                CHECK(seq_id == 7);
                protections.push_back(enabled);
            },
            [this](const llama_kv_action_request & value) {
                request = value;
                ++execute_calls;
                return next;
            },
        };
    }
};

static llama_kv_action_result completed(uint64_t decision_id) {
    llama_kv_action_result result;
    result.action = llama_kv_action::prefetch;
    result.decision_id = decision_id;
    result.outcome = llama_kv_action_outcome::completed;
    return result;
}

static void test_no_swap_noop_and_legacy_compatibility() {
    fake_core core;
    core.next.action = llama_kv_action::prefetch;
    core.next.decision_id = 1001;
    core.next.outcome = llama_kv_action_outcome::no_op;

    const auto result = server_kv_resume_gate(
            core.ops(), server_kv_resume_trigger::resume, 7, 1001);
    CHECK(result.graph_allowed);
    CHECK(core.execute_calls == 1);
    CHECK(core.request.action == llama_kv_action::prefetch);
    CHECK(core.request.seq_id == 7);
    CHECK(core.request.decision_id == 1001);
    CHECK(core.request.correctness_required && core.request.all_required);
    CHECK(core.request.max_blocks == 0 && core.request.target_bytes == 0);
    CHECK(core.protections.size() == 1 && core.protections[0]);

    const auto legacy = server_kv_resume_gate(
            server_kv_resume_ops {}, server_kv_resume_trigger::active_access, 7, 1002);
    CHECK(legacy.graph_allowed);
    CHECK(legacy.action.outcome == llama_kv_action_outcome::no_op);
}

static void test_success_keeps_protection_until_lifecycle_end() {
    fake_core core;
    core.next = completed(1101);
    core.next.core_transaction_id = 55;
    core.next.blocks = 2;
    core.next.bytes = 4096;

    const auto result = server_kv_resume_gate(
            core.ops(), server_kv_resume_trigger::active_access, 7, 1101);
    CHECK(result.graph_allowed);
    CHECK(core.protections.size() == 1 && core.protections.front());

    core.ops().set_protected(7, false);
    CHECK(core.protections.size() == 2 && !core.protections.back());
}


static void test_resume_events_bind_prefetch_to_graph_gate() {
    fake_core core;
    core.next = completed(1151);
    core.next.core_transaction_id = 91;

    const auto result = server_kv_resume_gate(
            core.ops(), server_kv_resume_trigger::active_access, 7, 1151);
    const auto prefetch = server_kv_resume_format_event(result, 7, 4, false);
    const auto graph_gate = server_kv_resume_format_event(result, 7, 4, true);
    CHECK(prefetch.find("kv_resume_order_event phase=prefetch") != std::string::npos);
    CHECK(prefetch.find("decision_id=1151") != std::string::npos);
    CHECK(prefetch.find("seq_id=7") != std::string::npos);
    CHECK(prefetch.find("claimant_epoch=4") != std::string::npos);
    CHECK(prefetch.find("transaction_id=91") != std::string::npos);
    CHECK(prefetch.find("graph_allowed=1") != std::string::npos);
    CHECK(graph_gate.find("phase=graph_gate") != std::string::npos);
}

static void test_resume_stage_timing_event() {
    const server_kv_resume_stage_timing timing {
        1151,
        7,
        91,
        2,
        4096,
        100,
        200,
        300,
        700,
    };
    const auto event = server_kv_resume_format_stage_timing(timing);
    CHECK(event.find("kv_resume_stage_timing") != std::string::npos);
    CHECK(event.find("decision_id=1151") != std::string::npos);
    CHECK(event.find("seq_id=7") != std::string::npos);
    CHECK(event.find("transaction_id=91") != std::string::npos);
    CHECK(event.find("restored_blocks=2") != std::string::npos);
    CHECK(event.find("restored_bytes=4096") != std::string::npos);
    CHECK(event.find("queue_us=100") != std::string::npos);
    CHECK(event.find("gate_us=200") != std::string::npos);
    CHECK(event.find("graph_us=300") != std::string::npos);
    CHECK(event.find("total_us=700") != std::string::npos);
}

static void test_first_block_failure_blocks_graph() {
    fake_core core;
    core.next.action = llama_kv_action::prefetch;
    core.next.decision_id = 1201;
    core.next.outcome = llama_kv_action_outcome::failed;
    core.next.reason = llama_kv_action_reason::prefetch_failed;
    core.next.io_failure = true;
    core.next.fail_stop = true;
    core.next.shortfall_bytes = 2048;

    const auto result = server_kv_resume_gate(
            core.ops(), server_kv_resume_trigger::active_access, 7, 1201);
    CHECK(!result.graph_allowed);
    CHECK(core.execute_calls == 1);
    const auto message = server_kv_resume_failure_message(result);
    CHECK(message.find("decision_id=1201") != std::string::npos);
    CHECK(message.find("completed_blocks=0") != std::string::npos);
    CHECK(message.find("reason=prefetch_failed") != std::string::npos);
}

static void test_partial_failure_preserves_result_and_blocks_graph() {
    fake_core core;
    core.next.action = llama_kv_action::prefetch;
    core.next.decision_id = 1301;
    core.next.core_transaction_id = 77;
    core.next.outcome = llama_kv_action_outcome::partial_failure;
    core.next.reason = llama_kv_action_reason::prefetch_failed;
    core.next.state_changed = true;
    core.next.io_failure = true;
    core.next.fail_stop = true;
    core.next.blocks = 1;
    core.next.bytes = 4096;
    core.next.shortfall_bytes = 4096;

    const auto result = server_kv_resume_gate(
            core.ops(), server_kv_resume_trigger::resume, 7, 1301);
    CHECK(!result.graph_allowed);
    CHECK(result.action.state_changed && result.action.core_transaction_id == 77);
    const auto message = server_kv_resume_failure_message(result);
    CHECK(message.find("transaction_id=77") != std::string::npos);
    CHECK(message.find("completed_blocks=1") != std::string::npos);
}

static void test_shortfall_fail_stop_and_context_invalid_block_graph() {
    fake_core shortfall;
    shortfall.next = completed(1401);
    shortfall.next.shortfall_bytes = 1;
    CHECK(!server_kv_resume_gate(
            shortfall.ops(), server_kv_resume_trigger::active_access, 7, 1401).graph_allowed);

    fake_core fail_stop;
    fail_stop.next = completed(1402);
    fail_stop.next.fail_stop = true;
    CHECK(!server_kv_resume_gate(
            fail_stop.ops(), server_kv_resume_trigger::active_access, 7, 1402).graph_allowed);

    fake_core invalid;
    invalid.next = completed(1403);
    invalid.next.capability.context_invalid = true;
    CHECK(!server_kv_resume_gate(
            invalid.ops(), server_kv_resume_trigger::active_access, 7, 1403).graph_allowed);
}

int main() {
    test_no_swap_noop_and_legacy_compatibility();
    test_success_keeps_protection_until_lifecycle_end();
    test_resume_events_bind_prefetch_to_graph_gate();
    test_resume_stage_timing_event();
    test_first_block_failure_blocks_graph();
    test_partial_failure_preserves_result_and_blocks_graph();
    test_shortfall_fail_stop_and_context_invalid_block_graph();

    std::printf("server KV resume tests: %d/%d passed\n", tests_total - tests_failed, tests_total);
    return tests_failed == 0 ? 0 : 1;
}
