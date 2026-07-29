#!/usr/bin/env python3

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
SERVER = (ROOT / "tools/server/server.cpp").read_text(encoding="utf-8")
LLAMA_CONTEXT = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
ACTION_H = (ROOT / "tools/server/server-kv-pressure-action.h").read_text(encoding="utf-8")
ACTION_CPP = (ROOT / "tools/server/server-kv-pressure-action.cpp").read_text(encoding="utf-8")


class UnifiedPressureActionStaticTest(unittest.TestCase):
    def test_startup_decision_rejects_before_runtime_wiring(self):
        self.assertIn("LLAMA_KV_PRESSURE_UNIFIED_ACTION", ACTION_CPP)
        self.assertIn("server_kv_pressure_unified_action_startup_status", ACTION_H)
        self.assertIn("server_kv_pressure_unified_action_startup_decide_from_env", ACTION_H)
        self.assertIn("server_kv_pressure_unified_action_startup_decide_from_env()", CONTEXT)
        self.assertIn("KV pressure unified action configuration error", CONTEXT)
        self.assertIn("server initialization rejected", CONTEXT)
        self.assertNotIn("KV pressure unified action disabled", CONTEXT)
        self.assertIn("if (!init_kv_pressure_sampler())", CONTEXT)
        self.assertLess(
            CONTEXT.index("server_kv_pressure_unified_action_startup_decide_from_env()"),
            CONTEXT.index("kv_pressure_sampler_environment_enablement()"))
        self.assertLess(
            CONTEXT.index("if (!init_kv_pressure_sampler())"),
            CONTEXT.index("queue_tasks.on_new_task"))
        self.assertLess(
            SERVER.index("ctx_server.load_model(params)"),
            SERVER.index("ctx_server.start_loop()"))
        self.assertIn("llama_decode", CONTEXT)
        self.assertIn("mctx->apply()", LLAMA_CONTEXT)

    def test_pressure_and_resume_share_one_decision_source(self):
        self.assertIn("uint64_t kv_decision_next = 0", CONTEXT)
        self.assertEqual(CONTEXT.count("++kv_decision_next"), 2)
        self.assertNotIn("kv_resume_decision_next", CONTEXT)

    def test_action_is_logical_and_single_destructive_release(self):
        self.assertIn("llama_kv_action::evaluate", ACTION_CPP)
        self.assertIn("llama_kv_action::release", ACTION_CPP)
        self.assertNotIn("llama_kv_action::offload", ACTION_CPP)
        for physical_detail in (
                "paged_block", "paged_free_list", "paged_resolve", "backing_store", "physical_block"):
            self.assertNotIn(physical_detail, ACTION_H)
            self.assertNotIn(physical_detail, ACTION_CPP)

    def test_evaluate_gates_release_and_marker_keeps_results_distinct(self):
        for gate in (
                "context_invalid", "write_transaction_open", "fail_stop", "can_release"):
            self.assertIn(gate, ACTION_CPP)
        for field in (
                "decision_id=", "transaction_id=", "outcome=", "reason=", "blocks=", "bytes=",
                "shortfall_bytes=", "io_failure=", "state_changed="):
            self.assertIn(field, ACTION_CPP)
        self.assertNotIn("rss_", ACTION_CPP)
        self.assertNotIn("mincore", ACTION_CPP)
        self.assertIn("decision_mismatch", ACTION_CPP)
        self.assertIn("release_submitted", ACTION_CPP)
        self.assertNotIn("release_complete", ACTION_CPP)
        for reason in (
                "zero_budget", "no_candidate", "scan_budget_exhausted", "target_satisfied",
                "target_shortfall", "unsupported", "blocked", "failed"):
            self.assertIn(f'return "{reason}"', ACTION_CPP)


if __name__ == "__main__":
    unittest.main()
