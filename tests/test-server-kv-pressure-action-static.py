#!/usr/bin/env python3

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
SERVER = (ROOT / "tools/server/server.cpp").read_text(encoding="utf-8")
LLAMA_CONTEXT = (ROOT / "src/llama-context.cpp").read_text(encoding="utf-8")
MEMORY_H = (ROOT / "src/llama-memory.h").read_text(encoding="utf-8")
KV_CACHE_H = (ROOT / "src/llama-kv-cache.h").read_text(encoding="utf-8")
KV_CACHE_CPP = (ROOT / "src/llama-kv-cache.cpp").read_text(encoding="utf-8")
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

    def test_production_capability_record_uses_live_kv_state(self):
        start = CONTEXT.index("void log_kv_governor_capability() const")
        end = CONTEXT.index("// unlike load_model()", start)
        record = CONTEXT[start:end]
        self.assertEqual(record.count("KV_GOVERNOR_CAPABILITY"), 1)
        self.assertIn("llama_get_memory(ctx_tgt)", record)
        self.assertIn("get_kv_runtime_capability()", record)
        self.assertIn("slots.size()", record)
        self.assertNotIn("params_base.n_parallel", record)
        for field in (
                "n_slots=", "n_seq_max=", "n_stream=", "kv_unified=",
                "paged_metadata=", "ingraph_gather=", "release_supported=",
                "offload_supported=", "prefetch_supported=", "backing_ready=",
                "swap_explicit_only="):
            self.assertIn(field, record)
        self.assertIn("virtual llama_kv_runtime_capability get_kv_runtime_capability() const", MEMORY_H)
        self.assertIn("virtual llama_kv_runtime_claimant get_kv_runtime_claimant", MEMORY_H)
        self.assertIn("llama_kv_runtime_capability get_kv_runtime_capability() const override", KV_CACHE_H)
        self.assertIn("llama_kv_runtime_claimant get_kv_runtime_claimant", KV_CACHE_H)
        capability = KV_CACHE_CPP[
            KV_CACHE_CPP.index("llama_kv_runtime_capability llama_kv_cache::get_kv_runtime_capability() const"):
            KV_CACHE_CPP.index("llama_kv_action_result llama_kv_cache::execute_action(")]
        self.assertIn("paged_swap_enabled", capability)
        self.assertIn("kv_swap_store", capability)
        self.assertIn("paged_swap_explicit_only", capability)
        self.assertIn("paged_write_context_invalid", capability)
        self.assertIn("LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY", KV_CACHE_CPP)
        self.assertIn("!kv->paged_swap_explicit_only", KV_CACHE_CPP)
        self.assertIn("requires LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY=1", CONTEXT)

    def test_g0_s1_resident_observation_wraps_real_offload_transaction(self):
        self.assertIn("LLAMA_KV_G0_S1_RESIDENT_OBSERVATION", CONTEXT)
        self.assertIn('std::strcmp(resident_observation, "1")', CONTEXT)
        self.assertIn('std::strcmp(resident_observation, "preflight")', CONTEXT)

        callback_start = CONTEXT.index("[mem, observe_resident = kv_g0_s1_resident_observation]")
        callback_end = CONTEXT.index("} : server_kv_pressure_action_ops {}", callback_start)
        callback = CONTEXT[callback_start:callback_end]
        self.assertIn("request.action != llama_kv_action::offload", callback)
        self.assertLess(
            callback.index("const auto before = mem->sample_kv_resident()"),
            callback.index("const auto action_result = mem->execute_action(request)"))
        self.assertLess(
            callback.index("const auto action_result = mem->execute_action(request)"),
            callback.index("const auto after = mem->sample_kv_resident()"))
        self.assertIn("action_result, (uint64_t) ::getpid(), before, after", callback)

        formatter_start = CONTEXT.index("format_kv_g0_s1_resident_observation(")
        formatter_end = CONTEXT.index("// state diagram:", formatter_start)
        formatter = CONTEXT[formatter_start:formatter_end]
        for field in (
                "kv_g0_s1_resident_observation", "source=paged_sample_mincore", "action=offload",
                "decision_id=", "seq_id=", "transaction_id=", "server_pid=",
                "before_available=", "before_object_id=", "before_generation=",
                "before_resident_bytes=", "before_resident_pages=",
                "after_available=", "after_object_id=", "after_generation=",
                "after_resident_bytes=", "after_resident_pages="):
            self.assertIn(field, formatter)

        metrics_start = CONTEXT.index("case SERVER_TASK_TYPE_METRICS:")
        metrics_end = CONTEXT.index("case SERVER_TASK_TYPE_SLOT_SAVE:", metrics_start)
        metrics = CONTEXT[metrics_start:metrics_end]
        self.assertIn("kv_g0_s1_resident_preflight && mem", metrics)
        self.assertIn("sample_kv_resident()", metrics)
        self.assertIn('slot_data["kv_resident"]', metrics)
        self.assertNotIn("kv_g0_s1_resident_observation && mem", metrics)
        self.assertIn("virtual llama_kv_resident_sample sample_kv_resident() const", MEMORY_H)
        self.assertIn("llama_kv_resident_sample sample_kv_resident() const override", KV_CACHE_H)
        self.assertIn("llama_kv_resident_sample llama_kv_cache::sample_kv_resident() const", KV_CACHE_CPP)
        self.assertIn("paged_sample_mincore()", KV_CACHE_CPP)

    def test_pressure_and_resume_share_one_decision_source(self):
        self.assertIn("uint64_t kv_decision_next = 0", CONTEXT)
        self.assertEqual(CONTEXT.count("++kv_decision_next"), 2)
        self.assertNotIn("kv_resume_decision_next", CONTEXT)

    def test_actions_are_logical_and_single_decision_bounded(self):
        legacy = ACTION_CPP[
            ACTION_CPP.index("server_kv_pressure_execute_unified_action("):
            ACTION_CPP.index("void server_kv_governor_state::reset()")]
        governor = ACTION_CPP[
            ACTION_CPP.index("server_kv_pressure_execute_governor("):
            ACTION_CPP.index("server_kv_pressure_unified_action_format_marker(")]
        self.assertIn("llama_kv_action::evaluate", legacy)
        self.assertIn("llama_kv_action::release", legacy)
        self.assertNotIn("llama_kv_action::offload", legacy)
        self.assertIn("llama_kv_action::release", governor)
        self.assertIn("llama_kv_action::offload", governor)
        self.assertIn("if (!state.offload_armed_)", governor)
        self.assertIn("result.release.reason == llama_kv_action_reason::no_candidate", governor)
        for physical_detail in (
                "paged_block", "paged_free_list", "paged_resolve", "backing_store", "physical_block"):
            self.assertNotIn(physical_detail, ACTION_H)
            self.assertNotIn(physical_detail, ACTION_CPP)

    def test_governor_scores_idle_claimants_without_active_producer_gate(self):
        governor = ACTION_CPP[
            ACTION_CPP.index("server_kv_pressure_execute_governor("):
            ACTION_CPP.index("server_kv_pressure_unified_action_format_marker(")]
        self.assertNotIn("no_active_claimant", governor)
        self.assertNotIn("has_active_claimant", governor)
        self.assertLess(
            governor.index("result.scores.reserve(claimants.size())"),
            governor.index("const auto selected = std::find_if("))

    def test_governor_snapshot_score_and_lifecycle_wiring(self):
        for token in (
                "server_kv_pressure_snapshot", "server_kv_claimant_snapshot",
                "server_kv_governor_state", "pressure_debt_bytes_",
                "idle_age_score", "logical_kv_score", "reclaimable_score",
                "lcp_n_past_penalty", "io_cost_penalty", "failure_penalty"):
            self.assertIn(token, ACTION_H + ACTION_CPP)
        self.assertGreaterEqual(CONTEXT.count("kv_governor_state.reset();"), 2)
        self.assertIn("server_kv_pressure_execute_governor", CONTEXT)
        self.assertIn("const server_kv_pressure_snapshot pressure_snapshot", CONTEXT)
        self.assertIn("std::vector<server_kv_claimant_snapshot> claimant_snapshots", CONTEXT)
        self.assertIn("std::vector<server_kv_claimant_runtime_observation> runtime_claimants", CONTEXT)
        self.assertIn("LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE", CONTEXT)
        self.assertIn("if (kv_governor_claimant_trace)", CONTEXT)
        self.assertIn("get_kv_runtime_claimant(slot.id)", CONTEXT)
        self.assertIn('slot_data["kv_claimant"]', CONTEXT)
        self.assertIn("claimant_exhausted", ACTION_H + ACTION_CPP)
        self.assertNotIn("std::thread", ACTION_H + ACTION_CPP)

    def test_evaluate_gates_release_and_marker_keeps_results_distinct(self):
        for gate in (
                "context_invalid", "write_transaction_open", "fail_stop", "can_release"):
            self.assertIn(gate, ACTION_CPP)
        for field in (
                "decision_id=", "transaction_id=", "outcome=", "reason=", "blocks=", "bytes=",
                "relieved_bytes=", "shortfall_bytes=", "io_failure=", "io_errno=", "state_changed=",
                "episode=", "debt_before_bytes=", "debt_after_bytes=",
                "offload_armed_before=", "offload_armed_after=", "selected_seq_id=",
                "claimants=", "scores="):
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
