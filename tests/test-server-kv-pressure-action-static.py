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
QUEUE_H = (ROOT / "tools/server/server-queue.h").read_text(encoding="utf-8")
QUEUE_CPP = (ROOT / "tools/server/server-queue.cpp").read_text(encoding="utf-8")


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
        self.assertIn('std::strcmp(resident_observation, "both")', CONTEXT)
        init_start = CONTEXT.index("const char * resident_observation =")
        init_end = CONTEXT.index("const char * resume_stage_timing =", init_start)
        init = CONTEXT[init_start:init_end]
        self.assertIn("resident_observation_both", init)
        self.assertIn("kv_g0_s1_resident_observation = resident_observation_both", init)
        self.assertIn("kv_g0_s1_resident_preflight = resident_observation_both", init)

        callback_start = CONTEXT.index("observe_resident = kv_g0_s1_resident_observation")
        callback_end = CONTEXT.index("} : server_kv_pressure_action_ops {}", callback_start)
        callback = CONTEXT[callback_start:callback_end]
        self.assertIn("request.action != llama_kv_action::offload", callback)
        self.assertLess(
            callback.index("const auto before = mem->sample_kv_resident()"),
            callback.index("auto action_result = mem->execute_action(request)"))
        self.assertLess(
            callback.index("auto action_result = mem->execute_action(request)"),
            callback.index("const auto after = mem->sample_kv_resident()"))
        self.assertIn("action_result, (uint64_t) ::getpid(), before, after", callback)
        self.assertIn("physical_relief_available", callback)
        self.assertIn("action_elapsed_us", callback)

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
        self.assertNotIn("llama_kv_action::prefetch", governor)
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
        self.assertIn("server_kv_pressure_snapshot pressure_snapshot", CONTEXT)
        self.assertIn("sample_kv_claimant_physical_views", CONTEXT)
        self.assertIn("std::vector<server_kv_claimant_snapshot> claimant_snapshots", CONTEXT)
        self.assertIn("std::vector<server_kv_claimant_runtime_observation> runtime_claimants", CONTEXT)
        self.assertIn("LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE", CONTEXT)
        self.assertIn("if (kv_governor_claimant_trace)", CONTEXT)
        self.assertIn("get_kv_runtime_claimant(slot.id)", CONTEXT)
        self.assertIn('slot_data["kv_claimant"]', CONTEXT)
        self.assertIn("claimant_exhausted", ACTION_H + ACTION_CPP)
        self.assertNotIn("std::thread", ACTION_H + ACTION_CPP)

    def test_idle_follow_up_uses_progress_bounded_rate_limited_queue_updates(self):
        self.assertIn("bool idle_follow_up_pending() const", ACTION_H)
        governor = ACTION_CPP[
            ACTION_CPP.index("server_kv_pressure_execute_governor("):
            ACTION_CPP.index("server_kv_pressure_unified_action_format_marker(")]
        self.assertIn("release_retry_scheduled", governor)
        self.assertIn("offload_retry_scheduled", governor)
        self.assertIn("release_progressed || release_scan_advanced || armed_offload ||", governor)
        self.assertIn(
            "offload_progressed || claimant_exhausted || offload_retry_scheduled", governor)
        self.assertIn("state.idle_follow_up_pending_ = false", governor)

        hook_start = CONTEXT.index("queue_tasks.on_idle_update_pending(")
        hook_end = CONTEXT.index("queue_tasks.on_sleeping_state(", hook_start)
        hook = CONTEXT[hook_start:hook_end]
        self.assertIn("kv_pressure_unified_action_config.enabled", hook)
        self.assertIn("kv_governor_state.idle_follow_up_pending()", hook)
        self.assertNotIn("queue_tasks.post", hook)

        self.assertIn("callback_idle_update_pending", QUEUE_H)
        wait_start = QUEUE_CPP.index("const bool idle_update_pending")
        wait_end = QUEUE_CPP.index("void server_queue::cleanup_pending_task", wait_start)
        idle_wait = QUEUE_CPP[wait_start:wait_end]
        self.assertIn("if (should_sleep() && !idle_update_pending)", idle_wait)
        self.assertIn("condition_tasks.wait_for(lock, max_wait_time", idle_wait)
        self.assertIn("if (res || idle_update_pending)", idle_wait)
        self.assertNotIn("queue_tasks.push", idle_wait)

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

    def test_v2_target_is_server_state_with_explicit_source_validation(self):
        for token in (
                "server_kv_resident_target_state",
                "server_kv_resident_target_from_env",
                "LLAMA_KV_RESIDENT_TARGET_BYTES",
                "LLAMA_KV_RESIDENT_TARGET_SOURCE",
                "env_static",
                "basis_generation"):
            self.assertIn(token, ACTION_H + ACTION_CPP + CONTEXT)
        self.assertIn("must be env_static", ACTION_CPP)
        self.assertIn("kv_resident_target_state", CONTEXT)
        self.assertIn("budget_target_enabled", CONTEXT)
        self.assertIn("budget_target_bytes", CONTEXT)
        self.assertNotIn("/kv_resident_target", CONTEXT)
        self.assertNotIn("OFFLOAD_GLOBAL", ACTION_CPP + ACTION_H)

    def test_v2_soft_chain_preserves_hard_priority_and_single_action(self):
        self.assertIn("budget_debt_bytes_", ACTION_H + ACTION_CPP)
        self.assertIn("soft_offload_armed_", ACTION_H + ACTION_CPP)
        self.assertIn("unmet_budget_bytes_", ACTION_H + ACTION_CPP)
        self.assertIn("budget_unmet_terminal", ACTION_CPP)
        self.assertIn("budget_view_unavailable", ACTION_CPP)
        self.assertIn("budget_target_satisfied", ACTION_CPP)
        self.assertIn("if (!state.soft_offload_armed_)", ACTION_CPP)
        self.assertIn("result.release.reason == llama_kv_action_reason::no_candidate", ACTION_CPP)
        self.assertIn("llama_kv_action::offload", ACTION_CPP)
        governor = ACTION_CPP[
            ACTION_CPP.index("server_kv_pressure_execute_governor("):
            ACTION_CPP.index("server_kv_pressure_unified_action_format_marker(")]
        basis_gate = governor.index("if (!pressure.pressure_basis_valid)")
        soft_start = governor.index("if (pressure.state != kv_pressure_state::PRESSURE")
        pressure_start = governor.index("result.observed_excess_bytes")
        self.assertLess(basis_gate, soft_start)
        self.assertLess(soft_start, pressure_start)
        for token in ("protected_or_shared", "budget_unmet_terminal", "budget_unmet_backoff_samples"):
            self.assertIn(token, governor)
        for field in (
                "budget_active=", "budget_target_enabled=", "budget_source=",
                "budget_basis_generation=", "budget_view_valid=",
                "budget_resident_available=", "budget_reclaimable_available=",
                "budget_resident_bytes=",
                "budget_dead_resident_reclaimable_bytes=",
                "budget_transient_staging_bound_bytes=", "budget_observed_excess_bytes=",
                "budget_debt_before_bytes=", "budget_debt_after_bytes=",
                "soft_offload_armed_before=", "soft_offload_armed_after=",
                "budget_next_action_sample=", "unmet_budget_bytes_after="):
            self.assertIn(field, ACTION_CPP)
        self.assertNotIn("std::thread", ACTION_H + ACTION_CPP)

    def test_v2_budget_view_is_one_scheduler_cadence_call_and_not_per_token(self):
        self.assertEqual(CONTEXT.count("mem->sample_kv_physical_budget_view()"), 1)
        self.assertIn("maybe_sample_kv_pressure", CONTEXT)
        self.assertLess(
            CONTEXT.index("mem->sample_kv_physical_budget_view()"),
            CONTEXT.index("server_kv_pressure_execute_governor"))
        update_start = CONTEXT.index("void update_slots()")
        update_end = CONTEXT.index("if (all_idle)", update_start)
        self.assertIn("maybe_sample_kv_pressure(all_idle)", CONTEXT[update_start:update_end])
        self.assertNotIn("sample_kv_physical_budget_view()", LLAMA_CONTEXT)

    def test_v2_reclaimable_view_does_not_reuse_swap_disabled_gate(self):
        sampler = KV_CACHE_CPP[
            KV_CACHE_CPP.index("llama_kv_release_budget_snapshot llama_kv_cache::sample_kv_release_budget()"):
            KV_CACHE_CPP.index("llama_kv_physical_budget_view llama_kv_cache::sample_kv_physical_budget_view()")]
        self.assertIn("paged_layout_valid", sampler)
        self.assertIn("paged_release_blocks_bounded_dry_run", sampler)
        self.assertNotIn("if (!bounded_release_can_enable()", sampler)
        self.assertNotIn("if (bounded_release_can_enable()", sampler)
        self.assertIn("llama_kv_release_collect_ownership", KV_CACHE_CPP)
        self.assertIn("paged_release_blocks_bounded_impl", KV_CACHE_CPP)
        self.assertIn("execute_action", KV_CACHE_CPP)

    def test_v2_keeps_target_separate_from_rss_and_staging(self):
        self.assertIn("transient_staging_bound_bytes", ACTION_H + ACTION_CPP)
        self.assertIn("not subtracted", ACTION_H)
        self.assertNotIn("pressure_current_bytes - config.budget_target_bytes", ACTION_CPP)
        self.assertNotIn("memory.current", ACTION_CPP)
        self.assertNotIn("cgroup", ACTION_CPP)


if __name__ == "__main__":
    unittest.main()
