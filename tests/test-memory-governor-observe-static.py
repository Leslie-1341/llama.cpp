#!/usr/bin/env python3

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVER_CONTEXT = (ROOT / "tools/server/server-context.cpp").read_text()
MOE_H = (ROOT / "src/llama-moe-buffer.h").read_text()
MOE_CPP = (ROOT / "src/llama-moe-buffer.cpp").read_text()


class MemoryGovernorObserveStaticTest(unittest.TestCase):
    def observe_function(self):
        start = SERVER_CONTEXT.index("publish_memory_governor_observation(")
        start = SERVER_CONTEXT.rfind("\n", 0, start) + 1
        return SERVER_CONTEXT[
            start:
            SERVER_CONTEXT.index("bool init_kv_pressure_sampler()")
        ]

    def test_observe_env_gate_and_marker_schema_present(self):
        self.assertIn("LLAMA_MEMORY_GOVERNOR_OBSERVE", SERVER_CONTEXT)
        self.assertIn("LLAMA_MEMORY_GOVERNOR_OBSERVE_MS", SERVER_CONTEXT)
        self.assertIn('"memory_governor_observe"', SERVER_CONTEXT)
        for field in [
            "dense_resident_bytes",
            "dense_reclaimable_bytes",
            "moe_resident_bytes",
            "moe_budget_bytes",
            "kv_resident_bytes",
            "kv_reclaimable_resident_bytes",
            "pressure_excess_bytes",
            "would_reclaim_candidates",
            "would_prefetch_candidates",
            "clean_reclaim_enabled",
            "clean_reclaim_attempted",
            "clean_reclaim_released_bytes",
            "kv_release_enabled",
            "kv_release_attempted",
            "kv_release_relieved_bytes",
            "kv_release_reason",
            "kv_offload_enabled",
            "kv_offload_attempted",
            "kv_offload_seq_id",
            "kv_offload_relieved_bytes",
            "kv_offload_reason",
            "prefetch_budget_enabled",
            "prefetch_budget_tick_bytes",
            "prefetch_budget_dense_bytes",
            "prefetch_budget_moe_bytes",
            "prefetch_budget_kv_resume_used_bytes",
        ]:
            self.assertIn(field, SERVER_CONTEXT)

    def test_observe_path_scores_would_candidates(self):
        observe = self.observe_function()
        self.assertIn("memory_governor_format_candidates(reclaim_candidates, 3)", observe)
        self.assertIn("memory_governor_format_candidates(prefetch_candidates, 3)", observe)
        for token in [
            '"dense_layer"',
            '"moe_expert"',
            '"kv_global"',
            '"kv_sequence"',
            '"reclaim_clean"',
            '"release"',
            '"offload"',
            '"prefetch"',
        ]:
            self.assertIn(token, observe)

    def test_clean_reclaim_is_gated_and_kv_safe(self):
        observe = self.observe_function()
        self.assertIn("memory_governor_clean_reclaim_enabled", observe)
        self.assertIn("llama_flex_reclaim_released(", observe)
        self.assertIn("llama_moe_buffer_reclaim_clean(", observe)
        forbidden = [
            "bounded_release(",
            "bounded_release_dry_run(",
            "llama_flex_request_layer(",
            "llama_flex_release_layer(",
            "llama_moe_buffer_prefetch(",
            "llama_moe_buffer_prefetch_ranked(",
        ]
        for token in forbidden:
            self.assertNotIn(token, observe)

    def test_kv_release_is_gated(self):
        observe = self.observe_function()
        self.assertIn("LLAMA_MEMORY_GOVERNOR_KV_RELEASE", SERVER_CONTEXT)
        self.assertIn("memory_governor_kv_release_enabled", observe)
        self.assertIn("memory_governor_select_kv_release_candidate", observe)
        self.assertIn("llama_kv_action::evaluate", observe)
        self.assertIn("llama_kv_action::release", observe)
        self.assertIn("telemetry->state != kv_pressure_state::PRESSURE", observe)
        self.assertIn("memory_governor_kv_release_next_sample", observe)

    def test_kv_offload_is_gated_release_first_and_seq_scoped(self):
        observe = self.observe_function()
        self.assertIn("LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD", SERVER_CONTEXT)
        self.assertIn("memory_governor_kv_offload_enabled", observe)
        self.assertIn("memory_governor_select_kv_offload_candidate", observe)
        self.assertIn('"kv_sequence"', observe)
        self.assertIn('"offload"', observe)
        self.assertIn("release_disabled", observe)
        self.assertIn("release_cooldown", observe)
        self.assertIn("pressure_satisfied", observe)
        self.assertIn("kv_release_result.relieved_bytes", observe)
        self.assertIn("llama_kv_action::offload", observe)
        self.assertIn("selected.id", observe)
        self.assertIn("llama_kv_io_class::capacity_write", observe)

    def test_memory_governor_kv_release_supersedes_old_unified_path(self):
        self.assertIn("LLAMA_MEMORY_GOVERNOR_KV_RELEASE/OFFLOAD supersedes", SERVER_CONTEXT)
        unified_path = SERVER_CONTEXT[
            SERVER_CONTEXT.index("--- EdgeKV Governor: RELEASE first"):
            SERVER_CONTEXT.index("--- Bounded destructive release evaluation")
        ]
        self.assertIn("!memory_governor_kv_release_enabled", unified_path)
        self.assertIn("!memory_governor_kv_offload_enabled", unified_path)

    def test_prefetch_budget_is_unified_across_dense_moe_and_kv_resume(self):
        observe = self.observe_function()
        self.assertIn("LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MB_PER_TICK", SERVER_CONTEXT)
        self.assertIn("memory_governor_prefetch_budget_enabled", observe)
        self.assertIn("llama_flex_set_prefetch_budget", observe)
        self.assertIn("llama_moe_buffer_set_prefetch_budget", observe)
        self.assertIn("memory_governor_prefetch_budget_available_bytes", observe)
        self.assertIn("prefetch_budget_dense_bytes", observe)
        self.assertIn("prefetch_budget_moe_bytes", observe)
        self.assertIn("memory_governor_prefetch_budget_reserve", SERVER_CONTEXT)

    def test_moe_stats_and_reclaim_accessors_are_present(self):
        self.assertIn("struct llama_moe_buffer_stats", MOE_H)
        self.assertIn("llama_moe_buffer_get_stats", MOE_H)
        self.assertIn("struct llama_moe_buffer_reclaim_result", MOE_H)
        self.assertIn("llama_moe_buffer_reclaim_clean", MOE_H)
        impl = MOE_CPP[
            MOE_CPP.index("llama_moe_buffer_stats llama_moe_buffer_get_stats("):
        ]
        self.assertIn("std::lock_guard<std::mutex> lk(ctx.mtx)", impl)
        self.assertIn("ctx.resident_bytes", impl)
        self.assertIn("ctx.params.budget_bytes", impl)
        self.assertIn("ctx.expert_total", impl)


if __name__ == "__main__":
    unittest.main()
