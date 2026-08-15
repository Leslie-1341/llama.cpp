#!/usr/bin/env python3

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")


class ResidentPreflightStaticTest(unittest.TestCase):
    def test_whole_view_marker_contains_required_authority(self) -> None:
        start = CONTEXT.index("format_kv_resident_preflight_observation(")
        end = CONTEXT.index("static std::string format_kv_resident_preflight_claimant", start)
        formatter = CONTEXT[start:end]
        for field in (
            "timestamp_mono_ns=", "sample_count=", "whole_valid=", "resident_available=",
            "object_id=", "generation=", "resident_bytes=", "transient_staging_bound_bytes=",
            "resident_block_count=", "swapped_block_count=", "released_block_count=",
            "pending_write_block_count=", "global_target_enabled=", "global_target_source=",
            "observation_only=",
        ):
            self.assertIn(field, formatter)

    def test_claimant_observation_is_explicitly_gated_and_read_only(self) -> None:
        start = CONTEXT.index("void emit_resident_preflight_observation(")
        end = CONTEXT.index("void maybe_sample_kv_pressure(", start)
        block = CONTEXT[start:end]
        self.assertIn("if (!kv_g0_s1_resident_preflight)", block)
        self.assertIn("sample_kv_claimant_physical_views", block)
        self.assertIn("format_kv_resident_preflight_claimant", block)
        self.assertNotIn("execute_action", block)
        self.assertNotIn("server_kv_pressure_execute_governor", block)
        self.assertNotIn("prefetch_seq", block)

    def test_preflight_marker_is_observation_only(self) -> None:
        helper_start = CONTEXT.index("void emit_resident_preflight_observation(")
        helper_end = CONTEXT.index("void maybe_sample_kv_pressure(", helper_start)
        helper = CONTEXT[helper_start:helper_end]
        self.assertIn("sample_kv_physical_budget_view", CONTEXT)
        self.assertIn("!kv_pressure_unified_action_config.enabled", helper)
        self.assertIn("!memory_governor_kv_release_enabled", helper)
        self.assertIn("!memory_governor_kv_offload_enabled", helper)
        self.assertIn("observation_only", helper)
        self.assertIn('SRV_INF("%s\\n", format_kv_resident_preflight_observation(', helper)
        sample_start = CONTEXT.index("void maybe_sample_kv_pressure(")
        sample_end = CONTEXT.index("// Per-sample defensive gating", sample_start)
        sample = CONTEXT[sample_start:sample_end]
        self.assertGreaterEqual(sample.count("if (kv_g0_s1_resident_preflight)"), 3)
        self.assertGreaterEqual(sample.count("return;"), 3)

    def test_preflight_environment_switch_is_explicit_and_default_off(self) -> None:
        init_start = CONTEXT.index('const char * resident_observation =')
        init_end = CONTEXT.index('const char * resume_stage_timing =', init_start)
        init = CONTEXT[init_start:init_end]
        self.assertNotIn('std::strcmp(resident_observation, "preflight")', init)
        self.assertIn('std::getenv("LLAMA_KV_RESIDENT_PREFLIGHT")', init)
        self.assertIn('std::strcmp(resident_preflight, "1")', init)
        self.assertIn('kv_g0_s1_resident_preflight = resident_preflight', init)
        self.assertNotIn('kv_g0_s1_resident_preflight = resident_observation_both', init)
        self.assertIn('kv_g0_s1_resident_observation = resident_observation_both', init)


if __name__ == "__main__":
    unittest.main()
