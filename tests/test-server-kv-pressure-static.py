#!/usr/bin/env python3

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "tools/server/server-context.cpp"
RUNTIME_H = ROOT / "tools/server/server-kv-pressure.h"
RUNTIME_CPP = ROOT / "tools/server/server-kv-pressure.cpp"
MEMORY_H = ROOT / "src/llama-memory.h"
KV_CACHE_H = ROOT / "src/llama-kv-cache.h"
KV_CACHE_CPP = ROOT / "src/llama-kv-cache.cpp"
KV_CACHE_RELEASE_H = ROOT / "src/llama-kv-cache-release.h"


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace:index + 1]
    raise AssertionError(f"unterminated function: {signature}")


class ServerKvPressureStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = CONTEXT.read_text(encoding="utf-8")
        cls.runtime = RUNTIME_H.read_text(encoding="utf-8") + RUNTIME_CPP.read_text(encoding="utf-8")
        cls.core = KV_CACHE_CPP.read_text(encoding="utf-8")
        cls.core_h = KV_CACHE_H.read_text(encoding="utf-8")
        cls.release_h = KV_CACHE_RELEASE_H.read_text(encoding="utf-8")
        cls.memory_h = MEMORY_H.read_text(encoding="utf-8")

    def test_single_owner_and_update_slots_only(self):
        self.assertIn("std::unique_ptr<kv_pressure_sampler> kv_pressure_sampler_owner", self.context)
        self.assertEqual(self.context.count("kv_pressure_sampler_owner->sample()"), 1)

        maybe_sample = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)")
        self.assertIn("kv_pressure_sampler_owner->sample()", maybe_sample)

        update_slots = function_body(self.context, "void update_slots()")
        self.assertIn("maybe_sample_kv_pressure(all_idle);", update_slots)
        self.assertLess(
            update_slots.index("maybe_sample_kv_pressure(all_idle);"),
            update_slots.index("if (all_idle)"),
        )

    def test_no_kv_mutation_or_reclaim_calls(self):
        integration = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)") + self.runtime
        forbidden = (
            "paged_release_blocks(",
            "release_blocks(",
            "swap_out(",
            "swap_in(",
            "prefetch_seq(",
            "prefetch_seq_step(",
            "madvise(",
            "MADV_DONTNEED",
        )
        for token in forbidden:
            self.assertNotIn(token, integration)

    def test_dry_run_allowed_tokens(self):
        """Dry-run scanner and release-status query are allowed in maybe_sample_kv_pressure."""
        integration = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)")
        allowed = (
            "bounded_release_dry_run(",
            "paged_release_status()",
        )
        for token in allowed:
            self.assertIn(token, integration)

    def test_no_background_thread_or_lock(self):
        forbidden = ("std::thread", "std::mutex", "std::shared_mutex", "condition_variable")
        for token in forbidden:
            self.assertNotIn(token, self.runtime)

    def test_structured_marker_fields(self):
        required = (
            "state=", "previous_state=", "source=", "sample_valid=", "stale=",
            "rss_kb=", "cgroup_current_bytes=", "cgroup_max_bytes=",
            "psi_some_avg10=", "psi_full_avg10=", "sample_latency_ns=",
            "sample_count=", "skip_count=",
        )
        for field in required:
            self.assertIn(field, self.runtime)

    def test_dry_run_marker_fields(self):
        required = (
            "kv_pressure_dry_run",
            "release_enabled=",
            "would_release_bytes=", "would_release_blocks=",
            "blocks_scanned=", "blocks_skipped_owned=", "blocks_skipped_state=",
            "shortfall_bytes=", "overshoot_bytes=",
            "block_scan_exhausted=", "ownership_aborted=",
            "target_bytes=", "max_scan_blocks=",
            "skipped_reason=", "cooldown_ms=",
        )
        for field in required:
            self.assertIn(field, self.runtime)
        # Verify precise skip-reason differentiation strings.
        # "release_disabled" is NOT a skip reason — it is purely observational
        # (reported as release_enabled=0).  Only structural blockers cause skips.
        precise_reasons = (
            '"not_paged"',
            '"layout_unsupported"',
        )
        for reason in precise_reasons:
            self.assertIn(reason, self.context)

    def test_dry_run_scanner_is_const_and_readonly(self):
        """The dry-run scanner must be declared const and must NOT call madvise or mutate KV state."""
        dry_fn = function_body(self.core, "paged_release_blocks_bounded_dry_run")
        # Must be declared const on the method
        self.assertIn(") const", self.core_h.split("paged_release_blocks_bounded_dry_run")[1].split("{")[0])
        # Must NOT call destructive operations or write to KV state.
        # (We check for mutation patterns: madvise, state assignment, free-list
        # manipulation, and block-used clearing.  == comparisons are fine.)
        dry_run_forbidden = (
            "paged_madvise_block(", "MADV_DONTNEED",
            "paged_free_list.push_back",
            "paged_blocks_in_use",
            "paged_block_release_bytes",
            "paged_blocks_released",
        )
        for token in dry_run_forbidden:
            self.assertNotIn(token, dry_fn)
        # Must NOT gate on paged_block_release_enabled — dry-run is decoupled
        # from LLAMA_KV_PAGED_RELEASE so the scanner works even when
        # destructive release is disabled.
        self.assertNotIn("paged_block_release_enabled", dry_fn)
        # Must call ownership collection (same as destructive path)
        self.assertIn("llama_kv_release_collect_ownership", dry_fn)
        # Must check state gate (PENDING_WRITE / SWAPPED / RELEASED)
        self.assertIn("paged_block_state::PENDING_WRITE", dry_fn)

    def test_dry_run_explicit_evaluation_gate(self):
        """The dry-run evaluation in maybe_sample_kv_pressure must use an
        explicit boolean guard (should_evaluate) and must NOT rely on empty
        branches or sentinel strings to gate the scanner call."""
        integration = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)")
        # Explicit gate must exist
        self.assertIn("should_evaluate", integration)
        # Empty-branch antipattern: an `if` or `else if` whose body is just a
        # comment line followed by closing brace — the NORMAL/RECOVERY and
        # cooldown gates must NOT leak into evaluation.
        self.assertNotIn("// NORMAL / RECOVERY — not an error, just not a trigger state.", integration)
        self.assertNotIn("// Cooldown not yet elapsed — no marker.", integration)

    def test_dry_run_config_default_off(self):
        self.assertIn("enabled         = false", self.runtime)
        self.assertIn("target_bytes    = 0", self.runtime)

    def test_dry_run_lifecycle_in_init(self):
        init_body = function_body(self.context, "void init_kv_pressure_sampler()")
        self.assertIn("kv_pressure_runtime.dry_run_disable()", init_body)
        self.assertIn("server_kv_pressure_dry_run_config_from_env", init_body)

    def test_dry_run_lifecycle_in_sleep_resume(self):
        lifecycle = function_body(self.context, "void handle_sleeping_state(bool new_state)")
        self.assertIn("kv_pressure_runtime.dry_run_disable();", lifecycle)

    def test_master_switch_preflight_precedes_sampler_init(self):
        init_body = function_body(self.context, "void init_kv_pressure_sampler()")
        self.assertLess(
            init_body.index("kv_pressure_sampler_environment_enablement()"),
            init_body.index("std::make_unique<kv_pressure_sampler>()"),
        )
        self.assertRegex(init_body, re.compile(r"DISABLED\) \{\s*return;", re.MULTILINE))

    def test_sleep_resume_resets_sampler_lifecycle(self):
        lifecycle = function_body(self.context, "void handle_sleeping_state(bool new_state)")
        self.assertIn("kv_pressure_sampler_owner.reset();", lifecycle)
        self.assertIn("kv_pressure_runtime.disable();", lifecycle)
        self.assertIn("init_kv_pressure_sampler();", lifecycle)

    def test_result_struct_in_shared_header(self):
        """llama_kv_bounded_release_result must live in the shared release header."""
        self.assertIn("struct llama_kv_bounded_release_result", self.release_h)
        self.assertIn("released_bytes", self.release_h)
        self.assertIn("ownership_aborted", self.release_h)

    def test_memory_virtual_methods_declared(self):
        self.assertIn("virtual llama_kv_bounded_release_result bounded_release_dry_run", self.memory_h)
        self.assertIn("virtual llama_kv_release_status paged_release_status() const", self.memory_h)


if __name__ == "__main__":
    unittest.main()
