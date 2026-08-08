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
            "pressure_basis_valid=", "pressure_current_bytes=",
            "pressure_low_water_bytes=", "pressure_basis_generation=",
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
            '"not_ingraph"',
            '"no_layers"',
            '"no_row_idx"',
            '"swap_enabled"',
            '"dry_run_active"',
            '"structurally_disabled"',
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
        init_body = function_body(self.context, "bool init_kv_pressure_sampler()")
        self.assertIn("kv_pressure_runtime.dry_run_disable()", init_body)
        self.assertIn("server_kv_pressure_dry_run_config_from_env", init_body)

    def test_dry_run_lifecycle_in_sleep_resume(self):
        lifecycle = function_body(self.context, "void handle_sleeping_state(bool new_state)")
        self.assertIn("kv_pressure_runtime.dry_run_disable();", lifecycle)

    def test_master_switch_preflight_precedes_sampler_init(self):
        init_body = function_body(self.context, "bool init_kv_pressure_sampler()")
        self.assertLess(
            init_body.index("kv_pressure_sampler_environment_enablement()"),
            init_body.index("std::make_unique<kv_pressure_sampler>()"),
        )
        self.assertRegex(
            init_body,
            re.compile(r"KV_PRESSURE_ENABLEMENT_DISABLED\) \{\s*return true;", re.MULTILINE),
        )

    def test_sleep_resume_resets_sampler_lifecycle(self):
        lifecycle = function_body(self.context, "void handle_sleeping_state(bool new_state)")
        self.assertIn("kv_pressure_sampler_owner.reset();", lifecycle)
        self.assertIn("kv_pressure_runtime.disable();", lifecycle)
        self.assertIn("if (!init_kv_pressure_sampler())", lifecycle)
        self.assertIn(
            'GGML_ABORT("invalid KV pressure action configuration after sleeping")',
            lifecycle,
        )

    def test_main_init_rejects_sampler_failure(self):
        main_init = function_body(self.context, "bool init()")
        self.assertRegex(
            main_init,
            re.compile(
                r"if \(!init_kv_pressure_sampler\(\)\) \{\s*return false;",
                re.MULTILINE,
            ),
        )

    def test_result_struct_in_shared_header(self):
        """llama_kv_bounded_release_result must live in the shared release header."""
        self.assertIn("struct llama_kv_bounded_release_result", self.release_h)
        self.assertIn("released_bytes", self.release_h)
        self.assertIn("ownership_aborted", self.release_h)

    def test_memory_virtual_methods_declared(self):
        self.assertIn("virtual llama_kv_bounded_release_result bounded_release_dry_run", self.memory_h)
        self.assertIn("virtual llama_kv_release_status paged_release_status() const", self.memory_h)
        self.assertIn("virtual llama_kv_bounded_release_result bounded_release", self.memory_h)
        self.assertIn("virtual bool bounded_release_can_enable() const", self.memory_h)
        self.assertIn("virtual llama_kv_bounded_release_capability bounded_release_can_enable_diagnose() const", self.memory_h)
        self.assertIn("virtual uint64_t sample_kv_resident_bytes() const", self.memory_h)

    # --- bounded release checks ---

    def test_bounded_release_marker_fields(self):
        required = (
            "kv_pressure_bounded_release",
            "released_bytes=", "released_blocks=",
            "blocks_scanned=", "blocks_skipped_owned=", "blocks_skipped_state=",
            "madvise_failures=", "shortfall_bytes=", "overshoot_bytes=",
            "block_scan_exhausted=", "ownership_aborted=",
            "target_bytes=", "max_scan_blocks=",
            "legacy_enabled=", "sample_count=", "episode=",
            "cooldown_ms=", "skipped_reason=",
            "mincore_before_bytes=", "mincore_after_bytes=",
            "target_mode=", "kv_budget_valid=", "kv_budget_ownership_aborted=", "kv_resident_bytes=",
            "kv_reclaimable_resident_bytes=", "water_excess_bytes=",
            "water_shortfall_bytes=", "water_overshoot_bytes=",
            "max_release_bytes=", "target_clamp=", "decision_reason=",
            "bounded_cnt_bytes_delta=", "bounded_cnt_blocks_delta=",
            "can_enable=", "cap_paged=", "cap_ingraph=", "cap_layers=",
            "cap_row_idx=", "cap_swap_disabled=", "cap_layout=",
        )
        for field in required:
            self.assertIn(field, self.runtime)

    def test_bounded_release_allowed_tokens_in_maybe_sample(self):
        integration = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)")
        allowed = (
            "bounded_release(",
            "bounded_release_can_enable()",
            "bounded_release_due(",
            "bounded_release_counter_bytes()",
            "bounded_release_counter_blocks()",
        )
        for token in allowed:
            self.assertIn(token, integration)

    def test_bounded_release_config_default_off(self):
        self.assertIn("enabled         = false", self.runtime)

    def test_bounded_release_lifecycle_in_init(self):
        init_body = function_body(self.context, "bool init_kv_pressure_sampler()")
        self.assertIn("kv_pressure_runtime.bounded_release_disable()", init_body)
        self.assertIn("server_kv_pressure_bounded_release_config_from_env", init_body)

    def test_bounded_release_lifecycle_in_sleep_resume(self):
        lifecycle = function_body(self.context, "void handle_sleeping_state(bool new_state)")
        self.assertIn("kv_pressure_runtime.bounded_release_disable();", lifecycle)

    def test_bounded_release_structural_query_in_memory_h(self):
        self.assertIn("virtual bool bounded_release_can_enable() const", self.memory_h)

    def test_bounded_release_counter_fields_in_kv_cache_h(self):
        self.assertIn("paged_bounded_release_calls", self.core_h)
        self.assertIn("paged_bounded_release_blocks", self.core_h)
        self.assertIn("paged_bounded_release_bytes", self.core_h)
        self.assertIn("paged_bounded_release_unused", self.core_h)
        self.assertIn("paged_bounded_release_dead", self.core_h)

    def test_final_release_stats_expose_independent_bounded_source_counters(self):
        self.assertIn("bounded_release_calls=%llu", self.core)
        self.assertIn("bounded_release_bytes=%llu", self.core)
        self.assertIn("bounded_release_blocks=%llu", self.core)
        self.assertIn("paged_bounded_release_calls", self.core)
        self.assertIn("paged_bounded_release_bytes", self.core)
        self.assertIn("paged_bounded_release_blocks", self.core)

    def test_capability_struct_in_shared_header(self):
        """llama_kv_bounded_release_capability must be in the shared release header."""
        self.assertIn("struct llama_kv_bounded_release_capability", self.release_h)
        self.assertIn("bool can_enable", self.release_h)
        self.assertIn("bool paged", self.release_h)
        self.assertIn("bool ingraph", self.release_h)
        self.assertIn("bool layers_supported", self.release_h)
        self.assertIn("bool row_idx", self.release_h)
        self.assertIn("bool swap_disabled", self.release_h)
        self.assertIn("bool layout_supported", self.release_h)

    def test_capability_diagnose_used_in_context(self):
        """bounded_release_can_enable_diagnose() must be used in server-context.cpp."""
        self.assertIn("bounded_release_can_enable_diagnose()", self.context)
        # The per-condition checks must appear
        self.assertIn("not_ingraph", self.context)
        self.assertIn("no_layers", self.context)
        self.assertIn("no_row_idx", self.context)

    def test_capability_startup_marker(self):
        """Startup capability diagnostic must emit kv_pressure_bounded_release_capability."""
        init_body = function_body(self.context, "bool init_kv_pressure_sampler()")
        self.assertIn("kv_pressure_bounded_release_capability", init_body)
        self.assertIn("can_enable=", init_body)

    def test_bounded_release_no_legacy_mutation_integration(self):
        """Bounded release Phase C must never call legacy release functions."""
        integration = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)")
        # bounded_release uses the new server path, never calls legacy release
        self.assertNotIn("paged_release_blocks(", integration)
        self.assertNotIn("paged_block_release_enabled", integration)
        # bounded_release() is the new path (own counters, no test seams)
        self.assertIn("bounded_release(", integration)

    # --- telemetry: no_dummy reason distinction and dummy candidate types ---

    def test_kv_paged_release_stats_ensure_pending_write_rejected(self):
        """KV_PAGED_BOUNDED_RELEASE_STATS must include ensure_pending_write_rejected."""
        self.assertIn("KV_PAGED_BOUNDED_RELEASE_STATS", self.core)
        self.assertIn("ensure_pending_write_rejected=", self.core)
        self.assertIn("paged_block_ensure_pending_write_rejected", self.core_h)

    def test_kv_paged_release_stats_no_dummy_pending_write(self):
        """KV_PAGED_BOUNDED_RELEASE_STATS must distinguish RELEASED vs PENDING_WRITE no_dummy."""
        self.assertIn("KV_PAGED_BOUNDED_RELEASE_STATS", self.core)
        self.assertIn("released_redirect_no_dummy_pending_write=", self.core)
        self.assertIn("paged_released_redirect_no_dummy_pending_write", self.core_h)

    def test_kv_paged_release_stats_dummy_candidate_types(self):
        """KV_PAGED_BOUNDED_RELEASE_STATS must include dummy candidate type counts."""
        self.assertIn("KV_PAGED_BOUNDED_RELEASE_STATS", self.core)
        self.assertIn("dummy_candidate_resident=", self.core)
        self.assertIn("dummy_candidate_pending_write_cell=", self.core)
        self.assertIn("paged_dummy_candidate_resident", self.core_h)
        self.assertIn("paged_dummy_candidate_pending_write_cell", self.core_h)

    def test_kv_paged_release_stats_marker_closure(self):
        """Cleanup-C2: producer/parser marker closure.

        The current producer must emit ONLY the new bounded marker token; the
        retired `KV_PAGED_RELEASE_STATS` token must not appear in core (legacy
        artifacts continue to bind to their original parser identity).
        """
        # New marker must be the exact, single release-stats token in producer
        self.assertIn("KV_PAGED_BOUNDED_RELEASE_STATS contract=", self.core)
        # Retired marker must NOT appear anywhere in core — token-level closure
        self.assertNotIn("KV_PAGED_RELEASE_STATS", self.core)
        # Header must not declare a KV_PAGED_RELEASE_STATS log helper either
        self.assertNotIn("KV_PAGED_RELEASE_STATS", self.core_h)

    def test_pending_write_cell_dummy_in_row_idx(self):
        """set_input_paged_row_idx must scan PENDING_WRITE cells as dummy candidates."""
        row_idx_fn = function_body(self.core, "llama_kv_cache::set_input_paged_row_idx(ggml_tensor * dst")
        # Transaction-local dummy: scan of paged_pending_write_cells
        self.assertIn("paged_pending_write_cells", row_idx_fn)
        self.assertIn("paged_dummy_candidate_pending_write_cell", row_idx_fn)

    def test_ensure_write_resident_pending_write_rejected_counter(self):
        """paged_ensure_write_resident must increment the rejected counter on stale PENDING_WRITE."""
        ensure_fn = function_body(self.core, "llama_kv_cache::paged_ensure_write_resident(uint32_t phys_cell")
        self.assertIn("paged_block_ensure_pending_write_rejected", ensure_fn)
        self.assertIn("paged_pending_write_cells[phys_cell]", ensure_fn)

    def test_residency_predicates_are_state_driven_in_bounded_only_mode(self):
        """Read/write safety must not be bypassed when both legacy switches are off."""
        ensure_fn = function_body(self.core, "llama_kv_cache::paged_ensure_write_resident(uint32_t phys_cell")
        read_fn = function_body(self.core, "llama_kv_cache::paged_check_read_resident_impl(")
        for body in (ensure_fn, read_fn):
            self.assertNotIn("(!paged_block_release_enabled && !paged_swap_enabled)", body)
            self.assertNotIn("(!paged_swap_enabled && !paged_block_release_enabled)", body)
            self.assertIn("paged_block_states", body)

    def test_read_resident_distinguishes_fresh_and_stale_pending_write(self):
        """Active reads accept only current-transaction PENDING_WRITE cells."""
        read_fn = function_body(self.core, "llama_kv_cache::paged_check_read_resident_impl(")
        self.assertIn("paged_block_state::PENDING_WRITE", read_fn)
        self.assertIn("paged_pending_write_cells[phys_cell]", read_fn)
        self.assertIn("paged_block_ensure_pending_write_rejected", read_fn)
        self.assertIn("ACTIVE_READ_RELEASED_BLOCK", read_fn)
        self.assertIn("return false", read_fn)

    def test_dummy_comment_uses_graph_order_not_metadata_write_claim(self):
        """Dummy safety rationale must cite mask/address/graph order, not metadata writes."""
        row_idx_fn = function_body(self.core, "llama_kv_cache::set_input_paged_row_idx(ggml_tensor * dst")
        self.assertIn("masked /", row_idx_fn)
        self.assertIn("active-invisible", row_idx_fn)
        self.assertIn("SET_ROWS", row_idx_fn)
        self.assertIn("GET_ROWS", row_idx_fn)
        self.assertIn("does NOT write tensor data", row_idx_fn)
        self.assertNotIn("finish_write_transaction runs AFTER", row_idx_fn)


if __name__ == "__main__":
    unittest.main()
