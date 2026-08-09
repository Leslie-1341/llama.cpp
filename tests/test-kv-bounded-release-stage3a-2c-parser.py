#!/usr/bin/env python3
"""Single-mutation fixtures for the Stage 3A-2C protocol parser."""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-kv-bounded-release-stage3a-2c.py"
RUNNER = ROOT / "scripts" / "run-kv-bounded-release-stage3a-2c.py"


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ident(path: pathlib.Path) -> dict[str, object]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


class ParserArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.art = pathlib.Path(tempfile.mkdtemp(prefix="stage3a_2c_parser_"))
        self._write_valid_artifact()

    def tearDown(self) -> None:
        shutil.rmtree(self.art, ignore_errors=True)

    @staticmethod
    def telemetry() -> str:
        # Real server output produces combinations like first,state,source.
        # Single-value triggers (periodic, wake_completion) are also valid.
        return ("kv_pressure_telemetry state=CRITICAL previous_state=PRESSURE "
                "source=RSS_ABSOLUTE sample_valid=1 stale=0 config_valid=1 rss_kb=1000 "
                "cgroup_current_bytes=0 cgroup_max_bytes=0 cgroup_current_kb=0 "
                "cgroup_max_kb=0 cgroup_high_kb=0 psi_some_avg10=0 psi_full_avg10=0 "
                "pressure_basis_valid=1 pressure_current_bytes=1024000 "
                "pressure_low_water_bytes=1024 pressure_basis_generation=1 "
                "sample_latency_ns=1 sample_count=1 skip_count=0 idle=0 "
                "trigger=first,state,source\n")

    @staticmethod
    def dry_marker() -> str:
        return ("kv_pressure_dry_run state=CRITICAL source=RSS_ABSOLUTE stale=0 "
                "release_enabled=0 would_release_bytes=4096 would_release_blocks=1 "
                "blocks_scanned=1 blocks_skipped_owned=0 blocks_skipped_state=0 "
                "shortfall_bytes=0 overshoot_bytes=0 block_scan_exhausted=0 "
                "ownership_aborted=0 target_bytes=4096 max_scan_blocks=1 "
                "skipped_reason=none cooldown_ms=500 sample_count=1 idle=0\n")

    @staticmethod
    def bounded_marker() -> str:
        return ("kv_pressure_bounded_release state=CRITICAL source=RSS_ABSOLUTE stale=0 "
                "released_bytes=4096 released_blocks=1 blocks_scanned=1 "
                "blocks_skipped_owned=0 blocks_skipped_state=0 madvise_failures=0 "
                "shortfall_bytes=0 overshoot_bytes=0 block_scan_exhausted=0 "
                "ownership_aborted=0 target_mode=fixed pressure_basis_valid=1 "
                "pressure_current_bytes=1024000 pressure_low_water_bytes=1024 "
                "pressure_basis_generation=1 kv_budget_valid=0 kv_budget_ownership_aborted=0 "
                "kv_resident_bytes=0 kv_reclaimable_resident_bytes=0 water_excess_bytes=0 "
                "water_shortfall_bytes=0 water_overshoot_bytes=0 max_release_bytes=4096 "
                "target_clamp=none decision_reason=fixed target_bytes=4096 max_scan_blocks=1 legacy_enabled=0 "
                "sample_count=1 episode=1 cooldown_ms=60000 skipped_reason=none idle=0 "
                "mincore_before_bytes=8192 mincore_after_bytes=4096 "
                "bounded_cnt_bytes_delta=4096 bounded_cnt_blocks_delta=1 "
                "can_enable=1 cap_paged=1 cap_ingraph=1 cap_layers=1 cap_row_idx=1 "
                "cap_swap_disabled=1 cap_layout=1\n")

    @staticmethod
    def stats(*, bounded: bool = False) -> str:
        values = {
            "contract": "no_backing_unrecoverable_dead_or_unused_only",
            "bounded_release_calls": 1 if bounded else 0,
            "bounded_release_bytes": 4096 if bounded else 0,
            "bounded_release_blocks": 1 if bounded else 0,
            "reuse_allocations": 1 if bounded else 0,
            "write_commits": 1 if bounded else 0,
            "write_rollbacks": 0,
            "dummy_candidate_pending_write_cell": 1 if bounded else 0,
            "released_redirect_no_dummy": 0,
            "released_redirect_no_dummy_pending_write": 0,
            "ensure_pending_write_rejected": 0,
            "input_setup_fatal": 0,
            "row_mapping_fatal": 0,
            "write_mapping_fatal": 0,
            "active_nonresident_fatal": 0,
        }
        return "KV_PAGED_BOUNDED_RELEASE_STATS " + " ".join(f"{k}={v}" for k, v in values.items()) + "\n"

    @staticmethod
    def base_env() -> dict[str, str]:
        return {
            "HOME": "/tmp", "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1",
            "LLAMA_KV_PAGED_MINCORE": "1", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
            "LLAMA_KV_PAGED_RELEASE": "0", "LLAMA_KV_PAGED_SWAP": "0", "LLAMA_KV_SWAP": "0",
            "LLAMA_KV_PRESSURE_SAMPLER": "1",
        }

    def _write_valid_artifact(self) -> None:
        binary = self.art / "llama-server"
        model = self.art / "model.gguf"
        binary.write_bytes(b"binary")
        model.write_bytes(b"model")

        snapshot_dir = self.art / "source_snapshot"
        tracked = snapshot_dir / "tracked.diff"
        untracked = snapshot_dir / "untracked" / "scripts" / "fixture.py"
        untracked.parent.mkdir(parents=True)
        tracked.write_text("fixture tracked diff\n", encoding="utf-8")
        untracked.write_text("fixture untracked source\n", encoding="utf-8")

        common_argv = [str(binary), "--host", "127.0.0.1", "--port", "0", "--model", str(model),
                       "--ctx-size", "1024", "--n-gpu-layers", "0", "--threads", "4",
                       "--batch-size", "128", "--ubatch-size", "128", "--parallel", "1",
                       "--cache-ram", "0", "--cache-type-k", "f32", "--cache-type-v", "f32",
                       "--no-warmup"]
        case_configs: dict[str, object] = {}
        cases = (
            ("bounded_off", "OFF", self.telemetry() + self.stats(), {}),
            ("dry_run_off", "DRY", self.dry_marker() + self.stats(), {
                "LLAMA_KV_PRESSURE_DRY_RUN": "1",
                "LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES": "4096",
                "LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS": "1",
                "LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS": "500",
            }),
            ("bounded_on", "BOUNDED",
             "kv_pressure_bounded_release_capability can_enable=1 paged=1 ingraph=1 "
             "layers_supported=1 row_idx=1 swap_disabled=1 layout_supported=1\n" +
             self.bounded_marker() + self.stats(bounded=True), {
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": "4096",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": "1",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000",
            }),
        )
        for index, (directory, label, stderr, extra) in enumerate(cases):
            case = self.art / directory
            case.mkdir()
            env = self.base_env()
            env.update(extra)
            argv = list(common_argv)
            argv[argv.index("--port") + 1] = str(9000 + index)
            execution = {"argv": argv, "strace": False, "environment": env,
                         "cleared_inherited_prefixes": ["LLAMA_KV_", "LLAMA_", "GGML_", "GGUF_"]}
            (case / "environment.json").write_text(json.dumps(env), encoding="utf-8")
            (case / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
            (case / "server.stderr").write_text(stderr, encoding="utf-8")
            (case / "result.json").write_text(
                json.dumps({"response_text": "same", "http_status": 200}), encoding="utf-8")
            (case / "strace.log").write_text(
                "123 madvise(0x1000, 4096, MADV_DONTNEED) = 0\n" if label == "BOUNDED" else "",
                encoding="utf-8")
            (case / "phases.json").write_text(json.dumps({"shutdown": {
                "pgid": 1000 + index, "exit_code": 0, "pgid_check_complete": True,
                "cleanup_kill_attempted": False, "residual_process": False,
            }}), encoding="utf-8")
            case_configs[label] = {"environment": env, "execution": execution}

        manifest = {
            "protocol": "kv_bounded_release_stage3a_2c", "protocol_version": 4,
            "head_sha": "0" * 40, "worktree_dirty": True,
            "worktree_status": [" M fixture", "?? scripts/fixture.py"],
            "capture_mode": "diagnostic_dirty",
            "source_snapshot": {
                "schema_version": 1, "head_sha": "0" * 40,
                "tracked_diff": ident(tracked),
                "untracked_files": [{"status": "??", "path": "scripts/fixture.py",
                                     "snapshot_path": str(untracked), "size": untracked.stat().st_size,
                                     "sha256": digest(untracked), "mode": untracked.stat().st_mode & 0o777}],
            },
            "diff": ident(tracked), "diff_sha256": digest(tracked),
            "binary": ident(binary), "model": ident(model), "runner": ident(RUNNER),
            "parser": ident(PARSER), "prompt": "fixture", "seed": 1, "target_bytes": 4096,
            "timestamp_utc": "2026-07-22T00:00:00Z",
            "completed_at_utc": "2026-07-22T00:00:01Z", "case_configs": case_configs,
        }
        (self.art / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (self.art / "summary.json").write_text(json.dumps({
            "runner_status": "run_complete", "case_failures": [], "response_identity": True,
        }), encoding="utf-8")

    def run_parser(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(PARSER), str(self.art)],
                              text=True, capture_output=True, check=False)

    def assert_failure(self, reason: str) -> None:
        result = self.run_parser()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(reason, result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def mutate_json(self, relative: str, mutate) -> None:
        path = self.art / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_valid_baseline_passes(self) -> None:
        result = self.run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_retired_release_stats_marker_fails_closed(self) -> None:
        path = self.art / "bounded_off" / "server.stderr"
        path.write_text(path.read_text().replace(
            "KV_PAGED_BOUNDED_RELEASE_STATS", "KV_PAGED_RELEASE_STATS"))
        self.assert_failure("OFF: KV_PAGED_RELEASE_STATS is retired")

    def test_off_bounded_source_nonzero_fails_for_that_counter(self) -> None:
        path = self.art / "bounded_off" / "server.stderr"
        path.write_text(path.read_text().replace("bounded_release_calls=0", "bounded_release_calls=1"))
        self.assert_failure("OFF: final bounded source bounded_release_calls=1, must be 0")

    def test_dry_bounded_source_nonzero_fails_for_that_counter(self) -> None:
        path = self.art / "dry_run_off" / "server.stderr"
        path.write_text(path.read_text().replace("bounded_release_bytes=0", "bounded_release_bytes=4096"))
        self.assert_failure("DRY: final bounded source bounded_release_bytes=4096, must be 0")

    def test_bounded_final_source_delta_mismatch_fails_for_that_counter(self) -> None:
        path = self.art / "bounded_on" / "server.stderr"
        path.write_text(path.read_text().replace("bounded_release_bytes=4096", "bounded_release_bytes=8192"))
        self.assert_failure("final bounded_release_bytes=8192 != bounded marker source delta total=4096")

    def test_bounded_mincore_observational_allows_drop_different_from_counter(self) -> None:
        """v4: mincore drop can differ from counter delta (observational, not exact)."""
        path = self.art / "bounded_on" / "server.stderr"
        # mincore_after=2048 → drop=6144 vs cnt_delta=4096 — valid observational
        path.write_text(path.read_text().replace("mincore_after_bytes=4096", "mincore_after_bytes=2048"))
        result = self.run_parser()
        self.assertEqual(result.returncode, 0,
                         f"observational mincore should pass; got: {result.stderr}")

    def test_missing_phases_fails_for_phases(self) -> None:
        (self.art / "dry_run_off" / "phases.json").unlink()
        self.assert_failure("DRY: phases.json missing")

    def test_incomplete_pgid_check_fails_for_pgid(self) -> None:
        self.mutate_json("bounded_on/phases.json",
                         lambda value: value["shutdown"].update(pgid_check_complete=False))
        self.assert_failure("BOUNDED: shutdown PGID residual check incomplete")

    def test_incomplete_shutdown_schema_fails_for_missing_field(self) -> None:
        self.mutate_json("bounded_on/phases.json",
                         lambda value: value["shutdown"].pop("pgid"))
        self.assert_failure("BOUNDED: shutdown record missing fields: ['pgid']")

    def test_residual_process_fails_for_residual(self) -> None:
        self.mutate_json("bounded_off/phases.json",
                         lambda value: value["shutdown"].update(residual_process=True))
        self.assert_failure("OFF: residual process after shutdown")

    def test_untracked_snapshot_content_tamper_fails_for_snapshot(self) -> None:
        (self.art / "source_snapshot/untracked/scripts/fixture.py").write_text("tampered")
        self.assert_failure("source_snapshot content identity drift: scripts/fixture.py")

    def test_tracked_diff_tamper_fails_for_snapshot(self) -> None:
        (self.art / "source_snapshot/tracked.diff").write_text("tampered")
        self.assert_failure("source_snapshot tracked_diff identity drift")

    def test_untracked_snapshot_omission_fails_for_status_closure(self) -> None:
        self.mutate_json("manifest.json",
                         lambda value: value["source_snapshot"].update(untracked_files=[]))
        self.assert_failure("worktree_status untracked paths differ from source_snapshot")

    def test_execution_environment_drift_fails_for_environment_closure(self) -> None:
        self.mutate_json("dry_run_off/execution.json",
                         lambda value: value["environment"].update(LLAMA_KV_PAGED="0"))
        self.assert_failure("DRY: execution.json environment differs from environment.json")

    def test_manifest_environment_drift_fails_for_manifest_closure(self) -> None:
        self.mutate_json("manifest.json",
                         lambda value: value["case_configs"]["OFF"]["environment"].update(LLAMA_KV_PAGED="0"))
        self.assert_failure("manifest environment differs from recorded OFF environment")

    def test_manifest_argv_drift_fails_for_execution_closure(self) -> None:
        self.mutate_json("manifest.json",
                         lambda value: value["case_configs"]["BOUNDED"]["execution"]["argv"].append("--bad"))
        self.assert_failure("manifest execution differs from recorded BOUNDED execution")

    def test_cross_case_argv_drift_fails_for_argv_identity(self) -> None:
        execution_path = self.art / "dry_run_off/execution.json"
        execution = json.loads(execution_path.read_text())
        execution["argv"][execution["argv"].index("--ctx-size") + 1] = "2048"
        execution_path.write_text(json.dumps(execution))
        self.mutate_json("manifest.json",
                         lambda value: value["case_configs"]["DRY"].update(execution=execution))
        self.assert_failure("OFF/DRY/BOUNDED argv must be identical")

    def test_archival_clean_still_rejects_dirty_identity(self) -> None:
        self.mutate_json("manifest.json",
                         lambda value: value.update(capture_mode="archival_clean"))
        self.assert_failure("archival_clean artifact must record a clean worktree")

    def test_marker_single_field_schema_error_is_explicit(self) -> None:
        path = self.art / "bounded_on/server.stderr"
        path.write_text(path.read_text().replace(" cap_layout=1", "", 1))
        self.assert_failure("schema mismatch missing=['cap_layout']")

    # --- v4 observational protocol tests ---

    def test_mincore_observational_drop_allowed_to_differ_from_counter(self) -> None:
        """mincore drop != counter delta is valid (observational, not exact match)."""
        path = self.art / "bounded_on/server.stderr"
        # Change mincore_after to produce a valid-but-different drop
        path.write_text(path.read_text().replace("mincore_after_bytes=4096",
                                                 "mincore_after_bytes=2048"))
        result = self.run_parser()
        self.assertEqual(result.returncode, 0,
                         f"mincore observational drop should pass; got: {result.stderr}")

    def test_mincore_no_drop_fails_observational_gate(self) -> None:
        """mincore with after >= before fails the observational direction check."""
        path = self.art / "bounded_on/server.stderr"
        path.write_text(path.read_text().replace("mincore_after_bytes=4096",
                                                 "mincore_after_bytes=16384"))
        self.assert_failure("mincore must show physical drop")

    def test_mincore_zero_samples_fails_observational_gate(self) -> None:
        """mincore with zero before or after fails the data-presence check."""
        path = self.art / "bounded_on/server.stderr"
        path.write_text(path.read_text().replace("mincore_before_bytes=8192",
                                                 "mincore_before_bytes=0"))
        self.assert_failure("mincore_before_bytes=0 must be >0")

    def test_mincore_implausibly_large_drop_fails_observational_gate(self) -> None:
        """mincore drop far exceeding counter delta + page margin fails."""
        path = self.art / "bounded_on/server.stderr"
        # Set mincore drop = 1000000 while counter delta = 4096 (1 block)
        path.write_text(path.read_text().replace("mincore_before_bytes=8192",
                                                 "mincore_before_bytes=1004096"))
        path.write_text(path.read_text().replace("mincore_after_bytes=4096",
                                                 "mincore_after_bytes=4096"))
        self.assert_failure("mincore drop")

    def test_ownership_aborted_marker_is_rejected(self) -> None:
        """Marker with ownership_aborted=1 must not be counted as a valid execution."""
        path = self.art / "bounded_on/server.stderr"
        path.write_text(path.read_text().replace("ownership_aborted=0",
                                                 "ownership_aborted=1"))
        self.assert_failure("ownership_aborted=1, must be 0")

    def test_quarantine_skip_reason_does_not_produce_false_execution(self) -> None:
        """Marker with skipped_reason=not_paged means no valid execution exists."""
        path = self.art / "bounded_on/server.stderr"
        # Replace the valid bounded marker with a skipped one that has
        # released_bytes=0 and skipped_reason=not_paged.
        lines = path.read_text().split("\n")
        new_lines = []
        for line in lines:
            if line.startswith("kv_pressure_bounded_release "):
                # Replace with a skipped variant (no released bytes, not_paged)
                line = (
                    "kv_pressure_bounded_release state=PRESSURE source=RSS_ABSOLUTE stale=0 "
                    "released_bytes=0 released_blocks=0 blocks_scanned=0 "
                    "blocks_skipped_owned=0 blocks_skipped_state=0 madvise_failures=0 "
                    "shortfall_bytes=4096 overshoot_bytes=0 block_scan_exhausted=1 "
                    "ownership_aborted=0 target_mode=fixed pressure_basis_valid=1 "
                    "pressure_current_bytes=1024000 pressure_low_water_bytes=1024 "
                    "pressure_basis_generation=1 kv_budget_valid=0 kv_budget_ownership_aborted=0 "
                    "kv_resident_bytes=0 kv_reclaimable_resident_bytes=0 water_excess_bytes=0 "
                    "water_shortfall_bytes=0 water_overshoot_bytes=0 max_release_bytes=4096 "
                    "target_clamp=none decision_reason=fixed target_bytes=4096 max_scan_blocks=1 legacy_enabled=0 "
                    "sample_count=1 episode=0 cooldown_ms=60000 skipped_reason=not_paged idle=0 "
                    "mincore_before_bytes=0 mincore_after_bytes=0 "
                    "bounded_cnt_bytes_delta=0 bounded_cnt_blocks_delta=0 "
                    "can_enable=0 cap_paged=0 cap_ingraph=0 cap_layers=0 cap_row_idx=0 "
                    "cap_swap_disabled=0 cap_layout=0")
            new_lines.append(line)
        path.write_text("\n".join(new_lines))
        self.assert_failure("No valid bounded release execution found")

    def test_skip_reason_unsupported_is_valid_not_error(self) -> None:
        """skip_reason=no_row_idx before valid execution is tolerated (pre-exec skip)."""
        path = self.art / "bounded_on/server.stderr"
        # Add a pre-exec skip marker with skip_reason=no_row_idx and
        # cap_row_idx=0.  This tests the row_idx lifecycle tracking.
        # All capability bool fields must be 0/1, not -1.
        extra = (
            "kv_pressure_bounded_release state=PRESSURE source=RSS_ABSOLUTE stale=0 "
            "released_bytes=0 released_blocks=0 blocks_scanned=0 "
            "blocks_skipped_owned=0 blocks_skipped_state=0 madvise_failures=0 "
            "shortfall_bytes=4096 overshoot_bytes=0 block_scan_exhausted=1 "
            "ownership_aborted=0 target_mode=fixed pressure_basis_valid=1 "
            "pressure_current_bytes=1024000 pressure_low_water_bytes=1024 "
            "pressure_basis_generation=1 kv_budget_valid=0 kv_budget_ownership_aborted=0 "
            "kv_resident_bytes=0 kv_reclaimable_resident_bytes=0 water_excess_bytes=0 "
            "water_shortfall_bytes=0 water_overshoot_bytes=0 max_release_bytes=4096 "
            "target_clamp=none decision_reason=fixed target_bytes=4096 max_scan_blocks=1 legacy_enabled=0 "
            "sample_count=1 episode=0 cooldown_ms=60000 skipped_reason=no_row_idx idle=0 "
            "mincore_before_bytes=0 mincore_after_bytes=0 "
            "bounded_cnt_bytes_delta=0 bounded_cnt_blocks_delta=0 "
            "can_enable=0 cap_paged=0 cap_ingraph=0 cap_layers=0 cap_row_idx=0 "
            "cap_swap_disabled=0 cap_layout=0\n")
        # bounded_marker has sample_count=1; give extra marker sample_count=1
        # and change bounded to sample_count=2 so monotonicity holds.
        bounded_line = self.bounded_marker().replace("sample_count=1", "sample_count=2")
        path.write_text(
            "kv_pressure_bounded_release_capability can_enable=1 paged=1 ingraph=1 "
            "layers_supported=1 row_idx=1 swap_disabled=1 layout_supported=1\n" +
            extra + bounded_line +
            self.stats(bounded=True))
        result = self.run_parser()
        # The no_row_idx skip before execution is expected; the valid execution
        # after it must still succeed.
        self.assertEqual(result.returncode, 0,
                         f"pre-exec no_row_idx is valid; got: {result.stderr}")

    def test_fake_pass_runner_verdict_is_rejected(self) -> None:
        """Runner writing verdict=PASS into summary.json is a protocol violation."""
        self.mutate_json("summary.json",
                         lambda value: value.update(verdict="PASS"))
        self.assert_failure("summary.json contains verdict=PASS")

    def test_case_order_preserved_in_manifest_configs(self) -> None:
        """Manifest case_configs must be present and cover OFF/DRY/BOUNDED."""
        self.mutate_json("manifest.json",
                         lambda value: value.update(case_configs={}))
        self.assert_failure("manifest missing case config for OFF")

    def test_marker_duplicate_field_detected(self) -> None:
        """Duplicate field in marker line must be rejected."""
        path = self.art / "bounded_on/server.stderr"
        # Find the bounded marker line and append a duplicate field
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("kv_pressure_bounded_release "):
                # Append a duplicate released_bytes=0 at the end
                lines[i] = line.rstrip() + " released_bytes=0"
                break
        path.write_text("\n".join(lines))
        self.assert_failure("duplicates field")

    # --- trigger field regression (Stage 3A-2C v4 parser fix) ---

    def test_trigger_periodic_single_value_valid(self) -> None:
        """trigger=periodic (single value) must be accepted."""
        path = self.art / "bounded_off/server.stderr"
        path.write_text(path.read_text().replace(
            "trigger=first,state,source", "trigger=periodic"))
        result = self.run_parser()
        self.assertEqual(result.returncode, 0,
                         f"periodic single trigger should pass; got: {result.stderr}")

    def test_trigger_first_single_value_valid(self) -> None:
        """trigger=first (single value) must be accepted."""
        path = self.art / "bounded_off/server.stderr"
        path.write_text(path.read_text().replace(
            "trigger=first,state,source", "trigger=first"))
        result = self.run_parser()
        self.assertEqual(result.returncode, 0,
                         f"first single trigger should pass; got: {result.stderr}")

    def test_trigger_wake_completion_single_value_valid(self) -> None:
        """trigger=wake_completion (single value) must be accepted."""
        path = self.art / "bounded_off/server.stderr"
        path.write_text(path.read_text().replace(
            "trigger=first,state,source", "trigger=wake_completion"))
        result = self.run_parser()
        self.assertEqual(result.returncode, 0,
                         f"wake_completion trigger should pass; got: {result.stderr}")

    def test_trigger_unknown_value_fail_closed(self) -> None:
        """trigger containing an unknown token must be rejected."""
        path = self.art / "bounded_off/server.stderr"
        path.write_text(path.read_text().replace(
            "trigger=first,state,source", "trigger=first,bogus"))
        self.assert_failure("trigger has unknown value 'bogus'")

    def test_trigger_empty_component_fail_closed(self) -> None:
        """trigger with an empty component (leading/trailing/double comma) must be rejected."""
        path = self.art / "bounded_off/server.stderr"
        path.write_text(path.read_text().replace(
            "trigger=first,state,source", "trigger=first,,state"))
        self.assert_failure("trigger has empty component")

    def test_trigger_duplicate_value_fail_closed(self) -> None:
        """trigger with duplicate entries must be rejected."""
        path = self.art / "bounded_off/server.stderr"
        path.write_text(path.read_text().replace(
            "trigger=first,state,source", "trigger=first,first"))
        self.assert_failure("trigger has duplicate values")

    def test_trigger_empty_fail_closed(self) -> None:
        """trigger with empty value must be rejected (caught as malformed field)."""
        path = self.art / "bounded_off/server.stderr"
        path.write_text(path.read_text().replace(
            "trigger=first,state,source", "trigger="))
        self.assert_failure("malformed field 'trigger='")


    # --- Stage 3B-1 dynamic target env isolation (review-fix) ---

    _DYNAMIC_ENV = {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1"}

    def _add_dynamic_target_to_case(self, case_dir: str, label: str) -> None:
        """Add DYNAMIC_TARGET=1 to all env records for a case (environment.json,
        execution.json, and both manifest locations)."""
        self.mutate_json(f"{case_dir}/environment.json",
                         lambda e: e.update(self._DYNAMIC_ENV))
        self.mutate_json(f"{case_dir}/execution.json",
                         lambda v: v["environment"].update(self._DYNAMIC_ENV))
        # manifest.json stores the environment in two places:
        #   case_configs[label].environment (top-level env identity)
        #   case_configs[label].execution.environment (nested inside execution identity)
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"][label]["environment"].update(
                             self._DYNAMIC_ENV))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"][label]["execution"]["environment"].update(
                             self._DYNAMIC_ENV))

    def test_bounded_dynamic_target_accepted(self) -> None:
        """BOUNDED variant with DYNAMIC_TARGET=1 must pass env isolation."""
        self._add_dynamic_target_to_case("bounded_on", "BOUNDED")
        result = self.run_parser()
        self.assertEqual(result.returncode, 0,
                         f"BOUNDED+DYNAMIC_TARGET should pass; got: {result.stderr}")

    def test_off_dynamic_target_rejected(self) -> None:
        """OFF variant with DYNAMIC_TARGET=1 must be rejected as bounded key leakage."""
        self._add_dynamic_target_to_case("bounded_off", "OFF")
        self.assert_failure("OFF env has bounded-release keys")

    def test_dry_dynamic_target_rejected(self) -> None:
        """DRY variant with DYNAMIC_TARGET=1 must be rejected as bounded key leakage."""
        self._add_dynamic_target_to_case("dry_run_off", "DRY")
        self.assert_failure("DRY env has")


if __name__ == "__main__":
    unittest.main()
