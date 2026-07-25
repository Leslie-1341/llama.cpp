#!/usr/bin/env python3
"""Single-mutation fixtures for Stage 3B-1 v2 protocol parser (calibrated + reuse)."""

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
PARSER = ROOT / "scripts" / "parse-kv-bounded-release-stage3b-1.py"
RUNNER = ROOT / "scripts" / "run-kv-bounded-release-stage3b-1.py"


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ident(path: pathlib.Path) -> dict[str, object]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


# ── Calibration fixture values ────────────────────────────────────────────────

CALIB = {
    "schema_version": 1,
    "rss_idle_kb": 6000000,
    "rss_peak_kb": 6200000,
    "delta_kb": 200000,
    "margin_kb": 100000,
    "pressure_trigger_kb": 6050000,
    "critical_trigger_kb": 6150000,
    "pressure_safe_kb": 6300000,
    "critical_safe_kb": 6300001,
    "low_water_trigger_kb": 6000000,
    "low_water_safe_kb": 6300000,
    "calibration_response_text": "same",
}


class Stage3B1ParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.art = pathlib.Path(tempfile.mkdtemp(prefix="stage3b_1_v2_parser_"))
        self._write_valid_artifact()

    def tearDown(self) -> None:
        shutil.rmtree(self.art, ignore_errors=True)

    # ── marker builders ───────────────────────────────────────────────────────

    @staticmethod
    def telemetry(state: str = "CRITICAL") -> str:
        return (f"kv_pressure_telemetry state={state} previous_state=PRESSURE "
                "source=RSS_ABSOLUTE sample_valid=1 stale=0 config_valid=1 rss_kb=6100000 "
                "cgroup_current_bytes=0 cgroup_max_bytes=0 cgroup_current_kb=0 "
                "cgroup_max_kb=0 cgroup_high_kb=0 psi_some_avg10=0 psi_full_avg10=0 "
                "pressure_basis_valid=1 pressure_current_bytes=6246400000 "
                "pressure_low_water_bytes=6144000000 pressure_basis_generation=1 "
                "sample_latency_ns=1 sample_count=1 skip_count=0 idle=0 "
                "trigger=first,state,source\n")

    @staticmethod
    def telemetry_normal() -> str:
        return Stage3B1ParserTest.telemetry(state="NORMAL")

    @staticmethod
    def bounded_marker_fixed() -> str:
        return ("kv_pressure_bounded_release state=CRITICAL source=RSS_ABSOLUTE stale=0 "
                "released_bytes=4096 released_blocks=1 blocks_scanned=1 "
                "blocks_skipped_owned=0 blocks_skipped_state=0 madvise_failures=0 "
                "shortfall_bytes=0 overshoot_bytes=0 block_scan_exhausted=0 "
                "ownership_aborted=0 target_mode=fixed pressure_basis_valid=1 "
                "pressure_current_bytes=1024000 pressure_low_water_bytes=1024 "
                "pressure_basis_generation=1 kv_budget_valid=0 kv_budget_ownership_aborted=0 "
                "kv_resident_bytes=0 kv_reclaimable_resident_bytes=0 water_excess_bytes=0 "
                "water_shortfall_bytes=0 water_overshoot_bytes=0 max_release_bytes=33554432 "
                "target_clamp=none decision_reason=fixed target_bytes=33554432 "
                "max_scan_blocks=64 legacy_enabled=0 "
                "sample_count=1 episode=1 cooldown_ms=60000 skipped_reason=none idle=0 "
                "mincore_before_bytes=8192 mincore_after_bytes=4096 "
                "bounded_cnt_bytes_delta=4096 bounded_cnt_blocks_delta=1 "
                "can_enable=1 cap_paged=1 cap_ingraph=1 cap_layers=1 cap_row_idx=1 "
                "cap_swap_disabled=1 cap_layout=1\n")

    @staticmethod
    def bounded_marker_dynamic_release() -> str:
        return ("kv_pressure_bounded_release state=CRITICAL source=RSS_ABSOLUTE stale=0 "
                "released_bytes=16384 released_blocks=4 blocks_scanned=4 "
                "blocks_skipped_owned=0 blocks_skipped_state=0 madvise_failures=0 "
                "shortfall_bytes=0 overshoot_bytes=0 block_scan_exhausted=0 "
                "ownership_aborted=0 target_mode=dynamic pressure_basis_valid=1 "
                "pressure_current_bytes=1024000 pressure_low_water_bytes=1024 "
                "pressure_basis_generation=1 kv_budget_valid=1 kv_budget_ownership_aborted=0 "
                "kv_resident_bytes=65536 kv_reclaimable_resident_bytes=16384 "
                "water_excess_bytes=1022976 water_shortfall_bytes=0 water_overshoot_bytes=0 "
                "max_release_bytes=1073741824 target_clamp=reclaimable "
                "decision_reason=dynamic target_bytes=16384 "
                "max_scan_blocks=64 legacy_enabled=0 "
                "sample_count=1 episode=1 cooldown_ms=60000 skipped_reason=none idle=0 "
                "mincore_before_bytes=81920 mincore_after_bytes=65536 "
                "bounded_cnt_bytes_delta=16384 bounded_cnt_blocks_delta=4 "
                "can_enable=1 cap_paged=1 cap_ingraph=1 cap_layers=1 cap_row_idx=1 "
                "cap_swap_disabled=1 cap_layout=1\n")

    @staticmethod
    def stats(*, calls: int = 0, bytes_val: int = 0, blocks: int = 0,
              reuse: int = 0, commit: int = 0, dummy: int = 0) -> str:
        values = {
            "contract": "no_backing_unrecoverable_dead_or_unused_only",
            "bounded_release_calls": calls,
            "bounded_release_bytes": bytes_val,
            "bounded_release_blocks": blocks,
            "reuse_allocations": reuse,
            "write_commits": commit,
            "write_rollbacks": 0,
            "dummy_candidate_pending_write_cell": dummy,
            "released_redirect_no_dummy": 0,
            "released_redirect_no_dummy_pending_write": 0,
            "ensure_pending_write_rejected": 0,
            "input_setup_fatal": 0,
            "row_mapping_fatal": 0,
            "write_mapping_fatal": 0,
            "active_nonresident_fatal": 0,
        }
        return "KV_PAGED_RELEASE_STATS " + " ".join(f"{k}={v}" for k, v in values.items()) + "\n"

    @staticmethod
    def base_env() -> dict[str, str]:
        return {
            "HOME": "/tmp", "PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
            "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1",
            "LLAMA_KV_PAGED_MINCORE": "1", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
            "LLAMA_KV_PAGED_RELEASE": "0", "LLAMA_KV_PAGED_SWAP": "0", "LLAMA_KV_SWAP": "0",
            "LLAMA_KV_PRESSURE_SAMPLER": "1",
            "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "250",
            "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
        }

    # ── artifact builder ──────────────────────────────────────────────────────

    def _write_valid_artifact(self) -> None:
        binary = self.art / "llama-server"
        model = self.art / "model.gguf"
        binary.write_bytes(b"binary")
        model.write_bytes(b"model")

        # Calibration
        calib_dir = self.art / "calibration"
        calib_dir.mkdir()
        (calib_dir / "calibration.json").write_text(json.dumps(CALIB), encoding="utf-8")
        (calib_dir / "phases.json").write_text(json.dumps({"shutdown": {
            "pgid": 9000, "exit_code": 0, "pgid_check_complete": True,
            "cleanup_kill_attempted": False, "residual_process": False,
        }}), encoding="utf-8")

        # Source snapshot
        snapshot_dir = self.art / "source_snapshot"
        tracked = snapshot_dir / "tracked.diff"
        untracked_file = snapshot_dir / "untracked" / "scripts" / "fixture.py"
        untracked_file.parent.mkdir(parents=True)
        tracked.write_text("fixture tracked diff\n", encoding="utf-8")
        untracked_file.write_text("fixture untracked source\n", encoding="utf-8")

        common_argv = [str(binary), "--host", "127.0.0.1", "--port", "0", "--model", str(model),
                       "--ctx-size", "1024", "--n-gpu-layers", "0", "--threads", "4",
                       "--batch-size", "128", "--ubatch-size", "128", "--parallel", "1",
                       "--cache-ram", "0", "--cache-type-k", "f32", "--cache-type-v", "f32",
                       "--no-warmup"]

        cap_marker = ("kv_pressure_bounded_release_capability can_enable=1 paged=1 ingraph=1 "
                      "layers_supported=1 row_idx=1 swap_disabled=1 layout_supported=1\n")

        # Per-case defs
        cases_def = (
            # OFF: safe thresholds, no bounded release
            ("bounded_off", "OFF",
             self.telemetry() + self.stats(),
             {"LLAMA_KV_PRESSURE_RSS_KB": str(CALIB["pressure_safe_kb"]),
              "LLAMA_KV_CRITICAL_RSS_KB": str(CALIB["critical_safe_kb"]),
              "LLAMA_KV_LOW_WATER_RSS_KB": str(CALIB["low_water_safe_kb"])},
             1, False),

            # FIXED: trigger thresholds, fixed target, two requests
            ("bounded_fixed", "FIXED",
             cap_marker + self.bounded_marker_fixed() + self.stats(
                 calls=1, bytes_val=4096, blocks=1, reuse=2, commit=2, dummy=1),
             {"LLAMA_KV_PRESSURE_RSS_KB": str(CALIB["pressure_trigger_kb"]),
              "LLAMA_KV_CRITICAL_RSS_KB": str(CALIB["critical_trigger_kb"]),
              "LLAMA_KV_LOW_WATER_RSS_KB": str(CALIB["low_water_trigger_kb"]),
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": "33554432",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": "64",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000"},
             2, True),

            # DYNAMIC_NOOP: safe thresholds, bounded+dyn enabled, NORMAL state
            ("bounded_dynamic_noop", "DYNAMIC_NOOP",
             cap_marker + self.telemetry_normal() + self.stats(),
             {"LLAMA_KV_PRESSURE_RSS_KB": str(CALIB["pressure_safe_kb"]),
              "LLAMA_KV_CRITICAL_RSS_KB": str(CALIB["critical_safe_kb"]),
              "LLAMA_KV_LOW_WATER_RSS_KB": str(CALIB["low_water_safe_kb"]),
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": "1073741824",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": "64",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000"},
             1, False),

            # DYNAMIC_RELEASE: trigger thresholds, dynamic target, two requests
            ("bounded_dynamic_release", "DYNAMIC_RELEASE",
             cap_marker + self.bounded_marker_dynamic_release() + self.stats(
                 calls=1, bytes_val=16384, blocks=4, reuse=2, commit=2, dummy=1),
             {"LLAMA_KV_PRESSURE_RSS_KB": str(CALIB["pressure_trigger_kb"]),
              "LLAMA_KV_CRITICAL_RSS_KB": str(CALIB["critical_trigger_kb"]),
              "LLAMA_KV_LOW_WATER_RSS_KB": str(CALIB["low_water_trigger_kb"]),
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": "1073741824",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": "64",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
              "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000"},
             2, True),
        )

        case_configs: dict[str, object] = {}
        for index, (directory, label, stderr, extra_env, n_req, has_release) in enumerate(cases_def):
            case = self.art / directory
            case.mkdir()
            env = self.base_env()
            env.update(extra_env)
            argv = list(common_argv)
            argv[argv.index("--port") + 1] = str(9000 + index)
            execution = {"argv": argv, "strace": False, "environment": env,
                         "cleared_inherited_prefixes": ["LLAMA_KV_", "LLAMA_", "GGML_", "GGUF_"]}
            (case / "environment.json").write_text(json.dumps(env), encoding="utf-8")
            (case / "execution.json").write_text(json.dumps(execution), encoding="utf-8")
            (case / "server.stderr").write_text(stderr, encoding="utf-8")

            # Result: single or double request
            if n_req == 2:
                result = {"requests_count": 2,
                          "http_status_1": 200, "response_text_1": "same",
                          "http_status_2": 200, "response_text_2": "same"}
            else:
                result = {"requests_count": 1,
                          "http_status_1": 200, "response_text_1": "same"}
            (case / "result.json").write_text(json.dumps(result), encoding="utf-8")

            # Strace
            strace_bytes = 0
            if has_release:
                if label == "FIXED":
                    strace_bytes = 4096
                else:
                    strace_bytes = 16384
            if strace_bytes:
                (case / "strace.log").write_text(
                    f"123 madvise(0x1000, {strace_bytes}, MADV_DONTNEED) = 0\n",
                    encoding="utf-8")
            else:
                (case / "strace.log").write_text("", encoding="utf-8")

            (case / "phases.json").write_text(json.dumps({"shutdown": {
                "pgid": 1000 + index, "exit_code": 0, "pgid_check_complete": True,
                "cleanup_kill_attempted": False, "residual_process": False,
            }}), encoding="utf-8")
            case_configs[label] = {"environment": env, "execution": execution}

        manifest = {
            "protocol": "kv_bounded_release_stage3b_1", "protocol_version": 2,
            "head_sha": "0" * 40, "worktree_dirty": True,
            "worktree_status": [" M fixture", "?? scripts/fixture.py"],
            "capture_mode": "diagnostic_dirty",
            "calibration": CALIB,
            "source_snapshot": {
                "schema_version": 1, "head_sha": "0" * 40,
                "tracked_diff": ident(tracked),
                "untracked_files": [{"status": "??", "path": "scripts/fixture.py",
                                     "snapshot_path": str(untracked_file),
                                     "size": untracked_file.stat().st_size,
                                     "sha256": digest(untracked_file),
                                     "mode": untracked_file.stat().st_mode & 0o777}],
            },
            "diff": ident(tracked), "diff_sha256": digest(tracked),
            "binary": ident(binary), "model": ident(model), "runner": ident(RUNNER),
            "parser": ident(PARSER), "prompt": "fixture", "seed": 1,
            "timestamp_utc": "2026-07-25T00:00:00Z",
            "completed_at_utc": "2026-07-25T00:00:01Z", "case_configs": case_configs,
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
        self.assertNotEqual(result.returncode, 0,
                            f"expected failure; got exit 0\nstdout: {result.stdout}")
        self.assertIn(reason, result.stderr,
                      f"expected '{reason}' in stderr\nstderr: {result.stderr}")
        self.assertNotIn("Traceback", result.stderr)

    def mutate_json(self, relative: str, mutate) -> None:
        path = self.art / relative
        value = json.loads(path.read_text(encoding="utf-8"))
        mutate(value)
        path.write_text(json.dumps(value), encoding="utf-8")

    # ═══════════════════════════════════════════════════════════════════════
    # Baseline
    # ═══════════════════════════════════════════════════════════════════════

    def test_valid_baseline_passes(self) -> None:
        result = self.run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)

    # ═══════════════════════════════════════════════════════════════════════
    # Calibration
    # ═══════════════════════════════════════════════════════════════════════

    def test_calibration_missing_rejected(self) -> None:
        self.mutate_json("manifest.json", lambda v: v.pop("calibration", None))
        self.assert_failure("manifest missing calibration")

    def test_calibration_hardcoded_threshold_rejected(self) -> None:
        self.mutate_json("manifest.json",
                         lambda v: v["calibration"].update(pressure_trigger_kb=2))
        self.assert_failure("hardcoded artificial threshold")

    def test_calibration_trigger_below_idle_rejected(self) -> None:
        self.mutate_json("manifest.json",
                         lambda v: v["calibration"].update(
                             pressure_trigger_kb=5000000))  # < rss_idle_kb=6000000
        self.assert_failure("pressure_trigger_kb")

    def test_calibration_trigger_above_safe_rejected(self) -> None:
        self.mutate_json("manifest.json",
                         lambda v: v["calibration"].update(
                             pressure_trigger_kb=6400000))  # > pressure_safe_kb=6300000
        self.assert_failure("pressure_trigger must be < pressure_safe")

    # ═══════════════════════════════════════════════════════════════════════
    # OFF case
    # ═══════════════════════════════════════════════════════════════════════

    def test_off_bounded_marker_rejected(self) -> None:
        path = self.art / "bounded_off" / "server.stderr"
        path.write_text(path.read_text() + self.bounded_marker_fixed())
        self.assert_failure("OFF must have zero bounded markers")

    def test_off_must_have_single_request(self) -> None:
        self.mutate_json("bounded_off/result.json",
                         lambda v: v.update(requests_count=2))
        self.assert_failure("OFF must have single request")

    # ═══════════════════════════════════════════════════════════════════════
    # FIXED case
    # ═══════════════════════════════════════════════════════════════════════

    def test_fixed_target_mode_must_be_fixed(self) -> None:
        path = self.art / "bounded_fixed" / "server.stderr"
        path.write_text(path.read_text().replace("target_mode=fixed", "target_mode=dynamic"))
        self.assert_failure("target_mode=dynamic, expected fixed")

    def test_fixed_must_have_two_requests(self) -> None:
        self.mutate_json("bounded_fixed/result.json",
                         lambda v: v.update(requests_count=1))
        self.assert_failure("FIXED must have 2 requests")

    def test_fixed_reuse_allocations_must_be_positive(self) -> None:
        path = self.art / "bounded_fixed" / "server.stderr"
        path.write_text(path.read_text().replace("reuse_allocations=2", "reuse_allocations=0"))
        self.assert_failure("reuse_allocations=0 must be >0")

    def test_fixed_write_commits_must_be_positive(self) -> None:
        path = self.art / "bounded_fixed" / "server.stderr"
        path.write_text(path.read_text().replace("write_commits=2", "write_commits=0"))
        self.assert_failure("write_commits=0 must be >0")

    def test_fixed_ownership_aborted_is_rejected(self) -> None:
        path = self.art / "bounded_fixed" / "server.stderr"
        path.write_text(path.read_text().replace("ownership_aborted=0", "ownership_aborted=1"))
        self.assert_failure("ownership_aborted=1")

    # ═══════════════════════════════════════════════════════════════════════
    # DYNAMIC_NOOP case
    # ═══════════════════════════════════════════════════════════════════════

    def test_dynamic_noop_zero_bounded_markers(self) -> None:
        path = self.art / "bounded_dynamic_noop" / "server.stderr"
        path.write_text(path.read_text() + self.bounded_marker_fixed())
        self.assert_failure("DYNAMIC_NOOP must have zero bounded markers")

    def test_dynamic_noop_telemetry_must_be_normal(self) -> None:
        """All telemetry in DYNAMIC_NOOP must show NORMAL state."""
        path = self.art / "bounded_dynamic_noop" / "server.stderr"
        path.write_text(path.read_text().replace(
            "state=NORMAL", "state=PRESSURE"))
        self.assert_failure("DYNAMIC_NOOP must have ALL telemetry states=NORMAL")

    def test_dynamic_noop_must_be_configured_with_bounded_and_dynamic(self) -> None:
        """DYNAMIC_NOOP env must have BOUNDED_RELEASE=1 + DYNAMIC_TARGET=1."""
        self.mutate_json("bounded_dynamic_noop/environment.json",
                         lambda e: e.pop("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET", None))
        self.mutate_json("bounded_dynamic_noop/execution.json",
                         lambda v: v["environment"].pop(
                             "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET", None))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["DYNAMIC_NOOP"]["environment"].pop(
                             "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET", None))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["DYNAMIC_NOOP"]["execution"]["environment"].pop(
                             "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET", None))
        self.assert_failure("DYNAMIC_NOOP must have DYNAMIC_TARGET=1")

    def test_dynamic_noop_single_request(self) -> None:
        self.mutate_json("bounded_dynamic_noop/result.json",
                         lambda v: v.update(requests_count=2,
                                           http_status_2=200,
                                           response_text_2="same"))
        self.assert_failure("DYNAMIC_NOOP must have single request")

    # ═══════════════════════════════════════════════════════════════════════
    # DYNAMIC_RELEASE case
    # ═══════════════════════════════════════════════════════════════════════

    def test_dynamic_release_target_formula_must_match(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("target_bytes=16384", "target_bytes=32768"))
        self.assert_failure("target=32768 != expected=16384")

    def test_dynamic_release_decision_reason_must_be_dynamic(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("decision_reason=dynamic", "decision_reason=fixed"))
        self.assert_failure("decision_reason must be dynamic")

    def test_dynamic_release_target_mode_must_be_dynamic(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("target_mode=dynamic", "target_mode=fixed"))
        self.assert_failure("target_mode must be dynamic")

    def test_dynamic_release_reuse_allocations_must_be_positive(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("reuse_allocations=2", "reuse_allocations=0"))
        self.assert_failure("reuse_allocations=0 must be >0")

    def test_dynamic_release_write_commits_must_be_positive(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("write_commits=2", "write_commits=0"))
        self.assert_failure("write_commits=0 must be >0")

    def test_dynamic_release_must_have_two_requests(self) -> None:
        self.mutate_json("bounded_dynamic_release/result.json",
                         lambda v: v.update(requests_count=1))
        self.assert_failure("DYNAMIC_RELEASE must have 2 requests")

    def test_dynamic_release_kv_budget_must_be_valid(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("kv_budget_valid=1", "kv_budget_valid=0"))
        self.assert_failure("kv_budget_valid=1")

    def test_dynamic_release_mincore_no_drop_fails(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("mincore_after_bytes=65536",
                                                 "mincore_after_bytes=163840"))
        self.assert_failure("mincore must show physical drop")

    # ═══════════════════════════════════════════════════════════════════════
    # Env isolation
    # ═══════════════════════════════════════════════════════════════════════

    def test_off_has_bounded_key_rejected(self) -> None:
        self.mutate_json("bounded_off/environment.json",
                         lambda e: e.update({"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"}))
        self.mutate_json("bounded_off/execution.json",
                         lambda v: v["environment"].update(
                             {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["OFF"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["OFF"]["execution"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"}))
        self.assert_failure("OFF env has bounded-release key")

    def test_fixed_has_dynamic_target_rejected(self) -> None:
        self.mutate_json("bounded_fixed/environment.json",
                         lambda e: e.update(
                             {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1"}))
        self.mutate_json("bounded_fixed/execution.json",
                         lambda v: v["environment"].update(
                             {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["FIXED"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["FIXED"]["execution"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1"}))
        self.assert_failure("FIXED must not have DYNAMIC_TARGET")

    def test_shared_key_differs_rejected(self) -> None:
        self.mutate_json("bounded_fixed/environment.json",
                         lambda e: e.update({"LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "999"}))
        self.mutate_json("bounded_fixed/execution.json",
                         lambda v: v["environment"].update(
                             {"LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "999"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["FIXED"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "999"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["FIXED"]["execution"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "999"}))
        self.assert_failure("Shared key 'LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS' differs")

    def test_calibrated_key_safe_trigger_mismatch_rejected(self) -> None:
        """OFF and DYNAMIC_NOOP must share the same SAFE thresholds."""
        self.mutate_json("bounded_dynamic_noop/environment.json",
                         lambda e: e.update({"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.mutate_json("bounded_dynamic_noop/execution.json",
                         lambda v: v["environment"].update(
                             {"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["DYNAMIC_NOOP"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["DYNAMIC_NOOP"]["execution"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.assert_failure("Calibrated key 'LLAMA_KV_PRESSURE_RSS_KB' differs")

    # ═══════════════════════════════════════════════════════════════════════
    # Response identity
    # ═══════════════════════════════════════════════════════════════════════

    def test_response_differs_rejected(self) -> None:
        path = self.art / "bounded_fixed" / "result.json"
        data = json.loads(path.read_text())
        data["response_text_1"] = "different"
        path.write_text(json.dumps(data))
        self.assert_failure("OFF/FIXED req1 responses must match")

    def test_req2_response_differs_rejected(self) -> None:
        path = self.art / "bounded_fixed" / "result.json"
        data = json.loads(path.read_text())
        data["response_text_2"] = "different"
        path.write_text(json.dumps(data))
        self.assert_failure("OFF/FIXED req2 responses must match")

    # ═══════════════════════════════════════════════════════════════════════
    # Process cleanup
    # ═══════════════════════════════════════════════════════════════════════

    def test_missing_phases_fails(self) -> None:
        (self.art / "bounded_dynamic_release" / "phases.json").unlink()
        self.assert_failure("DYNAMIC_RELEASE: phases.json missing")

    def test_residual_process_fails(self) -> None:
        self.mutate_json("bounded_dynamic_release/phases.json",
                         lambda value: value["shutdown"].update(residual_process=True))
        self.assert_failure("DYNAMIC_RELEASE: residual process after shutdown")

    # ═══════════════════════════════════════════════════════════════════════
    # Lifecycle counters
    # ═══════════════════════════════════════════════════════════════════════

    def test_off_nonzero_bounded_source_fails(self) -> None:
        path = self.art / "bounded_off" / "server.stderr"
        path.write_text(path.read_text().replace(
            "bounded_release_calls=0", "bounded_release_calls=1"))
        self.assert_failure("OFF: final bounded source bounded_release_calls=1")

    def test_dynamic_noop_nonzero_bounded_source_fails(self) -> None:
        path = self.art / "bounded_dynamic_noop" / "server.stderr"
        path.write_text(path.read_text().replace(
            "bounded_release_bytes=0", "bounded_release_bytes=4096"))
        self.assert_failure("DYNAMIC_NOOP: final bounded source bounded_release_bytes=4096")

    def test_dynamic_release_fatal_counter_nonzero_fails(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        path.write_text(path.read_text().replace("write_rollbacks=0", "write_rollbacks=1"))
        self.assert_failure("DYNAMIC_RELEASE: final lifecycle write_rollbacks=1")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary integrity
    # ═══════════════════════════════════════════════════════════════════════

    def test_runner_verdict_pass_is_rejected(self) -> None:
        self.mutate_json("summary.json",
                         lambda value: value.update(verdict="PASS"))
        self.assert_failure("summary.json contains verdict=PASS")

    # ═══════════════════════════════════════════════════════════════════════
    # Marker schema
    # ═══════════════════════════════════════════════════════════════════════

    def test_marker_schema_mismatch_rejected(self) -> None:
        path = self.art / "bounded_fixed" / "server.stderr"
        path.write_text(path.read_text().replace(" cap_layout=1", "", 1))
        self.assert_failure("schema mismatch missing=['cap_layout']")

    def test_marker_duplicate_field_rejected(self) -> None:
        path = self.art / "bounded_fixed" / "server.stderr"
        lines = path.read_text().split("\n")
        for i, line in enumerate(lines):
            if line.startswith("kv_pressure_bounded_release "):
                lines[i] = line.rstrip() + " released_bytes=0"
                break
        path.write_text("\n".join(lines))
        self.assert_failure("duplicates field")

    # ═══════════════════════════════════════════════════════════════════════
    # Server topology
    # ═══════════════════════════════════════════════════════════════════════

    def test_server_topology_parallel_not_1_rejected(self) -> None:
        path = self.art / "bounded_off" / "execution.json"
        exe = json.loads(path.read_text())
        idx = exe["argv"].index("--parallel")
        exe["argv"][idx + 1] = "4"
        path.write_text(json.dumps(exe))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["OFF"].update(execution=exe))
        self.assert_failure("--parallel must be 1")

    # ═══════════════════════════════════════════════════════════════════════
    # Dynamic target clamp verification
    # ═══════════════════════════════════════════════════════════════════════

    def test_target_clamp_must_match_binding_factor(self) -> None:
        path = self.art / "bounded_dynamic_release" / "server.stderr"
        content = path.read_text()
        content = content.replace("max_release_bytes=1073741824", "max_release_bytes=4096")
        content = content.replace("target_bytes=16384", "target_bytes=4096")
        content = content.replace("released_bytes=16384", "released_bytes=4096")
        content = content.replace("bounded_cnt_bytes_delta=16384", "bounded_cnt_bytes_delta=4096")
        content = content.replace("target_clamp=reclaimable", "target_clamp=max_release")
        content = content.replace("released_blocks=4", "released_blocks=1")
        content = content.replace("bounded_cnt_blocks_delta=4", "bounded_cnt_blocks_delta=1")
        content = content.replace("mincore_after_bytes=65536", "mincore_after_bytes=77824")
        content = content.replace("blocks_scanned=4", "blocks_scanned=1")
        content = content.replace("bounded_release_bytes=16384", "bounded_release_bytes=4096")
        content = content.replace("bounded_release_blocks=4", "bounded_release_blocks=1")
        path.write_text(content)
        result = self.run_parser()
        self.assertEqual(result.returncode, 0,
                         f"max_release-clamped target should pass; got: {result.stderr}")

    # ═══════════════════════════════════════════════════════════════════════
    # Threshold derivation from calibration
    # ═══════════════════════════════════════════════════════════════════════

    def test_case_thresholds_match_calibration(self) -> None:
        """Case env thresholds must use calibration-derived values."""
        # OFF should use safe thresholds
        self.mutate_json("bounded_off/environment.json",
                         lambda e: e.update({"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.mutate_json("bounded_off/execution.json",
                         lambda v: v["environment"].update(
                             {"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["OFF"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.mutate_json("manifest.json",
                         lambda v: v["case_configs"]["OFF"]["execution"]["environment"].update(
                             {"LLAMA_KV_PRESSURE_RSS_KB": "9999999"}))
        self.assert_failure("OFF PRESSURE_RSS_KB=9999999 !=")


if __name__ == "__main__":
    unittest.main()
