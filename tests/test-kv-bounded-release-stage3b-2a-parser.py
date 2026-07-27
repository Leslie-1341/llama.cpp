#!/usr/bin/env python3
"""Synthetic-fixture tests for Stage 3B-2A parser (long-context ladder + continuous requests).

Covers: mandatory file checks, manifest identity, environment closure, KV_PAGED_RELEASE_STATS
enforcement, round numbering, stderr windows, blocks_skipped_owned, config contamination,
ownership/madvise safety.
"""

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
PARSER = ROOT / "scripts" / "parse-kv-bounded-release-stage3b-2a.py"


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ident(path: pathlib.Path) -> dict[str, object]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


# ── synthetic token calibration ──────────────────────────────────────────────

TOKEN_CALIB = {
    "schema_version": 1,
    "targets": [1024, 2048],
    "results": {
        "1024": {
            "target_tokens": 1024, "actual_tokens": 1024, "delta": 0,
            "prompt_text": "test " * 1024,
            "prompt_text_sha256": hashlib.sha256(("test " * 1024).encode()).hexdigest(),
            "seed_tokens_per_rep": 3, "repetitions": 342, "iterations": 1,
        },
        "2048": {
            "target_tokens": 2048, "actual_tokens": 2048, "delta": 0,
            "prompt_text": "test " * 2048,
            "prompt_text_sha256": hashlib.sha256(("test " * 2048).encode()).hexdigest(),
            "seed_tokens_per_rep": 3, "repetitions": 683, "iterations": 2,
        },
    },
    "seed_text_sha256": "abc",
}

# ── synthetic RSS calibration ────────────────────────────────────────────────
# Each tier now carries the runner-recorded completion timing sample
# (completion_wall_s) and the derived completion_timeout_s.  1024 tier took
# 10s wall → timeout = max(10*2.5, 30) = 30s.  2048 tier took 20s → 50s.

RSS_CALIB_1024 = {
    "target_tokens": 1024, "source_target": 1024, "ctx_size": 1280,
    "rss_idle_kb": 6000000, "rss_peak_kb": 6200000, "delta_kb": 200000,
    "margin_kb": 100000,
    "pressure_trigger_kb": 6050000, "critical_trigger_kb": 6150000,
    "pressure_safe_kb": 6300000, "critical_safe_kb": 6300001,
    "low_water_trigger_kb": 6000000, "low_water_safe_kb": 6300000,
    "http_status": 200,
    "response_text_sha256": "abc", "response_text_len": 183,
    "completion_wall_s": 10.0,
    "completion_timeout_s": 30.0,
    "completion_timeout_margin": 2.5,
    "completion_timeout_floor": 30.0,
    "calibration_timeout_s": 180.0,
    "calibration_timeout_mode": "bootstrap",
    "calibration_timeout_bootstrap_s": 180.0,
    "calibration_timeout_previous_target": None,
    "calibration_timeout_previous_wall_s": None,
    "calibration_timeout_token_ratio": None,
    "calibration_timeout_safety_factor": 1.5,
}

RSS_CALIB_2048 = {
    "target_tokens": 2048, "source_target": 2048, "ctx_size": 2304,
    "rss_idle_kb": 6200000, "rss_peak_kb": 6500000, "delta_kb": 300000,
    "margin_kb": 150000,
    "pressure_trigger_kb": 6275000, "critical_trigger_kb": 6425000,
    "pressure_safe_kb": 6650000, "critical_safe_kb": 6650001,
    "low_water_trigger_kb": 6200000, "low_water_safe_kb": 6650000,
    "http_status": 200,
    "response_text_sha256": "abc", "response_text_len": 183,
    "completion_wall_s": 20.0,
    "completion_timeout_s": 50.0,
    "completion_timeout_margin": 2.5,
    "completion_timeout_floor": 30.0,
    "calibration_timeout_s": 180.0,
    "calibration_timeout_mode": "previous_success_scaled",
    "calibration_timeout_bootstrap_s": 180.0,
    "calibration_timeout_previous_target": 1024,
    "calibration_timeout_previous_wall_s": 10.0,
    "calibration_timeout_token_ratio": 2.0,
    "calibration_timeout_safety_factor": 1.5,
    "calibration_timeout_scaled_s": 30.0,
}

BASELINE_RESPONSE = "Deterministic tests ensure reproducible verification."
BASELINE_SHA = hashlib.sha256(BASELINE_RESPONSE.encode()).hexdigest()


class Stage3B2AParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.art = pathlib.Path(tempfile.mkdtemp(prefix="stage3b_2a_parser_"))
        self._write_valid_artifact()

    def tearDown(self) -> None:
        shutil.rmtree(self.art, ignore_errors=True)

    # ── marker builders ─────────────────────────────────────────────────────

    @staticmethod
    def telemetry(state: str = "CRITICAL") -> str:
        return ("kv_pressure_telemetry state={s} previous_state=PRESSURE "
                "source=RSS_ABSOLUTE sample_valid=1 stale=0 config_valid=1 rss_kb=6100000 "
                "cgroup_current_bytes=0 cgroup_max_bytes=0 cgroup_current_kb=0 "
                "cgroup_max_kb=0 cgroup_high_kb=0 psi_some_avg10=0 psi_full_avg10=0 "
                "pressure_basis_valid=1 pressure_current_bytes=6246400000 "
                "pressure_low_water_bytes=6144000000 pressure_basis_generation=1 "
                "sample_latency_ns=1 sample_count=1 skip_count=0 idle=0 "
                "trigger=first,state,source\n".format(s=state))

    @staticmethod
    def bounded_executed() -> str:
        return ("kv_pressure_bounded_release state=CRITICAL source=RSS_ABSOLUTE stale=0 "
                "released_bytes=35389440 released_blocks=9 blocks_scanned=20 "
                "blocks_skipped_owned=5 blocks_skipped_state=0 madvise_failures=0 "
                "shortfall_bytes=0 overshoot_bytes=1835008 block_scan_exhausted=0 "
                "ownership_aborted=0 target_mode=dynamic pressure_basis_valid=1 "
                "pressure_current_bytes=6246400000 pressure_low_water_bytes=6144000000 "
                "pressure_basis_generation=1 kv_budget_valid=1 kv_budget_ownership_aborted=0 "
                "kv_resident_bytes=268173312 kv_reclaimable_resident_bytes=35389440 "
                "water_excess_bytes=102400000 water_shortfall_bytes=0 "
                "water_overshoot_bytes=0 max_release_bytes=1073741824 "
                "target_clamp=none decision_reason=dynamic "
                "target_bytes=33554432 max_scan_blocks=256 "
                "legacy_enabled=0 sample_count=5 episode=1 "
                "cooldown_ms=2000 skipped_reason=none idle=0 "
                "mincore_before_bytes=268173312 mincore_after_bytes=232783872 "
                "bounded_cnt_bytes_delta=35389440 bounded_cnt_blocks_delta=9 "
                "can_enable=1 cap_paged=1 cap_ingraph=1 cap_layers=1 "
                "cap_row_idx=1 cap_swap_disabled=1 cap_layout=1\n")

    @staticmethod
    def bounded_noop(reason: str = "owned") -> str:
        marker = Stage3B2AParserTest.bounded_executed()
        marker = marker.replace("released_bytes=35389440 released_blocks=9", "released_bytes=0 released_blocks=0")
        marker = marker.replace("blocks_scanned=20 blocks_skipped_owned=5 blocks_skipped_state=0", "blocks_scanned=256 blocks_skipped_owned=256 blocks_skipped_state=0")
        marker = marker.replace("shortfall_bytes=0 overshoot_bytes=1835008 block_scan_exhausted=0", "shortfall_bytes=0 overshoot_bytes=0 block_scan_exhausted=1")
        marker = marker.replace("mincore_before_bytes=268173312 mincore_after_bytes=232783872", "mincore_before_bytes=268173312 mincore_after_bytes=268173312")
        marker = marker.replace("bounded_cnt_bytes_delta=35389440 bounded_cnt_blocks_delta=9", "bounded_cnt_bytes_delta=0 bounded_cnt_blocks_delta=0")
        marker = marker.replace("sample_count=5 episode=1", "sample_count=6 episode=2")
        if reason == "state":
            marker = marker.replace("blocks_scanned=256 blocks_skipped_owned=256 blocks_skipped_state=0", "blocks_scanned=4 blocks_skipped_owned=0 blocks_skipped_state=4")
            marker = marker.replace("block_scan_exhausted=1", "block_scan_exhausted=0")
        elif reason == "scan":
            marker = marker.replace("blocks_scanned=256 blocks_skipped_owned=256 blocks_skipped_state=0", "blocks_scanned=256 blocks_skipped_owned=0 blocks_skipped_state=0")
        return marker

    @staticmethod
    def release_stats(reuse: int = 5, release_calls: int = 3) -> str:
        return ("KV_PAGED_RELEASE_STATS reuse_allocations={r} write_commits={r} "
                "write_rollbacks=0 dummy_candidate_pending_write_cell=0 "
                "released_redirect_no_dummy=0 released_redirect_no_dummy_pending_write=0 "
                "ensure_pending_write_rejected=0 input_setup_fatal=0 "
                "row_mapping_fatal=0 write_mapping_fatal=0 "
                "active_nonresident_fatal=0 "
                "bounded_release_calls={c} bounded_release_bytes=35389440 "
                "bounded_release_blocks=9\n".format(r=reuse, c=release_calls))

    # ── fixture helpers ─────────────────────────────────────────────────────

    def _write_manifest(self, overrides: dict[str, object] | None = None) -> None:
        runner_id = (ident(PARSER.parent / "run-kv-bounded-release-stage3b-2a.py")
                     if (PARSER.parent / "run-kv-bounded-release-stage3b-2a.py").exists()
                     else {"path": "runner", "size": 100, "sha256": "c" * 64})
        base = {
            "protocol": "kv_bounded_release_stage3b_2a",
            "protocol_version": 1,
            "runner_status": "run_complete",
            "failure_phase": None,
            "failure_target": None,
            "failure_reason": None,
            "head_sha": "0" * 40,
            "head_short": "0" * 10,
            "worktree_dirty": True,
            "worktree_status": ["?? somefile"],
            "capture_mode": "diagnostic_dirty",
            "timestamp_utc": "2026-07-25T00:00:00Z",
            "binary": {"path": "bin", "size": 100, "sha256": "a" * 64},
            "model": {"path": "model", "size": 200, "sha256": "b" * 64},
            "runner": runner_id,
            "parser": ident(PARSER),
            "token_targets": [1024, 2048],
            "num_continuous": 20,
            "n_predict": 32,
            "seed": 1,
            "dynamic_hard_cap_bytes": 1073741824,
            "max_scan_blocks": 256,
            "diff": {"path": "diff", "size": 0, "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
            "diff_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "token_calibration": {
                "schema_version": 1,
                "results": {
                    "1024": TOKEN_CALIB["results"]["1024"],
                    "2048": TOKEN_CALIB["results"]["2048"],
                },
            },
            "rss_calibration": {
                "schema_version": 1,
                "levels": {
                    "1024": RSS_CALIB_1024,
                    "2048": RSS_CALIB_2048,
                },
            },
            "effective_context": {
                "requested_targets": [1024, 2048],
                "effective_n_ctx": 8192,
                "n_predict": 32,
                "special_overhead": 1,
                "safety": 95,
                "token_align": 32,
                "max_prompt_tokens": 8064,
                "effective_targets": [1024, 2048],
                "probe": {
                    "requested_ctx_size": 2176,
                    "requested_targets": [1024, 2048],
                    "effective_n_ctx": 8192,
                    "max_prompt_tokens": 8064,
                    "effective_targets": [1024, 2048],
                    "completion_requests": 0,
                    "http_completion_statuses": [],
                },
                "clamped_tiers": [
                    {"requested": 1024, "effective": 1024},
                    {"requested": 2048, "effective": 2048},
                ],
            },
            "source_snapshot": {
                "schema_version": 1,
                "head_sha": "0" * 40,
                "tracked_diff": {"path": "d", "size": 0, "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
                "untracked_files": [],
            },
        }
        if overrides:
            base.update(overrides)
        (self.art / "manifest.json").write_text(
            json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # summary.json
        (self.art / "summary.json").write_text(
            json.dumps({"runner_status": "run_complete"}, indent=2) + "\n", encoding="utf-8")

    def _write_token_calib(self) -> None:
        d = self.art / "token_calibration"
        d.mkdir(parents=True, exist_ok=True)
        (d / "calibration.json").write_text(
            json.dumps(TOKEN_CALIB, indent=2) + "\n", encoding="utf-8")
        for t in [1024, 2048]:
            (d / f"prompt_{t}t.json").write_text(
                json.dumps(TOKEN_CALIB["results"][str(t)], indent=2) + "\n",
                encoding="utf-8")

    def _write_rss_calib(self) -> None:
        d = self.art / "rss_calibration"
        d.mkdir(parents=True, exist_ok=True)
        (d / "rss_calibration.json").write_text(
            json.dumps({"schema_version": 1, "levels": {
                "1024": RSS_CALIB_1024, "2048": RSS_CALIB_2048,
            }}, indent=2) + "\n", encoding="utf-8")
        for t, cal in [(1024, RSS_CALIB_1024), (2048, RSS_CALIB_2048)]:
            ld = d / f"ctx_{t}"
            ld.mkdir(parents=True, exist_ok=True)
            (ld / "rss_calibration.json").write_text(
                json.dumps(cal, indent=2) + "\n", encoding="utf-8")

    def _write_case_files(self, d: pathlib.Path, env: dict[str, str],
                          result: dict[str, object],
                          stderr: str, phases: dict[str, object]) -> None:
        """Write all 5 mandatory case files."""
        (d / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
        exe_env = dict(env)
        (d / "execution.json").write_text(
            json.dumps({"argv": [], "environment": exe_env, "ctx_size": 1280}, indent=2) + "\n",
            encoding="utf-8")
        (d / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if stderr:
            (d / "server.stderr").write_text(stderr, encoding="utf-8")
        else:
            (d / "server.stderr").write_text("", encoding="utf-8")
        (d / "phases.json").write_text(json.dumps(phases, indent=2) + "\n", encoding="utf-8")

    def _write_ladder_case(self, target: int, case_label: str,
                           result_override: dict[str, object] | None = None,
                           stderr: str | None = None,
                           phases_override: dict[str, object] | None = None) -> None:
        d = self.art / "ladder" / f"t{target}_{case_label.lower()}"
        d.mkdir(parents=True, exist_ok=True)

        if result_override is not None:
            result = result_override
        else:
            src_cal = RSS_CALIB_1024 if target == 1024 else RSS_CALIB_2048
            result = {
                "response_text": BASELINE_RESPONSE,
                "response_text_sha256": BASELINE_SHA,
                "response_text_len": len(BASELINE_RESPONSE),
                "http_status": 200,
                "rss_after_request_kb": 6200000 if target == 1024 else 6500000,
                "rss_after_release_kb": 6100000 if target == 1024 else 6400000,
                # Runner records the per-tier derived completion timeout and the
                # effective n_ctx it ran inside; the source tier is itself.
                "effective_n_ctx": 8192,
                "source_target": target,
                "legal_tier": target,
                "completion_timeout_s": src_cal["completion_timeout_s"],
            }

        phases = phases_override or {"shutdown": {"residual_process": False}}

        if case_label == "OFF":
            env = {
                "LLAMA_KV_PRESSURE_RSS_KB": "6300000",
                "LLAMA_KV_CRITICAL_RSS_KB": "6300001",
                "LLAMA_KV_LOW_WATER_RSS_KB": "6300000",
            }
        else:
            env = {
                "LLAMA_KV_PRESSURE_RSS_KB": "6050000",
                "LLAMA_KV_CRITICAL_RSS_KB": "6150000",
                "LLAMA_KV_LOW_WATER_RSS_KB": "6000000",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": "1073741824",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": "256",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "2000",
                "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "10000",
            }

        self._write_case_files(d, env, result, stderr or "", phases)

        # Write stderr_windows for active-ownership evidence
        if case_label == "DYNAMIC_RELEASE" and stderr:
            stderr_bytes = stderr.encode()
            windows_data = {
                "windows": [{
                    "round": 1,
                    "start_byte": 0,
                    "end_byte": len(stderr_bytes),
                    "start_timestamp_s": 0.0,
                    "end_timestamp_s": 1.0,
                    "duration_s": 1.0,
                }],
                "request_count": 1,
            }
            (d / "stderr_windows.json").write_text(
                json.dumps(windows_data, indent=2) + "\n", encoding="utf-8")

    def _write_continuous(self, num_rounds: int = 20, all_identical: bool = True,
                         cumulative_errors: int = 0,
                         stderr: str | None = None,
                         phases_override: dict[str, object] | None = None) -> None:
        d = self.art / "continuous_requests"
        d.mkdir(parents=True, exist_ok=True)

        baseline_text = BASELINE_RESPONSE
        baseline_sha = BASELINE_SHA
        rounds = []
        for i in range(num_rounds):
            text = baseline_text if all_identical else f"variant round {i}"
            rounds.append({
                "round": i + 1,
                "http_status": 200,
                "response_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "response_text_len": len(text),
                "matches_baseline": text == baseline_text,
                "rss_after_request_kb": 6200000,
                "rss_after_release_kb": 6100000,
            })
        result = {
            "num_requests": 20,
            "completed_rounds": num_rounds,
            "all_responses_identical": all_identical,
            "cumulative_error_count": cumulative_errors,
            "baseline_text_sha256": baseline_sha,
            "baseline_text_len": len(baseline_text),
            "rounds": rounds,
            "completion_timeout_s": RSS_CALIB_1024["completion_timeout_s"],
            "effective_n_ctx": 8192,
            "ctx_size": 1280,
        }

        env = {
            "LLAMA_KV_PRESSURE_RSS_KB": "6050000",
            "LLAMA_KV_CRITICAL_RSS_KB": "6150000",
            "LLAMA_KV_LOW_WATER_RSS_KB": "6000000",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": "1073741824",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": "256",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "2000",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "10000",
        }
        phases = phases_override or {"shutdown": {"residual_process": False}}

        self._write_case_files(d, env, result, stderr or "", phases)

        # Write stderr_windows for active-ownership evidence.
        # In a real run, each window accumulates more stderr (cumulative).
        # For a synthetic fixture, each window covers from byte 0 to a growing
        # end_byte so every window contains all markers written so far.
        if stderr:
            stderr_bytes = stderr.encode()
            full_len = len(stderr_bytes)
            windows = []
            for i in range(num_rounds):
                e = max(1, full_len * (i + 1) // num_rounds)
                windows.append({
                    "round": i + 1,
                    "start_byte": 0,
                    "end_byte": min(e, full_len),
                    "start_timestamp_s": float(i),
                    "end_timestamp_s": float(i + 1),
                    "duration_s": 1.0,
                })
            (d / "stderr_windows.json").write_text(
                json.dumps({"windows": windows, "request_count": num_rounds},
                          indent=2) + "\n", encoding="utf-8")

    def _write_valid_artifact(self) -> None:
        self._write_manifest()
        manifest = json.loads((self.art / "manifest.json").read_text())
        probe_dir = self.art / "context_probe"
        probe_dir.mkdir()
        (probe_dir / "probe.json").write_text(
            json.dumps(manifest["effective_context"]["probe"], indent=2) + "\n",
            encoding="utf-8")
        self._write_token_calib()
        self._write_rss_calib()

        for t in [1024, 2048]:
            for case_label in ["OFF", "DYNAMIC_RELEASE"]:
                stderr = None
                if case_label == "DYNAMIC_RELEASE":
                    stderr = (self.telemetry(state="CRITICAL") +
                              self.bounded_executed() +
                              self.release_stats(reuse=3, release_calls=1))
                elif case_label == "OFF":
                    stderr = self.telemetry(state="NORMAL")
                self._write_ladder_case(t, case_label, stderr=stderr)

        cont_stderr = (self.telemetry(state="CRITICAL") +
                       self.bounded_executed() +
                       self.release_stats(reuse=20, release_calls=20))
        self._write_continuous(stderr=cont_stderr)

    def _run_parser(self) -> subprocess.CompletedProcess[str]:
        result_path = self.art / "parser.json"
        return subprocess.run(
            [sys.executable, str(PARSER), str(self.art),
             "--result-path", str(result_path)],
            text=True, capture_output=True, check=False,
        )

    # ── baseline ────────────────────────────────────────────────────────────

    def test_valid_baseline_passes(self) -> None:
        r = self._run_parser()
        self.assertEqual(r.returncode, 0,
                         f"parser should PASS\nstdout={r.stdout}\nstderr={r.stderr}")

    def test_parser_self_verify(self) -> None:
        result_path = self.art / "parser.json"
        subprocess.run(
            [sys.executable, str(PARSER), str(self.art),
             "--result-path", str(result_path)],
            text=True, capture_output=True, check=True,
        )
        verify = subprocess.run(
            [sys.executable, str(PARSER), str(self.art),
             "--verify-result", str(result_path)],
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(verify.returncode, 0,
                         f"--verify-result failed\nstderr={verify.stderr}")

    # ── mandatory file checks ──────────────────────────────────────────────

    def test_ladder_missing_stderr_rejected(self) -> None:
        (self.art / "ladder" / "t1024_off" / "server.stderr").unlink()
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing server.stderr must fail")

    def test_ladder_missing_environment_rejected(self) -> None:
        (self.art / "ladder" / "t1024_off" / "environment.json").unlink()
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing environment.json must fail")

    def test_ladder_missing_execution_rejected(self) -> None:
        (self.art / "ladder" / "t1024_off" / "execution.json").unlink()
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing execution.json must fail")

    def test_ladder_missing_result_rejected(self) -> None:
        (self.art / "ladder" / "t1024_off" / "result.json").unlink()
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing result.json must fail")

    def test_ladder_missing_phases_rejected(self) -> None:
        (self.art / "ladder" / "t1024_off" / "phases.json").unlink()
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing phases.json must fail")

    def test_continuous_missing_stderr_rejected(self) -> None:
        (self.art / "continuous_requests" / "server.stderr").unlink()
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "continuous missing stderr must fail")

    # ── manifest identity ──────────────────────────────────────────────────

    def test_manifest_binary_size_zero_rejected(self) -> None:
        self._write_manifest(overrides={"binary": {"path": "bin", "size": 0, "sha256": "a" * 64}})
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "binary size=0 must fail")

    def test_manifest_binary_sha256_bad_length_rejected(self) -> None:
        self._write_manifest(overrides={"binary": {"path": "bin", "size": 100, "sha256": "short"}})
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "binary sha256 bad length must fail")

    def test_manifest_model_missing_path_rejected(self) -> None:
        self._write_manifest(overrides={"model": {"size": 100, "sha256": "b" * 64}})
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "model missing path must fail")

    def test_wrong_protocol_rejected(self) -> None:
        self._write_manifest(overrides={"protocol": "wrong_protocol"})
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "wrong protocol must fail")

    def test_missing_manifest_rejected(self) -> None:
        (self.art / "manifest.json").unlink()
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing manifest must fail")

    # ── environment closure ────────────────────────────────────────────────

    def test_env_execution_mismatch_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_off"
        exe = json.loads((d / "execution.json").read_text())
        exe["environment"]["EXTRA_KEY"] = "value"
        (d / "execution.json").write_text(json.dumps(exe, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "env/execution mismatch must fail")

    # ── token calibration failures ─────────────────────────────────────────

    def test_token_calibration_missing_rejected(self) -> None:
        shutil.rmtree(self.art / "token_calibration")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing token calibration must fail")

    def test_token_calibration_delta_exceeds_tolerance(self) -> None:
        calib_data = json.loads((self.art / "token_calibration/calibration.json").read_text())
        calib_data["results"]["1024"]["actual_tokens"] = 1030
        (self.art / "token_calibration/calibration.json").write_text(
            json.dumps(calib_data, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "token delta > ±2 must fail")

    # ── RSS calibration failures ───────────────────────────────────────────

    def test_rss_calibration_missing_rejected(self) -> None:
        shutil.rmtree(self.art / "rss_calibration")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing RSS calibration must fail")

    def test_rss_calibration_trigger_not_less_than_safe(self) -> None:
        calib_data = json.loads((self.art / "rss_calibration/rss_calibration.json").read_text())
        calib_data["levels"]["1024"]["pressure_trigger_kb"] = 7000000
        (self.art / "rss_calibration/rss_calibration.json").write_text(
            json.dumps(calib_data, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "trigger >= safe must fail")

    # ── ladder: OFF contamination + response identity ──────────────────────

    def test_off_has_bounded_marker_rejected(self) -> None:
        off_dir = self.art / "ladder" / "t1024_off"
        (off_dir / "server.stderr").write_text(
            self.telemetry(state="CRITICAL") + self.bounded_executed(), encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "OFF must not have bounded_release markers")

    def test_off_has_bounded_env_key_rejected(self) -> None:
        off_dir = self.art / "ladder" / "t1024_off"
        env = json.loads((off_dir / "environment.json").read_text())
        env["LLAMA_KV_PRESSURE_BOUNDED_RELEASE"] = "1"
        (off_dir / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "OFF must not have bounded-release env keys")

    def test_ladder_response_mismatch_rejected(self) -> None:
        dyn_result = json.loads(
            (self.art / "ladder" / "t1024_dynamic_release" / "result.json").read_text())
        dyn_result["response_text_sha256"] = hashlib.sha256(
            "Different response.".encode()).hexdigest()
        (self.art / "ladder" / "t1024_dynamic_release" / "result.json").write_text(
            json.dumps(dyn_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "OFF != DYNAMIC_RELEASE must fail")

    # ── ladder: ownership / madvise / mincore ──────────────────────────────

    def test_ladder_ownership_aborted_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed().replace("ownership_aborted=0",
                                                   "ownership_aborted=1"))
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "ownership_aborted must fail")

    def test_ladder_blocks_skipped_owned_zero_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed().replace("blocks_skipped_owned=5",
                                                   "blocks_skipped_owned=0"))
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "blocks_skipped_owned=0 must fail")

    def test_ladder_no_bounded_execution_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        stderr = (self.telemetry(state="NORMAL") +
                  "kv_pressure_bounded_release state=NORMAL source=NONE stale=0 "
                  "released_bytes=0 released_blocks=0 blocks_scanned=0 "
                  "blocks_skipped_owned=0 blocks_skipped_state=0 madvise_failures=0 "
                  "shortfall_bytes=0 overshoot_bytes=0 block_scan_exhausted=0 "
                  "ownership_aborted=0 target_mode=dynamic pressure_basis_valid=0 "
                  "pressure_current_bytes=0 pressure_low_water_bytes=0 "
                  "pressure_basis_generation=0 kv_budget_valid=0 kv_budget_ownership_aborted=0 "
                  "kv_resident_bytes=0 kv_reclaimable_resident_bytes=0 water_excess_bytes=0 "
                  "water_shortfall_bytes=0 water_overshoot_bytes=0 max_release_bytes=0 "
                  "target_clamp=none decision_reason=not_pressure "
                  "target_bytes=0 max_scan_blocks=256 "
                  "legacy_enabled=0 sample_count=1 episode=0 "
                  "cooldown_ms=0 skipped_reason=not_pressure idle=0 "
                  "mincore_before_bytes=0 mincore_after_bytes=0 "
                  "bounded_cnt_bytes_delta=0 bounded_cnt_blocks_delta=0 "
                  "can_enable=1 cap_paged=1 cap_ingraph=1 cap_layers=1 "
                  "cap_row_idx=1 cap_swap_disabled=1 cap_layout=1\n")
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "no bounded execution must fail")

    def test_ladder_no_mincore_drop_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed().replace(
                      "mincore_after_bytes=232783872", "mincore_after_bytes=268173312"))
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "no mincore drop must fail")

    def test_ladder_action_and_owned_noop_pass(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed() + self.bounded_noop("owned"),
            encoding="utf-8")
        result = self._run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ladder_state_noop_pass(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed() + self.bounded_noop("state"),
            encoding="utf-8")
        result = self._run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ladder_scan_exhausted_noop_pass(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed() + self.bounded_noop("scan"),
            encoding="utf-8")
        result = self._run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ladder_partial_action_with_shortfall_passes(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        stderr = self.telemetry() + self.bounded_executed().replace(
            "shortfall_bytes=0 overshoot_bytes=1835008", "shortfall_bytes=1048576 overshoot_bytes=0")
        d.joinpath("server.stderr").write_text(stderr, encoding="utf-8")
        d.joinpath("stderr_windows.json").unlink()
        result = self._run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_ladder_all_noops_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_noop("owned"), encoding="utf-8")
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no real bounded-release action", result.stderr)

    def test_ladder_action_counter_delta_mismatch_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed().replace(
                "bounded_cnt_bytes_delta=35389440", "bounded_cnt_bytes_delta=1"),
            encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_ladder_noop_nonzero_counter_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed() + self.bounded_noop().replace(
                "bounded_cnt_blocks_delta=0", "bounded_cnt_blocks_delta=1"),
            encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_ladder_madvise_failure_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed().replace(
                "madvise_failures=0", "madvise_failures=1"), encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_ladder_mapping_capability_violation_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed().replace(
                "cap_layout=1", "cap_layout=0"), encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_ladder_duplicate_marker_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed() + self.bounded_executed(),
            encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_ladder_out_of_order_marker_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        out_of_order = self.bounded_noop().replace(
            "sample_count=6 episode=2", "sample_count=4 episode=0")
        d.joinpath("server.stderr").write_text(
            self.telemetry() + self.bounded_executed() + out_of_order,
            encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_ladder_duplicate_marker_field_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        malformed = self.bounded_executed().rstrip("\n") + " released_bytes=33554432\n"
        d.joinpath("server.stderr").write_text(self.telemetry() + malformed, encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_ladder_residual_process_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_off"
        (d / "phases.json").write_text(
            json.dumps({"shutdown": {"residual_process": True}}, indent=2) + "\n",
            encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "residual process must fail")

    # ── continuous: response identity + round numbering ────────────────────

    def test_continuous_not_all_identical_rejected(self) -> None:
        cont_result = json.loads(
            (self.art / "continuous_requests" / "result.json").read_text())
        cont_result["all_responses_identical"] = False
        cont_result["cumulative_error_count"] = 3
        (self.art / "continuous_requests" / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "non-identical continuous responses must fail")

    def test_continuous_insufficient_rounds_rejected(self) -> None:
        cont_result = json.loads(
            (self.art / "continuous_requests" / "result.json").read_text())
        cont_result["completed_rounds"] = 3
        cont_result["rounds"] = cont_result["rounds"][:3]
        (self.art / "continuous_requests" / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "insufficient rounds must fail")

    def test_continuous_round_gap_rejected(self) -> None:
        cont_result = json.loads(
            (self.art / "continuous_requests" / "result.json").read_text())
        cont_result["rounds"][5]["round"] = 99  # gap in numbering
        (self.art / "continuous_requests" / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "round gap must fail")

    def test_continuous_duplicate_round_rejected(self) -> None:
        cont_result = json.loads(
            (self.art / "continuous_requests" / "result.json").read_text())
        cont_result["rounds"][5]["round"] = 5  # duplicate
        (self.art / "continuous_requests" / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "duplicate rounds must fail")

    def test_continuous_http_error_rejected(self) -> None:
        cont_result = json.loads(
            (self.art / "continuous_requests" / "result.json").read_text())
        cont_result["rounds"][5]["http_status"] = 500
        cont_result["rounds"][5]["matches_baseline"] = False
        cont_result["cumulative_error_count"] += 1
        (self.art / "continuous_requests" / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "HTTP error must fail")

    # ── continuous: KV_PAGED_RELEASE_STATS enforcement ─────────────────────

    def test_continuous_no_release_stats_rejected(self) -> None:
        d = self.art / "continuous_requests"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed())
        # Missing KV_PAGED_RELEASE_STATS entirely
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "missing KV_PAGED_RELEASE_STATS must fail")

    def test_continuous_reuse_allocations_zero_rejected(self) -> None:
        d = self.art / "continuous_requests"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed() +
                  self.release_stats(reuse=0, release_calls=20))
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "reuse_allocations=0 must fail")

    def test_continuous_write_rollbacks_nonzero_rejected(self) -> None:
        d = self.art / "continuous_requests"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed() +
                  "KV_PAGED_RELEASE_STATS reuse_allocations=5 write_commits=5 "
                  "write_rollbacks=1 dummy_candidate_pending_write_cell=0 "
                  "released_redirect_no_dummy=0 released_redirect_no_dummy_pending_write=0 "
                  "ensure_pending_write_rejected=0 input_setup_fatal=0 "
                  "row_mapping_fatal=0 write_mapping_fatal=0 "
                  "active_nonresident_fatal=0 "
                  "bounded_release_calls=20 bounded_release_bytes=35389440 "
                  "bounded_release_blocks=9\n")
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "write_rollbacks>0 must fail")

    def test_continuous_fatal_nonzero_rejected(self) -> None:
        d = self.art / "continuous_requests"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed() +
                  "KV_PAGED_RELEASE_STATS reuse_allocations=5 write_commits=5 "
                  "write_rollbacks=0 dummy_candidate_pending_write_cell=0 "
                  "released_redirect_no_dummy=0 released_redirect_no_dummy_pending_write=0 "
                  "ensure_pending_write_rejected=0 input_setup_fatal=1 "
                  "row_mapping_fatal=0 write_mapping_fatal=0 "
                  "active_nonresident_fatal=0 "
                  "bounded_release_calls=20 bounded_release_bytes=35389440 "
                  "bounded_release_blocks=9\n")
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "input_setup_fatal>0 must fail")

    def test_continuous_bounded_release_calls_zero_rejected(self) -> None:
        d = self.art / "continuous_requests"
        stderr = (self.telemetry(state="CRITICAL") +
                  self.bounded_executed() +
                  self.release_stats(reuse=20, release_calls=0))
        (d / "server.stderr").write_text(stderr, encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "bounded_release_calls=0 must fail")

    def test_continuous_write_commits_zero_rejected(self) -> None:
        d = self.art / "continuous_requests"
        stderr = ("KV_PAGED_RELEASE_STATS reuse_allocations=5 write_commits=0 "
                  "write_rollbacks=0 dummy_candidate_pending_write_cell=0 "
                  "released_redirect_no_dummy=0 released_redirect_no_dummy_pending_write=0 "
                  "ensure_pending_write_rejected=0 input_setup_fatal=0 "
                  "row_mapping_fatal=0 write_mapping_fatal=0 "
                  "active_nonresident_fatal=0 "
                  "bounded_release_calls=20 bounded_release_bytes=35389440 "
                  "bounded_release_blocks=9\n")
        # Recreate continuous with this stderr
        shutil.rmtree(self.art / "continuous_requests")
        self._write_continuous(stderr=(self.telemetry(state="CRITICAL") +
                                        self.bounded_executed() + stderr))
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "write_commits=0 must fail")

    def test_continuous_residual_process_rejected(self) -> None:
        d = self.art / "continuous_requests"
        (d / "phases.json").write_text(
            json.dumps({"shutdown": {"residual_process": True}}, indent=2) + "\n",
            encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "continuous residual process must fail")

    # ── stderr window validation ───────────────────────────────────────────

    def test_stderr_window_no_markers_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        # Put non-marker text in the stderr window
        clean_stderr = "some random log output\nno markers here\n"
        (d / "server.stderr").write_text(clean_stderr, encoding="utf-8")
        (d / "stderr_windows.json").write_text(
            json.dumps({"windows": [{"round": 1, "start_byte": 0,
                                      "end_byte": len(clean_stderr.encode()),
                                      "start_timestamp_s": 0.0, "end_timestamp_s": 1.0,
                                      "duration_s": 1.0}],
                       "request_count": 1}, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "stderr window without markers must fail")

    # ── RSS consistency: reject distorted samples ─────────────────────────────

    def test_continuous_rss_far_below_calibration_rejected(self) -> None:
        """RSS 2008 KiB vs calibration idle 6,000,000 KiB → reject (wrong process)."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        # Inject the bug symptom: RSS sampled from strace wrapper instead of server
        for r in cont_result["rounds"]:
            r["rss_after_request_kb"] = 2008
            r["rss_after_release_kb"] = 1800
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "RSS far below calibration idle must be rejected")

    def test_continuous_rss_just_above_absolute_floor_still_rejected(self) -> None:
        """RSS 50,001 KiB with 6,000,000 KiB idle — ratio still < 0.1 → reject."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_after_request_kb"] = 50001
            r["rss_after_release_kb"] = 50001
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "RSS below ratio threshold must be rejected")

    def test_continuous_rss_within_consistency_passes(self) -> None:
        """RSS close to calibration idle should pass (not distorted)."""
        # Default fixture already has rss_after_request_kb=6200000, idle=6000000
        # Just verify it passes
        r = self._run_parser()
        self.assertEqual(r.returncode, 0,
                        f"RSS within consistency range should PASS\nstderr={r.stderr}")

    def test_ladder_rss_far_below_calibration_rejected(self) -> None:
        """Ladder case with RSS far below calibration → reject."""
        d = self.art / "ladder" / "t1024_dynamic_release"
        result = json.loads((d / "result.json").read_text())
        result["rss_after_request_kb"] = 2008
        result["rss_after_release_kb"] = 1800
        (d / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "ladder RSS far below calibration must be rejected")

    # ── PID identity: positive (valid identity passes) ────────────────────────

    def _add_valid_pid_identity_to_fixture(self) -> None:
        """Add valid rss_pid_identity to continuous_requests rounds."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = {
                "pid": 51234,
                "starttime": 9876543,
                "cmdline": "/path/to/build/bin/llama-server --host 127.0.0.1",
            }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")

    def test_continuous_valid_pid_identity_passes(self) -> None:
        """PID identity with valid pid, starttime, and cmdline should pass."""
        self._add_valid_pid_identity_to_fixture()
        r = self._run_parser()
        self.assertEqual(r.returncode, 0,
                        f"valid PID identity should PASS\nstderr={r.stderr}")

    def test_ladder_valid_pid_identity_passes(self) -> None:
        """Ladder case with valid PID identity should pass."""
        d = self.art / "ladder" / "t1024_dynamic_release"
        result = json.loads((d / "result.json").read_text())
        result["rss_pid_identity"] = {
            "pid": 45123,
            "starttime": 1234567,
            "cmdline": "/build/llama-server --host 127.0.0.1 --port 8123",
        }
        (d / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertEqual(r.returncode, 0,
                        f"ladder valid PID identity should PASS\nstderr={r.stderr}")

    # ── PID identity: negative (invalid identity rejected) ────────────────────

    def test_continuous_pid_cmdline_contains_strace_rejected(self) -> None:
        """PID identity cmdline containing 'strace' → sampled the wrong process."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = {
                "pid": 1234,
                "starttime": 5678,
                "cmdline": "strace -f -e trace=madvise -o /tmp/strace.log -- "
                           "/build/llama-server --host 127.0.0.1",
            }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "pid_identity with strace cmdline must be rejected")

    def test_continuous_pid_zero_rejected(self) -> None:
        """PID identity with pid=0 must be rejected."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = {
                "pid": 0,
                "starttime": 5678,
                "cmdline": "/build/llama-server",
            }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "pid=0 must be rejected")

    def test_continuous_pid_starttime_zero_rejected(self) -> None:
        """PID identity with starttime=0 must be rejected."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = {
                "pid": 55123,
                "starttime": 0,
                "cmdline": "/build/llama-server",
            }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "starttime=0 must be rejected")

    def test_continuous_pid_cmdline_empty_rejected(self) -> None:
        """PID identity with empty cmdline must be rejected."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = {
                "pid": 55123,
                "starttime": 5678,
                "cmdline": "",
            }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0, "empty cmdline must be rejected")

    def test_continuous_pid_cmdline_no_llama_no_path_rejected(self) -> None:
        """PID identity cmdline that doesn't look like a server binary → reject."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = {
                "pid": 55123,
                "starttime": 5678,
                "cmdline": "some random process",
            }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "cmdline without server binary must be rejected")

    # ── PID identity: cross-round consistency ────────────────────────────────

    def test_continuous_pid_cross_round_inconsistent_rejected(self) -> None:
        """Different PID across rounds → server may have restarted."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        base_ident = {
            "pid": 55123,
            "starttime": 5678,
            "cmdline": "/build/llama-server --host 127.0.0.1",
        }
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = dict(base_ident)
        # Inject a different PID in round 10
        cont_result["rounds"][9]["rss_pid_identity"] = {
            "pid": 55999,
            "starttime": 9999,
            "cmdline": "/build/llama-server --host 127.0.0.1",
        }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "PID change across rounds must be rejected")

    def test_continuous_pid_starttime_change_across_rounds_rejected(self) -> None:
        """Same PID but different starttime → PID was reused."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        base_ident = {
            "pid": 55123,
            "starttime": 5678,
            "cmdline": "/build/llama-server --host 127.0.0.1",
        }
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = dict(base_ident)
        # Same PID but different starttime in round 5
        cont_result["rounds"][4]["rss_pid_identity"] = {
            "pid": 55123,  # same PID
            "starttime": 88888,  # different starttime → PID was reused!
            "cmdline": "/build/llama-server --host 127.0.0.1",
        }
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "PID reuse (different starttime) must be rejected")

    def test_continuous_pid_identity_not_dict_rejected(self) -> None:
        """rss_pid_identity that is not a dict → must fail validation."""
        d = self.art / "continuous_requests"
        cont_result = json.loads((d / "result.json").read_text())
        for r in cont_result["rounds"]:
            r["rss_pid_identity"] = "not_a_dict"
        (d / "result.json").write_text(
            json.dumps(cont_result, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "non-dict pid_identity must be rejected")


class Stage3B2ATimeoutDerivationTest(Stage3B2AParserTest):
    """Per-tier completion timeout must be derived from each tier's real RSS
    calibration completion wall-clock — not the historical fixed 180s.  Positive
    (consistent derivation passes) and negative (missing/inconsistent fails)."""

    def test_rss_calib_timeout_consistency_passes(self) -> None:
        # Default fixture: 1024 tier wall=10s→30s, 2048 tier wall=20s→50s.
        r = self._run_parser()
        self.assertEqual(r.returncode, 0,
                        f"consistent completion_timeout_s should PASS\nstderr={r.stderr}")

    def test_rss_calib_timeout_inconsistent_with_wall_rejected(self) -> None:
        # Recorded timeout that disagrees with derive(wall) must fail.
        calib = json.loads((self.art / "rss_calibration/rss_calibration.json").read_text())
        calib["levels"]["1024"]["completion_timeout_s"] = 999.0
        (self.art / "rss_calibration/rss_calibration.json").write_text(
            json.dumps(calib, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "completion_timeout_s inconsistent with wall-clock must fail")

    def test_rss_calib_wall_missing_rejected(self) -> None:
        # No real timing sample (runner failed to measure) → fail-closed.
        calib = json.loads((self.art / "rss_calibration/rss_calibration.json").read_text())
        del calib["levels"]["1024"]["completion_wall_s"]
        (self.art / "rss_calibration/rss_calibration.json").write_text(
            json.dumps(calib, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "missing completion_wall_s must fail (no real timing sample)")

    def test_rss_calib_recorded_timeout_missing_rejected(self) -> None:
        calib = json.loads((self.art / "rss_calibration/rss_calibration.json").read_text())
        del calib["levels"]["1024"]["completion_timeout_s"]
        (self.art / "rss_calibration/rss_calibration.json").write_text(
            json.dumps(calib, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "missing completion_timeout_s must fail")

    def test_rss_calib_failed_tier_rejected(self) -> None:
        # A tier whose calibration HTTP-400'd must not be silently usable.
        calib = json.loads((self.art / "rss_calibration/rss_calibration.json").read_text())
        calib["levels"]["2048"]["calibration_failed"] = True
        calib["levels"]["2048"]["http_status"] = 400
        (self.art / "rss_calibration/rss_calibration.json").write_text(
            json.dumps(calib, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "calibration_failed tier must be rejected (no fake-pass)")

    def test_ladder_completion_timeout_missing_rejected(self) -> None:
        d = self.art / "ladder" / "t1024_dynamic_release"
        res = json.loads((d / "result.json").read_text())
        del res["completion_timeout_s"]
        (d / "result.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "ladder case missing completion_timeout_s must fail")

    def test_continuous_completion_timeout_missing_rejected(self) -> None:
        d = self.art / "continuous_requests"
        res = json.loads((d / "result.json").read_text())
        del res["completion_timeout_s"]
        (d / "result.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "continuous missing completion_timeout_s must fail")

    def test_calibration_timeout_derivation_tamper_rejected(self) -> None:
        path = self.art / "rss_calibration/rss_calibration.json"
        data = json.loads(path.read_text())
        data["levels"]["2048"]["calibration_timeout_s"] = 999.0
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0)


class Stage3B2AIncompleteArtifactTest(Stage3B2AParserTest):
    def test_missing_continuous_result_is_controlled_fail_without_traceback(self) -> None:
        (self.art / "continuous_requests/result.json").unlink()
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("missing mandatory file result.json", result.stderr)

    def test_runner_incomplete_is_controlled_incomplete_without_traceback(self) -> None:
        manifest_path = self.art / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update({
            "runner_status": "run_incomplete",
            "failure_phase": "rss_calibration",
            "failure_target": 8064,
            "failure_reason": "completion timed out",
        })
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        summary_path = self.art / "summary.json"
        summary_path.write_text(json.dumps({
            "runner_status": "run_incomplete",
            "failure_phase": "rss_calibration",
            "failure_target": 8064,
            "failure_reason": "completion timed out",
        }, indent=2) + "\n", encoding="utf-8")
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)
        self.assertIn("INCOMPLETE", result.stderr)
        parser_result = json.loads((self.art / "parser.json").read_text())
        self.assertEqual(parser_result["status"], "INCOMPLETE")

    def test_current_real_incomplete_artifact_has_no_traceback(self) -> None:
        artifact = pathlib.Path(
            "/root/oscomp/kv_logs/kv_bounded_release_stage3b_2a_20260726T095924Z_"
            "3941949abe_006735dfcb80")
        if not artifact.is_dir():
            self.skipTest("current incomplete artifact not present")
        result = subprocess.run(
            [sys.executable, str(PARSER), str(artifact)],
            text=True, capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("Traceback", result.stderr)


class Stage3B2AEffectiveLifecycleTest(Stage3B2AParserTest):
    EFFECTIVE = [1024, 2048, 4096, 8064]
    REQUESTED = [1024, 2048, 4096, 8192]

    def setUp(self) -> None:
        super().setUp()
        token_file = self.art / "token_calibration/calibration.json"
        token_data = json.loads(token_file.read_text())
        rss_file = self.art / "rss_calibration/rss_calibration.json"
        rss_data = json.loads(rss_file.read_text())

        previous_target = 2048
        previous_wall = 20.0
        for target, wall in [(4096, 30.0), (8064, 80.0)]:
            prompt = "tier " * target
            token = {
                "target_tokens": target, "actual_tokens": target, "delta": 0,
                "prompt_text": prompt,
                "prompt_text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "seed_tokens_per_rep": 3, "repetitions": target // 3,
                "iterations": 1,
            }
            token_data["results"][str(target)] = token
            (self.art / f"token_calibration/prompt_{target}t.json").write_text(
                json.dumps(token, indent=2) + "\n", encoding="utf-8")

            timeout = max(wall * 2.5, 30.0)
            cal = dict(RSS_CALIB_2048)
            calibration_timeout = max(
                180.0, previous_wall * (target / previous_target) * 1.5)
            cal.update({
                "target_tokens": target, "source_target": target,
                "ctx_size": target + 128, "completion_wall_s": wall,
                "completion_timeout_s": timeout, "http_status": 200,
                "calibration_timeout_s": calibration_timeout,
                "calibration_timeout_mode": "previous_success_scaled",
                "calibration_timeout_bootstrap_s": 180.0,
                "calibration_timeout_previous_target": previous_target,
                "calibration_timeout_previous_wall_s": previous_wall,
                "calibration_timeout_token_ratio": target / previous_target,
                "calibration_timeout_safety_factor": 1.5,
                "calibration_timeout_scaled_s": previous_wall * (target / previous_target) * 1.5,
            })
            rss_data["levels"][str(target)] = cal
            level_dir = self.art / f"rss_calibration/ctx_{target}"
            level_dir.mkdir()
            (level_dir / "rss_calibration.json").write_text(
                json.dumps(cal, indent=2) + "\n", encoding="utf-8")

            result = {
                "response_text": BASELINE_RESPONSE,
                "response_text_sha256": BASELINE_SHA,
                "response_text_len": len(BASELINE_RESPONSE),
                "http_status": 200,
                "rss_after_request_kb": 6500000,
                "rss_after_release_kb": 6400000,
                "effective_n_ctx": 8192,
                "source_target": target,
                "legal_tier": target,
                "completion_timeout_s": timeout,
            }
            for label in ["OFF", "DYNAMIC_RELEASE"]:
                stderr = (self.telemetry(state="NORMAL") if label == "OFF" else
                          self.telemetry(state="CRITICAL") + self.bounded_executed() +
                          self.release_stats(reuse=3, release_calls=1))
                self._write_ladder_case(target, label, result_override=result, stderr=stderr)
            previous_target = target
            previous_wall = wall

        token_data["targets"] = self.EFFECTIVE
        token_file.write_text(json.dumps(token_data, indent=2) + "\n", encoding="utf-8")
        rss_file.write_text(json.dumps(rss_data, indent=2) + "\n", encoding="utf-8")

        manifest = json.loads((self.art / "manifest.json").read_text())
        manifest["token_targets"] = self.REQUESTED
        manifest["token_calibration"]["results"] = token_data["results"]
        manifest["rss_calibration"]["levels"] = rss_data["levels"]
        manifest["effective_context"].update({
            "requested_targets": self.REQUESTED,
            "effective_targets": self.EFFECTIVE,
            "clamped_tiers": [
                {"requested": 1024, "effective": 1024},
                {"requested": 2048, "effective": 2048},
                {"requested": 4096, "effective": 4096},
                {"requested": 8192, "effective": 8064},
            ],
            "probe": {
                "requested_ctx_size": 8320,
                "requested_targets": self.REQUESTED,
                "effective_n_ctx": 8192,
                "max_prompt_tokens": 8064,
                "effective_targets": self.EFFECTIVE,
                "completion_requests": 0,
                "http_completion_statuses": [],
            },
        })
        (self.art / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        probe_dir = self.art / "context_probe"
        (probe_dir / "probe.json").write_text(
            json.dumps(manifest["effective_context"]["probe"], indent=2) + "\n",
            encoding="utf-8")

    def test_full_effective_lifecycle_passes(self) -> None:
        result = self._run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_requested_8192_formal_calibration_rejected(self) -> None:
        path = self.art / "token_calibration/calibration.json"
        data = json.loads(path.read_text())
        data["results"]["8192"] = dict(data["results"]["8064"],
                                         target_tokens=8192, actual_tokens=8192)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_missing_8064_rss_calibration_rejected(self) -> None:
        path = self.art / "rss_calibration/rss_calibration.json"
        data = json.loads(path.read_text())
        del data["levels"]["8064"]
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_8064_reusing_4096_source_rejected(self) -> None:
        for label in ["off", "dynamic_release"]:
            path = self.art / f"ladder/t8064_{label}/result.json"
            data = json.loads(path.read_text())
            data["source_target"] = 4096
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_8064_non_200_calibration_rejected(self) -> None:
        path = self.art / "rss_calibration/rss_calibration.json"
        data = json.loads(path.read_text())
        data["levels"]["8064"]["http_status"] = 400
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_8064_timeout_from_other_tier_rejected(self) -> None:
        path = self.art / "ladder/t8064_dynamic_release/result.json"
        data = json.loads(path.read_text())
        data["completion_timeout_s"] = RSS_CALIB_2048["completion_timeout_s"]
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_probe_completion_activity_rejected(self) -> None:
        path = self.art / "manifest.json"
        data = json.loads(path.read_text())
        data["effective_context"]["probe"]["completion_requests"] = 1
        data["effective_context"]["probe"]["http_completion_statuses"] = [400]
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_probe_completion_artifact_rejected(self) -> None:
        (self.art / "context_probe/completion.sse").write_text("data: error\n")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_stale_8192_rss_directory_rejected(self) -> None:
        stale = self.art / "rss_calibration/ctx_8192"
        stale.mkdir()
        (stale / "rss_calibration.json").write_text("{}\n")
        self.assertNotEqual(self._run_parser().returncode, 0)

    def test_stale_8192_ladder_directory_rejected(self) -> None:
        (self.art / "ladder/t8192_off").mkdir()
        self.assertNotEqual(self._run_parser().returncode, 0)


class Stage3B2AContextOverflowTest(Stage3B2AParserTest):
    """The ladder must run inside the server's effective n_ctx.  The high tier
    must converge to a legal prompt count (e.g. 8064 for an 8192 model), not
    send a prompt that overflows it (HTTP 400 fake-pass)."""

    def test_effective_context_passes_when_within_budget(self) -> None:
        # Default fixture: effective_n_ctx=8192, max_prompt=8064, tiers fit.
        r = self._run_parser()
        self.assertEqual(r.returncode, 0,
                        f"tiers within effective context should PASS\nstderr={r.stderr}")

    def test_missing_effective_context_rejected(self) -> None:
        m = json.loads((self.art / "manifest.json").read_text())
        del m["effective_context"]
        (self.art / "manifest.json").write_text(
            json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "missing effective_context (runner did not probe n_ctx) must fail")

    def test_effective_context_nonpositive_n_ctx_rejected(self) -> None:
        m = json.loads((self.art / "manifest.json").read_text())
        m["effective_context"]["effective_n_ctx"] = 0
        (self.art / "manifest.json").write_text(
            json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "effective_n_ctx<=0 must fail (runner must probe server context)")

    def test_max_prompt_inconsistent_with_derive_rejected(self) -> None:
        m = json.loads((self.art / "manifest.json").read_text())
        # Tamper: max_prompt would be 8064 for n_ctx=8192, claim 8000 instead.
        m["effective_context"]["max_prompt_tokens"] = 8000
        (self.art / "manifest.json").write_text(
            json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "max_prompt_tokens inconsistent with derive(n_ctx,...) must fail")

    def test_legal_targets_above_max_prompt_rejected(self) -> None:
        m = json.loads((self.art / "manifest.json").read_text())
        # legal_targets contains a tier beyond max_prompt (the overflow bug).
        m["effective_context"]["max_prompt_tokens"] = 1500
        m["effective_context"]["legal_targets"] = [1024, 2048]
        (self.art / "manifest.json").write_text(
            json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "legal tier exceeding max_prompt_tokens must fail")

    def test_legal_targets_fewer_than_two_rejected(self) -> None:
        m = json.loads((self.art / "manifest.json").read_text())
        m["effective_context"]["effective_n_ctx"] = 200  # tiny context
        m["effective_context"]["max_prompt_tokens"] = 64
        m["effective_context"]["legal_targets"] = [64]
        (self.art / "manifest.json").write_text(
            json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "<2 legal tiers must fail (context too small for ladder)")

    def test_ladder_tier_exceeds_max_prompt_rejected(self) -> None:
        # The runner recorded a tier above max_prompt_tokens — i.e. it failed
        # to converge and ran an overflowing prompt.
        m = json.loads((self.art / "manifest.json").read_text())
        m["effective_context"]["max_prompt_tokens"] = 1500
        (self.art / "manifest.json").write_text(
            json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "ladder tier exceeding max_prompt_tokens must fail")

    def test_ladder_prompt_plus_generation_exceeds_n_ctx_rejected(self) -> None:
        # A single tier whose actual prompt tokens + overhead + n_predict +
        # safety > effective_n_ctx must fail as a context-overflow risk.
        d = self.art / "ladder" / "t2048_dynamic_release"
        res = json.loads((d / "result.json").read_text())
        # actual_tokens for 2048 tier is 2049 (tolerance 1).  used =
        # 2049 + 1 + 32 + 95 = 2177.  Shrink the recorded case effective_n_ctx
        # below used so the invariant breaches — without touching token calib.
        res["effective_n_ctx"] = 1100
        (d / "result.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertIn("context overflow risk", r.stderr,
                      f"expected context-overflow check to reject\nstderr={r.stderr}")
        self.assertNotEqual(r.returncode, 0,
                           "prompt+generation exceeding effective n_ctx must fail")

    def test_ladder_http_400_rejected(self) -> None:
        # An overflowed prompt that slot returned HTTP 400 must not fake-pass.
        d = self.art / "ladder" / "t1024_dynamic_release"
        res = json.loads((d / "result.json").read_text())
        res["http_status"] = 400
        res["response_text"] = ""
        res["response_text_len"] = 0
        (d / "result.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        r = self._run_parser()
        self.assertNotEqual(r.returncode, 0,
                           "ladder HTTP 400 (context overflow) must fail, not fake-pass")


if __name__ == "__main__":
    unittest.main()
