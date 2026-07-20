#!/usr/bin/env python3
"""Synthetic negative tests for Stage 3A-2B dry-run parser."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts/parse-kv-dry-run-stage3a-2b.py"

# Minimal valid artifact scaffolding
def _write_case(dir: pathlib.Path, name: str, variant: str,
                stderr_lines: list[str], response_text: str = "hello",
                env_extra: dict | None = None,
                strace_files: list[str] | None = None,
                strace_returncode: int = 0) -> None:
    case = dir / name
    case.mkdir(parents=True, exist_ok=True)
    base_env = {
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_RELEASE": "0",
        "LLAMA_KV_PRESSURE_SAMPLER": "1",
        "LLAMA_KV_LOW_WATER_RSS_KB": "1",
        "LLAMA_KV_PRESSURE_RSS_KB": "2",
        "LLAMA_KV_CRITICAL_RSS_KB": "3",
        "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "500",
        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
        "TZ": "UTC", "TMPDIR": "/tmp",
    }
    if variant == "ON":
        base_env["LLAMA_KV_PRESSURE_DRY_RUN"] = "1"
        base_env["LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES"] = "4194304"
        base_env["LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS"] = "64"
        base_env["LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS"] = "500"
    if env_extra:
        base_env.update(env_extra)
    (case / "environment.json").write_text(json.dumps(base_env))
    (case / "server.stderr").write_text("\n".join(stderr_lines) + "\n")
    (case / "result.json").write_text(json.dumps({
        "variant": variant, "response_text": response_text,
        "response_length": len(response_text), "http_status": 200,
    }))
    (case / "phases.json").write_text(json.dumps({
        "shutdown": {"residual_process": False},
    }))
    if strace_files is not None:
        for sf in strace_files:
            (case / sf).write_text("")
        (case / "strace_process.json").write_text(json.dumps({
            "returncode": strace_returncode, "attached_to_pid": 12345,
        }))


class DryRunParserNegativeTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.art = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_parser(self) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(PARSER), str(self.art)],
            capture_output=True, text=True)

    # ------------------------------------------------------------------
    # Valid baseline
    # ------------------------------------------------------------------
    def test_valid_artifact_passes(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ])
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  kv_pressure_telemetry state=CRITICAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL stale=0 release_enabled=0 "
            "would_release_bytes=524288 would_release_blocks=1 "
            "blocks_scanned=3 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 "
            "skipped_reason=none cooldown_ms=500 sample_count=1 idle=0",
        ], strace_files=["strace.1"])
        result = self._run_parser()
        self.assertEqual(result.returncode, 0,
                         f"valid artifact should pass: {result.stderr[-500:]}")

    # ------------------------------------------------------------------
    # Missing ON markers
    # ------------------------------------------------------------------
    def test_on_zero_dry_run_markers_fails(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ])
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  kv_pressure_telemetry state=CRITICAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ], strace_files=["strace.1"])
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "zero dry-run markers in ON must fail")

    # ------------------------------------------------------------------
    # Configuration pollution: LLAMA_KV_PAGED_RELEASE=1 in ON
    # ------------------------------------------------------------------
    def test_config_pollution_release_enabled_fails(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ])
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  kv_pressure_telemetry state=CRITICAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ], env_extra={"LLAMA_KV_PAGED_RELEASE": "1"})  # forbidden
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "LLAMA_KV_PAGED_RELEASE=1 must fail configuration check")

    # ------------------------------------------------------------------
    # OFF must not have dry_run markers
    # ------------------------------------------------------------------
    def test_off_has_dry_run_markers_fails(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL stale=0 release_enabled=0 "
            "would_release_bytes=100 would_release_blocks=1 "
            "blocks_scanned=1 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 "
            "skipped_reason=none cooldown_ms=500 sample_count=1 idle=0",
        ])
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL stale=0 release_enabled=0 "
            "would_release_bytes=524288 would_release_blocks=1 "
            "blocks_scanned=3 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 "
            "skipped_reason=none cooldown_ms=500 sample_count=1 idle=0",
        ], strace_files=["strace.1"])
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "OFF with dry_run markers must fail")

    # ------------------------------------------------------------------
    # Duplicate sample_count in ON markers
    # ------------------------------------------------------------------
    def test_on_duplicate_sample_count_fails(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ])
        marker = (
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL stale=0 release_enabled=0 "
            "would_release_bytes=524288 would_release_blocks=1 "
            "blocks_scanned=3 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 "
            "skipped_reason=none cooldown_ms=500 sample_count=1 idle=0"
        )
        _write_case(self.art, "dry_run_on", "ON",
                    [marker, marker],  # duplicate sample_count=1
                    strace_files=["strace.1"])
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "duplicate sample_count must fail")

    # ------------------------------------------------------------------
    # Non-increasing sample_count in ON markers
    # ------------------------------------------------------------------
    def test_on_sample_count_not_increasing_fails(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ])
        m1 = (
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL release_enabled=0 "
            "would_release_bytes=524288 would_release_blocks=1 "
            "blocks_scanned=3 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 stale=0 "
            "skipped_reason=none cooldown_ms=500 sample_count=3 idle=0"
        )
        m2 = (
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL release_enabled=0 "
            "would_release_bytes=524288 would_release_blocks=1 "
            "blocks_scanned=3 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 stale=0 "
            "skipped_reason=none cooldown_ms=500 sample_count=2 idle=0"
        )  # sample_count=2 after 3 → not increasing
        _write_case(self.art, "dry_run_on", "ON",
                    [m1, m2], strace_files=["strace.1"])
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "non-increasing sample_count must fail")

    # ------------------------------------------------------------------
    # Destructive release evidence (non-zero counter)
    # ------------------------------------------------------------------
    def test_nonzero_destructive_counter_fails(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ])
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  kv_pressure_telemetry state=CRITICAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
            "srv  paged_block_release_bytes=1048576",  # non-zero → evidence
        ], strace_files=["strace.1"])
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "non-zero destructive counter must fail")

    # ------------------------------------------------------------------
    # Zero-value destructive counter is fine (diagnostic)
    # ------------------------------------------------------------------
    def test_zero_destructive_counter_passes(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ])
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  kv_pressure_telemetry state=CRITICAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL stale=0 release_enabled=0 "
            "would_release_bytes=524288 would_release_blocks=1 "
            "blocks_scanned=3 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 "
            "skipped_reason=none cooldown_ms=500 sample_count=1 idle=0",
            "srv  paged_block_release_bytes=0",   # zero → diagnostic, not evidence
            "srv  paged_blocks_released=0",       # zero → diagnostic, not evidence
        ], strace_files=["strace.1"])
        result = self._run_parser()
        self.assertEqual(result.returncode, 0,
                         f"zero-value counters must be ignored: {result.stderr[-300:]}")

    # ------------------------------------------------------------------
    # Response mismatch
    # ------------------------------------------------------------------
    def test_response_mismatch_fails(self):
        _write_case(self.art, "dry_run_off", "OFF", [
            "srv  kv_pressure_telemetry state=NORMAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ], response_text="answer_A")
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  kv_pressure_telemetry state=CRITICAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
        ], response_text="answer_B", strace_files=["strace.1"])
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "response mismatch must fail")

    # ------------------------------------------------------------------
    # Missing case directory
    # ------------------------------------------------------------------
    def test_missing_off_case_fails(self):
        _write_case(self.art, "dry_run_on", "ON", [
            "srv  kv_pressure_telemetry state=CRITICAL source=RSS_ABSOLUTE stale=0 "
            "sample_valid=1 config_valid=1 rss_kb=99999 sample_count=1 idle=0 trigger=first",
            "srv  maybe_sample: kv_pressure_dry_run source=RSS_ABSOLUTE state=CRITICAL stale=0 release_enabled=0 "
            "would_release_bytes=524288 would_release_blocks=1 "
            "blocks_scanned=3 blocks_skipped_owned=0 blocks_skipped_state=0 "
            "shortfall_bytes=0 overshoot_bytes=0 "
            "block_scan_exhausted=0 ownership_aborted=0 "
            "target_bytes=4194304 max_scan_blocks=64 "
            "skipped_reason=none cooldown_ms=500 sample_count=1 idle=0",
        ], strace_files=["strace.1"])
        result = self._run_parser()
        self.assertNotEqual(result.returncode, 0,
                            "missing OFF case must fail")


if __name__ == "__main__":
    unittest.main()
