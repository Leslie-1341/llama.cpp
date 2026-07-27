#!/usr/bin/env python3
"""Directory-ownership contract tests for Stage 3B-2A runner's continuous phase.

The regression guard for the FileExistsError that arose from `main` calling
`loop_dir.mkdir(exist_ok=False)` before handing `loop_dir` to
`run_continuous_requests`, which also mkdir's the same directory (its own
output dir) with `exist_ok=False`. The fix is to let
`run_continuous_requests` remain the sole creator of its output directory,
preserving fail-closed behavior. These tests do not start a server; they mock
the process/network layer exactly like `test-runner-bounded-release-stage3a-2c.py`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import shutil
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run-kv-bounded-release-stage3b-2a.py"
SPEC = importlib.util.spec_from_file_location("stage3b_2a_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = 0
        self.stdout = None
        self.stderr = None
        self.stage3b_2a_pgid = pid

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class ContinueousDirOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage3b_2a_runner_dir_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_continuous_requests_creates_output_dir_when_absent(self) -> None:
        """loop_dir does not exist → run_continuous_requests creates it and succeeds."""
        loop_dir = self.tmp / "continuous_requests"
        self.assertFalse(loop_dir.exists())
        proc = FakeProcess()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner.subprocess, "Popen", return_value=proc))
            stack.enter_context(mock.patch.object(runner.os, "getpgid", return_value=proc.pid))
            stack.enter_context(mock.patch.object(runner, "kill_server"))
            stack.enter_context(mock.patch.object(runner, "wait_health"))
            stack.enter_context(mock.patch.object(runner, "check_capability"))
            stack.enter_context(mock.patch.object(
                runner, "stream_completion",
                side_effect=[(200, "deterministic")] * 5))
            stack.enter_context(mock.patch.object(runner, "get_server_rss_kb", return_value=6_000_000))
            stack.enter_context(mock.patch.object(runner.time, "sleep"))
            stack.enter_context(mock.patch.object(
                runner.os, "killpg",
                side_effect=[None, *(ProcessLookupError,) * 3]))
            result = runner.run_continuous_requests(
                "CONTINUOUS_REQUESTS", "/bin/true", "/tmp/model.gguf", 8123,
                runner.controlled_env({"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"}),
                loop_dir, runner.Deadline(120),
                ctx_size=1280, calibrated_prompt="prompt",
                num_requests=5, strace=False)
        self.assertTrue(loop_dir.exists())
        self.assertEqual(result["completed_rounds"], 5)
        self.assertTrue(result["all_responses_identical"])
        # environment.json proves the directory was created by the runner itself.
        self.assertTrue((loop_dir / "environment.json").exists())

    def test_run_continuous_requests_fails_closed_when_output_dir_exists(self) -> None:
        """loop_dir already exists → mkdir(exist_ok=False) raises FileExistsError.

        This is the fail-closed contract: the runner must refuse to reuse a
        pre-existing artifact directory rather than silently overwrite it. A
        pre-created loop_dir is exactly the file-exists condition that the
        removed main-level mkdir used to trigger, and which must now be observed
        only at the function's own mkdir boundary.
        """
        loop_dir = self.tmp / "continuous_requests"
        loop_dir.mkdir(parents=False, exist_ok=False)
        self.assertTrue(loop_dir.exists())

        proc = FakeProcess()
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(runner.subprocess, "Popen", return_value=proc))
            stack.enter_context(mock.patch.object(runner.os, "getpgid", return_value=proc.pid))
            stack.enter_context(mock.patch.object(runner, "kill_server"))
            stack.enter_context(mock.patch.object(runner, "wait_health"))
            stack.enter_context(mock.patch.object(runner, "check_capability"))
            stack.enter_context(mock.patch.object(
                runner, "stream_completion",
                side_effect=[(200, "deterministic")] * 5))
            stack.enter_context(mock.patch.object(runner, "get_server_rss_kb", return_value=6_000_000))
            stack.enter_context(mock.patch.object(runner.time, "sleep"))
            stack.enter_context(mock.patch.object(
                runner.os, "killpg",
                side_effect=[None, *(ProcessLookupError,) * 3]))
            with self.assertRaises(FileExistsError):
                runner.run_continuous_requests(
                    "CONTINUOUS_REQUESTS", "/bin/true", "/tmp/model.gguf", 8123,
                    runner.controlled_env({"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"}),
                    loop_dir, runner.Deadline(120),
                    ctx_size=1280, calibrated_prompt="prompt",
                    num_requests=5, strace=False)
        # Directory was not wiped or repopulated by the runner.
        self.assertEqual(list(loop_dir.iterdir()), [])

    def test_main_does_not_pre_create_loop_dir_before_delegate(self) -> None:
        """Static contract: main must not mkdir loop_dir before delegating.

        Guards against re-introducing the dup-mkdir. We assert that the source
        contains no `loop_dir.mkdir` call anywhere other than (implicitly) inside
        run_continuous_requests via its `output_dir.mkdir`.
        """
        src = RUNNER_PATH.read_text(encoding="utf-8")
        # `loop_dir` is bound in main; ensure no pre-create call in main.
        # The only legal directory-creation for loop_dir is the function's own
        # `output_dir.mkdir(parents=False, exist_ok=False)` (where output_dir IS
        # loop_dir at the call site). Assert main's literal `loop_dir.mkdir`:
        self.assertNotIn("loop_dir.mkdir", src,
                         "main must not pre-create loop_dir; run_continuous_requests "
                         "is the sole creator of its output directory")
        # And confirm the delegate still expects to create it itself.
        self.assertIn("output_dir.mkdir(parents=False, exist_ok=False)", src)


class DerivedTimeoutTest(unittest.TestCase):
    """Per-tier completion timeout must be derived from real measured timing,
    not the historical fixed 180s.  Floor + margin + fallback behavior."""

    def test_timeout_uses_margin_above_floor(self) -> None:
        # 10s wall × 2.5 = 25s, floored to 30s
        self.assertAlmostEqual(runner.derive_completion_timeout(10.0), 30.0)

    def test_timeout_uses_margin_when_above_floor(self) -> None:
        # 50s wall × 2.5 = 125s, above floor
        self.assertAlmostEqual(runner.derive_completion_timeout(50.0), 125.0)

    def test_timeout_nonpositive_measured_falls_back(self) -> None:
        # No valid timing sample (e.g. calibration HTTP 400) → fallback.
        # This is an explicit error condition, not a happy path.
        self.assertEqual(runner.derive_completion_timeout(0.0),
                         runner.COMPLETION_TIMEOUT_FALLBACK_S)
        self.assertEqual(runner.derive_completion_timeout(-1.0),
                         runner.COMPLETION_TIMEOUT_FALLBACK_S)

    def test_timeout_not_fixed_180_for_slow_tier(self) -> None:
        # A 100s wall must NOT return 180 — it must return 250 (100×2.5).
        self.assertNotAlmostEqual(runner.derive_completion_timeout(100.0), 180.0)
        self.assertAlmostEqual(runner.derive_completion_timeout(100.0), 250.0)


class CalibrationTimeoutDerivationTest(unittest.TestCase):
    def test_first_tier_uses_explicit_bootstrap(self) -> None:
        result = runner.derive_calibration_timeout(1024)
        self.assertEqual(result["calibration_timeout_mode"], "bootstrap")
        self.assertEqual(result["calibration_timeout_s"], 180.0)
        self.assertIsNone(result["calibration_timeout_previous_target"])

    def test_later_tier_scales_previous_success(self) -> None:
        result = runner.derive_calibration_timeout(4096, 2048, 85.0)
        self.assertEqual(result["calibration_timeout_mode"], "previous_success_scaled")
        self.assertEqual(result["calibration_timeout_token_ratio"], 2.0)
        self.assertEqual(result["calibration_timeout_s"], 255.0)

    def test_real_8064_timeout_is_significantly_above_180(self) -> None:
        result = runner.derive_calibration_timeout(
            8064, 4096, 182.0764746620007)
        self.assertGreater(result["calibration_timeout_s"], 500.0)
        self.assertAlmostEqual(result["calibration_timeout_s"],
                               182.0764746620007 * (8064 / 4096) * 1.5)

    def test_invalid_previous_success_rejected(self) -> None:
        with self.assertRaises(ValueError):
            runner.derive_calibration_timeout(2048, 1024, 0.0)


class RunnerFailureStatusTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage3b_2a_failure_"))
        self.old_artifact = runner._ACTIVE_ARTIFACT
        self.old_manifest = runner._ACTIVE_MANIFEST
        self.old_failure = runner._FIRST_FAILURE
        self.old_reason = runner._LAST_FAILURE_REASON
        runner._ACTIVE_ARTIFACT = self.tmp
        runner._ACTIVE_MANIFEST = {"protocol": "test"}
        runner._FIRST_FAILURE = None
        runner._LAST_FAILURE_REASON = None

    def tearDown(self) -> None:
        runner._ACTIVE_ARTIFACT = self.old_artifact
        runner._ACTIVE_MANIFEST = self.old_manifest
        runner._FIRST_FAILURE = self.old_failure
        runner._LAST_FAILURE_REASON = self.old_reason
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_persist_incomplete_records_first_failure(self) -> None:
        runner.set_failure_context("rss_calibration", 8064)
        runner.mark_failure("completion timed out")
        runner.persist_runner_status("run_incomplete")
        manifest = __import__("json").loads((self.tmp / "manifest.json").read_text())
        summary = __import__("json").loads((self.tmp / "summary.json").read_text())
        for value in [manifest, summary]:
            self.assertEqual(value["runner_status"], "run_incomplete")
            self.assertEqual(value["failure_phase"], "rss_calibration")
            self.assertEqual(value["failure_target"], 8064)
            self.assertEqual(value["failure_reason"], "completion timed out")


class EffectiveContextDerivationTest(unittest.TestCase):
    """The high tier must converge to a legal value (≈8064 for an 8192 model),
    not send a prompt that overflows the model context.  Fail-closed when the
    model context cannot hold any tier."""

    def test_max_prompt_converges_to_8064_for_8192_model(self) -> None:
        # 8192 - 32(n_predict) - 1(special) - 95(safety) = 8064, aligned by 32.
        self.assertEqual(runner.derive_max_prompt_tokens(8192, 32), 8064)

    def test_max_prompt_zero_when_context_too_small(self) -> None:
        # n_predict(32)+overhead(1)+safety(95)=128 > effective ctx of 100 → 0.
        self.assertEqual(runner.derive_max_prompt_tokens(100, 32), 0)

    def test_clamp_drops_overflowing_high_tier_to_8064(self) -> None:
        legal = runner.clamp_token_targets([1024, 2048, 4096, 8192], 8064)
        # 8192 clamps to 8064; lower tiers kept as-is.
        self.assertEqual(legal, [1024, 2048, 4096, 8064])

    def test_clamp_below_256_dropped(self) -> None:
        # A target collapsing below the 256 floor must be dropped, not sent.
        legal = runner.clamp_token_targets([128, 1024, 8192], 8064)
        self.assertEqual(legal, [1024, 8064])
        self.assertNotIn(128, legal)

    def test_clamp_too_small_context_returns_empty(self) -> None:
        # max_prompt <=0 → no legal ladder at all → caller must fail-closed.
        self.assertEqual(runner.clamp_token_targets([1024, 8192], 0), [])

    def test_clamp_collapses_duplicate_converged_tiers(self) -> None:
        # Two identical requested tiers must not appear twice in the legal ladder.
        legal = runner.clamp_token_targets([8064, 8064], 8064)
        self.assertEqual(legal, [8064])

    def test_probe_effective_n_ctx_extracts_capped_value(self) -> None:
        stderr = ("0.03 I srv: loading\n"
                  "0.05 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 8192\n")
        self.assertEqual(runner.probe_effective_n_ctx(stderr), 8192)

    def test_probe_effective_n_ctx_none_when_marker_absent(self) -> None:
        # If the server never logged the slot marker (did not load) the runner
        # must fail-closed rather than assume a context.
        self.assertIsNone(runner.probe_effective_n_ctx("no marker here\n"))

    def test_probe_effective_n_ctx_takes_max_for_multi_slot(self) -> None:
        stderr = ("0.05 I slot   load_model: id  0 | new slot, n_ctx = 4096\n"
                  "0.06 I slot   load_model: id  1 | new slot, n_ctx = 8192\n")
        self.assertEqual(runner.probe_effective_n_ctx(stderr), 8192)


if __name__ == "__main__":
    unittest.main()
