#!/usr/bin/env python3
"""Branch-executing tests for the Stage 3A-2C runner."""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run-kv-bounded-release-stage3a-2c.py"
SPEC = importlib.util.spec_from_file_location("stage3a_2c_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


class FakeProcess:
    def __init__(self, pid: int = 4321) -> None:
        self.pid = pid
        self.returncode = 0
        self.stdout = None
        self.stderr = None
        self.stage3a_2c_pgid = pid

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class RunnerBranchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage3a_2c_runner_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_start_server_calls_popen_with_recorded_argv_and_exact_environment(self) -> None:
        case = self.tmp / "case"
        case.mkdir()
        env = runner.build_controlled_env()
        proc = FakeProcess()
        with mock.patch.object(runner.subprocess, "Popen", return_value=proc) as popen, \
             mock.patch.object(runner.os, "getpgid", return_value=4321):
            actual = runner.start_server("/bin/true", 8123, "/tmp/model.gguf", env, case, strace=True)
        self.assertIs(actual, proc)
        args, kwargs = popen.call_args
        execution = json.loads((case / "execution.json").read_text())
        self.assertEqual(args[0], execution["argv"])
        self.assertEqual(kwargs["env"], execution["environment"])
        self.assertEqual(kwargs["env"], env)
        self.assertIn("--parallel", args[0])
        self.assertEqual(args[0][args[0].index("--parallel") + 1], "1")
        self.assertIn("--cache-ram", args[0])
        self.assertEqual(args[0][args[0].index("--cache-ram") + 1], "0")
        self.assertIn("--no-warmup", args[0])
        kwargs["stdout"].close()
        kwargs["stderr"].close()

    def test_run_case_exception_branch_writes_incomplete_result_and_shutdown(self) -> None:
        proc = FakeProcess()
        case = self.tmp / "case"
        with mock.patch.object(runner, "start_server", return_value=proc), \
             mock.patch.object(runner, "wait_health", side_effect=SystemExit(7)), \
             mock.patch.object(runner, "kill_server"), \
             mock.patch.object(runner.time, "sleep"), \
             mock.patch.object(runner.os, "killpg", side_effect=ProcessLookupError):
            with self.assertRaises(SystemExit):
                runner.run_case("OFF", "/bin/true", "/tmp/model", 8123,
                                runner.build_controlled_env(), case, runner.Deadline(10))
        result = json.loads((case / "result.json").read_text())
        failure = json.loads((case / "failure.json").read_text())
        shutdown = json.loads((case / "phases.json").read_text())["shutdown"]
        self.assertEqual(result["case_status"], "incomplete")
        self.assertTrue(failure["case_failed"])
        self.assertTrue(shutdown["pgid_check_complete"])
        self.assertFalse(shutdown["residual_process"])

    def test_run_case_residual_branch_kills_saved_pgid_then_records_clean(self) -> None:
        proc = FakeProcess(pid=8765)
        case = self.tmp / "case"
        killpg = mock.Mock(side_effect=[None, None, ProcessLookupError])
        with mock.patch.object(runner, "start_server", return_value=proc), \
             mock.patch.object(runner, "wait_health"), \
             mock.patch.object(runner, "stream_completion", return_value=(200, "same")), \
             mock.patch.object(runner, "kill_server"), \
             mock.patch.object(runner.time, "sleep"), \
             mock.patch.object(runner.os, "killpg", killpg):
            result = runner.run_case("OFF", "/bin/true", "/tmp/model", 8123,
                                     runner.build_controlled_env(), case, runner.Deadline(10))
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(killpg.call_args_list, [
            mock.call(8765, 0), mock.call(8765, signal.SIGKILL), mock.call(8765, 0)])
        shutdown = json.loads((case / "phases.json").read_text())["shutdown"]
        self.assertTrue(shutdown["cleanup_kill_attempted"])
        self.assertTrue(shutdown["pgid_check_complete"])
        self.assertFalse(shutdown["residual_process"])

    def test_run_case_pgid_probe_error_fails_closed(self) -> None:
        proc = FakeProcess(pid=8765)
        case = self.tmp / "case"
        with mock.patch.object(runner, "start_server", return_value=proc), \
             mock.patch.object(runner, "wait_health"), \
             mock.patch.object(runner, "stream_completion", return_value=(200, "same")), \
             mock.patch.object(runner, "kill_server"), \
             mock.patch.object(runner.time, "sleep"), \
             mock.patch.object(runner.os, "killpg", side_effect=PermissionError):
            runner.run_case("OFF", "/bin/true", "/tmp/model", 8123,
                            runner.build_controlled_env(), case, runner.Deadline(10))
        shutdown = json.loads((case / "phases.json").read_text())["shutdown"]
        self.assertFalse(shutdown["pgid_check_complete"])
        self.assertTrue(shutdown["residual_process"])

    def test_capture_source_snapshot_copies_tracked_diff_and_untracked_bytes(self) -> None:
        repo = self.tmp / "repo"
        art = self.tmp / "artifact"
        repo.mkdir()
        art.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.email", "fixture@example.invalid"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "Fixture"], cwd=repo, check=True)
        tracked = repo / "tracked.txt"
        tracked.write_text("before\n")
        subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
        tracked.write_text("after\n")
        untracked = repo / "nested" / "new.py"
        untracked.parent.mkdir()
        untracked.write_bytes(b"untracked bytes\x00\n")
        with mock.patch.object(runner, "ROOT", repo):
            snapshot = runner.capture_source_snapshot(art)
        self.assertGreater(snapshot["tracked_diff"]["size"], 0)
        self.assertEqual([item["path"] for item in snapshot["untracked_files"]], ["nested/new.py"])
        copied = pathlib.Path(snapshot["untracked_files"][0]["snapshot_path"])
        self.assertEqual(copied.read_bytes(), untracked.read_bytes())

    def test_parser_nonzero_is_propagated_without_verification(self) -> None:
        failed = subprocess.CompletedProcess([], 7, "", "parser failed")
        with mock.patch.object(runner.subprocess, "run", return_value=failed) as run:
            code = runner.run_parser_protocol(self.tmp)
        self.assertEqual(code, 7)
        self.assertEqual(run.call_count, 1)

    def test_parser_pass_still_requires_saved_result_verification(self) -> None:
        passed = subprocess.CompletedProcess([], 0, "", "")
        failed_verify = subprocess.CompletedProcess([], 9, "", "closure failed")
        with mock.patch.object(runner.subprocess, "run", side_effect=[passed, failed_verify]) as run:
            code = runner.run_parser_protocol(self.tmp)
        self.assertEqual(code, 9)
        self.assertEqual(run.call_count, 2)

    def test_controlled_environment_does_not_inherit_host_knobs(self) -> None:
        with mock.patch.dict(os.environ, {"LLAMA_KV_PAGED_SWAP": "1", "PYTHONPATH": "polluted"}):
            env = runner.build_controlled_env()
        self.assertEqual(env["LLAMA_KV_PAGED_SWAP"], "0")
        self.assertNotIn("PYTHONPATH", env)


if __name__ == "__main__":
    unittest.main()
