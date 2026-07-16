#!/usr/bin/env python3
"""Parser, runner, and non-dry executor regression for the frozen E2G/E2I protocol."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/kv-paged-identity-controlled-ab.sh"
RUNNER = ROOT / "scripts/kv-paged-identity-controlled-ab-runner.py"
PARSER = ROOT / "scripts/parse-kv-paged-identity-controlled-ab.py"


FAKE_BINARY = r'''#!/usr/bin/python3
import os, sys
case = "E2I" if os.environ["LLAMA_KV_PAGED_IDENTITY_FAST_PATH"] == "1" else "E2G"
print("===SEQ1_ACTIVE_BEGIN===")
print("paired-seq1")
print("===SEQ1_ACTIVE_END===")
print("===SEQ0_RESUME_BEGIN===")
print("paired-seq0")
print("===SEQ0_RESUME_END===")
fast = case == "E2I"
tpot = 90 if fast else 100
tps = 11 if fast else 10
p95 = 100 if fast else 110
wall = 900 if fast else 1000
print("KV_TEST_SUMMARY result=PASS decode_calls=10 expected_swap_out_io_failure=0 prefetch_failures_observed=0 active_decode_failures_observed=0 active_decode_retries=0 active_decode_retry_successes=0", file=sys.stderr)
print(f"KV_IDLE_SWAP_RESUME_PERF total_wall_ms={wall} tokens_per_second={tps}", file=sys.stderr)
print(f"KV_ACTIVE_TOKEN_STATS active_token_count=10 avg_ms={tpot} p95_ms={p95}", file=sys.stderr)
print("KV_GRAPH_REUSE_STATS n_reused=4", file=sys.stderr)
safety = [
 "identity_fail", "write_resolve_fail", "row_idx_fail", "mapping_oob_fail", "logical_to_physical_fail",
 "paged_row_mapping_invalid_fatal", "paged_write_mapping_invalid_fatal", "paged_active_row_nonresident_fatal",
 "paged_input_setup_fatal", "paged_swapped_active_visible_violation", "paged_swapped_active_visible_violation_rows",
 "paged_swapped_active_visible_violation_blocks", "paged_write_to_swapped_block", "paged_swap_backend_failures",
 "paged_swap_read_swap_in_failures", "paged_swap_write_swap_in_failures", "paged_swap_in_fail_no_offset",
 "paged_swap_in_fail_bad_size", "paged_swap_in_fail_read_cell", "paged_swap_in_fail_tensor_set",
 "paged_prefetch_seq_failures", "paged_block_release_fail", "paged_swap_madvise_failures"]
values = {name: 0 for name in safety}
values.update({
 "paged_swap_enabled": 0, "paged_idle_swap_enabled": 0, "paged_block_release_enabled": 0,
 "paged_identity_fast_path_enabled": int(fast), "paged_identity_fast_path_layers": 2 if fast else 0,
 "paged_identity_fast_path_reject_reason": "none" if fast else "not_requested",
 "ingraph_gather_layers": 0 if fast else 4, "paged_row_idx_inputs_created": 0 if fast else 2,
 "paged_row_idx_set_calls": 0 if fast else 8, "non_identity_enabled": 0})
print("KV paged metadata stats: " + " ".join(f"{key}={value}" for key, value in values.items()), file=sys.stderr)
'''


def git(*args: str, cwd: pathlib.Path) -> str:
    return subprocess.check_output(["git", "-C", str(cwd), *args], text=True).strip()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProtocolFixture:
    def __init__(self, dry_run: bool = False):
        self.temp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.temp.name)
        self.repo = self.base / "repo"
        self.output = self.base / "output"
        scripts = self.repo / "scripts"
        scripts.mkdir(parents=True)
        for source in (WRAPPER, RUNNER, PARSER):
            target = scripts / source.name
            shutil.copy2(source, target)
            target.chmod(0o755)
        self.binary = self.repo / "fake-binary"
        self.binary.write_text(FAKE_BINARY)
        self.binary.chmod(0o755)
        self.model = self.repo / "model.gguf"
        self.model.write_bytes(b"model")
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "fixture@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Fixture"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True)
        env = dict(os.environ)
        env.update({"ROOT": str(self.repo), "BINARY": str(self.binary), "MODEL": str(self.model),
                    "OUTPUT_ROOT": str(self.output), "DRY_RUN": "1" if dry_run else "0",
                    "ALLOW_DIRTY": "1" if dry_run else "0"})
        self.result = subprocess.run([str(scripts / WRAPPER.name)], env=env, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    def close(self) -> None:
        self.temp.cleanup()

    def run(self, round_no: int, case: str) -> pathlib.Path:
        return next(path for path in (self.output / "runs").iterdir()
                    if (meta := json.loads((path / "run.json").read_text()))["round"] == round_no
                    and meta["case"] == case)

    def refresh(self, path: pathlib.Path) -> None:
        manifest_path = self.output / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        meta = json.loads((path / "run.json").read_text())
        key = meta["round"], meta["run_order"], meta["case"]
        for run in manifest["completed_runs"]:
            if (run["round"], run["run_order"], run["case"]) == key:
                run["artifacts"] = {name: {"path": str(path / name), "size": (path / name).stat().st_size,
                                                    "sha256": sha256(path / name)}
                                    for name in run["artifacts"]}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    def parse(self, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(self.repo / "scripts" / PARSER.name)]
        if dry_run:
            command.append("--dry-run")
        command.append(str(self.output))
        return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


class ControlledProtocolTest(unittest.TestCase):
    def fixture(self, dry_run: bool = False) -> ProtocolFixture:
        fixture = ProtocolFixture(dry_run)
        self.addCleanup(fixture.close)
        return fixture

    def test_formal_fake_run_is_valid_and_reports_frozen_statistics(self) -> None:
        fixture = self.fixture()
        self.assertEqual(fixture.result.returncode, 0, fixture.result.stderr)
        summary = json.loads((fixture.output / "summary.json").read_text())
        self.assertEqual(summary["artifact_status"], "VALID")
        self.assertEqual(summary["performance_judgment"], "FAST_PATH_FASTER")
        self.assertEqual(summary["metrics"], ["tpot_ms", "tps", "p95_ms", "wall_ms"])
        self.assertEqual(len(summary["paired_rounds"]), 3)
        for metric in summary["metrics"]:
            self.assertEqual(set(summary["aggregate"][metric]["E2G"]), {"median", "min", "max"})
            self.assertTrue(summary["aggregate"][metric]["direction_consistency"]["all_rounds_favorable"])

    def test_e2g_e2i_are_single_variable_and_use_empty_allowlisted_environment(self) -> None:
        fixture = self.fixture()
        self.assertEqual(fixture.result.returncode, 0, fixture.result.stderr)
        for round_no in range(1, 4):
            executions = {case: json.loads((fixture.run(round_no, case) / "execution.json").read_text())
                          for case in ("E2G", "E2I")}
            left, right = executions["E2G"], executions["E2I"]
            self.assertEqual(left["argv"], right["argv"])
            self.assertEqual(left["cwd"], right["cwd"])
            self.assertEqual(left["timeout"], right["timeout"])
            self.assertEqual(left["auxiliary"], right["auxiliary"])
            left_env, right_env = dict(left["env"]), dict(right["env"])
            self.assertEqual(left_env.pop("LLAMA_KV_PAGED_IDENTITY_FAST_PATH"), "0")
            self.assertEqual(right_env.pop("LLAMA_KV_PAGED_IDENTITY_FAST_PATH"), "1")
            self.assertEqual(left_env, right_env)
            self.assertEqual(left_env["LLAMA_KV_PAGED_GATHER_NONIDENTITY"], "0")
            self.assertNotIn("USER", left_env)

    def test_fixed_order_and_execution_json_derived_artifacts(self) -> None:
        fixture = self.fixture()
        self.assertEqual(fixture.result.returncode, 0, fixture.result.stderr)
        observed = [(item["round"], item["run_order"], item["case"])
                    for item in json.loads((fixture.output / "manifest.json").read_text())["planned_runs"]]
        self.assertEqual(observed, [(1, 1, "E2G"), (1, 2, "E2I"), (2, 1, "E2I"),
                                    (2, 2, "E2G"), (3, 1, "E2G"), (3, 2, "E2I")])
        for path in (fixture.output / "runs").iterdir():
            execution = json.loads((path / "execution.json").read_text())
            self.assertEqual(json.loads((path / "environment.json").read_text()), execution["env"])
            replay = (path / "replay.sh").read_text()
            self.assertIn("env -i", replay)
            self.assertNotIn("os.environ.copy", replay)

    def test_rss_is_optional_and_absent_from_validity_and_metrics(self) -> None:
        fixture = self.fixture()
        self.assertEqual(fixture.result.returncode, 0, fixture.result.stderr)
        for path in (fixture.output / "runs").iterdir():
            self.assertEqual((path / "rss_samples.tsv").read_text().count("\n"), 1)
        summary = json.loads((fixture.output / "summary.json").read_text())
        self.assertFalse(any("rss" in metric.lower() for metric in summary["metrics"]))

    def test_performance_judgment_does_not_change_artifact_validity(self) -> None:
        fixture = self.fixture()
        path = fixture.run(2, "E2I")
        stderr = path / "stderr"
        stderr.write_text(stderr.read_text().replace("avg_ms=90", "avg_ms=190")
                          .replace("p95_ms=100", "p95_ms=200")
                          .replace("total_wall_ms=900", "total_wall_ms=1900")
                          .replace("tokens_per_second=11", "tokens_per_second=1"))
        fixture.refresh(path)
        result = fixture.parse()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((fixture.output / "summary.json").read_text())
        self.assertEqual(summary["artifact_status"], "VALID")
        self.assertEqual(summary["performance_judgment"], "MIXED")

    def test_dry_run_validates_both_rotations_without_starting_binary(self) -> None:
        fixture = self.fixture(dry_run=True)
        self.assertEqual(fixture.result.returncode, 0, fixture.result.stderr)
        self.assertEqual(json.loads((fixture.output / "summary.json").read_text())["artifact_status"], "DRY_RUN")
        self.assertFalse(any(path.name == "observed.json" for path in fixture.output.rglob("*")))

    def test_unique_pass_summary_is_required(self) -> None:
        fixture = self.fixture()
        path = fixture.run(1, "E2I")
        stderr = path / "stderr"
        stderr.write_text(stderr.read_text().replace("KV_TEST_SUMMARY result=PASS ", ""))
        fixture.refresh(path)
        self.assertNotEqual(fixture.parse().returncode, 0)

    def test_identity_mapping_row_write_and_backend_fields_fail_closed(self) -> None:
        for field in ("identity_fail", "mapping_oob_fail", "row_idx_fail", "write_resolve_fail",
                      "logical_to_physical_fail", "paged_swap_backend_failures"):
            with self.subTest(field=field):
                fixture = ProtocolFixture()
                try:
                    path = fixture.run(2, "E2G")
                    stderr = path / "stderr"
                    stderr.write_text(stderr.read_text().replace(f"{field}=0", f"{field}=1", 1))
                    fixture.refresh(path)
                    self.assertNotEqual(fixture.parse().returncode, 0)
                finally:
                    fixture.close()

    def test_e2g_and_e2i_mechanism_proofs_fail_closed(self) -> None:
        for case, old, new in (("E2G", "ingraph_gather_layers=4", "ingraph_gather_layers=0"),
                               ("E2I", "paged_identity_fast_path_enabled=1", "paged_identity_fast_path_enabled=0"),
                               ("E2I", "paged_row_idx_set_calls=0", "paged_row_idx_set_calls=1")):
            with self.subTest(case=case, old=old):
                fixture = ProtocolFixture()
                try:
                    path = fixture.run(3, case)
                    stderr = path / "stderr"
                    stderr.write_text(stderr.read_text().replace(old, new, 1))
                    fixture.refresh(path)
                    self.assertNotEqual(fixture.parse().returncode, 0)
                finally:
                    fixture.close()

    def test_output_environment_and_artifact_tampering_fail_closed(self) -> None:
        for mutation in ("output", "environment", "artifact"):
            with self.subTest(mutation=mutation):
                fixture = ProtocolFixture()
                try:
                    path = fixture.run(1, "E2I")
                    if mutation == "output":
                        (path / "seq0").write_text("different\n")
                        fixture.refresh(path)
                    elif mutation == "environment":
                        env = json.loads((path / "environment.json").read_text())
                        env["POLLUTED"] = "1"
                        (path / "environment.json").write_text(json.dumps(env))
                        fixture.refresh(path)
                    else:
                        with (path / "stderr").open("a") as handle:
                            handle.write("tampered\n")
                    self.assertNotEqual(fixture.parse().returncode, 0)
                finally:
                    fixture.close()


class ExecutorIntegrationTest(unittest.TestCase):
    def execute(self, body: str, timeout: float = 2.0, grace: float = 0.2) -> tuple[pathlib.Path, dict, tempfile.TemporaryDirectory]:
        temp = tempfile.TemporaryDirectory()
        root = pathlib.Path(temp.name)
        binary = root / "fake.py"
        binary.write_text("#!/usr/bin/python3\n" + textwrap.dedent(body))
        binary.chmod(0o755)
        observed = root / "observed.json"
        env = {"LANG": "C", "LC_ALL": "C", "OBSERVED": str(observed), "ONLY_ALLOWED": "yes"}
        execution = {"cwd": str(root), "binary": str(binary), "argv": [str(binary), "arg one"], "env": env,
                     "timeout": {"seconds": timeout, "term_signal": "TERM", "termination_grace_seconds": grace},
                     "auxiliary": {"rss": {"enabled": False, "sample_interval_seconds": 0.1}}}
        path = root / "execution.json"
        path.write_text(json.dumps(execution))
        result = subprocess.run([sys.executable, str(RUNNER), "--execute", str(path)], check=False)
        self.assertEqual(result.returncode, 0)
        return root, execution, temp

    def test_real_argv_env_and_normal_exit_are_reconciled(self) -> None:
        root, execution, temp = self.execute('''
import json, os, pathlib, sys
pathlib.Path(os.environ["OBSERVED"]).write_text(json.dumps({"argv": sys.argv, "env": dict(os.environ)}))
''')
        self.addCleanup(temp.cleanup)
        observed = json.loads((root / "observed.json").read_text())
        self.assertEqual(observed, {"argv": execution["argv"], "env": execution["env"]})
        result = json.loads((root / "execution_result.json").read_text())
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["timed_out"])
        self.assertEqual((root / "exit_code").read_text(), "0\n")
        process = json.loads((root / "process.json").read_text())
        self.assertEqual(process["env"], execution["env"])
        self.assertEqual(process["argv"], execution["argv"])

    def test_nonzero_exit_is_recorded(self) -> None:
        root, _, temp = self.execute("raise SystemExit(7)\n")
        self.addCleanup(temp.cleanup)
        result = json.loads((root / "execution_result.json").read_text())
        self.assertEqual(result["returncode"], 7)
        self.assertFalse(result["term_sent"])
        self.assertEqual((root / "exit_code").read_text(), "7\n")

    def test_timeout_sends_term(self) -> None:
        root, _, temp = self.execute('''
import signal, time
def stop(*_):
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
while True: time.sleep(0.05)
''', timeout=0.2)
        self.addCleanup(temp.cleanup)
        result = json.loads((root / "execution_result.json").read_text())
        self.assertTrue(result["timed_out"])
        self.assertTrue(result["term_sent"])
        self.assertFalse(result["kill_sent"])
        self.assertEqual(result["returncode"], 0)

    def test_timeout_kills_after_grace(self) -> None:
        root, _, temp = self.execute('''
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
while True: time.sleep(0.05)
''', timeout=0.2, grace=0.2)
        self.addCleanup(temp.cleanup)
        result = json.loads((root / "execution_result.json").read_text())
        self.assertTrue(result["term_sent"])
        self.assertTrue(result["kill_sent"])
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        self.assertEqual((root / "exit_code").read_text(), f"{-signal.SIGKILL}\n")


if __name__ == "__main__":
    unittest.main()
