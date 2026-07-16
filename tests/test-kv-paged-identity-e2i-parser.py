#!/usr/bin/env python3

import importlib.util
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-kv-paged-identity-e2i.py"
RUNNER = ROOT / "scripts" / "kv-paged-identity-e2i.sh"
CASES = ("E0", "E2", "E2I", "E2I_NOREUSE", "SHIFT", "NONIDENTITY", "SWAP", "RELEASE", "MADVISE")
REJECT = {
    "E2": "not_requested", "SHIFT": "non_identity_mapping", "NONIDENTITY": "dynamic_remap",
    "SWAP": "swap", "RELEASE": "release", "MADVISE": "madvise",
}
SPEC = importlib.util.spec_from_file_location("kv_paged_identity_parser", PARSER)
assert SPEC is not None and SPEC.loader is not None
PARSER_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARSER_MODULE)


def metadata(case_id: str) -> str:
    if case_id == "E0":
        return "KV_GRAPH_REUSE_STATS n_reused=3\n"
    fast = int(case_id in {"E2I", "E2I_NOREUSE"})
    gather = 0 if fast else 4
    reason = "none" if fast else REJECT[case_id]
    values = {
        "enabled": 1,
        "ingraph_gather_layers": gather,
        "paged_identity_fast_path_enabled": fast,
        "paged_identity_fast_path_layers": 2 if fast else 0,
        "paged_identity_fast_path_reject_reason": reason,
        "paged_row_idx_inputs_created": 0 if fast else 1,
        "paged_row_idx_set_calls": 0 if fast else 2,
        "paged_row_mapping_invalid_fatal": 0,
        "paged_write_mapping_invalid_fatal": 0,
        "paged_active_row_nonresident_fatal": 0,
        "paged_input_setup_fatal": 0,
        "paged_swapped_active_visible_violation": 0,
        "paged_write_to_swapped_block": 3 if case_id == "SWAP" else 0,
        "paged_swap_backend_failures": 0,
        "paged_swap_in_calls": 5 if case_id == "SWAP" else 0,
        "paged_swap_write_swapped_hits": 3 if case_id == "SWAP" else 0,
        "paged_swap_write_swap_in_calls": 3 if case_id == "SWAP" else 0,
        "paged_swap_write_swap_in_failures": 0,
    }
    n_reused = 0 if case_id == "E2I_NOREUSE" else 3
    return ("llama_kv_cache: KV paged metadata stats: " +
            " ".join(f"{key}={value}" for key, value in values.items()) +
            f"\nKV_GRAPH_REUSE_STATS n_reused={n_reused}\n")


def identity(path: pathlib.Path, include_path: bool = False) -> dict[str, object]:
    result: dict[str, object] = {
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if include_path:
        result["path"] = str(path)
    return result


def execution_doc(source: pathlib.Path, binary: pathlib.Path, env: dict[str, str]) -> dict[str, object]:
    return {
        "cwd": str(source),
        "binary": str(binary),
        "argv": [str(binary), "--fixture"],
        "env_set": dict(sorted(env.items())),
        "env_unset": sorted((
            "LLAMA_FLEX", "LLAMA_FLEX_AHEAD", "LLAMA_FLEX_AUTO", "LLAMA_FLEX_BUFFERED",
            "LLAMA_FLEX_DEBUG", "LLAMA_FLEX_LOCK_GB", "LLAMA_FLEX_RING", "LLAMA_FLEX_THREADS",
            "LLAMA_FLEX_TYPE", "LLAMA_FLEX_FACTOR", "LLAMA_FLEX_ORIG_CTX",
        )),
        "timeout": {"seconds": 900, "term_signal": "TERM", "kill_after_seconds": 10},
    }


def repo_identity(repo: pathlib.Path) -> dict[str, object]:
    status = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).splitlines()
    return {
        "path": str(repo),
        "head": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def write_fixture(root: pathlib.Path) -> None:
    source = root / "source"
    source.mkdir()
    binary = source / "fixture-binary"
    model = source / "fixture-model"
    runner = source / RUNNER.name
    parser = source / PARSER.name
    binary.write_bytes(b"binary")
    model.write_bytes(b"model")
    runner.write_bytes(RUNNER.read_bytes())
    parser.write_bytes(PARSER.read_bytes())
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.email", "fixture@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    manifest = {
        "protocol": "kv_paged_identity_e2i", "version": 1,
        "repo": repo_identity(source),
        "binary": identity(binary, include_path=True),
        "model": identity(model, include_path=True),
        "framework": {
            "runner": identity(runner, include_path=True),
            "parser": identity(parser, include_path=True),
        },
        "workload": {"common_args": ["--fixture"]},
        "execution": {"dry_run": False, "timeout_sec": 900},
        "planned_runs": [{"order": index, "case": case_id} for index, case_id in enumerate(CASES, 1)],
        "completed_runs": [],
    }
    for index, case_id in enumerate(CASES, 1):
        run = root / "runs" / case_id
        run.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"order": index, "case": case_id}))
        env = dict(PARSER_MODULE.BASE_ENV)
        env.update(PARSER_MODULE.CASE_ENV[case_id])
        env["LLAMA_KV_SWAP_DIR"] = str((root / "swap" / case_id).resolve())
        (run / "environment").write_text("".join(f"{key}={value}\n" for key, value in env.items()))
        execution = execution_doc(source, binary, env)
        (run / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
        (run / "command").write_text(PARSER_MODULE.normalized_command(execution))
        (run / "stdout").write_text(
            "===SEQ1_ACTIVE_BEGIN===\nsame-seq1\n===SEQ1_ACTIVE_END===\n"
            "===SEQ0_RESUME_BEGIN===\nsame-seq0\n===SEQ0_RESUME_END===\n"
        )
        (run / "stderr").write_text(metadata(case_id))
        (run / "exit_code").write_text("0\n")
        (run / "seq0").write_bytes(b"same-seq0\n")
        (run / "seq1").write_bytes(b"same-seq1\n")
        (run / "sequence.sha256").write_text("fixture sequence hashes\n")
        complete = {"order": index, "case": case_id}
        complete["artifacts"] = {name: identity(run / name) for name in PARSER_MODULE.RAW_ARTIFACTS}
        manifest["completed_runs"].append(complete)
    (root / "manifest.json").write_text(json.dumps(manifest))


def refresh(root: pathlib.Path, case_id: str) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    run = root / "runs" / case_id
    for item in manifest["completed_runs"]:
        if item["case"] == case_id:
            item["artifacts"] = {name: identity(run / name) for name in PARSER_MODULE.RAW_ARTIFACTS}
            break
    (root / "manifest.json").write_text(json.dumps(manifest))


class ParserTest(unittest.TestCase):
    def run_parser(self, root: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PARSER), *args, str(root)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    def fixture(self) -> tuple[tempfile.TemporaryDirectory[str], pathlib.Path]:
        temp = tempfile.TemporaryDirectory()
        root = pathlib.Path(temp.name)
        write_fixture(root)
        return temp, root

    def test_valid_matrix_with_swap_write_reconciliation_passes(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        result = self.run_parser(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((root / "summary.json").read_text())["result"], "PASS")

    def test_dry_run_passes_without_telemetry(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        for case_id in CASES:
            (root / "runs" / case_id / "stderr").write_text("")
            (root / "runs" / case_id / "exit_code").write_text("DRY_RUN\n")
            refresh(root, case_id)
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["execution"]["dry_run"] = True
        (root / "manifest.json").write_text(json.dumps(manifest))
        result = self.run_parser(root, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fast_path_row_input_creation_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "E2I" / "stderr"
        path.write_text(path.read_text().replace("paged_row_idx_inputs_created=0", "paged_row_idx_inputs_created=1"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_shift_wrong_reject_reason_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "SHIFT" / "stderr"
        path.write_text(path.read_text().replace("non_identity_mapping", "none"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_fallback_without_gather_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "SWAP" / "stderr"
        path.write_text(path.read_text().replace("ingraph_gather_layers=4", "ingraph_gather_layers=0"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_swap_write_counter_mismatch_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "SWAP" / "stderr"
        path.write_text(path.read_text().replace("paged_swap_write_swapped_hits=3",
                                                 "paged_swap_write_swapped_hits=2"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_swap_write_failure_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "SWAP" / "stderr"
        path.write_text(path.read_text().replace("paged_swap_write_swap_in_calls=3",
                                                 "paged_swap_write_swap_in_calls=2")
                                         .replace("paged_swap_write_swap_in_failures=0",
                                                  "paged_swap_write_swap_in_failures=1"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_non_swap_write_to_swapped_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "E2I" / "stderr"
        path.write_text(path.read_text().replace("paged_write_to_swapped_block=0",
                                                 "paged_write_to_swapped_block=1"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_output_mismatch_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        (root / "runs" / "E2I_NOREUSE" / "seq1").write_bytes(b"different\n")
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_duplicate_telemetry_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "NONIDENTITY" / "stderr"
        path.write_text(path.read_text() + path.read_text())
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_graph_reuse_environment_mismatch_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "E2I_NOREUSE" / "environment"
        path.write_text(path.read_text().replace("LLAMA_GRAPH_REUSE_DISABLE=1", "LLAMA_GRAPH_REUSE_DISABLE=0"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_missing_environment_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "E2I" / "environment"
        path.write_text(path.read_text().replace("LLAMA_KV_PAGED_INGRAPH=1\n", ""))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_dynamic_environment_pollution_fails_closed(self) -> None:
        mutations = (
            ("SHIFT", "LLAMA_KV_PAGED_SHIFT=1", "LLAMA_KV_PAGED_SHIFT=0"),
            ("NONIDENTITY", "LLAMA_KV_PAGED_GATHER_NONIDENTITY=1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY=0"),
            ("SWAP", "LLAMA_KV_PAGED_SWAP=1", "LLAMA_KV_PAGED_SWAP=0"),
            ("RELEASE", "LLAMA_KV_PAGED_RELEASE=1", "LLAMA_KV_PAGED_RELEASE=0"),
            ("MADVISE", "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1", "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0"),
            ("E2I", "LLAMA_KV_PAGED_INGRAPH=1", "LLAMA_KV_PAGED_INGRAPH=0"),
            ("E2I", None, "LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE=read"),
            ("E2I", "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0", "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1"),
        )
        for case_id, old, new in mutations:
            with self.subTest(case_id=case_id, polluted=new):
                temp, root = self.fixture()
                try:
                    path = root / "runs" / case_id / "environment"
                    content = path.read_text()
                    if old is None:
                        content += new + "\n"
                    else:
                        content = content.replace(old, new)
                    path.write_text(content)
                    self.assertNotEqual(self.run_parser(root).returncode, 0)
                finally:
                    temp.cleanup()

    def test_execution_command_argv_or_env_mismatch_fails_closed(self) -> None:
        mutations = ("command", "argv", "env_set", "timeout")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                temp, root = self.fixture()
                try:
                    run = root / "runs" / "E2I"
                    if mutation == "command":
                        (run / "command").write_text("true\n")
                    else:
                        execution = json.loads((run / "execution.json").read_text())
                        if mutation == "argv":
                            execution["argv"] = [execution["binary"], "--wrong"]
                        elif mutation == "env_set":
                            execution["env_set"]["LLAMA_KV_PAGED"] = "0"
                        else:
                            execution["timeout"]["seconds"] = 1
                        (run / "execution.json").write_text(json.dumps(execution, indent=2, sort_keys=True) + "\n")
                    refresh(root, "E2I")
                    self.assertNotEqual(self.run_parser(root).returncode, 0)
                finally:
                    temp.cleanup()

    def test_runner_clears_inherited_profiler_and_stability_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            model = root / "model"
            model.write_bytes(b"model")
            output = root / "output"
            env = dict(os.environ)
            env.update({
                "ROOT": str(ROOT), "MODEL": str(model), "BINARY": "/bin/true",
                "OUTPUT_ROOT": str(output), "DRY_RUN": "1",
                "LLAMA_KV_E2_GET_ROWS_PROFILE": "1",
                "LLAMA_KV_STABILITY_CYCLES": "999",
                "LLAMA_KV_STABILITY_FUTURE_KNOB": "polluted",
                "LLAMA_GRAPH_RESULT_DEBUG": "1", "GGML_SCHED_DEBUG": "1",
            })
            result = subprocess.run([str(RUNNER)], env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            for run in (output / "runs").iterdir():
                recorded = (run / "environment").read_text()
                command = (run / "command").read_text()
                self.assertIn("LLAMA_KV_E2_GET_ROWS_PROFILE=0\n", recorded)
                self.assertNotIn("LLAMA_KV_STABILITY_", recorded)
                self.assertIn("LLAMA_KV_STABILITY_CYCLES", command)
                self.assertIn("LLAMA_KV_STABILITY_FUTURE_KNOB", command)
                self.assertIn("LLAMA_GRAPH_RESULT_DEBUG", command)
                self.assertIn("GGML_SCHED_DEBUG", command)

    def test_reuse_enabled_without_reuse_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "E2I" / "stderr"
        path.write_text(path.read_text().replace("n_reused=3", "n_reused=0"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_reuse_disabled_with_reuse_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        path = root / "runs" / "E2I_NOREUSE" / "stderr"
        path.write_text(path.read_text().replace("n_reused=0", "n_reused=1"))
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_artifact_mutation_without_manifest_refresh_fails_closed(self) -> None:
        temp, root = self.fixture()
        self.addCleanup(temp.cleanup)
        with (root / "runs" / "E2I" / "stdout").open("a") as handle:
            handle.write("tampered\n")
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_repo_head_or_dirty_identity_mismatch_fails_closed(self) -> None:
        for mutation in ("head", "dirty"):
            with self.subTest(mutation=mutation):
                temp, root = self.fixture()
                try:
                    if mutation == "head":
                        manifest = json.loads((root / "manifest.json").read_text())
                        manifest["repo"]["head"] = "0" * 40
                        (root / "manifest.json").write_text(json.dumps(manifest))
                    else:
                        (root / "source" / "untracked").write_text("dirty\n")
                    self.assertNotEqual(self.run_parser(root).returncode, 0)
                finally:
                    temp.cleanup()

    def test_binary_or_framework_hash_mismatch_fails_closed(self) -> None:
        for section in ("binary", "runner", "parser"):
            with self.subTest(section=section):
                temp, root = self.fixture()
                try:
                    manifest = json.loads((root / "manifest.json").read_text())
                    item = manifest[section] if section == "binary" else manifest["framework"][section]
                    item["sha256"] = "0" * 64
                    (root / "manifest.json").write_text(json.dumps(manifest))
                    self.assertNotEqual(self.run_parser(root).returncode, 0)
                finally:
                    temp.cleanup()

    def test_missing_duplicate_or_out_of_order_sequence_markers_fail_closed(self) -> None:
        mutations = (
            lambda text: text.replace("===SEQ0_RESUME_END===\n", "", 1),
            lambda text: text.replace("===SEQ1_ACTIVE_BEGIN===\n", "===SEQ1_ACTIVE_BEGIN===\n===SEQ1_ACTIVE_BEGIN===\n", 1),
            lambda text: text.replace(
                "===SEQ0_RESUME_BEGIN===\nsame-seq0\n===SEQ0_RESUME_END===\n",
                "===SEQ0_RESUME_END===\nsame-seq0\n===SEQ0_RESUME_BEGIN===\n",
                1,
            ),
            lambda text: text.replace("same-seq1\n", "", 1),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                temp, root = self.fixture()
                try:
                    path = root / "runs" / "E2I" / "stdout"
                    path.write_text(mutate(path.read_text()))
                    refresh(root, "E2I")
                    self.assertNotEqual(self.run_parser(root).returncode, 0)
                finally:
                    temp.cleanup()


if __name__ == "__main__":
    unittest.main()
