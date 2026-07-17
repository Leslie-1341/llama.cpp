#!/usr/bin/env python3

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts" / "parse-kv-paged-release-correctness.py"
CASES = ("R0", "R1", "R2", "R3", "R4", "R5", "N0", "N1", "N2")
RUNNER = ROOT / "scripts" / "run-kv-paged-release-correctness.sh"
COMMON_ENV = {
    "LLAMA_KV_TEST_MODE": "1", "LLAMA_KV_ACTIVE_TOKEN_STATS": "0",
    "LLAMA_KV_IDLE_NUM_IDLE_SEQS": "2", "LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS": "64",
    "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1",
    "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0", "LLAMA_KV_PAGED_BLOCK_SIZE": "16",
    "LLAMA_KV_PAGED_SHIFT": "0", "LLAMA_KV_PAGED_RELEASE": "0",
    "LLAMA_KV_PAGED_MINCORE": "0", "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0", "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_SHADOW_VALIDATE": "0", "LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE": "0",
    "LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ": "-1",
    "LLAMA_KV_RELEASE_TEST_SHARE_SEQ0": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
    "LLAMA_KV_PAGED_RELEASE_TEST_FRESH_VERIFY": "1",
    "LLAMA_KV_PAGED_RELEASE_TEST_REPEAT": "0", "LLAMA_KV_PAGED_RELEASE_TEST_REUSE": "0",
    "LLAMA_KV_SWAP": "0", "LLAMA_KV_LAZY_CLEAR": "0", "LLAMA_KV_LAZY_TAIL": "0",
}
CASE_ENV = {
    "R0": {"LLAMA_KV_PAGED_RELEASE": "0", "LLAMA_KV_PAGED_MINCORE": "0"},
    "R1": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "1"},
    "R2": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "1",
           "LLAMA_KV_PAGED_RELEASE_TEST_REPEAT": "1"},
    "R3": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "1",
           "LLAMA_KV_PAGED_RELEASE_TEST_REUSE": "1"},
    "R4": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "1",
           "LLAMA_KV_RELEASE_TEST_SHARE_SEQ0": "1"},
    "R5": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "0",
           "LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE": "1",
           "LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ": "1"},
    "N0": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "0",
           "LLAMA_KV_PAGED_INGRAPH": "0"},
    "N1": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "0"},
    "N2": {"LLAMA_KV_PAGED_RELEASE": "1", "LLAMA_KV_PAGED_MINCORE": "0",
           "LLAMA_KV_PAGED": "0"},
}

def telemetry(case_id: str) -> str:
    enabled = case_id in {"R1", "R2", "R3", "R4", "R5"}
    values = {
        "contract": "no_backing_unrecoverable_dead_or_unused_only",
        "release_calls": 8 if enabled else 0,
        "released_blocks": 10 if enabled else 0,
        "released_unused": 10 if enabled else 0,
        "released_dead": 0,
        "skip_owned": 6 if enabled else 0,
        "skip_shared": 2 if case_id == "R4" else 0,
        "ownership_invalid": 0,
        "backing_metadata_cleared": 160 if enabled else 0,
        "backing_metadata_stale": 0,
        "idempotent_skips": 20 if enabled else 0,
        "reuse_allocations": 2 if case_id == "R4" else 0,
        "write_commits": 2 if case_id == "R4" else 0,
        "write_rollbacks": 1 if case_id == "R5" else 0,
        "released_redirect_rows": 32 if enabled else 0,
        "released_redirect_blocks": 4 if enabled else 0,
        "released_redirect_no_dummy": 0,
        "release_violation": 1 if case_id == "R5" else 0,
        "active_release_violation": 1 if case_id == "R5" else 0,
        "padded_release_violation": 0,
        "identity_fail": 0,
        "mapping_oob_fail": 0,
        "logical_mapping_fail": 0,
        "write_resolve_fail": 0,
        "row_idx_fail": 0,
        "row_mapping_fatal": 0,
        "write_mapping_fatal": 0,
        "active_nonresident_fatal": 0,
        "input_setup_fatal": 0,
        "shadow_mismatch": 0,
        "shadow_fail": 0,
        "release_madvise_fail": 0,
        "mincore_failures": 0,
        "mincore_enabled": 1 if case_id in {"R1", "R2", "R3", "R4"} else 0,
        "mincore_samples": 2 if case_id in {"R1", "R2", "R3", "R4"} else 0,
        "mincore_before_last": 8192 if enabled else 0,
        "mincore_after_last": 0,
        "mincore_post_graph_last": 0,
        "mincore_released_total_last": 8192 if enabled else 0,
        "mincore_drop_max": 8192 if enabled else 0,
        "mincore_reaccess_last": 0,
        "mincore_reaccess_total_last": 0,
        "fresh_verify_bytes": 8192,
        "fresh_verify_hash": 123456789,
        "repeat_test_passes": 0,
        "reuse_test_commits": 0,
        "force_active_release_triggers": 1 if case_id == "R5" else 0,
    }
    return "KV_PAGED_RELEASE_STATS " + " ".join(f"{k}={v}" for k, v in values.items()) + "\n"

def config_marker(case_id: str) -> str:
    if case_id == "R0":
        return ""
    enabled = case_id not in {"N0", "N1", "N2"}
    reason = "ROW_INDEX_GATHER_ENABLED" if enabled else "ROW_INDEX_GATHER_UNAVAILABLE"
    return (f"KV_PAGED_RELEASE_CONFIG requested=1 enabled={1 if enabled else 0} reason={reason} "
            f"paged={0 if case_id == 'N2' else 1} ingraph={0 if case_id == 'N0' else 1} "
            f"layers_supported={0 if case_id == 'N1' else 1} row_idx={1 if enabled else 0}\n")

def r2_marker() -> str:
    return ("KV_PAGED_RELEASE_R2 first_transition=10 second_transition=0 first_abort=0 second_abort=0 "
            "candidate_hash_first=11 candidate_hash_second=11 state_hash_first=12 state_hash_second=12 "
            "metadata_hash_first=13 metadata_hash_second=13 range_hash_first=14 range_hash_second=14\n")

def r3_marker() -> str:
    return ("KV_PAGED_RELEASE_R3 released_block=3 reused_block=3 state_released=1 pending_write=1 "
            "commit=1 fresh_write_bytes=8192 reaccess_resident=4096 reaccess_total=4096 "
            "old_backing_metadata_absent=1 stale_content_used=0\n")

def write_fixture(root: pathlib.Path) -> None:
    model = root / "model.gguf"
    model.write_bytes(b"fixture-model")
    identities = {
        "binary": pathlib.Path(sys.executable).resolve(), "model": model.resolve(),
        "runner": RUNNER.resolve(), "parser": PARSER.resolve(),
    }
    (root / "head").write_text("a" * 40 + "\n")
    (root / "status").write_text("")
    (root / "plan").write_text("\n".join(CASES) + "\n")
    (root / "manifest").write_text(
        "protocol=kv_paged_release_correctness\nversion=2\ndry_run=0\n")
    (root / "command.base").write_text(f"{identities['binary']} -m {identities['model']}\n")
    for kind, path in identities.items():
        (root / f"identity.{kind}.path").write_text(str(path.resolve()) + "\n")
        (root / f"identity.{kind}.sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    for case_id in CASES:
        run = root / "runs" / case_id
        run.mkdir(parents=True)
        env = dict(COMMON_ENV)
        env.update(CASE_ENV[case_id])
        (run / "environment").write_text("".join(f"{key}={value}\n" for key, value in sorted(env.items())))
        command_env = " ".join(f"{key}={value}" for key, value in sorted(env.items()))
        cache_type = "f16" if case_id == "N1" else "f32"
        (run / "command").write_text(
            f"env {command_env} {identities['binary']} -m {identities['model']} "
            f"--cache-type-k {cache_type} --cache-type-v {cache_type}\n")
        if case_id == "R5":
            (run / "stdout").write_text("")
            (run / "stderr").write_text(
                "KV_TEST_DECODE_RESULT call_index=9 ret=-3\n"
                "KV_PAGED_TEST_FORCE_ACTIVE_RELEASE target_seq=1 logical_row=1 physical_block=0 physical_cell=1 trigger_count=1\n"
                "KV_PAGED_PRE_GRAPH_FAILURE reason=ACTIVE_READ_RELEASED_BLOCK graph_compute_skipped=1\n" +
                "KV_PAGED_RELEASE_R5 active_errors=1 transaction_open=1 rollback_blocks=1 rollback_complete=1\n" +
                config_marker(case_id) +
                telemetry(case_id))
            (run / "exit_code").write_text("1\n")
        else:
            (run / "stdout").write_text(
                "===SEQ1_ACTIVE_BEGIN===\nsame1\n===SEQ1_ACTIVE_END===\n"
                "===SEQ0_RESUME_BEGIN===\nsame0\n===SEQ0_RESUME_END===\n")
            special = r2_marker() if case_id == "R2" else r3_marker() if case_id == "R3" else ""
            (run / "stderr").write_text(
                "KV_TEST_SUMMARY result=PASS\n" + config_marker(case_id) + special + telemetry(case_id))
            (run / "exit_code").write_text("0\n")
        names = ("environment", "command", "stdout", "stderr", "exit_code")
        (run / "artifacts.sha256").write_text("".join(
            f"{hashlib.sha256((run / name).read_bytes()).hexdigest()}  {name}\n" for name in names))

def refresh(run: pathlib.Path) -> None:
    names = ("environment", "command", "stdout", "stderr", "exit_code")
    (run / "artifacts.sha256").write_text("".join(
        f"{hashlib.sha256((run / name).read_bytes()).hexdigest()}  {name}\n" for name in names))

class ParserTest(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        root = pathlib.Path(temp.name)
        write_fixture(root)
        self.addCleanup(temp.cleanup)
        return root

    def run_parser(self, root: pathlib.Path, *extra: str):
        env = dict(os.environ)
        env["LLAMA_KV_PARSER_TEST_FIXTURE"] = "1"
        return subprocess.run([sys.executable, str(PARSER), *extra, str(root)], env=env,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_valid_gate(self):
        root = self.fixture()
        result = self.run_parser(root)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((root / "summary.json").read_text())["result"], "PASS")

    def test_missing_field_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R1" / "stderr"
        path.write_text(path.read_text().replace(" mincore_drop_max=8192", ""))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_duplicate_marker_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R2" / "stderr"
        path.write_text(path.read_text() + telemetry("R2"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_duplicate_field_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R1" / "stderr"
        path.write_text(path.read_text().replace(
            "KV_PAGED_RELEASE_STATS ", "KV_PAGED_RELEASE_STATS released_blocks=10 "))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_wrong_summary_status_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R1" / "stderr"
        path.write_text(path.read_text().replace("KV_TEST_SUMMARY result=PASS", "KV_TEST_SUMMARY result=FAIL"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_normal_violation_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R3" / "stderr"
        path.write_text(path.read_text().replace("padded_release_violation=0", "padded_release_violation=1"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_compute_not_skipped_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "stderr"
        path.write_text(path.read_text().replace("graph_compute_skipped=1", "graph_compute_skipped=0"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_timeout_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "exit_code"
        path.write_text("124\n")
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_signal_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "exit_code"
        path.write_text("143\n")
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_crash_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "exit_code"
        path.write_text("139\n")
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_ret_minus_30_does_not_match_minus_3(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "stderr"
        path.write_text(path.read_text().replace("ret=-3", "ret=-30"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_duplicate_active_error_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "stderr"
        path.write_text(path.read_text() +
                        "KV_PAGED_PRE_GRAPH_FAILURE reason=ACTIVE_READ_RELEASED_BLOCK graph_compute_skipped=1\n")
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_rollback_mismatch_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "stderr"
        path.write_text(path.read_text().replace("rollback_blocks=1", "rollback_blocks=0"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_transaction_zero_all_zero_valid(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "stderr"
        path.write_text(path.read_text()
                        .replace("transaction_open=1", "transaction_open=0")
                        .replace("rollback_blocks=1", "rollback_blocks=0")
                        .replace("write_rollbacks=1", "write_rollbacks=0"))
        refresh(path.parent)
        self.assertEqual(self.run_parser(root).returncode, 0)

    def test_r5_transaction_zero_rollback_one_fails(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "stderr"
        path.write_text(path.read_text().replace("transaction_open=1", "transaction_open=0"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r5_write_rollback_counter_mismatch_fails(self):
        root = self.fixture()
        path = root / "runs" / "R5" / "stderr"
        path.write_text(path.read_text().replace("write_rollbacks=1", "write_rollbacks=0"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r2_second_transition_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R2" / "stderr"
        path.write_text(path.read_text().replace("second_transition=0", "second_transition=1"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r2_state_hash_change_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R2" / "stderr"
        path.write_text(path.read_text().replace("state_hash_second=12", "state_hash_second=99"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r3_different_reuse_block_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R3" / "stderr"
        path.write_text(path.read_text().replace("reused_block=3", "reused_block=4"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r3_missing_pending_state_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R3" / "stderr"
        path.write_text(path.read_text().replace("pending_write=1", "pending_write=0"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_negative_continuous_view_release_counter_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "N0" / "stderr"
        path.write_text(path.read_text().replace("release_calls=0", "release_calls=1"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_negative_release_config_enabled_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "N1" / "stderr"
        path.write_text(path.read_text().replace(
            "requested=1 enabled=0 reason=ROW_INDEX_GATHER_UNAVAILABLE",
            "requested=1 enabled=1 reason=ROW_INDEX_GATHER_ENABLED"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_zero_mincore_range_fails_closed(self):
        root = self.fixture()
        path = root / "runs" / "R1" / "stderr"
        path.write_text(path.read_text().replace("mincore_released_total_last=8192", "mincore_released_total_last=0"))
        refresh(path.parent)
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_missing_top_identity_fails_closed(self):
        root = self.fixture()
        (root / "identity.model.sha256").unlink()
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_dirty_formal_status_fails_closed(self):
        root = self.fixture()
        (root / "status").write_text(" M src/llama-kv-cache.cpp\n")
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_extra_run_directory_fails_closed(self):
        root = self.fixture()
        (root / "runs" / "R6").mkdir()
        self.assertNotEqual(self.run_parser(root).returncode, 0)

    def test_r3_does_not_use_cumulative_fresh_hash(self):
        root = self.fixture()
        path = root / "runs" / "R3" / "stderr"
        path.write_text(path.read_text().replace("fresh_verify_hash=123456789", "fresh_verify_hash=987654321"))
        refresh(path.parent)
        self.assertEqual(self.run_parser(root).returncode, 0)

    def test_dry_run(self):
        root = self.fixture()
        (root / "manifest").write_text(
            "protocol=kv_paged_release_correctness\nversion=2\ndry_run=1\n")
        for case_id in CASES:
            (root / "runs" / case_id / "exit_code").write_text("DRY_RUN\n")
            (root / "runs" / case_id / "stdout").write_text("")
            (root / "runs" / case_id / "stderr").write_text("")
            refresh(root / "runs" / case_id)
        self.assertEqual(self.run_parser(root, "--dry-run").returncode, 0)

if __name__ == "__main__":
    unittest.main()
