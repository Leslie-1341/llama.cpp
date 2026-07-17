#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CASES = ("R0", "R1", "R2", "R3", "R4", "R5", "N0", "N1", "N2")
COMMON_ENV = {
    "LLAMA_KV_TEST_MODE": "1", "LLAMA_KV_ACTIVE_TOKEN_STATS": "0",
    "LLAMA_KV_IDLE_NUM_IDLE_SEQS": "2", "LLAMA_KV_PAGED": "1",
    "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
    "LLAMA_KV_PAGED_BLOCK_SIZE": "16", "LLAMA_KV_PAGED_SHIFT": "0",
    "LLAMA_KV_PAGED_RELEASE": "0", "LLAMA_KV_PAGED_MINCORE": "0",
    "LLAMA_KV_PAGED_SWAP": "0", "LLAMA_KV_PAGED_IDLE_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0", "LLAMA_KV_PAGED_SHADOW_VALIDATE": "0",
    "LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE": "0",
    "LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ": "-1",
    "LLAMA_KV_RELEASE_TEST_SHARE_SEQ0": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
    "LLAMA_KV_PAGED_RELEASE_TEST_FRESH_VERIFY": "1",
    "LLAMA_KV_PAGED_RELEASE_TEST_REPEAT": "0", "LLAMA_KV_PAGED_RELEASE_TEST_REUSE": "0",
    "LLAMA_KV_SWAP": "0", "LLAMA_KV_LAZY_CLEAR": "0", "LLAMA_KV_LAZY_TAIL": "0",
}
FIELDS = (
    "release_calls", "released_blocks", "released_unused", "released_dead",
    "skip_owned", "skip_shared", "ownership_invalid", "backing_metadata_cleared",
    "backing_metadata_stale",
    "idempotent_skips", "reuse_allocations", "write_commits", "write_rollbacks",
    "released_redirect_rows", "released_redirect_blocks", "released_redirect_no_dummy",
    "release_violation", "active_release_violation", "padded_release_violation",
    "identity_fail", "mapping_oob_fail", "logical_mapping_fail", "write_resolve_fail",
    "row_idx_fail", "row_mapping_fatal", "write_mapping_fatal", "active_nonresident_fatal",
    "input_setup_fatal", "shadow_mismatch", "shadow_fail", "release_madvise_fail",
    "mincore_failures",
    "mincore_enabled", "mincore_samples", "mincore_before_last", "mincore_after_last",
    "mincore_post_graph_last", "mincore_released_total_last", "mincore_drop_max",
    "mincore_reaccess_last", "mincore_reaccess_total_last",
    "fresh_verify_bytes", "fresh_verify_hash", "repeat_test_passes", "reuse_test_commits",
    "force_active_release_triggers",
)
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

NEGATIVE_CASES = {"N0", "N1", "N2"}

class GateError(ValueError):
    pass

def strict_kv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.count("=") != 1:
            raise GateError(f"{path.name}: malformed key/value line")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise GateError(f"{path.name}: duplicate/invalid key")
        result[key] = value
    return result

def sha256_path(path: Path) -> str:
    if not path.is_file():
        raise GateError(f"identity path missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_top(root: Path, dry_run: bool, fixture: bool) -> dict[str, Path]:
    required = {"head", "status", "plan", "command.base", "manifest"}
    for kind in ("binary", "model", "runner", "parser"):
        required.update({f"identity.{kind}.path", f"identity.{kind}.sha256"})
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise GateError(f"missing top-level artifacts: {missing}")

    manifest = strict_kv(root / "manifest")
    expected_manifest = {
        "protocol": "kv_paged_release_correctness", "version": "2",
        "dry_run": "1" if dry_run else "0",
    }
    if manifest != expected_manifest:
        raise GateError(f"manifest mismatch: {manifest}")

    head = (root / "head").read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise GateError("invalid HEAD identity")
    status = (root / "status").read_text()
    if not dry_run and status != "":
        raise GateError("formal artifact was not produced from a clean worktree")

    identities: dict[str, Path] = {}
    for kind in ("binary", "model", "runner", "parser"):
        path_text = (root / f"identity.{kind}.path").read_text().strip()
        digest = (root / f"identity.{kind}.sha256").read_text().strip()
        if not path_text or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise GateError(f"invalid {kind} identity record")
        path = Path(path_text)
        if sha256_path(path) != digest:
            raise GateError(f"{kind} identity mismatch")
        identities[kind] = path

    repo = Path(__file__).resolve().parents[1]
    if identities["parser"].resolve() != Path(__file__).resolve():
        raise GateError("parser path identity mismatch")
    expected_runner = repo / "scripts" / "run-kv-paged-release-correctness.sh"
    if identities["runner"].resolve() != expected_runner.resolve():
        raise GateError("runner path identity mismatch")
    command_base = (root / "command.base").read_text()
    if str(identities["binary"]) not in command_base or str(identities["model"]) not in command_base:
        raise GateError("base command is not bound to binary/model identities")

    if not fixture:
        live_head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        if live_head != head:
            raise GateError("artifact HEAD differs from parser checkout HEAD")
        if not dry_run:
            live_status = subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain"], text=True)
            if live_status != "":
                raise GateError("parser checkout is not clean")
    return identities

def one_line(text: str, marker: str) -> str:
    lines = [line for line in text.splitlines() if marker in line]
    if len(lines) != 1:
        raise GateError(f"expected one {marker}, got {len(lines)}")
    return lines[0]

def fields(line: str) -> dict[str, str]:
    pairs = re.findall(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)", line)
    result: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate field {key}")
        result[key] = value
    return result

def integer(values: dict[str, str], key: str) -> int:
    if key not in values or re.fullmatch(r"-?[0-9]+", values[key]) is None:
        raise GateError(f"missing or invalid {key}")
    return int(values[key])

def extract(stdout: bytes, begin: bytes, end: bytes) -> bytes:
    if stdout.count(begin) != 1 or stdout.count(end) != 1:
        raise GateError("sequence marker count mismatch")
    body = stdout.split(begin, 1)[1].split(end, 1)[0]
    return body.strip()

def verify_artifacts(run: Path) -> None:
    checksum_path = run / "artifacts.sha256"
    if not checksum_path.is_file():
        raise GateError(f"{run.name}: missing artifact checksums")
    expected_names = {"environment", "command", "stdout", "stderr", "exit_code"}
    seen: set[str] = set()
    for line in checksum_path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        if match is None or match.group(2) in seen:
            raise GateError(f"{run.name}: invalid artifact checksum line")
        digest, name = match.groups()
        if name not in expected_names:
            raise GateError(f"{run.name}: unexpected checksummed artifact {name}")
        path = run / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise GateError(f"{run.name}: artifact identity mismatch: {name}")
        seen.add(name)
    if seen != expected_names:
        raise GateError(f"{run.name}: artifact checksum set mismatch")

def environment(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            raise GateError(f"{path.parent.name}: malformed environment line")
        key, value = line.split("=", 1)
        if not key or key in result:
            raise GateError(f"{path.parent.name}: duplicate/invalid environment key")
        result[key] = value
    return result

def parse_run(root: Path, case_id: str, dry_run: bool, identities: dict[str, Path]) -> dict[str, object]:
    run = root / "runs" / case_id
    required = ("environment", "command", "stdout", "stderr", "exit_code", "artifacts.sha256")
    if not run.is_dir() or any(not (run / name).is_file() for name in required):
        raise GateError(f"{case_id}: incomplete artifacts")
    verify_artifacts(run)
    env = environment(run / "environment")
    required_env = dict(COMMON_ENV)
    required_env.update(CASE_ENV[case_id])
    expected_keys = set(required_env) | {"LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS"}
    if set(env) != expected_keys or re.fullmatch(r"[1-9][0-9]*", env.get(
            "LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS", "")) is None:
        raise GateError(f"{case_id}: environment key set/warmup mismatch")
    for key, expected in required_env.items():
        if env.get(key) != expected:
            raise GateError(f"{case_id}: {key}={env.get(key)!r}, expected {expected!r}")
    command = (run / "command").read_text()
    if str(identities["binary"]) not in command or str(identities["model"]) not in command:
        raise GateError(f"{case_id}: command is not bound to binary/model identities")
    for key, value in env.items():
        if f"{key}={value}" not in command:
            raise GateError(f"{case_id}: command/environment mismatch for {key}")
    expected_cache_type = "f16" if case_id == "N1" else "f32"
    for option in ("--cache-type-k", "--cache-type-v"):
        if re.search(rf"(?:^|\s){option}\s+{expected_cache_type}(?:\s|$)", command) is None:
            raise GateError(f"{case_id}: command does not select {expected_cache_type} K/V")
    exit_text = (run / "exit_code").read_text().strip()
    if dry_run:
        if exit_text != "DRY_RUN":
            raise GateError(f"{case_id}: dry-run exit marker invalid")
        return {"case": case_id, "status": "PLANNED"}
    if re.fullmatch(r"[0-9]+", exit_text) is None:
        raise GateError(f"{case_id}: invalid exit code")
    rc = int(exit_text)
    stderr = (run / "stderr").read_text(errors="replace")
    stdout = (run / "stdout").read_bytes()
    line = one_line(stderr, "KV_PAGED_RELEASE_STATS")
    values = fields(line)
    if values.get("contract") != "no_backing_unrecoverable_dead_or_unused_only":
        raise GateError(f"{case_id}: release contract marker invalid")
    for name in FIELDS:
        integer(values, name)

    if case_id != "R0":
        config = fields(one_line(stderr, "KV_PAGED_RELEASE_CONFIG"))
        expected_enabled = "0" if case_id in NEGATIVE_CASES else "1"
        expected_reason = "ROW_INDEX_GATHER_UNAVAILABLE" if case_id in NEGATIVE_CASES else \
            "ROW_INDEX_GATHER_ENABLED"
        if config.get("requested") != "1" or config.get("enabled") != expected_enabled or \
                config.get("reason") != expected_reason:
            raise GateError(f"{case_id}: release configuration evidence invalid")

    if case_id != "R5":
        if rc != 0:
            raise GateError(f"{case_id}: exit {rc}")
        summary = fields(one_line(stderr, "KV_TEST_SUMMARY"))
        if summary.get("result") != "PASS":
            raise GateError(f"{case_id}: test summary not PASS")
        for name in ("release_violation", "active_release_violation", "padded_release_violation",
                     "released_redirect_no_dummy", "force_active_release_triggers",
                     "ownership_invalid", "backing_metadata_stale", "write_rollbacks",
                     "identity_fail", "mapping_oob_fail", "logical_mapping_fail", "write_resolve_fail",
                     "row_idx_fail", "row_mapping_fatal", "write_mapping_fatal", "active_nonresident_fatal",
                     "input_setup_fatal", "shadow_mismatch", "shadow_fail", "release_madvise_fail",
                     "mincore_failures"):
            if integer(values, name) != 0:
                raise GateError(f"{case_id}: {name} must be zero")
        seq0 = extract(stdout, b"===SEQ0_RESUME_BEGIN===", b"===SEQ0_RESUME_END===")
        seq1 = extract(stdout, b"===SEQ1_ACTIVE_BEGIN===", b"===SEQ1_ACTIVE_END===")
    else:
        if rc != 1:
            raise GateError(f"R5: expected driver exit 1, got {rc}")
        if integer(values, "force_active_release_triggers") != 1:
            raise GateError("R5: force trigger must be exactly one")
        if integer(values, "active_release_violation") != 1:
            raise GateError("R5: active violation must be exactly one")
        if integer(values, "release_violation") != 1:
            raise GateError("R5: release violation must be exactly one")
        for name in ("padded_release_violation", "released_redirect_no_dummy", "ownership_invalid",
                     "backing_metadata_stale", "identity_fail", "mapping_oob_fail",
                     "logical_mapping_fail", "write_resolve_fail", "row_idx_fail",
                     "row_mapping_fatal", "write_mapping_fatal", "active_nonresident_fatal",
                     "input_setup_fatal", "shadow_mismatch", "shadow_fail", "release_madvise_fail",
                     "mincore_failures"):
            if integer(values, name) != 0:
                raise GateError(f"R5: unexpected {name}")
        decode_lines = [line for line in stderr.splitlines() if "KV_TEST_DECODE_RESULT" in line]
        decode_values = [fields(line) for line in decode_lines]
        if any(set(value) != {"call_index", "ret"} for value in decode_values) or \
                sum(value.get("ret") == "-3" for value in decode_values) != 1 or any(
                    value.get("ret", "").startswith("-") and value.get("ret") != "-3"
                    for value in decode_values):
            raise GateError("R5: expected exactly one decode -3")
        if stderr.count("KV_PAGED_TEST_FORCE_ACTIVE_RELEASE ") != 1:
            raise GateError("R5: expected exactly one force-active-release trigger marker")
        pre = fields(one_line(stderr, "KV_PAGED_PRE_GRAPH_FAILURE"))
        if pre.get("reason") != "ACTIVE_READ_RELEASED_BLOCK" or pre.get("graph_compute_skipped") != "1":
            raise GateError("R5: pre-graph failure evidence invalid")
        r5 = fields(one_line(stderr, "KV_PAGED_RELEASE_R5"))
        if integer(r5, "active_errors") != 1:
            raise GateError("R5: active_errors must be exactly one")
        txn_open = integer(r5, "transaction_open")
        if txn_open not in (0, 1):
            raise GateError("R5: transaction_open must be 0 or 1")
        if integer(r5, "rollback_blocks") != txn_open:
            raise GateError("R5: rollback_blocks must equal transaction_open")
        if r5.get("rollback_complete") != "1":
            raise GateError("R5: rollback_complete must be 1")
        if integer(values, "write_rollbacks") != txn_open:
            raise GateError("R5: write_rollbacks must equal rollback_blocks")
        return {"case": case_id, "status": "EXPECTED_FAILURE", "exit": rc, "telemetry": values}

    if case_id == "R0" or case_id in NEGATIVE_CASES:
        zero_release_fields = (
            "release_calls", "released_blocks", "released_unused", "released_dead",
            "skip_owned", "skip_shared", "ownership_invalid", "backing_metadata_cleared",
            "backing_metadata_stale", "idempotent_skips", "reuse_allocations", "write_commits",
            "write_rollbacks", "released_redirect_rows", "released_redirect_blocks",
            "release_violation", "active_release_violation", "padded_release_violation",
            "release_madvise_fail", "repeat_test_passes", "reuse_test_commits",
            "force_active_release_triggers",
        )
        if any(integer(values, name) != 0 for name in zero_release_fields):
            raise GateError(f"{case_id}: destructive release counters must be zero")
    else:
        if integer(values, "released_blocks") <= 0 or integer(values, "released_unused") <= 0:
            raise GateError(f"{case_id}: unused release did not trigger")
        if integer(values, "skip_owned") <= 0:
            raise GateError(f"{case_id}: owned protection not observed")
        if integer(values, "mincore_enabled") != 1 or integer(values, "mincore_samples") <= 0:
            raise GateError(f"{case_id}: release mincore evidence missing")
        if integer(values, "mincore_released_total_last") <= 0:
            raise GateError(f"{case_id}: release mincore range is empty")
        if integer(values, "mincore_before_last") <= integer(values, "mincore_after_last") or \
                integer(values, "mincore_drop_max") <= 0:
            raise GateError(f"{case_id}: KV resident pages did not decrease")
        if integer(values, "mincore_after_last") != 0:
            raise GateError(f"{case_id}: released pages remained resident after release")
        if integer(values, "mincore_post_graph_last") != 0:
            raise GateError(f"{case_id}: released pages silently refaulted")
    if case_id == "R2":
        repeat = fields(one_line(stderr, "KV_PAGED_RELEASE_R2"))
        if integer(repeat, "first_transition") <= 0 or integer(repeat, "second_transition") != 0 or \
                integer(repeat, "first_abort") != 0 or integer(repeat, "second_abort") != 0:
            raise GateError("R2: two-release transition/abort evidence invalid")
        for prefix in ("candidate", "state", "metadata", "range"):
            if repeat.get(f"{prefix}_hash_first") != repeat.get(f"{prefix}_hash_second"):
                raise GateError(f"R2: {prefix} hash changed across the second release")
    if case_id == "R3":
        reuse = fields(one_line(stderr, "KV_PAGED_RELEASE_R3"))
        if reuse.get("released_block") != reuse.get("reused_block") or \
                any(reuse.get(name) != "1" for name in (
                    "state_released", "pending_write", "commit", "old_backing_metadata_absent")) or \
                integer(reuse, "fresh_write_bytes") <= 0 or integer(reuse, "reaccess_resident") <= 0 or \
                integer(reuse, "reaccess_total") <= 0 or reuse.get("stale_content_used") != "0":
            raise GateError("R3: deterministic same-block fresh reuse evidence invalid")
    if case_id == "R4" and integer(values, "skip_shared") <= 0:
        raise GateError("R4: shared-owner protection not observed")
    return {"case": case_id, "status": "PASS", "exit": rc, "seq0": hashlib.sha256(seq0).hexdigest(),
            "seq1": hashlib.sha256(seq1).hexdigest(), "telemetry": values}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("artifact", type=Path)
    args = ap.parse_args()
    try:
        fixture = os.environ.get("LLAMA_KV_PARSER_TEST_FIXTURE") == "1"
        identities = verify_top(args.artifact, args.dry_run, fixture)
        plan = tuple((args.artifact / "plan").read_text().split())
        if plan != CASES:
            raise GateError(f"plan mismatch: {plan}")
        runs_root = args.artifact / "runs"
        if not runs_root.is_dir() or {p.name for p in runs_root.iterdir()} != set(CASES) or \
                any(not (runs_root / case_id).is_dir() for case_id in CASES):
            raise GateError("run directory set does not exactly match the fixed plan")
        runs = [parse_run(args.artifact, case_id, args.dry_run, identities) for case_id in CASES]
        if not args.dry_run:
            baseline = runs[0]
            for run in runs[1:5] + runs[6:]:
                if run["seq0"] != baseline["seq0"] or run["seq1"] != baseline["seq1"]:
                    raise GateError(f"{run['case']}: output differs from R0")
        summary = {"protocol": "kv_paged_release_correctness", "version": 2,
                   "dry_run": args.dry_run, "result": "PASS", "runs": runs}
        (args.artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print("PASS: kv paged release correctness gate")
        return 0
    except (OSError, GateError, UnicodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
