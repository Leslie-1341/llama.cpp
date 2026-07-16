#!/usr/bin/env python3
"""Fail-closed parser for the independent paged identity E2I path."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


CASES = ("E0", "E2", "E2I", "E2I_NOREUSE", "SHIFT", "NONIDENTITY", "SWAP", "RELEASE", "MADVISE")
PLAN = tuple((index, case_id) for index, case_id in enumerate(CASES, 1))
REJECT = {
    "E2": "not_requested",
    "SHIFT": "non_identity_mapping",
    "NONIDENTITY": "dynamic_remap",
    "SWAP": "swap",
    "RELEASE": "release",
    "MADVISE": "madvise",
}
BASE_ENV = {
    "LLAMA_KV_ACTIVE_TOKEN_STATS": "1", "LLAMA_KV_PAGED_IO_STATS": "1",
    "LLAMA_KV_E2_GET_ROWS_PROFILE": "0",
    "LLAMA_KV_IDLE_NUM_IDLE_SEQS": "2", "LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS": "256",
    "LLAMA_KV_CACHE_DEBUG": "0", "LLAMA_KV_LAZY_CLEAR": "0", "LLAMA_KV_LAZY_TAIL": "0",
    "LLAMA_KV_PAGED_BLOCK_SIZE": "16", "LLAMA_KV_PAGED_SHIFT": "0",
    "LLAMA_KV_PAGED_RELEASE": "0", "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0", "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS": "1",
    "LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES": "0", "LLAMA_KV_PAGED_SHADOW_VALIDATE": "0",
    "LLAMA_KV_PAGED_MINCORE": "0", "LLAMA_KV_PAGED_TRACE": "0",
    "LLAMA_KV_PAGED_IDLE_TRACE": "0", "LLAMA_KV_PAGED_REFAULT_TRACE": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE": "0", "LLAMA_KV_PAGED_REFAULT_TRACE_MAX": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE_ONCE": "0", "LLAMA_KV_PAGED_TIMING": "0",
    "LLAMA_KV_PAGED_RESUME_TIMING": "0", "LLAMA_KV_PAGED_RESUME_TIMING_STEP": "0",
    "LLAMA_KV_PAGED_RESUME_PREFETCH": "0", "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "0", "LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS": "1", "LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP": "1",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS": "1",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP": "1",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE": "off", "LLAMA_KV_PAGED_RESUME_PENDING_TOKEN": "96",
    "LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS": "0", "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "0",
    "LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME": "0", "LLAMA_KV_SWAP": "0",
    "LLAMA_KV_SWAP_MODE": "exact", "LLAMA_KV_SWAP_WINDOW": "0", "LLAMA_KV_SWAP_SINK": "0",
    "LLAMA_KV_SWAP_RSS_SAMPLE": "0", "LLAMA_KV_SWAP_MADVISE": "0",
    "LLAMA_KV_SWAP_BACKEND_SELFTEST": "0", "LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST": "0",
}
CASE_ENV = {
    "E0": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
           "LLAMA_KV_PAGED": "0", "LLAMA_KV_PAGED_INGRAPH": "0", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0"},
    "E2": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
           "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "1"},
    "E2I": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "1",
            "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0"},
    "E2I_NOREUSE": {"LLAMA_GRAPH_REUSE_DISABLE": "1", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "1",
                    "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0"},
    "SHIFT": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "1",
              "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
              "LLAMA_KV_PAGED_SHIFT": "1"},
    "NONIDENTITY": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "1",
                    "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "1"},
    "SWAP": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "1",
             "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
             "LLAMA_KV_PAGED_SWAP": "1"},
    "RELEASE": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "1",
                "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
                "LLAMA_KV_PAGED_RELEASE": "1"},
    "MADVISE": {"LLAMA_GRAPH_REUSE_DISABLE": "0", "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "1",
                "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
                "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "1"},
}
SAFETY_ZERO = (
    "paged_row_mapping_invalid_fatal", "paged_write_mapping_invalid_fatal",
    "paged_active_row_nonresident_fatal", "paged_input_setup_fatal",
    "paged_swapped_active_visible_violation", "paged_swap_backend_failures",
)
RAW_ARTIFACTS = (
    "run.json", "execution.json", "command", "environment", "stdout", "stderr", "exit_code",
    "seq0", "seq1", "sequence.sha256",
)
SEQUENCE_MARKERS = (
    ("seq1", b"===SEQ1_ACTIVE_BEGIN===", b"===SEQ1_ACTIVE_END==="),
    ("seq0", b"===SEQ0_RESUME_BEGIN===", b"===SEQ0_RESUME_END==="),
)


class ArtifactError(ValueError):
    pass


def read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def file_identity(item: Any, label: str) -> Path:
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ArtifactError(f"manifest missing {label} identity")
    path = Path(item["path"])
    expected_sha = item.get("sha256")
    expected_size = item.get("size")
    if not path.is_file() or not isinstance(expected_size, int) or expected_size < 0 \
            or not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ArtifactError(f"manifest has invalid {label} identity")
    if path.stat().st_size != expected_size or sha256(path) != expected_sha:
        raise ArtifactError(f"{label} identity mismatch")
    return path


def validate_identity(manifest: dict[str, Any], dry_run: bool) -> None:
    repo = manifest.get("repo")
    if not isinstance(repo, dict) or not isinstance(repo.get("path"), str) \
            or not isinstance(repo.get("head"), str) or not isinstance(repo.get("dirty"), bool) \
            or not isinstance(repo.get("status_porcelain"), list) \
            or any(not isinstance(line, str) for line in repo["status_porcelain"]):
        raise ArtifactError("manifest missing repo identity")
    repo_path = Path(repo["path"])
    try:
        head = subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(repo_path), "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactError("cannot verify repository identity") from exc
    if head != repo["head"] or status != repo["status_porcelain"] or bool(status) != repo["dirty"]:
        raise ArtifactError("repository HEAD/dirty identity mismatch")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("dry_run") is not dry_run:
        raise ArtifactError("parser mode differs from manifest execution mode")
    if not dry_run and (repo["dirty"] or repo["status_porcelain"]):
        raise ArtifactError("non-dry-run E2I artifact requires a clean worktree")
    file_identity(manifest.get("binary"), "binary")
    file_identity(manifest.get("model"), "model")
    framework = manifest.get("framework")
    if not isinstance(framework, dict):
        raise ArtifactError("manifest missing framework identity")
    file_identity(framework.get("runner"), "framework.runner")
    file_identity(framework.get("parser"), "framework.parser")


def validate_artifacts(run_dir: Path, manifest_run: dict[str, Any]) -> None:
    identities = manifest_run.get("artifacts")
    if not isinstance(identities, dict) or set(identities) != set(RAW_ARTIFACTS):
        raise ArtifactError(f"{run_dir.name} artifact identity set mismatch")
    for name in RAW_ARTIFACTS:
        path = run_dir / name
        identity = identities[name]
        if not path.is_file() or not isinstance(identity, dict):
            raise ArtifactError(f"{run_dir.name} missing artifact {name}")
        if identity.get("size") != path.stat().st_size or identity.get("sha256") != sha256(path):
            raise ArtifactError(f"{run_dir.name} artifact identity mismatch: {name}")


def validate_sequences(run_dir: Path) -> None:
    stdout = (run_dir / "stdout").read_bytes()
    lines = stdout.splitlines(keepends=True)
    positions: list[int] = []
    for name, begin, end in SEQUENCE_MARKERS:
        normalized = [line.rstrip(b"\r\n") for line in lines]
        begin_positions = [index for index, line in enumerate(normalized) if line == begin]
        end_positions = [index for index, line in enumerate(normalized) if line == end]
        if len(begin_positions) != 1 or len(end_positions) != 1:
            raise ArtifactError(
                f"{run_dir.name} {name} marker count mismatch: "
                f"begin={len(begin_positions)} end={len(end_positions)}"
            )
        begin_index, end_index = begin_positions[0], end_positions[0]
        if begin_index >= end_index:
            raise ArtifactError(f"{run_dir.name} {name} markers are out of order")
        section = b"".join(lines[begin_index + 1:end_index])
        if not section:
            raise ArtifactError(f"{run_dir.name} {name} marker section is empty")
        if section != (run_dir / name).read_bytes():
            raise ArtifactError(f"{run_dir.name} {name} does not match raw stdout section")
        positions.extend((begin_index, end_index))
    if positions != sorted(positions):
        raise ArtifactError(f"{run_dir.name} sequence marker sections are globally out of order")


def fields(line: str) -> dict[str, str]:
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line)
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ArtifactError(f"duplicate telemetry key(s): {', '.join(duplicates)}")
    return dict(pairs)


def unique_line(text: str, marker: str) -> str:
    lines = [line for line in text.splitlines() if marker in line]
    if len(lines) != 1:
        raise ArtifactError(f"telemetry marker {marker!r} count is {len(lines)}, expected 1")
    return lines[0]


def integer(source: dict[str, str], name: str) -> int:
    try:
        return int(source[name])
    except (KeyError, ValueError) as exc:
        raise ArtifactError(f"missing or invalid integer telemetry: {name}") from exc


def environment(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in read(run_dir / "environment").splitlines():
        if "=" not in line:
            raise ArtifactError(f"malformed environment line in {run_dir}: {line!r}")
        key, value = line.split("=", 1)
        if key in result:
            raise ArtifactError(f"duplicate environment key {key} in {run_dir}")
        result[key] = value
    return result


def validate_matrix(root: Path, manifest: dict[str, Any], dry_run: bool) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    if manifest.get("protocol") != "kv_paged_identity_e2i" or manifest.get("version") != 1:
        raise ArtifactError("manifest protocol/version mismatch")
    validate_identity(manifest, dry_run)
    planned = tuple((int(item["order"]), str(item["case"])) for item in manifest.get("planned_runs", []))
    if planned != PLAN:
        raise ArtifactError("manifest does not match the fixed E2I plan")
    run_dirs: list[Path] = []
    actual: list[tuple[int, str]] = []
    for case_id in CASES:
        run_dir = root / "runs" / case_id
        try:
            meta = json.loads(read(run_dir / "run.json"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"invalid run.json for {case_id}") from exc
        actual.append((int(meta["order"]), str(meta["case"])))
        for name in ("command", "environment", "stdout", "stderr", "exit_code", "seq0", "seq1"):
            if not (run_dir / name).is_file():
                raise ArtifactError(f"{case_id} missing artifact {name}")
        run_dirs.append(run_dir)
    if tuple(actual) != PLAN:
        raise ArtifactError("run directories do not match the fixed E2I plan")
    completed_runs = manifest.get("completed_runs")
    if not isinstance(completed_runs, list) or len(completed_runs) != len(PLAN):
        raise ArtifactError("manifest completed_runs is missing or incomplete")
    completed: dict[str, dict[str, Any]] = {}
    completed_plan: list[tuple[int, str]] = []
    for index, item in enumerate(completed_runs):
        if not isinstance(item, dict):
            raise ArtifactError(f"completed_runs[{index}] must be an object")
        try:
            key = (int(item["order"]), str(item["case"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError(f"completed_runs[{index}] has invalid matrix fields") from exc
        if key[1] in completed:
            raise ArtifactError("manifest completed_runs contains duplicate cases")
        completed_plan.append(key)
        completed[key[1]] = item
    if tuple(completed_plan) != PLAN:
        raise ArtifactError("manifest completed_runs order/uniqueness mismatch")
    for run_dir in run_dirs:
        validate_artifacts(run_dir, completed[run_dir.name])
    return run_dirs, completed


def validate_environment(run_dir: Path, case_id: str) -> dict[str, str]:
    env = environment(run_dir)
    expected = dict(BASE_ENV)
    expected.update(CASE_ENV[case_id])
    expected["LLAMA_KV_SWAP_DIR"] = str((run_dir.parent.parent / "swap" / case_id).resolve())
    missing = sorted(expected.keys() - env.keys())
    unexpected = sorted(env.keys() - expected.keys())
    mismatches = sorted(name for name in expected.keys() & env.keys() if env[name] != expected[name])
    if missing or unexpected or mismatches:
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if unexpected:
            detail.append(f"unexpected={unexpected}")
        if mismatches:
            detail.append("mismatch=" + repr([(name, env[name], expected[name]) for name in mismatches]))
        raise ArtifactError(f"{case_id} environment mismatch: {'; '.join(detail)}")
    return env


def normalized_command(execution: dict[str, Any]) -> str:
    timeout = execution["timeout"]
    tokens = [
        "cd", shlex.quote(execution["cwd"]), "&&",
        "timeout", "--signal=TERM", "--kill-after=10", str(timeout["seconds"]),
        "env",
    ]
    tokens.extend(f"-u {shlex.quote(name)}" for name in execution["env_unset"])
    tokens.extend(
        f"{shlex.quote(name)}={shlex.quote(value)}"
        for name, value in sorted(execution["env_set"].items())
    )
    tokens.extend(shlex.quote(value) for value in execution["argv"])
    return " ".join(tokens) + "\n"


def validate_execution(run_dir: Path, manifest: dict[str, Any], case_id: str) -> None:
    try:
        execution = json.loads(read(run_dir / "execution.json"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{case_id} execution.json is invalid") from exc
    if set(execution) != {"cwd", "binary", "argv", "env_set", "env_unset", "timeout"}:
        raise ArtifactError(f"{case_id} execution.json field set mismatch")
    workload = manifest.get("workload")
    common_args = workload.get("common_args") if isinstance(workload, dict) else None
    if not isinstance(common_args, list) or any(not isinstance(item, str) for item in common_args):
        raise ArtifactError("manifest workload.common_args is invalid")
    if execution["cwd"] != manifest["repo"]["path"] or execution["binary"] != manifest["binary"]["path"]:
        raise ArtifactError(f"{case_id} execution cwd/binary mismatch")
    if execution["argv"] != [manifest["binary"]["path"], *common_args]:
        raise ArtifactError(f"{case_id} execution argv differs from manifest workload")
    if execution["env_set"] != environment(run_dir):
        raise ArtifactError(f"{case_id} execution env_set differs from environment artifact")
    if sorted(set(execution["env_unset"])) != execution["env_unset"]:
        raise ArtifactError(f"{case_id} execution env_unset is not sorted unique")
    timeout = execution["timeout"]
    if not isinstance(timeout, dict) or timeout.get("seconds") != manifest["execution"]["timeout_sec"] \
            or timeout.get("term_signal") != "TERM" or timeout.get("kill_after_seconds") != 10:
        raise ArtifactError(f"{case_id} execution timeout mismatch")
    if read(run_dir / "command") != normalized_command(execution):
        raise ArtifactError(f"{case_id} command is not the normalized execution.json command")


def graph_reuse_count(stderr: str, case_id: str) -> int:
    telemetry = fields(unique_line(stderr, "KV_GRAPH_REUSE_STATS"))
    return integer(telemetry, "n_reused")


def parse_run(run_dir: Path, manifest: dict[str, Any], baseline: Path, e2: Path) -> dict[str, Any]:
    meta = json.loads(read(run_dir / "run.json"))
    case_id = str(meta["case"])
    validate_environment(run_dir, case_id)
    validate_execution(run_dir, manifest, case_id)
    if read(run_dir / "exit_code").strip() != "0":
        raise ArtifactError(f"{case_id} did not exit successfully")
    seq0 = (run_dir / "seq0").read_bytes()
    seq1 = (run_dir / "seq1").read_bytes()
    validate_sequences(run_dir)
    if not seq0 or not seq1:
        raise ArtifactError(f"{case_id} has empty output evidence")
    if seq0 != (baseline / "seq0").read_bytes() or seq1 != (baseline / "seq1").read_bytes():
        raise ArtifactError(f"{case_id} output differs from E0")
    if case_id.startswith("E2I") and (seq0 != (e2 / "seq0").read_bytes() or seq1 != (e2 / "seq1").read_bytes()):
        raise ArtifactError(f"{case_id} output differs from E2")

    stderr = read(run_dir / "stderr")
    if "Segmentation fault" in stderr or "GGML_ASSERT" in stderr:
        raise ArtifactError(f"{case_id} contains a fatal signature")
    n_reused = graph_reuse_count(stderr, case_id)
    if case_id == "E2I" and n_reused <= 0:
        raise ArtifactError("E2I requires n_reused>0")
    if case_id == "E2I_NOREUSE" and n_reused != 0:
        raise ArtifactError("E2I_NOREUSE requires n_reused=0")
    if case_id == "E0":
        if "KV paged metadata stats:" in stderr:
            raise ArtifactError("E0 unexpectedly emitted paged telemetry")
        return {"case": case_id, "result": "PASS", "path": "baseline", "n_reused": n_reused}

    telemetry = fields(unique_line(stderr, "KV paged metadata stats:"))
    for name in SAFETY_ZERO:
        if integer(telemetry, name) != 0:
            raise ArtifactError(f"{case_id} safety field is nonzero: {name}")
    write_to_swapped = integer(telemetry, "paged_write_to_swapped_block")
    if case_id == "SWAP":
        write_swapped_hits = integer(telemetry, "paged_swap_write_swapped_hits")
        write_swap_in_successes = integer(telemetry, "paged_swap_write_swap_in_calls")
        write_swap_in_failures = integer(telemetry, "paged_swap_write_swap_in_failures")
        swap_in_calls = integer(telemetry, "paged_swap_in_calls")
        if write_to_swapped != write_swapped_hits or write_swapped_hits != (
                write_swap_in_successes + write_swap_in_failures):
            raise ArtifactError("SWAP write-to-swapped counters do not reconcile")
        if write_swap_in_failures != 0:
            raise ArtifactError("SWAP write-triggered swap-in failures are nonzero")
        if write_swap_in_successes > swap_in_calls:
            raise ArtifactError("SWAP write-triggered swap-in successes exceed global swap-in calls")
    elif write_to_swapped != 0:
        raise ArtifactError(f"{case_id} write-to-swapped count is nonzero")
    enabled = integer(telemetry, "paged_identity_fast_path_enabled")
    layers = integer(telemetry, "paged_identity_fast_path_layers")
    gathers = integer(telemetry, "ingraph_gather_layers")
    created = integer(telemetry, "paged_row_idx_inputs_created")
    set_calls = integer(telemetry, "paged_row_idx_set_calls")
    reason = telemetry.get("paged_identity_fast_path_reject_reason")
    if case_id in {"E2I", "E2I_NOREUSE"}:
        if enabled != 1 or layers <= 0 or reason != "none":
            raise ArtifactError(f"{case_id} did not enable the identity fast path")
        if gathers != 0 or created != 0 or set_calls != 0:
            raise ArtifactError(f"{case_id} created/filled row indices or executed gather")
        path = "continuous_view"
    else:
        if enabled != 0 or layers != 0 or reason != REJECT[case_id]:
            raise ArtifactError(f"{case_id} reject telemetry mismatch")
        if gathers <= 0 or created <= 0 or set_calls <= 0:
            raise ArtifactError(f"{case_id} did not fall back to row-index gather from its first graph")
        path = "gather_fallback"
    return {"case": case_id, "result": "PASS", "path": path,
            "reject_reason": reason, "n_reused": n_reused}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    try:
        manifest = json.loads(read(root / "manifest.json"))
        run_dirs, _ = validate_matrix(root, manifest, args.dry_run)
        if args.dry_run:
            for run_dir in run_dirs:
                case_id = str(json.loads(read(run_dir / "run.json"))["case"])
                validate_environment(run_dir, case_id)
                validate_execution(run_dir, manifest, case_id)
            (root / "summary.json").write_text(json.dumps({"result": "DRY_RUN", "cases": list(CASES)}, indent=2) + "\n")
            return 0
        baseline = root / "runs" / "E0"
        e2 = root / "runs" / "E2"
        results = [parse_run(run_dir, manifest, baseline, e2) for run_dir in run_dirs]
        summary = {"result": "PASS", "runs": results,
                   "correctness": "all outputs byte-exact to E0; E2I variants also byte-exact to E2"}
        (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return 0
    except (ArtifactError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
