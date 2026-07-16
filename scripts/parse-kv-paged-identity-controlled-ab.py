#!/usr/bin/env python3
"""Fail-closed parser for the frozen three-round E2G/E2I experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


PLAN = (
    (1, 1, "E2G"), (1, 2, "E2I"),
    (2, 1, "E2I"), (2, 2, "E2G"),
    (3, 1, "E2G"), (3, 2, "E2I"),
)
CASES = ("E2G", "E2I")
METRICS = ("tpot_ms", "tps", "p95_ms", "wall_ms")
LOWER_IS_BETTER = {"tpot_ms", "p95_ms", "wall_ms"}
RAW_ARTIFACTS = (
    "run.json", "execution.json", "environment.json", "replay.sh", "stdout", "stderr",
    "exit_code", "seq0", "seq1", "process.json", "execution_result.json", "rss_samples.tsv",
)
SEQUENCES = (
    ("seq1", b"===SEQ1_ACTIVE_BEGIN===", b"===SEQ1_ACTIVE_END==="),
    ("seq0", b"===SEQ0_RESUME_BEGIN===", b"===SEQ0_RESUME_END==="),
)
SAFETY_ZERO = (
    "identity_fail", "write_resolve_fail", "row_idx_fail", "mapping_oob_fail",
    "logical_to_physical_fail", "paged_row_mapping_invalid_fatal",
    "paged_write_mapping_invalid_fatal", "paged_active_row_nonresident_fatal",
    "paged_input_setup_fatal", "paged_swapped_active_visible_violation",
    "paged_swapped_active_visible_violation_rows", "paged_swapped_active_visible_violation_blocks",
    "paged_write_to_swapped_block", "paged_swap_backend_failures",
    "paged_swap_read_swap_in_failures", "paged_swap_write_swap_in_failures",
    "paged_swap_in_fail_no_offset", "paged_swap_in_fail_bad_size",
    "paged_swap_in_fail_read_cell", "paged_swap_in_fail_tensor_set",
    "paged_prefetch_seq_failures", "paged_block_release_fail", "paged_swap_madvise_failures",
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


def identity(item: Any, label: str, verify_live: bool = True) -> Path:
    if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
        raise ArtifactError(f"invalid {label} identity")
    path = Path(item["path"])
    if not isinstance(item["size"], int) or item["size"] < 0 \
            or not isinstance(item["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None:
        raise ArtifactError(f"invalid {label} identity fields")
    if verify_live and (not path.is_file() or path.stat().st_size != item["size"] or sha256(path) != item["sha256"]):
        raise ArtifactError(f"{label} identity mismatch")
    return path


def fields(line: str) -> dict[str, str]:
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line)
    result: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate telemetry key {key}")
        result[key] = value
    return result


def unique(text: str, marker: str) -> dict[str, str]:
    lines = [line for line in text.splitlines() if marker in line]
    if len(lines) != 1:
        raise ArtifactError(f"telemetry marker {marker!r} count is {len(lines)}, expected 1")
    return fields(lines[0])


def number(source: dict[str, str], name: str) -> float:
    try:
        value = float(source[name])
    except (KeyError, ValueError) as exc:
        raise ArtifactError(f"missing or invalid numeric telemetry {name}") from exc
    if not math.isfinite(value):
        raise ArtifactError(f"non-finite telemetry {name}")
    return value


def integer(source: dict[str, str], name: str) -> int:
    value = number(source, name)
    if not value.is_integer():
        raise ArtifactError(f"telemetry {name} is not an integer")
    return int(value)


def run_key(item: Any, label: str) -> tuple[int, int, str]:
    if not isinstance(item, dict):
        raise ArtifactError(f"{label} must be an object")
    try:
        key = int(item["round"]), int(item["run_order"]), str(item["case"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid {label} matrix fields") from exc
    if key[2] not in CASES:
        raise ArtifactError(f"unknown case {key[2]}")
    return key


def load_protocol_runner(manifest: dict[str, Any]) -> Any:
    runner_path = identity(manifest["framework"]["runner"], "framework.runner")
    spec = importlib.util.spec_from_file_location("identity_controlled_runner", runner_path)
    if spec is None or spec.loader is None:
        raise ArtifactError("cannot load frozen runner protocol")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_manifest(root: Path, manifest: dict[str, Any], dry_run: bool) -> Any:
    if manifest.get("protocol") != "kv_paged_identity_controlled_ab" or manifest.get("version") != 2:
        raise ArtifactError("manifest protocol/version mismatch")
    repo = manifest.get("repo")
    if not isinstance(repo, dict) or not isinstance(repo.get("path"), str) \
            or not isinstance(repo.get("head"), str) or not isinstance(repo.get("dirty"), bool) \
            or not isinstance(repo.get("status_porcelain"), list):
        raise ArtifactError("invalid repository provenance")
    try:
        current_head = subprocess.check_output(["git", "-C", repo["path"], "rev-parse", "HEAD"], text=True,
                                               stderr=subprocess.DEVNULL).strip()
        current_status = subprocess.check_output(["git", "-C", repo["path"], "status", "--porcelain"], text=True,
                                                 stderr=subprocess.DEVNULL).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactError("cannot verify repository provenance") from exc
    if current_head != repo["head"] or current_status != repo["status_porcelain"] or bool(current_status) != repo["dirty"]:
        raise ArtifactError("repository provenance mismatch")
    execution = manifest.get("execution")
    if not isinstance(execution, dict) or execution.get("dry_run") is not dry_run:
        raise ArtifactError("parser mode differs from manifest")
    if not dry_run and (repo["dirty"] or execution.get("allow_dirty") is not False):
        raise ArtifactError("formal experiment requires a clean worktree")
    identity(manifest.get("binary"), "binary")
    identity(manifest.get("model"), "model")
    framework = manifest.get("framework")
    if not isinstance(framework, dict) or set(framework) != {"wrapper", "runner", "parser"}:
        raise ArtifactError("invalid framework identity set")
    for name in framework:
        identity(framework[name], f"framework.{name}")
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict) or set(provenance) != {"host", "build"}:
        raise ArtifactError("missing host/build provenance")
    host, build = provenance["host"], provenance["build"]
    if not isinstance(host, dict) or not all(key in host for key in (
            "hostname", "os", "kernel", "machine", "cpu", "memory", "affinity_cpus", "cgroup")):
        raise ArtifactError("incomplete host/OS/CPU/memory/affinity/cgroup provenance")
    if not isinstance(build, dict) or not all(key in build for key in (
            "compiler", "build_type", "key_config", "cmake_cache")):
        raise ArtifactError("incomplete compiler/build/CMake provenance")
    runner = load_protocol_runner(manifest)
    if tuple(runner.PLAN) != PLAN:
        raise ArtifactError("runner plan differs from frozen parser plan")
    workload = manifest.get("workload")
    if not isinstance(workload, dict) or workload.get("argv_after_model") != list(runner.COMMON_ARGS) \
            or workload.get("rounds") != 3 or workload.get("order") != "R1 G-I, R2 I-G, R3 G-I":
        raise ArtifactError("workload differs from frozen protocol")
    planned = manifest.get("planned_runs")
    if not isinstance(planned, list) or tuple(run_key(item, f"planned_runs[{i}]") for i, item in enumerate(planned)) != PLAN:
        raise ArtifactError("planned runs differ from frozen protocol")
    order_path = root / "execution_order.log"
    identity(manifest.get("execution_order"), "execution_order", verify_live=False)
    if not order_path.is_file() or order_path.stat().st_size != manifest["execution_order"]["size"] \
            or sha256(order_path) != manifest["execution_order"]["sha256"]:
        raise ArtifactError("execution order artifact mismatch")
    observed = []
    for line in read(order_path).splitlines():
        match = re.fullmatch(r"round=(\d+) run_order=(\d+) case=(E2G|E2I)", line)
        if not match:
            raise ArtifactError("malformed execution order")
        observed.append((int(match[1]), int(match[2]), match[3]))
    if tuple(observed) != PLAN:
        raise ArtifactError("observed execution order differs from frozen protocol")
    return runner


def run_dirs(root: Path, manifest: dict[str, Any]) -> tuple[list[Path], dict[tuple[int, int, str], dict[str, Any]]]:
    actual: list[tuple[tuple[int, int, str], Path]] = []
    for path in (root / "runs").iterdir():
        if path.is_dir():
            try:
                meta = json.loads(read(path / "run.json"))
            except json.JSONDecodeError as exc:
                raise ArtifactError(f"invalid {path}/run.json") from exc
            actual.append((run_key(meta, str(path)), path))
    actual.sort(key=lambda item: item[0][:2])
    if tuple(key for key, _ in actual) != PLAN:
        raise ArtifactError("run directories differ from frozen plan")
    completed = manifest.get("completed_runs")
    if not isinstance(completed, list) or len(completed) != len(PLAN):
        raise ArtifactError("completed run manifest is incomplete")
    completed_map: dict[tuple[int, int, str], dict[str, Any]] = {}
    for index, item in enumerate(completed):
        key = run_key(item, f"completed_runs[{index}]")
        if key in completed_map:
            raise ArtifactError("duplicate completed run")
        completed_map[key] = item
    if tuple(completed_map) != PLAN:
        raise ArtifactError("completed run order differs from frozen plan")
    for key, path in actual:
        identities = completed_map[key].get("artifacts")
        if not isinstance(identities, dict) or set(identities) != set(RAW_ARTIFACTS):
            raise ArtifactError(f"{path.name} artifact set mismatch")
        for name in RAW_ARTIFACTS:
            artifact = path / name
            item = identities[name]
            if not artifact.is_file() or item.get("size") != artifact.stat().st_size or item.get("sha256") != sha256(artifact):
                raise ArtifactError(f"{path.name} artifact identity mismatch: {name}")
    return [path for _, path in actual], completed_map


def validate_replay(path: Path, execution: dict[str, Any]) -> None:
    text = read(path)
    if not text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n") or "env -i " not in text:
        raise ArtifactError(f"{path} is not an execution-derived readable replay")
    for token in (execution["cwd"], execution["binary"], str(execution["timeout"]["seconds"]),
                  str(execution["timeout"]["termination_grace_seconds"])):
        if shlex_quote(token) not in text and token not in text:
            raise ArtifactError(f"{path} does not reflect execution.json")


def shlex_quote(value: str) -> str:
    import shlex
    return shlex.quote(value)


def validate_execution(path: Path, manifest: dict[str, Any], runner: Any, dry_run: bool) -> dict[str, Any]:
    try:
        execution = json.loads(read(path / "execution.json"))
        recorded_env = json.loads(read(path / "environment.json"))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid execution/environment JSON in {path.name}") from exc
    if set(execution) != {"cwd", "binary", "argv", "env", "timeout", "auxiliary"}:
        raise ArtifactError(f"{path.name} execution field set mismatch")
    if execution["cwd"] != manifest["repo"]["path"] or execution["binary"] != manifest["binary"]["path"]:
        raise ArtifactError(f"{path.name} cwd/binary mismatch")
    expected_argv = [manifest["binary"]["path"], "-m", manifest["model"]["path"], *runner.COMMON_ARGS]
    if execution["argv"] != expected_argv:
        raise ArtifactError(f"{path.name} argv differs from frozen workload")
    if execution["env"] != recorded_env:
        raise ArtifactError(f"{path.name} Popen/artifact environment objects differ")
    timeout = execution["timeout"]
    protocol_execution = manifest["execution"]
    if timeout != {"seconds": protocol_execution["timeout_seconds"], "term_signal": "TERM",
                   "termination_grace_seconds": protocol_execution["termination_grace_seconds"]}:
        raise ArtifactError(f"{path.name} timeout differs from manifest")
    expected_aux = {"rss": {"enabled": protocol_execution["rss_auxiliary_enabled"],
                            "sample_interval_seconds": protocol_execution["rss_sample_interval_seconds"]}}
    if execution["auxiliary"] != expected_aux:
        raise ArtifactError(f"{path.name} auxiliary configuration mismatch")
    meta = json.loads(read(path / "run.json"))
    case = meta["case"]
    expected_env = dict(runner.BASE_ENV_DEFAULTS)
    expected_env.update(runner.PROTOCOL_ENV)
    expected_env["LLAMA_KV_PAGED_IDENTITY_FAST_PATH"] = "1" if case == "E2I" else "0"
    expected_env["LLAMA_KV_SWAP_DIR"] = str((path.parent.parent / "swap").resolve())
    if execution["env"] != dict(sorted(expected_env.items())):
        raise ArtifactError(f"{path.name} complete environment differs from fixed allowlist")
    validate_replay(path / "replay.sh", execution)
    exit_text = read(path / "exit_code").strip()
    if dry_run:
        if exit_text != "DRY_RUN" or read(path / "process.json") or read(path / "execution_result.json"):
            raise ArtifactError(f"{path.name} is not a clean dry-run artifact")
    else:
        try:
            process = json.loads(read(path / "process.json"))
            result = json.loads(read(path / "execution_result.json"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{path.name} process/result artifact invalid") from exc
        if process.get("argv") != execution["argv"] or process.get("cwd") != execution["cwd"] \
                or process.get("env") != execution["env"]:
            raise ArtifactError(f"{path.name} actual process differs from execution.json")
        if result.get("returncode") != 0 or exit_text != "0" or result.get("timed_out") is not False \
                or result.get("term_sent") is not False or result.get("kill_sent") is not False:
            raise ArtifactError(f"{path.name} did not exit normally")
    return execution


def validate_sequences(path: Path) -> None:
    lines = (path / "stdout").read_bytes().splitlines(keepends=True)
    normalized = [line.rstrip(b"\r\n") for line in lines]
    positions: list[int] = []
    for name, begin, end in SEQUENCES:
        if normalized.count(begin) != 1 or normalized.count(end) != 1:
            raise ArtifactError(f"{path.name} {name} marker count mismatch")
        left, right = normalized.index(begin), normalized.index(end)
        if left >= right:
            raise ArtifactError(f"{path.name} {name} marker order mismatch")
        content = b"".join(lines[left + 1:right])
        if not content or content != (path / name).read_bytes():
            raise ArtifactError(f"{path.name} {name} evidence mismatch")
        positions.extend((left, right))
    if positions != sorted(positions):
        raise ArtifactError(f"{path.name} sequence sections overlap/out of order")


def parse_run(path: Path, reference: Path) -> dict[str, Any]:
    meta = json.loads(read(path / "run.json"))
    round_no, order, case = run_key(meta, str(path))
    validate_sequences(path)
    for name in ("seq0", "seq1"):
        if (path / name).read_bytes() != (reference / name).read_bytes():
            raise ArtifactError(f"{path.name} output differs from paired run")
    stderr = read(path / "stderr")
    summary = unique(stderr, "KV_TEST_SUMMARY ")
    if summary.get("result") != "PASS":
        raise ArtifactError(f"{path.name} KV_TEST_SUMMARY is not PASS")
    perf = unique(stderr, "KV_IDLE_SWAP_RESUME_PERF ")
    active = unique(stderr, "KV_ACTIVE_TOKEN_STATS ")
    reuse = unique(stderr, "KV_GRAPH_REUSE_STATS ")
    metadata = unique(stderr, "KV paged metadata stats:")
    if integer(active, "active_token_count") <= 0 or integer(reuse, "n_reused") <= 0:
        raise ArtifactError(f"{path.name} lacks active samples or graph reuse")
    for name in SAFETY_ZERO:
        if integer(metadata, name) != 0:
            raise ArtifactError(f"{path.name} failure field is nonzero: {name}")
    for name in ("paged_swap_enabled", "paged_idle_swap_enabled", "paged_block_release_enabled"):
        if integer(metadata, name) != 0:
            raise ArtifactError(f"{path.name} unexpectedly enabled {name}")
    enabled = integer(metadata, "paged_identity_fast_path_enabled")
    layers = integer(metadata, "paged_identity_fast_path_layers")
    reason = metadata.get("paged_identity_fast_path_reject_reason")
    gathers = integer(metadata, "ingraph_gather_layers")
    created = integer(metadata, "paged_row_idx_inputs_created")
    set_calls = integer(metadata, "paged_row_idx_set_calls")
    nonidentity = integer(metadata, "non_identity_enabled")
    if case == "E2G":
        if enabled != 0 or layers != 0 or reason != "not_requested" or nonidentity != 0 \
                or gathers <= 0 or created <= 0 or set_calls <= 0:
            raise ArtifactError(f"{path.name} did not prove row-index/gather execution")
    elif enabled != 1 or layers <= 0 or reason != "none" or nonidentity != 0 \
            or gathers != 0 or created != 0 or set_calls != 0:
        raise ArtifactError(f"{path.name} did not prove identity fast-path topology")
    row = {
        "round": round_no, "run_order": order, "case": case,
        "tpot_ms": number(active, "avg_ms"), "tps": number(perf, "tokens_per_second"),
        "p95_ms": number(active, "p95_ms"), "wall_ms": number(perf, "total_wall_ms"),
    }
    if any(row[name] <= 0 for name in METRICS):
        raise ArtifactError(f"{path.name} has nonpositive core metric")
    return row


def rounded(value: float) -> float:
    return round(value, 6)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    paired: list[dict[str, Any]] = []
    for round_no in range(1, 4):
        cases = {row["case"]: row for row in rows if row["round"] == round_no}
        if set(cases) != set(CASES):
            raise ArtifactError(f"round {round_no} is not pairable")
        raw = {case: {metric: cases[case][metric] for metric in METRICS} for case in CASES}
        delta = {metric: rounded(cases["E2I"][metric] - cases["E2G"][metric]) for metric in METRICS}
        favorable = {metric: delta[metric] < 0 if metric in LOWER_IS_BETTER else delta[metric] > 0 for metric in METRICS}
        paired.append({"round": round_no, "raw": raw, "paired_delta_e2i_minus_e2g": delta, "favorable": favorable})
    aggregate: dict[str, Any] = {}
    all_favorable = True
    all_unfavorable = True
    for metric in METRICS:
        metric_result: dict[str, Any] = {}
        for case in CASES:
            values = [float(row[metric]) for row in rows if row["case"] == case]
            metric_result[case] = {"median": rounded(statistics.median(values)),
                                   "min": rounded(min(values)), "max": rounded(max(values))}
        deltas = [float(item["paired_delta_e2i_minus_e2g"][metric]) for item in paired]
        favorable_count = sum(bool(item["favorable"][metric]) for item in paired)
        metric_result["paired_delta_e2i_minus_e2g"] = {
            "median": rounded(statistics.median(deltas)), "min": rounded(min(deltas)), "max": rounded(max(deltas))}
        metric_result["direction_consistency"] = {
            "favorable_rounds": favorable_count, "total_rounds": 3,
            "all_rounds_favorable": favorable_count == 3,
            "all_rounds_unfavorable_or_equal": favorable_count == 0,
        }
        all_favorable &= favorable_count == 3
        all_unfavorable &= favorable_count == 0
        aggregate[metric] = metric_result
    judgment = "FAST_PATH_FASTER" if all_favorable else "FAST_PATH_NOT_FASTER" if all_unfavorable else "MIXED"
    return {
        "artifact_status": "VALID", "performance_judgment": judgment,
        "decision_rule": "FAST_PATH_FASTER only when all four core metrics favor E2I in all three paired rounds",
        "metrics": list(METRICS), "runs": rows, "paired_rounds": paired, "aggregate": aggregate,
        "correctness": "paired outputs byte-identical; unique KV_TEST_SUMMARY result=PASS; failure fields zero",
    }


def write_tables(root: Path, summary: dict[str, Any]) -> None:
    with (root / "runs.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("round", "run_order", "case", *METRICS),
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["runs"])
    with (root / "paired_deltas.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("round", "metric", "E2G_raw", "E2I_raw", "E2I_minus_E2G", "favorable"))
        for item in summary["paired_rounds"]:
            for metric in METRICS:
                writer.writerow((item["round"], metric, item["raw"]["E2G"][metric], item["raw"]["E2I"][metric],
                                 item["paired_delta_e2i_minus_e2g"][metric], item["favorable"][metric]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    try:
        manifest = json.loads(read(root / "manifest.json"))
        runner = validate_manifest(root, manifest, args.dry_run)
        paths, _ = run_dirs(root, manifest)
        for path in paths:
            validate_execution(path, manifest, runner, args.dry_run)
        if args.dry_run:
            (root / "summary.json").write_text(json.dumps({"artifact_status": "DRY_RUN", "plan": PLAN}, indent=2) + "\n")
            return 0
        references = {round_no: next(path for path in paths if json.loads(read(path / "run.json"))["round"] == round_no)
                      for round_no in range(1, 4)}
        rows = [parse_run(path, references[json.loads(read(path / "run.json"))["round"]]) for path in paths]
        summary = summarize(rows)
        (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        write_tables(root, summary)
        print(f"VALID performance_judgment={summary['performance_judgment']}")
        return 0
    except (ArtifactError, OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
