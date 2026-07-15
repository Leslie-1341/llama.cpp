#!/usr/bin/env python3
"""Parse the controlled KV E0-E5 workload without inventing unavailable metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable


NA = "NA"
CASES = ["E0", "E1", "E2", "E3", "E4", "E5"]
CASE_CONFIG = {
    "E0": "optimization off baseline",
    "E1": "lazy clear + lazy tail; paged path off",
    "E2": "paged row-index + in-graph gather infrastructure only",
    "E3": "E2 + idle paged swap/backing store; madvise/prefetch off",
    "E4": "E3 + idle-swap madvise; prefetch off",
    "E5": "E4 + delayed active prefetch (1 block every token) + resume defer",
}

RUN_COLUMNS = [
    "round", "run_order", "case", "label", "exit_code", "result", "correctness", "mechanism_verification",
    "seq0_exact", "seq1_exact", "seq0_sha256", "seq1_sha256",
    "active_visible_safety_violations", "active_visible_safety_violation_rows",
    "active_visible_safety_violation_blocks", "active_restore_required_rows", "active_restore_required_blocks",
    "write_swapped_violations", "fatal", "pending", "backend_io_failures",
    "active_token_count", "swapped_blocks", "swap_out_calls", "swap_in_calls", "restored_blocks",
    "fallback_blocks", "madvise_blocks", "madvise_bytes", "prefetch_calls", "prefetch_api_calls",
    "prefetch_restored_blocks", "prefetch_total_ms", "prefetch_max_ms",
    "backing_io_write_bytes", "backing_io_read_bytes", "backing_file_logical_size",
    "backing_file_allocated_bytes", "backing_capacity_bytes",
    "active_token_avg_ms", "active_token_p50_ms", "active_token_p95_ms", "active_token_p99_ms",
    "active_token_max_ms", "decode_avg_ms", "active_phase_wall_ms", "resume_first_token_ms", "total_wall_ms", "tps",
    "swap_out_avg_us", "swap_out_max_us", "swap_in_avg_us", "swap_in_max_us",
    "swap_out_validate_us", "swap_out_pack_us", "swap_out_write_us", "swap_out_metadata_us", "swap_out_madvise_us",
    "swap_in_validate_us", "swap_in_read_us", "swap_in_unpack_us", "swap_in_commit_us",
    "rss_before_active_prefetch_kb", "rss_after_active_prefetch_kb", "rss_before_prefetch_kb", "rss_after_prefetch_kb",
    "rss_before_resume_kb", "rss_after_resume_kb", "process_vmrss_sampled_max_kb", "process_vmhwm_kb",
    "kv_resident_bytes", "kv_nonresident_bytes", "mincore_enabled", "cgroup_memory_current_sampled_max_bytes",
    "cgroup_memory_peak_bytes", "lazy_clear_enabled", "lazy_tail_enabled", "paged_enabled", "ingraph_gather_layers",
    "nonidentity_enabled", "nonidentity_remap_rows", "paged_release_enabled", "paged_swap_enabled",
    "idle_swap_enabled", "madvise_enabled",
    "prefetch_auto_started", "defer_requested", "warnings",
]

SUMMARY_METRICS = [
    "active_token_avg_ms", "active_token_p50_ms", "active_token_p95_ms", "active_token_p99_ms",
    "active_token_max_ms", "decode_avg_ms", "active_phase_wall_ms", "resume_first_token_ms", "total_wall_ms", "tps",
    "rss_before_resume_kb", "rss_after_resume_kb", "process_vmrss_sampled_max_kb", "process_vmhwm_kb",
    "cgroup_memory_current_sampled_max_bytes", "swapped_blocks", "madvise_bytes", "prefetch_restored_blocks",
    "active_restore_required_rows", "active_restore_required_blocks", "fallback_blocks",
    "backing_file_logical_size", "backing_file_allocated_bytes",
]

RUN_PLANS = {
    1: [
        (1, 1, "E0"), (1, 2, "E1"), (1, 3, "E2"),
        (1, 4, "E3"), (1, 5, "E4"), (1, 6, "E5"),
    ],
    3: [
        (1, 1, "E0"), (1, 2, "E3"), (1, 3, "E1"),
        (1, 4, "E4"), (1, 5, "E2"), (1, 6, "E5"),
        (2, 1, "E5"), (2, 2, "E2"), (2, 3, "E4"),
        (2, 4, "E1"), (2, 5, "E3"), (2, 6, "E0"),
        (3, 1, "E2"), (3, 2, "E5"), (3, 3, "E0"),
        (3, 4, "E3"), (3, 5, "E1"), (3, 6, "E4"),
    ],
}

MARKERS = {
    "perf": "KV_IDLE_SWAP_RESUME_PERF ",
    "active": "KV_ACTIVE_TOKEN_STATS ",
    "metadata": "KV paged metadata stats:",
    "io": "KV_PAGED_IO_STATS ",
    "swap": "KV swap stats:",
    "test": "KV_TEST_SUMMARY ",
    "lazy_clear": "kv lazy-clear stats:",
    "lazy_tail": "kv lazy-tail stats:",
    "backing": "KV swap backing store ready",
}

COMMON_REQUIRED_METRICS = {
    "active_token_count", "active_token_avg_ms", "active_token_p50_ms", "active_token_p95_ms",
    "active_token_p99_ms", "active_token_max_ms", "decode_avg_ms", "active_phase_wall_ms",
    "resume_first_token_ms", "total_wall_ms", "tps", "prefetch_calls", "prefetch_total_ms",
    "prefetch_restored_blocks", "prefetch_max_ms", "fallback_blocks",
    "rss_before_active_prefetch_kb", "rss_after_active_prefetch_kb", "rss_before_prefetch_kb",
    "rss_after_prefetch_kb", "rss_before_resume_kb", "rss_after_resume_kb",
    "process_vmrss_sampled_max_kb", "process_vmhwm_kb", "backend_io_failures",
}

PAGED_REQUIRED_METRICS = {
    "active_visible_safety_violations", "active_visible_safety_violation_rows",
    "active_visible_safety_violation_blocks", "active_restore_required_rows",
    "active_restore_required_blocks", "write_swapped_violations", "fatal", "swapped_blocks",
    "swap_out_calls", "swap_in_calls", "restored_blocks", "madvise_blocks", "madvise_bytes",
    "prefetch_api_calls", "backing_io_write_bytes", "backing_io_read_bytes",
    "swap_out_avg_us", "swap_out_max_us", "swap_in_avg_us", "swap_in_max_us",
    "swap_out_validate_us", "swap_out_pack_us", "swap_out_write_us", "swap_out_metadata_us",
    "swap_out_madvise_us", "swap_in_validate_us", "swap_in_read_us", "swap_in_unpack_us",
    "swap_in_commit_us", "mincore_enabled", "ingraph_gather_layers", "nonidentity_enabled",
    "nonidentity_remap_rows", "paged_release_enabled", "paged_swap_enabled", "idle_swap_enabled",
}


class ArtifactError(ValueError):
    pass


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def unique_line(text: str, marker: str, required: bool = False) -> str:
    lines = [line for line in text.splitlines() if marker in line]
    if len(lines) > 1:
        raise ArtifactError(f"duplicate telemetry marker {marker.strip()!r}")
    if required and not lines:
        raise ArtifactError(f"missing telemetry marker {marker.strip()!r}")
    return lines[0] if lines else ""


def fields(line: str) -> dict[str, str]:
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line)
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ArtifactError(f"duplicate telemetry key(s): {', '.join(duplicates)}")
    return dict(pairs)


def run_tuple(meta: Any, source: str) -> tuple[int, int, str]:
    if not isinstance(meta, dict):
        raise ArtifactError(f"{source} must contain a JSON object")
    missing = [key for key in ("round", "run_order", "case") if key not in meta]
    if missing:
        raise ArtifactError(f"{source} missing matrix field(s): {', '.join(missing)}")
    try:
        item = (int(meta["round"]), int(meta["run_order"]), str(meta["case"]))
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"{source} has invalid round/run_order") from exc
    if item[2] not in CASES:
        raise ArtifactError(f"{source} has unknown case {item[2]!r}")
    return item


def validate_matrix(root: Path, manifest: dict[str, Any]) -> list[Path]:
    try:
        runs = int(manifest["workload"]["runs"])
        planned = manifest["planned_runs"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ArtifactError("manifest missing workload.runs or planned_runs") from exc
    expected = RUN_PLANS.get(runs)
    if expected is None:
        raise ArtifactError(f"manifest workload.runs must be 1 or 3 (got {runs})")
    if not isinstance(planned, list):
        raise ArtifactError("manifest planned_runs must be a list")

    planned_tuples = [run_tuple(meta, f"manifest planned_runs[{index}]") for index, meta in enumerate(planned)]
    if len(set(planned_tuples)) != len(planned_tuples):
        raise ArtifactError("manifest contains a duplicate (round, run_order, case) tuple")
    if planned_tuples != expected:
        raise ArtifactError(f"manifest run matrix does not match the fixed RUNS={runs} plan")

    run_dirs = sorted(path for path in (root / "runs").glob("*") if path.is_dir())
    actual_dirs: list[tuple[tuple[int, int, str], Path]] = []
    for run_dir in run_dirs:
        run_path = run_dir / "run.json"
        if not run_path.is_file():
            raise ArtifactError(f"run directory missing run.json: {run_dir}")
        try:
            meta = json.loads(read_text(run_path))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"invalid JSON in {run_path}") from exc
        actual_dirs.append((run_tuple(meta, str(run_path)), run_dir))
    actual_dirs.sort(key=lambda item: (item[0][0], item[0][1]))
    actual = [item for item, _ in actual_dirs]
    if len(set(actual)) != len(actual):
        raise ArtifactError("runs contain a duplicate (round, run_order, case) tuple")
    if actual != expected:
        raise ArtifactError(f"artifact run matrix does not match the fixed RUNS={runs} plan")
    for round_no in range(1, runs + 1):
        if sum(item[0] == round_no and item[2] == "E0" for item in actual) != 1:
            raise ArtifactError(f"round {round_no} must contain exactly one E0 reference")
    return [run_dir for _, run_dir in actual_dirs]


def value(source: dict[str, str], key: str) -> str:
    candidate = source.get(key, NA)
    return candidate if candidate != "" else NA


def numeric(candidate: Any) -> float | None:
    if candidate in (None, "", NA, "DRY_RUN"):
        return None
    try:
        result = float(candidate)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(candidate: Any) -> int | None:
    number = numeric(candidate)
    if number is None or not number.is_integer():
        return None
    return int(number)


def sum_fields(source: dict[str, str], names: Iterable[str], absent_zero: bool = False) -> str:
    values = [integer(source.get(name)) for name in names]
    if any(item is None for item in values):
        return "0" if absent_zero and not source else NA
    return str(sum(item for item in values if item is not None))


def sha256_file(path: Path) -> str:
    line = read_text(path).strip()
    return line.split()[0] if line else NA


def file_identity(path: Path) -> dict[str, Any]:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": str(path), "size": path.stat().st_size, "sha256": digest.hexdigest()}
    except OSError:
        return {"path": str(path), "size": NA, "sha256": NA}


def refresh_framework_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    repo = Path(manifest.get("repo", {}).get("path", ""))
    paths = {
        "runner": repo / "scripts/kv-final-controlled-e0-e5.sh",
        "parser": repo / "scripts/parse-kv-final-controlled-e0-e5.py",
        "protocol": repo / "docs/kv_final_controlled_e0_e5_protocol.md",
    }
    manifest["framework"] = {name: file_identity(path) for name, path in paths.items()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def max_column(path: Path, column: str) -> str:
    if not path.is_file():
        return NA
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            number = numeric(row.get(column))
            if number is not None:
                values.append(number)
    if not values:
        return NA
    result = max(values)
    return str(int(result)) if result.is_integer() else f"{result:.6f}"


def sequence_exact(run_dir: Path, reference: Path | None, exit_code: str) -> str:
    candidate = run_dir
    if exit_code != "0" or reference is None or not candidate.is_file() or not reference.is_file():
        return NA
    candidate_bytes = candidate.read_bytes()
    reference_bytes = reference.read_bytes()
    if not candidate_bytes or not reference_bytes:
        return NA
    return "YES" if candidate_bytes == reference_bytes else "NO"


def requested_env(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in read_text(run_dir / "environment").splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
            result[key] = val
    return result


def compare_rule(actual: str, operator: str, expected: int) -> bool | None:
    number = integer(actual)
    if number is None:
        return None
    if operator == "eq":
        return number == expected
    if operator == "gt":
        return number > expected
    raise ValueError(operator)


def verification(case_id: str, row: dict[str, str]) -> tuple[str, list[str]]:
    checks: list[tuple[str, bool | None]] = []

    def eq(name: str, expected: int) -> None:
        checks.append((f"{name}={expected}", compare_rule(row.get(name, NA), "eq", expected)))

    def gt(name: str) -> None:
        checks.append((f"{name}>0", compare_rule(row.get(name, NA), "gt", 0)))

    if case_id == "E0":
        eq("paged_enabled", 0)
        eq("lazy_clear_enabled", 0)
        eq("lazy_tail_enabled", 0)
        eq("paged_swap_enabled", 0)
        eq("prefetch_calls", 0)
        eq("madvise_bytes", 0)
    elif case_id == "E1":
        eq("lazy_clear_enabled", 1)
        eq("lazy_tail_enabled", 1)
        eq("paged_enabled", 0)
        eq("paged_swap_enabled", 0)
        eq("prefetch_calls", 0)
    elif case_id == "E2":
        eq("paged_enabled", 1)
        gt("ingraph_gather_layers")
        eq("nonidentity_enabled", 1)
        eq("fatal", 0)
        eq("paged_release_enabled", 0)
        eq("swap_out_calls", 0)
        eq("swap_in_calls", 0)
        eq("madvise_bytes", 0)
        eq("prefetch_calls", 0)
    elif case_id == "E3":
        eq("paged_enabled", 1)
        eq("paged_swap_enabled", 1)
        eq("idle_swap_enabled", 1)
        gt("swap_out_calls")
        gt("swapped_blocks")
        gt("nonidentity_remap_rows")
        gt("backing_io_write_bytes")
        eq("madvise_bytes", 0)
        eq("prefetch_calls", 0)
    elif case_id == "E4":
        eq("paged_enabled", 1)
        eq("paged_swap_enabled", 1)
        eq("idle_swap_enabled", 1)
        gt("swap_out_calls")
        gt("nonidentity_remap_rows")
        gt("madvise_blocks")
        gt("madvise_bytes")
        eq("prefetch_calls", 0)
    elif case_id == "E5":
        eq("paged_enabled", 1)
        eq("paged_swap_enabled", 1)
        eq("idle_swap_enabled", 1)
        gt("swap_out_calls")
        gt("nonidentity_remap_rows")
        gt("madvise_blocks")
        gt("madvise_bytes")
        gt("prefetch_calls")
        gt("prefetch_restored_blocks")
        eq("fallback_blocks", 0)
        eq("prefetch_auto_started", 1)
        eq("defer_requested", 1)

    failed = [name for name, outcome in checks if outcome is False]
    unknown = [name for name, outcome in checks if outcome is None]
    if failed:
        return "FAIL", [f"failed: {', '.join(failed)}"]
    if unknown:
        return "UNVERIFIED", [f"unverified: {', '.join(unknown)}"]
    return "PASS", ["all observable mechanism checks passed"]


def parse_run(run_dir: Path, references: dict[int, tuple[Path, Path]]) -> dict[str, str]:
    meta = json.loads(read_text(run_dir / "run.json"))
    case_id = meta["case"]
    round_no = int(meta["round"])
    stderr = read_text(run_dir / "stderr")
    stdout = read_text(run_dir / "stdout")
    exit_code = read_text(run_dir / "exit_code").strip() or NA
    env = requested_env(run_dir)

    for marker in MARKERS.values():
        unique_line(stderr, marker)

    perf = fields(unique_line(stderr, MARKERS["perf"], required=True))
    active = fields(unique_line(stderr, MARKERS["active"], required=True))
    metadata_line = unique_line(stderr, MARKERS["metadata"], required=case_id in {"E2", "E3", "E4", "E5"})
    metadata = fields(metadata_line)
    io = fields(unique_line(stderr, MARKERS["io"], required=case_id in {"E2", "E3", "E4", "E5"}))
    swap = fields(unique_line(stderr, MARKERS["swap"], required=True))
    test = fields(unique_line(stderr, MARKERS["test"]))
    lazy_clear_line = unique_line(stderr, MARKERS["lazy_clear"], required=case_id == "E1")
    lazy_tail_line = unique_line(stderr, MARKERS["lazy_tail"], required=case_id == "E1")
    backing_line = unique_line(stderr, MARKERS["backing"], required=case_id in {"E3", "E4", "E5"})
    lazy_clear = fields(lazy_clear_line)
    lazy_tail = fields(lazy_tail_line)
    backing = fields(backing_line)

    paged_enabled = "1" if metadata_line else "0"
    lazy_clear_enabled = "1" if integer(value(lazy_clear, "enabled")) == 1 else "0"
    lazy_tail_enabled = "1" if integer(value(lazy_tail, "enabled")) == 1 else "0"
    paged_absent = paged_enabled == "0"

    fatal = sum_fields(metadata, [
        "paged_row_mapping_invalid_fatal", "paged_write_mapping_invalid_fatal",
        "paged_active_row_nonresident_fatal", "paged_input_setup_fatal",
    ], absent_zero=paged_absent)
    active_visible = value(metadata, "paged_swapped_active_visible_violation") if metadata else "0"
    active_visible_rows = value(metadata, "paged_swapped_active_visible_violation_rows") if metadata else "0"
    active_visible_blocks = value(metadata, "paged_swapped_active_visible_violation_blocks") if metadata else "0"
    active_restore_required_rows = value(metadata, "paged_swapped_active_violation_rows") if metadata else "0"
    active_restore_required_blocks = value(metadata, "paged_swapped_active_violation_blocks") if metadata else "0"
    write_swapped = value(metadata, "paged_write_to_swapped_block") if metadata else "0"
    backend_failures = sum_fields(metadata, [
        "paged_swap_backend_failures", "paged_swap_read_swap_in_failures",
        "paged_swap_write_swap_in_failures", "paged_swap_in_fail_no_offset",
        "paged_swap_in_fail_bad_size", "paged_swap_in_fail_read_cell",
        "paged_swap_in_fail_tensor_set", "paged_prefetch_seq_failures",
        "paged_block_release_fail", "paged_swap_madvise_failures",
    ], absent_zero=paged_absent)
    if integer(backend_failures) is not None and integer(value(swap, "backend_failures")) is not None:
        backend_failures = str(integer(backend_failures) + integer(value(swap, "backend_failures")))

    madvise_blocks = value(metadata, "paged_swap_madvise_calls") if metadata else "0"
    madvise_bytes = value(metadata, "paged_swap_madvise_bytes") if metadata else "0"
    swap_out_calls = value(io, "block_swap_out_calls")
    swap_in_calls = value(io, "block_swap_in_calls")
    if swap_out_calls == NA:
        swap_out_calls = value(metadata, "paged_swap_out_calls") if metadata else "0"
    if swap_in_calls == NA:
        swap_in_calls = value(metadata, "paged_swap_in_calls") if metadata else "0"

    capacity_bytes = NA
    capacity_match = re.search(r"capacity=([0-9.]+) MiB", backing_line)
    if capacity_match:
        capacity_bytes = str(round(float(capacity_match.group(1)) * 1024 * 1024))

    reference = references.get(round_no)
    seq0_exact = sequence_exact(run_dir / "seq0", reference[0] if reference else None, exit_code)
    seq1_exact = sequence_exact(run_dir / "seq1", reference[1] if reference else None, exit_code)

    warning_lines = [
        line.strip() for line in stderr.splitlines()
        if ("warning:" in line.lower() or " disabled" in line.lower() or "falling back" in line.lower())
        and "sampling" not in line.lower()
    ]

    row = {column: NA for column in RUN_COLUMNS}
    row.update({
        "round": str(round_no), "run_order": str(meta["run_order"]), "case": case_id, "label": CASE_CONFIG[case_id],
        "exit_code": exit_code, "seq0_exact": seq0_exact, "seq1_exact": seq1_exact,
        "seq0_sha256": sha256_file(run_dir / "seq0.sha256"), "seq1_sha256": sha256_file(run_dir / "seq1.sha256"),
        "active_visible_safety_violations": active_visible,
        "active_visible_safety_violation_rows": active_visible_rows,
        "active_visible_safety_violation_blocks": active_visible_blocks,
        "active_restore_required_rows": active_restore_required_rows,
        "active_restore_required_blocks": active_restore_required_blocks,
        "write_swapped_violations": write_swapped,
        "fatal": fatal, "pending": NA, "backend_io_failures": backend_failures,
        "active_token_count": value(active, "active_token_count"),
        "swapped_blocks": value(metadata, "paged_blocks_swapped_out") if metadata else "0",
        "swap_out_calls": swap_out_calls, "swap_in_calls": swap_in_calls,
        "restored_blocks": value(metadata, "paged_blocks_swapped_in") if metadata else "0",
        "fallback_blocks": value(perf, "resume_pending_fallback_blocks"),
        "madvise_blocks": madvise_blocks, "madvise_bytes": madvise_bytes,
        "prefetch_calls": value(active, "prefetch_calls"),
        "prefetch_api_calls": value(metadata, "paged_prefetch_seq_calls") if metadata else "0",
        "prefetch_restored_blocks": value(perf, "prefetch_during_active_blocks"),
        "prefetch_total_ms": value(active, "prefetch_total_ms"),
        "prefetch_max_ms": value(perf, "prefetch_during_active_ms_max"),
        "backing_io_write_bytes": value(io, "bytes_written"), "backing_io_read_bytes": value(io, "bytes_read"),
        "backing_file_logical_size": max_column(run_dir / "memory_samples.tsv", "backing_logical_size"),
        "backing_file_allocated_bytes": max_column(run_dir / "memory_samples.tsv", "backing_allocated_bytes"),
        "backing_capacity_bytes": capacity_bytes,
        "active_token_avg_ms": value(active, "avg_ms"), "active_token_p50_ms": value(active, "p50_ms"),
        "active_token_p95_ms": value(active, "p95_ms"), "active_token_p99_ms": value(active, "p99_ms"),
        "active_token_max_ms": value(active, "max_ms"), "decode_avg_ms": value(active, "decode_avg_ms"),
        "active_phase_wall_ms": value(perf, "seq1_active_ms"),
        "resume_first_token_ms": value(perf, "seq0_resume_first_token_ms"),
        "total_wall_ms": value(perf, "total_wall_ms"), "tps": value(perf, "tokens_per_second"),
        "swap_out_avg_us": value(io, "avg_block_swap_out_latency_us"),
        "swap_out_max_us": value(io, "max_block_swap_out_latency_us"),
        "swap_in_avg_us": value(io, "avg_block_swap_in_latency_us"),
        "swap_in_max_us": value(io, "max_block_swap_in_latency_us"),
        "swap_out_validate_us": value(io, "block_out_validate_us"), "swap_out_pack_us": value(io, "block_out_pack_us"),
        "swap_out_write_us": value(io, "block_out_write_us"), "swap_out_metadata_us": value(io, "block_out_metadata_us"),
        "swap_out_madvise_us": value(io, "block_out_madvise_us"), "swap_in_validate_us": value(io, "block_in_validate_us"),
        "swap_in_read_us": value(io, "block_in_read_us"), "swap_in_unpack_us": value(io, "block_in_unpack_us"),
        "swap_in_commit_us": value(io, "block_in_commit_us"),
        "rss_before_active_prefetch_kb": value(perf, "rss_before_active_prefetch_kb"),
        "rss_after_active_prefetch_kb": value(perf, "rss_after_active_prefetch_kb"),
        "rss_before_prefetch_kb": value(perf, "rss_before_prefetch_kb"),
        "rss_after_prefetch_kb": value(perf, "rss_after_prefetch_kb"),
        "rss_before_resume_kb": value(perf, "rss_before_resume_kb"), "rss_after_resume_kb": value(perf, "rss_after_resume_kb"),
        "process_vmrss_sampled_max_kb": max_column(run_dir / "memory_samples.tsv", "vmrss_kb"),
        "process_vmhwm_kb": max_column(run_dir / "memory_samples.tsv", "vmhwm_kb"),
        "kv_resident_bytes": NA, "kv_nonresident_bytes": NA,
        "mincore_enabled": value(metadata, "kv_mincore_enabled") if metadata else "0",
        "cgroup_memory_current_sampled_max_bytes": max_column(run_dir / "memory_samples.tsv", "cgroup_memory_current_bytes"),
        "cgroup_memory_peak_bytes": NA,
        "lazy_clear_enabled": lazy_clear_enabled, "lazy_tail_enabled": lazy_tail_enabled,
        "paged_enabled": paged_enabled, "ingraph_gather_layers": value(metadata, "ingraph_gather_layers"),
        "nonidentity_enabled": value(metadata, "paged_nonidentity_enabled"),
        "nonidentity_remap_rows": value(metadata, "paged_nonidentity_remap_rows"),
        "paged_release_enabled": value(metadata, "paged_block_release_enabled") if metadata else "0",
        "paged_swap_enabled": value(metadata, "paged_swap_enabled") if metadata else "0",
        "idle_swap_enabled": value(metadata, "paged_idle_swap_enabled") if metadata else "0",
        "madvise_enabled": "1" if integer(madvise_blocks) is not None and integer(madvise_blocks) > 0 else "0",
        "prefetch_auto_started": value(perf, "prefetch_auto_started"),
        "defer_requested": env.get("LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME", NA),
        "warnings": " | ".join(warning_lines) if warning_lines else "none",
    })

    required_metrics = set(COMMON_REQUIRED_METRICS)
    if case_id in {"E2", "E3", "E4", "E5"}:
        required_metrics.update(PAGED_REQUIRED_METRICS)
    if case_id in {"E3", "E4", "E5"}:
        required_metrics.add("backing_capacity_bytes")
    if integer(exit_code) is None:
        required_metrics.add("exit_code")
    missing_metrics = {name for name in required_metrics if numeric(row.get(name)) is None}
    required_source_fields = {
        "perf": {
            "total_wall_ms", "seq1_active_ms", "seq0_resume_first_token_ms", "tokens_per_second",
            "prefetch_during_active_blocks", "prefetch_during_active_ms_max", "prefetch_auto_started",
            "resume_pending_fallback_blocks", "prefetch_failures", "rss_before_active_prefetch_kb",
            "rss_after_active_prefetch_kb", "rss_before_prefetch_kb", "rss_after_prefetch_kb",
            "rss_before_resume_kb", "rss_after_resume_kb",
        },
        "active": {
            "active_token_count", "avg_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
            "decode_avg_ms", "prefetch_calls", "prefetch_total_ms",
        },
        "swap": {"backend_failures"},
    }
    sources = {"perf": perf, "active": active, "swap": swap, "metadata": metadata, "io": io}
    if case_id in {"E2", "E3", "E4", "E5"}:
        required_source_fields["metadata"] = {
            "ingraph_gather_layers", "paged_nonidentity_enabled", "paged_nonidentity_remap_rows",
            "paged_block_release_enabled", "paged_swap_enabled", "paged_idle_swap_enabled",
            "paged_blocks_swapped_out", "paged_blocks_swapped_in", "paged_swap_out_calls",
            "paged_swap_in_calls", "paged_swap_madvise_calls", "paged_swap_madvise_bytes",
            "paged_prefetch_seq_calls", "kv_mincore_enabled", "paged_swapped_active_visible_violation",
            "paged_swapped_active_visible_violation_rows", "paged_swapped_active_visible_violation_blocks",
            "paged_swapped_active_violation_rows", "paged_swapped_active_violation_blocks",
            "paged_write_to_swapped_block", "paged_row_mapping_invalid_fatal",
            "paged_write_mapping_invalid_fatal", "paged_active_row_nonresident_fatal",
            "paged_input_setup_fatal", "paged_swap_backend_failures", "paged_swap_read_swap_in_failures",
            "paged_swap_write_swap_in_failures", "paged_swap_in_fail_no_offset", "paged_swap_in_fail_bad_size",
            "paged_swap_in_fail_read_cell", "paged_swap_in_fail_tensor_set", "paged_prefetch_seq_failures",
            "paged_block_release_fail", "paged_swap_madvise_failures",
        }
        required_source_fields["io"] = {
            "block_swap_out_calls", "block_swap_in_calls", "bytes_read", "bytes_written",
            "avg_block_swap_out_latency_us", "max_block_swap_out_latency_us",
            "avg_block_swap_in_latency_us", "max_block_swap_in_latency_us", "block_out_validate_us",
            "block_out_pack_us", "block_out_write_us", "block_out_metadata_us", "block_out_madvise_us",
            "block_in_validate_us", "block_in_read_us", "block_in_unpack_us", "block_in_commit_us",
        }
    for source_name, names in required_source_fields.items():
        for name in names:
            if numeric(sources[source_name].get(name)) is None:
                missing_metrics.add(f"{source_name}.{name}")
    missing_metrics = sorted(missing_metrics)

    zero_fields = [
        active_visible, active_visible_rows, active_visible_blocks, write_swapped, fatal, backend_failures,
        value(perf, "prefetch_failures"), value(test, "prefetch_failures_observed"),
    ]
    correctness_failed = (
        (integer(exit_code) is not None and integer(exit_code) != 0) or
        seq0_exact == "NO" or seq1_exact == "NO" or
        any(number is not None and number != 0 for number in (integer(item) for item in zero_fields)) or
        "Segmentation fault" in stderr or "GGML_ASSERT" in stderr
    )
    if correctness_failed:
        row["correctness"] = "FAIL"
    elif missing_metrics or any(item == NA for item in (seq0_exact, seq1_exact, fatal, backend_failures)):
        row["correctness"] = "UNVERIFIED"
    else:
        row["correctness"] = "PASS"

    mechanism, mechanism_notes = verification(case_id, row)
    if warning_lines and case_id in {"E1", "E2", "E3", "E4", "E5"}:
        requested_disabled = any("disabled" in line.lower() or "falling back" in line.lower() for line in warning_lines)
        if requested_disabled:
            mechanism = "FAIL"
            mechanism_notes.append("a requested path reported disabled/fallback")
    row["mechanism_verification"] = mechanism
    if row["correctness"] == "FAIL" or mechanism == "FAIL":
        row["result"] = "FAIL"
    elif row["correctness"] == "PASS" and mechanism == "PASS":
        row["result"] = "PASS"
    else:
        row["result"] = "UNVERIFIED"

    with (run_dir / "extracted_metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    (run_dir / "result").write_text(
        f"result={row['result']}\ncorrectness={row['correctness']}\n"
        f"correctness_notes={'missing required metrics: ' + ', '.join(missing_metrics) if missing_metrics else 'all required metrics present'}\n"
        f"mechanism_verification={mechanism}\nmechanism_notes={'; '.join(mechanism_notes)}\n"
    )
    return row


def compact_number(number: float) -> int | float:
    return int(number) if number.is_integer() else round(number, 6)


def metric_stats(rows: list[dict[str, str]], metric: str, baseline: float | None) -> dict[str, Any]:
    raw: list[Any] = []
    values: list[float] = []
    for row in sorted(rows, key=lambda item: int(item["round"])):
        number = numeric(row.get(metric))
        raw.append(NA if number is None else compact_number(number))
        if number is not None:
            values.append(number)
    result: dict[str, Any] = {"raw": raw, "median": NA, "min": NA, "max": NA, "relative_e0_percent": NA}
    if values:
        median = float(statistics.median(values))
        result.update({"median": compact_number(median), "min": compact_number(min(values)), "max": compact_number(max(values))})
        if baseline is not None and baseline != 0:
            result["relative_e0_percent"] = round((median - baseline) * 100.0 / baseline, 6)
    return result


def write_summaries(root: Path, rows: list[dict[str, str]], manifest: dict[str, Any]) -> None:
    grouped = {case_id: [row for row in rows if row["case"] == case_id] for case_id in CASES}
    baseline_medians: dict[str, float | None] = {}
    for metric in SUMMARY_METRICS:
        vals = [numeric(row.get(metric)) for row in grouped["E0"]]
        valid = [item for item in vals if item is not None]
        baseline_medians[metric] = float(statistics.median(valid)) if valid else None

    summary_columns = [
        "case", "label", "successful_runs", "total_runs", "correctness", "mechanism_verification",
        "raw_values_json", "median_json", "min_json", "max_json", "relative_e0_percent_json",
    ]
    summary_rows: list[dict[str, str]] = []
    stats_by_case: dict[str, dict[str, dict[str, Any]]] = {}
    for case_id in CASES:
        case_rows = grouped[case_id]
        stats = {metric: metric_stats(case_rows, metric, baseline_medians[metric]) for metric in SUMMARY_METRICS}
        stats_by_case[case_id] = stats
        correctness_states = {row["correctness"] for row in case_rows}
        mechanism_states = {row["mechanism_verification"] for row in case_rows}
        correctness = "PASS" if correctness_states == {"PASS"} else ("FAIL" if "FAIL" in correctness_states else "UNVERIFIED")
        mechanism = "PASS" if mechanism_states == {"PASS"} else ("FAIL" if "FAIL" in mechanism_states else "UNVERIFIED")
        summary_rows.append({
            "case": case_id, "label": CASE_CONFIG[case_id],
            "successful_runs": str(sum(row["correctness"] == "PASS" for row in case_rows)),
            "total_runs": str(len(case_rows)), "correctness": correctness, "mechanism_verification": mechanism,
            "raw_values_json": json.dumps({metric: data["raw"] for metric, data in stats.items()}, separators=(",", ":")),
            "median_json": json.dumps({metric: data["median"] for metric, data in stats.items()}, separators=(",", ":")),
            "min_json": json.dumps({metric: data["min"] for metric, data in stats.items()}, separators=(",", ":")),
            "max_json": json.dumps({metric: data["max"] for metric, data in stats.items()}, separators=(",", ":")),
            "relative_e0_percent_json": json.dumps(
                {metric: data["relative_e0_percent"] for metric, data in stats.items()}, separators=(",", ":")
            ),
        })

    with (root / "summary.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    def med(case_id: str, metric: str) -> Any:
        return stats_by_case[case_id][metric]["median"]

    dirty = bool(manifest.get("repo", {}).get("dirty"))
    allow_dirty = bool(manifest.get("execution", {}).get("allow_dirty"))
    lines = ["# KV final controlled E0-E5 summary", ""]
    if dirty:
        lines += ["> <span style=\"color:red\"><strong>WARNING: WORKTREE DIRTY.</strong></span> Results were produced with uncommitted changes"
                  + (" under `ALLOW_DIRTY=1`." if allow_dirty else "."), ""]
    lines += ["## Configurations", "", "| Case | Configuration |", "|---|---|"]
    lines += [f"| {case_id} | {CASE_CONFIG[case_id]} |" for case_id in CASES]
    lines += ["", "E1 and E2-E5 are independent paths: lazy tail is incompatible with the nonidentity paged mapping, so their gains must not be added.", ""]
    lines += ["## Correctness and mechanism verification", "", "| Case | Correct runs | Correctness | Mechanism |", "|---|---:|---|---|"]
    lines += [f"| {row['case']} | {row['successful_runs']}/{row['total_runs']} | {row['correctness']} | {row['mechanism_verification']} |" for row in summary_rows]
    lines += ["", "### Safety violations versus restore work", "",
              "`active_visible_safety_violations` and its row/block detail come only from the three `paged_swapped_active_visible_violation*` safety fields; any nonzero value is a correctness failure. `active_restore_required_rows` and `active_restore_required_blocks` preserve the raw sources `paged_swapped_active_violation_rows/blocks`: they count resume-time active-required data that is still SWAPPED and needs synchronous restoration, not data corruption. `fallback_blocks` is the resume-pending restoration workload reported by the driver.", "",
              f"Observed medians: E3 restore-required blocks={med('E3', 'active_restore_required_blocks')}, fallback blocks={med('E3', 'fallback_blocks')}; E4 restore-required blocks={med('E4', 'active_restore_required_blocks')}, fallback blocks={med('E4', 'fallback_blocks')}; E5 restore-required blocks={med('E5', 'active_restore_required_blocks')}, fallback blocks={med('E5', 'fallback_blocks')}. E3/E4 have no prefetch, so 17 fallback blocks are expected in this workload; E5 restores 17 blocks by prefetch and reaches fallback zero.", ""]
    lines += ["", "## Memory (median; KiB unless noted)", "", "| Case | RSS before resume | RSS after resume | sampled VmRSS max | madvise bytes |", "|---|---:|---:|---:|---:|"]
    lines += [f"| {case_id} | {med(case_id, 'rss_before_resume_kb')} | {med(case_id, 'rss_after_resume_kb')} | {med(case_id, 'process_vmrss_sampled_max_kb')} | {med(case_id, 'madvise_bytes')} |" for case_id in CASES]
    lines += ["", "Process VmRSS, process VmHWM, and cgroup `memory.current` are different accounting scopes. mmap-backed shared file pages can appear in process RSS while their page-cache charge is not necessarily charged to the current cgroup. Process RSS and cgroup current must not be directly compared or subtracted; E0-E5 comparisons are valid only within the same metric. A sampled or peak RSS is not used as a substitute for current-RSS savings. Driver phase RSS points are reported separately.", ""]
    lines += ["## Performance (median)", "", "| Case | Active avg ms | Active p95 ms | Resume first-token ms | Total wall ms | TPS |", "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {case_id} | {med(case_id, 'active_token_avg_ms')} | {med(case_id, 'active_token_p95_ms')} | {med(case_id, 'resume_first_token_ms')} | {med(case_id, 'total_wall_ms')} | {med(case_id, 'tps')} |" for case_id in CASES]
    lines += ["", "## Scope and unavailable telemetry", "",
              "This is a deterministic controlled workload. It does not represent semi-real, ShareGPT, server-like concurrency, or a cgroup `memory.max` capacity matrix, and cannot replace those experiments.", "",
              "With mincore intentionally disabled for the performance experiment, KV resident/nonresident bytes are `NA`; KV residency attribution belongs to a separate diagnostic pass. Per-run cgroup `memory.peak` is `NA` because the framework does not reset a shared cgroup peak. `pending` is `NA` outside stability mode. Backing-file size/allocation can be `NA` when the short-lived anonymous/O_TMPFILE descriptor is not observed by the sampler.", "",
              "The driver performs zero-block prefetch probes even in prefetch-off paged cases. Mechanism checks therefore require zero active restore calls/blocks, not a zero low-level prefetch API probe count. `LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS` is recorded as 0 but is not consumed by this driver.", ""]
    (root / "summary.md").write_text("\n".join(lines))


def dry_run(root: Path) -> None:
    with (root / "runs.tsv").open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(RUN_COLUMNS)
    summary_columns = ["case", "label", "successful_runs", "total_runs", "correctness", "mechanism_verification"]
    with (root / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(summary_columns)
        for case_id in CASES:
            writer.writerow([case_id, CASE_CONFIG[case_id], 0, 0, "UNVERIFIED", "UNVERIFIED"])
    (root / "summary.md").write_text("# KV final controlled E0-E5 dry-run\n\nNo model process was started.\n")
    for run_dir in sorted((root / "runs").glob("*")):
        (run_dir / "extracted_metrics.tsv").write_text("status\nDRY_RUN\n")
        (run_dir / "result").write_text("result=DRY_RUN\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate planned artifacts without parsing model output")
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if not (root / "manifest.json").is_file():
        parser.error(f"manifest.json not found in {root}")
    try:
        manifest = json.loads(read_text(root / "manifest.json"))
        run_dirs = validate_matrix(root, manifest)
    except (ArtifactError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.dry_run:
        dry_run(root)
        return 0

    references: dict[int, tuple[Path, Path]] = {}
    for run_dir in run_dirs:
        meta = json.loads(read_text(run_dir / "run.json"))
        if meta["case"] == "E0" and read_text(run_dir / "exit_code").strip() == "0":
            references[int(meta["round"])] = (run_dir / "seq0", run_dir / "seq1")
    try:
        rows = [parse_run(run_dir, references) for run_dir in run_dirs]
    except (ArtifactError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows.sort(key=lambda row: (int(row["round"]), int(row["run_order"])))
    with (root / "runs.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = refresh_framework_manifest(root, manifest)
    write_summaries(root, rows, manifest)
    return 1 if any(row["result"] != "PASS" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
