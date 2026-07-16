#!/usr/bin/env python3
"""Fail-closed parser for the E0/E2/E5 single-turn diagnostic artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


CASES = ("E0", "E2", "E5")
PLAN = tuple((order, case_id) for order, case_id in enumerate(CASES, 1))
RAW_ARTIFACTS = (
    "run.json", "command", "environment", "stdout", "stderr", "exit_code",
    "perf_status", "perf_report.txt", "perf_report.stderr", "seq0", "seq1",
)
TIMING_FIELDS = (
    "apply_calls", "apply_paged_total_us", "set_row_idx_calls", "set_row_idx_total_us",
    "row_idx_fill_us", "active_visible_us", "nonidentity_probe_us", "swapped_blocks_scan_us",
    "check_read_resident_us", "check_read_resident_calls", "paged_resolve_calls",
    "cells_scanned", "blocks_scanned", "row_idx_entries", "getenv_calls",
)
IO_FIELDS = (
    "block_swap_out_calls", "block_swap_in_calls", "bytes_read", "bytes_written",
    "avg_block_swap_in_latency_us", "max_block_swap_in_latency_us",
    "block_in_validate_us", "block_in_read_us", "block_in_unpack_us", "block_in_commit_us",
)
ACTIVE_FIELDS = (
    "active_token_count", "avg_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms",
    "decode_avg_ms", "prefetch_calls", "prefetch_total_ms",
)
PERF_FIELDS = (
    "prefetch_during_active_blocks", "prefetch_auto_started", "resume_pending_fallback_blocks",
    "prefetch_failures",
)
PREFETCH_FIELDS = ("token", "requested_blocks", "restored_blocks", "decode_ms", "prefetch_ms", "total_ms")
PHASE_CALL_FIELDS = ("call", "seq_id", "requested_blocks", "restored_blocks", "phase_events")
PHASE_BLOCK_FIELDS = (
    "call", "block_index", "physical_block", "validate_us", "read_us", "unpack_us",
    "commit_us", "phase_sum_us",
)
GET_ROWS_PROFILE_FIELDS = (
    "step", "layer", "kv", "src", "n_kv", "row_bytes", "segment_ending_at_get_rows_wall_us",
)
GET_ROWS_PROFILE_SCOPE = "scheduler_graph_segment_from_previous_callback_boundary_through_target_get_rows_completion"
PAGED_SAFETY_FIELDS = (
    "paged_swapped_active_visible_violation", "paged_swapped_active_visible_violation_rows",
    "paged_swapped_active_visible_violation_blocks", "paged_write_to_swapped_block",
    "paged_row_mapping_invalid_fatal", "paged_write_mapping_invalid_fatal",
    "paged_active_row_nonresident_fatal", "paged_input_setup_fatal",
    "paged_swap_backend_failures", "paged_swap_read_swap_in_failures",
    "paged_swap_write_swap_in_failures", "paged_swap_in_fail_no_offset",
    "paged_swap_in_fail_bad_size", "paged_swap_in_fail_read_cell",
    "paged_swap_in_fail_tensor_set", "paged_prefetch_seq_failures",
    "paged_block_release_fail", "paged_swap_madvise_failures",
)
EXPECTED_ENV = {
    "E0": {
        "LLAMA_KV_E2_GET_ROWS_PROFILE": "0",
        "LLAMA_KV_LAZY_CLEAR": "0", "LLAMA_KV_LAZY_TAIL": "0", "LLAMA_KV_PAGED": "0",
        "LLAMA_KV_PAGED_INGRAPH": "0", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
        "LLAMA_KV_PAGED_SWAP": "0", "LLAMA_KV_PAGED_IDLE_SWAP": "0",
        "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0", "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "0",
    },
    "E2": {
        "LLAMA_KV_E2_GET_ROWS_PROFILE": "1",
        "LLAMA_KV_LAZY_CLEAR": "0", "LLAMA_KV_LAZY_TAIL": "0", "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "1",
        "LLAMA_KV_PAGED_SWAP": "0", "LLAMA_KV_PAGED_IDLE_SWAP": "0",
        "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0", "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "0",
    },
    "E5": {
        "LLAMA_KV_E2_GET_ROWS_PROFILE": "0",
        "LLAMA_KV_LAZY_CLEAR": "0", "LLAMA_KV_LAZY_TAIL": "0", "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "1",
        "LLAMA_KV_PAGED_SWAP": "1", "LLAMA_KV_PAGED_IDLE_SWAP": "1",
        "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "1", "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "1",
        "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "1", "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "1",
        "LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME": "1",
    },
}


class ArtifactError(ValueError):
    pass


def read(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}: {exc}") from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def marker_lines(text: str, marker: str) -> list[str]:
    return [line for line in text.splitlines() if marker in line]


def unique_line(text: str, marker: str, required: bool = True) -> str:
    lines = marker_lines(text, marker)
    if len(lines) > 1:
        raise ArtifactError(f"duplicate telemetry marker {marker.strip()!r}")
    if required and not lines:
        raise ArtifactError(f"missing telemetry marker {marker.strip()!r}")
    return lines[0] if lines else ""


def fields(line: str) -> dict[str, str]:
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line)
    result: dict[str, str] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"duplicate telemetry key {key!r}")
        result[key] = value
    return result


def number(source: dict[str, str], name: str) -> float:
    if name not in source:
        raise ArtifactError(f"missing telemetry field {name!r}")
    try:
        value = float(source[name])
    except ValueError as exc:
        raise ArtifactError(f"invalid numeric field {name}={source[name]!r}") from exc
    if not math.isfinite(value) or value < 0:
        raise ArtifactError(f"invalid nonnegative field {name}={source[name]!r}")
    return value


def integer(source: dict[str, str], name: str) -> int:
    value = number(source, name)
    if not value.is_integer():
        raise ArtifactError(f"field {name} must be an integer")
    return int(value)


def compact(value: float) -> int | float:
    return int(value) if value.is_integer() else round(value, 6)


def parse_json(path: Path) -> Any:
    try:
        return json.loads(read(path))
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"invalid JSON in {path}") from exc


def require_identity(manifest: dict[str, Any]) -> None:
    if manifest.get("protocol") != "kv_e0_e2_e5_single_turn_diagnostic" or manifest.get("version") != 2:
        raise ArtifactError("manifest protocol/version mismatch")
    for section, names in {
        "repo": ("path", "branch", "head", "upstream", "dirty", "status_porcelain"),
        "binary": ("path", "sha256", "size"),
        "model": ("path", "sha256", "size"),
        "host": ("hostname", "platform", "python", "compiler_build"),
        "execution": ("dry_run", "use_perf", "perf_available", "perf_reason", "timeout_sec"),
    }.items():
        value = manifest.get(section)
        if not isinstance(value, dict) or any(name not in value for name in names):
            raise ArtifactError(f"manifest missing identity fields in {section}")
    framework = manifest.get("framework")
    if not isinstance(framework, dict):
        raise ArtifactError("manifest missing framework identity")
    for name in ("runner", "parser"):
        value = framework.get(name)
        if not isinstance(value, dict) or any(field not in value for field in ("path", "sha256", "size")):
            raise ArtifactError(f"manifest missing framework.{name} identity")


def validate_artifact_identity(root: Path, record: dict[str, Any]) -> None:
    case_id = record.get("case")
    directory = record.get("directory")
    artifacts = record.get("artifacts")
    if not isinstance(case_id, str) or not isinstance(directory, str) or not isinstance(artifacts, dict):
        raise ArtifactError("manifest run record is malformed")
    run_dir = root / directory
    if run_dir.resolve() != (root / "runs" / case_id).resolve():
        raise ArtifactError(f"{case_id} run directory mismatch")
    expected_names = set(RAW_ARTIFACTS)
    optional = {"perf.data"}
    if not expected_names.issubset(artifacts) or not set(artifacts).issubset(expected_names | optional):
        raise ArtifactError(f"{case_id} raw artifact set is incomplete or contains unknown entries")
    for name, identity in artifacts.items():
        path = run_dir / name
        if not path.is_file() or not isinstance(identity, dict):
            raise ArtifactError(f"{case_id} missing raw artifact {name}")
        if identity.get("size") != path.stat().st_size or identity.get("sha256") != sha256(path):
            raise ArtifactError(f"{case_id} raw artifact identity mismatch: {name}")


def load_manifest(root: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = parse_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest.json must contain an object")
    require_identity(manifest)
    planned = manifest.get("planned_runs")
    expected = [{"order": order, "case": case_id} for order, case_id in PLAN]
    if planned != expected:
        raise ArtifactError("manifest planned_runs must be exactly ordered E0,E2,E5")
    records = manifest.get("runs")
    if not isinstance(records, list) or [(item.get("order"), item.get("case")) for item in records if isinstance(item, dict)] != list(PLAN) or len(records) != len(PLAN):
        raise ArtifactError("manifest runs must be exactly ordered E0,E2,E5")
    actual_dirs = sorted(path.name for path in (root / "runs").glob("*") if path.is_dir())
    if actual_dirs != sorted(CASES):
        raise ArtifactError("artifact run directories must be exactly E0,E2,E5")
    run_dirs: list[Path] = []
    for (order, case_id), record in zip(PLAN, records):
        if not isinstance(record, dict) or record.get("order") != order or record.get("case") != case_id:
            raise ArtifactError("manifest run order/case mismatch")
        validate_artifact_identity(root, record)
        run_dir = root / "runs" / case_id
        meta = parse_json(run_dir / "run.json")
        if meta != {"order": order, "case": case_id}:
            raise ArtifactError(f"{case_id} run.json mismatch")
        run_dirs.append(run_dir)
    return manifest, run_dirs


def requested_env(run_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in read(run_dir / "environment").splitlines():
        if "=" not in line:
            raise ArtifactError(f"invalid environment line in {run_dir}: {line!r}")
        key, value = line.split("=", 1)
        if key in result:
            raise ArtifactError(f"duplicate environment key {key!r} in {run_dir}")
        result[key] = value
    return result


def validate_case_environment(run_dir: Path, case_id: str) -> dict[str, str]:
    env = requested_env(run_dir)
    expected = EXPECTED_ENV[case_id]
    mismatches = [f"{key}={env.get(key)!r} expected {value!r}" for key, value in expected.items() if env.get(key) != value]
    if mismatches:
        raise ArtifactError(f"{case_id} environment mismatch: {', '.join(mismatches)}")
    return env


def parse_perf(run_dir: Path, case_id: str) -> dict[str, Any]:
    status = fields(read(run_dir / "perf_status").strip())
    state = status.get("state")
    reason = status.get("reason", "unknown")
    if case_id == "E5":
        if state != "not_applicable" or reason != "e5_uses_swapin_phase_telemetry":
            raise ArtifactError("E5 perf status must be not_applicable")
        return {"status": "not_applicable", "reason": reason, "get_rows_kernel_sample_percent": "NA", "matched_symbols": []}
    if state == "unresolved":
        return {"status": "unresolved", "reason": reason, "get_rows_kernel_sample_percent": "NA", "matched_symbols": []}
    if state != "captured":
        raise ArtifactError(f"{case_id} invalid perf state {state!r}")
    report = read(run_dir / "perf_report.txt")
    matches: list[tuple[str, float]] = []
    for line in report.splitlines():
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)%.*\b(ggml_compute_forward_get_rows(?:_[A-Za-z0-9_]+)?)\b", line)
        if match and "_back" not in match.group(2):
            matches.append((match.group(2), float(match.group(1))))
    if not matches:
        return {"status": "unresolved", "reason": "get_rows_symbols_or_samples_missing", "get_rows_kernel_sample_percent": "NA", "matched_symbols": []}
    return {
        "status": "resolved", "reason": "sampled_symbol_overhead",
        "get_rows_kernel_sample_percent": round(sum(value for _, value in matches), 6),
        "matched_symbols": [name for name, _ in matches],
    }


def parse_timing(stderr: str) -> dict[str, int | float]:
    source = fields(unique_line(stderr, "KV_PAGED_TIMING_SUMMARY "))
    return {name: compact(number(source, name)) for name in TIMING_FIELDS}


def parse_get_rows_profile(stderr: str, case_id: str) -> dict[str, Any]:
    event_lines = marker_lines(stderr, "KV_E2_GET_ROWS_PROFILE ")
    summary_lines = marker_lines(stderr, "KV_E2_GET_ROWS_PROFILE_SUMMARY ")
    if case_id != "E2":
        if event_lines or summary_lines:
            raise ArtifactError(f"{case_id} unexpectedly emitted E2 GET_ROWS profiler telemetry")
        return {"status": "not_requested"}

    if len(summary_lines) != 1:
        raise ArtifactError(f"E2 GET_ROWS profiler summary count is {len(summary_lines)}, expected 1")
    summary = fields(summary_lines[0])
    steps = integer(summary, "steps")
    model_layers = integer(summary, "model_layers")
    kv_layers = integer(summary, "kv_layers")
    event_count = integer(summary, "events")
    capacity = integer(summary, "capacity")
    if summary.get("scope") != GET_ROWS_PROFILE_SCOPE:
        raise ArtifactError("E2 GET_ROWS profiler summary has an invalid or missing scope")
    if steps <= 0 or model_layers <= 0 or kv_layers <= 0 or kv_layers > model_layers:
        raise ArtifactError("E2 GET_ROWS profiler summary has invalid step/layer counts")
    expected_event_count = steps * kv_layers * 2
    if event_count != len(event_lines) or event_count != expected_event_count or capacity < event_count:
        raise ArtifactError(
            f"E2 GET_ROWS profiler event count mismatch: summary={event_count} "
            f"lines={len(event_lines)} expected={expected_event_count} capacity={capacity}"
        )

    events: list[dict[str, int | str]] = []
    nodes_by_step: dict[int, set[tuple[int, str]]] = {step: set() for step in range(1, steps + 1)}
    for line in event_lines:
        source = fields(line)
        if any(name not in source for name in GET_ROWS_PROFILE_FIELDS):
            raise ArtifactError("E2 GET_ROWS profiler event is missing a required field")
        step = integer(source, "step")
        layer = integer(source, "layer")
        n_kv = integer(source, "n_kv")
        row_bytes = integer(source, "row_bytes")
        wall_us = integer(source, "segment_ending_at_get_rows_wall_us")
        kv = source["kv"]
        src = source["src"]
        if step < 1 or step > steps or layer < 0 or layer >= model_layers:
            raise ArtifactError("E2 GET_ROWS profiler event has an out-of-range step or layer")
        if kv not in {"K", "V"} or src != f"cache_{kv.lower()}_l{layer}":
            raise ArtifactError("E2 GET_ROWS profiler event has an invalid KV/source pair")
        if n_kv <= 0 or row_bytes <= 0:
            raise ArtifactError("E2 GET_ROWS profiler event has a nonpositive shape")
        node = (layer, kv)
        if node in nodes_by_step[step]:
            raise ArtifactError(f"E2 GET_ROWS profiler duplicate node at step {step}: {node}")
        nodes_by_step[step].add(node)
        events.append({
            "step": step, "layer": layer, "kv": kv, "src": src, "n_kv": n_kv,
            "row_bytes": row_bytes, "segment_ending_at_get_rows_wall_us": wall_us,
        })

    expected_nodes = nodes_by_step[1]
    if len(expected_nodes) != kv_layers * 2:
        raise ArtifactError("E2 GET_ROWS profiler first step has incomplete KV nodes")
    for layer, _ in expected_nodes:
        if (layer, "K") not in expected_nodes or (layer, "V") not in expected_nodes:
            raise ArtifactError(f"E2 GET_ROWS profiler layer {layer} has incomplete K/V nodes")
    for step, nodes in nodes_by_step.items():
        if nodes != expected_nodes:
            raise ArtifactError(f"E2 GET_ROWS profiler step {step} has incomplete or inconsistent KV nodes")

    wall_values = [int(event["segment_ending_at_get_rows_wall_us"]) for event in events]
    wall_sum = sum(wall_values)
    return {
        "status": "resolved", "steps": steps, "model_layers": model_layers,
        "kv_layers": kv_layers, "events": events,
        "segment_ending_at_get_rows_wall_us": {
            "count": len(wall_values), "sum": wall_sum,
            "avg": compact(wall_sum / len(wall_values)),
            "min": min(wall_values), "max": max(wall_values),
        },
        "segment_ending_at_get_rows_wall_scope": GET_ROWS_PROFILE_SCOPE,
    }


def paged_sources(stderr: str) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict[str, str]]:
    metadata = fields(unique_line(stderr, "KV paged metadata stats:"))
    io = fields(unique_line(stderr, "KV_PAGED_IO_STATS "))
    active = fields(unique_line(stderr, "KV_ACTIVE_TOKEN_STATS "))
    perf = fields(unique_line(stderr, "KV_IDLE_SWAP_RESUME_PERF "))
    for name in IO_FIELDS:
        number(io, name)
    for name in ACTIVE_FIELDS:
        number(active, name)
    for name in PERF_FIELDS:
        number(perf, name)
    for name in PAGED_SAFETY_FIELDS:
        if integer(metadata, name) != 0:
            raise ArtifactError(f"paged safety/backend field is nonzero: {name}={metadata[name]}")
    return metadata, io, active, perf


def warning_failure(stderr: str) -> bool:
    return any(
        (" disabled" in line.lower() or "falling back" in line.lower()) and "sampling" not in line.lower()
        for line in stderr.splitlines()
    )


def parse_e5(stderr: str, env: dict[str, str]) -> dict[str, Any]:
    metadata, io, active, perf = paged_sources(stderr)
    required_positive = (
        "ingraph_gather_layers", "paged_nonidentity_remap_rows", "paged_swap_out_calls",
        "paged_swap_madvise_calls", "paged_swap_madvise_bytes",
    )
    for name in required_positive:
        if integer(metadata, name) <= 0:
            raise ArtifactError(f"E5 required mechanism field did not trigger: {name}")
    if integer(metadata, "paged_swap_enabled") != 1 or integer(metadata, "paged_idle_swap_enabled") != 1:
        raise ArtifactError("E5 paged swap/idle swap not enabled")
    if integer(perf, "prefetch_auto_started") != 1 or integer(perf, "resume_pending_fallback_blocks") != 0:
        raise ArtifactError("E5 auto prefetch did not start cleanly or fallback remained")
    if integer(perf, "prefetch_failures") != 0:
        raise ArtifactError("E5 observed prefetch failures")
    if env.get("LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME") != "1":
        raise ArtifactError("E5 defer was not requested")

    prefetch_lines = marker_lines(stderr, "KV_ACTIVE_TOKEN_PREFETCH ")
    prefetch_calls = integer(active, "prefetch_calls")
    if prefetch_calls <= 0 or len(prefetch_lines) != prefetch_calls:
        raise ArtifactError(f"E5 prefetch line count {len(prefetch_lines)} != prefetch_calls {prefetch_calls}")
    events: list[dict[str, int | float]] = []
    tokens: set[int] = set()
    for line in prefetch_lines:
        source = fields(line)
        event = {name: compact(number(source, name)) for name in PREFETCH_FIELDS}
        token = int(event["token"])
        if token in tokens:
            raise ArtifactError(f"duplicate prefetch token {token}")
        if int(event["restored_blocks"]) <= 0:
            raise ArtifactError(f"prefetch token {token} restored no blocks")
        tokens.add(token)
        events.append(event)
    if abs(sum(float(event["prefetch_ms"]) for event in events) - number(active, "prefetch_total_ms")) > max(0.002 * max(1, prefetch_calls), 0.002):
        raise ArtifactError("per-token prefetch_ms does not reconcile with prefetch_total_ms")

    call_lines = marker_lines(stderr, "KV_PAGED_PREFETCH_PHASE_CALL ")
    block_lines = marker_lines(stderr, "KV_PAGED_PREFETCH_BLOCK_PHASE ")
    if len(call_lines) != prefetch_calls:
        raise ArtifactError(f"E5 phase call count {len(call_lines)} != prefetch_calls {prefetch_calls}")
    calls: list[dict[str, int]] = []
    for line in call_lines:
        source = fields(line)
        calls.append({name: integer(source, name) for name in PHASE_CALL_FIELDS})
    call_ids = [item["call"] for item in calls]
    if len(set(call_ids)) != len(call_ids) or call_ids != sorted(call_ids):
        raise ArtifactError("E5 phase call IDs must be unique and increasing")

    blocks_by_call: dict[int, list[dict[str, int]]] = {call_id: [] for call_id in call_ids}
    physical_blocks: set[int] = set()
    for line in block_lines:
        source = fields(line)
        block = {name: integer(source, name) for name in PHASE_BLOCK_FIELDS}
        call_id = block["call"]
        if call_id not in blocks_by_call:
            raise ArtifactError(f"E5 block phase references unknown call {call_id}")
        phase_sum = block["validate_us"] + block["read_us"] + block["unpack_us"] + block["commit_us"]
        if block["phase_sum_us"] != phase_sum:
            raise ArtifactError("E5 per-block phase sum mismatch")
        if block["physical_block"] in physical_blocks:
            raise ArtifactError(f"E5 duplicate restored physical block {block['physical_block']}")
        physical_blocks.add(block["physical_block"])
        blocks_by_call[call_id].append(block)

    associated: list[dict[str, Any]] = []
    for event, call in zip(events, calls):
        call_blocks = blocks_by_call[call["call"]]
        if call["seq_id"] != 0 or call["requested_blocks"] != int(event["requested_blocks"]) or call["restored_blocks"] != int(event["restored_blocks"]):
            raise ArtifactError(f"E5 token {event['token']} does not reconcile with phase call {call['call']}")
        if call["phase_events"] != call["restored_blocks"] or len(call_blocks) != call["restored_blocks"]:
            raise ArtifactError(f"E5 phase event count mismatch for call {call['call']}")
        if [block["block_index"] for block in call_blocks] != list(range(len(call_blocks))):
            raise ArtifactError(f"E5 block indexes are not contiguous for call {call['call']}")
        associated.append({"token": event["token"], "call": call["call"], "blocks": call_blocks})

    restored_total = sum(int(event["restored_blocks"]) for event in events)
    if restored_total != integer(perf, "prefetch_during_active_blocks") or restored_total != integer(io, "block_swap_in_calls"):
        raise ArtifactError("E5 restored blocks do not reconcile with active-prefetch/global swap-in totals")
    phase_totals = {
        "validate_us": sum(block["validate_us"] for blocks in blocks_by_call.values() for block in blocks),
        "read_us": sum(block["read_us"] for blocks in blocks_by_call.values() for block in blocks),
        "unpack_us": sum(block["unpack_us"] for blocks in blocks_by_call.values() for block in blocks),
        "commit_us": sum(block["commit_us"] for blocks in blocks_by_call.values() for block in blocks),
    }
    for event_name, io_name in {
        "validate_us": "block_in_validate_us", "read_us": "block_in_read_us",
        "unpack_us": "block_in_unpack_us", "commit_us": "block_in_commit_us",
    }.items():
        if phase_totals[event_name] != integer(io, io_name):
            raise ArtifactError(f"E5 per-block {event_name} does not reconcile with {io_name}")
    phase_sum = sum(phase_totals.values())
    if phase_sum <= 0:
        raise ArtifactError("E5 did not observe nonzero swap-in phase timing")
    phase_percent = {name.replace("_us", "_percent"): round(value * 100.0 / phase_sum, 6) for name, value in phase_totals.items()}

    p95 = number(active, "p95_ms")
    p99 = number(active, "p99_ms")
    overlap_p95 = [int(event["token"]) for event in events if float(event["total_ms"]) >= p95]
    overlap_p99 = [int(event["token"]) for event in events if float(event["total_ms"]) >= p99]
    return {
        "swap_in": {name: compact(number(io, name)) for name in IO_FIELDS},
        "swap_in_phase_sum_us": phase_sum,
        "swap_in_phase_percent": phase_percent,
        "prefetch_calls": prefetch_calls,
        "prefetch_events": events,
        "token_block_phase_association": associated,
        "prefetch_tokens_at_or_above_p95": overlap_p95,
        "prefetch_tokens_at_or_above_p99": overlap_p99,
        "prefetch_overlap_p95_count": len(overlap_p95),
        "prefetch_overlap_p99_count": len(overlap_p99),
    }


def sequence_exact(run_dir: Path, reference: Path) -> bool:
    candidate = (run_dir / reference.name).read_bytes()
    baseline = reference.read_bytes()
    if not candidate or not baseline:
        raise ArtifactError(f"empty sequence evidence in {run_dir}")
    return candidate == baseline


def parse_run(root: Path, run_dir: Path, reference_dir: Path) -> dict[str, Any]:
    meta = parse_json(run_dir / "run.json")
    case_id = meta["case"]
    exit_code = read(run_dir / "exit_code").strip()
    if exit_code != "0":
        raise ArtifactError(f"{case_id} exit_code is {exit_code!r}, expected 0")
    env = validate_case_environment(run_dir, case_id)
    stderr = read(run_dir / "stderr")
    if "Segmentation fault" in stderr or "GGML_ASSERT" in stderr:
        raise ArtifactError(f"{case_id} stderr contains a fatal signature")
    if case_id in {"E2", "E5"} and warning_failure(stderr):
        raise ArtifactError(f"{case_id} requested path reported disabled or falling back")

    active = fields(unique_line(stderr, "KV_ACTIVE_TOKEN_STATS "))
    perf_line = fields(unique_line(stderr, "KV_IDLE_SWAP_RESUME_PERF "))
    swap = fields(unique_line(stderr, "KV swap stats:"))
    for name in ACTIVE_FIELDS:
        number(active, name)
    for name in PERF_FIELDS:
        number(perf_line, name)
    if integer(swap, "backend_failures") != 0 or integer(perf_line, "prefetch_failures") != 0:
        raise ArtifactError(f"{case_id} observed backend or prefetch failure")

    seq0_exact = sequence_exact(run_dir, reference_dir / "seq0")
    seq1_exact = sequence_exact(run_dir, reference_dir / "seq1")
    if not seq0_exact or not seq1_exact:
        raise ArtifactError(f"{case_id} output differs from E0")

    result: dict[str, Any] = {
        "case": case_id, "exit_code": 0, "correctness": "PASS", "mechanism_verification": "PASS",
        "seq0_exact": seq0_exact, "seq1_exact": seq1_exact,
    }
    result["get_rows_profile"] = parse_get_rows_profile(stderr, case_id)
    if case_id in {"E0", "E2"}:
        timing = parse_timing(stderr)
        result["paged_timing"] = timing
        if case_id == "E0":
            if marker_lines(stderr, "KV paged metadata stats:") or integer(active, "prefetch_calls") != 0:
                raise ArtifactError("E0 unexpectedly entered paged/prefetch path")
        else:
            metadata, io, _, _ = paged_sources(stderr)
            if integer(metadata, "ingraph_gather_layers") <= 0 or integer(metadata, "paged_nonidentity_enabled") != 1:
                raise ArtifactError("E2 did not enable in-graph/nonidentity paged gather")
            if timing["set_row_idx_calls"] == 0 or timing["row_idx_entries"] == 0:
                raise ArtifactError("E2 did not observe row-index calls and entries")
            if any(integer(io, name) != 0 for name in ("block_swap_out_calls", "block_swap_in_calls", "bytes_read", "bytes_written")):
                raise ArtifactError("E2 unexpectedly performed paged swap I/O")
            if any(integer(metadata, name) != 0 for name in ("paged_swap_enabled", "paged_idle_swap_enabled", "paged_swap_madvise_calls")):
                raise ArtifactError("E2 unexpectedly enabled swap/madvise")
            if integer(active, "prefetch_calls") != 0:
                raise ArtifactError("E2 unexpectedly performed active prefetch")
        result["perf_get_rows"] = parse_perf(run_dir, case_id)
        result["attribution"] = "NOT_REQUIRED" if case_id == "E0" else "GET_ROWS_PROFILE"
        result["result"] = "PASS"
    else:
        result["paged_io_prefetch"] = parse_e5(stderr, env)
        result["perf_get_rows"] = parse_perf(run_dir, case_id)
        result["attribution"] = "NOT_APPLICABLE"
        result["result"] = "PASS"
    return result


def write_summary(root: Path, results: list[dict[str, Any]]) -> None:
    by_case = {item["case"]: item for item in results}
    e0 = by_case["E0"]["paged_timing"]
    e2 = by_case["E2"]["paged_timing"]
    delta_fields = (
        "apply_paged_total_us", "set_row_idx_total_us", "row_idx_fill_us", "active_visible_us",
        "nonidentity_probe_us", "swapped_blocks_scan_us", "check_read_resident_us",
    )
    deltas = {name: compact(float(e2[name]) - float(e0[name])) for name in delta_fields}
    e2_profile = by_case["E2"]["get_rows_profile"]
    e2_wall = e2_profile["segment_ending_at_get_rows_wall_us"]
    overall = "PASS" if all(item["result"] == "PASS" for item in results) else "UNRESOLVED"
    summary = {
        "scope": "informal_single_turn_diagnostic_not_a_performance_conclusion",
        "result": overall,
        "runs": results,
        "e2_minus_e0_timing_us": deltas,
        "e2_get_rows_profile": {
            "steps": e2_profile["steps"], "kv_layers": e2_profile["kv_layers"],
            "event_count": len(e2_profile["events"]),
            "segment_ending_at_get_rows_wall_us": e2_wall,
            "segment_ending_at_get_rows_wall_scope": e2_profile["segment_ending_at_get_rows_wall_scope"],
        },
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with (root / "summary.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("section", "metric", "value"))
        writer.writerow(("overall", "result", overall))
        for case_id in CASES:
            writer.writerow((case_id, "result", by_case[case_id]["result"]))
            writer.writerow((case_id, "attribution", by_case[case_id]["attribution"]))
        for case_id in ("E0", "E2"):
            for name, value in by_case[case_id]["paged_timing"].items():
                writer.writerow((case_id, name, value))
            perf = by_case[case_id]["perf_get_rows"]
            writer.writerow((case_id, "perf_status", perf["status"]))
            writer.writerow((case_id, "get_rows_kernel_sample_percent", perf["get_rows_kernel_sample_percent"]))
        for name, value in deltas.items():
            writer.writerow(("E2-E0", name, value))
        writer.writerow((
            "E2-GET_ROWS",
            "segment_ending_at_get_rows_wall_scope",
            e2_profile["segment_ending_at_get_rows_wall_scope"],
        ))
        for name, value in e2_wall.items():
            writer.writerow(("E2-GET_ROWS", f"segment_ending_at_get_rows_wall_us_{name}", value))
        e5 = by_case["E5"]["paged_io_prefetch"]
        for name, value in e5["swap_in_phase_percent"].items():
            writer.writerow(("E5", name, value))
        writer.writerow(("E5", "associated_tokens", len(e5["token_block_phase_association"])))

    lines = [
        "# E0/E2/E5 single-turn bottleneck diagnostic", "",
        "> Informal diagnostic only. This artifact is not a controlled performance conclusion.", "",
        f"Overall result: **{overall}**", "",
        "## Case validation", "", "| case | correctness | mechanism | attribution | result |", "|---|---|---|---|---|",
    ]
    for case_id in CASES:
        item = by_case[case_id]
        lines.append(f"| {case_id} | {item['correctness']} | {item['mechanism_verification']} | {item['attribution']} | {item['result']} |")
    lines += ["", "## E0/E2 paged CPU timing", "", "| case | row-index fill us | active-visible us | resident-check us/calls | cells/blocks/rows scanned | GET_ROWS perf |", "|---|---:|---:|---:|---:|---|"]
    for case_id in ("E0", "E2"):
        timing = by_case[case_id]["paged_timing"]
        perf = by_case[case_id]["perf_get_rows"]
        perf_text = f"{perf['get_rows_kernel_sample_percent']}%" if perf["status"] == "resolved" else f"unresolved ({perf['reason']})"
        lines.append(
            f"| {case_id} | {timing['row_idx_fill_us']} | {timing['active_visible_us']} | "
            f"{timing['check_read_resident_us']}/{timing['check_read_resident_calls']} | "
            f"{timing['cells_scanned']}/{timing['blocks_scanned']}/{timing['row_idx_entries']} | {perf_text} |"
        )
    lines += [
        "", "## E2 scheduler segment ending at GET_ROWS wall-time summary", "",
        f"count={e2_wall['count']}, sum={e2_wall['sum']} us, avg={e2_wall['avg']} us, "
        f"min={e2_wall['min']} us, max={e2_wall['max']} us.", "",
        "Each value is outer wall time for the scheduler graph segment from the previous callback boundary "
        "through completion of the target GET_ROWS.",
        "It must not be described as single-node time or GET_ROWS kernel time; it is also not step or end-to-end wall time.",
    ]
    e5 = by_case["E5"]["paged_io_prefetch"]
    phases = e5["swap_in_phase_percent"]
    lines += [
        "", "## E5 token/block swap-in phase association", "",
        f"Associated {len(e5['token_block_phase_association'])} active-token prefetch calls with "
        f"{sum(len(item['blocks']) for item in e5['token_block_phase_association'])} restored physical blocks.", "",
        f"validate={phases['validate_percent']}%, read={phases['read_percent']}%, "
        f"unpack={phases['unpack_percent']}%, commit={phases['commit_percent']}% of associated block phases.", "",
        "Raw token, call, physical-block and phase records are preserved in summary.json.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n")


def validate_dry_run(root: Path, run_dirs: list[Path]) -> None:
    for run_dir, case_id in zip(run_dirs, CASES):
        validate_case_environment(run_dir, case_id)
        if read(run_dir / "exit_code").strip() != "DRY_RUN":
            raise ArtifactError(f"{case_id} dry-run exit_code is not DRY_RUN")
        parse_perf(run_dir, case_id)
    (root / "summary.json").write_text(json.dumps({
        "scope": "dry_run_only_model_not_started", "planned_cases": list(CASES), "result": "PASS",
    }, indent=2) + "\n")
    (root / "summary.md").write_text("# E0/E2/E5 single-turn diagnostic dry-run\n\nPASS: planned E0, E2, E5; model was not started.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        manifest, run_dirs = load_manifest(root)
        if bool(manifest["execution"]["dry_run"]) != args.dry_run:
            raise ArtifactError("parser dry-run mode does not match manifest")
        if args.dry_run:
            validate_dry_run(root, run_dirs)
            return 0
        reference_dir = root / "runs" / "E0"
        results = [parse_run(root, run_dir, reference_dir) for run_dir in run_dirs]
        write_summary(root, results)
    except (ArtifactError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1 if any(item["result"] != "PASS" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
