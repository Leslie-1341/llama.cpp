#!/usr/bin/env python3
"""Fail-closed verdict authority for canonical Formal OFFLOAD artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = "kv_offload_benchmark"
SCHEMA_VERSION = 2
SUPPORTED_POLICIES = {"resident", "release_only", "v2"}
BUDGET_POLICIES = {"release_only", "v2"}
SUPPORTED_KV_REPRESENTATIONS = {"paged"}
SUPPORTED_LOADING_MODES = {"exact"}
SUPPORTED_RESTORES = {"k1_sync", "k2_pipeline"}
SUPPORTED_PREFAULTS = {"off", "r2"}
SUPPORTED_RUN_KINDS = {"qualification", "formal"}
SUPPORTED_RUN_MODES = {"qualification", "characterization"}
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
UINT = re.compile(r"^[0-9]+$")
SIGNED_INT = re.compile(r"^-?[0-9]+$")
CANONICAL_SERVER_OPTIONS = {"-m", "--model", "--host", "--port"}
PRESSURE_BASIS_AUTHORITIES = {"cgroup_finite", "rss_absolute"}
SAMPLE_HEADER = [
    "elapsed_ms", "timestamp_mono_ns", "timestamp_realtime_ns", "pid", "starttime_ticks",
    "vmrss_kb", "vmhwm_kb", "vmswap_kb", "cgroup_memory_current_bytes",
    "cgroup_memory_peak_bytes", "cgroup_memory_swap_current_bytes", "cgroup_memory_events",
    "backing_logical_size", "backing_allocated_bytes",
]

MANIFEST_REQUIRED = {
    "schema_version", "protocol", "artifact_id", "runner_status", "created_at_utc", "framework",
    "provenance", "spec", "planned_runs", "run_results", "unsupported",
}
MANIFEST_OPTIONAL = {"dry_run", "finished_at_utc"}
SPEC_KEYS = {
    "schema_version", "protocol", "phase", "run_kind", "run_mode", "binary", "model", "model_quantization",
    "server_args", "environment", "pressure_basis", "workload", "cases", "run_order", "sampler", "cgroup",
    "max_blocks", "health_timeout_seconds", "request_timeout_seconds",
}
CASE_KEYS = {
    "case_id", "policy", "kv_representation", "loading_mode", "restore", "prefault",
    "kv_target_bytes", "action_target_bytes",
}
PLAN_KEYS = {
    "run_id", "round", "run_order", "case_id", "policy", "kv_representation", "loading_mode",
    "restore", "prefault", "kv_target_bytes", "action_target_bytes",
}
RUN_KEYS = {"run_id", "round", "run_order", "case_id", "execution_index", "case", "request_plan"}
EXECUTION_KEYS = {
    "run_id", "round", "run_order", "case_id", "execution_index", "run_mode", "argv", "environment", "server_identity",
    "server_cgroup", "pressure_basis", "sampler_identity", "sampler_schema", "sampler_argv", "request_loop_started",
    "request_count", "qualification", "characterization",
}
CLEANUP_KEYS = {"server", "sampler", "residual_process", "cleanup_complete"}
PROCESS_CLEANUP_KEYS = {
    "pid", "pgid", "exit_code", "stop_requested", "stop_signal", "term_timed_out", "kill_timed_out",
    "pgid_check_complete", "residual_process",
}
RESPONSE_KEYS = {
    "sequence", "request_id", "repeat_index", "measurement", "measurement_phase", "n_predict", "stream",
    "started_mono_ns", "started_realtime_ns", "first_byte_mono_ns", "finished_mono_ns",
    "http_status", "headers", "body_path", "body_bytes", "body_sha256", "response_json", "error",
}
ACTION_REQUIRED = {
    "state", "source", "sample_valid", "stale", "pressure_basis_valid", "decision_id", "episode", "target_bytes", "max_blocks",
    "observed_excess_bytes", "debt_before_bytes", "debt_after_bytes", "budget_active",
    "budget_target_enabled", "budget_source", "budget_target_bytes", "budget_basis_generation",
    "budget_view_valid", "budget_resident_available", "budget_reclaimable_available",
    "budget_resident_bytes", "budget_dead_resident_reclaimable_bytes",
    "budget_transient_staging_bound_bytes", "budget_observed_excess_bytes", "budget_debt_before_bytes",
    "budget_debt_after_bytes", "soft_offload_armed_before", "soft_offload_armed_after",
    "budget_next_action_sample", "unmet_budget_bytes_after", "offload_armed_before",
    "offload_armed_after", "next_action_sample", "evaluate_attempted", "evaluate_outcome",
    "evaluate_reason", "release_attempted", "offload_attempted", "selected_seq_id",
    "selected_claimant_epoch", "transaction_id", "outcome", "reason", "blocks", "bytes",
    "relieved_bytes", "shortfall_bytes", "io_failure", "io_errno", "state_changed",
    "decision_reason", "sample_count", "idle", "claimants", "scores",
}
ACTION_NUMERIC = {
    key for key in ACTION_REQUIRED if key not in {
        "state", "source", "budget_source", "evaluate_outcome", "evaluate_reason", "outcome",
        "reason", "decision_reason", "claimants", "scores", "selected_seq_id",
    }
}
ACTION_BOOL = {
    "sample_valid", "stale", "pressure_basis_valid", "budget_active", "budget_target_enabled", "budget_view_valid", "budget_resident_available",
    "budget_reclaimable_available", "soft_offload_armed_before", "soft_offload_armed_after",
    "offload_armed_before", "offload_armed_after", "evaluate_attempted", "release_attempted",
    "offload_attempted", "io_failure", "state_changed", "idle",
}
RESUME_BOOL = {"graph_allowed"}
IO_BOOL = {"k2_enabled", "restore_prefault_enabled"}
RESUME_REQUIRED = {
    "phase", "decision_id", "seq_id", "claimant_epoch", "transaction_id", "action",
    "outcome", "reason", "graph_allowed",
}
TIMING_REQUIRED = {
    "decision_id", "seq_id", "transaction_id", "restored_blocks", "restored_bytes", "queue_us",
    "gate_us", "graph_us", "total_us",
}
RESTORE_ACTIVITY_FIELDS = {
    "block_in_validate_us", "avg_block_in_validate_us", "block_in_read_us", "avg_block_in_read_us",
    "block_in_unpack_us", "avg_block_in_unpack_us", "block_in_commit_us", "avg_block_in_commit_us",
    "restore_prefault_groups", "restore_prefault_calls", "restore_prefault_us",
    "restore_prefault_minor_faults", "restore_prefault_major_faults", "restore_scatter_groups",
    "restore_scatter_us", "restore_scatter_fault_groups", "restore_scatter_minor_faults",
    "restore_scatter_major_faults", "k2_peak_staging_groups", "k2_peak_staging_bytes",
    "k2_pipeline_wall_us", "k2_exposed_read_wait_us", "k2_pipeline_stall_us",
    "k2_read_completed_ahead",
}
IO_REQUIRED = {
    "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls", "backing_write_syscalls",
    "bytes_read", "bytes_written", "avg_block_swap_out_latency_us", "max_block_swap_out_latency_us",
    "avg_block_swap_in_latency_us", "max_block_swap_in_latency_us", "staging_buffer_bytes",
    "k2_enabled", "k2_group_byte_cap", "k2_staging_bound_bytes", "k2_peak_staging_groups",
    "k2_peak_staging_bytes", "k2_pipeline_wall_us", "k2_exposed_read_wait_us",
    "k2_pipeline_stall_us", "k2_read_completed_ahead", "block_out_validate_us",
    "avg_block_out_validate_us", "block_out_pack_us", "avg_block_out_pack_us",
    "block_out_write_us", "avg_block_out_write_us", "block_out_metadata_us",
    "avg_block_out_metadata_us", "block_out_madvise_us", "avg_block_out_madvise_us",
    "block_in_validate_us", "avg_block_in_validate_us", "block_in_read_us",
    "avg_block_in_read_us", "block_in_unpack_us", "avg_block_in_unpack_us",
    "block_in_commit_us", "avg_block_in_commit_us", "restore_prefault_enabled",
    "restore_prefault_groups", "restore_prefault_calls", "restore_prefault_us",
    "restore_prefault_minor_faults", "restore_prefault_major_faults", "restore_scatter_groups",
    "restore_scatter_us", "restore_scatter_fault_groups", "restore_scatter_minor_faults",
    "restore_scatter_major_faults",
}
RESIDENT_OBSERVATION_REQUIRED = {
    "source", "action", "decision_id", "seq_id", "transaction_id", "server_pid",
    "before_available", "before_object_id", "before_generation", "before_page_size",
    "before_total_bytes", "before_resident_bytes", "before_total_pages", "before_resident_pages",
    "after_available", "after_object_id", "after_generation", "after_page_size",
    "after_total_bytes", "after_resident_bytes", "after_total_pages", "after_resident_pages",
}
RESIDENT_OBSERVATION_NUMERIC = RESIDENT_OBSERVATION_REQUIRED - {"source", "action"}
RESIDENT_OBSERVATION_BOOL = {"before_available", "after_available"}
SLOT_SNAPSHOT_KEYS = {"captured_mono_ns", "http_status", "body_path", "body_bytes", "body_sha256", "body_json", "error"}
TELEMETRY_GAPS = [
    {"field": "event timestamp/event_seq", "status": "unavailable", "reason": "current production markers have no monotonic event envelope"},
    {"field": "object_id/generation per action", "status": "unavailable", "reason": "unified action marker does not carry physical object identity"},
    {"field": "per-action read/write bytes", "status": "unavailable", "reason": "KV_PAGED_IO_STATS is cumulative teardown telemetry"},
    {"field": "per-action restore/page-population/stall", "status": "partial", "reason": "qualification correlates PREFETCH timing by decision/transaction/seq, but teardown IO remains cumulative rather than per-action"},
    {"field": "multi-session migration correlation", "status": "unavailable", "reason": "current telemetry has no stable migration correlation key across sessions; qualification proves only a single-slot round-trip"},
    {"field": "swapped_authoritative_bytes per observation", "status": "unavailable", "reason": "server marker does not expose the Physical Budget View swapped field"},
]


class ParseError(Exception):
    pass


def require_finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParseError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ParseError(f"{label} must be a finite positive number")
    return result


def reject_canonical_server_args(args: list[str], label: str) -> None:
    for item in args:
        if item in CANONICAL_SERVER_OPTIONS or item.startswith(("--model=", "--host=", "--port=")):
            raise ParseError(f"{label} must not override -m/--model/--host/--port")
        if item.startswith("-m") and not item.startswith("--"):
            raise ParseError(f"{label} must not override -m/--model/--host/--port")


def reject_canonical_kv_environment(environment: dict[str, str], label: str) -> None:
    conflicts = sorted(key for key in environment if key.startswith("LLAMA_KV_"))
    if conflicts:
        raise ParseError(f"{label} conflicts with canonical KV env: {conflicts}")


def exact(value: Any, expected: set[str], label: str, optional: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(f"{label} must be an object")
    optional = optional or set()
    missing = expected - set(value)
    extra = set(value) - expected - optional
    if missing or extra:
        raise ParseError(f"{label} schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
    return value


def read_json(path: pathlib.Path) -> Any:
    if not path.is_file():
        raise ParseError(f"missing {path.relative_to(path.parents[1]) if len(path.parents) > 1 else path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ParseError(f"invalid JSON {path}: {exc}") from exc


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_identity(value: Any, label: str, allow_missing: bool = False) -> None:
    identity = exact(value, {"path", "present", "size", "sha256"}, label)
    if not isinstance(identity["path"], str) or not isinstance(identity["present"], bool):
        raise ParseError(f"{label} has invalid path/present")
    if not identity["present"]:
        if not allow_missing:
            raise ParseError(f"{label} is missing")
        if identity["size"] is not None or identity["sha256"] is not None:
            raise ParseError(f"{label} missing identity must use null size/sha256")
        return
    path = pathlib.Path(identity["path"])
    if not path.is_file():
        raise ParseError(f"{label} file is absent: {path}")
    if identity["size"] != path.stat().st_size or identity["sha256"] != sha256_file(path):
        raise ParseError(f"{label} identity drift: {path}")
    if not isinstance(identity["sha256"], str) or len(identity["sha256"]) != 64:
        raise ParseError(f"{label} sha256 is malformed")


def reject_runner_verdict(value: Any, path: str = "artifact") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"verdict", "performance_verdict", "parser_status"}:
                raise ParseError(f"runner artifact contains parser-owned field {path}.{key}")
            reject_runner_verdict(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_runner_verdict(child, f"{path}[{index}]")


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def derived_run_id(round_id: int, order: int, case_id: str) -> str:
    return f"r{round_id:03d}_o{order:03d}_{safe_component(case_id)}"


def normalize_pressure_basis(value: Any) -> dict[str, Any]:
    basis = exact(value, {"authority", "low_water_kb", "pressure_kb", "critical_kb"}, "spec.pressure_basis")
    authority = basis["authority"]
    if authority not in PRESSURE_BASIS_AUTHORITIES:
        raise ParseError(f"spec.pressure_basis.authority is unsupported: {authority!r}")
    thresholds = [basis[key] for key in ("low_water_kb", "pressure_kb", "critical_kb")]
    if authority == "cgroup_finite":
        if any(item is not None for item in thresholds):
            raise ParseError("cgroup_finite pressure basis cannot define RSS thresholds")
    else:
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in thresholds):
            raise ParseError("rss_absolute pressure basis thresholds must be positive integers")
        if not (thresholds[0] < thresholds[1] < thresholds[2]):
            raise ParseError("RSS pressure basis thresholds must satisfy low < pressure < critical")
    return {
        "authority": authority,
        "low_water_kb": thresholds[0],
        "pressure_kb": thresholds[1],
        "critical_kb": thresholds[2],
    }


def pressure_basis_source(basis: dict[str, Any]) -> str:
    return "CGROUP_RATIO" if basis["authority"] == "cgroup_finite" else "RSS_ABSOLUTE"


def normalize_workload(workload: Any) -> dict[str, Any]:
    value = exact(
        workload,
        {"warmup", "requests", "repeat", "qualification", "characterization"},
        "spec.workload",
    )
    if not isinstance(value["warmup"], list) or not isinstance(value["requests"], list):
        raise ParseError("spec.workload warmup/requests must be arrays")
    all_requests = value["warmup"] + value["requests"]
    ids: list[str] = []
    for index, item in enumerate(all_requests):
        request = exact(item, {"request_id", "prompt", "n_predict", "stream"}, f"workload.request[{index}]")
        if not isinstance(request["request_id"], str) or not CASE_ID_RE.fullmatch(request["request_id"]):
            raise ParseError(f"workload.request[{index}] has invalid request_id")
        if not isinstance(request["prompt"], str) or isinstance(request["n_predict"], bool) or not isinstance(request["n_predict"], int) or request["n_predict"] < 0:
            raise ParseError(f"workload.request[{index}] has invalid prompt/n_predict")
        if not isinstance(request["stream"], bool):
            raise ParseError(f"workload.request[{index}].stream must be boolean")
        ids.append(request["request_id"])
    if len(ids) != len(set(ids)):
        raise ParseError("workload request_id values are duplicated")
    if isinstance(value["repeat"], bool) or not isinstance(value["repeat"], int) or not 1 <= value["repeat"] <= 1000:
        raise ParseError("workload.repeat is invalid")
    if not value["requests"]:
        raise ParseError("workload.requests must contain at least one measurement request")

    normalized = dict(value)
    qualification = value["qualification"]
    if qualification is not None:
        qualification = exact(
            qualification,
            {"idle_seconds", "offload_timeout_seconds", "resume_request_id"},
            "workload.qualification",
        )
        idle_seconds = require_finite_positive(
            qualification["idle_seconds"], "workload.qualification.idle_seconds")
        offload_timeout_seconds = require_finite_positive(
            qualification["offload_timeout_seconds"],
            "workload.qualification.offload_timeout_seconds",
        )
        resume_request_id = qualification["resume_request_id"]
        if not isinstance(resume_request_id, str) or resume_request_id not in {
                item["request_id"] for item in value["requests"]}:
            raise ParseError(
                "workload.qualification.resume_request_id must reference a measurement request")
        if not value["warmup"]:
            raise ParseError("qualification requires at least one warmup request")
        normalized["qualification"] = {
            "idle_seconds": idle_seconds,
            "offload_timeout_seconds": offload_timeout_seconds,
            "resume_request_id": resume_request_id,
        }

    characterization = value["characterization"]
    if characterization is not None:
        characterization = exact(
            characterization,
            {
                "idle_seconds", "settle_timeout_seconds", "target_tolerance_bytes",
                "resume_request_id",
            },
            "workload.characterization",
        )
        idle_seconds = require_finite_positive(
            characterization["idle_seconds"], "workload.characterization.idle_seconds")
        settle_timeout_seconds = require_finite_positive(
            characterization["settle_timeout_seconds"],
            "workload.characterization.settle_timeout_seconds",
        )
        tolerance = characterization["target_tolerance_bytes"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
            raise ParseError(
                "workload.characterization.target_tolerance_bytes is invalid")
        resume_request_id = characterization["resume_request_id"]
        if not isinstance(resume_request_id, str) or resume_request_id not in {
                item["request_id"] for item in value["requests"]}:
            raise ParseError(
                "workload.characterization.resume_request_id must reference a measurement request")
        if resume_request_id != value["requests"][0]["request_id"]:
            raise ParseError(
                "workload.characterization.resume_request_id must be the first measurement request")
        if not value["warmup"]:
            raise ParseError("characterization requires at least one warmup/fill request")
        normalized["characterization"] = {
            "idle_seconds": idle_seconds,
            "settle_timeout_seconds": settle_timeout_seconds,
            "target_tolerance_bytes": tolerance,
            "resume_request_id": resume_request_id,
        }
    return normalized


def expanded_request_plan(
        workload: dict[str, Any], case: dict[str, Any], run_mode: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    sequence = 0

    def append(
            item: dict[str, Any], request_id: str, repeat_index: int, measurement: bool,
            measurement_phase: str) -> None:
        nonlocal sequence
        result.append({
            "sequence": sequence,
            "request_id": request_id,
            "repeat_index": repeat_index,
            "measurement": measurement,
            "measurement_phase": measurement_phase,
            "n_predict": item["n_predict"],
            "prompt_sha256": sha256_bytes(item["prompt"].encode("utf-8")),
            "stream": item["stream"],
        })
        sequence += 1

    for item in workload["warmup"]:
        append(item, item["request_id"], 0, False, "fill")
    for repeat_index in range(1, workload["repeat"] + 1):
        for request_index, item in enumerate(workload["requests"]):
            if run_mode == "characterization" and case["policy"] == "v2":
                measurement_phase = (
                    "resume" if repeat_index == 1 and request_index == 0
                    else "post_resume_steady")
            elif run_mode == "characterization" and case["policy"] == "release_only":
                measurement_phase = "release_only_steady"
            elif run_mode == "characterization":
                measurement_phase = "resident_steady"
            else:
                measurement_phase = "qualification_measurement"
            append(item, item["request_id"], repeat_index, True, measurement_phase)
    if run_mode == "qualification" and workload["qualification"] is not None and case["policy"] == "v2":
        resume_request_id = workload["qualification"]["resume_request_id"]
        resume_request = next(
            item for item in workload["requests"] if item["request_id"] == resume_request_id)
        append(resume_request, resume_request_id, 0, False, "qualification_resume")
    return result


def validate_spec(spec: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    value = exact(spec, SPEC_KEYS, "spec")
    if value["schema_version"] != SCHEMA_VERSION or value["protocol"] != PROTOCOL:
        raise ParseError("spec protocol/schema mismatch")
    if value["phase"] not in {"resident_baseline", "coarse_target", "local_target", "representative"}:
        raise ParseError("spec.phase is invalid")
    if value["run_kind"] not in SUPPORTED_RUN_KINDS:
        raise ParseError("spec.run_kind is invalid")
    if value["run_mode"] not in SUPPORTED_RUN_MODES:
        raise ParseError("spec.run_mode is invalid")
    if not isinstance(value["server_args"], list) or any(not isinstance(item, str) for item in value["server_args"]):
        raise ParseError("spec.server_args is invalid")
    reject_canonical_server_args(value["server_args"], "spec.server_args")
    if not isinstance(value["environment"], dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value["environment"].items()
    ):
        raise ParseError("spec.environment is invalid")
    reject_canonical_kv_environment(value["environment"], "spec.environment")
    pressure_basis = normalize_pressure_basis(value["pressure_basis"])
    workload = normalize_workload(value["workload"])
    if value["run_mode"] == "qualification":
        if workload["qualification"] is None or workload["characterization"] is not None:
            raise ParseError(
                "qualification mode requires only workload.qualification configuration")
    elif workload["characterization"] is None or workload["qualification"] is not None:
        raise ParseError(
            "characterization mode requires only workload.characterization configuration")
    cases: dict[str, dict[str, Any]] = {}
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise ParseError("spec.cases must be non-empty")
    for index, raw_case in enumerate(value["cases"]):
        case = exact(raw_case, CASE_KEYS, f"spec.cases[{index}]")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise ParseError(f"spec.cases[{index}].case_id is invalid")
        if case_id in cases:
            raise ParseError(f"duplicate case_id: {case_id}")
        if any(not isinstance(case[factor], str) for factor in ("policy", "kv_representation", "loading_mode", "restore", "prefault")):
            raise ParseError(f"{case_id} factors must be strings")
        target = case["kv_target_bytes"]
        action_target = case["action_target_bytes"]
        if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target <= 0):
            raise ParseError(f"{case_id}.kv_target_bytes is invalid")
        if action_target is not None and (
                isinstance(action_target, bool) or not isinstance(action_target, int) or action_target <= 0):
            raise ParseError(f"{case_id}.action_target_bytes is invalid")
        if case["policy"] == "resident":
            if target is not None or action_target is not None:
                raise ParseError(f"{case_id}: resident policy cannot have resident/action targets")
        elif case["policy"] in BUDGET_POLICIES and (target is None or action_target is None):
            raise ParseError(
                f"{case_id}: {case['policy']} policy requires explicit kv_target_bytes and action_target_bytes")
        cases[case_id] = case
    action_targets = {
        case["action_target_bytes"] for case in cases.values()
        if case["policy"] in BUDGET_POLICIES
    }
    if len(action_targets) > 1:
        raise ParseError("all budget cases must use the same explicit action_target_bytes")
    sampler = exact(value["sampler"], {"interval_seconds"}, "spec.sampler")
    sampler_interval = require_finite_positive(sampler["interval_seconds"], "spec.sampler.interval_seconds")
    cgroup = exact(value["cgroup"], {"expected_memory_max"}, "spec.cgroup")
    if cgroup["expected_memory_max"] is not None and not isinstance(cgroup["expected_memory_max"], str):
        raise ParseError("spec.cgroup.expected_memory_max must be null or a string")
    max_blocks = value["max_blocks"]
    if isinstance(max_blocks, bool) or not isinstance(max_blocks, int) or max_blocks <= 0:
        raise ParseError("spec.max_blocks must be positive")
    health_timeout = require_finite_positive(value["health_timeout_seconds"], "spec.health_timeout_seconds")
    request_timeout = require_finite_positive(value["request_timeout_seconds"], "spec.request_timeout_seconds")
    plan: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for index, raw_run in enumerate(value["run_order"]):
        item = exact(raw_run, {"round", "run_order", "case_id"}, f"spec.run_order[{index}]")
        if isinstance(item["round"], bool) or not isinstance(item["round"], int) or item["round"] <= 0 or isinstance(item["run_order"], bool) or not isinstance(item["run_order"], int) or item["run_order"] <= 0:
            raise ParseError(f"spec.run_order[{index}] has invalid round/order")
        case_id = item["case_id"]
        if case_id not in cases:
            raise ParseError(f"spec.run_order[{index}] references unknown case {case_id}")
        key = (item["round"], item["run_order"])
        if key in seen:
            raise ParseError(f"duplicate planned run key: {key}")
        seen.add(key)
        case = cases[case_id]
        plan.append({
            "run_id": derived_run_id(item["round"], item["run_order"], case_id),
            "round": item["round"],
            "run_order": item["run_order"],
            "case_id": case_id,
            "policy": case["policy"],
            "kv_representation": case["kv_representation"],
            "loading_mode": case["loading_mode"],
            "restore": case["restore"],
            "prefault": case["prefault"],
            "kv_target_bytes": case["kv_target_bytes"],
            "action_target_bytes": case["action_target_bytes"],
        })
    if any(item["policy"] == "release_only" for item in plan) and value["run_mode"] != "characterization":
        raise ParseError("release_only policy is available only in characterization mode")
    if value["run_kind"] == "formal":
        if len({item["round"] for item in plan}) < 2:
            raise ParseError("formal run requires at least two independent rounds")
        if value["run_mode"] == "qualification" and workload["repeat"] < 2:
            raise ParseError("formal qualification run requires repeat >= 2")
    return cases, plan, workload


def parse_fields(line: str, token: str, required: set[str], label: str) -> dict[str, str]:
    words = line.split()
    if words.count(token) != 1:
        raise ParseError(f"{label}: duplicate or malformed {token} token")
    fields: dict[str, str] = {}
    for word in words[words.index(token) + 1:]:
        if word.count("=") != 1:
            raise ParseError(f"{label}: malformed field {word!r}")
        key, value = word.split("=", 1)
        if not key or not value or key in fields:
            raise ParseError(f"{label}: duplicate or empty field {key!r}")
        fields[key] = value
    missing = required - set(fields)
    if missing:
        raise ParseError(f"{label}: missing fields {sorted(missing)}")
    return fields


def numeric_fields(fields: dict[str, str], names: set[str], label: str) -> None:
    for key in names:
        if key in fields and not UINT.fullmatch(fields[key]):
            raise ParseError(f"{label}: {key} is not a non-negative integer")


def validate_action_fields(fields: dict[str, str], label: str) -> None:
    numeric_fields(fields, ACTION_NUMERIC, label)
    selected_seq_id = fields["selected_seq_id"]
    if not SIGNED_INT.fullmatch(selected_seq_id) or int(selected_seq_id) < -1:
        raise ParseError(f"{label}: selected_seq_id must be -1 or a non-negative integer")
    if fields["offload_attempted"] == "1" and fields["state_changed"] == "1":
        if int(selected_seq_id) < 0:
            raise ParseError(f"{label}: state-changing OFFLOAD requires a non-negative selected_seq_id")
        if int(fields["selected_claimant_epoch"]) <= 0:
            raise ParseError(f"{label}: state-changing OFFLOAD requires a positive selected_claimant_epoch")
        if int(fields["transaction_id"]) <= 0:
            raise ParseError(f"{label}: state-changing OFFLOAD requires a positive transaction_id")


def boolean_fields(fields: dict[str, str], names: set[str], label: str) -> None:
    for key in names:
        if key in fields and fields[key] not in {"0", "1"}:
            raise ParseError(f"{label}: {key} is not a boolean marker")


def marker_records(text: str, token: str, required: set[str], label: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for line in text.splitlines():
        if token not in line.split():
            continue
        fields = parse_fields(line, token, required, label)
        result.append(fields)
    return result


def is_qualifying_offload_action(
        action: dict[str, str], expected_source: str, allow_shortfall: bool = False) -> bool:
    return (
        action["state"] == "NORMAL"
        and action["source"] == expected_source
        and action["sample_valid"] == "1"
        and action["stale"] == "0"
        and action["pressure_basis_valid"] == "1"
        and action["budget_active"] == "1"
        and int(action["budget_observed_excess_bytes"]) > 0
        and action["offload_attempted"] == "1"
        and action["outcome"] == "completed"
        and action["state_changed"] == "1"
        and int(action["blocks"]) > 0
        and int(action["bytes"]) > 0
        and int(action["relieved_bytes"]) > 0
        and (allow_shortfall or int(action["shortfall_bytes"]) == 0)
        and action["io_failure"] == "0"
    )


def is_transaction_local_resident_drop(observation: dict[str, str]) -> bool:
    return (
        observation["source"] == "paged_sample_mincore"
        and observation["action"] == "offload"
        and observation["before_available"] == "1"
        and observation["after_available"] == "1"
        and observation["before_object_id"] == observation["after_object_id"]
        and observation["before_generation"] == observation["after_generation"]
        and int(observation["after_resident_bytes"]) < int(observation["before_resident_bytes"])
    )


def qualified_offload_pairs(
        actions: list[dict[str, str]],
        observations: list[dict[str, str]],
        expected_source: str,
        allow_shortfall: bool = False,
) -> list[dict[str, Any]]:
    observations_by_key: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for observation in observations:
        if not is_transaction_local_resident_drop(observation):
            continue
        key = (
            observation["decision_id"], observation["transaction_id"], observation["seq_id"])
        observations_by_key.setdefault(key, []).append(observation)
    result: list[dict[str, Any]] = []
    for action in actions:
        if not is_qualifying_offload_action(action, expected_source, allow_shortfall):
            continue
        key = (action["decision_id"], action["transaction_id"], action["selected_seq_id"])
        for observation in observations_by_key.get(key, []):
            result.append({
                "decision_id": int(action["decision_id"]),
                "transaction_id": int(action["transaction_id"]),
                "seq_id": int(action["selected_seq_id"]),
                "selected_claimant_epoch": int(action["selected_claimant_epoch"]),
                "before_object_id": int(observation["before_object_id"]),
                "before_generation": int(observation["before_generation"]),
                "before_resident_bytes": int(observation["before_resident_bytes"]),
                "after_object_id": int(observation["after_object_id"]),
                "after_generation": int(observation["after_generation"]),
                "after_resident_bytes": int(observation["after_resident_bytes"]),
                "resident_drop_bytes": (
                    int(observation["before_resident_bytes"])
                    - int(observation["after_resident_bytes"])),
                "action": action,
                "resident_observation": observation,
            })
    return result


def stderr_window(data: bytes, start_offset: int, end_offset: int, label: str) -> str:
    if (
        isinstance(start_offset, bool)
        or not isinstance(start_offset, int)
        or isinstance(end_offset, bool)
        or not isinstance(end_offset, int)
        or start_offset < 0
        or end_offset < start_offset
        or end_offset > len(data)
    ):
        raise ParseError(f"{label}: stderr window is outside the captured file")
    segment = data[start_offset:end_offset]
    if start_offset > 0 and data[start_offset - 1:start_offset] != b"\n":
        newline = segment.find(b"\n")
        if newline < 0:
            return ""
        segment = segment[newline + 1:]
    if segment and not segment.endswith(b"\n"):
        newline = segment.rfind(b"\n")
        segment = b"" if newline < 0 else segment[:newline + 1]
    return segment.decode("utf-8", errors="replace")


def validate_sampler(path: pathlib.Path, identity: dict[str, Any], label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ParseError(f"{label}: memory_samples.tsv missing")
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0].split("\t") != SAMPLE_HEADER:
        raise ParseError(f"{label}: memory sampler header mismatch")
    if len(rows) == 1:
        raise ParseError(f"{label}: memory sampler has no samples")
    samples: list[dict[str, Any]] = []
    previous_mono: int | None = None
    previous_elapsed: int | None = None
    for line_number, line in enumerate(rows[1:], start=2):
        columns = line.split("\t")
        if len(columns) != len(SAMPLE_HEADER):
            raise ParseError(f"{label}: sample line {line_number} column count mismatch")
        numeric_indices = {0, 1, 2, 3, 4, 5, 6, 7}
        for index in numeric_indices:
            if not UINT.fullmatch(columns[index]):
                raise ParseError(f"{label}: sample line {line_number} field {SAMPLE_HEADER[index]} is invalid")
        pid = int(columns[3])
        starttime = int(columns[4])
        if pid != identity["pid"] or starttime != identity["starttime_ticks"]:
            raise ParseError(f"{label}: sample line {line_number} process identity mismatch")
        mono = int(columns[1])
        elapsed = int(columns[0])
        if previous_mono is not None and mono <= previous_mono:
            raise ParseError(f"{label}: monotonic timestamp is not increasing")
        if previous_elapsed is not None and elapsed < previous_elapsed:
            raise ParseError(f"{label}: elapsed_ms moved backwards")
        previous_mono, previous_elapsed = mono, elapsed
        events = columns[11]
        if events != "NA":
            for entry in events.split(";"):
                if not re.fullmatch(r"[A-Za-z0-9_.-]+=[0-9]+", entry):
                    raise ParseError(f"{label}: malformed memory.events value")
        for index in (5, 6, 7, 8, 9, 10, 12, 13):
            if columns[index] != "NA" and not UINT.fullmatch(columns[index]):
                raise ParseError(f"{label}: optional field {SAMPLE_HEADER[index]} is invalid")
        samples.append({key: columns[index] for index, key in enumerate(SAMPLE_HEADER)})
    return samples


def validate_resident_observation(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParseError(f"{label}: resident observation is not an object")
    required = {"status", "source", "object_id", "generation", "page_size", "total_bytes", "resident_bytes", "total_pages", "resident_pages"}
    if set(value) != required:
        raise ParseError(f"{label}: resident observation schema mismatch")
    if value["status"] != "available" or not isinstance(value["source"], str) or not value["source"]:
        raise ParseError(f"{label}: resident observation is unavailable")
    for key in ("object_id", "generation", "page_size", "total_bytes", "resident_bytes", "total_pages", "resident_pages"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0:
            raise ParseError(f"{label}: {key} is invalid")
    if value["page_size"] <= 0 or value["total_bytes"] <= 0 or value["total_pages"] <= 0:
        raise ParseError(f"{label}: resident totals are invalid")
    if value["resident_bytes"] > value["total_bytes"] or value["resident_pages"] > value["total_pages"]:
        raise ParseError(f"{label}: resident observation exceeds total")
    return value


def validate_slot_snapshot(
        path: pathlib.Path, label: str, require_resident: bool = True) -> dict[str, Any]:
    snapshot = exact(read_json(path), SLOT_SNAPSHOT_KEYS, label)
    if not isinstance(snapshot["captured_mono_ns"], int) or snapshot["captured_mono_ns"] < 0:
        raise ParseError(f"{label}: captured_mono_ns is invalid")
    if snapshot["http_status"] != 200:
        raise ParseError(f"{label}: /slots must return HTTP 200")
    if not isinstance(snapshot["body_path"], str) or pathlib.Path(snapshot["body_path"]).is_absolute() or ".." in pathlib.Path(snapshot["body_path"]).parts:
        raise ParseError(f"{label}: body_path escapes artifact")
    body = path.parent / snapshot["body_path"]
    if not body.is_file():
        raise ParseError(f"{label}: raw body missing")
    raw = body.read_bytes()
    if not isinstance(snapshot["body_bytes"], int) or snapshot["body_bytes"] != len(raw):
        raise ParseError(f"{label}: raw body size mismatch")
    if not isinstance(snapshot["body_sha256"], str) or snapshot["body_sha256"] != sha256_bytes(raw):
        raise ParseError(f"{label}: raw body identity mismatch")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParseError(f"{label}: raw /slots body is not JSON") from exc
    if decoded != snapshot["body_json"] or not isinstance(decoded, list):
        raise ParseError(f"{label}: /slots body_json does not match raw body")
    residents = []
    for index, slot in enumerate(decoded):
        if not isinstance(slot, dict):
            raise ParseError(f"{label}: slot {index} is not an object")
        if "kv_resident" in slot:
            residents.append(validate_resident_observation(slot["kv_resident"], f"{label}.slot[{index}].kv_resident"))
    if require_resident and not residents:
        raise ParseError(f"{label}: no valid physical resident observation")
    return snapshot


def validate_execution_environment(execution: dict[str, Any], case: dict[str, Any], label: str, spec: dict[str, Any]) -> None:
    env = execution["environment"]
    if not isinstance(env, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()):
        raise ParseError(f"{label}: environment is invalid")
    if case["kv_representation"] != "paged" or case["loading_mode"] != "exact":
        raise ParseError(f"{label}: unsupported representation/loading factor reached workload")
    if env.get("LLAMA_KV_PAGED") != "1" or env.get("LLAMA_KV_PAGED_INGRAPH") != "1":
        raise ParseError(f"{label}: paged runtime is not explicitly enabled")
    for key, expected in {
        "LLAMA_KV_PAGED_SWAP": "1" if case["policy"] == "v2" else "0",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
        "LLAMA_KV_PAGED_MINCORE": "1",
        "LLAMA_KV_PAGED_IO_STATS": "1",
        "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "0",
    }.items():
        if env.get(key) != expected:
            raise ParseError(f"{label}: canonical KV environment mismatch for {key}")
    if not isinstance(env.get("LLAMA_KV_SWAP_DIR"), str) or not env["LLAMA_KV_SWAP_DIR"]:
        raise ParseError(f"{label}: canonical backing directory is missing")
    if env.get("LLAMA_KV_REPRESENTATION") not in (None, "paged"):
        raise ParseError(f"{label}: runtime representation disagrees with factor")
    if env.get("LLAMA_KV_LOADING_MODE") not in (None, "exact"):
        raise ParseError(f"{label}: runtime loading mode disagrees with factor")
    if env.get("LLAMA_KV_PAGED_RESTORE_K2") != ("1" if case["restore"] == "k2_pipeline" else "0"):
        raise ParseError(f"{label}: restore mechanism activation mismatch")
    if env.get("LLAMA_KV_PAGED_RESTORE_PREFAULT_PROBE") != ("1" if case["prefault"] == "r2" else "0"):
        raise ParseError(f"{label}: prefault mechanism activation mismatch")
    if env.get("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS") != "100" or env.get("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS") != "1000":
        raise ParseError(f"{label}: canonical pressure interval environment mismatch")
    if env.get("LLAMA_KV_PRESSURE_SAMPLER") != "1" or env.get("LLAMA_KV_RESUME_STAGE_TIMING") != "1":
        raise ParseError(f"{label}: canonical telemetry environment is incomplete")
    expected_resident_observation = "1" if case["policy"] == "v2" else "preflight"
    if env.get("LLAMA_KV_G0_S1_RESIDENT_OBSERVATION") != expected_resident_observation:
        raise ParseError(f"{label}: physical resident observation mode mismatch")
    basis = spec["pressure_basis"]
    if basis["authority"] == "rss_absolute":
        for key, expected in {
            "LLAMA_KV_LOW_WATER_RSS_KB": str(basis["low_water_kb"]),
            "LLAMA_KV_PRESSURE_RSS_KB": str(basis["pressure_kb"]),
            "LLAMA_KV_CRITICAL_RSS_KB": str(basis["critical_kb"]),
        }.items():
            if env.get(key) != expected:
                raise ParseError(f"{label}: RSS pressure basis environment mismatch for {key}")
    elif any(key in env for key in (
        "LLAMA_KV_LOW_WATER_RSS_KB", "LLAMA_KV_PRESSURE_RSS_KB", "LLAMA_KV_CRITICAL_RSS_KB",
    )):
        raise ParseError(f"{label}: cgroup pressure basis leaked RSS thresholds")
    if case["policy"] == "resident":
        if env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION") != "0" or env.get("LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE") != "0":
            raise ParseError(f"{label}: resident policy did not disable unified action")
        if any(key in env for key in (
            "LLAMA_KV_RESIDENT_TARGET_BYTES", "LLAMA_KV_RESIDENT_TARGET_SOURCE",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS",
        )):
            raise ParseError(f"{label}: resident policy leaked a budget target")
    elif case["policy"] in BUDGET_POLICIES:
        resident_target = str(case["kv_target_bytes"])
        action_target = str(case["action_target_bytes"])
        if env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION") != "1" or env.get("LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE") != "1":
            raise ParseError(f"{label}: {case['policy']} policy did not enable unified action")
        if env.get("LLAMA_KV_RESIDENT_TARGET_BYTES") != resident_target or env.get("LLAMA_KV_RESIDENT_TARGET_SOURCE") != "env_static":
            raise ParseError(f"{label}: {case['policy']} resident target environment mismatch")
        if env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES") != action_target:
            raise ParseError(f"{label}: {case['policy']} action target environment mismatch")
        if env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS") != str(spec["max_blocks"]):
            raise ParseError(f"{label}: {case['policy']} action max-blocks environment mismatch")


def validate_action_target_markers(
        actions: list[dict[str, str]],
        case: dict[str, Any],
        spec: dict[str, Any],
        label: str,
) -> None:
    if case["policy"] not in BUDGET_POLICIES:
        return
    expected_action_target = int(case["action_target_bytes"])
    expected_resident_target = int(case["kv_target_bytes"])
    for action in actions:
        if int(action["target_bytes"]) != expected_action_target:
            raise ParseError(f"{label}: action marker target differs from action_target_bytes")
        if int(action["max_blocks"]) != int(spec["max_blocks"]):
            raise ParseError(f"{label}: action marker max_blocks differs from manifest")
        if action["budget_target_enabled"] == "1" and int(action["budget_target_bytes"]) != expected_resident_target:
            raise ParseError(f"{label}: action marker budget target differs from kv_target_bytes")


def validate_responses(
        path: pathlib.Path,
        raw_dir: pathlib.Path,
        request_plan: list[dict[str, Any]],
        label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not path.is_file():
        raise ParseError(f"{label}: responses.jsonl missing")
    records: list[dict[str, Any]] = []
    service_failures: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseError(f"{label}: invalid response JSON line {line_number}: {exc}") from exc
        record = exact(record, RESPONSE_KEYS, f"{label}.response[{line_number}]")
        if not isinstance(record["request_id"], str) or not CASE_ID_RE.fullmatch(record["request_id"]):
            raise ParseError(f"{label}: response request_id is invalid")
        if not isinstance(record["measurement"], bool) or not isinstance(record["stream"], bool):
            raise ParseError(f"{label}: response boolean field is invalid")
        if not isinstance(record["measurement_phase"], str) or not record["measurement_phase"]:
            raise ParseError(f"{label}: response measurement phase is invalid")
        for key in ("sequence", "repeat_index", "n_predict", "started_mono_ns", "started_realtime_ns", "finished_mono_ns", "http_status", "body_bytes"):
            if isinstance(record[key], bool) or not isinstance(record[key], int) or record[key] < 0:
                raise ParseError(f"{label}: response {key} is invalid")
        if record["first_byte_mono_ns"] is not None and (
            isinstance(record["first_byte_mono_ns"], bool) or not isinstance(record["first_byte_mono_ns"], int)
            or record["first_byte_mono_ns"] < record["started_mono_ns"]
            or record["first_byte_mono_ns"] > record["finished_mono_ns"]
        ):
            raise ParseError(f"{label}: first byte timestamp is invalid")
        if not record["stream"] and record["first_byte_mono_ns"] is not None:
            raise ParseError(f"{label}: non-streaming response cannot provide TTFT timestamp")
        if record["finished_mono_ns"] < record["started_mono_ns"]:
            raise ParseError(f"{label}: response time moved backwards")
        if record["error"] is not None and not isinstance(record["error"], str):
            raise ParseError(f"{label}: response error is invalid")
        if not isinstance(record["body_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["body_sha256"]):
            raise ParseError(f"{label}: response body hash is invalid")
        body_path = pathlib.Path(record["body_path"])
        if body_path.is_absolute() or ".." in body_path.parts:
            raise ParseError(f"{label}: response body path escapes artifact")
        body = raw_dir.parent / body_path
        if not body.is_file():
            raise ParseError(f"{label}: response body is missing")
        if body.stat().st_size != record["body_bytes"] or sha256_file(body) != record["body_sha256"]:
            raise ParseError(f"{label}: response body identity mismatch")
        records.append(record)
        if record["http_status"] < 200 or record["http_status"] >= 300:
            service_failures.append({
                "sequence": record["sequence"], "request_id": record["request_id"],
                "http_status": record["http_status"], "error": record["error"],
            })
    if len(records) != len(request_plan):
        raise ParseError(f"{label}: response count {len(records)} != planned {len(request_plan)}")
    if not any(record["measurement"] for record in records):
        raise ParseError(f"{label}: no measurement response was recorded")
    for actual, expected in zip(records, request_plan):
        for key in (
            "sequence", "request_id", "repeat_index", "measurement", "measurement_phase",
            "n_predict", "stream",
        ):
            if actual[key] != expected[key]:
                raise ParseError(f"{label}: response/order mismatch at sequence {expected['sequence']} field {key}")
    return records, service_failures


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize_metric(values: list[float], unit: str) -> dict[str, Any]:
    if not values:
        return {"status": "UNAVAILABLE", "unit": unit, "n": 0, "p50": None, "p95": None, "p99": None}
    return {
        "status": "AVAILABLE", "unit": unit, "n": len(values),
        "p50": percentile(values, 0.50), "p95": percentile(values, 0.95), "p99": percentile(values, 0.99),
    }


def response_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    measurement = [record for record in records if record["measurement"]]
    ttft = [
        (record["first_byte_mono_ns"] - record["started_mono_ns"]) / 1_000_000
        for record in measurement if record["first_byte_mono_ns"] is not None
    ]
    e2e = [
        (record["finished_mono_ns"] - record["started_mono_ns"]) / 1_000_000
        for record in measurement
    ]
    tpot: list[float] = []
    throughput: list[float] = []
    for record in measurement:
        value = record.get("response_json")
        timings = value.get("timings") if isinstance(value, dict) else None
        if not isinstance(timings, dict):
            continue
        predicted_n = timings.get("predicted_n")
        predicted_ms = timings.get("predicted_ms")
        predicted_per_second = timings.get("predicted_per_second")
        if isinstance(predicted_n, (int, float)) and predicted_n > 0 and isinstance(predicted_ms, (int, float)) and predicted_ms >= 0:
            tpot.append(float(predicted_ms) / float(predicted_n))
        if isinstance(predicted_per_second, (int, float)) and predicted_per_second >= 0:
            throughput.append(float(predicted_per_second))
    return {
        "request_count": len(measurement),
        "ttft_ms": summarize_metric(ttft, "ms"),
        "tpot_ms_per_token": summarize_metric(tpot, "ms/token"),
        "e2e_ms": summarize_metric(e2e, "ms"),
        "throughput_tokens_per_second": summarize_metric(throughput, "tokens/s"),
    }


def memory_statistics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def numeric(field: str) -> list[int]:
        return [int(sample[field]) for sample in samples if sample[field] != "NA"]

    def extrema(field: str) -> dict[str, int | None]:
        values = numeric(field)
        return {"min": min(values) if values else None, "max": max(values) if values else None}

    return {
        "vmrss_kb": extrema("vmrss_kb"),
        "vmhwm_kb": extrema("vmhwm_kb"),
        "vmswap_kb": extrema("vmswap_kb"),
        "cgroup_memory_current_bytes": extrema("cgroup_memory_current_bytes"),
        "cgroup_memory_peak_bytes": extrema("cgroup_memory_peak_bytes"),
        "cgroup_memory_swap_current_bytes": extrema("cgroup_memory_swap_current_bytes"),
        "backing_logical_size": extrema("backing_logical_size"),
        "backing_allocated_bytes": extrema("backing_allocated_bytes"),
        "sample_count": len(samples),
    }


def validate_process_identity(value: Any, expected_argv: list[str], label: str) -> None:
    identity = exact(value, {"pid", "starttime_ticks", "cmdline", "cmdline_sha256"}, label)
    if isinstance(identity["pid"], bool) or not isinstance(identity["pid"], int) or identity["pid"] <= 0:
        raise ParseError(f"{label}: pid is invalid")
    if isinstance(identity["starttime_ticks"], bool) or not isinstance(identity["starttime_ticks"], int) or identity["starttime_ticks"] <= 0:
        raise ParseError(f"{label}: starttime_ticks is invalid")
    cmdline = identity["cmdline"]
    if not isinstance(cmdline, list) or not cmdline or any(not isinstance(item, str) or not item for item in cmdline):
        raise ParseError(f"{label}: cmdline is invalid")
    if not isinstance(expected_argv, list) or not expected_argv or any(not isinstance(item, str) or not item for item in expected_argv):
        raise ParseError(f"{label}: expected argv is invalid")
    if cmdline != expected_argv and (len(cmdline) < len(expected_argv) or cmdline[-len(expected_argv):] != expected_argv):
        raise ParseError(f"{label}: cmdline does not match launch argv")
    expected_hash = sha256_bytes(json.dumps(cmdline, separators=(",", ":")).encode("utf-8"))
    if identity["cmdline_sha256"] != expected_hash or not re.fullmatch(r"[0-9a-f]{64}", identity["cmdline_sha256"]):
        raise ParseError(f"{label}: cmdline hash mismatch")


def validate_cleanup_record(value: Any, label: str) -> dict[str, Any]:
    cleanup = exact(value, PROCESS_CLEANUP_KEYS, label)
    if cleanup["pid"] is not None and (isinstance(cleanup["pid"], bool) or not isinstance(cleanup["pid"], int) or cleanup["pid"] <= 0):
        raise ParseError(f"{label}: pid is invalid")
    if cleanup["pgid"] is not None and (isinstance(cleanup["pgid"], bool) or not isinstance(cleanup["pgid"], int) or cleanup["pgid"] <= 0):
        raise ParseError(f"{label}: pgid is invalid")
    if cleanup["exit_code"] is not None and (isinstance(cleanup["exit_code"], bool) or not isinstance(cleanup["exit_code"], int)):
        raise ParseError(f"{label}: exit code is invalid")
    for key in ("stop_requested", "term_timed_out", "kill_timed_out", "pgid_check_complete", "residual_process"):
        if not isinstance(cleanup[key], bool):
            raise ParseError(f"{label}: {key} is not boolean")
    if cleanup["stop_signal"] is not None and cleanup["stop_signal"] not in {"SIGTERM", "SIGKILL"}:
        raise ParseError(f"{label}: stop signal is invalid")
    if not cleanup["pgid_check_complete"]:
        raise ParseError(f"{label}: PGID residual check is incomplete")
    if cleanup["pid"] is None or cleanup["pgid"] is None or cleanup["exit_code"] is None:
        raise ParseError(f"{label}: cleanup identity is missing")
    return cleanup


def require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ParseError(f"{label} is not a non-negative integer")
    return value


def validate_qualification_record(
        value: Any,
        expected: dict[str, Any] | None,
        case: dict[str, Any],
        label: str,
) -> dict[str, Any]:
    qualification = exact(
        value,
        {
            "idle_seconds", "offload_timeout_seconds", "resume_request_id", "idle",
            "offload_barrier", "resume",
        },
        label,
    )
    if expected is None:
        if any(qualification[key] is not None for key in qualification):
            raise ParseError(f"{label}: non-qualification run contains qualification state")
        return qualification
    if qualification["idle_seconds"] != expected["idle_seconds"]:
        raise ParseError(f"{label}: idle_seconds differs from manifest workload")
    if qualification["offload_timeout_seconds"] != expected["offload_timeout_seconds"]:
        raise ParseError(f"{label}: offload_timeout_seconds differs from manifest workload")
    if qualification["resume_request_id"] != expected["resume_request_id"]:
        raise ParseError(f"{label}: resume_request_id differs from manifest workload")
    idle = exact(
        qualification["idle"],
        {"started_mono_ns", "finished_mono_ns", "duration_ns", "stderr_start_offset"},
        f"{label}.idle",
    )
    idle_started = require_nonnegative_int(idle["started_mono_ns"], f"{label}.idle.started_mono_ns")
    idle_finished = require_nonnegative_int(idle["finished_mono_ns"], f"{label}.idle.finished_mono_ns")
    idle_duration = require_nonnegative_int(idle["duration_ns"], f"{label}.idle.duration_ns")
    require_nonnegative_int(idle["stderr_start_offset"], f"{label}.idle.stderr_start_offset")
    minimum_idle_ns = int(float(expected["idle_seconds"]) * 1_000_000_000)
    if (
        idle_finished < idle_started
        or idle_duration != idle_finished - idle_started
        or idle_duration <= 0
        or idle_duration + 1_000_000 < minimum_idle_ns
    ):
        raise ParseError(f"{label}: idle timing is missing, inconsistent, or too short")
    if case["policy"] == "resident":
        if qualification["offload_barrier"] is not None or qualification["resume"] is not None:
            raise ParseError(f"{label}: resident qualification must not run OFFLOAD barrier or resume")
        return qualification
    barrier = exact(
        qualification["offload_barrier"],
        {
            "status", "started_mono_ns", "completed_mono_ns", "duration_ns",
            "stderr_start_offset", "stderr_end_offset", "decision_id", "transaction_id", "seq_id",
            "selected_claimant_epoch", "before_object_id", "before_generation", "before_resident_bytes", "after_object_id",
            "after_generation", "after_resident_bytes", "resident_drop_bytes",
        },
        f"{label}.offload_barrier",
    )
    if barrier["status"] != "passed":
        raise ParseError(f"{label}: real OFFLOAD barrier did not pass")
    for key in (
        "started_mono_ns", "completed_mono_ns", "duration_ns", "stderr_start_offset",
        "stderr_end_offset", "decision_id", "transaction_id", "seq_id", "selected_claimant_epoch",
        "before_object_id",
        "before_generation", "before_resident_bytes", "after_object_id", "after_generation",
        "after_resident_bytes", "resident_drop_bytes",
    ):
        require_nonnegative_int(barrier[key], f"{label}.offload_barrier.{key}")
    if (
        barrier["started_mono_ns"] < idle_finished
        or barrier["completed_mono_ns"] < barrier["started_mono_ns"]
        or barrier["duration_ns"] != barrier["completed_mono_ns"] - barrier["started_mono_ns"]
        or barrier["stderr_start_offset"] != idle["stderr_start_offset"]
        or barrier["stderr_end_offset"] < barrier["stderr_start_offset"]
        or barrier["before_object_id"] != barrier["after_object_id"]
        or barrier["before_generation"] != barrier["after_generation"]
        or barrier["selected_claimant_epoch"] <= 0
        or barrier["after_resident_bytes"] >= barrier["before_resident_bytes"]
        or barrier["resident_drop_bytes"]
            != barrier["before_resident_bytes"] - barrier["after_resident_bytes"]
        or barrier["resident_drop_bytes"] <= 0
    ):
        raise ParseError(f"{label}: OFFLOAD barrier timing or physical-drop record is inconsistent")
    resume = exact(
        qualification["resume"],
        {
            "sequence", "request_id", "started_mono_ns", "finished_mono_ns",
            "stderr_start_offset", "stderr_end_offset",
        },
        f"{label}.resume",
    )
    for key in (
        "sequence", "started_mono_ns", "finished_mono_ns", "stderr_start_offset",
        "stderr_end_offset",
    ):
        require_nonnegative_int(resume[key], f"{label}.resume.{key}")
    if (
        resume["request_id"] != expected["resume_request_id"]
        or resume["started_mono_ns"] <= barrier["completed_mono_ns"]
        or resume["finished_mono_ns"] < resume["started_mono_ns"]
        or resume["stderr_start_offset"] < barrier["stderr_end_offset"]
        or resume["stderr_end_offset"] < resume["stderr_start_offset"]
    ):
        raise ParseError(f"{label}: resume request did not occur strictly after the OFFLOAD barrier")
    return qualification


def validate_qualification_causality(
        qualification: dict[str, Any],
        records: list[dict[str, Any]],
        stderr_data: bytes,
        case: dict[str, Any],
        spec: dict[str, Any],
        label: str,
) -> dict[str, Any] | None:
    expected = spec["workload"]["qualification"]
    if expected is None:
        return None
    idle = qualification["idle"]
    if case["policy"] == "resident":
        fill_records = records
    else:
        if len(records) < 2:
            raise ParseError(f"{label}: V2 qualification is missing fill or resume requests")
        fill_records = records[:-1]
    if not fill_records or max(record["finished_mono_ns"] for record in fill_records) > idle["started_mono_ns"]:
        raise ParseError(f"{label}: qualification idle began before seed/fill completed")
    if case["policy"] == "resident":
        return None
    barrier = qualification["offload_barrier"]
    resume_record = records[-1]
    resume = qualification["resume"]
    if (
        resume_record["sequence"] != resume["sequence"]
        or resume_record["request_id"] != resume["request_id"]
        or resume_record["started_mono_ns"] != resume["started_mono_ns"]
        or resume_record["finished_mono_ns"] != resume["finished_mono_ns"]
        or resume_record["measurement"] is not False
    ):
        raise ParseError(f"{label}: recorded resume request does not match the real measurement request replay")
    barrier_text = stderr_window(
        stderr_data,
        barrier["stderr_start_offset"],
        barrier["stderr_end_offset"],
        f"{label}.offload_barrier",
    )
    barrier_actions = marker_records(
        barrier_text, "kv_pressure_unified_action", ACTION_REQUIRED, f"{label}.offload_barrier.action")
    for action in barrier_actions:
        validate_action_fields(action, f"{label}.offload_barrier.action")
        boolean_fields(action, ACTION_BOOL, f"{label}.offload_barrier.action")
    barrier_observations = marker_records(
        barrier_text,
        "kv_g0_s1_resident_observation",
        RESIDENT_OBSERVATION_REQUIRED,
        f"{label}.offload_barrier.resident",
    )
    for observation in barrier_observations:
        numeric_fields(
            observation, RESIDENT_OBSERVATION_NUMERIC, f"{label}.offload_barrier.resident")
        boolean_fields(
            observation, RESIDENT_OBSERVATION_BOOL, f"{label}.offload_barrier.resident")
    pairs = qualified_offload_pairs(
        barrier_actions, barrier_observations, pressure_basis_source(spec["pressure_basis"]))
    barrier_fields = {
        key: barrier[key]
        for key in (
            "decision_id", "transaction_id", "seq_id", "selected_claimant_epoch",
            "before_object_id", "before_generation", "before_resident_bytes", "after_object_id", "after_generation", "after_resident_bytes",
            "resident_drop_bytes",
        )
    }
    if not any(all(pair[key] == value for key, value in barrier_fields.items()) for pair in pairs):
        raise ParseError(
            f"{label}: OFFLOAD barrier has no matching successful action and transaction-local resident drop")
    resume_text = stderr_window(
        stderr_data,
        resume["stderr_start_offset"],
        resume["stderr_end_offset"],
        f"{label}.resume",
    )
    resume_events = marker_records(
        resume_text, "kv_resume_order_event", RESUME_REQUIRED, f"{label}.qualification_resume")
    for event in resume_events:
        numeric_fields(
            event, {"decision_id", "seq_id", "claimant_epoch", "transaction_id"},
            f"{label}.qualification_resume")
        boolean_fields(event, RESUME_BOOL, f"{label}.qualification_resume")
    resume_timings = marker_records(
        resume_text, "kv_resume_stage_timing", TIMING_REQUIRED, f"{label}.qualification_timing")
    for timing in resume_timings:
        numeric_fields(timing, TIMING_REQUIRED, f"{label}.qualification_timing")
    evidence = validate_resume_restore_evidence(
        resume_events,
        resume_timings,
        {barrier["seq_id"]: {barrier["selected_claimant_epoch"]}},
        f"{label}.qualification_resume",
    )
    return {
        "offload": barrier_fields,
        "prefetch": evidence["prefetch"],
        "graph_gate": evidence["graph_gate"],
        "timing": evidence["timing"],
        "resume_request": resume,
    }


def validate_snapshot_reference(value: Any, expected_path: str, label: str) -> dict[str, Any]:
    reference = exact(value, {"path", "captured_mono_ns"}, label)
    if reference["path"] != expected_path:
        raise ParseError(f"{label}: snapshot path mismatch")
    require_nonnegative_int(reference["captured_mono_ns"], f"{label}.captured_mono_ns")
    return reference


def validate_characterization_record(
        value: Any,
        expected: dict[str, Any] | None,
        case: dict[str, Any],
        max_blocks: int,
        label: str,
) -> dict[str, Any]:
    record = exact(
        value,
        {
            "idle_seconds", "settle_timeout_seconds", "target_tolerance_bytes",
            "resume_request_id", "requested_target_bytes", "action_target_bytes", "after_fill", "idle", "settle",
            "release_settled", "settled", "resume", "after_measurement",
        },
        label,
    )
    if expected is None:
        if any(record[key] is not None for key in record):
            raise ParseError(f"{label}: non-characterization run contains characterization state")
        return record
    for key in (
        "idle_seconds", "settle_timeout_seconds", "target_tolerance_bytes", "resume_request_id",
    ):
        if record[key] != expected[key]:
            raise ParseError(f"{label}: {key} differs from manifest workload")
    if record["requested_target_bytes"] != case["kv_target_bytes"]:
        raise ParseError(f"{label}: requested target differs from the case factor")
    if record["action_target_bytes"] != case["action_target_bytes"]:
        raise ParseError(f"{label}: action target differs from the case factor")
    validate_snapshot_reference(
        record["after_fill"], "slots_after_fill.json", f"{label}.after_fill")
    validate_snapshot_reference(
        record["after_measurement"], "slots_after_measurement.json",
        f"{label}.after_measurement",
    )
    if case["policy"] == "resident":
        if any(record[key] is not None for key in ("idle", "settle", "release_settled", "settled", "resume")):
            raise ParseError(f"{label}: resident characterization contains budget-settle state")
        return record

    idle = exact(
        record["idle"],
        {"started_mono_ns", "finished_mono_ns", "duration_ns", "stderr_start_offset"},
        f"{label}.idle",
    )
    idle_started = require_nonnegative_int(
        idle["started_mono_ns"], f"{label}.idle.started_mono_ns")
    idle_finished = require_nonnegative_int(
        idle["finished_mono_ns"], f"{label}.idle.finished_mono_ns")
    idle_duration = require_nonnegative_int(idle["duration_ns"], f"{label}.idle.duration_ns")
    require_nonnegative_int(idle["stderr_start_offset"], f"{label}.idle.stderr_start_offset")
    minimum_idle_ns = int(float(expected["idle_seconds"]) * 1_000_000_000)
    if (
        idle_finished < idle_started
        or idle_duration != idle_finished - idle_started
        or idle_duration <= 0
        or idle_duration + 1_000_000 < minimum_idle_ns
    ):
        raise ParseError(f"{label}: idle timing is missing, inconsistent, or too short")

    settle = exact(
        record["settle"],
        {
            "status", "started_mono_ns", "completed_mono_ns", "duration_ns",
            "stderr_start_offset", "stderr_end_offset", "terminal_decision",
            "physical_resident_bytes", "release_settle", "release_boundary", "decision_ids", "offload_decision_ids",
        },
        f"{label}.settle",
    )
    if settle["status"] not in {
        "target_reached", "unmet_floor", "release_settled", "release_no_candidate",
    }:
        raise ParseError(f"{label}: completed characterization has no terminal settle state")
    for key in (
        "started_mono_ns", "completed_mono_ns", "duration_ns", "stderr_start_offset",
        "stderr_end_offset",
    ):
        require_nonnegative_int(settle[key], f"{label}.settle.{key}")
    physical_resident_bytes = settle["physical_resident_bytes"]
    if physical_resident_bytes is None:
        if not (case["policy"] == "v2" and settle["status"] == "unmet_floor"):
            raise ParseError(f"{label}: settled physical resident observation is missing")
    else:
        require_nonnegative_int(
            physical_resident_bytes, f"{label}.settle.physical_resident_bytes")
    if (
        settle["started_mono_ns"] < idle_finished
        or settle["completed_mono_ns"] < settle["started_mono_ns"]
        or settle["duration_ns"] != settle["completed_mono_ns"] - settle["started_mono_ns"]
        or settle["stderr_start_offset"] != idle["stderr_start_offset"]
        or settle["stderr_end_offset"] < settle["stderr_start_offset"]
    ):
        raise ParseError(f"{label}: settle timing or stderr window is inconsistent")
    if case["policy"] == "v2":
        validate_snapshot_reference(
            settle["release_settle"], "slots_release_settled.json", f"{label}.settle.release_settle")
        boundary = exact(
            settle["release_boundary"],
            {
                "status", "decision_id", "budget_resident_bytes", "budget_debt_after_bytes",
                "unmet_budget_bytes_after", "state_changed", "bytes", "relieved_bytes",
                "soft_offload_armed_after",
            },
            f"{label}.settle.release_boundary",
        )
        if boundary["status"] not in {"release_no_candidate", "release_settled"}:
            raise ParseError(f"{label}: invalid RELEASE phase-boundary status")
        for key in (
            "decision_id", "budget_resident_bytes", "budget_debt_after_bytes",
            "unmet_budget_bytes_after", "bytes", "relieved_bytes",
        ):
            require_nonnegative_int(boundary[key], f"{label}.settle.release_boundary.{key}")
        if not isinstance(boundary["state_changed"], bool) or not isinstance(
                boundary["soft_offload_armed_after"], bool):
            raise ParseError(f"{label}: RELEASE phase-boundary boolean is malformed")
        if boundary["status"] == "release_no_candidate":
            if (
                boundary["state_changed"]
                or boundary["bytes"] != 0
                or boundary["relieved_bytes"] != 0
                or boundary["budget_debt_after_bytes"] <= 0
                or boundary["unmet_budget_bytes_after"] != 0
                or not boundary["soft_offload_armed_after"]
            ):
                raise ParseError(f"{label}: RELEASE no_candidate boundary is not an armed soft-budget terminal")
        elif (
            not boundary["state_changed"]
            or boundary["bytes"] <= 0
            or boundary["relieved_bytes"] <= 0
            or boundary["budget_debt_after_bytes"] != 0
            or boundary["unmet_budget_bytes_after"] != 0
            or boundary["soft_offload_armed_after"]
        ):
            raise ParseError(f"{label}: RELEASE target closure boundary is inconsistent")
    elif settle["release_settle"] is not None or settle["release_boundary"] is not None:
        raise ParseError(f"{label}: release_only settle contains an intermediate release boundary")
    for key in ("decision_ids", "offload_decision_ids"):
        values = settle[key]
        if not isinstance(values, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values):
            raise ParseError(f"{label}.settle.{key} is invalid")
        if len(values) != len(set(values)):
            raise ParseError(f"{label}.settle.{key} contains duplicates")
    if not set(settle["offload_decision_ids"]).issubset(set(settle["decision_ids"])):
        raise ParseError(f"{label}: OFFLOAD decision list escapes settle decisions")
    terminal = exact(
        settle["terminal_decision"],
        {
            "decision_id", "decision_reason", "action_target_bytes", "max_blocks",
            "budget_target_bytes", "budget_resident_bytes", "budget_observed_excess_bytes", "budget_debt_after_bytes",
            "unmet_budget_bytes_after", "budget_transient_staging_bound_bytes",
            "soft_offload_armed_before", "soft_offload_armed_after",
            "positive_offload", "positive_release",
        },
        f"{label}.settle.terminal_decision",
    )
    for key in (
        "decision_id", "action_target_bytes", "max_blocks", "budget_target_bytes", "budget_resident_bytes",
        "budget_observed_excess_bytes", "budget_debt_after_bytes",
        "unmet_budget_bytes_after", "budget_transient_staging_bound_bytes",
    ):
        require_nonnegative_int(terminal[key], f"{label}.settle.terminal_decision.{key}")
    if (
        not isinstance(terminal["decision_reason"], str)
        or not isinstance(terminal["positive_offload"], bool)
        or not isinstance(terminal["positive_release"], bool)
        or terminal["soft_offload_armed_before"] not in {"0", "1"}
        or terminal["soft_offload_armed_after"] not in {"0", "1"}
    ):
        raise ParseError(f"{label}: terminal decision fields are malformed")
    if (
        terminal["action_target_bytes"] != case["action_target_bytes"]
        or terminal["max_blocks"] != max_blocks
        or terminal["budget_target_bytes"] != case["kv_target_bytes"]
    ):
        raise ParseError(f"{label}: terminal decision target fields disagree with the case")
    if terminal["decision_id"] not in settle["decision_ids"]:
        raise ParseError(f"{label}: terminal decision is absent from settle decision list")
    validate_snapshot_reference(record["settled"], "slots_settled.json", f"{label}.settled")
    validate_snapshot_reference(record["release_settled"], "slots_settled.json" if case["policy"] == "release_only" else "slots_release_settled.json", f"{label}.release_settled")
    if case["policy"] == "release_only":
        if record["release_settled"] != record["settled"]:
            raise ParseError(f"{label}: release_only release/settled snapshots differ")
        if record["resume"] is not None:
            raise ParseError(f"{label}: release_only characterization contains a resume")
        return record
    resume = exact(
        record["resume"],
        {
            "sequence", "request_id", "started_mono_ns", "finished_mono_ns",
            "stderr_start_offset", "stderr_end_offset",
        },
        f"{label}.resume",
    )
    for key in (
        "sequence", "started_mono_ns", "finished_mono_ns", "stderr_start_offset",
        "stderr_end_offset",
    ):
        require_nonnegative_int(resume[key], f"{label}.resume.{key}")
    if (
        resume["request_id"] != expected["resume_request_id"]
        or resume["started_mono_ns"] <= settle["completed_mono_ns"]
        or resume["finished_mono_ns"] < resume["started_mono_ns"]
        or resume["stderr_start_offset"] < settle["stderr_end_offset"]
        or resume["stderr_end_offset"] < resume["stderr_start_offset"]
    ):
        raise ParseError(f"{label}: resume did not occur strictly after budget settle")
    return record


def optional_authoritative_slot_resident(
        snapshot: dict[str, Any], label: str) -> dict[str, Any] | None:
    value = snapshot.get("body_json")
    if not isinstance(value, list):
        raise ParseError(f"{label}: slot snapshot body is unavailable")
    observations = [
        slot["kv_resident"] for slot in value
        if isinstance(slot, dict) and isinstance(slot.get("kv_resident"), dict)
    ]
    if not observations:
        return None
    first = observations[0]
    if any(item != first for item in observations[1:]):
        raise ParseError(f"{label}: slots disagree on the global physical resident observation")
    return first


def authoritative_slot_resident(snapshot: dict[str, Any], label: str) -> dict[str, Any]:
    resident = optional_authoritative_slot_resident(snapshot, label)
    if resident is None:
        raise ParseError(f"{label}: physical resident observation is missing")
    return resident


def release_boundary_status(action: dict[str, str]) -> str | None:
    if action["release_attempted"] != "1" or action["offload_attempted"] != "0":
        return None
    if (
        action["outcome"] == "no_op"
        and action["reason"] == "no_candidate"
        and action["state_changed"] == "0"
        and action["io_failure"] == "0"
        and int(action["blocks"]) == 0
        and int(action["bytes"]) == 0
        and int(action["relieved_bytes"]) == 0
        and action["transaction_id"] == "0"
        and int(action["budget_debt_after_bytes"]) > 0
        and int(action["unmet_budget_bytes_after"]) == 0
        and action["soft_offload_armed_after"] == "1"
    ):
        return "release_no_candidate"
    if (
        action["outcome"] == "completed"
        and action["state_changed"] == "1"
        and action["io_failure"] == "0"
        and int(action["blocks"]) > 0
        and int(action["bytes"]) > 0
        and int(action["relieved_bytes"]) > 0
        and int(action["budget_debt_after_bytes"]) == 0
        and int(action["unmet_budget_bytes_after"]) == 0
        and action["soft_offload_armed_after"] == "0"
    ):
        return "release_settled"
    return None


def release_boundary_from_actions(
        actions: list[dict[str, str]],
) -> tuple[int, str, dict[str, str]] | None:
    for index, action in enumerate(actions):
        status = release_boundary_status(action)
        if status is not None:
            return index, status, action
    return None


def characterization_budget_observations(
        actions: list[dict[str, str]],
        expected_source: str,
        resident_target_bytes: int,
        action_target_bytes: int,
        max_blocks: int,
        tolerance_bytes: int,
        label: str,
        release_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    for action in actions:
        if (
            action["state"] != "NORMAL"
            or action["source"] != expected_source
            or action["sample_valid"] != "1"
            or action["stale"] != "0"
            or action["pressure_basis_valid"] != "1"
            or action["budget_active"] != "1"
            or action["budget_target_enabled"] != "1"
            or action["budget_source"] != "env_static"
            or int(action["target_bytes"]) != action_target_bytes
            or int(action["max_blocks"]) != max_blocks
            or int(action["budget_target_bytes"]) != resident_target_bytes
            or action["budget_view_valid"] != "1"
            or action["budget_resident_available"] != "1"
            or action["idle"] != "1"
        ):
            continue
        resident_bytes = int(action["budget_resident_bytes"])
        observed_excess = max(0, resident_bytes - resident_target_bytes)
        if int(action["budget_observed_excess_bytes"]) != observed_excess:
            raise ParseError(f"{label}: target and observed physical resident are inconsistent")
        positive_offload = (
            action["offload_attempted"] == "1"
            and action["outcome"] == "completed"
            and action["state_changed"] == "1"
            and action["io_failure"] == "0"
            and int(action["blocks"]) > 0
            and int(action["bytes"]) > 0
            and int(action["relieved_bytes"]) > 0
        )
        positive_release = (
            action["release_attempted"] == "1"
            and action["outcome"] == "completed"
            and action["state_changed"] == "1"
            and action["io_failure"] == "0"
            and int(action["blocks"]) > 0
            and int(action["relieved_bytes"]) > 0
        )
        release_no_candidate = release_only and release_boundary_status(action) == "release_no_candidate"
        summary = {
            "decision_id": int(action["decision_id"]),
            "decision_reason": action["decision_reason"],
            "action_target_bytes": action_target_bytes,
            "max_blocks": max_blocks,
            "budget_target_bytes": resident_target_bytes,
            "budget_resident_bytes": resident_bytes,
            "budget_observed_excess_bytes": observed_excess,
            "budget_debt_after_bytes": int(action["budget_debt_after_bytes"]),
            "unmet_budget_bytes_after": int(action["unmet_budget_bytes_after"]),
            "budget_transient_staging_bound_bytes": int(
                action["budget_transient_staging_bound_bytes"]),
            "soft_offload_armed_before": action["soft_offload_armed_before"],
            "soft_offload_armed_after": action["soft_offload_armed_after"],
            "positive_offload": positive_offload,
            "positive_release": positive_release,
            "action": action,
        }
        observations.append(summary)
        if release_no_candidate:
            terminal = {"status": "release_no_candidate", **summary}
            break
        if (
            (positive_offload or positive_release)
            and summary["budget_debt_after_bytes"] == 0
            and summary["unmet_budget_bytes_after"] == 0
        ):
            terminal = {
                "status": "release_settled" if release_only else "debt_closed",
                **summary,
            }
            break
        if (
            action["decision_reason"] == "budget_target_satisfied"
            and resident_bytes <= resident_target_bytes + tolerance_bytes
            and observed_excess == 0
            and summary["budget_debt_after_bytes"] == 0
            and summary["unmet_budget_bytes_after"] == 0
        ):
            terminal = {"status": "target_reached", **summary}
            break
        if (
            action["decision_reason"] == "budget_unmet_terminal"
            and observed_excess > 0
            and summary["budget_debt_after_bytes"] == observed_excess
            and summary["unmet_budget_bytes_after"] == observed_excess
        ):
            terminal = {"status": "unmet_floor", **summary}
            break
    if terminal is None:
        raise ParseError(f"{label}: no production target-closure or budget_unmet_terminal marker")
    return observations, terminal


def resume_event_base_key(event: dict[str, str]) -> tuple[str, str, str]:
    return event["decision_id"], event["transaction_id"], event["seq_id"]


def resume_event_key(event: dict[str, str]) -> tuple[str, str, str, str]:
    return (*resume_event_base_key(event), event["claimant_epoch"])


def validate_resume_restore_evidence(
        resumes: list[dict[str, str]],
        timings: list[dict[str, str]],
        selected_offload_epochs: dict[int, set[int]],
        label: str,
) -> dict[str, dict[str, str]]:
    if not resumes or not timings:
        raise ParseError(f"{label}: positive resume requires order and timing markers")

    phases_by_key: dict[tuple[str, str, str, str], set[str]] = {}
    events_by_base: dict[tuple[str, str, str], dict[str, dict[str, str]]] = {}
    for event in resumes:
        if (
            event["action"] != "prefetch"
            or event["outcome"] != "completed"
            or event["graph_allowed"] != "1"
            or event["phase"] not in {"prefetch", "graph_gate"}
        ):
            raise ParseError(
                f"{label}: resume contains no completed graph-allowed PREFETCH evidence")
        full_key = resume_event_key(event)
        phases = phases_by_key.setdefault(full_key, set())
        if event["phase"] in phases:
            raise ParseError(f"{label}: PREFETCH phase is duplicated")
        phases.add(event["phase"])
        base_key = resume_event_base_key(event)
        event_by_phase = events_by_base.setdefault(base_key, {})
        if event["phase"] in event_by_phase:
            raise ParseError(f"{label}: resume PREFETCH key is duplicated")
        event_by_phase[event["phase"]] = event

    if any(phases != {"prefetch", "graph_gate"} for phases in phases_by_key.values()):
        raise ParseError(f"{label}: PREFETCH lacks a paired graph gate with the same claimant epoch")
    if len({full_key[:3] for full_key in phases_by_key}) != len(phases_by_key):
        raise ParseError(f"{label}: resume key has multiple claimant epochs")

    timings_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for timing in timings:
        key = (timing["decision_id"], timing["transaction_id"], timing["seq_id"])
        if key in timings_by_key:
            raise ParseError(f"{label}: resume timing key is duplicated")
        timings_by_key[key] = timing
    if set(timings_by_key) != set(events_by_base):
        raise ParseError(f"{label}: PREFETCH, graph_gate, and timing keys do not match")

    positive: list[dict[str, dict[str, str]]] = []
    for base_key, events in events_by_base.items():
        timing = timings_by_key[base_key]
        if (
            int(timing["restored_blocks"]) <= 0
            or int(timing["restored_bytes"]) <= 0
            or int(timing["total_us"]) <= 0
        ):
            raise ParseError(f"{label}: completed PREFETCH lacks positive restore timing")
        prefetch = events["prefetch"]
        seq_id = int(prefetch["seq_id"])
        if seq_id not in selected_offload_epochs:
            raise ParseError(
                f"{label}: positive PREFETCH seq_id is not selected by this round's OFFLOAD actions")
        if int(prefetch["claimant_epoch"]) not in selected_offload_epochs[seq_id]:
            raise ParseError(
                f"{label}: positive PREFETCH claimant_epoch does not match the selected OFFLOAD claimant")
        positive.append({"prefetch": prefetch, "graph_gate": events["graph_gate"], "timing": timing})

    if not positive:
        raise ParseError(f"{label}: no positive resume restore evidence")
    return positive[0]


def validate_release_only_noop_restore(
        resumes: list[dict[str, str]], timings: list[dict[str, str]], label: str) -> None:
    if not resumes and not timings:
        return
    phases_by_key: dict[tuple[str, str, str, str], set[str]] = {}
    for event in resumes:
        if (
            event["action"] != "prefetch"
            or event["outcome"] != "no_op"
            or event["graph_allowed"] != "1"
            or event["phase"] not in {"prefetch", "graph_gate"}
        ):
            raise ParseError(
                f"{label}: RELEASE-only contains positive or malformed PREFETCH evidence")
        key = resume_event_key(event)
        phases = phases_by_key.setdefault(key, set())
        if event["phase"] in phases:
            raise ParseError(f"{label}: RELEASE-only PREFETCH phase is duplicated")
        phases.add(event["phase"])
    if any(phases != {"prefetch", "graph_gate"} for phases in phases_by_key.values()):
        raise ParseError(f"{label}: RELEASE-only PREFETCH lacks a paired graph gate")
    if len({key[:3] for key in phases_by_key}) != len(phases_by_key):
        raise ParseError(f"{label}: RELEASE-only PREFETCH key has multiple claimant epochs")

    timings_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for timing in timings:
        key = (timing["decision_id"], timing["transaction_id"], timing["seq_id"])
        if key in timings_by_key:
            raise ParseError(f"{label}: RELEASE-only restore timing is duplicated")
        if int(timing["restored_blocks"]) != 0 or int(timing["restored_bytes"]) != 0:
            raise ParseError(f"{label}: RELEASE-only contains positive restore evidence")
        timings_by_key[key] = timing
    if set(timings_by_key) != {key[:3] for key in phases_by_key}:
        raise ParseError(f"{label}: RELEASE-only PREFETCH timing does not match no-op markers")


def k2_statistics(io: dict[str, str]) -> dict[str, Any]:
    return {
        "enabled": int(io["k2_enabled"]),
        "group_byte_cap": int(io["k2_group_byte_cap"]),
        "pipeline_wall_us": int(io["k2_pipeline_wall_us"]),
        "read_wait_us": int(io["k2_exposed_read_wait_us"]),
        "pipeline_stall_us": int(io["k2_pipeline_stall_us"]),
        "read_completed_ahead": int(io["k2_read_completed_ahead"]),
        "validate_us": int(io["block_in_validate_us"]),
        "read_us": int(io["block_in_read_us"]),
        "unpack_us": int(io["block_in_unpack_us"]),
        "commit_us": int(io["block_in_commit_us"]),
        "prefault": {
            "enabled": int(io["restore_prefault_enabled"]),
            "groups": int(io["restore_prefault_groups"]),
            "calls": int(io["restore_prefault_calls"]),
            "us": int(io["restore_prefault_us"]),
            "minor_faults": int(io["restore_prefault_minor_faults"]),
            "major_faults": int(io["restore_prefault_major_faults"]),
        },
        "scatter": {
            "groups": int(io["restore_scatter_groups"]),
            "us": int(io["restore_scatter_us"]),
            "fault_groups": int(io["restore_scatter_fault_groups"]),
            "minor_faults": int(io["restore_scatter_minor_faults"]),
            "major_faults": int(io["restore_scatter_major_faults"]),
        },
    }


def validate_characterization_causality(
        record: dict[str, Any],
        responses: list[dict[str, Any]],
        stderr_data: bytes,
        case: dict[str, Any],
        spec: dict[str, Any],
        run_dir: pathlib.Path,
        io: dict[str, str],
        label: str,
        qualified_pairs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    warmup_count = len(spec["workload"]["warmup"])
    if len(responses) <= warmup_count:
        raise ParseError(f"{label}: characterization has no post-fill measurement")
    warmup = responses[:warmup_count]
    measurements = responses[warmup_count:]
    if any(item["measurement"] for item in warmup) or any(
            not item["measurement"] for item in measurements):
        raise ParseError(f"{label}: fill and measurement responses are not separated")
    if case["policy"] == "v2":
        if measurements[0]["measurement_phase"] != "resume" or any(
                item["measurement_phase"] != "post_resume_steady"
                for item in measurements[1:]):
            raise ParseError(
                f"{label}: characterization measurements are not classified as resume then post-resume steady")
    elif case["policy"] == "release_only":
        if any(item["measurement_phase"] != "release_only_steady" for item in measurements):
            raise ParseError(f"{label}: RELEASE-only measurements are not classified as release_only_steady")
    elif any(item["measurement_phase"] != "resident_steady" for item in measurements):
        raise ParseError(f"{label}: Resident measurements are not classified as resident_steady")

    after_fill_snapshot = validate_slot_snapshot(
        run_dir / "slots_after_fill.json",
        f"{label}.slots_after_fill",
        require_resident=case["policy"] != "v2",
    )
    after_measurement_snapshot = validate_slot_snapshot(
        run_dir / "slots_after_measurement.json",
        f"{label}.slots_after_measurement",
        require_resident=case["policy"] != "v2",
    )
    if (
        record["after_fill"]["captured_mono_ns"] != after_fill_snapshot["captured_mono_ns"]
        or record["after_measurement"]["captured_mono_ns"]
            != after_measurement_snapshot["captured_mono_ns"]
    ):
        raise ParseError(f"{label}: characterization snapshot reference timestamp mismatch")
    fill_finished = max(item["finished_mono_ns"] for item in warmup)
    first_measurement = measurements[0]
    if (
        after_fill_snapshot["captured_mono_ns"] < fill_finished
        or after_fill_snapshot["captured_mono_ns"] > first_measurement["started_mono_ns"]
        or after_measurement_snapshot["captured_mono_ns"] < first_measurement["finished_mono_ns"]
    ):
        raise ParseError(f"{label}: fill/measurement physical snapshots are out of order")
    if case["policy"] == "v2":
        resident_after_fill = optional_authoritative_slot_resident(
            after_fill_snapshot, f"{label}.slots_after_fill")
        resident_after_measurement = optional_authoritative_slot_resident(
            after_measurement_snapshot, f"{label}.slots_after_measurement")
    else:
        resident_after_fill = authoritative_slot_resident(
            after_fill_snapshot, f"{label}.slots_after_fill")
        resident_after_measurement = authoritative_slot_resident(
            after_measurement_snapshot, f"{label}.slots_after_measurement")

    if case["policy"] == "resident":
        return {
            "status": "RESIDENT_BASELINE",
            "performance_eligible": True,
            "release_terminal": None,
            "requested_target_bytes": None,
            "action_target_bytes": None,
            "resume_measurement": None,
            "post_resume_steady_measurements": [],
            "resident_after_fill": resident_after_fill["resident_bytes"],
            "resident_after_release_settle": None,
            "resident_after_offload_settle": None,
            "resident_settled": resident_after_fill["resident_bytes"],
            "resident_after_resume": resident_after_measurement["resident_bytes"],
            "release_physical_relief_bytes": 0,
            "release_physical_relief_authority": "none",
            "offload_physical_relief_bytes": 0,
            "offload_physical_relief_authority": "none",
            "total_physical_relief_bytes": 0,
            "total_physical_relief_authority": "sum_of_authorities",
            "physical_relief_bytes": 0,
            "memory_saved_bytes": 0,
            "memory_saved_ratio": 0.0,
            "budget_debt_after": 0,
            "settle_time_seconds": None,
            "unmet_budget_bytes": 0,
            "release": {
                "actions": 0, "blocks": 0, "bytes": 0,
                "physical_relieved_bytes": 0,
            },
            "offload": {
                "actions": 0, "blocks": 0, "bytes": 0, "physical_relieved_bytes": 0,
                "write_syscalls": int(io["backing_write_syscalls"]),
                "bytes_written": int(io["bytes_written"]),
            },
            "resume": {
                "restored_blocks": 0, "restored_bytes": 0,
                "read_syscalls": int(io["backing_read_syscalls"]),
                "bytes_read": int(io["bytes_read"]), "gate_us": None, "total_us": None,
            },
            "transient_staging_peak_bytes": int(io["k2_peak_staging_bytes"]),
            "transient_staging_bound_bytes": int(io["k2_staging_bound_bytes"]),
            "k2": k2_statistics(io),
        }

    if case["policy"] == "release_only":
        settled_snapshot = validate_slot_snapshot(
            run_dir / "slots_settled.json", f"{label}.slots_settled")
        if record["settled"]["captured_mono_ns"] != settled_snapshot["captured_mono_ns"]:
            raise ParseError(f"{label}: settled snapshot reference timestamp mismatch")
        if record["release_settled"]["captured_mono_ns"] != settled_snapshot["captured_mono_ns"]:
            raise ParseError(f"{label}: release settled snapshot reference timestamp mismatch")
        settled_resident = authoritative_slot_resident(settled_snapshot, f"{label}.slots_settled")
        identity_keys = ("object_id", "generation", "page_size", "total_bytes", "total_pages")
        if any(
            resident_after_fill[key] != settled_resident[key]
            or settled_resident[key] != resident_after_measurement[key]
            for key in identity_keys
        ):
            raise ParseError(f"{label}: physical resident identity changed across RELEASE-only characterization")
        if settled_resident["resident_bytes"] > resident_after_fill["resident_bytes"]:
            raise ParseError(f"{label}: RELEASE-only settled resident exceeds after-fill resident")
        idle = record["idle"]
        settle = record["settle"]
        if (
            settled_snapshot["captured_mono_ns"] < settle["completed_mono_ns"]
            or settled_snapshot["captured_mono_ns"] > first_measurement["started_mono_ns"]
        ):
            raise ParseError(f"{label}: RELEASE-only settled resident was not captured before measurement")
        settle_text = stderr_window(
            stderr_data, settle["stderr_start_offset"], settle["stderr_end_offset"],
            f"{label}.settle",
        )
        settle_actions = marker_records(
            settle_text, "kv_pressure_unified_action", ACTION_REQUIRED,
            f"{label}.settle.action",
        )
        for action in settle_actions:
            validate_action_fields(action, f"{label}.settle.action")
            boolean_fields(action, ACTION_BOOL, f"{label}.settle.action")
        observations, terminal = characterization_budget_observations(
            settle_actions,
            pressure_basis_source(spec["pressure_basis"]),
            int(case["kv_target_bytes"]),
            int(case["action_target_bytes"]),
            int(spec["max_blocks"]),
            int(spec["workload"]["characterization"]["target_tolerance_bytes"]),
            f"{label}.settle",
            release_only=True,
        )
        terminal_record = {key: value for key, value in terminal.items() if key not in {"status", "action"}}
        if (
            settle["status"] != terminal["status"]
            or settle["decision_ids"] != [item["decision_id"] for item in observations]
            or settle["offload_decision_ids"]
            or settle["terminal_decision"] != terminal_record
        ):
            raise ParseError(f"{label}: RELEASE-only settle record differs from production markers")
        tolerance = int(spec["workload"]["characterization"]["target_tolerance_bytes"])
        if terminal["status"] == "release_settled":
            if (
                settled_resident["resident_bytes"] > int(case["kv_target_bytes"]) + tolerance
                or terminal["budget_debt_after_bytes"] != 0
                or terminal["unmet_budget_bytes_after"] != 0
                or terminal["soft_offload_armed_after"] != "0"
            ):
                raise ParseError(f"{label}: RELEASE-only target closure lacks physical resident closure")
        elif terminal["status"] == "release_no_candidate":
            if abs(
                settled_resident["resident_bytes"] - terminal["budget_resident_bytes"]
            ) > tolerance:
                raise ParseError(f"{label}: RELEASE no_candidate differs from settled physical resident")
        else:
            raise ParseError(f"{label}: invalid RELEASE-only terminal status")
        release_actions = [
            item["action"] for item in observations if item["action"]["release_attempted"] == "1"
        ]
        release_physical_relief = (
            resident_after_fill["resident_bytes"] - settled_resident["resident_bytes"])
        performance_eligible = terminal["status"] == "release_settled"
        return {
            "status": "RELEASE_SETTLED" if performance_eligible else "RELEASE_FLOOR_PROBE",
            "release_terminal": terminal["status"],
            "release_probe_kind": "target_point" if performance_eligible else "floor_probe",
            "performance_eligible": performance_eligible,
            "requested_target_bytes": int(case["kv_target_bytes"]),
            "action_target_bytes": int(case["action_target_bytes"]),
            "action_target_bytes": int(case["action_target_bytes"]),
            "resume_measurement": None,
            "post_resume_steady_measurements": [],
            "resident_after_fill": resident_after_fill["resident_bytes"],
            "resident_after_release_settle": settled_resident["resident_bytes"],
            "resident_after_offload_settle": None,
            "resident_settled": settled_resident["resident_bytes"],
            "resident_after_resume": settled_resident["resident_bytes"],
            "release_physical_relief_bytes": release_physical_relief,
            "release_physical_relief_authority": "phase_boundary_slots",
            "offload_physical_relief_bytes": 0,
            "offload_physical_relief_authority": "not_applicable",
            "total_physical_relief_bytes": release_physical_relief,
            "total_physical_relief_authority": "sum_of_authorities",
            "physical_relief_bytes": release_physical_relief,
            "memory_saved_bytes": release_physical_relief,
            "memory_saved_ratio": (
                release_physical_relief / resident_after_fill["resident_bytes"]
                if resident_after_fill["resident_bytes"] else None),
            "budget_debt_after": terminal["budget_debt_after_bytes"],
            "settle_time_seconds": settle["duration_ns"] / 1_000_000_000,
            "unmet_budget_bytes": terminal["unmet_budget_bytes_after"],
            "release": {
                "actions": len(release_actions),
                "blocks": sum(int(item["blocks"]) for item in release_actions),
                "bytes": sum(int(item["bytes"]) for item in release_actions),
                "physical_relieved_bytes": release_physical_relief,
            },
            "offload": {
                "actions": 0, "blocks": 0, "bytes": 0,
                "physical_relieved_bytes": 0, "write_syscalls": int(io["backing_write_syscalls"]),
                "bytes_written": int(io["bytes_written"]),
            },
            "resume": {
                "restored_blocks": 0, "restored_bytes": 0,
                "read_syscalls": int(io["backing_read_syscalls"]),
                "bytes_read": int(io["bytes_read"]), "gate_us": None, "total_us": None,
            },
            "transient_staging_peak_bytes": int(io["k2_peak_staging_bytes"]),
            "transient_staging_bound_bytes": int(io["k2_staging_bound_bytes"]),
            "k2": k2_statistics(io),
        }

    settled_snapshot = validate_slot_snapshot(
        run_dir / "slots_settled.json",
        f"{label}.slots_settled",
        require_resident=False,
    )
    if record["settled"]["captured_mono_ns"] != settled_snapshot["captured_mono_ns"]:
        raise ParseError(f"{label}: settled snapshot reference timestamp mismatch")
    settled_resident = optional_authoritative_slot_resident(
        settled_snapshot, f"{label}.slots_settled")
    release_snapshot = validate_slot_snapshot(
        run_dir / "slots_release_settled.json",
        f"{label}.slots_release_settled",
        require_resident=False,
    )
    if record["release_settled"]["captured_mono_ns"] != release_snapshot["captured_mono_ns"]:
        raise ParseError(f"{label}: release settled snapshot reference timestamp mismatch")
    release_resident = optional_authoritative_slot_resident(
        release_snapshot, f"{label}.slots_release_settled")

    idle = record["idle"]
    settle = record["settle"]
    if idle["started_mono_ns"] < after_fill_snapshot["captured_mono_ns"]:
        raise ParseError(f"{label}: V2 idle began before after-fill resident capture")
    if (
        release_snapshot["captured_mono_ns"] < after_fill_snapshot["captured_mono_ns"]
        or release_snapshot["captured_mono_ns"] > settled_snapshot["captured_mono_ns"]
        or release_snapshot["captured_mono_ns"] > first_measurement["started_mono_ns"]
        or settled_snapshot["captured_mono_ns"] < settle["completed_mono_ns"]
        or settled_snapshot["captured_mono_ns"] > first_measurement["started_mono_ns"]
    ):
        raise ParseError(f"{label}: RELEASE/OFFLOAD resident phases are out of order")
    resume = record["resume"]
    if (
        resume["sequence"] != first_measurement["sequence"]
        or resume["request_id"] != first_measurement["request_id"]
        or resume["started_mono_ns"] != first_measurement["started_mono_ns"]
        or resume["finished_mono_ns"] != first_measurement["finished_mono_ns"]
    ):
        raise ParseError(f"{label}: resume record does not match first post-settle measurement")

    settle_text = stderr_window(
        stderr_data, settle["stderr_start_offset"], settle["stderr_end_offset"],
        f"{label}.settle",
    )
    settle_actions = marker_records(
        settle_text, "kv_pressure_unified_action", ACTION_REQUIRED,
        f"{label}.settle.action",
    )
    for action in settle_actions:
        validate_action_fields(action, f"{label}.settle.action")
        boolean_fields(action, ACTION_BOOL, f"{label}.settle.action")
    resident_target = int(case["kv_target_bytes"])
    action_target = int(case["action_target_bytes"])
    max_blocks = int(spec["max_blocks"])
    tolerance = int(spec["workload"]["characterization"]["target_tolerance_bytes"])
    observations, terminal = characterization_budget_observations(
        settle_actions,
        pressure_basis_source(spec["pressure_basis"]),
        resident_target,
        action_target,
        max_blocks,
        tolerance,
        f"{label}.settle",
    )
    terminal_record = {key: value for key, value in terminal.items() if key not in {"status", "action"}}
    boundary_entry = release_boundary_from_actions(settle_actions)
    if boundary_entry is None:
        raise ParseError(f"{label}: V2 has no valid RELEASE phase boundary")
    boundary_index, boundary_status, boundary_action = boundary_entry
    recorded_boundary = settle["release_boundary"]
    if (
        recorded_boundary["status"] != boundary_status
        or recorded_boundary["decision_id"] != int(boundary_action["decision_id"])
        or recorded_boundary["budget_resident_bytes"] != int(boundary_action["budget_resident_bytes"])
        or recorded_boundary["budget_debt_after_bytes"] != int(boundary_action["budget_debt_after_bytes"])
        or recorded_boundary["unmet_budget_bytes_after"] != int(boundary_action["unmet_budget_bytes_after"])
        or recorded_boundary["state_changed"] != (boundary_action["state_changed"] == "1")
        or recorded_boundary["bytes"] != int(boundary_action["bytes"])
        or recorded_boundary["relieved_bytes"] != int(boundary_action["relieved_bytes"])
        or recorded_boundary["soft_offload_armed_after"] != (boundary_action["soft_offload_armed_after"] == "1")
    ):
        raise ParseError(f"{label}: recorded RELEASE phase boundary differs from production marker")

    slot_resident_snapshots = (
        resident_after_fill, release_resident, settled_resident, resident_after_measurement)
    missing_slot_resident = any(resident is None for resident in slot_resident_snapshots)
    all_slot_resident_missing = all(resident is None for resident in slot_resident_snapshots)
    observed_slot_residents = [
        resident for resident in slot_resident_snapshots if resident is not None
    ]
    if terminal["status"] != "unmet_floor" and missing_slot_resident:
        raise ParseError(
            f"{label}: V2 physical resident snapshots are incomplete outside UNMET_FLOOR")
    if terminal["status"] == "unmet_floor" and missing_slot_resident and not all_slot_resident_missing:
        raise ParseError(f"{label}: V2 physical resident snapshots are partially missing")
    if len(observed_slot_residents) > 1:
        identity_keys = ("object_id", "generation", "page_size", "total_bytes", "total_pages")
        first_resident = observed_slot_residents[0]
        if any(
            first_resident[key] != resident[key]
            for resident in observed_slot_residents[1:]
            for key in identity_keys
        ):
            raise ParseError(f"{label}: physical resident identity changed across characterization")
    if not observations:
        raise ParseError(f"{label}: V2 characterization has no budget resident observations")
    budget_resident_views = {
        "after_fill": {
            "authority": "budget_view_marker",
            "resident_bytes": observations[0]["budget_resident_bytes"],
        },
        "after_release_settle": {
            "authority": "budget_view_marker",
            "resident_bytes": recorded_boundary["budget_resident_bytes"],
        },
        "settled": {
            "authority": "budget_view_marker",
            "resident_bytes": terminal["budget_resident_bytes"],
        },
        "after_resume": None,
    }
    if (
        budget_resident_views["after_release_settle"]["resident_bytes"]
        > budget_resident_views["after_fill"]["resident_bytes"]
        or budget_resident_views["settled"]["resident_bytes"]
        > budget_resident_views["after_release_settle"]["resident_bytes"]
    ):
        raise ParseError(f"{label}: budget-view resident observations are not monotonic")
    physical_resident_available = not missing_slot_resident
    if physical_resident_available:
        assert resident_after_fill is not None
        assert release_resident is not None
        assert settled_resident is not None
        if release_resident["resident_bytes"] > resident_after_fill["resident_bytes"]:
            raise ParseError(f"{label}: RELEASE settle exceeds after-fill B_full observation")
        if settled_resident["resident_bytes"] > release_resident["resident_bytes"]:
            raise ParseError(f"{label}: OFFLOAD settle exceeds RELEASE settle resident")
        tolerance = int(spec["workload"]["characterization"]["target_tolerance_bytes"])
        if abs(
            release_resident["resident_bytes"] - recorded_boundary["budget_resident_bytes"]
        ) > tolerance:
            raise ParseError(
                f"{label}: slots RELEASE snapshot disagrees with marker phase-boundary resident")
    positive_action_indices = [
        index for index, action in enumerate(settle_actions)
        if is_qualifying_offload_action(
            action, pressure_basis_source(spec["pressure_basis"]), allow_shortfall=True)
    ]
    if positive_action_indices and min(positive_action_indices) <= boundary_index:
        raise ParseError(
            f"{label}: first positive OFFLOAD overlaps or precedes the RELEASE boundary")
    if positive_action_indices and boundary_status != "release_no_candidate":
        raise ParseError(
            f"{label}: V2 OFFLOAD lacks the required RELEASE no_candidate boundary")

    expected_settle_status = (
        "target_reached" if terminal["status"] in {"debt_closed", "target_reached"}
        else "unmet_floor")
    if (
        settle["status"] != expected_settle_status
        or settle["decision_ids"] != [item["decision_id"] for item in observations]
        or settle["offload_decision_ids"] != [
            item["decision_id"] for item in observations if item["positive_offload"]]
        or settle["terminal_decision"] != terminal_record
    ):
        raise ParseError(f"{label}: runner settle record differs from production markers")
    terminal_resident = terminal["budget_resident_bytes"]
    if settle["physical_resident_bytes"] is not None:
        if not physical_resident_available or settled_resident is None:
            raise ParseError(f"{label}: runner claims physical resident without physical authority")
        if settle["physical_resident_bytes"] != settled_resident["resident_bytes"]:
            raise ParseError(f"{label}: runner settled resident differs from the physical snapshot")
    if terminal["status"] != "debt_closed" and physical_resident_available:
        assert settled_resident is not None
        if abs(settled_resident["resident_bytes"] - terminal_resident) > tolerance:
            raise ParseError(f"{label}: settled physical resident differs from terminal budget view")
    if terminal["status"] in {"debt_closed", "target_reached"}:
        if settled_resident is None or (
            settled_resident["resident_bytes"] > resident_target + tolerance
            or terminal["budget_debt_after_bytes"] != 0
            or terminal["unmet_budget_bytes_after"] != 0
        ):
            raise ParseError(f"{label}: target was declared reached without actual resident closure")
        status = "TARGET_REACHED"
    else:
        actual_unmet = max(0, terminal_resident - resident_target)
        if (
            actual_unmet <= 0
            or terminal["budget_debt_after_bytes"] != actual_unmet
            or terminal["unmet_budget_bytes_after"] != actual_unmet
        ):
            raise ParseError(f"{label}: unmet floor marker does not preserve actual resident debt")
        status = "UNMET_FLOOR"

    resume_text = stderr_window(
        stderr_data, resume["stderr_start_offset"], resume["stderr_end_offset"],
        f"{label}.resume",
    )
    resume_events = marker_records(
        resume_text, "kv_resume_order_event", RESUME_REQUIRED, f"{label}.resume.event")
    for event in resume_events:
        numeric_fields(
            event, {"decision_id", "seq_id", "claimant_epoch", "transaction_id"},
            f"{label}.resume.event")
        boolean_fields(event, RESUME_BOOL, f"{label}.resume.event")
    resume_timings = marker_records(
        resume_text, "kv_resume_stage_timing", TIMING_REQUIRED, f"{label}.resume.timing")
    for timing in resume_timings:
        numeric_fields(timing, TIMING_REQUIRED, f"{label}.resume.timing")
    positive_timing: dict[str, str] | None = None
    positive_resume_evidence: dict[str, dict[str, str]] | None = None
    positive_offloads = [item for item in observations if item["positive_offload"]]
    release_actions = [
        item["action"] for item in observations if item["action"]["release_attempted"] == "1"
    ]
    offload_pairs = qualified_pairs or []
    all_positive_actions = [
        action for action in marker_records(
            stderr_data.decode("utf-8", errors="replace"),
            "kv_pressure_unified_action", ACTION_REQUIRED, f"{label}.all_action")
        if is_qualifying_offload_action(
            action, pressure_basis_source(spec["pressure_basis"]), allow_shortfall=True)
    ]
    if len(all_positive_actions) != len(positive_offloads):
        raise ParseError(
            f"{label}: positive OFFLOAD exists outside the settled phase window")
    if positive_offloads and len(offload_pairs) != len(positive_offloads):
        raise ParseError(
            f"{label}: every positive OFFLOAD must have a transaction-local mincore resident drop")
    positive_keys = {
        (
            int(item["action"]["decision_id"]),
            int(item["action"]["transaction_id"]),
            int(item["action"]["selected_seq_id"]),
        )
        for item in positive_offloads
    }
    pair_keys = {
        (item["decision_id"], item["transaction_id"], item["seq_id"])
        for item in offload_pairs
    }
    if pair_keys != positive_keys:
        raise ParseError(f"{label}: transaction-local OFFLOAD pairs do not match settled actions")
    if positive_offloads:
        selected_offload_epochs: dict[int, set[int]] = {}
        for item in positive_offloads:
            action = item["action"]
            selected_offload_epochs.setdefault(
                int(action["selected_seq_id"]), set()).add(
                    int(action["selected_claimant_epoch"]))
        positive_resume_evidence = validate_resume_restore_evidence(
            resume_events,
            resume_timings,
            selected_offload_epochs,
            f"{label}.resume",
        )
        positive_timing = positive_resume_evidence["timing"]
    if terminal["status"] == "unmet_floor" and not positive_offloads:
        raise ParseError(f"{label}: UNMET_FLOOR requires positive OFFLOAD evidence")
    if terminal["status"] == "unmet_floor" and positive_timing is None:
        raise ParseError(f"{label}: UNMET_FLOOR requires positive resume restore evidence")
    if not release_actions and positive_offloads:
        raise ParseError(f"{label}: V2 OFFLOAD was not preceded by a Unified RELEASE decision")
    offload_physical_relief = sum(item["resident_drop_bytes"] for item in offload_pairs)
    if physical_resident_available:
        assert resident_after_fill is not None
        assert release_resident is not None
        assert settled_resident is not None
        release_physical_relief = (
            resident_after_fill["resident_bytes"] - release_resident["resident_bytes"])
        settled_physical_relief = (
            resident_after_fill["resident_bytes"] - settled_resident["resident_bytes"])
        total_physical_relief = release_physical_relief + offload_physical_relief
        if total_physical_relief != settled_physical_relief:
            raise ParseError(
                f"{label}: RELEASE plus OFFLOAD physical relief violates resident conservation")
    else:
        release_physical_relief = None
        settled_physical_relief = None
        total_physical_relief = None
    resume_measurement = {
        "sequence": first_measurement["sequence"],
        "request_id": first_measurement["request_id"],
        "repeat_index": first_measurement["repeat_index"],
        "measurement_phase": first_measurement["measurement_phase"],
    }
    post_resume_steady_measurements = [
        {
            "sequence": item["sequence"],
            "request_id": item["request_id"],
            "repeat_index": item["repeat_index"],
            "measurement_phase": item["measurement_phase"],
        }
        for item in measurements[1:]
    ]

    return {
        "status": status,
        "performance_eligible": True,
        "release_terminal": terminal["status"],
        "requested_target_bytes": resident_target,
        "action_target_bytes": action_target,
        "resume_measurement": resume_measurement,
        "post_resume_steady_measurements": post_resume_steady_measurements,
        "resident_after_fill": (
            resident_after_fill["resident_bytes"] if physical_resident_available else None),
        "resident_after_fill_authority": (
            "slots_physical" if physical_resident_available else "budget_view_marker"),
        "resident_after_release_settle": (
            release_resident["resident_bytes"] if physical_resident_available else None),
        "resident_after_release_settle_authority": (
            "slots_physical" if physical_resident_available else "budget_view_marker"),
        "resident_after_offload_settle": (
            settled_resident["resident_bytes"]
            if physical_resident_available and positive_offloads else None),
        "resident_after_offload_settle_authority": (
            "slots_physical" if physical_resident_available and positive_offloads
            else "budget_view_marker" if positive_offloads else "unavailable"),
        "resident_settled": (
            settled_resident["resident_bytes"] if physical_resident_available else None),
        "resident_settled_authority": (
            "slots_physical" if physical_resident_available else "budget_view_marker"),
        "resident_after_resume": (
            resident_after_measurement["resident_bytes"]
            if resident_after_measurement is not None else None),
        "resident_after_resume_authority": (
            "slots_physical" if resident_after_measurement is not None else "unavailable"),
        "resident_views": {
            "after_fill": (
                {"authority": "slots_physical", "resident_bytes": resident_after_fill["resident_bytes"]}
                if physical_resident_available else None),
            "after_release_settle": (
                {"authority": "slots_physical", "resident_bytes": release_resident["resident_bytes"]}
                if physical_resident_available else None),
            "settled": (
                {"authority": "slots_physical", "resident_bytes": settled_resident["resident_bytes"]}
                if physical_resident_available else None),
            "after_resume": (
                {"authority": "slots_physical", "resident_bytes": resident_after_measurement["resident_bytes"]}
                if resident_after_measurement is not None else None),
        },
        "budget_resident_views": budget_resident_views,
        "release_budget_view_relief_bytes": (
            budget_resident_views["after_fill"]["resident_bytes"]
            - budget_resident_views["after_release_settle"]["resident_bytes"]),
        "settled_budget_view_relief_bytes": (
            budget_resident_views["after_fill"]["resident_bytes"]
            - budget_resident_views["settled"]["resident_bytes"]),
        "budget_view_relief_authority": "budget_view_marker",
        "release_physical_relief_bytes": release_physical_relief,
        "release_physical_relief_authority": (
            "phase_boundary_slots" if physical_resident_available
            else "unavailable_no_independent_physical_authority"),
        "offload_physical_relief_bytes": offload_physical_relief,
        "offload_physical_relief_authority": "transaction_local_mincore",
        "total_physical_relief_bytes": total_physical_relief,
        "total_physical_relief_authority": (
            "sum_of_independent_physical_authorities"
            if physical_resident_available else "unavailable_no_independent_physical_authority"),
        "physical_relief_bytes": total_physical_relief,
        "memory_saved_bytes": total_physical_relief,
        "memory_saved_ratio": (
            total_physical_relief / resident_after_fill["resident_bytes"]
            if physical_resident_available and resident_after_fill["resident_bytes"] else None),
        "budget_debt_after": terminal["budget_debt_after_bytes"],
        "settle_time_seconds": settle["duration_ns"] / 1_000_000_000,
        "unmet_budget_bytes": terminal["unmet_budget_bytes_after"],
        "release": {
            "actions": len(release_actions),
            "blocks": sum(int(item["blocks"]) for item in release_actions),
            "bytes": sum(int(item["bytes"]) for item in release_actions),
            "physical_relieved_bytes": release_physical_relief,
        },
        "offload": {
            "actions": len(positive_offloads),
            "blocks": sum(int(item["action"]["blocks"]) for item in positive_offloads),
            "bytes": sum(int(item["action"]["bytes"]) for item in positive_offloads),
            "physical_relieved_bytes": offload_physical_relief,
            "action_relieved_bytes": sum(
                int(item["action"]["relieved_bytes"]) for item in positive_offloads),
            "write_syscalls": int(io["backing_write_syscalls"]),
            "bytes_written": int(io["bytes_written"]),
        },
        "resume": {
            "restored_blocks": int(positive_timing["restored_blocks"]) if positive_timing else 0,
            "restored_bytes": int(positive_timing["restored_bytes"]) if positive_timing else 0,
            "read_syscalls": int(io["backing_read_syscalls"]),
            "bytes_read": int(io["bytes_read"]),
            "gate_us": int(positive_timing["gate_us"]) if positive_timing else None,
            "total_us": int(positive_timing["total_us"]) if positive_timing else None,
        },
        "transient_staging_peak_bytes": int(io["k2_peak_staging_bytes"]),
        "transient_staging_bound_bytes": max(
            [int(io["k2_staging_bound_bytes"])]
            + [item["budget_transient_staging_bound_bytes"] for item in observations]),
        "k2": k2_statistics(io),
    }


def parse_run(artifact: pathlib.Path, plan: dict[str, Any], case: dict[str, Any], workload: dict[str, Any], expected_execution_index: int, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = artifact / "runs" / plan["run_id"]
    label = f"{plan['run_id']}"
    if not run_dir.is_dir():
        raise ParseError(f"{label}: run directory missing")
    run = exact(read_json(run_dir / "run.json"), RUN_KEYS, f"{label}.run")
    for key in ("run_id", "round", "run_order", "case_id"):
        if run[key] != plan[key]:
            raise ParseError(f"{label}: run identity mismatch at {key}")
    if run["execution_index"] != expected_execution_index:
        raise ParseError(f"{label}: declared execution index mismatch")
    if run["case"] != plan:
        raise ParseError(f"{label}: embedded case differs from planned factor")
    request_plan = run["request_plan"]
    expected_plan = expanded_request_plan(workload, case, spec["run_mode"])
    if request_plan != expected_plan:
        raise ParseError(f"{label}: request plan differs from manifest workload")
    execution = exact(read_json(run_dir / "execution.json"), EXECUTION_KEYS, f"{label}.execution")
    for key in ("run_id", "round", "run_order", "case_id"):
        if execution[key] != plan[key]:
            raise ParseError(f"{label}: execution identity mismatch at {key}")
    if execution["run_mode"] != spec["run_mode"]:
        raise ParseError(f"{label}: execution run mode differs from manifest spec")
    if execution["pressure_basis"] != spec["pressure_basis"]:
        raise ParseError(f"{label}: execution pressure basis differs from manifest spec")
    if isinstance(execution["execution_index"], bool) or not isinstance(execution["execution_index"], int) or execution["execution_index"] != expected_execution_index:
        raise ParseError(f"{label}: actual execution order does not match declaration")
    if execution["request_loop_started"] is not True or isinstance(execution["request_count"], bool) or not isinstance(execution["request_count"], int) or execution["request_count"] <= 0:
        raise ParseError(f"{label}: request loop/measurement execution is missing")
    validate_execution_environment(execution, case, label, spec)
    swap_dir = pathlib.Path(execution["environment"]["LLAMA_KV_SWAP_DIR"]).resolve()
    if swap_dir != (run_dir / "backing").resolve():
        raise ParseError(f"{label}: canonical backing directory does not match run artifact")
    argv = execution["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise ParseError(f"{label}: server argv is invalid")
    if argv[0] != spec["binary"]:
        raise ParseError(f"{label}: server argv binary mismatch")
    if any(item == "-m" or item.startswith(("--model=", "--host=", "--port=")) for item in argv[1:]):
        raise ParseError(f"{label}: argv contains an attached canonical override")
    for option, expected in (("--host", "127.0.0.1"), ("--model", spec["model"])):
        positions = [index for index, item in enumerate(argv) if item == option]
        if len(positions) != 1 or positions[0] + 1 >= len(argv) or argv[positions[0] + 1] != expected:
            raise ParseError(f"{label}: canonical argv field {option} mismatch")
    port_positions = [index for index, item in enumerate(argv) if item == "--port"]
    if len(port_positions) != 1 or port_positions[0] + 1 >= len(argv) or not UINT.fullmatch(argv[port_positions[0] + 1]) or int(argv[port_positions[0] + 1]) <= 0:
        raise ParseError(f"{label}: canonical port argv field is invalid")
    expected_argv = [
        spec["binary"], "--host", "127.0.0.1", "--port", argv[port_positions[0] + 1],
        "--model", spec["model"], *spec["server_args"],
    ]
    if argv != expected_argv:
        raise ParseError(f"{label}: server argv does not match spec")
    sampler_argv = execution["sampler_argv"]
    if not isinstance(sampler_argv, list) or not sampler_argv or any(not isinstance(item, str) or not item for item in sampler_argv):
        raise ParseError(f"{label}: sampler argv is invalid")
    if len(sampler_argv) != 8 or sampler_argv[:3] != ["bash", str(ROOT / "scripts/kv-controlled-memory-sampler.sh"), "--sample-process"]:
        raise ParseError(f"{label}: sampler argv prefix is invalid")
    server_identity = execution["server_identity"]
    validate_process_identity(server_identity, argv, f"{label}.server_identity")
    if sampler_argv[3] != str(server_identity["pid"]):
        raise ParseError(f"{label}: sampler argv is not bound to server pid")
    if pathlib.Path(sampler_argv[4]).resolve() != (run_dir / "memory_samples.tsv").resolve() or pathlib.Path(sampler_argv[5]).resolve() != (run_dir / "backing").resolve():
        raise ParseError(f"{label}: sampler output/backing argv mismatch")
    if sampler_argv[6] != str(spec["sampler"]["interval_seconds"]):
        raise ParseError(f"{label}: sampler interval argv mismatch")
    sampler_identity = execution["sampler_identity"]
    validate_process_identity(sampler_identity, sampler_argv, f"{label}.sampler_identity")
    if execution["sampler_schema"] != "v2":
        raise ParseError(f"{label}: sampler schema is not v2")
    if not isinstance(execution["server_cgroup"], dict):
        raise ParseError(f"{label}: server cgroup record missing")
    server_cgroup = execution["server_cgroup"]
    if server_cgroup.get("scope") not in {"none", "shared", "dedicated"}:
        raise ParseError(f"{label}: cgroup scope is invalid")
    if server_cgroup.get("scope") == "server_only":
        raise ParseError(f"{label}: cgroup current cannot be described as server-only")
    expected_memory_max = spec["cgroup"]["expected_memory_max"]
    if expected_memory_max is not None and server_cgroup.get("memory_max") != expected_memory_max:
        raise ParseError(f"{label}: cgroup memory.max does not match expected value")
    if spec["pressure_basis"]["authority"] == "cgroup_finite":
        if not isinstance(expected_memory_max, str) or not UINT.fullmatch(expected_memory_max) or int(expected_memory_max) <= 0:
            raise ParseError(f"{label}: finite cgroup pressure authority is not declared")
        if not isinstance(server_cgroup.get("memory_max"), str) or not UINT.fullmatch(server_cgroup["memory_max"]) or int(server_cgroup["memory_max"]) <= 0:
            raise ParseError(f"{label}: finite cgroup pressure authority is unavailable")
    if sampler_argv[7] != str(server_cgroup.get("memory_current_file") or ""):
        raise ParseError(f"{label}: sampler cgroup path does not match server cgroup")
    qualification = validate_qualification_record(
        execution["qualification"],
        workload["qualification"] if spec["run_mode"] == "qualification" else None,
        case,
        f"{label}.qualification",
    )
    characterization = validate_characterization_record(
        execution["characterization"],
        workload["characterization"] if spec["run_mode"] == "characterization" else None,
        case,
        spec["max_blocks"],
        f"{label}.characterization",
    )
    allow_missing_v2_run_resident = (
        spec["run_mode"] == "characterization"
        and case["policy"] == "v2"
        and characterization["settle"] is not None
        and characterization["settle"]["status"] == "unmet_floor"
    )
    slots_before = validate_slot_snapshot(
        run_dir / "slots_before.json",
        f"{label}.slots_before",
        require_resident=not allow_missing_v2_run_resident,
    )
    slots_after = validate_slot_snapshot(
        run_dir / "slots_after.json",
        f"{label}.slots_after",
        require_resident=not allow_missing_v2_run_resident,
    )
    if allow_missing_v2_run_resident:
        slots_before_resident = optional_authoritative_slot_resident(
            slots_before, f"{label}.slots_before")
        slots_after_resident = optional_authoritative_slot_resident(
            slots_after, f"{label}.slots_after")
        if (slots_before_resident is None) != (slots_after_resident is None):
            raise ParseError(f"{label}: V2 run-level resident snapshots are partially missing")
    cleanup = exact(read_json(run_dir / "cleanup.json"), CLEANUP_KEYS, f"{label}.cleanup")
    server_cleanup = validate_cleanup_record(cleanup["server"], f"{label}.cleanup.server")
    sampler_cleanup = validate_cleanup_record(cleanup["sampler"], f"{label}.cleanup.sampler")
    if not isinstance(cleanup["residual_process"], bool) or not isinstance(cleanup["cleanup_complete"], bool):
        raise ParseError(f"{label}: cleanup booleans are invalid")
    nested_residual = bool(server_cleanup["residual_process"] or sampler_cleanup["residual_process"])
    if cleanup["residual_process"] != nested_residual:
        raise ParseError(f"{label}: cleanup residual summary disagrees with PGID checks")
    if nested_residual or not cleanup["cleanup_complete"]:
        raise ParseError(f"{label}: cleanup is incomplete or has residual process")
    if server_cleanup["exit_code"] != 0 or sampler_cleanup["exit_code"] != 0:
        raise ParseError(f"{label}: process exit code is non-zero")
    for filename in ("server.stdout", "server.stderr", "sampler.stdout", "sampler.stderr"):
        if not (run_dir / filename).is_file():
            raise ParseError(f"{label}: missing {filename}")
    samples = validate_sampler(run_dir / "memory_samples.tsv", server_identity, label)
    records, service_failures = validate_responses(
        run_dir / "responses.jsonl", run_dir / "raw", request_plan, label)
    if execution["request_count"] != len(records):
        raise ParseError(f"{label}: execution request_count differs from responses.jsonl")
    stderr_data = (run_dir / "server.stderr").read_bytes()
    stderr = stderr_data.decode("utf-8", errors="replace")
    actions = marker_records(stderr, "kv_pressure_unified_action", ACTION_REQUIRED, f"{label}.action")
    for action in actions:
        validate_action_fields(action, f"{label}.action")
        boolean_fields(action, ACTION_BOOL, f"{label}.action")
    validate_action_target_markers(actions, case, spec, f"{label}.action")
    resident_observations = marker_records(
        stderr,
        "kv_g0_s1_resident_observation",
        RESIDENT_OBSERVATION_REQUIRED,
        f"{label}.resident",
    )
    for observation in resident_observations:
        numeric_fields(observation, RESIDENT_OBSERVATION_NUMERIC, f"{label}.resident")
        boolean_fields(observation, RESIDENT_OBSERVATION_BOOL, f"{label}.resident")
    decision_ids = [item["decision_id"] for item in actions]
    if len(decision_ids) != len(set(decision_ids)):
        raise ParseError(f"{label}: duplicate unified-action decision_id")
    resumes = marker_records(stderr, "kv_resume_order_event", RESUME_REQUIRED, f"{label}.resume")
    for resume in resumes:
        numeric_fields(resume, {"decision_id", "seq_id", "claimant_epoch", "transaction_id"}, f"{label}.resume")
        boolean_fields(resume, RESUME_BOOL, f"{label}.resume")
    timings = marker_records(stderr, "kv_resume_stage_timing", TIMING_REQUIRED, f"{label}.timing")
    for timing in timings:
        numeric_fields(timing, TIMING_REQUIRED, f"{label}.timing")
    io_records = marker_records(stderr, "KV_PAGED_IO_STATS", IO_REQUIRED, f"{label}.io")
    for io in io_records:
        numeric_fields(io, IO_REQUIRED, f"{label}.io")
        boolean_fields(io, IO_BOOL, f"{label}.io")
    if not io_records:
        raise ParseError(f"{label}: missing mandatory KV_PAGED_IO_STATS marker")
    io_last = io_records[-1]
    offload_actions = [action for action in actions if action["offload_attempted"] == "1"]
    expected_source = pressure_basis_source(spec["pressure_basis"])
    qualified_pairs = qualified_offload_pairs(
        actions,
        resident_observations,
        expected_source,
        allow_shortfall=spec["run_mode"] == "characterization",
    )
    qualification_round_trip: dict[str, Any] | None = None
    characterization_metrics: dict[str, Any] | None = None
    if spec["run_mode"] == "qualification":
        qualification_round_trip = validate_qualification_causality(
            qualification, records, stderr_data, case, spec, label)
    else:
        characterization_metrics = validate_characterization_causality(
            characterization, records, stderr_data, case, spec, run_dir, io_last, label,
            qualified_pairs)
        measurement_records = [record for record in records if record["measurement"]]
        if case["policy"] == "v2":
            characterization_metrics["performance"] = response_statistics(
                [record for record in measurement_records if record["measurement_phase"] == "resume"])
            characterization_metrics["post_resume_steady_performance"] = response_statistics(
                [record for record in measurement_records if record["measurement_phase"] == "post_resume_steady"])
        elif case["policy"] == "release_only":
            characterization_metrics["performance"] = response_statistics(
                [record for record in measurement_records if record["measurement_phase"] == "release_only_steady"]
                if characterization_metrics["performance_eligible"] else [])
            characterization_metrics["post_resume_steady_performance"] = response_statistics([])
        else:
            characterization_metrics["performance"] = response_statistics(measurement_records)
            characterization_metrics["post_resume_steady_performance"] = response_statistics([])
    if case["policy"] == "v2":
        if not actions:
            raise ParseError(f"{label}: missing mandatory kv_pressure_unified_action marker")
        if spec["run_mode"] == "qualification":
            if not offload_actions:
                raise ParseError(f"{label}: V2 qualification has no offload_attempted=1 action")
            if not qualified_pairs:
                raise ParseError(
                    f"{label}: V2 qualification has no matching successful OFFLOAD and transaction-local resident drop")
            if int(io_last["block_swap_out_calls"]) <= 0 or int(io_last["bytes_written"]) <= 0:
                raise ParseError(f"{label}: V2 qualification IO evidence has no swap-out/write")
            if (
                int(io_last["block_swap_in_calls"]) <= 0
                or int(io_last["backing_read_syscalls"]) <= 0
                or int(io_last["bytes_read"]) <= 0
            ):
                raise ParseError(f"{label}: V2 qualification IO evidence has no swap-in/read")
            if qualification_round_trip is None:
                raise ParseError(f"{label}: V2 qualification round-trip evidence is missing")
        else:
            if characterization_metrics is None:
                raise ParseError(f"{label}: V2 characterization metrics are missing")
            positive_offloads = characterization_metrics["offload"]["actions"]
            if positive_offloads > 0 and (
                int(io_last["block_swap_out_calls"]) <= 0
                or int(io_last["backing_write_syscalls"]) <= 0
                or int(io_last["bytes_written"]) <= 0
            ):
                raise ParseError(f"{label}: V2 characterization IO has no swap-out/write")
            if (
                positive_offloads > 0
                or characterization_metrics["status"] == "UNMET_FLOOR"
            ) and (
                int(io_last["block_swap_in_calls"]) <= 0
                or int(io_last["backing_read_syscalls"]) <= 0
                or int(io_last["bytes_read"]) <= 0
                or characterization_metrics["resume"]["restored_bytes"] <= 0
            ):
                raise ParseError(f"{label}: V2 characterization IO has no linked swap-in/read")
    elif case["policy"] == "release_only":
        if not actions:
            raise ParseError(f"{label}: RELEASE-only is missing the Unified RELEASE marker")
        release_actions = [action for action in actions if action["release_attempted"] == "1"]
        if not release_actions:
            raise ParseError(f"{label}: RELEASE-only did not execute or close a Unified RELEASE path")
        if any(action["offload_attempted"] != "0" for action in actions):
            raise ParseError(f"{label}: RELEASE-only contains OFFLOAD action evidence")
        migration_fields = (
            "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls",
            "backing_write_syscalls", "bytes_read", "bytes_written",
        )
        if any(
            int(io[key]) != 0
            for io in io_records
            for key in migration_fields
        ):
            raise ParseError(f"{label}: RELEASE-only contains swap or backing IO evidence")
        if any(
            int(io[key]) > 0
            for io in io_records
            for key in RESTORE_ACTIVITY_FIELDS
        ):
            raise ParseError(f"{label}: RELEASE-only contains positive restore activity")
        validate_release_only_noop_restore(resumes, timings, f"{label}.resume")
    else:
        if any(action["offload_attempted"] != "0" or action["release_attempted"] != "0" for action in actions):
            raise ParseError(f"{label}: resident case contains state-changing action evidence")
        resident_migration_fields = (
            "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls",
            "backing_write_syscalls", "bytes_read", "bytes_written",
        )
        if any(int(io_last[key]) != 0 for key in resident_migration_fields):
            raise ParseError(f"{label}: resident case contains migration IO evidence")
    if not (case["policy"] == "v2" and spec["run_mode"] == "qualification") \
            and int(io_last["block_swap_in_calls"]) > 0:
        if not resumes or not timings:
            raise ParseError(f"{label}: swap-in requires resume and timing markers")
        completed_resumes = [
            resume for resume in resumes
            if resume["outcome"] == "completed" and resume["graph_allowed"] == "1"]
        if not completed_resumes:
            raise ParseError(f"{label}: swap-in has no successful graph-gated resume")
        resume_ids = {resume["decision_id"] for resume in completed_resumes}
        if not any(
                timing["decision_id"] in resume_ids
                and int(timing["restored_bytes"]) > 0
                and int(timing["total_us"]) > 0
                for timing in timings):
            raise ParseError(f"{label}: swap-in has no corresponding positive resume timing")
    return {
        "run_id": plan["run_id"],
        "case_id": plan["case_id"],
        "round": plan["round"],
        "policy": plan["policy"],
        "server_exit_code": server_cleanup["exit_code"],
        "sampler_exit_code": sampler_cleanup["exit_code"],
        "samples": samples,
        "memory": memory_statistics(samples),
        "slots_before": slots_before,
        "slots_after": slots_after,
        "responses": records,
        "service_failures": service_failures,
        "actions": actions,
        "resident_observations": resident_observations,
        "qualified_offload_pairs": qualified_pairs,
        "qualification_round_trip": qualification_round_trip,
        "characterization": characterization_metrics,
        "resume_events": resumes,
        "resume_timings": timings,
        "io": io_last,
        "statistics": (
            characterization_metrics["performance"]
            if characterization_metrics is not None else response_statistics(records)
        ),
    }


def validate_formal_pressure_authority(
        spec: dict[str, Any], plan: list[dict[str, Any]], runner_status: str,
) -> None:
    if (
        runner_status != "UNSUPPORTED"
        and spec["run_kind"] == "formal"
        and any(item["policy"] in BUDGET_POLICIES for item in plan)
        and spec["pressure_basis"]["authority"] != "cgroup_finite"
    ):
        raise ParseError("formal budget run requires real finite cgroup pressure authority")


def validate_manifest(artifact: pathlib.Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest = exact(read_json(artifact / "manifest.json"), MANIFEST_REQUIRED, "manifest", MANIFEST_OPTIONAL)
    reject_runner_verdict(manifest)
    if manifest["schema_version"] != SCHEMA_VERSION or manifest["protocol"] != PROTOCOL:
        raise ParseError("manifest protocol/schema mismatch")
    if not isinstance(manifest["artifact_id"], str) or not manifest["artifact_id"]:
        raise ParseError("manifest artifact_id is invalid")
    framework = exact(manifest["framework"], {"runner", "parser", "memory_sampler"}, "manifest.framework")
    for key, identity in framework.items():
        verify_identity(identity, f"manifest.framework.{key}")
    parser_identity = framework["parser"]
    if pathlib.Path(parser_identity["path"]).resolve() != SCRIPT.resolve():
        raise ParseError("manifest parser identity does not point to canonical parser")
    provenance = exact(manifest["provenance"], {"git", "host", "binary", "model", "model_quantization", "runner_pid", "runner_cgroup"}, "manifest.provenance")
    verify_identity(provenance["binary"], "manifest.provenance.binary", allow_missing=True)
    verify_identity(provenance["model"], "manifest.provenance.model", allow_missing=True)
    if not isinstance(provenance["model_quantization"], str) or not provenance["model_quantization"]:
        raise ParseError("model_quantization is missing")
    if not isinstance(provenance["git"], dict) or not isinstance(provenance["host"], dict) or not isinstance(provenance["runner_cgroup"], dict):
        raise ParseError("manifest provenance subrecord is malformed")
    git = exact(provenance["git"], {"head", "branch", "dirty_status", "diff_sha256", "capture_mode"}, "manifest.provenance.git")
    if not isinstance(git["dirty_status"], list) or any(not isinstance(item, str) or not item for item in git["dirty_status"]):
        raise ParseError("manifest.provenance.git dirty_status is malformed")
    if git["capture_mode"] not in {"archival_clean", "diagnostic_dirty"}:
        raise ParseError("manifest.provenance.git capture_mode is invalid")
    cases, plan, workload = validate_spec(manifest["spec"])
    if manifest["spec"]["run_kind"] == "formal" and (git["capture_mode"] != "archival_clean" or git["dirty_status"]):
        raise ParseError("formal artifact is not from a clean worktree")
    for item in plan:
        if (
            item["policy"] not in SUPPORTED_POLICIES
            or item["kv_representation"] not in SUPPORTED_KV_REPRESENTATIONS
            or item["loading_mode"] not in SUPPORTED_LOADING_MODES
            or item["restore"] not in SUPPORTED_RESTORES
            or item["prefault"] not in SUPPORTED_PREFAULTS
        ):
            if manifest["runner_status"] != "UNSUPPORTED":
                raise ParseError(f"unsupported factor reached workload: {item}")
    planned = manifest["planned_runs"]
    if not isinstance(planned, list) or planned != plan:
        raise ParseError("manifest planned_runs differs from parser-owned spec plan")
    results = manifest["run_results"]
    if not isinstance(results, list):
        raise ParseError("manifest run_results must be an array")
    for index, result in enumerate(results):
        exact(result, {"run_id", "case_id", "round", "run_order", "execution_index", "directory", "status", "error"}, f"manifest.run_results[{index}]")
        if index < len(plan):
            expected = plan[index]
            for key in ("run_id", "case_id", "round", "run_order"):
                if result[key] != expected[key]:
                    raise ParseError(f"manifest.run_results[{index}] disagrees with declared plan at {key}")
            if result["execution_index"] != index:
                raise ParseError(f"manifest.run_results[{index}] execution order is not sequential")
        if result["status"] not in {"complete", "incomplete"} or (result["error"] is not None and not isinstance(result["error"], str)):
            raise ParseError(f"manifest.run_results[{index}] status/error is malformed")
    status = manifest["runner_status"]
    validate_formal_pressure_authority(manifest["spec"], plan, status)
    if status not in {"run_in_progress", "run_complete", "run_incomplete", "UNSUPPORTED", "DRY_RUN"}:
        raise ParseError(f"unknown runner_status: {status}")
    if status == "UNSUPPORTED":
        unsupported = exact(manifest["unsupported"], {"stage", "reason", "workload_started"}, "manifest.unsupported")
        if unsupported["workload_started"] or not str(unsupported["stage"]).startswith("pre_workload_"):
            raise ParseError("UNSUPPORTED is not fail-closed before workload")
        if results or any((artifact / "runs").iterdir()):
            raise ParseError("UNSUPPORTED artifact contains workload evidence")
    elif status == "DRY_RUN":
        if manifest["unsupported"] is not None or results or any((artifact / "runs").iterdir()):
            raise ParseError("DRY_RUN artifact contains workload evidence")
    else:
        if manifest["unsupported"] is not None:
            raise ParseError("completed artifact contains unsupported record")
        if status != "run_complete":
            raise ParseError(f"runner did not complete: {status}")
        if len(results) != len(plan):
            raise ParseError("manifest run_results count differs from plan")
        expected_ids = [item["run_id"] for item in plan]
        actual_ids = [item["run_id"] for item in results]
        if actual_ids != expected_ids:
            raise ParseError("manifest run_results order differs from planned order")
        run_root = artifact / "runs"
        actual_dirs = sorted(item.name for item in run_root.iterdir() if item.is_dir())
        if actual_dirs != sorted(expected_ids):
            raise ParseError(f"run directories differ from plan: actual={actual_dirs} expected={sorted(expected_ids)}")
    return manifest, cases, plan, workload


def write_verdict(artifact: pathlib.Path, status: str, result: dict[str, Any]) -> None:
    parser_record = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "status": status,
        "parser": {
            "path": str(SCRIPT), "present": True, "size": SCRIPT.stat().st_size, "sha256": sha256_file(SCRIPT),
        },
        "manifest_sha256": sha256_file(artifact / "manifest.json"),
        "result_file": "result.json",
    }
    (artifact / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact / "parser.json").write_text(json.dumps(parser_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def slot_resident_values(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    value = snapshot.get("body_json")
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for slot in value:
        if not isinstance(slot, dict):
            continue
        resident = slot.get("kv_resident")
        if isinstance(resident, dict):
            result.append(resident)
    return result


def performance_delta(
        candidate: dict[str, Any],
        baseline: dict[str, Any],
        candidate_phase: str,
) -> dict[str, Any]:
    metrics = {
        "ttft_ms": "ms",
        "tpot_ms_per_token": "ms/token",
        "e2e_ms": "ms",
        "throughput_tokens_per_second": "tokens/s",
    }
    result: dict[str, Any] = {
        "status": "UNAVAILABLE",
        "candidate_phase": candidate_phase,
        "baseline_phase": "resident_steady",
        "metrics": {},
        "reason": "no corresponding available measurement metric",
    }
    for name, unit in metrics.items():
        candidate_metric = candidate.get(name, {})
        baseline_metric = baseline.get(name, {})
        if (
            candidate_metric.get("status") == "AVAILABLE"
            and baseline_metric.get("status") == "AVAILABLE"
            and candidate_metric.get("p50") is not None
            and baseline_metric.get("p50") is not None
        ):
            result["metrics"][name] = {
                "unit": unit,
                "candidate_p50": candidate_metric["p50"],
                "baseline_p50": baseline_metric["p50"],
                "delta_p50": candidate_metric["p50"] - baseline_metric["p50"],
            }
    if result["metrics"]:
        result["status"] = "AVAILABLE"
        result.pop("reason")
    return result


def characterization_comparisons(
        runs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    resident_by_round: dict[int, list[dict[str, Any]]] = {}
    for item in runs:
        if item["status"] == "RESIDENT_BASELINE":
            resident_by_round.setdefault(item.get("round", 0), []).append(item)
    comparisons: list[dict[str, Any]] = []
    for item in runs:
        if item["status"] == "RESIDENT_BASELINE":
            continue
        policy = item.get("policy")
        if policy not in BUDGET_POLICIES:
            continue
        round_id = item.get("round", 0)
        baselines = resident_by_round.get(round_id, [])
        comparison: dict[str, Any] = {
            "round": round_id,
            "candidate_run_id": item["run_id"],
            "policy": policy,
            "baseline_run_id": None,
            "memory_saved_bytes": None,
            "memory_saved_ratio": None,
            "performance_delta": {
                "status": "UNAVAILABLE",
                "candidate_phase": (
                    "release_only_steady" if policy == "release_only"
                    else "post_resume_steady"),
                "baseline_phase": "resident_steady",
                "metrics": {},
                "reason": "no unique Resident baseline in the same round",
            },
        }
        if len(baselines) == 1:
            baseline = baselines[0]
            comparison["baseline_run_id"] = baseline["run_id"]
            physical_candidate = (
                (policy == "release_only" or item.get("resident_settled_authority") == "slots_physical")
                and baseline.get("resident_after_fill") is not None
                and item.get("resident_settled") is not None
            )
            if physical_candidate:
                saved = baseline["resident_after_fill"] - item["resident_settled"]
                comparison["memory_saved_bytes"] = saved
                comparison["memory_saved_ratio"] = (
                    saved / baseline["resident_after_fill"]
                    if baseline["resident_after_fill"] else None)
            candidate_performance = (
                item.get("performance", {}) if policy == "release_only"
                else item.get("post_resume_steady_performance", {}))
            comparison["performance_delta"] = performance_delta(
                candidate_performance,
                baseline.get("performance", {}),
                "release_only_steady" if policy == "release_only" else "post_resume_steady",
            )
        comparisons.append(comparison)

    aggregates: dict[str, Any] = {}
    for policy in ("release_only", "v2"):
        selected = [item for item in comparisons if item["policy"] == policy]
        memory_values = [item["memory_saved_bytes"] for item in selected if item["memory_saved_bytes"] is not None]
        ratio_values = [item["memory_saved_ratio"] for item in selected if item["memory_saved_ratio"] is not None]
        metric_values: dict[str, list[float]] = {}
        for item in selected:
            for name, value in item["performance_delta"].get("metrics", {}).items():
                metric_values.setdefault(name, []).append(float(value["delta_p50"]))
        aggregates[policy] = {
            "n": len(selected),
            "memory_saved_bytes": {
                "status": "AVAILABLE" if memory_values else "UNAVAILABLE",
                "min": min(memory_values) if memory_values else None,
                "max": max(memory_values) if memory_values else None,
                "p50": percentile([float(value) for value in memory_values], 0.50),
            },
            "memory_saved_ratio": {
                "status": "AVAILABLE" if ratio_values else "UNAVAILABLE",
                "min": min(ratio_values) if ratio_values else None,
                "max": max(ratio_values) if ratio_values else None,
                "p50": percentile([float(value) for value in ratio_values], 0.50),
            },
            "performance_delta_p50": {
                name: {
                    "status": "AVAILABLE",
                    "min": min(values),
                    "max": max(values),
                    "p50": percentile(values, 0.50),
                }
                for name, values in metric_values.items()
            },
        }
    return comparisons, aggregates


def summarize_characterization(
        run_results: list[dict[str, Any]], run_kind: str | None = None,
) -> dict[str, Any]:
    runs = [
        {
            "run_id": item["run_id"],
            "case_id": item["case_id"],
            "round": item.get("round"),
            "policy": item.get("policy"),
            **item["characterization"],
        }
        for item in run_results if item["characterization"] is not None
    ]
    resident_runs = [item for item in runs if item["status"] == "RESIDENT_BASELINE"]
    release_runs = [item for item in runs if item.get("policy") == "release_only"]
    release_floor_runs = [
        item for item in release_runs if item.get("release_terminal") == "release_no_candidate"
    ]
    release_target_runs = [
        item for item in release_runs if item.get("release_terminal") == "release_settled"
    ]
    v2_runs = [item for item in runs if item["status"] in {"TARGET_REACHED", "UNMET_FLOOR"}]
    full_values = [item["resident_after_fill"] for item in resident_runs]
    if full_values:
        b_full = {
            "status": "AVAILABLE",
            "definition": "same-workload Resident after-fill physical resident",
            "observations": [
                {"run_id": item["run_id"], "bytes": item["resident_after_fill"]}
                for item in resident_runs
            ],
            "min_bytes": min(full_values),
            "max_bytes": max(full_values),
            "p50_bytes": percentile([float(value) for value in full_values], 0.50),
        }
    else:
        b_full = {
            "status": "UNAVAILABLE",
            "definition": "same-workload Resident after-fill physical resident",
            "reason": "no Resident characterization case",
            "observations": [],
            "min_bytes": None,
            "max_bytes": None,
            "p50_bytes": None,
        }

    release_values = [item["resident_after_release_settle"] for item in release_floor_runs]
    rounds = {item.get("round") for item in runs}
    formal_floor_complete = True
    if run_kind == "formal":
        formal_floor_complete = bool(release_runs) and all(
            any(
                item.get("round") == round_id
                and item.get("release_terminal") == "release_no_candidate"
                for item in release_runs
            )
            for round_id in rounds
        ) and len(release_floor_runs) == len(release_runs)
    if release_values and formal_floor_complete:
        b_release_floor = {
            "status": "AVAILABLE",
            "definition": "same-workload RELEASE-only settled physical resident after a real Unified RELEASE no_candidate floor probe",
            "observations": [
                {
                    "run_id": item["run_id"],
                    "resident_after_release_settle": item["resident_after_release_settle"],
                    "release_terminal": item["release_terminal"],
                }
                for item in release_floor_runs
            ],
            "min_bytes": min(release_values),
            "max_bytes": max(release_values),
            "p50_bytes": percentile([float(value) for value in release_values], 0.50),
        }
    else:
        reason = "no RELEASE-only no_candidate floor probe"
        if run_kind == "formal" and release_runs and not formal_floor_complete:
            reason = "formal floor probe is not release_no_candidate in every round"
        b_release_floor = {
            "status": "UNAVAILABLE",
            "definition": "same-workload RELEASE-only settled physical resident after a real Unified RELEASE no_candidate floor probe",
            "reason": reason,
            "observations": [],
            "min_bytes": None,
            "max_bytes": None,
            "p50_bytes": None,
        }
    release_target_points = [
        {
            "run_id": item["run_id"],
            "resident_after_release_settle": item["resident_after_release_settle"],
            "requested_target_bytes": item["requested_target_bytes"],
            "release_terminal": item["release_terminal"],
        }
        for item in release_target_runs
    ]

    if v2_runs:
        lowest_target = min(item["requested_target_bytes"] for item in v2_runs)
        lowest_runs = [item for item in v2_runs if item["requested_target_bytes"] == lowest_target]
        physical_floor_runs = [
            item for item in lowest_runs
            if item["status"] == "UNMET_FLOOR"
            and item.get("resident_settled_authority") == "slots_physical"
            and item.get("resident_settled") is not None
        ]
        if (
            lowest_runs
            and len(physical_floor_runs) == len(lowest_runs)
            and all(item["status"] == "UNMET_FLOOR" for item in lowest_runs)
        ):
            floor_values = [item["resident_settled"] for item in physical_floor_runs]
            b_floor = {
                "status": "AVAILABLE",
                "definition": "actual settled resident from the lowest explicit target with terminal unmet budget",
                "requested_target_bytes": lowest_target,
                "observations": [
                    {
                        "run_id": item["run_id"],
                        "resident_settled": item["resident_settled"],
                        "resident_settled_authority": item["resident_settled_authority"],
                        "unmet_budget_bytes": item["unmet_budget_bytes"],
                    }
                    for item in physical_floor_runs
                ],
                "min_bytes": min(floor_values),
                "max_bytes": max(floor_values),
                "p50_bytes": percentile([float(value) for value in floor_values], 0.50),
            }
        else:
            reason = "lowest explicit target did not terminate as UNMET_FLOOR in every run"
            if lowest_runs and any(
                    item.get("resident_settled_authority") != "slots_physical"
                    for item in lowest_runs):
                reason = "independent physical resident authority is unavailable"
            b_floor = {
                "status": "UNAVAILABLE",
                "definition": "actual settled resident from the lowest explicit target with terminal unmet budget",
                "requested_target_bytes": lowest_target,
                "reason": reason,
                "observations": [],
                "min_bytes": None,
                "max_bytes": None,
                "p50_bytes": None,
            }
    else:
        b_floor = {
            "status": "UNAVAILABLE",
            "definition": "actual settled resident from the lowest explicit target with terminal unmet budget",
            "requested_target_bytes": None,
            "reason": "no V2 characterization case",
            "observations": [],
            "min_bytes": None,
            "max_bytes": None,
            "p50_bytes": None,
        }
    status = (
        "UNMET_FLOOR" if any(item["status"] == "UNMET_FLOOR" for item in v2_runs)
        else "TARGET_REACHED" if v2_runs
        else "RELEASE_FLOOR_PROBE" if release_floor_runs
        else "RELEASE_SETTLED" if release_runs
        else "TARGET_REACHED"
    )
    comparisons, comparison_aggregate = characterization_comparisons(runs)
    return {
        "status": status,
        "runs": runs,
        "b_full": b_full,
        "b_release_floor": b_release_floor,
        "release_target_points": release_target_points,
        "b_reachable_floor": b_floor,
        "baseline_ladder": {
            "resident_b_full": b_full,
            "release_only_b_release_floor": b_release_floor,
            "v2_b_reachable_floor": b_floor,
        },
        "comparisons": comparisons,
        "comparison_aggregate": comparison_aggregate,
    }


def parse_artifact(artifact: pathlib.Path) -> tuple[str, dict[str, Any]]:
    try:
        manifest, cases, plan, workload = validate_manifest(artifact)
        status = manifest["runner_status"]
        if status == "UNSUPPORTED":
            result = {
                "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "artifact_id": manifest["artifact_id"],
                "verdict": "UNSUPPORTED", "errors": [], "unsupported": manifest["unsupported"],
                "telemetry_gaps": TELEMETRY_GAPS,
            }
            return "UNSUPPORTED", result
        if status == "DRY_RUN":
            result = {
                "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL, "artifact_id": manifest["artifact_id"],
                "verdict": "DRY_RUN", "errors": [], "planned_runs": plan,
                "telemetry_gaps": TELEMETRY_GAPS,
            }
            return "DRY_RUN", result
        run_results: list[dict[str, Any]] = []
        service_failures: list[dict[str, Any]] = []
        for item in plan:
            parsed = parse_run(
                artifact, item, cases[item["case_id"]], workload, len(run_results), manifest["spec"])
            run_results.append(parsed)
            service_failures.extend(parsed["service_failures"])
        statistics = {
            "runs": len(run_results),
            "by_case": {
                item["case_id"]: {
                    "runs": [parsed["statistics"] for parsed in run_results if parsed["case_id"] == item["case_id"]]
                }
                for item in plan
            },
        }
        physical_observations = [
            {
                "run_id": item["run_id"],
                "authority": (
                    "transaction_local_mincore"
                    if item["qualified_offload_pairs"]
                    and item["characterization"] is not None
                    and item["characterization"]["total_physical_relief_bytes"] is not None
                    else "transaction_local_mincore_only"
                    if item["qualified_offload_pairs"]
                    else "phase_boundary_release"
                    if item["characterization"] is not None
                    and item["policy"] in BUDGET_POLICIES
                    else "slots_pre_post"),
                "release_physical_relief_bytes": (
                    item["characterization"]["release_physical_relief_bytes"]
                    if item["characterization"] is not None else 0),
                "offload_physical_relief_bytes": (
                    item["characterization"]["offload_physical_relief_bytes"]
                    if item["characterization"] is not None else 0),
                "release_authority": (
                    item["characterization"]["release_physical_relief_authority"]
                    if item["characterization"] is not None else "not_applicable"),
                "offload_authority": (
                    item["characterization"]["offload_physical_relief_authority"]
                    if item["characterization"] is not None else (
                        "transaction_local_mincore" if item["qualified_offload_pairs"] else "not_applicable")),
                "resident_views": (
                    item["characterization"].get("resident_views")
                    if item["characterization"] is not None else None),
                "budget_resident_views": (
                    item["characterization"].get("budget_resident_views")
                    if item["characterization"] is not None else None),
                "slots_before": slot_resident_values(item["slots_before"]),
                "slots_after": slot_resident_values(item["slots_after"]),
                "transaction_local_offload": item["qualified_offload_pairs"],
                "budget_resident_bytes": [
                    int(action["budget_resident_bytes"])
                    for action in item["actions"]
                    if action.get("budget_resident_bytes") is not None
                ],
            }
            for item in run_results
        ]
        characterization_runs = [
            item["characterization"] for item in run_results if item["characterization"] is not None
        ]
        action_summary = {
            "unified_action_records": sum(len(item["actions"]) for item in run_results),
            "release_attempts": sum(sum(int(action["release_attempted"]) for action in item["actions"]) for item in run_results),
            "offload_attempts": sum(sum(int(action["offload_attempted"]) for action in item["actions"]) for item in run_results),
            "offload_bytes": sum(sum(int(action["bytes"]) for action in item["actions"] if action["offload_attempted"] == "1") for item in run_results),
            "release_bytes": sum(sum(int(action["bytes"]) for action in item["actions"] if action["release_attempted"] == "1") for item in run_results),
            "release_blocks": sum(item["release"]["blocks"] for item in characterization_runs),
            "offload_blocks": sum(item["offload"]["blocks"] for item in characterization_runs),
            "release_physical_relief_bytes": (
                sum(item["release_physical_relief_bytes"] for item in characterization_runs)
                if all(item["release_physical_relief_bytes"] is not None
                       for item in characterization_runs) else None),
            "offload_physical_relief_bytes": sum(
                item["offload_physical_relief_bytes"] for item in characterization_runs),
            "total_physical_relief_bytes": (
                sum(item["total_physical_relief_bytes"] for item in characterization_runs)
                if all(item["total_physical_relief_bytes"] is not None
                       for item in characterization_runs) else None),
            "physical_relief_authority": {
                "release": (
                    "per_run_phase_boundary"
                    if all(item["release_physical_relief_bytes"] is not None
                           for item in characterization_runs)
                    else "UNAVAILABLE_NO_INDEPENDENT_PHYSICAL_AUTHORITY"),
                "offload": "transaction_local_mincore",
                "total": (
                    "release_plus_offload"
                    if all(item["total_physical_relief_bytes"] is not None
                           for item in characterization_runs)
                    else "UNAVAILABLE_NO_INDEPENDENT_PHYSICAL_AUTHORITY"),
            },
            "backing_bytes_written": sum(item["offload"]["bytes_written"] for item in characterization_runs),
            "backing_bytes_read": sum(item["resume"]["bytes_read"] for item in characterization_runs),
            "resume_restored_bytes": sum(item["resume"]["restored_bytes"] for item in characterization_runs),
            "staging_peak_bytes": max([item["transient_staging_peak_bytes"] for item in characterization_runs] or [0]),
            "budget_excess_bytes": [
                int(action["budget_observed_excess_bytes"])
                for item in run_results for action in item["actions"]
            ],
            "budget_unmet_bytes": [
                int(action["unmet_budget_bytes_after"])
                for item in run_results for action in item["actions"]
            ],
            "io_cumulative_last": [item["io"] for item in run_results],
            "source_note": "per-action migration I/O is unavailable; cumulative IO marker is retained without attribution",
        }
        characterization_summary = None
        if manifest["spec"]["run_mode"] == "characterization":
            if service_failures:
                raise ParseError("characterization contains a service failure")
            characterization_summary = summarize_characterization(
                run_results, manifest["spec"]["run_kind"])
        success_verdict = "FORMAL_PASS" if manifest["spec"]["run_kind"] == "formal" else "QUALIFICATION_PASS"
        result_verdict = (
            characterization_summary["status"]
            if characterization_summary is not None
            else ("VALID_SERVICE_FAILURE" if service_failures else success_verdict)
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "artifact_id": manifest["artifact_id"],
            "run_kind": manifest["spec"]["run_kind"],
            "run_mode": manifest["spec"]["run_mode"],
            "verdict": result_verdict,
            "errors": [],
            "service_failures": service_failures,
            "planned_runs": plan,
            "statistics": statistics,
            "memory_observations": [
                {"run_id": item["run_id"], "memory": item["memory"]}
                for item in run_results
            ],
            "physical_observations": physical_observations,
            "restore_observations": [
                {
                    "run_id": item["run_id"],
                    "qualification_round_trip": item["qualification_round_trip"],
                    "stage_timing": item["resume_timings"],
                    "io": item["io"],
                }
                for item in run_results
            ],
            "action_summary": action_summary,
            "characterization": characterization_summary,
            "telemetry_gaps": TELEMETRY_GAPS,
        }
        return result["verdict"], result
    except ParseError as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "verdict": "INVALID_ARTIFACT",
            "errors": [str(exc)],
            "telemetry_gaps": TELEMETRY_GAPS,
        }
        return "INVALID_ARTIFACT", result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=pathlib.Path)
    args = parser.parse_args()
    artifact = args.artifact
    if not artifact.is_dir():
        print(f"parser error: artifact directory missing: {artifact}", file=sys.stderr)
        return 2
    status, result = parse_artifact(artifact)
    write_verdict(artifact, status, result)
    print(json.dumps({"artifact": str(artifact), "status": status}))
    return {
        "FORMAL_PASS": 0,
        "QUALIFICATION_PASS": 0,
        "TARGET_REACHED": 0,
        "RELEASE_SETTLED": 0,
        "RELEASE_FLOOR_PROBE": 0,
        "UNMET_FLOOR": 0,
        "DRY_RUN": 0,
        "UNSUPPORTED": 3,
        "VALID_SERVICE_FAILURE": 4,
    }.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
