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
SCHEMA_VERSION = 1
SUPPORTED_POLICIES = {"resident", "v2"}
SUPPORTED_KV_REPRESENTATIONS = {"paged"}
SUPPORTED_LOADING_MODES = {"exact"}
SUPPORTED_RESTORES = {"k1_sync", "k2_pipeline"}
SUPPORTED_PREFAULTS = {"off", "r2"}
SUPPORTED_RUN_KINDS = {"qualification", "formal"}
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
    "schema_version", "protocol", "phase", "run_kind", "binary", "model", "model_quantization",
    "server_args", "environment", "pressure_basis", "workload", "cases", "run_order", "sampler", "cgroup",
    "max_blocks", "health_timeout_seconds", "request_timeout_seconds",
}
CASE_KEYS = {"case_id", "policy", "kv_representation", "loading_mode", "restore", "prefault", "kv_target_bytes"}
PLAN_KEYS = {
    "run_id", "round", "run_order", "case_id", "policy", "kv_representation", "loading_mode",
    "restore", "prefault", "kv_target_bytes",
}
RUN_KEYS = {"run_id", "round", "run_order", "case_id", "execution_index", "case", "request_plan"}
EXECUTION_KEYS = {
    "run_id", "round", "run_order", "case_id", "execution_index", "argv", "environment", "server_identity",
    "server_cgroup", "pressure_basis", "sampler_identity", "sampler_schema", "sampler_argv", "request_loop_started",
    "request_count", "qualification",
}
CLEANUP_KEYS = {"server", "sampler", "residual_process", "cleanup_complete"}
PROCESS_CLEANUP_KEYS = {
    "pid", "pgid", "exit_code", "stop_requested", "stop_signal", "term_timed_out", "kill_timed_out",
    "pgid_check_complete", "residual_process",
}
RESPONSE_KEYS = {
    "sequence", "request_id", "repeat_index", "measurement", "n_predict", "stream",
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
IO_REQUIRED = {
    "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls", "backing_write_syscalls",
    "bytes_read", "bytes_written", "staging_buffer_bytes", "k2_enabled", "k2_staging_bound_bytes",
    "k2_peak_staging_groups", "k2_peak_staging_bytes", "k2_pipeline_wall_us", "k2_exposed_read_wait_us",
    "k2_pipeline_stall_us", "k2_read_completed_ahead", "restore_prefault_enabled",
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
    value = exact(workload, {"warmup", "requests", "repeat", "qualification"}, "spec.workload")
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
        value = dict(value)
        value["qualification"] = {
            "idle_seconds": idle_seconds,
            "offload_timeout_seconds": offload_timeout_seconds,
            "resume_request_id": resume_request_id,
        }
    return value


def expanded_request_plan(workload: dict[str, Any], case: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    sequence = 0

    def append(item: dict[str, Any], request_id: str, repeat_index: int, measurement: bool) -> None:
        nonlocal sequence
        result.append({
            "sequence": sequence,
            "request_id": request_id,
            "repeat_index": repeat_index,
            "measurement": measurement,
            "n_predict": item["n_predict"],
            "prompt_sha256": sha256_bytes(item["prompt"].encode("utf-8")),
            "stream": item["stream"],
        })
        sequence += 1

    for item in workload["warmup"]:
        append(item, item["request_id"], 0, False)
    for repeat_index in range(1, workload["repeat"] + 1):
        for item in workload["requests"]:
            append(item, item["request_id"], repeat_index, True)
    if workload["qualification"] is not None and case["policy"] == "v2":
        resume_request_id = workload["qualification"]["resume_request_id"]
        resume_request = next(
            item for item in workload["requests"] if item["request_id"] == resume_request_id)
        append(resume_request, resume_request_id, 0, False)
    return result


def validate_spec(spec: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    value = exact(spec, SPEC_KEYS, "spec")
    if value["schema_version"] != SCHEMA_VERSION or value["protocol"] != PROTOCOL:
        raise ParseError("spec protocol/schema mismatch")
    if value["phase"] not in {"resident_baseline", "coarse_target", "local_target", "representative"}:
        raise ParseError("spec.phase is invalid")
    if value["run_kind"] not in SUPPORTED_RUN_KINDS:
        raise ParseError("spec.run_kind is invalid")
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
    if value["run_kind"] == "qualification" and workload["qualification"] is None:
        raise ParseError("qualification run requires explicit idle/offload-barrier/resume configuration")
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
        if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target <= 0):
            raise ParseError(f"{case_id}.kv_target_bytes is invalid")
        if case["policy"] == "resident" and target is not None:
            raise ParseError(f"{case_id}: resident target must be null")
        if case["policy"] == "v2" and target is None:
            raise ParseError(f"{case_id}: v2 target is missing")
        cases[case_id] = case
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
        })
    if value["run_kind"] == "formal" and (len({item["round"] for item in plan}) < 2 or workload["repeat"] < 2):
        raise ParseError("formal run requires explicit multi-round plan and repeat >= 2")
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


def is_qualifying_offload_action(action: dict[str, str], expected_source: str) -> bool:
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
        and int(action["shortfall_bytes"]) == 0
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
        if not is_qualifying_offload_action(action, expected_source):
            continue
        key = (action["decision_id"], action["transaction_id"], action["selected_seq_id"])
        for observation in observations_by_key.get(key, []):
            result.append({
                "decision_id": int(action["decision_id"]),
                "transaction_id": int(action["transaction_id"]),
                "seq_id": int(action["selected_seq_id"]),
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
        "LLAMA_KV_PAGED_SWAP": "0" if case["policy"] == "resident" else "1",
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
    elif case["policy"] == "v2":
        target = str(case["kv_target_bytes"])
        if env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION") != "1" or env.get("LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE") != "1":
            raise ParseError(f"{label}: v2 policy did not enable unified action")
        if env.get("LLAMA_KV_RESIDENT_TARGET_BYTES") != target or env.get("LLAMA_KV_RESIDENT_TARGET_SOURCE") != "env_static":
            raise ParseError(f"{label}: v2 resident target environment mismatch")
        if env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES") != target:
            raise ParseError(f"{label}: v2 action target environment mismatch")
        if env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS") != str(spec["max_blocks"]):
            raise ParseError(f"{label}: v2 action max-blocks environment mismatch")


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
        for key in ("sequence", "request_id", "repeat_index", "measurement", "n_predict", "stream"):
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
        if isinstance(predicted_n, (int, float)) and predicted_n > 0 and isinstance(predicted_ms, (int, float)) and predicted_ms >= 0:
            tpot.append(float(predicted_ms) / float(predicted_n))
            elapsed_s = (record["finished_mono_ns"] - record["started_mono_ns"]) / 1_000_000_000
            if elapsed_s > 0:
                throughput.append(float(predicted_n) / elapsed_s)
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
            "before_object_id", "before_generation", "before_resident_bytes", "after_object_id",
            "after_generation", "after_resident_bytes", "resident_drop_bytes",
        },
        f"{label}.offload_barrier",
    )
    if barrier["status"] != "passed":
        raise ParseError(f"{label}: real OFFLOAD barrier did not pass")
    for key in (
        "started_mono_ns", "completed_mono_ns", "duration_ns", "stderr_start_offset",
        "stderr_end_offset", "decision_id", "transaction_id", "seq_id", "before_object_id",
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
            "decision_id", "transaction_id", "seq_id", "before_object_id", "before_generation",
            "before_resident_bytes", "after_object_id", "after_generation", "after_resident_bytes",
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
    for prefetch in resume_events:
        if (
            prefetch["phase"] != "prefetch"
            or prefetch["action"] != "prefetch"
            or prefetch["outcome"] != "completed"
            or prefetch["graph_allowed"] != "1"
            or int(prefetch["seq_id"]) != barrier["seq_id"]
        ):
            continue
        key = (prefetch["decision_id"], prefetch["transaction_id"], prefetch["seq_id"])
        graph_gate = next((
            event for event in resume_events
            if event["phase"] == "graph_gate"
            and event["action"] == "prefetch"
            and event["outcome"] == "completed"
            and event["graph_allowed"] == "1"
            and (event["decision_id"], event["transaction_id"], event["seq_id"]) == key
        ), None)
        timing = next((
            item for item in resume_timings
            if (item["decision_id"], item["transaction_id"], item["seq_id"]) == key
            and int(item["restored_blocks"]) > 0
            and int(item["restored_bytes"]) > 0
            and int(item["total_us"]) > 0
        ), None)
        if graph_gate is not None and timing is not None:
            return {
                "offload": barrier_fields,
                "prefetch": prefetch,
                "graph_gate": graph_gate,
                "timing": timing,
                "resume_request": resume,
            }
    raise ParseError(
        f"{label}: qualification resume has no completed graph-allowed PREFETCH with positive restore timing")


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
    expected_plan = expanded_request_plan(workload, case)
    if request_plan != expected_plan:
        raise ParseError(f"{label}: request plan differs from manifest workload")
    execution = exact(read_json(run_dir / "execution.json"), EXECUTION_KEYS, f"{label}.execution")
    for key in ("run_id", "round", "run_order", "case_id"):
        if execution[key] != plan[key]:
            raise ParseError(f"{label}: execution identity mismatch at {key}")
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
    transaction_local_physical_authority = (
        case["policy"] == "v2" and spec["run_kind"] == "qualification")
    slots_before = validate_slot_snapshot(
        run_dir / "slots_before.json",
        f"{label}.slots_before",
        require_resident=not transaction_local_physical_authority,
    )
    slots_after = validate_slot_snapshot(
        run_dir / "slots_after.json",
        f"{label}.slots_after",
        require_resident=not transaction_local_physical_authority,
    )
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
    qualification = validate_qualification_record(
        execution["qualification"], workload["qualification"], case, f"{label}.qualification")
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
    qualified_pairs = qualified_offload_pairs(actions, resident_observations, expected_source)
    qualification_round_trip: dict[str, Any] | None = None
    if workload["qualification"] is not None and (
            case["policy"] == "resident" or spec["run_kind"] == "qualification"):
        qualification_round_trip = validate_qualification_causality(
            qualification, records, stderr_data, case, spec, label)
    if case["policy"] == "v2":
        if not actions:
            raise ParseError(f"{label}: missing mandatory kv_pressure_unified_action marker")
        if not offload_actions:
            raise ParseError(f"{label}: V2 has no offload_attempted=1 action")
        if spec["run_kind"] == "qualification":
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
            for action in offload_actions:
                if action["sample_valid"] != "1" or action["stale"] != "0" or action["pressure_basis_valid"] != "1":
                    raise ParseError(f"{label}: V2 offload action lacks a valid pressure basis")
                if action["state"] != "NORMAL" or action["source"] != expected_source:
                    raise ParseError(f"{label}: V2 offload action is not backed by a NORMAL {expected_source} basis")
                if action["budget_active"] != "1" or action["budget_target_enabled"] != "1" or action["budget_view_valid"] != "1" or action["budget_resident_available"] != "1":
                    raise ParseError(f"{label}: V2 offload action lacks an active valid budget view")
                if int(action["budget_observed_excess_bytes"]) <= 0 or int(action["budget_resident_bytes"]) <= int(action["budget_target_bytes"]):
                    raise ParseError(f"{label}: V2 offload action lacks positive budget excess")
                if action["outcome"] != "completed" or action["io_failure"] != "0" or action["state_changed"] != "1":
                    raise ParseError(f"{label}: V2 offload action did not complete successfully")
                if int(action["blocks"]) <= 0 or int(action["bytes"]) <= 0 or int(action["relieved_bytes"]) <= 0:
                    raise ParseError(f"{label}: V2 offload action has no positive physical relief")
                if int(action["shortfall_bytes"]) != 0:
                    raise ParseError(f"{label}: V2 offload action has a shortfall")
            if int(io_last["block_swap_out_calls"]) <= 0 or int(io_last["bytes_written"]) <= 0:
                raise ParseError(f"{label}: V2 IO evidence has no swap-out/write")
    else:
        if any(action["offload_attempted"] != "0" or action["release_attempted"] != "0" for action in actions):
            raise ParseError(f"{label}: resident case contains state-changing action evidence")
        resident_migration_fields = (
            "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls",
            "backing_write_syscalls", "bytes_read", "bytes_written",
        )
        if any(int(io_last[key]) != 0 for key in resident_migration_fields):
            raise ParseError(f"{label}: resident case contains migration IO evidence")
    if not (case["policy"] == "v2" and spec["run_kind"] == "qualification") \
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
        "resume_events": resumes,
        "resume_timings": timings,
        "io": io_last,
        "statistics": response_statistics(records),
    }


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
    if manifest["spec"]["run_kind"] == "formal" and manifest["spec"]["pressure_basis"]["authority"] != "cgroup_finite" and status != "UNSUPPORTED":
        raise ParseError("formal V2 qualification requires real finite cgroup pressure authority")
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
                    if item["qualified_offload_pairs"] else "slots_pre_post"),
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
        action_summary = {
            "unified_action_records": sum(len(item["actions"]) for item in run_results),
            "release_attempts": sum(sum(int(action["release_attempted"]) for action in item["actions"]) for item in run_results),
            "offload_attempts": sum(sum(int(action["offload_attempted"]) for action in item["actions"]) for item in run_results),
            "offload_bytes": sum(sum(int(action["bytes"]) for action in item["actions"] if action["offload_attempted"] == "1") for item in run_results),
            "release_bytes": sum(sum(int(action["bytes"]) for action in item["actions"] if action["release_attempted"] == "1") for item in run_results),
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
        success_verdict = "FORMAL_PASS" if manifest["spec"]["run_kind"] == "formal" else "QUALIFICATION_PASS"
        result = {
            "schema_version": SCHEMA_VERSION,
            "protocol": PROTOCOL,
            "artifact_id": manifest["artifact_id"],
            "run_kind": manifest["spec"]["run_kind"],
            "verdict": "VALID_SERVICE_FAILURE" if service_failures else success_verdict,
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
    return {"FORMAL_PASS": 0, "QUALIFICATION_PASS": 0, "DRY_RUN": 0, "UNSUPPORTED": 3, "VALID_SERVICE_FAILURE": 4}.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
