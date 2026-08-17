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
from typing import Any, Sequence

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from multi_session_replay import ReplayError, check_fidelity, expand_schedule, load_replay

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = pathlib.Path(__file__).resolve()
PROTOCOL = "kv_offload_benchmark"
SCHEMA_VERSION = 2
SUPPORTED_POLICIES = {"resident", "release_only", "v2", "idle_age", "v3"}
BUDGET_POLICIES = {"release_only", "v2", "idle_age", "v3"}
SWAP_POLICIES = {"v2", "idle_age", "v3"}
Q3_POLICIES = {"v2", "idle_age", "v3"}
SUPPORTED_KV_REPRESENTATIONS = {"paged"}
SUPPORTED_LOADING_MODES = {"exact"}
SUPPORTED_RESTORES = {"k1_sync", "k2_pipeline"}
SUPPORTED_PREFAULTS = {"off", "r2"}
SUPPORTED_RUN_KINDS = {"qualification", "formal"}
SUPPORTED_RUN_MODES = {"qualification", "characterization", "resident_preflight"}
RESIDENT_PREFLIGHT_DEFAULT_SAMPLE_INTERVAL = 0.10
RESIDENT_PREFLIGHT_MIN_SAMPLES = 10
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
UINT = re.compile(r"^[0-9]+$")
SIGNED_INT = re.compile(r"^-?[0-9]+$")
CANONICAL_BUDGET_RELEASE_TARGETS = (
    2_684_354_560,
    2_147_483_648,
    1_610_612_736,
)
CANONICAL_BUDGET_OFFLOAD_TARGETS = (
    1_073_741_824,
    805_306_368,
    536_870_912,
    268_435_456,
)
CANONICAL_BUDGET_ACTION_TARGET_BYTES = 268_435_456
CANONICAL_BUDGET_MAX_BLOCKS = 64
# Reference-only anchors from prior physical observations. They never supply
# actual resident values or participate in budget-curve eligibility gates.
REFERENCE_B_FULL_BYTES = 3_221_028_864
REFERENCE_B_RELEASE_FLOOR_BYTES = 1_386_479_616
BUDGET_SWEEP_ORDER_MODE = "interleaved_reverse"
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
MANIFEST_OPTIONAL = {
    "dry_run", "finished_at_utc", "profile", "runtime_contract", "global_governor_isolation",
}
SPEC_KEYS = {
    "schema_version", "protocol", "phase", "run_kind", "run_mode", "binary", "model", "model_quantization",
    "server_args", "environment", "pressure_basis", "workload", "cases", "run_order", "sampler", "cgroup",
    "max_blocks", "health_timeout_seconds", "request_timeout_seconds", "profile", "global_governor_isolation",
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
    "http_status", "headers", "body_path", "body_bytes", "body_sha256", "completion_sha256", "response_json", "error",
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
ACTION_V3_NUMERIC = {
    "action_elapsed_us", "physical_relief_bytes", "physical_object_id", "physical_generation",
}
ACTION_V3_BOOL = {"physical_relief_available"}
ACTION_V3_POLICIES = {"v2", "idle_age", "lru", "v3"}
# V3 marker authority.  The format marker emits `policy=` and `decision_fallback=`
# (empty when the decision was fully cost-aware / non-V3), so they are required by the
# V3 audit even though the legacy V2 marker never carried them.
ACTION_V3_MARKER_REQUIRED = {"policy", "decision_fallback"}
ACTION_V3_PHYSICAL_FEEDBACK = {"action_elapsed_us", "physical_relief_available",
                               "physical_relief_bytes", "physical_object_id", "physical_generation"}
# Per-claimant score schema, in the exact field order emitted by
# server_kv_pressure_unified_action_format_marker(): 32 fields ending with fallback_reason.
# cost_aware is "1"/"0"; fallback_reason is a bare token ("none" when cost-aware).
SCORE_FIELDS = (
    "seq_id", "eligible", "exclusion", "total", "idle_age_score", "logical_kv_score",
    "reclaimable_score", "lcp_n_past_penalty", "io_cost_penalty", "failure_penalty", "rank",
    "cost_aware", "physical_estimate_available", "physical_estimate_authoritative",
    "estimated_physical_bytes", "reuse_probability_ppm", "expected_offload_write_cost_us",
    "expected_restore_gate_cost_us", "expected_cost_us", "churn_penalty_us", "raw_idle_age_us",
    "raw_answer_tokens", "raw_lcp_hint_tokens", "physical_object_id", "physical_generation",
    "actual_relief_bytes", "resident_lease_until_sample", "round_trip_count",
    "last_offload_bytes", "last_restore_bytes", "fallback_reason",
)
SCORE_NUMERIC_FIELDS = {f for f in SCORE_FIELDS if f != "exclusion" and f != "fallback_reason"}
SCORE_BOOL_FIELDS = {"eligible", "cost_aware", "physical_estimate_available",
                     "physical_estimate_authoritative"}
V3_FALLBACK_REASONS = {
    "none", "physical_unavailable", "physical_not_authoritative", "write_cost_unavailable",
    "restore_cost_unavailable", "no_reuse_evidence",
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


def normalize_replay(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and "lifecycle" not in value:
        value = {**value, "lifecycle": None}
    replay = exact(
        value,
        {"source", "path", "time_dilation", "n_parallel", "session_ids", "admission_timeout_seconds", "lifecycle"},
        "spec.workload.replay",
        {"source_lineage_id", "alignment_family_id"},
    )
    if replay["source"] not in {"transcript", "fixture"}:
        raise ParseError("spec.workload.replay.source is invalid")
    if not isinstance(replay["path"], str) or not replay["path"]:
        raise ParseError("spec.workload.replay.path is invalid")
    if isinstance(replay["time_dilation"], bool) or not isinstance(replay["time_dilation"], (int, float)) or not math.isfinite(float(replay["time_dilation"])) or replay["time_dilation"] < 0:
        raise ParseError("spec.workload.replay.time_dilation is invalid")
    if isinstance(replay["n_parallel"], bool) or not isinstance(replay["n_parallel"], int) or replay["n_parallel"] <= 0:
        raise ParseError("spec.workload.replay.n_parallel is invalid")
    if replay["session_ids"] is not None and (
        not isinstance(replay["session_ids"], list)
        or any(not isinstance(item, str) or not item for item in replay["session_ids"])
        or len(replay["session_ids"]) != len(set(replay["session_ids"]))
    ):
        raise ParseError("spec.workload.replay.session_ids is invalid")
    admission_timeout = require_finite_positive(
        replay["admission_timeout_seconds"], "spec.workload.replay.admission_timeout_seconds")
    raw_lifecycle = replay["lifecycle"]
    if raw_lifecycle is None or raw_lifecycle is False:
        lifecycle = {"enabled": False, "drain_after_last_arrival": False}
    else:
        if raw_lifecycle is True:
            raw_lifecycle = {}
        if not isinstance(raw_lifecycle, dict):
            raise ParseError("spec.workload.replay.lifecycle must be null, boolean, or object")
        unknown = set(raw_lifecycle) - {
            "enabled", "drain_after_last_arrival", "ttl_seconds",
            "parent_manifest_path", "parent_manifest_sha256", "trace_identity",
        }
        if unknown:
            raise ParseError(f"spec.workload.replay.lifecycle schema mismatch extra={sorted(unknown)}")
        enabled = raw_lifecycle.get("enabled", True)
        drain = raw_lifecycle.get("drain_after_last_arrival", False)
        if not isinstance(enabled, bool) or not isinstance(drain, bool):
            raise ParseError("spec.workload.replay.lifecycle enabled/drain_after_last_arrival must be boolean")
        lifecycle = {"enabled": enabled, "drain_after_last_arrival": drain}
        if "ttl_seconds" in raw_lifecycle:
            lifecycle["ttl_seconds"] = require_finite_positive(raw_lifecycle["ttl_seconds"], "spec.workload.replay.lifecycle.ttl_seconds")
        for key in ("parent_manifest_path", "parent_manifest_sha256"):
            if key in raw_lifecycle and (not isinstance(raw_lifecycle[key], str) or not raw_lifecycle[key]):
                raise ParseError(f"spec.workload.replay.lifecycle.{key} must be a non-empty string")
            if key in raw_lifecycle:
                lifecycle[key] = raw_lifecycle[key]
        if "trace_identity" in raw_lifecycle:
            if not isinstance(raw_lifecycle["trace_identity"], dict):
                raise ParseError("spec.workload.replay.lifecycle.trace_identity must be an object")
            lifecycle["trace_identity"] = dict(raw_lifecycle["trace_identity"])
    if lifecycle["enabled"] and float(replay["time_dilation"]) <= 0:
        raise ParseError("spec.workload.replay.lifecycle requires time_dilation > 0")
    family_keys = {"source_lineage_id", "alignment_family_id"}
    if bool(family_keys & set(replay)):
        lineage = replay.get("source_lineage_id")
        family = replay.get("alignment_family_id")
        if isinstance(lineage, bool) or not isinstance(lineage, int) or lineage < 0:
            raise ParseError("spec.workload.replay.source_lineage_id is invalid")
        if not isinstance(family, str) or not family:
            raise ParseError("spec.workload.replay.alignment_family_id is invalid")
    return {**replay, "time_dilation": float(replay["time_dilation"]),
            "admission_timeout_seconds": admission_timeout, "lifecycle": lifecycle}


FINAL_F16_PROFILE = "final_f16"
FINAL_F16_ISOLATION = {
    "LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS": "0",
    "LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM": "0",
    "LLAMA_MEMORY_GOVERNOR_KV_RELEASE": "0",
    "LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD": "0",
    "LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET": "0",
    "LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO": "0",
    "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER": "0",
    "LLAMA_MEMORY_GOVERNOR_REALLOCATION": "0",
    "LLAMA_MEMORY_GOVERNOR_DENSE_REPIN": "0",
    "LLAMA_MEMORY_GOVERNOR_DENSE_RING_SHRINK": "0",
    "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC": "0",
    "LLAMA_MEMORY_GOVERNOR_ASYNC_ACTIONS": "0",
}

Q2Q3_RUNTIME_CONTRACT_KEYS = {
    "ctx_size", "executor", "kv_unified", "parallel", "cache_type_k", "cache_type_v",
    "no_cache_idle_slots", "no_context_shift", "paged_block_size",
    "action_target_bytes", "max_blocks",
}


def normalize_runtime_contract(
        value: Any, label: str = "runtime_contract", profile: str | None = None) -> dict[str, Any] | None:
    if value is None:
        return None
    final = profile == FINAL_F16_PROFILE
    expected = Q2Q3_RUNTIME_CONTRACT_KEYS | ({"restore_path", "prefault"} if final else set())
    contract = exact(value, expected, label)
    ctx_size = contract["ctx_size"]
    if isinstance(ctx_size, bool) or not isinstance(ctx_size, int) or ctx_size <= 0:
        raise ParseError(f"{label}.ctx_size is invalid")
    if not isinstance(contract["executor"], str) or not contract["executor"]:
        raise ParseError(f"{label}.executor is invalid")
    for key in ("kv_unified", "no_cache_idle_slots", "no_context_shift"):
        if contract[key] is not True:
            raise ParseError(f"{label}.{key} must be true")
    expected_parallel = 1 if final else 3
    if contract["parallel"] != expected_parallel:
        raise ParseError(f"{label}.parallel must be {expected_parallel}")
    for key in ("cache_type_k", "cache_type_v"):
        if contract[key] != "f16":
            raise ParseError(f"{label}.{key} must be f16")
    if contract["paged_block_size"] != 16:
        raise ParseError(f"{label}.paged_block_size must be 16")
    if final and (contract["restore_path"] != "k2_pipeline" or contract["prefault"] != "r2"):
        raise ParseError(f"{label} must bind restore_path=k2_pipeline and prefault=r2")
    for key in ("action_target_bytes", "max_blocks"):
        if isinstance(contract[key], bool) or not isinstance(contract[key], int) or contract[key] <= 0:
            raise ParseError(f"{label}.{key} is invalid")
    return dict(contract)


def _argv_option_value(argv: list[str], option: str) -> str | None:
    for index, item in enumerate(argv):
        if item == option and index + 1 < len(argv):
            return argv[index + 1]
        if item.startswith(option + "="):
            return item.split("=", 1)[1]
    return None

def normalize_resident_preflight(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    config = exact(
        value,
        {
            "sample_interval_seconds", "min_samples", "window_timeout_seconds",
            "max_sample_gap_seconds", "normalized_resident_spread_bytes",
            "normalized_resident_spread_ratio", "claimant_relief_spread_bytes",
            "claimant_relief_spread_ratio",
        },
        "spec.workload.resident_preflight",
    )
    interval = require_finite_positive(
        config["sample_interval_seconds"],
        "spec.workload.resident_preflight.sample_interval_seconds",
    )
    if abs(interval - RESIDENT_PREFLIGHT_DEFAULT_SAMPLE_INTERVAL) > 1e-9:
        raise ParseError("spec.workload.resident_preflight.sample_interval_seconds must be 0.1")
    min_samples = config["min_samples"]
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < RESIDENT_PREFLIGHT_MIN_SAMPLES:
        raise ParseError(
            f"spec.workload.resident_preflight.min_samples must be >= {RESIDENT_PREFLIGHT_MIN_SAMPLES}")
    timeout = require_finite_positive(
        config["window_timeout_seconds"],
        "spec.workload.resident_preflight.window_timeout_seconds",
    )
    if timeout < interval * min_samples:
        raise ParseError(
            "spec.workload.resident_preflight.window_timeout_seconds is too short "
            "for the requested sample window")
    max_gap = require_finite_positive(
        config["max_sample_gap_seconds"],
        "spec.workload.resident_preflight.max_sample_gap_seconds",
    )
    if max_gap < interval:
        raise ParseError("spec.workload.resident_preflight.max_sample_gap_seconds is shorter than sample interval")
    for key in ("normalized_resident_spread_bytes", "claimant_relief_spread_bytes"):
        item = config[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ParseError(f"spec.workload.resident_preflight.{key} is invalid")
    for key in ("normalized_resident_spread_ratio", "claimant_relief_spread_ratio"):
        ratio = config[key]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ParseError(f"spec.workload.resident_preflight.{key} must be in [0, 1]")
    return {
        "sample_interval_seconds": interval,
        "min_samples": min_samples,
        "window_timeout_seconds": timeout,
        "max_sample_gap_seconds": max_gap,
        "normalized_resident_spread_bytes": config["normalized_resident_spread_bytes"],
        "normalized_resident_spread_ratio": float(config["normalized_resident_spread_ratio"]),
        "claimant_relief_spread_bytes": config["claimant_relief_spread_bytes"],
        "claimant_relief_spread_ratio": float(config["claimant_relief_spread_ratio"]),
    }


def normalize_workload(workload: Any) -> dict[str, Any]:
    if not isinstance(workload, dict):
        raise ParseError("spec.workload must be an object")
    replay = normalize_replay(workload["replay"]) if "replay" in workload else None
    resident_preflight = normalize_resident_preflight(workload.get("resident_preflight")) if "resident_preflight" in workload else None
    expected_keys = {"warmup", "requests", "repeat", "qualification", "characterization"}
    if replay is not None:
        expected_keys.add("replay")
    if "resident_preflight" in workload:
        expected_keys.add("resident_preflight")
    value = exact(workload, expected_keys, "spec.workload")
    if not isinstance(value["warmup"], list) or not isinstance(value["requests"], list):
        raise ParseError("spec.workload warmup/requests must be arrays")
    if replay is not None:
        if value["warmup"] or value["requests"] or value["qualification"] is not None or value["characterization"] is not None:
            raise ParseError("spec.workload.replay cannot be combined with legacy workload")
        return {"warmup": [], "requests": [], "repeat": 1,
                "qualification": None, "characterization": None,
                "replay": replay, "resident_preflight": resident_preflight}
    all_requests = value["warmup"] + value["requests"]
    ids: list[str] = []
    for index, item in enumerate(all_requests):
        request = exact(
            item, {"request_id", "prompt", "n_predict", "stream"},
            f"workload.request[{index}]", {"temperature", "seed"},
        )
        if not isinstance(request["request_id"], str) or not CASE_ID_RE.fullmatch(request["request_id"]):
            raise ParseError(f"workload.request[{index}] has invalid request_id")
        if not isinstance(request["prompt"], str) or isinstance(request["n_predict"], bool) or not isinstance(request["n_predict"], int) or request["n_predict"] < 0:
            raise ParseError(f"workload.request[{index}] has invalid prompt/n_predict")
        if not isinstance(request["stream"], bool):
            raise ParseError(f"workload.request[{index}].stream must be boolean")
        if "temperature" in request and (
            isinstance(request["temperature"], bool) or not isinstance(request["temperature"], (int, float))
            or not math.isfinite(float(request["temperature"])) or float(request["temperature"]) < 0
        ):
            raise ParseError(f"workload.request[{index}].temperature is invalid")
        if "seed" in request and (isinstance(request["seed"], bool) or not isinstance(request["seed"], int)):
            raise ParseError(f"workload.request[{index}].seed is invalid")
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
        qualification = exact(qualification, {"idle_seconds", "offload_timeout_seconds", "resume_request_id"}, "workload.qualification")
        idle_seconds = require_finite_positive(qualification["idle_seconds"], "workload.qualification.idle_seconds")
        offload_timeout_seconds = require_finite_positive(qualification["offload_timeout_seconds"], "workload.qualification.offload_timeout_seconds")
        resume_request_id = qualification["resume_request_id"]
        if not isinstance(resume_request_id, str) or resume_request_id not in {item["request_id"] for item in value["requests"]}:
            raise ParseError("workload.qualification.resume_request_id must reference a measurement request")
        if not value["warmup"]:
            raise ParseError("qualification requires at least one warmup request")
        normalized["qualification"] = {"idle_seconds": idle_seconds, "offload_timeout_seconds": offload_timeout_seconds, "resume_request_id": resume_request_id}

    characterization = value["characterization"]
    if characterization is not None:
        characterization = exact(characterization, {"idle_seconds", "settle_timeout_seconds", "target_tolerance_bytes", "resume_request_id"}, "workload.characterization")
        idle_seconds = require_finite_positive(characterization["idle_seconds"], "workload.characterization.idle_seconds")
        settle_timeout_seconds = require_finite_positive(characterization["settle_timeout_seconds"], "workload.characterization.settle_timeout_seconds")
        tolerance = characterization["target_tolerance_bytes"]
        if isinstance(tolerance, bool) or not isinstance(tolerance, int) or tolerance < 0:
            raise ParseError("workload.characterization.target_tolerance_bytes is invalid")
        resume_request_id = characterization["resume_request_id"]
        if not isinstance(resume_request_id, str) or resume_request_id not in {item["request_id"] for item in value["requests"]}:
            raise ParseError("workload.characterization.resume_request_id must reference a measurement request")
        if resume_request_id != value["requests"][0]["request_id"]:
            raise ParseError("workload.characterization.resume_request_id must be the first measurement request")
        if not value["warmup"]:
            raise ParseError("characterization requires at least one warmup/fill request")
        normalized["characterization"] = {"idle_seconds": idle_seconds, "settle_timeout_seconds": settle_timeout_seconds, "target_tolerance_bytes": tolerance, "resume_request_id": resume_request_id}
    if resident_preflight is not None:
        raise ParseError("spec.workload.resident_preflight requires a replay workload")
    normalized["resident_preflight"] = None
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
            "temperature": item.get("temperature"),
            "seed": item.get("seed"),
            "stream": item["stream"],
        })
        sequence += 1

    for item in workload["warmup"]:
        append(item, item["request_id"], 0, False, "fill")
    for repeat_index in range(1, workload["repeat"] + 1):
        for request_index, item in enumerate(workload["requests"]):
            if run_mode == "characterization" and case["policy"] in SWAP_POLICIES:
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
    if run_mode == "qualification" and workload["qualification"] is not None and case["policy"] in SWAP_POLICIES:
        resume_request_id = workload["qualification"]["resume_request_id"]
        resume_request = next(
            item for item in workload["requests"] if item["request_id"] == resume_request_id)
        append(resume_request, resume_request_id, 0, False, "qualification_resume")
    return result


def budget_case_signature(item: dict[str, Any]) -> tuple[str, str | None, int | None]:
    return item["policy"], item.get("role"), item["kv_target_bytes"]


def validate_budget_sweep(
        value: Any,
        cases: dict[str, dict[str, Any]],
        plan: list[dict[str, Any]],
        max_blocks: int,
        run_kind: str,
        profile: str | None = None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    sweep = exact(value, {"release_targets_bytes", "offload_targets_bytes", "action_target_bytes", "max_blocks", "rounds", "order_mode"}, "spec.budget_sweep")
    def target_list(raw: Any, label: str) -> tuple[int, ...]:
        if not isinstance(raw, list) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in raw):
            raise ParseError(f"{label} must be a list of positive absolute byte values")
        if len(raw) != len(set(raw)) or raw != sorted(raw, reverse=True):
            raise ParseError(f"{label} must be unique and in descending explicit-byte order")
        return tuple(raw)
    release_targets = target_list(sweep["release_targets_bytes"], "spec.budget_sweep.release_targets_bytes")
    offload_targets = target_list(sweep["offload_targets_bytes"], "spec.budget_sweep.offload_targets_bytes")
    action_target = sweep["action_target_bytes"]
    if isinstance(action_target, bool) or not isinstance(action_target, int) or action_target <= 0:
        raise ParseError("spec.budget_sweep.action_target_bytes must be positive")
    sweep_max_blocks = sweep["max_blocks"]
    if isinstance(sweep_max_blocks, bool) or not isinstance(sweep_max_blocks, int) or sweep_max_blocks <= 0:
        raise ParseError("spec.budget_sweep.max_blocks must be positive")
    rounds = sweep["rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise ParseError("spec.budget_sweep.rounds must be positive")
    if run_kind == "formal" and rounds != 2:
        raise ParseError("formal budget_sweep requires exactly two independent rounds")
    if sweep["order_mode"] != BUDGET_SWEEP_ORDER_MODE:
        raise ParseError("spec.budget_sweep.order_mode must be interleaved_reverse")
    if any(item["policy"] in BUDGET_POLICIES and item["action_target_bytes"] != action_target for item in cases.values()):
        raise ParseError("spec.budget_sweep.action_target_bytes differs from a budget case")
    if profile == FINAL_F16_PROFILE:
        if max_blocks != sweep_max_blocks:
            raise ParseError("final_f16 budget_sweep.max_blocks must match spec.max_blocks")
        release_cases = [
            case for case in cases.values()
            if case["policy"] == "release_only" and case.get("role") == "release_target"
        ]
        floor_cases = [
            case for case in cases.values()
            if case["policy"] == "release_only" and case.get("role") == "release_floor_probe"
        ]
        offload_cases = [case for case in cases.values() if case["policy"] == "v2"]
        if release_targets != tuple(sorted((case["kv_target_bytes"] for case in release_cases), reverse=True)):
            raise ParseError("final_f16 release targets do not match ordinary RELEASE cases")
        if offload_targets != tuple(sorted((case["kv_target_bytes"] for case in offload_cases), reverse=True)):
            raise ParseError("final_f16 offload targets do not match explicit V2 cases")
        floor_targets = {case["kv_target_bytes"] for case in floor_cases}
        ordinary = [case["kv_target_bytes"] for case in release_cases]
        if len(floor_targets) != 1 or not ordinary or max(floor_targets) >= min(ordinary):
            raise ParseError("final_f16 floor probe target is not below ordinary RELEASE targets")
        expected_signatures = [(case["policy"], case.get("role"), case["kv_target_bytes"]) for case in cases.values()]
    else:
        if release_targets != CANONICAL_BUDGET_RELEASE_TARGETS or offload_targets != CANONICAL_BUDGET_OFFLOAD_TARGETS:
            raise ParseError("legacy budget_sweep targets must use canonical F32 byte values")
        if action_target != CANONICAL_BUDGET_ACTION_TARGET_BYTES:
            raise ParseError("legacy budget_sweep action_target_bytes must be exactly 256 MiB")
        if sweep_max_blocks != CANONICAL_BUDGET_MAX_BLOCKS or max_blocks != sweep_max_blocks:
            raise ParseError("legacy budget_sweep and spec.max_blocks must both be exactly 64")
        expected_signatures = [("resident", None, None)]
        expected_signatures.extend(("release_only", None, target) for target in release_targets)
        expected_signatures.extend(("v2", None, target) for target in offload_targets)
    expected_set = set(expected_signatures)
    planned_rounds: dict[int, list[dict[str, Any]]] = {}
    for item in plan:
        planned_rounds.setdefault(item["round"], []).append(item)
    if sorted(planned_rounds) != list(range(1, rounds + 1)):
        raise ParseError("spec.budget_sweep rounds must be numbered consecutively from one")
    for round_id, entries in planned_rounds.items():
        if [item["run_order"] for item in entries] != list(range(1, len(entries) + 1)):
            raise ParseError(f"spec.budget_sweep round {round_id} run_order must be consecutive")
        signatures = [budget_case_signature(item) for item in entries]
        if len(entries) != len(expected_signatures) or set(signatures) != expected_set:
            raise ParseError(f"spec.budget_sweep round {round_id} does not contain the declared case matrix")
        if len(signatures) != len(set(signatures)):
            raise ParseError(f"spec.budget_sweep round {round_id} contains duplicate target cases")
        budget_signatures = [signature for signature in signatures if signature[0] != "resident"]
        if any(budget_signatures[index][0] == budget_signatures[index + 1][0] for index in range(len(budget_signatures) - 1)):
            raise ParseError(f"spec.budget_sweep round {round_id} must interleave RELEASE and V2 cases")
    if rounds == 2:
        if [budget_case_signature(item) for item in planned_rounds[2]] != list(reversed([budget_case_signature(item) for item in planned_rounds[1]])):
            raise ParseError("spec.budget_sweep rounds must use reverse order without randomization")
    result = {"release_targets_bytes": list(release_targets), "offload_targets_bytes": list(offload_targets), "action_target_bytes": action_target, "max_blocks": sweep_max_blocks, "rounds": rounds, "order_mode": sweep["order_mode"]}
    if profile != FINAL_F16_PROFILE:
        result["reference_anchors"] = {"b_full_bytes": REFERENCE_B_FULL_BYTES, "b_release_floor_bytes": REFERENCE_B_RELEASE_FLOOR_BYTES}
    return result


def validate_cross_policy_single_switch(plan: list[dict[str, Any]], label: str = "cross-policy") -> None:
    policies = sorted({item["policy"] for item in plan if item["policy"] in SWAP_POLICIES})
    if len(policies) < 2:
        return
    comparable = (
        "kv_representation", "loading_mode", "restore", "prefault",
        "kv_target_bytes", "action_target_bytes",
    )
    by_policy: dict[str, list[dict[str, Any]]] = {policy: [] for policy in policies}
    for item in sorted(plan, key=lambda value: (value["round"], value["run_order"])):
        if item["policy"] in by_policy:
            by_policy[item["policy"]].append(item)
    reference_policy = policies[0]
    reference = [tuple(item[key] for key in comparable) for item in by_policy[reference_policy]]
    for policy in policies[1:]:
        current = [tuple(item[key] for key in comparable) for item in by_policy[policy]]
        if current != reference:
            raise ParseError(
                f"{label}: policies {reference_policy} and {policy} differ outside policy")

def load_target_freeze_record(
        reference: Any, runtime_contract: dict[str, Any], workload: dict[str, Any],
        label: str = "spec.target_freeze_record") -> dict[str, Any] | None:
    if reference is None:
        return None
    verify_identity(reference, label)
    path = pathlib.Path(reference["path"])
    record = read_json(path)
    if not isinstance(record, dict) or record.get("kind") != "resident_preflight_target_freeze":
        raise ParseError(f"{label} has an invalid record kind")
    if record.get("status") != "FROZEN" or record.get("frozen") is not True:
        raise ParseError(f"{label} is not frozen")
    frozen_t = record.get("frozen_T")
    if isinstance(frozen_t, bool) or not isinstance(frozen_t, int) or frozen_t <= 0:
        raise ParseError(f"{label}.frozen_T is invalid")
    if record.get("runtime_contract") != runtime_contract:
        raise ParseError(f"{label} runtime contract mismatch")
    identity = record.get("identity")
    required = {
        "transcript_sha256", "fixture_sha256", "transcript_identity", "fixture_identity",
        "parent_transcript_identity", "preflight_fixture_identity", "q2_evidence_identity",
        "source_lineage_id", "alignment_family_id",
    }
    if not isinstance(identity, dict) or not required.issubset(identity):
        raise ParseError(f"{label} identity is missing required Q2 provenance")
    replay_path = pathlib.Path(workload["replay"]["path"]).resolve()
    preflight_fixture = identity["preflight_fixture_identity"]
    verify_identity(preflight_fixture, f"{label}.preflight_fixture_identity")
    if pathlib.Path(preflight_fixture["path"]).resolve() != replay_path:
        raise ParseError(f"{label} preflight fixture path differs from workload replay")
    if identity["transcript_sha256"] != preflight_fixture["sha256"]:
        raise ParseError(f"{label} preflight fixture SHA mismatch")
    legacy_transcript = identity.get("transcript_identity")
    if not isinstance(legacy_transcript, dict) or legacy_transcript != preflight_fixture:
        raise ParseError(f"{label} transcript identity is not the preflight fixture")
    verify_identity(identity["parent_transcript_identity"], f"{label}.parent_transcript_identity")
    verify_identity(identity["q2_evidence_identity"], f"{label}.q2_evidence_identity")
    legacy_fixture = identity.get("fixture_identity")
    if not isinstance(legacy_fixture, dict) or legacy_fixture != identity["q2_evidence_identity"]:
        raise ParseError(f"{label} fixture identity is not Q2 evidence")
    if identity["fixture_sha256"] != identity["q2_evidence_identity"]["sha256"]:
        raise ParseError(f"{label} Q2 evidence SHA mismatch")
    source_lineage_id = identity["source_lineage_id"]
    if isinstance(source_lineage_id, bool) or not isinstance(source_lineage_id, int) or source_lineage_id < 0:
        raise ParseError(f"{label} source lineage identity is invalid")
    if not isinstance(identity["alignment_family_id"], str) or not identity["alignment_family_id"]:
        raise ParseError(f"{label} alignment family identity is invalid")
    return record


def bind_target_freeze_cases(
        cases: dict[str, dict[str, Any]], record: dict[str, Any] | None,
        runtime_contract: dict[str, Any] | None) -> None:
    if record is None:
        return
    if runtime_contract is None:
        raise ParseError("target_freeze_record requires runtime_contract")
    frozen_t = record["frozen_T"]
    action_target = runtime_contract["action_target_bytes"]
    for case_id, case in cases.items():
        if case["policy"] not in BUDGET_POLICIES:
            continue
        if case["kv_target_bytes"] is not None or case["action_target_bytes"] is not None:
            raise ParseError(
                f"{case_id}: target_freeze_record forbids manual resident/action targets")
        case["kv_target_bytes"] = frozen_t
        case["action_target_bytes"] = action_target

def validate_spec(spec: Any) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    spec_keys = set(SPEC_KEYS) - {"profile", "global_governor_isolation"}
    if isinstance(spec, dict) and "budget_sweep" in spec:
        spec_keys.add("budget_sweep")
    if isinstance(spec, dict) and "runtime_contract" in spec:
        spec_keys.add("runtime_contract")
    if isinstance(spec, dict) and "target_freeze_record" in spec:
        spec_keys.add("target_freeze_record")
    value = exact(spec, spec_keys, "spec", {"profile", "global_governor_isolation"})
    profile = value.get("profile")
    if profile not in {None, FINAL_F16_PROFILE}:
        raise ParseError("spec.profile is unsupported")
    if profile == FINAL_F16_PROFILE and (value["run_mode"] != "characterization" or value["run_kind"] != "formal"):
        raise ParseError("final_f16 profile requires formal characterization")
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
    if isinstance(value.get("workload"), dict) and "replay" in value["workload"]:
        replay_args = set(value["server_args"])
        if replay_args.intersection({"--parallel", "-np", "--cache-idle-slots", "--no-cache-idle-slots", "--context-shift", "--no-context-shift"}):
            raise ParseError("spec replay workload contains runner-owned multi-session server options")
    if not isinstance(value["environment"], dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value["environment"].items()
    ):
        raise ParseError("spec.environment is invalid")
    runtime_contract = normalize_runtime_contract(value.get("runtime_contract"), profile=value.get("profile"))
    if value["run_mode"] == "resident_preflight" and runtime_contract is None:
        raise ParseError("resident_preflight requires an explicit runtime_contract")
    if profile == FINAL_F16_PROFILE and runtime_contract is None:
        raise ParseError("final_f16 requires an explicit runtime_contract")
    if profile == FINAL_F16_PROFILE and value.get("global_governor_isolation") != FINAL_F16_ISOLATION:
        raise ParseError("final_f16 global_governor_isolation is missing or drifted")
    if runtime_contract is not None and runtime_contract["max_blocks"] != value["max_blocks"]:
        raise ParseError("spec.runtime_contract.max_blocks must match spec.max_blocks")
    reject_canonical_kv_environment(value["environment"], "spec.environment")
    pressure_basis = normalize_pressure_basis(value["pressure_basis"])
    workload = normalize_workload(value["workload"])
    if profile == FINAL_F16_PROFILE:
        for request in workload["warmup"] + workload["requests"]:
            if "temperature" not in request or "seed" not in request:
                raise ParseError(
                    "final_f16 workload requests require explicit temperature and seed")
    if ("replay" in workload and workload["replay"].get("lifecycle", {}).get("enabled", False)
            and any(item == "--slot-save-path" or item.startswith("--slot-save-path=") for item in value["server_args"])):
        raise ParseError("spec lifecycle replay contains runner-owned --slot-save-path")
    if "replay" in workload:
        if value["run_mode"] == "resident_preflight":
            if value["run_kind"] != "qualification" or workload["resident_preflight"] is None:
                raise ParseError(
                    "resident_preflight requires qualification run_kind and workload.resident_preflight")
            if workload["replay"]["lifecycle"]["enabled"]:
                raise ParseError("resident_preflight cannot enable replay lifecycle")
        elif value["run_mode"] != "qualification" or value["run_kind"] != "qualification":
            raise ParseError("replay workload is restricted to qualification mode")
    elif value["run_mode"] == "qualification":
        if workload["qualification"] is None or workload["characterization"] is not None:
            raise ParseError(
                "qualification mode requires only workload.qualification configuration")
    elif value["run_mode"] == "resident_preflight":
        raise ParseError("resident_preflight requires a replay workload")
    elif workload["characterization"] is None or workload["qualification"] is not None:
        raise ParseError(
            "characterization mode requires only workload.characterization configuration")
    target_freeze_record = None
    if value.get("target_freeze_record") is not None:
        if value["phase"] != "representative" or value["run_mode"] == "resident_preflight":
            raise ParseError("target_freeze_record is restricted to representative Q3")
        if runtime_contract is None:
            raise ParseError("target_freeze_record requires runtime_contract")
        target_freeze_record = load_target_freeze_record(
            value["target_freeze_record"], runtime_contract, workload)
    elif value["phase"] == "representative" and runtime_contract is not None \
            and value["run_mode"] != "resident_preflight":
        raise ParseError("representative Q3 requires target_freeze_record")
    cases: dict[str, dict[str, Any]] = {}
    if not isinstance(value["cases"], list) or not value["cases"]:
        raise ParseError("spec.cases must be non-empty")
    for index, raw_case in enumerate(value["cases"]):
        case = exact(raw_case, CASE_KEYS, f"spec.cases[{index}]", {"role"})
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise ParseError(f"spec.cases[{index}].case_id is invalid")
        if case_id in cases:
            raise ParseError(f"duplicate case_id: {case_id}")
        if case.get("role") is not None and case["role"] not in {"resident_baseline", "release_target", "release_floor_probe", "offload_target"}:
            raise ParseError(f"{case_id}.role is invalid")
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
            if profile == FINAL_F16_PROFILE and case.get("role") != "resident_baseline":
                raise ParseError(f"{case_id}: final_f16 Resident case must have role resident_baseline")
        elif case["policy"] in BUDGET_POLICIES and (target is None or action_target is None) and target_freeze_record is None:
            raise ParseError(
                f"{case_id}: {case['policy']} policy requires explicit kv_target_bytes and action_target_bytes")
        if profile == FINAL_F16_PROFILE:
            expected_roles = {"release_target", "release_floor_probe"} if case["policy"] == "release_only" else {"offload_target"} if case["policy"] == "v2" else {"resident_baseline"}
            if case.get("role") not in expected_roles:
                raise ParseError(f"{case_id}: final_f16 role must be one of {sorted(expected_roles)}")
            if case["policy"] in SWAP_POLICIES and (case["restore"] != "k2_pipeline" or case["prefault"] != "r2"):
                raise ParseError(f"{case_id}: final_f16 swap cases require k2_pipeline/r2")
        cases[case_id] = case
    if profile == FINAL_F16_PROFILE:
        release_cases = [case for case in cases.values() if case["policy"] == "release_only"]
        floor_cases = [case for case in release_cases if case.get("role") == "release_floor_probe"]
        if len(floor_cases) != 1 or len(release_cases) < 2:
            raise ParseError("final_f16 requires exactly one floor probe and at least one ordinary RELEASE target")
    bind_target_freeze_cases(cases, target_freeze_record, runtime_contract)
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
        planned = {
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
        }
        if profile == FINAL_F16_PROFILE:
            planned["role"] = case["role"]
        plan.append(planned)
    if value["run_mode"] == "resident_preflight" and any(
            item["policy"] != "resident" for item in plan):
        raise ParseError("resident_preflight is policy-neutral and requires resident cases")
    validate_cross_policy_single_switch(plan)
    if value["phase"] == "representative":
        q3_policies = {item["policy"] for item in plan}
        if q3_policies != Q3_POLICIES or any(
                sum(item["policy"] == policy for item in plan) != 1
                for policy in Q3_POLICIES):
            raise ParseError(
                "representative Q3 run must contain exactly one each of v2, idle_age, and v3")
    budget_sweep = validate_budget_sweep(
        value.get("budget_sweep"), cases, plan, max_blocks, value["run_kind"], profile)
    if profile == FINAL_F16_PROFILE and budget_sweep is None:
        raise ParseError("final_f16 requires an explicit budget_sweep")
    if (
        value.get("budget_sweep") is not None
        and value["run_kind"] == "formal"
        and value["run_mode"] == "characterization"
        and workload["repeat"] < 2
    ):
        raise ParseError("formal budget_sweep requires repeat >= 2 for post-resume steady measurements")
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
    extra = (
        set(fields) - required
        if required and token == "kv_resident_preflight_observation" else set()
    )
    if missing or extra:
        raise ParseError(
            f"{label}: marker schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
    return fields


def numeric_fields(fields: dict[str, str], names: set[str], label: str) -> None:
    for key in names:
        if key in fields and not UINT.fullmatch(fields[key]):
            raise ParseError(f"{label}: {key} is not a non-negative integer")


def validate_action_fields(fields: dict[str, str], label: str) -> None:
    numeric_fields(fields, ACTION_NUMERIC, label)
    numeric_fields(fields, ACTION_V3_NUMERIC, label)
    boolean_fields(fields, ACTION_V3_BOOL, label)
    if "policy" in fields and fields["policy"] not in ACTION_V3_POLICIES:
        raise ParseError(f"{label}: unsupported policy marker {fields['policy']!r}")
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


def _parse_v3_score_records(scores_value: str, label: str) -> list[dict[str, str]]:
    # `scores=` is "none" when no claimants were scored, otherwise a `;`-separated
    # list of `:`-separated per-claimant field groups matching SCORE_FIELDS order.
    if scores_value == "none":
        return []
    records: list[dict[str, str]] = []
    for index, group in enumerate(scores_value.split(";")):
        cells = group.split(":")
        if len(cells) != len(SCORE_FIELDS):
            raise ParseError(
                f"{label}: score[{index}] has {len(cells)} fields, expected {len(SCORE_FIELDS)}")
        records.append(dict(zip(SCORE_FIELDS, cells)))
    return records


def parse_claimant_epoch_map(value: str, label: str) -> dict[int, int]:
    if value == "none":
        return {}
    if not isinstance(value, str) or not value:
        raise ParseError(f"{label}: claimant runtime history is missing")
    result: dict[int, int] = {}
    for index, group in enumerate(value.split(";")):
        cells = group.split(":")
        if len(cells) != 10:
            raise ParseError(f"{label}: claimant[{index}] field count is invalid")
        if (
            not UINT.fullmatch(cells[0])
            or not UINT.fullmatch(cells[1])
            or int(cells[1]) <= 0
            or any(cell not in {"0", "1"} for cell in cells[2:5])
            or any(not UINT.fullmatch(cell) for cell in cells[5:])
        ):
            raise ParseError(f"{label}: claimant[{index}] epoch/runtime fields are invalid")
        seq_id = int(cells[0])
        epoch = int(cells[1])
        if seq_id in result:
            raise ParseError(f"{label}: claimant seq_id is duplicated")
        result[seq_id] = epoch
    return result


def _u64_add(lhs: int, rhs: int) -> int:
    return min((1 << 64) - 1, lhs + rhs)


def validate_v3_score_formula(
        scores: list[dict[str, str]], decision_fallback: str, label: str) -> None:
    eligible = [score for score in scores if score["eligible"] == "1"]
    expected_order = []
    for score in eligible:
        seq_id = int(score["seq_id"])
        if score["cost_aware"] == "1":
            reuse = int(score["reuse_probability_ppm"])
            restore = (int(score["expected_restore_gate_cost_us"]) * reuse) // 1_000_000 if reuse else 0
            expected_cost = _u64_add(
                _u64_add(int(score["expected_offload_write_cost_us"]), restore),
                int(score["churn_penalty_us"]))
            if expected_cost != int(score["expected_cost_us"]):
                raise ParseError(f"{label}: score {seq_id} expected_cost_us formula mismatch")
            scaled = min((1 << 64) - 1, expected_cost * 1_000_000)
            expected_total = min((1 << 63) - 1, scaled // max(int(score["estimated_physical_bytes"]), 1))
        else:
            expected_total = int(score["idle_age_score"])
        if int(score["total"]) != expected_total:
            raise ParseError(f"{label}: score {seq_id} total formula mismatch")
        expected_order.append((score, expected_total))
    if decision_fallback == "none":
        expected_order.sort(key=lambda item: (item[1], int(item[0]["seq_id"])))
    else:
        expected_order.sort(key=lambda item: (-int(item[0]["idle_age_score"]), int(item[0]["seq_id"])))
    expected_seq = [item[0]["seq_id"] for item in expected_order]
    actual_seq = [score["seq_id"] for score in eligible]
    if actual_seq != expected_seq:
        raise ParseError(f"{label}: score order does not match current C++ V3 authority")

def validate_action_v3_audit(action: dict[str, str], case: dict[str, Any], label: str) -> None:
    # Canonical V3 tightening (review-fix item 4): a V3 decision marker must carry
    # a single comparable ranking authority.  Fail-closed when the cost feedback or
    # per-claimant audit record is absent instead of silently passing it through.
    if case.get("policy") != "v3":
        return
    if "policy" in action and action["policy"] != "v3":
        raise ParseError(f"{label}: policy marker {action['policy']!r} does not match V3 case")
    for key in ACTION_V3_MARKER_REQUIRED - set(action):
        raise ParseError(f"{label}: missing V3 marker field {key!r}")
    if action["policy"] != "v3":
        raise ParseError(f"{label}: policy marker must be 'v3' for V3 case")

    # Determine whether this marker captured a state-changing OFFLOAD selected a
    # claimant; a no-op decision (claimant_no_candidate / not_pressure / evaluate
    # rejection / pure release) needs no physical feedback or victim score audit.
    is_offload = action["offload_attempted"] == "1"
    state_changed = is_offload and action["state_changed"] == "1"

    for key in ACTION_V3_PHYSICAL_FEEDBACK - set(action):
        raise ParseError(f"{label}: missing V3 physical feedback field {key!r}")

    score_records = _parse_v3_score_records(action.get("scores", "none"), label)
    if not is_offload:
        return
    selected_seq_id = int(action["selected_seq_id"])
    if selected_seq_id < 0:
        raise ParseError(f"{label}: OFFLOAD attempted without a non-negative selected_seq_id")

    selected_score: dict[str, str] | None = None
    seen_selected = False
    eligible_before: list[int] = []
    for score_index, record in enumerate(score_records):
        for key in SCORE_NUMERIC_FIELDS:
            value = record[key]
            if not UINT.fullmatch(value):
                raise ParseError(f"{label}: score {key} is not a non-negative integer")
        for key in SCORE_BOOL_FIELDS:
            if record[key] not in {"0", "1"}:
                raise ParseError(f"{label}: score {key} is not a boolean")
        if record["fallback_reason"] not in V3_FALLBACK_REASONS:
            raise ParseError(
                f"{label}: score fallback_reason {record['fallback_reason']!r} is not a known V3 reason")
        if record["exclusion"] not in {
            "none", "active", "protected", "shared", "write_open", "fail_stop", "exhausted",
            "epoch_mismatch", "stale_generation", "resident_lease", "no_physical_relief", "empty"}:
            raise ParseError(f"{label}: score exclusion {record['exclusion']!r} is unknown")
        if record["cost_aware"] == "1" and record["fallback_reason"] != "none":
            raise ParseError(f"{label}: cost-aware claimant must carry no fallback reason")
        if record["cost_aware"] == "0" and record["fallback_reason"] == "none" and record["eligible"] == "1":
            raise ParseError(f"{label}: eligible non-cost-aware claimant must record a fallback reason")
        if int(record["rank"]) != score_index:
            raise ParseError(f"{label}: score rank {record['rank']} is out of order")
        seq_id = int(record["seq_id"])
        if record["eligible"] == "1":
            eligible_before.append(seq_id)
            if seq_id == selected_seq_id:
                selected_score = record
                seen_selected = True
    if not eligible_before:
        raise ParseError(f"{label}: OFFLOAD attempted but no eligible claimant score was emitted")
    if not seen_selected:
        raise ParseError(
            f"{label}: selected_seq_id={selected_seq_id} has no eligible claimant score")

    # The selected victim must carry either a cost-aware authority or an explicit
    # fallback reason so the decision is auditable.  A bare INT_MAX/4 placeholder
    # would be silently unranked — fail-closed instead.
    if selected_score["cost_aware"] == "1":
        if selected_score["estimated_physical_bytes"] == "0":
            raise ParseError(f"{label}: cost-aware selected claimant reports zero physical bytes")
        if int(selected_score["expected_cost_us"]) <= 0:
            raise ParseError(f"{label}: cost-aware selected claimant has no expected cost")
    elif selected_score["fallback_reason"] == "none":
        raise ParseError(f"{label}: selected claimant score is neither cost-aware nor fallback-tagged")

    # Marker-level `decision_fallback` must agree with the per-claimant score
    # authority so the canonical audit cannot certify a mixed-evidence decision
    # as fully cost-aware, nor a fully cost-aware decision as fallen back.
    marker_fallback = action["decision_fallback"]
    eligible_records = [r for r in score_records if r["eligible"] == "1"]
    eligible_non_cost_aware = [r for r in eligible_records if r["cost_aware"] == "0"]
    if marker_fallback == "none":
        if eligible_non_cost_aware:
            raise ParseError(
                f"{label}: marker decision_fallback=none but eligible non-cost-aware claimant "
                f"{eligible_non_cost_aware[0]['seq_id']} carries a fallback reason")
    else:
        if not any(r["fallback_reason"] == marker_fallback for r in eligible_non_cost_aware):
            raise ParseError(
                f"{label}: marker decision_fallback={marker_fallback!r} is not carried by any "
                f"eligible non-cost-aware claimant")
        if not eligible_non_cost_aware:
            raise ParseError(
                f"{label}: marker decision_fallback={marker_fallback!r} but no eligible "
                f"non-cost-aware claimant exists to justify the fallback")

    validate_v3_score_formula(score_records, action["decision_fallback"], label)

    if state_changed:
        # Physical feedback is the V3 production signal: when an OFFLOAD actually
        # moved KV, the marker must record the action wall time and physical
        # lineage so the next decision can compare cost against actual relief.
        if int(action["action_elapsed_us"]) == 0:
            raise ParseError(f"{label}: state-changing OFFLOAD reports zero action_elapsed_us")
        if int(action["physical_object_id"]) == 0 or int(action["physical_generation"]) == 0:
            raise ParseError(f"{label}: state-changing OFFLOAD missing physical lineage feedback")


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


def offload_physical_relief_from_pairs(
        offload_pairs: list[dict[str, Any]]) -> int:
    # Sum the resident drops of already-validated transaction-local mincore
    # pairs. Qualification V2 contributes a single barrier pair; characterization
    # V2 contributes the same pairs this function sums. Empty (resident /
    # release_only / unmatched) yields 0. Only real physical authority is used;
    # action.bytes / relieved_bytes / logical KV bytes never substitute here.
    return sum(item["resident_drop_bytes"] for item in offload_pairs)


# Phase evidence contract: which logical sessions/seq set must carry positive
# physical evidence per phase. prepare-offload/prepare-restore need both A and B
# to have acted; final-competition binds A/B as candidates but only the single
# selected victim actually offloads; post-final-restore only the selected victim
# restores. These predicate the *evidence* set per phase, independent of the A/B
# slot binding (every phase still binds both A/B slots).
CANONICAL_PHASE_ORDER = ["prepare-offload", "prepare-restore",
                         "final-competition", "post-final-restore"]


def phase_requires_ab_pair_evidence(phase_id: str) -> bool:
    # prepare-offload: both A and B must each complete a real state-changing
    # OFFLOAD with a transaction-local physical resident drop.
    return phase_id == "prepare-offload"


def phase_requires_ab_restore_evidence(phase_id: str) -> bool:
    # prepare-restore: both A and B must each have a positive Exact restore.
    return phase_id == "prepare-restore"


def phase_requires_single_victim_evidence(phase_id: str) -> bool:
    # final-competition: A/B are both eligible candidates, but only one
    # state-changing OFFLOAD victim is permitted.
    return phase_id == "final-competition"


def phase_requires_selected_restore_evidence(phase_id: str) -> bool:
    # post-final-restore: only the selected victim may restore; non-selected
    # candidates must perform no positive restore.
    return phase_id == "post-final-restore"


def phase_evidence_seq_set(
        phase_id: str, window: dict[str, Any],
        actions: list[dict[str, str]],
        observations: list[dict[str, str]],
        resumes: list[dict[str, str]],
        expected_source: str) -> set[int]:
    # Return the set of seq_ids that carry the *positive physical evidence* the
    # contract requires for this phase. For prepare phases this is the A/B seq
    # set (so the caller can assert both acted); for final-competition this is
    # the set of A/B candidate seqs that actually performed a qualifying
    # state-changing OFFLOAD (caller asserts exactly one); for post-final-restore
    # this is the set of positive restore seqs (caller asserts it equals the
    # selected single-victim set).
    if phase_requires_ab_pair_evidence(phase_id):
        pairs = qualified_offload_pairs(actions, observations, expected_source)
        return {pair["seq_id"] for pair in pairs}
    if phase_requires_ab_restore_evidence(phase_id):
        positive = [event for event in resumes
                    if event["phase"] == "prefetch"
                    and event["outcome"] == "completed"
                    and event["graph_allowed"] == "1"]
        return {int(event["seq_id"]) for event in positive}
    if phase_requires_single_victim_evidence(phase_id):
        pairs = qualified_offload_pairs(actions, observations, expected_source)
        return {pair["seq_id"] for pair in pairs}
    if phase_requires_selected_restore_evidence(phase_id):
        positive = [event for event in resumes
                    if event["phase"] == "prefetch"
                    and event["outcome"] == "completed"
                    and event["graph_allowed"] == "1"]
        return {int(event["seq_id"]) for event in positive}
    return set()




def replay_prepare_offload_feedback(
        actions: list[dict[str, str]], observations: list[dict[str, str]],
        expected_source: str, expected_seq_ids: set[int], label: str) -> dict[int, dict[str, int]]:
    actions_by_seq: dict[int, list[dict[str, str]]] = {}
    for action in actions:
        if action.get("offload_attempted") == "1" and action.get("state_changed") == "1":
            seq_id = int(action["selected_seq_id"])
            if seq_id in expected_seq_ids:
                actions_by_seq.setdefault(seq_id, []).append(action)
    if set(actions_by_seq) != expected_seq_ids or any(
            len(items) != 1 for items in actions_by_seq.values()):
        raise ParseError(f"{label}: A/B state-changing OFFLOAD action is missing or duplicated")
    pairs = qualified_offload_pairs(actions, observations, expected_source)
    pairs_by_seq: dict[int, list[dict[str, Any]]] = {}
    for pair in pairs:
        pairs_by_seq.setdefault(pair["seq_id"], []).append(pair)
    if set(pairs_by_seq) != expected_seq_ids:
        raise ParseError(f"{label}: prepare OFFLOAD feedback does not cover exactly A/B")
    if any(len(items) != 1 for items in pairs_by_seq.values()):
        raise ParseError(f"{label}: prepare OFFLOAD feedback is missing or duplicated")

    feedback: dict[int, dict[str, int]] = {}
    for seq_id, items in pairs_by_seq.items():
        pair = items[0]
        action = pair["action"]
        observation = pair["resident_observation"]
        required = {
            "selected_claimant_epoch", "action_elapsed_us", "bytes",
            "physical_relief_available", "physical_relief_bytes",
            "physical_object_id", "physical_generation",
        }
        if not required.issubset(action):
            raise ParseError(f"{label}: prepare OFFLOAD feedback fields are missing")
        if any(not UINT.fullmatch(action[key]) for key in required - {"physical_relief_available"}):
            raise ParseError(f"{label}: prepare OFFLOAD feedback fields are invalid")
        epoch = int(action["selected_claimant_epoch"])
        elapsed_us = int(action["action_elapsed_us"])
        offload_bytes = int(action["bytes"])
        relief_bytes = int(action["physical_relief_bytes"])
        object_id = int(action["physical_object_id"])
        generation = int(action["physical_generation"])
        if (
            action["physical_relief_available"] != "1"
            or epoch <= 0
            or elapsed_us <= 0
            or offload_bytes <= 0
            or relief_bytes <= 0
            or object_id <= 0
            or generation <= 0
            or relief_bytes != pair["resident_drop_bytes"]
            or pair["before_object_id"] != object_id
            or pair["after_object_id"] != object_id
            or pair["before_generation"] != generation
            or pair["after_generation"] != generation
            or int(observation["before_object_id"]) != object_id
            or int(observation["after_object_id"]) != object_id
            or int(observation["before_generation"]) != generation
            or int(observation["after_generation"]) != generation
        ):
            raise ParseError(f"{label}: prepare OFFLOAD physical feedback lineage is inconsistent")
        feedback[seq_id] = {
            "seq_id": seq_id,
            "epoch": epoch,
            "decision_id": pair["decision_id"],
            "transaction_id": pair["transaction_id"],
            "object_id": object_id,
            "generation": generation,
            "offload_elapsed_us": elapsed_us,
            "offload_bytes": offload_bytes,
            "physical_relief_bytes": relief_bytes,
        }
    return feedback


def replay_prepare_restore_feedback(
        resumes: list[dict[str, str]], timings: list[dict[str, str]],
        offload_feedback: dict[int, dict[str, int]], label: str) -> dict[int, dict[str, int]]:
    expected_seq_ids = set(offload_feedback)
    if not expected_seq_ids:
        raise ParseError(f"{label}: prepare OFFLOAD feedback is empty")
    if any(int(event["seq_id"]) not in expected_seq_ids for event in resumes):
        raise ParseError(f"{label}: prepare restore contains an unknown seq_id")
    if any(int(timing["seq_id"]) not in expected_seq_ids for timing in timings):
        raise ParseError(f"{label}: prepare restore timing contains an unknown seq_id")

    feedback: dict[int, dict[str, int]] = {}
    for seq_id, offload in offload_feedback.items():
        seq_resumes = [event for event in resumes if int(event["seq_id"]) == seq_id]
        seq_timings = [timing for timing in timings if int(timing["seq_id"]) == seq_id]
        positive_prefetch = [
            event for event in seq_resumes
            if event["phase"] == "prefetch"
            and event["action"] == "prefetch"
            and event["outcome"] == "completed"
            and event["graph_allowed"] == "1"
        ]
        positive_graph_gate = [
            event for event in seq_resumes
            if event["phase"] == "graph_gate"
            and event["action"] == "prefetch"
            and event["outcome"] == "completed"
            and event["graph_allowed"] == "1"
        ]
        if len(positive_prefetch) != 1 or len(positive_graph_gate) != 1 or len(seq_timings) != 1:
            raise ParseError(f"{label}: prepare restore evidence is missing or duplicated for seq {seq_id}")
        evidence = validate_resume_restore_evidence(
            seq_resumes,
            seq_timings,
            {seq_id: {offload["epoch"]}},
            f"{label}.seq[{seq_id}]")
        prefetch = evidence["prefetch"]
        graph_gate = evidence["graph_gate"]
        timing = evidence["timing"]
        if (
            prefetch["decision_id"] != graph_gate["decision_id"]
            or prefetch["transaction_id"] != graph_gate["transaction_id"]
            or prefetch["seq_id"] != graph_gate["seq_id"]
            or prefetch["claimant_epoch"] != graph_gate["claimant_epoch"]
            or timing["decision_id"] != prefetch["decision_id"]
            or timing["transaction_id"] != prefetch["transaction_id"]
            or timing["seq_id"] != prefetch["seq_id"]
            or int(prefetch["claimant_epoch"]) != offload["epoch"]
            or int(timing["restored_bytes"]) <= 0
            or int(timing["gate_us"]) <= 0
        ):
            raise ParseError(f"{label}: prepare restore identity or timing is inconsistent for seq {seq_id}")
        feedback[seq_id] = {
            "seq_id": seq_id,
            "epoch": offload["epoch"],
            "decision_id": int(prefetch["decision_id"]),
            "transaction_id": int(prefetch["transaction_id"]),
            "restore_bytes": int(timing["restored_bytes"]),
            "restore_gate_us": int(timing["gate_us"]),
        }
    return feedback


def validate_prepare_feedback_history(
        final_action: dict[str, str], scores: list[dict[str, str]],
        offload_feedback: dict[int, dict[str, int]],
        restore_feedback: dict[int, dict[str, int]],
        expected_seq_ids: set[int], selected_seq_id: int, label: str) -> dict[int, dict[str, int]]:
    if set(offload_feedback) != expected_seq_ids or set(restore_feedback) != expected_seq_ids:
        raise ParseError(f"{label}: prepare feedback does not cover exactly A/B")
    claimant_epochs = parse_claimant_epoch_map(final_action.get("claimants", "none"), f"{label}.claimants")
    required_history = {
        "expected_offload_write_cost_us", "expected_restore_gate_cost_us",
        "last_offload_bytes", "last_restore_bytes", "actual_relief_bytes",
        "round_trip_count", "physical_object_id", "physical_generation",
    }
    history: dict[int, dict[str, int]] = {}
    for seq_id in sorted(expected_seq_ids):
        matching = [score for score in scores if int(score.get("seq_id", "-1")) == seq_id]
        if len(matching) != 1:
            raise ParseError(f"{label}: final score history for seq {seq_id} is missing or duplicated")
        score = matching[0]
        if not required_history.issubset(score):
            raise ParseError(f"{label}: final score history fields for seq {seq_id} are missing")
        if score.get("eligible") != "1" or score.get("cost_aware") != "1" or score.get("fallback_reason") != "none":
            raise ParseError(f"{label}: final score history for seq {seq_id} is not cost-aware")
        if any(not UINT.fullmatch(score[key]) for key in required_history):
            raise ParseError(f"{label}: final score history for seq {seq_id} is invalid")
        offload = offload_feedback[seq_id]
        restore = restore_feedback[seq_id]
        if (
            int(score["expected_offload_write_cost_us"]) != offload["offload_elapsed_us"]
            or int(score["expected_restore_gate_cost_us"]) != restore["restore_gate_us"]
            or int(score["last_offload_bytes"]) != offload["offload_bytes"]
            or int(score["last_restore_bytes"]) != restore["restore_bytes"]
            or int(score["actual_relief_bytes"]) != offload["physical_relief_bytes"]
            or int(score["round_trip_count"]) < 1
            or int(score["physical_object_id"]) != offload["object_id"]
            or int(score["physical_generation"]) != offload["generation"]
            or claimant_epochs.get(seq_id) != offload["epoch"]
        ):
            raise ParseError(f"{label}: final score history does not derive from prepare feedback for seq {seq_id}")
        history[seq_id] = {
            "seq_id": seq_id,
            "object_id": int(score["physical_object_id"]),
            "generation": int(score["physical_generation"]),
            "epoch": claimant_epochs[seq_id],
            "offload_elapsed_us": int(score["expected_offload_write_cost_us"]),
            "restore_gate_us": int(score["expected_restore_gate_cost_us"]),
            "offload_bytes": int(score["last_offload_bytes"]),
            "restore_bytes": int(score["last_restore_bytes"]),
            "physical_relief_bytes": int(score["actual_relief_bytes"]),
            "round_trip_count": int(score["round_trip_count"]),
        }
    if selected_seq_id not in expected_seq_ids:
        raise ParseError(f"{label}: selected seq is outside prepare feedback authority")
    if (
        int(final_action["selected_claimant_epoch"]) != claimant_epochs.get(selected_seq_id)
        or int(final_action["physical_object_id"]) != offload_feedback[selected_seq_id]["object_id"]
        or int(final_action["physical_generation"]) != offload_feedback[selected_seq_id]["generation"]
    ):
        raise ParseError(f"{label}: selected final claimant lineage differs from prepare feedback")
    return history

def final_competition_selected_victim(
        final_actions: list[dict[str, str]],
        final_observations: list[dict[str, str]],
        expected_source: str,
        ab_candidate_seqs: set[int]) -> dict[str, Any] | None:
    # The final-competition selected victim must NOT be derived from the last
    # action marker (which may be a trailing no-op). Instead it is the unique
    # qualifying state-changing OFFLOAD + transaction-local resident-drop pair
    # within the final window whose seq belongs to the A/B candidate set. Zero
    # or more than one such pair is fail-closed (returns None).
    pairs = qualified_offload_pairs(final_actions, final_observations, expected_source)
    victim_pairs = [pair for pair in pairs if pair["seq_id"] in ab_candidate_seqs]
    if len(victim_pairs) != 1:
        return None
    return victim_pairs[0]


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


RESIDENT_PREFLIGHT_WHOLE_REQUIRED = {
    "timestamp_mono_ns", "sample_count", "whole_valid", "resident_available",
    "reclaimable_available", "swapped_metadata_consistent", "object_id", "generation",
    "page_size", "native_block_bytes", "total_bytes", "resident_bytes", "dead_resident_reclaimable_bytes",
    "swapped_authoritative_bytes", "transient_staging_bound_bytes",
    "n_blocks", "n_owned_blocks", "n_shared_blocks", "resident_block_count",
    "swapped_block_count", "released_block_count", "pending_write_block_count",
    "invalid_block_count", "unused_block_count", "global_target_enabled",
    "global_target_source", "global_target_bytes", "global_target_basis_generation",
    "observation_only",
}
RESIDENT_PREFLIGHT_WHOLE_NUMERIC = RESIDENT_PREFLIGHT_WHOLE_REQUIRED - {
    "whole_valid", "resident_available", "reclaimable_available",
    "swapped_metadata_consistent", "global_target_enabled", "global_target_source",
    "observation_only",
}
RESIDENT_PREFLIGHT_WHOLE_BOOL = {
    "whole_valid", "resident_available", "reclaimable_available",
    "swapped_metadata_consistent", "global_target_enabled", "observation_only",
}
RESIDENT_PREFLIGHT_CLAIMANT_REQUIRED = {
    "timestamp_mono_ns", "sample_count", "seq_id", "claimant_epoch", "active",
    "valid", "available", "authoritative", "shared", "object_id", "generation",
    "exclusive_resident_bytes", "exclusive_resident_blocks", "swapped_bytes", "swapped_blocks",
}
RESIDENT_PREFLIGHT_CLAIMANT_NUMERIC = RESIDENT_PREFLIGHT_CLAIMANT_REQUIRED - {
    "active", "valid", "available", "authoritative", "shared",
}
RESIDENT_PREFLIGHT_CLAIMANT_BOOL = {"active", "valid", "available", "authoritative", "shared"}


def validate_optional_resident_observation(value: Any, label: str) -> dict[str, Any] | None:
    """Validate the /slots kv_resident union without inventing physical authority.

    The server's preflight observation is legitimately {"status":"unavailable"}
    before a resident sample can be established.  That state is acceptable only
    when the caller explicitly treats resident authority as optional.  Available
    observations still use the strict full physical schema.
    """
    if not isinstance(value, dict):
        raise ParseError(f"{label}: resident observation is not an object")
    status = value.get("status")
    if status == "unavailable":
        if set(value) != {"status"}:
            raise ParseError(f"{label}: unavailable resident observation schema mismatch")
        return None
    if status == "available":
        return validate_resident_observation(value, label)
    raise ParseError(f"{label}: resident observation status is invalid")


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
    residents: list[dict[str, Any]] = []
    saw_unavailable_resident = False
    for index, slot in enumerate(decoded):
        if not isinstance(slot, dict):
            raise ParseError(f"{label}: slot {index} is not an object")
        if "kv_resident" not in slot:
            continue
        resident = validate_optional_resident_observation(
            slot["kv_resident"], f"{label}.slot[{index}].kv_resident")
        if resident is None:
            saw_unavailable_resident = True
        else:
            residents.append(resident)
    if residents and saw_unavailable_resident:
        raise ParseError(f"{label}: slots disagree on physical resident availability")
    if require_resident and not residents:
        raise ParseError(f"{label}: no valid physical resident observation")
    return snapshot


def validate_runtime_execution_identity(
        execution: dict[str, Any], spec: dict[str, Any], label: str) -> None:
    contract = spec.get("runtime_contract")
    if contract is None:
        return
    argv = execution.get("argv")
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv) or not argv:
        raise ParseError(f"{label}: execution argv is invalid")
    if pathlib.Path(argv[0]).name != contract["executor"]:
        raise ParseError(f"{label}: runtime identity executor mismatch")
    expected_parallel = "1" if spec.get("profile") == FINAL_F16_PROFILE else "3"
    if _argv_option_value(argv, "--parallel") != expected_parallel:
        raise ParseError(f"{label}: runtime identity requires --parallel {expected_parallel}")
    if _argv_option_value(argv, "--cache-type-k") != "f16" or _argv_option_value(argv, "--cache-type-v") != "f16":
        raise ParseError(f"{label}: runtime identity requires F16 K/V cache")
    if "--kv-unified" not in argv or "--no-cache-idle-slots" not in argv or "--no-context-shift" not in argv:
        raise ParseError(f"{label}: runtime identity is missing unified/idle/context flags")
    if _argv_option_value(argv, "--ctx-size") != str(contract["ctx_size"]):
        raise ParseError(f"{label}: runtime identity ctx-size mismatch")
    env = execution.get("environment")
    if not isinstance(env, dict) or env.get("LLAMA_KV_PAGED_BLOCK_SIZE") != "16":
        raise ParseError(f"{label}: runtime identity block-size mismatch")

def validate_execution_environment(execution: dict[str, Any], case: dict[str, Any], label: str, spec: dict[str, Any]) -> None:
    validate_runtime_execution_identity(execution, spec, label)
    env = execution["environment"]
    if not isinstance(env, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items()):
        raise ParseError(f"{label}: environment is invalid")
    if spec.get("profile") == FINAL_F16_PROFILE:
        for key, expected in FINAL_F16_ISOLATION.items():
            if env.get(key) != expected:
                raise ParseError(f"{label}: final_f16 governor isolation mismatch for {key}")
    if case["kv_representation"] != "paged" or case["loading_mode"] != "exact":
        raise ParseError(f"{label}: unsupported representation/loading factor reached workload")
    if env.get("LLAMA_KV_PAGED") != "1" or env.get("LLAMA_KV_PAGED_INGRAPH") != "1":
        raise ParseError(f"{label}: paged runtime is not explicitly enabled")
    for key, expected in {
        "LLAMA_KV_PAGED_SWAP": "1" if case["policy"] in SWAP_POLICIES else "0",
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
    if case["policy"] in SWAP_POLICIES and env.get("LLAMA_KV_PRESSURE_POLICY", "v2") != case["policy"]:
        raise ParseError(f"{label}: policy environment does not match case")
    expected_resident_observation = (
        "preflight" if spec["run_mode"] == "resident_preflight"
        else "both" if case["policy"] in SWAP_POLICIES and spec["run_mode"] == "characterization"
        else "1" if case["policy"] in SWAP_POLICIES else "preflight")
    if env.get("LLAMA_KV_G0_S1_RESIDENT_OBSERVATION") != expected_resident_observation:
        raise ParseError(f"{label}: physical resident observation mode mismatch")
    if spec["run_mode"] == "resident_preflight":
        if env.get("LLAMA_KV_RESIDENT_PREFLIGHT") != "1":
            raise ParseError(f"{label}: resident_preflight explicit switch is missing or disabled")
    elif "LLAMA_KV_RESIDENT_PREFLIGHT" in env:
        raise ParseError(f"{label}: non-resident_preflight execution contains the Q2 switch")
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
    if spec["run_mode"] == "resident_preflight":
        if case["policy"] != "resident":
            raise ParseError(f"{label}: resident_preflight reached a non-resident policy")
        validate_resident_preflight_environment(env, label)



def validate_resident_preflight_environment(
        env: dict[str, str], label: str) -> None:
    expected = {
        "LLAMA_MEMORY_GOVERNOR": "0",
        "LLAMA_MEMORY_GOVERNOR_OBSERVE": "1",
        "LLAMA_MEMORY_GOVERNOR_OBSERVE_MS": "100",
        "LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS": "0",
        "LLAMA_MEMORY_GOVERNOR_KV_RELEASE": "0",
        "LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD": "0",
        "LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET": "0",
        "LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM": "0",
        "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER": "0",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION": "0",
        "LLAMA_MEMORY_GOVERNOR_DENSE_REPIN": "0",
        "LLAMA_MEMORY_GOVERNOR_DENSE_RING_SHRINK": "0",
        "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC": "0",
        "LLAMA_MEMORY_GOVERNOR_ASYNC_ACTIONS": "0",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "0",
        "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "0",
        "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "preflight",
    }
    if env.get("LLAMA_KV_RESIDENT_PREFLIGHT") != "1":
        raise ParseError(f"{label}: resident_preflight explicit switch is missing or disabled")
    for key, value in expected.items():
        if env.get(key) != value:
            raise ParseError(f"{label}: resident_preflight is not observation-only at {key}")
    for key in (
        "LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MB_PER_TICK",
        "LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO",
        "LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_RUNTIME_AUTO",
        "LLAMA_KV_RESIDENT_TARGET_BYTES", "LLAMA_KV_RESIDENT_TARGET_SOURCE",
    ):
        if key in env:
            raise ParseError(f"{label}: resident_preflight contains a Global target or prefetch budget")


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
    for action in actions:
        validate_action_v3_audit(action, case, label)


def normalized_completion_sha256(value: Any) -> str | None:
    content: Any = value
    if isinstance(value, dict):
        for key in ("content", "text", "completion", "generated_text"):
            if key in value:
                content = value[key]
                break
    if content is None or isinstance(content, (bool, int, float)):
        return None
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(encoded)

def validate_responses(
        path: pathlib.Path,
        raw_dir: pathlib.Path,
        request_plan: list[dict[str, Any]],
        label: str,
        require_completion_identity: bool = False,
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
        record = exact(
            record,
            RESPONSE_KEYS - {"completion_sha256"},
            f"{label}.response[{line_number}]",
            {"completion_sha256", "prompt_sha256", "temperature", "seed"},
        )
        if require_completion_identity and any(
                key not in record for key in ("prompt_sha256", "temperature", "seed")):
            raise ParseError(f"{label}: Final F16 response workload identity is incomplete")
        if not isinstance(record["request_id"], str) or not CASE_ID_RE.fullmatch(record["request_id"]):
            raise ParseError(f"{label}: response request_id is invalid")
        if not isinstance(record["measurement"], bool) or not isinstance(record["stream"], bool):
            raise ParseError(f"{label}: response boolean field is invalid")
        if "prompt_sha256" in record and (
                not isinstance(record["prompt_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["prompt_sha256"])
        ):
            raise ParseError(f"{label}: response prompt identity is invalid")
        if "temperature" in record and record["temperature"] is not None and (
                isinstance(record["temperature"], bool)
                or not isinstance(record["temperature"], (int, float))
                or not math.isfinite(float(record["temperature"]))
                or float(record["temperature"]) < 0
        ):
            raise ParseError(f"{label}: response temperature identity is invalid")
        if "seed" in record and record["seed"] is not None and (
                isinstance(record["seed"], bool) or not isinstance(record["seed"], int)
        ):
            raise ParseError(f"{label}: response seed identity is invalid")
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
        if record.get("completion_sha256") is not None and (
            not isinstance(record["completion_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", record["completion_sha256"])
        ):
            raise ParseError(f"{label}: response completion hash is invalid")
        if require_completion_identity and record.get("completion_sha256") is None:
            raise ParseError(f"{label}: Final F16 response completion identity is missing")
        body_path = pathlib.Path(record["body_path"])
        if body_path.is_absolute() or ".." in body_path.parts:
            raise ParseError(f"{label}: response body path escapes artifact")
        body = raw_dir.parent / body_path
        if not body.is_file():
            raise ParseError(f"{label}: response body is missing")
        if body.stat().st_size != record["body_bytes"] or sha256_file(body) != record["body_sha256"]:
            raise ParseError(f"{label}: response body identity mismatch")
        try:
            body_value = json.loads(body.read_text(encoding="utf-8")) if record["body_bytes"] else None
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ParseError(f"{label}: response body is not valid UTF-8 JSON") from exc
        if body_value != record["response_json"]:
            raise ParseError(f"{label}: response JSON differs from hash-verified response body")
        expected_completion = normalized_completion_sha256(body_value)
        if record.get("completion_sha256") is not None and record["completion_sha256"] != expected_completion:
            raise ParseError(f"{label}: response completion identity mismatch")
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
        for key in ("prompt_sha256", "temperature", "seed"):
            if key in actual and actual[key] != expected.get(key):
                raise ParseError(f"{label}: response/workload identity mismatch at sequence {expected['sequence']} field {key}")
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
        validate_action_v3_audit(action, case, f"{label}.offload_barrier.action")
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
        if not (case["policy"] in SWAP_POLICIES and settle["status"] == "unmet_floor"):
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
    if case["policy"] in SWAP_POLICIES:
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
    observations: list[dict[str, Any] | None] = []
    for index, slot in enumerate(value):
        if not isinstance(slot, dict) or "kv_resident" not in slot:
            continue
        observations.append(validate_optional_resident_observation(
            slot["kv_resident"], f"{label}.slot[{index}].kv_resident"))
    if not observations:
        return None
    available = [item for item in observations if item is not None]
    if not available:
        return None
    if len(available) != len(observations):
        raise ParseError(f"{label}: slots disagree on physical resident availability")
    first = available[0]
    if any(item != first for item in available[1:]):
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
    if case["policy"] in SWAP_POLICIES:
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
        require_resident=case["policy"] not in SWAP_POLICIES,
    )
    after_measurement_snapshot = validate_slot_snapshot(
        run_dir / "slots_after_measurement.json",
        f"{label}.slots_after_measurement",
        require_resident=case["policy"] not in SWAP_POLICIES,
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
    if case["policy"] in SWAP_POLICIES:
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
            "resident_after_fill_authority": "slots_physical",
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
            validate_action_v3_audit(action, case, f"{label}.settle.action")
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
        if spec.get("profile") == FINAL_F16_PROFILE:
            role = case.get("role")
            expected_terminal = (
                "release_no_candidate" if role == "release_floor_probe"
                else "release_settled" if role == "release_target"
                else None)
            if expected_terminal is None or terminal["status"] != expected_terminal:
                raise ParseError(
                    f"{label}: final_f16 RELEASE role/terminal mismatch: "
                    f"role={role!r} terminal={terminal['status']!r}")
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
            "resident_after_fill_authority": "slots_physical",
            "resident_after_release_settle": settled_resident["resident_bytes"],
            "resident_after_release_settle_authority": "slots_physical",
            "resident_after_offload_settle": None,
            "resident_after_offload_settle_authority": "unavailable",
            "resident_settled": settled_resident["resident_bytes"],
            "resident_settled_authority": "slots_physical",
            "resident_after_resume": settled_resident["resident_bytes"],
            "resident_after_resume_authority": "slots_physical",
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
        validate_action_v3_audit(action, case, f"{label}.settle.action")
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
    offload_attempt_indices = [
        index for index, action in enumerate(settle_actions)
        if action["offload_attempted"] == "1"
    ]
    if offload_attempt_indices and min(offload_attempt_indices) <= boundary_index:
        raise ParseError(
            f"{label}: OFFLOAD attempt overlaps or precedes the RELEASE boundary")
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
    if spec.get("profile") == FINAL_F16_PROFILE and (
        case.get("role") != "offload_target" or not positive_offloads or boundary_status != "release_no_candidate"
    ):
        raise ParseError(
            f"{label}: final_f16 V2 requires role=offload_target, a positive OFFLOAD, "
            "and a preceding RELEASE no_candidate boundary")
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


def validate_replay_erased_slot_snapshot(
        snapshot: dict[str, Any], slot_id: int, label: str) -> dict[str, Any]:
    slots = snapshot.get("body_json")
    if not isinstance(slots, list):
        raise ParseError(f"{label}: post-erase /slots body is unavailable")
    matches = [slot for slot in slots if isinstance(slot, dict) and slot.get("id") == slot_id]
    if len(matches) != 1:
        raise ParseError(f"{label}: post-erase slot identity is missing or duplicated")
    slot = matches[0]
    if slot.get("is_processing") is not False:
        raise ParseError(f"{label}: post-erase slot is still processing")
    n_prompt_tokens = slot.get("n_prompt_tokens")
    if isinstance(n_prompt_tokens, bool) or not isinstance(n_prompt_tokens, int) or n_prompt_tokens != 0:
        raise ParseError(f"{label}: post-erase prompt was not cleared")
    claimant = slot.get("kv_claimant")
    required = {"target_blocks", "eligible_resident_blocks", "swapped_blocks", "shared_blocks", "blocked_blocks"}
    if not isinstance(claimant, dict) or not required.issubset(claimant):
        raise ParseError(f"{label}: post-erase KV claimant evidence is missing")
    counts: dict[str, int] = {}
    for key in sorted(required):
        value = claimant.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ParseError(f"{label}: post-erase KV claimant {key} was not cleared")
        counts[key] = value
    return {"slot_id": slot_id, "n_prompt_tokens": n_prompt_tokens, "claimant_counts": counts}


def validate_replay_lifecycle(
        label: str, replay_plan: Any, replay_cfg: dict[str, Any],
        actual: list[dict[str, Any]], journal: list[dict[str, Any]],
        run_dir: pathlib.Path | None = None) -> dict[str, Any]:
    if not replay_cfg.get("lifecycle", {}).get("enabled", False):
        return {"status": "DISABLED", "errors": []}
    lifecycle = replay_plan.lifecycle or {}
    ttl_runtime_us = int(round(float(lifecycle["ttl_seconds"]) * replay_cfg["time_dilation"] * 1_000_000))
    if ttl_runtime_us <= 0:
        raise ParseError(f"{label}: lifecycle runtime TTL is invalid")
    by_request = {row["request_id"]: row for row in actual}
    if len(by_request) != len(actual):
        raise ParseError(f"{label}: lifecycle request identity is duplicated")
    allowed = {"TURN_START", "TURN_COMPLETE", "TTL_ARM", "REVISIT", "TTL_EXPIRY", "DEAD", "COLD_RESTART"}
    sessions = {
        session.logical_session_id: {
            "lineage_id": session.lineage_id, "state": "ABSENT", "generation": 0,
            "binding": None, "timer": None, "next_turn": 1, "inflight": None,
            "revisit_pending": False, "restart_pending": False, "last_event": None,
            "last_completion": None, "last_start": None,
        }
        for session in replay_plan.sessions
    }
    starts: dict[str, int] = {}
    completes: dict[str, int] = {}
    pending_idle: set[str] = set()
    previous_events: list[dict[str, Any]] = []
    timer_ids: set[int] = set()
    slot_owners: dict[int, str] = {}
    slot_generations: dict[int, int] = {}
    for index, event in enumerate(journal):
        if not isinstance(event, dict) or event.get("event") not in allowed:
            raise ParseError(f"{label}: lifecycle event {index} is invalid")
        sid = event.get("logical_session_id")
        if sid not in sessions or event.get("lineage_id") != sessions[sid]["lineage_id"]:
            raise ParseError(f"{label}: lifecycle event {index} session identity mismatch")
        session = sessions[sid]
        kind = event["event"]
        request_id = event.get("request_id")
        row = by_request.get(request_id) if request_id is not None else None
        binding = (event.get("slot_id"), event.get("seq_id"), event.get("runner_generation"))
        if event.get("slot_id") is not None:
            slot_id = event.get("slot_id")
            runner_generation = event.get("runner_generation")
            if (event.get("seq_id") != slot_id or isinstance(slot_id, bool)
                    or not isinstance(slot_id, int) or not 0 <= slot_id < replay_plan.n_parallel
                    or isinstance(runner_generation, bool) or not isinstance(runner_generation, int)
                    or runner_generation <= 0):
                raise ParseError(f"{label}: lifecycle event {index} slot/seq/generation is invalid")
        if kind == "COLD_RESTART":
            if session["state"] != "DEAD" or not session["last_event"] or session["last_event"].get("event") != "DEAD" or row is None:
                raise ParseError(f"{label}: COLD_RESTART is not after DEAD")
            if row.get("lifecycle_trigger") != "COLD_RESTART" or row.get("cache_prompt") is not False:
                raise ParseError(f"{label}: COLD_RESTART request cache_prompt is invalid")
            if event.get("turn") != session["next_turn"] or event.get("lifecycle_generation") != session["generation"] + 1:
                raise ParseError(f"{label}: COLD_RESTART generation/turn is invalid")
            if event.get("arrival_us") != row.get("planned_arrival_us"):
                raise ParseError(f"{label}: COLD_RESTART arrival differs from frozen replay arrival")
            session["generation"] += 1
            session["restart_pending"] = binding
        elif kind == "TURN_START":
            if row is None or request_id in starts or event.get("turn") != row.get("turn"):
                raise ParseError(f"{label}: lifecycle TURN_START is missing, duplicated, or mismatched")
            trigger = event.get("trigger")
            if trigger not in {"COLD_START", "CONTINUATION", "REVISIT", "COLD_RESTART"}:
                raise ParseError(f"{label}: lifecycle TURN_START trigger is invalid")
            expected_turn = session["next_turn"]
            if event.get("turn") != expected_turn or session["inflight"] is not None:
                raise ParseError(f"{label}: lifecycle TURN_START order is invalid")
            if trigger == "COLD_START":
                if session["state"] != "ABSENT" or session["generation"] != 0 or row.get("cache_prompt") is not False:
                    raise ParseError(f"{label}: invalid cold start lifecycle")
                session["generation"] = 1
            elif trigger == "COLD_RESTART":
                if (session["state"] != "DEAD" or row.get("cache_prompt") is not False
                        or session["restart_pending"] != binding):
                    raise ParseError(f"{label}: invalid cold restart lifecycle")
                session["restart_pending"] = False
            elif trigger == "REVISIT":
                timer = session["timer"]
                if session["state"] != "IDLE_TTL" or timer is None or row.get("cache_prompt") is not True:
                    raise ParseError(f"{label}: invalid revisit state or cache_prompt")
                if event.get("arrival_us") is None or event["arrival_us"] >= timer["expiry_us"]:
                    raise ParseError(f"{label}: revisit is not before expiry")
                if binding != session["binding"] or event.get("lifecycle_generation") != session["generation"] or not session["revisit_pending"]:
                    raise ParseError(f"{label}: revisit changed generation or slot")
                session["timer"] = None
                session["revisit_pending"] = False
            else:
                if session["state"] != "ACTIVE" or row.get("cache_prompt") is not True:
                    raise ParseError(f"{label}: invalid continuation state or cache_prompt")
                if session["last_completion"] is None or row.get("planned_arrival_us", 0) > session["last_completion"]:
                    raise ParseError(f"{label}: continuation did not arrive during prior request")
                pending_idle.discard(sid)
            if session["binding"] is not None and trigger not in {"COLD_RESTART", "COLD_START"} and binding != session["binding"]:
                raise ParseError(f"{label}: lifecycle slot changed without restart")
            slot_id, _seq_id, runner_generation = binding
            if trigger in {"COLD_START", "COLD_RESTART"}:
                owner = slot_owners.get(slot_id)
                if owner is not None:
                    raise ParseError(f"{label}: slot {slot_id} was reused before prior DEAD")
                expected_runner_generation = slot_generations.get(slot_id, 0) + 1
                if runner_generation != expected_runner_generation:
                    raise ParseError(f"{label}: slot {slot_id} runner_generation did not advance exactly once")
                slot_generations[slot_id] = runner_generation
                slot_owners[slot_id] = sid
            elif slot_owners.get(slot_id) != sid or slot_generations.get(slot_id) != runner_generation:
                raise ParseError(f"{label}: live session slot ownership/generation drift")
            session["binding"] = binding
            if event.get("lifecycle_generation") != session["generation"]:
                raise ParseError(f"{label}: lifecycle generation mismatch")
            if event.get("arrival_us") != row.get("planned_arrival_us"):
                raise ParseError(f"{label}: lifecycle arrival timestamp differs from frozen replay arrival")
            session["inflight"] = request_id
            session["last_start"] = event.get("arrival_us")
            starts[request_id] = index
        elif kind == "TURN_COMPLETE":
            if row is None or request_id not in starts or request_id in completes or session["inflight"] != request_id:
                raise ParseError(f"{label}: lifecycle TURN_COMPLETE is missing, duplicated, or out of order")
            if event.get("completion_us") != row.get("completed_us") or event.get("turn") != row.get("turn"):
                raise ParseError(f"{label}: lifecycle completion timestamp/turn mismatch")
            if event.get("lifecycle_generation") != session["generation"] or binding != session["binding"]:
                raise ParseError(f"{label}: lifecycle completion identity mismatch")
            session["inflight"] = None
            session["state"] = "ACTIVE"
            session["last_completion"] = event.get("completion_us")
            session["next_turn"] += 1
            pending_idle.add(sid)
            completes[request_id] = index
        elif kind == "TTL_ARM":
            if sid not in pending_idle or session["inflight"] is not None or session["binding"] is None:
                raise ParseError(f"{label}: invalid TTL_ARM ordering")
            timer_id = event.get("timer_id")
            if isinstance(timer_id, bool) or not isinstance(timer_id, int) or timer_id in timer_ids:
                raise ParseError(f"{label}: invalid or duplicate timer identity")
            if event.get("completion_us") != session["last_completion"] or event.get("expiry_us") != session["last_completion"] + ttl_runtime_us:
                raise ParseError(f"{label}: TTL_ARM timing mismatch")
            if binding != session["binding"] or event.get("lifecycle_generation") != session["generation"]:
                raise ParseError(f"{label}: TTL_ARM identity mismatch")
            session["timer"] = {"timer_id": timer_id, "expiry_us": event["expiry_us"], "turn": event["turn"]}
            session["state"] = "IDLE_TTL"
            pending_idle.remove(sid)
            timer_ids.add(timer_id)
        elif kind == "REVISIT":
            timer = session["timer"]
            if session["state"] != "IDLE_TTL" or timer is None or request_id not in by_request:
                raise ParseError(f"{label}: invalid REVISIT state")
            if event.get("arrival_us") is None or event["arrival_us"] >= timer["expiry_us"]:
                raise ParseError(f"{label}: REVISIT is not before expiry")
            if event.get("request_id") != request_id or event.get("lifecycle_generation") != session["generation"] or binding != session["binding"]:
                raise ParseError(f"{label}: REVISIT identity mismatch")
            if event.get("arrival_us") != by_request[request_id].get("planned_arrival_us"):
                raise ParseError(f"{label}: REVISIT arrival differs from frozen replay arrival")
            session["revisit_pending"] = True
        elif kind == "TTL_EXPIRY":
            timer = session["timer"]
            if timer is None or session["state"] != "IDLE_TTL" or session["inflight"] is not None:
                raise ParseError(f"{label}: stale, premature, or inflight TTL_EXPIRY")
            if event.get("timer_id") != timer["timer_id"] or event.get("expiry_us") != timer["expiry_us"]:
                raise ParseError(f"{label}: stale timer identity was not ignored")
            if event.get("lifecycle_generation") != session["generation"] or binding != session["binding"]:
                raise ParseError(f"{label}: TTL_EXPIRY generation/slot mismatch")
        elif kind == "DEAD":
            timer = session["timer"]
            if timer is None or session["state"] != "IDLE_TTL" or not session["last_event"] or session["last_event"].get("event") == "TURN_COMPLETE":
                raise ParseError(f"{label}: DEAD is not after a valid TTL expiry")
            if session["last_event"].get("event") != "TTL_EXPIRY" or event.get("timer_id") != timer["timer_id"]:
                raise ParseError(f"{label}: DEAD lacks valid TTL_EXPIRY evidence")
            if event.get("erase_http_status") != 200 or event.get("erase_id_slot") != session["binding"][0]:
                raise ParseError(f"{label}: DEAD erase HTTP evidence is invalid")
            if isinstance(event.get("erase_n_erased"), bool) or not isinstance(event.get("erase_n_erased"), int) or event["erase_n_erased"] <= 0:
                raise ParseError(f"{label}: DEAD erase did not prove a non-empty prompt was cleared")
            if not isinstance(event.get("erase_body_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", event["erase_body_sha256"]):
                raise ParseError(f"{label}: DEAD erase body hash is invalid")
            timer_id = timer["timer_id"]
            expected_verify_name = f"slots_after_erase_{timer_id:06d}.json"
            if event.get("erase_verify_path") != expected_verify_name:
                raise ParseError(f"{label}: DEAD post-erase snapshot identity is invalid")
            if not isinstance(event.get("erase_verify_body_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", event["erase_verify_body_sha256"]):
                raise ParseError(f"{label}: DEAD post-erase snapshot hash is invalid")
            if run_dir is None:
                raise ParseError(f"{label}: DEAD post-erase snapshot cannot be independently verified")
            verify_snapshot = validate_slot_snapshot(
                run_dir / expected_verify_name, f"{label}.{expected_verify_name}", require_resident=False)
            if verify_snapshot.get("body_sha256") != event["erase_verify_body_sha256"]:
                raise ParseError(f"{label}: DEAD post-erase snapshot hash drift")
            validate_replay_erased_slot_snapshot(
                verify_snapshot, session["binding"][0], f"{label}.{expected_verify_name}")
            slot_id = session["binding"][0]
            if slot_owners.get(slot_id) != sid:
                raise ParseError(f"{label}: DEAD slot ownership is inconsistent")
            del slot_owners[slot_id]
            session["state"] = "DEAD"
            session["binding"] = None
            session["timer"] = None
        session["last_event"] = event
        previous_events.append(event)
    for request_id in by_request:
        if starts.get(request_id) is None or completes.get(request_id) is None:
            raise ParseError(f"{label}: every request must have exactly one TURN_START and TURN_COMPLETE")
    if pending_idle:
        raise ParseError(f"{label}: completed lifecycle turn has no TTL_ARM or continuation")
    if replay_cfg.get("lifecycle", {}).get("drain_after_last_arrival", False) and any(item["binding"] is not None for item in sessions.values()):
        raise ParseError(f"{label}: drain_after_last_arrival did not reach DEAD")
    return {"status": "PASS", "errors": [], "event_count": len(journal),
            "turn_start_count": len(starts), "turn_complete_count": len(completes),
            "dead_count": sum(1 for item in journal if item.get("event") == "DEAD"),
            "revisit_count": sum(1 for item in journal if item.get("event") == "TURN_START" and item.get("trigger") == "REVISIT"),
            "ttl_expiry_count": sum(1 for item in journal if item.get("event") == "TTL_EXPIRY")}

def _resident_preflight_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        raise ParseError("resident_preflight window has no resident samples")
    return {
        "n": len(values),
        "min": min(values),
        "median": percentile(values, 0.50),
        "max": max(values),
        "spread": max(values) - min(values),
    }


def _resident_preflight_stability(
        values: list[int], config: dict[str, Any], label: str,
        bytes_key: str, ratio_key: str) -> dict[str, Any]:
    summary = _resident_preflight_summary(values)
    allowed = max(
        int(config[bytes_key]),
        int(math.ceil(summary["max"] * config[ratio_key])),
    )
    summary["allowed_spread"] = allowed
    if summary["spread"] > allowed:
        raise ParseError(f"{label}: stability threshold exceeded")
    return summary


def _resident_preflight_interval_summary(samples: list[dict[str, Any]], config: dict[str, Any], label: str) -> dict[str, Any]:
    intervals = [
        samples[index]["timestamp_mono_ns"] - samples[index - 1]["timestamp_mono_ns"]
        for index in range(1, len(samples))
    ]
    if not intervals:
        raise ParseError(f"{label}: resident_preflight window has no sample interval")
    max_gap_ns = int(config["max_sample_gap_seconds"] * 1_000_000_000)
    if max(intervals) > max_gap_ns:
        raise ParseError(f"{label}: resident_preflight sample gap exceeds frozen max_sample_gap_seconds")
    return {
        "min_seconds": min(intervals) / 1_000_000_000,
        "median_seconds": percentile(intervals, 0.50) / 1_000_000_000,
        "max_seconds": max(intervals) / 1_000_000_000,
    }


def validate_resident_preflight_action_free(
        stderr: str, actions: list[dict[str, str]], io_records: list[dict[str, str]],
        resumes: list[dict[str, str]], timings: list[dict[str, str]], label: str) -> None:
    if actions:
        raise ParseError(f"{label}: resident_preflight observed a unified action marker")
    if resumes or timings:
        raise ParseError(f"{label}: resident_preflight observed PREFETCH/restore telemetry")
    forbidden_tokens = {
        "kv_g0_s1_resident_observation", "kv_pressure_unified_action",
        "memory_governor_async_action", "kv_resume_order_event", "kv_resume_stage_timing",
    }
    for line in stderr.splitlines():
        words = set(line.split())
        observed_forbidden = forbidden_tokens & words
        if observed_forbidden:
            token = sorted(observed_forbidden)[0]
            raise ParseError(f"{label}: resident_preflight observed state-changing telemetry: {token}")
        if "erase" in line.lower():
            raise ParseError(f"{label}: resident_preflight observed slot erase telemetry")
        if "global_dynamic" in line:
            raise ParseError(f"{label}: resident_preflight observed a global_dynamic target")
        if "memory_governor_observe" in words:
            fields = parse_fields(line, "memory_governor_observe", set(), f"{label}.observe")
            for key in (
                "kv_release_attempted", "kv_offload_attempted", "kv_release_state_changed",
                "kv_offload_state_changed", "kv_soft_budget_enabled", "kv_soft_target_bytes",
                "kv_soft_release_target_bytes", "kv_soft_offload_target_bytes",
            ):
                if key in fields and (not UINT.fullmatch(fields[key]) or int(fields[key]) != 0):
                    raise ParseError(f"{label}: resident_preflight observed a Global/action change at {key}")
    io_fields = (
        "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls",
        "backing_write_syscalls", "bytes_read", "bytes_written",
    )
    for io in io_records:
        if any(int(io[key]) != 0 for key in io_fields):
            raise ParseError(f"{label}: resident_preflight observed swap I/O")
        if any(int(io[key]) != 0 for key in RESTORE_ACTIVITY_FIELDS):
            raise ParseError(f"{label}: resident_preflight observed restore activity")


def freeze_resident_preflight_target(
        preflight: dict[str, Any], runtime_contract: dict[str, Any],
        label: str = "resident_preflight") -> dict[str, Any]:
    if not isinstance(runtime_contract, dict):
        raise ParseError(f"{label}: runtime contract is missing for target freeze")
    required = {"normalized_R_AB", "normalized_R_ABC", "Relief_A", "Relief_B", "native_block_bytes"}
    if not required.issubset(preflight):
        raise ParseError(f"{label}: normalized target inputs are incomplete")
    ab_hi = int(preflight["normalized_R_AB"]["max"])
    abc_lo = int(preflight["normalized_R_ABC"]["min"])
    abc_hi = int(preflight["normalized_R_ABC"]["max"])
    relief_a = int(preflight["Relief_A"]["resident_bytes_lower_bound"])
    relief_b = int(preflight["Relief_B"]["resident_bytes_lower_bound"])
    blocks_a = int(preflight["Relief_A"].get("resident_blocks_lower_bound", 0))
    blocks_b = int(preflight["Relief_B"].get("resident_blocks_lower_bound", 0))
    relief_lower_bound = min(relief_a, relief_b)
    action_target = int(runtime_contract["action_target_bytes"])
    max_blocks = int(runtime_contract["max_blocks"])
    alignment_quantum = int(preflight["native_block_bytes"])
    if min(ab_hi, abc_lo, abc_hi, relief_lower_bound, action_target,
           max_blocks, alignment_quantum, blocks_a, blocks_b) <= 0:
        raise ParseError(f"{label}: target-freeze inputs are non-positive or claimant block authority is missing")
    if relief_a < blocks_a * alignment_quantum or relief_b < blocks_b * alignment_quantum:
        raise ParseError(f"{label}: claimant physical relief is inconsistent with native block capacity")
    conservative_block_capacity = alignment_quantum
    max_blocks_cap = max_blocks * conservative_block_capacity
    single_action_relief_cap = min(relief_lower_bound, action_target, max_blocks_cap)
    freeze_margin = max(
        int(preflight["normalized_R_AB"].get("allowed_spread", 0)),
        int(preflight["normalized_R_ABC"].get("allowed_spread", 0)),
        alignment_quantum,
    )
    lower_raw = max(
        ab_hi + freeze_margin,
        abc_hi - single_action_relief_cap,
    )
    upper_raw = abc_lo - freeze_margin
    lower = ((lower_raw + alignment_quantum - 1) // alignment_quantum) * alignment_quantum
    upper = (upper_raw // alignment_quantum) * alignment_quantum
    formula = (
        "T=ceil_q(max(AB_hi+margin, ABC_hi-single_action_relief_cap)); "
        "single_action_relief_cap=min(Relief_lower_bound, action_target, "
        "max_blocks*native_block_bytes); require T<=floor_q(ABC_lo-margin)"
    )
    record = {
        "status": "FROZEN" if lower <= upper else "TARGET_WINDOW_UNAVAILABLE",
        "frozen": lower <= upper,
        "frozen_T": upper if lower <= upper else None,
        "safe_interval": {"lower": lower, "upper": upper},
        "inputs": {
            "AB_hi": ab_hi, "ABC_lo": abc_lo, "ABC_hi": abc_hi,
            "Relief_A_lower_bound": relief_a,
            "Relief_B_lower_bound": relief_b,
            "Relief_A_blocks_lower_bound": blocks_a,
            "Relief_B_blocks_lower_bound": blocks_b,
            "Relief_lower_bound": relief_lower_bound,
            "action_cap": action_target,
            "max_blocks": max_blocks,
            "conservative_block_capacity": conservative_block_capacity,
            "max_blocks_cap": max_blocks_cap,
            "single_action_relief_cap": single_action_relief_cap,
        },
        "alignment_quantum": alignment_quantum,
        "freeze_margin": freeze_margin,
        "formula": formula,
        "normalized_summaries": {
            "AB": preflight["normalized_R_AB"],
            "ABC": preflight["normalized_R_ABC"],
        },
    }
    if lower > upper:
        raise ParseError(f"{label}: TARGET_WINDOW_UNAVAILABLE {record}")
    return record


def resident_preflight_identity(
        artifact: pathlib.Path, manifest: dict[str, Any], spec: dict[str, Any],
        preflight: dict[str, Any], run_dir: pathlib.Path) -> dict[str, Any]:
    provenance = manifest["provenance"]
    replay = spec["workload"]["replay"]
    replay_path = pathlib.Path(replay["path"]).resolve()
    samples_path = (run_dir / preflight["sample_path"]).resolve()

    def identity(path: pathlib.Path, label: str) -> dict[str, Any]:
        if not path.is_file():
            raise ParseError(f"{label} is missing: {path}")
        return {
            "path": str(path), "present": True,
            "size": path.stat().st_size, "sha256": sha256_file(path),
        }

    if "source_lineage_id" not in replay or "alignment_family_id" not in replay:
        raise ParseError("resident_preflight replay lineage/family identity is missing")
    source_lineage_id = replay["source_lineage_id"]
    alignment_family_id = replay["alignment_family_id"]
    if isinstance(source_lineage_id, bool) or not isinstance(source_lineage_id, int) or source_lineage_id < 0:
        raise ParseError("resident_preflight source lineage identity is invalid")
    if not isinstance(alignment_family_id, str) or not alignment_family_id:
        raise ParseError("resident_preflight alignment family identity is invalid")

    preflight_fixture_identity = identity(replay_path, "resident_preflight fixture")
    source = read_json(replay_path)
    if not isinstance(source, dict):
        raise ParseError("resident_preflight replay fixture is not an object")
    derived = source.get("derived_from")
    if isinstance(derived, dict):
        parent_path_value = derived.get("transcript_path")
        parent_sha = derived.get("transcript_sha256")
        if not isinstance(parent_path_value, str) or not parent_path_value:
            raise ParseError("resident_preflight parent transcript path is missing")
        if not isinstance(parent_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", parent_sha):
            raise ParseError("resident_preflight parent transcript SHA is invalid")
        parent_path = pathlib.Path(parent_path_value).expanduser().resolve()
        parent_transcript_identity = identity(parent_path, "resident_preflight parent transcript")
        if parent_transcript_identity["sha256"] != parent_sha:
            raise ParseError("resident_preflight parent transcript identity drift")
        parent_source = read_json(parent_path)
        if not isinstance(parent_source, dict) or parent_source.get("transcript_sha256") != parent_sha:
            raise ParseError("resident_preflight parent transcript oracle identity mismatch")
    elif source.get("schema_version") == "gt-trace-1b-a/v1":
        parent_transcript_identity = dict(preflight_fixture_identity)
    else:
        raise ParseError("resident_preflight replay has no verifiable parent transcript")

    q2_evidence_identity = identity(samples_path, "resident_preflight Q2 evidence")
    return {
        "artifact_id": manifest["artifact_id"],
        "head": provenance["git"]["head"],
        "binary_sha256": provenance["binary"].get("sha256"),
        "model_sha256": provenance["model"].get("sha256"),
        "transcript_sha256": preflight_fixture_identity["sha256"],
        "fixture_sha256": q2_evidence_identity["sha256"],
        "transcript_identity": dict(preflight_fixture_identity),
        "fixture_identity": dict(q2_evidence_identity),
        "parent_transcript_identity": parent_transcript_identity,
        "preflight_fixture_identity": preflight_fixture_identity,
        "q2_evidence_identity": q2_evidence_identity,
        "source_lineage_id": source_lineage_id,
        "alignment_family_id": alignment_family_id,
        "runtime_contract": spec["runtime_contract"],
        "cgroup": spec["cgroup"],
        "executor": pathlib.Path(spec["binary"]).name,
        "object_id": preflight["object_id"],
        "generation": preflight["generation"],
    }

def validate_resident_preflight_samples(
        run_dir: pathlib.Path, record: dict[str, Any], workload: dict[str, Any],
        label: str, admission_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    expected = workload["resident_preflight"]
    if expected is None:
        raise ParseError(f"{label}: resident_preflight configuration is missing")
    record = exact(
        record,
        {"mode", "sample_interval_seconds", "min_samples", "window_timeout_seconds",
         "samples_path", "bindings", "windows"},
        f"{label}.resident_preflight",
    )
    if record["mode"] != "resident_preflight":
        raise ParseError(f"{label}: resident_preflight mode identity is invalid")
    for key in ("sample_interval_seconds", "window_timeout_seconds"):
        if record[key] != expected[key]:
            raise ParseError(f"{label}: resident_preflight {key} differs from workload")
    if record["min_samples"] != expected["min_samples"]:
        raise ParseError(f"{label}: resident_preflight min_samples differs from workload")
    sample_path = pathlib.Path(record["samples_path"])
    if sample_path.is_absolute() or ".." in sample_path.parts:
        raise ParseError(f"{label}: resident_preflight samples path escapes artifact")
    sample_file = run_dir / sample_path
    if not sample_file.is_file():
        raise ParseError(f"{label}: resident_preflight samples file is missing")
    bindings = exact(record["bindings"], {"A", "B", "C"}, f"{label}.resident_preflight.bindings")
    for sid, binding in bindings.items():
        binding = exact(binding, {"slot_id", "seq_id", "runner_generation"},
                        f"{label}.resident_preflight.binding.{sid}")
        for key in ("slot_id", "seq_id", "runner_generation"):
            require_nonnegative_int(binding[key], f"{label}.resident_preflight.binding.{sid}.{key}")
        if binding["runner_generation"] <= 0:
            raise ParseError(f"{label}: resident_preflight runner generation is invalid")
    slot_ids = [bindings[sid]["slot_id"] for sid in ("A", "B", "C")]
    seq_ids = [bindings[sid]["seq_id"] for sid in ("A", "B", "C")]
    if len(set(slot_ids)) != 3 or len(set(seq_ids)) != 3:
        raise ParseError(f"{label}: resident_preflight A/B/C slot or seq identity is not unique")
    if admission_records is not None:
        if not isinstance(admission_records, list):
            raise ParseError(f"{label}: resident_preflight admission journal is invalid")
        observed: dict[str, tuple[int, int, int]] = {}
        for index, admission in enumerate(admission_records):
            if not isinstance(admission, dict) or admission.get("logical_session_id") not in bindings:
                raise ParseError(f"{label}: resident_preflight admission[{index}] session identity is invalid")
            sid = admission["logical_session_id"]
            binding = tuple(admission.get(key) for key in ("slot_id", "seq_id", "runner_generation"))
            if any(isinstance(value, bool) or not isinstance(value, int) for value in binding):
                raise ParseError(f"{label}: resident_preflight admission[{index}] binding is invalid")
            if sid in observed and observed[sid] != binding:
                raise ParseError(f"{label}: resident_preflight admission binding drift for {sid}")
            observed[sid] = binding
        if set(observed) != set(bindings) or any(
                observed[sid] != tuple(bindings[sid][key] for key in ("slot_id", "seq_id", "runner_generation"))
                for sid in bindings):
            raise ParseError(f"{label}: resident_preflight admission-to-seq lineage mismatch")
    windows = exact(record["windows"], {"AB_RESIDENT", "ABC_ACTIVE"}, f"{label}.resident_preflight.windows")
    rows: dict[str, list[dict[str, Any]]] = {"AB_RESIDENT": [], "ABC_ACTIVE": []}
    claimant_epochs: dict[int, int] = {}
    for line_number, line in enumerate(sample_file.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseError(f"{label}: invalid resident_preflight sample line {line_number}") from exc
        row = exact(row, {"window", "timestamp_mono_ns", "sample_count", "whole", "claimants"},
                    f"{label}.resident_preflight.sample[{line_number}]")
        if row["window"] not in rows:
            raise ParseError(f"{label}: unknown resident_preflight window")
        for key in ("timestamp_mono_ns", "sample_count"):
            require_nonnegative_int(row[key], f"{label}.resident_preflight.sample.{key}")
        whole = exact(row["whole"], RESIDENT_PREFLIGHT_WHOLE_REQUIRED,
                      f"{label}.resident_preflight.sample.whole")
        for key in RESIDENT_PREFLIGHT_WHOLE_NUMERIC:
            require_nonnegative_int(whole[key], f"{label}.resident_preflight.whole.{key}")
        for key in RESIDENT_PREFLIGHT_WHOLE_BOOL:
            if not isinstance(whole[key], bool):
                raise ParseError(f"{label}: resident_preflight whole boolean is invalid")
        if not isinstance(whole["global_target_source"], str) or not whole["global_target_source"]:
            raise ParseError(f"{label}: resident_preflight Global target source is invalid")
        if (
                not whole["whole_valid"] or not whole["resident_available"]
                or not whole["reclaimable_available"]
                or not whole["swapped_metadata_consistent"]
                or not whole["observation_only"]):
            raise ParseError(f"{label}: resident_preflight whole-KV authority is unavailable")
        if (whole["object_id"] <= 0 or whole["generation"] <= 0 or whole["page_size"] <= 0
                or whole["native_block_bytes"] <= 0 or whole["total_bytes"] <= 0):
            raise ParseError(f"{label}: resident_preflight whole-KV identity is invalid")
        if whole["dead_resident_reclaimable_bytes"] > whole["resident_bytes"]:
            raise ParseError(f"{label}: dead reclaimable bytes exceed resident bytes")
        if whole["swapped_authoritative_bytes"] > whole["total_bytes"]:
            raise ParseError(f"{label}: swapped authoritative bytes exceed total bytes")
        if whole["global_target_enabled"] or whole["global_target_bytes"] != 0 or whole["global_target_source"] not in {"none", "UNAVAILABLE"}:
            raise ParseError(f"{label}: resident_preflight contains a Global dynamic target")
        claimants = row["claimants"]
        if not isinstance(claimants, dict):
            raise ParseError(f"{label}: resident_preflight claimant map is invalid")
        decoded_claimants: dict[int, dict[str, Any]] = {}
        for seq_text, value in claimants.items():
            if not isinstance(seq_text, str) or not UINT.fullmatch(seq_text):
                raise ParseError(f"{label}: resident_preflight claimant id is invalid")
            claimant = exact(value, RESIDENT_PREFLIGHT_CLAIMANT_REQUIRED,
                             f"{label}.resident_preflight.claimant")
            for key in RESIDENT_PREFLIGHT_CLAIMANT_NUMERIC:
                require_nonnegative_int(claimant[key], f"{label}.resident_preflight.claimant.{key}")
            for key in RESIDENT_PREFLIGHT_CLAIMANT_BOOL:
                if not isinstance(claimant[key], bool):
                    raise ParseError(f"{label}: resident_preflight claimant boolean is invalid")
            seq_id = int(seq_text)
            if claimant["seq_id"] != seq_id or claimant["timestamp_mono_ns"] != row["timestamp_mono_ns"] or claimant["sample_count"] != row["sample_count"]:
                raise ParseError(f"{label}: resident_preflight claimant/sample identity drift")
            if claimant["claimant_epoch"] <= 0:
                raise ParseError(f"{label}: resident_preflight claimant epoch is invalid")
            previous_epoch = claimant_epochs.setdefault(seq_id, claimant["claimant_epoch"])
            if previous_epoch != claimant["claimant_epoch"]:
                raise ParseError(f"{label}: resident_preflight claimant epoch drift")
            if claimant["object_id"] != whole["object_id"] or claimant["generation"] != whole["generation"]:
                raise ParseError(f"{label}: resident_preflight claimant physical identity drift")
            decoded_claimants[seq_id] = claimant
        for sid in ("A", "B"):
            seq_id = bindings[sid]["seq_id"]
            claimant = decoded_claimants.get(seq_id)
            if claimant is None:
                raise ParseError(f"{label}: resident_preflight {sid} claimant sample is missing")
            if not claimant["valid"] or not claimant["available"] or not claimant["authoritative"] or claimant["shared"]:
                raise ParseError(f"{label}: INVALID_FIXTURE {sid} claimant authority is unavailable/shared")
            if claimant["active"]:
                raise ParseError(f"{label}: resident_preflight {sid} is not idle")
        c_claimant = decoded_claimants.get(bindings["C"]["seq_id"])
        if row["window"] == "AB_RESIDENT":
            if c_claimant is not None and c_claimant["active"]:
                raise ParseError(f"{label}: C became active during AB_RESIDENT")
        elif (
                c_claimant is None or not c_claimant["active"]
                or not c_claimant["valid"] or not c_claimant["available"]
                or not c_claimant["authoritative"] or c_claimant["shared"]):
            raise ParseError(f"{label}: C is not an authoritative active claimant during ABC_ACTIVE")
        rows[row["window"]].append({"whole": whole, "claimants": decoded_claimants,
                                     "timestamp_mono_ns": row["timestamp_mono_ns"],
                                     "sample_count": row["sample_count"]})
    all_rows = rows["AB_RESIDENT"] + rows["ABC_ACTIVE"]
    if not all_rows:
        raise ParseError(f"{label}: resident_preflight has no samples")
    identity = {(item["whole"]["object_id"], item["whole"]["generation"]) for item in all_rows}
    if len(identity) != 1:
        raise ParseError(f"{label}: resident_preflight object/generation drift")
    for window_name, samples in rows.items():
        metadata = exact(windows[window_name], {"started_mono_ns", "finished_mono_ns", "sample_count"} | ({"dispatch_started_mono_ns"} if window_name == "ABC_ACTIVE" else set()), f"{label}.resident_preflight.{window_name}")
        if len(samples) < expected["min_samples"] or metadata["sample_count"] != len(samples):
            raise ParseError(f"{label}: resident_preflight {window_name} sample count is insufficient")
        for key in ("started_mono_ns", "finished_mono_ns"):
            require_nonnegative_int(metadata[key], f"{label}.resident_preflight.{window_name}.{key}")
        if metadata["finished_mono_ns"] < metadata["started_mono_ns"]:
            raise ParseError(f"{label}: resident_preflight {window_name} timing is invalid")
        if samples[0]["timestamp_mono_ns"] < metadata["started_mono_ns"] or samples[-1]["timestamp_mono_ns"] > metadata["finished_mono_ns"]:
            raise ParseError(f"{label}: resident_preflight {window_name} sample is outside window metadata")
        if window_name == "ABC_ACTIVE":
            require_nonnegative_int(metadata["dispatch_started_mono_ns"], f"{label}.resident_preflight.ABC_ACTIVE.dispatch_started_mono_ns")
            if samples[0]["timestamp_mono_ns"] <= metadata["dispatch_started_mono_ns"]:
                raise ParseError(f"{label}: ABC_ACTIVE sample preceded C dispatch")
        previous = None
        expected_interval_ns = int(expected["sample_interval_seconds"] * 1_000_000_000)
        for sample in samples:
            if previous is not None:
                delta_ns = sample["timestamp_mono_ns"] - previous["timestamp_mono_ns"]
                if (
                        sample["sample_count"] != previous["sample_count"] + 1
                        or delta_ns < expected_interval_ns // 2
                        or delta_ns > int(expected["max_sample_gap_seconds"] * 1_000_000_000)):
                    raise ParseError(f"{label}: resident_preflight {window_name} samples are not consecutive within the frozen gap contract")
            previous = sample
    if windows["ABC_ACTIVE"]["dispatch_started_mono_ns"] <= windows["AB_RESIDENT"]["finished_mono_ns"]:
        raise ParseError(f"{label}: ABC_ACTIVE dispatch overlaps AB_RESIDENT window")
    interval_summaries = {
        name: _resident_preflight_interval_summary(samples, expected, f"{label}: {name}")
        for name, samples in rows.items()
    }
    resident_values = {
        name: [sample["whole"]["resident_bytes"] for sample in samples]
        for name, samples in rows.items()
    }
    normalized_values = {
        name: [sample["whole"]["resident_bytes"] - sample["whole"]["dead_resident_reclaimable_bytes"] for sample in samples]
        for name, samples in rows.items()
    }
    normalized_summaries = {
        name: _resident_preflight_stability(
            values, expected, f"{label}: {name} normalized resident",
            "normalized_resident_spread_bytes", "normalized_resident_spread_ratio")
        for name, values in normalized_values.items()
    }
    combined_samples = all_rows
    exclusive = {}
    relief_values: dict[str, list[int]] = {}
    for sid in ("A", "B"):
        seq_id = bindings[sid]["seq_id"]
        values = [sample["claimants"][seq_id]["exclusive_resident_bytes"] for sample in combined_samples]
        blocks = [sample["claimants"][seq_id]["exclusive_resident_blocks"] for sample in combined_samples]
        if min(values) <= 0:
            raise ParseError(f"{label}: INVALID_FIXTURE {sid} has no positive exclusive resident bytes")
        relief_values[sid] = values
        exclusive[sid] = {"resident_bytes_lower_bound": min(values), "resident_blocks_lower_bound": min(blocks)}
    relief_summaries = {
        sid: _resident_preflight_stability(
            values, expected, f"{label}: {sid} claimant relief",
            "claimant_relief_spread_bytes", "claimant_relief_spread_ratio")
        for sid, values in relief_values.items()
    }
    combined_bytes = [
        sample["claimants"][bindings["A"]["seq_id"]]["exclusive_resident_bytes"]
        + sample["claimants"][bindings["B"]["seq_id"]]["exclusive_resident_bytes"]
        for sample in combined_samples
    ]
    combined_blocks = [
        sample["claimants"][bindings["A"]["seq_id"]]["exclusive_resident_blocks"]
        + sample["claimants"][bindings["B"]["seq_id"]]["exclusive_resident_blocks"]
        for sample in combined_samples
    ]
    return {
        "status": "PASS",
        "sample_path": str(sample_path),
        "R_AB": _resident_preflight_summary(resident_values["AB_RESIDENT"]),
        "R_ABC": _resident_preflight_summary(resident_values["ABC_ACTIVE"]),
        "normalized_R_AB": normalized_summaries["AB_RESIDENT"],
        "normalized_R_ABC": normalized_summaries["ABC_ACTIVE"],
        "intervals": interval_summaries,
        "exclusive_estimate": {
            "authority": "claimant_physical_view",
            "A": exclusive["A"], "B": exclusive["B"],
            "combined_resident_bytes_lower_bound": min(combined_bytes),
            "combined_resident_blocks_lower_bound": min(combined_blocks),
        },
        "Relief_A": exclusive["A"],
        "Relief_B": exclusive["B"],
        "combined_exclusive_resident_bytes_lower_bound": min(combined_bytes),
        "combined_exclusive_resident_blocks_lower_bound": min(combined_blocks),
        "object_id": next(iter(identity))[0], "generation": next(iter(identity))[1],
        "page_size": all_rows[0]["whole"]["page_size"],
        "native_block_bytes": all_rows[0]["whole"]["native_block_bytes"],
        "windows": {name: {"sample_count": len(samples), "first_sample_count": samples[0]["sample_count"],
                            "last_sample_count": samples[-1]["sample_count"]}
                    for name, samples in rows.items()},
    }


def parse_replay_run(
        artifact: pathlib.Path,
        plan: dict[str, Any],
        case: dict[str, Any],
        workload: dict[str, Any],
        expected_execution_index: int,
        spec: dict[str, Any]) -> dict[str, Any]:
    label = plan["run_id"]
    run_dir = artifact / "runs" / label
    run = exact(read_json(run_dir / "run.json"), RUN_KEYS, f"{label}.run", {"replay"})
    for key in ("run_id", "round", "run_order", "case_id"):
        if run[key] != plan[key]:
            raise ParseError(f"{label}: replay run identity mismatch at {key}")
    if run["execution_index"] != expected_execution_index or run["case"] != plan:
        raise ParseError(f"{label}: replay run execution/case identity mismatch")
    replay_cfg = workload["replay"]
    try:
        replay_plan = load_replay(replay_cfg["source"], replay_cfg["path"],
                                  n_parallel=replay_cfg["n_parallel"],
                                  time_dilation=replay_cfg["time_dilation"],
                                  selected_session_ids=replay_cfg["session_ids"],
                                  lifecycle=replay_cfg.get("lifecycle", {}).get("enabled", False))
    except (OSError, ReplayError) as exc:
        raise ParseError(f"{label}: replay source cannot be reloaded: {exc}") from exc
    if replay_cfg.get("lifecycle", {}).get("enabled", False):
        source_lifecycle = replay_plan.lifecycle or {}
        lifecycle_cfg = replay_cfg["lifecycle"]
        if "ttl_seconds" in lifecycle_cfg and abs(float(lifecycle_cfg["ttl_seconds"]) - float(source_lifecycle["ttl_seconds"])) > 1e-9:
            raise ParseError(f"{label}: lifecycle TTL differs from source identity")
        if "parent_manifest_path" in lifecycle_cfg and pathlib.Path(lifecycle_cfg["parent_manifest_path"]).resolve() != pathlib.Path(source_lifecycle.get("parent_manifest_path", "")).resolve():
            raise ParseError(f"{label}: lifecycle parent manifest path mismatch")
        if "parent_manifest_sha256" in lifecycle_cfg and lifecycle_cfg["parent_manifest_sha256"] != source_lifecycle.get("parent_manifest_sha256"):
            raise ParseError(f"{label}: lifecycle parent manifest SHA mismatch")
        if "trace_identity" in lifecycle_cfg and lifecycle_cfg["trace_identity"] != source_lifecycle.get("trace_identity"):
            raise ParseError(f"{label}: lifecycle trace identity mismatch")
    expected_schedule = expand_schedule(replay_plan)
    if run.get("request_plan") != expected_schedule:
        raise ParseError(f"{label}: replay request schedule drift")
    execution = exact(
        read_json(run_dir / "execution.json"), EXECUTION_KEYS, f"{label}.execution",
        {"replay", "lifecycle", "resident_preflight"})
    if any(execution[key] != plan[key] for key in ("run_id", "round", "run_order", "case_id")):
        raise ParseError(f"{label}: replay execution identity mismatch")
    if execution["execution_index"] != expected_execution_index or execution["run_mode"] not in {"qualification", "resident_preflight"}:
        raise ParseError(f"{label}: replay execution mode/order mismatch")
    validate_execution_environment(execution, case, label, spec)
    if replay_cfg.get("lifecycle", {}).get("enabled", False):
        execution_lifecycle = execution.get("lifecycle")
        if not isinstance(execution_lifecycle, dict) or execution_lifecycle.get("enabled") is not True:
            raise ParseError(f"{label}: replay execution lifecycle identity is missing")
    expected_argv = [spec["binary"], "--host", "127.0.0.1", "--port", execution["argv"][execution["argv"].index("--port") + 1],
                     "--model", spec["model"], *spec["server_args"], "--parallel", str(replay_cfg["n_parallel"]),
                     "--no-cache-idle-slots", "--no-context-shift"]
    if replay_cfg.get("lifecycle", {}).get("enabled", False):
        expected_argv.extend(["--slot-save-path", str(run_dir / "slot-cache")])
        if execution.get("lifecycle", {}).get("slot_save_path") != str(run_dir / "slot-cache"):
            raise ParseError(f"{label}: lifecycle slot-save path identity mismatch")
    if execution["argv"] != expected_argv:
        raise ParseError(f"{label}: replay server argv does not match canonical multi-session options")
    validate_process_identity(execution["server_identity"], execution["argv"], f"{label}.server_identity")
    sampler_argv = execution["sampler_argv"]
    validate_process_identity(execution["sampler_identity"], sampler_argv, f"{label}.sampler_identity")
    if execution["request_loop_started"] is not True or execution["request_count"] != len(expected_schedule):
        raise ParseError(f"{label}: replay request lifecycle/count mismatch")
    cleanup = exact(read_json(run_dir / "cleanup.json"), CLEANUP_KEYS, f"{label}.cleanup")
    server_cleanup = validate_cleanup_record(cleanup["server"], f"{label}.cleanup.server")
    sampler_cleanup = validate_cleanup_record(cleanup["sampler"], f"{label}.cleanup.sampler")
    if cleanup["residual_process"] or not cleanup["cleanup_complete"] or server_cleanup["exit_code"] != 0 or sampler_cleanup["exit_code"] != 0:
        raise ParseError(f"{label}: replay cleanup is incomplete")
    if not (run_dir / "memory_samples.tsv").is_file():
        raise ParseError(f"{label}: replay memory samples missing")
    validate_sampler(run_dir / "memory_samples.tsv", execution["server_identity"], label)
    slots_before = validate_slot_snapshot(
        run_dir / "slots_before.json", f"{label}.slots_before", require_resident=False)
    slots = slots_before["body_json"]
    if not isinstance(slots, list):
        raise ParseError(f"{label}: replay /slots preflight body is unavailable")
    slot_map = {
        slot.get("id"): slot
        for slot in slots
        if isinstance(slot, dict)
        and isinstance(slot.get("id"), int)
        and not isinstance(slot.get("id"), bool)
    }
    required_ids = set(range(replay_cfg["n_parallel"]))
    if not required_ids.issubset(slot_map):
        raise ParseError(f"{label}: replay /slots does not expose all requested slots")
    max_required = max(
        len(event.prompt_tokens) + event.n_predict for event in replay_plan.events)
    for slot_id in sorted(required_ids):
        n_ctx = slot_map[slot_id].get("n_ctx")
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx < max_required:
            raise ParseError(
                f"{label}: replay slot {slot_id} context capacity is insufficient")
    responses_path = run_dir / "responses.jsonl"
    if not responses_path.is_file():
        raise ParseError(f"{label}: replay responses missing")
    actual: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(responses_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParseError(f"{label}: invalid replay response line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ParseError(f"{label}: replay response line {line_number} is not an object")
        required = {"event_seq", "logical_session_id", "lineage_id", "turn", "planned_ts_us", "planned_arrival_us",
                    "prompt_tokens", "prompt_sha256", "prompt_token_count", "n_predict", "request_id", "slot_id", "seq_id",
                    "source", "source_lineage_id", "source_turn", "phase_id", "reference_completion_tokens",
                    "reference_completion_sha256", "runner_generation", "cache_prompt", "claimant_epoch",
                    "physical_object_id", "physical_generation", "expected_live_kv_cells", "dispatch_order",
                    "admitted_us", "dispatched_us",
                    "completed_us", "arrival_lag_us", "admission_wait_us", "service_us", "started_mono_ns",
                    "started_us", "finished_mono_ns", "http_status", "headers", "request_sha256", "body_path",
                    "body_bytes", "body_sha256", "response_json", "response_tokens", "response_slot_id",
                    "tokens_evaluated", "tokens_predicted", "live_kv_pos_min", "live_kv_pos_max",
                    "live_kv_cells", "live_kv_blocks", "live_kv_object_id", "live_kv_generation",
                    "live_kv_block_aligned", "live_kv_authoritative", "live_kv_shared",
                    "response_token_sha256", "response_token_count", "error"}
        if replay_cfg.get("lifecycle", {}).get("enabled", False):
            required.update({"lifecycle_trigger", "lifecycle_generation"})
        if set(row) != required:
            raise ParseError(f"{label}: replay response schema mismatch at line {line_number}")
        if row["request_id"] in seen:
            raise ParseError(f"{label}: duplicate replay request_id {row['request_id']}")
        seen.add(row["request_id"])
        slot = row["slot_id"]
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < replay_cfg["n_parallel"] or row["seq_id"] != slot:
            raise ParseError(f"{label}: replay slot/seq binding is invalid")
        if row["http_status"] != 200 or row["response_slot_id"] != slot:
            raise ParseError(f"{label}: replay completion slot/status mismatch")
        if row["prompt_token_count"] != len(row["prompt_tokens"]):
            raise ParseError(f"{label}: replay prompt token count mismatch")
        if replay_cfg.get("lifecycle", {}).get("enabled", False):
            expected_cache = row.get("lifecycle_trigger") in {"CONTINUATION", "REVISIT"}
        else:
            expected_cache = row["turn"] > 1
        if row["cache_prompt"] is not expected_cache:
            raise ParseError(f"{label}: replay cache_prompt lifecycle mismatch")
        if isinstance(row["runner_generation"], bool) or not isinstance(row["runner_generation"], int) or row["runner_generation"] <= 0:
            raise ParseError(f"{label}: replay runner generation is invalid")
        for key in ("claimant_epoch", "physical_object_id", "physical_generation"):
            value = row[key]
            if value != "UNAVAILABLE" and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ParseError(f"{label}: replay {key} is neither a valid telemetry value nor UNAVAILABLE")
        live_keys = (
            "live_kv_pos_min", "live_kv_pos_max", "live_kv_cells", "live_kv_blocks",
            "live_kv_object_id", "live_kv_generation", "live_kv_block_aligned",
            "live_kv_authoritative", "live_kv_shared",
        )
        live_values = [row[key] for key in live_keys]
        if any(value is not None for value in live_values):
            if any(value is None for value in live_values):
                raise ParseError(f"{label}: replay live-KV telemetry is incomplete")
            if any(isinstance(row[key], bool) or not isinstance(row[key], int)
                   for key in live_keys[:6]):
                raise ParseError(f"{label}: replay live-KV numeric telemetry is invalid")
            if any(row[key] < 0 for key in live_keys[:2]) or any(
                    row[key] <= 0 for key in live_keys[2:6]):
                raise ParseError(f"{label}: replay live-KV numeric telemetry is out of range")
            if row["live_kv_pos_max"] < row["live_kv_pos_min"]:
                raise ParseError(f"{label}: replay live-KV position range is invalid")
            if any(not isinstance(row[key], bool)
                   for key in live_keys[6:]):
                raise ParseError(f"{label}: replay live-KV boolean telemetry is invalid")
            if not row["live_kv_block_aligned"] or not row["live_kv_authoritative"] \
                    or row["live_kv_shared"]:
                raise ParseError(f"{label}: replay live-KV authority is not exclusive and aligned")
            runtime_contract = spec.get("runtime_contract")
            quantum = runtime_contract.get("paged_block_size", 16) \
                if isinstance(runtime_contract, dict) else 16
            if row["live_kv_cells"] != row["live_kv_blocks"] * quantum:
                raise ParseError(f"{label}: replay live-KV cells do not fill paged blocks")
            expected_cells = row["expected_live_kv_cells"]
            if expected_cells is not None and (
                    isinstance(expected_cells, bool) or not isinstance(expected_cells, int)
                    or expected_cells <= 0 or row["live_kv_cells"] != expected_cells):
                raise ParseError(f"{label}: replay live-KV cells differ from expected footprint")
        elif row["expected_live_kv_cells"] is not None:
            raise ParseError(f"{label}: expected live-KV footprint lacks telemetry")
        if isinstance(row["dispatch_order"], bool) or not isinstance(row["dispatch_order"], int) or row["dispatch_order"] < 0:
            raise ParseError(f"{label}: replay dispatch order is invalid")
        expected_request = {
            "prompt": row["prompt_tokens"], "n_predict": row["n_predict"], "seed": 0,
            "temperature": 0.0, "top_k": 1, "top_p": 1.0,
            "cache_prompt": row["cache_prompt"], "return_tokens": True,
            "ignore_eos": True, "stream": False, "id_slot": slot,
        }
        expected_request_sha = sha256_bytes(json.dumps(expected_request, ensure_ascii=False,
                                                       separators=(",", ":"), sort_keys=True).encode("utf-8"))
        if row["request_sha256"] != expected_request_sha:
            raise ParseError(f"{label}: replay request identity mismatch")
        if row["prompt_sha256"] != sha256_bytes(json.dumps(row["prompt_tokens"], separators=(",", ":")).encode("utf-8")):
            raise ParseError(f"{label}: replay prompt SHA mismatch")
        payload = row["response_json"]
        if not isinstance(payload, dict) or payload.get("id_slot") != slot:
            raise ParseError(f"{label}: replay response id_slot mismatch")
        tokens = payload.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != row["n_predict"] or row["tokens_predicted"] != row["n_predict"] or row["tokens_evaluated"] != len(row["prompt_tokens"]):
            raise ParseError(f"{label}: replay token conservation mismatch")
        token_sha = sha256_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))
        if row["response_tokens"] != tokens or row["response_token_count"] != len(tokens):
            raise ParseError(f"{label}: replay response token payload/count mismatch")
        if row["response_token_sha256"] != token_sha:
            raise ParseError(f"{label}: replay response token SHA mismatch")
        expected_tokens = row.get("reference_completion_tokens", [])
        expected_sha = row.get("reference_completion_sha256", "")
        if expected_tokens and (tokens != expected_tokens or expected_sha != token_sha):
            raise ParseError(f"{label}: MODEL_BOUND_REAL reference completion oracle mismatch")
        body_path = pathlib.Path(row["body_path"])
        if body_path.is_absolute() or ".." in body_path.parts:
            raise ParseError(f"{label}: replay response body escapes artifact")
        body = run_dir / body_path
        if not body.is_file() or body.stat().st_size != row["body_bytes"] or sha256_file(body) != row["body_sha256"]:
            raise ParseError(f"{label}: replay response body identity mismatch")
        for key in ("admitted_us", "dispatched_us", "completed_us", "arrival_lag_us",
                    "admission_wait_us", "service_us", "started_us"):
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ParseError(f"{label}: replay {key} is invalid")
        if row["completed_us"] < row["dispatched_us"] or row["dispatched_us"] < row["admitted_us"]:
            raise ParseError(f"{label}: replay timestamps are not monotonic")
        if row["dispatched_us"] != row["started_us"]:
            raise ParseError(f"{label}: replay dispatched/start timestamp mismatch")
        if row["arrival_lag_us"] != max(0, row["dispatched_us"] - row["planned_arrival_us"]):
            raise ParseError(f"{label}: replay arrival_lag_us is inconsistent")
        if row["admission_wait_us"] != max(0, row["admitted_us"] - row["planned_arrival_us"]):
            raise ParseError(f"{label}: replay admission_wait_us is inconsistent")
        if row["service_us"] != row["completed_us"] - row["dispatched_us"]:
            raise ParseError(f"{label}: replay service_us is inconsistent")
        if replay_cfg.get("lifecycle", {}).get("enabled", False):
            if row["lifecycle_trigger"] not in {"COLD_START", "CONTINUATION", "REVISIT", "COLD_RESTART"}:
                raise ParseError(f"{label}: replay lifecycle trigger is invalid")
            if isinstance(row["lifecycle_generation"], bool) or not isinstance(row["lifecycle_generation"], int) or row["lifecycle_generation"] <= 0:
                raise ParseError(f"{label}: replay lifecycle generation is invalid")
        actual.append(row)
    if len(actual) != len(expected_schedule):
        raise ParseError(f"{label}: replay response count mismatch")
    lifecycle_records: list[dict[str, Any]] = []
    if replay_cfg.get("lifecycle", {}).get("enabled", False):
        lifecycle_path = run_dir / "lifecycle.jsonl"
        if not lifecycle_path.is_file():
            raise ParseError(f"{label}: lifecycle journal is missing")
        for line_number, line in enumerate(lifecycle_path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParseError(f"{label}: invalid lifecycle journal line {line_number}: {exc}") from exc
            if not isinstance(event, dict):
                raise ParseError(f"{label}: lifecycle journal line {line_number} is not an object")
            lifecycle_records.append(event)
    lifecycle_verdict = validate_replay_lifecycle(
        label, replay_plan, replay_cfg, actual, lifecycle_records, run_dir=run_dir)
    fidelity = check_fidelity(replay_plan, actual, n_parallel=replay_cfg["n_parallel"])
    fidelity["lifecycle"] = lifecycle_verdict
    if fidelity["status"] != "PASS":
        raise ParseError(f"{label}: workload_fidelity failed: {fidelity['errors']}")
    replay_artifact = exact(
        read_json(run_dir / "replay.json"),
        {"plan", "schedule", "admission", "events", "workload_fidelity"},
        f"{label}.replay", {"lifecycle", "phase_evidence", "resident_preflight"})
    admitted_sessions = {item.get("logical_session_id") for item in replay_artifact["admission"] if item.get("status") in {"admitted", "already_live", "finished"}}
    planned_sessions = {item["logical_session_id"] for item in expected_schedule}
    if admitted_sessions != planned_sessions:
        raise ParseError(f"{label}: planned sessions differ from admitted sessions")
    if replay_artifact["plan"] != replay_plan.to_dict():
        raise ParseError(f"{label}: replay plan identity drift")
    if replay_artifact["schedule"] != expected_schedule or replay_artifact["workload_fidelity"]["status"] != "PASS":
        raise ParseError(f"{label}: replay artifact schedule/fidelity record drift")
    if replay_artifact["events"] != actual:
        raise ParseError(f"{label}: replay artifact event journal differs from responses authority")
    if replay_cfg.get("lifecycle", {}).get("enabled", False):
        if replay_artifact.get("lifecycle") != lifecycle_records:
            raise ParseError(f"{label}: replay lifecycle journal differs from lifecycle.jsonl")
    if execution.get("replay") != replay_artifact:
        raise ParseError(f"{label}: execution replay journal differs from replay.json")
    if spec["run_mode"] == "resident_preflight":
        if replay_plan.fixture_contract is not None:
            raise ParseError(f"{label}: resident_preflight reused a Q1 phase contract")
        stderr = (run_dir / "server.stderr").read_text(encoding="utf-8", errors="replace")
        actions = marker_records(stderr, "kv_pressure_unified_action", ACTION_REQUIRED, f"{label}.action")
        io_records = marker_records(stderr, "KV_PAGED_IO_STATS", IO_REQUIRED, f"{label}.io")
        if not io_records:
            raise ParseError(f"{label}: resident_preflight is missing KV_PAGED_IO_STATS")
        for io in io_records:
            numeric_fields(io, IO_REQUIRED, f"{label}.io")
            boolean_fields(io, IO_BOOL, f"{label}.io")
        resumes = marker_records(stderr, "kv_resume_order_event", RESUME_REQUIRED, f"{label}.resume")
        timings = marker_records(stderr, "kv_resume_stage_timing", TIMING_REQUIRED, f"{label}.timing")
        validate_resident_preflight_action_free(stderr, actions, io_records, resumes, timings, label)
        preflight = validate_resident_preflight_samples(
            run_dir, execution.get("resident_preflight"), workload, label,
            replay_artifact["admission"])
        replay_preflight = exact(
            replay_artifact.get("resident_preflight"),
            {"samples_path", "windows", "bindings"}, f"{label}.replay.resident_preflight")
        execution_preflight = exact(
            execution.get("resident_preflight"),
            {"mode", "sample_interval_seconds", "min_samples", "window_timeout_seconds", "samples_path", "bindings", "windows"},
            f"{label}.execution.resident_preflight")
        if replay_preflight["samples_path"] != preflight["sample_path"] or replay_preflight["windows"] != execution_preflight["windows"] or replay_preflight["bindings"] != execution_preflight["bindings"]:
            raise ParseError(f"{label}: resident_preflight replay journal drift")
        return {"run_id": label, "case_id": plan["case_id"], "round": plan["round"], "policy": plan["policy"],
                "server_exit_code": server_cleanup["exit_code"], "sampler_exit_code": sampler_cleanup["exit_code"],
                "samples": validate_sampler(run_dir / "memory_samples.tsv", execution["server_identity"], label),
                "memory": memory_statistics(validate_sampler(run_dir / "memory_samples.tsv", execution["server_identity"], label)),
                "slots_before": slots_before, "slots_after": validate_slot_snapshot(run_dir / "slots_after.json", f"{label}.slots_after", require_resident=False),
                "responses": actual, "service_failures": [], "actions": [], "resident_observations": [],
                "qualified_offload_pairs": [], "offload_physical_relief_bytes": 0,
                "qualification_round_trip": None, "characterization": None,
                "resume_events": [], "resume_timings": [], "io": io_records[-1],
                "statistics": response_statistics(actual), "workload_fidelity": fidelity,
                "replay": replay_artifact, "resident_preflight": preflight,
                "R_AB": preflight["R_AB"], "R_ABC": preflight["R_ABC"],
                "exclusive_estimate": preflight["exclusive_estimate"],
                "Relief_A": preflight["Relief_A"], "Relief_B": preflight["Relief_B"],
                "combined_exclusive_resident_bytes_lower_bound": preflight["combined_exclusive_resident_bytes_lower_bound"],
                "combined_exclusive_resident_blocks_lower_bound": preflight["combined_exclusive_resident_blocks_lower_bound"]}
    if replay_plan.fixture_contract is None:
        return {"run_id": label, "case_id": plan["case_id"], "round": plan["round"], "policy": plan["policy"],
                "server_exit_code": server_cleanup["exit_code"], "sampler_exit_code": sampler_cleanup["exit_code"],
                "samples": validate_sampler(run_dir / "memory_samples.tsv", execution["server_identity"], label),
                "memory": memory_statistics(validate_sampler(run_dir / "memory_samples.tsv", execution["server_identity"], label)),
                "slots_before": slots_before, "slots_after": validate_slot_snapshot(run_dir / "slots_after.json", f"{label}.slots_after", require_resident=False),
                "responses": actual, "service_failures": [], "actions": [], "resident_observations": [],
                "qualified_offload_pairs": [], "offload_physical_relief_bytes": 0,
                "qualification_round_trip": None, "characterization": None,
                "resume_events": [], "resume_timings": [], "io": {},
                "statistics": {}, "workload_fidelity": fidelity,
                "replay": replay_artifact}
    phase_evidence = validate_replay_phase_evidence(
        execution.get("qualification"), replay_plan, actual, replay_artifact,
        run_dir, case, spec, label)
    raw_phase_evidence = {
        "phase_windows": phase_evidence["phase_windows"],
        "erase_actions": phase_evidence["erase_actions"],
    }
    if replay_artifact.get("phase_evidence") != raw_phase_evidence:
        raise ParseError(f"{label}: replay phase evidence differs from parser recomputation")
    if execution.get("replay", {}).get("phase_evidence") != raw_phase_evidence:
        raise ParseError(f"{label}: execution replay phase evidence differs from parser recomputation")
    return {"run_id": label, "case_id": plan["case_id"], "round": plan["round"], "policy": plan["policy"],
            "server_exit_code": server_cleanup["exit_code"], "sampler_exit_code": sampler_cleanup["exit_code"],
            "samples": validate_sampler(run_dir / "memory_samples.tsv", execution["server_identity"], label),
            "memory": memory_statistics(validate_sampler(run_dir / "memory_samples.tsv", execution["server_identity"], label)),
            "slots_before": slots_before, "slots_after": validate_slot_snapshot(run_dir / "slots_after.json", f"{label}.slots_after", require_resident=False),
            "responses": actual, "service_failures": [], "actions": phase_evidence["actions"],
            "resident_observations": phase_evidence["resident_observations"],
            "qualified_offload_pairs": phase_evidence["qualified_offload_pairs"],
            "offload_physical_relief_bytes": phase_evidence["offload_physical_relief_bytes"],
            "qualification_round_trip": phase_evidence["qualification_round_trip"], "characterization": None,
            "resume_events": phase_evidence["resume_events"], "resume_timings": phase_evidence["resume_timings"],
            "io": phase_evidence["io"], "statistics": response_statistics(actual),
            "workload_fidelity": fidelity, "replay": replay_artifact,
            "phase_evidence": phase_evidence}


def parse_run(artifact: pathlib.Path, plan: dict[str, Any], case: dict[str, Any], workload: dict[str, Any], expected_execution_index: int, spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = artifact / "runs" / plan["run_id"]
    label = f"{plan['run_id']}"
    if not run_dir.is_dir():
        raise ParseError(f"{label}: run directory missing")
    run = exact(read_json(run_dir / "run.json"), RUN_KEYS, f"{label}.run", {"replay"})
    for key in ("run_id", "round", "run_order", "case_id"):
        if run[key] != plan[key]:
            raise ParseError(f"{label}: run identity mismatch at {key}")
    if run["execution_index"] != expected_execution_index:
        raise ParseError(f"{label}: declared execution index mismatch")
    if run["case"] != plan:
        raise ParseError(f"{label}: embedded case differs from planned factor")
    if "replay" in workload:
        return parse_replay_run(artifact, plan, case, workload, expected_execution_index, spec)
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
        case["policy"] in SWAP_POLICIES
        and (
            spec["run_mode"] == "qualification"
            or (
                characterization["settle"] is not None
                and characterization["settle"]["status"] == "unmet_floor"
            )
        )
    )
    # Run-level slots_before / slots_after are lifecycle bookends, not physical
    # metric authorities.  slots_before is captured immediately after wait_health()
    # and before any warmup, so a real server's resident physical sample may not
    # yet be established at that instant (status=unavailable is legitimate).  The
    # characterization physical authorities are slots_after_fill / slots_settled /
    # slots_release_settled / slots_after_measurement (validated with their own
    # strict require_resident gates below) plus the transaction-local mincore
    # qualification evidence.  Requiring a physical resident observation in the
    # startup slots_before would therefore false-fail a genuine artifact with no
    # evidence value of its own, so slots_before is always relaxed: the snapshot
    # still passes the full HTTP/raw-body/hash/schema identity checks; only the
    # "must already observe a physical resident" requirement is dropped.
    slots_before = validate_slot_snapshot(
        run_dir / "slots_before.json",
        f"{label}.slots_before",
        require_resident=False,
    )
    # slots_after covers the post-run bookend and is treated symmetrically with
    # the existing exception for V2 qualification / unmet_floor runs, where a
    # missing terminal physical sample is a legitimate completion shape.  In all
    # other runs slots_after is plentiful (Resident / target-reached /
    # release-settled artifacts carry a real resident at run end) so require it.
    slots_after = validate_slot_snapshot(
        run_dir / "slots_after.json",
        f"{label}.slots_after",
        require_resident=not allow_missing_v2_run_resident,
    )
    # The former symmetric "slots_before-resident is None == slots_after-resident
    # is None" requirement encoded an artificial bookend parity between two
    # lifecycle moments (startup vs. post-completion) that the real server
    # startup sequence does not guarantee and the characterization physical
    # authorities above (slots_after_fill / slots_settled / slots_release_settled
    # / slots_after_measurement plus transaction-local mincore) do not depend on,
    # so it is removed rather than enforced when allow_missing_v2_run_resident.
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
        run_dir / "responses.jsonl", run_dir / "raw", request_plan, label,
        require_completion_identity=spec.get("profile") == FINAL_F16_PROFILE)
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
        if case["policy"] in SWAP_POLICIES:
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
    if case["policy"] in SWAP_POLICIES:
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
    if not (case["policy"] in SWAP_POLICIES and spec["run_mode"] == "qualification") \
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
        **({"role": plan["role"]} if spec.get("profile") == FINAL_F16_PROFILE else {}),
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
        "offload_physical_relief_bytes": offload_physical_relief_from_pairs(qualified_pairs),
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


def validate_replay_model_binding(
        model_binding: Any, artifact_model_sha: Any, artifact_binary_sha: Any,
        runtime_contract: Any = None) -> None:
    """Bind replay to the transcript model while retaining binary provenance validation."""
    if not isinstance(model_binding, dict):
        raise ParseError("replay transcript model identity is missing")
    transcript_model_sha = model_binding.get("model_sha256")
    transcript_binary_sha = model_binding.get("binary_sha256")
    if not isinstance(transcript_model_sha, str) or re.fullmatch(r"[0-9a-f]{64}", transcript_model_sha) is None:
        raise ParseError("replay transcript model SHA is invalid")
    if not isinstance(transcript_binary_sha, str) or re.fullmatch(r"[0-9a-f]{64}", transcript_binary_sha) is None:
        raise ParseError("replay transcript binary SHA is invalid")
    if transcript_model_sha != artifact_model_sha:
        raise ParseError("replay transcript model SHA differs from artifact model identity")
    if transcript_binary_sha != artifact_binary_sha:
        raise ParseError("replay transcript binary SHA differs from artifact binary identity")
    transcript_runtime_contract = model_binding.get("runtime_contract")
    if not isinstance(transcript_runtime_contract, dict) or set(transcript_runtime_contract) != Q2Q3_RUNTIME_CONTRACT_KEYS:
        raise ParseError("replay transcript runtime contract is missing or malformed")
    if runtime_contract is not None and transcript_runtime_contract != runtime_contract:
        raise ParseError("replay transcript runtime contract differs from spec runtime contract")


def validate_derived_replay_parent(
        source: dict[str, Any], artifact_model_sha: Any,
        artifact_binary_sha: Any, label: str,
        runtime_contract: Any = None) -> None:
    derived = source.get("derived_from")
    required_derived = {
        "transcript_path", "transcript_sha256", "parent_file_sha256",
        "source_lineage_id", "alignment_family_id",
        "q2_preflight_fixture_identity", "q2_evidence_identity",
    }
    if not isinstance(derived, dict) or set(derived) != required_derived:
        raise ParseError(f"{label}: derived_from provenance identity schema is invalid")
    verify_identity(derived["q2_preflight_fixture_identity"], f"{label}.q2_preflight_fixture_identity")
    verify_identity(derived["q2_evidence_identity"], f"{label}.q2_evidence_identity")
    source_lineage_id = derived["source_lineage_id"]
    if isinstance(source_lineage_id, bool) or not isinstance(source_lineage_id, int) or source_lineage_id < 0:
        raise ParseError(f"{label}: source lineage identity is invalid")
    alignment_family_id = derived["alignment_family_id"]
    if not isinstance(alignment_family_id, str) or not alignment_family_id:
        raise ParseError(f"{label}: alignment family identity is invalid")
    parent_path_value = derived.get("transcript_path")
    if not isinstance(parent_path_value, str) or not parent_path_value:
        raise ParseError(f"{label}: parent transcript path is invalid")
    parent_path = pathlib.Path(parent_path_value).expanduser().resolve()
    parent = read_json(parent_path)
    if not isinstance(parent, dict):
        raise ParseError(f"{label}: parent transcript is not an object")
    parent_file_sha = sha256_file(parent_path)
    if derived["parent_file_sha256"] != parent_file_sha:
        raise ParseError(f"{label}: parent file SHA differs from derived identity")
    parent_identity = derived["q2_preflight_fixture_identity"]
    if pathlib.Path(parent_identity["path"]).resolve() == parent_path:
        raise ParseError(f"{label}: Q2 preflight fixture must remain distinct from parent transcript")
    transcript_sha = derived.get("transcript_sha256")
    if not isinstance(transcript_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", transcript_sha):
        raise ParseError(f"{label}: parent transcript SHA is invalid")
    if parent.get("transcript_sha256") != transcript_sha:
        raise ParseError(f"{label}: parent transcript oracle SHA mismatch")
    try:
        from trace_compiler.runtime_materialize import load_transcript
        load_transcript(parent_path)
    except Exception as exc:
        raise ParseError(f"{label}: parent transcript validation failed: {exc}") from exc
    if parent.get("schema_version") != "gt-trace-1b-a/v1":
        raise ParseError(f"{label}: parent transcript schema is unsupported")
    if parent.get("materialization_status") != "MODEL_BOUND_REAL" or parent.get("materialize_mode") != "direct_token_ids":
        raise ParseError(f"{label}: parent transcript is not MODEL_BOUND_REAL direct_token_ids")
    reference_pass = parent.get("reference_pass")
    if not isinstance(reference_pass, dict) or reference_pass.get("completed") is not True:
        raise ParseError(f"{label}: parent reference completion is incomplete")
    if reference_pass.get("reference_completion_role") != "qualification_observation_only":
        raise ParseError(f"{label}: parent reference completion role is invalid")
    if reference_pass.get("formal_correctness_oracle") != "future_resident_multi_session_baseline":
        raise ParseError(f"{label}: parent correctness oracle is invalid")
    turns = parent.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ParseError(f"{label}: parent transcript has no turns")
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, dict):
            raise ParseError(f"{label}: parent turn {index} is not an object")
        completion = turn.get("reference_completion_tokens")
        completion_sha = turn.get("reference_completion_sha256")
        if not isinstance(completion, list) or not isinstance(completion_sha, str):
            raise ParseError(f"{label}: parent turn {index} lacks completion oracle")
        if turn.get("n_predict") != len(completion) or completion_sha != sha256_bytes(
                json.dumps(completion, separators=(",", ":")).encode("utf-8")):
            raise ParseError(f"{label}: parent turn {index} completion oracle mismatch")
    model = parent.get("model")
    if not isinstance(model, dict) or model.get("identity_status") != "REAL":
        raise ParseError(f"{label}: parent model identity is not REAL")
    for key in ("model_sha256", "binary_sha256"):
        if not isinstance(model.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", model[key]):
            raise ParseError(f"{label}: parent model {key} is invalid")
    if model["model_sha256"] != artifact_model_sha:
        raise ParseError(f"{label}: parent model SHA differs from artifact model")
    if isinstance(artifact_binary_sha, str) and model["binary_sha256"] != artifact_binary_sha:
        raise ParseError(f"{label}: parent binary SHA differs from artifact binary")
    parent_runtime_contract = parent.get("runtime_contract")
    if not isinstance(parent_runtime_contract, dict) or set(parent_runtime_contract) != Q2Q3_RUNTIME_CONTRACT_KEYS:
        raise ParseError(f"{label}: parent runtime contract is missing or malformed")
    if runtime_contract is not None and parent_runtime_contract != runtime_contract:
        raise ParseError(f"{label}: parent runtime contract differs from spec runtime contract")
    parent_lineages = {
        turn.get("lineage_id") for turn in turns
        if isinstance(turn, dict) and isinstance(turn.get("lineage_id"), int)
    }
    if source_lineage_id not in parent_lineages:
        raise ParseError(f"{label}: source lineage is absent from parent transcript")




def replay_admission_seq_authority(
        admission: Any, actual: list[dict[str, Any]], label: str) -> dict[str, int]:
    if not isinstance(admission, list):
        raise ParseError(f"{label}: replay admission journal is missing")
    bindings: dict[str, set[int]] = {"A": set(), "B": set()}
    binding_statuses = {"admitted", "already_live", "finished", "lineage_live"}
    for index, item in enumerate(admission):
        if not isinstance(item, dict):
            raise ParseError(f"{label}: admission record {index} is not an object")
        sid = item.get("logical_session_id")
        if sid not in bindings:
            continue
        status = item.get("status")
        seq_id = item.get("seq_id")
        slot_id = item.get("slot_id")
        if status == "queued":
            if seq_id is not None or slot_id is not None:
                raise ParseError(f"{label}: queued {sid} admission record carries a forged binding")
            continue
        if status not in binding_statuses:
            raise ParseError(f"{label}: {sid} admission record has an invalid binding status")
        if (
            isinstance(seq_id, bool)
            or not isinstance(seq_id, int)
            or seq_id < 0
            or isinstance(slot_id, bool)
            or not isinstance(slot_id, int)
            or slot_id != seq_id
        ):
            raise ParseError(f"{label}: {sid} admission binding is missing, forged, or inconsistent")
        bindings[sid].add(seq_id)
    for sid, seqs in bindings.items():
        if len(seqs) != 1:
            raise ParseError(f"{label}: {sid} admission mapping is missing or drifted")
    if len({next(iter(seqs)) for seqs in bindings.values()}) != 2:
        raise ParseError(f"{label}: A/B admission mappings share or duplicate a seq")
    authority = {sid: next(iter(seqs)) for sid, seqs in bindings.items()}
    for index, row in enumerate(actual):
        sid = row.get("logical_session_id")
        if sid not in authority:
            continue
        if row.get("seq_id") != authority[sid] or row.get("slot_id") != authority[sid]:
            raise ParseError(f"{label}: response event {index} disagrees with admission seq authority")
    return authority


def validate_replay_phase_window_bindings(
        phase_id: str, window: dict[str, Any], ab_seq_ids: set[int],
        selected_seq_id: int | None, label: str) -> set[int]:
    if window.get("session_ids") != ["A", "B"]:
        raise ParseError(f"{label}: {phase_id} window A/B binding is invalid")
    seq_ids = window.get("seq_ids")
    expected_seq_ids = window.get("expected_seq_ids")
    if (
        not isinstance(seq_ids, list)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in seq_ids)
        or seq_ids != sorted(seq_ids)
        or len(set(seq_ids)) != len(seq_ids)
        or expected_seq_ids != seq_ids
    ):
        raise ParseError(f"{label}: {phase_id} window seq binding is invalid or forged")
    window_seq_ids = set(seq_ids)
    if phase_id != "post-final-restore":
        if window_seq_ids != ab_seq_ids:
            raise ParseError(f"{label}: {phase_id} window seq_ids do not match admission authority")
    else:
        if len(window_seq_ids) != 1 or not window_seq_ids.issubset(ab_seq_ids):
            raise ParseError(f"{label}: post-final-restore window seq_id is outside A/B authority")
        if selected_seq_id is not None and window_seq_ids != {selected_seq_id}:
            raise ParseError(f"{label}: post-final-restore window seq_id differs from selected victim")
    return window_seq_ids

def validate_replay_phase_evidence(
        qualification: Any, replay_plan: Any, actual: list[dict[str, Any]],
        replay_artifact: dict[str, Any], run_dir: pathlib.Path, case: dict[str, Any],
        spec: dict[str, Any], label: str) -> dict[str, Any]:
    contract = replay_plan.fixture_contract
    if contract is None:
        if isinstance(qualification, dict) and (
                qualification.get("phase_windows") or qualification.get("erase_actions")):
            raise ParseError(f"{label}: non-derived replay contains qualification phase evidence")
        return {
            "status": "NOT_APPLICABLE", "phase_windows": [], "erase_actions": [],
            "actions": [], "resident_observations": [], "qualified_offload_pairs": [],
            "offload_physical_relief_bytes": 0, "resume_events": [], "resume_timings": [],
            "io": {}, "qualification_round_trip": None,
        }
    if not isinstance(qualification, dict):
        raise ParseError(f"{label}: derived replay qualification record is missing")
    phase_windows = qualification.get("phase_windows")
    erase_actions = qualification.get("erase_actions")
    if not isinstance(phase_windows, list) or not isinstance(erase_actions, list):
        raise ParseError(f"{label}: derived replay phase evidence is missing")
    phase_order = list(contract["phase_order"])
    if len(phase_windows) != len(phase_order):
        raise ParseError(f"{label}: derived replay does not contain all phase windows")
    stderr_data = (run_dir / "server.stderr").read_bytes()
    admission = replay_artifact.get("admission")
    admission_authority = replay_admission_seq_authority(admission, actual, label)
    ab_admission_seq_ids = set(admission_authority.values())
    previous_end = None
    normalized_windows: list[dict[str, Any]] = []
    all_actions: list[dict[str, str]] = []
    all_observations: list[dict[str, str]] = []
    all_resumes: list[dict[str, str]] = []
    all_timings: list[dict[str, str]] = []
    phase_records: dict[str, dict[str, Any]] = {}
    for index, window in enumerate(phase_windows):
        if not isinstance(window, dict):
            raise ParseError(f"{label}: phase window {index} is not an object")
        required = {
            "status", "phase_id", "started_mono_ns", "completed_mono_ns", "duration_ns",
            "stderr_start_offset", "stderr_end_offset", "expected_seq_ids", "evidence",
            "phase_index", "session_ids", "seq_ids",
        }
        if set(window) != required:
            raise ParseError(f"{label}: phase window {index} schema mismatch")
        if window["status"] != "passed" or window["phase_id"] != phase_order[index] or window["phase_index"] != index:
            raise ParseError(f"{label}: phase window order/status is invalid")
        phase_id = phase_order[index]
        window_seq_ids = validate_replay_phase_window_bindings(
            phase_id, window, ab_admission_seq_ids, None, f"{label}.phase[{index}]")
        for key in ("started_mono_ns", "completed_mono_ns", "duration_ns", "stderr_start_offset", "stderr_end_offset"):
            require_nonnegative_int(window[key], f"{label}.phase[{index}].{key}")
        if window["completed_mono_ns"] < window["started_mono_ns"] or window["duration_ns"] != window["completed_mono_ns"] - window["started_mono_ns"]:
            raise ParseError(f"{label}: phase window timing is inconsistent")
        if window["stderr_end_offset"] <= window["stderr_start_offset"] or window["stderr_end_offset"] > len(stderr_data):
            raise ParseError(f"{label}: phase window stderr offsets are invalid")
        if previous_end is not None and window["stderr_start_offset"] != previous_end:
            raise ParseError(f"{label}: phase stderr windows are not contiguous")
        previous_end = window["stderr_end_offset"]
        text = stderr_window(stderr_data, window["stderr_start_offset"], window["stderr_end_offset"], f"{label}.phase[{index}]")
        actions = marker_records(text, "kv_pressure_unified_action", ACTION_REQUIRED, f"{label}.phase[{index}].action")
        observations = marker_records(text, "kv_g0_s1_resident_observation", RESIDENT_OBSERVATION_REQUIRED, f"{label}.phase[{index}].resident")
        resumes = marker_records(text, "kv_resume_order_event", RESUME_REQUIRED, f"{label}.phase[{index}].resume")
        timings = marker_records(text, "kv_resume_stage_timing", TIMING_REQUIRED, f"{label}.phase[{index}].timing")
        for action in actions:
            validate_action_fields(action, f"{label}.phase[{index}].action")
            boolean_fields(action, ACTION_BOOL, f"{label}.phase[{index}].action")
        for observation in observations:
            numeric_fields(observation, RESIDENT_OBSERVATION_NUMERIC, f"{label}.phase[{index}].resident")
            boolean_fields(observation, RESIDENT_OBSERVATION_BOOL, f"{label}.phase[{index}].resident")
        for event in resumes:
            numeric_fields(event, {"decision_id", "seq_id", "claimant_epoch", "transaction_id"}, f"{label}.phase[{index}].resume")
            boolean_fields(event, RESUME_BOOL, f"{label}.phase[{index}].resume")
        for timing in timings:
            numeric_fields(timing, TIMING_REQUIRED, f"{label}.phase[{index}].timing")
        expected_source = pressure_basis_source(spec["pressure_basis"])
        evidence_seqs = phase_evidence_seq_set(
            phase_id, window, actions, observations, resumes, expected_source)
        expected_seq_ids = set(window["seq_ids"])
        if phase_requires_ab_pair_evidence(phase_id):
            # prepare-offload: both A and B must each complete a real OFFLOAD with
            # a transaction-local physical drop.
            if evidence_seqs != expected_seq_ids:
                raise ParseError(f"{label}: {phase_id} lacks A/B OFFLOAD physical-drop evidence")
        elif phase_requires_ab_restore_evidence(phase_id):
            # prepare-restore: both A and B must each have a positive Exact restore.
            if evidence_seqs != expected_seq_ids:
                raise ParseError(f"{label}: {phase_id} lacks positive A/B restore evidence")
        elif phase_requires_single_victim_evidence(phase_id):
            # final-competition: A/B must both be eligible candidates, but only a
            # single state-changing OFFLOAD victim is permitted. The unique-victim
            # fail-closed check runs in the final block against the A/B candidate
            # set; here the window only needs the A/B candidates to be present in
            # the phase's eligible score records (checked in the final block).
            if len(evidence_seqs) > 1:
                raise ParseError(f"{label}: final-competition window has multiple state-changing OFFLOAD victims")
        elif phase_requires_selected_restore_evidence(phase_id):
            # post-final-restore: only the selected victim may restore; the window
            # already binds exactly the selected victim seq, and non-selected
            # candidates must perform no positive restore.
            if evidence_seqs != expected_seq_ids:
                raise ParseError(f"{label}: post-final-restore evidence does not bind the selected victim")
        phase_records[phase_id] = {
            "actions": actions,
            "observations": observations,
            "resumes": resumes,
            "timings": timings,
        }
        all_actions.extend(actions); all_observations.extend(observations)
        all_resumes.extend(resumes); all_timings.extend(timings)
        normalized_windows.append(dict(window))

    if any(item.get("logical_session_id") in {"A", "B"} for item in erase_actions):
        raise ParseError(f"{label}: A/B lineage was erased before qualification ended")
    c_erases = [item for item in erase_actions if item.get("logical_session_id") == contract["control_session_id"]]
    if len(c_erases) != 1 or c_erases[0].get("phase_id") != contract["explicit_erase"]["phase_id"]:
        raise ParseError(f"{label}: derived replay C explicit erase evidence is missing or duplicated")
    for sid in ("A", "B"):
        live = [item for item in admission if item.get("logical_session_id") == sid and item.get("status") == "lineage_live"]
        if not live:
            raise ParseError(f"{label}: {sid} lineage was not kept live through qualification")
    c_records = [item for item in admission if item.get("logical_session_id") == "C" and item.get("seq_id") is not None]
    if not c_records:
        raise ParseError(f"{label}: control session C has no bound claimant")
    c_seq = c_records[-1]["seq_id"]
    expected_source = pressure_basis_source(spec["pressure_basis"])
    prepare_feedback_record: dict[str, Any] | None = None
    prepare_offload_feedback: dict[int, dict[str, int]] | None = None
    prepare_restore_feedback: dict[int, dict[str, int]] | None = None
    if case["policy"] == "v3":
        prepare_offload = phase_records.get("prepare-offload")
        prepare_restore = phase_records.get("prepare-restore")
        if prepare_offload is None or prepare_restore is None:
            raise ParseError(f"{label}: V3 prepare feedback phase records are missing")
        prepare_offload_feedback = replay_prepare_offload_feedback(
            prepare_offload["actions"], prepare_offload["observations"],
            expected_source, ab_admission_seq_ids, f"{label}.prepare-offload")
        prepare_restore_feedback = replay_prepare_restore_feedback(
            prepare_restore["resumes"], prepare_restore["timings"],
            prepare_offload_feedback, f"{label}.prepare-restore")
    final_text = stderr_window(stderr_data, phase_windows[2]["stderr_start_offset"], phase_windows[2]["stderr_end_offset"], f"{label}.final")
    final_actions = marker_records(final_text, "kv_pressure_unified_action", ACTION_REQUIRED, f"{label}.final.action")
    if not final_actions:
        raise ParseError(f"{label}: final competition has no action marker")
    validate_action_target_markers(final_actions, case, spec, f"{label}.final.action")
    ab_seq = set(ab_admission_seq_ids)
    expected_source = pressure_basis_source(spec["pressure_basis"])
    final_observations = marker_records(final_text, "kv_g0_s1_resident_observation", RESIDENT_OBSERVATION_REQUIRED, f"{label}.final.resident")
    # The selected final victim must NOT be derived from the last action marker
    # (a trailing no-op marker must not redefine the selection). It is the unique
    # qualifying state-changing OFFLOAD + transaction-local resident-drop pair in
    # the final window whose seq belongs to the A/B candidate set; zero or more
    # than one such pair is fail-closed.
    victim_pair = final_competition_selected_victim(
        final_actions, final_observations, expected_source, ab_seq)
    if victim_pair is None:
        raise ParseError(f"{label}: final-competition has no unique qualifying state-changing OFFLOAD victim within A/B candidates")
    selected_seq = victim_pair["seq_id"]
    # Bind final_action to the victim pair's own action marker, not the window
    # tail, so a trailing no-op cannot shadow the real selection.
    final_action = victim_pair["action"]
    score_records = _parse_v3_score_records(final_action.get("scores", "none"), f"{label}.final.score")
    eligible = [record for record in score_records if record["eligible"] == "1"]
    if len(eligible) < 2:
        raise ParseError(f"{label}: final competition has fewer than two eligible candidates")
    selected = next((record for record in eligible if int(record["seq_id"]) == selected_seq), None)
    if selected is None or int(selected["rank"]) != 0:
        raise ParseError(f"{label}: selected final claimant is not rank zero")
    for seq_id in ab_seq:
        record = next((item for item in eligible if int(item["seq_id"]) == seq_id), None)
        if record is None:
            raise ParseError(f"{label}: A/B claimant is not eligible in final score")
        if case["policy"] == "v3" and (record["cost_aware"] != "1" or record["fallback_reason"] != "none" or final_action.get("decision_fallback") != "none"):
            raise ParseError(f"{label}: V3 A/B final claimant is not fully cost-aware")
    history_record: dict[int, dict[str, int]] | None = None
    if prepare_offload_feedback is not None and prepare_restore_feedback is not None:
        history_record = validate_prepare_feedback_history(
            final_action,
            score_records,
            prepare_offload_feedback,
            prepare_restore_feedback,
            ab_seq,
            selected_seq,
            f"{label}.final.prepare-feedback")
        prepare_feedback_record = {
            "offload": {str(seq_id): dict(value) for seq_id, value in prepare_offload_feedback.items()},
            "restore": {str(seq_id): dict(value) for seq_id, value in prepare_restore_feedback.items()},
            "history": {str(seq_id): dict(value) for seq_id, value in history_record.items()},
        }
    c_score = next((item for item in score_records if int(item["seq_id"]) == int(c_seq)), None)
    if c_score is None or c_score["exclusion"] not in {"active", "protected"}:
        raise ParseError(f"{label}: control session C was not safety-excluded from ranking")
    matching = next((pair for pair in qualified_offload_pairs(final_actions, final_observations, expected_source) if pair["seq_id"] == selected_seq and pair["decision_id"] == int(final_action["decision_id"]) and pair["transaction_id"] == int(final_action["transaction_id"])), None)
    if matching is None or final_action.get("physical_relief_available") != "1" or int(final_action["physical_relief_bytes"]) != matching["resident_drop_bytes"]:
        raise ParseError(f"{label}: final physical relief does not match transaction-local resident drop")
    selected_score = selected
    if int(final_action["physical_object_id"]) != int(selected_score["physical_object_id"]) or int(final_action["physical_generation"]) != int(selected_score["physical_generation"]):
        raise ParseError(f"{label}: final OFFLOAD physical identity differs from selected score")
    io_records = marker_records(stderr_data.decode("utf-8", errors="replace"), "KV_PAGED_IO_STATS", IO_REQUIRED, f"{label}.io")
    if not io_records:
        raise ParseError(f"{label}: replay IO evidence is missing")
    final_events: dict[str, Any] = {}
    post_events: dict[str, Any] = {}
    for sid in ("A", "B"):
        final_candidates = [event for event in replay_plan.events if event.logical_session_id == sid and event.phase_id == "final-competition"]
        post_candidates = [event for event in replay_plan.events if event.logical_session_id == sid and event.phase_id == "post-final-restore"]
        if not final_candidates or not post_candidates:
            raise ParseError(f"{label}: final/post-final phase is missing a required {sid} event")
        final_events[sid] = max(final_candidates, key=lambda event: event.turn)
        post_events[sid] = max(post_candidates, key=lambda event: event.turn)
    for sid in ("A", "B"):
        if (final_events[sid].prompt_tokens, final_events[sid].n_predict, final_events[sid].reference_completion_tokens) != (post_events[sid].prompt_tokens, post_events[sid].n_predict, post_events[sid].reference_completion_tokens):
            raise ParseError(f"{label}: post-final request differs from final fixed request for {sid}")
    validate_replay_phase_window_bindings(
        "post-final-restore", phase_windows[3], ab_admission_seq_ids,
        selected_seq, f"{label}.phase[3]")
    post_window = phase_windows[3]
    post_text = stderr_window(
        stderr_data, post_window["stderr_start_offset"],
        post_window["stderr_end_offset"], f"{label}.post-final")
    post_events_marker = marker_records(
        post_text, "kv_resume_order_event", RESUME_REQUIRED, f"{label}.post-final.resume")
    post_positive = [event for event in post_events_marker
                     if event.get("phase") == "prefetch"
                     and event.get("outcome") == "completed"
                     and event.get("graph_allowed") == "1"]
    if not any(int(event["seq_id"]) == selected_seq for event in post_positive):
        raise ParseError(f"{label}: selected final claimant has no positive post-final restore")
    if any(int(event["seq_id"]) != selected_seq for event in post_positive):
        raise ParseError(f"{label}: non-selected claimant performed a post-final restore")
    return {
        "status": "PASS", "phase_windows": normalized_windows, "erase_actions": list(erase_actions),
        "actions": all_actions, "resident_observations": all_observations,
        "qualified_offload_pairs": qualified_offload_pairs(all_actions, all_observations, pressure_basis_source(spec["pressure_basis"])),
        "offload_physical_relief_bytes": sum(item["resident_drop_bytes"] for item in qualified_offload_pairs(all_actions, all_observations, pressure_basis_source(spec["pressure_basis"]))),
        "resume_events": all_resumes, "resume_timings": all_timings,
        "io": io_records[-1],
        "qualification_round_trip": {
            "status": "PASS",
            "selected_seq_id": selected_seq,
            **({"prepare_feedback": prepare_feedback_record}
               if prepare_feedback_record is not None else {}),
        },
    }

def _normalize_q3_argv(argv: list[str]) -> list[str]:
    normalized = list(argv)
    for index, item in enumerate(normalized[:-1]):
        if item == "--port":
            normalized[index + 1] = "<dynamic-port>"
    return normalized


def _normalize_q3_environment(environment: dict[str, str]) -> dict[str, str]:
    return {
        key: value for key, value in environment.items()
        if key not in {"LLAMA_KV_PRESSURE_POLICY", "LLAMA_KV_SWAP_DIR"}
    }


def validate_q3_policy_only_identity(
        artifact: pathlib.Path, manifest: dict[str, Any], plan: list[dict[str, Any]],
        spec: dict[str, Any]) -> None:
    if spec.get("phase") != "representative" or spec.get("runtime_contract") is None:
        return
    if [item["policy"] for item in plan].count("v2") != 1 or \
            [item["policy"] for item in plan].count("idle_age") != 1 or \
            [item["policy"] for item in plan].count("v3") != 1:
        raise ParseError("representative Q3 requires exactly one execution per pressure policy")
    baseline_argv: list[str] | None = None
    baseline_env: dict[str, str] | None = None
    baseline_cgroup: dict[str, Any] | None = None
    baseline_case: dict[str, Any] | None = None
    for item in plan:
        run_dir = artifact / "runs" / item["run_id"]
        execution = exact(
            read_json(run_dir / "execution.json"), EXECUTION_KEYS,
            f"{item['run_id']}.execution", {"replay", "lifecycle", "resident_preflight"})
        argv = execution.get("argv")
        environment = execution.get("environment")
        cgroup = execution.get("server_cgroup")
        if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
            raise ParseError(f"{item['run_id']}: Q3 execution argv is invalid")
        if not isinstance(environment, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in environment.items()):
            raise ParseError(f"{item['run_id']}: Q3 execution environment is invalid")
        if not isinstance(cgroup, dict):
            raise ParseError(f"{item['run_id']}: Q3 execution cgroup is missing")
        normalized_argv = _normalize_q3_argv(argv)
        normalized_env = _normalize_q3_environment(environment)
        normalized_cgroup = {
            key: cgroup.get(key) for key in ("scope", "memory_max")
        }
        if baseline_argv is None:
            baseline_argv = normalized_argv
            baseline_env = normalized_env
            baseline_cgroup = normalized_cgroup
            baseline_case = item
            continue
        if normalized_argv != baseline_argv:
            raise ParseError(f"{item['run_id']}: Q3 argv drifts outside dynamic port")
        if normalized_env != baseline_env:
            raise ParseError(f"{item['run_id']}: Q3 environment drifts outside pressure policy/swap directory")
        if normalized_cgroup != baseline_cgroup:
            raise ParseError(f"{item['run_id']}: Q3 cgroup contract drifts")
        for key in ("kv_representation", "loading_mode", "restore", "prefault", "kv_target_bytes", "action_target_bytes"):
            if item[key] != baseline_case[key]:
                raise ParseError(f"{item['run_id']}: Q3 case field drifts at {key}")

def validate_target_freeze_provenance(
        manifest: dict[str, Any], spec: dict[str, Any], workload: dict[str, Any]) -> None:
    reference = spec.get("target_freeze_record")
    if reference is None:
        return
    record = load_target_freeze_record(
        reference, spec["runtime_contract"], workload, "manifest.target_freeze_record")
    identity = record["identity"]
    provenance = manifest["provenance"]
    if identity.get("head") != provenance["git"]["head"]:
        raise ParseError("target-freeze HEAD identity does not match Q3 manifest")
    if identity.get("binary_sha256") != provenance["binary"].get("sha256"):
        raise ParseError("target-freeze binary identity does not match Q3 manifest")
    if identity.get("model_sha256") != provenance["model"].get("sha256"):
        raise ParseError("target-freeze model identity does not match Q3 manifest")
    if identity.get("executor") != pathlib.Path(spec["binary"]).name:
        raise ParseError("target-freeze executor identity does not match Q3 spec")
    if identity.get("cgroup") != spec["cgroup"]:
        raise ParseError("target-freeze cgroup contract does not match Q3 spec")
    replay_path = pathlib.Path(workload["replay"]["path"])
    replay_source = read_json(replay_path)
    derived = replay_source.get("derived_from") if isinstance(replay_source, dict) else None
    if not isinstance(derived, dict):
        raise ParseError("Q3 replay must be a derived four-phase fixture")
    if identity["source_lineage_id"] != derived.get("source_lineage_id"):
        raise ParseError("Q2/Q3 source lineage identity mismatch")
    if identity["alignment_family_id"] != derived.get("alignment_family_id"):
        raise ParseError("Q2/Q3 alignment family identity mismatch")
    parent_identity = identity["parent_transcript_identity"]
    if parent_identity["sha256"] != derived.get("transcript_sha256"):
        raise ParseError("Q2/Q3 parent transcript identity mismatch")
    if identity["preflight_fixture_identity"] != derived.get("q2_preflight_fixture_identity"):
        raise ParseError("Q2/Q3 preflight fixture identity mismatch")
    if identity["q2_evidence_identity"] != derived.get("q2_evidence_identity"):
        raise ParseError("Q2/Q3 evidence identity mismatch")

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
    if manifest["spec"].get("profile") == FINAL_F16_PROFILE:
        if manifest.get("profile") != FINAL_F16_PROFILE:
            raise ParseError("final_f16 manifest profile identity is missing or drifted")
        if manifest.get("runtime_contract") != manifest["spec"].get("runtime_contract"):
            raise ParseError("final_f16 manifest runtime_contract identity is missing or drifted")
        if manifest.get("global_governor_isolation") != FINAL_F16_ISOLATION:
            raise ParseError("final_f16 manifest governor isolation identity is missing or drifted")
    validate_target_freeze_provenance(manifest, manifest["spec"], workload)
    if "replay" in workload:
        replay_path = pathlib.Path(workload["replay"]["path"])
        replay_source = read_json(replay_path)
        if workload["replay"]["source"] == "transcript":
            model_binding = replay_source.get("model") if isinstance(replay_source, dict) else None
            validate_replay_model_binding(
                model_binding, provenance["model"]["sha256"], provenance["binary"].get("sha256"),
                manifest["spec"].get("runtime_contract"))
        elif isinstance(replay_source, dict) and replay_source.get("schema") == "gt-trace-1b-q1-derived/v1":
            validate_derived_replay_parent(
                replay_source, provenance["model"]["sha256"],
                provenance["binary"]["sha256"], "manifest.derived_replay",
                manifest["spec"].get("runtime_contract"))
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
        result_keys = {"run_id", "case_id", "round", "run_order", "execution_index", "directory", "status", "error"}
        result_optional = {"role"} if manifest["spec"].get("profile") == FINAL_F16_PROFILE else set()
        result = exact(result, result_keys, f"manifest.run_results[{index}]", result_optional)
        if index < len(plan):
            if manifest["spec"].get("profile") == FINAL_F16_PROFILE:
                if result.get("role") != plan[index].get("role"):
                    raise ParseError(f"manifest.run_results[{index}] role differs from planned case")
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


def write_target_freeze_record(artifact: pathlib.Path, result: dict[str, Any]) -> None:
    if result.get("verdict") != "RESIDENT_PREFLIGHT_PASS":
        return
    target = result.get("target_freeze")
    if (not isinstance(target, dict) or target.get("status") != "FROZEN"
            or target.get("frozen") is not True or not isinstance(target.get("frozen_T"), int)):
        raise ParseError("resident_preflight pass is missing a frozen target record")
    identity = target.get("identity")
    if not isinstance(identity, dict):
        raise ParseError("resident_preflight target freeze has ambiguous identity records")
    if not isinstance(identity.get("runtime_contract"), dict):
        raise ParseError("resident_preflight target freeze is missing runtime contract identity")
    record = dict(target)
    record.update({
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "kind": "resident_preflight_target_freeze",
        "status": "FROZEN",
        "frozen": True,
        "runtime_contract": identity["runtime_contract"],
    })
    dump_path = artifact / "target-freeze.json"
    dump_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

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


def curve_numeric_summary(values: list[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "status": "UNAVAILABLE", "n": 0, "min": None, "max": None, "p50": None,
        }
    numeric = [float(value) for value in values]
    return {
        "status": "AVAILABLE", "n": len(numeric), "min": min(numeric),
        "max": max(numeric), "p50": percentile(numeric, 0.50),
    }


def unavailable_curve_metric(unit: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE", "unit": unit, "n": 0,
        "p50": None, "p95": None, "p99": None,
    }


def curve_point(
        item: dict[str, Any], segment: str, baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    policy = item["policy"]
    if segment == "release_segment":
        actual = item.get("resident_after_release_settle")
        actual_authority = item.get("resident_after_release_settle_authority")
        steady = item.get("performance") or {}
        eligible = (
            policy == "release_only"
            and item.get("status") == "RELEASE_SETTLED"
            and item.get("performance_eligible") is True
            and isinstance(actual, int)
            and actual_authority == "slots_physical"
        )
        exclusion_reason = None if eligible else (
            "release_no_candidate_floor_probe"
            if item.get("release_terminal") == "release_no_candidate"
            else "RELEASE target is not physically settled")
    else:
        actual = item.get("resident_settled")
        actual_authority = item.get("resident_settled_authority")
        steady = item.get("post_resume_steady_performance") or {}
        eligible = (
            policy in SWAP_POLICIES
            and item.get("status") == "TARGET_REACHED"
            and isinstance(actual, int)
            and actual_authority == "slots_physical"
            and item.get("total_physical_relief_bytes") is not None
        )
        exclusion_reason = None if eligible else (
            "UNMET_FLOOR_without_independent_physical_authority"
            if item.get("status") == "UNMET_FLOOR"
            and actual_authority != "slots_physical"
            else "V2 point requires TARGET_REACHED and independent physical authority")
    requested = item.get("requested_target_bytes")
    io = item.get("io") or {}
    resume = item.get("resume") or {}
    k2 = item.get("k2") or {}
    if not isinstance(requested, int) or requested <= 0:
        raise ParseError(f"{item.get('run_id', 'budget run')}: requested target is invalid")
    baseline_bytes = baseline.get("resident_after_fill") if baseline else None
    if eligible and isinstance(baseline_bytes, int):
        memory_saved_bytes = baseline_bytes - actual
        memory_saved_ratio = memory_saved_bytes / baseline_bytes if baseline_bytes else None
    else:
        memory_saved_bytes = item.get("memory_saved_bytes") if eligible else None
        memory_saved_ratio = item.get("memory_saved_ratio") if eligible else None
    metrics = {
        name: steady.get(name, unavailable_curve_metric(unit))
        for name, unit in (
            ("e2e_ms", "ms"),
            ("tpot_ms_per_token", "ms/token"),
            ("throughput_tokens_per_second", "tokens/s"),
            ("ttft_ms", "ms"),
        )
    }
    point = {
        "run_id": item["run_id"],
        "case_id": item["case_id"],
        "round": item["round"],
        "policy": policy,
        "requested_target_bytes": requested,
        "action_target_bytes": item.get("action_target_bytes"),
        "actual_settled_resident_bytes": actual if eligible else None,
        "actual_resident_bytes": actual if eligible else None,
        "actual_resident_authority": actual_authority if eligible else "UNAVAILABLE",
        "target_error_bytes": actual - requested if eligible else None,
        "baseline_resident_bytes": baseline_bytes,
        "rss_cgroup_auxiliary": {
            "authority": "auxiliary_only",
            "memory_samples": item.get("memory"),
        },
        "release_physical_relief_bytes": item.get("release_physical_relief_bytes"),
        "release_physical_relief_authority": item.get("release_physical_relief_authority"),
        "offload_physical_relief_bytes": item.get("offload_physical_relief_bytes"),
        "offload_physical_relief_authority": item.get("offload_physical_relief_authority"),
        "total_physical_relief_bytes": item.get("total_physical_relief_bytes"),
        "total_physical_relief_authority": item.get("total_physical_relief_authority"),
        "physical_relief": {
            "release_bytes": item.get("release_physical_relief_bytes"),
            "release_authority": item.get("release_physical_relief_authority"),
            "offload_bytes": item.get("offload_physical_relief_bytes"),
            "offload_authority": item.get("offload_physical_relief_authority"),
            "total_bytes": item.get("total_physical_relief_bytes"),
            "total_authority": item.get("total_physical_relief_authority"),
        },
        "memory_saved_bytes": memory_saved_bytes,
        "memory_saved_ratio": memory_saved_ratio,
        "release": item.get("release"),
        "offload": item.get("offload"),
        "actions": {
            "release": (item.get("release") or {}).get("actions"),
            "offload": (item.get("offload") or {}).get("actions"),
        },
        "blocks": {
            "release": (item.get("release") or {}).get("blocks"),
            "offload": (item.get("offload") or {}).get("blocks"),
        },
        "backing_io": {
            key: int(io[key]) if key in io else None
            for key in (
                "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls",
                "backing_write_syscalls", "bytes_read", "bytes_written",
            )
        },
        "resume": resume,
        "resume_restored_bytes": resume.get("restored_bytes"),
        "resume_gate_us": resume.get("gate_us"),
        "resume_total_us": resume.get("total_us"),
        "k2": k2,
        "k2_read_us": k2.get("read_us"),
        "k2_unpack_us": k2.get("unpack_us"),
        "k2_prefault": k2.get("prefault"),
        "k2_scatter": k2.get("scatter"),
        "staging_peak_bytes": item.get("transient_staging_peak_bytes"),
        "staging_bound_bytes": item.get("transient_staging_bound_bytes"),
        "performance": metrics,
        "steady_performance": steady,
        "curve_eligible": eligible,
        "curve_exclusion_reason": exclusion_reason,
    }
    return point


def curve_target_aggregate(points: list[dict[str, Any]], target: int) -> dict[str, Any]:
    selected = [point for point in points if point["requested_target_bytes"] == target]
    def values(key: str) -> list[float | int]:
        return [point[key] for point in selected if point.get(key) is not None]
    performance = {
        name: curve_numeric_summary([
            metric["p50"] for point in selected
            for metric in [point["performance"].get(name, {})]
            if metric.get("status") == "AVAILABLE" and metric.get("p50") is not None
        ])
        for name in ("e2e_ms", "tpot_ms_per_token", "throughput_tokens_per_second", "ttft_ms")
    }
    return {
        "requested_target_bytes": target,
        "rounds": [
            {"round": point["round"], "run_id": point["run_id"],
             "actual_settled_resident_bytes": point["actual_settled_resident_bytes"]}
            for point in sorted(selected, key=lambda value: (value["round"], value["run_id"]))
        ],
        "actual_settled_resident_bytes": curve_numeric_summary(values("actual_settled_resident_bytes")),
        "target_error_bytes": curve_numeric_summary(values("target_error_bytes")),
        "memory_saved_bytes": curve_numeric_summary(values("memory_saved_bytes")),
        "memory_saved_ratio": curve_numeric_summary(values("memory_saved_ratio")),
        "release_physical_relief_bytes": curve_numeric_summary(values("release_physical_relief_bytes")),
        "offload_physical_relief_bytes": curve_numeric_summary(values("offload_physical_relief_bytes")),
        "total_physical_relief_bytes": curve_numeric_summary(values("total_physical_relief_bytes")),
        "performance": performance,
    }


def build_budget_curve(
        runs: list[dict[str, Any]], budget_sweep: dict[str, Any] | None,
        run_kind: str | None, final_f16: bool = False,
) -> dict[str, Any] | None:
    if budget_sweep is None:
        return None
    baseline_by_round: dict[int, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for item in runs:
        if item.get("status") == "RESIDENT_BASELINE":
            if item["round"] in baseline_by_round:
                diagnostics.append({
                    "run_id": item["run_id"], "round": item["round"],
                    "reason": "duplicate Resident baseline",
                })
            else:
                baseline_by_round[item["round"]] = item

    def segment(name: str, policy: str, targets: list[int]) -> dict[str, Any]:
        candidates = [
            item for item in runs
            if item.get("policy") == policy
            and (not final_f16 or (
                item.get("role") == "release_target" if policy == "release_only"
                else item.get("role") == "offload_target"))
        ]
        points: list[dict[str, Any]] = []
        segment_diagnostics: list[dict[str, Any]] = []
        for item in candidates:
            baseline = baseline_by_round.get(item["round"])
            point = curve_point(item, name, baseline)
            if point["curve_eligible"]:
                if baseline is None or not isinstance(point["baseline_resident_bytes"], int):
                    point["curve_eligible"] = False
                    point["curve_exclusion_reason"] = "missing Resident baseline in the same round"
                elif point["baseline_resident_bytes"] < point["actual_settled_resident_bytes"]:
                    point["curve_eligible"] = False
                    point["curve_exclusion_reason"] = "actual resident exceeds same-round Resident baseline"
            if point["curve_eligible"]:
                points.append(point)
            else:
                segment_diagnostics.append({
                    "run_id": point["run_id"], "case_id": point["case_id"],
                    "round": point["round"],
                    "requested_target_bytes": point["requested_target_bytes"],
                    "actual_settled_resident_bytes": None,
                    "actual_resident_authority": "UNAVAILABLE",
                    "status": item.get("status"),
                    "reason": point["curve_exclusion_reason"],
                })
        points.sort(key=lambda point: (
            point["actual_settled_resident_bytes"], point["round"], point["requested_target_bytes"],
        ))
        expected_count = len(targets) * int(budget_sweep["rounds"])
        missing = [
            {"round": round_id, "requested_target_bytes": target}
            for round_id in range(1, int(budget_sweep["rounds"]) + 1)
            for target in targets
            if not any(
                point["round"] == round_id and point["requested_target_bytes"] == target
                for point in points)
        ]
        target_aggregates = [curve_target_aggregate(points, target) for target in targets]
        per_round = [
            {
                "round": round_id,
                "points": [
                    point for point in points if point["round"] == round_id
                ],
            }
            for round_id in range(1, int(budget_sweep["rounds"]) + 1)
        ]
        performance = {
            name: curve_numeric_summary([
                metric["p50"] for point in points
                for metric in [point["performance"].get(name, {})]
                if metric.get("status") == "AVAILABLE" and metric.get("p50") is not None
            ])
            for name in ("e2e_ms", "tpot_ms_per_token", "throughput_tokens_per_second", "ttft_ms")
        }
        gate_pass = len(points) == expected_count and not missing
        return {
            "policy": policy,
            "requested_targets_bytes": list(targets),
            "action_target_bytes": budget_sweep["action_target_bytes"],
            "max_blocks": budget_sweep["max_blocks"],
            "axis": "actual_settled_resident_bytes",
            "axis_authority": "independent_physical_slots",
            "ordered_by": "actual_settled_resident_bytes_ascending",
            "points": points,
            "per_round": per_round,
            "target_aggregates": target_aggregates,
            "aggregate": {
                "n": len(points),
                "actual_settled_resident_bytes": curve_numeric_summary([
                    point["actual_settled_resident_bytes"] for point in points]),
                "target_error_bytes": curve_numeric_summary([
                    point["target_error_bytes"] for point in points]),
                "memory_saved_bytes": curve_numeric_summary([
                    point["memory_saved_bytes"] for point in points]),
                "memory_saved_ratio": curve_numeric_summary([
                    point["memory_saved_ratio"] for point in points]),
                "release_physical_relief_bytes": curve_numeric_summary([
                    point["release_physical_relief_bytes"] for point in points
                    if point.get("release_physical_relief_bytes") is not None]),
                "offload_physical_relief_bytes": curve_numeric_summary([
                    point["offload_physical_relief_bytes"] for point in points
                    if point.get("offload_physical_relief_bytes") is not None]),
                "total_physical_relief_bytes": curve_numeric_summary([
                    point["total_physical_relief_bytes"] for point in points
                    if point.get("total_physical_relief_bytes") is not None]),
                "performance": performance,
            },
            "gate": {
                "status": "PASS" if gate_pass else "UNAVAILABLE",
                "required_points": expected_count,
                "eligible_points": len(points),
                "missing_points": missing,
                "physical_authority_required": True,
                "reason": None if gate_pass else "not every target/round has TARGET_REACHED with independent physical resident authority",
            },
            "diagnostics": segment_diagnostics,
        }

    release_segment = segment(
        "release_segment", "release_only", list(budget_sweep["release_targets_bytes"]))
    offload_segment = segment(
        "offload_segment", "v2", list(budget_sweep["offload_targets_bytes"]))
    curve = {
        "status": "READY" if release_segment["gate"]["status"] == "PASS"
        and offload_segment["gate"]["status"] == "PASS" else "INCOMPLETE",
        "release_segment": release_segment,
        "offload_segment": offload_segment,
        "knee": None,
        "diagnostics": diagnostics,
    }
    if not final_f16:
        curve["reference_anchors"] = {
            "b_full_bytes": REFERENCE_B_FULL_BYTES,
            "b_release_floor_bytes": REFERENCE_B_RELEASE_FLOOR_BYTES,
        }
    if run_kind == "formal" and curve["status"] != "READY":
        raise ParseError("formal budget_sweep physical curve gate is incomplete")
    return curve


def summarize_characterization(
        run_results: list[dict[str, Any]], run_kind: str | None = None,
        budget_sweep: dict[str, Any] | None = None,
        completion_exact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runs = [
        {
            "run_id": item["run_id"],
            "case_id": item["case_id"],
            "round": item.get("round"),
            "policy": item.get("policy"),
            "role": item.get("role"),
            "io": item.get("io"),
            "statistics": item.get("statistics"),
            **item["characterization"],
        }
        for item in run_results if item["characterization"] is not None
    ]
    resident_runs = [item for item in runs if item["status"] == "RESIDENT_BASELINE"]
    release_runs = [item for item in runs if item.get("policy") == "release_only"]
    final_f16 = run_kind == "formal" and any(item.get("role") is not None for item in runs)
    release_floor_runs = [
        item for item in release_runs
        if (item.get("role") == "release_floor_probe" if final_f16
            else item.get("release_terminal") == "release_no_candidate")
    ]
    release_target_runs = [
        item for item in release_runs
        if (item.get("role") == "release_target" if final_f16
            else item.get("release_terminal") == "release_settled")
    ]
    v2_runs = [item for item in runs if item["status"] in {"TARGET_REACHED", "UNMET_FLOOR"}]
    rounds = {item.get("round") for item in runs}
    full_values = [item["resident_after_fill"] for item in resident_runs]
    if final_f16:
        for round_id in rounds:
            baselines = [item for item in resident_runs if item.get("round") == round_id]
            if len(baselines) != 1:
                raise ParseError(
                    f"final_f16 round {round_id} requires exactly one same-round Resident baseline")
            if baselines[0].get("resident_after_fill_authority") != "slots_physical":
                raise ParseError(
                    f"final_f16 round {round_id} Resident baseline lacks physical authority")
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
    formal_floor_complete = True
    if run_kind == "formal":
        if final_f16:
            formal_floor_complete = bool(release_runs) and all(
                sum(item.get("role") == "release_floor_probe" for item in release_runs
                    if item.get("round") == round_id) == 1
                for round_id in rounds
            ) and all(
                item.get("release_terminal") == "release_no_candidate"
                for item in release_floor_runs
            )
        else:
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
    budget_curve = build_budget_curve(runs, budget_sweep, run_kind, final_f16=final_f16)
    if budget_curve is not None:
        observed_anchor_status = (
            "AVAILABLE"
            if b_full["status"] == "AVAILABLE" and b_release_floor["status"] == "AVAILABLE"
            else "PARTIAL"
            if b_full["status"] == "AVAILABLE" or b_release_floor["status"] == "AVAILABLE"
            else "UNAVAILABLE"
        )
        budget_curve["observed_anchors"] = {
            "b_full_bytes": b_full["p50_bytes"],
            "b_release_floor_bytes": b_release_floor["p50_bytes"],
            "status": observed_anchor_status,
        }
    completion_summary = completion_exact or {
        "status": "UNAVAILABLE",
        "completion_exact": None,
        "comparisons": [],
    }
    final_kv_lifecycle_curve = {
        "status": "READY" if final_f16 and budget_curve is not None
        and budget_curve.get("status") == "READY" else "UNAVAILABLE",
        "profile": FINAL_F16_PROFILE if final_f16 else None,
        "cache_type_k": "f16" if final_f16 else None,
        "cache_type_v": "f16" if final_f16 else None,
        "b_full": b_full,
        "b_release_floor": b_release_floor,
        "release_targets": release_target_points,
        "budget_curve": budget_curve,
        "completion_exact": completion_summary,
        "physical_authority": {
            "b_full": "same_round_resident_after_fill_slots_physical" if final_f16 else "not_final_f16",
            "b_release_floor": "same_round_release_floor_probe_slots_physical" if final_f16 else "not_final_f16",
            "curve_axis": "actual_settled_resident_bytes",
            "curve_axis_authority": "independent_physical_slots",
        },
        "rss_cgroup_auxiliary": {
            "authority": "auxiliary_only_not_physical_kv_relief",
            "source": "memory_samples.tsv",
            "runs": [
                {
                    "run_id": item["run_id"],
                    "memory": item.get("memory"),
                }
                for item in runs
            ] if final_f16 else [],
        },
    }
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
        "budget_curve": budget_curve,
        "completion_exact": completion_summary,
        "final_kv_lifecycle_curve": final_kv_lifecycle_curve,
    }


def aggregate_target_freeze_records(
        records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ParseError("resident_preflight produced no target-freeze record")
    first = records[0]
    frozen_t = first.get("frozen_T")
    identity = first.get("identity")
    if (first.get("status") != "FROZEN" or first.get("frozen") is not True
            or isinstance(frozen_t, bool) or not isinstance(frozen_t, int)
            or not isinstance(identity, dict)):
        raise ParseError("resident_preflight target-freeze record is invalid")
    for index, record in enumerate(records[1:], start=1):
        if (record.get("status") != "FROZEN" or record.get("frozen") is not True
                or record.get("frozen_T") != frozen_t):
            raise ParseError(
                f"resident_preflight target-freeze round {index} disagrees on frozen_T")
        round_identity = record.get("identity")
        if not isinstance(round_identity, dict) or round_identity != identity:
            raise ParseError(
                f"resident_preflight target-freeze round {index} disagrees on identity")
    aggregate = dict(first)
    aggregate["round_count"] = len(records)
    aggregate["rounds"] = [dict(record) for record in records]
    aggregate["identity"] = dict(identity)
    return aggregate


def validate_final_f16_completion_exact(
        run_results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_round: dict[int, list[dict[str, Any]]] = {}
    for run in run_results:
        round_id = run.get("round")
        role = run.get("role")
        if not isinstance(round_id, int) or role not in {
                "resident_baseline", "release_target", "release_floor_probe", "offload_target"}:
            raise ParseError("final_f16 completion identity has invalid round or role")
        by_round.setdefault(round_id, []).append(run)

    comparisons: list[dict[str, Any]] = []
    for round_id, round_runs in sorted(by_round.items()):
        residents = [run for run in round_runs if run.get("role") == "resident_baseline"]
        if len(residents) != 1:
            raise ParseError(
                f"final_f16 round {round_id} requires exactly one Resident completion oracle")
        oracle = residents[0]
        oracle_records = oracle.get("responses")
        if not isinstance(oracle_records, list) or not oracle_records:
            raise ParseError(f"final_f16 round {round_id} Resident responses are missing")

        def index_records(run: dict[str, Any], label: str) -> dict[tuple[str, int], dict[str, Any]]:
            records = run.get("responses")
            if not isinstance(records, list) or not records:
                raise ParseError(f"{label}: completion responses are missing")
            indexed: dict[tuple[str, int], dict[str, Any]] = {}
            for record in records:
                request_id = record.get("request_id")
                repeat_index = record.get("repeat_index")
                key = (request_id, repeat_index)
                if not isinstance(request_id, str) or not isinstance(repeat_index, int) or key in indexed:
                    raise ParseError(f"{label}: duplicate or invalid completion request identity")
                completion = record.get("completion_sha256")
                if not isinstance(completion, str) or not re.fullmatch(r"[0-9a-f]{64}", completion):
                    raise ParseError(f"{label}: completion identity is missing or malformed")
                if record.get("http_status", 0) < 200 or record.get("http_status", 0) >= 300:
                    raise ParseError(f"{label}: completion response is not HTTP 2xx")
                if record.get("error") is not None:
                    raise ParseError(f"{label}: completion response contains an error")
                for field in (
                    "n_predict", "stream", "measurement", "prompt_sha256", "temperature", "seed",
                ):
                    if field not in record:
                        raise ParseError(
                            f"{label}: completion workload identity is missing field {field}")
                indexed[key] = record
            return indexed

        oracle_index = index_records(oracle, f"{oracle['run_id']}")
        for candidate in round_runs:
            if candidate is oracle:
                continue
            candidate_index = index_records(candidate, f"{candidate['run_id']}")
            if set(candidate_index) != set(oracle_index):
                raise ParseError(
                    f"final_f16 round {round_id} request identity differs between "
                    f"{oracle['run_id']} and {candidate['run_id']}")
            matched = 0
            for key, oracle_record in oracle_index.items():
                candidate_record = candidate_index[key]
                for field in (
                    "n_predict", "stream", "measurement", "prompt_sha256", "temperature", "seed",
                ):
                    if candidate_record.get(field) != oracle_record.get(field):
                        raise ParseError(
                            f"final_f16 completion workload identity drift at "
                            f"{candidate['run_id']} request {key[0]} repeat {key[1]} field {field}")
                if candidate_record["completion_sha256"] != oracle_record["completion_sha256"]:
                    raise ParseError(
                        f"final_f16 completion drift at {candidate['run_id']} request "
                        f"{key[0]} repeat {key[1]}")
                matched += 1
            comparisons.append({
                "round": round_id,
                "resident_run_id": oracle["run_id"],
                "candidate_run_id": candidate["run_id"],
                "candidate_role": candidate["role"],
                "request_count": matched,
                "status": "PASS",
            })
    if not comparisons:
        raise ParseError("final_f16 completion identity has no candidate comparisons")
    return {
        "status": "PASS",
        "completion_exact": True,
        "comparisons": comparisons,
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
        if "replay" in workload:
            replay_results = [
                parse_run(artifact, item, cases[item["case_id"]], workload, index, manifest["spec"])
                for index, item in enumerate(plan)
            ]
            validate_q3_policy_only_identity(artifact, manifest, plan, manifest["spec"])
            fidelity = {
                "status": "PASS" if all(item["workload_fidelity"]["status"] == "PASS" for item in replay_results) else "FAIL",
                "runs": [item["workload_fidelity"] for item in replay_results],
                "planned_sessions": sum(item["workload_fidelity"]["planned_sessions"] for item in replay_results),
                "executed_sessions": sum(item["workload_fidelity"]["executed_sessions"] for item in replay_results),
                "planned_turns": sum(item["workload_fidelity"]["planned_turns"] for item in replay_results),
                "completed_turns": sum(item["workload_fidelity"]["completed_turns"] for item in replay_results),
            }
            if manifest["spec"]["run_mode"] == "resident_preflight":
                preflight_runs = [item["resident_preflight"] for item in replay_results]
                target_records: list[dict[str, Any]] = []
                target_error: str | None = None
                if fidelity["status"] != "PASS":
                    verdict = "INVALID_ARTIFACT"
                else:
                    for index, preflight in enumerate(preflight_runs):
                        label = f"resident_preflight[{index}]"
                        try:
                            target = freeze_resident_preflight_target(
                                preflight, manifest["spec"]["runtime_contract"], label)
                        except ParseError as exc:
                            if str(exc).startswith(f"{label}: TARGET_WINDOW_UNAVAILABLE"):
                                target_error = str(exc)
                                break
                            raise
                        identity = resident_preflight_identity(
                            artifact, manifest, manifest["spec"], preflight,
                            artifact / "runs" / replay_results[index]["run_id"])
                        target["identity"] = identity
                        target_records.append(target)
                    if target_error is not None:
                        verdict = "TARGET_WINDOW_UNAVAILABLE"
                    elif not target_records:
                        raise ParseError("resident_preflight produced no target-freeze record")
                    else:
                        frozen_values = {record["frozen_T"] for record in target_records}
                        if len(frozen_values) != 1:
                            raise ParseError(
                                "resident_preflight target-freeze records disagree on frozen_T")
                        verdict = "RESIDENT_PREFLIGHT_PASS"
                target_record: dict[str, Any]
                if target_error is not None:
                    target_record = {
                        "status": "TARGET_WINDOW_UNAVAILABLE",
                        "frozen": False,
                        "frozen_T": None,
                        "error": target_error,
                        "partial_records": target_records,
                    }
                elif target_records:
                    target_record = aggregate_target_freeze_records(target_records)
                else:
                    target_record = {
                        "status": "NOT_COMPUTED",
                        "frozen": False,
                        "frozen_T": None,
                        "error": None,
                    }
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "protocol": PROTOCOL,
                    "artifact_id": manifest["artifact_id"],
                    "run_kind": manifest["spec"]["run_kind"],
                    "run_mode": manifest["spec"]["run_mode"],
                    "verdict": verdict,
                    "errors": [target_error] if target_error else [],
                    "planned_runs": plan,
                    "workload_fidelity": fidelity,
                    "replay_runs": replay_results,
                    "resident_preflight_runs": preflight_runs,
                    "R_AB": preflight_runs[0]["R_AB"] if len(preflight_runs) == 1 else [item["R_AB"] for item in preflight_runs],
                    "R_ABC": preflight_runs[0]["R_ABC"] if len(preflight_runs) == 1 else [item["R_ABC"] for item in preflight_runs],
                    "normalized_R_AB": preflight_runs[0]["normalized_R_AB"] if len(preflight_runs) == 1 else [item["normalized_R_AB"] for item in preflight_runs],
                    "normalized_R_ABC": preflight_runs[0]["normalized_R_ABC"] if len(preflight_runs) == 1 else [item["normalized_R_ABC"] for item in preflight_runs],
                    "exclusive_estimate": preflight_runs[0]["exclusive_estimate"] if len(preflight_runs) == 1 else [item["exclusive_estimate"] for item in preflight_runs],
                    "target": target_record,
                    "target_freeze": target_record,
                    "resident_preflight_identity": (
                        target_record.get("identity")
                        if isinstance(target_record.get("identity"), dict) else None),
                    "telemetry_gaps": TELEMETRY_GAPS,
                }
                return result["verdict"], result
            result = {
                "schema_version": SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "artifact_id": manifest["artifact_id"],
                "run_kind": manifest["spec"]["run_kind"],
                "run_mode": manifest["spec"]["run_mode"],
                "verdict": "INVALID_ARTIFACT",
                "errors": [
                    "generic replay lacks derived four-phase phase evidence; workload fidelity alone cannot qualify Q3"
                ],
                "planned_runs": plan,
                "workload_fidelity": fidelity,
                "replay_runs": replay_results,
                "telemetry_gaps": TELEMETRY_GAPS,
            }
            return result["verdict"], result
        run_results: list[dict[str, Any]] = []
        service_failures: list[dict[str, Any]] = []
        for item in plan:
            parsed = parse_run(
                artifact, item, cases[item["case_id"]], workload, len(run_results), manifest["spec"])
            run_results.append(parsed)
            service_failures.extend(parsed["service_failures"])
        completion_exact = None
        if manifest["spec"].get("profile") == FINAL_F16_PROFILE:
            completion_exact = validate_final_f16_completion_exact(run_results)
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
                "offload_physical_relief_bytes": item["offload_physical_relief_bytes"],
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
                item["offload_physical_relief_bytes"] for item in run_results),
            "total_physical_relief_bytes": (
                sum(item["release_physical_relief_bytes"] for item in characterization_runs)
                + sum(item["offload_physical_relief_bytes"] for item in run_results)
                if all(item["release_physical_relief_bytes"] is not None
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
                    if all(item["release_physical_relief_bytes"] is not None
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
                run_results,
                manifest["spec"]["run_kind"],
                manifest["spec"].get("budget_sweep"),
                completion_exact=completion_exact,
            )
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
            **({
                "profile": manifest["spec"]["profile"],
                "runtime_contract": dict(manifest["spec"]["runtime_contract"]),
                "global_governor_isolation": dict(manifest["spec"]["global_governor_isolation"]),
            } if manifest["spec"].get("profile") == FINAL_F16_PROFILE else {}),
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
            "completion_exact": completion_exact,
            "budget_curve": (
                characterization_summary.get("budget_curve")
                if characterization_summary is not None else None),
            "final_kv_lifecycle_curve": (
                characterization_summary.get("final_kv_lifecycle_curve")
                if characterization_summary is not None else None),
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
    if status == "RESIDENT_PREFLIGHT_PASS":
        write_target_freeze_record(artifact, result)
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
