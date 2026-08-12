#!/usr/bin/env python3
"""Canonical fact-only runner for the Formal OFFLOAD Benchmark.

The runner owns experiment planning, process/request lifecycle, and raw artifact
capture.  It never writes a performance verdict or derives aggregate metrics;
parse-kv-offload-benchmark.py is the only verdict authority.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import math
import json
import os
import pathlib
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from multi_session_replay import ReplayError, check_fidelity, expand_schedule, load_replay

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = pathlib.Path(__file__).resolve()
PARSER = ROOT / "scripts" / "parse-kv-offload-benchmark.py"
MEMORY_SAMPLER = ROOT / "scripts" / "kv-controlled-memory-sampler.sh"
PROTOCOL = "kv_offload_benchmark"
SCHEMA_VERSION = 2
SUPPORTED_POLICIES = {"resident", "release_only", "v2", "idle_age", "v3"}
BUDGET_POLICIES = {"release_only", "v2", "idle_age", "v3"}
SWAP_POLICIES = {"v2", "idle_age", "v3"}
SUPPORTED_KV_REPRESENTATIONS = {"paged"}
SUPPORTED_LOADING_MODES = {"exact"}
SUPPORTED_RESTORES = {"k1_sync", "k2_pipeline"}
SUPPORTED_PREFAULTS = {"off", "r2"}
SUPPORTED_RUN_KINDS = {"qualification", "formal"}
SUPPORTED_RUN_MODES = {"qualification", "characterization"}
CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAMPLE_SCHEMA = "v2"
DEFAULT_SAMPLE_INTERVAL = 0.10
DEFAULT_MAX_BLOCKS = 64
DEFAULT_HEALTH_TIMEOUT = 30.0
DEFAULT_REQUEST_TIMEOUT = 180.0
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
# Reference-only anchors retained for manifest metadata; they never drive a run.
REFERENCE_B_FULL_BYTES = 3_221_028_864
REFERENCE_B_RELEASE_FLOOR_BYTES = 1_386_479_616
BUDGET_SWEEP_ORDER_MODE = "interleaved_reverse"
CANONICAL_SERVER_OPTIONS = {"-m", "--model", "--host", "--port"}
IDENTITY_BIND_TIMEOUT_SEC = 0.5
PRESSURE_BASIS_AUTHORITIES = {"cgroup_finite", "rss_absolute"}
DEFAULT_PRESSURE_BASIS = {
    "authority": "cgroup_finite",
    "low_water_kb": None,
    "pressure_kb": None,
    "critical_kb": None,
}


class RunnerError(Exception):
    """A user-visible plan or lifecycle error without a parser verdict."""


class UnsupportedPlan(RunnerError):
    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


def require_finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunnerError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RunnerError(f"{label} must be a finite positive number")
    return result


def reject_canonical_server_args(args: list[str]) -> None:
    for item in args:
        if item in CANONICAL_SERVER_OPTIONS or item.startswith(("--model=", "--host=", "--port=")):
            raise RunnerError("spec.server_args must not override -m/--model/--host/--port")
        if item.startswith("-m") and not item.startswith("--"):
            raise RunnerError("spec.server_args must not override -m/--model/--host/--port")


def reject_canonical_kv_environment(environment: dict[str, str]) -> None:
    conflicts = sorted(key for key in environment if key.startswith("LLAMA_KV_"))
    if conflicts:
        raise RunnerError(f"spec.environment conflicts with canonical KV env: {conflicts}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def dump(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: pathlib.Path) -> dict[str, Any]:
    value: dict[str, Any] = {"path": str(path), "present": path.is_file()}
    if path.is_file():
        value.update({"size": path.stat().st_size, "sha256": sha256_file(path)})
    else:
        value.update({"size": None, "sha256": None})
    return value


def run_git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
        encoding="utf-8",
        stderr=subprocess.DEVNULL,
    ).strip()


def git_provenance() -> dict[str, Any]:
    status = run_git("status", "--short", "--untracked-files=all")
    diff = subprocess.check_output(
        ["git", "-C", str(ROOT), "diff", "--no-ext-diff", "--binary", "HEAD"],
        encoding="utf-8",
    )
    return {
        "head": run_git("rev-parse", "HEAD"),
        "branch": run_git("branch", "--show-current"),
        "dirty_status": status.splitlines() if status else [],
        "diff_sha256": sha256_bytes(diff.encode("utf-8")),
        "capture_mode": "diagnostic_dirty" if status else "archival_clean",
    }


def read_first_line(path: pathlib.Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as stream:
            return stream.readline().strip()
    except OSError:
        return None


def process_starttime(pid: int) -> str | None:
    stat_path = pathlib.Path(f"/proc/{pid}/stat")
    try:
        text = stat_path.read_text(encoding="utf-8")
    except OSError:
        return None
    tail = text.rsplit(") ", 1)[-1].split()
    return tail[19] if len(tail) > 19 and tail[19].isdigit() else None


def process_cmdline(pid: int) -> list[str]:
    cmdline_path = pathlib.Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except OSError as exc:
        raise RunnerError(f"cannot read process cmdline for pid {pid}") from exc
    if not raw or not raw.endswith(b"\0"):
        raise RunnerError(f"process cmdline is malformed for pid {pid}")
    try:
        cmdline = [item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item]
    except UnicodeDecodeError as exc:
        raise RunnerError(f"process cmdline is not UTF-8 for pid {pid}") from exc
    if not cmdline or any(not item for item in cmdline):
        raise RunnerError(f"process cmdline is empty for pid {pid}")
    return cmdline


def process_identity(
        pid: int,
        fallback_argv: list[str],
        timeout_seconds: float = IDENTITY_BIND_TIMEOUT_SEC,
) -> dict[str, Any]:
    if isinstance(pid, bool) or pid <= 0:
        raise RunnerError(f"cannot bind invalid process pid {pid}")
    timeout_seconds = require_finite_positive(timeout_seconds, "process identity bind timeout")
    deadline_ns = time.monotonic_ns() + int(timeout_seconds * 1_000_000_000)
    stable_starttime: str | None = None
    stable_cmdline: list[str] | None = None
    last_error = f"cannot bind process identity for pid {pid}"
    attempt = 0
    while True:
        starttime = process_starttime(pid)
        if starttime is None:
            last_error = f"cannot bind process starttime for pid {pid}"
        else:
            try:
                cmdline = process_cmdline(pid)
            except RunnerError as exc:
                last_error = str(exc)
            else:
                endtime = process_starttime(pid)
                if endtime is None:
                    last_error = f"cannot recheck process starttime for pid {pid}"
                elif endtime != starttime:
                    last_error = f"process starttime changed while binding pid {pid}"
                elif stable_starttime == starttime and stable_cmdline == cmdline:
                    return {
                        "pid": pid,
                        "starttime_ticks": int(starttime),
                        "cmdline": cmdline,
                        "cmdline_sha256": sha256_bytes(
                            json.dumps(cmdline, separators=(",", ":")).encode("utf-8")),
                    }
                else:
                    stable_starttime = starttime
                    stable_cmdline = cmdline
                    last_error = f"process identity is not yet stable for pid {pid}"

        now_ns = time.monotonic_ns()
        if now_ns >= deadline_ns:
            raise RunnerError(last_error)
        remaining_seconds = (deadline_ns - now_ns) / 1_000_000_000
        backoff_seconds = min(0.001 * (2 ** min(attempt, 5)), 0.05)
        time.sleep(min(backoff_seconds, remaining_seconds))
        attempt += 1


def cgroup_identity(pid: int) -> dict[str, Any]:
    path = pathlib.Path(f"/proc/{pid}/cgroup")
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    v2_rel: str | None = None
    v1_rel: str | None = None
    membership: list[str] = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, rel = parts
        membership.append(line)
        if hierarchy == "0" and not controllers:
            v2_rel = rel
        elif "memory" in controllers.split(","):
            v1_rel = rel
    if v2_rel is not None:
        root = pathlib.Path("/sys/fs/cgroup") / v2_rel.lstrip("/")
        if root.is_dir():
            return {
                "version": "v2",
                "path": str(root),
                "membership": membership,
                "memory_current_file": str(root / "memory.current"),
                "memory_peak_file": str(root / "memory.peak"),
                "memory_swap_current_file": str(root / "memory.swap.current"),
                "memory_events_file": str(root / "memory.events"),
                "memory_max": read_first_line(root / "memory.max"),
                "memory_high": read_first_line(root / "memory.high"),
                "memory_swap_max": read_first_line(root / "memory.swap.max"),
            }
    if v1_rel is not None:
        root = pathlib.Path("/sys/fs/cgroup/memory") / v1_rel.lstrip("/")
        if root.is_dir():
            return {
                "version": "v1",
                "path": str(root),
                "membership": membership,
                "memory_current_file": str(root / "memory.usage_in_bytes"),
                "memory_peak_file": str(root / "memory.max_usage_in_bytes"),
                "memory_swap_current_file": None,
                "memory_events_file": None,
                "memory_max": read_first_line(root / "memory.limit_in_bytes"),
                "memory_high": read_first_line(root / "memory.soft_limit_in_bytes"),
                "memory_swap_max": None,
            }
    return {
        "version": "none",
        "path": None,
        "membership": membership,
        "memory_current_file": None,
        "memory_peak_file": None,
        "memory_swap_current_file": None,
        "memory_events_file": None,
        "memory_max": None,
        "memory_high": None,
        "memory_swap_max": None,
    }


def cgroup_scope(record: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    path = value.get("path")
    reference_path = reference.get("path")
    if path is None:
        value["scope"] = "none"
    elif reference_path is not None and path == reference_path:
        value["scope"] = "shared"
    else:
        value["scope"] = "dedicated"
    return value


def finite_memory_limit(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[1-9][0-9]*", value))


def validate_pressure_authority(
        spec: dict[str, Any], plan: list[dict[str, Any]]) -> None:
    if not any(item["policy"] in BUDGET_POLICIES for item in plan):
        return
    basis = spec["pressure_basis"]
    if spec["run_kind"] == "formal" and basis["authority"] != "cgroup_finite":
        raise UnsupportedPlan(
            "pre_workload_pressure_authority",
            "formal budget run requires real finite cgroup pressure authority",
        )
    if basis["authority"] != "cgroup_finite":
        return
    expected = spec["cgroup"]["expected_memory_max"]
    if not finite_memory_limit(expected):
        raise UnsupportedPlan(
            "pre_workload_pressure_authority",
            "finite cgroup pressure authority requires cgroup.expected_memory_max",
        )
    observed = cgroup_identity(os.getpid())
    if observed.get("version") not in {"v1", "v2"} or not finite_memory_limit(observed.get("memory_max")):
        raise UnsupportedPlan(
            "pre_workload_pressure_authority",
            "current runner cgroup has no finite memory.max authority",
        )
    if observed.get("memory_max") != expected:
        raise UnsupportedPlan(
            "pre_workload_pressure_authority",
            f"runner cgroup memory.max does not match expected value: expected={expected!r} actual={observed.get('memory_max')!r}",
        )


def host_provenance() -> dict[str, Any]:
    uname = platform.uname()
    mem_total = None
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            mem_total = line.split()[1]
            break
    return {
        "platform": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        "cpu_count": os.cpu_count(),
        "page_size": os.sysconf("SC_PAGE_SIZE"),
        "mem_total_kb": int(mem_total) if mem_total and mem_total.isdigit() else None,
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def validate_mapping(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunnerError(f"{label} must be an object")
    missing = expected - set(value)
    extra = set(value) - expected
    if missing or extra:
        raise RunnerError(f"{label} schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
    return value


def normalize_request(value: Any, label: str) -> dict[str, Any]:
    item = validate_mapping(value, {"request_id", "prompt", "n_predict", "stream"}, label)
    request_id = item["request_id"]
    if not isinstance(request_id, str) or not CASE_ID_RE.fullmatch(request_id):
        raise RunnerError(f"{label}.request_id is invalid")
    if not isinstance(item["prompt"], str):
        raise RunnerError(f"{label}.prompt must be a string")
    if isinstance(item["n_predict"], bool) or not isinstance(item["n_predict"], int) or item["n_predict"] < 0:
        raise RunnerError(f"{label}.n_predict must be a non-negative integer")
    if not isinstance(item["stream"], bool):
        raise RunnerError(f"{label}.stream must be boolean")
    return dict(item)


def normalize_pressure_basis(value: Any) -> dict[str, Any]:
    basis = validate_mapping(
        value,
        {"authority", "low_water_kb", "pressure_kb", "critical_kb"},
        "pressure_basis",
    )
    authority = basis["authority"]
    if authority not in PRESSURE_BASIS_AUTHORITIES:
        raise RunnerError(f"pressure_basis.authority is unsupported: {authority!r}")
    thresholds = [basis[key] for key in ("low_water_kb", "pressure_kb", "critical_kb")]
    if authority == "cgroup_finite":
        if any(value is not None for value in thresholds):
            raise RunnerError("cgroup_finite pressure_basis cannot define RSS thresholds")
    else:
        if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in thresholds):
            raise RunnerError("rss_absolute pressure_basis thresholds must be positive integers")
        if not (thresholds[0] < thresholds[1] < thresholds[2]):
            raise RunnerError("RSS pressure basis thresholds must satisfy low < pressure < critical")
    return {
        "authority": authority,
        "low_water_kb": thresholds[0],
        "pressure_kb": thresholds[1],
        "critical_kb": thresholds[2],
    }


def normalize_replay(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and "lifecycle" not in value:
        value = {**value, "lifecycle": None}
    replay = validate_mapping(
        value,
        {"source", "path", "time_dilation", "n_parallel", "session_ids", "admission_timeout_seconds", "lifecycle"},
        "workload.replay",
    )
    if replay["source"] not in {"transcript", "fixture"}:
        raise RunnerError("workload.replay.source must be transcript or fixture")
    if not isinstance(replay["path"], str) or not replay["path"]:
        raise RunnerError("workload.replay.path must be a non-empty string")
    time_dilation = replay["time_dilation"]
    if isinstance(time_dilation, bool) or not isinstance(time_dilation, (int, float)) or not math.isfinite(float(time_dilation)) or time_dilation < 0:
        raise RunnerError("workload.replay.time_dilation must be finite and non-negative")
    n_parallel = replay["n_parallel"]
    if isinstance(n_parallel, bool) or not isinstance(n_parallel, int) or n_parallel <= 0:
        raise RunnerError("workload.replay.n_parallel must be positive")
    session_ids = replay["session_ids"]
    if session_ids is not None and (
        not isinstance(session_ids, list)
        or any(not isinstance(item, str) or not item for item in session_ids)
        or len(session_ids) != len(set(session_ids))
    ):
        raise RunnerError("workload.replay.session_ids must be null or a unique string array")
    admission_timeout = require_finite_positive(
        replay["admission_timeout_seconds"], "workload.replay.admission_timeout_seconds")
    raw_lifecycle = replay["lifecycle"]
    if raw_lifecycle is None or raw_lifecycle is False:
        lifecycle = {"enabled": False, "drain_after_last_arrival": False}
    else:
        if raw_lifecycle is True:
            raw_lifecycle = {}
        if not isinstance(raw_lifecycle, dict):
            raise RunnerError("workload.replay.lifecycle must be null, boolean, or object")
        unknown = set(raw_lifecycle) - {
            "enabled", "drain_after_last_arrival", "ttl_seconds",
            "parent_manifest_path", "parent_manifest_sha256", "trace_identity",
        }
        if unknown:
            raise RunnerError(f"workload.replay.lifecycle schema mismatch extra={sorted(unknown)}")
        enabled = raw_lifecycle.get("enabled", True)
        drain = raw_lifecycle.get("drain_after_last_arrival", False)
        if not isinstance(enabled, bool) or not isinstance(drain, bool):
            raise RunnerError("workload.replay.lifecycle enabled/drain_after_last_arrival must be boolean")
        lifecycle = {"enabled": enabled, "drain_after_last_arrival": drain}
        if "ttl_seconds" in raw_lifecycle:
            lifecycle["ttl_seconds"] = require_finite_positive(raw_lifecycle["ttl_seconds"], "workload.replay.lifecycle.ttl_seconds")
        for key in ("parent_manifest_path", "parent_manifest_sha256"):
            if key in raw_lifecycle and (not isinstance(raw_lifecycle[key], str) or not raw_lifecycle[key]):
                raise RunnerError(f"workload.replay.lifecycle.{key} must be a non-empty string")
            if key in raw_lifecycle:
                lifecycle[key] = raw_lifecycle[key]
        if "trace_identity" in raw_lifecycle:
            if not isinstance(raw_lifecycle["trace_identity"], dict):
                raise RunnerError("workload.replay.lifecycle.trace_identity must be an object")
            lifecycle["trace_identity"] = dict(raw_lifecycle["trace_identity"])
    if lifecycle["enabled"] and time_dilation <= 0:
        raise RunnerError("workload.replay.lifecycle requires time_dilation > 0")
    return {
        "source": replay["source"],
        "path": replay["path"],
        "time_dilation": float(time_dilation),
        "n_parallel": n_parallel,
        "session_ids": session_ids,
        "admission_timeout_seconds": admission_timeout,
        "lifecycle": lifecycle,
    }


def load_replay_plan(replay: dict[str, Any]) -> Any:
    lifecycle_cfg = replay.get("lifecycle", {"enabled": False})
    enabled = bool(lifecycle_cfg.get("enabled", False))
    try:
        plan = load_replay(
            replay["source"], replay["path"],
            n_parallel=replay["n_parallel"],
            time_dilation=replay["time_dilation"],
            selected_session_ids=replay["session_ids"],
            lifecycle=enabled,
        )
    except (OSError, ReplayError) as exc:
        raise RunnerError(f"replay input is invalid: {exc}") from exc
    if enabled:
        source_lifecycle = plan.lifecycle or {}
        if "ttl_seconds" in lifecycle_cfg and abs(float(lifecycle_cfg["ttl_seconds"]) - float(source_lifecycle["ttl_seconds"])) > 1e-9:
            raise RunnerError("workload.replay.lifecycle TTL differs from replay source identity")
        if "parent_manifest_path" in lifecycle_cfg and pathlib.Path(lifecycle_cfg["parent_manifest_path"]).resolve() != pathlib.Path(source_lifecycle.get("parent_manifest_path", "")).resolve():
            raise RunnerError("workload.replay.lifecycle parent manifest path mismatch")
        if "parent_manifest_sha256" in lifecycle_cfg and source_lifecycle.get("parent_manifest_sha256") != lifecycle_cfg["parent_manifest_sha256"]:
            raise RunnerError("workload.replay.lifecycle parent manifest SHA mismatch")
        expected_identity = lifecycle_cfg.get("trace_identity")
        if expected_identity is not None and expected_identity != source_lifecycle.get("trace_identity"):
            raise RunnerError("workload.replay.lifecycle trace identity mismatch")
    return plan


def normalize_workload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunnerError("workload must be an object")
    replay = normalize_replay(value["replay"]) if "replay" in value else None
    expected_keys = {"warmup", "requests", "repeat", "qualification", "characterization"}
    if replay is not None:
        expected_keys.add("replay")
    workload = validate_mapping(value, expected_keys, "workload")
    if not isinstance(workload["warmup"], list) or not isinstance(workload["requests"], list):
        raise RunnerError("workload warmup/requests must be arrays")
    warmup = [normalize_request(item, f"workload.warmup[{i}]") for i, item in enumerate(workload["warmup"])]
    requests = [normalize_request(item, f"workload.requests[{i}]") for i, item in enumerate(workload["requests"])]
    if replay is not None:
        if warmup or requests or workload["qualification"] is not None or workload["characterization"] is not None:
            raise RunnerError("workload.replay cannot be combined with legacy requests or qualification/characterization")
        return {
            "warmup": [], "requests": [], "repeat": 1,
            "qualification": None, "characterization": None, "replay": replay,
        }
    ids = [item["request_id"] for item in warmup + requests]
    if len(ids) != len(set(ids)):
        raise RunnerError("workload request_id values must be unique")
    repeat = workload["repeat"]
    if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= 1000:
        raise RunnerError("workload.repeat must be in [1, 1000]")
    if not requests:
        raise RunnerError("workload.requests must contain at least one measurement request")

    qualification = workload["qualification"]
    if qualification is not None:
        qualification = validate_mapping(
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
                item["request_id"] for item in requests}:
            raise RunnerError(
                "workload.qualification.resume_request_id must reference a measurement request")
        if not warmup:
            raise RunnerError("qualification requires at least one warmup request")
        qualification = {
            "idle_seconds": idle_seconds,
            "offload_timeout_seconds": offload_timeout_seconds,
            "resume_request_id": resume_request_id,
        }

    characterization = workload["characterization"]
    if characterization is not None:
        characterization = validate_mapping(
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
            raise RunnerError(
                "workload.characterization.target_tolerance_bytes must be a non-negative integer")
        resume_request_id = characterization["resume_request_id"]
        if not isinstance(resume_request_id, str) or resume_request_id not in {
                item["request_id"] for item in requests}:
            raise RunnerError(
                "workload.characterization.resume_request_id must reference a measurement request")
        if resume_request_id != requests[0]["request_id"]:
            raise RunnerError(
                "workload.characterization.resume_request_id must be the first measurement request")
        if not warmup:
            raise RunnerError("characterization requires at least one warmup/fill request")
        characterization = {
            "idle_seconds": idle_seconds,
            "settle_timeout_seconds": settle_timeout_seconds,
            "target_tolerance_bytes": tolerance,
            "resume_request_id": resume_request_id,
        }

    return {
        "warmup": warmup,
        "requests": requests,
        "repeat": repeat,
        "qualification": qualification,
        "characterization": characterization,
    }


def normalize_cases(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RunnerError("cases must be a non-empty array")
    cases: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(value):
        item = validate_mapping(
            raw,
            {
                "case_id", "policy", "kv_representation", "loading_mode", "restore", "prefault",
                "kv_target_bytes", "action_target_bytes",
            },
            f"cases[{index}]",
        )
        case_id = item["case_id"]
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise RunnerError(f"cases[{index}].case_id is invalid")
        if case_id in cases:
            raise RunnerError(f"duplicate case_id: {case_id}")
        for factor in ("policy", "kv_representation", "loading_mode", "restore", "prefault"):
            if not isinstance(item[factor], str):
                raise RunnerError(f"{case_id}.{factor} must be a string")
        target = item["kv_target_bytes"]
        action_target = item["action_target_bytes"]
        if target is not None and (isinstance(target, bool) or not isinstance(target, int) or target <= 0):
            raise RunnerError(f"{case_id}.kv_target_bytes must be null or positive")
        if action_target is not None and (
                isinstance(action_target, bool) or not isinstance(action_target, int) or action_target <= 0):
            raise RunnerError(f"{case_id}.action_target_bytes must be null or positive")
        if item["policy"] == "resident":
            if target is not None or action_target is not None:
                raise RunnerError(f"{case_id}: resident policy cannot have resident/action targets")
        elif item["policy"] in BUDGET_POLICIES and (target is None or action_target is None):
            raise RunnerError(
                f"{case_id}: {item['policy']} policy requires explicit kv_target_bytes and action_target_bytes")
        cases[case_id] = dict(item)
    action_targets = {
        case["action_target_bytes"] for case in cases.values()
        if case["policy"] in BUDGET_POLICIES
    }
    if len(action_targets) > 1:
        raise RunnerError(
            "all budget cases must use the same explicit action_target_bytes")
    return cases


def normalize_plan(value: Any, cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RunnerError("run_order must be a non-empty array")
    plan: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    run_ids: set[str] = set()
    for index, raw in enumerate(value):
        item = validate_mapping(raw, {"round", "run_order", "case_id"}, f"run_order[{index}]")
        round_id, order = item["round"], item["run_order"]
        case_id = item["case_id"]
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id <= 0 or isinstance(order, bool) or not isinstance(order, int) or order <= 0:
            raise RunnerError(f"run_order[{index}] round/run_order must be positive integers")
        if case_id not in cases:
            raise RunnerError(f"run_order[{index}] references unknown case: {case_id}")
        key = (round_id, order)
        if key in seen:
            raise RunnerError(f"duplicate run key: {key}")
        seen.add(key)
        run_id = f"r{round_id:03d}_o{order:03d}_{safe_component(case_id)}"
        if run_id in run_ids:
            raise RunnerError(f"run_id collision: {run_id}")
        run_ids.add(run_id)
        case = cases[case_id]
        plan.append({
            "run_id": run_id,
            "round": round_id,
            "run_order": order,
            "case_id": case_id,
            "policy": case["policy"],
            "kv_representation": case["kv_representation"],
            "loading_mode": case["loading_mode"],
            "restore": case["restore"],
            "prefault": case["prefault"],
            "kv_target_bytes": case["kv_target_bytes"],
            "action_target_bytes": case["action_target_bytes"],
        })
    return plan


def budget_case_signature(item: dict[str, Any]) -> tuple[str, int | None]:
    return item["policy"], item["kv_target_bytes"]


def normalize_budget_sweep(
        value: Any,
        cases: dict[str, dict[str, Any]],
        plan: list[dict[str, Any]],
        max_blocks: int,
        run_kind: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    sweep = validate_mapping(
        value,
        {
            "release_targets_bytes", "offload_targets_bytes", "action_target_bytes",
            "max_blocks", "rounds", "order_mode",
        },
        "budget_sweep",
    )

    def target_list(raw: Any, label: str) -> tuple[int, ...]:
        if not isinstance(raw, list) or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in raw):
            raise RunnerError(f"{label} must be a list of positive absolute byte values")
        if len(raw) != len(set(raw)) or raw != sorted(raw, reverse=True):
            raise RunnerError(f"{label} must be unique and in descending explicit-byte order")
        return tuple(raw)

    release_targets = target_list(
        sweep["release_targets_bytes"], "budget_sweep.release_targets_bytes")
    offload_targets = target_list(
        sweep["offload_targets_bytes"], "budget_sweep.offload_targets_bytes")
    if release_targets != CANONICAL_BUDGET_RELEASE_TARGETS:
        raise RunnerError(
            "budget_sweep.release_targets_bytes must use the canonical 2.50/2.00/1.50 GiB bytes")
    if offload_targets != CANONICAL_BUDGET_OFFLOAD_TARGETS:
        raise RunnerError(
            "budget_sweep.offload_targets_bytes must use the canonical 1.00/0.75/0.50/0.25 GiB bytes")
    action_target = sweep["action_target_bytes"]
    if isinstance(action_target, bool) or not isinstance(action_target, int) or action_target <= 0:
        raise RunnerError("budget_sweep.action_target_bytes must be a positive absolute byte value")
    if action_target != CANONICAL_BUDGET_ACTION_TARGET_BYTES:
        raise RunnerError("budget_sweep.action_target_bytes must be exactly 256 MiB")
    if any(
            item["policy"] in BUDGET_POLICIES
            and item["action_target_bytes"] != action_target
            for item in cases.values()):
        raise RunnerError("budget_sweep.action_target_bytes differs from a budget case")
    sweep_max_blocks = sweep["max_blocks"]
    if isinstance(sweep_max_blocks, bool) or not isinstance(sweep_max_blocks, int) or sweep_max_blocks <= 0:
        raise RunnerError("budget_sweep.max_blocks must be a positive integer")
    if sweep_max_blocks != CANONICAL_BUDGET_MAX_BLOCKS or max_blocks != sweep_max_blocks:
        raise RunnerError("budget_sweep and spec.max_blocks must both be exactly 64")
    rounds = sweep["rounds"]
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds <= 0:
        raise RunnerError("budget_sweep.rounds must be a positive integer")
    if run_kind == "formal" and rounds != 2:
        raise RunnerError("formal budget_sweep requires exactly two independent rounds")
    if sweep["order_mode"] != BUDGET_SWEEP_ORDER_MODE:
        raise RunnerError("budget_sweep.order_mode must be interleaved_reverse")

    expected_signatures = [("resident", None)]
    expected_signatures.extend(("release_only", target) for target in release_targets)
    expected_signatures.extend(("v2", target) for target in offload_targets)
    expected_set = set(expected_signatures)
    planned_rounds: dict[int, list[dict[str, Any]]] = {}
    for item in plan:
        planned_rounds.setdefault(item["round"], []).append(item)
    if sorted(planned_rounds) != list(range(1, rounds + 1)):
        raise RunnerError("budget_sweep rounds must be numbered consecutively from one")
    for round_id, entries in planned_rounds.items():
        if [item["run_order"] for item in entries] != list(range(1, len(entries) + 1)):
            raise RunnerError(f"budget_sweep round {round_id} run_order must be consecutive")
        signatures = [budget_case_signature(item) for item in entries]
        if len(entries) != len(expected_signatures) or set(signatures) != expected_set:
            raise RunnerError(
                f"budget_sweep round {round_id} must contain one Resident, all RELEASE targets, and all V2 targets")
        if len(signatures) != len(set(signatures)):
            raise RunnerError(f"budget_sweep round {round_id} contains duplicate target cases")
        budget_signatures = [signature for signature in signatures if signature[0] != "resident"]
        if any(
                budget_signatures[index][0] == budget_signatures[index + 1][0]
                for index in range(len(budget_signatures) - 1)):
            raise RunnerError(
                f"budget_sweep round {round_id} must interleave RELEASE and V2 cases")
    if rounds == 2:
        first = [budget_case_signature(item) for item in planned_rounds[1]]
        second = [budget_case_signature(item) for item in planned_rounds[2]]
        if second != list(reversed(first)):
            raise RunnerError("budget_sweep rounds must use reverse order without randomization")

    return {
        "release_targets_bytes": list(release_targets),
        "offload_targets_bytes": list(offload_targets),
        "action_target_bytes": action_target,
        "max_blocks": sweep_max_blocks,
        "rounds": rounds,
        "order_mode": sweep["order_mode"],
    }


def validate_spec(raw: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if isinstance(raw, dict):
        raw = dict(raw)
        raw.setdefault("pressure_basis", dict(DEFAULT_PRESSURE_BASIS))
    spec_keys = {
        "schema_version", "protocol", "phase", "run_kind", "run_mode", "binary", "model", "model_quantization",
        "server_args", "environment", "pressure_basis", "workload", "cases", "run_order", "sampler",
        "cgroup", "max_blocks", "health_timeout_seconds", "request_timeout_seconds",
    }
    if isinstance(raw, dict) and "budget_sweep" in raw:
        spec_keys.add("budget_sweep")
    spec = validate_mapping(raw, spec_keys, "spec")
    if spec["schema_version"] != SCHEMA_VERSION or spec["protocol"] != PROTOCOL:
        raise RunnerError("spec protocol/schema version is unsupported")
    if spec["phase"] not in {"resident_baseline", "coarse_target", "local_target", "representative"}:
        raise RunnerError("spec.phase is invalid")
    if spec["run_kind"] not in SUPPORTED_RUN_KINDS:
        raise RunnerError("spec.run_kind is invalid")
    if spec["run_mode"] not in SUPPORTED_RUN_MODES:
        raise RunnerError("spec.run_mode is invalid")
    for key in ("binary", "model", "model_quantization"):
        if not isinstance(spec[key], str) or not spec[key]:
            raise RunnerError(f"spec.{key} must be a non-empty string")
    if not isinstance(spec["server_args"], list) or any(not isinstance(item, str) for item in spec["server_args"]):
        raise RunnerError("spec.server_args must be a string array")
    reject_canonical_server_args(spec["server_args"])
    if isinstance(spec.get("workload"), dict) and "replay" in spec["workload"]:
        replay_args = set(spec["server_args"])
        if replay_args.intersection({"--parallel", "-np", "--cache-idle-slots", "--no-cache-idle-slots", "--context-shift", "--no-context-shift"}):
            raise RunnerError("replay workload owns --parallel/cache-idle/context-shift server options")
    if not isinstance(spec["environment"], dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in spec["environment"].items()
    ):
        raise RunnerError("spec.environment must map strings to strings")
    reject_canonical_kv_environment(spec["environment"])
    pressure_basis = normalize_pressure_basis(spec["pressure_basis"])
    sampler = validate_mapping(spec["sampler"], {"interval_seconds"}, "sampler")
    sampler_interval = require_finite_positive(sampler["interval_seconds"], "sampler.interval_seconds")
    cgroup = validate_mapping(spec["cgroup"], {"expected_memory_max"}, "cgroup")
    if cgroup["expected_memory_max"] is not None and not isinstance(cgroup["expected_memory_max"], str):
        raise RunnerError("cgroup.expected_memory_max must be null or a string")
    max_blocks = spec["max_blocks"]
    if isinstance(max_blocks, bool) or not isinstance(max_blocks, int) or max_blocks <= 0:
        raise RunnerError("max_blocks must be positive")
    health_timeout = require_finite_positive(spec["health_timeout_seconds"], "health_timeout_seconds")
    request_timeout = require_finite_positive(spec["request_timeout_seconds"], "request_timeout_seconds")
    workload = normalize_workload(spec["workload"])
    if ("replay" in workload and workload["replay"].get("lifecycle", {}).get("enabled", False)
            and any(item == "--slot-save-path" or item.startswith("--slot-save-path=") for item in spec["server_args"])):
        raise RunnerError("lifecycle replay owns --slot-save-path server option")
    cases = normalize_cases(spec["cases"])
    plan = normalize_plan(spec["run_order"], cases)
    budget_sweep = normalize_budget_sweep(
        spec.get("budget_sweep"), cases, plan, max_blocks, spec["run_kind"])
    if (
        budget_sweep is not None
        and spec["run_kind"] == "formal"
        and spec["run_mode"] == "characterization"
        and workload["repeat"] < 2
    ):
        raise RunnerError("formal budget_sweep requires repeat >= 2 for post-resume steady measurements")
    if any(item["policy"] == "release_only" for item in plan) and spec["run_mode"] != "characterization":
        raise RunnerError("release_only policy is available only in characterization mode")
    if "replay" in workload:
        if spec["run_mode"] != "qualification" or spec["run_kind"] != "qualification":
            raise RunnerError("replay workload is restricted to qualification mode")
    elif spec["run_mode"] == "qualification":
        if workload["qualification"] is None or workload["characterization"] is not None:
            raise RunnerError(
                "qualification mode requires only workload.qualification configuration")
    elif workload["characterization"] is None or workload["qualification"] is not None:
        raise RunnerError(
            "characterization mode requires only workload.characterization configuration")
    if spec["run_kind"] == "formal":
        if len({item["round"] for item in plan}) < 2:
            raise RunnerError("formal run requires at least two independent rounds")
        if spec["run_mode"] == "qualification" and workload["repeat"] < 2:
            raise RunnerError("formal qualification run requires repeat >= 2")
    normalized = dict(spec)
    normalized["pressure_basis"] = pressure_basis
    normalized["workload"] = workload
    normalized["sampler"] = {"interval_seconds": sampler_interval}
    normalized["cgroup"] = dict(cgroup)
    normalized["max_blocks"] = max_blocks
    normalized["health_timeout_seconds"] = health_timeout
    normalized["request_timeout_seconds"] = request_timeout
    if budget_sweep is not None:
        normalized["budget_sweep"] = budget_sweep
    return normalized, cases, plan, workload


def expanded_request_plan(
        workload: dict[str, Any], policy: str, run_mode: str) -> list[dict[str, Any]]:
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
            if run_mode == "characterization" and policy in SWAP_POLICIES:
                measurement_phase = (
                    "resume" if repeat_index == 1 and request_index == 0
                    else "post_resume_steady")
            elif run_mode == "characterization" and policy == "release_only":
                measurement_phase = "release_only_steady"
            elif run_mode == "characterization":
                measurement_phase = "resident_steady"
            else:
                measurement_phase = "qualification_measurement"
            append(item, item["request_id"], repeat_index, True, measurement_phase)
    if run_mode == "qualification" and workload["qualification"] is not None and policy in SWAP_POLICIES:
        resume_request_id = workload["qualification"]["resume_request_id"]
        resume_request = next(
            item for item in workload["requests"] if item["request_id"] == resume_request_id)
        append(resume_request, resume_request_id, 0, False, "qualification_resume")
    return result


def runtime_environment(
        spec: dict[str, Any], case: dict[str, Any], backing_dir: pathlib.Path) -> dict[str, str]:
    env = {
        "HOME": "/tmp",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_INGRAPH": "1",
        "LLAMA_KV_PAGED_SWAP": "1" if case["policy"] in SWAP_POLICIES else "0",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
        "LLAMA_KV_PAGED_MINCORE": "1",
        "LLAMA_KV_PAGED_IO_STATS": "1",
        "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "0",
        "LLAMA_KV_RESUME_STAGE_TIMING": "1",
        "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": (
            "both" if case["policy"] in SWAP_POLICIES and spec["run_mode"] == "characterization"
            else "1" if case["policy"] in SWAP_POLICIES else "preflight"),
        "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "100",
        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
        # The manifest environment may override this for V3 controlled A/B;
        # the default keeps canonical V2 behavior unchanged.
        "LLAMA_KV_PRESSURE_POLICY": case["policy"] if case["policy"] in SWAP_POLICIES else "v2",
        "LLAMA_KV_PRESSURE_SAMPLER": "1",
        "LLAMA_KV_PAGED_RESTORE_K2": "1" if case["restore"] == "k2_pipeline" else "0",
        "LLAMA_KV_PAGED_RESTORE_PREFAULT_PROBE": "1" if case["prefault"] == "r2" else "0",
        "LLAMA_KV_SWAP_DIR": str(backing_dir),
    }
    pressure_basis = spec["pressure_basis"]
    if pressure_basis["authority"] == "rss_absolute":
        env.update({
            "LLAMA_KV_LOW_WATER_RSS_KB": str(pressure_basis["low_water_kb"]),
            "LLAMA_KV_PRESSURE_RSS_KB": str(pressure_basis["pressure_kb"]),
            "LLAMA_KV_CRITICAL_RSS_KB": str(pressure_basis["critical_kb"]),
        })
    env.update(spec["environment"])
    if case["policy"] == "resident":
        env.update({
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "0",
            "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "0",
        })
        for key in (
            "LLAMA_KV_RESIDENT_TARGET_BYTES",
            "LLAMA_KV_RESIDENT_TARGET_SOURCE",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS",
        ):
            env.pop(key, None)
    elif case["policy"] in BUDGET_POLICIES:
        resident_target = str(case["kv_target_bytes"])
        action_target = str(case["action_target_bytes"])
        env.update({
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
            "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "1",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": action_target,
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": str(spec["max_blocks"]),
            "LLAMA_KV_RESIDENT_TARGET_BYTES": resident_target,
            "LLAMA_KV_RESIDENT_TARGET_SOURCE": "env_static",
        })
    return env


def server_argv(spec: dict[str, Any], port: int) -> list[str]:
    args = [
        spec["binary"],
        "--host", "127.0.0.1",
        "--port", str(port),
        "--model", spec["model"],
        *spec["server_args"],
    ]
    replay = spec.get("workload", {}).get("replay") if isinstance(spec.get("workload"), dict) else None
    if replay is not None:
        args.extend(["--parallel", str(replay["n_parallel"]), "--no-cache-idle-slots", "--no-context-shift"])
    return args


def wait_health(port: int, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "health endpoint not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunnerError(f"server exited before health check: {process.returncode}")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            connection.close()
            if 200 <= response.status < 300:
                return
            last_error = f"HTTP {response.status}"
        except (OSError, http.client.HTTPException) as exc:
            last_error = str(exc)
        time.sleep(0.05)
    raise RunnerError(f"server health timeout: {last_error}")


def capture_slots(port: int, path: pathlib.Path, timeout: float) -> dict[str, Any]:
    started_mono = time.monotonic_ns()
    status = 0
    body = b""
    error: str | None = None
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        connection.request("GET", "/slots")
        response = connection.getresponse()
        status = int(response.status)
        body = response.read()
        connection.close()
    except (OSError, http.client.HTTPException) as exc:
        error = str(exc)
    parsed: Any = None
    try:
        parsed = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    body_path = path.with_suffix(".body")
    body_path.write_bytes(body)
    snapshot = {
        "captured_mono_ns": started_mono,
        "http_status": status,
        "body_path": body_path.name,
        "body_bytes": len(body),
        "body_sha256": sha256_bytes(body),
        "body_json": parsed,
        "error": error,
    }
    dump(path, snapshot)
    return snapshot


def captured_resident_bytes(
        snapshot: dict[str, Any], *, allow_missing: bool = False) -> int | None:
    slots = snapshot.get("body_json")
    if not isinstance(slots, list):
        raise RunnerError("slot snapshot body is unavailable")
    values: list[int] = []
    for slot in slots:
        resident = slot.get("kv_resident") if isinstance(slot, dict) else None
        if not isinstance(resident, dict) or resident.get("status") != "available":
            continue
        value = resident.get("resident_bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RunnerError("slot physical resident value is invalid")
        values.append(value)
    if not values:
        if allow_missing:
            return None
        raise RunnerError("slot physical resident observation is missing or inconsistent")
    if any(value != values[0] for value in values[1:]):
        raise RunnerError("slot physical resident observation is missing or inconsistent")
    return values[0]


def request_completion(
        port: int,
        plan_item: dict[str, Any],
        request: dict[str, Any],
        raw_dir: pathlib.Path,
        timeout: float) -> dict[str, Any]:
    body = {
        "prompt": request["prompt"],
        "n_predict": request["n_predict"],
        "cache_prompt": True,
        "id_slot": 0,
        "stream": request["stream"],
    }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    started_mono = time.monotonic_ns()
    started_realtime = time.time_ns()
    first_byte_mono: int | None = None
    finished_mono = started_mono
    status = 0
    headers: list[list[str]] = []
    response_body = b""
    error: str | None = None
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        connection.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
        response = connection.getresponse()
        status = int(response.status)
        headers = [[key, value] for key, value in response.getheaders()]
        chunks: list[bytes] = []
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        response_body = b"".join(chunks)
        connection.close()
    except (OSError, http.client.HTTPException) as exc:
        error = str(exc)
    finished_mono = time.monotonic_ns()
    body_path = raw_dir / f"response_{plan_item['sequence']:06d}.body"
    body_path.write_bytes(response_body)
    parsed: Any = None
    try:
        parsed = json.loads(response_body.decode("utf-8")) if response_body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    return {
        "sequence": plan_item["sequence"],
        "request_id": plan_item["request_id"],
        "repeat_index": plan_item["repeat_index"],
        "measurement": plan_item["measurement"],
        "measurement_phase": plan_item["measurement_phase"],
        "n_predict": plan_item["n_predict"],
        "stream": plan_item["stream"],
        "started_mono_ns": started_mono,
        "started_realtime_ns": started_realtime,
        "first_byte_mono_ns": first_byte_mono,
        "finished_mono_ns": finished_mono,
        "http_status": status,
        "headers": headers,
        "body_path": str(body_path.relative_to(raw_dir.parent)),
        "body_bytes": len(response_body),
        "body_sha256": sha256_bytes(response_body),
        "response_json": parsed,
        "error": error,
    }


def pressure_basis_source(basis: dict[str, Any]) -> str:
    return "CGROUP_RATIO" if basis["authority"] == "cgroup_finite" else "RSS_ABSOLUTE"


def parse_marker_fields(line: str, token: str) -> dict[str, str] | None:
    words = line.split()
    if words.count(token) != 1:
        return None
    fields: dict[str, str] = {}
    for word in words[words.index(token) + 1:]:
        if word.count("=") != 1:
            return None
        key, value = word.split("=", 1)
        if not key or not value or key in fields:
            return None
        fields[key] = value
    return fields


def uint_marker_field(fields: dict[str, str], key: str) -> int | None:
    value = fields.get(key)
    if value is None or not re.fullmatch(r"[0-9]+", value):
        return None
    return int(value)


def qualifying_offload_action(
        fields: dict[str, str], expected_source: str) -> tuple[int, int, int] | None:
    decision_id = uint_marker_field(fields, "decision_id")
    transaction_id = uint_marker_field(fields, "transaction_id")
    seq_id = uint_marker_field(fields, "selected_seq_id")
    claimant_epoch = uint_marker_field(fields, "selected_claimant_epoch")
    budget_excess = uint_marker_field(fields, "budget_observed_excess_bytes")
    blocks = uint_marker_field(fields, "blocks")
    migrated_bytes = uint_marker_field(fields, "bytes")
    relieved_bytes = uint_marker_field(fields, "relieved_bytes")
    shortfall_bytes = uint_marker_field(fields, "shortfall_bytes")
    if None in (
            decision_id, transaction_id, seq_id, claimant_epoch, budget_excess, blocks,
            migrated_bytes, relieved_bytes, shortfall_bytes):
        return None
    if (
        fields.get("state") != "NORMAL"
        or fields.get("source") != expected_source
        or fields.get("sample_valid") != "1"
        or fields.get("stale") != "0"
        or fields.get("pressure_basis_valid") != "1"
        or fields.get("budget_active") != "1"
        or claimant_epoch <= 0
        or budget_excess <= 0
        or fields.get("offload_attempted") != "1"
        or fields.get("outcome") != "completed"
        or fields.get("state_changed") != "1"
        or blocks <= 0
        or migrated_bytes <= 0
        or relieved_bytes <= 0
        or shortfall_bytes != 0
        or fields.get("io_failure") != "0"
    ):
        return None
    return decision_id, transaction_id, seq_id


def qualifying_resident_drop(
        fields: dict[str, str]) -> tuple[tuple[int, int, int], dict[str, int]] | None:
    names = (
        "decision_id", "transaction_id", "seq_id", "before_object_id", "before_generation",
        "before_resident_bytes", "after_object_id", "after_generation", "after_resident_bytes",
    )
    values = {name: uint_marker_field(fields, name) for name in names}
    if any(value is None for value in values.values()):
        return None
    if (
        fields.get("source") != "paged_sample_mincore"
        or fields.get("action") != "offload"
        or fields.get("before_available") != "1"
        or fields.get("after_available") != "1"
        or values["before_object_id"] != values["after_object_id"]
        or values["before_generation"] != values["after_generation"]
        or values["after_resident_bytes"] >= values["before_resident_bytes"]
    ):
        return None
    key = (values["decision_id"], values["transaction_id"], values["seq_id"])
    return key, {name: int(value) for name, value in values.items()}


def find_offload_barrier_pair(text: str, expected_source: str) -> dict[str, int] | None:
    actions: dict[tuple[int, int, int], dict[str, str]] = {}
    observations: dict[tuple[int, int, int], dict[str, int]] = {}
    for line in text.splitlines():
        action = parse_marker_fields(line, "kv_pressure_unified_action")
        if action is not None:
            key = qualifying_offload_action(action, expected_source)
            if key is not None:
                actions[key] = action
        observation = parse_marker_fields(line, "kv_g0_s1_resident_observation")
        if observation is not None:
            qualified = qualifying_resident_drop(observation)
            if qualified is not None:
                key, values = qualified
                observations[key] = values
    for key in actions:
        observation = observations.get(key)
        if observation is None:
            continue
        decision_id, transaction_id, seq_id = key
        selected_claimant_epoch = uint_marker_field(
            actions[key], "selected_claimant_epoch")
        if selected_claimant_epoch is None or selected_claimant_epoch <= 0:
            continue
        return {
            "decision_id": decision_id,
            "transaction_id": transaction_id,
            "seq_id": seq_id,
            "selected_claimant_epoch": selected_claimant_epoch,
            "before_object_id": observation["before_object_id"],
            "before_generation": observation["before_generation"],
            "before_resident_bytes": observation["before_resident_bytes"],
            "after_object_id": observation["after_object_id"],
            "after_generation": observation["after_generation"],
            "after_resident_bytes": observation["after_resident_bytes"],
            "resident_drop_bytes": (
                observation["before_resident_bytes"] - observation["after_resident_bytes"]),
        }
    return None


def complete_stderr_suffix(path: pathlib.Path, start_offset: int) -> tuple[str, int]:
    data = path.read_bytes()
    if start_offset < 0 or start_offset > len(data):
        raise RunnerError("server stderr offset moved outside the captured file")
    suffix = data[start_offset:]
    effective_start = start_offset
    if start_offset > 0 and data[start_offset - 1:start_offset] != b"\n":
        newline = suffix.find(b"\n")
        if newline < 0:
            return "", start_offset
        effective_start += newline + 1
        suffix = suffix[newline + 1:]
    newline = suffix.rfind(b"\n")
    if newline < 0:
        return "", effective_start
    complete = suffix[:newline + 1]
    return complete.decode("utf-8", errors="replace"), effective_start + len(complete)


def wait_real_offload_barrier(
        stderr_path: pathlib.Path,
        start_offset: int,
        timeout_seconds: float,
        expected_source: str,
        server: subprocess.Popen[bytes],
) -> dict[str, Any]:
    started_mono_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    last_complete_offset = start_offset
    while True:
        text, complete_offset = complete_stderr_suffix(stderr_path, start_offset)
        last_complete_offset = max(last_complete_offset, complete_offset)
        pair = find_offload_barrier_pair(text, expected_source)
        if pair is not None:
            completed_mono_ns = time.monotonic_ns()
            return {
                "status": "passed",
                "started_mono_ns": started_mono_ns,
                "completed_mono_ns": completed_mono_ns,
                "duration_ns": completed_mono_ns - started_mono_ns,
                "stderr_start_offset": start_offset,
                "stderr_end_offset": complete_offset,
                **pair,
            }
        if server.poll() is not None:
            raise RunnerError("server exited while waiting for the real OFFLOAD barrier")
        if time.monotonic() >= deadline:
            completed_mono_ns = time.monotonic_ns()
            return {
                "status": "timeout",
                "started_mono_ns": started_mono_ns,
                "completed_mono_ns": completed_mono_ns,
                "duration_ns": completed_mono_ns - started_mono_ns,
                "stderr_start_offset": start_offset,
                "stderr_end_offset": last_complete_offset,
                "decision_id": None,
                "transaction_id": None,
                "seq_id": None,
                "before_object_id": None,
                "before_generation": None,
                "before_resident_bytes": None,
                "after_object_id": None,
                "after_generation": None,
                "after_resident_bytes": None,
                "resident_drop_bytes": None,
            }
        time.sleep(0.02)


def budget_settle_observations(
        text: str,
        expected_source: str,
        resident_target_bytes: int,
        action_target_bytes: int,
        max_blocks: int,
        tolerance_bytes: int,
        release_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    observations: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    seen_decisions: set[int] = set()
    numeric_names = (
        "decision_id", "target_bytes", "max_blocks", "budget_target_bytes", "budget_resident_bytes",
        "budget_observed_excess_bytes", "budget_debt_after_bytes",
        "unmet_budget_bytes_after", "budget_transient_staging_bound_bytes",
        "blocks", "bytes", "relieved_bytes",
    )
    for line in text.splitlines():
        fields = parse_marker_fields(line, "kv_pressure_unified_action")
        if fields is None:
            continue
        values = {name: uint_marker_field(fields, name) for name in numeric_names}
        if any(value is None for value in values.values()):
            continue
        decision_id = int(values["decision_id"])
        if decision_id in seen_decisions:
            continue
        if (
            fields.get("state") != "NORMAL"
            or fields.get("source") != expected_source
            or fields.get("sample_valid") != "1"
            or fields.get("stale") != "0"
            or fields.get("pressure_basis_valid") != "1"
            or fields.get("budget_active") != "1"
            or fields.get("budget_target_enabled") != "1"
            or fields.get("budget_source") != "env_static"
            or int(values["target_bytes"]) != action_target_bytes
            or int(values["max_blocks"]) != max_blocks
            or int(values["budget_target_bytes"]) != resident_target_bytes
            or fields.get("budget_view_valid") != "1"
            or fields.get("budget_resident_available") != "1"
            or fields.get("idle") != "1"
        ):
            continue
        resident_bytes = int(values["budget_resident_bytes"])
        observed_excess = max(0, resident_bytes - resident_target_bytes)
        if int(values["budget_observed_excess_bytes"]) != observed_excess:
            continue
        positive_offload = (
            fields.get("offload_attempted") == "1"
            and fields.get("outcome") == "completed"
            and fields.get("state_changed") == "1"
            and fields.get("io_failure") == "0"
            and int(values["blocks"]) > 0
            and int(values["bytes"]) > 0
            and int(values["relieved_bytes"]) > 0
        )
        positive_release = (
            fields.get("release_attempted") == "1"
            and fields.get("outcome") == "completed"
            and fields.get("state_changed") == "1"
            and fields.get("io_failure") == "0"
            and int(values["blocks"]) > 0
            and int(values["relieved_bytes"]) > 0
        )
        release_no_candidate = (
            release_only
            and fields.get("release_attempted") == "1"
            and fields.get("offload_attempted") == "0"
            and fields.get("outcome") == "no_op"
            and fields.get("reason") == "no_candidate"
            and fields.get("state_changed") == "0"
            and fields.get("io_failure") == "0"
            and int(values["blocks"]) == 0
            and int(values["bytes"]) == 0
            and int(values["relieved_bytes"]) == 0
            and fields.get("transaction_id") == "0"
            and int(values["budget_debt_after_bytes"]) > 0
            and int(values["unmet_budget_bytes_after"]) == 0
            and fields.get("soft_offload_armed_after") == "1"
        )
        summary = {
            "decision_id": decision_id,
            "decision_reason": fields.get("decision_reason"),
            "action_target_bytes": action_target_bytes,
            "max_blocks": max_blocks,
            "budget_target_bytes": resident_target_bytes,
            "budget_resident_bytes": resident_bytes,
            "budget_observed_excess_bytes": observed_excess,
            "budget_debt_after_bytes": int(values["budget_debt_after_bytes"]),
            "unmet_budget_bytes_after": int(values["unmet_budget_bytes_after"]),
            "budget_transient_staging_bound_bytes": int(
                values["budget_transient_staging_bound_bytes"]),
            "soft_offload_armed_before": fields.get("soft_offload_armed_before"),
            "soft_offload_armed_after": fields.get("soft_offload_armed_after"),
            "positive_offload": positive_offload,
            "positive_release": positive_release,
        }
        observations.append(summary)
        seen_decisions.add(decision_id)
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
            fields.get("decision_reason") == "budget_target_satisfied"
            and resident_bytes <= resident_target_bytes + tolerance_bytes
            and observed_excess == 0
            and summary["budget_debt_after_bytes"] == 0
            and summary["unmet_budget_bytes_after"] == 0
        ):
            terminal = {"status": "target_reached", **summary}
            break
        if (
            fields.get("decision_reason") == "budget_unmet_terminal"
            and observed_excess > 0
            and summary["budget_debt_after_bytes"] == observed_excess
            and summary["unmet_budget_bytes_after"] == observed_excess
        ):
            terminal = {"status": "unmet_floor", **summary}
            break
    return observations, terminal


def release_settle_boundary(
        text: str,
        expected_source: str,
        resident_target_bytes: int,
        action_target_bytes: int,
        max_blocks: int,
) -> dict[str, Any] | None:
    for line in text.splitlines():
        fields = parse_marker_fields(line, "kv_pressure_unified_action")
        if fields is None:
            continue
        values = {
            key: uint_marker_field(fields, key)
            for key in (
                "decision_id", "target_bytes", "max_blocks", "budget_target_bytes",
                "budget_resident_bytes", "budget_debt_after_bytes",
                "unmet_budget_bytes_after", "blocks", "bytes", "relieved_bytes",
            )
        }
        if any(value is None for value in values.values()):
            continue
        if (
            fields.get("state") != "NORMAL"
            or fields.get("source") != expected_source
            or fields.get("sample_valid") != "1"
            or fields.get("stale") != "0"
            or fields.get("pressure_basis_valid") != "1"
            or fields.get("budget_active") != "1"
            or fields.get("budget_target_enabled") != "1"
            or fields.get("budget_source") != "env_static"
            or int(values["target_bytes"]) != action_target_bytes
            or int(values["max_blocks"]) != max_blocks
            or int(values["budget_target_bytes"]) != resident_target_bytes
            or fields.get("budget_view_valid") != "1"
            or fields.get("budget_resident_available") != "1"
            or fields.get("idle") != "1"
            or fields.get("release_attempted") != "1"
            or fields.get("offload_attempted") != "0"
        ):
            continue
        no_candidate = (
            fields.get("outcome") == "no_op"
            and fields.get("reason") == "no_candidate"
            and fields.get("state_changed") == "0"
            and fields.get("io_failure") == "0"
            and int(values["blocks"]) == 0
            and int(values["bytes"]) == 0
            and int(values["relieved_bytes"]) == 0
            and fields.get("transaction_id") == "0"
            and int(values["budget_debt_after_bytes"]) > 0
            and int(values["unmet_budget_bytes_after"]) == 0
            and fields.get("soft_offload_armed_after") == "1"
        )
        target_closed = (
            fields.get("outcome") == "completed"
            and fields.get("state_changed") == "1"
            and fields.get("io_failure") == "0"
            and int(values["blocks"]) > 0
            and int(values["relieved_bytes"]) > 0
            and int(values["budget_debt_after_bytes"]) == 0
            and int(values["unmet_budget_bytes_after"]) == 0
            and fields.get("soft_offload_armed_after") == "0"
        )
        if no_candidate or target_closed:
            return {
                "status": "release_no_candidate" if no_candidate else "release_settled",
                "decision_id": int(values["decision_id"]),
                "budget_resident_bytes": int(values["budget_resident_bytes"]),
                "budget_debt_after_bytes": int(values["budget_debt_after_bytes"]),
                "unmet_budget_bytes_after": int(values["unmet_budget_bytes_after"]),
                "state_changed": fields.get("state_changed") == "1",
                "bytes": int(values["bytes"]),
                "relieved_bytes": int(values["relieved_bytes"]),
                "soft_offload_armed_after": fields.get("soft_offload_armed_after") == "1",
            }
    return None


def wait_budget_settle(
        stderr_path: pathlib.Path,
        start_offset: int,
        timeout_seconds: float,
        expected_source: str,
        resident_target_bytes: int,
        action_target_bytes: int,
        max_blocks: int,
        tolerance_bytes: int,
        server: subprocess.Popen[bytes],
        *,
        release_only: bool = False,
        capture_release_settle: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started_mono_ns = time.monotonic_ns()
    deadline = time.monotonic() + timeout_seconds
    last_complete_offset = start_offset
    release_settle: dict[str, Any] | None = None
    release_boundary: dict[str, Any] | None = None
    while True:
        text, complete_offset = complete_stderr_suffix(stderr_path, start_offset)
        last_complete_offset = max(last_complete_offset, complete_offset)
        observations, terminal = budget_settle_observations(
            text,
            expected_source,
            resident_target_bytes,
            action_target_bytes,
            max_blocks,
            tolerance_bytes,
            release_only=release_only,
        )
        boundary = release_settle_boundary(
            text,
            expected_source,
            resident_target_bytes,
            action_target_bytes,
            max_blocks,
        )
        if (
            capture_release_settle is not None
            and release_settle is None
            and boundary is not None
        ):
            release_settle = capture_release_settle()
            release_boundary = boundary
        if terminal is not None:
            completed_mono_ns = time.monotonic_ns()
            return {
                "status": terminal.pop("status"),
                "started_mono_ns": started_mono_ns,
                "completed_mono_ns": completed_mono_ns,
                "duration_ns": completed_mono_ns - started_mono_ns,
                "stderr_start_offset": start_offset,
                "stderr_end_offset": complete_offset,
                "terminal_decision": terminal,
                "physical_resident_bytes": None,
                "release_settle": release_settle,
                "release_boundary": release_boundary,
                "decision_ids": [item["decision_id"] for item in observations],
                "offload_decision_ids": [
                    item["decision_id"] for item in observations if item["positive_offload"]],
            }
        if server.poll() is not None:
            raise RunnerError("server exited while waiting for the budget settle marker")
        if time.monotonic() >= deadline:
            completed_mono_ns = time.monotonic_ns()
            return {
                "status": "timeout",
                "started_mono_ns": started_mono_ns,
                "completed_mono_ns": completed_mono_ns,
                "duration_ns": completed_mono_ns - started_mono_ns,
                "stderr_start_offset": start_offset,
                "stderr_end_offset": last_complete_offset,
                "terminal_decision": None,
                "physical_resident_bytes": None,
                "release_settle": release_settle,
                "release_boundary": release_boundary,
                "decision_ids": [item["decision_id"] for item in observations],
                "offload_decision_ids": [
                    item["decision_id"] for item in observations if item["positive_offload"]],
            }
        time.sleep(0.02)


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_process(
        process: subprocess.Popen[bytes] | None,
        name: str,
        timeout: float) -> dict[str, Any]:
    if process is None:
        return {
            "pid": None, "pgid": None, "exit_code": None, "stop_requested": False,
            "stop_signal": None, "term_timed_out": False, "kill_timed_out": False,
            "pgid_check_complete": False, "residual_process": False,
        }
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        pgid = process.pid
    stop_requested = process.poll() is None
    stop_signal: str | None = "SIGTERM" if stop_requested else None
    term_timed_out = False
    kill_timed_out = False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        term_timed_out = True
    if process_group_alive(pgid):
        stop_signal = "SIGKILL"
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + timeout
        while process_group_alive(pgid) and time.monotonic() < deadline:
            time.sleep(0.05)
        kill_timed_out = process_group_alive(pgid)
    residual = process_group_alive(pgid)
    return {
        "pid": process.pid,
        "pgid": pgid,
        "exit_code": process.returncode if not residual else None,
        "stop_requested": stop_requested,
        "stop_signal": stop_signal,
        "term_timed_out": term_timed_out,
        "kill_timed_out": kill_timed_out,
        "pgid_check_complete": True,
        "residual_process": residual,
    }


def request_completion_replay(
        port: int,
        event: dict[str, Any],
        slot_id: int,
        raw_dir: pathlib.Path,
        timeout: float,
        cache_prompt: bool,
        started_at_ns: int,
) -> dict[str, Any]:
    if isinstance(slot_id, bool) or not isinstance(slot_id, int) or slot_id < 0:
        raise RunnerError("replay id_slot must be a non-negative integer")
    if not isinstance(cache_prompt, bool):
        raise RunnerError("replay cache_prompt must be boolean")
    body = {
        "prompt": event["prompt_tokens"],
        "n_predict": event["n_predict"],
        "seed": 0,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
        "ignore_eos": True,
        "stream": False,
        "id_slot": slot_id,
    }
    encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    request_sha = sha256_bytes(encoded)
    started_mono = time.monotonic_ns()
    status = 0
    response_body = b""
    headers: list[list[str]] = []
    error: str | None = None
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        connection.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
        response = connection.getresponse()
        status = int(response.status)
        headers = [[key, value] for key, value in response.getheaders()]
        chunks: list[bytes] = []
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        response_body = b"".join(chunks)
        connection.close()
    except (OSError, http.client.HTTPException) as exc:
        error = str(exc)
    finished_mono = time.monotonic_ns()
    body_path = raw_dir / f"replay_response_{event['event_seq']:06d}.body"
    body_path.write_bytes(response_body)
    parsed: Any = None
    try:
        parsed = json.loads(response_body.decode("utf-8")) if response_body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    token_sha: str | None = None
    tokens_evaluated: int | None = None
    tokens_predicted: int | None = None
    response_slot: int | None = None
    if isinstance(parsed, dict):
        response_slot = parsed.get("id_slot")
        tokens_evaluated = parsed.get("tokens_evaluated")
        tokens_predicted = parsed.get("tokens_predicted")
        tokens = parsed.get("tokens")
        if isinstance(tokens, list) and all(isinstance(t, int) and not isinstance(t, bool) and t >= 0 for t in tokens):
            token_sha = sha256_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))
    return {
        **event,
        "slot_id": slot_id,
        "seq_id": slot_id,
        "cache_prompt": cache_prompt,
        "prompt_token_count": len(event["prompt_tokens"]),
        "claimant_epoch": "UNAVAILABLE",
        "physical_object_id": "UNAVAILABLE",
        "physical_generation": "UNAVAILABLE",
        "request_sha256": request_sha,
        "started_mono_ns": started_mono,
        "started_us": (started_mono - started_at_ns) // 1000,
        "finished_mono_ns": finished_mono,
        "completed_us": (finished_mono - started_at_ns) // 1000,
        "http_status": status,
        "headers": headers,
        "body_path": str(body_path.relative_to(raw_dir.parent)),
        "body_bytes": len(response_body),
        "body_sha256": sha256_bytes(response_body),
        "response_json": parsed,
        "response_slot_id": response_slot,
        "tokens_evaluated": tokens_evaluated,
        "tokens_predicted": tokens_predicted,
        "response_token_sha256": token_sha,
        "error": error,
    }


def validate_replay_model_binding(replay_plan: Any, spec: dict[str, Any]) -> None:
    """Bind a model-bound transcript to the replay model, not its materializer binary."""
    if replay_plan.schema != "gt-trace-1b-a/v1":
        return
    try:
        document = json.loads(pathlib.Path(replay_plan.source_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot reload model-bound replay transcript: {exc}") from exc
    model = document.get("model")
    if not isinstance(model, dict):
        raise RunnerError("model-bound replay transcript has no model identity")
    expected_model_sha = model.get("model_sha256")
    transcript_binary_sha = model.get("binary_sha256")
    if not isinstance(expected_model_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_model_sha):
        raise RunnerError("model-bound replay transcript model SHA is invalid")
    if not isinstance(transcript_binary_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", transcript_binary_sha):
        raise RunnerError("model-bound replay transcript binary SHA is invalid")
    if sha256_file(pathlib.Path(spec["model"])) != expected_model_sha:
        raise RunnerError("replay model bytes differ from the model-bound transcript")


def validate_replay_slot_capacity(
        snapshot: dict[str, Any], replay_plan: Any, n_parallel: int) -> None:
    """Require every replay slot to fit the largest selected request."""
    slots = snapshot.get("body_json")
    if not isinstance(slots, list):
        raise RunnerError("replay /slots preflight body is unavailable")
    by_id: dict[int, dict[str, Any]] = {}
    for slot in slots:
        if not isinstance(slot, dict):
            continue
        slot_id = slot.get("id")
        if isinstance(slot_id, bool) or not isinstance(slot_id, int):
            continue
        by_id[slot_id] = slot
    required_ids = set(range(n_parallel))
    if not required_ids.issubset(by_id):
        raise RunnerError(
            f"replay /slots does not expose all requested slots: expected={sorted(required_ids)} "
            f"actual={sorted(by_id)}")
    max_required = max(
        len(event.prompt_tokens) + event.n_predict
        for event in replay_plan.events
    )
    for slot_id in sorted(required_ids):
        n_ctx = by_id[slot_id].get("n_ctx")
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx <= 0:
            raise RunnerError(f"replay slot {slot_id} n_ctx is invalid")
        if n_ctx < max_required:
            raise RunnerError(
                f"replay slot {slot_id} context too small: n_ctx={n_ctx} "
                f"required={max_required}")


def validate_replay_erased_slot_snapshot(snapshot: dict[str, Any], slot_id: int) -> dict[str, Any]:
    if snapshot.get("error") is not None or snapshot.get("http_status") != 200:
        raise RunnerError("post-erase /slots verification failed")
    slots = snapshot.get("body_json")
    if not isinstance(slots, list):
        raise RunnerError("post-erase /slots body is unavailable")
    matches = [slot for slot in slots if isinstance(slot, dict) and slot.get("id") == slot_id]
    if len(matches) != 1:
        raise RunnerError("post-erase /slots slot identity is missing or duplicated")
    slot = matches[0]
    if slot.get("is_processing") is not False:
        raise RunnerError("post-erase slot is still processing")
    n_prompt_tokens = slot.get("n_prompt_tokens")
    if isinstance(n_prompt_tokens, bool) or not isinstance(n_prompt_tokens, int) or n_prompt_tokens != 0:
        raise RunnerError("post-erase slot prompt was not cleared")
    claimant = slot.get("kv_claimant")
    required = {"target_blocks", "eligible_resident_blocks", "swapped_blocks", "shared_blocks", "blocked_blocks"}
    if not isinstance(claimant, dict) or not required.issubset(claimant):
        raise RunnerError("post-erase KV claimant evidence is missing")
    counts: dict[str, int] = {}
    for key in sorted(required):
        value = claimant.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise RunnerError(f"post-erase KV claimant {key} was not cleared")
        counts[key] = value
    return {"slot_id": slot_id, "n_prompt_tokens": n_prompt_tokens, "claimant_counts": counts}


def validate_replay_erase_response(
        status: int, body: bytes, error: str | None, slot_id: int) -> dict[str, Any]:
    parsed: Any = None
    try:
        parsed = json.loads(body.decode("utf-8")) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    if error is not None or status != 200 or not isinstance(parsed, dict):
        raise RunnerError(
            f"replay erase failed slot={slot_id} status={status} error={error or 'invalid JSON'}")
    if set(parsed) != {"id_slot", "n_erased"} or parsed.get("id_slot") != slot_id:
        raise RunnerError("replay erase response identity/schema mismatch")
    n_erased = parsed.get("n_erased")
    if isinstance(n_erased, bool) or not isinstance(n_erased, int) or n_erased <= 0:
        raise RunnerError("replay erase n_erased must prove a non-empty prompt was cleared")
    return {"id_slot": parsed["id_slot"], "n_erased": n_erased, "body": parsed}


def replay_arrival_precedes_expiry(arrival_us: int, expiry_us: int) -> bool:
    return arrival_us < expiry_us


def erase_replay_slot(port: int, slot_id: int, timeout: float) -> dict[str, Any]:
    if isinstance(slot_id, bool) or not isinstance(slot_id, int) or slot_id < 0:
        raise RunnerError("replay erase slot is invalid")
    body = b""
    status = 0
    error: str | None = None
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        connection.request("POST", f"/slots/{slot_id}?action=erase", body, {"Content-Length": "0"})
        response = connection.getresponse()
        status = int(response.status)
        body = response.read()
        connection.close()
    except (OSError, http.client.HTTPException) as exc:
        error = str(exc)
    validated = validate_replay_erase_response(status, body, error, slot_id)
    return {"http_status": status, "body_sha256": sha256_bytes(body), **validated}


def lifecycle_event(
        event_type: str, *, session: dict[str, Any], turn: int, binding: dict[str, Any] | None,
        runner_generation: int | None, lifecycle_generation: int, arrival_us: int | None = None,
        completion_us: int | None = None, expiry_us: int | None = None, request_id: str | None = None,
        trigger: str | None = None, **extra: Any) -> dict[str, Any]:
    result = {
        "event": event_type,
        "logical_session_id": session["logical_session_id"],
        "lineage_id": session["lineage_id"],
        "turn": turn,
        "slot_id": binding.get("slot_id") if binding else None,
        "seq_id": binding.get("seq_id") if binding else None,
        "runner_generation": runner_generation,
        "lifecycle_generation": lifecycle_generation,
        "arrival_us": arrival_us,
        "completion_us": completion_us,
        "expiry_us": expiry_us,
        "request_id": request_id,
        "trigger": trigger,
    }
    result.update(extra)
    return result

def run_one_replay_lifecycle(
        artifact: pathlib.Path,
        run: dict[str, Any],
        case: dict[str, Any],
        spec: dict[str, Any],
        workload: dict[str, Any],
        execution_index: int,
        replay_plan: Any,
        schedule: list[dict[str, Any]],
) -> dict[str, Any]:
    run_dir = artifact / "runs" / run["run_id"]
    raw_dir = run_dir / "raw"
    backing_dir = run_dir / "backing"
    slot_save_dir = run_dir / "slot-cache"
    raw_dir.mkdir(parents=True)
    backing_dir.mkdir()
    slot_save_dir.mkdir()
    replay_cfg = workload["replay"]
    lifecycle_cfg = replay_cfg["lifecycle"]
    ttl_runtime_us = int(round(float(replay_plan.lifecycle["ttl_seconds"]) * replay_cfg["time_dilation"] * 1_000_000))
    if ttl_runtime_us <= 0:
        raise RunnerError("lifecycle runtime TTL is not positive")
    dump(run_dir / "run.json", {
        "run_id": run["run_id"], "round": run["round"], "run_order": run["run_order"],
        "case_id": run["case_id"], "execution_index": execution_index, "case": run,
        "request_plan": schedule, "replay": replay_plan.to_dict(),
    })
    port = free_port()
    env = runtime_environment(spec, case, backing_dir)
    argv = [*server_argv(spec, port), "--slot-save-path", str(slot_save_dir)]
    stdout_path, stderr_path = run_dir / "server.stdout", run_dir / "server.stderr"
    sampler_stdout_path, sampler_stderr_path = run_dir / "sampler.stdout", run_dir / "sampler.stderr"
    samples_path, responses_path = run_dir / "memory_samples.tsv", run_dir / "responses.jsonl"
    lifecycle_path = run_dir / "lifecycle.jsonl"
    server: subprocess.Popen[bytes] | None = None
    sampler: subprocess.Popen[bytes] | None = None
    server_identity_record: dict[str, Any] | None = None
    sampler_identity_record: dict[str, Any] | None = None
    server_cgroup: dict[str, Any] | None = None
    runner_cgroup = cgroup_identity(os.getpid())
    request_records: list[dict[str, Any]] = []
    admission_records: list[dict[str, Any]] = []
    lifecycle_records: list[dict[str, Any]] = []
    lifecycle_error: str | None = None
    server_cleanup = {"pid": None, "pgid": None, "exit_code": None, "stop_requested": False,
                       "stop_signal": None, "term_timed_out": False, "kill_timed_out": False,
                       "pgid_check_complete": False, "residual_process": False}
    sampler_cleanup = dict(server_cleanup)
    handles: list[Any] = []
    started_at_ns = time.monotonic_ns()
    request_loop_started = False
    admission = __import__("multi_session_replay", fromlist=["SlotAdmission"]).SlotAdmission(replay_cfg["n_parallel"])
    sessions = {
        session.logical_session_id: {
            "logical_session_id": session.logical_session_id,
            "lineage_id": session.lineage_id,
            "state": "ABSENT", "lifecycle_generation": 0, "next_turn": 1,
            "binding": None, "timer": None,
        }
        for session in replay_plan.sessions
    }

    def clock_us() -> int:
        return (time.monotonic_ns() - started_at_ns) // 1000

    def emit(record: dict[str, Any]) -> None:
        lifecycle_records.append(record)

    def timer_due(now_us: int, *, force: bool = False) -> bool:
        for session in sessions.values():
            timer = session["timer"]
            if timer is not None and (force or timer["expiry_us"] <= now_us):
                return True
        return False

    def expire_due(now_us: int, *, force: bool = False) -> bool:
        for session in sorted(sessions.values(), key=lambda item: (item["timer"]["expiry_us"], item["logical_session_id"]) if item["timer"] else (10**30, item["logical_session_id"])):
            timer = session["timer"]
            if timer is None or (not force and timer["expiry_us"] > now_us):
                continue
            if session["state"] != "IDLE_TTL" or session["binding"] is None or session["lifecycle_generation"] != timer["lifecycle_generation"]:
                session["timer"] = None
                continue
            binding = dict(session["binding"])
            emit(lifecycle_event("TTL_EXPIRY", session=session, turn=timer["turn"], binding=binding,
                                 runner_generation=binding["runner_generation"],
                                 lifecycle_generation=session["lifecycle_generation"],
                                 completion_us=timer["armed_at_us"], expiry_us=timer["expiry_us"],
                                 trigger="ttl_timer", timer_id=timer["timer_id"]))
            erase = erase_replay_slot(port, binding["slot_id"], spec["request_timeout_seconds"])
            verify_name = f"slots_after_erase_{timer['timer_id']:06d}.json"
            verify_snapshot = capture_slots(port, run_dir / verify_name, spec["request_timeout_seconds"])
            validate_replay_erased_slot_snapshot(verify_snapshot, binding["slot_id"])
            released = admission.release(session["logical_session_id"], now_us)
            session["binding"] = None
            session["timer"] = None
            session["state"] = "DEAD"
            emit(lifecycle_event("DEAD", session=session, turn=timer["turn"], binding=binding,
                                 runner_generation=binding["runner_generation"],
                                 lifecycle_generation=session["lifecycle_generation"],
                                 completion_us=timer["armed_at_us"], expiry_us=timer["expiry_us"],
                                 trigger="ttl_timer", timer_id=timer["timer_id"],
                                 erase_http_status=erase["http_status"], erase_body_sha256=erase["body_sha256"],
                                 erase_id_slot=erase["id_slot"], erase_n_erased=erase["n_erased"],
                                 erase_verify_path=verify_name, erase_verify_body_sha256=verify_snapshot["body_sha256"],
                                 released_us=now_us, release_status=released["status"]))
            return True
        return False

    try:
        server_handle = stdout_path.open("wb"); handles.append(server_handle)
        server_err_handle = stderr_path.open("wb"); handles.append(server_err_handle)
        server = subprocess.Popen(argv, cwd=run_dir, env=env, stdout=server_handle,
                                  stderr=server_err_handle, start_new_session=True)
        server_identity_record = process_identity(server.pid, argv)
        server_cgroup = cgroup_scope(cgroup_identity(server.pid), runner_cgroup)
        expected_memory_max = spec["cgroup"]["expected_memory_max"]
        if expected_memory_max is not None and server_cgroup.get("memory_max") != expected_memory_max:
            raise RunnerError("server cgroup memory.max mismatch")
        if spec["pressure_basis"]["authority"] == "cgroup_finite" and not finite_memory_limit(server_cgroup.get("memory_max")):
            raise RunnerError("server cgroup does not provide a finite pressure authority")
        current_file = server_cgroup.get("memory_current_file") or ""
        sampler_env = dict(os.environ)
        sampler_env["KV_CONTROLLED_SAMPLE_SCHEMA"] = SAMPLE_SCHEMA
        sampler_env["KV_CONTROLLED_CGROUP_DIR"] = server_cgroup.get("path") or ""
        sampler_command = ["bash", str(MEMORY_SAMPLER), "--sample-process", str(server.pid),
                           str(samples_path), str(backing_dir), str(spec["sampler"]["interval_seconds"]), current_file]
        sampler_stdout_handle = sampler_stdout_path.open("wb"); handles.append(sampler_stdout_handle)
        sampler_stderr_handle = sampler_stderr_path.open("wb"); handles.append(sampler_stderr_handle)
        sampler = subprocess.Popen(sampler_command, cwd=run_dir, env=sampler_env,
                                   stdout=sampler_stdout_handle, stderr=sampler_stderr_handle,
                                   start_new_session=True)
        sampler_identity_record = process_identity(sampler.pid, sampler_command)
        wait_health(port, server, spec["health_timeout_seconds"])
        slots_before = capture_slots(port, run_dir / "slots_before.json", spec["request_timeout_seconds"])
        validate_replay_slot_capacity(slots_before, replay_plan, replay_cfg["n_parallel"])
        started_at_ns = time.monotonic_ns()
        request_loop_started = True
        pending = list(schedule)
        inflight: dict[Any, dict[str, Any]] = {}
        inflight_sessions: set[str] = set()
        queued_recorded: set[str] = set()
        pre_registered_revisits: dict[int, dict[str, Any]] = {}
        dispatch_count = 0
        blocked_since: float | None = None
        timer_sequence = 0
        with ThreadPoolExecutor(max_workers=replay_cfg["n_parallel"]) as executor:
            while pending or inflight or (lifecycle_cfg["drain_after_last_arrival"] and any(item["binding"] is not None for item in sessions.values())):
                progressed = False
                done = [future for future in inflight if future.done()]
                for future in sorted(done, key=lambda item: inflight[item]["dispatch_order"]):
                    meta = inflight.pop(future)
                    first = meta["event"]
                    sid = first["logical_session_id"]
                    session = sessions[sid]
                    try:
                        actual = future.result()
                    except Exception as exc:
                        raise RunnerError(f"replay request failed for {first['request_id']}: {exc}") from exc
                    actual.update({
                        "dispatch_order": meta["dispatch_order"],
                        "admitted_us": meta["admitted_us"],
                        "dispatched_us": actual["started_us"],
                        "planned_arrival_us": first["planned_arrival_us"],
                        "arrival_lag_us": max(0, actual["started_us"] - first["planned_arrival_us"]),
                        "admission_wait_us": max(0, meta["admitted_us"] - first["planned_arrival_us"]),
                        "service_us": actual["completed_us"] - actual["started_us"],
                        "runner_generation": meta["binding"]["runner_generation"],
                        "lifecycle_trigger": meta["trigger"],
                        "lifecycle_generation": session["lifecycle_generation"],
                    })
                    request_records.append(actual)
                    inflight_sessions.remove(sid)
                    session["next_turn"] += 1
                    emit(lifecycle_event("TURN_COMPLETE", session=session, turn=first["turn"], binding=meta["binding"],
                                         runner_generation=meta["binding"]["runner_generation"],
                                         lifecycle_generation=session["lifecycle_generation"],
                                         arrival_us=meta["arrival_us"], completion_us=actual["completed_us"],
                                         request_id=first["request_id"], trigger=meta["trigger"]))
                    continuation_arrived = any(
                        item["logical_session_id"] == sid and item["turn"] == session["next_turn"]
                        and item["planned_arrival_us"] <= actual["completed_us"] for item in pending)
                    if continuation_arrived:
                        session["state"] = "ACTIVE"
                    else:
                        timer_sequence += 1
                        expiry = actual["completed_us"] + ttl_runtime_us
                        timer = {"timer_id": timer_sequence, "armed_at_us": actual["completed_us"],
                                 "expiry_us": expiry, "turn": first["turn"],
                                 "runner_generation": meta["binding"]["runner_generation"],
                                 "lifecycle_generation": session["lifecycle_generation"]}
                        session["timer"] = timer
                        session["state"] = "IDLE_TTL"
                        emit(lifecycle_event("TTL_ARM", session=session, turn=first["turn"], binding=meta["binding"],
                                             runner_generation=meta["binding"]["runner_generation"],
                                             lifecycle_generation=session["lifecycle_generation"],
                                             completion_us=actual["completed_us"], expiry_us=expiry,
                                             request_id=first["request_id"], trigger="completion", timer_id=timer_sequence))
                    progressed = True

                now_us = clock_us()
                # Arrival wins only when the frozen runtime arrival is strictly before expiry.
                # Register the revisit before consulting wall-clock timer expiry so scheduler
                # jitter cannot turn a pre-expiry trace arrival into a false DEAD transition.
                for first in sorted(
                        (item for item in pending if item["planned_arrival_us"] <= now_us),
                        key=lambda item: (item["planned_arrival_us"], item["event_seq"])):
                    sid = first["logical_session_id"]
                    session = sessions[sid]
                    timer = session["timer"]
                    if (session["state"] != "IDLE_TTL" or timer is None
                            or first["turn"] != session["next_turn"]
                            or first["event_seq"] in pre_registered_revisits):
                        continue
                    if not replay_arrival_precedes_expiry(
                            first["planned_arrival_us"], timer["expiry_us"]):
                        continue
                    emit(lifecycle_event(
                        "REVISIT", session=session, turn=first["turn"], binding=session["binding"],
                        runner_generation=session["binding"]["runner_generation"],
                        lifecycle_generation=session["lifecycle_generation"],
                        arrival_us=first["planned_arrival_us"], expiry_us=timer["expiry_us"],
                        request_id=first["request_id"], trigger="arrival", timer_id=timer["timer_id"],
                        observed_us=now_us))
                    pre_registered_revisits[first["event_seq"]] = timer
                    session["timer"] = None
                    session["state"] = "ACTIVE"
                if expire_due(now_us):
                    progressed = True
                    continue
                ready = [item for item in pending if item["planned_arrival_us"] <= now_us]
                for first in ready:
                    if len(inflight) >= replay_cfg["n_parallel"]:
                        break
                    sid = first["logical_session_id"]
                    session = sessions[sid]
                    if sid in inflight_sessions:
                        continue
                    if first["turn"] != session["next_turn"]:
                        raise ReplayError(f"session {sid} turn order is not contiguous")
                    trigger = "CONTINUATION"
                    cache_prompt = True
                    needs_admission = False
                    registered_revisit = pre_registered_revisits.pop(first["event_seq"], None)
                    if registered_revisit is not None:
                        trigger = "REVISIT"
                    elif session["state"] == "ABSENT":
                        trigger = "COLD_START"
                        cache_prompt = False
                        needs_admission = True
                    elif session["state"] == "DEAD":
                        trigger = "COLD_RESTART"
                        cache_prompt = False
                        needs_admission = True
                    elif session["state"] == "IDLE_TTL":
                        timer = session["timer"]
                        if timer is None or not replay_arrival_precedes_expiry(
                                first["planned_arrival_us"], timer["expiry_us"]):
                            continue
                        session["timer"] = None
                        session["state"] = "ACTIVE"
                        trigger = "REVISIT"
                        cache_prompt = True
                        emit(lifecycle_event("REVISIT", session=session, turn=first["turn"], binding=session["binding"],
                                             runner_generation=session["binding"]["runner_generation"],
                                             lifecycle_generation=session["lifecycle_generation"],
                                             arrival_us=first["planned_arrival_us"], expiry_us=timer["expiry_us"],
                                             request_id=first["request_id"], trigger="arrival", timer_id=timer["timer_id"],
                                             observed_us=now_us))
                    elif session["state"] != "ACTIVE":
                        raise ReplayError(f"session {sid} lifecycle state is invalid")
                    if needs_admission:
                        decision = admission.admit(sid, now_us)
                        if decision["status"] != "admitted":
                            continue
                        session["binding"] = {"slot_id": decision["slot_id"], "seq_id": decision["seq_id"],
                                              "runner_generation": decision["runner_generation"]}
                        if trigger == "COLD_START":
                            session["lifecycle_generation"] = 1
                        elif trigger == "COLD_RESTART":
                            session["lifecycle_generation"] += 1
                        admission_records.append({**first, **decision, "observed_us": now_us})
                        queued_recorded.discard(sid)
                    binding = dict(session["binding"])
                    pending.remove(first)
                    inflight_sessions.add(sid)
                    dispatch_order = dispatch_count
                    dispatch_count += 1
                    if trigger == "COLD_RESTART":
                        emit(lifecycle_event("COLD_RESTART", session=session, turn=first["turn"], binding=binding,
                                             runner_generation=binding["runner_generation"],
                                             lifecycle_generation=session["lifecycle_generation"], arrival_us=first["planned_arrival_us"],
                                             request_id=first["request_id"], trigger="arrival", observed_us=now_us))
                    emit(lifecycle_event("TURN_START", session=session, turn=first["turn"], binding=binding,
                                         runner_generation=binding["runner_generation"],
                                         lifecycle_generation=session["lifecycle_generation"], arrival_us=first["planned_arrival_us"],
                                         request_id=first["request_id"], trigger=trigger, observed_us=now_us))
                    future = executor.submit(request_completion_replay, port, first, binding["slot_id"], raw_dir,
                                             spec["request_timeout_seconds"], cache_prompt, started_at_ns)
                    inflight[future] = {"event": first, "binding": binding, "dispatch_order": dispatch_order,
                                       "admitted_us": max(first["planned_arrival_us"], now_us),
                                       "arrival_us": first["planned_arrival_us"], "trigger": trigger}
                    session["state"] = "ACTIVE"
                    progressed = True
                if progressed:
                    blocked_since = None
                    continue
                now_us = clock_us()
                if expire_due(now_us):
                    continue
                if inflight:
                    next_arrival = min((item["planned_arrival_us"] for item in pending), default=None)
                    next_timer = min((item["timer"]["expiry_us"] for item in sessions.values() if item["timer"]), default=None)
                    wake = min(value for value in (next_arrival, next_timer) if value is not None) if (next_arrival is not None or next_timer is not None) else None
                    timeout = None if wake is None or wake <= now_us else (wake - now_us) / 1_000_000.0
                    wait(list(inflight), timeout=timeout, return_when=FIRST_COMPLETED)
                    continue
                if not pending and lifecycle_cfg["drain_after_last_arrival"]:
                    timers = [item["timer"]["expiry_us"] for item in sessions.values() if item["timer"]]
                    if timers:
                        time.sleep(max(0.0, (min(timers) - now_us) / 1_000_000.0))
                        continue
                    break
                next_arrival = min((item["planned_arrival_us"] for item in pending), default=None)
                next_timer = min((item["timer"]["expiry_us"] for item in sessions.values() if item["timer"]), default=None)
                if next_arrival is not None or next_timer is not None:
                    wake = min(value for value in (next_arrival, next_timer) if value is not None)
                    if wake > now_us:
                        time.sleep((wake - now_us) / 1_000_000.0)
                        continue
                if blocked_since is None:
                    blocked_since = time.monotonic()
                if time.monotonic() - blocked_since >= replay_cfg["admission_timeout_seconds"]:
                    raise RunnerError("replay admission queue cannot drain without lifecycle release")
                time.sleep(0.01)
        request_records.sort(key=lambda item: item["dispatch_order"])
        with responses_path.open("w", encoding="utf-8") as responses:
            for actual in request_records:
                responses.write(json.dumps(actual, ensure_ascii=False, sort_keys=True) + "\n")
        with lifecycle_path.open("w", encoding="utf-8") as journal:
            for event in lifecycle_records:
                journal.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        capture_slots(port, run_dir / "slots_after.json", spec["request_timeout_seconds"])
    except (OSError, RunnerError, ReplayError, subprocess.SubprocessError) as exc:
        lifecycle_error = str(exc)
    finally:
        with lifecycle_path.open("w", encoding="utf-8") as journal:
            for event in lifecycle_records:
                journal.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if sampler is not None:
            sampler_cleanup = terminate_process(sampler, "sampler", 5.0)
        if server is not None:
            server_cleanup = terminate_process(server, "server", 10.0)
        for handle in handles:
            handle.close()
    cleanup = {"server": server_cleanup, "sampler": sampler_cleanup,
               "residual_process": bool(server_cleanup["residual_process"] or sampler_cleanup["residual_process"]),
               "cleanup_complete": lifecycle_error is None and server_cleanup["exit_code"] == 0
               and sampler_cleanup["exit_code"] == 0 and not (server_cleanup["residual_process"] or sampler_cleanup["residual_process"])}
    dump(run_dir / "cleanup.json", cleanup)
    fidelity = check_fidelity(replay_plan, request_records, n_parallel=replay_cfg["n_parallel"])
    lifecycle_status = "PASS" if lifecycle_error is None else "FAIL"
    fidelity["lifecycle"] = {"status": lifecycle_status, "errors": [] if lifecycle_error is None else [lifecycle_error]}
    replay_record = {"plan": replay_plan.to_dict(), "schedule": schedule,
                     "admission": admission_records, "events": request_records,
                     "lifecycle": lifecycle_records, "workload_fidelity": fidelity}
    dump(run_dir / "replay.json", replay_record)
    execution = {
        "run_id": run["run_id"], "round": run["round"], "run_order": run["run_order"],
        "case_id": run["case_id"], "execution_index": execution_index, "run_mode": spec["run_mode"],
        "argv": argv, "environment": env, "server_identity": server_identity_record,
        "server_cgroup": server_cgroup, "pressure_basis": dict(spec["pressure_basis"]),
        "sampler_identity": sampler_identity_record, "sampler_schema": SAMPLE_SCHEMA,
        "sampler_argv": ["bash", str(MEMORY_SAMPLER), "--sample-process",
                         str(server_identity_record["pid"]) if server_identity_record else "NOT_STARTED",
                         str(samples_path), str(backing_dir), str(spec["sampler"]["interval_seconds"]),
                         str(server_cgroup.get("memory_current_file", "")) if server_cgroup else ""],
        "request_loop_started": request_loop_started, "request_count": len(request_records),
        "qualification": {"idle_seconds": None, "offload_timeout_seconds": None, "resume_request_id": None,
                          "idle": None, "offload_barrier": None, "resume": None},
        "characterization": {"idle_seconds": None, "settle_timeout_seconds": None, "target_tolerance_bytes": None,
                             "resume_request_id": None, "requested_target_bytes": None, "action_target_bytes": None,
                             "after_fill": None, "idle": None, "settle": None, "release_settled": None,
                             "settled": None, "resume": None, "after_measurement": None},
        "replay": replay_record, "lifecycle": {"enabled": True, "ttl_runtime_us": ttl_runtime_us,
                                                   "slot_save_path": str(slot_save_dir)},
    }
    dump(run_dir / "execution.json", execution)
    if not responses_path.exists():
        responses_path.write_text("", encoding="utf-8")
    complete = cleanup["cleanup_complete"] and lifecycle_error is None and fidelity["status"] == "PASS" and lifecycle_status == "PASS"
    return {"run_id": run["run_id"], "case_id": run["case_id"], "round": run["round"],
            "run_order": run["run_order"], "execution_index": execution_index,
            "directory": str(run_dir.relative_to(artifact)), "status": "complete" if complete else "incomplete",
            "error": lifecycle_error or (None if fidelity["status"] == "PASS" else "workload fidelity failed")}

def run_one_replay(
        artifact: pathlib.Path,
        run: dict[str, Any],
        case: dict[str, Any],
        spec: dict[str, Any],
        workload: dict[str, Any],
        execution_index: int,
) -> dict[str, Any]:
    replay_cfg = workload["replay"]
    replay_plan = load_replay_plan(replay_cfg)
    schedule = expand_schedule(replay_plan)
    if replay_cfg.get("lifecycle", {}).get("enabled", False):
        return run_one_replay_lifecycle(artifact, run, case, spec, workload, execution_index, replay_plan, schedule)
    run_dir = artifact / "runs" / run["run_id"]
    raw_dir = run_dir / "raw"
    backing_dir = run_dir / "backing"
    raw_dir.mkdir(parents=True)
    backing_dir.mkdir()
    replay_cfg = workload["replay"]
    replay_plan = load_replay_plan(replay_cfg)
    validate_replay_model_binding(replay_plan, spec)
    schedule = expand_schedule(replay_plan)
    dump(run_dir / "run.json", {
        "run_id": run["run_id"], "round": run["round"], "run_order": run["run_order"],
        "case_id": run["case_id"], "execution_index": execution_index, "case": run,
        "request_plan": schedule, "replay": replay_plan.to_dict(),
    })
    port = free_port()
    env = runtime_environment(spec, case, backing_dir)
    argv = server_argv(spec, port)
    stdout_path, stderr_path = run_dir / "server.stdout", run_dir / "server.stderr"
    sampler_stdout_path, sampler_stderr_path = run_dir / "sampler.stdout", run_dir / "sampler.stderr"
    samples_path, responses_path = run_dir / "memory_samples.tsv", run_dir / "responses.jsonl"
    server: subprocess.Popen[bytes] | None = None
    sampler: subprocess.Popen[bytes] | None = None
    server_identity_record: dict[str, Any] | None = None
    sampler_identity_record: dict[str, Any] | None = None
    server_cgroup: dict[str, Any] | None = None
    runner_cgroup = cgroup_identity(os.getpid())
    request_records: list[dict[str, Any]] = []
    admission_records: list[dict[str, Any]] = []
    lifecycle_error: str | None = None
    server_cleanup = {"pid": None, "pgid": None, "exit_code": None, "stop_requested": False,
                       "stop_signal": None, "term_timed_out": False, "kill_timed_out": False,
                       "pgid_check_complete": False, "residual_process": False}
    sampler_cleanup = dict(server_cleanup)
    handles: list[Any] = []
    started_at_ns = time.monotonic_ns()
    request_loop_started = False
    try:
        server_handle = stdout_path.open("wb"); handles.append(server_handle)
        server_err_handle = stderr_path.open("wb"); handles.append(server_err_handle)
        server = subprocess.Popen(argv, cwd=run_dir, env=env, stdout=server_handle,
                                  stderr=server_err_handle, start_new_session=True)
        server_identity_record = process_identity(server.pid, argv)
        server_cgroup = cgroup_scope(cgroup_identity(server.pid), runner_cgroup)
        expected_memory_max = spec["cgroup"]["expected_memory_max"]
        if expected_memory_max is not None and server_cgroup.get("memory_max") != expected_memory_max:
            raise RunnerError("server cgroup memory.max mismatch")
        if spec["pressure_basis"]["authority"] == "cgroup_finite" and not finite_memory_limit(server_cgroup.get("memory_max")):
            raise RunnerError("server cgroup does not provide a finite pressure authority")
        current_file = server_cgroup.get("memory_current_file") or ""
        sampler_env = dict(os.environ)
        sampler_env["KV_CONTROLLED_SAMPLE_SCHEMA"] = SAMPLE_SCHEMA
        sampler_env["KV_CONTROLLED_CGROUP_DIR"] = server_cgroup.get("path") or ""
        sampler_command = ["bash", str(MEMORY_SAMPLER), "--sample-process", str(server.pid),
                           str(samples_path), str(backing_dir), str(spec["sampler"]["interval_seconds"]), current_file]
        sampler_stdout_handle = sampler_stdout_path.open("wb"); handles.append(sampler_stdout_handle)
        sampler_stderr_handle = sampler_stderr_path.open("wb"); handles.append(sampler_stderr_handle)
        sampler = subprocess.Popen(sampler_command, cwd=run_dir, env=sampler_env,
                                   stdout=sampler_stdout_handle, stderr=sampler_stderr_handle,
                                   start_new_session=True)
        sampler_identity_record = process_identity(sampler.pid, sampler_command)
        wait_health(port, server, spec["health_timeout_seconds"])
        slots_before = capture_slots(
            port, run_dir / "slots_before.json", spec["request_timeout_seconds"])
        validate_replay_slot_capacity(
            slots_before, replay_plan, replay_cfg["n_parallel"])
        request_loop_started = True
        admission = __import__("multi_session_replay", fromlist=["SlotAdmission"]).SlotAdmission(
            replay_cfg["n_parallel"])
        session_last_turn = {
            session.logical_session_id: session.turns[-1].turn
            for session in replay_plan.sessions
        }
        session_next_turn: dict[str, int] = {
            session.logical_session_id: 1 for session in replay_plan.sessions
        }
        session_available_us: dict[str, int] = {
            session.logical_session_id: 0 for session in replay_plan.sessions
        }
        pending = list(schedule)
        inflight: dict[Any, dict[str, Any]] = {}
        inflight_sessions: set[str] = set()
        queued_recorded: set[str] = set()
        dispatch_count = 0
        blocked_since: float | None = None

        with ThreadPoolExecutor(max_workers=replay_cfg["n_parallel"]) as executor:
            while pending or inflight:
                progressed = False

                # Retire completed HTTP requests first.  Results are kept in
                # dispatch order when persisted, but their own timestamps retain
                # the true completion order.
                done = [future for future in inflight if future.done()]
                for future in sorted(done, key=lambda item: inflight[item]["dispatch_order"]):
                    meta = inflight.pop(future)
                    first = meta["event"]
                    sid = first["logical_session_id"]
                    try:
                        actual = future.result()
                    except Exception as exc:
                        raise RunnerError(
                            f"replay request failed for {first['request_id']}: {exc}") from exc
                    actual.update({
                        "dispatch_order": meta["dispatch_order"],
                        "admitted_us": meta["admitted_us"],
                        "dispatched_us": actual["started_us"],
                        "planned_arrival_us": first["planned_arrival_us"],
                        "arrival_lag_us": max(
                            0, actual["started_us"] - first["planned_arrival_us"]),
                        "admission_wait_us": max(
                            0, meta["admitted_us"] - first["planned_arrival_us"]),
                        "service_us": actual["completed_us"] - actual["started_us"],
                        "runner_generation": meta["binding"]["runner_generation"],
                    })
                    request_records.append(actual)
                    inflight_sessions.remove(sid)
                    session_available_us[sid] = actual["completed_us"]
                    session_next_turn[sid] += 1
                    if first["turn"] == session_last_turn[sid]:
                        finish = admission.finish(sid, actual["completed_us"])
                        admission_records.append({**first, **finish})
                    progressed = True

                now_us = (time.monotonic_ns() - started_at_ns) // 1000
                ready = [
                    item for item in pending
                    if item["planned_arrival_us"] <= now_us
                ]

                # Dispatch independent sessions concurrently, up to n_parallel.
                for first in ready:
                    if len(inflight) >= replay_cfg["n_parallel"]:
                        break
                    sid = first["logical_session_id"]
                    if sid in inflight_sessions:
                        continue
                    if first["turn"] != session_next_turn[sid]:
                        # A later turn may have arrived while the previous turn is
                        # still running; defer it, but never reinterpret it as a
                        # new session/admission event.
                        if first["turn"] > session_next_turn[sid]:
                            continue
                        raise ReplayError(
                            f"session {sid} turn order is not contiguous")

                    if sid in admission.bindings:
                        binding = admission.binding(sid)
                        admitted_us = max(
                            first["planned_arrival_us"], session_available_us[sid])
                    else:
                        was_waiting = sid in admission.waiting
                        decision = admission.admit(sid, now_us)
                        if decision["status"] == "queued":
                            if not was_waiting and sid not in queued_recorded:
                                admission_records.append({
                                    **first, **decision, "observed_us": now_us})
                                queued_recorded.add(sid)
                            continue
                        if decision["status"] != "admitted":
                            raise RunnerError(
                                f"unexpected replay admission state {decision['status']!r}")
                        admission_records.append({
                            **first, **decision, "observed_us": now_us})
                        queued_recorded.discard(sid)
                        binding = {
                            "slot_id": decision["slot_id"],
                            "seq_id": decision["seq_id"],
                            "runner_generation": decision["runner_generation"],
                        }
                        admitted_us = max(first["planned_arrival_us"], now_us)

                    if not 0 <= binding["slot_id"] < replay_cfg["n_parallel"]:
                        raise RunnerError("replay slot binding is out of range")

                    pending.remove(first)
                    inflight_sessions.add(sid)
                    current_dispatch_order = dispatch_count
                    dispatch_count += 1
                    future = executor.submit(
                        request_completion_replay,
                        port,
                        first,
                        binding["slot_id"],
                        raw_dir,
                        spec["request_timeout_seconds"],
                        first["turn"] > 1,
                        started_at_ns,
                    )
                    inflight[future] = {
                        "event": first,
                        "binding": binding,
                        "dispatch_order": current_dispatch_order,
                        "admitted_us": admitted_us,
                    }
                    progressed = True

                if progressed:
                    blocked_since = None
                    continue

                if inflight:
                    next_arrival_us = min(
                        (item["planned_arrival_us"] for item in pending),
                        default=None,
                    )
                    timeout = None
                    if next_arrival_us is not None and next_arrival_us > now_us:
                        timeout = max(
                            0.0, (next_arrival_us - now_us) / 1_000_000.0)
                    wait(
                        list(inflight),
                        timeout=timeout,
                        return_when=FIRST_COMPLETED,
                    )
                    continue

                if not pending:
                    break

                # No request is in flight.  If all current slots are held by
                # idle sessions, a queued session must wait for one of those
                # sessions' future turns to complete; this qualification gate
                # deliberately does not invent eviction/TTL semantics.
                future_live = [
                    item for item in pending
                    if item["logical_session_id"] in admission.bindings
                ]
                next_wakeup_us = min(
                    (item["planned_arrival_us"] for item in future_live),
                    default=min(item["planned_arrival_us"] for item in pending),
                )
                if next_wakeup_us > now_us:
                    blocked_since = None
                    time.sleep((next_wakeup_us - now_us) / 1_000_000.0)
                    continue

                if blocked_since is None:
                    blocked_since = time.monotonic()
                if time.monotonic() - blocked_since >= replay_cfg["admission_timeout_seconds"]:
                    raise RunnerError(
                        "replay admission queue cannot drain without lifecycle release")
                time.sleep(0.01)

        request_records.sort(key=lambda item: item["dispatch_order"])
        with responses_path.open("w", encoding="utf-8") as responses:
            for actual in request_records:
                responses.write(
                    json.dumps(actual, ensure_ascii=False, sort_keys=True) + "\n")
        capture_slots(port, run_dir / "slots_after.json", spec["request_timeout_seconds"])
    except (OSError, RunnerError, ReplayError, subprocess.SubprocessError) as exc:
        lifecycle_error = str(exc)
    finally:
        if sampler is not None:
            sampler_cleanup = terminate_process(sampler, "sampler", 5.0)
        if server is not None:
            server_cleanup = terminate_process(server, "server", 10.0)
        for handle in handles:
            handle.close()
    cleanup = {"server": server_cleanup, "sampler": sampler_cleanup,
               "residual_process": bool(server_cleanup["residual_process"] or sampler_cleanup["residual_process"]),
               "cleanup_complete": lifecycle_error is None and server_cleanup["exit_code"] == 0
               and sampler_cleanup["exit_code"] == 0 and not (server_cleanup["residual_process"] or sampler_cleanup["residual_process"])}
    dump(run_dir / "cleanup.json", cleanup)
    fidelity = check_fidelity(replay_plan, request_records, n_parallel=replay_cfg["n_parallel"])
    replay_record = {"plan": replay_plan.to_dict(), "schedule": schedule,
                     "admission": admission_records, "events": request_records,
                     "workload_fidelity": fidelity}
    dump(run_dir / "replay.json", replay_record)
    execution = {
        "run_id": run["run_id"], "round": run["round"], "run_order": run["run_order"],
        "case_id": run["case_id"], "execution_index": execution_index, "run_mode": spec["run_mode"],
        "argv": argv, "environment": env, "server_identity": server_identity_record,
        "server_cgroup": server_cgroup, "pressure_basis": dict(spec["pressure_basis"]),
        "sampler_identity": sampler_identity_record, "sampler_schema": SAMPLE_SCHEMA,
        "sampler_argv": ["bash", str(MEMORY_SAMPLER), "--sample-process",
                         str(server_identity_record["pid"]) if server_identity_record else "NOT_STARTED",
                         str(samples_path), str(backing_dir), str(spec["sampler"]["interval_seconds"]),
                         str(server_cgroup.get("memory_current_file", "")) if server_cgroup else ""],
        "request_loop_started": request_loop_started, "request_count": len(request_records),
        "qualification": {"idle_seconds": None, "offload_timeout_seconds": None, "resume_request_id": None,
                          "idle": None, "offload_barrier": None, "resume": None},
        "characterization": {"idle_seconds": None, "settle_timeout_seconds": None, "target_tolerance_bytes": None,
                             "resume_request_id": None, "requested_target_bytes": None, "action_target_bytes": None,
                             "after_fill": None, "idle": None, "settle": None, "release_settled": None,
                             "settled": None, "resume": None, "after_measurement": None},
        "replay": replay_record,
    }
    dump(run_dir / "execution.json", execution)
    if not responses_path.exists():
        responses_path.write_text("", encoding="utf-8")
    complete = cleanup["cleanup_complete"] and lifecycle_error is None and fidelity["status"] == "PASS"
    return {"run_id": run["run_id"], "case_id": run["case_id"], "round": run["round"],
            "run_order": run["run_order"], "execution_index": execution_index,
            "directory": str(run_dir.relative_to(artifact)), "status": "complete" if complete else "incomplete",
            "error": lifecycle_error or (None if fidelity["status"] == "PASS" else "workload fidelity failed")}


def run_one(
        artifact: pathlib.Path,
        run: dict[str, Any],
        case: dict[str, Any],
        spec: dict[str, Any],
        workload: dict[str, Any],
        execution_index: int,
) -> dict[str, Any]:
    if "replay" in workload:
        return run_one_replay(artifact, run, case, spec, workload, execution_index)
    run_dir = artifact / "runs" / run["run_id"]
    raw_dir = run_dir / "raw"
    backing_dir = run_dir / "backing"
    raw_dir.mkdir(parents=True)
    backing_dir.mkdir()
    request_plan = expanded_request_plan(workload, case["policy"], spec["run_mode"])
    dump(run_dir / "run.json", {
        "run_id": run["run_id"],
        "round": run["round"],
        "run_order": run["run_order"],
        "case_id": run["case_id"],
        "execution_index": execution_index,
        "case": run,
        "request_plan": request_plan,
    })
    port = free_port()
    env = runtime_environment(spec, case, backing_dir)
    argv = server_argv(spec, port)
    stdout_path = run_dir / "server.stdout"
    stderr_path = run_dir / "server.stderr"
    sampler_stdout_path = run_dir / "sampler.stdout"
    sampler_stderr_path = run_dir / "sampler.stderr"
    samples_path = run_dir / "memory_samples.tsv"
    responses_path = run_dir / "responses.jsonl"
    server: subprocess.Popen[bytes] | None = None
    sampler: subprocess.Popen[bytes] | None = None
    server_identity_record: dict[str, Any] | None = None
    sampler_identity_record: dict[str, Any] | None = None
    server_cgroup: dict[str, Any] | None = None
    runner_cgroup = cgroup_identity(os.getpid())
    request_loop_started = False
    request_records: list[dict[str, Any]] = []
    qualification_config = (
        workload["qualification"] if spec["run_mode"] == "qualification" else None)
    characterization_config = (
        workload["characterization"] if spec["run_mode"] == "characterization" else None)
    qualification_record: dict[str, Any] = {
        "idle_seconds": (
            qualification_config["idle_seconds"] if qualification_config is not None else None),
        "offload_timeout_seconds": (
            qualification_config["offload_timeout_seconds"]
            if qualification_config is not None else None),
        "resume_request_id": (
            qualification_config["resume_request_id"] if qualification_config is not None else None),
        "idle": None,
        "offload_barrier": None,
        "resume": None,
    }
    characterization_record: dict[str, Any] = {
        "idle_seconds": (
            characterization_config["idle_seconds"] if characterization_config is not None else None),
        "settle_timeout_seconds": (
            characterization_config["settle_timeout_seconds"]
            if characterization_config is not None else None),
        "target_tolerance_bytes": (
            characterization_config["target_tolerance_bytes"]
            if characterization_config is not None else None),
        "resume_request_id": (
            characterization_config["resume_request_id"]
            if characterization_config is not None else None),
        "requested_target_bytes": (
            case["kv_target_bytes"] if characterization_config is not None else None),
        "action_target_bytes": (
            case["action_target_bytes"] if characterization_config is not None else None),
        "after_fill": None,
        "idle": None,
        "settle": None,
        "release_settled": None,
        "settled": None,
        "resume": None,
        "after_measurement": None,
    }
    lifecycle_error: str | None = None
    server_exit: int | None = None
    sampler_exit: int | None = None
    server_handle = None
    server_cleanup: dict[str, Any] = {
        "pid": None, "pgid": None, "exit_code": None, "stop_requested": False,
        "stop_signal": None, "term_timed_out": False, "kill_timed_out": False,
        "pgid_check_complete": False, "residual_process": False,
    }
    sampler_cleanup: dict[str, Any] = {
        "pid": None, "pgid": None, "exit_code": None, "stop_requested": False,
        "stop_signal": None, "term_timed_out": False, "kill_timed_out": False,
        "pgid_check_complete": False, "residual_process": False,
    }

    def snapshot_reference(filename: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {"path": filename, "captured_mono_ns": snapshot["captured_mono_ns"]}

    try:
        server_handle = stdout_path.open("wb")
        server_err_handle = stderr_path.open("wb")
        server = subprocess.Popen(
            argv,
            cwd=run_dir,
            env=env,
            stdout=server_handle,
            stderr=server_err_handle,
            start_new_session=True,
        )
        server_identity_record = process_identity(server.pid, argv)
        server_cgroup = cgroup_scope(cgroup_identity(server.pid), runner_cgroup)
        expected_memory_max = spec["cgroup"]["expected_memory_max"]
        if expected_memory_max is not None and server_cgroup.get("memory_max") != expected_memory_max:
            raise RunnerError(
                f"server cgroup memory.max mismatch: expected={expected_memory_max!r} "
                f"actual={server_cgroup.get('memory_max')!r}")
        if spec["pressure_basis"]["authority"] == "cgroup_finite" and not finite_memory_limit(
                server_cgroup.get("memory_max")):
            raise RunnerError("server cgroup does not provide a finite pressure authority")
        current_file = server_cgroup.get("memory_current_file") or ""
        sampler_env = dict(os.environ)
        sampler_env["KV_CONTROLLED_SAMPLE_SCHEMA"] = SAMPLE_SCHEMA
        sampler_env["KV_CONTROLLED_CGROUP_DIR"] = server_cgroup.get("path") or ""
        sampler_command = [
            "bash", str(MEMORY_SAMPLER), "--sample-process", str(server.pid),
            str(samples_path), str(backing_dir), str(spec["sampler"]["interval_seconds"]), current_file,
        ]
        sampler_stdout_handle = sampler_stdout_path.open("wb")
        sampler_stderr_handle = sampler_stderr_path.open("wb")
        sampler = subprocess.Popen(
            sampler_command,
            cwd=run_dir,
            env=sampler_env,
            stdout=sampler_stdout_handle,
            stderr=sampler_stderr_handle,
            start_new_session=True,
        )
        sampler_identity_record = process_identity(sampler.pid, sampler_command)
        wait_health(port, server, spec["health_timeout_seconds"])
        capture_slots(port, run_dir / "slots_before.json", spec["request_timeout_seconds"])
        request_loop_started = True
        request_by_id = {
            item["request_id"]: item for item in workload["warmup"] + workload["requests"]}

        with responses_path.open("w", encoding="utf-8") as responses:
            def issue(item: dict[str, Any]) -> dict[str, Any]:
                request = request_by_id.get(item["request_id"])
                if request is None:
                    raise RunnerError(f"missing request body for {item['request_id']}")
                record = request_completion(
                    port, item, request, raw_dir, spec["request_timeout_seconds"])
                responses.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                responses.flush()
                request_records.append(record)
                return record

            if spec["run_mode"] == "qualification":
                resume_planned = qualification_config is not None and case["policy"] in SWAP_POLICIES
                fill_plan = request_plan[:-1] if resume_planned else request_plan
                for item in fill_plan:
                    issue(item)
                if qualification_config is None:
                    raise RunnerError("qualification configuration is missing")
                stderr_start_offset = stderr_path.stat().st_size
                idle_started_mono_ns = time.monotonic_ns()
                time.sleep(qualification_config["idle_seconds"])
                idle_finished_mono_ns = time.monotonic_ns()
                qualification_record["idle"] = {
                    "started_mono_ns": idle_started_mono_ns,
                    "finished_mono_ns": idle_finished_mono_ns,
                    "duration_ns": idle_finished_mono_ns - idle_started_mono_ns,
                    "stderr_start_offset": stderr_start_offset,
                }
                if case["policy"] in SWAP_POLICIES:
                    barrier = wait_real_offload_barrier(
                        stderr_path,
                        stderr_start_offset,
                        qualification_config["offload_timeout_seconds"],
                        pressure_basis_source(spec["pressure_basis"]),
                        server,
                    )
                    qualification_record["offload_barrier"] = barrier
                    if barrier["status"] != "passed":
                        raise RunnerError(
                            "qualification timed out waiting for a real OFFLOAD with transaction-local resident drop")
                    resume_item = request_plan[-1]
                    while time.monotonic_ns() <= barrier["completed_mono_ns"]:
                        time.sleep(0)
                    resume_stderr_start_offset = stderr_path.stat().st_size
                    record = issue(resume_item)
                    resume_stderr_end_offset = stderr_path.stat().st_size
                    if record["started_mono_ns"] <= barrier["completed_mono_ns"]:
                        raise RunnerError("qualification resume did not start after the OFFLOAD barrier")
                    qualification_record["resume"] = {
                        "sequence": record["sequence"],
                        "request_id": record["request_id"],
                        "started_mono_ns": record["started_mono_ns"],
                        "finished_mono_ns": record["finished_mono_ns"],
                        "stderr_start_offset": resume_stderr_start_offset,
                        "stderr_end_offset": resume_stderr_end_offset,
                    }
            else:
                if characterization_config is None:
                    raise RunnerError("characterization configuration is missing")
                warmup_plan = request_plan[:len(workload["warmup"])]
                measurement_plan = request_plan[len(workload["warmup"]):]
                for item in warmup_plan:
                    issue(item)
                after_fill = capture_slots(
                    port, run_dir / "slots_after_fill.json", spec["request_timeout_seconds"])
                characterization_record["after_fill"] = snapshot_reference(
                    "slots_after_fill.json", after_fill)

                if case["policy"] in SWAP_POLICIES:
                    stderr_start_offset = stderr_path.stat().st_size
                    idle_started_mono_ns = time.monotonic_ns()
                    time.sleep(characterization_config["idle_seconds"])
                    idle_finished_mono_ns = time.monotonic_ns()
                    characterization_record["idle"] = {
                        "started_mono_ns": idle_started_mono_ns,
                        "finished_mono_ns": idle_finished_mono_ns,
                        "duration_ns": idle_finished_mono_ns - idle_started_mono_ns,
                        "stderr_start_offset": stderr_start_offset,
                    }
                    def capture_release_settle() -> dict[str, Any]:
                        snapshot = capture_slots(
                            port,
                            run_dir / "slots_release_settled.json",
                            spec["request_timeout_seconds"],
                        )
                        return snapshot_reference("slots_release_settled.json", snapshot)

                    settle = wait_budget_settle(
                        stderr_path,
                        stderr_start_offset,
                        characterization_config["settle_timeout_seconds"],
                        pressure_basis_source(spec["pressure_basis"]),
                        int(case["kv_target_bytes"]),
                        int(case["action_target_bytes"]),
                        int(spec["max_blocks"]),
                        characterization_config["target_tolerance_bytes"],
                        server,
                        capture_release_settle=capture_release_settle,
                    )
                    characterization_record["settle"] = settle
                    if settle["status"] == "timeout":
                        raise RunnerError(
                            "characterization timed out waiting for target closure or budget_unmet_terminal")
                    settled = capture_slots(
                        port, run_dir / "slots_settled.json", spec["request_timeout_seconds"])
                    settled_resident_bytes = captured_resident_bytes(
                        settled, allow_missing=settle["status"] == "unmet_floor")
                    if settled_resident_bytes is None:
                        settled_resident_bytes = int(
                            settle["terminal_decision"]["budget_resident_bytes"])
                        settle["physical_resident_bytes"] = None
                    else:
                        settle["physical_resident_bytes"] = settled_resident_bytes
                    target_bytes = int(case["kv_target_bytes"])
                    tolerance_bytes = characterization_config["target_tolerance_bytes"]
                    if settle["status"] in {"debt_closed", "target_reached"}:
                        if settled_resident_bytes > target_bytes + tolerance_bytes:
                            raise RunnerError(
                                "budget debt closed without actual physical resident target closure")
                        settle["status"] = "target_reached"
                    elif (
                        settled_resident_bytes <= target_bytes
                        or abs(
                            settled_resident_bytes
                            - settle["terminal_decision"]["budget_resident_bytes"])
                            > tolerance_bytes
                    ):
                        raise RunnerError(
                            "budget_unmet_terminal differs from actual settled physical resident")
                    characterization_record["release_settled"] = settle["release_settle"]
                    if (
                        characterization_record["release_settled"] is None
                        or settle["release_boundary"] is None
                    ):
                        raise RunnerError(
                            "V2 characterization has no production RELEASE settle boundary")
                    characterization_record["settled"] = snapshot_reference(
                        "slots_settled.json", settled)

                elif case["policy"] == "release_only":
                    stderr_start_offset = stderr_path.stat().st_size
                    idle_started_mono_ns = time.monotonic_ns()
                    time.sleep(characterization_config["idle_seconds"])
                    idle_finished_mono_ns = time.monotonic_ns()
                    characterization_record["idle"] = {
                        "started_mono_ns": idle_started_mono_ns,
                        "finished_mono_ns": idle_finished_mono_ns,
                        "duration_ns": idle_finished_mono_ns - idle_started_mono_ns,
                        "stderr_start_offset": stderr_start_offset,
                    }
                    settle = wait_budget_settle(
                        stderr_path,
                        stderr_start_offset,
                        characterization_config["settle_timeout_seconds"],
                        pressure_basis_source(spec["pressure_basis"]),
                        int(case["kv_target_bytes"]),
                        int(case["action_target_bytes"]),
                        int(spec["max_blocks"]),
                        characterization_config["target_tolerance_bytes"],
                        server,
                        release_only=True,
                    )
                    characterization_record["settle"] = settle
                    if settle["status"] == "timeout":
                        raise RunnerError(
                            "release_only characterization timed out waiting for Unified RELEASE closure")
                    settled = capture_slots(
                        port, run_dir / "slots_settled.json", spec["request_timeout_seconds"])
                    settled_resident_bytes = captured_resident_bytes(settled)
                    settle["physical_resident_bytes"] = settled_resident_bytes
                    terminal = settle["terminal_decision"]
                    if terminal is None:
                        raise RunnerError("release_only characterization has no terminal RELEASE decision")
                    tolerance_bytes = characterization_config["target_tolerance_bytes"]
                    if settle["status"] == "release_settled":
                        if (
                            settled_resident_bytes > int(case["kv_target_bytes"]) + tolerance_bytes
                            or terminal["budget_debt_after_bytes"] != 0
                            or terminal["unmet_budget_bytes_after"] != 0
                        ):
                            raise RunnerError(
                                "Unified RELEASE reported closure without actual physical resident closure")
                    elif settle["status"] == "release_no_candidate":
                        if abs(
                            settled_resident_bytes - terminal["budget_resident_bytes"]
                        ) > tolerance_bytes:
                            raise RunnerError(
                                "RELEASE no_candidate terminal differs from actual settled physical resident")
                    else:
                        raise RunnerError(
                            f"unexpected release_only settle status: {settle['status']}")
                    characterization_record["release_settled"] = snapshot_reference(
                        "slots_settled.json", settled)
                    characterization_record["settled"] = snapshot_reference(
                        "slots_settled.json", settled)

                if not measurement_plan:
                    raise RunnerError("characterization measurement plan is empty")
                first_item = measurement_plan[0]
                resume_stderr_start_offset = stderr_path.stat().st_size
                first_record = issue(first_item)
                resume_stderr_end_offset = stderr_path.stat().st_size
                if case["policy"] in SWAP_POLICIES:
                    settle = characterization_record["settle"]
                    if first_record["started_mono_ns"] <= settle["completed_mono_ns"]:
                        raise RunnerError(
                            "characterization resume started before budget settle completed")
                    characterization_record["resume"] = {
                        "sequence": first_record["sequence"],
                        "request_id": first_record["request_id"],
                        "started_mono_ns": first_record["started_mono_ns"],
                        "finished_mono_ns": first_record["finished_mono_ns"],
                        "stderr_start_offset": resume_stderr_start_offset,
                        "stderr_end_offset": resume_stderr_end_offset,
                    }
                after_measurement = capture_slots(
                    port, run_dir / "slots_after_measurement.json", spec["request_timeout_seconds"])
                characterization_record["after_measurement"] = snapshot_reference(
                    "slots_after_measurement.json", after_measurement)
                for item in measurement_plan[1:]:
                    issue(item)

        capture_slots(port, run_dir / "slots_after.json", spec["request_timeout_seconds"])
    except (OSError, RunnerError, subprocess.SubprocessError) as exc:
        lifecycle_error = str(exc)
    finally:
        if sampler is not None:
            sampler_cleanup = terminate_process(sampler, "sampler", 5.0)
            sampler_exit = sampler_cleanup["exit_code"]
        if server is not None:
            server_cleanup = terminate_process(server, "server", 10.0)
            server_exit = server_cleanup["exit_code"]
        for handle in (
            locals().get("server_handle"), locals().get("server_err_handle"),
            locals().get("sampler_stdout_handle"), locals().get("sampler_stderr_handle"),
        ):
            if handle is not None:
                handle.close()
    residual_process = bool(server_cleanup["residual_process"] or sampler_cleanup["residual_process"])
    cleanup = {
        "server": server_cleanup,
        "sampler": sampler_cleanup,
        "residual_process": residual_process,
        "cleanup_complete": lifecycle_error is None and server_exit == 0 and sampler_exit == 0
            and server_cleanup["pgid_check_complete"] and sampler_cleanup["pgid_check_complete"]
            and not residual_process,
    }
    dump(run_dir / "cleanup.json", cleanup)
    execution = {
        "run_id": run["run_id"],
        "round": run["round"],
        "run_order": run["run_order"],
        "case_id": run["case_id"],
        "execution_index": execution_index,
        "run_mode": spec["run_mode"],
        "argv": argv,
        "environment": env,
        "server_identity": server_identity_record,
        "server_cgroup": server_cgroup,
        "pressure_basis": dict(spec["pressure_basis"]),
        "sampler_identity": sampler_identity_record,
        "sampler_schema": SAMPLE_SCHEMA,
        "sampler_argv": [
            "bash", str(MEMORY_SAMPLER), "--sample-process",
            str(server_identity_record["pid"]) if server_identity_record else "NOT_STARTED",
            str(samples_path), str(backing_dir), str(spec["sampler"]["interval_seconds"]),
            str(server_cgroup.get("memory_current_file", "")) if server_cgroup else "",
        ],
        "request_loop_started": request_loop_started,
        "request_count": len(request_records),
        "qualification": qualification_record,
        "characterization": characterization_record,
    }
    dump(run_dir / "execution.json", execution)
    if not responses_path.exists():
        responses_path.write_text("", encoding="utf-8")
    complete = cleanup["cleanup_complete"] and lifecycle_error is None
    return {
        "run_id": run["run_id"],
        "case_id": run["case_id"],
        "round": run["round"],
        "run_order": run["run_order"],
        "execution_index": execution_index,
        "directory": str(run_dir.relative_to(artifact)),
        "status": "complete" if complete else "incomplete",
        "error": lifecycle_error,
    }


def framework_provenance() -> dict[str, Any]:
    return {
        "runner": file_identity(RUNNER),
        "parser": file_identity(PARSER),
        "memory_sampler": file_identity(MEMORY_SAMPLER),
    }


def base_manifest(
        spec: dict[str, Any],
        plan: list[dict[str, Any]],
        artifact_id: str,
        runner_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "artifact_id": artifact_id,
        "runner_status": runner_status,
        "created_at_utc": utc_now(),
        "framework": framework_provenance(),
        "provenance": {
            "git": git_provenance(),
            "host": host_provenance(),
            "binary": file_identity(pathlib.Path(spec["binary"])),
            "model": file_identity(pathlib.Path(spec["model"])),
            "model_quantization": spec["model_quantization"],
            "runner_pid": os.getpid(),
            "runner_cgroup": cgroup_identity(os.getpid()),
        },
        "spec": spec,
        "planned_runs": plan,
        "run_results": [],
        "unsupported": None,
    }


def write_manifest(artifact: pathlib.Path, manifest: dict[str, Any]) -> None:
    dump(artifact / "manifest.json", manifest)


def execute(spec_path: pathlib.Path, output: pathlib.Path, dry_run: bool) -> int:
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    spec, cases, plan, workload = validate_spec(raw)
    if spec["run_kind"] == "formal":
        provenance = git_provenance()
        if provenance["dirty_status"]:
            raise RunnerError("formal run requires a clean worktree")
    artifact = output
    artifact.mkdir(parents=True, exist_ok=False)
    (artifact / "runs").mkdir()
    artifact_id = uuid.uuid4().hex
    manifest = base_manifest(spec, plan, artifact_id, "run_in_progress")
    write_manifest(artifact, manifest)
    unsupported: UnsupportedPlan | None = None
    for item in plan:
        if item["policy"] not in SUPPORTED_POLICIES:
            unsupported = UnsupportedPlan(
                "pre_workload_policy",
                f"policy={item['policy']} is not implemented in KV-Bench-V0",
            )
            break
        if item["kv_representation"] not in SUPPORTED_KV_REPRESENTATIONS:
            unsupported = UnsupportedPlan(
                "pre_workload_kv_representation",
                f"kv_representation={item['kv_representation']} is not implemented in KV-Bench-V0",
            )
            break
        if item["loading_mode"] not in SUPPORTED_LOADING_MODES:
            unsupported = UnsupportedPlan(
                "pre_workload_loading_mode",
                f"loading_mode={item['loading_mode']} is not implemented in KV-Bench-V0",
            )
            break
        if item["restore"] not in SUPPORTED_RESTORES:
            unsupported = UnsupportedPlan(
                "pre_workload_restore",
                f"restore={item['restore']} is not implemented in KV-Bench-V0",
            )
            break
        if item["prefault"] not in SUPPORTED_PREFAULTS:
            unsupported = UnsupportedPlan(
                "pre_workload_prefault",
                f"prefault={item['prefault']} is not implemented in KV-Bench-V0",
            )
            break
    if unsupported is None:
        try:
            validate_pressure_authority(spec, plan)
        except UnsupportedPlan as exc:
            unsupported = exc
    if unsupported is None and not dry_run:
        if not pathlib.Path(spec["binary"]).is_file() or not os.access(spec["binary"], os.X_OK):
            unsupported = UnsupportedPlan("pre_workload_binary_model", "binary is missing or not executable")
        elif not pathlib.Path(spec["model"]).is_file():
            unsupported = UnsupportedPlan("pre_workload_binary_model", "model is missing")
    if unsupported is not None:
        manifest["runner_status"] = "UNSUPPORTED"
        manifest["unsupported"] = {
            "stage": unsupported.stage,
            "reason": unsupported.reason,
            "workload_started": False,
        }
        write_manifest(artifact, manifest)
        print(json.dumps({"artifact": str(artifact), "runner_status": "UNSUPPORTED"}))
        return 3
    if dry_run:
        manifest["runner_status"] = "DRY_RUN"
        manifest["dry_run"] = {
            "workload_started": False,
            "request_plans": {
                case_id: expanded_request_plan(workload, case["policy"], spec["run_mode"])
                for case_id, case in cases.items()
            },
        }
        write_manifest(artifact, manifest)
        print(json.dumps({"artifact": str(artifact), "runner_status": "DRY_RUN"}))
        return 0
    results: list[dict[str, Any]] = []
    for execution_index, run in enumerate(plan):
        result = run_one(artifact, run, cases[run["case_id"]], spec, workload, execution_index)
        results.append(result)
        manifest["run_results"] = results
        manifest["runner_status"] = "run_in_progress"
        write_manifest(artifact, manifest)
    manifest["runner_status"] = "run_complete" if all(
        item["status"] == "complete" for item in results
    ) else "run_incomplete"
    manifest["finished_at_utc"] = utc_now()
    write_manifest(artifact, manifest)
    print(json.dumps({"artifact": str(artifact), "runner_status": manifest["runner_status"]}))
    return 0 if manifest["runner_status"] == "run_complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        return execute(args.spec, args.output, args.dry_run)
    except UnsupportedPlan as exc:
        print(f"UNSUPPORTED: {exc}", file=sys.stderr)
        return 3
    except (OSError, json.JSONDecodeError, RunnerError, subprocess.CalledProcessError) as exc:
        print(f"runner error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
