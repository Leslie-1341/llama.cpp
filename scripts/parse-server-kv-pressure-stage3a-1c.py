#!/usr/bin/env python3
"""Fail-closed parser for Stage 3A-1C server pressure telemetry artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import signal
import statistics
import subprocess
import sys
from typing import Any, NoReturn


PLAN = (
    ("ab_r1_off", "ab", 1, 1, "OFF"), ("ab_r1_on", "ab", 1, 2, "ON"),
    ("ab_r2_on", "ab", 2, 1, "ON"), ("ab_r2_off", "ab", 2, 2, "OFF"),
    ("ab_r3_off", "ab", 3, 1, "OFF"), ("ab_r3_on", "ab", 3, 2, "ON"),
    ("on_lifecycle", "lifecycle", 0, 0, "ON"),
    ("on_idle_250ms", "idle_limit", 0, 0, "ON"),
    ("strace_off_correctness", "strace", 0, 1, "OFF"),
    ("strace_on_correctness", "strace", 0, 2, "ON"),
)
ON_ENV = {
    "LLAMA_KV_PRESSURE_SAMPLER": "1",
    "LLAMA_KV_LOW_WATER_RSS_KB": "1", "LLAMA_KV_PRESSURE_RSS_KB": "2",
    "LLAMA_KV_CRITICAL_RSS_KB": "3",
    "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "60000",
    "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "60000",
}
ZERO_ENV = (
    "LLAMA_KV_PAGED_RELEASE", "LLAMA_KV_PAGED_SWAP", "LLAMA_KV_PAGED_IDLE_SWAP",
    "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE", "LLAMA_KV_PAGED_RESUME_PREFETCH",
    "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE", "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED",
    "LLAMA_KV_SWAP", "LLAMA_KV_SWAP_MADVISE",
)
TIMEOUTS_S = {
    "startup": 5.0, "health": 180.0, "completion": 180.0,
    "sleep": 30.0, "resume": 180.0, "shutdown": 15.0, "case_total": 430.0,
}
PRESSURE_SETTINGS_PURPOSE = "FORCED_LIFECYCLE_STATE_VALIDATION_ONLY_NOT_REAL_DEPLOYMENT_THRESHOLDS"
MARKER_KEYS = {
    "state", "previous_state", "source", "sample_valid", "stale", "config_valid",
    "rss_kb", "cgroup_current_bytes", "cgroup_max_bytes", "cgroup_current_kb",
    "cgroup_max_kb", "cgroup_high_kb", "psi_some_avg10", "psi_full_avg10",
    "sample_latency_ns", "sample_count", "skip_count", "idle", "trigger",
}
TRIGGERS = {"first", "state", "source", "stale", "periodic"}
# Structured action evidence only.  Bare prose words are deliberately not banned.
FORBIDDEN_LOG = re.compile(
    r"(?i)(?:paged_release_blocks\s*\(|release_blocks\s*\(|swap_(?:in|out)\s*\(|"
    r"prefetch_seq(?:_step)?\s*\(|(?:kv|paged)_(?:release|swap|prefetch|madvise|reclaim)_"
    r"(?:action|event|marker|telemetry)\b|(?:(?:kv|paged)_)?(?:release|swap|prefetch|madvise|reclaim)_"
    r"(?:calls|blocks|bytes|actions?|enabled)=[1-9][0-9]*)"
)
METRICS = (
    "ttft_ms", "tpot_ms", "throughput_tps",
    "sse_chunk_interval_p95_ms", "sse_chunk_interval_p99_ms",
)
ROOT_OUTPUTS = {"summary.json", "performance.tsv"}
BASE_CASE_FILES = {
    "case.json", "execution.json", "environment.json", "process.json", "result.json",
    "health.json", "metrics.json", "completion.sse", "completion.events.json",
    "completion_window.json", "server.stdout", "server.stderr", "phases.json",
}


class ArtifactError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def load(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot read valid JSON: {path}") from exc


def text(path: pathlib.Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError as exc:
        raise ArtifactError(f"cannot read {path}") from exc


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ArtifactError(f"cannot hash {path}") from exc
    return digest.hexdigest()


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_identity(item: Any, label: str) -> pathlib.Path:
    if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
        raise ArtifactError(f"invalid {label} identity schema")
    path = pathlib.Path(item["path"])
    if (not path.is_file() or path.stat().st_size != item["size"] or
            sha256(path) != item["sha256"]):
        raise ArtifactError(f"{label} identity mismatch")
    return path.resolve()


def inventory_entries(root: pathlib.Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(item for item in root.rglob("*")
                           if item.is_file() and item.name != "inventory.sha256.json")
    ]


def verify_inventory(root: pathlib.Path) -> None:
    inventory = load(root / "inventory.sha256.json")
    if not isinstance(inventory, dict) or inventory.get("algorithm") != "sha256" or inventory.get("excludes") != ["inventory.sha256.json"]:
        raise ArtifactError("invalid inventory schema")
    files = inventory.get("files")
    if not isinstance(files, list) or any(not isinstance(item, dict) or set(item) != {"path", "size", "sha256"} for item in files):
        raise ArtifactError("invalid inventory entries")
    paths = [item["path"] for item in files]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ArtifactError("inventory paths are duplicate or out of order")
    actual = inventory_entries(root)
    if files != actual:
        raise ArtifactError("artifact file set, size, or checksum differs from inventory")


def write_inventory(root: pathlib.Path) -> None:
    write_json(root / "inventory.sha256.json", {
        "algorithm": "sha256", "excludes": ["inventory.sha256.json"],
        "files": inventory_entries(root),
    })


def marker_fields(line: str) -> dict[str, str]:
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_]*)=([^\s]+)", line)
    fields: dict[str, str] = {}
    for key, value in pairs:
        if key in fields:
            raise ArtifactError(f"duplicate telemetry key: {key}")
        fields[key] = value
    if set(fields) != MARKER_KEYS:
        raise ArtifactError(
            f"telemetry keys mismatch: missing={sorted(MARKER_KEYS-set(fields))} "
            f"extra={sorted(set(fields)-MARKER_KEYS)}")
    for key in MARKER_KEYS - {"state", "previous_state", "source", "trigger"}:
        if re.fullmatch(r"[0-9]+", fields[key]) is None:
            raise ArtifactError(f"invalid integer telemetry field {key}")
    if fields["state"] not in {"NORMAL", "PRESSURE", "CRITICAL", "RECOVERY"}:
        raise ArtifactError("invalid telemetry state")
    if fields["previous_state"] not in {"NORMAL", "PRESSURE", "CRITICAL", "RECOVERY"}:
        raise ArtifactError("invalid previous telemetry state")
    if fields["source"] not in {"NONE", "CGROUP_RATIO", "RSS_ABSOLUTE", "CGROUP_ABSOLUTE"}:
        raise ArtifactError("invalid telemetry source")
    triggers = fields["trigger"].split(",")
    if (not triggers or any(not trigger for trigger in triggers) or
            len(triggers) != len(set(triggers)) or any(trigger not in TRIGGERS for trigger in triggers)):
        raise ArtifactError("invalid or duplicate telemetry trigger")
    return fields


def markers(stderr: str) -> list[dict[str, str]]:
    result = []
    for line in stderr.splitlines():
        if "kv_pressure_telemetry" in line:
            if line.count("kv_pressure_telemetry") != 1:
                raise ArtifactError("duplicate marker token on one line")
            result.append(marker_fields(line))
    return result


def window_markers(case_dir: pathlib.Path, name: str) -> list[dict[str, str]]:
    window = load(case_dir / name)
    start, end = window.get("stderr_start"), window.get("stderr_end")
    raw = (case_dir / "server.stderr").read_bytes()
    if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or
            isinstance(end, bool) or start < 0 or end < start or end > len(raw)):
        raise ArtifactError(f"invalid log window: {name}")
    return markers(raw[start:end].decode(errors="replace"))


def trigger_set(marker: dict[str, str]) -> set[str]:
    return set(marker["trigger"].split(","))


def require_lifecycle_first(sequence: list[dict[str, str]], label: str) -> None:
    first_positions = [index for index, marker in enumerate(sequence) if "first" in trigger_set(marker)]
    if first_positions != [0]:
        raise ArtifactError(f"{label} must have exactly one earliest first trigger")
    first = sequence[0]
    if first["sample_count"] != "1":
        raise ArtifactError(f"{label} first marker must reset sample_count=1")
    if first["source"] != "RSS_ABSOLUTE" or first["config_valid"] != "1" or first["sample_valid"] != "1":
        raise ArtifactError(f"{label} first marker does not prove valid RSS sampling")


def close_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-6)


def finite_positive(value: Any, label: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool) or
            not math.isfinite(value) or value <= 0):
        raise ArtifactError(f"missing or invalid {label}")
    return float(value)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def parse_sse(raw_path: pathlib.Path, evidence_path: pathlib.Path) -> dict[str, Any]:
    try:
        raw = raw_path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"missing raw SSE stream: {raw_path}") from exc
    payloads: list[str] = []
    for line in raw.splitlines():
        if not line:
            continue
        if not line.startswith(b"data: "):
            raise ArtifactError("raw stream contains a malformed SSE line")
        try:
            payloads.append(line[6:].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ArtifactError("SSE payload is not UTF-8") from exc
    evidence = load(evidence_path)
    if not isinstance(evidence, dict) or set(evidence) != {
            "request_started_monotonic_ns", "http_status", "events",
            "stream_ended_monotonic_ns", "clock"}:
        raise ArtifactError("invalid completion timestamp evidence schema")
    if evidence["http_status"] != 200 or evidence["clock"] != "time.monotonic_ns":
        raise ArtifactError("completion timestamp evidence has invalid HTTP status or clock")
    started, ended, records = (evidence["request_started_monotonic_ns"],
                               evidence["stream_ended_monotonic_ns"], evidence["events"])
    if (not isinstance(started, int) or isinstance(started, bool) or not isinstance(ended, int) or
            isinstance(ended, bool) or ended < started or not isinstance(records, list)):
        raise ArtifactError("invalid completion monotonic timestamps")
    if len(payloads) != len(records) or not records:
        raise ArtifactError("raw SSE and timestamp event counts differ")
    arrivals: list[int] = []
    decoded: list[dict[str, Any]] = []
    for payload, record in zip(payloads, records):
        if not isinstance(record, dict) or set(record) != {"raw_payload", "arrival_monotonic_ns"} or record["raw_payload"] != payload:
            raise ArtifactError("raw SSE payload differs from timestamp evidence")
        arrival = record["arrival_monotonic_ns"]
        if (not isinstance(arrival, int) or isinstance(arrival, bool) or arrival < started or
                arrival > ended or (arrivals and arrival < arrivals[-1])):
            raise ArtifactError("missing or invalid SSE arrival timestamp")
        arrivals.append(arrival)
        try:
            event = json.loads(payload)
        except ValueError as exc:
            raise ArtifactError("malformed SSE JSON payload") from exc
        if not isinstance(event, dict) or type(event.get("stop")) is not bool:
            raise ArtifactError("SSE event lacks an exact boolean stop field")
        decoded.append(event)
    terminal_positions = [index for index, event in enumerate(decoded) if event["stop"]]
    if terminal_positions != [len(decoded) - 1]:
        raise ArtifactError("completion must have exactly one final terminal event")
    nonterminal = decoded[:-1]
    if not nonterminal:
        raise ArtifactError("completion has no valid streaming chunks")
    token_groups: list[list[int]] = []
    for event in nonterminal:
        tokens = event.get("tokens")
        if (not isinstance(tokens, list) or not tokens or
                any(not isinstance(token, int) or isinstance(token, bool) for token in tokens) or
                not isinstance(event.get("content"), str)):
            raise ArtifactError("nonterminal SSE chunk has invalid content/tokens")
        token_groups.append(tokens)
    terminal = decoded[-1]
    if not isinstance(terminal.get("content"), str) or not isinstance(terminal.get("stop_type"), str):
        raise ArtifactError("terminal SSE event lacks content or stop_type")
    timings = terminal.get("timings")
    if not isinstance(timings, dict):
        raise ArtifactError("terminal SSE event lacks server_timings")
    predicted_n = timings.get("predicted_n")
    if not isinstance(predicted_n, int) or isinstance(predicted_n, bool) or predicted_n <= 0:
        raise ArtifactError("invalid server_timings.predicted_n")
    valid_token_count = sum(len(group) for group in token_groups)
    if predicted_n != valid_token_count or terminal.get("tokens_predicted") != predicted_n:
        raise ArtifactError("predicted_n, terminal tokens_predicted, and streamed token count differ")
    predicted_ms = finite_positive(timings.get("predicted_ms"), "server_timings.predicted_ms")
    predicted_per_token = finite_positive(
        timings.get("predicted_per_token_ms"), "server_timings.predicted_per_token_ms")
    predicted_per_second = finite_positive(
        timings.get("predicted_per_second"), "server_timings.predicted_per_second")
    tpot = predicted_ms / predicted_n
    throughput = predicted_n * 1000.0 / predicted_ms
    if not close_number(predicted_per_token, tpot) or not close_number(predicted_per_second, throughput):
        raise ArtifactError("server timing fields are internally inconsistent")
    chunk_arrivals = arrivals[:-1]
    ttft = (chunk_arrivals[0] - started) / 1e6
    if ttft <= 0:
        raise ArtifactError("TTFT is not positive")
    intervals = [(right - left) / 1e6 for left, right in zip(chunk_arrivals, chunk_arrivals[1:])]
    if len(intervals) < 2:
        raise ArtifactError("fewer than two SSE chunk interval samples; quantiles are unavailable")
    metrics = {
        "ttft_ms": ttft, "tpot_ms": tpot, "throughput_tps": throughput,
        "sse_chunk_interval_p95_ms": percentile(intervals, 0.95),
        "sse_chunk_interval_p99_ms": percentile(intervals, 0.99),
        "predicted_n": predicted_n, "valid_token_count": valid_token_count,
        "valid_sse_chunk_count": len(nonterminal), "sse_chunk_interval_n": len(intervals),
        "all_chunks_single_token": all(len(group) == 1 for group in token_groups),
        "server_timings": timings,
    }
    if metrics["sse_chunk_interval_p95_ms"] > metrics["sse_chunk_interval_p99_ms"]:
        raise ArtifactError("SSE chunk interval p95 exceeds p99")
    return metrics


def verify_metrics(case_dir: pathlib.Path, stem: str = "completion") -> dict[str, Any]:
    prefix = "" if stem == "completion" else "wake_"
    derived = parse_sse(case_dir / f"{stem}.sse", case_dir / f"{stem}.events.json")
    recorded = load(case_dir / f"{prefix}metrics.json")
    if not isinstance(recorded, dict) or set(recorded) != set(derived):
        raise ArtifactError(f"runner metric schema contradicts raw evidence: {stem}")
    for key, expected in derived.items():
        actual = recorded[key]
        if isinstance(expected, float):
            if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not close_number(float(actual), expected):
                raise ArtifactError(f"runner metric contradicts raw evidence: {key}")
        elif actual != expected:
            raise ArtifactError(f"runner metric contradicts raw evidence: {key}")
    return derived


def exact_plan(manifest: dict[str, Any]) -> None:
    expected = [{"name": n, "kind": k, "round": r, "order": o, "variant": v}
                for n, k, r, o, v in PLAN]
    if manifest.get("planned_cases") != expected:
        raise ArtifactError("planned case matrix differs from frozen protocol")
    if manifest.get("completed_cases") != [item["name"] for item in expected]:
        raise ArtifactError("missing, duplicate, or out-of-order completed case")


def verify_repo(manifest: dict[str, Any]) -> None:
    repo = manifest.get("repo")
    if not isinstance(repo, dict) or repo.get("dirty") is not False or repo.get("status_porcelain") != []:
        raise ArtifactError("manifest does not bind a clean worktree")
    path = pathlib.Path(repo.get("path", ""))
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactError("cannot verify live repository identity") from exc
    if head != repo.get("head") or status:
        raise ArtifactError("HEAD identity mismatch or live worktree dirty")


def verify_post_run(manifest: dict[str, Any]) -> None:
    post = manifest.get("post_run")
    if not isinstance(post, dict) or set(post) != {"repo", "binary", "model", "framework"}:
        raise ArtifactError("missing post-run identity snapshot")
    expected_repo = {"head": manifest["repo"]["head"], "dirty": False, "status_porcelain": []}
    if post.get("repo") != expected_repo or post.get("binary") != manifest.get("binary") or post.get("model") != manifest.get("model"):
        raise ArtifactError("repository, binary, or model identity drifted during the run")
    framework = post.get("framework")
    if not isinstance(framework, dict) or framework != manifest.get("framework"):
        raise ArtifactError("runner or parser identity drifted during the run")


def expected_case_files(spec: dict[str, Any], actual_names: set[str]) -> set[str]:
    expected = set(BASE_CASE_FILES)
    if spec["kind"] == "lifecycle":
        expected.update({
            "sleep_observations.json", "resume_props.json", "wake_metrics.json",
            "wake_completion.sse", "wake_completion.events.json", "wake_completion_window.json",
        })
    if spec["kind"] == "idle_limit":
        expected.update({"idle_window.json", "idle_props.json"})
    if spec["kind"] == "strace":
        expected.add("strace_process.json")
        traces = {name for name in actual_names if re.fullmatch(r"strace\.[0-9]+", name)}
        if not traces:
            raise ArtifactError(f"strace case has no per-pid trace: {spec['name']}")
        expected.update(traces)
    return expected


def verify_schema(root: pathlib.Path, dry_run: bool) -> None:
    root_names = {item.name for item in root.iterdir()}
    allowed_root = {"manifest.json", "inventory.sha256.json", "cases"}
    valid_root_sets = ((allowed_root, allowed_root | {"summary.json"}) if dry_run else
                       (allowed_root, allowed_root | ROOT_OUTPUTS))
    if root_names not in valid_root_sets:
        raise ArtifactError("artifact root has missing or extra files")
    cases = root / "cases"
    if not cases.is_dir():
        raise ArtifactError("artifact cases directory is missing")
    expected_dirs = {item[0] for item in PLAN}
    actual_dirs = {item.name for item in cases.iterdir() if item.is_dir()}
    if actual_dirs != expected_dirs or any(not item.is_dir() for item in cases.iterdir()):
        raise ArtifactError("cases directory set is missing, duplicated, or extra")
    for name, kind, round_no, order, variant in PLAN:
        spec = {"name": name, "kind": kind, "round": round_no, "order": order, "variant": variant}
        case_dir = cases / name
        actual = {item.name for item in case_dir.iterdir() if item.is_file()}
        if any(not item.is_file() for item in case_dir.iterdir()):
            raise ArtifactError(f"case contains an extra directory: {name}")
        if dry_run:
            expected = {"case.json", "execution.json", "environment.json"}
        else:
            expected = expected_case_files(spec, actual)
        if actual != expected:
            raise ArtifactError(f"case file set mismatch: {name}; missing={sorted(expected-actual)} extra={sorted(actual-expected)}")


def verify_phases(case_dir: pathlib.Path, kind: str) -> None:
    phases = load(case_dir / "phases.json")
    if not isinstance(phases, dict) or set(phases) != {"startup", "health", "completion", "sleep", "resume", "shutdown", "case"}:
        raise ArtifactError("phase timeout evidence schema mismatch")
    for name, limit in TIMEOUTS_S.items():
        key = "case" if name == "case_total" else name
        record = phases[key]
        expected_status = "PASS" if key not in {"sleep", "resume"} or kind == "lifecycle" else "NOT_APPLICABLE"
        if not isinstance(record, dict) or record.get("status") != expected_status or record.get("timeout_s") != limit:
            raise ArtifactError(f"phase timeout or failure recorded: {key}")
        if expected_status == "PASS":
            start, end = record.get("started_monotonic_ns"), record.get("ended_monotonic_ns")
            if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or
                    isinstance(end, bool) or end < start or (end - start) > limit * 1e9):
                raise ArtifactError(f"phase exceeded hard timeout or timestamps invalid: {key}")


def normalize_proc_path(path: str) -> str:
    return re.sub(r"^/proc/[0-9]+/", "/proc/self/", path)


def trace_reads(case_dir: pathlib.Path) -> tuple[set[str], int]:
    sampler_paths: set[str] = set()
    other_reads = 0
    for trace in sorted(case_dir.glob("strace.*")):
        fds: dict[str, str] = {}
        for line in text(trace).splitlines():
            opened = re.search(r'openat\([^,]+, "([^"]+)"[^=]*= ([0-9]+)', line)
            if opened:
                fds[opened.group(2)] = normalize_proc_path(opened.group(1))
                continue
            read = re.search(r"read\(([0-9]+)(?:<([^>]+)>)?,", line)
            if read:
                path = normalize_proc_path(read.group(2) or fds.get(read.group(1), ""))
                if (path in {"/proc/self/statm", "/proc/pressure/memory"} or
                        pathlib.PurePosixPath(path).name in {
                            "memory.current", "memory.max", "memory.high", "memory.pressure",
                            "memory.usage_in_bytes", "memory.limit_in_bytes", "memory.soft_limit_in_bytes",
                        }):
                    sampler_paths.add(path)
                else:
                    other_reads += 1
                continue
            closed = re.search(r"close\(([0-9]+)(?:<[^>]+>)?\)", line)
            if closed:
                fds.pop(closed.group(1), None)
    return sampler_paths, other_reads


def verify_strace(case_dir: pathlib.Path, variant: str, server_pid: int) -> dict[str, Any]:
    info = load(case_dir / "strace_process.json")
    argv = info.get("argv") if isinstance(info, dict) else None
    if (not isinstance(argv, list) or "trace=openat,read,close" not in argv or "-p" not in argv or
            info.get("attach_after_health") is not True or info.get("attached_to_pid") != server_pid or
            info.get("shutdown_requested") is not True or info.get("returncode") not in {0, -signal.SIGINT} or
            info.get("residual_process") is not False):
        raise ArtifactError("strace was not an independent bounded attach-after-ready run")
    paths, other_reads = trace_reads(case_dir)
    if variant == "OFF":
        if paths:
            raise ArtifactError("OFF strace observed pressure sampler procfs/cgroup reads")
    else:
        if "/proc/self/statm" not in paths or "/proc/pressure/memory" not in paths:
            raise ArtifactError("ON strace lacks required sampler procfs reads")
        names = {pathlib.PurePosixPath(path).name for path in paths}
        v2 = {"memory.current", "memory.max", "memory.high", "memory.pressure"}
        v1 = {"memory.usage_in_bytes", "memory.limit_in_bytes", "memory.soft_limit_in_bytes"}
        if not (v2 <= names or v1 <= names):
            raise ArtifactError("ON strace lacks a complete cgroup sampler read set")
    return {"sampler_paths": sorted(paths), "other_read_count": other_reads}


def manifest_repo_path(root: pathlib.Path) -> str:
    return str(pathlib.Path(load(root / "manifest.json")["repo"]["path"]).resolve())


def verify_case(root: pathlib.Path, spec: dict[str, Any], binary: pathlib.Path,
                model: pathlib.Path) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any] | None]:
    case_dir = root / "cases" / spec["name"]
    if load(case_dir / "case.json") != spec:
        raise ArtifactError(f"case metadata mismatch: {spec['name']}")
    execution, env, process, result = (load(case_dir / "execution.json"),
                                       load(case_dir / "environment.json"),
                                       load(case_dir / "process.json"), load(case_dir / "result.json"))
    if env != execution.get("env") or env != process.get("env"):
        raise ArtifactError(f"environment identity mismatch: {spec['name']}")
    if process.get("server_argv") != execution.get("argv"):
        raise ArtifactError(f"server argv mismatch: {spec['name']}")
    argv = execution.get("argv")
    if not isinstance(argv, list) or not argv or pathlib.Path(argv[0]).resolve() != binary:
        raise ArtifactError(f"binary argv identity mismatch: {spec['name']}")
    port = execution.get("port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ArtifactError(f"invalid recorded server port: {spec['name']}")
    expected_argv = [
        str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", "1024", "--batch-size", "128", "--ubatch-size", "128",
        "--parallel", "1", "--seed", "1", "--temp", "0", "--no-warmup",
        "--sleep-idle-seconds", "2" if spec["kind"] == "lifecycle" else "-1",
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
    ]
    if (argv != expected_argv or execution.get("launch_argv") != argv or
            execution.get("cwd") != manifest_repo_path(root)):
        raise ArtifactError(f"startup argv/cwd differs from frozen protocol: {spec['name']}")
    if pathlib.Path(process.get("exe", "")).resolve() != binary or pathlib.Path(process.get("expected_exe", "")).resolve() != binary:
        raise ArtifactError(f"live process executable identity mismatch: {spec['name']}")
    if (not isinstance(result, dict) or set(result) != {"returncode", "shutdown_requested", "sigkill_used", "residual_process", "unexpected_exit"} or
            result.get("shutdown_requested") is not True or
            result.get("unexpected_exit") is not False or result.get("residual_process") is not False or
            result.get("returncode") not in {-signal.SIGTERM, -signal.SIGKILL, 0}):
        raise ArtifactError(f"server abnormal exit, timeout, or residual process: {spec['name']}")
    if result["returncode"] == -signal.SIGKILL and result.get("sigkill_used") is not True:
        raise ArtifactError(f"SIGKILL shutdown was not recorded consistently: {spec['name']}")
    verify_phases(case_dir, spec["kind"])
    if load(case_dir / "health.json") != {"status_code": 200, "body": {"status": "ok"}}:
        raise ArtifactError(f"health request failed: {spec['name']}")
    case_metrics = verify_metrics(case_dir)
    combined_log = text(case_dir / "server.stderr") + "\n" + text(case_dir / "server.stdout")
    if FORBIDDEN_LOG.search(combined_log):
        raise ArtifactError(f"structured KV mutation action observed: {spec['name']}")
    for name in ZERO_ENV:
        if env.get(name) != "0":
            raise ArtifactError(f"destructive/prefetch environment not disabled: {name}")
    observed = markers(combined_log)
    initial = window_markers(case_dir, "completion_window.json")
    if spec["variant"] == "OFF":
        expected_off = dict(ON_ENV); expected_off["LLAMA_KV_PRESSURE_SAMPLER"] = "0"
        if any(env.get(key) != value for key, value in expected_off.items()) or observed:
            raise ArtifactError(f"OFF case emitted telemetry: {spec['name']}")
    else:
        expected_env = dict(ON_ENV)
        if spec["kind"] == "idle_limit":
            expected_env.update({"LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "250", "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000"})
        if any(env.get(key) != value for key, value in expected_env.items()):
            raise ArtifactError(f"ON threshold/cadence mismatch: {spec['name']}")
        if spec["kind"] == "lifecycle":
            wake = window_markers(case_dir, "wake_completion_window.json")
            require_lifecycle_first(initial, "startup lifecycle")
            require_lifecycle_first(wake, "resume lifecycle")
            initial_window = load(case_dir / "completion_window.json")
            wake_window = load(case_dir / "wake_completion_window.json")
            if initial_window.get("stderr_end", -1) > wake_window.get("stderr_start", -2):
                raise ArtifactError("startup and resume lifecycle log windows overlap or are out of order")
            global_first = sum("first" in trigger_set(marker) for marker in observed)
            if global_first != 2 or sum("first" in trigger_set(marker) for marker in initial + wake) != 2:
                raise ArtifactError("old logs or an extra process supplied a lifecycle first marker")
            sleep, resume = load(case_dir / "sleep_observations.json"), load(case_dir / "resume_props.json")
            if (not sleep or sleep[-1].get("status_code") != 200 or
                    sleep[-1].get("body", {}).get("is_sleeping") is not True):
                raise ArtifactError("sleep lifecycle was not observed")
            if resume.get("status_code") != 200 or resume.get("body", {}).get("is_sleeping") is not False:
                raise ArtifactError("resume lifecycle was not observed")
            verify_metrics(case_dir, "wake_completion")
        else:
            require_lifecycle_first(initial, f"{spec['name']} lifecycle")
            if sum("first" in trigger_set(marker) for marker in observed) != 1:
                raise ArtifactError(f"duplicate or later first trigger: {spec['name']}")
        if spec["kind"] == "idle_limit":
            window = load(case_dir / "idle_window.json")
            start, end = window.get("stderr_start"), window.get("stderr_end")
            if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or
                    isinstance(end, bool) or start < 0 or end < start or
                    not isinstance(window.get("duration_ms"), (int, float)) or window["duration_ms"] < 1400):
                raise ArtifactError("invalid or too-short idle boundary evidence")
            raw_stderr = (case_dir / "server.stderr").read_bytes()
            if end > len(raw_stderr) or b"kv_pressure_telemetry" in raw_stderr[start:end]:
                raise ArtifactError("250 ms cadence emitted telemetry during pure idle")
            props = load(case_dir / "idle_props.json")
            if props.get("status_code") != 200 or props.get("body", {}).get("is_sleeping") is not False:
                raise ArtifactError("idle limitation observation invalid")
    server_pid = process.get("pid")
    if not isinstance(server_pid, int) or isinstance(server_pid, bool) or server_pid <= 0:
        raise ArtifactError(f"invalid server pid evidence: {spec['name']}")
    strace_result = verify_strace(case_dir, spec["variant"], server_pid) if spec["kind"] == "strace" else None
    return observed, case_metrics, strace_result


def rounded(value: float) -> float:
    return round(value, 6)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRICS:
        result[metric] = {}
        for variant in ("OFF", "ON"):
            values = [row[metric] for row in rows if row["variant"] == variant]
            if len(values) != 3:
                raise ArtifactError(f"performance aggregate does not have n=3: {metric}/{variant}")
            result[metric][variant] = {
                "n": 3, "median": rounded(statistics.median(values)),
                "min": rounded(min(values)), "max": rounded(max(values)),
                "tail_quantiles": "NOT_REPORTED_N_EQ_3_TOO_SMALL_FOR_RELIABLE_TAIL_LATENCY",
            }
    return result


def parse(root: pathlib.Path, dry_run: bool) -> dict[str, Any]:
    verify_inventory(root)
    verify_schema(root, dry_run)
    manifest = load(root / "manifest.json")
    if manifest.get("protocol") != "server_kv_pressure_stage3a_1c" or manifest.get("version") != 2:
        raise ArtifactError("protocol/version mismatch")
    if bool(manifest.get("dry_run")) != dry_run:
        raise ArtifactError("dry-run mode mismatch")
    if manifest.get("pressure_settings_purpose") != PRESSURE_SETTINGS_PURPOSE:
        raise ArtifactError("1/2/3 KiB forced-threshold scope is missing or misleading")
    frozen = manifest.get("frozen")
    if (not isinstance(frozen, dict) or frozen.get("on_env") != ON_ENV or
            frozen.get("timeouts_s") != TIMEOUTS_S):
        raise ArtifactError("frozen environment or timeout contract mismatch")
    verify_repo(manifest)
    verify_post_run(manifest)
    verify_identity(manifest.get("framework", {}).get("runner"), "runner")
    verify_identity(manifest.get("framework", {}).get("parser"), "parser")
    binary = verify_identity(manifest.get("binary"), "binary")
    model = verify_identity(manifest.get("model"), "model")
    exact_plan(manifest)
    if dry_run:
        return {"artifact_status": "DRY_RUN", "protocol": manifest["protocol"],
                "performance_conclusion": "NOT_RUN",
                "pressure_settings_purpose": PRESSURE_SETTINGS_PURPOSE}
    rows: list[dict[str, Any]] = []
    strace_results: dict[str, Any] = {}
    case_intervals: list[tuple[int, int]] = []
    for name, kind, round_no, order, variant in PLAN:
        spec = {"name": name, "kind": kind, "round": round_no, "order": order, "variant": variant}
        _, metrics, trace = verify_case(root, spec, binary, model)
        phase = load(root / "cases" / name / "phases.json")["case"]
        case_intervals.append((phase["started_monotonic_ns"], phase["ended_monotonic_ns"]))
        if kind == "ab":
            rows.append({"round": round_no, "order": order, "variant": variant,
                         **{key: metrics[key] for key in METRICS},
                         "sse_chunk_interval_n": metrics["sse_chunk_interval_n"],
                         "valid_token_count": metrics["valid_token_count"]})
        if trace is not None:
            strace_results[name] = trace
    if any(left[1] > right[0] for left, right in zip(case_intervals, case_intervals[1:])):
        raise ArtifactError("case execution order overlaps or differs from the frozen plan")
    ports = [load(root / "cases" / item[0] / "execution.json")["port"] for item in PLAN]
    if len(ports) != len(set(ports)):
        raise ArtifactError("case ports are not unique within the protocol")
    for round_no in (1, 2, 3):
        pair = [item for item in PLAN if item[1] == "ab" and item[2] == round_no]
        off_name = next(item[0] for item in pair if item[4] == "OFF")
        on_name = next(item[0] for item in pair if item[4] == "ON")
        off_exec = load(root / "cases" / off_name / "execution.json")
        on_exec = load(root / "cases" / on_name / "execution.json")
        off_env, on_env = dict(off_exec["env"]), dict(on_exec["env"])
        if (off_env.pop("LLAMA_KV_PRESSURE_SAMPLER", None) != "0" or
                on_env.pop("LLAMA_KV_PRESSURE_SAMPLER", None) != "1" or off_env != on_env):
            raise ArtifactError(f"round {round_no} OFF/ON environment is not single-switch")
        def normalized_argv(source: dict[str, Any]) -> list[str]:
            argv = list(source["argv"])
            index = argv.index("--port") + 1
            argv[index] = "<dynamic-port>"
            return argv
        if normalized_argv(off_exec) != normalized_argv(on_exec):
            raise ArtifactError(f"round {round_no} OFF/ON startup argv mismatch")
    return {
        "artifact_status": "VALID", "protocol": manifest["protocol"],
        "correctness": "PASS: OFF/ON, independent startup/resume lifecycles, bounded attach strace, read-only structured guards",
        "pressure_settings_purpose": PRESSURE_SETTINGS_PURPOSE,
        "idle_sampling_limit": {
            "status": "OBSERVED_CURRENT_LIMITATION_NOT_FAILURE",
            "observation": "after request-driven first sampling, 1.5 s of continuous idle emitted no periodic telemetry at 250 ms sampling cadence",
            "scope": "known lack of an active idle scheduler tick in this workload; not evidence of continuous monitoring",
        },
        "strace_attribution": strace_results,
        "performance_conclusion": "EXPLORATORY_ONLY_NO_FORMAL_BENEFIT_CLAIM",
        "performance_scope": "three runs per variant (n=3); no aggregate tail p95/p99 claim",
        "performance_runs": rows, "performance_aggregate": summarize(rows),
    }


def write_outputs(root: pathlib.Path, summary: dict[str, Any]) -> None:
    write_json(root / "summary.json", summary)
    rows = summary.get("performance_runs", [])
    performance = root / "performance.tsv"
    if rows:
        fields = ("round", "order", "variant", *METRICS, "sse_chunk_interval_n", "valid_token_count")
        with performance.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
    elif performance.exists():
        performance.unlink()
    write_inventory(root)
    verify_inventory(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("artifact", type=pathlib.Path)
    args = parser.parse_args()
    root = args.artifact.resolve()
    try:
        summary = parse(root, args.dry_run)
        write_outputs(root, summary)
    except ArtifactError as exc:
        fail(str(exc))
    print(f"PASS: {summary['artifact_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
