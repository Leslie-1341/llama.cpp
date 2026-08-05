#!/usr/bin/env python3
"""Real-server runner for the G0-S1 single-session KV roundtrip gate."""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "kv_governor_g0_s1"
PROTOCOL_VERSION = 1
SOURCE_MARKER_SCHEMA = "kv_governor_stage3c_1c_2b_1r/v6"
CAPABILITY_MARKER = "KV_GOVERNOR_CAPABILITY"
MARKER = "kv_pressure_unified_action"
RESIDENT_OBSERVATION_MARKER = "kv_g0_s1_resident_observation"
RESUME_TIMING_MARKER = "kv_resume_stage_timing"
PREFETCH_BLOCK_PHASE_MARKER = "KV_PAGED_PREFETCH_BLOCK_PHASE"
PREFETCH_PHASE_CALL_MARKER = "KV_PAGED_PREFETCH_PHASE_CALL"
IO_STATS_MARKER = "KV_PAGED_IO_STATS"
CAPABILITY_FIELDS = {
    "n_slots", "n_seq_max", "n_stream", "kv_unified", "paged_metadata",
    "ingraph_gather", "release_supported", "offload_supported", "prefetch_supported",
    "backing_ready", "swap_explicit_only",
}
REQUIRED_CAPABILITY = {
    "kv_unified", "paged_metadata", "ingraph_gather", "release_supported",
    "offload_supported", "prefetch_supported", "backing_ready", "swap_explicit_only",
}
CLAIMANT_FIELDS = {
    "epoch", "exhausted", "valid", "target_blocks", "eligible_resident_blocks",
    "swapped_blocks", "shared_blocks", "blocked_blocks",
}
RESIDENT_FIELDS = {
    "status", "source", "object_id", "generation", "page_size", "total_bytes",
    "resident_bytes", "total_pages", "resident_pages",
}

CTX_SIZE = 2048
SUPPORTED_CTX_SIZES = (1024, CTX_SIZE, 4096, 8064)
PAGED_BLOCK_SIZE = 64
MIN_PREFIX_BLOCKS = 2
MIN_PREFIX_TOKENS = PAGED_BLOCK_SIZE * MIN_PREFIX_BLOCKS
N_PREDICT = 32
SEED = 1
TEMPERATURE = 0.0
GOVERNOR_TARGET_BYTES = 1 << 30
GOVERNOR_MAX_BLOCKS = 64
HEALTH_TIMEOUT_SECONDS = 60.0
CAPABILITY_TIMEOUT_SECONDS = 30.0
MARKER_TIMEOUT_SECONDS = 60.0
REQUEST_TIMEOUT_SECONDS = 180.0
SERVER_TERM_TIMEOUT_SECONDS = 10.0
SERVER_KILL_TIMEOUT_SECONDS = 5.0
BASE_ENV = {"HOME": "/tmp", "LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "")}
PREFIX_TEXT_UNIT = "G0 S1 deterministic scalable prefix token block.\n"
MAX_PREFIX_TEXT_UNITS = 2048
PREFIX_TEXT = PREFIX_TEXT_UNIT * MAX_PREFIX_TEXT_UNITS
QUERY_TEXT = "Continue with one deterministic concise answer."


class RunnerInterrupted(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


class WorkloadFailure(Exception):
    pass


def dump(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha(path)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()


def tracked_diff_fingerprint() -> str:
    return sha_bytes(subprocess.check_output(["git", "-C", str(ROOT), "diff", "--no-ext-diff", "--binary", "HEAD"]))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fields_after_token(line: str, token: str) -> dict[str, str] | None:
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


def token_records(raw: bytes, token: str, start: int = 0, end: int | None = None) -> list[dict[str, Any]]:
    limit = len(raw) if end is None else end
    if start < 0 or limit < start or limit > len(raw):
        raise WorkloadFailure(f"invalid {token} byte scope")
    records: list[dict[str, Any]] = []
    offset = start
    for line in raw[start:limit].splitlines(keepends=True):
        line_end = offset + len(line)
        if token.encode("utf-8") in line:
            fields = fields_after_token(line.decode("utf-8", errors="replace"), token)
            if fields is None:
                raise WorkloadFailure(f"malformed {token} record")
            records.append({"offset": offset, "end": line_end, "fields": fields})
        offset = line_end
    return records


def unsigned_fields(fields: dict[str, str], required: set[str], label: str) -> dict[str, int]:
    if set(fields) != required:
        raise WorkloadFailure(
            f"{label} schema mismatch missing={sorted(required-set(fields))} "
            f"extra={sorted(set(fields)-required)}")
    values: dict[str, int] = {}
    for key, value in fields.items():
        if not value.isdigit():
            raise WorkloadFailure(f"{label}.{key} is not an unsigned integer")
        values[key] = int(value)
    return values


def capture_resume_timing(case: pathlib.Path, step2: dict[str, Any]) -> dict[str, Any]:
    scope = json.loads((case / "resume_scope.json").read_text(encoding="utf-8"))
    start, end = scope.get("start"), scope.get("end")
    if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool):
        raise WorkloadFailure("resume timing scope is malformed")
    raw = (case / "server.stderr").read_bytes()

    stage_records = token_records(raw, RESUME_TIMING_MARKER, start, end)
    if len(stage_records) != 1:
        raise WorkloadFailure(f"expected one {RESUME_TIMING_MARKER} record, got {len(stage_records)}")
    stage = unsigned_fields(stage_records[0]["fields"], {
        "decision_id", "seq_id", "transaction_id", "restored_blocks", "restored_bytes",
        "queue_us", "gate_us", "graph_us", "total_us",
    }, RESUME_TIMING_MARKER)

    call_records = token_records(raw, PREFETCH_PHASE_CALL_MARKER, start, end)
    if len(call_records) != 1:
        raise WorkloadFailure(f"expected one {PREFETCH_PHASE_CALL_MARKER} record, got {len(call_records)}")
    prefetch_call = unsigned_fields(call_records[0]["fields"], {
        "call", "seq_id", "requested_blocks", "restored_blocks", "phase_events",
    }, PREFETCH_PHASE_CALL_MARKER)

    block_records = token_records(raw, PREFETCH_BLOCK_PHASE_MARKER, start, end)
    block_phases = [unsigned_fields(record["fields"], {
        "call", "block_index", "physical_block", "validate_us", "read_us", "unpack_us",
        "commit_us", "phase_sum_us",
    }, PREFETCH_BLOCK_PHASE_MARKER) for record in block_records]
    if prefetch_call["phase_events"] != len(block_phases):
        raise WorkloadFailure("prefetch phase-event count does not match block records")
    if prefetch_call["restored_blocks"] != stage["restored_blocks"]:
        raise WorkloadFailure("prefetch call and server timing restored-block counts differ")
    for index, block in enumerate(block_phases):
        if block["call"] != prefetch_call["call"] or block["block_index"] != index:
            raise WorkloadFailure("prefetch block phase order/correlation is invalid")
        if block["phase_sum_us"] != block["validate_us"] + block["read_us"] + block["unpack_us"] + block["commit_us"]:
            raise WorkloadFailure("prefetch block phase sum is inconsistent")

    io_records = token_records(raw, IO_STATS_MARKER)
    if not io_records:
        raise WorkloadFailure(f"missing {IO_STATS_MARKER} record")
    io_keys = {
        "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls", "backing_write_syscalls",
        "bytes_read", "bytes_written", "avg_block_swap_in_latency_us", "max_block_swap_in_latency_us",
        "block_in_validate_us", "block_in_read_us", "block_in_unpack_us", "block_in_commit_us",
    }
    io_candidates: list[tuple[dict[str, int], int]] = []
    for record in io_records:
        io_fields = record["fields"]
        if not io_keys <= set(io_fields):
            raise WorkloadFailure(f"{IO_STATS_MARKER} is missing required fields")
        io_stats = {key: int(io_fields[key]) for key in sorted(io_keys) if io_fields[key].isdigit()}
        if set(io_stats) != io_keys:
            raise WorkloadFailure(f"{IO_STATS_MARKER} contains non-integer required fields")
        io_candidates.append((io_stats, record["offset"]))
    io_stats, _io_offset = max(io_candidates, key=lambda item: (
        item[0]["bytes_read"], item[0]["bytes_written"], item[0]["block_swap_in_calls"], item[1]))

    response_timings = step2.get("response_timings")
    timing_keys = {"prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_token_ms"}
    if not isinstance(response_timings, dict) or not timing_keys <= set(response_timings):
        raise WorkloadFailure("step2 response timings are unavailable")
    if any(not isinstance(response_timings[key], (int, float)) or isinstance(response_timings[key], bool)
           for key in timing_keys):
        raise WorkloadFailure("step2 response timings are malformed")

    read_us = sum(block["read_us"] for block in block_phases)
    restore_us = sum(block["validate_us"] + block["unpack_us"] for block in block_phases)
    commit_us = sum(block["commit_us"] for block in block_phases)
    phase_us = read_us + restore_us + commit_us
    explained_us = stage["queue_us"] + phase_us + stage["graph_us"]
    derived = {
        "server_ttft_ms": stage["total_us"] / 1000.0,
        "server_prompt_ms": float(response_timings["prompt_ms"]),
        "tpot_ms": float(response_timings["predicted_per_token_ms"]),
        "queue_us": stage["queue_us"],
        "read_us": read_us,
        "restore_us": restore_us,
        "commit_us": commit_us,
        "graph_us": stage["graph_us"],
        "gate_us": stage["gate_us"],
        "gate_overhead_us": max(0, stage["gate_us"] - phase_us),
        "residual_us": max(0, stage["total_us"] - explained_us),
        "io_bytes": stage["restored_bytes"],
        "io_syscalls": io_stats["backing_read_syscalls"],
        "io_service_us": read_us,
    }
    value = {
        "schema_version": 1,
        "scope": scope,
        "stage": stage,
        "prefetch_call": prefetch_call,
        "block_phases": block_phases,
        "io_stats_record_count": len(io_records),
        "io_stats": io_stats,
        "response_timings": {key: response_timings[key] for key in sorted(timing_keys)},
        "derived": derived,
    }
    dump(case / "timing.json", value)
    return value


def server_argv(
        binary: pathlib.Path,
        model: pathlib.Path,
        port: int,
        ctx_size: int = CTX_SIZE) -> list[str]:
    return [
        str(binary), "--host", "127.0.0.1", "--port", str(port), "--model", str(model),
        "--ctx-size", str(ctx_size), "--parallel", "1", "--kv-unified", "--no-cache-idle-slots",
        "--timeout", "300", "--threads", "4", "--n-gpu-layers", "0", "--cache-type-k", "f32",
        "--cache-type-v", "f32", "--no-warmup",
    ]


def wait_health(port: int, proc: subprocess.Popen[bytes]) -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline and proc.poll() is None:
        con: http.client.HTTPConnection | None = None
        try:
            con = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            con.request("GET", "/health")
            response = con.getresponse()
            response.read()
            if response.status == 200:
                return True
        except OSError:
            pass
        finally:
            if con is not None:
                try:
                    con.close()
                except OSError:
                    pass
        time.sleep(0.2)
    return False


def wait_capability(case: pathlib.Path, proc: subprocess.Popen[bytes]) -> tuple[dict[str, str] | None, dict[str, Any] | None, str | None]:
    deadline = time.monotonic() + CAPABILITY_TIMEOUT_SECONDS
    stderr = case / "server.stderr"
    while time.monotonic() < deadline:
        raw = stderr.read_bytes() if stderr.is_file() else b""
        records: list[tuple[dict[str, str] | None, int, int]] = []
        offset = 0
        for line in raw.splitlines(keepends=True):
            end = offset + len(line)
            if CAPABILITY_MARKER.encode("utf-8") in line:
                records.append((fields_after_token(line.decode("utf-8", errors="replace"), CAPABILITY_MARKER), offset, end))
            offset = end
        if len(records) > 1:
            return None, None, "duplicate production capability records"
        if len(records) == 1:
            fields, start, end = records[0]
            if fields is None or set(fields) != CAPABILITY_FIELDS:
                return None, None, "malformed production capability record"
            return fields, {"offset": start, "end": end, "fields": fields}, None
        if proc.poll() is not None:
            return None, None, f"server exited before capability marker (exit={proc.returncode})"
        time.sleep(0.1)
    return None, None, "timed out waiting for production capability marker"


def query_slots_raw(port: int) -> tuple[int, bytes, list[dict[str, Any]] | None]:
    con: http.client.HTTPConnection | None = None
    try:
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        con.request("GET", "/slots")
        response = con.getresponse()
        raw = response.read()
        value = json.loads(raw) if response.status == 200 and raw else None
        return response.status, raw, value if isinstance(value, list) else None
    except (OSError, json.JSONDecodeError):
        return 0, b"", None
    finally:
        if con is not None:
            try:
                con.close()
            except OSError:
                pass


def read_process_identity(pid: int) -> dict[str, Any] | None:
    try:
        stat_text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat_text.rsplit(")", 1)[1].split()
        starttime = int(tail[19])
        cmdline = [part.decode("utf-8", errors="replace") for part in pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
        if starttime <= 0 or not cmdline:
            return None
        return {
            "pid": pid,
            "starttime_ticks": starttime,
            "cmdline": cmdline,
            "cmdline_sha256": sha_bytes(b"\0".join(item.encode("utf-8") for item in cmdline)),
        }
    except (IndexError, OSError, ValueError):
        return None


def prepare_backing(case: pathlib.Path) -> pathlib.Path:
    backing = case / "backing"
    backing.mkdir()
    return backing


def cleanup_backing(case: pathlib.Path, backing: pathlib.Path) -> dict[str, Any]:
    error: str | None = None
    try:
        shutil.rmtree(backing)
    except FileNotFoundError:
        pass
    except OSError as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "path": str(backing.resolve()),
        "environment_value": "backing",
        "created": True,
        "cleanup_attempted": True,
        "exists_after_cleanup": backing.exists(),
        "cleanup_error": error,
    }


def stop(proc: subprocess.Popen[bytes]) -> dict[str, Any]:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = proc.pid

    def group_exists() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False

    term_timed_out = False
    kill_timed_out = False
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=SERVER_TERM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        term_timed_out = True
    if group_exists():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=SERVER_KILL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        deadline = time.monotonic() + SERVER_KILL_TIMEOUT_SECONDS
        while group_exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        kill_timed_out = group_exists()
    for handle in (proc.stdout, proc.stderr):
        if handle and not handle.closed:
            handle.close()
    return {
        "pid": proc.pid,
        "pgid": pgid,
        "exit_code": proc.returncode,
        "term_timed_out": term_timed_out,
        "kill_timed_out": kill_timed_out,
        "residual_process": group_exists(),
    }


def governor_env(enabled: bool) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update({
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_INGRAPH": "1",
        "LLAMA_KV_PAGED_SWAP": "1",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
        "LLAMA_KV_PAGED_MINCORE": "1",
        "LLAMA_KV_PAGED_BLOCK_SIZE": str(PAGED_BLOCK_SIZE),
        "LLAMA_KV_SWAP_DIR": "backing",
        "LLAMA_KV_PRESSURE_SAMPLER": "1",
        "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "1",
        "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "1",
        "LLAMA_KV_PAGED_IO_STATS": "1",
        "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "1",
        "LLAMA_KV_RESUME_STAGE_TIMING": "1",
        "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "100",
        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
        "LLAMA_KV_LOW_WATER_RSS_KB": "1",
        "LLAMA_KV_PRESSURE_RSS_KB": "2",
        "LLAMA_KV_CRITICAL_RSS_KB": "3",
    })
    if enabled:
        env.update({
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": str(GOVERNOR_TARGET_BYTES),
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": str(GOVERNOR_MAX_BLOCKS),
        })
    return env


def start(
        case: pathlib.Path,
        argv: list[str],
        env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        argv,
        cwd=case,
        stdout=(case / "server.stdout").open("wb"),
        stderr=(case / "server.stderr").open("wb"),
        env=env,
        preexec_fn=os.setsid,
    )


def normalize_claimant(value: Any, slot_id: int, active: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != CLAIMANT_FIELDS or not isinstance(active, bool):
        return None
    epoch = value.get("epoch")
    counts = ("target_blocks", "eligible_resident_blocks", "swapped_blocks", "shared_blocks", "blocked_blocks")
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 or
            not isinstance(value.get("exhausted"), bool) or not isinstance(value.get("valid"), bool) or
            any(not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < 0 for key in counts)):
        return None
    if value["target_blocks"] != sum(value[key] for key in counts if key != "target_blocks"):
        return None
    return {
        "seq_id": slot_id,
        "epoch": epoch,
        "active": active,
        "exhausted": value["exhausted"],
        "valid": value["valid"],
        "target_blocks": value["target_blocks"],
        "eligible_resident_blocks": value["eligible_resident_blocks"],
        "swapped_blocks": value["swapped_blocks"],
        "shared_blocks": value["shared_blocks"],
        "blocked_blocks": value["blocked_blocks"],
    }


def normalize_resident(value: Any) -> tuple[dict[str, Any] | None, str]:
    if isinstance(value, dict) and value == {"status": "unavailable"}:
        return None, "server declared KV physical resident observation unavailable"
    if not isinstance(value, dict) or set(value) != RESIDENT_FIELDS:
        return None, "server does not expose the G0-S1 KV physical resident observation"
    if value.get("status") != "available" or value.get("source") != "paged_sample_mincore":
        return None, "server returned an invalid KV physical resident observation status"
    numeric = ("object_id", "generation", "page_size", "total_bytes", "resident_bytes", "total_pages", "resident_pages")
    if any(not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < 0 for key in numeric):
        return None, "server returned a malformed KV physical resident observation"
    if (value["object_id"] == 0 or value["generation"] == 0 or value["page_size"] == 0 or
            value["total_bytes"] == 0 or value["total_pages"] == 0 or
            value["resident_bytes"] > value["total_bytes"] or value["resident_pages"] > value["total_pages"] or
            value["total_bytes"] != value["total_pages"] * value["page_size"] or
            value["resident_bytes"] != value["resident_pages"] * value["page_size"]):
        return None, "server returned an inconsistent KV physical resident observation"
    return dict(value), ""


def physical_probe(slots: list[dict[str, Any]] | None, raw: bytes, identity_record: dict[str, Any] | None) -> dict[str, Any]:
    logical_fields = {
        "epoch", "exhausted", "valid", "target_blocks", "eligible_resident_blocks",
        "swapped_blocks", "shared_blocks", "blocked_blocks",
    }
    claimant_keys: list[str] = []
    resident: dict[str, Any] | None = None
    reason = "invalid /slots physical-observation probe"
    if isinstance(slots, list) and len(slots) == 1 and isinstance(slots[0], dict):
        claimant = slots[0].get("kv_claimant")
        if isinstance(claimant, dict):
            claimant_keys = sorted(str(key) for key in claimant)
        resident, reason = normalize_resident(slots[0].get("kv_resident"))
    return {
        "source": "GET /slots.kv_resident",
        "server_pid": identity_record.get("pid") if identity_record else None,
        "slots_raw_sha256": sha_bytes(raw),
        "slots_shape": len(slots) if isinstance(slots, list) else None,
        "claimant_keys": claimant_keys,
        "logical_claimant_fields_present": logical_fields <= set(claimant_keys),
        "physical_resident_sample_available": resident is not None,
        "resident": resident,
        "reason": reason,
    }


def capture_snapshot(case: pathlib.Path, port: int, label: str) -> dict[str, Any]:
    status, raw, slots = query_slots_raw(port)
    observed = time.monotonic_ns()
    stderr = case / "server.stderr"
    stderr_end = stderr.stat().st_size if stderr.is_file() else 0
    raw_path = case / f"{label}.raw.json"
    raw_path.write_bytes(raw)
    if status != 200 or not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict):
        raise WorkloadFailure(f"{label}: GET /slots did not return exactly one slot")
    slot = slots[0]
    if slot.get("id") != 0 or not isinstance(slot.get("is_processing"), bool):
        raise WorkloadFailure(f"{label}: GET /slots has an invalid slot identity")
    claimant = normalize_claimant(slot.get("kv_claimant"), 0, slot["is_processing"])
    if claimant is None:
        raise WorkloadFailure(f"{label}: GET /slots has an invalid KV claimant")
    value = {
        "raw_path": raw_path.name,
        "raw_sha256": sha_bytes(raw),
        "observed_monotonic_ns": observed,
        "stderr_end": stderr_end,
        "claimant": claimant,
    }
    dump(case / f"{label}.json", value)
    return value


def marker_records(case: pathlib.Path, start: int = 0) -> tuple[list[tuple[dict[str, str], int, int]], int]:
    raw = (case / "server.stderr").read_bytes()
    records: list[tuple[dict[str, str], int, int]] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        end = offset + len(line)
        if offset >= start and MARKER.encode("utf-8") in line:
            fields = fields_after_token(line.decode("utf-8", errors="replace"), MARKER)
            if fields is not None:
                records.append((fields, offset, end))
        offset = end
    return records, len(raw)


def resident_observation_records(
        case: pathlib.Path) -> list[tuple[dict[str, str] | None, int, int]]:
    raw = (case / "server.stderr").read_bytes()
    records: list[tuple[dict[str, str] | None, int, int]] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        end = offset + len(line)
        if RESIDENT_OBSERVATION_MARKER.encode("utf-8") in line:
            fields = fields_after_token(
                line.decode("utf-8", errors="replace"), RESIDENT_OBSERVATION_MARKER)
            records.append((fields, offset, end))
        offset = end
    return records


def capture_resident_observation(
        case: pathlib.Path,
        offload: dict[str, str]) -> dict[str, Any]:
    matches = [
        record for record in resident_observation_records(case)
        if record[0] is not None and
        record[0].get("decision_id") == offload.get("decision_id") and
        record[0].get("seq_id") == offload.get("selected_seq_id") and
        record[0].get("transaction_id") == offload.get("transaction_id")
    ]
    if len(matches) != 1:
        raise WorkloadFailure(
            "OFFLOAD transaction does not have exactly one resident observation")
    fields, start, end = matches[0]
    assert fields is not None
    return {"offset": start, "end": end, "fields": fields}


def marker_is_changed_offload(fields: dict[str, str]) -> bool:
    try:
        return (
            fields.get("offload_attempted") == "1" and
            fields.get("state_changed") == "1" and
            int(fields.get("transaction_id", "0")) > 0)
    except ValueError:
        return False


def resident_sample(fields: dict[str, str], prefix: str) -> dict[str, int]:
    if fields.get(f"{prefix}_available") != "1":
        raise WorkloadFailure(
            f"OFFLOAD {prefix} resident observation is unavailable")
    keys = (
        "object_id", "generation", "page_size", "total_bytes", "resident_bytes",
        "total_pages", "resident_pages",
    )
    try:
        sample = {key: int(fields[f"{prefix}_{key}"]) for key in keys}
    except (KeyError, ValueError) as exc:
        raise WorkloadFailure(
            f"OFFLOAD {prefix} resident observation is malformed") from exc
    if (
            sample["object_id"] <= 0 or sample["generation"] <= 0 or
            sample["page_size"] <= 0 or sample["total_bytes"] <= 0 or
            sample["total_pages"] <= 0 or
            sample["resident_bytes"] > sample["total_bytes"] or
            sample["resident_pages"] > sample["total_pages"] or
            sample["total_bytes"] != sample["total_pages"] * sample["page_size"] or
            sample["resident_bytes"] != sample["resident_pages"] * sample["page_size"]):
        raise WorkloadFailure(
            f"OFFLOAD {prefix} resident observation accounting is inconsistent")
    return sample


def offload_collection_record(
        scope_start: int,
        expected_blocks: int,
        selected_seq_id: int,
        selected_claimant_epoch: int,
        transactions: list[dict[str, Any]],
        status: str) -> dict[str, Any]:
    first_resident = int(
        transactions[0]["resident"]["fields"]["before_resident_bytes"])
    last_resident = int(
        transactions[-1]["resident"]["fields"]["after_resident_bytes"])
    cumulative = {
        "transaction_count": len(transactions),
        "blocks": sum(int(item["fields"]["blocks"]) for item in transactions),
        "bytes": sum(int(item["fields"]["bytes"]) for item in transactions),
        "relieved_bytes": sum(
            int(item["fields"]["relieved_bytes"]) for item in transactions),
        "first_resident_bytes": first_resident,
        "last_resident_bytes": last_resident,
        "resident_drop_bytes": first_resident - last_resident,
    }
    if cumulative["resident_drop_bytes"] != cumulative["relieved_bytes"]:
        raise WorkloadFailure(
            "cumulative OFFLOAD resident drop does not equal cumulative relieved_bytes")
    return {
        "status": status,
        "scope_start": scope_start,
        "scope_end": transactions[-1]["end"],
        "expected_blocks": expected_blocks,
        "selected_seq_id": selected_seq_id,
        "selected_claimant_epoch": selected_claimant_epoch,
        "transactions": transactions,
        "cumulative": cumulative,
    }


def collect_offload_transactions(
        case: pathlib.Path,
        proc: subprocess.Popen[bytes],
        port: int,
        start: int,
        expected_blocks: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if expected_blocks <= 0:
        raise WorkloadFailure("OFFLOAD expected block count must be positive")
    deadline = time.monotonic() + MARKER_TIMEOUT_SECONDS
    seen_offsets: set[tuple[int, int]] = set()
    transaction_ids: set[int] = set()
    transactions: list[dict[str, Any]] = []
    selected_seq_id: int | None = None
    selected_claimant_epoch: int | None = None
    cumulative_blocks = 0
    previous_after: dict[str, int] | None = None

    while time.monotonic() < deadline:
        records, _ = marker_records(case, start)
        for fields, begin, end in records:
            marker_span = (begin, end)
            if marker_span in seen_offsets:
                continue
            seen_offsets.add(marker_span)
            if not marker_is_changed_offload(fields):
                continue
            try:
                seq_id = int(fields["selected_seq_id"])
                epoch = int(fields["selected_claimant_epoch"])
                transaction_id = int(fields["transaction_id"])
                blocks = int(fields["blocks"])
                byte_count = int(fields["bytes"])
                relieved_bytes = int(fields["relieved_bytes"])
            except (KeyError, ValueError) as exc:
                raise WorkloadFailure(
                    "state-changing OFFLOAD marker has malformed numeric fields") from exc
            if (
                    seq_id < 0 or epoch <= 0 or transaction_id <= 0 or blocks <= 0 or
                    byte_count <= 0 or relieved_bytes <= 0 or
                    fields.get("outcome") != "completed" or
                    fields.get("release_attempted") != "0" or
                    fields.get("io_failure") != "0" or fields.get("idle") != "1"):
                raise WorkloadFailure(
                    "state-changing OFFLOAD marker violates the transaction contract")
            if selected_seq_id is None:
                selected_seq_id = seq_id
                selected_claimant_epoch = epoch
            elif seq_id != selected_seq_id or epoch != selected_claimant_epoch:
                raise WorkloadFailure(
                    "state-changing OFFLOAD changed seq_id or claimant_epoch before closure")
            if transaction_id in transaction_ids:
                raise WorkloadFailure("duplicate state-changing OFFLOAD transaction_id")
            if cumulative_blocks + blocks > expected_blocks:
                raise WorkloadFailure(
                    "cumulative OFFLOAD blocks exceed the complete step1 workload")

            resident = capture_resident_observation(case, fields)
            resident_fields = resident["fields"]
            if resident_fields.get("server_pid") != str(proc.pid):
                raise WorkloadFailure(
                    "OFFLOAD resident observation does not bind the server PID")
            before = resident_sample(resident_fields, "before")
            after = resident_sample(resident_fields, "after")
            identity_keys = (
                "object_id", "generation", "page_size", "total_bytes", "total_pages")
            if any(before[key] != after[key] for key in identity_keys):
                raise WorkloadFailure(
                    "OFFLOAD resident observation changed KV object or generation")
            if previous_after is not None and before != previous_after:
                raise WorkloadFailure(
                    "ordered OFFLOAD resident observations are not contiguous")
            resident_drop_bytes = before["resident_bytes"] - after["resident_bytes"]
            if resident_drop_bytes <= 0 or resident_drop_bytes != relieved_bytes:
                raise WorkloadFailure(
                    "OFFLOAD resident drop does not equal relieved_bytes")

            transaction_ids.add(transaction_id)
            cumulative_blocks += blocks
            transactions.append({
                "index": len(transactions),
                "offset": begin,
                "end": end,
                "fields": fields,
                "resident": resident,
                "resident_drop_bytes": resident_drop_bytes,
            })
            previous_after = after
            assert selected_seq_id is not None and selected_claimant_epoch is not None
            evidence = offload_collection_record(
                start, expected_blocks, selected_seq_id, selected_claimant_epoch,
                transactions, "blocks_complete" if cumulative_blocks == expected_blocks else "collecting")
            dump(case / "offload.json", evidence)

            if cumulative_blocks == expected_blocks:
                post = capture_snapshot(case, port, "post_claimant")
                claimant = post["claimant"]
                if (
                        claimant["seq_id"] != selected_seq_id or
                        claimant["epoch"] != selected_claimant_epoch or
                        claimant["active"] or claimant["exhausted"] or not claimant["valid"] or
                        claimant["target_blocks"] != expected_blocks or
                        claimant["eligible_resident_blocks"] != 0 or
                        claimant["swapped_blocks"] != expected_blocks or
                        claimant["shared_blocks"] != 0 or claimant["blocked_blocks"] != 0):
                    raise WorkloadFailure(
                        "final claimant does not close the complete OFFLOAD workload")
                evidence["status"] = "complete"
                dump(case / "offload.json", evidence)
                return evidence, post

        if proc.poll() is not None:
            raise WorkloadFailure(
                f"server exited while waiting for cumulative OFFLOAD (exit={proc.returncode})")
        time.sleep(0.05)
    raise WorkloadFailure(
        f"timed out waiting for cumulative OFFLOAD blocks "
        f"({cumulative_blocks}/{expected_blocks})")


def tokenize(port: int, content: str, add_special: bool) -> list[int]:
    payload = json.dumps({"content": content, "add_special": add_special}, separators=(",", ":")).encode("utf-8")
    con: http.client.HTTPConnection | None = None
    status = 0
    value: Any = {}
    try:
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        con.request("POST", "/tokenize", payload, {"Content-Type": "application/json"})
        response = con.getresponse()
        status = response.status
        raw = response.read()
        value = json.loads(raw) if raw else {}
    finally:
        if con is not None:
            con.close()
    tokens = value.get("tokens") if status == 200 and isinstance(value, dict) else None
    if not isinstance(tokens, list) or not tokens or any(not isinstance(token, int) or isinstance(token, bool) for token in tokens):
        raise WorkloadFailure(f"tokenization failed (status={status})")
    return tokens


def token_ids_sha256(tokens: list[int]) -> str:
    return sha_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))


def common_token_prefix_count(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def aligned_prefix_token_count(
        prefix_candidate: list[int],
        continuation_prompt: list[int],
        target_prefix_tokens: int) -> int:
    common_prefix = common_token_prefix_count(prefix_candidate, continuation_prompt)
    shared_boundary_limit = min(common_prefix, max(0, len(continuation_prompt) - 1))
    boundary_limit = min(shared_boundary_limit, target_prefix_tokens)
    return boundary_limit // PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE


def target_prefix_record(
        target_prefix_tokens: int,
        prompt_p: list[int],
        continuation_prompt: list[int]) -> dict[str, Any]:
    target_is_valid = (
        isinstance(target_prefix_tokens, int) and
        not isinstance(target_prefix_tokens, bool) and
        target_prefix_tokens >= MIN_PREFIX_TOKENS)
    return {
        "target_prefix_tokens": target_prefix_tokens,
        "actual_p_token_count": len(prompt_p),
        "actual_pq_token_count": len(continuation_prompt),
        "actual_p_block_count": len(prompt_p) // PAGED_BLOCK_SIZE,
        "target_prefix_token_delta": (
            target_prefix_tokens - len(prompt_p) if target_is_valid else None),
    }


def build_token_workload(
        prefix_candidate: list[int],
        continuation_prompt: list[int],
        ctx_size: int,
        target_prefix_tokens: int) -> tuple[list[int], dict[str, Any]]:
    target_is_valid = (
        isinstance(target_prefix_tokens, int) and
        not isinstance(target_prefix_tokens, bool) and
        target_prefix_tokens >= MIN_PREFIX_TOKENS)
    ctx_is_valid = (
        isinstance(ctx_size, int) and not isinstance(ctx_size, bool) and ctx_size > 0)
    common_prefix = common_token_prefix_count(prefix_candidate, continuation_prompt)
    prefix_token_count = (
        aligned_prefix_token_count(
            prefix_candidate, continuation_prompt, target_prefix_tokens)
        if target_is_valid else 0)
    prompt_p = prefix_candidate[:prefix_token_count]
    strict_prefix = (
        bool(prompt_p) and len(prompt_p) < len(continuation_prompt) and
        continuation_prompt[:len(prompt_p)] == prompt_p)
    context_token_count = len(continuation_prompt) + N_PREDICT
    first_mismatch = (
        common_prefix
        if common_prefix < len(prefix_candidate) and common_prefix < len(continuation_prompt)
        else None)
    mismatch_tokens = (
        {
            "prefix_candidate": prefix_candidate[first_mismatch],
            "continuation_prompt": continuation_prompt[first_mismatch],
        }
        if first_mismatch is not None else None)
    selection = target_prefix_record(target_prefix_tokens, prompt_p, continuation_prompt)
    target_delta = selection["target_prefix_token_delta"]
    failure_reasons: list[str] = []
    if not target_is_valid:
        failure_reasons.append(
            f"target_prefix_tokens must be an integer at least {MIN_PREFIX_TOKENS}")
    if not ctx_is_valid:
        failure_reasons.append("ctx_size must be a positive integer")
    if len(prompt_p) < MIN_PREFIX_TOKENS:
        failure_reasons.append(
            f"selected P has {len(prompt_p)} tokens ({len(prompt_p) // PAGED_BLOCK_SIZE} full "
            f"{PAGED_BLOCK_SIZE}-token blocks); requires at least {MIN_PREFIX_TOKENS} tokens "
            f"({MIN_PREFIX_BLOCKS} blocks); target_prefix_tokens={target_prefix_tokens}; "
            f"tokenize(P)={len(prefix_candidate)} tokenize(P+Q)={len(continuation_prompt)} "
            f"common_prefix={common_prefix}")
    if target_is_valid and (
            not isinstance(target_delta, int) or target_delta < 0 or
            target_delta >= PAGED_BLOCK_SIZE):
        failure_reasons.append(
            f"selected P misses target by {target_delta} tokens; requires "
            f"0<=target-actual<{PAGED_BLOCK_SIZE} with target={target_prefix_tokens} "
            f"actual={len(prompt_p)} common_prefix={common_prefix}")
    if not strict_prefix:
        failure_reasons.append(
            f"selected tokens(P) is not a strict prefix of tokens(P+Q): "
            f"P={len(prompt_p)} P+Q={len(continuation_prompt)} common_prefix={common_prefix}")
    if ctx_is_valid and context_token_count > ctx_size:
        failure_reasons.append(
            f"tokens(P+Q)+n_predict exceeds ctx_size: "
            f"{len(continuation_prompt)}+{N_PREDICT}={context_token_count}>{ctx_size}")
    evidence = {
        "status": "ready" if not failure_reasons else "invalid",
        "token_boundary_policy": "largest_paged_block_boundary_within_tokenized_common_prefix",
        "prefix_candidate_token_count": len(prefix_candidate),
        "continuation_prompt_token_count": len(continuation_prompt),
        "common_prefix_token_count": common_prefix,
        "prefix_token_count": len(prompt_p),
        "prefix_block_count": len(prompt_p) // PAGED_BLOCK_SIZE,
        "continuation_suffix_token_count": len(continuation_prompt) - len(prompt_p),
        "n_predict": N_PREDICT,
        "context_token_count": context_token_count,
        "minimum_prefix_blocks": MIN_PREFIX_BLOCKS,
        "minimum_prefix_token_count": MIN_PREFIX_TOKENS,
        "ctx_size": ctx_size,
        "prefix_covers_minimum_blocks": len(prompt_p) >= MIN_PREFIX_TOKENS,
        "target_prefix_satisfied": (
            isinstance(target_delta, int) and 0 <= target_delta < PAGED_BLOCK_SIZE),
        "strict_prefix": strict_prefix,
        "context_fits": ctx_is_valid and context_token_count <= ctx_size,
        "first_token_mismatch_index": first_mismatch,
        "first_token_mismatch": mismatch_tokens,
        "prefix_candidate_tokens_sha256": token_ids_sha256(prefix_candidate),
        "prefix_tokens_sha256": token_ids_sha256(prompt_p),
        "continuation_prompt_tokens_sha256": token_ids_sha256(continuation_prompt),
        "failure_reasons": failure_reasons,
        **selection,
    }
    return prompt_p, evidence


def scalable_prefix_text(unit_count: int) -> str:
    if (not isinstance(unit_count, int) or isinstance(unit_count, bool) or
            not 1 <= unit_count <= MAX_PREFIX_TEXT_UNITS):
        raise ValueError("prefix text unit count is out of range")
    return PREFIX_TEXT_UNIT * unit_count


def tokenization_failure_evidence(
        stage: str,
        reason: str,
        ctx_size: int,
        target_prefix_tokens: int,
        prefix_text_unit_count: int | None = None,
        prefix_candidate: list[int] | None = None) -> dict[str, Any]:
    selected_text = (
        scalable_prefix_text(prefix_text_unit_count)
        if prefix_text_unit_count is not None else None)
    return {
        "status": "tokenization_failed",
        "token_boundary_policy": "largest_paged_block_boundary_within_tokenized_common_prefix",
        "tokenization_stage": stage,
        "prefix_text_unit_count": prefix_text_unit_count,
        "prefix_text_max_units": MAX_PREFIX_TEXT_UNITS,
        "selected_prefix_text_sha256": (
            sha_bytes(selected_text.encode("utf-8")) if selected_text is not None else None),
        "prefix_candidate_token_count": len(prefix_candidate) if prefix_candidate is not None else None,
        "continuation_prompt_token_count": None,
        "common_prefix_token_count": None,
        "prefix_token_count": None,
        "prefix_block_count": None,
        "continuation_suffix_token_count": None,
        "n_predict": N_PREDICT,
        "context_token_count": None,
        "minimum_prefix_blocks": MIN_PREFIX_BLOCKS,
        "minimum_prefix_token_count": MIN_PREFIX_TOKENS,
        "ctx_size": ctx_size,
        "target_prefix_tokens": target_prefix_tokens,
        "actual_p_token_count": None,
        "actual_pq_token_count": None,
        "actual_p_block_count": None,
        "target_prefix_token_delta": None,
        "prefix_covers_minimum_blocks": None,
        "target_prefix_satisfied": False,
        "strict_prefix": None,
        "context_fits": None,
        "first_token_mismatch_index": None,
        "first_token_mismatch": None,
        "prefix_candidate_tokens_sha256": token_ids_sha256(prefix_candidate) if prefix_candidate is not None else None,
        "prefix_tokens_sha256": None,
        "continuation_prompt_tokens_sha256": None,
        "failure_reasons": [reason],
    }


def prepare_token_workload(
        case: pathlib.Path,
        port: int,
        ctx_size: int,
        target_prefix_tokens: int) -> tuple[list[int], list[int]]:
    if (not isinstance(target_prefix_tokens, int) or isinstance(target_prefix_tokens, bool) or
            target_prefix_tokens < MIN_PREFIX_TOKENS):
        reason = f"target_prefix_tokens must be an integer at least {MIN_PREFIX_TOKENS}"
        dump(case / "workload.json", tokenization_failure_evidence(
            "selection", reason, ctx_size, target_prefix_tokens))
        raise WorkloadFailure(reason)

    candidates: dict[int, tuple[str, list[int], list[int]]] = {}

    def target_satisfied(
            prefix_candidate: list[int],
            continuation_prompt: list[int]) -> bool:
        actual = aligned_prefix_token_count(
            prefix_candidate, continuation_prompt, target_prefix_tokens)
        return 0 <= target_prefix_tokens - actual < PAGED_BLOCK_SIZE

    def tokenized(unit_count: int) -> tuple[str, list[int], list[int]]:
        cached = candidates.get(unit_count)
        if cached is not None:
            return cached
        prefix_text = scalable_prefix_text(unit_count)
        try:
            prefix_candidate = tokenize(port, prefix_text, True)
        except Exception as exc:
            reason = f"tokenize(P) failed: {type(exc).__name__}: {exc}"
            dump(case / "workload.json", tokenization_failure_evidence(
                "P", reason, ctx_size, target_prefix_tokens, unit_count))
            raise WorkloadFailure(reason) from exc
        try:
            continuation_prompt = tokenize(port, prefix_text + QUERY_TEXT, True)
        except Exception as exc:
            reason = f"tokenize(P+Q) failed: {type(exc).__name__}: {exc}"
            dump(case / "workload.json", tokenization_failure_evidence(
                "P+Q", reason, ctx_size, target_prefix_tokens, unit_count, prefix_candidate))
            raise WorkloadFailure(reason) from exc
        value = (prefix_text, prefix_candidate, continuation_prompt)
        candidates[unit_count] = value
        return value

    lower_units = 0
    upper_units = 1
    while True:
        _text, prefix_candidate, continuation_prompt = tokenized(upper_units)
        if target_satisfied(prefix_candidate, continuation_prompt):
            break
        if upper_units == MAX_PREFIX_TEXT_UNITS:
            prompt_p, evidence = build_token_workload(
                prefix_candidate, continuation_prompt, ctx_size, target_prefix_tokens)
            evidence.update({
                "prefix_text_unit_count": upper_units,
                "prefix_text_max_units": MAX_PREFIX_TEXT_UNITS,
                "selected_prefix_text_sha256": sha_bytes(_text.encode("utf-8")),
            })
            dump(case / "workload.json", evidence)
            raise WorkloadFailure(
                "scalable P/P+Q token workload contract failed: " +
                "; ".join(evidence["failure_reasons"]))
        lower_units = upper_units
        upper_units = min(MAX_PREFIX_TEXT_UNITS, upper_units * 2)

    while lower_units + 1 < upper_units:
        middle_units = (lower_units + upper_units) // 2
        _text, prefix_candidate, continuation_prompt = tokenized(middle_units)
        if target_satisfied(prefix_candidate, continuation_prompt):
            upper_units = middle_units
        else:
            lower_units = middle_units

    prefix_text, prefix_candidate, continuation_prompt = tokenized(upper_units)
    prompt_p, evidence = build_token_workload(
        prefix_candidate, continuation_prompt, ctx_size, target_prefix_tokens)
    evidence.update({
        "prefix_text_unit_count": upper_units,
        "prefix_text_max_units": MAX_PREFIX_TEXT_UNITS,
        "selected_prefix_text_sha256": sha_bytes(prefix_text.encode("utf-8")),
    })
    dump(case / "workload.json", evidence)
    if evidence["failure_reasons"]:
        raise WorkloadFailure(
            "scalable P/P+Q token workload contract failed: " +
            "; ".join(evidence["failure_reasons"]))
    return prompt_p, continuation_prompt


def completion_body(prompt: list[int], n_predict: int) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "cache_prompt": True,
        "id_slot": 0,
        "stream": False,
    }


def request_completion(port: int, body: dict[str, Any], label: str, record: pathlib.Path) -> dict[str, Any]:
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    item: dict[str, Any] = {
        "label": label,
        "request": body,
        "request_sha256": sha_bytes(encoded),
        "started_monotonic_ns": time.monotonic_ns(),
    }
    con: http.client.HTTPConnection | None = None
    status = 0
    raw = b""
    value: Any = {}
    try:
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=REQUEST_TIMEOUT_SECONDS)
        con.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
        response = con.getresponse()
        status = response.status
        raw = response.read()
        value = json.loads(raw) if raw else {}
    except (OSError, json.JSONDecodeError) as exc:
        value = {"transport_error": f"{type(exc).__name__}: {exc}"}
    finally:
        if con is not None:
            try:
                con.close()
            except OSError:
                pass
    text = value.get("content", "") if isinstance(value, dict) and isinstance(value.get("content"), str) else ""
    if not text and isinstance(value, dict) and isinstance(value.get("choices"), list) and value["choices"]:
        choice = value["choices"][0]
        if isinstance(choice, dict) and isinstance(choice.get("text"), str):
            text = choice["text"]
    response_timings = value.get("timings") if isinstance(value, dict) else None
    item.update({
        "http_status": status,
        "response_raw": raw.decode("utf-8", errors="replace"),
        "response_text": text,
        "response_sha256": sha_bytes(text.encode("utf-8")),
        "response_timings": response_timings,
        "finished_monotonic_ns": time.monotonic_ns(),
    })
    with record.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")
    return item


def write_execution(
        case: pathlib.Path,
        binary: pathlib.Path,
        model: pathlib.Path,
        argv: list[str],
        env: dict[str, str],
        server_identity: dict[str, Any]) -> None:
    dump(case / "execution.json", {
        "argv": argv,
        "cwd": str(case.resolve()),
        "environment": env,
        "binary": identity(binary),
        "model": identity(model),
        "server_identity": server_identity,
    })


def verify_capability(capability: dict[str, str]) -> str | None:
    unavailable = {key: capability[key] for key in REQUIRED_CAPABILITY if capability[key] != "1"}
    if unavailable:
        return f"required production capability unavailable: {unavailable}"
    if any(capability[key] != "1" for key in ("n_slots", "n_seq_max", "n_stream")):
        return "production capability does not describe a single-session server"
    return None


def run_physical_preflight(
        binary: pathlib.Path,
        model: pathlib.Path,
        root: pathlib.Path,
        ctx_size: int,
        target_prefix_tokens: int) -> tuple[str, str]:
    case = root / "PREFLIGHT"
    case.mkdir()
    backing = prepare_backing(case)
    env = governor_env(False)
    env["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"] = "preflight"
    dump(case / "environment.json", env)
    port = free_port()
    argv = server_argv(binary, model, port, ctx_size)
    result: dict[str, Any] = {
        "status": "startup_failed",
        "request_loop_started": False,
        "port": port,
        "ctx_size": ctx_size,
        "target_prefix_tokens": target_prefix_tokens,
    }
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = start(case, argv, env)
        server_identity = read_process_identity(proc.pid)
        if server_identity is None:
            return "failure", "could not bind server PID/starttime/cmdline identity"
        write_execution(case, binary, model, argv, env, server_identity)
        if not wait_health(port, proc):
            return "failure", f"server listener was not healthy (exit={proc.poll()})"
        capability, evidence, capability_error = wait_capability(case, proc)
        if capability_error:
            return "failure", capability_error
        assert capability is not None and evidence is not None
        dump(case / "capability.json", evidence)
        capability_error = verify_capability(capability)
        if capability_error:
            return "unsupported", capability_error
        status, raw, slots = query_slots_raw(port)
        (case / "physical_probe.raw.json").write_bytes(raw)
        probe = physical_probe(slots, raw, server_identity)
        probe["slots_http_status"] = status
        dump(case / "physical_probe.json", probe)
        if status != 200 or slots is None:
            return "failure", "could not acquire a valid /slots physical-observation probe"
        if not probe["physical_resident_sample_available"]:
            return "unsupported", str(probe["reason"])
        result["status"] = "supported"
        return "supported", ""
    except RunnerInterrupted:
        raise
    except OSError as exc:
        return "failure", f"server execution failed: {type(exc).__name__}: {exc}"
    finally:
        server_cleanup = stop(proc) if proc is not None else {
            "pid": None, "pgid": None, "exit_code": None,
            "term_timed_out": False, "kill_timed_out": False, "residual_process": False,
        }
        backing_cleanup = cleanup_backing(case, backing)
        result["process"] = server_cleanup
        result["backing"] = backing_cleanup
        dump(case / "cleanup.json", {"server": server_cleanup, "backing": backing_cleanup})
        dump(case / "result.json", result)


def run_case(
        name: str,
        binary: pathlib.Path,
        model: pathlib.Path,
        enabled: bool,
        root: pathlib.Path,
        ctx_size: int,
        target_prefix_tokens: int) -> dict[str, Any]:
    case = root / name
    case.mkdir()
    backing = prepare_backing(case)
    env = governor_env(enabled)
    dump(case / "environment.json", env)
    port = free_port()
    argv = server_argv(binary, model, port, ctx_size)
    result: dict[str, Any] = {
        "status": "startup_failed",
        "request_loop_started": False,
        "port": port,
        "ctx_size": ctx_size,
        "target_prefix_tokens": target_prefix_tokens,
    }
    proc: subprocess.Popen[bytes] | None = None
    step2: dict[str, Any] | None = None
    resume_boundary: int | None = None
    try:
        proc = start(case, argv, env)
        server_identity = read_process_identity(proc.pid)
        if server_identity is None:
            raise WorkloadFailure("could not bind server PID/starttime/cmdline identity")
        write_execution(case, binary, model, argv, env, server_identity)
        if not wait_health(port, proc):
            raise WorkloadFailure(f"server listener was not healthy (exit={proc.poll()})")
        capability, evidence, capability_error = wait_capability(case, proc)
        if capability_error:
            raise WorkloadFailure(capability_error)
        assert capability is not None and evidence is not None
        dump(case / "capability.json", evidence)
        capability_error = verify_capability(capability)
        if capability_error:
            raise WorkloadFailure(capability_error)

        prompt_p, prompt_pq = prepare_token_workload(
            case, port, ctx_size, target_prefix_tokens)
        result["workload"] = target_prefix_record(
            target_prefix_tokens, prompt_p, prompt_pq)

        records = case / "requests.jsonl"
        result["request_loop_started"] = True
        step1_stderr_start = (case / "server.stderr").stat().st_size
        step1 = request_completion(port, completion_body(prompt_p, 0), "step1", records)
        if step1["http_status"] != 200:
            raise WorkloadFailure("step1 did not receive HTTP 200")

        if enabled:
            _offload, post = collect_offload_transactions(
                case, proc, port, step1_stderr_start,
                len(prompt_p) // PAGED_BLOCK_SIZE)
            resume_boundary = post["stderr_end"]

        resume_start = (
            resume_boundary if resume_boundary is not None else
            (case / "server.stderr").stat().st_size)
        step2 = request_completion(port, completion_body(prompt_pq, N_PREDICT), "step2", records)
        resume_end = (case / "server.stderr").stat().st_size
        dump(case / "resume_scope.json", {
            "start": resume_start,
            "end": resume_end,
            "request_label": "step2",
            "request_started_monotonic_ns": step2["started_monotonic_ns"],
            "request_finished_monotonic_ns": step2["finished_monotonic_ns"],
        })

        if step2["http_status"] != 200:
            raise WorkloadFailure("step2 did not receive HTTP 200")
        result.update({
            "status": "complete",
            "capability": capability,
            "http_statuses": {"step1": step1["http_status"], "step2": step2["http_status"]},
        })
    except RunnerInterrupted:
        raise
    except (OSError, WorkloadFailure) as exc:
        result.update({"status": "request_failed", "workload_error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        result.update({"status": "request_failed", "workload_error": f"{type(exc).__name__}: {exc}"})
    finally:
        server_cleanup = stop(proc) if proc is not None else {
            "pid": None, "pgid": None, "exit_code": None,
            "term_timed_out": False, "kill_timed_out": False, "residual_process": False,
        }
        if result.get("status") == "complete" and step2 is not None:
            try:
                timing = capture_resume_timing(case, step2)
                result["timing"] = timing["derived"]
            except (OSError, ValueError, json.JSONDecodeError, WorkloadFailure) as exc:
                result["status"] = "request_failed"
                result["timing_error"] = f"{type(exc).__name__}: {exc}"
        backing_cleanup = cleanup_backing(case, backing)
        result["process"] = server_cleanup
        result["backing"] = backing_cleanup
        if server_cleanup["residual_process"] or server_cleanup["term_timed_out"] or server_cleanup["kill_timed_out"]:
            result["status"] = "request_failed"
            result["cleanup_error"] = "server cleanup incomplete"
        if backing_cleanup["exists_after_cleanup"] or backing_cleanup["cleanup_error"] is not None:
            result["status"] = "request_failed"
            result["cleanup_error"] = "backing cleanup incomplete"
        dump(case / "cleanup.json", {"server": server_cleanup, "backing": backing_cleanup})
        dump(case / "result.json", result)
    return result


def base_manifest(
        binary: pathlib.Path,
        model: pathlib.Path | None,
        stamp: str,
        ctx_size: int,
        target_prefix_tokens: int) -> dict[str, Any]:
    dirty = git("status", "--porcelain").splitlines()
    return {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "source_marker_schema": SOURCE_MARKER_SCHEMA,
        "timestamp_utc": stamp,
        "finished_timestamp_utc": stamp,
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "dirty_status": dirty,
        "tracked_diff_fingerprint": tracked_diff_fingerprint(),
        "capture_mode": "diagnostic_dirty" if dirty else "archival_clean",
        "runner": identity(pathlib.Path(__file__)),
        "parser": identity(ROOT / "scripts/parse-kv-governor-g0-s1.py"),
        "binary_requested": str(binary),
        "model_requested": str(model) if model is not None else None,
        "parameters": {
            "parallel": 1,
            "n_stream": 1,
            "id_slot": 0,
            "cache_prompt": True,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "step1_n_predict": 0,
            "step2_n_predict": N_PREDICT,
            "mincore_requested": True,
            "source_marker_schema": SOURCE_MARKER_SCHEMA,
            "paged_block_size": PAGED_BLOCK_SIZE,
            "ctx_size": ctx_size,
            "target_prefix_tokens": target_prefix_tokens,
            "governor_target_bytes": GOVERNOR_TARGET_BYTES,
            "governor_max_blocks": GOVERNOR_MAX_BLOCKS,
            "prefix_text_sha256": sha_bytes(PREFIX_TEXT.encode("utf-8")),
            "query_text_sha256": sha_bytes(QUERY_TEXT.encode("utf-8")),
        },
        "case_names": ["OFF", "GOVERNOR_ON"],
        "runner_status": "run_in_progress",
    }


def invoke_parser(root: pathlib.Path) -> int:
    parser = ROOT / "scripts/parse-kv-governor-g0-s1.py"
    return subprocess.run(
        [sys.executable, str(parser), str(root), "--result-path", str(root / "parser.json")],
        text=True,
        encoding="utf-8",
    ).returncode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("KV_GOVERNOR_BINARY", "build/bin/llama-server"))
    ap.add_argument("--model", default=os.environ.get("KV_GOVERNOR_MODEL"))
    ap.add_argument("--output-dir")
    ap.add_argument("--ctx-size", type=int, choices=SUPPORTED_CTX_SIZES, default=CTX_SIZE)
    ap.add_argument("--target-prefix-tokens", type=int, required=True)
    args = ap.parse_args()
    if args.target_prefix_tokens < MIN_PREFIX_TOKENS:
        ap.error(f"--target-prefix-tokens must be at least {MIN_PREFIX_TOKENS}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(
        f"/root/oscomp/kv_logs/{PROTOCOL}_{stamp}_{uuid.uuid4().hex[:10]}")
    if output.exists():
        raise SystemExit(f"refusing existing artifact directory: {output}")
    output.mkdir(parents=True)
    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).resolve() if args.model else None
    manifest = base_manifest(
        binary, model, stamp, args.ctx_size, args.target_prefix_tokens)
    signal_seen: list[int] = []

    def handle_signal(signum: int, _frame: Any) -> None:
        if not signal_seen:
            signal_seen.append(signum)
            raise RunnerInterrupted(signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        if not binary.is_file() or model is None or not model.is_file():
            if binary.is_file():
                manifest["binary"] = identity(binary)
            if model is not None and model.is_file():
                manifest["model"] = identity(model)
            manifest.update({
                "runner_status": "UNSUPPORTED",
                "unsupported_stage": "pre_workload_binary_model",
                "unsupported_reason": "binary or model is unavailable; no server/model workload evidence was fabricated",
            })
        else:
            manifest.update({"binary": identity(binary), "model": identity(model)})
            outcome, reason = run_physical_preflight(
                binary, model, output, args.ctx_size, args.target_prefix_tokens)
            if outcome == "unsupported":
                manifest.update({
                    "runner_status": "UNSUPPORTED",
                    "unsupported_stage": "pre_workload_physical_probe",
                    "unsupported_reason": reason,
                })
            elif outcome == "failure":
                manifest.update({"runner_status": "run_incomplete", "runner_error": reason})
            else:
                manifest["OFF"] = run_case(
                    "OFF", binary, model, False, output, args.ctx_size, args.target_prefix_tokens)
                manifest["GOVERNOR_ON"] = run_case(
                    "GOVERNOR_ON", binary, model, True, output, args.ctx_size, args.target_prefix_tokens)
                manifest["runner_status"] = "run_complete"
    except RunnerInterrupted as exc:
        manifest.update({
            "runner_status": "run_interrupted",
            "interruption": {"signal": exc.signum, "message": str(exc)},
        })
    except Exception as exc:
        manifest.update({
            "runner_status": "run_incomplete",
            "runner_error": f"{type(exc).__name__}: {exc}",
        })
    finally:
        manifest["finished_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump(output / "manifest.json", manifest)
    if manifest["runner_status"] == "run_interrupted":
        raise SystemExit(128 + int(manifest["interruption"]["signal"]))
    raise SystemExit(invoke_parser(output))


if __name__ == "__main__":
    main()
