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
PREFIX_TEXT = ("G0 S1 fixed long prefix token. " * 128).strip()
QUERY_TEXT = "\nContinue with one deterministic concise answer."


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


def server_argv(binary: pathlib.Path, model: pathlib.Path, port: int) -> list[str]:
    return [
        str(binary), "--host", "127.0.0.1", "--port", str(port), "--model", str(model),
        "--ctx-size", str(CTX_SIZE), "--parallel", "1", "--kv-unified", "--no-cache-idle-slots",
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


def start(binary: pathlib.Path, model: pathlib.Path, case: pathlib.Path, port: int, env: dict[str, str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        server_argv(binary, model, port),
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
        offload: dict[str, str]) -> dict[str, Any] | None:
    matches = [
        record for record in resident_observation_records(case)
        if record[0] is not None and
        record[0].get("decision_id") == offload.get("decision_id") and
        record[0].get("seq_id") == offload.get("selected_seq_id") and
        record[0].get("transaction_id") == offload.get("transaction_id")
    ]
    if not matches:
        return None
    fields, start, end = matches[0]
    assert fields is not None
    value = {"offset": start, "end": end, "fields": fields}
    dump(case / "resident.json", value)
    return value


def marker_is_changed_offload(fields: dict[str, str]) -> bool:
    try:
        return (
            fields.get("offload_attempted") == "1" and
            fields.get("state_changed") == "1" and
            int(fields.get("transaction_id", "0")) > 0)
    except ValueError:
        return False


def wait_changed_offload(case: pathlib.Path, proc: subprocess.Popen[bytes], start: int) -> tuple[dict[str, str], int, int]:
    deadline = time.monotonic() + MARKER_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        records, _ = marker_records(case, start)
        for fields, begin, end in records:
            if marker_is_changed_offload(fields):
                return fields, begin, end
        if proc.poll() is not None:
            raise WorkloadFailure(f"server exited while waiting for OFFLOAD (exit={proc.returncode})")
        time.sleep(0.05)
    raise WorkloadFailure("timed out waiting for a state-changing OFFLOAD")


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


def build_token_workload(prefix_candidate: list[int], continuation_prompt: list[int]) -> tuple[list[int], dict[str, Any]]:
    common_prefix = common_token_prefix_count(prefix_candidate, continuation_prompt)
    boundary_limit = min(common_prefix, max(0, len(continuation_prompt) - 1))
    prefix_token_count = boundary_limit // PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE
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
    failure_reasons: list[str] = []
    if len(prompt_p) < MIN_PREFIX_TOKENS:
        failure_reasons.append(
            f"selected P has {len(prompt_p)} tokens ({len(prompt_p) // PAGED_BLOCK_SIZE} full "
            f"{PAGED_BLOCK_SIZE}-token blocks); requires at least {MIN_PREFIX_TOKENS} tokens "
            f"({MIN_PREFIX_BLOCKS} blocks); tokenize(P)={len(prefix_candidate)} "
            f"tokenize(P+Q)={len(continuation_prompt)} common_prefix={common_prefix}")
    if not strict_prefix:
        failure_reasons.append(
            f"selected tokens(P) is not a strict prefix of tokens(P+Q): "
            f"P={len(prompt_p)} P+Q={len(continuation_prompt)} common_prefix={common_prefix}")
    if context_token_count > CTX_SIZE:
        failure_reasons.append(
            f"tokens(P+Q)+n_predict exceeds ctx_size: "
            f"{len(continuation_prompt)}+{N_PREDICT}={context_token_count}>{CTX_SIZE}")
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
        "ctx_size": CTX_SIZE,
        "prefix_covers_minimum_blocks": len(prompt_p) >= MIN_PREFIX_TOKENS,
        "strict_prefix": strict_prefix,
        "context_fits": context_token_count <= CTX_SIZE,
        "first_token_mismatch_index": first_mismatch,
        "first_token_mismatch": mismatch_tokens,
        "prefix_candidate_tokens_sha256": token_ids_sha256(prefix_candidate),
        "prefix_tokens_sha256": token_ids_sha256(prompt_p),
        "continuation_prompt_tokens_sha256": token_ids_sha256(continuation_prompt),
        "failure_reasons": failure_reasons,
    }
    return prompt_p, evidence


def tokenization_failure_evidence(
        stage: str,
        reason: str,
        prefix_candidate: list[int] | None = None) -> dict[str, Any]:
    return {
        "status": "tokenization_failed",
        "token_boundary_policy": "largest_paged_block_boundary_within_tokenized_common_prefix",
        "tokenization_stage": stage,
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
        "ctx_size": CTX_SIZE,
        "prefix_covers_minimum_blocks": None,
        "strict_prefix": None,
        "context_fits": None,
        "first_token_mismatch_index": None,
        "first_token_mismatch": None,
        "prefix_candidate_tokens_sha256": token_ids_sha256(prefix_candidate) if prefix_candidate is not None else None,
        "prefix_tokens_sha256": None,
        "continuation_prompt_tokens_sha256": None,
        "failure_reasons": [reason],
    }


def prepare_token_workload(case: pathlib.Path, port: int) -> tuple[list[int], list[int]]:
    prefix_candidate: list[int] | None = None
    try:
        prefix_candidate = tokenize(port, PREFIX_TEXT, True)
    except Exception as exc:
        reason = f"tokenize(P) failed: {type(exc).__name__}: {exc}"
        dump(case / "workload.json", tokenization_failure_evidence("P", reason))
        raise WorkloadFailure(reason) from exc
    try:
        continuation_prompt = tokenize(port, PREFIX_TEXT + QUERY_TEXT, True)
    except Exception as exc:
        reason = f"tokenize(P+Q) failed: {type(exc).__name__}: {exc}"
        dump(case / "workload.json", tokenization_failure_evidence("P+Q", reason, prefix_candidate))
        raise WorkloadFailure(reason) from exc
    prompt_p, evidence = build_token_workload(prefix_candidate, continuation_prompt)
    dump(case / "workload.json", evidence)
    if evidence["failure_reasons"]:
        raise WorkloadFailure(
            "fixed P/P+Q token workload contract failed: " + "; ".join(evidence["failure_reasons"]))
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
    item.update({
        "http_status": status,
        "response_raw": raw.decode("utf-8", errors="replace"),
        "response_text": text,
        "response_sha256": sha_bytes(text.encode("utf-8")),
        "finished_monotonic_ns": time.monotonic_ns(),
    })
    with record.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")
    return item


def write_execution(case: pathlib.Path, binary: pathlib.Path, model: pathlib.Path, port: int, env: dict[str, str], server_identity: dict[str, Any]) -> None:
    dump(case / "execution.json", {
        "argv": server_argv(binary, model, port),
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


def run_physical_preflight(binary: pathlib.Path, model: pathlib.Path, root: pathlib.Path) -> tuple[str, str]:
    case = root / "PREFLIGHT"
    case.mkdir()
    backing = prepare_backing(case)
    env = governor_env(False)
    env["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"] = "preflight"
    dump(case / "environment.json", env)
    port = free_port()
    result: dict[str, Any] = {"status": "startup_failed", "request_loop_started": False, "port": port}
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = start(binary, model, case, port, env)
        server_identity = read_process_identity(proc.pid)
        if server_identity is None:
            return "failure", "could not bind server PID/starttime/cmdline identity"
        write_execution(case, binary, model, port, env, server_identity)
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


def run_case(name: str, binary: pathlib.Path, model: pathlib.Path, enabled: bool, root: pathlib.Path) -> dict[str, Any]:
    case = root / name
    case.mkdir()
    backing = prepare_backing(case)
    env = governor_env(enabled)
    dump(case / "environment.json", env)
    port = free_port()
    result: dict[str, Any] = {"status": "startup_failed", "request_loop_started": False, "port": port}
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = start(binary, model, case, port, env)
        server_identity = read_process_identity(proc.pid)
        if server_identity is None:
            raise WorkloadFailure("could not bind server PID/starttime/cmdline identity")
        write_execution(case, binary, model, port, env, server_identity)
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

        prompt_p, prompt_pq = prepare_token_workload(case, port)

        records = case / "requests.jsonl"
        result["request_loop_started"] = True
        step1_stderr_start = (case / "server.stderr").stat().st_size
        step1 = request_completion(port, completion_body(prompt_p, 0), "step1", records)
        if step1["http_status"] != 200:
            raise WorkloadFailure("step1 did not receive HTTP 200")

        if enabled:
            offload_fields, offload_start, offload_end = wait_changed_offload(
                case, proc, step1_stderr_start)
            dump(case / "offload.json", {
                "offset": offload_start,
                "end": offload_end,
                "fields": offload_fields,
            })
            capture_resident_observation(case, offload_fields)
            capture_snapshot(case, port, "post_claimant")
            resume_start = (case / "server.stderr").stat().st_size
            step2 = request_completion(port, completion_body(prompt_pq, N_PREDICT), "step2", records)
            resume_end = (case / "server.stderr").stat().st_size
            dump(case / "resume_scope.json", {
                "start": resume_start,
                "end": resume_end,
                "request_label": "step2",
                "request_started_monotonic_ns": step2["started_monotonic_ns"],
                "request_finished_monotonic_ns": step2["finished_monotonic_ns"],
            })
        else:
            step2 = request_completion(port, completion_body(prompt_pq, N_PREDICT), "step2", records)

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


def base_manifest(binary: pathlib.Path, model: pathlib.Path | None, stamp: str) -> dict[str, Any]:
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
            "ctx_size": CTX_SIZE,
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
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(
        f"/root/oscomp/kv_logs/{PROTOCOL}_{stamp}_{uuid.uuid4().hex[:10]}")
    if output.exists():
        raise SystemExit(f"refusing existing artifact directory: {output}")
    output.mkdir(parents=True)
    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).resolve() if args.model else None
    manifest = base_manifest(binary, model, stamp)
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
            outcome, reason = run_physical_preflight(binary, model, output)
            if outcome == "unsupported":
                manifest.update({
                    "runner_status": "UNSUPPORTED",
                    "unsupported_stage": "pre_workload_physical_probe",
                    "unsupported_reason": reason,
                })
            elif outcome == "failure":
                manifest.update({"runner_status": "run_incomplete", "runner_error": reason})
            else:
                manifest["OFF"] = run_case("OFF", binary, model, False, output)
                manifest["GOVERNOR_ON"] = run_case("GOVERNOR_ON", binary, model, True, output)
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
