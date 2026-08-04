#!/usr/bin/env python3
"""Real-model, fail-closed Stage 3C Governor smoke for unified multi-slot KV."""
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
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "kv_governor_stage3c_1c_2b_1r"
PROTOCOL_VERSION = 6
PARALLELS = (2, 3)
N_PREDICT = 0
ACTIVE_N_PREDICT = -1
SEED = 1
PAGED_BLOCK_SIZE = 64
GOVERNOR_TARGET_BYTES = 1073741824
GOVERNOR_MAX_BLOCKS = 64
# parallel=3 worst-case prefill of three 576-token prompts measured ~30.74s on the
# reference host (server print_timing). A 30s client timeout fires before the slowest
# response returns, turning completed requests into transport_error and cascading to
# missing gate boundaries. Defaults cover that worst case with margin and remain
# configurable via environment for slower hosts.
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("KV_GOVERNOR_REQUEST_TIMEOUT_SECONDS", "90"))
ACTIVE_SOCKET_TIMEOUT_SECONDS = float(os.environ.get("KV_GOVERNOR_ACTIVE_SOCKET_TIMEOUT_SECONDS", "90"))
ACTIVE_CANCEL_TIMEOUT_SECONDS = 8.0
MARKER_TIMEOUT_SECONDS = 20.0
LAYOUT_TIMEOUT_SECONDS = 20.0
SERVER_TERM_TIMEOUT_SECONDS = 10.0
SERVER_KILL_TIMEOUT_SECONDS = 5.0
PROMPT = ("KV governor unified multi-slot smoke context. Keep the answer deterministic. " * 30).strip()
BASE_ENV = {"HOME": "/tmp", "LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "")}
CAPABILITY_MARKER = "KV_GOVERNOR_CAPABILITY"
CAPABILITY_FIELDS = {
    "n_slots", "n_seq_max", "n_stream", "kv_unified", "paged_metadata",
    "ingraph_gather", "release_supported", "offload_supported",
    "prefetch_supported", "backing_ready", "swap_explicit_only",
}
REQUIRED_CAPABILITY = {
    "kv_unified", "paged_metadata", "ingraph_gather", "release_supported",
    "offload_supported", "prefetch_supported", "backing_ready", "swap_explicit_only",
}
MARKER = "kv_pressure_unified_action"


def dump(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha(path)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(
        port: int,
        body: dict[str, Any],
        label: str,
        record: pathlib.Path,
        record_lock: threading.Lock,
        timeout: float = REQUEST_TIMEOUT_SECONDS) -> int:
    encoded = json.dumps(body, separators=(",", ":")).encode()
    item: dict[str, Any] = {
        "label": label,
        "request": body,
        "request_sha256": hashlib.sha256(encoded).hexdigest(),
        "started_monotonic_ns": time.monotonic_ns(),
    }
    status, raw, data = 0, b"", {}
    try:
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        con.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
        response = con.getresponse()
        status, raw = response.status, response.read()
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            data = {"non_object": True}
    except Exception as exc:
        data = {"transport_error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            con.close()
        except (UnboundLocalError, OSError):
            pass
    text = data.get("content", "") if isinstance(data.get("content"), str) else ""
    if not text and isinstance(data.get("choices"), list) and data["choices"]:
        text = str(data["choices"][0].get("text", ""))
    item.update({
        "http_status": status,
        "response": data,
        "response_raw": raw.decode(errors="replace"),
        "response_text": text,
        "response_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finished_monotonic_ns": time.monotonic_ns(),
    })
    with record_lock:
        with record.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, sort_keys=True) + "\n")
    return status


class WorkloadFailure(Exception):
    pass


class RunnerInterrupted(BaseException):
    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


class ActiveRequest:
    def __init__(
            self,
            port: int,
            body: dict[str, Any],
            label: str,
            record: pathlib.Path,
            record_lock: threading.Lock,
            start_event: threading.Event):
        self.port = port
        self.body = body
        self.label = label
        self.record = record
        self.record_lock = record_lock
        self.start_event = start_event
        self.stop_event = threading.Event()
        self.stop_reason: str | None = None
        self.stop_requested_monotonic_ns: int | None = None
        self.result: dict[str, Any] = {}
        self.connection: http.client.HTTPConnection | None = None
        self.response: http.client.HTTPResponse | None = None
        self.io_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name=f"kv-governor-{label}", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def _run(self) -> None:
        encoded = json.dumps(self.body, separators=(",", ":")).encode()
        item: dict[str, Any] = {
            "label": self.label,
            "request": self.body,
            "request_sha256": hashlib.sha256(encoded).hexdigest(),
            "started_monotonic_ns": time.monotonic_ns(),
        }
        status = 0
        stream_bytes = 0
        stream_chunks = 0
        stream_sample = bytearray()
        stream_digest = hashlib.sha256()
        completion = "not_started"
        transport_error: str | None = None
        try:
            if not self.start_event.wait(timeout=5.0):
                raise TimeoutError("active request start barrier timed out")
            if self.stop_event.is_set():
                completion = "runner_cancelled"
                return
            connection = http.client.HTTPConnection(
                "127.0.0.1", self.port, timeout=ACTIVE_SOCKET_TIMEOUT_SECONDS)
            with self.io_lock:
                self.connection = connection
            connection.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
            response = connection.getresponse()
            with self.io_lock:
                self.response = response
            status = response.status
            if status != 200:
                payload = response.read()
                stream_digest.update(payload)
                stream_sample.extend(payload[:4096])
                stream_bytes = len(payload)
                stream_chunks = 1 if payload else 0
                completion = "http_error"
            else:
                completion = "server_eof"
                while not self.stop_event.is_set():
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    stream_digest.update(chunk)
                    stream_bytes += len(chunk)
                    stream_chunks += 1
                    if len(stream_sample) < 4096:
                        stream_sample.extend(chunk[:4096 - len(stream_sample)])
                if self.stop_event.is_set():
                    completion = "runner_cancelled"
        except Exception as exc:
            transport_error = f"{type(exc).__name__}: {exc}"
            completion = "runner_cancelled" if self.stop_event.is_set() else "transport_error"
        finally:
            with self.io_lock:
                response = self.response
                connection = self.connection
                self.response = None
                self.connection = None
            if response is not None:
                try:
                    response.close()
                except OSError:
                    pass
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            finished_ns = time.monotonic_ns()
            item.update({
                "http_status": status,
                "response": {
                    "completion": completion,
                    "stream_bytes": stream_bytes,
                    "stream_chunks": stream_chunks,
                    "transport_error": transport_error,
                },
                "response_raw": stream_sample.decode(errors="replace"),
                "response_text": "",
                "response_sha256": stream_digest.hexdigest(),
                "cancelled_by_runner": completion == "runner_cancelled",
                "stop_reason": self.stop_reason,
                "stop_requested_monotonic_ns": self.stop_requested_monotonic_ns,
                "finished_monotonic_ns": finished_ns,
            })
            with self.record_lock:
                with self.record.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(item, sort_keys=True) + "\n")
            self.result = {
                "http_status": status,
                "completion": completion,
                "transport_error": transport_error,
                "finished_monotonic_ns": finished_ns,
            }

    def cancel(self, reason: str, timeout: float = ACTIVE_CANCEL_TIMEOUT_SECONDS) -> dict[str, Any]:
        if self.stop_requested_monotonic_ns is None:
            self.stop_reason = reason
            self.stop_requested_monotonic_ns = time.monotonic_ns()
            self.stop_event.set()
        with self.io_lock:
            response = self.response
            connection = self.connection
            sock = connection.sock if connection is not None else None
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if response is not None:
                try:
                    response.close()
                except OSError:
                    pass
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
        self.thread.join(timeout=timeout)
        return {
            "reason": reason,
            "requested_monotonic_ns": self.stop_requested_monotonic_ns,
            "finished_monotonic_ns": self.result.get("finished_monotonic_ns"),
            "thread_joined": not self.thread.is_alive(),
            "http_status": self.result.get("http_status", 0),
            "completion": self.result.get("completion", "thread_alive"),
            "transport_error": self.result.get("transport_error"),
        }


def wait_health(port: int, proc: subprocess.Popen[bytes], seconds: float = 60) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            con = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            con.request("GET", "/health")
            ready = con.getresponse().status == 200
            con.close()
            if ready:
                return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


def query_slots(port: int) -> list[dict[str, Any]] | None:
    try:
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        con.request("GET", "/slots")
        response = con.getresponse()
        raw = response.read()
        if response.status != 200:
            return None
        value = json.loads(raw) if raw else None
        return value if isinstance(value, list) else None
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        try:
            con.close()
        except (UnboundLocalError, OSError):
            pass


def normalize_runtime_claimant(value: Any, slot: int, active: bool) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    required = {
        "epoch", "exhausted", "valid", "target_blocks", "eligible_resident_blocks",
        "swapped_blocks", "shared_blocks", "blocked_blocks",
    }
    if set(value) != required:
        return None
    epoch = value.get("epoch")
    counts = {key: value.get(key) for key in required if key.endswith("_blocks")}
    if (not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 1 or
            not isinstance(value.get("exhausted"), bool) or
            not isinstance(value.get("valid"), bool) or
            any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values())):
        return None
    if value["target_blocks"] != sum(count for key, count in counts.items() if key != "target_blocks"):
        return None
    return {
        "seq_id": slot,
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


def layout_snapshot(
        slots: list[dict[str, Any]] | None,
        parallel: int,
        require_capacity: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    if not isinstance(slots, list) or len(slots) != parallel:
        return None
    roles: dict[int, bool] = {}
    claimants: dict[int, dict[str, Any]] = {}
    for slot in slots:
        if not isinstance(slot, dict):
            return None
        slot_id, processing = slot.get("id"), slot.get("is_processing")
        if not isinstance(slot_id, int) or isinstance(slot_id, bool) or not isinstance(processing, bool):
            return None
        if slot_id in roles:
            return None
        roles[slot_id] = processing
        claimant = normalize_runtime_claimant(slot.get("kv_claimant"), slot_id, processing)
        if claimant is not None:
            claimants[slot_id] = claimant
    if set(roles) != set(range(parallel)):
        return None
    active_slot = parallel - 1
    if any(roles[slot] != (slot == active_slot) for slot in range(parallel)):
        return None
    if require_capacity:
        if set(claimants) != set(range(parallel)):
            return None
        for slot in range(active_slot):
            claimant = claimants[slot]
            if (claimant["active"] or claimant["exhausted"] or not claimant["valid"] or
                    claimant["eligible_resident_blocks"] < 2):
                return None
        if not claimants[active_slot]["active"]:
            return None
    return (
        [{"id": slot, "is_processing": roles[slot]} for slot in range(parallel)],
        [claimants[slot] for slot in range(parallel)] if set(claimants) == set(range(parallel)) else [],
    )


def tokenize(port: int, content: str, add_special: bool) -> list[int]:
    encoded = json.dumps({"content": content, "add_special": add_special}, separators=(",", ":")).encode()
    con = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        con.request("POST", "/tokenize", encoded, {"Content-Type": "application/json"})
        response = con.getresponse()
        raw = response.read()
        data = json.loads(raw) if raw else {}
    finally:
        con.close()
    tokens = data.get("tokens") if response.status == 200 and isinstance(data, dict) else None
    if not isinstance(tokens, list) or not tokens or not all(isinstance(token, int) for token in tokens):
        raise RuntimeError(f"tokenization failed (status={response.status})")
    return tokens


def aligned_prompts(port: int, parallel: int) -> tuple[dict[int, list[int]], list[dict[str, int]]]:
    padding_tokens = tokenize(port, " .", False)
    pad_token = padding_tokens[-1]
    prompts: dict[int, list[int]] = {}
    records: list[dict[str, int]] = []
    for slot in range(parallel):
        tokens = tokenize(port, slot_prompt(slot), True)
        target = max(PAGED_BLOCK_SIZE * 3, ((len(tokens) + PAGED_BLOCK_SIZE - 1) // PAGED_BLOCK_SIZE) * PAGED_BLOCK_SIZE)
        prompts[slot] = tokens + [pad_token] * (target - len(tokens))
        records.append({"slot": slot, "token_count": len(tokens), "aligned_token_count": target, "block_size": PAGED_BLOCK_SIZE})
    return prompts, records


def stop(proc: subprocess.Popen[bytes], case: pathlib.Path) -> dict[str, Any]:
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
        if proc.poll() is None:
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
    residual = group_exists()
    record = {
        "pid": proc.pid,
        "pgid": pgid,
        "exit_code": proc.returncode,
        "term_timed_out": term_timed_out,
        "kill_timed_out": kill_timed_out,
        "residual_process": residual,
    }
    dump(case / "process.json", record)
    return record


def prepare_backing(case: pathlib.Path) -> pathlib.Path:
    backing = case / "backing"
    backing.mkdir()
    return backing.resolve()


def cleanup_backing(case: pathlib.Path, backing: pathlib.Path) -> dict[str, Any]:
    before = sorted(item.name for item in backing.iterdir()) if backing.is_dir() else []
    record: dict[str, Any] = {
        "environment_value": "backing",
        "path": str(backing),
        "created": True,
        "contents_before_cleanup": before,
        "cleanup_attempted": True,
    }
    try:
        shutil.rmtree(backing)
        record["cleanup_error"] = None
    except OSError as exc:
        record["cleanup_error"] = f"{type(exc).__name__}: {exc}"
    record["exists_after_cleanup"] = backing.exists()
    dump(case / "backing.json", record)
    return record


def governor_env(enabled: bool) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update({
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_INGRAPH": "1",
        "LLAMA_KV_PAGED_SWAP": "1",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
        "LLAMA_KV_PAGED_BLOCK_SIZE": str(PAGED_BLOCK_SIZE),
        "LLAMA_KV_SWAP_DIR": "backing",
        "LLAMA_KV_PRESSURE_SAMPLER": "1",
        "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "1",
        "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "100",
        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "100",
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
        binary: str,
        model: str,
        env: dict[str, str],
        case: pathlib.Path,
        port: int,
        parallel: int) -> subprocess.Popen[bytes]:
    argv = [
        binary, "--host", "127.0.0.1", "--port", str(port), "--model", model,
        "--ctx-size", "2048", "--parallel", str(parallel), "--kv-unified",
        "--no-cache-idle-slots", "--timeout", "300", "--threads", "4", "--n-gpu-layers", "0",
        "--cache-type-k", "f32", "--cache-type-v", "f32", "--no-warmup",
    ]
    dump(case / "execution.json", {
        "argv": argv,
        "cwd": str(case.resolve()),
        "environment": env,
        "binary": identity(pathlib.Path(binary)),
        "model": identity(pathlib.Path(model)),
    })
    return subprocess.Popen(
        argv,
        cwd=case,
        stdout=(case / "server.stdout").open("wb"),
        stderr=(case / "server.stderr").open("wb"),
        env=env,
        preexec_fn=os.setsid,
    )


def slot_prompt(slot: int) -> str:
    isolated_prefix = " ".join(f"slot-{slot}-isolation-{index}" for index in range(24))
    return f"{isolated_prefix} {PROMPT}"


def completion_body(prompt: list[int], slot: int, n_predict: int = N_PREDICT) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": SEED,
        "cache_prompt": True,
        "id_slot": slot,
        "stream": False,
    }


def active_completion_body(prompt: list[int], slot: int) -> dict[str, Any]:
    body = completion_body(prompt, slot, ACTIVE_N_PREDICT)
    body.update({"stream": True, "ignore_eos": True})
    return body


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


def wait_capability(case: pathlib.Path, proc: subprocess.Popen[bytes], seconds: float = 30) -> tuple[dict[str, str] | None, str | None]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        text = (case / "server.stderr").read_text(errors="replace")
        records = [fields_after_token(line, CAPABILITY_MARKER) for line in text.splitlines() if CAPABILITY_MARKER in line]
        if len(records) > 1:
            return None, "duplicate production capability records"
        if len(records) == 1:
            capability = records[0]
            if capability is None or set(capability) != CAPABILITY_FIELDS:
                return None, "malformed production capability record"
            return capability, None
        if proc.poll() is not None:
            return None, f"server exited before production capability record (exit={proc.returncode})"
        time.sleep(0.1)
    return None, "timed out waiting for production capability record"


def marker_records(
        case: pathlib.Path,
        start: int,
        stop: int | None = None) -> tuple[list[tuple[dict[str, str], int, int]], int]:
    raw = (case / "server.stderr").read_bytes()
    limit = len(raw) if stop is None else min(stop, len(raw))
    records: list[tuple[dict[str, str], int, int]] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        end = offset + len(line)
        if end > limit:
            break
        if offset >= start and MARKER.encode() in line:
            fields = fields_after_token(line.decode(errors="replace"), MARKER)
            if fields is not None:
                records.append((fields, offset, end))
        offset = end
    return records, len(raw)


def wait_marker(
        case: pathlib.Path,
        proc: subprocess.Popen[bytes],
        active: ActiveRequest,
        start: int,
        predicate: Callable[[dict[str, str]], bool],
        label: str,
        seconds: float = MARKER_TIMEOUT_SECONDS,
        reject: Callable[[dict[str, str]], str | None] | None = None) -> tuple[dict[str, str] | None, int, str | None]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        records, size = marker_records(case, start)
        for fields, _, end in records:
            if predicate(fields):
                return fields, end, None
            rejection = reject(fields) if reject is not None else None
            if rejection:
                return None, end, rejection
        if proc.poll() is not None:
            return None, size, f"server exited while waiting for {label} (exit={proc.returncode})"
        if not active.is_alive():
            return None, size, f"active request completed before {label}"
        time.sleep(0.1)
    _, size = marker_records(case, start)
    return None, size, f"timed out waiting for {label}"


def marker_is_arm(fields: dict[str, str]) -> bool:
    return (
        fields.get("release_attempted") == "1" and
        fields.get("reason") == "no_candidate" and
        fields.get("offload_armed_after") == "1"
    )


def marker_is_changed_offload(slot: int) -> Callable[[dict[str, str]], bool]:
    return lambda fields: (
        fields.get("offload_attempted") == "1" and
        fields.get("selected_seq_id") == str(slot) and
        fields.get("state_changed") == "1" and
        int(fields.get("blocks", "0")) >= 2
    )


def marker_is_exhausted_offload(slot: int) -> Callable[[dict[str, str]], bool]:
    return lambda fields: (
        fields.get("offload_attempted") == "1" and
        fields.get("selected_seq_id") == str(slot) and
        fields.get("outcome") == "no_op" and
        fields.get("reason") == "no_candidate"
    )


def marker_slot_exclusion(fields: dict[str, str], slot: int) -> str | None:
    for score in fields.get("scores", "").split(";"):
        parts = score.split(":")
        if len(parts) == 10 and parts[0] == str(slot):
            return parts[2]
    return None


def marker_has_active_slot(slot: int) -> Callable[[dict[str, str]], bool]:
    return lambda fields: marker_slot_exclusion(fields, slot) == "active"


def marker_has_eligible_slot(fields: dict[str, str], slot: int) -> bool:
    for score in fields.get("scores", "").split(";"):
        parts = score.split(":")
        if len(parts) == 10 and parts[0] == str(slot):
            return parts[1] == "1" and parts[2] == "none"
    return False


def marker_runtime_claimants(fields: dict[str, str], parallel: int) -> list[dict[str, Any]] | None:
    encoded = fields.get("claimants")
    if not encoded or encoded == "none":
        return None
    result: dict[int, dict[str, Any]] = {}
    for item in encoded.split(";"):
        parts = item.split(":")
        if len(parts) != 10 or any(not part.isdigit() for part in parts):
            return None
        seq_id, epoch, active, exhausted, valid, target, eligible, swapped, shared, blocked = map(int, parts)
        if (seq_id in result or epoch < 1 or active not in (0, 1) or exhausted not in (0, 1) or
                valid not in (0, 1) or target != eligible + swapped + shared + blocked):
            return None
        result[seq_id] = {
            "seq_id": seq_id,
            "epoch": epoch,
            "active": bool(active),
            "exhausted": bool(exhausted),
            "valid": bool(valid),
            "target_blocks": target,
            "eligible_resident_blocks": eligible,
            "swapped_blocks": swapped,
            "shared_blocks": shared,
            "blocked_blocks": blocked,
        }
    if set(result) != set(range(parallel)):
        return None
    return [result[slot] for slot in range(parallel)]


def runtime_layout_ready(claimants: list[dict[str, Any]] | None, parallel: int) -> bool:
    if not isinstance(claimants, list) or len(claimants) != parallel:
        return False
    active_slot = parallel - 1
    for slot, claimant in enumerate(claimants):
        if claimant.get("seq_id") != slot or claimant.get("active") != (slot == active_slot):
            return False
        if slot < active_slot and (
                claimant.get("exhausted") is not False or claimant.get("valid") is not True or
                int(claimant.get("eligible_resident_blocks", 0)) < 2):
            return False
    return True


def marker_is_any_offload(fields: dict[str, str]) -> bool:
    return fields.get("offload_attempted") == "1"


def first_a_offload_rejection(fields: dict[str, str]) -> str | None:
    if not marker_is_any_offload(fields):
        return None
    if fields.get("selected_seq_id") != "0":
        return "wrong claimant OFFLOAD occurred before claimant A"
    if fields.get("outcome") == "no_op" and fields.get("reason") == "no_candidate":
        return "claimant A exhausted before its required multi-block OFFLOAD"
    if fields.get("state_changed") == "1":
        if int(fields.get("blocks", "0")) < 2:
            return "first claimant A OFFLOAD completed fewer than 2 blocks"
        return None
    return "unexpected claimant A OFFLOAD ordering"


def reject_any_offload(label: str) -> Callable[[dict[str, str]], str | None]:
    return lambda fields: f"unexpected OFFLOAD before {label}" if marker_is_any_offload(fields) else None


def wait_layout_ready(
        case: pathlib.Path,
        proc: subprocess.Popen[bytes],
        active: ActiveRequest,
        port: int,
        parallel: int,
        start: int,
        governor_enabled: bool,
        seconds: float = LAYOUT_TIMEOUT_SECONDS) -> tuple[dict[str, Any] | None, str | None]:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return None, f"server exited before layout_ready (exit={proc.returncode})"
        if not active.is_alive():
            return None, "active request completed before layout_ready"

        slot_snapshot = layout_snapshot(
            query_slots(port), parallel, require_capacity=not governor_enabled)
        if not governor_enabled and slot_snapshot is not None:
            roles, claimants = slot_snapshot
            return {
                "stderr_end": (case / "server.stderr").stat().st_size,
                "evidence_marker_end": 0,
                "decision_id": 0,
                "source": "slots",
                "observed_monotonic_ns": time.monotonic_ns(),
                "slots": roles,
                "claimants": claimants,
            }, None

        records, _ = marker_records(case, start)
        for fields, begin, end in records:
            claimants = marker_runtime_claimants(fields, parallel)
            if claimants and claimants[0]["exhausted"]:
                return None, "claimant A exhausted before layout_ready"
            if runtime_layout_ready(claimants, parallel):
                rejection = first_a_offload_rejection(fields)
                if rejection:
                    return None, rejection
                roles_only = layout_snapshot(query_slots(port), parallel, require_capacity=False)
                if roles_only is not None and active.is_alive():
                    roles, _ = roles_only
                    return {
                        "stderr_end": begin,
                        "evidence_marker_end": end,
                        "decision_id": int(fields["decision_id"]),
                        "source": "governor_pre_action",
                        "observed_monotonic_ns": time.monotonic_ns(),
                        "slots": roles,
                        "claimants": claimants,
                    }, None
            if marker_is_any_offload(fields):
                return None, first_a_offload_rejection(fields) or "OFFLOAD occurred before layout_ready"
        time.sleep(0.05)
    return None, f"timed out waiting for parallel={parallel} layout_ready"


def seed_slots(
        port: int,
        prompts: dict[int, list[int]],
        slots: tuple[int, ...],
        record: pathlib.Path,
        record_lock: threading.Lock,
        statuses: dict[str, int],
        start_event: threading.Event) -> str | None:
    def run(slot: int) -> None:
        if not start_event.wait(timeout=5.0):
            statuses[f"seed_s{slot}"] = 0
            return
        statuses[f"seed_s{slot}"] = request(
            port, completion_body(prompts[slot], slot), f"seed_s{slot}", record, record_lock)

    workers = [
        threading.Thread(target=run, args=(slot,), name=f"kv-governor-seed-{slot}", daemon=True)
        for slot in slots
    ]
    for worker in workers:
        worker.start()
    start_event.set()
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS + 5.0
    for worker in workers:
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
    if any(worker.is_alive() for worker in workers):
        return "seed HTTP worker exceeded its bounded timeout"
    if any(statuses.get(f"seed_s{slot}") != 200 for slot in slots):
        return "resident claimant seed request did not complete"
    return None


def start_active_request(
        case: pathlib.Path,
        port: int,
        prompts: dict[int, list[int]],
        slot: int,
        record: pathlib.Path,
        record_lock: threading.Lock,
        start_event: threading.Event) -> tuple[ActiveRequest, int]:
    start = (case / "server.stderr").stat().st_size
    active = ActiveRequest(
        port, active_completion_body(prompts[slot], slot), f"active_s{slot}",
        record, record_lock, start_event)
    active.start()
    return active, start


def finish_active(
        case: pathlib.Path,
        active: ActiveRequest,
        active_slot: int,
        statuses: dict[str, int],
        reason: str) -> tuple[dict[str, Any], str | None]:
    stderr_start = (case / "server.stderr").stat().st_size
    record = active.cancel(reason)
    record["stderr_start"] = stderr_start
    statuses[f"active_s{active_slot}"] = int(record.get("http_status", 0))
    dump(case / "active.json", record)
    if not record["thread_joined"]:
        return record, "active HTTP worker did not stop within the bounded cancellation window"
    if record["http_status"] != 200:
        return record, "active request was not accepted before cancellation"
    if record["completion"] != "runner_cancelled":
        return record, "active request ended before runner cancellation"
    return record, None


def drive_enabled_workload(
        case: pathlib.Path,
        proc: subprocess.Popen[bytes],
        port: int,
        parallel: int,
        prompts: dict[int, list[int]],
        record: pathlib.Path,
        record_lock: threading.Lock,
        statuses: dict[str, int]) -> tuple[dict[str, Any], str | None]:
    active_slot = parallel - 1
    phases: dict[str, Any] = {"active_slot": active_slot, "resume_scopes": {}}
    start_event = threading.Event()
    active, active_start = start_active_request(
        case, port, prompts, active_slot, record, record_lock, start_event)
    phases["active_stderr_start"] = active_start
    failure: str | None = None
    gates_complete = False

    try:
        seed_error = seed_slots(
            port, prompts, tuple(range(active_slot)), record, record_lock,
            statuses, start_event)
        if seed_error:
            raise WorkloadFailure(seed_error)

        layout_ready, error = wait_layout_ready(
            case, proc, active, port, parallel, active_start, True)
        if error:
            raise WorkloadFailure(error)
        assert layout_ready is not None
        phases["layout_ready"] = layout_ready

        arm, after_arm, error = wait_marker(
            case, proc, active, active_start, marker_is_arm,
            "RELEASE full-tour no_candidate arm",
            reject=reject_any_offload("RELEASE arm"))
        if error:
            raise WorkloadFailure(error)
        phases["release_arm"] = arm

        def offload_a_when_layout_ready(fields: dict[str, str]) -> bool:
            return (marker_is_changed_offload(0)(fields) and
                    marker_has_active_slot(active_slot) and
                    (parallel == 2 or marker_has_eligible_slot(fields, 1)))

        first_a, after_first_a, error = wait_marker(
            case, proc, active, max(after_arm, int(layout_ready["stderr_end"])),
            offload_a_when_layout_ready,
            "state-changing OFFLOAD A with the intended active/idle layout",
            reject=first_a_offload_rejection)
        if error:
            raise WorkloadFailure(error)
        assert first_a is not None
        phases["offload_a"] = first_a

        if parallel == 3:
            def exhausted_a_when_layout_ready(fields: dict[str, str]) -> bool:
                return (marker_is_exhausted_offload(0)(fields) and
                        marker_has_active_slot(2) and marker_has_eligible_slot(fields, 1))

            exhausted_a, after_exhausted_a, error = wait_marker(
                case, proc, active, after_first_a, exhausted_a_when_layout_ready,
                "OFFLOAD A exhaustion while B remains idle and C active",
                reject=reject_any_offload("claimant A exhaustion"))
            if error:
                raise WorkloadFailure(error)
            phases["offload_a_exhausted"] = exhausted_a

            def offload_b_when_layout_ready(fields: dict[str, str]) -> bool:
                return (marker_is_changed_offload(1)(fields) and
                        marker_has_active_slot(2) and
                        marker_slot_exclusion(fields, 0) == "exhausted")

            offload_b, after_offload_b, error = wait_marker(
                case, proc, active, after_exhausted_a, offload_b_when_layout_ready,
                "state-changing OFFLOAD B after A exhaustion",
                reject=reject_any_offload("claimant B advance"))
            if error:
                raise WorkloadFailure(error)
            phases["offload_b"] = offload_b
            reaccess_floor = after_offload_b
        else:
            reaccess_floor = after_first_a

        reaccess_start = (case / "server.stderr").stat().st_size
        if reaccess_start < reaccess_floor:
            raise WorkloadFailure("stderr regressed before controlled reaccess")
        statuses["reaccess_a"] = request(
            port, completion_body(prompts[0], 0), "reaccess_a", record, record_lock)
        reaccess_end = (case / "server.stderr").stat().st_size
        phases["reaccess_stderr_start"] = reaccess_start
        phases["reaccess_stderr_end"] = reaccess_end
        phases["resume_scopes"]["a_initial"] = {"start": reaccess_start, "end": reaccess_end}
        if statuses["reaccess_a"] != 200:
            raise WorkloadFailure("reaccess_a did not complete")

        if parallel == 3:
            initial_epoch = int(first_a["selected_claimant_epoch"])

            def reused_a_when_layout_ready(fields: dict[str, str]) -> bool:
                return (marker_is_changed_offload(0)(fields) and
                        int(fields.get("selected_claimant_epoch", "0")) > initial_epoch and
                        marker_has_active_slot(2))

            reused, _, error = wait_marker(
                case, proc, active, reaccess_end, reused_a_when_layout_ready,
                "reused A OFFLOAD with a new epoch",
                reject=reject_any_offload("reused claimant A OFFLOAD"))
            if error:
                raise WorkloadFailure(error)
            phases["offload_a_reused"] = reused

            reaccess_b_start = (case / "server.stderr").stat().st_size
            statuses["reaccess_b"] = request(
                port, completion_body(prompts[1], 1), "reaccess_b", record, record_lock)
            reaccess_b_end = (case / "server.stderr").stat().st_size
            phases["resume_scopes"]["b"] = {"start": reaccess_b_start, "end": reaccess_b_end}
            if statuses["reaccess_b"] != 200:
                raise WorkloadFailure("reaccess_b did not complete")

            reused_a_start = (case / "server.stderr").stat().st_size
            statuses["reuse_a"] = request(
                port, completion_body(prompts[0], 0), "reuse_a", record, record_lock)
            reused_a_end = (case / "server.stderr").stat().st_size
            phases["resume_scopes"]["a_reused"] = {"start": reused_a_start, "end": reused_a_end}
            if statuses["reuse_a"] != 200:
                raise WorkloadFailure("reuse_a did not complete")
        gates_complete = True
    except WorkloadFailure as exc:
        failure = str(exc)
    finally:
        active_record, active_error = finish_active(
            case, active, active_slot, statuses,
            "gate_complete" if gates_complete else "workload_abort")
        phases["active_stop"] = active_record
        if failure is None:
            failure = active_error
        elif active_error:
            failure = f"{failure}; {active_error}"
    return phases, failure


def drive_off_workload(
        case: pathlib.Path,
        proc: subprocess.Popen[bytes],
        port: int,
        parallel: int,
        prompts: dict[int, list[int]],
        record: pathlib.Path,
        record_lock: threading.Lock,
        statuses: dict[str, int]) -> tuple[dict[str, Any], str | None]:
    active_slot = parallel - 1
    phases: dict[str, Any] = {"active_slot": active_slot, "resume_scopes": {}}
    start_event = threading.Event()
    active, active_start = start_active_request(
        case, port, prompts, active_slot, record, record_lock, start_event)
    phases["active_stderr_start"] = active_start
    failure: str | None = None
    gates_complete = False

    try:
        seed_error = seed_slots(
            port, prompts, tuple(range(active_slot)), record, record_lock,
            statuses, start_event)
        if seed_error:
            raise WorkloadFailure(seed_error)
        layout_ready, error = wait_layout_ready(
            case, proc, active, port, parallel, active_start, False)
        if error:
            raise WorkloadFailure(error)
        assert layout_ready is not None
        phases["layout_ready"] = layout_ready

        reaccess_start = (case / "server.stderr").stat().st_size
        statuses["reaccess_a"] = request(
            port, completion_body(prompts[0], 0), "reaccess_a", record, record_lock)
        reaccess_end = (case / "server.stderr").stat().st_size
        phases["reaccess_stderr_start"] = reaccess_start
        phases["reaccess_stderr_end"] = reaccess_end
        phases["resume_scopes"]["a_initial"] = {"start": reaccess_start, "end": reaccess_end}
        if statuses["reaccess_a"] != 200:
            raise WorkloadFailure("reaccess_a did not complete")

        if parallel == 3:
            reaccess_b_start = (case / "server.stderr").stat().st_size
            statuses["reaccess_b"] = request(
                port, completion_body(prompts[1], 1), "reaccess_b", record, record_lock)
            reaccess_b_end = (case / "server.stderr").stat().st_size
            phases["resume_scopes"]["b"] = {"start": reaccess_b_start, "end": reaccess_b_end}
            if statuses["reaccess_b"] != 200:
                raise WorkloadFailure("reaccess_b did not complete")

            reused_a_start = (case / "server.stderr").stat().st_size
            statuses["reuse_a"] = request(
                port, completion_body(prompts[0], 0), "reuse_a", record, record_lock)
            reused_a_end = (case / "server.stderr").stat().st_size
            phases["resume_scopes"]["a_reused"] = {"start": reused_a_start, "end": reused_a_end}
            if statuses["reuse_a"] != 200:
                raise WorkloadFailure("reuse_a did not complete")
        gates_complete = True
    except WorkloadFailure as exc:
        failure = str(exc)
    finally:
        active_record, active_error = finish_active(
            case, active, active_slot, statuses,
            "gate_complete" if gates_complete else "workload_abort")
        phases["active_stop"] = active_record
        if failure is None:
            failure = active_error
        elif active_error:
            failure = f"{failure}; {active_error}"
    return phases, failure


def run_case(name: str, binary: str, model: str, enabled: bool, root: pathlib.Path, parallel: int) -> dict[str, Any]:
    case = root / name
    case.mkdir()
    backing = prepare_backing(case)
    env, record, port = governor_env(enabled), case / "requests.jsonl", free_port()
    dump(case / "environment.json", env)
    result: dict[str, Any] = {"status": "startup_failed", "port": port, "request_loop_started": False}
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = start(binary, model, env, case, port, parallel)
        if not wait_health(port, proc):
            result["startup_reason"] = f"listener was not healthy (exit={proc.poll()})"
            return result
        capability, capability_error = wait_capability(case, proc)
        if capability_error:
            result.update({"status": "unsupported", "unsupported_reason": capability_error})
            return result
        unsupported = {key: capability[key] for key in REQUIRED_CAPABILITY if capability[key] != "1"}
        if unsupported:
            result.update({
                "status": "unsupported",
                "unsupported_reason": f"production capability unavailable: {unsupported}",
                "capability": capability,
            })
            return result
        try:
            prompts, alignment = aligned_prompts(port, parallel)
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            result.update({
                "status": "startup_failed",
                "startup_reason": f"could not construct block-aligned prompts: {type(exc).__name__}: {exc}",
                "capability": capability,
            })
            return result
        dump(case / "workload.json", {"prompt_alignment": alignment, "block_size": PAGED_BLOCK_SIZE})

        statuses: dict[str, int] = {}
        record_lock = threading.Lock()
        if enabled:
            phases, workload_error = drive_enabled_workload(case, proc, port, parallel, prompts, record, record_lock, statuses)
        else:
            phases, workload_error = drive_off_workload(
                case, proc, port, parallel, prompts, record, record_lock, statuses)
        result.update({
            "status": "complete" if workload_error is None and all(status == 200 for status in statuses.values()) else "request_failed",
            "http_statuses": statuses,
            "capability": capability,
            "phases": phases,
            "request_loop_started": True,
        })
        if workload_error:
            result["workload_error"] = workload_error
        if "active_stderr_start" in phases:
            result["active_stderr_start"] = phases["active_stderr_start"]
        if "layout_ready" in phases:
            result["layout_ready"] = phases["layout_ready"]
        if "active_stop" in phases:
            result["active_stop"] = phases["active_stop"]
        if "resume_scopes" in phases:
            result["resume_scopes"] = phases["resume_scopes"]
        if "reaccess_stderr_start" in phases:
            result["reaccess_stderr_start"] = phases["reaccess_stderr_start"]
            result["reaccess_stderr_end"] = phases["reaccess_stderr_end"]
        return result
    except RunnerInterrupted as exc:
        result.update({
            "status": "interrupted",
            "interruption": {"signal": exc.signum, "message": str(exc)},
        })
        return result
    except OSError as exc:
        result.update({"status": "startup_failed", "startup_reason": f"server exec failed: {type(exc).__name__}: {exc}"})
        return result
    except Exception as exc:
        result.update({
            "status": "request_failed",
            "workload_error": f"{type(exc).__name__}: {exc}",
        })
        return result
    finally:
        active_path = case / "active.json"
        if active_path.is_file() and "active_stop" not in result:
            try:
                result["active_stop"] = json.loads(active_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                result["active_stop_error"] = f"{type(exc).__name__}: {exc}"
                result["status"] = "request_failed"
        if proc is not None:
            process_record = stop(proc, case)
            result["process"] = process_record
            if process_record["residual_process"] or process_record["kill_timed_out"]:
                result["status"] = "request_failed"
                result["residual_process"] = True
        backing_record = cleanup_backing(case, backing)
        result["backing"] = backing_record
        if backing_record["exists_after_cleanup"] or backing_record["cleanup_error"]:
            result["status"] = "request_failed"
            result["backing_cleanup_failed"] = True
        dump(case / "result.json", result)


def run_negative(name: str, binary: str, model: str, extra: dict[str, str], root: pathlib.Path, parallel: int) -> dict[str, Any]:
    case = root / name
    case.mkdir()
    backing = prepare_backing(case)
    env = governor_env(True)
    env.update(extra)
    dump(case / "environment.json", env)
    port = free_port()
    result: dict[str, Any] = {"status": "complete", "port": port, "request_loop_started": False}
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = start(binary, model, env, case, port, parallel)
        health = wait_health(port, proc, seconds=3)
        try:
            proc.wait(timeout=SERVER_TERM_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            result["status"] = "request_failed"
        result.update({
            "health_reached": health,
            "exit_code": proc.returncode,
            "rejected_before_request_loop": not health and proc.returncode not in (None, 0),
        })
        return result
    except RunnerInterrupted as exc:
        result.update({
            "status": "interrupted",
            "interruption": {"signal": exc.signum, "message": str(exc)},
            "health_reached": False,
            "exit_code": proc.returncode if proc is not None else None,
            "rejected_before_request_loop": True,
        })
        return result
    except OSError as exc:
        result.update({
            "status": "startup_failed",
            "startup_reason": f"server exec failed: {type(exc).__name__}: {exc}",
            "health_reached": False,
            "exit_code": None,
            "rejected_before_request_loop": True,
        })
        return result
    except Exception as exc:
        result.update({
            "status": "request_failed",
            "startup_reason": f"{type(exc).__name__}: {exc}",
            "health_reached": False,
            "exit_code": proc.returncode if proc is not None else None,
            "rejected_before_request_loop": True,
        })
        return result
    finally:
        if proc is not None:
            process_record = stop(proc, case)
            result["process"] = process_record
            if process_record["residual_process"] or process_record["kill_timed_out"]:
                result["status"] = "request_failed"
                result["residual_process"] = True
        backing_record = cleanup_backing(case, backing)
        result["backing"] = backing_record
        if backing_record["exists_after_cleanup"] or backing_record["cleanup_error"]:
            result["status"] = "request_failed"
            result["backing_cleanup_failed"] = True
        dump(case / "result.json", result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=os.environ.get("KV_GOVERNOR_BINARY", "build/bin/llama-server"))
    ap.add_argument("--model", default=os.environ.get("KV_GOVERNOR_MODEL"))
    ap.add_argument("--output-dir")
    args = ap.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(f"/root/oscomp/kv_logs/{PROTOCOL}_{stamp}_{uuid.uuid4().hex[:10]}")
    if out.exists():
        raise SystemExit(f"refusing existing artifact directory: {out}")
    out.mkdir(parents=True)
    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).resolve() if args.model else None
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL,
        "protocol_version": PROTOCOL_VERSION,
        "timestamp_utc": stamp,
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "dirty_status": git("status", "--porcelain").splitlines(),
        "runner": identity(pathlib.Path(__file__)),
        "parser": identity(ROOT / "scripts/parse-kv-governor-stage3c-1c-2b-1r.py"),
        "binary_requested": str(binary),
        "model_requested": str(model) if model else None,
        "parameters": {
            "parallels": list(PARALLELS),
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "seed": SEED,
            "n_predict": N_PREDICT,
            "active_n_predict": ACTIVE_N_PREDICT,
            "temperature": 0.0,
            "kv_unified": True,
            "paged_block_size": PAGED_BLOCK_SIZE,
            "governor_target_bytes": GOVERNOR_TARGET_BYTES,
            "governor_max_blocks": GOVERNOR_MAX_BLOCKS,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "active_socket_timeout_seconds": ACTIVE_SOCKET_TIMEOUT_SECONDS,
            "active_cancel_timeout_seconds": ACTIVE_CANCEL_TIMEOUT_SECONDS,
            "marker_timeout_seconds": MARKER_TIMEOUT_SECONDS,
            "layout_timeout_seconds": LAYOUT_TIMEOUT_SECONDS,
            "server_timeout_seconds": 300,
            "server_term_timeout_seconds": SERVER_TERM_TIMEOUT_SECONDS,
            "server_kill_timeout_seconds": SERVER_KILL_TIMEOUT_SECONDS,
        },
        "runner_status": "run_in_progress",
    }
    if not binary.is_file() or model is None or not model.is_file():
        manifest.update({
            "runner_status": "UNSUPPORTED",
            "unsupported_reason": "binary or model is unavailable; no real server/model evidence was fabricated",
        })
        if binary.is_file():
            manifest["binary"] = identity(binary)
        dump(out / "manifest.json", manifest)
        raise SystemExit(3)
    manifest.update({"binary": identity(binary), "model": identity(model)})
    dump(out / "manifest.json", manifest)

    signal_seen: list[int] = []

    def handle_signal(signum: int, _frame: Any) -> None:
        if signal_seen:
            return
        signal_seen.append(signum)
        raise RunnerInterrupted(signum)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    runs: dict[str, Any] = {}
    incomplete = False
    unsupported: list[str] = []
    interruption: dict[str, Any] | None = None
    try:
        negatives = (
            ("INVALID_UNIFIED", {"LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "invalid"}),
            ("CONFLICT_UNIFIED_LEGACY", {"LLAMA_KV_PAGED_RELEASE": "1"}),
            ("CONFLICT_UNIFIED_DRY_RUN", {"LLAMA_KV_PRESSURE_DRY_RUN": "1"}),
            ("CONFLICT_UNIFIED_BOUNDED", {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"}),
        )
        for parallel in PARALLELS:
            root = out / f"parallel_{parallel}"
            root.mkdir()
            cases: dict[str, Any] = {}
            for name, enabled in (("OFF", False), ("GOVERNOR_ON", True)):
                case_result = run_case(name, str(binary), str(model), enabled, root, parallel)
                cases[name] = case_result
                if case_result.get("status") == "interrupted":
                    interruption = case_result.get("interruption", {"message": "interrupted"})
                    break
                if case_result.get("status") == "unsupported":
                    unsupported.append(str(case_result.get("unsupported_reason", "unspecified")))
                    break
            if not interruption and not unsupported:
                for name, extra in negatives:
                    case_result = run_negative(name, str(binary), str(model), extra, root, parallel)
                    cases[name] = case_result
                    if case_result.get("status") == "interrupted":
                        interruption = case_result.get("interruption", {"message": "interrupted"})
                        break
            incomplete |= any(case.get("status") != "complete" for case in cases.values())
            runs[str(parallel)] = {"parallel": parallel, "cases": cases}
            if interruption or unsupported:
                break
    except RunnerInterrupted as exc:
        interruption = {"signal": exc.signum, "message": str(exc)}
    except Exception as exc:
        incomplete = True
        manifest["runner_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if interruption:
            manifest.update({
                "runs": runs,
                "runner_status": "run_interrupted",
                "interruption": interruption,
            })
        elif unsupported:
            manifest.update({
                "runs": runs,
                "runner_status": "UNSUPPORTED",
                "unsupported_reason": "; ".join(unsupported),
            })
        else:
            manifest.update({
                "runs": runs,
                "runner_status": "run_incomplete" if incomplete else "run_complete",
            })
        manifest["finished_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dump(out / "manifest.json", manifest)

    if interruption:
        raise SystemExit(128 + int(interruption.get("signal", 0) or 0))
    if unsupported:
        raise SystemExit(3)
    parser = ROOT / "scripts/parse-kv-governor-stage3c-1c-2b-1r.py"
    raise SystemExit(subprocess.run(
        [sys.executable, str(parser), str(out), "--result-path", str(out / "parser.json")],
        text=True).returncode)


if __name__ == "__main__":
    main()
