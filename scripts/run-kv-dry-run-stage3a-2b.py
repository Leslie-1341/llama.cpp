#!/usr/bin/env python3
"""Stage 3A-2B: real-server dry-run OFF/ON controlled validation.

OFF/ON share the same binary, model, parameters, request, random seed, and
observation window.  Only LLAMA_KV_PRESSURE_DRY_RUN differs.  Neither case
sets LLAMA_KV_PAGED_RELEASE — dry-run is decoupled from destructive release.

The 1/2/3 KiB RSS thresholds intentionally force PRESSURE/CRITICAL state
transitions and are FORCED_LIFECYCLE_STATE_VALIDATION_ONLY, not deployment
recommendations.

Strace is attached *after* server health-ready so early startup syscalls are
excluded.  The ON case must show zero MADV_DONTNEED calls.

Protocol:  ./run-kv-dry-run-stage3a-2b.py --binary build/bin/llama-server
           --model <path> [--output-dir /path/to/artifact]
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import pathlib
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, NoReturn

ROOT = pathlib.Path(__file__).resolve().parents[1]

BASE_ENV: dict[str, str] = {
    "HOME": "/tmp", "LANG": "C", "LC_ALL": "C",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "TMPDIR": "/tmp", "TZ": "UTC",
}

# Explicitly disable destructive release — dry-run is decoupled.
FORBIDDEN_ENV: dict[str, str] = {
    "LLAMA_KV_PAGED_RELEASE": "0",
    "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_SWAP": "0",
}

SHARED_ENV: dict[str, str] = {
    "LLAMA_KV_PAGED": "1",
    "LLAMA_KV_PRESSURE_SAMPLER": "1",
    # Forced thresholds: trigger PRESSURE/CRITICAL immediately (RSS always > 3 KiB).
    "LLAMA_KV_LOW_WATER_RSS_KB": "1",
    "LLAMA_KV_PRESSURE_RSS_KB": "2",
    "LLAMA_KV_CRITICAL_RSS_KB": "3",
    # Short interval so the sampler fires during the request window.
    "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "500",
    "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
}

ON_ENV: dict[str, str] = {
    "LLAMA_KV_PRESSURE_DRY_RUN": "1",
    "LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES": "4194304",
    "LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS": "64",
    "LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS": "500",
}

THRESHOLD_PURPOSE = (
    "FORCED_LIFECYCLE_STATE_VALIDATION_ONLY_NOT_REAL_DEPLOYMENT_THRESHOLDS"
)

PROMPT = "In one short sentence, explain why deterministic tests are useful."
N_PREDICT = 32
SEED = 1

TIMEOUTS_S = {
    "startup": 5.0, "health": 180.0, "completion": 120.0,
    "shutdown": 15.0, "case_total": 360.0, "post_attach_wait": 0.75,
}


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(args_list: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args_list], text=True).strip()


def assert_repo(head: str) -> dict[str, Any]:
    current = git(["rev-parse", "HEAD"])
    status = git(["status", "--porcelain"])
    info: dict[str, Any] = {"head_matches": current == head}
    if current != head:
        fail("HEAD changed during the protocol")
    if status:
        info["dirty"] = True
        info["dirty_files"] = status.split("\n")
    else:
        info["dirty"] = False
    return info


class Deadline:
    def __init__(self, seconds: float):
        self.end = time.monotonic() + seconds

    def remaining(self, phase_limit: float | None = None) -> float:
        value = self.end - time.monotonic()
        if phase_limit is not None:
            value = min(value, phase_limit)
        if value <= 0:
            raise TimeoutError("case total deadline expired")
        return value


def free_port(used: set[int]) -> int:
    for _ in range(100):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        if port not in used:
            used.add(port)
            return port
    fail("could not allocate a unique loopback port")


def request(port: int, method: str, path: str, deadline: Deadline,
            phase_remaining: float, body: Any | None = None) -> tuple[int, dict[str, Any]]:
    encoded = None if body is None else json.dumps(body).encode()
    headers = {} if encoded is None else {"Content-Type": "application/json"}
    conn = http.client.HTTPConnection(
        "127.0.0.1", port, timeout=deadline.remaining(phase_remaining))
    try:
        conn.request(method, path, encoded, headers)
        response = conn.getresponse()
        payload = response.read()
        parsed = json.loads(payload) if payload else {}
        if not isinstance(parsed, dict):
            fail(f"{path} returned a non-object JSON body")
        return response.status, parsed
    finally:
        conn.close()


def stream_completion(port: int, raw_path: pathlib.Path, evidence_path: pathlib.Path,
                      deadline: Deadline, phase_seconds: float) -> dict[str, Any]:
    body = json.dumps({
        "prompt": PROMPT, "n_predict": N_PREDICT, "stream": True,
        "seed": SEED, "temperature": 0.0, "cache_prompt": False,
    }).encode()
    phase_end = time.monotonic() + min(phase_seconds, deadline.remaining())
    started = time.monotonic_ns()
    conn = http.client.HTTPConnection("127.0.0.1", port,
                                      timeout=max(0.001, phase_end - time.monotonic()))
    raw = bytearray()
    records: list[dict[str, Any]] = []
    http_status: int | None = None
    try:
        conn.request("POST", "/completion", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        http_status = response.status
        if response.status != 200:
            payload = response.read()
            fail(f"completion returned HTTP {response.status}: {payload[:200]!r}")
        while True:
            remaining = min(phase_end - time.monotonic(), deadline.remaining())
            if remaining <= 0:
                raise TimeoutError("completion deadline expired")
            if conn.sock is not None:
                conn.sock.settimeout(remaining)
            line = response.readline()
            if not line:
                break
            raw.extend(line)
            stripped = line.rstrip(b"\r\n")
            if not stripped:
                continue
            if not stripped.startswith(b"data: "):
                fail("completion stream contains a non-SSE data line")
            payload = stripped[6:]
            try:
                event = json.loads(payload)
            except (UnicodeDecodeError, ValueError) as exc:
                fail(f"completion stream contains malformed JSON: {exc}")
            if not isinstance(event, dict):
                fail("completion SSE payload is not an object")
            records.append({
                "raw_payload": payload.decode("utf-8"),
                "arrival_monotonic_ns": time.monotonic_ns(),
                "content": event.get("content", ""),
            })
    finally:
        conn.close()
        raw_path.write_bytes(raw)
        write_json(evidence_path, {
            "request_started_monotonic_ns": started, "http_status": http_status,
            "events": records, "stream_ended_monotonic_ns": time.monotonic_ns(),
            "clock": "time.monotonic_ns",
        })
    # Extract response text for cross-case comparison.
    response_text = "".join(e.get("content", "") for e in records)
    return {
        "response_text": response_text,
        "response_length": len(response_text),
        "chunk_count": len(records),
        "http_status": http_status,
    }


def wait_health(proc: subprocess.Popen[bytes], port: int, deadline: Deadline) -> dict[str, Any]:
    phase_end = time.monotonic() + min(TIMEOUTS_S["health"], deadline.remaining())
    last = "not attempted"
    while time.monotonic() < phase_end:
        if proc.poll() is not None:
            fail(f"server exited during health wait with status {proc.returncode}")
        try:
            remaining = phase_end - time.monotonic()
            status, payload = request(port, "GET", "/health", deadline, min(1.0, remaining))
            if status == 200 and payload.get("status") == "ok":
                return {"status_code": status, "body": payload}
            last = f"HTTP {status}: {payload!r}"
        except (OSError, ValueError, TimeoutError) as exc:
            last = str(exc)
        time.sleep(min(0.1, max(0.0, phase_end - time.monotonic())))
    raise TimeoutError(f"server health timeout: {last}")


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate(proc: subprocess.Popen[bytes], deadline: Deadline) -> dict[str, Any]:
    pgid = proc.pid
    requested = process_group_alive(pgid)
    if requested:
        os.killpg(pgid, signal.SIGTERM)
    killed = False
    try:
        returncode = proc.wait(timeout=deadline.remaining(10.0))
    except subprocess.TimeoutExpired:
        killed = True
        if process_group_alive(pgid):
            os.killpg(pgid, signal.SIGKILL)
        try:
            returncode = proc.wait(timeout=deadline.remaining(5.0))
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("server leader was not reaped after SIGKILL") from exc
    if process_group_alive(pgid):
        killed = True
        os.killpg(pgid, signal.SIGKILL)
    kill_end = time.monotonic() + min(1.0, deadline.remaining())
    while process_group_alive(pgid) and time.monotonic() < kill_end:
        time.sleep(min(0.05, max(0.0, kill_end - time.monotonic())))
    residual = process_group_alive(pgid)
    if residual:
        raise TimeoutError("server process group remains after SIGKILL")
    return {"returncode": returncode, "shutdown_requested": requested,
            "sigkill_used": killed, "residual_process": False, "unexpected_exit": False}


def stop_attached_strace(trace_proc: subprocess.Popen[bytes], deadline: Deadline) -> dict[str, Any]:
    requested = trace_proc.poll() is None
    if requested:
        trace_proc.send_signal(signal.SIGINT)
    try:
        code = trace_proc.wait(timeout=deadline.remaining(5.0))
    except subprocess.TimeoutExpired:
        trace_proc.kill()
        try:
            code = trace_proc.wait(timeout=deadline.remaining(5.0))
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("strace did not exit after SIGKILL") from exc
    return {"shutdown_requested": requested, "returncode": code,
            "residual_process": trace_proc.poll() is None}


def attach_strace(strace_bin: pathlib.Path, proc: subprocess.Popen[bytes],
                  prefix: pathlib.Path, deadline: Deadline,
                  trace_syscalls: str = "madvise,openat,read") -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    argv = [str(strace_bin), "-ff", "-qq", "-yy", "-s", "4096",
            "-e", f"trace={trace_syscalls}", "-o", str(prefix), "-p", str(proc.pid)]
    attached = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    attach_end = time.monotonic() + min(5.0, deadline.remaining())
    while time.monotonic() < attach_end:
        if attached.poll() is not None:
            error = (attached.stderr.read() if attached.stderr else b"").decode(errors="replace")
            fail(f"strace attach failed: {error.strip()}")
        if list(prefix.parent.glob(prefix.name + ".*")):
            time.sleep(min(0.1, deadline.remaining()))
            if attached.poll() is not None:
                error = (attached.stderr.read() if attached.stderr else b"").decode(errors="replace")
                fail(f"strace attach failed: {error.strip()}")
            return attached, {"argv": argv, "attached_to_pid": proc.pid,
                              "attach_after_health": True,
                              "attach_monotonic_ns": time.monotonic_ns()}
        time.sleep(0.05)
    attached.kill()
    attached.wait(timeout=deadline.remaining(1.0))
    raise TimeoutError("strace attach timeout")


def build_env(variant: str) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update(FORBIDDEN_ENV)
    env.update(SHARED_ENV)
    if variant == "ON":
        env.update(ON_ENV)
    # OFF: no LLAMA_KV_PRESSURE_DRY_RUN set at all
    return env


def server_argv(binary: pathlib.Path, model: pathlib.Path, port: int) -> list[str]:
    return [
        str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", "1024", "--batch-size", "128", "--ubatch-size", "128",
        "--parallel", "1", "--seed", str(SEED), "--temp", "0", "--no-warmup",
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
    ]


def execute_case(case_dir: pathlib.Path, variant: str,
                 binary_id: dict[str, Any], model_id: dict[str, Any],
                 head: str, strace_bin: pathlib.Path,
                 used_ports: set[int]) -> dict[str, Any]:
    case_started_ns = time.monotonic_ns()
    deadline = Deadline(TIMEOUTS_S["case_total"])
    proc: subprocess.Popen[bytes] | None = None
    trace_proc: subprocess.Popen[bytes] | None = None
    repo_info = assert_repo(head)
    port = free_port(used_ports)
    env = build_env(variant)
    argv = server_argv(pathlib.Path(binary_id["path"]), pathlib.Path(model_id["path"]), port)
    has_strace = variant == "ON"

    write_json(case_dir / "execution.json", {
        "argv": argv, "cwd": str(ROOT), "env_keys_sorted": sorted(env.keys()),
        "port": port, "strace_attach_after_health": has_strace,
        "threshold_purpose": THRESHOLD_PURPOSE,
        "repo": repo_info,
    })
    write_json(case_dir / "environment.json", env)
    stdout_path = case_dir / "server.stdout"
    stderr_path = case_dir / "server.stderr"

    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        startup_ns = time.monotonic_ns()
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=True)
        if time.monotonic_ns() - startup_ns > TIMEOUTS_S["startup"] * 1e9:
            raise TimeoutError("startup process creation timeout")
        write_json(case_dir / "process.json", {
            "pid": proc.pid, "server_argv": argv, "env": env,
            "exe": os.readlink(f"/proc/{proc.pid}/exe"),
            "expected_exe": str(pathlib.Path(binary_id["path"]).resolve()),
        })

        health_ns = time.monotonic_ns()
        health = wait_health(proc, port, deadline)
        write_json(case_dir / "health.json", health)

        # Post-health wait: let the sampler observe RSS before the request.
        # With 500ms sample interval, 750ms gives ~1-2 samples.
        post_wait_ms = int(TIMEOUTS_S["post_attach_wait"] * 1000)
        post_wait_start = time.monotonic_ns()
        time.sleep(TIMEOUTS_S["post_attach_wait"])
        post_wait_end = time.monotonic_ns()

        if has_strace:
            trace_proc, trace_info = attach_strace(strace_bin, proc, case_dir / "strace", deadline)
            write_json(case_dir / "strace_process.json", trace_info)
            # Additional wait after strace attach to ensure at least one sample is observed.
            time.sleep(TIMEOUTS_S["post_attach_wait"])
            trace_info["post_attach_wait_requested_ms"] = post_wait_ms
            trace_info["post_attach_wait_actual_ms"] = (time.monotonic_ns() - post_wait_end) / 1e6
            write_json(case_dir / "strace_process.json", trace_info)

        # Snapshot stderr position before the request for windowing.
        stderr.flush()
        log_start = stderr_path.stat().st_size

        completion_ns = time.monotonic_ns()
        raw_path = case_dir / "completion.sse"
        events_path = case_dir / "completion.events.json"
        response_info = stream_completion(port, raw_path, events_path, deadline,
                                          TIMEOUTS_S["completion"])

        stderr.flush()
        log_end = stderr_path.stat().st_size
        write_json(case_dir / "completion_window.json", {
            "stderr_start": log_start, "stderr_end": log_end,
            "post_health_wait_start_ns": post_wait_start,
            "post_health_wait_end_ns": post_wait_end,
        })

        if trace_proc is not None:
            trace_info.update(stop_attached_strace(trace_proc, deadline))
            trace_proc = None
            write_json(case_dir / "strace_process.json", trace_info)

        if proc.poll() is not None:
            fail(f"server exited before shutdown: {proc.returncode}")

        shutdown_ns = time.monotonic_ns()
        proc_info = terminate(proc, deadline)
        proc = None

    write_json(case_dir / "result.json", {
        "variant": variant, "response_text": response_info["response_text"],
        "response_length": response_info["response_length"],
        "chunk_count": response_info["chunk_count"],
        "http_status": response_info["http_status"],
        "started_monotonic_ns": case_started_ns,
        "ended_monotonic_ns": time.monotonic_ns(),
    })
    write_json(case_dir / "phases.json", {
        "startup": {"started_ns": startup_ns, "status": "PASS"},
        "health": {"started_ns": health_ns, "status": "PASS"},
        "completion": {"started_ns": completion_ns, "status": "PASS"},
        "shutdown": {"started_ns": shutdown_ns, "status": "PASS", **proc_info},
    })
    return response_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3A-2B dry-run OFF/ON runner")
    parser.add_argument("--binary", required=True, help="Path to llama-server binary")
    parser.add_argument("--model", required=True, help="Path to GGUF model file")
    parser.add_argument("--strace", default="/usr/bin/strace", help="Path to strace binary")
    parser.add_argument("--output-dir", help="Artifact output directory (auto-generated if omitted)")
    args = parser.parse_args()

    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).resolve()
    strace_bin = pathlib.Path(args.strace).resolve()

    if not binary.is_file():
        fail(f"binary not found: {binary}")
    if not model.is_file():
        fail(f"model not found: {model}")
    if not strace_bin.is_file():
        fail(f"strace not found: {strace_bin}")
    if platform.system() != "Linux":
        fail("this protocol requires Linux (/proc, strace, madvise)")

    head = git(["rev-parse", "HEAD"])
    binary_id = identity(binary)
    model_id = identity(model)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short_head = head[:10]
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(
        f"/root/oscomp/kv_logs/kv_dry_run_stage3a_2b_{ts}_{short_head}")
    if out_dir.exists():
        fail(f"output directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "protocol": "stage3a-2b-dry-run-off-on",
        "timestamp_utc": ts,
        "head_sha": head,
        "binary": binary_id,
        "model": model_id,
        "threshold_purpose": THRESHOLD_PURPOSE,
        "prompt": PROMPT, "n_predict": N_PREDICT, "seed": SEED,
        "cases": [],
        "parser": str(ROOT / "scripts/parse-kv-dry-run-stage3a-2b.py"),
    }

    used_ports: set[int] = set()

    for variant in ("OFF", "ON"):
        case_name = f"dry_run_{variant.lower()}"
        case_dir = out_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=False)
        print(f"=== {case_name} ===", flush=True)
        try:
            response_info = execute_case(case_dir, variant, binary_id, model_id,
                                         head, strace_bin, used_ports)
            manifest["cases"].append({
                "name": case_name, "variant": variant,
                "response_text": response_info["response_text"],
                "response_length": response_info["response_length"],
                "status": "COMPLETED",
            })
            print(f"  response: {response_info['response_text'][:60]}...", flush=True)
        except Exception as exc:
            manifest["cases"].append({
                "name": case_name, "variant": variant,
                "status": "FAILED", "error": str(exc),
            })
            print(f"  FAILED: {exc}", flush=True)

    # Run parser
    parser_path = ROOT / "scripts/parse-kv-dry-run-stage3a-2b.py"
    parser_cmd = [sys.executable, str(parser_path), str(out_dir)]
    parser_result = subprocess.run(parser_cmd, capture_output=True, text=True)
    manifest["parser_exit_code"] = parser_result.returncode
    manifest["parser_stdout"] = parser_result.stdout.strip().split("\n")[-5:]
    if parser_result.stderr:
        manifest["parser_stderr_tail"] = parser_result.stderr.strip().split("\n")[-10:]

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "summary.json", {
        "artifact": str(out_dir),
        "head_sha": head,
        "cases": [c["name"] for c in manifest["cases"]],
        "parser_exit_code": parser_result.returncode,
        "verdict": "PASS" if parser_result.returncode == 0 else "FAIL",
    })

    print(f"\nArtifact: {out_dir}", flush=True)
    print(f"Parser exit: {parser_result.returncode}", flush=True)
    if parser_result.stdout:
        print(parser_result.stdout, flush=True)
    if parser_result.returncode != 0:
        raise SystemExit(parser_result.returncode)


if __name__ == "__main__":
    main()
