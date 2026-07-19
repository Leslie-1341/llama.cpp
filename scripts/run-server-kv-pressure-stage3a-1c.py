#!/usr/bin/env python3
"""Stage 3A-1C clean-HEAD real-server read-only validation runner.

The 1/2/3 KiB RSS settings intentionally force pressure-state lifecycle
transitions.  They are not deployment thresholds and must not be interpreted as
empirical pressure limits.  This runner never builds the server or finds a model.
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
from typing import Any, NoReturn


ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts/parse-server-kv-pressure-stage3a-1c.py"
PLAN = (
    ("ab_r1_off", "ab", 1, 1, "OFF"), ("ab_r1_on", "ab", 1, 2, "ON"),
    ("ab_r2_on", "ab", 2, 1, "ON"), ("ab_r2_off", "ab", 2, 2, "OFF"),
    ("ab_r3_off", "ab", 3, 1, "OFF"), ("ab_r3_on", "ab", 3, 2, "ON"),
    ("on_lifecycle", "lifecycle", 0, 0, "ON"),
    ("on_idle_250ms", "idle_limit", 0, 0, "ON"),
    ("strace_off_correctness", "strace", 0, 1, "OFF"),
    ("strace_on_correctness", "strace", 0, 2, "ON"),
)
FORBIDDEN_ENV = {
    "LLAMA_KV_PAGED_RELEASE": "0", "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0", "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_RESUME_PREFETCH": "0", "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "0", "LLAMA_KV_SWAP": "0",
    "LLAMA_KV_SWAP_MADVISE": "0",
}
BASE_ENV = {
    "HOME": "/tmp", "LANG": "C", "LC_ALL": "C",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "TMPDIR": "/tmp", "TZ": "UTC", **FORBIDDEN_ENV,
}
ON_ENV = {
    "LLAMA_KV_PRESSURE_SAMPLER": "1",
    "LLAMA_KV_LOW_WATER_RSS_KB": "1",
    "LLAMA_KV_PRESSURE_RSS_KB": "2",
    "LLAMA_KV_CRITICAL_RSS_KB": "3",
    "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "60000",
    "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "60000",
}
PRESSURE_SETTINGS_PURPOSE = (
    "FORCED_LIFECYCLE_STATE_VALIDATION_ONLY_NOT_REAL_DEPLOYMENT_THRESHOLDS"
)
PROMPT = "In one short sentence, explain why deterministic tests are useful."
N_PREDICT = 32
TIMEOUTS_S = {
    "startup": 5.0, "health": 180.0, "completion": 180.0,
    "sleep": 30.0, "resume": 180.0, "shutdown": 15.0, "case_total": 430.0,
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


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def assert_repo(head: str) -> None:
    if git(["rev-parse", "HEAD"]) != head:
        fail("HEAD changed during the protocol")
    if git(["status", "--porcelain"]):
        fail("worktree became dirty during the protocol")


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


def phase_record(phases: dict[str, Any], name: str, started_ns: int,
                 status: str, error: str | None = None) -> None:
    record: dict[str, Any] = {
        "status": status, "started_monotonic_ns": started_ns,
        "ended_monotonic_ns": time.monotonic_ns(), "timeout_s": TIMEOUTS_S[name],
    }
    if error is not None:
        record["error"] = error
    phases[name] = record


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
        "seed": 1, "temperature": 0.0, "cache_prompt": False,
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
            })
    finally:
        conn.close()
        raw_path.write_bytes(raw)
        write_json(evidence_path, {
            "request_started_monotonic_ns": started, "http_status": http_status,
            "events": records, "stream_ended_monotonic_ns": time.monotonic_ns(),
            "clock": "time.monotonic_ns",
        })
    return derive_runner_metrics(evidence_path)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    rank = max(1, int(math.ceil(q * len(ordered))))
    return ordered[rank - 1]


def derive_runner_metrics(evidence_path: pathlib.Path) -> dict[str, Any]:
    evidence = json.loads(evidence_path.read_text())
    decoded = [json.loads(item["raw_payload"]) for item in evidence["events"]]
    nonterminal = [item for item in decoded if item.get("stop") is False]
    terminal = [item for item in decoded if item.get("stop") is True]
    if len(terminal) != 1 or decoded[-1] is not terminal[0] or not nonterminal:
        fail("completion must contain nonterminal chunks and one final terminal event")
    tokens = [event.get("tokens") for event in nonterminal]
    if any(not isinstance(group, list) or not group or
           any(not isinstance(token, int) or isinstance(token, bool) for token in group)
           for group in tokens):
        fail("completion contains an invalid token chunk")
    timings = terminal[0].get("timings")
    if not isinstance(timings, dict):
        fail("terminal completion event lacks server timings")
    predicted_n = timings.get("predicted_n")
    predicted_ms = timings.get("predicted_ms")
    if not isinstance(predicted_n, int) or isinstance(predicted_n, bool) or predicted_n <= 0:
        fail("invalid predicted_n")
    if not isinstance(predicted_ms, (int, float)) or isinstance(predicted_ms, bool) or predicted_ms <= 0:
        fail("invalid predicted_ms")
    arrivals = [item["arrival_monotonic_ns"] for item, event in zip(evidence["events"], decoded)
                if event.get("stop") is False]
    intervals = [(right - left) / 1e6 for left, right in zip(arrivals, arrivals[1:])]
    result: dict[str, Any] = {
        "ttft_ms": (arrivals[0] - evidence["request_started_monotonic_ns"]) / 1e6,
        "tpot_ms": float(predicted_ms) / predicted_n,
        "throughput_tps": predicted_n * 1000.0 / float(predicted_ms),
        "predicted_n": predicted_n, "valid_token_count": sum(len(group) for group in tokens),
        "valid_sse_chunk_count": len(nonterminal),
        "sse_chunk_interval_n": len(intervals),
        "all_chunks_single_token": all(len(group) == 1 for group in tokens),
        "server_timings": timings,
    }
    if len(intervals) >= 2:
        result["sse_chunk_interval_p95_ms"] = percentile(intervals, 0.95)
        result["sse_chunk_interval_p99_ms"] = percentile(intervals, 0.99)
    return result


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


def wait_sleep(proc: subprocess.Popen[bytes], port: int, deadline: Deadline) -> list[dict[str, Any]]:
    phase_end = time.monotonic() + min(TIMEOUTS_S["sleep"], deadline.remaining())
    observations: list[dict[str, Any]] = []
    while time.monotonic() < phase_end:
        if proc.poll() is not None:
            fail(f"server exited while waiting for sleep: {proc.returncode}")
        remaining = phase_end - time.monotonic()
        status, payload = request(port, "GET", "/props", deadline, min(2.0, remaining))
        observations.append({"status_code": status, "body": payload})
        if status == 200 and payload.get("is_sleeping") is True:
            return observations
        time.sleep(min(0.25, max(0.0, phase_end - time.monotonic())))
    raise TimeoutError("server did not enter sleeping state")


def common_argv(binary: pathlib.Path, model: pathlib.Path, port: int, sleeping: bool) -> list[str]:
    return [
        str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
        "--ctx-size", "1024", "--batch-size", "128", "--ubatch-size", "128",
        "--parallel", "1", "--seed", "1", "--temp", "0", "--no-warmup",
        "--sleep-idle-seconds", "2" if sleeping else "-1",
        "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps",
    ]


def case_environment(variant: str, kind: str) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update(ON_ENV)
    if variant == "OFF":
        env["LLAMA_KV_PRESSURE_SAMPLER"] = "0"
    if kind == "idle_limit":
        env["LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS"] = "250"
        env["LLAMA_KV_PRESSURE_LOG_INTERVAL_MS"] = "1000"
    if kind == "strace":
        env["LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS"] = "250"
        env["LLAMA_KV_PRESSURE_LOG_INTERVAL_MS"] = "60000"
    return env


def process_group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


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
    return {"shutdown_requested": requested, "returncode": code, "residual_process": trace_proc.poll() is None}


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
    # Reaping the leader first avoids mistaking its zombie for a live process.
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


def attach_strace(strace: pathlib.Path, proc: subprocess.Popen[bytes], prefix: pathlib.Path,
                  deadline: Deadline) -> tuple[subprocess.Popen[bytes], dict[str, Any]]:
    argv = [str(strace), "-ff", "-qq", "-yy", "-s", "4096",
            "-e", "trace=openat,read,close", "-o", str(prefix), "-p", str(proc.pid)]
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
                              "attach_after_health": True, "attach_monotonic_ns": time.monotonic_ns()}
        time.sleep(0.05)
    attached.kill()
    attached.wait(timeout=deadline.remaining(1.0))
    raise TimeoutError("strace attach timeout")


def execute_case(case_dir: pathlib.Path, spec: dict[str, Any], head: str,
                 binary_id: dict[str, Any], model_id: dict[str, Any], strace: pathlib.Path,
                 used_ports: set[int]) -> None:
    case_started_ns = time.monotonic_ns()
    deadline = Deadline(TIMEOUTS_S["case_total"])
    phases: dict[str, Any] = {}
    proc: subprocess.Popen[bytes] | None = None
    trace_proc: subprocess.Popen[bytes] | None = None
    result: dict[str, Any] | None = None
    assert_repo(head)
    port = free_port(used_ports)
    kind, variant = spec["kind"], spec["variant"]
    env = case_environment(variant, kind)
    argv = common_argv(pathlib.Path(binary_id["path"]), pathlib.Path(model_id["path"]), port,
                       sleeping=kind == "lifecycle")
    write_json(case_dir / "execution.json", {
        "argv": argv, "launch_argv": argv, "cwd": str(ROOT), "env": env, "port": port,
        "strace_attach_after_health": kind == "strace",
    })
    write_json(case_dir / "environment.json", env)
    stdout_path, stderr_path = case_dir / "server.stdout", case_dir / "server.stderr"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        startup_ns = time.monotonic_ns()
        try:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                    start_new_session=True)
            if time.monotonic_ns() - startup_ns > TIMEOUTS_S["startup"] * 1e9:
                raise TimeoutError("startup process creation timeout")
            write_json(case_dir / "process.json", {
                "pid": proc.pid, "server_argv": argv, "env": env,
                "exe": os.readlink(f"/proc/{proc.pid}/exe"), "expected_exe": str(pathlib.Path(binary_id["path"]).resolve()),
            })
            phase_record(phases, "startup", startup_ns, "PASS")

            health_ns = time.monotonic_ns()
            health = wait_health(proc, port, deadline)
            write_json(case_dir / "health.json", health)
            phase_record(phases, "health", health_ns, "PASS")

            if kind == "strace":
                trace_proc, trace_info = attach_strace(strace, proc, case_dir / "strace", deadline)
                write_json(case_dir / "strace_process.json", trace_info)
                # Bounded post-attach wait: ensure the sampler has at least one
                # sample window before the request begins.  250 ms interval +
                # 250 ms safety margin = 500 ms minimum.
                post_wait_requested_ms = 500
                post_wait_start_ns = time.monotonic_ns()
                post_wait_end = time.monotonic() + post_wait_requested_ms / 1000.0
                while time.monotonic() < post_wait_end:
                    time.sleep(min(0.05, max(0.0, post_wait_end - time.monotonic())))
                post_wait_actual_ns = time.monotonic_ns()
                trace_info["post_attach_wait_requested_ms"] = post_wait_requested_ms
                trace_info["post_attach_wait_actual_ms"] = (post_wait_actual_ns - post_wait_start_ns) / 1e6

            completion_ns = time.monotonic_ns()
            if kind == "strace":
                trace_info["completion_start_monotonic_ns"] = completion_ns
                write_json(case_dir / "strace_process.json", trace_info)
            stderr.flush(); log_start = stderr_path.stat().st_size
            metrics = stream_completion(port, case_dir / "completion.sse",
                                        case_dir / "completion.events.json", deadline,
                                        TIMEOUTS_S["completion"])
            stderr.flush(); log_end = stderr_path.stat().st_size
            write_json(case_dir / "metrics.json", metrics)
            write_json(case_dir / "completion_window.json", {"stderr_start": log_start, "stderr_end": log_end})
            phase_record(phases, "completion", completion_ns, "PASS")

            if trace_proc is not None:
                trace_info.update(stop_attached_strace(trace_proc, deadline))
                trace_proc = None
                write_json(case_dir / "strace_process.json", trace_info)

            if kind == "lifecycle":
                sleep_ns = time.monotonic_ns()
                write_json(case_dir / "sleep_observations.json", wait_sleep(proc, port, deadline))
                phase_record(phases, "sleep", sleep_ns, "PASS")
                resume_ns = time.monotonic_ns()
                stderr.flush(); wake_start = stderr_path.stat().st_size
                wake_metrics = stream_completion(port, case_dir / "wake_completion.sse",
                                                 case_dir / "wake_completion.events.json", deadline,
                                                 TIMEOUTS_S["resume"])
                stderr.flush(); wake_end = stderr_path.stat().st_size
                write_json(case_dir / "wake_metrics.json", wake_metrics)
                write_json(case_dir / "wake_completion_window.json", {"stderr_start": wake_start, "stderr_end": wake_end})
                status, props = request(port, "GET", "/props", deadline, deadline.remaining(5.0))
                write_json(case_dir / "resume_props.json", {"status_code": status, "body": props})
                phase_record(phases, "resume", resume_ns, "PASS")
            else:
                phases["sleep"] = {"status": "NOT_APPLICABLE", "timeout_s": TIMEOUTS_S["sleep"]}
                phases["resume"] = {"status": "NOT_APPLICABLE", "timeout_s": TIMEOUTS_S["resume"]}

            if kind == "idle_limit":
                stderr.flush(); idle_start_ns = time.monotonic_ns(); idle_start = stderr_path.stat().st_size
                idle_end = time.monotonic() + min(1.5, deadline.remaining())
                while time.monotonic() < idle_end:
                    time.sleep(min(0.05, max(0.0, idle_end - time.monotonic())))
                stderr.flush(); idle_log_end = stderr_path.stat().st_size
                write_json(case_dir / "idle_window.json", {
                    "duration_ms": (time.monotonic_ns() - idle_start_ns) / 1e6,
                    "stderr_start": idle_start, "stderr_end": idle_log_end,
                })
                status, props = request(port, "GET", "/props", deadline, deadline.remaining(5.0))
                write_json(case_dir / "idle_props.json", {"status_code": status, "body": props})
            if proc.poll() is not None:
                fail(f"server exited before shutdown: {proc.returncode}")
        except TimeoutError as exc:
            phases.setdefault("case", {"status": "TIMEOUT", "error": str(exc),
                                       "timeout_s": TIMEOUTS_S["case_total"]})
            raise
        finally:
            cleanup_deadline = deadline
            try:
                deadline.remaining()
            except TimeoutError:
                # A failed total deadline must still receive bounded emergency cleanup.
                cleanup_deadline = Deadline(TIMEOUTS_S["shutdown"])
            if trace_proc is not None:
                stop_attached_strace(trace_proc, cleanup_deadline)
            shutdown_ns = time.monotonic_ns()
            if proc is not None:
                result = terminate(proc, cleanup_deadline)
                phase_record(phases, "shutdown", shutdown_ns, "PASS")
    phases.setdefault("case", {
        "status": "PASS", "started_monotonic_ns": case_started_ns,
        "ended_monotonic_ns": time.monotonic_ns(), "timeout_s": TIMEOUTS_S["case_total"],
    })
    write_json(case_dir / "phases.json", phases)
    if result is None:
        fail("case ended without a shutdown result")
    write_json(case_dir / "result.json", result)
    assert_repo(head)
    if identity(pathlib.Path(binary_id["path"])) != binary_id:
        fail("binary identity changed during the protocol")
    if identity(pathlib.Path(model_id["path"])) != model_id:
        fail("model identity changed during the protocol")


def write_dry_run_case(case_dir: pathlib.Path, spec: dict[str, Any], binary: pathlib.Path,
                       model: pathlib.Path, port: int) -> None:
    env = case_environment(spec["variant"], spec["kind"])
    argv = common_argv(binary, model, port, sleeping=spec["kind"] == "lifecycle")
    write_json(case_dir / "execution.json", {
        "argv": argv, "launch_argv": argv, "cwd": str(ROOT), "env": env, "port": port,
        "dry_run_only": True, "strace_attach_after_health": spec["kind"] == "strace",
    })
    write_json(case_dir / "environment.json", env)


def host_info() -> dict[str, Any]:
    return {
        "hostname": platform.node(), "kernel": platform.release(), "machine": platform.machine(),
        "python": sys.version, "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
    }


def write_inventory(root: pathlib.Path) -> None:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "inventory.sha256.json"):
        entries.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(root / "inventory.sha256.json", {"algorithm": "sha256", "excludes": ["inventory.sha256.json"], "files": entries})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    binary, model = args.binary.resolve(), args.model.resolve()
    for path, label in ((binary, "binary"), (model, "model"), (PARSER, "parser")):
        if not path.is_file():
            fail(f"{label} not found: {path}")
    if not os.access(binary, os.X_OK):
        fail(f"binary is not executable: {binary}")
    head = git(["rev-parse", "HEAD"])
    if git(["status", "--porcelain"]):
        fail("clean worktree is mandatory, including dry-run")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = (args.output or pathlib.Path(
        f"/root/oscomp/kv_logs/server_kv_pressure_stage3a_1c_{timestamp}_{os.getpid()}")).resolve()
    if output.exists():
        fail(f"refusing to overwrite output directory: {output}")
    try:
        output.relative_to(ROOT)
        fail("output must be outside the repository so the worktree stays clean")
    except ValueError:
        pass
    strace_name = shutil.which("strace")
    if not args.dry_run and not strace_name:
        fail("strace is required for the independent correctness cases")
    output.joinpath("cases").mkdir(parents=True)
    binary_id, model_id = identity(binary), identity(model)
    specs = [{"name": n, "kind": k, "round": r, "order": o, "variant": v}
             for n, k, r, o, v in PLAN]
    manifest: dict[str, Any] = {
        "protocol": "server_kv_pressure_stage3a_1c", "version": 2,
        "repo": {"path": str(ROOT), "head": head, "dirty": False, "status_porcelain": []},
        "binary": binary_id, "model": model_id,
        "framework": {"runner": identity(pathlib.Path(__file__).resolve()), "parser": identity(PARSER)},
        "host": host_info(), "dry_run": args.dry_run,
        "pressure_settings_purpose": PRESSURE_SETTINGS_PURPOSE,
        "frozen": {"prompt": PROMPT, "n_predict": N_PREDICT, "on_env": ON_ENV,
                   "forbidden_env": FORBIDDEN_ENV, "timeouts_s": TIMEOUTS_S,
                   "ab_order": [item[0] for item in PLAN if item[1] == "ab"]},
        "planned_cases": specs, "completed_cases": [],
    }
    write_json(output / "manifest.json", manifest)
    used_ports: set[int] = set()
    for index, spec in enumerate(specs):
        case_dir = output / "cases" / spec["name"]
        case_dir.mkdir()
        write_json(case_dir / "case.json", spec)
        if args.dry_run:
            write_dry_run_case(case_dir, spec, binary, model, 18080 + index)
        else:
            execute_case(case_dir, spec, head, binary_id, model_id, pathlib.Path(strace_name), used_ports)
        manifest["completed_cases"].append(spec["name"])
        write_json(output / "manifest.json", manifest)
    assert_repo(head)
    for item, label in ((binary_id, "binary"), (model_id, "model"),
                        (manifest["framework"]["runner"], "runner"),
                        (manifest["framework"]["parser"], "parser")):
        if identity(pathlib.Path(item["path"])) != item:
            fail(f"{label} identity changed before parsing")
    manifest["post_run"] = {
        "repo": {"head": git(["rev-parse", "HEAD"]), "dirty": False, "status_porcelain": []},
        "binary": identity(binary), "model": identity(model),
        "framework": {"runner": identity(pathlib.Path(__file__).resolve()), "parser": identity(PARSER)},
    }
    write_json(output / "manifest.json", manifest)
    write_inventory(output)
    command = [sys.executable, str(PARSER)]
    if args.dry_run:
        command.append("--dry-run")
    command.append(str(output))
    result = subprocess.run(command, check=False)
    print(f"artifact={output}")
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
