#!/usr/bin/env python3
"""Global Fast-A1 Route-A benchmark runner.

This runner only orchestrates the public HTTP workload.  It never calls KV or
MoE private APIs and leaves verdict authority to parse-global-route-a-benchmark.py.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "global_route_a_fast_a1"
PROTOCOL_VERSION = 1
CASES = ("static_low", "global_route_a", "static_high")
MARKER_NAMES = (
    "memory_governor_observe", "kv_pressure_unified_action", "kv_resume_order_event",
    "kv_resume_stage_timing", "KV_PAGED_PREFETCH_PHASE_CALL", "KV_PAGED_PREFETCH_BLOCK_PHASE",
    "KV_PAGED_PREFETCH_PIPELINE",
)
DEFAULT_INTERVAL_MS = 150
DEFAULT_SETTLE_MS = 1000
DEFAULT_REQUEST_TIMEOUT = 180.0


class WorkloadFailure(Exception):
    pass


class RunnerInterrupted(BaseException):
    pass


def dump(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def file_identity(path: pathlib.Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha(path)}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()


def git_provenance() -> dict[str, Any]:
    dirty = git("status", "--porcelain").splitlines()
    diff = subprocess.check_output(["git", "-C", str(ROOT), "diff", "--no-ext-diff", "--binary", "HEAD"])
    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "dirty_status": dirty,
        "tracked_diff_fingerprint": sha_bytes(diff),
        "capture_mode": "DIRTY_DEV_ONLY" if dirty else "archival_clean",
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def monotonic_ns() -> int:
    return time.monotonic_ns()


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


def parse_latest_runtime(stderr: pathlib.Path) -> dict[str, Any]:
    if not stderr.is_file():
        return {}
    latest: dict[str, Any] = {}
    try:
        lines = stderr.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return latest
    for line in lines[-500:]:
        for marker in ("memory_governor_observe", "kv_pressure_unified_action"):
            if marker in line:
                fields = fields_after_token(line, marker)
                if fields is not None:
                    latest.update({f"{marker}.{key}": value for key, value in fields.items()})
    return latest


def read_int(path: pathlib.Path) -> int | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_memory_stat(path: pathlib.Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(" ")
            try:
                result[key] = int(value)
            except ValueError:
                continue
    except OSError:
        pass
    return result


def proc_sample(pid: int) -> dict[str, Any]:
    result: dict[str, Any] = {"pid": pid}
    status = pathlib.Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem"}:
                try:
                    result[key] = int(value.strip().split()[0]) * 1024
                except (IndexError, ValueError):
                    result[key] = None
    except OSError:
        result["proc_unavailable"] = True
    smaps = pathlib.Path(f"/proc/{pid}/smaps_rollup")
    try:
        for line in smaps.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss", "Private_Clean", "Private_Dirty"}:
                try:
                    result[f"smaps_{key}"] = int(value.strip().split()[0]) * 1024
                except (IndexError, ValueError):
                    result[f"smaps_{key}"] = None
    except OSError:
        result["smaps_unavailable"] = True
    try:
        for line in pathlib.Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"read_bytes", "write_bytes", "syscr", "syscw"}:
                result[key] = int(value.strip())
    except (OSError, ValueError):
        result["io_unavailable"] = True
    return result


def cgroup_sample(cgroup_path: pathlib.Path | None) -> dict[str, Any]:
    if cgroup_path is None:
        return {"available": False}
    result: dict[str, Any] = {"available": cgroup_path.is_dir(), "path": str(cgroup_path)}
    for name in ("memory.current", "memory.peak", "memory.high", "memory.max"):
        result[name.replace(".", "_")] = read_int(cgroup_path / name)
    result["memory_stat"] = read_memory_stat(cgroup_path / "memory.stat")
    result["memory_events"] = read_memory_stat(cgroup_path / "memory.events")
    psi = pathlib.Path("/proc/pressure/memory")
    try:
        result["pressure_memory"] = psi.read_text(encoding="utf-8").splitlines()
    except OSError:
        result["pressure_memory"] = None
    return result


class Sampler(threading.Thread):
    def __init__(self, pid: int, stderr: pathlib.Path, out: pathlib.Path, interval_ms: int,
                 cgroup_path: pathlib.Path | None, phase: dict[str, str]):
        super().__init__(daemon=True)
        self.pid = pid
        self.stderr = stderr
        self.out = out
        self.interval = max(0.1, interval_ms / 1000.0)
        self.cgroup_path = cgroup_path
        self.phase = phase
        self.stop_event = threading.Event()
        self.samples = 0
        self.warnings: list[str] = []

    def run(self) -> None:
        try:
            with self.out.open("w", encoding="utf-8") as handle:
                while not self.stop_event.is_set():
                    item: dict[str, Any] = {
                        "monotonic_ns": monotonic_ns(),
                        "phase": self.phase.get("name", "unknown"),
                        "process": proc_sample(self.pid),
                        "cgroup": cgroup_sample(self.cgroup_path),
                        "runtime": parse_latest_runtime(self.stderr),
                    }
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
                    handle.flush()
                    self.samples += 1
                    self.stop_event.wait(self.interval)
        except OSError as exc:
            self.warnings.append(f"sampler output failure: {type(exc).__name__}: {exc}")

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.join(timeout=max(2.0, self.interval * 4))
        return {"samples": self.samples, "warnings": self.warnings, "alive": self.is_alive()}


def process_identity(pid: int) -> dict[str, Any] | None:
    try:
        stat = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        cmdline = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return None
    right = stat.rfind(")")
    if right < 0:
        return None
    fields = stat[right + 2:].split()
    if len(fields) < 20:
        return None
    return {
        "pid": pid,
        "starttime_ticks": int(fields[19]),
        "cmdline": cmdline.decode("utf-8", errors="replace").split("\0")[:-1],
        "cmdline_sha256": sha_bytes(cmdline),
    }


def stop_process(proc: subprocess.Popen[bytes] | None) -> dict[str, Any]:
    if proc is None:
        return {"exit_code": None, "term_timed_out": False, "kill_timed_out": False, "residual_process": False}
    term_timed_out = kill_timed_out = False
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        term_timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_timed_out = True
    residual = False
    try:
        os.killpg(proc.pid, 0)
        residual = True
    except OSError:
        pass
    return {
        "exit_code": proc.returncode,
        "term_timed_out": term_timed_out,
        "kill_timed_out": kill_timed_out,
        "residual_process": residual,
    }


def request_completion(port: int, body: dict[str, Any], phase: str, label: str,
                       out: pathlib.Path, timeout: float) -> dict[str, Any]:
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    started = monotonic_ns()
    status = 0
    raw = b""
    error = None
    value: Any = {}
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        connection.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
        response = connection.getresponse()
        status = response.status
        raw = response.read()
        connection.close()
        value = json.loads(raw) if raw else {}
    except (OSError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        value = {"transport_error": error}
    finished = monotonic_ns()
    text = ""
    tokens: Any = None
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            text = value["content"]
        elif isinstance(value.get("choices"), list) and value["choices"] and isinstance(value["choices"][0], dict):
            text = value["choices"][0].get("text", "")
        if isinstance(value.get("tokens"), list) and all(isinstance(x, int) and not isinstance(x, bool) for x in value["tokens"]):
            tokens = value["tokens"]
    item = {
        "label": label, "phase": phase, "request": body,
        "request_sha256": sha_bytes(encoded), "started_monotonic_ns": started,
        "finished_monotonic_ns": finished, "duration_ms": (finished - started) / 1e6,
        "http_status": status, "transport_error": error,
        "response_raw": raw.decode("utf-8", errors="replace"),
        "response_text": text, "response_sha256": sha_bytes(text.encode("utf-8")),
        "response_tokens": tokens,
        "response_tokens_sha256": sha_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8")) if tokens is not None else None,
        "response_timings": value.get("timings") if isinstance(value, dict) else None,
    }
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
    return item


def health(port: int, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    attempts = 0
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        attempts += 1
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            connection.close()
            last = {"status": response.status, "body": body}
            if response.status == 200:
                return {"ok": True, "attempts": attempts, **last}
        except OSError as exc:
            last = {"error": f"{type(exc).__name__}: {exc}"}
        time.sleep(0.2)
    return {"ok": False, "attempts": attempts, **last}


def stderr_size(path: pathlib.Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def phase_record(case: pathlib.Path, phases: list[dict[str, Any]], name: str, phase_state: dict[str, str],
                 fn: Any) -> Any:
    start = monotonic_ns()
    begin = stderr_size(case / "server.stderr")
    phase_state["name"] = name
    record: dict[str, Any] = {"name": name, "started_monotonic_ns": start, "stderr_start": begin, "status": "RUNNING"}
    phases.append(record)
    try:
        value = fn()
        record["status"] = "PASS"
        return value
    except Exception as exc:
        record["status"] = "FAIL"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["ended_monotonic_ns"] = monotonic_ns()
        record["stderr_end"] = stderr_size(case / "server.stderr")
        with (case / "phase_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def normalize_request(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkloadFailure(f"workload.{label} must be an object")
    result = dict(value)
    result.setdefault("id_slot", 0)
    result.setdefault("cache_prompt", True)
    result.setdefault("stream", False)
    result.setdefault("seed", 0)
    result.setdefault("temperature", 0.0)
    result.setdefault("top_k", 1)
    result.setdefault("top_p", 1.0)
    if "prompt" not in result and "messages" not in result:
        raise WorkloadFailure(f"workload.{label} needs prompt or messages")
    return result


def load_spec(path: pathlib.Path) -> dict[str, Any]:
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkloadFailure(f"invalid spec: {exc}") from exc
    if not isinstance(spec, dict):
        raise WorkloadFailure("spec must be an object")
    workload = spec.get("workload")
    if not isinstance(workload, dict):
        raise WorkloadFailure("spec.workload is required")
    for label in ("session_a_turn1", "session_a_turn2", "session_b"):
        normalize_request(workload.get(label), label)
    repeats = workload.get("moe_demand_repeats", 4)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or not 4 <= repeats <= 8:
        raise WorkloadFailure("moe_demand_repeats must be an integer in [4, 8]")
    for key in ("kv_resident_target_bytes", "initial_moe_budget_bytes", "static_high_moe_budget_bytes", "memory_cap_bytes"):
        if not isinstance(spec.get(key), int) or isinstance(spec[key], bool) or spec[key] <= 0:
            raise WorkloadFailure(f"spec.{key} must be a positive integer")
    for key in ("initial_moe_budget_bytes", "static_high_moe_budget_bytes"):
        if spec[key] % (1 << 20) != 0:
            raise WorkloadFailure(f"spec.{key} must be MiB-aligned for *_MB runtime controls")
    return spec


def build_env(spec: dict[str, Any], case_name: str, case_dir: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"HOME": "/tmp", "LANG": "C", "LC_ALL": "C"})
    for key in list(env):
        if key.startswith(("LLAMA_KV_", "LLAMA_MEMORY_GOVERNOR_", "LLAMA_LAZY_MOE_")) or key in {"LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"}:
            del env[key]
    target = int(spec["kv_resident_target_bytes"])
    initial = int(spec["initial_moe_budget_bytes"])
    high = int(spec["static_high_moe_budget_bytes"])
    global_case = case_name == "global_route_a"
    moe_budget = initial if case_name != "static_high" else high
    env.update({
        "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_SWAP": "1",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1", "LLAMA_KV_PAGED_MINCORE": "1",
        "LLAMA_KV_PAGED_BLOCK_SIZE": str(spec.get("paged_block_size", 64)),
        "LLAMA_KV_SWAP_DIR": "backing", "LLAMA_KV_PRESSURE_SAMPLER": "1",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": str(spec.get("action_target_bytes", target)),
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": str(spec.get("action_max_blocks", 64)),
        "LLAMA_KV_RESIDENT_TARGET_BYTES": str(target),
        "LLAMA_KV_RESIDENT_TARGET_SOURCE": "env_static",
        "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": str(spec.get("pressure_sample_interval_ms", 100)),
        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": str(spec.get("pressure_log_interval_ms", 1000)),
        "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "1",
        "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "1",
        "LLAMA_KV_RESUME_ORDER_TRACE": "1",
        "LLAMA_KV_RESUME_STAGE_TIMING": "1",
        "LLAMA_KV_PAGED_IO_STATS": "1",
        "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER": "1" if global_case else "0",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION": "1" if global_case else "0",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_APPLY_MOE": "1" if global_case else "0",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_CONFIRM": "1",
        "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC": "1" if global_case else "0",
        "LLAMA_MEMORY_GOVERNOR_MOE_FAST_START": "0",
        "LLAMA_MEMORY_GOVERNOR_MOE_MIN_MB": str(max(1, initial // (1 << 20))),
        "LLAMA_MEMORY_GOVERNOR_MOE_WARM_MB": str(max(1, moe_budget // (1 << 20))),
        "LLAMA_MEMORY_GOVERNOR_MOE_MAX_MB": str(max(1, (high if global_case else moe_budget) // (1 << 20))),
        "LLAMA_MEMORY_GOVERNOR_MOE_GROW_MB": str(max(1, spec.get("moe_grow_bytes", 64 << 20) // (1 << 20))),
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_CREDIT_CAP_MB": str(max(1, spec.get("credit_cap_bytes", 256 << 20) // (1 << 20))),
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_MAX_GRANT_MB": str(max(1, spec.get("max_grant_bytes", 64 << 20) // (1 << 20))),
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_MIN_GRANT_MB": str(max(1, spec.get("min_grant_bytes", 1 << 20) // (1 << 20))),
    })
    sidecar = spec.get("moe_sidecar")
    if sidecar:
        env["LLAMA_LAZY_MOE_SIDECAR"] = str(pathlib.Path(sidecar).resolve())
    protected_prefixes = ("LLAMA_KV_", "LLAMA_MEMORY_GOVERNOR_", "LLAMA_LAZY_MOE_")
    protected_exact = {"LLAMA_ARG_CACHE_TYPE_K", "LLAMA_ARG_CACHE_TYPE_V"}
    for key, value in spec.get("environment", {}).items():
        if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
            raise WorkloadFailure("environment overrides must be scalar")
        if key.startswith(protected_prefixes) or key in protected_exact:
            raise WorkloadFailure(f"environment override is protected: {key}")
        env[key] = str(value)
    dump(case_dir / "environment.json", {"effective": env, "case": case_name, "target_bytes": target,
                                         "initial_moe_budget_bytes": initial, "case_moe_budget_bytes": moe_budget})
    return env


def server_argv(binary: pathlib.Path, model: pathlib.Path, port: int, spec: dict[str, Any]) -> list[str]:
    argv = [str(binary), "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(spec.get("ctx_size", 8192)), "--parallel", "1",
            "--cache-type-k", "f16", "--cache-type-v", "f16"]
    extra = spec.get("server_args", [])
    if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
        raise WorkloadFailure("server_args must be a list of strings")
    return argv + extra


def attach_to_cgroup(pid: int, cgroup_path: pathlib.Path) -> None:
    procs = cgroup_path / "cgroup.procs"
    try:
        procs.write_text(f"{pid}\n", encoding="utf-8")
        members = procs.read_text(encoding="utf-8").split()
    except OSError as exc:
        raise WorkloadFailure(f"cannot attach server pid {pid} to cgroup: {exc}") from exc
    if str(pid) not in members:
        raise WorkloadFailure(f"server pid {pid} is not present in cgroup {cgroup_path}")

def run_case(case_name: str, spec: dict[str, Any], output: pathlib.Path, binary: pathlib.Path,
             model: pathlib.Path, cgroup_path: pathlib.Path | None) -> dict[str, Any]:
    case = output / case_name
    case.mkdir()
    (case / "backing").mkdir()
    phases: list[dict[str, Any]] = []
    phase_state = {"name": "P0"}
    port = free_port()
    env = build_env(spec, case_name, case)
    argv = server_argv(binary, model, port, spec)
    dump(case / "execution.json", {"argv": argv, "cwd": str(case.resolve()), "port": port,
                                    "binary": file_identity(binary), "model": file_identity(model),
                                    "cgroup_path": str(cgroup_path) if cgroup_path else None})
    server_stdout = case / "server.stdout"
    server_stderr = case / "server.stderr"
    requests = case / "requests.jsonl"
    phases_path = case / "phase_events.jsonl"
    proc: subprocess.Popen[bytes] | None = None
    sampler: Sampler | None = None
    result: dict[str, Any] = {"case": case_name, "status": "startup_failed", "port": port}
    try:
        proc = subprocess.Popen(argv, cwd=case, stdout=server_stdout.open("wb"), stderr=server_stderr.open("wb"),
                                env=env, preexec_fn=os.setsid)
        if cgroup_path is None:
            raise WorkloadFailure("cgroup path is required for Fast-A1")
        attach_to_cgroup(proc.pid, cgroup_path)
        identity = process_identity(proc.pid)
        dump(case / "process.json", identity or {"pid": proc.pid, "identity_unavailable": True})
        sampler = Sampler(proc.pid, server_stderr, case / "memory_timeline.jsonl",
                          int(spec.get("sampler_interval_ms", DEFAULT_INTERVAL_MS)), cgroup_path, phase_state)
        sampler.start()
        phase_record(case, phases, "P0_preflight_identity", phase_state, lambda: identity or (_ for _ in ()).throw(WorkloadFailure("process identity unavailable")))
        health_result = phase_record(case, phases, "P0_health", phase_state, lambda: health(port, float(spec.get("health_timeout_s", 60))))
        dump(case / "health.json", health_result)
        if not health_result.get("ok"):
            raise WorkloadFailure(f"health failed: {health_result}")
        workload = spec["workload"]
        normalized = {key: normalize_request(workload[key], key) for key in ("session_a_turn1", "session_a_turn2", "session_b")}
        warmups = workload.get("warmup", [])
        if not isinstance(warmups, list):
            raise WorkloadFailure("workload.warmup must be a list")
        for index, warmup in enumerate(warmups):
            body = normalize_request(warmup, f"warmup[{index}]")
            phase_record(case, phases, "P1_fixed_warmup", phase_state,
                         lambda body=body, index=index: request_completion(port, body, "P1", f"warmup_{index}", requests, DEFAULT_REQUEST_TIMEOUT))
        phase_record(case, phases, "P1_fixed_warmup", phase_state, lambda: None)
        phase_record(case, phases, "P2_A_turn1_fill", phase_state,
                     lambda: request_completion(port, normalized["session_a_turn1"], "P2", "A_turn1", requests, DEFAULT_REQUEST_TIMEOUT))
        settle_ms = int(spec.get("settle_ms", DEFAULT_SETTLE_MS))
        phase_record(case, phases, "P3_A_completion_idle_release", phase_state, lambda: time.sleep(max(0, settle_ms) / 1000.0))
        phase_record(case, phases, "P4_route_a_credit_observation", phase_state, lambda: time.sleep(max(0, settle_ms) / 1000.0))
        b_results = []
        for index in range(int(workload.get("moe_demand_repeats", 4))):
            b_results.append(phase_record(case, phases, "P5_B_moe_demand", phase_state,
                                          lambda index=index: request_completion(port, normalized["session_b"], "P5", f"B_{index}", requests, DEFAULT_REQUEST_TIMEOUT)))
        turn2 = phase_record(case, phases, "P6_A_turn2_exact_restore", phase_state,
                             lambda: request_completion(port, normalized["session_a_turn2"], "P6", "A_turn2", requests, DEFAULT_REQUEST_TIMEOUT))
        phase_record(case, phases, "P7_short_steady_measurement", phase_state,
                     lambda: time.sleep(float(spec.get("steady_ms", 250)) / 1000.0))
        result.update({"status": "complete", "turn2_response_sha256": turn2["response_sha256"],
                       "turn2_tokens_sha256": turn2.get("response_tokens_sha256"),
                       "b_completed": sum(1 for item in b_results if item.get("http_status") == 200)})
    except RunnerInterrupted:
        result.update({"status": "interrupted"})
        raise
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        sampler_result = sampler.stop() if sampler is not None else {"samples": 0, "warnings": [], "alive": False}
        server_result = stop_process(proc)
        dump(case / "exit_codes.json", {"server": server_result, "sampler": sampler_result})
        dump(case / "cleanup.json", {"server": server_result, "sampler": sampler_result,
                                      "completed": not server_result.get("residual_process", False)})
        dump(case / "result.json", result)
    return result


def build_manifest(spec_path: pathlib.Path, output: pathlib.Path, binary: pathlib.Path, model: pathlib.Path,
                   parser: pathlib.Path, cgroup_path: pathlib.Path | None) -> dict[str, Any]:
    spec = load_spec(spec_path)
    return {
        "protocol": PROTOCOL, "protocol_version": PROTOCOL_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git": git_provenance(), "runner": file_identity(pathlib.Path(__file__)),
        "parser": file_identity(parser), "spec": file_identity(spec_path),
        "binary": file_identity(binary), "model": file_identity(model),
        "moe_sidecar": file_identity(pathlib.Path(spec["moe_sidecar"]).resolve()) if spec.get("moe_sidecar") else None,
        "workload": {"path": str(spec_path), "sha256": sha(spec_path)},
        "cgroup_path": str(cgroup_path) if cgroup_path else None,
        "case_names": list(CASES), "parameters": {key: spec[key] for key in (
            "kv_resident_target_bytes", "initial_moe_budget_bytes", "static_high_moe_budget_bytes", "memory_cap_bytes")},
        "runner_status": "run_in_progress", "required_artifacts": [
            "manifest.json", "case_spec.json", "server.stdout", "server.stderr", "requests.jsonl",
            "phase_events.jsonl", "memory_timeline.jsonl", "cleanup.json", "exit_codes.json", "result.json",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--cgroup-path", required=True)
    args = parser.parse_args()
    spec_path = pathlib.Path(args.spec).resolve()
    binary = pathlib.Path(args.binary).resolve()
    model = pathlib.Path(args.model).resolve()
    if not spec_path.is_file() or not binary.is_file() or not model.is_file():
        print("missing --spec/--binary/--model", file=sys.stderr)
        return 2
    try:
        spec = load_spec(spec_path)
        cgroup_path = pathlib.Path(args.cgroup_path).resolve() if args.cgroup_path else None
        if cgroup_path is not None:
            if not cgroup_path.is_dir():
                raise WorkloadFailure(f"cgroup path does not exist: {cgroup_path}")
            memory_max = read_int(cgroup_path / "memory.max")
            if memory_max != int(spec["memory_cap_bytes"]):
                raise WorkloadFailure(
                    f"cgroup memory.max {memory_max!r} does not equal spec memory_cap_bytes {spec['memory_cap_bytes']}"
                )
        output = pathlib.Path(args.output_dir).resolve() if args.output_dir else pathlib.Path(
            f"/root/oscomp/kv_logs/{PROTOCOL}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}")
        if output.exists():
            print(f"refusing existing artifact directory: {output}", file=sys.stderr)
            return 2
        output.mkdir(parents=True)
        manifest = build_manifest(spec_path, output, binary, model,
                                  ROOT / "scripts/parse-global-route-a-benchmark.py", cgroup_path)
        dump(output / "case_spec.json", spec)
        dump(output / "manifest.json", manifest)
        results = []
        for case_name in CASES:
            results.append(run_case(case_name, spec, output, binary, model, cgroup_path))
        manifest["runner_status"] = "run_complete" if all(item.get("status") == "complete" for item in results) else "run_failed"
        manifest["finished_timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest["case_results"] = results
        dump(output / "manifest.json", manifest)
        parser_path = ROOT / "scripts/parse-global-route-a-benchmark.py"
        parser_env = dict(os.environ)
        parser_env["PYTHONUTF8"] = "1"
        parsed = subprocess.run([sys.executable, str(parser_path), str(output)], env=parser_env, encoding="utf-8", text=True)
        return parsed.returncode
    except (WorkloadFailure, OSError, subprocess.CalledProcessError) as exc:
        print(f"global Route-A runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerInterrupted:
        raise SystemExit(130)
