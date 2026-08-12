#!/usr/bin/env python3
"""Profile LLM inference memory behavior from llama-server logs and runtime samples.

The script is evidence-first: by default it enables only observation markers and
explicitly disables legacy destructive KV actions.  It does not create or modify
cgroups and only terminates the process group it starts.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import pathlib
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import Any, NoReturn

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "build/bin/llama-server"
DEFAULT_PROMPT = "Explain memory scheduling briefly."

MARKERS = (
    "memory_governor_observe",
    "kv_pressure_telemetry",
    "kv_pressure_dry_run",
    "kv_pressure_bounded_release",
    "kv_pressure_unified_action",
    "KV_GOVERNOR_CAPABILITY",
    "kv_g0_s1_resident_observation",
    "kv_resume_order_event",
)

RECOMMENDED_FIELDS = {
    "memory_governor_observe": {
        "sample_count",
        "effective_pressure_state",
        "dense_resident_bytes",
        "moe_resident_bytes",
        "kv_effective_resident_bytes",
    },
    "kv_pressure_telemetry": {
        "state",
        "source",
        "sample_valid",
        "rss_kb",
        "sample_count",
    },
}

FORBIDDEN_DEFAULT_ENV = {
    "LLAMA_KV_PAGED_RELEASE": "0",
    "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_RESUME_PREFETCH": "0",
    "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "0",
    "LLAMA_KV_SWAP": "0",
    "LLAMA_KV_SWAP_MADVISE": "0",
}

ACTION_ENV_KEYS = {
    "LLAMA_MEMORY_GOVERNOR_KV_RELEASE",
    "LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD",
    "LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM",
    "LLAMA_KV_PAGED_RELEASE",
    "LLAMA_KV_PAGED_SWAP",
}

KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if not path.is_file():
        return {"path": str(path), "exists": True, "is_file": False}
    return {"path": str(path), "exists": True, "size": path.stat().st_size, "sha256": sha256(path)}


def write_inventory(out_dir: pathlib.Path) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "inventory.sha256.json":
            continue
        rel = path.relative_to(out_dir)
        records.append({"path": str(rel), "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(out_dir / "inventory.sha256.json", {"files": records})


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        fail(f"expected KEY=VALUE, got: {value}")
    key, val = value.split("=", 1)
    if not key:
        fail(f"empty env key in: {value}")
    return key, val


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return pathlib.Path(args.prompt_file).read_text(encoding="utf-8")
    return args.prompt


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "LLAMA_MEMORY_GOVERNOR": "1",
        "LLAMA_MEMORY_GOVERNOR_OBSERVE": "1",
        "LLAMA_MEMORY_GOVERNOR_OBSERVE_MS": str(args.observe_ms),
    })
    if args.auto_backends:
        env["LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS"] = "1"
    if args.flex_trace:
        env["LLAMA_FLEX_TRACE"] = str(pathlib.Path(args.flex_trace))

    if not args.allow_unsafe_env:
        env.update(FORBIDDEN_DEFAULT_ENV)

    for item in args.env or []:
        key, val = parse_key_value(item)
        env[key] = val

    validate_env_safety(env, args)
    return env


def validate_env_safety(env: dict[str, str], args: argparse.Namespace) -> None:
    if args.enable_actions:
        return
    unsafe = []
    for key in ACTION_ENV_KEYS:
        if env.get(key) not in (None, "", "0", "false", "False", "FALSE"):
            unsafe.append(f"{key}={env[key]}")
    if unsafe:
        fail("action env requires --enable-actions: " + ", ".join(sorted(unsafe)))


def build_server_argv(args: argparse.Namespace, port: int) -> list[str]:
    argv = [
        str(pathlib.Path(args.binary)),
        "--model", str(pathlib.Path(args.model)),
        "--host", args.host,
        "--port", str(port),
        "--ctx-size", str(args.ctx_size),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--threads", str(args.threads),
        "--parallel", str(args.parallel),
        "--seed", str(args.seed),
        "--temp", str(args.temp),
        "--log-verbosity", str(args.log_verbosity),
        "--no-log-prefix",
        "--no-log-timestamps",
        "--no-webui",
    ]
    if args.no_warmup:
        argv.append("--no-warmup")
    argv.extend(args.extra_server_arg or [])
    return argv


def wait_health(host: str, port: int, proc: subprocess.Popen[bytes], timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    attempts = 0
    last_error = ""
    while time.monotonic() < deadline:
        attempts += 1
        if proc.poll() is not None:
            return {"ok": False, "attempts": attempts, "error": f"server exited rc={proc.returncode}"}
        conn: http.client.HTTPConnection | None = None
        try:
            conn = http.client.HTTPConnection(host, port, timeout=1.0)
            conn.request("GET", "/health")
            response = conn.getresponse()
            payload = response.read()
            if response.status == 200:
                parsed: Any = None
                try:
                    parsed = json.loads(payload) if payload else None
                except ValueError:
                    parsed = payload.decode("utf-8", errors="replace")
                return {"ok": True, "attempts": attempts, "status": response.status, "body": parsed}
            last_error = f"HTTP {response.status}"
        except OSError as exc:
            last_error = str(exc)
        finally:
            if conn is not None:
                conn.close()
        time.sleep(0.2)
    return {"ok": False, "attempts": attempts, "error": last_error or "timeout"}


def terminate_process_group(proc: subprocess.Popen[bytes], timeout_s: float = 10.0) -> dict[str, Any]:
    if proc.poll() is not None:
        return {"already_exited": True, "returncode": proc.returncode}
    result: dict[str, Any] = {"already_exited": False}
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        result["sigterm_sent"] = True
    except ProcessLookupError:
        result["sigterm_sent"] = False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            result["returncode"] = proc.returncode
            return result
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
        result["sigkill_sent"] = True
    except ProcessLookupError:
        result["sigkill_sent"] = False
    proc.wait(timeout=5)
    result["returncode"] = proc.returncode
    return result


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def stream_completion(host: str, port: int, prompt: str, args: argparse.Namespace,
                      raw_path: pathlib.Path, events_path: pathlib.Path) -> dict[str, Any]:
    body = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "stream": args.stream,
        "seed": args.seed,
        "temperature": args.temp,
        "cache_prompt": args.cache_prompt,
    }
    encoded = json.dumps(body).encode("utf-8")
    started = time.monotonic_ns()
    conn = http.client.HTTPConnection(host, port, timeout=args.request_timeout_sec)
    raw = bytearray()
    records: list[dict[str, Any]] = []
    http_status: int | None = None
    response_body: Any = None
    try:
        conn.request("POST", "/completion", encoded, {"Content-Type": "application/json"})
        response = conn.getresponse()
        http_status = response.status
        if response.status != 200:
            payload = response.read()
            raw.extend(payload)
            fail(f"completion returned HTTP {response.status}: {payload[:200]!r}")
        if args.stream:
            while True:
                line = response.readline()
                if not line:
                    break
                raw.extend(line)
                stripped = line.rstrip(b"\r\n")
                if not stripped:
                    continue
                if not stripped.startswith(b"data: "):
                    records.append({
                        "arrival_monotonic_ns": time.monotonic_ns(),
                        "raw_payload": stripped.decode("utf-8", errors="replace"),
                        "parse_error": "non_sse_data_line",
                    })
                    continue
                payload = stripped[6:]
                record: dict[str, Any] = {
                    "arrival_monotonic_ns": time.monotonic_ns(),
                    "raw_payload": payload.decode("utf-8", errors="replace"),
                }
                try:
                    record["parsed"] = json.loads(payload)
                except ValueError as exc:
                    record["parse_error"] = str(exc)
                records.append(record)
        else:
            payload = response.read()
            raw.extend(payload)
            response_body = json.loads(payload) if payload else {}
            records.append({"arrival_monotonic_ns": time.monotonic_ns(), "parsed": response_body})
    finally:
        conn.close()
        raw_path.write_bytes(raw)
        write_json(events_path, {
            "request_started_monotonic_ns": started,
            "stream": args.stream,
            "http_status": http_status,
            "events": records,
            "response_body": response_body,
            "stream_ended_monotonic_ns": time.monotonic_ns(),
            "clock": "time.monotonic_ns",
        })
    return derive_runner_metrics(events_path)


def derive_runner_metrics(events_path: pathlib.Path) -> dict[str, Any]:
    evidence = json.loads(events_path.read_text(encoding="utf-8"))
    events = evidence.get("events", [])
    parsed = [e.get("parsed") for e in events if isinstance(e.get("parsed"), dict)]
    timings = None
    for obj in reversed(parsed):
        if isinstance(obj.get("timings"), dict):
            timings = obj["timings"]
            break
    arrivals = [int(e["arrival_monotonic_ns"]) for e in events if "arrival_monotonic_ns" in e]
    intervals_ms = [(b - a) / 1e6 for a, b in zip(arrivals, arrivals[1:])]
    metrics: dict[str, Any] = {
        "http_status": evidence.get("http_status"),
        "event_count": len(events),
        "ttft_ms": ((arrivals[0] - evidence["request_started_monotonic_ns"]) / 1e6) if arrivals else None,
        "wall_ms": ((evidence["stream_ended_monotonic_ns"] - evidence["request_started_monotonic_ns"]) / 1e6),
        "sse_chunk_interval_p50_ms": percentile(intervals_ms, 0.50),
        "sse_chunk_interval_p95_ms": percentile(intervals_ms, 0.95),
        "sse_chunk_interval_p99_ms": percentile(intervals_ms, 0.99),
        "server_timings": timings,
    }
    if timings:
        for key in ("predicted_n", "predicted_ms", "predicted_per_second", "prompt_n", "prompt_ms", "prompt_per_second"):
            if key in timings:
                metrics[key] = timings[key]
        predicted_n = timings.get("predicted_n")
        predicted_ms = timings.get("predicted_ms")
        if isinstance(predicted_n, (int, float)) and isinstance(predicted_ms, (int, float)) and predicted_n > 1:
            metrics["server_tpot_ms"] = predicted_ms / max(1, predicted_n - 1)
    return metrics


def parse_colon_file(path: pathlib.Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    except OSError:
        pass
    return data


def read_proc_status(pid: int) -> dict[str, str]:
    return parse_colon_file(pathlib.Path(f"/proc/{pid}/status"))


def read_proc_smaps_rollup(pid: int) -> dict[str, str]:
    return parse_colon_file(pathlib.Path(f"/proc/{pid}/smaps_rollup"))


def read_proc_io(pid: int) -> dict[str, str]:
    return parse_colon_file(pathlib.Path(f"/proc/{pid}/io"))


def kb_value(value: str | None) -> str:
    if not value:
        return "NA"
    return value.split()[0] if value.split() else "NA"


def plain_int(value: str | None) -> str:
    if value is None:
        return "NA"
    first = value.split()[0] if value.split() else value
    return first if re.fullmatch(r"-?\d+", first) else "NA"


def read_cgroup_events(path: pathlib.Path) -> dict[str, str]:
    result = {"low": "NA", "high": "NA", "max": "NA", "oom": "NA", "oom_kill": "NA"}
    try:
        for line in (path / "memory.events").read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] in result:
                result[parts[0]] = parts[1]
    except OSError:
        pass
    return result


def read_cgroup_scalar(path: pathlib.Path, name: str) -> str:
    try:
        return (path / name).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return "NA"


class SamplerThread(threading.Thread):
    def __init__(self, pid: int, interval_ms: int, out_dir: pathlib.Path, cgroup_path: pathlib.Path | None):
        super().__init__(daemon=True)
        self.pid = pid
        self.interval_s = max(0.01, interval_ms / 1000.0)
        self.out_dir = out_dir
        self.cgroup_path = cgroup_path
        self.stop_event = threading.Event()
        self.warnings: list[str] = []

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        proc_path = self.out_dir / "proc-samples.tsv"
        cg_path = self.out_dir / "cgroup-samples.tsv"
        with proc_path.open("w", encoding="utf-8") as proc_handle:
            proc_handle.write("monotonic_ns\tpid\tVmRSS_kB\tVmHWM_kB\tRssAnon_kB\tRssFile_kB\tRssShmem_kB\tsmaps_Rss_kB\tsmaps_Pss_kB\tread_bytes\twrite_bytes\tsyscr\tsyscw\n")
            cg_handle = cg_path.open("w", encoding="utf-8")
            try:
                cg_handle.write("monotonic_ns\tmemory_current\tmemory_peak\tmemory_high\tmemory_max\tevents_low\tevents_high\tevents_max\tevents_oom\tevents_oom_kill\n")
                while not self.stop_event.is_set():
                    ts = time.monotonic_ns()
                    status = read_proc_status(self.pid)
                    smaps = read_proc_smaps_rollup(self.pid)
                    io = read_proc_io(self.pid)
                    proc_handle.write("\t".join([
                        str(ts), str(self.pid),
                        kb_value(status.get("VmRSS")), kb_value(status.get("VmHWM")),
                        kb_value(status.get("RssAnon")), kb_value(status.get("RssFile")), kb_value(status.get("RssShmem")),
                        kb_value(smaps.get("Rss")), kb_value(smaps.get("Pss")),
                        plain_int(io.get("read_bytes")), plain_int(io.get("write_bytes")),
                        plain_int(io.get("syscr")), plain_int(io.get("syscw")),
                    ]) + "\n")
                    proc_handle.flush()
                    if self.cgroup_path is not None:
                        events = read_cgroup_events(self.cgroup_path)
                        cg_handle.write("\t".join([
                            str(ts),
                            read_cgroup_scalar(self.cgroup_path, "memory.current"),
                            read_cgroup_scalar(self.cgroup_path, "memory.peak"),
                            read_cgroup_scalar(self.cgroup_path, "memory.high"),
                            read_cgroup_scalar(self.cgroup_path, "memory.max"),
                            events["low"], events["high"], events["max"], events["oom"], events["oom_kill"],
                        ]) + "\n")
                        cg_handle.flush()
                    time.sleep(self.interval_s)
            finally:
                cg_handle.close()


def coerce_value(value: str) -> Any:
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)(?:[eE]-?\d+)?", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_marker_line(line: str, marker: str) -> dict[str, Any] | None:
    idx = line.find(marker)
    if idx < 0:
        return None
    before = line[:idx]
    after_idx = idx + len(marker)
    if before and not before[-1].isspace():
        return None
    if after_idx < len(line) and not line[after_idx].isspace():
        return None
    rest = line[after_idx:].strip()
    fields: dict[str, Any] = {}
    duplicate_keys: list[str] = []
    unparsed_tokens: list[str] = []
    for token in rest.split():
        if token.count("=") != 1:
            unparsed_tokens.append(token)
            continue
        key, value = token.split("=", 1)
        if not key or not KEY_RE.match(key):
            unparsed_tokens.append(token)
            continue
        if key in fields:
            duplicate_keys.append(key)
        fields[key] = coerce_value(value)
    missing = sorted(RECOMMENDED_FIELDS.get(marker, set()) - set(fields))
    return {
        "marker": marker,
        "fields": fields,
        "duplicate_keys": duplicate_keys,
        "unparsed_tokens": unparsed_tokens,
        "missing_recommended_fields": missing,
    }


def parse_log_file(log_path: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    offset = 0
    line_no = 0
    try:
        with log_path.open("rb") as handle:
            for raw_line in handle:
                line_no += 1
                start = offset
                end = start + len(raw_line)
                offset = end
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                for marker in MARKERS:
                    parsed = parse_marker_line(line, marker)
                    if parsed is None:
                        continue
                    parsed.update({
                        "line_no": line_no,
                        "byte_start": start,
                        "byte_end": end,
                        "raw": line,
                    })
                    if parsed["duplicate_keys"]:
                        warnings.append(f"line {line_no}: duplicate keys in {marker}: {','.join(parsed['duplicate_keys'])}")
                    if parsed["unparsed_tokens"]:
                        warnings.append(f"line {line_no}: unparsed tokens in {marker}: {','.join(parsed['unparsed_tokens'][:5])}")
                    if parsed["missing_recommended_fields"]:
                        warnings.append(f"line {line_no}: missing recommended fields in {marker}: {','.join(parsed['missing_recommended_fields'])}")
                    events.append(parsed)
                    break
    except OSError as exc:
        fail(f"cannot read log {log_path}: {exc}")
    if not any(e["marker"] == "memory_governor_observe" for e in events):
        warnings.append("no memory_governor_observe markers found")
    return events, warnings


def write_events_jsonl(path: pathlib.Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")


def marker_events(events: list[dict[str, Any]], marker: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("marker") == marker]


def field_values(events: list[dict[str, Any]], key: str) -> list[Any]:
    values = []
    for event in events:
        fields = event.get("fields", {})
        if key in fields:
            values.append(fields[key])
    return values


def numeric_values(events: list[dict[str, Any]], key: str) -> list[int | float]:
    return [v for v in field_values(events, key) if isinstance(v, (int, float))]


def last_value(events: list[dict[str, Any]], key: str, default: Any = None) -> Any:
    values = field_values(events, key)
    return values[-1] if values else default


def max_value(events: list[dict[str, Any]], key: str) -> int | float | None:
    values = numeric_values(events, key)
    return max(values) if values else None


def min_value(events: list[dict[str, Any]], key: str) -> int | float | None:
    values = numeric_values(events, key)
    return min(values) if values else None


def delta_value(events: list[dict[str, Any]], key: str) -> int | float | None:
    values = numeric_values(events, key)
    if not values:
        return None
    return values[-1] - values[0]


def count_truthy(events: list[dict[str, Any]], key: str) -> int:
    count = 0
    for value in field_values(events, key):
        if value in (1, "1", True, "true", "True", "yes", "on"):
            count += 1
    return count


def aggregate_dense(obs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "enabled_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "dense_flex_enabled")),
        "resident_bytes_max": max_value(obs, "dense_resident_bytes"),
        "resident_bytes_last": last_value(obs, "dense_resident_bytes"),
        "reclaimable_bytes_max": max_value(obs, "dense_reclaimable_bytes"),
        "ring_bytes_max": max_value(obs, "dense_ring_bytes"),
        "locked_bytes_last": last_value(obs, "dense_locked_bytes"),
        "delta_locked_bytes_last": last_value(obs, "dense_delta_locked_bytes"),
        "stream_per_token_bytes_last": last_value(obs, "dense_stream_per_token_bytes"),
        "bytes_streamed_delta": delta_value(obs, "dense_bytes_streamed"),
        "bytes_read_phys_delta": delta_value(obs, "dense_bytes_read_phys"),
        "read_ops_delta": delta_value(obs, "dense_read_ops"),
        "effective_ahead_last": last_value(obs, "dense_effective_ahead"),
        "prefetch_budget_dropped_delta": delta_value(obs, "dense_prefetch_budget_dropped"),
        "repin_attempts": count_truthy(obs, "dense_repin_attempted"),
        "resize_attempts": count_truthy(obs, "dense_resize_attempted"),
        "repin_reason_last": last_value(obs, "dense_repin_reason"),
        "resize_reason_last": last_value(obs, "dense_resize_reason"),
    }


def aggregate_moe(obs: list[dict[str, Any]]) -> dict[str, Any]:
    hits_last = last_value(obs, "moe_cache_hits", 0)
    misses_last = last_value(obs, "moe_cache_misses", 0)
    hit_rate = None
    if isinstance(hits_last, (int, float)) and isinstance(misses_last, (int, float)) and hits_last + misses_last > 0:
        hit_rate = hits_last / (hits_last + misses_last)
    actions = Counter(str(v) for v in field_values(obs, "moe_budget_action"))
    return {
        "enabled_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "moe_enabled")),
        "resident_bytes_max": max_value(obs, "moe_resident_bytes"),
        "resident_bytes_last": last_value(obs, "moe_resident_bytes"),
        "budget_bytes_last": last_value(obs, "moe_budget_bytes"),
        "budget_unbounded_last": last_value(obs, "moe_budget_unbounded"),
        "expert_bytes_last": last_value(obs, "moe_expert_bytes"),
        "streams_delta": delta_value(obs, "moe_streams"),
        "hits_delta": delta_value(obs, "moe_hits"),
        "evictions_delta": delta_value(obs, "moe_evictions"),
        "bytes_read_delta": delta_value(obs, "moe_bytes_read"),
        "cache_hits_last": hits_last,
        "cache_misses_last": misses_last,
        "cache_hit_rate": hit_rate,
        "prefetch_hits_delta": delta_value(obs, "moe_prefetch_hits"),
        "prefetch_late_delta": delta_value(obs, "moe_prefetch_late"),
        "prefetch_unused_delta": delta_value(obs, "moe_prefetch_unused"),
        "warm_working_set_bytes_last": last_value(obs, "moe_warm_working_set_bytes"),
        "warm_working_set_groups_last": last_value(obs, "moe_warm_working_set_groups"),
        "budget_actions": dict(actions),
        "budget_reason_last": last_value(obs, "moe_budget_reason"),
    }


def aggregate_kv(obs: list[dict[str, Any]]) -> dict[str, Any]:
    sources = sorted({str(v) for v in field_values(obs, "kv_effective_budget_source")})
    return {
        "memory_present_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "kv_memory_present")),
        "resident_bytes_max": max_value(obs, "kv_resident_bytes"),
        "resident_bytes_last": last_value(obs, "kv_resident_bytes"),
        "reclaimable_resident_bytes_max": max_value(obs, "kv_reclaimable_resident_bytes"),
        "slot_budget_valid_seen": any(v in (1, "1", True, "true", "True") for v in field_values(obs, "kv_slot_budget_valid")),
        "slot_resident_bytes_max": max_value(obs, "kv_slot_resident_bytes"),
        "slot_reclaimable_resident_bytes_max": max_value(obs, "kv_slot_reclaimable_resident_bytes"),
        "effective_budget_sources_seen": sources,
        "effective_resident_bytes_max": max_value(obs, "kv_effective_resident_bytes"),
        "effective_reclaimable_resident_bytes_max": max_value(obs, "kv_effective_reclaimable_resident_bytes"),
        "soft_excess_bytes_max": max_value(obs, "kv_soft_excess_bytes"),
        "release_attempts": count_truthy(obs, "kv_release_attempted"),
        "release_relieved_bytes_sum": sum(v for v in numeric_values(obs, "kv_release_relieved_bytes")),
        "release_reason_last": last_value(obs, "kv_release_reason"),
        "offload_attempts": count_truthy(obs, "kv_offload_attempted"),
        "offload_relieved_bytes_sum": sum(v for v in numeric_values(obs, "kv_offload_relieved_bytes")),
        "offload_backend_last": last_value(obs, "kv_offload_backend"),
        "offload_reason_last": last_value(obs, "kv_offload_reason"),
    }


def aggregate_pressure(obs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "states_seen": sorted({str(v) for v in field_values(obs, "effective_pressure_state")}),
        "reason_last": last_value(obs, "effective_pressure_reason"),
        "pressure_current_bytes_max": max_value(obs, "pressure_current_bytes"),
        "pressure_excess_bytes_max": max_value(obs, "pressure_excess_bytes"),
        "rss_observed_bytes_max": max_value(obs, "rss_observed_bytes"),
        "prefetch_budget_effective_bytes_last": last_value(obs, "prefetch_budget_effective_bytes"),
        "prefetch_budget_tick_bytes_last": last_value(obs, "prefetch_budget_tick_bytes"),
        "prefetch_budget_dense_bytes_last": last_value(obs, "prefetch_budget_dense_bytes"),
        "prefetch_budget_moe_bytes_last": last_value(obs, "prefetch_budget_moe_bytes"),
        "prefetch_budget_kv_resume_used_bytes_delta": delta_value(obs, "prefetch_budget_kv_resume_used_bytes"),
        "auction_selected_allocation_counts": dict(Counter(str(v) for v in field_values(obs, "auction_selected_allocation"))),
        "auction_selected_reclaim_counts": dict(Counter(str(v) for v in field_values(obs, "auction_selected_reclaim"))),
        "global_optimizer_decision_last": last_value(obs, "global_optimizer_decision"),
        "reallocation_reason_last": last_value(obs, "reallocation_reason"),
    }


def read_tsv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    records = []
    for line in lines[1:]:
        parts = line.split("\t")
        records.append({key: parts[i] if i < len(parts) else "NA" for i, key in enumerate(header)})
    return records


def int_or_none(value: str | None) -> int | None:
    if value is None or value == "NA" or value == "max":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def aggregate_tsv_samples(path: pathlib.Path, fields: list[str], deltas: list[str] | None = None) -> dict[str, Any]:
    records = read_tsv(path)
    result: dict[str, Any] = {"sample_count": len(records)}
    for field in fields:
        vals = [v for v in (int_or_none(r.get(field)) for r in records) if v is not None]
        result[f"{field}_max"] = max(vals) if vals else None
        result[f"{field}_last"] = vals[-1] if vals else None
    for field in deltas or []:
        vals = [v for v in (int_or_none(r.get(field)) for r in records) if v is not None]
        result[f"{field}_delta"] = vals[-1] - vals[0] if vals else None
    return result


def aggregate_all(events: list[dict[str, Any]], out_dir: pathlib.Path, mode: str, warnings: list[str],
                  request_metrics: dict[str, Any] | None, action_enabled: bool,
                  inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    obs = marker_events(events, "memory_governor_observe")
    marker_counts = Counter(e["marker"] for e in events)
    summary = {
        "schema": "memory_behavior_profile/v1",
        "mode": mode,
        "observation_only": not action_enabled,
        "action_enabled": action_enabled,
        "inputs": inputs or {},
        "request_metrics": request_metrics or {},
        "markers": {"counts": dict(marker_counts)},
        "dense": aggregate_dense(obs),
        "moe": aggregate_moe(obs),
        "kv": aggregate_kv(obs),
        "pressure": aggregate_pressure(obs),
        "proc": aggregate_tsv_samples(out_dir / "proc-samples.tsv", ["VmRSS_kB", "VmHWM_kB", "RssAnon_kB", "RssFile_kB", "smaps_Rss_kB"], ["read_bytes", "write_bytes", "syscr", "syscw"]),
        "cgroup": aggregate_tsv_samples(out_dir / "cgroup-samples.tsv", ["memory_current", "memory_peak"], ["events_high", "events_max", "events_oom", "events_oom_kill"]),
        "warnings": warnings,
        "evidence_limits": [
            "memory_governor_observe marker is sampled; it is not a per-token complete trace.",
            "MoE expert activation is inferred from runtime counters; exact router logits are not captured.",
            "Dense parameter loading is inferred from Dense Flex counters; this is not hardware PMU-level memory tracing.",
            "KV usage is inferred from resident/reclaimable/slot/action markers; observation-only runs do not validate release/offload correctness.",
            "This script records evidence and does not claim performance improvement.",
        ],
    }
    return summary


def write_summary_tsv(path: pathlib.Path, summary: dict[str, Any], out_dir: pathlib.Path) -> None:
    def get(path_str: str) -> Any:
        cur: Any = summary
        for part in path_str.split("."):
            if not isinstance(cur, dict):
                return None
            cur = cur.get(part)
        return cur

    columns = [
        ("status", "ok"),
        ("mode", "mode"),
        ("observation_only", "observation_only"),
        ("marker_memory_governor_observe_count", "markers.counts.memory_governor_observe"),
        ("marker_kv_pressure_telemetry_count", "markers.counts.kv_pressure_telemetry"),
        ("ttft_ms", "request_metrics.ttft_ms"),
        ("tpot_ms", "request_metrics.server_tpot_ms"),
        ("predicted_per_second", "request_metrics.predicted_per_second"),
        ("rss_observed_bytes_max", "pressure.rss_observed_bytes_max"),
        ("proc_VmRSS_kB_max", "proc.VmRSS_kB_max"),
        ("cgroup_memory_current_max", "cgroup.memory_current_max"),
        ("dense_enabled", "dense.enabled_seen"),
        ("dense_resident_bytes_max", "dense.resident_bytes_max"),
        ("dense_locked_bytes_last", "dense.locked_bytes_last"),
        ("dense_bytes_read_phys_delta", "dense.bytes_read_phys_delta"),
        ("dense_read_ops_delta", "dense.read_ops_delta"),
        ("dense_prefetch_budget_dropped_delta", "dense.prefetch_budget_dropped_delta"),
        ("moe_enabled", "moe.enabled_seen"),
        ("moe_resident_bytes_max", "moe.resident_bytes_max"),
        ("moe_bytes_read_delta", "moe.bytes_read_delta"),
        ("moe_streams_delta", "moe.streams_delta"),
        ("moe_evictions_delta", "moe.evictions_delta"),
        ("moe_cache_hit_rate", "moe.cache_hit_rate"),
        ("kv_memory_present", "kv.memory_present_seen"),
        ("kv_resident_bytes_max", "kv.resident_bytes_max"),
        ("kv_slot_resident_bytes_max", "kv.slot_resident_bytes_max"),
        ("kv_reclaimable_resident_bytes_max", "kv.reclaimable_resident_bytes_max"),
        ("kv_release_attempts", "kv.release_attempts"),
        ("kv_offload_attempts", "kv.offload_attempts"),
        ("pressure_states_seen", "pressure.states_seen"),
        ("prefetch_budget_effective_bytes_last", "pressure.prefetch_budget_effective_bytes_last"),
        ("warnings_count", "warnings"),
        ("out_dir", str(out_dir)),
    ]
    headers = [name for name, _ in columns]
    values = []
    for _, spec in columns:
        if spec == "ok" or spec == str(out_dir):
            value = spec
        elif spec == "warnings":
            value = len(summary.get("warnings", []))
        else:
            value = get(spec)
        if isinstance(value, list):
            value = ",".join(map(str, value))
        elif value is None:
            value = "NA"
        values.append(str(value))
    path.write_text("\t".join(headers) + "\n" + "\t".join(values) + "\n", encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return ", ".join(map(str, value)) if value else "NA"
    return str(value)


def write_report_md(path: pathlib.Path, summary: dict[str, Any]) -> None:
    dense = summary["dense"]
    moe = summary["moe"]
    kv = summary["kv"]
    pressure = summary["pressure"]
    req = summary.get("request_metrics", {})
    lines = [
        "# LLM Memory Behavior Profile",
        "",
        "## 结论摘要",
        "",
        f"- 运行模式：`{summary['mode']}`。",
        f"- observation-only：`{summary['observation_only']}`；action-enabled：`{summary['action_enabled']}`。",
        f"- `memory_governor_observe` 条数：`{summary['markers']['counts'].get('memory_governor_observe', 0)}`。",
        f"- `kv_pressure_telemetry` 条数：`{summary['markers']['counts'].get('kv_pressure_telemetry', 0)}`。",
        "",
        "## 请求指标",
        "",
        f"- TTFT(ms)：{fmt(req.get('ttft_ms'))}",
        f"- Server TPOT(ms)：{fmt(req.get('server_tpot_ms'))}",
        f"- predicted tokens/s：{fmt(req.get('predicted_per_second'))}",
        "",
        "## Dense 参数加载观察",
        "",
        f"- Dense Flex enabled seen：{fmt(dense.get('enabled_seen'))}",
        f"- resident bytes max：{fmt(dense.get('resident_bytes_max'))}",
        f"- locked bytes last：{fmt(dense.get('locked_bytes_last'))}",
        f"- bytes read phys delta：{fmt(dense.get('bytes_read_phys_delta'))}",
        f"- read ops delta：{fmt(dense.get('read_ops_delta'))}",
        f"- effective ahead last：{fmt(dense.get('effective_ahead_last'))}",
        f"- prefetch dropped delta：{fmt(dense.get('prefetch_budget_dropped_delta'))}",
        "",
        "## MoE 专家激活/缓存观察",
        "",
        f"- MoE enabled seen：{fmt(moe.get('enabled_seen'))}",
        f"- resident bytes max：{fmt(moe.get('resident_bytes_max'))}",
        f"- budget bytes last：{fmt(moe.get('budget_bytes_last'))}",
        f"- streams delta：{fmt(moe.get('streams_delta'))}",
        f"- bytes read delta：{fmt(moe.get('bytes_read_delta'))}",
        f"- evictions delta：{fmt(moe.get('evictions_delta'))}",
        f"- cache hit rate：{fmt(moe.get('cache_hit_rate'))}",
        f"- prefetch late delta：{fmt(moe.get('prefetch_late_delta'))}",
        "",
        "## KV Cache 使用观察",
        "",
        f"- KV memory present seen：{fmt(kv.get('memory_present_seen'))}",
        f"- resident bytes max：{fmt(kv.get('resident_bytes_max'))}",
        f"- slot resident bytes max：{fmt(kv.get('slot_resident_bytes_max'))}",
        f"- reclaimable resident bytes max：{fmt(kv.get('reclaimable_resident_bytes_max'))}",
        f"- effective budget sources：{fmt(kv.get('effective_budget_sources_seen'))}",
        f"- release attempts：{fmt(kv.get('release_attempts'))}",
        f"- offload attempts：{fmt(kv.get('offload_attempts'))}",
        "",
        "## Pressure / Prefetch / Auction 观察",
        "",
        f"- pressure states seen：{fmt(pressure.get('states_seen'))}",
        f"- pressure current bytes max：{fmt(pressure.get('pressure_current_bytes_max'))}",
        f"- pressure excess bytes max：{fmt(pressure.get('pressure_excess_bytes_max'))}",
        f"- rss observed bytes max：{fmt(pressure.get('rss_observed_bytes_max'))}",
        f"- prefetch budget effective bytes last：{fmt(pressure.get('prefetch_budget_effective_bytes_last'))}",
        f"- global optimizer decision last：{fmt(pressure.get('global_optimizer_decision_last'))}",
        "",
        "## Proc / Cgroup 观察",
        "",
        f"- proc samples：{fmt(summary['proc'].get('sample_count'))}",
        f"- proc VmRSS_kB max：{fmt(summary['proc'].get('VmRSS_kB_max'))}",
        f"- proc read_bytes delta：{fmt(summary['proc'].get('read_bytes_delta'))}",
        f"- cgroup samples：{fmt(summary['cgroup'].get('sample_count'))}",
        f"- cgroup memory.current max：{fmt(summary['cgroup'].get('memory_current_max'))}",
        "",
        "## 证据边界",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("evidence_limits", []))
    if summary.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {w}" for w in summary["warnings"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(out_dir: pathlib.Path, events: list[dict[str, Any]], warnings: list[str], mode: str,
                  request_metrics: dict[str, Any] | None, action_enabled: bool,
                  inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    write_events_jsonl(out_dir / "memory-events.jsonl", events)
    summary = aggregate_all(events, out_dir, mode, warnings, request_metrics, action_enabled, inputs)
    write_json(out_dir / "summary.json", summary)
    write_summary_tsv(out_dir / "summary.tsv", summary, out_dir)
    write_report_md(out_dir / "report.md", summary)
    write_inventory(out_dir)
    return summary


def command_parse(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = pathlib.Path(args.log)
    write_json(out_dir / "parse-input.json", {"log": str(log), "stdout": args.stdout})
    events, warnings = parse_log_file(log)
    if args.strict and not marker_events(events, "memory_governor_observe"):
        fail("strict mode requires at least one memory_governor_observe marker")
    write_outputs(out_dir, events, warnings, "parse", None, False, {"log": identity(log)})
    print(f"wrote {out_dir}")


def command_run(args: argparse.Namespace) -> None:
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binary = pathlib.Path(args.binary)
    model = pathlib.Path(args.model)
    if not binary.is_file():
        fail(f"missing binary: {binary}")
    if not model.is_file():
        fail(f"missing model: {model}")
    port = args.port or free_port()
    argv = build_server_argv(args, port)
    env = build_env(args)
    prompt = load_prompt(args)
    action_enabled = bool(args.enable_actions)
    execution = {
        "argv": argv,
        "cwd": str(ROOT),
        "host": args.host,
        "port": port,
        "dry_run": args.dry_run,
        "started_wall_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    effective_env = {key: env[key] for key in sorted(env) if key.startswith("LLAMA_")}
    write_json(out_dir / "execution.json", execution)
    write_json(out_dir / "environment.json", {"effective_llama_env": effective_env})
    write_json(out_dir / "request.json", {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "n_predict": args.n_predict, "stream": args.stream})
    if args.dry_run:
        write_inventory(out_dir)
        print(f"dry-run wrote {out_dir}")
        return

    stdout_path = out_dir / "server.stdout"
    stderr_path = out_dir / "server.stderr"
    proc: subprocess.Popen[bytes] | None = None
    sampler: SamplerThread | None = None
    request_metrics: dict[str, Any] | None = None
    lifecycle: dict[str, Any] = {}
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            proc = subprocess.Popen(argv, cwd=str(ROOT), env=env, stdout=stdout, stderr=stderr, start_new_session=True)
            write_json(out_dir / "process.json", {"pid": proc.pid, "binary": identity(binary), "model": identity(model)})
            cgroup_path = pathlib.Path(args.cgroup_path) if args.cgroup_path else None
            if cgroup_path is not None and not cgroup_path.exists():
                cgroup_path = None
                lifecycle.setdefault("warnings", []).append("cgroup path does not exist; cgroup sampling disabled")
            sampler = SamplerThread(proc.pid, args.sample_ms, out_dir, cgroup_path)
            sampler.start()
            health = wait_health(args.host, port, proc, args.startup_timeout_sec)
            write_json(out_dir / "health.json", health)
            if not health.get("ok"):
                fail(f"server health failed: {health}")
            request_metrics = stream_completion(args.host, port, prompt, args, out_dir / ("response.sse" if args.stream else "response.json"), out_dir / "response.events.json")
            lifecycle["request_metrics"] = request_metrics
    finally:
        if sampler is not None:
            sampler.stop()
            sampler.join(timeout=5)
        if proc is not None:
            lifecycle["termination"] = terminate_process_group(proc)
        write_json(out_dir / "execution-result.json", lifecycle)

    events, warnings = parse_log_file(stderr_path)
    warnings.extend(lifecycle.get("warnings", []))
    inputs = {
        "binary": identity(binary),
        "model": identity(model),
        "argv": argv,
        "env_effective": effective_env,
    }
    write_outputs(out_dir, events, warnings, "run", request_metrics, action_enabled, inputs)
    print(f"wrote {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile llama-server memory behavior from runtime markers and samples.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="start llama-server, run one completion, and profile memory behavior")
    run.add_argument("--binary", default=str(DEFAULT_BINARY))
    run.add_argument("--model", required=True)
    run.add_argument("--out-dir", required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=0)
    run.add_argument("--ctx-size", type=int, default=2048)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--ubatch-size", type=int, default=64)
    run.add_argument("--threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--temp", type=float, default=0.0)
    run.add_argument("--n-predict", type=int, default=64)
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--prompt-file")
    stream_group = run.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", default=True)
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    run.add_argument("--cache-prompt", action="store_true")
    run.add_argument("--startup-timeout-sec", type=float, default=240.0)
    run.add_argument("--request-timeout-sec", type=float, default=300.0)
    run.add_argument("--log-verbosity", type=int, default=4)
    run.add_argument("--no-warmup", action="store_true", default=True)
    run.add_argument("--observe-ms", type=int, default=500)
    run.add_argument("--auto-backends", action="store_true")
    run.add_argument("--env", action="append", default=[])
    run.add_argument("--flex-trace")
    run.add_argument("--sample-ms", type=int, default=200)
    run.add_argument("--cgroup-path")
    run.add_argument("--enable-actions", action="store_true")
    run.add_argument("--allow-unsafe-env", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--extra-server-arg", action="append", default=[])

    parse = sub.add_parser("parse", help="parse an existing llama-server log")
    parse.add_argument("--log", required=True)
    parse.add_argument("--stdout")
    parse.add_argument("--out-dir", required=True)
    parse.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "run":
        if args.prompt_file and args.prompt != DEFAULT_PROMPT:
            fail("--prompt and --prompt-file are mutually exclusive")
        command_run(args)
    elif args.command == "parse":
        command_parse(args)
    else:
        fail(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
