#!/usr/bin/env python3
"""Real-model, fail-closed Stage 3C Governor smoke for unified multi-slot KV."""
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
PROTOCOL = "kv_governor_stage3c_1c_2b_1r"
PARALLELS = (2, 3)
N_PREDICT = 8
SEED = 1
PROMPT = ("KV governor unified multi-slot smoke context. Keep the answer deterministic. " * 48).strip()
BASE_ENV = {"HOME": "/tmp", "LANG": "C", "LC_ALL": "C", "PATH": os.environ.get("PATH", "")}


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


def request(port: int, body: dict[str, Any], label: str, record: pathlib.Path) -> int:
    encoded = json.dumps(body, separators=(",", ":")).encode()
    item: dict[str, Any] = {"label": label, "request": body, "request_sha256": hashlib.sha256(encoded).hexdigest()}
    status, raw, data = 0, b"", {}
    try:
        con = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
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
        except Exception:
            pass
    text = data.get("content", "") if isinstance(data.get("content"), str) else ""
    if not text and isinstance(data.get("choices"), list) and data["choices"]:
        text = str(data["choices"][0].get("text", ""))
    item.update({"http_status": status, "response": data, "response_raw": raw.decode(errors="replace"),
                 "response_text": text, "response_sha256": hashlib.sha256(text.encode()).hexdigest()})
    with record.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")
    return status


def wait_health(port: int, proc: subprocess.Popen[bytes], seconds: float = 120) -> bool:
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


def stop(proc: subprocess.Popen[bytes], case: pathlib.Path) -> None:
    pgid = os.getpgid(proc.pid)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(pgid, signal.SIGKILL)
        proc.wait()
    for handle in (proc.stdout, proc.stderr):
        if handle and not handle.closed:
            handle.close()
    try:
        os.killpg(pgid, 0)
        residual = True
    except ProcessLookupError:
        residual = False
    dump(case / "process.json", {"pid": proc.pid, "pgid": pgid, "exit_code": proc.returncode, "residual_process": residual})


def governor_env(enabled: bool) -> dict[str, str]:
    env = dict(BASE_ENV)
    env.update({
        "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1",
        "LLAMA_KV_PRESSURE_SAMPLER": "1", "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "100",
        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "100",
        "LLAMA_KV_LOW_WATER_RSS_KB": "1", "LLAMA_KV_PRESSURE_RSS_KB": "2", "LLAMA_KV_CRITICAL_RSS_KB": "3",
    })
    if enabled:
        env.update({"LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
                    "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "1048576",
                    "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": "64"})
    return env


def start(binary: str, model: str, env: dict[str, str], case: pathlib.Path, port: int, parallel: int) -> subprocess.Popen[bytes]:
    argv = [binary, "--host", "127.0.0.1", "--port", str(port), "--model", model, "--ctx-size", "2048",
            "--parallel", str(parallel), "--kv-unified", "--threads", "4", "--n-gpu-layers", "0",
            "--cache-type-k", "f32", "--cache-type-v", "f32", "--no-warmup"]
    dump(case / "execution.json", {"argv": argv, "environment": env, "binary": identity(pathlib.Path(binary)), "model": identity(pathlib.Path(model))})
    return subprocess.Popen(argv, stdout=(case / "server.stdout").open("wb"), stderr=(case / "server.stderr").open("wb"), env=env, preexec_fn=os.setsid)


def completion_body(slot: int, suffix: str, n_predict: int = N_PREDICT) -> dict[str, Any]:
    return {"prompt": f"{PROMPT} {suffix}", "n_predict": n_predict, "temperature": 0.0, "seed": SEED,
            "cache_prompt": True, "id_slot": slot, "stream": False}


def run_case(name: str, binary: str, model: str, enabled: bool, root: pathlib.Path, parallel: int) -> dict[str, Any]:
    case = root / name
    case.mkdir()
    env, record, port = governor_env(enabled), case / "requests.jsonl", free_port()
    dump(case / "environment.json", env)
    proc = start(binary, model, env, case, port, parallel)
    if not wait_health(port, proc):
        stop(proc, case)
        dump(case / "result.json", {"status": "startup_failed", "port": port})
        return {"status": "startup_failed", "port": port}
    statuses: dict[str, int] = {}
    try:
        for cycle in range(2):
            for slot in range(parallel):
                label = f"seed_c{cycle}_s{slot}"
                statuses[label] = request(port, completion_body(slot, f"cycle={cycle} slot={slot}"), label, record)
        active_slot = parallel - 1
        active: dict[str, int] = {}
        worker = threading.Thread(target=lambda: active.setdefault("status", request(port, completion_body(active_slot, f"active slot={active_slot}", 128), f"active_s{active_slot}", record)), daemon=True)
        worker.start()
        time.sleep(0.35)
        time.sleep(3.0)
        worker.join(timeout=150)
        statuses[f"active_s{active_slot}"] = active.get("status", 0)
        before = (case / "server.stderr").read_bytes()
        for slot in range(parallel):
            label = f"reaccess_s{slot}"
            statuses[label] = request(port, completion_body(slot, f"cycle=1 slot={slot}"), label, record)
        time.sleep(1.0)
        after = (case / "server.stderr").read_bytes()
        for slot in range(parallel):
            label = f"reuse_c2_s{slot}"
            statuses[label] = request(port, completion_body(slot, f"cycle=2 slot={slot}"), label, record)
        time.sleep(2.0)
        result = {"status": "complete" if all(code == 200 for code in statuses.values()) else "request_failed",
                  "http_statuses": statuses, "reaccess_stderr_start": len(before), "reaccess_stderr_end": len(after)}
        dump(case / "result.json", result)
        return {"status": result["status"], "port": port}
    finally:
        stop(proc, case)


def run_negative(name: str, binary: str, model: str, extra: dict[str, str], root: pathlib.Path, parallel: int) -> dict[str, Any]:
    case = root / name
    case.mkdir()
    env = governor_env(True)
    env.update(extra)
    dump(case / "environment.json", env)
    port = free_port()
    proc = start(binary, model, env, case, port, parallel)
    health = wait_health(port, proc, seconds=3)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        stop(proc, case)
    else:
        for handle in (proc.stdout, proc.stderr):
            if handle and not handle.closed:
                handle.close()
        dump(case / "process.json", {"pid": proc.pid, "exit_code": proc.returncode, "residual_process": False})
    result = {"health_reached": health, "exit_code": proc.returncode,
              "rejected_before_request_loop": not health and proc.returncode not in (None, 0)}
    dump(case / "result.json", result)
    return result


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
    binary, model = pathlib.Path(args.binary).resolve(), pathlib.Path(args.model).resolve() if args.model else None
    manifest: dict[str, Any] = {"protocol": PROTOCOL, "protocol_version": 2, "timestamp_utc": stamp,
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"), "dirty_status": git("status", "--porcelain").splitlines(),
        "runner": identity(pathlib.Path(__file__)), "parser": identity(ROOT / "scripts/parse-kv-governor-stage3c-1c-2b-1r.py"),
        "binary_requested": str(binary), "model_requested": str(model) if model else None,
        "parameters": {"parallels": list(PARALLELS), "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(), "seed": SEED, "n_predict": N_PREDICT, "temperature": 0.0, "kv_unified": True}, "runner_status": "run_in_progress"}
    if not binary.is_file() or model is None or not model.is_file():
        manifest.update({"runner_status": "UNSUPPORTED", "unsupported_reason": "binary or model is unavailable; no real server/model evidence was fabricated"})
        if binary.is_file(): manifest["binary"] = identity(binary)
        dump(out / "manifest.json", manifest)
        raise SystemExit(3)
    manifest.update({"binary": identity(binary), "model": identity(model)})
    dump(out / "manifest.json", manifest)
    runs, incomplete = {}, False
    for parallel in PARALLELS:
        root = out / f"parallel_{parallel}"
        root.mkdir()
        cases = {"OFF": run_case("OFF", str(binary), str(model), False, root, parallel), "GOVERNOR_ON": run_case("GOVERNOR_ON", str(binary), str(model), True, root, parallel), "INVALID_UNIFIED": run_negative("INVALID_UNIFIED", str(binary), str(model), {"LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "invalid"}, root, parallel), "CONFLICT_UNIFIED_LEGACY": run_negative("CONFLICT_UNIFIED_LEGACY", str(binary), str(model), {"LLAMA_KV_PAGED_RELEASE": "1"}, root, parallel)}
        incomplete |= any(cases[name].get("status") != "complete" for name in ("OFF", "GOVERNOR_ON"))
        runs[str(parallel)] = {"parallel": parallel, "cases": cases}
    manifest.update({"runs": runs, "runner_status": "run_incomplete" if incomplete else "run_complete"})
    dump(out / "manifest.json", manifest)
    parser = ROOT / "scripts/parse-kv-governor-stage3c-1c-2b-1r.py"
    raise SystemExit(subprocess.run([sys.executable, str(parser), str(out), "--result-path", str(out / "parser.json")], text=True).returncode)


if __name__ == "__main__":
    main()
