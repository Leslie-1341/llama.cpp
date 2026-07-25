#!/usr/bin/env python3
"""Stage 3B-1 v2: calibrated dynamic target bounded release with reuse verification.

Phase 0 — RSS Calibration:
  Start server with same binary/model/request.  Measure idle and post-completion
  RSS, then derive pressure/water thresholds so that:
    - OFF / DYNAMIC_NOOP → thresholds above peak → stay NORMAL
    - FIXED / DYNAMIC_RELEASE → thresholds between idle and peak → PRESSURE/CRITICAL

Phase 1–4 — Four cases:
  OFF            — single request, no bounded release
  FIXED          — bounded release (fixed target), two requests → reuse/commit >0
  DYNAMIC_NOOP   — bounded release + dynamic target, NORMAL state, zero release
  DYNAMIC_RELEASE — bounded release + dynamic target, real release, two requests

Dynamic target formula: target = min(water_excess, hard_cap, KV_resident, KV_reclaimable_resident)

All cases share the same binary, model, prompt, seed, and topology.
No hard-coded 1/2/3 KiB or 100 GB artificial thresholds.

Protocol: ./run-kv-bounded-release-stage3b-1.py --binary build/bin/llama-server
          --model <path> [--output-dir /path/to/artifact]
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
import time
import uuid
from datetime import datetime, timezone
from typing import Any, NoReturn

ROOT = pathlib.Path(__file__).resolve().parents[1]

# ── constants ────────────────────────────────────────────────────────────────

BASE_ENV: dict[str, str] = {
    "HOME": "/tmp", "LANG": "C", "LC_ALL": "C",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "TMPDIR": "/tmp", "TZ": "UTC",
}

FORBIDDEN_ENV: dict[str, str] = {
    "LLAMA_KV_PAGED_RELEASE": "0",
    "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_SWAP": "0",
}

UNSET_ENV: set[str] = {
    "LLAMA_KV_LAZY_TAIL",
    "LLAMA_KV_LAZY_CLEAR",
}

EXPERIMENT_ENV_PREFIXES = ("LLAMA_KV_", "LLAMA_", "GGML_", "GGUF_")

# Shared across all four cases — must be byte-identical (except per-case keys).
# PRESSURE_RSS_KB, CRITICAL_RSS_KB, and LOW_WATER_RSS_KB come from calibration.
SHARED_ENV: dict[str, str] = {
    "LLAMA_KV_PAGED": "1",
    "LLAMA_KV_PAGED_INGRAPH": "1",
    "LLAMA_KV_PAGED_MINCORE": "1",
    "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
    "LLAMA_KV_PRESSURE_SAMPLER": "1",
    "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "250",
    "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
}

# Keys that are set per-case from calibration (may legitimately differ).
CALIBRATED_KEYS = {
    "LLAMA_KV_PRESSURE_RSS_KB",
    "LLAMA_KV_CRITICAL_RSS_KB",
    "LLAMA_KV_LOW_WATER_RSS_KB",
}

# Bounded-release keys
BOUNDED_ONLY_KEYS = {
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET",
}

FIXED_TARGET_BYTES = 33554432       # 32 MiB
DYNAMIC_HARD_CAP_BYTES = 1073741824  # 1 GiB
MAX_SCAN_BLOCKS = 64

PROMPT = "In one short sentence, explain why deterministic tests are useful."
N_PREDICT = 32
SEED = 1

TIMEOUTS_S = {
    "startup": 5.0, "health": 180.0, "completion": 120.0,
    "shutdown": 15.0, "case_total": 360.0, "post_attach_wait": 0.75,
    "calibration_health": 120.0, "release_wait": 10.0,
}

# ── helpers ───────────────────────────────────────────────────────────────────

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


def git_bytes(args_list: list[str]) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args_list])


def get_server_rss_kb(pid: int) -> int:
    """Read RSS of server process via /proc/<pid>/statm. Returns KiB."""
    try:
        with open(f"/proc/{pid}/statm", "r") as f:
            parts = f.read().split()
            if len(parts) >= 2:
                return int(parts[1]) * 4  # pages → KiB
    except Exception:
        pass
    return 0


def capture_source_snapshot(art_dir: pathlib.Path) -> dict[str, Any]:
    snapshot_dir = art_dir / "source_snapshot"
    untracked_dir = snapshot_dir / "untracked"
    snapshot_dir.mkdir()
    untracked_dir.mkdir()

    diff_path = snapshot_dir / "tracked.diff"
    diff_path.write_bytes(git_bytes(["diff", "--binary", "HEAD"]))

    raw = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    repo_paths = sorted(p.decode("utf-8", errors="strict") for p in raw.split(b"\0") if p)
    files: list[dict[str, Any]] = []
    for repo_path in repo_paths:
        rel = pathlib.PurePosixPath(repo_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise RuntimeError(f"unsafe untracked path: {repo_path!r}")
        source = ROOT / pathlib.Path(*rel.parts)
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"untracked snapshot source is not a regular file: {repo_path}")
        target = untracked_dir / pathlib.Path(*rel.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        target.chmod(source.stat().st_mode & 0o777)
        captured = identity(target)
        files.append({
            "status": "??",
            "path": repo_path,
            "snapshot_path": captured["path"],
            "size": captured["size"],
            "sha256": captured["sha256"],
            "mode": source.stat().st_mode & 0o777,
        })
    return {
        "schema_version": 1,
        "head_sha": git(["rev-parse", "HEAD"]),
        "tracked_diff": identity(diff_path),
        "untracked_files": files,
    }


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


def stream_completion(port: int, raw_path: pathlib.Path,
                      deadline: Deadline, phase_seconds: float) -> tuple[int, str]:
    body = json.dumps({
        "prompt": PROMPT, "n_predict": N_PREDICT, "stream": True,
        "seed": SEED, "temperature": 0.0, "cache_prompt": False,
    }).encode()
    phase_end = time.monotonic() + min(phase_seconds, deadline.remaining())
    conn = http.client.HTTPConnection("127.0.0.1", port,
                                      timeout=max(0.001, phase_end - time.monotonic()))
    raw = bytearray()
    status: int | None = None
    try:
        conn.request("POST", "/v1/completions", body,
                     {"Content-Type": "application/json"})
        response = conn.getresponse()
        status = response.status
        while time.monotonic() < phase_end:
            chunk = response.read(4096)
            if not chunk:
                break
            raw.extend(chunk)
    except Exception as exc:
        print(f"completion error: {exc}", file=sys.stderr)
    finally:
        conn.close()
        raw_path.write_bytes(raw)

    text_parts: list[str] = []
    for line in raw.decode(errors="replace").split("\n"):
        if line.startswith("data: ") and not line.startswith("data: [DONE]"):
            try:
                data = json.loads(line[6:])
                for choice in data.get("choices", []):
                    t = choice.get("text", "")
                    if t:
                        text_parts.append(t)
            except (json.JSONDecodeError, KeyError):
                pass
    return status or 0, "".join(text_parts)


def make_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    result = dict(BASE_ENV)
    result.update(FORBIDDEN_ENV)
    result.update(SHARED_ENV)
    if extra:
        result.update(extra)
    for key in UNSET_ENV:
        result.pop(key, None)
    return result


def controlled_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    result = make_env(extra)
    all_seeds = {*FORBIDDEN_ENV, *SHARED_ENV, *(extra or {})}
    if any(key.startswith(EXPERIMENT_ENV_PREFIXES) for key in result if key not in all_seeds):
        raise RuntimeError("controlled environment contains an unexpected experiment variable")
    return result


def wait_health(port: int, deadline: Deadline) -> None:
    start = time.monotonic()
    delay = 0.1
    while time.monotonic() < start + deadline.remaining():
        try:
            status, _ = request(port, "GET", "/health", deadline, 3.0)
            if status == 200:
                return
        except Exception:
            pass
        time.sleep(delay)
        delay = min(delay * 1.5, 1.0)
    fail("health endpoint did not respond")


def start_server(binary: str, port: int, model: str, env_vars: dict[str, str],
                 output_dir: pathlib.Path, strace: bool = False) -> subprocess.Popen:
    cmd = []
    if strace:
        cmd = ["strace", "-f", "-e", "trace=madvise",
               "-o", str(output_dir / "strace.log"), "--"]
    cmd += [binary, "--host", "127.0.0.1", "--port", str(port),
            "--model", model, "--ctx-size", "1024",
            "--n-gpu-layers", "0", "--threads", "4",
            "--batch-size", "128", "--ubatch-size", "128",
            "--parallel", "1",
            "--cache-ram", "0",
            "--cache-type-k", "f32", "--cache-type-v", "f32",
            "--no-warmup"]

    write_json(output_dir / "execution.json", {
        "argv": cmd, "strace": strace, "environment": env_vars,
        "cleared_inherited_prefixes": list(EXPERIMENT_ENV_PREFIXES),
    })

    log = (output_dir / "server.stdout").open("wb")
    err = (output_dir / "server.stderr").open("wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=err, env=env_vars,
                           preexec_fn=os.setsid)
    proc.stage3b_1_pgid = os.getpgid(proc.pid)  # type: ignore[attr-defined]
    return proc


def kill_server(proc: subprocess.Popen, pgid: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        proc.wait(timeout=TIMEOUTS_S["shutdown"])
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait()


def record_cleanup(output_dir: pathlib.Path, proc: subprocess.Popen, pgid: int) -> None:
    """Write phases.json shutdown record after server exit."""
    time.sleep(0.5)

    residual = False
    pgid_check_complete = False
    cleanup_kill_attempted = False
    try:
        os.killpg(pgid, 0)
        residual = True
        pgid_check_complete = True
    except ProcessLookupError:
        pgid_check_complete = True
    except OSError:
        residual = True
    if residual:
        try:
            cleanup_kill_attempted = True
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        time.sleep(0.1)
        try:
            os.killpg(pgid, 0)
            pgid_check_complete = True
        except ProcessLookupError:
            residual = False
            pgid_check_complete = True
        except OSError:
            residual = True
            pgid_check_complete = False
    phase_path = output_dir / "phases.json"
    phases: dict[str, Any] = {}
    if phase_path.exists():
        phases = json.loads(phase_path.read_text(encoding="utf-8"))
    phases["shutdown"] = {
        "pgid": pgid, "exit_code": proc.returncode,
        "pgid_check_complete": pgid_check_complete,
        "cleanup_kill_attempted": cleanup_kill_attempted,
        "residual_process": residual,
    }
    write_json(phase_path, phases)
    for f in [proc.stdout, proc.stderr]:
        if f and not f.closed:
            f.close()


def check_capability(name: str, env_vars: dict[str, str],
                     output_dir: pathlib.Path) -> None:
    """Verify startup capability marker for bounded-release cases."""
    bounded_requested = env_vars.get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "0") == "1"
    if not bounded_requested:
        return
    stderr_text = (output_dir / "server.stderr").read_text(errors="replace")
    cap_match = re.search(
        r"kv_pressure_bounded_release_capability\s+"
        r"can_enable=(\d+)\s+paged=(\d+)\s+ingraph=(\d+)\s+"
        r"layers_supported=(\d+)\s+row_idx=(\d+)\s+"
        r"swap_disabled=(\d+)\s+layout_supported=(\d+)",
        stderr_text)
    if not cap_match:
        fail(f"{name}: capability marker not found in server stderr")
    cap = {k: int(v) for k, v in zip(
        ("can_enable", "paged", "ingraph", "layers_supported",
         "row_idx", "swap_disabled", "layout_supported"),
        cap_match.groups())}
    write_json(output_dir / "capability.json", cap)
    STARTUP_HARD = ("paged", "ingraph", "layers_supported",
                    "swap_disabled", "layout_supported")
    for field in STARTUP_HARD:
        if cap.get(field, -1) != 1:
            details = " ".join(f"{k}={v}" for k, v in cap.items())
            fail(f"{name}: startup capability {field}={cap.get(field)} — must be 1; "
                 f"full: {details}")
    if cap["row_idx"] != 1:
        print(f"  {name}: startup row_idx=0 (deferred)")


# ── RSS calibration ──────────────────────────────────────────────────────────

def run_calibration(binary: str, model: str, art_dir: pathlib.Path) -> dict[str, Any]:
    """Measure baseline and peak RSS, then derive threshold values.

    Returns calibration dict with all derived threshold values (KiB).
    """
    print("\n=== Phase 0: RSS Calibration ===")
    calib_dir = art_dir / "calibration"
    calib_dir.mkdir(parents=False, exist_ok=False)

    # Calibration server: no pressure sampler, no bounded release — just measure RSS
    calib_env = controlled_env({})
    calib_env.pop("LLAMA_KV_PRESSURE_SAMPLER", None)
    calib_env.pop("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", None)
    calib_env.pop("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS", None)
    # Remove any pressure threshold keys
    for k in list(calib_env):
        if k.startswith("LLAMA_KV_PRESSURE_RSS") or k.startswith("LLAMA_KV_CRITICAL"):
            calib_env.pop(k, None)

    port = free_port(set())
    proc = start_server(binary, port, model, calib_env, calib_dir, strace=False)
    pgid = proc.stage3b_1_pgid

    try:
        deadline = Deadline(TIMEOUTS_S["calibration_health"] + TIMEOUTS_S["completion"] + 30)
        wait_health(port, deadline)

        # Idle RSS (model loaded, no KV usage)
        rss_idle_kb = get_server_rss_kb(proc.pid)
        print(f"  RSS idle (after load): {rss_idle_kb} KiB")

        # Send request, wait for completion
        raw_path = calib_dir / "completion.sse"
        http_status, response_text = stream_completion(
            port, raw_path, deadline, TIMEOUTS_S["completion"])
        print(f"  calibration request: HTTP {http_status}  "
              f"len={len(response_text)}")

        # Post-completion RSS (KV cache populated)
        rss_peak_kb = get_server_rss_kb(proc.pid)
        print(f"  RSS peak (post-completion): {rss_peak_kb} KiB")

    finally:
        kill_server(proc, pgid)
        record_cleanup(calib_dir, proc, pgid)

    # ── derive thresholds ──
    delta_kb = max(1024, rss_peak_kb - rss_idle_kb)
    margin_kb = max(1024, delta_kb // 2)  # half delta as safety margin

    # TRIGGER: thresholds between idle and peak → PRESSURE/CRITICAL when KV used
    pressure_trigger_kb = rss_idle_kb + max(1, delta_kb // 4)
    critical_trigger_kb = rss_idle_kb + max(2, delta_kb * 3 // 4)

    # SAFE: thresholds above peak → always stay NORMAL
    pressure_safe_kb = rss_peak_kb + margin_kb
    critical_safe_kb = pressure_safe_kb + 1

    # LOW_WATER: for water_excess computation
    low_water_trigger_kb = rss_idle_kb   # at baseline → water_excess ≈ KV contribution
    low_water_safe_kb = pressure_safe_kb  # above peak → water_excess = 0

    calibration = {
        "schema_version": 1,
        "rss_idle_kb": rss_idle_kb,
        "rss_peak_kb": rss_peak_kb,
        "delta_kb": delta_kb,
        "margin_kb": margin_kb,
        "pressure_trigger_kb": pressure_trigger_kb,
        "critical_trigger_kb": critical_trigger_kb,
        "pressure_safe_kb": pressure_safe_kb,
        "critical_safe_kb": critical_safe_kb,
        "low_water_trigger_kb": low_water_trigger_kb,
        "low_water_safe_kb": low_water_safe_kb,
        "calibration_response_text": response_text,
    }
    write_json(calib_dir / "calibration.json", calibration)
    print(f"  derived: pressure_trigger={pressure_trigger_kb} KiB  "
          f"critical_trigger={critical_trigger_kb} KiB  "
          f"pressure_safe={pressure_safe_kb} KiB")
    return calibration


# ── case runner ───────────────────────────────────────────────────────────────

def run_case(name: str, binary: str, model: str, port: int,
             env_vars: dict[str, str], output_dir: pathlib.Path,
             deadline: Deadline, strace: bool = False,
             two_requests: bool = False) -> dict[str, Any]:
    """Run one variant.  When two_requests=True, send two completions and
    record both responses, verifying KV reuse after bounded release."""
    output_dir.mkdir(parents=False, exist_ok=False)
    write_json(output_dir / "environment.json", env_vars)

    proc = start_server(binary, port, model, env_vars, output_dir, strace=strace)
    pgid = proc.stage3b_1_pgid

    try:
        wait_health(port, deadline)
        if strace:
            time.sleep(TIMEOUTS_S["post_attach_wait"])

        # Startup capability gate
        check_capability(name, env_vars, output_dir)

        # ── request 1 ──
        raw1 = output_dir / "completion_1.sse"
        status1, text1 = stream_completion(
            port, raw1, deadline, deadline.remaining(TIMEOUTS_S["completion"]))
        print(f"  req1: HTTP {status1}  len={len(text1)}")

        rss_after_req1_kb = get_server_rss_kb(proc.pid)

        text2 = ""
        status2 = 0
        rss_after_req2_kb = 0
        if two_requests:
            # Wait for bounded release to fire via pressure sampler
            print(f"  waiting {TIMEOUTS_S['release_wait']}s for bounded release ...")
            time.sleep(TIMEOUTS_S["release_wait"])

            # ── request 2 (reuses released blocks) ──
            raw2 = output_dir / "completion_2.sse"
            status2, text2 = stream_completion(
                port, raw2, deadline, deadline.remaining(TIMEOUTS_S["completion"]))
            print(f"  req2: HTTP {status2}  len={len(text2)}")
            rss_after_req2_kb = get_server_rss_kb(proc.pid)

        result = {
            "requests_count": 2 if two_requests else 1,
            "http_status_1": status1,
            "response_text_1": text1,
            "rss_after_req1_kb": rss_after_req1_kb,
        }
        if two_requests:
            result["http_status_2"] = status2
            result["response_text_2"] = text2
            result["rss_after_req2_kb"] = rss_after_req2_kb

        write_json(output_dir / "result.json", result)
        return result

    except SystemExit:
        write_json(output_dir / "result.json", {
            "requests_count": 0, "response_text_1": "",
            "error": "case aborted", "case_status": "incomplete",
        })
        write_json(output_dir / "failure.json", {
            "case_failed": True, "failure_type": "runner_abort",
        })
        raise

    finally:
        kill_server(proc, pgid)
        record_cleanup(output_dir, proc, pgid)


# ── parser protocol ──────────────────────────────────────────────────────────

def run_parser_protocol(art_dir: pathlib.Path) -> int:
    parser_path = ROOT / "scripts" / "parse-kv-bounded-release-stage3b-1.py"
    parser_cmd = [sys.executable, str(parser_path), str(art_dir),
                  "--result-path", str(art_dir / "parser.json")]
    parser_run = subprocess.run(parser_cmd, text=True, capture_output=True, check=False)
    print(parser_run.stdout, end="")
    print(parser_run.stderr, end="", file=sys.stderr)
    if parser_run.returncode != 0:
        return parser_run.returncode

    verify_cmd = [sys.executable, str(parser_path), str(art_dir),
                  "--verify-result", str(art_dir / "parser.json")]
    verify_run = subprocess.run(verify_cmd, text=True, capture_output=True, check=False)
    print(verify_run.stdout, end="")
    print(verify_run.stderr, end="", file=sys.stderr)
    return verify_run.returncode


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 3B-1 v2: calibrated dynamic target with reuse verification")
    ap.add_argument("--binary", required=True, help="Path to llama-server binary")
    ap.add_argument("--model", required=True, help="Path to GGUF model file")
    ap.add_argument("--output-dir", help="Output artifact directory")
    args = ap.parse_args()

    binary = str(pathlib.Path(args.binary).resolve())
    model = str(pathlib.Path(args.model).resolve())
    for p in [binary, model]:
        if not pathlib.Path(p).exists():
            fail(f"file not found: {p}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    head = git(["rev-parse", "--short=10", "HEAD"])
    if not args.output_dir:
        art_dir = pathlib.Path(
            f"/root/oscomp/kv_logs/kv_bounded_release_stage3b_1_{ts}_{head}_"
            f"{uuid.uuid4().hex[:12]}")
    else:
        art_dir = pathlib.Path(args.output_dir)
    if art_dir.exists():
        fail(f"artifact directory already exists (refusing overwrite): {art_dir}")
    art_dir.mkdir(parents=True, exist_ok=False)

    # ── manifest stub (finalized after all cases) ──
    initial_status = git(["status", "--porcelain"])
    identity_info = {
        "protocol": "kv_bounded_release_stage3b_1",
        "protocol_version": 2,
        "head_sha": git(["rev-parse", "HEAD"]),
        "head_short": head,
        "worktree_dirty": bool(initial_status),
        "worktree_status": initial_status.splitlines(),
        "capture_mode": "archival_clean" if not initial_status else "diagnostic_dirty",
        "timestamp_utc": ts,
        "binary": identity(pathlib.Path(binary)),
        "model": identity(pathlib.Path(model)),
        "runner": identity(pathlib.Path(__file__)),
        "parser": identity(ROOT / "scripts" / "parse-kv-bounded-release-stage3b-1.py"),
        "prompt": PROMPT,
        "n_predict": N_PREDICT,
        "seed": SEED,
        "fixed_target_bytes": FIXED_TARGET_BYTES,
        "dynamic_hard_cap_bytes": DYNAMIC_HARD_CAP_BYTES,
        "max_scan_blocks": MAX_SCAN_BLOCKS,
    }
    identity_info["source_snapshot"] = capture_source_snapshot(art_dir)
    identity_info["diff"] = identity_info["source_snapshot"]["tracked_diff"]
    identity_info["diff_sha256"] = identity_info["diff"]["sha256"]
    write_json(art_dir / "manifest.json", identity_info)

    used_ports: set[int] = set()

    # ── Phase 0: RSS calibration ──
    calib = run_calibration(binary, model, art_dir)
    identity_info["calibration"] = calib

    # ── Build per-case env ──
    # OFF: no bounded release, safe thresholds (stay NORMAL)
    off_env: dict[str, str] = {
        "LLAMA_KV_PRESSURE_RSS_KB": str(calib["pressure_safe_kb"]),
        "LLAMA_KV_CRITICAL_RSS_KB": str(calib["critical_safe_kb"]),
        "LLAMA_KV_LOW_WATER_RSS_KB": str(calib["low_water_safe_kb"]),
    }

    # FIXED: bounded release (fixed target), trigger thresholds
    fixed_env: dict[str, str] = {
        "LLAMA_KV_PRESSURE_RSS_KB": str(calib["pressure_trigger_kb"]),
        "LLAMA_KV_CRITICAL_RSS_KB": str(calib["critical_trigger_kb"]),
        "LLAMA_KV_LOW_WATER_RSS_KB": str(calib["low_water_trigger_kb"]),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": str(FIXED_TARGET_BYTES),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": str(MAX_SCAN_BLOCKS),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000",
    }

    # DYNAMIC_NOOP: bounded release + dynamic target, SAFE thresholds → NORMAL
    dynamic_noop_env: dict[str, str] = {
        "LLAMA_KV_PRESSURE_RSS_KB": str(calib["pressure_safe_kb"]),
        "LLAMA_KV_CRITICAL_RSS_KB": str(calib["critical_safe_kb"]),
        "LLAMA_KV_LOW_WATER_RSS_KB": str(calib["low_water_safe_kb"]),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": str(DYNAMIC_HARD_CAP_BYTES),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": str(MAX_SCAN_BLOCKS),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000",
    }

    # DYNAMIC_RELEASE: bounded release + dynamic target, TRIGGER thresholds → PRESSURE/CRITICAL
    dynamic_release_env: dict[str, str] = {
        "LLAMA_KV_PRESSURE_RSS_KB": str(calib["pressure_trigger_kb"]),
        "LLAMA_KV_CRITICAL_RSS_KB": str(calib["critical_trigger_kb"]),
        "LLAMA_KV_LOW_WATER_RSS_KB": str(calib["low_water_trigger_kb"]),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": str(DYNAMIC_HARD_CAP_BYTES),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": str(MAX_SCAN_BLOCKS),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000",
    }

    cases: list[tuple[str, str, dict[str, str], bool, bool]] = [
        ("bounded_off", "OFF", off_env, True, False),
        ("bounded_fixed", "FIXED", fixed_env, True, True),
        ("bounded_dynamic_noop", "DYNAMIC_NOOP", dynamic_noop_env, True, False),
        ("bounded_dynamic_release", "DYNAMIC_RELEASE", dynamic_release_env, True, True),
    ]

    # ── Run cases ──
    results: dict[str, Any] = {}
    case_failures: list[str] = []
    for dir_name, label, extra_env, use_strace, two_req in cases:
        print(f"\n=== {label} ({dir_name}) {'[2-request]' if two_req else '[1-request]'} ===")
        port = free_port(used_ports)
        case_dir = art_dir / dir_name
        env = controlled_env(extra_env)

        deadline = Deadline(TIMEOUTS_S["case_total"])
        try:
            result = run_case(label, binary, model, port, env, case_dir, deadline,
                             strace=use_strace, two_requests=two_req)
            results[label] = result
            r1 = result.get("response_text_1", "")
            r2 = result.get("response_text_2", "")
            print(f"  result: req1={r1[:40]!r}..."
                  + (f"  req2={r2[:40]!r}..." if two_req else ""))
        except SystemExit as exc:
            print(f"  CASE FAILED: {exc}")
            case_failures.append(label)
            results[label] = {"response_text_1": "", "response_text_2": "",
                              "error": str(exc), "requests_count": 0}
        time.sleep(2)

    # ── Response identity check ──
    all_match = False
    if not case_failures:
        off_text = results["OFF"]["response_text_1"]
        all_match = len(off_text) > 0
        for label in ["FIXED", "DYNAMIC_NOOP", "DYNAMIC_RELEASE"]:
            r = results[label]
            all_match = all_match and r["response_text_1"] == off_text
            if r.get("requests_count", 1) >= 2:
                all_match = all_match and r.get("response_text_2", "") == off_text

    runner_status = "run_complete" if not case_failures else "run_incomplete"
    summary = {
        "runner_status": runner_status,
        "response_identity": bool(all_match),
        "off_text_len": len(results["OFF"].get("response_text_1", "")),
        "case_failures": case_failures,
        "_note": "runner records observable facts only — final verdict is parser's responsibility",
    }
    write_json(art_dir / "summary.json", summary)

    # ── Finalize manifest ──
    repo_after = assert_repo(identity_info["head_sha"])
    if hashlib.sha256(git_bytes(["diff", "--binary", "HEAD"])).hexdigest() != identity_info["diff_sha256"]:
        fail("tracked diff changed during the protocol")
    identity_info["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    identity_info["case_configs"] = {
        label: {
            "environment": json.loads(
                (art_dir / directory / "environment.json").read_text(encoding="utf-8")),
            "execution": json.loads(
                (art_dir / directory / "execution.json").read_text(encoding="utf-8")),
        }
        for directory, label, _, _, _ in cases
        if (art_dir / directory / "environment.json").is_file() and
           (art_dir / directory / "execution.json").is_file()
    }
    write_json(art_dir / "manifest.json", identity_info)

    # ── Parser verdict ──
    parser_code = run_parser_protocol(art_dir)
    if parser_code != 0:
        raise SystemExit(parser_code)

    print(f"\n=== Summary ===")
    print(f"  artifact: {art_dir}")
    print(f"  response identity: {'YES' if all_match else 'NO'}")
    print(f"\n  Run parser:")
    print(f"  python scripts/parse-kv-bounded-release-stage3b-1.py {art_dir}")


if __name__ == "__main__":
    main()
