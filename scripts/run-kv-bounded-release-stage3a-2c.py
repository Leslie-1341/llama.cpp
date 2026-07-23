#!/usr/bin/env python3
"""Stage 3A-2C: real-server bounded destructive release OFF/DRY/BOUNDED validation.

Three variants share the same binary, model, parameters, request, and seed:
  - OFF:  no release switches at all (zero bounded/destructive source evidence)
  - DRY:  LLAMA_KV_PRESSURE_DRY_RUN=1 (read-only would-release prediction)
  - BOUNDED: LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1 (real MADV_DONTNEED)

All three explicitly set LLAMA_KV_PAGED_RELEASE=0.  The 1/2/3 KiB RSS thresholds
force PRESSURE/CRITICAL state transitions (FORCED_LIFECYCLE_VALIDATION_ONLY).

Strace captures process-wide MADV_DONTNEED calls for all cases.  It is not a
source attribution mechanism: OFF/DRY calls are reported as background, while
BOUNDED uses its source counter plus mincore as the proof and reports any
strace surplus as background.  The release target_bytes must exceed RSS noise.

Protocol: ./run-kv-bounded-release-stage3a-2c.py --binary build/bin/llama-server
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

# Env keys to UNSET from subprocess environment (remove entirely, not set to "0").
# LLAMA_KV_LAZY_TAIL / LLAMA_KV_LAZY_CLEAR use atoi(val) != 0 semantics which
# correctly parses "0" as disabled, but to eliminate any future ambiguity
# about presence-based enablement, we strip them from the child env entirely.
UNSET_ENV: set[str] = {
    "LLAMA_KV_LAZY_TAIL",
    "LLAMA_KV_LAZY_CLEAR",
}

# The child is deliberately *not* based on os.environ.  Keep this explicit
# deny-list as an auditable record of the experiment knobs that would otherwise
# be inherited from a developer shell.  build_controlled_env() removes all
# LLAMA_KV_* keys and these non-KV experimental families before it installs the
# protocol whitelist below.
EXPERIMENT_ENV_PREFIXES = ("LLAMA_KV_", "LLAMA_", "GGML_", "GGUF_")

SHARED_ENV: dict[str, str] = {
    "LLAMA_KV_PAGED": "1",
    # Explicitly enable in-graph path — bounded release requires paged_ingraph_enabled
    # for row-index gather mode.  Without this (or if overridden by identity fast path),
    # paged_row_idx_enabled remains false and bounded_release_can_enable() fails.
    "LLAMA_KV_PAGED_INGRAPH": "1",
    "LLAMA_KV_PAGED_MINCORE": "1",  # enable KV resident-page tracking
    # Disable identity fast path — bounded release requires row-index gather
    # mode for ownership collection.  Without this, paged_row_idx_enabled
    # remains false and bounded_release_can_enable() returns false.
    "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
    "LLAMA_KV_PRESSURE_SAMPLER": "1",
    "LLAMA_KV_LOW_WATER_RSS_KB": "1",
    "LLAMA_KV_PRESSURE_RSS_KB": "2",
    "LLAMA_KV_CRITICAL_RSS_KB": "3",
    "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "250",
    "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
}

DRY_ONLY_ENV: dict[str, str] = {
    "LLAMA_KV_PRESSURE_DRY_RUN": "1",
    "LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES": "33554432",  # 32 MiB
    "LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS": "64",
    "LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS": "500",
}

BOUNDED_ONLY_ENV: dict[str, str] = {
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": "33554432",  # 32 MiB
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": "64",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "60000",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "60000",
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


def git_bytes(args_list: list[str]) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args_list])


def capture_source_snapshot(art_dir: pathlib.Path) -> dict[str, Any]:
    """Capture the inputs needed to reconstruct this dirty tree from HEAD."""
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
    """Send completion request via SSE, return (http_status, full_text)."""
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

    # Extract text from SSE stream
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


def env_dict(extra: dict[str, str] | None = None) -> dict[str, str]:
    result = dict(BASE_ENV)
    result.update(FORBIDDEN_ENV)
    result.update(SHARED_ENV)
    if extra:
        result.update(extra)
    # Explicitly unset lazy env vars from child process environment to
    # eliminate any risk of presence-based enablement.
    for key in UNSET_ENV:
        result.pop(key, None)
    return result


def build_controlled_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return the exact, isolated environment passed to Popen.

    Do not inherit host variables: clearing is represented both by the empty
    construction and by ``cleared_inherited`` in the per-case record.
    """
    result = env_dict(extra)
    if any(key.startswith(EXPERIMENT_ENV_PREFIXES) for key in result if key not in {
            *FORBIDDEN_ENV, *SHARED_ENV, *(extra or {})}):
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
    """Start llama-server, optionally under strace.

    Topology is locked to single-slot row-index gather:
      --parallel 1   → single sequence slot (not auto/MAX)
      --cache-ram 0  → disable prompt-cache so no cached-prompt
                       shortcut bypasses the KV paged path
      --no-warmup    → disable warmup so the single inference
                       exercises the full cold path
      --cache-type-k f32 --cache-type-v f32
                     → F32 K/V tensors required for row-index
                       gather mode and bounded release ownership
    """
    cmd = []
    if strace:
        cmd = ["strace", "-f", "-e", "trace=madvise",
               "-o", str(output_dir / "strace.log"),
               "--"]
    cmd += [binary, "--host", "127.0.0.1", "--port", str(port),
            "--model", model, "--ctx-size", "1024",
            "--n-gpu-layers", "0", "--threads", "4",
            "--batch-size", "128", "--ubatch-size", "128",
            "--parallel", "1",
            "--cache-ram", "0",
            "--cache-type-k", "f32", "--cache-type-v", "f32",
            "--no-warmup"]

    # Record the *actual* Popen command and environment, not an intended
    # overlay.  This must stay byte-for-byte identical to ``env=`` below.
    write_json(output_dir / "execution.json", {
        "argv": cmd,
        "strace": strace,
        "environment": env_vars,
        "cleared_inherited_prefixes": list(EXPERIMENT_ENV_PREFIXES),
    })

    log = (output_dir / "server.stdout").open("wb")
    err = (output_dir / "server.stderr").open("wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=err, env=env_vars,
                           preexec_fn=os.setsid)
    # Keep the process-group identity while the leader is alive.  Looking it
    # up during teardown loses the group as soon as the leader exits and can
    # silently miss orphaned strace/server descendants.
    proc.stage3a_2c_pgid = os.getpgid(proc.pid)  # type: ignore[attr-defined]
    return proc


def kill_server(proc: subprocess.Popen, pgid: int) -> None:
    """Send SIGTERM, wait, escalate to SIGKILL."""
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


def run_parser_protocol(art_dir: pathlib.Path) -> int:
    """Run the parser-owned verdict, then verify its saved closure record."""
    parser_path = ROOT / "scripts" / "parse-kv-bounded-release-stage3a-2c.py"
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


def run_case(name: str, binary: str, model: str, port: int, env_vars: dict[str, str],
             output_dir: pathlib.Path, deadline: Deadline,
             strace: bool = False) -> dict[str, Any]:
    """Run one variant: start server, send request, capture output, stop."""
    output_dir.mkdir(parents=False, exist_ok=False)
    write_json(output_dir / "environment.json", env_vars)

    proc = start_server(binary, port, model, env_vars, output_dir, strace=strace)
    pgid = proc.stage3a_2c_pgid  # type: ignore[attr-defined]

    try:
        wait_health(port, deadline)
        if strace:
            time.sleep(TIMEOUTS_S["post_attach_wait"])

        # --- Capability gate ---
        bounded_requested = (
            env_vars.get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE", "0") == "1"
        )
        if bounded_requested:
            stderr_path = output_dir / "server.stderr"
            stderr_text = stderr_path.read_text(errors="replace")
            cap_match = re.search(
                r"kv_pressure_bounded_release_capability\s+"
                r"can_enable=(\d+)\s+paged=(\d+)\s+ingraph=(\d+)\s+"
                r"layers_supported=(\d+)\s+row_idx=(\d+)\s+"
                r"swap_disabled=(\d+)\s+layout_supported=(\d+)",
                stderr_text)
            if not cap_match:
                fail(
                    f"{name}: capability marker not found in server stderr — "
                    f"bounded release is requested but the structural "
                    f"capability diagnostic never appeared")
            cap = {k: int(v) for k, v in zip(
                ("can_enable", "paged", "ingraph", "layers_supported",
                 "row_idx", "swap_disabled", "layout_supported"),
                cap_match.groups())}
            write_json(output_dir / "capability.json", cap)

            # Five non-deferred fields must be 1 at startup
            STARTUP_HARD = ("paged", "ingraph", "layers_supported",
                            "swap_disabled", "layout_supported")
            for field in STARTUP_HARD:
                if cap.get(field, -1) != 1:
                    details = " ".join(f"{k}={v}" for k, v in cap.items())
                    fail(
                        f"{name}: startup capability {field}={cap.get(field)} — "
                        f"must be 1; server topology incompatible. "
                        f"full: {details}")
            if cap["row_idx"] != 1:
                print(f"  {name}: startup row_idx=0 (deferred — graph input "
                      f"not yet created); will be re-diagnosed per-sample")

        raw_path = output_dir / "completion.sse"
        http_status, response_text = stream_completion(
            port, raw_path, deadline,
            deadline.remaining(TIMEOUTS_S["completion"]))

        # Capture RSS of server process (coarse)
        rss_before_kb = 0
        rss_after_kb = 0
        try:
            with open(f"/proc/{proc.pid}/statm", "r") as f:
                parts = f.read().split()
                if len(parts) >= 2:
                    rss_after_kb = int(parts[1]) * 4  # pages → KB
        except Exception:
            pass

        result = {
            "http_status": http_status,
            "response_text": response_text,
            "rss_before_kb": rss_before_kb,
            "rss_after_kb": rss_after_kb,
        }
        write_json(output_dir / "result.json", result)

        return result

    except SystemExit as exc:
        # Case-level failure: record the error in the artifact so the
        # parser sees an explicit incomplete marker and does NOT treat
        # the case as a valid silent skip.
        write_json(output_dir / "result.json", {
            "http_status": 0,
            "response_text": "",
            "error": f"case aborted: {exc}",
            "case_status": "incomplete",
        })
        write_json(output_dir / "failure.json", {
            "case_failed": True, "failure_type": "runner_abort",
        })
        # Re-raise so the caller (main) can decide whether to continue
        raise

    finally:
        kill_server(proc, pgid)

        # Wait for strace to flush
        time.sleep(0.5)

        # Check residual processes
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
            # The leader may already be gone while descendants still own the
            # original process group.  Clean that exact group, then report the
            # post-cleanup state rather than a guessed leader-derived PGID.
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
        # Never overwrite failure metadata from the exception path.
        phase_path = output_dir / "phases.json"
        phases: dict[str, Any] = {}
        if phase_path.exists():
            phases = json.loads(phase_path.read_text(encoding="utf-8"))
        phases["shutdown"] = {
            "pgid": pgid,
            "exit_code": proc.returncode,
            "pgid_check_complete": pgid_check_complete,
            "cleanup_kill_attempted": cleanup_kill_attempted,
            "residual_process": residual,
        }
        write_json(phase_path, phases)

        # Close log files
        for f in [proc.stdout, proc.stderr]:
            if f and not f.closed:
                f.close()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 3A-2C bounded release OFF/DRY/BOUNDED validation")
    ap.add_argument("--binary", required=True,
                    help="Path to llama-server binary")
    ap.add_argument("--model", required=True,
                    help="Path to GGUF model file")
    ap.add_argument("--output-dir",
                    help="Output artifact directory (default: auto-generated)")
    ap.add_argument("--target-bytes", type=int, default=33554432,
                    help="Release target in bytes (default: 33554432 = 32 MiB)")
    args = ap.parse_args()

    binary = str(pathlib.Path(args.binary).resolve())
    model = str(pathlib.Path(args.model).resolve())

    # Validate inputs
    for p in [binary, model]:
        if not pathlib.Path(p).exists():
            fail(f"file not found: {p}")

    # Create output artifact directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    head = git(["rev-parse", "--short=10", "HEAD"])
    if not args.output_dir:
        art_dir = pathlib.Path(
            f"/root/oscomp/kv_logs/kv_bounded_release_stage3a_2c_{ts}_{head}_{uuid.uuid4().hex[:12]}")
    else:
        art_dir = pathlib.Path(args.output_dir)
    if art_dir.exists():
        fail(f"artifact directory already exists (refusing overwrite): {art_dir}")
    art_dir.mkdir(parents=True, exist_ok=False)

    # Adjust target bytes in env
    target_str = str(args.target_bytes)
    dry_env = dict(DRY_ONLY_ENV)
    dry_env["LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES"] = target_str
    bounded_env = dict(BOUNDED_ONLY_ENV)
    bounded_env["LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES"] = target_str

    # Write identity info
    initial_status = git(["status", "--porcelain"])
    identity_info = {
        "protocol": "kv_bounded_release_stage3a_2c",
        "protocol_version": 4,
        "head_sha": git(["rev-parse", "HEAD"]),
        "head_short": head,
        "worktree_dirty": bool(initial_status),
        "worktree_status": initial_status.splitlines(),
        "capture_mode": "archival_clean" if not initial_status else "diagnostic_dirty",
        "timestamp_utc": ts,
        "binary": identity(pathlib.Path(binary)),
        "model": identity(pathlib.Path(model)),
        "runner": identity(pathlib.Path(__file__)),
        "parser": identity(ROOT / "scripts" / "parse-kv-bounded-release-stage3a-2c.py"),
        "threshold_purpose": THRESHOLD_PURPOSE,
        "prompt": PROMPT,
        "n_predict": N_PREDICT,
        "seed": SEED,
        "target_bytes": args.target_bytes,
    }
    identity_info["source_snapshot"] = capture_source_snapshot(art_dir)
    identity_info["diff"] = identity_info["source_snapshot"]["tracked_diff"]
    identity_info["diff_sha256"] = identity_info["diff"]["sha256"]
    write_json(art_dir / "manifest.json", identity_info)

    used_ports: set[int] = set()

    cases = [
        ("bounded_off", "OFF", dict(BASE_ENV), True),   # strace: background report
        ("dry_run_off", "DRY", dry_env, True),           # strace: background report
        ("bounded_on", "BOUNDED", bounded_env, True),    # source counter + mincore attribution
    ]

    results: dict[str, Any] = {}
    case_failures: list[str] = []
    for dir_name, label, extra_env, use_strace in cases:
        print(f"\n=== {label} ({dir_name}) ===")
        port = free_port(used_ports)
        case_dir = art_dir / dir_name
        env = build_controlled_env(extra_env)

        deadline = Deadline(TIMEOUTS_S["case_total"])
        try:
            result = run_case(label, binary, model, port, env, case_dir, deadline,
                             strace=use_strace)
            results[label] = result
            print(f"  HTTP {result['http_status']}  "
                  f"response: {result['response_text'][:60]!r}...")
        except SystemExit as exc:
            print(f"  CASE FAILED: {exc}")
            case_failures.append(label)
            results[label] = {
                "http_status": 0,
                "response_text": "",
                "error": str(exc),
            }
        # Cooldown between cases
        time.sleep(2)

    # Verify response identity
    all_match = False
    if case_failures:
        print(f"\n  {len(case_failures)} case(s) failed: {case_failures}")
    else:
        off_text = results["OFF"]["response_text"]
        dry_text = results["DRY"]["response_text"]
        bounded_text = results["BOUNDED"]["response_text"]
        all_match = (off_text == dry_text == bounded_text and len(off_text) > 0)

    # Runner records observable facts only — the final verdict is the
    # parser's responsibility.  Runner reports "run_complete" when all
    # cases finished without runner-level failure; "run_incomplete"
    # otherwise.  Response identity is reported as an observable, not
    # as a PASS/FAIL verdict.
    runner_status = "run_complete" if not case_failures else "run_incomplete"
    summary = {
        "runner_status": runner_status,
        "response_identity": bool(all_match),
        "off_text_len": len(results["OFF"].get("response_text", "")),
        "dry_text_len": len(results["DRY"].get("response_text", "")),
        "bounded_text_len": len(results["BOUNDED"].get("response_text", "")),
        "case_failures": case_failures,
        "_note": "runner records observable facts only — final verdict is parser's responsibility",
    }
    write_json(art_dir / "summary.json", summary)

    # Close the manifest only after every actual Popen invocation has been
    # recorded.  It is intentionally a manifest of facts, not a planned argv.
    repo_after = assert_repo(identity_info["head_sha"])
    if repo_after.get("dirty_files", []) != identity_info["worktree_status"]:
        fail("worktree changed during the protocol")
    if hashlib.sha256(git_bytes(["diff", "--binary", "HEAD"])).hexdigest() != identity_info["diff_sha256"]:
        fail("tracked diff changed during the protocol")
    raw_untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    current_untracked = sorted(
        p.decode("utf-8", errors="strict") for p in raw_untracked.split(b"\0") if p)
    captured_untracked = [item["path"] for item in identity_info["source_snapshot"]["untracked_files"]]
    if current_untracked != captured_untracked:
        fail("untracked file set changed during the protocol")
    for item in identity_info["source_snapshot"]["untracked_files"]:
        current = identity(ROOT / item["path"])
        if current["size"] != item["size"] or current["sha256"] != item["sha256"] or \
                ((ROOT / item["path"]).stat().st_mode & 0o777) != item["mode"]:
            fail(f"untracked file changed during the protocol: {item['path']}")
    identity_info["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    identity_info["case_configs"] = {
        label: {
            "environment": json.loads((art_dir / directory / "environment.json").read_text(encoding="utf-8")),
            "execution": json.loads((art_dir / directory / "execution.json").read_text(encoding="utf-8")),
        }
        for directory, label, _, _ in cases
        if (art_dir / directory / "environment.json").is_file() and
           (art_dir / directory / "execution.json").is_file()
    }
    write_json(art_dir / "manifest.json", identity_info)

    # The parser is the only verdict authority.  Preserve its complete command
    # and streams even when it rejects an incomplete runner artifact.
    # Parser rejection or closure verification failure is a runner failure.
    parser_code = run_parser_protocol(art_dir)
    if parser_code != 0:
        raise SystemExit(parser_code)

    print(f"\n=== Summary ===")
    print(f"  artifact: {art_dir}")
    print(f"  response identity: {'YES' if all_match else 'NO'}")
    print(f"  output dir: {art_dir}")
    print(f"\n  Run parser:")
    print(f"  python scripts/parse-kv-bounded-release-stage3a-2c.py {art_dir}")


if __name__ == "__main__":
    main()
