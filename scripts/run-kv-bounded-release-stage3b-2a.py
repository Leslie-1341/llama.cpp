#!/usr/bin/env python3
"""Stage 3B-2A: long-context ladder + continuous-request release-only boundary test.

Phase 0 — Tokenizer Calibration:
  Uses the server /tokenize endpoint to build prompts of exact token counts
  (1024, 2048, 4096, 8192).  Records the actual prompt text and verified token
  count for each ladder step — not just changing --ctx-size.

Phase 1 — RSS Calibration (per context length):
  For each ladder step, start server at that ctx-size, measure idle and
  post-completion RSS, derive per-level trigger/safe thresholds.

Phase 2 — Long Context Ladder:
  For each token count / ctx-size, run OFF and DYNAMIC_RELEASE cases:
    - OFF: no bounded release, safe thresholds → stays NORMAL
    - DYNAMIC_RELEASE: bounded release + dynamic target, trigger thresholds
  Verify response identity, no ownership/madvise errors, release metrics
  consistent with context length.

Phase 3 — Continuous Requests (20):
  Single server with DYNAMIC_RELEASE config.  Send 20 identical requests
  with release-wait between each.  Verify all responses match OFF baseline
  and no cumulative errors emerge.

Protocol: ./run-kv-bounded-release-stage3b-2a.py --binary build/bin/llama-server
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

SHARED_ENV: dict[str, str] = {
    "LLAMA_KV_PAGED": "1",
    "LLAMA_KV_PAGED_INGRAPH": "1",
    "LLAMA_KV_PAGED_MINCORE": "1",
    "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
    "LLAMA_KV_PRESSURE_SAMPLER": "1",
    "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "250",
    "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
}

CALIBRATED_KEYS = {
    "LLAMA_KV_PRESSURE_RSS_KB",
    "LLAMA_KV_CRITICAL_RSS_KB",
    "LLAMA_KV_LOW_WATER_RSS_KB",
}

BOUNDED_ONLY_KEYS = {
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET",
}

TOKEN_TARGETS = [1024, 2048, 4096, 8192]
DYNAMIC_HARD_CAP_BYTES = 1073741824  # 1 GiB
MAX_SCAN_BLOCKS = 256  # increased for long contexts

N_PREDICT = 32
SEED = 1

# Effective-context derivation (阻塞项:模型有效上下文越界).
# server 把 --ctx-size clamp 到模型 n_ctx_train 后,实际生效 n_ctx 可能
# 小于请求的 ctx_size.  prompt 的 token 数(含特殊 token) + N_PREDICT + 安全
# 余量之和必须 <= 有效 n_ctx,否则 completion 必然 HTTP 400 越界失败.
#
#   max_prompt_tokens = floor((effective_n_ctx - N_PREDICT - SPECIAL_OVERHEAD - SAFETY)
#                             / TOKEN_ALIGN) * TOKEN_ALIGN
#
# 实测 Llama-3 /tokenize(add_special=False) 的 prompt 进入 completion 后会
# 增加 1 个特殊 token (n_prompt_tokens = prompt_tokens + 1), 故 SPECIAL_OVERHEAD=1.
# SAFETY 留白用于 prompt tokenize 的尾部差异与解码边界,使 8192 模型收敛到
#   8192 - 32 - 1 - SAFETY_DEFAULT(=95) = 8064  (与项目验收预期一致).
# 阶梯收敛时再按 TOKEN_ALIGN 向下取整,避免发送任意奇数档 prompt.
SPECIAL_OVERHEAD = 1
SAFETY_DEFAULT = 95
TOKEN_ALIGN = 32

# Regex for the server stderr line that exposes the *effective* slot n_ctx
# (after model-n_ctx_train capping).  Example:
#   "0.05.204.324 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 8192"
EFFECTIVE_N_CTX_RE = re.compile(r"new slot,\s*n_ctx\s*=\s*(\d+)")
# Base prompt for single-request scenarios (used in Phase 1 calibration)
BASE_PROMPT = "In one short sentence, explain why deterministic tests are useful."

# Token calibration template — a long, descriptive paragraph that can be
# repeated and tokenized to reach exact target token counts.
TOKEN_CALIBRATION_SEED = (
    "The operating system kernel manages memory allocation across multiple "
    "concurrent processes by maintaining page tables, tracking resident set "
    "size, and implementing demand paging with copy-on-write semantics. "
    "When physical memory pressure increases beyond configured thresholds, "
    "the kernel invokes the out-of-memory killer or triggers reclaim mechanisms "
    "such as page frame reclamation, swap, and memory compaction. "
    "User-space applications can advise the kernel about memory usage patterns "
    "through system calls like madvise, fadvise, and mlock, which influence "
    "page cache eviction, readahead, and residency guarantees. "
    "Modern systems employ control groups to enforce hierarchical resource "
    "limits including memory.max, memory.high, and memory.low boundaries, "
    "with pressure stall information exposing per-cgroup contention levels. "
    "Key-value caches in large language model inference servers store attention "
    "key and value tensors for each token position across all transformer layers, "
    "creating a memory footprint proportional to context length times hidden "
    "dimension. Deterministic testing ensures that cache management policies "
    "produce reproducible results regardless of system load or timing. "
)

TIMEOUTS_S = {
    "startup": 5.0, "health": 180.0, "completion": 180.0,
    "tokenize": 30.0, "shutdown": 15.0, "case_total": 600.0,
    "calibration_startup": 180.0, "release_wait": 8.0,
    "post_attach_wait": 0.75, "loop_total": 1800.0,
}

# Completion timeout derivation (Phase 0已校准 / ladder 复用).
# 每档 completion 的 timeout 不再固定 180s,而是基于该档 RSS calibration
# 中真实 completion 的 wall-clock 耗时乘以显式余量倍数,再取一个下限。
#   timeout_s = max(measured_s * TIMEOUT_MARGIN, TIMEOUT_FLOOR_S)
# 任意一档未取得真实耗时(measured<=0)时该档使用 FALLBACK.
TIMEOUT_MARGIN = 2.5
TIMEOUT_FLOOR_S = 30.0
COMPLETION_TIMEOUT_FALLBACK_S = 180.0
CALIBRATION_TIMEOUT_BOOTSTRAP_S = 180.0
CALIBRATION_TIMEOUT_SAFETY_FACTOR = 1.5

_ACTIVE_ARTIFACT: pathlib.Path | None = None
_ACTIVE_MANIFEST: dict[str, Any] | None = None
_CURRENT_PHASE = "startup"
_CURRENT_TARGET: int | None = None
_FIRST_FAILURE: dict[str, Any] | None = None
_LAST_FAILURE_REASON: str | None = None


# ── helpers ───────────────────────────────────────────────────────────────────

def set_failure_context(phase: str, target: int | None = None) -> None:
    global _CURRENT_PHASE, _CURRENT_TARGET
    _CURRENT_PHASE = phase
    _CURRENT_TARGET = target


def mark_failure(reason: str, phase: str | None = None,
                 target: int | None = None) -> None:
    global _FIRST_FAILURE, _LAST_FAILURE_REASON
    _LAST_FAILURE_REASON = reason
    if _FIRST_FAILURE is None:
        _FIRST_FAILURE = {
            "runner_status": "run_incomplete",
            "failure_phase": phase if phase is not None else _CURRENT_PHASE,
            "failure_target": target if target is not None else _CURRENT_TARGET,
            "failure_reason": reason,
        }


def persist_runner_status(status: str, phase: str | None = None,
                          target: int | None = None,
                          reason: str | None = None) -> None:
    if _ACTIVE_ARTIFACT is None or _ACTIVE_MANIFEST is None:
        return
    failure = _FIRST_FAILURE or {
        "failure_phase": phase,
        "failure_target": target,
        "failure_reason": reason,
    }
    payload = {
        "runner_status": status,
        "failure_phase": failure.get("failure_phase") if status != "run_complete" else None,
        "failure_target": failure.get("failure_target") if status != "run_complete" else None,
        "failure_reason": failure.get("failure_reason") if status != "run_complete" else None,
    }
    _ACTIVE_MANIFEST.update(payload)
    write_json(_ACTIVE_ARTIFACT / "manifest.json", _ACTIVE_MANIFEST)
    summary_path = _ACTIVE_ARTIFACT / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            value = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                summary = value
        except (OSError, json.JSONDecodeError):
            pass
    summary.update(payload)
    write_json(summary_path, summary)


def fail(message: str) -> NoReturn:
    mark_failure(message)
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
    try:
        with open(f"/proc/{pid}/statm", "r") as f:
            parts = f.read().split()
            if len(parts) >= 2:
                return int(parts[1]) * 4
    except Exception:
        pass
    return 0


# ── derived timeout / context-limit (pure, unit-testable) ─────────────────────

def derive_completion_timeout(measured_s: float,
                              margin: float = TIMEOUT_MARGIN,
                              floor: float = TIMEOUT_FLOOR_S,
                              fallback: float = COMPLETION_TIMEOUT_FALLBACK_S) -> float:
    """Per-tier completion timeout from that tier's measured wall-clock.

    timeout_s = max(measured_s * margin, floor).  measured<=0 (no valid sample,
    e.g. calibration HTTP 400) → fall back to a fixed safe value.  Every path
    that reaches the ladder must have a positive measured value; relying on the
    fallback downstream is an explicit error condition, not a happy path.
    """
    if measured_s <= 0.0 or measured_s is None:
        return fallback
    return max(measured_s * margin, floor)


def derive_calibration_timeout(target_tokens: int,
                               previous_target_tokens: int | None = None,
                               previous_completion_wall_s: float | None = None,
                               bootstrap_s: float = CALIBRATION_TIMEOUT_BOOTSTRAP_S,
                               safety_factor: float = CALIBRATION_TIMEOUT_SAFETY_FACTOR,
                               ) -> dict[str, Any]:
    if previous_target_tokens is None or previous_completion_wall_s is None:
        return {
            "calibration_timeout_s": bootstrap_s,
            "calibration_timeout_mode": "bootstrap",
            "calibration_timeout_bootstrap_s": bootstrap_s,
            "calibration_timeout_previous_target": None,
            "calibration_timeout_previous_wall_s": None,
            "calibration_timeout_token_ratio": None,
            "calibration_timeout_safety_factor": safety_factor,
        }
    if previous_target_tokens <= 0 or previous_completion_wall_s <= 0:
        raise ValueError("previous successful calibration timing must be positive")
    token_ratio = target_tokens / previous_target_tokens
    derived = previous_completion_wall_s * token_ratio * safety_factor
    timeout_s = max(bootstrap_s, derived)
    return {
        "calibration_timeout_s": timeout_s,
        "calibration_timeout_mode": "previous_success_scaled",
        "calibration_timeout_bootstrap_s": bootstrap_s,
        "calibration_timeout_previous_target": previous_target_tokens,
        "calibration_timeout_previous_wall_s": previous_completion_wall_s,
        "calibration_timeout_token_ratio": token_ratio,
        "calibration_timeout_safety_factor": safety_factor,
        "calibration_timeout_scaled_s": derived,
    }


def derive_max_prompt_tokens(effective_n_ctx: int, n_predict: int = N_PREDICT,
                             special_overhead: int = SPECIAL_OVERHEAD,
                             safety: int = SAFETY_DEFAULT,
                             align: int = TOKEN_ALIGN) -> int:
    """Maximum prompt token count (as reported by /tokenize, add_special=False)
    that will fit inside effective_n_ctx together with n_predict.

    Constraints (invariant): prompt_tokens + special_overhead + n_predict
    <= effective_n_ctx, with `safety` headroom reserved to absorb tokenizer
    tail variance and decode boundaries; result aligned down to `align`.
    Returns 0 when the effective context cannot even hold generation alone.
    """
    budget = effective_n_ctx - n_predict - special_overhead - safety
    if budget <= 0:
        return 0
    if align > 1:
        budget = (budget // align) * align
    return max(budget, 0)


def clamp_token_targets(targets: list[int], max_prompt_tokens: int,
                        align: int = TOKEN_ALIGN) -> list[int]:
    """Converge ladder targets to legal values given the effective context.

    Each requested target: kept as-is when <= max_prompt_tokens; otherwise
    replaced by the largest aligned value <= max_prompt_tokens.  Targets that
    collapse below resolution (256 tokens minimum) are dropped.  Duplicates
    produced by clamping two tiers into the same legal value are collapsed,
    preserving order and uniqueness.  Returns the (possibly shorter,
    possibly identical) legal ladder.
    """
    if max_prompt_tokens <= 0:
        return []
    seen: set[int] = set()
    out: list[int] = []
    for t in sorted(targets):
        if t <= max_prompt_tokens:
            legal = t
        else:
            legal = (max_prompt_tokens // align) * align if align > 1 else max_prompt_tokens
        if legal < 256:
            continue
        if legal in seen:
            continue
        seen.add(legal)
        out.append(legal)
    return out


def probe_effective_n_ctx(stderr_text: str) -> int | None:
    """Extract the server's *effective* slot n_ctx from startup stderr.

    llama.cpp logs `slot load_model: ... new slot, n_ctx = N` after clamping
    --ctx-size to model n_ctx_train, so N is the actual context budget served
    by each slot.  Returns the highest such N if several slots are reported
    (they should be equal under --parallel 1), or None when the marker never
    appeared (server did not finish loading → caller must fail-closed).
    """
    matches = EFFECTIVE_N_CTX_RE.findall(stderr_text)
    if not matches:
        return None
    return max(int(m) for m in matches)


def capture_pid_identity(pid: int) -> dict[str, Any]:
    """Capture /proc/<pid> starttime and cmdline to prevent PID-reuse ambiguity."""
    starttime: int = 0
    cmdline: str = ""
    try:
        with open(f"/proc/{pid}/stat", "r") as f:
            fields = f.read().strip().split()
        # field 22 (0-indexed: 21) is starttime — monotonically unique per boot
        if len(fields) > 21:
            starttime = int(fields[21])
    except Exception:
        pass
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
        cmdline = raw.decode(errors="replace").replace("\0", " ").strip()
    except Exception:
        pass
    return {"pid": pid, "starttime": starttime, "cmdline": cmdline[:512]}


def resolve_server_pid(proc: subprocess.Popen, timeout: float = 5.0) -> int:
    """Return the real llama-server PID, resolving through strace if needed.

    When strace -f wraps the server, proc.pid is the strace process (negligible
    RSS).  This function finds the child process whose cmdline matches the
    server binary.  The result is cached on proc.stage3b_2a_server_pid.
    """
    if getattr(proc, "stage3b_2a_server_pid", 0) > 0:
        return proc.stage3b_2a_server_pid

    if not getattr(proc, "stage3b_2a_has_strace", False):
        proc.stage3b_2a_server_pid = proc.pid  # type: ignore[attr-defined]
        return proc.pid

    binary = getattr(proc, "stage3b_2a_binary", "llama-server")
    binary_name = os.path.basename(binary)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            children_path = f"/proc/{proc.pid}/task/{proc.pid}/children"
            with open(children_path, "r") as f:
                child_pids = f.read().strip().split()
            for cp in child_pids:
                if not cp:
                    continue
                cpid = int(cp)
                try:
                    with open(f"/proc/{cpid}/cmdline", "rb") as f:
                        cmdline = f.read()
                    exe = cmdline.split(b"\0")[0].decode(errors="replace")
                    if binary_name in exe or "llama-server" in exe:
                        proc.stage3b_2a_server_pid = cpid  # type: ignore[attr-defined]
                        return cpid
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        time.sleep(0.1)
    # Fallback: return strace PID (wrong, but preserves old behaviour)
    return proc.pid


def sample_server_rss(proc: subprocess.Popen) -> tuple[int, dict[str, Any]]:
    """Sample RSS of the real server process, returning (rss_kb, pid_identity)."""
    server_pid = resolve_server_pid(proc)
    rss_kb = get_server_rss_kb(server_pid)
    identity = capture_pid_identity(server_pid)
    return rss_kb, identity


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


def tokenize_content(port: int, content: str, deadline: Deadline) -> list[int]:
    """Tokenize text via server /tokenize, returning token IDs."""
    status, data = request(port, "POST", "/tokenize", deadline,
                           TIMEOUTS_S["tokenize"],
                           {"content": content, "add_special": False,
                            "parse_special": True})
    if status != 200:
        fail(f"/tokenize returned HTTP {status}")
    tokens = data.get("tokens", [])
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        fail(f"/tokenize returned unexpected token format: {type(tokens)}")
    return tokens


def stream_completion(port: int, raw_path: pathlib.Path,
                      prompt: str, deadline: Deadline,
                      phase_seconds: float) -> tuple[int, str]:
    body = json.dumps({
        "prompt": prompt, "n_predict": N_PREDICT, "stream": True,
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
                 output_dir: pathlib.Path, ctx_size: int,
                 strace: bool = False) -> subprocess.Popen:
    cmd = []
    if strace:
        cmd = ["strace", "-f", "-e", "trace=madvise",
               "-o", str(output_dir / "strace.log"), "--"]
    cmd += [binary, "--host", "127.0.0.1", "--port", str(port),
            "--model", model, "--ctx-size", str(ctx_size),
            "--n-gpu-layers", "0", "--threads", "4",
            "--batch-size", "128", "--ubatch-size", "128",
            "--parallel", "1",
            "--cache-ram", "0",
            "--cache-type-k", "f32", "--cache-type-v", "f32",
            "--no-warmup"]

    write_json(output_dir / "execution.json", {
        "argv": cmd, "strace": strace, "environment": env_vars,
        "cleared_inherited_prefixes": list(EXPERIMENT_ENV_PREFIXES),
        "ctx_size": ctx_size,
    })

    log = (output_dir / "server.stdout").open("wb")
    err = (output_dir / "server.stderr").open("wb")
    proc = subprocess.Popen(cmd, stdout=log, stderr=err, env=env_vars,
                           preexec_fn=os.setsid)
    proc.stage3b_2a_pgid = os.getpgid(proc.pid)  # type: ignore[attr-defined]
    proc.stage3b_2a_has_strace = strace  # type: ignore[attr-defined]
    proc.stage3b_2a_binary = binary  # type: ignore[attr-defined]
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


# ── Phase 0: Effective-context probe ─────────────────────────────────────────

def probe_server_effective_context(binary: str, model: str, art_dir: pathlib.Path,
                                   requested_targets: list[int],
                                   used_ports: set[int]) -> dict[str, Any]:
    """Discover effective n_ctx on an isolated server without a completion."""
    print("\n=== Phase 0: Effective Context Probe ===")
    probe_dir = art_dir / "context_probe"
    probe_dir.mkdir(parents=False, exist_ok=False)
    requested_ctx_size = max(requested_targets) + N_PREDICT + SPECIAL_OVERHEAD + SAFETY_DEFAULT
    probe_env = controlled_env({})
    probe_env.pop("LLAMA_KV_PRESSURE_SAMPLER", None)
    probe_env.pop("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", None)
    probe_env.pop("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS", None)
    for key in list(probe_env):
        if key.startswith("LLAMA_KV_PRESSURE_RSS") or key.startswith("LLAMA_KV_CRITICAL"):
            probe_env.pop(key, None)

    port = free_port(used_ports)
    proc = start_server(binary, port, model, probe_env, probe_dir,
                        ctx_size=requested_ctx_size)
    pgid = os.getpgid(proc.pid)
    try:
        wait_health(port, Deadline(TIMEOUTS_S["calibration_startup"]))
    finally:
        kill_server(proc, pgid)
        record_cleanup(probe_dir, proc, pgid)

    stderr_path = probe_dir / "server.stderr"
    effective_n_ctx = probe_effective_n_ctx(
        stderr_path.read_text(errors="replace") if stderr_path.is_file() else "")
    if effective_n_ctx is None or effective_n_ctx <= 0:
        fail("context probe did not expose a positive effective n_ctx — fail-closed")

    max_prompt_tokens = derive_max_prompt_tokens(effective_n_ctx, N_PREDICT)
    effective_targets = clamp_token_targets(requested_targets, max_prompt_tokens)
    if len(effective_targets) < 2:
        fail(f"effective n_ctx={effective_n_ctx} allows max_prompt_tokens="
             f"{max_prompt_tokens}; effective targets={effective_targets} has <2 tiers")

    result = {
        "requested_ctx_size": requested_ctx_size,
        "requested_targets": requested_targets,
        "effective_n_ctx": effective_n_ctx,
        "max_prompt_tokens": max_prompt_tokens,
        "effective_targets": effective_targets,
        "completion_requests": 0,
        "http_completion_statuses": [],
    }
    write_json(probe_dir / "probe.json", result)
    print(f"  requested_ctx={requested_ctx_size} effective_n_ctx={effective_n_ctx} "
          f"effective_targets={effective_targets}")
    return result


# ── Phase 1: Tokenizer Calibration ──────────────────────────────────────────

def calibrate_token_prompts(binary: str, model: str, art_dir: pathlib.Path,
                            targets: list[int]) -> dict[int, dict[str, Any]]:
    """Build prompts of exact token counts using the model's own tokenizer.

    Starts a temporary server at max needed ctx-size, then iteratively
    tokenizes and adjusts prompt text until each target token count is hit
    within a tolerance of ±2 tokens.
    """
    print("\n=== Phase 0: Tokenizer Calibration ===")
    calib_dir = art_dir / "token_calibration"
    calib_dir.mkdir(parents=False, exist_ok=False)

    max_ctx = max(targets) + 256  # headroom
    calib_env = controlled_env({})
    calib_env.pop("LLAMA_KV_PRESSURE_SAMPLER", None)
    calib_env.pop("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", None)
    calib_env.pop("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS", None)
    for k in list(calib_env):
        if k.startswith("LLAMA_KV_PRESSURE_RSS") or k.startswith("LLAMA_KV_CRITICAL"):
            calib_env.pop(k, None)

    port = free_port(set())
    proc = start_server(binary, port, model, calib_env, calib_dir, ctx_size=max_ctx)
    pgid = os.getpgid(proc.pid)

    results: dict[int, dict[str, Any]] = {}
    try:
        deadline = Deadline(TIMEOUTS_S["calibration_startup"] + 120)
        wait_health(port, deadline)

        # Measure how many tokens per repetition of the seed text
        seed_tokens = tokenize_content(port, TOKEN_CALIBRATION_SEED, deadline)
        seed_len = len(seed_tokens)
        print(f"  seed text: {seed_len} tokens per repetition")

        for target in targets:
            # Estimate repetitions needed: target / seed_len, round up
            reps = max(1, (target + seed_len - 1) // seed_len)
            text = " ".join([TOKEN_CALIBRATION_SEED] * reps)
            tokens = tokenize_content(port, text, deadline)
            current = len(tokens)

            # Formal tiers require an exact prompt token count.  Detokenizing the
            # first target token IDs gives the model tokenizer an exact fixed point.
            iteration = 0
            while current != target and iteration < 20:
                iteration += 1
                if current > target:
                    # Trim: remove tokens from the end
                    tokens = tokens[:target]
                    # Detokenize back to text for recording (approximate)
                    status, data = request(port, "POST", "/detokenize", deadline,
                                           TIMEOUTS_S["tokenize"],
                                           {"tokens": tokens})
                    if status == 200:
                        text = data.get("content", text)
                    else:
                        # Fallback: just use fewer repetitions
                        reps = max(1, reps - 1)
                        text = " ".join([TOKEN_CALIBRATION_SEED] * reps)
                else:
                    # Extend: add one more repetition
                    reps += 1
                    text = " ".join([TOKEN_CALIBRATION_SEED] * reps)
                tokens = tokenize_content(port, text, deadline)
                current = len(tokens)

            if current != target:
                fail(f"tokenizer calibration failed for target={target}: "
                     f"exact token count required, best={current} after {iteration} iterations")

            print(f"  target={target:5d}  actual={current:5d}  delta={current - target:+d}  "
                  f"repetitions={reps}  iterations={iteration}")

            results[target] = {
                "target_tokens": target,
                "actual_tokens": current,
                "delta": current - target,
                "prompt_text": text,
                "prompt_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "seed_tokens_per_rep": seed_len,
                "repetitions": reps,
                "iterations": iteration,
            }

            write_json(calib_dir / f"prompt_{target}t.json", results[target])

    finally:
        kill_server(proc, pgid)
        record_cleanup(calib_dir, proc, pgid)

    write_json(calib_dir / "calibration.json", {
        "schema_version": 1,
        "targets": targets,
        "results": {str(t): r for t, r in results.items()},
        "seed_text_sha256": hashlib.sha256(TOKEN_CALIBRATION_SEED.encode()).hexdigest(),
    })
    return results


# ── Phase 1: RSS Calibration per Context Length ──────────────────────────────

def calibrate_rss_per_ctx(binary: str, model: str, art_dir: pathlib.Path,
                          prompts: dict[int, dict[str, Any]],
                          used_ports: set[int]) -> dict[int, dict[str, Any]]:
    """Measure idle and peak RSS at each context length, derive thresholds."""
    print("\n=== Phase 1: RSS Calibration per Context Length ===")
    calib_dir = art_dir / "rss_calibration"
    calib_dir.mkdir(parents=False, exist_ok=False)

    results: dict[int, dict[str, Any]] = {}
    previous_success_target: int | None = None
    previous_success_wall_s: float | None = None

    for target in sorted(prompts):
        set_failure_context("rss_calibration", target)
        prompt_info = prompts[target]
        prompt_text = prompt_info["prompt_text"]
        ctx_size = target + 256  # headroom for generation

        level_dir = calib_dir / f"ctx_{target}"
        level_dir.mkdir(parents=False, exist_ok=False)

        calib_env = controlled_env({})
        calib_env.pop("LLAMA_KV_PRESSURE_SAMPLER", None)
        calib_env.pop("LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS", None)
        calib_env.pop("LLAMA_KV_PRESSURE_LOG_INTERVAL_MS", None)
        for k in list(calib_env):
            if k.startswith("LLAMA_KV_PRESSURE_RSS") or k.startswith("LLAMA_KV_CRITICAL"):
                calib_env.pop(k, None)

        port = free_port(used_ports)
        proc = start_server(binary, port, model, calib_env, level_dir, ctx_size=ctx_size)
        pgid = os.getpgid(proc.pid)

        timeout_derivation = derive_calibration_timeout(
            target, previous_success_target, previous_success_wall_s)
        calibration_timeout_s = float(timeout_derivation["calibration_timeout_s"])

        try:
            deadline = Deadline(
                TIMEOUTS_S["calibration_startup"] + calibration_timeout_s + 60)
            wait_health(port, deadline)

            rss_idle_kb = get_server_rss_kb(proc.pid)
            raw_path = level_dir / "completion.sse"
            comp_start_ts = time.monotonic()
            http_status, response_text = stream_completion(
                port, raw_path, prompt_text, deadline, calibration_timeout_s)
            completion_wall_s = time.monotonic() - comp_start_ts
            rss_peak_kb = get_server_rss_kb(proc.pid)
            # Calibration completion is the per-tier timing sample.  A non-200
            # here (e.g. effective n_ctx < prompt+n_predict) means this tier
            # is not usable: record it explicitly rather than silently
            # producing an empty response that downstream treats as success.
            if http_status != 200 or not response_text:
                level = {
                    "target_tokens": target,
                    "source_target": target,
                    "ctx_size": ctx_size,
                    **timeout_derivation,
                    "http_status": http_status,
                    "completion_wall_s": completion_wall_s,
                    "response_text_len": 0,
                    "calibration_failed": True,
                    "error": "calibration completion non-200 or empty response",
                }
                write_json(level_dir / "rss_calibration.json", level)
                results[target] = level
                write_json(calib_dir / "rss_calibration.json", {
                    "schema_version": 1,
                    "levels": {str(t): r for t, r in results.items()},
                })
                fail(f"RSS calibration target={target} returned HTTP {http_status} "
                     f"or empty response — formal effective tier is unusable")
            completion_timeout_s = derive_completion_timeout(completion_wall_s)

            delta_kb = max(1024, rss_peak_kb - rss_idle_kb)
            margin_kb = max(1024, delta_kb // 2)

            pressure_trigger_kb = rss_idle_kb + max(1, delta_kb // 4)
            critical_trigger_kb = rss_idle_kb + max(2, delta_kb * 3 // 4)
            pressure_safe_kb = rss_peak_kb + margin_kb
            critical_safe_kb = pressure_safe_kb + 1
            low_water_trigger_kb = rss_idle_kb
            low_water_safe_kb = pressure_safe_kb

            print(f"  ctx={target:5d}  idle={rss_idle_kb} KiB  peak={rss_peak_kb} KiB  "
                  f"delta={delta_kb} KiB  trigger={pressure_trigger_kb} KiB  "
                  f"safe={pressure_safe_kb} KiB  response_len={len(response_text)}  "
                  f"completion={completion_wall_s:.1f}s  "
                  f"calibration_timeout={calibration_timeout_s:.1f}s  "
                  f"ladder_timeout={completion_timeout_s:.1f}s")

            level = {
                "target_tokens": target,
                "source_target": target,
                "ctx_size": ctx_size,
                **timeout_derivation,
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
                "http_status": http_status,
                "response_text_sha256": hashlib.sha256(response_text.encode()).hexdigest(),
                "response_text_len": len(response_text),
                # Real completion wall-clock measured this tier, used to derive
                # the per-tier ladder/continuous completion timeout.
                "completion_wall_s": completion_wall_s,
                "completion_timeout_s": completion_timeout_s,
                "completion_timeout_margin": TIMEOUT_MARGIN,
                "completion_timeout_floor": TIMEOUT_FLOOR_S,
            }
            write_json(level_dir / "rss_calibration.json", level)
            results[target] = level
            previous_success_target = target
            previous_success_wall_s = completion_wall_s

        finally:
            kill_server(proc, pgid)
            record_cleanup(level_dir, proc, pgid)

    write_json(calib_dir / "rss_calibration.json", {
        "schema_version": 1,
        "levels": {str(t): r for t, r in results.items()},
    })
    return results


# ── Phase 2: Long Context Ladder ─────────────────────────────────────────────

def run_ladder_case(name: str, binary: str, model: str, port: int,
                    env_vars: dict[str, str], output_dir: pathlib.Path,
                    deadline: Deadline, ctx_size: int,
                    calibrated_prompt: str, strace: bool = False,
                    completion_timeout_s: float | None = None,
                    source_target: int | None = None,
                    effective_n_ctx: int | None = None) -> dict[str, Any]:
    """Run a single ladder case (OFF or DYNAMIC_RELEASE for a given ctx_size).

    completion_timeout_s overrides the fixed TIMEOUTS_S["completion"] — each
    tier uses the timeout derived from its own measured completion wall-clock
    in RSS calibration.  source_target and effective_n_ctx are recorded into
    the result so the parser can verify the ladder converged to legal values.
    """
    output_dir.mkdir(parents=False, exist_ok=False)
    write_json(output_dir / "environment.json", env_vars)

    # Probe server-effective n_ctx from this case's own startup stderr so the
    # parser can confirm each tier ran inside the effective context.
    proc = start_server(binary, port, model, env_vars, output_dir,
                       ctx_size=ctx_size, strace=strace)
    pgid = os.getpgid(proc.pid)
    stderr_windows: list[dict[str, Any]] = []
    case_effective_n_ctx: int | None = effective_n_ctx

    try:
        wait_health(port, deadline)
        if strace:
            time.sleep(TIMEOUTS_S["post_attach_wait"])
        check_capability(name, env_vars, output_dir)

        # Record stderr byte offset before request
        stderr_path = output_dir / "server.stderr"
        req_start_byte = stderr_path.stat().st_size if stderr_path.exists() else 0
        req_start_ts = time.monotonic()

        # Single request — the ladder case sends the calibrated prompt
        comp_timeout = completion_timeout_s if completion_timeout_s else TIMEOUTS_S["completion"]
        raw1 = output_dir / "completion_1.sse"
        status1, text1 = stream_completion(
            port, raw1, calibrated_prompt, deadline,
            min(comp_timeout, deadline.remaining()))
        print(f"  {name}: HTTP {status1}  response_len={len(text1)}  "
              f"comp_timeout={comp_timeout:.1f}s")
        rss_after_kb, rss_identity = sample_server_rss(proc)

        # If bounded release is active, wait for release to fire
        if env_vars.get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE") == "1":
            print(f"  {name}: waiting {TIMEOUTS_S['release_wait']}s for release ...")
            time.sleep(TIMEOUTS_S["release_wait"])
            rss_after_release_kb, _ = sample_server_rss(proc)
        else:
            rss_after_release_kb = rss_after_kb

        # Record stderr byte offset after request + release wait
        req_end_byte = stderr_path.stat().st_size if stderr_path.exists() else req_start_byte
        req_end_ts = time.monotonic()
        stderr_windows.append({
            "round": 1,
            "start_byte": req_start_byte,
            "end_byte": req_end_byte,
            "start_timestamp_s": req_start_ts,
            "end_timestamp_s": req_end_ts,
            "duration_s": req_end_ts - req_start_ts,
        })

        # Read this case's actual effective n_ctx (post-capping) from its
        # own stderr; fall back to the inherited probe if the marker is absent.
        if stderr_path.is_file():
            case_probe = probe_effective_n_ctx(stderr_path.read_text(errors="replace"))
            if case_probe:
                case_effective_n_ctx = case_probe

        result = {
            "http_status": status1,
            "response_text": text1,
            "response_text_sha256": hashlib.sha256(text1.encode()).hexdigest(),
            "response_text_len": len(text1),
            "rss_after_request_kb": rss_after_kb,
            "rss_after_release_kb": rss_after_release_kb,
            "rss_pid_identity": rss_identity,
            "completion_timeout_s": comp_timeout,
            "source_target": source_target,
            "effective_n_ctx": case_effective_n_ctx,
            "legal_tier": source_target if source_target is not None else ctx_size,
        }
        write_json(output_dir / "result.json", result)
        return result

    except SystemExit:
        write_json(output_dir / "result.json", {
            "response_text": "", "response_text_len": 0,
            "error": "case aborted", "case_status": "incomplete",
        })
        raise

    finally:
        kill_server(proc, pgid)
        record_cleanup(output_dir, proc, pgid)
        # Write stderr windows after cleanup (stderr is closed and flushed)
        if stderr_windows:
            write_json(output_dir / "stderr_windows.json",
                      {"windows": stderr_windows, "request_count": len(stderr_windows)})


def build_ladder_envs(rss_calib: dict[int, dict[str, Any]],
                      ) -> dict[int, dict[str, dict[str, str]]]:
    """Build per-context-length OFF and DYNAMIC_RELEASE env configs."""
    envs: dict[int, dict[str, dict[str, str]]] = {}
    for target, cal in sorted(rss_calib.items()):
        off_env: dict[str, str] = {
            "LLAMA_KV_PRESSURE_RSS_KB": str(cal["pressure_safe_kb"]),
            "LLAMA_KV_CRITICAL_RSS_KB": str(cal["critical_safe_kb"]),
            "LLAMA_KV_LOW_WATER_RSS_KB": str(cal["low_water_safe_kb"]),
        }

        dyn_env: dict[str, str] = {
            "LLAMA_KV_PRESSURE_RSS_KB": str(cal["pressure_trigger_kb"]),
            "LLAMA_KV_CRITICAL_RSS_KB": str(cal["critical_trigger_kb"]),
            "LLAMA_KV_LOW_WATER_RSS_KB": str(cal["low_water_trigger_kb"]),
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": str(DYNAMIC_HARD_CAP_BYTES),
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": str(MAX_SCAN_BLOCKS),
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "2000",
            "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "10000",
        }

        envs[target] = {"OFF": off_env, "DYNAMIC_RELEASE": dyn_env}
    return envs


# ── Phase 3: Continuous Requests (20) ────────────────────────────────────────

def run_continuous_requests(name: str, binary: str, model: str, port: int,
                            env_vars: dict[str, str], output_dir: pathlib.Path,
                            deadline: Deadline, ctx_size: int,
                            calibrated_prompt: str, num_requests: int = 20,
                            strace: bool = False,
                            completion_timeout_s: float | None = None,
                            effective_n_ctx: int | None = None) -> dict[str, Any]:
    """Send num_requests completions on the same server, recording all metrics.

    Each request-cycle: send completion → wait for release → measure RSS.
    Verifies response identity across all rounds and detects cumulative errors.
    Records stderr byte-offset windows per request for parser validation.
    """
    output_dir.mkdir(parents=False, exist_ok=False)
    write_json(output_dir / "environment.json", env_vars)
    rounds: list[dict[str, Any]] = []
    stderr_windows: list[dict[str, Any]] = []

    proc = start_server(binary, port, model, env_vars, output_dir,
                       ctx_size=ctx_size, strace=strace)
    pgid = os.getpgid(proc.pid)
    all_identical = True
    baseline_text = ""
    cumulative_errors = 0

    try:
        wait_health(port, deadline)
        if strace:
            time.sleep(TIMEOUTS_S["post_attach_wait"])
        check_capability(name, env_vars, output_dir)

        stderr_path = output_dir / "server.stderr"

        for i in range(num_requests):
            # Record stderr byte offset before request
            req_start_byte = stderr_path.stat().st_size if stderr_path.exists() else 0
            req_start_ts = time.monotonic()

            raw_path = output_dir / f"completion_{i + 1:02d}.sse"
            comp_timeout = completion_timeout_s if completion_timeout_s else TIMEOUTS_S["completion"]
            status, text = stream_completion(
                port, raw_path, calibrated_prompt, deadline,
                min(comp_timeout, deadline.remaining()))
            rss_after_req_kb, rss_identity = sample_server_rss(proc)

            if status != 200:
                cumulative_errors += 1
                print(f"  round {i + 1:2d}/{num_requests}: HTTP {status} ERROR  "
                      f"len={len(text)}")
            else:
                print(f"  round {i + 1:2d}/{num_requests}: HTTP {status}  "
                      f"len={len(text)}  rss={rss_after_req_kb} KiB")

            if i == 0:
                baseline_text = text
            elif text != baseline_text:
                all_identical = False
                cumulative_errors += 1

            # Wait for release to fire between requests
            if i < num_requests - 1:
                time.sleep(TIMEOUTS_S["release_wait"])
                rss_after_release_kb, _ = sample_server_rss(proc)
            else:
                rss_after_release_kb = rss_after_req_kb

            # Record stderr byte offset after request + release wait
            req_end_byte = stderr_path.stat().st_size if stderr_path.exists() else req_start_byte
            req_end_ts = time.monotonic()
            stderr_windows.append({
                "round": i + 1,
                "start_byte": req_start_byte,
                "end_byte": req_end_byte,
                "start_timestamp_s": req_start_ts,
                "end_timestamp_s": req_end_ts,
                "duration_s": req_end_ts - req_start_ts,
            })

            rounds.append({
                "round": i + 1,
                "http_status": status,
                "response_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "response_text_len": len(text),
                "matches_baseline": text == baseline_text,
                "rss_after_request_kb": rss_after_req_kb,
                "rss_after_release_kb": rss_after_release_kb,
                "rss_pid_identity": rss_identity,
            })

        result = {
            "num_requests": num_requests,
            "completed_rounds": len(rounds),
            "all_responses_identical": all_identical,
            "cumulative_error_count": cumulative_errors,
            "baseline_text_sha256": hashlib.sha256(baseline_text.encode()).hexdigest()
            if baseline_text else "",
            "baseline_text_len": len(baseline_text),
            "rounds": rounds,
            "completion_timeout_s": (completion_timeout_s
                                      if completion_timeout_s else TIMEOUTS_S["completion"]),
            "effective_n_ctx": effective_n_ctx,
            "ctx_size": ctx_size,
        }
        write_json(output_dir / "result.json", result)
        return result

    except SystemExit:
        write_json(output_dir / "result.json", {
            "num_requests": num_requests,
            "completed_rounds": len(rounds),
            "error": "case aborted", "case_status": "incomplete",
        })
        raise

    finally:
        kill_server(proc, pgid)
        record_cleanup(output_dir, proc, pgid)
        if stderr_windows:
            write_json(output_dir / "stderr_windows.json",
                      {"windows": stderr_windows, "request_count": len(stderr_windows)})


# ── parser protocol ──────────────────────────────────────────────────────────

def run_parser_protocol(art_dir: pathlib.Path) -> int:
    parser_path = ROOT / "scripts" / "parse-kv-bounded-release-stage3b-2a.py"
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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global _ACTIVE_ARTIFACT, _ACTIVE_MANIFEST, _FIRST_FAILURE, _LAST_FAILURE_REASON
    _FIRST_FAILURE = None
    _LAST_FAILURE_REASON = None
    ap = argparse.ArgumentParser(
        description="Stage 3B-2A: long-context ladder + continuous-request release-only boundary test")
    ap.add_argument("--binary", required=True, help="Path to llama-server binary")
    ap.add_argument("--model", required=True, help="Path to GGUF model file")
    ap.add_argument("--output-dir", help="Output artifact directory")
    ap.add_argument("--token-targets", type=int, nargs="+",
                    default=TOKEN_TARGETS,
                    help="Target token counts for ladder (default: 1024 2048 4096 8192)")
    ap.add_argument("--num-continuous", type=int, default=20,
                    help="Number of continuous requests (default: 20)")
    args = ap.parse_args()

    binary = str(pathlib.Path(args.binary).resolve())
    model = str(pathlib.Path(args.model).resolve())
    for p in [binary, model]:
        if not pathlib.Path(p).exists():
            fail(f"file not found: {p}")

    targets = args.token_targets
    if len(targets) < 2:
        fail("at least 2 token targets required for ladder")
    if len(targets) != len(set(targets)):
        fail("token targets must be unique")
    for t in targets:
        if t < 256:
            fail(f"token target {t} below minimum (256)")

    num_continuous = args.num_continuous
    if num_continuous < 5:
        fail("num_continuous must be >= 5")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    head = git(["rev-parse", "--short=10", "HEAD"])
    if not args.output_dir:
        art_dir = pathlib.Path(
            f"/root/oscomp/kv_logs/kv_bounded_release_stage3b_2a_{ts}_{head}_"
            f"{uuid.uuid4().hex[:12]}")
    else:
        art_dir = pathlib.Path(args.output_dir)
    if art_dir.exists():
        fail(f"artifact directory already exists (refusing overwrite): {art_dir}")
    art_dir.mkdir(parents=True, exist_ok=False)

    # ── manifest stub ──
    initial_status = git(["status", "--porcelain"])
    identity_info = {
        "protocol": "kv_bounded_release_stage3b_2a",
        "protocol_version": 1,
        "head_sha": git(["rev-parse", "HEAD"]),
        "head_short": head,
        "worktree_dirty": bool(initial_status),
        "worktree_status": initial_status.splitlines(),
        "capture_mode": "archival_clean" if not initial_status else "diagnostic_dirty",
        "timestamp_utc": ts,
        "binary": identity(pathlib.Path(binary)),
        "model": identity(pathlib.Path(model)),
        "runner": identity(pathlib.Path(__file__)),
        "parser": identity(ROOT / "scripts" / "parse-kv-bounded-release-stage3b-2a.py"),
        "token_targets": targets,
        "num_continuous": num_continuous,
        "n_predict": N_PREDICT,
        "seed": SEED,
        "dynamic_hard_cap_bytes": DYNAMIC_HARD_CAP_BYTES,
        "max_scan_blocks": MAX_SCAN_BLOCKS,
    }
    identity_info["source_snapshot"] = capture_source_snapshot(art_dir)
    identity_info["diff"] = identity_info["source_snapshot"]["tracked_diff"]
    identity_info["diff_sha256"] = identity_info["diff"]["sha256"]
    identity_info.update({
        "runner_status": "run_in_progress",
        "failure_phase": None,
        "failure_target": None,
        "failure_reason": None,
    })
    _ACTIVE_ARTIFACT = art_dir
    _ACTIVE_MANIFEST = identity_info
    write_json(art_dir / "manifest.json", identity_info)
    persist_runner_status("run_in_progress")

    used_ports: set[int] = set()

    # Probe first.  The probe server reaches health and exits without sending a
    # completion, so an overflowing requested high tier can never enter formal
    # calibration or become calibration_failed evidence.
    set_failure_context("context_probe")
    context_probe = probe_server_effective_context(
        binary, model, art_dir, targets, used_ports)
    effective_n_ctx = context_probe["effective_n_ctx"]
    max_prompt_tokens = context_probe["max_prompt_tokens"]
    effective_targets = context_probe["effective_targets"]

    identity_info["effective_context"] = {
        "requested_targets": targets,
        "effective_targets": effective_targets,
        "effective_n_ctx": effective_n_ctx,
        "n_predict": N_PREDICT,
        "special_overhead": SPECIAL_OVERHEAD,
        "safety": SAFETY_DEFAULT,
        "token_align": TOKEN_ALIGN,
        "max_prompt_tokens": max_prompt_tokens,
        "probe": context_probe,
        "clamped_tiers": [
            {"requested": t,
             "effective": (t if t <= max_prompt_tokens else
                           (max_prompt_tokens // TOKEN_ALIGN) * TOKEN_ALIGN)}
            for t in targets
        ],
    }
    write_json(art_dir / "manifest.json", identity_info)

    # Formal tokenizer and RSS calibration are keyed only by effective targets.
    # Every effective tier is independently tokenized and completed.
    set_failure_context("token_calibration")
    token_prompts = calibrate_token_prompts(
        binary, model, art_dir, effective_targets)
    identity_info["token_calibration"] = {
        "schema_version": 1,
        "targets": effective_targets,
        "results": {str(t): r for t, r in token_prompts.items()},
    }

    set_failure_context("rss_calibration")
    rss_calib = calibrate_rss_per_ctx(
        binary, model, art_dir, token_prompts, used_ports)
    identity_info["rss_calibration"] = {
        "schema_version": 1,
        "targets": effective_targets,
        "levels": {str(t): c for t, c in rss_calib.items()},
    }
    write_json(art_dir / "manifest.json", identity_info)

    ladder_dir = art_dir / "ladder"
    ladder_dir.mkdir(parents=False, exist_ok=False)
    ladder_envs = build_ladder_envs(rss_calib)

    ladder_results: dict[str, dict[str, Any]] = {}
    ladder_failures: list[str] = []
    for tier in effective_targets:
        cal = rss_calib[tier]
        prompt_text = token_prompts[tier]["prompt_text"]
        ctx_size = cal["ctx_size"]
        measured_s = cal["completion_wall_s"]
        comp_timeout_s = cal["completion_timeout_s"]
        envs = ladder_envs[tier]

        for case_label in ["OFF", "DYNAMIC_RELEASE"]:
            set_failure_context(f"ladder_{case_label.lower()}", tier)
            full_label = f"LADDER_{case_label}_t{tier}"
            case_dir = ladder_dir / f"t{tier}_{case_label.lower()}"
            extra_env = envs[case_label]
            env = controlled_env(extra_env)
            port = free_port(used_ports)
            deadline = Deadline(TIMEOUTS_S["case_total"])

            print(f"\n=== {full_label} ctx={ctx_size} comp_timeout={comp_timeout_s:.1f}s ===")
            try:
                result = run_ladder_case(
                    full_label, binary, model, port, env, case_dir, deadline,
                    ctx_size=ctx_size, calibrated_prompt=prompt_text,
                    strace=True,
                    completion_timeout_s=comp_timeout_s,
                    source_target=tier, effective_n_ctx=effective_n_ctx)
                # HTTP non-200 is an evidence-fake-pass risk: the ladder tier
                # must not silently appear to complete.  Treat any non-200 /
                # empty response as a hard failure rather than a verdict input.
                if not isinstance(result, dict) or result.get("http_status") != 200 \
                        or result.get("response_text_len", 0) <= 0:
                    raise RuntimeError(
                        f"{full_label}: completion HTTP {result.get('http_status') if isinstance(result, dict) else '?'} "
                        f"len={result.get('response_text_len') if isinstance(result, dict) else 0} — "
                        f"effective context exhausted; recorded as ladder failure")
                ladder_results[full_label] = result
            except SystemExit as exc:
                print(f"  LADDER CASE FAILED: {exc}")
                mark_failure(str(exc), f"ladder_{case_label.lower()}", tier)
                ladder_failures.append(full_label)
                ladder_results[full_label] = {
                    "response_text": "", "response_text_len": 0,
                    "error": str(exc),
                }
            except RuntimeError as exc:
                print(f"  LADDER CASE FAILED: {exc}")
                mark_failure(str(exc), f"ladder_{case_label.lower()}", tier)
                ladder_failures.append(full_label)
                ladder_results[full_label] = {
                    "response_text": "", "response_text_len": 0,
                    "http_status": result.get("http_status") if isinstance(result, dict) else 0,
                    "error": str(exc),
                }
            time.sleep(2)

    # ── Phase 3: Continuous Requests ──
    # loop_dir is created exclusively by run_continuous_requests (exist_ok=False),
    # preserving fail-closed behavior on pre-existing artifacts; do not mkdir here.
    loop_dir = art_dir / "continuous_requests"

    contiguous_target = min(effective_targets)
    min_cal = rss_calib[contiguous_target]
    loop_prompt = token_prompts[contiguous_target]["prompt_text"]
    loop_ctx = min_cal["ctx_size"]
    loop_completion_timeout = min_cal["completion_timeout_s"]

    loop_env = controlled_env({
        "LLAMA_KV_PRESSURE_RSS_KB": str(min_cal["pressure_trigger_kb"]),
        "LLAMA_KV_CRITICAL_RSS_KB": str(min_cal["critical_trigger_kb"]),
        "LLAMA_KV_LOW_WATER_RSS_KB": str(min_cal["low_water_trigger_kb"]),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET": "1",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES": str(DYNAMIC_HARD_CAP_BYTES),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS": str(MAX_SCAN_BLOCKS),
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS": "2000",
        "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS": "10000",
    })

    loop_port = free_port(used_ports)
    print(f"\n=== CONTINUOUS_REQUESTS x{num_continuous} ctx={loop_ctx} "
          f"comp_timeout={loop_completion_timeout:.1f}s ===")
    loop_deadline = Deadline(TIMEOUTS_S["loop_total"])
    continuous_result = None
    set_failure_context("continuous_requests", contiguous_target)
    try:
        continuous_result = run_continuous_requests(
            "CONTINUOUS_REQUESTS", binary, model, loop_port, loop_env, loop_dir,
            loop_deadline, ctx_size=loop_ctx, calibrated_prompt=loop_prompt,
            num_requests=num_continuous, strace=True,
            completion_timeout_s=loop_completion_timeout,
            effective_n_ctx=effective_n_ctx)
    except (SystemExit, RuntimeError, TimeoutError, OSError) as exc:
        print(f"  CONTINUOUS CASE FAILED: {exc}")
        mark_failure(str(exc), "continuous_requests", contiguous_target)
        ladder_failures.append("CONTINUOUS_REQUESTS")
        continuous_result = {"error": str(exc), "num_requests": num_continuous,
                            "completed_rounds": 0}

    # ── Summary ──
    runner_status = "run_complete" if not ladder_failures else "run_incomplete"
    summary = {
        "runner_status": runner_status,
        "ladder_failures": ladder_failures,
        "ladder_off_responses": {
            f"t{t}": {
                "sha256": ladder_results.get(f"LADDER_OFF_t{t}", {}).get(
                    "response_text", ""),
            }
            for t in effective_targets
        },
        "continuous_all_identical": continuous_result.get("all_responses_identical", False)
        if continuous_result else False,
        "_note": "runner records observable facts only — final verdict is parser's responsibility",
    }
    write_json(art_dir / "summary.json", summary)
    persist_runner_status(runner_status)

    # ── Finalize manifest ──
    repo_after = assert_repo(identity_info["head_sha"])
    if hashlib.sha256(git_bytes(["diff", "--binary", "HEAD"])).hexdigest() != \
            identity_info["diff_sha256"]:
        fail("tracked diff changed during the protocol")
    identity_info["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    persist_runner_status(runner_status)

    # ── Parser verdict ──
    parser_code = run_parser_protocol(art_dir)
    if parser_code != 0:
        raise SystemExit(parser_code)

    print(f"\n=== Summary ===")
    print(f"  artifact: {art_dir}")
    print(f"  ladder failures: {ladder_failures if ladder_failures else 'none'}")
    if continuous_result:
        print(f"  continuous all_identical: {continuous_result.get('all_responses_identical')}")
    print(f"\n  Run parser:")
    print(f"  python scripts/parse-kv-bounded-release-stage3b-2a.py {art_dir}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if _ACTIVE_MANIFEST is not None and \
                _ACTIVE_MANIFEST.get("runner_status") != "run_complete":
            reason = _LAST_FAILURE_REASON or f"{type(exc).__name__}: {exc}"
            mark_failure(reason)
            persist_runner_status("run_incomplete")
        raise
