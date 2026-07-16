#!/usr/bin/env python3
"""Frozen three-round E2G/E2I controlled runner.

The internal --execute mode receives one execution.json path.  No execution
setting is accepted through argv or inherited environment in that mode.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import shlex
import signal
import subprocess
import sys
import time
from typing import Any


PLAN = (
    (1, 1, "E2G"), (1, 2, "E2I"),
    (2, 1, "E2I"), (2, 2, "E2G"),
    (3, 1, "E2G"), (3, 2, "E2I"),
)
COMMON_ARGS = (
    "--ctx-size", "2048", "--n-predict", "128", "--batch-size", "128",
    "--ubatch-size", "128", "--seed", "1", "--temp", "0",
    "--cache-type-k", "f32", "--cache-type-v", "f32", "--kv-unified",
    "--parallel", "4", "--log-verbosity", "4", "--no-log-prefix",
    "--no-log-timestamps",
)
PROTOCOL_ENV = {
    "LLAMA_GRAPH_REUSE_DISABLE": "0",
    "LLAMA_KV_ACTIVE_TOKEN_STATS": "1",
    "LLAMA_KV_TEST_MODE": "1",
    "LLAMA_KV_PAGED_IO_STATS": "0",
    "LLAMA_KV_IDLE_NUM_IDLE_SEQS": "2",
    "LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS": "256",
    "LLAMA_KV_CACHE_DEBUG": "0",
    "LLAMA_KV_LAZY_CLEAR": "0",
    "LLAMA_KV_LAZY_TAIL": "0",
    "LLAMA_KV_PAGED": "1",
    "LLAMA_KV_PAGED_INGRAPH": "1",
    "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
    "LLAMA_KV_PAGED_BLOCK_SIZE": "16",
    "LLAMA_KV_PAGED_SHIFT": "0",
    "LLAMA_KV_PAGED_RELEASE": "0",
    "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS": "1",
    "LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS": "0",
    "LLAMA_KV_PAGED_RESUME_PREFETCH": "0",
    "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "0",
    "LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS": "1",
    "LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP": "1",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS": "1",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP": "1",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE": "off",
    "LLAMA_KV_PAGED_RESUME_PENDING_TOKEN": "96",
    "LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS": "0",
    "LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME": "0",
    "LLAMA_KV_SWAP": "0",
    "LLAMA_KV_SWAP_MODE": "exact",
    "LLAMA_KV_SWAP_WINDOW": "0",
    "LLAMA_KV_SWAP_SINK": "0",
    "LLAMA_KV_SWAP_RSS_SAMPLE": "0",
    "LLAMA_KV_SWAP_MADVISE": "0",
    "LLAMA_KV_SWAP_BACKEND_SELFTEST": "0",
    "LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST": "0",
    "LLAMA_KV_E2_GET_ROWS_PROFILE": "0",
    "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "0",
    "LLAMA_KV_PAGED_TIMING": "0",
    "LLAMA_KV_PAGED_RESUME_TIMING": "0",
    "LLAMA_KV_PAGED_RESUME_TIMING_STEP": "0",
    "LLAMA_KV_PAGED_TRACE": "0",
    "LLAMA_KV_PAGED_IDLE_TRACE": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE_MAX": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE_ONCE": "0",
    "LLAMA_KV_PAGED_MINCORE": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES": "0",
    "LLAMA_KV_PAGED_SHADOW_VALIDATE": "0",
}
BASE_ENV_DEFAULTS = {
    "HOME": "/tmp",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
}
TERM_SIGNALS = {"TERM": signal.SIGTERM}


def fail(message: str) -> "NoReturn":
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    if value not in {"0", "1"}:
        fail(f"{name} must be 0 or 1")
    return value == "1"


def positive_number(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        value = float(raw)
    except ValueError:
        fail(f"{name} must be a positive number")
    if value <= 0:
        fail(f"{name} must be a positive number")
    return value


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: pathlib.Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: pathlib.Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def proc_starttime(pid: int) -> int:
    return int(pathlib.Path(f"/proc/{pid}/stat").read_text().split()[21])


def proc_memory(pid: int) -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0])
    return values["VmRSS"], values["VmHWM"]


def execute(execution_path: pathlib.Path) -> int:
    execution = json.loads(execution_path.read_text())
    run_dir = execution_path.parent
    stdout_path, stderr_path = run_dir / "stdout", run_dir / "stderr"
    timeout = float(execution["timeout"]["seconds"])
    grace = float(execution["timeout"]["termination_grace_seconds"])
    term_signal = TERM_SIGNALS[execution["timeout"]["term_signal"]]
    auxiliary = execution["auxiliary"]["rss"]
    sample_path = run_dir / "rss_samples.tsv"
    sample_path.write_text("elapsed_ms\tpid\tstarttime\tvmrss_kb\tvmhwm_kb\n")
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        proc = subprocess.Popen(
            execution["argv"], cwd=execution["cwd"], env=execution["env"],
            stdout=stdout, stderr=stderr, start_new_session=True,
        )
        starttime = proc_starttime(proc.pid)
        process = {
            "pid": proc.pid,
            "starttime": starttime,
            "argv": execution["argv"],
            "cwd": execution["cwd"],
            "env": execution["env"],
            "exe": os.readlink(f"/proc/{proc.pid}/exe"),
        }
        write_json(run_dir / "process.json", process)
        term_sent_at: float | None = None
        term_sent = False
        kill_sent = False
        samples = sample_path.open("a") if auxiliary["enabled"] else None
        try:
            while proc.poll() is None:
                now = time.monotonic()
                if not term_sent and now - started >= timeout:
                    os.killpg(proc.pid, term_signal)
                    term_sent = True
                    term_sent_at = now
                elif term_sent and not kill_sent and now - term_sent_at >= grace:
                    os.killpg(proc.pid, signal.SIGKILL)
                    kill_sent = True
                if samples is not None:
                    try:
                        if proc_starttime(proc.pid) == starttime:
                            rss, hwm = proc_memory(proc.pid)
                            samples.write(f"{int((now-started)*1000)}\t{proc.pid}\t{starttime}\t{rss}\t{hwm}\n")
                            samples.flush()
                    except (FileNotFoundError, ProcessLookupError, KeyError):
                        pass
                time.sleep(min(float(auxiliary["sample_interval_seconds"]), 0.05))
        finally:
            if samples is not None:
                samples.close()
        returncode = proc.wait()
    result = {
        "returncode": returncode,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
        "timed_out": term_sent,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
    }
    write_json(run_dir / "execution_result.json", result)
    (run_dir / "exit_code").write_text(f"{returncode}\n")
    return 0


def replay(execution: dict[str, Any]) -> str:
    timeout = execution["timeout"]
    env_items = " \\\n+  ".join(f"{shlex.quote(k)}={shlex.quote(v)}" for k, v in sorted(execution["env"].items()))
    argv = " \\\n+  ".join(shlex.quote(item) for item in execution["argv"])
    return (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(execution['cwd'])}\n"
        f"exec timeout --signal={timeout['term_signal']} "
        f"--kill-after={timeout['termination_grace_seconds']} {timeout['seconds']} \\\n+  env -i {env_items} \\\n+  {argv}\n"
    )


def cmake_provenance(binary: pathlib.Path) -> dict[str, Any]:
    cache = binary.parents[1] / "CMakeCache.txt"
    selected: dict[str, str] = {}
    compiler = ""
    if cache.is_file():
        for line in cache.read_text(errors="replace").splitlines():
            if line.startswith("//") or line.startswith("#") or "=" not in line or ":" not in line.split("=", 1)[0]:
                continue
            left, value = line.split("=", 1)
            key = left.split(":", 1)[0]
            if key in {"CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "GGML_NATIVE", "GGML_CPU", "GGML_CPU_ALL_VARIANTS", "GGML_CPU_REPACK", "GGML_OPENMP", "GGML_BLAS"}:
                selected[key] = value
        compiler = selected.get("CMAKE_CXX_COMPILER", "")
    version = "unavailable"
    if compiler and pathlib.Path(compiler).is_file():
        version = subprocess.run([compiler, "--version"], text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, check=False).stdout.splitlines()[0]
    return {"cmake_cache": str(cache), "build_type": selected.get("CMAKE_BUILD_TYPE", "unknown"),
            "key_config": selected, "compiler": {"path": compiler or "unknown", "version": version}}


def host_provenance() -> dict[str, Any]:
    os_release: dict[str, str] = {}
    release = pathlib.Path("/etc/os-release")
    if release.is_file():
        for line in release.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip('"')
    cpu_model = "unknown"
    for line in pathlib.Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    mem_total = "unknown"
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            mem_total = line.split(":", 1)[1].strip()
            break
    cgroup = pathlib.Path("/proc/self/cgroup").read_text(errors="replace").splitlines()
    cgroup_files: dict[str, str] = {}
    for name in ("cpu.max", "memory.max", "memory.current", "cpuset.cpus.effective"):
        path = pathlib.Path("/sys/fs/cgroup") / name
        if path.is_file():
            cgroup_files[name] = path.read_text(errors="replace").strip()
    return {
        "hostname": platform.node(), "os": os_release, "kernel": platform.release(),
        "machine": platform.machine(), "cpu": {"model": cpu_model, "logical_count": os.cpu_count()},
        "memory": {"mem_total": mem_total}, "affinity_cpus": sorted(os.sched_getaffinity(0)),
        "cgroup": {"membership": cgroup, "v2_root_files": cgroup_files},
    }


def extract_sequences(run_dir: pathlib.Path) -> None:
    lines = (run_dir / "stdout").read_bytes().splitlines(keepends=True)
    normalized = [line.rstrip(b"\r\n") for line in lines]
    for name, begin, end in (("seq1", b"===SEQ1_ACTIVE_BEGIN===", b"===SEQ1_ACTIVE_END==="),
                             ("seq0", b"===SEQ0_RESUME_BEGIN===", b"===SEQ0_RESUME_END===")):
        if normalized.count(begin) == 1 and normalized.count(end) == 1:
            left, right = normalized.index(begin), normalized.index(end)
            content = b"".join(lines[left + 1:right]) if left < right else b""
        else:
            content = b""
        (run_dir / name).write_bytes(content)


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--execute":
        return execute(pathlib.Path(sys.argv[2]).resolve())
    if len(sys.argv) != 1:
        fail("runner takes configuration only from its documented environment")

    root = pathlib.Path(os.environ.get("ROOT", pathlib.Path(__file__).resolve().parents[1])).resolve()
    binary = pathlib.Path(os.environ.get("BINARY", root / "build/bin/llama-kv-idle-swap-resume")).resolve()
    model = pathlib.Path(os.environ.get("MODEL", "/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf")).resolve()
    parser = root / "scripts/parse-kv-paged-identity-controlled-ab.py"
    wrapper = root / "scripts/kv-paged-identity-controlled-ab.sh"
    runner = pathlib.Path(__file__).resolve()
    dry_run = env_bool("DRY_RUN")
    allow_dirty = env_bool("ALLOW_DIRTY")
    collect_rss = env_bool("COLLECT_RSS")
    timeout = positive_number("CASE_TIMEOUT_SEC", "900")
    grace = positive_number("TERMINATION_GRACE_SEC", "10")
    interval = positive_number("RSS_SAMPLE_INTERVAL_SEC", "0.10")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = pathlib.Path(os.environ.get("OUTPUT_ROOT", f"/root/oscomp/kv_logs/kv_paged_identity_controlled_ab_{timestamp}_{os.getpid()}")).resolve()
    if allow_dirty and not dry_run:
        fail("ALLOW_DIRTY is permitted only for dry-run planning")
    for path, label in ((root / ".git", "repository"), (binary, "binary"), (model, "model"),
                        (parser, "parser"), (wrapper, "wrapper")):
        if not path.exists():
            fail(f"{label} not found: {path}")
    if not os.access(binary, os.X_OK):
        fail(f"binary not executable: {binary}")
    if output.exists():
        fail(f"output path already exists: {output}")
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).splitlines()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if not dry_run and status:
        fail("formal run requires a clean worktree")

    output.joinpath("runs").mkdir(parents=True)
    output.joinpath("swap").mkdir()
    execution_order = output / "execution_order.log"
    execution_order.write_text("")
    binary_id, model_id = identity(binary), identity(model)
    framework = {"wrapper": identity(wrapper), "runner": identity(runner), "parser": identity(parser)}
    manifest: dict[str, Any] = {
        "protocol": "kv_paged_identity_controlled_ab", "version": 2,
        "repo": {"path": str(root), "head": head, "dirty": bool(status), "status_porcelain": status},
        "binary": binary_id, "model": model_id, "framework": framework,
        "provenance": {"host": host_provenance(), "build": cmake_provenance(binary)},
        "workload": {"argv_after_model": list(COMMON_ARGS), "rounds": 3,
                     "order": "R1 G-I, R2 I-G, R3 G-I"},
        "execution": {"dry_run": dry_run, "allow_dirty": allow_dirty, "timeout_seconds": timeout,
                      "termination_grace_seconds": grace, "rss_auxiliary_enabled": collect_rss,
                      "rss_sample_interval_seconds": interval},
        "planned_runs": [{"round": r, "run_order": o, "case": c} for r, o, c in PLAN],
        "completed_runs": [],
    }
    write_json(output / "manifest.json", manifest)

    for round_no, order, case_id in PLAN:
        name = f"round_{round_no}_order_{order:02d}_{case_id}"
        run_dir = output / "runs" / name
        run_dir.mkdir()
        execution_order.write_text(execution_order.read_text() + f"round={round_no} run_order={order} case={case_id}\n")
        child_env = dict(BASE_ENV_DEFAULTS)
        child_env.update(PROTOCOL_ENV)
        child_env["LLAMA_KV_PAGED_IDENTITY_FAST_PATH"] = "1" if case_id == "E2I" else "0"
        child_env["LLAMA_KV_SWAP_DIR"] = str(output / "swap")
        argv = [str(binary), "-m", str(model), *COMMON_ARGS]
        execution = {
            "cwd": str(root), "binary": str(binary), "argv": argv,
            "env": dict(sorted(child_env.items())),
            "timeout": {"seconds": timeout, "term_signal": "TERM", "termination_grace_seconds": grace},
            "auxiliary": {"rss": {"enabled": collect_rss, "sample_interval_seconds": interval}},
        }
        meta = {"round": round_no, "run_order": order, "case": case_id,
                "binary_sha256": binary_id["sha256"], "model_sha256": model_id["sha256"]}
        write_json(run_dir / "run.json", meta)
        write_json(run_dir / "execution.json", execution)
        write_json(run_dir / "environment.json", execution["env"])
        replay_path = run_dir / "replay.sh"
        replay_path.write_text(replay(execution))
        replay_path.chmod(0o755)
        for artifact in ("stdout", "stderr", "seq0", "seq1", "process.json", "execution_result.json"):
            (run_dir / artifact).write_text("")
        (run_dir / "rss_samples.tsv").write_text("elapsed_ms\tpid\tstarttime\tvmrss_kb\tvmhwm_kb\n")
        if dry_run:
            (run_dir / "exit_code").write_text("DRY_RUN\n")
        else:
            subprocess.run([sys.executable, str(runner), "--execute", str(run_dir / "execution.json")], check=True)
            extract_sequences(run_dir)
        artifacts = {name: identity(run_dir / name) for name in (
            "run.json", "execution.json", "environment.json", "replay.sh", "stdout", "stderr",
            "exit_code", "seq0", "seq1", "process.json", "execution_result.json", "rss_samples.tsv")}
        manifest["completed_runs"].append({**meta, "artifacts": artifacts})
        write_json(output / "manifest.json", manifest)

    manifest["execution_order"] = identity(execution_order)
    write_json(output / "manifest.json", manifest)
    command = [sys.executable, str(parser), "--dry-run", str(output)] if dry_run else [sys.executable, str(parser), str(output)]
    write_json(output / "parser_command.json", command)
    parsed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    (output / "parser.stdout").write_text(parsed.stdout)
    (output / "parser.stderr").write_text(parsed.stderr)
    (output / "parser.exit_code").write_text(f"{parsed.returncode}\n")
    manifest["parser_run"] = {
        name: identity(output / name)
        for name in ("parser_command.json", "parser.stdout", "parser.stderr", "parser.exit_code")
    }
    write_json(output / "manifest.json", manifest)
    if parsed.stdout:
        print(parsed.stdout, end="")
    if parsed.stderr:
        print(parsed.stderr, end="", file=sys.stderr)
    if parsed.returncode != 0:
        return parsed.returncode
    print(("dry-run VALID" if dry_run else "artifact VALID") + f": {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
