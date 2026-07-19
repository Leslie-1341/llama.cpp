#!/usr/bin/env python3
"""Structural positive and fail-closed negatives for the Stage 3A-1C protocol."""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-server-kv-pressure-stage3a-1c.py"
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
ZERO_ENV = {
    "LLAMA_KV_PAGED_RELEASE": "0", "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0", "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_RESUME_PREFETCH": "0", "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "0", "LLAMA_KV_SWAP": "0",
    "LLAMA_KV_SWAP_MADVISE": "0",
}
ON_ENV = {
    "LLAMA_KV_PRESSURE_SAMPLER": "1", "LLAMA_KV_LOW_WATER_RSS_KB": "1",
    "LLAMA_KV_PRESSURE_RSS_KB": "2", "LLAMA_KV_CRITICAL_RSS_KB": "3",
    "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "60000",
    "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "60000",
}
TIMEOUTS = {
    "startup": 5.0, "health": 180.0, "completion": 180.0,
    "sleep": 30.0, "resume": 180.0, "shutdown": 15.0, "case_total": 430.0,
}
PURPOSE = "FORCED_LIFECYCLE_STATE_VALIDATION_ONLY_NOT_REAL_DEPLOYMENT_THRESHOLDS"
MARKER = (
    "kv_pressure_telemetry state=CRITICAL previous_state=NORMAL source=RSS_ABSOLUTE "
    "sample_valid=1 stale=0 config_valid=1 rss_kb=100 cgroup_current_bytes=200 "
    "cgroup_max_bytes=300 cgroup_current_kb=1 cgroup_max_kb=2 cgroup_high_kb=0 "
    "psi_some_avg10=0 psi_full_avg10=0 sample_latency_ns=1000 sample_count=1 "
    "skip_count=0 idle=0 trigger=first,state,source\n"
)


def dump(path: pathlib.Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: pathlib.Path) -> dict:
    return {"path": str(path), "size": path.stat().st_size, "sha256": digest(path)}


def stream_objects() -> list[dict]:
    chunks = [
        {"stop": False, "content": text, "tokens": [index]}
        for index, text in enumerate(("one", " two", " three", "."), 1)
    ]
    chunks.append({
        "stop": True, "content": "", "tokens": [], "stop_type": "limit",
        "tokens_predicted": 4,
        "timings": {"predicted_n": 4, "predicted_ms": 20.0,
                    "predicted_per_token_ms": 5.0, "predicted_per_second": 200.0},
    })
    return chunks


def write_stream(directory: pathlib.Path, stem: str = "completion",
                 objects: list[dict] | None = None, arrivals: list[int] | None = None) -> None:
    objects = stream_objects() if objects is None else objects
    payloads = [json.dumps(item, separators=(",", ":")) for item in objects]
    (directory / f"{stem}.sse").write_text("".join(f"data: {item}\n\n" for item in payloads))
    arrivals = arrivals or [1_010_000_000, 1_020_000_000, 1_030_000_000, 1_040_000_000, 1_041_000_000]
    dump(directory / f"{stem}.events.json", {
        "request_started_monotonic_ns": 1_000_000_000, "http_status": 200,
        "events": [{"raw_payload": payload, "arrival_monotonic_ns": arrival}
                   for payload, arrival in zip(payloads, arrivals)],
        "stream_ended_monotonic_ns": 1_050_000_000, "clock": "time.monotonic_ns",
    })


def valid_metrics() -> dict:
    timings = {"predicted_n": 4, "predicted_ms": 20.0,
               "predicted_per_token_ms": 5.0, "predicted_per_second": 200.0}
    return {
        "ttft_ms": 10.0, "tpot_ms": 5.0, "throughput_tps": 200.0,
        "sse_chunk_interval_p95_ms": 10.0, "sse_chunk_interval_p99_ms": 10.0,
        "predicted_n": 4, "valid_token_count": 4, "valid_sse_chunk_count": 4,
        "sse_chunk_interval_n": 3, "all_chunks_single_token": True,
        "server_timings": timings,
    }


class Fixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.repo = self.root / "repo"; self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.email", "fixture@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "config", "user.name", "Fixture"], check=True)
        (self.repo / "tracked").write_text("fixture\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "tracked"], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-qm", "fixture"], check=True)
        self.binary = self.root / "llama-server"
        self.binary.write_text("#!/bin/sh\nexit 0\n"); self.binary.chmod(0o755)
        self.model = self.root / "model.gguf"; self.model.write_bytes(b"model")
        self.artifact = self.root / "artifact"; (self.artifact / "cases").mkdir(parents=True)
        specs = [{"name": n, "kind": k, "round": r, "order": o, "variant": v}
                 for n, k, r, o, v in PLAN]
        head = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        framework = {"runner": identity(RUNNER), "parser": identity(PARSER)}
        binary_id, model_id = identity(self.binary), identity(self.model)
        dump(self.artifact / "manifest.json", {
            "protocol": "server_kv_pressure_stage3a_1c", "version": 2,
            "repo": {"path": str(self.repo), "head": head, "dirty": False, "status_porcelain": []},
            "binary": binary_id, "model": model_id, "framework": framework,
            "dry_run": False, "pressure_settings_purpose": PURPOSE,
            "frozen": {"on_env": ON_ENV, "timeouts_s": TIMEOUTS},
            "planned_cases": specs, "completed_cases": [spec["name"] for spec in specs],
            "post_run": {"repo": {"head": head, "dirty": False, "status_porcelain": []},
                         "binary": binary_id, "model": model_id, "framework": framework},
        })
        for spec in specs:
            self._case(spec)
        self.seal()

    def _case(self, spec: dict) -> None:
        directory = self.artifact / "cases" / spec["name"]; directory.mkdir()
        dump(directory / "case.json", spec)
        env = dict(ZERO_ENV); env.update(ON_ENV)
        if spec["variant"] == "OFF":
            env["LLAMA_KV_PRESSURE_SAMPLER"] = "0"
        if spec["kind"] == "idle_limit":
            env.update({"LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "250",
                        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000"})
        sleep = "2" if spec["kind"] == "lifecycle" else "-1"
        case_index = next(index for index, item in enumerate(PLAN) if item[0] == spec["name"])
        port = 12000 + case_index
        argv = [str(self.binary), "-m", str(self.model), "--host", "127.0.0.1", "--port", str(port),
                "--ctx-size", "1024", "--batch-size", "128", "--ubatch-size", "128",
                "--parallel", "1", "--seed", "1", "--temp", "0", "--no-warmup",
                "--sleep-idle-seconds", sleep, "--log-verbosity", "4", "--no-log-prefix", "--no-log-timestamps"]
        dump(directory / "execution.json", {
            "argv": argv, "launch_argv": argv, "env": env, "cwd": str(self.repo), "port": port,
            "strace_attach_after_health": spec["kind"] == "strace",
        })
        dump(directory / "environment.json", env)
        dump(directory / "process.json", {"pid": 123, "server_argv": argv, "env": env,
                                           "exe": str(self.binary), "expected_exe": str(self.binary)})
        dump(directory / "result.json", {"returncode": -15, "shutdown_requested": True,
                                          "sigkill_used": False, "residual_process": False,
                                          "unexpected_exit": False})
        dump(directory / "health.json", {"status_code": 200, "body": {"status": "ok"}})
        write_stream(directory); dump(directory / "metrics.json", valid_metrics())
        (directory / "server.stdout").write_text("")
        count = 0 if spec["variant"] == "OFF" else (2 if spec["kind"] == "lifecycle" else 1)
        (directory / "server.stderr").write_text(MARKER * count)
        initial_end = 0 if spec["variant"] == "OFF" else len(MARKER.encode())
        dump(directory / "completion_window.json", {"stderr_start": 0, "stderr_end": initial_end})
        phases = {}
        for name, limit in TIMEOUTS.items():
            key = "case" if name == "case_total" else name
            if key in {"sleep", "resume"} and spec["kind"] != "lifecycle":
                phases[key] = {"status": "NOT_APPLICABLE", "timeout_s": limit}
            else:
                phases[key] = {"status": "PASS", "timeout_s": limit,
                               "started_monotonic_ns": (case_index + 1) * 1_000_000_000,
                               "ended_monotonic_ns": (case_index + 1) * 1_000_000_000 + 1_000_000}
        dump(directory / "phases.json", phases)
        if spec["kind"] == "lifecycle":
            dump(directory / "sleep_observations.json", [{"status_code": 200, "body": {"is_sleeping": True}}])
            dump(directory / "resume_props.json", {"status_code": 200, "body": {"is_sleeping": False}})
            write_stream(directory, "wake_completion"); dump(directory / "wake_metrics.json", valid_metrics())
            dump(directory / "wake_completion_window.json", {
                "stderr_start": len(MARKER.encode()), "stderr_end": 2 * len(MARKER.encode())})
        if spec["kind"] == "idle_limit":
            dump(directory / "idle_props.json", {"status_code": 200, "body": {"is_sleeping": False}})
            size = (directory / "server.stderr").stat().st_size
            dump(directory / "idle_window.json", {"duration_ms": 1500.0, "stderr_start": size, "stderr_end": size})
        if spec["kind"] == "strace":
            dump(directory / "strace_process.json", {
                "argv": ["/usr/bin/strace", "-ff", "-qq", "-yy", "-s", "4096",
                         "-e", "trace=openat,read,close", "-o", "strace", "-p", "123"],
                "attached_to_pid": 123, "attach_after_health": True,
                "shutdown_requested": True, "returncode": 0, "residual_process": False,
            })
            if spec["variant"] == "ON":
                paths = ["/proc/self/statm", "/proc/pressure/memory", "/sys/fs/cgroup/a/memory.current",
                         "/sys/fs/cgroup/a/memory.max", "/sys/fs/cgroup/a/memory.high",
                         "/sys/fs/cgroup/a/memory.pressure"]
            else:
                paths = ["/etc/localtime"]
            lines = []
            for fd, path in enumerate(paths, 3):
                lines.extend([f'openat(AT_FDCWD, "{path}", O_RDONLY) = {fd}<{path}>',
                              f'read({fd}<{path}>, "x", 1) = 1', f'close({fd}<{path}>) = 0'])
            (directory / "strace.123").write_text("\n".join(lines) + "\n")

    def seal(self) -> None:
        entries = []
        for path in sorted(item for item in self.artifact.rglob("*")
                           if item.is_file() and item.name != "inventory.sha256.json"):
            entries.append({"path": path.relative_to(self.artifact).as_posix(),
                            "size": path.stat().st_size, "sha256": digest(path)})
        dump(self.artifact / "inventory.sha256.json", {
            "algorithm": "sha256", "excludes": ["inventory.sha256.json"], "files": entries})

    def close(self) -> None:
        self.temp.cleanup()

    def run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, str(PARSER), str(self.artifact)], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


class ParserSyntheticTest(unittest.TestCase):
    def fixture(self) -> Fixture:
        fixture = Fixture(); self.addCleanup(fixture.close); return fixture

    def assert_rejected(self, mutate, *, reseal: bool = True) -> None:
        fixture = self.fixture(); mutate(fixture)
        if reseal:
            fixture.seal()
        result = fixture.run()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_complete_artifact_is_valid_and_exploratory_only(self) -> None:
        fixture = self.fixture(); result = fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((fixture.artifact / "summary.json").read_text())
        self.assertEqual(summary["artifact_status"], "VALID")
        self.assertEqual(summary["pressure_settings_purpose"], PURPOSE)
        self.assertEqual(summary["performance_conclusion"], "EXPLORATORY_ONLY_NO_FORMAL_BENEFIT_CLAIM")
        self.assertEqual(summary["idle_sampling_limit"]["status"], "OBSERVED_CURRENT_LIMITATION_NOT_FAILURE")
        aggregate = summary["performance_aggregate"]["ttft_ms"]["ON"]
        self.assertEqual(aggregate["n"], 3)
        self.assertIn("NOT_REPORTED", aggregate["tail_quantiles"])
        self.assertEqual(summary["strace_attribution"]["strace_off_correctness"]["sampler_paths"], [])
        self.assertIn("/proc/self/statm", summary["strace_attribution"]["strace_on_correctness"]["sampler_paths"])

    def test_first_trigger_set_duplicate_order_and_resume_isolation_fail_closed(self) -> None:
        mutations = []
        mutations.append(lambda f: (f.artifact / "cases/ab_r1_on/server.stderr").write_text(
            MARKER.replace("trigger=first,state,source", "trigger=first,first,state")))
        mutations.append(lambda f: (f.artifact / "cases/ab_r1_on/server.stderr").write_text(
            MARKER.replace("sample_count=1", "sample_count=2").replace("trigger=first,state,source", "trigger=periodic") + MARKER))
        mutations.append(lambda f: (f.artifact / "cases/ab_r1_on/server.stderr").write_text(MARKER + MARKER))
        def old_resume(f):
            path = f.artifact / "cases/on_lifecycle/wake_completion_window.json"
            dump(path, {"stderr_start": 0, "stderr_end": len(MARKER.encode())})
        mutations.append(old_resume)
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_rejected(mutation)

    def test_sse_malformed_terminal_timing_metric_and_timestamp_negatives(self) -> None:
        def malformed(f):
            directory = f.artifact / "cases/ab_r1_off"
            evidence = json.loads((directory / "completion.events.json").read_text())
            evidence["events"][0]["raw_payload"] = "{bad"
            dump(directory / "completion.events.json", evidence)
            lines = (directory / "completion.sse").read_text().splitlines()
            lines[0] = "data: {bad"
            (directory / "completion.sse").write_text("\n".join(lines) + "\n")
        def no_terminal(f):
            directory = f.artifact / "cases/ab_r1_off"; objects = stream_objects()[:-1]
            write_stream(directory, objects=objects, arrivals=[1_010_000_000, 1_020_000_000, 1_030_000_000, 1_040_000_000])
        def timing_missing(f):
            directory = f.artifact / "cases/ab_r1_off"; objects = stream_objects()
            del objects[-1]["timings"]["predicted_ms"]
            write_stream(directory, objects=objects)
        def metrics_conflict(f):
            path = f.artifact / "cases/ab_r1_off/metrics.json"; value = json.loads(path.read_text())
            value["ttft_ms"] = 999.0; dump(path, value)
        def timestamp_missing(f):
            path = f.artifact / "cases/ab_r1_off/completion.events.json"; value = json.loads(path.read_text())
            del value["events"][0]["arrival_monotonic_ns"]; dump(path, value)
        def timestamp_insufficient(f):
            directory = f.artifact / "cases/ab_r1_off"; objects = [stream_objects()[0], stream_objects()[-1]]
            objects[-1]["tokens_predicted"] = 1
            objects[-1]["timings"] = {"predicted_n": 1, "predicted_ms": 5.0,
                                       "predicted_per_token_ms": 5.0, "predicted_per_second": 200.0}
            write_stream(directory, objects=objects, arrivals=[1_010_000_000, 1_011_000_000])
        for name, mutation in (("malformed", malformed), ("terminal", no_terminal),
                               ("timing", timing_missing), ("metric", metrics_conflict),
                               ("timestamp", timestamp_missing), ("samples", timestamp_insufficient)):
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_exact_case_file_set_and_checksum_fail_closed(self) -> None:
        self.assert_rejected(lambda f: (f.artifact / "cases/unexpected").mkdir())
        self.assert_rejected(lambda f: (f.artifact / "cases/ab_r1_off/extra.txt").write_text("extra"))
        self.assert_rejected(lambda f: (f.artifact / "cases/ab_r1_off/health.json").unlink())
        self.assert_rejected(lambda f: (f.artifact / "cases/ab_r1_off/metrics.json").write_text("tampered"), reseal=False)

    def test_all_identity_drift_modes_fail_closed(self) -> None:
        def framework(label):
            def mutate(f):
                path = f.artifact / "manifest.json"; value = json.loads(path.read_text())
                value["framework"][label]["sha256"] = "0" * 64; dump(path, value)
            return mutate
        def binary(f): f.binary.write_text("changed\n")
        def model(f): f.model.write_bytes(b"changed")
        def head(f):
            (f.repo / "tracked").write_text("second\n")
            subprocess.run(["git", "-C", str(f.repo), "add", "tracked"], check=True)
            subprocess.run(["git", "-C", str(f.repo), "commit", "-qm", "second"], check=True)
        def dirty(f): (f.repo / "untracked").write_text("dirty\n")
        for name, mutation in (("runner", framework("runner")), ("parser", framework("parser")),
                               ("binary", binary), ("model", model), ("head", head), ("dirty", dirty)):
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_off_telemetry_and_structured_action_markers_fail_closed(self) -> None:
        self.assert_rejected(lambda f: (f.artifact / "cases/ab_r1_off/server.stderr").write_text(MARKER))
        structured = (
            "paged_release_blocks()\n", "kv_swap_action bytes=1\n", "kv_prefetch_event count=1\n",
            "kv_madvise_marker advice=DONTNEED\n", "reclaim_calls=1\n",
        )
        for value in structured:
            with self.subTest(value=value):
                self.assert_rejected(lambda f, value=value: (
                    f.artifact / "cases/ab_r3_on/server.stderr").write_text(MARKER + value))
        # Bare prose is not a mutation marker and must not become a broad false positive.
        fixture = self.fixture()
        path = fixture.artifact / "cases/ab_r3_on/server.stderr"
        path.write_text(MARKER + "release swap prefetch madvise reclaim are discussed in prose\n")
        fixture.seal()
        self.assertEqual(fixture.run().returncode, 0)

    def test_every_phase_timeout_and_residual_process_fail_closed(self) -> None:
        for phase in ("startup", "health", "completion", "sleep", "resume", "shutdown", "case"):
            with self.subTest(phase=phase):
                def mutation(f, phase=phase):
                    case = "on_lifecycle" if phase in {"sleep", "resume"} else "ab_r1_off"
                    path = f.artifact / "cases" / case / "phases.json"
                    value = json.loads(path.read_text()); value[phase]["status"] = "TIMEOUT"; dump(path, value)
                self.assert_rejected(mutation)
        def residual(f):
            path = f.artifact / "cases/ab_r1_off/result.json"; value = json.loads(path.read_text())
            value["residual_process"] = True; dump(path, value)
        self.assert_rejected(residual)

    def test_request_exit_order_strace_and_idle_boundaries_fail_closed(self) -> None:
        def health(f): dump(f.artifact / "cases/ab_r2_off/health.json", {"status_code": 503, "body": {}})
        def exit_bad(f):
            path = f.artifact / "cases/ab_r2_off/result.json"; value = json.loads(path.read_text())
            value["returncode"] = 7; value["unexpected_exit"] = True; dump(path, value)
        def order(f):
            path = f.artifact / "manifest.json"; value = json.loads(path.read_text())
            value["completed_cases"][0], value["completed_cases"][1] = value["completed_cases"][1], value["completed_cases"][0]
            dump(path, value)
        def overlap(f):
            path = f.artifact / "cases/ab_r1_on/phases.json"; value = json.loads(path.read_text())
            value["case"]["started_monotonic_ns"] = 1_000_500_000; dump(path, value)
        def reused_port(f):
            source = json.loads((f.artifact / "cases/ab_r1_off/execution.json").read_text())["port"]
            directory = f.artifact / "cases/ab_r1_on"
            execution = json.loads((directory / "execution.json").read_text())
            old = str(execution["port"]); execution["port"] = source
            execution["argv"][execution["argv"].index("--port") + 1] = str(source)
            execution["launch_argv"] = list(execution["argv"]); dump(directory / "execution.json", execution)
            process = json.loads((directory / "process.json").read_text())
            process["server_argv"] = list(execution["argv"]); dump(directory / "process.json", process)
        def strace_on_missing(f):
            (f.artifact / "cases/strace_on_correctness/strace.123").write_text(
                'openat(AT_FDCWD, "/etc/localtime", O_RDONLY) = 3</etc/localtime>\nread(3</etc/localtime>, "x", 1) = 1\nclose(3</etc/localtime>) = 0\n')
        def strace_off_polluted(f):
            (f.artifact / "cases/strace_off_correctness/strace.123").write_text(
                'openat(AT_FDCWD, "/proc/self/statm", O_RDONLY) = 3</proc/1/statm>\nread(3</proc/1/statm>, "x", 1) = 1\nclose(3</proc/1/statm>) = 0\n')
        def idle(f):
            directory = f.artifact / "cases/on_idle_250ms"; path = directory / "server.stderr"
            start = path.stat().st_size; path.write_text(path.read_text() + MARKER)
            dump(directory / "idle_window.json", {"duration_ms": 1500.0, "stderr_start": start,
                                                   "stderr_end": path.stat().st_size})
        for name, mutation in (("health", health), ("exit", exit_bad), ("order", order),
                               ("overlap", overlap), ("port", reused_port),
                               ("strace_on", strace_on_missing), ("strace_off", strace_off_polluted),
                               ("idle", idle)):
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_plain_on_first_marker_before_completion_window_passes(self) -> None:
        """Regression: real ab_r1_on first marker precedes completion window (sleep/resume NOT_APPLICABLE)."""
        fixture = self.fixture()
        case_dir = fixture.artifact / "cases/ab_r1_on"
        prefix = b"server startup and health log lines\n"
        suffix = b"\npost-completion log\n"
        stderr = prefix + MARKER.encode() + suffix
        (case_dir / "server.stderr").write_bytes(stderr)
        dump(case_dir / "completion_window.json", {
            "stderr_start": len(prefix) + len(MARKER.encode()),
            "stderr_end": len(stderr),
        })
        fixture.seal()
        result = fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((fixture.artifact / "summary.json").read_text())
        self.assertEqual(summary["artifact_status"], "VALID")

    def test_plain_on_missing_first_and_late_first_fail_closed(self) -> None:
        """Plain ON case with observed[0] lacking first, or first only at a later position."""
        def no_first_at_all(f):
            case_dir = f.artifact / "cases/ab_r1_on"
            (case_dir / "server.stderr").write_text(
                MARKER.replace("trigger=first,state,source", "trigger=state,source"))
        def late_first(f):
            case_dir = f.artifact / "cases/ab_r1_on"
            (case_dir / "server.stderr").write_text(
                MARKER.replace("trigger=first,state,source", "trigger=periodic") + MARKER)
        for name, mutation in (("missing", no_first_at_all), ("late", late_first)):
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_lifecycle_startup_first_marker_before_completion_window_passes(self) -> None:
        """Regression: lifecycle startup first marker precedes completion window stderr_start."""
        fixture = self.fixture()
        case_dir = fixture.artifact / "cases/on_lifecycle"
        marker_bytes = MARKER.encode()
        prefix = b"server startup and health check log lines\n"
        mid = b"\ncompletion request processing log\n"
        suffix = b"\npost-shutdown log\n"
        stderr = prefix + marker_bytes + mid + marker_bytes + suffix
        (case_dir / "server.stderr").write_bytes(stderr)
        # completion window starts AFTER the first (pre-request) marker
        completion_end = len(prefix) + len(marker_bytes) + len(mid)
        dump(case_dir / "completion_window.json", {
            "stderr_start": len(prefix) + len(marker_bytes),
            "stderr_end": completion_end,
        })
        # wake window covers the second (resume) marker
        wake_start = completion_end
        wake_end = completion_end + len(marker_bytes) + len(suffix)
        dump(case_dir / "wake_completion_window.json", {
            "stderr_start": wake_start,
            "stderr_end": wake_end,
        })
        fixture.seal()
        result = fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads((fixture.artifact / "summary.json").read_text())
        self.assertEqual(summary["artifact_status"], "VALID")

    def test_lifecycle_startup_missing_late_duplicate_first_fail_closed(self) -> None:
        """Lifecycle startup must have exactly one earliest first with sample_count=1."""
        def no_startup_first(f):
            case_dir = f.artifact / "cases/on_lifecycle"
            marker_bytes = MARKER.encode()
            prefix = b"startup\n"
            mid = b"\ncompletion\n"
            suffix = b"\npost\n"
            # startup marker lacks "first" trigger; resume marker is normal
            no_first_marker = MARKER.replace("trigger=first,state,source", "trigger=periodic").encode()
            stderr = prefix + no_first_marker + mid + marker_bytes + suffix
            (case_dir / "server.stderr").write_bytes(stderr)
            completion_end = len(prefix) + len(no_first_marker) + len(mid)
            dump(case_dir / "completion_window.json", {"stderr_start": 0, "stderr_end": completion_end})
            dump(case_dir / "wake_completion_window.json", {
                "stderr_start": completion_end, "stderr_end": len(stderr)})

        def late_startup_first(f):
            case_dir = f.artifact / "cases/on_lifecycle"
            marker_bytes = MARKER.encode()
            periodic_bytes = MARKER.replace("trigger=first,state,source", "trigger=periodic").encode()
            prefix = b"startup\n"
            mid = b"\ncompletion\n"
            suffix = b"\npost\n"
            # periodic marker first, first marker later → first_positions != [0]
            stderr = prefix + periodic_bytes + mid + marker_bytes + marker_bytes + suffix
            (case_dir / "server.stderr").write_bytes(stderr)
            completion_end = len(prefix) + len(periodic_bytes) + len(mid) + len(marker_bytes)
            dump(case_dir / "completion_window.json", {"stderr_start": 0, "stderr_end": completion_end})
            wake_start = completion_end
            dump(case_dir / "wake_completion_window.json", {
                "stderr_start": wake_start, "stderr_end": len(stderr)})

        def duplicate_startup_first(f):
            case_dir = f.artifact / "cases/on_lifecycle"
            marker_bytes = MARKER.encode()
            prefix = b"startup\n"
            mid = b"\ncompletion\n"
            suffix = b"\npost\n"
            # two markers both with "first" in startup range
            stderr = prefix + marker_bytes + mid + marker_bytes + marker_bytes + suffix
            (case_dir / "server.stderr").write_bytes(stderr)
            completion_end = len(prefix) + len(marker_bytes) + len(mid) + len(marker_bytes)
            dump(case_dir / "completion_window.json", {"stderr_start": 0, "stderr_end": completion_end})
            dump(case_dir / "wake_completion_window.json", {
                "stderr_start": completion_end, "stderr_end": len(stderr)})

        for name, mutation in (("missing", no_startup_first), ("late", late_startup_first),
                               ("duplicate", duplicate_startup_first)):
            with self.subTest(name=name):
                self.assert_rejected(mutation)

    def test_static_pressure_integration_chain_has_no_kv_mutation(self) -> None:
        runtime = (ROOT / "tools/server/server-kv-pressure.cpp").read_text()
        header = (ROOT / "tools/server/server-kv-pressure.h").read_text()
        context = (ROOT / "tools/server/server-context.cpp").read_text()
        start = context.index("void maybe_sample_kv_pressure(")
        end = context.index("\n    void ", start + 10)
        integration = context[start:end]
        forbidden = re.compile(r"paged_release|swap_(?:in|out)|prefetch_seq|madvise|reclaim", re.I)
        self.assertIsNone(forbidden.search(runtime + header + integration))
        runner = RUNNER.read_text()
        self.assertIn("SIGKILL", runner)
        self.assertIn('trace=openat,read,close', runner)
        self.assertNotIn('trace=madvise', runner)


if __name__ == "__main__":
    unittest.main()
