#!/usr/bin/env python3
"""Fail-closed fixtures for the B0-B2 case-aware server baseline parser."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run-kv-server-b0-b2.py"
PARSER_PATH = ROOT / "scripts/parse-kv-server-b0-b2.py"
SAMPLER_PATH = ROOT / "scripts/kv-controlled-memory-sampler.sh"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(RUNNER_PATH, "kv_server_b0_b2_runner")
PARSER = load_module(PARSER_PATH, "kv_server_b0_b2_parser")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ident(path: pathlib.Path) -> dict:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha_bytes(path.read_bytes())}


def put(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


SAMPLE_HEADER = [
    "elapsed_ms", "pid", "starttime_ticks", "vmrss_kb", "vmhwm_kb",
    "cgroup_memory_current_bytes", "backing_logical_size", "backing_allocated_bytes",
]


def proc_starttime(pid: int) -> int:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return int(text.rsplit(")", 1)[1].split()[19])


def proc_state(pid: int) -> str:
    text = pathlib.Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    return text.rsplit(")", 1)[1].split()[0]


def wait_for_path(path: pathlib.Path, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {path}")


def read_sample_rows(path: pathlib.Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0].split("\t") == SAMPLE_HEADER
    return [line.split("\t") for line in lines[1:]]


def offset(text: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    at = -1
    search = 0
    for _ in range(occurrence + 1):
        at = text.index(needle, search)
        search = at + len(needle)
    end = text.index("\n", at) + 1
    return len(text[:at].encode("utf-8")), len(text[:end].encode("utf-8"))


def raw_claimant(eligible: int, swapped: int, epoch: int = 2) -> dict:
    return {
        "epoch": epoch,
        "exhausted": False,
        "valid": True,
        "target_blocks": eligible + swapped,
        "eligible_resident_blocks": eligible,
        "swapped_blocks": swapped,
        "shared_blocks": 0,
        "blocked_blocks": 0,
    }


def claimant(eligible: int, swapped: int, epoch: int = 2) -> dict:
    return {"seq_id": 0, "active": False, **raw_claimant(eligible, swapped, epoch)}


def resident(pages: int, total_pages: int = 256) -> dict:
    page_size = 4096
    return {
        "status": "available",
        "source": "paged_sample_mincore",
        "object_id": 7,
        "generation": 3,
        "page_size": page_size,
        "total_bytes": total_pages * page_size,
        "resident_bytes": pages * page_size,
        "total_pages": total_pages,
        "resident_pages": pages,
    }


def capability_fields(flex: bool) -> dict[str, str]:
    fields = {key: "1" for key in PARSER.CAPABILITY_REQUIRED}
    if not flex:
        for key in PARSER.ENABLED_CAPABILITY - {"kv_unified"}:
            fields[key] = "0"
    return fields


def capability_line(flex: bool) -> tuple[str, dict[str, str]]:
    fields = capability_fields(flex)
    return "KV_GOVERNOR_CAPABILITY " + " ".join(
        f"{key}={fields[key]}" for key in sorted(fields)) + "\n", fields


def resident_observation(before: dict, after: dict) -> tuple[str, dict[str, str]]:
    fields = {
        "source": "paged_sample_mincore",
        "action": "offload",
        "decision_id": "1",
        "seq_id": "0",
        "transaction_id": "1",
        "server_pid": "0",  # replaced by the fixture writer
        "before_available": "1",
        "before_object_id": str(before["object_id"]),
        "before_generation": str(before["generation"]),
        "before_page_size": str(before["page_size"]),
        "before_total_bytes": str(before["total_bytes"]),
        "before_resident_bytes": str(before["resident_bytes"]),
        "before_total_pages": str(before["total_pages"]),
        "before_resident_pages": str(before["resident_pages"]),
        "after_available": "1",
        "after_object_id": str(after["object_id"]),
        "after_generation": str(after["generation"]),
        "after_page_size": str(after["page_size"]),
        "after_total_bytes": str(after["total_bytes"]),
        "after_resident_bytes": str(after["resident_bytes"]),
        "after_total_pages": str(after["total_pages"]),
        "after_resident_pages": str(after["resident_pages"]),
    }
    return PARSER.RESIDENT_OBSERVATION_MARKER + " " + " ".join(
        f"{key}={value}" for key, value in fields.items()) + "\n", fields


def offload_marker() -> tuple[str, dict[str, str]]:
    claim = claimant(16, 0)
    claimant_value = ":".join(str(value) for value in (
        claim["seq_id"], claim["epoch"], int(claim["active"]), int(claim["exhausted"]),
        int(claim["valid"]), claim["target_blocks"], claim["eligible_resident_blocks"],
        claim["swapped_blocks"], claim["shared_blocks"], claim["blocked_blocks"],
    ))
    fields = {
        "state": "CRITICAL",
        "source": "rss",
        "stale": "0",
        "decision_id": "1",
        "episode": "1",
        "target_bytes": "131072",
        "max_blocks": "64",
        "observed_excess_bytes": "131072",
        "debt_before_bytes": "131072",
        "debt_after_bytes": "65536",
        "offload_armed_before": "1",
        "offload_armed_after": "1",
        "next_action_sample": "2",
        "evaluate_attempted": "1",
        "evaluate_outcome": "completed",
        "evaluate_reason": "none",
        "release_attempted": "0",
        "offload_attempted": "1",
        "selected_seq_id": "0",
        "selected_claimant_epoch": "2",
        "transaction_id": "1",
        "outcome": "completed",
        "reason": "target_satisfied",
        "blocks": "16",
        "bytes": "131072",
        "relieved_bytes": "65536",
        "shortfall_bytes": "0",
        "io_failure": "0",
        "io_errno": "0",
        "state_changed": "1",
        "decision_reason": "offload_submitted",
        "sample_count": "1",
        "idle": "1",
        "claimants": claimant_value,
        "scores": "0:1:none:100:10:20:90:0:20:0",
    }
    return PARSER.MARKER + " " + " ".join(
        f"{key}={value}" for key, value in fields.items()) + "\n", fields


def resume_line(phase: str) -> str:
    return (
        f"{PARSER.RESUME_MARKER} phase={phase} decision_id=1 seq_id=0 "
        "claimant_epoch=2 transaction_id=1 action=prefetch outcome=completed "
        "reason=none graph_allowed=1\n"
    )


def timing_lines() -> tuple[str, dict, dict, list[dict], dict]:
    stage = {
        "decision_id": 1,
        "seq_id": 0,
        "transaction_id": 1,
        "restored_blocks": 2,
        "restored_bytes": 4096,
        "queue_us": 100,
        "gate_us": 260,
        "graph_us": 300,
        "total_us": 900,
    }
    call = {"call": 1, "seq_id": 0, "requested_blocks": 16, "restored_blocks": 2, "phase_events": 2}
    blocks = [
        {"call": 1, "block_index": 0, "physical_block": 3, "validate_us": 10, "read_us": 50, "unpack_us": 20, "commit_us": 5, "phase_sum_us": 85},
        {"call": 1, "block_index": 1, "physical_block": 4, "validate_us": 10, "read_us": 50, "unpack_us": 20, "commit_us": 5, "phase_sum_us": 85},
    ]
    io = {
        "block_swap_out_calls": 16,
        "block_swap_in_calls": 2,
        "backing_read_syscalls": 2,
        "backing_write_syscalls": 16,
        "bytes_read": 4096,
        "bytes_written": 65536,
        "avg_block_swap_in_latency_us": 85,
        "max_block_swap_in_latency_us": 90,
        "block_in_validate_us": 20,
        "block_in_read_us": 100,
        "block_in_unpack_us": 40,
        "block_in_commit_us": 10,
    }
    text = (
        "kv_resume_stage_timing " + " ".join(f"{key}={value}" for key, value in stage.items()) + "\n" +
        "KV_PAGED_PREFETCH_BLOCK_PHASE " + " ".join(f"{key}={value}" for key, value in blocks[0].items()) + "\n" +
        "KV_PAGED_PREFETCH_BLOCK_PHASE " + " ".join(f"{key}={value}" for key, value in blocks[1].items()) + "\n" +
        "KV_PAGED_PREFETCH_PHASE_CALL " + " ".join(f"{key}={value}" for key, value in call.items()) + "\n" +
        "KV_PAGED_IO_STATS " + " ".join(f"{key}={value}" for key, value in io.items()) + "\n"
    )
    return text, stage, call, blocks, io


class BaselineFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.root = self.tmp / "artifact"
        self.root.mkdir()
        self.current_binary = self.tmp / "current-llama-server"
        self.model = self.tmp / "model.gguf"
        self.current_binary.write_bytes(b"current")
        self.model.write_bytes(b"model")
        self.cgroup = {
            "version": "none",
            "path": PARSER.NOT_APPLICABLE,
            "memory_current_file": PARSER.NOT_APPLICABLE,
            "memory_max_file": PARSER.NOT_APPLICABLE,
            "memory_peak_file": PARSER.NOT_APPLICABLE,
            "memory_current": PARSER.NOT_APPLICABLE,
            "memory_max": PARSER.NOT_APPLICABLE,
            "memory_peak_snapshot": PARSER.NOT_APPLICABLE,
        }
        self.timings = {
            "prompt_n": 1025,
            "prompt_ms": 0.9,
            "predicted_n": 32,
            "predicted_ms": 320.0,
            "predicted_per_token_ms": 10.0,
        }
        self.build()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def metadata(self, round_no: int, run_order: int, name: str) -> dict:
        return PARSER.baseline_metadata(round_no, run_order, name)

    def write_requests(self, case: pathlib.Path, prompt_p: list[int], prompt_pq: list[int], start: int) -> dict:
        step1_request = RUNNER.completion_body(prompt_p, 0)
        step2_request = RUNNER.completion_body(prompt_pq, 32, stream=True)
        step1 = {
            "label": "step1",
            "request": step1_request,
            "request_sha256": sha_bytes(json.dumps(step1_request, separators=(",", ":"), sort_keys=True).encode("utf-8")),
            "http_status": 200,
            "response_raw": "{}",
            "response_text": "",
            "response_sha256": sha_bytes(b""),
            "response_timings": {},
            "started_monotonic_ns": start,
            "finished_monotonic_ns": start + 20,
        }
        text = "deterministic continuation"
        step2 = {
            "label": "step2",
            "request": step2_request,
            "request_sha256": sha_bytes(json.dumps(step2_request, separators=(",", ":"), sort_keys=True).encode("utf-8")),
            "http_status": 200,
            "response_raw": 'data: {"content":"deterministic continuation"}\n\ndata: {"stop":true}\n\n',
            "response_text": text,
            "response_sha256": sha_bytes(text.encode("utf-8")),
            "response_timings": dict(self.timings),
            "started_monotonic_ns": start + 100,
            "first_content_monotonic_ns": start + 200,
            "streaming": True,
            "content_type": "text/event-stream; charset=utf-8",
            "stream_event_count": 2,
            "stream_terminal_received": True,
            "stream_error": None,
            "finished_monotonic_ns": start + 1_100,
        }
        (case / "requests.jsonl").write_text(
            json.dumps(step1, sort_keys=True) + "\n" + json.dumps(step2, sort_keys=True) + "\n",
            encoding="utf-8")
        return {"step1": step1, "step2": step2}

    def write_memory(self, case: pathlib.Path, name: str, pid: int, starttime: int) -> None:
        phases = []
        labels = ["server_ready", "after_step1"]
        if name == "FLEXKV_K1_SYNC":
            labels.append("after_offload")
        labels.append("after_step2")
        for index, label in enumerate(labels):
            backing = {"status": "not_observed"} if PARSER.baseline_case_flags(name)["backing"] else PARSER.NOT_APPLICABLE
            phases.append({
                "phase": label,
                "monotonic_ns": 10_000 + index,
                "server_pid": pid,
                "vmrss_kb": 100 + index,
                "vmhwm_kb": 150 + index,
                "cgroup_memory_current_bytes": PARSER.NOT_APPLICABLE,
                "backing": backing,
            })
        put(case / "memory_phases.json", {
            "schema_version": 1,
            "sample_interval_seconds": 0.10,
            "cgroup": self.cgroup,
            "phases": phases,
        })
        (case / "memory_sampler.stderr").write_bytes(b"")
        (case / "memory_samples.tsv").write_text(
            "elapsed_ms\tpid\tstarttime_ticks\tvmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes\n"
            f"0\t{pid}\t{starttime}\t100\t150\t{PARSER.SAMPLE_NOT_APPLICABLE}\t{PARSER.SAMPLE_NOT_APPLICABLE}\t{PARSER.SAMPLE_NOT_APPLICABLE}\n",
            encoding="utf-8")

    def write_execution(self, case: pathlib.Path, metadata: dict, name: str, pid: int, starttime: int) -> dict:
        argv = PARSER.baseline_expected_argv(str(self.current_binary), str(self.model), 2048, name)
        argv[argv.index("<port>")] = str(9000 + pid)
        identity = {
            "pid": pid,
            "starttime_ticks": starttime,
            "cmdline": argv,
            "cmdline_sha256": sha_bytes(b"\0".join(item.encode("utf-8") for item in argv)),
        }
        environment = PARSER.baseline_expected_environment(name)
        environment["PATH"] = "/usr/bin"
        backing = case / "backing" if PARSER.baseline_case_flags(name)["backing"] else None
        sampler_argv = [
            "bash", str(SAMPLER_PATH), "--sample-process", str(pid), str(case / "memory_samples.tsv"),
            str(backing) if backing else "", "0.10", "",
        ]
        execution = {
            "case": metadata["case"],
            "round": metadata["round"],
            "run_order": metadata["run_order"],
            "host": "127.0.0.1",
            "argv": argv,
            "cwd": str(case.resolve()),
            "environment": environment,
            "environment_closure": {"inherits_parent_environment": False, "base_environment_keys": ["HOME", "LANG", "LC_ALL", "PATH"]},
            "binary": ident(self.current_binary),
            "model": ident(self.model),
            "server_identity": identity,
            "server_cgroup": PARSER.NOT_APPLICABLE,
            "cgroup": self.cgroup,
            "memory_sampler": ident(SAMPLER_PATH),
            "memory_sampler_argv": sampler_argv,
        }
        put(case / "environment.json", environment)
        put(case / "execution.json", execution)
        return execution

    def write_resident(self, case: pathlib.Path, pid: int) -> None:
        current = claimant(16, 0)
        raw = [{"id": 0, "is_processing": False, "kv_claimant": raw_claimant(16, 0), "kv_resident": resident(80)}]
        raw_path = case / "resident_after_step1.raw.json"
        raw_path.write_text(json.dumps(raw), encoding="utf-8")
        probe = {
            "source": "GET /slots.kv_resident",
            "server_pid": pid,
            "slots_raw_sha256": sha_bytes(raw_path.read_bytes()),
            "slots_shape": 1,
            "claimant_keys": sorted(raw[0]["kv_claimant"]),
            "logical_claimant_fields_present": True,
            "physical_resident_sample_available": True,
            "resident": raw[0]["kv_resident"],
            "reason": "",
        }
        put(case / "resident_after_step1.json", {
            "raw_path": raw_path.name,
            "raw_sha256": sha_bytes(raw_path.read_bytes()),
            "observed_monotonic_ns": 500,
            "stderr_end": 0,
            "slots_http_status": 200,
            "claimant": current,
            "physical_probe": probe,
            "resident_scope": "whole_kv_mincore_sample_not_per_prefix_proof",
        })

    def write_snapshot(self, case: pathlib.Path, filename: str, raw_name: str, item: dict, stderr_end: int, observed: int) -> None:
        raw_path = case / raw_name
        raw_path.write_text(json.dumps([{"id": 0, "is_processing": False, "kv_claimant": raw_claimant(item["eligible_resident_blocks"], item["swapped_blocks"], item["epoch"])}]), encoding="utf-8")
        put(case / filename, {
            "raw_path": raw_path.name,
            "raw_sha256": sha_bytes(raw_path.read_bytes()),
            "observed_monotonic_ns": observed,
            "stderr_end": stderr_end,
            "claimant": item,
        })

    def write_k1_evidence(self, case: pathlib.Path, pid: int, rows: dict) -> tuple[dict, dict]:
        cap_line, cap_fields = capability_line(True)
        before, after = resident(80), resident(64)
        observation, observation_fields = resident_observation(before, after)
        observation_fields["server_pid"] = str(pid)
        observation = PARSER.RESIDENT_OBSERVATION_MARKER + " " + " ".join(
            f"{key}={value}" for key, value in observation_fields.items()) + "\n"
        marker_line, marker_fields = offload_marker()
        timing_text, stage, call, blocks, io = timing_lines()
        stderr = cap_line + observation + marker_line + resume_line("prefetch") + resume_line("graph_gate") + timing_text
        (case / "server.stderr").write_text(stderr, encoding="utf-8")
        cap_start, cap_end = offset(stderr, PARSER.CAPABILITY_MARKER)
        observation_start, observation_end = offset(stderr, PARSER.RESIDENT_OBSERVATION_MARKER)
        marker_start, marker_end = offset(stderr, PARSER.MARKER)
        resume_start, _ = offset(stderr, PARSER.RESUME_MARKER, 0)
        _, resume_end = offset(stderr, "KV_PAGED_IO_STATS")
        put(case / "capability.json", {"offset": cap_start, "end": cap_end, "fields": cap_fields})
        pre = claimant(16, 0)
        post = claimant(0, 16)
        self.write_snapshot(case, "pre_offload_claimant.json", "pre_offload_claimant.raw.json", pre, cap_end, 400)
        self.write_snapshot(case, "post_claimant.json", "post_claimant.raw.json", post, marker_end, 500)
        cumulative = {
            "transaction_count": 1,
            "blocks": 16,
            "bytes": 131072,
            "relieved_bytes": 65536,
            "first_resident_bytes": before["resident_bytes"],
            "last_resident_bytes": after["resident_bytes"],
            "resident_drop_bytes": before["resident_bytes"] - after["resident_bytes"],
        }
        put(case / "offload.json", {
            "status": "complete",
            "scope_start": cap_end,
            "scope_end": marker_end,
            "expected_blocks": 16,
            "selected_seq_id": 0,
            "selected_claimant_epoch": 2,
            "transactions": [{
                "index": 0,
                "offset": marker_start,
                "end": marker_end,
                "fields": marker_fields,
                "resident": {"offset": observation_start, "end": observation_end, "fields": observation_fields},
                "resident_drop_bytes": 65536,
            }],
            "cumulative": cumulative,
        })
        scope = {
            "start": resume_start,
            "end": resume_end,
            "request_label": "step2",
            "request_started_monotonic_ns": rows["step2"]["started_monotonic_ns"],
            "request_finished_monotonic_ns": rows["step2"]["finished_monotonic_ns"],
        }
        put(case / "resume_scope.json", scope)
        selected_timings = {key: self.timings[key] for key in sorted({"prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_token_ms"})}
        put(case / "timing.json", {
            "schema_version": 1,
            "scope": scope,
            "stage": stage,
            "prefetch_call": call,
            "block_phases": blocks,
            "io_stats_record_count": 1,
            "io_stats": io,
            "response_timings": selected_timings,
            "derived": {
                "server_ttft_ms": 0.9,
                "server_prompt_ms": 0.9,
                "tpot_ms": 10.0,
                "queue_us": 100,
                "read_us": 100,
                "restore_us": 60,
                "commit_us": 10,
                "graph_us": 300,
                "gate_us": 260,
                "gate_overhead_us": 90,
                "residual_us": 330,
                "io_bytes": 4096,
                "io_syscalls": 2,
                "io_service_us": 100,
            },
        })
        return cumulative, cap_fields

    def write_case(self, round_no: int, run_order: int, name: str) -> None:
        metadata = self.metadata(round_no, run_order, name)
        case = self.root / "runs" / f"round_{round_no}_order_{run_order:02d}_{name}"
        case.mkdir(parents=True)
        put(case / "run.json", metadata)
        pid = 1000 + round_no * 100 + run_order
        starttime = 5000 + pid
        execution = self.write_execution(case, metadata, name, pid, starttime)
        prompt_p = list(range(1024))
        prompt_pq = prompt_p + [2000]
        rows = self.write_requests(case, prompt_p, prompt_pq, 1_000_000 * (round_no * 10 + run_order))
        _selected, workload = RUNNER.build_token_workload(prompt_p, prompt_pq, 2048, 1024)
        workload.update({
            "prefix_text_unit_count": 128,
            "prefix_text_max_units": RUNNER.MAX_PREFIX_TEXT_UNITS,
            "selected_prefix_text_sha256": sha_bytes(RUNNER.scalable_prefix_text(128).encode("utf-8")),
        })
        put(case / "workload.json", workload)
        (case / "server.stdout").write_text("", encoding="utf-8")
        stderr = ""
        cap = PARSER.NOT_APPLICABLE
        offload = PARSER.NOT_APPLICABLE
        timing = PARSER.NOT_APPLICABLE
        if name == "CURRENT_E0":
            line, cap = capability_line(False)
            stderr = line
            (case / "server.stderr").write_text(stderr, encoding="utf-8")
            begin, end = offset(stderr, PARSER.CAPABILITY_MARKER)
            put(case / "capability.json", {"offset": begin, "end": end, "fields": cap})
        elif name == "FLEXKV_RESIDENT":
            line, cap = capability_line(True)
            stderr = line
            (case / "server.stderr").write_text(stderr, encoding="utf-8")
            begin, end = offset(stderr, PARSER.CAPABILITY_MARKER)
            put(case / "capability.json", {"offset": begin, "end": end, "fields": cap})
            self.write_resident(case, pid)
        elif name == "FLEXKV_K1_SYNC":
            offload, cap = self.write_k1_evidence(case, pid, rows)
            timing = json.loads((case / "timing.json").read_text(encoding="utf-8"))["derived"]
        if name != "FLEXKV_K1_SYNC":
            boundary = len(stderr.encode("utf-8"))
            put(case / "resume_scope.json", {
                "start": boundary,
                "end": boundary,
                "request_label": "step2",
                "request_started_monotonic_ns": rows["step2"]["started_monotonic_ns"],
                "request_finished_monotonic_ns": rows["step2"]["finished_monotonic_ns"],
            })
        self.write_memory(case, name, pid, starttime)
        metrics = {
            "schema_version": 1,
            "request_label": "step2",
            "ttft_source": "http_stream_first_nonempty_content",
            "tpot_source": "response.timings.predicted_per_token_ms",
            "total_duration_source": "http_stream_wall",
            "tps_source": "n_predict/http_stream_wall",
            "ttft_ms": 0.0001,
            "tpot_ms": 10.0,
            "total_duration_ms": 0.001,
            "tps": 32_000_000.0,
            "response_timings": dict(self.timings),
        }
        put(case / "step2_metrics.json", metrics)
        backing = ({
            "path": str((case / "backing").resolve()),
            "environment_value": "backing",
            "created": True,
            "cleanup_attempted": True,
            "exists_after_cleanup": False,
            "cleanup_error": None,
        } if PARSER.baseline_case_flags(name)["backing"] else {"status": PARSER.NOT_APPLICABLE})
        process = {
            "pid": pid, "pgid": pid, "exit_code": 0, "stop_requested": True,
            "stop_signal": "TERM", "term_timed_out": False, "kill_timed_out": False,
            "residual_process": False,
        }
        sampler = {
            "started": True, "pid": pid + 10_000, "stop_requested": True,
            "stop_signal": "TERM", "exit_code": 0, "timed_out": False,
        }
        result = {
            "status": "complete",
            "case": name,
            "round": round_no,
            "run_order": run_order,
            "request_loop_started": True,
            "port": 9000 + pid,
            "ctx_size": 2048,
            "target_prefix_tokens": 1024,
            "evidence": PARSER.baseline_evidence_paths(name),
            "capability": cap,
            "workload": {key: workload[key] for key in PARSER.WORKLOAD_RESULT_FIELDS},
            "offload": offload,
            "step2_metrics": metrics,
            "http_statuses": {"step1": 200, "step2": 200},
            "timing": timing,
            "process": process,
            "memory_sampler": sampler,
            "backing": backing,
        }
        put(case / "cleanup.json", {"server": process, "memory_sampler": sampler, "backing": backing})
        put(case / "result.json", result)

    def build(self) -> None:
        for round_no, run_order, name in PARSER.BASELINE_RUN_PLAN:
            self.write_case(round_no, run_order, name)
        manifest = {
            "protocol": PARSER.BASELINE_PROTOCOL,
            "protocol_version": PARSER.BASELINE_PROTOCOL_VERSION,
            "source_marker_schema": PARSER.SOURCE_MARKER_SCHEMA,
            "timestamp_utc": "20260806T000000Z",
            "finished_timestamp_utc": "20260806T000001Z",
            "branch": "baseline-fixture",
            "head": "a" * 40,
            "dirty_status": [],
            "tracked_diff_fingerprint": "b" * 64,
            "capture_mode": "archival_clean",
            "runner": ident(RUNNER_PATH),
            "parser": ident(PARSER_PATH),
            "memory_sampler": ident(SAMPLER_PATH),
            "binary_requested": str(self.current_binary),
            "model_requested": str(self.model),
            "binary": ident(self.current_binary),
            "model": ident(self.model),
            "host": {"hostname": "fixture", "kernel": "fixture kernel", "os_release": "fixture os"},
            "cgroup": self.cgroup,
            "execution": {"dry_run": False, "allow_dirty": False, "smoke": False},
            "parameters": {
                "parallel": 1,
                "n_stream": 1,
                "id_slot": 0,
                "cache_prompt": True,
                "temperature": 0.0,
                "seed": 1,
                "step1_n_predict": 0,
                "step2_n_predict": 32,
                "step2_stream": True,
                "mincore_requested": True,
                "source_marker_schema": PARSER.SOURCE_MARKER_SCHEMA,
                "paged_block_size": 64,
                "ctx_size": 2048,
                "target_prefix_tokens": 1024,
                "governor_target_bytes": 1 << 30,
                "governor_max_blocks": 64,
                "sample_interval_seconds": 0.10,
                "rounds": 3,
                "prefix_text_sha256": sha_bytes(RUNNER.PREFIX_TEXT.encode("utf-8")),
                "query_text_sha256": sha_bytes(RUNNER.QUERY_TEXT.encode("utf-8")),
            },
            "case_names": list(PARSER.BASELINE_CASES),
            "comparison_edges": PARSER.BASELINE_COMPARISON_EDGES,
            "planned_runs": PARSER.baseline_plan(),
            "run_results": [{**self.metadata(round_no, run_order, name), "status": "complete"} for round_no, run_order, name in PARSER.BASELINE_RUN_PLAN],
            "runner_status": "run_complete",
        }
        put(self.root / "manifest.json", manifest)

    def parse(self) -> tuple[str, list[str]]:
        return PARSER.main_parse(self.root)

    def test_valid_three_round_fixture_passes(self) -> None:
        self.assertEqual(self.parse(), ("PASS", []))
        self.assertTrue((self.root / "summary.tsv").is_file())
        self.assertTrue((self.root / "comparison.json").is_file())

    def test_exact_output_difference_fails_on_b0_edge(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["response_text"] = "different"
        rows[1]["response_sha256"] = sha_bytes(b"different")
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("B0: step2 output is not Exact" in detail for detail in details))

    def test_k1_missing_timing_fails_closed(self) -> None:
        (self.root / "runs" / "round_1_order_03_FLEXKV_K1_SYNC" / "timing.json").unlink()
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("timing.json" in detail for detail in details))

    def test_resident_raw_evidence_missing_fails_closed(self) -> None:
        (self.root / "runs" / "round_1_order_02_FLEXKV_RESIDENT" / "resident_after_step1.raw.json").unlink()
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("resident" in detail and "raw file is missing" in detail for detail in details))

    def test_dirty_formal_capture_requires_explicit_opt_in(self) -> None:
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dirty_status"] = [" M scripts/run-kv-server-b0-b2.py"]
        manifest["capture_mode"] = "diagnostic_dirty"
        manifest["execution"]["allow_dirty"] = False
        put(path, manifest)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("captured dirty without explicit allow_dirty" in detail for detail in details))

    def test_dirty_diagnostic_capture_never_returns_formal_pass(self) -> None:
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dirty_status"] = [" M scripts/run-kv-server-b0-b2.py"]
        manifest["capture_mode"] = "diagnostic_dirty"
        manifest["execution"]["allow_dirty"] = True
        put(path, manifest)
        self.assertEqual(self.parse()[0], "DIAGNOSTIC")

    def test_zero_memory_samples_fail_closed(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "memory_samples.tsv"
        path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("captured zero samples" in detail for detail in details))

    def test_malformed_memory_sample_non_numeric_identity_fail_closed(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "memory_samples.tsv"
        path.write_text(
            "elapsed_ms\tpid\tstarttime_ticks\tvmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes\n"
            "0\tNA\t5000\t100\t150\tNA\tNA\tNA\n",
            encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("lacks numeric process identity/RSS" in detail for detail in details))

    def test_malformed_memory_sample_wrong_starttime_fail_closed(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "memory_samples.tsv"
        path.write_text(
            "elapsed_ms\tpid\tstarttime_ticks\tvmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes\n"
            "0\t6001\t99999\t100\t150\tNA\tNA\tNA\n",
            encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("does not bind the server process identity" in detail for detail in details))

    def test_manifest_extra_field_fails_closed(self) -> None:
        path = self.root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["unexpected"] = True
        put(path, manifest)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("extra=['unexpected']" in detail for detail in details))

    def test_memory_sample_missing_column_fails_closed(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "memory_samples.tsv"
        path.write_text("\t".join(SAMPLE_HEADER) + "\n0\t1\t2\t100\t150\tNA\tNA\n", encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("has 7 columns; expected 8" in detail for detail in details))

    def test_memory_sample_extra_column_fails_closed(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "memory_samples.tsv"
        path.write_text("\t".join(SAMPLE_HEADER) + "\n0\t1\t2\t100\t150\tNA\tNA\tNA\textra\n", encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("has 9 columns; expected 8" in detail for detail in details))

    def test_memory_sample_none_sentinel_fails_closed(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "memory_samples.tsv"
        path.write_text("\t".join(SAMPLE_HEADER) + "\n0\t1\t2\t100\t150\tNA\tNone\tNA\n", encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("invalid sentinel" in detail for detail in details))

    def test_two_failed_runs_are_all_reported_with_full_identity(self) -> None:
        for round_no, order, name in PARSER.BASELINE_RUN_PLAN[:2]:
            path = self.root / "runs" / f"round_{round_no}_order_{order:02d}_{name}" / "result.json"
            result = json.loads(path.read_text(encoding="utf-8"))
            result.update({"status": "request_failed", "workload_error": f"failure-{name}"})
            put(path, result)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("round=1 order=1 case=CURRENT_E0" in detail for detail in details))
        self.assertTrue(any("round=1 order=2 case=FLEXKV_RESIDENT" in detail for detail in details))

    def test_server_exit_code_must_bind_active_stop(self) -> None:
        path = self.root / "runs" / "round_1_order_01_CURRENT_E0" / "cleanup.json"
        cleanup = json.loads(path.read_text(encoding="utf-8"))
        cleanup["server"]["stop_requested"] = False
        cleanup["server"]["stop_signal"] = PARSER.NOT_APPLICABLE
        cleanup["server"]["exit_code"] = -15
        put(path, cleanup)
        result_path = path.parent / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["process"] = cleanup["server"]
        put(result_path, result)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("exit_code binding" in detail for detail in details))


class RealSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="kv-real-sampler-"))
        self.processes: list[subprocess.Popen[bytes]] = []

    def tearDown(self) -> None:
        for proc in self.processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def target(self, duration: float = 2.0) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen([
            sys.executable, "-c", f"import time; time.sleep({duration})",
        ])
        self.processes.append(proc)
        return proc

    def sampler(self, mode: str, pid: int, output: pathlib.Path, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
        return subprocess.Popen([
            "bash", str(SAMPLER_PATH), mode, str(pid), str(output), "", "0.01", "",
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    @staticmethod
    def sampler_result(sampler: subprocess.Popen[bytes], timeout: float = 3.0) -> tuple[int, str]:
        _stdout, stderr = sampler.communicate(timeout=timeout)
        return sampler.returncode, stderr.decode("utf-8", errors="replace")

    def test_direct_pid_binds_fixed_identity_and_finishes_after_exit(self) -> None:
        target = self.target()
        starttime = proc_starttime(target.pid)
        output = self.tmp / "direct.tsv"
        sampler = self.sampler("--sample-process", target.pid, output)
        time.sleep(0.05)
        target.terminate()
        target.wait(timeout=2)
        sampler_rc, sampler_stderr = self.sampler_result(sampler, 2)
        self.assertEqual(sampler_rc, 0, sampler_stderr)
        rows = read_sample_rows(output)
        self.assertTrue(rows)
        self.assertTrue(all(row[1] == str(target.pid) and row[2] == str(starttime) for row in rows))

    def test_wrapper_child_is_bound_once(self) -> None:
        wrapper = subprocess.Popen([
            "timeout", "0.35", sys.executable, "-c", "import time; time.sleep(5)",
        ])
        self.processes.append(wrapper)
        output = self.tmp / "wrapper.tsv"
        sampler = self.sampler("--sample-wrapper", wrapper.pid, output)
        sampler_rc, sampler_stderr = self.sampler_result(sampler)
        self.assertEqual(sampler_rc, 0, sampler_stderr)
        wrapper.wait(timeout=3)
        rows = read_sample_rows(output)
        self.assertTrue(rows)
        self.assertNotEqual(rows[0][1], str(wrapper.pid))
        self.assertTrue(all(row[1] == rows[0][1] and row[2] == rows[0][2] for row in rows))

    def test_unwaited_zombie_is_a_normal_end(self) -> None:
        child_path = self.tmp / "child.pid"
        script = (
            "import os, pathlib, time\n"
            "pid = os.fork()\n"
            f"if pid != 0: pathlib.Path({str(child_path)!r}).write_text(str(pid))\n"
            "if pid == 0:\n"
            "    time.sleep(0.15); os._exit(0)\n"
            "time.sleep(1.5)\n"
        )
        parent = subprocess.Popen([sys.executable, "-c", script])
        self.processes.append(parent)
        wait_for_path(child_path)
        child_pid = int(child_path.read_text(encoding="utf-8"))
        output = self.tmp / "zombie.tsv"
        sampler = self.sampler("--sample-process", child_pid, output)
        sampler_rc, sampler_stderr = self.sampler_result(sampler)
        self.assertEqual(sampler_rc, 0, sampler_stderr)
        self.assertEqual(proc_state(child_pid), "Z")
        parent.terminate()
        parent.wait(timeout=2)

    def test_active_sampler_stop_returns_zero_without_stopping_target(self) -> None:
        target = self.target(3.0)
        output = self.tmp / "stop.tsv"
        sampler = self.sampler("--sample-process", target.pid, output)
        time.sleep(0.05)
        sampler.terminate()
        sampler_rc, sampler_stderr = self.sampler_result(sampler, 2)
        self.assertEqual(sampler_rc, 0, sampler_stderr)
        self.assertIsNone(target.poll())

    def test_live_read_failure_is_not_treated_as_process_exit(self) -> None:
        target = self.target(3.0)
        fake_bin = self.tmp / "fake-bin"
        fake_bin.mkdir()
        fake_awk = fake_bin / "awk"
        fake_awk.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake_awk.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        output = self.tmp / "read-failure.tsv"
        sampler = self.sampler("--sample-process", target.pid, output, env)
        sampler_rc, sampler_stderr = self.sampler_result(sampler, 2)
        self.assertEqual(sampler_rc, 8)
        self.assertIn("live process RSS read failed", sampler_stderr)

    def test_sleep_failure_propagates(self) -> None:
        target = self.target(3.0)
        fake_bin = self.tmp / "fake-sleep-bin"
        fake_bin.mkdir()
        fake_sleep = fake_bin / "sleep"
        fake_sleep.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake_sleep.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        output = self.tmp / "sleep-failure.tsv"
        sampler = self.sampler("--sample-process", target.pid, output, env)
        sampler_rc, sampler_stderr = self.sampler_result(sampler, 2)
        self.assertEqual(sampler_rc, 12)
        self.assertIn("sleep failed", sampler_stderr)

    def test_starttime_change_fails_closed(self) -> None:
        target = self.target(3.0)
        output = self.tmp / "starttime.tsv"
        starttime = proc_starttime(target.pid)
        shell = f"""
source {str(SAMPLER_PATH)!r}
calls=0
kv_controlled_read_proc_stat() {{
    calls=$((calls + 1))
    KV_CONTROLLED_PROC_STATE=R
    if (( calls == 1 )); then
        KV_CONTROLLED_PROC_STARTTIME={starttime}
    else
        KV_CONTROLLED_PROC_STARTTIME=$(( {starttime} + 1 ))
    fi
}}
kv_controlled_sample_bound_process {target.pid} {starttime} {str(output)!r} '' 0.01 ''
exit $?
"""
        completed = subprocess.run(["bash", "-c", shell], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 9, completed.stderr)
        self.assertIn("starttime changed", completed.stderr)

    def test_output_write_failure_propagates(self) -> None:
        target = self.target()
        output_dir = self.tmp / "output-dir"
        output_dir.mkdir()
        sampler = self.sampler("--sample-process", target.pid, output_dir)
        sampler_rc, sampler_stderr = self.sampler_result(sampler, 2)
        self.assertEqual(sampler_rc, 10)
        self.assertIn("cannot write output header", sampler_stderr)

    def test_row_write_failure_propagates(self) -> None:
        target = self.target(3.0)
        fake_bin = self.tmp / "fake-awk-bin"
        fake_bin.mkdir()
        fake_awk = fake_bin / "awk"
        real_awk = shutil.which("awk")
        self.assertIsNotNone(real_awk)
        fake_awk.write_text(
            f"#!/bin/sh\n/usr/bin/sleep 0.2\nexec {real_awk} \"$@\"\n",
            encoding="utf-8")
        fake_awk.chmod(0o755)
        env = dict(os.environ)
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        output = self.tmp / "row-failure.tsv"
        sampler = self.sampler("--sample-process", target.pid, output, env)
        wait_for_path(output)
        output.unlink()
        output.mkdir()
        sampler_rc, sampler_stderr = self.sampler_result(sampler, 2)
        self.assertEqual(sampler_rc, 11)
        self.assertIn("cannot write sample row", sampler_stderr)

    def test_one_hundred_short_lifetimes_leave_no_sampler_failure(self) -> None:
        for index in range(100):
            target = self.target(0.2)
            output = self.tmp / f"short-{index}.tsv"
            sampler = self.sampler("--sample-process", target.pid, output)
            deadline = time.monotonic() + 1.0
            while not output.exists() and sampler.poll() is None and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(output.exists(), f"iteration {index}: sampler did not bind")
            target.terminate()
            target.wait(timeout=2)
            sampler_rc, sampler_stderr = self.sampler_result(sampler, 2)
            self.assertEqual(sampler_rc, 0, f"iteration {index}: {sampler_stderr}")

    def test_runner_stop_reaps_process_group(self) -> None:
        case = self.tmp / "runner-case"
        case.mkdir()
        command = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
            "time.sleep(30)"
        )
        proc = RUNNER.start(case, [sys.executable, "-c", command], dict(os.environ))
        cleanup = RUNNER.stop(proc)
        self.assertTrue(cleanup["stop_requested"])
        self.assertEqual(cleanup["stop_signal"], "TERM")
        self.assertFalse(cleanup["residual_process"])
        self.assertFalse(cleanup["term_timed_out"])
        self.assertFalse(cleanup["kill_timed_out"])
        self.assertEqual(cleanup["exit_code"], -15)
        members = []
        for entry in pathlib.Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if os.getpgid(int(entry.name)) == cleanup["pgid"]:
                    members.append(int(entry.name))
            except (OSError, ProcessLookupError):
                pass
        self.assertEqual(members, [])


class RunnerContractTest(unittest.TestCase):
    def test_fixed_three_round_plan_and_b0_b1_edges(self) -> None:
        self.assertEqual(len(RUNNER.BASELINE_RUN_PLAN), 9)
        self.assertEqual(
            [RUNNER.baseline_run_metadata(*item) for item in RUNNER.BASELINE_RUN_PLAN],
            PARSER.baseline_plan())
        self.assertEqual(RUNNER.BASELINE_COMPARISON_EDGES, PARSER.BASELINE_COMPARISON_EDGES)
        for round_no in range(1, 4):
            cases = [name for current_round, _order, name in RUNNER.BASELINE_RUN_PLAN if current_round == round_no]
            self.assertEqual(set(cases), set(RUNNER.BASELINE_CASES))

    def test_current_e0_explicitly_closes_every_experimental_switch(self) -> None:
        env = RUNNER.baseline_env("CURRENT_E0")
        self.assertEqual(
            {key: env[key] for key in RUNNER.KV_EXPERIMENTAL_DISABLED_ENV},
            RUNNER.KV_EXPERIMENTAL_DISABLED_ENV)
        self.assertEqual(env["LLAMA_KV_PAGED"], "0")
        self.assertEqual(env["LLAMA_KV_PRESSURE_UNIFIED_ACTION"], "0")
        self.assertEqual(env["LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE"], "0")
        self.assertEqual(env["LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED"], "0")

    def test_k1_sync_is_not_legacy_e5_delayed_prefetch(self) -> None:
        resident = RUNNER.baseline_env("FLEXKV_RESIDENT")
        k1 = RUNNER.baseline_env("FLEXKV_K1_SYNC")
        self.assertEqual(resident["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"], "preflight")
        self.assertEqual(k1["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"], "1")
        self.assertEqual(resident["LLAMA_KV_PRESSURE_UNIFIED_ACTION"], "0")
        self.assertEqual(k1["LLAMA_KV_PRESSURE_UNIFIED_ACTION"], "1")
        self.assertEqual(k1["LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE"], "0")
        self.assertEqual(k1["LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED"], "0")
        self.assertEqual(k1["LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME"], "0")


if __name__ == "__main__":
    unittest.main()
