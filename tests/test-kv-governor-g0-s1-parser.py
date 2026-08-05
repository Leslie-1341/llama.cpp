#!/usr/bin/env python3
"""Fail-closed fixtures for the G0-S1 single-session roundtrip parser."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "scripts/parse-kv-governor-g0-s1.py"
RUNNER_PATH = ROOT / "scripts/run-kv-governor-g0-s1.py"


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PARSER = load_module(PARSER_PATH, "kv_governor_g0_s1_parser")
RUNNER = load_module(RUNNER_PATH, "kv_governor_g0_s1_runner")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def ident(path: pathlib.Path) -> dict:
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha_bytes(path.read_bytes())}


def put(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def row(label: str, prompt: list[int], n_predict: int, text: str, start: int, finish: int) -> dict:
    request = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "seed": 1,
        "cache_prompt": True,
        "id_slot": 0,
        "stream": False,
    }
    return {
        "label": label,
        "request": request,
        "request_sha256": sha_bytes(json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")),
        "http_status": 200,
        "response_text": text,
        "response_sha256": sha_bytes(text.encode("utf-8")),
        "started_monotonic_ns": start,
        "finished_monotonic_ns": finish,
    }


def claimant(eligible: int, swapped: int, epoch: int = 1) -> dict:
    return {
        "seq_id": 0,
        "epoch": epoch,
        "active": False,
        "exhausted": False,
        "valid": True,
        "target_blocks": eligible + swapped,
        "eligible_resident_blocks": eligible,
        "swapped_blocks": swapped,
        "shared_blocks": 0,
        "blocked_blocks": 0,
    }


def raw_claimant(value: dict) -> dict:
    return {key: value[key] for key in PARSER.RAW_CLAIMANT_FIELDS}


def resident(resident_pages: int, object_id: int = 7, generation: int = 3) -> dict:
    page_size = 4096
    return {
        "status": "available",
        "source": "paged_sample_mincore",
        "object_id": object_id,
        "generation": generation,
        "page_size": page_size,
        "total_bytes": 16 * page_size,
        "resident_bytes": resident_pages * page_size,
        "total_pages": 16,
        "resident_pages": resident_pages,
    }


def resident_observation(
        decision: int,
        transaction: int,
        server_pid: int,
        before: dict,
        after: dict) -> str:
    fields = {
        "source": "paged_sample_mincore",
        "action": "offload",
        "decision_id": str(decision),
        "seq_id": "0",
        "transaction_id": str(transaction),
        "server_pid": str(server_pid),
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
        f"{key}={value}" for key, value in fields.items()) + "\n"


def marker(
        sample: int,
        decision: int,
        *,
        release: int = 0,
        offload: int = 0,
        armed_before: int = 0,
        armed_after: int = 0,
        transaction: int = 0,
        blocks: int = 0,
        byte_count: int = 0,
        relief: int = 0,
        outcome: str = "no_op",
        reason: str = "no_candidate",
        state_changed: int = 0,
        seq: int = -1,
        epoch: int = 0,
        claim: dict | None = None) -> str:
    claim = claimant(4, 0) if claim is None else claim
    claimants = ":".join(str(value) for value in (
        claim["seq_id"], claim["epoch"], int(claim["active"]), int(claim["exhausted"]),
        int(claim["valid"]), claim["target_blocks"], claim["eligible_resident_blocks"],
        claim["swapped_blocks"], claim["shared_blocks"], claim["blocked_blocks"],
    ))
    fields = {
        "state": "CRITICAL",
        "source": "rss",
        "stale": "0",
        "decision_id": str(decision),
        "episode": "1",
        "target_bytes": "1000",
        "max_blocks": "64",
        "observed_excess_bytes": "1000",
        "debt_before_bytes": "1000",
        "debt_after_bytes": str(1000 - relief),
        "offload_armed_before": str(armed_before),
        "offload_armed_after": str(armed_after),
        "next_action_sample": str(sample + 1),
        "evaluate_attempted": "1",
        "evaluate_outcome": "completed",
        "evaluate_reason": "none",
        "release_attempted": str(release),
        "offload_attempted": str(offload),
        "selected_seq_id": str(seq),
        "selected_claimant_epoch": str(epoch),
        "transaction_id": str(transaction),
        "outcome": outcome,
        "reason": reason,
        "blocks": str(blocks),
        "bytes": str(byte_count),
        "relieved_bytes": str(relief),
        "shortfall_bytes": str(1000 - byte_count),
        "io_failure": "0",
        "io_errno": "0",
        "state_changed": str(state_changed),
        "decision_reason": "offload_submitted" if offload else "release_submitted",
        "sample_count": str(sample),
        "idle": "1",
        "claimants": claimants,
        "scores": f"{seq}:1:none:100:10:20:90:0:20:0" if seq >= 0 else "none",
    }
    return "kv_pressure_unified_action " + " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def capability_line() -> str:
    fields = {
        "n_slots": "1",
        "n_seq_max": "1",
        "n_stream": "1",
        "kv_unified": "1",
        "paged_metadata": "1",
        "ingraph_gather": "1",
        "release_supported": "1",
        "offload_supported": "1",
        "prefetch_supported": "1",
        "backing_ready": "1",
        "swap_explicit_only": "1",
    }
    return "KV_GOVERNOR_CAPABILITY " + " ".join(f"{key}={value}" for key, value in fields.items()) + "\n"


def resume_line(phase: str) -> str:
    return (
        f"kv_resume_order_event phase={phase} decision_id=4 seq_id=0 claimant_epoch=2 "
        "transaction_id=2 action=prefetch outcome=completed reason=none graph_allowed=1\n"
    )


def resume_timing_line() -> str:
    return (
        "kv_resume_stage_timing decision_id=4 seq_id=0 transaction_id=2 "
        "restored_blocks=2 restored_bytes=4096 queue_us=100 gate_us=260 "
        "graph_us=300 total_us=900\n"
    )


def prefetch_block_line(index: int, physical_block: int) -> str:
    return (
        f"KV_PAGED_PREFETCH_BLOCK_PHASE call=1 block_index={index} physical_block={physical_block} "
        "validate_us=10 read_us=50 unpack_us=20 commit_us=5 phase_sum_us=85\n"
    )


def prefetch_call_line() -> str:
    return "KV_PAGED_PREFETCH_PHASE_CALL call=1 seq_id=0 requested_blocks=0 restored_blocks=2 phase_events=2\n"


def io_stats_line(active: bool = True) -> str:
    values = {
        "block_swap_out_calls": 2 if active else 0,
        "block_swap_in_calls": 2 if active else 0,
        "backing_read_syscalls": 2 if active else 0,
        "backing_write_syscalls": 2 if active else 0,
        "bytes_read": 4096 if active else 0,
        "bytes_written": 4096 if active else 0,
        "avg_block_swap_in_latency_us": 85 if active else 0,
        "max_block_swap_in_latency_us": 90 if active else 0,
        "block_in_validate_us": 20 if active else 0,
        "block_in_read_us": 100 if active else 0,
        "block_in_unpack_us": 40 if active else 0,
        "block_in_commit_us": 10 if active else 0,
    }
    return "KV_PAGED_IO_STATS " + " ".join(f"{key}={value}" for key, value in values.items()) + "\n"


def offsets(text: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    from_at = 0
    for _ in range(occurrence + 1):
        start = text.index(needle, from_at)
        from_at = start + len(needle)
    end = text.index("\n", start) + 1
    return len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))


class RunnerTimingCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = pathlib.Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.case)

    def write_valid(self) -> dict:
        prelude = io_stats_line(False)
        body = resume_timing_line() + prefetch_block_line(0, 3) + prefetch_block_line(1, 4) + prefetch_call_line()
        stderr = prelude + body + io_stats_line()
        (self.case / "server.stderr").write_text(stderr, encoding="utf-8")
        put(self.case / "resume_scope.json", {
            "start": len(prelude.encode("utf-8")),
            "end": len((prelude + body).encode("utf-8")),
            "request_label": "step2",
            "request_started_monotonic_ns": 100,
            "request_finished_monotonic_ns": 200,
        })
        return {
            "response_timings": {
                "prompt_n": 1,
                "prompt_ms": 0.9,
                "predicted_n": 32,
                "predicted_ms": 320.0,
                "predicted_per_token_ms": 10.0,
            },
        }

    def test_capture_closes_resume_stage_decomposition(self) -> None:
        value = RUNNER.capture_resume_timing(self.case, self.write_valid())
        self.assertEqual(value["derived"]["read_us"], 100)
        self.assertEqual(value["derived"]["restore_us"], 60)
        self.assertEqual(value["derived"]["commit_us"], 10)
        self.assertEqual(value["derived"]["graph_us"], 300)
        self.assertEqual(value["derived"]["gate_overhead_us"], 90)
        self.assertEqual(value["derived"]["residual_us"], 330)
        self.assertEqual(value["derived"]["io_bytes"], 4096)
        self.assertEqual(value["derived"]["io_syscalls"], 2)
        self.assertEqual(value["io_stats_record_count"], 2)
        self.assertEqual(value["derived"]["tpot_ms"], 10.0)
        self.assertTrue((self.case / "timing.json").is_file())

    def test_capture_rejects_duplicate_stage_marker(self) -> None:
        step2 = self.write_valid()
        path = self.case / "server.stderr"
        scope = json.loads((self.case / "resume_scope.json").read_text(encoding="utf-8"))
        text = path.read_text(encoding="utf-8")
        path.write_text(text[:scope["start"]] + resume_timing_line() + text[scope["start"]:], encoding="utf-8")
        scope["end"] += len(resume_timing_line().encode("utf-8"))
        put(self.case / "resume_scope.json", scope)
        with self.assertRaises(RUNNER.WorkloadFailure):
            RUNNER.capture_resume_timing(self.case, step2)


class RawClaimantSchemaTest(unittest.TestCase):
    def test_production_slot_fields_normalize_to_snapshot_claimant(self) -> None:
        expected = claimant(4, 0)
        for active in (False, True):
            errors: list[str] = []
            actual = PARSER.validate_raw_claimant({
                "id": 0,
                "is_processing": active,
                "kv_claimant": raw_claimant(expected),
            }, "snapshot raw", errors)
            self.assertEqual(errors, [])
            self.assertEqual(actual, {**expected, "active": active})

    def test_raw_claimant_keeps_exact_fields_and_types(self) -> None:
        value = raw_claimant(claimant(4, 0))
        invalid_slots = {
            "missing nested field": {
                "id": 0, "is_processing": False,
                "kv_claimant": {key: item for key, item in value.items() if key != "blocked_blocks"},
            },
            "derived field in nested object": {
                "id": 0, "is_processing": False, "kv_claimant": {**value, "active": False},
            },
            "boolean block count": {
                "id": 0, "is_processing": False, "kv_claimant": {**value, "eligible_resident_blocks": True},
            },
            "boolean slot id": {
                "id": False, "is_processing": False, "kv_claimant": value,
            },
            "integer active state": {
                "id": 0, "is_processing": 0, "kv_claimant": value,
            },
        }
        for label, slot in invalid_slots.items():
            with self.subTest(label=label):
                errors: list[str] = []
                self.assertIsNone(PARSER.validate_raw_claimant(slot, "snapshot raw", errors))
                self.assertTrue(errors)


class ParserFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.build_valid()

    def tearDown(self) -> None:
        shutil.rmtree(self.root)

    def environment(self, enabled: bool) -> dict[str, str]:
        env = {
            "HOME": "/tmp",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": "/bin",
            "LLAMA_KV_PAGED": "1",
            "LLAMA_KV_PAGED_INGRAPH": "1",
            "LLAMA_KV_PAGED_SWAP": "1",
            "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
            "LLAMA_KV_PAGED_MINCORE": "1",
            "LLAMA_KV_PAGED_BLOCK_SIZE": "64",
            "LLAMA_KV_SWAP_DIR": "backing",
            "LLAMA_KV_PRESSURE_SAMPLER": "1",
            "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "1",
            "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "1",
            "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "100",
            "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
            "LLAMA_KV_LOW_WATER_RSS_KB": "1",
            "LLAMA_KV_PRESSURE_RSS_KB": "2",
            "LLAMA_KV_CRITICAL_RSS_KB": "3",
        }
        if enabled:
            env.update({
                "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
                "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "1073741824",
                "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": "64",
            })
        return env

    def server_identity(self, pid: int, cmdline: list[str]) -> dict:
        return {
            "pid": pid,
            "starttime_ticks": 1000 + pid,
            "cmdline": cmdline,
            "cmdline_sha256": sha_bytes(b"\0".join(item.encode("utf-8") for item in cmdline)),
        }

    def write_case(self, name: str, enabled: bool, text: str, pid: int) -> None:
        case = self.root / name
        case.mkdir()
        env = self.environment(enabled)
        argv = [
            str(PARSER_PATH), "--host", "127.0.0.1", "--port", str(8000 + pid),
            "--model", str(PARSER_PATH), "--ctx-size", "2048", "--parallel", "1",
            "--kv-unified", "--no-cache-idle-slots", "--timeout", "300", "--threads", "4",
            "--n-gpu-layers", "0", "--cache-type-k", "f32", "--cache-type-v", "f32", "--no-warmup",
        ]
        server_identity = self.server_identity(pid, argv)
        put(case / "environment.json", env)
        put(case / "execution.json", {
            "argv": argv,
            "cwd": str(case.resolve()),
            "environment": env,
            "binary": ident(PARSER_PATH),
            "model": ident(PARSER_PATH),
            "server_identity": server_identity,
        })
        (case / "server.stdout").write_text("", encoding="utf-8")
        if not enabled:
            stderr = capability_line()
            (case / "server.stderr").write_text(stderr, encoding="utf-8")
            capability_start, capability_end = offsets(stderr, "KV_GOVERNOR_CAPABILITY")
            put(case / "capability.json", {
                "offset": capability_start,
                "end": capability_end,
                "fields": {part.split("=", 1)[0]: part.split("=", 1)[1] for part in capability_line().split()[1:]},
            })
        else:
            candidate = claimant(17, 0, epoch=2)
            post = claimant(0, 17, epoch=2)
            release = marker(
                1, 1, release=1, transaction=1, blocks=2, byte_count=100, relief=100,
                outcome="completed", reason="target_shortfall", state_changed=1)
            arm = marker(2, 2, release=1, armed_after=1, claim=candidate)
            observation = resident_observation(3, 2, pid, resident(12), resident(11))
            offload = marker(
                3, 3, offload=1, armed_before=1, armed_after=1, transaction=2,
                blocks=17, byte_count=900, relief=900, outcome="completed", reason="target_shortfall",
                state_changed=1, seq=0, epoch=2, claim=candidate)
            noop_observation = resident_observation(4, 0, pid, resident(11), resident(11))
            noop = marker(
                4, 4, offload=1, armed_before=1, armed_after=1, transaction=0,
                seq=0, epoch=2, claim=post)
            stderr = (
                capability_line() + release + arm + observation + offload +
                noop_observation + noop + resume_line("prefetch") + resume_line("graph_gate"))
            (case / "server.stderr").write_text(stderr, encoding="utf-8")
            cap_start, cap_end = offsets(stderr, "KV_GOVERNOR_CAPABILITY")
            observation_start, observation_end = offsets(stderr, PARSER.RESIDENT_OBSERVATION_MARKER)
            offload_start, offload_end = offsets(stderr, "kv_pressure_unified_action", 2)
            _, noop_end = offsets(stderr, "kv_pressure_unified_action", 3)
            resume_start, _ = offsets(stderr, "kv_resume_order_event", 0)
            _, resume_end = offsets(stderr, "kv_resume_order_event", 1)
            fields = lambda line: {part.split("=", 1)[0]: part.split("=", 1)[1] for part in line.split()[1:]}
            put(case / "capability.json", {"offset": cap_start, "end": cap_end, "fields": fields(capability_line())})
            put(case / "offload.json", {"offset": offload_start, "end": offload_end, "fields": fields(offload)})
            put(case / "resident.json", {
                "offset": observation_start,
                "end": observation_end,
                "fields": fields(observation),
            })
            post_slots = [{"id": 0, "is_processing": False, "kv_claimant": raw_claimant(post)}]
            raw_path = case / "post_claimant.raw.json"
            raw_path.write_text(json.dumps(post_slots), encoding="utf-8")
            put(case / "post_claimant.json", {
                "raw_path": raw_path.name,
                "raw_sha256": sha_bytes(raw_path.read_bytes()),
                "observed_monotonic_ns": 400,
                "stderr_end": noop_end,
                "claimant": post,
            })
            put(case / "resume_scope.json", {
                "start": resume_start,
                "end": resume_end,
                "request_label": "step2",
                "request_started_monotonic_ns": 500,
                "request_finished_monotonic_ns": 600,
            })
        prompt_p = list(range(1088))
        rows = [
            row("step1", prompt_p, 0, "", 100, 150),
            row("step2", prompt_p + [2000], 32, text, 500, 600),
        ]
        (case / "requests.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows), encoding="utf-8")
        put(case / "result.json", {"status": "complete", "request_loop_started": True})
        put(case / "cleanup.json", {
            "server": {"pid": pid, "pgid": pid, "exit_code": 0, "term_timed_out": False, "kill_timed_out": False, "residual_process": False},
            "backing": {
                "path": str((case / "backing").resolve()), "environment_value": "backing",
                "created": True, "cleanup_attempted": True, "exists_after_cleanup": False, "cleanup_error": None,
            },
        })

    def build_valid(self) -> None:
        self.write_case("OFF", False, "deterministic continuation", 111)
        self.write_case("GOVERNOR_ON", True, "deterministic continuation", 222)
        put(self.root / "manifest.json", {
            "protocol": "kv_governor_g0_s1",
            "protocol_version": 1,
            "source_marker_schema": "kv_governor_stage3c_1c_2b_1r/v6",
            "timestamp_utc": "20260804T000000Z",
            "finished_timestamp_utc": "20260804T000001Z",
            "branch": "fix/kv-p0-b1-bounded-store",
            "head": "a" * 40,
            "dirty_status": [],
            "tracked_diff_fingerprint": "b" * 64,
            "capture_mode": "archival_clean",
            "runner": ident(RUNNER_PATH),
            "parser": ident(PARSER_PATH),
            "binary": ident(PARSER_PATH),
            "model": ident(PARSER_PATH),
            "parameters": {
                "parallel": 1,
                "n_stream": 1,
                "id_slot": 0,
                "cache_prompt": True,
                "temperature": 0.0,
                "seed": 1,
                "step1_n_predict": 0,
                "step2_n_predict": 32,
                "mincore_requested": True,
                "source_marker_schema": "kv_governor_stage3c_1c_2b_1r/v6",
                "paged_block_size": 64,
                "ctx_size": 2048,
                "governor_target_bytes": 1073741824,
                "governor_max_blocks": 64,
                "prefix_text_sha256": "c" * 64,
                "query_text_sha256": "d" * 64,
            },
            "case_names": ["OFF", "GOVERNOR_ON"],
            "runner_status": "run_complete",
        })

    def parse(self) -> tuple[str, list[str]]:
        return PARSER.main_parse(self.root)

    def write_on_stderr(self, text: str) -> None:
        (self.root / "GOVERNOR_ON" / "server.stderr").write_text(text, encoding="utf-8")

    def update_resident_observation(self, **updates: str) -> None:
        case = self.root / "GOVERNOR_ON"
        evidence_path = case / "resident.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        fields = evidence["fields"]
        fields.update(updates)
        log_path = case / "server.stderr"
        log = log_path.read_text(encoding="utf-8")
        old_line = next(line for line in log.splitlines(True) if PARSER.RESIDENT_OBSERVATION_MARKER in line)
        new_line = PARSER.RESIDENT_OBSERVATION_MARKER + " " + " ".join(
            f"{key}={value}" for key, value in fields.items()) + "\n"
        self.assertEqual(len(old_line.encode("utf-8")), len(new_line.encode("utf-8")))
        log_path.write_text(log.replace(old_line, new_line), encoding="utf-8")
        put(evidence_path, evidence)

    def test_17_block_roundtrip_ignores_transaction_zero_noop(self) -> None:
        self.assertEqual(self.parse(), ("PASS", []))

    def test_complete_unbound_physical_artifact_fails(self) -> None:
        path = self.root / "GOVERNOR_ON" / "resident.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["offset"] += 1
        put(path, evidence)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("does not bind an exact raw observation" in detail for detail in details))

    def test_missing_marker_field_fails(self) -> None:
        log = (self.root / "GOVERNOR_ON" / "server.stderr").read_text(encoding="utf-8")
        self.write_on_stderr(log.replace(" transaction_id=1", "", 1))
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("schema mismatch" in detail for detail in details))

    def test_output_mismatch_fails(self) -> None:
        path = self.root / "GOVERNOR_ON" / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["response_text"] = "different continuation"
        rows[1]["response_sha256"] = sha_bytes(rows[1]["response_text"].encode("utf-8"))
        path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows), encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("continuation output differs" in detail for detail in details))

    def test_prefetch_must_precede_graph_gate(self) -> None:
        case = self.root / "GOVERNOR_ON"
        lines = (case / "server.stderr").read_text(encoding="utf-8").splitlines(True)
        resume_indices = [index for index, line in enumerate(lines) if "kv_resume_order_event" in line]
        lines[resume_indices[0]], lines[resume_indices[1]] = lines[resume_indices[1]], lines[resume_indices[0]]
        self.write_on_stderr("".join(lines))
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("resume events are out of order" in detail for detail in details))

    def test_missing_physical_evidence_fails(self) -> None:
        (self.root / "GOVERNOR_ON" / "resident.json").unlink()
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("resident.json" in detail for detail in details))

    def test_resident_generation_mismatch_fails(self) -> None:
        self.update_resident_observation(after_generation="4")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("one KV object/generation" in detail for detail in details))

    def test_resident_server_pid_mismatch_fails(self) -> None:
        self.update_resident_observation(server_pid="999")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("server PID" in detail for detail in details))

    def test_resident_without_byte_and_page_drop_fails(self) -> None:
        path = self.root / "GOVERNOR_ON" / "resident.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        fields = evidence["fields"]
        self.update_resident_observation(
            after_resident_bytes=fields["before_resident_bytes"],
            after_resident_pages=fields["before_resident_pages"],
        )
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("resident bytes did not decline" in detail for detail in details))

    def test_resident_unavailable_after_offload_fails(self) -> None:
        self.update_resident_observation(after_available="0")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("sampling is unavailable" in detail for detail in details))

    def test_resident_transaction_mismatch_fails(self) -> None:
        self.update_resident_observation(transaction_id="9")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("OFFLOAD transaction" in detail for detail in details))

    def test_zero_transaction_fails(self) -> None:
        case = self.root / "GOVERNOR_ON"
        offload = json.loads((case / "offload.json").read_text(encoding="utf-8"))
        offload["fields"]["transaction_id"] = "0"
        put(case / "offload.json", offload)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("does not bind" in detail or "transaction contract" in detail for detail in details))

    def test_preworkload_unsupported_passes_through(self) -> None:
        shutil.rmtree(self.root / "OFF")
        shutil.rmtree(self.root / "GOVERNOR_ON")
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        manifest.update({
            "runner_status": "UNSUPPORTED",
            "unsupported_stage": "pre_workload_physical_probe",
            "unsupported_reason": "no bindable same-KV mincore surface",
        })
        put(self.root / "manifest.json", manifest)
        self.assertEqual(self.parse(), ("UNSUPPORTED", ["no bindable same-KV mincore surface"]))

    def test_postworkload_unsupported_fails(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        manifest.update({
            "runner_status": "UNSUPPORTED",
            "unsupported_stage": "pre_workload_physical_probe",
            "unsupported_reason": "no bindable same-KV mincore surface",
        })
        put(self.root / "manifest.json", manifest)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("request evidence" in detail or "executed" in detail for detail in details))

    def test_missing_capability_fails_closed(self) -> None:
        (self.root / "GOVERNOR_ON" / "capability.json").unlink()
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("capability.json" in detail for detail in details))

    def test_missing_case_fails_closed(self) -> None:
        shutil.rmtree(self.root / "OFF")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("OFF: missing" in detail or "missing" in detail for detail in details))

    def test_invalid_identity_fails_closed(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        manifest["binary"] = {"path": "", "size": 0, "sha256": "not-a-sha"}
        put(self.root / "manifest.json", manifest)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("identity binary" in detail for detail in details))

    def test_http_failure_fails_closed(self) -> None:
        path = self.root / "GOVERNOR_ON" / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["http_status"] = 500
        path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows), encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("did not receive HTTP 200" in detail for detail in details))

    def test_upstream_failure_reports_only_first_cause(self) -> None:
        case = self.root / "GOVERNOR_ON"
        put(case / "result.json", {
            "status": "request_failed",
            "request_loop_started": True,
            "workload_error": "WorkloadFailure: first real failure",
        })
        for filename in (
                "offload.json", "resident.json", "post_claimant.json",
                "post_claimant.raw.json", "resume_scope.json"):
            (case / filename).unlink()
        self.assertEqual(
            self.parse(),
            ("FAIL", ["GOVERNOR_ON: WorkloadFailure: first real failure"]),
        )

    def test_step2_before_step1_fails(self) -> None:
        path = self.root / "GOVERNOR_ON" / "requests.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[1]["started_monotonic_ns"] = 1
        rows[1]["finished_monotonic_ns"] = 2
        path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in rows), encoding="utf-8")
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("step2 does not begin" in detail for detail in details))

    def test_server_cmdline_mismatch_fails(self) -> None:
        path = self.root / "GOVERNOR_ON" / "execution.json"
        execution = json.loads(path.read_text(encoding="utf-8"))
        execution["server_identity"]["cmdline"] = ["wrong-server"]
        execution["server_identity"]["cmdline_sha256"] = sha_bytes(b"wrong-server")
        put(path, execution)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("cmdline differs" in detail for detail in details))


class RunnerTokenWorkloadTest(unittest.TestCase):
    def test_terminal_retokenization_uses_previous_shared_block_boundary(self) -> None:
        prefix_candidate = list(range(1151)) + [13]
        continuation_prompt = list(range(1151)) + [624, 23526, 448, 825, 72349, 63594, 4226, 13]

        prompt_p, evidence = RUNNER.build_token_workload(prefix_candidate, continuation_prompt)

        self.assertEqual(len(prefix_candidate), 1152)
        self.assertEqual(len(continuation_prompt), 1159)
        self.assertEqual(len(prompt_p), 1088)
        self.assertEqual(prompt_p, continuation_prompt[:1088])
        self.assertEqual(evidence["common_prefix_token_count"], 1151)
        self.assertEqual(evidence["first_token_mismatch_index"], 1151)
        self.assertEqual(evidence["first_token_mismatch"], {
            "prefix_candidate": 13,
            "continuation_prompt": 624,
        })
        self.assertEqual(evidence["prefix_block_count"], 17)
        self.assertTrue(evidence["strict_prefix"])
        self.assertEqual(evidence["failure_reasons"], [])

    def test_context_exactly_at_limit_is_valid(self) -> None:
        prefix_candidate = list(range(1984))
        continuation_prompt = prefix_candidate + list(range(10000, 10032))

        prompt_p, evidence = RUNNER.build_token_workload(prefix_candidate, continuation_prompt)

        self.assertEqual(len(prompt_p), 1984)
        self.assertEqual(evidence["context_token_count"], 2048)
        self.assertTrue(evidence["context_fits"])
        self.assertEqual(evidence["status"], "ready")

    def test_insufficient_shared_prefix_reports_counts(self) -> None:
        prefix_candidate = list(range(200))
        continuation_prompt = list(range(100)) + [999] + list(range(1000, 1199))

        prompt_p, evidence = RUNNER.build_token_workload(prefix_candidate, continuation_prompt)

        self.assertEqual(len(prompt_p), 64)
        self.assertEqual(evidence["prefix_candidate_token_count"], 200)
        self.assertEqual(evidence["continuation_prompt_token_count"], 300)
        self.assertEqual(evidence["common_prefix_token_count"], 100)
        self.assertEqual(evidence["status"], "invalid")
        self.assertTrue(any("requires at least 128 tokens" in reason for reason in evidence["failure_reasons"]))

    def test_invalid_context_is_persisted_before_failure(self) -> None:
        prefix_candidate = list(range(1920))
        continuation_prompt = prefix_candidate + list(range(10000, 10097))
        with tempfile.TemporaryDirectory() as directory:
            case = pathlib.Path(directory)
            with mock.patch.object(RUNNER, "tokenize", side_effect=[prefix_candidate, continuation_prompt]):
                with self.assertRaisesRegex(RUNNER.WorkloadFailure, r"2017\+32=2049>2048"):
                    RUNNER.prepare_token_workload(case, 12345)
            evidence = json.loads((case / "workload.json").read_text(encoding="utf-8"))

        self.assertEqual(evidence["status"], "invalid")
        self.assertEqual(evidence["prefix_candidate_token_count"], 1920)
        self.assertEqual(evidence["continuation_prompt_token_count"], 2017)
        self.assertEqual(evidence["prefix_token_count"], 1920)
        self.assertEqual(evidence["context_token_count"], 2049)
        self.assertFalse(evidence["context_fits"])
        self.assertTrue(any("exceeds ctx_size" in reason for reason in evidence["failure_reasons"]))


class RunnerResidentObservationTest(unittest.TestCase):
    def test_runtime_claimant_snapshot_does_not_require_resident_sample(self) -> None:
        slots = [{
            "id": 0,
            "is_processing": False,
            "kv_claimant": raw_claimant(claimant(4, 0)),
        }]
        raw = json.dumps(slots).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            case = pathlib.Path(directory)
            (case / "server.stderr").write_bytes(b"")
            with mock.patch.object(RUNNER, "query_slots_raw", return_value=(200, raw, slots)):
                value = RUNNER.capture_snapshot(case, 12345, "post_claimant")
        self.assertNotIn("resident", value)
        self.assertEqual(value["claimant"], claimant(4, 0))

    def test_capture_binds_raw_transaction_record(self) -> None:
        line = resident_observation(2, 1, 222, resident(12), resident(11))
        offload = {"decision_id": "2", "selected_seq_id": "0", "transaction_id": "1"}
        with tempfile.TemporaryDirectory() as directory:
            case = pathlib.Path(directory)
            (case / "server.stderr").write_text(line, encoding="utf-8")
            value = RUNNER.capture_resident_observation(case, offload)
            saved = json.loads((case / "resident.json").read_text(encoding="utf-8"))
        self.assertEqual(value, saved)
        self.assertEqual(saved["fields"]["transaction_id"], "1")
        self.assertEqual(saved["fields"]["before_resident_pages"], "12")
        self.assertEqual(saved["fields"]["after_resident_pages"], "11")

    def test_early_17_block_offload_continues_to_step2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "llama-server"
            model = root / "model.gguf"
            binary.write_bytes(b"binary")
            model.write_bytes(b"model")
            candidate = claimant(17, 0, epoch=2)
            post = claimant(0, 17, epoch=2)
            early_log = (
                resident_observation(2, 0, 222, resident(12), resident(12)) +
                marker(2, 2, offload=1, armed_before=1, armed_after=1, seq=0, epoch=2, claim=candidate) +
                resident_observation(3, 2, 222, resident(12), resident(1)) +
                marker(
                    3, 3, offload=1, armed_before=1, armed_after=1, transaction=2,
                    blocks=17, byte_count=900, relief=900, outcome="completed",
                    reason="target_shortfall", state_changed=1, seq=0, epoch=2,
                    claim=candidate))
            post_slots = [{
                "id": 0,
                "is_processing": False,
                "kv_claimant": raw_claimant(post),
            }]
            post_raw = json.dumps(post_slots).encode("utf-8")
            proc = mock.Mock(pid=222, returncode=None)
            proc.poll.return_value = None
            labels: list[str] = []

            def fake_start(
                    _binary: pathlib.Path,
                    _model: pathlib.Path,
                    case: pathlib.Path,
                    _port: int,
                    _env: dict[str, str]):
                (case / "server.stdout").write_bytes(b"")
                (case / "server.stderr").write_bytes(b"")
                return proc

            def fake_request(
                    _port: int,
                    _body: dict,
                    label: str,
                    _record: pathlib.Path) -> dict:
                labels.append(label)
                if label == "step1":
                    (root / "GOVERNOR_ON" / "server.stderr").write_text(
                        early_log, encoding="utf-8")
                    return {
                        "http_status": 200,
                        "started_monotonic_ns": 100,
                        "finished_monotonic_ns": 200,
                    }
                return {
                    "http_status": 200,
                    "started_monotonic_ns": 300,
                    "finished_monotonic_ns": 400,
                }

            capability = {key: "1" for key in RUNNER.CAPABILITY_FIELDS}
            clean_process = {
                "pid": 222,
                "pgid": 222,
                "exit_code": 0,
                "term_timed_out": False,
                "kill_timed_out": False,
                "residual_process": False,
            }
            with (
                    mock.patch.object(RUNNER, "start", side_effect=fake_start),
                    mock.patch.object(RUNNER, "read_process_identity", return_value={
                        "pid": 222,
                        "starttime_ticks": 1,
                        "cmdline": [str(binary)],
                        "cmdline_sha256": "0" * 64,
                    }),
                    mock.patch.object(RUNNER, "wait_health", return_value=True),
                    mock.patch.object(RUNNER, "wait_capability", return_value=(
                        capability, {"offset": 0, "end": 0, "fields": capability}, None)),
                    mock.patch.object(RUNNER, "prepare_token_workload", return_value=(
                        list(range(1088)), list(range(1089)))),
                    mock.patch.object(RUNNER, "request_completion", side_effect=fake_request),
                    mock.patch.object(RUNNER, "query_slots_raw", return_value=(
                        200, post_raw, post_slots)),
                    mock.patch.object(RUNNER, "capture_resume_timing", return_value={"derived": {}}),
                    mock.patch.object(RUNNER, "stop", return_value=clean_process)):
                result = RUNNER.run_case(
                    "GOVERNOR_ON", binary, model, True, root)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(labels, ["step1", "step2"])
            self.assertFalse((root / "GOVERNOR_ON" / "pre_claimant.json").exists())
            saved = json.loads(
                (root / "GOVERNOR_ON" / "resident.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["fields"]["transaction_id"], "2")
            self.assertTrue((root / "GOVERNOR_ON" / "post_claimant.json").is_file())


class RunnerPhysicalProbeTest(unittest.TestCase):
    def test_bound_slots_resident_sample_is_available(self) -> None:
        probe = RUNNER.physical_probe(
            [{"id": 0, "is_processing": False, "kv_claimant": {
                "epoch": 1, "exhausted": False, "valid": True, "target_blocks": 4,
                "eligible_resident_blocks": 4, "swapped_blocks": 0, "shared_blocks": 0,
                "blocked_blocks": 0,
            }, "kv_resident": resident(12)}],
            b"[]",
            {"pid": 7},
        )
        self.assertTrue(probe["physical_resident_sample_available"])
        self.assertEqual(probe["resident"]["object_id"], 7)

    def test_unavailable_slots_resident_sample_is_explicit(self) -> None:
        probe = RUNNER.physical_probe(
            [{"id": 0, "is_processing": False, "kv_claimant": {
                "epoch": 1, "exhausted": False, "valid": True, "target_blocks": 4,
                "eligible_resident_blocks": 4, "swapped_blocks": 0, "shared_blocks": 0,
                "blocked_blocks": 0,
            }, "kv_resident": {"status": "unavailable"}}],
            b"[]",
            {"pid": 7},
        )
        self.assertFalse(probe["physical_resident_sample_available"])
        self.assertIn("declared", probe["reason"])

    def test_logical_slots_are_not_physical_resident_evidence(self) -> None:
        probe = RUNNER.physical_probe(
            [{"id": 0, "is_processing": False, "kv_claimant": {
                "epoch": 1, "exhausted": False, "valid": True, "target_blocks": 4,
                "eligible_resident_blocks": 4, "swapped_blocks": 0, "shared_blocks": 0,
                "blocked_blocks": 0,
            }}],
            b"[]",
            {"pid": 7},
        )
        self.assertFalse(probe["physical_resident_sample_available"])
        self.assertTrue(probe["logical_claimant_fields_present"])


if __name__ == "__main__":
    unittest.main()
