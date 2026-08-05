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


def workload_evidence(
        prompt_p: list[int],
        prompt_pq: list[int],
        ctx_size: int,
        target_prefix_tokens: int) -> dict:
    _selected, evidence = RUNNER.build_token_workload(
        prompt_p, prompt_pq, ctx_size, target_prefix_tokens)
    evidence.update({
        "prefix_text_unit_count": 128,
        "prefix_text_max_units": RUNNER.MAX_PREFIX_TEXT_UNITS,
        "selected_prefix_text_sha256": sha_bytes(
            RUNNER.scalable_prefix_text(128).encode("utf-8")),
    })
    return evidence


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


def resident(
        resident_pages: int,
        object_id: int = 7,
        generation: int = 3,
        total_pages: int = 256) -> dict:
    page_size = 4096
    return {
        "status": "available",
        "source": "paged_sample_mincore",
        "object_id": object_id,
        "generation": generation,
        "page_size": page_size,
        "total_bytes": total_pages * page_size,
        "resident_bytes": resident_pages * page_size,
        "total_pages": total_pages,
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
    target_bytes = max(1_000_000, byte_count)
    debt_before_bytes = max(2_000_000, relief)
    fields = {
        "state": "CRITICAL",
        "source": "rss",
        "stale": "0",
        "decision_id": str(decision),
        "episode": "1",
        "target_bytes": str(target_bytes),
        "max_blocks": "64",
        "observed_excess_bytes": str(debt_before_bytes),
        "debt_before_bytes": str(debt_before_bytes),
        "debt_after_bytes": str(debt_before_bytes - relief),
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
        "shortfall_bytes": str(target_bytes - byte_count),
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


def resume_line(phase: str, epoch: int = 2, transaction: int = 20) -> str:
    return (
        f"kv_resume_order_event phase={phase} decision_id=20 seq_id=0 "
        f"claimant_epoch={epoch} transaction_id={transaction} action=prefetch "
        "outcome=completed reason=none graph_allowed=1\n"
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

    def write_case(
            self,
            name: str,
            enabled: bool,
            text: str,
            pid: int,
            ctx_size: int = 2048,
            target_prefix_tokens: int = 1024,
            transaction_blocks: tuple[int, ...] = (16,)) -> None:
        case = self.root / name
        case.mkdir()
        env = self.environment(enabled)
        argv = [
            str(PARSER_PATH), "--host", "127.0.0.1", "--port", str(8000 + pid),
            "--model", str(PARSER_PATH), "--ctx-size", str(ctx_size), "--parallel", "1",
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
            expected_blocks = target_prefix_tokens // 64
            self.assertEqual(sum(transaction_blocks), expected_blocks)
            initial = claimant(expected_blocks, 0, epoch=2)
            release = marker(
                1, 1, release=1, transaction=1, blocks=2, byte_count=8192,
                relief=8192, outcome="completed", reason="target_shortfall",
                state_changed=1)
            arm = marker(2, 2, release=1, armed_after=1, claim=initial)
            parts = [capability_line(), release, arm]
            transaction_lines: list[tuple[str, str, int]] = []
            cumulative_blocks = 0
            first_resident_pages = expected_blocks + 64
            for index, blocks in enumerate(transaction_blocks):
                decision = 3 + index
                transaction = 2 + index
                candidate = claimant(
                    expected_blocks - cumulative_blocks, cumulative_blocks, epoch=2)
                before = resident(first_resident_pages - cumulative_blocks)
                after = resident(first_resident_pages - cumulative_blocks - blocks)
                observation = resident_observation(
                    decision, transaction, pid, before, after)
                offload = marker(
                    decision, decision, offload=1, armed_before=1, armed_after=1,
                    transaction=transaction, blocks=blocks,
                    byte_count=blocks * 8192, relief=blocks * 4096,
                    outcome="completed", reason="target_satisfied", state_changed=1,
                    seq=0, epoch=2, claim=candidate)
                parts.extend((observation, offload))
                transaction_lines.append((observation, offload, blocks))
                cumulative_blocks += blocks
            post = claimant(0, expected_blocks, epoch=2)
            noop_decision = 3 + len(transaction_blocks)
            final_resident = resident(first_resident_pages - expected_blocks)
            noop_observation = resident_observation(
                noop_decision, 0, pid, final_resident, final_resident)
            noop = marker(
                noop_decision, noop_decision, offload=1, armed_before=1,
                armed_after=1, transaction=0, seq=0, epoch=2, claim=post)
            parts.extend((
                noop_observation, noop, resume_line("prefetch"),
                resume_line("graph_gate")))
            stderr = "".join(parts)
            (case / "server.stderr").write_text(stderr, encoding="utf-8")
            cap_start, cap_end = offsets(stderr, "KV_GOVERNOR_CAPABILITY")
            _, noop_end = offsets(
                stderr, "kv_pressure_unified_action", 2 + len(transaction_blocks))
            resume_start, _ = offsets(stderr, "kv_resume_order_event", 0)
            _, resume_end = offsets(stderr, "kv_resume_order_event", 1)

            def fields(line: str) -> dict[str, str]:
                return {
                    part.split("=", 1)[0]: part.split("=", 1)[1]
                    for part in line.split()[1:]
                }

            put(case / "capability.json", {
                "offset": cap_start,
                "end": cap_end,
                "fields": fields(capability_line()),
            })
            transactions: list[dict] = []
            cumulative_blocks = 0
            cumulative_bytes = 0
            cumulative_relief = 0
            for index, (observation, offload, blocks) in enumerate(transaction_lines):
                observation_start, observation_end = offsets(
                    stderr, PARSER.RESIDENT_OBSERVATION_MARKER, index)
                offload_start, offload_end = offsets(
                    stderr, "kv_pressure_unified_action", 2 + index)
                observation_fields = fields(observation)
                offload_fields = fields(offload)
                resident_drop_bytes = (
                    int(observation_fields["before_resident_bytes"]) -
                    int(observation_fields["after_resident_bytes"]))
                transactions.append({
                    "index": index,
                    "offset": offload_start,
                    "end": offload_end,
                    "fields": offload_fields,
                    "resident": {
                        "offset": observation_start,
                        "end": observation_end,
                        "fields": observation_fields,
                    },
                    "resident_drop_bytes": resident_drop_bytes,
                })
                cumulative_blocks += blocks
                cumulative_bytes += int(offload_fields["bytes"])
                cumulative_relief += int(offload_fields["relieved_bytes"])
            first_resident_bytes = int(
                transactions[0]["resident"]["fields"]["before_resident_bytes"])
            last_resident_bytes = int(
                transactions[-1]["resident"]["fields"]["after_resident_bytes"])
            put(case / "offload.json", {
                "status": "complete",
                "scope_start": cap_end,
                "scope_end": transactions[-1]["end"],
                "expected_blocks": expected_blocks,
                "selected_seq_id": 0,
                "selected_claimant_epoch": 2,
                "transactions": transactions,
                "cumulative": {
                    "transaction_count": len(transactions),
                    "blocks": cumulative_blocks,
                    "bytes": cumulative_bytes,
                    "relieved_bytes": cumulative_relief,
                    "first_resident_bytes": first_resident_bytes,
                    "last_resident_bytes": last_resident_bytes,
                    "resident_drop_bytes": first_resident_bytes - last_resident_bytes,
                },
            })
            post_slots = [{
                "id": 0,
                "is_processing": False,
                "kv_claimant": raw_claimant(post),
            }]
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
        prompt_p = list(range(target_prefix_tokens))
        prompt_pq = prompt_p + [2000]
        rows = [
            row("step1", prompt_p, 0, "", 100, 150),
            row("step2", prompt_pq, 32, text, 500, 600),
        ]
        evidence = workload_evidence(
            prompt_p, prompt_pq, ctx_size, target_prefix_tokens)
        put(case / "workload.json", evidence)
        (case / "requests.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
            encoding="utf-8")
        put(case / "result.json", {
            "status": "complete",
            "request_loop_started": True,
            "ctx_size": ctx_size,
            "target_prefix_tokens": target_prefix_tokens,
            "workload": {
                key: evidence[key] for key in PARSER.WORKLOAD_RESULT_FIELDS
            },
        })
        put(case / "cleanup.json", {
            "server": {
                "pid": pid,
                "pgid": pid,
                "exit_code": 0,
                "term_timed_out": False,
                "kill_timed_out": False,
                "residual_process": False,
            },
            "backing": {
                "path": str((case / "backing").resolve()),
                "environment_value": "backing",
                "created": True,
                "cleanup_attempted": True,
                "exists_after_cleanup": False,
                "cleanup_error": None,
            },
        })

    def build_valid(
            self,
            transaction_blocks: tuple[int, ...] = (16,),
            ctx_size: int = 2048,
            target_prefix_tokens: int = 1024) -> None:
        self.write_case(
            "OFF", False, "deterministic continuation", 111,
            ctx_size, target_prefix_tokens, transaction_blocks)
        self.write_case(
            "GOVERNOR_ON", True, "deterministic continuation", 222,
            ctx_size, target_prefix_tokens, transaction_blocks)
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
                "ctx_size": ctx_size,
                "target_prefix_tokens": target_prefix_tokens,
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

    def update_resident_observation(
            self,
            transaction_index: int = 0,
            **updates: str) -> None:
        case = self.root / "GOVERNOR_ON"
        evidence_path = case / "offload.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        resident_evidence = evidence["transactions"][transaction_index]["resident"]
        fields = resident_evidence["fields"]
        transaction_id = fields["transaction_id"]
        decision_id = fields["decision_id"]
        fields.update(updates)
        log_path = case / "server.stderr"
        log = log_path.read_text(encoding="utf-8")
        old_line = next(
            line for line in log.splitlines(True)
            if PARSER.RESIDENT_OBSERVATION_MARKER in line and
            f" transaction_id={transaction_id}" in line and
            f" decision_id={decision_id}" in line)
        new_line = PARSER.RESIDENT_OBSERVATION_MARKER + " " + " ".join(
            f"{key}={value}" for key, value in fields.items()) + "\n"
        self.assertEqual(len(old_line.encode("utf-8")), len(new_line.encode("utf-8")))
        log_path.write_text(log.replace(old_line, new_line), encoding="utf-8")
        put(evidence_path, evidence)

    def test_16_block_single_transaction_ignores_transaction_zero_noop(self) -> None:
        self.assertEqual(self.parse(), ("PASS", []))

    def test_120_block_three_transaction_roundtrip(self) -> None:
        shutil.rmtree(self.root)
        self.root.mkdir()
        self.build_valid(
            transaction_blocks=(43, 43, 34),
            ctx_size=8064,
            target_prefix_tokens=7680)

        self.assertEqual(self.parse(), ("PASS", []))
        evidence = json.loads(
            (self.root / "GOVERNOR_ON" / "offload.json").read_text(
                encoding="utf-8"))
        self.assertEqual(
            [int(item["fields"]["blocks"]) for item in evidence["transactions"]],
            [43, 43, 34])
        self.assertEqual(evidence["cumulative"]["blocks"], 120)
        self.assertEqual(
            evidence["cumulative"]["resident_drop_bytes"],
            evidence["cumulative"]["relieved_bytes"])

    def test_step2_new_epoch_offload_is_excluded_from_closed_transactions(self) -> None:
        case = self.root / "GOVERNOR_ON"
        path = case / "server.stderr"
        text = path.read_text(encoding="utf-8")
        post_step_observation = resident_observation(
            99, 99, 222, resident(64), resident(63))
        post_step_offload = marker(
            99, 99, offload=1, armed_before=1, armed_after=1,
            transaction=99, blocks=1, byte_count=8192, relief=4096,
            outcome="completed", reason="target_satisfied", state_changed=1,
            seq=0, epoch=3, claim=claimant(1, 0, epoch=3))
        path.write_text(
            text + post_step_observation + post_step_offload,
            encoding="utf-8")

        self.assertEqual(self.parse(), ("PASS", []))

    def test_cumulative_resident_drop_mismatch_fails(self) -> None:
        path = self.root / "GOVERNOR_ON" / "offload.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["cumulative"]["resident_drop_bytes"] += 1
        put(path, evidence)

        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any(
            "cumulative OFFLOAD summary differs" in detail for detail in details))

    def test_complete_unbound_physical_artifact_fails(self) -> None:
        path = self.root / "GOVERNOR_ON" / "offload.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["transactions"][0]["resident"]["offset"] += 1
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
        path = self.root / "GOVERNOR_ON" / "offload.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["transactions"][0]["resident"] = None
        put(path, evidence)
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("resident evidence schema" in detail for detail in details))

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
        path = self.root / "GOVERNOR_ON" / "offload.json"
        evidence = json.loads(path.read_text(encoding="utf-8"))
        fields = evidence["transactions"][0]["resident"]["fields"]
        self.update_resident_observation(
            after_resident_bytes=fields["before_resident_bytes"],
            after_resident_pages=fields["before_resident_pages"],
        )
        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("resident bytes/pages did not decline" in detail for detail in details))

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
        offload["transactions"][0]["fields"]["transaction_id"] = "0"
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
                "offload.json", "post_claimant.json",
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

    def test_target_cannot_be_one_full_block_above_actual_prefix(self) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["parameters"]["target_prefix_tokens"] = 1152
        put(manifest_path, manifest)
        for name in PARSER.CASES:
            case = self.root / name
            workload_path = case / "workload.json"
            workload = json.loads(workload_path.read_text(encoding="utf-8"))
            workload["target_prefix_tokens"] = 1152
            workload["target_prefix_token_delta"] = 64
            workload["target_prefix_satisfied"] = True
            put(workload_path, workload)
            result_path = case / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["target_prefix_tokens"] = 1152
            result["workload"]["target_prefix_tokens"] = 1152
            result["workload"]["target_prefix_token_delta"] = 64
            put(result_path, result)

        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("differ by at least one full block" in detail for detail in details))

    def test_actual_block_count_must_match_step1_request(self) -> None:
        case = self.root / "OFF"
        workload_path = case / "workload.json"
        workload = json.loads(workload_path.read_text(encoding="utf-8"))
        workload["actual_p_block_count"] = 15
        workload["prefix_block_count"] = 15
        put(workload_path, workload)
        result_path = case / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["workload"]["actual_p_block_count"] = 15
        put(result_path, result)

        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("actual_p_block_count" in detail for detail in details))

    def test_manifest_ctx_size_must_match_real_server_argv(self) -> None:
        manifest_path = self.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["parameters"]["ctx_size"] = 4096
        put(manifest_path, manifest)
        for name in PARSER.CASES:
            case = self.root / name
            result_path = case / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["ctx_size"] = 4096
            put(result_path, result)
            workload_path = case / "workload.json"
            workload = json.loads(workload_path.read_text(encoding="utf-8"))
            workload["ctx_size"] = 4096
            put(workload_path, workload)

        status, details = self.parse()
        self.assertEqual(status, "FAIL")
        self.assertTrue(any("--ctx-size differs" in detail for detail in details))


class RunnerTokenWorkloadTest(unittest.TestCase):
    def test_terminal_retokenization_cannot_silently_drop_one_full_block(self) -> None:
        prefix_candidate = list(range(1151)) + [13]
        continuation_prompt = list(range(1151)) + [624, 23526, 448, 825, 72349, 63594, 4226, 13]

        prompt_p, evidence = RUNNER.build_token_workload(
            prefix_candidate, continuation_prompt, 2048, 1152)

        self.assertEqual(len(prompt_p), 1088)
        self.assertEqual(evidence["common_prefix_token_count"], 1151)
        self.assertEqual(evidence["target_prefix_token_delta"], 64)
        self.assertFalse(evidence["target_prefix_satisfied"])
        self.assertEqual(evidence["status"], "invalid")
        self.assertTrue(any(
            "target-actual<64" in reason for reason in evidence["failure_reasons"]))

    def test_context_exactly_at_configured_limit_is_valid(self) -> None:
        for ctx_size in RUNNER.SUPPORTED_CTX_SIZES:
            with self.subTest(ctx_size=ctx_size):
                prefix_token_count = ctx_size - 2 * RUNNER.N_PREDICT
                prefix_candidate = list(range(prefix_token_count))
                continuation_prompt = prefix_candidate + list(range(10000, 10000 + RUNNER.N_PREDICT))

                prompt_p, evidence = RUNNER.build_token_workload(
                    prefix_candidate, continuation_prompt, ctx_size, prefix_token_count)

                self.assertEqual(len(prompt_p), prefix_token_count)
                self.assertEqual(evidence["ctx_size"], ctx_size)
                self.assertEqual(evidence["context_token_count"], ctx_size)
                self.assertTrue(evidence["context_fits"])
                self.assertEqual(evidence["status"], "ready")

    def test_five_target_profiles_use_monotonic_prefixes_from_one_text_source(self) -> None:
        targets = (1024, 2048, 4096, 6144, 7680)
        selected_units: list[int] = []
        actual_tokens: list[int] = []

        def fake_tokenize(_port: int, content: str, add_special: bool) -> list[int]:
            self.assertTrue(add_special)
            has_query = content.endswith(RUNNER.QUERY_TEXT)
            prefix_text = content[:-len(RUNNER.QUERY_TEXT)] if has_query else content
            self.assertEqual(len(prefix_text) % len(RUNNER.PREFIX_TEXT_UNIT), 0)
            unit_count = len(prefix_text) // len(RUNNER.PREFIX_TEXT_UNIT)
            self.assertEqual(prefix_text, RUNNER.scalable_prefix_text(unit_count))
            prefix_tokens = list(range(unit_count * 8 + 1))
            return prefix_tokens + (list(range(1_000_000, 1_000_008)) if has_query else [])

        for target_prefix_tokens in targets:
            with self.subTest(target_prefix_tokens=target_prefix_tokens), tempfile.TemporaryDirectory() as directory:
                case = pathlib.Path(directory)
                with mock.patch.object(RUNNER, "tokenize", side_effect=fake_tokenize):
                    prompt_p, prompt_pq = RUNNER.prepare_token_workload(
                        case, 12345, 8064, target_prefix_tokens)
                evidence = json.loads((case / "workload.json").read_text(encoding="utf-8"))

                selected_units.append(evidence["prefix_text_unit_count"])
                actual_tokens.append(len(prompt_p))
                self.assertEqual(prompt_p, prompt_pq[:len(prompt_p)])
                self.assertEqual(evidence["status"], "ready")
                self.assertEqual(evidence["target_prefix_tokens"], target_prefix_tokens)
                self.assertEqual(evidence["actual_p_token_count"], len(prompt_p))
                self.assertEqual(evidence["actual_pq_token_count"], len(prompt_pq))
                self.assertEqual(evidence["actual_p_block_count"], len(prompt_p) // 64)
                self.assertGreaterEqual(target_prefix_tokens - len(prompt_p), 0)
                self.assertLess(target_prefix_tokens - len(prompt_p), 64)
                self.assertTrue(evidence["target_prefix_satisfied"])
                self.assertTrue(evidence["strict_prefix"])
                self.assertTrue(evidence["context_fits"])

        self.assertEqual(actual_tokens, list(targets))
        self.assertEqual(selected_units, sorted(selected_units))
        self.assertEqual(len(set(selected_units)), len(selected_units))

    def test_target_uses_largest_complete_block_not_exceeding_limit(self) -> None:
        prefix_candidate = list(range(3000))
        continuation_prompt = list(range(3000))

        prompt_p, evidence = RUNNER.build_token_workload(
            prefix_candidate, continuation_prompt, 8064, 2050)

        self.assertEqual(len(prompt_p), 2048)
        self.assertEqual(evidence["actual_p_token_count"], 2048)
        self.assertEqual(evidence["actual_p_block_count"], 32)
        self.assertEqual(evidence["target_prefix_token_delta"], 2)
        self.assertEqual(evidence["status"], "ready")

    def test_target_below_two_blocks_is_an_explicit_workload_failure(self) -> None:
        _prompt_p, evidence = RUNNER.build_token_workload(
            list(range(3000)), list(range(3000)), 8064, 127)

        self.assertEqual(evidence["status"], "invalid")
        self.assertEqual(evidence["target_prefix_tokens"], 127)
        self.assertEqual(evidence["actual_p_token_count"], 0)
        self.assertTrue(any("target_prefix_tokens must be" in reason for reason in evidence["failure_reasons"]))

    def test_insufficient_shared_prefix_reports_counts(self) -> None:
        prefix_candidate = list(range(200))
        continuation_prompt = list(range(100)) + [999] + list(range(1000, 1199))

        prompt_p, evidence = RUNNER.build_token_workload(
            prefix_candidate, continuation_prompt, 2048, 128)

        self.assertEqual(len(prompt_p), 64)
        self.assertEqual(evidence["prefix_candidate_token_count"], 200)
        self.assertEqual(evidence["continuation_prompt_token_count"], 300)
        self.assertEqual(evidence["common_prefix_token_count"], 100)
        self.assertEqual(evidence["status"], "invalid")
        self.assertTrue(any("requires at least 128 tokens" in reason for reason in evidence["failure_reasons"]))

    def test_invalid_context_is_persisted_before_failure(self) -> None:
        ctx_size = 1024
        prefix_candidate = list(range(960))
        continuation_prompt = prefix_candidate + list(range(10000, 10033))
        with tempfile.TemporaryDirectory() as directory:
            case = pathlib.Path(directory)
            with mock.patch.object(RUNNER, "tokenize", side_effect=[prefix_candidate, continuation_prompt]):
                with self.assertRaisesRegex(RUNNER.WorkloadFailure, r"993\+32=1025>1024"):
                    RUNNER.prepare_token_workload(case, 12345, ctx_size, 960)
            evidence = json.loads((case / "workload.json").read_text(encoding="utf-8"))

        self.assertEqual(evidence["status"], "invalid")
        self.assertEqual(evidence["ctx_size"], ctx_size)
        self.assertEqual(evidence["prefix_candidate_token_count"], 960)
        self.assertEqual(evidence["continuation_prompt_token_count"], 993)
        self.assertEqual(evidence["prefix_token_count"], 960)
        self.assertEqual(evidence["context_token_count"], 1025)
        self.assertFalse(evidence["context_fits"])
        self.assertTrue(any("exceeds ctx_size" in reason for reason in evidence["failure_reasons"]))


class RunnerContextSizeTest(unittest.TestCase):
    def test_default_ctx_size_is_used_by_server_and_manifest(self) -> None:
        binary = pathlib.Path("/tmp/llama-server")
        model = pathlib.Path("/tmp/model.gguf")
        argv = RUNNER.server_argv(binary, model, 12345)
        with (
                mock.patch.object(RUNNER, "git", return_value=""),
                mock.patch.object(RUNNER, "tracked_diff_fingerprint", return_value="0" * 64)):
            manifest = RUNNER.base_manifest(
                binary, model, "20260805T000000Z", RUNNER.CTX_SIZE, 1024)

        self.assertEqual(RUNNER.CTX_SIZE, 2048)
        self.assertEqual(argv[argv.index("--ctx-size") + 1], "2048")
        self.assertEqual(manifest["parameters"]["ctx_size"], 2048)
        self.assertEqual(manifest["parameters"]["target_prefix_tokens"], 1024)

    def test_start_passes_the_final_argv_object_to_popen(self) -> None:
        binary = pathlib.Path("/tmp/llama-server")
        model = pathlib.Path("/tmp/model.gguf")
        argv = RUNNER.server_argv(binary, model, 12345, 8064)
        env = {"LANG": "C"}
        proc = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            case = pathlib.Path(directory)
            with mock.patch.object(RUNNER.subprocess, "Popen", return_value=proc) as popen:
                started = RUNNER.start(case, argv, env)
            stdout = popen.call_args.kwargs["stdout"]
            stderr = popen.call_args.kwargs["stderr"]
            stdout.close()
            stderr.close()

        self.assertIs(started, proc)
        self.assertIs(popen.call_args.args[0], argv)
        self.assertEqual(popen.call_args.kwargs["cwd"], case)
        self.assertIs(popen.call_args.kwargs["env"], env)

    def test_preflight_off_and_on_share_argv_and_parameter_evidence(self) -> None:
        ctx_size = 8064
        target_prefix_tokens = 1024
        clean_process = {
            "pid": 321,
            "pgid": 321,
            "exit_code": 0,
            "term_timed_out": False,
            "kill_timed_out": False,
            "residual_process": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            base = pathlib.Path(directory)
            binary = base / "llama-server"
            model = base / "model.gguf"
            binary.write_bytes(b"binary")
            model.write_bytes(b"model")
            with (
                    mock.patch.object(RUNNER, "git", return_value=""),
                    mock.patch.object(RUNNER, "tracked_diff_fingerprint", return_value="0" * 64)):
                manifest = RUNNER.base_manifest(
                    binary, model, "20260805T000000Z", ctx_size, target_prefix_tokens)

            observed: dict[str, tuple[dict, dict]] = {}
            for name, enabled in (("PREFLIGHT", None), ("OFF", False), ("GOVERNOR_ON", True)):
                root = base / f"root-{name}"
                root.mkdir()
                captured: list[list[str]] = []
                proc = mock.Mock(pid=321, returncode=None)
                proc.poll.return_value = None

                def fake_start(
                        _case: pathlib.Path,
                        argv: list[str],
                        _env: dict[str, str]) -> mock.Mock:
                    captured.append(argv)
                    return proc

                def fake_identity(_pid: int) -> dict:
                    argv = captured[0]
                    return {
                        "pid": 321,
                        "starttime_ticks": 1234,
                        "cmdline": list(argv),
                        "cmdline_sha256": sha_bytes(b"\0".join(
                            item.encode("utf-8") for item in argv)),
                    }

                with (
                        mock.patch.object(RUNNER, "free_port", return_value=12345),
                        mock.patch.object(RUNNER, "start", side_effect=fake_start),
                        mock.patch.object(RUNNER, "read_process_identity", side_effect=fake_identity),
                        mock.patch.object(RUNNER, "wait_health", return_value=False),
                        mock.patch.object(RUNNER, "stop", return_value=clean_process)):
                    if name == "PREFLIGHT":
                        RUNNER.run_physical_preflight(
                            binary, model, root, ctx_size, target_prefix_tokens)
                    else:
                        RUNNER.run_case(
                            name, binary, model, bool(enabled), root,
                            ctx_size, target_prefix_tokens)

                case = root / name
                execution = json.loads((case / "execution.json").read_text(encoding="utf-8"))
                result = json.loads((case / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(len(captured), 1)
                self.assertEqual(execution["argv"], captured[0])
                self.assertEqual(execution["server_identity"]["cmdline"], captured[0])
                self.assertEqual(captured[0][captured[0].index("--ctx-size") + 1], str(ctx_size))
                self.assertNotIn("--target-prefix-tokens", captured[0])
                self.assertEqual(result["ctx_size"], ctx_size)
                self.assertEqual(result["target_prefix_tokens"], target_prefix_tokens)
                observed[name] = (execution, result)

        self.assertEqual(manifest["parameters"]["ctx_size"], ctx_size)
        self.assertEqual(
            manifest["parameters"]["target_prefix_tokens"], target_prefix_tokens)
        normalized_argv = []
        for name in ("PREFLIGHT", "OFF", "GOVERNOR_ON"):
            argv = list(observed[name][0]["argv"])
            argv[argv.index("--port") + 1] = "<port>"
            normalized_argv.append(argv)
        self.assertEqual(normalized_argv[0], normalized_argv[1])
        self.assertEqual(normalized_argv[1], normalized_argv[2])

    def test_main_forwards_supported_ctx_sizes(self) -> None:
        target_prefix_tokens = 128
        for ctx_size in RUNNER.SUPPORTED_CTX_SIZES:
            with self.subTest(ctx_size=ctx_size), tempfile.TemporaryDirectory() as directory:
                root = pathlib.Path(directory)
                binary = root / "llama-server"
                model = root / "model.gguf"
                output = root / "artifact"
                binary.write_bytes(b"binary")
                model.write_bytes(b"model")
                case_result = {
                    "status": "complete",
                    "request_loop_started": True,
                    "ctx_size": ctx_size,
                    "target_prefix_tokens": target_prefix_tokens,
                }
                with (
                        mock.patch.object(
                            RUNNER, "run_physical_preflight", return_value=("supported", "")) as preflight,
                        mock.patch.object(RUNNER, "run_case", side_effect=[case_result, case_result]) as run_case,
                        mock.patch.object(RUNNER, "invoke_parser", return_value=0),
                        mock.patch.object(RUNNER.signal, "signal"),
                        mock.patch.object(RUNNER, "git", return_value=""),
                        mock.patch.object(RUNNER, "tracked_diff_fingerprint", return_value="0" * 64),
                        mock.patch.object(RUNNER.sys, "argv", [
                            str(RUNNER_PATH), "--binary", str(binary), "--model", str(model),
                            "--output-dir", str(output), "--ctx-size", str(ctx_size),
                            "--target-prefix-tokens", str(target_prefix_tokens),
                        ])):
                    with self.assertRaises(SystemExit) as caught:
                        RUNNER.main()

                self.assertEqual(caught.exception.code, 0)
                preflight.assert_called_once_with(
                    binary.resolve(), model.resolve(), output, ctx_size, target_prefix_tokens)
                self.assertEqual(run_case.call_args_list, [
                    mock.call(
                        "OFF", binary.resolve(), model.resolve(), False, output,
                        ctx_size, target_prefix_tokens),
                    mock.call(
                        "GOVERNOR_ON", binary.resolve(), model.resolve(), True, output,
                        ctx_size, target_prefix_tokens),
                ])
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["parameters"]["ctx_size"], ctx_size)
                self.assertEqual(
                    manifest["parameters"]["target_prefix_tokens"], target_prefix_tokens)
                self.assertEqual(manifest["OFF"]["ctx_size"], ctx_size)
                self.assertEqual(manifest["GOVERNOR_ON"]["ctx_size"], ctx_size)

    def test_main_forwards_target_prefix_tokens_and_records_workload(self) -> None:
        ctx_size = 8064
        target_prefix_tokens = 6144
        workload = {
            "target_prefix_tokens": target_prefix_tokens,
            "actual_p_token_count": target_prefix_tokens,
            "actual_pq_token_count": 8000,
            "actual_p_block_count": 96,
            "target_prefix_token_delta": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "llama-server"
            model = root / "model.gguf"
            output = root / "artifact"
            binary.write_bytes(b"binary")
            model.write_bytes(b"model")
            case_result = {
                "status": "complete",
                "request_loop_started": True,
                "ctx_size": ctx_size,
                "target_prefix_tokens": target_prefix_tokens,
                "workload": workload,
            }
            with (
                    mock.patch.object(
                        RUNNER, "run_physical_preflight", return_value=("supported", "")) as preflight,
                    mock.patch.object(RUNNER, "run_case", side_effect=[case_result, case_result]) as run_case,
                    mock.patch.object(RUNNER, "invoke_parser", return_value=0),
                    mock.patch.object(RUNNER.signal, "signal"),
                    mock.patch.object(RUNNER, "git", return_value=""),
                    mock.patch.object(RUNNER, "tracked_diff_fingerprint", return_value="0" * 64),
                    mock.patch.object(RUNNER.sys, "argv", [
                        str(RUNNER_PATH), "--binary", str(binary), "--model", str(model),
                        "--output-dir", str(output), "--ctx-size", str(ctx_size),
                        "--target-prefix-tokens", str(target_prefix_tokens),
                    ])):
                with self.assertRaises(SystemExit) as caught:
                    RUNNER.main()

            self.assertEqual(caught.exception.code, 0)
            preflight.assert_called_once_with(
                binary.resolve(), model.resolve(), output, ctx_size, target_prefix_tokens)
            self.assertEqual(run_case.call_args_list, [
                mock.call(
                    "OFF", binary.resolve(), model.resolve(), False, output, ctx_size, target_prefix_tokens),
                mock.call(
                    "GOVERNOR_ON", binary.resolve(), model.resolve(), True, output, ctx_size, target_prefix_tokens),
            ])
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["parameters"]["target_prefix_tokens"], target_prefix_tokens)
        self.assertEqual(manifest["OFF"]["workload"], workload)
        self.assertEqual(manifest["GOVERNOR_ON"]["workload"], workload)

    def test_main_rejects_invalid_ctx_sizes(self) -> None:
        for ctx_size in ("0", "1023", "8065", "8192", "invalid"):
            with self.subTest(ctx_size=ctx_size):
                with (
                        mock.patch.object(RUNNER.sys, "argv", [
                            str(RUNNER_PATH), "--ctx-size", ctx_size,
                            "--target-prefix-tokens", "128",
                        ]),
                        mock.patch.object(RUNNER.sys, "stderr")):
                    with self.assertRaises(SystemExit) as caught:
                        RUNNER.main()
                self.assertEqual(caught.exception.code, 2)

    def test_main_rejects_invalid_target_prefix_tokens(self) -> None:
        for target_prefix_tokens in ("-1", "0", "127", "invalid"):
            with self.subTest(target_prefix_tokens=target_prefix_tokens):
                with (
                        mock.patch.object(RUNNER.sys, "argv", [
                            str(RUNNER_PATH), "--target-prefix-tokens", target_prefix_tokens]),
                        mock.patch.object(RUNNER.sys, "stderr")):
                    with self.assertRaises(SystemExit) as caught:
                        RUNNER.main()
                self.assertEqual(caught.exception.code, 2)

    def test_main_requires_target_prefix_tokens(self) -> None:
        with (
                mock.patch.object(RUNNER.sys, "argv", [str(RUNNER_PATH)]),
                mock.patch.object(RUNNER.sys, "stderr")):
            with self.assertRaises(SystemExit) as caught:
                RUNNER.main()
        self.assertEqual(caught.exception.code, 2)

    def test_preflight_result_records_ctx_size(self) -> None:
        ctx_size = 8064
        target_prefix_tokens = 6144
        clean_process = {
            "pid": 123,
            "pgid": 123,
            "exit_code": 0,
            "term_timed_out": False,
            "kill_timed_out": False,
            "residual_process": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "llama-server"
            model = root / "model.gguf"
            proc = mock.Mock(pid=123)
            with (
                    mock.patch.object(RUNNER, "start", return_value=proc) as start,
                    mock.patch.object(RUNNER, "read_process_identity", return_value=None),
                    mock.patch.object(RUNNER, "stop", return_value=clean_process)):
                outcome, _reason = RUNNER.run_physical_preflight(
                    binary, model, root, ctx_size, target_prefix_tokens)
            result = json.loads((root / "PREFLIGHT" / "result.json").read_text(encoding="utf-8"))

        self.assertEqual(outcome, "failure")
        called_argv = start.call_args.args[1]
        self.assertEqual(called_argv[called_argv.index("--ctx-size") + 1], str(ctx_size))
        self.assertEqual(result["ctx_size"], ctx_size)
        self.assertEqual(result["target_prefix_tokens"], target_prefix_tokens)


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
        self.assertEqual(value["fields"]["transaction_id"], "1")
        self.assertEqual(value["fields"]["before_resident_pages"], "12")
        self.assertEqual(value["fields"]["after_resident_pages"], "11")

    def test_collects_120_blocks_as_three_transactions(self) -> None:
        blocks_by_transaction = (43, 43, 34)
        expected_blocks = sum(blocks_by_transaction)
        cumulative_blocks = 0
        resident_pages = 184
        lines: list[str] = []
        for index, blocks in enumerate(blocks_by_transaction):
            decision = 10 + index
            transaction = 20 + index
            candidate = claimant(
                expected_blocks - cumulative_blocks, cumulative_blocks, epoch=2)
            lines.append(resident_observation(
                decision, transaction, 222,
                resident(resident_pages - cumulative_blocks),
                resident(resident_pages - cumulative_blocks - blocks)))
            lines.append(marker(
                decision, decision, offload=1, armed_before=1, armed_after=1,
                transaction=transaction, blocks=blocks,
                byte_count=blocks * 8192, relief=blocks * 4096,
                outcome="completed", reason="target_satisfied", state_changed=1,
                seq=0, epoch=2, claim=candidate))
            cumulative_blocks += blocks
        post = claimant(0, expected_blocks, epoch=2)
        post_slots = [{
            "id": 0,
            "is_processing": False,
            "kv_claimant": raw_claimant(post),
        }]
        post_raw = json.dumps(post_slots).encode("utf-8")
        proc = mock.Mock(pid=222, returncode=None)
        proc.poll.return_value = None

        with tempfile.TemporaryDirectory() as directory:
            case = pathlib.Path(directory)
            (case / "server.stderr").write_text("".join(lines), encoding="utf-8")
            with mock.patch.object(
                    RUNNER, "query_slots_raw",
                    return_value=(200, post_raw, post_slots)):
                evidence, post_evidence = RUNNER.collect_offload_transactions(
                    case, proc, 12345, 0, expected_blocks)
            saved = json.loads(
                (case / "offload.json").read_text(encoding="utf-8"))

        self.assertEqual(evidence, saved)
        self.assertEqual(post_evidence["claimant"], post)
        self.assertEqual(evidence["status"], "complete")
        self.assertEqual(
            [int(item["fields"]["blocks"]) for item in evidence["transactions"]],
            [43, 43, 34])
        self.assertEqual(evidence["cumulative"]["blocks"], 120)
        self.assertEqual(
            evidence["cumulative"]["resident_drop_bytes"],
            evidence["cumulative"]["relieved_bytes"])

    def test_16_block_single_offload_continues_to_step2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            binary = root / "llama-server"
            model = root / "model.gguf"
            binary.write_bytes(b"binary")
            model.write_bytes(b"model")
            candidate = claimant(16, 0, epoch=2)
            post = claimant(0, 16, epoch=2)
            early_log = (
                resident_observation(2, 0, 222, resident(80), resident(80)) +
                marker(2, 2, offload=1, armed_before=1, armed_after=1, seq=0, epoch=2, claim=candidate) +
                resident_observation(3, 2, 222, resident(80), resident(64)) +
                marker(
                    3, 3, offload=1, armed_before=1, armed_after=1, transaction=2,
                    blocks=16, byte_count=131072, relief=65536, outcome="completed",
                    reason="target_satisfied", state_changed=1, seq=0, epoch=2,
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
            started_argv: list[str] = []

            def fake_start(
                    case: pathlib.Path,
                    argv: list[str],
                    _env: dict[str, str]):
                started_argv[:] = argv
                self.assertEqual(argv[argv.index("--ctx-size") + 1], "4096")
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
                evidence = json.loads(
                    (root / "GOVERNOR_ON" / "offload.json").read_text(
                        encoding="utf-8"))
                self.assertEqual(evidence["status"], "complete")
                self.assertEqual(evidence["cumulative"]["blocks"], 16)
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
                    mock.patch.object(RUNNER, "read_process_identity", side_effect=lambda _pid: {
                        "pid": 222,
                        "starttime_ticks": 1,
                        "cmdline": list(started_argv),
                        "cmdline_sha256": sha_bytes(b"\0".join(
                            item.encode("utf-8") for item in started_argv)),
                    }),
                    mock.patch.object(RUNNER, "wait_health", return_value=True),
                    mock.patch.object(RUNNER, "wait_capability", return_value=(
                        capability, {"offset": 0, "end": 0, "fields": capability}, None)),
                    mock.patch.object(RUNNER, "prepare_token_workload", return_value=(
                        list(range(1024)), list(range(1025)))) as prepare_token_workload,
                    mock.patch.object(RUNNER, "request_completion", side_effect=fake_request),
                    mock.patch.object(RUNNER, "query_slots_raw", return_value=(
                        200, post_raw, post_slots)),
                    mock.patch.object(RUNNER, "capture_resume_timing", return_value={"derived": {}}),
                    mock.patch.object(RUNNER, "stop", return_value=clean_process)):
                result = RUNNER.run_case(
                    "GOVERNOR_ON", binary, model, True, root, 4096, 1024)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["ctx_size"], 4096)
            self.assertEqual(result["target_prefix_tokens"], 1024)
            self.assertEqual(prepare_token_workload.call_args.args[-2], 4096)
            self.assertEqual(prepare_token_workload.call_args.args[-1], 1024)
            self.assertEqual(result["workload"], {
                "target_prefix_tokens": 1024,
                "actual_p_token_count": 1024,
                "actual_pq_token_count": 1025,
                "actual_p_block_count": 16,
                "target_prefix_token_delta": 0,
            })
            self.assertEqual(labels, ["step1", "step2"])
            self.assertFalse((root / "GOVERNOR_ON" / "pre_claimant.json").exists())
            saved_result = json.loads(
                (root / "GOVERNOR_ON" / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_result["ctx_size"], 4096)
            self.assertEqual(saved_result["workload"], result["workload"])
            execution = json.loads(
                (root / "GOVERNOR_ON" / "execution.json").read_text(encoding="utf-8"))
            self.assertEqual(execution["argv"], started_argv)
            self.assertEqual(execution["server_identity"]["cmdline"], started_argv)
            saved = json.loads(
                (root / "GOVERNOR_ON" / "offload.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(saved["cumulative"]["blocks"], 16)
            self.assertEqual(
                saved["transactions"][0]["fields"]["transaction_id"], "2")
            self.assertEqual(
                saved["cumulative"]["resident_drop_bytes"],
                saved["cumulative"]["relieved_bytes"])
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
