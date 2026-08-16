#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Short GT-trace-1B-A protocol tests.

The fake server exercises request/response shape only.  Every fake-derived
transcript must remain UNVERIFIED/OPEN and is never presented as real model
qualification evidence.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from trace_compiler.runtime_materialize import (  # type: ignore[reportMissingImports]
    ApiResponse,
    CalibrationError,
    ContextOverflowError,
    DirectTokenContractError,
    PartialBlockConflictError,
    ParentArtifactError,
    SafeAlphabetMaterializer,
    ServerProcessIdentityError,
    ServerAuthority,
    TranscriptDriftError,
    build_dry_run_plan,
    build_reference_request,
    calibrate_tokenizer,
    capture_process_start_identity_snapshot,
    canonical_json_bytes,
    load_frozen_slice,
    load_transcript,
    materialize_reference_transcript,
    preflight_direct_token_contract,
    sha256_bytes,
    validate_policy_completion,
    verify_server_process_identity,
    MATERIALIZE_MODE_REAL,
)


class FakeServer:
    """Protocol fixture; it is deliberately not a real model/tokenizer."""

    def __init__(
        self,
        n_ctx: int = 128,
        unstable_tokenize: bool = False,
        token_ids=None,
        consume_prompt: bool = True,
    ) -> None:
        self.n_ctx = n_ctx
        self.unstable_tokenize = unstable_tokenize
        self.token_ids = tuple(token_ids or (11, 13, 17, 19))
        self.consume_prompt = consume_prompt
        self.tokenize_calls = 0
        self.completion_requests: list[dict] = []

    def get(self, path: str):
        if path == "/props":
            return ApiResponse.from_payload({
                "default_generation_settings": {"n_ctx": self.n_ctx},
                "model_path": "FAKE_MODEL",
            })
        if path == "/slots":
            return ApiResponse.from_payload([{
                "id": 0,
                "n_ctx": self.n_ctx,
                "is_processing": False,
            }])
        raise AssertionError(path)

    def post(self, path: str, payload: dict):
        if path == "/tokenize":
            self.tokenize_calls += 1
            ids = list(self.token_ids)
            if self.unstable_tokenize and self.tokenize_calls % 2 == 0:
                ids = ids[:-1] + [23]
            return ApiResponse.from_payload({
                "tokens": [{"id": token_id, "special": False} for token_id in ids]
            })
        if path == "/completion":
            self.completion_requests.append(dict(payload))
            count = int(payload["n_predict"])
            generated = [self.token_ids[index % len(self.token_ids)] for index in range(count)]
            return ApiResponse.from_payload({
                "id_slot": payload["id_slot"],
                "tokens_evaluated": len(payload["prompt"]) if self.consume_prompt else 0,
                "tokens_predicted": count,
                "tokens": generated,
            })
        raise AssertionError(path)


class StatefulFakeServer(FakeServer):
    """Single-slot fixture that exposes cache reuse and cold-start boundaries.

    The fake cannot reproduce Alibaba's real block-level KV reuse, so it models
    reuse loosely: a same-session continuation (cache_prompt=True) must share a
    common prefix with the slot's resident tokens, and a cold start
    (cache_prompt=False) must reset the slot.  The exact reuse contract is
    validated upstream via expected_full_block_lcp; here we only check the
    cold-start/continuation boundary shape, not strict prompt+completion
    inclusion (which the trace semantics explicitly reject).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_tokens: tuple[int, ...] = ()
        self.reference_events: list[dict] = []

    def post(self, path: str, payload: dict):
        if path != "/completion":
            return super().post(path, payload)
        prompt = tuple(payload["prompt"])
        cache_prompt = payload.get("cache_prompt")
        if cache_prompt:
            cache_hit = bool(self.cache_tokens) and _common_prefix_len(prompt, self.cache_tokens) > 0
            if not cache_hit:
                raise AssertionError("same-session continuation did not hit cached prefix")
        else:
            cache_hit = False
        response = super().post(path, payload)
        generated = tuple(response.payload["tokens"])
        self.cache_tokens = prompt + generated
        if payload["n_predict"] > 0:
            self.reference_events.append({
                "prompt": prompt,
                "cache_prompt": cache_prompt,
                "cache_hit": cache_hit,
            })
        return response


class ConflictingMaterializer(SafeAlphabetMaterializer):
    def __init__(self, alphabet):
        super().__init__(alphabet)
        self.calls = 0

    def materialize_block(self, lineage_id: int, block_id: str):
        self.calls += 1
        result = super().materialize_block(lineage_id, block_id)
        if self.calls >= 2 and block_id == "L1:H1":
            return tuple(reversed(result))
        return result


def _common_prefix_len(left, right) -> int:
    count = 0
    for a, b in zip(left, right):
        if a != b:
            break
        count += 1
    return count


def _event(lineage_id, turn, timestamp, input_length, n_predict, block_ids):
    return {
        "line_msg_index": 0,
        "lineage_id": lineage_id,
        "lifecycle_generation": 1,
        "event_type": "TURN_START",
        "ts": timestamp,
        "turn": turn,
        "request_plan": {
            "target_output_length": n_predict,
            "n_predict": n_predict,
            "required_context_tokens": input_length + n_predict,
        },
        "feature": {
            "lineage_id": lineage_id,
            "turn": turn,
            "relative_arrival_offset": timestamp,
            "timestamp_abs": timestamp,
            "input_length": input_length,
            "type": "text",
            "block_count": len(block_ids),
            "session_namespaced_block_ids": list(block_ids),
        },
    }


def _write_parent(root: Path, sessions=None):
    if sessions is None:
        sessions = [
            (1, 1, 0.0, 15, 2, ["L1:H1"]),
            (2, 1, 1.0, 15, 1, ["L2:H1"]),
            (1, 2, 2.0, 25, 3, ["L1:H1", "L1:H2"]),
        ]
    events = []
    for item in sessions:
        events.append(_event(*item))
    for index, event in enumerate(events, 1):
        event["line_msg_index"] = index
    event_text = "\n".join(
        json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for event in events
    ) + ("\n" if events else "")
    events_path = root / "events_typical.jsonl"
    events_path.write_text(event_text, encoding="utf-8")
    lineages = sorted({item[0] for item in sessions})
    max_prompt = max((item[3] for item in sessions), default=0)
    max_required = max((item[3] + item[4] for item in sessions), default=0)
    manifest = {
        "schema_version": "gt-trace-1a/v2",
        "slice_class": "typical",
        "trace_key": "traceA",
        "trace_repo_head": "trace-head",
        "trace_file": "qwen_traceA_blksz_16.jsonl",
        "trace_file_sha256": "trace-file-sha",
        "source_window_ts": [0.0, 10.0],
        "trace_span": [0.0, 10.0],
        "calibration_window": [0.0, 5.0],
        "evaluation_window": [5.0, 10.0],
        "selected_event_span": [0.0, 2.0],
        "selection_rule": "fixture",
        "seed": 7,
        "session_count": len(lineages),
        "turn_count": len(events),
        "time_dilation": 1.0,
        "prefix_mode": "session_namespaced",
        "materialize_mode": "placeholder_token_ids",
        "runtime_event_scope": "arrival_and_request_plan_only",
        "ttl_replay_scope": "offline_projected_characterization_only",
        "tokenization_calibration_requirement": {
            "status": "OPEN",
            "materialize_mode_now": "placeholder_token_ids",
            "required_by_gate": "GT-trace-1B",
        },
        "ttl_calibration_identity": {},
        "max_prompt_tokens": max_prompt,
        "max_required_context_tokens": max_required,
        "max_context_footprint_tokens": max_prompt,
        "event_stream_sha256": sha256_bytes(event_text.encode("utf-8")),
        "event_count": len(events),
        "session_ids": lineages,
        "descriptive_stats": {},
    }
    manifest_path = root / "manifest_typical.json"
    manifest_path.write_bytes(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
        + b"\n"
    )
    return manifest_path, events_path


def _calibrate(server: FakeServer):
    return calibrate_tokenizer(
        server,
        corpus=("固定 UTF-8 corpus", "second sample"),
        repetitions=2,
        model_sha256=None,
        binary_sha256=None,
        real_model=False,
    )


def _write_modified_transcript(path: Path, document: dict) -> None:
    unsigned = dict(document)
    unsigned.pop("transcript_sha256", None)
    document["transcript_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _promote_realish(transcript) -> dict:
    document = json.loads(transcript.to_canonical_json())
    model_sha = "a" * 64
    binary_sha = "b" * 64
    identity = {
        "pid": 4321,
        "base_url": "http://127.0.0.1:8080",
        "port": 8080,
        "proc_exe": "/tmp/llama-server",
        "proc_exe_sha256": binary_sha,
        "proc_cwd": "/tmp",
        "proc_root": "/proc",
        "proc_stat_starttime": "4242",
        "cmdline": ["/tmp/llama-server", "--model", "/tmp/model.gguf", "--port", "8080"],
        "model_path": "/tmp/model.gguf",
        "model_sha256": model_sha,
        "binary_path": "/tmp/llama-server",
        "binary_sha256": binary_sha,
        "props_model_path": "/tmp/model.gguf",
    }
    identity["binding_sha256"] = sha256_bytes(canonical_json_bytes(identity))
    document["materialize_mode"] = MATERIALIZE_MODE_REAL
    document["materialization_status"] = "MODEL_BOUND_REAL"
    document["model"] = {
        "model_sha256": model_sha,
        "binary_sha256": binary_sha,
        "identity_status": "REAL",
        "server_process_identity": identity,
    }
    document["runtime_contract"] = {
        "ctx_size": 128,
        "executor": "llama-server",
        "kv_unified": True,
        "parallel": 3,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "no_cache_idle_slots": True,
        "no_context_shift": True,
        "paged_block_size": 16,
        "action_target_bytes": 4096,
        "max_blocks": 8,
    }
    authority = document["effective_n_ctx_authority"]
    authority.update({
        "source": "server_props_and_slots",
        "props_model_path": "/tmp/model.gguf",
        "props_sha256": "c" * 64,
        "slots_sha256": "d" * 64,
        "probe_request_sha256": "e" * 64,
        "probe_response_sha256": "f" * 64,
        "process_identity": identity,
    })
    calibration = document["token_calibration"]
    calibration.update({
        "model_sha256": model_sha,
        "binary_sha256": binary_sha,
        "real_model": True,
    })
    document["reference_pass"].update({
        "process_start_identity_before": {
            "pid": 4321,
            "proc_stat_starttime": "4242",
        },
        "process_start_identity_after": {
            "pid": 4321,
            "proc_stat_starttime": "4242",
        },
    })
    unsigned = dict(document)
    unsigned.pop("transcript_sha256", None)
    document["transcript_sha256"] = sha256_bytes(canonical_json_bytes(unsigned))
    return document


class RuntimeMaterializeTest(unittest.TestCase):
    def test_direct_token_schema_and_authority(self):
        server = FakeServer()
        calibration = _calibrate(server)
        authority = preflight_direct_token_contract(
            server, probe_token_id=calibration.safe_alphabet[0], slot_id=0
        )
        self.assertEqual(authority.source, "server_props_and_slots")
        self.assertEqual(authority.effective_n_ctx, 128)
        request = build_reference_request((1, 2, 3), 4, 0)
        self.assertEqual(request["prompt"], [1, 2, 3])
        self.assertTrue(request["cache_prompt"])
        self.assertTrue(request["return_tokens"])
        self.assertTrue(request["ignore_eos"])
        self.assertNotIn("stop", request)

    def test_preflight_is_explicit_one_token_generation(self):
        server = FakeServer()
        preflight_direct_token_contract(server, probe_token_id=11)
        request = server.completion_requests[-1]
        self.assertEqual(request["prompt"], [11])
        self.assertEqual(request["n_predict"], 1)
        self.assertFalse(request["cache_prompt"])
        self.assertTrue(request["return_tokens"])
        self.assertTrue(request["ignore_eos"])
        self.assertFalse(request["stream"])

    def test_preflight_requires_prompt_consumption(self):
        server = FakeServer(consume_prompt=False)
        with self.assertRaises(DirectTokenContractError):
            preflight_direct_token_contract(server, probe_token_id=11)

    def test_preflight_requires_returned_probe_token(self):
        server = FakeServer()
        original_post = server.post

        def bad_post(path, payload):
            response = original_post(path, payload)
            if path == "/completion":
                broken = dict(response.payload)
                broken["tokens"] = []
                return ApiResponse.from_payload(broken)
            return response

        server.post = bad_post
        with self.assertRaises(DirectTokenContractError):
            preflight_direct_token_contract(server, probe_token_id=11)

    def test_preflight_rejects_slot_context_mismatch(self):
        server = FakeServer()
        original_get = server.get

        def bad_get(path):
            value = original_get(path)
            if path == "/slots":
                return ApiResponse.from_payload([{"id": 0, "n_ctx": 64, "is_processing": False}])
            return value

        server.get = bad_get
        with self.assertRaises(DirectTokenContractError):
            preflight_direct_token_contract(server, probe_token_id=11)

    def test_deterministic_alphabet_and_unstable_rejection(self):
        calibration = _calibrate(FakeServer())
        self.assertEqual(calibration.safe_alphabet, tuple(sorted(calibration.safe_alphabet)))
        self.assertEqual(
            calibration.safe_alphabet_sha256,
            sha256_bytes(json.dumps(list(calibration.safe_alphabet), separators=(",", ":"), sort_keys=True).encode("utf-8")),
        )
        with self.assertRaises(CalibrationError):
            _calibrate(FakeServer(unstable_tokenize=True))

    def test_illegal_alphabet_fails_closed(self):
        with self.assertRaises(CalibrationError):
            SafeAlphabetMaterializer([1, -2])
        with self.assertRaises(CalibrationError):
            SafeAlphabetMaterializer([1, 1])

    def test_exact_length_prefix_isolation_and_open_status(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(Path(directory))
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            calibration = _calibrate(server)
            authority = ServerAuthority.fixture(128)
            transcript = materialize_reference_transcript(
                frozen, calibration, server, authority=authority
            )
            turns = transcript.turns
            first = turns[0]
            third = turns[2]
            self.assertEqual(len(first.prompt_tokens), 15)
            self.assertEqual(len(third.prompt_tokens), 25)
            # New per-turn materialization: each turn's prompt comes from its
            # OWN input_length + session_namespaced_block_ids.  Turn (1,1) has a
            # 15-token input below one block, so it is entirely a partial tail
            # of L1:H1.  Turn (1,2) has a 25-token input = one full block of
            # L1:H1 plus a 9-token partial tail of L1:H2.  The two turns share
            # zero full blocks (turn 1 has no complete block), so the reuse
            # contract declares expected_full_block_lcp == 0 and the two prompts
            # must NOT be required to share any prefix beyond the session
            # discriminator at position 0.
            self.assertEqual(
                (first.input_length, third.input_length), (15, 25)
            )
            constraint = SafeAlphabetMaterializer(calibration.safe_alphabet)
            constraint.bind_session_ids((1, 2))
            first_full = constraint.materialize_block(1, "L1:H1")
            # Same full block id materializes to identical 16 tokens both times.
            self.assertEqual(first_full, first_full)
            # Turn (1,2)'s full-block prefix must equal re-materialized L1:H1.
            self.assertEqual(third.prompt_tokens[:16], first_full)
            # Turn (1,1) is a 15-token remainder tail of L1:H1; it differs from
            # the full-block materialization because the tail is request-local
            # identity, not stable cross-request content.
            self.assertNotEqual(first.prompt_tokens, first_full[:15])
            self.assertNotEqual(turns[0].prompt_tokens[0], turns[1].prompt_tokens[0])
            self.assertNotEqual(turns[0].prompt_tokens, turns[1].prompt_tokens)
            self.assertEqual(transcript.materialization_status, "UNVERIFIED")
            self.assertEqual(transcript.materialize_mode, "OPEN")
            self.assertEqual(transcript.token_calibration["real_model"], False)
            self.assertEqual(len(server.completion_requests), 3)
            self.assertTrue(all("prompt" in request and isinstance(request["prompt"], list) for request in server.completion_requests))

    def test_reference_execution_is_session_contiguous_and_artifact_order_is_parent_order(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(Path(directory))
            frozen = load_frozen_slice(manifest_path, events_path)
            server = StatefulFakeServer()
            transcript = materialize_reference_transcript(
                frozen,
                _calibrate(server),
                server,
                authority=ServerAuthority.fixture(128),
            )
            reference_events = server.reference_events
            self.assertEqual(
                [event["prompt"] for event in reference_events],
                [
                    transcript.turns[0].prompt_tokens,
                    transcript.turns[2].prompt_tokens,
                    transcript.turns[1].prompt_tokens,
                ],
            )
            self.assertEqual(
                [event["cache_prompt"] for event in reference_events],
                [False, True, False],
            )
            self.assertEqual(
                [event["cache_hit"] for event in reference_events],
                [False, True, False],
            )
            self.assertEqual(
                [(turn.lineage_id, turn.turn) for turn in transcript.turns],
                [(1, 1), (2, 1), (1, 2)],
            )
            self.assertEqual(transcript.reference_pass["execution_order"], "session_contiguous")
            self.assertEqual(transcript.reference_pass["artifact_order"], "parent_event_order")
            self.assertEqual(
                transcript.reference_pass["reference_completion_role"],
                "qualification_observation_only",
            )
            self.assertEqual(
                transcript.reference_pass["reuse_contract"],
                "expected_full_block_lcp",
            )
            self.assertIsInstance(
                transcript.reference_pass["block_lcp_observations"], list
            )

    def test_session_discriminator_scales_and_alphabet_shortage_fails(self):
        sessions = [
            (lineage_id, 1, float(lineage_id), 8, 1, [f"L{lineage_id}:H1"])
            for lineage_id in range(1, 101)
        ]
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(Path(directory), sessions=sessions)
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer(token_ids=range(1000, 1100))
            transcript = materialize_reference_transcript(
                frozen,
                _calibrate(server),
                server,
                authority=ServerAuthority.fixture(128),
            )
            first_tokens = [turn.prompt_tokens[0] for turn in transcript.turns]
            self.assertEqual(len(first_tokens), 100)
            self.assertEqual(len(set(first_tokens)), 100)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(Path(directory), sessions=sessions)
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            with self.assertRaises(CalibrationError):
                materialize_reference_transcript(
                    frozen,
                    _calibrate(server),
                    server,
                    authority=ServerAuthority.fixture(128),
                )

    def test_real_server_identity_rejects_wrong_model_binary_or_user_sha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            wrong_model = root / "wrong-model.gguf"
            binary = root / "llama-server"
            wrong_binary = root / "wrong-server"
            model.write_bytes(b"model-a")
            wrong_model.write_bytes(b"model-b")
            binary.write_bytes(b"binary-a")
            wrong_binary.write_bytes(b"binary-b")
            proc = root / "proc" / "1234"
            proc.mkdir(parents=True)
            (proc / "exe").symlink_to(binary)
            (proc / "cwd").symlink_to(root)
            (proc / "cmdline").write_bytes(
                f"{binary}\0--model\0{model}\0--port\0{8080}\0".encode("utf-8")
            )
            stat_tail = ["S"] + ["0"] * 18 + ["4242"]
            (proc / "stat").write_text(
                f"1234 (llama-server) {' '.join(stat_tail)}", encoding="utf-8"
            )
            authority = replace(
                ServerAuthority.fixture(128),
                source="server_props_and_slots",
                props_model_path=str(model),
            )
            kwargs = {
                "authority": authority,
                "base_url": "http://127.0.0.1:8080",
                "server_pid": 1234,
                "proc_root": root / "proc",
            }
            bound = verify_server_process_identity(
                model_path=model,
                binary_path=binary,
                **kwargs,
            )
            self.assertEqual(bound["model_sha256"], hashlib.sha256(b"model-a").hexdigest())
            self.assertEqual(bound["proc_stat_starttime"], "4242")
            self.assertEqual(
                capture_process_start_identity_snapshot(bound, "test snapshot"),
                {"pid": 1234, "proc_stat_starttime": "4242"},
            )
            with self.assertRaises(ServerProcessIdentityError):
                verify_server_process_identity(model_path=wrong_model, binary_path=binary, **kwargs)
            with self.assertRaises(ServerProcessIdentityError):
                verify_server_process_identity(model_path=model, binary_path=wrong_binary, **kwargs)
            with self.assertRaises(ServerProcessIdentityError):
                verify_server_process_identity(
                    model_path=model,
                    binary_path=binary,
                    model_sha256="0" * 64,
                    **kwargs,
                )
            self.assertEqual(MATERIALIZE_MODE_REAL, "direct_token_ids")

    def test_real_server_identity_binds_runtime_contract_to_proc_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.gguf"
            binary = root / "llama-server"
            model.write_bytes(b"model-a")
            binary.write_bytes(b"binary-a")
            proc = root / "proc" / "1234"
            proc.mkdir(parents=True)
            (proc / "exe").symlink_to(binary)
            (proc / "cwd").symlink_to(root)
            runtime_contract = {
                "ctx_size": 128,
                "executor": "llama-server",
                "kv_unified": True,
                "parallel": 3,
                "cache_type_k": "f16",
                "cache_type_v": "f16",
                "no_cache_idle_slots": True,
                "no_context_shift": True,
                "paged_block_size": 16,
                "action_target_bytes": 4096,
                "max_blocks": 8,
            }
            def write_cmdline(cache_type: str = "f16", parallel: int = 3):
                argv = [
                    str(binary), "--model", str(model), "--port", "8080",
                    "--ctx-size", "128", "--cache-type-k", cache_type,
                    "--cache-type-v", cache_type, "--parallel", str(parallel),
                    "--kv-unified", "--no-cache-idle-slots", "--no-context-shift",
                ]
                (proc / "cmdline").write_bytes("\0".join(argv).encode("utf-8") + b"\0")
            write_cmdline()
            stat_tail = ["S"] + ["0"] * 18 + ["4242"]
            (proc / "stat").write_text(
                f"1234 (llama-server) {' '.join(stat_tail)}", encoding="utf-8"
            )
            authority = replace(
                ServerAuthority.fixture(128),
                source="server_props_and_slots",
                props_model_path=str(model),
            )
            kwargs = {
                "authority": authority,
                "base_url": "http://127.0.0.1:8080",
                "server_pid": 1234,
                "model_path": model,
                "binary_path": binary,
                "proc_root": root / "proc",
                "runtime_contract": runtime_contract,
            }
            verify_server_process_identity(**kwargs)
            write_cmdline(cache_type="f32")
            with self.assertRaises(ServerProcessIdentityError):
                verify_server_process_identity(**kwargs)
            write_cmdline(parallel=2)
            with self.assertRaises(ServerProcessIdentityError):
                verify_server_process_identity(**kwargs)

    def test_previous_partial_tail_reminder_change_is_legal(self):
        # Alibaba semantics (A): a sub-block partial block is NOT stable
        # cross-request content identity.  Two turns may legitimately rematerialize
        # different tokens at the same block position when that position is a
        # partial tail.  This must therefore PASS, not fail-closed.
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(
                Path(directory),
                sessions=[
                    (1, 1, 0.0, 5, 1, ["L1:H1"]),
                    (1, 2, 1.0, 20, 1, ["L1:H1", "L1:H2"]),
                ],
            )
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            calibration = _calibrate(server)
            partial_materializer = ConflictingMaterializer(calibration.safe_alphabet)
            transcript = materialize_reference_transcript(
                frozen,
                calibration,
                server,
                authority=ServerAuthority.fixture(128),
                materializer=partial_materializer,
            )
            first, second = transcript.turns[0], transcript.turns[1]
            # Both turns' first block id is L1:H1, but turn 1 is a 5-token tail and
            # turn 2 is a full 16-token block of L1:H1 followed by a 4-token tail.
            # expected_full_block_lcp = min(0, 1) = 0, so no shared full-block
            # prefix is declared and the reuse contract is satisfied vacuously.
            self.assertEqual(first.block_ids[0], "L1:H1")
            self.assertEqual(second.block_ids[:1], ("L1:H1",))
            self.assertEqual(transcript.materialization_status, "UNVERIFIED")
            # The two turns' partial tails at block position 0 are allowed to
            # diverge because a partial block is request-local identity.
            self.assertEqual(len(first.prompt_tokens), 5)
            self.assertEqual(len(second.prompt_tokens), 20)

    def test_full_block_hash_divergence_cannot_fake_reuse(self):
        # Alibaba semantics (B): a divergence in a SHARED FULL block's content
        # must still fail-closed.  Turn 1 and turn 2 declare the same full
        # block L1:H1 at position 0, so expected_full_block_lcp = 16; if the
        # materializer returns different content for L1:H1 on the second call,
        # the frozen transcript must reject the fight.
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(
                Path(directory),
                sessions=[
                    (1, 1, 0.0, 20, 1, ["L1:H1", "L1:H2"]),
                    (1, 2, 1.0, 36, 1, ["L1:H1", "L1:H2", "L1:H3"]),
                ],
            )
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            calibration = _calibrate(server)
            with self.assertRaises(PartialBlockConflictError):
                materialize_reference_transcript(
                    frozen,
                    calibration,
                    server,
                    authority=ServerAuthority.fixture(128),
                    materializer=ConflictingMaterializer(calibration.safe_alphabet),
                )

    def test_same_full_block_prefix_produces_same_token_prefix(self):
        # Alibaba semantics (C): a full 16-token block keyed by its namespaced
        # block id always materializes to the same 16-token sequence.  Two
        # first turns that reuse L1:H1 must therefore share the first 16 tokens.
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(
                Path(directory),
                sessions=[
                    (1, 1, 0.0, 32, 1, ["L1:H1", "L1:H2"]),
                    (1, 2, 1.0, 48, 1, ["L1:H1", "L1:H2", "L1:H3"]),
                ],
            )
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            calibration = _calibrate(server)
            transcript = materialize_reference_transcript(
                frozen,
                calibration,
                server,
                authority=ServerAuthority.fixture(128),
            )
            first, second = transcript.turns[0], transcript.turns[1]
            self.assertEqual(first.prompt_tokens[:16], second.prompt_tokens[:16])
            self.assertEqual(first.prompt_tokens[:32], second.prompt_tokens[:32])
            # The actual token LCP is capped at the shorter prompt (32): the two
            # full shared blocks (L1:H1, L1:H2) agree, and the third block
            # L1:H3 only exists in turn 2.  actual_token_lcp >= expected.
            observations = transcript.reference_pass["block_lcp_observations"]
            matching = [item for item in observations if item["from_turn"] == 1]
            self.assertEqual(matching[0]["expected_full_block_lcp"], 32)
            self.assertGreaterEqual(matching[0]["actual_token_lcp"], 32)
            self.assertLessEqual(matching[0]["actual_token_lcp"], 32)

    def test_partial_tail_has_exactly_remainder_length_no_padding(self):
        # Alibaba semantics (D): the partial tail must be exactly the
        # remainder length, never padded out to a full block.  Verify for a few
        # non-multiple lengths and for an exact-block-multiple length.
        for input_length, expected_tail in [(5, 5), (10, 10), (16, 0), (25, 9), (32, 0)]:
            with self.subTest(input_length=input_length):
                with tempfile.TemporaryDirectory() as directory:
                    block_count = (input_length + 15) // 16 or 1
                    block_ids = [f"L1:H{idx}" for idx in range(1, block_count + 1)]
                    manifest_path, events_path = _write_parent(
                        Path(directory),
                        sessions=[(1, 1, 0.0, input_length, 1, block_ids)],
                    )
                    frozen = load_frozen_slice(manifest_path, events_path)
                    server = FakeServer()
                    calibration = _calibrate(server)
                    transcript = materialize_reference_transcript(
                        frozen,
                        calibration,
                        server,
                        authority=ServerAuthority.fixture(128),
                    )
                    turn = transcript.turns[0]
                    self.assertEqual(len(turn.prompt_tokens), input_length)
                    self.assertEqual(len(turn.prompt_tokens) % 16, expected_tail % 16)

    def test_next_turn_need_not_include_previous_completion(self):
        # Alibaba semantics (E): the next turn's input may strip special
        # tokens from the previous reference completion, so
        # previous.prompt + previous.reference_completion is NOT a strict prefix
        # invariant of the next turn's prompt.  A transcript whose next turn
        # does NOT strictly include the previous completion must be accepted by
        # the loader.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, events_path = _write_parent(
                root,
                sessions=[
                    (1, 1, 0.0, 20, 3, ["L1:H1", "L1:H2"]),
                    (1, 2, 1.0, 36, 1, ["L1:H1", "L1:H2", "L1:H3"]),
                ],
            )
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            calibration = _calibrate(server)
            transcript = materialize_reference_transcript(
                frozen,
                calibration,
                server,
                authority=ServerAuthority.fixture(128),
            )
            first, second = transcript.turns[0], transcript.turns[1]
            carried = first.prompt_tokens + first.reference_completion_tokens
            # The next prompt explicitly does NOT strictly extend the carried
            # prompt+completion prefix in general; only the shared full-block
            # prefix (16 tokens of L1:H1) must agree.
            self.assertEqual(second.prompt_tokens[:16], first.prompt_tokens[:16])
            self.assertNotEqual(second.prompt_tokens[: len(carried)], carried)
            transcript_path = root / "transcript.json"
            transcript.write(transcript_path)
            loaded = load_transcript(transcript_path)
            self.assertEqual(loaded.transcript_sha256, transcript.transcript_sha256)

    def test_tampered_full_block_prefix_is_rejected_even_after_rehash(self):
        # Alibaba semantics (F): tamper a shared full-block prefix token AND
        # recompute the transcript SHA.  The loader must still refuse because
        # the expected_full_block_lcp reuse contract no longer holds.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, events_path = _write_parent(
                root,
                sessions=[
                    (1, 1, 0.0, 32, 1, ["L1:H1", "L1:H2"]),
                    (1, 2, 1.0, 48, 1, ["L1:H1", "L1:H2", "L1:H3"]),
                ],
            )
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            calibration = _calibrate(server)
            transcript = materialize_reference_transcript(
                frozen,
                calibration,
                server,
                authority=ServerAuthority.fixture(128),
            )
            real_document = json.loads(transcript.to_canonical_json())
            # Tamper a token inside the SHARED full-block prefix (position 16 is
            # the first token of block L1:H2, shared by both turns).
            real_document["turns"][1]["prompt_tokens"][16] += 1
            real_document["turns"][1]["prompt_sha256"] = sha256_bytes(
                canonical_json_bytes(real_document["turns"][1]["prompt_tokens"])
            )
            tampered_path = root / "tampered.json"
            _write_modified_transcript(tampered_path, real_document)
            with self.assertRaises(ValueError):
                load_transcript(tampered_path)

    def test_context_overflow_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(Path(directory))
            frozen = load_frozen_slice(manifest_path, events_path)
            server = FakeServer()
            with self.assertRaises(ContextOverflowError):
                materialize_reference_transcript(
                    frozen,
                    _calibrate(server),
                    server,
                    authority=ServerAuthority.fixture(20),
                )

    def test_v1_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(Path(directory))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["schema_version"] = "gt-trace-1a/v1"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ParentArtifactError):
                load_frozen_slice(manifest_path, events_path)

    def test_transcript_round_trip_without_source_trace_and_drift_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, events_path = _write_parent(root)
            frozen = load_frozen_slice(manifest_path, events_path)
            transcript = materialize_reference_transcript(
                frozen,
                _calibrate(FakeServer()),
                FakeServer(),
                authority=ServerAuthority.fixture(128),
            )
            transcript_path = root / "transcript.json"
            transcript.write(transcript_path)
            loaded = load_transcript(transcript_path)
            self.assertEqual(loaded.transcript_sha256, transcript.transcript_sha256)
            self.assertEqual(loaded.to_canonical_json(), transcript.to_canonical_json())
            turn = loaded.turns[0]
            validate_policy_completion(
                loaded,
                lineage_id=turn.lineage_id,
                turn=turn.turn,
                completion_tokens=turn.reference_completion_tokens,
            )
            with self.assertRaises(TranscriptDriftError):
                validate_policy_completion(
                    loaded,
                    lineage_id=turn.lineage_id,
                    turn=turn.turn,
                    completion_tokens=turn.reference_completion_tokens[:-1] + (999,),
                )
            # The loader only reads this child transcript; source trace identity
            # is metadata, not an online dependency.
            (root / "source-trace-deleted").write_text("gone", encoding="utf-8")
            loaded_again = load_transcript(transcript_path)
            self.assertEqual(loaded_again.transcript_sha256, loaded.transcript_sha256)

    def test_loader_trust_boundary_rejects_rewritten_real_tampering_and_allows_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, events_path = _write_parent(root)
            frozen = load_frozen_slice(manifest_path, events_path)
            transcript = materialize_reference_transcript(
                frozen,
                _calibrate(FakeServer()),
                FakeServer(),
                authority=ServerAuthority.fixture(128),
            )
            open_path = root / "open.json"
            transcript.write(open_path)
            self.assertEqual(load_transcript(open_path).materialization_status, "UNVERIFIED")

            real_document = _promote_realish(transcript)
            valid_real_path = root / "real-valid.json"
            _write_modified_transcript(valid_real_path, real_document)
            self.assertEqual(load_transcript(valid_real_path).materialization_status, "MODEL_BOUND_REAL")

            def mutate_mode(document):
                document["materialize_mode"] = "OPEN"

            def mutate_status(document):
                document["materialization_status"] = "UNVERIFIED"

            def mutate_model_identity(document):
                document["model"]["binary_sha256"] = "0" * 64

            def mutate_authority(document):
                document["effective_n_ctx_authority"]["source"] = "fixture"

            def mutate_prefix(document):
                prompt = document["turns"][2]["prompt_tokens"]
                # Turn (1,2) has a 25-token prompt = one full 16-token block
                # (L1:H1) plus a 9-token tail.  Tamper a token INSIDE the
                # reproducible full block (position 8) and recompute its SHA.
                # The loader re-derives the full-block synthetic content from
                # the materializer, so it must reject this even after the
                # transcript SHA is recomputed.
                prompt[8] += 1
                document["turns"][2]["prompt_sha256"] = sha256_bytes(
                    canonical_json_bytes(prompt)
                )

            def mutate_discriminator(document):
                entries = document["materializer"]["session_discriminators"]
                entries[1]["token_id"] = entries[0]["token_id"]
                document["materializer"]["session_discriminators_sha256"] = sha256_bytes(
                    canonical_json_bytes(entries)
                )

            def mutate_scalar_type(document):
                document["turns"][0]["input_length"] = "15"

            def mutate_process_start_snapshot(document):
                # Both snapshots are kept internally consistent with each other
                # but deliberately detached from the bound authority process.
                # Rehashing the transcript must not make this acceptable.
                forged = {
                    "pid": 9999,
                    "proc_stat_starttime": "999999",
                }
                document["reference_pass"]["process_start_identity_before"] = dict(forged)
                document["reference_pass"]["process_start_identity_after"] = dict(forged)

            mutations = (
                mutate_mode,
                mutate_status,
                mutate_model_identity,
                mutate_authority,
                mutate_prefix,
                mutate_discriminator,
                mutate_scalar_type,
                mutate_process_start_snapshot,
            )
            for mutate in mutations:
                candidate = json.loads(json.dumps(real_document))
                mutate(candidate)
                candidate_path = root / f"candidate-{mutations.index(mutate)}.json"
                _write_modified_transcript(candidate_path, candidate)
                with self.assertRaises(ValueError):
                    load_transcript(candidate_path)

    def test_dry_run_stays_open(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, events_path = _write_parent(Path(directory))
            plan = build_dry_run_plan(load_frozen_slice(manifest_path, events_path))
            self.assertEqual(plan["materialization_status"], "UNVERIFIED")
            self.assertEqual(plan["materialize_mode"], "OPEN")
            self.assertFalse(plan["formal_replay"])

    def test_real_flag_requires_file_identity(self):
        with self.assertRaises(CalibrationError):
            calibrate_tokenizer(FakeServer(), real_model=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
