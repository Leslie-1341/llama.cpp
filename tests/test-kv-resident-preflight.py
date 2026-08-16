#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PARSER = load_module(ROOT / "scripts/parse-kv-offload-benchmark.py", "resident_preflight_parser")
RUNNER = load_module(ROOT / "scripts/run-kv-offload-benchmark.py", "resident_preflight_runner")


RUNTIME_CONTRACT = {
    "ctx_size": 4096,
    "executor": "server",
    "kv_unified": True,
    "parallel": 3,
    "cache_type_k": "f16",
    "cache_type_v": "f16",
    "no_cache_idle_slots": True,
    "no_context_shift": True,
    "paged_block_size": 16,
    "action_target_bytes": 4096,
    "max_blocks": 64,
}


class ResidentPreflightParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="resident_preflight_")
        self.root = pathlib.Path(self.temp.name)
        self.workload = {
            "resident_preflight": {
                "sample_interval_seconds": 0.1,
                "min_samples": 10,
                "window_timeout_seconds": 5.0,
                "max_sample_gap_seconds": 0.5,
                "normalized_resident_spread_bytes": 64,
                "normalized_resident_spread_ratio": 0.0,
                "claimant_relief_spread_bytes": 0,
                "claimant_relief_spread_ratio": 0.0,
            }
        }
        self.record = {
            "mode": "resident_preflight",
            "sample_interval_seconds": 0.1,
            "min_samples": 10,
            "window_timeout_seconds": 5.0,
            "samples_path": "resident_preflight_samples.jsonl",
            "bindings": {
                "A": {"slot_id": 0, "seq_id": 0, "runner_generation": 1},
                "B": {"slot_id": 1, "seq_id": 1, "runner_generation": 1},
                "C": {"slot_id": 2, "seq_id": 2, "runner_generation": 1},
            },
            "windows": {
                "AB_RESIDENT": {
                    "started_mono_ns": 1_000_000_000,
                    "finished_mono_ns": 1_900_000_000,
                    "sample_count": 10,
                },
                "ABC_ACTIVE": {
                    "started_mono_ns": 3_000_000_000,
                    "finished_mono_ns": 3_900_000_000,
                    "dispatch_started_mono_ns": 2_900_000_000,
                    "sample_count": 10,
                },
            },
        }

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _whole(self, timestamp: int, sample_count: int, *, generation: int = 3,
               global_target_enabled: bool = False,
               global_target_source: str = "none") -> dict:
        return {
            "timestamp_mono_ns": timestamp,
            "sample_count": sample_count,
            "whole_valid": True,
            "resident_available": True,
            "reclaimable_available": True,
            "swapped_metadata_consistent": True,
            "object_id": 7,
            "generation": generation,
            "page_size": 4096,
            "native_block_bytes": 2048,
            "total_bytes": 65536,
            "resident_bytes": 32768 + sample_count,
            "dead_resident_reclaimable_bytes": 1024,
            "swapped_authoritative_bytes": 0,
            "transient_staging_bound_bytes": 0,
            "n_blocks": 16,
            "n_owned_blocks": 4,
            "n_shared_blocks": 0,
            "resident_block_count": 8,
            "swapped_block_count": 0,
            "released_block_count": 0,
            "pending_write_block_count": 0,
            "invalid_block_count": 0,
            "unused_block_count": 8,
            "global_target_enabled": global_target_enabled,
            "global_target_source": global_target_source,
            "global_target_bytes": 0 if not global_target_enabled else 1024,
            "global_target_basis_generation": 0 if not global_target_enabled else 1,
            "observation_only": True,
        }

    def _claimant(self, timestamp: int, sample_count: int, seq_id: int, *, active: bool = False,
                  authoritative: bool = True, shared: bool = False) -> dict:
        return {
            "timestamp_mono_ns": timestamp,
            "sample_count": sample_count,
            "seq_id": seq_id,
            "claimant_epoch": 1,
            "active": active,
            "valid": True,
            "available": True,
            "authoritative": authoritative,
            "shared": shared,
            "object_id": 7,
            "generation": 3,
            "exclusive_resident_bytes": 4096 if seq_id in (0, 1) else 0,
            "exclusive_resident_blocks": 1 if seq_id in (0, 1) else 0,
            "swapped_bytes": 0,
            "swapped_blocks": 0,
        }

    def _write_samples(self, *, short: bool = False, drift: bool = False,
                       claimant_bad: bool = False, c_inactive: bool = False,
                       dynamic_target: bool = False) -> pathlib.Path:
        rows = []
        count = 9 if short else 10
        for window, start, base_count in (
                ("AB_RESIDENT", 1_000_000_000, 10),
                ("ABC_ACTIVE", 3_000_000_000, 30)):
            for index in range(count):
                timestamp = start + index * 100_000_000
                sample_count = base_count + index
                whole = self._whole(
                    timestamp,
                    sample_count,
                    generation=4 if drift and window == "ABC_ACTIVE" and index == count - 1 else 3,
                    global_target_enabled=dynamic_target,
                    global_target_source="global_dynamic" if dynamic_target else "none",
                )
                claimants = {
                    "0": self._claimant(timestamp, sample_count, 0,
                                        authoritative=not claimant_bad),
                    "1": self._claimant(timestamp, sample_count, 1,
                                        shared=claimant_bad),
                    "2": self._claimant(
                        timestamp, sample_count, 2,
                        active=window == "ABC_ACTIVE" and not c_inactive),
                }
                rows.append({
                    "window": window,
                    "timestamp_mono_ns": timestamp,
                    "sample_count": sample_count,
                    "whole": whole,
                    "claimants": claimants,
                })
        path = self.root / self.record["samples_path"]
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def _parse(self):
        return PARSER.validate_resident_preflight_samples(
            self.root, self.record, self.workload, "fixture")

    def test_ab_and_abc_windows_pass_and_produce_raw_summaries(self) -> None:
        self._write_samples()
        result = self._parse()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["R_AB"]["n"], 10)
        self.assertEqual(result["R_ABC"]["n"], 10)
        self.assertEqual(result["R_AB"]["min"], 32778)
        self.assertEqual(result["R_AB"]["max"], 32787)
        self.assertEqual(result["normalized_R_AB"]["min"], 31754)
        self.assertEqual(result["normalized_R_ABC"]["max"], 31783)
        self.assertEqual(result["intervals"]["AB_RESIDENT"]["max_seconds"], 0.1)
        self.assertEqual(result["exclusive_estimate"]["combined_resident_bytes_lower_bound"], 8192)

    def test_sample_shortfall_fails_closed(self) -> None:
        self._write_samples(short=True)
        with self.assertRaisesRegex(PARSER.ParseError, "sample count is insufficient"):
            self._parse()

    def test_object_generation_drift_fails_closed(self) -> None:
        self._write_samples(drift=True)
        with self.assertRaisesRegex(PARSER.ParseError, "physical identity drift|object/generation drift"):
            self._parse()

    def test_whole_reclaim_metadata_fails_closed(self) -> None:
        path = self._write_samples()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["whole"]["reclaimable_available"] = False
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PARSER.ParseError, "whole-KV authority"):
            self._parse()

    def test_large_sample_gap_fails_closed(self) -> None:
        path = self._write_samples()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        ab_rows = [row for row in rows if row["window"] == "AB_RESIDENT"]
        for row in ab_rows[5:]:
            row["timestamp_mono_ns"] += 30_000_000_000
            row["whole"]["timestamp_mono_ns"] = row["timestamp_mono_ns"]
            for claimant in row["claimants"].values():
                claimant["timestamp_mono_ns"] = row["timestamp_mono_ns"]
        self.record["windows"]["AB_RESIDENT"]["finished_mono_ns"] += 30_000_000_000
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PARSER.ParseError, "not consecutive"):
            self._parse()

    def test_claimant_authority_or_shared_fails_closed(self) -> None:
        self._write_samples(claimant_bad=True)
        with self.assertRaisesRegex(PARSER.ParseError, "authority is unavailable/shared"):
            self._parse()

    def test_c_must_be_active_in_abc_window(self) -> None:
        self._write_samples(c_inactive=True)
        with self.assertRaisesRegex(PARSER.ParseError, "C is not (?:active|an authoritative active claimant)"):
            self._parse()

    def test_state_change_and_io_fail_closed(self) -> None:
        io = {key: "0" for key in PARSER.IO_REQUIRED}
        io["bytes_written"] = "1"
        with self.assertRaisesRegex(PARSER.ParseError, "swap I/O"):
            PARSER.validate_resident_preflight_action_free(
                "", [], [io], [], [], "fixture")
        with self.assertRaisesRegex(PARSER.ParseError, "unified action marker"):
            PARSER.validate_resident_preflight_action_free(
                "", [{}], [], [], [], "fixture")
        with self.assertRaisesRegex(PARSER.ParseError, "Global/action change"):
            PARSER.validate_resident_preflight_action_free(
                "srv  memory_governor_observe kv_release_attempted=1",
                [], [], [], [], "fixture")

    def test_global_dynamic_target_fails_closed(self) -> None:
        self._write_samples(dynamic_target=True)
        with self.assertRaisesRegex(PARSER.ParseError, "Global dynamic target"):
            self._parse()

    def test_observation_only_environment_is_explicit(self) -> None:
        env = {
            "LLAMA_MEMORY_GOVERNOR": "0",
            "LLAMA_MEMORY_GOVERNOR_OBSERVE": "1",
            "LLAMA_MEMORY_GOVERNOR_OBSERVE_MS": "100",
            "LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS": "0",
            "LLAMA_MEMORY_GOVERNOR_KV_RELEASE": "0",
            "LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD": "0",
            "LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET": "0",
            "LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM": "0",
            "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER": "0",
            "LLAMA_MEMORY_GOVERNOR_REALLOCATION": "0",
            "LLAMA_MEMORY_GOVERNOR_DENSE_REPIN": "0",
            "LLAMA_MEMORY_GOVERNOR_DENSE_RING_SHRINK": "0",
            "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC": "0",
            "LLAMA_MEMORY_GOVERNOR_ASYNC_ACTIONS": "0",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "0",
            "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "0",
            "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "preflight",
            "LLAMA_KV_RESIDENT_PREFLIGHT": "1",
        }
        PARSER.validate_resident_preflight_environment(env, "fixture")
        env["LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD"] = "1"
        with self.assertRaisesRegex(PARSER.ParseError, "observation-only"):
            PARSER.validate_resident_preflight_environment(env, "fixture")

    def test_runner_accepts_only_resident_policy_and_forces_observation_env(self) -> None:
        spec = {
            "schema_version": 2,
            "protocol": "kv_offload_benchmark",
            "phase": "resident_baseline",
            "run_kind": "qualification",
            "run_mode": "resident_preflight",
            "binary": "/tmp/server",
            "model": "/tmp/model.gguf",
            "model_quantization": "synthetic",
            "server_args": [
                "--ctx-size", "4096", "--cache-type-k", "f16", "--cache-type-v", "f16",
                "--kv-unified",
            ],
            "runtime_contract": {
                "ctx_size": 4096, "executor": "server", "kv_unified": True, "parallel": 3,
                "cache_type_k": "f16", "cache_type_v": "f16", "no_cache_idle_slots": True,
                "no_context_shift": True, "paged_block_size": 16, "action_target_bytes": 4096,
                "max_blocks": 64,
            },
            "environment": {},
            "pressure_basis": {
                "authority": "rss_absolute",
                "low_water_kb": 1,
                "pressure_kb": 2,
                "critical_kb": 3,
            },
            "workload": {
                "warmup": [], "requests": [], "repeat": 1,
                "qualification": None, "characterization": None,
                "replay": {
                    "source": "fixture", "path": "/tmp/replay.json",
                    "time_dilation": 1.0, "n_parallel": 3, "session_ids": None,
                    "admission_timeout_seconds": 1.0, "lifecycle": None,
                    "source_lineage_id": 0, "alignment_family_id": "family-q2",
                },
                "resident_preflight": {
                    "sample_interval_seconds": 0.1,
                    "min_samples": 10,
                    "window_timeout_seconds": 5.0,
                "max_sample_gap_seconds": 0.5,
                "normalized_resident_spread_bytes": 64,
                "normalized_resident_spread_ratio": 0.0,
                "claimant_relief_spread_bytes": 0,
                "claimant_relief_spread_ratio": 0.0,
                },
            },
            "cases": [{
                "case_id": "resident",
                "policy": "resident",
                "kv_representation": "paged",
                "loading_mode": "exact",
                "restore": "k1_sync",
                "prefault": "off",
                "kv_target_bytes": None,
                "action_target_bytes": None,
            }],
            "run_order": [{"round": 1, "run_order": 1, "case_id": "resident"}],
            "sampler": {"interval_seconds": 0.1},
            "cgroup": {"expected_memory_max": None},
            "max_blocks": 64,
            "health_timeout_seconds": 1.0,
            "request_timeout_seconds": 1.0,
        }
        normalized, cases, _, workload = RUNNER.validate_spec(spec)
        parser_cases, parser_plan, _ = PARSER.validate_spec(spec)
        self.assertEqual(parser_plan[0]["policy"], "resident")
        self.assertIn("resident", parser_cases)
        self.assertEqual(normalized["run_mode"], "resident_preflight")
        self.assertIsNotNone(workload["resident_preflight"])
        env = RUNNER.runtime_environment(normalized, cases["resident"], self.root)
        self.assertEqual(env["LLAMA_KV_PRESSURE_UNIFIED_ACTION"], "0")
        self.assertEqual(env["LLAMA_MEMORY_GOVERNOR_OBSERVE_MS"], "100")
        self.assertEqual(env["LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS"], "0")
        self.assertEqual(env["LLAMA_KV_RESIDENT_PREFLIGHT"], "1")
        self.assertEqual(env["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"], "preflight")
        ordinary_spec = dict(normalized)
        ordinary_spec["run_mode"] = "qualification"
        ordinary_case = dict(cases["resident"])
        ordinary_env = RUNNER.runtime_environment(ordinary_spec, ordinary_case, self.root)
        self.assertNotIn("LLAMA_KV_RESIDENT_PREFLIGHT", ordinary_env)
        spec["workload"]["resident_preflight"]["window_timeout_seconds"] = 0.1
        with self.assertRaisesRegex(RUNNER.RunnerError, "too short"):
            RUNNER.validate_spec(spec)

    def test_q2_switch_is_required_and_non_q2_switch_is_rejected(self) -> None:
        env = {
            "LLAMA_MEMORY_GOVERNOR": "0",
            "LLAMA_MEMORY_GOVERNOR_OBSERVE": "1",
            "LLAMA_MEMORY_GOVERNOR_OBSERVE_MS": "100",
            "LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS": "0",
            "LLAMA_MEMORY_GOVERNOR_KV_RELEASE": "0",
            "LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD": "0",
            "LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET": "0",
            "LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM": "0",
            "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER": "0",
            "LLAMA_MEMORY_GOVERNOR_REALLOCATION": "0",
            "LLAMA_MEMORY_GOVERNOR_DENSE_REPIN": "0",
            "LLAMA_MEMORY_GOVERNOR_DENSE_RING_SHRINK": "0",
            "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC": "0",
            "LLAMA_MEMORY_GOVERNOR_ASYNC_ACTIONS": "0",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "0",
            "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "0",
            "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "preflight",
            "LLAMA_KV_RESIDENT_PREFLIGHT": "1",
        }
        PARSER.validate_resident_preflight_environment(env, "fixture")
        env.pop("LLAMA_KV_RESIDENT_PREFLIGHT")
        with self.assertRaisesRegex(PARSER.ParseError, "explicit switch"):
            PARSER.validate_resident_preflight_environment(env, "fixture")

    def test_model_bound_replay_binds_model_and_binary(self) -> None:
        model_path = self.root / "model.bin"
        binary_path = self.root / "server.bin"
        model_path.write_bytes(b"model-v1")
        binary_path.write_bytes(b"binary-v1")
        model_sha = RUNNER.sha256_file(model_path)
        binary_sha = RUNNER.sha256_file(binary_path)
        transcript_path = self.root / "transcript.json"
        transcript_path.write_text(json.dumps({
            "schema": "gt-trace-1b-a/v1",
            "model": {"model_sha256": model_sha, "binary_sha256": binary_sha},
            "runtime_contract": RUNTIME_CONTRACT,
        }), encoding="utf-8")
        replay_plan = types.SimpleNamespace(
            schema="gt-trace-1b-a/v1", source_path=str(transcript_path), fixture_contract=None)
        spec = {
            "model": str(model_path), "binary": str(binary_path),
            "runtime_contract": RUNTIME_CONTRACT,
        }
        RUNNER.validate_replay_model_binding(replay_plan, spec)
        PARSER.validate_replay_model_binding(
            {"model_sha256": model_sha, "binary_sha256": binary_sha,
             "runtime_contract": RUNTIME_CONTRACT},
            model_sha, binary_sha, RUNTIME_CONTRACT)

    def test_model_bound_replay_rejects_stale_binary(self) -> None:
        model_path = self.root / "model.bin"
        binary_path = self.root / "server.bin"
        model_path.write_bytes(b"model-v1")
        binary_path.write_bytes(b"binary-v2")
        model_sha = RUNNER.sha256_file(model_path)
        stale_binary_sha = "0" * 64
        transcript_path = self.root / "transcript.json"
        transcript_path.write_text(json.dumps({
            "schema": "gt-trace-1b-a/v1",
            "model": {"model_sha256": model_sha, "binary_sha256": stale_binary_sha},
        }), encoding="utf-8")
        replay_plan = types.SimpleNamespace(
            schema="gt-trace-1b-a/v1", source_path=str(transcript_path), fixture_contract=None)
        with self.assertRaisesRegex(RUNNER.RunnerError, "binary bytes"):
            RUNNER.validate_replay_model_binding(
                replay_plan, {"model": str(model_path), "binary": str(binary_path)})
        with self.assertRaisesRegex(PARSER.ParseError, "binary SHA"):
            PARSER.validate_replay_model_binding(
                {"model_sha256": model_sha, "binary_sha256": stale_binary_sha},
                model_sha, RUNNER.sha256_file(binary_path))


    def test_target_freeze_record_persists_and_rejects_fixture_drift(self) -> None:
        replay_path = self.root / "replay.json"
        parent_path = self.root / "parent.json"
        fixture_path = self.root / "fixture.jsonl"
        replay_path.write_text("replay\n", encoding="utf-8")
        parent_path.write_text("parent\n", encoding="utf-8")
        fixture_path.write_text("fixture\n", encoding="utf-8")
        runtime_contract = {
            "ctx_size": 4096, "executor": "server", "kv_unified": True, "parallel": 3,
            "cache_type_k": "f16", "cache_type_v": "f16", "no_cache_idle_slots": True,
            "no_context_shift": True, "paged_block_size": 16, "action_target_bytes": 4096,
            "max_blocks": 64,
        }
        preflight_identity = {
            "path": str(replay_path), "present": True,
            "size": replay_path.stat().st_size, "sha256": PARSER.sha256_file(replay_path),
        }
        evidence_identity = {
            "path": str(fixture_path), "present": True,
            "size": fixture_path.stat().st_size, "sha256": PARSER.sha256_file(fixture_path),
        }
        parent_identity = {
            "path": str(parent_path), "present": True,
            "size": parent_path.stat().st_size, "sha256": PARSER.sha256_file(parent_path),
        }
        target = {
            "status": "FROZEN", "frozen": True, "frozen_T": 8192,
            "identity": {
                "transcript_sha256": preflight_identity["sha256"],
                "fixture_sha256": evidence_identity["sha256"],
                "transcript_identity": preflight_identity,
                "fixture_identity": evidence_identity,
                "parent_transcript_identity": parent_identity,
                "preflight_fixture_identity": preflight_identity,
                "q2_evidence_identity": evidence_identity,
                "source_lineage_id": 0,
                "alignment_family_id": "family-q2",
                "runtime_contract": runtime_contract,
            },
        }
        PARSER.write_target_freeze_record(
            self.root, {"verdict": "RESIDENT_PREFLIGHT_PASS", "target_freeze": target})
        freeze_path = self.root / "target-freeze.json"
        record = json.loads(freeze_path.read_text(encoding="utf-8"))
        reference = {
            "path": str(freeze_path), "present": True,
            "size": freeze_path.stat().st_size, "sha256": PARSER.sha256_file(freeze_path),
        }
        loaded = PARSER.load_target_freeze_record(
            reference, runtime_contract, {"replay": {"path": str(replay_path)}}, "fixture")
        self.assertEqual(loaded["frozen_T"], 8192)
        fixture_path.write_text("drifted fixture\n", encoding="utf-8")
        with self.assertRaisesRegex(PARSER.ParseError, "fixture_identity|q2_evidence_identity"):
            PARSER.load_target_freeze_record(
                reference, runtime_contract, {"replay": {"path": str(replay_path)}}, "fixture")

    def test_multi_round_target_freeze_aggregates_and_round_trips(self) -> None:
        replay_path = self.root / "replay-multi.json"
        parent_path = self.root / "parent-multi.json"
        fixture_path = self.root / "fixture-multi.jsonl"
        replay_path.write_text("replay\n", encoding="utf-8")
        parent_path.write_text("parent\n", encoding="utf-8")
        fixture_path.write_text("fixture\n", encoding="utf-8")
        runtime_contract = dict(RUNTIME_CONTRACT)
        replay_identity = {
            "path": str(replay_path), "present": True,
            "size": replay_path.stat().st_size,
            "sha256": PARSER.sha256_file(replay_path),
        }
        evidence_identity = {
            "path": str(fixture_path), "present": True,
            "size": fixture_path.stat().st_size,
            "sha256": PARSER.sha256_file(fixture_path),
        }
        parent_identity = {
            "path": str(parent_path), "present": True,
            "size": parent_path.stat().st_size,
            "sha256": PARSER.sha256_file(parent_path),
        }
        identity = {
            "transcript_sha256": replay_identity["sha256"],
            "fixture_sha256": evidence_identity["sha256"],
            "transcript_identity": replay_identity,
            "fixture_identity": evidence_identity,
            "parent_transcript_identity": parent_identity,
            "preflight_fixture_identity": replay_identity,
            "q2_evidence_identity": evidence_identity,
            "source_lineage_id": 0,
            "alignment_family_id": "family-q2-multi",
            "runtime_contract": runtime_contract,
        }
        round_one = {"status": "FROZEN", "frozen": True, "frozen_T": 8192,
                     "identity": identity}
        round_two = json.loads(json.dumps(round_one))
        aggregate = PARSER.aggregate_target_freeze_records([round_one, round_two])
        self.assertEqual(aggregate["round_count"], 2)
        self.assertIsInstance(aggregate["identity"], dict)
        PARSER.write_target_freeze_record(
            self.root, {"verdict": "RESIDENT_PREFLIGHT_PASS", "target_freeze": aggregate})
        freeze_path = self.root / "target-freeze.json"
        reference = {
            "path": str(freeze_path), "present": True,
            "size": freeze_path.stat().st_size,
            "sha256": PARSER.sha256_file(freeze_path),
        }
        loaded = PARSER.load_target_freeze_record(
            reference, runtime_contract, {"replay": {"path": str(replay_path)}}, "multi")
        self.assertEqual(loaded["frozen_T"], 8192)
        self.assertEqual(loaded["round_count"], 2)
        drifted = json.loads(json.dumps(round_one))
        drifted["identity"]["source_lineage_id"] = 1
        with self.assertRaisesRegex(PARSER.ParseError, "identity"):
            PARSER.aggregate_target_freeze_records([round_one, drifted])

    def test_target_freeze_binding_rejects_manual_override_and_injects_cap(self) -> None:
        runtime_contract = dict(RUNTIME_CONTRACT)
        record = {"frozen_T": 8192}
        base = {
            "case_id": "v2", "policy": "v2", "kv_target_bytes": None,
            "action_target_bytes": None,
        }
        cases = {"v2": dict(base)}
        PARSER.bind_target_freeze_cases(cases, record, runtime_contract)
        self.assertEqual(cases["v2"]["kv_target_bytes"], 8192)
        self.assertEqual(cases["v2"]["action_target_bytes"], 4096)
        cases["v2"]["kv_target_bytes"] = 123
        with self.assertRaisesRegex(PARSER.ParseError, "forbids manual"):
            PARSER.bind_target_freeze_cases(cases, record, runtime_contract)

    def test_target_freeze_uses_normalized_bounds_and_alignment(self) -> None:
        preflight = {
            "normalized_R_AB": {"max": 16384, "allowed_spread": 0},
            "normalized_R_ABC": {"min": 24576, "max": 24576, "allowed_spread": 0},
            "Relief_A": {"resident_bytes_lower_bound": 4096, "resident_blocks_lower_bound": 2},
            "Relief_B": {"resident_bytes_lower_bound": 8192, "resident_blocks_lower_bound": 4},
            "native_block_bytes": 2048,
        }
        record = PARSER.freeze_resident_preflight_target(
            preflight, RUNTIME_CONTRACT)
        self.assertEqual(record["status"], "FROZEN")
        self.assertTrue(record["frozen"])
        self.assertEqual(record["frozen_T"], 22528)
        self.assertEqual(record["safe_interval"], {"lower": 20480, "upper": 22528})
        self.assertEqual(record["inputs"]["AB_hi"], 16384)
        self.assertEqual(record["inputs"]["Relief_lower_bound"], 4096)

    def test_target_freeze_reports_unavailable_window(self) -> None:
        preflight = {
            "normalized_R_AB": {"max": 30000, "allowed_spread": 0},
            "normalized_R_ABC": {"min": 30001, "max": 30001, "allowed_spread": 0},
            "Relief_A": {"resident_bytes_lower_bound": 4096, "resident_blocks_lower_bound": 2},
            "Relief_B": {"resident_bytes_lower_bound": 4096, "resident_blocks_lower_bound": 2},
            "native_block_bytes": 2048,
        }
        with self.assertRaisesRegex(PARSER.ParseError, "TARGET_WINDOW_UNAVAILABLE"):
            PARSER.freeze_resident_preflight_target(
                preflight, RUNTIME_CONTRACT)


    def test_production_resident_marker_schema_is_exact_and_native_block_bytes_passes(self) -> None:
        whole = self._whole(1_000_000_000, 10)
        token = "kv_resident_preflight_observation"
        line = "srv " + token + " " + " ".join(
            f"{key}={int(value) if isinstance(value, bool) else value}"
            for key, value in whole.items())
        fields = RUNNER.parse_marker_fields(line, token)
        self.assertIsNotNone(fields)
        assert fields is not None
        required = set(fields)
        numeric = required - {
            "whole_valid", "resident_available", "reclaimable_available",
            "swapped_metadata_consistent", "global_target_enabled", "observation_only",
            "global_target_source",
        }
        booleans = {
            "whole_valid", "resident_available", "reclaimable_available",
            "swapped_metadata_consistent", "global_target_enabled", "observation_only",
        }
        decoded = RUNNER._resident_preflight_decode_marker(
            fields, numeric, booleans, required, "resident observation")
        self.assertEqual(decoded["native_block_bytes"], 2048)
        self.assertEqual(
            PARSER.parse_fields(
                line, token, required, "fixture.marker"),
            fields,
        )

        missing = dict(fields)
        missing.pop("native_block_bytes")
        with self.assertRaisesRegex(RUNNER.RunnerError, "schema mismatch"):
            RUNNER._resident_preflight_decode_marker(
                missing, numeric, booleans, required, "resident observation")
        duplicate_line = line + " native_block_bytes=2048"
        self.assertIsNone(RUNNER.parse_marker_fields(duplicate_line, token))
        extra = dict(fields)
        extra["unexpected"] = "1"
        with self.assertRaisesRegex(RUNNER.RunnerError, "schema mismatch"):
            RUNNER._resident_preflight_decode_marker(
                extra, numeric, booleans, required, "resident observation")
        with self.assertRaisesRegex(PARSER.ParseError, "schema mismatch"):
            PARSER.parse_fields(
                line + " unexpected=1", token,
                required, "fixture.marker")

    def test_completion_live_kv_authority_requires_real_aligned_exclusive_view(self) -> None:
        event = {"expected_live_kv_cells": 32}
        actual = {
            "live_kv_pos_min": 10,
            "live_kv_pos_max": 100,
            "live_kv_cells": 32,
            "live_kv_blocks": 2,
            "live_kv_object_id": 7,
            "live_kv_generation": 3,
            "live_kv_block_aligned": True,
            "live_kv_authoritative": True,
            "live_kv_shared": False,
        }
        RUNNER.validate_resident_prepare_footprint(
            event, actual, RUNTIME_CONTRACT, "prepare-A")
        for field, value in (
                ("live_kv_block_aligned", False),
                ("live_kv_authoritative", False),
                ("live_kv_shared", True)):
            bad = dict(actual)
            bad[field] = value
            with self.assertRaisesRegex(RUNNER.RunnerError, "authority is invalid"):
                RUNNER.validate_resident_prepare_footprint(
                    event, bad, RUNTIME_CONTRACT, f"bad-{field}")
        bad = dict(actual, live_kv_cells=31)
        with self.assertRaisesRegex(RUNNER.RunnerError, "do not fill"):
            RUNNER.validate_resident_prepare_footprint(
                event, bad, RUNTIME_CONTRACT, "bad-cell-alignment")
        bad = dict(actual, live_kv_blocks=1)
        with self.assertRaisesRegex(RUNNER.RunnerError, "do not fill"):
            RUNNER.validate_resident_prepare_footprint(
                event, bad, RUNTIME_CONTRACT, "bad-block-count")

    def test_target_freeze_uses_max_blocks_in_single_action_cap(self) -> None:
        preflight = {
            "normalized_R_AB": {"max": 16384, "allowed_spread": 0},
            "normalized_R_ABC": {"min": 32768, "max": 32768, "allowed_spread": 0},
            "Relief_A": {"resident_bytes_lower_bound": 65536, "resident_blocks_lower_bound": 32},
            "Relief_B": {"resident_bytes_lower_bound": 65536, "resident_blocks_lower_bound": 32},
            "native_block_bytes": 2048,
        }
        limited = dict(RUNTIME_CONTRACT, action_target_bytes=65536, max_blocks=2)
        record = PARSER.freeze_resident_preflight_target(preflight, limited)
        self.assertEqual(record["inputs"]["max_blocks_cap"], 4096)
        self.assertEqual(record["inputs"]["single_action_relief_cap"], 4096)
        self.assertEqual(record["frozen_T"], record["safe_interval"]["upper"])
        uncapped = PARSER.freeze_resident_preflight_target(
            preflight, dict(limited, max_blocks=64))
        self.assertGreater(
            uncapped["inputs"]["single_action_relief_cap"],
            record["inputs"]["single_action_relief_cap"],
        )


if __name__ == "__main__":
    unittest.main()
