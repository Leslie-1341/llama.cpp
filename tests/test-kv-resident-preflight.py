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


class ResidentPreflightParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="resident_preflight_")
        self.root = pathlib.Path(self.temp.name)
        self.workload = {
            "resident_preflight": {
                "sample_interval_seconds": 0.1,
                "min_samples": 10,
                "window_timeout_seconds": 5.0,
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
            "total_bytes": 65536,
            "resident_bytes": 32768 + sample_count,
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
            "server_args": [],
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
                },
                "resident_preflight": {
                    "sample_interval_seconds": 0.1,
                    "min_samples": 10,
                    "window_timeout_seconds": 5.0,
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
        }), encoding="utf-8")
        replay_plan = types.SimpleNamespace(
            schema="gt-trace-1b-a/v1", source_path=str(transcript_path), fixture_contract=None)
        spec = {"model": str(model_path), "binary": str(binary_path)}
        RUNNER.validate_replay_model_binding(replay_plan, spec)
        PARSER.validate_replay_model_binding(
            {"model_sha256": model_sha, "binary_sha256": binary_sha}, model_sha, binary_sha)

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


if __name__ == "__main__":
    unittest.main()
