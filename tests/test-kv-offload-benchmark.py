#!/usr/bin/env python3
"""Synthetic tests for the canonical Formal OFFLOAD Benchmark evidence spine."""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run-kv-offload-benchmark.py"
PARSER = ROOT / "scripts" / "parse-kv-offload-benchmark.py"


def load_runner_module():
    module_spec = importlib.util.spec_from_file_location("kv_offload_runner_test_module", RUNNER)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load benchmark runner")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_parser_module():
    module_spec = importlib.util.spec_from_file_location("kv_offload_parser_test_module", PARSER)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load benchmark parser")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


ACTION_FIELDS = {
    "state": "NORMAL", "source": "RSS_ABSOLUTE", "sample_valid": "1", "stale": "0", "pressure_basis_valid": "1",
    "decision_id": "1", "episode": "0",
    "target_bytes": "4096", "max_blocks": "64", "observed_excess_bytes": "4096", "debt_before_bytes": "4096",
    "debt_after_bytes": "0", "budget_active": "1", "budget_target_enabled": "1", "budget_source": "env_static",
    "budget_target_bytes": "4096", "budget_basis_generation": "1", "budget_view_valid": "1",
    "budget_resident_available": "1", "budget_reclaimable_available": "1", "budget_resident_bytes": "8192",
    "budget_dead_resident_reclaimable_bytes": "0", "budget_transient_staging_bound_bytes": "0",
    "budget_observed_excess_bytes": "4096", "budget_debt_before_bytes": "4096", "budget_debt_after_bytes": "0",
    "soft_offload_armed_before": "1", "soft_offload_armed_after": "0", "budget_next_action_sample": "2",
    "unmet_budget_bytes_after": "0", "offload_armed_before": "1", "offload_armed_after": "0",
    "next_action_sample": "2", "evaluate_attempted": "1", "evaluate_outcome": "completed",
    "evaluate_reason": "none", "release_attempted": "0", "offload_attempted": "1", "selected_seq_id": "0",
    "selected_claimant_epoch": "1", "transaction_id": "7", "outcome": "completed", "reason": "none",
    "blocks": "1", "bytes": "4096", "relieved_bytes": "4096", "shortfall_bytes": "0", "io_failure": "0",
    "io_errno": "0", "state_changed": "1", "decision_reason": "budget_excess", "sample_count": "1", "idle": "1",
    "claimants": "0:1:0:0:1:1:1:0:0:0", "scores": "none",
}
IO_FIELDS = {
    "block_swap_out_calls": "1", "block_swap_in_calls": "1", "backing_read_syscalls": "1", "backing_write_syscalls": "1",
    "bytes_read": "4096", "bytes_written": "4096", "staging_buffer_bytes": "4096", "k2_enabled": "0",
    "k2_staging_bound_bytes": "0", "k2_peak_staging_groups": "0", "k2_peak_staging_bytes": "0", "k2_pipeline_wall_us": "0",
    "k2_exposed_read_wait_us": "0", "k2_pipeline_stall_us": "0", "k2_read_completed_ahead": "0",
    "restore_prefault_enabled": "0", "restore_prefault_groups": "0", "restore_prefault_calls": "0", "restore_prefault_us": "0",
    "restore_prefault_minor_faults": "0", "restore_prefault_major_faults": "0", "restore_scatter_groups": "1",
    "restore_scatter_us": "10", "restore_scatter_fault_groups": "0", "restore_scatter_minor_faults": "0",
    "restore_scatter_major_faults": "0",
}


def marker(token: str, fields: dict[str, str]) -> str:
    return token + " " + " ".join(f"{key}={value}" for key, value in fields.items())


class CanonicalBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="kv_offload_benchmark_")
        self.root = pathlib.Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"synthetic-model")
        self.fake_server = self.root / "fake-server.py"
        self.fake_server.write_text(textwrap.dedent(f"""
            #!/usr/bin/env python3
            import argparse, json, os, signal, sys, threading, time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            offload_enabled = os.environ.get('LLAMA_KV_PAGED_SWAP') == '1'
            mode = os.environ.get('KV_SYNTHETIC_MODE', 'complete')
            expose_v2_resident = os.environ.get('KV_SYNTHETIC_V2_SLOTS_RESIDENT') == '1'
            action_fields = {ACTION_FIELDS!r}
            io_fields = {IO_FIELDS!r}
            state = {{'requests': 0, 'offload_emitted': False, 'prefetch_completed': False}}
            output_lock = threading.Lock()

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--port', type=int, required=True)
            parser.add_argument('--host', default='127.0.0.1')
            args, _ = parser.parse_known_args()

            def emit_marker(token, fields):
                with output_lock:
                    print(token + " " + " ".join(f"{{key}}={{value}}" for key, value in fields.items()), file=sys.stderr, flush=True)

            def emit_offload_after_fill():
                time.sleep(0.08)
                if not offload_enabled or mode == 'no_offload':
                    return
                observation = {{
                    'source': 'paged_sample_mincore', 'action': 'offload',
                    'decision_id': '1', 'seq_id': '0', 'transaction_id': '7', 'server_pid': str(os.getpid()),
                    'before_available': '1', 'before_object_id': '1', 'before_generation': '1',
                    'before_page_size': '4096', 'before_total_bytes': '8192', 'before_resident_bytes': '8192',
                    'before_total_pages': '2', 'before_resident_pages': '2',
                    'after_available': '1', 'after_object_id': '1', 'after_generation': '1',
                    'after_page_size': '4096', 'after_total_bytes': '8192', 'after_resident_bytes': '4096',
                    'after_total_pages': '2', 'after_resident_pages': '1',
                }}
                if mode == 'mismatch_decision':
                    observation['decision_id'] = '2'
                elif mode == 'mismatch_transaction':
                    observation['transaction_id'] = '8'
                elif mode == 'mismatch_seq':
                    observation['seq_id'] = '1'
                elif mode == 'drop_zero':
                    observation['after_resident_bytes'] = observation['before_resident_bytes']
                    observation['after_resident_pages'] = observation['before_resident_pages']
                if mode != 'action_only':
                    emit_marker('kv_g0_s1_resident_observation', observation)
                emit_marker('kv_pressure_unified_action', action_fields)
                state['offload_emitted'] = True

            def emit_resume_evidence():
                if mode == 'prefetch_noop':
                    outcome = 'no_op'
                    restored_blocks = '0'
                    restored_bytes = '0'
                    total_us = '0'
                else:
                    outcome = 'completed'
                    restored_blocks = '1'
                    restored_bytes = '4096'
                    total_us = '6'
                    state['prefetch_completed'] = True
                base = {{
                    'decision_id': '2', 'seq_id': '0', 'claimant_epoch': '1', 'transaction_id': '8',
                    'action': 'prefetch', 'outcome': outcome, 'reason': 'none', 'graph_allowed': '1',
                }}
                emit_marker('kv_resume_order_event', {{'phase': 'prefetch', **base}})
                emit_marker('kv_resume_order_event', {{'phase': 'graph_gate', **base}})
                emit_marker('kv_resume_stage_timing', {{
                    'decision_id': '2', 'seq_id': '0', 'transaction_id': '8',
                    'restored_blocks': restored_blocks, 'restored_bytes': restored_bytes,
                    'queue_us': '1', 'gate_us': '2', 'graph_us': '3', 'total_us': total_us,
                }})

            def stop(_signum, _frame):
                values = dict(io_fields)
                for key in ('block_swap_out_calls', 'block_swap_in_calls', 'backing_read_syscalls', 'backing_write_syscalls', 'bytes_read', 'bytes_written'):
                    values[key] = '0'
                if offload_enabled and state['offload_emitted']:
                    values['block_swap_out_calls'] = '1'
                    values['backing_write_syscalls'] = '1'
                    values['bytes_written'] = '4096'
                if offload_enabled and state['prefetch_completed'] and mode != 'swap_in_zero':
                    values['block_swap_in_calls'] = '1'
                    values['backing_read_syscalls'] = '1'
                    values['bytes_read'] = '4096'
                emit_marker('KV_PAGED_IO_STATS', values)
                raise SystemExit(0)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def do_GET(self):
                    if self.path == '/health':
                        self.send_response(200); self.end_headers(); self.wfile.write(b'{{"status":"ok"}}'); return
                    if self.path == '/slots':
                        slot = {{'id': 0, 'is_processing': False}}
                        if not offload_enabled or expose_v2_resident:
                            slot['kv_resident'] = {{
                                'status': 'available', 'source': 'synthetic', 'object_id': 1, 'generation': 1,
                                'page_size': 4096, 'total_bytes': 8192, 'resident_bytes': 8192,
                                'total_pages': 2, 'resident_pages': 2,
                            }}
                        encoded = json.dumps([slot]).encode()
                        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length', str(len(encoded))); self.end_headers(); self.wfile.write(encoded); return
                    self.send_response(404); self.end_headers()

                def do_POST(self):
                    length = int(self.headers.get('Content-Length', '0'))
                    self.rfile.read(length)
                    state['requests'] += 1
                    ordinal = state['requests']
                    print(f"synthetic_request ordinal={{ordinal}}", flush=True)
                    if offload_enabled and ordinal == 2:
                        threading.Thread(target=emit_offload_after_fill, daemon=True).start()
                    elif offload_enabled and ordinal == 3:
                        emit_resume_evidence()
                    time.sleep(0.02)
                    payload = {{"content":"ok", "timings":{{"predicted_n":2,"predicted_ms":4}}}}
                    encoded = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(encoded)))
                    self.end_headers(); self.wfile.write(encoded)

            signal.signal(signal.SIGTERM, stop)
            server = ThreadingHTTPServer((args.host, args.port), Handler)
            server.daemon_threads = True
            server.serve_forever()
        """).strip() + "\n", encoding="utf-8")
        self.fake_server.chmod(self.fake_server.stat().st_mode | stat.S_IXUSR)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def spec(self, *, policy: str = "v2") -> dict[str, object]:
        target = 4096 if policy == "v2" else None
        return {
            "schema_version": 1,
            "protocol": "kv_offload_benchmark",
            "phase": "coarse_target",
            "run_kind": "qualification",
            "binary": str(self.fake_server),
            "model": str(self.model),
            "model_quantization": "synthetic",
            "server_args": [],
            "environment": {"KV_SYNTHETIC_MODE": "complete"},
            "pressure_basis": {
                "authority": "rss_absolute",
                "low_water_kb": 64 * 1024 * 1024,
                "pressure_kb": 128 * 1024 * 1024,
                "critical_kb": 256 * 1024 * 1024,
            },
            "workload": {
                "warmup": [{"request_id": "seed", "prompt": "seed", "n_predict": 1, "stream": False}],
                "requests": [{"request_id": "request", "prompt": "hello", "n_predict": 2, "stream": False}],
                "repeat": 1,
                "qualification": {
                    "idle_seconds": 0.01,
                    "offload_timeout_seconds": 1.0,
                    "resume_request_id": "request",
                },
            },
            "cases": [{
                "case_id": "case",
                "policy": policy,
                "kv_representation": "paged",
                "loading_mode": "exact",
                "restore": "k1_sync",
                "prefault": "off",
                "kv_target_bytes": target,
            }],
            "run_order": [{"round": 1, "run_order": 1, "case_id": "case"}],
            "sampler": {"interval_seconds": 1.0},
            "cgroup": {"expected_memory_max": None},
            "max_blocks": 64,
            "health_timeout_seconds": 10,
            "request_timeout_seconds": 10,
        }

    def write_spec(self, value: dict[str, object], name: str = "spec.json") -> pathlib.Path:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def run_runner(self, spec: pathlib.Path, artifact: pathlib.Path, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(RUNNER), "--spec", str(spec), "--output", str(artifact)]
        if dry_run:
            command.append("--dry-run")
        return subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONUTF8": "1"})

    def run_parser(self, artifact: pathlib.Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PARSER), str(artifact)],
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONUTF8": "1"},
        )

    def run_real_artifact(
            self,
            *,
            policy: str = "v2",
            mode: str = "complete",
            expose_v2_resident: bool = False,
    ) -> pathlib.Path:
        value = self.spec(policy=policy)
        value["environment"]["KV_SYNTHETIC_MODE"] = mode
        if expose_v2_resident:
            value["environment"]["KV_SYNTHETIC_V2_SLOTS_RESIDENT"] = "1"
        spec = self.write_spec(value)
        artifact = self.root / f"real-{policy}-{mode}-{len(list(self.root.glob('real-*')))}"
        runner = self.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        return artifact

    def run_incomplete_artifact(self, mode: str) -> tuple[pathlib.Path, subprocess.CompletedProcess[str]]:
        value = self.spec()
        value["environment"]["KV_SYNTHETIC_MODE"] = mode
        value["workload"]["qualification"]["offload_timeout_seconds"] = 0.2
        spec = self.write_spec(value, f"{mode}.json")
        artifact = self.root / f"incomplete-{mode}"
        return artifact, self.run_runner(spec, artifact)

    def test_dry_run_parser_and_runner_do_not_write_verdict(self) -> None:
        spec = self.write_spec(self.spec())
        artifact = self.root / "dry-run"
        runner = self.run_runner(spec, artifact, dry_run=True)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("verdict", json.dumps(manifest))
        self.assertFalse((artifact / "result.json").exists())
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "DRY_RUN")

    def test_v0_rejects_slo_facade_and_canonical_overrides(self) -> None:
        for mutation in (
            lambda value: value["workload"].update({"slo": {"ttft_ms": 1, "tpot_ms": 1, "attainment": 1}}),
            lambda value: value.update({"server_args": ["--port=9"]}),
            lambda value: value.update({"environment": {"LLAMA_KV_PAGED_SWAP": "0"}}),
        ):
            value = self.spec()
            mutation(value)
            spec = self.write_spec(value)
            artifact = self.root / "rejected"
            result = self.run_runner(spec, artifact, dry_run=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(artifact.exists())

    def test_nonfinite_timeout_and_interval_are_rejected(self) -> None:
        for field in ("health_timeout_seconds", "request_timeout_seconds"):
            value = self.spec()
            value[field] = float("nan")
            result = self.run_runner(self.write_spec(value), self.root / field, dry_run=True)
            self.assertEqual(result.returncode, 2, result.stderr)
        value = self.spec()
        value["sampler"]["interval_seconds"] = float("inf")
        result = self.run_runner(self.write_spec(value), self.root / "interval", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        value = self.spec()
        value["workload"]["qualification"]["offload_timeout_seconds"] = float("nan")
        result = self.run_runner(self.write_spec(value), self.root / "offload-timeout", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_resume_request_id_must_reference_measurement_request(self) -> None:
        value = self.spec()
        value["workload"]["qualification"]["resume_request_id"] = "seed"
        result = self.run_runner(self.write_spec(value), self.root / "bad-resume-id", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_v2_requires_positive_successful_offload_evidence(self) -> None:
        artifact = self.run_real_artifact()
        stderr = next(artifact.glob("runs/*/server.stderr"))
        stderr.write_text(stderr.read_text(encoding="utf-8").replace("offload_attempted=1", "offload_attempted=0"), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("matching successful action", result["errors"][0])

    def test_action_selection_identity_uses_contextual_seq_sentinel(self) -> None:
        parser = load_parser_module()

        def validate(action: dict[str, str]) -> None:
            fields = parser.marker_records(
                marker("kv_pressure_unified_action", action),
                "kv_pressure_unified_action",
                parser.ACTION_REQUIRED,
                "action",
            )[0]
            parser.validate_action_fields(fields, "action")

        for selected_seq_id in ("0", "7"):
            action = dict(ACTION_FIELDS)
            action["selected_seq_id"] = selected_seq_id
            validate(action)

        invalid = dict(ACTION_FIELDS)
        invalid["selected_seq_id"] = "-1"
        with self.assertRaisesRegex(parser.ParseError, "state-changing OFFLOAD"):
            validate(invalid)

        for field in ("selected_claimant_epoch", "transaction_id"):
            invalid = dict(ACTION_FIELDS)
            invalid[field] = "0"
            with self.subTest(field=field):
                with self.assertRaisesRegex(parser.ParseError, "positive"):
                    validate(invalid)

        noop = dict(ACTION_FIELDS)
        noop.update({
            "offload_attempted": "0", "release_attempted": "0", "outcome": "no_op",
            "state_changed": "0", "selected_seq_id": "-1", "selected_claimant_epoch": "0",
            "transaction_id": "0", "blocks": "0", "bytes": "0", "relieved_bytes": "0",
            "shortfall_bytes": "0",
        })
        validate(noop)

    def test_v2_requires_normal_valid_basis_and_budget_excess(self) -> None:
        artifact = self.run_real_artifact()
        stderr = next(artifact.glob("runs/*/server.stderr"))
        text = stderr.read_text(encoding="utf-8")
        self.assertIn("state=NORMAL", text)
        self.assertIn("sample_valid=1", text)
        self.assertIn("pressure_basis_valid=1", text)
        self.assertIn("budget_observed_excess_bytes=4096", text)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)

        for field, replacement in (
            ("pressure_basis_valid=1", "pressure_basis_valid=0"),
            ("budget_observed_excess_bytes=4096", "budget_observed_excess_bytes=0"),
            ("state=NORMAL", "state=PRESSURE"),
        ):
            with self.subTest(field=field):
                invalid = self.run_real_artifact()
                invalid_stderr = next(invalid.glob("runs/*/server.stderr"))
                invalid_text = invalid_stderr.read_text(encoding="utf-8")
                self.assertIn(field, invalid_text)
                invalid_stderr.write_text(invalid_text.replace(field, replacement, 1), encoding="utf-8")
                invalid_result = self.run_parser(invalid)
                self.assertNotEqual(invalid_result.returncode, 0)
                verdict = json.loads((invalid / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(verdict["verdict"], "INVALID_ARTIFACT")

    def test_offload_barrier_enforces_production_action_contract(self) -> None:
        runner = load_runner_module()
        parser = load_parser_module()
        observation = {
            "source": "paged_sample_mincore", "action": "offload", "decision_id": "1",
            "seq_id": "0", "transaction_id": "7", "server_pid": "123",
            "before_available": "1", "before_object_id": "1", "before_generation": "1",
            "before_page_size": "4096", "before_total_bytes": "8192",
            "before_resident_bytes": "8192", "before_total_pages": "2", "before_resident_pages": "2",
            "after_available": "1", "after_object_id": "1", "after_generation": "1",
            "after_page_size": "4096", "after_total_bytes": "8192",
            "after_resident_bytes": "4096", "after_total_pages": "2", "after_resident_pages": "1",
        }
        text = marker("kv_g0_s1_resident_observation", observation) + "\n" \
            + marker("kv_pressure_unified_action", ACTION_FIELDS) + "\n"
        self.assertIsNotNone(runner.find_offload_barrier_pair(text, "RSS_ABSOLUTE"))
        self.assertEqual(
            len(parser.qualified_offload_pairs([ACTION_FIELDS], [observation], "RSS_ABSOLUTE")), 1)
        for field, invalid in (
            ("state", "PRESSURE"),
            ("source", "CGROUP_RATIO"),
            ("sample_valid", "0"),
            ("stale", "1"),
            ("pressure_basis_valid", "0"),
            ("budget_active", "0"),
            ("budget_observed_excess_bytes", "0"),
            ("offload_attempted", "0"),
            ("outcome", "no_op"),
            ("state_changed", "0"),
            ("blocks", "0"),
            ("bytes", "0"),
            ("relieved_bytes", "0"),
            ("shortfall_bytes", "1"),
            ("io_failure", "1"),
        ):
            with self.subTest(field=field):
                action = dict(ACTION_FIELDS)
                action[field] = invalid
                invalid_text = marker("kv_g0_s1_resident_observation", observation) + "\n" \
                    + marker("kv_pressure_unified_action", action) + "\n"
                self.assertIsNone(runner.find_offload_barrier_pair(invalid_text, "RSS_ABSOLUTE"))
                self.assertEqual(
                    parser.qualified_offload_pairs([action], [observation], "RSS_ABSOLUTE"), [])
        for field, invalid in (
            ("before_available", "0"),
            ("after_available", "0"),
            ("after_object_id", "2"),
            ("after_generation", "2"),
            ("after_resident_bytes", "8192"),
        ):
            with self.subTest(resident_field=field):
                resident = dict(observation)
                resident[field] = invalid
                invalid_text = marker("kv_g0_s1_resident_observation", resident) + "\n" \
                    + marker("kv_pressure_unified_action", ACTION_FIELDS) + "\n"
                self.assertIsNone(runner.find_offload_barrier_pair(invalid_text, "RSS_ABSOLUTE"))
                self.assertEqual(
                    parser.qualified_offload_pairs([ACTION_FIELDS], [resident], "RSS_ABSOLUTE"), [])

    def test_offload_barrier_timeout_fails_closed_without_resume(self) -> None:
        artifact, runner = self.run_incomplete_artifact("no_offload")
        self.assertEqual(runner.returncode, 1, runner.stderr)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
        self.assertIsNone(execution["qualification"]["resume"])
        responses = next(artifact.glob("runs/*/responses.jsonl")).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(responses), 2)
        stdout = next(artifact.glob("runs/*/server.stdout")).read_text(encoding="utf-8")
        self.assertNotIn("ordinal=3", stdout)

    def test_offload_barrier_requires_matching_resident_drop(self) -> None:
        artifact, runner = self.run_incomplete_artifact("action_only")
        self.assertEqual(runner.returncode, 1, runner.stderr)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
        self.assertIsNone(execution["qualification"]["resume"])

    def test_offload_barrier_rejects_mismatched_correlation_key(self) -> None:
        for mode in ("mismatch_decision", "mismatch_transaction", "mismatch_seq"):
            with self.subTest(mode=mode):
                artifact, runner = self.run_incomplete_artifact(mode)
                self.assertEqual(runner.returncode, 1, runner.stderr)
                execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
                self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
                self.assertIsNone(execution["qualification"]["resume"])

    def test_offload_barrier_rejects_zero_physical_drop(self) -> None:
        artifact, runner = self.run_incomplete_artifact("drop_zero")
        self.assertEqual(runner.returncode, 1, runner.stderr)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
        self.assertIsNone(execution["qualification"]["resume"])

    def test_resume_is_real_measurement_replay_after_barrier(self) -> None:
        artifact = self.run_real_artifact()
        run = json.loads(next(artifact.glob("runs/*/run.json")).read_text(encoding="utf-8"))
        self.assertEqual(
            [item["request_id"] for item in run["request_plan"]],
            ["seed", "request", "request"],
        )
        self.assertEqual(
            [item["measurement"] for item in run["request_plan"]],
            [False, True, False],
        )
        self.assertNotIn("__qualification", json.dumps(run))
        runner_source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("X-KV-Benchmark-Phase", runner_source)
        self.assertNotIn("__qualification_offload__", runner_source)
        self.assertNotIn("__qualification_resume__", runner_source)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        barrier = execution["qualification"]["offload_barrier"]
        resume = execution["qualification"]["resume"]
        self.assertEqual(barrier["status"], "passed")
        self.assertGreater(resume["started_mono_ns"], barrier["completed_mono_ns"])
        self.assertEqual(resume["request_id"], "request")

    def test_v2_swap_out_without_swap_in_is_invalid(self) -> None:
        artifact = self.run_real_artifact(mode="swap_in_zero")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("no swap-in/read", result["errors"][0])

    def test_v2_prefetch_noop_and_zero_restore_are_invalid(self) -> None:
        artifact = self.run_real_artifact(mode="prefetch_noop")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("completed graph-allowed PREFETCH", result["errors"][0])

    def test_v2_transaction_local_drop_is_authoritative_without_slot_resident(self) -> None:
        artifact = self.run_real_artifact()
        slots_before = json.loads(next(artifact.glob("runs/*/slots_before.json")).read_text(encoding="utf-8"))
        self.assertNotIn("kv_resident", slots_before["body_json"][0])
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "QUALIFICATION_PASS")
        physical = result["physical_observations"][0]
        self.assertEqual(physical["authority"], "transaction_local_mincore")
        self.assertEqual(physical["transaction_local_offload"][0]["resident_drop_bytes"], 4096)

    def test_resident_no_swap_in_does_not_require_resume(self) -> None:
        artifact = self.run_real_artifact(policy="resident")
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertIsNone(execution["qualification"]["offload_barrier"])
        self.assertIsNone(execution["qualification"]["resume"])
        self.assertEqual(execution["request_count"], 2)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "QUALIFICATION_PASS")
        self.assertEqual(result["physical_observations"][0]["authority"], "slots_pre_post")

        stderr = next(artifact.glob("runs/*/server.stderr"))
        text = stderr.read_text(encoding="utf-8")
        self.assertIn("bytes_written=0", text)
        stderr.write_text(text.replace("bytes_written=0", "bytes_written=1", 1), encoding="utf-8")
        invalid = self.run_parser(artifact)
        self.assertNotEqual(invalid.returncode, 0)
        invalid_result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("migration IO", invalid_result["errors"][0])

    def test_nonstreaming_first_byte_is_rejected_as_ttft(self) -> None:
        artifact = self.run_real_artifact()
        response = next(artifact.glob("runs/*/responses.jsonl"))
        record = json.loads(response.read_text(encoding="utf-8").splitlines()[0])
        record["first_byte_mono_ns"] = record["started_mono_ns"]
        response.write_text(json.dumps(record) + "\n" + "\n".join(response.read_text(encoding="utf-8").splitlines()[1:]) + "\n", encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("non-streaming", result["errors"][0])

    def test_slots_and_cleanup_are_fail_closed(self) -> None:
        artifact = self.run_real_artifact()
        slots = next(artifact.glob("runs/*/slots_before.json"))
        value = json.loads(slots.read_text(encoding="utf-8"))
        value["http_status"] = 404
        slots.write_text(json.dumps(value), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("HTTP 200", result["errors"][0])

        artifact = self.run_real_artifact()
        cleanup = next(artifact.glob("runs/*/cleanup.json"))
        value = json.loads(cleanup.read_text(encoding="utf-8"))
        value["server"]["residual_process"] = True
        cleanup.write_text(json.dumps(value), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("residual", result["errors"][0])

    def test_expected_memory_max_mismatch_is_rejected(self) -> None:
        value = self.spec()
        value["cgroup"]["expected_memory_max"] = "not-the-current-limit"
        result = self.run_runner(self.write_spec(value), self.root / "bad-cgroup")
        self.assertNotEqual(result.returncode, 0)

    def test_missing_finite_cgroup_authority_is_unsupported_before_workload(self) -> None:
        value = self.spec()
        value["pressure_basis"] = {
            "authority": "cgroup_finite",
            "low_water_kb": None,
            "pressure_kb": None,
            "critical_kb": None,
        }
        value["cgroup"]["expected_memory_max"] = None
        artifact = self.root / "missing-pressure-authority"
        result = self.run_runner(self.write_spec(value), artifact)
        self.assertEqual(result.returncode, 3, result.stderr)
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runner_status"], "UNSUPPORTED")
        self.assertEqual(manifest["unsupported"]["stage"], "pre_workload_pressure_authority")
        self.assertFalse(any((artifact / "runs").iterdir()))
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 3, parsed.stderr)
        verdict = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["verdict"], "UNSUPPORTED")

    def test_process_identity_startup_transient_retries_and_timeout_fails_closed(self) -> None:
        runner = load_runner_module()
        expected = ["bash", "sampler.sh", "--sample-process"]
        with mock.patch.object(
                runner, "process_starttime", side_effect=["10", "10", "10", "10", "10"]), \
             mock.patch.object(
                runner, "process_cmdline", side_effect=[
                    runner.RunnerError("process cmdline is malformed for pid 123"),
                    expected,
                    expected,
                ]):
            identity = runner.process_identity(123, expected, timeout_seconds=0.1)
        self.assertEqual(identity["pid"], 123)
        self.assertEqual(identity["starttime_ticks"], 10)
        self.assertEqual(identity["cmdline"], expected)

        with mock.patch.object(runner, "process_starttime", return_value="10"), \
             mock.patch.object(
                runner, "process_cmdline",
                side_effect=runner.RunnerError("process cmdline is malformed for pid 123")):
            with self.assertRaisesRegex(runner.RunnerError, "process cmdline is malformed"):
                runner.process_identity(123, expected, timeout_seconds=0.01)

    def test_wrapper_binding_uses_one_monotonic_deadline_domain(self) -> None:
        script = ROOT / "scripts/kv-controlled-memory-sampler.sh"
        shell = f"""
            source {str(script)!r}
            KV_CONTROLLED_BIND_TIMEOUT_NS=5
            clock_file=$(mktemp)
            date_called=0
            trap 'rm -f "$clock_file"' EXIT
            printf '0' >"$clock_file"
            kv_controlled_read_proc_stat() {{ KV_CONTROLLED_PROC_STATE=S; KV_CONTROLLED_PROC_STARTTIME=1; return 0; }}
            kv_controlled_descendant_pid() {{ printf '123'; }}
            kv_controlled_read_monotonic_ns() {{
                clock_calls=$(<"$clock_file")
                clock_calls=$((clock_calls + 1))
                printf '%s' "$clock_calls" >"$clock_file"
                if (( clock_calls == 1 )); then printf '100'; else printf '106'; fi
            }}
            date() {{ date_called=1; return 99; }}
            rc=0
            kv_controlled_bind_wrapper_child 123 >/dev/null || rc=$?
            printf '%s:%s' "$rc" "$date_called"
        """
        result = subprocess.run(["bash", "-c", shell], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "7:0")

    def test_cleanup_order_is_sampler_then_server(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        sampler_stop = source.index('if sampler is not None:\n            sampler_cleanup = terminate_process')
        server_stop = source.index('if server is not None:\n            server_cleanup = terminate_process')
        self.assertLess(sampler_stop, server_stop)

    def test_process_identity_hash_and_execution_order_are_checked(self) -> None:
        artifact = self.run_real_artifact()
        execution = next(artifact.glob("runs/*/execution.json"))
        value = json.loads(execution.read_text(encoding="utf-8"))
        value["server_identity"]["cmdline_sha256"] = "0" * 64
        execution.write_text(json.dumps(value), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("cmdline hash", result["errors"][0])

    def test_formal_requires_clean_worktree_and_measurements(self) -> None:
        value = self.spec()
        value["run_kind"] = "formal"
        value["workload"]["repeat"] = 2
        value["run_order"] = [
            {"round": 1, "run_order": 1, "case_id": "case"},
            {"round": 2, "run_order": 1, "case_id": "case"},
        ]
        result = self.run_runner(self.write_spec(value), self.root / "formal-dirty", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        value["workload"]["requests"] = []
        value["run_kind"] = "qualification"
        value["run_order"] = [{"round": 1, "run_order": 1, "case_id": "case"}]
        result = self.run_runner(self.write_spec(value), self.root / "no-measurement", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_unimplemented_policy_is_unsupported_before_workload(self) -> None:
        value = self.spec(policy="v3")
        value["cases"][0]["kv_target_bytes"] = None
        spec = self.write_spec(value, "unsupported.json")
        artifact = self.root / "unsupported"
        runner = self.run_runner(spec, artifact, dry_run=True)
        self.assertEqual(runner.returncode, 3, runner.stderr)
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runner_status"], "UNSUPPORTED")
        self.assertFalse(any((artifact / "runs").iterdir()))
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 3, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "UNSUPPORTED")

    def test_unimplemented_factor_is_unsupported_before_workload(self) -> None:
        for field, value in (("kv_representation", "quantized"), ("loading_mode", "selective"), ("restore", "k3"), ("prefault", "r3")):
            with self.subTest(field=field):
                spec_value = self.spec()
                spec_value["cases"][0][field] = value
                spec = self.write_spec(spec_value, f"unsupported-{field}.json")
                artifact = self.root / f"unsupported-{field}"
                runner = self.run_runner(spec, artifact, dry_run=True)
                self.assertEqual(runner.returncode, 3, runner.stderr)
                self.assertFalse(any((artifact / "runs").iterdir()))
                parsed = self.run_parser(artifact)
                self.assertEqual(parsed.returncode, 3, parsed.stderr)
                result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["verdict"], "UNSUPPORTED")

    def test_plan_mutations_fail_closed(self) -> None:
        spec_value = self.spec()
        spec_value["cases"].append({
            "case_id": "case-2", "policy": "resident", "kv_representation": "paged",
            "loading_mode": "exact", "restore": "k1_sync", "prefault": "off", "kv_target_bytes": None,
        })
        spec_value["run_order"].append({"round": 1, "run_order": 2, "case_id": "case-2"})
        spec = self.write_spec(spec_value, "plan.json")
        artifact = self.root / "plan"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 0)
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["planned_runs"] = list(reversed(manifest["planned_runs"]))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("planned_runs", result["errors"][0])

    def test_formal_requires_explicit_interleaved_repeats(self) -> None:
        spec_value = self.spec()
        spec_value["run_kind"] = "formal"
        spec = self.write_spec(spec_value, "formal.json")
        artifact = self.root / "formal"
        runner = self.run_runner(spec, artifact, dry_run=True)
        self.assertEqual(runner.returncode, 2, runner.stderr)
        self.assertFalse(artifact.exists())

    def test_unknown_case_fails_closed(self) -> None:
        spec_value = self.spec()
        spec_value["run_order"][0]["case_id"] = "missing-case"
        spec = self.write_spec(spec_value, "unknown-case.json")
        artifact = self.root / "unknown-case"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 2)
        self.assertFalse(artifact.exists())

    def test_duplicate_case_fails_closed(self) -> None:
        spec_value = self.spec()
        spec_value["cases"].append(dict(spec_value["cases"][0]))
        spec = self.write_spec(spec_value, "duplicate-case.json")
        artifact = self.root / "duplicate-case"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 2)
        self.assertFalse(artifact.exists())

    def test_missing_case_in_plan_fails_closed(self) -> None:
        spec_value = self.spec()
        spec_value["cases"].append({
            "case_id": "case-2", "policy": "resident", "kv_representation": "paged",
            "loading_mode": "exact", "restore": "k1_sync", "prefault": "off", "kv_target_bytes": None,
        })
        spec_value["run_order"].append({"round": 1, "run_order": 2, "case_id": "case-2"})
        spec = self.write_spec(spec_value, "missing-case.json")
        artifact = self.root / "missing-case"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 0)
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["spec"]["cases"] = manifest["spec"]["cases"][:1]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")

    def test_sampler_monotonic_fraction_conversion_regression(self) -> None:
        shell = f"source {str(ROOT / 'scripts/kv-controlled-memory-sampler.sh')!r}; " \
            "for value in 7.01 7.45 7.123456; do " \
            "kv_controlled_monotonic_ns_from_uptime \"$value\"; printf '\\n'; done"
        result = subprocess.run(["bash", "-c", shell], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["7010000000", "7450000000", "7123456000"])

    def test_real_short_synthetic_http_and_sampler_path(self) -> None:
        spec = self.write_spec(self.spec(), "real.json")
        artifact = self.root / "real"
        runner = self.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        self.assertFalse((artifact / "result.json").exists())
        samples = (artifact / "runs/r001_o001_case/memory_samples.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(samples[0].split("\t")[:5], [
            "elapsed_ms", "timestamp_mono_ns", "timestamp_realtime_ns", "pid", "starttime_ticks",
        ])
        self.assertGreaterEqual(len(samples), 2)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "QUALIFICATION_PASS")
        self.assertEqual(result["statistics"]["runs"], 1)
        self.assertEqual(result["statistics"]["by_case"]["case"]["runs"][0]["request_count"], 1)
        self.assertEqual(result["statistics"]["by_case"]["case"]["runs"][0]["ttft_ms"]["status"], "UNAVAILABLE")
        self.assertEqual(result["statistics"]["by_case"]["case"]["runs"][0]["e2e_ms"]["status"], "AVAILABLE")
        self.assertEqual(result["action_summary"]["offload_bytes"], 4096)
        round_trip = result["restore_observations"][0]["qualification_round_trip"]
        self.assertEqual(round_trip["offload"]["resident_drop_bytes"], 4096)
        self.assertEqual(round_trip["prefetch"]["outcome"], "completed")
        self.assertEqual(round_trip["prefetch"]["graph_allowed"], "1")
        self.assertEqual(round_trip["timing"]["restored_blocks"], "1")
        self.assertEqual(round_trip["timing"]["restored_bytes"], "4096")
        self.assertGreater(int(result["restore_observations"][0]["io"]["block_swap_in_calls"]), 0)
        self.assertGreater(int(result["restore_observations"][0]["io"]["backing_read_syscalls"]), 0)
        self.assertGreater(int(result["restore_observations"][0]["io"]["bytes_read"]), 0)

        execution_path = next(artifact.glob("runs/*/execution.json"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        self.assertEqual(execution["environment"]["LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE"], "0")
        execution["server_identity"]["starttime_ticks"] += 1
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        tampered = self.run_parser(artifact)
        self.assertNotEqual(tampered.returncode, 0)
        tampered_result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(tampered_result["verdict"], "INVALID_ARTIFACT")


if __name__ == "__main__":
    unittest.main()
