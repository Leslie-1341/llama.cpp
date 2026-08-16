#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER_PATH = ROOT / "scripts/parse-global-route-a-benchmark.py"
spec = importlib.util.spec_from_file_location("global_route_a_parser", PARSER_PATH)
assert spec and spec.loader
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


BASE_SPEC = {
    "kv_resident_target_bytes": 1024,
    "initial_moe_budget_bytes": 2 << 20,
    "static_high_moe_budget_bytes": 4 << 20,
    "memory_cap_bytes": 8192,
    "ctx_size": 8192,
    "workload": {
        "warmup": [{"prompt": [1, 2], "n_predict": 1}],
        "session_a_turn1": {"prompt": [1, 2, 3], "n_predict": 2, "id_slot": 0, "cache_prompt": True, "seed": 0, "temperature": 0, "top_k": 1, "top_p": 1},
        "session_a_turn2": {"prompt": [1, 2, 3, 4], "n_predict": 2, "id_slot": 0, "cache_prompt": True, "seed": 0, "temperature": 0, "top_k": 1, "top_p": 1},
        "session_b": {"prompt": [9, 8], "n_predict": 2, "id_slot": 0, "cache_prompt": True, "seed": 0, "temperature": 0, "top_k": 1, "top_p": 1},
        "moe_demand_repeats": 4,
    },
}


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_artifact(root: pathlib.Path, *, dirty: bool = True) -> None:
    write_json(root / "case_spec.json", BASE_SPEC)
    write_json(root / "manifest.json", {
        "protocol": parser.PROTOCOL,
        "protocol_version": 1,
        "runner_status": "run_complete",
        "case_names": list(parser.CASES),
        "parameters": {key: BASE_SPEC[key] for key in ("kv_resident_target_bytes", "initial_moe_budget_bytes", "static_high_moe_budget_bytes", "memory_cap_bytes")},
        "git": {"capture_mode": "DIRTY_DEV_ONLY" if dirty else "archival_clean", "head": "abc", "branch": "test", "dirty_status": [" M scripts/parse-kv-offload-benchmark.py"] if dirty else []},
    })
    for case in parser.CASES:
        make_case(root / case, case)


def marker(case: str, *, global_case: bool) -> list[str]:
    lines = [
        "preflight ok\n",
        "health ok\n",
        "warmup ok\n",
        "fill ok\n",
        "kv_pressure_unified_action action=offload offload_attempted=1 decision_id=1 transaction_id=11 outcome=completed reason=none state_changed=1 io_failure=0 physical_relief_available=1 physical_relief_bytes=100 physical_object_id=2 physical_generation=4\n",
    ]
    if global_case:
        lines.append("memory_governor_observe reallocation_confirmed_kv_physical_credit_bytes=100 reallocation_confirmed_kv_physical_object_id=2 reallocation_confirmed_kv_physical_generation=4 reallocation_kv_physical_credit_earned_bytes=64 reallocation_credit_earned_bytes=64 reallocation_credit_available_bytes=64 reallocation_credit_remaining_bytes=64 reallocation_credit_spent_bytes=0 reallocation_moe_grant_bytes=0 moe_resident_bytes=100 moe_cache_hits=1 moe_cache_misses=0\n")
    else:
        lines.append("memory_governor_observe reallocation_confirmed_kv_physical_credit_bytes=0 reallocation_confirmed_kv_physical_object_id=0 reallocation_confirmed_kv_physical_generation=0 reallocation_credit_earned_bytes=0 reallocation_credit_available_bytes=0 reallocation_credit_remaining_bytes=0 reallocation_credit_spent_bytes=0 reallocation_moe_grant_bytes=0 moe_resident_bytes=100\n")
    lines.append("b demand\n")
    if global_case:
        lines.append("memory_governor_observe reallocation_reason=moe_credit_grant reallocation_confirmed_kv_physical_credit_bytes=0 reallocation_confirmed_kv_physical_object_id=0 reallocation_confirmed_kv_physical_generation=0 reallocation_kv_physical_credit_earned_bytes=64 reallocation_credit_earned_bytes=64 reallocation_credit_available_bytes=0 reallocation_credit_remaining_bytes=0 reallocation_credit_spent_bytes=64 reallocation_moe_grant_bytes=64 reallocation_moe_old_budget_bytes=2048 reallocation_moe_new_budget_bytes=2112 moe_resident_bytes=200 moe_cache_hits=2 moe_cache_misses=1 moe_bytes_read=128 moe_budget_action=grow\n")
    else:
        lines.append("memory_governor_observe reallocation_confirmed_kv_physical_credit_bytes=0 reallocation_confirmed_kv_physical_object_id=0 reallocation_confirmed_kv_physical_generation=0 reallocation_credit_earned_bytes=0 reallocation_credit_available_bytes=0 reallocation_credit_remaining_bytes=0 reallocation_credit_spent_bytes=0 reallocation_moe_grant_bytes=0 moe_resident_bytes=100\n")
    lines.extend([
        "kv_resume_order_event phase=prefetch seq_id=0 claimant_epoch=1 transaction_id=22 decision_id=44 action=prefetch outcome=completed reason=none graph_allowed=0\n",
        "kv_resume_order_event phase=graph_gate seq_id=0 claimant_epoch=1 transaction_id=22 decision_id=44 action=prefetch outcome=completed reason=none graph_allowed=1\n",
        "KV_PAGED_PREFETCH_PHASE_CALL call=1 seq_id=0 requested_blocks=2 restored_blocks=2 phase_events=2\n",
        "steady ok\n",
    ])
    return lines


def make_case(case_dir: pathlib.Path, case: str) -> None:
    case_dir.mkdir()
    lines = marker(case, global_case=case == "global_route_a")
    stderr = "".join(lines)
    (case_dir / "server.stderr").write_text(stderr, encoding="utf-8")
    (case_dir / "server.stdout").write_text("", encoding="utf-8")
    offsets = []
    offset = 0
    for line in lines:
        offsets.append((offset, offset + len(line.encode("utf-8"))))
        offset += len(line.encode("utf-8"))
    names = [
        "P0_preflight_identity", "P0_health", "P1_fixed_warmup", "P2_A_turn1_fill",
        "P3_A_completion_idle_release", "P4_route_a_credit_observation", "P5_B_moe_demand",
        "P6_A_turn2_exact_restore", "P7_short_steady_measurement",
    ]
    windows = []
    ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 11), (11, 12)]
    for name, (lo, hi) in zip(names, ranges):
        start = offsets[lo][0]
        end = offsets[hi - 1][1]
        windows.append({"name": name, "stderr_start": start, "stderr_end": end, "status": "PASS"})
    (case_dir / "phase_events.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in windows), encoding="utf-8")
    requests = []
    for label, request in (("A_turn1", BASE_SPEC["workload"]["session_a_turn1"]), ("B_0", BASE_SPEC["workload"]["session_b"]), ("B_1", BASE_SPEC["workload"]["session_b"]), ("B_2", BASE_SPEC["workload"]["session_b"]), ("B_3", BASE_SPEC["workload"]["session_b"]), ("A_turn2", BASE_SPEC["workload"]["session_a_turn2"])):
        tokens = [7, 8]
        request_bytes = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
        token_bytes = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
        requests.append({"label": label, "phase": "P6" if label == "A_turn2" else "P5", "request": request,
                         "request_sha256": parser.sha_bytes(request_bytes), "http_status": 200, "transport_error": None,
                         "response_text": "ok", "response_sha256": parser.sha_bytes(b"ok"), "response_tokens": tokens,
                         "response_tokens_sha256": parser.sha_bytes(token_bytes)})
    (case_dir / "requests.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in requests), encoding="utf-8")
    timeline = {"monotonic_ns": 1, "phase": "P6_A_turn2_exact_restore", "process": {"VmRSS": 1}, "cgroup": {"available": False}, "runtime": {}}
    (case_dir / "memory_timeline.jsonl").write_text(json.dumps(timeline) + "\n", encoding="utf-8")
    write_json(case_dir / "cleanup.json", {"completed": True})
    write_json(case_dir / "exit_codes.json", {"server": {"exit_code": 0, "residual_process": False}, "sampler": {"alive": False}})
    write_json(case_dir / "result.json", {"case": case, "status": "complete"})
    write_json(case_dir / "execution.json", {"argv": ["/bin/server"], "binary": {"path": "/bin/server", "size": 1, "sha256": "a"}, "model": {"path": "/model.gguf", "size": 1, "sha256": "b"}, "cgroup_path": "/cg"})
    write_json(case_dir / "process.json", {"pid": 1, "starttime_ticks": 1, "cmdline": ["/bin/server"]})
    env = {
        "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER": "1" if case == "global_route_a" else "0",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION": "1" if case == "global_route_a" else "0",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_APPLY_MOE": "1" if case == "global_route_a" else "0",
        "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC": "1" if case == "global_route_a" else "0",
        "LLAMA_MEMORY_GOVERNOR_MOE_WARM_MB": "2" if case != "static_high" else "4",
        "LLAMA_MEMORY_GOVERNOR_MOE_MAX_MB": "2" if case == "static_low" else "4",
        "LLAMA_KV_RESIDENT_TARGET_BYTES": "1024", "LLAMA_KV_RESIDENT_TARGET_SOURCE": "env_static",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "1024",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_CONFIRM": "1",
        "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1", "LLAMA_KV_PAGED_SWAP": "1",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1", "LLAMA_KV_PAGED_MINCORE": "1",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
    }
    write_json(case_dir / "environment.json", {"effective": env})


class GlobalRouteAParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="global-route-a-test-")
        self.root = pathlib.Path(self.temp.name)
        make_artifact(self.root)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def parse(self) -> dict:
        return parser.parse_artifact(self.root)

    def test_complete_route_a_passes_mechanism_utilization_and_correctness(self) -> None:
        result = self.parse()
        self.assertEqual(result["verdict"]["verdict"], "GLOBAL_FAST_A1_READY_FOR_INTEGRATION")
        self.assertEqual(result["summary"]["gates"]["route_a_mechanism"], "GLOBAL_ROUTE_A_MECHANISM_PASS")
        self.assertEqual(result["summary"]["gates"]["route_a_utilization"], "GLOBAL_ROUTE_A_UTILIZATION_PASS")
        self.assertEqual(result["summary"]["gates"]["cross_case_correctness"], "CROSS_CASE_CORRECTNESS_PASS")
        self.assertTrue(result["verdict"]["dirty_dev_only"])
        self.assertFalse(result["verdict"]["performance_claims_allowed"])

    def test_static_case_credit_grant_is_rejected(self) -> None:
        path = self.root / "static_low" / "server.stderr"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[5] = "memory_governor_observe reallocation_reason=moe_credit_grant reallocation_credit_earned_bytes=64 reallocation_credit_available_bytes=0 reallocation_credit_remaining_bytes=0 reallocation_credit_spent_bytes=64 reallocation_moe_grant_bytes=64 reallocation_moe_old_budget_bytes=1 reallocation_moe_new_budget_bytes=2 moe_resident_bytes=100"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = self.parse()
        self.assertEqual(result["verdict"]["verdict"], "NOT_READY")
        self.assertTrue(any(error.endswith("static_low_has_route_a_credit_grant") for error in result["summary"]["errors"]))

    def test_global_missing_physical_relief_fails(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        path.write_text(path.read_text(encoding="utf-8").replace("physical_relief_available=1 physical_relief_bytes=100", "physical_relief_available=0 physical_relief_bytes=0"), encoding="utf-8")
        result = self.parse()
        self.assertEqual(result["verdict"]["verdict"], "NOT_READY")
        self.assertEqual(result["summary"]["gates"]["route_a_mechanism"], "INVALID")

    def test_object_generation_mismatch_fails(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        text = path.read_text(encoding="utf-8").replace(
            "reallocation_confirmed_kv_physical_credit_bytes=100 reallocation_confirmed_kv_physical_object_id=2 reallocation_confirmed_kv_physical_generation=4",
            "reallocation_confirmed_kv_physical_credit_bytes=100 reallocation_confirmed_kv_physical_object_id=3 reallocation_confirmed_kv_physical_generation=4",
            1,
        )
        path.write_text(text, encoding="utf-8")
        result = self.parse()
        self.assertTrue(any(error.endswith("physical_object_generation_mismatch") for error in result["summary"]["errors"]))

    def test_confirmation_generation_mismatch_fails(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        text = path.read_text(encoding="utf-8").replace(
            "reallocation_confirmed_kv_physical_generation=4",
            "reallocation_confirmed_kv_physical_generation=5",
            1,
        )
        path.write_text(text, encoding="utf-8")
        result = self.parse()
        self.assertTrue(any(error.endswith("physical_object_generation_mismatch") for error in result["summary"]["errors"]))

    def test_missing_confirmation_identity_fails(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        text = path.read_text(encoding="utf-8").replace(
            "reallocation_confirmed_kv_physical_object_id=2 reallocation_confirmed_kv_physical_generation=4",
            "reallocation_confirmed_kv_physical_object_id=0 reallocation_confirmed_kv_physical_generation=0",
            1,
        )
        path.write_text(text, encoding="utf-8")
        result = self.parse()
        self.assertTrue(any(error.endswith("credit_missing_physical_object_generation") for error in result["summary"]["errors"]))

    def test_intervening_unrelated_action_fails(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        lines = path.read_text(encoding="utf-8").splitlines()
        intervening = ("kv_pressure_unified_action action=offload offload_attempted=1 decision_id=88 transaction_id=99 "
                       "outcome=completed reason=none state_changed=1 io_failure=0 physical_relief_available=1 "
                       "physical_relief_bytes=10 physical_object_id=2 physical_generation=4")
        lines.insert(6, intervening)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        events_path = self.root / "global_route_a" / "phase_events.jsonl"
        events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
        events[5]["stderr_end"] = events[5]["stderr_end"] + len(intervening) + 1
        events[6]["stderr_start"] = events[6]["stderr_start"] + len(intervening) + 1
        events_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in events), encoding="utf-8")
        result = self.parse()
        self.assertTrue(any(error.endswith("physical_object_generation_mismatch") or "duplicate_intervening_action" in error
                            for error in result["summary"]["errors"]))

    def test_credit_ledger_must_close(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        path.write_text(path.read_text(encoding="utf-8").replace("reallocation_credit_spent_bytes=64 reallocation_moe_grant_bytes=64", "reallocation_credit_spent_bytes=64 reallocation_moe_grant_bytes=32"), encoding="utf-8")
        result = self.parse()
        self.assertTrue(any(error.endswith("credit_spent_grant_not_closed:64!=32") for error in result["summary"]["errors"]))

    def test_emergency_grow_is_not_live_credit(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        text = path.read_text(encoding="utf-8").replace("reallocation_reason=moe_credit_grant", "reallocation_reason=moe_emergency_working_set")
        path.write_text(text, encoding="utf-8")
        result = self.parse()
        self.assertEqual(result["summary"]["gates"]["route_a_mechanism"], "INVALID")

    def test_budget_growth_without_demand_is_unverified(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        text = path.read_text(encoding="utf-8").replace("moe_cache_hits=1 moe_cache_misses=0", "moe_cache_hits=0 moe_cache_misses=0")
        text = text.replace("moe_cache_hits=2", "moe_cache_hits=0")
        text = text.replace("moe_cache_misses=1", "moe_cache_misses=0")
        text = text.replace("moe_bytes_read=128", "moe_bytes_read=0")
        text = text.replace("moe_resident_bytes=200", "moe_resident_bytes=100")
        path.write_text(text, encoding="utf-8")
        result = self.parse()
        self.assertEqual(result["summary"]["gates"]["route_a_utilization"], "GLOBAL_ROUTE_A_UTILIZATION_UNVERIFIED")
        self.assertEqual(result["verdict"]["verdict"], "NOT_READY")

    def test_exact_output_drift_fails(self) -> None:
        path = self.root / "static_high" / "requests.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[-1] = lines[-1].replace("[7, 8]", "[7, 9]")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = self.parse()
        self.assertIn("exact_token_output_drift", result["summary"]["errors"])

    def test_missing_artifact_is_fail_closed(self) -> None:
        (self.root / "global_route_a" / "memory_timeline.jsonl").unlink()
        result = self.parse()
        self.assertEqual(result["verdict"]["verdict"], "INVALID")
        self.assertTrue(any("missing artifacts" in error for error in result["verdict"]["errors"]))

    def test_phase_order_and_graph_gate_are_required(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        text = path.read_text(encoding="utf-8").replace("phase=graph_gate", "phase=wrong_gate")
        path.write_text(text, encoding="utf-8")
        result = self.parse()
        self.assertTrue(any("unknown_restore_phase" in error or "missing_exact_restore_markers" in error for error in result["summary"]["errors"]))

    def test_unexpected_case_environment_difference_fails(self) -> None:
        env_path = self.root / "static_high" / "environment.json"
        env = json.loads(env_path.read_text(encoding="utf-8"))
        env["effective"]["LLAMA_KV_PAGED"] = "0"
        write_json(env_path, env)
        result = self.parse()
        self.assertIn("unexpected_case_environment_diff:LLAMA_KV_PAGED", result["summary"]["errors"])

    def test_identity_mismatch_fails(self) -> None:
        execution_path = self.root / "static_high" / "execution.json"
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        execution["model"]["sha256"] = "different"
        write_json(execution_path, execution)
        result = self.parse()
        self.assertIn("common_execution_identity_mismatch", result["summary"]["errors"])

    def test_dirty_dev_boundary_is_recorded_not_promoted_to_performance_pass(self) -> None:
        result = self.parse()
        self.assertTrue(result["summary"]["dirty_dev_only"])
        self.assertFalse(result["verdict"]["performance_claims_allowed"])

    def test_startup_credit_is_not_route_a_grant(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "moe_resident_bytes=100 moe_cache_hits=1",
            "moe_budget_reason=fast_start moe_resident_bytes=100 moe_cache_hits=1"), encoding="utf-8")
        result = self.parse()
        self.assertEqual(result["summary"]["gates"]["route_a_mechanism"], "GLOBAL_ROUTE_A_MECHANISM_PASS")


    def test_empty_timeline_fails_closed(self) -> None:
        (self.root / "global_route_a" / "memory_timeline.jsonl").write_text("", encoding="utf-8")
        result = self.parse()
        self.assertTrue(any("empty_memory_timeline" in error for error in result["summary"]["errors"]))

    def test_missing_common_environment_key_fails(self) -> None:
        env_path = self.root / "global_route_a" / "environment.json"
        env = json.loads(env_path.read_text(encoding="utf-8"))
        del env["effective"]["LLAMA_KV_PAGED_INGRAPH"]
        write_json(env_path, env)
        result = self.parse()
        self.assertTrue(any("missing_common_environment_key:LLAMA_KV_PAGED_INGRAPH" in error for error in result["summary"]["errors"]))

    def test_process_binary_mismatch_fails(self) -> None:
        path = self.root / "static_high" / "execution.json"
        execution = json.loads(path.read_text(encoding="utf-8"))
        execution["argv"][0] = "/other/server"
        write_json(path, execution)
        result = self.parse()
        self.assertTrue(any(error.endswith("process_binary_mismatch") for error in result["summary"]["errors"]))

    def test_credit_before_physical_relief_fails(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[4], lines[5] = lines[5], lines[4]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = self.parse()
        self.assertTrue(any("credit_before_physical_relief" in error for error in result["summary"]["errors"]))

    def test_dense_grant_fails(self) -> None:
        path = self.root / "global_route_a" / "server.stderr"
        path.write_text(path.read_text(encoding="utf-8").replace(
            "reallocation_moe_grant_bytes=64", "reallocation_dense_grant_bytes=1 reallocation_moe_grant_bytes=64", 1), encoding="utf-8")
        result = self.parse()
        self.assertIn("global_route_a:dense_grant_present", result["summary"]["errors"])

    def test_restore_phase_overlap_fails_closed(self) -> None:
        path = self.root / "global_route_a" / "phase_events.jsonl"
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        events[7]["stderr_start"] = events[6]["stderr_start"]
        path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in events), encoding="utf-8")
        result = self.parse()
        self.assertTrue(any("phase windows overlap" in error for error in result["verdict"]["errors"]))


if __name__ == "__main__":
    unittest.main()
