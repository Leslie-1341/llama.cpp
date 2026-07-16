#!/usr/bin/env python3
"""Synthetic fail-closed tests for the E0/E2/E5 single-turn diagnostic parser."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARSER = REPO / "scripts/parse-kv-e0-e2-e5-single-turn.py"
CASES = ("E0", "E2", "E5")
RAW_ARTIFACTS = (
    "run.json", "command", "environment", "stdout", "stderr", "exit_code",
    "perf_status", "perf_report.txt", "perf_report.stderr", "seq0", "seq1",
)


def key_values(values: dict[str, int | float]) -> str:
    return " ".join(f"{key}={value}" for key, value in values.items())


def timing_line() -> str:
    values = {
        "apply_calls": 1, "apply_paged_total_us": 10, "set_row_idx_calls": 1,
        "set_row_idx_total_us": 2, "row_idx_fill_us": 2, "active_visible_us": 1,
        "nonidentity_probe_us": 1, "swapped_blocks_scan_us": 1,
        "check_read_resident_us": 1, "check_read_resident_calls": 1,
        "paged_resolve_calls": 1, "cells_scanned": 1, "blocks_scanned": 1,
        "row_idx_entries": 1, "getenv_calls": 1,
    }
    return "KV_PAGED_TIMING_SUMMARY " + key_values(values)


def metadata(case_id: str) -> dict[str, int]:
    e5 = int(case_id == "E5")
    values = {
        "ingraph_gather_layers": 1, "paged_nonidentity_enabled": 1,
        "paged_nonidentity_remap_rows": e5, "paged_swap_enabled": e5,
        "paged_idle_swap_enabled": e5, "paged_swap_out_calls": e5,
        "paged_swap_madvise_calls": e5, "paged_swap_madvise_bytes": e5 * 4096,
    }
    for name in (
        "paged_swapped_active_visible_violation", "paged_swapped_active_visible_violation_rows",
        "paged_swapped_active_visible_violation_blocks", "paged_write_to_swapped_block",
        "paged_row_mapping_invalid_fatal", "paged_write_mapping_invalid_fatal",
        "paged_active_row_nonresident_fatal", "paged_input_setup_fatal",
        "paged_swap_backend_failures", "paged_swap_read_swap_in_failures",
        "paged_swap_write_swap_in_failures", "paged_swap_in_fail_no_offset",
        "paged_swap_in_fail_bad_size", "paged_swap_in_fail_read_cell",
        "paged_swap_in_fail_tensor_set", "paged_prefetch_seq_failures",
        "paged_block_release_fail", "paged_swap_madvise_failures",
    ):
        values[name] = 0
    return values


def io_values(case_id: str) -> dict[str, int]:
    e5 = int(case_id == "E5")
    return {
        "block_swap_out_calls": e5, "block_swap_in_calls": e5,
        "bytes_read": e5 * 4096, "bytes_written": e5 * 4096,
        "avg_block_swap_in_latency_us": e5 * 100, "max_block_swap_in_latency_us": e5 * 100,
        "block_in_validate_us": e5 * 10, "block_in_read_us": e5 * 20,
        "block_in_unpack_us": e5 * 30, "block_in_commit_us": e5 * 40,
    }


def environment(case_id: str) -> str:
    common = {
        "LLAMA_KV_E2_GET_ROWS_PROFILE": int(case_id == "E2"),
        "LLAMA_KV_LAZY_CLEAR": 0, "LLAMA_KV_LAZY_TAIL": 0,
        "LLAMA_KV_PAGED": int(case_id != "E0"),
        "LLAMA_KV_PAGED_INGRAPH": int(case_id != "E0"),
        "LLAMA_KV_PAGED_GATHER_NONIDENTITY": int(case_id != "E0"),
        "LLAMA_KV_PAGED_SWAP": int(case_id == "E5"),
        "LLAMA_KV_PAGED_IDLE_SWAP": int(case_id == "E5"),
        "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": int(case_id == "E5"),
        "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": int(case_id == "E5"),
    }
    if case_id == "E5":
        common.update({
            "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": 1,
            "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": 1,
            "LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME": 1,
        })
    return "\n".join(f"{key}={value}" for key, value in common.items()) + "\n"


def stderr_for(case_id: str) -> str:
    e5 = int(case_id == "E5")
    active = {
        "active_token_count": 10, "avg_ms": 1, "p50_ms": 1, "p95_ms": 2,
        "p99_ms": 3, "max_ms": 3, "decode_avg_ms": 1,
        "prefetch_calls": e5, "prefetch_total_ms": e5,
    }
    perf = {
        "prefetch_during_active_blocks": e5, "prefetch_auto_started": e5,
        "resume_pending_fallback_blocks": 0, "prefetch_failures": 0,
    }
    lines = [
        "KV_IDLE_SWAP_RESUME_PERF " + key_values(perf),
        "KV_ACTIVE_TOKEN_STATS " + key_values(active),
        "llama_kv_cache: KV swap stats: backend_failures=0",
    ]
    if case_id in {"E0", "E2"}:
        lines.append(timing_line())
    if case_id in {"E2", "E5"}:
        lines.append("llama_kv_cache: KV paged metadata stats: " + key_values(metadata(case_id)))
        lines.append("KV_PAGED_IO_STATS " + key_values(io_values(case_id)))
    if case_id == "E2":
        for step in (1, 2):
            for layer in (0, 1):
                for kv in ("K", "V"):
                    lines.append(
                        f"KV_E2_GET_ROWS_PROFILE step={step} layer={layer} kv={kv} "
                        f"src=cache_{kv.lower()}_l{layer} n_kv={step * 8} row_bytes=256 "
                        "segment_ending_at_get_rows_wall_us=1"
                    )
        lines.append(
            "KV_E2_GET_ROWS_PROFILE_SUMMARY steps=2 model_layers=2 kv_layers=2 events=8 capacity=16 "
            "scope=scheduler_graph_segment_from_previous_callback_boundary_through_target_get_rows_completion"
        )
    if case_id == "E5":
        lines += [
            "KV_PAGED_PREFETCH_BLOCK_PHASE call=1 block_index=0 physical_block=7 "
            "validate_us=10 read_us=20 unpack_us=30 commit_us=40 phase_sum_us=100",
            "KV_PAGED_PREFETCH_PHASE_CALL call=1 seq_id=0 requested_blocks=1 restored_blocks=1 phase_events=1",
            "KV_ACTIVE_TOKEN_PREFETCH token=96 requested_blocks=1 restored_blocks=1 "
            "decode_ms=1 prefetch_ms=1 total_ms=2",
        ]
    return "\n".join(lines) + "\n"


def identity(path: Path) -> dict[str, int | str]:
    data = path.read_bytes()
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def seal_artifacts(root: Path) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    runs = []
    for order, case_id in enumerate(CASES, 1):
        run_dir = root / "runs" / case_id
        runs.append({
            "order": order, "case": case_id, "directory": f"runs/{case_id}",
            "artifacts": {name: identity(run_dir / name) for name in RAW_ARTIFACTS},
        })
    manifest["runs"] = runs
    manifest_path.write_text(json.dumps(manifest) + "\n")


def create_artifacts(root: Path, perf_state: str = "captured") -> None:
    root.mkdir()
    manifest = {
        "protocol": "kv_e0_e2_e5_single_turn_diagnostic", "version": 2,
        "scope": "informal_diagnostic_not_a_controlled_performance_conclusion",
        "repo": {"path": str(REPO), "branch": "test", "head": "0" * 40, "upstream": "NA", "dirty": False, "status_porcelain": []},
        "binary": {"path": "/tmp/binary", "sha256": "1" * 64, "size": 1},
        "model": {"path": "/tmp/model", "sha256": "2" * 64, "size": 1},
        "framework": {
            "runner": {"path": "/tmp/runner", "sha256": "3" * 64, "size": 1},
            "parser": {"path": str(PARSER), "sha256": "4" * 64, "size": 1},
        },
        "host": {"hostname": "test", "platform": "test", "python": "test", "compiler_build": "test"},
        "execution": {"dry_run": False, "use_perf": "auto", "perf_available": perf_state == "captured", "perf_reason": "test", "timeout_sec": 1},
        "planned_runs": [{"order": order, "case": case_id} for order, case_id in enumerate(CASES, 1)],
        "runs": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    runs = root / "runs"
    runs.mkdir()
    for order, case_id in enumerate(CASES, 1):
        run_dir = runs / case_id
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({"order": order, "case": case_id}, sort_keys=True) + "\n")
        (run_dir / "command").write_text("true\n")
        (run_dir / "environment").write_text(environment(case_id))
        (run_dir / "stdout").write_text("raw stdout\n")
        (run_dir / "stderr").write_text(stderr_for(case_id))
        (run_dir / "exit_code").write_text("0\n")
        if case_id == "E5":
            (run_dir / "perf_status").write_text("state=not_applicable reason=e5_uses_swapin_phase_telemetry\n")
            (run_dir / "perf_report.txt").write_text("")
        else:
            (run_dir / "perf_status").write_text(f"state={perf_state} reason=test\n")
            report = "  12.34%  binary  [.] ggml_compute_forward_get_rows\n" if perf_state == "captured" else ""
            (run_dir / "perf_report.txt").write_text(report)
        (run_dir / "perf_report.stderr").write_text("")
        (run_dir / "seq0").write_text("same seq0\n")
        (run_dir / "seq1").write_text("same seq1\n")
    seal_artifacts(root)


class ParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = Path(tempfile.mkdtemp(prefix="single-turn-parser-test-"))
        self.root = self.tempdir / "artifacts"
        create_artifacts(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def parse(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["python3", str(PARSER), str(self.root)], text=True, capture_output=True, check=False)

    def run_dir(self, case_id: str) -> Path:
        return self.root / "runs" / case_id

    def test_complete_resolved_artifact_passes(self) -> None:
        result = self.parse()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["result"], "PASS")
        self.assertEqual(summary["runs"][0]["attribution"], "NOT_REQUIRED")
        self.assertEqual(summary["runs"][1]["attribution"], "GET_ROWS_PROFILE")
        self.assertEqual(summary["e2_get_rows_profile"]["segment_ending_at_get_rows_wall_us"], {
            "count": 8, "sum": 8, "avg": 1, "min": 1, "max": 1,
        })
        self.assertEqual(
            summary["e2_get_rows_profile"]["segment_ending_at_get_rows_wall_scope"],
            "scheduler_graph_segment_from_previous_callback_boundary_through_target_get_rows_completion",
        )
        self.assertIn(
            "E2-GET_ROWS\tsegment_ending_at_get_rows_wall_us_sum\t8",
            (self.root / "summary.tsv").read_text(),
        )
        self.assertIn(
            "E2-GET_ROWS\tsegment_ending_at_get_rows_wall_scope\t"
            "scheduler_graph_segment_from_previous_callback_boundary_through_target_get_rows_completion",
            (self.root / "summary.tsv").read_text(),
        )
        self.assertIn("previous callback boundary", (self.root / "summary.md").read_text())
        self.assertIn("not be described as single-node time or GET_ROWS kernel time", (self.root / "summary.md").read_text())
        self.assertIn("not step or end-to-end wall time", (self.root / "summary.md").read_text())

    def test_perf_unavailable_is_optional_for_e0_and_e2(self) -> None:
        shutil.rmtree(self.root)
        create_artifacts(self.root, perf_state="unresolved")
        result = self.parse()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["result"], "PASS")
        self.assertEqual([run["result"] for run in summary["runs"]], ["PASS", "PASS", "PASS"])
        self.assertEqual(summary["runs"][0]["perf_get_rows"]["status"], "unresolved")
        self.assertEqual(summary["runs"][1]["perf_get_rows"]["status"], "unresolved")

    def test_perf_report_without_get_rows_samples_is_optional(self) -> None:
        for case_id in ("E0", "E2"):
            (self.run_dir(case_id) / "perf_report.txt").write_text("")
        seal_artifacts(self.root)
        result = self.parse()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["result"], "PASS")
        self.assertEqual([run["perf_get_rows"]["status"] for run in summary["runs"][:2]],
                         ["unresolved", "unresolved"])

    def test_e2_profiler_is_required_even_when_perf_resolves(self) -> None:
        stderr = self.run_dir("E2") / "stderr"
        stderr.write_text("\n".join(
            line for line in stderr.read_text().splitlines()
            if not line.startswith("KV_E2_GET_ROWS_PROFILE")
        ) + "\n")
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

    def test_manifest_and_raw_evidence_fail_closed(self) -> None:
        manifest = json.loads((self.root / "manifest.json").read_text())
        manifest["planned_runs"].append("ignored")
        (self.root / "manifest.json").write_text(json.dumps(manifest) + "\n")
        self.assertEqual(self.parse().returncode, 2)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        (self.run_dir("E0") / "command").write_text("false\n")
        self.assertEqual(self.parse().returncode, 2)

    def test_case_environment_and_fallback_fail_closed(self) -> None:
        environment_path = self.run_dir("E0") / "environment"
        environment_path.write_text(environment_path.read_text().replace("LLAMA_KV_PAGED=0", "LLAMA_KV_PAGED=1"))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        stderr = self.run_dir("E2") / "stderr"
        stderr.write_text(stderr.read_text() + "warning: requested path disabled; falling back\n")
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

    def test_e5_token_block_phase_mismatch_fails(self) -> None:
        stderr = self.run_dir("E5") / "stderr"
        stderr.write_text(stderr.read_text().replace("restored_blocks=1 phase_events=1", "restored_blocks=0 phase_events=0", 1))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        stderr = self.run_dir("E5") / "stderr"
        stderr.write_text(stderr.read_text().replace("phase_sum_us=100", "phase_sum_us=99"))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

    def test_e2_profiler_missing_or_duplicate_marker_fails(self) -> None:
        stderr = self.run_dir("E2") / "stderr"
        stderr.write_text("\n".join(
            line for line in stderr.read_text().splitlines()
            if not line.startswith("KV_E2_GET_ROWS_PROFILE_SUMMARY ")
        ) + "\n")
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        stderr = self.run_dir("E2") / "stderr"
        summary = next(line for line in stderr.read_text().splitlines() if line.startswith("KV_E2_GET_ROWS_PROFILE_SUMMARY "))
        stderr.write_text(stderr.read_text() + summary + "\n")
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

    def test_e2_profiler_kv_completeness_and_event_count_fail_closed(self) -> None:
        stderr = self.run_dir("E2") / "stderr"
        incomplete = stderr.read_text().replace(
            "layer=1 kv=V src=cache_v_l1", "layer=2 kv=V src=cache_v_l2", 1)
        stderr.write_text(incomplete.replace("model_layers=2", "model_layers=3"))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        stderr = self.run_dir("E2") / "stderr"
        stderr.write_text(stderr.read_text().replace("events=8 capacity=16", "events=7 capacity=16"))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

    def test_e2_profiler_wall_time_and_case_pollution_fail_closed(self) -> None:
        stderr = self.run_dir("E2") / "stderr"
        stderr.write_text(stderr.read_text().replace(
            "segment_ending_at_get_rows_wall_us=1", "segment_ending_at_get_rows_wall_us=-1", 1))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        environment_path = self.run_dir("E0") / "environment"
        environment_path.write_text(environment_path.read_text().replace(
            "LLAMA_KV_E2_GET_ROWS_PROFILE=0", "LLAMA_KV_E2_GET_ROWS_PROFILE=1"))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

    def test_e2_profiler_scope_fails_closed(self) -> None:
        stderr = self.run_dir("E2") / "stderr"
        stderr.write_text(stderr.read_text().replace(
            "scope=scheduler_graph_segment_from_previous_callback_boundary_through_target_get_rows_completion",
            "scope=single_node_get_rows",
        ))
        seal_artifacts(self.root)
        self.assertEqual(self.parse().returncode, 2)

if __name__ == "__main__":
    unittest.main()
