#!/usr/bin/env python3
"""Synthetic fail-closed tests for the controlled E0-E5 artifact parser."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PARSER = REPO / "scripts/parse-kv-final-controlled-e0-e5.py"
PLAN = [(1, order, case_id) for order, case_id in enumerate(("E0", "E1", "E2", "E3", "E4", "E5"), 1)]


def key_values(values: dict[str, int | float]) -> str:
    return " ".join(f"{key}={value}" for key, value in values.items())


def metadata(case_id: str) -> dict[str, int]:
    paged_swap = int(case_id in {"E3", "E4", "E5"})
    madvise = int(case_id in {"E4", "E5"})
    active_prefetch = int(case_id == "E5")
    result = {
        "ingraph_gather_layers": 1,
        "paged_nonidentity_enabled": 1,
        "paged_nonidentity_remap_rows": paged_swap,
        "paged_block_release_enabled": 0,
        "paged_swap_enabled": paged_swap,
        "paged_idle_swap_enabled": paged_swap,
        "paged_blocks_swapped_out": paged_swap,
        "paged_blocks_swapped_in": paged_swap,
        "paged_swap_out_calls": paged_swap,
        "paged_swap_in_calls": paged_swap,
        "paged_swap_madvise_calls": madvise,
        "paged_swap_madvise_bytes": madvise * 4096,
        "paged_prefetch_seq_calls": active_prefetch,
        "kv_mincore_enabled": 0,
        "paged_swapped_active_visible_violation": 0,
        "paged_swapped_active_visible_violation_rows": 0,
        "paged_swapped_active_visible_violation_blocks": 0,
        "paged_swapped_active_violation_rows": 0,
        "paged_swapped_active_violation_blocks": 0,
        "paged_write_to_swapped_block": 0,
        "paged_row_mapping_invalid_fatal": 0,
        "paged_write_mapping_invalid_fatal": 0,
        "paged_active_row_nonresident_fatal": 0,
        "paged_input_setup_fatal": 0,
        "paged_swap_backend_failures": 0,
        "paged_swap_read_swap_in_failures": 0,
        "paged_swap_write_swap_in_failures": 0,
        "paged_swap_in_fail_no_offset": 0,
        "paged_swap_in_fail_bad_size": 0,
        "paged_swap_in_fail_read_cell": 0,
        "paged_swap_in_fail_tensor_set": 0,
        "paged_prefetch_seq_failures": 0,
        "paged_block_release_fail": 0,
        "paged_swap_madvise_failures": 0,
    }
    return result


def stderr_for(case_id: str) -> str:
    active_prefetch = int(case_id == "E5")
    perf = {
        "total_wall_ms": 100, "seq1_active_ms": 20, "seq0_resume_first_token_ms": 2,
        "tokens_per_second": 10, "prefetch_during_active_blocks": active_prefetch,
        "prefetch_during_active_ms_max": active_prefetch, "prefetch_auto_started": active_prefetch,
        "resume_pending_fallback_blocks": 0, "prefetch_failures": 0,
        "rss_before_active_prefetch_kb": 100, "rss_after_active_prefetch_kb": 100,
        "rss_before_prefetch_kb": 100, "rss_after_prefetch_kb": 100,
        "rss_before_resume_kb": 100, "rss_after_resume_kb": 100,
    }
    active = {
        "active_token_count": 10, "avg_ms": 1, "p50_ms": 1, "p95_ms": 1,
        "p99_ms": 1, "max_ms": 1, "decode_avg_ms": 1,
        "prefetch_calls": active_prefetch, "prefetch_total_ms": active_prefetch,
    }
    lines = [
        "KV_IDLE_SWAP_RESUME_PERF " + key_values(perf),
        "KV_ACTIVE_TOKEN_STATS " + key_values(active),
        "llama_kv_cache: KV swap stats: backend_failures=0",
    ]
    if case_id == "E1":
        lines += [
            "llama_kv_cache: kv lazy-clear stats: enabled=1",
            "llama_kv_cache: kv lazy-tail stats: enabled=1",
        ]
    if case_id in {"E2", "E3", "E4", "E5"}:
        lines.append("llama_kv_cache: KV paged metadata stats: " + key_values(metadata(case_id)))
        io = {
            "block_swap_out_calls": int(case_id in {"E3", "E4", "E5"}),
            "block_swap_in_calls": int(case_id in {"E3", "E4", "E5"}),
            "bytes_read": int(case_id in {"E3", "E4", "E5"}),
            "bytes_written": int(case_id in {"E3", "E4", "E5"}),
            "avg_block_swap_out_latency_us": 0, "max_block_swap_out_latency_us": 0,
            "avg_block_swap_in_latency_us": 0, "max_block_swap_in_latency_us": 0,
            "block_out_validate_us": 0, "block_out_pack_us": 0, "block_out_write_us": 0,
            "block_out_metadata_us": 0, "block_out_madvise_us": 0, "block_in_validate_us": 0,
            "block_in_read_us": 0, "block_in_unpack_us": 0, "block_in_commit_us": 0,
        }
        lines.append("KV_PAGED_IO_STATS " + key_values(io))
    if case_id in {"E3", "E4", "E5"}:
        lines.append("llama_kv_cache: KV swap backing store ready capacity=1.00 MiB")
    return "\n".join(lines) + "\n"


def create_artifacts(root: Path) -> None:
    planned = [
        {"round": round_no, "run_order": order, "case": case_id}
        for round_no, order, case_id in PLAN
    ]
    manifest = {
        "repo": {"path": str(REPO), "dirty": False},
        "workload": {"runs": 1},
        "execution": {"dry_run": False, "allow_dirty": False},
        "planned_runs": planned,
    }
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n")
    runs = root / "runs"
    runs.mkdir()
    for round_no, order, case_id in PLAN:
        run_dir = runs / f"round_{round_no}_order_{order:02d}_{case_id}"
        run_dir.mkdir()
        (run_dir / "run.json").write_text(json.dumps({
            "round": round_no, "run_order": order, "case": case_id,
        }) + "\n")
        (run_dir / "exit_code").write_text("0\n")
        (run_dir / "environment").write_text(
            f"LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME={int(case_id == 'E5')}\n"
        )
        (run_dir / "stdout").write_text("")
        (run_dir / "stderr").write_text(stderr_for(case_id))
        for sequence in ("seq0", "seq1"):
            content = f"fixed {sequence}\n"
            (run_dir / sequence).write_text(content)
            digest = hashlib.sha256(content.encode()).hexdigest()
            (run_dir / f"{sequence}.sha256").write_text(f"{digest}  {sequence}\n")
        (run_dir / "memory_samples.tsv").write_text(
            "vmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes\n"
            "100\t110\t1000\tNA\tNA\n"
        )


class ParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = Path(tempfile.mkdtemp(prefix="e0-e5-parser-test-"))
        self.root = self.tempdir / "artifacts"
        create_artifacts(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.tempdir)

    def parse(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", str(PARSER), str(self.root)], text=True, capture_output=True, check=False,
        )

    def run_dir(self, case_id: str) -> Path:
        return next((self.root / "runs").glob(f"*_{case_id}"))

    def test_complete_matrix_passes(self) -> None:
        result = self.parse()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unverified_is_nonzero(self) -> None:
        (self.run_dir("E0") / "seq1").write_text("")
        self.assertNotEqual(self.parse().returncode, 0)

    def test_missing_and_duplicate_matrix_entries_are_nonzero(self) -> None:
        shutil.rmtree(self.run_dir("E5"))
        self.assertNotEqual(self.parse().returncode, 0)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        duplicate = self.root / "runs/duplicate_E0"
        shutil.copytree(self.run_dir("E0"), duplicate)
        self.assertNotEqual(self.parse().returncode, 0)

    def test_missing_required_metric_is_nonzero(self) -> None:
        stderr = self.run_dir("E0") / "stderr"
        stderr.write_text(stderr.read_text().replace(" tokens_per_second=10", ""))
        self.assertNotEqual(self.parse().returncode, 0)

    def test_duplicate_marker_and_key_are_nonzero(self) -> None:
        stderr = self.run_dir("E0") / "stderr"
        text = stderr.read_text()
        perf = next(line for line in text.splitlines() if line.startswith("KV_IDLE_SWAP_RESUME_PERF "))
        stderr.write_text(text + perf + "\n")
        self.assertNotEqual(self.parse().returncode, 0)

        shutil.rmtree(self.root)
        create_artifacts(self.root)
        stderr = self.run_dir("E0") / "stderr"
        stderr.write_text(stderr.read_text().replace(" total_wall_ms=100", " total_wall_ms=100 total_wall_ms=100"))
        self.assertNotEqual(self.parse().returncode, 0)


if __name__ == "__main__":
    unittest.main()
