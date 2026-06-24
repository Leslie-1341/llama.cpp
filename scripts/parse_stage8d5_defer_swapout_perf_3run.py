#!/usr/bin/env python3
"""Parse Stage 8D-5 defer swap-out formal 3-run perf logs.

seq1_equal and resume_equal use cmp-style status: 0 means the extracted output
block matches F0 for the same run, non-zero means it differs or is unavailable.
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import parse_stage8a_policy_matrix as base


POLICY = {
    "F0_paged_swap_off": "paged_swap_off",
    "F1_full_prefetch_no_defer": "full_prefetch_no_defer",
    "F2_full_prefetch_defer": "full_prefetch_defer",
    "F3_full_prefetch_earlier_defer": "full_prefetch_earlier_defer",
}

DEFAULTS = {
    "F0_paged_swap_off": {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    },
}
for case in (
    "F1_full_prefetch_no_defer",
    "F2_full_prefetch_defer",
    "F3_full_prefetch_earlier_defer",
):
    DEFAULTS[case] = {
        "num_idle": 2,
        "pressure_mode": "off",
        "prefetch_enabled": 1,
        "prefetch_every_tokens": 4,
        "prefetch_blocks_per_step": 4,
        "prefetch_safety_tokens": 0,
    }

RUN_RE = re.compile(r"^(?P<case>.+)_run(?P<run_id>[0-9]+)$")

EXTRA_FIELDS = ["seq1_equal", "resume_equal"]
ALL_FIELDS = ["case", "run_id", "policy"] + [
    field for field in base.FIELDS if field not in {"case", "policy"}
] + EXTRA_FIELDS
MEDIAN_FIELDS = ["case", "policy"] + [
    field for field in base.FIELDS if field not in {"case", "policy"}
] + EXTRA_FIELDS
COMPACT_MEDIAN_FIELDS = [
    "case",
    "policy",
    "resume_first_ms",
    "active_tps",
    "seq1_active_ms",
    "total_wall_ms",
    "process_rss_drop_mib",
    "madvise_mib",
    "prefetch_blocks",
    "fallback_blocks",
    "prefetch_remaining_blocks_before_resume",
    "seq1_equal",
    "resume_equal",
    "seq1_decoded",
    "resume_decoded",
    "real_abnormal",
]
DELTA_FIELDS = [
    "delta",
    "resume_first_ms",
    "active_tps",
    "seq1_active_ms",
    "total_wall_ms",
    "process_rss_drop_mib",
    "madvise_mib",
    "prefetch_blocks",
    "fallback_blocks",
    "prefetch_remaining_blocks_before_resume",
]
DELTA_PAIRS = [
    ("F1-F0", "F1_full_prefetch_no_defer", "F0_paged_swap_off"),
    ("F2-F1", "F2_full_prefetch_defer", "F1_full_prefetch_no_defer"),
    ("F3-F1", "F3_full_prefetch_earlier_defer", "F1_full_prefetch_no_defer"),
    ("F2-F0", "F2_full_prefetch_defer", "F0_paged_swap_off"),
    ("F3-F0", "F3_full_prefetch_earlier_defer", "F0_paged_swap_off"),
]


def fmt(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def discover_runs(log_dir: Path) -> list[tuple[str, int, str]]:
    runs: list[tuple[str, int, str]] = []
    for err_path in sorted(log_dir.glob("*_run*.err")):
        match = RUN_RE.match(err_path.stem)
        if not match:
            continue
        runs.append((match.group("case"), int(match.group("run_id")), err_path.stem))
    return sorted(runs, key=lambda item: (item[0], item[1]))


def register_case(stem: str, case_name: str) -> None:
    base.CASE_DEFAULTS[stem] = DEFAULTS.get(case_name, base.GENERIC_DEFAULTS)
    base.POLICY[stem] = POLICY.get(case_name, case_name)


def extract_block(text: str, begin: str, end: str) -> str | None:
    pattern = re.compile(
        rf"^{re.escape(begin)}\n(?P<body>.*?)^{re.escape(end)}$",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        return None
    return match.group("body")


def read_out(log_dir: Path, stem: str) -> str:
    path = log_dir / f"{stem}.out"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def output_blocks(log_dir: Path, stem: str) -> tuple[str | None, str | None]:
    out_text = read_out(log_dir, stem)
    seq1 = extract_block(out_text, "===SEQ1_ACTIVE_BEGIN===", "===SEQ1_ACTIVE_END===")
    resume = extract_block(out_text, "===SEQ0_RESUME_BEGIN===", "===SEQ0_RESUME_END===")
    return seq1, resume


def cmp_status(actual: str | None, expected: str | None) -> int:
    if actual is None or expected is None:
        return 1
    return 0 if actual == expected else 1


def correctness_status(log_dir: Path, case_name: str, run_id: int, stem: str) -> tuple[int, int]:
    seq1, resume = output_blocks(log_dir, stem)
    if case_name == "F0_paged_swap_off":
        return (0 if seq1 is not None else 1, 0 if resume is not None else 1)

    base_stem = f"F0_paged_swap_off_run{run_id}"
    base_seq1, base_resume = output_blocks(log_dir, base_stem)
    return cmp_status(seq1, base_seq1), cmp_status(resume, base_resume)


def parse_run(log_dir: Path, case_name: str, run_id: int, stem: str) -> dict[str, Any]:
    register_case(stem, case_name)
    parsed = base.parse_case(log_dir, stem)
    seq1_equal, resume_equal = correctness_status(log_dir, case_name, run_id, stem)
    row: dict[str, Any] = {"case": case_name, "run_id": run_id, "policy": parsed["policy"]}
    for field in base.FIELDS:
        if field not in {"case", "policy"}:
            row[field] = parsed[field]
    row["seq1_equal"] = seq1_equal
    row["resume_equal"] = resume_equal
    return row


def median_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case"]), []).append(row)

    medians: list[dict[str, Any]] = []
    for case_name in sorted(by_case):
        case_rows = sorted(by_case[case_name], key=lambda row: int(row["run_id"]))
        median: dict[str, Any] = {
            "case": case_name,
            "policy": case_rows[0]["policy"],
        }
        for field in MEDIAN_FIELDS:
            if field in {"case", "policy"}:
                continue
            values = [row[field] for row in case_rows]
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
                median[field] = int(statistics.median(values))
            elif all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
                median[field] = float(statistics.median(values))
            else:
                median[field] = values[len(values) // 2]
        medians.append(median)
    return medians


def delta_rows(medians: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case = {str(row["case"]): row for row in medians}
    rows: list[dict[str, Any]] = []
    for label, lhs, rhs in DELTA_PAIRS:
        if lhs not in by_case or rhs not in by_case:
            continue
        row: dict[str, Any] = {"delta": label}
        for field in DELTA_FIELDS:
            if field == "delta":
                continue
            row[field] = by_case[lhs][field] - by_case[rhs][field]
        rows.append(row)
    return rows


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row[field]) for field in fields})


def print_table(fields: list[str], rows: list[dict[str, Any]]) -> None:
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(fmt(row[field])) for field in fields))


def main() -> int:
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/oscomp/kv_logs/stage8d5_defer_swapout_perf_3run")
    runs = discover_runs(log_dir)
    rows = [parse_run(log_dir, case_name, run_id, stem) for case_name, run_id, stem in runs]
    medians = median_rows(rows)

    write_tsv(log_dir / "summary_all.tsv", ALL_FIELDS, rows)
    write_tsv(log_dir / "summary_median.tsv", MEDIAN_FIELDS, medians)

    print_table(ALL_FIELDS, rows)
    print()
    print("compact median view")
    print_table(COMPACT_MEDIAN_FIELDS, medians)
    print()
    print("compact delta view")
    print_table(DELTA_FIELDS, delta_rows(medians))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
