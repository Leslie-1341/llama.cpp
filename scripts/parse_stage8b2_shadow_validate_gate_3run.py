#!/usr/bin/env python3
"""Parse Stage 8B-2b shadow validation gate 3-run logs."""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import parse_stage8a_policy_matrix as base


POLICY = {
    "S0_paged_off": "paged_off",
    "S1_paged_bookkeeping_shadow_on": "paged_bookkeeping_shadow_on",
    "S2_paged_bookkeeping_shadow_off": "paged_bookkeeping_shadow_off",
    "S3_paged_ingraph_shadow_off": "paged_ingraph_shadow_off",
    "S4_paged_ingraph_gather_shadow_off": "paged_ingraph_gather_shadow_off",
    "S5_prefetch_full_shadow_off": "prefetch_full_shadow_off",
}

DEFAULTS = {
    case: {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    }
    for case in POLICY
}
DEFAULTS["S5_prefetch_full_shadow_off"] = {
    "num_idle": 2,
    "pressure_mode": "off",
    "prefetch_enabled": 1,
    "prefetch_every_tokens": 4,
    "prefetch_blocks_per_step": 4,
    "prefetch_safety_tokens": 0,
}

RUN_RE = re.compile(r"^(?P<case>.+)_run(?P<run_id>[0-9]+)$")

ALL_FIELDS = ["case", "run_id", "policy"] + [
    field for field in base.FIELDS if field not in {"case", "policy"}
]
MEDIAN_FIELDS = ["case", "policy"] + [
    field for field in base.FIELDS if field not in {"case", "policy"}
]
COMPACT_MEDIAN_FIELDS = [
    "case",
    "policy",
    "resume_first_ms",
    "active_tps",
    "seq1_active_ms",
    "total_wall_ms",
    "prefetch_blocks",
    "fallback_blocks",
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
    "prefetch_blocks",
    "fallback_blocks",
]
DELTA_PAIRS = [
    ("S1-S0", "S1_paged_bookkeeping_shadow_on", "S0_paged_off"),
    ("S2-S0", "S2_paged_bookkeeping_shadow_off", "S0_paged_off"),
    ("S3-S2", "S3_paged_ingraph_shadow_off", "S2_paged_bookkeeping_shadow_off"),
    ("S4-S3", "S4_paged_ingraph_gather_shadow_off", "S3_paged_ingraph_shadow_off"),
    ("S5-S4", "S5_prefetch_full_shadow_off", "S4_paged_ingraph_gather_shadow_off"),
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


def parse_run(log_dir: Path, case_name: str, run_id: int, stem: str) -> dict[str, Any]:
    register_case(stem, case_name)
    parsed = base.parse_case(log_dir, stem)
    row: dict[str, Any] = {"case": case_name, "run_id": run_id, "policy": parsed["policy"]}
    for field in base.FIELDS:
        if field not in {"case", "policy"}:
            row[field] = parsed[field]
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
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/oscomp/kv_logs/stage8b2_shadow_validate_gate_3run")
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
