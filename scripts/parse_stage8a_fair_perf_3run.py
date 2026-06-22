#!/usr/bin/env python3
"""Parse Stage 8A fair performance 3-run logs and report medians."""

from __future__ import annotations

import csv
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import parse_stage8a_policy_matrix as base


FAIR_POLICY = {
    "P0_paged_off": "paged_off",
    "P1_paged_on_swap_off_min": "paged_on_swap_off_min",
    "P2_swap_only_min": "swap_only_min",
    "P3_prefetch_g2_min": "prefetch_g2_min",
    "P4_prefetch_g3_min": "prefetch_g3_min",
}

FAIR_DEFAULTS = {
    "P0_paged_off": {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    },
    "P1_paged_on_swap_off_min": {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    },
    "P2_swap_only_min": {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    },
    "P3_prefetch_g2_min": {
        "num_idle": 2,
        "pressure_mode": "off",
        "prefetch_enabled": 1,
        "prefetch_every_tokens": 2,
        "prefetch_blocks_per_step": 1,
        "prefetch_safety_tokens": 0,
    },
    "P4_prefetch_g3_min": {
        "num_idle": 2,
        "pressure_mode": "off",
        "prefetch_enabled": 1,
        "prefetch_every_tokens": 4,
        "prefetch_blocks_per_step": 4,
        "prefetch_safety_tokens": 0,
    },
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
    "prefetch_blocks",
    "fallback_blocks",
    "seq1_decoded",
    "resume_decoded",
    "real_abnormal",
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


def register_fair_case(stem: str, case_name: str) -> None:
    defaults = FAIR_DEFAULTS.get(case_name, base.GENERIC_DEFAULTS)
    base.CASE_DEFAULTS[stem] = defaults
    base.POLICY[stem] = FAIR_POLICY.get(case_name, case_name)


def parse_run(log_dir: Path, case_name: str, run_id: int, stem: str) -> dict[str, Any]:
    register_fair_case(stem, case_name)
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
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/oscomp/kv_logs/stage8a_fair_perf_3run")
    runs = discover_runs(log_dir)
    rows = [parse_run(log_dir, case_name, run_id, stem) for case_name, run_id, stem in runs]
    medians = median_rows(rows)

    write_tsv(log_dir / "summary_all.tsv", ALL_FIELDS, rows)
    write_tsv(log_dir / "summary_median.tsv", MEDIAN_FIELDS, medians)

    print_table(ALL_FIELDS, rows)
    print()
    print("compact median view")
    print_table(COMPACT_MEDIAN_FIELDS, medians)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
