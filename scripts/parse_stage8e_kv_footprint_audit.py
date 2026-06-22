#!/usr/bin/env python3
"""Parse the Stage 8E-B KV footprint / residency audit.

This is a diagnostic memory-accounting parser. It does not introduce any new
optimization; it only reads the existing key=value telemetry emitted by
build/bin/llama-kv-idle-swap-resume and reports KV capacity, in-use, idle-owned,
swapped, residency, process RSS drop and a set of derived ratios.

It reuses the key=value extraction helpers from parse_stage8a_policy_matrix.py
so the value-scraping logic stays in one place.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_stage8a_policy_matrix as base  # noqa: E402


# B0/B1 map onto the Stage 8D-5 F0/F2 cases (see runner header).
POLICY = {
    "B0_paged_swap_off_mincore": "paged_on_swap_off",
    "B1_full_prefetch_defer_mincore": "full_prefetch_defer",
}

BASELINE_CASE = "B0_paged_swap_off_mincore"

MIB = 1024.0 * 1024.0


def mib_bytes(vals: dict[str, list[str]], *keys: str) -> float:
    return base.max_num(vals, *keys) / MIB


def last_nonzero_num(vals: dict[str, list[str]], key: str, default: float = 0.0) -> float:
    for raw in reversed(vals.get(key, [])):
        value = base.number(raw, default)
        if value != 0:
            return value
    return default


def last_nonzero_mib(vals: dict[str, list[str]], key: str) -> float:
    return last_nonzero_num(vals, key) / MIB


def last_mib(vals: dict[str, list[str]], key: str) -> float:
    return base.last_num(vals, key) / MIB


def max_first_available(vals: dict[str, list[str]], *keys: str) -> float:
    for key in keys:
        if key in vals and vals[key]:
            return base.max_num(vals, key)
    return 0.0


def pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den * 100.0


FIELDS = [
    # basics
    "case",
    "run_id",
    "exit_code",
    "policy",
    # process RSS
    "process_rss_before_mib",
    "process_rss_after_madvise_mib",
    "process_rss_drop_mib",
    "process_rss_drop_sum_mib",
    "process_rss_drop_max_mib",
    "process_rss_drop_pct",
    # KV capacity / in-use
    "kv_capacity_blocks",
    "kv_block_size_cells",
    "kv_total_capacity_mib",
    "kv_block_size_mib",
    "blocks_in_use",
    "kv_in_use_mib",
    "kv_free_blocks",
    # idle / swapped / madvise
    "idle_owned_blocks",
    "idle_owned_mib",
    "swapped_blocks",
    "swapped_mib",
    "madvise_calls",
    "madvise_mib",
    "madvise_failures",
    # mincore residency
    "kv_mincore_total_mib",
    "kv_mincore_resident_mib",
    "kv_mincore_nonresident_mib",
    "kv_mincore_resident_pct",
    "swapped_total_mib",
    "swapped_resident_mib",
    "swapped_nonresident_mib",
    "swapped_nonresident_peak_mib",
    "swapped_nonresident_pct",
    "kv_resident_before_madvise_mib",
    "kv_resident_after_madvise_mib",
    "kv_resident_after_resume_mib",
    "kv_resident_drop_mib",
    "kv_resident_recover_mib",
    # ratios
    "rss_drop_over_madvise_pct",
    "rss_drop_over_idle_owned_pct",
    "rss_drop_over_kv_capacity_pct",
    "rss_drop_over_kv_inuse_pct",
    "kv_nonresident_over_capacity_pct",
    "kv_nonresident_over_inuse_pct",
    "swapped_nonresident_over_swapped_pct",
    "kv_capacity_over_process_rss_pct",
    "kv_inuse_over_process_rss_pct",
    "kv_resident_over_process_rss_pct",
    # correctness / safety
    "seq1_decoded",
    "resume_decoded",
    "seq1_equal",
    "resume_equal",
    "real_abnormal",
    "active_visible_violation",
    "visible_violation_rows",
    "active_violation_rows",
]

COMPACT_FIELDS = [
    "case",
    "kv_total_capacity_mib",
    "kv_in_use_mib",
    "kv_mincore_nonresident_mib",
    "idle_owned_mib",
    "swapped_nonresident_mib",
    "swapped_nonresident_peak_mib",
    "madvise_mib",
    "process_rss_drop_mib",
    "kv_resident_drop_mib",
    "rss_drop_over_kv_inuse_pct",
    "rss_drop_over_madvise_pct",
    "seq1_equal",
    "resume_equal",
    "real_abnormal",
]

CORRECTNESS_FIELDS = [
    "case",
    "seq1_decoded",
    "resume_decoded",
    "seq1_equal",
    "resume_equal",
    "active_visible_violation",
    "visible_violation_rows",
    "active_violation_rows",
    "real_abnormal",
]

# abnormal filtering: drop these zero-valued counters before scanning.
ABNORMAL_DROP_PATTERNS = [
    re.compile(r"\bfailures?=0\b"),
    re.compile(r"\bviolations?=0\b"),
    re.compile(r"\bbackend_failures=0\b"),
    re.compile(r"\bpaged_swap_madvise_failures=0\b"),
    re.compile(r"\bkv_mincore_failures=0\b"),
    re.compile(r"[a-z_]*failures?=0\b"),
    re.compile(r"[a-z_]*violations?=0\b"),
]

ABNORMAL_KEEP = re.compile(
    r"\b("
    r"warn|error|failed|failure|nan|backend_fail|violation|assert|abort|"
    r"segmentation|sigsegv|segv"
    r")\b",
    re.IGNORECASE,
)


def extract_block(text: str, begin: str, end: str) -> str | None:
    pattern = re.compile(
        rf"^{re.escape(begin)}\n(?P<body>.*?)^{re.escape(end)}$",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group("body") if match else None


def read_text(log_dir: Path, case: str, suffix: str) -> str:
    path = log_dir / f"{case}.{suffix}"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def output_blocks(log_dir: Path, case: str) -> tuple[str | None, str | None]:
    out_text = read_text(log_dir, case, "out")
    seq1 = extract_block(out_text, "===SEQ1_ACTIVE_BEGIN===", "===SEQ1_ACTIVE_END===")
    resume = extract_block(out_text, "===SEQ0_RESUME_BEGIN===", "===SEQ0_RESUME_END===")
    return seq1, resume


def cmp_status(actual: str | None, expected: str | None) -> int:
    if actual is None or expected is None:
        return 1
    return 0 if actual == expected else 1


def correctness_status(
    log_dir: Path,
    case: str,
    base_blocks: tuple[str | None, str | None],
) -> tuple[int, int]:
    seq1, resume = output_blocks(log_dir, case)
    if case == BASELINE_CASE:
        return (0 if seq1 is not None else 1, 0 if resume is not None else 1)
    base_seq1, base_resume = base_blocks
    return cmp_status(seq1, base_seq1), cmp_status(resume, base_resume)


def collect_abnormal(log_dir: Path, case: str) -> tuple[int, int]:
    text = read_text(log_dir, case, "out") + "\n" + read_text(log_dir, case, "err")
    raw = 0
    real = 0
    real_lines: list[str] = []
    for line in text.splitlines():
        if not ABNORMAL_KEEP.search(line):
            continue
        raw += 1
        scrubbed = line
        for pat in ABNORMAL_DROP_PATTERNS:
            scrubbed = pat.sub("", scrubbed)
        if ABNORMAL_KEEP.search(scrubbed):
            real += 1
            real_lines.append(line.strip())
    return raw, real


def abnormal_lines(log_dir: Path, case: str) -> tuple[list[str], list[str]]:
    text = read_text(log_dir, case, "out") + "\n" + read_text(log_dir, case, "err")
    raw_lines: list[str] = []
    real_lines: list[str] = []
    for line in text.splitlines():
        if not ABNORMAL_KEEP.search(line):
            continue
        raw_lines.append(line.strip())
        scrubbed = line
        for pat in ABNORMAL_DROP_PATTERNS:
            scrubbed = pat.sub("", scrubbed)
        if ABNORMAL_KEEP.search(scrubbed):
            real_lines.append(line.strip())
    return raw_lines, real_lines


def parse_case(log_dir: Path, case: str, base_blocks: tuple[str | None, str | None]) -> dict[str, object]:
    text = read_text(log_dir, case, "out") + "\n" + read_text(log_dir, case, "err")
    vals = base.parse_values(text)

    # --- process RSS ---
    process_rss_before_mib = last_nonzero_num(vals, "paged_swap_rss_before_first_kb") / 1024.0
    process_rss_after_madvise_mib = last_nonzero_num(vals, "paged_swap_rss_after_last_kb") / 1024.0
    process_rss_drop_mib = max(0.0, process_rss_before_mib - process_rss_after_madvise_mib)
    process_rss_drop_sum_mib = base.max_num(vals, "paged_swap_rss_drop_sum_kb") / 1024.0
    process_rss_drop_max_mib = base.max_num(vals, "paged_swap_rss_drop_max_kb") / 1024.0
    process_rss_drop_pct = pct(process_rss_drop_mib, process_rss_before_mib)

    # --- idle / swapped / madvise ---
    idle_owned_blocks = base.max_num(vals, "paged_cov_idle_owned_blocks", "idle_owned_blocks")
    idle_owned_mib = mib_bytes(vals, "paged_cov_idle_owned_bytes")
    swapped_blocks = base.max_num(vals, "kv_mincore_swapped_block_count")
    swapped_mib = mib_bytes(vals, "kv_mincore_swapped_total_bytes")
    madvise_calls = base.max_num(vals, "paged_swap_madvise_calls")
    madvise_mib = mib_bytes(vals, "paged_swap_madvise_bytes")
    madvise_failures = base.max_num(vals, "paged_swap_madvise_failures")

    # --- KV capacity / in-use ---
    kv_capacity_blocks = base.max_num(vals, "paged_n_blocks")
    kv_block_size_cells = base.max_num(vals, "paged_block_size")
    kv_total_capacity_mib = last_nonzero_mib(vals, "kv_mincore_total_bytes")
    if idle_owned_blocks > 0:
        kv_block_size_mib = idle_owned_mib / idle_owned_blocks
    elif swapped_blocks > 0:
        kv_block_size_mib = swapped_mib / swapped_blocks
    elif kv_total_capacity_mib > 0 and kv_capacity_blocks > 0:
        kv_block_size_mib = kv_total_capacity_mib / kv_capacity_blocks
    else:
        kv_block_size_mib = 0.0
    blocks_in_use = max_first_available(vals, "paged_blocks_in_use", "blocks_in_use", "non_empty_blocks")
    kv_in_use_mib = blocks_in_use * kv_block_size_mib
    kv_free_blocks = kv_capacity_blocks - blocks_in_use if kv_capacity_blocks > 0 else 0

    # --- mincore residency ---
    kv_mincore_total_mib = last_nonzero_mib(vals, "kv_mincore_total_bytes")
    kv_mincore_resident_mib = last_nonzero_mib(vals, "kv_mincore_resident_bytes")
    kv_mincore_nonresident_mib = max(0.0, kv_mincore_total_mib - kv_mincore_resident_mib)
    kv_mincore_resident_pct = pct(kv_mincore_resident_mib, kv_mincore_total_mib)

    swapped_total_mib = mib_bytes(vals, "kv_mincore_swapped_total_bytes")
    swapped_resident_mib = last_mib(vals, "kv_mincore_swapped_resident_bytes")
    swapped_nonresident_mib = last_nonzero_mib(vals, "kv_mincore_swapped_nonresident_bytes")
    swapped_nonresident_peak_mib = mib_bytes(vals, "kv_mincore_swapped_nonresident_bytes")
    swapped_nonresident_pct = pct(swapped_nonresident_mib, swapped_total_mib)

    kv_resident_before_madvise_mib = mib_bytes(vals, "kv_mincore_before_madvise_resident_bytes")
    kv_resident_after_madvise_mib = mib_bytes(vals, "kv_mincore_after_madvise_resident_bytes")
    kv_resident_after_resume_mib = mib_bytes(vals, "kv_mincore_after_resume_resident_bytes")
    kv_resident_drop_mib = mib_bytes(vals, "kv_mincore_madvise_drop_bytes")
    kv_resident_recover_mib = mib_bytes(vals, "kv_mincore_resume_recover_bytes")

    # --- ratios ---
    rss_drop_over_madvise_pct = pct(process_rss_drop_mib, madvise_mib)
    rss_drop_over_idle_owned_pct = pct(process_rss_drop_mib, idle_owned_mib)
    rss_drop_over_kv_capacity_pct = pct(process_rss_drop_mib, kv_total_capacity_mib)
    rss_drop_over_kv_inuse_pct = pct(process_rss_drop_mib, kv_in_use_mib)
    kv_nonresident_over_capacity_pct = pct(kv_mincore_nonresident_mib, kv_total_capacity_mib)
    kv_nonresident_over_inuse_pct = pct(kv_mincore_nonresident_mib, kv_in_use_mib)
    swapped_nonresident_over_swapped_pct = pct(swapped_nonresident_mib, swapped_total_mib)
    kv_capacity_over_process_rss_pct = pct(kv_total_capacity_mib, process_rss_before_mib)
    kv_inuse_over_process_rss_pct = pct(kv_in_use_mib, process_rss_before_mib)
    kv_resident_over_process_rss_pct = pct(kv_mincore_resident_mib, process_rss_before_mib)

    # --- correctness / safety ---
    seq1_equal, resume_equal = correctness_status(log_dir, case, base_blocks)
    active_visible_violation = base.max_num(
        vals, "active_visible_violation", "paged_swapped_active_visible_violation"
    )
    visible_violation_rows = base.max_num(
        vals, "visible_violation_rows", "paged_swapped_active_visible_violation_rows"
    )
    active_violation_rows = base.max_num(
        vals, "active_violation_rows", "paged_swapped_active_violation_rows"
    )

    row: dict[str, object] = {
        "case": case,
        "run_id": 1,
        "exit_code": base.read_exit(log_dir / f"{case}.exit"),
        "policy": POLICY.get(case, case),
        "process_rss_before_mib": process_rss_before_mib,
        "process_rss_after_madvise_mib": process_rss_after_madvise_mib,
        "process_rss_drop_mib": process_rss_drop_mib,
        "process_rss_drop_sum_mib": process_rss_drop_sum_mib,
        "process_rss_drop_max_mib": process_rss_drop_max_mib,
        "process_rss_drop_pct": process_rss_drop_pct,
        "kv_capacity_blocks": int(kv_capacity_blocks),
        "kv_block_size_cells": int(kv_block_size_cells),
        "kv_total_capacity_mib": kv_total_capacity_mib,
        "kv_block_size_mib": kv_block_size_mib,
        "blocks_in_use": int(blocks_in_use),
        "kv_in_use_mib": kv_in_use_mib,
        "kv_free_blocks": int(kv_free_blocks),
        "idle_owned_blocks": int(idle_owned_blocks),
        "idle_owned_mib": idle_owned_mib,
        "swapped_blocks": int(swapped_blocks),
        "swapped_mib": swapped_mib,
        "madvise_calls": int(madvise_calls),
        "madvise_mib": madvise_mib,
        "madvise_failures": int(madvise_failures),
        "kv_mincore_total_mib": kv_mincore_total_mib,
        "kv_mincore_resident_mib": kv_mincore_resident_mib,
        "kv_mincore_nonresident_mib": kv_mincore_nonresident_mib,
        "kv_mincore_resident_pct": kv_mincore_resident_pct,
        "swapped_total_mib": swapped_total_mib,
        "swapped_resident_mib": swapped_resident_mib,
        "swapped_nonresident_mib": swapped_nonresident_mib,
        "swapped_nonresident_peak_mib": swapped_nonresident_peak_mib,
        "swapped_nonresident_pct": swapped_nonresident_pct,
        "kv_resident_before_madvise_mib": kv_resident_before_madvise_mib,
        "kv_resident_after_madvise_mib": kv_resident_after_madvise_mib,
        "kv_resident_after_resume_mib": kv_resident_after_resume_mib,
        "kv_resident_drop_mib": kv_resident_drop_mib,
        "kv_resident_recover_mib": kv_resident_recover_mib,
        "rss_drop_over_madvise_pct": rss_drop_over_madvise_pct,
        "rss_drop_over_idle_owned_pct": rss_drop_over_idle_owned_pct,
        "rss_drop_over_kv_capacity_pct": rss_drop_over_kv_capacity_pct,
        "rss_drop_over_kv_inuse_pct": rss_drop_over_kv_inuse_pct,
        "kv_nonresident_over_capacity_pct": kv_nonresident_over_capacity_pct,
        "kv_nonresident_over_inuse_pct": kv_nonresident_over_inuse_pct,
        "swapped_nonresident_over_swapped_pct": swapped_nonresident_over_swapped_pct,
        "kv_capacity_over_process_rss_pct": kv_capacity_over_process_rss_pct,
        "kv_inuse_over_process_rss_pct": kv_inuse_over_process_rss_pct,
        "kv_resident_over_process_rss_pct": kv_resident_over_process_rss_pct,
        "seq1_decoded": int(base.last_num(vals, "seq1_decoded_tokens")),
        "resume_decoded": int(base.last_num(vals, "seq0_resume_decoded_tokens")),
        "seq1_equal": seq1_equal,
        "resume_equal": resume_equal,
        "real_abnormal": base.has_real_abnormal(text),
        "active_visible_violation": int(active_visible_violation),
        "visible_violation_rows": int(visible_violation_rows),
        "active_violation_rows": int(active_violation_rows),
    }
    return row


def discover_cases(log_dir: Path) -> list[str]:
    cases = sorted(path.stem for path in log_dir.glob("*.exit"))
    if cases:
        return cases
    return [BASELINE_CASE, "B1_full_prefetch_defer_mincore"]


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: base.fmt(row[field]) for field in fields})


def print_table(fields: list[str], rows: list[dict[str, object]]) -> None:
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(base.fmt(row[field])) for field in fields))


def write_abnormal_summary(path: Path, log_dir: Path, cases: list[str]) -> None:
    lines: list[str] = []
    for case in cases:
        raw_lines, real_lines = abnormal_lines(log_dir, case)
        lines.append(f"===== {case} =====")
        lines.append(f"raw_abnormal_matches={len(raw_lines)}")
        lines.append(f"real_abnormal_matches={len(real_lines)}")
        if real_lines:
            lines.append("--- real_abnormal lines ---")
            lines.extend(real_lines)
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    log_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("/root/oscomp/kv_logs/stage8e_kv_footprint_audit")
    )
    cases = discover_cases(log_dir)
    base_blocks = output_blocks(log_dir, BASELINE_CASE)
    rows = [parse_case(log_dir, case, base_blocks) for case in cases]

    write_tsv(log_dir / "summary.tsv", FIELDS, rows)
    # single-run: median is the same row set, aggregated one row per case.
    write_tsv(log_dir / "summary_median.tsv", FIELDS, rows)
    write_abnormal_summary(log_dir / "abnormal_summary.txt", log_dir, cases)

    print("===== full summary =====")
    print_table(FIELDS, rows)

    print()
    print("===== compact footprint view =====")
    compact_rows = [{field: row[field] for field in COMPACT_FIELDS} for row in rows]
    print_table(COMPACT_FIELDS, compact_rows)

    print()
    print("===== correctness view =====")
    correctness_rows = [{field: row[field] for field in CORRECTNESS_FIELDS} for row in rows]
    print_table(CORRECTNESS_FIELDS, correctness_rows)

    print()
    print("===== abnormal summary =====")
    for case in cases:
        raw_lines, real_lines = abnormal_lines(log_dir, case)
        print(f"{case}\traw_abnormal_matches={len(raw_lines)}\treal_abnormal_matches={len(real_lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
