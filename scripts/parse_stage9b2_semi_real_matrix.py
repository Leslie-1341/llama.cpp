#!/usr/bin/env python3
"""Parse Stage 9-B2 semi-real multi-session S0-S5 matrix logs."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


DEFAULT_CASES = [
    "S0_baseline_paged_off",
    "S1_paged_on_reclaim_off",
    "S2_tail_lazy_only",
    "S3_idle_swap_only",
    "S4_idle_swap_prefetch_defer",
    "S5_tail_idle_prefetch_defer",
]

FIELDS = [
    "case",
    "exit_code",
    "total_wall_ms",
    "active_tps",
    "active_decode_tokens",
    "active_decode_ms",
    "rss_kb",
    "summary_count",
    "finished_count",
    "resumed_count",
    "decoded_total",
    "resume_first_A_ms",
    "resume_first_B_ms",
    "resume_first_C_ms",
    "resume_first_D_ms",
    "paged_idle_swap_out_calls",
    "paged_idle_swap_candidates",
    "paged_swap_madvise_calls",
    "paged_swap_madvise_mib",
    "paged_swap_madvise_failures",
    "paged_cov_idle_owned_blocks",
    "paged_cov_idle_owned_mib",
    "paged_cov_resident_safe_blocks",
    "paged_cov_nonidentity_remapped_blocks",
    "paged_idle_swap_skip_protected",
    "paged_idle_swap_skip_deferred",
    "paged_active_restore_from_swapped_blocks",
    "paged_write_to_swapped_block",
    "paged_swapped_active_visible_violation",
    "paged_swapped_active_visible_violation_rows",
    "paged_swapped_active_violation_rows",
    "paged_swapped_active_visible_restore_rows",
    "paged_swap_rss_total_drop_mib",
    "paged_swap_rss_drop_sum_mib",
    "kv_mincore_enabled",
    "fallback_blocks",
    "prefetch_events",
    "prefetch_step_events",
    "real_abnormal",
]

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
ABNORMAL_RE = re.compile(
    r"\b(warn\w*|error|failed|failure|nan|backend_fail|violation|assert|abort|segmentation|sigsegv|segv)\b",
    re.IGNORECASE,
)

ZERO_FIELD_RE = re.compile(
    r"\b(?:"
    r"failures|failure|violation|violations|backend_failures|"
    r"paged_swap_madvise_failures|kv_mincore_failures|"
    r"paged_swapped_active_visible_violation|"
    r"paged_swapped_active_visible_violation_rows"
    r")=0\b",
    re.IGNORECASE,
)

NON_ABNORMAL_FIELD_RE = re.compile(
    r"\bpaged_swapped_active_violation_rows=\d+\b",
    re.IGNORECASE,
)


def parse_values(text: str) -> dict[str, list[str]]:
    vals: dict[str, list[str]] = {}
    for key, val in KV_RE.findall(text):
        vals.setdefault(key.strip(), []).append(val.strip().rstrip(",;"))
    return vals


def parse_line_values(line: str) -> dict[str, str]:
    return {key.strip(): val.strip().rstrip(",;") for key, val in KV_RE.findall(line)}


def num(raw: str | None, default: float = 0.0) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def int_num(raw: str | None, default: int = 0) -> int:
    return int(num(raw, float(default)))


def last_num(vals: dict[str, list[str]], key: str, default: float = 0.0) -> float:
    values = vals.get(key)
    return num(values[-1], default) if values else default


def first_present_last_num(vals: dict[str, list[str]], keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        values = vals.get(key)
        if values:
            return num(values[-1], default)
    return default


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_exit(path: Path) -> int:
    if not path.exists():
        return 127
    try:
        return int(read_text(path).strip() or "0")
    except ValueError:
        return 127


def fmt(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def event_count(text: str, event: str) -> int:
    total = 0
    for line in text.splitlines():
        if "KV_SEMI_SESSION" not in line:
            continue
        vals = parse_line_values(line)
        if vals.get("event") == event:
            total += 1
    return total


def last_idle_trace_values(text: str) -> dict[str, list[str]]:
    last = ""
    for line in text.splitlines():
        if "KV_PAGED_IDLE_TRACE" in line:
            last = line
    return parse_values(last) if last else {}


def abnormal_lines(text: str) -> list[str]:
    matches: list[str] = []
    for line in text.splitlines():
        if "n_ctx_seq" in line and "n_ctx_train" in line:
            continue
        scrubbed = ZERO_FIELD_RE.sub("", line)
        scrubbed = NON_ABNORMAL_FIELD_RE.sub("", scrubbed)
        if ABNORMAL_RE.search(scrubbed):
            matches.append(line)
    return matches


def discover_cases(log_dir: Path) -> list[str]:
    names: set[str] = set()
    for suffix in ("*.out", "*.err", "*.exit"):
        names.update(path.stem for path in log_dir.glob(suffix))
    if not names:
        return DEFAULT_CASES
    return [case for case in DEFAULT_CASES if case in names] + sorted(names - set(DEFAULT_CASES))


def parse_case(log_dir: Path, case: str) -> tuple[dict[str, object], list[str]]:
    out_text = read_text(log_dir / f"{case}.out")
    err_text = read_text(log_dir / f"{case}.err")
    text = out_text + "\n" + err_text
    vals = parse_values(text)
    idle_vals = last_idle_trace_values(text)

    summaries = []
    for line in text.splitlines():
        if "KV_SEMI_SUMMARY" in line:
            summaries.append(parse_line_values(line))

    by_name = {summary.get("name", ""): summary for summary in summaries}
    abnormal = abnormal_lines(text)

    row: dict[str, object] = {
        "case": case,
        "exit_code": read_exit(log_dir / f"{case}.exit"),
        "total_wall_ms": last_num(vals, "total_wall_ms"),
        "active_tps": last_num(vals, "active_tps"),
        "active_decode_tokens": int(last_num(vals, "active_decode_tokens")),
        "active_decode_ms": last_num(vals, "active_decode_ms"),
        "rss_kb": int(last_num(vals, "rss_kb")),
        "summary_count": len(summaries),
        "finished_count": sum(int_num(summary.get("finished")) for summary in summaries),
        "resumed_count": sum(int_num(summary.get("resumed")) for summary in summaries),
        "decoded_total": sum(int_num(summary.get("decoded")) for summary in summaries),
        "resume_first_A_ms": num(by_name.get("A", {}).get("resume_first_ms")),
        "resume_first_B_ms": num(by_name.get("B", {}).get("resume_first_ms")),
        "resume_first_C_ms": num(by_name.get("C", {}).get("resume_first_ms")),
        "resume_first_D_ms": num(by_name.get("D", {}).get("resume_first_ms")),
        "paged_idle_swap_out_calls": int(last_num(idle_vals, "paged_idle_swap_out_calls")),
        "paged_idle_swap_candidates": int(last_num(idle_vals, "paged_idle_swap_candidates")),
        "paged_swap_madvise_calls": int(last_num(idle_vals, "paged_swap_madvise_calls")),
        "paged_swap_madvise_mib": last_num(idle_vals, "paged_swap_madvise_bytes") / 1048576.0,
        "paged_swap_madvise_failures": int(last_num(idle_vals, "paged_swap_madvise_failures")),
        "paged_cov_idle_owned_blocks": int(last_num(idle_vals, "paged_cov_idle_owned_blocks")),
        "paged_cov_idle_owned_mib": last_num(idle_vals, "paged_cov_idle_owned_bytes") / 1048576.0,
        "paged_cov_resident_safe_blocks": int(last_num(idle_vals, "paged_cov_resident_safe_blocks")),
        "paged_cov_nonidentity_remapped_blocks": int(last_num(idle_vals, "paged_cov_nonidentity_remapped_blocks")),
        "paged_idle_swap_skip_protected": int(last_num(idle_vals, "paged_idle_swap_skip_protected")),
        "paged_idle_swap_skip_deferred": int(last_num(idle_vals, "paged_idle_swap_skip_deferred")),
        "paged_active_restore_from_swapped_blocks": int(last_num(idle_vals, "paged_active_restore_from_swapped_blocks")),
        "paged_write_to_swapped_block": int(last_num(idle_vals, "paged_write_to_swapped_block")),
        "paged_swapped_active_visible_violation": int(last_num(idle_vals, "paged_swapped_active_visible_violation")),
        "paged_swapped_active_visible_violation_rows": int(last_num(idle_vals, "paged_swapped_active_visible_violation_rows")),
        "paged_swapped_active_violation_rows": int(last_num(idle_vals, "paged_swapped_active_violation_rows")),
        "paged_swapped_active_visible_restore_rows": int(last_num(idle_vals, "paged_swapped_active_visible_restore_rows")),
        "paged_swap_rss_total_drop_mib": last_num(idle_vals, "paged_swap_rss_total_drop_kb") / 1024.0,
        "paged_swap_rss_drop_sum_mib": last_num(idle_vals, "paged_swap_rss_drop_sum_kb") / 1024.0,
        "kv_mincore_enabled": int(last_num(idle_vals, "kv_mincore_enabled")),
        "fallback_blocks": int(first_present_last_num(vals, ["fallback_blocks", "paged_fallback_blocks", "paged_swap_fallback_blocks"])),
        "prefetch_events": event_count(text, "prefetch"),
        "prefetch_step_events": event_count(text, "prefetch_step"),
        "real_abnormal": len(abnormal),
    }
    return row, abnormal


def write_abnormal_summary(path: Path, case_lines: list[tuple[str, list[str]]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        total = 0
        for case, lines in case_lines:
            total += len(lines)
            f.write(f"{case}\treal_abnormal={len(lines)}\n")
            for line in lines[:20]:
                f.write(f"  {line}\n")
            if len(lines) > 20:
                f.write(f"  ... {len(lines) - 20} more\n")
        f.write(f"TOTAL\treal_abnormal={total}\n")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {Path(argv[0]).name} LOGDIR", file=sys.stderr)
        return 2

    log_dir = Path(argv[1])
    rows = []
    case_lines = []
    for case in discover_cases(log_dir):
        row, lines = parse_case(log_dir, case)
        rows.append(row)
        case_lines.append((case, lines))

    with (log_dir / "summary.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key, 0)) for key in FIELDS})

    write_abnormal_summary(log_dir / "abnormal_summary.txt", case_lines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
