#!/usr/bin/env python3
"""Parse one Stage 8A policy sanity matrix run.

net_rss_drop_before_resume_mib is intentionally emitted as 0 for now: the
current log stream does not expose a reliable before-idle-swap to before-resume
RSS pair across all policy cases.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


DEFAULT_CASES = [
    "A0_paged_off",
    "A1_paged_on_swap_off",
    "B0_swap_only",
    "C1_medium",
    "C3_off",
]

POLICY = {
    "A0_paged_off": "paged_off",
    "A1_paged_on_swap_off": "paged_on_swap_off",
    "B0_swap_only": "swap_only",
    "C1_medium": "medium_prefetch",
    "C3_off": "full_active_window_prefetch",
}

CASE_DEFAULTS = {
    "A0_paged_off": {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    },
    "A1_paged_on_swap_off": {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    },
    "B0_swap_only": {
        "num_idle": 2,
        "pressure_mode": "none",
        "prefetch_enabled": 0,
        "prefetch_every_tokens": 0,
        "prefetch_blocks_per_step": 0,
        "prefetch_safety_tokens": 0,
    },
    "C1_medium": {
        "num_idle": 2,
        "pressure_mode": "medium",
        "prefetch_enabled": 1,
        "prefetch_every_tokens": 4,
        "prefetch_blocks_per_step": 1,
        "prefetch_safety_tokens": 0,
    },
    "C3_off": {
        "num_idle": 2,
        "pressure_mode": "off",
        "prefetch_enabled": 1,
        "prefetch_every_tokens": 4,
        "prefetch_blocks_per_step": 1,
        "prefetch_safety_tokens": 0,
    },
}

GENERIC_DEFAULTS = {
    "num_idle": 2,
    "pressure_mode": "none",
    "prefetch_enabled": 0,
    "prefetch_every_tokens": 0,
    "prefetch_blocks_per_step": 0,
    "prefetch_safety_tokens": 0,
}

FIELDS = [
    "case",
    "policy",
    "exit_code",
    "num_idle",
    "pressure_mode",
    "prefetch_enabled",
    "prefetch_every_tokens",
    "prefetch_blocks_per_step",
    "prefetch_safety_tokens",
    "idle_owned_blocks",
    "swapped_blocks",
    "madvise_mib",
    "process_rss_drop_mib",
    "kv_nonresident_mib",
    "kv_resident_mib",
    "whole_kv_range_drop_mib",
    "net_rss_drop_before_resume_mib",
    "prefetch_blocks",
    "restore_blocks",
    "fallback_blocks",
    "target_restore_blocks",
    "prefetch_auto_window_ok",
    "prefetch_auto_need_steps",
    "prefetch_auto_start_token",
    "prefetch_auto_effective_start_token",
    "prefetch_remaining_blocks_before_resume",
    "resume_first_ms",
    "resume_total_ms",
    "seq1_active_ms",
    "seq1_active_tokens",
    "active_tps",
    "total_wall_ms",
    "seq1_decoded",
    "resume_decoded",
    "active_visible_violation",
    "visible_violation_rows",
    "active_violation_rows",
    "real_abnormal",
]

COMPACT_FIELDS = [
    "case",
    "policy",
    "swapped",
    "kv_nonresident",
    "rss_drop",
    "prefetch",
    "fallback",
    "resume_first",
    "active_tps",
    "decoded",
    "abnormal",
]

KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def parse_values(text: str) -> dict[str, list[str]]:
    vals: dict[str, list[str]] = {}
    for key, val in KV_RE.findall(text):
        vals.setdefault(key, []).append(val.rstrip(",;"))
    return vals


def number(raw: str | None, default: float = 0.0) -> float:
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def last_num(vals: dict[str, list[str]], key: str, default: float = 0.0) -> float:
    if key not in vals or not vals[key]:
        return default
    return number(vals[key][-1], default)


def max_num(vals: dict[str, list[str]], *keys: str) -> float:
    found: list[float] = []
    for key in keys:
        found.extend(number(v) for v in vals.get(key, []))
    return max(found) if found else 0.0


def max_num_in_prefixed_lines(text: str, prefix: str, key: str) -> float:
    found: list[float] = []
    pattern = re.compile(rf"\b{re.escape(key)}=([^\s]+)")
    for line in text.splitlines():
        if prefix not in line:
            continue
        found.extend(number(match.group(1).rstrip(",;")) for match in pattern.finditer(line))
    return max(found) if found else 0.0


def max_sum_in_lines(text: str, *keys: str) -> float:
    found: list[float] = []
    patterns = {
        key: re.compile(rf"\b{re.escape(key)}=([^\s]+)")
        for key in keys
    }
    for line in text.splitlines():
        total = 0.0
        seen = False
        for pattern in patterns.values():
            match = pattern.search(line)
            if match:
                seen = True
                total += number(match.group(1).rstrip(",;"))
        if seen:
            found.append(total)
    return max(found) if found else 0.0


def last_str(vals: dict[str, list[str]], key: str, default: str) -> str:
    if key not in vals or not vals[key]:
        return default
    return vals[key][-1]


def mib(bytes_value: float) -> float:
    return bytes_value / 1024.0 / 1024.0


def fmt(value: object) -> object:
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def read_exit(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8", errors="replace").strip() or "0")
    except (OSError, ValueError):
        return 127


def discover_cases(log_dir: Path) -> list[str]:
    cases = sorted(path.stem for path in log_dir.glob("*.err"))
    return cases if cases else DEFAULT_CASES


def has_real_abnormal(text: str) -> int:
    for line in text.splitlines():
        low = line.lower()
        if "sigsegv" in low or "segmentation" in low or "assert" in low or "abort" in low:
            return 1
        if re.search(r"\bnan\b", low):
            return 1
        if re.search(r"backend_failures=[1-9][0-9]*", low):
            return 1
        if re.search(r"active_visible_violation=[1-9][0-9]*", low):
            return 1
        if re.search(r"visible_violation_rows=[1-9][0-9]*", low):
            return 1
        if re.search(r"\b(error|failed)\b", low):
            scrubbed = re.sub(r"[a-z_]*fail(?:ure|ures)?=0\b", "", low)
            scrubbed = re.sub(r"[a-z_]*errors?=0\b", "", scrubbed)
            scrubbed = re.sub(r"[a-z_]*violation[a-z_]*=0\b", "", scrubbed)
            if re.search(r"\b(error|failed)\b", scrubbed):
                return 1
    return 0


def parse_case(log_dir: Path, case: str) -> dict[str, object]:
    out_text = (log_dir / f"{case}.out").read_text(encoding="utf-8", errors="replace") if (log_dir / f"{case}.out").exists() else ""
    err_text = (log_dir / f"{case}.err").read_text(encoding="utf-8", errors="replace") if (log_dir / f"{case}.err").exists() else ""
    text = out_text + "\n" + err_text
    vals = parse_values(text)
    defaults = CASE_DEFAULTS.get(case, GENERIC_DEFAULTS)

    resident_samples = [number(v) for v in vals.get("kv_mincore_resident_bytes", [])]
    whole_kv_range_drop_mib = 0.0
    if resident_samples:
        whole_kv_range_drop_mib = mib(max(resident_samples) - min(resident_samples))

    prefetch_blocks = (
        last_num(vals, "prefetch_during_active_blocks")
        + last_num(vals, "prefetch_blocks")
    )
    swapped_blocks = max_num(vals, "kv_mincore_swapped_block_count")
    if swapped_blocks == 0:
        swapped_blocks = max_num_in_prefixed_lines(text, "KV_PAGED_TRACE", "swapped_blocks")

    restore_blocks = max_num(vals, "paged_swap_in_calls")
    if restore_blocks == 0:
        restore_blocks = max_sum_in_lines(
            text,
            "paged_swap_read_swap_in_calls",
            "paged_swap_write_swap_in_calls",
        )

    seq1_active_ms = last_num(vals, "seq1_active_ms")
    seq1_active_tokens = last_num(vals, "seq1_active_tokens")
    active_tps = 1000.0 * seq1_active_tokens / seq1_active_ms if seq1_active_ms > 0 else 0.0

    active_visible_violation = max_num(
        vals,
        "active_visible_violation",
        "paged_swapped_active_visible_violation",
    )
    visible_violation_rows = max_num(
        vals,
        "visible_violation_rows",
        "paged_swapped_active_visible_violation_rows",
    )
    active_violation_rows = max_num(
        vals,
        "active_violation_rows",
        "paged_swapped_active_violation_rows",
    )

    prefetch_enabled = max_num(
        vals,
        "prefetch_enabled",
        "prefetch_during_active_enabled",
        "prefetch_auto_enabled",
    )
    if prefetch_enabled == 0 and defaults["prefetch_enabled"]:
        prefetch_enabled = defaults["prefetch_enabled"]

    row: dict[str, object] = {
        "case": case,
        "policy": POLICY.get(case, case),
        "exit_code": read_exit(log_dir / f"{case}.exit"),
        "num_idle": int(last_num(vals, "num_idle_seqs", defaults["num_idle"])),
        "pressure_mode": last_str(vals, "pressure_mode", str(defaults["pressure_mode"])),
        "prefetch_enabled": int(prefetch_enabled),
        "prefetch_every_tokens": int(last_num(vals, "prefetch_auto_every_tokens", defaults["prefetch_every_tokens"])),
        "prefetch_blocks_per_step": int(last_num(vals, "prefetch_auto_blocks_per_step", defaults["prefetch_blocks_per_step"])),
        "prefetch_safety_tokens": int(last_num(vals, "prefetch_auto_safety_tokens", defaults["prefetch_safety_tokens"])),
        "idle_owned_blocks": int(max_num(vals, "paged_cov_idle_owned_blocks", "idle_owned_blocks")),
        "swapped_blocks": int(swapped_blocks),
        "madvise_mib": mib(max_num(vals, "paged_swap_madvise_bytes")),
        "process_rss_drop_mib": max_num(vals, "paged_swap_rss_total_drop_kb") / 1024.0,
        "kv_nonresident_mib": mib(max_num(vals, "kv_mincore_swapped_nonresident_bytes")),
        "kv_resident_mib": mib(max_num(vals, "kv_mincore_resident_bytes")),
        "whole_kv_range_drop_mib": whole_kv_range_drop_mib,
        "net_rss_drop_before_resume_mib": 0.0,
        "prefetch_blocks": int(prefetch_blocks),
        "restore_blocks": int(restore_blocks),
        "fallback_blocks": int(last_num(vals, "prefetch_remaining_blocks_before_resume", max_num(vals, "prefetch_auto_fallback_blocks", "resume_pending_fallback_blocks"))),
        "target_restore_blocks": int(last_num(vals, "target_restore_blocks")),
        "prefetch_auto_window_ok": int(last_num(vals, "prefetch_auto_window_ok")),
        "prefetch_auto_need_steps": int(last_num(vals, "prefetch_auto_need_steps")),
        "prefetch_auto_start_token": int(last_num(vals, "prefetch_auto_start_token")),
        "prefetch_auto_effective_start_token": int(last_num(vals, "effective_start_token")),
        "prefetch_remaining_blocks_before_resume": int(last_num(vals, "prefetch_remaining_blocks_before_resume")),
        "resume_first_ms": last_num(vals, "seq0_resume_first_token_ms"),
        "resume_total_ms": last_num(vals, "seq0_resume_total_ms"),
        "seq1_active_ms": seq1_active_ms,
        "seq1_active_tokens": int(seq1_active_tokens),
        "active_tps": active_tps,
        "total_wall_ms": last_num(vals, "total_wall_ms"),
        "seq1_decoded": int(last_num(vals, "seq1_decoded_tokens")),
        "resume_decoded": int(last_num(vals, "seq0_resume_decoded_tokens")),
        "active_visible_violation": int(active_visible_violation),
        "visible_violation_rows": int(visible_violation_rows),
        "active_violation_rows": int(active_violation_rows),
        "real_abnormal": has_real_abnormal(text),
    }
    return row


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row[field]) for field in fields})


def print_table(fields: list[str], rows: list[dict[str, object]]) -> None:
    print("\t".join(fields))
    for row in rows:
        print("\t".join(str(fmt(row[field])) for field in fields))


def main() -> int:
    log_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/root/oscomp/kv_logs/stage8a_policy_matrix_once")
    rows = [parse_case(log_dir, case) for case in discover_cases(log_dir)]
    summary_path = log_dir / "summary.tsv"
    write_tsv(summary_path, FIELDS, rows)

    print_table(FIELDS, rows)
    print()
    print("compact decision view")
    compact_rows = []
    for row in rows:
        compact = {
            "case": row["case"],
            "policy": row["policy"],
            "swapped": row["swapped_blocks"],
            "kv_nonresident": row["kv_nonresident_mib"],
            "rss_drop": row["process_rss_drop_mib"],
            "prefetch": row["prefetch_blocks"],
            "fallback": row["fallback_blocks"],
            "resume_first": row["resume_first_ms"],
            "active_tps": row["active_tps"],
            "decoded": f"{row['seq1_decoded']}/{row['resume_decoded']}",
            "abnormal": row["real_abnormal"],
        }
        compact_rows.append(compact)
    print_table(COMPACT_FIELDS, compact_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
