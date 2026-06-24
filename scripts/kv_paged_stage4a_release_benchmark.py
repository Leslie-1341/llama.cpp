#!/usr/bin/env python3
"""Run the Stage 4A block-aware release benchmark matrix.

This runner only orchestrates benchmark commands and writes summary.csv. It
does not build llama.cpp, change C++ code, or interpret the benchmark result.
"""

import argparse
import csv
import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
DEFAULT_BINARY = "build/bin/llama-completion"
DEFAULT_OUT_DIR = "/root/oscomp/kv_logs/paged_stage4a_release_benchmark/"
DEFAULT_PROMPT = "Hello, how are you?"
DEFAULT_CTX_LIST = "512,2048,4096"
DEFAULT_N_LIST = "16,128,512"
SEED = "42"

FIXED_ARGS = [
    "-t", "12",
    "-ngl", "0",
    "-fa", "on",
    "-no-cnv",
    "-ctk", "f32",
    "-ctv", "f32",
    "-nkvo",
    "-v",
]

MODES = [
    ("baseline", {}),
    ("paged_no_release", {
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_SHIFT": "1",
    }),
    ("paged_release", {
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_SHIFT": "1",
        "LLAMA_KV_PAGED_RELEASE": "1",
    }),
]

PAGED_ENV_KEYS = [
    "LLAMA_KV_PAGED",
    "LLAMA_KV_PAGED_SHIFT",
    "LLAMA_KV_PAGED_RELEASE",
    "LLAMA_KV_PAGED_SWAP",
    "LLAMA_KV_PAGED_TRACE",
    "LLAMA_KV_PAGED_IDLE_TRACE",
    "LLAMA_KV_LAZY_TAIL",
    "LLAMA_KV_LAZY_CLEAR",
    "LLAMA_LOW_MEM_WARMUP",
    "LLAMA_KV_EXACT_SWAP",
    "LLAMA_KV_APPROX",
    "LLAMA_KV_APPROX_WINDOW",
    "LLAMA_KV_APPROX_DEBUG",
]

PAGED_FIELDS = [
    "paged_blocks_released",
    "paged_blocks_released_unused",
    "paged_block_release_bytes",
    "paged_block_release_rss_drop_last_kb",
    "paged_block_release_rss_drop_max_kb",
    "paged_block_release_rss_before_last_kb",
    "paged_block_release_rss_after_last_kb",
    "paged_release_violation",
    "paged_active_release_violation",
    "paged_padded_release_violation",
    "paged_block_release_fail",
    "paged_row_idx_fail",
    "paged_logical_to_physical_fail",
    "paged_swap_backend_failures",
]

PERF_FIELDS = [
    "prompt_eval_ms",
    "eval_ms",
    "ms_per_token",
    "tokens_per_second",
    "total_ms",
    "graphs_reused",
]

CSV_FIELDS = [
    "mode",
    "ctx",
    "n_tokens",
    "status",
    "exit_code",
    "out_path",
    "log_path",
    "time_path",
    "sha256",
    "sha256_equal_to_baseline",
    *PERF_FIELDS,
    "peak_rss_kb",
    *PAGED_FIELDS,
    "advised_mib",
    "rss_drop_max_mib",
    "rss_drop_over_advised_ratio",
]

STATS_PREFIX = "KV paged metadata stats:"


def parse_int_list(value):
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer list item: {item!r}") from exc
    if not out:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    return out


def make_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Stage 4A block-aware release benchmark matrix and write summary.csv.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--binary", default=DEFAULT_BINARY)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--ctx-list", type=parse_int_list, default=parse_int_list(DEFAULT_CTX_LIST))
    parser.add_argument("--n-list", type=parse_int_list, default=parse_int_list(DEFAULT_N_LIST))
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def case_name(mode, ctx, n_tokens):
    return f"{mode}_ctx{ctx}_n{n_tokens}"


def build_command(args, ctx, n_tokens):
    return [
        "/usr/bin/time",
        "-v",
        "-o",
        None,
        args.binary,
        "-m", args.model,
        "-p", args.prompt,
        "-s", SEED,
        "-c", str(ctx),
        "-n", str(n_tokens),
        *FIXED_ARGS,
    ]


def clean_env(mode_env):
    env = os.environ.copy()
    for key in PAGED_ENV_KEYS:
        env.pop(key, None)
    env.update(mode_env)
    return env


def format_command(env_delta, cmd, out_path, log_path, time_path):
    env_parts = ["env", *[f"-u {key}" for key in PAGED_ENV_KEYS]]
    env_parts.extend(f"{key}={shlex.quote(value)}" for key, value in sorted(env_delta.items()))
    printable_cmd = [time_path if part is None else part for part in cmd]
    cmd_parts = [shlex.quote(str(part)) for part in printable_cmd]
    redirects = [
        ">", shlex.quote(str(out_path)),
        "2>", shlex.quote(str(log_path)),
    ]
    return " ".join([*env_parts, *cmd_parts, *redirects])


def command_with_time_path(cmd, time_path):
    return [str(time_path) if part is None else part for part in cmd]


def sha256_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


def parse_time_file(path):
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    return match.group(1) if match else ""


def parse_perf(log_path):
    values = {key: "" for key in PERF_FIELDS}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return values

    prompt_matches = re.findall(
        r"common_perf_print:\s+prompt eval time =\s*([0-9.]+) ms /"
        r"\s*\d+ tokens \(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\)",
        text,
    )
    if prompt_matches:
        values["prompt_eval_ms"] = prompt_matches[-1][0]

    eval_matches = re.findall(
        r"common_perf_print:\s+eval time =\s*([0-9.]+) ms /"
        r"\s*\d+ runs\s+\(\s*([0-9.]+) ms per token,\s*([0-9.]+) tokens per second\)",
        text,
    )
    if eval_matches:
        values["eval_ms"] = eval_matches[-1][0]
        values["ms_per_token"] = eval_matches[-1][1]
        values["tokens_per_second"] = eval_matches[-1][2]

    total_matches = re.findall(r"common_perf_print:\s+total time =\s*([0-9.]+) ms /", text)
    if total_matches:
        values["total_ms"] = total_matches[-1]

    graphs_matches = re.findall(r"common_perf_print:\s+graphs reused =\s*(\d+)", text)
    if graphs_matches:
        values["graphs_reused"] = graphs_matches[-1]

    return values


def parse_paged_stats(log_path):
    values = {key: "" for key in PAGED_FIELDS}
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return values

    stats_lines = [line for line in lines if STATS_PREFIX in line]
    if not stats_lines:
        return values

    last = stats_lines[-1]
    pairs = dict(re.findall(r"([A-Za-z0-9_]+)=(-?\d+)", last))
    aliases = {
        "paged_row_idx_fail": "row_idx_fail",
        "paged_logical_to_physical_fail": "logical_to_physical_fail",
    }
    for key in PAGED_FIELDS:
        source_key = aliases.get(key, key)
        if source_key in pairs:
            values[key] = pairs[source_key]
    return values


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def add_derived_fields(row):
    advised = to_int(row.get("paged_block_release_bytes"))
    drop_max = to_int(row.get("paged_block_release_rss_drop_max_kb"))

    if advised is not None:
        row["advised_mib"] = f"{advised / (1024 * 1024):.2f}"
    if drop_max is not None:
        row["rss_drop_max_mib"] = f"{drop_max / 1024:.2f}"
    if advised and drop_max is not None:
        row["rss_drop_over_advised_ratio"] = f"{drop_max * 1024 / advised:.6f}"


def skipped_row(mode, ctx, n_tokens, out_path, log_path, time_path):
    row = {key: "" for key in CSV_FIELDS}
    row.update({
        "mode": mode,
        "ctx": ctx,
        "n_tokens": n_tokens,
        "status": "skipped",
        "exit_code": "",
        "out_path": str(out_path),
        "log_path": str(log_path),
        "time_path": str(time_path),
    })
    return row


def run_case(args, mode, mode_env, ctx, n_tokens, out_path, log_path, time_path):
    cmd = build_command(args, ctx, n_tokens)
    if args.dry_run:
        print(format_command(mode_env, cmd, out_path, log_path, time_path))
        row = {key: "" for key in CSV_FIELDS}
        row.update({
            "mode": mode,
            "ctx": ctx,
            "n_tokens": n_tokens,
            "status": "dry_run",
            "exit_code": "",
            "out_path": str(out_path),
            "log_path": str(log_path),
            "time_path": str(time_path),
        })
        return row

    cmd = command_with_time_path(cmd, time_path)
    try:
        with open(out_path, "wb") as out_f, open(log_path, "wb") as err_f:
            proc = subprocess.run(
                cmd,
                stdout=out_f,
                stderr=err_f,
                env=clean_env(mode_env),
                check=False,
            )
        exit_code = proc.returncode
    except OSError as exc:
        log_path.write_text(f"failed to execute command: {exc}\n", errors="replace")
        time_path.write_text("", errors="replace")
        exit_code = 127

    row = {key: "" for key in CSV_FIELDS}
    row.update({
        "mode": mode,
        "ctx": ctx,
        "n_tokens": n_tokens,
        "status": "ok" if exit_code == 0 else "failed",
        "exit_code": str(exit_code),
        "out_path": str(out_path),
        "log_path": str(log_path),
        "time_path": str(time_path),
        "sha256": sha256_file(out_path),
        "peak_rss_kb": parse_time_file(time_path),
    })
    row.update(parse_perf(log_path))
    row.update(parse_paged_stats(log_path))
    add_derived_fields(row)
    return row


def write_summary(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fill_sha256_comparisons(rows):
    baselines = {}
    for row in rows:
        if row["mode"] == "baseline" and row["status"] == "ok" and row["sha256"]:
            baselines[(row["ctx"], row["n_tokens"])] = row["sha256"]

    for row in rows:
        if row["status"] not in ("ok", "failed") or not row["sha256"]:
            row["sha256_equal_to_baseline"] = ""
            continue
        baseline = baselines.get((row["ctx"], row["n_tokens"]))
        if row["mode"] == "baseline":
            row["sha256_equal_to_baseline"] = "yes" if baseline else ""
        elif baseline:
            row["sha256_equal_to_baseline"] = "yes" if row["sha256"] == baseline else "no"
        else:
            row["sha256_equal_to_baseline"] = ""


def main():
    args = make_arg_parser().parse_args()
    out_dir = Path(args.out_dir)

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ctx in args.ctx_list:
        for n_tokens in args.n_list:
            invalid = n_tokens >= ctx
            for mode, mode_env in MODES:
                name = case_name(mode, ctx, n_tokens)
                out_path = out_dir / f"{name}.out"
                log_path = out_dir / f"{name}.log"
                time_path = out_dir / f"{name}.time"

                if invalid and not args.include_invalid:
                    rows.append(skipped_row(mode, ctx, n_tokens, out_path, log_path, time_path))
                    continue

                rows.append(run_case(args, mode, mode_env, ctx, n_tokens, out_path, log_path, time_path))

    fill_sha256_comparisons(rows)

    summary_path = out_dir / "summary.csv"
    if not args.dry_run:
        write_summary(summary_path, rows)
        print(f"wrote {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
