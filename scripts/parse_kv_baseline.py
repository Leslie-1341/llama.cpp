#!/usr/bin/env python3
"""parse_kv_baseline.py — 解析 KV baseline 实验日志，汇总成 CSV。

读取 results/kv_baseline/ 下由 run_kv_baseline.sh 产生的:
  - ctx<CTX>_run<R>.log   —— llama-completion 原始输出(含 common_perf_print 行)
  - ctx<CTX>_run<R>.time  —— /usr/bin/time -v 输出(含 Maximum resident set size)

提取每组实验的:
  - context length
  - run id
  - Maximum resident set size (KB)
  - prompt eval speed (tokens/s) 及 prompt eval time (ms)
  - decode (eval) speed (tokens/s) 及 decode eval time (ms)
  - total time (ms)

输出:
  - results/kv_baseline/baseline_summary.csv
  - 控制台简要汇总

本脚本【不触发推理、不修改源码】,只解析已有日志。
"""

import csv
import os
import re
import sys
from collections import defaultdict

# ----------------------------------------------------------------------------
# 结果目录:默认 <repo>/results/kv_baseline,可用 RESULTS_DIR 环境变量覆盖。
# ----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR", os.path.join(REPO_ROOT, "results", "kv_baseline")
)
OUT_CSV = os.path.join(RESULTS_DIR, "baseline_summary.csv")

# 文件名形如 ctx512_run3.log / ctx512_run3.time
NAME_RE = re.compile(r"^ctx(?P<ctx>\d+)_run(?P<run>\d+)\.(?P<ext>log|time)$")

# common_perf_print 输出行(stderr),示例:
#   common_perf_print: prompt eval time =     123.45 ms /    42 tokens (    2.94 ms per token,   340.21 tokens per second)
#   common_perf_print:        eval time =    5678.90 ms /   127 runs   (   44.72 ms per token,    22.36 tokens per second)
#   common_perf_print:       total time =    5802.35 ms /   169 tokens
PROMPT_EVAL_RE = re.compile(
    r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([\d.]+)\s*tokens per second"
)
DECODE_EVAL_RE = re.compile(
    r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second"
)
TOTAL_TIME_RE = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")

# /usr/bin/time -v 行:
#   Maximum resident set size (kbytes): 1234567
MAX_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def parse_log(path):
    """从 llama-completion 原始日志提取 perf 指标。返回 dict(缺失项为 None)。"""
    out = {
        "prompt_eval_ms": None,
        "prompt_eval_tokens": None,
        "prompt_eval_tps": None,
        "decode_eval_ms": None,
        "decode_eval_runs": None,
        "decode_eval_tps": None,
        "total_time_ms": None,
    }
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = PROMPT_EVAL_RE.search(line)
            if m:
                out["prompt_eval_ms"] = float(m.group(1))
                out["prompt_eval_tokens"] = int(m.group(2))
                out["prompt_eval_tps"] = float(m.group(3))
                continue
            # decode 行必须排除 "prompt eval time",故要求行内不含 "prompt"。
            if "prompt" not in line:
                m = DECODE_EVAL_RE.search(line)
                if m:
                    out["decode_eval_ms"] = float(m.group(1))
                    out["decode_eval_runs"] = int(m.group(2))
                    out["decode_eval_tps"] = float(m.group(3))
                    continue
            m = TOTAL_TIME_RE.search(line)
            if m:
                out["total_time_ms"] = float(m.group(1))
    return out


def parse_time(path):
    """从 /usr/bin/time -v 输出提取 Maximum resident set size(KB)。"""
    max_rss_kb = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = MAX_RSS_RE.search(line)
            if m:
                max_rss_kb = int(m.group(1))
                break
    return max_rss_kb


def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"ERROR: 结果目录不存在: {RESULTS_DIR}", file=sys.stderr)
        print("请先运行 scripts/run_kv_baseline.sh 生成日志。", file=sys.stderr)
        sys.exit(1)

    # 按 (ctx, run) 归并 .log 与 .time。
    runs = defaultdict(dict)  # key=(ctx, run) -> {"log": path, "time": path}
    for name in sorted(os.listdir(RESULTS_DIR)):
        m = NAME_RE.match(name)
        if not m:
            continue
        key = (int(m.group("ctx")), int(m.group("run")))
        runs[key][m.group("ext")] = os.path.join(RESULTS_DIR, name)

    if not runs:
        print(f"ERROR: 在 {RESULTS_DIR} 未找到 ctx<N>_run<R>.log/.time 文件。", file=sys.stderr)
        print("请先运行 scripts/run_kv_baseline.sh。", file=sys.stderr)
        sys.exit(1)

    fieldnames = [
        "ctx_length",
        "run_id",
        "max_rss_kb",
        "max_rss_mb",
        "prompt_eval_ms",
        "prompt_eval_tokens",
        "prompt_eval_tps",
        "decode_eval_ms",
        "decode_eval_runs",
        "decode_eval_tps",
        "total_time_ms",
    ]

    rows = []
    for (ctx, run) in sorted(runs.keys()):
        entry = runs[(ctx, run)]
        row = {"ctx_length": ctx, "run_id": run}
        # 默认全部 None。
        for k in fieldnames[2:]:
            row[k] = None

        if "log" in entry:
            perf = parse_log(entry["log"])
            row.update(perf)
        else:
            print(f"WARN: ctx={ctx} run={run} 缺少 .log 文件", file=sys.stderr)

        if "time" in entry:
            rss_kb = parse_time(entry["time"])
            row["max_rss_kb"] = rss_kb
            row["max_rss_mb"] = round(rss_kb / 1024.0, 1) if rss_kb is not None else None
        else:
            print(f"WARN: ctx={ctx} run={run} 缺少 .time 文件", file=sys.stderr)

        rows.append(row)

    # 写 CSV。
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"已写出 CSV: {OUT_CSV}  (共 {len(rows)} 行)")
    print_summary(rows)


def _fmt(v, nd=2):
    if v is None:
        return "    n/a"
    if isinstance(v, float):
        return f"{v:8.{nd}f}"
    return f"{v:8}"


def print_summary(rows):
    """控制台简要汇总:逐行 + 按 ctx 聚合均值。"""
    print()
    print("=== 逐次实验 ===")
    header = (
        f"{'ctx':>6} {'run':>4} {'RSS(MB)':>9} "
        f"{'p_eval(ms)':>11} {'p_tps':>8} {'d_eval(ms)':>11} {'d_tps':>8} {'total(ms)':>11}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['ctx_length']:>6} {r['run_id']:>4} "
            f"{_fmt(r['max_rss_mb'], 1)} "
            f"{_fmt(r['prompt_eval_ms'])} {_fmt(r['prompt_eval_tps'])} "
            f"{_fmt(r['decode_eval_ms'])} {_fmt(r['decode_eval_tps'])} "
            f"{_fmt(r['total_time_ms'])}"
        )

    # 按 ctx 聚合均值。
    by_ctx = defaultdict(list)
    for r in rows:
        by_ctx[r["ctx_length"]].append(r)

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    print()
    print("=== 按 context length 聚合(均值) ===")
    print(header)
    print("-" * len(header))
    for ctx in sorted(by_ctx.keys()):
        group = by_ctx[ctx]
        print(
            f"{ctx:>6} {'avg':>4} "
            f"{_fmt(mean([r['max_rss_mb'] for r in group]), 1)} "
            f"{_fmt(mean([r['prompt_eval_ms'] for r in group]))} "
            f"{_fmt(mean([r['prompt_eval_tps'] for r in group]))} "
            f"{_fmt(mean([r['decode_eval_ms'] for r in group]))} "
            f"{_fmt(mean([r['decode_eval_tps'] for r in group]))} "
            f"{_fmt(mean([r['total_time_ms'] for r in group]))}"
        )


if __name__ == "__main__":
    main()
