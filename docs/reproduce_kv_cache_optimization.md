# KV Cache 内存优化复现指南

本文档用于复现本项目在 `llama.cpp` 上实现的 KV cache 运行时内存优化结果。重点包括：

```text
1. 构建相关 example driver；
2. 运行最小 smoke；
3. 生成 ShareGPT-backed trace；
4. 复现 Stage 12-C final V5 配置；
5. 复现 ctx4096 / ctx8192 RSS 与性能结果；
6. 复现 mincore KV resident page 诊断；
7. 判断 correctness / safety 是否通过。
```

本文档不依赖本机绝对路径。所有路径均用变量表示。文中给出的 `/root/oscomp/...` 仅作为本项目实验环境示例，不是源码依赖。

---

## 1. 复现范围

### 1.1 本文档复现什么

本文档复现的是：

```text
ShareGPT-backed trace replay workload 下的 KV cache reclaim 效果；
fast-maintenance V5 最终配置；
ctx4096 / ctx8192 下的 process RSS drop；
mincore 诊断下的 KV resident page drop；
trace replay 的 correctness / safety 检查。
```

### 1.2 本文档不复现什么

本文档不覆盖：

```text
1. 早期所有 stage 的完整矩阵；
2. llama-server HTTP benchmark；
3. GPU backend；
4. 真实线上生产 trace；
5. cgroup / PSI 真实 memory pressure 策略；
6. 完整异步 prefetch thread。
```

历史阶段材料位于：

```text
docs/archive/kv-stage-history/
```

最终详细结果位于：

```text
docs/kv_trace_replay_stage12c_real_sharegpt_results.md
```

---

## 2. 环境要求

### 2.1 系统环境

推荐环境：

```text
OS: Ubuntu 22.04 或兼容 Linux
CPU: x86_64
Compiler: 支持 C++17
Build system: CMake
Python: Python 3.8+
```

需要 Linux 支持：

```text
madvise(MADV_DONTNEED)
mincore(2)    # 仅诊断时需要
/proc/self/status 或等价 RSS 读取路径
```

### 2.2 模型要求

本项目实验使用：

```text
Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

复现时需要自行准备模型文件。模型文件不应提交到仓库。

设置变量：

```bash
export MODEL=/path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### 2.3 仓库路径

设置仓库路径：

```bash
export REPO=/path/to/llama.cpp
cd "$REPO"
```

本项目实验环境示例：

```bash
export REPO=/root/oscomp/llama.cpp
export MODEL=/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### 2.4 日志目录

设置日志目录：

```bash
export LOG_ROOT=/path/to/kv_logs
mkdir -p "$LOG_ROOT"
```

实验环境示例：

```bash
export LOG_ROOT=/root/oscomp/kv_logs
```

---

## 3. 构建

### 3.1 配置 CMake

```bash
cd "$REPO"

cmake -S . -B build
```

### 3.2 构建最终 driver

```bash
cmake --build build -j --target llama-kv-trace-replay
```

### 3.3 构建相关辅助 driver

可选：

```bash
cmake --build build -j --target \
  llama-kv-trace-replay \
  llama-kv-semi-real-multisession \
  llama-kv-idle-swap-resume
```

### 3.4 构建产物检查

```bash
ls -lh build/bin/llama-kv-trace-replay
```

预期：

```text
build/bin/llama-kv-trace-replay 存在
```

---

## 4. 最小 smoke

最小 smoke 用于确认 trace replay driver 可运行，不用于复现最终性能数字。

### 4.1 准备 smoke corpus

`examples/kv-trace-replay/traces/smoke_4s2t.tsv` 使用 `corpus:` prompt source，因此需要设置 `LLAMA_KV_TRACE_CORPUS_FILE`。

准备一个临时 corpus：

```bash
python3 - <<'PY'
from pathlib import Path

p = Path("/tmp/kv_trace_smoke_corpus.txt")
unit = "This is a temporary corpus for KV trace replay smoke testing. "
p.write_text(unit * 20000)
print(p, p.stat().st_size)
PY

export CORPUS=/tmp/kv_trace_smoke_corpus.txt
```

也可以使用 WikiText-2：

```bash
export CORPUS="$REPO/wikitext-2-raw/wiki.test.raw"
```

### 4.2 默认路径 smoke

不开 paged / swap：

```bash
cd "$REPO"

mkdir -p "$LOG_ROOT/smoke"

LLAMA_KV_TRACE_FILE=examples/kv-trace-replay/traces/smoke_4s2t.tsv \
LLAMA_KV_TRACE_CORPUS_FILE="$CORPUS" \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 512 \
  --batch-size 128 \
  --ubatch-size 64 \
  --parallel 4 \
  --seed 1 \
  --temp 0 \
  > "$LOG_ROOT/smoke/default.out" \
  2> "$LOG_ROOT/smoke/default.err"

echo "exit=$?"
grep -H 'KV_TRACE_SUMMARY\|KV_TRACE_PERF' "$LOG_ROOT/smoke/default.out" "$LOG_ROOT/smoke/default.err" || true
```

### 4.3 Paged + idle swap smoke

```bash
cd "$REPO"

LLAMA_KV_PAGED=1 \
LLAMA_KV_PAGED_INGRAPH=1 \
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0 \
LLAMA_KV_TRACE_FILE=examples/kv-trace-replay/traces/smoke_4s2t.tsv \
LLAMA_KV_TRACE_CORPUS_FILE="$CORPUS" \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 512 \
  --batch-size 128 \
  --ubatch-size 64 \
  --parallel 4 \
  --seed 1 \
  --temp 0 \
  > "$LOG_ROOT/smoke/paged.out" \
  2> "$LOG_ROOT/smoke/paged.err"

echo "exit=$?"
grep -H 'KV_TRACE_SUMMARY\|KV_TRACE_PERF' "$LOG_ROOT/smoke/paged.out" "$LOG_ROOT/smoke/paged.err" || true
```

### 4.4 Smoke 异常检查

```bash
grep -HniE 'Segmentation fault|Assertion|assertion failed|GGML_ASSERT|abort|Aborted|KV_TRACE_ERROR|KV_PAGED_ERROR|active_visible_violation=1|paged_swapped_active_visible_violation=1|write_to_swapped=1|paged_write_to_swapped_block=1|\b[Nn]a[Nn]\b|out of memory|OOM' \
  "$LOG_ROOT/smoke/default.out" "$LOG_ROOT/smoke/default.err" \
  "$LOG_ROOT/smoke/paged.out" "$LOG_ROOT/smoke/paged.err" || true
```

通过标准：

```text
1. default / paged 均 exit=0；
2. 每个 session 有 KV_TRACE_SUMMARY；
3. 有 KV_TRACE_PERF；
4. abnormal grep 无真实异常。
```

---

## 5. 生成 ShareGPT-backed trace

最终实验使用 ShareGPT-backed trace replay。ShareGPT 数据文件不提交到仓库，需要自行准备。

### 5.1 输入数据

支持 JSON / JSONL，推荐先准备为严格 JSON 或可被脚本解析的 JSONL：

```bash
export SHAREGPT=/path/to/sharegpt.json
```

### 5.2 生成 trace

```bash
cd "$REPO"

export TRACE_DIR=/path/to/trace_out
mkdir -p "$TRACE_DIR"

python3 scripts/kv_trace_from_sharegpt.py \
  --input "$SHAREGPT" \
  --output-dir "$TRACE_DIR" \
  --num-sessions 8 \
  --max-turns-per-session 3 \
  --start-gap-ms 0 \
  --turn-gap-ms 400 \
  --jitter-ms 50 \
  --idle-ms-low 1200 \
  --idle-ms-high 2400 \
  --min-decode-tokens 16 \
  --max-decode-tokens 96 \
  --chars-per-token 4 \
  --max-user-chars 2048 \
  --seed 1 \
  --overwrite
```

生成结果：

```text
$TRACE_DIR/trace.tsv
$TRACE_DIR/prompts/
```

设置 trace 变量：

```bash
export TRACE="$TRACE_DIR/trace.tsv"
```

### 5.3 检查 trace

```bash
head -20 "$TRACE"
find "$TRACE_DIR/prompts" -type f | head
wc -l "$TRACE"
```

trace TSV 字段一般为：

```text
session_id
turn_id
arrival_ms
prompt_source
target_decode_tokens
idle_ms
```

`file:` prompt source 使用相对路径，由 `llama-kv-trace-replay` 按 trace 文件所在目录解析。脚本会避免依赖本机绝对 prompt 路径。

---

## 6. 通用运行参数

最终实验使用以下通用参数：

```bash
COMMON_ARGS_4096=(
  -m "$MODEL"
  --ctx-size 4096
  --batch-size 512
  --ubatch-size 128
  --parallel 8
  --seed 1
  --temp 0
  --cache-type-k f32
  --cache-type-v f32
  --kv-unified
)

COMMON_ARGS_8192=(
  -m "$MODEL"
  --ctx-size 8192
  --batch-size 512
  --ubatch-size 128
  --parallel 8
  --seed 1
  --temp 0
  --cache-type-k f32
  --cache-type-v f32
  --kv-unified
)
```

---

## 7. Final V5 配置

最终推荐配置：

```bash
export LLAMA_KV_LAZY_TAIL=1
export LLAMA_KV_LAZY_CLEAR=1

export LLAMA_KV_PAGED=1
export LLAMA_KV_PAGED_INGRAPH=1
export LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
export LLAMA_KV_PAGED_SWAP=1
export LLAMA_KV_PAGED_IDLE_SWAP=1
export LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1

export LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
export LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2
export LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2
export LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
export LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12
export LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1

export LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
export LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8
export LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8
export LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16
```

为了避免污染不同 case，建议在脚本中使用 `env -i` 或用函数显式传 env。下面给出更易读的函数式写法。

---

## 8. 单次运行：T0 baseline 与 V5

### 8.1 T0 baseline

T0 不开启 paged / lazy / swap：

```bash
cd "$REPO"

mkdir -p "$LOG_ROOT/stage12_reproduce_ctx4096"

LLAMA_KV_TRACE_FILE="$TRACE" \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 128 \
  --parallel 8 \
  --seed 1 \
  --temp 0 \
  --cache-type-k f32 \
  --cache-type-v f32 \
  --kv-unified \
  > "$LOG_ROOT/stage12_reproduce_ctx4096/T0.out" \
  2> "$LOG_ROOT/stage12_reproduce_ctx4096/T0.err"

echo "exit=$?"
grep -H 'KV_TRACE_SUMMARY\|KV_TRACE_PERF' \
  "$LOG_ROOT/stage12_reproduce_ctx4096/T0.out" \
  "$LOG_ROOT/stage12_reproduce_ctx4096/T0.err" || true
```

### 8.2 V5 final

```bash
cd "$REPO"

LLAMA_KV_LAZY_TAIL=1 \
LLAMA_KV_LAZY_CLEAR=1 \
LLAMA_KV_PAGED=1 \
LLAMA_KV_PAGED_INGRAPH=1 \
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2 \
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2 \
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0 \
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12 \
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1 \
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0 \
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8 \
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8 \
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16 \
LLAMA_KV_TRACE_FILE="$TRACE" \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 128 \
  --parallel 8 \
  --seed 1 \
  --temp 0 \
  --cache-type-k f32 \
  --cache-type-v f32 \
  --kv-unified \
  > "$LOG_ROOT/stage12_reproduce_ctx4096/V5.out" \
  2> "$LOG_ROOT/stage12_reproduce_ctx4096/V5.err"

echo "exit=$?"
grep -H 'KV_TRACE_SUMMARY\|KV_TRACE_PERF\|paged_' \
  "$LOG_ROOT/stage12_reproduce_ctx4096/V5.out" \
  "$LOG_ROOT/stage12_reproduce_ctx4096/V5.err" | tail -120 || true
```

### 8.3 单次异常检查

```bash
grep -HniE 'Segmentation fault|Assertion|assertion failed|GGML_ASSERT|abort|Aborted|KV_TRACE_ERROR|KV_PAGED_ERROR|active_visible_violation=1|paged_swapped_active_visible_violation=1|write_to_swapped=1|paged_write_to_swapped_block=1|\b[Nn]a[Nn]\b|out of memory|OOM' \
  "$LOG_ROOT/stage12_reproduce_ctx4096/T0.out" \
  "$LOG_ROOT/stage12_reproduce_ctx4096/T0.err" \
  "$LOG_ROOT/stage12_reproduce_ctx4096/V5.out" \
  "$LOG_ROOT/stage12_reproduce_ctx4096/V5.err" || true
```

---

## 9. 3-run median 复现脚本

建议保存脚本，而不是直接粘贴长命令到终端。

### 9.1 保存脚本

保存为：

```text
/tmp/run_kv_stage12_reproduce_3run.sh
```

内容如下：

```bash
cat > /tmp/run_kv_stage12_reproduce_3run.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

: "${REPO:?set REPO=/path/to/llama.cpp}"
: "${MODEL:?set MODEL=/path/to/model.gguf}"
: "${TRACE:?set TRACE=/path/to/trace.tsv}"
: "${LOG_ROOT:?set LOG_ROOT=/path/to/logs}"
: "${CTX_SIZE:=4096}"

cd "$REPO"

OUT_DIR="$LOG_ROOT/stage12_reproduce_ctx${CTX_SIZE}_3run"
mkdir -p "$OUT_DIR"

common_args=(
  -m "$MODEL"
  --ctx-size "$CTX_SIZE"
  --batch-size 512
  --ubatch-size 128
  --parallel 8
  --seed 1
  --temp 0
  --cache-type-k f32
  --cache-type-v f32
  --kv-unified
)

run_t0() {
  local run="$1"
  echo "===== T0 run $run ====="
  set +e
  LLAMA_KV_TRACE_FILE="$TRACE" \
  ./build/bin/llama-kv-trace-replay "${common_args[@]}" \
    > "$OUT_DIR/T0_run${run}.out" \
    2> "$OUT_DIR/T0_run${run}.err"
  local status=$?
  set -e
  echo "$status" > "$OUT_DIR/T0_run${run}.exit"
}

run_v5() {
  local run="$1"
  echo "===== V5 run $run ====="
  set +e
  LLAMA_KV_LAZY_TAIL=1 \
  LLAMA_KV_LAZY_CLEAR=1 \
  LLAMA_KV_PAGED=1 \
  LLAMA_KV_PAGED_INGRAPH=1 \
  LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
  LLAMA_KV_PAGED_SWAP=1 \
  LLAMA_KV_PAGED_IDLE_SWAP=1 \
  LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
  LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
  LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2 \
  LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2 \
  LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0 \
  LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12 \
  LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1 \
  LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0 \
  LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8 \
  LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8 \
  LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16 \
  LLAMA_KV_TRACE_FILE="$TRACE" \
  ./build/bin/llama-kv-trace-replay "${common_args[@]}" \
    > "$OUT_DIR/V5_run${run}.out" \
    2> "$OUT_DIR/V5_run${run}.err"
  local status=$?
  set -e
  echo "$status" > "$OUT_DIR/V5_run${run}.exit"
}

for i in 1 2 3; do
  run_t0 "$i"
  run_v5 "$i"
done

echo "===== exits ====="
cat "$OUT_DIR"/*.exit

echo "===== summaries ====="
grep -H 'KV_TRACE_SUMMARY\|KV_TRACE_PERF' "$OUT_DIR"/*.out "$OUT_DIR"/*.err || true

echo "===== abnormal grep ====="
grep -HniE 'Segmentation fault|Assertion|assertion failed|GGML_ASSERT|abort|Aborted|KV_TRACE_ERROR|KV_PAGED_ERROR|active_visible_violation=1|paged_swapped_active_visible_violation=1|write_to_swapped=1|paged_write_to_swapped_block=1|\b[Nn]a[Nn]\b|out of memory|OOM' \
  "$OUT_DIR"/*.out "$OUT_DIR"/*.err || true

echo "out_dir=$OUT_DIR"
SH

chmod +x /tmp/run_kv_stage12_reproduce_3run.sh
```

### 9.2 运行 ctx4096 3-run

```bash
export CTX_SIZE=4096
/tmp/run_kv_stage12_reproduce_3run.sh
```

### 9.3 运行 ctx8192 3-run

```bash
export CTX_SIZE=8192
/tmp/run_kv_stage12_reproduce_3run.sh
```

---

## 10. 解析 3-run median

保存 parser：

```bash
cat > /tmp/parse_kv_trace_perf.py <<'PY'
#!/usr/bin/env python3
import re
import sys
import statistics
from pathlib import Path

def parse_kv_line(line):
    d = {}
    for item in line.strip().split():
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
    return d

def to_float(d, k, default=None):
    try:
        return float(d[k])
    except Exception:
        return default

def median(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return statistics.median(xs)

def read_case(out_dir, case):
    rows = []
    for p in sorted(Path(out_dir).glob(f"{case}_run*.out")) + sorted(Path(out_dir).glob(f"{case}_run*.err")):
        text = p.read_text(errors="ignore")
        for line in text.splitlines():
            if "KV_TRACE_PERF" in line:
                d = parse_kv_line(line)
                d["_file"] = str(p)
                rows.append(d)
    return rows

def summarize(out_dir, case):
    rows = read_case(out_dir, case)
    return {
        "case": case,
        "n": len(rows),
        "active_tps": median([to_float(r, "active_tps") for r in rows]),
        "active_decode_ms": median([to_float(r, "active_decode_ms") for r in rows]),
        "total_wall_ms": median([to_float(r, "total_wall_ms") for r in rows]),
        "rss_kb": median([to_float(r, "rss_kb") for r in rows]),
        "active_decode_tokens": median([to_float(r, "active_decode_tokens") for r in rows]),
    }

def fmt(x):
    if x is None:
        return "NA"
    return f"{x:.6f}"

def main():
    if len(sys.argv) != 2:
        print("usage: parse_kv_trace_perf.py OUT_DIR", file=sys.stderr)
        sys.exit(2)
    out_dir = sys.argv[1]
    t0 = summarize(out_dir, "T0")
    v5 = summarize(out_dir, "V5")

    print("case\tn\tactive_tps\tactive_decode_ms\ttotal_wall_ms\trss_kb\tactive_decode_tokens")
    for s in (t0, v5):
        print(
            f"{s['case']}\t{s['n']}\t{fmt(s['active_tps'])}\t{fmt(s['active_decode_ms'])}\t"
            f"{fmt(s['total_wall_ms'])}\t{fmt(s['rss_kb'])}\t{fmt(s['active_decode_tokens'])}"
        )

    if t0["rss_kb"] is not None and v5["rss_kb"] is not None:
        rss_drop_mib = (t0["rss_kb"] - v5["rss_kb"]) / 1024.0
        total_rss_drop_pct = (t0["rss_kb"] - v5["rss_kb"]) / t0["rss_kb"] * 100.0
        print()
        print(f"rss_drop_mib={rss_drop_mib:.3f}")
        print(f"total_rss_drop_pct={total_rss_drop_pct:.3f}")

    if t0["active_tps"] and v5["active_tps"]:
        tps_delta_pct = (v5["active_tps"] - t0["active_tps"]) / t0["active_tps"] * 100.0
        print(f"tps_delta_pct={tps_delta_pct:.3f}")

    if t0["active_decode_ms"] and v5["active_decode_ms"]:
        decode_delta_pct = (v5["active_decode_ms"] - t0["active_decode_ms"]) / t0["active_decode_ms"] * 100.0
        print(f"active_decode_ms_delta_pct={decode_delta_pct:.3f}")

    if t0["total_wall_ms"] and v5["total_wall_ms"]:
        wall_delta_pct = (v5["total_wall_ms"] - t0["total_wall_ms"]) / t0["total_wall_ms"] * 100.0
        print(f"total_wall_delta_pct={wall_delta_pct:.3f}")

if __name__ == "__main__":
    main()
PY

chmod +x /tmp/parse_kv_trace_perf.py
```

解析 ctx4096：

```bash
/tmp/parse_kv_trace_perf.py "$LOG_ROOT/stage12_reproduce_ctx4096_3run"
```

解析 ctx8192：

```bash
/tmp/parse_kv_trace_perf.py "$LOG_ROOT/stage12_reproduce_ctx8192_3run"
```

---

## 11. mincore KV resident 诊断

`LLAMA_KV_PAGED_MINCORE=1` 用于诊断 KV buffer resident page。该模式会增加采样开销，不作为性能基准，只用于证明 RSS drop 与 KV resident drop 对应。

### 11.1 为什么需要 M0 baseline-like case

纯 T0 baseline 不启用 paged path，因此不会输出 `kv_mincore_*` 字段。为了获得 baseline-like KV resident reference，需要运行：

```text
M0_no_madvise_mincore:
  paged enabled
  swap enabled
  idle swap enabled
  madvise disabled
  lazy disabled
  mincore enabled
```

该配置用于证明：未 madvise 时 KV buffer 基本 100% resident。

### 11.2 运行 M0_no_madvise_mincore

ctx4096 示例：

```bash
cd "$REPO"

mkdir -p "$LOG_ROOT/mincore_ctx4096"

LLAMA_KV_PAGED=1 \
LLAMA_KV_PAGED_INGRAPH=1 \
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP=1 \
LLAMA_KV_PAGED_MINCORE=1 \
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0 \
LLAMA_KV_TRACE_FILE="$TRACE" \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 128 \
  --parallel 8 \
  --seed 1 \
  --temp 0 \
  --cache-type-k f32 \
  --cache-type-v f32 \
  --kv-unified \
  > "$LOG_ROOT/mincore_ctx4096/M0.out" \
  2> "$LOG_ROOT/mincore_ctx4096/M0.err"
```

ctx8192 只需修改：

```bash
--ctx-size 8192
```

### 11.3 运行 V5_mincore

ctx4096 示例：

```bash
cd "$REPO"

LLAMA_KV_LAZY_TAIL=1 \
LLAMA_KV_LAZY_CLEAR=1 \
LLAMA_KV_PAGED=1 \
LLAMA_KV_PAGED_INGRAPH=1 \
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2 \
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2 \
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0 \
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12 \
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1 \
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0 \
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8 \
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8 \
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16 \
LLAMA_KV_PAGED_MINCORE=1 \
LLAMA_KV_TRACE_FILE="$TRACE" \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 128 \
  --parallel 8 \
  --seed 1 \
  --temp 0 \
  --cache-type-k f32 \
  --cache-type-v f32 \
  --kv-unified \
  > "$LOG_ROOT/mincore_ctx4096/V5.out" \
  2> "$LOG_ROOT/mincore_ctx4096/V5.err"
```

ctx8192 同样修改：

```bash
--ctx-size 8192
```

### 11.4 提取 mincore 字段

```bash
grep -H 'kv_mincore\|kv_total\|kv_resident\|swapped_nonresident\|KV_TRACE_PERF' \
  "$LOG_ROOT"/mincore_ctx*/*.out \
  "$LOG_ROOT"/mincore_ctx*/*.err || true
```

关注字段：

```text
kv_mincore_total_bytes
kv_mincore_resident_bytes
kv_mincore_k_total_bytes
kv_mincore_v_total_bytes
kv_mincore_k_resident_bytes
kv_mincore_v_resident_bytes
kv_mincore_swapped_total_bytes
kv_mincore_swapped_resident_bytes
kv_mincore_swapped_nonresident_bytes
```

判断：

```text
M0_no_madvise_mincore:
  kv_resident ≈ kv_total

V5_mincore:
  kv_resident 明显小于 kv_total

process RSS drop:
  应与 kv resident drop 数量级一致
```

---

## 12. 通过标准

### 12.1 Correctness 标准

每个正式 case 应满足：

```text
exit = 0
KV_TRACE_SUMMARY count = session count
all sessions finished
real_abnormal = 0
```

常用检查：

```bash
grep -H 'KV_TRACE_SUMMARY\|KV_TRACE_PERF' "$LOG_ROOT"/stage12_reproduce_*/*.out "$LOG_ROOT"/stage12_reproduce_*/*.err || true
```

### 12.2 Safety 标准

必须没有：

```text
active_visible_violation=1
paged_swapped_active_visible_violation=1
write_to_swapped=1
paged_write_to_swapped_block=1
backend_failures > 0
KV_TRACE_ERROR
KV_PAGED_ERROR
NaN
assert / abort / segfault
OOM
```

检查命令：

```bash
grep -HniE 'Segmentation fault|Assertion|assertion failed|GGML_ASSERT|abort|Aborted|KV_TRACE_ERROR|KV_PAGED_ERROR|active_visible_violation=1|paged_swapped_active_visible_violation=1|write_to_swapped=1|paged_write_to_swapped_block=1|\b[Nn]a[Nn]\b|out of memory|OOM|backend_failures=[1-9]' \
  "$LOG_ROOT"/stage12_reproduce_*/*.out \
  "$LOG_ROOT"/stage12_reproduce_*/*.err \
  "$LOG_ROOT"/mincore_ctx*/*.out \
  "$LOG_ROOT"/mincore_ctx*/*.err || true
```

### 12.3 性能结果判断

结果不要求逐字节等于提交文档中的数值，因为硬件、内核、编译参数、后台负载、trace 再生成方式都会影响时间和 RSS。但趋势应符合：

```text
ctx4096:
  V5 相比 T0 有明显 RSS drop；
  TPS 回退应显著小于 aggressive idle swap；
  abnormal grep 为空。

ctx8192:
  V5 RSS drop 应明显大于 ctx4096；
  total RSS drop ratio 应提高；
  KV resident drop 应与 process RSS drop 数量级一致。
```

---

## 13. 预期参考结果

本项目最终提交时的参考结果如下。

### 13.1 ctx4096 final V5

```text
Process RSS drop:
  611.883 MiB

Total RSS drop:
  6.713%

RSS drop / theoretical KV capacity:
  59.754%

Active TPS delta:
  -3.007%

Active decode time delta:
  +3.100%

Total wall time delta:
  +2.713%

Resume first-token weighted avg delta:
  +1.242 ms
```

### 13.2 ctx8192 final V5

```text
Process RSS drop:
  1619.195 MiB

Total RSS drop:
  15.997%

RSS drop / theoretical KV capacity:
  79.062%

Active TPS delta:
  -1.006%

Active decode time delta:
  +1.017%

Total wall time delta:
  +0.966%
```

### 13.3 mincore 参考结果

ctx4096：

```text
baseline-like KV resident:
  1023.75 MiB

V5 KV resident:
  426.0 MiB

KV resident drop:
  597.75 MiB
```

ctx8192：

```text
baseline-like KV resident:
  2047.75 MiB

V5 KV resident:
  426.0 MiB

KV resident drop:
  1621.75 MiB
```

---

## 14. 常见问题

### 14.1 `LLAMA_KV_TRACE_CORPUS_FILE is required`

原因：

```text
trace 使用了 corpus: prompt source，但没有设置 LLAMA_KV_TRACE_CORPUS_FILE。
```

解决：

```bash
export LLAMA_KV_TRACE_CORPUS_FILE=/path/to/corpus.txt
```

对于 ShareGPT converter 生成的 `file:` prompt trace，一般只需要：

```bash
export LLAMA_KV_TRACE_FILE=/path/to/trace.tsv
```

### 14.2 找不到 `wikitext-2-raw/wiki.test.raw`

下载脚本依赖 `unzip`：

```bash
apt-get update
apt-get install -y unzip
bash scripts/get-wikitext-2.sh
```

或使用 Python：

```bash
python3 - <<'PY'
import urllib.request
import zipfile
from pathlib import Path

url = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"
zip_path = Path("/tmp/wikitext-2-raw-v1.zip")
urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(".")
PY
```

### 14.3 pure T0 没有 `kv_mincore_*` 字段

这是正常现象。T0 不启用 paged path，因此没有 mincore telemetry。使用 `M0_no_madvise_mincore` 作为 baseline-like KV resident reference。

### 14.4 V5_mincore 性能和普通 V5 不一致

这是正常现象。`LLAMA_KV_PAGED_MINCORE=1` 是诊断开关，会增加采样开销。不要用 mincore run 的 TPS 作为最终性能结论。

### 14.5 RSS drop 和 `madvise_bytes` 不完全相等

正常。`madvise` 表示向内核建议释放某段虚拟地址范围；实际 RSS drop 受页对齐、内核策略、是否被重新 fault-in、page cache、同时期其他内存活动影响。最终应以 process RSS 和 mincore resident page 为准。

---

## 15. 不应提交的文件

复现过程中不要提交：

```text
模型文件
ShareGPT 原始数据
生成的 prompts/
生成的 trace_out/
kv_logs/
*.out
*.err
/tmp 下的临时脚本
本机绝对路径数据
```

建议检查：

```bash
git status --short
```

如需清理未跟踪实验产物，先人工确认路径，再删除。

---

## 16. 最小复现流程摘要

从干净仓库到最小复现：

```bash
export REPO=/path/to/llama.cpp
export MODEL=/path/to/model.gguf
export LOG_ROOT=/path/to/kv_logs
export SHAREGPT=/path/to/sharegpt.json
export TRACE_DIR=/path/to/trace_out
export TRACE="$TRACE_DIR/trace.tsv"

cd "$REPO"

cmake -S . -B build
cmake --build build -j --target llama-kv-trace-replay

python3 scripts/kv_trace_from_sharegpt.py \
  --input "$SHAREGPT" \
  --output-dir "$TRACE_DIR" \
  --num-sessions 8 \
  --max-turns-per-session 3 \
  --start-gap-ms 0 \
  --turn-gap-ms 400 \
  --jitter-ms 50 \
  --idle-ms-low 1200 \
  --idle-ms-high 2400 \
  --min-decode-tokens 16 \
  --max-decode-tokens 96 \
  --chars-per-token 4 \
  --max-user-chars 2048 \
  --seed 1 \
  --overwrite

export CTX_SIZE=4096
/tmp/run_kv_stage12_reproduce_3run.sh

/tmp/parse_kv_trace_perf.py "$LOG_ROOT/stage12_reproduce_ctx4096_3run"
```

完整复现包括：

```text
1. smoke；
2. ctx4096 T0/V5 3-run median；
3. ctx8192 T0/V5 3-run median；
4. ctx4096 M0/V5 mincore；
5. ctx8192 M0/V5 mincore；
6. abnormal grep；
7. 结果与 docs/kv_trace_replay_stage12c_real_sharegpt_results.md 对照。
```

---

## 17. 复现结论写法

如果结果通过，应写：

```text
在 ShareGPT-backed trace replay workload 下，V5 fast-maintenance 配置相对 T0 baseline 实现了显著 process RSS drop；ctx8192 下 RSS drop 更明显，且 mincore 诊断显示 KV resident pages 从约 2 GiB 降至约 426 MiB，说明 RSS 下降主要来自 KV cache 物理驻留页释放。所有正式 run 均应满足 exit=0、real_abnormal=0、all sessions finished，且无 active-visible violation 与 write-to-swapped violation。
```

不要写：

```text
1. 已完成真实线上 serving trace 验证；
2. 已完整复刻 vLLM PagedAttention；
3. 已优化 peak RSS；
4. 已支持所有 backend；
5. KV cache 永久减少了固定比例。
```
