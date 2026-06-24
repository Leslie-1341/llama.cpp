# llama.cpp KV Cache 内存优化

本项目面向“操作系统功能赛 / 边缘设备上的 LLM 推理优化”方向，基于 `llama.cpp` 实现 KV cache 运行时内存优化。项目目标是在真实推理过程中降低进程 current RSS，并通过 resident page 诊断证明 KV cache 对应物理页确实被释放，同时控制吞吐和 resume latency 的回退。

## 1. 项目概述

大语言模型推理过程中，KV cache 会随上下文长度和并发会话数增长而快速膨胀。原始 `llama.cpp` 的 KV cache 主要采用“构造期连续预分配大块 K/V tensor + cell 级元数据管理”的方式：写侧可以按 cell scatter 写入，但读侧长期依赖 `[0, n_kv)` 的连续视图，再通过 attention mask 屏蔽无效 token。该设计简单稳定，但在长上下文、多会话、暂停/恢复类 workload 中容易导致大量暂时不活跃的 KV 仍然常驻物理内存。

本项目围绕这一问题，逐步实现了：

```text
tail / lazy reclaim
paged row index / non-identity gather
idle KV block swap-out
madvise-based resident page release
resume swap-in
prefetch / defer
fast maintenance
trace replay workload
mincore-based KV resident diagnostics
```

最终形成了一套面向 idle session 的 KV cache 回收机制：

```text
active session:
  保持当前推理所需 KV block resident。

paused / idle session:
  其 idle-owned KV block 可被 swap-out，并通过 MADV_DONTNEED 释放物理页。

resume session:
  根据上层调度信号，可提前 prefetch，或在 resume path 中按需 swap-in。

fast maintenance:
  降低 idle swap maintenance 对 decode 热路径的影响。
```

## 2. 核心成果

最终推荐配置为 fast-maintenance V5。该配置在保留主要 RSS 收益的同时，大幅降低了早期 aggressive idle swap 带来的吞吐损失。

### 2.1 ctx4096 结果

在 ShareGPT-backed trace replay workload、Llama-3-8B Q4_K_M、KV cache K/V 为 f32 的配置下：

| 指标                                    | 结果          |
| ------------------------------------- | ----------- |
| Process RSS drop                      | 611.883 MiB |
| Total RSS drop                        | 6.713%      |
| RSS drop / theoretical KV capacity    | 59.754%     |
| Active TPS delta                      | -3.007%     |
| Active decode time delta              | +3.100%     |
| Total wall time delta                 | +2.713%     |
| Resume first-token weighted avg delta | +1.242 ms   |

### 2.2 ctx8192 结果

在 ctx-size 扩大到 8192 后，KV cache 占进程总内存比例更高，因此收益更明显：

| 指标                                 | 结果           |
| ---------------------------------- | ------------ |
| Process RSS drop                   | 1619.195 MiB |
| Total RSS drop                     | 15.997%      |
| RSS drop / theoretical KV capacity | 79.062%      |
| Active TPS delta                   | -1.006%      |
| Active decode time delta           | +1.017%      |
| Total wall time delta              | +0.966%      |

### 2.3 KV resident page 诊断

使用 `LLAMA_KV_PAGED_MINCORE=1` 进行 resident page 诊断后，结果表明 RSS 下降并非统计噪声，而是与 KV resident page 下降直接对应。

ctx8192 下：

```text
baseline-like KV resident:
  2047.75 MiB

V5 final KV resident:
  426.0 MiB

KV resident drop:
  1621.75 MiB
```

这说明最终 RSS drop 主要来自 KV cache 物理驻留页减少，而不是日志、page cache 或其他偶然因素。

## 3. 关键设计

### 3.1 原始问题：连续读窗口限制

`llama.cpp` 原始 KV cache 具有如下特征：

```text
1. 构造期按 n_ctx 预分配大连续 K/V tensor；
2. cell metadata 记录 token position、seq ownership 等信息；
3. 写侧通过 ggml_set_rows 按 slot_info.idxs scatter 写入；
4. 读侧 get_k/get_v 长期返回 [0, n_kv) 连续 view；
5. attention mask 只影响数值结果，不等价于避免物理读取。
```

因此，早期 exact swap 原型虽然能完成 swap-out / swap-in correctness，但只要读侧仍要求 `[0, n_kv)` 全区间 resident，被释放的页就会很快被换回，RSS 下降难以稳定保留。

### 3.2 Paged row index

项目引入 paged row index 机制，在 graph 内通过 row index 控制 K/V 读取行：

```text
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
```

对于 active-visible KV：

```text
row_idx 指向真实 physical row。
```

对于当前 active session 不可见的 idle-only KV：

```text
row_idx 可重映射到 resident dummy row。
```

这样可以让 idle block 退出 active read window，为后续安全 swap-out / madvise 创造条件。

### 3.3 Idle KV block swap-out

当一个 session 暂停后，其拥有的 KV block 会变成 idle-owned block。系统在确认这些 block 不会被当前 active session 读取后，可以执行：

```text
RESIDENT -> SWAPPED
```

swap-out 会将 block 内容写入 backing store，并记录 offset / size / state。配合 madvise 后，其物理页可以被系统回收。

关键开关：

```bash
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
```

### 3.4 Resume swap-in

当 idle session 重新 resume 时，如果其历史 KV block 已经处于 `SWAPPED` 状态，系统会在读写前恢复：

```text
SWAPPED -> RESIDENT
```

这保证 resume 后仍能使用原历史上下文，不把 idle reclaim 变成简单丢弃。

### 3.5 Prefetch 与 defer

为了减少 resume first-token latency，项目实现了 prefetch 机制：

```text
llama_memory_prefetch_seq(...)
llama_memory_prefetch_seq_step(...)
llama_memory_set_seq_prefetch_protected(...)
```

调度层可以在 session 即将 resume 前逐步恢复 KV block。

同时，`LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1` 用于避免 resume first-token 阶段同步触发新的 idle swap-out，从而减少首 token 暴露延迟。

### 3.6 Fast maintenance

早期 aggressive idle swap 虽然可以带来更大 RSS drop，但吞吐下降较明显。最终引入 fast-maintenance 控制项：

```bash
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16
```

其核心思想是：

```text
不在每一步都做完整 idle maintenance；
限制每次维护处理的 block 数；
只回收已 idle 足够久的 block；
关闭非必要 debug probe。
```

最终 V5 配置用约 100 MiB 的 RSS 收益让渡，换回约 30 个百分点的 TPS 损失恢复。

## 4. 最终推荐配置

以下为 Stage 12-C 最终推荐配置。实际使用时可按 workload 调整。

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

## 5. 主要代码路径

### 5.1 核心源码

```text
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-graph.cpp
src/llama-graph.h
src/llama-memory.h
include/llama.h
```

其中：

```text
src/llama-kv-cache.cpp / .h:
  KV block metadata、swap-out、swap-in、madvise、mincore、prefetch、fast maintenance。

src/llama-graph.cpp / .h:
  paged row index 接入 graph attention 读路径。

src/llama-memory.h、include/llama.h:
  暴露少量 memory-level experimental API，用于 prefetch / protected 控制。
```

### 5.2 Example drivers

```text
examples/kv-trace-replay/
examples/kv-semi-real-multisession/
examples/kv-idle-swap-resume/
examples/kv-idle-telemetry/
```

其中最终推荐入口是：

```text
examples/kv-trace-replay/
```

它支持 trace-driven workload，能够表达多 session、多 turn、arrival time、idle interval、resume 等行为。

### 5.3 Scripts

```text
scripts/kv_trace_from_sharegpt.py
```

该脚本用于将 ShareGPT 格式数据转换为 `kv-trace-replay` 可读取的 trace 与 prompt 文件。

## 6. 快速构建

```bash
cmake -S . -B build
cmake --build build -j --target llama-kv-trace-replay
```

也可以同时构建相关 driver：

```bash
cmake --build build -j --target \
  llama-kv-trace-replay \
  llama-kv-semi-real-multisession \
  llama-kv-idle-swap-resume
```

## 7. 快速 smoke

设置模型路径：

```bash
MODEL=/path/to/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

运行默认 trace replay smoke：

```bash
LLAMA_KV_TRACE_FILE=examples/kv-trace-replay/traces/smoke_4s2t.tsv \
LLAMA_KV_TRACE_CORPUS_FILE=/path/to/corpus.txt \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 512 \
  --batch-size 128 \
  --ubatch-size 64 \
  --parallel 4 \
  --seed 1 \
  --temp 0
```

运行 paged + idle swap smoke：

```bash
LLAMA_KV_PAGED=1 \
LLAMA_KV_PAGED_INGRAPH=1 \
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP=1 \
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0 \
LLAMA_KV_TRACE_FILE=examples/kv-trace-replay/traces/smoke_4s2t.tsv \
LLAMA_KV_TRACE_CORPUS_FILE=/path/to/corpus.txt \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 512 \
  --batch-size 128 \
  --ubatch-size 64 \
  --parallel 4 \
  --seed 1 \
  --temp 0
```

通过标准：

```text
1. 每个 session 有 KV_TRACE_SUMMARY；
2. KV_TRACE_PERF 正常输出；
3. exit code = 0；
4. 无 assert / abort / segmentation fault / NaN；
5. 无 active-visible violation；
6. 无 write-to-swapped violation。
```

## 8. ShareGPT-backed trace replay

本项目最终 workload 使用 ShareGPT 对话数据生成 trace。需要先准备 ShareGPT JSON / JSONL 数据，然后转换：

```bash
python3 scripts/kv_trace_from_sharegpt.py \
  --input /path/to/sharegpt.json \
  --output-dir /path/to/trace_out \
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

转换后会生成：

```text
trace.tsv
prompts/
```

运行 trace replay：

```bash
TRACE=/path/to/trace_out/trace.tsv
MODEL=/path/to/model.gguf

LLAMA_KV_TRACE_FILE="$TRACE" \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 4096 \
  --batch-size 512 \
  --ubatch-size 128 \
  --parallel 8 \
  --seed 1 \
  --temp 0
```

注意：该 workload 是 **ShareGPT-backed synthetic trace**，不是线上生产 trace。真实部分来自 ShareGPT 对话内容与多轮结构；synthetic 部分包括 arrival time、idle interval、target decode tokens 和 long-idle 分布。

## 9. 结果文档

最终详细结果见：

```text
docs/kv_trace_replay_stage12c_real_sharegpt_results.md
```

历史阶段文档已归档到：

```text
docs/archive/kv-stage-history/
```

建议阅读顺序：

```text
1. README_KV_OPT.md
2. docs/reproduce_kv_cache_optimization.md
3. docs/final_technical_report.md
4. docs/kv_trace_replay_stage12c_real_sharegpt_results.md
5. docs/archive/kv-stage-history/
```

## 10. Correctness 与 Safety

项目所有关键实验均检查以下安全项：

```text
exit = 0
real_abnormal = 0
all_finished = 1
active_visible_violation = 0
write_to_swapped = 0
paged_swapped_active_visible_violation = 0
paged_write_to_swapped_block = 0
backend_failures = 0
```

同时通过以下路径验证正确性：

```text
1. baseline vs paged / swap 输出对比；
2. seq active 段输出一致性；
3. seq resume 段输出一致性；
4. swap-out / swap-in counter；
5. mincore resident page diagnostic；
6. trace replay 多 session 完成情况。
```

## 11. 当前边界与限制

本项目当前仍是比赛原型 / research prototype，不是完整生产级 serving runtime。需要注意以下边界：

```text
1. 重点优化 current RSS，不声称系统 peak RSS 已被稳定优化；
2. ShareGPT-backed trace 不是生产线上真实请求 trace；
3. 当前策略主要在 example / trace driver 中表达，并非完整 server scheduler；
4. public memory API 属于实验性扩展，不代表最终 upstream API 设计；
5. 尚未实现真正异步 prefetch thread；
6. 尚未完整接入 llama-server HTTP serving benchmark；
7. 当前主要验证 CPU KV path；
8. 不声称完整复刻 vLLM PagedAttention。
```

更准确的表述是：

```text
本项目在 llama.cpp 上实现并验证了 block-level KV cache reclaim 机制，
证明 idle session 的 KV block 可以被安全换出、释放物理页，并在 resume 时恢复；
在 ShareGPT-backed trace replay workload 下，最终配置实现了可观 RSS 下降和极少量性能回退。
```

## 12. 后续工作

后续可继续推进：

```text
1. 接入真实 llama-server slot / request scheduler；
2. 将 trace replay 扩展到更真实的 arrival distribution；
3. 引入真实 memory pressure signal，例如 cgroup memory pressure / PSI；
4. 实现真正异步 prefetch thread；
5. 扩展 GPU backend 兼容性；
6. 收敛 experimental public API；
7. 将 reclaim policy 工程化为统一策略开关；
8. 增加更多模型、上下文长度和并发规模验证。
```

## 13. 关键结论

本项目最终证明：

```text
1. llama.cpp 原始 KV cache 的连续预分配设计存在 idle KV 物理页常驻问题；
2. 通过 paged row index，可以让 idle-only block 退出 active read window；
3. idle KV block 可以被 swap-out + madvise，从而真实降低 RSS；
4. resume swap-in / prefetch / defer 可以控制恢复成本；
5. fast maintenance 能显著降低 idle swap 维护开销；
6. mincore 诊断证明最终 RSS 下降与 KV resident page 下降高度一致；
7. 在 ctx8192 ShareGPT-backed trace replay 中，最终配置实现约 1619 MiB RSS 下降，约 16.0% total RSS drop，TPS 回退约 1.0%。
```
