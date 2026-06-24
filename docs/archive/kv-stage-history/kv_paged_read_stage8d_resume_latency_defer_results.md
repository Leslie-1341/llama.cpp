# KV Paged Read — Stage 8D Resume Latency Defer Results

## 1. Goal

Stage 8D 的目标是收束 Stage 8C clean performance matrix 中暴露出的一个问题：

> full prefetch 已经消除了 resume fallback，但 resume first-token latency 仍然高于 paged swap-off baseline。

本阶段重点不是继续扩大 RSS 下降，而是定位并降低 full-prefetch 路径中的 residual resume latency。

具体目标如下：

1. 解释为什么 `fallback_blocks=0` 后 `resume_first_ms` 仍然偏高；
2. 判断 residual latency 来自 prefetch、fallback、check-read，还是来自其他同步路径；
3. 通过 timing probe 定位 resume 首 token 路径中的主要耗时；
4. 设计并验证 `LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1`；
5. 在正式 3-run clean perf matrix 中验证：

   * resume first-token latency 回到 baseline 范围；
   * current RSS drop / madvise 收益基本保持；
   * fallback 仍为 0；
   * correctness 和异常检查保持干净。

本阶段仍属于 **controlled multi-seq idle/resume benchmark**。它不是真实数据集测试，也不是 server scheduler 测试。底层 idle-owned KV block 的识别是运行时自动完成的，但上层 workload 的 idle/resume 时机仍由 example driver 构造。

---

## 2. Background from Stage 8C

Stage 8C clean performance matrix 已经证明，在关闭 shadow validation、关闭 mincore 和重日志的 clean perf 路径下，paged bookkeeping 本身的开销较低；idle KV swap/madvise 能带来明确的 current RSS drop；full prefetch 能消除 resume fallback。

Stage 8C 的核心现象是：

```text
paged swap-off baseline:
  resume_first_ms ≈ 80–82 ms

swap + madvise + full prefetch:
  resume_first_ms ≈ 99–103 ms
  fallback_blocks = 0
  process_rss_drop ≈ 124 MiB
```

也就是说，full prefetch 已经把 resume 需要的旧 KV blocks 提前恢复，`fallback_blocks=0`，但 resume first-token latency 仍然比 baseline 高约 17–20 ms。

这说明 residual latency 不是由 fallback restore 直接造成的。Stage 8D 的任务就是继续定位这部分剩余延迟。

---

## 3. Stage 8D-2 Ablation Summary

Stage 8D-2 通过 ablation matrix 分离了几类可能来源：

1. idle trace / idle maintenance 是否本身造成主要延迟；
2. logical swap metadata path 是否有成本；
3. madvise / physical reclaim 是否带来额外成本；
4. prefetch timing 是否足够早；
5. check-read 是否是主要开销。

关键判断如下：

```text
D0_paged_swap_off:
  baseline path

D1_idle_trace_no_swap:
  idle trace 本身只带来很小差异，不是主因

D2_swap_metadata_no_madvise_no_prefetch:
  logical swap / metadata / fallback path 明显增加 resume latency

D3_swap_metadata_no_madvise_full_prefetch:
  full prefetch 能降低 fallback 成本，但仍有 residual

D4/D5/D6:
  madvise 会增加部分成本，但不是 residual 的唯一来源
```

该阶段得出的主要结论是：

```text
residual penalty 主要来自 logical swap / idle maintenance / resume path；
check_read 不是主因；
madvise 是次要贡献，不足以解释全部 residual latency。
```

因此 Stage 8D 继续转向更细粒度的 timing probe。

---

## 4. Stage 8D-3 / 8D-4 Timing Diagnosis

Stage 8D-3 增加了累计 timing，用于拆分：

```text
set_input_us
idle_maintenance_us
swap_out_us
check_read_us
```

累计 timing 显示：

1. `check_read_us` 很小，不是主因；
2. set_input / idle maintenance / swap-out 在 full-prefetch swap path 中显著增加；
3. 需要进一步确认这些开销是否集中在 resume first-token step。

Stage 8D-4 进一步增加 per-step timing 和 resume first-token marker：

```text
KV_RESUME_FIRST_BEGIN
KV_PAGED_STEP_TIMING ...
KV_RESUME_FIRST_END
```

关键 resume first-token step 结果为：

```text
R5/R6 resume first step:
  swap_out_calls = 7
  swap_out_us ≈ 12–14 ms
  idle_maintenance_us ≈ 12–14 ms
  set_input_us ≈ 12–14 ms
  remaining_prefetch_blocks = 0
```

这说明：

```text
full prefetch 已经恢复了 resume 所需旧 KV；
resume 首 token 仍然变慢，是因为同一个 decode step 内又同步执行了新的 idle swap-out。
```

换句话说，full-prefetch 后剩余的首 token penalty 不是来自旧 KV 没恢复，而是来自 **resume 首步内触发的新一轮 idle reclaim**。

正式 performance matrix 中没有开启这些 timing probe。timing 结果来自单独诊断 run；正式 perf 数据使用：

```text
LLAMA_KV_PAGED_RESUME_TIMING=0
LLAMA_KV_PAGED_RESUME_TIMING_STEP=0
```

因此 timing instrumentation 没有污染 Stage 8D-5 的正式性能结果。

---

## 5. Stage 8D-5 Policy

为解决 resume first-token step 被新 idle swap-out 阻塞的问题，Stage 8D-5 增加了一个默认关闭的 policy gate：

```text
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

其语义是：

```text
当 driver 即将执行 resume first-token decode 时，
请求 KV cache 在下一次 row_idx fill / idle maintenance 中跳过新的 idle swap-out。
```

该策略只跳过新的 idle swap-out，不影响以下路径：

1. prefetch restore；
2. fallback restore；
3. row_idx remap；
4. active-visible correctness scan；
5. `paged_check_read_resident()`；
6. existing swapped block redirect；
7. ordinary paged-off / single-request path。

实现边界：

```text
defer(1) 当前按 row_idx fill 计数，而不是严格按 llama_decode 次数计数。
```

也就是说，`defer(1)` 表示下一次 row_idx fill 中跳过 idle swap-out。当前 controlled benchmark 已经验证该 defer 命中 resume first-token step；未来接入 server scheduler 或更复杂的多 turn workload 时，需要重新审视这个计数单位是否仍然合适。

---

## 6. Once Probe Result

Stage 8D-5 once probe 用于验证 policy 是否真的把 resume 首步的新 idle swap-out 延后。

对比结果如下：

```text
E5_full_prefetch_no_defer:
  set_input_us = 13913
  idle_maintenance_us = 13520
  swap_out_us = 13334
  swap_out_calls = 7
  defer_idle_swapout = 0
  swapout_deferred_blocks = 0
```

```text
E6_full_prefetch_defer:
  set_input_us = 575
  idle_maintenance_us = 177
  swap_out_us = 0
  swap_out_calls = 0
  defer_idle_swapout = 1
  swapout_deferred_blocks = 7
```

该结果说明：

```text
resume first-token step 内原本发生的 7 次新 idle swap-out 被成功延后；
首步 set_input_us 从约 13.9 ms 降至约 0.6 ms；
swap_out_us 从约 13.3 ms 降至 0。
```

这证明 `LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1` 的机制生效。

---

## 7. Formal 3-run Performance Matrix

### 7.1 Experimental Setup

Stage 8D-5 formal perf matrix 使用 3-run median，固定 workload 配置如下：

```text
ctx = 2048
parallel = 4
warmup = 256
n = 128
LLAMA_KV_IDLE_NUM_IDLE_SEQS = 2
cache-type-k = f32
cache-type-v = f32
kv-unified = on
temp = 0
3 runs per case
```

该 benchmark 是 controlled multi-seq idle/resume workload。其基本结构是：

```text
seq0 / seq2 构造为 idle seq；
seq1 作为 active decode seq；
随后 seq0 resume。
```

其中，F0/F1/F2/F3 不是四个真实对话场景，而是同一 workload 下的四种 policy case。

正式 perf matrix 显式关闭：

```text
LLAMA_KV_PAGED_RESUME_TIMING=0
LLAMA_KV_PAGED_RESUME_TIMING_STEP=0
LLAMA_KV_PAGED_MINCORE=0
LLAMA_KV_PAGED_TRACE=0
LLAMA_KV_PAGED_REFAULT_TRACE=0
LLAMA_KV_CACHE_DEBUG=0
LLAMA_KV_PAGED_SHADOW_VALIDATE=0
```

swap/madvise case 保留：

```text
LLAMA_KV_PAGED_IDLE_TRACE=1
```

当前实现中，`LLAMA_KV_PAGED_IDLE_TRACE=1` 不只是日志开关，还承载 idle maintenance / swap-out 路径，因此在 swap/madvise case 中需要保留。

---

### 7.2 Case Design

| case                           | policy                      | 含义                                            |
| ------------------------------ | --------------------------- | --------------------------------------------- |
| F0_paged_swap_off              | paged_swap_off              | paged on, swap off baseline                   |
| F1_full_prefetch_no_defer      | full_prefetch_no_defer      | swap + madvise + full prefetch, no defer      |
| F2_full_prefetch_defer         | full_prefetch_defer         | swap + madvise + full prefetch + resume defer |
| F3_full_prefetch_earlier_defer | full_prefetch_earlier_defer | earlier full prefetch + resume defer          |

---

### 7.3 Median Results

| case                           | policy                      | resume_first_ms | madvise_mib | process_rss_drop_mib | prefetch_blocks | fallback_blocks | active_tps |
| ------------------------------ | --------------------------- | --------------: | ----------: | -------------------: | --------------: | --------------: | ---------: |
| F0_paged_swap_off              | paged_swap_off              |          87.089 |       0.000 |                0.000 |               0 |               0 |     11.263 |
| F1_full_prefetch_no_defer      | full_prefetch_no_defer      |         104.210 |     153.750 |              124.176 |              17 |               0 |     11.604 |
| F2_full_prefetch_defer         | full_prefetch_defer         |          84.819 |     153.750 |              124.238 |              17 |               0 |     11.646 |
| F3_full_prefetch_earlier_defer | full_prefetch_earlier_defer |          83.643 |     153.750 |              124.070 |              17 |               0 |     11.734 |

关键 delta：

```text
F1 - F0 = +17.121 ms
F2 - F1 = -19.391 ms
F3 - F1 = -20.567 ms
```

解释：

1. F1 相对 F0 仍有约 17 ms residual latency，说明 full prefetch no-defer 路径仍然把新 idle swap-out 放在 resume 首 token 关键路径内；
2. F2 开启 defer 后，`resume_first_ms` 从 104.210 ms 降到 84.819 ms，回到 baseline 范围；
3. F3 使用 earlier prefetch + defer，`resume_first_ms` 为 83.643 ms，与 F2 接近；
4. F1/F2/F3 的 `madvise_mib` 均为 153.750 MiB；
5. F1/F2/F3 的 `process_rss_drop_mib` 均约为 124 MiB；
6. F1/F2/F3 的 `prefetch_blocks` 均为 17，`fallback_blocks` 均为 0。

因此可以得出：

```text
defer policy 降低了 resume first-token latency；
defer 不是取消 swap-out，而是把 resume 首 token 内的新 swap-out 延后；
长期 madvise / RSS 收益基本保持。
```

需要注意的是，F2/F3 的 `resume_first_ms` 略低于 F0 baseline，但不应宣传为“比 baseline 更快”。更稳妥的表述是：

```text
defer 后 resume_first_ms 回到 baseline 范围。
```

---

## 8. Correctness and Clean Perf Conditions

Stage 8D-5 formal 3-run matrix 的 correctness 和异常检查结果如下：

```text
12 runs exit = 0
seq1_decoded = 128
resume_decoded = 128
seq1_equal = 0
resume_equal = 0
real_abnormal = 0
active_visible_violation = 0
visible_violation_rows = 0
active_violation_rows = 0
```

此外：

```text
no timing/debug leakage check: empty
abnormal grep: empty
```

说明：

1. 所有 run 均正常退出；
2. seq1 active 段输出一致；
3. seq0 resume 段输出一致；
4. 没有 active-visible violation；
5. 没有真实异常；
6. 没有 timing / debug trace 泄漏到正式 perf 日志中。

提交前额外 smoke 也已通过：

```text
paged-off / ordinary path smoke: passed
defer no-op smoke: passed
```

这说明新增 defer policy 和 timing wrapper 没有污染 paged-off / ordinary path，也没有在无 idle swap 配置下产生副作用。

---

## 9. Implementation Notes and Semantic Boundary

### 9.1 `paged_trace_step` semantic change

Stage 8D 的实现中，`paged_trace_step` 的自增语义发生了一个诊断层面的变化：

```text
paged_trace_step 现在在 row_idx fill 路径上无条件推进，
而不是只在 paged trace enabled 时推进。
```

这样做的目的是让 timing / refault / step-level diagnosis 在 regular paged trace 关闭时仍然拥有连续可读的 step 编号。

该变化影响的是 diagnostic step numbering，不影响默认 perf 路径。正式 perf matrix 中 timing / refault trace 均关闭，因此该变化不会污染 Stage 8D-5 的性能数据。

该语义变化需要在 commit message 和结果文档中明确说明，不能将 Stage 8D 改动描述为“完全无行为变化”。

---

### 9.2 Defer counter unit

当前 defer 的计数单位是：

```text
row_idx fill
```

而不是严格意义上的：

```text
llama_decode call
```

因此：

```text
defer(1) means skipping idle swap-out for the next row_idx fill.
```

在当前 controlled benchmark 中，once probe 已经证明该 defer 命中 resume first-token step：

```text
swap_out_calls: 7 -> 0
swapout_deferred_blocks: 7
```

但未来如果进入 server scheduler、更复杂的 multi-turn workload，或者 resume first-token 前存在额外 row_idx fill，该计数语义需要重新审视。

---

## 10. Boundary and Non-claims

Stage 8D 的结果需要保持以下边界：

1. 这是 controlled multi-seq idle/resume benchmark，不是真实数据集结果；
2. 这不是 server scheduler 结果；
3. F0/F1/F2/F3 是 policy cases，不是四个真实对话场景；
4. 底层 KV 层会根据当前 active seq 自动识别 idle-owned blocks，但 workload 层面的 idle/resume 时机仍由 example driver 构造；
5. 当前方案不是完整 vLLM PagedAttention；
6. 当前结果证明的是 current RSS reduction，不是 peak RSS optimization；
7. 当前结果不能直接外推为真实业务场景下一定获得相同的 124 MiB RSS drop；
8. 当前 defer policy 的 `defer(1)` 在真实 server scheduler 中是否足够，需要 Stage 9 继续验证。

当前阶段可以严谨表述为：

```text
在 controlled multi-seq idle/resume benchmark 中，Stage 8D 验证了 idle KV swap/madvise + full prefetch + resume-step defer 的完整机制链路。该机制在保持 correctness 的前提下，获得约 124 MiB current RSS drop，并将 full-prefetch no-defer 的 resume first-token latency 从 104.210 ms 降至 84.819 ms，回到 baseline 范围。
```

不应表述为：

```text
真实数据集上已经验证；
真实 server scheduler 已完成；
完整 PagedAttention 已实现；
系统一定比 baseline 更快；
peak RSS 已优化。
```

---

## 11. Next Step

Stage 8D 已经足够收束。下一阶段建议进入 Stage 9，但不要直接进入真实 dataset / real trace。

推荐路线是：

```text
Stage 8D results doc + commit
→ Stage 9 semi-real multi-turn workload
→ real prompt / real trace workload
→ server scheduler integration
```

Stage 9 第一阶段建议目标：

```text
构造 semi-real multi-turn workload：
使用真实 prompt / 多轮对话文本，
但 idle/resume 的轮转仍由 driver 控制，
以便保持可归因。
```

Stage 9 需要回答的问题包括：

1. 在更接近真实对话的 prompt 长度和 turn-taking 下，idle-owned KV coverage 是否仍然足够；
2. swap/madvise 是否仍然能获得稳定 current RSS drop；
3. full prefetch + resume defer 是否仍能控制 resume first-token latency；
4. `defer(1)` 在更复杂的 multi-turn 场景中是否仍然足够；
5. 是否需要把 defer 从固定 1 次 row_idx fill 扩展为 N-step 或 resume-sensitive window；
6. 是否需要更自然的 idle timeout / resume signal。

只有 Stage 9 semi-real workload 站稳后，才适合进一步进入真实 trace 或 server scheduler。

---

## 12. Summary

Stage 8D 的最终结论如下：

```text
Stage 8D 定位并解决了 full prefetch 后的 residual resume first-token latency。
```

具体来说：

1. Stage 8C 发现 full prefetch 后 fallback 已为 0，但 resume first-token latency 仍然偏高；
2. Stage 8D-2/3/4 证明 residual latency 的主要来源是 resume 首步内同步执行的新 idle swap-out；
3. Stage 8D-5 新增 `LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1`，在 resume first-token step 延后新的 idle swap-out；
4. once probe 证明 `swap_out_calls` 从 7 降到 0；
5. formal 3-run matrix 证明 `resume_first_ms` 从 104.210 ms 降至 84.819 ms，回到 baseline 范围；
6. `madvise_mib=153.750 MiB` 和 `process_rss_drop≈124 MiB` 基本保持；
7. `fallback_blocks=0`，correctness 和 abnormal 检查均通过；
8. 该结果仍限定在 controlled multi-seq idle/resume benchmark 中，下一阶段需要进入 semi-real multi-turn workload。

因此，Stage 8D 可以作为当前 idle KV swap/madvise + prefetch + resume defer 机制的一个完整阶段性收束点。
