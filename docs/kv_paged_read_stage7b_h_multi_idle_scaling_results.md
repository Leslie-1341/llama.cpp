# Stage 7B-H：Multi-idle KV Scaling Matrix Results

## 1. 背景

本阶段目标是验证：

> 在多 idle request 场景下，idle-owned KV blocks 增加后，当前 block-level idle swap + madvise 机制是否能够同步放大 RSS 下降收益。

前序 Stage 7B 中已经完成以下工作：

1. **S0 controlled smoke**
   验证最小配置下 idle KV swap/madvise 链路有效。

2. **S1 controlled smoke**
   将 `ctx` 和 `warmup` 放大后，`idle-owned blocks` 和 `RSS drop` 均明显增加，说明单 idle seq 的 history 变长可以放大 RSS 收益。

3. **S2 controlled smoke**
   将 `--parallel` 从 2 提到 4 后，RSS 收益没有继续放大。后续审计确认：原 example driver 实际只构造了 `seq0 idle/resume` 和 `seq1 active decode`，`--parallel=4` 只是扩大 KV capacity，没有真实构造 `seq2/seq3` workload。

4. **Stage 7B-G multi-idle workload implementation**
   在 `examples/kv-idle-swap-resume/idle-swap-resume.cpp` 中新增：

   ```text
   LLAMA_KV_IDLE_NUM_IDLE_SEQS
   ```

   默认值为 1，保持旧行为不变。开启后可以构造多个 idle seq：

   ```text
   NUM_IDLE_SEQS=1 -> idle seqs = {0}, active seq = 1
   NUM_IDLE_SEQS=2 -> idle seqs = {0,2}, active seq = 1
   NUM_IDLE_SEQS=3 -> idle seqs = {0,2,3}, active seq = 1
   ```

Stage 7B-H 在此基础上运行 `NUM_IDLE_SEQS=1/2/3` matrix，形成更干净的 multi-idle scaling 曲线。

---

## 2. 实验目标

本实验重点回答三个问题：

1. `LLAMA_KV_IDLE_NUM_IDLE_SEQS` 是否真正构造了多个 idle seq？
2. idle seq 数量从 1 增加到 3 时，`idle-owned blocks` 和 `swapped blocks` 是否按预期放大？
3. RSS drop 是否随 idle-owned KV 规模近似线性放大？

本阶段仍属于 **controlled microbenchmark / release smoke**，不是最终真实 workload 结果，也不是最终性能优化结果。

---

## 3. 实验配置

固定配置如下：

```text
ctx=2048
parallel=4
warmup=256
n=128
pressure_mode=high
cache-type-k=f32
cache-type-v=f32
kv-unified=enabled
```

关键环境变量：

```text
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_IDLE_TRACE=1
LLAMA_KV_PAGED_TRACE=1
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1

LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=high
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1
LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
```

其中：

```text
pressure_mode=high
```

表示：

```text
target_restore_blocks=0
prefetch_blocks=0
```

也就是在 active decode 阶段不主动 prefetch，尽量保留 RSS 下降收益，将恢复代价留到 resume fallback 阶段。

---

## 4. Matrix 设计

本阶段只改变一个变量：

```text
LLAMA_KV_IDLE_NUM_IDLE_SEQS = 1 / 2 / 3
```

三组 workload 分别为：

| workload   | idle seqs | active seq | 说明          |
| ---------- | --------- | ---------- | ----------- |
| h_numidle1 | `{0}`     | `1`        | 旧行为回归       |
| h_numidle2 | `{0,2}`   | `1`        | 两个 idle seq |
| h_numidle3 | `{0,2,3}` | `1`        | 三个 idle seq |

---

## 5. 结果总表

| 指标                             | NUM_IDLE_SEQS=1 | NUM_IDLE_SEQS=2 | NUM_IDLE_SEQS=3 |
| ------------------------------ | --------------: | --------------: | --------------: |
| max_seen_seq_count             |               2 |               3 |               4 |
| max_active_seq_count           |               1 |               1 |               1 |
| max_idle_seq_count             |               1 |               2 |               3 |
| max_idle_owned_blocks          |              17 |              34 |              51 |
| max_swapped_blocks             |              17 |              34 |              51 |
| blocks_in_use_at_max_swapped   |              19 |              36 |              54 |
| capacity_blocks_at_max_swapped |             128 |             128 |             128 |
| idle_cov_inuse_pct             |         89.474% |         94.444% |         94.444% |
| idle_cov_capacity_pct          |         13.281% |         26.562% |         39.844% |
| rss_drop_mib                   |      63.656 MiB |     123.965 MiB |     185.668 MiB |
| process_rss_drop_pct           |          0.753% |          1.467% |          2.197% |
| drop_over_idle_pct             |         93.612% |         91.151% |         91.014% |
| release_ratio_capacity_pct     |         12.433% |         24.212% |         36.263% |
| release_ratio_inuse_pct        |         83.758% |         86.087% |         85.957% |
| rss_drop_over_advised_pct      |         67.900% |         80.628% |         85.365% |
| max_madvise_calls              |              25 |              41 |              58 |
| max_madvise_bytes_mib          |      93.750 MiB |     153.750 MiB |     217.500 MiB |
| madvise_failures               |               0 |               0 |               0 |
| resume_first_ms                |      166.890 ms |      172.184 ms |      181.211 ms |
| tokens_per_second              |       10.671657 |        9.692740 |        9.114787 |
| net_rss_drop_before_resume_mib |      62.441 MiB |     123.965 MiB |     185.668 MiB |
| seq1_decoded_tokens            |             128 |             128 |             128 |
| resume_decoded_tokens          |             128 |             128 |             128 |
| correctness_status             |         partial |         partial |         partial |
| real_abnormal_matches          |               0 |               0 |               0 |

---

## 6. Workload 生效性验证

日志分布显示：

```text
NUM_IDLE_SEQS=1:
  max_seen_seq_count=2
  max_idle_seq_count=1

NUM_IDLE_SEQS=2:
  max_seen_seq_count=3
  max_idle_seq_count=2

NUM_IDLE_SEQS=3:
  max_seen_seq_count=4
  max_idle_seq_count=3
```

这说明 `LLAMA_KV_IDLE_NUM_IDLE_SEQS` 已经真实改变 workload 结构。

也就是说：

```text
NUM_IDLE_SEQS=3
```

不再只是扩大 KV capacity，而是实际构造了：

```text
seq0 idle
seq2 idle
seq3 idle
seq1 active
```

这解决了 S2 阶段暴露的问题：原先 `--parallel=4` 只是容量参数，没有真实产生多个 idle request。

---

## 7. Idle-owned KV scaling

`max_idle_owned_blocks` 结果为：

```text
17 -> 34 -> 51
```

这是严格线性增长：

```text
1 idle seq: 17 blocks
2 idle seq: 34 blocks = 2 × 17
3 idle seq: 51 blocks = 3 × 17
```

`max_swapped_blocks` 同样为：

```text
17 -> 34 -> 51
```

说明：

1. 多 idle seq 的 KV block 被正确识别为 idle-owned；
2. idle-owned blocks 可以被当前 block-level swap 机制处理；
3. multi-idle workload 下 swap-out 规模随 idle seq 数量线性放大。

---

## 8. RSS drop scaling

RSS drop 结果为：

```text
63.656 MiB -> 123.965 MiB -> 185.668 MiB
```

相对 1 idle seq：

```text
2 idle seq: 123.965 / 63.656 ≈ 1.95x
3 idle seq: 185.668 / 63.656 ≈ 2.92x
```

这说明 RSS 下降收益基本随 idle seq 数量近似线性增长。

`process_rss_drop_pct` 同样单调提升：

```text
0.753% -> 1.467% -> 2.197%
```

由于当前模型权重和其他 runtime buffer 占据了大量 RSS，进程总 RSS 百分比下降看起来不大；但绝对 RSS drop 从约 64 MiB 增至约 186 MiB，说明 KV idle swap/madvise 的物理内存释放效果是明确可观测的。

---

## 9. Release ratio 分析

`release_ratio_capacity_pct` 从：

```text
12.433% -> 24.212% -> 36.263%
```

单调增加，基本对应 idle seq 数量增长。

这说明：

```text
idle seq 数量增加
  -> idle-owned KV 占总 KV capacity 的比例增加
  -> 可释放 KV capacity 比例增加
  -> RSS drop 同步放大
```

`drop_over_idle_pct` 保持在约 91% 到 94%：

```text
93.612% -> 91.151% -> 91.014%
```

说明在当前 matrix 中，idle-owned KV 到实际 RSS drop 的转化效率较高，并且随 idle seq 数量增加保持稳定。

---

## 10. madvise 行为

`max_madvise_bytes_mib` 为：

```text
93.750 MiB -> 153.750 MiB -> 217.500 MiB
```

`madvise_failures` 始终为：

```text
0
```

说明：

1. 多 idle seq 场景下，madvise 调用规模随可释放 KV 增加；
2. 当前 page-level release 路径没有出现 madvise failure；
3. RSS drop 与 madvise bytes 呈稳定正相关。

`rss_drop_over_advised_pct` 为：

```text
67.900% -> 80.628% -> 85.365%
```

该比例不是严格线性指标，因为 advised bytes 与实际 RSS drop 之间受到 page residency、页对齐、邻接保护、OS 回收行为等因素影响。但整体趋势显示，多 idle seq 场景下 madvise 的实际 RSS 转化效果没有恶化。

---

## 11. Resume latency 与 prefetch 解释

`resume_first_ms` 为：

```text
166.890 ms -> 172.184 ms -> 181.211 ms
```

它没有随 idle seq 数量成倍增加。

原因是本阶段 G/H 设计中：

```text
只 resume seq0；
seq2 / seq3 只参与 idle swap/madvise，不参与本轮 resume。
```

因此：

```text
RSS drop 统计的是所有 idle seq 的释放收益；
resume_first_ms 主要反映 seq0 resume 的 first-token latency。
```

这不是矛盾，而是本阶段设计的刻意隔离：

1. Stage 7B-H 先验证多 idle seq 对 RSS release 的放大作用；
2. 不在同一实验中混入“依次恢复所有 idle seq”的总代价；
3. 后续若要评估所有 idle seq resume 的总代价，应单独设计 G2-B / H2。

此外，本阶段使用：

```text
pressure_mode=high
```

所以：

```text
target_restore_blocks=0
prefetch_blocks=0
fallback_blocks=17
```

active decode 阶段不会主动 prefetch，目的是尽可能保留 RSS 收益。延迟结果应作为 “high pressure / no-active-prefetch” 下的恢复代价，而不是最终性能优化结果。

---

## 12. tokens/s 解释

`tokens_per_second` 为：

```text
10.671657 -> 9.692740 -> 9.114787
```

随 idle seq 数量增加而下降。

原因是：

1. `NUM_IDLE_SEQS` 增加后，example driver 执行了更多 idle seq prefill/warmup；
2. 当前 tokens/s 包含 driver 总执行路径开销；
3. 该指标反映 controlled workload 的整体执行成本，不等价于真实 serving throughput；
4. 本阶段重点是 release scaling，不是最终吞吐优化。

正式表述应为：

```text
tokens/s decreases as the controlled multi-idle workload becomes larger, but this is a cost-side indicator of the current example driver path rather than a final serving throughput result.
```

---

## 13. Correctness 状态

三组实验均满足：

```text
real_abnormal_matches=0
madvise_failures=0
seq1_decoded_tokens=128
resume_decoded_tokens=128
```

但当前仍未产出 token-level equality 字段：

```text
base_vs_swap_equal
seq0_resume_equal
seq1_active_equal
```

因此本阶段 correctness 状态只能标记为：

```text
correctness_status=partial
```

含义是：

```text
当前没有发现真实异常，decode 数量正常；
但还没有完成 baseline/token-level equality 验证。
```

正式 results 或最终提交前，需要补充 baseline 对照或 token-level diff，不能直接写作 correctness 全绿。

---

## 14. 阶段性结论

Stage 7B-H 通过。

在固定 `ctx=2048`、`parallel=4`、`warmup=256`、`pressure_mode=high` 的 controlled multi-idle workload 中，`NUM_IDLE_SEQS` 从 1 增至 3 时：

```text
max_idle_owned_blocks: 17 -> 34 -> 51
max_swapped_blocks:    17 -> 34 -> 51
rss_drop_mib:          63.656 -> 123.965 -> 185.668
process_rss_drop_pct:  0.753% -> 1.467% -> 2.197%
```

这证明：

```text
多 idle request 数量增加后，
idle-owned KV 规模可以线性放大；
当前 block-level idle swap/madvise 机制可以随 idle KV 体量放大 RSS 下降收益。
```

同时，`real_abnormal_matches=0`、`madvise_failures=0`，说明 release path 在该 matrix 下运行稳定。

---

## 15. 与前序阶段的关系

### S0

S0 验证最小配置下 release path 可用：

```text
ctx=1024
parallel=2
warmup=64
max_idle_owned_blocks=8
rss_drop≈28 MiB
```

### S1

S1 放大 ctx/warmup：

```text
ctx=2048
parallel=2
warmup=256
max_idle_owned_blocks=17
rss_drop≈64 MiB
```

说明单 idle seq 的 history 增长可以放大 RSS release。

### S2

S2 将 `parallel` 提高到 4，但 RSS 没有继续放大：

```text
parallel=4
max_idle_owned_blocks=17
rss_drop≈64 MiB
```

审计发现原 driver 实际只构造了 seq0/seq1，`--parallel=4` 只扩大 capacity，不增加 idle seq。

### Stage 7B-G

新增 `LLAMA_KV_IDLE_NUM_IDLE_SEQS`，让 example driver 真正支持 multi-idle controlled workload。

### Stage 7B-H

通过 `NUM_IDLE_SEQS=1/2/3` matrix 证明：多 idle seq 数量增加后，RSS drop 近似线性放大。

---

## 16. 当前限制

本阶段仍有三个限制：

1. **correctness 仍是 partial**
   当前只确认 `real_abnormal=0` 和 decode token 数正常；还没有 baseline/token-level equality。

2. **workload 是 controlled microbenchmark**
   本阶段人为控制 idle seq 数量和 warmup 长度，用于验证机制 scaling，不应表述为真实生产 trace。

3. **当前是 high-pressure no-active-prefetch 场景**
   `pressure_mode=high` 保留 RSS 收益，但会把恢复代价留到 resume fallback；后续需要单独评估 medium/low prefetch 策略对 latency 的改善和对 RSS 的抵消。

---

## 17. 后续工作

建议下一步按以下顺序推进：

### 17.1 补 correctness equality

目标：

```text
同一模型、同一 seed、同一 prompt、同一 workload 下，
比较 baseline 与 swap/madvise 路径输出。
```

需要恢复或新增以下字段：

```text
base_vs_swap_equal=0
seq0_resume_equal=0
seq1_active_equal=0
```

正式结果前必须补这一项。

### 17.2 运行 3-run median

对 Stage 7B-H matrix 进行 3-run median：

```text
NUM_IDLE_SEQS=1
NUM_IDLE_SEQS=2
NUM_IDLE_SEQS=3
```

重点统计：

```text
rss_drop_mib
process_rss_drop_pct
max_idle_owned_blocks
max_swapped_blocks
release_ratio_capacity_pct
resume_first_ms
tokens_per_second
real_abnormal_matches
```

### 17.3 可选：all-idle resume 代价评估

当前只 resume seq0。后续可以新增实验：

```text
依次 resume seq0 / seq2 / seq3
```

用于评估：

```text
释放多个 idle seq 后，如果所有 idle seq 都恢复，总恢复代价是多少。
```

### 17.4 Stage 7C-A：KV footprint / residency audit

在完成 Stage 7B-H median 后，可以进入 Stage 7C-A，进一步回答：

```text
KV cache 在总进程 RSS 中占多少？
idle-owned KV 理论上限是多少？
madvise 后实际 RSS drop 达到理论上限的多少？
```

---

## 18. 对外汇报口径

可以简要表述为：

```text
我们在 llama.cpp 的连续 KV cache 结构上实现了 controlled multi-idle workload，用于验证 block-level idle KV swap/madvise 的可扩展性。

在 ctx=2048、parallel=4、warmup=256 的配置下，idle seq 数量从 1 增加到 3 时，idle-owned blocks 从 17 增至 51，RSS drop 从 63.7 MiB 增至 185.7 MiB，进程 RSS 下降比例从 0.75% 增至 2.20%，且 real_abnormal_matches=0。

这说明多 idle request 场景下，RSS 下降收益可以随 idle KV 体量近似线性放大。
```

需要避免的表述：

```text
不要说实现了完整 vLLM PagedAttention。
不要直接对标 vLLM throughput 2-4x。
不要把 controlled workload 说成真实生产 workload。
不要在 correctness partial 的情况下写 correctness 全绿。
```
