# Stage 5G-1：high-idle-coverage workload 结果

> 本文档整理 Stage 5G-1 high-idle-coverage workload no-code sweep 结果，仅为结果归档，不包含任何源码 / examples / CMake 修改。
> 文件路径：`docs/kv_paged_read_stage5g_high_idle_coverage_results.md`
> 关联文档：
> - `docs/kv_paged_read_stage5g_high_idle_coverage_plan.md`（Stage 5G-0 high-idle-coverage workload 计划）
> - `docs/kv_paged_read_stage5f_releasable_coverage_results.md`（Stage 5F-1 coverage matrix 结果）

---

## 1. 背景

Stage 5F-1 coverage matrix 已定位当前约 12% release ratio 的主瓶颈：

```text
small/mid/large 的 idle-owned coverage ≈ 12.5%
最终 release ratio ≈ 11.7%
read-window = 0%
resident-safe gate = 100%
nonidentity-remap gate = 100%
KV drop / idle-owned ≈ 93.75%
```

结论：

```text
当前瓶颈不是 madvise 机制失效，也不是 read-window / remap gate 过严，
而是当前 workload 中可释放 idle-owned KV 覆盖范围太小。
```

Stage 5G-0 提出路线 A：

```text
通过 high-idle-coverage workload 提高 idle-owned KV coverage；
观察 release ratio 是否随 idle-owned coverage 同步提高；
验证当前释放机制的收益上限是否主要由 workload 中 idle KV 占比决定。
```

---

## 2. 本阶段目标

整理 Stage 5G-1 no-code sweep 结果，回答：

```text
1. 增大 n/ctx 是否能提高 idle-owned coverage？
2. release ratio 是否随 idle-owned coverage 同步提高？
3. read-window / resident-safe / remap gate 是否仍不是瓶颈？
4. high-idle-coverage workload 下 release ratio 是否能达到 30%+？
5. 本轮是否能评价 latency / perf tradeoff？
```

---

## 3. 实验设置

本轮不改源码，只使用现有 binary 和 Stage 5F-1 coverage telemetry。

实验组：

```text
small_12p: ctx=512,  n=64    n/ctx=0.1250
small_25p: ctx=512,  n=128   n/ctx=0.2500
small_31p: ctx=512,  n=160   n/ctx=0.3125
small_37p: ctx=512,  n=192   n/ctx=0.3750

mid_12p:   ctx=1024, n=128   n/ctx=0.1250
mid_25p:   ctx=1024, n=256   n/ctx=0.2500
mid_31p:   ctx=1024, n=320   n/ctx=0.3125
mid_37p:   ctx=1024, n=384   n/ctx=0.3750
```

共同参数：

```text
--batch-size 128
--ubatch-size 128
--seed 1
--temp 0
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 2
```

对比两组：

```text
base: paged + nonidentity + mincore + coverage，不开 swap/madvise
madv: idle swap + madvise + resume swap-in + mincore + coverage
```

---

## 4. Correctness 结果

所有 case 均满足：

```text
base_exit=0
madv_exit=0
base_vs_madv_equal=0
seq1_equal=0
seq0_equal=0
```

`real_abnormal` 原始 grep 中出现的命中均为正常 warning：

```text
W llama_context: n_ctx_seq (...) < n_ctx_train (8192) -- the full capacity of the model will not be utilized
```

这些 warning 只表示当前 ctx-size 小于模型训练上下文长度，是 small / mid ctx 测试下的预期日志，不表示 correctness 或 runtime failure。因此本轮真实异常视为 0。

---

## 5. Summary 表

| case | ctx | n | n_over_ctx | base_exit | madv_exit | base_vs_madv_equal | seq1_equal | seq0_equal | real_abnormal | idle_owned_blocks | in_read_window_blocks | not_in_read_window_blocks | resident_safe_blocks | nonidentity_remapped_blocks | idle_owned_bytes | in_read_window_bytes | resident_safe_bytes | nonidentity_remapped_bytes | kv_total_bytes | kv_drop_bytes | madvise_bytes | idle_coverage_pct | read_window_block_pct | safe_candidate_pct | remap_gate_pct | drop_over_idle_pct | release_ratio_total_pct | total_wall_ms | tokens_per_second |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small_12p | 512 | 64 | 0.1250 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 4 | 0 | 4 | 4 | 4 | 16777216 | 0 | 16777216 | 16777216 | 133955584 | 15728640 | 19660800 | 12.52 | 0.00 | 100.00 | 100.00 | 93.75 | 11.74 | 12503.798 | 10.631810 |
| small_25p | 512 | 128 | 0.2500 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 8 | 0 | 8 | 8 | 8 | 33554432 | 0 | 33554432 | 33554432 | 133955584 | 31457280 | 35389440 | 25.05 | 0.00 | 100.00 | 100.00 | 93.75 | 23.48 | 27262.709 | 9.553242 |
| small_31p | 512 | 160 | 0.3125 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 10 | 0 | 10 | 10 | 10 | 41943040 | 0 | 41943040 | 41943040 | 133955584 | 39321600 | 43253760 | 31.31 | 0.00 | 100.00 | 100.00 | 93.75 | 29.35 | 35587.293 | 9.113982 |
| small_37p | 512 | 192 | 0.3750 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 12 | 0 | 12 | 12 | 12 | 50331648 | 0 | 50331648 | 50331648 | 133955584 | 47185920 | 51118080 | 37.57 | 0.00 | 100.00 | 100.00 | 93.75 | 35.23 | 42801.790 | 9.073293 |
| mid_12p | 1024 | 128 | 0.1250 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 8 | 0 | 8 | 8 | 8 | 33554432 | 0 | 33554432 | 33554432 | 268173312 | 31457280 | 35389440 | 12.51 | 0.00 | 100.00 | 100.00 | 93.75 | 11.73 | 25380.196 | 10.278653 |
| mid_25p | 1024 | 256 | 0.2500 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 16 | 0 | 16 | 16 | 16 | 67108864 | 0 | 67108864 | 67108864 | 268173312 | 62914560 | 66846720 | 25.02 | 0.00 | 100.00 | 100.00 | 93.75 | 23.46 | 51788.164 | 9.977767 |
| mid_31p | 1024 | 320 | 0.3125 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 20 | 0 | 20 | 20 | 20 | 83886080 | 0 | 83886080 | 83886080 | 268173312 | 78643200 | 82575360 | 31.28 | 0.00 | 100.00 | 100.00 | 93.75 | 29.33 | 74509.784 | 8.643940 |
| mid_37p | 1024 | 384 | 0.3750 | 0 | 0 | 0 | 0 | 0 | benign_ctx_warning | 24 | 0 | 24 | 24 | 24 | 100663296 | 0 | 100663296 | 100663296 | 268173312 | 94371840 | 98304000 | 37.54 | 0.00 | 100.00 | 100.00 | 93.75 | 35.19 | 89520.716 | 8.625040 |

注意：`resume_first_ms` 本轮未成功提取，不能给出 resume first-token latency 结论。

---

## 6. 结果解释

### 6.1 idle coverage 随 n/ctx 线性提高

small：

```text
n/ctx=0.1250 -> idle_coverage_pct=12.52%
n/ctx=0.2500 -> idle_coverage_pct=25.05%
n/ctx=0.3125 -> idle_coverage_pct=31.31%
n/ctx=0.3750 -> idle_coverage_pct=37.57%
```

mid：

```text
n/ctx=0.1250 -> idle_coverage_pct=12.51%
n/ctx=0.2500 -> idle_coverage_pct=25.02%
n/ctx=0.3125 -> idle_coverage_pct=31.28%
n/ctx=0.3750 -> idle_coverage_pct=37.54%
```

idle-owned coverage 与 n/ctx 呈稳定对应关系，说明 high-idle-coverage workload 能有效放大可释放 idle KV 覆盖范围。

### 6.2 release ratio 随 idle coverage 同步提高

small：

```text
idle_coverage=12.52% -> release_ratio=11.74%
idle_coverage=25.05% -> release_ratio=23.48%
idle_coverage=31.31% -> release_ratio=29.35%
idle_coverage=37.57% -> release_ratio=35.23%
```

mid：

```text
idle_coverage=12.51% -> release_ratio=11.73%
idle_coverage=25.02% -> release_ratio=23.46%
idle_coverage=31.28% -> release_ratio=29.33%
idle_coverage=37.54% -> release_ratio=35.19%
```

```text
release_ratio_total_pct ≈ idle_coverage_pct × 93.75%
```

当 idle coverage 从约 12.5% 提高到约 37.5% 时，release ratio 从约 11.7% 提高到约 35.2%。这说明当前机制在 high-idle-coverage workload 下可以达到 30%+ KV resident release ratio。

### 6.3 read-window / safe / remap gate 仍不是瓶颈

所有 case：

```text
read_window_block_pct = 0.00%
safe_candidate_pct = 100.00%
remap_gate_pct = 100.00%
drop_over_idle_pct = 93.75%
```

提高 n/ctx 后，idle-owned blocks 仍未被 read-window 吃掉，全部成为 resident-safe candidate，并全部通过 nonidentity-remap gate。后段 KV drop / idle-owned 稳定为 93.75%，说明释放转换效率稳定。

因此当前不应优先修改：

```text
read-window gate
RESIDENT state gate
nonidentity remap gate
madvise 后段机制
```

### 6.4 memory scaling 成立，但 latency 未闭环

```text
本轮验证的是 memory release scaling，不是完整 perf tradeoff。
```

原因：

```text
resume_first_ms 本轮未提取成功；
total_wall_ms 和 tokens_per_second 随 n 增大自然变化，不能直接作为 swap/madvise 代价结论；
后续需要单独跑 perf matrix，比对相同 n/ctx 下 base / nomadv / madv 的 resume first-token latency 和 total wall time。
```

---

## 7. 总结结论

Stage 5G-1 no-code sweep 证明路线 A 成立：只要 workload 中 idle-owned KV coverage 提高，KV release ratio 就会同步提高。small/mid 两档中，n/ctx 从 0.125 提高到 0.375 时，idle_coverage_pct 从约 12.5% 提高到约 37.5%，release_ratio_total_pct 从约 11.7% 提高到约 35.2%。read-window、resident-safe、nonidentity-remap gate 仍不是瓶颈。因此此前约 12% 的低释放比例主要由 workload 中 idle KV 覆盖不足造成，而不是释放机制失效。

---

## 8. 后续方向

```text
1. 补 Stage 5G perf matrix，重点比较 12.5% / 31.25% / 37.5% high-idle coverage 下的 resume first-token latency；
2. 如时间允许，扩展 large_25p / large_31p / large_37p；
3. 后续可设计更贴近真实多会话暂停-恢复场景的 high-idle workload；
4. 若要进一步优化体验，应研究 prefetch / async swap-in 来降低 resume 首字延迟。
```

---

## 9. 结论

Stage 5G-1 证明，在 high-idle-coverage workload 下，idle KV swap + madvise 的 release ratio 可随 idle-owned coverage 从约 12% 提升到 30%+；当前机制的收益上限主要由 workload 中 idle KV 占比决定，而不是由 read-window、remap 或 madvise 后段 gate 限制。
