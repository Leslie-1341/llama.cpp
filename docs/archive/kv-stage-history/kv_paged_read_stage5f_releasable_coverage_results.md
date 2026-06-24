# Stage 5F-1：releasable coverage matrix 结果

> 本文档整理 small / mid / large 三档 releasable coverage telemetry 结果，仅为结果归档，不包含任何源码 / examples / CMake 修改。
> 文件路径：`docs/kv_paged_read_stage5f_releasable_coverage_results.md`
> 关联文档：`docs/kv_paged_read_stage5f_releasable_coverage_plan.md`（Stage 5F-0 coverage funnel 计划）

---

## 1. 背景

Stage 5E-1B 已确认 idle swap + madvise 链路能真实释放 KV resident pages，但释放比例稳定在约 12%：

```text
small: KV resident drop = 15 MiB, release ratio = 11.74%
mid:   KV resident drop = 30 MiB, release ratio = 11.73%
large: KV resident drop = 60 MiB, release ratio = 11.72%

resume recover = 100%
process RSS drop 与 KV resident drop 同量级
correctness 全 0
```

Stage 5F-0 提出需要把约 12% 的释放比例拆成 coverage funnel，判断瓶颈到底来自 idle coverage 少，还是 read-window / resident-safe / remap / madvise gate 损耗。

Stage 5F-1 已实现 coverage telemetry，并完成 small / mid / large matrix。本阶段只整理结果，不修改释放策略。

---

## 2. 本阶段目标

本阶段回答：

```text
1. 当前 12% release ratio 的主要瓶颈在哪一层？
2. idle-owned KV 覆盖率是多少？
3. read-window 是否吃掉大量 idle KV？
4. resident-safe / nonidentity-remap gate 是否造成明显损耗？
5. 后续应该优先改 workload / idle coverage，还是改 madvise 后段策略？
```

---

## 3. 实验矩阵

三档：

```text
small: ctx=512,  n=64
mid:   ctx=1024, n=128
large: ctx=2048, n=256
```

三档均使用：

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

三档全部为 0：

```text
base_exit=0
madv_exit=0
base_vs_madv_equal=0
seq1_equal=0
seq0_equal=0
real_abnormal=0
```

结论：

```text
coverage telemetry 不破坏 correctness；madvise + resume swap-in 仍保持
full / seq1 active / seq0 resume 逐字节一致。
```

---

## 5. Summary 表

| case | ctx | n | base_exit | madv_exit | base_vs_madv_equal | seq1_equal | seq0_equal | real_abnormal | idle_owned_blocks | in_read_window_blocks | not_in_read_window_blocks | resident_safe_blocks | nonidentity_remapped_blocks | idle_owned_bytes | in_read_window_bytes | resident_safe_bytes | nonidentity_remapped_bytes | kv_total_bytes | kv_drop_bytes | madvise_bytes | idle_coverage_pct | read_window_block_pct | safe_candidate_pct | remap_gate_pct | madvise_to_idle_pct | resident_release_efficiency_pct | release_ratio_total_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 512 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 4 | 4 | 4 | 16777216 | 0 | 16777216 | 16777216 | 133955584 | 15728640 | 19660800 | 12.52 | 0.00 | 100.00 | 100.00 | 117.19 | 80.00 | 11.74 |
| mid | 1024 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 8 | 8 | 8 | 33554432 | 0 | 33554432 | 33554432 | 268173312 | 31457280 | 35389440 | 12.51 | 0.00 | 100.00 | 100.00 | 105.47 | 88.89 | 11.73 |
| large | 2048 | 256 | 0 | 0 | 0 | 0 | 0 | 0 | 16 | 0 | 16 | 16 | 16 | 67108864 | 0 | 67108864 | 67108864 | 536608768 | 62914560 | 66846720 | 12.51 | 0.00 | 100.00 | 100.00 | 99.61 | 94.12 | 11.72 |

---

## 6. 结果解释

### 6.1 主瓶颈：idle-owned coverage 只有约 12.5%

```text
small: idle_coverage_pct = 12.52%, release_ratio_total = 11.74%
mid:   idle_coverage_pct = 12.51%, release_ratio_total = 11.73%
large: idle_coverage_pct = 12.51%, release_ratio_total = 11.72%
```

当前释放比例约 12%，主要由 idle-owned KV 覆盖率决定。三档中进入该轮释放漏斗的 idle-owned KV 只占总 KV resident 的约 12.5%，而最终释放比例约 11.7%，说明当前瓶颈首先是 workload 中可释放 idle KV 覆盖范围太小。

### 6.2 read-window 不是瓶颈

```text
read_window_block_pct = 0.00%
in_read_window_bytes = 0
```

在发生 madvise 的这一轮中，idle-owned block 没有被 read-window 保护吃掉。

### 6.3 resident-safe / remap gate 不是瓶颈

```text
safe_candidate_pct = 100.00%
remap_gate_pct = 100.00%
```

idle-owned block 全部成为 resident-safe candidate，并全部通过 nonidentity-remapped gate。

所以当前不应优先修改：

```text
read-window gate
RESIDENT state gate
nonidentity remap gate
```

### 6.4 后段 madvise 转化不是主瓶颈

```text
KV drop / idle-owned:
small = 15 / 16 = 93.75%
mid   = 30 / 32 = 93.75%
large = 60 / 64 = 93.75%
```

进入 idle-owned 漏斗的 KV 中，大部分最终体现为 KV resident drop。剩余损耗主要来自 page alignment / tensor 边界 / resident 转化差异，但不是当前 12% release ratio 的主要来源。

### 6.5 `madvise_to_idle_pct` 口径说明

`madvise_to_idle_pct` 在 small / mid 中超过 100%，不能解释为“释放超过 idle-owned”。原因是 `paged_cov_idle_owned_bytes` 是 latch 到发生 madvise 的某一轮 coverage 字节，而 `paged_swap_madvise_bytes` 是全程累计字段，二者不是严格同一窗口口径。

因此本文不把 `madvise_to_idle_pct` 作为主要结论。本阶段主要使用 `idle_coverage_pct`、`safe_candidate_pct`、`remap_gate_pct`、`release_ratio_total_pct` 以及 KV drop / idle-owned 进行判断。若后续需要更精确的同窗口后段效率，可新增 `paged_cov_madvise_bytes` 或 `paged_cov_swap_bytes_out`。

---

## 7. 总结结论

Stage 5F-1 coverage matrix 表明，当前约 12% 的 release ratio 主要由 idle-owned coverage 决定。small/mid/large 三档中，idle-owned KV 仅占总 KV resident 的约 12.5%，而 read-window、resident-safe 和 nonidentity-remap gate 均未造成额外损失；进入 idle-owned 漏斗的 KV 中约 93.75% 最终表现为 KV resident drop。因此当前瓶颈不是 madvise 机制失效，也不是 remap/read-window gate 过严，而是当前 workload 下可释放 idle KV 覆盖范围太小。

---

## 8. 后续方向

如果目标是把释放比例提高到 30% 以上，优先方向不是放宽 read-window / remap / madvise 后段 gate，而是扩大 idle-owned KV coverage。

可能方向：

```text
1. 更多 idle seq；
2. 更长 idle history；
3. 更大比例的 cold / inactive KV；
4. 更贴近多会话暂停-恢复场景的 workload；
5. 后续再考虑 prefetch / async swap-in 降低 resume first-token latency。
```

边界：

```text
本阶段不修改释放策略，只定位瓶颈；是否扩大 idle-owned coverage
需要后续设计 workload 或策略后再验证 correctness 和 latency。
```

---

## 9. 文档结论

Stage 5F-1 表明，当前约 12% 的 KV release ratio 主要受限于 idle-owned KV coverage：发生 madvise 的这一轮中，idle-owned KV 只占总 KV resident 的约 12.5%，而 read-window、resident-safe、nonidentity-remap 和 madvise 后段均不是主要瓶颈。
