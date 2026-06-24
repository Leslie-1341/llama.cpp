# Stage 5E-1B：KV mincore resident matrix 结果

> 本文档整理 small / mid / large 三档 KV mincore resident telemetry 结果，仅为结果归档，不包含任何源码 / examples / CMake 修改。
> 文件路径：`docs/kv_paged_read_stage5e_mincore_resident_matrix_results.md`
> 关联文档：
> - `docs/kv_paged_read_stage5e_memory_attribution_plan.md`（5E-0 no-source attribution）
> - `docs/kv_paged_read_stage5e_mincore_resident_telemetry_plan.md`（5E-1 mincore 设计）
> - `docs/kv_paged_read_stage5c_scale_cumulative_rss_results.md`（5C-scale-B cumulative RSS）
> - `docs/kv_paged_read_stage5d_clean_perf_tradeoff_results.md`（5D-1B clean perf tradeoff）

---

## 1. 背景

Stage 5E-0 已通过 no-source memory attribution 说明：

```text
1. unified KV 下 slots_factor=1；
2. ctx=512/1024/2048/4096 的理论 KV 容量为 128/256/512/1024 MiB；
3. RSS 随 ctx 的增量与理论 KV 增量高度吻合（比值 0.98–1.01）；
4. Stage 5C/5D 的 observed process RSS drop 约为理论 KV 容量的 ~11%；
5. 但 process RSS drop 不能精确说明 KV resident pages 释放了多少。
```

Stage 5E-1-code 已实现只读 mincore telemetry：

```text
LLAMA_KV_PAGED_MINCORE=1
```

只统计 KV tensor 区间 resident pages，不改状态机，不改 graph/kernel，不改 swap 语义，复用 `paged_madvise_block` 同款地址推导与 page-align。

本阶段整理 small / mid / large mincore matrix，回答：

```text
1. process RSS drop 是否主要来自 KV resident pages？
2. KV resident drop 是否随规模放大？
3. resume swap-in 后 KV resident 是否恢复？
4. 当前释放比例到底是多少？
```

---

## 2. 实验矩阵

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
base: paged + nonidentity + mincore，不开 swap/madvise
madv: idle swap + madvise + resume swap-in + mincore
```

---

## 3. correctness 结果

三档全部为 0：

```text
base_exit=0
madv_exit=0
base_vs_madv_equal=0
seq1_equal=0
seq0_equal=0
real_abnormal=0
kv_mincore_failures=0
paged_swap_madvise_failures=0
```

结论：

```text
mincore telemetry 不破坏 correctness；madvise + resume swap-in 保持
full / seq1 active / seq0 resume 逐字节一致。
```

---

## 4. summary 表

原始 summary 表：

```text
case    ctx     n       base_exit       madv_exit       base_vs_madv_equal      seq1_equal      seq0_equal      real_abnormal   kv_mincore_total_bytes  kv_mincore_before_madvise_resident_bytes        kv_mincore_after_madvise_resident_bytes    kv_mincore_after_resume_resident_bytes  kv_mincore_madvise_drop_bytes   kv_mincore_resume_recover_bytes kv_total_mib    kv_drop_mib     kv_recover_mib  kv_drop_over_kv_total_pct       resume_recover_over_drop_pct    paged_swap_rss_total_drop_kb    paged_swap_rss_drop_sum_kb      rss_total_drop_mib      rss_drop_sum_mib        rss_total_drop_over_kv_drop_pct paged_swap_madvise_bytes        paged_swap_madvise_calls        paged_swap_madvise_failures     kv_mincore_failures     kv_mincore_sample_calls
small   512     64      0       0       0       0       0       0       133955584       133955584       118226944       133955584       15728640        15728640        127.75  15.00   15.00   11.74   100.00     13684   18816   13.36   18.38   89.09   19660800        5       0       0       133
mid     1024    128     0       0       0       0       0       0       268173312       268173312       236716032       268173312       31457280        31457280        255.75  30.00   30.00   11.73   100.00     29164   34296   28.48   33.49   94.93   35389440        9       0       0       261
large   2048    256     0       0       0       0       0       0       536608768       536608768       473694208       536608768       62914560        62914560        511.75  60.00   60.00   11.72   100.00     59168   65120   57.78   63.59   96.30   66846720        17      0       0       517
```

---

## 5. 结果解释

### 5.1 KV resident drop 随规模放大

```text
small: 15 MiB
mid:   30 MiB
large: 60 MiB
```

结论：

```text
KV resident drop 随 ctx/n 放大近似线性增长。
```

### 5.2 resume recover = 100%

```text
small: 15 MiB recover / 15 MiB drop = 100%
mid:   30 MiB recover / 30 MiB drop = 100%
large: 60 MiB recover / 60 MiB drop = 100%
```

解释：

```text
madvise 释放的 KV resident pages 在 resume swap-in 后完整恢复。
```

### 5.3 process RSS drop 与 KV resident drop 同量级

```text
small: RSS total drop 13.36 MiB vs KV resident drop 15 MiB
mid:   RSS total drop 28.48 MiB vs KV resident drop 30 MiB
large: RSS total drop 57.78 MiB vs KV resident drop 60 MiB
```

结论：

```text
Stage 5C/5D 中观测到的 process current RSS drop 主要来自 KV tensor
resident pages 被 madvise 释放。
```

口径说明（不要写成“完全相等”）：

```text
process RSS 是全进程口径（含权重/compute/allocator/其它），mincore 是 KV
tensor 区间口径；二者同量级但不应要求严格相等。rss_total_drop_over_kv_drop_pct
随规模从 89.09% 上升到 96.30%，反映规模越大、KV resident 释放在进程 RSS
drop 中的占比越主导。
```

### 5.4 当前释放比例仍偏低

`kv_drop_over_kv_total_pct`：

```text
small 11.74%
mid   11.73%
large 11.72%
```

解释：

```text
当前策略只释放了 KV resident 的约 12%，还没有达到明显收益所需的 30% 以上目标。
```

这不是 failure，而是后续优化方向：

```text
后续需要扩大 cold/releasable KV 覆盖范围，减少只能释放少量 idle block 的限制。
```

---

## 6. 与 Stage 5D 性能结果结合

引用 Stage 5D-1B 结论（large case）：

```text
KV resident drop          = 60 MiB
process RSS drop          = 57.78 MiB
resume first-token latency 增加约 42 ms
```

结论：

```text
当前方案证明了 idle KV swap + madvise 可以真实释放 KV resident pages，并在
resume 时恢复；但释放比例只有约 12%，同时存在 resume 首 token latency 代价。
因此当前阶段更适合作为机制验证和 tradeoff 基线，不应包装为最终高收益优化。
```

---

## 7. 边界说明

```text
1. mincore 只统计 KV tensor 区间，不统计 model weights / compute / allocator；
2. process RSS drop 与 KV mincore drop 口径不同，不要求严格相等；
3. 当前结果只覆盖 Linux + CPU backend；
4. mincore telemetry 有 syscall 开销，不用于默认性能矩阵；
5. 当前释放比例约 12%，后续优化重点是提升可释放 KV resident 覆盖比例，
   而不是继续证明 madvise 是否有效。
```

---

## 8. 一句话结论

```text
Stage 5E-1B 通过 mincore 直接确认：madvise 后 KV tensor resident pages 随规模
下降 15/30/60 MiB，并在 resume swap-in 后 100% 恢复；process RSS drop 与 KV
resident drop 同量级，说明前面观测到的 RSS 下降主要来自 KV resident pages
释放。但当前释放比例稳定约 12%，收益仍偏低。
```
