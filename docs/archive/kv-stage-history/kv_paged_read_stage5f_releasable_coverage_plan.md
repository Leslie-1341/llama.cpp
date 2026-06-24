# Stage 5F: Releasable Coverage Analysis Plan

## 1. Stage 5F-0 目标

Stage 5E-1B 已确认 idle swap + madvise 链路能真实释放 KV resident pages，但释放比例稳定在约 12%：

```text
small: KV resident drop = 15 MiB, release ratio = 11.74%
mid:   KV resident drop = 30 MiB, release ratio = 11.73%
large: KV resident drop = 60 MiB, release ratio = 11.72%

resume recover = 100%
process RSS drop 与 KV resident drop 同量级
correctness 全 0
```

当前问题：

- madvise 确实释放了 KV resident pages；
- 但释放比例只有约 12%，低于明显收益所需的约 30% 目标。

Stage 5F-0 的目标是：

- 解释为什么当前只能释放约 12%；
- 判断瓶颈是 idle KV 覆盖少，还是 safe / swap / madvise gate 太保守；
- 在定位瓶颈前，不修改释放策略。

本阶段（5F-0）只做源码勘察、指标设计与文档整理，不改源码。

## 2. 为什么当前 12% release ratio 需要拆解

总释放比例当前由单一比值描述：

```text
release_ratio_total = kv_mincore_madvise_drop_bytes / kv_mincore_total_bytes ≈ 12%
```

这个比值是整条链路的端到端结果，无法单独区分以下两类截然不同的原因：

1. **覆盖不足**：workload 中真正可释放的 idle KV 本身就少，或被 read-window / state gate 大量保护，进入 swap-out 的候选就不多；
2. **gate 过保守**：候选其实不少，但被 nonidentity remap gate、swap-out 后端、madvise page-align / neighbor 等逐层削减。

三个 case（small/mid/large）释放比例几乎相同（11.72%–11.74%），说明瓶颈是**结构性的、与规模无关的比例约束**，而不是某个绝对字节上限。因此需要把端到端比例拆成漏斗，看损失主要发生在哪一层。

## 3. 当前 idle swap / madvise 链路源码现状

链路核心在 `src/llama-kv-cache.cpp` 的 idle-trace 块（约 3819–4156 行），辅以 `paged_swap_out_block`（约 1872 行）与 `paged_madvise_block`（约 2154 行）。

候选集合的包含关系：

```text
madvise candidate ⊆ swap-out candidate ⊆ safe idle candidate
```

当前 block 进入 swap-out 的主要条件：

1. block owner 非空；
2. 非 mixed seq（单 seq 占用）；
3. owner 只属于 idle seq，不含 active seq；
4. 不在当前 read window（`trace_read_blocks` 未命中）；
5. block state == RESIDENT；
6. paged_idle_swap_requested 为真，且 idle_swap_ready 成立（requires swap_enabled + backing store + nonidentity_probe + idle_trace）；
7. block 已在 nonidentity_remapped_blocks 内。

madvise 不单独选块：在 swap-out 成功（block state 置为 SWAPPED）之后，如果 `idle_swap_madvise_ready` 为 true，`paged_swap_out_block` 才调用 `paged_madvise_block` 对该 block 的 K/V 张量区间执行 `MADV_DONTNEED`。`paged_madvise_block` 内部做 page-align 内缩（round-up start / round-down end）并对未对齐边界的相邻保护块计数（skip_no_full_page / skip_neighbor）。

## 4. coverage funnel 指标设计

把端到端比例拆成下列漏斗。每层标注对应源码判断或已有字段。bytes 口径使用 per-block 字节单位（每 cell 的 K/V 各 layer `nb[1]` 之和 × `paged_block_size`，与 `paged_swap_out_block` 内 `total_size` 推导一致）。

| # | 层级 | 来源 |
|---|------|------|
| 0 | total KV bytes | `kv_mincore_total_bytes`（字节分母）；total blocks 用 `paged_n_blocks` |
| 1 | resident blocks | `paged_block_states[block] == RESIDENT` |
| 2 | non-empty owner blocks | `owner.none()` 反向；已有 `paged_idle_non_empty_blocks` |
| 3 | single-seq blocks | `owner_count == 1`；已有 `paged_idle_single_seq_blocks` |
| 4 | idle-owned blocks | `!mixed && only_seen_idle_seq`（局部变量 `cold_candidates`，需提升为字段） |
| 5 | in-read-window blocks | `trace_read_blocks` 命中（局部变量 `cold_in_read_window`） |
| 6 | not-in-read-window blocks | 局部变量 `cold_not_in_read_window` |
| 7 | resident safe candidates | #6 ∩ RESIDENT（局部变量 `safe_swap_candidates`） |
| 8 | nonidentity-remapped candidates | `nonidentity_remapped_blocks` 命中（= #7 − `paged_idle_swap_skip_not_remapped`） |
| 9 | swap-out attempted blocks | 调 `paged_swap_out_block`；= `paged_idle_swap_candidates` |
| 10 | swap-out success blocks / bytes | `paged_swap_out_calls` / `paged_blocks_swapped_out`；bytes = `paged_swap_bytes_out` |
| 11 | madvise attempted / advised bytes | `paged_swap_madvise_calls` / `paged_swap_madvise_bytes` |
| 12 | actual KV resident drop bytes | `kv_mincore_madvise_drop_bytes`（before−after madvise） |

漏斗 #4–#8 当前只在 idle-trace 一步内作为局部变量算出并 printf，没有可累计的成员字段，也没有 bytes 口径——这是主要缺口。

## 5. 可复用字段

后段（swap / madvise / mincore）已有字段较完整，可直接复用：

```text
paged_idle_non_empty_blocks
paged_idle_single_seq_blocks
paged_idle_swap_candidates
paged_idle_swap_out_calls
paged_idle_swap_skip_not_remapped
paged_idle_swap_skip_not_resident

paged_swap_out_calls
paged_blocks_swapped_out
paged_swap_bytes_out

paged_swap_madvise_calls
paged_swap_madvise_bytes
paged_swap_madvise_skip_no_full_page
paged_swap_madvise_skip_neighbor

kv_mincore_total_bytes
kv_mincore_madvise_drop_bytes
```

说明：

- 后段 swap / madvise / mincore 已有字段较完整；
- 缺口主要在 funnel 中段：idle-owned、read-window、resident-safe、nonidentity-remapped 的**累计字段**和 **bytes 口径**目前都缺失（仅有局部 block 计数）。

## 6. 建议新增字段

只补 funnel 中段的累计计数与 bytes 口径，最小必要：

```text
paged_cov_idle_owned_blocks
paged_cov_in_read_window_blocks
paged_cov_not_in_read_window_blocks
paged_cov_resident_safe_blocks
paged_cov_nonidentity_remapped_blocks

paged_cov_idle_owned_bytes
paged_cov_in_read_window_bytes
paged_cov_resident_safe_bytes
paged_cov_nonidentity_remapped_bytes
```

不新增（可由已有字段替代）：

- total blocks 用 `paged_n_blocks`；
- nonempty / single-seq 复用 `paged_idle_non_empty_blocks` / `paged_idle_single_seq_blocks`；
- swap / madvise / mincore 后段字段已存在；
- page alignment 损耗先通过 `paged_swap_madvise_bytes / paged_swap_bytes_out` 间接观察；当前 `advised_bytes` 已等于 `paged_swap_madvise_bytes`（neighbor 只计数不缩区间），故不单独新增 page-aligned advisable bytes，后续证据不足再拆 raw candidate bytes 与 aligned advisable bytes。

bytes 单位建议在 idle-trace 入口算一次 per-block 字节数，各 bytes 字段 = 对应 block 计数 × 该单位，避免逐块重复求和。

## 7. 派生比例设计

```text
release_ratio_total =
  kv_mincore_madvise_drop_bytes / kv_mincore_total_bytes

idle_coverage_ratio =
  paged_cov_idle_owned_bytes / kv_mincore_total_bytes

read_window_block_ratio =
  paged_cov_in_read_window_bytes / paged_cov_idle_owned_bytes

safe_candidate_ratio =
  paged_cov_resident_safe_bytes / paged_cov_idle_owned_bytes

remap_gate_ratio =
  paged_cov_nonidentity_remapped_bytes / paged_cov_resident_safe_bytes

swap_success_ratio =
  paged_swap_bytes_out / paged_cov_nonidentity_remapped_bytes

madvise_advice_ratio =
  paged_swap_madvise_bytes / paged_swap_bytes_out

resident_release_efficiency =
  kv_mincore_madvise_drop_bytes / paged_swap_madvise_bytes
```

每个比例定位的瓶颈：

- `idle_coverage_ratio` 低：workload 中 idle KV 本身少（覆盖不足，非 gate 问题）；
- `read_window_block_ratio` 高：read window 保护吃掉大量 idle KV；
- `safe_candidate_ratio` 低：read-window / state gate 综合损耗大；
- `remap_gate_ratio` 低：nonidentity remap gate（条件 7）是瓶颈；
- `swap_success_ratio` 低：swap-out 后端失败或未成功；
- `madvise_advice_ratio` 低：swap 后真正可 advise 字节少（page-align / neighbor 损耗）；
- `resident_release_efficiency` 低：advise 后实际 resident drop 转化低。

链式乘积应近似等于 `release_ratio_total`；哪一段比值最小，即主要损失所在层。

## 8. 可能瓶颈分析

以下为假设，需 5F-1 telemetry 定量验证，本阶段不下最终结论：

1. 更可能的瓶颈在 candidate funnel 前段，即 idle-owned / read-window / resident-safe / remap gate；
2. page-align / neighbor 损耗需要统计，但从 5E-1B 看（large case madvise_bytes ≈ 63.75 MiB 与 drop 60 MiB 同量级）可能不是最大瓶颈；
3. resume recover = 100% 不是释放比例低的原因，它影响释放持续时间和 resume first-token latency；
4. 后续必须用 5F-1 telemetry 定量判断哪一层损失最大。

## 9. Stage 5F-1 最小 telemetry 实现建议

Stage 5F-1 只加 coverage telemetry，不修改 swap / madvise 策略。

允许修改文件：

```text
src/llama-kv-cache.h
src/llama-kv-cache.cpp
```

禁止：

```text
修改 swap/madvise 策略
修改 correctness 语义
修改 graph/kernel
修改 stdout marker
修改 driver
```

实现要点：

- 将 idle-trace 内的局部变量（`cold_candidates` / `cold_in_read_window` / `cold_not_in_read_window` / `safe_swap_candidates` 及 remap 命中计数）提升为 `paged_cov_*` 累计成员字段；
- 为上述各层新增 bytes 口径字段（block 计数 × per-block 字节单位）；
- 在已有 idle-trace / `paged_log_stats` 输出行追加这些字段，沿用现有 telemetry 行格式，不新增 stdout marker。

## 10. 结论一句话

Stage 5F 应先把当前约 12% 的释放比例拆解为 idle 覆盖、read-window 保护、resident-safe 候选、nonidentity remap gate、swap/madvise 成功率和实际 resident drop 等层级，找出主要损失发生在哪一层，再决定是否修改释放策略。

## 11. 后续优化方向

- 若 `idle_coverage_ratio` / `safe_candidate_ratio` 偏低：考虑放宽 read-window 或 state gate（需评估 correctness 与 resume 延迟）；
- 若 `remap_gate_ratio` 偏低：评估 nonidentity remap gate 是否过严；
- 若 `madvise_advice_ratio` / `resident_release_efficiency` 偏低：再投入 page-align / neighbor 区间优化；
- resume recover 高带来的首字延迟，作为独立维度在后续阶段单独优化。
