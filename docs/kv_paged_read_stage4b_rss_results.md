# Stage 4B-RSS — swapped-block madvise（current RSS 回收）结果

前置：

- [docs/kv_paged_read_stage4b_rss_plan.md](kv_paged_read_stage4b_rss_plan.md)
- [docs/kv_paged_read_stage4b_block_swap_results.md](kv_paged_read_stage4b_block_swap_results.md)
- [docs/kv_paged_read_stage4a_block_release_results.md](kv_paged_read_stage4a_block_release_results.md)

本文是 Stage 4B-RSS 的**结果文档**。Stage 4B-RSS 在 Stage 4B block-aware swap correctness 闭环之上，对 swap-out 写盘成功后的 `SWAPPED` physical block 执行 page-aligned `MADV_DONTNEED`，尝试回收历史 KV block 的物理页；读/写路径遇到 `SWAPPED` block 时先整块 swap-in 再继续。

结论先行：**swap+madvise correctness 闭环已打通，且观测到 current RSS 下降** —— 输出与 baseline 逐位一致，swap-out/swap-in 数量与字节完全闭合，madvise 真实发生并采到 RSS drop；同时 Stage 4A release 路径在抽取通用 `paged_madvise_block()` 后仍逐位通过，无回归。

---

## 0. 实验设置

- 单序列（`n_stream==1`）、`!v_trans`、贪心解码、单轮、`ctx=512`。
- swap 路径四方对比：
  1. baseline：`LLAMA_KV_PAGED=0`；
  2. no-swap：`LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1`，swap 关；
  3. swap+madvise：`LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1 LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_SWAP_MADVISE=1`（release 强制关闭）。
- Stage 4A release 回归单独一跑：`LLAMA_KV_PAGED=1` + release 开（swap 关），验证通用 `paged_madvise_block()` 抽取后旧路径不变。
- physical block swap，block 内逐 cell 复用 exact swap 的 `write_cell`/`read_cell`；swap-out 后对 `SWAPPED` 块叠加 page-aligned madvise。

---

## 1. swap+madvise correctness 与 swap 闭合结果

| 字段 | 值 |
|---|---|
| `base_vs_noswap_equal` | `0`（逐位一致） |
| `base_vs_swap_rss_equal` | `0`（逐位一致） |
| sha256（三份完全一致） | `5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006` |
| `paged_release_violation` | `0` |
| `paged_swap_out_calls` | `116` |
| `paged_swap_in_calls` | `116` |
| `paged_blocks_swapped_out` | `116` |
| `paged_blocks_swapped_in` | `116` |
| `paged_swap_bytes_out` | `486539264` |
| `paged_swap_bytes_in` | `486539264` |
| `paged_swap_backend_failures` | `0` |

## 2. 读路径 / 写路径 SWAPPED 处理结果

| 字段 | 值 |
|---|---|
| `paged_swap_read_swapped_hits` | `66` |
| `paged_swap_read_swap_in_calls` | `66` |
| `paged_swap_read_swap_in_failures` | `0` |
| `paged_swap_write_swapped_hits` | `50` |
| `paged_swap_write_swap_in_calls` | `50` |
| `paged_swap_write_swap_in_failures` | `0` |
| `paged_swap_in_fail_no_offset` | `0` |
| `paged_swap_in_fail_bad_size` | `0` |
| `paged_swap_in_fail_read_cell` | `0` |
| `paged_swap_in_fail_tensor_set` | `0` |

读路径命中 SWAPPED 块 66 次、全部成功 swap-in；写路径命中 50 次、全部成功 swap-in；两路合计 `66+50=116`，与 `paged_swap_in_calls=116` 完全吻合。四类 swap-in 失败计数器全 0。

## 3. madvise 与 current RSS 结果

| 字段 | 值 |
|---|---|
| `paged_swap_madvise_calls` | `116` |
| `paged_swap_madvise_bytes` | `456130560` |
| `paged_swap_madvise_failures` | `0` |
| `paged_swap_rss_samples` | `116` |
| `paged_swap_rss_drop_last_kb` | `3840` |
| `paged_swap_rss_drop_max_kb` | `3840` |

## 4. Stage 4A release 回归结果

| 字段 | 值 |
|---|---|
| `base_vs_release_equal` | `0`（逐位一致） |
| sha256 | `5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006` |
| `paged_block_release_enabled` | `1` |
| `paged_block_release_calls` | `65` |
| `paged_blocks_released` | `32` |
| `paged_blocks_released_unused` | `32` |
| `paged_block_release_bytes` | `125829120` |
| `paged_block_release_rss_drop_last_kb` | `60832` |
| `paged_block_release_rss_drop_max_kb` | `61292` |
| `paged_release_violation` | `0` |
| `paged_block_release_fail` | `0` |
| `graphs reused` | `62` |

---

## 5. 结果解释

1. **correctness 通过**：`base_vs_swap_rss_equal=0` 且三份 sha256 完全一致，证明在 swap 之上叠加 madvise 后输出仍与 baseline 逐位等价。swap-out 后丢弃物理页、swap-in 重新提交并写回，没有引入任何寻址或内容回归。

2. **swap-out / swap-in 完全闭合**：`paged_swap_out_calls = paged_swap_in_calls = 116`，`paged_blocks_swapped_out = paged_blocks_swapped_in = 116`，`paged_swap_bytes_out = paged_swap_bytes_in = 486539264`。相比 Stage 4B（`out=120/in=66`，`bytes_out > bytes_in`）的窗口尾部滞留，本阶段每个 swap-out 的块都被读/写回，达成严格对称——这是 Stage 4B plan §5.3 原始 INV-3 期望的形态，本次自然满足。

3. **读路径与写路径都成功处理 SWAPPED block**：读命中 66、写命中 50，全部成功 swap-in，零失败。证明 SWAPPED 块在读前（`set_input_paged_row_idx`）和写前（`paged_ensure_write_resident`）两条挂点都被正确拦截并换回，无漏网。

4. **独立 physical-cell swap metadata 修复了 no_offset 问题**：`paged_swap_in_fail_no_offset=0`（连同 `bad_size`/`read_cell`/`tensor_set` 全 0）。说明 swap metadata 以 physical cell 为键独立记录后，swap-in 总能定位到 backing store 中正确的 offset，不再出现「换回时找不到落盘位置」的失败路径。这是相对早期实现的关键修复点。

5. **madvise 真实发生且观测到 RSS drop**：`paged_swap_madvise_calls=116`、`paged_swap_madvise_bytes=456130560`、`madvise_failures=0`，且 `rss_samples=116`、`rss_drop_last_kb=3840`、`rss_drop_max_kb=3840`。即每次 swap-out 后都成功 madvise，并在采样中观测到 current RSS 下降。满足 plan §12 的 current RSS 通过标准（`madvise_calls>0` ∧ `madvise_bytes>0` ∧ `rss_drop_last_kb>0`）。

6. **madvise 字节 < swap-out 字节属正常**：`paged_swap_madvise_bytes=456130560 < paged_swap_bytes_out=486539264`。差额来自 page-aligned 内缩——block 首/尾未对齐的部分页被边界保护跳过,不计入 madvise。这是页对齐策略的预期行为,不是失败。

7. **Stage 4A release 回归通过**：`base_vs_release_equal=0`、sha256 不变、`graphs reused=62`、`release_violation=0`、`release_fail=0`，release 计数与 Stage 4A 基线一致（`released=32`，`bytes=125829120`，`rss_drop` 量级 ~60 MiB）。证明把页对齐逻辑抽成通用 `paged_madvise_block()` 同时供 release 与 swap 两条路径调用后，旧 release 路径行为零漂移。

---

## 6. 修订后的 Stage 4B-RSS 通过标准

| 条件 | 本次结果 | 通过 |
|---|---|---|
| sha256 与 baseline 一致 | `5c16…06` ×3 | ✅ |
| `paged_swap_out_calls > 0` | `116` | ✅ |
| `paged_swap_in_calls > 0` | `116` | ✅ |
| swap-out / swap-in 数量闭合 | `116 == 116` | ✅ |
| `paged_swap_bytes_out == paged_swap_bytes_in` | `486539264 == 486539264` | ✅ |
| 读/写 swap-in 失败 | 全 `0` | ✅ |
| `paged_swap_in_fail_*` | 全 `0` | ✅ |
| `paged_swap_backend_failures == 0` | `0` | ✅ |
| `paged_release_violation == 0` | `0` | ✅ |
| `paged_swap_madvise_calls > 0` | `116` | ✅ |
| `paged_swap_madvise_bytes > 0` | `456130560` | ✅ |
| `paged_swap_madvise_failures == 0` | `0` | ✅ |
| `paged_swap_rss_drop_last_kb > 0` | `3840` | ✅ |
| Stage 4A release 回归逐位一致 | `5c16…06`，`graphs reused=62` | ✅ |

---

## 7. 边界与不夸大

- **只声明 current RSS 方向的机制成立，不声明 peak RSS 下降**：madvise 发生在 KV 已写入、历史块进入 swap 窗口之后；peak 出现在 prefill / 首批 decode 全量 KV resident 时刻，此时尚无可 swap 的历史块，本阶段不改高水位。这与 plan §13 一致。
- **当前 RSS drop 量级小（`rss_drop_max_kb=3840` ≈ 3.75 MiB）**：原因是 `ctx=512` 下历史 KV 总量本就有限，可 swap 的块少、单块字节少。`rss_drop` 只证明机制方向正确，未证明收益规模。对比 Stage 4A release 在同跑下 `rss_drop_max_kb=61292`（≈59.8 MiB，但那是释放 32 个 UNUSED 块、字节口径不同），本阶段 swap+madvise 的 current RSS 收益规模需要更长上下文才能体现。
- 仅覆盖单序列、`!v_trans`、贪心、单轮、`ctx=512`。多序列、`v_trans`、采样、长上下文均未验证。
- swap+madvise 与 release 第一版仍互斥，未验证二者同 step 共存。

---

## 8. 下一步建议

1. **Stage 4B-RSS ctx=4096 扩展测试**：在更长上下文下重跑三方对比，历史块更多、单块更大，才能衡量 current RSS 收益的真实规模（预期 `rss_drop` 量级显著高于 `ctx=512` 的 3.75 MiB）。
2. **冷热策略 / 预取（Stage 4C 方向）**：用访问预测替代强制 sink/window，减少无谓 swap-out，并在 swap-in 前预取以掩盖 IO 延迟。
3. **peak RSS 方向**：分块 prefill / 延迟提交，与 lazy-clear 合流，降低 prefill 全量 resident 的高水位——这是比 current RSS 更大的改动，需单独立项。
4. 上述任一阶段开工前先确认是否需要先支持多序列 / `v_trans`，避免后续返工。
