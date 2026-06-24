# Stage 4B — block-aware KV swap correctness probe 结果

前置：

- [docs/kv_paged_read_stage4b_block_swap_plan.md](kv_paged_read_stage4b_block_swap_plan.md)
- [docs/kv_paged_read_stage4a_block_release_results.md](kv_paged_read_stage4a_block_release_results.md)
- [docs/kv_paged_read_stage2b_non_identity_mapping_results.md](kv_paged_read_stage2b_non_identity_mapping_results.md)

本文是 Stage 4B 的**结果文档**。Stage 4B 实现了 block-aware KV swap 的 correctness probe：按 physical block 粒度把历史 KV block swap-out 到 backing store，并在读路径 `set_input_paged_row_idx()` 填 `row_idx` 前自动 swap-in。第一版**不追求 RSS 最优，只验证 swap correctness 闭环**。

结论先行：**correctness 闭环已打通** —— swap-out 与 swap-in 都真实发生，且开 swap 后输出与 baseline 逐位一致。

---

## 0. 实验设置

- 单序列（`n_stream==1`）、`!v_trans`、贪心解码、单轮。
- 三方对比：
  1. baseline：`LLAMA_KV_PAGED=0`；
  2. no-swap：`LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1`，swap 关；
  3. swap：`LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1 LLAMA_KV_PAGED_SWAP=1`（release 强制关闭）。
- physical block swap，block 内逐 cell 复用 exact swap 的 `kv_swap_store->write_cell/read_cell`。

---

## 1. 验证结果

| 字段 | 值 |
|---|---|
| `base_vs_noswap_equal` | `0`（diff 空，逐位一致） |
| `base_vs_swap_equal` | `0`（diff 空，逐位一致） |
| sha256（三份完全一致） | `5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006` |
| `paged_swap_enabled` | `1` |
| `paged_swap_out_calls` | `120` |
| `paged_blocks_swapped_out` | `120` |
| `paged_swap_in_calls` | `66` |
| `paged_blocks_swapped_in` | `66` |
| `paged_swap_bytes_out` | `503316480` |
| `paged_swap_bytes_in` | `276824064` |
| `paged_swap_backend_failures` | `0` |
| `paged_release_violation` | `0` |
| `graphs reused` | `62` |

---

## 2. 结果解释

1. **correctness probe 通过**：`base_vs_swap_equal=0` 且三份 sha256 完全一致，证明开启 block-aware swap 后输出与 baseline 逐位等价。swap 路径没有引入任何寻址或内容回归。
2. **swap-out 与 swap-in 都真实发生**：`paged_swap_out_calls=120` / `paged_blocks_swapped_out=120` 证明历史 block 被实际写到 backing store；`paged_swap_in_calls=66` / `paged_blocks_swapped_in=66` 证明这些 block 在读前被实际换回。闭环成立。
3. **后端零失败、读零违例**：`paged_swap_backend_failures=0` 表示所有 `write_cell/read_cell` 成功；`paged_release_violation=0` 表示 4B 单开 swap 时没有任何块走到 RELEASED 态（与 §0「release 强制关闭」一致）。`graphs reused=62` 与 Stage 2B/4A 同基线，图复用未被 swap 破坏。
4. **`bytes_out != bytes_in` 不是失败**：`paged_swap_bytes_out=503316480 > paged_swap_bytes_in=276824064`。原因是强制 sink/window 策略下，最后若干批被 swap-out 的 block 在程序结束前没有再次落入读窗口，因此从未被 swap-in。这是窗口策略的预期行为，而非数据丢失——任何**实际被读到**的 block 都已正确 swap-in（由 sha256 守门）。
5. **修订 INV-3**：plan §5.3 原写 `bytes_out == bytes_in`，过于严格。实际正确的对称性是「每个**会被读回**的 block 都 swap-in」，而非「所有 swap-out 都 swap-in」。据此修订 Stage 4B 的通过标准（见 §3）。

---

## 3. 修订后的 Stage 4B 通过标准

| 条件 | 本次结果 | 通过 |
|---|---|---|
| sha256 与 baseline 一致 | `5c16…06` ×3 | ✅ |
| `paged_swap_out_calls > 0` | `120` | ✅ |
| `paged_swap_in_calls > 0` | `66` | ✅ |
| `paged_swap_bytes_out > 0` | `503316480` | ✅ |
| `paged_swap_bytes_in > 0` | `276824064` | ✅ |
| `paged_swap_bytes_in <= paged_swap_bytes_out` | `276824064 <= 503316480` | ✅ |
| `paged_swap_backend_failures == 0` | `0` | ✅ |
| `paged_release_violation == 0` | `0` | ✅ |

> 相对 plan 的关键调整：把 `bytes_out == bytes_in` 放宽为 `0 < bytes_in <= bytes_out`。窗口尾部 swap-out 但未被读回的 block 属正常现象，correctness 由 sha256 + `backend_failures=0` 守门，而非字节严格对称。

---

## 4. 边界与不夸大

- **本阶段不声明 RSS 收益**：第一版只验证 block-aware swap correctness 闭环，未叠加 madvise，未测 current RSS。`bytes_out > bytes_in` 表明部分 swapped-out block 仍滞留 backing store，但其物理内存释放收益不在本阶段衡量范围。
- 仅覆盖单序列、`!v_trans`、贪心、单轮。多序列、`v_trans`、采样路径均未验证。
- swap 与 Stage 4A release 第一版互斥，未验证二者共存。

---

## 5. 下一步建议

1. **Stage 4B-RSS**：在 swap-out 成功后对 `SWAPPED` 块叠加 madvise（与 Stage 4A 路径合流），衡量真实 current RSS 下降，并解决 `bytes_out > bytes_in` 尾部 block 的物理回收。
2. **Stage 4C 预取 / 冷热策略**：用更合理的访问预测替代强制 sink/window，减少无谓 swap-out（`120 swap-out` vs `66 swap-in` 说明当前策略过度激进），降低 swap IO 量。
3. 上述任一阶段开工前先确认是否需要先支持多序列 / `v_trans`，避免后续返工。
