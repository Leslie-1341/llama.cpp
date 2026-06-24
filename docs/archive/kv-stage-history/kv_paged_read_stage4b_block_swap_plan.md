# Stage 4B — block-aware KV swap（correctness probe）设计

前置：

- [docs/kv_paged_read_stage4a_block_release_plan.md](kv_paged_read_stage4a_block_release_plan.md)
- [docs/kv_paged_read_stage4a_block_release_results.md](kv_paged_read_stage4a_block_release_results.md)
- [docs/kv_paged_read_stage2b_non_identity_mapping_results.md](kv_paged_read_stage2b_non_identity_mapping_results.md)
- [docs/kv_paged_read_stage3b_ingraph_gather_results.md](kv_paged_read_stage3b_ingraph_gather_results.md)
- [docs/kv_runtime_swap_stage2_exact_swap_results.md](kv_runtime_swap_stage2_exact_swap_results.md)

本文是 Stage 4B 的**正式设计文档**，只描述设计，不含实现，不改源码、不 build、不跑实验。Stage 4B 第一版的目标从 Stage 4A 的「按块 madvise 释放降 RSS」收窄为**单一闭环命题**：证明一个历史 KV physical block 可以被 swap-out 到 backing store，再在读前 swap-in，最终输出与 baseline 逐位一致。第一版**不追求 RSS 最优、不做冷热策略、不做多序列、不做预取**。

---

## 0. 前置基线（已证，不重证）

Stage 4B 建立在两条已端到端验证的能力之上，本阶段不重新证明：

1. **非连续寻址正确**（Stage 2B / 3B）：`LLAMA_KV_PAGED_SHIFT=1` 下读写共用 `paged_block_table`，输出 sha256 与 baseline 逐位等价。
2. **physical-block 状态机与挂载点可用**（Stage 4A）：`paged_block_states[]` 以 `physical_block = phys / paged_block_size` 为索引；读路径在 `set_input_paged_row_idx` 填 `row_idx` 前已有 `paged_check_read_resident` 探针，写路径在 `paged_write_resolve` 后已有 `paged_ensure_write_resident` 探针。

任何 Stage 4B 引入的 sha256 回归都意味着 swap 路径有 bug，而非寻址问题。

---

## 1. Stage 4B 目标

1. 引入 **physical-block 级 swap-out / swap-in**：把不在当前 step `row_idx` 引用窗口内的历史 block 内容写到 backing store，并把块状态置为 `SWAPPED`。
2. 在读前（`set_input_paged_row_idx` 填值循环内）与写前（`set_input_k_idxs`/`set_input_v_idxs` 的 ensure 挂点）按需把 `SWAPPED` 块 swap-in 回 `RESIDENT`。
3. 默认关闭（env 门控），开启后输出与 baseline **逐位一致**（sha256 守门），同时 swap 计数器证明确实发生了 swap-out 与 swap-in。
4. 复用 Stage 2 exact swap 已验证的序列化布局与 `kv_swap_store->write_cell/read_cell` 接口，block 级只是「块内按 cell 循环」，不新增 backing-store IO 接口。

---

## 2. 非目标

- **不追求 RSS 最优**：第一版只证 correctness 闭环，RSS 收益留待后续阶段。
- **不做冷热/不活跃判定**：单序列 causal decode 下中间历史块始终 active，任何「按冷热触发」都无法稳定触发 swap。第一版用**强制 sink/window** 触发。
- **不做多序列**：仅 `n_stream == 1`。
- **不做预取**：swap-in 严格惰性，挂在读/写填值点按需触发。
- **第一版不与 Stage 4A release 同时开启**：见 §3，二者对块内容的语义冲突。
- **不做 `v_trans`**：仅 `!v_trans`，与 Stage 2 exact swap 一致。

---

## 3. Stage 4A release 与 Stage 4B swap 的区别

| 维度 | Stage 4A release | Stage 4B swap |
|---|---|---|
| 机制 | `madvise(MADV_DONTNEED)` | 序列化到 `kv_swap_store` + 回读 |
| 块内容 | **丢弃**（下次写入前重新提交，零页） | **保留**（存盘，swap-in 时逐位恢复） |
| 终态 | `RELEASED` | `SWAPPED` |
| 能否释放 active history | 否（active 块必须留） | 是（active 历史块也可 swap，读前再换回） |
| 第一版同时开启 | —— | **不**与 release 同开 |

关键冲突：`RELEASED` 表示「内容已丢、靠重新触摸得零页」，`SWAPPED` 表示「内容已存盘、靠 read-back 恢复」。若同一块在同一 step 既被 release 又被 swap，恢复语义二义。第一版 env 门控**二选一**：`LLAMA_KV_PAGED_SWAP=1` 时强制 `paged_block_release_enabled=false`，使 correctness 判定不被 release 路径污染。`paged_release_blocks` 现有对 `SWAPPED` 块的跳过（[llama-kv-cache.cpp:2518](../src/llama-kv-cache.cpp#L2518)）在 4B 单开 swap 时不会被触发，但保留该跳过作为未来共存的安全网。

---

## 4. 为什么按 physical block swap

1. **与现有状态机同维度**：`paged_block_states[]`、`paged_release_blocks` 的 `active[]`、`paged_ensure_write_resident`/`paged_check_read_resident` 全部以 physical block 为索引。physical block swap 复用同一套结构，零新增映射层。
2. **logical block swap 需第二套状态**：logical→physical 已由 `paged_block_table` 承担，若再按 logical 维度记录 swap 状态，会出现「同一物理块被多个 logical 视图引用」时的状态归属问题。第一版坚决避免。
3. **IO 单位天然对齐**：staging 缓冲、`write_cell` key、madvise 页对齐都以连续 cell 区间 `[b*bs, (b+1)*bs)` 表达，physical block 是最自然的批量单位。

结论：第一版 swap 单位 = **physical block**。

---

## 5. block 状态机：UNUSED / RESIDENT / RELEASED / SWAPPED

复用 [llama-kv-cache.h:458](../src/llama-kv-cache.h#L458) 已有的 `paged_block_state` 枚举（`UNUSED=0`, `RESIDENT`/`FREE`, `RELEASED`, `SWAPPED=3`），不新增枚举值。

### 5.1 状态定义（4B 视角）

- **UNUSED**：尚未被任何 logical cell 映射。
- **RESIDENT**：内容驻留在 KV tensor 内存，可直接被 graph 读。
- **RELEASED**：Stage 4A 状态；4B 单开 swap 时不产生此态。读路径若遇到则记 `paged_release_violation`。
- **SWAPPED**：内容已写到 backing store，KV tensor 内对应行内容**不可信**，读前必须 swap-in。

### 5.2 状态转移（4B 第一版只两条边）

```
RESIDENT --paged_swap_out_block--> SWAPPED   （强制 sink/window 触发）
SWAPPED  --paged_swap_in_block --> RESIDENT  （读前/写前 ensure 触发）
```

- 写路径遇 `SWAPPED`：直接置 `RESIDENT`，**无需 read-back**（写覆盖整 cell，backing-store 内容可弃）。
- 读路径遇 `SWAPPED`：必须 `paged_swap_in_block` 回读后再置 `RESIDENT`。

### 5.3 不变式

- INV-1：graph 真正读 `phys` 行之前，其所属块必为 `RESIDENT`（由 §10 读挂点保证）。
- INV-2：写 `phys` 行之前，其所属块必为 `RESIDENT`（由 §11 写挂点保证）。
- INV-3：`paged_swap_bytes_out == paged_swap_bytes_in`（同一闭环内每个 swap-out 的块都被 swap-in，且字节数对称）。
- INV-4：4B 单开 swap 时 `paged_release_violation == 0`。

---

## 6. backing store 设计

**复用旧 exact swap 接口，不新增 block 级 IO。**

- backing store 仍是 `kv_swap_store`（[llama-kv-cache.h:80/115](../src/llama-kv-cache.h#L80) 的 `write_cell`/`read_cell`）。
- block 级 swap-out = 对 `cell ∈ [b*bs, (b+1)*bs)`（上界 `min` 到 `paged_kv_size`）逐 cell 复用 `swap_out_cell` 的 staging 序列化布局（固定顺序：layer0 K, layer0 V, layer1 K, layer1 V, …）+ `write_cell(0, cell, …)`。
- block 级 swap-in = 对同区间逐 cell 复用 `swap_in_cell` 的 `read_cell` + `ggml_backend_tensor_set` 回填。
- **block 级状态机 + cell 级 IO**：状态在 `paged_block_states[block]` 上记录，字节搬运沿用已验证的 per-cell 代码。这样第一版不引入未验证的 IO 路径。

字节会计：每搬运一个 cell 累加其 staging size 到 `paged_swap_bytes_out` / `paged_swap_bytes_in`。

---

## 7. block 级 swap-out 设计

新增 `paged_swap_out_block(uint32_t physical_block)`（参考 [swap_out_cell:1963](../src/llama-kv-cache.cpp#L1963)）：

1. 门控：`kv_paged_enabled && paged_swap_enabled && !v_trans && n_stream==1 && kv_swap_store`，否则 no-op。
2. 仅当 `paged_block_states[physical_block] == RESIDENT` 时执行；否则跳过。
3. 对块内每个 cell：staging 序列化 → `kv_swap_store->write_cell`；失败累加 `paged_swap_backend_failures` 并中止该块（保持 RESIDENT，不进入半 swap 态）。
4. 全块成功后：`paged_block_states[physical_block] = SWAPPED`；`paged_swap_out_calls += 1`；`paged_blocks_swapped_out += 1`；`paged_swap_bytes_out += <块字节数>`。

---

## 8. block 级 swap-in / ensure_resident 设计

新增 `paged_swap_in_block(uint32_t physical_block)`（参考 [swap_in_cell:2023](../src/llama-kv-cache.cpp#L2023)）：

1. 门控同 §7。
2. 仅当 `paged_block_states[physical_block] == SWAPPED` 时执行。
3. 对块内每个 cell：`read_cell` → `ggml_backend_tensor_set` 回填；失败累加 `paged_swap_backend_failures`。
4. 成功后：`paged_block_states[physical_block] = RESIDENT`；`paged_swap_in_calls += 1`；`paged_blocks_swapped_in += 1`；`paged_swap_bytes_in += <块字节数>`。

`ensure_resident(block)` 最小语义（在现有探针内实现，不新增 call site）：

- **RESIDENT** → no-op。
- **SWAPPED** → 读前调 `paged_swap_in_block`；写前直接置 `RESIDENT`（写覆盖，无需 read-back）。
- **RELEASED** → 4B 单开 swap 不应出现；读路径记 `paged_release_violation`，写路径沿用 Stage 4A「翻回 RESIDENT」。

---

## 9. 强制 sink/window correctness probe

单序列 causal decode 每步都读 `[0, n_kv)`，所有历史块都是 active，冷热策略无法触发 swap。第一版用**强制 sink/window**：

新增 `paged_swap_out_window(uint32_t n_kv)`（参考 [swap_out_window:2106](../src/llama-kv-cache.cpp#L2106)）：

1. 门控同 §7；`n_kv_blocks = ceil(n_kv / paged_block_size)`。
2. 若 `n_kv_blocks <= sink_blocks + window_blocks` → 跳过，`paged_swap_window_skipped += 1`。
3. 对 `b ∈ [sink_blocks, n_kv_blocks - window_blocks)` 且状态为 `RESIDENT` 的块调 `paged_swap_out_block(b)`。
4. 默认 `sink_blocks = 1`、`window_blocks = 1`（env 可调），保证中等长度 prompt 即可触发。

触发点：[apply():4189](../src/llama-kv-cache.cpp#L4189) 处，紧邻原 `paged_release_blocks` 调用（4B 单开 swap 时 release 关闭）。这些被 swap-out 的中间历史块在下一 decode step 必被 `[0,n_kv)` 读窗口引用，因而必然触发 swap-in —— 同时稳定满足 `swap_out_calls>0` 与 `swap_in_calls>0`。

---

## 10. 读路径接入点

[set_input_paged_row_idx:3105](../src/llama-kv-cache.cpp#L3105) 填值循环内，`data[r] = phys`（[行 3123](../src/llama-kv-cache.cpp#L3123)）之前、现 `paged_check_read_resident(phys)`（[行 3119](../src/llama-kv-cache.cpp#L3119)）的位置：

- 把 `paged_check_read_resident` 升级：若 `phys` 所属块为 `SWAPPED`，调 `paged_swap_in_block`，置 `RESIDENT`，使 graph 读该行前内容已恢复（INV-1）。
- 这是唯一一个「已知 `phys`→block、且在 graph 实际 gather 之前」的点，无需新增 call site。

graph 侧触发链不变：[llama-graph.cpp:455](../src/llama-graph.cpp#L455) `llm_graph_input_attn_kv::set_input` → `mctx->set_input_paged_row_idx`。

---

## 11. 写路径接入点

[set_input_k_idxs:3047](../src/llama-kv-cache.cpp#L3047) / [set_input_v_idxs:3066](../src/llama-kv-cache.cpp#L3066) 循环内，`phys = paged_write_resolve(cell)` 后的 `paged_ensure_write_resident(phys)`（[1741](../src/llama-kv-cache.cpp#L1741)）：

- 扩展 SWAPPED 分支：目标块为 `SWAPPED` 时直接置 `RESIDENT`（写覆盖整 cell，丢弃 backing-store 内容，**不**做 read-back）。
- 三处调用点（[3060](../src/llama-kv-cache.cpp#L3060)/[3080](../src/llama-kv-cache.cpp#L3080)/[3096](../src/llama-kv-cache.cpp#L3096)）已就位，无需新增。

---

## 12. 新增统计字段

放在 [llama-kv-cache.h](../src/llama-kv-cache.h#L496) paged 计数区（`mutable` 视读写路径而定）：

| 字段 | 含义 |
|---|---|
| `paged_swap_out_calls` | `paged_swap_out_block` 成功次数 |
| `paged_swap_in_calls` | `paged_swap_in_block` 成功次数 |
| `paged_blocks_swapped_out` | 成功 swap-out 的块数 |
| `paged_blocks_swapped_in` | 成功 swap-in 的块数 |
| `paged_swap_bytes_out` | 累计 swap-out 字节 |
| `paged_swap_bytes_in` | 累计 swap-in 字节 |
| `paged_swap_backend_failures` | backing-store IO 失败次数 |
| `paged_swap_window_skipped` | `paged_swap_out_window` 因窗口过短跳过次数 |

在 [paged_log_stats](../src/llama-kv-cache.cpp#L1900) 内打印（参考既有 paged 计数打印格式）。

---

## 13. correctness 验证标准

同一 prompt、贪心（`temp=0`）、单序列、单轮，两方对比：

- baseline：`LLAMA_KV_PAGED=1` 但 swap 关（或 `LLAMA_KV_PAGED=0`）。
- probe：`LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1 LLAMA_KV_PAGED_SWAP=1`（release 强制关闭）。

通过条件（全部满足）：

- baseline vs swap 输出**逐 token 完全一致**；
- **sha256 完全一致**；
- `paged_swap_out_calls > 0`；
- `paged_blocks_swapped_out > 0`；
- `paged_swap_in_calls > 0`；
- `paged_blocks_swapped_in > 0`；
- `paged_swap_bytes_out == paged_swap_bytes_in`；
- `paged_swap_backend_failures == 0`；
- `paged_release_violation == 0`；
- `graphs reused == 62`（与 Stage 2B/4A 同基线，证明图复用未被 swap 破坏）。

---

## 14. 风险表

| 风险 | 等级 | 触发条件 | 缓解 |
|---|---|---|---|
| 半 swap 态（块内部分 cell 写盘后 IO 失败） | 高 | `write_cell` 中途失败 | 失败即中止该块并保持 `RESIDENT`，不置 `SWAPPED`；累加 `paged_swap_backend_failures`，由 §13 守门 |
| 读前漏 swap-in 导致读到陈旧/零内容 | 高 | 读挂点未覆盖某 `phys` | sha256 守门；INV-1；唯一读入口 `set_input_paged_row_idx` 逐行覆盖 |
| swap 与 release 语义冲突 | 中 | 同 step 二者都开 | env 门控二选一，4B 单开 swap 时强制 `paged_block_release_enabled=false`（§3） |
| 块字节非对称导致 INV-3 失败 | 中 | swap-out / swap-in 区间或 staging 布局不一致 | swap-in 严格镜像 swap-out 的 cell 区间与 staging 顺序；`bytes_out==bytes_in` 守门 |
| 窗口过短从不触发（假阴性） | 中 | prompt 太短 | `paged_swap_window_skipped` 暴露；probe 选用足够长 prompt 使 `n_kv_blocks > sink+window` |
| `v_trans` / 多序列误入 | 低 | 配置不满足约束 | 门控 `!v_trans && n_stream==1`，否则 no-op |
| RSS 不降甚至略升 | 低（第一版可接受） | 第一版只 correctness | 非目标；§2 已声明，留待后续阶段 |

---

## 15. Codex 实现提示词草案

> 在 `src/llama-kv-cache.{h,cpp}` 中实现 Stage 4B block-aware KV swap 的 correctness probe，复用现有 `paged_block_state` 枚举与 `kv_swap_store`。
>
> 1. 新增成员函数 `paged_swap_out_block(uint32_t physical_block)`、`paged_swap_in_block(uint32_t physical_block)`、`paged_swap_out_window(uint32_t n_kv)`，门控 `kv_paged_enabled && paged_swap_enabled && !v_trans && n_stream==1 && kv_swap_store`。
> 2. block 级搬运按块内 `cell ∈ [b*bs, min((b+1)*bs, paged_kv_size))` 逐 cell 复用 `swap_out_cell`/`swap_in_cell` 的 staging 布局与 `write_cell`/`read_cell`，但状态记录在 `paged_block_states[block]`。
> 3. swap-out 成功置 `SWAPPED`，swap-in 成功置 `RESIDENT`；IO 失败中止该块、保持 `RESIDENT`、累加 `paged_swap_backend_failures`。
> 4. 读路径：在 `set_input_paged_row_idx` 的 `data[r]=phys` 之前升级 `paged_check_read_resident`，对 `SWAPPED` 块调 `paged_swap_in_block`。
> 5. 写路径：在 `paged_ensure_write_resident` 增加 `SWAPPED` 分支，直接置 `RESIDENT`，不 read-back。
> 6. 在 `apply()` 现 `paged_release_blocks` 调用旁新增 `paged_swap_out_window(n_kv)`；env `LLAMA_KV_PAGED_SWAP=1` 时强制关闭 `paged_block_release_enabled`。
> 7. 新增 §12 全部统计字段并在 `paged_log_stats` 打印。
> 8. 默认关闭；不动 graph 侧；不动 read/write 填值循环结构（只在既有探针内扩展）。
>
> 约束：不做多序列、不做预取、不做冷热策略、不追求 RSS。

---

## 16. 下一步建议

1. 本 plan 落地实现后，按 §13 跑 baseline vs swap 双跑，产出 `kv_paged_read_stage4b_block_swap_results.md`，重点核对 sha256 与 8 个 swap 计数器。
2. correctness 闭环通过后，再单独立项做 Stage 4B-RSS：swap-out 后对 `SWAPPED` 块叠加 madvise（与 Stage 4A 路径合流），并衡量真实 current RSS 下降。
3. RSS 阶段通过后再考虑冷热/预取/多序列，均不属于本 correctness probe。
