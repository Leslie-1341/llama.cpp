# Stage 4B-RSS — swapped-block madvise（current RSS 回收）设计

前置：

- [docs/kv_paged_read_stage4b_block_swap_plan.md](kv_paged_read_stage4b_block_swap_plan.md)
- [docs/kv_paged_read_stage4b_block_swap_results.md](kv_paged_read_stage4b_block_swap_results.md)
- [docs/kv_paged_read_stage4a_block_release_plan.md](kv_paged_read_stage4a_block_release_plan.md)
- [docs/kv_paged_read_stage4a_block_release_results.md](kv_paged_read_stage4a_block_release_results.md)

本文是 Stage 4B-RSS 的**正式设计文档**，只描述设计，不含实现，不改源码、不 build、不跑实验。Stage 4B 已证明 historical KV physical block 可以 swap-out 到 backing store 并在读前 swap-in，输出逐位一致；Stage 4B-RSS 在此之上**回收物理页**：swap-out 成功后对 `SWAPPED` block 执行 page-aligned `MADV_DONTNEED`，并在读前 swap-in 时让物理页重新 resident。

结论先行：Stage 4B-RSS 把 Stage 4A（只释放 UNUSED block）的回收能力扩展到**有内容的历史 active block**，因为这些内容已安全存盘，物理页可丢弃后再从 backing store 恢复。

---

## 0. 前置基线（已证，不重证）

| 能力 | 来源 | 状态 |
|---|---|---|
| 非连续寻址逐位等价 | Stage 2B/3B | sha256 `5c16…06` |
| UNUSED block madvise 释放降 RSS ≈ 892 MiB | Stage 4A | 已证 |
| historical block swap-out / 读前 swap-in 逐位等价 | Stage 4B | `base_vs_swap_equal=0`，`backend_failures=0`，`release_violation=0`，`graphs reused=62` |

Stage 4B 的 swap 计数基线：`swap_out_calls=120`、`swap_in_calls=66`、`bytes_out=503316480`、`bytes_in=276824064`。本阶段不重证寻址与 swap 正确性，只在 swap 闭环上叠加 madvise 并衡量 current RSS。

---

## 1. Stage 4B-RSS 与 Stage 4B correctness probe 的区别

| 维度 | Stage 4B（correctness probe） | Stage 4B-RSS |
|---|---|---|
| 命题 | 内容能存盘并逐位读回 | 存盘后**物理页能释放并恢复** |
| swap-out 后动作 | 仅置 `SWAPPED` | 置 `SWAPPED` **+ page-aligned `MADV_DONTNEED`** |
| swap-in 前提 | block 内存仍在（madvise 未触） | block 物理页可能已被丢弃，回读即重新提交 |
| RSS 声明 | 不声明 | 声明 **current RSS 下降** |
| 守门 | sha256 + swap 计数 | sha256 + swap 计数 + RSS 采样 |

核心增量只有一处：在 `paged_swap_out_block()` 写完 backing store、置 `SWAPPED` 之后，对该 block 的物理页做 madvise。其余路径不变。

---

## 2. 为什么 SWAPPED block 可以 madvise，而 RELEASED block 不能恢复

- **RELEASED（Stage 4A）**：madvise 后内容**丢失**，恢复方式是「下次写入前重新提交得零页」。它只对 **UNUSED / 不再被读** 的 block 安全，因为没人需要旧内容。读到 RELEASED block 是 bug（记 `paged_release_violation`）。
- **SWAPPED（Stage 4B）**：madvise 前内容已**完整写入 backing store**。madvise 丢弃的只是物理页副本，源数据仍在盘上。读前 `paged_swap_in_block()` 用 `read_cell` + `ggml_backend_tensor_set` 把内容写回，触发缺页 → 物理页重新提交 → 内容逐位恢复。

因此 SWAPPED 是「内容有备份的可丢弃态」，RELEASED 是「内容无备份的可重写态」。只有前者能对**仍会被读回的 active 历史块**安全 madvise。这正是 Stage 4B-RSS 能回收 active history、而 Stage 4A 不能的根本原因。

---

## 3. swap-out 后 madvise 的安全时机

唯一安全时机：**`write_cell` 全部成功、block 状态已置 `SWAPPED` 之后**，在 `paged_swap_out_block()` 内部。

理由：

1. 必须在内容确实落盘后才能丢弃物理页，否则 swap-in 读回的是不完整数据。
2. 必须在置 `SWAPPED` 之后，保证后续任何读到该 block 的路径都会先走 swap-in（INV-1），不会直接 gather 到被 madvise 的零页。
3. 若 `write_cell` 中途失败，block 保持 `RESIDENT`、不 madvise（§7 风险缓解），避免「内容没存盘却把页丢了」。

不在 `paged_swap_out_window()` 批量末尾统一 madvise，而是**每块成功即 madvise**，使状态与物理页严格同步，便于失败回滚。

---

## 4. SWAPPED block 的状态语义（4B-RSS 强化）

`SWAPPED` 在本阶段承载三条不变式：

1. **backing store 中有完整 K/V 内容**：block 内每个 cell 的 staging 都已 `write_cell` 成功。
2. **物理页可以被释放**：madvise 后 KV tensor 对应行的物理页可能不存在；逻辑地址保留，内容不可信。
3. **读前必须 swap-in**：任何 `row_idx` 引用到该 block 的 step，必须先 `paged_swap_in_block()` 回读，才能让 graph gather 读取。

读路径 INV-1（Stage 4B 已建立）在本阶段升级为「swap-in 不仅恢复内容，还重新提交物理页」。

---

## 5. `paged_swap_out_block()` 是否应写完 backing store 后立即 madvise

**是。** 在 `paged_swap_out_block()` 内、置 `SWAPPED` 之后立即对该 block 物理页 madvise（受 `LLAMA_KV_PAGED_SWAP_MADVISE` 子开关门控）。

- 立即 madvise 让「状态=SWAPPED」与「物理页已丢弃」原子对应，无中间窗口。
- 失败处理：`write_cell` 失败 → 不置 SWAPPED、不 madvise；madvise 失败 → 块仍是 SWAPPED（内容已存盘，读前 swap-in 仍正确），只累加 `paged_swap_madvise_failures`，不影响 correctness。
- madvise 是 best-effort 的 RSS 优化，不是 correctness 前提：即使一次 madvise 都没成功，输出仍逐位正确。

---

## 6. `paged_swap_in_block()` 是否需要在读回前确保物理页重新 resident

**不需要显式操作**，由 `ggml_backend_tensor_set` 写回自然触发。

- `read_cell` 把内容读到 staging（host 缓冲），`ggml_backend_tensor_set` 把 staging 写入 KV tensor 对应行。
- 写入被 madvise 过的地址会触发缺页 → 内核分配新物理页 → 内容写入。物理页在 swap-in 完成时即重新 resident，无需额外 `madvise(MADV_WILLNEED)` 或手动 touch。
- swap-in 完成后置 `RESIDENT`，物理页与内容同时恢复。

因此 swap-in 路径相对 Stage 4B **零改动**，madvise 的恢复语义完全由现有 `ggml_backend_tensor_set` 覆盖。

---

## 7. 如何复用 Stage 4A 的 page-align madvise 逻辑

Stage 4A 的 [paged_release_blocks:2478](../src/llama-kv-cache.cpp#L2478) 内 `advise_block` lambda 已实现：

- 按 layer 遍历 K/V tensor，`row = ggml_row_size(...)`；
- block 物理区间 `[lo_cell, hi_cell) = [b*bs, min((b+1)*bs, paged_kv_size))`；
- 地址 `lo_a/hi_a`，**页对齐内缩**（`a_start = align_up(lo_a)`，`a_end = align_down(hi_a)`），`a_end <= a_start` 则跳过；
- 边界保护：`lo_a`/`hi_a` 未对齐且相邻块 active 时跳过该侧，绝不误伤 live 页。

Stage 4B-RSS **抽出**该 page-align + 边界保护逻辑为可复用 helper（如 `paged_madvise_block(physical_block, &bytes)`），同时被 `paged_release_blocks`（RELEASED 路径）与 `paged_swap_out_block`（SWAPPED 路径）调用。两条路径共用同一套页对齐内缩，避免逻辑分叉。

---

## 8. 如何避免 active block 被 madvise

两层保护：

1. **触发层**：`paged_swap_out_window()` 只对 `[sink_blocks, n_kv_blocks - window_blocks)` 区间的块 swap-out。窗口尾部 `window_blocks` 个块（含当前 step 必读的近期块）永不进入 swap，因此不会被 madvise。
2. **页对齐层**（复用 §7）：`paged_madvise_block` 的边界保护——block 首/尾地址未页对齐且相邻物理块 active 时，跳过未对齐的那一页，绝不丢弃跨块共享页里 active 块的数据。

二者叠加保证：被 madvise 的物理页要么属于已 SWAPPED 的非窗口块，要么是完全落在该块内的对齐页。当前 step `row_idx` 引用的 active 块的物理页永不被丢弃；下一 step 若引用到已 swap 的块，读前 swap-in 先恢复。

---

## 9. 是否继续禁止 `LLAMA_KV_PAGED_RELEASE` 与 `LLAMA_KV_PAGED_SWAP` 同开

**第一版继续禁止同开。**

- RELEASED（内容丢弃）与 SWAPPED（内容存盘）对同一物理块的恢复语义不同，同 step 对同块同时施加二者会产生二义。
- `LLAMA_KV_PAGED_SWAP=1` 时强制 `paged_block_release_enabled=false`（沿用 Stage 4B §3）。
- 注意：Stage 4B-RSS 的 madvise 是在 **swap 路径内部**对 SWAPPED 块做的，与 Stage 4A 的 release 路径是两条独立通道，复用 §7 的 helper 但不互相调用。本阶段不验证二者共存。
- 未来若要合流（同一 step 既 release UNUSED 又 swap+madvise history），需单独立项设计状态优先级，不在本阶段范围。

---

## 10. 新增统计字段

放在 [llama-kv-cache.h](../src/llama-kv-cache.h#L496) paged 计数区：

| 字段 | 含义 |
|---|---|
| `paged_swap_madvise_calls` | swap-out 后对 SWAPPED 块成功发起 madvise 的次数 |
| `paged_swap_madvise_bytes` | 累计经 madvise 释放的字节（页对齐内缩后） |
| `paged_swap_madvise_failures` | `madvise()` 返回非零的次数 |
| `paged_swap_rss_before_last_kb` | 最近一次 swap-out 批 madvise 前的 current RSS（KB） |
| `paged_swap_rss_after_last_kb` | 最近一次 swap-out 批 madvise 后的 current RSS（KB） |
| `paged_swap_rss_drop_last_kb` | `before - after`（下界 0） |

RSS 采样复用 Stage 4A 的 `get_current_rss_kb()`，在 `paged_swap_out_window()` 一批 swap-out（含 madvise）的前后各采一次，避免逐块 `/proc` 读取开销。统计在 [paged_log_stats](../src/llama-kv-cache.cpp#L1900) 打印。

---

## 11. correctness 验证标准

与 Stage 4B 一致，madvise 不得破坏任何一条：

- baseline vs swap+madvise 输出**逐 token 完全一致**；
- **sha256 完全一致**（`5c16…06`）；
- `paged_swap_out_calls > 0`、`paged_swap_in_calls > 0`；
- `0 < paged_swap_bytes_in <= paged_swap_bytes_out`（沿用 Stage 4B 修订标准）；
- `paged_swap_backend_failures == 0`；
- `paged_release_violation == 0`；
- `graphs reused == 62`。

> 关键：`paged_swap_madvise_failures > 0` **不**判失败（madvise 是 best-effort），只要 sha256 与 swap 计数守门通过。

---

## 12. current RSS 验证标准

- `paged_swap_madvise_calls > 0`（确实发生了 madvise）；
- `paged_swap_madvise_bytes > 0`；
- `paged_swap_rss_drop_last_kb > 0`（至少一次采样观测到 current RSS 下降）；
- 与 Stage 4B（swap 但不 madvise）同 prompt 对比，swap+madvise 的 current RSS 峰后稳态应明显更低；
- 与 Stage 4A（只 release UNUSED）对比，本阶段额外回收了 active history block 的物理页，current RSS 下降应**不小于** Stage 4A。

RSS 为带噪指标，以多次采样的稳态趋势为准，不以单点为准。

---

## 13. 为什么本阶段仍不保证 peak RSS 下降

- madvise 只回收**已 swap-out 的历史块**的物理页，发生在 prompt 已构建、KV 已写入之后。**peak** 往往出现在 prefill / 首批 decode 时全量 KV 仍 resident 的时刻，此时尚无可 swap 的历史块。
- 强制 sink/window 在 `n_kv` 足够大后才触发 swap-out，peak 之前不释放。
- backing store 尾部滞留（`bytes_out > bytes_in`）说明部分块换出后未换回，能降 current RSS，但不改变 peak 时刻的内存高水位。
- 降 peak 需要 prefill 期就分块写盘 / 延迟提交，属后续阶段（与 lazy-clear / 分块 prefill 合流），不在本阶段范围。

本阶段只声明 **current（稳态）RSS 下降**，不声明 peak。

---

## 14. 风险表

| 风险 | 等级 | 触发条件 | 缓解 |
|---|---|---|---|
| madvise 前内容未落盘，swap-in 读到残缺数据 | 高 | madvise 时机早于 `write_cell` 完成 | madvise 严格在置 SWAPPED 之后（§3/§5）；write_cell 失败则不置 SWAPPED 不 madvise |
| 误 madvise 当前 step active 块物理页 | 高 | 触发窗口或页对齐错误 | 触发层窗口排除 + 页对齐内缩边界保护双层（§8）；sha256 守门 |
| 跨块共享页误伤相邻 active 块 | 高 | block 字节非页整除且邻块 active | 复用 Stage 4A 边界保护 lambda（§7），未对齐侧整页跳过 |
| madvise 失败被误判 correctness 失败 | 中 | `madvise()` 返回非零 | best-effort，仅记 `paged_swap_madvise_failures`，不进 correctness 判据（§11） |
| swap-in 未重新提交物理页 | 中 | 误以为需手动 touch | `ggml_backend_tensor_set` 写回自然触发缺页恢复（§6） |
| release 与 swap madvise 路径分叉导致逻辑漂移 | 中 | 两处各写一份页对齐 | 抽 `paged_madvise_block` helper 共用（§7） |
| RSS 不降（madvise 被合并/页未实际归还） | 低 | block 不足一页 / 内核策略 | `paged_swap_madvise_bytes`、`rss_drop_last_kb` 暴露；§12 以稳态趋势判定 |
| 与 Stage 4A release 同开语义冲突 | 低 | 两开关同开 | 第一版强制互斥（§9） |

---

## 15. Codex 实现提示词草案

> 在 `src/llama-kv-cache.{h,cpp}` 实现 Stage 4B-RSS：在已有 Stage 4B block swap 之上，对 swap-out 的 SWAPPED 块做 page-aligned madvise 以回收 current RSS。
>
> 1. 从 `paged_release_blocks` 内的 `advise_block` lambda 抽出可复用 helper `paged_madvise_block(uint32_t physical_block, uint64_t * out_bytes)`，保留页对齐内缩与相邻-active 边界保护；`paged_release_blocks` 改为调用它（行为不变）。
> 2. 在 `paged_swap_out_block()` 内，`write_cell` 全部成功并置 `SWAPPED` 之后，若 `LLAMA_KV_PAGED_SWAP_MADVISE=1`，调 `paged_madvise_block` 释放该块物理页；累加 `paged_swap_madvise_calls` / `paged_swap_madvise_bytes`；madvise 失败累加 `paged_swap_madvise_failures`，不回滚状态。
> 3. `paged_swap_in_block()` 不改：`ggml_backend_tensor_set` 写回自然重新提交物理页。
> 4. 在 `paged_swap_out_window()` 的一批 swap-out 前后用 `get_current_rss_kb()` 各采一次，写 `paged_swap_rss_before_last_kb` / `after` / `drop_last_kb`。
> 5. 新增 §10 全部统计字段并在 `paged_log_stats` 打印。
> 6. 门控：`LLAMA_KV_PAGED_SWAP=1` 仍强制 `paged_block_release_enabled=false`；`LLAMA_KV_PAGED_SWAP_MADVISE` 为 swap 下的子开关。
> 7. 不动 graph、不动 attention kernel、不动 read/write 填值循环结构、仅 `!v_trans && n_stream==1`。
>
> 约束：不做冷热策略、不做预取、不做多序列、不追求 peak RSS。correctness 由 sha256 + swap 计数守门，madvise 失败不判 correctness 失败。

---

## 16. 下一步建议

1. 本 plan 落地后跑三方对比（baseline / Stage 4B swap-only / Stage 4B-RSS swap+madvise），产出 `kv_paged_read_stage4b_rss_results.md`，核对 sha256 不变 + `swap_madvise_bytes>0` + `rss_drop_last_kb>0`。
2. current RSS 闭环通过后，再单独立项做 **peak RSS**（分块 prefill / 延迟提交），这是与 lazy-clear 合流的更大改动。
3. RSS 路径稳定后再考虑 Stage 4C 预取 / 冷热策略，用访问预测替代强制窗口，减少 `swap_out(120) >> swap_in(66)` 的过度换出。
