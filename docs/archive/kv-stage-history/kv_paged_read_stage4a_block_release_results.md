# Stage 4A — block-aware release (madvise-only) results

前置：

- [docs/kv_paged_read_stage4a_block_release_plan.md](kv_paged_read_stage4a_block_release_plan.md)
- [docs/kv_paged_read_stage2b_non_identity_mapping_results.md](kv_paged_read_stage2b_non_identity_mapping_results.md)
- [docs/kv_paged_read_stage3b_ingraph_gather_results.md](kv_paged_read_stage3b_ingraph_gather_results.md)
- [docs/kv_lazy_block_stage_f1_design.md](kv_lazy_block_stage_f1_design.md)

本文记录 Stage 4A（block-aware madvise-only release）的实现与验证结果。Stage 4A 不做 swap、不做 backing store、不回收 active history，只释放当前不被 active `row_idx` 引用的 physical block，目标是验证 release 的 correctness 并观察 **current RSS** 是否下降。

---

## 1. Stage 4A 目标回顾

Stage 2B/3B 已端到端证明非连续寻址成立（读写共用同一张 `paged_block_table`，输出逐位等价）。Stage 4A 的目标**不再是证明寻址**，而是首次以 RSS 为目标：按 **physical block** 粒度 `MADV_DONTNEED` 释放当前不被引用的物理块，争取降低 current RSS，同时输出与 baseline 完全一致。

旧 lazy-tail 的物理连续前缀假设在 non-identity 下不成立（见 plan §3），故 Stage 4A 改以 physical block 为释放单位。

---

## 2. 实现要点

1. **新增 env `LLAMA_KV_PAGED_RELEASE=1`**：默认关闭，零回归。
2. **新增 physical block 状态机**：
   - `UNUSED`（无 active logical cell 引用，≈ 旧 `paged_block_used==0`）；
   - `RESIDENT`（已引用且物理页驻留）；
   - `RELEASED`（已 `MADV_DONTNEED`，页让出，内容不可恢复）；
   - `SWAPPED`（**预留**，本阶段不实现）。
3. **release 单位为 physical block**：判定输入是 logical（哪些不在读窗），执行对象经 `paged_resolve` 翻译为 physical block 再页对齐。
4. **release 前构造 active physical block 集合**：由本 step `row_idx`（`paged_resolve(r)`，`r ∈ [0,n_kv)`）的命中块组成。
5. **不释放 active block**：active 集合内的块一律不动（INV-2）。
6. **对 inactive 块执行 page-aligned `MADV_DONTNEED`**：UNUSED / 已离开读窗的 RESIDENT 块，页对齐内缩后 advise，置 RELEASED。
7. **写路径命中 RELEASED → RESIDENT**：append 整行覆盖（`ggml_set_rows`），页按需重新提交，不读旧值，语义安全（INV-3）。
8. **读路径命中 RELEASED 计入 `paged_release_violation`**：INV-1 守卫，必须恒为 0。
9. **新增 current RSS trace**：记录每次 release 前后 `/proc/self/statm` RSS。

---

## 3. 验证配置

> 均 `-fa on --temp 0 --seed 42`，确定性贪心；`-ctk f32 -ctv f32 -nkvo`；单序列；`LLAMA_KV_PAGED=1`。

参考 baseline sha256：`5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006`

---

## 4. 验证结果一：ctx=512 correctness smoke

| 字段 | 值 |
|---|---|
| `base_vs_release_equal` | `0`（diff 空） |
| sha256 | `5c16d81e…931e4006`（一致） |
| `paged_blocks_released` | `28` |
| `paged_blocks_released_unused` | `28` |
| `paged_block_release_bytes` | `110100480`（≈ 105.0 MiB） |
| `paged_release_violation` | `0` |
| `mapping_oob_fail` | `0` |
| `logical_to_physical_fail` | `0` |
| `graphs reused` | `62` |

判读：`paged_blocks_released=28 > 0` 且 `paged_block_release_bytes>0` → block-aware madvise **真实发生，不是 no-op**；`base_vs_release_equal=0` 且 sha256 一致 → 释放物理块后输出仍逐位等价；`paged_release_violation=0` → 没有释放任何被 active `row_idx` 引用的块（INV-1 成立）；`graphs reused=62` → 拓扑/形状未变，graph reuse 未破坏。`paged_blocks_released_unused=28` 等于总释放数 → 本轮释放的全部是 UNUSED 块（未触碰 active history，符合 Stage 4A 边界）。

---

## 5. 验证结果二：ctx=4096 RSS trace

### 5.1 correctness（大 ctx 下仍逐位一致）

| 字段 | 值 |
|---|---|
| `ctx4096_norelease_vs_release_equal` | `0`（diff 空） |
| sha256 | `5c16d81e…931e4006`（一致） |
| `paged_blocks_released` | `476` |
| `paged_blocks_released_unused` | `476` |
| `paged_block_release_bytes` | `1871708160`（≈ 1785.0 MiB） |
| `paged_release_violation` | `0` |
| `graphs reused` | `62` |

大 ctx 下释放规模显著放大（28 → 476 块，105 MiB → 1785 MiB advise），correctness 不变：no-release 与 release 输出逐位一致，sha256 同为 `5c16d81e…`。

### 5.2 current RSS trace

| 字段 | 值（KB） |
|---|---|
| `paged_block_release_rss_before_last_kb` | `9177728` |
| `paged_block_release_rss_after_last_kb` | `8264476` |
| `paged_block_release_rss_drop_last_kb` | `913252` |
| `paged_block_release_rss_drop_max_kb` | `913668` |

最后一次 release 的 current RSS 从 `9177728 KB` 降到 `8264476 KB`，**下降 `913252 KB ≈ 891.8 MiB`**（`drop_last` 与 `before-after` 一致，校验无误）；单次最大降幅 `913668 KB ≈ 892.2 MiB`。current RSS 实测确实下降。

### 5.3 peak RSS（`Maximum resident set size`）

| 配置 | Maximum resident set size |
|---|---|
| no release | `9179784 KB` |
| release | `9178336 KB` |
| 差值 | `1448 KB ≈ 1.41 MiB` |

peak RSS **基本不降**（仅 ≈ 1.4 MiB，属噪声量级）。原因：peak 发生在 release 之前 —— 物理页在被引用、提交、达到峰值之后才被 advise，advise 只能压低**此后**的 current RSS，无法回溯压低已发生的峰值。这与 lazy-tail / exact-rss 阶段对 peak 的结论一致（peak 由构造期 buffer 提交钉住）。

---

## 6. 通过标准（全部满足）

1. ✅ `base_vs_release_equal=0`、`ctx4096_norelease_vs_release_equal=0`，两 ctx 下 sha256 均为 `5c16d81e…931e4006`（release 后输出逐位等价）；
2. ✅ `paged_blocks_released` ctx512=28 / ctx4096=476，`paged_block_release_bytes` 均 > 0（madvise 真实发生）；
3. ✅ `paged_release_violation=0`（INV-1：无释放被 active row_idx 引用的块）；
4. ✅ `mapping_oob_fail=0`、`logical_to_physical_fail=0`（寻址未被破坏）；
5. ✅ `graphs reused=62`（reuse 未破坏）；
6. ✅ current RSS 实测下降 ≈ 891.8 MiB（ctx4096）；
7. ✅ peak RSS 基本不降（≈ 1.4 MiB），如实记录、不夸大。

判定核心：**`paged_blocks_released>0` 且 diff 仍空，current RSS 实测下降而 peak 不降** —— 物理块真被回收、输出仍逐位等价、收益限于 current RSS。

---

## 7. 结论

1. **Stage 4A correctness 通过**：两 ctx 下 `base_vs_release_equal=0` / `ctx4096_norelease_vs_release_equal=0`，sha256 与 baseline 完全一致，`paged_release_violation=0`、`graphs reused=62`。
2. **block-aware madvise 真实发生，不是 no-op**：`paged_blocks_released` 28（ctx512）/ 476（ctx4096），`paged_block_release_bytes` 105 MiB / 1785 MiB。
3. **current RSS 下降约 892 MiB**（ctx4096，`913252 KB`）：按物理块 `MADV_DONTNEED` 确实压低了运行期常驻内存。
4. **peak RSS 基本不降**（≈ 1.4 MiB）：peak 发生在 release 之前，advise 无法回溯压低已达到的峰值。
5. **Stage 4A 只证明 current RSS 优化，不声称 peak RSS 优化**：peak 仍由构造期内存提交钉住。
6. **仍未做 swap / backing store，也未回收 active history**：本轮释放的全部是 UNUSED 块（`paged_blocks_released_unused` 等于总数），RELEASED 内容不可恢复，符合 Stage 4A 边界。

---

## 8. 不夸大边界

可诚实声称：

- block-level madvise release 在 paged KV 下 correctness 成立（两 ctx + sha256 守门）；
- 物理块确实被回收（`paged_blocks_released>0`）；
- current RSS 实测下降约 892 MiB（ctx4096，特定配置）。

不可声称：

- 不声称降低 **peak** RSS（实测仅 ≈ 1.4 MiB，噪声量级）；
- 不声称回收了 active history（4A 只回收 UNUSED 块）；
- 不声称已实现 swap（4A 是 madvise-only，RELEASED 不可恢复）；
- 不声称该 RSS 收益在所有模型 / ctx / backend 下都成立（单配置实测，需更多覆盖）。

---

## 9. 下一步建议

1. **Stage 4B — block-aware history eviction / swap**：在预留的 `SWAPPED` 状态上实现 backing store，回收 **evicted history block**（`logical < n_kv` 但已逐出窗口）的物理页，读回时 swap-in。需重新评估 I/O 时序与 INV-1（read 前 ensure resident）。
2. **更广的 RSS 覆盖**：不同模型规模 / ctx / KV dtype（F16）下复测 current RSS，确认收益不是单配置偶然；F16 路径需先解除 Stage 3B 的 F32 gather 限制。
3. **峰值方向另议**：若要动 peak，需回到构造期 lazy-clear / 按需提交方向（Stage P2 已有 clear-frontier 基础），与 4A 的 release 正交。
