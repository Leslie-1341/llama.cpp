# Stage 4A — block-aware release (madvise-only) 设计

前置：

- [docs/kv_paged_read_stage2b_non_identity_mapping_results.md](kv_paged_read_stage2b_non_identity_mapping_results.md)
- [docs/kv_paged_read_stage3b_ingraph_gather_results.md](kv_paged_read_stage3b_ingraph_gather_results.md)
- [docs/kv_lazy_block_stage_f1_design.md](kv_lazy_block_stage_f1_design.md)
- [docs/kv_runtime_swap_stage2_summary_and_roadmap.md](kv_runtime_swap_stage2_summary_and_roadmap.md)

本文是 Stage 4A 的**正式设计文档**，只描述设计，不含实现。Stage 4A 的目标从「证明寻址正确」转为「按物理 block 安全 release，争取真实降低 current RSS」。第一版**只做 madvise-only**，不做 swap、不回收 active history。

---

## 0. 前置基线（Stage 2B 已证）

Stage 4A 建立在已经端到端验证的非连续寻址之上。`LLAMA_KV_PAGED_SHIFT=1`（non-identity ring-shift）下：

| 字段 | 值 |
|---|---|
| `base_vs_shift1_equal` | `0`（diff 空） |
| sha256 | `5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006` |
| `write_resolve_changed` | `108` |
| `row_idx_changed` | `15600` |
| `ingraph_gather_layers` | `512` |
| `mapping_oob_fail` | `0` |
| `logical_to_physical_fail` | `0` |
| `graphs reused` | `62` |

读写两端共用同一张 `paged_block_table`，寻址被打散后输出仍逐位等价。**Stage 4A 不重新证明寻址，只在此基础上回收物理块。** 任何寻址相关的 sha256 回归都意味着 Stage 4A 引入了 bug，而非寻址问题。

---

## 1. Stage 4A 目标

1. 引入 **physical-block 级**的驻留状态机，使物理块可以被安全 `MADV_DONTNEED` 释放并在未来写入前重新提交。
2. 只释放**当前 logical prefix `[0, n_kv)` 之外、不会被本 step `row_idx` 引用**的 physical block。
3. 释放路径以 **physical block 为单位**，页对齐内缩，绝不误伤仍 live 的块。
4. 默认关闭（env 门控），开启后输出与 baseline 逐位一致（sha256 守门），同时 `paged_blocks_released > 0`、current RSS 实测下降。
5. 为后续 Stage 4B（history block swap）铺好状态机与挂载点。

一句话：**Stage 4A 是「lazy-tail 的 block 化、且在 non-identity 下成立」的版本** —— 收益来源相同（回收未被引用的容量），但释放轴从「连续 cell 区间」换成「physical block」。

---

## 2. 非目标

Stage 4A **明确不做**以下事项（对齐 [roadmap §6](kv_runtime_swap_stage2_summary_and_roadmap.md) 的「禁止夸大」）：

1. **不做 swap / backing store**：RELEASED 块内容不持久化、不可恢复。swap 留到 Stage 4B。
2. **不回收 active history**：`[0, n_kv)` 内被 `row_idx` 引用的块一律不动。回收 evicted history 留到 Stage 4B。
3. **不声称降低 peak RSS**：peak 由构造期 buffer clear 钉住，Stage 4A 只针对 **current RSS**，且最终以实测为准——**允许 negative result**。
4. **不改寻址公式**：`paged_resolve` / `get_k`/`get_v` gather / `n_kv` 计算保持 Stage 2B/3B 已验证的形态。
5. **不扩展支持矩阵**：沿用首版限定 `n_stream==1 && !v_trans && F32 K/V`，不满足则 release 整体禁用，行为回退。
6. **不引入新 ggml op、不动 attention kernel、不动 graph 拓扑**：release 只改物理页驻留，不改张量形状/节点数，graph reuse 必须保持。

---

## 3. 为什么旧 lazy-tail 在 non-identity paged KV 下不安全

lazy-tail（`madvise_tail`，[src/llama-kv-cache.cpp:2316](../src/llama-kv-cache.cpp#L2316)）建立在**物理连续前缀假设**上：

> 所有有效 KV 内容驻留在物理 `[0, n_kv)` 连续前缀；尾部 `[GGML_PAD(n_kv,256), kv_size)` 永远在读窗外，可安全 advise。

non-identity 下该假设被打破。`paged_resolve(cell)` 把 logical block 经 ring-shift 映射到非连续 physical block（block 0 固定、其余环形平移，[src/llama-kv-cache.cpp:1628-1631](../src/llama-kv-cache.cpp#L1628)）：

- 有效内容**散布在 `[0, kv_size)` 各处**，不再聚在连续前缀；
- 物理尾部可能藏着 live block（某个 `r < n_kv` 映射过去）；
- 物理前缀可能有空洞（对应的 logical 尚未进入读窗）。

因此现有代码已在 [src/llama-kv-cache.cpp:2321](../src/llama-kv-cache.cpp#L2321) 用 `paged_non_identity_enabled` **强制禁用** lazy-tail，并打印：

```
KV lazy-tail disabled because paged non-identity mapping breaks physical-prefix assumption
```

**根因**：lazy-tail 的释放单位是「连续 cell 区间」，而 non-identity 的安全释放单位必须是「physical block」。Stage 4A 要换的正是这条轴 —— 不再问「哪段连续尾部安全」，而是问「哪些 physical block 当前没有任何 `r < n_kv` 映射进来」。

---

## 4. 为什么 release 单位必须是 physical block

1. **madvise 操作物理页**。`MADV_DONTNEED` 作用于 host 虚拟地址区间，只有 physical block 对应一段**连续的 host 字节区间**：`phys_cell = physical_block * block_size + offset`（[src/llama-kv-cache.cpp:1700](../src/llama-kv-cache.cpp#L1700)）。
2. **logical block 在物理上是打散的**。non-identity 下一段连续 logical 区间映射到非连续 physical 块；按 logical 区间 advise 会误伤其它 logical 映射到的同一物理页。
3. **判定与执行分离**：「能否释放」可以从 logical 视角推导（哪些 logical 不在 `[0, n_kv)`），但**执行 release 的对象必须翻译成 physical block** 再页对齐。

> 设计规则：**release 的判定输入是 logical，执行对象是 physical**。两者之间唯一的翻译器是 `paged_resolve` / `paged_block_table`，不得另起一套映射。

---

## 5. block 状态机

Stage 4A 在现有的物理块占用账本（`paged_block_used` / `paged_free_list`，[src/llama-kv-cache.h:463-464](../src/llama-kv-cache.h#L463)）之上，新增一个 **physical-block 级**状态数组 `paged_block_state[paged_n_blocks]`。

注意：cell 级状态机 `llama_kv_cell_state{UNTOUCHED,RESIDENT,SWAPPED,RELEASED}` 已存在（[src/llama-kv-cells.h:31-36](../src/llama-kv-cells.h#L31)），属于 **exact-swap 老路径**（`kv_swap_mode::exact` 门控）。Stage 4A **不复用、不混用**它，新建 block 级状态机，避免两套语义打架。

### 5.1 状态定义

| 状态 | 含义 | 本阶段 |
|---|---|---|
| `FREE` / `UNUSED` | 当前没有 active logical cell 引用该物理块 | 实现 |
| `RESIDENT` | 已被某 logical cell 引用且物理页驻留，可被 gather/写入 | 实现 |
| `RELEASED` | 已 `MADV_DONTNEED`，物理页可被内核回收；映射仍在，但内容**已不可恢复**（下次访问得零页） | 实现 |
| `SWAPPED` | 内容已写入 backing store、物理页释放，可 swap-in 恢复 | **预留，本阶段不实现** |

### 5.2 状态转移

```
FREE ──(logical cell 首次映射进来, paged_note_cells)──▶ RESIDENT
RESIDENT ──(logical 离开 [0,n_kv) 且不会被本 step row_idx 引用 + 页对齐成立)──▶ RELEASED   [madvise]
RELEASED ──(未来写入命中该块, set_input_k/v_idxs 写前)──▶ RESIDENT                       [页按需重新提交, 内容视为新写入]
RELEASED ──(若 logical 永不再映射)──▶ 保持 RELEASED
（SWAPPED 相关转移留空：Stage 4B）
```

关键：`RELEASED → RESIDENT` 只在**未来新写入**前发生，**绝不用于读回旧历史**。因为 `MADV_DONTNEED` 后旧 KV 不可恢复（设计边界 §5）。

### 5.3 命名说明：`FREE` 不等于 OS 内存未占用

`FREE` 这个词容易误解为「该物理内存已归还给 OS」。实际语义是：

> **当前没有 active logical cell 引用该物理块**（≈ 现有 `paged_block_used[i]==0`）。

它**不保证**底层物理页已被回收 —— 一个块可以是 `FREE` 但其页仍被进程占用（从未 advise 过）。真正「页已让出」的状态是 `RELEASED`。

**命名建议**：为避免误导，推荐用 **`UNUSED`** 或 **`UNREFERENCED`** 替代 `FREE`，更准确表达「无 active 引用」而非「内存已释放」。本文后续以 `UNUSED` 为主名，`FREE` 作别名。语义轴明确为：

- **引用轴**：`UNUSED`（无 logical 引用） vs `RESIDENT`（有引用且页在）；
- **驻留轴**：`RELEASED`（页已 advise 让出） vs 其它（页在）。

---

## 6. Stage 4A 关键不变式

实现与验证都以以下不变式为准，任一被破坏即为致命 bug：

- **INV-1（读安全）**：对所有 `r < n_kv`，`paged_resolve(r)` 所在 physical block 必须为 `RESIDENT`。
- **INV-2（不释放 live）**：任何会被本 step 或未来 step `row_idx` 引用的 physical block 都不得 release。
- **INV-3（写前提交）**：release 后若未来写入命中该 block，必须先 `RELEASED → RESIDENT`（ensure resident）再写。
- **INV-4（不可恢复）**：`RELEASED` block 的旧内容不可读回；只能作为「待新写入」的空块重新进入 RESIDENT。
- **INV-5（零回归）**：release 关闭时行为与 Stage 2B/3B 逐位一致；开启时输出仍与 baseline sha256 一致。

INV-1 是核心安全网：只要它成立，graph 内 `ggml_get_rows(row_idx)` 读到的每一行都来自 RESIDENT 块，不会撞上零页。

---

## 7. 挂载点

所有挂载点都在 **CPU 端、`graph_compute` 之前**的 set_input / apply 阶段，时序与 Stage 3B 的 row_idx 填值一致（填值在 compute 前，gather 在 compute 内）。

### 7.1 读路径挂载点 — `set_input_paged_row_idx`

位置：[src/llama-kv-cache.cpp:2891](../src/llama-kv-cache.cpp#L2891)（经 [src/llama-graph.cpp:461](../src/llama-graph.cpp#L461) 在 set_input 阶段调用）。

每步填 `row_idx[r] = paged_resolve(r)` for `r ∈ [0, n_kv)` 之前/之时：

- 对 `paged_resolve(r)` 命中的 physical block，断言其为 `RESIDENT`（INV-1）；
- 若命中 `RELEASED` 块 → 这是逻辑 bug（不该发生，因为 `r < n_kv` 的块都该 RESIDENT），计 `paged_release_violation`（必须恒为 0），并 fail-safe 翻回 RESIDENT 以免读零页。

> Stage 4A 设计上 `set_input_paged_row_idx` 不主动触发 swap-in（无 backing store）；它只做 violation 守卫。真正的「确保 resident」由 release 判定保证不释放 live 块。

### 7.2 写路径挂载点 — `set_input_k_idxs` / `set_input_v_idxs`

位置：[src/llama-kv-cache.cpp:2848](../src/llama-kv-cache.cpp#L2848) / [src/llama-kv-cache.cpp:2867](../src/llama-kv-cache.cpp#L2867)。

`paged_write_resolve(cell)` 得到 phys 之后、写入 `data[]` 之前：

- 计算目标 physical block；若为 `RELEASED` → `RELEASED → RESIDENT`（INV-3），计 `paged_block_ensure_released`；
- madvise-only 下无需 I/O：页会在下次访问时按需重新提交为零页，而 append 写入是整行覆盖（`ggml_set_rows`），不依赖旧内容，故语义安全；
- 计 `paged_block_ensure_calls`。

### 7.3 release 挂载点 — 新增 `paged_release_blocks(n_kv)`

位置：每步 `apply_ubatch` 之后的钩子序列，紧邻现有 `madvise_tail` 调用 [src/llama-kv-cache.cpp:3970](../src/llama-kv-cache.cpp#L3970)（即 `swap_out_window` / `ensure_resident` / `clear_frontier_advance` / `madvise_tail` 同一时机）。

**不复用** `madvise_tail` 的连续 tail 假设，只复用它的 page-align 内缩逻辑（见 §8）。`n_kv` 已知后调用，不改 `n_kv` / row_idx 长度 / 节点拓扑。

---

## 8. page-aligned madvise 策略

复用 `madvise_tail` 的页对齐内缩思想（[src/llama-kv-cache.cpp:2366-2382](../src/llama-kv-cache.cpp#L2366)），但作用对象从「连续 tail」改为「单个 physical block 的字节区间」：

对每个待 release 的 physical block，在每层 K/V 张量上：

1. 计算块的字节区间 `[block_lo_byte, block_hi_byte) = [phys_block*block_size*row_size, (phys_block+1)*block_size*row_size)`（绝对 host 地址，非相对 offset）。
2. **向内对齐**：`a_start = ceil(lo, page)`，`a_end = floor(hi, page)`。
3. **跳过条件**：
   - `a_end <= a_start`（块字节范围不足一页或对齐后为空）→ 计 `paged_block_release_skip_unaligned`；
   - 块与相邻 **live（RESIDENT）** 块共享同一物理页 → 跳过该页边界，绝不 advise 跨界页（计 `paged_block_release_skip_live`）。
4. 对内缩后的区间 `madvise(a_start, len, MADV_DONTNEED)`；成功计 `paged_blocks_released` / `paged_block_release_bytes`，失败计 `paged_block_release_fail`。

> 页对齐内缩是 INV-2 的物理实现：宁可少释放（保守内缩），绝不误伤 live 页。`block_size * row_size` 不足一页时该块整体跳过——这也是收益受块粒度限制的来源（§11 风险）。

---

## 9. 新增统计字段

挂在 `paged_log_stats`（[src/llama-kv-cache.cpp:1851](../src/llama-kv-cache.cpp#L1851)），全部默认 0，release 关闭时保持 0：

| 字段 | 含义 | 守门期望 |
|---|---|---|
| `paged_block_release_enabled` | release 是否启用 | env=1 时为 1 |
| `paged_block_release_calls` | `paged_release_blocks` 调用次数 | > 0（启用后） |
| `paged_blocks_released` | 成功 advise 的 physical block 次数 | **> 0**（生效证据） |
| `paged_block_release_bytes` | 累计 advise 字节 | **> 0** |
| `paged_block_release_skip_live` | 因与 live 块共页而跳过 | 记录，可 > 0 |
| `paged_block_release_skip_unaligned` | 因不足一页/对齐为空而跳过 | 记录，可 > 0 |
| `paged_block_release_fail` | `madvise` 返回非 0 | **= 0** |
| `paged_block_ensure_calls` | 写前 ensure-resident 检查次数 | ≥ 0 |
| `paged_block_ensure_released` | 写前把 RELEASED 翻回 RESIDENT 次数 | 记录 |
| `paged_release_violation` | 读路径命中 RELEASED 块（INV-1 破坏） | **必须 = 0** |

`paged_release_violation` 是最关键的安全计数：非 0 即说明释放了仍被读取的块。

---

## 10. correctness 验证标准

> 配置沿用 `-fa on --temp 0 --seed 42 -ctk f32 -ctv f32 -nkvo`，确定性贪心，单序列。

1. **零回归**：`LLAMA_KV_PAGED_RELEASE=0` 时 `base_vs_off_equal=0`，sha256 = `5c16d81e…931e4006`。
2. **开启逐位一致**：`LLAMA_KV_PAGED_RELEASE=1` 时 `base_vs_release_equal=0`，sha256 与 baseline 完全一致。
3. `paged_release_violation = 0`（INV-1 成立）。
4. `row_idx_fail = 0`（无 INVALID 物理行）。
5. `mapping_oob_fail = 0`、`logical_to_physical_fail = 0`（寻址未被破坏）。
6. `graphs reused = 62`（拓扑/形状不变，reuse 未破坏）。
7. 与 non-identity（`LLAMA_KV_PAGED_SHIFT=1`）同开仍 `base_vs_release_shift1_equal=0`（release 在打散映射下也安全）。

判定核心：**`paged_blocks_released > 0` 且 diff 仍空** —— 物理块真被回收，输出仍逐位等价。

---

## 11. RSS 验证标准

1. `paged_blocks_released > 0`（确实回收了物理块，否则是 no-op）。
2. `paged_block_release_bytes > 0`。
3. **current RSS**（`/proc/self/statm`，复用 `get_current_rss_kb`）：release 开启后低于关闭时。建议大 ctx + F32 KV（F32 行宽是 F16 两倍，收益更可见）下测量 before/after。
4. **peak RSS 可能不降**：peak 由构造期 buffer clear 钉住，必须如实记录，不得声称 Stage 4A 降低 peak。
5. **允许 negative result**：若收益被块粒度 / backend 分配粒度淹没，如实记录，并据此决定是否转 Stage 4B —— 绝不声称未验证的收益。

---

## 12. 风险表

| 风险 | 等级 | 触发条件 | 缓解 |
|---|---|---|---|
| release 仍被未来 row_idx 引用的块 | **致命** | `n_kv` 增长使新 logical 映射到已 RELEASED 块，gather 读零页且静默 | 只 release `logical_block > (n_kv-1)/block_size` 的块；INV-1 + `paged_release_violation=0`；sha256 守门 |
| 写入 RELEASED 块读到零值 | 高 | append 命中刚回收块 | 写前 `RELEASED→RESIDENT`（INV-3）；确认 `ggml_set_rows` 整行覆盖不读旧值 |
| 页对齐误伤相邻 live block | 高 | 块字节区间 < 1 页或与邻块共页 | 绝对地址向内对齐（§8）；共页则 `skip_live`；不足一页整块跳过 |
| current RSS 不降 | 中 | 块太小 / 小 ctx / backend 分配粗 | 大 ctx + F32 实测；negative result 如实记录 |
| graph reuse 破坏 | 中 | row_idx 形状/节点数随 release 变化 | release 只改页驻留，不改 `n_kv`/row_idx 长度/拓扑；守门 `graphs reused=62` |
| 非支持布局误启用 | 中 | 多序列 / v_trans / 量化 | 复用 `paged_ingraph_gather_supported`（[src/llama-kv-cache.cpp:1827](../src/llama-kv-cache.cpp#L1827)）门控，不满足则 release 禁用 |
| 与 exact-swap 老路径状态混淆 | 中 | 误复用 `llama_kv_cell_state` | 新建独立 `paged_block_state`，不碰 `ensure_resident`/`swap_out_window` |

---

## 13. Stage 4A 最小实现路线

env 门控 `LLAMA_KV_PAGED_RELEASE`，默认关闭，零回归：

1. `paged_build_block_table`（[src/llama-kv-cache.cpp:1600](../src/llama-kv-cache.cpp#L1600)）：初始化 `paged_block_state[]` 全 `UNUSED`。
2. `paged_note_cells`（[src/llama-kv-cache.cpp:1650](../src/llama-kv-cache.cpp#L1650)）：块首次被映射时置 `RESIDENT`（扩展现有占用逻辑为状态机）。
3. 新增 `paged_release_blocks(n_kv)`，挂 [src/llama-kv-cache.cpp:3970](../src/llama-kv-cache.cpp#L3970)：计算 `max_logical = (n_kv-1)/block_size`；对 `logical > max_logical` 的 logical 块，其 physical 块若 `RESIDENT` 则页对齐 `MADV_DONTNEED` → `RELEASED`（§8）。
4. 读守卫：`set_input_paged_row_idx`（[src/llama-kv-cache.cpp:2891](../src/llama-kv-cache.cpp#L2891)）命中 `RELEASED` → `paged_release_violation++`，fail-safe 翻回 RESIDENT。
5. 写守卫：`set_input_k_idxs` / `set_input_v_idxs`（[src/llama-kv-cache.cpp:2848](../src/llama-kv-cache.cpp#L2848) / [2867](../src/llama-kv-cache.cpp#L2867)）命中 `RELEASED` → `RESIDENT`（INV-3）。
6. `paged_log_stats`（[src/llama-kv-cache.cpp:1851](../src/llama-kv-cache.cpp#L1851)）：追加 §9 统计。
7. 头文件（[src/llama-kv-cache.h](../src/llama-kv-cache.h)）：声明 `paged_block_state`、env 门控、统计字段、`paged_release_blocks`。

**本轮绝对不碰**：`paged_resolve`/`paged_write_resolve`、`get_k`/`get_v` gather 公式、`ensure_resident`/`swap_out_window`/`madvise_swapped_runs`（exact-swap 老路径）、`madvise_tail` 本体、attention kernel / ggml op / `find_slot` / `n_kv` 计算。

> 本质：Stage 4A = lazy-tail 的 block 化 + non-identity 安全版。收益来源与 lazy-tail 相同（回收未被引用容量），但释放轴为 physical block，且为 Stage 4B（回收 evicted history）准备好状态机。

---

## 14. 后续展望

- **Stage 4B — history block swap**：在 `SWAPPED` 状态上实现 backing store，回收 evicted history block（`logical < n_kv` 但已逐出窗口）的物理页，读回时 swap-in。需重新评估 I/O 时序与 INV-1。
- **Stage 4C — block-aware prefetch / smarter eviction**：基于访问模式预取即将进入读窗的块，配合 sink+recent 等非连续保留策略，把物理 residency 与实际读取集合对齐，逼近 PagedAttention 终态。

---

## 15. 不夸大边界

对齐 [roadmap §6](kv_runtime_swap_stage2_summary_and_roadmap.md)，Stage 4A 可诚实声称 / 不可声称：

可声称（待实测验证后）：

- block-level madvise release 在 non-identity paged KV 下 correctness 成立（sha256 守门）；
- 物理块确实被回收（`paged_blocks_released > 0`）。

不可声称：

- 不声称降低 **peak** RSS；
- 不声称一定降低 current RSS（以实测为准，允许 negative result）；
- 不声称回收了 active history（4A 只回收未引用块）；
- 不声称已实现 swap（4A 是 madvise-only，RELEASED 不可恢复）；
- 不声称已接近 PagedAttention 终态。
