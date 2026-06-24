# Stage 4C-3 — idle-request / idle-sequence KV swap 实施计划

配套设计：[docs/kv_paged_routeb_idle_request_swap_design.md](kv_paged_routeb_idle_request_swap_design.md)

本文把 Stage 4C-3 拆成可并行交付的原子任务，适合后续分给 Codex 实现。**本轮不实现，只规划。**
每个任务给出：目标 / 涉及文件 / 允许修改函数 / 禁止修改函数 / 新增统计 / smoke / 回滚标准 / 是否需 Claude Code 审查。

总体约束（适用于所有任务）：

- 全程要求 `n_stream==1 && !v_trans`，且仅当 `LLAMA_KV_PAGED_IDLE_SWAP=1` 才改变行为；否则整条路径 no-op。
- 任何任务的回归底线：单序列 ctx=512 / ctx=4096 smoke 与 baseline **逐位一致 / sha256 相同**。
- 禁止改 attention kernel、`build_*` 图构造、公开 API、`paged_swap_out_block` / `paged_swap_in_block` 的换出/换回内核。
- 禁止 commit / push（由人工审查后决定）。

推荐路线（按依赖顺序）：**A → B → C → D**。A、C 的子任务内部可并行；B 依赖 A 的 bitmap；D 依赖 B。

---

## 路线 A — seq/block ownership telemetry（先行，零行为漂移）

### A1. per-seq idle 计数

- **目标**：维护 `seq_last_active_step[seq_id]` 与全局 step；提供 `idle_steps(seq)` 查询。`seq_rm` 时重置对应 seq。
- **涉及文件**：[src/llama-kv-cache.h](../src/llama-kv-cache.h)、[src/llama-kv-cache.cpp](../src/llama-kv-cache.cpp)。
- **允许修改函数**：`apply_ubatch`（刷新本 ubatch seq 的 last_active）、`seq_rm`（重置）；新增成员
  `std::vector<uint64_t> paged_seq_last_active_step`、`uint64_t paged_idle_step`。
- **禁止修改函数**：`paged_resolve` / `paged_write_resolve` / `find_slot` / 任何 `build_*` / swap 内核。
- **新增统计**：`paged_idle_step`、可选 per-seq idle 直方图（trace only）。
- **smoke**：ctx=512 单序列，开关开/关均与 baseline sha256 一致（单序列下计数变动但不影响输出）。
- **回滚标准**：sha256 与 baseline 不一致，或 `seq_rm` 后计数未重置。
- **Claude Code 审查**：否（机械计数）。

### A2. block→seq bitmap 构建

- **目标**：新增 `std::vector<seq_set_t> paged_block_seq`（长度 `paged_n_blocks`），每步在 `paged_note_cells`
  之后重建：遍历 `v_cells[0]` used cell，经 `paged_block_table` 映射到 physical block，OR 其 `seq` bitset。
- **涉及文件**：`src/llama-kv-cache.h` / `.cpp`、读取 [src/llama-kv-cells.h](../src/llama-kv-cells.h) 的 `seq_has`/`seq` 访问器。
- **允许修改函数**：新增 `paged_build_block_seq()`；在 `apply()` 中 `paged_note_cells` 之后调用（trace 开时）。
- **禁止修改函数**：`paged_note_cells` 内核逻辑、`paged_build_block_table`、swap 内核。
- **新增统计**：`paged_block_seq_rebuilds`、`paged_blocks_multi_seq`（混 seq 块数）。
- **smoke**：单序列下所有块 `block_seq` 仅含 seq 0；sha256 与 baseline 一致。
- **回滚标准**：bitmap 与逐 cell `seq_has` 不一致，或行为漂移。
- **Claude Code 审查**：**是**（logical/physical 映射方向是最大出错点）。

### A3. cold-candidate telemetry（只统计不换出）

- **目标**：每步计算 `active_seq_set` / `idle_seq_set`，统计 cold candidate（`block_seq ⊆ idle` 且非空）、
  mixed-active（含 active）块数，经 `LLAMA_KV_PAGED_TRACE` 输出。**不换出。**
- **涉及文件**：`src/llama-kv-cache.cpp`、`paged_trace_emit_step` / `paged_log_stats`。
- **允许修改函数**：`paged_trace_emit_step`（追加字段）；新增 `paged_compute_idle_sets()`。
- **禁止修改函数**：`set_input_paged_row_idx` 的 row_idx 写入逻辑、swap 内核。
- **新增统计**：`paged_idle_cold_candidates`、`paged_idle_skip_mixed_active`、`paged_idle_seq_count`。
- **smoke**：ctx=4096 单序列 trace，cold_candidates 趋势与 4C-1/4C-2B 仿真一致；sha256 与 baseline 一致。
- **回滚标准**：sha256 不一致，或统计与 §3 定义不符。
- **Claude Code 审查**：否（A2 通过后此项为纯统计）。

### A3.5. read-window membership telemetry（**关键 gate，不换出**）

- **目标**：实测 cold block 是否仍落在 padded physical read window（row_idx 覆盖范围）里。**只统计不换出。**
  这是判断「当前 graph 是否支持 request-level physical read isolation」、能否进入 B1 的硬 gate（见设计 §4.1 / §7.3）。
- **涉及文件**：`src/llama-kv-cache.cpp`、`set_input_paged_row_idx` / `paged_trace_emit_step` / `paged_log_stats`。
- **允许修改函数**：`set_input_paged_row_idx`（仅收集本 step row_idx 覆盖的 physical block 集合到一个临时
  `read_window_blocks`，不改 row_idx 写入的值）；`paged_trace_emit_step`（追加字段）；新增
  `paged_compute_read_window_blocks()`。
- **禁止修改函数**：row_idx 张量的实际写入值（`data[r] = phys`）、`paged_resolve`、swap 内核、attention kernel。
- **新增统计（设计 §7.3 全集）**：
  - `paged_idle_cold_candidates` — `block_seq` 非空 ∧ 不含 active seq ∧ 只含 idle seq；
  - `paged_idle_read_window_blocks` — 本 step row_idx / physical read window 覆盖的 block 数；
  - `paged_idle_cold_in_read_window` — cold candidate ∩ read window（**不可换出**）；
  - `paged_idle_cold_not_in_read_window` — cold candidate ∖ read window；
  - `paged_idle_safe_swap_candidates` — `cold_not_in_read_window ∧ state==RESIDENT`（B1 唯一可换集合）；
  - `paged_idle_skip_mixed_active` — 含 active seq 被跳过数。
- **smoke / 通过标准**：
  1. 单序列 ctx=512 / ctx=4096 sha256 **完全一致**（纯统计，零行为漂移）；
  2. synthetic 多 seq trace 能输出 `cold_candidate` / `cold_in_read_window` / `safe_swap_candidate` 三组数；
  3. **gate**：若 `paged_idle_safe_swap_candidates` 长期为 0（cold block 全在 read window 里），
     **不得进入 B1**——说明 graph 尚无物理读隔离，应转向设计 §10 延伸研究（seq-aware row_idx / masked dummy remap）。
- **回滚标准**：sha256 漂移，或 read_window 集合与 row_idx 实际覆盖不符。
- **Claude Code 审查**：**是**（read-window 判定是 B1 安全条件第 3 条的数据来源，错判直接导致 thrashing）。

---

## 路线 B — idle-seq swap-out（首个行为改变）

### B1. idle 换出选择器

- **前置 gate**：A3.5 实测 `paged_idle_safe_swap_candidates > 0` 持续出现，否则**不得开工**（见设计 §4.1 / §7.3）。
- **目标**：新增 `paged_swap_out_idle_seqs(uint32_t n_kv)`：仅对满足设计 §4 + §4.1 全部安全条件的 block 调
  `paged_swap_out_block`。`LLAMA_KV_PAGED_IDLE_SWAP=1` 时**替换** apply 流程中 `paged_swap_out_window` 的调用。
- **涉及文件**：`src/llama-kv-cache.cpp` / `.h`。
- **允许修改函数**：新增 `paged_swap_out_idle_seqs`；`llama_kv_cache_context::apply`
  （[src/llama-kv-cache.cpp:4706-4710](../src/llama-kv-cache.cpp#L4706-L4710) / 4738-4739）的 swap 分流；env 解析。
- **禁止修改函数**：`paged_swap_out_block` / `paged_swap_in_block`（换出/换回内核不动）、`paged_swap_out_window`
  （保持原状供开关关闭时使用）、读写 resume 钩子、row_idx 写入。
- **安全条件（扩展自原 `block_seq ∩ active == ∅ && block_seq.any()`）**——换出前必须**全部成立**：
  - `cold_candidate`：`block_seq[b].any() && block_seq[b] ∩ active_seq_set == ∅`（只含 idle seq）；
  - `!in_physical_read_window`：block 不在本 step row_idx 覆盖范围（复用 A3.5 的 `read_window_blocks`）；
  - `state == RESIDENT`。
  即 `cold_candidate && !in_physical_read_window && state==RESIDENT`。
  - 若 `cold_candidate && in_physical_read_window`：**必须 skip**，计入 `paged_idle_cold_in_read_window`，
    **不得 swap-out**（否则下一步被 `paged_check_read_resident` 立即换回 → 4B-RSS thrashing）。
  - 若含 active seq：skip，计入 `paged_idle_skip_mixed_active`。
- **新增统计**：`paged_idle_blocks_swapped_out`、`paged_idle_cold_in_read_window`、`paged_idle_skip_mixed_active`、`paged_idle_seq_swap_enabled`。
- **smoke**：(1) 单序列开关开 → idle 永不触发，sha256 与 baseline 一致；(2) synthetic 双 seq 错峰 → 仅
  `safe_swap_candidate` 块进 SWAPPED，mixed 块与仍在 read window 的 cold 块始终 resident。
- **回滚标准**：单序列 sha256 漂移；任何含 active seq 的块被换出；**任何仍在 read window 的块被换出**（thrashing 复现）。
- **Claude Code 审查**：**是**（安全条件是 exact 的全部依据）。

### B2. resume 换回闭合验证

- **目标**：确认 idle seq resume 时经现有读/写钩子换回，计数闭合（swap_out == swap_in）。**不新增换回路径。**
- **涉及文件**：`src/llama-kv-cache.cpp`（仅加 idle 专属计数）。
- **允许修改函数**：`paged_check_read_resident` / `paged_ensure_write_resident`（仅追加 idle resume 计数，不改换回逻辑）。
- **禁止修改函数**：`paged_swap_in_block`、row_idx 写入。
- **新增统计**：`paged_idle_resume_swap_in`、`paged_idle_thrash_in_4_steps`（目标 0）。
- **smoke**：synthetic 双 seq，`idle_blocks_swapped_out == idle_resume_swap_in`，thrash=0，resume 后输出与全程 resident 基线逐位一致。
- **回滚标准**：计数不闭合，thrash>0，或 resume 输出错。
- **Claude Code 审查**：**是**（exact 守门）。

---

## 路线 C — resume swap-in / optional prefetch（可选，v1 之后）

### C1. resume I/O 尖峰度量

- **目标**：度量 resume 单步换回字节 / 块数（对照 4C-2B 的最高 336 MiB），不改行为。
- **允许修改函数**：resume 钩子内的计数；`paged_log_stats`。
- **新增统计**：`paged_idle_resume_bytes_max` / `_last`。
- **smoke**：synthetic 双 seq 采到非零尖峰；sha256 不变。
- **回滚标准**：行为漂移。
- **Claude Code 审查**：否。

### C2.（可选，明列 non-goal 的延伸）异步 prefetch

- **状态**：v1 **不做**。仅在 C1 量化后、若延迟成为瓶颈再立项。本计划不展开任务细节。

---

## 路线 D — 真实多请求 demo

### D1. llama-parallel / server 集成 smoke

- **目标**：在 `llama-parallel` 或 server 多 slot 下跑 idle swap，采集真实 idle 分布，与 4C-2B 仿真对照；
  验证 `seq_rm` slot 复用时 idle 计数正确重置。
- **涉及文件**：测试脚本（`scripts/`）、可能少量 server slot↔seq 映射只读探针。
- **允许修改函数**：仅测试 / 脚本；KV 内核不动。
- **禁止修改函数**：所有 KV swap 内核与选择器（D 阶段只验证不改实现）。
- **新增统计**：复用 B/C 统计，新增 demo 级聚合脚本。
- **smoke**：多 slot 下 idle seq 块被换出、active 块保留、整体输出与单跑基线一致（可复现子集上 sha256）。
- **回滚标准**：多 seq 下输出错，或 slot 复用导致误换。
- **Claude Code 审查**：**是**（首次真实多 seq 暴露 §9 兼容性风险）。

---

## 任务依赖图

```
A1 ─┐
A2 ─┼─► A3 ─► A3.5 ─►[gate: safe_swap_candidates>0]─► B1 ─► B2 ─► (C1) ─► D1
A1 ─┘                                                  ▲
A2 ──────────────────────────────────────────────────┘   (B1 依赖 A2 bitmap + A1 idle 计数 + A3.5 read-window 判定)
```

并行建议：A1 / A2 可并行；A3 等 A2；A3.5 等 A3（共用 idle sets）；**B1 等 A3.5 的 gate 通过**（`safe_swap_candidates>0`）；
B2 紧随 B1；C1 与 D1 准备工作可在 B2 通过后并行启动。

**硬 gate**：若 A3.5 在 multi-seq smoke 中 `paged_idle_safe_swap_candidates` 长期为 0，则当前 graph 无
request-level physical read isolation，**整条 B 路线暂停**，转设计 §10 延伸研究（seq-aware row_idx /
masked dummy remap）。这是本计划相对初版**改变推荐路线**的地方：B1 不再是 A 之后的必然下一步，而是被 A3.5 的实测 gate 守住。

每个任务结束都必须跑「单序列 ctx=512 sha256 == baseline」回归，作为零漂移底线。
