# Stage 4C-3 — idle-request / idle-sequence KV swap 设计

前置：

- [docs/kv_paged_routeb_multi_request_sim.md](kv_paged_routeb_multi_request_sim.md) — multi-request / idle 仿真（结论：idle 请求是真冷数据，exact，平均省 ~54%，峰值降不了）
- [docs/kv_paged_trace_sim_stage4c1_results.md](kv_paged_trace_sim_stage4c1_results.md) — 单请求 exact 无安全冷块
- [docs/kv_paged_read_stage4b_rss_results.md](kv_paged_read_stage4b_rss_results.md) — swap+madvise correctness 闭环（复用其换出/换回机制）

本文是 Stage 4C-3 的**设计文档**。本轮**只做源码分析与设计，不改源码、不 build、不跑实验、不 commit**。落地拆分见
[docs/kv_paged_routeb_idle_request_swap_implementation_plan.md](kv_paged_routeb_idle_request_swap_implementation_plan.md)。

结论先行：llama.cpp 没有 request 抽象，但 paged 路径强制 `n_stream==1`（unified cache），所有序列共享一份
`v_cells[0]`，因此**用 `seq_id` 近似 request 是唯一可行且自洽的做法**。idle-seq swap 的换出/换回**机制可以
100% 复用 Stage 4B-RSS**（`paged_swap_out_block` / `paged_swap_in_block` + 读写路径 ensure 钩子），
真正要新增的只有一个**安全的换出选择器**：替换 `paged_swap_out_window` 里制造 thrashing 的强制 sink+window
启发式。因为 unified cache 下 block 普遍混有多个 seq，第一版的核心安全约束是「**只要 block 含任何 active
seq 的 cell 就必须 resident**」。

---

## 1. llama.cpp 是否存在 request 级抽象？

**没有 request 抽象。** 调度的最小单位是 `seq_id`（`llama_seq_id`，范围 `[0, LLAMA_MAX_SEQ)`，上限
`n_seq_max`）。一次 `llama_decode` 的 `llama_batch` 里每个 token 携带一组 `seq_id`，KV cache 按 token 写入
cell。没有「请求开始 / 请求结束 / 请求 idle」这一层语义；请求的边界由上层（server slot）通过
`seq_rm` / `seq_cp` / `seq_keep` 间接表达。

### 1.1 用 seq_id 近似 request

可以，且是唯一自洽选项。一个 server slot 对应一个 `seq_id`，slot 的生命周期就是请求的生命周期。Stage 4C-3
把「**idle request**」定义为「**idle sequence**」：某 `seq_id` 在最近若干 decode step 中没有出现在 ubatch 的
seq 集合里。这与 4C-2B 仿真里的 per-request idle 阈值一一对应。

### 1.2 KV cell 如何记录 seq ownership

见 [src/llama-kv-cells.h:574-577](../src/llama-kv-cells.h#L574-L577)：

```cpp
using seq_set_t = std::bitset<LLAMA_MAX_SEQ>;
std::vector<seq_set_t> seq;   // seq[i] = 占用第 i 个 cell 的序列位集
```

每个 cell 持有一个 `std::bitset<LLAMA_MAX_SEQ>`。访问器：

- `seq_has(i, seq_id)` — cell i 是否属于 seq_id（[src/llama-kv-cells.h:385](../src/llama-kv-cells.h#L385)）；
- `seq_count(i)` — cell i 被多少个 seq 占用（[src/llama-kv-cells.h:377](../src/llama-kv-cells.h#L377)）；
- `seq_get(i)` — 当 `count==1` 时返回唯一 seq（断言 count==1，[src/llama-kv-cells.h:404](../src/llama-kv-cells.h#L404)）。

### 1.3 一个 KV block 是否可能混多个 seq_id？

**会，而且在 unified cache 下是常态。** paged 路径强制 `n_stream==1`（见
`paged_swap_out_block` 的 `n_stream != 1` 早退，[src/llama-kv-cache.cpp:1834](../src/llama-kv-cache.cpp#L1834)），
即所有序列共享 `v_cells[0]`。`find_slot` 把不同 seq 的 token 顺序填进同一 ring buffer，故一个
`paged_block_size`（默认 16）大小的 physical block 完全可能同时含 seq 0 和 seq 1 的 cell。

这是本设计的**核心难点**：block 不是 per-seq 的。这条事实直接决定了第 4 节的安全条件和第 6 节的最小范围。

---

## 2. idle sequence 如何定义

### 2.1 度量口径

第一版用 **decode step 计数**，不用 wall time、不用显式 API：

- 维护 per-seq 计数 `seq_last_active_step[seq_id]`，在 `apply_ubatch` 时把本 ubatch 出现的每个
  seq 的值刷新为当前全局 step；
- `idle_steps(seq) = current_step - seq_last_active_step[seq]`；
- `idle(seq) := idle_steps(seq) >= LLAMA_KV_PAGED_IDLE_THRESHOLD`。

理由：step 计数确定、可复现、与现有 trace step（`paged_trace_step`）同源，便于 sha256 守门和与
4C-2B 仿真的 `idle_swap_T` 直接对照。wall time 不可复现、API 标记需要改公开接口（non-goal）。

### 2.2 第一版能力边界（与 mechanism 复用对齐）

第一版**只在以下条件成立时启用** idle swap，否则整条路径 no-op（与 `paged_swap_out_block` 现有早退完全一致）：

- `n_stream == 1`（unified cache）；
- `!v_trans`；
- 单 backend、F32/F16 等 `paged_swap_out_block` 已支持的 cell 类型；
- 已 `LLAMA_KV_PAGED_SWAP=1`（复用 swap backing store）。

beam search / parallel sampling / 量化 V / 多 stream 一律不在 v1 范围（见 §10 non-goals）。

---

## 3. sequence → physical block 映射如何维护

### 3.1 seq_id → logical cells

遍历 `v_cells[0]`：cell i 属于 seq 当且仅当 `cells.seq_has(i, seq_id)`。没有现成倒排索引，但
`used` 集合 + `seq_has` 足够；规模是 `used_max_p1()` 量级，per-step 一次扫描可接受（也可缓存，见 §3.4）。

### 3.2 logical cell → physical block

复用 `paged_resolve(cell)`（[src/llama-kv-cache.cpp:1724](../src/llama-kv-cache.cpp#L1724)）：
`logical_block = cell / paged_block_size` → `paged_block_table[logical_block]` → `physical_block`。
physical block index = `phys_cell / paged_block_size`。

### 3.3 physical block 混多个 seq 的处理

按 block 聚合一个 **block→seq bitmap**：

```
block_seq[b] = OR over cells c in physical block b of cells.seq[c_logical]
```

注意方向：`paged_block_states` 以 **physical** block 为索引，而 `seq` 以 **logical** cell 为索引。
需要先经 `paged_block_table` 把 logical block 映射到 physical block，再把该 logical block 内所有 cell 的
seq bitset OR 进对应 physical block 的 `block_seq`。

判定：

- `block_seq[b]` 与 `active_seq_set`（本 step ubatch 的 seq 集合，或更严格地「非 idle seq 集合」）**有交集**
  → block 是 **hot**，禁止换出；
- `block_seq[b]` ⊆ `idle_seq_set` 且非空 → block 是 **cold**，可换出；
- `block_seq[b]` 为空 → 该 block 无 live cell，属 Stage 4A release 的范畴，不归本路径。

### 3.4 是否需要 ownership / refcount / seq bitmap

需要 **block→seq bitmap**（`std::vector<seq_set_t> paged_block_seq`，长度 `paged_n_blocks`）。
不需要 refcount：bitset 的「是否含 active seq」判定已足够，refcount 反而引入额外失配风险。
bitmap 在 `paged_note_cells` 之后、换出决策之前重建一次即可（per-step O(used) 一次扫描）。
v1 优先「每步重建」求简单可验证，缓存增量更新留作后续优化。

---

## 4. idle swap-out 的安全条件

这是 exact（保 sha256）的全部依据，按强弱排列：

1. **active seq 的 block 绝不换出。** 只要 `block_seq[b] ∩ active_seq_set ≠ ∅` 就保留 resident。
   active_seq_set 至少包含本 step ubatch 的所有 seq。
2. **mixed block 只要含一个 active seq 就必须 resident。** 这是 unified cache 的直接后果：block 混 seq，
   不能因为「大部分是 idle seq」就换出——会连带换走 active seq 的 KV，下一步立即被 read 钩子换回 → 退化为
   Stage 4B-RSS 的 thrashing。**「含 active 即保留」是第一版不 thrash 的硬保证。**
3. **swapped block 必须已有完整 backing store。** 复用 Stage 4B-RSS 的 per-physical-cell swap metadata
   （`paged_swap_offsets` / `paged_swap_sizes`）。换出走 `paged_swap_out_block`，它本身要求 RESIDENT 且
   逐 cell 落盘成功才置 SWAPPED；不引入新落盘路径。
4. **release 与 swap 语义区分。** RELEASED = 块无 live cell、内容可丢弃、读到即 violation（Stage 4A）；
   SWAPPED = 块有 live cell、内容已落盘、读到必须先换回（Stage 4B）。idle swap 产出的永远是 **SWAPPED**，
   绝不能误标 RELEASED（否则 resume 读到会触发 `paged_release_violation` 且数据错）。

> 安全口径建议比「ubatch seq」更宽：`active_seq_set` = 本 step ubatch seq ∪ 所有 `idle_steps < T` 的 seq。
> 即只有越过 idle 阈值的 seq 才算 cold。这样短暂停顿（停 1~2 步）的 seq 不会被换出又立刻换回。

### 4.1 physical read-window safety（关键补充）

§4 的「含 active 即保留」是 **逻辑 request 视角** 的安全条件，但它**不充分**。在 llama.cpp 当前的
unified KV cache + attention graph 下，存在一个更底层的约束：

**「block 不含 active seq」≠「block 不会被物理读取」。**

原因：attention graph 仍可能按全局 `[0, n_kv)` 物理读取 K/V（`set_input_paged_row_idx` 的 row_idx
张量宽度是 `GGML_PAD(used_max_p1, n_pad)`，覆盖整个 padded read window，见
[src/llama-kv-cache.cpp:3557](../src/llama-kv-cache.cpp#L3557)），seq 之间的不可见性是靠
**attention mask 隐藏 token**，而**不是缩小物理读窗**。我们在前序阶段已经验证过：**mask-only 不等于物理
read window 缩小**。因此一个只属于 idle seq 的 block，**逻辑上无人读，物理上仍可能落在 row_idx 覆盖范围里**。

如果在这种情况下换出该 block，`set_input_paged_row_idx` 下一步仍会对它调
`paged_check_read_resident()`（[src/llama-kv-cache.cpp:3563](../src/llama-kv-cache.cpp#L3563)），命中
SWAPPED → 立即 `paged_swap_in_block` 换回 → **退化为 Stage 4B-RSS 的 thrashing**。这与 §8.3 想避免的根因
完全一致——区别只在于这次的「读」来自 graph 的全局物理读窗，而非逻辑 attention 可见性。

因此 idle block 可 swap-out 的安全条件必须**在 §4 基础上再加一条物理层约束**，完整列表：

1. block 不含 active seq（`block_seq[b] ∩ active_seq_set == ∅`）；
2. block 非空且只含 idle seq（`block_seq[b].any()` 且 ⊆ idle_seq_set）；
3. **block 当前不在 active graph 的 physical read window / row_idx 覆盖范围内**；
4. block 有完整 backing store（复用 4B-RSS 的 per-cell swap metadata）；
5. block 当前 `state == RESIDENT`。

其中第 3 条是本次新增的**硬约束**。判定方法：在 `set_input_paged_row_idx` 构建 row_idx 时收集本 step 实际
写入张量的 physical block 集合（`read_window_blocks`），cold candidate 只有**不在**该集合里才允许换出。

**除非后续实现 masked-column dummy remap（把 idle 列重映射到一个 dummy/reserve row）或 seq-aware row_idx
缩窗（让 row_idx 只覆盖 active seq 的物理块），否则不能仅凭 seq mask 认为某 block 不会被物理读取。** 这两项
均超出 v1 范围（见 §10），是「真正的 request-level physical read isolation」的前置研究。

> 推论：在当前 graph 不缩窗的前提下，单 step 内**几乎所有 used block 都在 read window 里**（与 4C-1 实测
> `active_read_blocks` 覆盖 `blocks_in_use` 一致）。这意味着 idle block 的安全换出窗口，依赖于 **idle seq 的
> 块何时离开 row_idx 覆盖范围**——很可能要等到 `n_kv` 收缩（idle seq 的 cell 被 `seq_rm` 或不再计入
> `used_max_p1()`）才出现。v1-A 的 telemetry（§7、A3.5）就是用来**实测这个安全窗口到底存不存在**，再决定能否进 B1。

---

## 5. resume / swap-in 触发点

**完全复用现有钩子，不新增同步换入路径。**

- **读路径**：`set_input_paged_row_idx` → `paged_check_read_resident(phys)`
  （[src/llama-kv-cache.cpp:3563](../src/llama-kv-cache.cpp#L3563),
  [1800](../src/llama-kv-cache.cpp#L1800)）。遇到 SWAPPED block 调 `paged_swap_in_block` 换回。
  idle seq 一旦重新进入 ubatch，其 block 出现在 row_idx 覆盖范围里，读前自动换回。
- **写路径**：`paged_write_resolve` → `paged_ensure_write_resident(phys_cell)`
  （[src/llama-kv-cache.cpp:1771](../src/llama-kv-cache.cpp#L1771)）。resume 的新 token 写入时换回所在 block。

Stage 4B-RSS 结果已验证两条钩子对 SWAPPED block 各命中 66 / 50 次、零失败、计数闭合，**resume 机制现成可用**。

**v1 不做主动 prefetch。** resume 首步在读/写钩子里同步换回整请求 block（4C-2B 实测单次最高 336 MiB I/O 尖峰），
会拉长 resume 首 token 延迟。把异步 prefetch 列为 §10 non-goal，留给 4C-3 之后的 C 阶段可选项。

---

## 6. 第一版最小设计

**从 sequence-level idle swap 起步，不做完整 request scheduler。** server slot 已经把请求映射到 seq_id，
KV 层只需「按 seq idle 状态选择安全冷块换出」，调度策略留在上层。

最小落地形态（telemetry → swap 两步走）：

1. **v1-A（telemetry-only，先行）**：维护 `seq_last_active_step` 和 `paged_block_seq` bitmap，
   每步记录「哪些 seq idle、哪些 block 是 cold candidate（`block_seq ⊆ idle` 且非空）」，
   **只统计不换出**。验证 ownership bitmap 正确、cold 判定与 4C-2B 仿真趋势一致。**输出与 baseline 必须逐位一致**
   （纯计数，零行为改变）。
2. **v1-B（idle-seq swap-out）**：新增选择器 `paged_swap_out_idle_seqs(n_kv)`，**替换** apply 流程里
   `paged_swap_out_window` 的调用（仅当 idle swap 开关打开时）。它只换出满足 §4 全部条件的 cold block。

新增 env：

- `LLAMA_KV_PAGED_IDLE_SWAP=1` — 总开关（默认关；关时完全 no-op，走原 `paged_swap_out_window` 或不 swap）；
- `LLAMA_KV_PAGED_IDLE_THRESHOLD=<steps>` — idle 阈值 T，默认 8（对齐 4C-2B 推荐的 8~16 偏小值）；
- 复用既有 `LLAMA_KV_PAGED_TRACE=1` 输出 ownership / cold-candidate 计数（v1-A 即可观测）。

是否先 telemetry-only：**是，强烈建议**。v1-A 把「ownership bitmap 正确性」与「换出动作」解耦，前者可在
零行为漂移下单独守门，是 unified-cache 混 seq 场景下最大风险（block ownership 不清）的最便宜验证手段。

---

## 7. 如何测试

### 7.1 命令行能否构造多 seq / multi-request？

不容易。`llama-cli` 默认单序列贪心解码（4B-RSS / 4C-1 trace 都是单 seq）。多 seq 路径主要在
`llama-server`（多 slot）和 `llama-parallel`。但 server 的 idle gap 由真实客户端时序决定，不可复现，
不适合做 sha256 守门。

### 7.2 第一版测试策略

- **correctness 守门（必须）**：沿用 4B-RSS 的单序列 ctx=512 / ctx=4096 smoke。即便单序列下 idle 永远不触发
  （seq 每步 active），也要验证「开关打开但无 idle」时输出与 baseline **逐位一致 / sha256 相同**——证明
  telemetry 与选择器零行为漂移。这是回归底线。
- **multi-seq 行为验证（synthetic / debug hook）**：用 `llama-parallel` 或新增一个 debug 驱动，
  构造两个 seq 错峰 decode（seq A 解码若干步后停，seq B 解码，期间 A 越过 idle 阈值被换出，随后 A resume）。
  断言：(a) A idle 期间其专属 block 进入 SWAPPED；(b) A、B 共享的 mixed block **始终 resident**；
  (c) A resume 后输出与「全程 resident」基线逐位一致。
- **sha256 一致性**：v1-A 必然一致（纯计数）。v1-B 在 §4 安全条件成立时**应当**一致——因为换出的 block
  在 idle 窗口内无人读，换回逐位还原。守门即比对 multi-seq 场景下 idle-swap on/off 两跑的 token 序列 sha256。

### 7.3 需要的日志与统计

经 `LLAMA_KV_PAGED_TRACE` / `paged_log_stats` 暴露：

- `paged_idle_seq_swap_enabled`；
- `paged_idle_cold_candidates`（`block_seq` 非空、不含 active seq、只含 idle seq 的块数）；
- `paged_idle_read_window_blocks`（本 step row_idx / physical read window 覆盖的 block 数）；
- `paged_idle_cold_in_read_window`（cold candidate 且仍在 read window —— **不可换出**，否则 thrash）；
- `paged_idle_cold_not_in_read_window`（cold candidate 且已离开 read window）；
- `paged_idle_safe_swap_candidates`（`cold_not_in_read_window && state==RESIDENT` —— 唯一允许 v1-B 真实换出的集合）；
- `paged_idle_skip_mixed_active`（因含 active seq 被跳过的块数 —— 逻辑安全条件的证据）；
- `paged_idle_blocks_swapped_out` / `paged_idle_blocks_swapped_in`（区别于 4B-RSS 的 window swap 计数）；
- `paged_idle_resume_swap_in`（resume 触发的换回数）；
- `paged_idle_thrash_in_4_steps`（换出后 ≤4 步又换回的块数，目标 0，对照 4C-2B 的 thrash=0）。

**进入 B1 的判断标准**：在 multi-seq telemetry smoke 中，如果 `paged_idle_safe_swap_candidates` 长期为 0
（即大量 idle cold block 仍落在 read window 里，`cold_in_read_window` ≈ `cold_candidates`），说明**当前
llama.cpp graph 尚未支持 request-level physical read isolation**——此时**不得进入 B1**，下一步应先研究
seq-aware row_idx / masked dummy row / per-seq graph read window（见 §10 延伸研究）。只有当
`safe_swap_candidates > 0` 持续出现，idle swap 才有 exact 的落地空间。

---

## 8. 与 Stage 4B-RSS 的关系

### 8.1 可复用（不改）

- `paged_swap_out_block` / `paged_swap_in_block` — 物理块换出/换回机制，逐 cell 落盘/还原 + madvise；
- `paged_check_read_resident` / `paged_ensure_write_resident` — resume 的读/写换回钩子；
- per-physical-cell swap metadata（`paged_swap_offsets` / `paged_swap_sizes`）；
- `paged_resolve` / `paged_block_table` — logical→physical 映射；
- swap backing store（`kv_swap_store`）与全部 swap 失败计数器。

### 8.2 不复用 / 替换

- `paged_swap_out_window`（[src/llama-kv-cache.cpp:2601](../src/llama-kv-cache.cpp#L2601)）—— 它的
  **选择策略**（强制 sink=1 + window=1，换出中间所有块）是 thrashing 根源，被
  `paged_swap_out_idle_seqs` **替换**（而非修改）。开关分流：idle swap 开 → 走新选择器；否则保持原行为。

### 8.3 为什么这次不 thrash

Stage 4B-RSS 在单请求里强制换出 live history，撞上「每步读全历史」（4C-1 已证 `active_read_blocks` 覆盖
`blocks_in_use`），换出立即被换回 → `swap_out=swap_in=8000`。本设计**只换出不含任何 active seq 的 block**：
idle seq 在 idle 窗口内一个 block 都不读，换出期间无人访问，换回只发生在 resume 那一步（4C-2B 实测
thrash=0）。区别只有一句：**换出的东西当下到底有没有人读。**§4.2 的「含 active 即保留」把这条保证落到了
mixed block 上。

---

## 9. 风险表

| 风险 | 后果 | 缓解 |
|---|---|---|
| **mixed block 误换出 active seq** | active seq KV 被换走，下一步换回 → thrashing 且可能延迟错 | §4.2 硬约束「block_seq ∩ active ≠ ∅ 即保留」；`paged_idle_skip_mixed_active` 计数守门；v1-A telemetry 先验证 ownership |
| **cold block 仍在物理 read window**（mask-only ≠ 缩窗） | 只属 idle seq 但仍被 graph 全局读 → 换出后 `paged_check_read_resident` 立即换回 → 退化为 4B-RSS thrashing | §4.1 第 3 条硬约束「不在 read window 才可换」；`paged_idle_cold_in_read_window` 守门；`safe_swap_candidates==0` 则不进 B1 |
| **block ownership 不清**（logical/physical、shift 映射错位） | bitmap 错 → cold 判定错 → 误换或漏换 | bitmap 必经 `paged_block_table` 解析；v1-A 在零行为下单独验证 bitmap；shift 场景（`paged_non_identity_enabled`）单列 smoke |
| **resume 同步 swap-in 延迟** | 整请求换回挤在 resume 首步，最高 ~336 MiB I/O，拉长首 token | 接受为 v1 已知代价（4C-2B 已量化）；prefetch 列 non-goal 留后续 |
| **backing store 元数据失效** | swap-in 找不到 offset / size 错 → 数据错或失败 | 复用 4B-RSS 已验证的 per-cell metadata（实测 fail_no_offset=0）；不新增落盘路径 |
| **多 seq / beam / parallel decode 兼容** | 非 unified / v_trans / beam 下语义未验证 | v1 强制 `n_stream==1 && !v_trans`，其余早退 no-op；beam/parallel 明列 non-goal |
| **release 与 swap 状态冲突** | 同块既想 RELEASE 又想 SWAP，状态机歧义 | RELEASED↔SWAPPED 互斥；idle 路径只产 SWAPPED；保持 4B-RSS「swap 与 release 第一版互斥」约定，不在同 step 对同块两者并发 |
| **seq idle 计数与 server slot 复用错配** | slot 复用 seq_id 后旧 idle 计数串到新请求 | `seq_rm` 时重置该 seq 的 `seq_last_active_step`；v1 先在 synthetic 场景验证，server 集成留后续 |

---

## 10. 明确 non-goals（v1 不做）

- **不做 sink+recent approximate**（4C-2A）—— 那条改注意力可见 KV，不保证 sha256；本路线是 exact。
- **不做 attention pruning / sparse / sliding-window**。
- **不做 GPU offload**。
- **不做 prefetch v1**（resume 同步换回，异步预取留后续 C 阶段）。
- **不做 v_trans / 量化 V / 多 stream**（强制 `n_stream==1 && !v_trans`，否则 no-op）。
- **不改 attention kernel**、不改公开 API（不加显式 idle 标记接口）。
- **不做完整 request scheduler**（调度留上层 server，KV 层只按 seq idle 选块）。
- **不追求降绝对峰值**（4C-2B 已证峰值在全 active 时刻，idle-swap 只降平均 / 持续 footprint）。

### 延伸研究（v1 不做，但 §4.1 / §7 指向它）

当 telemetry 实测 `paged_idle_safe_swap_candidates` 长期为 0（idle cold block 仍被全局物理读窗覆盖），说明
当前 graph 缺少 request-level physical read isolation。下一步研究方向（任选其一，均为独立立项）：

- **seq-aware row_idx 缩窗**：让 `set_input_paged_row_idx` 只为 active seq 的物理块生成 row，把 idle seq 的
  块排除出物理读窗——这才真正让 idle block「无人物理读」，是 exact idle swap 的前置条件。
- **masked-column dummy remap**：把被 mask 隐藏的 idle 列 row_idx 重映射到一个 dummy/reserve row，使其物理读
  不触达真实 idle block。

二者都改 row_idx 构建（不改 attention kernel），但需独立 correctness 验证，超出 4C-3 v1。
