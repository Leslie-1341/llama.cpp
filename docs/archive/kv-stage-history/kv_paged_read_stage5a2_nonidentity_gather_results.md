# Stage 5A-2 — non-identity gather / row_idx remap probe results

前置：

- [docs/kv_paged_routeb_idle_telemetry_validation_results.md](kv_paged_routeb_idle_telemetry_validation_results.md) — Stage 4C-3：idle cold block 仍在物理 read window 内的负向 gate
- [docs/kv_paged_read_stage3b_ingraph_gather_results.md](kv_paged_read_stage3b_ingraph_gather_results.md) — in-graph `ggml_get_rows` gather path
- driver：`examples/kv-idle-telemetry/idle-telemetry.cpp`
- 改动文件：`src/llama-kv-cache.cpp`、`src/llama-kv-cache.h`

---

## 1. 背景

Stage 4C-3 在 A-idle / B-active 多请求场景下得到一个明确的负向 gate：

```text
cold_candidates=1
cold_in_read_window=1
cold_not_in_read_window=0
safe_swap_candidates=0
```

引擎能识别出 idle 请求留下的冷 KV block（`cold_candidates=1`），但它仍落在当前物理 read window 内（`cold_in_read_window=1`），因此没有任何安全换出候选（`safe_swap_candidates=0`）。

根因：read window 由 `set_input_paged_row_idx()` 对每个逻辑行 `r∈[0,n_kv)` 做恒等 `paged_resolve(r)` 构建，逐行覆盖全部已写 cell，不区分 seq。只要 idle cell 的逻辑行进入 row_idx，它的物理 block 就一定进入 read window。

下一步问题因此被改写为：**怎么让 idle seq 的 block 退出物理 read window**，而不是“怎么换出 idle KV”。

## 2. Stage 5A-1 基础

Stage 5A-1 先验证 in-graph gather path 可用、可控：

- 新增环境变量 `LLAMA_KV_PAGED_INGRAPH`，可显式在“连续 view 读取”与“`ggml_get_rows` gather 读取”之间切换。
- identity gather 验证通过：row_idx 为恒等映射时，view path 与 gather path 的输出 **sha256 完全一致**。
- gather 实际生效（`ingraph_gather_layers>0`），证明现有 `ggml_get_rows` gather path 数学等价于原连续 view，可以作为 row_idx remap 的载体。

5A-1 是 5A-2 的前提：只有 gather path 本身被证明无损，才能在其上改 row_idx 内容。

## 3. Stage 5A-2 设计

- 新增环境变量 `LLAMA_KV_PAGED_GATHER_NONIDENTITY=1`（默认关闭）。
- 在 `set_input_paged_row_idx()` 内，对 idle-only cold cell 做 **row_idx remap**（重映射逻辑行指向的物理 cell）。
- **不压缩 `n_kv`**：row_idx 宽度仍为 `dst->ne[0] == n_kv`。
- **不改变 mask shape**、**不改变 graph shape**。
- **不改** `get_k` / `get_v` / attention / `llama-graph`。
- **不做** 真实 swap / release / madvise / prefetch。

这是方案 B（保持宽度、重定向被 mask 的行），而非方案 A（压缩 `n_kv` / 动态 mask 形状）。方案 B 保持图形状恒定，规避动态 shape 风险，并让输出天然不变。

## 4. 核心机制

对每个逻辑行 `r`：

- **active-visible cell**：保持原 `paged_resolve(r)` 物理映射，不动。
- **idle-only、当前 active seq 不可见、理论上被 mask 的行**：把物理目标重定向到一个 **dummy physical cell**。

dummy cell 的选取约束（全部满足才用）：

- 来自 active-visible cell（对某个当前 active seq `seq_has`）；
- 非空（`!is_empty`）；
- 所在 block 处于 `RESIDENT` 状态。

remap 前对该行再做一次**行级 mask 复核**：只有当该行对所有当前 active seq 都不可见（`r>=active_n_kv`、空 cell、或所有 active seq `!seq_has`）时才允许 remap。任一条件不满足则跳过（计入 `skip_not_masked` / `skip_not_resident` / `skip_no_dummy`）。

效果：被 remap 的 idle row 指向的 dummy cell 本就会被 attention mask 屏蔽成 -INF（idle cell 对 active seq 不可见），所以 attention 读到的有效内容不变；而 idle block 的**真实物理 block 不再进入 `trace_read_blocks`（read window）**。

## 5. 实验结果

> 由于仓库内无 `.gguf` 模型，验证需用户提供 `-m $MODEL`（F32 KV，`-fa -nkvo`，`--kv-unified`）。以下为已记录的 remap smoke 结果。

Stage 5A-2 remap smoke：

```text
base_vs_remap_equal=0
sha256 完全一致
warnings 为空
swapped_blocks=0
```

base（`LLAMA_KV_PAGED_GATHER_NONIDENTITY` 未设）：

```text
cold_candidates=1
read_window_blocks=16
cold_in_read_window=1
cold_not_in_read_window=0
safe_swap_candidates=0
paged_nonidentity_enabled=0
```

remap（`LLAMA_KV_PAGED_GATHER_NONIDENTITY=1`）：

```text
cold_candidates=1
read_window_blocks=15
cold_in_read_window=0
cold_not_in_read_window=1
safe_swap_candidates=1
paged_nonidentity_enabled=1
paged_nonidentity_remap_rows=528
paged_nonidentity_remap_blocks=33
paged_nonidentity_skip_no_dummy=0
paged_nonidentity_skip_not_masked=0
paged_nonidentity_skip_not_resident=0
paged_nonidentity_cold_in_read_window_before=1
paged_nonidentity_cold_in_read_window_after=0
paged_nonidentity_safe_candidates_after=1
```

逐项解释：

1. `base_vs_remap_equal=0` + sha256 一致：remap **不改变模型输出**（被 remap 的行本就被 mask 屏蔽）。
2. `read_window_blocks` 16→15：read window 少覆盖 1 个 block，正是退出的 idle cold block。
3. `cold_in_read_window` 1→0、`cold_not_in_read_window` 0→1：那个 idle cold block 被移出物理 read window。
4. `safe_swap_candidates` 0→1：在不改变输出、不触发任何 swap 的前提下，**出现了一个理论上可安全换出的候选**。
5. 三个 `skip_*=0`：没有发生静默跳过/回退，所有打算 remap 的行都满足 mask+resident+dummy 三重门控。
6. `swapped_blocks=0`：本实验没有触发任何真实 swap（telemetry-only）。

## 6. 回归结果

默认关闭回归（`LLAMA_KV_PAGED_GATHER_NONIDENTITY` 未设）：

```text
exit=0
base_vs_trace_equal=0
sha256 一致
cold_candidates=1
cold_in_read_window=1
cold_not_in_read_window=0
safe_swap_candidates=0
swapped_blocks=0
```

→ 与 Stage 4C-3 完全一致：默认关闭时**零行为变化**，不影响已有 idle telemetry。

Stage 5A-1 identity gather 回归：

```text
view_vs_gather_equal=0
sha256 一致
swapped_blocks=0
```

→ identity gather path 仍无损，5A-2 的改动未污染 5A-1。

## 7. CC review 结论

- 实现正确、保守，可作为 probe 提交；
- 默认关闭零行为变化（remap 分支被 env 门控短路，`data[r]` 逐位等价原逻辑，`active_seq` 计算仅前移、值不变）；
- remap 判定保守：仅 `owner.count()==1` 的 idle-only block，且对每行做行级 `seq_has` mask 复核，不会误 remap active-visible cell；
- dummy cell 选择安全：一定来自 active-visible、非空、RESIDENT cell，不会选到空 / swapped / released / 越界 cell，且在 swap/release 未启用时不触发任何 swap-in 或 release；
- telemetry 可信：`*_before` 用 remap 前物理块、`*_after`（含 `trace_read_blocks`）用 remap 后物理块，前后窗口计算正确；
- 无必须修复项。

## 8. 限制和边界

必须明确，避免夸大：

- 这**不是**完整 PagedAttention；
- 这**不是**压缩 `n_kv`（row_idx 宽度、mask shape、graph shape 均未变）；
- 这**不是**实际 KV swap；
- 这**没有**实际释放内存，没有任何 RSS 实测收益；
- `safe_swap_candidates=1` 只说明**理论上**出现了可安全换出的候选，并未换出任何东西；
- dummy cell 的正确性**依赖 mask 屏蔽被 remap 的 idle row** 这条不变量。代码已用行级 `masked` 复核强制此前提（不满足则不 remap），但未来若改动 mask 逻辑，这是回归风险点；
- 计数器语义：
  - `paged_nonidentity_remap_rows` / `paged_nonidentity_remap_blocks` / 三个 `skip_*` 是**跨所有 decode step 的累计值**（例如 `remap_blocks=33` 是累计，不代表存在 33 个不同 block）；
  - `paged_nonidentity_cold_in_read_window_before/after`、`paged_nonidentity_safe_candidates_after`、`paged_nonidentity_enabled` 是**最后一步的快照值**。

## 9. 阶段结论

Stage 5A-2 证明：在不改变输出、不触发 swap/madvise/release 的前提下，non-identity row_idx remap 可以让 idle-only cold block 退出物理 read window，使 `safe_swap_candidates` 从 0 变为 1。

## 10. 下一步

- Stage 5B 可考虑基于 `safe_swap_candidate` 做真实 idle block swap；
- 或先做更严格的多 prompt / 多 seed / 长上下文验证，确认 remap 在更大 idle 分布下仍保持输出不变与窗口收缩；
- 真实 swap 前必须确认 resume / swap-in 正确性——`safe_swap_candidate>0` 只是理论窗口，真实换出需保证换回后逐字节一致。
