# Stage 4C-3 — 多请求 / idle-request telemetry 验证方案

前置：

- [docs/kv_paged_routeb_idle_request_swap_design.md](kv_paged_routeb_idle_request_swap_design.md) — idle swap 设计（§4.1 物理 read-window safety、§7.3 进入 B1 的 gate）
- [docs/kv_paged_routeb_idle_request_swap_implementation_plan.md](kv_paged_routeb_idle_request_swap_implementation_plan.md) — A1/A2/A3/A3.5 telemetry 任务（已落地）
- [docs/kv_paged_routeb_multi_request_sim.md](kv_paged_routeb_multi_request_sim.md) — 多请求 idle swap 离线仿真
- [docs/kv_paged_read_stage4b_rss_results.md](kv_paged_read_stage4b_rss_results.md) — 复用的 swap 机制

本文是 Stage 4C-3 的**验证方案设计**。本轮**只写文档，不改代码、不 build、不跑实验、不 commit**。

本方案是一个 **gate 验证实验**，不是内存优化结果。它的唯一产出是回答一个二元问题：在当前 llama.cpp graph 下，idle seq 的 KV block 是否存在 exact 安全换出窗口（`safe_swap_candidate > 0`）。无论结论是 >0 还是 =0，都直接决定 B1 能否开工——见 §8。

---

## 1. 验证目标

- 用一个确定性 driver 构造 **A-idle / B-active** 场景：seq 0（=A）写入 KV 后停止进入 ubatch，seq 1（=B）持续 decode；
- 借已落地的 A1/A2/A3/A3.5 telemetry，实测 A 的 block 是否被记为
  `cold_candidate` / `cold_in_read_window` / `cold_not_in_read_window` / `safe_swap_candidate`；
- **本阶段只做 telemetry，不做真实 swap**：swap 开关全程关闭，A3.5 是纯计数，block 状态恒为 RESIDENT。

明确边界：本实验**不度量内存收益**，不换出任何块，不产生 RSS drop。它只验证「安全换出窗口是否存在」。

---

## 2. 为什么不用 llama-completion

`llama-completion` / `llama-cli` 默认单序列贪心解码（4B-RSS / 4C-1 trace 都是单 seq）。单请求下 A3.5 已实测 `safe_swap_candidate=0`（full attention 每步读全历史，4C-1 结论）。它无法让「A 写入后 idle、B 同时 active」并存，因此无法验证多请求场景的安全窗口。

---

## 3. 为什么暂时不用 llama-server

`llama-server` 多 slot 能产生真实 idle，更贴近生产，但**不适合作为 gate 实验**：

- idle gap 由客户端时序决定，**不确定、不可复现**；
- decode step 步序随并发抖动，telemetry 无法稳定对齐；
- 无法做 sha256 守门（输出顺序依赖到达时序）。

server 适合 B 路线落地后的 D 阶段真实 demo（见实施计划 D1），不适合现在判定安全窗口是否存在。

---

## 4. 为什么选择仿 examples/batched 写最小 C++ driver

[examples/batched/batched.cpp](../examples/batched/batched.cpp) 是最省力且最可控的起点：

- 用 `common_batch_add(batch, token, pos, {seq_id}, logits)` 可**显式把 token 定向到指定 seq**；
- 由调度循环决定每步放哪些 seq → 可**确定性构造** seq 0 写入后 idle、seq 1 持续 active；
- 单 ctx、`n_stream==1`、贪心解码 → 可与「单 seq 单跑」做 **sha256 对照**。

对比其他 example：

| example | 多 seq | 确定性 | idle 模式 | 结论 |
|---|---|---|---|---|
| `examples/idle` | 否（单 seq，每轮 `llama_memory_clear`） | 是 | 无 | 不可用 |
| `examples/batched` | 是（`common_batch_add` 定向 seq） | 是 | 无（需改调度） | **改造起点** |
| `examples/parallel` | 是（多 client + `seq_cp`/`seq_rm`） | 否（连续批处理 + 随机调度） | 有但不可控 | 不适合 gate |

`examples/batched` 唯一缺的是 idle 模式——这正是 driver 要改的调度循环。

---

## 5. 最小实验设计

确定性 timeline（单 ctx，unified cache，`n_stream==1`，`!v_trans`，贪心）：

1. **A 写入阶段**：seq 0 写入一段 prompt 并解码若干 token → A 占据若干 KV block；
2. **A idle / B active 阶段**：进入循环，**只对 seq 1** `common_batch_add` + `llama_decode`，连续 N 步；seq 0 全程不入 ubatch；
3. A 越过 `LLAMA_KV_PAGED_IDLE_THRESHOLD` 后成为 idle seq；
4. 每步读 trace 行，观察 A 的 block 落入 cold / read-window 哪一类。

约束（保证零行为漂移、零换出）：

- **不对 seq 0 调用 `seq_rm`**——让 A 的 KV 留在 `v_cells[0]` 里自然变冷（这正是要观测 used_max_p1 是否仍覆盖它）；
- 不对 seq 0 调用 `seq_cp`；
- 不调用任何 swap / madvise / prefetch；
- **不设** `LLAMA_KV_PAGED_SWAP` / `LLAMA_KV_PAGED_SWAP_MADVISE` / `LLAMA_KV_PAGED_IDLE_SWAP`。

swap 关闭的天然保证：`paged_swap_out_block` 在 `paged_swap_enabled==0` 时早退；madvise 仅在 swap 路径内调用；A3.5 统计纯读不写状态，block 恒为 RESIDENT。

### 5.1 sha256 对照口径（trace-off vs trace-on，同一 driver）

correctness 守门**必须**用同一个 `idle-telemetry` driver、**同一调度、同一 seed**的两次运行对照，**不与任何 single-seq 运行比较**：

- **base（trace off）**：跑 driver，不开 trace 子开关（telemetry 路径不构建）；输出写入 `base.out`；
- **trace on**：同一 driver、同一调度、同一 seed，打开
  `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1 LLAMA_KV_PAGED_IDLE_TRACE=1`（必要时叠加 `LLAMA_KV_PAGED_TRACE=1`）；输出写入 `trace.out`。

比较对象只有 `base.out` 与 `trace.out`，证明「打开 idle telemetry 不改变同一 multi-seq workload 的输出」。

> **不将 multi-seq 输出与 single-seq 输出直接比较。** multi-seq 与 single-seq 的 token 序列本就不同（B 的存在改变了 ubatch 组成与 cell 布局），二者 sha256 不可能一致，拿来对照没有意义。single-seq 单跑**只能作为额外 sanity check**（确认 driver 在退化为单 seq 时仍正常解码），**不作为 multi-seq gate 的主 correctness 对照**。multi-seq gate 的主正确性判断,是同一 multi-seq driver 的 trace-off 与 trace-on 对照。

---

## 6. 需要新增文件

- `examples/kv-idle-telemetry/idle-telemetry.cpp` — 仿 `batched.cpp` 的最小 driver，实现 §5 的调度；
- `examples/kv-idle-telemetry/CMakeLists.txt` + examples 构建接入；
- `scripts/kv_paged_idle_telemetry_smoke.sh` — 跑 driver、grep trace、汇总六字段、比对 sha256。

**不改** `src/llama-kv-cache.{h,cpp}`（telemetry 已落地）、**不改** attention kernel / row_idx 写入 / 任何 swap 路径。

---

## 7. 通过标准

- **correctness 主守门**：同一 driver、同一调度、同一 seed 的 trace-off（`base.out`）与 trace-on（`trace.out`）两次运行，
  `cmp -s base.out trace.out` 返回 0，且两文件 sha256 完全一致——证明打开 idle telemetry 对同一 multi-seq workload 零行为漂移；
- （sanity check，非主守门）single-seq 单跑能正常解码完成，**不与 multi-seq 输出做 sha256 比较**；
- `paged_swap_enabled=0`；
- `paged_swap_out_calls=0`；
- `paged_swap_in_calls=0`；
- `paged_swap_madvise_calls=0`；
- `paged_swap_backend_failures=0`；
- multi-seq trace 能**稳定输出** `cold_candidate`、`cold_in_read_window`、`cold_not_in_read_window`、`safe_swap_candidate` 四组非平凡数；
- **明确报告 `safe_swap_candidate` 是 >0 还是 =0**。

---

## 8. 两种结果解释

### 8.1 `safe_swap_candidate > 0`

A 的 idle block 既不含 active seq、又**已离开物理 read window**（row_idx 不再覆盖它）且仍 RESIDENT → 存在 exact 安全换出窗口 → **可以进入 B1 真实 idle swap 设计**。说明当前 graph 在「某 seq idle 后其 block 自然退出 `used_max_p1()` / row_idx 覆盖」上已具备 request-level 物理读隔离的雏形。

### 8.2 `safe_swap_candidate = 0`（当前预计结果，合理且有价值）

这是**基于源码的预期结果**，不是失败。row_idx 宽度 = `GGML_PAD(used_max_p1, n_pad)`（[src/llama-kv-cache.cpp:3557](../src/llama-kv-cache.cpp#L3557)），`used_max_p1` 是全局最大已用 cell。只要 A 的 KV 还驻留在 `v_cells[0]`（没被 `seq_rm`），它就被算进 `used_max_p1`，A 的 block **始终落在 read window 里** → cold candidate 全部 `cold_in_read_window`，`safe_swap_candidate=0`。这正是设计 §4.1 的核心论断：**mask-only 不缩物理读窗**。

结论：**不得进入 B1**，转设计 §10 延伸研究——seq-aware row_idx 缩窗（让 row_idx 只覆盖 active seq 的物理块）或 masked-column dummy remap（把 idle 列重映射到 dummy row），让 idle seq 的块真正退出 row_idx 覆盖。

**价值**：这个实验要么发现安全窗口存在（>0，放行 B1），要么用真实多 seq 数据**坐实** §4.1 的物理读窗约束（=0，挡住 B1 并指明唯一可行的研究方向）。两种结论都直接决定下一步——这就是 gate 的意义。

---

## 9. 给 Codex 的最小执行任务

> **任务：KV idle-request telemetry 验证 driver（只读 telemetry，不换出）**
>
> - 复制 `examples/batched/batched.cpp` 为 `examples/kv-idle-telemetry/idle-telemetry.cpp` + `CMakeLists.txt`，接入 examples 构建；
> - 改调度：seq 0 解码 prompt 后，循环**只对 seq 1** `common_batch_add` + `llama_decode` N 步；seq 0 全程不入 ubatch、不 `seq_rm`、不 `seq_cp`；
> - 新增 `scripts/kv_paged_idle_telemetry_smoke.sh`：
>   - **base（trace off）**：跑 driver 写 `base.out`；
>   - **trace on**：同一 driver、同一调度、同一 seed，以
>     `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1 LLAMA_KV_PAGED_IDLE_TRACE=1`（必要时叠加 `LLAMA_KV_PAGED_TRACE=1`、`LLAMA_KV_PAGED_IDLE_THRESHOLD=8`）跑 driver 写 `trace.out`；
>   - 比对 `cmp -s base.out trace.out`（返回 0）+ 两文件 sha256 一致；
>   - grep trace 出六字段逐步打印；
>   - **不与 single-seq 输出做 sha256 比较**（multi-seq 与 single-seq 输出本就不同）；
> - **禁止**：改 `src/llama-kv-cache.*`、改 attention、改 row_idx；设置 `LLAMA_KV_PAGED_SWAP` / `LLAMA_KV_PAGED_IDLE_SWAP`；调用任何 swap / madvise / prefetch / `seq_rm`-on-A；commit / push；
> - **通过标准**：`base.out` 与 `trace.out` `cmp`/sha256 一致；`paged_swap_*_calls=0`、`paged_swap_madvise_calls=0`；trace 稳定输出四组 cold / window 字段；明确报告 `safe_swap_candidate` 是否 >0；
> - **回滚标准**：`base.out` 与 `trace.out` sha256 漂移，或任何 swap / madvise 计数非 0；
> - **Claude Code 审查：是**（driver 的 seq 调度正确性 + gate 结论解读）。
