# Stage 4C-3 — idle-request KV telemetry validation results

前置：

- [docs/kv_paged_trace_sim_stage4c1_results.md](kv_paged_trace_sim_stage4c1_results.md) — 单请求 exact 仿真：无安全冷块
- [docs/kv_paged_routeb_sink_recent_sweep.md](kv_paged_routeb_sink_recent_sweep.md) — sink+recent approximate 扫描
- [docs/kv_paged_routeb_multi_request_sim.md](kv_paged_routeb_multi_request_sim.md) — 多请求 idle swap 离线仿真（理论收益）
- driver：`examples/kv-idle-telemetry/idle-telemetry.cpp`、`examples/kv-idle-telemetry/CMakeLists.txt`
- smoke 脚本：[scripts/kv_paged_idle_telemetry_smoke.sh](../scripts/kv_paged_idle_telemetry_smoke.sh)

---

## 1. Purpose

Stage 4C-2B 的离线仿真表明：多请求场景下，idle 请求的 KV 在理论上是冷数据，可在 exact correctness 下换出。本阶段（4C-3）不再停留在仿真，而是用真实 driver + telemetry 在引擎内**实测验证一个 gate 问题**：

> 在 A-idle / B-active 的多请求场景下，idle 请求留下的 KV block 是否真的能成为 `safe_swap_candidate`？

这是一个 **gate 验证**，不是内存优化结果。本阶段**不做真实 swap、不做 madvise、不做 prefetch**，只用 telemetry 判断当前机制下是否存在安全换出窗口。

## 2. Validation setup

实验构造最小多请求场景：

- seq 0：先写入 KV，然后不再进入 ubatch，变成 **idle**；
- seq 1：当前 **active**，持续 decode；
- **不**对 seq 0 调用 `seq_rm`（不删除 idle 的 KV）；
- **不**对 seq 0 调用 `seq_cp`；
- **不**设置 `LLAMA_KV_PAGED_SWAP`（以及 `_MADVISE` / `_IDLE_SWAP`），保证 telemetry-only。

base / trace 两次运行共用同一个 driver、同一调度、同一 seed，唯一差别是 telemetry 开关：

| run | 环境 | 说明 |
|---|---|---|
| base  | `LLAMA_KV_PAGED=1`、`LLAMA_KV_PAGED_SHIFT=1` | paged on，trace off |
| trace | base + `LLAMA_KV_PAGED_IDLE_TRACE=1`（必要时 `LLAMA_KV_PAGED_TRACE=1`） | paged on，idle telemetry on |

correctness 通过 `base.out` vs `trace.out` 的 `cmp` 与 `sha256sum` 判断（见 smoke 脚本 76–84 行）。

## 3. Driver and smoke script

- driver：`examples/kv-idle-telemetry/idle-telemetry.cpp`（`--parallel 2`、`--temp 0`、`--seed 1`、f32 KV、`--kv-unified`），制造 seq 0 idle / seq 1 active。
- smoke：`scripts/kv_paged_idle_telemetry_smoke.sh` 跑 base 与 trace 两次，比对输出，再从 telemetry 抽取 idle gate 字段。
- 字段抽取顺序：脚本优先读最终 `KV paged metadata stats` 行；若该行缺失，则回退到最后一条 `KV_PAGED_IDLE_TRACE` / `KV_PAGED_TRACE`（脚本 119–128、191 行）。这一回退是本阶段 gate 判断的实际来源（见 §9）。

## 4. Correctness result

```text
exit=0
base_vs_trace_equal=0
sha256 完全一致
```

- `base_vs_trace_equal=0`：base 与 trace 的输出逐字节相同（`cmp -s` 成功）。
- sha256 一致：进一步确认两次输出完全相同。

结论：**idle telemetry 不改变模型输出**——它纯粹是观测，没有触碰 KV 内容或调度。

## 5. Telemetry result

```text
idle_trace_lines=34
paged_idle_cold_candidates=1
paged_idle_read_window_blocks=16
paged_idle_cold_in_read_window=1
paged_idle_cold_not_in_read_window=0
paged_idle_skip_mixed_active=1
paged_idle_safe_swap_candidates=0
safe_swap_candidate_gate=0
final_stats_present=0
swapped_blocks=0
```

逐项解释：

1. `idle_trace_lines=34`：trace 运行产生了 34 行 `KV_PAGED_IDLE_TRACE`，说明 idle telemetry 路径确实被触发、在工作。
2. `paged_idle_cold_candidates=1`：在 A-idle / B-active 场景下，引擎识别出 **1 个 idle cold block** ——多请求场景下确实存在 idle 留下的冷 KV，这与 Stage 4C-2B 的前提一致。
3. `paged_idle_read_window_blocks=16`：当前物理 read window 覆盖 16 个 block。
4. `paged_idle_cold_in_read_window=1`：那个 idle cold block **仍落在当前物理 read window 内**——尽管 seq 0 已 idle，它的 block 仍被 active decode 的 read window 覆盖。
5. `paged_idle_cold_not_in_read_window=0` 且 `paged_idle_safe_swap_candidates=0`：没有任何 cold block 落在 read window 之外，因此**没有安全换出候选**。
6. `swapped_blocks=0`：本实验没有触发任何真实 swap（符合 telemetry-only 设定，`LLAMA_KV_PAGED_SWAP` 未设置）。
7. `paged_idle_skip_mixed_active=1`：存在一次 mixed/active 相关的**保守跳过**——当 idle 与 active 在同一物理结构上混用时，引擎选择不把它当安全候选，符合“安全优先”策略。
8. `final_stats_present=0`：最终 `paged_log_stats` 未稳定打印，gate 字段来自最后一条 trace 行（见 §9 诚实说明）。

## 6. Gate conclusion

**多请求 A-idle / B-active 场景下，idle cold block 能被识别出来（`cold_candidates=1`），但它仍然落在物理 read window 内（`cold_in_read_window=1`、`cold_not_in_read_window=0`），因此 `safe_swap_candidates=0`。当前机制下不存在安全换出窗口，不能进入真实 idle block swap。**

这是一个**明确的负向 gate 结论**，而且是有价值的：它精确定位了阻塞点——不是“没有冷数据”，而是“冷数据还没退出物理 read window”。

## 7. Why safe_swap_candidate=0 matters

idle 请求虽然在调度语义上已经不活跃（不再进入 ubatch），但只要它的 block 仍被当前 graph 的物理 read window 覆盖，引擎在 decode 时仍可能读到这些行。此时换出/释放它们就不再是 exact —— 与 Stage 4C-1 单请求结论同源：**只要 block 仍在 read window 内，就没有安全换出窗口**。

因此 `safe_swap_candidate=0` 不是 telemetry 的失败，恰恰是它**正确地拦住了一次不安全的换出**。要让 idle KV 可安全换出，必须先让 idle seq 的 block 真正退出物理 read window。

## 8. Implication for next stage

gate 把下一步问题从“怎么换出 idle KV”改写为“**怎么让 idle seq 的 block 退出物理 read window**”。在 block 仍被 read window 覆盖时强推 swap，只会重蹈 Stage 4B-RSS 强制 live-history swap 的 thrashing（参见 [多请求仿真文档 §4.2](kv_paged_routeb_multi_request_sim.md)）。先缩窗，再谈 swap/release。

## 9. Limitations

- `final_stats_present=0`：当前自定义 driver 未稳定打印最终 `paged_log_stats`，因此 smoke 脚本退化为用**最后一条** `KV_PAGED_IDLE_TRACE` / `KV_PAGED_TRACE` 作为 gate 判断来源（脚本 119–128、191 行）。这**不影响本阶段 gate 结论**（safe_swap_candidate 的判定只依赖 idle trace 字段），但意味着 `paged_swap_*` / `row_idx_fail` 等最终统计字段在本次运行不可得。后续可改进 driver 的释放路径或最终统计打印，让 `final_stats` 稳定出现。
- 本实验是**最小合成场景**（2 seq、短 prompt、ctx=512），只验证 gate 是否成立，不代表真实多并发负载下的 idle 分布。
- 本阶段**未做**真实 swap / madvise / prefetch，因此不涉及任何内存实测收益。

明确避免夸大：

- 本阶段**没有**实现 idle request swap；
- 本阶段**没有**降低任何内存；
- 当前机制**不能**安全换出 idle KV；
- 仅完成了 gate 验证，并证明当前机制下**没有安全换出窗口**。

## 10. Next steps

不进入真实 idle swap。优先让 idle seq 的 block 退出物理 read window，再重新考虑 swap/release：

1. **seq-aware row_idx 缩窗**：让 graph 的 read window 不再覆盖 idle seq 的 block（最直接，先把 `cold_not_in_read_window` 做到 > 0）。
2. **dummy remap**：把 idle seq 的逻辑行临时映射走，使其物理上退出 read window。
3. **按需 gather / paged read**（更接近 PagedAttention）：read 集合按 seq 实际需要构建，idle seq 自然不进入。

目标统一：**让 idle seq 的 block 真正退出物理 read window，使 `safe_swap_candidate > 0`，再重新评估 swap/release**。届时可复用本 smoke 脚本作为回归 gate（safe_swap_candidate 从 0 翻正即为该阶段成功信号）。
