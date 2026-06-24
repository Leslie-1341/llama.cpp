# Stage 5C-scale-B：cumulative RSS telemetry 结果

> 本文档记录 Stage 5C-scale-B 的实验结果与 telemetry 解释，仅为结果归档，不包含源码修改。
> 文件路径：`docs/kv_paged_read_stage5c_scale_cumulative_rss_results.md`
> 关联文档：
> - `docs/kv_paged_read_stage5c_idle_swap_release_plan.md`（5C 计划）
> - `docs/kv_paged_read_stage5c_idle_swap_madvise_results.md`（5C madvise / current RSS 结果）

---

## 1. Stage 5C-scale-B 目标

在 Stage 5C 已闭环（safe idle swapped block 可 madvise、保持 `SWAPPED`、resume swap-in 逐字节一致、current RSS 单次 drop 为正）的基础上，本阶段验证：

- 新增的**累计 RSS telemetry** 是否能量化「整个 idle swap madvise 阶段累计降低了多少 current RSS」；
- 累计 RSS drop 是否随 idle swap madvise 规模（`madvise_calls` / `madvise_bytes`）放大；
- 放大 workload（ctx / n）后 resume correctness 是否仍然逐字节一致。

本阶段只做 telemetry 增强与 small / mid / large 三档矩阵观测，不改 swap / madvise 语义，不改状态机。

---

## 2. 为什么需要累计 RSS telemetry

Stage 5C 的 RSS 字段（`rss_before_last` / `rss_after_last` / `rss_drop_last` / `rss_drop_max`）只反映**单次** madvise 的 last/max drop。它们能证明「某一次 madvise 后 current RSS 下降为正」，但无法回答：

```text
整个 idle swap madvise 阶段，current RSS 总共下降了多少？
这个累计下降是否随 madvise_calls / madvise_bytes 放大？
```

因此 5C-scale-B 新增两个不同口径的累计字段（`total_drop` / `drop_sum`），配合首次采样的 `before_first`，才能在矩阵放大时观测累计 current RSS 收益的趋势。

---

## 3. 新增 telemetry 字段解释

| 字段 | 含义 |
|---|---|
| `paged_swap_rss_before_first_kb` | 第一次成功进入 madvise RSS 采样（advised range 有效、计入 `madvise_calls`）时的 RSS before。**记录一次后不再覆盖**，代表整个 madvise 窗口起点的常驻内存。 |
| `paged_swap_rss_after_last_kb` | 最后一次 madvise 后的 RSS after，随每次 madvise 更新，代表窗口终点的常驻内存。 |
| `paged_swap_rss_total_drop_kb` | 窗口净下降：`max(0, rss_before_first - rss_after_last)`。回答「从第一次 madvise 前到最后一次 madvise 后，current RSS 总体下降多少」。 |
| `paged_swap_rss_drop_sum_kb` | 逐次正向 drop 累加：`sum(max(0, rss_before_this_call - rss_after_this_call))`。回答「每次 madvise 前后局部观测到的正向下降之和」。 |

关键区别：

- `total_drop` 是**端到端净值**（起点 before 减终点 after）；
- `drop_sum` 是**逐次局部正向值之和**；
- 两者口径不同：窗口内若有中间 RSS 回升（进程其它内存活动），`drop_sum` 会大于 `total_drop`。本阶段三档均出现 `drop_sum > total_drop`，属预期，非 bug。

---

## 4. 实验配置

实验 driver：`build/bin/llama-kv-idle-swap-resume`（复用 5B-2 / 5C 的 resume driver，未新增）。

公共参数（small 档示例，mid / large 仅放大 `--ctx-size` 与 `-n`）：

```text
-m /root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
--batch-size 128
--ubatch-size 128
--seed 1
--temp 0
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 2
```

三档分别为：

| 档 | ctx-size | n |
|---|---|---|
| small | 512 | 64 |
| mid | 1024 | 128 |
| large | 2048 | 256 |

每档三组（base / nomadv / madv），同 seed / prompt / ctx / n，除目标 env 外完全同构。determinism：greedy、`temp=0`、固定 seed、`kv_unified=true`。

---

## 5. small / mid / large 矩阵

| 档 | ctx | n | `madvise_calls` | `madvise_bytes` | `rss_total_drop_kb` | `rss_drop_sum_kb` | `real_abnormal` |
|---|---|---|---|---|---|---|---|
| small | 512 | 64 | 5 | 19660800 | 13840 | 18972 | 0 |
| mid | 1024 | 128 | 9 | 35389440 | 29052 | 34184 | 0 |
| large | 2048 | 256 | 17 | 66846720 | 59292 | 65200 | 0 |

---

## 6. Correctness 结果

三档全部退出码为 0，且 full stdout / seq1 active / seq0 resume 三段在 base / nomadv / madv 三组逐字节一致：

```text
base_exit=0
nomadv_exit=0
madv_exit=0

full_nomadv=0   full_madv=0
seq1_nomadv=0   seq1_madv=0
seq0_nomadv=0   seq0_madv=0
```

> seq0 resume 段三档独立比对为 0 是关键：swap-in 缺陷只会反映在 seq0 输出上。放大 ctx / n 后 resume swap-in 仍逐字节恢复 KV 内容，madvise 未破坏 swap roundtrip。

---

## 7. madvise 规模趋势

```text
small : calls=5   bytes=19660800
mid   : calls=9   bytes=35389440
large : calls=17  bytes=66846720
```

- madvise 调用数随 ctx / n 单调放大（5 → 9 → 17）；
- advised bytes 同步放大（≈18.75 MiB → ≈33.75 MiB → ≈63.75 MiB）；
- 三档 `madvise_failures=0`，证明放大 workload 未引入 madvise 失败。

---

## 8. cumulative RSS drop 趋势

```text
small : total_drop_kb=13840  drop_sum_kb=18972
mid   : total_drop_kb=29052  drop_sum_kb=34184
large : total_drop_kb=59292  drop_sum_kb=65200
```

- `total_drop_kb` 随 madvise 规模单调放大（13840 → 29052 → 59292），证明窗口净 current RSS 下降随 idle swap madvise 体量放大；
- `drop_sum_kb` 同步单调放大（18972 → 34184 → 65200）；
- 三档均 `drop_sum > total_drop`，差值为窗口内中间 RSS 回升所致，两口径互不相等是预期行为；
- **RSS drop ≠ madvise_bytes**：`madvise_bytes` 是向内核 advise 的字节数，`total_drop` / `drop_sum` 是 `/proc` 观测到的常驻内存差值。两者不应相等——内核回收时机、页对齐、邻居跳过及进程其它内存活动都会使观测 drop 与 advise 量不一致。本阶段只声明 observed cumulative current RSS drop is positive 且随规模放大，不声明二者数值相等。

---

## 9. 结论

**Stage 5C-scale-B verified that cumulative current RSS drop scales with idle swap madvise volume while preserving resume correctness.**

- 新增累计 RSS telemetry（`rss_before_first_kb` / `rss_total_drop_kb` / `rss_drop_sum_kb`）在三档矩阵中正确透出、非负、可 grep；
- madvise 规模随 ctx / n 放大：`calls` 5 → 9 → 17，`bytes` 19660800 → 35389440 → 66846720，`failures=0`；
- 累计 current RSS drop 随 madvise 规模单调放大：`total_drop_kb` 13840 → 29052 → 59292，`drop_sum_kb` 18972 → 34184 → 65200；
- 三档 full / seq1 active / seq0 resume 在 base / nomadv / madv 全部逐字节一致，resume correctness 在放大 workload 下保持；
- 三档 `real_abnormal=0`，无真实 warning / error / NaN / backend failure / release violation。

---

## 10. 边界说明

- 只验证 **current RSS**，不验证 **peak RSS**；
- `total_drop` 与 `drop_sum` 是两个不同口径，不应混用，也不应相互等同（窗口内 RSS 回升会使二者不等）；
- 累计 drop 与 `madvise_bytes` 不相等，本阶段不声明数值等同；
- 不是 server-like workload（单进程 driver，固定 prompt / seq）；
- 不是生产级 PagedAttention，不声明生产级收益；
- 不支持非 F32 KV、不支持 `v_trans`；
- 不做多 idle seq / 多轮 resume 压力场景；
- RSS drop 的绝对值受 ctx / block_size / idle block 数量与页对齐影响，本阶段仅证明 drop 为正且随规模放大，不量化上限。

---

## 11. 后续方向

- 更大 ctx / block_size，量化更大的累计 RSS drop 并观察 `total_drop` 与 `drop_sum` 差值随规模的变化；
- 多 idle seq / 多轮 resume 压力场景；
- server-like workload 下的调度与累计 RSS 行为；
- prefetch / async swap-in，降低 resume swap-in 延迟；
- 探索 peak RSS（而不仅 current RSS）的优化路径。
