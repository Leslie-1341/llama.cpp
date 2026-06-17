# Stage 4C-1 — KV block cold/hot policy offline simulator 结果（草稿）

前置：

- [docs/kv_paged_read_stage4b_rss_results.md](kv_paged_read_stage4b_rss_results.md)
- Stage 4C-0 trace 开关：`LLAMA_KV_PAGED_TRACE=1`（telemetry only）
- 脚本：[scripts/kv_paged_trace_sim.py](../scripts/kv_paged_trace_sim.py)
- trace：`/root/oscomp/kv_logs/paged_stage4c_trace_active_ctx4096_n512/trace.txt`
- 摘要 CSV：`/root/oscomp/kv_logs/paged_stage4c_trace_active_ctx4096_n512/sim_summary.csv`

本文是 Stage 4C-1 的**结果文档草稿**。Stage 4C-1 只新增一个离线 Python 仿真器，回放 Stage 4C-0 采集的 block 访问轨迹，估算不同冷热置换策略的理论 RSS 收益与 swap I/O 代价。**不改 C++，不 build，不跑 llama，不 commit。**

结论先行：**在当前单请求 exact full attention 场景下，所有保证正确性的策略（no_swap / stage4a_release_only / lru_idle_threshold / anti_thrash_cooldown / belady_oracle）都无法 swap 出任何 live history，RSS 收益为 0**。原因是每个 decode step 的 `active_read_blocks` 完整覆盖了 `blocks_in_use`：历史每一块每步都被读，没有“空闲块”可换出，连 Belady 最优也无能为力。唯一产生 RSS 收益的是 `sink_recent_approx`，但它**主动丢弃中间历史的读**，因此**不保证 sha256 一致**，只能作为 RSS 上限估计。

---

## 0. 仿真设置

- trace：`ctx=4096`、`n=512`，共 513 个 decode step。
- 仿真器**只用 `active_read_blocks`**（真实已写入的读集合）作为读集合，**不用 `read_blocks`**（padded graph row_idx 覆盖范围，含 reserve/padding block）。
- 每步的“live set” = `active_read_blocks ∪ write_block`。
- block 大小默认 `--block-bytes 4194304`（≈4 MiB/block）、`--block-size 16`。
- 默认参数：`--idle-threshold 8 --cooldown 8 --sink-blocks 2 --recent-blocks 4`，`belady_budget` 自动取 `peak_blocks_in_use // 2 = 16`。

swap 计数语义：

- 从 `swapped` 集合的进出沿计数。一个块**首次分配**（写入新块）不计 swap-in（它从未在盘上）。
- swap-out = resident→swapped；swap-in = swapped→resident。
- thrash：某次 swap-in 的块在 ≤K 步前刚被 swap-out（K=1 / K=4）。

---

## 1. 仿真结果（默认参数）

```
trace steps=513 block_bytes=4194304 peak_blocks_in_use=33 belady_budget=16
```

| policy | avg_resident | peak_resident | resident_mb_avg | resident_mb_peak | saved_mb_avg | saved_mb_peak | swap_out | swap_in | swap_io_gib | thrash_4 | thrash_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no_swap | 16.84 | 33 | 67.4 | 132.0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 | 0.0 |
| stage4a_release_only | 16.84 | 33 | 67.4 | 132.0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 | 0.0 |
| lru_idle_threshold | 16.84 | 33 | 67.4 | 132.0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 | 0.0 |
| anti_thrash_cooldown | 16.84 | 33 | 67.4 | 132.0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 | 0.0 |
| sink_recent_approx | 5.58 | 6 | 22.3 | 24.0 | 45.1 | 108.0 | 27 | 0 | 0.105 | 0 | 0.0 |
| belady_oracle | 16.84 | 33 | 67.4 | 132.0 | 0.0 | 0.0 | 0 | 0 | 0.0 | 0 | 0.0 |

完整指标见 `sim_summary.csv`。

---

## 2. 为什么 exact full attention 下 swap live history 必然 thrashing / 无收益

trace 的每一行都满足：`active_read_blocks` 等于 `0..blocks_in_use-1`（block 1 因 shift 映射缺位，其余连续）。也就是说：

- 每个 step，所有历史块都被读；
- 唯一的“新块”是当前 `write_block`（最近一块）。

在这种访问模式下：

- **idle-threshold（lru / anti-thrash）**：任何块的“空闲步数”永远是 0（每步都被读），永远不会超过阈值 → 永不换出 → swap=0、收益=0。即使把 `--idle-threshold 0` 仍然如此，因为“本步被读”的块被显式保护。
- **belady_oracle**：最优换出要求换出“下次最晚再用”的块，但**所有 resident 块本步都在被读**，没有可换出的候选；即便强行换出，下一步立刻要 swap-in → 这正是 thrashing 的根因。Stage 4B-RSS 实测 `ctx=4096/n=512` 出现 `swap_out=8000/swap_in=8000` 就是强制换出 live history 撞上这个访问模式的直接后果。
- **stage4a_release_only**：只回收 free/unused block，对 live history 不动 → 与 no_swap 同样的 resident，但 swap=0，是 exact 场景下最稳的策略（resident == `blocks_in_use`）。

一句话：**只要 `active_read_blocks` 覆盖 `blocks_in_use`，live history 就没有安全的换出窗口**；任何换出都会在下一步被强制换回，表现为高 swap I/O、低 RSS drop 的 thrashing——与 Stage 4B-RSS 的强制 swap 观测一致。

---

## 3. sink_recent_approx：近似策略（不保证 sha256 一致）

`sink_recent_approx` 保留最旧 `sink_blocks`（attention sink）+ 最近 `recent_blocks`，把中间历史当冷数据换出。它在默认参数下把 resident 从 33 块（peak）压到 6 块，估算 peak RSS 节省 ~108 MiB、avg ~45 MiB。

但必须强调：

- 它**主动丢弃中间历史块的读**（这些块本步其实在被 attention 读），因此**改变了注意力的可见 KV，不保证输出 / sha256 一致**。
- 表中数字是**理论 RSS 上限收益估计**，不是 correctness-preserving 的运行结果。
- 若要在真实推理里使用，需要配合 sparse / windowed attention 等语义变更，超出当前 exact full attention 的范围。

swap_in=0 是因为该策略一旦换出某块就不再换回（中间历史不会重新进入 hot 集合），所以没有 thrash；代价是正确性。

---

## 4. 对下一阶段的提示

- 在**单请求 exact full attention** 下，KV block swap 几乎没有安全收益空间；继续在该场景强推 swap 只会复现 thrashing。
- 真正的收益要来自以下方向之一（均超出本仿真的 exact 假设）：
  1. 多请求 / 多序列：非活跃序列的 block 才是安全的冷数据；
  2. 语义层 sparse / sliding-window / sink attention：让 `active_read_blocks` 不再覆盖全历史；
  3. 仅回收 free block（Stage 4A 路线），不碰 live history。

建议把 trace 采集扩展到多请求 / 长 idle gap 场景后再跑本仿真，验证 idle-threshold / belady 在那里是否出现非零安全收益。

---

## 5. 复现实验

```bash
python3 scripts/kv_paged_trace_sim.py \
  --trace /root/oscomp/kv_logs/paged_stage4c_trace_active_ctx4096_n512/trace.txt \
  --csv   /root/oscomp/kv_logs/paged_stage4c_trace_active_ctx4096_n512/sim_summary.csv
```

可调参数：`--block-bytes --block-size --recent-blocks --sink-blocks --idle-threshold --cooldown --belady-budget`。
