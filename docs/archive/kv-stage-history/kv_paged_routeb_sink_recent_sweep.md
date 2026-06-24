# Stage 4C-2A — sink+recent 参数扫描结果（approximate policy）

前置：

- [docs/kv_paged_trace_sim_stage4c1_results.md](kv_paged_trace_sim_stage4c1_results.md)
- 脚本：[scripts/kv_paged_trace_sim.py](../scripts/kv_paged_trace_sim.py)
- trace：`/root/oscomp/kv_logs/paged_stage4c_trace_active_ctx4096_n512/trace.txt`

本文是 Stage 4C-2A 的结果文档：在 Stage 4C-1 仿真器之上，对 `sink_recent_approx` 策略做 `sink_blocks` × `recent_blocks` 参数扫描，估算理论 RSS 收益。

结论先行：`sink+recent` 是一个 **approximate（近似）策略，不保证 sha256 一致**。它的收益完全来自“不再读取中间历史 block”。在 exact full attention 要求下这些 block 仍会被读回，收益归零；只有放弃中间历史读、并 release/madvise 这些 cold middle block 时，下表的 RSS 节省才成立。

---

## 0. 基线

当前 trace 的峰值占用：

```
peak blocks_in_use = 33  →  约 132 MiB KV resident（4 MiB/block）
```

这也是 `no_swap` / `stage4a_release_only` 等所有 exact 策略的 resident 水平（参见 Stage 4C-1 结论：exact 场景下安全 swap 收益为 0）。

---

## 1. sink+recent 参数扫描

`sink_recent_approx` 保留最旧 `sink` 个 attention sink block + 最近 `recent` 个 block，把中间历史当冷数据换出/释放。

关键观察：当 `recent` 足够大（32 / 64）时，`sink+recent` 的保留窗口已经覆盖了几乎全部 live KV block，因此 **saved_peak=0**——没有中间历史可丢弃。收益只在 `recent` 较小时出现：

| sink | recent | peak resident | saved_peak | 说明 |
|---|---|---|---|---|
| 1 | 4  | 20 MiB | 112 MiB | 最激进，保留窗口最小 |
| 2 | 4  | 24 MiB | 108 MiB | aggressive 候选 |
| 2 | 8  | 40 MiB |  92 MiB | balanced 候选 |
| 2 | 16 | 72 MiB |  60 MiB | conservative 候选 |
| 4 | 16 | 80 MiB |  52 MiB | sink 更宽，收益略降 |
| * | 32 | 132 MiB | 0 MiB | 窗口覆盖全部 live block |
| * | 64 | 132 MiB | 0 MiB | 窗口覆盖全部 live block |

（peak resident + saved_peak ≈ 132 MiB 基线；差异来自 sink/recent 窗口与 block 边界的取整。）

---

## 2. 推荐候选参数

| 档位 | 参数 | peak resident | saved_peak |
|---|---|---|---|
| aggressive   | `sink=2, recent=4`  | 24 MiB | 108 MiB |
| balanced     | `sink=2, recent=8`  | 40 MiB |  92 MiB |
| conservative | `sink=2, recent=16` | 72 MiB |  60 MiB |

三档都以 `sink=2` 为基准（保留两块 attention sink），用 `recent` 调节激进程度：`recent` 越小 RSS 越省、对中间历史的近似越激进。

---

## 3. 正确性边界（重点）

- `sink+recent` 是 **approximate policy**：它主动丢弃中间历史 block 的读，改变了注意力可见的 KV，**不保证输出 / sha256 一致**。上表数字是**理论 RSS 上限收益估计**，不是 correctness-preserving 的运行结果。
- 收益来源单一：**不再读取中间历史 block**。因此实现时可以对这些 cold middle block 做 release / madvise，把 RSS 真正降下来。
- 反过来，**若要求 exact full attention**，中间历史 block 每个 decode step 仍会被读回，cold 集合不存在，上述 saved_peak 全部归零——这与 Stage 4C-1 中 `active_read_blocks` 覆盖 `blocks_in_use` 的观测一致。

一句话：sink+recent 的 RSS 收益和 exact correctness 是互斥的，二选一。

---

## 4. 下一步建议

继续做 **multi-request / idle request swap 仿真**：在 exact correctness 前提下，寻找天然产生冷 KV 的场景——非活跃序列、长 idle gap 的请求，其 block 在不被读的窗口内可以安全 swap/release，而不像单请求 full attention 那样每步都被读。目标是找到“不牺牲正确性也能腾出冷 KV”的路径，作为 sink+recent 近似路线之外的 exact 备选。

---

## 5. 复现实验

```bash
python3 scripts/kv_paged_trace_sim.py \
  --trace /root/oscomp/kv_logs/paged_stage4c_trace_active_ctx4096_n512/trace.txt \
  --sink-blocks 2 --recent-blocks 8
```

扫描时改变 `--sink-blocks` / `--recent-blocks` 即可复现上表各行。
