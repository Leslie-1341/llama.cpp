# Stage 9-D：Semi-real Multi-session KV Reclaim Result Panel

## 1. Purpose

本阶段整理 Stage 9-C 的 3-run median 结果，形成最终结果面板与汇报口径。

Stage 9-D 不引入新的源码修改，也不新增实验机制。它的目标是把已有结果整理成可以用于比赛汇报、阶段文档和最终报告的简洁证据链，重点回答：

1. 当前优化针对什么 workload；
2. S0-S5 分别验证什么机制；
3. RSS 下降多少；
4. 相对 KV buffer 总容量收益是多少；
5. 吞吐量是否下降；
6. resume first-token latency 是否可控；
7. safety / correctness signal 是否干净；
8. 对外汇报时应该如何表述，避免夸大为真实生产 workload。

---

## 2. Workload Scope

当前测试 workload 是：

```text
semi-real multi-session paused/resume workload
```

它是一个半真实、多会话、固定 timeline、可复现的 workload，用于模拟多轮聊天服务中常见的 session 状态变化：

```text
active decode -> paused idle -> resume pending -> resuming -> finished
```

### 2.1 Session 构成

Stage 9-B1 example driver 内置 4 个 session：

| Session | 类型                      | 作用                |
| ------- | ----------------------- | ----------------- |
| A       | `long_context_session`  | 模拟较长上下文会话，产生较多 KV |
| B       | `short_context_session` | 模拟短会话             |
| C       | `bursty_session`        | 模拟短暂活跃、暂停、恢复的突发会话 |
| D       | `short_context_session` | 模拟中途进入的新短请求       |

### 2.2 Workload 定位

该 workload 比早期 controlled workload 更接近真实多会话场景，因为 idle KV 来自 session paused 状态，而不是只通过参数强行指定 idle seq 数量。

但它仍不是完整生产 workload。当前没有覆盖：

1. `llama-server`；
2. HTTP request queue；
3. slot reuse；
4. continuous batching；
5. context shift；
6. streaming；
7. 真实用户 prompt trace；
8. 随机请求到达分布。

因此，本阶段结果应表述为：

```text
已在 semi-real multi-session paused/resume workload 中完成验证。
```

不应表述为：

```text
已在真实生产 server workload 中完成验证。
```

---

## 3. Case Matrix

Stage 9-C 使用 S0-S5 六组配置，每组运行 3 次，取 median。

| Case | 名称                            | 机制                                       | 目的                                   |
| ---- | ----------------------------- | ---------------------------------------- | ------------------------------------ |
| S0   | `S0_baseline_paged_off`       | baseline                                 | 不启用 paged / reclaim，作为基线             |
| S1   | `S1_paged_on_reclaim_off`     | paged bookkeeping only                   | 观察 paged 机制本身开销                      |
| S2   | `S2_tail_lazy_only`           | tail/lazy reclaim                        | 验证 unused/free/tail KV 物理页释放         |
| S3   | `S3_idle_swap_only`           | idle swap/madvise                        | 验证 paused session 的 idle-owned KV 释放 |
| S4   | `S4_idle_swap_prefetch_defer` | idle swap + prefetch + defer             | 验证 idle reclaim 的 resume latency 控制  |
| S5   | `S5_tail_idle_prefetch_defer` | tail/lazy + idle swap + prefetch + defer | 验证组合策略的最大收益和代价                       |

---

## 4. Core Result Panel

KV buffer 总容量参照值：

```text
511.750 MiB
```

该值用于计算：

```text
RSS drop vs S0 / KV buffer total
```

注意：这里的比例表示 **进程 RSS 下降量相当于 KV buffer 总容量的多少**，不是说 KV cache 逻辑容量减少了多少，也不是说 KV cache 内容本身释放了多少比例。

| Case | Mechanism                                | RSS drop vs S0 / MiB | RSS drop / KV capacity | TPS delta vs S0 | Resume delta vs S0 / ms | Safety |
| ---- | ---------------------------------------- | -------------------: | ---------------------: | --------------: | ----------------------: | ------ |
| S0   | baseline                                 |                0.000 |                   0.0% |          0.000% |                   0.000 | clean  |
| S1   | paged only                               |               -2.367 |                  -0.5% |         -1.318% |                  +1.672 | clean  |
| S2   | tail/lazy only                           |              319.758 |                  62.5% |         +0.738% |                  -0.931 | clean  |
| S3   | idle swap only                           |               84.109 |                  16.4% |         -0.819% |                  +6.070 | clean  |
| S4   | idle swap + prefetch + defer             |               84.152 |                  16.4% |         -0.647% |                  +2.975 | clean  |
| S5   | tail/lazy + idle swap + prefetch + defer |              403.621 |                  78.9% |         -0.807% |                  +3.729 | clean  |

核心结论：

```text
S5 组合策略在 3-run median 下实现约 403.621 MiB 进程 RSS 下降，
该下降量约等于 511.750 MiB KV buffer 总容量的 78.9%，
active TPS 仅下降约 0.807%，
resume first-token latency 平均增加约 3.729 ms。
```

---

## 5. Memory Result

### 5.1 Tail/lazy reclaim

S2 相对 S0：

```text
RSS drop vs S0 = 319.758 MiB
TPS delta vs S0 = +0.738%
Resume delta vs S0 = -0.931 ms
```

这说明 tail/lazy reclaim 可以有效释放 unused/free/tail KV 区域的物理驻留页。该部分主要对应还没有被有效 KV 内容占用，或不需要保留有效内容的 tail/free 区域。

该机制的特点是：

1. 不改变 KV buffer 的逻辑容量；
2. 不重新分配 KV buffer；
3. 通过 lazy clear / tail madvise 降低物理页驻留；
4. 后续如果写入这些区域，OS 可以重新分配物理页。

因此，S2 证明了 **unused/free/tail KV capacity reclaim** 的有效性。

### 5.2 Idle swap/madvise

S3 相对 S0：

```text
RSS drop vs S0 = 84.109 MiB
TPS delta vs S0 = -0.819%
Resume delta vs S0 = +6.070 ms
```

S3 中 idle swap/madvise 的核心统计为：

```text
swap_calls = 35
madvise_mib = 131.250 MiB
idle_owned_mib = 92.000 MiB
```

这说明 paused session 中的 idle-owned KV 能够被识别，并通过 swap/madvise 释放物理页，从而带来约 84 MiB 级别的进程 RSS 下降。

### 5.3 Prefetch + defer 对 idle reclaim 的作用

S4 相比 S3：

```text
RSS drop 基本保持不变；
swap_calls 仍为 35；
madvise_mib 仍为 131.250 MiB；
prefetch_step = 70；
resume delta 从 +6.070 ms 降至 +2.975 ms。
```

这说明 prefetch + defer 的作用不是扩大 RSS 收益，而是控制恢复 paused session 时的 first-token latency：

1. prefetch 在 resume 前逐步恢复即将需要的 KV；
2. defer 避免 resume first-token 阶段同步触发新的 idle swap-out；
3. 两者共同降低 resume latency 的额外开销。

### 5.4 组合策略

S5 相比 S0：

```text
RSS drop vs S0 = 403.621 MiB
TPS delta vs S0 = -0.807%
Resume delta vs S0 = +3.729 ms
```

S5 相比 S4 的额外 RSS 下降：

```text
403.621 - 84.152 = 319.469 MiB
```

该额外收益基本对应 tail/lazy reclaim 的贡献。

因此，S5 证明：

```text
tail/lazy reclaim 与 idle swap/madvise 是互补机制。
```

两者分别覆盖不同 KV 生命周期状态：

| 机制                | 处理对象                         | 语义                              |
| ----------------- | ---------------------------- | ------------------------------- |
| tail/lazy reclaim | unused/free/tail KV capacity | 丢弃未使用物理页，后续写入再 fault-in         |
| idle swap/madvise | used-but-idle KV content     | 保存/保护 idle KV，释放物理驻留，resume 前恢复 |

---

## 6. Throughput Result

S0 median active TPS：

```text
13.010
```

S5 median active TPS：

```text
12.911
```

S5 相对 S0：

```text
TPS delta vs S0 = -0.807%
```

这说明在当前 semi-real multi-session workload 下，组合策略获得约 403.6 MiB RSS 下降，同时 active TPS 下降低于 1%。

推荐表述：

```text
S5 在获得约 403.6 MiB RSS 下降的同时，active TPS 仅下降约 0.8%，未观察到明显吞吐损失。
```

不建议表述为：

```text
吞吐完全没有下降。
```

因为 3-run median 仍是小规模实验，且不同 workload 下 TPS 可能波动。

---

## 7. Resume First-token Latency Result

本阶段统计 A/B/C 三个会 resume 的 session。D 没有 resume，因此不纳入 resume average。

S0 baseline：

```text
Resume ABC avg = 73.472 ms
```

S3 idle swap only：

```text
Resume ABC avg = 80.014 ms
Resume delta vs S0 = +6.070 ms
```

S4 idle swap + prefetch + defer：

```text
Resume ABC avg = 76.696 ms
Resume delta vs S0 = +2.975 ms
```

S5 tail + idle + prefetch + defer：

```text
Resume ABC avg = 76.802 ms
Resume delta vs S0 = +3.729 ms
```

结论：

```text
单独 idle swap/madvise 会增加 resume first-token latency；
加入 prefetch + defer 后，resume latency 额外开销明显下降；
S5 组合策略的 resume first-token latency 额外开销约 3.7 ms，属于可控范围。
```

从机制上看：

1. S3 只做 idle swap/madvise，会在 resume 阶段暴露恢复成本；
2. S4 提前 prefetch，并在 resume first-token 阶段 defer 新 swap-out，因此降低首字延迟；
3. S5 叠加 tail/lazy 后，首字延迟略高于 S4，但仍明显低于 S3 idle-swap-only 的延迟开销。

---

## 8. Safety Result

Stage 9-C 共运行：

```text
3 runs × 6 cases = 18 workloads
```

全部正常退出：

```text
18 / 18 exit_code = 0
```

异常统计：

```text
run1    real_abnormal=0
run2    real_abnormal=0
run3    real_abnormal=0
TOTAL   real_abnormal=0
```

关键 safety signal：

| Signal                                   | Result |
| ---------------------------------------- | -----: |
| `real_abnormal`                          |      0 |
| `paged_swapped_active_visible_violation` |      0 |
| `paged_write_to_swapped_block`           |      0 |
| `paged_swap_madvise_failures`            |      0 |
| `active_visible_violation`               |      0 |

S3/S4/S5 中 idle swap/madvise 稳定触发：

```text
swap_calls = 35
madvise_mib = 131.250 MiB
idle_owned_mib = 92.000 MiB
```

S4/S5 中 prefetch 稳定触发：

```text
prefetch_step = 70
```

因此可以表述为：

```text
在该 semi-real workload 的 18 次运行中，未观察到真实异常、active-visible violation 或写入 swapped block，说明 active-needed safety、prefetch protection 和 defer 机制在该场景下稳定工作。
```

---

## 9. Correct Interpretation of “80%”

S5 的 3-run median RSS drop 为：

```text
403.621 MiB
```

KV buffer 总容量参照值为：

```text
511.750 MiB
```

因此：

```text
403.621 / 511.750 ≈ 78.9%
```

推荐表述：

```text
S5 的进程 RSS 下降量约等于当前 KV buffer 总容量的 78.9%，接近 80%。
```

或：

```text
The observed process RSS reduction is equivalent to about 78.9% of the KV buffer capacity.
```

不推荐表述：

```text
KV cache 释放了 80%。
```

原因：

1. RSS 是进程级指标，不是纯 KV resident 的直接百分比；
2. KV buffer 的逻辑容量没有缩小；
3. tail/lazy 释放的是物理驻留页，不是删除 KV buffer；
4. idle swap/madvise 释放的是 idle-owned KV 对应的 resident pages；
5. 该比例是以 KV buffer capacity 为参照的“等价值”，不是 KV 内容释放比例。

更严谨的中文表达：

```text
以当前 KV buffer 总容量为参照，S5 的最终进程 RSS 下降量达到约 79% 的 KV 容量等价值。
```

---

## 10. Recommended Reporting Wording

### 10.1 一句话结论

```text
在 semi-real 多会话暂停/恢复 workload 中，S5 组合策略实现了约 403.6 MiB 的 3-run median 进程 RSS 下降，约等于 KV buffer 总容量的 78.9%，同时 active TPS 仅下降约 0.8%，resume 首字延迟平均增加约 3.7 ms。
```

### 10.2 1 分钟汇报口径

```text
我们构造了一个 semi-real multi-session workload，用 4 个固定 session 模拟长会话、短会话、突发会话的 active、paused idle 和 resume 过程。基于 S0-S5 六组配置做 3-run median 验证后，组合策略 S5 同时启用 tail/lazy reclaim、idle swap/madvise、prefetch 和 resume defer。结果显示，S5 相对 baseline 的进程 RSS 下降约 403.6 MiB，约等于当前 KV buffer 总容量的 78.9%；active TPS 仅下降约 0.8%；resume first-token latency 平均增加约 3.7 ms。18 次运行全部 exit=0，real_abnormal=0，没有 active-visible violation，也没有写入 swapped block。这说明 tail/lazy 与 idle swap/madvise 可以互补叠加，prefetch + defer 可以把 resume 延迟控制在较小范围内。
```

### 10.3 更简短的答辩口径

```text
当前最强结果是在 semi-real 多会话 workload 下得到的：组合策略 S5 可以把进程 RSS 降低约 403.6 MiB，相当于 KV buffer 容量的 78.9%；吞吐只下降约 0.8%；恢复会话的首字延迟平均只增加约 3.7 ms；并且 18 次运行没有真实异常。
```

---

## 11. Limitations

本阶段仍有以下限制：

1. workload 是 semi-real example workload，不是完整生产 server workload；
2. timeline 固定，没有随机请求到达；
3. prompt 是内置固定文本，没有使用真实用户 trace；
4. 没有覆盖 `llama-server` 的 slot reuse、request queue、continuous batching、context shift 和 streaming；
5. 当前结果基于 Q4_K_M 模型、ctx-size 2048、parallel 4；
6. 3-run median 提高了稳定性，但仍不是大规模 benchmark；
7. S5 内部收益归因仍需要更细粒度的 residency audit 才能完全拆分 tail/lazy 与 idle swap 的 page-level 贡献；
8. 当前还没有实现统一 `LLAMA_KV_RECLAIM_POLICY=auto` 策略，S0-S5 仍是手动组合配置。

因此，对外表述时应避免：

```text
已完成真实生产 workload 验证。
```

应表述为：

```text
已在 semi-real multi-session paused/resume workload 中完成 3-run median 验证。
```

---

## 12. Next Step

Stage 9-D 之后，建议有两个方向。

### 12.1 更稳妥方向：Stage 9-E workload 扩展

继续在不改核心机制的前提下扩展 workload 变体，包括：

1. 更长 context；
2. 更多并发 session；
3. 更多 paused session；
4. 更频繁 resume；
5. 不同 long/short/bursty session 比例。

目标是验证 S5 结论是否对 workload 变化稳定。

### 12.2 工程推进方向：Stage 10 unified auto policy

在已有手动 S0-S5 组合验证基础上，进一步实现统一策略：

```text
LLAMA_KV_RECLAIM_POLICY=off|tail_only|idle_only|auto
```

其中 `auto` 根据 KV 生命周期状态、idle age、resume pending、memory pressure 和 active-needed safety 自动选择 reclaim 行为。

Stage 10 应建立在 Stage 9-C/9-D 的结果之上，但需要注意它会重新引入源码风险，因此进入前应先确认当前文档和结果已经完整提交。

---

## 13. Final Summary

Stage 9-D 将 Stage 9-C 的 3-run median 结果整理为结果面板，核心结论如下：

1. 当前 workload 是 semi-real multi-session paused/resume workload；
2. S5 组合策略获得最大内存收益；
3. S5 相对 baseline 的进程 RSS 下降为 `403.621 MiB`；
4. 该下降量约等于 `511.750 MiB` KV buffer 总容量的 `78.9%`；
5. S5 active TPS 仅下降 `0.807%`；
6. S5 resume first-token latency 平均增加 `3.729 ms`；
7. 18 次运行全部 `exit=0`；
8. `real_abnormal=0`；
9. `active_visible_violation=0`；
10. `write_swapped=0`。

最终可概括为：

```text
在 semi-real 多会话暂停/恢复场景中，tail/lazy reclaim + idle swap/madvise + prefetch/defer 的组合策略可以稳定获得接近 KV buffer 容量 80% 等价值的进程 RSS 下降，同时保持吞吐基本不变，并将 resume 首字延迟额外开销控制在约 4 ms 以内。
```
