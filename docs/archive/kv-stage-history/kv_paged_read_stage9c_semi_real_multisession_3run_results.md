# Stage 9-C：Semi-real Multi-session Workload 3-run Median Results

## 1. 背景与目标

本阶段在 Stage 9-B1 / Stage 9-B2 的基础上，对 semi-real multi-session workload 进行 3-run median 稳定性验证。

Stage 9-B1 已新增 example driver：

```text
build/bin/llama-kv-semi-real-multisession
```

该 driver 在 example 层构造多个 session 的固定调度流程，用于模拟多会话聊天场景中的 active、paused idle、resume pending 和 resuming 状态。

Stage 9-B2 已新增 S0-S5 single-run matrix runner / parser：

```text
scripts/kv_stage9b2_semi_real_matrix.sh
scripts/parse_stage9b2_semi_real_matrix.py
```

Stage 9-C 的目标是将 single-run 结果升级为 3-run median 结果，验证以下问题：

1. semi-real workload 下，tail/lazy、idle swap/madvise、prefetch/defer 组合是否稳定；
2. S5 组合策略是否稳定带来最高 RSS 收益；
3. 吞吐量是否基本不下降；
4. resume first-token latency 是否保持在可控范围；
5. 是否存在真实异常、active-visible violation 或写入 swapped block。

---

## 2. Workload 定义

本阶段测试的是：

```text
semi-real multi-session paused/resume workload
```

它不是完整真实 server workload，而是一个半真实、固定调度、可复现、可归因的多会话 workload。

### 2.1 Session 构成

driver 内置 4 个 session：

| Session | 类型                      | 用途                |
| ------- | ----------------------- | ----------------- |
| A       | `long_context_session`  | 模拟较长上下文会话，产生较多 KV |
| B       | `short_context_session` | 模拟短会话             |
| C       | `bursty_session`        | 模拟短活跃、暂停、恢复的突发会话  |
| D       | `short_context_session` | 模拟中途进入的新短请求       |

### 2.2 Session 状态

每个 session 在执行过程中可能经历以下状态：

```text
WAITING
PREFILL
ACTIVE_DECODE
PAUSED_IDLE
RESUME_PENDING
RESUMING
FINISHED
```

其中：

* `ACTIVE_DECODE`：该 session 正在生成 token，其 KV 必须保持 active-visible；
* `PAUSED_IDLE`：该 session 暂停生成，但历史 KV 仍保留在 KV cache 中；
* `RESUME_PENDING`：该 session 即将恢复生成，适合触发 prefetch；
* `RESUMING`：该 session 开始恢复生成，需要关注 first-token latency；
* `FINISHED`：该 session 达到目标 decode token 数。

### 2.3 与真实 workload 的关系

当前 workload 模拟的是多会话聊天服务中的一种典型模式：

```text
一个长会话暂停；
其他短会话继续生成；
部分 paused session 稍后恢复；
新的短请求中途进入。
```

它比早期 controlled workload 更接近真实多会话场景，因为 idle KV 不再完全依赖手工指定的 idle seq 数量，而是来自 session 的 paused 状态。

但它仍不是完整生产 workload，原因包括：

1. 未接入 `llama-server`；
2. 未引入 HTTP 请求；
3. 未模拟真实 request queue；
4. 未使用真实 continuous batching；
5. 未覆盖 slot reuse / context shift / streaming 等 server 行为；
6. prompt 和 timeline 是固定内置的；
7. 请求到达不是随机分布。

因此，本阶段结果应表述为：

```text
semi-real multi-session workload result
```

不应表述为：

```text
production server benchmark result
```

---

## 3. 实验配置

### 3.1 公共运行参数

所有 case 使用相同公共参数：

```text
-n 128
--ctx-size 2048
--batch-size 128
--ubatch-size 128
--seed 1
--temp 0
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 4
```

模型路径：

```text
/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

binary：

```text
build/bin/llama-kv-semi-real-multisession
```

结果目录：

```text
/root/oscomp/kv_logs/stage9c_3run_median
```

### 3.2 Case Matrix

本阶段使用 S0-S5 六组配置。

| Case | 名称                            | 目的                              |
| ---- | ----------------------------- | ------------------------------- |
| S0   | `S0_baseline_paged_off`       | baseline，不启用 paged / reclaim    |
| S1   | `S1_paged_on_reclaim_off`     | 观察 paged bookkeeping 本身开销       |
| S2   | `S2_tail_lazy_only`           | 单独验证 tail/lazy reclaim          |
| S3   | `S3_idle_swap_only`           | 单独验证 idle swap/madvise          |
| S4   | `S4_idle_swap_prefetch_defer` | 验证 idle swap + prefetch + defer |
| S5   | `S5_tail_idle_prefetch_defer` | 验证 tail/lazy 与 idle swap 组合策略   |

### 3.3 关键机制开关

S2 启用：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1
```

S3 启用：

```text
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
```

S4 在 S3 基础上启用：

```text
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=4
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=4
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

S5 在 S4 基础上额外启用：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1
```

---

## 4. 运行完整性检查

Stage 9-C 共执行：

```text
3 runs × 6 cases = 18 workloads
```

所有 case 均正常退出：

| Run  | S0 | S1 | S2 | S3 | S4 | S5 |
| ---- | -: | -: | -: | -: | -: | -: |
| run1 |  0 |  0 |  0 |  0 |  0 |  0 |
| run2 |  0 |  0 |  0 |  0 |  0 |  0 |
| run3 |  0 |  0 |  0 |  0 |  0 |  0 |

异常统计：

```text
run1    real_abnormal=0
run2    real_abnormal=0
run3    real_abnormal=0
TOTAL   real_abnormal=0
```

因此，本阶段可确认：

```text
18 个 workload 均正常完成；
未观察到真实异常；
未观察到 crash / assert / segfault / backend failure；
未观察到 active-visible violation；
未观察到 write-to-swapped-block。
```

---

## 5. 3-run Median 结果

### 5.1 Derived median summary

以 S0 baseline 为参照，得到以下 3-run median 派生结果：

| Case                              | run_count | RSS drop vs S0 / MiB | TPS delta vs S0 / % | Resume ABC avg / ms | Resume delta vs S0 / ms |
| --------------------------------- | --------: | -------------------: | ------------------: | ------------------: | ----------------------: |
| S0 baseline                       |         3 |                0.000 |               0.000 |              73.472 |                   0.000 |
| S1 paged only                     |         3 |               -2.367 |              -1.318 |              75.790 |                  +1.672 |
| S2 tail/lazy only                 |         3 |              319.758 |              +0.738 |              72.879 |                  -0.931 |
| S3 idle swap only                 |         3 |               84.109 |              -0.819 |              80.014 |                  +6.070 |
| S4 idle swap + prefetch + defer   |         3 |               84.152 |              -0.647 |              76.696 |                  +2.975 |
| S5 tail + idle + prefetch + defer |         3 |              403.621 |              -0.807 |              76.802 |                  +3.729 |

### 5.2 Raw median summary

| Case                              | RSS / KB |    TPS | Wall / ms | A first / ms | B first / ms | C first / ms | swap calls | madvise / MiB | idle-owned / MiB | internal RSS drop / MiB | prefetch steps | abnormal |
| --------------------------------- | -------: | -----: | --------: | -----------: | -----------: | -----------: | ---------: | ------------: | ---------------: | ----------------------: | -------------: | -------: |
| S0 baseline                       |  8683532 | 13.010 | 27591.947 |       71.903 |       76.351 |       72.948 |          0 |         0.000 |            0.000 |                   0.000 |              0 |        0 |
| S1 paged only                     |  8685828 | 12.744 | 27994.135 |       72.904 |       77.660 |       71.992 |          0 |         0.000 |            0.000 |                   0.000 |              0 |        0 |
| S2 tail/lazy only                 |  8356100 | 13.007 | 27471.514 |       71.912 |       75.827 |       70.501 |          0 |         0.000 |            0.000 |                   0.000 |              0 |        0 |
| S3 idle swap only                 |  8597424 | 12.780 | 27975.665 |       77.261 |       87.099 |       76.076 |         35 |       131.250 |           92.000 |                  81.707 |              0 |        0 |
| S4 idle + prefetch + defer        |  8597448 | 12.851 | 27862.847 |       80.407 |       79.400 |       70.941 |         35 |       131.250 |           92.000 |                  81.699 |             70 |        0 |
| S5 tail + idle + prefetch + defer |  8270236 | 12.911 | 27712.703 |       81.461 |       76.575 |       70.618 |         35 |       131.250 |           92.000 |                  17.848 |             70 |        0 |

---

## 6. RSS 结果分析

### 6.1 S1：paged bookkeeping 本身没有内存收益

S1 相比 S0：

```text
RSS drop vs S0 = -2.367 MiB
TPS delta vs S0 = -1.318%
```

这说明仅打开 paged bookkeeping，但不启用 reclaim，不会带来内存收益，反而可能引入轻微开销。

该结果符合预期，因为 S1 只是打开 paged metadata / graph / nonidentity 等路径，但没有执行 tail/lazy reclaim，也没有执行 idle swap/madvise。

### 6.2 S2：tail/lazy 单独带来约 319.8 MiB RSS 下降

S2 相比 S0：

```text
RSS drop vs S0 = 319.758 MiB
TPS delta vs S0 = +0.738%
Resume delta vs S0 = -0.931 ms
```

这说明 tail/lazy reclaim 能够有效释放 unused/free/tail 部分的物理驻留内存。

该路径主要对应：

```text
未写入或不再需要保留有效 KV 内容的 tail/free 区域；
通过 lazy clear / tail madvise 降低物理页驻留；
不需要保存具体 KV 内容；
后续如果写入，OS 会重新分配物理页。
```

因此，S2 证明了 tail/lazy 对 “未使用 KV 容量” 的 RSS 优化是有效且低开销的。

### 6.3 S3：idle swap/madvise 单独带来约 84.1 MiB RSS 下降，但增加 resume 延迟

S3 相比 S0：

```text
RSS drop vs S0 = 84.109 MiB
TPS delta vs S0 = -0.819%
Resume delta vs S0 = +6.070 ms
```

S3 中：

```text
swap_calls = 35
madvise_mib = 131.250 MiB
idle_owned_mib = 92.000 MiB
```

这说明 idle-owned block 能够被识别并执行 swap/madvise，带来约 84 MiB 级别的最终进程 RSS 下降。

但由于 S3 不启用 prefetch/defer，恢复 paused session 时需要同步处理更多 swapped KV，导致 resume first-token latency 明显增加。

因此，S3 证明：

```text
idle swap/madvise 能释放 used-but-idle KV；
但如果没有 prefetch/defer，resume latency 会变差。
```

### 6.4 S4：prefetch + defer 将 resume 延迟从 +6.1 ms 降至 +3.0 ms

S4 相比 S0：

```text
RSS drop vs S0 = 84.152 MiB
TPS delta vs S0 = -0.647%
Resume delta vs S0 = +2.975 ms
```

S4 相比 S3：

```text
RSS 收益基本相同；
swap_calls 稳定为 35；
madvise_mib 稳定为 131.250 MiB；
prefetch_step = 70；
resume delta 从 +6.070 ms 降至 +2.975 ms。
```

这说明 prefetch + defer 的作用是有效的：

```text
prefetch 在 resume 前逐步恢复即将需要的 KV；
defer 避免 resume first-token 阶段同步触发新的 idle swap-out；
从而降低 resume first-token latency。
```

S4 证明 idle swap/madvise 的内存收益可以保留，同时 resume 延迟代价可以被控制。

### 6.5 S5：tail/lazy + idle swap/madvise 可以叠加，RSS 收益最大

S5 相比 S0：

```text
RSS drop vs S0 = 403.621 MiB
TPS delta vs S0 = -0.807%
Resume delta vs S0 = +3.729 ms
```

S5 相比 S4：

```text
额外 RSS drop ≈ 403.621 - 84.152 = 319.469 MiB
```

该额外收益基本对应 tail/lazy path 的贡献。

因此，S5 证明：

```text
tail/lazy reclaim 与 idle swap/madvise 是互补机制；
tail/lazy 处理 unused/free/tail KV 容量；
idle swap/madvise 处理 used-but-idle KV 内容；
二者组合后可以获得更大的 RSS 收益。
```

---

## 7. 相对 KV 容量与总进程 RSS 的解释

如果沿用 Stage 8E 中约 `511.750 MiB` 的 KV buffer 总容量作为参照，则 S5 的 3-run median RSS 下降量为：

```text
403.621 MiB / 511.750 MiB ≈ 78.9%
```

因此，可以表述为：

```text
在该 semi-real multi-session workload 下，S5 组合策略的最终进程 RSS 下降量约等于 KV buffer 总容量的 78.9%，接近 80%。
```

但不应表述为：

```text
KV cache 本身释放了 80%。
```

更准确的说法是：

```text
以当前 KV buffer 总容量为参照，S5 的进程 RSS 下降量达到约 79% 的 KV 容量等价值。
```

如果以整个进程 RSS 为分母，则 S5 相对 S0 的总进程 RSS 降幅约为：

```text
403.621 MiB / (8683532 KB / 1024) ≈ 4.8%
```

因此，推荐同时报告两个口径：

| 口径               |     数值 | 含义                          |
| ---------------- | -----: | --------------------------- |
| 相对 KV buffer 总容量 | ≈78.9% | 说明 KV 相关内存优化强度高             |
| 相对总进程 RSS        |  ≈4.8% | 说明整个 llama.cpp 进程的真实 RSS 降幅 |

---

## 8. 吞吐量结果分析

S5 相比 S0：

```text
TPS delta vs S0 = -0.807%
```

这说明在 3-run median 下，组合策略的吞吐量下降低于 1%。

从 raw median 看：

```text
S0 active_tps = 13.010
S5 active_tps = 12.911
```

因此可以表述为：

```text
S5 在获得约 403.6 MiB RSS 下降的同时，active TPS 仅下降约 0.8%，未观察到明显吞吐损失。
```

注意，该结果来自 semi-real example workload，不代表完整 server workload 下的吞吐结论。

---

## 9. Resume first-token latency 结果分析

本阶段关注 A/B/C 三个会 resume 的 session。D 没有 resume，因此不计入 resume average。

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
prefetch + defer 能显著缓解该问题；
组合 tail/lazy 后，S5 的 resume first-token latency 额外开销仍保持在约 3.7 ms，属于可控范围。
```

---

## 10. Safety / Correctness Signals

Stage 9-C 中所有 case 均满足：

```text
exit_code = 0
real_abnormal = 0
paged_swapped_active_visible_violation = 0
paged_write_to_swapped_block = 0
```

S3/S4/S5 中：

```text
swap_calls = 35
madvise_mib = 131.250 MiB
idle_owned_mib = 92.000 MiB
```

S4/S5 中：

```text
prefetch_step = 70
```

这说明：

1. idle swap/madvise 在 S3/S4/S5 中稳定触发；
2. prefetch 在 S4/S5 中稳定触发；
3. active-visible safety gate 没有被破坏；
4. 没有发生对 swapped block 的写入；
5. 没有观察到真实异常。

需要注意：

```text
paged_swap_rss_total_drop_mib
```

是 idle swap/madvise 内部统计的 RSS drop，不等同于相对 S0 的总进程 RSS drop。尤其 S5 中该字段为 `17.848 MiB`，不应被解读为 S5 总收益只有 17.8 MiB。S5 的最终总收益应使用：

```text
S0 rss_kb - S5 rss_kb
```

对应 3-run median derived result：

```text
403.621 MiB
```

---

## 11. 阶段结论

Stage 9-C 的核心结论如下：

1. 在 semi-real multi-session paused/resume workload 下，S0-S5 共 18 次运行全部正常退出，`real_abnormal=0`。
2. Tail/lazy reclaim 单独带来约 `319.758 MiB` RSS 下降，且没有观察到吞吐损失。
3. Idle swap/madvise 单独带来约 `84.109 MiB` RSS 下降，但会使 resume first-token latency 增加约 `6.070 ms`。
4. 加入 prefetch + defer 后，idle swap/madvise 的 RSS 收益基本保持，同时 resume first-token latency 额外开销从约 `6.070 ms` 降至约 `2.975 ms`。
5. Tail/lazy 与 idle swap/madvise 可以叠加。S5 组合策略达到约 `403.621 MiB` 的 RSS 下降，约等于当前 KV buffer 总容量的 `78.9%`。
6. S5 的 active TPS 相比 S0 仅下降约 `0.807%`，resume first-token latency 平均增加约 `3.729 ms`，整体性能代价可控。
7. S5 未出现 active-visible violation、write-to-swapped-block 或真实异常，说明现有 active-needed safety、prefetch protection 和 defer 机制在该 workload 下可以稳定工作。

简要概括：

```text
Stage 9-C 证明：在半真实多会话暂停/恢复场景中，tail/lazy + idle swap/madvise + prefetch/defer 的组合策略可以稳定获得约 403.6 MiB RSS 下降，吞吐基本不变，resume 首字延迟仅小幅增加。
```

---

## 12. 限制

本阶段仍存在以下限制：

1. 当前 workload 是 semi-real example workload，不是完整 llama-server workload；
2. session timeline 固定，未覆盖随机请求到达；
3. prompt 内置，未使用真实用户 trace 或外部 prompt 数据集；
4. 未覆盖 server slot reuse、continuous batching、context shift、streaming 等复杂行为；
5. 当前结果基于 Q4_K_M 模型、ctx-size 2048、parallel 4，不代表所有模型和上下文长度；
6. 3-run median 提高了稳定性，但仍不是大规模 benchmark；
7. S5 中 tail/lazy 与 idle swap/madvise 的内部 RSS attribution 仍需要更细粒度的 KV residency audit 进一步拆解。

因此，正式汇报时应避免声称：

```text
已完成真实生产 workload 验证。
```

建议表述为：

```text
已在 semi-real multi-session paused/resume workload 中完成 3-run median 验证。
```

---

## 13. 下一步方向

后续可选方向：

### 13.1 Stage 9-D：结果文档与图表整理

将 Stage 9-C 的 3-run median 结果整理为比赛汇报用图表，包括：

1. RSS drop vs S0；
2. TPS delta vs S0；
3. Resume first-token latency delta；
4. S0-S5 case matrix；
5. safety summary。

### 13.2 Stage 9-E：更多 workload 变体

增加多个 semi-real workload 变体，例如：

1. 更长上下文；
2. 更多并发 session；
3. 更多 paused session；
4. 更频繁 resume；
5. 不同 prompt 长度组合。

目标是验证当前结论是否对 workload 变化稳定。

### 13.3 Stage 10：统一策略或 server-like workload

进一步方向包括：

1. 实现统一 reclaim policy；
2. 引入真实 trace 或 server-like scheduler；
3. 接入更接近 llama-server 的 slot/request 行为；
4. 做更完整的性能/内存 trade-off 分析。

---

## 14. 推荐汇报口径

对老师/队友可以这样概括：

```text
我们在 Stage 9-C 中构造了一个 semi-real multi-session paused/resume workload，用 4 个固定 session 模拟长会话、短会话、突发会话的 active、paused 和 resume 行为。基于 S0-S5 六组配置做 3-run median 验证后，组合策略 S5 实现了约 403.6 MiB 的最终进程 RSS 下降，约等于当前 KV buffer 总容量的 78.9%。同时 active TPS 仅下降约 0.8%，resume 首字延迟平均增加约 3.7 ms，18 次运行均 exit=0 且 real_abnormal=0。结果说明 tail/lazy reclaim 与 idle swap/madvise 可以互补叠加，prefetch + defer 能控制 resume 延迟，是目前最完整、最接近多会话真实使用场景的一组验证结果。
```
