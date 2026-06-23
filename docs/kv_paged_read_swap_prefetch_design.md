# KV Cache Swap + Prefetch 机制设计说明

## 1. 背景与目标

在大模型推理过程中，KV cache 会随着上下文长度和并发请求数量增长而持续占用内存。对于长上下文、多会话并发场景，KV cache 可能成为进程 RSS 的重要组成部分。

本项目的目标不是简单减少 KV cache 的逻辑容量，而是在运行时识别暂时不用的 idle KV block，将其物理内存释放给操作系统，并在请求恢复时再换入，从而降低进程 current RSS。

核心目标可以概括为：

```text
在不破坏推理正确性的前提下，
对 idle request 的 KV block 做 swap-out + madvise，
在 request resume 前后做 swap-in / prefetch，
以 RSS 下降换取可控的性能代价。
```

本机制主要面向：

```text
多会话；
长上下文；
请求存在 pause / idle / resume；
KV cache 占进程内存比例较高；
内存压力比极限吞吐更重要。
```

不适合主要强调：

```text
单请求 full attention；
所有 KV 都马上会被读取；
没有 idle request；
极低延迟优先且内存压力不明显。
```

---

## 2. 总体设计思路

整体机制由五部分组成：

```text
1. Paged KV / block-level 管理；
2. idle KV block 识别；
3. swap-out + madvise 释放物理页；
4. resume 时 swap-in 恢复 KV；
5. prefetch + defer 降低恢复延迟。
```

可以理解为：

```text
请求活跃时：
  它的 KV 必须常驻，不能释放。

请求暂停/idle 时：
  它独占且当前不会被 active request 读取的 KV block 可以换出。

请求即将恢复时：
  提前或同步把它需要的 KV block 换回来。

请求恢复后：
  避免立即又把刚恢复的 KV block 换出去。
```

---

## 3. 为什么需要 block-level KV 管理

传统连续 KV cache 更适合按连续 token 范围访问。但对于多请求、多会话场景，一个请求的 KV 可能与其他请求交错存在。

如果只按连续区间处理，会遇到两个问题：

```text
1. 粒度太粗：
   可能为了保留少量 active KV，不得不保留大量 idle KV。

2. 安全性难判断：
   很难精确知道某一段物理 KV 是否仍被 active request 读取。
```

因此本项目采用 paged / block-level 思路：

```text
将 KV cache 按 block 管理；
每个 block 有状态；
通过 block table / row remap 判断逻辑 KV 到物理 block 的映射；
以 block 为单位做 resident / swapped / released 状态转换。
```

这样可以更精确地回答：

```text
这个 block 是否属于 idle request？
这个 block 是否在当前 read window 中？
这个 block 是否被 active request 需要？
这个 block 是否已经 safe remap？
这个 block 是否可以 swap-out？
```

---

## 4. idle KV block 的含义

idle KV block 指的是：

```text
当前没有 active request 需要读取，
主要归属于暂停或 idle session，
且可以安全从物理内存中释放的 KV block。
```

它不是“无用数据”。它仍然属于某个 request，将来 resume 时还需要恢复。

因此它和 tail/lazy reclaim 不同：

| 机制                | 处理对象                    | 是否需要恢复       |
| ----------------- | ----------------------- | ------------ |
| tail/lazy reclaim | 未使用或可重写的 tail 区域        | 不需要保留原内容     |
| idle KV swap      | idle request 已生成的 KV 内容 | resume 时需要恢复 |

简单说：

```text
tail/lazy 是丢掉暂时不用的空白区域；
idle swap 是把暂时不用但未来还要用的 KV 内容搬走。
```

---

## 5. swap-out 的核心流程

当系统发现某些 block 满足 idle 条件后，会尝试执行 swap-out。

核心流程如下：

```text
1. 识别当前 active seq；
2. 统计哪些 block 属于 active read window；
3. 排除 active-owned / active-visible block；
4. 找到 idle-owned 或 cold block；
5. 检查该 block 是否已经 nonidentity remap；
6. 检查是否 resident；
7. 检查是否受 prefetch protection / defer protection 保护；
8. 将该 block 内容保存到 swap store；
9. 将 block 标记为 swapped；
10. 对对应物理页调用 madvise；
11. 更新统计计数器。
```

关键安全原则：

```text
不能换出 active request 当前可见的 KV；
不能换出仍会被当前 attention read window 读取的 KV；
不能写入已经 swapped 的 block；
不能在 resume 过程中立刻把刚恢复的 block 再换出。
```

---

## 6. madvise 为什么能降低 RSS

swap-out 只是逻辑上把 KV 内容保存起来。如果物理页仍然 resident，进程 RSS 不一定下降。

因此需要配合：

```text
madvise(..., MADV_DONTNEED)
```

其作用是告诉操作系统：

```text
这段内存当前内容可以丢弃；
后续访问时再重新 fault-in。
```

在本项目中，完整路径是：

```text
KV block 内容保存到 swap store；
对应 KV buffer 页被 madvise；
操作系统回收物理页；
进程 current RSS 下降。
```

需要注意：

```text
madvise bytes 不等于实际 RSS drop。
```

原因包括：

```text
页对齐限制；
相邻 page 仍被使用；
OS 是否立即回收；
同一页中是否混有不可释放内容；
RSS 统计粒度和时机差异。
```

因此本项目同时关注：

```text
madvise bytes；
KV resident / nonresident；
process RSS drop。
```

最终更重要的是：

```text
进程 current RSS 是否真实下降。
```

---

## 7. resume swap-in

当 idle session 恢复生成时，它之前换出的 KV block 必须重新可用。

恢复路径是：

```text
1. session 进入 RESUME_PENDING；
2. 检查该 seq 需要的 KV block；
3. 对 swapped block 执行 swap-in；
4. 将内容从 swap store 恢复到 KV buffer；
5. 更新 block 状态为 resident；
6. 允许 attention 正常读取。
```

如果没有提前恢复，resume 时会一次性触发较多 swap-in，导致首 token latency 明显升高。

因此需要 prefetch。

---

## 8. 为什么需要 prefetch

idle swap 降低 RSS，但也引入恢复成本：

```text
swap-out 越多，resume 时需要恢复的 KV 越多；
恢复越晚，resume first-token latency 越高；
如果恢复发生在 active decode 路径中，也会拖慢整体 TPS。
```

prefetch 的目标是：

```text
在 request 真正恢复前，
提前把它可能需要的 KV block 恢复一部分，
减少 resume 当下的一次性阻塞。
```

本项目目前实现的是同步预取，不引入后台线程。

同步预取包括两类：

```text
1. during-active incremental prefetch；
2. final limited sync prefetch。
```

---

## 9. during-active incremental prefetch

during-active incremental prefetch 指的是：

```text
当其他 session 仍在 active decode 时，
对即将 resume 的 session 每隔一定 token 步恢复少量 block。
```

当前关键参数：

```text
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
```

含义：

```text
每隔 2 个 active decode token，
为 RESUME_PENDING session 最多恢复 2 个 block。
```

优点：

```text
将恢复成本分散到 request 真正 resume 之前；
减少 resume 当下的集中 swap-in。
```

缺点：

```text
仍然发生在主推理流程中；
会抢占 active decode 的时间；
可能降低 active TPS。
```

---

## 10. final limited sync prefetch

during-active prefetch 可能无法完全覆盖 resume 需要的 block。因此 Stage 10-E 进一步加入 final limited sync prefetch。

新增参数：

```text
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12
```

含义：

```text
在 session 真正 resume 前，
最多再同步恢复 12 个 block。
```

它不是 full prefetch，而是 limited prefetch。

设计原因：

```text
full prefetch 可能一次恢复过多 block，造成明显同步阻塞；
limited prefetch 控制恢复上限，在延迟和吞吐之间折中。
```

Stage 10-E 的结果显示：

```text
final limited sync prefetch 可以明显降低 resume first-token latency。
```

例如：

```text
Stage 10-D-C e2_b2_defer:
  resume delta ≈ +27.564 ms

Stage 10-E-G final12 no-core-trace:
  resume delta ≈ +20.198 ms
```

---

## 11. defer swapout on resume

resume 后的请求刚刚恢复了部分 KV block。如果这些 block 立刻又被 idle swap 逻辑换出，会造成反复 swap-in / swap-out，增加延迟并降低收益。

因此引入：

```text
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

核心作用：

```text
对刚 resume 的 session/block 做短暂保护；
避免刚换入就被再次换出。
```

该机制在 Stage 8 已证明可以降低 resume first-token latency。

简单说：

```text
刚搬回来的 KV，先别马上再搬走。
```

---

## 12. prefetch protection

prefetch protection 与 defer 类似，但更偏向保护正在预取或刚预取的 block。

目的：

```text
避免一个 block 刚被预取恢复，
下一步又被 idle swap-out 选中。
```

否则会出现：

```text
预取恢复 → idle 判断为可换出 → 又换出 → resume 时仍然缺页
```

这会抵消 prefetch 的收益。

---

## 13. 低日志执行模式

早期实现中：

```text
LLAMA_KV_PAGED_IDLE_TRACE
```

既控制 trace 打印，又控制 idle maintenance 执行，导致：

```text
关闭 IDLE_TRACE 后，idle swap 不执行；
RSS drop 从约 363 MiB 降到约 194 MiB。
```

Stage 10-E-F 已经解耦：

```text
LLAMA_KV_PAGED_IDLE_SWAP 控制功能；
LLAMA_KV_PAGED_IDLE_TRACE 控制打印。
```

最终低日志方案为：

```text
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1

不设置：
LLAMA_KV_PAGED_TRACE
LLAMA_KV_PAGED_IDLE_TRACE
```

这样可以做到：

```text
执行 idle swap / madvise；
不打印逐步 KV_PAGED_TRACE；
不打印逐步 KV_PAGED_IDLE_TRACE；
仍然获得完整 RSS 下降。
```

Stage 10-E-G 结果：

```text
RSS drop ≈ 363.184 MiB
KV_PAGED_TRACE lines = 0
KV_PAGED_IDLE_TRACE lines = 0
```

说明低日志完整 S5 已经跑通。

---

## 14. 最终同步方案配置

当前同步稳定方案推荐配置如下：

```bash
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1

LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1

LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

低日志下不设置：

```bash
LLAMA_KV_PAGED_TRACE
LLAMA_KV_PAGED_IDLE_TRACE
```

对应实验简称：

```text
S5_final12_no_core_trace
```

---

## 15. 当前最终结果

在 WikiText-2 corpus-backed semi-real multi-session workload 下，Stage 10-E-G 3-run median：

| case                       | RSS drop vs S0 | active TPS delta | resume delta |
| -------------------------- | -------------: | ---------------: | -----------: |
| `S0_baseline`              |      0.000 MiB |           0.000% |     0.000 ms |
| `S2_tail_lazy`             |    191.461 MiB |          -0.552% |    -0.431 ms |
| `S5_final12_no_core_trace` |    363.184 MiB |         -18.706% |   +20.198 ms |

通过性：

```text
exit_sum = 0
real_abnormal_sum = 0
all_finished_sum = 3
KV_PAGED_TRACE lines = 0
KV_PAGED_IDLE_TRACE lines = 0
```

结论：

```text
当前同步方案能够稳定降低 RSS，
但仍有明显吞吐代价。
```

---

## 16. 收益与代价

### 16.1 收益

当前方案的主要收益：

```text
进程 current RSS 下降约 363.184 MiB；
低日志模式下仍然成立；
真实文本语料驱动下成立；
多会话 pause/resume 场景下成立；
不引入异步线程，工程风险较低。
```

### 16.2 代价

当前方案的主要代价：

```text
active TPS 下降约 18.706%；
resume first-token 平均增加约 20.198 ms；
同步 prefetch 仍在主推理路径中执行；
swap-in / madvise / restore 机制本身有明显成本。
```

因此不能将当前方案表述为无损优化。

更准确的表述是：

```text
当前方案以可观但明确的性能代价换取稳定 RSS 下降。
```

---

## 17. 为什么后续考虑异步预取

Stage 10-E-G 已经排除了一个关键疑问：

```text
TPS 下降不是主要由 trace 打印造成的。
```

因为关闭：

```text
KV_PAGED_TRACE
KV_PAGED_IDLE_TRACE
```

后，TPS 仍下降约 18.706%。

因此性能瓶颈更可能来自：

```text
swap-out / madvise；
restore / swap-in；
prefetch step；
final sync prefetch；
page fault / memory bandwidth；
主推理路径中同步执行恢复工作。
```

异步预取的目标是：

```text
把部分恢复工作从主推理路径中移到后台；
在 session 真正 resume 之前提前恢复 KV block；
减少 active decode 被同步恢复打断；
降低 resume first-token latency。
```

但异步会引入：

```text
线程安全问题；
锁竞争；
KV block 状态一致性；
后台任务队列；
worker 生命周期管理；
context 销毁时退出顺序；
偶发竞态 bug。
```

因此异步应作为新分支实验，不应直接覆盖当前同步稳定版。

---

## 18. 推荐分支策略

当前同步方案建议作为稳定主线：

```text
kv-paged-stage4c-idle-telemetry
```

建议打 tag：

```text
stage10e-sync-stable
```

后续异步实验建议新开分支：

```text
kv-paged-stage11-async-prefetch
```

这样可以保证：

```text
同步稳定版本可随时提交；
异步实验失败也不影响主线；
后续可以对比 sync vs async。
```

---

## 19. 当前方案的边界

当前 workload 应称为：

```text
WikiText-2 corpus-backed semi-real multi-session workload
```

或：

```text
真实语料驱动的半真实多会话工作负载
```

因为：

```text
prompt 内容来自真实语料；
session 生命周期仍由 driver 构造；
不是完整真实 server workload；
不是 HTTP / llama-server / k6 压测；
不是 ShareGPT / LMSYS / WildChat 真实请求流。
```

当前结论的适用范围是：

```text
多会话 pause/resume 场景；
idle KV 较多；
可以接受一定性能代价换取 RSS 下降。
```

不应外推为：

```text
所有推理场景都有效；
单请求也有收益；
线上服务无损；
真实 server workload 已完全验证。
```

---

## 20. 对外汇报口径

推荐汇报口径：

```text
我们实现了面向多会话 idle KV cache 的运行时 swap-out / madvise / resume swap-in 机制，并结合 during-active incremental prefetch、resume defer 和 final limited sync prefetch 降低恢复延迟。在 WikiText-2 真实语料驱动的半真实多会话 workload 中，最终低日志同步方案在关闭逐步 KV trace 的情况下，实现约 363.2 MiB 进程 RSS 下降，3-run 均正常完成，无 real abnormal。代价是 active TPS 下降约 18.7%，resume first-token 平均增加约 20.2 ms。该结果证明 idle KV 运行时换出能够真实降低内存占用，但同步恢复路径仍有明显性能代价，后续将通过异步预取继续优化。
```

---

## 21. 后续工作

建议下一阶段：

```text
Stage 11-A：async prefetch feasibility audit
```

审计重点：

```text
llama_memory_prefetch_seq_step 是否能在后台线程调用；
KV cache 是否已有可复用锁；
swap-in / swap-out / write path 是否会并发冲突；
prefetch protection 是否需要原子化；
worker 如何启动和停止；
context 销毁时如何安全退出；
如何避免后台线程恢复已经结束的 seq。
```

然后再做：

```text
Stage 11-B：minimal async prefetch prototype
```

要求：

```text
默认关闭；
env 开启；
只在 semi-real workload 里验证；
先不追求通用 server 集成；
先证明 TPS / resume 是否优于同步方案。
```
