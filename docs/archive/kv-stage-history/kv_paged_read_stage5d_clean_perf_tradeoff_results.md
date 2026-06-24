# Stage 5D-1B：clean perf matrix tradeoff 结果

> 本文档记录 Stage 5D-1B 的实验结果与性能/内存 tradeoff 解释，仅为结果归档，不包含源码修改。
> 文件路径：`docs/kv_paged_read_stage5d_clean_perf_tradeoff_results.md`
> 关联文档：
> - `docs/kv_paged_read_stage5c_idle_swap_madvise_results.md`（5C madvise / current RSS 结果）
> - `docs/kv_paged_read_stage5c_scale_cumulative_rss_results.md`（5C-scale-B cumulative RSS 结果）

---

## 1. Stage 5D-1B 目标

在更干净的性能测试条件下，对比 base / nomadv / madv 的整体耗时、tokens/s 和 seq0 resume first-token latency，判断 idle swap + madvise 的性能代价。

注意：

- 本阶段不是新增优化；
- 不重新证明 peak RSS；
- 不声明生产级 server 性能；
- 只基于当前 standalone driver 做 clean perf observation。

---

## 2. 为什么需要 clean perf matrix

前面阶段已经闭环了 correctness 与 current RSS 收益：

- Stage 5B-2：idle KV block swap-out / resume swap-in correctness（逐字节一致）；
- Stage 5C：safe idle swapped block 可 madvise，current RSS 正向下降；
- Stage 5C-scale-B：补充 cumulative RSS telemetry，验证 current RSS drop 随 madvise 规模放大。

Stage 5C-scale-B 的核心数据：

```text
small ctx=512  n=64   madvise_bytes=19660800  rss_total_drop≈13.4 MiB
mid   ctx=1024 n=128  madvise_bytes=35389440  rss_total_drop≈28.4 MiB
large ctx=2048 n=256  madvise_bytes=66846720  rss_total_drop≈57.4 MiB
```

但仅有 RSS 收益不足以判断这条路径是否值得：还要量化性能代价，尤其是：

```text
total wall time / tokens per second
resume first-token latency
```

Stage 5D-1 初测受 verbose trace 与固定运行顺序干扰，噪声偏大。本阶段用更干净的测试条件重测，才能给出可信的 tradeoff 判断。

---

## 3. 实验设置

相比 Stage 5D-1 初测的改进：

```text
1. 关闭 LLAMA_KV_PAGED_TRACE，减少 verbose trace 对性能的干扰；
2. 保留必要的 idle/swap gate；
3. base / nomadv / madv 轮换运行顺序，避免固定顺序导致后跑更快；
4. 每组 RUNS=6；
5. 使用 median 作为主要统计口径，mean 作为辅助；
6. correctness 仍然对 full stdout / seq1 active / seq0 resume 做逐字节比对。
```

实验三档：

```text
small: ctx=512,  n=64
mid:   ctx=1024, n=128
large: ctx=2048, n=256
```

三组：

```text
base:   paged + nonidentity，不开 swap，不开 madvise
nomadv: idle swap-out + resume swap-in，不 madvise
madv:   idle swap-out + madvise + resume swap-in
```

---

## 4. correctness 结果

all correctness/abnormal fields are zero：

```text
base_exit=0
nomadv_exit=0
madv_exit=0

full_nomadv=0
full_madv=0

seq1_nomadv=0
seq1_madv=0

seq0_nomadv=0
seq0_madv=0

real_abnormal=0
```

结论：性能插桩和 clean perf matrix 不破坏 correctness；base / nomadv / madv 在 small / mid / large 下均保持逐字节一致。

---

## 5. memory result：cumulative current RSS drop

madv 组的 cumulative current RSS drop 随 workload 放大：

```text
small 约 13.41 MiB
mid   约 28.39 MiB
large 约 57.45 MiB
```

这与 Stage 5C-scale-B 的结果一致，说明 idle swap + madvise 的内存收益随可 madvise idle block 规模放大。

> 边界：这里只声明 observed cumulative current RSS drop 随规模放大。RSS drop 不等于 madvise bytes（内核回收时机、页对齐、邻居跳过及进程其它内存活动都会使观测 drop 与 advise 量不一致）；不声明 peak RSS 下降；不声明生产级 server 收益。

---

## 6. throughput / total wall time result

在当前 clean perf matrix 中，没有观察到 madv 带来的整体 total wall time / tokens/s 下降。

grouped means / medians（重点保留 median）：

```text
case    mode    total_wall_ms_mean  total_wall_ms_median  resume_first_ms_mean  resume_first_ms_median  resume_total_ms_mean  resume_total_ms_median  tokens_per_second_mean  tokens_per_second_median  madvise_bytes_mean  rss_total_drop_kb_mean  rss_drop_sum_kb_mean
small   base    13698.242  13678.121  101.915  103.293  6596.487  6681.495  9.689771   9.692301   0         0      0
small   nomadv  12947.603  12878.409  102.992  103.031  6093.905  6067.498  10.265854  10.322218  0         0      0
small   madv    12827.730  12588.160  109.124  107.018  6079.378  5997.033  10.379832  10.557614  19660800  13728  18860

mid     base    27423.258  27339.591  101.267  100.504  14116.192  14170.484  9.500940   9.528223   0         0      0
mid     nomadv  26460.773  26512.130  112.321  111.337  12757.246  12549.005  9.863119   9.835724   0         0      0
mid     madv    25068.850  24983.609  121.757  119.078  12321.494  12257.398  10.411274  10.440376  35389440  29070  34202

large   base    56612.880  56602.010  115.562  115.023  29926.207  29933.084  9.133591   9.121232   0         0      0
large   nomadv  56506.447  56648.058  144.725  140.319  30193.900  30435.644  9.157135   9.115118   0         0      0
large   madv    55316.255  55586.446  158.567  157.270  28882.383  28560.055  9.342857   9.288470   66846720  58832  65089
```

madv median total_wall_ms 相比 base：

```text
small -7.969%
mid   -8.617%
large -1.794%
```

解释（克制）：负数表示 madv 在当前 standalone driver 中没有变慢，median total_wall_ms 甚至略低。但这不能直接解释为生产级吞吐提升，可能与当前 driver 的 active read window 缩小、运行噪声、单机调度等因素有关。

> 不声明 madv 提升吞吐量；不声明 swap/madvise 一定没有性能代价。

---

## 7. resume first-token latency result

这是当前最稳定的性能代价信号，必须重点看。

median-based tradeoff：

```text
case    nomadv_total_slowdown_vs_base_%  madv_total_slowdown_vs_base_%  madv_total_slowdown_vs_nomadv_%  nomadv_resume_first_delta_ms  madv_resume_first_delta_ms  madv_resume_first_delta_vs_nomadv_ms  rss_total_drop_mib
small   -5.847  -7.969  -2.254  -0.262  3.725   3.987   13.41
mid     -3.027  -8.617  -5.765  10.833  18.575  7.742   28.39
large   0.081   -1.794  -1.874  25.296  42.247  16.952  57.45
```

madv 相比 base 的 seq0 resume first-token latency 增加：

```text
small +3.725 ms
mid   +18.575 ms
large +42.247 ms
```

madv 相比 nomadv：

```text
small +3.987 ms
mid   +7.742 ms
large +16.952 ms
```

解释：这符合机制预期。idle KV block 被 madvise 后，seq0 resume 首步需要 swap-in，swap-in 开销主要暴露在 resume first token；释放规模越大，首字延迟越明显。

---

## 8. tradeoff 总结

当前 driver 下，idle KV swap + madvise 的主要性能代价不是整体吞吐下降，而是 idle request resume 的首 token 延迟增加。

- 内存侧：cumulative current RSS drop 随 workload 放大（≈13.41 / 28.39 / 57.45 MiB）；
- 吞吐侧：median total_wall_ms / tokens/s 未观察到下降（madv vs base 为 -7.969% / -8.617% / -1.794%），但不声明吞吐提升；
- 延迟侧：madv 的 resume first-token latency 单调增加（vs base +3.725 / +18.575 / +42.247 ms），释放规模越大越明显。

---

## 9. 边界说明

```text
1. 本阶段只验证 current RSS，不验证 peak RSS；
2. 本阶段是 standalone driver，不是 production server workload；
3. total_wall_ms / tokens_per_second 未观察到下降，但不声明生产级吞吐提升；
4. resume first-token latency 是当前更稳定的性能代价信号；
5. 后续还需要多 idle seq、多轮 resume、server-like workload、prefetch / async swap-in 验证。
```

---

## 10. 后续方向

- 多 idle seq / 多轮 resume 压力场景下的吞吐与 resume 延迟；
- server-like workload 下的调度、累计 RSS 与首 token 延迟行为；
- prefetch / async swap-in，降低 resume swap-in 暴露在首 token 上的延迟；
- 探索 peak RSS（而不仅 current RSS）的优化路径；
- 在更大 ctx / block_size 下复测 tradeoff 曲线，观察延迟代价随规模的增长形态。

---

## 11. 结论

**Stage 5D-1B 表明，在当前 standalone driver 中，idle KV swap + madvise 可以降低 cumulative current RSS，未观察到整体吞吐下降；主要可测性能代价体现在 resume first-token latency 增加。**
