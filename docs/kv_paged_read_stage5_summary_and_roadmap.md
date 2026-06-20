# KV Paged Read — Stage 5 Summary and Roadmap

> 本文档收束 Stage 5F–5G 的结果，用于阶段总结与进入下一阶段 prefetch / async swap-in 之前的路线说明。
> 关联文档：
> - [docs/kv_paged_read_stage5f_releasable_coverage_results.md](docs/kv_paged_read_stage5f_releasable_coverage_results.md)
> - [docs/kv_paged_read_stage5g_high_idle_coverage_plan.md](docs/kv_paged_read_stage5g_high_idle_coverage_plan.md)
> - [docs/kv_paged_read_stage5g_high_idle_coverage_results.md](docs/kv_paged_read_stage5g_high_idle_coverage_results.md)
> - [docs/kv_paged_read_stage5g_high_idle_perf_tradeoff_results.md](docs/kv_paged_read_stage5g_high_idle_perf_tradeoff_results.md)
> - 补充参考：[docs/kv_paged_read_stage5d_clean_perf_tradeoff_results.md](docs/kv_paged_read_stage5d_clean_perf_tradeoff_results.md)

---

## 1. 背景与目标

Stage 5 的目标不是继续改推理算法，而是把 idle KV 生命周期这条链路闭环：在 seq / request 进入 idle 后，释放其独占的 KV resident pages；在其恢复时，再把对应 KV blocks swap-in，保证输出 correctness。

本阶段解决的问题是：如何在不破坏推理语义的前提下，把 idle KV 从 resident memory 中真正释放出去，并量化这条路径的收益、边界和代价。

## 2. 本阶段机制概述

本阶段实现并验证的是 idle KV block swap-out + madvise + resume swap-in。

其目标是当某些 seq / request 进入 idle 状态时，将其独占的 KV blocks 从 resident memory 中释放；当该 seq / request 恢复时，再将对应 KV blocks swap-in，保证输出 correctness。

这一机制的核心判断顺序是：先识别 idle-owned KV coverage，再经过 read-window、resident-safe 和 nonidentity-remap gate，最后由 madvise 将可释放页真正交还给内核。

## 3. Correctness 结论

Stage 5B-2 / 5C / 5F / 5G 的多组实验都保持了语义一致性。base / nomadv / madv 多组实验中，seq1 active 输出和 seq0 resume 输出均保持一致，exit_code=0，real_abnormal=0。

结论很直接：当前 idle KV swap + madvise 没有破坏推理语义。

## 4. Memory release 结论

Stage 5F 的低 idle coverage workload 下，release ratio 约 11.7%。Stage 5G 的 high-idle-coverage workload 下，release ratio 可随 idle-owned coverage 提升到约 35.2%。

这说明约 12% 不是机制上限，而是原 workload 中 idle-owned KV coverage 低导致的结果。

## 5. Coverage attribution：为什么原始释放比例只有约 12%

Stage 5F 的 coverage matrix 给出的关键归因是：

- idle-owned coverage ≈ 12.5%
- release_ratio_total ≈ 11.7%
- read_window_block_pct = 0.00%
- safe_candidate_pct = 100.00%
- remap_gate_pct = 100.00%
- drop_over_idle_pct ≈ 93.75%

解释是：当前瓶颈不是 read-window / resident-safe / nonidentity-remap gate，也不是 madvise 后段失效，而是可释放 idle-owned KV 覆盖范围不足。进入 idle-owned 漏斗的 KV 中，大约 93.75% 最终表现为 KV resident drop，所以机制本身是有效的，限制主要来自 workload 结构。

## 6. High-idle coverage scaling：为什么 30%+ release ratio 成立

Stage 5G-1 说明，当 n / ctx 从 0.125 提高到 0.375 时，idle_coverage_pct 从约 12.5% 提高到约 37.5%，release_ratio_total_pct 从约 11.7% 提高到约 35.2%。

同时，数据近似满足：

release_ratio_total_pct ≈ idle_coverage_pct × 93.75%

这说明机制收益主要受 workload 中 idle KV 占比决定；在长上下文、多会话、存在 idle request 的场景中，收益会更明显。high-idle coverage 不是为了改变机制，而是为了验证释放上限会随 idle-owned coverage 线性放大。

## 7. Perf tradeoff：首字延迟代价

Stage 5G-2 的性能权衡结论是：

- mid_12p: madv RSS drop ≈ 28.39 MiB，resume first-token latency vs base +19.14 ms
- mid_31p: madv RSS drop ≈ 70.88 MiB，resume first-token latency vs base +42.86 ms
- mid_37p: madv RSS drop ≈ 86.05 MiB，resume first-token latency vs base +54.33 ms

同时，madv vs nomadv 的额外首字延迟为 +5.30 ms / +9.58 ms / +11.68 ms，说明 madvise 释放后的 page fault / resident recovery 额外代价小于整体 swap-out / swap-in 机制代价。

保守地说，当前 standalone driver 和 3-run median 下未观察到 total_wall_ms / tokens_per_second 退化，但不能解释为生产级吞吐提升。主要可重复观测的性能代价集中在 resume first-token latency 增加。

## 8. 当前局限

1. 当前实验基于 standalone driver，不是完整 server scheduler。
2. high-idle workload 是为了验证收益上限，不等同生产负载。
3. 当前 resume swap-in 是同步路径，首字延迟仍偏高。
4. total_wall_ms / tokens_per_second 结果不能直接推导生产吞吐。
5. 尚未实现 prefetch / async swap-in。

## 9. 下一阶段：prefetch / async swap-in

下一阶段命名为 Stage 6: Prefetch / Async Swap-in for Resume Latency Mitigation。

当前 swap / madvise 负责降低内存；下一阶段 prefetch / async swap-in 负责将恢复 IO 从用户首字路径中移出，降低恢复时的交互延迟。

建议路线：

1. Stage 6A: prefetch design plan
2. Stage 6B: synchronous prefetch probe before resume
3. Stage 6C: async prefetch overlapping with active request decode
4. Stage 6D: compare resume first-token latency with and without prefetch

建议评估指标：

- RSS drop 是否保持
- resume first-token latency 是否下降
- seq0 / seq1 correctness 是否保持
- prefetch 是否引入额外 RSS 回升
- 是否能与 active request decode 重叠

## 10. 阶段性结论

本阶段我们完成了 idle KV swap + madvise 的可行性验证。结果表明，在存在 idle request 的长上下文 / 多会话场景中，该机制可以真实释放 KV resident pages；低 idle coverage 下释放比例约 12%，而 high-idle coverage 下可提升到 30%+。性能侧，主要代价集中在 idle request 恢复时的首字延迟，且随释放规模增加。因此下一阶段将引入 prefetch / async swap-in，将恢复 IO 从首字路径中移出，降低交互延迟。

### 对外汇报口径

本阶段我们完成了 idle KV swap + madvise 的可行性验证。结果表明，在存在 idle request 的长上下文 / 多会话场景中，该机制可以真实释放 KV resident pages；低 idle coverage 下释放比例约 12%，而 high-idle coverage 下可提升到 30%+。性能侧，主要代价集中在 idle request 恢复时的首字延迟，且随释放规模增加。因此下一阶段将引入 prefetch / async swap-in，将恢复 IO 从首字路径中移出，降低交互延迟。