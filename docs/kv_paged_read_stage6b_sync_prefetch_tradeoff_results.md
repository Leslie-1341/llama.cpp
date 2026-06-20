# KV Paged Read — Stage 6B Synchronous Prefetch Tradeoff Results

> 本文档记录 Stage 6B 的同步 prefetch 结果与 tradeoff 分析。本文只新增文档，不修改源码，不 build。

---

## 1. Goal

Stage 5 已完成 idle KV swap-out + madvise + resume swap-in，证明 idle KV blocks 可以真实释放 RSS。

Stage 6B 在此基础上引入 synchronous prefetch：在 seq0 resume decode 之前，显式恢复即将访问的 SWAPPED KV blocks，目标是降低 resume first-token latency。

本阶段还引入 long-idle workload：LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=64。它让 seq0 在进入 idle 前先 decode 64 个 warmup tokens，从而扩大 seq0 的 idle-owned KV footprint，避免 prefetch 只恢复 1 个 block，导致样本太小、结论不稳定。

本阶段的核心问题不是“prefetch 是否能让延迟更低”，而是“prefetch 是否以回收 RSS 为代价换来首字延迟改善，以及这个 tradeoff 是否符合预期”。

---

## 2. Implementation Summary

Stage 6B 的关键工程改动如下：

1. 新增 public prefetch hook：llama_memory_prefetch_seq(mem, seq_id)
2. prefetch_seq(seq_id) 只恢复 SWAPPED blocks
3. RELEASED blocks 仍然 skip，不会被重新置回 RESIDENT
4. 修复 prefetch_seq 中 logical block / physical block 映射问题：
   - 错误：cell / paged_block_size
   - 正确：paged_resolve(cell) / paged_block_size
5. 新增 prefetch coverage debug 字段：
   - prefetch_owned_blocks
   - prefetch_swapped_blocks
   - prefetch_resident_blocks
   - prefetch_released_blocks
   - prefetch_invalid_cells
   - prefetch_failures
6. 新增 long-idle workload env：LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS
7. 修正 RSS checkpoint telemetry：
   - rss_before_prefetch_kb 在所有模式下都记录
   - no-prefetch 下 rss_after_prefetch_kb = rss_before_prefetch_kb

这组改动的共同目标是让 prefetch 的触发对象、覆盖范围、内存回升、以及 latency 收益都可以被单独观察，而不是混在 resume first-token 路径里无法拆分。

---

## 3. Why Long-Idle Warmup Was Needed

如果 seq0 在进入 idle 前只积累很少的 idle-owned KV blocks，那么 synchronous prefetch 只会恢复极少量数据，RSS 回升和 first-token latency 改善都可能过于微弱，难以稳定体现 tradeoff。

LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=64 的作用，就是人为扩大 seq0 的 idle-owned KV footprint，让 seq0 在 idle 后拥有足够多的 SWAPPED blocks 可供 prefetch。这样可以验证两个关键问题：

1. prefetch 是否真的只恢复该 seq 所拥有的 SWAPPED blocks；
2. prefetch 的 RSS 回升量是否与 idle swap/madvise 的释放量大体对称。

换句话说，warmup 不是为了制造更好的结果，而是为了让同步 prefetch 的 tradeoff 暴露得足够清楚。

---

## 4. Experiment Setup

实验采用 warmup=64、3-run median 的配置：

LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=64
-n 128
--ctx-size 1024
--parallel 2
--batch-size 128
--ubatch-size 128
--cache-type-k f32
--cache-type-v f32
--kv-unified
--temp 0

三种模式如下：

1. base_no_swap：paged read enabled，no idle swap，no prefetch
2. swap_no_prefetch：idle swap + madvise enabled，no prefetch
3. swap_prefetch：idle swap + madvise enabled，synchronous prefetch enabled

这里的比较顺序很重要：base_no_swap 用来定义无释放时的基线，swap_no_prefetch 用来观察“仅释放”对 RSS 和 latency 的影响，swap_prefetch 用来观察“释放后再同步拉回”的 tradeoff。

---

## 5. Raw Median Results

以下是 3-run median 原始结果。

| mode | first_ms | total_wall_ms | tps | prefetch_ms | prefetch_blocks | owned | swapped | resident | rss_before_kb | rss_after_kb | rss_resume_kb |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_no_swap | 79.269 | 28356.179 | 11.265870 | 0.000 | 0 | 0 | 0 | 0 | 8395012 | 8395012 | 8397524 |
| swap_no_prefetch | 115.846 | 27606.876 | 11.667935 | 0.000 | 0 | 0 | 0 | 0 | 8375872 | 8375872 | 8366924 |
| swap_prefetch | 99.613 | 27444.146 | 11.787884 | 15.932 | 5 | 6 | 5 | 1 | 8375840 | 8395112 | 8366824 |

说明：

1. no-prefetch 模式下 prefetch coverage 字段为 0 是正常的，因为没有调用 prefetch helper。
2. swap_prefetch 中 prefetch_blocks=5，说明同步 prefetch 实际恢复了 5 个 SWAPPED blocks。
3. owned=6、swapped=5、resident=1 的组合表明 prefetch 覆盖到了该 seq 所拥有的目标块，其中大部分原本处于 SWAPPED 状态，少量已经是 RESIDENT。

---

## 6. Memory / Latency Derived Metrics

下面是按 median 原始结果推导的关键指标。

| metric | value |
|---|---:|
| idle_rss_drop_mib_base_vs_swap_no_prefetch | 18.691 MiB |
| prefetch_rss_rebound_mib | 18.820 MiB |
| net_rss_drop_after_prefetch_mib_base_vs_prefetch_after | -0.098 MiB |
| first_token_gain_ms_swap_no_prefetch_minus_prefetch | 16.233 ms |
| prefetch_rebound_over_idle_drop_pct | 100.690% |
| net_drop_after_prefetch_over_idle_drop_pct | -0.522% |

解释口径如下：

1. idle swap/madvise 在 resume 前降低 RSS 约 18.69 MiB。
2. synchronous prefetch 提前恢复 5 个 SWAPPED KV blocks，使 RSS 回升约 18.82 MiB。
3. prefetch 回升量基本等于 swap/madvise 释放量，因此 prefetch 完成后的净 RSS 收益约为 0。
4. synchronous prefetch 将 resume first-token latency 从 115.846 ms 降至 99.613 ms，降低约 16.23 ms。
5. 但 prefetch 后 first-token latency 仍高于 base_no_swap 的 79.269 ms。
6. prefetch_ms median 为 15.932 ms，与 first-token latency gain 接近，说明 synchronous prefetch 主要是把 swap-in 成本从 first-token path 前移到 prefetch phase。

---

## 7. Interpretation

Stage 6B 证明了 synchronous prefetch 机制是可行的：它确实能在 resume 之前恢复目标 SWAPPED blocks，也确实能降低 resume first-token latency。

但更重要的是，它把 tradeoff 量化得很清楚：

1. swap_no_prefetch 能降低 RSS，但会显著增加 resume first-token latency。
2. swap_prefetch 能降低部分 resume first-token latency，但会把已经释放的 KV blocks 提前拉回内存，基本抵消本轮 RSS 收益。
3. prefetch_ms 与 latency gain 接近，说明成本主要只是从首字路径前移到了 prefetch phase，而不是被真正消除。

因此，这个阶段的结论不是“prefetch 让系统同时变快又更省内存”，而是“prefetch 证明了恢复路径可以前移，但如果它仍然是同步执行，就会吃回大部分内存收益”。

---

## 8. Limitation of Synchronous Prefetch

同步 prefetch 的限制非常明确：它本质上仍然是把 swap-in 工作放在用户可见路径附近，只是从 seq0 resume first-token 的同步构图段，挪到了 resume 前的显式 prefetch 阶段。

这会带来三个直接后果：

1. RSS 回升几乎与释放量对称，内存收益窗口很短。
2. 如果 prefetch 过早执行，RSS drop 会被提前吃回，内存优化目标被削弱。
3. 即使 first-token latency 得到改善，它仍然没有接近 base_no_swap，说明同步 prefetch 只是缓解，不是最终优化形态。

所以，Stage 6B 的同步 prefetch 更像是机制验证，而不是最终方案。它证明“提前恢复”是可行的，但没有解决“恢复成本如何被隐藏”这个更关键的问题。

---

## 9. Implication for Stage 6C

Stage 6C 应该从 synchronous prefetch 走向 async / overlapped prefetch。

目标是：在保留 idle-stage RSS drop 的同时，把 swap-in latency 隐藏到 active decode 或 scheduler waiting time 之中。

理想行为应该是：

1. 在 idle 期间，swapped KV blocks 保持 non-resident，从而保住 RSS drop。
2. 在 seq resume 之前，async prefetch 提前启动。
3. prefetch 成本与 active request decode 或等待时间重叠。
4. 用户可见的 resume first-token latency 接近 base_no_swap。
5. 吞吐不应出现明显退化。

也就是说，Stage 6B 负责证明“要恢复什么、恢复多少、同步恢复会付出什么代价”，Stage 6C 则要解决“怎样把这个代价藏起来”。

---

## 10. Summary

Stage 6B 证明了 synchronous prefetch 可以工作，并且可以显著降低 resume first-token latency，但它同时也证明了一个更重要的事实：同步 prefetch 会把被 idle swap/madvise 释放出去的 KV blocks 提前拉回内存，几乎抵消本轮 RSS 收益。

因此，在无内存压力、单个 resume checkpoint 下，base_no_swap 仍然是最优方案。

本阶段的正确结论应当是：

1. swap_no_prefetch 能降低 RSS，但显著增加 resume first-token latency。
2. swap_prefetch 能降低部分 resume first-token latency，但会基本回收掉 RSS 释放收益。
3. synchronous prefetch 不是最终优化形态，只是 Stage 6C async / overlap prefetch 的机制基础。

---

文件生成于 Stage 6B 结果整理，本轮仅新增文档，未修改源码。