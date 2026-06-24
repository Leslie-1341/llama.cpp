# KV Paged Read — Stage 6C-1 Interleaved Prefetch Results

> 本文档记录 Stage 6C-1（interleaved / incremental single-thread prefetch）的实现、smoke 测试、调度修复与调优结果，以及对后续 Stage 6C-2 的建议。本文仅新增文档，不修改源码。

---

## 1. Goal

Stage 6B 已证明 synchronous prefetch 可行，但存在明显局限：

1. idle swap/madvise 能降低 RSS；
2. resume 前 synchronous prefetch 能降低 first-token latency；
3. 但 synchronous prefetch 会把已经释放的 KV blocks 提前拉回内存；
4. 因此 prefetch 后净 RSS 收益基本归零；
5. prefetch_ms 仍然是同步等待，只是从 first-token path 前移到了 resume 前。

Stage 6C-1 的目标是在不引入后台线程的前提下，先做 single-thread interleaved / incremental prefetch probe：

- 把 KV swap-in 成本拆成多次小恢复，插入 `seq1` active decode 阶段；
- 在 `seq0` resume 前尽量恢复完目标 blocks，降低 `seq0` resume first-token latency；
- 同时观测 `seq1` active latency / tokens_per_second 是否明显退化。

核心验证点：

1. 恢复成本能否从 resume first-token 路径移出；
2. 一次性 15~16 ms 的 synchronous prefetch 能否拆成多个约 3~4 ms 的小恢复；
3. active prefetch 后是否能稳定保留 resident progress；
4. 是否可把 RSS 回升窗口推迟到 active 后段。

---

## 2. Why Not Direct Async Thread Yet (Stage 6C-0 Audit)

Stage 6C-0 审计结论（摘要）：

1. 不建议直接做真正后台线程 async prefetch；
2. 原因是 `paged_block_states` 没有锁，`ggml_backend_tensor_set` 与 active decode 并发写同一 K/V tensor 的安全性不明确；
3. 更稳妥的路线是先做 single-thread interleaved prefetch；它不涉及线程安全问题，但可以验证 overlap / 分摊恢复成本的调度价值；
4. 真正 async thread 留到后续 Stage 6C-2 再评估。

---

## 3. Stage 6C-1 Implementation Overview

关键改动（只为文档记录，实际实现已在代码库中）：

1. 新增 public API：`llama_memory_prefetch_seq_step(mem, seq_id, max_blocks)`；
2. `max_blocks > 0`：最多恢复 `max_blocks` 个 SWAPPED blocks；
3. `max_blocks == 0`：probe-only，只扫描并更新 last-call stats，不恢复 block；
4. `prefetch_seq(seq_id)` 保留 full restore 语义，内部转发到 `prefetch_seq_step(seq_id, UINT32_MAX)`；
5. driver 新增 active-stage prefetch env：
   - `LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE`
   - `LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS`
   - `LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS`
   - `LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP`
6. active loop 插入点：在 `seq1` active decode loop 中，`decode_batch(...)` 成功之后调用 `prefetch_seq_step`；
7. 新增 telemetry：
   - `prefetch_during_active_enabled`
   - `prefetch_during_active_calls`
   - `prefetch_during_active_blocks`
   - `prefetch_during_active_ms_total`
   - `prefetch_during_active_ms_max`
   - `prefetch_remaining_blocks_before_resume`
   - `rss_before_active_prefetch_kb`
   - `rss_after_active_prefetch_kb`
   - `rss_before_resume_kb`

此实现为 single-thread、非并发的 interleaved prefetch，主要用于验证 schedule / protected-gate 的有效性。

---

## 4. First Smoke: Interleaved Prefetch Without Protection

首次 smoke 测试结果（未加保护 gate）：

```
prefetch_during_active_enabled=1
prefetch_during_active_calls=9
prefetch_during_active_blocks=9
prefetch_during_active_ms_total=30.512
prefetch_during_active_ms_max=3.444
prefetch_remaining_blocks_before_resume=4
rss_before_active_prefetch_kb=8375676
rss_after_active_prefetch_kb=8379604
seq0_resume_first_token_ms=111.121
```

解释：

1. active prefetch 已经触发，且单次 prefetch 延时被拆小（ms_max ≈ 3.444 ms）；
2. 但 `during_blocks=9` 大于理论目标 `≈5`，`remaining=4`，说明 resume 前仍有 4 个 blocks 是 SWAPPED；
3. RSS 只回升约 3.84 MiB（≈1 个 block），说明刚 prefetch 回来的 seq0 blocks 又被 idle swap-out 策略重新换出。

根因：idle swap-out 嵌在每次 `llama_decode` 的图输入构建回调 `set_input_paged_row_idx` 中，`seq1` active decode 每一步都会触发 idle-trace + swap-out 扫描；由于 `seq0` 仍被视为 idle，刚 prefetch 回来的 seq0-owned resident blocks 又被选作 victim。

---

## 5. Stage 6C-1A: Prefetch-Protected Gate (修复)

为解决上面问题，我们引入 per-seq prefetch-protected / resume-pending bitset：`paged_prefetch_protected_seq`，以及 public API：

```
llama_memory_set_seq_prefetch_protected(mem, seq_id, enabled)
```

语义：

- `enabled=true`：该 seq 拥有的 blocks 不再被 idle swap-out 选为 victim，但其 SWAPPED blocks 仍允许被 prefetch / swap-in；
- `enabled=false`：恢复默认 idle swap-out 策略。

driver 使用：

- 第一次 active prefetch 前：`set_seq_prefetch_protected(mem, 0, true)`；
- `seq0` resume 完成后：`set_seq_prefetch_protected(mem, 0, false)`。

新增 telemetry：`prefetch_protect_enabled`、`paged_idle_swap_skip_protected`。

修复后 smoke 结果示例：

```
prefetch_during_active_enabled=1
prefetch_protect_enabled=1
prefetch_during_active_calls=9
prefetch_during_active_blocks=5
prefetch_during_active_ms_total=16.602
prefetch_during_active_ms_max=3.755
prefetch_remaining_blocks_before_resume=0
rss_before_active_prefetch_kb=8375760
rss_after_active_prefetch_kb=8395032
seq0_resume_first_token_ms=96.786
paged_idle_swap_skip_protected>0
```

解释：

1. protected gate 生效，active-prefetched blocks 不再被立即重新 swap-out；
2. `during_blocks` 从 9 下降到 5，`remaining` 从 4 变为 0；
3. RSS 回升约 18.82 MiB，符合恢复 5 个 blocks 的预期；
4. first-token latency 降到 96.786 ms。

结论：protected gate 对 interleaved prefetch 的效果至关重要，它确保恢复进度在 resume 前能稳定保留。

---

## 6. Four-Mode 3-run Median (Stage 6C-1A)

测试配置：

```
warmup=64
-n 128
--ctx-size 1024
--parallel 2
--batch-size 128
--ubatch-size 128
--cache-type-k f32
--cache-type-v f32
--kv-unified
--temp 0
```

四种模式：`base_no_swap`、`swap_no_prefetch`、`swap_sync_prefetch`、`swap_interleaved_prefetch`。

关键派生指标：

- `idle_rss_drop_mib_base_vs_swap_no_prefetch = 18.832 MiB`
- `sync_prefetch_rss_rebound_mib = 18.820 MiB`
- `interleaved_active_prefetch_rss_rebound_mib = 18.820 MiB`
- `net_rss_drop_before_resume_mib_base_vs_interleaved = 0.012 MiB`

- `first_token_gain_ms_sync_vs_no_prefetch = 15.860 ms`
- `first_token_gain_ms_interleaved_vs_no_prefetch = 14.544 ms`
- `first_token_delta_ms_interleaved_minus_base = 19.917 ms`
- `first_token_delta_ms_interleaved_minus_sync = 1.316 ms`

- `seq1_active_overhead_ms_interleaved_minus_no_prefetch = 249.906 ms`
- `tps_delta_interleaved_minus_no_prefetch = -0.127`
- `tps_delta_interleaved_minus_base = 0.333`

- `sync_rebound_over_idle_drop_pct = 99.938%`
- `interleaved_rebound_over_idle_drop_pct = 99.938%`
- `interleaved_net_drop_over_idle_drop_pct = 0.062%`

解释：

1. `swap_no_prefetch` 使 RSS 降低约 18.832 MiB，但 `first-token` latency 增加到 113.115 ms（median）；
2. `sync_prefetch` 将 `first-token` latency 降到 97.255 ms，但 RSS 回升约 18.820 MiB；
3. `interleaved_prefetch` 将 `first-token` latency 降到 98.571 ms，仅比 sync 慢 1.316 ms；
4. interleaved 单次最大恢复耗时约 4.192 ms，明显低于 sync 的一次性 15.843 ms；
5. interleaved 在 active 阶段完成恢复并在 resume 前基本恢复所有被释放 blocks（`remaining=0`），因此它的作用是把恢复成本拆小并移出 first-token path，而不是保留净 RSS drop。

---

## 7. Stage 6C-1B: Prefetch Schedule Tuning

目的：在保持 `remaining=0` 和低 first-token latency 的前提下，尽量推迟 prefetch 开始时间，延长 RSS 低位窗口。

测试了若干 schedule（以 token 计）：

配置示例：

- A: AFTER=64, EVERY=8, STEP=1
- B: AFTER=88, EVERY=8, STEP=1
- C: AFTER=96, EVERY=8, STEP=1
- D: AFTER=112, EVERY=4, STEP=1

测试结论摘要：

1. 四组 schedule 全部满足 `remaining=0`、`blocks=5`、`ms_max < 5 ms`；
2. C 的 `first_ms` 最低（约 96.065 ms）；
3. D 的 prefetch 开始最晚（AFTER=112），`ms_max` 最低（≈3.345 ms），并且 `tps` 未见明显退化；
4. 因此 D（after=112/every=4/step=1）被推荐为折衷最优：尽量推迟 RSS 回升，同时保持低 first-token latency。

推荐：

```
LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS=112
LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=4
LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=1
```

---

## 8. Current Conclusions

必须诚实写明：

```
Stage 6C-1 证明了 protected interleaved prefetch 的可行性，但它仍不是最终能同时保留净 RSS 收益并完全消除 resume first-token latency 的方案。
```

要点：

1. partial prefetch API 能把 KV swap-in 拆成多次小恢复；
2. active-stage interleaved prefetch 能把恢复成本移出 `seq0` resume first-token path；
3. protected gate 是必要的，否则刚 prefetch 回来的 blocks 会被 idle swap-out 重新换出；
4. protected gate 修复后，median 下 `during_blocks=5`、`remaining=0`，说明恢复进度可稳定保留；
5. 与 `swap_no_prefetch` 相比，interleaved prefetch 将 `first-token` latency 从 113.115 ms 降到 98.571 ms，改善约 14.544 ms；
6. 与 sync prefetch 相比，interleaved first-token 仅慢约 1.316 ms；
7. interleaved 把一次性 15.843 ms 的恢复拆成多次单步最大约 4.192 ms 的恢复；
8. schedule tuning 可进一步推迟回升时机。

---

## 9. Limitations & Boundaries

必须诚实指出：

1. Stage 6C-1 仍然不是最终“同时降低净 RSS 与降低 first-token latency”的方案；
2. interleaved prefetch 在 `seq0` resume 前仍基本恢复了所有被释放 blocks；
3. active-stage RSS rebound 约 18.820 MiB，几乎等于 idle swap/madvise 释放量（≈18.832 MiB）；
4. 因此 resume 前净 RSS drop 仍接近 0；
5. 当前收益主要是延长低 RSS 窗口并降低用户可见的首字延迟，后续仍需要 scheduler-level / async overlap 来进一步扩大低 RSS 时间窗口。

---

## 10. Next Direction: Stage 6C-2 (Recommendation)

建议 Stage 6C-2 从 example-driven interleaved prefetch，过渡到 scheduler-level delayed prefetch policy：

候选方向：

1. scheduler-level resume prediction：仅在有较高 resume 预测置信度时开启 protected prefetch；
2. dynamic prefetch window：根据剩余 active tokens、remaining swapped blocks、blocks_per_step、单 block 恢复耗时动态决定何时启动；
3. memory-pressure-aware policy：根据系统内存压力调整 swap/prefetch 行为；
4. true async thread 暂不作为首选，直到并发写安全性（`paged_block_states`、`ggml_backend_tensor_set`）问题被彻底评估并解决。

推荐下一阶段命名：

```
Stage 6C-2: Scheduler-level delayed prefetch policy
```

---

## 11. Summary

Stage 6C-1 通过 protected interleaved prefetch 验证了“把恢复成本拆小并移出 first-token 路径”的思路：

- 它显著减少了单次阻塞时延（单步 <~4 ms），并把大块恢复工作分摊到 active decode；
- protected gate 是保证恢复进度不会被即时 swap-out 吞噬的关键；
- schedule tuning 能推迟 RSS 回升，延长低 RSS 窗口，但在本阶段仍未显著保留 net RSS drop；
- 下一步应把调度策略向 scheduler 层迁移，并在明确并发安全与内存压力量化后再考虑真正的后台 async prefetch。

---

文件新增：`docs/kv_paged_read_stage6c1_interleaved_prefetch_results.md`（本次提交仅新增文档，无源码变更）。
