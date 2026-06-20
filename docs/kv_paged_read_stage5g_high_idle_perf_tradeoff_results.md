# KV Paged Read — Stage 5G-2: High-idle-coverage Perf Tradeoff 结果

## 0. 目的
补齐 Stage 5G-1 在 high-idle-coverage 条件下的性能权衡评估，特别关注 resume first-token latency（seq0_resume_first_token_ms）的代价，并记录 medians 与 tradeoff 汇总。

## 1. 背景
Stage 5F-1 已定位：当前约 12% release ratio 主要受限于 idle-owned KV coverage，而不是 read-window / resident-safe / nonidentity-remap gate。

Stage 5G-1 已证明：提高 `n/ctx` 可以提高 idle-owned KV coverage，并使 release ratio 同步提高。small/mid 中，当 `n/ctx` 从 0.125 提高到 0.375 时，release ratio 从约 11.7% 提高到约 35.2%。该阶段只验证 memory release scaling，未评价 resume first-token latency。参考：

`docs/kv_paged_read_stage5g_high_idle_coverage_results.md`

Stage 5G-2 目标：补齐 high-idle-coverage 下的 perf tradeoff，尤其是 resume first-token latency。

## 2. 实验设置
使用现有 driver timing marker: `KV_IDLE_SWAP_RESUME_PERF`。

关键 timing 字段：

- `total_wall_ms`
- `seq1_active_ms`
- `seq0_resume_first_token_ms`（resume first-token latency）
- `seq0_resume_total_ms`
- `seq1_active_tokens`
- `seq0_resume_tokens`
- `total_measured_tokens`
- `tokens_per_second`
- `seq0_prefill_ms`
- `seq1_prefill_ms`

仅跑 mid 三档：

- `mid_12p`: ctx=1024, n=128, n/ctx=0.1250
- `mid_31p`: ctx=1024, n=320, n/ctx=0.3125
- `mid_37p`: ctx=1024, n=384, n/ctx=0.3750

每档三种模式：

- `base`: paged + nonidentity + idle_trace，不开 swap/madvise
- `nomadv`: idle swap-out / swap-in，但不开 madvise
- `madv`: idle swap-out + madvise + resume swap-in

每个 case/mode 跑 3 次，取 median。

注意：本轮 perf matrix 不开 `LLAMA_KV_PAGED_MINCORE=1`，以避免 mincore 统计影响 timing。

## 3. correctness 结果
所有 case/mode 均满足：

- `exit_code_max=0`
- `seq1_equal_max=0`
- `seq0_equal_max=0`
- `real_abnormal_max=0`

结论：base / nomadv / madv 输出保持一致；timing 实验未破坏 correctness。

## 4. medians_by_mode 表
以下为各 case/mode 的 median 结果（数据按原表保留）：

| case | mode | ctx | n | n_over_ctx | total_wall_ms | seq1_active_ms | seq0_resume_first_token_ms | seq0_resume_total_ms | tokens_per_second | seq0_prefill_ms | seq1_prefill_ms | paged_swap_rss_total_drop_kb | paged_swap_madvise_bytes | exit_code_max | seq1_equal_max | seq0_equal_max | real_abnormal_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid_12p | base | 1024 | 128 | 0.1250 | 24568.972 | 11807.26 | 90.188 | 12243.198 | 10.607814 | 434.948 | 171.088 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| mid_12p | madv | 1024 | 128 | 0.1250 | 23013.96 | 11325.823 | 109.329 | 11252.928 | 11.338094 | 434.235 | 173.148 | 29072.0 | 35389440.0 | 0 | 0 | 0 | 0 |
| mid_12p | nomadv | 1024 | 128 | 0.1250 | 24447.765 | 11395.865 | 104.033 | 12229.103 | 10.663909 | 428.557 | 172.235 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| mid_31p | base | 1024 | 320 | 0.3125 | 65158.665 | 29337.081 | 103.244 | 35367.944 | 9.88768 | 427.977 | 171.217 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| mid_31p | madv | 1024 | 320 | 0.3125 | 61583.492 | 29052.584 | 146.102 | 32263.817 | 10.465596 | 429.83 | 170.789 | 72584.0 | 82575360.0 | 0 | 0 | 0 | 0 |
| mid_31p | nomadv | 1024 | 320 | 0.3125 | 64455.064 | 29235.233 | 136.523 | 34791.197 | 9.995872 | 427.701 | 169.783 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| mid_37p | base | 1024 | 384 | 0.3750 | 79937.498 | 35974.477 | 102.094 | 43469.065 | 9.660096 | 434.005 | 170.427 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |
| mid_37p | madv | 1024 | 384 | 0.3750 | 75169.017 | 35273.133 | 156.425 | 39487.884 | 10.275802 | 431.306 | 169.807 | 88116.0 | 98304000.0 | 0 | 0 | 0 | 0 |
| mid_37p | nomadv | 1024 | 384 | 0.3750 | 79074.302 | 35624.316 | 144.745 | 43064.488 | 9.765866 | 430.977 | 168.733 | 0.0 | 0.0 | 0 | 0 | 0 | 0 |

## 5. tradeoff_summary 表
以下为 tradeoff 汇总（数据按原表保留）：

| case | base_resume_first_ms | nomadv_resume_first_ms | madv_resume_first_ms | madv_resume_first_delta_vs_base_ms | madv_resume_first_delta_vs_nomadv_ms | base_total_wall_ms | nomadv_total_wall_ms | madv_total_wall_ms | madv_total_slowdown_vs_base_pct | base_tokens_per_second | nomadv_tokens_per_second | madv_tokens_per_second | madv_tokens_per_second_delta_vs_base_pct | base_rss_drop_mib | nomadv_rss_drop_mib | madv_rss_drop_mib | base_madvise_mib | nomadv_madvise_mib | madv_madvise_mib |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mid_12p | 90.188 | 104.033 | 109.329 | 19.14099999999999 | 5.295999999999992 | 24568.972 | 24447.765 | 23013.96 | -6.329169979110249 | 10.607814 | 10.663909 | 11.338094 | 6.884359020623854 | 0.0 | 0.0 | 28.390625 | 0.0 | 0.0 | 33.75 |
| mid_31p | 103.244 | 136.523 | 146.102 | 42.858000000000004 | 9.579000000000008 | 65158.665 | 64455.064 | 61583.492 | -5.486872697591338 | 9.88768 | 9.995872 | 10.465596 | 5.844808893491704 | 0.0 | 0.0 | 70.8828125 | 0.0 | 0.0 | 78.75 |
| mid_37p | 102.094 | 144.745 | 156.425 | 54.33100000000002 | 11.680000000000007 | 79937.498 | 79074.302 | 75169.017 | -5.965261759881447 | 9.660096 | 9.765866 | 10.275802 | 6.373704774776567 | 0.0 | 0.0 | 86.05078125 | 0.0 | 0.0 | 93.75 |

## 6. 结果解释

### 6.1 内存收益随 idle coverage 提高

- mid_12p: madv RSS drop = 28.39 MiB
- mid_31p: madv RSS drop = 70.88 MiB
- mid_37p: madv RSS drop = 86.05 MiB

解释：high-idle-coverage 下，madvise 可释放的 current RSS 随 idle KV 覆盖范围增大而增加。即释放规模越大，madvise 可收回的内存越多。

### 6.2 resume first-token latency 是主要性能代价

- mid_12p: madv vs base = +19.14 ms
- mid_31p: madv vs base = +42.86 ms
- mid_37p: madv vs base = +54.33 ms

结论：释放规模越大，resume first-token latency 增量越大，resume 首字延迟是主要的交互性能成本。

### 6.3 madvise 的额外代价小于整体 swap 机制代价

- mid_12p: madv vs nomadv = +5.30 ms
- mid_31p: madv vs nomadv = +9.58 ms
- mid_37p: madv vs nomadv = +11.68 ms

解释：base -> nomadv 的增加主要来自 swap-out / swap-in 机制本身；nomadv -> madv 的增加才是 madvise 释放后 page fault / resident recovery 的额外代价。

### 6.4 total_wall_ms / tokens/s 没有观察到退化，但不能解释为性能提升

- mid_12p: madv_total_slowdown_vs_base_pct = -6.33%
- mid_31p: madv_total_slowdown_vs_base_pct = -5.49%
- mid_37p: madv_total_slowdown_vs_base_pct = -5.97%

以及：madv tokens/s 相比 base 为 +5.84% 到 +6.88%。

保守解释：在当前 standalone driver 和 3-run median 口径下，未观察到 total_wall_ms 或 tokens_per_second 退化；该现象不解释为生产级吞吐提升。主要可重复观测的性能代价集中在 `seq0_resume_first_token_ms` 增加。

## 7. 总结结论

Stage 5G-2 表明，high-idle-coverage 下 30%+ KV release ratio 的主要性能代价是 resume first-token latency 增加。mid_37p 中，madv 可释放约 86.05 MiB current RSS，resume first-token latency 相比 base 增加约 54.33 ms，相比 nomadv 增加约 11.68 ms。当前 standalone driver 下未观察到 total_wall_ms / tokens_per_second 退化，但这一点不解释为生产级吞吐提升。

## 8. 后续方向

1. 若要降低交互代价，应研究 prefetch / async swap-in；
2. 后续可增加 6-run median 或 large 档验证；
3. 当前结果已经足以支撑“内存收益随 idle coverage 增加，但 resume 首字延迟随释放规模增加”的 tradeoff 结论。

## 9. 结论

Stage 5G-2 证明，high-idle-coverage workload 下 idle KV swap + madvise 可获得 30%+ KV resident/current RSS 释放收益，但主要性能代价集中在 resume first-token latency，且该代价随释放规模增大而增加。

---

文件生成于 Stage 5G-2 结果整理（仅新增文档，未修改源码）。
