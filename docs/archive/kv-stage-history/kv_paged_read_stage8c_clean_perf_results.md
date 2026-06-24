# Stage 8C: Clean Performance Matrix after Shadow Validation Gate

## 1. Background

Stage 8A 验证了 idle KV swap / prefetch / pressure policy 的基本机制：

1. `swap-only` 可以将 idle-owned KV blocks 换出，并通过 `madvise(DONTNEED)` 降低物理驻留内存。
2. 在 `NUM_IDLE_SEQS=2` 的 controlled benchmark 中，34 个 idle-owned blocks 可带来约 `127.5 MiB` KV nonresident，并对应约 `123–124 MiB` process RSS drop。
3. `swap-only` 会在 seq0 resume 时触发 fallback restore，导致 resume first-token latency 明显升高。
4. interleaved prefetch 可以提前恢复 resume 所需 blocks，将 `fallback_blocks` 降为 0，从而降低 resume first-token latency。

但 Stage 8A fair performance matrix 发现，full prefetch 虽然能消除 swap fallback 延迟，但 `paged-on + swap-off` 自身仍比 paged-off baseline 慢约 29 ms。

随后 Stage 8B-1 / 8B-2 定位并解决了 paged path 的主要基础开销来源：

```text
paged_shadow_validate() 是之前 paged performance path 的主要基础开销来源。
```

Stage 8B-2 新增：

```text
LLAMA_KV_PAGED_SHADOW_VALIDATE
```

语义：

```text
默认值：1，保持原行为；
显式设置 LLAMA_KV_PAGED_SHADOW_VALIDATE=0 时，跳过 paged_shadow_validate()。
```

Stage 8B-2 3-run median 结果表明：

| case                               | 含义                            | resume_first_ms | active_tps |
| ---------------------------------- | ----------------------------- | --------------: | ---------: |
| S0_paged_off                       | paged off baseline            |          80.024 |     12.284 |
| S1_paged_bookkeeping_shadow_on     | paged on + shadow on          |         108.477 |      9.150 |
| S2_paged_bookkeeping_shadow_off    | paged on + shadow off         |          80.095 |     12.328 |
| S4_paged_ingraph_gather_shadow_off | ingraph + gather + shadow off |          80.765 |     12.194 |
| S5_prefetch_full_shadow_off        | full prefetch + shadow off    |          99.359 |     12.173 |

关键结论：

```text
关闭 paged_shadow_validate() 后，paged bookkeeping 基本回到 paged-off baseline。
```

因此 Stage 8C 在 clean performance path 下重新评估 swap / prefetch 策略，明确在没有 debug validation 污染时的性能-内存权衡。

---

## 2. Goal

Stage 8C 的目标是重新回答：

```text
在 LLAMA_KV_PAGED_SHADOW_VALIDATE=0 的 clean performance path 下，
swap-only 到底带来多少 resume penalty？
prefetch 能恢复多少 latency？
active_tps 与 baseline 相比损失多少？
```

本阶段重点不是证明页真实 nonresident，而是评估 clean performance mode 下的延迟和吞吐。

页真实换出 / nonresident 已在 Stage 7D-B / Stage 8A memory validation mode 中通过 `mincore` 证明。Stage 8C 中 `LLAMA_KV_PAGED_MINCORE=0`，因此 `kv_nonresident_mib` 不作为本阶段的内存真实性指标。

---

## 3. Experiment Setup

### 3.1 Benchmark

Benchmark binary:

```text
build/bin/llama-kv-idle-swap-resume
```

Model:

```text
/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

Common arguments:

```text
--n-predict 128
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

Workload environment:

```text
LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
```

Performance mode environment:

```text
LLAMA_KV_PAGED_MINCORE=0
LLAMA_KV_PAGED_TRACE=0
LLAMA_KV_PAGED_REFAULT_TRACE=0
LLAMA_KV_CACHE_DEBUG=0
LLAMA_KV_PAGED_SHADOW_VALIDATE=0
```

### 3.2 Case Matrix

| case                            | 含义                                                      |
| ------------------------------- | ------------------------------------------------------- |
| C0_paged_off                    | paged off baseline                                      |
| C1_paged_on_swap_off_shadow_off | paged on, swap off, shadow validation off               |
| C2_swap_only_shadow_off         | swap-only, shadow validation off                        |
| C3_prefetch_g2_shadow_off       | swap + full prefetch, every 2 tokens, 1 block per step  |
| C4_prefetch_g4b4_shadow_off     | swap + full prefetch, every 4 tokens, 4 blocks per step |

### 3.3 Output

Logs:

```text
/root/oscomp/kv_logs/stage8c_clean_perf_3run
```

Summary files:

```text
/root/oscomp/kv_logs/stage8c_clean_perf_3run/summary_all.tsv
/root/oscomp/kv_logs/stage8c_clean_perf_3run/summary_median.tsv
```

Scripts:

```text
scripts/kv_stage8c_clean_perf_3run.sh
scripts/parse_stage8c_clean_perf_3run.py
```

---

## 4. 3-run Median Result

### 4.1 Compact Median View

| case                            | policy                       | resume_first_ms | active_tps | seq1_active_ms | total_wall_ms | process_rss_drop_mib | madvise_mib | prefetch_blocks | fallback_blocks | prefetch_remaining_blocks_before_resume | seq1/resume decoded | real_abnormal |
| ------------------------------- | ---------------------------- | --------------: | ---------: | -------------: | ------------: | -------------------: | ----------: | --------------: | --------------: | --------------------------------------: | ------------------- | ------------: |
| C0_paged_off                    | paged_off                    |          79.785 |     12.385 |      10335.144 |     59660.135 |                0.000 |       0.000 |               0 |               0 |                                       0 | 128/128             |             0 |
| C1_paged_on_swap_off_shadow_off | paged_on_swap_off_shadow_off |          82.361 |     11.999 |      10667.202 |     61094.507 |                0.000 |       0.000 |               0 |               0 |                                       0 | 128/128             |             0 |
| C2_swap_only_shadow_off         | swap_only_shadow_off         |         139.877 |     12.071 |      10604.119 |     60916.718 |              124.203 |     153.750 |               0 |              17 |                                      17 | 128/128             |             0 |
| C3_prefetch_g2_shadow_off       | prefetch_g2_shadow_off       |         101.371 |     12.051 |      10621.085 |     60786.458 |              124.023 |     153.750 |              17 |               0 |                                       0 | 128/128             |             0 |
| C4_prefetch_g4b4_shadow_off     | prefetch_g4b4_shadow_off     |          99.116 |     12.055 |      10617.618 |     60548.269 |              124.062 |     153.750 |              17 |               0 |                                       0 | 128/128             |             0 |

### 4.2 Compact Delta View

| delta   | resume_first_ms | active_tps | seq1_active_ms | total_wall_ms | process_rss_drop_mib | madvise_mib | prefetch_blocks | fallback_blocks | prefetch_remaining_blocks_before_resume |
| ------- | --------------: | ---------: | -------------: | ------------: | -------------------: | ----------: | --------------: | --------------: | --------------------------------------: |
| C1 - C0 |          +2.576 |     -0.386 |       +332.058 |     +1434.372 |                0.000 |       0.000 |               0 |               0 |                                       0 |
| C2 - C1 |         +57.516 |     +0.071 |        -63.083 |      -177.789 |             +124.203 |    +153.750 |               0 |             +17 |                                     +17 |
| C3 - C2 |         -38.506 |     -0.019 |        +16.966 |      -130.260 |               -0.180 |       0.000 |             +17 |             -17 |                                     -17 |
| C4 - C2 |         -40.761 |     -0.015 |        +13.499 |      -368.449 |               -0.141 |       0.000 |             +17 |             -17 |                                     -17 |
| C3 - C1 |         +19.010 |     +0.052 |        -46.117 |      -308.049 |             +124.023 |    +153.750 |             +17 |               0 |                                       0 |
| C4 - C1 |         +16.755 |     +0.056 |        -49.584 |      -546.238 |             +124.062 |    +153.750 |             +17 |               0 |                                       0 |

---

## 5. Interpretation

### 5.1 Shadow-off 后 paged 基础路径接近 baseline

Baseline:

```text
C0_paged_off:
  resume_first_ms = 79.785 ms
  active_tps = 12.385
```

Paged-on, swap-off, shadow-off:

```text
C1_paged_on_swap_off_shadow_off:
  resume_first_ms = 82.361 ms
  active_tps = 11.999
```

差值：

```text
C1 - C0:
  resume_first_ms +2.576 ms
  active_tps -0.386
```

解释：

```text
在 LLAMA_KV_PAGED_SHADOW_VALIDATE=0 后，
paged framework 的基础开销已经明显降低。
```

这与 Stage 8B-2 的结论一致：此前 paged path 的主要基础开销来自 `paged_shadow_validate()`，而不是 paged route / ingraph / gather 本身。

---

### 5.2 Swap-only 可带来 RSS drop，但 resume penalty 明显

Swap-only:

```text
C2_swap_only_shadow_off:
  resume_first_ms = 139.877 ms
  active_tps = 12.071
  process_rss_drop_mib = 124.203
  madvise_mib = 153.750
  fallback_blocks = 17
  prefetch_remaining_blocks_before_resume = 17
```

相对 C1：

```text
C2 - C1:
  resume_first_ms +57.516 ms
  process_rss_drop_mib +124.203 MiB
  fallback_blocks +17
```

解释：

```text
swap-only 能取得约 124 MiB process RSS drop，
但 resume 时需要现场恢复 17 个 blocks，
导致 resume first-token latency 增加约 57.5 ms。
```

这说明 swap-only 代表了“最大内存收益 / 较高恢复延迟”的策略端点。

---

### 5.3 Interleaved prefetch 显著降低 swap-only resume penalty

Prefetch g2:

```text
C3_prefetch_g2_shadow_off:
  resume_first_ms = 101.371 ms
  active_tps = 12.051
  process_rss_drop_mib = 124.023
  prefetch_blocks = 17
  fallback_blocks = 0
```

Prefetch g4b4:

```text
C4_prefetch_g4b4_shadow_off:
  resume_first_ms = 99.116 ms
  active_tps = 12.055
  process_rss_drop_mib = 124.062
  prefetch_blocks = 17
  fallback_blocks = 0
```

相对 swap-only：

```text
C3 - C2:
  resume_first_ms -38.506 ms
  fallback_blocks -17

C4 - C2:
  resume_first_ms -40.761 ms
  fallback_blocks -17
```

解释：

```text
interleaved prefetch 可以提前恢复 resume 所需 17 个 blocks，
将 fallback_blocks 从 17 降为 0，
并将 resume first-token latency 从约 139.9 ms 降至约 99.1–101.4 ms。
```

因此 prefetch 的作用已经明确成立：

```text
prefetch 能显著缓解 swap-only 的 resume penalty。
```

---

### 5.4 Full prefetch 后 active_tps 接近 baseline

Active throughput:

| case                            | active_tps |
| ------------------------------- | ---------: |
| C0_paged_off                    |     12.385 |
| C1_paged_on_swap_off_shadow_off |     11.999 |
| C2_swap_only_shadow_off         |     12.071 |
| C3_prefetch_g2_shadow_off       |     12.051 |
| C4_prefetch_g4b4_shadow_off     |     12.055 |

相对 baseline，C3/C4 的 active_tps 约为：

```text
C3: 12.051 / 12.385 ≈ 97.3%
C4: 12.055 / 12.385 ≈ 97.3%
```

解释：

```text
在 clean performance path 下，swap / prefetch 对 active decode throughput 的影响较小。
```

该结果比 Stage 8A 更合理，因为 Stage 8A 仍受到 `paged_shadow_validate()` 的 debug validation 开销污染。

---

### 5.5 Full prefetch 后仍有约 17–19 ms resume 剩余代价

相对 paged-off baseline：

```text
C3 - C0:
  resume_first_ms +21.586 ms

C4 - C0:
  resume_first_ms +19.331 ms
```

相对 paged-on swap-off：

```text
C3 - C1:
  resume_first_ms +19.010 ms

C4 - C1:
  resume_first_ms +16.755 ms
```

解释：

```text
full prefetch 已经消除 fallback restore，
但 swap/prefetch/resume 状态路径仍有约 17–19 ms 额外 resume first-token latency。
```

这部分剩余代价可能来自：

1. resume state transition；
2. prefetch restore 后的元数据维护；
3. block-table / row remap resume path；
4. idle-swap bookkeeping；
5. benchmark driver 的 resume 调度逻辑；
6. KV pages 被换出再恢复后的 cache locality 影响。

该问题留给后续 Stage 8D 继续定位。

---

## 6. Correctness

所有 Stage 8C cases 均满足：

```text
exit_code = 0
seq1_decoded = 128
resume_decoded = 128
active_visible_violation = 0
visible_violation_rows = 0
real_abnormal = 0
```

其中 C2 `active_violation_rows = 272` 是现有 diagnostic counter，不等价于真实 correctness failure。真实判据仍为：

```text
active_visible_violation = 0
visible_violation_rows = 0
real_abnormal = 0
seq1_decoded = 128
resume_decoded = 128
```

C3 / C4 中：

```text
prefetch_blocks = 17
fallback_blocks = 0
```

说明 resume 所需 blocks 已被提前恢复，resume 阶段没有 fallback restore。

---

## 7. Memory Interpretation

Stage 8C 是 clean performance mode，关闭了：

```text
LLAMA_KV_PAGED_MINCORE=0
```

因此：

```text
kv_nonresident_mib = 0
```

该字段不能用于本阶段的内存真实性判断。

本阶段主要使用：

```text
process_rss_drop_mib
madvise_mib
idle_owned_blocks
prefetch_blocks
fallback_blocks
resume_first_ms
active_tps
```

其中 C2/C3/C4 均显示：

```text
idle_owned_blocks = 34
madvise_mib = 153.750
process_rss_drop_mib ≈ 124 MiB
```

这说明 performance mode 中仍然执行了 idle swap / madvise 路径，并记录到 process RSS drop。

但需要注意：

```text
process_rss_drop_mib 是 swap-out 过程中观测到的 RSS drop；
不等价于 full prefetch 后 resume 前仍保留的净内存收益。
```

对于页真实 nonresident 和 KV resident/nonresident 关系，应引用 Stage 7D-B / Stage 8A 的 mincore 结果。

---

## 8. Stage 8C Core Conclusion

Stage 8C 证明：

```text
在关闭 paged_shadow_validate() 的 clean performance path 下，
paged-on swap-off 相比 paged-off baseline 仅增加约 2.6 ms resume first-token latency，
说明 paged framework 的基础开销已被压低。
```

同时：

```text
swap-only 可以带来约 124 MiB process RSS drop，
但会因 17 个 fallback blocks 使 resume first-token latency 增加约 57.5 ms。
```

开启 full interleaved prefetch 后：

```text
17 个 resume blocks 全部提前恢复；
fallback_blocks 从 17 降为 0；
resume first-token latency 从约 139.9 ms 降至约 99.1–101.4 ms；
active_tps 维持在 baseline 的约 97% 以上。
```

因此当前系统已经形成清晰的 memory-latency tradeoff：

| policy        | 内存收益                                 | resume latency               | active throughput |
| ------------- | ------------------------------------ | ---------------------------- | ----------------- |
| paged off     | 无                                    | 最低                           | 最高                |
| swap-only     | 高                                    | 高                            | 接近 baseline       |
| full prefetch | 仍有 swap-out RSS drop 记录，但部分内存收益被恢复抵消 | 明显低于 swap-only，但仍高于 baseline | 接近 baseline       |

---

## 9. Limitations

当前结果仍有以下限制：

1. Stage 8C 是 controlled benchmark，不是 ShareGPT realistic workload。
2. idle seq 数量由 `LLAMA_KV_IDLE_NUM_IDLE_SEQS=2` 人为构造。
3. ctx-size 仍为 2048，还没有扩展到更长上下文。
4. parallel=4，仍不是高并发服务端压力场景。
5. Stage 8C 关闭 mincore，因此不能单独证明 KV pages 的真实 resident/nonresident 状态。
6. `process_rss_drop_mib≈124 MiB` 不能等同于 full prefetch 后 resume 前的净内存收益。
7. full prefetch 后仍有约 17–19 ms resume first-token 额外代价。
8. auto-idle detection 尚未实现。
9. ShareGPT / server-level continuous batching workload 尚未接入。

---

## 10. Next Steps

### 10.1 保存并提交 Stage 8C 结果

建议提交文件：

```text
scripts/kv_stage8c_clean_perf_3run.sh
scripts/parse_stage8c_clean_perf_3run.py
docs/kv_paged_read_stage8c_clean_perf_results.md
```

建议 commit message：

```text
docs: record stage8c clean kv paged perf results
```

### 10.2 Stage 8D: 定位 full prefetch 后剩余 resume penalty

当前 C4 相对 baseline 仍有：

```text
C4 - C0 = +19.331 ms resume_first_ms
```

下一阶段应定位该剩余代价来源：

```text
resume state transition
prefetch restore metadata
block-table resume path
row remap resume path
idle state bookkeeping
driver resume scheduling
cache locality impact
```

### 10.3 重新做 memory-validation / performance 双模式对照

后续文档中应明确区分：

```text
memory validation mode:
  mincore / trace / shadow validation enabled
  用于证明 KV nonresident 和 RSS drop 真实性

clean performance mode:
  mincore / trace / shadow validation disabled
  用于评估 latency / throughput
```

### 10.4 扩展 workload

后续应逐步扩展：

```text
NUM_IDLE_SEQS
ctx-size
parallel
long-context workload
auto-idle detection
ShareGPT-driven workload
server-level continuous batching workload
```

目标是从 controlled benchmark 过渡到更接近真实多轮对话服务的 workload。

---

## 11. One-sentence Summary

Stage 8C 在 `LLAMA_KV_PAGED_SHADOW_VALIDATE=0` 的 clean performance path 下重新评估 swap / prefetch。3-run median 结果显示，paged-on swap-off 相比 paged-off baseline 仅增加约 2.6 ms resume first-token latency；swap-only 可带来约 124 MiB process RSS drop，但因 17 个 fallback blocks 使 resume latency 增加约 57.5 ms；full interleaved prefetch 将 17 个 blocks 全部提前恢复，fallback 降为 0，使 resume latency 从约 139.9 ms 降至约 99.1–101.4 ms，同时 active_tps 维持在 baseline 的约 97% 以上。
