# Stage 8E: KV Footprint / Residency Audit

## 1. 背景

前序 Stage 8C / Stage 8D 已经证明：

1. idle-aware KV swap/madvise 可以降低进程 current RSS；
2. full prefetch 可以消除 resume fallback；
3. resume first-token defer 可以避免 resume 首 token 阶段同步触发新 swap-out，从而把恢复延迟压回 baseline 附近。

但是，Stage 8D 主要关注的是 **clean performance path**，即：

```text
LLAMA_KV_PAGED_MINCORE=0
```

因此 Stage 8D 能说明：

```text
RSS 确实下降；
latency / TPS 开销可控；
resume first-token latency 已被 defer 优化。
```

但它还没有完整回答一个问题：

```text
RSS 下降是否确实对应 KV cache resident pages 被释放？
```

Stage 8E 的目标就是补齐这部分内存账，量化：

```text
KV 总容量；
实际 in-use KV；
idle-owned KV；
KV resident / nonresident；
swapped nonresident；
process RSS drop；
二者之间的对应关系。
```

本阶段是 diagnostic audit，不作为 latency / TPS 性能结论来源。正式性能结论仍以 Stage 8D clean perf 结果为准。

---

## 2. 目标

Stage 8E 的目标包括：

1. 打开 `LLAMA_KV_PAGED_MINCORE=1`，用 mincore 采样 KV cache resident 状态；
2. 对比 baseline 与 optimized policy 下的 KV residency；
3. 判断 RSS 下降是否与 KV nonresident 增长相匹配；
4. 解释总 KV capacity、in-use KV、idle-owned KV、swapped KV、RSS drop 之间的关系；
5. 为后续 unified KV reclaim policy 提供 footprint 数据依据。

---

## 3. 实验性质

本阶段是 **single-run diagnostic smoke**，不是正式性能 benchmark。

原因：

```text
MINCORE 会引入额外 page-table sampling 开销；
Stage 8E 关注 memory accounting，不关注 latency / TPS；
Stage 8D 已经提供 MINCORE=0 的 clean performance 结论。
```

因此，Stage 8E 中只使用 single-run 对照：

```text
B0: paged on, swap off, mincore on
B1: full prefetch + resume defer + mincore on
```

---

## 4. 新增脚本

本阶段新增两个脚本：

```text
scripts/kv_stage8e_kv_footprint_audit.sh
scripts/parse_stage8e_kv_footprint_audit.py
```

未修改：

```text
src/
examples/
CMake
public API
```

本阶段没有新增核心机制，只新增 diagnostic runner 和 parser。

---

## 5. Workload 配置

实验复用 Stage 8D-5 的核心 workload 设置，以便与前序 clean perf 结果保持可比性。

### 5.1 模型

```text
/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### 5.2 Binary

```text
build/bin/llama-kv-idle-swap-resume
```

### 5.3 Decode 配置

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

### 5.4 Idle workload 配置

```text
LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
```

含义：

```text
当前 workload 中构造多个 seq；
其中部分 seq 暂停并保留历史 KV；
另一个 seq 继续 active decode；
之后 paused seq resume。
```

这个 workload 是 controlled benchmark，用于验证机制链路和内存账，不等价于真实线上 scheduler。

---

## 6. Case 设计

### 6.1 B0: paged on, swap off, mincore on

Case 名称：

```text
B0_paged_swap_off_mincore
```

作用：

```text
作为 KV residency baseline；
打开 paged / in-graph / gather；
关闭 swap / idle swap / madvise；
打开 mincore。
```

核心 env：

```text
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_IDLE_TRACE=1
LLAMA_KV_PAGED_MINCORE=1
LLAMA_KV_PAGED_SWAP=0
LLAMA_KV_PAGED_IDLE_SWAP=0
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
```

### 6.2 B1: full prefetch + defer + mincore on

Case 名称：

```text
B1_full_prefetch_defer_mincore
```

作用：

```text
验证 optimized policy 下的 KV resident / nonresident 状态；
使用 Stage 8D-5 F2 对应策略；
打开 idle swap / madvise / full prefetch / resume defer；
打开 mincore。
```

核心 env：

```text
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_IDLE_TRACE=1
LLAMA_KV_PAGED_MINCORE=1
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=4
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=4
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

---

## 7. Diagnostic 设置

两个 case 都强制打开：

```text
LLAMA_KV_PAGED_MINCORE=1
```

并关闭会干扰性能和日志解释的 debug path：

```text
LLAMA_KV_PAGED_SHADOW_VALIDATE=0
LLAMA_KV_PAGED_TRACE=0
LLAMA_KV_PAGED_REFAULT_TRACE=0
LLAMA_KV_CACHE_DEBUG=0
LLAMA_KV_PAGED_RESUME_TIMING=0
LLAMA_KV_PAGED_RESUME_TIMING_STEP=0
```

说明：

```text
Stage 8E 只用于 residency audit；
latency / TPS 不从本阶段读取；
正式 performance 仍引用 Stage 8D clean perf。
```

---

## 8. 输出目录

```text
/root/oscomp/kv_logs/stage8e_kv_footprint_audit
```

主要输出文件：

```text
B0_paged_swap_off_mincore.out
B0_paged_swap_off_mincore.err
B0_paged_swap_off_mincore.exit

B1_full_prefetch_defer_mincore.out
B1_full_prefetch_defer_mincore.err
B1_full_prefetch_defer_mincore.exit

summary.tsv
summary_median.tsv
abnormal_summary.txt
```

其中：

```text
summary.tsv 和 summary_median.tsv 在 single-run 下内容相同。
```

---

## 9. Correctness / Safety 结果

两个 case 均正常退出：

| case                           | exit_code |
| ------------------------------ | --------: |
| B0_paged_swap_off_mincore      |         0 |
| B1_full_prefetch_defer_mincore |         0 |

Correctness / abnormal 检查：

| case                           | seq1_equal | resume_equal | real_abnormal | active_visible_violation | visible_violation_rows | active_violation_rows |
| ------------------------------ | ---------: | -----------: | ------------: | -----------------------: | ---------------------: | --------------------: |
| B0_paged_swap_off_mincore      |          0 |            0 |             0 |                        0 |                      0 |                     0 |
| B1_full_prefetch_defer_mincore |          0 |            0 |             0 |                        0 |                      0 |                     0 |

解释：

```text
seq1_equal=0 表示 seq1 active 输出与 baseline 对齐；
resume_equal=0 表示 seq0 resume 输出与 baseline 对齐；
real_abnormal=0 表示 abnormal grep 过滤后没有真实异常；
active_visible_violation=0 表示没有 active-visible safety violation。
```

因此，Stage 8E diagnostic run 没有发现 correctness 或 safety 异常。

---

## 10. Compact Footprint 结果

修正 parser 口径后，compact footprint 如下：

| case                           | kv_total | kv_resident | kv_nonresident | kv_inuse | idle_owned | swapped_nonres | swapped_nonres_peak | madvise | rss_before | rss_after | rss_drop | kv_res_drop | seq1_eq | resume_eq | abnormal |
| ------------------------------ | -------: | ----------: | -------------: | -------: | ---------: | -------------: | ------------------: | ------: | ---------: | --------: | -------: | ----------: | ------: | --------: | -------: |
| B0_paged_swap_off_mincore      |  511.750 |     511.750 |          0.000 |    0.000 |      0.000 |          0.000 |               0.000 |   0.000 |      0.000 |     0.000 |    0.000 |       0.000 |       0 |         0 |        0 |
| B1_full_prefetch_defer_mincore |  511.750 |     421.750 |         90.000 |  208.000 |    136.000 |         90.000 |             127.500 | 153.750 |   8449.750 |  8363.328 |   86.422 |      63.750 |       0 |         0 |        0 |

单位：

```text
MiB unless otherwise noted.
```

注意：

```text
B0 主要作为 mincore resident baseline。
B0 中 block-level in-use / idle-owned accounting 不作为重点解释对象。
B1 是 optimized policy 下完整 footprint / residency 解释对象。
```

---

## 11. 关键结果解读

### 11.1 KV 总容量

B1 中：

```text
kv_total = 511.750 MiB
```

表示当前配置下 llama.cpp 为 KV cache 预留的总容量。

这个值不是本次可释放上限。它包含：

```text
active-owned KV；
idle-owned KV；
mixed/shared KV；
free / unused KV capacity；
未来 token 增长所需预留空间。
```

因此不能直接理解为：

```text
KV 有 511.750 MiB，所以应该释放 511.750 MiB。
```

---

### 11.2 实际 in-use KV

B1 中：

```text
kv_inuse = 208.000 MiB
```

表示当前 workload 中实际已经写入有效 KV 内容的 block 规模。

这说明：

```text
511.750 MiB 是总容量；
当前真正承载有效 KV 的约为 208.000 MiB；
剩余部分主要是未写入 / free / tail capacity。
```

---

### 11.3 idle-owned KV

B1 中：

```text
idle_owned = 136.000 MiB
```

表示当前 in-use KV 中，有约 136 MiB 属于 idle seq 持有，并且不是当前 active seq 的直接需求对象。

这个值来自 controlled multi-seq idle/resume workload，不是自然线上 workload 的统计。

它说明当前机制针对的主要对象是：

```text
used-but-currently-idle KV
```

而不是：

```text
整个 KV capacity。
```

---

### 11.4 Final KV nonresident

B1 最终采样点：

```text
kv_nonresident = 90.000 MiB
swapped_nonres = 90.000 MiB
```

表示运行末尾有约 90 MiB KV pages 处于 nonresident 状态。

这说明：

```text
idle KV swap/madvise 确实让部分 KV pages 从 resident memory 中释放。
```

---

### 11.5 Peak swapped nonresident

B1 中：

```text
swapped_nonres_peak = 127.500 MiB
```

表示运行过程中 swapped nonresident 曾达到约 127.5 MiB。

最终值低于 peak：

```text
final swapped_nonres = 90.000 MiB
peak swapped_nonres = 127.500 MiB
```

原因是：

```text
resume / prefetch / later access 会把部分 swapped pages 重新拉回 resident；
因此 final snapshot 和 peak snapshot 不完全相同。
```

这也是 parser 二次修正时区分：

```text
swapped_nonresident_mib
swapped_nonresident_peak_mib
```

的原因。

---

### 11.6 madvise 覆盖与 RSS drop

B1 中：

```text
madvise = 153.750 MiB
rss_before = 8449.750 MiB
rss_after  = 8363.328 MiB
rss_drop   = 86.422 MiB
```

这说明：

```text
运行过程中对约 153.75 MiB KV-related pages 执行 madvise；
最终同一时间口径下，进程 current RSS 下降约 86.42 MiB。
```

`madvise` 不等于 `rss_drop`，原因包括：

```text
madvise 是被建议释放的累计/覆盖规模；
部分页可能已经 nonresident；
部分页可能被后续 prefetch / resume 拉回；
kernel RSS accounting 存在采样时机差异；
neighbor block / page alignment / safety policy 会影响转化率。
```

因此，本阶段更重要的对应关系是：

```text
final KV nonresident ≈ 90.000 MiB
final process RSS drop ≈ 86.422 MiB
```

这两个数数量级高度接近，说明 RSS drop 与 KV resident page 释放具有直接对应关系。

---

## 12. 派生比例

基于 B1：

```text
kv_total = 511.750 MiB
kv_inuse = 208.000 MiB
idle_owned = 136.000 MiB
final kv_nonresident = 90.000 MiB
peak swapped_nonresident = 127.500 MiB
final process RSS drop = 86.422 MiB
madvise = 153.750 MiB
```

可得到：

| 指标                                    |                计算 |     结果 |
| ------------------------------------- | ----------------: | -----: |
| final RSS drop / KV total             |  86.422 / 511.750 | 16.89% |
| final KV nonresident / KV total       |  90.000 / 511.750 | 17.59% |
| final RSS drop / KV in-use            |  86.422 / 208.000 | 41.55% |
| idle-owned / KV in-use                | 136.000 / 208.000 | 65.38% |
| final RSS drop / idle-owned           |  86.422 / 136.000 | 63.55% |
| peak swapped nonresident / idle-owned | 127.500 / 136.000 | 93.75% |
| final RSS drop / madvise              |  86.422 / 153.750 | 56.21% |
| final RSS drop / process RSS before   | 86.422 / 8449.750 |  1.02% |

这些比例说明：

1. 如果以整个进程 RSS 为分母，下降比例较小，因为模型权重和 runtime buffers 占据大头；
2. 如果以 KV total 为分母，最终下降约 16.9%；
3. 如果以实际 in-use KV 为分母，最终下降约 41.6%；
4. 如果以 idle-owned KV 为分母，过程中 peak swapped nonresident 达到约 93.8%，说明 idle-owned KV 的可释放转化较高；
5. final snapshot 中 RSS drop 低于 peak，因为 resume / prefetch 之后部分 KV 会重新 resident。

---

## 13. 与 Stage 8D 的关系

Stage 8D 是 clean performance run：

```text
LLAMA_KV_PAGED_MINCORE=0
```

它证明：

```text
full prefetch + resume defer 可以消除 fallback；
resume first-token latency 可以回到 baseline 附近；
active TPS 保持可接受。
```

Stage 8E 是 diagnostic residency run：

```text
LLAMA_KV_PAGED_MINCORE=1
```

它证明：

```text
RSS drop 对应 KV resident pages 的真实释放；
madvise 不是单纯计数器；
KV nonresident 与 process RSS drop 数量级一致。
```

因此，两者分工不同：

| 阶段       | 主要目标                           | 是否使用 MINCORE | 是否作为性能结论 |
| -------- | ------------------------------ | -----------: | -------: |
| Stage 8D | latency / TPS / resume defer   |            否 |        是 |
| Stage 8E | KV footprint / residency audit |            是 |        否 |

---

## 14. 为什么不是释放整个 KV capacity

本阶段 B1 中：

```text
kv_total = 511.750 MiB
kv_inuse = 208.000 MiB
idle_owned = 136.000 MiB
```

说明总 KV capacity 中只有一部分当前承载有效 KV 内容。

当前机制释放的是：

```text
已经写入有效 KV；
属于 idle seq；
当前 active request 不需要；
通过 safety check；
可以 swap/madvise；
未来 resume 前可以 prefetch。
```

即：

```text
used-but-idle KV
```

它不负责释放：

```text
尚未写入有效 token 的 free / unused tail capacity。
```

unused / free capacity 属于另一类优化对象：

```text
tail madvise；
lazy clear；
free block madvise；
lazy allocation。
```

这与当前 idle KV swap/madvise 不矛盾，而是可以在后续统一成更完整的 KV resident memory manager。

---

## 15. 机制定位

当前 idle-aware KV swap/madvise 机制适合：

```text
高并发；
长上下文；
多 session；
部分请求 paused / idle；
后续可能 resume；
KV cache 占用显著。
```

它不适合单独解释为：

```text
压缩整个 KV cache；
释放所有预分配 KV capacity；
替代 lazy allocation。
```

更准确的定位是：

```text
针对 used-but-currently-idle KV 的运行时 resident memory 回收机制。
```

后续与 tail / lazy clear 融合后，可覆盖两类场景：

| 场景          | 主要浪费来源                     | 适合机制                                   |
| ----------- | -------------------------- | -------------------------------------- |
| 低并发 / 中短上下文 | KV capacity 预留较大，但实际用不满    | tail / lazy clear / free block release |
| 高并发 / 长上下文  | 已写入大量 KV，其中部分 session idle | idle KV swap/madvise + prefetch/defer  |
| resume 阶段   | swapped KV 需要恢复访问          | prefetch + resume defer                |

---

## 16. 结论

Stage 8E 完成了 KV footprint / residency audit，证明 optimized policy 下 RSS 下降确实对应 KV resident page 释放。

核心结果：

```text
KV total capacity:             511.750 MiB
KV in-use:                     208.000 MiB
idle-owned KV:                 136.000 MiB
final KV nonresident:           90.000 MiB
peak swapped nonresident:      127.500 MiB
madvise coverage:              153.750 MiB
final process RSS drop:         86.422 MiB
correctness:                   clean
real abnormal:                 0
```

可以总结为：

```text
在 controlled multi-seq idle/resume workload 中，优化机制在不破坏 active/resume correctness 的前提下，将最终约 90 MiB KV pages 转为 nonresident，并带来约 86.4 MiB 进程 current RSS 下降。该结果说明内存收益来自 KV resident pages 的真实释放，而不是单纯 telemetry 计数。
```

---

## 17. 限制

本阶段仍有以下限制：

1. workload 是 controlled benchmark，不是自然线上请求流；
2. idle-owned KV 的规模来自人为构造的 active / paused / resume 时序；
3. MINCORE 打开后不用于性能评估；
4. single-run diagnostic 不提供统计稳定性结论；
5. 当前机制只覆盖 used-but-idle KV，没有同时释放 unused/free tail capacity；
6. 尚未接入 llama-server 的真实 slot / continuous batching scheduler。

---

## 18. 下一步

建议下一步分两条推进：

### 18.1 Stage 8F: Unified KV Reclaim Policy 设计

目标：

```text
将 tail / lazy clear 与 idle KV swap/madvise 统一成 KV lifecycle-aware resident memory management。
```

设计方向：

```text
unused/free KV capacity -> tail/lazy/free-block release
used-but-idle KV        -> idle swap/madvise
resume-needed KV        -> prefetch/defer
active-needed KV        -> always protected
```

可设计统一策略：

```text
LLAMA_KV_RECLAIM_POLICY=off|tail_only|idle_only|auto
```

### 18.2 Stage 9: Semi-real Multi-session Workload

目标：

```text
用真实/半真实多轮 prompt 构造 multi-session active / paused / resume workload；
验证 idle-owned KV 是否能在更自然文本场景下稳定产生；
验证 RSS drop 与 latency trade-off。
```

Stage 9 暂不建议直接接入 llama-server，优先新增 simplified scheduler driver，保持实验可控、可复现、可归因。