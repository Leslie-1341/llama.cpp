# Stage 8B-2: Shadow Validation Gate 与 Paged Path 性能开销定位结果

## 1. Background

Stage 8A 已经验证了 idle KV swap / prefetch / pressure policy 的基本机制：

1. `swap-only` 可以将 idle-owned KV blocks 换出并通过 `madvise(DONTNEED)` 降低物理驻留内存。
2. 在 `NUM_IDLE_SEQS=2` 的 controlled benchmark 中，34 个 idle-owned blocks 可带来约 `127.5 MiB` KV nonresident 和约 `123–124 MiB` process RSS drop。
3. `swap-only` 会在 seq0 resume 时产生 fallback restore，导致 resume first-token latency 明显升高。
4. interleaved prefetch 可以提前恢复 resume 所需 blocks，将 fallback 降到 0，从而降低 resume first-token latency。

但 Stage 8A fair performance matrix 也暴露了一个问题：

```text
full prefetch 能消除 swap fallback 延迟，
但 full prefetch 后的 resume first-token latency 仍明显高于 paged-off baseline。
```

Stage 8A fair performance 3-run median：

| case                     | 含义                 | resume_first_ms | active_tps | prefetch_blocks | fallback_blocks |
| ------------------------ | ------------------ | --------------: | ---------: | --------------: | --------------: |
| P0_paged_off             | 原始 baseline        |          82.452 |     12.126 |               0 |               0 |
| P1_paged_on_swap_off_min | paged on, swap off |         111.291 |      8.850 |               0 |               0 |
| P2_swap_only_min         | swap-only          |         152.840 |     10.907 |               0 |              17 |
| P3_prefetch_g2_min       | full prefetch      |         111.398 |     10.794 |              17 |               0 |
| P4_prefetch_g3_min       | full prefetch      |         111.291 |     10.943 |              17 |               0 |

可以看到：

```text
P3/P4 full prefetch 已经把 fallback_blocks 降到 0，
但 resume_first_ms 只能回到 P1 paged-on swap-off 水平，
仍然比 P0 paged-off baseline 慢约 29 ms。
```

因此 Stage 8B 的重点从 prefetch 策略转向 paged framework 自身开销定位。

---

## 2. Problem: Paged Path Baseline Overhead

Stage 8B-1 overhead ablation 进一步拆分了 paged path 的开销来源。

Stage 8B-1 3-run median：

| case                                | 含义                                               | resume_first_ms | active_tps |
| ----------------------------------- | ------------------------------------------------ | --------------: | ---------: |
| Q0_paged_off                        | paged off baseline                               |          80.966 |     12.397 |
| Q1_paged_bookkeeping_only           | paged=1, ingraph=0, gather_nonidentity=0, swap=0 |         108.924 |      9.125 |
| Q2_paged_ingraph_only               | Q1 + ingraph                                     |         110.973 |      8.783 |
| Q4_paged_ingraph_gather_nonidentity | Q2 + gather_nonidentity                          |         110.403 |      8.906 |
| Q5_paged_idle_trace_no_swap         | Q4 + idle_trace, no swap                         |         110.785 |      8.962 |
| Q6_swap_only_full_path              | full swap-only path                              |         154.758 |     10.844 |

关键 delta：

| delta   | resume_first_ms 变化 | 解释                              |
| ------- | -----------------: | ------------------------------- |
| Q1 - Q0 |         +27.958 ms | paged bookkeeping 基础开销          |
| Q2 - Q1 |          +2.049 ms | ingraph / row_idx / gather 额外开销 |
| Q4 - Q2 |          -0.570 ms | gather_nonidentity scan，噪声范围内   |
| Q5 - Q4 |          +0.382 ms | idle maintenance scan，噪声范围内     |
| Q6 - Q5 |         +43.973 ms | swap fallback 代价                |

该结果说明：

```text
最大基础开销在 Q1 阶段已经出现。
也就是说，即使 ingraph=0、gather_nonidentity=0、swap=0，
只要 LLAMA_KV_PAGED=1，resume_first_ms 就已经比 paged-off baseline 慢约 28 ms。
```

结合源码审计，最大嫌疑是 `paged_shadow_validate()`：

1. 它在 paged path 中每 step 执行。
2. 它会对 K/V raw memory 做逐层、逐行 byte-level 内容校验。
3. 它是 debug validation，不是推理功能路径所必需。
4. 在 performance mode 中继续执行会严重污染延迟和吞吐测试。

---

## 3. Change: `LLAMA_KV_PAGED_SHADOW_VALIDATE`

本阶段新增环境变量：

```text
LLAMA_KV_PAGED_SHADOW_VALIDATE
```

语义：

```text
默认值：1
显式设置为 0 时，跳过 paged_shadow_validate()
```

设计原则：

1. 默认行为保持不变。
2. debug / correctness 验证时仍可开启 shadow validation。
3. performance benchmark 中可以显式关闭，避免 debug validation 污染性能路径。
4. 不删除 `paged_shadow_validate()`。
5. 不改变 `swap-out` / `madvise` / `swap-in` / `prefetch` / `row remap` 功能路径。
6. 不影响 `paged_note_cells()`、`paged_assert_identity()`、active-visible violation 检查等其他机制。

修改文件：

```text
src/llama-kv-cache.h
src/llama-kv-cache.cpp
```

核心改动：

```text
新增 paged_shadow_validate_enabled 成员；
在 llama_kv_cache 构造函数中读取 LLAMA_KV_PAGED_SHADOW_VALIDATE；
在 llama_kv_cache_context::next() 中对 paged_shadow_validate() 调用加 gate。
```

该改动的定位是：

```text
将 debug validation path 与 performance path 分离。
```

---

## 4. Experiment Setup

### Benchmark

使用现有 benchmark：

```text
build/bin/llama-kv-idle-swap-resume
```

固定配置：

```text
model: /root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
n=128
ctx-size=2048
batch-size=128
ubatch-size=128
seed=1
temp=0
cache-type-k=f32
cache-type-v=f32
kv-unified
parallel=4
LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
```

Performance mode 中关闭测量型开销：

```text
LLAMA_KV_PAGED_MINCORE=0
LLAMA_KV_PAGED_TRACE=0
LLAMA_KV_PAGED_REFAULT_TRACE=0
LLAMA_KV_CACHE_DEBUG=0
```

### Case Design

| case                               | 含义                                                               |
| ---------------------------------- | ---------------------------------------------------------------- |
| S0_paged_off                       | paged off baseline                                               |
| S1_paged_bookkeeping_shadow_on     | paged on, ingraph off, gather off, shadow validate on            |
| S2_paged_bookkeeping_shadow_off    | paged on, ingraph off, gather off, shadow validate off           |
| S3_paged_ingraph_shadow_off        | paged on, ingraph on, shadow validate off                        |
| S4_paged_ingraph_gather_shadow_off | paged on, ingraph on, gather_nonidentity on, shadow validate off |
| S5_prefetch_full_shadow_off        | full prefetch path, shadow validate off                          |

验证脚本：

```text
scripts/kv_stage8b2_shadow_validate_gate_once.sh
scripts/kv_stage8b2_shadow_validate_gate_3run.sh
scripts/parse_stage8b2_shadow_validate_gate_3run.py
```

---

## 5. Once Result

Stage 8B-2 once 结果：

| case                               | resume_first_ms | active_tps | prefetch_blocks | fallback_blocks |
| ---------------------------------- | --------------: | ---------: | --------------: | --------------: |
| S0_paged_off                       |          79.131 |     12.287 |               0 |               0 |
| S1_paged_bookkeeping_shadow_on     |         110.961 |      8.854 |               0 |               0 |
| S2_paged_bookkeeping_shadow_off    |          79.215 |     12.252 |               0 |               0 |
| S3_paged_ingraph_shadow_off        |          81.827 |     12.082 |               0 |               0 |
| S4_paged_ingraph_gather_shadow_off |          81.307 |     12.036 |               0 |               0 |
| S5_prefetch_full_shadow_off        |          99.552 |     11.972 |              17 |               0 |

关键观察：

```text
S1 - S0 = +31.830 ms
S2 - S0 = +0.084 ms
```

也就是说，单轮结果中，关闭 `paged_shadow_validate()` 后，paged bookkeeping path 几乎回到 paged-off baseline。

---

## 6. 3-run Median Result

Stage 8B-2 3-run median 结果：

| case                               | 含义                           | resume_first_ms | active_tps | prefetch_blocks | fallback_blocks | seq1/resume decoded | real_abnormal |
| ---------------------------------- | ---------------------------- | --------------: | ---------: | --------------: | --------------: | ------------------- | ------------: |
| S0_paged_off                       | baseline                     |          80.024 |     12.284 |               0 |               0 | 128/128             |             0 |
| S1_paged_bookkeeping_shadow_on     | shadow on                    |         108.477 |      9.150 |               0 |               0 | 128/128             |             0 |
| S2_paged_bookkeeping_shadow_off    | shadow off                   |          80.095 |     12.328 |               0 |               0 | 128/128             |             0 |
| S3_paged_ingraph_shadow_off        | ingraph, shadow off          |          84.124 |     12.042 |               0 |               0 | 128/128             |             0 |
| S4_paged_ingraph_gather_shadow_off | ingraph + gather, shadow off |          80.765 |     12.194 |               0 |               0 | 128/128             |             0 |
| S5_prefetch_full_shadow_off        | full prefetch, shadow off    |          99.359 |     12.173 |              17 |               0 | 128/128             |             0 |

关键 delta：

| delta   | resume_first_ms 变化 |   active_tps 变化 |
| ------- | -----------------: | --------------: |
| S1 - S0 |         +28.453 ms |  12.284 → 9.150 |
| S2 - S0 |          +0.071 ms | 12.284 → 12.328 |
| S3 - S2 |          +4.029 ms | 12.328 → 12.042 |
| S4 - S3 |          -3.359 ms | 12.042 → 12.194 |
| S5 - S4 |         +18.594 ms | 12.194 → 12.173 |

核心结论：

```text
paged_shadow_validate() 是之前 paged performance path 基础开销的主要来源。
```

关闭该 debug validation 后：

```text
paged bookkeeping 基本回到 paged-off baseline；
ingraph / gather_nonidentity 在当前 benchmark 下只带来小幅开销；
full prefetch + shadow off 后 active_tps 接近 baseline；
full prefetch + shadow off 后 resume_first_ms 仍比 baseline 高约 19 ms。
```

---

## 7. Interpretation

### 7.1 Shadow validation 是主要基础开销来源

对比：

```text
S0_paged_off:
  resume_first_ms = 80.024 ms

S1_paged_bookkeeping_shadow_on:
  resume_first_ms = 108.477 ms

S2_paged_bookkeeping_shadow_off:
  resume_first_ms = 80.095 ms
```

可以看到：

```text
开启 shadow validation 时，paged bookkeeping path 比 baseline 慢约 28.5 ms；
关闭 shadow validation 后，paged bookkeeping path 与 baseline 几乎一致。
```

这说明 `paged_shadow_validate()` 原本作为 debug 校验逻辑，不应放在 performance path 中默认执行。

### 7.2 Ingraph / gather_nonidentity 不是主要瓶颈

关闭 shadow validation 后：

```text
S2_paged_bookkeeping_shadow_off:
  resume_first_ms = 80.095 ms

S3_paged_ingraph_shadow_off:
  resume_first_ms = 84.124 ms

S4_paged_ingraph_gather_shadow_off:
  resume_first_ms = 80.765 ms
```

这说明在当前 controlled benchmark 中：

```text
INGRAPH / GATHER_NONIDENTITY 的额外开销较小；
之前的主要 overhead 不是 ggml_get_rows gather，而是 shadow validation。
```

### 7.3 Full prefetch 消除了 fallback，但仍有剩余 resume 代价

S5 结果：

```text
S5_prefetch_full_shadow_off:
  prefetch_blocks = 17
  fallback_blocks = 0
  resume_first_ms = 99.359 ms
  active_tps = 12.173
```

对比 baseline：

```text
S0_paged_off:
  resume_first_ms = 80.024 ms
  active_tps = 12.284
```

说明：

```text
full prefetch + shadow off 后，active_tps 基本接近 baseline；
resume_first_ms 仍有约 19.3 ms 额外代价。
```

该剩余代价可能来自：

1. prefetch / restore 后的状态维护；
2. idle swap-out 与 resume restore 的元数据路径；
3. row remap / block-table 在 resume path 中的残余成本；
4. benchmark driver 调度差异；
5. 被提前恢复的 KV pages 对 cache locality 的影响。

这部分属于后续优化目标。

---

## 8. Correctness

所有 Stage 8B-2 3-run median case 均满足：

```text
exit_code = 0
seq1_decoded = 128
resume_decoded = 128
active_visible_violation = 0
visible_violation_rows = 0
real_abnormal = 0
```

S5 full prefetch case 中：

```text
prefetch_blocks = 17
fallback_blocks = 0
```

说明 resume 所需 blocks 已经通过 interleaved prefetch 提前恢复，resume 阶段没有 fallback restore。

关闭 `paged_shadow_validate()` 后，模型功能路径仍正常完成。但需要注意：

```text
关闭 shadow validation 会减少内部 byte-level K/V 内容校验能力。
```

因此建议使用方式为：

```text
debug / correctness 开发阶段：
  LLAMA_KV_PAGED_SHADOW_VALIDATE=1

performance benchmark / final evaluation：
  LLAMA_KV_PAGED_SHADOW_VALIDATE=0
```

并继续通过以下外部指标保证正确性：

```text
decoded token count
baseline output cmp
active_visible_violation
visible_violation_rows
abnormal grep
```

---

## 9. Impact

本阶段的核心影响是将 debug path 和 performance path 解耦。

修改前：

```text
LLAMA_KV_PAGED=1 后，paged_shadow_validate() 默认每 step 执行；
即使 swap off、ingraph off、gather_nonidentity off，仍产生明显性能开销。
```

修改后：

```text
默认行为保持不变；
performance mode 可显式设置 LLAMA_KV_PAGED_SHADOW_VALIDATE=0；
关闭 shadow validation 后，paged bookkeeping 基本回到 baseline。
```

这使后续评估更公平：

```text
memory validation mode:
  开启 mincore / trace / shadow validation，验证机制正确性和内存释放真实性。

performance mode:
  关闭 mincore / trace / shadow validation，评估真实推理性能。
```

---

## 10. Limitations

当前结果仍有以下限制：

1. 仍是 controlled benchmark，不是 ShareGPT realistic workload。
2. idle seq 仍由 `LLAMA_KV_IDLE_NUM_IDLE_SEQS` 人为构造。
3. S5 full prefetch 虽然 active_tps 接近 baseline，但 resume_first_ms 仍有约 19 ms 额外代价。
4. 当前 `kv_nonresident_mib` 在 performance mode 下关闭了 mincore，因此为 0，不用于判断内存释放。
5. S5 中 `process_rss_drop_mib≈124.250` 表示 swap-out 曾达到的 RSS drop，不等价于 full prefetch 后 resume 前剩余的净内存收益。
6. 还没有在更高并发、更长上下文下验证总 RSS 下降比例。
7. 还没有实现 auto-idle detection。
8. 还没有接入 ShareGPT / server-level continuous batching workload。

---

## 11. Next Steps

建议后续路线：

### 11.1 固化 shadow validation gate

将以下文件作为阶段 checkpoint 提交：

```text
src/llama-kv-cache.h
src/llama-kv-cache.cpp
scripts/kv_stage8b2_shadow_validate_gate_once.sh
scripts/kv_stage8b2_shadow_validate_gate_3run.sh
scripts/parse_stage8b2_shadow_validate_gate_3run.py
docs/kv_paged_read_stage8b2_shadow_validate_gate_results.md
```

建议 commit message：

```text
kv-cache: gate paged shadow validation for perf path
```

### 11.2 继续定位 S5 剩余 resume 代价

当前 S5 仍比 baseline 慢约 19 ms：

```text
S0 resume_first_ms = 80.024 ms
S5 resume_first_ms = 99.359 ms
```

后续可继续定位：

```text
prefetch restore path
block-table resume path
row remap resume path
idle state transition
driver resume scheduling
```

### 11.3 恢复内存-延迟综合评估

在 performance path 清理后，应重新做一组 representative matrix：

```text
baseline
swap-only
partial prefetch
full prefetch
```

同时区分：

```text
memory validation mode：证明 KV nonresident / RSS drop
performance mode：证明 resume latency / active_tps
```

### 11.4 扩展到更高并发和长上下文

当前 `NUM_IDLE_SEQS=2` 只带来约 `127.5 MiB` KV nonresident。后续应扩大：

```text
NUM_IDLE_SEQS
ctx-size
parallel
long-context workload
```

以验证高并发长上下文场景下的总 RSS 下降比例。

### 11.5 Auto-idle 与 ShareGPT workload

最后再推进：

```text
auto-idle detection
ShareGPT-driven workload
server-level continuous batching evaluation
```

目标是从 controlled benchmark 过渡到更真实的对话推理场景。

---

## 12. One-sentence Summary

Stage 8B-2 通过新增 `LLAMA_KV_PAGED_SHADOW_VALIDATE` 开关，将 debug-only shadow validation 从 performance path 中分离。3-run median 结果显示，开启 shadow validation 时 paged bookkeeping 比 paged-off baseline 慢约 28.5 ms；关闭后几乎回到 baseline。full prefetch + shadow off 后，fallback 降为 0，active_tps 接近 baseline，但 resume first-token latency 仍有约 19 ms 额外代价。该阶段确认了此前 paged path 的主要基础开销来自 `paged_shadow_validate()`，并为后续公平评估 swap/prefetch 机制提供了更干净的 performance path。
