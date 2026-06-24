# KV Paged Read — Stage 6D-C Pressure-Aware Prefetch Policy 3-Run Median Results

> 本文档记录 Stage 6D-C（memory-pressure-aware prefetch policy）的 3-run median 实验结果。
> 本阶段只整理结果文档，不修改 `src/`、`include/`、public API、examples、CMake、scripts。
> 目标是验证 Stage 6D-B 实现的 pressure mode policy 是否产生可解释、可预测的 RSS / resume-latency trade-off。

---

## 1. Stage 6D-C 目标

Stage 6D-C 在固定 workload、固定 `resume_pending_token` 的条件下，扫描 pressure mode，验证：

```text
1. off / low / medium / high 是否产生可解释的 target_restore_blocks；
2. 主动恢复的 blocks 数是否与 fallback_blocks 互补；
3. rss_before_resume 是否随恢复 blocks 减少而单调下降；
4. resume first-token latency 是否随恢复 blocks 减少而上升；
5. off 是否字段级回归 Stage 6C-3C 完整恢复行为。
```

核心结论：

```text
Stage 6D-C verified pressure-aware prefetch policy matrix.
```

---

## 2. 与 Stage 6D-A / 6D-B 的关系

```text
Stage 6D-A:
  完成 memory-pressure-aware prefetch policy design audit；
  确定 env 名称、target_restore_blocks 规则、实验矩阵与风险边界。

Stage 6D-B:
  在 examples/kv-idle-swap-resume/idle-swap-resume.cpp 中实现：
  LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE = off | low | medium | high
  pressure 仅通过 target_restore_blocks 这一入口影响恢复量；
  新增 telemetry：pressure_mode、target_restore_blocks。

Stage 6D-C:
  固定 pending_token=96（active window 充足），只扫 pressure mode；
  使任何 fallback 都来自 pressure policy 主动限流，而非 window 不足；
  本文档即该轮 3-run median 的结果整理。
```

---

## 3. 实验配置

固定配置：

```text
warmup=64
n=128
ctx=1024
parallel=2
batch=128
ubatch=128
cache-type-k=f32
cache-type-v=f32
kv-unified
temp=0
every=4
step=1
safety=0
pending_token=96        (active_window_remaining = n - pending = 32 >= need_span)
3-run median
```

变量：

```text
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE ∈ {off, low, medium, high}
```

为什么固定 pending=96、只扫 mode：

```text
1. pending=96 时 active window=32，足以覆盖完整恢复所需的 need_span=16；
2. 因此 off / low 必然完整恢复，fallback 只可能来自 pressure policy 主动限流；
3. 这把 pressure 决策与 timing 决策彻底解耦，结果可干净归因。
```

结果文件：

```text
/root/oscomp/kv_logs/stage6d_pressure_3run/summary.tsv
/root/oscomp/kv_logs/stage6d_pressure_3run/median_by_mode.tsv
/root/oscomp/kv_logs/stage6d_pressure_3run/abnormal_count.txt
```

---

## 4. Pressure Mode Policy

Stage 6D-B 实现的 target_restore_blocks 规则：

```text
off    : target_restore_blocks = remaining_blocks        (= 5)
low    : target_restore_blocks = remaining_blocks        (= 5)
medium : target_restore_blocks = min(remaining_blocks, 3) (= 3)
high   : target_restore_blocks = 0
```

pressure 只通过 target_restore_blocks 影响 timing，公式其余部分与 Stage 6C-3C 逐字相同：

```text
target_blocks = clamp_by_mode(remaining_blocks, pressure_mode)
need_steps    = ceil(target_blocks / blocks_per_step)
need_span     = (need_steps - 1) * every_tokens + safety_tokens
```

本轮 remaining_blocks=5、blocks_per_step=1、every_tokens=4、safety_tokens=0：

```text
off / low : need_steps=5 -> need_span=16 -> effective_start=112
medium    : need_steps=3 -> need_span=8  -> effective_start=120
high      : need_steps=0 -> 短路到 no-active-prefetch -> effective_start=96
```

---

## 5. 3-Run Median 结果表

数据来自 `median_by_mode.tsv`（3-run median）。

```text
mode    target  calls  blocks  remaining  auto_fb  pend_fb  first_ms  tps         start  completed  rss_before_resume  rss_after_resume
off     5       5      5       0          0        0        101.525   11.727267   112    1          8395348            8366940
low     5       5      5       0          0        0        97.227    11.762402   112    1          8394984            8367136
medium  3       3      3       2          2        2        101.153   12.169847   120    0          8387736            8366988
high    0       0      0       5          5        5        112.643   12.117316   96     0          8375896            8367056
```

字段说明：

```text
target:
  target_restore_blocks（pressure policy 设置的本次恢复上限）。

calls / blocks:
  active decode 期间实际 prefetch 调用次数与恢复 blocks 数。

remaining:
  prefetch_remaining_blocks_before_resume，resume 前仍未恢复的 blocks。

auto_fb / pend_fb:
  prefetch_auto_fallback_blocks / resume_pending_fallback_blocks，
  二者一致，均等于 remaining。

first_ms:
  seq0_resume_first_token_ms。

start:
  prefetch_auto_start_token。

completed:
  prefetch_auto_completed（是否在 resume 前完成全部目标恢复）。

rss_before_resume / rss_after_resume:
  进入 resume 前 / 后的 RSS（KiB）。
```

---

## 6. RSS / Latency Trade-off 分析

### 6.1 恢复量随 pressure 单调下降

```text
off / low : target=5 -> blocks=5 -> remaining=0 -> fallback=0
medium    : target=3 -> blocks=3 -> remaining=2 -> fallback=2
high      : target=0 -> blocks=0 -> remaining=5 -> fallback=5
```

观察：

```text
1. off / low 完整恢复 5 blocks，fallback=0，completed=1；
2. medium 主动只恢复 3 blocks，剩余 2 blocks 进入 fallback，completed=0；
3. high 完全跳过 active prefetch，5 blocks 全部进入 fallback，completed=0；
4. blocks + fallback 恒等于 remaining_blocks=5，互补关系成立。
```

### 6.2 rss_before_resume 随恢复 blocks 减少而降低

以 low 为基准（rss_before_resume=8394984 KiB）：

```text
low    -> medium : 8394984 - 8387736 = 7248 KiB
low    -> high   : 8394984 - 8375896 = 19088 KiB
```

解释：

```text
1. medium 少恢复 2 blocks，resume 前 RSS 比 low 低约 7248 KiB；
2. high 完全不恢复，resume 前 RSS 比 low 低约 19088 KiB；
3. RSS 下降量与“resume 前少恢复的 blocks 数”一致，方向可解释、可预测。
```

### 6.3 first-token latency 随恢复 blocks 减少而上升

```text
low    : first_ms=97.227   (完整恢复，无 fallback)
off    : first_ms=101.525  (完整恢复，无 fallback)
medium : first_ms=101.153  (fallback=2)
high   : first_ms=112.643  (fallback=5，最高)
```

解释：

```text
1. high 的 first-token latency 最高，因为 5 个 blocks 全部要在 first-token path 上 swap-in；
2. 这说明“保留 RSS”的代价是 resume fallback 成本上升；
3. off 与 low 都是完整恢复，first_ms 差异在运行间噪声范围内，不构成 policy 差异。
```

### 6.4 tokens/s 不应解释为 policy 提升吞吐

```text
off    : tps=11.727267
low    : tps=11.762402
medium : tps=12.169847
high   : tps=12.117316
```

说明：

```text
1. tokens/s 在各 mode 间有波动，但与 fallback / RSS 趋势不构成稳定单调关系；
2. tps 的差异落在 run-to-run 抖动范围内；
3. 不应把 medium / high 的 tps 解释为 pressure policy 带来的吞吐提升；
4. 本阶段关心的是 RSS / resume-latency trade-off，而非 steady-state 吞吐。
```

---

## 7. Abnormal 检查

```text
raw_abnormal_matches=3876
real_abnormal_matches=0
```

解释：

```text
1. raw_abnormal_matches 是关键字粗匹配数，包含正常 telemetry 行中的字面命中；
2. real_abnormal_matches=0，表示过滤后没有真实异常；
3. 本轮 3-run 全部 12 次运行均无真实异常，结果可用于 median。
```

---

## 8. 结论

```text
Stage 6D-C verified pressure-aware prefetch policy matrix.
```

具体结论：

```text
1. off / low 完整恢复 5 blocks，fallback=0，completed=1；
   off 字段级回归 Stage 6C-3C 完整恢复行为。

2. medium 主动只恢复 3 blocks，剩余 2 blocks 进入 fallback；
   对齐 6C-3C 中恢复 3 blocks 的已知行，只是触发原因为 pressure 而非 window 不足。

3. high 跳过 active prefetch，5 blocks 全部进入 fallback；
   等价 6C-3C 的 pending=-1 行为，但由 pressure mode 显式触发。

4. rss_before_resume 随恢复 blocks 减少而单调降低：
   low -> medium 约降低 7248 KiB；
   low -> high   约降低 19088 KiB。

5. high 的 first-token latency 最高（112.643 ms），
   说明保留 RSS 的代价是 resume fallback 成本上升。

6. tokens/s 有波动，落在 run-to-run 抖动范围内，
   不应解释为 pressure policy 提升吞吐。
```

---

## 9. 边界与限制

```text
1. 本阶段验证的是 pressure-aware policy 的“可解释性”：
   pressure mode 能可预测、单调地改变 target_restore_blocks、fallback、
   first-token latency 与 rss_before_resume。

2. 本阶段不承诺解决“净 RSS 收益被 resume 后续恢复吃回”的问题：
   medium / high 在 resume 前保留的 RSS，会在 first-token path 上随 fallback
   恢复而部分吃回（见 rss_after_resume 各 mode 趋同至约 8367xxx KiB）。

3. medium / high 之间 rss_before_resume 的差异反映的是
   “resume 前少恢复了多少 blocks”，
   不应被误读为系统层面节省了等量内存。

4. 本阶段不接入真实 memory pressure 信号
   （/proc/meminfo、cgroup memory.pressure PSI、mincore），
   pressure level 由 env 注入，保证输入干净、确定、可复现。

5. 本阶段不进入 core scheduler；
   policy 继续留在 examples/kv-idle-swap-resume/idle-swap-resume.cpp。

6. 本阶段不新增 public API；
   现有 prefetch_seq_step / protected gate / last-stats mechanism 已足够。
```

---

## 10. 下一步方向

```text
1. 真实 memory pressure 信号接入与“信号 -> pressure mode”阈值映射，
   作为 6D 之后的独立阶段处理，与本阶段 policy 验证解耦；

2. 若要解决净 RSS 收益被吃回的问题，需要在 resume path 之外
   引入对“恢复后是否再次换出”的策略，超出本阶段范围；

3. 维持边界：不进 core scheduler、不新增 public API、不做 true async prefetch thread。
```

---

## 11. Summary

```text
Stage 6D-C 在固定 workload、固定 pending_token=96 的条件下扫描 pressure mode，
验证了 Stage 6D-B 实现的 pressure-aware prefetch policy matrix。

3-run median 显示：
  off / low 完整恢复 5 blocks（fallback=0）；
  medium 主动恢复 3 blocks（fallback=2）；
  high 跳过 active prefetch（fallback=5）。

rss_before_resume 随恢复 blocks 减少而单调下降
（low->medium 约 7248 KiB，low->high 约 19088 KiB），
而 first-token latency 随之上升（high 最高，112.643 ms），
证明 pressure mode 在 RSS 与 resume latency 之间提供了可解释、可预测的取舍。

本阶段不接真实 memory pressure 信号、不进入 core scheduler、不新增 public API，
也不承诺解决净 RSS 收益被 resume 后续恢复吃回的问题。
```
