# KV Paged Read — Stage 6C-3B Resume-Pending Prefetch Results

> 本文档记录 Stage 6C-3B 的 example-level pseudo-scheduler 实现和 4-mode 3-run median 结果。

---

## 1. Goal

Stage 6C-3B 在 `kv-idle-swap-resume` example 中引入 `resume_pending_token`，模拟上层 scheduler 在 active decode 中途判断 idle seq 即将 resume 的场景。

核心结论：

```text
Stage 6C-3B 在 kv-idle-swap-resume example 中引入 resume_pending_token，模拟上层 scheduler 在 active decode 中途判断 idle seq 即将 resume 的场景。实验表明：

1. pending=0 可复现 Stage 6C-2C baseline；
2. pending=96 时 active window 仍足够，remaining=0、fallback=0；
3. pending=124 时 active window 不足，系统执行 partial prefetch 并通过 fallback_blocks=3 优雅退化；
4. pending=-1 时不做 prediction、不 prefetch，RSS 保留更久，但 first-token latency 最高。
```

---

## 2. Background

Stage 6C-2B/2C 的 auto delayed prefetch 假设 seq0 从 active window 一开始就已知会 resume。
Stage 6C-3B 进一步模拟真实 scheduler：seq0 只有在 active decode 的某个 token 才被标记为 RESUME_PENDING。

6C-3A 审计结论：

```text
不进入 core scheduler；
不新增 public API；
不做 true async thread；
不引入 memory pressure；
policy 留在 example driver；
library 只提供 mechanism。
```

相关审计记录见 [Stage 6C-3A Scheduler-Level Resume-Aware Prefetch Policy Audit](kv_paged_read_stage6c3a_scheduler_policy_audit.md)。

---

## 3. Implementation Summary

只修改：

```text
examples/kv-idle-swap-resume/idle-swap-resume.cpp
```

新增 env：

```text
LLAMA_KV_PAGED_RESUME_PENDING_TOKEN
```

语义：

```text
>=0:
  seq1 active decode 到该 token 时，将 seq0 标记为 resume_pending。

0:
  active 一开始就标记，等价 Stage 6C-2C baseline。

<0:
  不标记 resume_pending；
  不 set protected；
  不执行 active prefetch；
  用作 no_resume_prediction 对照。
```

新增 telemetry：

```text
resume_pending_token
resume_pending_started
active_window_remaining
effective_start_token
resume_pending_window_ok
resume_pending_fallback_blocks
```

说明：

```text
本阶段未新增 public API，未修改 src/。
```

---

## 4. Formula

相对窗口公式：

```text
active_window_remaining = n_decode - resume_pending_token

need_steps = ceil(remaining_blocks / blocks_per_step)
need_span  = (need_steps - 1) * every_tokens + safety_tokens
```

若窗口足够：

```text
need_span <= active_window_remaining

resume_pending_window_ok = 1
effective_start_token = resume_pending_token + (active_window_remaining - need_span)
```

若窗口不足：

```text
need_span > active_window_remaining

resume_pending_window_ok = 0
effective_start_token = resume_pending_token
尽早 prefetch，并允许 fallback_blocks > 0
```

no prediction：

```text
resume_pending_token < 0

resume_pending_started = 0
active prefetch calls = 0
fallback_blocks = remaining_before_resume
```

---

## 5. Experiment Setup

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
3-run median
```

模式：

```text
A_baseline_pending0:
  LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=0

B_late_pending96:
  LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96

C_too_late_pending124:
  LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=124

D_no_prediction:
  LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=-1
```

---

## 6. Median Results

```text
mode                         first_ms  seq1_active_ms  total_wall_ms  tps      pending  pending_started  active_window  effective_start  window_ok  calls  blocks  ms_total  ms_max  remaining  fallback  auto_start  auto_remaining  auto_completed  rss_before_active  rss_after_active  active_rebound_mib  rss_before_resume  rss_after_resume
A_baseline_pending0          95.815    9892.888        26431.286      12.171   0        1                128            112              1          5      5       16.673    3.385   0          0         112         5               1               8375952            8395224           18.820              8395224            8366588
B_late_pending96             95.890    9787.972        26317.180      12.209   96       1                32             112              1          5      5       16.025    3.296   0          0         112         5               1               8375888            8395160           18.820              8395160            8367368
C_too_late_pending124        106.547   9789.952        26284.252      12.243   124      1                4              124              0          2      2       6.373     3.257   3          3         124         5               0               8375828            8383484           7.477               8383484            8367048
D_no_prediction              112.237    9724.882        26192.841      12.270   -1       0                0              0                1          0      0       0.000     0.000   5          5         0           0               0               0                  0                 NA                  8375672            8366932
```

说明：

```text
D_no_prediction 中 rss_before_active/rss_after_active 为 0，active_rebound_mib 为 NA，是因为没有 active prefetch，因此没有 active prefetch RSS rebound 统计点。
```

---

## 7. Interpretation

### A: baseline

```text
pending=0
active_window=128
effective_start=112
calls=5
blocks=5
remaining=0
fallback=0
```

解释：

```text
该模式复现 Stage 6C-2C 中 warmup=64、n=128 的 auto_safety0 baseline，说明 6C-3B 的 resume_pending 机制没有破坏原有 auto delayed prefetch 行为。
```

### B: late but enough window

```text
pending=96
active_window=32
effective_start=112
calls=5
blocks=5
remaining=0
fallback=0
```

解释：

```text
即使 seq0 到 active 中后段才被标记为 resume_pending，只要 remaining active window 足够覆盖 need_span，系统仍能按相同 effective_start=112 完成全部 5 blocks prefetch。
```

### C: too late with fallback

```text
pending=124
active_window=4
effective_start=124
window_ok=0
calls=2
blocks=2
remaining=3
fallback=3
first_ms=106.547
```

解释：

```text
当 resume_pending 到来过晚，active window 不足以恢复全部 5 blocks，系统会尽早执行 partial prefetch，恢复 2 blocks，并把剩余 3 blocks 通过 fallback 暴露给 resume first-token path。
```

强调：

```text
这不是失败，而是 Stage 6C-3B 需要验证的优雅退化路径。
```

### D: no prediction

```text
pending=-1
pending_started=0
calls=0
blocks=0
remaining=5
fallback=5
first_ms=112.237
rss_before_resume=8375672
```

解释：

```text
不标记 resume_pending 时，driver 不 set protected，也不 active prefetch。结果是 RSS 保留更久，rss_before_resume 最低，但 first-token latency 最高。
```

---

## 8. Main Findings

```text
1. Resume-aware timing works:
   pending=0 和 pending=96 都能计算到 effective_start=112，并完成全部 prefetch。

2. Fallback works:
   pending=124 时 window_ok=0，系统只恢复 2/5 blocks，fallback_blocks=3，first_ms 升高但低于 no-prediction。

3. No-prediction trade-off is quantified:
   pending=-1 时不 prefetch，rss_before_resume 最低，但 first_ms 最高。
```

用数值写清楚：

```text
B first_ms = 95.890 ms
C first_ms = 106.547 ms
D first_ms = 112.237 ms

B rss_before_resume = 8395160 KiB
C rss_before_resume = 8383484 KiB
D rss_before_resume = 8375672 KiB
```

解释：

```text
越早恢复，resume first-token latency 越低，但 resume 前 RSS 越高；
越晚或不恢复，RSS 保留更久，但 first-token latency 上升。
```

---

## 9. Limitations

```text
Stage 6C-3B 仍没有解决 prefetch 抵消净 RSS 收益的问题。
```

原因：

```text
A/B 中 remaining=0，说明 resume 前已恢复全部 blocks，因此 RSS 已回升。
C/D 虽然保留更多 RSS，但代价是 fallback_blocks>0 和 first-token latency 上升。
```

本阶段真正解决的是：

```text
1. 让 prefetch policy 对 resume_pending 时机敏感；
2. 支持窗口不足时 partial prefetch + fallback；
3. 量化 no-prediction 的 RSS/latency trade-off。
```

---

## 10. Next Direction

建议下一阶段：

```text
Stage 6C-3C:
  扩展 resume_pending token / safety / blocks_per_step 矩阵；
  或进入 Stage 6D memory-pressure-aware policy 设计。
```

但当前不建议直接进入：

```text
1. core scheduler；
2. true async prefetch；
3. memory-pressure-aware implementation；
4. public API redesign。
```

---

## 11. Summary

```text
Stage 6C-3B 在 example driver 中实现了 resume-aware pseudo-scheduler。通过 LLAMA_KV_PAGED_RESUME_PENDING_TOKEN，driver 可以模拟上层 scheduler 在 active decode 中途标记 seq0 即将 resume。4-mode 3-run median 结果显示：pending=0 与 pending=96 均能完成全部 prefetch，remaining=0、fallback=0；pending=124 因 active window 不足只恢复 2/5 blocks，并通过 fallback_blocks=3 优雅退化；pending=-1 不执行 active prefetch，RSS 保留更久但 first-token latency 最高。该阶段证明了 resume-aware timing、partial prefetch fallback 和 no-prediction trade-off 均可由 example-level pseudo-scheduler 表达。
```