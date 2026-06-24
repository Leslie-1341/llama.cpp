# KV Paged Read — Stage 6C-3C Resume-Pending Boundary Sweep Results

> 本文档记录 Stage 6C-3C（resume-pending boundary sweep）的实验结果。
> 本阶段目标是在 Stage 6C-3B pseudo-scheduler 已经跑通的基础上，扫描 `resume_pending_token`，找出 complete prefetch 到 fallback 的边界点，并量化 fallback、first-token latency 与 RSS 之间的关系。

---

## 1. Goal

Stage 6C-3B 已经在 `kv-idle-swap-resume` example 中引入 `LLAMA_KV_PAGED_RESUME_PENDING_TOKEN`，用 driver 显式模拟上层 scheduler 在 active decode 中途判断 idle seq 即将 resume 的场景。

Stage 6C-3C 进一步固定 workload 和 prefetch 参数，只扫描 `resume_pending_token`，验证 resume-aware pseudo-scheduler 的边界行为。

核心结论：

```text
Stage 6C-3C 在 warmup=64、n=128、every=4、step=1、safety=0 的固定配置下扫描 resume_pending_token。

结果显示：
1. pending <= 112 时，active window 足够，系统可以完整恢复 5 个 blocks，remaining=0、fallback=0；
2. pending=116 开始出现 fallback；
3. pending=116/120/124 分别对应 fallback=1/2/3；
4. pending=-1 时不预测、不 active prefetch，fallback=5，first-token latency 最高，rss_before_resume 最低。
```

---

## 2. Background

Stage 6C-3B 已经验证 4-mode 行为：

```text
A pending=0:
  baseline，active window 从一开始就知道 seq0 即将 resume。

B pending=96:
  late but enough window，resume_pending 较晚出现，但 active window 仍足够完整恢复。

C pending=124:
  too late with fallback，resume_pending 太晚出现，只能 partial prefetch。

D pending=-1:
  no prediction，不标记 resume_pending，不执行 active prefetch。
```

Stage 6C-3C 的目的不是改变策略，而是进一步扫描 pending token，从 early 到 late，找出 complete prefetch 到 fallback 的临界点。

---

## 3. Experiment Setup

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
3-run median
```

扫描范围：

```text
resume_pending_token ∈ {-1, 0, 64, 96, 104, 108, 112, 116, 120, 124}
```

其中：

```text
pending=-1:
  no prediction baseline，不 active prefetch。

pending>=0:
  active decode 到该 token 时，将 seq0 标记为 resume_pending。
```

---

## 4. Boundary Formula

在本轮固定配置中：

```text
remaining_blocks = 5
blocks_per_step = 1
every_tokens = 4
safety_tokens = 0

need_steps = ceil(remaining_blocks / blocks_per_step)
           = ceil(5 / 1)
           = 5

need_span = (need_steps - 1) * every_tokens + safety_tokens
          = (5 - 1) * 4 + 0
          = 16 tokens
```

因此，完整恢复 5 个 blocks 至少需要 16 个 active tokens 的窗口。

```text
latest complete-prefetch pending token = n_decode - need_span
                                      = 128 - 16
                                      = 112
```

解释：

```text
pending <= 112:
  active window 至少有 16 tokens，可以完整恢复 5 blocks。

pending > 112:
  active window 不足，必然出现 fallback。
```

---

## 5. Median Results

结果表采用 3-run median。原始 terminal 输出中部分列因 tab/空格挤在一起显示不清，本文档采用 `summary.tsv` 与 `decision_hints` 对齐后的字段。

```text
mode                   pending  first_ms  seq1_active_ms  tps     pending_started  active_window  effective_start  window_ok  calls  blocks  ms_total  ms_max  remaining  fallback  auto_start  auto_remaining  auto_completed  rss_before_resume  rss_after_resume
pending_no_prediction  -1       113.875   9837.463        12.133  0                0              0                1          0      0       0.000     0.000   5          5         0           0               0               8375964            8366756
pending_000            0        95.861    9879.792        12.166  1                128            112              1          5      5       16.124    3.293   0          0         112         5               1               8395440            8367056
pending_064            64       97.291    9785.301        12.218  1                64             112              1          5      5       16.225    3.280   0          0         112         5               1               8395156            8367012
pending_096            96       97.739    9768.125        12.263  1                32             112              1          5      5       15.887    3.282   0          0         112         5               1               8394992            8367092
pending_104            104      97.373    9791.757        12.205  1                24             112              1          5      5       16.128    3.284   0          0         112         5               1               8395176            8366772
pending_108            108      95.222    9776.642        12.248  1                20             112              1          5      5       16.114    3.265   0          0         112         5               1               8395160            8366736
pending_112            112      95.852    9783.517        12.229  1                16             112              1          5      5       16.768    3.407   0          0         112         5               1               8395324            8366664
pending_116            116      97.540    9773.632        12.216  1                12             116              0          4      4       13.262    3.352   1          1         116         5               0               8391344            8367168
pending_120            120      102.331   9976.847        12.127  1                8              120              0          3      3       9.946     3.391   2          2         120         5               0               8387976            8367080
pending_124            124      106.249   9738.476        12.281  1                4              124              0          2      2       6.586     3.324   3          3         124         5               0               8383780            8366816
```

字段说明：

```text
pending:
  LLAMA_KV_PAGED_RESUME_PENDING_TOKEN。

active_window:
  n_decode - pending。

effective_start:
  实际开始 active prefetch 的 token。

window_ok:
  active window 是否足够覆盖 need_span。

calls / blocks:
  active decode 期间实际 prefetch 调用次数与恢复 blocks 数。

remaining / fallback:
  resume 前仍未恢复的 blocks 数。

rss_before_resume:
  进入 resume 前的 RSS。
```

---

## 6. Interpretation

### 6.1 Complete-prefetch region

完整恢复区间：

```text
pending ∈ {0, 64, 96, 104, 108, 112}
```

共同结果：

```text
effective_start=112
calls=5
blocks=5
remaining=0
fallback=0
window_ok=1
```

解释：

```text
只要 resume_pending 不晚于 token 112，active window 就足以恢复全部 5 blocks。
```

这说明 Stage 6C-3B 中的相对窗口公式在更密集的 pending-token sweep 中仍然成立。

---

### 6.2 Fallback boundary

第一个 fallback 点：

```text
pending=116
```

结果：

```text
active_window=12
effective_start=116
window_ok=0
calls=4
blocks=4
remaining=1
fallback=1
```

解释：

```text
pending=116 时，active window 只剩 12 tokens，小于 need_span=16。
因此系统无法完整恢复 5 blocks，只能恢复 4 blocks，剩余 1 block 进入 fallback。
```

这与公式推导一致：

```text
latest complete-prefetch pending token = 112
first fallback token = 116
```

---

### 6.3 Increasing fallback region

fallback 随 pending 变晚而增加：

```text
pending=116 -> fallback=1
pending=120 -> fallback=2
pending=124 -> fallback=3
```

对应 active prefetch blocks：

```text
pending=116 -> blocks=4
pending=120 -> blocks=3
pending=124 -> blocks=2
```

解释：

```text
pending 越晚，active window 越短；
active stage 能恢复的 blocks 越少；
resume 前剩余 blocks 越多；
fallback_blocks 随之增加。
```

这说明 fallback 不是异常行为，而是 active window 不足时的可解释退化路径。

---

### 6.4 No-prediction baseline

no-prediction 模式：

```text
pending=-1
calls=0
blocks=0
remaining=5
fallback=5
first_ms=113.875
rss_before_resume=8375964
```

解释：

```text
不做 resume prediction 时，系统完全不 active prefetch。
因此 resume 前 RSS 保留最多，但 resume first-token latency 最高。
```

它给出了当前 workload 下“不预测、不恢复”的 latency 上界和 RSS 下界。

---

## 7. Main Findings

### 7.1 Boundary is formula-consistent

本轮参数下：

```text
need_span=16
latest complete-prefetch pending token=112
```

实验结果显示：

```text
pending<=112:
  remaining=0
  fallback=0

pending=116:
  fallback=1
```

因此，边界与公式完全一致。

---

### 7.2 Fallback is monotonic after boundary

边界之后：

```text
pending=116 -> fallback=1
pending=120 -> fallback=2
pending=124 -> fallback=3
```

这说明 fallback blocks 与 active window 缩短之间具有稳定、可解释的关系。

---

### 7.3 Latency/RSS trade-off is visible

完整 prefetch 区间：

```text
pending=0~112:
  first_ms ≈ 95~98 ms
  rss_before_resume ≈ 8395xxx KiB
```

部分 prefetch 区间：

```text
pending=116:
  fallback=1
  first_ms=97.540 ms
  rss_before_resume=8391344 KiB

pending=120:
  fallback=2
  first_ms=102.331 ms
  rss_before_resume=8387976 KiB

pending=124:
  fallback=3
  first_ms=106.249 ms
  rss_before_resume=8383780 KiB
```

no-prediction：

```text
pending=-1:
  fallback=5
  first_ms=113.875 ms
  rss_before_resume=8375964 KiB
```

结论：

```text
prefetch 越完整：
  resume first-token latency 越低；
  resume 前 RSS 越高。

prefetch 越少：
  resume 前 RSS 越低；
  fallback 越多；
  first-token latency 越高。
```

---

## 8. What This Stage Solves

Stage 6C-3C 解决的是：

```text
1. 确定当前 workload 下 resume_pending 的最晚安全边界；
2. 验证 pending 超过边界后 fallback 会随窗口缩短递增；
3. 量化 fallback_blocks、first-token latency、rss_before_resume 三者之间的关系；
4. 为后续 memory-pressure-aware policy 提供依据。
```

更具体地说，在当前配置下：

```text
如果系统希望 zero fallback：
  resume_pending 必须不晚于 token 112；
  即至少提前 16 active tokens。

如果系统希望保留更多 RSS：
  可以故意推迟 prefetch；
  但需要接受 fallback 和 first-token latency 上升。
```

---

## 9. Limitations

Stage 6C-3C 仍没有解决 prefetch 抵消净 RSS 收益的问题。

原因：

```text
pending<=112 的完整恢复区域仍会让 RSS 回升；
pending>112 和 no-prediction 虽然保留更多 RSS，但代价是 fallback 和 first-token latency 上升。
```

本阶段不是最终 memory-saving policy，而是边界验证：

```text
1. 它证明了 resume-aware pseudo-scheduler 的时机公式成立；
2. 它证明了 fallback 行为可预测；
3. 它为后续“何时故意不完全 prefetch”提供了数据依据。
```

---

## 10. Next Direction

建议下一阶段进入：

```text
Stage 6D: memory-pressure-aware prefetch policy design
```

理由：

```text
Stage 6C-3C 已经明确了 RSS/latency trade-off：

完整 prefetch:
  降低 resume first-token latency；
  但会吃回 RSS。

partial / no prefetch:
  保留更多 RSS；
  但增加 fallback 和 first-token latency。
```

Stage 6D 可以开始研究：

```text
memory_pressure=low:
  允许完整 prefetch，优先降低 latency。

memory_pressure=medium:
  partial prefetch，在 latency 与 RSS 之间折中。

memory_pressure=high:
  skip 或 delay prefetch，优先保留 RSS。
```

但仍不建议立即进入：

```text
1. core scheduler；
2. true async prefetch；
3. public API redesign；
4. complex resume-probability prediction。
```

---

## 11. Summary

Stage 6C-3C 通过 pending-token sweep 找到了 resume-aware prefetch 的边界。在 `warmup=64`、`n=128`、`remaining=5`、`every=4`、`step=1`、`safety=0` 的条件下，`need_span=16`，因此 `pending=112` 是最后一个可以完整恢复 5 blocks 的 resume-pending token。

实验结果显示：

```text
pending<=112:
  remaining=0
  fallback=0

pending=116/120/124:
  fallback=1/2/3

pending=-1:
  不做 active prefetch
  fallback=5
  first-token latency 最高
  RSS 保留最多
```

该结果说明当前 pseudo-scheduler 的边界行为与公式一致，并为后续 memory-pressure-aware policy 提供了明确依据。
