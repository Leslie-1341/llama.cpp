# KV Paged Read — Stage 6C-2B Auto Delayed Prefetch Results

> 本文档记录 Stage 6C-2B（driver-level auto delayed prefetch policy）的实现、smoke 测试、3-run median 对照结果，以及对后续 Stage 6C-2C 的建议。

---

## 1. Background

Stage 6C-1 已完成 protected interleaved prefetch，证明可以把 KV swap-in 拆成 active 阶段的多次小恢复，降低 resume first-token 暴露的恢复成本。

Stage 6C-1B 中推荐的 fixed schedule 是：

```text
AFTER=112
EVERY=4
STEP=1
```

但 fixed D 是手写调参，不具备 workload 自适应能力。

Stage 6C-2B 的目标是把 fixed D 抽象成 driver-level auto delayed prefetch policy：

```text
1. active decode 前 probe seq0 remaining swapped blocks；
2. 根据 active_total_tokens / remaining_blocks / every / step / safety 自动计算 start token；
3. 复用 Stage 6C-1 的 prefetch_seq_step 和 protected gate；
4. 不新增 public API；
5. 不修改 src/；
6. 只在 example driver 中实现 policy。
```

---

## 2. Implementation Summary

只修改：

```text
examples/kv-idle-swap-resume/idle-swap-resume.cpp
```

新增 env：

```text
LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS
```

新增 telemetry：

```text
prefetch_auto_enabled
prefetch_auto_active_total_tokens
prefetch_auto_remaining_blocks
prefetch_auto_need_steps
prefetch_auto_start_token
prefetch_auto_every_tokens
prefetch_auto_blocks_per_step
prefetch_auto_safety_tokens
prefetch_auto_window_ok
prefetch_auto_started
prefetch_auto_completed
prefetch_auto_fallback_blocks
```

策略边界：

```text
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE 仍是 active-stage prefetch 总开关；
AUTO_DELAYED 只决定 start token 来源；
AUTO_DELAYED 未启用时，原手写 AFTER/EVERY/STEP 路径保持不变；
不新增 public API；
不进入 core scheduler；
不做 true async thread。
```

---

## 3. Auto Start Formula

```text
need_steps = ceil(remaining_blocks / blocks_per_step)

auto_start_token =
    active_total_tokens
  - (need_steps - 1) * every_tokens
  - safety_tokens
```

钳制规则：

```text
remaining_blocks == 0:
  不启动 active prefetch；
  auto_start_token = active_total_tokens；

auto_start_token < 0:
  auto_start_token = 0；
  window_ok = 0；

否则：
  window_ok = 1。
```

复现 fixed D：

```text
active_total_tokens = 128
remaining_blocks = 5
blocks_per_step = 1
every_tokens = 4
safety_tokens = 0

need_steps = 5
auto_start_token = 128 - (5 - 1) * 4 = 112
```

---

## 4. Smoke Result

single-run smoke：

```text
prefetch_auto_enabled=1
prefetch_auto_active_total_tokens=128
prefetch_auto_remaining_blocks=5
prefetch_auto_need_steps=5
prefetch_auto_start_token=112
prefetch_auto_every_tokens=4
prefetch_auto_blocks_per_step=1
prefetch_auto_safety_tokens=0
prefetch_auto_window_ok=1
prefetch_auto_started=1
prefetch_auto_completed=1
prefetch_auto_fallback_blocks=0
prefetch_during_active_calls=5
prefetch_during_active_blocks=5
prefetch_remaining_blocks_before_resume=0
seq0_resume_first_token_ms=99.277
prefetch_during_active_ms_max=3.438
rss_before_active_prefetch_kb=8375652
rss_after_active_prefetch_kb=8394924
```

解释：

```text
auto policy 正确计算 start_token=112，并在 active 阶段完成 5 次 partial prefetch，恢复 5 个 blocks，resume 前 remaining=0。
```

---

## 5. Minimal 3-run Matrix

配置：

```text
warmup=64
n=128
ctx=1024
parallel=2
f32 K/V
temp=0
3-run median
```

模式：

```text
fixed_D_after112_every4_step1
auto_safety0
auto_safety4
```

结果表（核心字段）：

```text
mode                         first_ms  seq1_active_ms  total_wall_ms  tps      calls  blocks  ms_total  ms_max  remaining  auto_enabled  auto_active_total  auto_remaining  auto_need_steps  auto_start  auto_every  auto_step  auto_safety  auto_window_ok  auto_started  auto_completed  auto_fallback  rss_before_active  rss_after_active  active_rebound_mib  rss_before_resume  rss_after_resume
auto_safety0                 98.664    9973.514        26927.806      11.937   5      5       16.526    3.353   0          1             128                5               5                112         4           1          0            1               1             1               0              8375864            8395136           18.820              8395136            8366844
auto_safety4                 97.308    9891.608        26934.632      11.923   6      5       16.466    3.322   0          1             128                5               5                108         4           1          4            1               1             1               0              8376132            8395404           18.820              8395404            8366972
fixed_D_after112_every4_step1 96.876    10184.536       27178.912      11.784   5      5       16.658    3.412   0          0             0                  0               0                112         4           1          0            0               0             0               0              8375844            8395116           18.820              8395116            8366872
```

---

## 6. Interpretation

```text
1. auto_safety0 自动计算 start_token=112，成功复现 fixed D；
2. auto_safety4 自动计算 start_token=108，提前一个 safety window；
3. 两个 auto 模式均 blocks=5、remaining=0、fallback=0；
4. ms_max 均小于 5 ms；
5. first-token latency 与 fixed D 接近；
6. auto delayed policy 可以替代手写 AFTER=112。
```

---

## 7. Recommended Default

推荐默认：

```text
auto_safety0
```

理由：

```text
1. 精确复现 fixed D 的最晚启动点；
2. calls=5，刚好恢复 5 个 blocks；
3. 没有多余 prefetch 调用；
4. tps 高于 fixed_D；
5. 更符合尽量晚恢复、延长低 RSS 窗口的目标。
```

说明：

```text
auto_safety4 是保守配置。如果后续 workload 更大或运行波动导致 remaining>0 / fallback_blocks>0，可使用 safety=4。
```

---

## 8. Limitations

必须写：

```text
Stage 6C-2B 仍不表示 resume 前保留了净 RSS drop。
```

原因：

```text
1. active_rebound_mib 仍约 18.820 MiB；
2. resume 前目标 blocks 已恢复完；
3. RSS 在 resume 前已经基本回升；
4. 当前收益是自动化 delayed prefetch 起点，减少手写参数，并尽量推迟 RSS 回升。
```

---

## 9. Summary

```text
Stage 6C-2B 将 Stage 6C-1B 的 fixed D schedule 抽象为 driver-level auto delayed prefetch policy。auto_safety0 能自动计算 start_token=112，复现 fixed D 行为；auto_safety4 能按 safety margin 计算 start_token=108。两个 auto 模式均完成 blocks=5、remaining=0、fallback=0，单步 prefetch 最大耗时低于 5 ms，first-token latency 与 fixed D 接近。该结果说明 auto delayed policy 可以替代手写 after/every/step，为后续 scheduler-level policy 和多 workload 自适应打基础。
```
