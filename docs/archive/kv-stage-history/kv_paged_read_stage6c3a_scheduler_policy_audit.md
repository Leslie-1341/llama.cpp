# KV Paged Read — Stage 6C-3A Scheduler-Level Resume-Aware Prefetch Policy Audit

## 目标
记录 Stage 6C-3A 的设计审计结论，明确后续 Stage 6C-3B 的实现边界。

核心结论：

```
Stage 6C-3 不建议直接进入 core scheduler。
当前 llama.cpp core 缺少 request-level scheduler 和 idle→resume-pending 信号源。
因此下一步应先在 kv-idle-swap-resume example 中实现 pseudo-scheduler，模拟 resume-aware prefetch policy。
```

## 背景（此前阶段回顾）

Stage 6C-1:
```
  完成 protected interleaved prefetch；
  解决刚 prefetch 回来的 blocks 被 idle swap-out 重新换出的问题；
  证明可以把 KV swap-in 成本拆成 active 阶段多次小恢复。
```

Stage 6C-2B:
```
  完成 driver-level auto delayed prefetch；
  用 remaining_blocks、active_total_tokens、every、step、safety 自动计算 start token；
  auto_safety0 可复现 fixed D 的 start_token=112。
```

Stage 6C-2C:
```
  完成 scaling matrix；
  warmup=32/64/128、n=128/256 下均满足 start_ok=True、remaining=0、fallback=0、ms_max<5 ms。
```

当前问题：
```
6C-2B/2C 仍然只是 example driver 中的确定性策略：
seq0 一定会 resume；
active window 已知；
remaining blocks 可提前 probe；
因此还不是通用 scheduler-level policy。
```

## 为什么不直接进入 core scheduler（源码审计结论）

```
1. src/ 中没有 request-level scheduler；
2. core 的调度粒度是 llama_decode 的 batch / ubatch；
3. core 中没有 request、slot、idle queue 的概念；
4. tools/server 中虽然有 slot / queue，但它与 KV cache mechanism 解耦；
5. core 无法知道哪个 idle seq 即将 resume；
6. idle→resume-pending 信号只能由上层 driver / server / 应用提供；
7. 如果把 policy 塞进 set_input_paged_row_idx，会污染每次 decode 的关键路径；
8. 普通单请求路径也会被迫承担额外分支和扫描成本。
```

结论：

```
Stage 6C-3 不应直接修改 core scheduler。
policy 继续留在 example driver 中验证；
library 继续只提供 mechanism。
```

## 推荐状态机（采用 5 态）

状态：

```
ACTIVE:
  正在 decode 的 seq。

IDLE_RESIDENT:
  idle，但 KV 仍 resident。

IDLE_SWAPPED:
  idle，且部分/全部 KV blocks 已 swapped。

RESUME_PENDING:
  driver 已标记该 seq 即将 resume；
  set protected=true；
  probe remaining_blocks；
  根据 active_window_remaining 启动 incremental prefetch。

RESUMED:
  已进入 resume decode。
```

转移：

```
ACTIVE -> IDLE_RESIDENT
  decode 切到其他 seq。

IDLE_RESIDENT -> IDLE_SWAPPED
  idle gate swap-out 命中。

IDLE_RESIDENT / IDLE_SWAPPED -> RESUME_PENDING
  driver mark resume_pending。

RESUME_PENDING -> RESUMED
  进入 first-token resume path。

RESUMED -> ACTIVE
  resume 完成，清 protected。
```

说明：

```
PREFETCHING 不作为独立状态，因为它是 RESUME_PENDING 的行为。
READY_TO_RESUME 不作为独立状态，因为它等价于 RESUME_PENDING && remaining_blocks==0。
```

## 最小 policy 输入（Stage 6C-3B 需求）

最小输入：

```
remaining_blocks
active_window_remaining
every_tokens
blocks_per_step
safety_tokens
```

其中：

```
active_window_remaining = n_decode - resume_pending_token
```

区别于 Stage 6C-2B：

```
6C-2B 使用 active_total_tokens；
6C-3B 中 resume_pending 可能在 active 中后段才出现，因此应使用 active_window_remaining。
```

公式：

```
need_steps = ceil(remaining_blocks / blocks_per_step)
need_span  = (need_steps - 1) * every_tokens + safety_tokens
```

若：

```
need_span <= active_window_remaining
```
则：

```
window_ok = 1
effective_start_token = resume_pending_token + (active_window_remaining - need_span)
```
否则：

```
window_ok = 0
effective_start_token = resume_pending_token
尽早 prefetch，并允许 fallback_blocks > 0
```

## Stage 6C-3B 最小策略（推荐先做 example-level pseudo-scheduler）

流程：

```
1. seq0 idle 后被 swapped；
2. driver 在 seq1 active decode 的某个 token 标记 seq0 为 RESUME_PENDING；
3. 到达 resume_pending_token 时：
   - set_seq_prefetch_protected(seq0, true)
   - probe remaining_blocks
   - active_window_remaining = n_decode - resume_pending_token
   - 计算 effective_start_token
4. 后续 active decode 中执行 incremental prefetch；
5. resume 前再次 probe：
   - remaining=0：ready
   - remaining>0：记录 fallback_blocks，first-token path 承担剩余 swap-in
6. resume 完成后清 protected。
```

## 是否需要新增 API

结论：

```
Stage 6C-3B 不需要新增 public API。
```

现有 API 已足够：

```
llama_memory_prefetch_seq_step(mem, seq, 0)
  probe remaining blocks。

llama_memory_prefetch_seq_step(mem, seq, k)
  incremental prefetch。

llama_kv_cache_set_seq_prefetch_protected(...)
  protected gate。

llama_kv_cache_prefetch_seq_last_stats(...)
  telemetry / remaining blocks。
```

原则：

```
library 只提供 mechanism；
policy 留在 driver / pseudo-scheduler；
优先不新增 public API。
```

## 是否引入 memory pressure

结论：

```
Stage 6C-3 不引入 memory pressure。
```

理由：

```
1. 当前阶段目标是验证 resume-aware timing 和 fallback；
2. memory pressure 需要独立信号源，例如 /proc/meminfo、cgroup、mincore；
3. 如果现在加入 pressure，会和“prefetch 抵消净 RSS 收益”问题耦合，难以归因；
4. memory-pressure-aware policy 留到 Stage 6D。
```

未来方向（预留）：

```
LLAMA_KV_PAGED_PREFETCH_MEMORY_PRESSURE_MODE:
  off
  delay
  skip
```

## Stage 6C-3B 最小实验矩阵

固定配置：

```
warmup=64
n=128
ctx=1024
parallel=2
f32 K/V
temp=0
3-run median
```

模式：

```
mode A: auto_delayed_baseline
  resume_pending_token=0；
  等价 Stage 6C-2C baseline；
  window=full n。

mode B: resume_pending_late
  seq0 在 active 中后段才标记 resume_pending；
  active window 仍足够；
  预期 window_ok=1、remaining=0、fallback=0。

mode C: resume_pending_too_late
  seq0 很晚才标记 resume_pending；
  active window 不足；
  预期 window_ok=0、fallback_blocks>0、first-ms 升高。

mode D: no_resume_prediction
  不标记 resume_pending；
  不 prefetch；
  预期 RSS 保留更久，但 first-token latency 最高。
```

核心指标：

```
resume_pending_token
active_window_remaining
auto_start_token
effective_start_token
remaining_blocks
window_ok
fallback_blocks
first_ms
ms_max
seq1_active_ms
rss_before_active_prefetch
rss_after_active_prefetch
rss_before_resume
rss_after_resume
```

成败判据：

```
A / B:
  window_ok=1
  remaining=0
  fallback=0
  ms_max<5

C:
  window_ok=0
  fallback>0
  first_ms 介于 B 与 D 之间

D:
  first_ms 最高
  rss_before_resume 最低
```

## 风险与不建议做的方案

不建议：

```
1. 不直接进 core scheduler；
2. 不新增 public API；
3. 不做 true async prefetch thread；
4. 不引入 memory pressure；
5. 不做复杂 resume_probability 预测；
6. 不把 policy 写进 set_input_paged_row_idx。
```

风险：

```
1. 如果忘记在第一次 prefetch 前 set protected=true，prefetched blocks 会被 idle gate 重新换出；
2. 如果 mode C 没有明确 fallback telemetry，容易误判成成功；
3. 如果 policy 放在 decode 关键路径，会污染普通单请求；
4. 即使 Stage 6C-3 成功，remaining=0 时净 RSS drop 仍会接近 0。
```

## 下一步实现建议（Stage 6C-3B 最小改动）

只改动：

```
examples/kv-idle-swap-resume/idle-swap-resume.cpp
```

新增 env：

```
LLAMA_KV_PAGED_RESUME_PENDING_TOKEN
```

默认：

```
0
```

含义：

```
seq1 active decode 到第几个 token 时，将 seq0 标记为 resume_pending。
0 表示从 active 一开始就已知会 resume，等价 Stage 6C-2C baseline。
```

新增 telemetry：

```
resume_pending_token
active_window_remaining
effective_start_token
resume_pending_window_ok
resume_pending_started
resume_pending_fallback_blocks
```

实现路线：

```
1. 加 env；
2. 把 protected=true 和 probe remaining 从 active loop 前移到 resume_pending_token 时刻；
3. 把 auto_start 公式从 active_total_tokens 改为 active_window_remaining；
4. 保持 prefetch_seq_step / protected gate / last stats API 不变；
5. 先用 mode A 回归 6C-2C；
6. 再跑 mode B/C/D。
```

## Summary

```
Stage 6C-3A 审计表明，目前不适合将 resume-aware prefetch policy 直接放入 llama.cpp core。core 缺少 request-level scheduler 和 idle→resume-pending 信号源，policy 放入图输入构建路径会污染普通 decode 关键路径。因此 Stage 6C-3B 应继续采用 example-level pseudo-scheduler：由 driver 显式注入 resume_pending_token，用 active_window_remaining 计算 effective_start_token，并验证 normal / late / too-late / no-prediction 四类场景。该阶段目标是验证 resume-aware prefetch 的状态机、时机决策和 fallback 行为，而不是解决最终净 RSS drop 保留问题。
```
