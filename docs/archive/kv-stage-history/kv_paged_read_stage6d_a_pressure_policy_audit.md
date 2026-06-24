# KV Paged Read — Stage 6D-A Memory-Pressure-Aware Prefetch Policy Design Audit

> 本文档记录 Stage 6D-A（memory-pressure-aware prefetch policy）的设计审计结论。
> 本阶段只做设计审计，不修改 `src/`、`include/`、public API，也不修改 examples 代码、不运行实验。
> 目标是在进入 Stage 6D-B 实现之前，确定 pressure-aware policy 的边界、env 设计、target_restore_blocks 规则与实验矩阵。

---

## 1. 目标

Stage 6C-3C 已经量化了 resume-aware prefetch 的 RSS / first-token latency trade-off：

```text
完整 prefetch:
  降低 resume first-token latency；
  但会吃回 RSS。

partial / no prefetch:
  保留更多 RSS；
  但增加 fallback 和 first-token latency。
```

Stage 6D 的目标是把这条 trade-off 曲线变成一个**可解释、可触发**的 policy：

```text
当系统内存压力低时，可以积极 prefetch，优先降低 resume latency；
当系统内存压力高时，应该限制或跳过 prefetch，优先保留 RSS；
让系统在 RSS 与 resume latency 之间做可解释的取舍。
```

Stage 6D-A 是其中的设计审计阶段，回答：

```text
1. 本阶段是否先用 env 模拟 pressure；
2. 是否读取真实 memory pressure 信号；
3. 推荐的 env 名称与语义；
4. 推荐的 low / medium / high policy；
5. 是否需要新增 public API；
6. 是否需要修改 src/；
7. 推荐的最小实验矩阵；
8. 风险与不建议做的方案；
9. 下一步实现建议。
```

---

## 2. 为什么本阶段先用 env 模拟 pressure

结论：

```text
Stage 6D-A / 6D-B 先用 env 注入 pressure level，policy 继续留在
examples/kv-idle-swap-resume/idle-swap-resume.cpp。
```

理由：

```text
1. 本阶段要验证的是 policy 行为是否可解释，而不是 pressure 信号采集本身；
2. env 注入让 pressure level 成为一个干净、确定、可复现的输入；
3. 这与 Stage 6C 系列保持一致：library 只提供 mechanism，policy 留在 driver；
4. 先解耦“信号”与“决策”，可以避免一开始就引入两个同时变化的变量。
```

---

## 3. 为什么暂不读取真实 memory pressure 信号

结论：

```text
Stage 6D 暂不接入 /proc/meminfo、cgroup memory.pressure (PSI)、mincore 等真实信号。
```

理由（归因陷阱）：

```text
1. PSI (cgroup memory.pressure) 是全机 / 全 cgroup 级别的信号，
   无法归因到单个 seq 的 prefetch 行为；

2. /proc/meminfo 的 MemAvailable 受 page cache 影响，
   而 prefetch 自身会扰动 page cache，形成反馈回路；

3. mincore 只能反映某段映射的驻留情况，
   仍需要额外逻辑把它翻译成“是否应该 prefetch”的决策；

4. Stage 6C-3C §9 已经表明：完整 prefetch 区间净 RSS 收益本就接近 0；
   叠加真实信号噪声后，pressure mode 之间的差异几乎不可读；

5. 如果现在接入真实信号，会把“信号噪声”与“policy 决策”耦合在一起，
   导致实验结果难以归因。
```

处理方式：

```text
真实信号采集与“信号 -> pressure mode”的阈值映射，
留到 Stage 6D 之后的独立阶段单独处理，与 policy 验证解耦。
```

---

## 4. 为什么 policy 继续留在 example，而不进入 src/ 或 public API

结论：

```text
Stage 6D-A/6D-B 只改 examples/kv-idle-swap-resume/idle-swap-resume.cpp。
不修改 src/、include/，不新增 public API。
```

延续 Stage 6C-3A 的源码审计结论：

```text
1. src/ 中没有 request-level scheduler；
2. core 的调度粒度是 llama_decode 的 batch / ubatch；
3. core 中没有 request / slot / idle queue 的概念；
4. core 无法知道哪个 idle seq 即将 resume，更无法知道系统内存压力；
5. idle->resume-pending 信号、memory-pressure 信号都只能由上层 driver / server / 应用提供；
6. 如果把 pressure policy 塞进 decode 关键路径（如 set_input_paged_row_idx），
   会污染普通单请求路径，迫使其承担额外分支与扫描成本。
```

现有 mechanism 已足够支撑 pressure policy：

```text
llama_memory_prefetch_seq_step(mem, seq, 0)
  probe remaining blocks。

llama_memory_prefetch_seq_step(mem, seq, k)
  incremental prefetch。

llama_kv_cache_set_seq_prefetch_protected(...)
  protected gate。

llama_kv_cache_prefetch_seq_last_stats(...)
  telemetry / remaining blocks。
```

`target_restore_blocks` 只是 driver 在调用 step 前，对“本次 resume 总恢复量”设置的上限，属于纯 policy，不需要进 library。

---

## 5. 推荐 env：名称与语义

推荐采用 mode 语义而非数字 level：

```text
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE = off | low | medium | high
默认值: off   (完全保持 Stage 6C-3C 行为)
```

语义：

```text
off:
  不启用 pressure-aware policy；
  逐字段回归 Stage 6C-3C 行为；
  用作回归基线与实现自检。

low:
  低内存压力；
  完整 prefetch，优先降低 resume first-token latency。

medium:
  中等内存压力；
  partial prefetch，只恢复一部分 blocks；
  在 latency 与 RSS 之间折中。

high:
  高内存压力；
  skip active prefetch，优先保留 RSS。
```

为什么选 mode 而不是 `LEVEL=0|1|2`：

```text
1. off 是数字方案无法自然表达的档位；
   而回归基线必须有一个显式的“禁用 policy”状态；

2. low / medium / high 自带语义，实验表和文档可直接阅读，
   无需把 0 / 1 / 2 再翻译成压力等级；

3. 未来接入真实信号时，“信号 -> low/medium/high”比“信号 -> 0/1/2”更自然，
   low/medium/high 是内存压力分级的通用表达；

4. 数字方案唯一优势是“可排序”，但 mode 本身已隐含顺序，
   不值得为此牺牲可读性。
```

---

## 6. 推荐 target_restore_blocks 规则

在现有公式之前插入一个 `target_restore_blocks` 上限，**不改变 timing 公式**：

```text
off    : target_restore_blocks = remaining_blocks
low    : target_restore_blocks = remaining_blocks
medium : target_restore_blocks = min(remaining_blocks, 3)
high   : target_restore_blocks = 0
```

四档恰好张成 Stage 6C-3C 已验证的 trade-off 曲线：

```text
off / low : 完整恢复       -> fallback=0,        rss_before_resume 最高
medium    : partial 恢复    -> fallback 中等,      rss_before_resume 中等
high      : skip            -> fallback=remaining, rss_before_resume 最低
```

### 6.1 为什么 medium 采用固定 cap=3，而不是比例恢复

```text
1. 比例恢复（例如 ceil(remaining/2)）会让 target 随 workload 漂移，
   跨配置不可比；

2. 固定 cap 让“medium 恢复 N blocks”成为一个可直接对照的常量；

3. 本轮 remaining_blocks=5，cap=3 正好落在完整(5)与 skip(0)之间，
   且对齐 Stage 6C-3C 已知的 fallback 阶梯
   （pending=120 对应 fallback=2，恢复 3 blocks）；

4. 固定 cap 也更容易做交叉验证：medium 行应接近 6C-3C 中
   恢复 3 blocks 的那一行，只是触发原因不同（pressure 而非 window 不足）。
```

### 6.2 为什么 high 采用 skip，而不是 delay

```text
1. 在固定 workload 且 active window 充足时，delay 最终仍会恢复全部 blocks，
   RSS 收益归零，无法与 low 区分；

2. skip 才能产生一个干净的“保留 RSS、fallback=remaining”的上界点；

3. high=skip 等价于 6C-3C 的 pending=-1 行为，
   但由 pressure mode 显式触发，而非因缺失 resume prediction；

4. delay 引入了额外的时间维度，会与 resume-aware timing 混在一起，
   增加归因难度（见 §9 风险）。
```

---

## 7. 关键公式（pressure-aware 版本）

pressure 只通过 `target_restore_blocks` 这一个入口影响 timing，公式其余部分与 Stage 6C-3C 逐字相同：

```text
target_blocks = clamp_by_mode(remaining_blocks, pressure_mode)
need_steps    = ceil(target_blocks / blocks_per_step)
need_span     = (need_steps - 1) * every_tokens + safety_tokens
```

合理性：

```text
1. off 时 target_blocks = remaining_blocks，公式与 6C-3C 完全一致，必然回归；

2. high 时 target_blocks = 0 -> need_steps = 0，
   需在代码里短路成“不 active prefetch”分支
   （对齐现有 remaining==0 的处理路径）；

3. fallback_blocks = remaining_blocks - actually_restored_blocks 自动成立；

4. pressure 不触碰 effective_start_token 的窗口判定逻辑，
   因此 timing 决策与 pressure 决策彼此独立、各自可解释。
```

---

## 8. 推荐 Stage 6D-B 最小实现范围

只改动：

```text
examples/kv-idle-swap-resume/idle-swap-resume.cpp
```

最小改动清单：

```text
1. 新增 env LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE，解析为 enum，默认 off；

2. 在现有 auto-start 公式之前插入：
   target_blocks = clamp_by_mode(remaining_blocks, pressure_mode)；
   high 短路到 no-active-prefetch 分支；

3. active decode loop 中，prefetch step 累计到 target_blocks 后停止；

4. 新增 telemetry 字段：
   pressure_mode
   target_restore_blocks；

5. 保持 prefetch_seq_step / protected gate / last-stats API 调用不变；

6. 先跑 off 回归 6C-3C（逐字段比对），再跑 low / medium / high。
```

不做：

```text
1. 不修改 src/ 或 include/；
2. 不新增 public API；
3. 不接入真实 memory pressure 信号；
4. 不引入 true async prefetch thread。
```

---

## 9. 推荐 Stage 6D-C 最小实验矩阵

固定配置（沿用 6C-3C，并固定 pending_token）：

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
pending_token=96        (单一固定值，active window 充足)
3-run median
```

变量：

```text
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE ∈ {off, low, medium, high}
```

为什么固定 pending=96、只扫 mode：

```text
1. pending=96 时 active_window = n - pending = 32 >= need_span，window 充足；
2. 因此任何 fallback 都来自 pressure policy 主动限流，而非 window 不足；
3. 这把 pressure 决策与 timing 决策彻底解耦，结果可干净归因；
4. off 与 low 的核心行为指标应一致；但 pressure_mode 字段本身不同。可用作实现自检与旁路验证。
```

核心指标（沿用 6C-3C 字段 + 新增）：

```text
pressure_mode
target_restore_blocks
prefetch_during_active_blocks
prefetch_remaining_blocks_before_resume
fallback_blocks
first_ms
rss_before_resume
rss_after_resume
tokens_per_second
```

预期（基于 6C-3C §6.3 实测外推，最终以实验为准）：

```text
off / low:
  target=5  blocks=5  fallback=0  first_ms 最低   rss_before_resume 最高

medium:
  target=3  blocks=3  fallback=2  first_ms 中等   rss_before_resume 中等

high:
  target=0  blocks=0  fallback=5  first_ms 最高   rss_before_resume 最低
```

交叉验证锚点：

```text
medium 行应接近 6C-3C 的 pending=120 行（恢复 3 blocks、fallback=2）；
high   行应接近 6C-3C 的 pending=-1 行（恢复 0 blocks、fallback=5）；
只是触发原因从“window 不足”变成“pressure 限流”。
```

---

## 10. 风险与限制

### 10.1 风险

```text
1. 归因混淆（最高优先）：
   若同时变 pending 和 mode，无法区分 fallback 来自 window 不足还是 pressure 限流。
   缓解：实验矩阵固定 pending=96，只扫 mode。

2. off 与 low 实现分叉：
   若二者走不同分支，会引入偏差。
   缓解：把二者做成字段级一致，当作实现自检。

3. medium 采用比例会跨配置不可比：
   缓解：固定 cap=3。

4. high 采用 delay 在足够 window 下 RSS 收益归零、与 low 不可区分：
   缓解：high=skip。

5. 净 RSS 收益仍接近 0：
   Stage 6C-3C §9 的未解决问题在本阶段依旧存在。
```

### 10.2 限制（必须明确说明）

```text
本阶段验证的是 pressure-aware policy 的“可解释性”，
即：pressure mode 是否能可预测地、单调地改变 target_restore_blocks、
fallback、first-token latency、rss_before_resume。

本阶段不承诺解决“净 RSS 收益被 resume prefetch 吃回”的问题。

medium / high 之间 rss_before_resume 的差异反映的是
“在 resume 前少恢复了多少 blocks”，
不应被误读为系统层面节省了等量内存。
```

### 10.3 不建议做的方案

```text
1. 不接入真实 /proc/meminfo / cgroup PSI / mincore（本阶段）；
2. 不进入 core scheduler；
3. 不新增 public API；
4. 不做 true async prefetch thread；
5. medium 不用比例恢复，用固定 cap=3；
6. high 不用 delay，用 skip；
7. 不做 resume-probability 概率预测。
```

---

## 11. 下一步实现建议

```text
1. 本阶段（6D-A）只产出设计审计文档，不改代码、不跑实验；

2. Stage 6D-B：在 idle-swap-resume.cpp 中实现最小 pressure policy
   （env + target_restore_blocks + telemetry），先 off 回归 6C-3C；

3. Stage 6D-C：按 §9 矩阵固定 pending=96，扫描 off/low/medium/high，
   产出 kv_paged_read_stage6d_c_pressure_policy_results.md；

4. 结果文档需显式声明本阶段验证的是 policy 可解释性，而非净 RSS 节省；

5. 真实 memory pressure 信号接入留到 6D 之后独立阶段。
```

---

## 12. Summary

```text
Stage 6D-A 审计表明，memory-pressure-aware prefetch policy 应继续采用
example-level pseudo-scheduler 方案：用 env LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE
（off|low|medium|high）注入 pressure level，policy 留在
examples/kv-idle-swap-resume/idle-swap-resume.cpp，library 继续只提供 mechanism。

pressure 通过单一入口 target_restore_blocks 影响恢复量：
off/low 完整恢复、medium 固定 cap=3、high skip。
该设计复用 Stage 6C-3C 的 timing 公式，off 必然逐字段回归，
medium/high 分别对齐 6C-3C 中恢复 3 blocks 与 0 blocks 的已知行，提供交叉验证锚点。

本阶段不接入真实 memory pressure 信号，不修改 src/、include/、public API，
不新增 API，也不解决净 RSS 收益被 resume prefetch 吃回的问题——
它验证的是 pressure-aware policy 在 RSS 与 resume latency 之间取舍的可解释性。
```
