# KV Paged Read — Stage 6E-A Real Memory Pressure Signal Design Audit

> 本文档记录 Stage 6E-A（real runtime memory pressure signal）的设计审计结论。
> 本阶段只做设计审计，不修改 `src/`、`include/`、public API，也不修改 examples 代码、不运行实验。
> 目标是在进入 Stage 6E-B 实现之前，确定“真实运行时内存压力信号 -> low / medium / high”的信号选择、阈值映射、env 设计与 fallback 边界。

---

## 1. Goal

Stage 6D 已经把 RSS / resume-latency 的 trade-off 变成了一个可解释、可触发的 policy，但 pressure level 仍由 env 手动指定：

```text
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE = off | low | medium | high
```

Stage 6E-A 的目标是回答：

```text
1. 有哪些候选的真实运行时 memory pressure 信号；
2. 哪个信号适合做 primary，哪个做 secondary，哪个只做 fallback / 诊断；
3. 为什么 PSI / mincore 不适合做 primary；
4. signal -> pressure_mode（low / medium / high）的初始阈值映射；
5. 接入信号需要新增哪些 env，如何默认保持 6D 行为不变；
6. 读取失败如何 fallback；
7. Stage 6E-B 的最小实现范围与边界。
```

本阶段只产出设计审计文档，不改代码、不跑实验。

---

## 2. Background from Stage 6D

Stage 6D 系列（参见 `kv_paged_read_stage6d_a_pressure_policy_audit.md`、`kv_paged_read_stage6d_c_pressure_policy_results.md`）已确立：

```text
1. pressure 只通过单一入口 target_restore_blocks 影响恢复量；
2. policy 留在 examples/kv-idle-swap-resume/idle-swap-resume.cpp，library 只提供 mechanism；
3. env 注入 pressure level，保证输入干净、确定、可复现；
4. 暂不接入真实信号，以避免“信号噪声”与“policy 决策”耦合。
```

Stage 6D-C 的 3-run median 已验证 policy 矩阵可解释、可预测：

```text
off / low : 完整恢复 5 blocks，fallback=0   (rss_before_resume 最高，first_ms 最低)
medium    : 主动恢复 3 blocks，fallback=2   (rss_before_resume 中等)
high      : 跳过 active prefetch，fallback=5 (rss_before_resume 最低，first_ms 最高)
```

Stage 6E-A 不改变这套 policy 行为，只替换“pressure level 从哪里来”：把手动 env 注入，扩展为可从真实运行时信号自动映射。`off` 仍作为显式禁用档保留。

---

## 3. Candidate Pressure Signals

本阶段评估以下五个候选信号：

```text
1. application-level memory budget   (应用自维护的内存预算)
2. cgroup memory.current / memory.max (cgroup v2 内存限额比值)
3. /proc/meminfo MemAvailable        (内核全机可用内存估算)
4. PSI memory.pressure               (cgroup / 全机 stall 统计)
5. mincore                           (某段映射的驻留情况)
```

评估维度：

```text
attribution  : 能否归因到本 workload / 本进程；
noise        : 信号自身噪声、是否与 prefetch 行为形成反馈回路；
portability  : 可移植范围；
cost         : 采集代价；
semantics    : 信号语义与“是否应该 prefetch”决策的契合度。
```

---

## 4. Signal-by-Signal Analysis

### 4.1 application-level memory budget

```text
来源       : driver / 应用自维护的配额 (used vs limit)
attribution: 强 — 完全归因到本进程 / 本 workload
noise      : 极低 — 自己记账，不受 page cache 扰动，无反馈回路
portability: 全平台 (纯应用态)
cost       : 近零 (内存读)
semantics  : 与 pressure_mode 直接对齐 (used_ratio -> 档位)
```

重要修正（语义边界）：

```text
application-level budget 是“人为设定的内存预算压力”，不是真实系统压力本身。
  优点: 可控、可复现、归因干净，与 6D 的 env 注入精神一致；
  缺点: 不能直接反映系统真实剩余内存——它衡量的是
        “本 workload 用了多少自己声明的预算”，而非“系统还剩多少”。
因此它适合做 reproducible policy evaluation 的 primary，
但不能被误读为系统级真实压力探针。
```

### 4.2 cgroup memory.current / memory.max

```text
来源       : cgroup v2 memory controller
attribution: 中-强 — 归因到本容器 / 本 cgroup
noise      : 低 — current/max 是确定限额比值，不是统计量
portability: Linux + cgroup v2 + 进程在受限 cgroup 内
cost       : 低 (读两个文件)
semantics  : used_ratio = current/max，直接映射档位
缺点       : 非容器化或 memory.max=max 时退化为无意义
```

适合做容器 / 部署环境下反映真实压力的 secondary：在受限 cgroup 内，它是最接近“本部署单元真实余量”的确定信号。

### 4.3 /proc/meminfo MemAvailable

```text
来源       : 内核全机估算
attribution: 弱 — 全机级，无法归因到本 seq / 本进程
noise      : 高 — 受 page cache 影响，prefetch 自身扰动 page cache -> 反馈回路
portability: 全 Linux
cost       : 低
semantics  : 全机余量，需额外换算成本进程压力
```

只适合做 fallback / 诊断参考：当 budget 与 cgroup 都不可用时，可作为最后的粗略余量估计，但不应作为 primary（反馈回路 + 全机噪声，难以稳定映射）。

### 4.4 PSI (memory.pressure)

```text
来源       : cgroup / 全机 stall 统计 (some/full, avg10/60/300)
attribution: 弱 — 全机 / 全 cgroup 滞后统计量
noise      : 高 — 时间窗滞后指标，与单次 prefetch 决策时间尺度不匹配
portability: Linux >= 4.20 且开启 PSI
cost       : 低
semantics  : 反映已发生的 stall，不是“当前是否该 prefetch”
```

只作为诊断 / 旁路参考，不作为 primary（理由见 §5）。

### 4.5 mincore

```text
来源       : 查询某段映射的驻留 page
attribution: 中 — 只反映该映射驻留情况，不反映系统压力
noise      : 中 — 反映“是否已在内存”，不是“是否该 prefetch”
portability: Linux / Unix-like available, but not a portable cross-platform policy signal
cost       : 中 — 需逐页扫描映射区间
semantics  : 语义错位 — 驻留率 != 系统压力
```

只作为诊断 / 旁路参考，不作为 primary（理由见 §5）。

---

## 5. Recommended Signal Source

```text
primary   (reproducible policy evaluation):
  application-level memory budget

secondary (real container / deployment pressure):
  cgroup memory.current / memory.max

fallback / diagnostic only:
  /proc/meminfo MemAvailable

not primary (diagnostic / side-channel only):
  PSI memory.pressure
  mincore
```

理由：

```text
1. 6D 系列的核心方法论是“干净归因”——signal 必须能归因到本 workload；
   application budget 与 cgroup 满足，全机信号不满足。

2. application budget 是确定、可复现、零反馈回路的输入，
   与 6D 用 env 注入 pressure 的“干净输入”精神一致，
   迁移到真实信号时不破坏 off 回归与可解释性；
   但它是人为预算压力，不是真实系统压力 (见 §4.1 修正)。

3. cgroup memory.current/memory.max 是确定比值而非统计量，
   在受限 cgroup 内能反映本部署单元的真实余量，适合做 secondary。

4. /proc/meminfo 全机、有反馈回路，仅在 budget / cgroup 都不可用时
   作为 fallback / 诊断。
```

为什么 PSI / mincore 不做 primary：

```text
PSI:
  1. 全机 / 全 cgroup 级滞后统计量，无法归因到单个 seq 的 prefetch 行为；
  2. avg10/60/300 的时间尺度与单次 resume prefetch 决策不匹配，
     决策时它仍是旧值；
  3. 与 prefetch 行为之间存在反馈滞后，破坏“干净归因”。

mincore:
  1. 语义是“某段映射是否驻留”，不是“系统是否有压力”，与 pressure_mode 语义错位；
  2. 仍需额外逻辑把驻留率翻译成决策，引入第二个变化变量；
  3. 逐页扫描有非平凡代价，且不反映系统层面余量。

共同点: 二者都把“信号噪声”与“policy 决策”耦合，
违背 Stage 6D-A §3 已确立的解耦原则，因此只作诊断 / 旁路参考。
```

---

## 6. Signal to Pressure Mode Mapping

以使用率 `used_ratio` 为统一输入：

```text
application budget : used_ratio = used / LLAMA_KV_PAGED_PREFETCH_MEM_LIMIT_BYTES
cgroup             : used_ratio = memory.current / memory.max
meminfo (fallback) : used_ratio = 1 - MemAvailable / MemTotal
```

初始阈值映射（可调，后续实验校准）：

```text
used_ratio < 0.70            -> low      (完整 prefetch，优先降 resume latency)
0.70 <= used_ratio < 0.85    -> medium   (cap=3，折中)
used_ratio >= 0.85           -> high     (skip active prefetch，优先保 RSS)
```

要点：

```text
1. 自动映射只产出 low / medium / high，复用 Stage 6D-B 既有 target_restore_blocks 规则，
   不改 timing 公式；

2. off 只作为显式禁用档，永远不由自动 signal 产生——
   off 是回归基线与显式停用 policy 的状态，必须保持手动可控；

3. 阈值留 hysteresis 余地（上调 / 下调可用不同边界以避免抖动），
   初版可先用上述单阈值，抖动问题留待 6E-B 实测后处理。
```

---

## 7. Environment Design

新增 / 复用 env：

```text
LLAMA_KV_PAGED_PREFETCH_PRESSURE_SOURCE = env | budget | cgroup | meminfo
  默认 env，完全保持 Stage 6D 行为。

LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE = off | low | medium | high
  env source 下沿用 6D 语义，由 env 显式指定，默认 off。

LLAMA_KV_PAGED_PREFETCH_MEM_LIMIT_BYTES = <bytes>
  budget source 的 limit；缺省 / 解析失败则 budget 不可用。

LLAMA_KV_PAGED_PREFETCH_PRESSURE_T_MED  = 0.70
  low -> medium 阈值。

LLAMA_KV_PAGED_PREFETCH_PRESSURE_T_HIGH = 0.85
  medium -> high 阈值。
```

默认行为保持 6D 不变：

```text
SOURCE 默认 = env；此时完全走 Stage 6D-B 路径，
PRESSURE_MODE 仍由 env 显式指定，off 字段级回归 Stage 6C-3C。
只有显式设 SOURCE=budget|cgroup|meminfo 才启用自动 signal -> mode 映射。
```

---

## 8. Failure and Fallback Behavior

```text
budget   : 未设置 MEM_LIMIT_BYTES 或解析失败            -> 回退 env source；
cgroup   : 文件不存在 / memory.max=max / 解析失败       -> 回退 env source；
meminfo  : 读取 / 解析失败                              -> 回退 env source。
```

统一原则：

```text
1. 任一自动映射失败都不报错、不中断 resume，只回退到 env source；
2. 回退后默认行为等价 Stage 6D，缺省 off——
   即“信号缺失 = 安全回到手动 / off”，不引入新风险；
3. fallback 是“失败即降级”，不向更激进档位偏移。
```

新增 telemetry 建议：

```text
pressure_source           : 实际生效的 source (env|budget|cgroup|meminfo)；
pressure_used_ratio       : 自动映射时计算出的 used_ratio (env source 下为 N/A)；
pressure_source_fallback  : 是否发生了 source 回退 (0/1)。
```

这三个字段让“信号读取是否成功、映射依据是什么、是否降级”可在结果中干净归因，延续 6D-C 的字段级自检思路。

---

## 9. Integration Boundary

```text
1. 只改 examples/kv-idle-swap-resume/idle-swap-resume.cpp —— 是。
   signal 采集 + signal -> mode 映射都属 policy，与 Stage 6D-A §4 一致。

2. 不改 src/ —— 是。
   core 无 request / slot / pressure 概念，不应把信号读取塞进 decode 关键路径。

3. 不改 include/ / 不新增 public API —— 是。
   现有 prefetch_seq_step / protected gate / last-stats mechanism 已足够，
   signal -> mode 纯属上层决策。

4. 不进 llama_decode core path —— 是。
   信号读取与映射在 driver 的 resume 决策点完成，不污染普通单请求路径。
```

---

## 10. Minimal Stage 6E-B Plan

只改动：

```text
examples/kv-idle-swap-resume/idle-swap-resume.cpp
```

最小改动清单：

```text
1. 新增 env LLAMA_KV_PAGED_PREFETCH_PRESSURE_SOURCE，解析为 enum，默认 env；

2. 实现 used_ratio 采集:
   budget  : used / MEM_LIMIT_BYTES；
   cgroup  : memory.current / memory.max；
   meminfo : 1 - MemAvailable / MemTotal；

3. 实现 used_ratio -> {low, medium, high} 映射 (T_MED / T_HIGH)，
   off 不由自动映射产生；

4. 实现 source fallback: 任一读取 / 解析失败 -> 回退 env source (缺省 off)；

5. 新增 telemetry: pressure_source、pressure_used_ratio、pressure_source_fallback；

6. 复用 Stage 6D-B 的 target_restore_blocks 规则与 timing 公式，不改 mechanism；

7. 验证顺序: 先 SOURCE=env 回归 6D（逐字段比对），
   再分别验证 budget / cgroup / meminfo 在已知 used_ratio 下命中预期档位。
```

---

## 11. Risks and Non-Goals

风险：

```text
1. budget 被误读为系统真实压力 (最高优先)：
   budget 是人为预算，不反映系统真实余量。
   缓解: 文档显式声明 (§4.1)，telemetry 记 pressure_source 以区分。

2. 阈值抖动：
   used_ratio 在阈值附近波动会导致 mode 频繁切换。
   缓解: 预留 hysteresis，初版单阈值，实测后引入双边界。

3. meminfo 反馈回路：
   prefetch 扰动 page cache 影响 MemAvailable。
   缓解: meminfo 仅作 fallback / 诊断，不做 primary。

4. source 实现分叉破坏 6D 回归：
   缓解: SOURCE=env 默认路径与 6D-B 字段级一致，作实现自检。
```

非目标（Non-Goals）：

```text
1. 不做 true async prefetch thread；
2. 不做 production memory scheduler；
3. 不做 resume probability prediction；
4. 不承诺解决 resume 后净 RSS 被重新吃回的问题
   (Stage 6C-3C / 6D-C 的未解决问题在本阶段依旧存在)；
5. 不进入 core scheduler、不新增 public API。
```

---

## 12. Summary

```text
Stage 6E-A 审计确定了“真实运行时 memory pressure 信号 -> pressure_mode”的设计：

primary   (reproducible policy evaluation): application-level memory budget；
secondary (real container / deployment)   : cgroup memory.current / memory.max；
fallback / diagnostic only                : /proc/meminfo MemAvailable；
not primary (diagnostic / side-channel)   : PSI、mincore。

关键修正: application budget 是人为设定的预算压力，优点是可控、可复现、归因干净，
缺点是不直接反映系统真实剩余内存；PSI / mincore 因不可归因或语义错位只作旁路参考；
mincore 仅 Linux / Unix-like available, 不是可移植的跨平台 policy 信号。

映射: used_ratio < 0.70 -> low；0.70~0.85 -> medium；>= 0.85 -> high；
off 只作显式禁用档，不由自动信号产生。

接入通过新增 LLAMA_KV_PAGED_PREFETCH_PRESSURE_SOURCE=env|budget|cgroup|meminfo
（默认 env，完全保持 6D 行为）与阈值 env 实现，任一读取失败即回退 env / 缺省 off，
并新增 pressure_source / pressure_used_ratio / pressure_source_fallback telemetry。

Stage 6E-B 只改 examples/kv-idle-swap-resume/idle-swap-resume.cpp，
不改 src/、include/，不新增 public API，不进 llama_decode core path，
不做 async prefetch / production scheduler / resume probability prediction，
也不解决净 RSS 收益被 resume 后续恢复吃回的问题。
```
