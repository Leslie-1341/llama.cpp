# KV Paged Read — Stage 7B Idle Coverage Scaling Workload Plan

> 本文档设计 Stage 7B 的 workload 方案：通过扩大 idle-owned KV coverage，
> 让 idle KV swap/madvise 产生更明显、更可信的 current RSS drop。
> 所有指标与结果表沿用 Stage 7A 统一面板
> （docs/kv_paged_read_stage7a_metric_panel_and_eval_plan.md）。

---

## 1. Goal

```text
1. 扩大 idle-owned KV coverage（被判定为 idle 且可释放的 KV 占比）；
2. 验证 release_ratio_total_pct / rss_drop_mib 是否随 idle_coverage_pct 增加而 scaling；
3. 用 Stage 7A 统一面板同时报告 RSS 收益与 resume latency 代价；
4. 明确区分 controlled workload（机制验证 microbenchmark）与
   semi-real workload（自然 idle/resume），不把前者伪装成真实业务负载收益。
```

核心问题（贯穿全文）：

```text
在 llama.cpp 连续 KV cache 之上，如何把更多 KV 推入 idle 状态并释放，
使 current RSS 下降更明显、更可信，同时让 resume first-token latency 代价可量化、可解释。
```

---

## 2. Why Idle Coverage Is the Main Scaling Lever

```text
1. 只有 idle-owned KV 才是可释放对象:
   active 序列的 KV 不能 swap/madvise（会破坏 correctness 或抬高 resume 代价）；
   因此可释放体量的上界 = idle_coverage。

2. RSS drop 受 idle coverage 上界约束:
   rss_drop_mib 不可能超过 (idle KV 常驻物理页量 × madvise 落地率)；
   提升 idle coverage 等价于抬高这条上界。

3. 低 idle coverage 时 RSS drop 被权重稀释:
   在低并发/中短上下文场景，KV 占总 RSS 比例低，
   即便释放全部 idle KV，process_rss_drop_pct 仍接近噪声（见 Stage 7A §5.1）；
   要得到"明显且可信"的进程总 RSS 下降，必须把场景推向 KV 占比更高的区域
   （更长 ctx、更多 parallel、更多 idle 序列）。

4. 因此 Stage 7B 的主调节杆是 idle coverage:
   通过 idle 序列数 / idle history 长度 / ctx / parallel 四个旋钮抬高 coverage，
   观察 release_ratio 与 rss_drop 是否随之单调 scaling。
```

---

## 3. Controlled Workload Design

人工控制 idle 体量，用于验证机制 scaling。**这是 microbenchmark，不是业务负载收益声明。**

### 3.1 控制旋钮

```text
ctx                                 : -c，上下文长度（抬高单序列 KV 体量）
parallel                            : -np，并发序列数（抬高总 KV 体量）
n                                    : -n，active decode token 数
LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS    : idle 序列在进入 idle 前预热/积累的 KV token 数
                                       （直接抬高单个 idle 序列的 owned KV）
idle history 长度                    : 通过 idle prompt + warmup 控制 idle 序列常驻 KV 体量
resume 点                           : 何时 resume idle 序列（决定 idle 持续时长与释放窗口）
```

### 3.2 scaling 设计

```text
固定 active 侧（n / 一个 active 序列），单调抬高 idle 侧体量：
  扫 LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS（idle history 长度）；
  或扫 parallel（增加 idle 序列数）；
  或扫 ctx（抬高单序列 KV 上界）。
观察 idle_coverage_pct 随之上升时，release_ratio_total_pct / rss_drop_mib 是否同向 scaling。
```

### 3.3 边界声明

```text
1. controlled workload 是机制验证，证明"释放量随 idle coverage scaling 正确"；
2. 不得被表述为真实业务负载下的端到端 RSS 节省；
3. idle 由参数显式构造，idle/resume 时机由 env / 参数注入，不是自然对话节奏。
```

---

## 4. Semi-Real Paused/Resume Workload Design

用 multi-round / paused-resume conversation 模拟自然 idle/resume，更适合最终汇报展示。

```text
形态:
  多个会话序列轮流 active / paused / resume：
  会话 A 正在 decode（active）时，会话 B/C 处于 paused（idle）；
  轮到 B resume 时，A/C 进入 paused；
  idle 由"该会话当前没有在生成"这一自然状态触发，而非 env 强制。

与 controlled 的区别:
  idle/resume 时机来自对话轮转，而非参数注入；
  idle history 长度来自真实多轮上下文积累，而非 warmup token 注入。

纪律:
  1. 仍是 example 层驱动，不进 core scheduler；
  2. 报告需说明 idle/resume 由对话轮转触发；
  3. 同样用 Stage 7A 统一表，同时报 RSS 收益与 resume latency 代价。
```

---

## 5. Minimal Smoke Matrix

先跑 smoke（单次、低成本、确认管线与 correctness），再进 3-run median。

```text
smoke 目标:
  1. 确认 idle swap-out / madvise / resume 管线在该配置下不崩、correctness 全绿；
  2. 确认 idle_coverage_pct / release_ratio_total_pct / rss_drop_mib 字段能正常产出；
  3. 探测 OOM 边界，确定 3-run median 安全配置。
```

smoke 配置（单 run，由低到高，先小后大）：

```text
id     workload     ctx    par  warmup/idle_history                  n     resume点
S0     controlled   1024   2    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=64  128   active decode 末
S1     controlled   2048   2    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256 128   active decode 末
S2     controlled   2048   4    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256 128   active decode 末
SR0    semi-real    2048   3    multi-round（2-3 轮自然积累）         128   轮转 resume
```

```text
说明:
  S0 = 对齐既有 stage 的安全基线（ctx=1024, par=2），先确认无回归；
  S1 抬高 ctx + idle history，单序列 idle KV 上界拉高；
  S2 再加 parallel，抬高 idle 序列数 → idle coverage 上界进一步拉高；
  SR0 = semi-real 形态的最小 smoke；
  每升一档先观察 RSS 与是否接近 OOM，再决定是否进入下一档。
```

---

## 6. 3-Run Median Matrix

smoke 全绿且确认不 OOM 后，对选定配置跑 3-run median（沿用既有 stage 的 median 方法）。

```text
固定（除非该行显式扫描）:
  batch=128 ubatch=128
  cache-type-k=f32 cache-type-v=f32
  kv-unified
  temp=0
  prefetch / resume policy 固定一组（如 RESUME_PENDING_TOKEN 固定、PRESSURE_MODE=off）
  3-run median
  每组均报告 real_abnormal_matches
```

median 矩阵（每行 3 run 取 median，含 baseline 对照）：

```text
id     workload     ctx    par  warmup/idle_history    n     idle_coverage 目标
M-base baseline     2048   4    -- / --                128   0（无 idle 释放对照）
M1     controlled   2048   4    WARMUP=256             128   中
M2     controlled   2048   4    WARMUP=512             128   高
M3     controlled   4096   4    WARMUP=512             128   更高（OOM 风险，依 smoke 决定是否跑）
MR1    semi-real    2048   3    multi-round 2-3 轮      128   自然
```

```text
扫描轴说明:
  M1 -> M2 固定 ctx/par，只抬 idle history（WARMUP 256 -> 512），
         观察 idle_coverage / release_ratio / rss_drop 是否单调 scaling；
  M3 抬 ctx 到 4096，仅在 smoke 确认不 OOM 时纳入；
  MR1 给出 semi-real 形态的 median，用于汇报展示。
OOM 防护:
  3-run median 只跑 smoke 已确认安全的配置；
  任何 smoke 接近内存上限的档不进入 median。
```

---

## 7. Metrics and Result Table

沿用 Stage 7A 统一结果表（§6 模板）。Stage 7B 每行必填以下列：

```text
mode/workload  ctx  par  warmup/idle_history  idle_cov%  rel_total%  drop/idle%
  rss_drop_mib  proc_rss_drop%  rss/advised%  resume_ms  delta_ms  tok/s
  net_rss_mib  correctness  real_abn
```

字段对应 Stage 7A 分类：

```text
correctness / real_abnormal_matches          : §4.1（门槛，必须全绿 / 0）
idle_coverage_pct                             : §4.3（本阶段主调节杆）
release_ratio_total_pct / drop_over_idle_pct  : §4.3
rss_drop_mib / process_rss_drop_pct           : §4.2（核心收益）
rss_drop_over_advised_pct                     : §4.4
resume_first_token_ms / first_token_delta_ms  : §4.5（核心代价）
tokens_per_second                             : 代价侧，确认未显著劣化
net_rss_drop_before_resume_mib                : §4.7（区分释放与净节省）
```

占位表（Stage 7B smoke/median 填实测）：

```text
mode/workload  ctx   par  warmup/idle  idle_cov%  rel_total%  drop/idle%  rss_drop_mib  proc_rss_drop%  rss/advised%  resume_ms  delta_ms  tok/s  net_rss_mib  correct  real_abn
baseline       2048  4    --/--        --         --          --          0             0               --            --         0         --     0            green    0
controlled M1  2048  4    256/--       --         --          --          --            --              --            --         --        --     --           green    0
controlled M2  2048  4    512/--       --         --          --          --            --              --            --         --        --     --           green    0
semi-real MR1  2048  3    --/2-3轮     --         --          --          --            --              --            --         --        --     --           green    0
```

报告纪律（同 Stage 7A §6）：

```text
1. baseline 行必填，作为 delta / pct 分母；
2. correctness 非全绿或 real_abnormal_matches != 0 的行不参与 trade-off 比较；
3. 每个 workload 必须同时给出 RSS 收益列与 resume latency 代价列。
```

---

## 8. Expected Outcomes

结果一律用 trade-off 格式，不写"好/很好"，不对标 vLLM waste<4% / throughput 2-4x。

```text
预期 1（机制 scaling）:
  随 idle_coverage_pct 上升（WARMUP 256 -> 512、par 2 -> 4），
  release_ratio_total_pct 与 rss_drop_mib 单调上升；
  若不单调，说明释放被某处吃回或 idle 判定未生效，需归因。

预期 2（场景差异）:
  高 par / 高 ctx 配置下 process_rss_drop_pct 更可见（KV 占比更高，Stage 7A §5.2）；
  低配置下应改看 KV 侧指标（release_ratio / drop_over_idle，Stage 7A §5.1）。

trade-off 陈述模板（每个 workload 各填一句）:
  "在释放 X MiB / Y% KV / Z% process RSS 的情况下，
   resume first-token 增加 A ms，tokens/s 变化 B%，correctness 全绿。"

净收益要求:
  必须同时报 net_rss_drop_before_resume_mib；
  若 net 远小于 rss_drop，说明收益被 prefetch / resume 吃回（Stage 7A §4.7），
  按净值而非毛值主张节省。
```

---

## 9. Risks and Non-Goals

```text
Non-Goals:
  1. 不实现 PagedAttention kernel、不实现 continuous batching；
  2. 不进 core scheduler，workload 驱动留在 examples 层；
  3. 不新增 public API；
  4. 不对标 vLLM 的 KV waste < 4% 与 throughput 2-4x。

Risks:
  1. controlled workload 人工性强:
     idle 由参数构造，必须标注为 microbenchmark，不得当真实负载收益（§3.3）。

  2. 高 idle coverage 抬高 resume latency:
     更多 idle KV 被释放 → resume 时更多 blocks 需 swap-in，
     first_token_delta_ms 可能上升，必须随 RSS 收益一同报告。

  3. RSS drop 被 prefetch / resume 吃回:
     用 net_rss_drop_before_resume_mib 区分释放与净节省（Stage 7A §4.7、6D-C §9）。

  4. high ctx / high parallel 可能 OOM:
     先 smoke 探边界，3-run median 只跑已确认安全配置（§5、§6）。

  5. tokens/s run-to-run 抖动:
     tok/s 是代价侧指标，用 3-run median，且不解释为吞吐提升（Stage 7A §3、§8 风险3）。
```

---

## 10. Summary

```text
Stage 7B 通过扩大 idle-owned KV coverage 放大 current RSS drop，
主调节杆是 idle coverage（idle 序列数 / idle history 长度 / ctx / parallel）。

workload 分两类：
  controlled —— 人工构造 idle，验证 release_ratio / rss_drop 随 idle_coverage scaling，
                明确为 microbenchmark，不当真实负载收益；
  semi-real —— multi-round / paused-resume conversation 模拟自然 idle/resume，用于汇报展示。

先跑最小 smoke（S0 安全基线 -> S1 抬 ctx+idle history -> S2 加 parallel -> SR0 semi-real），
确认 correctness 全绿、字段产出正常、不接近 OOM，再对安全配置跑 3-run median。

所有结果用 Stage 7A 统一表，每个 workload 同时报 RSS 收益与 resume latency 代价，
评价固定为 trade-off 格式，并以 net_rss_drop_before_resume 为净收益口径，
```
