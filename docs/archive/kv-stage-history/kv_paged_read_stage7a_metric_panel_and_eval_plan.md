# KV Paged Read — Stage 7A Metric Panel and Evaluation Plan

> 本文档建立 Stage 7 后续实验的统一评价指标面板与评测方法。
> 本阶段只写 docs，不修改 `src/`、`include/`、public API、examples、CMake、scripts。
> 目标是为后续所有 Stage 7 实验提供一致、可复现、可解释的指标口径，
> 避免只看单一 RSS drop 或单一 tokens/s 而产生误导性结论。

---

## 1. Goal

Stage 7A 的目标是定义一套统一的指标面板（metric panel）与评测口径，使后续
Stage 7B 及以后的实验结果可以横向比较、可复现、可归因。

```text
1. 固定一组核心指标分类，覆盖 correctness / RSS / KV / madvise / latency / prefetch / net benefit；
2. 明确我们不直接对标 vLLM 的哪些指标，以及原因；
3. 定义低并发-中短上下文与高并发-长上下文两种场景下不同的评价重心；
4. 给出统一结果表模板，使每个 workload 都同时报告 RSS 收益与 resume latency 代价；
5. 把评价标准固定为 trade-off 格式，而非主观的“好/很好”。
```

赛题核心约束（贯穿全文）：

```text
扩大 KV cache 内存优化场景，获得更明显、更可信的 current RSS 下降，
同时保持 correctness 全绿、resume latency 代价可量化、可解释。
```

---

## 2. Why We Need a Unified Metric Panel

到 Stage 6D-C 为止，单轮实验已经能产出多类 telemetry（RSS、fallback、
first-token latency、tokens/s 等），但缺少统一口径会带来三个问题。

```text
1. 单一指标误导：
   只报 RSS drop 会掩盖 resume latency 代价；
   只报 tokens/s 会把 run-to-run 抖动误读为吞吐提升（见 Stage 6D-C §6.4）。

2. 跨阶段不可比：
   不同 stage 用不同字段名、不同 workload、不同 run 次数，
   结果无法横向比较，难以判断某个 policy 是否真的更优。

3. 收益不可归因：
   "RSS 下降"可能来自 KV release、madvise、padding recovery，
   也可能被 prefetch / resume 吃回（见 Stage 6D-C §9.2）；
   没有 net benefit 口径就无法区分"释放了"与"净节省了"。
```

统一面板要求每个实验都按相同分类、相同字段、相同 trade-off 格式报告，
使"释放 X MiB 的代价是 resume 增加 A ms"这类陈述可以被直接比较。

---

## 3. Why We Do Not Directly Use vLLM Metrics

vLLM 的两个常被引用的指标：

```text
1. KV waste < 4%      (PagedAttention 的内存碎片浪费极低)
2. throughput 2-4x    (相对 baseline serving 的吞吐倍数)
```

我们不直接对标这两个指标，原因是系统形态不同：

```text
vLLM:
  完整 serving engine
  + PagedAttention CUDA kernel（block 级非连续 KV 寻址）
  + continuous batching（请求级动态拼批）
  + 调度器（scheduler，抢占 / swap / 重排）。
  其 KV waste < 4% 来自 block 分配器，throughput 2-4x 来自 batching + kernel。

我们当前路线:
  llama.cpp 连续 KV cache 之上的运行时 KV block lifecycle 管理
  （block-aware / idle-aware / swap-aware / madvise-aware / prefetch-aware）。
  我们不实现 PagedAttention kernel、不实现 continuous batching、不进 core scheduler。
```

因此对标口径调整为：

```text
1. 不报 "KV waste < 4%"：
   我们不引入 block 分配器，碎片率不是我们的优化对象；
   我们关心的是 idle KV 的 release / swap-out / madvise 覆盖率。

2. 不报 "throughput 2-4x"：
   我们不做 continuous batching，steady-state 吞吐不是我们的提升来源；
   tokens/s 在我们这里是"代价侧"指标（确认未显著劣化），不是"收益侧"指标。

3. 我们报的是:
   idle KV 的 current RSS 下降（绝对 MiB + 进程占比 %），
   以及为换取该下降付出的 resume first-token latency 代价。
```

---

## 4. Core Metric Categories

所有 Stage 7 实验按以下 7 类指标报告。每类给出字段与含义。

### 4.1 Correctness（前置门槛）

```text
output_consistency      : resume 后输出是否与 baseline（无 swap）逐 token 一致
real_abnormal_matches   : 过滤 telemetry 字面命中后的真实异常数，必须 = 0
```

correctness 是门槛指标：未全绿的实验结果不进入任何 trade-off 比较。

### 4.2 RSS（核心收益）

```text
rss_drop_mib            : absolute current RSS drop（MiB），相对同配置 baseline
process_rss_drop_pct    : process RSS drop pct = rss_drop / baseline_process_rss
```

### 4.3 KV（释放质量）

```text
release_ratio_total_pct : 已释放 KV blocks / 总 KV blocks
idle_coverage_pct       : 被判定为 idle 且纳入释放候选的 KV 占比
drop_over_idle_pct      : 实际释放量 / idle 候选量（idle 中真正落地释放的比例）
```

### 4.4 madvise（释放是否落到物理页）

```text
advised_bytes           : MADV_DONTNEED 等 advise 覆盖的字节数
rss_drop_over_advised_pct : rss_drop / advised_bytes（advise 转化为 RSS 下降的效率）
skip_neighbor           : 因 page 邻接保护而跳过 advise 的块数
```

### 4.5 latency（核心代价）

```text
resume_first_token_ms   : resume 后首 token 延迟
first_token_delta_ms    : resume_first_token_ms - baseline_first_token_ms
```

### 4.6 prefetch（恢复时机）

```text
remaining_blocks_before_resume : resume 前仍未恢复的 blocks（落到 fallback path）
prefetch_ms_max                : 单次 prefetch 调用的最大耗时
```

### 4.7 net benefit（净收益，区分"释放"与"净节省"）

```text
rss_rebound_due_to_prefetch    : 因 active prefetch / resume 恢复而被吃回的 RSS
net_rss_drop_before_resume     : resume 前实际净下降 = rss_drop - rss_rebound
```

净收益口径专门回应 Stage 6D-C §9 的限制：resume 前保留的 RSS 会在
first-token path 上随 fallback 恢复而部分吃回，net_rss_drop 才是可主张的净节省。

---

## 5. Scenario-Specific Evaluation

KV cache 占总进程 RSS 的比例随场景变化，导致评价重心不同。
报告结果时必须先声明场景，再选对应重心指标。

### 5.1 低并发 / 中短上下文

```text
特征:
  KV cache 占总 RSS 比例低，模型权重占主导；
  total process RSS drop 会被权重稀释，绝对 MiB 偏小。

评价重心（看 KV 侧，而非进程总 RSS）:
  release_ratio_total_pct      KV 释放比例
  idle_coverage_pct            idle 覆盖
  drop_over_idle_pct           idle 中真正落地释放
  unused tail / padding recovery   尾块与 padding 回收
  KV resident drop（mincore）  KV 常驻物理页下降

不应作为主指标:
  process_rss_drop_pct（会被权重稀释，趋近噪声）
```

### 5.2 高并发 / 长上下文

```text
特征:
  KV cache 占总 RSS 比例显著变大，是核心优化场景；
  idle KV 的释放能直接体现在进程总 RSS 上。

评价重心（看进程总 RSS）:
  process_rss_drop_pct         进程总 RSS 下降占比
  rss_drop_mib                 绝对 RSS 下降
  net_rss_drop_before_resume   净下降
  相同内存预算下可承载的上下文长度 / 并发请求规模

辅助指标:
  release_ratio_total_pct / idle_coverage_pct 仍报告，用于解释 RSS 下降来源
```

场景声明规则：

```text
每个结果表行必须带 ctx 与 parallel 两列，使读者能判断该行属于哪个场景，
并据此选择正确的重心指标进行比较。
```

---

## 6. Recommended Result Table Template

所有 Stage 7 实验使用统一结果表。列定义如下（每行一个 mode/workload）。

```text
mode/workload                  : policy 模式或 workload 名
ctx                            : 上下文长度
parallel                       : 并发序列数
warmup / idle_history          : warmup token 数 / idle history 配置
idle_coverage_pct              : §4.3
release_ratio_total_pct        : §4.3
drop_over_idle_pct             : §4.3
rss_drop_mib                   : §4.2
process_rss_drop_pct           : §4.2
rss_drop_over_advised_pct      : §4.4
resume_first_token_ms          : §4.5
first_token_delta_ms           : §4.5
tokens_per_second              : 代价侧，确认未显著劣化
net_rss_drop_before_resume_mib : §4.7
correctness                    : §4.1（全绿 / 否）
real_abnormal_matches          : §4.1（必须 0）
```

模板表（占位行，Stage 7B 填入实测）：

```text
mode/workload  ctx  par  warmup/idle  idle_cov%  rel_total%  drop/idle%  rss_drop_mib  proc_rss_drop%  rss/advised%  resume_ms  delta_ms  tok/s  net_rss_mib  correct  real_abn
baseline       ----  --  --/--        --         --          --          0             0               --            --         0         --     0            green    0
<policy_A>     ----  --  --/--        --         --          --          --            --              --            --         --        --     --           green    0
<policy_B>     ----  --  --/--        --         --          --          --            --              --            --         --        --     --           green    0
```

报告纪律：

```text
1. baseline 行必填，作为 delta / pct 的分母；
2. correctness 非全绿或 real_abnormal_matches != 0 的行不参与 trade-off 比较；
3. 每个 workload 必须同时给出 RSS 收益列与 resume latency 代价列，缺一不可。
```

---

## 7. Stage 7B Experiment Direction

Stage 7B 的实验原则。

### 7.1 Controlled workload — 验证 scaling，不伪装成真实负载

```text
用途:
  通过可控参数（idle history 长度、idle 序列数、ctx）扫描 idle_coverage_pct，
  验证 release_ratio / rss_drop 是否随 idle coverage 单调、可预测地变化。

纪律:
  controlled workload 是机制验证，不得被表述为真实业务负载下的收益；
  结果只用于证明"机制随 coverage scaling 正确"，不用于声明端到端节省。
```

### 7.2 Semi-real workload — 更自然的 idle / resume

```text
用途:
  用 multi-round / paused-resume conversation 模拟更自然的 idle/resume 节奏，
  让某些序列在多轮对话间真实进入 idle，再被 resume。

纪律:
  仍是 example 层驱动，不进 core scheduler；
  报告需说明 idle/resume 由对话节奏触发，而非 env 强制注入时机。
```

### 7.3 两类 workload 的共同要求

```text
1. 每个 workload 都必须同时报告 RSS 收益与 resume latency 代价；
2. 都使用 §6 统一结果表；
3. 都先声明场景（§5），再选重心指标；
4. 都用 3-run median，并报告 real_abnormal_matches。
```

---

## 8. Risks and Non-Goals

```text
Non-Goals:
  1. 不实现 PagedAttention kernel、不实现 continuous batching；
  2. 不进 core scheduler，policy 继续留在 examples 层；
  3. 不新增 public API；
  4. 不对标 vLLM 的 KV waste < 4% 与 throughput 2-4x。

Risks:
  1. RSS 收益被 resume 吃回:
     必须用 net_rss_drop_before_resume 区分"释放"与"净节省"（§4.7）。

  2. 场景错配:
     在低并发/中短上下文场景用 process_rss_drop_pct 当主指标，
     会因权重稀释得到趋近噪声的结论（§5.1）。

  3. tokens/s 误读:
     run-to-run 抖动可能被误读为吞吐提升；
     tokens/s 在本面板是代价侧指标，只用于确认未显著劣化（§3、Stage 6D-C §6.4）。

  4. controlled workload 越界:
     controlled scaling 实验不得被表述为真实负载收益（§7.1）。
```

---

## 9. Summary

```text
Stage 7A 定义了 Stage 7 后续实验的统一评价指标面板与评测口径。

核心面板分 7 类：correctness（门槛）、RSS（核心收益）、KV、madvise、
latency（核心代价）、prefetch、net benefit（区分释放与净节省）。

我们不直接对标 vLLM 的 KV waste < 4% 与 throughput 2-4x，
因为 vLLM 是完整 serving engine + PagedAttention kernel + continuous batching + scheduler，
而我们是 llama.cpp 连续 KV cache 上的运行时 KV block lifecycle 管理。

评价分两类场景：低并发/中短上下文看 KV 侧（release ratio / idle coverage / KV resident drop），
高并发/长上下文看进程总 RSS（process RSS drop pct / absolute drop / 可承载规模）。

所有结果用统一结果表，每个 workload 同时报告 RSS 收益与 resume latency 代价，
评价标准固定为 trade-off 格式（释放 X MiB / Y% KV / Z% RSS，resume +A ms，tok/s 变 B%，correctness 全绿），
不写主观的"好/很好"。

Stage 7B 用 controlled workload 验证 idle coverage scaling（不伪装成真实负载），
用 semi-real multi-round / paused-resume conversation 模拟更自然的 idle/resume。
```
