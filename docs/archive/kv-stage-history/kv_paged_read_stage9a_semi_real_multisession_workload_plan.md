# Stage 9-A：Semi-real Multi-session Workload 设计文档

## 0. 文档性质与边界

本文件是 **设计文档**，只新增文档，不修改任何源码。

严格区分两类内容：

```text
[已实现]   当前仓库中已有的 env gate / 函数 / 机制（Stage 8C–8F 已验证）；
[Stage 9-B] Stage 9 后续实现部分（driver / workload / case matrix runner）；
[Stage 10] unified auto policy，本阶段不实现。
```

Stage 9 只做 semi-real multi-session **workload driver**，用现有底层 env gate 做手动组合验证；**不实现** `LLAMA_KV_RECLAIM_POLICY=auto`，auto 留到 Stage 10。

---

## 1. 背景与目标

当前进展：

1. Stage 8D：证明 idle KV swap/madvise + full prefetch + resume defer 可降低 current RSS，并把 resume first-token latency 拉回 baseline 附近；
2. Stage 8E：通过 MINCORE diagnostic audit 证明 RSS drop 对应 KV resident pages 真实释放；
3. Stage 8F：完成 unified KV reclaim policy 设计文档，把 tail/lazy clear 与 idle swap/madvise 统一为 KV lifecycle-aware resident memory management 的设计口径。

当前局限：

```text
idle seq 数量、warmup token、resume 时机仍由 example driver 人为写死；
底层 idle-owned block detection 是自动的，但上层 workload 还不够接近真实多会话请求流；
比赛汇报中还需要证明机制不只适用于固定 microbenchmark。
```

Stage 9 要补的是 **semi-real multi-session scheduler / workload driver**，让 active / paused / resume 状态来自更接近真实多轮对话的调度流，而不是单一 seq0/seq1 固定时序。目标是为后续验证 tail/lazy 与 idle swap 两条机制在更自然多会话场景下是否互补，提供一个可控、可复现、可归因的 workload 基座。

---

## 2. 为什么暂不直接接入 llama-server

llama-server 已有 slot / continuous batching / request queue 等真实调度逻辑。但直接接入会一次性引入：

```text
HTTP 收发；
slot 复用与回收；
context shift；
并发排队；
streaming 输出。
```

这些变量同时出现会让内存/延迟归因变得困难。Stage 9 先做 simplified scheduler driver，保持：

```text
可控：调度 timeline 固定、可枚举；
可复现：prompt 内置、temp=0、seed 固定；
可定位：单变量 case matrix 可逐项归因；
可对比：直接复用现有 env gate，与 Stage 8 结果同口径。
```

只有 Stage 9 semi-real workload 站稳后，才适合进一步进入真实 trace 或 server scheduler 集成。

---

## 3. Session 状态机设计

每个 session 是一个独立逻辑请求，绑定一个 `llama_seq_id`，在 deterministic timeline 下经过以下状态：

```text
WAITING
PREFILL
ACTIVE_DECODE
PAUSED_IDLE
RESUME_PENDING
RESUMING
FINISHED
```

| 状态 | 进入条件 | 参与 active batch | 是否持有 KV | 可能成为 idle-owned KV | 是否需要 prefetch / defer |
| --- | --- | --- | --- | --- | --- |
| `WAITING` | session 已登记但 prompt 未进入 | 否 | 否 | 否 | 否 |
| `PREFILL` | timeline 到达该 session 的入场 step | 是（prefill batch） | 开始写入 | 否（刚写入即 active） | 否 |
| `ACTIVE_DECODE` | prefill 完成、持续生成 token | 是 | 是 | 否（active-owned，受保护） | 否 |
| `PAUSED_IDLE` | timeline 标记暂停，不再进入 batch | 否 | 是（保留历史 KV） | **是**（idle-owned，可 swap/madvise） | 否（被动 swap-out） |
| `RESUME_PENDING` | timeline 标记即将 resume | 否（仍未解码） | 是（部分可能 swapped） | 是（但应被 prefetch protect） | 是（prefetch during active / auto delayed） |
| `RESUMING` | resume first-token decode | 是 | 是（需恢复访问） | 否（恢复中） | 是（defer 新 swap-out 保护首 token） |
| `FINISHED` | 解码达到目标或 EOG | 否 | 释放 | 否 | 否 |

关键不变量（与 Stage 8F §4.2 一致）：

```text
ACTIVE_DECODE / RESUMING 的 KV 始终受保护，不被 idle swap-out；
只有 PAUSED_IDLE 持有的 used-but-idle KV 是 idle swap/madvise 的主要对象；
RESUME_PENDING 期间应通过 set_seq_prefetch_protected 防止刚 prefetch 的 block 被再次 evict；
RESUMING 首 token 通过 defer_idle_swapout 避免在关键路径同步触发新一轮 swap-out。
```

---

## 4. Session 类型设计

至少三类 session，prompt 使用内置固定文本（不联网、不依赖外部数据集），重点是可复现：

```text
long_context_session:
  较长 prompt + 多轮续写，产生较多 KV；
  暂停后是制造 idle-owned KV 的主要来源。

short_context_session:
  中短 prompt，模拟普通短问答；
  KV in-use 较小，适合观察 tail / free capacity 是否明显。

bursty_session:
  prefill 后短暂 active，随即 paused，之后 resume；
  专门用于压 pause/resume 与 prefetch/defer 路径。
```

内置 prompt 可直接复用并扩展现有 driver 中的固定文本风格（参考 `examples/kv-idle-swap-resume/idle-swap-resume.cpp` 的 `idle_prompt` / `active_prompt`），按类型给定不同长度，保证 temp=0、seed 固定时输出确定。

---

## 5. Deterministic timeline

给出一个可实现的固定调度表（step 为逻辑调度步，不必等于单次 `llama_decode`）：

```text
step 0-64:
  session A (long_context)   PREFILL -> ACTIVE_DECODE
  session B (short_context)  WAITING
  session C (bursty)         WAITING

step 65-160:
  session A   PAUSED_IDLE        # A 暂停，成为 idle-owned KV 来源
  session B   PREFILL -> ACTIVE_DECODE
  session C   PREFILL -> ACTIVE_DECODE(短) -> PAUSED_IDLE

step 161-240:
  session A   RESUME_PENDING      # 触发 prefetch during active
  session B   PAUSED_IDLE
  session C   RESUMING -> ACTIVE_DECODE

step 241-320:
  session A   RESUMING -> ACTIVE_DECODE   # 首 token 触发 defer
  session B   RESUME_PENDING -> RESUMING
  session D (short_context)  PREFILL -> ACTIVE_DECODE  # 短请求中途进入
  session C   FINISHED
```

该表只是参考设计；Stage 9-B 可调整 step 边界，但必须保持 deterministic（固定、可枚举、可复现），以便单变量归因。timeline 由 driver 用一张静态调度表驱动，不引入随机到达。

---

## 6. 与现有机制对接

Stage 9-B driver 应尽量复用现有底层 env gate，不新增 policy 语义：

```text
[已实现] LLAMA_KV_PAGED
[已实现] LLAMA_KV_PAGED_INGRAPH
[已实现] LLAMA_KV_PAGED_GATHER_NONIDENTITY
[已实现] LLAMA_KV_PAGED_IDLE_SWAP
[已实现] LLAMA_KV_PAGED_IDLE_SWAP_MADVISE
[已实现] LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE
[已实现] LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED
[已实现] LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME
[已实现] LLAMA_KV_LAZY_TAIL
[已实现] LLAMA_KV_LAZY_CLEAR
```

明确边界：

```text
Stage 9 不实现 LLAMA_KV_RECLAIM_POLICY=auto；
Stage 9 只用现有 env gate 做手动组合验证；
auto 选路逻辑（先 tail/free、再 idle、始终保护 active）留到 Stage 10。
```

也就是说，Stage 9 的「策略」体现在 case matrix 的 env gate 组合上，而不是代码里的自动决策。

---

## 7. Case matrix

设计后续可跑的单变量 case matrix（clean perf 关闭 MINCORE）：

| case | 组合 | 主要回答的问题 |
| --- | --- | --- |
| `S0_baseline_paged_off` | paged off | semi-real workload 下的 latency / TPS / RSS baseline |
| `S1_paged_on_reclaim_off` | paged on，reclaim 全关 | paged bookkeeping 本身在多 session 下的额外开销 |
| `S2_tail_lazy_only` | `LLAMA_KV_LAZY_TAIL` + `LLAMA_KV_LAZY_CLEAR` | 仅回收 unused/free/tail capacity 的收益（short_context 主导） |
| `S3_idle_swap_only` | `LLAMA_KV_PAGED_IDLE_SWAP(_MADVISE)`，无 prefetch/defer | 仅回收 used-but-idle KV 的 RSS 收益与 resume latency 代价 |
| `S4_idle_swap_prefetch_defer` | idle swap + prefetch during active + auto delayed + defer | prefetch/defer 是否能在多 session 下仍把 resume first-token 拉回 baseline |
| `S5_tail_idle_prefetch_defer` | tail/lazy + idle swap + prefetch + defer | 两条机制叠加是否互补，是否出现 page fault / resume latency 叠加 |

归因逻辑：S2/S3 单独验证两条线的独立收益；S4 验证 idle 线的 latency 可控性；S5 验证融合，重点观察 tail 与 idle 叠加的相互影响（这正是 Stage 8F §6.6 提到、尚未实测的部分）。

---

## 8. 指标

每个 case 至少采集以下指标：

```text
process_rss_drop_mib
kv_total_capacity_mib
kv_inuse_mib
unused_or_tail_mib
idle_owned_mib
swapped_nonresident_mib
madvise_mib
resume_first_ms
active_tps
fallback_blocks
prefetch_blocks
page_fault_or_refault_count
correctness_status
real_abnormal
```

口径说明（沿用 Stage 8D / 8E 分工）：

```text
clean perf run 关闭 MINCORE / refault trace，用于 latency / TPS / RSS 结论；
MINCORE / refault trace 仅用于 diagnostic run（footprint / residency audit）；
page_fault_or_refault_count 属于 diagnostic 信号，不与 clean perf latency 同 run 读取；
correctness_status / real_abnormal 每个 case 都必须检查，作为前置门槛。
```

多 session 下，`resume_first_ms` / `idle_owned_mib` 等应能按 session（或至少按 long/bursty 类型）分别上报，避免被聚合掩盖。

---

## 9. Stage 9-B 实现建议

`[Stage 9-B]` 后续实现，本文档不实现：

```text
新增 example driver 目录：
  examples/kv-semi-real-multisession/

复用策略：
  尽量复用 examples/kv-idle-swap-resume/idle-swap-resume.cpp 已有机制：
    warmup_idle_seq / prefill / active decode / resume 循环；
    llama_kv_cache_set_seq_prefetch_protected()；
    llama_kv_cache_defer_idle_swapout()；
    llama_memory_prefetch_seq() / _seq_step()；
    current RSS / prefetch stats telemetry 输出格式。
  把单一 seq0/seq1 时序泛化为「N 个 session + 静态调度表」驱动。

最小 CLI / env：
  -m / -n / --ctx-size / --parallel 等沿用现有参数；
  新增一个内置 timeline 选择（如 LLAMA_KV_SEMI_REAL_TIMELINE=default）；
  session 类型与 prompt 长度用内置常量，先不做外部配置文件。

需要补的 telemetry：
  per-session resume_first_ms / idle_owned / swapped_nonresident；
  case 级 unused_or_tail_mib（配合 tail/lazy gate）。

先不做：
  外部 prompt 数据集加载；
  随机到达 / 真实并发；
  完整 server scheduler；
  auto policy。
```

建议路径选 `examples/kv-semi-real-multisession/`，与现有 `examples/kv-idle-swap-resume/`、`examples/kv-idle-telemetry/` 命名风格一致，便于复用 CMake 与 telemetry 约定。

---

## 10. 风险与非目标

非目标：

```text
不接入 llama-server；
不实现完整生产 scheduler；
不实现 unified auto policy（留到 Stage 10）；
不实现 vLLM-style full paged KV allocator；
不追求自然语言任务质量。
```

风险：

```text
semi-real workload 仍不是线上真实 workload，只是更自然的多 session 构造；
prompt 长度和调度 timeline 仍需人为设计；
tail/lazy 与 idle swap 同时开启（S5）可能导致 page fault / resume latency 叠加；
需要后续单变量矩阵逐项归因，避免把叠加效应误读为单一机制收益；
defer 计数单位仍是 row_idx fill（Stage 8D §9.2），多 session resume 下是否仍足够需要验证。
```

---

## 11. 下一步

```text
1. Stage 9-B：实现 examples/kv-semi-real-multisession/ driver，复用现有 env gate 与 telemetry；
2. 跑 S0–S5 case matrix（clean perf 关闭 MINCORE），并补一组 diagnostic MINCORE run 做 footprint 对照；
3. 归因 tail/lazy 与 idle swap 在多 session 下是否互补、是否叠加产生 latency 代价；
4. 站稳后再进入 Stage 10 unified auto policy（LLAMA_KV_RECLAIM_POLICY=auto）。
```
