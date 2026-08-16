# 基于 llama.cpp 的 KV Cache 运行时内存优化技术报告

## 摘要

大语言模型推理过程中，KV cache 会随上下文长度、并发会话数和多轮对话历史快速增长。在边缘设备或内存受限环境下，KV cache 的物理内存占用会显著压缩模型推理可用资源，甚至限制长上下文和多会话服务能力。原始 `llama.cpp` 的 KV cache 采用构造期连续预分配的大块 K/V tensor，并通过 cell 级元数据管理 token、position 和 sequence ownership。该设计稳定且便于通用推理，但在长上下文、多会话、暂停/恢复类 workload 中，会导致大量暂时不活跃的 KV 仍常驻物理内存。

本项目面向“操作系统功能赛 / 边缘设备上的 LLM 推理优化”方向，在 `llama.cpp` 上实现并验证了一套 KV cache 运行时内存回收机制。项目从原始 KV cache 访存路径分析出发，先后实现 tail/lazy reclaim、paged row index、idle KV block swap-out、`madvise(MADV_DONTNEED)` 物理页释放、resume swap-in、prefetch/defer、fast maintenance、trace replay workload 和 mincore resident page 诊断。最终系统能够在 idle session 暂停期间将其 KV block 写入 backing store 并释放对应物理页，在 session resume 时按需恢复，从而在保持 correctness / safety 的同时降低进程 current RSS。

历史 ShareGPT-backed synthetic trace replay 的代表配置为 fast-maintenance V5。在 Llama-3-8B-Instruct Q4_K_M、CPU KV path、K/V cache f32 配置下，ctx4096 场景实现 611.883 MiB process RSS drop，约 6.713% total RSS drop，active TPS 回退约 3.007%；ctx8192 场景实现 1619.195 MiB process RSS drop，约 15.997% total RSS drop，active TPS 回退约 1.006%。mincore 诊断进一步显示，ctx8192 下 baseline-like KV resident 约 2047.75 MiB，而 V5 final KV resident 约 426.0 MiB，KV resident drop 约 1621.75 MiB，证明该历史实验中的 RSS 下降主要来自 KV cache 物理驻留页减少，而非统计噪声或其他进程内存波动。该结果不是当前 Final F16 formal benchmark，也不是 Global Route A formal performance result。

需要强调的是，本项目当前是比赛原型与研究性系统实现，不声称已经完成生产级 llama-server scheduler、完整异步 prefetch、GPU backend 支持或完整复刻 vLLM PagedAttention。项目核心贡献在于：在 `llama.cpp` 现有架构内，以较小改动引入 block-level KV reclaim 机制，并用可复现实验链证明 idle KV 的物理页可以被安全释放和恢复。

**关键词**：LLM 推理；KV cache；llama.cpp；RSS；madvise；mincore；Paged KV；Swap；Prefetch；边缘设备

---

## 1. 项目背景与问题定义

### 1.1 背景

大语言模型推理主要由模型权重、计算中间缓冲区和 KV cache 共同占用内存。对于固定模型权重而言，随着上下文长度和并发会话数增加，KV cache 成为运行时内存增长的重要来源。尤其在多轮对话、长上下文推理和多 session 服务场景下，不同会话的历史 KV 可能长期保留在内存中，即使其中一部分会话已经暂停或短期内不会继续生成。

在边缘设备上，物理内存通常有限。若 KV cache 始终以 full resident 的方式常驻，会带来以下问题：

1. 长上下文可用长度受限；
2. 多会话并发能力受限；
3. 系统更容易触发内存压力或 OOM；
4. 推理服务难以在低内存设备上稳定运行；
5. KV cache 与模型权重、compute buffer 争用物理内存。

因此，本项目选择从操作系统内存管理视角切入，将虚拟内存、按需提交、换出、物理页释放和预取思想引入 LLM 推理运行时。

### 1.2 问题定义

本项目关注的问题可以定义为：

```text
在不破坏 llama.cpp 推理正确性的前提下，
识别当前 active request 不再需要立即访问的 idle KV blocks，
将这些 blocks 对应的数据保存到 backing store，
释放其物理驻留页，
并在 request resume 前或 resume 时正确恢复，
最终降低进程 current RSS，同时控制吞吐和首字延迟回退。
```

与单纯减少逻辑 KV 使用量不同，本项目强调：

```text
1. 进程 RSS 是否真实下降；
2. KV cache resident pages 是否真实减少；
3. resume 后上下文是否仍可正确使用；
4. active request 是否不会读到已经换出的 KV；
5. 写入路径是否不会写入 swapped block；
6. 性能损失是否可解释、可控制。
```

### 1.3 目标与非目标

#### 目标

本项目的主要目标包括：

1. 分析 `llama.cpp` 原始 KV cache 的内存行为；
2. 找到导致 idle KV 常驻物理内存的结构性原因；
3. 在 `llama.cpp` 内实现 block-level KV reclaim 机制；
4. 支持 idle block swap-out、madvise 和 resume swap-in；
5. 支持 prefetch/defer，降低 resume first-token latency；
6. 使用 trace replay workload 验证多 session 暂停/恢复场景；
7. 使用 mincore 证明 KV resident page 真实减少；
8. 建立可复现的测试、日志和结果解析流程。

#### 非目标

本项目当前不声称完成：

1. 生产级 llama-server slot scheduler；
2. 真实线上 request trace 验证；
3. GPU backend 完整支持；
4. 完整异步 prefetch thread；
5. 完整复刻 vLLM PagedAttention；
6. 稳定降低进程 peak RSS；
7. 上游可直接合并的最终 public API 设计。

---

## 2. llama.cpp 原始 KV Cache 机制分析

### 2.1 原始结构

`llama.cpp` 的 KV cache 以连续 K/V tensor 为核心。构造期根据上下文长度、层数、KV head 数和 head dimension 分配 K/V 存储空间，并通过 cell metadata 管理每个 token 的 position、sequence ownership 和生命周期状态。

可以概括为：

```text
KV cache = large contiguous K tensor + large contiguous V tensor + cell metadata
```

其中：

1. K/V 数据存放在预分配 buffer 中；
2. cell metadata 描述 token 和 sequence 状态；
3. 写入时通过索引将 token 的 K/V 写入指定 cell；
4. attention 读取时通常基于 `[0, n_kv)` 的连续窗口构造 K/V view；
5. mask 用于数值屏蔽无效 token，但不等价于避免底层物理页被触碰。

### 2.2 写侧与读侧的不对称

通过源码分析和实验，我们确认 `llama.cpp` 的写侧与读侧存在显著不对称：

```text
写侧：
  具备一定 scatter 能力，可以按 cell / row 写入。

读侧：
  仍依赖连续 [0, n_kv) view，再通过 mask 屏蔽无效 cell。
```

这意味着，只要某些 KV cell 仍落在 `[0, n_kv)` 的连续读窗口内，即使它们属于 idle request、当前 active request 不需要，其对应物理页也可能在 attention 计算中被触碰或被 correctness 路径要求恢复。

### 2.3 早期 exact swap 的限制

项目早期实现过 runtime KV swap 原型，完成了如下闭环：

```text
swap-out -> poison / madvise probe -> swap-in -> output correctness
```

该阶段证明 KV 数据可以从原 buffer 复制到 backing store，并在需要时原位恢复。但 exact swap 仍然存在根本限制：

```text
只要 attention 读侧仍要求 [0, n_kv) 全区间 resident，
被换出的 KV 很快会在 ensure_resident 路径中被恢复，
从而导致 RSS 难以稳定下降。
```

因此，简单的 exact swap + madvise 不能解决核心问题。后续必须让 idle block 退出 active read window，或者将读路径改造成可以按 row/block 非连续读取的形式。

### 2.4 设计转向

基于上述分析，项目从“按 cell 换出”转向“block-level / paged-read / idle-request reclaim”主线：

```text
1. 将 KV 管理粒度从 cell 提升到 block；
2. 识别 idle request 拥有的 blocks；
3. 使用 row_idx / gather 让 idle-only block 退出 active read window；
4. 对安全 block 执行 swap-out / madvise；
5. 在 resume 时恢复；
6. 通过 prefetch/defer 控制恢复成本。
```

---

## 3. 总体设计

### 3.1 设计原则

本项目遵循以下设计原则：

1. **默认关闭**：新增机制均通过环境变量显式开启，默认不影响普通 `llama.cpp` 行为；
2. **机制与策略分离**：library/core 提供 swap、madvise、prefetch 等机制，上层 example/trace driver 表达调度策略；
3. **安全优先**：active-visible KV 不允许被错误换出；写入路径不允许写入 swapped block；
4. **可观测性**：每个阶段都输出 telemetry，支持 RSS、resident、swap、prefetch、latency 解析；
5. **逐步验证**：先 correctness，再 RSS，再 performance，再 workload realism；
6. **边界明确**：严格区分 current RSS、peak RSS、theoretical KV capacity、actual KV resident drop。

### 3.2 系统架构

最终系统可分为六层：

```text
┌──────────────────────────────────────────────┐
│ Trace / workload driver                      │
│ - session lifecycle                          │
│ - arrival / idle / resume                    │
│ - prefetch policy                            │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│ Public / experimental memory API             │
│ - prefetch_seq                               │
│ - prefetch_seq_step                          │
│ - set_seq_prefetch_protected                 │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│ KV cache state machine                       │
│ - RESIDENT                                   │
│ - SWAPPED                                    │
│ - released / nonresident pages               │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│ Paged row index / gather path                │
│ - active-visible rows -> real row            │
│ - idle-only rows -> dummy resident row       │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│ Backing store + madvise                      │
│ - swap-out data copy                         │
│ - MADV_DONTNEED page release                 │
│ - swap-in recovery                           │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│ Diagnostics                                  │
│ - RSS                                        │
│ - mincore resident pages                     │
│ - safety counters                            │
│ - latency / TPS                              │
└──────────────────────────────────────────────┘
```

### 3.3 KV block 状态机

核心状态机如下：

```text
RESIDENT
  |
  | idle-owned + not active-visible + safe candidate
  v
SWAPPED
  |
  | resume / prefetch / read-before-use
  v
RESIDENT
```

在启用 madvise 后：

```text
RESIDENT -> SWAPPED -> physical pages nonresident
```

注意，`SWAPPED` 表示数据已保存到 backing store，可恢复；这不同于简单丢弃。

### 3.4 active / idle / resume 语义

在多 session workload 中，一个 session 可能处于：

```text
WAITING
PREFILL
ACTIVE_DECODE
PAUSED_IDLE
RESUME_PENDING
RESUMING
FINISHED
```

其中：

1. `ACTIVE_DECODE`：session 当前正在生成，必须保持其可见 KV resident；
2. `PAUSED_IDLE`：session 暂停，历史 KV 可成为 reclaim candidate；
3. `RESUME_PENDING`：上层知道该 session 即将 resume，可提前 prefetch；
4. `RESUMING`：session 重新进入 active batch，需要恢复被 swap-out 的历史 KV。

---

## 4. 核心模块实现

### 4.1 Tail / Lazy reclaim

早期优化围绕 unused / tail KV 区域展开。对于已分配但当前未实际使用的 KV tail，可以通过 lazy clear 和 tail madvise 降低 current RSS。

核心思想：

```text
逻辑容量保持不变；
当前未使用的尾部页可以释放物理驻留；
未来写入时由 OS 重新 fault-in。
```

该阶段证明：

```text
1. tail / lazy reclaim 可以真实降低 current RSS；
2. peak RSS 不容易通过 KV tail 清理显著下降；
3. 后续应重点优化运行时 current RSS。
```

### 4.2 Paged row index 与 non-identity gather

为解决 idle KV 位于 active physical read window 内的问题，项目引入 paged row index。该机制利用 graph 内的 row index / gather 路径，将 K/V 读取从简单连续 view 改为可控 row mapping。

对于 active-visible row：

```text
row_idx[row] = real_physical_row
```

对于当前 active request 不可见的 idle-only row：

```text
row_idx[row] = dummy_resident_row
```

这样，idle-only block 不再被 active attention 真实读取，从而成为 safe swap candidate。

相关开关：

```bash
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
```

### 4.3 Idle KV block swap-out

当 idle-owned block 已经退出 active read window 后，可以执行 swap-out：

```text
1. 将 block 的 K/V 数据写入 backing store；
2. 记录 offset / size；
3. 将 block state 设为 SWAPPED；
4. 更新 swap-out counters；
5. 在启用 madvise 时释放原 block 对应物理页。
```

对应开关：

```bash
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
```

为了保证安全，系统会检查：

```text
1. active-visible block 不应被 swap-out；
2. 当前写入目标 block 必须 resident；
3. read path 需要的 swapped block 必须先恢复；
4. backend failure 必须记录并暴露。
```

### 4.4 Resume swap-in

当 session resume 后，之前属于该 session 的 swapped block 需要恢复。恢复路径包括：

```text
1. 从 backing store 读取 block 数据；
2. 写回原 K/V tensor 对应位置；
3. 将 block state 从 SWAPPED 改回 RESIDENT；
4. 更新 swap-in counters；
5. 继续正常 decode。
```

该机制保证 idle reclaim 不等于丢弃上下文。session resume 后仍可以使用原历史 KV。

### 4.5 Prefetch

如果所有 swap-in 都暴露在 resume first-token path 上，会增加首字延迟。因此项目实现了 prefetch 机制，使上层 driver 可以在 resume 前逐步恢复目标 session 的 KV block。

相关接口包括：

```text
llama_memory_prefetch_seq(...)
llama_memory_prefetch_seq_step(...)
llama_memory_set_seq_prefetch_protected(...)
```

典型策略：

```text
1. probe remaining swapped blocks；
2. 根据 active window 和 blocks_per_step 计算 start token；
3. active decode 期间分批恢复；
4. resume 前如果 remaining=0，则 first-token path 不再承担完整恢复成本；
5. 如果窗口不足，则允许 partial prefetch + fallback。
```

### 4.6 Defer

在 resume first-token 阶段，如果同步触发新的 idle swap-out，会放大首字延迟。项目引入 defer 机制：

```bash
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

其作用是：

```text
在 resume first-token 关键阶段，暂缓 idle swap-out maintenance，
避免恢复和回收同时争用热路径。
```

### 4.7 Fast maintenance

早期 idle swap maintenance 在每一步 decode 中执行较多检查和候选维护，造成明显吞吐回退。最终 fast-maintenance V5 引入以下控制：

```bash
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16
```

语义：

1. `DEBUG_PROBES=0`：关闭非必要诊断 probe；
2. `EVERY_TOKENS=8`：每 8 个 token 执行一次 idle maintenance；
3. `MAX_BLOCKS_PER_STEP=8`：限制单次维护处理的 block 数；
4. `MIN_IDLE_STEPS=16`：只有 idle 足够久的 block 才进入回收候选。

fast maintenance 的关键效果是减少不必要的短 idle thrashing，避免刚进入 idle 的 KV 频繁 swap-out / swap-in。

---

## 5. Workload 与评测方法

### 5.1 Controlled workload

早期通过 `kv-idle-swap-resume` driver 构造双 session 场景：

```text
seq0:
  prefill -> idle -> resume

seq1:
  prefill -> active decode
```

该 workload 用于验证：

```text
1. idle block 能否 swap-out；
2. seq1 active 输出是否不变；
3. seq0 resume 输出是否不变；
4. resume 是否触发 swap-in；
5. madvise 是否降低 RSS；
6. prefetch/defer 是否降低首字延迟。
```

### 5.2 Semi-real multi-session workload

随后项目实现 `kv-semi-real-multisession` driver，用固定 timeline 模拟多个聊天 session 的暂停和恢复。

该阶段验证：

```text
1. 多 session lifecycle；
2. S0-S5 策略组合；
3. tail/lazy + idle swap + prefetch + defer 的组合收益；
4. 3-run median 稳定性。
```

### 5.3 Corpus-backed semi-real workload

项目进一步支持从真实文本 corpus 中切分 prompt，使多 session workload 的 prompt 来源更接近真实文本。

### 5.4 ShareGPT-backed trace replay workload

最终 workload 使用 ShareGPT 对话数据生成 trace replay 输入。trace replay driver 支持：

```text
session_id
turn_id
arrival_ms
prompt_source
target_decode_tokens
idle_ms
```

支持的 prompt source 包括：

```text
corpus:<offset>:<chars>
text:<inline_text>
file:<relative_path>
```

其中 `file:` prompt 路径相对 trace 文件所在目录解析，并拒绝绝对路径、`..` 越界、空路径和不安全路径。

ShareGPT-backed trace 的含义是：

```text
真实部分：
  对话内容、多轮结构、用户 prompt 文本来自 ShareGPT。

合成部分：
  arrival_ms、idle_ms、target_decode_tokens、long-idle 分布由脚本生成。
```

因此它不是生产线上真实 trace，但比固定 prompt / 固定 timeline 更接近真实多轮对话 workload。

### 5.5 评价指标

本项目主要使用以下指标：

#### 内存指标

```text
rss_kb:
  进程当前 RSS。

rss_drop_mib:
  baseline RSS - optimized RSS。

total_rss_drop_pct:
  rss_drop / baseline RSS。

theoretical_kv_capacity:
  根据层数、KV head、head dim、cache type 和 ctx-size 计算的理论 KV 容量。

RSS drop / theoretical KV capacity:
  RSS 下降量与理论 KV 容量之比，只作容量参照，不等价于“KV 释放比例”。

kv_mincore_resident_bytes:
  mincore 诊断得到的 KV resident bytes。

kv_resident_drop:
  baseline-like KV resident - optimized KV resident。
```

#### 性能指标

```text
active_tps:
  active decode tokens per second。

active_decode_ms:
  active decode 总耗时。

total_wall_ms:
  workload 总墙钟时间。

resume_first_avg_ms:
  session resume first-token latency。
```

#### Correctness / safety 指标

```text
exit=0
real_abnormal=0
all_finished=1
active_visible_violation=0
write_to_swapped=0
paged_swapped_active_visible_violation=0
paged_write_to_swapped_block=0
backend_failures=0
```

---

## 6. 实验结果

### 6.1 早期 Tail / Lazy reclaim

早期 tail madvise 在不同 ctx-size 下已能降低 current RSS：

```text
ctx=2048:
  current RSS drop ≈ 219.6 MiB

ctx=4096:
  current RSS drop ≈ 475.8 MiB

decode speed regression:
  about 3% to 5%
```

该阶段结论：

```text
1. tail/lazy reclaim 可以降低 current RSS；
2. 它不是 peak RSS 优化；
3. 它释放的是 unused/tail 物理页，未来写入会重新 fault-in；
4. 它不能解决 idle session 历史 KV 常驻问题。
```

### 6.2 Stage 5：idle swap + madvise correctness

Stage 5 验证 idle block 的完整生命周期：

```text
RESIDENT -> SWAPPED / madvise -> resume -> RESIDENT
```

核心结论：

```text
1. idle KV block 可以被识别；
2. non-identity gather 可以让 idle-only block 退出 active read window；
3. safe candidate 可以 swap-out；
4. resume 时可以 swap-in；
5. active output 和 resume output 均可保持一致；
6. madvise 后 RSS 可以真实下降。
```

### 6.3 Stage 6：Prefetch / resume-aware policy

Stage 6 解决的问题是：

```text
idle swap 降低 RSS，但 resume 时需要恢复 KV；
如果完全在 first-token path 恢复，会增加首字延迟。
```

#### 同步 prefetch

同步 prefetch 可以降低 resume first-token latency，但会提前恢复所有 block，使 RSS 收益在 resume 前被吃回。

#### Interleaved prefetch

将 prefetch 拆成小步：

```text
prefetch_seq_step(seq_id, max_blocks)
```

可以把恢复成本分摊到 active decode 窗口中，降低单次阻塞。

#### Auto delayed prefetch

根据 remaining blocks 和 active window 自动计算最晚启动点：

```text
need_steps = ceil(remaining_blocks / blocks_per_step)

auto_start_token =
    active_total_tokens
  - (need_steps - 1) * every_tokens
  - safety_tokens
```

#### Resume-aware pseudo-scheduler

引入 resume-pending 概念后，系统可以表达：

```text
complete prefetch:
  窗口充足，resume 前全部恢复。

partial prefetch:
  窗口不足，只恢复部分 block，剩余 fallback 到 resume path。

no prefetch:
  不提前恢复，RSS 保留最多，first-token latency 最高。
```

这一阶段证明了 RSS 与 latency trade-off 可以被调度策略显式控制。

### 6.4 Stage 9：Semi-real multi-session 结果

Stage 9 的 semi-real multi-session workload 中，S5 组合策略包含：

```text
tail/lazy reclaim
idle swap
madvise
prefetch
defer
```

3-run median 代表结果：

```text
S5 RSS drop vs S0:
  ≈ 403.6 MiB

S5 active TPS delta:
  ≈ -0.8%

S5 resume first-token average delta:
  ≈ +3.7 ms

runs:
  18 次运行均 exit=0
  real_abnormal=0
  active-visible violation=0
  write-to-swapped=0
```

相对于当时约 511.750 MiB 的 KV buffer 总容量，403.6 MiB 的 process RSS drop 约等于其 78.9%。严谨表述是：

```text
S5 的进程 RSS 下降量约等于当前 KV buffer 总容量的 78.9%。
```

### 6.5 Historical Stage 12-C：ShareGPT-backed synthetic trace replay 结果

最终结果使用 ShareGPT-backed trace replay workload，并采用 fast-maintenance V5 配置。

#### ctx4096 3-run median

| 指标                                    |          结果 |
| ------------------------------------- | ----------: |
| Process RSS drop                      | 611.883 MiB |
| Total RSS drop                        |      6.713% |
| RSS drop / theoretical KV capacity    |     59.754% |
| Active TPS delta                      |     -3.007% |
| Active decode time delta              |     +3.100% |
| Total wall time delta                 |     +2.713% |
| Resume first-token weighted avg delta |   +1.242 ms |

解释：

```text
ctx4096 下，V5 在保持低性能回退的同时，
实现约 612 MiB process RSS drop。
```

#### ctx8192 3-run median

| 指标                                 |           结果 |
| ---------------------------------- | -----------: |
| Process RSS drop                   | 1619.195 MiB |
| Total RSS drop                     |      15.997% |
| RSS drop / theoretical KV capacity |      79.062% |
| Active TPS delta                   |      -1.006% |
| Active decode time delta           |      +1.017% |
| Total wall time delta              |      +0.966% |

解释：

```text
ctx8192 下，KV cache 理论容量更大，
idle KV reclaim 的收益更显著。
V5 实现约 1.58 GiB RSS 下降，
同时 TPS 回退约 1%。
```

### 6.6 Fast maintenance 效果

早期 aggressive idle swap 可带来约 716 MiB RSS drop，但 TPS 回退约 33% 到 34%。fast-maintenance V5 的设计取舍是：

```text
牺牲约 100 MiB RSS drop；
换回约 30 个百分点的 TPS 损失。
```

最终 V5 的 ctx4096 结果：

```text
RSS drop:
  ≈ 611.9 MiB

TPS delta:
  ≈ -3.0%
```

说明 fast maintenance 能有效降低 idle maintenance 对 decode 热路径的影响。

### 6.7 mincore resident page 诊断

为了证明 RSS drop 来自 KV cache 物理页释放，项目使用 `LLAMA_KV_PAGED_MINCORE=1` 对 KV buffer resident page 进行诊断。

#### ctx4096

```text
baseline-like KV resident:
  1023.75 MiB

V5 KV resident:
  426.0 MiB

KV resident drop:
  597.75 MiB

process RSS drop:
  ≈ 611.746 MiB
```

#### ctx8192

```text
baseline-like KV resident:
  2047.75 MiB

V5 KV resident:
  426.0 MiB

KV resident drop:
  1621.75 MiB

process RSS drop:
  ≈ 1619.168 MiB
```

结论：

```text
process RSS drop 与 KV resident drop 数量级高度一致，
说明最终 RSS 下降主要来自 KV resident pages 减少。
```

在该历史 trace workload 中，ctx4096 和 ctx8192 的 V5 final KV resident 都约为 426 MiB，这说明当时优化后的实际驻留 KV 与“当前活跃/必要 KV”更相关，而不是与预分配 ctx-size 线性绑定。该观察不外推为当前 Final F16 或 Global Route A 的 formal 结果。

---

## 7. Correctness 与 Safety 验证

### 7.1 输出正确性

项目在不同阶段分别使用以下方式验证 correctness：

```text
1. baseline vs paged 输出一致；
2. baseline vs swap 输出一致；
3. seq1 active 段输出一致；
4. seq0 resume 段输出一致；
5. trace replay session 全部 finished；
6. exit code = 0。
```

### 7.2 active-visible safety

核心安全要求：

```text
active request 当前可能读取的 block 不能被 swap-out。
```

系统通过 row_idx remap 和 safety counters 检查：

```text
active_visible_violation=0
paged_swapped_active_visible_violation=0
```

### 7.3 write-to-swapped safety

写入路径必须保证目标 block resident：

```text
write_to_swapped=0
paged_write_to_swapped_block=0
```

如果写入目标处于 `SWAPPED` 状态，应先恢复再写入。

### 7.4 backend failure

swap-in / swap-out / tensor set / backing store 操作失败必须记录。正式结果要求：

```text
backend_failures=0
```

### 7.5 诊断开关默认关闭

以下诊断开关默认关闭，不影响普通运行：

```text
LLAMA_KV_PAGED_IDLE_TRACE
LLAMA_KV_PAGED_MINCORE
LLAMA_KV_PAGED_TRACE
LLAMA_KV_PAGED_REFAULT_TRACE
```

其中 `LLAMA_KV_PAGED_MINCORE=1` 只用于 read-only diagnostic，不参与 swap/madvise/state-machine 决策。

---

## 8. 工程实现与代码边界

### 8.1 修改范围

主要涉及：

```text
src/llama-kv-cache.cpp
src/llama-kv-cache.h
src/llama-graph.cpp
src/llama-graph.h
src/llama-memory.h
include/llama.h
examples/kv-trace-replay/
examples/kv-semi-real-multisession/
examples/kv-idle-swap-resume/
scripts/kv_trace_from_sharegpt.py
```

### 8.2 默认兼容性

默认情况下：

```text
LLAMA_KV_PAGED 未设置；
LLAMA_KV_LAZY_TAIL 未设置；
LLAMA_KV_LAZY_CLEAR 未设置；
LLAMA_KV_PAGED_SWAP 未设置；
LLAMA_KV_PAGED_IDLE_SWAP 未设置。
```

因此普通 `llama.cpp` 运行路径不启用 paged / swap / madvise / prefetch 逻辑。

### 8.3 Public API 边界

项目新增少量 experimental memory-level API，用于 prefetch 和 protected 控制。这些 API 服务于比赛原型验证，不代表最终 upstream API 设计已经定稿。

### 8.4 Policy 位置

项目明确区分 mechanism 与 policy：

```text
library / core:
  提供 KV block 状态、swap、madvise、prefetch、diagnostic 等机制。

example / trace driver:
  表达 session lifecycle、resume_pending、prefetch timing、pressure mode 等策略。
```

原因是 `llama.cpp` core 当前没有完整 request-level scheduler；idle→resume_pending 信号通常由上层 server / application / trace driver 知道。把 policy 硬塞进 core 会污染普通 decode 热路径，也不利于后续接入真实 server scheduler。

### 8.5 文档与复现

最终文档结构：

```text
README.md
docs/reproduce_kv_cache_optimization.md
docs/final_technical_report.md
docs/kv_trace_replay_stage12c_real_sharegpt_results.md
docs/kv_lifecycle_evidence_protocol.md
docs/kv_block_lifecycle_contract.md
docs/kv_pressure_scheduler_contract.md
docs/current_memory_governor_architecture.md
docs/archive/kv-stage-history/
```

README.md 是决赛仓库唯一高层入口；其他文档分别承担复现、历史结果、目标契约和架构说明职责。历史阶段文档被归档保留，不与当前入口竞争 authority。

---

## 9. 局限性

### 9.1 当前不是生产级 server scheduler

最终 workload 是 ShareGPT-backed trace replay，而不是 llama-server HTTP production workload。trace replay 能表达多 session、多 turn、arrival、idle、resume，但仍然没有真实 server slot、HTTP 请求、streaming、continuous batching、真实用户到达分布等全部因素。

### 9.2 尚未完整支持 GPU backend

当前主要验证 CPU KV path。GPU backend 下 K/V buffer 管理、host/device transfer、madvise/mincore 语义均不同，需要单独适配。

### 9.3 未实现 true async prefetch

当前 prefetch 是调度层分步触发，不是独立异步线程。它可以把恢复成本分摊到 active decode 窗口，但仍可能在调用点产生同步开销。

### 9.4 不优化 peak RSS

项目重点是 current RSS。早期 trace 显示，peak RSS 很大程度受模型权重加载和 compute buffer 影响，因此不能把本项目结果表述为稳定降低 peak RSS。

### 9.5 ShareGPT-backed trace 不是线上真实 trace

ShareGPT 提供真实对话内容和多轮结构，但 arrival time、idle interval、target decode tokens 等由脚本生成。因此应表述为：

```text
ShareGPT-backed synthetic trace
```

不能表述为：

```text
真实线上 trace
```

### 9.6 Experimental API 需要后续收敛

当前新增的 memory-level API 主要服务于实验 driver 和 trace replay。若要进入上游或生产系统，需要重新设计 API 命名、返回语义、错误处理和 backend 兼容性。

---

## 10. 创新点

### 10.1 在 llama.cpp 内实现 block-level KV reclaim

本项目在 `llama.cpp` 内部实现 KV block 状态、swap-out、madvise、swap-in 和 telemetry，能够实际运行和测量 RSS。

### 10.2 利用 row_idx/gather 规避连续 read window 限制

通过 paged row index 和 non-identity gather，idle-only block 可以退出 active read window，从而避免 active attention 读取已回收 block。

### 10.3 将操作系统内存管理思想引入 LLM 推理运行时

项目将以下 OS 概念引入 KV cache 管理：

```text
resident / nonresident pages
swap-out / swap-in
madvise
lazy commit
prefetch
working set
idle page reclaim
```

### 10.4 使用 mincore 验证 resident page 变化

不仅测量 process RSS，还通过 mincore 观察 KV buffer resident pages，证明内存下降来自 KV cache 物理驻留减少。

### 10.5 建立 trace replay 验证链

通过 `kv-trace-replay` 和 ShareGPT converter，项目形成可控且可复现的多 session trace workload。

### 10.6 fast maintenance 控制性能回退

通过 maintenance interval、per-step budget、minimum idle steps 和 debug probe gate，项目显著降低 aggressive idle swap 的吞吐损失。

---

## 11. 未来工作

后续可以继续从以下方向推进：

### 11.1 接入 llama-server scheduler

将机制接入真实 server slot / request scheduler，使 idle detection、resume_pending、prefetch policy 来自真实 serving 层。

### 11.2 引入真实 memory pressure signal

后续可接入：

```text
/proc/meminfo
cgroup memory pressure
PSI
runtime RSS threshold
```

但应将 signal collection 与 policy decision 分阶段验证，避免归因混淆。

### 11.3 实现异步 prefetch thread

通过独立线程或 event-driven 机制执行 prefetch，使恢复成本更少暴露在 decode 调用路径上。

### 11.4 支持 GPU backend

GPU backend 需要重新处理 device memory、host staging、DMA、unified memory 或 explicit transfer。

### 11.5 扩大 workload 覆盖

后续可扩展：

```text
1. 更多模型规模；
2. 更长 ctx-size；
3. 更高 parallel；
4. 更复杂 ShareGPT trace；
5. server benchmark；
6. 长尾 idle 分布；
7. 多轮长对话。
```

### 11.6 收敛统一策略接口

未来可以将分散的 env 收敛为更高层策略：

```text
LLAMA_KV_RECLAIM_POLICY=off|tail_only|idle_only|auto
```

并根据 memory pressure、latency target 和 request priority 自动选择 reclaim / prefetch 行为。

---

## 12. 总结

本项目围绕 `llama.cpp` 的 KV cache 内存占用问题，完成了从源码分析、机制设计、原型实现、正确性验证、RSS 验证、resident page 诊断到 trace replay workload 的完整优化链路。

项目最终证明：

1. `llama.cpp` 原始连续预分配 KV cache 在多 session / 长上下文场景中存在 idle KV 物理页常驻问题；
2. 简单 exact swap 不能稳定降低 RSS，原因是读侧仍依赖 `[0, n_kv)` 连续 view；
3. 通过 paged row index / non-identity gather，可以让 idle-only block 退出 active read window；
4. idle KV block 可以被安全 swap-out，并通过 `madvise(MADV_DONTNEED)` 释放物理页；
5. resume swap-in 保证 idle session 恢复后仍能使用历史上下文；
6. prefetch / defer 能控制 resume first-token latency；
7. fast maintenance 能显著降低 idle swap maintenance 对吞吐的影响；
8. ShareGPT-backed trace replay 证明该机制在更接近多轮对话的 workload 下有效；
9. mincore 诊断证明最终 process RSS drop 与 KV resident page drop 高度一致。

最终，在 ctx8192 ShareGPT-backed trace replay workload 中，V5 配置实现约 1619 MiB process RSS drop，约 15.997% total RSS drop，active TPS 回退约 1.006%，并通过 safety counters 和 mincore diagnostics 验证 correctness 与 resident page reduction。该结果表明，在 `llama.cpp` 现有架构上引入 block-level idle KV reclaim 是可行的，并为后续在真实 serving 系统中进一步实现 memory-pressure-aware KV cache 管理提供了基础。
