# FlexKV-OS：权重–KV 协同的 LLM 运行时内存系统

FlexKV-OS 是基于 `llama.cpp` / `ggml` 的研究型运行时内存系统，面向边缘设备、低内存环境和长上下文多会话推理。项目将模型权重、MoE expert working set 和 KV Cache 视为可调度的物理内存对象，目标不是改变模型语义，而是在可验证的门禁下控制驻留、换出、恢复和跨组件预算分配。

> **项目定位**：比赛原型与研究性实现。当前结论以源码、测试和已记录实验为准；未完成 formal benchmark 的路径不写成性能 PASS。

## 1. 当前结论速览

| 模块 | 当前状态 | 证据口径 |
|---|---|---|
| Dense Flex | 已实现 | 有历史 cgroup `memory.current`、RSS、吞吐和受限内存实测证据 |
| MoE Buffer + CLG | 已实现 | 有历史 cgroup RSS、吞吐、expert working-set 和 PPL 证据 |
| KV V5 reclaim | 已实现并完成历史验证 | ShareGPT-backed **synthetic trace replay**；不是线上生产 trace |
| Native F16 paged KV | 已实现，已有单测基础 | Final F16 formal benchmark 仍 pending |
| Global-KV-A | mechanism/correctness qualification 已完成并冻结 | `GLOBAL_KV_A_FINAL_PASS`；不代表性能收益 |
| Global Route A | runner/parser 与实验边界正在冻结 | formal performance benchmark 仍 pending |
| Global Route B | 尚未实现 | 不宣称自动 workload-aware 最优分配 |

`GLOBAL_KV_A_FINAL_PASS` 的准确含义是：

```text
KV live OFFLOAD
  → authoritative physical relief
  → object/generation-bound physical credit
  → Global reallocation
  → MoE budget grant
  → Exact Restore
  → fixed-budget resident control token exact
```

该 qualification 证明机制和 correctness 闭环，不代表 TPS、TTFT、TPOT、resume latency 或任何百分比性能收益。Global Route A 的 formal performance benchmark 仍须独立完成。

## 2. 系统架构

最终方向是**共享物理内存预算与跨组件协同**，而不是四个互不相干的优化模块：

```text
┌──────────────────────────────────────────────────────────────┐
│                   Global Memory Governor                     │
│ physical headroom · ROI · credit · reallocation · grants    │
└───────────────┬──────────────────┬───────────────────────────┘
                │                  │
        ┌───────▼───────┐  ┌───────▼───────┐
        │   Dense Flex   │  │   MoE Buffer  │
        │ layer ring     │  │ expert cache  │
        └───────┬────────┘  └───────┬───────┘
                │                   │
                └─────────┬─────────┘
                          ▼
                 ┌─────────────────┐
                 │ KV Cache Control│
                 │ paged/reclaim   │
                 └─────────────────┘
```

### 2.1 Dense Flex

Dense 模型每个 token 通常需要按层访问大量权重。Dense Flex 以 decoder layer 为粒度维护匿名 ring working set，支持：

- layer-level ring slot 与顺序读取；
- LRU / clean reclaim；
- adaptive ring size；
- 受限物理预算下的显式驻留边界；
- 计算前同步加载作为 correctness 兜底。

Dense Flex 的历史结果表明，匿名 backing 的 working set 控制可以区别于单纯 `mmap` + `madvise` 的“逻辑映射仍在、物理 RSS 未真实下降”路径。详见 [TEST_RESULTS.md](TEST_RESULTS.md) 和 [final technical report](docs/final_technical_report.md)。

### 2.2 MoE Buffer 与 CLG

MoE Buffer 以 expert slice 为粒度维护工作集，核心状态为 `COLD / INFLIGHT / RESIDENT`，并结合：

- LRU 驱逐与 hot expert 保护；
- 多 worker 预取；
- 动态 budget；
- CLG（Current-state / Locality Guided）预测下一层可能激活的 expert；
- `weight stream callback` 在 kernel 前同步兜底。

CLG 只影响预取时机和性能，不改变真实 router 结果。预测不命中时，执行路径仍必须同步取得正确 expert 数据。

### 2.3 KV Cache Controller

KV 路径将 **logical KV state** 与 **physical residency** 解耦：逻辑上下文可以继续保留，而当前 active request 不可见的 idle-owned block 可以被安全换出。

典型生命周期为：

```text
active-visible KV
  → idle-owned + active-invisible
  → RELEASE / OFFLOAD
  → backing store authoritative
  → MADV_DONTNEED physical relief
  → PREFETCH / Exact Restore
  → active-visible again
```

主要机制包括：

- paged row index 与 non-identity gather；
- block ownership、active visibility 和 generation 约束；
- `RELEASE / OFFLOAD / PREFETCH / Exact Restore`；
- backing store、`madvise(MADV_DONTNEED)` 和 `mincore` 诊断；
- resume-aware prefetch/defer 与 fast maintenance。

这里的 paged KV 是当前实现中的 paged row index / reclaim 路径，**不宣称完整复刻 vLLM PagedAttention**。若没有完整 block table、non-contiguous physical page 和对应 attention 读取语义，就不能使用“完整 PagedAttention”这一表述。

### 2.4 Global Memory Governor

Global Governor 统一观察 Dense、MoE 和 KV 的物理预算与 headroom，并以 object/generation-bound physical credit 连接释放和再分配：

- 采集 RSS / cgroup / KV resident 等物理观测；
- 区分 logical bytes、allocated bytes、resident bytes 和 physical relief；
- 依据 headroom、ROI、cooldown 和 action priority 选择候选；
- 将 KV 的 authoritative physical relief 转化为有身份约束的 credit；
- 向 MoE 或其他组件授予固定预算下的 resident control token。

当前 Governor 已有协同治理骨架，但不能把骨架自动等价成 Global Route A formal performance result；完整生命周期 evidence protocol 与 Route B 仍有明确实现边界。

## 3. 代表性历史结果

以下结果来自 **historical ShareGPT-backed synthetic trace replay**，使用 CPU KV path、Llama-3-8B-Instruct Q4_K_M、K/V f32 和 fast-maintenance V5。它们不是当前 Final F16 formal benchmark，也不是线上生产 trace。

| 场景 | Process RSS drop | Total RSS drop | Active TPS delta |
|---|---:|---:|---:|
| ctx4096 | 611.883 MiB | 6.713% | -3.007% |
| ctx8192 | 1619.195 MiB | 15.997% | -1.006% |

ctx8192 的 historical `mincore` diagnostic 记录约 `2047.75 MiB → 426.0 MiB` 的 KV resident 变化，说明当时 RSS 下降主要对应 KV 物理驻留页减少。该证据不能外推为 peak RSS、生产 serving、Final F16 或 Global Route A 的性能结论。

权重侧历史 cgroup 结果显示：

- Dense Flex 的匿名 ring working set 能在受限内存下真实控制物理驻留；
- MoE Buffer 在 expert working set 足够时可使 evictions 降至 0，并保留较高吞吐；
- MoE 侧有 PPL / bit-exact 对照，CLG 只改变预取，不改变 routing。

组合结果、测试环境和历史数据表见 [TEST_RESULTS.md](TEST_RESULTS.md)；KV V5 的详细历史记录见 [ShareGPT-backed trace replay report](docs/kv_trace_replay_stage12c_real_sharegpt_results.md)。

## 4. 复现与快速开始

### 4.1 基础构建

```bash
cmake -S . -B build
cmake --build build -j
```

KV trace driver 的最小构建入口：

```bash
cmake --build build -j --target llama-kv-trace-replay
```

### 4.2 最小 trace smoke

准备模型和 corpus 后运行：

```bash
MODEL=/path/to/model.gguf

LLAMA_KV_TRACE_FILE=examples/kv-trace-replay/traces/smoke_4s2t.tsv \
LLAMA_KV_TRACE_CORPUS_FILE=/path/to/corpus.txt \
./build/bin/llama-kv-trace-replay \
  -m "$MODEL" \
  --ctx-size 512 \
  --batch-size 128 \
  --ubatch-size 64 \
  --parallel 4 \
  --seed 1 \
  --temp 0
```

通过标准至少包括：`exit=0`、所有 session 完成、无 abnormal、无 active-visible violation、无 write-to-swapped violation。

### 4.3 ShareGPT-backed synthetic trace

```bash
python3 scripts/kv_trace_from_sharegpt.py \
  --input /path/to/sharegpt.json \
  --output-dir /path/to/trace_out \
  --num-sessions 8 \
  --max-turns-per-session 3 \
  --start-gap-ms 0 \
  --turn-gap-ms 400 \
  --jitter-ms 50 \
  --idle-ms-low 1200 \
  --idle-ms-high 2400 \
  --min-decode-tokens 16 \
  --max-decode-tokens 96 \
  --chars-per-token 4 \
  --max-user-chars 2048 \
  --seed 1 \
  --overwrite
```

这里的 prompt / 多轮内容来自 ShareGPT，arrival、idle、jitter、decode target 等 timing 是合成的，因此结果应称为 ShareGPT-backed synthetic trace replay。

### 4.4 资格测试入口

当前 Final F16 与 Global Route A 的 runner/parser/test 文件属于受保护的并行实验边界，不在本 README cleanup 中修改：

- [Final F16 runner](scripts/run-kv-offload-benchmark.py)
- [Final F16 parser](scripts/parse-kv-offload-benchmark.py)
- [Final F16 tests](tests/test-kv-offload-benchmark.py)
- [Global Route A runner](scripts/run-global-route-a-benchmark.py)
- [Global Route A parser](scripts/parse-global-route-a-benchmark.py)
- [Global Route A tests](tests/test-global-route-a-benchmark.py)

这些入口的存在不等于 formal performance PASS；正式结果必须以对应 parser/verdict 和完整 benchmark artifact 为准。

## 5. 证据口径与正确性

必须区分以下概念：

- `logical KV bytes`：逻辑上下文仍保留的容量；
- `allocated KV bytes`：运行时分配的 KV 存储范围；
- `resident KV bytes`：采样时实际驻留的物理页；
- process RSS / cgroup `memory.current`：进程或 cgroup 层面的观测；
- peak RSS：全程峰值，不能由 current/final RSS drop 代替。

`mincore` 只用于系统观测和诊断，不是 lifecycle state authority。KV block 的 ownership、generation、transaction 和 authoritative copy 必须由运行时状态与 evidence protocol 维护。

正确性门禁包括：

1. active request 不读取已换出的不可见 block；
2. backing 写入完成并成为权威副本后才释放物理页；
3. resume / prefetch 后 Exact Restore 恢复可读内容；
4. write-to-swapped、active-visible、generation 和 transaction 错配必须 fail-closed；
5. 权重预测失败时由同步路径兜底；
6. PPL、输出或 trace replay correctness 对照不能被 RSS 结果替代。

## 6. 当前限制

当前系统仍是研究原型，主要限制为：

- Final F16 formal benchmark 尚未完成；
- Global Route A formal performance benchmark 尚未完成；
- Global Route B 尚未实现；
- 当前主要验证 Linux CPU KV path，GPU device-memory reclaim 尚未形成同等证据；
- 尚未完整接入 llama-server HTTP production workload、真实线上 request trace、continuous batching 和完整 slot scheduler；
- 尚未形成完整异步 prefetch thread 的生产级实现；
- 重点优化 current RSS，不宣称 peak RSS 已稳定降低；
- 当前 paged KV 不宣称完整 vLLM PagedAttention；
- target lifecycle contracts 与当前 telemetry/runner/parser 仍存在实现差距，详见对应契约文档。

## 7. 文档与交付物索引

### 当前入口

- [Final technical report](docs/final_technical_report.md)：系统设计、机制、限制和历史实验汇总。
- [KV cache reproduction guide](docs/reproduce_kv_cache_optimization.md)：构建、smoke、trace replay 和历史 V5 复现步骤。
- [KV lifecycle evidence protocol](docs/kv_lifecycle_evidence_protocol.md)：目标证据身份、事件闭包和 parser 判定边界。
- [KV block lifecycle contract](docs/kv_block_lifecycle_contract.md)：block/cell、驻留、事务、quarantine 和 generation 目标语义。
- [KV pressure scheduler contract](docs/kv_pressure_scheduler_contract.md)：scheduler、action priority 和 lifecycle core 目标边界。
- [Global memory governor architecture](docs/current_memory_governor_architecture.md)：Dense/MoE/KV 协同治理骨架及当前缺口。

### 演示与图示

- [参赛文档](docs/参赛文档.pdf)
- [参赛演示文档](docs/操作系统功能赛.pptx)
- [系统架构图](figures/flow/main.png)
- [Dense Flex 图](figures/flow/weight.png)
- [MoE Buffer 图](figures/flow/moe.png)
- [CLG 预取图](figures/flow/pre.png)
- [Paged KV 图](figures/flow/swap.png)
- [测试结果图目录](figures/test/)

### 历史归档

- [KV 阶段性文档](docs/archive/kv-stage-history/)
- [Lazy-loading 历史最终报告](docs-archive/lazy-loading/LAZY_LOADING_FINAL_REPORT.md)
- [Lazy-loading VM 发现](docs-archive/lazy-loading/LAZY_LOADING_VM_DISCOVERY.md)
- [Lazy-loading 实现状态](docs-archive/lazy-loading/IMPLEMENTATION_STATUS_REPORT.md)
- [Lazy-loading 设计对比](docs-archive/lazy-loading/IMPLEMENTATION_VS_DESIGN_COMPARISON.md)

历史文档用于追溯技术路线，不是当前能力和 formal benchmark 的 authority。

## 8. 四个模块的最终边界

```text
Dense Flex:
  layer-level weight working set；有历史实测证据。

MoE Buffer + CLG:
  expert-level working set + prediction；有历史 RSS / throughput / PPL 证据。

KV Cache Controller:
  logical state 与 physical residency 解耦；Final F16 formal benchmark pending。

Global Memory Governor:
  共享物理预算、credit 和 reallocation；Global-KV-A mechanism/correctness qualification 已完成并冻结（`GLOBAL_KV_A_FINAL_PASS`）。Route A performance pending，Route B 未实现。
```

所有新的性能数字都必须注明 workload、模型、memory kind、RSS 口径、是否为 peak、是否为 synthetic trace，以及对应的 runner/parser/verdict。