# Architecture

> 记录当前源码可验证的稳定结构。讨论方案必须标记为 proposed，不得混入已实现架构。

- Last verified: 2026-07-16
- Evidence commit: `a744830e90969a2298785cdd994901f8f448995a`
- Verification scope: 当前源码、Git diff、clean-HEAD E2I 功能 artifact 与三轮 E2G/E2I controlled A/B artifact；本轮未运行构建、测试或实验

## System Boundary

- 基础系统为 `llama.cpp` / `ggml`。项目扩展位于模型加载、CPU weight-stream callback、KV cache、attention graph 输入和独立 example/runner 层。
- 当前已核对路径主要面向 Linux CPU/host memory。KV paged 路径要求 `n_stream == 1 && !v_trans`；`madvise`/`mincore` 不适用于 GPU device memory。
- Dense Flex、MoE-Buffer/CLG、lazy KV 与 paged KV 均由环境变量显式启用；普通未配置路径不主动进入这些实验机制。

## Runtime Data Flow

### 权重路径

```text
model loader / GGUF file descriptors
  -> Dense: llama_flex layer registration -> ring-slot pread/prefetch
  -> MoE: expert tensor registration -> anonymous expert slots -> worker prefetch/LRU
  -> CPU weight-stream callback before kernel
  -> synchronous wait/load fallback
  -> GGML CPU kernel
```

- Dense Flex 以 decoder layer 为注册和驻留单位，后台 I/O 将层权重装入有限 ring slot，并在 callback 中重定向 tensor data。
- MoE-Buffer 为 `*_exps.weight` 分配匿名虚拟缓冲区，按 expert slice 从模型文件读取；预算和 LRU 控制 resident expert，冷页可用 `MADV_DONTNEED` 回收。
- CLG 在当前层 hidden state 上预测下一层 expert，并向 MoE-Buffer 提交异步提示；真实 kernel 前的 weight-stream callback 仍负责同步兜底，因此预测不直接决定 routing 正确性。

### KV 路径

```text
slot allocation / KV writes
  -> paged logical-to-physical metadata
  -> paged_row_idx graph input
  -> get_k/get_v row gather for attention

idle-owned + active-invisible RESIDENT block
  -> pack cell-major K/V bytes
  -> one write_cells() fixed-slot range write
  -> publish offsets + state SWAPPED
  -> optional MADV_DONTNEED

resume/read/write requires block
  -> read_cells() into staging
  -> validate/unpack
  -> commit tensor contents + state RESIDENT
  -> attention/decode continues
```

### Static paged identity fast path

```text
context construction
  -> resolve eligibility once for the context lifetime
  -> eligible static identity mapping: no paged_row_idx input, continuous K/V views
  -> otherwise: paged_row_idx input, per-step row fill, K/V GET_ROWS gather

graph cache reuse
  -> compare cached graph row-index topology with current context topology
  -> topology matches: normal shape checks may permit reuse
  -> topology differs: reject reuse and rebuild the graph
```

- Eligibility is context-lifetime state, not a per-token speculation. It requires explicit fast-path request, paged and in-graph modes, `n_stream == 1`, `!v_trans`, no approximate-dynamic view, F32 K/V layers, identity block mapping, and no requested dynamic remap, swap, release, madvise or mapping/swap-in/backing-I/O fault injection.
- Eligible E2I omits `paged_row_idx` creation/fill and bypasses K/V `GET_ROWS`, exposing continuous K/V views. E2G and every rejected configuration retain the row-index/gather topology.
- This optimization does not turn a dynamic mapping back into a continuous view. Any feature that may invalidate identity is rejected before the context starts and uses the existing gather path for the whole context lifetime.

## Module Responsibilities

- `src/llama-model.cpp`, `src/llama-flex.*`：Dense layer 注册、ring sizing、stream/prefetch 与 compute callback 接入。
- `src/llama-moe-buffer.*`：MoE expert 匿名缓冲、resident/inflight 管理、LRU/budget、worker prefetch 和同步 callback。
- `src/llama-window.*`：window 机制及 CLG predictor；CLG buffer mode 将预测结果转发给 MoE-Buffer。
- `src/llama-kv-cache.*`：KV cell/block metadata、backing store、swap/madvise/restore/prefetch、错误状态和 telemetry。
- `src/llama-graph.cpp`：构造并填充 paged row index，将其传给 K/V attention 读取路径。
- `src/llama-context.cpp`, `src/llama-memory.h`：memory-level experimental hook 与 paged 错误向 decode/graph 状态的传播。
- `examples/kv-*`：构造 idle/resume/trace workload 和上层策略信号；不拥有 core swap 状态机。
- `scripts/run-kv-p0-*.sh`、`scripts/kv-final-controlled-e0-e5.sh` 及 parser：回归、稳定性、受控消融和证据采集；E0-E5 parser 对固定矩阵、必要 telemetry/指标、唯一性和最终 `PASS` 状态执行 fail-closed 门禁。
- `examples/kv-idle-swap-resume/idle-swap-resume.cpp`：Stage 1 的 E2 outer-segment profiler 通过已有 eval callback 链观察 cache K/V `GET_ROWS`；默认关闭，仅在 `LLAMA_KV_E2_GET_ROWS_PROFILE=1` 时启用。它预分配事件容量并对 callback 配对、K/V 完整性、节点唯一性和溢出 fail-closed。
- `src/llama-kv-cache.*`：Stage 1 的 E5 prefetch block-phase trace 默认关闭，仅在 `LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=1` 时开启；每个实际恢复 block 输出 validate/read/unpack/commit 的差值，且该开关同时开启既有 I/O stats。
- `scripts/kv-e0-e2-e5-single-turn-diagnose.sh` 与 `scripts/parse-kv-e0-e2-e5-single-turn.py`：固定 E0/E2/E5 顺序、环境去污染、artifact hash/identity、输出精确一致性、机制与安全字段、E2/E5 telemetry 对账均为 fail-closed；任一缺失、重复、失配或非零 run 均拒绝 artifact。

## Integration Points

- 模型加载阶段根据环境和模型形状选择 Flex 或 MoE-Buffer；两者不会同时启用，MoE-Buffer 还要求 mmap、MoE tensor 且非 vocab-only/check-tensors 路径。
- CPU graph compute 安装 weight-stream callback；CLG node callback 只提供预取提示。
- KV graph input 在 `llama-graph.cpp` 调用 `build_input_paged_row_idx()` / `set_input_paged_row_idx()`，attention 的 `get_k()`/`get_v()` 接收 row index。
- 上层通过 `prefetch_seq()` / `prefetch_seq_step()` 等 memory hook 表达 resume 预取；core 当前没有生产级 request scheduler。

## Invariants and Error Propagation

- 默认关闭：未显式设置相关环境变量时，不启用 Flex/MoE/CLG/paged swap/madvise/prefetch。
- backing store 容量固定为物理 KV cell 数乘每 cell 全层 K/V stride；cell offset 是固定槽位映射，重复 swap-out 不应追加增长。
- paged block 写入整块成功后才发布 offsets 和 `SWAPPED`；失败时清除待发布 metadata，不执行 madvise，block 保持可重试状态。
- swap-in 先读入 staging，再提交 tensor 与 `RESIDENT` 状态；active 必需的恢复失败记录 context-local error，并在 graph compute 前失败返回。
- 只有当前 active attention 不可见的 idle-owned block 才可进入 idle swap；write/read 路径均检查 mapping 与 resident 状态。
- `mincore`、trace 和 fault injection 是诊断/测试机制，默认关闭；它们的观测不能替代正式 correctness gate。
- E0-E5 artifact 必须与 `RUNS=1/3` 固定计划精确一致；缺失/重复/乱序 tuple、缺失必要字段或指标、重复 marker/key、以及任一非 `PASS` run 都使 parser 非零退出。dry-run 只验证规划产物，不构成模型正确性或性能证据。
- E2 `segment_ending_at_get_rows_wall_us` 的范围严格是“scheduler graph 中前一 callback 边界至目标 GET_ROWS 完成”；它是带 callback 的 outer segment，不是单个 GET_ROWS kernel、CPU backend 内层时间、单 token 时间或端到端 wall time。
- E5 的 per-block phase 值是既有累计 swap-in 阶段计数器在一次 `prefetch_seq_step()` 恢复前后的差值；parser 要求每 block 的 phase sum 与四阶段相加一致，并与 token prefetch、全局 swap-in 与累计 I/O 字段对账。它描述被关联的恢复阶段，不等同于完整 decode latency。
- Identity fast-path eligibility is immutable for a constructed context. `paged_identity_fast_path_enabled` and `paged_row_idx_enabled` select exactly one topology: continuous view without row input, or gather with row input.
- Graph reuse must preserve topology, not only tensor sizes. `llm_graph_input_attn_kv::can_reuse()` requires `(cached graph has paged_row_idx) == context uses paged_row_idx`; a mismatch rejects reuse and forces rebuild. Within the same topology, existing `n_kv` and mask/shape checks still apply.
- Enabling E2I must imply identity mapping for the full context lifetime. Dynamic remap, swap, release, madvise, non-identity/shifted mapping, approximate dynamic view, unsupported layout/layer type and injected fault scopes are fail-closed eligibility rejections, never runtime permission to keep using the continuous view.

## Modification Boundaries

- core 提供 block state、I/O、madvise、restore、prefetch 与诊断机制；session lifecycle、resume timing、pressure policy 保持在 server/application/example 层。
- Stage 1 未向 ggml CPU backend 的 `GET_ROWS` 实现加入内层计时，也未修改 scheduler；诊断挂接在 example 的可选 eval callback 链和 KV prefetch telemetry，默认路径不输出这些事件。
- 不应把 ShareGPT-backed synthetic trace 描述为真实线上 trace，也不应把 current RSS 收益描述为 peak RSS 收益。
- public/memory-level API 仍属 experimental；在没有 backend 与错误语义设计前，不视为稳定 upstream API。
- 修改 KV 状态机时必须同步核对 row mapping、active visibility、swap metadata 发布顺序和 graph 前错误传播。
- Any future feature that can change logical-to-physical mapping or residency after construction must either remain incompatible with E2I or introduce a proven invalidation/rebuild contract before relaxing the context-lifetime gate.

## Known Limitations

- 尚未验证 GPU/device-memory KV reclaim；host `madvise`/`mincore` 语义不能外推到 GPU。
- KV paged 主路径当前只接受兼容 layout（源码检查为 `n_stream == 1 && !v_trans`）；部分 non-identity probe 还限制 F32 K/V。
- KV prefetch 由上层分步触发，不是独立异步恢复线程；真实 server queue/continuous batching 尚未接入。
- 权重侧 Flex 与 MoE-Buffer 是分离路径，不是统一 shared weight/KV I/O budget scheduler。
- `src/llama-kv-cache.cpp` deferred-construction 处仍有“block swap 每 cell I/O”的旧注释；当前实现实际使用 `write_cells()`/`read_cells()` 做 block range I/O，该注释已陈旧。
- README/历史报告中的性能数字缺少仓库内原始 artifacts 与 commit/worktree 绑定，目前无法确认其对当前 HEAD 的适用性。
- Stage 1 的 E2 outer segment 含 callback/scheduler 边界，存在观测扰动；它仅用于定位。Stage 2 已改用未插桩 identity fast path 的端到端 A/B；该证据只支持当前阶段保留决策。
- Static identity fast path is intentionally narrower than paged KV: it does not apply to swap/reclaim/madvise, non-identity mapping, multi-stream, `v_trans`, approximate-dynamic views, non-F32 K/V layers, fault-injection configurations, GPU/device KV, or any context that may remap after construction.
- The current E2G/E2I evidence covers one CPU host, one F32-KV layout, one model and one fixed ctx-2048 example workload. It validates the topology and stage decision, not general backend or final-competition performance.
