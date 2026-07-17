# Architecture

> 记录当前源码可验证的稳定结构。讨论方案必须标记为 proposed，不得混入已实现架构。

- Last verified: 2026-07-17
- Evidence commit: `adfe671367f0cdc17327786c2b5c6182939cbf09`
- Verification scope: 当前源码、Git diff、clean-HEAD R0–R5/N0–N2 release correctness artifact；本轮未运行构建、测试或实验

## System Boundary

- 基础系统为 `llama.cpp` / `ggml`。项目扩展位于模型加载、CPU weight-stream callback、KV cache、attention graph 输入和独立 example/runner 层。
- 当前已核对路径主要面向 Linux CPU/host memory。KV paged 路径要求 `n_stream == 1 && !v_trans`；`madvise`/`mincore` 不适用于 GPU device memory。
- Dense Flex、MoE-Buffer/CLG、lazy KV、paged KV、destructive release 均由环境变量显式启用；普通未配置路径不主动进入这些实验机制。

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

### RELEASED destructive reclaim 路径

```text
paged_release_blocks() called from idle/resume boundary
  -> llama_kv_release_collect_ownership: per-block owned/shared bitmap
     invalid mapping → ABORT, destructive_release_skipped
  -> for each physical block:
     if owned[block]      → skip (含 shared skip 计数)
     if RELEASED          → idempotent skip
     if SWAPPED           → skip (recoverable history 保护)
     otherwise (RESIDENT or UNUSED):
       -> paged_madvise_block(MADV_DONTNEED) on K/V pages
       -> clear backing metadata (swap offsets/sizes)
       -> state → RELEASED, push to free list
       -> 计数: released_unused (state was UNUSED) 或 released_dead (state was RESIDENT, 无 live owner)

PENDING_WRITE transaction (reuse allocation 写入新 K/V):
  paged_finish_write_transaction(success=true)
    -> verify fresh writes
    -> PENDING_WRITE → RESIDENT (commit)
    -> mincore reaccess sampling
  paged_finish_write_transaction(success=false)
    -> paged_madvise_block(DONTNEED) on each pending block
    -> PENDING_WRITE → RELEASED (rollback)
    -> push to free list
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

- Eligibility is context-lifetime state, not a per-token speculation.
- Eligible E2I omits `paged_row_idx` creation/fill and bypasses K/V `GET_ROWS`, exposing continuous K/V views.
- This optimization does not turn a dynamic mapping back into a continuous view.

## KV Cache Block State Machine

### States (`paged_block_state`)

| State | Value | 含义 |
|-------|-------|------|
| UNUSED | 0 | 未分配，无 K/V 内容 |
| RESIDENT | 1 | K/V tensor 页面在 host memory 中有效驻留 |
| RELEASED | 2 | **destructive**：页面已被 MADV_DONTNEED，无 backing，旧 KV 不可恢复 |
| SWAPPED | 3 | 可恢复：页面被 madvise'd 但 backing store 持有权威副本 |
| PENDING_WRITE | 4 | 事务中间态：block 已被 release 回收并分配给新写入，但事务未提交 |

### State transition invariants

1. **RELEASED 是终态的、不可逆的**。进入 RELEASED 后，K/V tensor 页面已无权威 backing；后续 reuse 是全新分配（UNUSED→RESIDENT 的道德等价），旧内容不可恢复。
2. **只有无 live/owned cell 的 block 才能进入 RELEASED**。`llama_kv_release_collect_ownership` 在每次 release 调用前重算 ownership bitmap；任何 invalid mapping 导致 release 立即 ABORT。
3. **SWAPPED block 不得被 release**。可恢复历史（idle/shared sequence 的 offloaded block）必须经 backing store→swap-in→RESIDENT 路径恢复，不能被 RELEASED 销毁。
4. **PENDING_WRITE 事务原子性**：commit → RESIDENT；rollback → RELEASED + free list。不在中间态留下可观察的 block。
5. **Destructive release 与 swap 互斥**：`llama_kv_destructive_release_can_enable` 要求 `!swap_enabled`；启用 swap 时 release 自动禁用。

### Dummy row redirect

- 目的：在 row-index fill 阶段，将非 active visible 行从 SWAPPED/RELEASED block 重定向到 resident dummy row，避免 touch non-resident 页面（防止 refault/污染 residency 结论）。
- SWAPPED redirect：`paged_swapped_redirect_rows`/`paged_swapped_redirect_blocks` 计数。
- RELEASED redirect：`paged_released_redirect_rows`/`paged_released_redirect_blocks` 计数。
- Active visible violation：SWAPPED/RELEASED 行仍被 active seq 需要时，`paged_active_row_dummy_redirect_blocked` 阻止 redirect 并记录错误；release 侧还会将 `paged_active_release_violation` 递增。

### Ownership model

- `llama_kv_release_block_ownership`：per-block `owned[]`（是否有任何 logical cell 映射到此物理 block）和 `shared[]`（是否有 ≥2 个不同 seq 共享此 block）。
- Shared block 的 shared 状态单独计数（`paged_block_release_skip_shared`），但同样受 owned gate 保护——shared 不等于 unowned，release 不回收仍被任何 seq 引用的 block。

## Module Responsibilities

- `src/llama-model.cpp`、`src/llama-flex.*`：Dense layer 注册、ring sizing、stream/prefetch 与 compute callback 接入。
- `src/llama-moe-buffer.*`：MoE expert 匿名缓冲、resident/inflight 管理、LRU/budget、worker prefetch 和同步 callback。
- `src/llama-window.*`：window 机制及 CLG predictor；CLG buffer mode 将预测结果转发给 MoE-Buffer。
- `src/llama-kv-cache.*`：KV cell/block metadata、backing store、swap/madvise/restore/prefetch、**destructive release 状态机、ownership 收集、事务提交/回滚、dummy redirect**、错误状态和 telemetry。
- `src/llama-kv-cache-release.h`：release 准入门禁（`llama_kv_destructive_release_can_enable`）与 ownership 收集（`llama_kv_release_collect_ownership`），与核心 KV 实现分离以便静态审计。
- `src/llama-graph.cpp`：构造并填充 paged row index，将其传给 K/V attention 读取路径；**dummy row redirect 在此层发生**。
- `src/llama-context.cpp`、`src/llama-memory.h`：memory-level experimental hook 与 paged 错误向 decode/graph 状态的传播。
- `examples/kv-*`：构造 idle/resume/trace workload 和上层策略信号；不拥有 core swap/release 状态机。
- `scripts/run-kv-p0-*.sh`、`scripts/run-kv-paged-release-correctness.sh` 及 parser：P0 回归、稳定性、release correctness 协议和 fail-closed artifact 门禁。
- `tests/test-kv-paged-release-ownership.cpp`、`tests/test-kv-paged-release-correctness-parser.py`：ownership 逻辑单元测试与 parser 合成负例回归。

## Integration Points

- 模型加载阶段根据环境和模型形状选择 Flex 或 MoE-Buffer。
- CPU graph compute 安装 weight-stream callback；CLG node callback 只提供预取提示。
- KV graph input 在 `llama-graph.cpp` 调用 `build_input_paged_row_idx()` / `set_input_paged_row_idx()`，attention 的 `get_k()`/`get_v()` 接收 row index；**dummy redirect 在 row-index fill 中发生**。
- 上层通过 `prefetch_seq()` / `prefetch_seq_step()` 等 memory hook 表达 resume 预取。
- **Release 在 idle/resume boundary 由 example driver 调用**；core 不自动触发 release。

## Invariants and Error Propagation

- 默认关闭：未显式设置相关环境变量时，不启用 Flex/MoE/CLG/paged swap/madvise/release/prefetch。
- backing store 容量固定为物理 KV cell 数乘每 cell 全层 K/V stride。
- paged block 写入整块成功后才发布 offsets 和 `SWAPPED`；失败时清除待发布 metadata，block 保持可重试状态。
- swap-in 先读入 staging，再提交 tensor 与 `RESIDENT` 状态；active 必需的恢复失败记录 context-local error，并在 graph compute 前失败返回。
- `paged_release_blocks()` 每次调用前重算 ownership；invalid mapping 导致 ABORT 且 `destructive_release_skipped=1`。
- PENDING_WRITE 事务：commit 后 block 为 RESIDENT；rollback 后 block 为 RELEASED 并回收至 free list。不在中间态遗留。
- Release 过程中不跳过任何 RESIDENT block 的 madvise；只有 owned、RELEASED 和 SWAPPED block 被跳过。
- R0–R5/N0–N2 artifact 必须与 `RUN_PLAN=(R0 R1 R2 R3 R4 R5 N0 N1 N2)` 精确一致；parser 对 contract marker、安全字段、机制触发和最终 `PASS` 状态 fail-closed。
- Identity fast-path eligibility is immutable for a constructed context；graph reuse 必须 preserve topology。
- **Release 是 correctness 机制，不是压力调度策略**。当前证据验证的是选择性（不误伤）、事务原子性和互斥门禁；不覆盖回收时机、并发 request interference 或与 swap 的融合调度。

## Modification Boundaries

- core 提供 block state、I/O、madvise、restore、prefetch、release 与诊断机制；session lifecycle、release timing、pressure policy 保持在 server/application/example 层。
- 修改 `paged_block_state` 枚举或状态转换时必须同步核对：ownership collection、dummy redirect、事务提交/回滚、swap gate 和 release gate。
- 任何可能改变 post-construction mapping/residency 的新机制必须与 release 的 ownership gate 互斥或提供经证明的失效契约。
- 不应把 release correctness 协议结果描述为压力调度性能结论。

## Known Limitations

- 尚未验证 GPU/device-memory KV reclaim；host `madvise`/`mincore` 语义不能外推到 GPU。
- KV paged 主路径当前只接受兼容 layout（`n_stream == 1 && !v_trans`）；部分路径还限制 F32 K/V。
- KV prefetch 由上层分步触发，不是独立异步恢复线程；真实 server queue/continuous batching 尚未接入。
- 权重侧 Flex 与 MoE-Buffer 是分离路径，不是统一 shared weight/KV I/O budget scheduler。
- Release 当前只在 example driver 的 idle/resume boundary 调用；没有请求级压力信号触发条件 release。
- 当前证据覆盖单机 CPU、Llama-3-8B Q4_K_M、ctx 1024、parallel 4、固定 idle/resume workload。不覆盖多模型、长上下文、server/continuous batching 或不同 backend/layout。
- 历史 README 中的性能数字缺少仓库内原始 artifacts 与 commit/worktree 绑定，目前无法确认其对当前 HEAD 的适用性。
