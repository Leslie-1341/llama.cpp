# Architecture

> 记录当前源码和运行证据可验证的稳定结构。目标契约、计划和未验证机制必须明确标记，不能混入当前 runtime。

- Last verified: 2026-07-27
- Runtime evidence commits: `cffe4f5ae`、`0fe0aed12`、`a94381a31`
- Evidence scope: Stage 3A-2C historical diagnostic + Stage 3B-2A clean-HEAD long-context correctness and limited-stability artifact

## 1. Authority and Scope

### 1.1 Current runtime authority

事实优先级：当前源码和 diff、真实运行结果、四份工程账本、最新设计文档、历史文档与聊天记录、理论推测。

当前主要支持 Linux CPU/host memory。`madvise`、`mincore` 和 procfs/cgroup pressure sampling 不适用于 GPU device KV；不能把 CPU 结果外推到 GPU。

所有实验机制默认关闭，通过环境变量显式启用。普通 llama.cpp 路径不应自动进入 Flex、MoE buffer、paged swap/release、pressure sampler、dry-run 或 bounded release。

### 1.2 Frozen target contracts — not current runtime

以下文档是后续统一 lifecycle/scheduler/evidence 的目标 authority，不代表当前源码已经实现：

- `docs/kv_block_lifecycle_contract.md`：内容、驻留、事务三轴状态；per-cell overlay；唯一 commit visibility；quarantine/reset/generation。
- `docs/kv_pressure_scheduler_contract.md`：server/core/runner/parser 分层；`NOOP/EVALUATE/RELEASE/OFFLOAD/PREFETCH`；active-required prefetch 优先。
- `docs/kv_lifecycle_evidence_protocol.md`：instance/epoch/decision/transaction/block generation、事件顺序和完整闭包。

当前 Stage 3A-2C 仍使用单一 `paged_block_state`、聚合 marker 和 v4 diagnostic protocol；不得描述成完整三轴 lifecycle、统一五动作 scheduler 或完整 lifecycle v5。

## 2. System Boundary

### 2.1 Weight paths

```text
GGUF/model loader
  -> Dense Flex: decoder-layer registration -> bounded reusable buffers -> pread/prefetch
  -> MoE Buffer: expert slice registration -> resident/inflight/LRU budget -> worker prefetch
  -> CPU weight-stream callback before kernel
  -> synchronous wait/load fallback
  -> GGML CPU kernel
```

- Dense 和 MoE 的访问模式不同，因此维持分离实现。
- CLG 只提供 expert prefetch hint；真实 routing 和同步 callback 仍决定正确性。
- 当前没有统一的 weight/KV physical-memory budget 或 I/O arbiter。

### 2.2 KV paths

```text
slot allocation / KV write
  -> logical cell metadata
  -> paged logical-to-physical mapping
  -> paged_row_idx graph input
  -> K/V row gather
  -> attention
```

Paged layout 当前要求兼容 CPU host geometry，核心限制包括 single stream、非 `v_trans` 和受支持 layer/KV type。具体 capability 由 core 公开接口判定，server 不读取 private block state。

## 3. KV State and Safety

### 3.1 Current block states

| State | Meaning |
|---|---|
| `UNUSED` | 未承载有效旧 KV，可作为新写入候选 |
| `RESIDENT` | K/V 页面当前有效驻留 |
| `RELEASED` | 旧内容已 destructive discard、无 backing；状态对象可被复用于全新内容 |
| `SWAPPED` | 页面可不驻留，但 backing store 保存权威副本 |
| `PENDING_WRITE` | 新写事务已申请 block，但尚未完成当前提交 |
| `INVALID` | compute-started 失败后的隔离状态；需要显式 clear/reset |

`RELEASED` 是**旧内容不可恢复的终态**，不是 allocation object 永不复用。复用后写入的是全新 KV，不能恢复旧内容。

### 3.2 Shared destructive-release eligibility

当前 unbounded、legacy bounded、server bounded 和 dry-run 使用一致的关键安全语义：

```text
cache/context fail-stop invalid -> reject evaluation
invalid mapping during ownership collection -> ABORT, zero destructive mutation
owned/shared/active-visible block -> skip
RELEASED/SWAPPED/PENDING_WRITE/INVALID -> skip
RESIDENT or UNUSED and unowned -> candidate
```

两个 destructive 路径在每个 candidate 的 `madvise` 前重新读取 context validity 与 block state。当前 server queue 以单 scheduler owner 同步调用；ownership collection 到最后一个 `madvise` 之间没有 callback、unlock、yield 或其他 mutation owner，因此 ownership bitmap 在本次调用内稳定。

### 3.3 Unbounded release

```text
paged_release_blocks()
  -> structural gate + context fail-stop gate
  -> collect full ownership bitmap
  -> invalid mapping: abort
  -> scan physical blocks
     owned -> skip
     RELEASED/SWAPPED/PENDING_WRITE/INVALID -> skip
     recheck context + state
     -> page-aligned MADV_DONTNEED
     -> clear stale backing metadata
     -> state RELEASED
     -> unique free-list insertion
```

该路径现在已补齐 PENDING_WRITE、INVALID 和 fail-stop 门禁。历史 D-0010 中“原 unbounded 缺口未修”的描述只适用于当时提交，不代表当前状态。

### 3.4 Bounded release core

```text
paged_release_blocks_bounded_impl(target_bytes, max_scan_blocks, counters)
  -> shared structural/safety gates
  -> target=0: zero side effect
  -> max_scan=0: exhausted + full shortfall
  -> collect ownership
  -> bounded scan
     unsafe state/owned -> skip
     madvise failure -> count failure, preserve state, continue
     success -> RELEASED + free-list + counter update
  -> stop when released_bytes >= target or scan budget exhausted
  -> return released/shortfall/overshoot/scan/safety result
```

- Legacy API `paged_release_blocks_bounded()` 由 `LLAMA_KV_PAGED_RELEASE` 授权并更新 legacy counters。
- Server API `bounded_release()` 由 pressure policy 授权并更新独立 bounded counters。
- 两者共享同一实现和 safety gates；server production path 不访问 test seams。
- 释放粒度是完整物理 block，允许最多一个 block 的 overshoot。

### 3.5 Dry-run

```text
paged_release_blocks_bounded_dry_run(target_bytes, max_scan_blocks) const
  -> same structural, ownership, fail-stop and four-state skip logic
  -> compute page-aligned would-release bytes
  -X no MADV_DONTNEED
  -X no block-state/free-list/backing/counter mutation
```

Dry-run 零目标 destructive side effect 已由 `const` 接口、源码审计和 Stage 3A-2B 单请求 strace 验证；并发和长上下文尚未覆盖。

## 4. Swap, Restore and Write Transactions

```text
idle/shared recoverable block
  -> pack cell-major K/V
  -> one fixed-range write_cells()
  -> publish backing metadata only after complete write
  -> state SWAPPED
  -> optional MADV_DONTNEED

resume/active requires block
  -> read_cells() into staging
  -> validate and unpack
  -> commit tensor contents
  -> state RESIDENT
  -> graph/decode continues
```

- Backing store 使用固定 physical-cell slots，逻辑容量有界。
- Swap-out 先完成整块 I/O 再发布 metadata；部分写失败不能成为可见 SWAPPED block。
- Swap-in 先读 staging，再提交 tensor/state；active-required restore failure 通过 context-local error 在 graph compute 前失败传播。
- Destructive release 与 swap 在当前 capability 上互斥；SWAPPED block 永不进入 RELEASED。
- Current transaction states are `CLOSED/APPLIED/COMPUTE_STARTED` plus block `PENDING_WRITE/INVALID` behavior；这不是冻结目标契约中的完整 PREPARED/per-cell overlay 实现。

## 5. Static Paged Identity Fast Path

```text
context construction
  -> one-time eligibility
  -> eligible static identity mapping: continuous K/V view, no paged_row_idx
  -> otherwise: paged_row_idx + GET_ROWS gather

graph cache reuse
  -> compare row-index topology
  -> mismatch -> rebuild
```

Eligibility is context-lifetime immutable and fail-closed. Dynamic remap/swap/release/madvise configurations must fall back to gather unless an explicit invalidation contract exists.

## 6. Pressure Sampling and Server Policy

### 6.1 Sampler

```text
LLAMA_KV_PRESSURE_SAMPLER=1
  -> resolve RSS or cgroup source
  -> read RSS/cgroup and telemetry-only PSI/high
  -> validate sample and stale state
  -> NORMAL/PRESSURE/CRITICAL/RECOVERY state machine
  -> publish telemetry snapshot
```

Sampler only produces pressure state and telemetry; it does not directly mutate KV.

State invariants:

- CRITICAL threshold entry is immediate.
- Downgrade requires valid low-water hysteresis and cooldown.
- Stale sample keeps prior pressure state and cannot trigger destructive action.
- PSI only upgrades sustained main-source pressure; PSI alone does not trigger reclaim.
- Source/basis changes reset continuity counters.
- Default disabled.

### 6.2 Server scheduler ownership

`server::update_slots()` is the single scheduler owner. `maybe_sample_kv_pressure(all_idle)` is synchronous and does not create a reclaim thread.

```text
Phase A: sample/log telemetry
Phase B: optional bounded dry-run
Phase C: optional bounded destructive release
```

Dry-run and bounded release use independent config and cooldown/backoff state. They **can be enabled together**; Phase B runs before Phase C in the same scheduler call. Stage 3A-2C runner separates DRY and BOUNDED variants to obtain a clear controlled comparison.

### 6.3 Phase C gates

```text
1. bounded master switch + target > 0
2. memory object exists
3. core capability can_enable
4. telemetry not stale
5. pressure state is PRESSURE or CRITICAL
6. cooldown/backoff due
```

Only after all gates pass does server call `mem->bounded_release(target, max_scan)`.

Unsupported hybrid/recurrent wrappers inherit `bounded_release_can_enable() == false` and `paged_release_status() == not_paged`; server emits/records skip reason and does not call the default no-op release method.

### 6.4 Marker and trigger format

`kv_pressure_bounded_release` contains **32 key/value fields** after the marker name:

```text
state source stale released_bytes released_blocks blocks_scanned
blocks_skipped_owned blocks_skipped_state madvise_failures shortfall_bytes
overshoot_bytes block_scan_exhausted ownership_aborted target_bytes
max_scan_blocks legacy_enabled sample_count episode cooldown_ms
mincore_before_bytes mincore_after_bytes bounded_cnt_bytes_delta
bounded_cnt_blocks_delta can_enable cap_paged cap_ingraph cap_layers
cap_row_idx cap_swap_disabled cap_layout skipped_reason idle
```


Telemetry `trigger` is a comma-separated token set, not a single enum. Allowed tokens are `first/state/source/stale/periodic/wake_completion`; empty, duplicate or unknown tokens are invalid.

## 7. Evidence Architecture — v4 Diagnostic

### 7.1 Responsibility split

- Producer/core/server: emit raw structured facts.
- Runner: launch isolated variants, record command/env/source/binary/model/script identity, process lifecycle and raw outputs; never writes PASS.
- Parser: validates schema, identity, case isolation, required mechanism execution, response identity and cleanup; owns PASS/FAIL.

### 7.2 Release evidence layers

1. `released_bytes/released_blocks`: per-call core result.
2. `bounded_cnt_bytes_delta/blocks_delta`: independent server-observed delta of core bounded counters; **primary target-action attribution** and must match the per-call result.
3. `mincore_before/after`: independent KV resident-page observation; v4 requires valid direction and plausibility, not universal exact equality.
4. Process-wide `strace MADV_DONTNEED`: background report only; proves syscall activity exists but cannot assign all bytes/calls to bounded release.
5. Response identity and zero safety errors: correctness gate.

The Stage 3A-2C run happened to satisfy exact equality among core result, counter delta and mincore observed drop. This is a result of that run, not a universal parser requirement.

### 7.3 Current protocol limit

v4 has aggregate/sample-level correlation only. It does not provide cache instance/epoch, decision/transaction IDs, block generation or a strict event stream. It cannot be called the frozen lifecycle v5 protocol.

## 8. Harness and Agent Infrastructure

### 8.1 Stable Harness v1 evidence

Commit `fd51455b7` registered a diff-aware `scripts/os-agent/gate-runner` with 15/15 E2E evidence. It classifies changed files, maps C/C++ targets, runs mode-specific checks, writes artifacts outside the repo and emits one `OS_AGENT_GATE_RESULT` marker.

Stable mode behavior:

| Mode | Main purpose | C/C++ build | clang-tidy | Parser/skill checks |
|---|---|---:|---:|---:|
| implement | implementation gate | yes | yes when available | yes |
| review | diff review gate | yes | yes when available | yes |
| review-fix | targeted fix gate | yes | no | yes |
| audit | read-only classification/mapping | no | no | yes |

### 8.2 Newer snapshot state — evidence insufficient

The uploaded source snapshot additionally contains:

- `EXIT_UNVERIFIED=5` and final priority `FAIL > UNRESOLVED > UNVERIFIED > INCOMPLETE > PASS > NO_CHANGES`;
- marker field `unverified=<n>`;
- known-fail selection logic and E2E-16–E2E-18 tests.

These are visible in source, but the supplied ledgers do not contain a commit-bound full Harness result for that newer state. Until the planned Skill/Harness task completes review and self-tests, they remain **implemented but evidence insufficient**, not a stable Harness v2 claim.

## 8.3 Stage 3B-2A clean-HEAD long-context evidence lifecycle

`run-kv-bounded-release-stage3b-2a.py` 将正式验证固定为 fail-closed 的四段生命周期：

```text
effective-context probe (server stderr effective n_ctx)
  -> exact token calibration (each legal tier)
  -> independent RSS calibration (fresh real llama-server PID/RSS per tier)
  -> OFF/DYNAMIC_RELEASE ladder + continuous DYNAMIC_RELEASE requests
  -> runner artifact facts / parser schema, identity, action, response and cleanup verdict
```

- Probe 从 server 实际 slot `n_ctx` 推导可发送 prompt 上限；请求档位必须 clamp 到有效档位，不能以 context overflow 得到假 PASS。
- Token calibration 与 RSS calibration 独立：前者通过 `/tokenize` 记录 exact prompt count，后者为每个有效档位启动真实 `llama-server`、记录 `(pid, starttime, cmdline)` 与 RSS 后再派生该档阈值/timeout。
- Ladder 的 `OFF` 使用安全阈值且不配置 bounded action；`DYNAMIC_RELEASE` 使用动态 target 与触发阈值。每档均要求 HTTP 200 和与该档 OFF baseline 的 response identity。
- `DYNAMIC_RELEASE` marker 的 action 与 safe zero-release no-op 分开：有正 `released_bytes` 时须有有效、方向正确的 KV `mincore` drop；若候选均 owned/shared/active-visible，则 `released_bytes=0` 是允许的安全 no-op，不得伪报 action。
- RSS 只能从真实 `llama-server` PID 采样，不能取 `strace` wrapper；连续请求还要求跨 round PID identity 一致、20 个 HTTP 200、零累计错误和 response identity。
- Runner 只持久化输入、身份、原始事实和首个失败 phase/target/reason；任一必需目录、文件、身份、上下文、action/no-op、PID/RSS、响应或 cleanup 不满足，parser 以 fail-closed 方式拒绝 artifact。仅 parser 写 PASS/FAIL。

## 9. Module Responsibilities

- `src/llama-flex.*` / weight-stream files: Dense layer registration, bounded buffers, prefetch and compute callback.
- `src/llama-moe-buffer.*`, `src/llama-window.*`: expert residency/prefetch and CLG hinting.
- `src/llama-kv-cache.*`: KV metadata, mapping, backing I/O, swap/restore, release/dry-run, transactions, dummy redirect, counters and error state.
- `src/llama-kv-cache-release.h`: release capability/result/ownership helpers shared by core and server-facing API.
- `src/llama-kv-pressure.*`: pressure source parsing, sampling and state transitions only.
- `tools/server/server-kv-pressure.*`: pressure cadence, dry-run/bounded config, cooldown/backoff, event formatting.
- `tools/server/server-context.cpp`: single-owner lifecycle and Phase A/B/C invocation.
- `src/llama-memory.h`: memory-level capability and action virtual interfaces.
- `src/llama-graph.cpp`: paged row-index input and dummy row redirect.
- `scripts/run-*`, `scripts/parse-*`, `tests/test-*-parser.py`: stage-local runner/parser and negative fixtures.
- `scripts/os-agent/`: agent gate harness.

## 10. Current Invariants

- Experimental mechanisms are opt-in and fail-closed.
- Server policy chooses whether/when/how much; KV core owns final candidate resolution and mutation.
- Server never selects or mutates private physical block state directly.
- Owned/shared/active-visible KV must not be destructively released.
- SWAPPED history is recoverable and must not be converted to RELEASED.
- PENDING_WRITE and INVALID are not release candidates.
- Fail-stop context cannot continue release or decode until explicit reset.
- Invalid mapping aborts destructive release before mutation.
- A failed `madvise` leaves that block state unchanged and is counted.
- Dry-run performs no target destructive action.
- Runner does not write verdict; parser is fail-closed authority.
- System observation is not state-transition truth.
- Stage-local gate PASS does not replace real-model or formal performance evidence.

## 11. Known Limitations

- No GPU/device-memory reclaim evidence.
- No multi-stream or `v_trans` support for the current paged release path.
- No complete PagedAttention block-table design; current implementation is lightweight paged row mapping for edge CPU runtime.
- No unified weight/KV memory budget or I/O arbiter.
- No active-access/resume asynchronous prefetch thread in server scheduler.
- Stage 3B-2A 已在单模型、单次协议中覆盖有效 8064-token 档位与同一 server 的 20 次连续请求；仍无并发、多模型/quantization、不同 block/page size、长期重复 release/refault 的证明。
- DYNAMIC target 与 per-tier calibrated RSS 的正确性证据不等于生产阈值或性能策略；Stage 3A-2C 的 forced CRITICAL/fixed target 仅保留为历史 diagnostic。
- `mincore`、RSS 和 strace 是诊断/正确性观测；它们会带来开销，正式性能比较必须关闭或单独测量。
- Current target contracts for three-axis lifecycle, unified five-action scheduler and v5 evidence remain unimplemented.

## 12. Next Architecture Gate

Stage 3B-2A 已关闭单模型、单次 long-context/repeated-request correctness 门禁。下一架构门禁是以 clean-HEAD artifacts 扩展到真实压力、并发、多模型/quantization 和重复 release/refault，并在每个正释放 case 维持 mincore drop、在 owned-only case 维持安全 no-op。仅在这些 KV 正确性和性能矩阵稳定后，再接入 release/offload/prefetch 与权重的共享预算和 I/O 仲裁。
