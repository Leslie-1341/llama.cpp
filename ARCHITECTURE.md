# Architecture

> 记录当前源码可验证的稳定结构。讨论方案必须标记为 proposed，不得混入已实现架构。

- Last verified: 2026-07-21
- Evidence commit: `fd51455b79af08f629810621578965e772ce0685`
- Verification scope: 当前源码、Git history（`02f8cd5ed`→`fd51455b7` diff）、Harness v1 实现（`scripts/os-agent/` 全部文件）、test-harness.sh 15/15 E2E PASS、audit gate verdict PASS、四模式 marker 验证

## System Boundary

- 基础系统为 `llama.cpp` / `ggml`。项目扩展位于模型加载、CPU weight-stream callback、KV cache、attention graph 输入、server scheduler 和独立 example/runner 层。
- 当前已核对路径主要面向 Linux CPU/host memory。KV paged 路径要求 `n_stream == 1 && !v_trans`；`madvise`/`mincore` 不适用于 GPU device memory。
- Dense Flex、MoE-Buffer/CLG、lazy KV、paged KV、destructive release、pressure sampler 和 **pressure-driven dry-run** 均由环境变量显式启用；普通未配置路径不主动进入这些实验机制。
- Pressure sampler 已编入 `llama` 库，并在 server scheduler 的 `update_slots()` 中以单 owner、限频方式接入。**Stage 3A-2B 新增 pressure-driven dry-run scanner 在同一 scheduler 上下文中以只读方式运行**；destructive reclaim 尚未接入 server。
- **Harness v1**（`scripts/os-agent/gate-runner`）是 diff-aware agent gate 框架——仅在显式调用时运行，不 hook 到 git、build 或 editor；artifact 写入 `/tmp/os-agent-gate/`（repo 外）。

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

### Bounded release 路径（`paged_release_blocks_bounded`）

```text
paged_release_blocks_bounded(target_bytes, max_scan_blocks)
  target=0 → return empty result (zero side effects, seam reset)
  max_scan=0 → return exhausted + full shortfall (zero scan)
  → llama_kv_release_collect_ownership: per-block owned/shared bitmap
    invalid mapping → ABORT, ownership_aborted=true, zero state changes
    test_force_ownership_abort seam → ABORT (single-shot, auto-reset)
  → for physical_block in [0, min(max_scan_blocks, paged_n_blocks)):
    if owned[block] → blocks_skipped_owned++, continue
    resolve state via test_block_state_override seam (if set) else real state
    if PENDING_WRITE | SWAPPED | RELEASED → blocks_skipped_state++, continue
    (RESIDENT or UNUSED):
      test_madvise_fail_block seam → inject failure, no DONTNEED
      → paged_madvise_block(MADV_DONTNEED) on K/V pages
      madvise fails → madvise_failures++, continue (state unchanged, scan continues)
      madvise succeeds:
        → clear backing metadata (swap offsets/sizes)
        → state → RELEASED, push to free list
        → released_bytes += advised_bytes, released_blocks++
        → update global counters (released_unused / released_dead)
    if released_bytes >= target_bytes → stop (one-block overshoot allowed)
  → result: released_bytes, shortfall_bytes, overshoot_bytes, blocks_scanned,
    blocks_skipped_owned, blocks_skipped_state, madvise_failures,
    block_scan_exhausted, ownership_aborted
  → ALL test-only seams auto-reset (single-shot guarantee)
```

- `target_bytes` 控制单次调用可释放的字节预算；`released_bytes` 可能因单 block 粒度 overshoot。
- `max_scan_blocks` 控制最大扫描 block 数，在 remaining 全为 owned/RELEASED/SWAPPED/PENDING_WRITE 时产生 shortfall。
- **PENDING_WRITE gate 是相对于原 `paged_release_blocks()` 的防御增强**。
- ownership ABORT 路径严格零 state change。
- Bounded release 当前 **未接入 server scheduler 的 destructive reclaim 路径**（仅 dry-run 已接入）。

### Dry-run bounded release 路径（`paged_release_blocks_bounded_dry_run`）— Stage 3A-2B 新增

```text
paged_release_blocks_bounded_dry_run(target_bytes, max_scan_blocks) const
  → 前置条件（fail-silent）：
    !kv_paged_enabled → return empty
    v_trans || n_stream!=1 || block_size==0 || n_blocks==0 → return empty
  target=0 → return empty (zero side effects)
  max_scan=0 → return exhausted + full shortfall (zero scan, zero ownership check)
  → llama_kv_release_collect_ownership: per-block owned/shared bitmap
    invalid mapping → return ownership_aborted=true (零 state change)
  → for physical_block in [0, min(max_scan_blocks, paged_n_blocks)):
    if owned[block] → blocks_skipped_owned++, continue
    resolve state from paged_block_states[block]
    if PENDING_WRITE | SWAPPED | RELEASED → blocks_skipped_state++, continue
    (RESIDENT or UNUSED) — candidate:
      → walk each layer's K/V tensors, compute page-aligned would-release bytes
        (same page-alignment + block-boundary neighbor-protection logic as destructive)
      → block_bytes == 0 → skip (page-sharing prevented all pages)
      → released_blocks++, released_bytes += block_bytes
    if released_bytes >= target_bytes → stop (one-block overshoot allowed)
  → blocks_scanned, block_scan_exhausted, shortfall/overshoot computed
  → result returned as llama_kv_bounded_release_result (would-release semantics)
  -X NO paged_madvise_block() call — 零 state mutation
  -X NO paged_block_states modification
  -X NO paged_free_list modification
  -X NO backing-metadata clear
  -X NO global-counter increment
  -X NO test-only seam access
  → const 方法，编译器强制零 mutation
```

- Dry-run 的 would-release 字节计算与 destructive 路径的 page-alignment 逻辑完全一致：`(lo_a + pg - 1) & ~(pg - 1)` 到 `hi_a & ~(pg - 1)` 的对齐区间。block 边界共享页面被自动排除。
- Dry-run scanner **不检查 `LLAMA_KV_PAGED_RELEASE`**——只需 paged KV enabled + valid layout + !swap。因此即使 destructive release 因 `LLAMA_KV_PAGED_RELEASE != 1` 而 disabled，dry-run 仍可评估 would-release 候选。
- Dry-run **不接入 test-only seam**（`force_ownership_abort`、`madvise_fail_block`、`block_state_override` 仅作用于 destructive `paged_release_blocks_bounded()`）。

### Pressure sampler → server telemetry → dry-run 路径（Stage 3A-2B 新增）

```text
server init:
  -> kv_pressure_sampler_environment_enablement()  [master switch]
  -> server_kv_pressure_config_from_env()           [cadence config]
  -> server_kv_pressure_dry_run_config_from_env()   [dry-run config — 独立解析]
  -> kv_pressure_sampler::init(enablement)          [source 解析、cgroup 路径]
  -> server_kv_pressure_runtime::enable(config)     [重置 counters/deadlines/state]
  -> server_kv_pressure_runtime::dry_run_enable()   [重置 cooldown/state-entry]

server sleep:
  -> kv_pressure_sampler_owner.reset()              [destroy sampler]
  -> kv_pressure_runtime.disable()                  [清除 state/deadlines]
  -> kv_pressure_runtime.dry_run_disable()          [清除 cooldown state]

server resume:
  -> init_kv_pressure_sampler()                     [全新 lifecycle，含 dry-run config 重新解析]

server update_slots() — single-threaded scheduler owner:
  -> maybe_sample_kv_pressure(all_idle)
     // Phase A: telemetry sampling (unchanged from 3A-1C)
     if (!sampler_owner) return;
     if (!runtime.sample_due(now)) { skip_count++; return; }
     -> sampler_owner->sample()
     -> runtime.record_sample(now, idle, sampler_owner->telemetry())
     -> if event.should_log(): SRV_INF marker

     // Phase B: dry-run bounded release evaluation (3A-2B 新增) — read-only
     should_evaluate = false
     // Gate 1: master switch
     if (!dry_run_config.enabled || target_bytes==0) → skip (zero markers)
     // Gate 2: memory exists
     else if (!ctx_tgt || !llama_get_memory(ctx_tgt)) → skip_reason="no_memory"
     else:
       // Gate 3: release capability check
       release_status = llama_get_memory(ctx_tgt)->paged_release_status()
       switch (release_status):
         available     → release_enabled=true
         disabled      → release_enabled=false (observational, not a skip)
         swap_enabled  → skip_reason="swap_enabled"
         not_paged     → skip_reason="not_paged"
         layout_unsupported → skip_reason="layout_unsupported"
       // Gate 4: stale rejection
       if (!skip_reason && telemetry.stale) → skip_reason="stale"
       // Gate 5: trigger-state gate
       if (!skip_reason && state != PRESSURE && state != CRITICAL)
         → no evaluation, no marker (NORMAL/RECOVERY 不触发)
       // Gate 6: cooldown/backoff time gate
       else if (!skip_reason)
         should_evaluate = dry_run_due(now, state, stale)

     if should_evaluate:
       result = mem->bounded_release_dry_run(target_bytes, max_scan_blocks)
       had_shortfall = result.shortfall_bytes > 0 && result.block_scan_exhausted
       dry_run_record(had_shortfall, now)  // 推进 cooldown/backoff
       skip_reason = "none"

     if skip_reason:
       emit kv_pressure_dry_run marker
       ownership_aborted → SRV_WRN (escalate)
       else → SRV_INF
```

- **Dry-run 与 telemetry 共享同一个 scheduler 单线程 owner**：`maybe_sample_kv_pressure()` 在 telemetry 采样后立即执行 dry-run 评估，不创建额外线程或锁。
- **`should_evaluate` 显式门控**：6 道独立 gate，任意一道不满足即不调用 scanner。NORMAL/RECOVERY 状态下不输出 marker（不是 "skipped"，而是评估不发生）。只有触发评估但 skip 时才产生 marker（含 `skipped_reason`），或评估完成后产生 marker（含 `would_release_*` 字段）。
- **Dry-run config 独立于 sampler init**：即使 sampler init 失败，dry-run config 仍可解析；但当前 dry-run 读取 sampler 的 pressure state，因此 sampler 必须初始化成功才能触发评估。
- **Cooldown/backoff 策略**：
  - 首次进入 PRESSURE/CRITICAL（state-entry）：立即评估一次，重置 cooldown timer。
  - CRITICAL entry：绕过 cooldown 进行首次评估；后续 sustained CRITICAL 评估仍受 cooldown 限制。
  - 同一 state episode 内：`elapsed >= cooldown_ms` 才允许再次评估。
  - `dry_run_record()` 后：无 shortfall → 回 base cooldown；有 shortfall + scan exhausted → 延长到 backoff_ms。
  - 状态退出（PRESSURE→NORMAL 等）：下一轮 `dry_run_due()` 中的 `state_entered` 再次重置。
  - Stale 清空时：`dry_run_due()` 返回 false（stale 拒绝先于 cooldown 检查）。
- **Sleep/resume 完全隔离**：sleep 时 `dry_run_disable()` 清除 config、cooldown state 和 state-entry tracking；resume 时全新解析 env 并重新 enable。

### `llama_kv_release_status` 枚举与 `paged_release_status()`

```text
enum class llama_kv_release_status {
    available,           // 可执行 release: paged + ingraph + layers + row_idx + !swap
    not_paged,           // KV paging 未启用
    layout_unsupported,  // v_trans || n_stream!=1 || block_size==0
    swap_enabled,        // swap 与 release 互斥
    disabled,            // paged_block_release_enabled 为 false
};
```

- `paged_release_status()` 快速路径：`paged_block_release_enabled==true` → 立即返回 `available`（该 bool 已编码完整的 `llama_kv_destructive_release_can_enable` 检查）。
- 慢速路径（dry-run 所需）：逐项检查 paged KV 前提、layout、swap 互斥，返回精确原因。`disabled` 表示前提满足但 `LLAMA_KV_PAGED_RELEASE!=1`——对 dry-run 是观测性而非 skip reason。
- 用于 server policy 区分"不能 release"（hard skip）与"未启用 release"（observational），替代旧版单一 boolean `can_enable` 过载。

### Sampler-only 核心路径（`llama` 库内）

```text
LLAMA_KV_PRESSURE_SAMPLER=1 + validated thresholds
  -> resolve one unambiguous cgroup v1/v2 memory path
  -> read selected primary source
     - RSS_ABSOLUTE: /proc/self/statm
     - CGROUP_ABSOLUTE: memory.current / memory.usage_in_bytes
     - CGROUP_RATIO: memory.current ÷ finite memory.max
  -> read telemetry-only memory.high and PSI
     - /proc/pressure/memory
     - cgroup v2 memory.pressure
     - PSI upgrade signal uses max(system, cgroup)
  -> validate selected-source sample
  -> evaluate NORMAL/PRESSURE/CRITICAL/RECOVERY
  -> publish read-only kv_pressure_telemetry
  -X no reclaim / swap / prefetch / bounded-store / dry-run action
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

## Memory Pressure State Machine

### States (`kv_pressure_state`)

| State | 含义 |
|-------|------|
| NORMAL | 主压力源低于 pressure threshold；默认关闭时固定返回此状态 |
| PRESSURE | 主压力源连续达到 pressure threshold，且满足滞回与 cooldown |
| CRITICAL | 达到 critical threshold 时立即进入；也可由持续 PRESSURE + PSI 滞回升级 |
| RECOVERY | PRESSURE/CRITICAL 连续低于 low-water 且满足滞回与 cooldown 后进入的恢复中间态 |

### Transition invariants

1. **CRITICAL entry 不受 cooldown 或普通滞回阻塞**：NORMAL、PRESSURE、RECOVERY 达到 critical threshold 时立即进入 CRITICAL。
2. **降级必须 fail-closed**：PRESSURE/CRITICAL 只有连续有效样本低于 low-water，并满足 hysteresis + cooldown，才进入 RECOVERY；RECOVERY 再次满足独立 low-water hysteresis + cooldown 才回 NORMAL。
3. **stale 不自动降级**：选中 source 读取失败时立即标记 stale，清空 pressure/low-water/PSI 连续计数，保持原状态。
4. **PSI 不是独立 destructive trigger**：PSI 只在主 source 仍高于 pressure 且状态已为 PRESSURE 时，经过连续样本升级到 CRITICAL；PSI 单独升高不改变 NORMAL。
5. **source/basis 切换隔离生命周期**：source、阈值、cgroup unlimited 标志或 ratio maximum 改变时重置累计计数；PSI 升级计数也只属于一次连续 PRESSURE 生命周期。
6. **默认关闭且 sampler/telemetry-only**：`LLAMA_KV_PRESSURE_SAMPLER` 未显式启用时 `sample()` 返回 NORMAL；server runtime 不创建 sampler。状态机不拥有或调用 reclaim。
7. **Dry-run 仅在 PRESSURE/CRITICAL 状态触发**（Stage 3A-2B 新增）：NORMAL/RECOVERY 状态不产生 dry-run 评估或 marker。此 gate 在 `should_evaluate` 门控的第 5 道检查执行。

## Server Pressure Runtime State Machine

### server_kv_pressure_runtime lifecycle

| 事件 | 行为 |
|------|------|
| `enable(config)` | 重置 enabled=true, sample_count=0, skip_count=0, next_sample_=epoch, last_log_=epoch, last_state_=NORMAL, last_source_=NONE, last_stale_=false |
| `disable()` | enabled=false, 清空所有 counters/deadlines/state 快照 |
| `sample_due(now)` | enabled=false → false；首次调用 → true；now < next_sample_ → skip_count++ & false；否则 → true |
| `record_sample(now, idle, telemetry)` | sample_count++，计算 first_sample/state_changed/source_changed/stale_changed/periodic，刷新 next_sample_ 和 last_* 快照，可能刷新 last_log_ |

### Dry-run lifecycle（Stage 3A-2B 新增）

| 事件 | 行为 |
|------|------|
| `dry_run_enable(cfg)` | 设置 dry_run_config_, 重置 last_dry_run_=epoch, last_dry_run_state_=NORMAL, current_cooldown_ms_=cfg.cooldown_ms |
| `dry_run_disable()` | dry_run_config_.enabled=false, 清空 cooldown state |
| `dry_run_due(now, state, stale)` | enabled=false or target=0 → false; stale → false; state ∉ {PRESSURE, CRITICAL} → false; state-entry (state changed) → reset cooldown + return true; otherwise → elapsed >= current_cooldown_ms_ |
| `dry_run_record(had_shortfall, now)` | 记录 last_dry_run_=now; had_shortfall → current_cooldown_ms_=backoff_ms; else → current_cooldown_ms_=cooldown_ms |

### Logging triggers (`should_log()`)

- `first_sample`：首个样本
- `state_changed`：状态转换（NORMAL↔PRESSURE↔CRITICAL↔RECOVERY）
- `source_changed`：主 source 切换
- `stale_changed`：stale 标记变化
- `periodic`：距上次 log ≥ `log_interval`（默认 60s，最小 1s）

不满足任一 trigger 时，sample 仍完成但不输出 marker。

**Dry-run marker 输出**：独立于 telemetry marker——仅当 `skip_reason != nullptr`（评估完成或明确跳过）时输出。NORMAL/RECOVERY 状态下不输出任何 marker。

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
6. **Dry-run 零 state change**（Stage 3A-2B 新增）：`paged_release_blocks_bounded_dry_run()` 为 `const` 方法，不修改 `paged_block_states`、`paged_free_list`、backing metadata 或 global counters。RELEASED/SWAPPED/PENDING_WRITE block 在 state gate 中被跳过（与 destructive 路径一致），但不会被推进到新状态。

### Dummy row redirect

- 目的：在 row-index fill 阶段，将非 active visible 行从 SWAPPED/RELEASED block 重定向到 resident dummy row，避免 touch non-resident 页面（防止 refault/污染 residency 结论）。
- SWAPPED redirect：`paged_swapped_redirect_rows`/`paged_swapped_redirect_blocks` 计数。
- RELEASED redirect：`paged_released_redirect_rows`/`paged_released_redirect_blocks` 计数。
- Active visible violation：SWAPPED/RELEASED 行仍被 active seq 需要时，`paged_active_row_dummy_redirect_blocked` 阻止 redirect 并记录错误；release 侧还会将 `paged_active_release_violation` 递增。

### Ownership model

- `llama_kv_release_block_ownership`：per-block `owned[]`（是否有任何 logical cell 映射到此物理 block）和 `shared[]`（是否有 ≥2 个不同 seq 共享此 block）。
- Shared block 的 shared 状态单独计数（`paged_block_release_skip_shared`），但同样受 owned gate 保护——shared 不等于 unowned，release 不回收仍被任何 seq 引用的 block。
- **Dry-run scanner 复用同一 ownership collection 逻辑**：与 destructive 路径完全一致的 bitmap 计算和 ABORT 语义。

## Agent Infrastructure（Harness v1）

Harness v1 是一个 diff-aware 的 agent gate 框架，位于 `scripts/os-agent/`，为 implement/review/review-fix/audit 四种工程模式提供自动化的变更验证门禁。它在 repo 根通过 `gate-runner <mode>` 调用，输出唯一结构化 marker `OS_AGENT_GATE_RESULT`。

### Harness 架构

```text
gate-runner <mode>
  -> diff_analyze()              [diff-analyzer.sh]
     -> git diff --name-only HEAD  (tracked: MADRC)
     -> git diff --name-only --diff-filter=D HEAD  (deleted)
     -> git ls-files --others     (untracked, excluding artifacts+fixtures)
     -> _classify_files(): ext-based 分类 → DIFF_CLASSES[HAS_*]
     -> diff_report(): 结构化 DIFF_* 输出到 artifact
  -> run_checks_for_mode <mode>   [checks/run-checks.sh]
     -> gate_implement | gate_review | gate_review_fix | gate_audit  [gates/define-gates.sh]
        -> check_git_diff_check()    (trailing whitespace, conflict markers)
        -> check_shell_syntax()      (bash -n 对所有变更 .sh)
        -> check_python_syntax()     (python3 -m py_compile)
        -> target_mapper_init()      (compile_commands.json → verified targets)
        -> check_cmake_configure()   (cmake --build --target help)
        -> check_compile_commands()  (compile_commands.json 存在且覆盖变更 .cpp)
        -> check_clang_tidy()        (clang-tidy --warnings-as-errors，无工具时 SKIP)
        -> check_incremental_build() (cmake --build，仅编译变更 target)
        -> check_parser_test()       (Python parser 合成负例)
        -> check_skill_validation()  (.claude/skills + .agents/skills 一致性)
        -> check_memory_check()      (工程账本验证)
  -> gate_final_verdict()         [common.sh]
     -> 优先级: FAIL > UNRESOLVED > INCOMPLETE > PASS > NO_CHANGES
  -> gate_emit_summary <code>     [common.sh]
     -> 唯一 OS_AGENT_GATE_RESULT marker + summary.txt
```

### 四种模式

| 模式 | 用途 | C/C++ build | clang-tidy | parser test |
|------|------|-------------|------------|-------------|
| `implement` | 实现完成后的完整门禁 | incremental build | ✓ | ✓ |
| `review` | diff 审查 | incremental build | ✓ | ✓ |
| `review-fix` | 按审查结论修复后 | incremental build | ✗ | ✓ |
| `audit` | 只读分析（分类、语法、mapping） | target resolution only | ✗ | ✓ |

### Target Mapping

- `.cpp/.c/.cc` 源文件：从 `compile_commands.json` 的 `-o CMakeFiles/<target>.dir/` 提取 target。
- `.h/.hpp` 头文件：按目录 umbrealla mapping——`src/`→`llama`、`tools/server/`→`llama-server`、`common/`→`llama-common`、`ggml/src/`→`ggml`、`examples/kv-*`→对应 example target。
- 所有 resolved target 必须通过 `cmake --build <dir> --target help` 验证存在。
- Mapping 失败 → UNRESOLVED (code=2)，不阻塞但需人工确认。

### Unique Marker

```
OS_AGENT_GATE_RESULT mode=<mode> verdict=PASS|FAIL|UNRESOLVED|INCOMPLETE|NO_CHANGES code=<0-4> checks=<n> pass=<p> fail=<f> skip=<s> unresolved=<u> artifact=<path>
```

- 全局唯一输出点：`gate_emit_summary()`（`common.sh:80-115`）。
- Artifacts 写入 `/tmp/os-agent-gate/gate-<mode>-<timestamp>/`（仓库外，防自污染），包含 `full.log` 与 `summary.txt`。
- Exit codes: 0=PASS 1=FAIL 2=UNRESOLVED 3=INCOMPLETE 4=NO_CHANGES。

### Harness Self-tests

- `scripts/os-agent/tests/test-harness.sh`：15 个合成 E2E 测试，覆盖：
  - E2E-1: NO_CHANGES（clean repo → code 4）
  - E2E-2: audit PASS（只读分析通过）
  - E2E-3: Python 语法 FAIL 检测
  - E2E-4: Shell 语法 FAIL 检测
  - E2E-5: Untracked 文件检测
  - E2E-6: Deleted 文件追踪
  - E2E-7: C++ target mapping（compile_commands.json -o flag）
  - E2E-8: Multi-target mapping（同一源文件多 target）
  - E2E-9: UNRESOLVED cpp（未知文件无 target）
  - E2E-10: 真实 incremental build（fixture repo + cmake）
  - E2E-11: Artifact 目录在 repo 外（防自污染）
  - E2E-12: 单一 summary marker（无重复）
  - E2E-13: Parser test UNRESOLVED
  - E2E-14: implement/review/review-fix/audit 四模式 dispatch
  - E2E-15: --build-dir flag 正确转发
- 已确认：**15/15 E2E PASS**（运行于 HEAD `fd51455b7`）。

### Harness Invariants

- **默认不运行**：只有显式调用 `gate-runner` 时才执行；不 hook 到 git、build 或 editor。
- **Fail-closed marker**：唯一 marker 由 `gate_emit_summary()` 生成；不会在 stdout 出现第二个 `OS_AGENT_GATE_RESULT` 行。
- **Artifact 防自污染**：artifact 目录 `/tmp/os-agent-gate/` 在 repo 外，diff-analyzer 自动排除该前缀路径。
- **Synthetic fixture 隔离**：harness self-test 在 `/tmp/os-agent-gate-test-*/` 临时目录创建 fixture repos，exit 时 trap cleanup。
- **模式不自行扩大范围**：audit 不执行 build，review 不执行 clang-tidy（仅 implement）。

## Module Responsibilities

- `src/llama-model.cpp`、`src/llama-flex.*`：Dense layer 注册、ring sizing、stream/prefetch 与 compute callback 接入。
- `src/llama-moe-buffer.*`：MoE expert 匿名缓冲、resident/inflight 管理、LRU/budget、worker prefetch 和同步 callback。
- `src/llama-window.*`：window 机制及 CLG predictor；CLG buffer mode 将预测结果转发给 MoE-Buffer。
- `src/llama-kv-cache.*`：KV cell/block metadata、backing store、swap/madvise/restore/prefetch、**destructive release 状态机、bounded release（destructive + dry-run 双变体）、ownership 收集、事务提交/回滚、dummy redirect**、错误状态和 telemetry。
- `src/llama-kv-cache-release.h`：release 准入门禁（`llama_kv_destructive_release_can_enable`）、ownership 收集（`llama_kv_release_collect_ownership`）、**`llama_kv_release_status` 枚举、`llama_kv_bounded_release_result` struct（提升为全局类型，供 core 与 server 共享）**。
- `src/llama-kv-pressure.*`：Linux RSS/cgroup/PSI 直接读取、严格解析、cgroup v1/v2 路径解析、source 选择、stale 处理、四态状态机和结构化 telemetry；不包含 reclaim policy 或 KV block 操作。
- `tools/server/server-kv-pressure.*`：**server-side pressure runtime**——`server_kv_pressure_runtime` 管理 sampling deadline、skip/event 计数、状态快照、结构化 marker 输出、**dry-run cooldown/backoff 状态机**；`server_kv_pressure_dry_run_config` 与 `server_kv_pressure_dry_run_event` 类型；不包含 procfs 读取或状态转换逻辑。
- `tools/server/server-context.cpp`：**single-threaded scheduler owner**——`maybe_sample_kv_pressure()` 在 `update_slots()` 中调用（含 Phase A telemetry + Phase B dry-run）、`init_kv_pressure_sampler()` 在 server init 和 resume 中创建 sampler + dry-run lifecycle。
- `scripts/probe-pressure-inputs.sh`、`tests/test-probe-pressure-inputs.sh`：Stage 3A-1A 只读输入探测与静态回归。
- `tests/test-kv-pressure-sampler.cpp`：fixture/synthetic 状态机、解析、source 切换、stale、溢出与 cgroup 路径回归。
- `tests/test-server-kv-pressure.cpp`：server runtime（含 dry-run config 解析、cooldown/backoff、state-entry、skip-reason 路径）。
- `tests/test-server-kv-pressure-static.py`：静态集成检查（含 dry-run decoupling、marker field schema、config isolation）。
- `tests/test-server-kv-pressure-stage3a-1c-parser.py`：9 parser 合成负例（3A-1C telemetry）。
- `tests/test-kv-dry-run-stage3a-2b-parser.py`：**dry-run parser 合成负例**——marker field schema、OFF/ON isolation、response divergence、destructive release contamination、MADV_DONTNEED leakage、config isolation。
- `scripts/run-server-kv-pressure-stage3a-1c.py`、`scripts/parse-server-kv-pressure-stage3a-1c.py`：Stage 3A-1C server validation protocol。
- `scripts/run-kv-dry-run-stage3a-2b.py`、`scripts/parse-kv-dry-run-stage3a-2b.py`：**Stage 3A-2B dry-run OFF/ON controlled A/B runner 与 fail-closed parser**。
- `src/llama-graph.cpp`：构造并填充 paged row index，将其传给 K/V attention 读取路径；**dummy row redirect 在此层发生**。
- `src/llama-context.cpp`、`src/llama-memory.h`：memory-level experimental hook（含 **`bounded_release_dry_run()` 与 `paged_release_status()` virtual 接口**）与 paged 错误向 decode/graph 状态的传播。
- `examples/kv-*`：构造 idle/resume/trace workload 和上层策略信号；不拥有 core swap/release 状态机。
- `scripts/run-kv-p0-*.sh`、`scripts/run-kv-paged-release-correctness.sh` 及 parser：P0 回归、稳定性、release correctness 协议和 fail-closed artifact 门禁。
- `tests/test-kv-paged-release-ownership.cpp`、`tests/test-kv-paged-release-correctness-parser.py`：ownership 逻辑单元测试与 parser 合成负例回归。
- `tests/test-kv-paged-release-bounded.cpp`：**bounded release 正确性测试**（CTest #29）——Part A 无模型 ownership fault fixture（4 组）；Part B 需模型（B1–B11）。
- `scripts/os-agent/gate-runner`：**gate harness 单入口**——diff 分析 → target mapping → mode-specific checks → unique marker。
- `scripts/os-agent/lib/common.sh`：**shared constants**——exit codes（0–4）、gate_log、gate_record_check、gate_final_verdict、gate_emit_summary。
- `scripts/os-agent/lib/diff-analyzer.sh`：**变更检测**——ext-based 分类（cpp/c/h/py/sh/cmake/skill/ledger + deleted variants），设置 HAS_* flags 驱动 check dispatch。
- `scripts/os-agent/lib/target-mapper.sh`：**C/C++ target mapping**——compile_commands.json -o flag 提取 + UMBRELLA_MAP header 解析 + cmake --target help 验证。
- `scripts/os-agent/lib/artifact.sh`：**artifact 管理**——/tmp/os-agent-gate/ 目录创建、full.log/summary.txt、防自污染。
- `scripts/os-agent/gates/define-gates.sh`：**四模式 gate 定义**——implement/review/review-fix/audit 的 check 组合与 dispatch 逻辑。
- `scripts/os-agent/checks/run-checks.sh`：**check 函数库**——git-diff-check、shell/python 语法、cmake-configure、compile-commands、clang-tidy、incremental-build、parser-test、skill-validation、memory-check。
- `scripts/os-agent/tests/test-harness.sh`：**15 E2E 自测**——合成 fixture repos、四模式 dispatch、target mapping、marker 唯一性、artifact 防自污染。

## Integration Points

- 模型加载阶段根据环境和模型形状选择 Flex 或 MoE-Buffer。
- CPU graph compute 安装 weight-stream callback；CLG node callback 只提供预取提示。
- KV graph input 在 `llama-graph.cpp` 调用 `build_input_paged_row_idx()` / `set_input_paged_row_idx()`，attention 的 `get_k()`/`get_v()` 接收 row index；**dummy redirect 在 row-index fill 中发生**。
- 上层通过 `prefetch_seq()` / `prefetch_seq_step()` 等 memory hook 表达 resume 预取。
- **Release 在 idle/resume boundary 由 example driver 调用**；core 不自动触发 release。
- **Pressure sampler → server telemetry → dry-run 集成点**（Stage 3A-2B）：
  - Server init: `init_kv_pressure_sampler()` 中解析 dry-run config 并 `dry_run_enable()`。
  - Server scheduler: `maybe_sample_kv_pressure(all_idle)` 的 Phase B（telemetry 采样之后）执行 `should_evaluate` 门控 + `bounded_release_dry_run()`。
  - Server sleep: `dry_run_disable()` 清除 cooldown state。
  - Server resume: `init_kv_pressure_sampler()` 中重新解析并 `dry_run_enable()`。
  - 调用侧为 `#if defined(__linux__)` 条件编译；非 Linux 平台无压力采样或 dry-run 代码路径。
  - **Dry-run 与 destructive release 不共享调用路径**：dry-run 通过 `bounded_release_dry_run()` virtual 接口 → `paged_release_blocks_bounded_dry_run() const`；destructive 通过 `paged_release_blocks_bounded()` → `paged_madvise_block()`。

## Invariants and Error Propagation

- 默认关闭：未显式设置相关环境变量时，不启用 Flex/MoE/CLG/paged swap/madvise/release/prefetch/pressure sampler/**dry-run**。
- Pressure sampler 配置、cgroup 路径或 selected source 无效时 transition fail-closed；telemetry 可保持 enabled，但 source 为 NONE，状态不得据此驱动 reclaim 或 dry-run。
- **Dry-run 配置 fail-closed**：env 解析错误 → `dry_run_config.enabled=false`，零 marker、零 scanner 调用。target_bytes=0 即使 master switch=1 也自动禁用。
- **Dry-run 不执行 destructive reclaim**：`paged_release_blocks_bounded_dry_run()` 为 `const` 方法，编译器强制零 mutation。Stage 3A-2B VALID artifact 的 strace 确认零 MADV_DONTNEED、zero destructive release marker。
- **Pressure state 与 KV block state 的解耦已延伸至 dry-run**：dry-run scanner 读取 pressure state 作为触发 gate，但 scanner 本身不推进任何 block state。PRESSURE/CRITICAL 状态下 dry-run 仅输出 would-release 预测，不改变 KV 状态。
- **Server owner 为单线程**：`update_slots()` 由 server queue 的单线程 scheduler 调用；`kv_pressure_sampler_owner`、`kv_pressure_runtime`（含 dry-run state）仅在此上下文中被访问，无 mutex 或 background thread。
- **Sleep/resume 隔离**：sleep 时 sampler 销毁、runtime 清零、dry-run state 清除；resume 时全新初始化，不继承旧 pressure state、transition counters、sampling deadlines、stale 状态、cooldown state 或 state-entry tracking。
- **Idle 不绕过限频**：`maybe_sample_kv_pressure()` 在 idle 检查前调用，但 `sample_due()` 与 `dry_run_due()` 的时间门控与 idle 标记独立；idle 期间不持续读取 procfs/cgroup，不持续评估 dry-run。
- backing store 容量固定为物理 KV cell 数乘每 cell 全层 K/V stride。
- paged block 写入整块成功后才发布 offsets 和 `SWAPPED`；失败时清除待发布 metadata，block 保持可重试状态。
- swap-in 先读入 staging，再提交 tensor 与 `RESIDENT` 状态；active 必需的恢复失败记录 context-local error，并在 graph compute 前失败返回。
- `paged_release_blocks()` 每次调用前重算 ownership；invalid mapping 导致 ABORT 且 `destructive_release_skipped=1`。
- PENDING_WRITE 事务：commit 后 block 为 RESIDENT；rollback 后 block 为 RELEASED 并回收至 free list。不在中间态遗留。
- Release 过程中不跳过任何 RESIDENT block 的 madvise；只有 owned、RELEASED 和 SWAPPED block 被跳过。
- **Dry-run 复用相同 ownership + state gate 但不 madvise**：与 destructive 路径的 owned/PENDING_WRITE/SWAPPED/RELEASED skip 逻辑一致；差异仅在于 RESIDENT/UNUSED candidate 的计算方式（page-aligned byte count vs madvise + state mutation）。
- R0–R5/N0–N2 artifact 必须与 `RUN_PLAN=(R0 R1 R2 R3 R4 R5 N0 N1 N2)` 精确一致；parser 对 contract marker、安全字段、机制触发和最终 `PASS` 状态 fail-closed。
- Stage 3A-1C validation protocol parser 对 manifest identity drift、case 缺失/额外、SSE 格式错误、OFF variant 出现 telemetry marker、structured action marker、timeout、residual process、case 顺序/端口重复和 strace 归因均 fail-closed。
- **Stage 3A-2B dry-run parser** 对 dry-run marker field schema、OFF/ON isolation、response divergence、destructive release contamination、MADV_DONTNEED leakage、config isolation 均 fail-closed。
- Identity fast-path eligibility is immutable for a constructed context；graph reuse 必须 preserve topology。
- **Release 是 correctness 机制，不是压力调度策略**。当前证据验证的是选择性（不误伤）、事务原子性和互斥门禁。Dry-run 扩展验证的是控制链路、只读性和协议——不代表真实阈值下的回收效果或性能收益。
- **Harness v1 不变量**：
  - 默认不运行：只有显式 `gate-runner <mode>` 才执行，不 hook 到 git/build/editor。
  - Fail-closed marker：`OS_AGENT_GATE_RESULT` 全局唯一输出点——无重复、无歧义、无隐式 fallback。
  - Artifact 防自污染：`/tmp/os-agent-gate/` 在 repo 外，diff-analyzer 排除此前缀。
  - Synthetic fixture 隔离：harness self-test 在 `/tmp/os-agent-gate-test-*/` 创建临时目录，exit 时 trap cleanup。
  - 模式不扩大范围：audit 不执行 build/review/clang-tidy；review-fix 不执行 clang-tidy。
  - Mapping 失败不阻塞：target mapping UNRESOLVED 返回 code 2（不是 FAIL/1）——标记需要人工确认但不阻止 audit/review 完成。

## Modification Boundaries

- core 提供 block state、I/O、madvise、restore、prefetch、release（destructive + dry-run）与诊断机制；session lifecycle、release timing、pressure policy 保持在 server/application/example 层。
- `kv_pressure_sampler` 只提供输入采样、状态判定和 telemetry；不得在该组件内直接调用 destructive release 或 dry-run scanner。
- server runtime (`server-kv-pressure.*`) 管理 sampling cadence、event 发布、marker 输出、dry-run cooldown/backoff 状态机和 dry-run marker 格式化；同样不得调用 destructive reclaim。
- bounded reclaim 的预算、候选集、频率和生命周期 owner 属于 server policy；core 提供 `paged_release_blocks_bounded()`（destructive）和 `paged_release_blocks_bounded_dry_run()`（只读）双原语供 policy 层选择。
- `llama_kv_bounded_release_result` 已提升至 `llama-kv-cache-release.h` 作为全局 struct——修改其字段布局会影响 core（paged_release_blocks_bounded）、dry-run scanner、server marker 格式化和 parser。
- `llama_kv_release_status` 枚举的新增 variant 会影响 server policy 的 skip reason 分支——必须同步更新 `paged_release_status()` 实现和 server `should_evaluate` 门控。
- 修改 `paged_block_state` 枚举或状态转换时必须同步核对：ownership collection、dummy redirect、事务提交/回滚、swap gate、release gate **和 dry-run scanner 的 state gate**。
- 任何可能改变 post-construction mapping/residency 的新机制必须与 release 的 ownership gate 互斥或提供经证明的失效契约。
- 不应把 release correctness 协议结果或 dry-run would-release 计数描述为压力调度性能结论。
- 修改 `scripts/os-agent/` 文件时必须同步更新 `test-harness.sh` 中的对应 E2E 测试、确保 15/15 继续 PASS；`target-mapper.sh` 的 UMBRELLA_MAP 必须在 CMake target 名称变化时同步更新。
- `DIFF_CLASSES`、`HAS_*` flags 或 check dispatch 逻辑的修改会影响所有四种 gate 模式的行为——必须通过 audit gate 回归后再推进其他模式。**修改 `gate_emit_summary()` 的 marker 格式时，必须同步更新所有 parser 和 test-harness.sh 的 marker 断言。**

## Known Limitations

- 尚未验证 GPU/device-memory KV reclaim；host `madvise`/`mincore` 语义不能外推到 GPU。
- KV paged 主路径当前只接受兼容 layout（`n_stream == 1 && !v_trans`）；部分路径还限制 F32 K/V。
- KV prefetch 由上层分步触发，不是独立异步恢复线程；真实 server queue/continuous batching 尚未接入。
- 权重侧 Flex 与 MoE-Buffer 是分离路径，不是统一 shared weight/KV I/O budget scheduler。
- Release 当前只在 example driver 的 idle/resume boundary 调用；**dry-run scanner 已接入 server scheduler，但 destructive bounded release 尚未接入**。
- **Stage 3A-2B dry-run VALID artifact 仅覆盖单请求、单模型（Llama-3-8B Q4_K_M）、forced threshold（1/2/3 KiB RSS）、ctx 1024、32 token 输出**。不覆盖并发请求、长上下文、真实内存压力、不同模型/quantization。
- Stage 3A-1C 性能结论为 EXPLORATORY_ONLY（n=3）。Stage 3A-2B dry-run 未做性能 profiling——其 overhead 来自 ownership collection + page-aligned byte counting（与 destructive madvise 路径对等但不执行 syscall），尚未有 TTFT/TPOT 测量。
- **Dry-run 的 zero-side-effect 属性当前仅由 `const` 方法签名和源码审计保证**；尚无 strace/mincore/ASan 在并发或长上下文下的独立验证。
- **Dry-run cooldown/backoff 参数（default cooldown=2s, backoff=10s, min=500ms）为初始值**，未经过真实 workload 调优；过短可能导致不必要的 scanner CPU 开销，过长可能导致 pressure response 滞后。
- Stage 3A-1C validation protocol 的 10-case 矩阵使用 1/2/3 KiB RSS 阈值，仅为 state lifecycle forced validation，不代表部署推荐值或经验性压力限制。Stage 3A-2B 共享相同 forced thresholds。
- 尚未验证真实模型/server 请求下的四态转换、source 动态切换与 stale 恢复；当前状态机证据来自 synthetic/fixture 单测和 forced-threshold server artifact。
- **Bounded release 原语已实现但 destructive 变体未接入任何调度层**：`paged_release_blocks_bounded()` 无 server scheduler 调用点。并发 request 下的 ownership 不变性、RSS 降幅、TTFT/TPOT/吞吐影响均未验证。
- **内部对象布局变化**：`llama_kv_cache` 新增三个 `mutable` test-only seam 字段和两个 test-only accessor 方法，以及四个 global counter accessor。默认值保证生产路径零行为差异，但对象 size 和 cache line 布局改变是静默的——当前没有 CI 门禁或 sizeof assertion 检测未来字段增删对布局的累积影响。
- **原 unbounded `paged_release_blocks()` 缺少 PENDING_WRITE gate**：只检查 RELEASED 和 SWAPPED，不检查 PENDING_WRITE。PENDING_WRITE block 当前可被 madvise 误伤。该缺陷被 bounded release（destructive + dry-run 双变体）的显式 PENDING_WRITE skip 间接记录为已知差距，但原路径尚未修复。
- **`test-kv-paged-release-bounded` 的 `decode_prompt()` 硬编码 BOS token 128000**（Llama-3 tokenizer）。非 Llama-3 模型（如 Qwen、DeepSeek）的 BOS 不同，测试将因 token 无效而失败。
- **`llama_kv_bounded_release_result` 提升至 `llama-kv-cache-release.h`**：将该 struct 从 `llama_kv_cache` 内部类型提升为全局类型，使得 `llama-memory.h` 可引用而无需 include `llama-kv-cache.h`。未来若 `llama_kv_bounded_release_result` 字段变化，server marker 格式化和 parser 均需同步更新。
- 当前 release 证据覆盖单机 CPU、Llama-3-8B Q4_K_M、ctx 1024、parallel 4、固定 idle/resume workload。不覆盖多模型、长上下文、server/continuous batching 或不同 backend/layout。
- 历史 README 中的性能数字缺少仓库内原始 artifacts 与 commit/worktree 绑定，目前无法确认其对当前 HEAD 的适用性。
- **Harness v1 限制**：
  - Harness 仅验证 diff 范围内的文件与合成 fixture——不执行完整 CTest suite 或长时间稳定性测试。`incremental-build` 只编译受影响 targets，不等同于全量构建。
  - `clang-tidy` 在缺少 clang-tidy 二进制或 `CMAKE_EXPORT_COMPILE_COMMANDS` 时自动 SKIP（非 FAIL），当前环境无 clang-tidy。
  - `target-mapper` 依赖 `compile_commands.json`；该文件由 CMake `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 生成，非项目强制依赖。缺失时 target resolution → UNRESOLVED。
  - Harness self-test 的 E2E-10（incremental build）使用 synthetic fixture repo（最小 CMakeLists.txt），非对真实 `llama.cpp` build 的验证。
  - `check_memory_check()` 验证工程账本文件存在但不验证内容正确性——内容一致性由人工 `memory update` 和 `os-agent-task` 流程保证。
  - Gate 仅覆盖四种显式 mode 调用；不会自动 hook 到 git commit、push 或 PR workflow。
