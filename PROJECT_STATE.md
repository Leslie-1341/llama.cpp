# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-23
- Evidence commit: `a532c53ad3e21c42d032f097b97fb96da48d9df0`（三份 KV 目标契约冻结）
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: **dirty**（25 tracked modified + 4 untracked；其中包含 Stage 3A-2C runtime/协议改动，尚未构建或验证）
- Upstream relation: 目前无法确认；本轮未查询远端引用

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

**KV 生命周期、pressure scheduler 与统一证据的目标契约已冻结；对应 runtime 尚未实现。** HEAD `a532c53ad` 新增三份仅定义目标 authority 与完成判定的文档：`docs/kv_block_lifecycle_contract.md`、`docs/kv_pressure_scheduler_contract.md`、`docs/kv_lifecycle_evidence_protocol.md`。三份文档均明确：目标语义、统一 action/事件与证据闭包不等于当前源码已满足；现有 runtime 只具有部分基础，仍存在 P0/P1 语义和证据差距。

**Stage 3A-2C runtime/协议工作树仍未验证。** 当前有 25 tracked modified + 4 untracked；其中包括 bounded destructive release 的 server 接入、生命周期相关 core 改动及 runner/parser/test。该工作树尚未构建、测试、review 或通过 implement gate；不得将其或三份冻结契约描述为 runtime 已实现。Harness v1 的既有提交证据仍见下文，但不替代本轮 runtime 验证。

## Implemented and Verified

- **Harness v1 — diff-aware validation harness**（`fd51455b7`）：
  - `scripts/os-agent/gate-runner`：单入口，支持 `implement` | `review` | `review-fix` | `audit` 四种模式。自动 diff 分析（`diff-analyzer.sh`）→ target mapping（`target-mapper.sh`，compile_commands.json -o flag 提取 + cmake --target help 验证）→ 模式特定检查（git-diff-check、shell/python 语法、cmake-configure、clang-tidy、incremental-build、parser-test、skill-validation、memory-check）→ 唯一结构化 marker `OS_AGENT_GATE_RESULT mode=<mode> verdict=<v> code=<c> checks=<n> pass=<p> fail=<f> skip=<s> unresolved=<u> artifact=<path>`。Exit codes: 0=PASS 1=FAIL 2=UNRESOLVED 3=INCOMPLETE 4=NO_CHANGES。Artifact 写入 `/tmp/os-agent-gate/`（仓库外，防自污染）。
  - `scripts/os-agent/lib/target-mapper.sh`：从 compile_commands.json 的 `-o CMakeFiles/<target>.dir/` 提取源文件→target；header 通过 UMBRELLA_MAP（`src→llama`、`tools/server→llama-server` 等）解析；所有 target 经 `cmake --build <dir> --target help` 验证。
  - `scripts/os-agent/lib/diff-analyzer.sh`：分类 changed/deleted/untracked 文件到 ext-based 类别（cpp/c/h/py/sh/cmake/skill/ledger），设置 HAS_* 标志驱动后续 check dispatch。
  - `scripts/os-agent/tests/test-harness.sh`：**15/15 E2E PASS**——覆盖 NO_CHANGES、audit readonly、Python/shell 语法 FAIL 检测、untracked 检测、deleted 追踪、C++ target mapping、multi-target mapping、UNRESOLVED cpp、incremental build PASS、artifact 防自污染、单一 summary marker、parser UNRESOLVED、implement/review/review-fix/audit 四模式 dispatch、--build-dir flag。
  - Audit gate 在 repo 根实际运行：verdict=PASS，checks=8，pass=5，skip=3，unresolved=0。
  - 已知限制：clang-tidy 在无 clang-tidy 环境会 SKIP（非 FAIL）；E2E-10 incremental build 使用 fixture repo（非真实编译）；target mapping 依赖 compile_commands.json 存在；gate 仅验证 diff 范围内的文件，不执行完整 CTest suite。

- **Stage 3A-2B pressure dry-run 控制链路**（`02f8cd5ed`）：
  - `paged_release_blocks_bounded_dry_run(target_bytes, max_scan_blocks) const`：只读扫描——与 destructive `paged_release_blocks_bounded()` 共享 ownership collection + state gate 逻辑，但不调用 `paged_madvise_block()`、不修改 `paged_block_states`、不操作 `paged_free_list`、不清除 backing metadata、不递增 global release counters。结果通过 `llama_kv_bounded_release_result` 返回 would-release 语义。
  - `bounded_release_dry_run()`：`llama_memory_i` virtual 接口，默认 no-op；paged KV 实现委托 `paged_release_blocks_bounded_dry_run()`。
  - `llama_kv_release_status` 枚举 + `paged_release_status()`：精确区分 release 可用/禁用原因（`available` / `not_paged` / `layout_unsupported` / `swap_enabled` / `disabled`），替代单一 boolean 过载。server policy 据此区分硬性 skip reason（swap/layout/not_paged）与观测性 disabled。
  - `server_kv_pressure_dry_run_config`：env 解析 `LLAMA_KV_PRESSURE_DRY_RUN`（master switch）、`LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES`（零值自动禁用）、`LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS`（≥1）、`LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS`/`BACKOFF_MS`（≥500ms），fail-closed 解析。
  - `server_kv_pressure_runtime` 扩展：`dry_run_enable()` / `dry_run_disable()` / `dry_run_due()` / `dry_run_record()` — cooldown/backoff 策略，state-entry 语义（进入 PRESSURE/CRITICAL 时立即评估一次），CRITICAL entry 绕过 cooldown，shortfall 时延长 backoff。
  - **`should_evaluate` 显式门控**（server-context.cpp `maybe_sample_kv_pressure()` 内）：
    1. master switch enabled + target_bytes > 0
    2. `llama_memory_i` 存在 → `paged_release_status()` 检查（hard skip reasons: swap_enabled/not_paged/layout_unsupported）
    3. `telemetry.stale` → skip
    4. pressure state 必须为 PRESSURE 或 CRITICAL（NORMAL/RECOVERY 不触发评估，无 marker）
    5. `dry_run_due()` cooldown/backoff 时间门控
    ——任意条件不满足即不调用 scanner、不输出 marker。
  - **dry-run marker**：`kv_pressure_dry_run state=... source=... stale=... release_enabled=... would_release_bytes=... would_release_blocks=... blocks_scanned=... blocks_skipped_owned=... blocks_skipped_state=... shortfall_bytes=... overshoot_bytes=... block_scan_exhausted=... ownership_aborted=... target_bytes=... max_scan_blocks=... skipped_reason=... cooldown_ms=... sample_count=... idle=...`，字段固定可解析。ownership ABORT 时升级为 SRV_WRN。
  - **dry-run 与 destructive release 解耦**：
    - dry-run 不检查 `LLAMA_KV_PAGED_RELEASE`；仅需 paged KV + valid layout + !swap。
    - dry-run 不接入 test-only seam（`force_ownership_abort`、`madvise_fail_block`、`block_state_override` 仅作用于 destructive `paged_release_blocks_bounded()`）。
    - dry-run 为 `const` 方法，编译器强制零 mutation。
  - Sleep/resume 生命周期：`dry_run_disable()` 清除 config 和 cooldown state；resume 后独立重新解析 env。
  - **Stage 3A-2B VALID artifact**（`/root/oscomp/kv_logs/kv_dry_run_stage3a_2b_20260720T161209Z_02f8cd5ed3`）：parser exit 0、verdict PASS、OFF=0 dry_run markers、ON=7 dry_run markers、response byte-identical、zero destructive release、zero MADV_DONTNEED (strace)、zero residual processes。Forced threshold 1/2/3 KiB RSS 触发 CRITICAL → dry-run scanner 产生 would-release 候选。
  - Static check 扩展（`test-server-kv-pressure-static.py`）：新增 dry-run 相关检查（dry-run decoupling from destructive release、marker field schema、config isolation）。
  - Dry-run parser synthetic negative 回归（`test-kv-dry-run-stage3a-2b-parser.py`）：fail-closed 验证 marker field schema、OFF/ON isolation、response divergence、destructive release contamination、MADV_DONTNEED leakage、config isolation。
  - C++ 集成测试扩展（`test-server-kv-pressure.cpp`）：dry-run config 解析、cooldown/backoff、state-entry、skip-reason 路径。
- **Stage 3A-2A core bounded KV release 原语**（`949fbd0c8`）：
  - `paged_release_blocks_bounded(target_bytes, max_scan_blocks)`：per-call budget 控制，target_bytes=0 立即返回（无副作用），max_scan_blocks=0 返回 exhausted+full shortfall。
  - 结果语义：`llama_kv_bounded_release_result`（已提升至 `llama-kv-cache-release.h` 作为全局 struct，供 core 与 server 共享）。
  - 状态门禁（按优先级）：ownership ABORT（invalid mapping → 立即返回）→ owned[block] skip → PENDING_WRITE/SWAPPED/RELEASED state skip → madvise → RELEASED + free list。**PENDING_WRITE gate 是相对于原 unbounded release 的防御增强**。
  - Test-only seam（全部 single-shot 自复位，生产路径零影响）。
  - 构建、CTest 注册和 `git diff --check` 通过。
- **Stage 3A-1C server-only telemetry 集成**（`6f59f66b6`、`726d977b2`）：10/10 cases PASS、strace 归因、lifecycle 隔离、idle 限频。
- **Stage 3A-1B pressure sampler-only**（`befd7a894`）：864/864 assertions、ASan/UBSan 通过。
- **Stage 3A-0 RELEASED 生命周期**（`73c2aa9ef`、`708626cd0`、`adfe67136`）：R0–R5/N0–N2 VALID。
- Stage 2 static paged identity fast path（`a744830e9`）已保留并通过 E2I 功能门禁与三轮 controlled A/B。
- 以下 Stage 2 事实仍然成立：Backing store 固定槽位、block I/O 原子发布、active 恢复错误传播、P0 build/backing-store unit、E0-E5 parser fail-closed 门禁、opt-in 环境变量。

## Implemented but Evidence Insufficient

- dry-run **已接入 server scheduler 并在 forced-pressure 下产出 VALID artifact**，但仅覆盖单请求、单模型（Llama-3-8B Q4_K_M）、forced threshold（1/2/3 KiB RSS）、ctx 1024、32 token 输出。不代表真实内存压力下的 dry-run 行为、候选 block 分布、cooldown/backoff 调优效果或多请求并发下的 ownership 安全性。
- `paged_release_blocks_bounded_dry_run()` 的 would-release 字节计算与 destructive 路径的 page-alignment 逻辑一致，但 dry-run 的零 side-effect 属性仅在代码级（`const` 方法签名 + 源码审计）保证，尚无 strace/mincore/ASan 在并发或长上下文下的独立验证。
- bounded release 测试（`test-kv-paged-release-bounded`）Part B（B1–B11，real-model correctness）因当前环境未提供 `LLAMACPP_TEST_MODELFILE` 而 CTest SKIP。
- 原 `paged_release_blocks()`（unbounded）缺少 PENDING_WRITE gate——该缺陷未修复且未触发已知 failure。
- Stage 3A-1C 性能结论仅为 EXPLORATORY_ONLY（n=3）。
- Stage 3A-1B 尚未在真实模型/server workload 中观察 NORMAL/PRESSURE/CRITICAL/RECOVERY 状态转换（状态转换由 forced threshold 触发，非真实压力）。
- 当前 HEAD 的 E0-E5 正式模型矩阵（RUNS=3）仍未运行。
- README 与历史文档的性能数字缺少仓库内原始 artifact、hash 和 commit/worktree 绑定。

## In Progress

- **Stage 3A-2C 及生命周期 runtime/协议改动**：源码、测试、runner/parser 与 Harness 改动共 25 个 tracked 文件和 4 个 untracked 文件；当前工作树状态是**尚未提交、构建、测试、review 或 gate 验证**。三份冻结契约已将其目标语义与 P0/P1 缺口列明，但不构成对该 runtime 的实现或验证声明。

## Blocked

- 目标契约的 runtime 实现存在 P0 语义与证据缺口：三轴状态/overlay、prepare 零 committed mutation、唯一 visibility commit、quarantine 闭合、统一 scheduler action/result/event 与完整 lifecycle parser 尚未实现。当前 Stage 3A-2C 工作树亦未完成构建和验证。

## Next Gate

**唯一下一门禁：先将当前 runtime 改动与冻结目标契约逐项对照，关闭 P0 生命周期/调度/证据缺口并完成定向构建、测试、review 和 implement gate；随后才能进行 Stage 3A-2C 固定 target 的真实 bounded destructive reclaim 受控验证。** 该受控验证仍需将 `paged_release_blocks_bounded()` 接入 PRESSURE/CRITICAL 分支（替换 dry-run scanner），执行真实 `MADV_DONTNEED`。既有验收要求：
1. OFF/ON controlled A/B：OFF 零 release marker + 零 MADV_DONTNEED；ON 产生真实 release marker（`paged_block_release_bytes > 0`、`paged_blocks_released > 0`）。
2. 响应 byte-identical（release 不误伤 active-owned block）。
3. strace 确认 MADV_DONTNEED 仅发生于 ON variant 且数量与 release marker 一致。
4. ownership ABORT 零发生（排除 invalid mapping）。
5. 不引入 swap、prefetch、dry-run 或 backing store 变更。
6. 使用固定 target_bytes 和 max_scan_blocks（非 forced-pressure 动态阈值），以便与 dry-run 的 would-release 预测对比验证。
7. 实现完成后须通过 `gate-runner implement` 门禁再声明完成。
8. 通过后进入 Stage 3B：压力阈值调优、动态 target 策略、与 swap/prefetch 的互斥调度融合。
