# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-20
- Evidence commit: `02f8cd5ed33a13ffac3cce444d6d3ea7eb987650`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean（untracked `Testing/` 不与源码重叠）
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 10

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

**Stage 3A-2B pressure-driven KV reclaim dry-run 稳定节点已提交并 VALID。** HEAD `02f8cd5ed` 将 Stage 3A-1C 已验证的 server pressure telemetry 与 Stage 3A-2A 的 bounded release 原语通过 `paged_release_blocks_bounded_dry_run()` 只读扫描连接：在 PRESSURE/CRITICAL 状态下对符合 budget 约束的候选 block 计算 would-release 决策，但**零 KV state mutation、零 MADV_DONTNEED、零 backing metadata 清除**。dry-run 与 destructive release 完全解耦：dry-run 不接入 test-only seam、不检查 `LLAMA_KV_PAGED_RELEASE`（仅需 paged KV + valid layout + !swap）、通过 `should_evaluate` 显式门控（master switch、release_status gate、stale 拒绝、状态触发 gate、cooldown/backoff）控制评估频率。OFF/ON controlled A/B artifact parser exit 0、verdict PASS、response byte-identical、strace 确认零 MADV_DONTNEED、零 destructive release。**该证据仅验证 forced-pressure dry-run 控制链路、只读性和协议，不代表真实阈值或性能收益。**

## Implemented and Verified

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

- 无源码、脚本或测试改动正在进行；本轮仅同步四个工程账本。

## Blocked

- 无已知功能门禁阻塞。dry-run 控制链路已验证通过；下一门禁为 destructive reclaim 受控验证（Stage 3A-2C）。

## Next Gate

**唯一下一门禁：Stage 3A-2C — 固定 target 的真实 bounded destructive reclaim 受控验证。** 在 dry-run 已验证的控制链路基础上，将 `paged_release_blocks_bounded()` 接入 PRESSURE/CRITICAL 分支（替换 dry-run scanner），执行真实 `MADV_DONTNEED`。验收要求：
1. OFF/ON controlled A/B：OFF 零 release marker + 零 MADV_DONTNEED；ON 产生真实 release marker（`paged_block_release_bytes > 0`、`paged_blocks_released > 0`）。
2. 响应 byte-identical（release 不误伤 active-owned block）。
3. strace 确认 MADV_DONTNEED 仅发生于 ON variant 且数量与 release marker 一致。
4. ownership ABORT 零发生（排除 invalid mapping）。
5. 不引入 swap、prefetch、dry-run 或 backing store 变更。
6. 使用固定 target_bytes 和 max_scan_blocks（非 forced-pressure 动态阈值），以便与 dry-run 的 would-release 预测对比验证。
7. 通过后进入 Stage 3B：压力阈值调优、动态 target 策略、与 swap/prefetch 的互斥调度融合。
