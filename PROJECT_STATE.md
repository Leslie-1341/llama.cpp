# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-20
- Evidence commit: `949fbd0c850b7413302907df20cc4022f623d8db`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean（untracked `Testing/` 不与源码重叠）
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 7

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

**Stage 3A-2A core bounded KV release 稳定节点已提交。** HEAD `949fbd0c8` 在 Stage 3A-1C VALID server pressure telemetry（`726d977b2`）基础上新增 `paged_release_blocks_bounded()` 核心原语及 533 行 correctness 测试。bounded release 提供 per-call target_bytes 预算和 max_scan_blocks 扫描上限，复用 `llama_kv_release_collect_ownership` 门禁，并在状态检查中显式保护 PENDING_WRITE block（原 unbounded `paged_release_blocks()` 缺少此 gate）。test-only seam（ownership ABORT、madvise failure、block state override）均为 single-shot 自复位，零生产行为影响。CTest 注册（#29 `test-kv-paged-release-bounded`）、build 通过、`git diff --check` 通过。bounded release 未接入 server、pressure sampler、swap 或 prefetch；未验证并发安全、RSS 降幅、时延收益。

## Implemented and Verified

- **Stage 3A-2A core bounded KV release 原语**（`949fbd0c8`）：
  - `paged_release_blocks_bounded(target_bytes, max_scan_blocks)`：per-call budget 控制，target_bytes=0 立即返回（无副作用），max_scan_blocks=0 返回 exhausted+full shortfall。
  - 结果语义：`llama_kv_bounded_release_result` 含 released_bytes/shortfall_bytes/overshoot_bytes/released_blocks/blocks_scanned/blocks_skipped_owned/blocks_skipped_state/madvise_failures/block_scan_exhausted/ownership_aborted。
  - 状态门禁（按优先级）：ownership ABORT（invalid mapping → 立即返回）→ owned[block] skip → PENDING_WRITE/SWAPPED/RELEASED state skip → madvise → RELEASED + free list。**PENDING_WRITE gate 是相对于原 unbounded release 的防御增强**（原版只检查 RELEASED 和 SWAPPED，PENDING_WRITE block 可能被误伤）。
  - Budget 检查：单 block overshoot 允许（`released_bytes >= target_bytes` 时停止扫描）；shortfall/overshoot 由结果字段精确计算。
  - Test-only seam（全部 single-shot 自复位，生产路径零影响）：
    - `paged_release_bounded_test_force_ownership_abort`：强制 ownership ABORT 路径，验证零 state change。
    - `paged_release_bounded_test_madvise_fail_block`：按物理 block index 注入 madvise 失败，跳过实际 `MADV_DONTNEED`，验证 block state 不变且扫描继续。
    - `paged_release_bounded_test_block_state_override`：按单 block 覆盖观测 state（含 PENDING_WRITE=4），用于验证 state gate 在 ownership gate 之后独立触发。
    - `paged_release_bounded_test_read_block_state()` / `paged_release_bounded_test_block_in_free_list()`：post-ABORT state invariance 验证。
  - 所有 test seam 在以下退出路径自动复位：target=0、max_scan=0、ownership ABORT、正常扫描结束。严格 single-shot。
  - 构建、CTest 注册和 `git diff --check` 通过。
- **Stage 3A-1C server-only telemetry 集成**（`6f59f66b6`）：
  - 单 owner、限频采样、默认关闭、sleep/resume 重建、idle 路径、与 reclaim 完全解耦、配置 fail-closed。
  - Go 验证：`test-server-kv-pressure.cpp`（9 C++ 集成回归）、`test-server-kv-pressure-static.py`（6 静态集成检查）已通过并注册于 CMake。
- **Stage 3A-1C 验证协议**（`297eed939`）：
  - Runner、parser、9 parser 合成负例、py_compile 全部通过。
- **Stage 3A-1B pressure sampler-only**（`befd7a894`）：
  - 864/864 assertions、ASan/UBSan、sampler-only strict warning build 全部通过。
  - 四态状态机、source 选择、stale、滞回/cooldown、PSI upgrade 隔离已实现并通过 fixture/synthetic 验证。
- **Stage 3A-0 RELEASED 生命周期**（`73c2aa9ef`、`708626cd0`、`adfe67136`）：
  - R0–R5/N0–N2 长门禁 artifact：parser exit 0、overall result PASS、全部 passing case 输出 byte-exact 一致。
- **Stage 3A-1C 真实 server pressure telemetry VALID**（`726d977b2`）：
  - Artifact: `/root/oscomp/kv_logs/server_kv_pressure_stage3a_1c_20260719T134156Z_726d977b26bb`，parser exit 0、status VALID、10/10 cases PASS。
  - Strace 归因、lifecycle 隔离、idle 限频、OFF 零 marker 全部验证通过。
  - Performance: EXPLORATORY_ONLY（n=3，不声明正式收益）。
- Stage 2 static paged identity fast path（`a744830e9`）已保留并通过 E2I 功能门禁与三轮 E2G/E2I controlled A/B。
- 以下 Stage 2 事实仍然成立（本轮未修改）：
  - Backing store 固定槽位、block I/O 原子发布、active 恢复错误传播至 graph 前失败。
  - P0 build/backing-store unit、B2B、I/O fault 与 duration stability 回归已提交。
  - E0-E5 parser fail-closed 门禁已就绪（`d9d2e3b80`）。
  - Opt-in Dense Flex、MoE-Buffer/CLG、paged KV 路径均通过环境变量显式启用。

## Implemented but Evidence Insufficient

- bounded release **未接入 server、pressure sampler、swap 或 prefetch**；`paged_release_blocks_bounded()` 仅在 example driver 层可调用，server scheduler 无调用点。无并发 request 下的 ownership 不变性、RSS 降幅、TTFT/TPOT/吞吐影响证据。
- bounded release 测试（`test-kv-paged-release-bounded`）Part A（ownership fault fixture，4 组）无需模型即可运行，Part B（B1–B11，real-model correctness）因当前环境未提供 `LLAMACPP_TEST_MODELFILE` 而 CTest SKIP；测试逻辑已通过 build+CTest 注册验证，但正式 run 日志和 assertion 计数未产生。
- 原 `paged_release_blocks()`（unbounded）缺少 PENDING_WRITE gate——PENDING_WRITE block 在当前代码中可能被 madvise 误伤，该缺陷未修复且未触发已知 failure。
- Stage 3A-1C 性能结论仅为 EXPLORATORY_ONLY（n=3，单模型、单提示、无并发请求、无长上下文）。
- Stage 3A-1B 尚未在真实模型/server workload 中观察 NORMAL/PRESSURE/CRITICAL/RECOVERY 状态转换。
- 当前 HEAD 的 E0-E5 正式模型矩阵（RUNS=3）仍未运行。
- README 与历史文档的性能数字缺少仓库内原始 artifact、hash 和 commit/worktree 绑定。

## In Progress

- 无源码、脚本或测试改动正在进行；本轮仅同步四个工程账本。

## Blocked

- 无已知功能门禁阻塞。bounded release 原语已就绪，进入 server pressure policy dry-run 接入前需先完成 policy 设计（调度 owner、dry-run 模式、日志 vs 真实 reclaim 的切换契约、与 swap/prefetch 互斥保持不变）。

## Next Gate

**唯一下一门禁：server pressure policy 的 dry-run 接入。** 将 Stage 3A-1C 已验证的 pressure state 与 Stage 3A-2A 的 bounded release 原语通过 server scheduler 连接：在 `maybe_sample_kv_pressure()` 的 PRESSURE/CRITICAL 分支中调用 `paged_release_blocks_bounded()` 的 dry-run 模式（实际不执行 `MADV_DONTNEED`，仅记录 would-release 决策、候选 block 和预算消耗），验证调度时序、ownership 安全性、压力状态与回收决策的一致性。dry-run 期间不执行 destructive reclaim；不得引入 swap、prefetch 或 backing store 变更。验收要求：dry-run marker 字段固定可解析、OFF/ON/PRESSURE/CRITICAL 各状态下的 would-release 决策可审计、ownership ABORT 正确传播、与 swap 互斥保持、strace 确认真实 reclaim 零发生。通过后再激活真实 madvise 的 destructive reclaim。
