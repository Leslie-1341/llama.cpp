# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-17
- Evidence commit: `adfe671367f0cdc17327786c2b5c6182939cbf09`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 3（`73c2aa9ef`、`708626cd0`、`adfe67136` 尚未推送）

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

**Stage 3A-0 RELEASED 生命周期正确性基线已通过。** clean-HEAD `adfe67136` 的 R0–R5/N0–N2 长门禁全部以预期结果通过（R0–R4 PASS、R5 EXPECTED_FAILURE、N0–N2 PASS），parser exit 0。确认 destructive release 仅允许 no-backing dead/unused block、可恢复历史必须经 SWAPPED、ownership 与 transactional commit/rollback 语义正确、dummy redirect 不污染 active visible 行，以及 release 与 swap、non-paged、non-ingraph、non-F32 K/V 的互斥门禁生效。

## Implemented and Verified

- Stage 2 static paged identity fast path（`a744830e9`）已保留并通过 E2I 功能门禁与三轮 E2G/E2I controlled A/B。
- **Stage 3A-0 RELEASED 生命周期**（`73c2aa9ef`、`708626cd0`、`adfe67136`）：
  - 四态 + PENDING_WRITE：UNUSED / RESIDENT / RELEASED / SWAPPED / PENDING_WRITE。
  - Ownership 收集：`llama_kv_release_collect_ownership` 逐 block 标记 owned/shared，含 invalid mapping 检测。
  - Destructive release gating：`llama_kv_destructive_release_can_enable` = paged ∧ ingraph ∧ layers_supported ∧ row_idx ∧ ¬swap。
  - `paged_release_blocks()` 仅释放 !owned 且非 SWAPPED 的 block；owned block 始终保护。
  - Transactional write：`paged_finish_write_transaction(success=true)` → PENDING_WRITE→RESIDENT（commit）；`success=false` → PENDING_WRITE→RELEASED + madvise + free list（rollback）。
  - Dummy row redirect：SWAPPED/RELEASED block 的非 active visible 行重定向到 resident dummy row；active visible 行被阻止并计数 `active_row_dummy_redirect_blocked`。
  - R0–R5/N0–N2 长门禁 artifact `/root/oscomp/kv_logs/kv_paged_release_20260717T134951Z_6071`：parser exit 0、overall result PASS、全部 passing case 输出 byte-exact 一致。
- 以下 Stage 2 事实仍然成立（本轮未修改，仅核对账本一致性）：
  - Backing store 固定槽位、block I/O 原子发布、active 恢复错误传播至 graph 前失败。
  - P0 build/backing-store unit、B2B、I/O fault 与 duration stability 回归已提交。
  - E0-E5 parser fail-closed 门禁已就绪（`d9d2e3b80`）。
  - Opt-in Dense Flex、MoE-Buffer/CLG、paged KV 路径均通过环境变量显式启用。

## Implemented but Evidence Insufficient

- 当前 HEAD 的 E0-E5 正式模型矩阵（RUNS=3）仍未运行；旧 dry-run artifact 绑定 `a4cd67e61` 不适用于当前 HEAD。
- README 与 `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 的历史性能数字缺少仓库内原始 artifact、hash 和 commit/worktree 绑定。
- `results/kv_baseline/` 的单次 ctx512 baseline 缺少 commit、构建信息和完整机器信息。
- Stage 2 E2G/E2I controlled A/B 仅覆盖单机 CPU、单模型、ctx 2048、固定 workload；尚无多模型、长上下文、server/continuous batching 证据；p95 也仅 2/3 轮有利。

## In Progress

- 无源码、脚本或测试改动正在进行；本轮只同步四个工程账本。

## Blocked

- Stage 3A-0 无功能门禁阻塞。下一道门禁需要真实压力采样与 bounded reclaim 的实现和受控正确性/行为证据。

## Next Gate

从 RELEASED 正确性基线进入真实压力下的 bounded reclaim：在多请求/多 session 并发或无 idle→active 等压力场景中，验证 release 的选择性（不误伤 live/shared block）、RSS 回收幅度、idempotent skip 与 reuse allocation 的比率，以及事务提交/回滚在压力下的正确性。该门禁仍是 correctness 基线性质，不得作为性能结论交付。
