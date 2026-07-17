# Experiment Ledger

> 追加式实验索引。只记录可追溯协议与证据入口，不复制大日志，不用未运行或失败结果支撑正式结论。

## E-0007 — Stage 3A-0 RELEASED 生命周期 R0–R5/N0–N2 长门禁

- Status: valid
- Date: 2026-07-17
- Commit/worktree: `adfe671367f0cdc17327786c2b5c6182939cbf09`；manifest 记录 clean worktree
- Runner/parser: `scripts/run-kv-paged-release-correctness.sh` / `scripts/parse-kv-paged-release-correctness.py`
- Raw artifact: `/root/oscomp/kv_logs/kv_paged_release_20260717T134951Z_6071`
- Evidence hashes: manifest 记录 protocol `kv_paged_release_correctness` version 2 dry_run=0；identity 文件记录 binary/model/runner/parser SHA256

**Question and protocol**

验证 destructive release 的正确性基线：选择性（owned/shared block 保护）、事务原子性（PENDING_WRITE commit/rollback）、互斥门禁（swap、non-paged、non-ingraph、non-F32 K/V 下 release 禁用）、idempotent skip、reuse allocation 提交，以及 force-active-release violation 的 fail-closed 行为。

固定 matrix: R0, R1, R2, R3, R4, R5, N0, N1, N2，每 case 各一次运行。

| Case | 配置 | 验证目标 |
|------|------|----------|
| R0 | release=0, mincore=0 | baseline: 无 release，输出作为 byte-exact 参照 |
| R1 | release=1, mincore=1 | 基本 release: unused block 回收，mincore 采样 before/after/reaccess |
| R2 | release=1, mincore=1, repeat=1 | 重复 release: idempotent skip 正确计数，无重复 madvise |
| R3 | release=1, mincore=1, reuse=1 | reuse allocation: PENDING_WRITE→commit→RESIDENT，fresh write verify |
| R4 | release=1, mincore=1, share_seq0=1 | shared 保护: shared block 仍被 owned gate 保护，不误伤 |
| R5 | release=1, mincore=0, force_active_release=1 seq=1 | fail-closed: active release violation 触发，事务 rollback 正确，exit 1（预期失败） |
| N0 | release=1, mincore=0, ingraph=0 | 互斥: ingraph 关闭时 release 自动禁用 |
| N1 | release=1, mincore=0, K/V f16 | 互斥: 非 F32 K/V 时 release 自动禁用 |
| N2 | release=1, mincore=0, paged=0 | 互斥: paged 关闭时所有机制禁用 |

**Environment and workload**

Binary: Release build, GCC 11.4, GGML CPU/OpenMP/native, 12 logical CPU (Xeon Platinum 8358), Ubuntu 22.04/Linux 5.15, ~24 GB RAM.
Model: Meta-Llama-3-8B-Instruct Q4_K_M, ctx 1024, n-predict 16, batch/ubatch 128, parallel 4, seed 1, temp 0, K/V F32 (N1 除外 f16), unified KV.
Warmup tokens: 64, idle seqs: 2.
Timeout: 900 s/case, non-dry-run.

**Correctness gate**

- 每 case exit 0（R5 除外：EXPECTED_FAILURE，exit 1）。
- 全部 passing case (R0–R4, N0–N2) 的 Seq0/Seq1 SHA256 与 R0 baseline 精确一致。
- `contract=no_backing_unrecoverable_dead_or_unused_only` marker 存在。
- 安全字段（`active_nonresident_fatal`, `active_release_violation`, `identity_fail`, `input_setup_fatal`, `mapping_oob_fail`, `logical_mapping_fail`, `backing_metadata_stale`）全部为 0。
- Release cases (R1–R4) 必须实际触发 release（`release_calls > 0`, `released_blocks > 0`）。
- Neg cases (N0–N2) 必须实际禁用 release（`release_enabled=0`）。
- R5 必须触发 `active_release_violation=1` 且 `force_active_release_triggers=1`。
- Parser: fail-closed——任一 case 状态非 PASS（且非 EXPECTED_FAILURE）、环境/commit/worktree/binary/model hash 失配、必要 marker 缺失、重复 marker/key、per-run artifact hash 不完整时，非零退出。

**Key mincore / correctness results**

All passing cases: Seq0 SHA256 `cbb2257d...`、Seq1 SHA256 `b9f8c2d9...`——全部与 R0 baseline byte-exact 一致。

| Case | released_blocks | released_unused | released_dead | idempotent_skips | reuse_allocations | write_commits | write_rollbacks |
|------|-----------------|-----------------|---------------|------------------|--------------------|---------------|-----------------|
| R0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| R1 | 62 | 62 | 0 | 9,020 | 12 | 12 | 0 |
| R2 | 62 | 62 | 0 | 9,082 | 12 | 12 | 0 |
| R3 | 62 | 62 | 0 | 9,020 | 12 | 12 | 0 |
| R4 | 62 | 62 | 0 | 9,020 | 12 | 12 | 0 |
| R5 | 62 | 62 | 0 | 7,400 | 10 | 10 | 0 |
| N0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| N1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| N2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- R0–R4 全部 release_calls 在 163–164 之间，released_blocks 稳定为 62，均为 UNUSED block（released_dead=0）。
- R1–R4 idempotent skips 稳定在 ~9,020–9,082，表明重复 release 调用不会重复 madvise 同一 block。
- 12 个 reuse allocation 全部 commit 成功（write_commits=12, write_rollbacks=0）。
- R5: force_active_release 正确触发 1 次 violation；release_calls 略少（131），因 active error 使 context 提前退出；2 个 reuse allocation 未完成（commit=10 vs 12 in R1–R4）。
- N0: ingraph=0 → release 禁用。
- N1: f16 K/V → layers_supported=0 → release 禁用；fresh_verify_bytes 降低（因 f16 存储减半）。
- N2: paged=0 → release 及所有 paged 机制禁用；fresh_verify_bytes=0。

R1 mincore: `mincore_before_last=243,793,920` bytes (~232 MiB) 在 madvise 前 resident；`mincore_post_total_last` 和 `mincore_reaccess_total_last=3,932,160` bytes (~3.75 MiB) reaccess 检测。Release 后 RSS 追踪由 per-case `rss_drop_last_kb`/`rss_drop_max_kb` 记录（具体值见 artifact summary.json，本轮不复制大日志）。

**Supported conclusion and limits**

- Stage 3A-0 destructive release 正确性基线通过：选择性（owned/shared 保护）、事务原子性（commit/rollback）、idempotent skip、reuse allocation、互斥门禁（swap/ingraph/layers/paged）和 force-active-release fail-closed 均以预期结果通过 R0–R5/N0–N2。
- **本阶段验证的是 release correctness 基线，不是压力调度性能结论。** 不覆盖：回收时机、并发 request interference、与 swap 的融合调度、真实 RSS 回收幅度在压力下的稳定性。
- 固定 example workload（ctx 1024、n-predict 16、parallel 4、idle seqs 2）不等同于 server continuous batching 或生产 ShareGPT trace。
- 不覆盖 GPU/device memory、multi-stream、v_trans、不同 block_size 或非 Llama 模型。
- 62 个 released block 全部是 UNUSED block（无 live owner），符合 destructive release 仅回收 no-backing dead/unused block 的契约；released_dead 在本 workload 中为 0，不表示 dead block release 路径异常——该路径的单元测试覆盖由 `tests/test-kv-paged-release-ownership.cpp` 提供。

## E-0005 — clean-HEAD paged identity E2I 功能门禁

- Status: valid
- Date: 2026-07-16
- Commit/worktree: `a744830e90969a2298785cdd994901f8f448995a`；manifest 记录 clean worktree
- Runner/parser: `scripts/kv-paged-identity-e2i.sh` / `scripts/parse-kv-paged-identity-e2i.py`
- Raw artifact: `/root/oscomp/kv_logs/kv_paged_identity_e2i_smoke_20260716T135342Z`

_(remaining content unchanged — refer to previous version of this experiment ledger)_

## E-0006 — static identity E2G/E2I 三轮 controlled A/B

- Status: valid
- Date: 2026-07-16
- Commit/worktree: `a744830e90969a2298785cdd994901f8f448995a`；manifest 记录 clean worktree
- Raw artifact: `/root/oscomp/kv_logs/kv_paged_identity_controlled_ab_20260716T142722Z`

_(remaining content unchanged — refer to previous version of this experiment ledger)_

## E-0001 — 当前 HEAD 的 KV controlled E0-E5

- Status: planned
- Date: 2026-07-15
- Commit/worktree: 目标 commit `d9d2e3b80acff27b3ff793003bb1f47ee212613b`；当前 HEAD `adfe67136` 包含 release correctness 基础设施，E0-E5 正式模型矩阵尚未运行
- Runner/protocol: `scripts/kv-final-controlled-e0-e5.sh`；`docs/kv_final_controlled_e0_e5_protocol.md`
- Parser: `scripts/parse-kv-final-controlled-e0-e5.py`
- Raw artifacts: 尚未生成

_(remaining content unchanged — refer to previous version of this experiment ledger)_

## E-0004 — Stage 1 E0/E2/E5 单 token 诊断框架

- Status: invalid（不能作为当前 HEAD 的有效模型实验或性能结论）

_(remaining content unchanged — refer to previous version of this experiment ledger)_

## E-0002 — Stage 12-C ShareGPT-backed historical report

- Status: historical-unverified

_(remaining content unchanged — refer to previous version of this experiment ledger)_

## E-0003 — 仓库内 ctx512 单次 baseline

- Status: historical-single-run

_(remaining content unchanged — refer to previous version of this experiment ledger)_

## Unconfirmed Claims Not Registered as Valid Experiments

- `README.md` 中 Dense Flex、MoE-Buffer、CLG、极端内存和"KV + flex auto"组合表缺少仓库内 raw logs、hash、运行 commit/worktree 与完整协议，目前无法确认。
- `README_KV_OPT.md` 与归档 stage docs 可作为历史设计和结果入口，但不能覆盖当前源码或替代 E-0001 的当前 HEAD 受控验证。
