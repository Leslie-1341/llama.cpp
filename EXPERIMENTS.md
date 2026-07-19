# Experiment Ledger

> 追加式实验索引。只记录可追溯协议与证据入口，不复制大日志，不用未运行或失败结果支撑正式结论。

## E-0001 — 当前 HEAD 的 KV controlled E0-E5

- Status: planned
- Date: 2026-07-15
- Commit/worktree: 目标 commit `d9d2e3b80acff27b3ff793003bb1f47ee212613b`；当前 HEAD `adfe67136` 包含 release correctness 基础设施，E0-E5 正式模型矩阵尚未运行
- Runner/protocol: `scripts/kv-final-controlled-e0-e5.sh`；`docs/kv_final_controlled_e0_e5_protocol.md`
- Parser: `scripts/parse-kv-final-controlled-e0-e5.py`
- Raw artifacts: 尚未生成

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

## E-0004 — Stage 1 E0/E2/E5 单 token 诊断框架

- Status: invalid（不能作为当前 HEAD 的有效模型实验或性能结论）

_(remaining content unchanged — refer to previous version of this experiment ledger)_

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

## E-0008 — Stage 3A-1B pressure sampler-only 单元、sanitizer 与严格 warning 验证

- Status: valid（仅限 sampler-only 正确性/构建门禁）
- Date: 2026-07-18
- Commit/worktree: `befd7a8944f44528cc6a44d1968114fc1294c326`；提交时 clean
- Target/source: `test-kv-pressure-sampler`；`src/llama-kv-pressure.*`、`tests/test-kv-pressure-sampler.cpp`
- Registered result: 864/864 assertions PASS；ASan/UBSan PASS；sampler-only strict warning build PASS
- Evidence entry: commit `befd7a894` message and committed test/CMake registration
- Raw artifact: 未在工程账本登记独立原始日志或精确构建命令；本轮未重跑验证

**Question and scope**

验证默认关闭的 Linux pressure sampler 是否能在 fixture/synthetic 输入下正确处理 RSS/cgroup/PSI 读取与严格解析、cgroup v1/v2 路径解析、source 选择与动态切换、NORMAL/PRESSURE/CRITICAL/RECOVERY 转换、stale、滞回/cooldown、PSI 生命周期隔离及溢出边界；同时确认该独立目标通过 sanitizer 和严格 warning 构建。

**Correctness/build gate**

- 独立测试程序全部 assertions 通过：Total 864、Passed 864、Failed 0。
- ASan/UBSan 运行通过，无已报告 sanitizer failure。
- sampler-only 严格 warning build 通过，无已报告 warning failure。
- `src/CMakeLists.txt` 将 `llama-kv-pressure.cpp` 编入 `llama`；`tests/CMakeLists.txt` 在 UNIX 下注册 `test-kv-pressure-sampler.cpp`。
- 当前源码无 server/context/decode/reclaim 调用点；因此本条不得解释为运行时集成或内存回收验证。

**Supported conclusion and limits**

- 可支持：Stage 3A-1B sampler-only 节点已提交，提交绑定的单元、ASan/UBSan 与严格 warning 门禁记录为通过；默认关闭、source fail-closed、stale 和四态状态机具备代码与测试证据。
- 证据限制：工程账本未保存原始 stdout/stderr、编译器版本、sanitizer flags 或精确命令，因此无法从本条独立复放当次构建；结论限定为 commit-bound 验证摘要，不外推为正式 server/性能实验。
- **尚未验证真实模型状态转换。** 当前状态转换证据来自 synthetic/fixture，不代表真实 RSS/cgroup/PSI 序列。
- **尚未验证 server 时延。** 未测 TTFT、TPOT、吞吐、p50/p95/p99 或不同采样频率的累计开销。
- **尚未验证 bounded reclaim。** sampler 与 `paged_release_blocks()` 无运行时连接，未验证回收预算、RSS 降幅、并发干扰或 live/shared block 安全性。

## E-0009 — Stage 3A-1C server pressure telemetry 集成与验证协议（planned）

- Status: planned（合成测试与静态检查已通过；真实模型实验尚未运行）
- Date: 2026-07-19
- Commit/worktree: `297eed939bb830ee85d426e54b67f78322955044`；clean
- Runner: `scripts/run-server-kv-pressure-stage3a-1c.py`（10-case matrix）
- Parser: `scripts/parse-server-kv-pressure-stage3a-1c.py`（fail-closed）
- Static checks: `tests/test-server-kv-pressure-static.py`（6 tests）
- Parser synthetic negatives: `tests/test-server-kv-pressure-stage3a-1c-parser.py`（9 tests）
- C++ integration: `tests/test-server-kv-pressure.cpp`（9 tests）
- Raw artifacts: 尚未生成（planned only）

**Pre-experiment verification (已通过)**

- 6/6 静态集成检查通过：single owner、no reclaim calls、no thread/lock、structured marker fields、master switch preflight、sleep/resume lifecycle reset。
- 9/9 parser 合成负例通过：valid artifact、trigger duplicates/first-set order、SSE malformation/terminal/timing/metrics conflict/timestamp missing/insufficient samples、case file set checksum、identity drift（runner/parser/binary/model/head/dirty）、OFF telemetry ban + structured action marker ban（含 prose false-positive 豁免）、timeout/residual process、request/exit/order/strace/idle boundary、static no-mutation chain。
- 9 C++ 集成测试通过：default-off、interval config、rate-limit + idle path、first/critical/stale logging、change/periodic logging、deadline overflow、lifecycle reset、marker fields、init failure stays disabled。
- py_compile: runner、parser、static tests、parser tests 全部通过。

**Question and scope**

验证 Stage 3A-1C server pressure telemetry 集成在真实 Linux server 进程中的行为：默认关闭、OFF/ON master switch、sleep/resume 正确重建 sampler lifecycle、idle 不持续采样、strace 确认真实 procfs/cgroup openat/read/close 调用链，以及采样对 TTFT/TPOT/吞吐/p95/p99 的短回归。

**Planned matrix (10 cases)**

| Case | kind | round | order | variant | 验证目标 |
|------|------|-------|-------|---------|----------|
| ab_r1_off | ab | 1 | 1 | OFF | 默认关闭：无 telemetry marker、无 procfs 访问 |
| ab_r1_on | ab | 1 | 2 | ON | 默认打开：NORMAL→CRITICAL 转换、marker 字段完整 |
| ab_r2_on | ab | 2 | 1 | ON | A/B 重排：不同于 r1 的 ordering 仍正确 |
| ab_r2_off | ab | 2 | 2 | OFF | A/B 重排：OFF 仍无 marker |
| ab_r3_off | ab | 3 | 1 | OFF | 第三轮 OFF 稳定性（无回归） |
| ab_r3_on | ab | 3 | 2 | ON | 第三轮 ON marker 一致性 |
| on_lifecycle | lifecycle | 0 | 0 | ON | sleep 时 sampler 销毁、resume 后重建、wake completion marker 独立 |
| on_idle_250ms | idle_limit | 0 | 0 | ON | 250ms 采样间隔、idle 窗口内无新 marker |
| strace_off_correctness | strace | 0 | 1 | OFF | strace 观察：仅 /etc/localtime 等非 sampler 路径 |
| strace_on_correctness | strace | 0 | 2 | ON | strace 观察：含 /proc/self/statm、/proc/pressure/memory、cgroup memory.* |

**Forced lifecycle thresholds (NOT deployment values)**

RSS thresholds 1/2/3 KiB 仅为强制 NORMAL→PRESSURE→CRITICAL 状态转换的验证值；不得作为经验性压力限制或部署推荐。`pressure_settings_purpose` marker 封印：`FORCED_LIFECYCLE_STATE_VALIDATION_ONLY_NOT_REAL_DEPLOYMENT_THRESHOLDS`。

**Environment and workload**

- Binary: 当前 HEAD Release build, GCC 11.4, GGML CPU/OpenMP/native, 12 logical CPU (Xeon Platinum 8358), Ubuntu 22.04/Linux 5.15, ~24 GB RAM。
- Model: 待定（推荐 Meta-Llama-3-8B-Instruct Q4_K_M）。
- Workload: ctx 1024, batch/ubatch 128, parallel 1, seed 1, temp 0, n-predict 32。
- Prompt: "In one short sentence, explain why deterministic tests are useful."
- Timeout: 430s/case total；startup 5s、health 180s、completion 180s、sleep 30s、resume 180s、shutdown 15s。
- Clean worktree 要求：`LLAMA_KV_PAGED_RELEASE=0`、`LLAMA_KV_PAGED_SWAP=0` 及全部 KV mutation 环境变量强制为零。

**Correctness gate (parser fail-closed)**

- 每个 case exit 0（正常 shutdown）；无 SIGKILL 使用（`sigkill_used=false`）；无 residual process。
- 所有 phase 状态为 PASS（或 lifecycle case 的 sleep/resume 为 NOT_APPLICABLE 豁免）。
- OFF variant：server stderr 中零 `kv_pressure_telemetry` marker。
- ON variant：至少一个 marker 含 `state=CRITICAL`、`trigger=first,state,source`、`sample_valid=1`、`stale=0`、`config_valid=1`。
- lifecycle case：pre-sleep 和 post-resume 各至少一个 marker，且 post-resume marker 的 `sample_count` 重置为 < pre-sleep（证明 lifecycle 隔离）。
- idle_limit case：idle 窗口（≥1500ms）内 stderr 无新增 marker。
- strace_on_correctness：trace 含 `/proc/self/statm` 或 cgroup `memory.current` 路径。
- strace_off_correctness：trace 不含 sampler procfs/cgroup 路径。
- identity drift（manifest head/binary/model/runner/parser hash）、case 顺序/端口重复、SSE reconstruction、metrics/sse_chunk 一致性、structured action marker（paged_release_blocks/swap/prefetch/madvise/reclaim）零出现，全部 fail-closed。
- 性能结论强制为 `EXPLORATORY_ONLY_NO_FORMAL_BENEFIT_CLAIM`；tail quantiles 记录为 `NOT_REPORTED`。

**Planned exit criteria**

1. Parser exit 0、artifact status `VALID`。
2. 全部 10 cases PASS（含 lifecycle 的 sleep/resume 和 idle 的 window check）。
3. TTFT、TPOT、吞吐、p95/p99 在 OFF vs ON 之间无显著退化（短回归判据：中位数偏移 <5% 且 p95 偏移 <10%）。
4. Strace 归因确认 ON 读取预期 procfs/cgroup 路径，OFF 不读取。
5. 原始 artifact 含完整 inventory.sha256.json、manifest.json、per-case logs 和 summary.json。

**Current status and next step**

- 合成验证（9 parser + 6 static + 9 C++ integration + py_compile）已通过，标记为 planned 实验的前置门禁。
- 真实 server + 真实模型尚未运行；需用户编译当前 HEAD、准备模型文件并执行 `run-server-kv-pressure-stage3a-1c.py`。
- 本条在真实模型实验完成并取得 parser PASS 结果前不得升级为 valid。