# Experiment Ledger

> 追加式实验索引。只记录可追溯协议与证据入口，不复制大日志，不用未运行或失败结果支撑正式结论。

## E-0001 — 当前 HEAD 的 KV controlled E0–E5

- Status: planned
- Date: 2026-07-15 首次入账；截至 2026-07-24 正式矩阵仍未运行
- Protocol/parser evidence commit: `d9d2e3b80acff27b3ff793003bb1f47ee212613b`
- Runner/protocol: `scripts/kv-final-controlled-e0-e5.sh`；`docs/kv_final_controlled_e0_e5_protocol.md`
- Parser: `scripts/parse-kv-final-controlled-e0-e5.py`
- Raw artifacts: 尚未生成

**Question**

在固定 idle/resume workload 下，lazy、paged gather、swap、madvise、prefetch/defer 是否正确触发，并如何影响 current RSS、active-token latency、TPS、wall time 和 backing I/O？

**Variants**

- E0: experimental KV mechanisms off
- E1: lazy clear/tail
- E2: paged row-index + in-graph gather
- E3: E2 + paged idle swap, no madvise
- E4: E3 + madvise
- E5: E4 + active delayed prefetch/defer

**Protocol**

Driver `llama-kv-idle-swap-resume`；ctx 2048、n-predict 128、batch/ubatch 128、parallel 4、seed 1、temp 0、K/V F32、unified KV。Functional smoke uses RUNS=1；formal mode uses RUNS=3 with interleaved order。Warmup 256 tokens、idle seqs 2、resume pending 96。

Correctness requires exit 0、same-round Seq0/Seq1 text and SHA256 equal to E0、all required safety/backend/I/O fields zero、case mechanism actually triggered。Missing mandatory data is UNVERIFIED；identity/order/case errors fail closed。

**Result and limits**

尚未运行，不能支持当前 HEAD 的 E0–E5 correctness 或 performance 结论。固定 example workload 也不等同于 HTTP server、ShareGPT 或 production continuous batching。
## E-0002 — Stage 12-C ShareGPT-backed historical report

- Status: historical-unverified（不能作为当前 HEAD 正式证据）
- Date: historical docs 2026-06-23 to 2026-06-24；2026-07-15 入账
- Commit/worktree: 无法确认
- Runner/report: `docs/reproduce_kv_cache_optimization.md`、`examples/kv-trace-replay/`、`scripts/kv_trace_from_sharegpt.py`、`docs/kv_trace_replay_stage12c_real_sharegpt_results.md`
- Raw artifacts/hash index: 仓库内缺失

**Question and workload**

在 ShareGPT-backed synthetic multi-session idle/resume trace 下，fast-maintenance Paged-KV 是否降低 current RSS，并控制 TPS 与 resume first-token 回退？ShareGPT 只提供文本/多轮结构，arrival、idle 和 concurrency 为合成。

**Historical report**

文档报告 Ubuntu 22.04、Linux 5.15、x86-64 8 cores/23 GB、SSD、Llama-3-8B Q4_K_M、K/V F32，并给出 baseline、aggressive S5、fast-maintenance V4/V5、ctx8192、mincore diagnostic 和组件消融。

Reported values: V5 ctx4096 RSS drop 611.883 MiB、TPS delta -3.007%；ctx8192 RSS drop 1619.195 MiB、TPS delta -1.006%。

**Supported conclusion and limits**

只能证明仓库历史文档曾报告这些口径和值。缺少 raw logs、binary/model hashes、run commit/worktree 和完整 run order，不能用于当前 HEAD 正式正确性、性能或可复现性结论。
## E-0003 — 仓库内 ctx512 单次 baseline

- Status: historical-single-run / invalid for comparison
- Date: 无法确认；2026-07-15 入账
- Commit/worktree: 无法确认
- Raw files: `results/kv_baseline/ctx512_run1.log`、`ctx512_run1.time`、`baseline_summary.csv`、`run_meta.txt`

**Workload**

Llama-3-8B-Instruct Q4_K_M、CPU、threads 12、ngl 0、ctx 512、n_predict 16、seed 42、prompt `Hello, how are you?`。OS、CPU、compiler/build、model/binary hash 和 commit 缺失。

**Observed single-run result**

- max RSS: 8,192,968 KiB
- prompt throughput: 42.90 tokens/s
- decode throughput: 15.37 tokens/s
- total: 1,145.84 ms

**Supported conclusion and limits**

只能证明一次未绑定 commit 的 baseline invocation 成功。单次运行、无 variant、无 correctness comparison，不能支持性能比较或回归。
## Unconfirmed Claims Not Registered as Valid Experiments

- `README.md` 中 Dense Flex、MoE-Buffer、CLG、极端内存和“KV + flex auto”组合表缺少当前仓库可核对的 raw logs、hash、run commit/worktree 和完整协议，不能覆盖正式受控实验。
- `README_KV_OPT.md` 与归档 stage docs 可作为历史设计/结果入口，但不能替代当前 HEAD artifact。
- 理论估计、单次最好结果、exit 0、marker 出现、内部 counter 下降或 gate PASS 均不能单独成为正式性能结论。
## E-0004 — Stage 1 E0/E2/E5 single-turn diagnostic

- Status: diagnostic-valid for bottleneck localization；invalid for formal performance
- Date: 2026-07-15 to 2026-07-16
- Artifact: `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_20260715T165518Z_104953`
- Runner/parser: `scripts/kv-e0-e2-e5-single-turn-diagnose.sh` / `scripts/parse-kv-e0-e2-e5-single-turn.py`

**Question**

定位 E2 paged gather 和 E5 active prefetch 的主要成本，不修改 scheduler 或 CPU kernel hot path。

**Result**

- E0/E2 correctness and mechanism checks passed, but performance attribution was explicitly disabled/unavailable and remained UNRESOLVED.
- E5 token→prefetch call→physical block→phase correlation passed.
- 17 active-token prefetch calls corresponded to 17 restored physical blocks.
- Within the measured block restore phases: unpack/tensor write-back about 70.76%, backing read about 29.20%, validate about 0.035%, commit about 0.004%.
- Diagnostic exit was nonzero because unresolved performance attribution was not converted into PASS.

**Supported conclusion and limits**

The evidence supports a Stage 1 diagnostic direction: E5 cost was dominated by unpack/write-back, then backing read; metadata commit was negligible in that run. It does not provide formal E0/E2/E5 performance comparison or p95/p99 conclusion.
## E-0005 — clean-HEAD paged identity E2I functional gate

- Status: valid
- Date: 2026-07-16
- Commit/worktree: `a744830e90969a2298785cdd994901f8f448995a`；manifest clean
- Runner/parser: `scripts/kv-paged-identity-e2i.sh` / `scripts/parse-kv-paged-identity-e2i.py`
- Artifact: `/root/oscomp/kv_logs/kv_paged_identity_e2i_smoke_20260716T135342Z`
- Manifest SHA256: `feee0b1f8951a35b597ce9ccbbe82dd0fe492bb67f90d897352bd80c90d51c82`
- Summary SHA256: `ad4276ec29f62a92d8b323892d29a889912c6285a59d9aa1f17c6890a55e8d`

**Question**

Static identity context 是否能省略 paged row-index/gather，同时保证 graph topology、mechanism activation 和 output correctness？

**Result**

Parser exit 0；functional artifact valid。Eligible E2I context 使用 continuous K/V view；不合格或动态配置保持 E2G gather。Graph reuse topology mismatch 时拒绝复用。

**Supported conclusion and limits**

支持保留 fast path 的功能正确性，不单独支持性能收益。适用范围限定为 context-lifetime static identity、supported CPU layout 和无动态 residency/mapping mutation。
## E-0006 — static identity E2G/E2I three-round controlled A/B

- Status: valid artifact；performance judgment `MIXED`
- Date: 2026-07-16
- Commit/worktree: `a744830e90969a2298785cdd994901f8f448995a`；manifest clean
- Artifact: `/root/oscomp/kv_logs/kv_paged_identity_controlled_ab_20260716T142722Z`
- Manifest SHA256: `46dff8b8f42d87435e6a9bdab3b44600cdbc0cbb1d4509a6d2ad4ad8939372bf`

**Question and protocol**

Compare paged gather E2G with static identity E2I using three paired/interleaved rounds, while requiring output correctness and actual mechanism selection.

**Result**

- TPOT, TPS and wall time were favorable to E2I in 3/3 rounds.
- p95 was favorable in 2/3 rounds.
- Strict aggregate judgment remained `MIXED`.

**Supported conclusion and limits**

The result is sufficient for the engineering decision to retain the fast path, but not for a final competition performance claim. It covers one model/workload/machine and does not establish server continuous-batching, long-context or multi-model benefit.
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
## E-0009 — Stage 3A-1C server pressure telemetry 集成与验证协议

- Status: **valid**（真实 server + Meta-Llama-3-8B Q4_K_M，10/10 cases PASS）
- Date: 2026-07-19
- Commit/worktree: `726d977b26bba375edd8e79c1c04a46cece942e4`；clean
- Runner: `scripts/run-server-kv-pressure-stage3a-1c.py`（10-case matrix；SHA256 `a01fabb2…`）
- Parser: `scripts/parse-server-kv-pressure-stage3a-1c.py`（fail-closed；SHA256 `23f59e01…`）
- Static checks: `tests/test-server-kv-pressure-static.py`（6 tests）
- Parser synthetic negatives: `tests/test-server-kv-pressure-stage3a-1c-parser.py`（9 tests）
- C++ integration: `tests/test-server-kv-pressure.cpp`（9 tests）
- Raw artifact: `/root/oscomp/kv_logs/server_kv_pressure_stage3a_1c_20260719T134156Z_726d977b26bb`
- Runner log: `/root/oscomp/kv_logs/server_kv_pressure_stage3a_1c_20260719T134156Z_726d977b26bb.runner.log`
- Evidence hashes: manifest `inventory.sha256.json` 覆盖全部 10 cases × 13 files = 130 files SHA256；binary SHA256 `f0801153…`；model SHA256 `b2f95e15…`

**Pre-experiment verification (已通过；与 planned 阶段一致)**

- 6/6 静态集成检查通过。
- 9/9 parser 合成负例通过。
- 9 C++ 集成测试通过。
- py_compile: runner、parser、static tests、parser tests 全部通过。

**Result: VALID — correctness PASS**

- Parser exit 0；artifact status `VALID`；10/10 cases PASS。
- 6 A/B rounds (r1 OFF→ON, r2 ON→OFF, r3 OFF→ON)：OFF variant 零 `kv_pressure_telemetry` marker；ON variant 均产生 state=CRITICAL marker（`trigger=first,state,source`，`sample_valid=1, stale=0, config_valid=1`）。
- Lifecycle (`on_lifecycle`)：pre-sleep marker `sample_count` 与 post-resume marker 独立（resume 后重置为 < pre-sleep 值）；post-sleep wake completion marker 含独立 `trigger=wake_completion`。
- Idle limit (`on_idle_250ms`)：request-driven 首样本后 1.5s 持续 idle 窗口内无新 telemetry marker；idle 标记进入 event 但不绕过 `sample_due` 时间门控。
- Strace ON: 6 条 sampler procfs/cgroup 路径确认（`/proc/self/statm`、`/proc/pressure/memory`、cgroup `memory.{current,high,max,pressure}`）。
- Strace OFF: 零 sampler procfs/cgroup 路径。
- 全部 cases: zero structured action marker（paged_release_blocks/swap/prefetch/madvise/reclaim）。

**Performance result: EXPLORATORY_ONLY**

| Metric | OFF median | ON median | OFF range | ON range |
|--------|-----------|----------|-----------|----------|
| TTFT (ms) | 233.9 | 247.8 | 233.0–236.0 | 245.3–249.7 |
| TPOT (ms) | 65.1 | 71.2 | 59.6–66.1 | 69.8–73.4 |
| Throughput (tps) | 15.4 | 14.0 | 15.1–16.8 | 13.6–14.3 |
| SSE chunk p95 (ms) | 73.6 | 75.9 | 63.4–76.8 | 75.8–78.5 |
| SSE chunk p99 (ms) | 74.0 | 78.5 | 63.7–82.4 | 78.0–79.5 |

- n=3 per variant（3 轮 A/B）；tail quantiles NOT_REPORTED（n 过小不可靠）。
- 性能结论 `EXPLORATORY_ONLY_NO_FORMAL_BENEFIT_CLAIM`：不作正式收益/退化声明。
- 观测到的 ON vs OFF 差异（TTFT +14ms、TPOT +6ms、TPS −1.4）在单请求、32 token 输出、250ms 采样间隔下获取；不推广到并发/长上下文/不同模型。

**Key correctness results**

- `pressure_settings_purpose`: `FORCED_LIFECYCLE_STATE_VALIDATION_ONLY_NOT_REAL_DEPLOYMENT_THRESHOLDS`。
- OFF variant: `config_valid=false, config_fail_reason=[not enabled]`；`sample_valid=0`。
- ON variant: `source=RSS_ABSOLUTE, sample_valid=1, stale=0, config_valid=1`；state 转换 NORMAL→PRESSURE→CRITICAL（1/2/3 KiB forced thresholds）。
- Lifecycle: sleep 期间 sampler destroyed + runtime disabled；resume 后全新 init；wake completion marker 为独立触发。
- Idle: `idle=true` marker 存在但触发为 `periodic`（非 idle 专用）；idle 不产生持续采样。

**Pre-fix INVALID artifacts (retained as diagnostic evidence)**

| Artifact | Commit | Failure reason |
|----------|--------|---------------|
| `…073412Z_4ac1919ec2ed` | `4ac1919ec2ed` | strace ON 无法观测 procfs/cgroup 路径（post-attach wait 不足 250ms sample period） |
| `…082058Z_31e81656d1e6` | `31e81656d1e6` | pre-request 首 marker 缺失（`sample_due` 在首个 request 到达前不触发） |
| `…125819Z_ad92e603f7b8` | `ad92e603f7b8` | ON strace lacks required sampler procfs reads（strace 在首次采样后 attach，采样周期 60000ms 导致 completion 结束前无后续采样；trace 文件全空） |

三个 INVALID artifact 均为 runner 10/10 cases 完成但 parser 拒绝；保留为集成缺陷定位与修复验证的诊断证据。

**Supported conclusion and limits**

- Stage 3A-1C server pressure telemetry 集成正确性门禁已通过：真实 server + Meta-Llama-3-8B Q4_K_M 下默认关闭、OFF/ON master switch、sleep/resume lifecycle 隔离、idle 限频、strace 归因全部验证通过。
- 全链路确认 pressure state 与 KV block 操作完全解耦：真实 server 下 zero structured action marker。
- **性能结论为 EXPLORATORY_ONLY**（n=3，单模型、单提示、无并发请求、ctx 1024、32 token 输出）。不构成正式 TTFT/TPOT/吞吐收益或退化声明。
- Pressure thresholds 1/2/3 KiB 为 forced lifecycle validation，非部署推荐值。
- 不覆盖：并发请求、长上下文、真实内存压力下的 CRITICAL 状态转换、不同模型/quantization、GPU/device memory。
## E-0010 — Stage 3A-2A bounded release 原语正确性测试

- Status: **registered**（test binary build 通过、CTest 注册、Part A 无需模型可运行；Part B 因环境缺少 `LLAMACPP_TEST_MODELFILE` 而 SKIP）
- Date: 2026-07-20
- Commit/worktree: `949fbd0c850b7413302907df20cc4022f623d8db`；clean（untracked `Testing/` 不与测试重叠）
- Target/source: `test-kv-paged-release-bounded`（CTest #29）；`src/llama-kv-cache.{h,cpp}`、`tests/test-kv-paged-release-bounded.cpp`
- Build artifact: `build/bin/test-kv-paged-release-bounded` 存在
- CTest: 注册为 test #29（label: `main`），无模型时返回 SKIP_EXIT_CODE (77)
- Registered result: build 通过、`git diff --check` 通过、CTest 注册通过；Part A ownership fault fixture（4 组）在 build 中 verified-compiles；Part B 正式 run 日志未产生

**Question and scope**

验证 `paged_release_blocks_bounded()` 的 per-call budget 控制、状态门禁（ownership ABORT、owned skip、PENDING_WRITE/SWAPPED/RELEASED state skip）、madvise failure 恢复、test seam single-shot 自复位、active-owned block 保护（logits consistency）、shortfall/overshoot/idempotent 语义，以及 ownership ABORT 的零 state change 不变性。

**Test structure**

| Part | 名称 | 模型需求 | 验证目标 |
|------|------|----------|----------|
| A | FAULT_1–4 | 无 | 合成 ownership fault fixture：identity valid、invalid mapping detected、all invalid、OOB block |
| B1 | target=0 | 是 | 零 budget 立即返回，零副作用 |
| B2 | max_scan=0 | 是 | 零 scan budget → exhausted + full shortfall |
| B3 | overshoot | 是 | target=1 → 至少 1 block released，overshoot == released - target |
| B4 | scan budget | 是 | max_scan_blocks=2 → blocks_scanned==2, exhausted==true |
| B5 | ownership ABORT | 是（fresh ctx） | force_ownership_abort seam → ABORT，zero released，pre/post state+free-list snapshot 不变，seam auto-reset |
| B6 | PENDING_WRITE gate | 是（fresh ctx） | block_state_override=4 on dead block → skipped by state gate（非 owned gate），real state+free-list 不变 |
| B7 | madvise failure | 是（fresh ctx） | madvise_fail_block injection → block state unchanged，scan continued，subsequent blocks released，seam auto-reset |
| B8 | active-owned | 是（dual ctx） | bounded release between prompt and continuation → zero blocks released，blocks_skipped_owned>0，dual-context logits byte-exact match |
| B9 | shortfall | 是 | all blocks RELEASED → exhausted==true，shortfall==target-released |
| B10 | idempotent | 是 | repeat call on all-RELEASED → released_blocks==0，same skipped_state count both calls |
| B11 | scan precision | 是 | max_scan=3 → blocks_scanned==3；target=0,max_scan=0 corner |

**Correctness gate (Part B)**

- B1–B4、B8–B11 各 CHECK 不触发 `failures++`。
- B5：ownership_aborted=true、released_blocks==0、released_bytes==0、blocks_scanned==0、pre/post state snapshot（4 blocks）+ free list snapshot 全等、全部三个 seam 自动复位。
- B6：ownership_aborted=false、blocks_skipped_owned==0（证明 block 不受 ownership gate 保护）、blocks_skipped_state>=1、real state+free list 不变、override 自动复位。
- B7：madvise_fail_block==-1 post-call、blocks_scanned>1（扫描继续）、released_blocks>0（后续 block 释放）、madvise_failures==1、failed block state+free list 不变。
- B8：released_blocks==0、released_bytes==0、blocks_skipped_owned>0、ctx1 vs ctx2 logits 全词汇表精确一致。
- B9–B10：exhausted shortfall 正确计算、repeat release 幂等。
- 全部 B 组：无 fake pass（`failures` 计数器全局递增，最终非零则 exit 1）。

**Environment**

Binary: Release build, GCC 11.4, GGML CPU/OpenMP/native, 12 logical CPU (Xeon Platinum 8358), Ubuntu 22.04/Linux 5.15, ~24 GB RAM.
Model requirement: GGUF 格式（推荐 Meta-Llama-3-8B-Instruct Q4_K_M），由 `LLAMACPP_TEST_MODELFILE` 环境变量或 argv[1] 指定。
Test config: `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_BLOCK_SIZE=16`；ctx 256、K/V F32。

**Known test limitations**

- `decode_prompt()` 硬编码 BOS token `128000`（Llama-3 tokenizer）；非 Llama-3 模型运行时 BOS token 无效，decode 失败。
- B5–B7 需要 fresh context（避免 all-RELEASED block reuse edge case），函数内创建独立 `ContextGuard`。
- B8 创建 dual context（baseline + variant），各 prompt→release→continuation 管线隔离。
- Part B 全部 11 组均需模型；无模型时 test 返回 SKIP_EXIT_CODE (77)，CTest 标记为 `Skipped`（非 FAIL）。

**Supported conclusion and limits**

- 可支持：Stage 3A-2A bounded release 原语已提交并通过 build/CTest 注册/`git diff --check`；Part A ownership fault fixture 在 build 中 verified-compiles；4 条退出路径的 seam 自动复位有源码级覆盖。
- 证据限制：Part B 正式 run 因缺少模型而 SKIP——本条目为 `registered` 状态，不得升级为 `valid` 直至提供模型运行并通过全部 B1–B11 assertions。
- **不覆盖**：并发 request、server scheduler 集成、pressure-driven 触发、RSS 降幅、TTFT/TPOT/吞吐影响、长上下文、不同模型/quantization/block_size。

**Later status addendum — Stage 3A-2C (2026-07-23)**

Stage 3A-2C 的定向短验证已运行当前扩展后的 WT0–WT23 并全部通过，同时 server integration/static tests 通过。E-0010 仍保留 `registered`，因为它记录的是 `949fbd0c8` 当时的独立 Part B 归档状态；后续短验证不能反向伪造该旧 artifact。端到端 server 证据见 E-0012。
## E-0011 — Stage 3A-2B pressure-driven KV reclaim dry-run OFF/ON controlled A/B

- Status: **valid**（真实 server + Meta-Llama-3-8B Q4_K_M，OFF/ON controlled A/B，parser exit 0、verdict PASS）
- Date: 2026-07-20
- Commit/worktree: `02f8cd5ed33a13ffac3cce444d6d3ea7eb987650`；clean
- Runner: `scripts/run-kv-dry-run-stage3a-2b.py`（OFF/ON paired A/B）
- Parser: `scripts/parse-kv-dry-run-stage3a-2b.py`（fail-closed）
- Parser synthetic negatives: `tests/test-kv-dry-run-stage3a-2b-parser.py`
- Static checks: `tests/test-server-kv-pressure-static.py`（含 dry-run decoupling、marker field schema、config isolation 检查）
- C++ integration: `tests/test-server-kv-pressure.cpp`（含 dry-run config 解析、cooldown/backoff、state-entry、skip-reason 路径）
- Raw artifact: `/root/oscomp/kv_logs/kv_dry_run_stage3a_2b_20260720T161209Z_02f8cd5ed3`
- Evidence hashes: manifest `manifest.json` 覆盖 OFF/ON case 的 binary/model/parser SHA256；summary `summary.json` verdict PASS

**Question and protocol**

验证 pressure-driven dry-run 控制链路：在 forced CRITICAL 状态下，dry-run scanner 是否能正确执行只读候选 block 评估、产出 would-release 预测，同时保持零 KV state mutation、零 MADV_DONTNEED、零 destructive release，且不伤害推理正确性。

Fixed OFF/ON paired protocol: OFF variant（`LLAMA_KV_PRESSURE_DRY_RUN=0`）vs ON variant（`LLAMA_KV_PRESSURE_DRY_RUN=1`、`TARGET_BYTES=33554432`（32 MiB）、`MAX_SCAN_BLOCKS=64`、`COOLDOWN_MS=500`、`BACKOFF_MS=5000`）。两者共享 `LLAMA_KV_PRESSURE_SAMPLER=1` + forced thresholds（1/2/3 KiB RSS）触发 CRITICAL 状态。ON strace 捕获确认 zero MADV_DONTNEED。

**Environment and workload**

Binary: Release build, GCC 11.4, GGML CPU/OpenMP/native, 12 logical CPU (Xeon Platinum 8358), Ubuntu 22.04/Linux 5.15, ~24 GB RAM.
Model: Meta-Llama-3-8B-Instruct Q4_K_M, ctx 1024, n-predict 32, seed 1, K/V F32.
Prompt: "In one short sentence, explain why deterministic tests are useful."
Timeout: 60s, non-dry-run（runner 参数命名；server 行为是 dry-run）。

**Correctness gate**

- Parser exit 0、verdict PASS。
- OFF case: zero `kv_pressure_dry_run` markers、telemetry markers present（dry-run 独立于 telemetry）。
- ON case: ≥1 `kv_pressure_dry_run` marker with `release_enabled=0`（`LLAMA_KV_PAGED_RELEASE` 未设置）、`skipped_reason=none`、`would_release_bytes>0`、`ownership_aborted=0`。
- OFF 与 ON 的 response 必须 byte-identical。
- ON strace: zero MADV_DONTNEED calls。
- 两者: zero destructive release markers（`paged_release_blocks`/`paged_block_release_bytes=[1-9]`/`paged_blocks_released=[1-9]`）。
- Zero residual processes。
- OFF/ON 环境差异仅限于 `LLAMA_KV_PRESSURE_DRY_RUN*` 系列变量。

**Key results**

| Metric | OFF | ON |
|--------|-----|----|
| dry_run markers | 0 | 7 |
| response length | 183 | 183 |
| response text | byte-identical | byte-identical |
| destructive release | NONE | NONE |
| MADV_DONTNEED (strace) | 0 | 0 |
| residual processes | 0 | 0 |

- OFF/ON response 完全一致——dry-run scanner 不影响推理输出。
- ON 产出的 7 个 dry_run marker 均由 forced CRITICAL 状态触发：`state=CRITICAL source=RSS_ABSOLUTE stale=0 release_enabled=0 would_release_bytes>0 ownership_aborted=0 skipped_reason=none`。
- `release_enabled=0` 符合预期：`LLAMA_KV_PAGED_RELEASE` 未设置，dry-run scanner 以 `disabled` 观测性状态运行（非 skip reason）。
- strace 归因零 MADV_DONTNEED——dry-run 的 `const` 只读性在 syscall 层面得到确认。

**Supported conclusion and limits**

- Stage 3A-2B pressure-driven dry-run 控制链路门禁已通过：forced CRITICAL 状态下的 would-release 预测正确产出、响应 byte-identical、零 KV state mutation（strace 确认）、零 destructive release。
- **该证据仅验证 dry-run 控制链路、只读性和协议。** 不声明真实阈值下的回收效果或性能收益。
- **不代表真实 pressure 行为**：CRITICAL 状态由 1/2/3 KiB RSS forced threshold 触发（`FORCED_LIFECYCLE_STATE_VALIDATION_ONLY_NOT_REAL_DEPLOYMENT_THRESHOLDS`），不代表真实内存压力场景下的 dry-run 触发模式或 would-release 候选分布。
- **不覆盖**：并发请求、长上下文、真实内存压力、NORMAL/RECOVERY 状态下的 dry-run 零触发（parser 验证 OFF marker=0，但未验证 NORMAL 状态下的 marker suppression）、cooldown/backoff 动态行为（forced CRITICAL 下 state 持续，cooldown 影响 marker 间隔但非本实验验证目标）、不同模型/quantization/block_size、dry-run scanner 的 CPU 开销 profiling。
- Dry-run 的 `const` 零 mutation 属性在 C++ 类型系统和源码审计层面成立；strace 的零 MADV_DONTNEED 确认了 syscall 层面的零 destructive 行为。但尚未验证 `const_cast` 绕过或其他 indirect mutation 路径是否存在。
- 本实验固定使用 32 MiB target_bytes 和 64 max_scan_blocks——这些值与 dry-run 的 would-release 预测相关，但当前 artifact 的 would-release 字节/block 计数的绝对值不构成正式容量或回收率结论。

**Relation to later gate**

Stage 3A-2C subsequently completed the destructive server path and is recorded in E-0012. E-0011 remains the independent evidence that the policy control chain could first be exercised read-only, with would-release output and no target destructive action.

## E-0012 — Stage 3A-2C bounded destructive release OFF/DRY/BOUNDED controlled validation

- Status: **valid diagnostic**（dirty-tree real-model PASS）
- Date: 2026-07-23
- Runtime base: `cffe4f5ae`
- Parser fix in working tree was later committed as `0fe0aed12`
- Capture mode: `diagnostic_dirty`；not clean-HEAD archival
- Runner: `scripts/run-kv-bounded-release-stage3a-2c.py`
- Parser: `scripts/parse-kv-bounded-release-stage3a-2c.py`
- Parser tests: `tests/test-kv-bounded-release-stage3a-2c-parser.py`（35/35 after trigger fix）
- Runner tests: 8/8
- Artifact: `/root/oscomp/kv_logs/kv_bounded_release_stage3a_2c_20260723T161538Z_cffe4f5aea_ba3095de5079`
- Console log: `/root/oscomp/kv_logs/stage3a_2c_console_20260723T161538Z.log`

**Question**

在 forced CRITICAL 下，server Phase C 是否能按固定 budget 安全执行 destructive bounded release，产生真实 KV resident-page drop，并保持与 OFF/DRY 相同的模型输出？

**Environment and workload**

- Meta-Llama-3-8B-Instruct Q4_K_M
- Linux CPU；runtime log: Intel Xeon Silver 4316
- single slot / single request
- ctx 1024、13-token prompt、32 generated tokens、seed 1、K/V F32
- prompt: `In one short sentence, explain why deterministic tests are useful.`
- forced RSS thresholds 1/2/3 KiB, only to force CRITICAL
- target 33,554,432 bytes、max_scan_blocks 64
- timeout 60 s per variant

**Variants and gates**

| Variant | Target action | Required evidence |
|---|---|---|
| OFF | bounded release disabled | zero bounded marker/counter; response baseline |
| DRY | read-only candidate evaluation | would-release marker; zero bounded destructive result/counter |
| BOUNDED | real bounded release | released bytes/blocks > 0; counter delta match; mincore physical observation; zero safety errors |

Process-wide strace is reported for all variants but is not used to attribute target bytes because llama.cpp performs background `MADV_DONTNEED` outside bounded release.

**Key result**

| Metric | OFF | DRY | BOUNDED |
|---|---:|---:|---:|
| response length | 183 | 183 | 183 |
| response identity | baseline | byte-equal | byte-equal |
| bounded release calls | 0 | 0 | 1 |
| released bytes | 0 | would-release > 0 | **35,389,440** |
| released blocks | 0 | would-release > 0 | **9** |
| ownership_aborted | — | 0 | **0** |
| madvise_failures | — | — | **0** |
| bounded counter bytes delta | 0 | 0 | **35,389,440** |
| bounded counter blocks delta | 0 | 0 | **9** |
| mincore before | — | — | 268,173,312 |
| mincore after | — | — | 232,783,872 |
| observed mincore drop | — | — | **35,389,440** |
| process cleanup | PASS | PASS | PASS |

Budget result: target 32 MiB, actual 33.75 MiB, overshoot 1,835,008 bytes due to whole-block granularity.

Process-wide strace report from the successful run:

- OFF: 50,085,888 bytes / 6 `MADV_DONTNEED` calls
- DRY: 41,738,240 bytes / 5 calls
- BOUNDED: 85,475,328 bytes / 582 calls

These values include background operations and are not bounded-release byte attribution.

**Parser verdict**

- Configuration isolation PASS
- Server topology/capability PASS
- OFF, DRY and BOUNDED case gates PASS
- Core result == independent bounded counter delta
- mincore directional/plausibility gate PASS; exact equality happened in this run but is not a universal requirement
- response identity PASS
- process cleanup PASS
- parser exit 0; runner exit 0

**Supported conclusion**

Stage 3A-2C endpoint is feasible and correct in the stated controlled boundary: pressure state can drive bounded release, 9 safe blocks were physically de-residented, no ownership abort or madvise failure occurred, and output remained byte-identical.

**Limits**

- Not real deployment pressure: CRITICAL was forced with 1/2/3 KiB thresholds.
- Not a formal performance or total-RSS result.
- Single run, single model, single request, ctx 1024, fixed target.
- Does not cover concurrency, long context, repeated episodes, dynamic target, different page/block sizes, different models, GPU, or release/offload/prefetch arbitration.
- Dirty-tree diagnostic evidence is not clean archival evidence.

## E-0013 — Stage 3B-2A clean-HEAD long-context OFF/DYNAMIC release-only validation

- Status: **valid clean-HEAD archival correctness and limited-stability evidence**
- Date: 2026-07-27
- Commit: `a94381a31ea7f127352d996861355559aac2b469` (`test: add Stage 3B-2A long-context release validation`)
- Capture mode: `archival_clean`；worktree clean，artifact `tracked.diff` 为空
- Model: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`（Meta-Llama-3-8B-Instruct，Q4_K_M；模型 SHA-256 见 manifest）
- Runner / parser: `scripts/run-kv-bounded-release-stage3b-2a.py` / `scripts/parse-kv-bounded-release-stage3b-2a.py`
- Command entry: `python3 scripts/run-kv-bounded-release-stage3b-2a.py --binary build/bin/llama-server --model /root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --output-dir <artifact>`；随后对同一 artifact 运行 parser。
- Artifact: `/root/oscomp/kv_logs/kv_bounded_release_stage3b_2a_20260727T115210Z_a94381a31e_d82cf2a44751`

**Question**

在 clean committed HEAD 上，动态 bounded release 是否能在真实 effective context 内通过四档 OFF/DYNAMIC ladder 与一次 20-request 连续序列，保持输出正确、action/no-op 语义、真实 server PID/RSS 采样和 fail-closed artifact 闭包？

**Protocol and calibration**

- effective-context probe：请求 `1024/2048/4096/8192`，实际 `n_ctx=8192`，使用 `n_predict=32`、special overhead 1、safety 95、32-token align，得到有效档位 **`1024/2048/4096/8064`**，最大 prompt 8064。
- token calibration：四档实际 token count 分别为 `1024/2048/4096/8064`，均 `delta=0`。
- RSS calibration：每档独立启动真实 `llama-server`，记录 PID identity 与 idle/peak RSS；四档 calibration completion 均 HTTP 200。对应 `ctx_size` 为 `1280/2304/4352/8320`，RSS delta 分别为 `23416/25456/41276/76688 KiB`。
- Ladder：每个有效档位执行 OFF 与 DYNAMIC_RELEASE；DYNAMIC hard cap 为 1 GiB、`max_scan_blocks=256`。连续阶段以最小合法档位的 DYNAMIC_RELEASE server 发送 20 个相同请求。

**Key result**

- 四个有效档位的 OFF/DYNAMIC_RELEASE 均 HTTP 200；每个 DYNAMIC 响应与同档 OFF baseline byte-identical。
- 20 次连续 DYNAMIC_RELEASE 请求：**20/20** completed，全部 HTTP 200、全部匹配 OFF baseline，`cumulative_error_count=0`，且每轮记录真实 `llama-server` PID/starttime/cmdline 与 RSS。
- parser 对 dynamic target、正 action 的 core/counter 与 mincore drop、以及 owned-only safe zero-release no-op 分别 fail-closed 验证；没有把 no-op 当作 release action。
- Runner status `run_complete`，**`RUN_RC=0`**；parser status `PASS`，**`PARSER_RC=0`**；failure phase/reason/target 均为空。

**Supported conclusion**

本 artifact 证明 Stage 3B-2A 已在所述单模型、单次完整协议边界内实现并验证：有效 context 探测、独立 token/RSS calibration、OFF/DYNAMIC ladder、20-request 连续正确性、真实 server PID/RSS 以及完整 artifact/parser fail-closed 生命周期。

**Limits**

- 这是正确性与有限稳定性验证，**不得**用于正式性能、总 RSS 收益、TTFT/TPOT/TPS/吞吐、多模型/quantization、并发请求或长期稳定性结论。
- 仅一次 clean-HEAD 运行；未覆盖不同 block/page size、真实生产阈值、多轮 release/refault 或 release/swap/offload/prefetch 与权重侧仲裁。
- RSS/mincore 为观测/正确性证据，不是完整 lifecycle state truth；诊断开销必须在性能协议中关闭或单独测量。

**离线定量摘要（同一 E-0013 artifact，非新增运行）**

- 四档 OFF/DYNAMIC ladder 共记录 **82** 个正 action、**20** 个 safe no-op、**628** 个 blocks；cumulative release traffic 为 **2355 MiB**，**82/82** 正 action 均匹配有效、方向正确的 `mincore` physical-drop 观测。
- 连续请求阶段为 **20/20** completed；其中 action/no-op 为 **24/2**，reuse/commit/rollback 为 **81/81/0**。
- cumulative release traffic 是本协议下累计 destructive release 字节流量，**不是**净 RSS 收益、回收率或内存节省；E-0013 不支持正式性能结论。

## E-0014 — Stage 3C-1 unified KV action evidence

- Status: **partially verified — Stage 3C-1B core directed unit test only**
- Identity: `fix/kv-p0-b1-bounded-store`，clean committed HEAD `bd418879aaf08154d67e1ea2f0722d8853f2159a`；implementation/test files: `src/llama-kv-cache-action.h`、`src/llama-kv-cache.cpp`、`src/llama-kv-cache.h`、`tests/test-kv-paged-release-bounded.cpp`。
- Command and result: `./build/bin/test-kv-paged-release-bounded`，exit 0，末行 `PASS: paged release bounded correctness`。
- Directed evidence: WT24 covers read-only EVALUATE, structured NOOP and unified RELEASE transaction correlation; WT25 covers unified OFFLOAD/PREFETCH and byte-exact real K/V tensor roundtrip; WT26 fault-injects backing-store I/O, including partial PREFETCH read failure with completed/failed block-state separation, exact shortfall and correctness-required fail-stop.
- Evidence entry only: this is no server runner/parser artifact and records no performance observation or conclusion.
- Not covered: Stage 3C-1C server arbitration/priority, response correctness in server, real-model runs, multi-slot, long-duration/repeated episodes, concurrency, performance, or weight–KV integration. These remain unimplemented or unverified as applicable; broader extensions belong to Stage 3C-2.
