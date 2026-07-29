# Project State

> 当前项目快照。只记录可验证事实；历史决策和实验索引分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-29
- Evidence commit: `ca4c952101656b078d1d1169efb781e3aa8981b1`（Stage 3C-1C-2A static-test synchronization）
- Runtime implementation commit: `64301af3db0a33974269a9fb30760ca22fb9f1ac`（Stage 3C-1C-2A unified pressure action）；`ca4c95210` 补齐静态测试接口
- Branch: `fix/kv-p0-b1-bounded-store`
- Source-validation worktree: clean；`git status --short` 为空，`git diff --check` 通过
- Upstream relation: `HEAD...origin/fix/kv-p0-b1-bounded-store = 0 0`；当前 HEAD 已推送且与远端同一提交

## Goal

面向内存受限的 Linux CPU 大模型推理，在不破坏模型输出和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 KV Cache 的物理驻留，并逐步形成权重–KV 协同的内存预算与 I/O 调度系统。

## Current Stage

**Current Stage：Stage 3C-1C-2A — unified pressure `EVALUATE→RELEASE` code-level stable node 已关闭；下一门禁为 Stage 3C-1C-2B-0 EdgeKV Governor v1 architecture audit 与 implementation-contract freeze。** Stage 3B-2A 的 release-only clean-HEAD archival correctness and limited-stability 结论保持不变；2A 只关闭单 owner、单 slot、显式 opt-in 的 server/core unified pressure action 结构与短验证，不构成真实模型 HTTP、真实压力、多 slot、长周期或性能结论。

## Stage 3A-2C Verified Result

- Model: Meta-Llama-3-8B-Instruct Q4_K_M
- Backend: Linux CPU；实际日志为 Intel Xeon Silver 4316
- Workload: `ctx=1024`、单 slot、单请求、13-token prompt、32-token generation、seed 1
- Pressure: RSS 1/2/3 KiB forced thresholds，仅用于强制进入 CRITICAL，不是部署阈值
- Budget: target `33,554,432` bytes（32 MiB），`max_scan_blocks=64`
- OFF: zero bounded-release marker/counter；存在进程级后台 `MADV_DONTNEED`，不用于目标 release 归因
- DRY: 产生 would-release 预测；zero bounded destructive release/counter；响应与 OFF 字节一致
- BOUNDED: `released_bytes=35,389,440`、`released_blocks=9`、`overshoot_bytes=1,835,008`
- Safety: `ownership_aborted=0`、`madvise_failures=0`，active/mapping/write/fail-stop violation 均未出现
- Physical observation: `mincore_before=268,173,312`、`mincore_after=232,783,872`，本次 observed drop 为 `35,389,440` bytes
- Primary attribution: `bounded_cnt_bytes_delta=35,389,440`、`bounded_cnt_blocks_delta=9`，与 core result 一致
- Correctness: OFF/DRY/BOUNDED response byte-identical，均为 183 chars
- Parser: exit 0，verdict PASS；runner exit 0；process cleanup PASS
- Artifact: `/root/oscomp/kv_logs/kv_bounded_release_stage3a_2c_20260723T161538Z_cffe4f5aea_ba3095de5079`
- Capture mode: `diagnostic_dirty`

## Implemented and Verified

### Stage 3B-2A — clean-HEAD long-context release-only validation

- 正式 artifact：`/root/oscomp/kv_logs/kv_bounded_release_stage3b_2a_20260727T115210Z_a94381a31e_d82cf2a44751`；`capture_mode=archival_clean`，HEAD 为 `a94381a31ea7f127352d996861355559aac2b469`，source snapshot 的 tracked diff 为空。
- effective-context probe 将请求档位 `1024/2048/4096/8192` 收敛为有效档位 `1024/2048/4096/8064`；最大 prompt 为 8064 tokens，`n_predict=32`。
- token calibration 四档均为 exact token count（delta 0）；RSS calibration 每档独立启动真实 `llama-server`、记录 PID identity/RSS，并均完成 HTTP 200。
- OFF 与 DYNAMIC_RELEASE 在四个有效档位均 HTTP 200，响应分别与该档 OFF baseline 一致；DYNAMIC 的 release action 与“all candidates owned”导致的 safe zero-release no-op 由 parser 分开验证。
- 连续 DYNAMIC_RELEASE 20/20 请求全部 HTTP 200、与 OFF baseline 一致、`cumulative_error_count=0`；runner `run_complete`、`RUN_RC=0`、parser `PASS`、`PARSER_RC=0`。
- 当前证据只关闭该单模型、单次完整协议的正确性和有限稳定性门禁。

### Stage 3C-1B — unified KV core action stable node

- Git identity：source validation 在 branch `fix/kv-p0-b1-bounded-store` 的 clean committed HEAD `bd418879aaf08154d67e1ea2f0722d8853f2159a` 上执行（写入账本前 `git status --short` 为空）。本轮账本同步后工作树仅因这四份账本 dirty；本次提交文件为 `src/llama-kv-cache-action.h`、`src/llama-kv-cache.cpp`、`src/llama-kv-cache.h`、`tests/test-kv-paged-release-bounded.cpp`。
- Core 已提供统一 `llama_kv_action_request/result` 与 `execute_action()`，覆盖 `NOOP/EVALUATE/PREFETCH/RELEASE/OFFLOAD`；`EVALUATE` 为只读，state-changing action 返回同一 decision 的 core transaction ID。
- `PREFETCH` 发生部分 backing-store read failure 时返回 `partial_failure`、已完成 block/byte、准确 shortfall 与 `fail_stop`，已恢复 block 保持 `RESIDENT`，失败 block 保持 `SWAPPED`；首次失败则返回 `failed` 且不创建 transaction。
- 定向测试 `./build/bin/test-kv-paged-release-bounded` 在该 HEAD 通过（末行 `PASS: paged release bounded correctness`）：WT24 验证 EVALUATE/RELEASE，WT25 验证 OFFLOAD→PREFETCH 的真实 K/V byte-exact roundtrip，WT26 覆盖 swap-out 与 backing-store read fault、partial PREFETCH failure 和 fail-stop。
- 这是 clean-HEAD core 单测证据，不是 server、真实模型、多 slot、长周期或性能验证。

### Stage 3C-1C-1 — server request resume gate stable node

- Git identity：在分支 `fix/kv-p0-b1-bounded-store` 的 clean committed HEAD `d3bc743d9479e4e46a1941373d5b4ba1a2c49b77` 上验证；写入账本前 `git status --short` 为空。本轮账本同步后工作树仅因四份账本 dirty。
- `server::update_slots()` 在 `n_past` 确定后、batch setup/graph compute 前调用 resume gate；gate 仅提交 logical `PREFETCH` intent，`correctness_required=true`、`all_required=true`，不读取或改写 private physical KV state。
- 仅当 decision ID 匹配、outcome 为 `completed` 或 `no_op`、无 I/O failure/fail-stop/context-invalid 且 `shortfall_bytes==0` 时允许 graph；其余情形释放 slot 并跳过本轮 graph/decode。sequence protection 在 gate 前设定，且只由 prompt clear 或 slot release 生命周期清除，不受 pressure gate 控制。
- Release 构建成功；定向 CTest 3/3 PASS：`test-kv-paged-release-bounded`、`test-server-kv-resume`、`test-server-kv-resume-static`。这是 clean-HEAD 的构建与定向单测证据，不是 HTTP 或真实模型验证。

### Stage 3C-1C-2A — unified pressure `EVALUATE→RELEASE` code-level stable node

- Git identity：runtime implementation 为 `64301af3db0a33974269a9fb30760ca22fb9f1ac`，静态测试同步为当前 clean committed HEAD `ca4c952101656b078d1d1169efb781e3aa8981b1`；branch 为 `fix/kv-p0-b1-bounded-store`，与 `origin/fix/kv-p0-b1-bounded-store` ahead/behind 均为 0。验证开始时工作树 clean，`git diff --check` 通过。
- unified 路径默认关闭，必须显式 `LLAMA_KV_PRESSURE_UNIFIED_ACTION=1`，并要求非零的 target/max-blocks；与 `LLAMA_KV_PAGED_RELEASE`、`LLAMA_KV_PRESSURE_DRY_RUN` 或 `LLAMA_KV_PRESSURE_BOUNDED_RELEASE` 同时请求时，以及解析、范围或缺失配置无效时，startup decision 以 disabled/invalid/conflict fail-closed，不进入 action。
- 有效、非 stale 的 `PRESSURE/CRITICAL` 采样仅在 server 提交同一 decision ID 的只读 `EVALUATE` 后，且 core capability/结果允许时，至多提交一次 `RELEASE`；decision mismatch、context-invalid、write-transaction-open、fail-stop、unsupported 或 evaluation rejection 均不 release。server 的稳定提交原因是 `release_submitted`，physical candidate、ownership/recheck、state transition、backing I/O、core reason 与 transaction ID 仍由 core 权威返回。
- 短验证入口：`python3 tests/test-server-kv-pressure-static.py` 为 37/37 PASS；定向 `ctest --test-dir build --output-on-failure -R '^(test-kv-paged-release-bounded|test-server-kv-pressure|test-server-kv-resume|test-server-kv-pressure-action|test-server-kv-pressure-static|test-server-kv-resume-static|test-server-kv-pressure-action-static)$'` 为 7/7 PASS、0 failed。账本同步 review Gate artifact 为 `/tmp/os-agent-gate/gate-review-20260729T105915Z-6555`（`OS_AGENT_GATE_RESULT ... verdict=PASS`；仅覆盖四份账本 diff，不替代 source/test evidence）。

### Stage 3A-2C — pressure-driven bounded destructive release

- Server `maybe_sample_kv_pressure()` 已形成三阶段：Phase A telemetry、Phase B dry-run、Phase C bounded release。
- Phase C 使用独立环境配置、capability、cooldown/backoff 和 6 道门控；server 只调用公开 memory API，不读取 KV core private block state。
- Legacy bounded 与 server bounded 共用 `paged_release_blocks_bounded_impl()`；原 unbounded、bounded、dry-run 三条 release 路径均拒绝 `RELEASED/SWAPPED/PENDING_WRITE/INVALID`，并在 fail-stop context 下拒绝执行。
- 两个 destructive 路径在 `madvise` 前重新读取 context validity 与 block state。当前 server queue 为单 scheduler owner，ownership collection 到 `madvise` 之间不存在并发 mutation 窗口。
- Hybrid/recurrent 等不支持 memory wrapper 通过默认 capability 返回 `not_paged`，server 不会把 unsupported 误判为 supported-no-candidate。
- Directed tests: `test-kv-paged-release-bounded` WT0–WT23 PASS。
- Server integration: `test-server-kv-pressure` 183/183 PASS；static test 36/36 PASS。
- F2 parser tests: 35/35 PASS；runner tests: 8/8 PASS。
- Incremental builds and agent gates PASS；最终 F1+F2 review PASS。

### Earlier stable nodes

- Stage 3A-2B pressure-driven dry-run：真实 server OFF/ON artifact PASS，would-release 可见、response identity、zero target destructive action。
- Stage 3A-2A bounded release core primitive：固定 target/max-scan、shortfall/overshoot、ownership abort 和 fault seams 已实现。
- Stage 3A-1C server pressure telemetry：10/10 cases PASS；仅支持 correctness，性能为 exploratory-only。
- Stage 3A-1B sampler-only：864/864 assertions、ASan/UBSan 和严格 warning 验证通过。
- Stage 3A-0 RELEASED lifecycle：R0–R5/N0–N2 correctness artifact valid。
- Stage 2 static paged identity fast path：功能门禁 PASS，controlled A/B judgment `MIXED`，决定保留。
- Stage 2 及更早：fixed-slot backing store、block I/O 原子发布、active restore error propagation、E0–E5 fail-closed parser 基础均已提交。
- Harness v1：commit `fd51455b7` 的 15/15 E2E 结果为已登记稳定证据。

## Implemented but Evidence Insufficient

- Stage 3B-2A 虽覆盖至有效 8064-token 档位和一次 20-request 连续序列，但仍只是一台 Linux CPU、单 slot、单模型、单次运行；未覆盖并发、多模型/quantization、不同 block/page size 或长期重复 episode。
- DYNAMIC target 的机制与 marker 已受 parser 验证，但本 artifact 不得用于声称正式总 RSS 收益、回收率、TTFT/TPOT/TPS、吞吐或 p95/p99 改善。
- RSS 是独立真实 server PID 的进程观测；它不替代 KV resident/mincore 或 lifecycle state truth。正释放的 mincore 下降是本协议的正确性证据，非性能指标。
- Stage 3C-1C-2A 已实现并通过 code-level 短验证，但真实模型 HTTP、真实压力触发、多 slot、长周期/重复 episode、并发、真实 server OFFLOAD→PREFETCH、长上下文、性能与权重–KV 融合仍未验证；本节点的静态/定向 CTest 不得推广为这些结论。`all_required` 可能多恢复 tail block，仍是未量化的 P1 性能边界。
- EdgeKV Governor v1 尚未实现，也没有收益证据；其 pressure debt/水位、slot 生命周期与复用信号、backing I/O 拆分、预测 PREFETCH、共享 memory claimant/I/O priority/byte budget 和线程边界仅是下一 architecture audit 的审计对象。
- Stage 3A-2C 的 dirty-tree 单次固定-target 结论保留为历史 diagnostic；新 clean-HEAD artifact 不自动将其推广为生产策略。

## In Progress

**Stage 3C-1C-2A 已关闭；进入 Stage 3C-1C-2B-0 EdgeKV Governor v1 architecture audit 与 implementation-contract freeze。** 本阶段仅审计并冻结可实现边界，不实现 Governor，也不声称收益。

## Blocked

无已确认的 2A runtime 阻塞项。Governor 设计尚未审计，故不存在可宣称已关闭的 Governor 实现/性能结论。

## Next Gate

### Stage 3C-1C-2B-0 — EdgeKV Governor v1 architecture audit 与 implementation-contract freeze

1. 审计 pressure debt 与 high/low-water 的可观测来源、更新时机、饱和/复位语义，并与现有 pressure telemetry、decision ID 和 fail-stop 语义对齐。
2. 审计 slot 生命周期、prompt clear/release/复用信号，定位 backing I/O 的可拆分点，并冻结预测性 `PREFETCH` 的 logical request/result 接口，不把 physical block authority 上移到 server。
3. 定义未来 Dense/MoE 可复用的 memory claimant、I/O priority 与 byte-budget 接口边界，明确现有 KV core/server 的所有权、可回退路径和线程/锁风险。
4. 产出 architecture audit 与 implementation contract；不得实现 Governor、不得冻结简单 round-robin `OFFLOAD`，也不得以此阶段声称真实压力、性能或融合收益。

原定持续压力、重复 episode、多 slot、并发与正式 KV-only/combined 性能矩阵继续后移至后续验证阶段，需另有模型/服务器 artifact。