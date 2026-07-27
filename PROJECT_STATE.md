# Project State

> 当前项目快照。只记录可验证事实；历史决策和实验索引分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-27
- Evidence commit: `a94381a31ea7f127352d996861355559aac2b469`（Stage 3B-2A clean-HEAD formal validation）
- Runtime implementation commit: `a94381a31`（Stage 3B-2A long-context release-only protocol）
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean；`git status --short` 为空，artifact 中 `tracked.diff` SHA-256 为空 diff
- Upstream relation: 目前无法确认

## Goal

面向内存受限的 Linux CPU 大模型推理，在不破坏模型输出和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 KV Cache 的物理驻留，并逐步形成权重–KV 协同的内存预算与 I/O 调度系统。

## Current Stage

**Stage 3B-2A 已实现并完成单模型、单次完整协议验证。** 在 clean committed HEAD `a94381a31` 上，Meta-Llama-3-8B-Instruct Q4_K_M 的 effective-context probe、独立 token/RSS calibration、OFF/DYNAMIC long-context ladder 和 20 次连续请求均完成；有效档位为 `1024/2048/4096/8064`，OFF/DYNAMIC 全部 HTTP 200，连续请求 20/20，`RUN_RC=0`、`PARSER_RC=0`。

该结论是 **clean-HEAD archival correctness and limited-stability PASS**。它不构成正式性能、总 RSS 收益、多模型或长期稳定性结论。

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
- release、swap/offload、prefetch 的统一互斥/优先级、与权重共享预算/I/O 仲裁尚未实现或验证。
- Stage 3A-2C 的 dirty-tree 单次固定-target 结论保留为历史 diagnostic；新 clean-HEAD artifact 不自动将其推广为生产策略。

## In Progress

**Agent Skill 与 Harness 收敛准备。** 目标是减少局部反复 review-fix、先证明问题真实可达、保持任务提示词精简，并让真实 producer 输出进入 fixture 回归。

仓库快照中可见 `UNVERIFIED`、known-fail 分类和更多 Harness E2E 代码，但目前缺少绑定当前 HEAD 的完整审查与测试输出，因此不能先记为稳定完成。

## Blocked

无已确认的 Stage 3B-2A runtime 阻塞项。

## Next Gate

### Stage 3B 后续覆盖与性能门禁

1. 在固定 baseline、KV-only、weight-only、combined 四组下，执行多轮交错的性能协议；明确 TTFT、TPOT、TPS、吞吐和分位数定义，并将诊断开销单独消融。
2. 扩展到至少多模型/quantization、真实内存/cgroup 压力、并发请求和重复 release/refault episode；每项均保留 clean-HEAD artifact 与 fail-closed parser verdict。
3. 验证 DYNAMIC release 的正释放 mincore drop、all-candidates-owned 安全 no-op、PID/RSS identity 与 response correctness 在上述矩阵中仍成立。
4. 在独立 KV 门禁稳定后，再建立 release、swap/offload、prefetch 与权重侧之间的共享预算和 I/O 仲裁；融合验证必须具备 baseline、KV-only、weight-only、combined 对照。
