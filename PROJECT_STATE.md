# Project State

> 当前项目快照。只记录可验证事实；历史决策和实验索引分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-24
- Evidence commit: `0fe0aed12e7f324d3959c40e6c04985a4e2af31e`（composite pressure trigger parser 修复）
- Runtime implementation commit: `cffe4f5ae`（Stage 3A-2C bounded destructive release）
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: 本文件生成时无法读取服务器当前 Git 状态；提交前必须由用户执行 `git status --short` 确认
- Upstream relation: 目前无法确认

## Goal

面向内存受限的 Linux CPU 大模型推理，在不破坏模型输出和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 KV Cache 的物理驻留，并逐步形成权重–KV 协同的内存预算与 I/O 调度系统。

## Current Stage

**Stage 3A-2C 已完成。** 当前结论限定为：在单机 CPU、单 slot、单请求、forced CRITICAL、固定 32 MiB target 的受控场景中，server 能安全执行 bounded destructive release，真实降低 KV resident pages，并保持 OFF/DRY/BOUNDED 三路响应字节一致。

该结论是 **dirty-tree real-model diagnostic PASS**，不是 clean-HEAD archival 证据，也不是正式内存或性能收益结论。

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

- Stage 3A-2C 只覆盖单模型、单请求、forced threshold、ctx 1024、固定 target；不代表真实压力、长上下文或并发场景。
- 本次约 33.75 MiB 下降证明 KV resident pages 可按预算真实回收；不等同于整体 RSS 大幅下降。该值约占本次 mincore 可观测 KV resident 的 13%，只占约 8.4 GiB server RSS 的约 0.4%。
- 未形成正式 TTFT、TPOT、TPS、吞吐、p95/p99 或长期稳定性对照；单次 timing 不能用于性能结论。
- `mincore` 本次恰好与 core result 精确一致，但 v4 协议只把它作为独立物理观测，不要求普遍精确相等。
- Process-wide strace 包含 llama.cpp 其他后台 `MADV_DONTNEED`；只能证明 syscall 存在，不能按字节归因 bounded release。
- Bounded release cooldown/backoff、真实 pressure thresholds、dynamic target 尚未调优。
- 多请求并发下 ownership、安全性、scheduler latency 和 release/refault 代价尚未验证。
- 与 swap/offload/prefetch 的统一互斥和优先级只存在目标契约，尚未形成统一 runtime scheduler。
- 当前 HEAD 的正式 E0–E5 三轮模型矩阵仍未运行。
- Stage 3A-2A 原始 E-0010 的模型 Part B 当时因环境变量缺失而 SKIP；当前 WT0–WT23 短验证已通过，但没有为原 E-0010 另建 clean archival artifact。

## In Progress

**Agent Skill 与 Harness 收敛准备。** 目标是减少局部反复 review-fix、先证明问题真实可达、保持任务提示词精简，并让真实 producer 输出进入 fixture 回归。

仓库快照中可见 `UNVERIFIED`、known-fail 分类和更多 Harness E2E 代码，但目前缺少绑定当前 HEAD 的完整审查与测试输出，因此不能先记为稳定完成。

## Blocked

无 Stage 3A-2C runtime 阻塞项。

## Next Gate

### Immediate gate — Skill + Harness

1. Skill 强制任务先定义阶段目标、支持边界和真实触发路径，再产生 implement/review-fix。
2. 提示词只保留目标、验收和特殊边界，不重复 Skill 已承担的固定规则。
3. Harness 区分 PASS、FAIL、UNRESOLVED、UNVERIFIED、INCOMPLETE、NO_CHANGES；必要检查被跳过时不能返回 PASS。
4. Fixture 必须覆盖真实 producer 日志形态，包括组合 trigger。
5. 完成短验证后形成独立 infra commit 并 push。

### Following gate — Stage 3B

1. 用真实内存上限和真实 workload 替代 forced 1/2/3 KiB thresholds。
2. 根据当前压力、目标水位和可回收量计算 dynamic target。
3. 调优 cooldown/backoff，量化 scheduler scan 与 mincore 开销。
4. 验证多请求、长上下文下 ownership、安全性、RSS/KV resident、TTFT/TPOT/TPS 和恢复代价。
5. 建立 release、swap/offload、prefetch 的统一互斥和优先级，再进入权重–KV 共享预算与 I/O 仲裁。
