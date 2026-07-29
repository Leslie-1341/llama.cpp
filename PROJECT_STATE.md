# Project State

> 当前项目快照。只记录可验证事实；历史决策和实验索引分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-28
- Evidence commit: `a94381a31ea7f127352d996861355559aac2b469`（Stage 3B-2A clean-HEAD formal validation）
- Runtime implementation commit: `a94381a31`（Stage 3B-2A long-context release-only protocol）
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean；`git status --short` 为空，artifact 中 `tracked.diff` SHA-256 为空 diff
- Upstream relation: 目前无法确认

## Goal

面向内存受限的 Linux CPU 大模型推理，在不破坏模型输出和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 KV Cache 的物理驻留，并逐步形成权重–KV 协同的内存预算与 I/O 调度系统。

## Current Stage

**Current Stage：Stage 3C-1 — 统一 KV 动作仲裁（最小闭环）。** Stage 3B-2A 已作为 release-only 稳定节点关闭：在 clean committed HEAD `a94381a31` 上，Meta-Llama-3-8B-Instruct Q4_K_M 的 effective-context probe、独立 token/RSS calibration、OFF/DYNAMIC long-context ladder 和 20 次连续请求均完成；有效档位为 `1024/2048/4096/8064`，OFF/DYNAMIC 全部 HTTP 200，连续请求 20/20，`RUN_RC=0`、`PARSER_RC=0`。

该结论是 **clean-HEAD archival correctness and limited-stability PASS**，只覆盖 release-only 的单模型、单次完整协议边界。Stage 3C-1 转向单 owner、单 slot 的最小统一 KV 动作仲裁；不再单独开展 release-only 的容量上限、持续压力或正式性能阶段。它不构成正式性能、总 RSS 收益、多模型或长期稳定性结论。

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
- Stage 3C-1B 已验证 core 内的统一 action request/result 与单 action transaction 边界；server 对五类 action 的仲裁、互斥/优先级和真实路径错误传播仍尚未实现或验证。权重–KV 共享预算/I/O 仲裁也尚未实现或验证。
- Stage 3A-2C 的 dirty-tree 单次固定-target 结论保留为历史 diagnostic；新 clean-HEAD artifact 不自动将其推广为生产策略。

## In Progress

**Stage 3C-1B core action 已关闭；进入 Stage 3C-1C server arbitration。** 下一节点在既有单 owner、单 slot server 路径上，按 D-0014 authority 只提交 logical intent 给 core；当前不新增异步线程，也不接入权重模块。

## Blocked

无已确认的 Stage 3C-1C server arbitration runtime 阻塞项；但 server 仲裁尚未实现，不能将 Stage 3C-1B core 单测推广为 server 证据。

## Next Gate

### Stage 3C-1C — 单 owner、单 slot 的 server arbitration

1. 将已有 server 路径改为只基于不可变 pressure/slot 快照选择 `NOOP/EVALUATE/PREFETCH/RELEASE/OFFLOAD` 的 logical intent，并为每个 decision 调用一次 core `execute_action()`；server 不读取或改写 private physical block state。
2. 保持 core 独占 physical candidate 解析、ownership/recheck、state transition、backing-store authority 与 transaction ID；同一 decision 不得组合多个 state-changing action，release shortfall 仅交由后续 decision 重新评估。
3. 验证单 owner、单 slot server 的 response correctness、error propagation、safe no-op、decision/result correlation 与 fail-stop；不把 core 单测替代为真实模型或长期/多 slot 证据。

原定 release-only 的持续压力、重复 episode、多 slot、并发及正式 KV-only 性能矩阵合并后移至 **Stage 3C-2**；Stage 3C-2 之前不再单独开展 release-only 容量上限、持续压力或正式性能阶段。
