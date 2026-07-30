# Project State

> 当前项目快照。只记录可验证事实；历史决策和实验索引分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-29
- Immutable snapshot: branch `fix/kv-p0-b1-bounded-store`，HEAD `55717bb322757a3edde73e8ce35a439f09796747`（`server: add EdgeKV pressure governor`）。
- Remote synchronization: `HEAD...origin/fix/kv-p0-b1-bounded-store = 0 0`；HEAD 已推送且与远端同一提交。
- Source validation: 验证开始前 worktree clean；`git diff --check` PASS。

## Goal

面向内存受限的 Linux CPU 大模型推理，在不破坏模型输出和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 KV Cache 的物理驻留，并逐步形成权重–KV 协同的内存预算与 I/O 调度系统。

## Current Stage

**Current Stage：Stage 3C-1C-2B-1 — EdgeKV Governor Policy + synchronous bounded OFFLOAD 已关闭为 code-level stable；下一门禁为 Stage 3C-1C-2B-1R real-model multi-slot Governor integration gate。** Stage 3B-2A 的 release-only clean-HEAD archival correctness and limited-stability 结论保持不变；2B-1 不构成真实模型 HTTP、真实压力/RSS、多 slot、长周期或性能结论。

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

- Git identity：runtime implementation 为 `64301af3db0a33974269a9fb30760ca22fb9f1ac`，静态测试同步为 `ca4c952101656b078d1d1169efb781e3aa8981b1`；其历史短验证结论保持不变。
- unified 路径默认关闭，必须显式 `LLAMA_KV_PRESSURE_UNIFIED_ACTION=1`，并要求非零的 target/max-blocks；与 legacy release/dry-run/bounded-release 冲突或无效配置时 fail-closed。
- 有效、非 stale 的 `PRESSURE/CRITICAL` 采样以同一 decision ID 先 `EVALUATE`，再至多一次 `RELEASE`；server 不拥有 physical candidate、state 或 backing I/O authority。

### Stage 3C-1C-2B-1 — EdgeKV Governor Policy + synchronous bounded OFFLOAD

- Immutable snapshot：branch `fix/kv-p0-b1-bounded-store`，HEAD `55717bb322757a3edde73e8ce35a439f09796747`；工作树在验证开始前 clean，`HEAD...origin/fix/kv-p0-b1-bounded-store = 0 0`，已提交并推送。
- Governor 每个 sample 使用 immutable pressure snapshot 与 logical claimant snapshots。server policy state 保存 pressure debt/episode、basis generation、OFFLOAD arm、backoff、claimant epoch/exhaustion 与 I/O failure penalty；stale sample hold，NORMAL/reset 清空这些 policy 状态。
- 同一 decision 只会改变一次状态：先 `EVALUATE`，有 debt 时 RELEASE 优先；core RELEASE 为 `no_candidate` 只 arm 后续 decision 的 OFFLOAD，绝不在本 decision 内链式 OFFLOAD。后续 eligible sample 才同步提交 bounded OFFLOAD。
- claimant 仅由逻辑 snapshot 确定性评分：idle age、logical KV、reclaimable bytes 减去 LCP、I/O cost、failure penalty；active/protected/shared/write-open/fail-stop/empty/epoch mismatch/exhausted 被排除。排序与输入顺序无关，epoch 复用解除 exhaustion。
- server 仅传递 logical sequence、budget、KV claimant 与 capacity-write I/O class；core 独占 physical candidate、ownership/recheck、state transition、backing I/O、transaction/outcome/reason/fail-stop。core `relieved_bytes` 是唯一 debt 偿还来源；零 relief/I/O failure 不还债，并推进 exhaustion 或 backoff。
- `./build/bin/test-server-kv-pressure-action`：**325/325 PASS**；六项定向 CTest：**6/6 PASS，0 failed**；完整命令、scope 和 Gate artifact 见 E-0017。该节点仅为 code-level stable。

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
- Stage 3C-1C-2A/2B-1 已通过 code-level 短验证，但真实模型 HTTP、真实压力/RSS、多 slot、长周期/重复 episode、并发、真实 server OFFLOAD→PREFETCH、长上下文、性能与权重–KV 融合仍未验证；静态/定向 CTest 不得推广为这些结论。`all_required` 的 restore tail 与 Governor 的 I/O/claimant score 都是未量化的 P1 性能边界。
- Stage 3A-2C 的 dirty-tree 单次固定-target 结论保留为历史 diagnostic；新 clean-HEAD artifact 不自动将其推广为生产策略。

## In Progress

**Stage 3C-1C-2B-1 已关闭为 code-level stable；进入 Stage 3C-1C-2B-1R real-model multi-slot Governor integration gate。** 该门禁验证已实现结构在真实模型服务器路径中的正确性与互斥，不产生性能收益结论。

## Blocked

无已确认的 code-level P0 阻塞项。Governor 的真实模型、多 slot、真实 pressure/RSS、恢复链路、长周期与性能尚未验证，不能宣称已关闭这些 runtime/性能结论。

## Next Gate

### Stage 3C-1C-2B-1R — real-model multi-slot Governor integration gate

1. 在真实模型、多 slot server 下，制造可审计的双 claimant 场景，验证确定性 claimant 推进、exhaustion/epoch reuse 与旧路径互斥。
2. 验证一次 bounded OFFLOAD 实际改变多个 block state，且 pressure debt、core `relieved_bytes`、server marker、decision/claimant epoch 一致；记录真实 RSS/pressure 为观测而非 lifecycle truth。
3. 证明 OFFLOAD 后 request-resume 的 correctness-required PREFETCH 在 graph/decode 前完成，I/O failure/shortfall 保持 fail-closed；核对 HTTP 输出与 OFF baseline 的正确性要求。
4. 覆盖持续/重复 episode 与 reset/backoff 的长周期行为。性能、KV-only/weight-only/combined 四组对照及 Dense/MoE 融合仍需独立正式实验协议。