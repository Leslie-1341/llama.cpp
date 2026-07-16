# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-16
- Evidence commit: `a9faa532be58c9d821fca77df9b6bf449db3fe7b`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: dirty；仅本轮修改 `PROJECT_STATE.md`、`ARCHITECTURE.md`、`DECISIONS.md`、`EXPERIMENTS.md`，无 staged diff，源码/脚本/测试文件无修改
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 5；相对本地 `master` ahead 16

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

当前分支处于 Stage 1 诊断框架稳定节点。`master` 之后的 16 个提交覆盖固定容量 backing store、paged swap 错误传播、mapping/I/O 故障注入与回归、长周期稳定性、block-level 原子 I/O 诊断、active-token latency telemetry、统一 E0-E5 runner/parser/protocol、parser fail-closed 加固，以及 E0/E2/E5 单 token 瓶颈诊断。提交 `a9faa532` 新增默认关闭的 E2 outer-segment profiler、E5 per-block phase telemetry 及 fail-closed runner/parser；未修改 CPU backend 内层计时或 scheduler。正式模型实验尚未运行。

## Implemented and Verified

以下仅表示已在当前 HEAD 源码或 Git 元数据中静态核对，不代表本轮运行过构建、测试或模型实验：

- `src/llama-model.cpp`、`src/llama-flex.*`、`src/llama-moe-buffer.*` 和 `src/llama-window.*` 中存在 opt-in 的 Dense Flex、MoE-Buffer 与 CLG prefetch 路径；未设置对应环境变量时不启用。
- `src/llama-kv-cache.*` 与 `src/llama-graph.cpp` 中存在 paged row-index、idle block swap-out、`MADV_DONTNEED`、resume swap-in/prefetch、`mincore` 诊断和状态/安全计数器。
- backing store 使用 `n_slots * cell_stride` 固定逻辑容量；paged block 写入通过 `write_cells()` 完成，整块写成功后才发布 offsets 和 `SWAPPED` 状态。
- active 路径的 paged swap-in/mapping/input 错误可通过 `llama_memory_i` 传播到 `llama-context.cpp`，并在 graph compute 前返回失败。
- `tests/test-kv-backing-store.cpp` 已接入 `tests/CMakeLists.txt`；三个 P0 regression/stability runner 和 E0-E5 runner/parser/protocol 均已提交。
- E0-E5 parser 在任一 run 非 `PASS` 时返回非零，并对 manifest/固定 RUN_PLAN 与实际 `(round, run_order, case)` 做完整性、顺序和唯一性校验；必要 telemetry/指标缺失、重复 marker 或重复 key 均硬失败。
- E0-E5 parser 与其 5-case 合成测试仍在当前源码中；本轮未重跑该测试。历史 dry-run artifact `/root/oscomp/kv_logs/kv_final_controlled_e0_e5_20260715T133400Z` 绑定 `a4cd67e61`，不是当前 HEAD，且只证明未启动模型的规划路径。
- Stage 1 的 `tests/test-kv-e0-e2-e5-single-turn-parser.py` 已在当前 HEAD 通过（11 tests）；`scripts/kv-e0-e2-e5-single-turn-diagnose.sh` 已通过 `bash -n`。这些验证只覆盖 parser fixture 和 shell 语法，不包含 build 或模型执行。
- `a9faa532` 的 E2 profiler 默认关闭，仅在 `LLAMA_KV_E2_GET_ROWS_PROFILE=1` 时由 example 的 eval callback 记录“前一 callback 边界至目标 GET_ROWS 完成”的 outer segment；E5 phase trace 默认关闭，仅在 `LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=1` 时逐 block 输出 validate/read/unpack/commit，并由 parser 对账。

## Implemented but Evidence Insufficient

- 当前 HEAD 的 backing-store unit、B2B、I/O fault、duration stability 与 E0-E5 模型功能/性能结论：本轮按要求未运行，仓库内也没有与 `d9d2e3b80` 绑定的模型运行结果目录。
- README 与 `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 报告的 ctx4096/ctx8192、mincore、权重侧及组合收益：当前仓库缺少对应原始日志、哈希和当时 commit/worktree，不能升级为当前 HEAD 的正式结论。
- `results/kv_baseline/` 只有一次 ctx512 baseline，元数据未记录 commit、构建信息和完整机器信息，不能支撑比较或正式性能结论。
- 当前 HEAD 未运行 build、E0/E2/E5 runner dry-run 或模型 A/B；因此 Stage 1 的运行时启用、指标完整性和端到端差异尚无当前提交证据。
- `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_*` 的五份既有 artifact 都绑定 `3ecb212c3` 而非当前 HEAD：三份非 dry-run 中两份 parser 返回 1（其中一份 summary 虽写 PASS，进程结果仍非零）且一份被 parser 以 E2 outer compute count 不匹配拒绝（返回 2）；两份返回 0 的 artifact 是 dry-run。均不能作为当前 HEAD 的有效模型实验或性能结论。

## In Progress

- 无源码、脚本或测试改动正在进行；本轮账本更新前工作树 clean。

## Blocked

- 当前 HEAD 的 P0 build、backing-store unit、B2B、I/O-fault 与 stability 回归尚未取得本轮运行证据；同时 Stage 1 未有当前 HEAD 的模型 artifact。二者均限制端到端性能判断。

## Next Gate

先审计 paged identity fast path，确认最小实现位置与不引入计时扰动的正确性边界；随后在未插桩路径上执行 E0/E2/E5 端到端 A/B。该 A/B 仍以前述 P0 正确性证据和当前 HEAD 可追溯 artifact 为前提，不能从 Stage 1 fixture、dry-run 或历史诊断外推。
