# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-15
- Evidence commit: `d9d2e3b80acff27b3ff793003bb1f47ee212613b`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: dirty；无 tracked/staged diff，仅有未跟踪的 `.agents/`、`.claude/`、四个工程账本、`INSTALL.md` 和 `docs/os-agent-task-usage.md`
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 3；相对本地 `master` ahead 14

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

当前分支处于 KV P0 加固与受控验证准备阶段。`master` 之后的 14 个提交依次覆盖固定容量 backing store、paged swap 错误传播、mapping/I/O 故障注入与回归、长周期稳定性、block-level 原子 I/O 诊断、active-token latency telemetry、统一 E0-E5 runner/parser/protocol，以及 parser fail-closed 加固。提交 `d9d2e3b80` 已落实进程级 fail-closed、固定矩阵对账、必要字段/指标校验和重复 telemetry/key 拒绝；合成回归与不启动模型的 dry-run 已通过。正式模型实验尚未运行。

## Implemented and Verified

以下仅表示已在当前 HEAD 源码或 Git 元数据中静态核对，不代表本轮运行过构建、测试或模型实验：

- `src/llama-model.cpp`、`src/llama-flex.*`、`src/llama-moe-buffer.*` 和 `src/llama-window.*` 中存在 opt-in 的 Dense Flex、MoE-Buffer 与 CLG prefetch 路径；未设置对应环境变量时不启用。
- `src/llama-kv-cache.*` 与 `src/llama-graph.cpp` 中存在 paged row-index、idle block swap-out、`MADV_DONTNEED`、resume swap-in/prefetch、`mincore` 诊断和状态/安全计数器。
- backing store 使用 `n_slots * cell_stride` 固定逻辑容量；paged block 写入通过 `write_cells()` 完成，整块写成功后才发布 offsets 和 `SWAPPED` 状态。
- active 路径的 paged swap-in/mapping/input 错误可通过 `llama_memory_i` 传播到 `llama-context.cpp`，并在 graph compute 前返回失败。
- `tests/test-kv-backing-store.cpp` 已接入 `tests/CMakeLists.txt`；三个 P0 regression/stability runner 和 E0-E5 runner/parser/protocol 均已提交。
- E0-E5 parser 在任一 run 非 `PASS` 时返回非零，并对 manifest/固定 RUN_PLAN 与实际 `(round, run_order, case)` 做完整性、顺序和唯一性校验；必要 telemetry/指标缺失、重复 marker 或重复 key 均硬失败。
- `python3 tests/test-kv-final-controlled-e0-e5-parser.py` 已通过（5 tests），覆盖正常矩阵及四类异常：`UNVERIFIED`、矩阵缺失/重复、必要指标缺失、重复 telemetry marker/key。
- `DRY_RUN=1 RUNS=1 ... scripts/kv-final-controlled-e0-e5.sh` 已通过，生成并校验 6 个 planned runs，输出明确确认模型未启动。

## Implemented but Evidence Insufficient

- 当前 HEAD 的 backing-store unit、B2B、I/O fault、duration stability 与 E0-E5 模型功能/性能结论：本轮按要求未运行，仓库内也没有与 `d9d2e3b80` 绑定的模型运行结果目录。
- README 与 `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 报告的 ctx4096/ctx8192、mincore、权重侧及组合收益：当前仓库缺少对应原始日志、哈希和当时 commit/worktree，不能升级为当前 HEAD 的正式结论。
- `results/kv_baseline/` 只有一次 ctx512 baseline，元数据未记录 commit、构建信息和完整机器信息，不能支撑比较或正式性能结论。

## In Progress

- 无法确认存在未提交的源码实现工作；`git status` 只显示上述未跟踪的代理/账本文档资产。

## Blocked

- `RUNS=3 scripts/kv-final-controlled-e0-e5.sh` 默认要求 clean worktree；当前未跟踪文件会使正式运行拒绝启动，除非先由用户纳入版本控制/移出工作树，或显式使用会被记录为 dirty override 的 `ALLOW_DIRTY=1`。
- 当前 HEAD 的 P0 build、backing-store unit、B2B、I/O-fault 与 stability 回归尚未取得本轮运行证据；因此还不能进入正式 E0-E5 模型实验。

## Next Gate

先在 `d9d2e3b80` 上完成 P0 build、backing-store unit、B2B、I/O-fault 与 stability 回归，并处理正式模式的 clean-worktree 要求。上述正确性门禁通过后，再运行 E0-E5 模型功能 smoke；正式 `RUNS=3` 性能矩阵仍由用户执行。
