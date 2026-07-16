# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-16
- Evidence commit: `a744830e90969a2298785cdd994901f8f448995a`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: 本轮开始时 clean；仅同步四个工程账本，不修改源码、脚本或测试
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 1；相对本地 `master` ahead 18

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

Stage 2 静态 paged identity fast path 已在 `a744830e9` 实现，通过 clean-HEAD 功能门禁，并基于三轮 E2G/E2I 受控配对证据决定保留。该路径在 context 构造期一次性判定 eligibility；仅对静态 identity mapping 使用连续 K/V view，其他配置回退到 paged row-index gather。当前证据是阶段决策级，不是决赛正式性能结论。

## Implemented and Verified

以下事实来自当前 HEAD 源码/Git 元数据及账本列明的既有 artifacts；本轮只核对证据，未重新运行构建、测试或模型实验：

- `src/llama-model.cpp`、`src/llama-flex.*`、`src/llama-moe-buffer.*` 和 `src/llama-window.*` 中存在 opt-in 的 Dense Flex、MoE-Buffer 与 CLG prefetch 路径；未设置对应环境变量时不启用。
- `src/llama-kv-cache.*` 与 `src/llama-graph.cpp` 中存在 paged row-index、idle block swap-out、`MADV_DONTNEED`、resume swap-in/prefetch、`mincore` 诊断和状态/安全计数器。
- backing store 使用 `n_slots * cell_stride` 固定逻辑容量；paged block 写入通过 `write_cells()` 完成，整块写成功后才发布 offsets 和 `SWAPPED` 状态。
- active 路径的 paged swap-in/mapping/input 错误可通过 `llama_memory_i` 传播到 `llama-context.cpp`，并在 graph compute 前返回失败。
- `tests/test-kv-backing-store.cpp` 已接入 `tests/CMakeLists.txt`；三个 P0 regression/stability runner 和 E0-E5 runner/parser/protocol 均已提交。
- E0-E5 parser 在任一 run 非 `PASS` 时返回非零，并对 manifest/固定 RUN_PLAN 与实际 `(round, run_order, case)` 做完整性、顺序和唯一性校验；必要 telemetry/指标缺失、重复 marker 或重复 key 均硬失败。
- E0-E5 parser 与其 5-case 合成测试仍在当前源码中；本轮未重跑该测试。历史 dry-run artifact `/root/oscomp/kv_logs/kv_final_controlled_e0_e5_20260715T133400Z` 绑定 `a4cd67e61`，不是当前 HEAD，且只证明未启动模型的规划路径。
- Stage 1 的 `tests/test-kv-e0-e2-e5-single-turn-parser.py` 已在当前 HEAD 通过（11 tests）；`scripts/kv-e0-e2-e5-single-turn-diagnose.sh` 已通过 `bash -n`。这些验证只覆盖 parser fixture 和 shell 语法，不包含 build 或模型执行。
- `a9faa532` 的 E2 profiler 默认关闭，仅在 `LLAMA_KV_E2_GET_ROWS_PROFILE=1` 时由 example 的 eval callback 记录“前一 callback 边界至目标 GET_ROWS 完成”的 outer segment；E5 phase trace 默认关闭，仅在 `LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=1` 时逐 block 输出 validate/read/unpack/commit，并由 parser 对账。
- `a744830e9` 增加 opt-in 的 `LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1`：只有 paged/in-graph、single-stream、非 `v_trans`、非 approximate-dynamic、F32 K/V、初始 identity mapping，且未请求 remap/swap/release/madvise/fault injection 时才启用；否则记录明确 reject reason 并保留 gather topology。
- clean-HEAD E2I 功能 artifact `/root/oscomp/kv_logs/kv_paged_identity_e2i_smoke_20260716T135342Z` 绑定 `a744830e9`、parser exit 0。E0/E2/E2I/E2I_NOREUSE 及 SHIFT/NONIDENTITY/SWAP/RELEASE/MADVISE 共 9 case 全部 PASS，输出相对 E0 byte-exact；E2I 与 E2 也 byte-exact。E2I graph reuse 为 `n_reused=762`，禁用 reuse 时为 0，所有动态/非 identity case 均按预期回退 gather。
- controlled A/B artifact `/root/oscomp/kv_logs/kv_paged_identity_controlled_ab_20260716T142722Z` 绑定同一 clean HEAD、binary/model hash 和三轮交错顺序，parser exit 0、artifact `VALID`。E2I 相对 E2G 的 TPOT、TPS、wall 三轮均有利；p95 为 2/3 轮有利。严格“四项三轮全有利”判据为 `MIXED`，但现有方向一致性与保守失效回退足以支持 Stage 2 保留决策。

## Implemented but Evidence Insufficient

- 当前 HEAD 的 backing-store unit、B2B、I/O fault、duration stability 与 E0-E5 模型功能/性能结论：本轮按要求未运行，仓库内也没有与 `d9d2e3b80` 绑定的模型运行结果目录。
- README 与 `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 报告的 ctx4096/ctx8192、mincore、权重侧及组合收益：当前仓库缺少对应原始日志、哈希和当时 commit/worktree，不能升级为当前 HEAD 的正式结论。
- `results/kv_baseline/` 只有一次 ctx512 baseline，元数据未记录 commit、构建信息和完整机器信息，不能支撑比较或正式性能结论。
- 当前受控 A/B 只覆盖单机 CPU、Llama-3-8B Q4_K_M、ctx 2048、固定 idle/resume example workload；尚无多模型、长上下文、server continuous batching、不同 backend/layout 或更大重复数证据。
- `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_*` 的五份既有 artifact 都绑定 `3ecb212c3` 而非当前 HEAD：三份非 dry-run 中两份 parser 返回 1（其中一份 summary 虽写 PASS，进程结果仍非零）且一份被 parser 以 E2 outer compute count 不匹配拒绝（返回 2）；两份返回 0 的 artifact 是 dry-run。均不能作为当前 HEAD 的有效模型实验或性能结论。

## In Progress

- 无源码、脚本或测试改动正在进行；Stage 2 已验收并决定保留，本轮只同步工程账本。

## Blocked

- Stage 2 无功能门禁阻塞。决赛正式性能结论仍缺少扩展 workload/模型/上下文/重复数及 server 场景验证；p95 也仅 2/3 轮有利。

## Next Gate

冻结并保留 Stage 2 静态 identity fast path；下一道门禁是在不改当前判定边界的前提下，将 E2G/E2I 比较扩展到决赛正式矩阵。正式表述前必须补足长上下文、更多模型/重复、server/continuous batching，并继续单独报告 p95 波动与所有退化轮次。
