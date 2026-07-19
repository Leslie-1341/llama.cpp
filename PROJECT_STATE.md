# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-19
- Evidence commit: `726d977b26bba375edd8e79c1c04a46cece942e4`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 6

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

**Stage 3A-1C 只读 server pressure telemetry 门禁已通过。** HEAD `726d977b2` 在 `297eed939` 基础上修复三处集成缺陷（`31e81656d` pre-request 首 marker、`ad92e603f` lifecycle 首 marker、`726d977b2` strace 观测窗），形成最终 VALID artifact。10-case 真实 server + Meta-Llama-3-8B Q4_K_M 实验全部 PASS：OFF/ON correctness、sleep/resume lifecycle 隔离、idle 限频、strace 归因均验证通过。性能数据已收集但标记为 EXPLORATORY_ONLY（n=3，不做正式收益声明）。全链路不连接 reclaim、swap、prefetch 或 bounded-store 决策。

## Implemented and Verified

- **Stage 3A-1C server-only telemetry 集成**（`6f59f66b6`）：
  - 单 owner：server `update_slots()` 独占 sampler 的 `sample()` 调用，无 background thread、无 mutex。
  - 限频采样：`server_kv_pressure_runtime::sample_due()` 在 owner 边界按可配置间隔（默认 250ms，最小 100ms）限频；skip 计数记录在 telemetry marker。
  - 默认关闭：`init_kv_pressure_sampler()` 先通过 `kv_pressure_sampler_environment_enablement()` 检查 master switch，未设置 `LLAMA_KV_PRESSURE_SAMPLER=1` 时 sampler 不创建。
  - sleep/resume 重建：`handle_sleeping_state()` 在 sleep 时 destroy sampler 并 disable runtime；resume 后 `init_kv_pressure_sampler()` 重建全新 lifecycle，不携带旧 pressure state、counters 或 deadline。
  - idle 路径：`maybe_sample_kv_pressure(all_idle)` 在 idle 检查前调用；idle 标记进入 event 但 `sample_due()` 仍按时间门控，idle 不持续采样。
  - 与 reclaim 完全解耦：`maybe_sample_kv_pressure()` 仅调用 `sampler->sample()`、`runtime.record_sample()` 和 `server_kv_pressure_format_marker()` 输出日志；源码无 `paged_release_blocks`、swap、prefetch、madvise 调用。
  - 配置 fail-closed：invalid interval/cadence 或 sampler init 失败时保持 disabled，仅输出 warning。
  - Go 验证：`test-server-kv-pressure.cpp`（9 C++ 集成回归）、`test-server-kv-pressure-static.py`（6 静态集成检查）已通过并注册于 CMake。
- **Stage 3A-1C 验证协议**（`297eed939`）：
  - Runner: `scripts/run-server-kv-pressure-stage3a-1c.py` — 10-case 矩阵（6 A/B OFF/ON、lifecycle、idle_limit、strace OFF/ON）；clean worktree、固定 binary/model framework hash、bounded process cleanup、SIGKILL 兜底。
  - Parser: `scripts/parse-server-kv-pressure-stage3a-1c.py` — fail-closed artifact 验证：identity 漂移、case 顺序/端口重复、timeout、residual process、SSE 重构、metrics/telemetry 一致性、strace 归因、structured action marker 禁止。
  - Parser 合成测试: `tests/test-server-kv-pressure-stage3a-1c-parser.py` — 9 tests 通过。
  - py_compile: runner、parser、static tests、parser tests 全部通过。
- **Stage 3A-1B pressure sampler-only**（`befd7a894`）：
  - 864/864 assertions、ASan/UBSan、sampler-only strict warning build 全部通过。
  - 四态状态机（NORMAL/PRESSURE/CRITICAL/RECOVERY）、source 选择、stale、滞回/cooldown、PSI upgrade 隔离已实现并通过 fixture/synthetic 验证。
- **Stage 3A-0 RELEASED 生命周期**（`73c2aa9ef`、`708626cd0`、`adfe67136`）：
  - R0–R5/N0–N2 长门禁 artifact：parser exit 0、overall result PASS、全部 passing case 输出 byte-exact 一致。
- **Stage 3A-1C 真实 server pressure telemetry VALID**（HEAD `726d977b2`）：
  - Artifact: `/root/oscomp/kv_logs/server_kv_pressure_stage3a_1c_20260719T134156Z_726d977b26bb`，parser exit 0、status VALID。
  - 10/10 cases PASS：6 A/B OFF/ON 三轮交错、lifecycle sleep/resume 隔离、idle_limit 250ms 窗口、strace OFF/ON correctness。
  - Strace 归因：ON 确认读取 `/proc/self/statm`、`/proc/pressure/memory`、cgroup `memory.{current,high,max,pressure}`；OFF 零 sampler 路径。
  - Lifecycle：sleep 销毁 sampler + runtime disable、resume 全新 `init_kv_pressure_sampler()`、post-resume `sample_count` 独立重置。
  - Idle 限频观测：request-driven 首样本后 1.5s 持续 idle 无 periodic telemetry marker（已知限制：当前 workload 无 active idle scheduler tick）。
  - OFF variant：零 `kv_pressure_telemetry` marker；zero structured action marker（paged_release_blocks/swap/prefetch/madvise/reclaim）。
  - Performance: EXPLORATORY_ONLY（n=3，tail quantiles NOT_REPORTED，不声明正式收益）。
  - Pressure thresholds 1/2/3 KiB: FORCED_LIFECYCLE_STATE_VALIDATION_ONLY，非部署推荐。
  - 此前失败 artifacts 保留为 INVALID 诊断证据：`server_kv_pressure_stage3a_1c_20260719T073412Z_4ac1919ec2ed`、`…T082058Z_31e81656d1e6`、`…T125819Z_ad92e603f7b8`（runner 10/10 cases 完成但 parser 因 strace/pre-request/strace 缺陷拒收；`ad92e603f7b8` 的失败为 ON strace 缺少 sampler procfs 读取，与 lifecycle post-resume marker 无关）。
- Stage 2 static paged identity fast path（`a744830e9`）已保留并通过 E2I 功能门禁与三轮 E2G/E2I controlled A/B。
- 以下 Stage 2 事实仍然成立（本轮未修改）：
  - Backing store 固定槽位、block I/O 原子发布、active 恢复错误传播至 graph 前失败。
  - P0 build/backing-store unit、B2B、I/O fault 与 duration stability 回归已提交。
  - E0-E5 parser fail-closed 门禁已就绪（`d9d2e3b80`）。
  - Opt-in Dense Flex、MoE-Buffer/CLG、paged KV 路径均通过环境变量显式启用。

## Implemented but Evidence Insufficient

- bounded reclaim 尚未接入，pressure state 与 `paged_release_blocks()` 之间没有运行时控制链；回收选择性、上限、RSS 降幅和并发干扰均未验证。
- Stage 3A-1C 性能结论仅为 EXPLORATORY_ONLY（n=3，单模型、单提示、无并发请求、无长上下文）；TTFT/TPOT/吞吐的正式性能结论需更大规模实验矩阵支撑。
- Stage 3A-1B 尚未在真实模型/server workload 中观察 NORMAL/PRESSURE/CRITICAL/RECOVERY 状态转换；当前 3A-1C artifact 使用 1/2/3 KiB forced thresholds，synthetic/fixture 单测不能替代真实 RSS/cgroup/PSI 行为证据。
- 当前 HEAD 的 E0-E5 正式模型矩阵（RUNS=3）仍未运行；旧 dry-run artifact 绑定 `a4cd67e61` 不适用于当前 HEAD。
- README 与 `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 的历史性能数字缺少仓库内原始 artifact、hash 和 commit/worktree 绑定。
- `results/kv_baseline/` 的单次 ctx512 baseline 缺少 commit、构建信息和完整机器信息。
- Stage 2 E2G/E2I controlled A/B 仅覆盖单机 CPU、单模型、ctx 2048、固定 workload。

## In Progress

- 无源码、脚本或测试改动正在进行；本轮只同步四个工程账本。

## Blocked

- 无已知功能门禁阻塞。进入 bounded reclaim 实现前需先完成 reclaim 设计（预算、候选集选择、频率、concurrency model、与 swap/prefetch 互斥契约）。

## Next Gate

**唯一下一门禁：bounded reclaim。** 基于 Stage 3A-1C 已验证的 pressure state 输入，设计并实现有界的 KV block reclaim：压力状态下以有限预算（max blocks 或 max bytes per call）、有限频率（cooldown/timer gate）、ownership gate（复用 `llama_kv_release_collect_ownership`）选择性回收 dead/unused block，并保留 shared block 保护。回收动作限定为 `paged_release_blocks()` 现有路径；不得引入新的 swap、prefetch 或 backing store 变更。验收要求：correctness 门禁（选择性、事务原子性、idempotent skip、互斥与 force-active-violation）不低于 Stage 3A-0 R0–R5/N0–N2 标准；pressure-driven release vs 无 release 的 controlled A/B 验证 RSS 降幅与 TTFT/TPOT/吞吐影响；并发 request 下 ownership 不变性保持。
