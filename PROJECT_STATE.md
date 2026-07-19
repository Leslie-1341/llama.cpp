# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-19
- Evidence commit: `297eed939bb830ee85d426e54b67f78322955044`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 4（`3b2502ca6`、`befd7a894`、`6f59f66b6`、`297eed939` 尚未推送）

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

**Stage 3A-1C 只读 server pressure telemetry 集成与验证协议稳定节点已提交。** clean-HEAD `297eed939` 在 sampler-only (`befd7a894`) 之上新增两笔提交：`6f59f66b6` 将 pressure sampler 以单 owner、限频、默认关闭、telemetry-only 方式接入 server scheduler；`297eed939` 提交 fail-closed 验证协议（runner + parser + 9 合成负例 + 6 静态集成检查）。三笔提交均不连接 reclaim、swap、prefetch 或 bounded-store 决策。真实模型、strace 和时延证据尚未产生。

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
- Stage 2 static paged identity fast path（`a744830e9`）已保留并通过 E2I 功能门禁与三轮 E2G/E2I controlled A/B。
- 以下 Stage 2 事实仍然成立（本轮未修改）：
  - Backing store 固定槽位、block I/O 原子发布、active 恢复错误传播至 graph 前失败。
  - P0 build/backing-store unit、B2B、I/O fault 与 duration stability 回归已提交。
  - E0-E5 parser fail-closed 门禁已就绪（`d9d2e3b80`）。
  - Opt-in Dense Flex、MoE-Buffer/CLG、paged KV 路径均通过环境变量显式启用。

## Implemented but Evidence Insufficient

- Stage 3A-1C server telemetry 集成尚未在真实模型下运行；当前证据为 C++/Python 合成测试、静态集成检查和 py_compile。
- 尚未对真实 server 进程执行 strace 验证（`strace_on_correctness`/`strace_off_correctness` case 为合成 fixture）。
- 尚未测量 server 接入后的采样频率、单次/累计采样开销及 TTFT、TPOT、吞吐、p95/p99 时延影响。
- bounded reclaim 尚未接入，pressure state 与 `paged_release_blocks()` 之间没有运行时控制链；回收选择性、上限、RSS 降幅和并发干扰均未验证。
- Stage 3A-1B 尚未在真实模型/server workload 中观察 NORMAL/PRESSURE/CRITICAL/RECOVERY 状态转换；synthetic/fixture 单测不能替代真实 RSS/cgroup/PSI 行为证据。
- 当前 HEAD 的 E0-E5 正式模型矩阵（RUNS=3）仍未运行；旧 dry-run artifact 绑定 `a4cd67e61` 不适用于当前 HEAD。
- README 与 `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 的历史性能数字缺少仓库内原始 artifact、hash 和 commit/worktree 绑定。
- `results/kv_baseline/` 的单次 ctx512 baseline 缺少 commit、构建信息和完整机器信息。
- Stage 2 E2G/E2I controlled A/B 仅覆盖单机 CPU、单模型、ctx 2048、固定 workload。

## In Progress

- 无源码、脚本或测试改动正在进行；本轮只同步四个工程账本。

## Blocked

- 无已知功能门禁阻塞。进入真实模型验证前仍缺少 strace 观测和端到端时延证据。

## Next Gate

**唯一下一门禁：clean-HEAD 真实 server OFF/ON、sleep/resume、strace 与探索性 A/B。** 使用当前 HEAD `297eed939` 编译 server、加载真实模型（如 Llama-3-8B Q4_K_M），以 `LLAMA_KV_PRESSURE_SAMPLER=0` 和 `=1` 分别运行，观测 telemetry marker、状态转换、sleep/resume 正确重建、strace 确认真实 procfs/cgroup openat/read/close 调用链，并收集 TTFT/TPOT/吞吐/p95/p99 短回归。通过后再单独设计 bounded reclaim 门禁。验证协议 runner/parser 的 10-case 矩阵作为正式实验模板，但 1/2/3 KiB RSS 阈值仅为 state lifecycle 验证值，不得作为部署推荐。
