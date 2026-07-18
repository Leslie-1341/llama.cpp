# Project State

> 当前项目快照。仅记录可验证事实；历史决策和实验结果分别进入 `DECISIONS.md` 与 `EXPERIMENTS.md`。

- Last updated: 2026-07-18
- Evidence commit: `befd7a8944f44528cc6a44d1968114fc1294c326`
- Branch: `fix/kv-p0-b1-bounded-store`
- Worktree: clean（账本更新前）
- Upstream relation: 相对 `origin/fix/kv-p0-b1-bounded-store` ahead 2（`3b2502ca6`、`befd7a894` 尚未推送）

## Goal

面向内存受限的 Linux CPU LLM 推理，在保留模型语义和可恢复 KV 历史的前提下，控制 Dense/MoE 权重与 idle KV Cache 的物理驻留，并建立可复现、可失败、可审计的正确性与性能证据链。

## Current Stage

**Stage 3A-1B sampler-only 稳定节点已提交并完成限定验证。** clean-HEAD `befd7a894` 新增默认关闭的 Linux RSS/cgroup/PSI 压力采样器与 NORMAL/PRESSURE/CRITICAL/RECOVERY 四态状态机；提交绑定的验证记录为 864/864 assertions、ASan/UBSan 通过、sampler-only 严格 warning build 通过。采样器当前只编入 `llama` 库并由独立单测调用，没有 server/推理运行时调用点，也不触发 reclaim、swap、prefetch 或 bounded-store 决策。

## Implemented and Verified

- **Stage 3A-1A 压力输入 probe**（`3b2502ca6`）：提供只读 cgroup v1/v2、RSS、PSI 输入探测与静态回归；只确认输入和读取代价，不实现调度或 reclaim。
- **Stage 3A-1B pressure sampler-only**（`befd7a894`）：
  - 数据源：`/proc/self/statm` RSS、cgroup v1/v2 current/max/high、cgroup v2 `memory.pressure` 与 `/proc/pressure/memory`。
  - 主状态机：NORMAL / PRESSURE / CRITICAL / RECOVERY；CRITICAL 进入立即生效，降级要求 low-water、连续有效样本、滞回和 cooldown。
  - fail-closed：选中源读取失败即标记 stale 且不自动降级；配置不完整、cgroup 路径歧义或 source 不可用时禁用 transition。
  - source：显式 RSS absolute 优先，其次 cgroup absolute，否则在 finite `memory.max` 下使用 cgroup ratio；运行中 finite/max 变化可切换 source，并重置跨 source/阈值基础的累计计数。
  - PSI 仅用于持续 PRESSURE 的确认/升级，不单独触发 destructive transition；系统 PSI 与 cgroup PSI 取较高值作为升级信号。
  - 默认关闭：未设置 `LLAMA_KV_PRESSURE_SAMPLER=1` 时保持 NORMAL；当前实现不调用任何 reclaim。
  - 提交记录验证：864/864 assertions、ASan/UBSan、sampler-only 严格 warning build 全部通过。
- Stage 2 static paged identity fast path（`a744830e9`）已保留并通过 E2I 功能门禁与三轮 E2G/E2I controlled A/B。
- **Stage 3A-0 RELEASED 生命周期**（`73c2aa9ef`、`708626cd0`、`adfe67136`）：
  - 四态 + PENDING_WRITE：UNUSED / RESIDENT / RELEASED / SWAPPED / PENDING_WRITE。
  - Ownership 收集：`llama_kv_release_collect_ownership` 逐 block 标记 owned/shared，含 invalid mapping 检测。
  - Destructive release gating：`llama_kv_destructive_release_can_enable` = paged ∧ ingraph ∧ layers_supported ∧ row_idx ∧ ¬swap。
  - `paged_release_blocks()` 仅释放 !owned 且非 SWAPPED 的 block；owned block 始终保护。
  - Transactional write：`paged_finish_write_transaction(success=true)` → PENDING_WRITE→RESIDENT（commit）；`success=false` → PENDING_WRITE→RELEASED + madvise + free list（rollback）。
  - Dummy row redirect：SWAPPED/RELEASED block 的非 active visible 行重定向到 resident dummy row；active visible 行被阻止并计数 `active_row_dummy_redirect_blocked`。
  - R0–R5/N0–N2 长门禁 artifact `/root/oscomp/kv_logs/kv_paged_release_20260717T134951Z_6071`：parser exit 0、overall result PASS、全部 passing case 输出 byte-exact 一致。
- 以下 Stage 2 事实仍然成立（本轮未修改，仅核对账本一致性）：
  - Backing store 固定槽位、block I/O 原子发布、active 恢复错误传播至 graph 前失败。
  - P0 build/backing-store unit、B2B、I/O fault 与 duration stability 回归已提交。
  - E0-E5 parser fail-closed 门禁已就绪（`d9d2e3b80`）。
  - Opt-in Dense Flex、MoE-Buffer/CLG、paged KV 路径均通过环境变量显式启用。

## Implemented but Evidence Insufficient

- Stage 3A-1B 尚未在真实模型/server workload 中观察 NORMAL/PRESSURE/CRITICAL/RECOVERY 状态转换；synthetic/fixture 单测不能替代真实 RSS/cgroup/PSI 行为证据。
- 尚未测量 server 接入后的采样频率、单次/累计采样开销及 TTFT、TPOT、吞吐、p95/p99 时延影响。
- bounded reclaim 尚未接入，压力状态与 `paged_release_blocks()` 之间没有运行时控制链；回收选择性、上限、RSS 降幅和并发干扰均未验证。
- 当前 HEAD 的 E0-E5 正式模型矩阵（RUNS=3）仍未运行；旧 dry-run artifact 绑定 `a4cd67e61` 不适用于当前 HEAD。
- README 与 `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 的历史性能数字缺少仓库内原始 artifact、hash 和 commit/worktree 绑定。
- `results/kv_baseline/` 的单次 ctx512 baseline 缺少 commit、构建信息和完整机器信息。
- Stage 2 E2G/E2I controlled A/B 仅覆盖单机 CPU、单模型、ctx 2048、固定 workload；尚无多模型、长上下文、server/continuous batching 证据；p95 也仅 2/3 轮有利。

## In Progress

- 无源码、脚本或测试改动正在进行；本轮只同步四个工程账本。

## Blocked

- sampler-only 节点无已知功能门禁阻塞。进入 reclaim 前仍缺少 server 真实生命周期中的只读状态观测与采样开销证据。

## Next Gate

**唯一下一门禁：只读 server 运行时集成。** 由单一 server/scheduler owner 在非 per-token/per-layer 热路径中限频调用 `kv_pressure_sampler::sample()`，只暴露结构化 telemetry 和状态转换，不连接 `paged_release_blocks()`、swap、prefetch 或任何 reclaim 动作。验收至少覆盖默认关闭行为、真实模型请求中的状态/源/stale 可观测性、并发生命周期安全，以及对 TTFT/TPOT/吞吐和 p95/p99 的短回归；通过后再单独设计 bounded reclaim 门禁。
