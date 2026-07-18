# Decision Log

> 追加式记录。不得删除历史决策；变化通过新条目 supersede 旧条目。

## D-0001 — 按 Dense 与 MoE 访问模式拆分权重驻留机制

- Date: 2026-07-15（首次入账；实现早于本日期）
- Status: accepted
- Evidence commit/worktree: `a4cd67e61`；当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

Dense 每 token 访问全部 decoder layer，MoE 每 token 只访问 router 选中的少量 expert；二者的可利用局部性和 I/O 粒度不同。

**Decision**

Dense 使用 layer-level Flex ring；MoE 使用 expert-level anonymous buffer、budget/LRU 与可选 CLG prefetch。CLG 只做预测提示，kernel 前 callback 保留同步兜底。

**Alternatives rejected**

- 用单一 layer/expert cache 策略覆盖两类模型：无法同时表达 Dense 顺序全层访问和 MoE 稀疏 expert 访问。
- 让 CLG 预测替代真实 routing：预测失败会改变语义，当前源码明确保留同步 callback。

**Consequences and limits**

- 两条权重路径不会同时启用，当前也没有统一权重/KV I/O budget。
- 主要适用于 CPU 模型文件读取；GPU 路径目前无法确认。

**Evidence**

- `README.md` 2.2.2、2.6.1-2.6.3。
- `src/llama-model.cpp` 的 `use_flex` / `use_moe_buffer_pre` gate 与 CLG 初始化。
- `src/llama-flex.h`、`src/llama-moe-buffer.h`、`src/llama-window.cpp`。

## D-0002 — 实验内存机制保持 opt-in，默认路径不启用

- Date: 2026-07-15（首次入账；实现早于本日期）
- Status: accepted
- Evidence commit/worktree: `a4cd67e61`；当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

Flex/MoE streaming 与 paged KV 依赖模型形状、CPU host memory 和操作系统能力，不能默认外推到所有 llama.cpp backend。

**Decision**

通过 `LLAMA_FLEX`、`LLAMA_LAZY_MOE_BUFFER`、`LLAMA_LAZY_CLG`、`LLAMA_KV_PAGED` 等环境变量显式启用；未设置时保留普通路径。KV paged 还在构造阶段检查 layout/backend 前提，不满足则禁用并记录警告。

**Alternatives rejected**

- 默认全局启用实验路径：会改变未覆盖 backend/layout 的行为和性能。
- 仅依赖文档约束而不在源码 gate：不能防止不兼容配置进入机制路径。

**Consequences and limits**

- 用户必须显式选择配置；不同环境变量组合需要受控消融验证。
- 这些环境变量和 memory-level API 仍是竞赛原型接口，不代表稳定 upstream contract。

**Evidence**

- `docs/final_technical_report.md` 8.2、8.3。
- `src/llama-model.cpp` 权重 gate；`src/llama-kv-cache.cpp` paged/swap gate。

## D-0003 — core 提供 KV 机制，上层 driver/scheduler 提供生命周期策略

- Date: 2026-07-15（首次入账；实现早于本日期）
- Status: accepted
- Evidence commit/worktree: `a4cd67e61`；当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

llama.cpp core 没有完整 request-level scheduler；idle、resume-pending 和压力策略依赖 application/server 已知的 session 生命周期。

**Decision**

core 维护 block state、swap/madvise/restore/prefetch 与错误；example/trace driver 负责 workload、idle/resume 时序、prefetch timing 和 pressure policy。

**Alternatives rejected**

- 把 request policy 固化到 core decode 热路径：会耦合特定 workload，并扩大普通运行开销与接口影响。
- 让 driver 直接维护 swap 状态机：会复制 core correctness 状态和错误处理。

**Consequences and limits**

- 当前 trace replay 不是生产 server scheduler；后续 server 接入需提供真实 lifecycle 信号。
- core API 只表达机制，不能单独证明真实线上调度效果。

**Evidence**

- `docs/final_technical_report.md` 8.4、9.1。
- `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 4。
- `src/llama-memory.h` 与 `examples/kv-*`。

## D-0004 — backing store 固定槽位、block I/O 原子发布并传播 active 恢复错误

- Date: 2026-07-15
- Status: accepted
- Evidence commit/worktree: `2f5464270`、`9e8961f08`、`41cbe43c7`、`7d063f757`、`bb31d6a1f`、`663ec053e`；当前 HEAD `adfe67136`
- Supersedes: none
- Superseded by: none

**Context**

append 式 backing file 会随重复 swap 周期增长；逐 cell 直接发布会暴露部分写入状态；active swap-in/mapping 失败若仅记录计数，graph 可能继续使用不完整 KV。

**Decision**

每个物理 cell 映射到固定 slot，容量为 `n_slots * cell_stride`；paged block 先打包并通过一次 `write_cells()` 完成 range 写入，成功后统一发布 metadata/`SWAPPED`。swap-in 先读 staging 再提交；active 必需恢复或 mapping/input 失败写入 context-local error，并在 graph compute 前失败。

**Alternatives rejected**

- 按文件尾追加：长期 swap 会无界增长。
- 每写完一个 cell 就发布：中途 I/O 失败可能留下部分可见 block。
- active 恢复失败后继续 graph：可能读取未恢复或错误映射的 KV。

**Consequences and limits**

- backing file 的逻辑容量有界，但累计读写字节与 syscall 仍随周期增长。
- block I/O 使用 grow-only staging，降低反复分配但增加一个 block 大小的 host buffer。

**Evidence**

- `src/llama-kv-cache.h` 的 fixed-slot contract 与 backing stats。
- `src/llama-kv-cache.cpp` 的 `write_cells()`/`read_cells()`、全有或全无 metadata publication。
- `src/llama-memory.h`、`src/llama-context.cpp`、`src/llama-graph.cpp` 的错误传播。
- `tests/test-kv-backing-store.cpp` 与三个 `scripts/run-kv-p0-*.sh`。

## D-0005 — 当前正式 KV 对比采用固定 E0-E5 协议

- Date: 2026-07-15
- Status: accepted
- Evidence commit/worktree: `d9d2e3b80`; 当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

历史文档混合了不同 workload、统计口径和诊断开关；当前 P0 修改需要一个绑定源码、二进制、模型、环境、运行顺序和原始产物的统一比较入口。

**Decision**

使用 `docs/kv_final_controlled_e0_e5_protocol.md` 定义 E0 baseline、E1 lazy、E2 paged gather、E3 swap、E4 madvise、E5 prefetch/defer；正式运行采用三轮交错顺序，并要求同轮输出精确一致、安全字段为零、机制字段实际触发。

**Alternatives rejected**

- 直接复用历史 README 数字：缺少当前 commit/worktree 与原始 artifacts 绑定。
- 只比较单次最好结果或只看退出码/RSS：不能排除输出错误、机制未触发和运行波动。
- 把 mincore 混入性能矩阵：诊断开销会污染受控性能比较。

**Consequences and limits**

- `RUNS=3` 默认要求 clean worktree。
- E0-E5 只覆盖 `llama-kv-idle-swap-resume` 固定 workload，不等同于 server/ShareGPT/生产结论。

**Evidence**

- `docs/kv_final_controlled_e0_e5_protocol.md`。
- `scripts/kv-final-controlled-e0-e5.sh`、`scripts/parse-kv-final-controlled-e0-e5.py`。

**Historical static audit addendum — protocol requirement vs parser capability before `d9d2e3b80` (2026-07-15)**

_(content unchanged — refer to previous version of this decision log)_

**Resolution addendum — parser fail-closed gate completed (2026-07-15)**

_(content unchanged)_

## D-0006 — 撤销 CPU backend GET_ROWS 内层计时，采用最小 identity fast path 的未插桩 A/B

- Date: 2026-07-16
- Status: accepted
- Evidence commit/worktree: `a9faa532be58c9d821fca77df9b6bf449db3fe7b`；本决策入账前工作树 clean
- Supersedes: none
- Superseded by: none

**Context**

Stage 1 需要区分 E2 paged gather 的调度/图构造开销与 E5 active prefetch 的恢复阶段，但当前没有绑定真实模型、当前 HEAD 与完整 A/B 条件的证据表明 CPU backend 内层 `GET_ROWS` 是端到端瓶颈或值得为其引入计时。

**Decision**

撤销向 CPU backend `GET_ROWS` 内层插桩的方案。保留默认关闭的 E2 outer-segment profiler 和 E5 block-phase telemetry，仅用于结构定位和 fail-closed artifact 对账。下一步先审计最小 paged identity fast path；性能判断改为该 fast path 未插桩时的端到端 E0/E2/E5 A/B，并要求当前 HEAD、正确性门槛和原始 artifact 可追溯。

**Alternatives rejected**

- 在 CPU backend `GET_ROWS` 内层加入计时：热路径计时会改变待测路径，且目前缺少真实模型证据证明该内层是主要瓶颈。
- 将 E2 outer-segment 或 E5 phase telemetry 直接解释为 kernel 或端到端收益：二者的采样范围不同，不能替代未插桩 A/B。
- 修改 scheduler 以获得更细粒度事件：超出本阶段诊断与最小 fast-path 审计范围，并会改变被比较的调度行为。

**Consequences and limits**

- 当前源码没有 CPU backend 内层计时和 scheduler 改动；Stage 1 事件默认关闭。
- E2 指标是 callback 边界到目标 GET_ROWS 完成的 outer segment；E5 指标是关联 block 的 validate/read/unpack/commit 阶段，不是完整 decode latency。
- identity fast path 与未插桩 A/B 尚未实现或运行；本决策不构成性能结论。

**Evidence**

- `a9faa532`：`examples/kv-idle-swap-resume/idle-swap-resume.cpp`、`src/llama-kv-cache.*`、`scripts/kv-e0-e2-e5-single-turn-diagnose.sh`、`scripts/parse-kv-e0-e2-e5-single-turn.py`。
- 当前 HEAD：`tests/test-kv-e0-e2-e5-single-turn-parser.py` 11 tests 通过；runner `bash -n` 通过；未运行 build、模型或新的 dry-run。

## D-0007 — 保留 context-lifetime static paged identity fast path

- Date: 2026-07-16
- Status: accepted
- Evidence commit/worktree: `a744830e90969a2298785cdd994901f8f448995a`；功能与性能 artifacts 均记录 clean worktree
- Supersedes: D-0006 中"identity fast path 尚未实现或运行"的状态描述
- Superseded by: none

**Context**

E2 paged gather 在 logical-to-physical mapping 始终为 identity 时仍为每层 K/V 构造 `GET_ROWS`，增加 row-index input、gather 节点和调度开销。优化必须保证 graph cache reuse 不复用错误 topology，也不能让 swap/remap/reclaim 等动态配置错误进入连续视图。

**Decision**

保留 `a744830e9` 的 opt-in static identity fast path。eligibility 在 context 构造期一次性、fail-closed 地解析；合格 context 在整个生命周期使用 continuous K/V view 并省略 `paged_row_idx`，不合格 context 保持 E2G gather。graph reuse 显式比较 cached graph 与当前 context 是否都具有相同 row-index topology，拓扑变化时拒绝复用。

**Alternatives rejected**

- 删除 fast path、始终使用 E2G：放弃已由三轮配对端到端证据显示的稳定 TPOT/TPS/wall 方向收益。
- 每 token 动态判断 identity：把状态检查和 topology 切换带入热路径，并扩大 graph reuse 与映射失效风险。
- 在 swap/remap/release/madvise 中乐观启用 continuous view：这些机制可破坏 context-lifetime identity/residency，不具备安全失效契约。
- 以当前结果宣称决赛正式性能胜出：严格四指标门槛的结论仍为 `MIXED`，p95 只有 2/3 轮有利，且 workload/模型/机器覆盖有限。

**Consequences and limits**

- 收益：移除 eligible identity context 的 row-index input/fill 与每层 K/V gather；controlled A/B 中 TPOT、TPS、wall 三轮均有利，p95 两轮有利。
- 代价：增加 context eligibility 状态、reject telemetry、graph-topology reuse 检查和两套 graph topology；配置组合测试与维护面扩大。
- 适用范围：显式请求、paged/in-graph、single-stream、非 `v_trans`、非 approximate-dynamic、F32 K/V、静态 identity mapping，且无 remap/swap/release/madvise/fault injection。
- 失效边界：任何可能改变 mapping/residency 的新机制、unsupported layer/layout、multi-stream、GPU/device KV 或缺少 graph invalidation 契约的 topology 切换都必须回退 gather。
- 当前是 Stage 2 决策级证据；正式决赛结论仍需扩展模型、长上下文、server/continuous batching、重复数与硬件覆盖，并如实保留 p95 退化轮次。

**Evidence**

- Functional artifact: `/root/oscomp/kv_logs/kv_paged_identity_e2i_smoke_20260716T135342Z`；manifest SHA256 `feee0b1f8951a35b597ce9ccbbe82dd0fe492bb67f90d897352bd80c90d51c82`；summary SHA256 `ad4276ec29f62a92d0d8b323892d29a889912c6285a59d9aa1f17c6890a55e8d`；parser exit 0。
- Controlled A/B artifact: `/root/oscomp/kv_logs/kv_paged_identity_controlled_ab_20260716T142722Z`；manifest SHA256 `46dff8b8f42d87435e6a9bdab3b44600cdbc0cbb1d4509a6d2ad4ad8939372bf`；summary SHA256 `5eb66dc2d48cc40924170ba3763ce0bd4c31e2344c7ce157fecc318f88bcbabe`；parser exit 0、artifact `VALID`、performance judgment `MIXED`。
- Source: `src/llama-kv-cache-identity.h`、`src/llama-kv-cache.cpp`、`src/llama-graph.cpp`；unit coverage: `tests/test-kv-paged-identity-fast-path.cpp`。

## D-0008 — Destructive release 仅允许 no-backing dead/unused block；可恢复历史必须经 SWAPPED

- Date: 2026-07-17
- Status: accepted
- Evidence commit/worktree: `adfe671367f0cdc17327786c2b5c6182939cbf09`（clean）；前置实现 `73c2aa9ef`、`708626cd0`；R0–R5/N0–N2 artifact `/root/oscomp/kv_logs/kv_paged_release_20260717T134951Z_6071`
- Supersedes: none
- Superseded by: none

**Context**

idle/resume workload 中，不再被任何 sequence 引用的 block（dead）或从未被分配过的 block（unused）占据了物理内存，但没有机制回收这些页面。单纯依赖 swap 会累积 backing file I/O 并保留可恢复元数据，而真正的垃圾 block 不需要可恢复性。同时，仍有 idle/shared owner 的 block 不能直接丢弃——它们可能需要后续 swap-in 恢复。

**Decision**

1. **RELEASED 是 destructive、不可逆状态**。进入 RELEASED 的 block 页面通过 `MADV_DONTNEED` 丢弃，无 backing store 副本，旧 KV 永远不可恢复。后续 reuse 是全新分配并写入新 K/V 内容。
2. **只有无 live/owned cell 的 block 才能 release**。每次 `paged_release_blocks()` 调用前通过 `llama_kv_release_collect_ownership` 重算全量 ownership bitmap；任何 invalid mapping 导致 release ABORT 并记录 `destructive_release_skipped=1`。
3. **SWAPPED block 永远不进入 RELEASED**。可恢复 idle/shared 历史必须先经 swap-in→RESIDENT 再决策；release 遍历中遇到 SWAPPED block 直接跳过。
4. **Destructive release 与 swap 互斥**。`llama_kv_destructive_release_can_enable` 要求 `!swap_enabled`；两者同时请求时 release 自动禁用。
5. **Reuse allocation 走事务路径**。release 回收的 block 被新 K/V 写入分配后进入 PENDING_WRITE；写入成功 commit→RESIDENT，失败 rollback→RELEASED+free list。不在中间态遗留。
6. **dummy row redirect 保护 active visible 行**。SWAPPED 或 RELEASED block 的非 active visible 行在 row-index fill 时重定向到 resident dummy row；active visible 行被阻止并递增 violation 计数器。
7. **Release 是 correctness 机制，不是压力调度策略**。当前实现验证选择性、原子性和互斥门禁；回收频率、批量大小和触发时机由上层 driver 控制。

**Alternatives rejected**

- 统一用 swap + MADV_DONTNEED 回收所有 block：为无价值的 dead/unused block 引入 backing file 和元数据开销；且 swap 路径与 release 的不可逆语义冲突。
- 让 release 同时处理 SWAPPED block：SWAPPED block 的 backing store 副本是恢复的唯一权威来源，madvise 后再删除 backing 会留下无任何副本的 block；release 语义要求无 backing。
- 乐观 release 而不重算 ownership：在并发或 batch 边界可能误伤刚刚被引用的 block。
- 让 release 与 swap 共存：两者的回收范围和安全契约不同（destructive vs recoverable），强制共存会引入优先级、竞态和重复回收问题。
- 将 release 结果直接作为性能结论：当前 RESIDENT→RELEASED 的 RSS 回收是受控 correctness 验证，不是代表性压力下的回收效果。

**Consequences and limits**

- 收益：为真正 dead/unused block 提供零 backing 开销的直接物理回收路径；ownership gate 防止误伤；事务路径保证 reuse allocation 的原子性；R0–R5/N0–N2 长门禁确认选择性、互斥和回滚正确。
- 代价：每次 release 调用需遍历全部 physical block 并重算 ownership（O(n_blocks × n_seqs × cells_per_seq)）；release 与 swap 不能同时启用；R5 确认 force-active-release 场景必然失败（这是正确的 fail-closed 行为）。
- 适用范围：paged、ingraph、F32 K/V、row-index gather mode、非 swap、CPU host memory。
- 失效边界：启用 swap 时 release 禁用；GPU/device memory 不支持 `madvise`/`mincore`；非 paged 或非 ingraph 模式不支持。
- 当前验证范围：单机 CPU、ctx 1024、parallel 4、固定 idle/resume workload、R0–R5/N0–N2 correctness 矩阵。压力调度下的回收幅度、idempotent skip 与 reuse allocation 比率、事务提交/回滚在并发下的正确性尚未验证。

**Evidence**

- `src/llama-kv-cache.h:662-671`：RELEASED 声明与 paged_block_state 枚举。
- `src/llama-kv-cache-release.h:6-13`：`llama_kv_destructive_release_can_enable` 准入门禁。
- `src/llama-kv-cache-release.h:22-71`：`llama_kv_release_collect_ownership` 全量 ownership bitmap。
- `src/llama-kv-cache.cpp:6039-6140`：`paged_release_blocks()` 遍历与 selective release。
- `src/llama-kv-cache.cpp:4302-4416`：`paged_finish_write_transaction` 事务提交/回滚。
- `src/llama-kv-cache.cpp:7090-7540`：dummy row redirect for SWAPPED/RELEASED blocks。
- `scripts/run-kv-paged-release-correctness.sh`：R0–R5/N0–N2 完整矩阵 runner。
- `scripts/parse-kv-paged-release-correctness.py`：fail-closed parser。
- Artifact `/root/oscomp/kv_logs/kv_paged_release_20260717T134951Z_6071`：parser exit 0、overall PASS、全部 passing case seq0/seq1 hash 一致、R5 EXPECTED_FAILURE 正确触发。

## D-0009 — 压力采样与 reclaim 解耦，先做单 owner 限频只读 server 集成

- Date: 2026-07-18
- Status: accepted
- Evidence commit/worktree: sampler-only `befd7a8944f44528cc6a44d1968114fc1294c326`（clean）；输入 probe `3b2502ca6f0d202f380b7efbcc8b18fe8f86e059`
- Supersedes: D-0008 中“尚无真实压力采样”的阶段状态；不改变 D-0008 的 release correctness 契约
- Superseded by: none

**Context**

Stage 3A-0 已验证 destructive release 的 ownership、事务和 fail-closed 语义，但没有真实压力输入。Stage 3A-1B 已实现 RSS/cgroup/PSI 采样与四态状态机，但 server runtime 尚无调用点。若在首次接入时同时触发 reclaim，状态误判、采样开销、request 生命周期竞态和回收副作用将无法独立归因。

**Decision**

1. **采样与 reclaim 解耦。** `kv_pressure_sampler` 只负责读取、source 选择、stale/滞回/状态转换和 telemetry，不直接调用 `paged_release_blocks()`、swap、prefetch 或 bounded-store policy。
2. **先做只读 server 集成。** 下一门禁只在真实 server 生命周期中实例化并观测 sampler；任何压力状态均不得触发 reclaim。只有状态行为与时延开销通过后，才另立 bounded reclaim 门禁。
3. **单线程 owner。** 一个 server/scheduler owner 独占 sampler 的初始化、`sample()` 调用和 telemetry 发布；request worker 不并发推进同一实例的 state/counters。跨线程消费者只读取 owner 发布的快照。
4. **调用侧限频。** 不在 per-token、per-layer、attention 或 kernel 热路径采样；由 owner 按时间门限在调度/维护边界采样，并复用最近快照。具体采样周期必须由真实 server 采样时延与 TTFT/TPOT 回归决定，本决策不预设未经验证的数值。
5. **默认关闭不变。** 未显式设置 `LLAMA_KV_PRESSURE_SAMPLER=1` 时不创建运行时压力行为；配置/source 无效或 stale 时不得触发后续 destructive action。

**Alternatives rejected**

- sampler 首次接入即绑定 bounded reclaim：无法区分状态机问题、生命周期问题和 reclaim correctness/性能问题，扩大 P0 风险。
- 每个 request/worker 各自维护 sampler：会产生重复 procfs/cgroup 读取、状态分叉和 owner 不清晰的问题。
- 每 token 或每层采样：文件读取和解析进入热路径，可能直接污染 TPOT/吞吐，且 probe 已表明读取代价需要单独预算。
- 由 PSI 单独触发 reclaim：PSI 只作为持续主压力的确认/升级信号，不能替代 RSS/cgroup 主 source 与 ownership gate。

**Consequences and limits**

- 收益：把输入正确性、状态行为、运行时开销和 reclaim 副作用拆成可独立验收的门禁；单 owner 避免当前非线程安全 counters/state 被并发推进。
- 代价：在只读阶段不会释放内存；需要维护 snapshot 发布与采样节流，压力变化的检测延迟受采样周期约束。
- 当前实现状态：决策 1 和默认关闭已由 sampler-only 源码实现；决策 2–4 是下一阶段 server 集成约束，尚未实现和验证。
- 失效条件：若后续 server 架构要求多 owner 或异步 sampler，必须先定义线程安全、clock、snapshot 一致性和重复采样去重契约，再以新决策 supersede 本条。

**Evidence**

- `src/llama-kv-pressure.h`：明确 sampler-only、默认关闭、无 reclaim，并定义 NORMAL/PRESSURE/CRITICAL/RECOVERY 与 telemetry。
- `src/llama-kv-pressure.cpp`：直接文件读取、source fail-closed、stale、滞回、cooldown、PSI upgrade 和 source/basis 切换计数隔离。
- `tests/test-kv-pressure-sampler.cpp`：864 assertions 的 fixture/synthetic 覆盖；提交摘要记录 ASan/UBSan 与 sampler-only strict warning build 通过。
- 当前源码搜索仅发现 `src/CMakeLists.txt` 和 sampler 单测引用该组件；server/context/decode/reclaim 无调用点。
