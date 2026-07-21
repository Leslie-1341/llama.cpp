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

- Date: 2026-07-18（决策入账）；2026-07-19（server 集成与验证协议提交；真实模型 VALID artifact 通过）
- Status: accepted；server 集成（`6f59f66b6`）、验证协议（`297eed939`）、集成修复（`31e81656d`、`ad92e603f`、`726d977b2`）与真实模型 VALID artifact 全部就绪
- Evidence commit/worktree: 当前 HEAD `726d977b26bba375edd8e79c1c04a46cece942e4`（clean）；VALID artifact `/root/oscomp/kv_logs/server_kv_pressure_stage3a_1c_20260719T134156Z_726d977b26bb`；sampler-only `befd7a894`；输入 probe `3b2502ca6`
- Supersedes: D-0008 中”尚无真实压力采样”的阶段状态；不改变 D-0008 的 release correctness 契约
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
- 当前实现状态：决策全部五项已由源码实现——1（sampler 与 reclaim 解耦）在 `befd7a894`；2–4（只读 server 集成、单 owner、限频）在 `6f59f66b6`；5（默认关闭）贯穿全部提交。验证协议（`297eed939`）已提交 runner/parser/合成负例。**真实模型 VALID artifact 已通过**（HEAD `726d977b2`，`31e81656d`/`ad92e603f`/`726d977b2` 三笔修复后 10/10 cases PASS）；性能结论为 EXPLORATORY_ONLY。
- 失效条件：若后续 server 架构要求多 owner 或异步 sampler，必须先定义线程安全、clock、snapshot 一致性和重复采样去重契约，再以新决策 supersede 本条。

**Evidence**

- `src/llama-kv-pressure.h`：明确 sampler-only、默认关闭、无 reclaim，并定义 NORMAL/PRESSURE/CRITICAL/RECOVERY 与 telemetry。
- `src/llama-kv-pressure.cpp`：直接文件读取、source fail-closed、stale、滞回、cooldown、PSI upgrade 和 source/basis 切换计数隔离。
- `tests/test-kv-pressure-sampler.cpp`：864 assertions 的 fixture/synthetic 覆盖；提交摘要记录 ASan/UBSan 与 sampler-only strict warning build 通过。
- `tools/server/server-kv-pressure.h`：`server_kv_pressure_runtime` 单 owner runtime——sample_due 限频、record_sample 事件发布、enable/disable lifecycle、无 thread/mutex。
- `tools/server/server-kv-pressure.cpp`：cadence config 解析（最小 100ms 采样/1s 日志）、deadline 推进、should_log() trigger 判定、结构化 marker 格式。
- `tools/server/server-context.cpp`：`init_kv_pressure_sampler()` master switch + fail-closed init、`maybe_sample_kv_pressure()` telemetry-only 调用、`handle_sleeping_state()` sleep/resume 重建、`#if defined(__linux__)` 条件编译。
- `tests/test-server-kv-pressure.cpp`：9 C++ 集成测试通过。
- `tests/test-server-kv-pressure-static.py`：6 静态集成检查通过（single owner、no reclaim calls、no thread/lock、marker fields、master switch preflight、sleep/resume reset）。
- `scripts/run-server-kv-pressure-stage3a-1c.py`、`scripts/parse-server-kv-pressure-stage3a-1c.py`、`tests/test-server-kv-pressure-stage3a-1c-parser.py`：9 合成负例通过、py_compile 通过。
- **Stage 3A-1C VALID artifact**: `/root/oscomp/kv_logs/server_kv_pressure_stage3a_1c_20260719T134156Z_726d977b26bb` — parser exit 0、artifact status VALID、10/10 cases PASS。Strace ON 归因 6 条 sampler procfs/cgroup 路径、OFF 零路径；lifecycle post-resume sample_count 独立重置；idle 1.5s 窗口零 periodic marker。
- 此前失败 artifacts 保留为 INVALID 诊断证据：`…073412Z_4ac1919ec2ed`（strace ON 无法观测 procfs 路径）、`…082058Z_31e81656d1e6`（pre-request marker 缺失）、`…125819Z_ad92e603f7b8`（ON strace lacks required sampler procfs reads；strace 在首次采样后 attach，60000ms 采样周期导致 completion 结束前无后续采样，trace 文件全空；与 lifecycle post-resume marker 无关）；均 runner 10/10 cases 完成但 parser 拒绝。

## D-0010 — Bounded release 作为 per-call budget 控制原语，先入 core 再连 pressure policy

- Date: 2026-07-20
- Status: accepted
- Evidence commit/worktree: `949fbd0c850b7413302907df20cc4022f623d8db`（clean）；CTest #29 `test-kv-paged-release-bounded` 注册并 build 通过
- Supersedes: D-0009 中"bounded reclaim 尚未实现"的阶段描述——本决策提交 core 原语，server 接入留待下道门禁
- Superseded by: none

**Context**

Stage 3A-0 的 `paged_release_blocks()` 是全量扫描、无 budget 控制——每次调用遍历全部 physical block 并释放所有符合条件的 dead/unused block。真实 server scheduler 需要以有限频率、有限量调用 release，避免单次调用阻塞 scheduler loop 太久或过量释放导致后续 request 无可用 block。此外，原 unbounded release 的 state gate 未检查 PENDING_WRITE——事务中间态 block 存在被误 madvise 的风险。

**Decision**

1. **新增 `paged_release_blocks_bounded(target_bytes, max_scan_blocks)` 核心原语**，提供 per-call budget（释放字节上限 + 扫描 block 上限）控制。target_bytes=0 立即返回零副作用；max_scan_blocks=0 返回 exhausted + 完整 shortfall。
2. **复用 `llama_kv_release_collect_ownership` 门禁**：ownership ABORT 路径与 unbounded release 一致——invalid mapping 立即返回、零 state change。
3. **显式 PENDING_WRITE gate**：在 state check 中显式跳过 PENDING_WRITE（以及 SWAPPED、RELEASED），修复原 unbounded release 的防御缺口。B6 test 验证 PENDING_WRITE gate 在 ownership gate 之后独立触发。
4. **Test-only seam 机制**：三个 mutable 字段（force_ownership_abort、madvise_fail_block、block_state_override）直接嵌入 `llama_kv_cache` 对象，默认值保证生产路径零行为差异。所有 seam 在每次调用退出时自动复位（四条退出路径全覆盖），严格 single-shot。两个只读 accessor 用于 post-condition 验证。
5. **Result struct 语义**：`llama_kv_bounded_release_result` 提供 released_bytes、shortfall_bytes、overshoot_bytes（单 block 粒度）、blocks_scanned、skipped_owned、skipped_state、madvise_failures、block_scan_exhausted、ownership_aborted——caller 可据此判断是否需要再次调用（shortfall > 0 且 scan 未穷尽）。
6. **先入 core，不连 pressure**：bounded release 当前无 server scheduler 调用点，无 pressure-driven 触发，无 dry-run 模式。server pressure policy 接入作为独立下道门禁。

**Alternatives rejected**

- 直接修改 `paged_release_blocks()` 签名增加 budget 参数：破坏已有 R0–R5/N0–N2 correctness 验证的调用接口和 artifact 格式。保留原版、新增 bounded variant 降低回退风险。
- 在 server 层封装 budget 逻辑：需要访问 `paged_block_states`、`paged_madvise_block`、ownership bitmap 等 core-private 状态，会破坏 D-0003 的 core/策略分离边界。
- 将 test seam 放在独立 test helper 中：需要 friend class 或 `#ifdef TEST` 条件编译，增加构建复杂度且难以覆盖 ABORT/state-override/madvise-failure 的真实调用路径。
- 让 bounded release 跳过 PENDING_WRITE 检查（与原版一致）：PENDING_WRITE 是事务中间态，madvise 会留下不可恢复的中间 block；防御性检查代价极低（一次枚举比较），收益明确。

**Consequences and limits**

- 收益：per-call budget 控制使上层 scheduler 可以限制单次 release 时延和回收量；PENDING_WRITE gate 修复原版防御缺口；test seam 提供可审计的故障注入路径；result struct 支持 caller 决策循环。
- 代价：`llama_kv_cache` 对象增加三个 mutable 字段和两个 accessor——对象布局/size 改变；test seam 是公开 mutable 字段，虽然命名约定为 `test_*`，但无编译期访问控制。原 unbounded release 的 PENDING_WRITE 缺陷未在本提交修复（只在新原语中防御）。
- 当前验证范围：build 通过、CTest 注册、`git diff --check` 通过、Part A ownership fault fixture（4 组）无需模型即可运行。Part B（B1–B11）需模型文件，当前环境因缺少 `LLAMACPP_TEST_MODELFILE` 而 CTest SKIP。
- 失效边界：与 unbounded release 共享 `paged_madvise_block`、ownership collection、RELEASED state transition 和 swap 互斥 gate——这些公共路径的变更会同时影响两个 release 原语。Test seam 为 mutable public 字段，未来字段语义变更可能与测试预期不一致。

**Evidence**

- `src/llama-kv-cache.h:450-497`：`llama_kv_bounded_release_result` struct、`paged_release_blocks_bounded()` 声明、六个 test-only 成员。
- `src/llama-kv-cache.cpp:6273-6440`：`paged_release_blocks_bounded()` 实现（~170 行），含四条退出路径的 seam 自动复位。
- `tests/test-kv-paged-release-bounded.cpp`：533 行，Part A 4 组 ownership fault fixture + Part B B1–B11（含 B5 ABORT snapshot invariance、B6 PENDING_WRITE gate、B7 madvise failure、B8 dual-context logits match）。
- `tests/CMakeLists.txt`：注册 test #29 `test-kv-paged-release-bounded`。
- Build artifact: `build/bin/test-kv-paged-release-bounded` 存在，CTest 可在有模型时运行 Part B。
- `git diff --check`：通过（无 whitespace 错误）。

## D-0011 — Dry-run bounded release 先于 destructive release 接入 server scheduler，通过只读控制链路验证调度时序与压力状态一致性

- Date: 2026-07-20
- Status: accepted
- Evidence commit/worktree: `02f8cd5ed33a13ffac3cce444d6d3ea7eb987650`（clean）；VALID artifact `/root/oscomp/kv_logs/kv_dry_run_stage3a_2b_20260720T161209Z_02f8cd5ed3`
- Supersedes: D-0010 中"bounded release 先入 core，不连 pressure"的阶段描述——本决策提交 server scheduler 的 dry-run 连接，destructive 接入留待 Stage 3A-2C
- Superseded by: none

**Context**

D-0010 提交的 `paged_release_blocks_bounded()` 是 per-call budget 控制的 destructive 原语，但无 server scheduler 调用点。若在首次接入 server 时直接执行 destructive `MADV_DONTNEED`，调度时序错误、ownership ABORT、cooldown 不足导致的过度回收、或 NORMAL 状态下误触发释放等问题将无法与 dry-run 的只读预测解耦归因。此外，dry-run 与 destructive release 的解耦本身就是一个需要独立验证的设计假设。

**Decision**

1. **先接入 dry-run，不接入 destructive release。** 在 `maybe_sample_kv_pressure()` 的 telemetry 采样后新增 Phase B：调用 `paged_release_blocks_bounded_dry_run()`——与 destructive `paged_release_blocks_bounded()` 共享 ownership collection + state gate 逻辑，但**零 KV state mutation、零 MADV_DONTNEED、零 backing metadata 清除**。
2. **Dry-run 与 destructive release 完全解耦**：
   - Dry-run scanner 为 `const` 方法，编译器强制零 mutation。
   - Dry-run 不检查 `LLAMA_KV_PAGED_RELEASE`——仅需 paged KV + valid layout + !swap。即使 destructive release 因 `LLAMA_KV_PAGED_RELEASE != 1` 而 disabled，dry-run 仍可独立评估。
   - Dry-run 不接入 test-only seam（`force_ownership_abort`、`madvise_fail_block`、`block_state_override`）——这些 seam 仅服务于 destructive 路径的正确性测试。
3. **显式 `should_evaluate` 六道门控**（无隐式 fallthrough、无 sentinel 字符串）：
   - Gate 1: master switch enabled + target_bytes > 0
   - Gate 2: `llama_memory_i` 存在
   - Gate 3: `paged_release_status()` 精确区分 hard skip（swap_enabled/not_paged/layout_unsupported）与 observational disabled
   - Gate 4: `telemetry.stale` 拒绝
   - Gate 5: pressure state ∈ {PRESSURE, CRITICAL}——NORMAL/RECOVERY 不触发评估、不输出 marker
   - Gate 6: `dry_run_due()` cooldown/backoff 时间门控
   - 任意条件不满足即不调用 scanner、不输出 marker。
4. **Cooldown/backoff 策略**：
   - State-entry 语义：进入 PRESSURE/CRITICAL 时立即评估一次（重置 cooldown timer）。
   - CRITICAL entry：绕过 cooldown 进行首次评估；后续 sustained CRITICAL 评估仍受 cooldown 限制。
   - 同一 state episode 内：`elapsed >= cooldown_ms` 才允许再次评估。
   - 评估后 shortfall + scan exhausted → 延长到 backoff_ms（避免在无法满足 target 时频繁无效扫描）；否则回 base cooldown。
5. **`llama_kv_release_status` 枚举**：替代旧版单一 boolean `can_enable` 过载，精确区分 release 可用性原因（`available` / `not_paged` / `layout_unsupported` / `swap_enabled` / `disabled`）。server policy 据此区分 hard skip reason 与 observational disabled。
6. **`llama_kv_bounded_release_result` 提升为全局类型**：从 `llama_kv_cache` 内部 struct 移至 `llama-kv-cache-release.h`，供 `llama-memory.h` virtual 接口引用（避免 core/server header 循环依赖）。
7. **验收仅验证 forced-pressure 控制链路、只读性和协议，不代表真实阈值或性能收益。**

**Alternatives rejected**

- 首次接入即同时执行 dry-run + destructive release：无法独立归因调度时序问题、ownership 问题和 madvise syscall 副作用。dry-run 只读路径是自然的中间门禁。
- 在 sampler 初始化成功前允许 dry-run 触发：dry-run 需要 sampler 提供的 pressure state 进行触发判定。允许独立运行需额外 state source abstraction——当前阶段不需要此复杂度。
- 使用单个 boolean `can_enable` 表达所有 release 状态：无法区分"swap 互斥"（hard skip）与"release 未启用"（observational），导致 server policy 过度跳过或错误触发。
- 让 dry-run scanner 也访问 test-only seam：dry-run 是生产路径——test seam 的语义（ABORT 注入、madvise failure 模拟）与只读语义冲突。保持两套代码路径（destructive + seam vs dry-run + no seam）降低了 seam 泄漏到生产 dry-run 的风险。
- 不做 cooldown/backoff，每次 telemetry sample 都评估 dry-run：ownership collection + page-aligned byte counting 是 O(n_blocks × n_layers) 操作，在 250ms 采样间隔下可能造成不必要的 CPU 开销。cooldown/backoff 将评估频率限制在 seconds 量级。

**Consequences and limits**

- 收益：控制链路只读验证独立于 destructive release 的正确性/性能风险；dry-run marker 提供可审计的 would-release 预测（候选 block 数、字节数、skip reason、cooldown 状态），可在 destructive 接入前确认 policy 决策一致性；`should_evaluate` 门控零隐式行为——所有 skip 原因通过 `skipped_reason` 字段可见。
- 代价：server scheduler 每次 telemetry sample 后额外执行门控检查（即使不评估也需通过 Gates 1–5）；ownership collection 在每次 dry-run 评估时执行（与 destructive release 对等的 O(n_blocks × n_seqs × cells_per_seq) 开销）；`llama_kv_cache` 对象进一步扩大（新增 4 个 global counter accessor）；`llama_kv_bounded_release_result` 提升为全局类型增加了 header 依赖面。
- 当前验证范围：单请求、单模型（Llama-3-8B Q4_K_M）、forced thresholds（1/2/3 KiB RSS）、ctx 1024、32 token 输出。不覆盖并发请求、长上下文、真实内存压力、不同模型/quantization。
- 失效边界：若后续 sampler 被移除或替换，dry-run 的 pressure state 输入随之消失。若 cooldown/backoff 参数与实际 workload 的 pressure 变化速率不匹配，可能导致过度评估（CPU 浪费）或响应滞后（CRITICAL 持续但 dry-run 被 cooldown 阻塞）。Cooldown 默认值（2s base / 10s backoff / 500ms min）为初始值，未经过真实 workload 调优。
- Dry-run 的 `const` 零 mutation 属性仅由 C++ 类型系统和源码审计保证——编译器阻止对 `this` 的非 mutable 成员写入，但 `paged_block_states` 等核心数组在 `const` 方法内仍可被误用（通过 `const_cast` 或其他路径）。当前实现未使用此类绕过。

**Evidence**

- `src/llama-kv-cache-release.h:6-59`：`llama_kv_release_status` 枚举、`llama_kv_bounded_release_result` struct（全局类型）。
- `src/llama-kv-cache.h:381-383`：`bounded_release_dry_run()` 声明。
- `src/llama-kv-cache.h:470-484`：`paged_release_blocks_bounded_dry_run() const` 声明与零副作用契约注释。
- `src/llama-kv-cache.h:507-519`：4 个 global counter test-only accessor。
- `src/llama-kv-cache.cpp:6444-6620`：`paged_release_blocks_bounded_dry_run()` ~177 行实现——ownership collection、state gate、page-aligned byte counting、budget check、zero mutation。
- `src/llama-kv-cache.cpp:6622-6648`：`paged_release_status()` 实现——快速路径 `available` 与慢速路径逐项检查。
- `src/llama-memory.h:162-176`：`bounded_release_dry_run()` 与 `paged_release_status()` virtual 接口。
- `tools/server/server-kv-pressure.h:44-134`：`server_kv_pressure_dry_run_config`、`server_kv_pressure_dry_run_event` structs 与 env 解析声明。
- `tools/server/server-kv-pressure.h:148-178`：`dry_run_enable()` / `dry_run_disable()` / `dry_run_due()` / `dry_run_record()` / `dry_run_config()` 接口。
- `tools/server/server-kv-pressure.cpp:193-254`：dry-run config env 解析（parse_bool_env、parse_uint64_env、parse_uint32_env 及 cooldown/backoff minimum clamping）。
- `tools/server/server-kv-pressure.cpp:256-296`：`dry_run_due()` 与 `dry_run_record()` 实现——state-entry 语义、cooldown elapsed 比较、backoff 扩展。
- `tools/server/server-kv-pressure.cpp:298-324`：`server_kv_pressure_dry_run_format_marker()`——16 字段结构化 marker。
- `tools/server/server-context.cpp:1098-1129`：`init_kv_pressure_sampler()` 内 dry-run config 解析与 `dry_run_enable()`。
- `tools/server/server-context.cpp:1153-1198`：`maybe_sample_kv_pressure()` 内 Phase B `should_evaluate` 六道门控 + `bounded_release_dry_run()` 调用 + marker 输出。
- `scripts/run-kv-dry-run-stage3a-2b.py`：546 行 OFF/ON controlled A/B runner。
- `scripts/parse-kv-dry-run-stage3a-2b.py`：314 行 fail-closed parser。
- `tests/test-kv-dry-run-stage3a-2b-parser.py`：dry-run parser 合成负例。
- `tests/test-server-kv-pressure.cpp`：dry-run config 解析、cooldown/backoff、state-entry、skip-reason 路径。
- `tests/test-server-kv-pressure-static.py`：dry-run decoupling、marker field schema、config isolation 静态检查。
- **Stage 3A-2B VALID artifact**：`/root/oscomp/kv_logs/kv_dry_run_stage3a_2b_20260720T161209Z_02f8cd5ed3`——parser exit 0、verdict PASS、OFF=0 ON=7 dry_run markers、response byte-identical、zero destructive release、zero MADV_DONTNEED (strace)。

## D-0012 — Diff-aware validation harness 作为 implement/review/review-fix/audit 的强制门禁

- Date: 2026-07-21
- Status: accepted
- Evidence commit/worktree: `fd51455b79af08f629810621578965e772ce0685`（clean）；harness self-test 15/15 E2E PASS；audit gate verdict PASS
- Supersedes: none
- Superseded by: none

**Context**

此前 implement/review/review-fix/audit 任务的"完成"判定依赖人工检查 agent 输出，存在三大问题：(1) 无法机器化判定变更是否引入语法错误、whitespace 问题或构建失败；(2) C/C++ 源文件与 CMake build target 的映射依赖人工知识（例如 `tools/server/server-context.cpp`→`llama-server`），容易遗漏增量构建；(3) parser test、skill validation 和 memory check 的遗漏仅能在后续 code review 中发现，反馈周期长。

**Decision**

1. **新增 `scripts/os-agent/gate-runner`** 作为 implement/review/review-fix/audit 四种模式的统一门禁入口。每个模式定义不同的 check 组合——implement 最全（git-diff-check、build、clang-tidy、shell/python 语法、parser test、skill、memory），audit 仅做只读分类+mapping。
2. **Diff 分析自动驱动 check dispatch**：`diff-analyzer.sh` 自动检测变更文件并按扩展名分类（cpp/c/h/py/sh/cmake/skill/ledger），设置 HAS_* flags——无 C/C++ 变更时自动 SKIP build/clang-tidy。
3. **真实 CMake target mapping**：`target-mapper.sh` 从 `compile_commands.json` 的 `-o CMakeFiles/<target>.dir/` 提取源文件→target，header 通过 UMBRELLA_MAP 解析到 umbrella target，所有 target 经 `cmake --target help` 验证。Mapping 失败 → UNRESOLVED (code=2)，不强阻塞但需人工确认。
4. **唯一结构化 marker**：`OS_AGENT_GATE_RESULT mode=<mode> verdict=<v> code=<c> checks=<n> pass=<p> fail=<f> skip=<s> unresolved=<u> artifact=<path>`。全局唯一输出点——无重复、无歧义。Exit codes: 0=PASS 1=FAIL 2=UNRESOLVED 3=INCOMPLETE 4=NO_CHANGES。
5. **Artifact 写入 repo 外**：`/tmp/os-agent-gate/gate-<mode>-<timestamp>/`，防自污染，包含 `full.log` 与 `summary.txt`。
6. **Harness self-test 回归**：`test-harness.sh` 15 个合成 E2E 测试覆盖全部四种模式、target mapping、marker 唯一性、artifact 防自污染、grammar FAIL 检测——15/15 PASS。
7. **不作为 implicit hook**：gate 仅在显式调用 `bash scripts/os-agent/gate-runner <mode>` 时运行，不自动 hook 到 git commit/push/PR workflow。

**Alternatives rejected**

- 将 gate 逻辑嵌入 `os-agent-task` SKILL.md 的 prose 指令：prose 无法 machine-verify——语法错误、构建失败和 mapping 缺失仍需人工发现。
- 使用 CI workflow（GitHub Actions/Jenkins）：当前竞赛私有仓库缺少 CI infrastructure；本地 gate 提供即时反馈且不依赖外部服务。
- 让 gate 自动 commit/push：违反 CLAUDE.md 规定（git commit/push 由用户完成）；gate 只做验证不做变更。
- 仅提供 check 脚本、不提供统一 gate-runner：分散的 check 脚本难以确保每次使用相同顺序和完整组合；gate-runner 的 `OS_AGENT_GATE_RESULT` marker 提供机器可解析的统一 verdict。
- 让 gate 在缺少 compile_commands.json 时 FAIL：该文件由 CMake `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` 生成，非项目强制依赖；UNRESOLVED 比 FAIL 更合适——标记了需要人工清理，但不阻塞非 C/C++ 任务的 audit。

**Consequences and limits**

- 收益：implement/review/review-fix/audit 任务获得机器可验证的完成标准；`OS_AGENT_GATE_RESULT` marker 提供可审计的通过/失败证据；自动 diff 分类避免遗漏跨语言变更的语法检查；target mapping 消除"该编译哪个 target"的不确定性。
- 代价：每次 implement/review/review-fix/audit 任务需额外运行 gate-runner（通常 <10s 不含 build）；需维护 compile_commands.json（已有构建流程）；harness 自身代码 ~1900 行需作为 infra 维护。
- 当前验证范围：自测 15/15 PASS 使用 synthetic fixture repos；audit gate 在 repo 根实际运行通过（verdict=PASS）；真实 incremental build（E2E-10）使用 synthetic fixture，未验证完整 `llama.cpp` build。
- 失效边界：若 compile_commands.json 缺失或过期，target mapping → UNRESOLVED；若 CMake 构建系统变更 target 命名，UMBRELLA_MAP 需同步更新；clang-tidy 在缺少工具时自动 SKIP（非 FAIL）。Gate 不替代完整 CTest suite 或长时间稳定性测试——它仅验证 diff 范围内的语法、构建和基本正确性。

**Evidence**

- `scripts/os-agent/gate-runner`：107 行主入口，arg 解析、diff 分析、check dispatch、verdict + marker 输出。
- `scripts/os-agent/lib/common.sh`：119 行，exit codes（0–4）、gate_log、gate_record_check、gate_final_verdict（优先级 FAIL>UNRESOLVED>INCOMPLETE>PASS>NO_CHANGES）、gate_emit_summary（唯一 marker 输出点）。
- `scripts/os-agent/lib/diff-analyzer.sh`：187 行，ext-based 分类 + HAS_* flags + deleted 文件追踪。
- `scripts/os-agent/lib/target-mapper.sh`：200 行，compile_commands.json -o flag 提取 + UMBRELLA_MAP + cmake --target help 验证。
- `scripts/os-agent/lib/artifact.sh`：60 行，/tmp/os-agent-gate/ 目录创建与防自污染。
- `scripts/os-agent/gates/define-gates.sh`：126 行，四模式 check 组合。
- `scripts/os-agent/checks/run-checks.sh`：534 行，10 个 check 函数。
- `scripts/os-agent/tests/test-harness.sh`：526 行，15 E2E 合成测试——**15/15 PASS**。
- Audit gate 实际运行：`bash scripts/os-agent/gate-runner audit` → `OS_AGENT_GATE_RESULT mode=audit verdict=PASS code=0 checks=8 pass=5 fail=0 skip=3 unresolved=0`。
