# KV Pressure Scheduler 目标契约

> 状态：**目标契约已冻结，当前源码尚未完全实现**
>
> 范围：server pressure scheduler 到 paged KV lifecycle core 的职责边界、动作、门禁和结构化结果。
>
> 非范围：动态阈值与自适应参数、异步 worker、权重–KV 融合、block 内部三轴状态的重复定义。

本文定义 pressure scheduler 如何请求生命周期动作。Block/cell 的内容、驻留、事务、commit/rollback 和 quarantine 语义以 [`kv_block_lifecycle_contract.md`](kv_block_lifecycle_contract.md) 为唯一 authority。文中的“必须”描述目标契约，不表示当前工作树已经满足。

## 1. 模块边界

### 1.1 Server scheduler

Server **只做策略**，负责：

- 消费已发布的 pressure sample 和 request lifecycle 快照；
- 维护 pressure episode、决策频率、静态预算和动作优先级；
- 在 `NOOP / EVALUATE / RELEASE / OFFLOAD / PREFETCH` 中完成一次仲裁；
- 选择逻辑 request/sequence 集合，而不是直接选择最终 physical block；
- 生成 `decision_id`，提交统一 action request，并发布 decision 与 server action observation；core action result 由 lifecycle core 发布；
- 将 core 的失败、partial、abort 和 fail-stop 原样传播，不把失败降级成成功或普通 skip。

Server 不得：

- 修改 block/cell state、committed metadata、backing metadata、used/free-list 或 write transaction；
- 根据 private block enum 绕过 core 门禁；
- 把 `madvise`、backing I/O、mincore/RSS 观测本身解释为状态转换成功；
- 在 active-required prefetch 未闭合时先执行 reclaim。

### 1.2 Lifecycle core

Lifecycle core **独占状态转换**，负责：

- logical cell/sequence 到 physical block 的解析和候选枚举；
- ownership、shared owner、reservation、transaction 和 quarantine 检查；
- destructive mutation 前的最终 ownership recheck；
- backing read/write、staging、validation、`madvise` 和失败收敛；
- block state、backing authority、used/free-list、overlay、commit/rollback 和 fail-stop 修改；
- 分配 `transaction_id`，维护 physical block generation，并发布独立 block events 与 core action result；
- 对 server 已做过的门禁再次执行 correctness 检查。Server 门禁是策略优化，core 门禁才是正确性 authority。

### 1.3 Runner 与 parser

Runner 只记录 request、event、stdout/stderr、系统观测和运行身份。Parser 是最终 protocol verdict authority。二者不得拥有调度策略或 core 状态转换权限。统一证据要求见 [`kv_lifecycle_evidence_protocol.md`](kv_lifecycle_evidence_protocol.md)。

## 2. Scheduler 输入快照

每次 decision 必须基于一份不可变输入快照；同一 `decision_id` 不得混用不同采样时刻的数据。Decision 必须先声明触发域：

```text
trigger = PRESSURE | ACTIVE_ACCESS | RESUME
```

- `PRESSURE` 用于压力驱动的 `RELEASE/OFFLOAD` 或 observation-only `EVALUATE`，必须引用有效且非 stale 的 pressure sample；
- `ACTIVE_ACCESS` 与 `RESUME` 用于 correctness-required `PREFETCH`，由 request lifecycle 触发，不依赖 pressure sample；此时 `episode_id/sample_id/pressure_snapshot` 必须显式为 `null`；
- 不得为了满足 schema 给 correctness prefetch 伪造 pressure sample。

| 类别 | 必需字段 | 语义 |
|---|---|---|
| 身份 | `server_instance_id`, `cache_instance_id`, `cache_epoch` | 隔离进程重启、cache reset 和不同 memory 实例 |
| Trigger | trigger、request/graph operation identity | 区分压力策略与 correctness-required access/resume |
| Pressure | `episode_id`, `sample_id`, state, source, valid, stale；仅 `PRESSURE` 必需 | sample 无效或 stale 时禁止 destructive action；不阻塞 correctness prefetch |
| 原始采样 | RSS/cgroup/PSI/KV resident 值及 validity；仅 pressure path 适用 | 保留事实，不由 scheduler marker 重写 |
| Request lifecycle | active、idle、resume-pending 的 request/sequence 集合 | 只影响策略和合法访问集合，不写入 block 状态 |
| Core capability | 每个 action 的 supported/reason、memory kind、backing 和 fail-stop 状态 | unsupported 必须显式表达，不能与“执行后无候选”混淆 |
| Transaction health | write transaction 是否 `CLOSED`、是否存在 reservation/quarantine/error latch | 开放事务或 fail-stop 时 fail-closed |
| Policy state | 上一 decision/result、cooldown/backoff 是否到期 | 只决定 pressure action 是否请求，不得延迟 correctness prefetch |
| 静态预算 | `target_bytes`, `max_candidates`/`max_blocks`, logical target set | 本契约不定义动态阈值算法 |

Server 不得把 physical block index、free-list 成员资格、backing offset 或 private block enum 作为可自行修改的策略输入。Core 可以返回聚合 candidate/capability summary，但最终候选必须在 transaction 内重新验证。

## 3. 动作模型

| 动作 | 语义 | 是否允许状态转换 |
|---|---|---|
| `NOOP` | 已完成仲裁，但当前不应调用 lifecycle core | 否 |
| `EVALUATE` | 对 `evaluated_action` 执行只读候选评估 | 否 |
| `RELEASE` | 不可逆丢弃无 owner、无恢复价值的 dead/empty resident block | 是 |
| `OFFLOAD` | 为未来可能恢复的 idle-owned `VALID` 内容建立完整 backing authority | 是 |
| `PREFETCH` | 把 active/resume-required 的完整 backing 恢复为 resident authority | 是 |

规则：

- `EVALUATE` 仅允许 `PRESSURE` trigger，必须携带 `evaluated_action=RELEASE|OFFLOAD|PREFETCH` 并复用对应候选门禁；它不得执行 I/O、`madvise`、state/free-list/backing publication 或 generation 修改。`ACTIVE_ACCESS/RESUME` 不得以 `EVALUATE(PREFETCH)` 代替 correctness prefetch。
- `NOOP` 仍必须产生 `decision_id` 和稳定 reason code，但不得产生 core `transaction_id`。
- clear/reset 是显式管理操作，不属于 pressure scheduler 动作，也不得通过 `RELEASE` 解除 quarantine。

## 4. 固定动作优先级

当多个动作同时可选时，固定优先级为：

1. **Fail-stop/error propagation**：存在 active restore failure、context-invalid、quarantine 泄漏风险或不可判定状态时，禁止普通 pressure action；
2. **`PREFETCH`**：满足 active 或 resume-pending 正确性需求；成功前不得进入 graph compute；
3. **`RELEASE`**：处理 unowned、无恢复价值的 dead/empty block，不产生 backing I/O；
4. **`OFFLOAD`**：处理仍有恢复价值的 idle-owned 内容，需要 backing I/O；
5. **`NOOP`**：无合法候选、sample stale、capability 不满足或频率门禁未到。

`EVALUATE` 是 observation mode，不参与 state-changing 动作的优先级竞争。`PREFETCH` 的 `ACTIVE_ACCESS/RESUME` correctness trigger 独立于 pressure sampling、pressure state 与 pressure cooldown：只要访问集合要求 resident authority，就必须先闭合 prefetch 或传播失败。Scheduler 每次 decision 最多提交一个 state-changing core transaction。

## 5. Server 策略门禁

### 5.1 Pressure-triggered action

`PRESSURE` 触发的 `RELEASE/OFFLOAD/EVALUATE` 依次经过：

1. action 配置有效且默认关闭路径未被意外启用；
2. pressure sample `valid=true` 且 `stale=false`；
3. 目标 cache instance/epoch 存在且与 capability snapshot 一致；
4. core 显式支持目标 action；
5. 无 fail-stop 或未处理的 active-required restore failure；
6. request lifecycle 允许目标动作；
7. correctness-required `PREFETCH` 不存在或已闭合；
8. 当前 pressure state 允许该动作；
9. cooldown/backoff 到期；
10. 静态预算非零且格式有效。

### 5.2 Correctness-triggered prefetch

`ACTIVE_ACCESS/RESUME` 触发的 `PREFETCH` 不经过 pressure sample、pressure state 或 pressure cooldown 门禁。它必须验证：目标 cache instance/epoch、request/sequence 身份、`VALID + OFFLOADED` authority、完整 backing metadata、无冲突 transaction/reservation、非 quarantine，以及 core capability。失败必须在 graph compute 前传播，不得降级为 pressure `NOOP` 后继续访问。

门禁失败产生 `NOOP` 或 `SKIPPED` decision result；只有 pressure destructive path 的失败才表述为“不调用 destructive core API”。Stable reason code 必须区分至少：

```text
feature_disabled
invalid_config
sample_invalid
sample_stale
unsupported
memory_instance_changed
transaction_open
fail_stop
active_required
resume_required
cooldown
no_budget
no_candidate
```

## 6. Core correctness 门禁

### 6.1 全局门禁

所有 action request 到达 core 后必须验证：

- `cache_instance_id + cache_epoch` 与当前实例匹配；
- request schema/version/action 合法；
- write transaction 和 reservation 状态允许该动作；
- logical→physical mapping 完整；invalid mapping 必须整体 abort、零状态变化；
- block 不处于 `QUARANTINED`；
- ownership、authority 和 generation 未在 decision 后失效；
- partial failure 有唯一、稳定且 fail-closed 的收敛状态。

### 6.2 `RELEASE`

每个候选在 destructive mutation 前必须重新验证：

- transaction 为 `CLOSED`；
- 无 live/shared owner，无 transaction reservation；
- 只接受 `VALID + RESIDENT` 的 dead 内容或可丢弃的 `EMPTY + RESIDENT`；
- 拒绝 `OFFLOADED/EVICTING/PREFETCHING/QUARANTINED`；
- 不丢弃承担唯一 backing/resident authority 的有效内容；
- discard 成功后才提交 `EMPTY + DISCARDED`、清 backing metadata，并恰好一次加入 free-list。

### 6.3 `OFFLOAD`

每个候选必须满足：

- `VALID + RESIDENT`，且不属于 active/resume-required 访问集合；
- 有 idle owner 或其他明确恢复价值；
- 无开放 write transaction/reservation，非 quarantine；
- backing store 可用；
- 完整 block 写入成功前 resident tensor 始终是权威副本；
- 完整写入后一次性发布 backing metadata 和 `VALID + OFFLOADED`；
- commit 后允许可选 residency-hint `madvise`；它只影响 OS 驻留提示，失败不得回滚已提交 backing authority；
- backing write/publication 失败保持 `VALID + RESIDENT`，不得发布部分 backing authority。

### 6.4 `PREFETCH`

每个候选必须满足：

- `VALID + OFFLOADED`，backing metadata 完整；
- 属于 active-required 或 resume-pending 逻辑集合；
- 非 quarantine，且无冲突 transaction/reservation；
- 完整 read、validation、staging unpack 成功后才提交 `VALID + RESIDENT`；
- 失败保持 `VALID + OFFLOADED`；
- active-required 失败必须在 graph compute 前传播。

## 7. Unified action request

统一 request 至少包含：

```text
schema_version
server_instance_id
cache_instance_id
cache_epoch
trigger                   // PRESSURE | ACTIVE_ACCESS | RESUME
episode_id                // 仅 PRESSURE；否则 null
sample_id                 // 仅 PRESSURE；否则 null
request_operation_id      // ACTIVE_ACCESS/RESUME 必需；PRESSURE 可为 null
decision_id
action
evaluated_action          // 仅 EVALUATE

request_lifecycle:
  active_ids[]
  idle_ids[]
  resume_pending_ids[]

policy_snapshot:
  pressure_state          // 非 PRESSURE trigger 时为 null
  pressure_source         // 非 PRESSURE trigger 时为 null
  sample_valid            // 非 PRESSURE trigger 时为 null
  sample_stale            // 非 PRESSURE trigger 时为 null

budget:
  target_bytes
  max_candidates
  max_blocks

logical_targets:
  seq_ids[]               // 可为空，空表示由 core 在合法全局集合中枚举
```

约束：

- Server 不得在 request 中指定“必须转换”的 physical block 列表；
- request 是意图和预算，不是 state-transition command；
- core 必须为 state-changing invocation 分配 `transaction_id`；
- correctness prefetch 必须引用 `ACTIVE_ACCESS/RESUME` trigger 和 request operation，不得依赖或伪造 pressure sample；
- retry 必须生成新 `decision_id` 和新 `transaction_id`，并通过 `retry_of` 关联，禁止复用旧 transaction identity。

## 8. Core action result 与 server/runner observation

### 8.1 Core action result

Core 只返回由 core authority 直接产生的候选、状态转换、I/O 和错误收敛事实。`NOOP`/server-side skip 没有 core result；只读 `EVALUATE` 返回 `core.evaluation_result`，state-changing action 在 transaction terminal 后返回 `core.action_result`。

```text
schema_version
cache_instance_id
cache_epoch
decision_id
transaction_id            // EVALUATE 为空
action
mode                       // evaluate | execute
outcome                    // completed | partial | aborted | failed
reason_code
retry_of

requested:
  target_bytes
  max_candidates
  max_blocks
  seq_ids[]

capability:
  supported
  reason_code
  snapshot_id

candidates:
  scanned
  eligible
  skipped_owned
  skipped_shared
  skipped_state
  skipped_transaction
  skipped_quarantined
  invalid_mapping

transition_aggregate:
  attempted_blocks
  committed_blocks
  aborted_blocks
  quarantined_blocks
  logical_bytes

physical_operation_aggregate:
  io_read_bytes
  io_write_bytes
  madvise_attempted_bytes
  madvise_accepted_bytes
  operation_failures

correctness:
  ownership_rechecked
  transaction_closed
  active_visibility_preserved
  backing_authority_valid
  fail_stop_latched

block_event_refs[]         // 必需；引用独立 block events
```

Core result **不得**包含 mincore/RSS/cgroup/PSI 等 server/runner observation。Aggregate 只能由独立 block events 重建；`block_results[]` 内嵌副本可作为诊断冗余，但不得成为 parser authority。

### 8.2 Block terminal 与 transaction terminal

每个进入 transition 的 physical block 都必须产生恰好一个 block terminal：`block.commit`、`block.abort` 或 `block.quarantine`。它只结束该 block 的 transition，不结束整笔 transaction。

多 block transaction 必须在全部候选 block 到达 terminal 后再产生恰好一个 transaction terminal：

- `completed`：所有进入 transition 的 block 均 commit；允许因 scan exhaustion 产生业务 shortfall；
- `partial`：至少一个 block commit，且至少一个其他 block abort/quarantine；已 commit block 保持已提交，失败 block 收敛到原 authority 或 quarantine，禁止把聚合 partial 误写成全部 rollback；
- `aborted`：零 block commit，且所有已开始 mutation 均回到原稳定 authority；
- `failed`：零或多个 block 已 commit，但至少一个 block 无法保持普通可用状态并触发 quarantine/fail-stop。

Transaction terminal 与 block terminal 是不同事件和不同闭包层级。Transaction terminal payload 至少包含：

```text
entered_block_transition_ids[]
committed_block_transition_ids[]
aborted_block_transition_ids[]
quarantined_block_transition_ids[]
entered_count
committed_count
aborted_count
quarantined_count
fail_stop_latched
```

Core 在每个 `block.transition.begin` 分配唯一 `block_transition_id`，对应 terminal event 必须复用同一 ID。`committed/aborted/quarantined` 三个 terminal ID 集合必须两两互斥，三者并集严格等于 `entered_block_transition_ids[]`；每个 entered transition 恰好落入一个 terminal 集合，counts 与各集合长度一致；`failed` 必须把触发 fail-stop 的 quarantine refs 明确关联。独立 block events 仍是 parser authority，terminal payload 是闭包索引。Core action result 必须在 transaction terminal 之后发布，并与 terminal payload 和独立 block event 聚合完全一致。

### 8.3 Server action observation 与 runner observation

Server 在 decision 后发布 `server.action_observation`，记录调度侧事实：trigger、decision outcome、是否调用 core、core result reference、queue/cooldown/backoff、错误向 request 的传播以及 server-side timing。它不得重述或覆盖 block transition truth。

Runner 另行记录 `runner.system_observation`：mincore/RSS/cgroup/PSI、strace、HTTP、进程退出和采集 validity。Runner observation 可以引用 decision/transaction/block event，但不得写入 core result，也不得据此制造 core commit。

`outcome` 与 `reason_code` 必须分离；server/runner observation 与 core result 也必须分离，parser 分别验证后再形成 protocol verdict。

## 9. 默认关闭、失败与重试

- Feature 关闭或 memory kind 不支持时，server action observation 必须明确记录 `skipped/unsupported`，不能与“支持但无候选”混淆；不得伪造 core result；
- capability snapshot 只用于 server 策略，core 必须在 transaction 内重新验证；
- invalid mapping、wrong owner、generation mismatch 和 quarantine 命中必须 fail-closed；
- multi-block `partial` 中，已 commit block 可按其独立 block event 生效；abort/quarantine block 不得被复用，transaction terminal 必须列全三类集合；
- offload/prefetch I/O 失败必须保持原 authority；
- compute-started failure 进入 quarantine/fail-stop，只能由显式 clear/reset 恢复；
- retry 不得继承未关闭 owner、reservation、overlay 或 transaction ID。

## 10. 目标契约—当前实现—P0/P1 差距

| 目标契约 | 当前实现 | 状态 |
|---|---|---|
| Server 只做策略，core 执行 release transition | Bounded release 已从 server 经 `llama_memory_i` 调用 KV core | **已实现但证据不足** |
| OFFLOAD/PREFETCH 也由统一 scheduler 仲裁 | Swap-out 候选策略仍在 core window scan；prefetch 没有对称的 server action path | **尚未实现，P1** |
| 固定优先级保证 active-required prefetch 先于 reclaim | 当前 server 只仲裁 dry-run/bounded release | **尚未实现，P1；接入前必须关闭** |
| 五种 action 共享 request；core result 与 server/runner observation 分层 | Release 有专用 result；offload 返回 `void`，prefetch 返回整数；marker 混合 observation | **尚未实现，P0 证据/错误传播差距** |
| Core 对 quarantine fail-closed | 当前 bounded/legacy release gate 未完整排除 `INVALID` | **尚未实现，P0 correctness** |
| Destructive mutation 前逐候选 ownership recheck | 当前 bounded release 每次调用开始收集一次 ownership | **尚未实现，P0** |
| Server 不指定最终 physical block | 当前 bounded release 由 core 扫描 physical candidates | **已实现但证据不足** |
| Invalid mapping 整体 abort、零状态变化 | Core 已有部分 ownership-abort/result 基础，但统一动作契约未覆盖 | **部分实现，P0 测试缺口** |
| Unified capability 区分 unsupported 与 no-candidate | 仅 bounded release 有细分 capability；其他 memory 默认 no-op 可能混淆语义 | **尚未实现，P1** |
| Outcome/reason、block/transaction terminal、core result/observation 分层 | 当前 marker 聚合 policy、transition、physical observation，且无 transaction/physical-block-generation ID | **尚未实现，P0 证据差距** |
| 三轴状态与 overlay/commit authority | 仍使用单一 block enum 和 block-level `PENDING_WRITE` | **尚未实现，P0；由生命周期契约跟踪** |

当前实现入口主要位于：

- [`tools/server/server-context.cpp`](../tools/server/server-context.cpp)：pressure sample、policy gate 和 bounded release 调用；
- [`tools/server/server-kv-pressure.h`](../tools/server/server-kv-pressure.h) 与 [`server-kv-pressure.cpp`](../tools/server/server-kv-pressure.cpp)：runtime 和 marker；
- [`src/llama-memory.h`](../src/llama-memory.h)：当前 release/prefetch memory interface；
- [`src/llama-kv-cache.cpp`](../src/llama-kv-cache.cpp)：release、swap、prefetch 和 write transaction transition。

## 11. 完成判定

只有同时满足以下条件，才能将本文件从“目标契约已冻结”提升为“当前实现已满足”：

1. 五种 action、统一 request、core evaluation/action result 与 server/runner observation schema 已实现，默认关闭和 unsupported 路径可区分；
2. Server 只提交策略意图和逻辑对象，release/offload/prefetch 的最终候选与状态转换均由 core 独占；
3. 固定优先级和 active/resume-required prefetch 门禁有真实 server 路径证据；
4. Release 对 ownership、transaction、offloaded 和 quarantine 的逐候选 destructive recheck 闭合；
5. Offload/prefetch 的 authority publication、I/O failure 和错误传播满足生命周期契约；
6. 独立 block event、transaction terminal、core result 与 server/runner observation 严格分层，并关联统一证据 ID；
7. [`kv_block_lifecycle_contract.md`](kv_block_lifecycle_contract.md) 的相关 P0 状态/transaction/quarantine 条件已实现；
8. 相关单元、短集成、implement/review gate 通过，且至少一个真实短模型路径覆盖 prefetch、release→reuse→commit/rollback 和 offload→restore；
9. clean-HEAD 正式协议产物通过 parser 后，才能形成可归档结论。

在此之前，正确表述是：**scheduler→lifecycle core 目标边界已冻结；当前实现部分具备，但仍有 P0/P1 语义和证据差距。**
