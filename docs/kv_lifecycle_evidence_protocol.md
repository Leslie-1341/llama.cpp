# KV Lifecycle 统一证据协议

> 状态：**目标协议已冻结，当前 telemetry/runner/parser 尚未完全实现**
>
> 范围：pressure decision、lifecycle transaction、block transition、物理操作、系统观测和后续 correctness 的身份、事件顺序与闭包规则。
>
> 非范围：动态调度参数、异步 worker、权重–KV 融合、把 mincore/RSS 当作 block state authority。

本文定义生命周期验收所需的机器可判定证据。Block/cell 状态与合法转换以 [`kv_block_lifecycle_contract.md`](kv_block_lifecycle_contract.md) 为 authority，scheduler 行为以 [`kv_pressure_scheduler_contract.md`](kv_pressure_scheduler_contract.md) 为 authority。本文只定义如何证明一次行为，不重新定义状态语义。

## 1. 协议 authority 与证据层

### 1.1 角色

- **Core event producer**：发布 evaluation/transaction、candidate、独立 block transition、I/O、`madvise`、commit/rollback/quarantine 与 core action result 事实；
- **Server event producer**：发布 pressure sample、scheduler decision、server action observation 和 request correctness 关联；
- **Runner**：记录原始事件、argv/env、binary/model/source 身份、HTTP、退出和系统观测，不写最终 PASS；
- **Parser**：是 protocol verdict authority，对身份、schema、顺序、数量、关联和闭包 fail-closed；
- **Harness/gate**：只证明 diff-aware 短检查完整性，不替代真实 server/model protocol。

### 1.2 证据分层

| 层级 | 证明内容 | 不能证明 |
|---|---|---|
| Transition | Core 提交了哪个 block 的哪次合法状态转换 | OS 是否立即回收页面 |
| Physical operation | backing I/O 或 `madvise` 是否尝试/被接受 | logical transition 必然正确；实际 resident drop 等于范围长度 |
| System observation | mincore/RSS/cgroup/PSI 在采样点的观测变化 | 某个 block 的内容/authority 状态 |
| Reuse/correctness | 后续 reserve、commit/rollback、read/decode 与原 transaction 的关系 | 未记录对象的隐式生命周期 |
| Identity/provenance | 事件属于哪个 source、binary、model、run 和进程实例 | 行为本身正确 |

任何一层都不得冒充另一层。完整 verdict 必须按动作闭合所需的全部层级。

## 2. 身份模型

所有 entity ID 必须由其 authority 生成，且不得被重新分配给另一个实体；同一 entity ID 可以且必须在关联事件中重复引用。逐事件唯一键是 `event_stream_id + event_seq`。

### 2.1 Run/source identity

Artifact 至少记录：

```text
protocol_name
protocol_version
head_sha
worktree_dirty
source_snapshot identity
binary identity
model identity
runner identity
parser identity
argv
environment
case_id
run_id
```

Dirty-tree artifact 只能标记为 diagnostic；clean HEAD、身份完整且 parser PASS 的 artifact 才能标记为 archival candidate。

### 2.2 `server_instance_id`

- Server 每次进程启动生成；
- 进程重启后必须变化；
- 所有 server 事件必须携带；
- 不得用 PID 单独代替，因为 PID 可重用。

### 2.3 `cache_instance_id` 与 `cache_epoch`

- 每个 lifecycle core/cache 实例生成 `cache_instance_id`；cache 销毁重建必须变化；
- cache 创建时 `cache_epoch=0`；
- 本协议冻结 explicit reset 身份策略：保留 `cache_instance_id`，原子递增 `cache_epoch`，并在新 epoch 内把所有 physical block generation counter 初始化为 `0`；
- reset 前的 transaction、block、backing 和 observation 引用不得跨 epoch 继续使用；
- transaction 与 physical block identity 必须同时引用 instance 和 epoch。

### 2.4 `episode_id`

表示一次连续高压力生命周期：

- 从 `NORMAL/RECOVERY` 首次进入 `PRESSURE/CRITICAL` 时创建；
- `PRESSURE ↔ CRITICAL` 不创建新 episode；
- stale sample 不关闭 episode，但禁止 destructive decision；
- 进入 `RECOVERY/NORMAL` 时关闭；
- episode 内可有多个 sample、decision 和 transaction；
- 不能把“成功执行次数”命名为 episode。

### 2.5 `sample_id`

- 每次 pressure sample 唯一；
- 包括 valid/invalid/stale sample；
- 只有 `PRESSURE` trigger 的 decision 必须精确引用一个 sample；
- sample count 可用于统计，但不能替代跨重启唯一 ID。

### 2.6 Trigger identity

每个 scheduler decision 必须声明：

```text
trigger = PRESSURE | ACTIVE_ACCESS | RESUME
```

- `PRESSURE` 必须引用 `episode_id + sample_id`；
- correctness-required `ACTIVE_ACCESS/RESUME` 必须引用 request/graph operation identity，且 `episode_id/sample_id` 显式为 `null`；
- parser 必须拒绝为 correctness prefetch 伪造 pressure sample，或让其等待 pressure sample/cooldown。

### 2.7 `decision_id`

- 每次 scheduler 仲裁唯一；
- `NOOP`、server-side skip、`EVALUATE` 和 state-changing action 均必须有；
- 必须按 trigger 引用 pressure identity 或 request operation identity；
- retry 创建新 decision，并通过 `retry_of_decision_id` 关联。

### 2.8 `transaction_id`

- 由 lifecycle core 在接受 state-changing invocation 时生成；
- `NOOP`、纯 server skip 和只读 `EVALUATE` 不得伪造 state-changing transaction；
- 推荐命名空间：`release/offload/prefetch/write`；
- transaction 必须引用触发它的 `decision_id`；内部 write transaction 若非 pressure decision 触发，可以引用 request/graph operation ID，并令 `decision_id` 为空；
- retry 不得复用旧 ID。

### 2.9 Physical block identity 与 generation

```text
physical_block_id = cache_instance_id : cache_epoch : physical_block
```

规则：

- physical block identity 表示槽位，不把 generation 编入 ID；
- 每个 block event 必须携带 `physical_block_id + generation_before + generation_after`；
- cache 创建和每次递增 epoch 的 reset 后，generation counter 初始化为 `0`；`0` 表示该 epoch 尚未提交 fresh content；
- fresh content 成功 `write.commit` 时 `generation_after = generation_before + 1`；
- release、offload、prefetch、prepare、apply 和 rollback 不递增 generation，前后值相等；release 后 counter 保留最后 committed 值；
- reuse reservation 不提前发布新 generation；
- offload/prefetch 通过相同 before/after 值证明 authority 迁移但内容身份不变；
- parser 必须拒绝同一 epoch 内 generation 清零、倒退、跳号或旧 epoch 引用。

## 3. Event envelope

每条机器可读事件至少包含：

```text
schema_version
event_stream_id
event_seq
event_type
timestamp_monotonic_ns
timestamp_wall_utc          // 仅诊断
run_id
case_id
server_instance_id
cache_instance_id
cache_epoch
episode_id
sample_id
decision_id
transaction_id
block_transition_id
physical_block_id
generation_before
generation_after
parent_event_seq
producer                    // server | core | runner-observer
payload
```

约束：

- `event_stream_id` 标识一个由 artifact event sequencer 合并的权威事件流；每次 recorder/server lifecycle 启动必须生成新值，stream rotation 也必须生成新值并显式链接 predecessor；
- server、core 与 runner-observer 都向同一 sequencer 提交事件；`event_seq` 只在同一 `event_stream_id` 内从 `1` 开始严格连续递增，是排序 authority；不同 producer 不得各自生成互相不可比较的 sequence；
- wall-clock timestamp 不得作为事件顺序判定依据；
- event batch 必须保留每个原始 `event_stream_id + event_seq`，不得只留下聚合后顺序；
- 缺少不适用 ID 时使用显式 `null`，不得省略字段导致 schema 歧义；
- parser 必须拒绝重复、缺号、倒序、跳到未知 parent、跨 stream/instance 错误关联或未声明的 stream rotation。

## 4. 事件类型与强制顺序

所有顺序都以同一 `event_stream_id` 内的 `event_seq` 判定。Trigger identity 先于 decision；只有 `PRESSURE` trigger 存在 `pressure.sample`，`ACTIVE_ACCESS/RESUME` 以 request/graph operation event 作为触发源。

### 4.1 `NOOP`

```text
pressure.sample 或 request.operation     // 按 trigger 二选一
scheduler.decision                       // action=NOOP
server.action_observation                // core_called=false
```

`NOOP` 不得产生 core evaluation、transaction、block transition 或 physical operation 事件。

### 4.2 `EVALUATE`

```text
pressure.sample                          // EVALUATE 仅允许 PRESSURE trigger
scheduler.decision                       // action=EVALUATE
core.evaluation.begin
core.candidate.summary
core.block.gate                          // 0..N，只读
core.evaluation.result
server.action_observation
```

`EVALUATE` 仅允许 `PRESSURE` trigger，不分配 state-changing `transaction_id`，不得产生 transition begin/terminal、backing I/O、`madvise`、generation 或 state/free-list publication。`ACTIVE_ACCESS/RESUME` correctness trigger 只能产生实际 `PREFETCH` decision 或 fail-closed 错误传播，不得以 `EVALUATE` 代替恢复并继续 graph。

### 4.3 State-changing action

```text
pressure.sample 或 request.operation     // PREFETCH correctness path 不需要 pressure.sample
scheduler.decision
core.transaction.begin
core.candidate.summary
core.block.gate                          // 0..N
core.block.transition.begin              // 每个进入 transition 的 block
core.physical_operation                  // 0..N，commit 前置 I/O/syscall
core.block.transition.commit
  或 core.block.transition.abort
  或 core.block.transition.quarantine
core.physical_operation                  // 仅 OFFLOAD commit 后可选 residency_hint madvise
...                                      // 其余 block 重复上述序列
core.transaction.completed
  或 core.transaction.partial
  或 core.transaction.aborted
  或 core.transaction.failed
core.action_result
server.action_observation
```

Block transition terminal 与 transaction terminal 必须分离：

- core 在每个 `transition.begin` 分配唯一 `block_transition_id`，对应 terminal event 必须复用同一 ID；每个 begin 恰好对应一个 block terminal；
- transaction 只有在全部已进入 transition 的 block 到达 terminal 后才能结束；
- `partial` 必须至少包含一个 committed block 和一个 abort/quarantine block；已 commit block 保持提交，失败 block 必须收敛回原 authority 或 quarantine；
- `aborted` 表示零 block commit 且所有已开始 mutation 均恢复；
- `failed` 表示存在无法保持普通可用状态的 block，可伴随已 commit block，但必须设置相应 quarantine/fail-stop；
- `core.action_result` 必须在 transaction terminal 后发布，且由独立 block events 重建；
- transaction terminal payload 必须列出 `entered/committed/aborted/quarantined_block_transition_ids[]` 及对应 counts；`committed/aborted/quarantined` 三个 terminal ID 集合两两互斥、并集严格等于 entered，每个 entered transition 恰好对应一个 terminal，且与携带相同 `block_transition_id` 的独立 block terminal events 完全一致；
- `server.action_observation` 只能引用 core result，不得成为 core transition truth。

### 4.4 Write transaction 与后续 correctness

```text
core.block.reuse_reserved
core.write.prepared
core.write.applied
core.write.compute_started

成功：
  graph success + required sync
  core.write.commit                      // 原子 visibility publication + write transaction 成功终态
  core.block.write_commit                // 1..N，每个 fresh block 的独立证明
失败：
  core.write.rollback
    或 core.write.quarantine

correctness.access_check
correctness.response
```

`write.commit` 只有在 overlay/metadata/state/generation publication、used/free-list 更新和 owner/reservation 清理已于同一原子边界完成后才能发布。它是 write transaction 成功终态和唯一 visibility 发布点，其 payload 必须包含 `expected_write_blocks[]`（每项为 `physical_block_id + generation_before + generation_after`）与 `expected_write_block_count`。该闭包索引只声明应出现的逐 block 证据集合，不替代 block truth。独立 `core.block.write_commit` events 必须紧随其后，作为对已提交逐 block 状态与 generation 的 parser-authoritative 证明；它们不产生新的 visibility。任一缺失都会使 artifact 不完整，但不会把已发生的 commit 解释为 rollback。

### 4.5 Explicit reset

```text
management.reset.request
core.reset.begin                        // epoch_before, epoch_after, expected_physical_blocks

commit path:
  core.reset.commit                     // 原子 epoch/state/free-list/backing publication
  core.block.reset                      // 1..N，逐 block 事后证明
abort path:
  core.reset.abort                      // 零 reset mutation，epoch 保持不变
```

Reset 规则：

- `reset.begin` 前所有 lifecycle/write transaction 和 block transition 必须已有 terminal；否则不得进入 commit path；
- `reset.begin` 只建立 reset staging/plan，不发布新 epoch、block state、free-list 或 backing mutation；payload 必须声明 `cache_instance_id`、`epoch_before`、`epoch_after = epoch_before + 1` 和 `expected_physical_blocks`；
- commit path 必须在一个原子边界内保留 instance、切换到 `epoch_after`，将所有 block generation 初始化为 `0`，清 backing/owner/overlay/reservation/pending/quarantine/error latch，并重建 free-list；随后发布 `core.reset.commit`；
- 每个 post-commit `core.block.reset` 使用新 epoch 的 envelope：`cache_epoch=epoch_after`、`physical_block_id=physical_block_id_after`、`generation_after=0`；payload 额外携带 `physical_block_id_before + generation_before`。这是通用单一 physical identity envelope 的明确跨 epoch 例外；
- `core.block.reset` 数量必须等于 `expected_physical_blocks`，after identities 覆盖新 epoch 全部 physical blocks且无重复；第一个普通新 epoch 事件必须晚于全部 block reset evidence；
- abort path 在任何 commit 前失败时发布 `core.reset.abort`，必须丢弃 staging，保持旧 epoch 的 block/backing/free-list/quarantine/error 状态逐项不变，且不得产生 `core.block.reset`；
- `reset.commit` 后失败或证据丢失不得回退为 abort；artifact 缺少任一 block reset evidence 时判 incomplete/fail-closed；
- cache 销毁重建使用新的 `cache_instance_id` 和独立 `core.cache.created`，不得伪装成 reset。

### 4.6 顺序不变量

- `scheduler.decision` 必须引用已经出现且与 trigger 匹配的 source event；
- state-changing `core.transaction.begin` 必须引用已经出现的 decision；
- candidate/gate/transition/physical events 必须引用开放 evaluation 或 transaction；
- commit 前置 physical operation 必须位于对应 block transition begin 与 terminal 之间；唯一例外是 offload commit 后的 `residency_hint` madvise，它必须标记 `phase=post_commit_hint`，失败不改变已提交 authority；
- core result 必须在 evaluation/transaction terminal 之后，server observation 必须在 decision 之后且引用适用的 core result；
- process/server 退出时不得遗留开放 lifecycle/write transaction 或未终结 block transition；
- reuse event 必须引用使旧内容进入可分配状态的 release transaction 和相同 physical block identity；
- correctness.response 不得早于相关 active-required prefetch 和 write transaction 终态。

## 5. Transition、物理操作与系统观测分离

### 5.1 Transition event

描述 core authority 下的状态事实，至少包含：

```text
block_transition_id
physical_block_id
generation_before
generation_after
from_content_state
from_residency_state
to_content_state
to_residency_state
transaction_state
ownership_rechecked
authority_before
authority_after
free_list_before/free_list_after
backing_publication_changed
outcome
reason_code
```

Transition commit 是 logical truth，不由 runner 从日志文本猜测。

### 5.2 Physical operation event

描述 syscall/I/O 事实：

```text
operation             // madvise | backing_read | backing_write | staging_unpack
phase                 // pre_commit_required | post_commit_hint
physical_block_id
generation_before
generation_after
requested_bytes
completed_or_accepted_bytes
return_code
error_code
address_or_backing_range_identity
```

规则：

- `madvise` 返回成功只证明内核接受建议范围；
- backing write 成功必须覆盖完整 block 后才能支持 offload commit；
- backing read/staging 必须完整验证后才能支持 prefetch commit；
- offload commit 后允许记录 `phase=post_commit_hint` 的 residency-hint `madvise`；它不参与 backing authority publication，失败不得把 committed offload 改写为 abort；
- process-wide strace 只能作为旁证；没有 transaction/block correlation 时不能单独归因。

### 5.3 System observation event

描述 mincore/RSS/cgroup/PSI 观测：

```text
observation_type
scope
before_or_after
value
valid
sample_latency_ns
related_transaction_id
related_core_result_event_seq
producer                    // server-observer | runner-observer
```

规则：

- mincore 表示采样时刻页面 residency，不表示 KV 内容有效性或 authority；
- RSS/cgroup 受其他分配、回收和 refault 影响，只能作为进程/系统级观测；
- `madvise_accepted_bytes` 不必等于 `mincore_before - mincore_after`；
- 只有协议额外证明目标页 before 全部 resident、观测区间无并发 refault 且 after 同范围 nonresident 时，parser 才可要求精确相等；
- 不满足精确前提时，parser 只能验证方向、范围和 validity，不能制造字节级闭包。

### 5.4 Core result 与 observation

- `core.evaluation.result` / `core.action_result` 只聚合 core 直接产生的 candidate、独立 block transition、I/O 与错误收敛事实；
- `server.action_observation` 记录调度调用、reason、cooldown/backoff、传播和 server timing；
- `runner.system_observation` 记录 mincore/RSS/cgroup/PSI、strace、HTTP、退出和采集 validity；
- server/runner observation 不得嵌入 core result，也不得覆盖独立 block event；
- parser 必须先以独立 block event 重建 core aggregate，再独立验证 server/runner observation。

## 6. `RELEASE` 证据闭包

每个 committed block 必须证明：

1. decision 选择 `RELEASE`，sample 有效且非 stale；
2. core transaction 已开始；
3. destructive mutation 前 ownership recheck 成功；
4. block 为合法 `CLOSED`、unowned、非 reserved、非 offloaded、非 quarantined 候选；
5. discard physical operation 被接受；
6. transition commit 为 `EMPTY + DISCARDED`；
7. backing authority 已清除；
8. free-list 成员资格从不可分配变为恰好一次可分配；
9. 每个 block transition terminal 已闭合，transaction terminal 在所有 block terminal 后发布；multi-block partial 的 committed/aborted/quarantined 集合可由独立 block event 重建；
10. core action result 的 committed block/bytes 与独立 block event 聚合一致，shortfall、overshoot、scan exhaustion 和 skipped reason 可由 candidate events 重建。

禁止以以下任一单项作为 release 成功：

- `madvise()` 返回 0；
- mincore/RSS 下降；
- aggregate counter 增加；
- HTTP 请求成功。

## 7. `OFFLOAD` 证据闭包

每个 committed block 必须证明：

1. decision 选择 `OFFLOAD`，逻辑对象 idle-owned 且非 active/resume-required；
2. 初始状态为 `VALID + RESIDENT`，非 reserved/quarantined；
3. transition begin 发布 `EVICTING`，resident tensor 在 commit 前仍是 authority；
4. backing write 覆盖完整 block；
5. offsets/sizes 等 metadata 在完整 write 后一次性发布；
6. transition commit 为 `VALID + OFFLOADED`，backing 成为唯一 authority；
7. publication 后允许可选 `phase=post_commit_hint` residency-hint `madvise`；其失败不回滚 offload commit；
8. backing write/publication failure 路径收敛回 `VALID + RESIDENT`，无部分 metadata publication；
9. 后续 prefetch 能以同一 `physical_block_id` 及相等的 generation before/after 找到该 backing。

## 8. `PREFETCH` 证据闭包

每个 committed block 必须证明：

1. decision 选择 `PREFETCH`，trigger 为 `ACTIVE_ACCESS` 或 `RESUME`，对象为 active-required 或 resume-pending；该 decision 不要求 pressure sample；
2. 初始状态为 `VALID + OFFLOADED`，backing metadata 完整；
3. transition begin 发布 `PREFETCHING`，backing 在 commit 前仍是 authority；
4. 完整 backing read、validation 和 staging unpack 成功；
5. transition commit 为 `VALID + RESIDENT`；
6. graph compute 只能在所有 active-required prefetch commit 后开始；
7. read/validation/unpack 失败保持 `VALID + OFFLOADED`；
8. active-required failure 必须闭合到 request/HTTP 错误，不得继续 decode。

## 9. Release 后 reuse 与 write transaction 闭包

### 9.1 Reservation

`core.block.reuse_reserved` 必须：

- 引用相同 `physical_block_id`、当前 `cache_epoch`、保留的旧 generation counter 和 release transaction；
- 引用新的 write `transaction_id`；
- 证明 block 只在 `EMPTY + DISCARDED` 且 free-list 唯一成员条件下被选择；
- 不提前发布新 generation 或 committed visibility。

### 9.2 Commit

成功路径必须按顺序证明：

```text
write.prepared
write.applied
write.compute_started
graph success + required sync
authority/overlay/state/generation/free-list atomic publication
write.commit
block.write_commit                  // 1..N，独立逐 block 证明
```

`core.write.commit` 是 write transaction 的成功终态和唯一 committed visibility 发布点。它必须在 graph 成功与 required sync 后，并且以下事实全部完成后才能发布：

- 验证 owner/reservation/fresh-write completeness；
- 原子 publication 中每个 fresh physical block 的 generation 已递增，随后发布的独立 `core.block.write_commit` event 满足 `generation_after = generation_before + 1`；
- 原子发布 overlay 和新 `VALID + RESIDENT`；
- 清旧 backing authority；
- 更新 used/free-list；
- 清 owner、overlay、journal 和 reservation，transaction 回到 `CLOSED`；
- 后续 read/decode 使用 `physical_block_id + cache_epoch + generation_after`，并与 baseline/variant correctness 关联。

Parser 必须要求紧随其后的独立 `core.block.write_commit` events 与 `expected_write_blocks[]` 精确集合相等、数量一致、无重复，且每项满足 `generation_after = generation_before + 1`。闭包索引用于发现缺失/额外事件；逐 block 状态事实的 authority 仍是独立 event。

### 9.3 Pre-compute rollback

`PREPARED/APPLIED` 失败必须证明：

- 只撤销当前 owner 的 reservation/overlay；
- 不发布新 generation；
- block 回到 release 后的 `EMPTY + DISCARDED` 或原稳定空状态；
- free-list 恢复且无重复；
- committed metadata/backing authority 与 transaction 前一致；
- owner/journal/pending 全部清理；
- 可立即重试并生成新 transaction ID。

Speculative page discard 失败时不得伪造 rollback success，必须进入 quarantine closure。

## 10. Quarantine 与 fail-stop 闭包

进入 quarantine 的事件必须记录：

- 触发 transaction 和首个失败 operation；
- 所有可能被部分写入或状态未知的 `physical_block_id + generation_before/after`；
- 稳定 residency 收敛结果；
- block 从 free-list 隔离；
- context/cache fail-stop latch；
- transaction owner、overlay 和 reservation 最终清理。

随后必须证明：

- quarantined block 的 read/write/release/offload/prefetch/reuse 均 fail-closed；
- backing 看似完整不能自动恢复；
- 普通 rollback 不能解除 quarantine；
- 只有 explicit clear/reset 事件可以退出；
- reset 保留 `cache_instance_id`、递增 `cache_epoch`、把新 epoch 的 generation counters 初始化为 `0`，清 backing、owner、overlay、pending 和 error latch，并重建无重复 free-list；cache 销毁重建才生成新 instance。

Parser 遇到 quarantine 后仍继续普通 decode、reclaim 或 reuse，必须判 FAIL。

## 11. Correctness closure

一次动作不能只闭合到 transaction result，还必须覆盖适用的后续行为：

| 场景 | 必需证据 |
|---|---|
| Release 无 reuse | active/owned/shared 内容未受影响；结果与 block events 聚合一致 |
| Release→reuse→commit | 新 generation 唯一可见；输出/token/字节与基线一致 |
| Release→reuse→rollback | 无新 generation、无 stale owner/overlay、可立即重试 |
| Offload→prefetch | 同一 generation 字节一致，authority 顺序唯一 |
| Active-required prefetch failure | graph 未开始，错误传播到 request/HTTP |
| Compute-started failure | 受影响 block 全部 quarantine，后续普通访问拒绝 |
| Repeated cycles | generation 单调，free-list/counter/owner/backing 不漂移 |
| Default/unsupported | 无 lifecycle transaction、无目标 counter/state 变化，原输出一致 |

Response identity 只能证明端到端输出结果，不能替代 block lifecycle 事件；block events 也不能替代响应正确性。

## 12. Runner 记录规则

Runner 必须：

- 使用受控 argv/env，并记录继承变量清理；
- 记录 source、binary、model、runner、parser 的路径、大小和 digest；
- 每个 case 使用唯一输出目录，禁止静默覆盖；
- 原样保存事件流、stdout/stderr、HTTP、退出码和 shutdown/PGID 状态；
- 保存 system observation 的采样范围、validity 和时间；
- 对 incomplete、timeout、残留进程或采集失败写明确运行状态；
- 不写最终 protocol PASS；
- dirty tree 必须保存 tracked diff 和 untracked source snapshot。

## 13. Parser fail-closed 规则

Parser 必须对以下情况非零退出：

### 13.1 Identity/schema

- protocol/schema version 不支持；
- source/binary/model/runner/parser 身份缺失或漂移；
- archival artifact 来自 dirty tree；
- 必需字段缺失、多余字段未由版本允许、enum/reason 不一致；
- server 与 parser 对同一 reason code 使用不同词汇。

### 13.2 Event order/correlation

- 同一 `event_stream_id` 内 `event_seq` 重复、缺号、倒序，或未声明 stream rotation/跨 server instance 混排；
- 引用未知 trigger source、decision、transaction、block transition、physical block、generation、core result 或 parent；
- decision 早于其 trigger source；`PRESSURE` decision 缺 sample，或 `ACTIVE_ACCESS/RESUME` decision 伪造/依赖 sample；
- commit 前置 physical operation 在 transition begin 前或 block terminal 后；offload post-commit hint 未标记 `phase=post_commit_hint`；
- `block_transition_id` 重用、begin/terminal ID 不匹配、缺 terminal/有多个 terminal，或 transaction 在 block terminal 前结束；
- `partial` 没有同时包含 committed 与 abort/quarantine block，或遗漏任一 block closure；
- core action result 在 transaction terminal 前，server observation 冒充 core result，或进程退出时仍开放；
- `write.commit` 缺少 expected block closure index，或后续 `core.block.write_commit` 与 expected set/count 不精确一致、重复、缺失、额外或 generation 非 +1；
- physical block reuse 后 generation 未递增、同 epoch 清零/倒退/跳号，或 reset 后仍引用旧 epoch；
- reset 缺少 begin/commit-or-abort terminal，commit 后 block reset evidence 与 expected count/全 physical-block 集合不一致，`epoch_after != epoch_before + 1`，任一新 generation 非 0，reset 与开放 transaction 重叠，普通新 epoch 事件早于全部 block reset evidence，或 abort 路径产生 reset mutation/block evidence；
- core result、aggregate counter 与独立 block event 无法重建一致；内嵌 `block_results[]` 与独立 block event 冲突。

### 13.3 Lifecycle correctness

- stale/invalid sample 触发 pressure destructive action；
- active-required prefetch 因无 pressure sample/cooldown 被跳过，或未完成就开始 graph；
- owned/shared/reserved/offloaded/quarantined block 被 release；
- partial backing I/O 后发布 valid authority，或 offload post-commit residency hint 失败后错误回滚 backing authority；
- rollback 发布新 generation 或遗留 owner/reservation；
- quarantine 后发生普通 read/write/release/offload/prefetch/reuse；
- 默认关闭 case 出现目标 action transaction；
- 目标 case 没有真实执行目标分支。

### 13.4 Physical observation

- 缺少 observation validity 却使用其做 verdict；
- 把 process-wide strace syscall 无关联归因给目标 transaction；
- 未证明精确前提却强制 `madvise bytes == mincore drop`；
- 用 RSS/mincore 变化替代 core transition commit；
- 将来自同一 mutation source 的 result/counter equality 称为独立物理证据。

Parser 不得因当前实现失败而放宽 schema、response identity、安全字段或 baseline。

## 14. 最小 case 矩阵

| Case | 必须证明 |
|---|---|
| OFF | 无 target decision/transaction/state/counter 变化；原输出一致 |
| NOOP | trigger→decision→server observation 顺序完整；零 core event |
| EVALUATE | trigger→decision→evaluation result 顺序完整；候选统计可由独立 gate event 重建；零 transaction/physical operation/transition terminal |
| RELEASE | ownership/state gate、discard、transition、free-list 闭合 |
| RELEASE→REUSE→COMMIT | generation、overlay visibility、输出一致 |
| RELEASE→REUSE→ROLLBACK | 原空状态恢复、无 owner 泄漏、可重试 |
| OFFLOAD | 完整 write 后 publication；failure 保持 resident |
| PREFETCH | ACTIVE_ACCESS/RESUME 无 pressure sample 也可触发；完整 read/unpack 后 resident；failure 保持 offloaded |
| MULTI-BLOCK PARTIAL | 每个 block terminal 完整；transaction partial 在全部 block terminal 后；已 commit 与 abort/quarantine 集合闭合 |
| ACTIVE PREFETCH FAILURE | graph 未开始、错误传播 |
| QUARANTINE | 所有普通动作拒绝；只有 reset 恢复 |
| RESET IDENTITY | instance 保留、epoch +1、generation=0、旧 epoch 引用全部拒绝 |
| REPEATED | 多轮 state/free-list/backing/generation/counter 不漂移 |
| UNSUPPORTED | 明确 capability reason，无目标 transaction |

模型相关 case 必须证明目标 memory kind、数据类型、拓扑和代码分支真实执行。正常退出的 skip 不能作为覆盖证据。

## 15. 目标协议—当前实现—P0/P1 差距

| 目标协议 | 当前实现 | 状态 |
|---|---|---|
| server/cache/episode/sample/decision/transaction/block-generation 全链 ID | 当前主要有 sample count 和按执行次数递增的 `episode`，没有其余关联 ID | **尚未实现，P0 证据差距** |
| 严格单调 `event_seq` | 当前依赖 stderr marker 出现顺序 | **尚未实现，P0** |
| Decision→transaction→block→physical→result 分阶段事件 | 当前 bounded marker 混合 policy、result、counter、mincore 和 capability | **尚未实现，P0** |
| Transition、physical operation、system observation 分层 | Core result/counter 与 `madvise` 范围同源；mincore 是聚合观测 | **部分实现，P0/P1** |
| `madvise` 与 mincore 精确相等仅在前提成立时检查 | 当前 Stage 3A-2C parser 无条件要求精确相等 | **尚未实现，P0 protocol** |
| Release block-level ownership/state/free-list closure | 当前只有 aggregate result/counter/final stats | **尚未实现，P0** |
| Release→reuse→commit/rollback 按 generation 关联 | 当前只有聚合 reuse/write counter 和 response identity | **尚未实现，P0** |
| Offload/prefetch I/O 和 authority publication 事件 | 当前主要通过内部 counters/return value，offload `void`、prefetch 整数 | **尚未实现，P0 证据差距** |
| Quarantine 所有动作 fail-closed 并由 reset 唯一解除 | 当前 `INVALID` 基础 fail-stop 已存在，但 release 门禁不完整 | **尚未实现，P0 correctness** |
| Parser reason/schema 与 producer 唯一一致 | 当前 bounded skip reason 存在 producer/parser vocabulary 不一致风险 | **尚未实现，P0 tooling** |
| Runner 记录 source/binary/model/argv/env/shutdown | 当前 Stage 3A-2C runner 已具备 dirty snapshot、身份和 PGID 记录基础 | **已实现但证据不足** |
| Parser 对缺失、漂移、残留进程 fail-closed | 当前 parser fixture 已覆盖部分 schema、identity、shutdown 负例 | **已实现但仅限现有 Stage 3A-2C schema** |
| 真实完整 lifecycle case | 当前没有统一 release/offload/prefetch/quarantine 协议产物 | **尚未实现，P0 验收差距** |

当前证据入口主要位于：

- [`tools/server/server-context.cpp`](../tools/server/server-context.cpp) 与 [`server-kv-pressure.cpp`](../tools/server/server-kv-pressure.cpp)：pressure/bounded marker；
- [`src/llama-kv-cache.cpp`](../src/llama-kv-cache.cpp)：release、swap、prefetch、write transaction 与 counters；
- [`scripts/run-kv-bounded-release-stage3a-2c.py`](../scripts/run-kv-bounded-release-stage3a-2c.py)：当前 bounded diagnostic runner；
- [`scripts/parse-kv-bounded-release-stage3a-2c.py`](../scripts/parse-kv-bounded-release-stage3a-2c.py)：当前 bounded parser；
- [`tests/test-kv-bounded-release-stage3a-2c-parser.py`](../tests/test-kv-bounded-release-stage3a-2c-parser.py) 与 [`test-runner-bounded-release-stage3a-2c.py`](../tests/test-runner-bounded-release-stage3a-2c.py)：当前 tooling fixture。

## 16. 完成判定

只有同时满足以下条件，才能将本文件从“目标协议已冻结”提升为“当前证据链已满足”：

1. 所有必需 ID、`event_stream_id`、physical block generation before/after 和严格连续 `event_seq` 已由 authority producer 实现；
2. PRESSURE 与 ACTIVE_ACCESS/RESUME trigger、decision、evaluation/transaction、独立 block transition、physical operation、core result、server/runner observation 和 correctness 事件可按 ID/顺序完整关联；
3. Release、offload、prefetch、multi-block partial、reuse→commit/rollback 和 quarantine 各自满足本文闭包；
4. Transition truth、syscall/I/O truth 和 mincore/RSS observation 在 schema 与 parser 中严格分层；
5. Parser 对 identity、schema、顺序、generation、reset epoch、block/transaction terminal、生命周期和物理归因全部 fail-closed；
6. OFF/EVALUATE/UNSUPPORTED 等零 mutation 路径可证明，不能靠无日志推断；
7. 正负 fixture 覆盖单字段缺失、重复/倒序事件、跨 transaction 错配、旧 generation、partial I/O、错误归因和假 PASS；
8. 相关 core 生命周期单元测试和无模型短集成通过；
9. 至少一个 dirty-tree 真实短模型协议完整通过，表述为“已诊断验证，尚不可归档”；
10. 用户形成 clean HEAD 后，以同一 runner/parser 重跑并获得身份完整的 archival PASS，才可形成正式结论。

在此之前，正确表述是：**统一生命周期证据目标协议已冻结；当前 telemetry/runner/parser 只有部分基础，尚不能闭合完整 lifecycle。**
