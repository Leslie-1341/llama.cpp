# KV Block 生命周期目标契约

> 状态：**目标契约已冻结，当前源码尚未完全实现**
>
> 范围：paged KV block/cell 的内容、驻留、事务、错误恢复与测试语义。
>
> 非范围：server 调度策略、证据 parser、权重–KV 融合和性能参数。

本文冻结 KV 生命周期的目标语义，作为后续实现、review 和测试的判定基准。文中的“必须”描述目标契约，不表示当前工作树已经满足。当前实现状态见第 9 节。

## 1. 状态模型

KV 生命周期由三个正交状态轴表达。请求级 `ACTIVE / IDLE / RESUME_PENDING` 属于上层生命周期，不得编码进 block 状态。

### 1.1 内容状态

| 状态 | 语义 |
|---|---|
| `EMPTY` | block 不含任何可被读取的已提交 KV；旧内容没有恢复价值。 |
| `VALID` | block 含完整、已提交且可恢复的 KV 内容。 |
| `QUARANTINED` | block 可能含部分写入、未知或不一致内容；禁止读取、写入、回收、换出和复用。 |

### 1.2 驻留状态

| 状态 | 语义 |
|---|---|
| `RESIDENT` | 当前权威内容位于 KV tensor 的可访问页面中。 |
| `EVICTING` | 正在生成完整 backing 副本；提交前 tensor 内容仍是权威副本。 |
| `OFFLOADED` | backing store 是权威副本；tensor 页面不得被假定为有效驻留，即使页面尚未实际回收。 |
| `PREFETCHING` | 正在从完整 backing 恢复；成功提交前 backing 仍是权威副本。 |
| `DISCARDED` | 无有效 resident 内容，也无可恢复 backing。 |

`OFFLOADED` 表达权威副本的位置，而不是操作系统页表的瞬时 resident bit。

### 1.3 事务状态

| 状态 | 语义 |
|---|---|
| `CLOSED` | 没有开放事务、owner、overlay、reservation 或 journal。 |
| `PREPARED` | owner、slot、reservation、journal 和 overlay 已建立，但 committed state 尚未改变。 |
| `APPLIED` | owner 可通过 overlay 看到本事务 cell metadata；其他观察者仍只看到 committed state。 |
| `COMPUTE_STARTED` | backend 可能已经写入 K/V；失败后禁止普通 rollback。 |

事务状态属于 write transaction，不属于 block。一个 block 可以同时包含已提交 cell 和同一事务预留的 overlay cell。

## 2. 合法组合

| 内容状态 | 合法驻留状态 | 说明 |
|---|---|---|
| `EMPTY` | `RESIDENT`, `DISCARDED` | `RESIDENT` 只表示页面可访问，不赋予其中字节任何读取语义。 |
| `VALID` | `RESIDENT`, `EVICTING`, `OFFLOADED`, `PREFETCHING` | 任一时刻必须存在且只能存在一个权威副本。 |
| `QUARANTINED` | `RESIDENT`, `OFFLOADED`, `DISCARDED` | 只记录保守的稳定驻留结果；不得启动新的 eviction/prefetch。 |

非法组合包括：

- `VALID + DISCARDED`：没有任何权威副本；
- `EMPTY + OFFLOADED/EVICTING/PREFETCHING`：空内容不得拥有或生成 backing；
- `QUARANTINED + EVICTING/PREFETCHING`：失败必须先收敛到稳定驻留状态再发布 quarantine；
- 将请求状态或 `PENDING_WRITE` 作为第四种 block 状态轴。

## 3. Transaction-owned cell overlay

`PENDING_WRITE` **不是 block 状态**。目标实现必须用 transaction-owned、per-cell overlay 表达未提交写入，至少包含：

- transaction owner identity；
- logical/physical cell reservation；
- 新 cell metadata；
- 被覆盖 committed metadata 的 journal；
- 涉及的 physical blocks；
- fresh-write bitmap 或等价的逐 cell 完整性信息。

可见性规则：

1. owner 在 `APPLIED` 和 `COMPUTE_STARTED` 中读取本事务 cell 时使用 overlay；
2. 非 owner 始终只能读取 committed view；
3. “存在某个开放 owner”不足以授权 pending cell 访问，访问者 identity 必须与 overlay owner 精确匹配；
4. block state、committed `v_cells/head`、free-list 和 backing metadata 不得承担 overlay 角色；
5. rollback 删除 overlay，commit 原子发布 overlay，任何路径结束后不得遗留 owner、reservation 或 pending bitmap。

## 4. 正常转换

### 4.1 Prepare 与 apply

```text
CLOSED
  -- prepare: 建立 owner/reservation/journal/overlay，零 committed mutation --> PREPARED
PREPARED
  -- apply: overlay 成为 owner 的 graph/input view --> APPLIED
APPLIED
  -- backend 即将获得写权限 --> COMPUTE_STARTED
```

`prepare()` 必须满足：

- 不修改 committed cell metadata、head、block 内容/状态、free-list 或 backing publication；
- 任一步失败均返回 `CLOSED`，且全局状态与调用前逐项相同；
- slot 探测若需要模拟 mutation，只能作用于临时副本。

### 4.2 Commit

正常 KV 写事务只能在 graph 成功且必要 backend synchronization 完成后 commit：

```text
COMPUTE_STARTED
  -- graph success + sync + atomic publication --> CLOSED
```

**`write.commit` 是唯一的 committed visibility 发布点，也是 write transaction 的成功终态。** 它不是“graph 已返回成功”的别名；只有 graph 成功、必要 backend synchronization 完成，并且下列 publication 在同一逻辑原子边界内完成后，才能发布 `write.commit`：

1. 验证 owner、reservation 和所有 fresh writes；
2. 在同一原子边界内递增每个 fresh physical block 的 generation counter；
3. 发布 overlay cell metadata/head；
4. 对新写入 block 发布 `VALID + RESIDENT`；
5. 清理旧 backing authority；
6. 更新 used/free-list；
7. 清除 overlay、journal、reservation 和 owner，进入 `CLOSED`。

`write.commit` payload 必须携带闭包索引 `expected_write_blocks[]` 与 `expected_write_block_count`；每项只声明预期的 `physical_block_id + generation_before + generation_after`，不替代 block truth。原子 publication 完成后，core 必须为每个受影响 physical block 发布独立 `core.block.write_commit` event，记录 `generation_before` 与递增后的 `generation_after`。这些事件是 parser 对逐 block 事实的 authority，但只是对已经发生的 `write.commit` 原子 publication 的事后证明，不构成更早的 visibility 发布点；transaction-level 聚合 block 列表不得替代它们。在 commit 前，其他事务、release scanner、ownership collector 和普通读路径不得看到新 metadata。

### 4.3 Destructive release 与 reuse

```text
VALID + RESIDENT + CLOSED + unowned
  -- ownership recheck + successful discard --> EMPTY + DISCARDED

EMPTY + DISCARDED
  -- reserve in PREPARED/APPLIED -> 状态保持；新内容仅存在于 overlay
  -- successful commit --> VALID + RESIDENT
```

Release 必须：

- 在 destructive mutation 前重新计算 physical-block ownership；
- 只处理 `CLOSED`、无 live/shared owner 的 `VALID + RESIDENT` 或可丢弃 `EMPTY + RESIDENT`；
- 对 invalid mapping 整体 abort，零状态变化；
- 对 transaction-reserved、`OFFLOADED`、`QUARANTINED` block fail-closed；
- 成功后清除 backing metadata，设置 `EMPTY + DISCARDED`，并把 block 恰好一次加入 free-list。

Reuse 不新增 block 级中间状态。其 pending 内容只由 overlay 表达。

### 4.4 Swap-out 与 swap-in

```text
VALID + RESIDENT
  -- begin swap-out --> VALID + EVICTING
  -- full backing write + atomic metadata publication --> VALID + OFFLOADED
  -- failure before publication --> VALID + RESIDENT

VALID + OFFLOADED
  -- begin restore --> VALID + PREFETCHING
  -- full read + validation + unpack + atomic commit --> VALID + RESIDENT
  -- failure before commit --> VALID + OFFLOADED
```

要求：

- swap-out 成功前 resident tensor 是唯一权威副本；
- backing offsets/sizes 只能在完整 block 写成功后一次性发布；
- `VALID + OFFLOADED` commit 后允许另行执行 residency-hint `madvise`；该调用只提示 OS 回收已失去 authority 的 tensor 页面，不属于 offload commit 前置条件，失败不得回滚已提交 backing authority；
- swap-in 必须先完整读入 staging，再提交 tensor 和 `RESIDENT`；
- 部分 I/O 不得产生 `VALID` 的半发布状态；
- active-required restore 失败必须在 graph compute 前传播失败。

## 5. 失败转换

### 5.1 Pre-compute rollback

`PREPARED` 或 `APPLIED` 失败时：

```text
PREPARED/APPLIED -- rollback --> CLOSED
```

rollback 必须：

- 只撤销当前 owner 的 reservation、overlay 和 speculative pages；
- committed metadata、其他事务和 backing authority保持不变；
- 新 block 回到原 `EMPTY` 驻留状态；
- used/free-list 恢复到事务前快照；
- 清除 owner、journal 和 pending bitmap；
- 支持失败后立即重试。

若 speculative page discard 失败，相关 block 不得回 free-list，必须进入 `QUARANTINED`。

### 5.2 Compute-started failure

`COMPUTE_STARTED` 后 backend 可能已写入部分 K/V，旧内容和新内容都不能再被信任：

```text
COMPUTE_STARTED -- compute/sync/commit failure --> affected blocks QUARANTINED, transaction CLOSED
```

要求：

- 禁止降级为普通 pre-compute rollback；
- 所有可能被写入的 block 从 free-list 隔离；
- memory/context 设置 fail-stop latch，后续 decode 拒绝继续使用；
- 错误处理本身必须最终清除 transaction owner，避免永久 owner 泄漏；
- 只有显式 `clear/reset` 可以解除 quarantine 和 fail-stop。

### 5.3 Reset

```text
任意稳定状态 -- explicit clear/reset --> EMPTY + RESIDENT 或 EMPTY + DISCARDED, transaction CLOSED
```

Reset 必须清除：

- `QUARANTINED` 和错误 latch；
- 所有 overlay、journal、owner 和 reservation；
- swap offsets/sizes 和 backing authority；
- pending bitmap；
- block used 标记，并按初始化策略重建无重复 free-list。

本目标契约冻结 reset 身份策略：**explicit clear/reset 保留 `cache_instance_id`，原子递增 `cache_epoch`，并在新 epoch 内把所有 physical block 的 generation counter 初始化为 `0`。** Cache 销毁后重建则必须生成新的 `cache_instance_id`，其 `cache_epoch=0`、所有 generation counter 也从 `0` 开始。旧 epoch 的 transaction、block 或 backing 引用在 reset 后一律失效。

`QUARANTINED` 不得通过 release、swap、普通 rollback 或 reuse 离开；**唯一出口是 explicit clear/reset**。

## 6. 核心不变量

### 6.1 Free-list

- `EMPTY` 且可分配的 block：`used=false`，在 free-list 中恰好一次；
- `VALID`、transaction-reserved、`QUARANTINED` block：`used=true` 或等价地不可分配，不得在 free-list；
- commit、rollback、release、reset 和重复调用后不得出现重复项或计数漂移；
- free-list 成员资格不得被未提交 overlay提前发布。

### 6.2 Owner

- 同一 cache 同时只能有契约允许数量的开放 write transaction；当前目标按单 owner 定义；
- begin、访问、finish 和析构清理都必须校验同一个 owner identity；
- wrong-owner begin/finish/access 必须 fail-closed，且不得清除合法 owner 的资源；
- `CLOSED` 必须等价于 owner、overlay、journal 和 reservation 全部为空。

### 6.3 Backing

- `VALID + OFFLOADED` 必须具有完整且一致的 backing metadata；
- `EMPTY + DISCARDED` 不得保留权威 backing；
- backing write 失败不得发布部分 offsets/sizes；
- restore 失败不得覆盖原 backing authority；
- quarantine 不因 backing 看似完整而自动恢复；
- reuse/commit 不得继承旧内容的 backing authority。

### 6.4 Visibility

- committed `VALID` 只按合法 sequence ownership 对普通读者可见；
- overlay 仅对匹配 owner 可见；
- `OFFLOADED` 数据必须成功 restore 后才能被 tensor 读写；
- `EMPTY + DISCARDED` 的 active-visible read 必须失败；
- `QUARANTINED` 的 read/write/release/swap/reuse 必须全部失败；
- request 的 `ACTIVE / IDLE / RESUME_PENDING` 只影响策略和合法访问集合，不改变 block 内容状态定义。

### 6.5 Physical block identity 与 generation

Block 槽位身份固定为：

```text
physical_block_id = cache_instance_id : cache_epoch : physical_block
```

Generation 是该槽位在当前 epoch 内的单调 content-lifecycle counter，不嵌入 `physical_block_id`：

- cache 创建或 reset 后，所有 physical block 的 generation counter 初始化为 `0`；`0` 表示当前 epoch 内尚未提交 fresh content；
- fresh content 的成功 `write.commit` 在其原子 publication 中将 counter 从 `generation_before` 递增为 `generation_after = generation_before + 1`；随后发布的独立 `core.block.write_commit` event 证明该变化；
- prepare、apply、rollback、release、offload 和 prefetch 不递增 counter；它们必须在独立 block event 中同时记录 `generation_before` 与 `generation_after`，未换代时两者相等；
- release 后槽位虽为 `EMPTY + DISCARDED`，counter 仍保留最后一次 committed generation，直到下一次 fresh `write.commit`；
- offload/prefetch 只迁移同一 generation 的 authority；
- reset 只能通过递增 `cache_epoch` 重新从 generation `0` 开始，禁止在同一 epoch 内清零或复用旧 generation；
- parser、runner 或 server 不得把 physical block index 单独当作内容身份。

## 7. 测试契约

生命周期测试必须断言状态、owner、used/free-list、backing 和 visibility，不得只检查函数返回值。

| 场景 | 必须证明的结果 | 优先级 |
|---|---|---|
| prepare success/failure | 进入 `PREPARED` 前后 committed state 零 mutation；失败完整恢复 | P0 |
| release | owned/shared/transaction/offloaded/quarantined 均不被释放；invalid mapping 零 mutation | P0 |
| release → reuse → commit | overlay owner 隔离；commit 后新内容唯一可见且 free-list 正确 | P0 |
| release → reuse → rollback | 回到原 `EMPTY + DISCARDED`；无 stale owner/overlay；可立即重试 | P0 |
| wrong owner | begin/access/finish 全部 fail-closed，不破坏合法事务 | P0 |
| swap-out/in | 真实 `RESIDENT → OFFLOADED → RESIDENT`、字节一致、原子 metadata publication | P0 |
| swap I/O failure | write 失败保持 resident；read 失败保持 offloaded；无部分发布 | P0 |
| compute-started failure | 所有可能写入 block quarantine；普通 rollback 自动拒绝/升级；clear 后恢复 | P0 |
| quarantine isolation | release、free-list、reuse、read/write 均无法绕过 | P0 |
| destructor cleanup | `PREPARED/APPLIED` 回滚；`COMPUTE_STARTED` quarantine | P0 |
| reset | 保留 instance、递增 epoch、generation 重置为 0；清 owner/backing/error/pending，free-list 无重复并恢复服务 | P0 |
| repeated cycles | 多轮 release/reuse/commit/rollback/swap 后状态和计数不漂移 | P1 |
| request lifecycle separation | ACTIVE/IDLE/RESUME_PENDING 切换不篡改 block 内容状态 | P1 |
| default/unsupported path | feature 关闭或 capability 不满足时保持原行为并 fail-closed | P1 |

所有模型相关测试必须证明目标模型类型和目标分支确实执行；普通退出 `0` 的 skip 不能作为覆盖证据。

## 8. 现有状态的目标映射

| 当前 `paged_block_state` | 目标映射 | 当前状态 |
|---|---|---|
| `UNUSED` | `EMPTY + RESIDENT/DISCARDED` | 已实现但单一枚举无法表达真实驻留。 |
| `RESIDENT` | 通常为 `VALID + RESIDENT` | 已实现但在开放事务中混入未提交共享 metadata。 |
| `RELEASED` | `EMPTY + DISCARDED` | 已实现但应由内容/驻留两轴表达。 |
| `SWAPPED` | `VALID + OFFLOADED` | 已实现但没有显式 `EVICTING/PREFETCHING`。 |
| `PENDING_WRITE` | transaction-owned per-cell overlay | **尚未实现目标模型**；当前仍是 block enum。 |
| `INVALID` | `QUARANTINED` | 已实现 fail-stop 基础语义，但隔离门禁不完整。 |

当前实现入口主要位于：

- [`src/llama-kv-cache.h`](../src/llama-kv-cache.h)：block enum、write transaction state 和 owner/pending 字段；
- [`src/llama-kv-cache.cpp`](../src/llama-kv-cache.cpp)：prepare/apply、release/reuse、swap、finish transaction；
- [`src/llama-context.cpp`](../src/llama-context.cpp)：graph compute、sync、commit/rollback/failure 时序；
- [`src/llama-memory-recurrent.cpp`](../src/llama-memory-recurrent.cpp)：recurrent prepare/find-slot/fail-stop；
- [`tests/test-kv-paged-release-bounded.cpp`](../tests/test-kv-paged-release-bounded.cpp) 与 [`tests/test-recurrent-state-rollback.cpp`](../tests/test-recurrent-state-rollback.cpp)：当前生命周期测试。

## 9. 目标契约—当前实现—缺失测试矩阵

状态标记：

- **已实现但证据不足**：源码已有对应机制，但当前工作树缺少足够动态证据；
- **尚未实现**：源码结构或语义不符合本目标契约；
- **P0 差距**：在关闭前不得声称生命周期契约已经实现。

| 目标契约 | 当前实现 | 缺失测试/证据 | 状态 |
|---|---|---|---|
| 三轴状态正交 | 单一 `UNUSED/RESIDENT/RELEASED/SWAPPED/PENDING_WRITE/INVALID` block enum | 三轴合法组合与非法访问测试 | **尚未实现，P0** |
| `CLOSED/PREPARED/APPLIED/COMPUTE_STARTED` | 只有 `CLOSED/APPLIED/COMPUTE_STARTED` | PREPARED 零 mutation 与失败恢复 | **尚未实现，P0** |
| prepare 零 committed mutation | KV prepare 会调用 mutation 型 `apply_ubatch()` 后恢复；recurrent prepare 也调用 mutation 型 `find_slot()` | KV 间接 purge、recurrent `n/rs_z/fail latch` 全快照对照 | **尚未实现，P0** |
| transaction-owned per-cell overlay | block 级 `PENDING_WRITE` + per-cell bitmap + cache-global owner | wrong-owner access/begin/finish；真实 mid-transaction release | **尚未实现，P0** |
| commit 是唯一 committed visibility 发布点 | `APPLIED` 已直接修改共享 cell metadata/head；commit 主要完成 block state 收尾 | 非 owner 在 commit 前只能看到旧 committed view | **尚未实现，P0** |
| pre-compute rollback | 已有 metadata/block journal 与 discard failure quarantine | 独立 clean rollback、double finish、立即重试 | **已实现但证据不足，P0 测试缺口** |
| compute-started failure quarantine | 已将可能写入 block 标记 `INVALID` 并 latch fail-stop | rollback 自动升级、析构 invalidate、wrong-owner finish | **已实现但证据不足，P0 测试缺口** |
| quarantine 仅 clear/reset | `INVALID` read/write 基本 fail-closed；bounded/legacy release gate 未完整排除 `INVALID` | `INVALID` 不可 release/free/reuse 的闭环测试 | **尚未实现，P0** |
| atomic swap publication | swap-out 完整写后发布；swap-in staging 完整读后提交 | 真实 `RESIDENT → SWAPPED → RESIDENT` 和 I/O failure 生命周期 | **已实现但证据不足，P0 测试缺口** |
| free-list 唯一性 | release/rollback/reset 已维护 used/free-list | 多轮、double finish、partial rollback、quarantine 隔离 | **已实现但证据不足** |
| request/block 状态分离 | request 状态未进入 block enum | ACTIVE/IDLE/RESUME_PENDING 切换的 block-state 不变性 | **已实现但证据不足** |
| recurrent/hybrid failure contract | 已有 post-find-slot fail-stop 与 clear recovery 测试逻辑 | 当前默认 fixture 为非 recurrent/hybrid 模型时直接成功退出，目标分支未执行 | **证据无效，P0 假绿** |

## 10. 完成判定

只有同时满足以下条件，才能将本文件从“目标契约已冻结”提升为“当前实现已满足”：

1. 当前源码移除 block 级 `PENDING_WRITE` 语义并实现 owner-checked cell overlay；
2. `PREPARED` 存在，prepare 对 committed state 可证明零 mutation；
3. commit 成为唯一 committed visibility 发布点；
4. quarantine 的 read/write/release/swap/reuse 全部门禁闭合，且仅 clear/reset 可解除；
5. 第 7 节全部 P0 测试真实执行目标分支并通过；
6. 相关 implement/review gate 通过；
7. 至少一个真实短模型路径覆盖 release→reuse→commit/rollback、swap 和 compute-started failure。

在此之前，正确表述是：**目标契约已冻结；当前实现部分具备，但仍有 P0 语义和证据差距。**
