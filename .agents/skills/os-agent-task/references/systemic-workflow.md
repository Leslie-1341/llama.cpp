# 系统级分析与高风险状态任务策略

## 目的

防止代理只追逐最后一条错误、只修当前函数或只让测试变绿。低风险任务做轻量全局检查；高风险、跨模块或连续失败任务才展开完整状态/生命周期契约。

## 1. 轻量全局检查

所有非 `memory` 任务开始前内部确认：

1. 当前阶段和被阻塞门禁；
2. 表面症状、首个真实错误、根因是否已有直接证据；
3. build 身份、argv/env、默认参数、数据类型、拓扑和 capability；
4. 跨模块状态、不变量和生命周期；
5. 当前证据层能证明什么、不能证明什么；
6. 是否有尚未排除的替代解释；
7. 下一动作能否证伪判断，而不只是增加日志或修补下游。

低风险任务不机械输出该清单。

## 2. 高风险任务触发条件

命中任一项即使用完整契约：

- release/reclaim/swap、madvise、backing store 等破坏性操作；
- block/cell/page、pending、commit/rollback、ownership、active visibility；
- 并发、异步、共享预算或 I/O 仲裁；
- server slot、跨请求复用、错误闩锁和跨模块错误传播；
- runner/parser/Harness/artifact 等最终证据链；
- 单元测试通过但真实 server/模型失败；
- 同一问题连续两轮未关闭；
- 修改跨三个以上模块或改变公共边界。

## 3. 根因分析顺序

不得从最后一条报错直接跳到局部修改。

1. **全局位置**：阶段目标、上一/下一门禁、问题属于 core、lifecycle、tooling、performance 还是 docs。
2. **首个真实错误**：按时间线找最早异常；HTTP 500、parser FAIL、cleanup error 常为次生结果。
3. **上游条件**：HEAD/dirty tree/binary、argv/env、默认参数、模型与 cache 类型、parallel/topology、capability 构造条件。
4. **状态与不变量**：状态集合、合法/非法转移、owner/visibility、commit/rollback、默认关闭和无 backing 边界。
5. **调用链与时序**：对象实例、wrapper 转发、server/context/memory/backend 顺序、快照有效期、并发与共享资源。
6. **证据工具**：先看原始日志，再看 runner、parser、Harness；不得优先修改工具来回避 core failure。
7. **竞争假设**：根因未确认时至少保留两个解释，并给出一轮可区分的最小观测。

强制自问：

> 当前看到的是根因，还是根因造成的最后一个可见症状？

## 4. 高风险状态/生命周期契约

`implement` 前内部明确以下内容；`contract` 模式写入必要部分。

### 状态和转移

- 当前源码中的真实状态；
- 每条合法转移及负责函数；
- 非法转移和非法访问。

例如：

```text
RELEASED -> PENDING_WRITE      allocate/reuse
PENDING_WRITE -> RESIDENT      success commit
PENDING_WRITE -> RELEASED      failure rollback
RELEASED active-visible read   fatal
```

### 所有权和可见性

- block/cell/page 的 owner；
- active/shared/pending/masked 对当前和其他事务的可见范围；
- snapshot 有效窗口与并发限制。

### commit、rollback 和错误传播

- 何时才算提交；
- rollback 只撤销当前事务，不影响已提交或其他事务数据；
- error latch、返回码、HTTP 和 telemetry 如何一致传播。

### 默认关闭与退化

- feature 关闭时行为不变；
- capability 不满足时 fail-closed 或安全 fallback；
- 无 backing 数据不得伪造恢复语义；
- 关闭时不得产生目标 counter。

### 生命周期测试

至少覆盖：

- 正常路径；
- release/reclaim 后 reuse；
- 成功 commit；
- 失败 rollback；
- 重复循环；
- 非当前事务访问；
- active/owned/shared 保护；
- 默认关闭；
- 错误传播；
- 一个短真实 server/模型路径。

函数级返回值正确不等于生命周期闭环。

## 5. 模式约束

### audit

根因未确认时保持只读，输出症状、首错、不变量、根因置信度、替代假设和最小证伪动作。

### implement

高风险契约不完整或新证据推翻原根因时停止修改。先生命周期单测，再早期短集成；不要先反复美化 parser/report。

### review

独立重建上游条件和状态转移，至少尝试一个反例或失败路径；检查验证工具是否掩盖 core failure。

### review-fix

只有“直接证据 + 被破坏不变量 + 最小修复范围 + 原失败路径测试”齐全才允许修改；否则输出“证据不足，退回 audit”。

### script

runner 只记录事实；parser 才能形成正式 protocol verdict；Harness 只证明短检查。测试必须实际执行目标分支。

## 6. 证据层

| 证据 | 可支持结论 |
|---|---|
| 静态/build | 文件可解析、目标可编译 |
| 单元测试 | 隔离函数和确定性状态转移 |
| 短集成 | 指定短调用链成立 |
| dirty-tree 真实实验 | 当前工作树诊断成立，不可归档 |
| clean-HEAD 受控复跑 | 当前提交正式协议成立 |
| 多轮正式实验 | 指定条件下性能与稳定性 |

Gate PASS 不得抬高证据层。

## 7. 强制重新审计信号

出现任一情况，停止局部修补并重新 `audit`：

- 同一问题连续两轮未关闭；
- 错误点每轮向下游移动；
- 单元测试持续 PASS、真实 server 持续失败；
- parser/Harness 修改次数超过 core 修改次数；
- 修复需要放宽安全门禁或 baseline；
- 当前解释无法覆盖完整 lifecycle。

## 8. 输出原则

系统级思考主要在内部完成。高风险任务最终只保留：全局位置、症状与根因状态、不变量、关键证据/替代假设、阶段影响和唯一下一动作。低风险任务保持原有精简风格。
