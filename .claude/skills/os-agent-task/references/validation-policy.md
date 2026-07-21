# 验证与证据策略

## 1. 验证原则

根据改动风险选择最低但充分的验证，不机械运行全量测试。验证必须与结论证据层匹配：build 不能证明真实 server 正确，单元测试不能证明跨请求生命周期，Gate PASS 不能证明正式实验成立。

高风险状态任务必须同时验证：

- 目标操作本身；
- 操作后的 reuse/read/write；
- 成功 commit；
- 失败 rollback；
- 重复循环和幂等；
- active/owned/shared/非当前事务保护；
- 默认关闭和 fallback；
- 错误传播；
- 至少一个真实短集成路径。

## 2. 验证阶梯

### 2.1 静态层

- 直接核对源码、状态转移和调用链；
- 核对 argv/env、默认参数和 capability 条件；
- `git diff --check`；
- Shell/Python 语法检查；
- 编译器警告和类型/边界检查。

只能证明文件和目标具备基础可执行性。

### 2.2 单元层

- 运行直接相关单测；
- 覆盖正常、边界、错误和回滚路径；
- 测试必须实际执行目标分支；
- 不得只依赖 import、grep、py_compile 或修改后的错误标准；
- 高风险改动按状态转移和生命周期组织测试，而不是只按函数组织。

### 2.3 短集成层

- 构建相关 target；
- 小规模 deterministic smoke；
- 固定 seed、温度、模型/cache 类型、拓扑和关键参数；
- 验证退出码、HTTP、机器可读字段和精确输出；
- 尽早验证真实调用链，不在真实路径未通时先反复美化 parser/report。

代理负责与修改强相关的定向构建、单元测试和无模型短 smoke。真实模型/server 短 smoke 通常由用户执行；若环境允许且任务明确允许，代理可执行极短诊断，但不得运行正式矩阵。

### 2.4 dirty-tree 真实诊断层

- 使用当前工作树 binary；
- 记录 HEAD、dirty 状态、diff、binary、model 和 runner 身份；
- 可证明当前工作树在指定协议下的诊断行为；
- 不得作为可归档正式结论。

### 2.5 clean-HEAD 受控验证层

- 用户形成稳定提交；
- clean HEAD 重新构建；
- 使用同一 runner/parser 和固定协议复跑；
- artifact 身份完整、parser PASS；
- 才可进入工程账本和正式材料。

### 2.6 用户长验证层

- 多轮模型回归；
- 一小时稳定性；
- 正式三轮性能/消融矩阵；
- 真实并发和 server workload；
- cgroup、PSI、strace、mincore 和长期 I/O 故障。

## 3. 生命周期验证模板

状态或破坏性操作至少设计以下场景：

| 场景 | 核心检查 |
|---|---|
| 首次正常路径 | 状态、输出、counter、错误均正确 |
| release/reclaim | 只处理合法候选，物理效果可证 |
| release 后 reuse | 分配、写入、row mapping、decode 成功 |
| commit | 只在 graph/IO 成功后成为可见状态 |
| rollback | 只撤销当前事务，无陈旧 owner/cell/page |
| 重复循环 | 多轮后状态、free list、counter 不漂移 |
| 非当前事务 | fail-closed，不读取 pending/invalid 数据 |
| active/owned/shared | 不被回收或错误重映射 |
| 默认关闭 | 行为、输出和目标 counter 不变 |
| 错误传播 | 首错、error latch、返回码和 HTTP 一致 |

只测试 `release()` 返回值或 `madvise()` 成功不等于生命周期闭环。

## 4. 当前仓库可复用入口

根据任务相关性选择，不要默认全部运行：

- `tests/test-kv-backing-store.cpp`
- `scripts/run-kv-p0-b2b-regression.sh`
- `scripts/run-kv-p0-io-fault-regression.sh`
- `scripts/run-kv-p0-stability.sh`
- `scripts/kv-final-controlled-e0-e5.sh`
- `examples/kv-idle-swap-resume/`
- `examples/kv-semi-real-multisession/`
- `examples/kv-trace-replay/`

运行前必须从当前源码、脚本和 build 目录确认真实参数，不能复制历史命令后直接假定可用。

## 5. 正确性结论

至少区分：

- 进程成功退出；
- HTTP、decode 和生成输出正确；
- 无 fatal/pending/active-visible/write-to-swapped 等错误；
- baseline 与 variant token/字节精确一致；
- 故障后状态保持、错误传播和重试一致；
- release 后 reuse/commit/rollback 闭环；
- 量化/压缩路径的精度或 PPL 变化。

“没有崩溃”“build 通过”“Gate PASS”“机制日志出现”均不等于正确。

## 6. 验证工具规则

### runner

- 记录完整命令、argv/env、binary/model、stdout/stderr、退出码、HTTP 和 artifact 身份；
- 只写 `run_complete`、`incomplete`、`unverified` 等运行状态；
- 不写最终 `PASS`。

### parser

- 是正式 protocol verdict authority；
- 缺失、重复、顺序错误、格式错误、身份冲突、未执行目标 case 均非零退出；
- 不得把背景机制计数归因给目标机制；
- 不得因当前实现失败而放宽 baseline、response identity 或安全字段。

### Harness

- 只证明 diff-aware 短检查；
- 必要检查跳过必须 `UNVERIFIED`；
- 相关 knownfail 必须 `UNRESOLVED`；
- unrelated knownfail 需有基线证据，不得永久人工豁免；
- Gate PASS 不抬高到真实模型或正式实验结论。

## 7. 性能结论

必须记录：

- baseline、variant 和当前 commit/工作树；
- build 类型、硬件、系统、模型和量化；
- prompt/output、ctx、batch、并发、预热；
- 指标定义和采样位置；
- 重复次数、中心值和波动；
- 异常运行及剔除原因。

至少关注：RSS/current/peak、KV resident bytes、TPS、TTFT、TPOT、p50/p95/p99、I/O bytes/syscalls、CPU/iowait，以及适用的正确性指标。

禁止：

- 用单次最好结果作为正式结论；
- 将开关消融误称为完整 baseline 对比；
- 将调用次数下降直接等同于端到端性能提升；
- 在正确性门槛失败时继续比较性能；
- 静默删除退化或失败结果。

## 8. 结果措辞

- 静态/单元层通过：`代码已实现并通过短验证，真实路径尚未验证`；
- dirty-tree 真实协议通过：`已诊断验证，尚不可归档`；
- clean-HEAD 正式协议通过：`已实现并验证`；
- 仅设计：`尚未实现`；
- 推测或证据不完整：`目前无法确认`。
