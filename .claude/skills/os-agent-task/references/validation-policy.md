# 验证与证据策略

## 验证阶梯

根据改动风险选择最低但充分的验证，不机械运行全量测试。

1. **静态层**
   - 直接核对源码和调用链；
   - `git diff --check`；
   - Shell/Python 语法检查；
   - 编译器警告和类型/边界检查。

2. **单元层**
   - 运行直接相关单测；
   - 覆盖正常路径、边界和错误路径；
   - 测试不能依赖修改后的错误标准来通过。

3. **短集成层**
   - 构建相关 target；
   - 小规模 deterministic smoke；
   - 固定 seed、温度和关键参数；
   - 验证退出码、机器可读字段和精确输出。

4. **用户长验证层**
   - 多轮模型回归；
   - 一小时稳定性；
   - 正式三轮性能/消融矩阵；
   - 真实 server workload。

代理负责准备命令、脚本和判定规则，用户负责执行第 4 层。

## 当前仓库可复用入口

根据任务相关性选择，不要默认全部运行：

- `tests/test-kv-backing-store.cpp`
- `scripts/run-kv-p0-b2b-regression.sh`
- `scripts/run-kv-p0-io-fault-regression.sh`
- `scripts/run-kv-p0-stability.sh`
- `scripts/kv-final-controlled-e0-e5.sh`
- `examples/kv-idle-swap-resume/`
- `examples/kv-semi-real-multisession/`
- `examples/kv-trace-replay/`

运行前必须从当前脚本和 build 目录确认真实参数，不能复制历史命令后直接假定可用。

## 正确性结论

至少区分：

- 进程成功退出；
- 无 fatal/pending/active-visible/write-to-swapped 等错误；
- baseline 与 variant token/字节精确一致；
- 故障后状态保持、错误传播和重试一致；
- 量化/压缩路径的精度或 PPL 变化。

“没有崩溃”不等于“正确”。

## 性能结论

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

## 结果措辞

- 已运行且通过：`已实现并验证`；
- 代码完成但只通过静态/短测：`已实现，正式证据不足`；
- 仅设计：`尚未实现`；
- 推测：`仅供讨论，目前无法确认`。
