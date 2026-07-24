# 高风险状态与证据门禁

仅在 context-policy Level 2 触发时读取。

## 状态/生命周期契约

必须明确：

1. 状态集合及 authority；
2. 合法/非法转移；
3. owner、shared、active visibility；
4. prepare/apply/compute/commit/rollback；
5. I/O 或 destructive operation 失败后的权威副本和错误传播；
6. 默认关闭、unsupported 和 fallback；
7. release/offload→reuse、重复循环与 reset。

根因或契约未明确时：`audit` 继续；`implement` 停止；`review-fix` 退回 audit；`review` 必须主动找反例。

## 根因顺序

症状 → 首个真实错误 → 实际 argv/env/binary/model/topology → capability/default → 上游调用链 → 状态/所有权 → 物理操作 → 外部观测。

至少保留一个竞争解释，直到最小观测能区分。不能从 HTTP 500、最后一行 stderr、RSS 变化或 marker 直接跳到 core 根因。

## 生命周期最低测试矩阵

- 首次正常路径；
- destructive/reclaim/offload；
- 后续 reuse；
- 成功 commit；
- pre-compute rollback；
- compute-started failure/fail-stop；
- 重复循环/幂等；
- 非当前事务；
- active/owned/shared；
- 默认关闭/unsupported；
- I/O、mapping、madvise 或 parser failure propagation。

只测试函数返回值或 syscall 成功不构成闭环。

## 证据分层

- 静态：调用链、类型、配置和不变量；
- 单元：真实执行目标分支和失败路径；
- 短集成：相关 target + deterministic smoke；
- dirty-tree 真实诊断：记录完整身份，不能归档；
- clean-HEAD 正式协议：可进入账本；
- 用户长测：并发、长上下文、cgroup、性能和稳定性。

Transition、physical operation、system observation、correctness 和 identity 不能互相冒充。
