# 调用示例

## 1. 生成普通 CC/Codex 指令

```text
$os-agent-task contract
为 active token 延迟增加 p50、p95、p99 机器可读统计；默认关闭，不改变生成结果。
```

输出应包含推荐模型，但不重复仓库固定规则。

## 2. 只读瓶颈审计

```text
$os-agent-task audit
审计当前同步分步预取是否造成 active token 尾延迟，并判断后台异步化是否值得做。
```

## 3. 高风险状态机审计

```text
$os-agent-task audit
定位 RELEASED block 在新请求复用后的 HTTP 500；先核对真实 argv/env、首个错误、状态转移、commit/rollback 和 row visibility，不修改代码。
```

审计不得从 HTTP 500 直接跳到最后一个报错函数，必须区分症状、首错、根因和替代假设。

## 4. 高风险任务契约

```text
$os-agent-task contract
为 bounded release 的 release→reuse→commit/rollback 闭环生成实现任务；保留 active/ownership/fail-closed，不运行正式三组模型实验。
```

契约应包含推荐模型、状态与不变量、失败回滚和生命周期测试。

## 5. 直接实现

```text
$os-agent-task implement
修复 E0–E5 parser 假通过：UNVERIFIED、矩阵缺失/重复、必要指标缺失或重复关键字段均非零退出；保留现有协议和 dry-run，用小型 fixture 覆盖。
```

## 6. 根因不足时不得实现

```text
$os-agent-task implement
修复 server 中偶发的 KV HTTP 500。
```

若没有原始 artifact、首错、不变量和可复现路径，应停止修改并输出“证据不足，需先 audit”。

## 7. 审查当前 diff

```text
$os-agent-task review
审查当前 bounded release 工作区修改，独立检查上游配置、release 后 reuse、成功 commit、失败 rollback 和默认关闭路径。
```

## 8. 定向修复

```text
$os-agent-task review-fix
只修复审查确认的两个阻塞项：range read 失败前存在 tensor 部分写回；统计脚本缺失字段仍可能 PASS。
```

若审查结论缺少直接证据或最小验收测试，应退回 audit，不边猜边改。

## 9. 写正式实验脚本

```text
$os-agent-task script
编写 off、1/1、4/4、8/8 四种预取策略的三轮交错复测脚本；runner 只记录事实，parser 唯一给最终 verdict，正式模型实验由用户执行。
```

## 10. 低价值方案判断

```text
$os-agent-task audit
评估把 KV pread/pwrite 改成 io_uring 的投入产出比；没有当前耗时证据不得建议实现。
```

## 11. 模块融合审计

```text
$os-agent-task audit
定位 KV resume prefetch 与 Dense/MoE 权重预取的统一 I/O 仲裁接入点，不修改代码。
```

## 12. 初始化工程账本

```text
$os-agent-task memory init
```

## 13. 检查账本是否陈旧

```text
$os-agent-task memory check
核对四个工程账本与当前分支、diff、源码和 E0–E5 审计结论是否一致。
```

## 14. 同步已验证成果

```text
$os-agent-task memory update
把 clean-HEAD 正式协议已通过的成果同步到项目状态与决策日志；dirty-tree 诊断和未运行性能结果不得写成正式证据。
```
