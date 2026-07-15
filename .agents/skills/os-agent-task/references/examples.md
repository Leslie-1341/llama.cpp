# 调用示例

## 1. 生成 CC/Codex 指令

```text
$os-agent-task contract
为 active token 延迟增加 p50、p95、p99 机器可读统计；默认关闭，不改变生成结果。
```

## 2. 只读瓶颈审计

```text
$os-agent-task audit
审计当前同步分步预取是否造成 active token 尾延迟，并判断后台异步化是否值得做。
```

## 3. 直接实现

```text
$os-agent-task implement
修复 E0–E5 parser 假通过：UNVERIFIED、矩阵缺失/重复、必要指标缺失或重复关键字段均非零退出；保留现有协议和 dry-run，用小型 fixture 覆盖。
```

## 4. 审查当前 diff

```text
$os-agent-task review
审查当前 B6 block-I/O 工作区修改，只报告阻塞问题、证据不足和最小补测。
```

## 5. 定向修复

```text
$os-agent-task review-fix
只修复审查确认的两个阻塞项：range read 失败前存在 tensor 部分写回；统计脚本缺失字段仍可能 PASS。
```

## 6. 写正式实验脚本

```text
$os-agent-task script
编写 off、1/1、4/4、8/8 四种预取策略的三轮交错复测脚本；正式模型实验由用户执行。
```

## 7. 低价值方案判断

```text
$os-agent-task audit
评估把 KV pread/pwrite 改成 io_uring 的投入产出比；没有当前耗时证据不得建议实现。
```

## 8. 模块融合审计

```text
$os-agent-task audit
定位 KV resume prefetch 与 Dense/MoE 权重预取的统一 I/O 仲裁接入点，不修改代码。
```

## 9. 初始化工程账本

```text
$os-agent-task memory init
```

## 10. 检查账本是否陈旧

```text
$os-agent-task memory check
核对四个工程账本与当前分支、diff、源码和 E0–E5 审计结论是否一致。
```

## 11. 同步已验证成果

```text
$os-agent-task memory update
把已通过 review 和短验证的 E0–E5 parser 修复同步到项目状态与决策日志；未运行的正式性能结果不得写入实验账本。
```
