---
name: os-agent-task
description: Lightweight evidence-driven workflow for one bounded OS competition engineering task: contract, audit, implement, review, review-fix, script, or memory. Invoke explicitly with $os-agent-task or /os-agent-task.
argument-hint: "<contract|audit|implement|review|review-fix|script|memory> <目标与验收>"
disable-model-invocation: true
---

# OS Agent Task

把用户目标收敛为一次可独立验收的任务。优先当前源码、diff 和运行证据；不把局部报错、代理摘要遗漏或未来理论风险直接升级为阻塞项。

## 1. 解析输入

提取：`mode`、目标、验收、本轮特殊边界。未给 mode 时按语义选择：

- 指令/提示词 → `contract`
- 审计/定位/评估 → `audit`
- 实现/修改/接入 → `implement`
- 审查当前改动 → `review`
- 按已确认阻塞修复 → `review-fix`
- 实验/回归/解析脚本 → `script`
- 工程账本初始化/检查/同步 → `memory`

只有权限或目标确实歧义时询问一次；不要为恢复固定背景追问。

## 2. 最小上下文策略

先运行：

```bash
bash "${SKILL_DIR:-.agents/skills/os-agent-task}/scripts/collect-task-context.sh"
```

然后按 [references/context-policy.md](references/context-policy.md) 取上下文。核心规则：

1. 默认只读当前 Git 身份、diff 摘要、`PROJECT_STATE.md` 当前阶段/阻塞/下一门禁，以及本轮直接相关源码和测试。
2. `ARCHITECTURE.md`、`DECISIONS.md`、`EXPERIMENTS.md` 先看标题索引，只读取匹配章节；禁止默认整本加载。
3. `EXPERIMENTS.md` 仅用于实验、脚本、正式结果或证据口径任务。
4. 高风险任务才读取 [references/high-risk.md](references/high-risk.md) 和对应目标契约；不读取无关历史章节。
5. 定位工具一次只选一种：优先 CodeGraph 或 `rg`；首轮不足再换工具，避免 CodeGraph、grep、全仓搜索重复定位。
6. 日志、diff、Gate 默认只读摘要和命中片段，不把完整大文件灌入上下文。

仓库客户端通常已注入 `AGENTS.md` 或 `CLAUDE.md`。除非规则缺失、冲突、发生变化或本轮修改治理文件，不再主动完整读取两份。

## 3. 固定契约

执行 [references/project-contract.md](references/project-contract.md)。只在一个地方维护固定 Git、证据、分工和账本规则，不在各模式重复加载。

## 4. 阻塞项准入

新增阻塞项必须同时满足：

- **源码可达**：真实调用链可到达；
- **阶段内**：属于当前节点目标；
- **真实风险**：会导致错误行为、错误传播或证据假通过。

代理未提及某路径、parser 未覆盖某字段、理论上可能出错、未来扩展不完整，都不能单独构成阻塞项。同一问题连续两轮未关闭时停止局部 `review-fix`，退回一次系统级 `audit`。

## 5. 选择并读取当前模式

只读取一个模式文件：

```text
references/modes/<mode>.md
```

模型推荐仅在 `contract` 或用户明确询问时读取 [references/model-selection.md](references/model-selection.md)。不要读取其他模式、示例或通用输出模板。

## 6. Gate 与验证

- `implement`、`review-fix`：修改完成后运行对应 Gate。
- `review`：运行 review Gate，除非本次只审查用户已提供且身份完整的同一 diff Gate artifact；复用时必须明确 artifact、HEAD 和当前 diff 未变化。
- `audit`：只读 audit Gate；clean tree 的 `NO_CHANGES` 不证明正确性。
- `contract`、纯 `memory` 不强制 Gate；`script` 修改 Harness/parser 时按实际修改运行相应 Gate。

```bash
bash scripts/os-agent/gate-runner <mode>
```

默认只读取 `OS_AGENT_GATE_RESULT` 和 `summary.txt`。非 PASS 时先按失败 check 名称在 `full.log` 中提取局部上下文，不直接读取完整日志。

Gate verdict：`PASS / FAIL / UNRESOLVED / INCOMPLETE / NO_CHANGES / UNVERIFIED`。只有 PASS 可声明该层短检查通过；Gate 不替代真实 server、模型、clean-HEAD 或正式实验。

## 7. 账本边界

普通任务不自动修改四份账本。只有显式 `memory update` 才同步。任务结束仅在确有影响时给一行账本建议。

- `PROJECT_STATE.md`：当前状态，可覆盖；
- `ARCHITECTURE.md`：已验证稳定结构；
- `DECISIONS.md`：追加/supersede；
- `EXPERIMENTS.md`：追加实验索引，不复制大日志。

## 8. 完成措辞

- 静态/单元/短集成：`代码已实现并通过短验证，真实路径尚未验证`；
- dirty-tree 真实短协议：`已诊断验证，尚不可归档`；
- clean-HEAD 正式协议：`已实现并验证`；
- 证据不足：`目前无法确认`。

最终输出结论先行，列修改/证据/未验证/唯一下一步；不复述固定背景和常规工具过程。
