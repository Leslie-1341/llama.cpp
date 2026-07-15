---
name: os-agent-task
description: Explicit OS competition engineering workflow for turning a concise goal into one controlled task: contract, audit, implement, review, review-fix, script, or project-memory maintenance. Use for requests such as “给我 CC/Codex 指令”, “审计这条源码链路”, “修改这部分代码”, “审查当前 diff”, “按审查结论修复”, “写实验/回归脚本”, or “初始化/同步工程账本”. Do not auto-trigger for ordinary questions; invoke explicitly with $os-agent-task or /os-agent-task.
argument-hint: "<contract|audit|implement|review|review-fix|script|memory> <目标与验收标准>"
disable-model-invocation: true
---

# OS Agent Task

把用户的简短目标转换为一次边界明确、证据驱动、可验收的工程任务。结果优先，不把提示词写成冗长施工步骤。

## 1. 输入

优先从调用参数或用户消息中提取：

- `mode`：`contract`、`audit`、`implement`、`review`、`review-fix`、`script`、`memory`；
- `目标结果`：完成后系统应表现出的行为或应得到的结论；
- `验收标准`：可观察、可测试、可判定；
- `特殊限制`：仅记录本轮新增边界。

若未显式给出 mode，按语义推断：

- “给我指令/任务词/提示词” → `contract`
- “审计/分析/定位/评估/是否值得做” → `audit`
- “实现/修改/修复/接入” → `implement`
- “审查 diff/检查当前改动/能否验收” → `review`
- “按审查结论修复/只修阻塞项” → `review-fix`
- “写脚本/实验矩阵/回归工具/解析器” → `script`
- “初始化/检查/同步项目状态、架构、决策或实验账本” → `memory`

只有权限边界确实无法判断时才询问一次。不要为了补齐固定背景而追问。

## 2. 开始前

1. 阅读仓库根目录的 `AGENTS.md` 和 `CLAUDE.md`。
2. 阅读 [references/project-contract.md](references/project-contract.md) 与 [references/ledger-policy.md](references/ledger-policy.md)。
3. 读取仓库根目录中已存在的工程账本：
   - `PROJECT_STATE.md`
   - `ARCHITECTURE.md`
   - `DECISIONS.md`
   - `EXPERIMENTS.md`
4. 按任务最小读取：
   - `contract`：优先读取 `PROJECT_STATE.md`、`ARCHITECTURE.md`、`DECISIONS.md`；
   - `audit`、`implement`、`review`、`review-fix`：读取前三个，涉及实验口径时再读 `EXPERIMENTS.md`；
   - `script`、`memory`：读取全部四个。
5. 除纯 `contract` 模式外，运行只读脚本：

   ```bash
   bash "${SKILL_DIR:-.agents/skills/os-agent-task}/scripts/collect-task-context.sh"
   ```

   若当前工具不提供 `SKILL_DIR`，直接从仓库根目录执行对应脚本路径。
6. 以事实来源优先级工作：当前源码、当前 diff、实际测试输出 > 工程账本 > README/docs > 历史总结或聊天记录。
7. 工程账本是快速恢复上下文的索引，不得覆盖当前代码事实。账本与源码冲突时，以源码和直接证据为准，并报告账本陈旧。
8. 只读取与当前目标直接相关的代码、测试和文档。陌生跨文件调用链、公共接口或核心状态修改前可使用 CodeGraph；最终结论仍需直接核对源码。

## 3. 通用任务契约

内部先形成以下五项，不必机械复述给用户：

1. **目标结果**：最终要实现或证明什么；
2. **必要上下文**：哪些当前事实会改变实现判断；
3. **关键边界**：不得破坏什么、允许改什么；
4. **验收标准**：怎样机器可判定地通过；
5. **最终输出**：需要报告哪些证据。

若一个请求包含多个独立目标，优先完成最高优先级且可独立验收的一项，不自行扩大范围。

## 4. 选择模式并执行

读取 [references/modes.md](references/modes.md) 中对应模式。严格遵守该模式的允许动作、停止条件和交付物。

验证要求读取 [references/validation-policy.md](references/validation-policy.md)。

最终输出使用 [references/output-templates.md](references/output-templates.md) 中对应模板，但删去不适用栏目，避免空泛填充。

## 5. 工程账本更新边界

- 普通 `audit`、`implement`、`review`、`review-fix`、`script` 不自动修改四个账本。
- 任务结束时，仅在确有影响时输出一行 `工程账本影响`，指出建议更新的文件和事实；无影响则省略。
- 只有用户显式调用 `memory update`，或明确要求“同步工程账本”，才允许修改账本。
- `PROJECT_STATE.md` 可更新当前快照；`ARCHITECTURE.md` 只记录已验证结构；`DECISIONS.md` 与 `EXPERIMENTS.md` 采用追加或显式 supersede，不重写历史。
- 不把计划写成已完成，不把单次或未验证结果写成正式实验结论。

## 6. 全局边界

- 不自动执行 `git commit`、`git push`、merge、rebase、tag、release、`git reset --hard` 或 `git clean -fd`。
- 默认不生成 patch。只有用户明确要求，或需要跨环境交付增量差异、严格审查或回滚时才生成。
- 不覆盖与当前任务无关的已有改动。若工作区已有修改与目标文件重叠且无法安全区分，停止并报告。
- 可执行与修改范围匹配的构建、静态检查、单元测试和短 smoke；长时间模型回归、稳定性测试和正式性能实验只编写脚本与执行命令，由用户运行。
- 不修改测试标准、baseline 参数或解析规则来掩盖失败。
- 不把理论收益、单次结果、probe 或历史数据写成当前实测结论。
- 发现 P0 问题时停止低优先级优化，先报告 P0。
- 不顺手重构、不增加无关功能、不生成提交信息或 PR 文案，除非用户明确要求且用于本私有竞赛仓库。
- 回复使用中文，直接给结论；不输出内部推理过程，不逐条叙述常规工具调用。

## 7. 完成判定

只有同时满足以下条件才可写“已完成”：

- 目标行为已实现或目标结论已有直接证据；
- 必要的短验证已通过，或明确说明由用户执行的长验证尚未完成；
- 没有隐藏的失败项；
- 修改范围与任务目标一致；
- 已列出未确认事项和剩余风险。

否则使用“实现完成但证据不足”“审计完成，尚未实现”或“目前无法确认”等准确表述。

## 8. 示例

需要调用示例时阅读 [references/examples.md](references/examples.md)。
