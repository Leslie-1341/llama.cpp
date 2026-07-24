---
name: os-agent-task
description: Explicit OS competition engineering workflow for turning a concise goal into one controlled task: contract, audit, implement, review, review-fix, script, or project-memory maintenance. Use for requests such as “给我 CC/Codex 指令”, “审计这条源码链路”, “修改这部分代码”, “审查当前 diff”, “按审查结论修复”, “写实验/回归脚本”, or “初始化/同步工程账本”. Do not auto-trigger for ordinary questions; invoke explicitly with $os-agent-task or /os-agent-task.
argument-hint: "<contract|audit|implement|review|review-fix|script|memory> <目标与验收标准>"
disable-model-invocation: true
---

# OS Agent Task

把用户的简短目标转换为一次边界明确、证据驱动、可验收的工程任务。结果优先，但不得把局部报错直接当作根因；先确认当前阶段、系统不变量和完整生命周期，再决定是否修改。

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

## 2. 系统级视角门禁

除纯 `memory` 外，开始任务前必须执行一次内部“全局检查点”，但不必把内部过程逐条输出：

1. 当前任务位于哪个阶段、阻塞哪一道门禁；
2. 当前看到的是表面症状、首个真实错误，还是已经有证据的根因；
3. 是否存在更上游的构建身份、argv/env、默认参数、模型/数据类型、拓扑或 capability 前置条件；
4. 涉及哪些跨模块状态、不变量、资源所有权、可见性和完整生命周期；
5. 当前结论由哪一层证据支持，哪一层仍未验证；
6. 是否至少存在一个尚未排除的替代解释；
7. 最小下一动作能否证伪当前判断，而不只是继续修补最后一条报错。

高风险或连续失败任务读取 [references/systemic-workflow.md](references/systemic-workflow.md)；`contract` 或需要推荐模型时读取 [references/model-selection.md](references/model-selection.md)。

### 任务粒度与阻塞项规则

- 任务保持中等粒度：一个任务对应一个可独立验收的目标结果。不把多个不相关的小修复合并为一个任务，也不把一个需要多轮 audit→implement→review 的复杂问题压缩为单轮。
- **新增阻塞项必须先证明**：源码可达性（真实调用链可达，非推测）、阶段范围（在当前阶段目标范围内）、真实风险（会产生可观测错误行为或假通过）。纯理论风险、未经证实的“可能存在问题”或 agent 输出未提及某路径，本身不构成阻塞项。
- **禁止将 agent 输出遗漏当作源码缺陷**：agent 未提及某调用链 ≠ 该调用链缺失；parser/Harness 未覆盖字段 ≠ core 未输出字段。覆盖缺口只有在当前协议明确要求该字段，且能够证明会造成假通过、错误归因或漏报时，才作为 tooling 阻塞项；否则记录为非阻塞建议。
- 同一问题连续两轮未关闭时，必须停止局部修补，重新执行系统级 audit。

以下任一情况视为**高风险系统任务**，必须使用完整状态/生命周期契约：

- destructive release/reclaim/swap、持久化或 backing store；
- block/cell/page、pending、commit/rollback、ownership、active visibility；
- 并发、异步、跨线程、共享资源仲裁；
- server slot、请求生命周期、错误闩锁和跨模块错误传播；
- parser/runner/Harness 等最终证据链；
- 同时修改三个以上模块，或真实路径与单元测试结论冲突；
- 同一问题已经连续两轮以上未关闭。

高风险任务在根因、不变量、合法/非法状态转移和失败回滚尚未明确时：

- `audit` 继续审计，不得猜测性给出实现结论；
- `implement` 停止修改，报告缺失契约；
- `review-fix` 退回只读审计，不得边猜边修；
- `review` 必须尝试推翻实现者的核心假设，而不是只确认局部代码。

## 3. Gate 门禁（必执行）

`implement`、`review`、`review-fix`、`audit` 四种模式在完成主要工作后、声明任务状态前，**必须**执行 diff-aware agent-gate：

```bash
bash scripts/os-agent/gate-runner <mode>
```

- `implement` → `bash scripts/os-agent/gate-runner implement`
- `review` → `bash scripts/os-agent/gate-runner review`
- `review-fix` → `bash scripts/os-agent/gate-runner review-fix`
- `audit` → `bash scripts/os-agent/gate-runner audit`（只读，不构建）

Gate 输出格式：`OS_AGENT_GATE_RESULT mode=<mode> verdict=PASS|FAIL|UNRESOLVED|UNVERIFIED|NO_CHANGES|INCOMPLETE code=<n> ...`

- **PASS (code=0)**：仅表示该 gate 中必要的短检查通过；不能替代真实 server、模型、长周期或 clean-HEAD 证据。
- **FAIL (code=1)**：存在已执行并失败的检查，禁止声明完成。
- **UNRESOLVED (code=2)**：存在相关 knownfail、未注册测试或未关闭歧义，禁止声明完成。
- **INCOMPLETE (code=3)**：检查链未完整执行，禁止声明完成。
- **NO_CHANGES (code=4)**：无改动可检查，可报告无改动，但不能据此证明行为正确。
- **UNVERIFIED (code=5)**：必要检查被跳过或证据层不足，禁止声明完全完成。

非 PASS 时必须检查 artifact 中的 `full.log` 和 `summary.txt`。不得通过删除测试、放宽 parser、修改 baseline 或把相关 knownfail 标成 unrelated 来获得绿色结果。

`contract`、`script`、`memory` 模式不强制运行 gate；若实际修改源码或 Harness，仍需运行对应模式 gate。

## 4. 开始前

1. 阅读仓库根目录的 `AGENTS.md` 和 `CLAUDE.md`，但不要重复其中固定规则。
2. 始终阅读 [references/project-contract.md](references/project-contract.md) 与 [references/ledger-policy.md](references/ledger-policy.md)。高风险/连续失败任务再读 [references/systemic-workflow.md](references/systemic-workflow.md)；`contract` 或模型选择任务再读 [references/model-selection.md](references/model-selection.md)。
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
6. 事实来源优先级：当前源码、当前 diff、实际测试/实验原始输出 > 当前构建身份和机器可读元数据 > 工程账本 > README/docs > 历史总结或聊天记录。
7. 工程账本是快速恢复上下文的索引，不得覆盖当前代码事实。冲突时指出账本陈旧，不自动调和。
8. 只读取与当前目标相关的代码、测试和文档，但跨文件状态机、公共接口或核心生命周期修改不得只读单个函数；必要时使用 CodeGraph 定位，最终结论直接核对源码。

## 5. 通用任务契约

内部先形成以下七项，不必机械复述给用户：

1. **全局位置**：当前阶段、上一门禁、下一门禁；
2. **目标结果**：最终要实现或证明什么；
3. **症状与根因状态**：已知症状、首个错误、根因置信度；
4. **必要上下文**：会改变判断的配置、默认值、拓扑和当前事实；
5. **关键边界与不变量**：不得破坏什么、允许改什么；
6. **验收标准与证据层**：怎样通过、由哪一层证据证明；
7. **最终输出**：需要报告哪些证据、未确认事项和唯一下一动作。

若一个请求包含多个独立目标，优先完成最高优先级且可独立验收的一项，不自行扩大范围。

## 6. 选择模式并执行

读取 [references/modes.md](references/modes.md) 中对应模式，严格遵守允许动作、系统级检查、停止条件和交付物。

验证要求读取 [references/validation-policy.md](references/validation-policy.md)。

最终输出使用 [references/output-templates.md](references/output-templates.md) 中对应模板，删去不适用栏目。低风险任务保持简洁；高风险任务只输出必要的全局结论，不倾倒内部推理过程。

## 7. 工程账本更新边界

- 普通 `audit`、`implement`、`review`、`review-fix`、`script` 不自动修改四个账本。
- 任务结束时，仅在确有影响时输出一行 `工程账本影响`；无影响则省略。
- 只有用户显式调用 `memory update`，或明确要求同步工程账本，才允许修改账本。
- `PROJECT_STATE.md` 可更新当前快照；`ARCHITECTURE.md` 只记录已验证结构；`DECISIONS.md` 与 `EXPERIMENTS.md` 追加或显式 supersede，不重写历史。
- 高风险状态机契约只有在源码、review 和对应证据通过后，才可进入 `ARCHITECTURE.md` 或 `DECISIONS.md`。
- 不把计划写成已完成，不把 dirty-tree 单次结果写成正式实验结论。

## 8. 全局边界

- 不自动执行 `git commit`、`git push`、merge、rebase、tag、release、`git reset --hard` 或 `git clean -fd`。
- 默认不生成 patch。只有用户明确要求，或需要跨环境交付增量差异、严格审查或回滚时才生成。
- 不覆盖与当前任务无关的已有改动。若工作区已有修改与目标文件重叠且无法安全区分，停止并报告。
- 代理执行与修改强相关的定向构建、单元测试、fixture 和短 smoke；全量构建、真实模型 server、strace/mincore/cgroup、长稳定性和正式实验由用户执行。
- 不修改测试标准、baseline 参数、parser 或 gate 规则来掩盖 core 正确性失败。
- 不把构建成功、Gate PASS、理论收益、单次结果、probe 或历史数据写成更高证据层的结论。
- 发现 P0 问题时停止低优先级优化，先关闭正确性和证据门禁。
- 同一问题连续两轮未关闭时，必须停止局部修补，重新执行系统级 audit。
- 不顺手重构、不增加无关功能、不生成提交信息或 PR 文案，除非用户明确要求。
- 回复使用中文，直接给结论；不输出内部推理过程，不逐条叙述常规工具调用。

## 9. 完成判定

只有同时满足以下条件，才可使用与证据层相匹配的完成措辞：

- 目标行为已实现或目标结论已有直接证据；
- 必要短验证通过；
- 高风险任务的完整生命周期、失败回滚和默认关闭路径已覆盖；
- 没有隐藏失败、相关 skip、未解释的配置差异或验证工具矛盾；
- 修改范围与任务目标一致；
- 已列出未确认事项和剩余风险；
- 需要真实 server/模型或 clean-HEAD 的结论，已经获得对应层证据。

措辞规则：

- 只有静态/单元层通过：`代码已实现并通过短验证，真实路径尚未验证`；
- dirty-tree 真实短实验通过：`已诊断验证，尚不可归档`；
- clean-HEAD 正式协议通过：`已实现并验证`；
- 根因或证据不足：`目前无法确认`；
- 不得只因 Gate PASS 写“阻塞项已关闭”或“阶段已完成”。

## 10. 示例

需要调用示例时阅读 [references/examples.md](references/examples.md)。
