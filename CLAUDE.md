IMPORTANT: Ensure you've thoroughly reviewed the [AGENTS.md](AGENTS.md) file before beginning any work.
# OS Competition Project Instructions

## 1. 项目背景

本仓库基于 llama.cpp，用于全国大学生操作系统大赛功能赛道决赛。

- 用户主要负责 KV Cache 管理与优化。
- 队友主要负责权重管理和专家激活优化。
- 后续需完成 KV Cache、权重管理、专家调度之间的融合。
- 决赛重点：创新性、工程完成度、性能、稳定性、融合效果、演示质量、答辩可信度。
- 项目状态持续变化，任何结论必须以当前分支源码、git diff 和测试结果为准。

## 2. 事实优先级

1. 当前源码、当前 diff、实际测试/实验原始证据
2. 工程账本（PROJECT_STATE / ARCHITECTURE / DECISIONS / EXPERIMENTS）
3. README、设计文档
4. 历史对话

冲突时以更高优先级为准，并明确指出哪个账本条目陈旧。不得将计划、理论推测或历史文档描述成已实现功能。

## 3. 回答原则

- 回复使用中文，先给结论。
- 必须区分：已实现并经过验证 / 已实现但证据不足 / 尚未实现 / 仅供讨论。
- 无法确认时写"目前无法确认"并说明缺少什么证据。
- 不得编造性能收益、测试结果、代码行为或创新点；不得用理论分析代替真实实验。
- 不得隐藏失败测试、性能退化或兼容性问题。

## 4. 任务优先级

- P0：正确性、崩溃、数据损坏、内存错误、并发错误、构建失败、演示失败。
- P1：性能、创新性、模块融合、实验可信度、评委可能质疑的问题。
- P2：代码整理、体验、表达、非关键功能。

发现 P0 问题时停止低优先级优化。

## 5. Git 与安全边界

不得主动执行：git commit / push / merge / rebase / tag / release / git reset --hard / git clean -fd。
Git 提交、推送、合并和发布由用户本人完成。
修改源码前先定位真实调用链和数据流；只做范围明确、可验证、可回滚的修改。
不覆盖与当前任务无关的已有改动。

不得：
- 修改系统级网络、SSH、代理或认证配置
- 安装或升级系统软件包
- 删除仓库外文件或已有构建目录
- 杀死不属于当前任务的进程
- 修改其他用户文件
- 输出 Token、密码、密钥或认证文件内容
- 修改 /root/.claude/settings.json

## 6. 权重–KV 协同主线

KV Cache 侧重点：内存占用与碎片、分配/回收/复用策略、数据布局与访存局部性、Prefix Cache 命中率、
并发隔离、长上下文性能、TTFT/TPOT/吞吐。

权重侧：Dense Flex ring、MoE-Buffer/LRU、CLG predictor。

融合检查：内存/显存竞争、KV 调度与权重加载/专家激活的时序冲突、并发优先级与反压、prefetch 带宽争用、
独立优化有效但组合后退化的情况、错误恢复与降级路径、重复缓存或重复预测。

融合验证必须提供 baseline、KV-only、weight-only、combined 四组对照及必要消融实验。
涉及压缩、量化或近似计算时必须增加正确性/精度对照实验。

以上领域的详细审查清单见工程账本 DECISIONS.md 与 ARCHITECTURE.md。

## 7. 工作入口

所有任务通过 `/os-agent-task` 驱动，模式包括 contract / audit / implement / review / review-fix / script / memory。
具体流程、交付物和停止条件由 Skill（`.claude/skills/os-agent-task/`）及其 references 定义。

implement、review、review-fix、audit 四种模式完成后必须通过 gate 门禁：

```bash
bash scripts/os-agent/gate-runner <mode>
```

gate 输出 `OS_AGENT_GATE_RESULT verdict=PASS` 方可声明完成。

## 8. 工程账本

项目可变状态统一记录在仓库根目录四个文件中，CLAUDE.md 不再重复：

- [PROJECT_STATE.md](PROJECT_STATE.md) — 当前快照（阶段、阻塞、门禁）
- [ARCHITECTURE.md](ARCHITECTURE.md) — 稳定架构（模块职责、数据流、不变量、修改边界）
- [DECISIONS.md](DECISIONS.md) — 追加式技术决策日志
- [EXPERIMENTS.md](EXPERIMENTS.md) — 追加式实验协议与证据索引

Task 执行时按需读取对应账本。普通任务不自动修改账本。账本与源码冲突时以源码为准。
