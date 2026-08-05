# 本地 Agent 工作流说明

`os-agent-task` 现在仅是可选的个人本地辅助工作流。本仓库不再跟踪或分发该 Skill，也不再包含通用 Harness。

## 共享边界

- `.claude/skills/os-agent-task/` 与 `.agents/skills/os-agent-task/` 是本地忽略目录；保留、调整或删除它们不会改变共享工程行为。
- `PROJECT_STATE.md`、`ARCHITECTURE.md`、`DECISIONS.md` 与 `EXPERIMENTS.md` 是本地忽略文件，不构成团队任务的强制读取项。
- `CLAUDE.md` 只保留全队通用规则；具体任务的范围与验证由当前任务契约和实际改动决定。
- 不应调用已移除的通用 Harness，也不应把本地 Skill 的结果描述为共享 Gate 结论。

## 本地路径

以下路径由仓库根目录的 `.gitignore` 精确忽略：

```text
.claude/skills/os-agent-task/
.agents/skills/os-agent-task/
PROJECT_STATE.md
ARCHITECTURE.md
DECISIONS.md
EXPERIMENTS.md
CLAUDE.local.md
.claude/settings.local.json
```

个人需要时可以在这些路径维护自己的工作流，但不得将本地配置、私有账本或本地验证要求写入共享源码。

## 共享改动的最低检查

本地化工作流或文档变更只做与改动直接相关的结构检查、失效引用扫描和空白错误检查：

```bash
git diff --check
git status --short
```

业务源码、业务测试和正式实验 runner 应按各自任务单独验证，不因本地 Agent 工作流迁移而重复执行。
