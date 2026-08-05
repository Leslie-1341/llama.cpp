# 本地 Agent 工作流

本仓库不再分发或安装 `os-agent-task` Skill，也不再包含通用 Harness。相关配置属于个人本地工作流，不是共享源码、构建依赖或发布物。

如需保留本地 Agent 工作流，只能在下列已忽略路径中维护，且不得将其作为团队任务或提交的强制前置条件：

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

共享仓库的安装、构建和运行说明请参阅 [docs/install.md](docs/install.md)。本类治理改动的最低检查为：

```bash
git diff --check
git status --short
```
