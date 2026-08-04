# 安装或升级 `os-agent-task` 轻量版

本目录内容以仓库根目录为相对路径。安装包只更新两套 Skill 和使用说明，不初始化或覆盖四份工程账本。

```bash
cd /root/oscomp/llama.cpp

tar -tzf ./os-agent-task-skill.tar.gz
tar -xzf ./os-agent-task-skill.tar.gz

bash .claude/skills/os-agent-task/scripts/validate-skill.sh
git diff --check
git status --short
```

预期核心文件：

```text
.agents/skills/os-agent-task/SKILL.md
.agents/skills/os-agent-task/agents/openai.yaml
.agents/skills/os-agent-task/references/high-risk.md
.agents/skills/os-agent-task/references/project-contract.md
.agents/skills/os-agent-task/scripts/validate-skill.sh
.claude/skills/os-agent-task/（相同内容）
docs/os-agent-task-usage.md
INSTALL.md
```

validator 会同时检查 Skill 的任务字段、核心策略、文件集合、脚本语法和两套镜像一致性。已运行的 Claude Code 或其他客户端如未发现更新，重启对应会话。
