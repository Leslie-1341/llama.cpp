# 安装或升级 `os-agent-task` v1.1

本目录内容以仓库根目录为相对路径。v1.1 可直接覆盖 v1.0 的 Skill 文件，不会覆盖已有工程账本。

```bash
cd /root/oscomp/llama.cpp

tar -tzf ./os-agent-task-skill-v1.1.tar.gz | sed -n '1,40p'
tar -xzf ./os-agent-task-skill-v1.1.tar.gz

bash .agents/skills/os-agent-task/scripts/validate-skill.sh
bash .claude/skills/os-agent-task/scripts/validate-skill.sh

bash .agents/skills/os-agent-task/scripts/init-project-ledger.sh
bash .agents/skills/os-agent-task/scripts/init-project-ledger.sh check

git status --short
```

预期新增或更新：

```text
.agents/skills/os-agent-task/
.claude/skills/os-agent-task/
docs/os-agent-task-usage.md
INSTALL.md
PROJECT_STATE.md
ARCHITECTURE.md
DECISIONS.md
EXPERIMENTS.md
```

初始化脚本只创建缺失账本，不覆盖已有文件。先用 `memory check` 和一个真实 `implement` 任务验证，再决定提交。
