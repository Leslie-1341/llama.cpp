# `os-agent-task` 使用说明

## 定位

该 Skill 封装项目中反复出现的七类工作，而不是封装某个 KV 阶段：

- `contract`：生成精简 CC/Codex 指令；
- `audit`：只读源码与工程资产审计；
- `implement`：最小范围实现；
- `review`：审查当前 diff；
- `review-fix`：只修阻塞问题；
- `script`：编写实验、回归或解析脚本；
- `memory`：初始化、检查或同步长期工程账本。

它不会替代网页端的架构决策、实验审查和阶段门禁，也不会自动 commit 或 push。

## v1.1 新增：四个工程账本

仓库根目录长期维护：

```text
PROJECT_STATE.md
ARCHITECTURE.md
DECISIONS.md
EXPERIMENTS.md
```

普通任务会读取这些文件，但不会自动修改。只有显式调用 `memory update` 或明确要求同步账本时才写入。

初始化：

```bash
bash .agents/skills/os-agent-task/scripts/init-project-ledger.sh
bash .claude/skills/os-agent-task/scripts/init-project-ledger.sh check
```

## 安装或从 v1.0 升级

将 v1.1 压缩包解压到仓库根目录，会覆盖 Skill 自身文件，不会覆盖四个工程账本：

```bash
cd /root/oscomp/llama.cpp
tar -xzf ./os-agent-task-skill-v1.1.tar.gz

bash .agents/skills/os-agent-task/scripts/validate-skill.sh
bash .claude/skills/os-agent-task/scripts/validate-skill.sh
bash .agents/skills/os-agent-task/scripts/init-project-ledger.sh

git status --short
```

Codex 或 Claude Code 已经运行时，如未发现更新，重启对应会话。

## 推荐输入

每次只需给：

```text
mode
目标结果
验收标准或本轮特殊限制（必要时）
```

例如 implement 保持简洁：

```text
$os-agent-task implement
修复 E0–E5 parser 假通过：UNVERIFIED、矩阵缺失/重复、必要指标缺失或重复关键字段均非零退出；保留协议和 dry-run，用小型 fixture 覆盖。
```

固定仓库规则、Git 分工、长测分工、账本读取和输出格式不再重复。

## 工程账本调用

```text
$os-agent-task memory init
```

```text
$os-agent-task memory check
核对四个账本与当前源码、分支和 diff 是否一致。
```

```text
$os-agent-task memory update
同步本轮已验证成果；未运行的正式实验不得写入 EXPERIMENTS.md。
```

## 维护

`.agents` 与 `.claude` 保存相同内容。修改其中一份后，应同步另一份并运行两个 validator：

```bash
rsync -a --delete .agents/skills/os-agent-task/ .claude/skills/os-agent-task/
bash .agents/skills/os-agent-task/scripts/validate-skill.sh
bash .claude/skills/os-agent-task/scripts/validate-skill.sh
```
