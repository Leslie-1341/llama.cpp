# 按需上下文策略

目标：只加载会改变本轮判断的信息，避免工程账本、规则、日志和重复搜索长期占满上下文。

## Level 0 — 每轮固定

1. `collect-task-context.sh` 输出的 branch、HEAD、dirty、diff stat、recent commits、文件 hash/行数；
2. `PROJECT_STATE.md`：若不超过 160 行可读全文；更长时只读 Current Stage、Blocked/In Progress、Next Gate；
3. 当前 diff 的文件列表/统计；
4. 用户目标直接涉及的源码、测试、脚本或 artifact。

不默认读取完整 `ARCHITECTURE.md`、`DECISIONS.md`、`EXPERIMENTS.md`。

## Level 1 — 按章节检索

先列标题：

```bash
bash "${SKILL_DIR}/scripts/read-ledger-sections.sh" --index ARCHITECTURE.md
```

再用任务关键词读取匹配章节：

```bash
bash "${SKILL_DIR}/scripts/read-ledger-sections.sh" ARCHITECTURE.md 'bounded release|pressure'
bash "${SKILL_DIR}/scripts/read-ledger-sections.sh" DECISIONS.md 'D-0016|dynamic target'
```

默认每个文件最多 3 节、220 行。没有命中时再扩大关键词，不直接读整本。

读取矩阵：

- `contract`：PROJECT_STATE；必要时 1–2 个 architecture/decision 章节。
- `audit/implement/review/review-fix`：PROJECT_STATE + 当前 diff + 相关源码/测试；只按需补 architecture/decision。
- `script`：只在协议/实验任务读取相关 experiment 条目。
- `memory`：四份账本都要核对，但仍先按标题/变化区域定位；`memory update` 才允许完整处理。

## Level 2 — 高风险展开

命中 destructive release、swap/backing、commit/rollback、ownership/visibility、并发/异步、server 生命周期、parser/Harness verdict 或连续两轮失败时：

1. 读取 `high-risk.md`；
2. 读取本任务直接相关的冻结契约章节；
3. 重建完整调用链和状态转移；
4. 不读取无关历史模块或完整实验账本。

## 搜索纪律

- 第一次定位只选 CodeGraph 或 `rg`；
- 找到符号后直接读定义、调用者和相关测试；
- 首轮不足才换另一种搜索，不同时做多套全仓扫描；
- 同一文件已经读过的区间不重复读取，除非 diff 发生变化；
- 大 diff 先看 `--stat`、文件名和目标函数 hunk，不先输出全 diff。

## 日志与 artifact

- 先读退出码、summary、首个错误和末尾 100–200 行；
- 用 `grep -n`/`rg` 定位 marker、FAIL、fatal、首错；
- 只展开命中前后必要窗口；
- Gate PASS 只读 marker/summary；非 PASS 按失败 check 定位 full.log；
- 不把完整构建日志、strace 或模型 stderr 持续带入上下文。
