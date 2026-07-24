# 模型选择

仅在 `contract` 或用户明确询问时读取。

- Flash：查找、日志、简单脚本、文档、小改动。
- Pro：常规实现、单测、parser/runner、根因明确的 review-fix。
- GLM：多文件实现、状态机、生命周期、错误传播。
- Kimi：长上下文、架构和历史 diff 审计。
- Grok：P0 根因、反例、失效场景、最终对抗 review。

默认：audit/review 用 Kimi；高风险用 Grok。implement/review-fix 用 Pro；跨模块高风险用 GLM。contract 常规用 Pro，架构任务用 Kimi。关键 P0 尽量由不同模型分别审计、实现和最终 review。
