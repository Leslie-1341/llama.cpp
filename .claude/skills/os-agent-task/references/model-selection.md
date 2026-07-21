# 模型选择策略

`contract` 模式生成的任务词首行写：

```text
推荐模型：<model>
```

必要时再写备选模型。模型推荐不替代状态契约和验证门禁。

## 模型分工

- `claude-go-deepseek-v4-flash`：grep、日志、fixture、简单脚本、文档和低风险小改动。不得优先用于 destructive memory、并发状态机或 P0 根因。
- `claude-go-deepseek-v4-pro`：常规实现、单元测试、parser/runner 和明确根因后的普通 review-fix。
- `claude-go-glm-5.2`：多文件实现、状态机、生命周期、错误传播和需要连续构建反馈的任务。
- `claude-go-kimi-k3`：长上下文、历史 diff、工程账本、架构梳理和状态转移契约。
- `claude-go-grok-4.5`：P0 根因审计、反例/失效场景、对抗式 review 和稳定节点前最终审查；不宜同时负责同一改动的主要实现与最终确认。

## 按模式默认推荐

| 模式 | 常规 | 高风险/跨模块 |
|---|---|---|
| contract | Pro | Kimi |
| audit | Kimi | Grok |
| implement | Pro | GLM |
| review | Kimi | Grok |
| review-fix | Pro | GLM；完成后 Grok review |
| script | Flash/Pro | GLM |
| memory | Flash | Kimi |

关键 P0 推荐使用不同模型分工：Kimi/Grok 审计 → GLM 实现 → Pro 定向检查 → Grok 最终 review。

若模型工具调用不稳定，即使文本推理较强，也不用于关键文件修改。可根据额度、成功率和上下文长度调整。
