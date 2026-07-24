# memory

支持：`init`、`check`、`update`。

- `init`：缺失时从 templates 创建，不覆盖已有文件。
- `check`：核对账本与当前源码、diff、commit 和 artifact；冲突要指出，不自动调和。
- `update`：只有用户显式要求才修改。

更新规则：

- PROJECT_STATE 可覆盖当前状态；
- ARCHITECTURE 只写已验证结构；
- DECISIONS 追加或 supersede，不删历史；
- EXPERIMENTS 追加协议/证据索引，不复制长日志；
- dirty 单次诊断不写成 clean 正式结果；
- 不为“看起来完整”填入无法确认的 commit、数值或证据。

memory 模式可以处理四份账本，但先用标题索引和 diff 定位变化，避免无差别反复读写。输出修改文件、事实依据、冲突、验证和下一门禁。
