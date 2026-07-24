# implement

目标：在根因、边界和验收已明确时完成一个稳定节点。

开始前确认真实路径和允许文件。根因不足、工作区重叠、验收矛盾、出现更高 P0 或需要扩大架构时停止并建议 audit。

规则：

- 最小修改，不顺手重构或引入下一阶段；
- 保持默认关闭、ownership、visibility、commit/rollback 和错误传播；
- 同步直接相关单测/fixture；
- 高风险按 `high-risk.md` 覆盖完整生命周期；
- 运行定向构建、单测和无模型短 smoke；正式模型/长测交给用户；
- 完成后运行 implement Gate，只读取 summary，失败时局部读取 full.log。

输出：实现状态、修改行为、生命周期变化、短验证、未验证、风险、diff stat/status、唯一下一步。静态/短测通过不得写成正式验证完成。
