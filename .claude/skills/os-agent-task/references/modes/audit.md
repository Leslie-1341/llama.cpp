# audit

目标：只读确认真实调用链、根因、阶段价值和最小实现边界。

必须：

- 核对 HEAD、dirty、目标 diff、实际 argv/env/binary/model/topology/capability；
- 找首个真实错误，区分症状、根因和至少一个替代解释；
- 对高风险任务按 `high-risk.md` 重建状态、所有权、commit/rollback；
- 判断测试是否真实命中目标分支；
- 只将“源码可达 + 阶段内 + 真实风险”列为阻塞；
- 给出一个中等粒度 implement 节点，不展开后续阶段。

禁止修改文件、把建议写成已实现、用账本或最后一条错误代替源码证据。

完成后运行 audit Gate；clean tree `NO_CHANGES` 只说明无 diff。

输出：结论、当前门禁、根因置信度、关键调用链/不变量、阻塞与非阻塞、最小实现边界、缺失证据、唯一下一步。
