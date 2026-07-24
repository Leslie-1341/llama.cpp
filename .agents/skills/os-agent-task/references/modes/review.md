# review

目标：独立判断当前 diff 是否可进入真实验证或稳定提交，不修改文件。

必须：

- 从 diff 和源码独立重建关键调用链，不复述实现摘要；
- 至少尝试一个真实可达反例；
- 检查上游配置、默认关闭、错误传播和目标分支测试；
- 高风险检查 release/offload→reuse→commit/rollback、active/owned/shared；
- 检查 parser/Harness 是否通过放宽标准吸收 core failure；
- 区分阻塞、非阻塞和证据不足；未来优化不阻塞当前阶段。

运行 review Gate，或在同一 HEAD/同一 diff 且身份完整时复用刚完成的机械 Gate artifact，并明确复用范围；对抗式审查本身不能省略。

输出先给：通过 / 有非阻塞问题 / 有阻塞问题 / 证据不足。随后给核心假设、问题、已有/缺失证据和唯一下一步。
