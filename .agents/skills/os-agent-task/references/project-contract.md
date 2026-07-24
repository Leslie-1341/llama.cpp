# 项目固定契约

本文件是固定规则唯一入口。各模式不得重复加载或复述这些内容。

## 事实优先级

当前源码、Git diff、真实运行结果 > 构建/二进制/模型/脚本身份 > 四份工程账本 > README/设计文档 > 历史聊天与理论推测。

缺代码、日志、commit、构建或实验依据时写“目前无法确认”，并说明缺失证据。历史 README 数值、单次运行、理论收益和旧分支结果不能自动成为当前 HEAD 结论。

## 分工与 Git

- 代理可做：源码修改、定向构建、单测、fixture、无模型短 smoke、脚本和只读审计。
- 用户执行：正式模型/server/cgroup/strace/mincore、长稳定性、正式性能矩阵、commit、push、merge、rebase、tag、发布。
- 不自动 commit/push/merge/rebase/tag；不使用 destructive Git；不覆盖无关 dirty 修改。
- 默认不生成 patch；只有跨环境交付、严格差异审查或回滚需要时使用。

## 工程节奏

一个任务只对应一个可独立验收的稳定节点。先明确阶段目标和边界，再实现；短验证后只做一次阶段级 review。没有新证据不反复打开已冻结阶段，不用后续功能掩盖当前 P0。

P0：正确性、稳定性、证据可信度和演示；P1：性能、创新和融合；P2：体验与非关键优化。

## 证据与工具

- runner 记录事实，parser 独占 verdict；缺 case、重复、顺序/身份/字段错误必须 fail-closed。
- build、exit=0、marker、Gate PASS、内部计数下降都不能单独证明正确性或性能。
- 正式性能必须具备固定基线、身份、硬件/系统/cgroup、指标定义、预热、重复、交错顺序、原始 artifact、正确性门槛和波动。
- Gate 只证明 diff-aware 短检查，不抬高证据层。

## KV 与融合边界

KV 任务重点核对 physical resident、ownership、active visibility、release/offload/restore/reuse、backing I/O 原子性、commit/rollback、fail-stop、prefetch/defer/fallback 和多会话。

不能把轻量 paged row mapping 描述成完整 PagedAttention，除非源码具备真实 block table、非连续物理页管理和对应 attention 读取。

权重–KV协同必须存在共享资源和联合决策，例如共享物理内存预算或统一 I/O 仲裁；同时打开两个独立模块不构成协同创新。
