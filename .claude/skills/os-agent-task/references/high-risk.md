# 高风险任务补充

只在公共接口、跨模块契约、并发/异步、ownership/visibility、destructive release/offload、backing I/O、commit/rollback 或证据协议会影响正确性时读取。

## 最小契约

只核对与本次目标直接相关的内容：

1. 状态与唯一 authority；
2. owner、shared、active visibility；
3. 合法状态转移及禁止条件；
4. prepare、apply/compute、commit、rollback 或 fail-stop 的边界；
5. 物理操作失败后的权威副本、错误传播和 cleanup；
6. release/offload 后的 reuse、重复循环与 reset；
7. 默认关闭、unsupported 与 fallback。

根因或上述必要契约不清时停止 implement，转为只读 audit。不得用 HTTP 状态、最后一行 stderr、单个 marker 或 RSS 变化直接替代首错与调用链。

## 最小验证

从真实生产入口选择能够覆盖本次状态变化的最小集合。通常优先：正常路径、目标破坏性路径、后续 reuse，以及本次实际涉及的 commit/rollback 或失败传播。

只有源码可达且会改变本次正确性结论时才增加 active/owned/shared、重复循环、默认关闭或 I/O 失败用例。禁止为了“矩阵完整”扩张无关负例。

## P0 判定

高风险不等于 P0。仍须同时具备：生产路径可达、影响正确性或核心结论、存在真实证据。缺任一项时记录为非阻塞风险或证据不足，不得阻塞当前稳定节点。
