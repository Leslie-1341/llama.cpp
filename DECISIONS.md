# Decision Log

> 追加式记录。不得删除历史决策；变化通过新条目 supersede 旧条目。

## D-0001 — 按 Dense 与 MoE 访问模式拆分权重驻留机制

- Date: 2026-07-15（首次入账；实现早于本日期）
- Status: accepted
- Evidence commit/worktree: `a4cd67e61`；当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

Dense 每 token 访问全部 decoder layer，MoE 每 token 只访问 router 选中的少量 expert；二者的可利用局部性和 I/O 粒度不同。

**Decision**

Dense 使用 layer-level Flex ring；MoE 使用 expert-level anonymous buffer、budget/LRU 与可选 CLG prefetch。CLG 只做预测提示，kernel 前 callback 保留同步兜底。

**Alternatives rejected**

- 用单一 layer/expert cache 策略覆盖两类模型：无法同时表达 Dense 顺序全层访问和 MoE 稀疏 expert 访问。
- 让 CLG 预测替代真实 routing：预测失败会改变语义，当前源码明确保留同步 callback。

**Consequences and limits**

- 两条权重路径不会同时启用，当前也没有统一权重/KV I/O budget。
- 主要适用于 CPU 模型文件读取；GPU 路径目前无法确认。

**Evidence**

- `README.md` 2.2.2、2.6.1-2.6.3。
- `src/llama-model.cpp` 的 `use_flex` / `use_moe_buffer_pre` gate 与 CLG 初始化。
- `src/llama-flex.h`、`src/llama-moe-buffer.h`、`src/llama-window.cpp`。

## D-0002 — 实验内存机制保持 opt-in，默认路径不启用

- Date: 2026-07-15（首次入账；实现早于本日期）
- Status: accepted
- Evidence commit/worktree: `a4cd67e61`；当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

Flex/MoE streaming 与 paged KV 依赖模型形状、CPU host memory 和操作系统能力，不能默认外推到所有 llama.cpp backend。

**Decision**

通过 `LLAMA_FLEX`、`LLAMA_LAZY_MOE_BUFFER`、`LLAMA_LAZY_CLG`、`LLAMA_KV_PAGED` 等环境变量显式启用；未设置时保留普通路径。KV paged 还在构造阶段检查 layout/backend 前提，不满足则禁用并记录警告。

**Alternatives rejected**

- 默认全局启用实验路径：会改变未覆盖 backend/layout 的行为和性能。
- 仅依赖文档约束而不在源码 gate：不能防止不兼容配置进入机制路径。

**Consequences and limits**

- 用户必须显式选择配置；不同环境变量组合需要受控消融验证。
- 这些环境变量和 memory-level API 仍是竞赛原型接口，不代表稳定 upstream contract。

**Evidence**

- `docs/final_technical_report.md` 8.2、8.3。
- `src/llama-model.cpp` 权重 gate；`src/llama-kv-cache.cpp` paged/swap gate。

## D-0003 — core 提供 KV 机制，上层 driver/scheduler 提供生命周期策略

- Date: 2026-07-15（首次入账；实现早于本日期）
- Status: accepted
- Evidence commit/worktree: `a4cd67e61`；当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

llama.cpp core 没有完整 request-level scheduler；idle、resume-pending 和压力策略依赖 application/server 已知的 session 生命周期。

**Decision**

core 维护 block state、swap/madvise/restore/prefetch 与错误；example/trace driver 负责 workload、idle/resume 时序、prefetch timing 和 pressure policy。

**Alternatives rejected**

- 把 request policy 固化到 core decode 热路径：会耦合特定 workload，并扩大普通运行开销与接口影响。
- 让 driver 直接维护 swap 状态机：会复制 core correctness 状态和错误处理。

**Consequences and limits**

- 当前 trace replay 不是生产 server scheduler；后续 server 接入需提供真实 lifecycle 信号。
- core API 只表达机制，不能单独证明真实线上调度效果。

**Evidence**

- `docs/final_technical_report.md` 8.4、9.1。
- `docs/kv_trace_replay_stage12c_real_sharegpt_results.md` 4。
- `src/llama-memory.h` 与 `examples/kv-*`。

## D-0004 — backing store 固定槽位、block I/O 原子发布并传播 active 恢复错误

- Date: 2026-07-15
- Status: accepted
- Evidence commit/worktree: `2f5464270`、`9e8961f08`、`41cbe43c7`、`7d063f757`、`bb31d6a1f`、`663ec053e`；当前 HEAD `d9d2e3b80`
- Supersedes: none
- Superseded by: none

**Context**

append 式 backing file 会随重复 swap 周期增长；逐 cell 直接发布会暴露部分写入状态；active swap-in/mapping 失败若仅记录计数，graph 可能继续使用不完整 KV。

**Decision**

每个物理 cell 映射到固定 slot，容量为 `n_slots * cell_stride`；paged block 先打包并通过一次 `write_cells()` 完成 range 写入，成功后统一发布 metadata/`SWAPPED`。swap-in 先读 staging 再提交；active 必需恢复或 mapping/input 失败写入 context-local error，并在 graph compute 前失败。

**Alternatives rejected**

- 按文件尾追加：长期 swap 会无界增长。
- 每写完一个 cell 就发布：中途 I/O 失败可能留下部分可见 block。
- active 恢复失败后继续 graph：可能读取未恢复或错误映射的 KV。

**Consequences and limits**

- backing file 的逻辑容量有界，但累计读写字节与 syscall 仍随周期增长。
- block I/O 使用 grow-only staging，降低反复分配但增加一个 block 大小的 host buffer。
- 运行时正确性仍需当前 HEAD 的 unit/fault/stability 回归证明；本轮未运行。

**Evidence**

- `src/llama-kv-cache.h` 的 fixed-slot contract 与 backing stats。
- `src/llama-kv-cache.cpp` 的 `write_cells()`/`read_cells()`、全有或全无 metadata publication。
- `src/llama-memory.h`、`src/llama-context.cpp`、`src/llama-graph.cpp` 的错误传播。
- `tests/test-kv-backing-store.cpp` 与三个 `scripts/run-kv-p0-*.sh`。

## D-0005 — 当前正式 KV 对比采用固定 E0-E5 协议

- Date: 2026-07-15
- Status: accepted
- Evidence commit/worktree: `d9d2e3b80`; 当前工作树无 tracked diff
- Supersedes: none
- Superseded by: none

**Context**

历史文档混合了不同 workload、统计口径和诊断开关；当前 P0 修改需要一个绑定源码、二进制、模型、环境、运行顺序和原始产物的统一比较入口。

**Decision**

使用 `docs/kv_final_controlled_e0_e5_protocol.md` 定义 E0 baseline、E1 lazy、E2 paged gather、E3 swap、E4 madvise、E5 prefetch/defer；正式运行采用三轮交错顺序，并要求同轮输出精确一致、安全字段为零、机制字段实际触发。

**Alternatives rejected**

- 直接复用历史 README 数字：缺少当前 commit/worktree 与原始 artifacts 绑定。
- 只比较单次最好结果或只看退出码/RSS：不能排除输出错误、机制未触发和运行波动。
- 把 mincore 混入性能矩阵：诊断开销会污染受控性能比较。

**Consequences and limits**

- `RUNS=3` 默认要求 clean worktree；当前工作树尚不满足。
- E0-E5 只覆盖 `llama-kv-idle-swap-resume` 固定 workload，不等同于 server/ShareGPT/生产结论。
- 当前只有协议与工具，尚无本轮运行结果；实验状态见 E-0001。

**Evidence**

- `docs/kv_final_controlled_e0_e5_protocol.md`。
- `scripts/kv-final-controlled-e0-e5.sh`、`scripts/parse-kv-final-controlled-e0-e5.py`。

**Historical static audit addendum — protocol requirement vs parser capability before `d9d2e3b80` (2026-07-15)**

- 协议要求：每轮 E0-E5 各出现且仅出现一次；required field 缺失必须是 `UNVERIFIED` 而非 presumed pass；同轮输出精确一致、安全与 backend/I/O failure 字段为零、case-specific mechanism 实际触发后，结果才可接受。
- 当时的 parser 能力：可从现存 run 目录抽取字段、标记 per-run `PASS`/`FAIL`/`UNVERIFIED` 并生成汇总，但没有对 manifest/RUN_PLAN 做矩阵完整性、顺序和唯一性校验。
- 当时的 parser 能力：`main()` 只因 `FAIL` 返回非零，纯 `UNVERIFIED` 仍返回 0；因此协议中的“缺字段不得通过”尚未落实为进程级 fail-closed 门禁。
- 当时的 parser 能力：没有统一必要指标集合；部分缺失值可保留为 `NA`，`prefetch_failures_observed` 缺失被接受，关键性能/内存字段缺失不必然阻止成功退出。
- 当时的 parser 能力：`last_line()` 取最后一条 marker，`fields()` 用 `dict(...)` 保留重复 key 的最后值，重复 telemetry 或字段不会硬失败。
- 当时结论：D-0005 接受的是协议目标，不代表 `a4cd67e61` 的 parser 已满足该协议。E-0001 在 parser fail-closed 修复及合成负例验证前为 `planned-blocked`，不得进入正式运行。

**Addendum evidence**

- `scripts/parse-kv-final-controlled-e0-e5.py:70-76`：last-line 与 dict 字段解析。
- `scripts/parse-kv-final-controlled-e0-e5.py:392-419`：per-run correctness/result 聚合。
- `scripts/parse-kv-final-controlled-e0-e5.py:549-563`：扫描现存目录且只对 `FAIL` 返回非零。
- `docs/kv_final_controlled_e0_e5_protocol.md` 的 Execution order 与 Observable acceptance rules。

**Resolution addendum — parser fail-closed gate completed (2026-07-15)**

- 提交 `d9d2e3b80` 将 parser 最终门禁改为任一 run 非 `PASS` 即非零退出，并加入 manifest/固定 RUN_PLAN/实际 artifact 的完整性、顺序和唯一性对账。
- parser 现在要求 case 对应的 telemetry marker 和必要指标存在，拒绝重复 marker 与重复 key；artifact 结构错误返回 2，验证不通过返回 1。
- `tests/test-kv-final-controlled-e0-e5-parser.py` 的 5 个合成测试已通过，包含正常矩阵和四类负例：`UNVERIFIED`、矩阵缺失/重复、必要指标缺失、重复 marker/key。
- runner 的 `DRY_RUN=1 RUNS=1` 已通过并生成 6 个 planned runs；该结果仅证明规划与 parser dry-run 路径可用，不代表任何模型运行结论。
- 先前 static audit addendum 的四项 parser 阻塞已由 `d9d2e3b80` 解决；D-0005 协议决策保持 accepted。正式模型实验仍未运行。

**Resolution evidence**

- Commit: `d9d2e3b80acff27b3ff793003bb1f47ee212613b`。
- Parser: `scripts/parse-kv-final-controlled-e0-e5.py`，SHA256 `ea7d5702a89ea52bdec2bc333841bb0ccd9cb2444fcf544e120fd87b4e6b8f82`。
- Synthetic regression: `tests/test-kv-final-controlled-e0-e5-parser.py`，SHA256 `10f19b1d37c03c4e1878d58521e31003d74debf13cffacf50b43775628c420e7`；`Ran 5 tests ... OK`。
