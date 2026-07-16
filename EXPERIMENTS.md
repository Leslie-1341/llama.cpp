# Experiment Ledger

> 追加式实验索引。只记录可追溯协议与证据入口，不复制大日志，不用未运行或失败结果支撑正式结论。

## E-0005 — clean-HEAD paged identity E2I 功能门禁

- Status: valid
- Date: 2026-07-16
- Commit/worktree: `a744830e90969a2298785cdd994901f8f448995a`；manifest 记录 clean worktree
- Runner/parser: `scripts/kv-paged-identity-e2i.sh` / `scripts/parse-kv-paged-identity-e2i.py`
- Raw artifact: `/root/oscomp/kv_logs/kv_paged_identity_e2i_smoke_20260716T135342Z`
- Evidence hashes: manifest `feee0b1f8951a35b597ce9ccbbe82dd0fe492bb67f90d897352bd80c90d51c82`；summary `ad4276ec29f62a92d0d8b323892d29a889912c6285a59d9aa1f17c6890a55e8d`；runner `18fb84b4f3cfdbf7611c65f00f38475743ac698b9316091451ddce96929cebc0`；parser `2c3459d3db7136c3edb90c106b508201b36074d8d1fe735dc2150c8ef8e2587f`

**Question and protocol**

验证 E2I continuous-view topology、E2G gather fallback、graph reuse/no-reuse，以及 SHIFT/NONIDENTITY/SWAP/RELEASE/MADVISE 的 fail-closed eligibility。固定顺序为 E0、E2、E2I、E2I_NOREUSE、SHIFT、NONIDENTITY、SWAP、RELEASE、MADVISE，各运行一次；parser 对 commit/worktree、binary/model、固定计划、环境、artifact hash、唯一 marker、机制与安全字段 fail-closed。

**Environment and workload**

Binary SHA256 `ab31ba2b...b7a8be0`；Meta-Llama-3-8B-Instruct Q4_K_M，model SHA256 `b2f95e15...b47410ce`；ctx 2048、n-predict 128、batch/ubatch 128、parallel 4、seed 1、temp 0、K/V F32、unified KV；timeout 900 s，非 dry-run。

**Correctness gate and result**

全部 run exit 0、唯一 `KV_TEST_SUMMARY result=PASS`、安全/故障字段为零；所有输出相对 E0 byte-exact，E2I variants 同时相对 E2 byte-exact。9 case 全部 PASS；E2I 为 continuous view 且 `n_reused=762`，E2I_NOREUSE 为 continuous view 且 `n_reused=0`；五个动态/非 identity case 均以预期 reject reason 回退 gather。parser exit 0。

**Supported conclusion and limits**

Stage 2 identity eligibility、continuous/gather topology、reuse/no-reuse 和主要失效边界通过该 clean-HEAD 功能门禁。单次固定 example 功能矩阵不提供性能结论，也不覆盖 GPU、multi-stream、非 F32 K/V 或生产 server。

## E-0006 — static identity E2G/E2I 三轮 controlled A/B

- Status: valid
- Date: 2026-07-16
- Commit/worktree: `a744830e90969a2298785cdd994901f8f448995a`；manifest 记录 clean worktree
- Runner/parser: `scripts/kv-paged-identity-controlled-ab.sh`、`scripts/kv-paged-identity-controlled-ab-runner.py` / `scripts/parse-kv-paged-identity-controlled-ab.py`
- Raw artifact: `/root/oscomp/kv_logs/kv_paged_identity_controlled_ab_20260716T142722Z`
- Evidence hashes: manifest `46dff8b8f42d87435e6a9bdab3b44600cdbc0cbb1d4509a6d2ad4ad8939372bf`；summary `5eb66dc2d48cc40924170ba3763ce0bd4c31e2344c7ce157fecc318f88bcbabe`；paired deltas `44c1f3066445673a9631ed2e4d80ea15baca5b3f057b310d06405a9a48a62f62`；runs `d7eb9c0a10616e4c90f955da1d5ebc861d70b7ed05a8780d97d72e626b5cb149`

**Question, baseline and variant**

比较 E2G（paged row-index + K/V `GET_ROWS` gather，identity fast path off）与 E2I（相同 paged identity 配置，fast path on、continuous K/V view）。唯一实验变量为 `LLAMA_KV_PAGED_IDENTITY_FAST_PATH=0/1`。

**Protocol and environment**

三轮成对交错：R1 G-I、R2 I-G、R3 G-I；每 case 每轮一次。固定 Llama-3-8B-Instruct Q4_K_M（SHA256 `b2f95e15...b47410ce`）、Release build、GCC 11.4、GGML CPU/OpenMP/native、12 logical CPU（Xeon Platinum 8358）、Ubuntu 22.04/Linux 5.15、约 24 GB RAM。workload 为 ctx 2048、n-predict 128、batch/ubatch 128、parallel 4、seed 1、temp 0、K/V F32、unified KV；无单独剔除的 warmup run。

**Correctness gate and metrics**

每次 exit 0；paired outputs byte-identical；唯一 `KV_TEST_SUMMARY result=PASS`；failure fields 为零；路径 telemetry 必须分别证明 E2G gather 与 E2I continuous view。指标来自同一 example telemetry：TPOT=`KV_ACTIVE_TOKEN_LATENCY avg_ms`，p95=`KV_ACTIVE_TOKEN_LATENCY p95_ms`，TPS=`KV_PERF tokens_per_second`，wall=`KV_PERF total_wall_ms`。低 TPOT/p95/wall、高 TPS 为有利；严格决策规则要求四指标三轮全部有利才标记 `FAST_PATH_FASTER`。

**Results**

- TPOT E2I-E2G：R1 `-2.490 ms`，R2 `-2.333 ms`，R3 `-1.480 ms`；3/3 有利。
- TPS E2I-E2G：R1 `+0.275151`，R2 `+0.357052`，R3 `+0.271673 token/s`；3/3 有利。
- wall E2I-E2G：R1 `-2251.650 ms`，R2 `-964.319 ms`，R3 `-734.402 ms`；3/3 有利。
- p95 E2I-E2G：R1 `-11.043 ms`，R2 `-4.744 ms`，R3 `+2.285 ms`；2/3 有利。
- parser exit 0，artifact `VALID`；因 p95 第三轮不利，严格 performance judgment 为 `MIXED`。

**Supported conclusion and limits**

该 artifact 是保留 static identity fast path 的决策级端到端证据：三个中心/总量指标在三轮中方向一致，p95 多数轮有利，且正确性门禁通过。它不是决赛正式性能结论；不能外推到更多模型、长上下文、server/continuous batching、不同硬件/backend/layout，且必须保留 p95 第三轮退化，不得改写为“四指标全面胜出”。

## E-0001 — 当前 HEAD 的 KV controlled E0-E5

- Status: planned
- Date: 2026-07-15
- Commit/worktree: 目标 commit `d9d2e3b80acff27b3ff793003bb1f47ee212613b`；当前 worktree dirty，尚不满足正式模式默认门禁
- Runner/protocol: `scripts/kv-final-controlled-e0-e5.sh`；`docs/kv_final_controlled_e0_e5_protocol.md`
- Parser: `scripts/parse-kv-final-controlled-e0-e5.py`
- Raw artifacts: 尚未生成；默认目标为 `/root/oscomp/kv_logs/kv_final_controlled_e0_e5_<UTC timestamp>/`
- Evidence hashes: protocol `6f0812883b997e2e1fcc8e0e0a2d60a982d4b9ea517a3076a5d40329de5704b0`；runner `d115ff23eced6d7aae8a9689e249362098ce852b578437c8a09fc39ddc0b12dc`；parser `ea7d5702a89ea52bdec2bc333841bb0ccd9cb2444fcf544e120fd87b4e6b8f82`；synthetic regression `10f19b1d37c03c4e1878d58521e31003d74debf13cffacf50b43775628c420e7`

**Question**

在固定 idle/resume workload 下，lazy、paged gather、swap、madvise、prefetch/defer 各自是否正确触发，并如何影响 current RSS、active token latency、TPS、wall time和 backing I/O？

**Baseline and variants**

- E0: all experimental KV mechanisms off。
- E1: lazy clear/tail。
- E2: paged row-index + in-graph gather，无 swap/reclaim。
- E3: E2 + paged idle swap，无 madvise。
- E4: E3 + madvise。
- E5: E4 + active delayed prefetch/defer。

**Environment and workload**

固定 driver `llama-kv-idle-swap-resume`；ctx 2048、n-predict 128、batch/ubatch 128、parallel 4、seed 1、temp 0、K/V f32、unified KV；模型、binary、host、compiler、cgroup 与 hashes 由 manifest 记录。当前实际环境与模型可用性尚未运行确认。

**Correctness gate**

每次 exit 0；同轮 Seq0/Seq1 文本和 SHA256 与 E0 精确一致；required safety/backend/I/O failure fields 为 0；缺字段为 `UNVERIFIED`；E1-E5 的 case-specific mechanism checks 必须实际触发。

**Metrics and calculation**

保留 per-run raw 值并报告 median/min/max 与相对同轮 E0；区分 driver RSS phase、sampled VmRSS/VmHWM、cgroup current 和 backing allocation，不跨 accounting scope 相减。性能矩阵关闭 mincore，KV resident attribution 单独诊断。

**Runs, warmup and order**

功能 smoke 为 `RUNS=1`；正式为 `RUNS=3`，三轮均包含 E0-E5 且顺序交错。warmup token 256，idle seq 2，resume pending token 96。

**Result**

正式模型实验尚未运行，目前无法确认任何 E0-E5 模型正确性或性能结论。当前源码仍包含 parser 合成测试与 dry-run 路径；本轮未重跑它们，且 `/root/oscomp/kv_logs/` 中可见的 E0-E5 dry-run artifact 绑定 `a4cd67e61` 而非目标 `d9d2e3b80`，故不将旧账本中的 d9 dry-run 描述保留为可核对运行证据。

**Framework state (源码已实现；当前 HEAD 未重跑 E0-E5 验证)**

- `d9d2e3b80` 已使任一 `UNVERIFIED`/非 `PASS` run 导致 parser 非零退出。
- manifest planned runs、固定 `RUNS=1/3` 计划与实际 `(round, run_order, case)` 现做精确对账，缺失、重复、额外或乱序均硬失败。
- common/paged 必要指标与 case-specific marker 现为显式 required set，缺失直接拒绝 artifact。
- 同 marker 多行或单行重复 key 现直接硬失败，不再采用“最后值覆盖”。
- 当前源码包含覆盖以上异常的合成回归和固定 6-case dry-run 路径；它们在本轮未重跑，不能替代当前 HEAD 的模型证据。

**Supported conclusion**

可确认协议、runner、fail-closed parser 及对应合成负例已提交。不能支持任何当前 HEAD 的模型正确性或性能结论；即使获得 parser 单独零退出码，也不能替代 build、P0 回归和正式模型运行。

**Limitations**

固定 example workload；不是 ShareGPT trace、HTTP server 或生产 continuous batching。正式模式要求 clean worktree；当前仍需先完成 `d9d2e3b80` 的 P0 build、backing-store unit、B2B、I/O-fault 与 stability 回归，再进入模型功能 smoke 和正式 `RUNS=3` 矩阵。

## E-0004 — Stage 1 E0/E2/E5 单 token 诊断框架

- Status: invalid（不能作为当前 HEAD 的有效模型实验或性能结论）
- Date: 2026-07-16
- Commit/worktree: 当前框架 commit `a9faa532be58c9d821fca77df9b6bf449db3fe7b`；核对开始时 worktree clean。既有运行 artifact 全部绑定 `3ecb212c3dc8f21c4a1d539eeda2659a54540c86`，不绑定当前 HEAD。
- Runner/parser: `scripts/kv-e0-e2-e5-single-turn-diagnose.sh`；`scripts/parse-kv-e0-e2-e5-single-turn.py`
- Framework sources: `examples/kv-idle-swap-resume/idle-swap-resume.cpp`；`src/llama-kv-cache.*`

**Question**

在不把诊断指标误写为 kernel 或端到端性能的前提下，E2 能否采集完整的 cache K/V `GET_ROWS` outer-segment 事件，E5 能否将 active-token prefetch 与逐 block swap-in validate/read/unpack/commit 阶段精确关联？

**Protocol and metric scope**

- E0、E2、E5 固定顺序；runner 记录 commit、工作树、binary/model、framework hash、环境和每个 raw artifact 的 hash，parser 对这些 identity、case 环境、输出、机制、安全字段及 telemetry 一律 fail-closed。
- E2 只在 `LLAMA_KV_E2_GET_ROWS_PROFILE=1` 启用；`segment_ending_at_get_rows_wall_us` 是 scheduler graph 中前一 callback 边界至目标 GET_ROWS 完成的 outer segment，不是 GET_ROWS kernel、CPU backend 内层、单 token 或端到端 wall time。
- E5 只在 `LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=1` 启用；逐 block 的 validate/read/unpack/commit 是 `prefetch_seq_step()` 中关联恢复的阶段差值，parser 要求与 token prefetch、全局 swap-in 和累计 I/O 对账。该值不等于完整 decode latency。
- runner 将 scope 标为 `informal_diagnostic_not_a_controlled_performance_conclusion`；即使 parser 接受，也不得产生正式性能结论。

**Current verification**

- 当前 HEAD 已运行 `PYTHONDONTWRITEBYTECODE=1 python3 tests/test-kv-e0-e2-e5-single-turn-parser.py`：11 tests，PASS。
- 当前 HEAD 已运行 `bash -n scripts/kv-e0-e2-e5-single-turn-diagnose.sh`：PASS。
- 本阶段未运行 build、runner dry-run、模型运行或 A/B；因此没有当前 HEAD 的 runtime artifact。

**Existing artifacts and status**

- `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_20260715T165518Z_104953`：非 dry-run，parser exit 1，summary `UNRESOLVED`；无效。
- `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_20260715T183236Z_116670`：非 dry-run，parser exit 1；虽然 summary 写 `PASS`，进程 exit 非零，按 fail-closed 规则无效。
- `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_20260716T024359Z_127924`：非 dry-run，parser exit 2，拒绝原因为 `E2 GET_ROWS CPU graph compute count 0 != outer event count 49344`；无效。
- `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_review_fix_20260716` 与 `/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_scope_fix_20260716`：dry-run，parser exit 0，明确未启动模型；只证明当时（旧 HEAD）的规划/解析路径，不是模型实验。

**Supported conclusion**

可确认当前源码具备默认关闭的 Stage 1 诊断结构，当前 parser fixture 和 shell 语法通过。不能支持 build 成功、模型正确性、CPU backend 内层耗时、scheduler 行为、identity fast path 收益或任何性能结论。

**Next evidence required**

先完成 paged identity fast path 审计；再以未插桩路径在当前 HEAD 采集可追溯的端到端 E0/E2/E5 A/B。P0 build/backing-store、B2B、I/O-fault 和 stability 证据仍是该 A/B 的前置正确性门槛。

## E-0002 — Stage 12-C ShareGPT-backed historical report

- Status: historical-unverified
- Date: 文档提交历史为 2026-06-23 至 2026-06-24；2026-07-15 首次入账
- Commit/worktree: 无法确认；结果文档未记录运行 commit/worktree，当前 HEAD 为 `d9d2e3b80`
- Runner/protocol: `docs/reproduce_kv_cache_optimization.md`、`examples/kv-trace-replay/`、`scripts/kv_trace_from_sharegpt.py`
- Raw artifacts: 仓库内不存在；文档未提供可在当前仓库核对的 artifact hash/index
- Report: `docs/kv_trace_replay_stage12c_real_sharegpt_results.md`，SHA256 `dc64eaa0218da2e948a748e64837ad923621592a9ba05b350886429a618203c9`

**Question**

在 ShareGPT-backed synthetic multi-session idle/resume trace 下，fast-maintenance Paged-KV 是否降低 current RSS，并控制 TPS 与 resume first-token 回退？

**Baseline and variants**

文档报告 baseline、aggressive S5、fast-maintenance V4/V5、ctx8192 和 mincore diagnostic；另含组件与 prefetch 消融。

**Environment and workload**

文档记录 Ubuntu 22.04、Linux 5.15、x86-64 8 核/23 GB、SSD、Llama-3-8B Q4_K_M、K/V f32；ShareGPT 只提供内容/多轮结构，arrival/idle/concurrency 为合成。实际机器、模型 hash、binary hash、build 与运行 commit 目前无法确认。

**Correctness gate**

文档使用 exit、session summaries、all-finished、abnormal grep 和 safety counters；仓库内无原始输出，无法重新核对字段完整性或 token/hash 对等。

**Metrics and calculation**

文档报告 current process RSS、KV resident bytes、active TPS/decode time、wall time、resume first-token，并对部分正式矩阵取 3-run median。原始解析输入与计算产物缺失。

**Runs, warmup and order**

文档称 fast-maintenance 与 ctx8192 使用 3-run median，另有单次 diagnostic；完整运行顺序和每次原始日志目前无法确认。

**Result**

历史文档报告 V5 ctx4096 RSS drop 611.883 MiB、TPS delta -3.007%，ctx8192 RSS drop 1619.195 MiB、TPS delta -1.006%，并报告 mincore resident drop。以上只作为“文档报告值”保留，不视为当前 HEAD 复现实测。

**Supported conclusion**

只能支持“仓库文档曾报告这些口径和数值”；不能支持当前 `d9d2e3b80` 的正式性能、正确性或可复现性结论。

**Limitations**

缺少 raw artifacts、hash、commit/worktree；当前分支在历史结果后又引入 fixed-slot、错误传播、原子 block I/O 和新 telemetry/protocol。

## E-0003 — 仓库内 ctx512 单次 baseline

- Status: historical-single-run
- Date: 无法确认；2026-07-15 首次入账
- Commit/worktree: 无法确认；`results/kv_baseline/run_meta.txt` 未记录
- Runner/protocol: 无 runner 或正式 protocol 入口；命令保存在 `ctx512_run1.time`
- Raw artifacts: `results/kv_baseline/ctx512_run1.log`、`ctx512_run1.time`、`baseline_summary.csv`、`run_meta.txt`
- Evidence hashes: summary `1f3b36cf7a56759506e4ae17ef28ef626d267fb07ce9e7411304d096c98c7f21`；log `3f34a99b00d95fb5e945003f959b3c3295db73b68bc004339bf1ffa9f9de96da`；time `0d58cf37b2fe11cef42896d55f680c3d779ecb20035ab9420d6921d34d3cae42`；meta `9e14dbf8cffefba7da502928ca8e9ae50a3384af45444c9cecabfad10e0800c1`

**Question**

一次 CPU ctx512 completion 的资源与吞吐观测是什么？

**Baseline and variants**

只有单个 baseline；无 variant。

**Environment and workload**

Llama-3-8B-Instruct Q4_K_M 路径、CPU、threads 12、ngl 0、ctx 512、n_predict 16、seed 42、prompt `Hello, how are you?`；OS、CPU 型号、compiler/build、model/binary hash 和 commit 缺失。

**Correctness gate**

进程 exit 0；没有 token/reference equality、异常字段或模型语义门槛。

**Metrics and calculation**

单次 max RSS 8192968 KiB、prompt TPS 42.90、decode TPS 15.37、total 1145.84 ms，来自仓库内 summary/time/log。

**Runs, warmup and order**

一次运行，程序默认 warmup；无重复、交错或波动范围。

**Result**

原始文件可核对该次命令成功和上述单次观测。

**Supported conclusion**

只能证明一次未绑定 commit 的 baseline invocation 曾成功；不能支持性能比较、回归或当前状态结论。

**Limitations**

单次运行、缺少 commit/build/完整环境、无 variant 和 correctness comparison。

## Unconfirmed Claims Not Registered as Valid Experiments

- `README.md` 中 Dense Flex、MoE-Buffer、CLG、极端内存和“KV + flex auto”组合表缺少仓库内 raw logs、hash、运行 commit/worktree 与完整协议，目前无法确认。
- `README_KV_OPT.md` 与归档 stage docs 可作为历史设计和结果入口，但不能覆盖当前源码或替代 E-0001 的当前 HEAD 受控验证。
