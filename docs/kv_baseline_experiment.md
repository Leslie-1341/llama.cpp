# KV cache baseline 实验说明

> 操作系统功能赛技术报告素材 · baseline 固化篇
>
> **定位**:在进入 [runtime KV swap 最小 demo](kv_runtime_swap_minimal_demo_plan.md) 实现之前,先用未修改的 llama.cpp 跑出一组可复现的 CPU-only baseline,作为后续 swap / offloading / prefetch demo 的对照基线。
>
> **约束**:本阶段【不修改源码、不实现 runtime swap、不改 attention 逻辑】,只新增 `scripts/` 与 `docs/` 下的实验辅助文件。
>
> 文档日期:2026/06/02

---

## 一、baseline 实验目的

赛题方向是"内存受限环境的大语言模型推理优化",核心指标是**运行时物理内存**,辅以吞吐 / 延迟。后续 runtime KV swap demo 要证明的命题是:**换出冷 KV 可降低峰值物理内存,且输出与 baseline 一致**(详见 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md) 第七节指标)。

要量化"降低了多少""回退了多少",必须先有一个**固定、可复现、未改源码**的对照基线。本 baseline 的作用:

1. 固化未修改 llama.cpp 在多个 context length 下的**峰值 RSS**(主指标,对应赛题"减少运行时物理内存")。
2. 固化 **prompt eval / decode eval 速度、tokens/s、总耗时**,作为 swap demo 性能回退的分母。
3. 提供**统一 prompt + 固定 seed + 多次重复**的实验协议,保证 baseline 与 demo 在同一口径下可比。

> 注:本 baseline 跑的是标准 llama.cpp,KV cache 走 [llama_kv_cache_analysis.md](llama_kv_cache_analysis.md) 描述的"预分配大连续张量 + cell 级元数据"路径,**无任何 swap 行为**。

---

## 二、实验环境

| 项 | 值 |
|---|---|
| 仓库路径 | `/root/oscomp/llama.cpp` |
| 分支 | `kv-runtime-swap-demo` |
| 可执行文件 | `build/bin/llama-completion`(已编译) |
| 模型 | `/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` |
| 量化 | Q4_K_M |
| 后端 | CPU-only(`-ngl 0`) |
| 线程数 | 12(`-t 12`) |
| RSS 采集 | `/usr/bin/time -v` 的 `Maximum resident set size` |

> 与 demo 边界一致性:demo 计划收敛到"单 sequence + CPU 单 backend + 同步 swap"(见 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md) 第二节)。baseline 同样用 **CPU-only、单序列、固定 prompt**,保证对照口径一致。

---

## 三、命令参数说明

baseline 调用 `llama-completion`,核心参数:

| 参数 | 含义 | baseline 取值 |
|---|---|---|
| `-m` | 模型路径 | 上述 GGUF |
| `-t` | CPU 线程数 | `12` |
| `-ngl` | offload 到 GPU 的层数 | `0`(CPU-only) |
| `-c` | context length(KV cache 容量上界) | `512 / 1024 / 2048` |
| `-n` | 生成 token 数 | `128` |
| `-s` | 随机种子(可复现) | `42` |
| `-no-cnv` | 关闭对话模式,走单轮 prompt→生成 | 固定开启 |
| `-p` | prompt 文本 | 固定 prompt(见脚本) |

**实验矩阵**:

| 维度 | 取值 | 说明 |
|---|---|---|
| context length | 512, 1024, 2048 | 主对照轴:KV cache 越大,常驻内存越高,越能体现后续 swap 收益 |
| 生成 token 数 | 128 | 固定,保证 decode 段工作量一致 |
| 后端 | `-ngl 0` | CPU-only |
| 线程 | `-t 12` | 固定 |
| prompt | 固定 | 各组一致,排除 prompt 长度/内容干扰 |
| 重复次数 | 3 | 每组重复 3 次,取均值并观察方差 |

共 3 (ctx) × 3 (repeats) = **9 次完整实验**。

> 为何 `-c` 是主对照轴:KV cache 的物理占用与 `n_ctx` 成正比(预分配大连续张量,见 [llama_kv_cache_analysis.md](llama_kv_cache_analysis.md) 生命周期阶段 2)。context 越大,峰值 RSS 越高,后续 swap demo "换出冷 KV 降内存"的空间也越大。baseline 把不同 ctx 下的 RSS 曲线固定下来,便于 demo 逐点对比。

---

## 四、采集指标

每次实验产出两个原始文件:

- `ctx<CTX>_run<R>.log` —— `llama-completion` 原始输出(stdout+stderr,含 `common_perf_print` 行)
- `ctx<CTX>_run<R>.time` —— `/usr/bin/time -v` 输出(含 `Maximum resident set size`)

解析脚本从中提取:

| 指标 | 来源 | 对应赛题 / demo 用途 |
|---|---|---|
| Maximum resident set size (KB/MB) | `.time` | **主指标**:峰值物理内存,swap demo 的对照分母 |
| prompt eval time (ms) + tokens/s | `.log` perf 行 | prompt 阶段吞吐 baseline |
| decode (eval) time (ms) + tokens/s | `.log` perf 行 | decode 阶段吞吐 baseline(同步 swap 最易在此回退) |
| total time (ms) | `.log` perf 行 | 端到端耗时 baseline |

> 这些指标对应 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md) 第七节指标表中的 "RSS / peak memory""decode latency""tokens/s"。swap demo 阶段会额外采集 swapped bytes / swap-in latency / 有效占用率等 baseline 没有的项。

---

## 五、如何运行短烟雾测试

完整 baseline 跑 9 次 × Llama3-8B,耗时较长。**先用短烟雾测试**确认脚本、模型、解析链路都通,再跑完整实验。短烟雾测试通过环境变量覆盖参数,只跑一组、各 1 次、生成很少 token:

```bash
cd /root/oscomp/llama.cpp

# 短烟雾:只跑 ctx=512,重复 1 次,生成 16 token,短 prompt
CTX_LENGTHS="512" REPEATS=1 N_PREDICT=16 PROMPT="Hello, how are you?" \
    bash scripts/run_kv_baseline.sh
```

预期:
- 终端打印计划运行 1 次,进度 `[1/1] ctx=512 run=1`,最后提示完成;
- `results/kv_baseline/` 下出现 `ctx512_run1.log`、`ctx512_run1.time`、`run_meta.txt`；
- `ctx512_run1.log` 末尾含 `common_perf_print:` 的 prompt/eval/total 行；
- `ctx512_run1.time` 含 `Maximum resident set size (kbytes):`。

随后跑解析(见第七节),确认能从烟雾日志里提取到指标。烟雾测试通过即可清掉烟雾日志再跑完整实验(或换 `RESULTS_DIR` 隔离):

```bash
rm -f results/kv_baseline/ctx512_run1.*   # 可选:清掉烟雾日志
```

---

## 六、如何运行完整 baseline

确认烟雾测试通过后,直接用默认参数跑完整矩阵:

```bash
cd /root/oscomp/llama.cpp
bash scripts/run_kv_baseline.sh
```

默认参数:`MODEL` = 上述 GGUF,`THREADS=12`,`NGL=0`,`N_PREDICT=128`,`REPEATS=3`,`CTX_LENGTHS="512 1024 2048"`,`SEED=42`,固定长 prompt。共 9 次实验,日志写入 `results/kv_baseline/`。

如需调整(例如只跑某几个 ctx、或换更长 prompt),用环境变量覆盖,例如:

```bash
CTX_LENGTHS="512 2048" REPEATS=5 bash scripts/run_kv_baseline.sh
```

脚本特性:
- `set -euo pipefail`:任一实验失败(非零退出)立即停止,不会产出半截数据;
- 每组实验独立日志文件,互不覆盖;
- 关键路径(BIN / MODEL / time)缺失时**前置检查直接退出**,不创建任何文件;
- 自动创建 `results/kv_baseline/` 并写 `run_meta.txt` 记录本次参数。

---

## 七、如何解析结果

实验跑完后,用解析脚本汇总(**不触发推理,只读日志**):

```bash
cd /root/oscomp/llama.cpp
python3 scripts/parse_kv_baseline.py
```

输出:
- `results/kv_baseline/baseline_summary.csv` —— 逐次实验一行,列含 ctx / run / RSS / prompt eval / decode eval / total time;
- 控制台打印两段汇总:逐次实验表 + 按 context length 聚合的均值表。

CSV 列说明:

| 列 | 含义 |
|---|---|
| `ctx_length` | context length(`-c`) |
| `run_id` | 该 ctx 下第几次重复 |
| `max_rss_kb` / `max_rss_mb` | 峰值 RSS(KB / MB) |
| `prompt_eval_ms` / `prompt_eval_tokens` / `prompt_eval_tps` | prompt 阶段耗时 / token 数 / 速度 |
| `decode_eval_ms` / `decode_eval_runs` / `decode_eval_tps` | decode 阶段耗时 / 步数 / 速度 |
| `total_time_ms` | 端到端总耗时 |

如用了自定义 `RESULTS_DIR`,解析时传同一个变量:

```bash
RESULTS_DIR=/path/to/results python3 scripts/parse_kv_baseline.py
```

---

## 八、后续如何与 runtime KV swap demo 对比

runtime KV swap demo 实现后(见 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md)),应在**相同矩阵**(相同 ctx / prompt / seed / 线程 / 重复次数)下采集同样指标,与本 baseline 逐点对比:

| 对比维度 | baseline | swap demo | 期望结论 |
|---|---|---|---|
| 峰值 RSS | 本 CSV `max_rss_mb` | demo 开启 swap 后的 RSS | **demo 显著更低**(赛题主结论) |
| 输出 token | baseline 生成文本 | demo 生成文本 | **逐 token 一致 / 数值误差内一致**(正确性) |
| decode tokens/s | 本 CSV `decode_eval_tps` | demo 的 decode 速度 | demo 同步 swap 阶段**可能回退**,记录回退幅度 |
| total time | 本 CSV `total_time_ms` | demo 端到端耗时 | 记录延迟代价,后续由 prefetch 阶段优化 |

对比方法建议:
1. **复用同一脚本协议**:swap demo 阶段可在 `run_kv_baseline.sh` 基础上加一个开关(如 `enable_kv_swap`)对应的脚本,保持 ctx / prompt / seed / 重复次数完全一致,确保对照公平。
2. **以 ctx 聚合均值作主对照**:用第七节"按 context length 聚合"的均值表对齐两侧,画 RSS vs ctx 曲线,直观展示 swap 在大 ctx 下的内存收益。
3. **正确性优先于性能**:先确认 demo 输出与 baseline 一致(见 demo plan 第六节正确性约束),再讨论 RSS 下降与延迟回退的权衡。
4. **关闭开关等价校验**:demo 关闭 swap 时应与本 baseline 数值一致(零侵入回退,见 demo plan 第六节约束 7),这本身就是一次对 baseline 的回归验证。

> 本 baseline 是**对照基准**,不含任何 swap 行为;它的价值在跑出 demo 数据后才完全体现——没有 baseline,demo 的"降了多少内存、回退了多少速度"就无从量化。

