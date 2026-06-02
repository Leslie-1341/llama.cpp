# KV cache baseline 实验结果

> 操作系统功能赛技术报告素材 · baseline 结果篇
>
> **定位**:汇总 [scripts/run_kv_baseline.sh](../scripts/run_kv_baseline.sh) 跑出、[scripts/parse_kv_baseline.py](../scripts/parse_kv_baseline.py) 解析的一组 CPU-only baseline 数据,作为后续 [runtime KV swap 最小 demo](kv_runtime_swap_minimal_demo_plan.md) 的对照基线。实验配置与采集口径见 [kv_baseline_experiment.md](kv_baseline_experiment.md)。
>
> **约束**:本阶段【未修改源码、未实现 runtime swap、未改实验脚本】,只新增本结果文档。
>
> 数据来源:`results/kv_baseline/baseline_summary.csv`、`results/kv_baseline/run_meta.txt`
>
> 文档日期:2026/06/02

---

## 一、baseline 实验配置

| 项 | 值 |
|---|---|
| 模型路径 | `/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` |
| 量化 | Q4_K_M |
| backend | **CPU-only**(`-ngl 0`) |
| 线程数 | 12(`-t 12`) |
| context length | 512 / 1024 / 2048(`-c`) |
| 每组重复次数 | 3 |
| 生成 token 数 | 128(`-n 128`) |
| 随机种子 | 42(`-s 42`,可复现) |
| 使用工具 | `build/bin/llama-completion` |
| RSS 采集 | `/usr/bin/time -v` 的 `Maximum resident set size` |
| 固定 prompt | "Explain the concept of virtual memory in an operating system, including paging, demand loading, and page replacement, in a few clear paragraphs." |

采集指标:**RSS(峰值物理内存)、prompt eval(时间 / 速度)、decode eval(时间 / 速度)、total time**。

共 3 (ctx) × 3 (repeats) = **9 次实验**,全部成功。

---

## 二、baseline 原始结果表

> 直接来自 `baseline_summary.csv` 的 9 行。`prompt_eval_tokens` 恒为 29,`decode_eval_runs` 恒为 127(即生成 128 token 中 127 步计入 eval),各组一致。

| ctx | run | RSS (MB) | prompt eval (ms) | prompt t/s | decode eval (ms) | decode t/s | total (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512  | 1 | 8010.8 | 437.00 | 66.36 | 9844.17 | 12.90 | 10332.27 |
| 512  | 2 | 8010.6 | 431.88 | 67.15 | 9604.56 | 13.22 | 10089.12 |
| 512  | 3 | 8011.0 | 436.31 | 66.47 | 9365.09 | 13.56 |  9849.15 |
| 1024 | 1 | 8074.5 | 432.48 | 67.05 | 9420.99 | 13.48 |  9907.37 |
| 1024 | 2 | 8074.7 | 427.80 | 67.79 | 9390.66 | 13.52 |  9871.88 |
| 1024 | 3 | 8074.9 | 432.19 | 67.10 | 9403.73 | 13.51 |  9883.95 |
| 2048 | 1 | 8202.6 | 433.42 | 66.91 | 9346.39 | 13.59 |  9829.26 |
| 2048 | 2 | 8202.1 | 425.27 | 68.19 | 9324.33 | 13.62 |  9803.27 |
| 2048 | 3 | 8202.8 | 425.51 | 68.15 | 9290.95 | 13.67 |  9776.74 |

> 同一 ctx 内 3 次重复方差极小(RSS 差异 < 0.5 MB,decode t/s 差异 < 0.7),说明实验稳定、可复现。

---

## 三、按 context length 的均值表

| ctx | avg RSS (MB) | avg prompt eval (ms) | avg prompt t/s | avg decode eval (ms) | avg decode t/s | avg total (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 512  | 8010.8 | 435.06 | 66.66 | 9604.61 | 13.23 | 10090.18 |
| 1024 | 8074.7 | 430.82 | 67.31 | 9405.13 | 13.50 |  9887.73 |
| 2048 | 8202.5 | 428.07 | 67.75 | 9320.56 | 13.63 |  9803.09 |

---

## 四、关键观察

**1. RSS 随 context length 单调增长**

峰值 RSS 随 `-c` 增大而上升:8010.8 → 8074.7 → 8202.5 MB。增量分布:

- **ctx 512 → 1024:+63.9 MB ≈ 64 MB**
- **ctx 1024 → 2048:+127.8 MB ≈ 128 MB**

后者约为前者两倍,与 context 翻倍后新增 token 数翻倍(+512 vs +1024 token)一致。

**2. 增长趋势与 Llama3-8B KV cache 估算基本一致**

Llama3-8B 为 32 层、GQA 8 个 KV head、head_dim 128,故每层每 token 的 KV 维度 `n_embd_k_gqa = n_embd_v_gqa = 8 × 128 = 1024`。F16 下每 token 的 KV 字节:

```
32 层 × (K + V) × 1024 × 2 字节 = 32 × 2 × 1024 × 2 = 131072 字节 = 128 KB / token
```

- 增 512 token(512→1024)→ 512 × 128 KB = **64 MB**,实测 +63.9 MB ✓
- 增 1024 token(1024→2048)→ 1024 × 128 KB = **128 MB**,实测 +127.8 MB ✓

实测增量与按 KV cache 容量公式的理论估算几乎完全吻合,印证了 [llama_kv_cache_analysis.md](llama_kv_cache_analysis.md) "KV 在构造期按 `n_ctx` 全量预分配连续张量"的结论:**RSS 增长主要来自 KV cache 预分配,与 context length 成正比**。这正是后续 runtime swap "换出冷 KV 降低常驻物理内存"的发力点。

> 注:模型权重(Q4_K_M ≈ 4.9 GB)与运行时缓冲构成 RSS 的固定大头(~8 GB),KV cache 增量(几十~上百 MB)叠加其上。swap demo 的内存收益应聚焦于 KV cache 这部分**可变增量**,而非总 RSS 的大比例下降。

**3. decode 速度基本稳定,baseline 无明显性能异常**

decode t/s 在三个 ctx 下分别为 13.23 / 13.50 / 13.63,**随 ctx 增大略有上升而非下降**,且组内方差极小。原因:本实验各组都只生成 128 token,实际 attend 的有效 KV 长度相近(prompt 29 token + 生成),并未因 `-c` 增大而显著拉长每步 attention 计算;更大的 `-c` 只是预留更大 KV 容量。整体看 decode 速度平稳(~13 t/s),prompt 速度平稳(~67 t/s),**说明 baseline 性能正常、无异常抖动或退化**,可作为可信对照。

**4. 可作为后续 runtime KV swap demo 的对照基线**

数据稳定、可复现、增长趋势符合理论,且实验协议(固定 prompt / seed / 线程 / 重复次数)已固化。**本组 baseline 可直接作为 runtime KV swap demo 的对照基线**,用于量化 demo 的内存收益与性能回退。

---

## 五、与后续 runtime KV swap demo 的关系

runtime KV swap demo(见 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md))实现后,应在**相同矩阵**(相同 ctx / prompt / seed / 线程 / 重复次数)下采集指标,与本 baseline 逐点对比。至少需对比:

| 对比项 | baseline 对应值 | demo 期望 / 关注点 |
|---|---|---|
| **RSS / peak memory 是否下降** | 本文均值表 `avg RSS`(512→2048:8010.8→8202.5 MB) | 开启 swap 后 KV 增量部分应下降,尤其大 ctx(2048)收益最明显 |
| **decode t/s 是否下降** | `avg decode t/s`(13.23 / 13.50 / 13.63) | 同步 swap 阶段预计回退,记录回退幅度;prefetch 阶段再优化 |
| **swap-in / swap-out 次数与延迟** | baseline 无(=0) | demo 新增指标,衡量换页频率与单次/累计延迟 |
| **输出是否保持一致** | baseline 生成文本(固定 seed) | demo 应与 baseline **逐 token 一致 / 数值误差内一致**(正确性前提) |
| **KV resident bytes / swapped bytes** | baseline resident = 全部 KV,swapped = 0 | demo 衡量"换出了多少、驻留多少",直接对应内存收益 |
| **换入放大比例** | baseline 无(连续 view 全 resident) | demo 衡量 `[0,n_kv)` 实际换入字节 / 理论必需字节,判断 swap 是否被放大效应吞噬收益 |

对比要点:
1. **以 ctx 聚合均值作主对照轴**,画 RSS-vs-ctx 与 decode_t/s-vs-ctx 两条曲线,直观展示 swap 的"内存降、速度退"权衡。
2. **正确性优先**:先确认 demo 输出与 baseline 一致,再讨论内存与延迟的权衡。
3. **关闭开关等价校验**:demo 关闭 swap 时数值应与本 baseline 一致(零侵入回退),即一次对 baseline 的回归验证。

> 本 baseline 不含任何 swap 行为;它的价值在 demo 数据产出后才完整体现——没有它,demo 的"降了多少内存、退了多少速度"无从量化。

---

## 六、数据可复现性

如需重跑并复核本结果:

```bash
cd /root/oscomp/llama.cpp
bash scripts/run_kv_baseline.sh          # 默认矩阵,9 次实验
python3 scripts/parse_kv_baseline.py     # 解析 → baseline_summary.csv + 控制台汇总
```

固定 `-s 42` + 固定 prompt 保证可复现;`results/kv_baseline/run_meta.txt` 记录了本次实验的完整参数快照。

> **注**:`results/kv_baseline/` 下的 raw logs(`ctx*_run*.log` / `.time`)体积较大且为派生产物,**不纳入版本提交**;需要时由上述脚本重新生成。本文档与 CSV 汇总已固化关键结论,足以支撑后续对比。
