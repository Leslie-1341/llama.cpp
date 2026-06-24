# Compute-Buffer / Warmup 阶段 G1：memory-aware warmup policy 结果

> 操作系统功能赛技术报告素材 · G1 结果篇
>
> **定位**：承接 [G0 计划](kv_compute_buffer_stage_g0_plan.md)、[P2 结果](kv_lazy_block_stage_p2_results.md)。本篇汇总 G1-code（`LLAMA_LOW_MEM_WARMUP` memory-aware warmup policy）的实测结果。**只整理结果、不改源码、不重跑实验。**
>
> 文档日期：2026/06/09｜分支：`kv-runtime-swap-e2-approx`

---

## 1. 阶段定位

P2 已证明 `LLAMA_KV_LAZY_CLEAR=1` 在 `--no-warmup` 下降 peak ≈477 MiB，但默认 warmup 引入 ~491 MiB 峰值掩盖该收益。G0 分析确认：对 Llama3 dense 模型，warmup 的 `llama_decode(2 token)` **不负责任何结构性初始化**——backend init、KV 分配、`sched_reserve`、`output_reserve` 均在 **context 构造期**完成；warmup 仅做一次性能预热（提前 commit compute buffer 物理页）。

G1 据此实现 **memory-aware warmup policy**：在内存受限场景跳过制造额外 peak 的预热实算，而**非简单关闭 warmup 功能**，默认行为零变化。

---

## 2. G1 实现机制

| 项 | 内容 |
|---|---|
| 入口 | 新增 env `LLAMA_LOW_MEM_WARMUP`，仅改 [common/common.cpp](../common/common.cpp) warmup 块前 |
| 决策点 | `if (params.warmup)` 之前读 env，置 `low_mem_warmup_skip` 标志 |
| 改动量 | +24 / −1 行，单文件 |
| 触碰范围 | **未改** ggml allocator、`sched_reserve`/`graph_reserve`、KV lazy-clear/lazy-tail、CLI 参数 |

env 取值语义：

| 取值 | 行为 | 日志 |
|---|---|---|
| 未设置 / 空 / `default` | **完全保持原 warmup 行为** | 原 `warming up the model...` |
| `off` | 跳过整个 warmup 块（等价 `--no-warmup`）| `LLAMA_LOW_MEM_WARMUP=off - ... skipping full warmup` |
| `minimal` | 与 `off` 共用跳过路径，但日志强调 memory-aware 语义 | `... structural init already done during context construction; skipping redundant warmup decode pre-touch to lower peak RSS` |
| 未知值 | **保持原行为**（不 silent 改变）| 一次 `LOG_WRN` 提示无效值 |

> **明确**：`minimal` 当前实现**不调用 `llama_decode`**，与 `off` 在内存效果上等价；差别仅在日志叙事（保留“结构性初始化已在构造期完成”的表述）。G0 已论证：在不碰 ggml allocator 的前提下，“调用 decode 但不 commit 全幅 compute buffer”无法做出有意义差异，故 `minimal` 取“不跑 decode”形态。

---

## 3. 默认行为与低内存行为

| 场景 | warmup decode | 日志 | peak 影响 |
|---|---|---|---|
| env 未设（默认）| ✅ 照常执行 | `warming up the model...` | 原行为（含 ~491 MiB 预热峰值）|
| `off` | ❌ 跳过 | low-memory 跳过 full warmup | 降 peak |
| `minimal` | ❌ 跳过 | memory-aware 表述 | 降 peak（同 off）|

**默认 warmup 没有被修改**：env 未设时 `low_mem_warmup_skip=false`，分支条件 `params.warmup && !low_mem_warmup_skip` 退化为原 `params.warmup`，逐字保留原逻辑。已实测默认运行仍打印 `warming up the model`。

---

## 4. 实验配置

| 项 | 取值 |
|---|---|
| 模型 | Meta-Llama-3-8B-Instruct-Q4_K_M.gguf |
| 后端 | CPU（`-ngl 0`），`-t 12` |
| ctx / n | `-c 4096` / `-n 16` |
| FlashAttention | `-fa on`（→ v_trans=false）|
| 其他 | `-s 42 -no-cnv -p "Hello, how are you?"` |
| RSS 口径 | `/proc/self/status` Maximum RSS（VmHWM）|

---

## 5. 结果表

`LLAMA_LOW_MEM_WARMUP=minimal LLAMA_KV_LAZY_CLEAR=1`（ctx=4096, n=16）：

| 指标 | 值 |
|---|---:|
| Maximum RSS | 8,163,684 KiB |
| lazy-clear skipped_bytes | 960 MiB |
| lazy-clear init_bytes | 64 MiB |
| lazy-clear grow_bytes | 0 |
| prompt eval | 171.10 ms / 7 tok = 40.91 tok/s |
| eval | 1056.52 ms / 15 runs = 14.20 tok/s |

日志确认：`LLAMA_LOW_MEM_WARMUP=minimal - structural init already done during context construction; skipping redundant warmup decode pre-touch to lower peak RSS`，且 `warming up the model` 出现 0 次。

---

## 6. 与 `--no-warmup` 对比

| 组（ctx=4096, n=16）| Maximum RSS (KiB) | 相对基线 |
|---|---:|---:|
| baseline + `--no-warmup` | 8,652,372 | — |
| `LLAMA_KV_LAZY_CLEAR=1` + `--no-warmup` | 8,163,896 | −488,476 KiB（≈477.0 MiB）|
| `LLAMA_LOW_MEM_WARMUP=minimal` + `LLAMA_KV_LAZY_CLEAR=1` | 8,163,684 | −488,688 KiB（≈477.2 MiB）|

- **G1 minimal 与 `--no-warmup` 的 peak 效果一致**（8,163,684 vs 8,163,896，差 ≈0.2 MiB，噪声级）——印证二者共用同一跳过路径；
- 相比 baseline + `--no-warmup`，G1+P2 降 peak **488,688 KiB ≈ 477.2 MiB**。

---

## 7. 正确性与性能影响

| 维度 | 结论 |
|---|---|
| 正确性 | ✅ 输出正常、无 NaN（warmup 不负责正确性，所有 init 在构造期完成）|
| peak RSS | ✅ 降 ≈477 MiB（配 P2）|
| 首 token 延迟 | 首个真实 prompt 略增一次一次性预热开销（线程池/算子 first-touch/compute buffer commit 挪到首 prompt）；prompt/eval 稳态 tok/s 与基线接近 |
| 默认行为 | ✅ 零变化 |

> 不声称改了 ggml allocator：peak 收益完全来自“不跑那次预热 decode”，compute buffer 的虚拟尺寸仍由构造期 `sched_reserve` 按 worst-case 定死，未触碰分配器。

---

## 8. 汇报口径

1. G1 不是“关闭 llama.cpp warmup”，而是 **memory-aware warmup policy**：默认仍 warmup；仅在显式低内存模式（env）下跳过制造额外 peak 的冗余 decode 预热实算。
2. 结构性初始化（backend/KV/sched_reserve/output_reserve）**始终在 context 构造期完成**，不依赖 warmup。
3. `minimal` 当前实现**不调用 `llama_decode`**，与 `off` 内存效果等价，差别仅在叙事日志。
4. G1+P2 在 `-fa on / v_trans=false / n_stream=1` 边界下降 peak ≈477 MiB，默认行为不变、可回滚、零 ggml 侵入。

---

## 9. 后续方向

1. **prompt 首算 compute buffer commit 仍是剩余 peak 来源**：G1 消除的是 warmup 预热那一份，首个真实 prompt 的大批 graph_compute 仍按 worst-case commit；要再降需改 ggml allocator（G3–G5），**改动面与跨 backend 风险大，暂缓、需独立立项**。
2. **G1 本身不再加码**：参数/common 层已达成设计目标。
3. **维持边界**：非 FA / 多 stream / 转置 V 路径不在 P2 收益范围内；G1 warmup policy 本身与 backend 无关，但其 peak 收益依赖 P2 的 lazy-clear。

---

## 10. 回滚

G1 代码回滚：

```bash
git checkout -- common/common.cpp
```

本文档回滚：

```bash
rm docs/kv_compute_buffer_stage_g1_results.md
```
