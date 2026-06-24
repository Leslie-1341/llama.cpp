# KV Lazy-Block Memory 阶段 P1：tail-madvise current RSS 验证结果

> 操作系统功能赛技术报告素材 · P1 结果篇
>
> **定位**：承接 [F1 设计](kv_lazy_block_stage_f1_design.md) P1 路径与 [F0 可行性](kv_lazy_block_stage_f0_feasibility.md) §9 分层结论。本篇汇总 P1-code（`LLAMA_KV_LAZY_TAIL=1`）的实测结果并分析。**本篇只整理结果、不改源码、不重跑实验、不进入 P2。**
>
> 文档日期：2026/06/09｜分支：`kv-runtime-swap-e2-approx`

---

## 1. 阶段定位

P1 是 **tail-madvise 的 current RSS 验证**，**不是 peak RSS 优化**：

- 机制：每 decode step `apply()` 末尾、n_kv 已知后，对每层 K/V 张量未使用的 tail 区间 `[GGML_PAD(n_kv,256), kv_size)` 做绝对地址 page-aligned `madvise(MADV_DONTNEED)`（[F1 §3](kv_lazy_block_stage_f1_design.md)）。
- tail 在 `[0,n_kv)` 读区间之外，不被 `ensure_resident` 换回，规避了 [E1-lite](kv_runtime_swap_stage_e1_lite.md) 的换入放大。
- 按 [F0 §6](kv_lazy_block_stage_f0_feasibility.md) 的判断：**peak RSS 由构造期 `ggml_backend_buffer_clear(buf,0)` 的 full memset 钉死，P1 事后 madvise 只能压 current、压不了 peak。** P2 clear-frontier 才是 peak 路径，本轮不做。

---

## 2. 实验配置

| 项 | 值 |
|---|---|
| 模型 | `/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`（Q4_K_M）|
| backend | CPU-only（`-ngl 0`）|
| 线程 | `-t 12` |
| context length | `-c 2048` / `-c 4096` |
| 生成 token | `-n 16` |
| seed | `-s 42` |
| FlashAttention | `-fa on`（→ `v_trans=false`，满足 P1 支持条件）|
| 开关 | lazy 组 `LLAMA_KV_LAZY_TAIL=1`；baseline 组不设 |
| peak RSS 采集 | `/usr/bin/time -v` 的 `Maximum resident set size` |
| current RSS 采集 | `madvise_tail` 内读 `/proc/self/statm`（rss_before / rss_after，KiB）|

> 边界：仅 `v_trans=false`、`n_stream=1`、单 seq（沿用 F1 非目标约束）。

---

## 3. 实验结果表

| ctx | 组 | peak RSS (KiB) | decode t/s | lazy calls | lazy bytes (累计 advise) | failures | rss_before (KiB) | rss_after (KiB) | current RSS drop |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 | baseline | 8,390,068 | 13.09 | — | — | — | — | — | — |
| 2048 | lazy | 8,387,928 | 12.68 | 1088 | 3,988,520,960 (3803.75 MiB) | 0 | 8,386,188 | 8,161,276 | 224,912 KiB ≈ 219.6 MiB |
| 4096 | baseline | 8,652,256 | 12.76 | — | — | — | — | — | — |
| 4096 | lazy | 8,650,144 | 8,551,923,712 → 见下 | — | 8,551,923,712 (8155.75 MiB) | 0 | 8,648,556 | 8,161,372 | 487,184 KiB ≈ 475.8 MiB |

> 4096-lazy 关键值：peak RSS = 8,650,144 KiB；decode t/s = 12.18；lazy calls = 1088；lazy bytes = 8,551,923,712 (8155.75 MiB)；failures = 0。

### 3.1 派生对照

| 对照项 | ctx=2048 | ctx=4096 |
|---|---|---|
| peak RSS 差（lazy − baseline）| −2,140 KiB（≈ −2.1 MiB，噪声级）| −2,112 KiB（≈ −2.1 MiB，噪声级）|
| current RSS drop（rss_before − rss_after）| 224,912 KiB ≈ 219.6 MiB | 487,184 KiB ≈ 475.8 MiB |
| decode t/s（lazy vs baseline）| 12.68 vs 13.09（−3.1%）| 12.18 vs 12.76（−4.5%）|
| lazy 后稳态 rss_after | 8,161,276 KiB | 8,161,372 KiB（**两 ctx 几乎相同**）|

---

## 4. 关键观察

1. **failures = 0**：两组所有 madvise 调用均被内核接受，绝对地址页对齐逻辑正确，tail 页确被标记可回收。验证「page release 有效」（[F0 §9](kv_lazy_block_stage_f0_feasibility.md) 第 1 条）。

2. **current RSS 明显下降**：ctx=2048 降 ≈219.6 MiB，ctx=4096 降 ≈475.8 MiB。这是 P1 与 E1-lite 的本质区别——tail 不在读区间，释放后不被换回，净 current RSS 真正下降。

3. **ctx 越大下降越明显**：tail 容量 = `kv_size − used`，ctx 翻倍而 used 不变（同样 `-n 16`）使 tail 近似翻倍，current RSS drop 从 219.6 → 475.8 MiB（约 2.17×）。**两组 lazy 后稳态 rss_after 几乎相同（8,161,2xx KiB）**，印证「不论预留多大 `-c`，实际驻留被压回到约『模型权重 + 有效 KV + 运行缓冲』的同一水位」——正对准赛题「内存受限 + 大 ctx 预留」场景。

4. **peak RSS 基本不变**：lazy 与 baseline 的 peak 差仅 ≈2 MiB（噪声级，且 lazy 略低可能是运行时抖动）。这**精确符合 F0 §6 预测**：peak 在构造期 full clear 瞬间被顶满，P1 事后释放无法回收已发生的峰值。**P1 不降 peak 是设计内结果，不是缺陷。**

5. **decode t/s 小幅下降**：−3.1%（2048）/ −4.5%（4096）。来自每步 1088 次 madvise 系统调用 + tail 页被释放后若再生长触碰需重新缺页。开销可接受且随 tail 增大略升，属预期回退。

6. **lazy bytes 是累计 advise 字节，不等于唯一释放字节**：`lazy bytes`（2048: 3803.75 MiB；4096: 8155.75 MiB）是**每步重复 advise 同一片 tail 的累加**（calls=1088 = 层数 × K/V × step 数），远大于 current RSS drop（219.6 / 475.8 MiB）。真实「被释放的唯一物理内存」应以 current RSS drop 为准，`lazy bytes` 仅反映 advise 工作量、可用于衡量重复开销。

---

## 5. 结论

| 命题 | P1 结论 | 依据 |
|---|---|---|
| 未使用 tail capacity 可被释放、降低 current RSS | ✅ **成功验证**（降 219.6 / 475.8 MiB，failures=0）| §4.1–§4.3 |
| P1 能降低 peak RSS | ❌ **不能**（peak 差 ≈2 MiB 噪声级）| §4.4，符合 [F0 §6](kv_lazy_block_stage_f0_feasibility.md) |
| 要降 peak 须改构造期 full clear | ⏭️ 需 **P2 clear-frontier**（本轮未做）| [F1 §4](kv_lazy_block_stage_f1_design.md) |
| 收益与 tail 大小正相关 | ✅ ctx 越大 / used 越小，current RSS drop 越大 | §4.3 |

**一句话**：P1 证明了「未使用 KV tail capacity 能在运行时被 `madvise` 真正释放、压低 current RSS，且 ctx 越大收益越大」，但**无法降低 peak RSS**——降 peak 必须进入 P2。

---

## 6. 后续建议

1. **P1 可优化重复 madvise**：当前每步对同一片 tail 重复 advise 1088 次（`lazy bytes` 远大于唯一释放量）。可只在 tail 起点 `GGML_PAD(n_kv,256)` **变化时**才 advise（记录上次 lo_cell），减少系统调用、收回部分 decode t/s 回退。属 P1 内增量优化，非必须。

2. **P2 前必须做 padding / NaN 风险验证**：P2 clear-frontier 跳过的不仅是纯 tail，还可能触及落入 `[0,n_kv)` 的 padding `[used_max_p1, n_kv)`（会被物理读取）。进入 P2-code 前必须先按 [F1 §6.3](kv_lazy_block_stage_f1_design.md) 读 `build_attn_mha` / CPU fattn 读路径，确认未初始化字节是否经 mask 后仍产生 NaN 传播，定出 clear-frontier 安全下界。**仍需源码 + 小实验验证。**

3. **暂不改 attention kernel、不做完整 PagedAttention**：维持 F1 边界——不碰 `get_k`/`get_v`/graph/kernel，不做非连续 gather read。P2 仅条件化构造期 clear，最小侵入、可回滚。

---

## 7. 回滚

本篇仅新增文档：

```bash
rm docs/kv_lazy_block_stage_p1_results.md
```
