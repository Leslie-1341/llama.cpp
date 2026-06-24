# Compute-Buffer / Warmup 阶段 G0：计划文档（peak RSS 第二瓶颈）

> 操作系统功能赛技术报告素材 · G0 计划篇
>
> **定位**：KV Lazy-Block（P1/P2）已收束 KV 侧 peak。RSS trace 定位默认 warmup 后、prompt decode 前约 491 MiB jump 来自 **compute buffer / graph allocator 首次实算 commit**，与 KV 正交。本篇为 G 阶段（compute buffer / warmup 峰值）设计路线。**只写计划，不改源码、不 build、不跑实验、不进入 G1-code。**
>
> 文档日期：2026/06/09｜分支：`kv-runtime-swap-e2-approx`
>
> 关联：[P2 结果](kv_lazy_block_stage_p2_results.md)、[P2-read](kv_lazy_block_stage_p2_read.md)、[P1 结果](kv_lazy_block_stage_p1_results.md)

---

## 1. 为什么 KV Lazy-Block 阶段可以收束

| 命题 | 状态 | 依据 |
|---|---|---|
| 未使用 KV tail 页可被 OS 回收（current RSS）| ✅ P1 验证（ctx4096 降 475.8 MiB）| [P1 §4](kv_lazy_block_stage_p1_results.md) |
| 构造期可绕过 KV tail full clear（peak）| ✅ P2 验证（clear 增量 512→32 MiB）| [P2 §5.1](kv_lazy_block_stage_p2_results.md) |
| 排除 warmup 后 P2 真实降 peak | ✅ `--no-warmup` 降 477 MiB | [P2 §6](kv_lazy_block_stage_p2_results.md) |
| KV 侧还有剩余 peak 杠杆 | ❌ 无（P1+P2 不叠加，tail 已不 commit）| [P2 §7](kv_lazy_block_stage_p2_results.md) |

**结论**：KV Lazy-Block 在 `-fa on / v_trans=false / n_stream=1` 边界下已把 KV 侧 peak 收益拿满。继续在 KV tail 上加码无法触及默认 warmup 下的 process peak。**KV 主线收束。**

---

## 2. 为什么默认 warmup 下 process peak 仍不降

RSS trace（[P2 §5.2](kv_lazy_block_stage_p2_results.md)）：

- warmup `decode-exit`（n_tokens=2）rss ≈ 8,158,652 KiB；
- prompt `decode-enter`（n_tokens=7）rss ≈ 8,649,920 KiB；
- jump ≈ 491 MiB，落在 warmup 结束后 / 首次 prompt `graph_compute` 实算；
- 已排除：`output_reserve`、入口 `graph_reserve`/`sched_reserve`、`process_ubatch` 的 apply/alloc/setinputs、`graph_compute` 后同步、logits/embd get（四点 rss 全相同）。

**推断**：peak 来自 **compute buffer 首次按 worst-case 实算 commit**。warmup 入口 `llama_set_warmup(true)`（[common/common.cpp:1371](../common/common.cpp#L1371)）可能触发 worst-case graph reserve，但 reserve 多为虚拟分配；真正 commit 发生在首次 `graph_compute` 写入。**G0 须先定量确认 commit 的确切来源（warmup 实算 vs prompt 首算）。**

---

## 3. G0 核心问题

| 问题 | 待答 |
|---|---|
| Q1 warmup 是否必须 | warmup 目的是预热/预 reserve；内存受限下能否跳过/缩小/延迟而不致首次 decode 卡顿或重 reserve |
| Q2 warmup 能否缩小 | warmup 是否可用更小 ctx / ubatch / graph，使其 reserve 的 compute buffer 不按满 ctx |
| Q3 compute buffer 是否 worst-case reserve | `sched_reserve`/`graph_reserve` 是否按 `n_ubatch`（默认 512）× `n_ctx` 预留，而非实际 n_tokens |
| Q4 是否虚拟分配后首次实算 commit | reserve 占虚拟地址、`graph_compute` 首次写入才 commit 物理页（trace 已强烈指向此） |
| Q5 能否按实际 n_tokens 分配 | graph allocator 能否按当前 ubatch 实际 token 数而非 worst-case 分配/commit |

---

## 4. 需分析的源码位置

| 层 | 位置 | 关注点 |
|---|---|---|
| 应用入口 | `tools/`（llama-completion main）| decode 调用序、是否可注入 `--no-warmup` / 小 warmup |
| common warmup | [common/common.cpp:1368-1407](../common/common.cpp#L1368) | `llama_set_warmup(true)`、warmup decode 2 token、`llama_memory_clear`、`llama_set_warmup(false)` |
| decode 路径 | [src/llama-context.cpp:1657](../src/llama-context.cpp#L1657) `decode` | `output_reserve`（[:1792](../src/llama-context.cpp#L1792)）、do/while ubatch loop |
| graph reserve | [src/llama-context.cpp:411](../src/llama-context.cpp#L411) `sched_reserve` / `graph_reserve`（[:2222](../src/llama-context.cpp#L2222)）/ [:582/:602/:617](../src/llama-context.cpp#L582) | worst-case n_tokens/n_ubatch reserve 序列 |
| process_ubatch | [src/llama-context.cpp:1249](../src/llama-context.cpp#L1249) | `sched_alloc_graph`（[:1292](../src/llama-context.cpp#L1292)）→ `graph_compute`（[:1309](../src/llama-context.cpp#L1309)）|
| `llama_set_warmup` | `src/llama-context.cpp`（待定位）| warmup 标志如何影响 graph 大小 / reserve |
| ggml 调度器/分配器 | `ggml/src/ggml-alloc.c`、`ggml-backend.cpp`（gallocr / sched）| 是否 lazy、能否 madvise、commit 时机（**改动面大，G0 仅分析不动**）|

---

## 5. 候选优化路径

| 路径 | 思路 | 修改范围 | 风险 | 影响默认行为 | 能否降 peak | 适合 demo |
|---|---|---|---|---|---|---|
| **G1 禁用/缩小 warmup** | 内存受限下跳过 warmup，或仅 warmup 而不预 reserve worst-case | 参数 / common 层 | 低（首次 decode 略慢 / 可能触发一次 reserve）| 否（开关控制）| ⚠️ 取决于 peak 是否在 warmup commit | ✅ 最适合 |
| **G2 小 graph warmup** | warmup 用小 ctx/ubatch，使其不 commit 满 compute buffer | common 层 + set_warmup 语义 | 中（warmup 失去预热满图意义，prompt 时可能重 reserve）| 否 | ⚠️ 若 prompt 仍按 worst-case 则无效 | ✅ |
| **G3 compute buffer lazy commit / madvise** | 对 compute buffer 未用区做 madvise，或延迟 commit | ggml backend / sched | 高（触 ggml 分配器，跨 backend）| 可能 | ✅ 若 commit 是瓶颈 | ❌ 改动过大 |
| **G4 按实际 n_tokens 分配** | graph allocator 按当前 ubatch 实际 token 而非 worst-case | ggml-alloc | 高（核心分配逻辑，影响 graph reuse）| 是（破坏固定图复用前提）| ✅ | ❌ 风险高 |
| **G5 复用/拆分 compute buffer** | 拆分大 compute buffer，分块按需 | ggml-alloc / sched | 高 | 可能 | ⚠️ | ❌ |

> 优先级：**G1 > G2 ≫ G3/G4/G5**。G1/G2 是参数/common 层、可回滚、零默认侵入；G3–G5 触及 ggml 分配器，改动面与跨 backend 风险大，G0 暂不进入。

---

## 6. 最小实验矩阵（设计，不执行）

| 组 | 开关 | ctx | n | 采集 | 预期 |
|---|---|---:|---:|---|---|
| baseline | 默认 warmup | 4096 | 16 | peak RSS、RSS trace、t/s | 对照 |
| P2 | `LLAMA_KV_LAZY_CLEAR=1` + 默认 warmup | 4096 | 16 | 同上 | peak 不降（warmup 覆盖）|
| P2 + no-warmup | `LLAMA_KV_LAZY_CLEAR=1` + `--no-warmup` | 4096 | 16 | 同上 | peak 降 ≈477 MiB（已验证）|
| G1 候选 | `--no-warmup` 或缩小 warmup（无 P2）| 4096 | 16 | peak RSS、首 token 延迟 | 判断 warmup 单独贡献多少 peak |
| G1+P2 | 缩小 warmup + `LLAMA_KV_LAZY_CLEAR=1` | 4096 | 16 | 同上 | 目标：默认路径也降 peak |

关键对照：**G1 候选（无 P2）的 peak vs baseline** → 量化 warmup 单独引入的 peak；若 G1 单独已大幅降 peak，则 warmup commit 是主因，参数级 demo 即可。

---

## 7. 停止条件

1. **若必须大改 ggml allocator（G3/G4/G5）才能降 peak → 暂缓**，记录为「需分配器级改造」的结论，不在比赛窗口内强行做。
2. **若仅靠参数/common 层（G1/G2）即可降低 warmup peak → 优先做参数级 demo**，最小侵入、可回滚。
3. 若 G1 分析发现 peak 实际发生在 prompt 首算（而非 warmup）→ 说明 warmup 非主因，转而评估 prompt 阶段 compute buffer，可能落入 G3+（暂缓）。

---

## 8. 下一步建议

1. **先源码分析 warmup 调用点**：`llama_set_warmup` 语义（warmup 标志是否使 graph 走 worst-case）、common warmup decode 路径（[common/common.cpp:1368](../common/common.cpp#L1368)）、它与 `sched_reserve` 的关系——确认 491 MiB 是 warmup 实算还是 prompt 首算。
2. **再做 G1 最小参数/逻辑验证**：对照 `--no-warmup` / 缩小 warmup 的 peak 与首 token 延迟，判断 warmup 能否安全跳过/缩小。
3. **暂不直接改 ggml allocator**：G3–G5 列为后续、需独立立项。

---

## 9. 是否建议进入 G1 warmup-minimize 分析

✅ **建议进入 G1（warmup-minimize 源码分析）**。理由：

- RSS trace 已把 peak 第二瓶颈定位到 warmup→prompt 之间的 compute commit，G1 是验证「warmup 是否为主因、能否参数级消除」的最小一步；
- G1 改动局限在参数 / common 层，零默认侵入、可回滚，最适合比赛 demo；
- 先做 G1 源码分析（不改码），确认 `llama_set_warmup` 是否让 warmup 走 worst-case graph，再决定是否需要 G2 小图 warmup。**暂不进入 G3–G5 的 ggml 分配器改造。**

---

## 10. 回滚

本篇仅新增文档：

```bash
rm docs/kv_compute_buffer_stage_g0_plan.md
```
