# KV Lazy-Block Memory 阶段 P2-read：conditional-clear / clear-frontier 安全性源码分析

> 操作系统功能赛技术报告素材 · P2-read 篇
>
> **定位**：承接 [P1 结果](kv_lazy_block_stage_p1_results.md)（current RSS 已降、peak 不变）与 [F1 §4/§6](kv_lazy_block_stage_f1_design.md)。本篇**只做源码分析**，回答「构造期 `ggml_backend_buffer_clear(buf,0)` 能否改成 clear-frontier、安全下界是什么」。**不改源码、不 build、不跑实验、不进入 P2-code。**
>
> 文档日期：2026/06/09｜分支：`kv-runtime-swap-e2-approx`｜仅核 `-fa on`（`v_trans=false`）路径

---

## 1. 读到的关键源码位置

| 关注点 | 位置 | 结论 |
|---|---|---|
| 构造期 full clear | [src/llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275) `ggml_backend_buffer_clear(buf,0)`；CPU 实现 `memset(ctx,0,size)` [ggml-backend.cpp:2264](../ggml/src/ggml-backend.cpp#L2264) | 整块 buffer memset，含 `[0,n_kv)` 与 tail `[n_kv,kv_size)` |
| n_kv 计算 | [src/llama-kv-cache.cpp:1671-1685](../src/llama-kv-cache.cpp#L1671) `get_n_kv` | `n_kv = max(min(size, max(256, PAD(used_max_p1,256))))`；对齐基准 256 |
| 读视图范围 | [src/llama-kv-cache.cpp:1687-1737](../src/llama-kv-cache.cpp#L1687) `get_k`/`get_v` | dim-2 大小 = `n_kv`；view 严格覆盖 `[0,n_kv)`，**tail `[n_kv,kv_size)` 不在 view 内** |
| mask 构建 | [src/llama-kv-cache.cpp:2048-2118](../src/llama-kv-cache.cpp#L2048) `set_input_kq_mask_impl` | 遍历 `[0,n_kv)`，`is_empty(j)` / 非本序列 / 未来 token → `data=-INFINITY`（[:2115](../src/llama-kv-cache.cpp#L2115)）；有效 cell → `0.0` |
| CPU fattn 内循环 | [ggml/src/ggml-cpu/ops.cpp:8338-8347](../ggml/src/ggml-cpu/ops.cpp#L8338) | **`if (mv == -INFINITY) continue;` 在读 `k_data`/`v_data` 之前**（[:8340-8342](../ggml/src/ggml-cpu/ops.cpp#L8340)）|

---

## 2. padding / NaN 风险判断

### 2.1 三类区间的归属

- **tail `[GGML_PAD(n_kv,256), kv_size)`**：在任何 `get_k/get_v` view 之外（§1 dim-2=n_kv），graph/kernel **永不读取**。P1 madvise 它安全；P2 不 clear 它也安全。
- **padding `[used_max_p1, n_kv)`**：在 `[0,n_kv)` view **之内**，会进入 mask 与 fattn 循环。这是唯一的风险区。
- **live `[0, used_max_p1)`**：有效 cell，必须正确（本就由 cpy_k/cpy_v 写入）。

### 2.2 关键发现：`-fa on` 路径对 masked 位置「先判后读」

CPU flash-attention 内循环（[ops.cpp:8338-8347](../ggml/src/ggml-cpu/ops.cpp#L8338)）：

```c
for (ic ...) {
    const float mv = mp ? slope*FP16_TO_FP32(mp[ic]) : 0.0f;
    if (mv == -INFINITY) { continue; }              // ← 先判 mask
    ...
    const char * k_data = k->data + ic*nbk1 + ...;  // ← 后读 K
    kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);
    ...
    const char * v_data = v->data + ic*nbv1 + ...;  // ← 后读 V
}
```

padding cell 是 `is_empty(j)==true` → mask 置 `-INFINITY`（[:2062-2063,2115](../src/llama-kv-cache.cpp#L2062)）→ fattn 在 `continue` 处**跳过，不读取 k_data/v_data**。因此：

> **`-fa on`（`v_trans=false`）路径下，`[used_max_p1, n_kv)` 未初始化字节不会被物理读取，NaN 不会产生、更不会传播。** 这是「先判后读」结构的直接推论，**仍需小实验复核**（见 §6）。

### 2.3 与注释「avoid NaNs in the padding」的关系

构造期注释说 clear 是「避免 padding 的 NaN」。结合 §2.2：在 `-fa on` 的「先判后读」结构下该清零对正确性**并非必需**（masked 位置不读）。但注释的存在说明**其他路径**（非 FA、`v_trans=true`、或不做 `mv==-INFINITY` 短路的 kernel）可能**确实读取** padding 并依赖其为 0 → 这些路径下不 clear 会产生 NaN。**因此 P2 必须限定在 `-fa on` 边界，其他路径保持 full clear。仍需验证非 FA 路径是否短路。**

---

## 3. clear-frontier 安全下界

| 候选下界 | 是否安全 | 理由 |
|---|---|---|
| 256（固定最小 block）| ❌ 不安全（单独） | n_kv 会随生成增长超过 256，`[256, n_kv)` padding 落入 view；虽 `-fa on` 短路不读，但**构造期无法预知未来 n_kv**，静态只清 256 缺乏保证 |
| 当前 `n_kv` | n/a | 构造期 n_kv 尚不存在（cells 全空，used=0）|
| `GGML_PAD(used_max_p1,256)` | n/a | 构造期 used=0 |
| **运行期动态 `n_kv_seen` 高水位** | ✅ **安全且充分** | 只要保证「凡进入过 `[0,n_kv)` view 的 block 都已清零」，masked padding 即便被读也是 0，非 FA 路径也安全 |

**结论（安全下界）**：

- **静态构造期**：full clear 不可在构造期安全地缩成固定小前缀（未来 n_kv 未知）。
- **正确做法 = 动态 clear-frontier**：构造期只清初始最小 block（`max(256,n_pad)`），运行期每当 n_kv 增长跨过旧 frontier，在新 block **进入读区间之前** memset 清零，再推进 `clear_frontier`。安全下界 = **`clear_frontier ≥ 运行期出现过的最大 n_kv`**。
- 这样 tail `[clear_frontier, kv_size)` 始终未被 commit（lazy）→ peak 不含 tail，而进入过 view 的区间始终已清零 → 即使非 FA 路径读 padding 也安全。

---

## 4. 推荐 P2-code 最小方案

| 项 | 方案 |
|---|---|
| 开关 | `LLAMA_KV_LAZY_CLEAR=1`，debug-only，默认关闭 → 关闭时 [:275](../src/llama-kv-cache.cpp#L275) 原 full clear 不变，完全等价 baseline |
| 构造期 | 开启时**只清** `[0, max(256,n_pad))` 的前缀（每层 K/V），tail 不 memset（用 `ggml_backend_tensor_memset` 限定 offset/size，[ggml-backend.cpp:2230](../ggml/src/ggml-backend.cpp#L2230)）；记 `clear_frontier = max(256,n_pad)` |
| 动态推进 | 在 `apply()` 内 `get_n_kv()` 之后、graph 读 K/V 之前（即 [src/llama-kv-cache.cpp:2836](../src/llama-kv-cache.cpp#L2836) `ensure_resident`/`madvise_tail` 同区域），若 `n_kv > clear_frontier`：对 `[clear_frontier, GGML_PAD(n_kv,256))` 各层 K/V 行 memset 0，再 `clear_frontier = GGML_PAD(n_kv,256)` |
| 字段 | `bool kv_lazy_clear` + `uint32_t clear_frontier` + 统计 `lazy_clear_init_bytes` / `lazy_clear_grow_bytes` / `lazy_clear_skipped_bytes` |
| 边界 | 仅 `v_trans=false && n_stream==1`；否则一次 warning + 退回 full clear |
| 改动文件 | **只改** `src/llama-kv-cache.cpp` / `.h`（不碰 graph/kernel/ggml/CMake/include/common）|
| 与 P1 关系 | P1 `LLAMA_KV_LAZY_TAIL` 保持独立可选；P2 降 peak、P1 降 current，可同开。**注意**：P2 动态 clear 会触碰刚进入 view 的 block，与 P1 madvise tail 区间不重叠（P1 只动 `[PAD(n_kv,256),kv_size)`，P2 只动 `[0,PAD(n_kv,256))`），二者正交 |

> block-level `clear_frontier`（单个游标）即足够，无需每 block 位图——因为读区间始终是连续前缀 `[0,n_kv)`，frontier 单调右移。

---

## 5. 是否建议进入 P2-code

✅ **建议进入 P2-code（动态 clear-frontier 方案）**，理由：

1. peak RSS 是赛题主指标，P1 已证明 current 可降但 peak 被 full clear 钉死（[P1 §4.4](kv_lazy_block_stage_p1_results.md)）；clear-frontier 是唯一能让 tail 从不 commit、从而压低 peak 的最小侵入路径。
2. 源码层面风险已大幅澄清：`-fa on` 路径「先判 mask 后读 K/V」（[ops.cpp:8340](../ggml/src/ggml-cpu/ops.cpp#L8340)），masked padding 不被读 → 不 clear 进入 view 的区间在 FA 路径下也不致命；动态 frontier 进一步保证「进入过 view 的区间已清零」，覆盖非 FA 顾虑。
3. 改动小、可回滚、默认零侵入，与既有 P1 / swap 边界一致。

**前置约束**：必须限定 `-fa on`（`v_trans=false`）、`n_stream=1`；其他路径退回 full clear（§2.3）。

---

## 6. 仍需小实验验证的点

1. **NaN 复核**：开 `LLAMA_KV_LAZY_CLEAR=1`，短序列比对输出是否与 baseline 逐 token 一致、logits 无 NaN（验证 §2.2 推论）。
2. **peak RSS 实测**：`/usr/bin/time -v` 对照 baseline，确认 tail 不 commit 后 peak 是否随 ctx 下降（尤其 ctx≫used 用例）。
3. **frontier 推进无遗漏**：确认动态 memset 在「block 进入 `[0,n_kv)` view」之前完成、无竞态（CPU 单图构建顺序，应安全，仍需确认 `apply()` → graph 读取的时序）。
4. **非 FA / `v_trans=true` 短路行为**：核对非 flash-attention CPU 路径是否同样 `mv==-INFINITY` 短路；若否，确认这些路径在 P2 开启时已被边界挡住退回 full clear。
5. **`ggml_backend_tensor_memset` 适用性**：确认其对 KV 张量按 offset/size 部分清零语义正确、CPU backend 支持。

---

## 7. 回滚

本篇仅新增文档：

```bash
rm docs/kv_lazy_block_stage_p2_read.md
```
