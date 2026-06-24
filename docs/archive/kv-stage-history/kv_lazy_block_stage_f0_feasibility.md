# KV Lazy-Block Memory 阶段 F0：可行性报告（源码分析，不改代码）

> 操作系统功能赛技术报告素材 · F0 篇
>
> **定位**：主线从 runtime swap 正确性 demo（A+B/C/D/D4 闭环、E1-lite 探针）转向「KV Lazy-Block Memory」——块式 / 惰性 / 可换出 / 可释放的 KV 内存管理。本篇只做**源码可行性分析**，回答「能否让未使用的 KV tail capacity 不常驻物理内存」，不改代码、不跑长实验。
>
> **约束**：只读源码 + 既有文档。所有结论附文件与行号；推断处标注「仍需验证」。
>
> 文档日期：2026/06/09｜分支：`kv-runtime-swap-stage-d`
>
> 关联：[llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)、[kv_baseline_results.md](kv_baseline_results.md)、[kv_runtime_swap_stage_e1_lite.md](kv_runtime_swap_stage_e1_lite.md)、[kv_runtime_swap_stage_d_summary.md](kv_runtime_swap_stage_d_summary.md)

---

## 0. 一句话结论

E1-lite 在「换出已写 cell」上失败，根因是 `[0,n_kv)` 全区间 resident + 换入放大。**F0 改换目标**：不碰任何 live cell，只针对**从未写入的 KV tail capacity**（`[round_up(used, 256), kv_size)`）。源码层面这部分页确实**从未被业务触碰**——唯一触碰它的是构造期 `ggml_backend_buffer_clear` 的一次性 full memset（[src/llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275)）。因此结论分层（详见 §9）：**能降 current RSS（tail madvise，前提是不被 n_kv 读区间覆盖）**；**降 peak RSS 必须避免构造期 full clear**；只改 madvise 而不改 clear，最多证明 page release 有效，无法压低 peak。

---

## 1. KV tensor 底层 buffer 在 CPU backend 下如何分配？

调用链（均已读源码确认）：

1. 构造函数为每个 `buft` 建一个 `no_alloc=true` 的 ggml_context（[src/llama-kv-cache.cpp:118-124](../src/llama-kv-cache.cpp#L118)），此时张量只有 meta、无数据。
2. 每层 `ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)`（K）与对应 V（[:215-216](../src/llama-kv-cache.cpp#L215)）。**第二维就是 `kv_size`（= `-c`），即 tail capacity 在此刻就被计入张量形状**。
3. 真正分配在 [:267](../src/llama-kv-cache.cpp#L267) `ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)`，CPU backend 的 `buft` 来自 [:196](../src/llama-kv-cache.cpp#L196) `ggml_backend_cpu_buffer_type()`。
4. CPU alloc 落到 `ggml_backend_cpu_buffer_type_alloc_buffer` → `ggml_aligned_malloc(size)`（[ggml/src/ggml-backend.cpp:2305-2306](../ggml/src/ggml-backend.cpp#L2305)）。
5. `ggml_aligned_malloc` 在 Linux 走 `posix_memalign(&p, 64, size)`（[ggml/src/ggml.c:367](../ggml/src/ggml.c#L367)），对齐 **64 字节**（非页对齐）。

**关键结论**：

- buffer 是 **单块 `posix_memalign` 匿名堆内存**，一层一个连续 K 张量 + 一个连续 V 张量（每个 `n_embd_*_gqa × kv_size × n_stream`）。
- 大块（Llama3-8B、ctx2048、F16 时每层 K/V 约 8MB+，总计 §baseline 与 ctx 成正比）通常被 glibc 走 `mmap` 直接分配 → **匿名页、可被 `MADV_DONTNEED` 真正还给 OS**（E1-lite `failures=0` 已侧面印证）。**仍需验证**：小 ctx 下若 size 落在 malloc arena（brk）内，`MADV_DONTNEED` 释放的页可能不还给 OS（[阶段 D 风险清单](kv_runtime_swap_stage_d_summary.md#L118) 已列此风险）。
- 对齐是 64B 不是页（4KB），所以**任何 madvise 都必须按绝对地址做页对齐裁剪**——E1-lite 已踩过这个坑并修正（[E1-lite §4](kv_runtime_swap_stage_e1_lite.md#L37)）。

---

## 2. KV buffer 构造期是否被完整 clear / memset / 预触碰？

**是，被完整 memset 一次。** [src/llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275)：

```cpp
ggml_backend_buffer_clear(buf, 0);   // 注释：initialize the buffers to avoid NaNs in the padding
```

CPU 实现 [ggml/src/ggml-backend.cpp:2262-2265](../ggml/src/ggml-backend.cpp#L2262)：

```cpp
static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);   // 整块 buffer，全 kv_size
}
```

**这是决定性事实**：`memset` 覆盖**整个 buffer**，包括 `[used, kv_size)` 的 tail capacity。memset 写入 → 内核为每一页分配物理页框并填零 → **构造结束时整块 KV（含从未使用的 tail）已全部 resident**。注释自承目的只是「避免 padding 里的 NaN」，并非业务必需地把 tail 清零。

---

## 3. baseline 中 RSS 随 ctx 增长，发生在哪个阶段？

源码层面定位为 **KV 初始化阶段（构造期 alloc + clear）**，而非首次 decode：

- alloc（`posix_memalign`）本身只保留虚拟地址，**不一定**立刻占物理页（Linux lazy）；
- 但紧接着的 [:275](../src/llama-kv-cache.cpp#L275) `ggml_backend_buffer_clear` 的 `memset(0)` **逐页写入**，强制 commit 全部物理页；
- 此处 size 与 `kv_size`（`-c`）成正比。[baseline §4](kv_baseline_results.md#L67) 实测 RSS 增量（+64MB / +128MB）与 KV 容量公式（128KB/token）**逐点吻合**，且 baseline 只生成 128 token、有效 KV 远小于 `kv_size`——若增长来自 decode 触碰，则应只随生成长度增、不随 `-c` 增。实测随 `-c` 增 → **增长锁定在构造期 clear，不是 decode**。

> 结论：**peak RSS 的 KV 部分在 context/KV 初始化阶段（`ggml_backend_buffer_clear`）就已顶满**，decode 阶段不再新增 KV 物理页（cell 只是写进已 resident 的行）。这对 §6 至关重要。

---

## 4. 能否只对未使用 tail range 做 madvise(DONTNEED) 而不碰 [0,n_kv)？

**能，地址上完全可分离**。每层 K 张量第二维是 cell 维，`row = ggml_row_size(type_k, n_embd_k_gqa)`，cell `i` 占 `[base + i*row, base+(i+1)*row)`。tail capacity = cell 索引 `[used_hi, kv_size)`，其绝对字节区间向内做页对齐即可，与 E1-lite 的 range 计算同构（[E1-lite §4](kv_runtime_swap_stage_e1_lite.md#L37)）。

**与 E1-lite 的本质差别**：E1-lite madvise 的是「已写、已换出的 live-history cell」，它们落在 `[0,n_kv)` 读区间内，下一 step 被 `ensure_resident` 立刻 fault 回来（换入放大）。**tail capacity 从未被任何 cell 占用，不在 `[0,n_kv)` 内，没有任何读路径会碰它** → madvise 后不会被换回。

**但有约束**：`n_kv = max(min(size, max(256, PAD(used_max_p1,256))), ...)`（[src/llama-kv-cache.cpp:1553-1558](../src/llama-kv-cache.cpp#L1553)）。安全 tail 必须是 `[round_up(used_max_p1, 256), kv_size)`——**对齐基准取 256（n_pad_cur），不是任意值**，否则 n_kv 增长一步就会把刚释放的页纳入读区间触发回填。只要 tail 起点 ≥ 当前及下一步可能的 n_kv 上界，就绝对安全。

---

## 5. 只释放 [round_up(n_kv,256), kv_size) 的 tail，能否降低 current RSS？

**能降 current RSS（条件成立时）**，且这是 F0 与 E1-lite 的根本区别：

- tail 页**不在任何读区间** → madvise(DONTNEED) 后内核回收物理页，**下一 step 不 fault 回来**（无换入放大）；
- 收益大小 = `(kv_size - used_hi) × per-cell-bytes × n_layer × (K+V)`。小 ctx 高占用时 tail 小、收益小；**大 ctx + 短序列**（典型「预留大 `-c` 但实际只用前缀」）时 tail 巨大，收益最大——正好对准赛题「内存受限」场景。

**前提/仍需验证**：

1. 大 buffer 须经 mmap 分配（见 §1），否则 arena 内页不还 OS；
2. tail 起点严格 ≥ n_kv 的运行时上界，且每当 used 增长、tail 缩小时需相应「收回」madvise 区间（隐式：再写到该页会自然 fault 一张零页回来，glibc 不需显式动作，但要确保不在 mask 读到脏值——tail 本就在 `[0,n_kv)` 之外，安全）；
3. 这是 **current（瞬时）RSS** 下降，**不是 peak**（见 §6）。

---

## 6. 要降低 peak RSS，是否必须避免初始 full clear？

**是，必须。** 这是本报告最重要的判断：

- `MAX resident set size`（`/usr/bin/time -v`）是**整个进程生命周期峰值**；
- §2/§3 已确认构造期 [:275](../src/llama-kv-cache.cpp#L275) 的 `memset(0)` 会把**整块 KV（含 tail）一次性 commit 成 resident**——峰值在初始化瞬间就被顶满；
- 之后再 madvise tail，只能把 current RSS 拉回来，**peak 已经发生、无法回收**（E1-lite §6.1 第 2 点同此口径：「即便某瞬间页被丢弃，峰值已被顶到 baseline」[E1-lite:79](kv_runtime_swap_stage_e1_lite.md#L79)）。

**要降 peak，唯一路径是让 tail 页从一开始就不被 commit**，即避免对 tail 做 full clear：

- 方案 A（最小侵入）：把 [:275](../src/llama-kv-cache.cpp#L275) 的「整块 clear」改成「只 clear `[0, 初始 used 上界)` 或只 clear 各 cell 写入前必要的 padding」，tail 不 memset → 内核 lazy、tail 页直到被写才 commit；
- 方案 B：clear 后立刻对 tail 做一次 madvise(DONTNEED)（构造期一次性）——peak 仍含那一瞬的 memset commit，**降不了 peak，只降稳态 current**；
- 方案 C（更彻底，改 backend）：tail capacity 根本不纳入初始 alloc/clear，按 block 惰性分配。

> 因此：**只动 madvise → 降 current、不降 peak；要降 peak → 必须改 clear（方案 A）或改 backend 分配（方案 C）**。注释「avoid NaNs in the padding」提示 clear 有正确性意图，改前需验证 padding NaN 是否真会进入 attention（mask 已置 -INF 屏蔽空洞 cell，**仍需验证 tail 未写行是否会被 kernel 读到产生 NaN 传播**）。

---

## 7. 最小代码插桩点

按「降 current」与「降 peak」分别给出，均小、可回滚、默认零侵入：

| 目标 | 插桩点 | 动作 |
|---|---|---|
| **降 current RSS** | `apply()` 末尾（紧邻 E1-lite 的 `ensure_resident` 区域）或新增 `madvise_tail()`，在 n_kv 确定后调用 | 对每层 K/V 的 `[round_up(n_kv,256), kv_size)` 绝对地址页对齐区间调 `madvise(MADV_DONTNEED)`；复用 E1-lite 的 `madvise_swapped_range` 页对齐逻辑（[E1-lite §4](kv_runtime_swap_stage_e1_lite.md#L37)） |
| **降 peak RSS（方案 A）** | 构造函数 [src/llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275) | 把无条件 `ggml_backend_buffer_clear(buf,0)` 换成「条件 clear」：env 开关下只清必要区间 / 或 clear 后立即 madvise tail（注意此法不降 peak，见 §6 方案 B） |
| **降 peak RSS（方案 C，重）** | `ggml_backend_alloc_ctx_tensors_from_buft` / CPU buft alloc | 块式惰性分配，超出 F0 边界，列为 F2+ |

最小验证插桩：复用 E1-lite 已有的 `kv_swap_madvise` 开关样式新增 `LLAMA_KV_LAZY_TAIL=1`，仅统计 `tail_madvise_bytes` + 采样 current RSS（读 `/proc/self/statm` RSS 字段），对照 baseline。

---

## 8. 需进一步读的文件 / 函数

- [src/llama-kv-cache.cpp:1548-1558](../src/llama-kv-cache.cpp#L1548) `get_n_kv` — 确认 n_kv 运行时上界、`n_pad_cur=max(n_pad,256)` 的 padding 规则（tail 起点基准）。
- [src/llama-kv-cache.cpp:1208](../src/llama-kv-cache.cpp#L1208) `cells.used_max_p1()` 及 `find_slot`（[:818](../src/llama-kv-cache.cpp#L818)）— 环形复用是否会让 used 高水位回退（影响 tail 能否变大）。**仍需验证**：环形 wrap 后 used_max_p1 是否单调。
- [ggml/src/ggml-backend.cpp:2305](../ggml/src/ggml-backend.cpp#L2305) `..._alloc_buffer` / [ggml/src/ggml.c:331](../ggml/src/ggml.c#L331) `ggml_aligned_malloc` — 确认大 size 走 mmap（可 glibc `M_MMAP_THRESHOLD` 或 `strace` 验证，超 128KB 默认 mmap，KV buffer 远超）。
- attention kernel 对 tail/padding 行的读行为（`build_attn_mha` [src/llama-graph.cpp:1953](../src/llama-graph.cpp#L1953) 及 CPU fattn [ggml/src/ggml-cpu/ops.cpp:8214](../ggml/src/ggml-cpu/ops.cpp#L8214)）— **验证不 clear tail 是否安全**（是否读到未初始化页产生 NaN）。
- `ggml_backend_buffer_clear` 调用方是否仅此一处（grep `ggml_backend_buffer_clear` 全仓，确认改它不影响别处）。

---

## 9. F0 可行性结论（分层）

| 命题 | 结论 | 依据 |
|---|---|---|
| **能证明 page release 有效** | ✅ 已由 E1-lite `failures=0` 侧证；tail madvise 同机制更干净（无换入放大） | [E1-lite §6](kv_runtime_swap_stage_e1_lite.md#L64) |
| **能降低 current（稳态）RSS** | ✅ **能**，条件：tail 起点 ≥ n_kv 上界、大 buffer 走 mmap。收益随 `kv_size - used` 增大，大 ctx 短序列最显著 | §4、§5 |
| **能降低 peak RSS** | ⚠️ **仅 madvise 不能**；**必须避免构造期 full clear（方案 A）或改 backend 惰性分配（方案 C）** | §2、§3、§6 |
| **必须改 backend buffer 分配才能做到** | 否（降 current 不必，改构造期 clear 即可降 peak 方案 A）；**彻底块式惰性（不预留 tail 虚拟+物理）才需改 backend**（方案 C，F2+） | §1、§6、§7 |

**一句话路线**：F0 先做「tail madvise」验证 current RSS 可降（最小、安全、不碰 live cell，规避 E1-lite 的换入放大）；同时确认「peak 受构造期 `ggml_backend_buffer_clear` 的 full memset 钉死」，把「条件化 clear」列为降 peak 的 F1 主攻点。

---

## 10. Codex 下一步提示词（只写 docs，不改代码）

```
任务：基于 docs/kv_lazy_block_stage_f0_feasibility.md，撰写 F1 设计文档
docs/kv_lazy_block_stage_f1_design.md。只写文档，不改任何源码，不跑实验。

读取（务必先读全）：
- docs/kv_lazy_block_stage_f0_feasibility.md（本可行性结论，尤其 §6/§9 分层）
- docs/kv_runtime_swap_stage_e1_lite.md（madvise 页对齐经验、换入放大根因）
- docs/llama_kv_cache_analysis.md（get_n_kv/n_pad、构造期分配、mask 不避免物理读）

F1 设计文档须覆盖：
1. 两条独立可回滚路径分别成文：
   (P1) tail-madvise 降 current RSS：插桩在 apply() 末尾，对每层 K/V
        [round_up(n_kv,256), kv_size) 绝对地址页对齐区间调 MADV_DONTNEED；
        env 开关 LLAMA_KV_LAZY_TAIL；统计 tail_madvise_bytes + 采样 /proc/self/statm。
   (P2) conditional-clear 降 peak RSS：构造期 src/llama-kv-cache.cpp:275 的
        ggml_backend_buffer_clear(buf,0) 改为条件化（env 开关下不 clear tail）。
2. 对每条路径给出：精确插桩行、伪代码、开关命名、统计字段、回滚命令。
3. 正确性风险专章：P2 必须论证「不 clear tail 是否导致 attention 读到未初始化页
   产生 NaN」——给出验证方案（检查 build_attn_mha 与 CPU fattn 是否读 [used,n_kv)
   之外的行；mask 是否覆盖；padding NaN 传播路径），结论标注「仍需源码/实验验证」。
4. 明确边界：仍只支持 v_trans=false(-fa on)、n_stream=1、单 seq；不改 backend 分配
   （块式惰性分配列为 F2，仅在文末作为 future work 一句带过）。
5. 实验矩阵设计（不执行）：对照 baseline 同矩阵（ctx 512/1024/2048，但增设
   used<<kv_size 的短序列用例），分别测 current RSS（P1）与 peak RSS（P2），
   并预测哪条路径降哪个指标、预期收益量级。
6. 结论沿用 F0 分层口径：能降 current / 能降 peak / 仅证 page release /
   是否必须改 backend，逐条对应到 P1、P2。

文风：与现有 docs 一致（中文、表格化、附文件:行号、推断标注「仍需验证」、
带回滚段）。不得声称已实测未验证的数字。
```

---

## 11. 回滚

本篇仅新增文档，回滚：

```bash
rm docs/kv_lazy_block_stage_f0_feasibility.md
```
