# KV Lazy-Block Memory 阶段 F1：设计文档（只写设计，不改源码）

> 操作系统功能赛技术报告素材 · F1 篇
>
> **定位**：承接 [F0 可行性报告](kv_lazy_block_stage_f0_feasibility.md)，把 F0 §9 的分层结论落成**两条独立、可回滚、默认零侵入**的实现设计：P1（tail-madvise 降 current RSS）与 P2（conditional-clear / clear-frontier 降 peak RSS）。本篇**只写设计**，不改源码、不跑实验、不 build、不 commit。
>
> **约束**：所有源码引用沿用 F0 已确认的文件/函数/行号；不声称任何未实测数据；推断处标注「仍需验证」。
>
> 文档日期：2026/06/09｜分支：`kv-runtime-swap-stage-d`
>
> 关联：[kv_lazy_block_stage_f0_feasibility.md](kv_lazy_block_stage_f0_feasibility.md)、[kv_runtime_swap_stage_e1_lite.md](kv_runtime_swap_stage_e1_lite.md)、[llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)

---

## 1. 当前 F0 结论摘要

F0 已对照源码确认（[F0 §1–§9](kv_lazy_block_stage_f0_feasibility.md)）：

| 项 | F0 结论 | 源码锚点 |
|---|---|---|
| CPU buffer 分配 | 每层 K/V 各一块连续 `posix_memalign(64B)` 匿名堆内存，第二维即 `kv_size`（`-c`）| [ggml.c:367](../ggml/src/ggml.c#L367)、[llama-kv-cache.cpp:215-216](../src/llama-kv-cache.cpp#L215) |
| 构造期 full clear | `ggml_backend_buffer_clear(buf,0)` 对**整块 buffer**（含 tail）做一次 `memset(0)`，逐页 commit 全部物理页 | [llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275)、[ggml-backend.cpp:2264](../ggml/src/ggml-backend.cpp#L2264) |
| RSS 增长阶段 | 锁定在 **KV 初始化（clear）阶段**，非 decode；与 `kv_size` 线性 | [baseline §4](kv_baseline_results.md#L67) |
| tail 可分离性 | tail = cell 索引 `[round_up(used_max_p1,256), kv_size)`，地址上与 `[0,n_kv)` 完全可分离 | [llama-kv-cache.cpp:1553-1558](../src/llama-kv-cache.cpp#L1553) |
| n_kv padding | `n_kv = max(min(size, max(256, PAD(used_max_p1,256))), …)`，对齐基准 **256** | [llama-kv-cache.cpp:1553-1558](../src/llama-kv-cache.cpp#L1553) |

**分层结论（F0 §9，F1 直接继承）**：
- 证明 page release 有效 → E1-lite 已侧证；
- 降 **current** RSS → **能**（tail madvise，无换入放大）→ **P1**；
- 降 **peak** RSS → 仅 madvise **不能**，必须避免构造期 full clear → **P2**；
- 改 backend 分配 → 降 current/peak 均**非必须**，仅彻底块式惰性才需要 → F2+。

---

## 2. F1 目标与非目标

### 目标

| 编号 | 目标 | 主指标 | 路径 |
|---|---|---|---|
| G1 | 让未使用 tail capacity 不**常驻**物理内存 | current RSS / smaps / statm | P1 |
| G2 | 让未使用 tail capacity 从初始化起就**不被 commit** | peak RSS（`/usr/bin/time -v`）| P2 |
| G3 | 两条路径默认关闭、env 开关、可一行回滚、关闭时与 baseline 逐 token 一致 | 正确性回归 | P1+P2 |

### 非目标（明确边界）

- ❌ 不改 backend buffer 分配（`ggml_backend_alloc_ctx_tensors_from_buft` / CPU buft alloc）——块式惰性分配是 **F2**，本篇仅文末一句带过；
- ❌ 不做 prefetch / 异步 / latency hiding；
- ❌ 不改 `get_k`/`get_v`/attention kernel/graph/mask 逻辑；
- ❌ 仅支持 `v_trans=false`（`-fa on`）、`n_stream=1`、单 seq（沿用 E1-lite 边界，[E1-lite §1](kv_runtime_swap_stage_e1_lite.md#L10)）；
- ❌ 不改 `include/llama.h`、`common`、CMake、scripts。

---

## 3. P1：tail-madvise 降 current RSS 设计

### 3.1 思路

在 n_kv 确定后，对每层 K/V 张量的 tail 区间 `[round_up(n_kv,256), kv_size)` 做绝对地址 page-aligned `madvise(MADV_DONTNEED)`。tail 不在任何读区间内（F0 §4），释放后**不会被 `ensure_resident` 换回**，因此与 E1-lite 不同——能真正压低 current RSS。

### 3.2 插桩点

`llama_kv_cache_context::apply()` 末尾，n_kv 已知处（即 E1-lite `ensure_resident` 同一区域；n_kv 来源 [llama-kv-cache.cpp:2836](../src/llama-kv-cache.cpp#L2836) `n_kv = kv->get_n_kv(sinfos[i_cur])`）。新增成员函数 `madvise_tail(n_kv)`，复用 E1-lite `madvise_swapped_range` 的**绝对地址页对齐**逻辑（[E1-lite §4](kv_runtime_swap_stage_e1_lite.md#L37)）。

### 3.3 伪代码

```cpp
// 仅当 LLAMA_KV_LAZY_TAIL=1；v_trans==false && n_stream==1
void llama_kv_cache::madvise_tail(uint32_t n_kv) {
    if (!kv_lazy_tail) return;
    const long pg = sysconf(_SC_PAGESIZE);
    const uint32_t lo_cell = GGML_PAD(n_kv, 256);     // tail 起点（>= n_kv 上界）
    if (lo_cell >= get_size()) return;                // 无 tail
    for (auto & layer : layers) {
        for (ggml_tensor * t : { layer.k, layer.v }) {
            if (!t) continue;
            const size_t row = ggml_row_size(t->type, t->ne[0]);
            uintptr_t base = (uintptr_t) t->data;
            uintptr_t lo_a = base + (size_t) lo_cell * row;   // tail 字节起点
            uintptr_t hi_a = base + (size_t) get_size() * row;// 张量末尾
            uintptr_t a_start = (lo_a + pg - 1) & ~(uintptr_t)(pg - 1); // 向上取整
            uintptr_t a_end   =  hi_a            & ~(uintptr_t)(pg - 1); // 向下取整
            tail_madvise_calls++;
            if (a_end > a_start) {
                if (madvise((void*)a_start, a_end - a_start, MADV_DONTNEED) == 0)
                    tail_madvise_bytes += (a_end - a_start);
                else
                    tail_madvise_failures++;
            }
        }
    }
}
```

> 注：n_stream==1 时 `t->ne[2]==1`，tail 是张量尾部单一连续段，无需逐 stream 处理；张量基址取 `t->data`（buffer 已 alloc）。

### 3.4 开关、统计、回滚

| 项 | 内容 |
|---|---|
| env 开关 | `LLAMA_KV_LAZY_TAIL=1`（仅当 `v_trans==false && n_stream==1` 生效；构造期读取 + 打横幅，沿用 E1-lite 样式 [E1-lite §2](kv_runtime_swap_stage_e1_lite.md#L19)）|
| 统计字段 | `tail_madvise_calls` / `tail_madvise_bytes` / `tail_madvise_failures` / `current_rss_before` / `current_rss_after`（后两者读 `/proc/self/statm` 第 2 字段 × 页大小，析构或末步采样）|
| 调用频率 | 每 decode step `apply()` 末尾调一次；tail 起点随 used 增长而上移，幂等安全 |
| 回滚 | `git checkout -- src/llama-kv-cache.cpp src/llama-kv-cache.h` |

### 3.5 P1 关键约束（沿用 F0 §5）

1. tail 起点必须 `= GGML_PAD(n_kv, 256)` 且 `>= n_kv 运行时上界`，否则下一 step n_kv 增长会把刚释放页纳入读区间 → 回填抖动；
2. 大 buffer 须走 mmap 才能真正还 OS（**仍需验证**，见 §6）；
3. P1 只降 **current RSS**，**不声称降 peak**（peak 已被构造期 clear 顶满，F0 §6）。

---

## 4. P2：conditional-clear / clear-frontier 降 peak RSS 设计

### 4.1 思路

peak RSS 的 KV 部分由构造期 [llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275) 的 full `memset(0)` 一次性顶满（F0 §3/§6）。P2 **不删除 clear**，而是引入 **clear frontier**：只清零可能落入 `[0,n_kv)` 连续读区间的**前缀**，tail block 不触碰 → 内核 lazy → tail 页直到被写才 commit → peak 不含 tail。

### 4.2 clear frontier 定义

- 构造期初始 `used == 0`，初始 n_kv 下界为 `max(256, n_pad)`（[llama-kv-cache.cpp:1553](../src/llama-kv-cache.cpp#L1553)）；
- **frontier = `GGML_PAD(初始安全前缀, 256)`**，保守可取一个固定上界 `clear_frontier_cells`（如 256 或 env 指定），只 clear `[0, clear_frontier_cells)` 的 cell 行；
- 之后 used 增长时，新进入 `[0,n_kv)` 的 cell 在写入前由 P2 增量清零其行（见 §4.4），或依赖「写整行」覆盖未初始化字节（仍需验证，见 §6）。

### 4.3 插桩点与伪代码

替换 [llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275) 的无条件 clear：

```cpp
// 原: ggml_backend_buffer_clear(buf, 0);
if (!kv_lazy_clear) {
    ggml_backend_buffer_clear(buf, 0);          // 默认：原行为，full clear
} else {
    // clear-frontier: 只清前缀，tail 不触碰
    for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t; t = ggml_get_next_tensor(ctx.get(), t)) {
        const size_t row = ggml_row_size(t->type, t->ne[0]);
        const size_t frontier_bytes = (size_t) clear_frontier_cells * row; // 每 stream
        // n_stream==1：清张量头部 frontier_bytes，tail 留给 lazy
        ggml_backend_tensor_memset(t, 0, /*offset=*/0, /*size=*/std::min(frontier_bytes, ggml_nbytes(t)));
    }
    kv_lazy_clear_frontier_bytes += ...;        // 统计
}
```

> `ggml_backend_tensor_memset` 经 `ggml_backend_cpu_buffer_memset_tensor`（[ggml-backend.cpp:2230](../ggml/src/ggml-backend.cpp#L2230)）只写指定 offset/size，不触碰 tail。**仍需验证**：该 API 对 KV 张量可直接调用、offset/size 语义符合预期。

### 4.4 frontier 推进（used 增长时）

两种候选，F1 仅设计、不定稿：

| 方案 | 做法 | 风险 |
|---|---|---|
| P2-a 静态 frontier | frontier 固定为初始 n_kv 下界（256），依赖「cpy_k/cpy_v 写整行」覆盖后续 cell 的未初始化字节 | 依赖写整行假设，**仍需验证**写是否覆盖全 row（量化/padding 字节）|
| P2-b 动态 frontier | `apply_ubatch` 写新 cell 前，对刚跨过旧 frontier 的行做增量 memset，再推进 frontier | 改动面更大，触及 `apply_ubatch`（[llama-kv-cache.cpp:1017](../src/llama-kv-cache.cpp#L1017)），偏离 §2 边界，列为 P2 次选 |

推荐 P2-a（最小侵入），但其安全性强依赖 §6 的 NaN 论证。

### 4.5 开关、统计、回滚

| 项 | 内容 |
|---|---|
| env 开关 | `LLAMA_KV_LAZY_CLEAR=1` + 可选 `LLAMA_KV_CLEAR_FRONTIER=<cells>`（默认 256）|
| 统计字段 | `kv_lazy_clear_frontier_bytes` / `kv_lazy_clear_skipped_bytes`（= 全 buffer − frontier）/ peak RSS 由外部 `/usr/bin/time -v` 采集 |
| 回滚 | `git checkout -- src/llama-kv-cache.cpp src/llama-kv-cache.h` |

---

## 5. P1 / P2 对比表

| 维度 | P1（tail-madvise）| P2（clear-frontier）|
|---|---|---|
| 主指标 | **current RSS** ↓ | **peak RSS** ↓ |
| 是否降 current RSS | ✅ 直接 | ✅（tail 从不 commit，稳态也低）|
| 是否降 peak RSS | ❌（peak 已被构造期 clear 顶满）| ✅（目标）|
| 插桩点 | `apply()` 末尾 `madvise_tail()` | 构造期 [:275](../src/llama-kv-cache.cpp#L275) clear |
| 机制 | 运行时回收已 commit 页 | 初始化期不 commit tail |
| 改动范围 | 新增 1 函数 + 4 统计字段 + env | 替换 1 处 clear + 1~2 字段 + env |
| 依赖假设 | 大 buffer 走 mmap（仍需验证）| 不 clear tail 不产生 NaN（仍需验证，§6）|
| 正确性风险 | 低（tail 在 `[0,n_kv)` 外，不被读）| 中（依赖未写行永不进入 attention 数值路径）|
| 影响默认行为 | 否（默认关）| 否（默认 full clear）|
| 与 E1-lite 区别 | 释放对象是 tail 而非 live cell → 无换入放大 | 不同机制（构造期）|

---

## 6. 正确性风险专章

### 6.1 三个核心问题

| 问题 | 当前判断 | 状态 |
|---|---|---|
| Q1：`[0,n_kv)` 连续读是否物理读取被 mask 的 cell？ | **是**，mask 仅置 -INF 屏蔽 softmax 数值，不避免物理读取（[分析文档 机制3](llama_kv_cache_analysis.md#L86)）| 已确认 |
| Q2：tail（`[n_kv, kv_size)`）是否会被 attention 读到？ | **否**（理论上）。读区间是 `[0,n_kv)`，tail 在其外。P1 释放 tail 安全 | 仍需验证 kernel 不越界读 |
| Q3：P2 不 clear tail/未写行，是否产生 NaN 传播？ | **关键风险**。若未写 cell 落入 `[0,n_kv)`（used < n_kv 的 padding 区），其未初始化字节会被物理读取 → 若为 NaN 字节模式，经 mask -INF 后理论上 `exp(-INF)=0`，但 **NaN × 0 = NaN**，可能污染 softmax | **仍需验证** |

### 6.2 Q3 的细化论证（P2 成败关键）

构造期注释自承 clear 目的是「avoid NaNs in the padding」（[:275](../src/llama-kv-cache.cpp#L275)）——这暗示 padding 行**确实会被读到**。关键区分两类未初始化区域：

1. **tail `[round_up(n_kv,256), kv_size)`**：在读区间外，P1/P2 均安全（除非 kernel 越界，Q2）；
2. **padding `[used_max_p1, n_kv)`**：在读区间**内**（n_kv 向上 pad 到 256），会被物理读取。**这是 P2 真正的风险区**——若不 clear 这段，未初始化字节可能产生 NaN。

**因此 P2 的 clear frontier 必须至少覆盖 `[0, n_kv 上界)`，只能省掉 `[n_kv 上界, kv_size)` 的纯 tail。** clear_frontier_cells 不能小于运行时 n_kv 峰值——但 n_kv 峰值在构造期未知（取决于实际生成长度）。两种安全策略：

| 策略 | 描述 | 取舍 |
|---|---|---|
| 保守 frontier | clear `[0, 预期最大 used + pad)`，留 tail | 需预估 used 上界；预估过小 → padding 落入未清区 → NaN 风险 |
| 动态推进（P2-b）| 写 cell 前增量清零跨界行 | 安全但改动面大 |

> **结论（仍需源码/实验验证）**：P2 只有在「保证 `[0, 运行时 n_kv 峰值)` 全程已清零」的前提下才安全。最稳妥是 P2-b 动态推进，或 P2-a + 严格约束 `clear_frontier_cells ≥ 实际 n_kv 峰值`。**验证方案见 §6.3**。

### 6.3 验证方案（F1 仅设计，不执行）

1. **读路径确认**：核读 `build_attn_mha`（[llama-graph.cpp:1953](../src/llama-graph.cpp#L1953)）与 CPU fattn `ggml_compute_forward_flash_attn_ext_f16_one_chunk`（[ggml-cpu/ops.cpp:8214](../ggml/src/ggml-cpu/ops.cpp#L8214)）——确认其遍历范围是否严格 `[0,n_kv)`、是否触碰 `[n_kv, kv_size)`（验证 Q2）；
2. **mask × NaN 行为**：确认被 -INF mask 的 cell 的 K/V 值是否仍参与 `vec_dot`/`mad`（若参与且为 NaN → 传播）；FA 路径下 K 在量化域点积、V 经 `to_float`（[分析文档 机制7](llama_kv_cache_analysis.md#L108)）——需确认未初始化字节能否解出 NaN；
3. **小实验（F1 之后）**：开 `LLAMA_KV_LAZY_CLEAR=1` + 极小 frontier，跑短序列，比对输出是否与 baseline 逐 token 一致 / 是否出现 NaN logits；
4. **frontier 下界扫描**：逐步降低 clear_frontier_cells，找出首次出现数值偏差的临界点，反推安全下界。

---

## 7. 实验矩阵设计（不执行）

对照 [baseline](kv_baseline_results.md) 同协议（固定 prompt、seed 42、`-t 12`、CPU-only、重复 3），新增 used << kv_size 的短序列用例：

| 组别 | 开关 | ctx (`-c`) | 生成 `-n` | 采集指标 | 预期 |
|---|---|---|---|---|---|
| baseline | 全关 | 512 / 1024 / 2048 | 128 | peak RSS、current RSS、decode t/s、输出 | 对照基线 |
| P1 | `LLAMA_KV_LAZY_TAIL=1` | 512 / 1024 / 2048 | 128 | + tail_madvise_bytes、current_rss_before/after | **current RSS ↓**，peak 不变 |
| P1-短序列 | `LLAMA_KV_LAZY_TAIL=1` | **4096 / 8192** | **32** | 同上 | current RSS 显著 ↓（tail 巨大）|
| P2 | `LLAMA_KV_LAZY_CLEAR=1` | 512 / 1024 / 2048 | 128 | + peak RSS、clear_skipped_bytes、输出一致性 | **peak RSS ↓**（若 NaN 安全）|
| P2-短序列 | `LLAMA_KV_LAZY_CLEAR=1` | **4096 / 8192** | **32** | 同上 | peak RSS 显著 ↓ |
| P1+P2 | 两者均开 | 4096 | 32 | current + peak | 两指标均 ↓ |

**预期收益量级（理论估算，非实测）**：按 Llama3-8B F16 ≈ 128KB/token（[baseline §4](kv_baseline_results.md#L78)），ctx8192、used≈32+prompt 时 tail ≈ (8192−~300)×128KB ≈ **~960 MB**——这是 P1 current / P2 peak 的理论上限收益。**实测仍需验证**（受 mmap 归还、frontier 安全下界制约）。

**正确性优先**：每组先确认输出与 baseline 逐 token 一致（P2 尤其要查 NaN logits），再读内存指标。

---

## 8. 回滚方式

```bash
# 代码（P1 / P2 落地后）
git checkout -- src/llama-kv-cache.cpp src/llama-kv-cache.h

# 本设计文档
rm docs/kv_lazy_block_stage_f1_design.md
```

两条路径默认关闭（env 不设即原行为），关闭等价于 baseline，构成零侵入回归。

---

## 9. 是否建议进入 P1-code 的判断

| 路径 | 建议 | 理由 |
|---|---|---|
| **P1（tail-madvise）** | ✅ **建议先进入 P1-code** | 风险低（tail 在读区间外，不触碰 live cell，规避 E1-lite 换入放大）、改动小（1 函数 + env）、可独立验证 current RSS；唯一前置「大 buffer 走 mmap」可在 P1-code 中用 statm/smaps 当场证实或证伪 |
| **P2（clear-frontier）** | ⚠️ **暂缓，先补 §6 验证** | peak RSS 是赛题主指标，价值更高，但 §6 的 NaN 风险（padding 落入 `[0,n_kv)` 被物理读取）必须**先读 kernel 源码 + 小实验**确认安全下界，否则可能产生错误输出。建议 P1-code 完成后，先做 §6.3 的「读路径确认 + NaN 行为」专项，再决定 P2-code |

**推荐节奏**：P1-code（验证 current RSS 可降 + 证实 mmap 归还）→ §6.3 NaN 专项验证 → 据结果决定 P2-code（clear-frontier）或退回 P2-b 动态推进。保持「可回滚、默认零侵入、先验证后扩展」一贯节奏。
