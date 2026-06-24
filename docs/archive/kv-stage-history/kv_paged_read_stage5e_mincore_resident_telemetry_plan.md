# Stage 5E-1：KV buffer mincore resident telemetry 设计与最小实现计划

> 本文档是 Stage 5E-1 的**设计计划**，不含源码修改、不 build、不跑实验、不 commit。
> 文件路径：`docs/kv_paged_read_stage5e_mincore_resident_telemetry_plan.md`
> 关联文档：
> - `docs/kv_paged_read_stage5e_memory_attribution_plan.md`（5E-0 no-source attribution）
> - `docs/kv_paged_read_stage5c_idle_swap_madvise_results.md`（5C madvise / current RSS）
> - `docs/kv_paged_read_stage5c_scale_cumulative_rss_results.md`（5C-scale-B cumulative RSS）

---

## 1. Stage 5E-1 目标

Stage 5E-0 用 buffer-size 日志 + peak RSS 差分确认了 KV 是 ctx 扩大时的主要 RSS 增量来源，并把 Stage 5C/5D 的 ~57 MiB current RSS drop 放进 attribution 框架（≈ 理论 KV 容量的 ~11%）。但仍有缺口：

```text
process RSS drop 只说明进程 RSS 降了多少；
不能精确说明 KV buffer 里实际有多少 resident pages 被 madvise 释放。
```

本阶段先做设计，回答：

```text
1. llama.cpp 中如何拿到 CPU KV buffer 的虚拟地址范围？
2. K 和 V 是否连续？是否需要分别统计？
3. unified KV 下如何统计整段 KV buffer resident pages？
4. mincore 需要哪些 page align 处理？
5. 如何在不改变 KV 状态机的情况下输出 telemetry？
6. 应该在哪些时刻采样？
```

---

## 2. 代码现状勘察（已读源码确认）

以下结论来自 `src/llama-kv-cache.cpp` / `src/llama-kv-cache.h` 实读，是设计的事实基础。

### 2.1 KV tensor / buffer 创建

[src/llama-kv-cache.cpp:586-696](src/llama-kv-cache.cpp#L586-L696)：

```text
- buft = ggml_backend_cpu_buffer_type()（CPU path）；
- 每个 KV layer 创建独立 tensor：
    k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream)
    v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream)
- buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)；
- ctxs_bufs.emplace_back(ctx, buf) 保存 (ctx, buffer) 对。
```

即 KV 由若干 `(ggml_context, ggml_backend_buffer)` 持有，每 buffer 内含多层的 K/V tensor。

### 2.2 K/V 是否连续

[src/llama-kv-cache.h:363-373](src/llama-kv-cache.h#L363-L373) 的 `kv_layer` 持有 `ggml_tensor * k, * v` 和 `k_stream / v_stream`。已确认：

```text
- K 与 V 是各自独立的 ggml_tensor，不保证在地址空间相邻；
- 同一 buffer 内多层 tensor 的相对布局由 ggml allocator 决定，
  本设计不假设跨 tensor 连续；
- 因此 mincore 必须以「单个 tensor 的地址区间」为最小统计单元，
  逐 tensor 累加，而不是对整段 buffer 一次 mincore。
```

### 2.3 已有的 host pointer 获取方式（关键复用点）

[src/llama-kv-cache.cpp:2134-2225](src/llama-kv-cache.cpp#L2134-L2225) 的 `paged_madvise_block` 已经在用与本设计**完全相同**的指针来源：

```text
ggml_tensor * k = layer.k_stream[0];
ggml_tensor * v = layer.v_stream[0];
char * base = (char *) t->data;            // host 基址
row = ggml_row_size(t->type, hparams.n_embd_k_gqa(il));
区间 = [base + lo_cell*row, base + hi_cell*row)
page-align: a_start = (lo_a + pg-1) & ~(pg-1); a_end = hi_a & ~(pg-1)
madvise((void*)a_start, len, MADV_DONTNEED)
```

**结论：mincore telemetry 可以直接复用 `t->data` + `ggml_row_size` + 同款 page-align，无需新增 buffer 解析逻辑。** mincore 与 madvise 走同一套地址推导，二者口径天然可比。

### 2.4 已有 RSS / telemetry 基础设施

```text
- get_current_rss_kb()  -> /proc/self/statm    [src/llama-kv-cache.cpp:2913]
- get_peak_rss_kb()     -> /proc/self/status VmHWM [src/llama-kv-cache.cpp:2938]
- paged_swap_rss_* 累计字段与打印在 paged_log_stats()
  [src/llama-kv-cache.cpp:2400-2517]
- madvise gate: paged_madvise_block 已被 #if defined(__unix__) 包裹。
```

mincore telemetry 字段应紧贴 `paged_swap_rss_*` 声明（[src/llama-kv-cache.h:558-570](src/llama-kv-cache.h#L558-L570)）并在同一 `paged_log_stats()` 打印，复用既有 grep 习惯。

---

## 3. mincore telemetry 设计

### 3.1 env gate（仅建议名）

```text
LLAMA_KV_PAGED_MINCORE=1
```

默认关闭。开启条件建议与 madvise path 对齐：要求 `kv_paged_enabled && !v_trans && n_stream==1`，否则打印一次 warning 并禁用（与 `paged_madvise_block` 的前置一致），不 crash。非 Linux / 非 CPU backend 直接禁用。

### 3.2 核心采样函数（仅建议签名，本轮不实现）

```cpp
// 只读：对所有 KV layer 的 K/V tensor 做 page-aligned mincore，
// 累加 resident / total bytes 与 pages，更新 telemetry 计数器。
void paged_sample_mincore(const char * tag) const;   // tag 标注采样点
```

实现要点：

```text
for layer in layers:
    for t in {k_stream[0], v_stream[0]}:
        base = t->data; len_bytes = ggml_nbytes(t)（或 row*kv_size*n_stream）
        a_start = align_up(base, pg)
        a_end   = align_down(base + len_bytes, pg)
        n_pages = (a_end - a_start) / pg
        vec<unsigned char> v(n_pages)
        rc = mincore((void*)a_start, a_end-a_start, v.data())
        if rc != 0: failures++ ; continue
        resident_pages += count of (v[i] & 1)
        total_pages    += n_pages
```

注意：mincore 第三参数每页 1 字节，bit0=resident。统计 `v[i] & 1`。

### 3.3 拟输出字段

聚合：

```text
kv_mincore_enabled
kv_mincore_total_bytes        (= total_pages * pagesize)
kv_mincore_resident_bytes     (= resident_pages * pagesize)
kv_mincore_resident_pages
kv_mincore_total_pages
kv_mincore_resident_ratio     (resident_pages / total_pages, 打印为千分比或浮点)
kv_mincore_sample_calls
kv_mincore_failures
```

K/V 分开（建议，便于看 V 是否因 v_trans/layout 行为不同）：

```text
kv_mincore_k_total_bytes
kv_mincore_k_resident_bytes
kv_mincore_v_total_bytes
kv_mincore_v_resident_bytes
```

为支持「前/后对照」，建议每个采样点存一组快照（见 §4），而非只存 last。最小实现可先存 `before_madvise` / `after_madvise` / `after_resume` 三组 resident_bytes。

---

## 4. 采样点设计

至少四个采样点（tag 区分）：

```text
1. prefill 后 / 初始化稳定后        -> baseline KV resident
2. idle swap + madvise 前           -> pre_madvise
3. idle swap + madvise 后           -> post_madvise
4. resume swap-in 后                -> post_resume
```

落点（基于已读代码）：

```text
- 点 2/3 应夹在现有 idle madvise 调用两侧。idle swap-out + madvise 的累计
  RSS 采样已在 paged_madvise_block 调用点附近完成
  （[src/llama-kv-cache.cpp:1955-1984] 的 paged_swap_rss_* 更新），
  mincore 点 2 紧贴 rss_before、点 3 紧贴 rss_after，即可与现有
  current-RSS 口径逐点并列。
- 点 4 落在 resume swap-in 完成后（paged_swap_in_block 成功返回后）。
- 点 1 落在首个 decode/prefill 结束、进入 idle 之前。
```

目标问答：

```text
madvise 前 KV resident bytes 是多少？        -> 点2
madvise 后 KV resident bytes 降了多少？      -> 点2 - 点3
resume swap-in 后 KV resident 是否回升？     -> 点4 vs 点3
```

这样可得到 **KV resident drop**（点2−点3），与 Stage 5C/5D 的 **process current RSS drop** 并列，直接回答 5E-0 留下的「57 MiB 占 KV resident 多少」。

---

## 5. page alignment 处理

与 `paged_madvise_block` 完全同款，避免两套口径：

```text
pg      = sysconf(_SC_PAGESIZE)
a_start = (lo_a + pg - 1) & ~(uintptr_t)(pg - 1)   // 向上对齐
a_end   =  hi_a          & ~(uintptr_t)(pg - 1)    // 向下对齐
若 a_end <= a_start: 跳过该 tensor（计 skipped，不计 failure）
mincore 的 addr 必须页对齐，length 任意但内部按页计；用 a_end-a_start。
```

要点：

```text
1. mincore 的起始地址必须页对齐，否则 EINVAL；
2. 与 madvise 用同样的「向上对齐起点、向下对齐终点」，
   使 mincore 统计的区间 ⊆ madvise 实际 advise 的区间，口径一致；
3. tensor 首尾不足一页的边缘 page 被排除，属预期（madvise 也排除），
   这部分边缘字节不计入 total/resident。
```

---

## 6. 安全边界（必须写清）

```text
1. mincore 只读 page residency，不写、不改内存内容；
2. 不改变 RESIDENT / SWAPPED / RELEASED / UNUSED 状态机；
3. 不改 graph / kernel / get_k / get_v / mask / cpy_k / cpy_v；
4. 不改 swap-out / madvise / swap-in 逻辑，只在其前后「旁观」采样；
5. 只支持 Linux + CPU backend（#if defined(__linux__)）；非 Linux 或
   非 CPU buffer 直接禁用并返回 0；
6. mincore 需要 page-aligned addr，未对齐会 EINVAL，必须先对齐（§5）；
7. mincore 每次遍历所有 layer 的 K/V tensor，有系统调用开销，
   不应在性能矩阵 / 默认路径开启，仅 LLAMA_KV_PAGED_MINCORE=1 时启用。
```

---

## 7. 风险点分析

```text
1. host pointer 稳定性：CPU backend 下 t->data 是有效 host 基址
   （paged_madvise_block 已依赖它），风险低；但非 CPU backend
   （GPU/RPC）t->data 不是可 mincore 的 host 地址，必须 gate 掉。

2. 同 allocator 区域混淆：CPU_Mapped(权重 mmap) / CPU_REPACK / KV buffer
   是不同 ggml buffer。本设计只对 layer.k/v tensor 的 t->data 区间
   mincore，不触碰权重 buffer，因此不会把权重 resident 误计入 KV。
   （5E-0 已确认 KV buffer 是独立的 "CPU KV buffer"。）

3. K/V 连续性：K、V 是独立 tensor，跨 tensor 不保证连续；本设计逐 tensor
   mincore 后累加，不假设连续，规避此风险（§2.2）。

4. backend 可移植性：仅 Linux CPU 支持；其它 backend / OS 禁用并输出
   kv_mincore_enabled=0，不影响 correctness。

5. madvise 后 resident 反映：MADV_DONTNEED 对匿名页是即时丢弃，
   mincore 在 madvise 后应能看到对应页 resident bit 归 0；但被
   skip_neighbor 保护、未满整页的边缘页不会被释放，故 resident drop
   会小于理论 block 字节——这正是要量化的量，不是 bug。

6. mincore vs process RSS 不完全一致：mincore 只统计 KV tensor 区间，
   process RSS 含权重/compute/allocator/其它；且内核 RSS 与 mincore
   resident 的采样时点、page accounting 口径略有差异。二者本就不应相等，
   mincore 的价值正是给出「KV 专属」的 resident 口径。

7. 采样开销与噪声：点1-4 各一次全 tensor 遍历，ctx 越大 tensor 越大、
   页数越多，遍历越久；故只在 telemetry run 开启，性能矩阵关闭。
```

---

## 8. 与现有 telemetry 的拼接

```text
现有: paged_swap_rss_total_drop_kb  -> process current RSS 窗口净降（57 MiB）
新增: kv_mincore_resident_bytes(点2) - kv_mincore_resident_bytes(点3)
                                    -> KV 专属 resident drop

可计算:
  kv_resident_drop / process_rss_total_drop  -> madvise 释放的 RSS 中 KV 占比
  kv_resident_drop / kv_mincore_total_bytes  -> 释放了 KV resident 的多少比例
  process_rss_total_drop / 理论KV容量(5E-0)  -> 已知 ~11%
三者并列，才能把 5E-0 的「57 MiB ≈ 理论KV 11%」升级为
「57 MiB 中有多少真正来自 KV resident 页释放」。
```

---

## 9. 最小实现规模预估（若进入 5E-1-code）

```text
- 新增私有方法 paged_sample_mincore(tag) const  （~50 行，复用 madvise 地址推导）
- 新增 mutable 计数器字段（紧贴 paged_swap_rss_* 声明）  （~16 行）
- 4 个采样点调用插入（点1-4）  （~8 行）
- paged_log_stats() 增打印块  （~20 行）
- env 解析 paged_mincore_requested  （~3 行，紧贴现有 env 解析）
全部在 src/llama-kv-cache.{h,cpp} 内，不动 graph/kernel/examples/CMake。
```

---

## 10. 结论与建议

设计已落到具体函数与行号，实现路径清晰、风险可控、可完全复用现有 madvise 地址推导与 RSS telemetry 基础设施。建议在用户确认后进入 **Stage 5E-1-code**，按 §9 的最小规模实现，仅改 `src/llama-kv-cache.h` 与 `src/llama-kv-cache.cpp` 两个文件。
