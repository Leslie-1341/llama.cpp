# Stage 2 exact KV swap RSS measurement results

## 1. 目标

本阶段目标不是实现 `madvise`，也不是引入新的 swap 策略，而是测量当前 exact 全量 `ensure_resident([0,n_kv))` 架构下，runtime exact swap 是否已经能够降低 RSS。

当前 exact swap 已经完成 correctness 闭环：

```text
swap_out_window
-> swap_out_cell
-> file-backed backing store
-> ensure_resident
-> swap_in_cell
-> graph read
```

本轮只新增 RSS measurement：

- `LLAMA_KV_SWAP_RSS_SAMPLE=1`
- current RSS sampling
- `/proc/self/status` 的 `VmHWM` peak RSS
- 析构阶段统计打印

## 2. 实验设置

模型与参数：

- model：`Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- `ctx=512`
- `n=64`
- `seed=42`
- `-fa on`
- prompt：`Hello, how are you?`

baseline：

```text
LLAMA_LOW_MEM_WARMUP=minimal
LLAMA_KV_LAZY_CLEAR=1
LLAMA_KV_SWAP_RSS_SAMPLE=1
```

swap-enabled：

```text
LLAMA_KV_SWAP=1
LLAMA_KV_SWAP_MODE=exact
LLAMA_KV_SWAP_WINDOW=1
LLAMA_KV_SWAP_RSS_SAMPLE=1
LLAMA_LOW_MEM_WARMUP=minimal
LLAMA_KV_LAZY_CLEAR=1
```

## 3. 输出正确性

输出一致性结果：

```text
base_vs_w1_equal=0
sha256(base.out)=5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006
sha256(w1.out)=5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006
```

说明：`base_vs_w1_equal=0` 表示 `cmp` 返回 0，即 baseline 与 swap-enabled 输出逐字节一致。

## 4. RSS 结果

baseline：

```text
RSS peak_kb=8163568 current_last_kb=8160600 current_min_kb=8156964 current_max_kb=8160600 rss_samples=64
```

swap-enabled：

```text
RSS peak_kb=8163848 current_last_kb=8160700 current_min_kb=8157260 current_max_kb=8160700 rss_samples=64
```

对比：

- peak RSS：swap-enabled 比 baseline 高约 `280 KiB`。
- current last RSS：swap-enabled 比 baseline 高约 `100 KiB`。
- current max RSS：swap-enabled 比 baseline 高约 `100 KiB`。
- 两组差异为噪声级。
- 没有观察到 RSS 下降。

该结果符合当前 exact 架构预期：runtime swap path 已经运行，但没有释放或回收原 KV tensor 页。

## 5. Swap stats

swap-enabled runtime stats：

```text
swap_out_calls=2464
swap_in_calls=2464
ensure_calls=64
window_calls=64
window_skipped=0
backend_failures=0
bytes_written=322961408
bytes_read=322961408
write_calls=2464
read_calls=2464
release_calls=0
```

解释：

- `swap_out_calls=2464`、`swap_in_calls=2464`，说明 swap path 确实大量触发。
- `bytes_written == bytes_read == 322961408`，说明 file-backed backing store 读写字节量一致。
- `write_calls == read_calls == 2464`，说明每次 swap-out 都有对应 swap-in。
- `backend_failures=0`，说明 backing store I/O 未报告失败。
- 输出仍与 baseline 逐字节一致。
- 但 RSS 没有下降。

## 6. 原因分析

当前 exact mode 不降低 RSS 的原因是结构性的：

- `ensure_resident([0,n_kv))` 每步都会把 read window 内的 `SWAPPED` cell 全量换回。
- 当前没有对原 KV tensor 页执行 `madvise(MADV_DONTNEED)`。
- 原 KV tensor allocation 仍然存在，且 graph 读取前需要保持 resident。
- file-backed backing store 当前是额外副本，而不是原 KV 物理页的替代驻留位置。
- 因此，当前 exact swap 只证明 correctness，不产生 RSS 收益。

这不是失败结果，而是对 Stage 2 exact correctness 架构边界的确认：在不回收原 KV 页、不改变 exact attention 连续 read view 的前提下，RSS 不下降是预期行为。

## 7. 后续路线

- `exact-rss-madvise`：尝试对已换出原 KV 页做 page-aligned `MADV_DONTNEED`，但必须默认关闭，并先计算 safe run，避免破坏仍需读取的 KV row。
- `approximate/window mode`：如果不再对 `[0,n_kv)` 全量 `ensure_resident`，才可能降低 active RSS；该路线需要明确近似语义。
- Stage 3 prefetch：在 swap-in 前加入预取，掩盖 file-backed I/O 延迟。
- Stage 4 paged read：引入 PagedAttention-style paged read，解决 exact 连续 view 下必须全量 resident 的根本约束。
