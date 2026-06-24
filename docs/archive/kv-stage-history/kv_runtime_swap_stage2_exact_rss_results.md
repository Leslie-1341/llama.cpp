# Stage 2 exact RSS results

## 1. 阶段目标

exact-rss2 的目标是验证 page-aligned `MADV_DONTNEED` 路径是否安全：在 exact swap 已经把 KV cell 写入 file-backed backing store 后，对对应的原 KV tensor 页执行 page release，并确认后续 `ensure_resident([0,n_kv))` 能正确换回，不破坏输出。

本阶段不是 active RSS 优化阶段，也不声称当前已经降低 RSS。当前 exact mode 仍然在 graph read 前全量 `ensure_resident([0,n_kv))`，因此主要结论是 correctness / safety，而不是内存收益。

## 2. 实现路径

当前 exact-rss2 在线路径：

```text
swap_out_window
-> swap_out_cell
-> file-backed backing store
-> madvise_swapped_runs
-> MADV_DONTNEED
-> ensure_resident
-> swap_in_cell
-> graph read
```

含义：

- `swap_out_window` 选择 window 外的 cold KV cell。
- `swap_out_cell` 将 cell 的 K/V 字节写入 file-backed backing store。
- `madvise_swapped_runs` 对连续 `SWAPPED` run 计算 safe page-aligned range。
- `MADV_DONTNEED` 只在 `LLAMA_KV_SWAP_MADVISE=1` 时实际调用。
- `ensure_resident` 在 graph read 前把 `[0,n_kv)` 内的 `SWAPPED` cell 换回。
- graph 仍读取连续 `[0,n_kv)` KV view。

## 3. Safe page range 算法

`madvise_swapped_runs(n_kv)` 的 safe range 计算规则：

- 扫描 `[0, min(n_kv, cells.size()))` 内连续 `SWAPPED` run。
- 对每个 run `[c_lo, c_hi)`，逐层、逐 K/V tensor 独立计算。
- 对 tensor 使用：

```text
lo_byte = base + c_lo * row
hi_byte = base + c_hi * row
a_start = page_align_up(lo_byte)
a_end   = page_align_down(hi_byte)
```

- 只对 `a_end > a_start` 的整页区间调用 `MADV_DONTNEED`。
- inward clipping 避免释放包含 `RESIDENT / UNTOUCHED` cell 的边界页。
- 若 `row != t->nb[1]`，说明布局假设不成立，跳过该 tensor-run，并计入 failure / skipped。
- 该路径默认关闭，仅由 `LLAMA_KV_SWAP_MADVISE=1` 显式开启。

## 4. Correctness 结果

输出一致性：

```text
base_vs_w1_madvise_equal=0
nomadvise_vs_madvise_equal=0

sha256(base.out)         = 5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006
sha256(w1_nomadvise.out) = 5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006
sha256(w1_madvise.out)   = 5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006
```

说明：`cmp` 返回值为 `0` 表示输出逐字节一致。baseline、exact swap without madvise、exact swap with madvise 三组输出完全一致。

## 5. Runtime stats

swap stats：

```text
swap_out_calls=2464
swap_in_calls=2464
ensure_calls=64
window_calls=64
backend_failures=0
bytes_written=322961408
bytes_read=322961408
```

madvise stats：

```text
madvise_enabled=1
madvise_calls=64
madvise_candidate_runs=4096
madvise_advised_runs=4096
madvise_advised_bytes=301989888
madvise_failures=0
madvise_skipped_bytes=0
```

解释：

- swap path 大量触发，且 file-backed backing store 读写字节量一致。
- `madvise_candidate_runs == madvise_advised_runs == 4096`，说明 dry-run 识别出的 safe ranges 均成功进入 `MADV_DONTNEED` 路径。
- `madvise_failures=0`，说明 page release syscall 未报告失败。
- 输出仍保持 bit-exact。

## 6. RSS 结果

RSS measurement：

```text
base peak_kb=8163844 current_last_kb=8161512 current_max_kb=8161512
nomadvise peak_kb=8163628 current_last_kb=8161172 current_max_kb=8161172
madvise peak_kb=8163952 current_last_kb=8161180 current_max_kb=8161436
```

结果解释：

- RSS 没有实质下降。
- madvise 组 peak 甚至略高，差异属于噪声 / fault-back 开销量级。
- 原因是 exact mode 下 `ensure_resident([0,n_kv))` 紧跟在 `madvise_swapped_runs` 后全量换回；被 `MADV_DONTNEED` 的页会在 `swap_in_cell` 写回时立即 fault back。
- 当前架构中，file-backed backing store 是 correctness backup，不能替代 graph read 前必须 resident 的连续 KV view。

## 7. 阶段结论

exact-rss2 的结论：

- page release 算法路径安全。
- `MADV_DONTNEED` 没有破坏输出正确性。
- safe page-aligned inward clipping 没有触发 syscall failure。
- 当前 exact 全量 ensure 架构无法带来有效 RSS 收益。
- exact-rss 路线到此收束，不继续在 exact 全量 resident 语义下追求 RSS 降低。

## 8. 下一步

下一阶段应转向 `approximate-window-plan`：

- 不再对 `[0,n_kv)` 全量 `ensure_resident`。
- 让窗口外 KV 真正保持 `SWAPPED / RELEASED`，从而可能降低 active RSS。
- correctness 评价从 sha256 bit-exact 转为质量、perplexity、输出可接受性等指标。
- Stage 3 可在 approximate/window swap 基础上加入 prefetch，掩盖 swap-in I/O 延迟。
- Stage 4 再考虑 PagedAttention-style paged read，从根本上解决 exact 连续 view 需要全量 resident 的约束。
