# Stage 2 exact KV swap correctness results

## 1. 目标和边界

Stage 2 exact swap 的目标是验证 runtime KV swap 的无损正确性闭环：已经写入的 KV cell 可以被换出到 file-backed backing store，再在 attention 读取前换回原 KV tensor，并保持生成输出不变。

本阶段的边界：

- exact mode 会通过 `ensure_resident([0,n_kv))` 在每步 graph 读取前把 read window 内的 `SWAPPED` cell 全量换回。
- 本阶段不追求 RSS 下降，也不声称已经降低 active RSS 或 idle RSS。
- 本阶段不涉及 prefetch。
- 本阶段不涉及 PagedAttention-style paged read。
- 本阶段不涉及 approximate/window attention。

因此，当前结果用于证明 correctness，而不是证明内存收益。

## 2. 已完成的在线链路

当前 exact swap 在线路径如下：

```text
swap_out_window(n_kv)
-> swap_out_cell(cell)
-> file-backed backing store
-> ensure_resident(n_kv)
-> swap_in_cell(cell)
-> graph reads [0,n_kv)
```

其中 `swap_out_window(n_kv)` 根据 window/sink 选择冷区 cell；`swap_out_cell(cell)` 将该 cell 的所有层 K/V 字节写入 file-backed backing store；`ensure_resident(n_kv)` 在 graph 读取前遍历 `[0,n_kv)`，对 `SWAPPED` cell 调用 `swap_in_cell(cell)` 恢复原 KV tensor。

## 3. 关键数据结构和开关

关键状态和 metadata：

- `llama_kv_cell_state`：当前包含 `UNTOUCHED / RESIDENT / SWAPPED / RELEASED`。
- `swap_offset`：记录 cell 在 backing store 中的文件偏移；`offset == 0` 是合法偏移。
- `swap_size`：记录 cell 写入 backing store 的字节数；`swap_size == 0` 表示没有 backing slot。
- file-backed backend：当前使用临时文件 + append-only `pwrite/pread`，非持久化，不做 fsync。

运行期开关：

- `LLAMA_KV_SWAP=1`：启用 runtime KV swap 框架。
- `LLAMA_KV_SWAP_MODE=exact`：启用 exact mode。
- `LLAMA_KV_SWAP_WINDOW`：保留的 resident window 大小。
- `LLAMA_KV_SWAP_SINK`：保留的 sink token 数量。

## 4. 字节布局

`swap_out_cell()` 使用固定 staging buffer 布局：

```text
layer0 K
layer0 V
layer1 K
layer1 V
...
```

`swap_in_cell()` 使用完全对称的层序和 K/V 顺序，将 backing store 中的 staging buffer 写回原 KV tensor。两端均按同一 row size 和 offset 计算方式处理 `!v_trans && n_stream == 1` 的 CPU host buffer 布局。

## 5. `apply()` 调用顺序

当前 `apply()` 中的顺序为：

```text
apply_ubatch
get_n_kv
swap_out_window
ensure_resident
clear_frontier_advance
madvise_tail
graph
```

设计含义：

- `swap_out_window` 位于 `ensure_resident` 前，用于先触发 cold cell 换出。
- `ensure_resident` 随后保证 graph 读取 `[0,n_kv)` 时不会读到 `SWAPPED` cell。
- `clear_frontier_advance` 和 `madvise_tail` 仍属于 Stage 1 的 lazy-clear / lazy-tail 逻辑，目标区间和 Stage 2 live-history swap 不同。

## 6. Correctness 实验设置

实验配置：

- model：`Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
- `ctx=512`
- `n=64`
- `seed=42`
- `-fa on`
- prompt：`Hello, how are you?`

baseline 环境：

```text
LLAMA_LOW_MEM_WARMUP=minimal
LLAMA_KV_LAZY_CLEAR=1
```

swap-enabled 环境：

```text
LLAMA_KV_SWAP=1
LLAMA_KV_SWAP_MODE=exact
LLAMA_KV_SWAP_WINDOW=1
LLAMA_LOW_MEM_WARMUP=minimal
LLAMA_KV_LAZY_CLEAR=1
```

输出一致性结果：

```text
base_vs_w1_equal=0
sha256(base.out) = 5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006
sha256(w1.out)   = 5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006
```

说明：`cmp`/`diff` 风格命令中返回值 `0` 表示两份输出逐字节一致。

runtime stats：

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

round-trip selftest：

```text
KV swap roundtrip selftest pass: cell=0 bytes=131072
```

## 7. 结果解释

`LLAMA_KV_SWAP_WINDOW=1` 是激进触发条件：除最后 1 个 token 外，`[sink, n_kv - window)` 中的 resident cell 都会被视为冷区候选。该配置用于最大化触发 exact swap 路径，而不是用于性能优化。

结果表明：

- `swap_out_calls=2464` 且 `swap_in_calls=2464`，说明 swap-out / swap-in 路径大量触发。
- `ensure_calls=64` 且 `window_calls=64`，说明每步均执行了 window trigger 和 resident 保证逻辑。
- `bytes_written == bytes_read == 322961408`，说明写出和读回字节量一致。
- `write_calls == read_calls == 2464`，说明每次 cell swap-out 都有对应 swap-in。
- `backend_failures=0`，说明 backing store I/O 路径未报告失败。
- baseline 与 swap-enabled 输出 sha256 完全一致，证明 exact swap 在线正确性闭环成立。

## 8. 为什么当前不降 RSS

当前 exact mode 不预期降低 RSS，原因如下：

- `ensure_resident(n_kv)` 每步都会把 `[0,n_kv)` 内的 `SWAPPED` cell 全量换回，确保 exact attention 读取连续 KV view 时语义不变。
- 当前没有对已经换出的原 KV 页执行 page-aligned `madvise`。
- 原 KV tensor allocation 仍存在，backing store 只是新增了可恢复副本。
- 因此 active RSS / idle RSS 暂时不下降是预期行为。

本阶段的目标是 correctness，而不是 RSS 收益。后续如要证明 RSS 下降，需要引入明确的页回收路径或改变 attention 读取方式。

## 9. 下一步路线

- `exact-rss`：对已换出的原 KV 页做 page-aligned `madvise`，尝试降低 idle/current RSS，并验证换回正确性。
- `exact-correctness` 扩展：覆盖更长上下文、多 prompt、多 seed，并保留输出 hash 和 runtime stats。
- `approximate/window mode`：不再对 `[0,n_kv)` 全量换回，用 window/sink attention 展示 active RSS 下降，但需要明确这是近似策略。
- Stage 3 prefetch：在 swap-in 前加入预取策略，隐藏 I/O 延迟。
- Stage 4 PagedAttention-style paged read：引入 block table / physical block pool / 非连续 paged read，避免 exact 连续 view 的全量换回约束。
