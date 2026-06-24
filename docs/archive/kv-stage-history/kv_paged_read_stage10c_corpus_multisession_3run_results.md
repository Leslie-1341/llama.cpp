# KV Paged Read - Stage 10-C-C WikiText-2 Corpus-backed Multi-session 3-run 结果记录

## 1. 阶段目标

Stage 10-C-C 的目标是在真实语料驱动的 semi-real 多会话 paused/resume workload 下，验证不同 KV reclaim 策略的 RSS、吞吐和 resume first-token latency 取舍。

本阶段基于 Stage 9 的 semi-real multi-session driver，但将固定 prompt 替换为 WikiText-2 语料切分得到的 prompt。

本阶段验证的问题是：

```text
在 WikiText-2 真实语料驱动的多会话暂停/恢复 workload 下，
tail/lazy、idle swap、prefetch、defer 等策略的 RSS 收益和性能代价是否仍然成立。
```

本阶段仍不是 llama-server / HTTP / ShareGPT / k6 server benchmark。

---

## 2. 实验配置

### 2.1 模型

```text
/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

### 2.2 语料

```text
wikitext-2-raw/wiki.test.raw
```

corpus mode 参数：

```text
LLAMA_KV_SEMI_CORPUS_FILE=wikitext-2-raw/wiki.test.raw
LLAMA_KV_SEMI_CORPUS_CHARS=1024
LLAMA_KV_SEMI_CORPUS_OFFSET=0
```

### 2.3 运行参数

```text
--ctx-size 2048
--batch-size 512
--ubatch-size 128
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 4
--seed 1
--temp 0
```

### 2.4 corpus prompt token 数

四个 session 的 prompt token 数在所有 case 和 run 中保持一致：

```text
A_tokens=232
B_tokens=257
C_tokens=232
D_tokens=246
```

这说明不同策略之间的 RSS / TPS / latency 差异主要来自策略本身，而不是 prompt token 数变化。

---

## 3. 实验矩阵

本阶段选取四组策略，每组运行 3 次并取 median：

| case | 含义 |
|---|---|
| S0_baseline_paged_off | 基线，不启用 paged / lazy / swap |
| S2_tail_lazy_only | 只启用 tail/lazy reclaim |
| S4_idle_swap_prefetch_defer | 启用 idle swap + madvise + prefetch + defer，不启用 tail/lazy |
| S5_tail_idle_prefetch_defer | 启用 tail/lazy + idle swap + madvise + prefetch + defer |

---

## 4. 正确性和安全性结果

12 次运行全部正常退出：

```text
S0_baseline_paged_off   1   0
S2_tail_lazy_only       1   0
S4_idle_swap_prefetch_defer     1   0
S5_tail_idle_prefetch_defer     1   0
S0_baseline_paged_off   2   0
S2_tail_lazy_only       2   0
S4_idle_swap_prefetch_defer     2   0
S5_tail_idle_prefetch_defer     2   0
S0_baseline_paged_off   3   0
S2_tail_lazy_only       3   0
S4_idle_swap_prefetch_defer     3   0
S5_tail_idle_prefetch_defer     3   0
```

安全性检查结果：

```text
real_abnormal_sum = 0
all_finished_sum = 3
write_to_swapped = 0
active_visible_violation = 0
active_visible_violation_rows = 0
```

这说明：

1. 所有 case 的 3 次运行均无异常退出；
2. 四个 session 均完成；
3. 未出现 write-to-swapped；
4. 未出现 active-visible violation；
5. 未观察到 crash、NaN、assert、backend failure 等异常。

---

## 5. 3-run median 结果

| case | RSS KiB | RSS drop vs S0 MiB | active TPS | TPS delta vs S0 | resume avg ms | resume delta vs S0 ms |
|---|---:|---:|---:|---:|---:|---:|
| S0_baseline_paged_off | 8,714,852 | 0.00 | 11.901 | 0.00% | 86.804 | 0.000 |
| S2_tail_lazy_only | 8,518,540 | 191.71 | 12.076 | +1.47% | 84.692 | -2.113 |
| S4_idle_swap_prefetch_defer | 8,538,940 | 171.79 | 9.897 | -16.83% | 114.036 | +27.232 |
| S5_tail_idle_prefetch_defer | 8,342,704 | 363.43 | 9.906 | -16.76% | 113.804 | +26.999 |

---

## 6. 分策略分析

### 6.1 S2：tail/lazy only

S2 相对 S0：

```text
RSS drop = 191.71 MiB
TPS delta = +1.47%
resume delta = -2.11 ms
```

说明在 WikiText-2 corpus-backed workload 下，tail/lazy reclaim 单独即可带来稳定 RSS 下降，并且没有观察到性能回退。

该结果支持：

```text
tail/lazy reclaim 对真实语料连续 prompt 的 semi-real 多会话 workload 仍然有效。
```

---

### 6.2 S4：idle swap + prefetch + defer

S4 相对 S0：

```text
RSS drop = 171.79 MiB
TPS delta = -16.83%
resume delta = +27.23 ms
idle_swap_out_calls = 89
swap_in_calls = 44
madvise_calls = 89
swapped_blocks = 45
```

说明 idle swap / madvise / prefetch / defer 路径在 corpus-backed workload 中真实触发，并带来了 RSS 下降。

但代价也明显：

```text
active TPS 下降约 16.83%
resume first-token 平均增加约 27.23 ms
```

因此 S4 的结论应表述为：

```text
idle swap 路径在真实语料多会话 workload 中有效，但当前策略仍有明显性能代价。
```

---

### 6.3 S5：tail/lazy + idle swap + prefetch + defer

S5 相对 S0：

```text
RSS drop = 363.43 MiB
TPS delta = -16.76%
resume delta = +27.00 ms
idle_swap_out_calls = 89
swap_in_calls = 44
madvise_calls = 89
swapped_blocks = 45
```

S5 获得本阶段最大 RSS 收益：

```text
363.43 MiB
```

说明 tail/lazy 和 idle swap 的收益可以叠加。

但性能代价仍明显：

```text
active TPS 下降约 16.76%
resume first-token 平均增加约 27.00 ms
```

因此 S5 的严谨结论是：

```text
组合策略在真实语料驱动的 semi-real workload 中仍能显著降低 RSS，
但相比 Stage 9 fixed-prompt workload，性能代价明显增大。
```

---

## 7. 与 Stage 9 fixed-prompt 结果的关系

Stage 9 fixed-prompt semi-real workload 中，S5 组合策略的 3-run median 大致表现为：

```text
RSS drop ≈ 403.6 MiB
TPS delta ≈ -0.8%
resume first-token delta ≈ +3.7 ms
```

Stage 10-C-C corpus-backed workload 中，S5 表现为：

```text
RSS drop = 363.43 MiB
TPS delta = -16.76%
resume first-token delta = +27.00 ms
```

因此，两个阶段的结论应区分：

1. Stage 9 证明在固定 semi-real prompt 下，S5 组合策略可以获得约 403 MiB RSS 下降，并且吞吐代价较小；
2. Stage 10-C-C 证明在 WikiText-2 真实语料 prompt 下，S5 仍有约 363 MiB RSS 下降；
3. 但真实语料下的 TPS 和 resume latency 代价明显更高。

这说明当前策略仍有优化空间，尤其是：

```text
swap-in / prefetch 时机；
prefetch block 数量；
defer window；
resume 前恢复策略；
真实语料下的 active read window 与 swapped block 交互。
```

---

## 8. 关键结论

Stage 10-C-C 已完成。

核心结论：

```text
在 WikiText-2 corpus-backed semi-real 多会话 paused/resume workload 下，
S5 组合策略相对 S0 实现了 363.43 MiB 的 RSS 下降；
同时 active TPS 下降约 16.76%，resume first-token 平均增加约 27.00 ms；
所有 12 次运行均无异常、无 write-to-swapped、无 active-visible violation。
```

这说明：

1. 真实语料下 RSS 下降仍然成立；
2. tail/lazy 和 idle swap 的收益可以叠加；
3. 当前 idle swap + prefetch + defer 策略在真实语料下有明显性能代价；
4. 后续需要继续优化 prefetch / swap-in / resume policy。

---

## 9. 边界与限制

本阶段仍有以下限制：

1. 不是 llama-server benchmark；
2. 没有 HTTP request；
3. 没有 ShareGPT 请求分布；
4. 没有真实 server slot；
5. 没有 continuous batching scheduler 改造；
6. 仍然是 example driver 中构造的 semi-real paused/resume workload；
7. RSS / TPS / resume latency 结论只在当前固定配置下成立：

```text
model = Meta-Llama-3-8B-Instruct-Q4_K_M
corpus = WikiText-2 test raw
corpus_chars = 1024
corpus_offset = 0
ctx-size = 2048
parallel = 4
cache-type-k/v = f32
runs = 3
```

因此，不能把本阶段结果直接表述为真实 server workload 结论。

---

## 10. 下一步方向

下一步建议进入 Stage 10-D，目标不是继续扩大矩阵，而是解释和优化 Stage 10-C-C 中暴露出的性能代价。

建议方向：

1. 分析 S4/S5 的 TPS 下降来源；
2. 检查 prefetch 是否过晚、过少或与 active decode 争用；
3. 检查 swap-in calls 与 resume latency 的对应关系；
4. 比较 `LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP` 不同取值；
5. 比较 `LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS` 不同取值；
6. 判断是否需要 corpus-backed 下的 prefetch 参数重新调优；
7. 后续再考虑完整 S0-S5 或 llama-server / ShareGPT benchmark。
