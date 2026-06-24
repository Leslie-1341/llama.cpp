# Stage 11：paged shadow validation 性能回退定位与最终结果

## 1. 背景

本项目围绕 `llama.cpp` 的 KV cache 运行时内存优化展开。

前期已经实现：

1. KV cache 分页 block 管理；
2. idle request KV block 识别；
3. idle KV block swap-out；
4. `madvise(MADV_DONTNEED)` 释放物理页；
5. resume 时 swap-in；
6. during-active incremental prefetch；
7. final limited sync prefetch；
8. defer swap-out on resume；
9. WikiText-2 corpus-backed semi-real multi-session workload。

Stage 10-E 已经证明：在 WikiText-2 真实语料驱动的半真实多会话 workload 中，组合方案可以稳定降低进程 RSS，但当时观察到较明显的吞吐下降：

```text
S5_final12_no_core_trace:
  RSS drop ≈ 363 MiB
  active TPS delta ≈ -18% ~ -19%
  resume first-token delta ≈ +20 ms
```

这个结果说明内存收益成立，但性能代价偏高。因此 Stage 11 的目标是定位并降低 paged 路径的性能开销。

---

## 2. workload 说明

当前 workload 是：

```text
corpus-backed semi-real multi-session workload
```

含义是：

```text
prompt 来自真实语料；
session 生命周期和 pause/resume 时间线由 driver 人为设定。
```

当前使用语料：

```text
wikitext-2-raw/wiki.test.raw
```

corpus 配置：

```text
LLAMA_KV_SEMI_CORPUS_FILE=wikitext-2-raw/wiki.test.raw
LLAMA_KV_SEMI_CORPUS_CHARS=1024
LLAMA_KV_SEMI_CORPUS_OFFSET=0
```

实际 prompt token：

```text
A_tokens=232
B_tokens=257
C_tokens=232
D_tokens=246
```

该 workload 不是完整真实线上请求流，但比固定 prompt workload 更接近真实文本输入，可用于验证 KV cache 机制在真实语料下的内存收益和性能代价。

---

## 3. Stage 11 初始问题

在接入真实语料后，观察到 paged 相关配置存在较大吞吐下降。

最初怀疑方向包括：

1. during-active prefetch 过重；
2. final sync prefetch 过重；
3. `madvise` 造成额外开销；
4. `row_idx` 构造过重；
5. `ggml_get_rows` 替代普通 view 导致 K/V 读取变慢；
6. paged 主路径中的 `paged_resolve()` 过多。

通过多轮拆分实验后发现，早期判断需要修正：

```text
真正导致大幅 TPS 回退的主因，不是 prefetch、madvise、row_idx 填充，也不是 get_k/get_v 主读取路径本身，而是默认开启的 paged_shadow_validate 调试校验。
```

---

## 4. prefetch timing 排查

Stage 11-A.5 对 prefetch 增加 timing telemetry。

关键结果：

```text
S5_final12_no_core_trace:
  active_decode 增加 ≈ 1025.576 ms
  step + final prefetch elapsed ≈ 65.816 ms
```

比例：

```text
65.816 / 1025.576 ≈ 6.4%
```

结论：

```text
prefetch 本身有成本，但不是 -18% ~ -19% TPS 回退的主要来源。
```

因此直接做异步 prefetch 的收益有限，且后台线程直接调用 `llama_memory_prefetch_seq_step()` 存在线程安全风险，不应优先推进真异步实现。

---

## 5. paged base path timing

为定位 `LLAMA_KV_PAGED=1` 的基础路径成本，新增了：

```text
LLAMA_KV_PAGED_TIMING=1
```

新增 summary 行：

```text
KV_PAGED_TIMING_SUMMARY ...
```

主要字段包括：

```text
apply_calls
apply_paged_total_us
set_row_idx_calls
set_row_idx_total_us
row_idx_fill_us
active_visible_us
nonidentity_probe_us
check_read_resident_us
paged_resolve_calls
cells_scanned
blocks_scanned
row_idx_entries
getenv_calls
```

初步结果显示：

```text
P0_paged_only_no_ingraph:
  active_decode_delta ≈ +1570 ms
  apply_paged_total_us ≈ 108 ms
  set_row_idx_total_us ≈ 13 ms
```

这说明：

```text
apply 阶段和 set_input_paged_row_idx 的 CPU 填充开销无法解释主要 TPS 回退。
```

---

## 6. INGRAPH 语义修正

后续审计发现：

```text
不设置 LLAMA_KV_PAGED_INGRAPH 并不等于关闭 in-graph。
```

源码语义是：

```text
只有 LLAMA_KV_PAGED_INGRAPH=0 时才关闭 row_idx input；
未设置时默认允许创建 row_idx。
```

因此之前命名为 `P0_paged_only_no_ingraph` 的实验实际并不是 no-ingraph，而是默认 ingraph-on。

修正后显式设置：

```text
LLAMA_KV_PAGED_INGRAPH=0
```

得到：

```text
set_row_idx_calls=0
row_idx_entries=0
```

但性能仍然下降明显，说明问题不只在 row_idx。

---

## 7. paged_resolve 调用审计

进一步审计 `paged_resolve()` 的调用来源。

关键发现：

```text
get_k() / get_v() 在 row_idx == nullptr 时不会调用 paged_resolve()；
它们会直接走普通 ggml_view_4d。
```

`paged_resolve_calls ≈ 300 万` 的主要来源不是 K/V 读取路径，而是：

```text
paged_shadow_validate()
```

`paged_shadow_validate()` 默认开启，并在每个 ubatch 中进行完整校验：

```text
每个 ubatch
  × n_layer
  × K/V 两份
  × n_kv row
```

以当前模型和 workload 估算：

```text
64 次 ubatch × 32 层 × 2 × 平均约 733 个 KV row
≈ 3,001,353 次 paged_resolve
```

这与实测完全吻合。

同时，`paged_shadow_validate()` 内部还会进行大量 row memcpy，这进一步放大了性能开销。

---

## 8. shadow_validate 配置对照实验

为验证该判断，进行配置对照：

```text
P0_ingraph0_shadow_on:
  LLAMA_KV_PAGED=1
  LLAMA_KV_PAGED_INGRAPH=0
  LLAMA_KV_PAGED_SHADOW_VALIDATE=1

P0_ingraph0_shadow_off:
  LLAMA_KV_PAGED=1
  LLAMA_KV_PAGED_INGRAPH=0
  LLAMA_KV_PAGED_SHADOW_VALIDATE=0
```

结果：

| case                   | active_tps | TPS delta | paged_resolve_calls |
| ---------------------- | ---------: | --------: | ------------------: |
| P0_ingraph0_shadow_on  |   9.266866 |  -25.211% |           3,001,353 |
| P0_ingraph0_shadow_off |  12.429794 |   +0.316% |               3,081 |

结论：

```text
关闭 shadow_validate 后，paged 基础路径性能基本恢复。
```

这证明先前 paged 基础路径的大幅 TPS 回退主要来自默认开启的调试校验，而不是 idle swap/madvise/prefetch 机制本身。

---

## 9. 源码修改

为避免正式 benchmark 被调试校验污染，将 `LLAMA_KV_PAGED_SHADOW_VALIDATE` 默认语义修改为：

```text
未设置：
  关闭 shadow validate

LLAMA_KV_PAGED_SHADOW_VALIDATE=0：
  关闭 shadow validate

LLAMA_KV_PAGED_SHADOW_VALIDATE=1：
  显式开启 shadow validate
```

修改文件：

```text
src/llama-kv-cache.cpp
src/llama-kv-cache.h
```

修改后保留了调试能力：

```text
LLAMA_KV_PAGED_SHADOW_VALIDATE=1
```

仍可恢复旧的重校验路径。

本次修改没有改变以下机制语义：

```text
paged_resolve
paged write
paged swap-out
paged swap-in
madvise
prefetch
defer
row_idx
get_k/get_v
public API
CMake
examples
scripts
docs
```

---

## 10. shadow_validate 默认关闭验证

修改后进行 1-run 验证。

关键结果：

| case                       | active_tps | TPS delta | resume delta | paged_resolve_calls |
| -------------------------- | ---------: | --------: | -----------: | ------------------: |
| P0_ingraph0_default_shadow |  11.926219 |   -0.828% |    -1.037 ms |               3,081 |
| P0_ingraph0_shadow_on      |   8.976657 |  -25.355% |   +30.146 ms |           3,001,353 |
| S5_final12_default_shadow  |  11.434722 |   -4.915% |    +3.008 ms |             161,454 |
| S5_final12_shadow_on       |   9.734921 |  -19.050% |   +19.623 ms |           3,159,726 |

结论：

```text
默认不设 LLAMA_KV_PAGED_SHADOW_VALIDATE 时，shadow_validate 已经默认关闭；
显式设置 LLAMA_KV_PAGED_SHADOW_VALIDATE=1 时，旧的重校验路径仍可复现。
```

---

## 11. 最终稳定 3-run median

最终进行 3-run median 验证。

最终配置：

```text
S5_final12_default_shadow
```

包含：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2
LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=2
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=12
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

关键点：

```text
不设置 LLAMA_KV_PAGED_SHADOW_VALIDATE
```

即使用源码默认关闭 shadow validation。

### 11.1 逐 run 结果

| case                      | run | RSS drop MiB | active TPS | TPS delta | resume avg ms | resume delta ms | all_finished |
| ------------------------- | --: | -----------: | ---------: | --------: | ------------: | --------------: | -----------: |
| S0_baseline               |   1 |        0.000 |  12.022071 |    0.000% |        83.471 |           0.000 |            1 |
| S0_baseline               |   2 |        0.000 |  11.499981 |    0.000% |        92.880 |           0.000 |            1 |
| S0_baseline               |   3 |        0.000 |  11.733393 |    0.000% |        86.897 |           0.000 |            1 |
| S2_tail_lazy              |   1 |      191.422 |  12.163841 |   +1.179% |        84.061 |          +0.591 |            1 |
| S2_tail_lazy              |   2 |      192.238 |  11.732388 |   +2.021% |        88.341 |          -4.539 |            1 |
| S2_tail_lazy              |   3 |      191.551 |  11.687110 |   -0.394% |        87.419 |          +0.522 |            1 |
| S5_final12_default_shadow |   1 |      363.262 |  11.384861 |   -5.300% |        89.435 |          +5.965 |            1 |
| S5_final12_default_shadow |   2 |      363.617 |  11.089148 |   -3.572% |        93.077 |          +0.197 |            1 |
| S5_final12_default_shadow |   3 |      363.406 |  11.240960 |   -4.197% |        91.305 |          +4.408 |            1 |

### 11.2 3-run median

| case                      | RSS drop MiB | active TPS | TPS delta | active decode delta ms | resume avg ms | resume delta ms | exit_sum | real_abnormal_sum | all_finished_sum |
| ------------------------- | -----------: | ---------: | --------: | ---------------------: | ------------: | --------------: | -------: | ----------------: | ---------------: |
| S0_baseline               |        0.000 |     11.733 |    0.000% |                  0.000 |        86.897 |           0.000 |        0 |                 0 |                3 |
| S2_tail_lazy              |      191.551 |     11.732 |   +1.179% |                -58.168 |        87.419 |          +0.522 |        0 |                 0 |                3 |
| S5_final12_default_shadow |      363.406 |     11.241 |   -4.197% |               +224.012 |        91.305 |          +4.408 |        0 |                 0 |                3 |

最终结论：

```text
S5_final12_default_shadow 在 WikiText-2 corpus-backed semi-real multi-session workload 中，3-run median 实现：

RSS drop = 363.406 MiB
active TPS delta = -4.197%
resume first-token delta = +4.408 ms
exit_sum = 0
real_abnormal_sum = 0
all_finished_sum = 3 / 3
```

---

## 12. 与 Stage 10-E 的对比

Stage 10-E 低日志 S5 结果约为：

```text
RSS drop ≈ 363.184 MiB
active TPS delta ≈ -18.706%
resume delta ≈ +20.198 ms
```

Stage 11 修正后：

```text
RSS drop ≈ 363.406 MiB
active TPS delta ≈ -4.197%
resume delta ≈ +4.408 ms
```

对比：

| 指标               |   Stage 10-E | Stage 11 final |   变化 |
| ---------------- | -----------: | -------------: | ---: |
| RSS drop         | ≈363.184 MiB |    363.406 MiB | 基本保持 |
| active TPS delta |    ≈-18.706% |        -4.197% | 明显改善 |
| resume delta     |  ≈+20.198 ms |      +4.408 ms | 明显改善 |

结论：

```text
关闭默认 shadow_validate 后，内存收益不变，性能代价显著下降。
```

---

## 13. 当前机制解决的问题

当前方案解决的是：

```text
多并发 / 长上下文场景下，paused 或 idle request 的 KV cache 继续占用物理内存的问题。
```

机制是：

```text
1. 将 KV cache 划分为 block；
2. 找出 idle request 独占且 active request 不需要读取的 KV block；
3. 将这些 block swap-out；
4. 对原 KV buffer 对应页面执行 madvise(MADV_DONTNEED)；
5. 让 OS 回收物理页，从而降低进程 RSS；
6. 请求 resume 前后再 swap-in；
7. 用 during-active prefetch 和 final limited sync prefetch 降低 resume 延迟；
8. 用 defer 避免刚恢复的 block 立即再次 swap-out。
```

---

## 14. 适用边界

当前收益依赖 workload 中 idle KV 的比例。

适合场景：

```text
多并发；
长上下文；
部分请求处于 pause / wait / idle；
idle 请求 KV 占比较高；
内存压力明显。
```

不适合或收益较小的场景：

```text
单请求持续生成；
所有请求持续 active；
没有 pause / idle；
短上下文；
KV cache 本身占比不高。
```

因此不能说：

```text
所有真实线上 workload 都能获得 363 MiB RSS 降低；
所有场景都只损失约 4% TPS；
单请求长上下文也有同等收益。
```

准确表述应为：

```text
在存在足够 idle KV 的多会话暂停/恢复场景下，该机制可以以较小性能代价显著降低运行时 RSS。
```

---

## 15. 当前阶段结论

Stage 11 的核心结论是：

```text
先前 paged 路径的大幅 TPS 回退主要来自默认开启的 paged_shadow_validate 调试校验，而非 idle KV swap / madvise / prefetch 机制本身。
```

将 shadow validation 改为默认关闭后，最终结果为：

```text
在 WikiText-2 真实语料驱动的半真实多会话 workload 中，
S5_final12_default_shadow 3-run median 实现：

363.406 MiB RSS 下降；
active TPS 仅下降 4.197%；
resume first-token 平均增加 4.408 ms；
3 次运行均 exit=0、real_abnormal=0、all_finished=1。
```

这说明：

```text
idle KV swap-out + madvise + resume swap-in + prefetch/defer 机制本身是有效的；
其内存收益稳定；
关闭调试校验后，性能代价处于可接受范围。
```

---

## 16. 后续工作

后续建议分三条线推进。

### 16.1 idle ratio sweep

当前 workload 的 idle 数量和 pause/resume 时间线是人为设定的。

为了量化收益边界，应增加：

```text
低 idle 占比；
中 idle 占比；
高 idle 占比；
不同 idle 持续时间；
不同 idle 历史长度。
```

目标是说明：

```text
RSS drop 与 idle KV 占比之间的关系。
```

推荐输出指标：

```text
idle_kv_bytes
swapped_nonres_bytes
rss_drop_mib
active_tps_delta
resume_delta_ms
```

### 16.2 更真实请求流

当前 workload 是真实 prompt + 人造调度。

后续可以进一步接近真实服务：

```text
ShareGPT / LMSYS / WildChat 风格对话长度分布；
请求到达间隔；
真实 pause/wait/resume pattern；
server-like continuous batching；
HTTP/k6 压测。
```

但这属于下一阶段，不应在当前稳定结果前继续扩大范围。

### 16.3 继续降低 S5 剩余性能成本

当前 S5 仍有约 4.2% TPS 回退。

可能来源：

```text
row_idx / nonidentity probe；
active-visible 检查；
prefetch step；
final sync prefetch；
swap-in/out；
madvise；
block state 维护。
```

可选优化方向：

```text
缓存 env bool；
减少不必要 row_idx 填充；
只有存在 swapped/nonidentity 时才构建复杂 row_idx；
缓存 block state 聚合信息；
进一步优化 prefetch 调度。
```

但当前阶段不建议继续大改，应先固化结果与文档。

---

## 17. 推荐汇报口径

可以对老师/队友这样说：

```text
我们针对多并发长上下文场景下 idle 请求 KV cache 持续占用物理内存的问题，实现了 block-level KV cache swap-out、madvise 释放物理页、resume swap-in，以及 prefetch/defer 降低恢复延迟的机制。

在 WikiText-2 真实语料驱动的半真实多会话 workload 中，最初观察到 paged 路径吞吐下降较大。通过 Stage 11 timing telemetry，我们定位到主要原因不是 swap/madvise/prefetch，而是默认开启的 paged_shadow_validate 调试校验在每个 ubatch 中全量扫描 KV，造成约 300 万次 paged_resolve 和大量 memcpy。我们将该调试校验改为默认关闭，仅在 LLAMA_KV_PAGED_SHADOW_VALIDATE=1 时显式开启。

最终 3-run median 结果显示，S5 组合方案实现约 363.4 MiB RSS 下降，active TPS 仅下降约 4.2%，resume first-token 平均增加约 4.4 ms，所有运行正常完成。这说明 idle KV swap/madvise 机制能够真实降低运行时物理内存占用，并且在关闭调试校验后性能代价较小。
```

---

## 18. 当前阶段状态

当前阶段可以视为稳定节点。

建议后续操作：

```text
1. 本地 commit shadow_validate 默认关闭修改；
2. 保存本文档到 docs/kv_paged_read_stage11_shadow_validate_results.md；
3. commit 文档；
4. 后续再决定是否 push。
```
