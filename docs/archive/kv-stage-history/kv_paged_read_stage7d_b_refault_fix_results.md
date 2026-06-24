# Stage 7D-B: refault source tracing 与 shadow validation 自我 refault 修复结果

## 1. 背景

本阶段承接 Stage 7C-G 的问题。

Stage 7C-G 已经修复了 swap-out gate 过度保守的问题，使 idle-owned blocks 能够重新进入 swap/madvise 路径。修复后，`NUM_IDLE_SEQS=1/2/3` 时，`swapped_blocks` 可以随 idle seq 数量增长：

```text
NUM_IDLE_SEQS=1 -> swapped_blocks=17
NUM_IDLE_SEQS=2 -> swapped_blocks=34
NUM_IDLE_SEQS=3 -> swapped_blocks=51
```

但是当时仍然存在一个关键异常现象：

```text
whole-KV resident drop / swapped nonresident 容量没有随 swapped_blocks 线性增长。
```

具体表现为：

```text
NUM_IDLE_SEQS=1:
  swapped_blocks=17
  swapped_nonresident≈63.75 MiB

NUM_IDLE_SEQS=2:
  swapped_blocks=34
  swapped_nonresident≈63.75 MiB
  swapped_resident≈63.75 MiB

NUM_IDLE_SEQS=3:
  swapped_blocks=51
  swapped_nonresident≈63.75 MiB
  swapped_resident≈127.50 MiB
```

也就是说，blocks 逻辑上已经处于 `SWAPPED` 状态，但旧的 SWAPPED KV pages 在后续运行中又重新变成 resident。该问题被定义为：

```text
SWAPPED blocks 能换出，但守不住 non-resident。
```

因此 Stage 7D 的目标是定位这些 SWAPPED pages 被重新触碰 / refault 的来源。

---

## 2. Stage 7D-A: refault source tracing

### 2.1 方法

Stage 7D-A 引入 debug-only refault tracing：

```text
mprotect(PROT_NONE) + SIGSEGV handler
```

在 block swap-out + madvise 成功后，对对应 SWAPPED block 的 K/V page 执行 `mprotect(PROT_NONE)`。如果后续任何路径读取该 page，就会触发 SIGSEGV。handler 记录：

```text
step
addr
kind=K/V
layer
block
page
offset
fault_count
backtrace
```

然后恢复 page 权限并返回，让程序继续执行。

该机制只用于 debug/refault tracing，不改变普通 swap/madvise 语义。

### 2.2 7D-A 最小复现结果

测试条件：

```text
NUM_IDLE_SEQS=2
ctx=2048
parallel=4
warmup=256
prefetch pressure mode=high
TRACE_MAX=128
```

结果摘要：

```text
max_swapped_blocks=34
max_idle_owned_blocks=34
max_madvise_calls=41
max_madvise_mib=153.750

max_kv_mincore_swapped_block_count=34
max_kv_mincore_swapped_resident_mib=127.500
max_kv_mincore_swapped_nonresident_mib=63.750

kv_refault_trace_lines=128
by_kind={'K': 68, 'V': 60}
top_steps=[('257', 128)]
```

说明：

```text
swap-out 确实发生；
refault 也确实发生；
第一批 refault 全部集中在 step=257。
```

随后打开 backtrace，抓到前 4 条 refault 的调用栈：

```text
KV_REFAULT_TRACE step=257 kind=K layer=0 block=0 state=SWAPPED
KV_REFAULT_TRACE_BT
libllama.so.0(...)
llama_kv_cache::paged_shadow_validate(...)
llama_kv_cache_context::next()
llama_context::decode(...)
llama_decode(...)
llama-kv-idle-swap-resume(...)
```

结论：

```text
第一批 SWAPPED K/V page refault 并不是来自 ggml graph / attention 计算路径，
而是来自 llama_kv_cache::paged_shadow_validate()。
```

---

## 3. 问题原因

`paged_shadow_validate()` 原本用于验证 paged row remap / shadow gather 的正确性。

但原实现存在一个问题：它会遍历 K/V tensor 的行，并无条件读取原始 K/V tensor memory。典型路径包括：

```cpp
std::memcpy(shadow, base + phys * row_size, ...);
std::memcmp(..., base + r * row_size, ...);
```

其中：

```text
phys: resolved physical row
r: logical row
base: 原始 K/V tensor memory
```

问题在于：

```text
当某个 block 已经处于 SWAPPED 状态，并且已经 madvise(DONTNEED) 后，
paged_shadow_validate() 仍然读取该 block 对应的 raw K/V memory。
```

这会导致：

```text
1. OS 将对应 page refault 回 resident；
2. block state 仍然显示 SWAPPED；
3. mincore 却看到这些 pages 又 resident；
4. residency 结果被 validation 自己污染。
```

因此，Stage 7C-G 中 “SWAPPED blocks 守不住 non-resident” 的主要原因并不是 graph attention 直接读取 SWAPPED pages，而是内部 validation 逻辑自身触碰了 SWAPPED KV pages。

---

## 4. Stage 7D-B 修复

### 4.1 修改文件

```text
src/llama-kv-cache.cpp
src/llama-kv-cache.h
```

### 4.2 修改原则

修复原则：

```text
paged_shadow_validate() 不应读取 SWAPPED block 的原始 K/V tensor memory。
```

如果某一行对应的 logical block 或 resolved physical block 处于 `SWAPPED` 状态，则：

```text
跳过 byte-level 内容校验；
保留 paged_resolve / row remap 元数据路径；
不触碰 raw K/V tensor address。
```

### 4.3 具体修改

在 `paged_shadow_validate()` 中增加 row/block 状态判断：

```text
row_block_swapped(row):
  block = row / paged_block_size
  if block 越界或 paged_block_states[block] == SWAPPED:
      return true
```

在 K 与 V 的内容校验循环中，在任何 `memcpy` / `memcmp` 之前判断：

```text
row_block_swapped(phys) || row_block_swapped(r)
```

如果命中，则：

```text
跳过该 row 的 raw memory read；
统计 skipped 行数和 skipped bytes；
continue。
```

修复后：

```text
SWAPPED block 仍可参与元数据校验；
但不会再被 shadow validation 触碰原始 K/V memory；
因此不会因为 validation 自身导致 refault。
```

### 4.4 新增 telemetry

新增或输出以下统计项：

```text
paged_shadow_validate_calls
paged_shadow_validate_blocks_checked
paged_shadow_validate_swapped_blocks_skipped
paged_shadow_validate_fault_risk_skipped
paged_shadow_validate_bytes_skipped
```

含义：

| 指标                                             | 含义                       |
| ---------------------------------------------- | ------------------------ |
| `paged_shadow_validate_calls`                  | shadow validation 调用次数   |
| `paged_shadow_validate_blocks_checked`         | 实际执行 raw K/V 内容读取校验的行数   |
| `paged_shadow_validate_swapped_blocks_skipped` | 因涉及 SWAPPED block 而跳过的行数 |
| `paged_shadow_validate_fault_risk_skipped`     | 因可能触发 refault 而跳过的行数     |
| `paged_shadow_validate_bytes_skipped`          | 跳过未读取的 raw K/V 字节数       |

---

## 5. 验证一：7D-B minimal refault trace

### 5.1 测试条件

```text
NUM_IDLE_SEQS=2
TRACE_MAX=8
LLAMA_KV_PAGED_REFAULT_TRACE=1
LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE=1
```

### 5.2 结果

```text
max_step=770
max_seen_seq_count=3
max_idle_seq_count=2
max_swapped_blocks=34
max_idle_owned_blocks=34
max_madvise_calls=41
max_madvise_mib=153.750

max_swapped_block_count=34
max_swapped_resident_mib=0.000
max_swapped_nonresident_mib=127.500

kv_refault_trace_lines=0
contains_paged_shadow_validate=False
```

### 5.3 结论

修复后：

```text
1. swap-out 正常发生；
2. 34 个 SWAPPED blocks 对应 127.500 MiB KV pages 全部保持 non-resident；
3. 没有任何 KV_REFAULT_TRACE；
4. paged_shadow_validate() 不再触发 refault。
```

这说明：

```text
shadow validation 自我 refault 问题已被排除。
```

---

## 6. 验证二：7D-B residency matrix

### 6.1 测试目的

验证修复后，KV nonresident 容量是否能够随 idle seq 数量线性增长。

测试变量：

```text
LLAMA_KV_IDLE_NUM_IDLE_SEQS=1
LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
LLAMA_KV_IDLE_NUM_IDLE_SEQS=3
```

固定配置：

```text
ctx=2048
parallel=4
warmup=256
cache-type-k=f32
cache-type-v=f32
pressure mode=high
mincore=enabled
```

### 6.2 结果表

| case        | num_idle | idle_owned_blocks | swapped_blocks | madvise_mib | rss_drop_mib | whole_kv_range_drop_mib | swapped_total_mib | max_swapped_resident_mib | max_swapped_nonresident_mib | seq1_decoded | resume_decoded | real_abnormal |
| ----------- | -------: | ----------------: | -------------: | ----------: | -----------: | ----------------------: | ----------------: | -----------------------: | --------------------------: | -----------: | -------------: | ------------: |
| db_numidle1 |        1 |                17 |             17 |      93.750 |       63.723 |                  63.750 |            63.750 |                    0.000 |                      63.750 |          128 |            128 |             0 |
| db_numidle2 |        2 |                34 |             34 |     153.750 |      124.062 |                 127.500 |           127.500 |                    0.000 |                     127.500 |          128 |            128 |             0 |
| db_numidle3 |        3 |                51 |             51 |     217.500 |      185.805 |                 191.250 |           191.250 |                    0.000 |                     191.250 |          128 |            128 |             0 |

### 6.3 结果解读

修复后，核心指标呈线性增长：

```text
idle seq 数量:
  1 -> 2 -> 3

idle-owned blocks:
  17 -> 34 -> 51

swapped blocks:
  17 -> 34 -> 51

SWAPPED KV nonresident:
  63.750 MiB -> 127.500 MiB -> 191.250 MiB

whole-KV resident drop:
  63.750 MiB -> 127.500 MiB -> 191.250 MiB
```

同时：

```text
max_swapped_resident_mib=0.000
seq1_decoded=128
resume_decoded=128
real_abnormal=0
```

说明：

```text
1. 所有 SWAPPED blocks 都能保持 non-resident；
2. KV resident memory 下降随 idle KV 规模线性扩大；
3. active decode 正常；
4. resume decode 正常；
5. 未观察到真实异常。
```

---

## 7. 验证三：plain smoke

### 7.1 测试目的

验证关闭 refault tracer、关闭 mincore matrix 后，普通运行路径是否仍然正常。

### 7.2 结果摘要

```text
max_step=770
max_seen_seq_count=3
max_idle_seq_count=2
max_swapped_blocks=34
max_idle_owned_blocks=34
max_madvise_calls=41
max_rss_drop_mib=123.625

active_visible_violation=0
visible_violation_rows=0

seq1_decoded_tokens=128
seq0_resume_decoded_tokens=128
```

### 7.3 关于 active_violation_rows=272

plain smoke 中出现：

```text
active_violation_rows=272
```

该字段不作为最终 correctness failure。结合日志上下文，它出现在 resume 阶段附近：

```text
active_seq=0
paged_swap_in_calls=17
seq0_resume_decoded_tokens=128
```

更准确的解释是：

```text
seq0 resume 时，需要访问原先被 SWAPPED 的历史 KV；
系统执行 swap-in / restore；
诊断计数记录了 active-needed swapped rows 的处理过程。
```

真正用于判断 correctness 的字段是：

```text
active_visible_violation=0
visible_violation_rows=0
seq1_decoded_tokens=128
seq0_resume_decoded_tokens=128
无 fatal abnormal / assert / segv
```

因此 plain smoke 判定为通过。

---

## 8. 本阶段核心结论

Stage 7D-B 的核心结论如下：

```text
通过 refault source tracing，我们发现此前 SWAPPED KV pages 被 refault 的第一来源并不是模型 graph attention，而是内部 paged_shadow_validate() 验证逻辑无条件读取了已经 SWAPPED 的 raw K/V tensor memory。
```

修复后：

```text
paged_shadow_validate() 会跳过 SWAPPED block 的 raw K/V 内容读取，只保留安全的元数据校验路径。
```

最终结果表明：

```text
1. paged_shadow_validate() 不再触发 refault；
2. NUM_IDLE_SEQS=2 时，34 个 SWAPPED blocks 对应 127.500 MiB KV pages 可全部保持 non-resident；
3. NUM_IDLE_SEQS=1/2/3 时，KV nonresident 容量从 63.750 MiB 线性增长到 191.250 MiB；
4. whole-KV resident drop 同步从 63.750 MiB 线性增长到 191.250 MiB；
5. active decode 和 resume decode 均正常完成。
```

因此可以认为：

```text
idle KV physical resident memory release 机制已经在 controlled multi-idle benchmark 中成立。
```

---

## 9. 当前成果的准确表述

对外建议表述为：

```text
在固定 KV cache 总容量的 controlled multi-idle workload 中，我们验证了 idle KV blocks 的运行时物理页释放能力。通过 refault tracing 定位并修复 shadow validation 自我 refault 后，SWAPPED KV pages 可以稳定保持 non-resident。随着 idle seq 数量从 1 增至 3，idle-owned blocks 从 17 增至 51，KV nonresident 容量从 63.750 MiB 增至 191.250 MiB，whole-KV resident drop 同步线性增长，同时 active/resume decode 正常。
```

需要避免的表述：

```text
不要说：KV cache 分配空间减少了。
```

准确说法应为：

```text
KV cache 的物理驻留内存减少了。
```

或者：

```text
KV resident memory / process RSS 在运行时下降。
```

---

## 10. 局限性

当前阶段仍然有以下局限：

### 10.1 仍是 controlled benchmark

当前 benchmark 是自定义 controlled multi-idle benchmark：

```text
人为控制 idle seq 数量；
人为设置 warmup token；
人为安排 active/resume 时间线。
```

它不是 ShareGPT realistic workload，也不是完整 server-level workload。

因此当前结果证明的是：

```text
机制在受控多 idle seq 场景中成立。
```

尚未证明：

```text
真实对话分布 / 真实请求到达模式下的整体收益。
```

### 10.2 idle detection 仍需进一步真实化

当前通过：

```text
LLAMA_KV_IDLE_NUM_IDLE_SEQS=1/2/3
```

构造 idle seq 数量。

后续需要将其扩展为：

```text
根据真实请求是否离开当前 batch；
slot 是否长时间无新 token；
用户思考时间 / 请求暂停时间；
自动判断 request idle。
```

### 10.3 ShareGPT workload 尚未接入

后续应考虑使用 ShareGPT 或类似对话数据集，构造更真实的 workload：

```text
真实 prompt length 分布；
真实多轮上下文长度；
多用户请求到达；
用户思考时间；
idle / resume 生命周期；
server-level continuous batching。
```

这样可以从 microbenchmark 过渡到 macrobenchmark，评估真实推理场景下的：

```text
process RSS drop
KV resident drop
tokens/s
TTFT
resume latency
p50/p95 latency
madvise/restore overhead
```

---

## 11. 后续方向

建议后续分为三个方向推进：

### 11.1 固化 Stage 7D-B

将当前修改与文档提交：

```text
src/llama-kv-cache.cpp
src/llama-kv-cache.h
docs/kv_paged_read_stage7d_b_refault_fix_results.md
```

建议 commit message：

```text
kv-cache: avoid shadow validation refault of swapped blocks
```

### 11.2 auto-idle detection

将当前人为指定 idle seq 的机制进一步升级为运行时自动识别：

```text
current batch seqs -> active
seen but absent for >= idle_timeout_steps -> idle candidate
resume_pending / prefetch protected / active-owned blocks -> exclude
idle-only resident blocks -> swap/madvise candidate
```

### 11.3 ShareGPT-driven realistic workload

基于 ShareGPT 构造真实对话 workload：

```text
每个 conversation 映射为一个 seq / slot；
prompt 和 response 长度来自 ShareGPT；
模拟多用户请求到达；
模拟用户思考时间；
部分 conversation idle；
idle 一段时间后 resume；
评估真实分布下的 memory-latency tradeoff。
```

该阶段的目标是证明：

```text
该机制不只是 controlled benchmark 下成立，
也能在真实/半真实多轮对话推理 workload 中带来可观的运行时内存收益。
```

---

## 12. 最终一句话总结

Stage 7D-B 通过 refault tracing 找到了并修复了 shadow validation 自我 refault 问题。修复后，SWAPPED idle KV blocks 可以稳定保持 non-resident，且 KV resident memory 下降量随 idle seq 数量线性增长。在 controlled multi-idle benchmark 中，idle seq 从 1 增至 3 时，KV nonresident 容量从 63.750 MiB 增至 191.250 MiB，active/resume decode 均正常完成。这证明 idle KV runtime physical memory release 机制已经成立，后续应接入 auto-idle detection 与 ShareGPT-driven realistic workload 评估真实推理场景下的综合收益。
