# Historical Stage 12-C: ShareGPT-backed Synthetic KV Trace Replay Results

## 1. 阶段目标

本阶段目标是将 KV cache 优化评测从固定 A/B/C/D 半真实 workload 推进到更接近真实对话形态的 **ShareGPT-backed trace replay workload**。

前一阶段的 semi-real workload 已经验证：

* 多 session 生命周期可以触发 idle KV block；
* idle KV block 可以被 swap-out；
* swap-out 后可以通过 `madvise` 降低进程 RSS；
* resume 时可以 restore / prefetch，保证继续推理正确性。

但固定 A/B/C/D workload 的问题是：

* session 数、turn 数、arrival、idle、resume 都由源码逻辑固定；
* prompt 内容虽然可以来自 corpus，但对话结构不是真实多轮对话；
* 很难说明优化在真实对话形态下仍然有效。

因此，本阶段新增并验证 `kv-trace-replay` driver，用 trace 文件描述多 session / 多 turn / arrival / idle / resume / decode token 数，使 workload 可以由真实 ShareGPT 数据转换得到。

本阶段不追求真实线上流量复现，而是验证：

```text
真实 ShareGPT 对话内容 + 合成 timing / concurrency
```

下，KV cache idle swap / madvise / restore 机制是否能稳定运行，并量化其 RSS 收益、KV 容量占比和性能代价。

---

## 2. Workload 口径

本阶段使用的 workload 应明确称为：

```text
ShareGPT-backed synthetic trace
```

含义如下：

1. **对话内容和多轮结构来自真实 ShareGPT 数据。**

   * user prompt 来自 ShareGPT conversation；
   * session / turn 来自 ShareGPT 中的 user → assistant pair；
   * 每个 session 对应一个 ShareGPT conversation；
   * 每个 turn 对应一个 user message。

2. **arrival / idle / concurrency 是合成的。**

   * ShareGPT 数据本身不包含真实线上请求到达时间；
   * 因此 `arrival_ms`、`idle_ms`、session 并发关系由 converter 参数合成；
   * 这使我们可以构造 tiny、medium、long-idle 等不同压力场景。

3. **assistant 输出不是复现 ShareGPT 原回复。**

   * ShareGPT assistant 回复只用于估算 `target_decode_tokens`；
   * replay 阶段由本地 Llama 模型实际 decode；
   * 因此该 workload 用于制造 KV 压力和多会话生命周期，不用于评估输出语义质量。

4. **后续 turn 只输入新增 user message。**

   * 不把完整历史重新塞进 prompt；
   * 历史上下文通过同一个 `seq_id` 上保留的 KV cache 继承；
   * 这样才能真实触发 idle / resume / restore 语义。

---

## 3. 数据处理链路

本阶段数据链路如下：

```text
ShareGPT 原始数据
  → strict JSON 修复
  → scripts/kv_trace_from_sharegpt.py
  → trace.tsv
  → prompts/*.txt
  → llama-kv-trace-replay
```

### 3.1 原始 JSONL 问题

从 Hugging Face 导出的 ShareGPT JSONL 文件中，部分记录包含未严格转义的换行，导致：

```text
文件名是 .jsonl，但并不是严格的一行一个 JSON object。
```

直接按 JSONL 逐行解析会在第 72 行附近报错：

```text
Unterminated string
```

因此，本阶段先使用宽松 JSON decoder 从原始文本中解析出完整 object，再写成 strict JSON array：

```text
/root/oscomp/sharegpt_data/sharegpt_gpt4_first2000.strict.json
```

后续 converter 使用该 strict JSON 文件作为输入。

### 3.2 Converter 输出

converter 输出目录包括：

```text
trace.tsv
summary.json
prompts/*.txt
```

其中 `trace.tsv` 格式为：

```text
session_id    turn_id    arrival_ms    prompt_source    target_decode_tokens    idle_ms
```

`prompt_source` 使用：

```text
file:prompts/sXXXXXX_tXXXXXX.txt
```

这样可以避免把真实 prompt 内联进 TSV，避免 tab / newline / UTF-8 内容破坏 trace 格式。

---

## 4. Driver 与实现边界

本阶段新增 driver：

```text
examples/kv-trace-replay/kv-trace-replay.cpp
```

新增目标：

```text
llama-kv-trace-replay
```

该 driver 的职责是：

* 读取 trace；
* 按 virtual time 驱动 session lifecycle；
* 支持多 session / 多 turn；
* 对同一 session 复用同一 `seq_id`；
* 在 session 暂停时进入 idle；
* 在后续 turn 到达时 resume；
* 调用已有 prefetch / restore / defer 机制；
* 输出 `KV_TRACE_*` telemetry。

该 driver 不负责直接 swap-out。idle swap-out 仍由 core KV cache 逻辑在 decode maintenance 中完成。

本阶段保持边界：

```text
driver 负责构造 workload；
core 负责 KV cache swap / madvise / restore；
不接 llama-server；
不做 HTTP / k6；
不引入真实 request queue；
不修改 public API。
```

---

## 5. 主要实验配置

### 5.1 模型与基础参数

```text
model:
  /root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

binary:
  ./build/bin/llama-kv-trace-replay

ctx-size:
  4096 for main long-idle / fast-maintenance result
  8192 for larger KV-ratio validation

batch-size:
  512

ubatch-size:
  128

cache-type-k:
  f32

cache-type-v:
  f32

kv-unified:
  enabled

parallel:
  4 for tiny
  8 for medium / long-idle
```

### 5.2 S5 配置

完整 S5 配置包括：

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

调试时可额外开启：

```text
LLAMA_KV_PAGED_IDLE_TRACE=1
```

但正式性能评测中不应开启该变量，因为它会输出大量 `KV_PAGED_IDLE_TRACE` 长日志，显著污染性能和日志收尾行为。

---

## 6. Tiny real ShareGPT-backed smoke

### 6.1 Trace 配置

```text
num_sessions = 4
max_turns_per_session = 2
actual_sessions = 4
actual_turns = 7
max_decode_tokens = 48
max_user_chars = 1024
parallel = 4
```

### 6.2 结果

| case             | exit | abnormal | summary_count | all_finished | active_tps | active_decode_tokens | active_decode_ms |  rss_kb |
| ---------------- | ---: | -------: | ------------: | -----------: | ---------: | -------------------: | ---------------: | ------: |
| T0_tiny_baseline |    0 |        0 |             4 |            1 |  13.437755 |                  336 |        25004.177 | 9240304 |
| T5_tiny_s5       |    0 |        0 |             4 |            1 |  10.125248 |                  336 |        33184.373 | 8375388 |

### 6.3 换算

```text
RSS drop = 9240304 - 8375388 = 864916 KiB
RSS drop = 844.6 MiB
TPS delta = 10.125248 / 13.437755 - 1 = -24.65%
```

### 6.4 结论

tiny real ShareGPT-backed smoke 通过，说明：

* ShareGPT prompt file replay 链路可用；
* trace driver 可以正确处理真实 prompt；
* S5 在小规模真实 ShareGPT-backed workload 下可以完整完成；
* idle swap / madvise / restore 未出现 correctness 错误。

---

## 7. Medium real ShareGPT-backed smoke

### 7.1 Trace 配置

```text
num_sessions = 8
max_turns_per_session = 4
actual_sessions = 8
actual_turns = 24
max_decode_tokens = 96
max_user_chars = 2048
parallel = 8
```

### 7.2 首轮现象：idle trace 过重

在开启：

```text
LLAMA_KV_PAGED_IDLE_TRACE=1
```

时，T5 没有输出完整 `KV_TRACE_SUMMARY` / `KV_TRACE_PERF`。但 abnormal grep 为空，core safety counters 没有显示 active-visible violation 或 write-to-swapped 错误。

判断：

```text
medium trace + LLAMA_KV_PAGED_IDLE_TRACE=1 日志过重，导致性能和日志收尾行为被明显污染。
```

因此正式 smoke 应关闭 `LLAMA_KV_PAGED_IDLE_TRACE`。

### 7.3 关闭 idle trace 后结果

| case                       | exit | abnormal | summary_count | all_finished | active_tps | active_decode_tokens | active_decode_ms |  rss_kb |
| -------------------------- | ---: | -------: | ------------: | -----------: | ---------: | -------------------: | ---------------: | ------: |
| T0_medium_baseline         |    0 |        0 |             8 |            1 |   8.370995 |                 2135 |       255047.354 | 9526668 |
| T5_medium_s5_no_idle_trace |    0 |        0 |             8 |            1 |   6.858006 |                 2135 |       311314.998 | 9461672 |

### 7.4 换算

```text
RSS drop = 9526668 - 9461672 = 64996 KiB
RSS drop = 63.5 MiB
TPS delta = 6.858006 / 8.370995 - 1 = -18.1%
```

### 7.5 结论

medium real ShareGPT-backed smoke 通过，证明：

* 8-session / 24-turn 真实 ShareGPT-backed trace 可以稳定 replay；
* T5 在关闭 heavy debug trace 后可完整完成；
* 该 workload 下最终 RSS drop 较小，说明频繁 resume 会抵消 idle swap/madvise 的最终内存收益；
* medium trace 更适合用于稳定性验证，不适合作为最终内存收益展示主场景。

---

## 8. Long-idle real ShareGPT-backed probe

### 8.1 Trace 配置

为突出 KV cache 优化适用场景，构造 long-idle trace：

```text
num_sessions = 8
max_turns_per_session = 2
actual_sessions = 8
actual_turns = 14
idle_ms_low = 1500
idle_ms_high = 4000
max_decode_tokens = 64
max_user_chars = 2048
parallel = 8
```

该 trace 更接近 KV cache 优化的目标场景：

```text
多 session；
长 prompt / history；
会话中间存在较长 idle；
resume 不过于频繁。
```

### 8.2 结果

| case                 | exit | abnormal | summary_count | all_finished | active_tps | active_decode_tokens | active_decode_ms |  rss_kb |
| -------------------- | ---: | -------: | ------------: | -----------: | ---------: | -------------------: | ---------------: | ------: |
| T0_longidle_baseline |    0 |        0 |             8 |            1 |  10.922327 |                  848 |        77639.134 | 9334036 |
| T5_longidle_s5       |    0 |        0 |             8 |            1 |   7.254124 |                  848 |       116899.019 | 8599540 |

### 8.3 换算

```text
RSS drop = 9334036 - 8599540 = 734496 KiB
RSS drop = 717.281 MiB

TPS delta = 7.254124 / 10.922327 - 1 = -33.584%

active_decode_ms delta = 116899.019 / 77639.134 - 1 = +50.567%
```

### 8.4 结论

long-idle trace 下，S5 正确性通过，并获得明显 RSS 收益：

```text
RSS drop ≈ 717.3 MiB
```

但 TPS 回退较大：

```text
active_tps delta ≈ -33.6%
```

因此该 workload 证明了 aggressive reclaim 的内存潜力，但也暴露了当前 idle swap 主路径维护开销。

---

## 9. Fast-maintenance 前 KV cache 理论容量估算与收益占比

### 9.1 ctx4096 KV cache 理论容量

当前配置：

```text
model = Llama-3-8B
n_layer ≈ 32
n_kv_head ≈ 8
head_dim ≈ 128
cache-type-k = f32 = 4 bytes
cache-type-v = f32 = 4 bytes
ctx-size = 4096
```

每 token KV 占用：

```text
32 × 8 × 128 × (4 + 4)
= 262144 bytes
= 256 KiB/token
```

总 theoretical KV capacity：

```text
4096 × 256 KiB
= 1048576 KiB
= 1024 MiB
= 1 GiB
```

### 9.2 从 block 反推验证

日志中出现过：

```text
paged_cov_idle_owned_blocks=133
paged_cov_idle_owned_bytes=557842432
```

反推：

```text
557842432 / 133 = 4194304 bytes = 4 MiB/block
```

每 block 约 16 token：

```text
4 MiB / 16 = 256 KiB/token
```

这与理论估算一致。

### 9.3 Fast-maintenance 前 long-idle 收益占理论 KV 容量比例

long-idle T5 RSS drop：

```text
717.281 MiB
```

占 1 GiB KV cache：

```text
717.281 / 1024 = 70.0%
```

因此可以表述为：

```text
在 4096 ctx、f32 KV 配置下，Llama-3-8B 的 theoretical KV capacity 约为 1 GiB。fast-maintenance 前的 long-idle ShareGPT-backed trace 中，aggressive S5 将最终 RSS 降低约 717 MiB，约等于 theoretical KV capacity 的 70%。
```

该比例是 RSS drop 与 theoretical KV capacity 的对比，不表示 KV buffer capacity 本身被改变。后续第 19 节使用 mincore 进一步补齐实际 KV resident 口径。

### 9.4 总 RSS 下降比例

long-idle baseline 总 RSS：

```text
9334036 KiB = 9115.27 MiB
```

S5 RSS drop：

```text
717.281 MiB
```

占总 RSS 比例：

```text
717.281 / 9115.27 = 7.87%
```

因此本阶段应同时报告两种比例：

```text
总进程 RSS 下降约 7.9%；
释放量约等于 theoretical KV capacity 的 70%。
```

这能避免只看总 RSS 时低估 KV cache 优化效果。

---

## 10. Prefetch 消融

为判断 TPS 回退是否主要来自 prefetch，本阶段测试了三个变体：

| case              |  during-active prefetch | final sync blocks |    RSS drop | TPS delta | active_decode_ms delta |
| ----------------- | ----------------------: | ----------------: | ----------: | --------: | ---------------------: |
| T5_final_only     |                     off |                12 | 717.070 MiB |  -33.700% |               +50.829% |
| T5_light_prefetch | every 8 tokens, 1 block |                12 | 716.867 MiB |  -33.825% |               +51.115% |
| T5_more_final     |                     off |                32 | 716.754 MiB |  -34.257% |               +52.107% |

结论：

```text
prefetch 不是当前 TPS 回退的主要原因。
```

原因是：

* 关闭 during-active prefetch 后，TPS 没有明显恢复；
* 降低 prefetch 频率后，TPS 没有明显恢复；
* 增加 final sync blocks 没有带来更好 RSS，也没有改善 TPS。

因此，后续性能优化重点不应继续放在 prefetch 参数上。

---

## 11. 组件级消融

为定位主要开销来源，本阶段进一步测试了 5 个组件级变体。

### 11.1 实验结果

| case                    | 说明                               |    RSS drop | RSS drop / theoretical KV | TPS delta | active_decode_ms delta |
| ----------------------- | -------------------------------- | ----------: | ---------------: | --------: | ---------------------: |
| T1_lazy_only            | lazy tail / lazy clear           | 447.691 MiB |          43.720% |   -2.827% |                +2.909% |
| T2_paged_no_swap        | paged path, no swap              | 462.164 MiB |          45.133% |   -3.478% |                +3.603% |
| T3_idle_swap_no_madvise | idle swap, no madvise            | 462.000 MiB |          45.117% |  -27.462% |               +37.858% |
| T4_madvise_no_prefetch  | idle swap + madvise, no prefetch | 716.945 MiB |          70.014% |  -33.465% |               +50.297% |
| T5_full_s5              | full S5                          | 716.562 MiB |          69.977% |  -34.005% |               +51.527% |

### 11.2 关键观察

#### 观察 1：lazy-only 已有较好收益且低开销

```text
T1_lazy_only:
  RSS drop ≈ 447.7 MiB
  占 KV cache ≈ 43.7%
  TPS delta ≈ -2.8%
```

说明 lazy tail / lazy clear 是一个有实际价值的低开销模式。

#### 观察 2：paged read path 本身开销较小

```text
T2_paged_no_swap:
  RSS drop ≈ 462.2 MiB
  TPS delta ≈ -3.5%
```

T2 相比 T1 只增加很少开销，说明 paged path 本身不是主要 TPS 瓶颈。

#### 观察 3：idle swap 状态维护是主要瓶颈

```text
T3_idle_swap_no_madvise:
  RSS drop ≈ 462.0 MiB
  TPS delta ≈ -27.5%
```

T3 相比 T2 没有获得额外 RSS 收益，却让 TPS 从约 -3.5% 扩大到约 -27.5%。

这说明主要开销来自 idle swap 相关状态维护，而不是 `madvise` 本身。

最可能的开销来源包括：

* idle block candidate 扫描；
* active-visible 检查；
* swapped block redirect 判断；
* restore 判断；
* swapped / resident / released 状态维护；
* detailed probe counters / coverage counters；
* 每 decode step 或高频 maintenance 中的 block / row / seq ownership 遍历。

#### 观察 4：madvise 提供主要额外 RSS 收益

```text
T4_madvise_no_prefetch:
  RSS drop ≈ 716.9 MiB
  占 KV cache ≈ 70.0%
  TPS delta ≈ -33.5%
```

T4 相比 T3 将 RSS drop 从约 462 MiB 提高到约 717 MiB，说明 `madvise` 是获得额外物理内存回收的关键。

但 T4 的 TPS 进一步下降，说明 physical reclaim 也有额外成本。

#### 观察 5：full S5 相比 T4 没有明显收益

```text
T5_full_s5:
  RSS drop ≈ 716.6 MiB
  TPS delta ≈ -34.0%
```

T5 相比 T4：

* RSS drop 几乎不变；
* TPS 略差；
* prefetch 没有带来可见收益。

因此，在当前 workload 下，`T4_madvise_no_prefetch` 比 full S5 更适合作为 aggressive reclaim 配置。

---

## 12. Fast-maintenance 前的结果口径

在加入 fast-maintenance 前，本阶段形成了两个阶段性结果口径。本历史阶段推荐的 fast-maintenance V5 配置见第 17 节。

### 12.1 Low-overhead 模式

配置：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1
```

结果：

```text
RSS drop ≈ 447.7 MiB
RSS drop / theoretical KV ≈ 43.7%
TPS delta ≈ -2.8%
```

适合表述为：

```text
在真实 ShareGPT-backed long-idle workload 下，low-overhead lazy-only 模式可在 TPS 仅下降约 2.8% 的情况下，将进程 RSS 降低约 447.7 MiB，约等于 4096 ctx / f32 theoretical KV capacity 的 43.7%。
```

### 12.2 Aggressive reclaim 模式

推荐配置：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1
LLAMA_KV_PAGED=1
LLAMA_KV_PAGED_INGRAPH=1
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
LLAMA_KV_PAGED_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP=1
LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
```

不推荐默认启用 prefetch 作为该 workload 下的最终配置，因为 prefetch 未显著改善 RSS 或 TPS。

结果：

```text
RSS drop ≈ 716.9 MiB
RSS drop / theoretical KV ≈ 70.0%
TPS delta ≈ -33.5%
```

适合表述为：

```text
在真实 ShareGPT-backed long-idle workload 下，aggressive reclaim 模式可将最终 RSS 降低约 716.9 MiB，约等于 4096 ctx / f32 theoretical KV capacity 的 70.0%。但 fast-maintenance 前 idle swap 主路径维护开销较大，导致 active TPS 回退约 33.5%。该问题已在第 17 节通过 fast-maintenance V5 显著缓解。
```

---

## 13. 为什么需要构造 KV cache 占比更大的场景

fast-maintenance 前的 ctx4096 long-idle baseline 总 RSS 约 9.1 GiB，而 theoretical KV capacity 约 1 GiB。

因此，即使释放 717 MiB：

```text
对 theoretical KV capacity：约 70%
对总进程 RSS：约 7.9%
```

这说明：

```text
如果要在总 RSS 指标上更明显地展示优化效果，需要构造 KV cache 占总内存比例更大的场景。
```

可行方向包括：

* 增大 `ctx-size`，例如 8192；
* 增加 session 并发；
* 增加长 prompt / 长 history；
* 构造更长 idle；
* 使用 KV cache 更显著的模型和参数组合；
* 保持 `cache-type-k=f32`、`cache-type-v=f32` 作为压力配置，便于放大 KV 内存占比。

在当时不应立即扩大场景，因为 aggressive reclaim 的 TPS 回退仍偏大。更合理顺序是：

```text
先优化 idle swap maintenance 开销；
再构造更大 KV 占比 workload；
最后做正式 3-run median。
```

该顺序已经在本阶段后续完成：第 17 节完成 fast-maintenance V5，降低 TPS 回退；第 18 节完成 ctx8192 larger KV-ratio validation；第 19 节进一步用 mincore 补齐实际 KV resident / memory composition 口径。

---

## 14. Fast-maintenance 前的主要瓶颈

在加入 fast-maintenance 前，本阶段定位出的主要瓶颈是：

```text
不是 paged read path；
不是 prefetch；
主要是 idle swap 开启后的状态维护、active-visible 检查、swapped redirect / restore 判断。
```

具体依据：

```text
T2_paged_no_swap:
  TPS delta ≈ -3.5%

T3_idle_swap_no_madvise:
  TPS delta ≈ -27.5%
```

T3 没有额外 `madvise` 收益，却引入主要 TPS 回退。因此，后续第 17 节优先优化 idle swap maintenance，而不是继续调 prefetch。

---

## 15. Fast-maintenance 优化设计方向

组件级消融后，本阶段没有新开阶段，而是在 Stage 12-C 内继续围绕真实 ShareGPT-backed workload 优化 idle swap maintenance。以下方向构成了后续第 17 节 fast-maintenance patch 的设计基础。

### 15.1 优化目标

当时的优化目标不是进一步追求更大 RSS drop，而是：

```text
保持 RSS drop 500–700 MiB；
将 TPS delta 从约 -33% 降低到 -10% ~ -20%；
保持 correctness counters 为 0；
保持 trace replay 完整收尾。
```

### 15.2 可能优化方向

#### 1. 降低 idle swap maintenance 频率

当前实现可能在 decode 热路径中过于频繁地做 idle swap 检查。可以引入：

```text
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS
```

使 maintenance 从每步执行变为每 N token 执行一次。

#### 2. 限制每次处理 block 数

引入：

```text
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP
```

限制每次 maintenance 最多处理 K 个 candidate block，避免单步维护过重。

#### 3. 增加 minimum idle threshold

引入：

```text
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS
```

只有 seq idle 超过一定步数后才允许 swap-out，避免短 idle 下 swap / restore 抖动。

#### 4. 将 detailed counters 改为 debug-only

将以下统计降级为 debug-only：

```text
paged_swapped_redirect_probe_rows
paged_swapped_redirect_probe_swapped_rows
paged_swapped_redirect_probe_resident_rows
paged_cov_*
detailed active-visible rows/blocks counters
```

正式性能模式只保留必要 safety counters：

```text
paged_swapped_active_visible_violation
paged_write_to_swapped_block
paged_swap_madvise_failures
```

#### 5. 避免 restore 后立即 reswap

为刚 restore 的 seq 或 block 设置短保护窗口，减少：

```text
swap-out → restore → swap-out
```

带来的抖动。

### 15.3 暂不优先做的方向

暂不建议优先进行大规模重构，例如：

* 重写 block table；
* 改 public API；
* 接真实 request queue；
* 引入后台线程；
* 重写 driver；
* 改 tokenizer / ggml graph。

这些方向最终收敛为第 17 节的 fast-maintenance V5 配置。

---

## 16. Fast-maintenance 前阶段小结

在加入 fast-maintenance 前，本阶段已经完成了从 semi-real workload 到 real ShareGPT-backed trace replay 的关键推进。

主要成果包括：

1. 新增 `kv-trace-replay` driver，使多 session / 多 turn / arrival / idle / resume 可以由 trace 文件驱动；
2. 新增 ShareGPT converter，将真实 ShareGPT conversation 转换为 `trace.tsv` 和 `file:prompts/...`；
3. 解决 ShareGPT 原始 JSONL 不严格的问题，通过 strict JSON array 输入稳定转换；
4. tiny real ShareGPT-backed smoke 通过；
5. medium real ShareGPT-backed smoke 在关闭 heavy idle trace 后通过；
6. long-idle real ShareGPT-backed probe 通过，并获得约 717 MiB RSS drop；
7. theoretical KV capacity 估算显示该 RSS drop 约等于 4096 ctx / f32 theoretical KV capacity 的 70%；
8. 组件级消融显示 low-overhead lazy-only 模式可用，约 447.7 MiB RSS drop，TPS 仅下降约 2.8%；
9. aggressive reclaim 模式可释放约 70% KV 容量级别 RSS，但当前 TPS 回退约 33%；
10. 当前主要瓶颈已定位为 idle swap 状态维护 / active-visible 检查 / swapped redirect / restore 判断，而不是 prefetch。

因此，在加入 fast-maintenance 前，本阶段阶段性结论是：

```text
真实 ShareGPT-backed workload 下，KV cache 内存优化机制已经完成端到端验证。低开销模式具备较好实用性；激进模式证明了接近 KV 容量上限的内存回收潜力，但仍需要优化 idle swap maintenance 开销，才能作为最终高性能方案。
```

该问题已经在第 17 节中通过 fast-maintenance V5 继续优化，并完成 3-run median、resume first-token latency 与 diagnostic counter 验证。

## 17. Fast-maintenance optimization result

在组件级消融后，本阶段继续优化 idle swap maintenance 主路径。审计显示，原 S5 aggressive reclaim 的主要 TPS 回退来自 `set_input_paged_row_idx` 中每 decode step 重复执行的 idle swap maintenance，包括 block/cell/seq ownership 扫描、idle candidate 重建、swap-out 判断、active-visible probe 和 debug counters。

因此，本阶段加入 fast-maintenance 控制项：

```text
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES
```

默认不设置这些变量时，保持原 S5 行为。推荐 fast-maintenance 配置为：

```text
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16
```

该配置含义是：

* 关闭非必要 debug probe / coverage counters；
* 每 8 个 decode step 执行一次 idle swap maintenance；
* 每次 maintenance 最多处理 8 个 idle block swap-out；
* seq 至少 idle 16 个 step 后才允许 swap-out，避免短 idle 下 swap / restore 抖动。

需要注意，fast-maintenance 只降低 idle candidate 维护和 swap-out 频率，不跳过 correctness safety checks。active-visible restore、SWAPPED row redirect、write-to-swapped 检查等 safety path 仍保持每步执行。

### 17.1 Fast-maintenance 3-run median

在 real ShareGPT-backed long-idle trace 上进行 3-run median 验证，结果如下：

| case                                     |   ok | median TPS | median decode ms | median total wall ms | median RSS KB |    RSS drop | RSS drop / theoretical KV | TPS delta | decode ms delta |
| ---------------------------------------- | ---: | ---------: | ---------------: | -------------------: | ------------: | ----------: | ---------------: | --------: | --------------: |
| T0 baseline                              | True |  10.875037 |        77976.745 |            96200.524 |       9333632 |   0.000 MiB |           0.000% |    0.000% |          0.000% |
| V4 every8 + budget8 + debug off          | True |  10.092739 |        84020.801 |           102442.304 |       8787704 | 533.133 MiB |          52.064% |   -7.194% |         +7.751% |
| V5 every8 + budget8 + idle16 + debug off | True |  10.548073 |        80393.828 |            98810.429 |       8707064 | 611.883 MiB |          59.754% |   -3.007% |         +3.100% |

在本历史 workload 中，V5 是阶段内相对 V4 的较优配置。相比 V4，V5 同时获得了更大的 RSS drop 和更小的 TPS 回退：

```text
V4:
  RSS drop = 533.133 MiB
  TPS delta = -7.194%

V5:
  RSS drop = 611.883 MiB
  TPS delta = -3.007%
```

这说明 `MIN_IDLE_STEPS=16` 的防抖机制有效。它避免了短 idle 下过早 swap-out，减少 swap / restore 抖动，并把有限 swap-out budget 更集中地用于真正长期 idle 的 KV block。

### 17.2 Resume first-token latency

除了吞吐和 RSS，本阶段还解析了 resume first-token latency。该指标来自：

```text
KV_TRACE_SUMMARY ... resumed=<N> resume_first_avg_ms=<N>
```

它表示 session 从 idle 状态 resume 后，到恢复后首个 token 输出的平均延迟。该指标比初始 prefill first-token 更能反映 KV swap 场景下的用户感知延迟。

3-run median 结果如下：

| case                                     | resumed total | resume first weighted avg | max seq resume avg | prefetch final elapsed sum | prefetch final elapsed max | prefetch restored blocks |
| ---------------------------------------- | ------------: | ------------------------: | -----------------: | -------------------------: | -------------------------: | -----------------------: |
| T0 baseline                              |             6 |                100.251 ms |         103.217 ms |                   0.000 ms |                   0.000 ms |                        0 |
| V4 every8 + budget8 + debug off          |             6 |                101.140 ms |         104.828 ms |                  34.097 ms |                  22.448 ms |                       15 |
| V5 every8 + budget8 + idle16 + debug off |             6 |                101.493 ms |         104.076 ms |                  36.913 ms |                  25.700 ms |                       17 |

V5 相比 baseline：

```text
resume first weighted avg delta:
  101.493 ms - 100.251 ms = +1.242 ms

max seq resume avg delta:
  104.076 ms - 103.217 ms = +0.859 ms
```

因此，V5 在显著降低 RSS 的同时，resume first-token latency 基本保持稳定。final prefetch 的总耗时约 36.9 ms，单次最大约 25.7 ms，但它没有造成明显的 resume first-token 延迟恶化。

### 17.3 Updated final result

更新后的历史 Stage 12-C 推荐结果为：

```text
workload:
  real ShareGPT-backed long-idle trace

baseline:
  median active_tps = 10.875037
  median active_decode_ms = 77976.745
  median total_wall_ms = 96200.524
  median rss_kb = 9333632
  median resume_first_weighted_avg_ms = 100.251

fast-maintenance V5:
  median active_tps = 10.548073
  median active_decode_ms = 80393.828
  median total_wall_ms = 98810.429
  median rss_kb = 8707064
  median resume_first_weighted_avg_ms = 101.493

RSS drop:
  626568 KiB = 611.883 MiB

RSS drop / theoretical KV capacity:
  59.754%

TPS delta:
  -3.007%

decode ms delta:
  +3.100%

total wall delta:
  +2.713%

resume first-token weighted avg delta:
  +1.242 ms

correctness:
  exit = 0
  real_abnormal = 0
  summary_count = 8
  all_finished = 1
```

因此，本阶段最终可以表述为：

```text
在真实 ShareGPT-backed long-idle workload 下，fast-maintenance V5 配置将最终 RSS 降低约 611.9 MiB，约等于 4096 ctx / f32 theoretical KV capacity 的 59.8%；active TPS 仅下降约 3.0%，resume first-token weighted average latency 仅增加约 1.24 ms。相比原 aggressive reclaim 的约 33% TPS 回退，fast-maintenance 显著降低了 idle swap 主路径开销，证明该 KV cache 内存优化具备实际可用的性能-内存权衡。
```

### 17.4 Current stage conclusion after fast-maintenance

加入 fast-maintenance 后，本阶段结论更新为：

1. real ShareGPT-backed trace replay 已完整跑通；
2. low-overhead lazy-only 模式可释放约 447.7 MiB RSS，TPS 回退约 2.8%；
3. 原 aggressive reclaim 可释放约 716 MiB RSS，但 TPS 回退约 33%；
4. fast-maintenance V5 可释放约 611.9 MiB RSS，约等于 4096 ctx / f32 theoretical KV capacity 的 59.8%，TPS 仅回退约 3.0%；
5. fast-maintenance V5 对 resume first-token latency 影响很小，加权平均仅增加约 1.24 ms；
6. 主要优化来自 idle swap maintenance 降频、block budget、min idle steps 防抖和 debug probe gating；
7. 已在第 18 节基于该配置完成 ctx8192 larger KV-ratio validation，并在第 19 节补齐 KV resident / memory composition diagnostic。

### 17.5 Diagnostic counters and final metric checklist

除 3-run median 性能结果外，本阶段额外执行了 V5 diagnostic run，用于验证 fast-maintenance 配置下 idle swap / madvise 机制确实被触发，并确认 safety counters 没有异常。该 diagnostic run 开启 `LLAMA_KV_PAGED_IDLE_TRACE=1`，因此只用于机制和 correctness 佐证，不用于性能统计。

diagnostic run 中观察到：

```text
paged_idle_swap_enabled = 1
paged_idle_swap_out_calls = 63
paged_idle_swap_candidates = 3932
paged_swap_madvise_calls = 63
paged_swap_madvise_bytes = 247726080
paged_swap_madvise_failures = 0
paged_swapped_active_visible_violation = 0
paged_write_to_swapped_block = 0
paged_write_to_swapped_block_seq = 0
paged_idle_only_swapped_blocks = 265
paged_swap_rss_drop_last_kb = 3840
paged_swap_rss_drop_max_kb = 3840
paged_swap_rss_drop_sum_kb = 241920
```

这说明：

1. V5 fast-maintenance 下 idle swap-out 确实发生；
2. `madvise` 物理页回收路径被触发；
3. 单次 block 回收 RSS drop 约为 3840 KiB，接近一个 4 MiB KV block 的实际物理页回收量；
4. 没有出现 active-visible violation；
5. 没有出现 write-to-swapped block；
6. 没有出现 madvise failure；
7. `DEBUG_PROBES=0` 生效，非必要 detailed probe / coverage counters 被关闭。

因此，V5 的最终结果不仅包含 RSS/TPS/latency 指标，也有机制和 correctness 证据支撑。

### 17.6 Final complete metric summary

本历史阶段采用 fast-maintenance V5 配置作为推荐结果：

```text
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16
```

配合原 S5 其他开关：

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

完整指标如下。

| category    |                     metric | T0 baseline | V5 fast-maintenance |                     delta |
| ----------- | -------------------------: | ----------: | ------------------: | ------------------------: |
| workload    |                   sessions |           8 |                   8 |                      same |
| workload    |                      turns |          14 |                  14 |                      same |
| workload    |              resumed turns |           6 |                   6 |                      same |
| workload    |       active decode tokens |         848 |                 848 |                      same |
| workload    |                   ctx-size |        4096 |                4096 |                      same |
| workload    |                   parallel |           8 |                   8 |                      same |
| memory      |                 median RSS | 9333632 KiB |         8707064 KiB |               -626568 KiB |
| memory      |                   RSS drop |           0 |         611.883 MiB |              -611.883 MiB |
| memory      | RSS drop / theoretical KV |           0 |             59.754% | +59.754 percentage points |
| memory      |       total RSS drop ratio |           0 |              6.713% |  +6.713 percentage points |
| throughput  |          median active TPS |   10.875037 |           10.548073 |                   -3.007% |
| throughput  |    median active decode ms |   77976.745 |           80393.828 |                   +3.100% |
| latency     |       median total wall ms |   96200.524 |           98810.429 |                   +2.713% |
| latency     |  resume first weighted avg |  100.251 ms |          101.493 ms |                 +1.242 ms |
| latency     |         max seq resume avg |  103.217 ms |          104.076 ms |                 +0.859 ms |
| prefetch    | final prefetch elapsed sum |    0.000 ms |           36.913 ms |                +36.913 ms |
| prefetch    | final prefetch elapsed max |    0.000 ms |           25.700 ms |                +25.700 ms |
| prefetch    |      final restored blocks |           0 |                  17 |                       +17 |
| correctness |                       exit |           0 |                   0 |                      pass |
| correctness |              real_abnormal |           0 |                   0 |                      pass |
| correctness |              summary_count |           8 |                   8 |                      pass |
| correctness |               all_finished |           1 |                   1 |                      pass |
| diagnostic  |   active-visible violation |           0 |                   0 |                      pass |
| diagnostic  |     write-to-swapped block |           0 |                   0 |                      pass |
| diagnostic  |           madvise failures |           0 |                   0 |                      pass |
| diagnostic  |        idle swap-out calls |           0 |                  63 |       mechanism triggered |
| diagnostic  |              madvise calls |           0 |                  63 |       mechanism triggered |
| diagnostic  |              madvise bytes |           0 |           247726080 |       mechanism triggered |

最终完整结论为：

```text
在 real ShareGPT-backed long-idle workload 上，fast-maintenance V5 取得 3-run median：
最终 RSS 降低 611.883 MiB，
约等于 4096 ctx / f32 theoretical KV capacity 的 59.754%，
约占 baseline 总进程 RSS 的 6.713%；
active TPS 仅下降 3.007%，
active decode time 增加 3.100%，
total wall time 增加 2.713%，
resume first-token weighted average latency 仅增加 1.242 ms。
diagnostic run 进一步确认 idle swap-out 和 madvise 路径被实际触发，且 active-visible violation、write-to-swapped block、madvise failure 均为 0。
```

相比原 aggressive reclaim：

```text
原 aggressive reclaim:
  RSS drop ≈ 716 MiB
  TPS delta ≈ -33% ~ -34%

fast-maintenance V5:
  RSS drop ≈ 611.9 MiB
  TPS delta ≈ -3.0%
  resume first-token weighted avg delta ≈ +1.24 ms
```

因此，在本历史 Stage 12-C workload 中，fast-maintenance V5 在保留主要 KV cache 物理页回收收益的同时，显著降低了 idle swap 主路径开销，是该阶段的推荐配置。它不承担当前 Final F16 或 Global Route A 的 formal benchmark authority。

## 18. Larger KV-ratio ctx8192 validation

在 ctx4096 主结果之外，本阶段进一步构造更大 KV cache 占比场景，用于验证 fast-maintenance V5 在更大上下文容量下是否仍能稳定运行，并观察总进程 RSS 下降比例是否随 KV cache 占比增大而放大。

### 18.1 Motivation

ctx4096 / f32 KV 配置下，Llama-3-8B 的 theoretical KV capacity 约为：

```text
1024 MiB
```

而 baseline 总 RSS 约为 9.1 GiB。因此即使 V5 释放约 611.9 MiB RSS，总进程 RSS 下降比例仍为约 6.7%。

为验证 KV cache 占比更大时的收益，本节将 `ctx-size` 提高到 8192。此时 theoretical KV capacity 为：

```text
8192 × 256 KiB = 2048 MiB = 2 GiB
```

### 18.2 Experiment setup

实验仍使用 real ShareGPT-backed long-idle trace，保持 session / turn / token 数一致，只改变 `ctx-size`：

```text
ctx-size = 8192
parallel = 8
cache-type-k = f32
cache-type-v = f32
active decode tokens = 848
sessions = 8
turns = 14
resumed turns = 6
```

V5 fast-maintenance 配置保持不变：

```text
LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=8
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=8
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=16
```

### 18.3 ctx8192 3-run median

| case                        |   ok | median TPS | median decode ms | median total wall ms | median RSS KB | resume first weighted avg | max seq resume avg | prefetch elapsed sum | prefetch elapsed max | prefetch restored blocks |
| --------------------------- | ---: | ---------: | ---------------: | -------------------: | ------------: | ------------------------: | -----------------: | -------------------: | -------------------: | -----------------------: |
| T0 ctx8192 baseline         | True |  10.846081 |        78184.923 |            96365.403 |      10364540 |                103.225 ms |         108.933 ms |             0.000 ms |             0.000 ms |                        0 |
| V5 ctx8192 fast-maintenance | True |  10.736931 |        78979.742 |            97296.760 |       8706484 |                100.830 ms |         104.047 ms |            38.715 ms |            26.835 ms |                       17 |

Derived metrics:

```text
RSS drop:
  10364540 KiB - 8706484 KiB = 1658056 KiB = 1619.195 MiB

RSS drop / theoretical KV capacity:
  1619.195 MiB / 2048 MiB = 79.062%

total RSS drop ratio:
  1619.195 MiB / (10364540 KiB / 1024) = 15.997%

TPS delta:
  -1.006%

active decode ms delta:
  +1.017%

total wall delta:
  +0.966%

resume first-token weighted avg delta:
  -2.395 ms

max seq resume avg delta:
  -4.886 ms
```

### 18.4 Interpretation

ctx8192 结果显示，fast-maintenance V5 在更大 theoretical KV capacity 下仍然稳定：

1. `exit = 0`；
2. `real_abnormal = 0`；
3. `summary_count = 8`；
4. `all_finished = 1`；
5. resume first-token latency 没有恶化；
6. final prefetch / restore 路径被触发，median restored blocks 为 17。

与 ctx4096 主结果相比：

| setting    |     RSS drop | RSS drop / theoretical KV | total RSS drop | TPS delta |
| ---------- | -----------: | ---------------: | -------------: | --------: |
| ctx4096 V5 |  611.883 MiB |          59.754% |         6.713% |   -3.007% |
| ctx8192 V5 | 1619.195 MiB |          79.062% |        15.997% |   -1.006% |

这说明，当 KV cache 占总进程 RSS 的比例增大时，V5 的总 RSS 下降比例也显著放大。ctx8192 下，V5 将总进程 RSS 降低约 16.0%，而 TPS 仅下降约 1.0%。

### 18.5 Updated conclusion with ctx8192 validation

ctx8192 验证进一步强化了该历史 Stage 12-C 结论，但不替代当前 Final F16 或 Global Route A 的 formal benchmark：

```text
fast-maintenance V5 不仅在 ctx4096 下能以约 3% TPS 回退释放约 611.9 MiB RSS；
在 ctx8192 下还能以约 1% TPS 回退释放约 1.58 GiB RSS，
约等于 2 GiB theoretical KV capacity 的 79.1%，
约占 baseline 总进程 RSS 的 16.0%。
```

因此，本阶段最终成果可以表述为：

```text
在真实 ShareGPT-backed long-idle workload 中，fast-maintenance V5 能够稳定触发 idle KV swap-out 与 madvise 物理页回收；在 ctx4096 下释放约 611.9 MiB RSS，TPS 回退约 3.0%；在 ctx8192 下释放约 1619.2 MiB RSS，总进程 RSS 下降约 16.0%，TPS 回退仅约 1.0%。第 19 节的 mincore diagnostic 进一步确认这些 RSS 下降主要来自 KV resident pages 的减少。这证明该优化在 KV cache 占比更高的长上下文场景中收益更加明显，并具备较好的性能-内存权衡。
```
## 19. KV resident and memory composition diagnostic

前述结果主要基于进程 RSS 差分和 theoretical KV capacity 进行解释。为进一步确认 RSS 下降是否确实来自 KV cache resident pages，本节补充 `LLAMA_KV_PAGED_MINCORE=1` 诊断实验。

需要说明的是，theoretical KV capacity、KV allocated size 和 KV resident size 不是同一概念：

```text
theoretical KV capacity:
  由模型结构、ctx-size、K/V dtype 决定的理论 KV 容量上限。

KV allocated size:
  llama.cpp 实际创建的 KV buffer 大小，通常接近 theoretical KV capacity。

KV resident size:
  通过 mincore 采样得到的当前常驻物理内存中的 KV pages。
```

因此，本节重点关注 `kv_mincore_resident_bytes`，即实际运行过程中 KV cache 对 RSS 的真实贡献。该 telemetry 为只读采样，不改变 KV block 内容或调度语义。

### 19.1 Diagnostic setup

本节使用与 Stage 12-C 主结果相同的 real ShareGPT-backed long-idle trace，仅开启只读 mincore telemetry。由于 `LLAMA_KV_PAGED_MINCORE=1` 依赖 paged KV path，因此这里使用一个 diagnostic baseline：

```text
M0_no_madvise_mincore:
  paged enabled
  idle swap enabled
  madvise disabled
  lazy disabled
  mincore enabled
```

该配置不作为性能 baseline，只用于观察在未进行 madvise 物理页回收时，KV buffer 是否基本全部 resident。由于纯 T0 baseline 不启用 paged KV path，无法直接输出 `kv_mincore_*` 字段，因此本节使用 M0 作为 baseline-like KV resident 参照。

V5 diagnostic 配置为：

```text
V5_mincore:
  final V5 fast-maintenance configuration
  mincore enabled
```

也就是说：

```text
M0_no_madvise_mincore 用于估计 baseline-like KV resident；
V5_mincore 用于估计优化后的 KV resident。
```

### 19.2 ctx4096 diagnostic result

| case                  |  process RSS |    KV total | KV resident | KV resident ratio | KV resident share of total RSS | swapped nonresident |
| --------------------- | -----------: | ----------: | ----------: | ----------------: | -----------------------------: | ------------------: |
| T0 pure               | 9114.793 MiB |         N/A |         N/A |               N/A |                            N/A |                 N/A |
| M0 no-madvise mincore | 9100.367 MiB | 1023.75 MiB | 1023.75 MiB |            100.0% |                        11.232% |             0.0 MiB |
| V5 mincore            | 8503.047 MiB | 1023.75 MiB |  426.00 MiB |           41.612% |                         5.010% |           150.0 MiB |

Derived metrics:

```text
process RSS drop:
  9114.793 MiB - 8503.047 MiB = 611.746 MiB

KV resident drop:
  1023.75 MiB - 426.00 MiB = 597.75 MiB

KV resident drop ratio:
  597.75 MiB / 1023.75 MiB = 58.388%

baseline-like KV resident share of total RSS:
  1023.75 MiB / 9114.793 MiB = 11.232%

V5 KV resident share of total RSS:
  426.00 MiB / 8503.047 MiB = 5.010%
```

ctx4096 下，M0 diagnostic 显示 KV buffer 约 1023.75 MiB，且基本全部 resident。V5 后，KV resident 降至约 426.00 MiB，降低约 597.75 MiB，占 baseline-like KV resident 的约 58.4%。

这说明 ctx4096 下的约 611.7 MiB process RSS drop 主要可以由 KV resident drop 解释。

### 19.3 ctx8192 diagnostic result

| case                  |   process RSS |    KV total | KV resident | KV resident ratio | KV resident share of total RSS | swapped nonresident |
| --------------------- | ------------: | ----------: | ----------: | ----------------: | -----------------------------: | ------------------: |
| T0 pure               | 10121.414 MiB |         N/A |         N/A |               N/A |                            N/A |                 N/A |
| M0 no-madvise mincore | 10123.938 MiB | 2047.75 MiB | 2047.75 MiB |            100.0% |                        20.232% |             0.0 MiB |
| V5 mincore            |  8502.246 MiB | 2047.75 MiB |  426.00 MiB |           20.803% |                         5.010% |           150.0 MiB |

Derived metrics:

```text
process RSS drop:
  10121.414 MiB - 8502.246 MiB = 1619.168 MiB

KV resident drop:
  2047.75 MiB - 426.00 MiB = 1621.75 MiB

KV resident drop ratio:
  1621.75 MiB / 2047.75 MiB = 79.197%

baseline-like KV resident share of total RSS:
  2047.75 MiB / 10121.414 MiB = 20.232%

V5 KV resident share of total RSS:
  426.00 MiB / 8502.246 MiB = 5.010%
```

ctx8192 下，M0 diagnostic 显示 KV buffer 约 2047.75 MiB，且基本全部 resident。V5 后，KV resident 仍降至约 426.00 MiB，降低约 1621.75 MiB，占 baseline-like KV resident 的约 79.2%。

这解释了为什么 ctx8192 的总进程 RSS 下降比例明显高于 ctx4096：ctx8192 中 KV cache 在 baseline 总 RSS 中的占比从约 11.2% 提高到约 20.2%，而 V5 最终将 KV resident 压到几乎相同的 426 MiB 级别。

### 19.4 Memory composition interpretation

基于 mincore diagnostic，可以得到更清晰的内存构成解释：

| setting | baseline-like KV resident | V5 KV resident | KV resident drop | process RSS drop |
| ------- | ------------------------: | -------------: | ---------------: | ---------------: |
| ctx4096 |               1023.75 MiB |     426.00 MiB |       597.75 MiB |      611.746 MiB |
| ctx8192 |               2047.75 MiB |     426.00 MiB |      1621.75 MiB |     1619.168 MiB |

可以看到，process RSS drop 与 KV resident drop 高度一致。因此，本阶段 RSS 下降不是单纯的理论容量推算，而是能够通过 mincore 直接观察到 KV resident pages 的减少。

模型权重、compute buffer、runtime allocator、线程栈、shared libraries 等非 KV 内存并不是本优化直接作用对象。它们构成总 RSS 的背景项，约为 8 GiB 级别。ctx-size 增大主要增加 KV cache resident，而 V5 能够将最终 KV resident 压到约 426 MiB。因此 ctx4096 和 ctx8192 的 V5 最终 RSS 都稳定在约 8.5 GiB 附近。

### 19.5 About swapped nonresident bytes

V5 diagnostic 中，`kv_swapped_nonresident_mib` 为 150 MiB。这个数值不等于全部释放的 KV resident。

全部 KV nonresident 可以由以下差值得到：

```text
ctx4096:
  1023.75 MiB - 426.00 MiB = 597.75 MiB

ctx8192:
  2047.75 MiB - 426.00 MiB = 1621.75 MiB
```

其中，`kv_swapped_nonresident_mib = 150 MiB` 表示当前处于 swapped block 状态并且 mincore 采样为 nonresident 的 KV block。剩余 nonresident KV 区域主要来自 lazy tail / lazy clear 避免未使用 KV 区域被物理提交，以及 madvise 后不再 resident 的 KV pages。

因此，正确解释是：

```text
V5 使最终 KV resident 从 baseline-like 的 1023.75 / 2047.75 MiB 降至约 426.00 MiB；
其中当前 swapped-block nonresident 为 150 MiB；
整体 KV resident drop 分别为 597.75 MiB 和 1621.75 MiB。
```

### 19.6 Updated conclusion

KV resident diagnostic 补齐了本阶段的内存构成解释链：

```text
ctx4096:
  baseline-like KV resident ≈ 1023.75 MiB
  V5 KV resident ≈ 426.00 MiB
  KV resident drop ≈ 597.75 MiB
  process RSS drop ≈ 611.75 MiB

ctx8192:
  baseline-like KV resident ≈ 2047.75 MiB
  V5 KV resident ≈ 426.00 MiB
  KV resident drop ≈ 1621.75 MiB
  process RSS drop ≈ 1619.17 MiB
```

这说明 V5 的主要收益确实来自 KV cache resident pages 的减少。随着 ctx-size 从 4096 增至 8192，baseline-like KV resident 从约 1 GiB 增至约 2 GiB，而 V5 最终 KV resident 仍维持在约 426 MiB。因此，总进程 RSS drop 从约 6.7% 提高到约 16.0%。

该结果进一步证明：在长上下文、多 session、idle/resume 的 KV memory pressure 场景中，fast-maintenance V5 能够稳定降低 KV resident memory，并且当 KV cache 占总内存比例更高时，总 RSS 收益更明显。
