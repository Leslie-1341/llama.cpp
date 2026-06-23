# Stage 12-C: Real ShareGPT-backed KV Trace Replay Results

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
  4096

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

## 9. KV cache 容量估算与收益占比

### 9.1 KV cache 理论容量

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

总 KV cache 容量：

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

### 9.3 long-idle 收益占 KV cache 容量比例

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
在 4096 ctx、f32 KV 配置下，Llama-3-8B 的 KV cache 容量约为 1 GiB。long-idle ShareGPT-backed trace 中，S5 将最终 RSS 降低约 717 MiB，相当于 KV cache 总容量的约 70%。
```

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
释放量约等于 KV cache 容量的 70%。
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

| case                    | 说明                               |    RSS drop | KV capacity drop | TPS delta | active_decode_ms delta |
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

## 12. 当前最优结果口径

本阶段可以形成两个结果口径。

### 12.1 Low-overhead 模式

配置：

```text
LLAMA_KV_LAZY_TAIL=1
LLAMA_KV_LAZY_CLEAR=1
```

结果：

```text
RSS drop ≈ 447.7 MiB
占 KV cache 容量 ≈ 43.7%
TPS delta ≈ -2.8%
```

适合表述为：

```text
在真实 ShareGPT-backed long-idle workload 下，low-overhead lazy-only 模式可在 TPS 仅下降约 2.8% 的情况下，将进程 RSS 降低约 447.7 MiB，约等于 4096 ctx / f32 KV cache 容量的 43.7%。
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
占 KV cache 容量 ≈ 70.0%
TPS delta ≈ -33.5%
```

适合表述为：

```text
在真实 ShareGPT-backed long-idle workload 下，aggressive reclaim 模式可将最终 RSS 降低约 716.9 MiB，约等于 4096 ctx / f32 KV cache 容量的 70.0%。但当前 idle swap 主路径维护开销较大，导致 active TPS 回退约 33.5%，后续仍需继续优化 maintenance 策略。
```

---

## 13. 为什么需要构造 KV cache 占比更大的场景

当前 long-idle baseline 总 RSS 约 9.1 GiB，而 KV cache 容量约 1 GiB。

因此，即使释放 717 MiB：

```text
对 KV cache 容量：约 70%
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

但当前不应立即扩大场景，因为 aggressive reclaim 的 TPS 回退仍偏大。更合理顺序是：

```text
先优化 idle swap maintenance 开销；
再构造更大 KV 占比 workload；
最后做正式 3-run median。
```

---

## 14. 当前瓶颈

本阶段已经定位出当前主要瓶颈：

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

T3 没有额外 `madvise` 收益，却引入主要 TPS 回退。因此下一步应优化 idle swap maintenance，而不是继续调 prefetch。

---

## 15. 后续优化方向

后续仍属于 Stage 12-C 的结果完善，不需要单独新开阶段。建议继续围绕真实 ShareGPT-backed workload 完善最终成果。

### 15.1 优化目标

目标不是进一步追求更大 RSS drop，而是：

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

当前更适合做最小 fast-maintenance patch。

---

## 16. 阶段结论

本阶段完成了从 semi-real workload 到 real ShareGPT-backed trace replay 的关键推进。

主要成果包括：

1. 新增 `kv-trace-replay` driver，使多 session / 多 turn / arrival / idle / resume 可以由 trace 文件驱动；
2. 新增 ShareGPT converter，将真实 ShareGPT conversation 转换为 `trace.tsv` 和 `file:prompts/...`；
3. 解决 ShareGPT 原始 JSONL 不严格的问题，通过 strict JSON array 输入稳定转换；
4. tiny real ShareGPT-backed smoke 通过；
5. medium real ShareGPT-backed smoke 在关闭 heavy idle trace 后通过；
6. long-idle real ShareGPT-backed probe 通过，并获得约 717 MiB RSS drop；
7. KV cache 容量估算显示该 RSS drop 约等于 4096 ctx / f32 KV cache 容量的 70%；
8. 组件级消融显示 low-overhead lazy-only 模式可用，约 447.7 MiB RSS drop，TPS 仅下降约 2.8%；
9. aggressive reclaim 模式可释放约 70% KV 容量级别 RSS，但当前 TPS 回退约 33%；
10. 当前主要瓶颈已定位为 idle swap 状态维护 / active-visible 检查 / swapped redirect / restore 判断，而不是 prefetch。

因此，本阶段当前最重要结论是：

```text
真实 ShareGPT-backed workload 下，KV cache 内存优化机制已经完成端到端验证。低开销模式具备较好实用性；激进模式证明了接近 KV 容量上限的内存回收潜力，但仍需要优化 idle swap maintenance 开销，才能作为最终高性能方案。
```

后续仍应在 Stage 12-C 内继续完善，而不是另开新阶段：

```text
先优化 idle swap maintenance 性能；
再构造 KV cache 占总 RSS 比例更大的 workload；
最后做正式 3-run median 和最终汇报。
```
