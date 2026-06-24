# Stage 5E-0：KV cache memory attribution / no-source rough accounting

> 本文档记录 Stage 5E-0 的内存归因计划与无源码修改的粗测结果，仅为计划 + rough accounting，不包含任何源码 / examples / CMake 修改。
> 文件路径：`docs/kv_paged_read_stage5e_memory_attribution_plan.md`
> 关联文档：
> - `docs/kv_paged_read_stage5c_idle_swap_madvise_results.md`（5C madvise / current RSS 结果）
> - `docs/kv_paged_read_stage5c_scale_cumulative_rss_results.md`（5C-scale-B cumulative RSS 结果）
> - `docs/kv_paged_read_stage5d_clean_perf_tradeoff_results.md`（5D-1B clean perf tradeoff 结果）

---

## 1. Stage 5E-0 目标

通过**不改源码**的方式，做一版粗略 memory attribution，回答：

```text
1. 进程总 RSS 大概是多少？
2. 模型权重 / KV cache / compute buffer / other 各自大概占多少？
3. KV cache 理论容量是多少？
4. RSS 随 ctx-size 增长的增量是否接近理论 KV 增量？
5. 当前 Stage 5C/5D 的 RSS drop 占理论 KV 容量的大致比例是多少？
```

本阶段只做 rough accounting，不要求精确到 page-level resident。精确 page residency 留给 Stage 5E-1（mincore）。

---

## 2. 为什么需要 memory attribution

Stage 5B / 5C / 5D 已经闭环：

- Stage 5B-2：idle KV block swap-out / resume swap-in correctness（逐字节一致）；
- Stage 5C：safe idle swapped block 可 madvise，current RSS 正向下降；
- Stage 5C-scale-B：累计 RSS telemetry 显示 current RSS drop 随 madvise 规模放大（large case `rss_total_drop ≈ 57 MiB`）；
- Stage 5D-1B：clean perf matrix 显示 current RSS 有收益（large case `rss_total_drop ≈ 57.45 MiB`），standalone driver 下未观察到整体吞吐下降，主要代价是 resume first-token latency 增加。

当前缺口：

```text
我们已经知道 madvise 后 current RSS 下降了 ~57 MiB，
但还不知道 KV cache 在进程 RSS 中的大致占比。
不知道 KV resident bytes，就无法判断这 57 MiB drop 的收益上限和释放比例。
```

因此需要先把进程 RSS 的组成拆开，给 57 MiB drop 一个可解释的参照系（理论 KV 容量）。

---

## 3. 进程 RSS 的组成

基于 `llama-kv-idle-swap-resume -v` 的 info 日志（ctx=512，本阶段实测，下同），进程内存大致分为四块：

| 组成 | 来源 | ctx=512 观测/估算 | 是否随 ctx 增长 |
|---|---|---|---|
| **model weights** | `CPU_Mapped model buffer size` + `CPU_REPACK model buffer size` | 4653.80 + 3204.00 MiB ≈ 7857.8 MiB | 否（与 ctx 无关） |
| **KV cache** | `CPU KV buffer size` | 128.00 MiB | **是**（与 n_ctx 线性） |
| **compute / graph buffer** | `CPU compute buffer size` | 66.63 MiB | 本矩阵中近似常数（见 §7） |
| **allocator / runtime / other** | tokenizer cache、vocab、采样、栈、glibc allocator 等 | 余量（peak RSS 减上述三项） | 弱相关 |

> 注意：`CPU_Mapped model buffer size = 4653.80 MiB` 是 mmap 的权重，按页 fault-in，**allocated（映射）大小不等于始终 resident 的 RSS**；`CPU_REPACK 3204.00 MiB` 是 repack 后的权重副本。两者相加（≈7857.8 MiB）大于模型文件本身（4.58 GiB），因为 repack 产生了额外常驻副本。这部分是本测得 peak RSS（≈8069 MiB）的主体。

---

## 4. 理论 KV 容量公式

### 4.1 模型参数（从 GGUF metadata / `-v` 日志提取）

```text
n_layer        (llama.block_count)              = 32
n_kv_heads     (llama.attention.head_count_kv)  = 8
head_dim       (llama.attention.key_length      = 128
                = llama.attention.value_length) = 128
n_ctx_train    (llama.context_length)           = 8192
cache-type-k = f32  -> bytes_K = 4
cache-type-v = f32  -> bytes_V = 4
kv_unified   = true
parallel     = 2  (n_seq_max = 2)
```

### 4.2 公式

```text
KV theoretical bytes =
  n_layer × n_ctx × n_kv_heads × head_dim × (bytes_K + bytes_V) × slots_factor
```

**关键：`slots_factor = 1`，不是 `parallel`。**

理由（基于实际日志，不硬套 parallel）：

```text
日志:  n_seq_max = 2
       kv_unified = true
       CPU KV buffer size = 128.00 MiB  (ctx=512)
```

代入 `slots_factor=1`：

```text
32 × 512 × 8 × 128 × (4+4) × 1
= 32 × 512 × 8 × 128 × 8
= 137,438,953,472 / ... 
= 134,217,728 bytes
= 128.00 MiB   ← 与日志 CPU KV buffer size = 128.00 MiB 完全一致
```

如果错误地乘以 `parallel=2`，会得到 256 MiB，与日志的 128 MiB 不符。

因此在 **unified KV** 下，`n_ctx` 是 **所有序列共享的 cell 总预算**，KV 容量按 `n_ctx` 一次计入，**不再乘 parallel**。`--parallel 2` 只影响这 `n_ctx` 个 cell 如何在 2 条序列间切分，不改变 KV 总分配大小。

可化简为：

```text
KV theoretical bytes = 262,144 × n_ctx  (bytes)
                     = 256 KiB × n_ctx
```

### 4.3 各 ctx 理论 KV 容量

| ctx | 理论 KV bytes | 理论 KV MiB | 日志 `CPU KV buffer size` |
|---|---|---|---|
| 512  | 134,217,728   | 128.00  | 128.00 ✓ |
| 1024 | 268,435,456   | 256.00  | 256.00 ✓ |
| 2048 | 536,870,912   | 512.00  | 512.00 ✓ |
| 4096 | 1,073,741,824 | 1024.00 | 1024.00 ✓ |
| 8192 | 2,147,483,648 | 2048.00 | （未跑，理论值） |

四档理论值与日志 `CPU KV buffer size` **逐档完全一致**，确认公式与 `slots_factor=1` 正确。ctx=8192 仅列理论值，未强制运行。

---

## 5. no-source RSS 差分实验设计

### 5.1 binary 选择

选用 `build/bin/llama-kv-idle-swap-resume`，理由：

```text
1. 这是 Stage 5B-2 / 5C / 5C-scale-B / 5D-1B 全程使用的同一 driver，
   与既有 57 MiB drop 数据同源、可直接对照；
2. 它接受 --kv-unified / --parallel（实测 llama-cli 在本 build 下报
   "error: invalid argument: --kv-unified"，无法满足任务要求的 unified KV 配置）；
3. 加 -v 即可透出 model buffer / KV buffer / compute buffer 等 info-level 日志。
```

### 5.2 测量矩阵

固定参数：

```text
-m /root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
--batch-size 128
--ubatch-size 128
--seed 1
--temp 0
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 2
-v                      (透出 buffer-size info 日志)
-n 16                   (粗测只需正常启动 + 少量 decode)
LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
```

变量：

```text
ctx = 512 / 1024 / 2048 / 4096
```

每档用 `/usr/bin/time -v` 包裹采集 peak RSS。

### 5.3 是否执行

**已执行。** 四档全部运行成功（exit 0）。summary 表见 §6。

---

## 6. 采集字段与结果

summary 表：`/root/oscomp/kv_logs/stage5e_memory_attribution/rss_diff_summary.tsv`
逐档原始日志：`/root/oscomp/kv_logs/stage5e_memory_attribution/run_ctx{512,1024,2048,4096}.log`

| ctx | model_mmap MiB | model_repack MiB | kv_buffer MiB | compute MiB | peak RSS kb | peak RSS MiB |
|---|---|---|---|---|---|---|
| 512  | 4653.80 | 3204.00 | 128.00  | 66.63 | 8263096 | 8069.4 |
| 1024 | 4653.80 | 3204.00 | 256.00  | 66.63 | 8395364 | 8198.6 |
| 2048 | 4653.80 | 3204.00 | 512.00  | 66.63 | 8652140 | 8449.4 |
| 4096 | 4653.80 | 3204.00 | 1024.00 | 66.63 | 9182772 | 8967.6 |

采集到的字段：

```text
1. process peak RSS（/usr/bin/time -v "Maximum resident set size"）；
2. CPU_Mapped model buffer size / CPU_REPACK model buffer size；
3. CPU KV buffer size（= KV self size）；
4. CPU compute buffer size（graph buffer）；
5. n_seq_max / n_ctx / kv_unified（确认 slots_factor）。
```

> 说明：`/usr/bin/time -v` 给出 peak RSS（VmHWM 等价）。本 driver 默认日志级别只到 warning，无独立 final-RSS info 行；current/final RSS 的精细对照由 Stage 5C/5D 既有 telemetry 字段（`paged_swap_rss_*`）承担，本阶段不重复。VmRSS/VmHWM 若需逐时刻采样，留待 5E-1。

---

## 7. 如何解释结果

### 7.1 RSS 增量 vs 理论 KV 增量

| ctx 跨档 | 理论 KV 增量 MiB | peak RSS 增量 MiB | 比值 RSS/KV |
|---|---|---|---|
| 512 → 1024  | +128 | +129.2 | 1.01 |
| 1024 → 2048 | +256 | +250.8 | 0.98 |
| 2048 → 4096 | +512 | +518.2 | 1.01 |

**结论：peak RSS 随 ctx 的增量与理论 KV 增量高度吻合（比值 0.98–1.01）。** 这说明：

```text
1. compute buffer 在本矩阵中基本不随 ctx 变（恒为 66.63 MiB，
   因为 batch/ubatch 固定为 128，graph 形状未随 ctx 改变）；
2. model weights 与 ctx 无关（恒 4653.80 + 3204.00 MiB）；
3. 因此 ctx 扩大时 RSS 的增量主体就是 KV cache，
   KV 是 ctx 扩大时最重要的内存增量来源。
```

### 7.2 KV 在总 RSS 中的占比（rough）

以 peak RSS 为分母，KV buffer 为分子（allocated 口径）：

```text
ctx=512 : 128 / 8069  ≈ 1.6%
ctx=1024: 256 / 8199  ≈ 3.1%
ctx=2048: 512 / 8449  ≈ 6.1%
ctx=4096: 1024 / 8968 ≈ 11.4%
```

即在 8B Q4_K_M、f32 KV 配置下，**模型权重（含 repack ≈7.7 GiB）是 RSS 绝对主体**，KV 只在 ctx 放大时才逐渐变成可观占比。

---

## 8. 当前 Stage 5C/5D 的 RSS drop 放进 attribution 框架

Stage 5C-scale-B / 5D-1B 的 `rss_total_drop`（madv vs madvise 窗口净下降）：

| 档 | ctx | 理论 KV 容量 MiB | observed rss_total_drop MiB | drop / 理论 KV |
|---|---|---|---|---|
| small | 512  | 128 | ≈13.4  | ≈10.5% |
| mid   | 1024 | 256 | ≈28.4  | ≈11.1% |
| large | 2048 | 512 | ≈57.45 | ≈11.2% |

解释（克制）：

```text
1. 三档 drop / 理论KV 比例稳定在 ~10.5%–11.2%，说明 madvise 释放的
   current RSS 与理论 KV 容量保持一个稳定的小比例关系；
2. 这个 ~11% 不能解释为"释放了 11% 的 KV"——它是 observed current RSS drop，
   不是 KV resident drop（见 §9 边界）；
3. 之所以远小于 100%，符合机制预期：只有 idle 且满整页、且不波及邻居
   （Stage 5C 中 skip_neighbor 占比可观）的 SWAPPED block 才会被 madvise，
   active / 部分页 / 邻居保护页都不释放。
```

因此 large case 的 **57 MiB ≈ 理论 KV 容量(512 MiB) 的 ~11%**，这是当前可解释的参照系。要知道这 57 MiB 占 **实际 KV resident bytes** 的比例，必须先测出 KV resident（5E-1 mincore）。

---

## 9. 边界说明

必须写清的解释：

```text
1. process RSS drop ≠ KV resident drop。57 MiB 是进程 current RSS 的下降，
   不等于"KV 的常驻内存下降了 57 MiB"。
2. KV theoretical size ≠ KV resident bytes。理论 128/256/512/1024 MiB 是
   按公式算的容量上限，实际 resident 取决于已写入多少 cell、页是否 fault-in。
3. llama.cpp 日志中的 buffer size（model/KV/compute）多数是 allocated/映射大小，
   不一定是 resident RSS。尤其 CPU_Mapped 4653.80 MiB 是 mmap 映射，按页 fault-in。
4. current RSS 与 peak RSS 是不同指标。本矩阵采集的是 peak RSS(VmHWM)；
   Stage 5C/5D 的 drop 说的是 current RSS。二者不可混用。
5. Stage 5C/5D 的 57 MiB drop 只能说明 observed current RSS drop，
   不能直接说明"释放了 KV 理论容量中的 57 MiB"。
6. RSS 差分不是精确归因：ctx 变化理论上也可能影响 compute graph buffer。
   本矩阵中 compute buffer 恰好恒为 66.63 MiB（batch/ubatch 固定），
   所以差分干净；若改变 batch/ubatch 该假设不再成立。
7. 要精确知道 KV 实际 resident bytes，需要 mincore 或类似 page residency 统计（5E-1）。
```

---

## 10. 后续 Stage 5E-1：mincore 精确 resident accounting

本阶段是 rough accounting，只用 buffer-size 日志 + peak RSS 差分。下一阶段建议：

```text
1. 用 mincore(2) 对 KV buffer 的虚拟地址区间逐页查询 resident 位，
   得到 KV 实际 resident bytes（而非理论容量）；
2. 在 idle swap madvise 前后各做一次 KV-region mincore，
   得到 KV resident drop（而非进程 current RSS drop）；
3. 把 57 MiB current RSS drop 与 KV resident drop 对照，
   给出"madvise 真正释放了多少 KV 常驻页"的精确口径；
4. 这需要能拿到 KV buffer 的虚拟地址 + 长度（涉及读取/插桩 KV cache 内部指针），
   因此 5E-1 很可能需要源码侧 telemetry，超出本阶段 no-source 范围。
```

---

## 11. 边界总结（一句话）

**Stage 5E-0 用 no-source buffer-size 日志 + peak RSS 差分，确认 KV 是 ctx 扩大时的主要 RSS 增量来源（RSS/KV 增量比 0.98–1.01），并给 Stage 5C/5D 的 ~57 MiB current RSS drop 找到参照系（≈理论 KV 容量的 ~11%）；但 RSS drop ≠ KV resident drop，精确 KV resident 口径留给 5E-1 mincore。**
