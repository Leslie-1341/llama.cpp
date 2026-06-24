# KV Paged Read — Stage 6A Prefetch Design Plan

> 本文档为 Stage 6 的设计勘察阶段（Stage 6A）。本轮只读源码与已有文档，仅新增本 doc，不改任何源码。
> 关联文档：
> - [docs/kv_paged_read_stage5_summary_and_roadmap.md](kv_paged_read_stage5_summary_and_roadmap.md)
> - [docs/kv_paged_read_stage5g_high_idle_perf_tradeoff_results.md](kv_paged_read_stage5g_high_idle_perf_tradeoff_results.md)
> - [docs/kv_paged_read_stage5g_high_idle_coverage_results.md](kv_paged_read_stage5g_high_idle_coverage_results.md)
> - 补充参考：[docs/kv_paged_read_stage5d_clean_perf_tradeoff_results.md](kv_paged_read_stage5d_clean_perf_tradeoff_results.md)、[docs/kv_paged_read_stage5f_releasable_coverage_results.md](kv_paged_read_stage5f_releasable_coverage_results.md)

---

## 1. 背景

Stage 5 已经把 idle KV 生命周期闭环：idle seq / request 进入 idle 后，释放其独占 KV resident pages（swap-out + madvise）；恢复时再 swap-in，保证 correctness。

Stage 5 的核心结论：

```text
1. idle KV swap + madvise 不破坏 correctness；
2. low-idle coverage workload 下 release ratio 约 11.7%；
3. high-idle coverage workload 下 release ratio 可提高到约 35.2%；
4. 约 12% 不是机制上限，而是 idle-owned KV coverage 低导致；
5. 主要性能代价集中在 resume first-token latency；
6. 当前 resume swap-in 是同步路径，首字延迟仍偏高。
```

Stage 5G-2 的 perf tradeoff（3-run median）：

```text
mid_12p: madv RSS drop ≈ 28.39 MiB，resume first-token latency vs base +19.14 ms
mid_31p: madv RSS drop ≈ 70.88 MiB，resume first-token latency vs base +42.86 ms
mid_37p: madv RSS drop ≈ 86.05 MiB，resume first-token latency vs base +54.33 ms
```

Stage 6 的目标：

```text
通过 prefetch / async swap-in，把恢复 IO / page fault 成本从用户 resume first-token 路径中移出，
在保持 RSS drop 的前提下降低 resume first-token latency。
```

## 2. Stage 5 暴露的问题

Stage 5G-2 把代价定位得很清楚（见 5G-2 §6.2/6.3）：

- 释放规模越大（idle coverage 越高），resume first-token latency 增量越大；
- `base -> nomadv` 的增量来自 swap-out / swap-in 机制本身（状态恢复 + read_cell + tensor_set）；
- `nomadv -> madv` 的增量来自 madvise 释放后首次访问触发的 page fault / resident recovery。

也就是说，首字延迟的成本几乎全部落在 **seq0 resume 的第一个 `llama_decode`** 上：在该 decode 构图阶段，所有被 swap-out / madvise 掉的 KV blocks 必须被同步恢复回 RESIDENT。这正是 prefetch 想要搬离的工作。

## 3. 当前同步 resume 恢复路径

以下行号基于本轮阅读的 [src/llama-kv-cache.cpp](../src/llama-kv-cache.cpp) 与 [src/llama-kv-cache.h](../src/llama-kv-cache.h)。

### 3.1 idle block：RESIDENT -> SWAPPED / RELEASED

- idle 判定与 swap-out 决策都发生在 `llama_kv_cache::set_input_paged_row_idx(...)`（[src/llama-kv-cache.cpp:3797](../src/llama-kv-cache.cpp#L3797) 起），即构建 paged row_idx 输入的同一函数里。
- idle gate 标志 `idle_swap_ready` 在 [src/llama-kv-cache.cpp:3833](../src/llama-kv-cache.cpp#L3833) 计算；madvise gate `idle_swap_madvise_ready` 在 [src/llama-kv-cache.cpp:3847](../src/llama-kv-cache.cpp#L3847)。
- 在 idle-trace 循环中，`cold_not_in_read_window` 且仍 RESIDENT 的 block 进入 `safe_swap_candidates`（[src/llama-kv-cache.cpp:4154](../src/llama-kv-cache.cpp#L4154)）。当 `paged_idle_swap_requested` 成立、block 经过 nonidentity-remap、仍 RESIDENT 时，调用 `paged_swap_out_block(block, idle_swap_madvise_ready)`（[src/llama-kv-cache.cpp:4170](../src/llama-kv-cache.cpp#L4170)）。
- `paged_swap_out_block(...)`（[src/llama-kv-cache.cpp:1872](../src/llama-kv-cache.cpp#L1872)）把 block 写入 `kv_swap_store`，置 `paged_block_states[block] = SWAPPED`（[src/llama-kv-cache.cpp:1963](../src/llama-kv-cache.cpp#L1963)），并在 `do_madvise` 为真时调用 `paged_madvise_block(...)`（[src/llama-kv-cache.cpp:1975](../src/llama-kv-cache.cpp#L1975)）把页交还内核。
- 纯 RELEASED（madvise-only，无 swap backend）路径走 `paged_release_blocks(n_kv)`（[src/llama-kv-cache.cpp:3215](../src/llama-kv-cache.cpp#L3215)，从 `apply()` 的 [src/llama-kv-cache.cpp:5493](../src/llama-kv-cache.cpp#L5493) 调用），状态置 RELEASED。

### 3.2 resume：SWAPPED / RELEASED -> RESIDENT

- 读路径恢复：`paged_check_read_resident(phys, active)`（定义 [src/llama-kv-cache.cpp:1829](../src/llama-kv-cache.cpp#L1829)），从 `set_input_paged_row_idx` 的 [src/llama-kv-cache.cpp:3984](../src/llama-kv-cache.cpp#L3984) 对**每一行** row_idx 调用一次。
  - 若 block 为 SWAPPED：调用 `paged_swap_in_block(physical_block)`（[src/llama-kv-cache.cpp:1849](../src/llama-kv-cache.cpp#L1849)）。
  - 若 block 为 RELEASED 且被读到：计为 `paged_release_violation`（[src/llama-kv-cache.cpp:1840](../src/llama-kv-cache.cpp#L1840)）——说明 RELEASED 不应再被读，恢复语义主要由 SWAPPED 承担。
- 写路径恢复：`paged_ensure_write_resident(phys_cell)`（定义 [src/llama-kv-cache.cpp:1800](../src/llama-kv-cache.cpp#L1800)），从 `set_input_k_idxs` / `set_input_v_idxs`（[src/llama-kv-cache.cpp:3751](../src/llama-kv-cache.cpp#L3751)/[3772](../src/llama-kv-cache.cpp#L3772)/[3788](../src/llama-kv-cache.cpp#L3788)）调用；RELEASED 直接翻 RESIDENT，SWAPPED 调 `paged_swap_in_block`。
- 实际搬运：`paged_swap_in_block(physical_block)`（[src/llama-kv-cache.cpp:2009](../src/llama-kv-cache.cpp#L2009)）逐 cell 从 `kv_swap_store->read_cell(...)`（[src/llama-kv-cache.cpp:2090](../src/llama-kv-cache.cpp#L2090)）读入 staging，再 `ggml_backend_tensor_set(...)`（[src/llama-kv-cache.cpp:2115](../src/llama-kv-cache.cpp#L2115)）写回 K/V 张量，最后置回 RESIDENT。

### 3.3 恢复是否在 seq0 resume decode 的同步路径上

是。`set_input_paged_row_idx` / `set_input_k_idxs` / `set_input_v_idxs` 都是图输入构建回调，运行在 `llama_decode()` 内部。seq0 resume 的第一个 `llama_decode`（driver 中 [examples/kv-idle-swap-resume/idle-swap-resume.cpp:215](../examples/kv-idle-swap-resume/idle-swap-resume.cpp#L215) 的 `seq0-resume-decode`，其耗时计入 [idle-swap-resume.cpp:223](../examples/kv-idle-swap-resume/idle-swap-resume.cpp#L223) 的 `seq0_resume_first_token_ms`）在构图时会逐行触发 `paged_check_read_resident -> paged_swap_in_block`，因此所有 read_cell IO + tensor_set + madvise 后的 page fault 全部串行落在首字延迟里。

### 3.4 关键函数 / 标志位清单

```text
swap-out:      paged_swap_out_block (1872)，paged_swap_out_window (2842)，paged_release_blocks (3215)
swap-in:       paged_swap_in_block (2009)
ensure 读恢复: paged_check_read_resident (1829)  <- set_input_paged_row_idx:3984
ensure 写恢复: paged_ensure_write_resident (1800) <- set_input_k_idxs/v_idxs:3751/3772/3788
状态枚举:      paged_block_state { RESIDENT, RELEASED, SWAPPED }  (llama-kv-cache.h:478)
状态数组:      paged_block_states
gate 标志:     idle_swap_ready (3833)，idle_swap_madvise_ready (3847)，paged_idle_swap_requested
backend:       kv_swap_store->read_cell / write，ggml_backend_tensor_set
madvise:       paged_madvise_block (1975 调用点)
```

## 4. prefetch 的目标与非目标

目标：

```text
prefetch 不是为了进一步降低 RSS，而是在保持已有 RSS drop 的前提下，
提前把"即将 resume 的 seq 所拥有的" swapped/released KV blocks 恢复成 RESIDENT，
从而把 read_cell IO + tensor_set + page fault 从 resume first-token 同步路径中移走，
降低 seq0_resume_first_token_ms。
```

非目标：

```text
1. 不改变 swap-out / madvise 的释放策略与覆盖范围；
2. 不追求更高 release ratio；
3. 不在 Stage 6A 改任何源码；
4. 不全局预取所有 idle blocks（见 §6）。
```

## 5. prefetch 触发时机

至少四种可选时机：

```text
1. seq 从 idle -> pending：调度器把某 idle seq 标记为"将要恢复"时立即预取其 blocks。
2. 用户输入到达、但 llama_decode 尚未开始前：在 driver / server 收到 resume 请求与首个 decode 之间预取。
3. scheduler 预测某 idle seq 即将恢复：基于启发式（最近活跃、优先级）提前预取，最具投机性。
4. active seq decode 期间后台预取：与 seq1 active decode 计算重叠，把恢复时间藏进计算窗口（Stage 6C 方向）。
```

Stage 6B 应优先选择：

```text
synchronous prefetch probe before resume decode（对应时机 2）
```

理由：它最小、可控、容易验证——只是在 seq0 resume 循环之前显式触发一次"把 seq0 需要的 blocks 恢复成 RESIDENT"的探针，不引入线程，不改释放策略，时间点紧贴 resume，RSS 回升窗口可控且可度量。

## 6. prefetch 对象选择

只预取：

```text
即将 resume 的那个 seq 所拥有的 swapped / released blocks。
```

不要全局预取所有 idle blocks，原因：

```text
1. 全局预取会提前把其它仍处于 idle 的 seq 的 RSS 收益吃回去；
2. 浪费 IO / 内存在不会马上被读的 block 上；
3. 与"保持 RSS drop"的目标冲突。
```

实现上，目标 block 集合即"该 seq 拥有、且当前状态为 SWAPPED（或 RELEASED）"的物理 block——与 §3.1 swap-out 时记录的集合对称。Stage 6B 可借助现有 per-block `paged_block_states` 与 seq 归属信息（idle-trace 已统计 single/multi-seq blocks）来圈定。

## 7. Stage 6B: synchronous prefetch probe 方案

### 7.1 实验思想

```text
在 seq0 resume loop（idle-swap-resume.cpp 的 seq0_resume_total_t0 之前）插入一次显式
prefetch / ensure-resident / touch 调用，把 seq0 需要的 swapped/released KV blocks
提前恢复成 RESIDENT；然后再测 seq0_resume_first_token_ms。
```

预取探针在概念上等价于"对 seq0 拥有的每个非 RESIDENT block 主动调用一次 swap-in"，使首个 resume decode 构图时 `paged_check_read_resident` 命中的全是 RESIDENT，从而首字 decode 不再承担 IO。

### 7.2 对比模式

```text
base                       : paged + nonidentity + idle_trace，不开 swap/madvise
madv_no_prefetch           : idle swap-out + madvise + 同步 resume swap-in（Stage 5 现状）
madv_prefetch_before_resume: 在 resume 前显式 prefetch 后再 resume
```

### 7.3 测试 case

```text
mid_12p: ctx=1024, n=128
mid_31p: ctx=1024, n=320
mid_37p: ctx=1024, n=384
```

### 7.4 重点指标

```text
1. seq0_resume_first_token_ms     （核心：是否下降）
2. seq0_resume_total_ms
3. total_wall_ms
4. tokens_per_second
5. RSS drop before prefetch       （prefetch 前的释放收益）
6. RSS after prefetch             （prefetch 引入的回升）
7. RSS after resume
8. seq0_equal / seq1_equal        （correctness 不变）
```

### 7.5 预期现象

```text
1. madv_prefetch_before_resume 的 first-token latency 应低于 madv_no_prefetch；
2. prefetch 会让 RSS 在 resume 前提前回升；
3. 若 prefetch 时机足够贴近 resume，则总体内存收益窗口仍有意义（释放-回升间隔越短，收益越实）。
```

## 8. Stage 6C: async prefetch overlap 方向

设计方向（Stage 6A 不要求实现）：

```text
async prefetch overlapping with active request decode。
```

思想：

```text
在 seq1 active decode 仍在运行时，用后台路径恢复 seq0 的 idle KV blocks；
若恢复耗时被 active decode 的计算覆盖，则 seq0 resume first-token latency 下降，
且用户等待路径变短——恢复 IO 被藏进了原本就要花的计算时间里。
```

Stage 6C 需要解决的额外问题（不在 6B 范围）：线程模型、与 `kv_swap_store` / `ggml_backend_tensor_set` 的并发安全、预取与 active decode 对同一 block 的竞争、以及恢复完成与状态翻转的可见性。

## 9. 评估指标

沿用 driver 的 `KV_IDLE_SWAP_RESUME_PERF` marker（[idle-swap-resume.cpp:250](../examples/kv-idle-swap-resume/idle-swap-resume.cpp#L250)）：

```text
首要：seq0_resume_first_token_ms（prefetch 前后对比）
其次：seq0_resume_total_ms，total_wall_ms，tokens_per_second
内存：RSS drop before prefetch / RSS after prefetch / RSS after resume
正确性：seq0_equal，seq1_equal，exit_code，real_abnormal
```

判定 prefetch 有效的标准：

```text
madv_prefetch_before_resume.seq0_resume_first_token_ms
    显著低于 madv_no_prefetch，且
correctness 不变（seq0_equal/seq1_equal/exit_code/real_abnormal 全 0），且
RSS drop 在 prefetch 前仍然存在（释放确有其事，prefetch 只是把回升点前移）。
```

## 10. 风险与边界

```text
1. prefetch 太早会提前吃回 RSS drop —— 释放与回升之间的时间窗越长，内存收益越被稀释；
2. prefetch 错误对象（非该 seq 拥有 / 不会被读的 block）会浪费内存和 IO；
3. async prefetch（6C）需处理线程安全 / 状态竞争 / 完成可见性；
4. 预取后 block 状态（paged_block_states）必须与 resident 实况一致，否则 paged_check_read_resident
   会重复 swap-in 或误判 release_violation；
5. prefetch 不能改变 seq0 / seq1 输出 —— 它只是把同步恢复提前，不得触碰张量内容或 row 映射语义。
```

## 11. 一句话结论

当前 resume swap-in 完全发生在 seq0 首个 `llama_decode` 的同步构图路径（`set_input_paged_row_idx -> paged_check_read_resident -> paged_swap_in_block`）上，首字延迟随释放规模线性增大；Stage 6B 应以"resume 前对该 seq 拥有的 swapped 块做一次同步 prefetch probe"为最小、可控、可验证的起点，把恢复 IO 从首字路径移走，再在 Stage 6C 探索与 active decode 重叠的 async prefetch。
