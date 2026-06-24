# Stage 5C：idle swap madvise / current RSS 结果

> 本文档记录 Stage 5C 的实验结果与 telemetry 解释，仅为结果归档，不包含源码修改。
> 文件路径：`docs/kv_paged_read_stage5c_idle_swap_madvise_results.md`
> 关联文档：
> - `docs/kv_paged_read_stage5c_idle_swap_release_plan.md`（5C 计划）
> - `docs/kv_paged_read_stage5b2_resume_swapin_results.md`（5B-2 resume 结果）
> - `docs/kv_paged_read_stage5b1_idle_swap_out_results.md`（5B-1 swap-out 结果）

---

## 1. Stage 5C 目标

在 Stage 5B correctness 已闭环（swap-out → 保持 SWAPPED → resume swap-in，逐字节一致）的基础上，验证：

- safe idle swapped block 是否可以执行 `madvise(MADV_DONTNEED)` 释放物理页；
- 释放后 resume 是否仍能正确 swap-in，输出与 baseline 逐字节一致；
- 是否能观测到 **current RSS** 的正向下降。

目标只承诺 **current RSS**，不承诺 peak RSS。

---

## 2. 代码行为摘要

- **新增 debug-only env `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1`**：作为 idle swap-out 路径是否启用 madvise 的 gate。
- **默认关闭时行为等同 Stage 5B-2**：不设置该 env 时，idle swap-out 仍走 `paged_swap_out_block(block, false)`，correctness 路径与 5B-2 完全一致。
- **gate 条件**：`idle_swap_madvise_ready = idle_swap_ready && paged_idle_swap_madvise_requested`。即必须 `LLAMA_KV_PAGED_IDLE_SWAP=1`（含 SWAP / NONIDENTITY / IDLE_TRACE / backing store 全部前置）且 `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1` 才启用，不绕过 `idle_swap_ready`。
- **开启后复用 `paged_swap_out_block(block, true)`**：safe idle candidate 路径把第二参数切到 `idle_swap_madvise_ready`，触发已有 `do_madvise=true` 分支与 `paged_madvise_block`，**不新增 madvise 实现**。
- **madvise 不改变状态机**：madvise 仅在 swap-out 成功（block 已置 `SWAPPED`）之后执行，且 **保持 block 为 `SWAPPED`**，不进入 `RELEASED`。`RELEASED` 仅由 `paged_release_blocks()` 设置，与本路径无关。resume 时仍走 `SWAPPED → RESIDENT` swap-in 路径。
- **misconfig 行为**：`LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1` 但 idle swap 未 ready 时，打印一次 warning、madvise 禁用、输出不变、不 crash。
- 不改 graph / kernel / get_k / get_v / mask / examples / CMake。

---

## 3. 实验配置

实验 driver：`build/bin/llama-kv-idle-swap-resume`（复用 Stage 5B-2 的 resume driver，未新增）。

公共参数：

```text
-m /root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
-n 64
--ctx-size 512
--batch-size 128
--ubatch-size 128
--seed 1
--temp 0
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 2
```

determinism：greedy、`temp=0`、固定 seed、`kv_unified=true`，三组同 seed / prompt / ctx / n，除目标 env 外完全同构。

---

## 4. 三组矩阵说明

| 组 | 环境变量 | 含义 |
|---|---|---|
| **base** | `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 LLAMA_KV_PAGED_IDLE_TRACE=1 LLAMA_KV_PAGED_TRACE=1` | 不开 backing store，不做 idle swap。输出基线 |
| **nomadv** | base + `LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=1` | 启用 idle swap-out + resume swap-in，但不 madvise（= 5B-2 行为） |
| **madv** | nomadv + `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1` | idle swap-out 后对 SWAPPED block madvise，再 resume swap-in |

---

## 5. Correctness 结果

三组（base / nomadv / madv）全部逐字节一致。

### 5.1 full stdout

```text
base_vs_nomadv_equal=0
base_vs_madv_equal=0
sha256（三组一致）= 0c2fada2df57cb797fcc5e3b6950c21b65a6197c6b54a89d4a02543b8fc148dc
```

### 5.2 seq1 active 段

```text
seq1_base_vs_nomadv_equal=0
seq1_base_vs_madv_equal=0
sha256（三组一致）= 371c5a41831ad024c928df9d629dda9c51181f485fe6e07261f3269c38831419
```

### 5.3 seq0 resume 段

```text
seq0_base_vs_nomadv_equal=0
seq0_base_vs_madv_equal=0
sha256（三组一致）= 873ea6bd90bcc548a5ceb6b78ae356df1900218b1e8906f4ccffc94839907b31
```

> seq0 resume 段单独比对是关键：swap-in 缺陷只会反映在 seq0 输出上。三组 resume 段 sha256 一致，证明 madvise 后 resume swap-in 完全恢复了 KV 内容。

---

## 6. Telemetry 结果

### 6.1 nomadv

```text
paged_idle_swap_enabled=1
paged_idle_swap_candidates=5
paged_idle_swap_out_calls=5
paged_idle_swap_madvise_enabled=0
paged_swap_madvise_calls=0
paged_swap_madvise_bytes=0
paged_swap_madvise_failures=0
paged_swap_rss_before_last_kb=0
paged_swap_rss_after_last_kb=0
paged_swap_rss_drop_last_kb=0
paged_swap_rss_drop_max_kb=0
swapped_blocks=4
paged_swap_in_calls=1
paged_swap_bytes_in=4194304
paged_swap_in_last_block=0
```

解释：idle swap-out 与 resume swap-in 均发生（`paged_idle_swap_out_calls=5`、`paged_swap_in_calls=1`），madvise 全程关闭，RSS 字段为 0，行为等同 5B-2。

### 6.2 madv

```text
paged_idle_swap_enabled=1
paged_idle_swap_candidates=5
paged_idle_swap_out_calls=5
paged_idle_swap_madvise_enabled=1
paged_swap_madvise_calls=5
paged_swap_madvise_bytes=19660800
paged_swap_madvise_failures=0
paged_swap_madvise_skip_no_full_page=0
paged_swap_madvise_skip_neighbor=384
paged_swap_rss_before_last_kb=8251324
paged_swap_rss_after_last_kb=8247484
paged_swap_rss_drop_last_kb=3840
paged_swap_rss_drop_max_kb=3840
swapped_blocks=4
paged_swap_in_calls=1
paged_swap_bytes_in=4194304
paged_swap_in_last_block=0
```

解释：

- madvise 真实执行：`paged_swap_madvise_calls=5`、`paged_swap_madvise_bytes=19660800`、`paged_swap_madvise_failures=0`；
- idle swap-out / resume swap-in 与 nomadv 完全一致（`paged_idle_swap_out_calls=5`、`paged_swap_in_calls=1`、`paged_swap_bytes_in=4194304`），证明 madvise 没有破坏 swap roundtrip；
- `paged_swap_madvise_skip_neighbor=384`：部分页因邻居保护被跳过，这是 `paged_madvise_block` 的预期安全行为（不波及相邻 RESIDENT/active 页），不是失败。

---

## 7. RSS 结果

```text
paged_swap_rss_before_last_kb = 8251324 KiB
paged_swap_rss_after_last_kb  = 8247484 KiB
paged_swap_rss_drop_last_kb   = 3840 KiB   (≈ 3.75 MiB)
paged_swap_madvise_bytes      = 19660800 bytes (≈ 18.75 MiB)
```

说明：

- madvise path executed with advised bytes > 0, and current RSS showed a positive drop of 3840 KiB；
- **RSS drop ≠ madvise_bytes**：`madvise_bytes` 是向内核 advise 的字节数，`rss_drop` 是同一时刻 `/proc` 观测到的常驻内存差值。两者不应相等——内核回收时机、页对齐、邻居跳过（`skip_neighbor=384`）、以及进程其它内存活动都会使观测 drop 小于 advise 量。本阶段只声明 **observed current RSS drop is positive**，不声明二者数值相等。

---

## 8. Abnormal Check

```text
raw_abnormal_matches=390
real_abnormal_matches=0
```

`raw_abnormal_matches` 大于 0 是因为每行 trace 都含 `paged_swap_madvise_failures=` / `paged_release_violation=` / `paged_swap_backend_failures=` 等字段名，被关键字 grep 命中。过滤掉值为 0 的字段（`failures=0` / `violation=0` / `backend_failures=0`）后 `real_abnormal_matches=0`。

结论：无真实 warning / error / NaN / backend failure / release violation / crash。misconfig 路径（MADVISE=1 但 idle swap 未 ready）另行验证为：warning 恰好一次、输出等同 base、madvise 禁用、不 crash。

---

## 9. 结论

**Stage 5C verified safe idle swapped block madvise with current RSS drop and resume correctness.**

- safe idle swapped block 可以被 madvise 释放物理页，`madvise_calls=5`、`madvise_bytes>0`、`madvise_failures=0`；
- madvise 后 block 保持 `SWAPPED`，未进入 `RELEASED`；resume 仍触发 swap-in（`paged_swap_in_calls=1`、`paged_swap_bytes_in=4194304`）；
- full stdout / seq1 active / seq0 resume 三段在 base / nomadv / madv 三组逐字节一致；
- current RSS 观测到正向下降（`rss_drop_last_kb=3840`），madvise path executed with advised bytes > 0；
- 无真实 warning / error / NaN / backend failure / release violation。

---

## 10. 边界说明

- 只验证 **current RSS**，不验证 **peak RSS**；
- 不做 prefetch；
- 不做 async swap-in；
- 不做 server 调度；
- 不是生产级 PagedAttention；
- 不支持非 F32 KV；
- 不支持 `v_trans`；
- 不做多 idle seq / 多轮 resume 的压力场景；
- RSS drop 的绝对值受 ctx / block_size / idle block 数量与页对齐影响，本阶段规模较小，仅证明 drop 为正，不量化上限。

---

## 11. 后续方向

- 放大 ctx / block_size / idle blocks，量化更大的 RSS drop，并降低 `skip_neighbor` 占比；
- 多 idle seq / 多轮 resume 压力场景；
- prefetch / async swap-in，降低 resume swap-in 延迟；
- server-like workload 下的调度与 RSS 行为；
- 探索 peak RSS（而不仅 current RSS）的优化路径。
