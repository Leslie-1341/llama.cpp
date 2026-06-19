# Stage 5B-2：idle request resume / swap-in correctness 结果

> 本文档仅记录实验结果与分析，不包含任何源码修改说明。
> 文件路径：`docs/kv_paged_read_stage5b2_resume_swapin_results.md`
> 关联文档：`docs/kv_paged_read_stage5b2_resume_swapin_plan.md`（5B-2 计划）、`docs/kv_paged_read_stage5b_idle_swap_design.md`（Stage 5B 总设计）

---

## 1. Stage 5B-2 目标

验证一个 idle sequence 的 KV block 被真实 swap-out（写入 backing store，`RESIDENT → SWAPPED`）后，在该 sequence **resume** 时能够正确 swap-in（`SWAPPED → RESIDENT`），且：

- resume 段输出与 no-swap baseline **逐字节一致**；
- swap-out / swap-in roundtrip 的内容一致（bytes 对账）；
- 全程无 backend failure、无 madvise、无 release、无 NaN、无 crash。

即闭合 5B-1 只证明的 swap-out 半程，补齐 swap-in / resume 这一半。

---

## 2. Driver 设计与流程

target：`build/bin/llama-kv-idle-swap-resume`（新增 `examples/kv-idle-swap-resume/`，不改旧 5A/5B-1 driver）。

流程拓扑（A idle → B active → A resume）：

1. **seq0 prefill**：写入 prompt A 的 KV（`logits=true`，便于 resume 首 token 采样）；
2. **seq1 active**：prefill prompt B 并 decode N 步——此阶段 seq0 为 idle-only，其 cold block 被 swap-out；
3. **seq0 idle**：seq1 decode 期间 seq0 不出现在任何 ubatch，持续被 remap 到 dummy、保持 SWAPPED；
4. **seq0 resume**：从 seq0 prefill 末位 logits 采样首 token，pos 接续 seq0 长度，token 绑定 `{ 0 }`，再 decode M 步——`active_seq.test(0)=true`，remap 前提失效，row_idx 指回真实 SWAPPED block，read path 触发 swap-in。

stdout 分段打印使用实际 driver marker：

- `===SEQ1_ACTIVE_BEGIN===` / `===SEQ1_ACTIVE_END===`：seq1 active 段；
- `===SEQ0_RESUME_BEGIN===` / `===SEQ0_RESUME_END===`：seq0 resume 段。

因此可以分别抽取 seq1 active 与 seq0 resume 输出，并与 no-swap baseline 做逐字节比对。

determinism：greedy、`temp=0`、固定 `seed=1`、`kv_unified=true`、`backend_sampling=false`。

---

## 3. 环境变量与运行配置

公共运行参数：

```
-n 64 --ctx-size 512 --batch-size 128 --ubatch-size 128
--seed 1 --temp 0 --cache-type-k f32 --cache-type-v f32
--kv-unified --parallel 2
model: /root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

| 组 | 环境变量 |
|---|---|
| base | `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 LLAMA_KV_PAGED_IDLE_TRACE=1 LLAMA_KV_PAGED_TRACE=1` |
| swap | base + `LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=1` |

两组同 seed / prompt / ctx / n，仅 swap 相关开关不同。

---

## 4. correctness 结果

| 比对范围 | 指标 | 结果 |
|---|---|---|
| full stdout | `base_vs_swap_equal` | 0（一致） |
| full stdout | sha256 | base 与 swap 一致 |
| seq1 active 段 | `seq1_active_equal` | 0（一致） |
| seq1 active 段 | sha256 | `371c5a41831ad024c928df9d629dda9c51181f485fe6e07261f3269c38831419` |
| **seq0 resume 段** | `seq0_resume_equal` | **0（一致）** |
| **seq0 resume 段** | sha256 | `873ea6bd90bcc548a5ceb6b78ae356df1900218b1e8906f4ccffc94839907b31` |

seq0 resume 段逐字节一致是核心结论：swap-out → resume → swap-in roundtrip 未污染 idle sequence 的历史 KV。

---

## 5. swap-out / swap-in telemetry 结果

swap-out（idle 阶段）：

```
paged_idle_swap_enabled   = 1
paged_idle_swap_candidates= 5
paged_idle_swap_out_calls = 5
swapped_blocks            = 4
```

swap-in（resume 阶段）：

```text
paged_swap_in_calls       = 1
paged_swap_bytes_in       = 4194304
paged_swap_in_last_block  = 0
```

异常检查：

```text
grep warning/error/failed/NaN/backend failure/release violation/crash 为空
```

说明未观察到 backend failure、release violation、NaN、crash 或 warning/error 相关异常。由于本次最终 counter 提取中未直接打印 `paged_swap_backend_failures=0` 字段，本文不将其作为独立计数结论，只以异常 grep 为空作为负面证据。

---

## 6. 关键 trace 解释

- **resume 已进入**：trace 中出现 `active_seq=0`，证明 seq0 token 重新入 ubatch，进入 resume 阶段；此前 idle 阶段 `active_seq` 不含 0。
- **remap 自动撤销**：resume step 中 seq0 拥有的 block `has_active_seq=true → only_seen_idle_seq=false`，cold 标记失效，row_idx 指回真实 physical block（不再指向 dummy）。`cold_in_read_window=0`、`cold_not_in_read_window=4` 与 idle 期一致，表明 resume 触发的 swap-in 发生在仍解析回真实 SWAPPED block 的那一步。
- **SWAPPED idle block 跨 active-only step 持续 remap**：在 seq1 active 阶段，seq0 的 idle-only SWAPPED block 不会被立即 swap-in，而是继续通过 non-identity remap 指向 resident dummy block；只有当 seq0 resume、`active_seq.test(0)=true` 后，该 block 不再满足 idle-only remap 条件，row_idx 才重新指向真实 physical block，从而触发所需的 swap-in。这说明本阶段验证的是“idle 阶段保持 swapped，resume 阶段按需复活”的 roundtrip 行为。
- **swap-in 仅触发一次**：`paged_swap_in_calls=1`，`paged_swap_bytes_in=4194304`（一个 block 的 K+V 容量），`paged_swap_in_last_block=0` 指 swap-in 的是 block 0。说明 resume 当前实际命中并读回所需的那个 SWAPPED block，而非每步反复 swap-in——符合 §5.1「SWAPPED idle block 持续 remap、resume 才复活」的不变量。
- **candidates(5) vs swapped_blocks(4)**：候选计数按 step 累加（同一逻辑 block 可在多 step 被识别为候选），而 `swapped_blocks` 是去重后实际处于 SWAPPED 的物理 block 数；两者不等是预期的「候选事件数 ≠ 物理 block 数」，非异常。
- **无副作用**：grep warning/error/failed/NaN/backend failure/release violation/crash 全为空。

---

## 7. 结论

**Stage 5B-2 verified idle KV block swap-out / resume swap-in roundtrip correctness.**

- idle-only cold block 被真实 swap-out（`swapped_blocks=4`，`paged_idle_swap_out_calls=5`）；
- seq0 resume 触发 swap-in（`paged_swap_in_calls=1`，`paged_swap_bytes_in=4194304`）；
- seq0 resume 段输出与 no-swap baseline **逐字节一致**（`seq0_resume_equal=0`，sha256 相同）；
- seq1 active 段与 full stdout 同样逐字节一致；
- 异常 grep 为空，未观察到 warning / error / failed / NaN / backend failure / release violation / crash；本阶段未启用 madvise / release。

---

## 8. 边界说明

本阶段**只验证 correctness**，明确不涉及：

- 不做 RSS benchmark；
- 不做 `MADV_DONTNEED`（madvise）；
- 不做 release；
- 不做 prefetch / async swap-in；
- 不是生产级 PagedAttention；
- 不是 server 多请求真实调度（拓扑为单 driver 内 seq0/seq1 显式编排）；
- 不支持非 F32 KV、不支持 `v_trans`、不做多 idle seq / 多轮 resume 压力场景。

本质上这是一个 **idle KV block swap-out / resume swap-in correctness probe**，而非完整生产方案。

---

## 9. 后续方向

- **Stage 5C**：在 correctness 已立的前提下，引入 RSS / madvise / release 组合，量化内存收益；
- 或 prefetch / async swap-in，降低 resume 首步延迟；
- 多 idle seq、多轮 resume、与真实 server 调度的集成压力场景。

> 注意：本结果仅证明 correctness，不应据此外推为完整生产方案。RSS 收益、调度策略、并发安全均需后续 stage 单独验证。
