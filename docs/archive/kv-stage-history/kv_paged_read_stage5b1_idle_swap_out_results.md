# Stage 5B-1 Safe-Candidate Idle KV Swap-Out Results

> 本文档记录 Stage 5B-1 的实验结果与 telemetry 解释。
> 对应提交：`dc4dcef95 feat: add safe-candidate idle kv swap-out probe`

---

## 1. Purpose

Stage 5A 已经通过 `LLAMA_KV_PAGED_GATHER_NONIDENTITY=1` 证明：idle-only cold block 可以通过 row_idx remap 退出当前 active request 的 physical read window，并形成 safe candidate。

Stage 5B-1 的目标是在此基础上验证：

- safe-candidate based idle block 可以执行真实 block-level swap-out；
- 当前 active request 输出保持逐字节一致；
- swap-out 不依赖旧 positional window policy；
- 本阶段不引入 madvise、release、RSS 优化或 resume swap-in 验证。

---

## 2. Implementation Scope

本阶段实现范围：

- 新增 `LLAMA_KV_PAGED_IDLE_SWAP=1`；
- 复用 `LLAMA_KV_PAGED_SWAP=1` 创建 backing store；
- `paged_swap_out_block()` 增加 `do_madvise` 参数；
- idle safe-candidate 路径调用 `paged_swap_out_block(block, false)`；
- `do_madvise=false` 时只跳过 madvise，不跳过写盘、状态更新和 swap 统计；
- `LLAMA_KV_PAGED_IDLE_SWAP=1` 时禁用旧 `paged_swap_out_window()` positional policy；
- 允许已经 `SWAPPED` 的 idle block 在 active-only 阶段继续 remap 到 resident dummy，避免每步 swap-in 抖动。

本阶段明确不做：

- `MADV_DONTNEED`；
- release；
- RSS 优化或 RSS benchmark；
- resume 后 swap-in correctness；
- 完整 idle request swap；
- 生产可用 PagedAttention。

---

## 3. Test Matrix

| Smoke | 目的 | 关键检查 |
|---|---|---|
| Primary smoke | 验证 Stage 5B-1 idle swap-out 主路径 | 输出一致、idle candidate 真实 swap-out、无 madvise/release/swap-in/backend failure、无 positional policy 污染 |
| Smoke A: n=128 continuation | 验证较长 continuation 下 idle block 持续保持 swapped | `paged_idle_swap_out_calls=1`、`swapped_blocks=1`、无 swap-in、输出一致 |
| Smoke B: misconfig no-op | 验证配置缺失时安全 no-op | warning 一次、swap disabled、输出一致、不崩溃 |
| Smoke C: positional-regression | 验证新增 idle-swap gate 不破坏普通 paged path | IDLE_SWAP 未设置时输出一致、无异常 warning/error |

---

## 4. Primary Smoke Result

Result:

```text
base_vs_swap_equal=0
base sha256 = 6bf045c4971db88adeb534f8e91dd251ae7ec20c2a1c45a1b8257a32acea3deb
swap sha256 = 6bf045c4971db88adeb534f8e91dd251ae7ec20c2a1c45a1b8257a32acea3deb

cold_in_read_window=0
cold_not_in_read_window=1
paged_idle_swap_enabled=1
paged_idle_swap_candidates=1
paged_idle_swap_out_calls=1
paged_idle_swap_skip_not_remapped=0
paged_idle_swap_skip_not_resident=0
paged_nonidentity_enabled=1
paged_nonidentity_cold_in_read_window_before=1
paged_nonidentity_cold_in_read_window_after=0
swapped_blocks=1
released_blocks=0

madvise / swap-in / backend failure / paged_swap_out_window check: no output
```

Conclusion:

- active request 输出逐字节一致；
- idle safe candidate 被真实 swap-out；
- block 状态进入 `SWAPPED`；
- 没有 release；
- 没有 madvise；
- 没有 swap-in；
- 没有 backend failure；
- 没有旧 positional swap policy 污染。

---

## 5. Continuation Smoke Result

Result:

```text
n128_base_vs_swap_equal=0
base sha256 = 698ace6843e02cf564317fc6ad7a50274897b78b383af664e1676fe0ee16bc99
swap sha256 = 698ace6843e02cf564317fc6ad7a50274897b78b383af664e1676fe0ee16bc99

tail:
paged_idle_swap_enabled=1
paged_idle_swap_candidates=1
paged_idle_swap_out_calls=1
cold_in_read_window=0
cold_not_in_read_window=1
safe_swap_candidates=0
paged_nonidentity_remap_rows=2064
paged_nonidentity_remap_blocks=129
paged_nonidentity_cold_in_read_window_before=1
paged_nonidentity_cold_in_read_window_after=0
swapped_blocks=1
released_blocks=0

madvise / swap-in / backend failure / warning / error check: no output
```

Conclusion:

- n=128 长一点的 continuation 下输出仍逐字节一致；
- safe candidate 被 swap-out 后，后续步骤没有反复 swap-out；
- `paged_idle_swap_out_calls` 保持 1；
- `swapped_blocks` 保持 1；
- 没有 swap-in；
- 没有 madvise；
- `SWAPPED`-remap 分支可以让 idle block 在 active-only 阶段持续保持 swapped。

---

## 6. Misconfiguration No-op Result

Result:

```text
misconfig_base_equal=0
base sha256 = 6bf045c4971db88adeb534f8e91dd251ae7ec20c2a1c45a1b8257a32acea3deb
misconfig sha256 = 6bf045c4971db88adeb534f8e91dd251ae7ec20c2a1c45a1b8257a32acea3deb

warning:
LLAMA_KV_PAGED_IDLE_SWAP=1 requires LLAMA_KV_PAGED_SWAP=1, LLAMA_KV_PAGED_GATHER_NONIDENTITY=1, LLAMA_KV_PAGED_IDLE_TRACE=1, and a backing store; idle swap disabled for this run

tail:
paged_idle_swap_enabled=0
paged_idle_swap_candidates=0
paged_idle_swap_out_calls=0
paged_nonidentity_enabled=0
swapped_blocks=0
released_blocks=0

unexpected errors: no output
```

Conclusion:

- 配置不完整时 warning 一次；
- idle swap 自动 disabled；
- 输出仍逐字节一致；
- 不发生 swap；
- 不崩溃；
- 行为是安全 no-op。

---

## 7. Positional Regression Result

Result:

```text
positional_base_equal=0
base sha256 = 6bf045c4971db88adeb534f8e91dd251ae7ec20c2a1c45a1b8257a32acea3deb
positional sha256 = 6bf045c4971db88adeb534f8e91dd251ae7ec20c2a1c45a1b8257a32acea3deb

tail:
swapped_blocks=0
released_blocks=0
resident_blocks 正常增长
无异常 warning/error
```

Conclusion:

- `LLAMA_KV_PAGED_IDLE_SWAP` 未设置时，新增 gate 没有破坏输出；
- 本短 workload 下旧 positional swap 没实际换出 block，但没有崩溃、没有异常、输出一致；
- 本 smoke 证明新增 idle-swap gate 没有破坏普通 paged path。

---

## 8. Telemetry Interpretation

以下组合不是失败：

```text
safe_swap_candidates=0
paged_nonidentity_safe_candidates_after=0
swapped_blocks=1
paged_idle_swap_candidates=1
paged_idle_swap_out_calls=1
```

原因：

- Stage 5A 中 safe candidate 表示“仍是 `RESIDENT` 且可以被 swap 的候选 block”；
- Stage 5B-1 中该 candidate 已被 swap-out 消费；
- block 状态从 `RESIDENT` 变为 `SWAPPED`；
- 因此它不再是“待 swap candidate”，而是“已 swapped block”；
- 所以后续 tail telemetry 中 `safe_swap_candidates=0` 可以与 `swapped_blocks=1` 同时成立。

Stage 5B-1 的成功签名应看：

- `paged_idle_swap_candidates > 0`；
- `paged_idle_swap_out_calls > 0`；
- `swapped_blocks > 0`；
- swap-in、madvise、backend failure 均不出现。

---

## 9. SWAPPED-remap Correctness

`SWAPPED`-remap 分支是 Stage 5B-1 correctness 的关键部分。

Step N:

- idle cold block 被 non-identity gather remap 到 resident dummy；
- 当前 active request 的 physical read window 不再包含该 block；
- safe-candidate idle swap 调用 `paged_swap_out_block(block, false)`；
- block 状态从 `RESIDENT` 变为 `SWAPPED`。

Step N+1:

- 该 block 仍然 idle；
- 如果不继续 remap，row_idx 会指向真实 `SWAPPED` physical block；
- read path 会触发 swap-in；
- 这会造成每步 swap-in 抖动，并让 idle swap-out 失去意义；
- 因此 idle 期间，`SWAPPED` idle block 必须继续 remap 到 resident dummy。

Resume 时：

- 该 block 重新含 active seq；
- idle-only remap 条件失效；
- row_idx 指回真实 physical block；
- read path 再触发 swap-in；
- 这正是 Stage 5B-2 resume correctness 的基础。

Smoke A 中 `paged_idle_swap_out_calls=1`、`swapped_blocks=1` 且无 swap-in，说明 active-only 阶段已经能持续保持 swapped，而不是每步 swap-in/swap-out 抖动。

---

## 10. What This Proves

Stage 5B-1 已验证：

- safe-candidate based idle KV block 可以真实 swap-out；
- active request 输出逐字节一致；
- idle swapped block 可在 active-only 阶段持续保持 `SWAPPED`；
- misconfig 时安全 no-op；
- 未混入 madvise；
- 未混入 release；
- 未做 RSS 优化；
- 未触发 swap-in；
- 未出现 backend failure；
- `LLAMA_KV_PAGED_IDLE_SWAP=1` 时旧 `paged_swap_out_window()` positional policy 没有污染结果。

---

## 11. What This Does Not Prove

Stage 5B-1 尚未证明：

- 完整 idle request swap；
- resume 后 swap-in correctness；
- RSS 下降；
- peak RSS 优化；
- production-ready PagedAttention；
- server 场景；
- 非 F32 KV；
- `v_trans` 场景。

---

## 12. Next Step: Stage 5B-2 Resume / Swap-in Correctness

Stage 5B-2 应验证 resume 路径：

- idle block 已经处于 `SWAPPED`；
- 对应 seq 重新成为 active；
- remap 条件自动失效；
- row_idx 指回真实 physical block；
- read/write path 在 graph compute 前完成 swap-in；
- resume 输出与 no-swap baseline 逐字节一致；
- swap-in 发生一次且无 backend failure；
- 不引入 madvise、release 或 RSS benchmark。

Stage 5B-1 的结论应被限定为：safe-candidate idle block 的真实 swap-out correctness probe 已通过；resume correctness 留给 Stage 5B-2。
