# Stage 5A-3 — non-identity gather robustness validation results

前置：

- [docs/kv_paged_read_stage5a2_nonidentity_gather_results.md](kv_paged_read_stage5a2_nonidentity_gather_results.md) — Stage 5A-2：non-identity row_idx remap probe（单次 smoke）
- [docs/kv_paged_routeb_idle_telemetry_validation_results.md](kv_paged_routeb_idle_telemetry_validation_results.md) — Stage 4C-3：idle cold block 仍在物理 read window 内的负向 gate
- driver：`examples/kv-idle-telemetry/idle-telemetry.cpp`
- 改动文件（沿用 5A-2，本阶段不改源码）：`src/llama-kv-cache.cpp`、`src/llama-kv-cache.h`

---

## 1. 背景

Stage 5A-2 实现了 non-identity row_idx remap probe：

- 环境变量 `LLAMA_KV_PAGED_GATHER_NONIDENTITY=1`（默认关闭）；
- 在 `set_input_paged_row_idx()` 中对 idle-only cold cell 做 row_idx remap；
- **不改变** `n_kv`、mask shape、graph shape；
- **不改** `get_k` / `get_v` / attention；
- **不做** 真实 swap / release / madvise / prefetch。

Stage 5A-2 单次 smoke 已证明：

```text
base_vs_remap_equal=0
sha256 一致
cold_in_read_window  1 -> 0
cold_not_in_read_window  0 -> 1
safe_swap_candidates  0 -> 1
swapped_blocks=0
```

但这只是**单个 case**。本阶段（5A-3）的唯一目标：验证 5A-2 的结论是否**稳定**，而非偶然单例。

## 2. 验证矩阵

| 维度 | 取值 |
|---|---|
| prompt | 3 个（`prompt_id` = 0 / 1 / 2） |
| seed | 1、42 |
| decode length `n` | 32、64、128 |
| case 总数 | 3 × 2 × 3 = **18** |

每个 case 跑 base 与 remap 两组，共用同一 driver、同一调度、同一 seed，唯一差别是 remap 开关：

| run | 环境 | 说明 |
|---|---|---|
| base  | `LLAMA_KV_PAGED=1`、`LLAMA_KV_PAGED_INGRAPH=1` | paged on，gather on，remap **off** |
| remap | base + `LLAMA_KV_PAGED_GATHER_NONIDENTITY=1` | paged on，gather on，remap **on** |

两组均**不**启用 swap / release / madvise / prefetch。correctness 通过每个 case 内 `base.out` vs `remap.out` 的 `cmp` 与 `sha256sum` 判断。

> 验证范围仍是 `examples/kv-idle-telemetry` driver、`ctx=512`、小规模 prompt/seed/n 矩阵（见 §6 边界）。

## 3. 异常提取（全空）

为避免“逐 case 肉眼核对”遗漏，验证脚本对 18 个 case 统一抽取**异常**集合。所有异常集合均为空：

```text
failed cmp:            (空)
sha mismatch:          (空)
no remap rows:         (空)
safe_after zero:       (空)
cold_after not reduced:(空)
warnings:              (空)
```

即：没有任何 case 出现输出不一致、未发生 remap、缺少安全候选、冷块未退出窗口或告警。

## 4. Summary 表格（18 case 全通过）

`remap_rows` / `remap_blocks` 随 `n` 变化（累计值，见 §5）；其余字段在所有 case 中恒定：`cmp=0`、`sha_base==sha_remap`、`skip_*=0`、`cold_before=1`、`cold_after=0`、`safe_after=1`、`swapped_blocks=0`、`warnings=0`。

| case | prompt_id | seed | n | cmp | remap_rows | remap_blocks | skip_no_dummy | skip_not_masked | skip_not_resident | cold_before | cold_after | safe_after | swapped_blocks | warnings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1  | 0 | 1  | 32  | 0 | 528  | 33  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 2  | 0 | 1  | 64  | 0 | 1040 | 65  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 3  | 0 | 1  | 128 | 0 | 2064 | 129 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 4  | 0 | 42 | 32  | 0 | 528  | 33  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 5  | 0 | 42 | 64  | 0 | 1040 | 65  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 6  | 0 | 42 | 128 | 0 | 2064 | 129 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 7  | 1 | 1  | 32  | 0 | 528  | 33  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 8  | 1 | 1  | 64  | 0 | 1040 | 65  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 9  | 1 | 1  | 128 | 0 | 2064 | 129 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 10 | 1 | 42 | 32  | 0 | 528  | 33  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 11 | 1 | 42 | 64  | 0 | 1040 | 65  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 12 | 1 | 42 | 128 | 0 | 2064 | 129 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 13 | 2 | 1  | 32  | 0 | 528  | 33  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 14 | 2 | 1  | 64  | 0 | 1040 | 65  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 15 | 2 | 1  | 128 | 0 | 2064 | 129 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 16 | 2 | 42 | 32  | 0 | 528  | 33  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 17 | 2 | 42 | 64  | 0 | 1040 | 65  | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| 18 | 2 | 42 | 128 | 0 | 2064 | 129 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |

每个 case 满足的判据：

- `cmp=0` 且 `sha_base == sha_remap`：remap 不改变输出；
- `remap_rows > 0`、`remap_blocks > 0`：remap 确实发生；
- `skip_no_dummy = skip_not_masked = skip_not_resident = 0`：无静默跳过/回退；
- `cold_before=1`、`cold_after=0`：idle cold block 退出物理 read window；
- `safe_after=1`：出现理论安全换出候选；
- `swapped_blocks=0`、`warnings=0`：无真实 swap、无告警。

## 5. 观察规律

`remap_rows` / `remap_blocks` 随 decode length `n` 单调增长：

| n | remap_rows | remap_blocks |
|---|---|---|
| 32  | 528  | 33  |
| 64  | 1040 | 65  |
| 128 | 2064 | 129 |

- 这两个量是**跨所有 decode step 的累计统计**：decode 步数越多，被 remap 的行/块累计越多，因此随 `n` 增长。它们**不**表示某一步存在 2064 行或 129 个不同 block。
- 与之相对，`cold_before` / `cold_after` / `safe_after` 是**末态快照**：无论 `n` 取多少，末步都稳定为 `1 / 0 / 1`。
- 两类指标在所有 prompt / seed 下表现一致，说明结论不依赖具体 prompt 文本或采样种子。

## 6. 限制和边界

必须明确，避免夸大：

- 这仍然是 **feasibility probe**，不是完整 PagedAttention；
- **没有**真实 swap；
- **没有**释放内存，**没有** RSS 收益；
- `safe_after=1` 表示**理论上**出现了安全换出候选，**不**表示已经实际换出任何 block；
- dummy cell 的正确性**依赖 mask 屏蔽被 remap 的 idle row** 这条不变量（代码已用行级 mask 复核强制；未来改动 mask 逻辑是回归风险点）；
- 当前验证范围仍是 `examples/kv-idle-telemetry` driver、`ctx=512`、小规模 prompt/seed/n 矩阵，不代表真实多并发负载下的 idle 分布；
- `remap_rows` / `remap_blocks` 为累计值，`cold_before/after`、`safe_after` 为末态快照（见 §5）。

## 7. 阶段结论

Stage 5A-3 robustness validation 在 18 个 case（3 prompt × 2 seed × 3 decode length）中**全部通过**。non-identity row_idx remap 在不改变输出、不触发 swap/madvise/release 的前提下，**稳定**使 idle-only cold block 退出物理 read window，并稳定产生 `safe_swap_candidate`。5A-2 的结论不是偶然单例。

## 8. 下一步

- 可进入 **Stage 5B**：基于 `safe_swap_candidate` 的真实 idle block swap 设计；
- 但进入真实 swap 前，必须先明确 **resume / swap-in correctness** 验证方案——`safe_swap_candidate>0` 只是理论窗口，真实换出需保证换回后逐字节一致；
- 也可先补充**更长 ctx / 更多 idle block** 的扩展验证，确认 remap 在更大 idle 分布下仍保持输出不变与窗口收缩。
