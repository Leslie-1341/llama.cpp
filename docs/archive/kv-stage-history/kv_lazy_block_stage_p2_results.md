# KV Lazy-Block Memory 阶段 P2：clear-frontier peak RSS 验证结果

> 操作系统功能赛技术报告素材 · P2 结果篇
>
> **定位**：承接 [P2-read](kv_lazy_block_stage_p2_read.md) 与 [P1 结果](kv_lazy_block_stage_p1_results.md)。本篇汇总 P2-code（`LLAMA_KV_LAZY_CLEAR=1` clear-frontier）的实测结果。**只整理结果、不改源码、不重跑实验。**
>
> 文档日期：2026/06/09｜分支：`kv-runtime-swap-e2-approx`

---

## 1. 阶段定位

P2 是 **peak RSS 路径**，与 P1（current RSS）互补：

- [P1](kv_lazy_block_stage_p1_results.md)：运行期对未使用 tail 做 `madvise(MADV_DONTNEED)`，降 current RSS，**但无法降 peak**——peak 在构造期 full clear 一瞬被顶满。
- P2：从源头避免构造期 `ggml_backend_buffer_clear(buf,0)` 对整块 KV buffer 的 full memset，使 tail 页**从不被 commit**，从而压低 peak RSS。

---

## 2. 机制简介：clear-frontier

- 构造期（[src/llama-kv-cache.cpp:275](../src/llama-kv-cache.cpp#L275) 区域）：开启时不再 full clear，只清 `[0, clear_frontier)` 前缀（初始 `clear_frontier = min(kv_size, 256)`），tail `[clear_frontier, kv_size)` 不触碰。
- 运行期（`apply()` 内 `clear_frontier_advance(n_kv)`）：当 `n_kv` 增长超过 `clear_frontier`，把新进入 `[0,n_kv)` 读区间的 `[clear_frontier, GGML_PAD(n_kv,256))` 行增量清零，再单调右移 frontier。
- 保证：凡进入过读区间的 block 都已清零；从未进入的 tail 永不 commit。

---

## 3. 正确性边界

仅在以下边界启用，否则一次 warning 后**退回 full clear**（默认行为）：

| 约束 | 取值 |
|---|---|
| FlashAttention | `-fa on`（→ `v_trans=false`）|
| stream | `n_stream == 1`（单 unified stream）|
| 序列 | 单 seq |

依据（[P2-read §2](kv_lazy_block_stage_p2_read.md)）：tail `[n_kv,kv_size)` 不在 `get_k/get_v` view 内；padding `[used_max_p1,n_kv)` 虽在 view 内，但 CPU flash-attention 内循环先判 mask，`mv==-INFINITY` 时 `continue`、不读 K/V，故未清零字节不产生 NaN 传播。**不声称完整 PagedAttention，不声称支持所有 backend**——非 FA / 转置 V / 多 stream 路径均退回 full clear。

---

## 4. 默认 warmup 下的现象：process peak 未降

| 组 | ctx | n | peak RSS (KiB) | skipped_bytes |
|---|---:|---:|---:|---:|
| baseline | 4096 | 16 | 8,652,696 | — |
| P2 lazy-clear | 4096 | 16 | 8,651,796 | 960 MiB |

**结论**：P2 的 skipped tail clear 确实生效（skipped=960 MiB），但**默认 warmup 在 warmup decode 后、正式 prompt decode 前引入额外约 480–500 MiB 峰值**，把 P2 在 KV 构造期省下的内存重新顶上去，**覆盖了 process 级 peak 收益**。两组 peak 差仅 ≈0.9 MiB（噪声级）。

---

## 5. RSS trace 诊断

### 5.1 构造期：P2 确实绕过 full clear

| 采样点 | baseline rss (KiB) | P2 rss (KiB) |
|---|---:|---:|
| pre-clear | 8,124,248 | 8,123,996 |
| post-clear | 8,648,344 | 8,157,052 |
| **clear 阶段增量** | **≈ 512 MiB** | **≈ 32 MiB** |

P2 partial clear 只 commit 前缀（+32 MiB），baseline full clear commit 整块（+512 MiB），skipped_bytes=960 MiB。**证明 P2 在构造期有效绕过 KV tail full clear。**

### 5.2 decode lifecycle：warmup 干扰定位

| 采样点 | rss (KiB) |
|---|---:|
| warmup `decode-exit`（n_tokens=2）| 8,158,652 |
| prompt `decode-enter`（n_tokens=7）| 8,649,920 |

jump（≈ 491 MiB）发生在 **warmup decode 结束后、正式 prompt decode 进入前 / 首次 prompt graph_compute 实算**。多轮插桩已排除：`output_reserve`、`graph_reserve`、`process_ubatch` 的 apply/alloc/setinputs 记录点、`graph_compute` 后同步、logits/embd get——均非该 jump 的直接贡献点。**jump 来自 compute buffer 首次大批实算 commit，与 KV 正交。**

---

## 6. `--no-warmup` 关键对照：P2 peak 真实下降

排除 warmup 干扰后：

| 组 | Maximum RSS (KiB) | prompt eval (tok/s) | eval (tok/s) |
|---|---:|---:|---:|
| baseline + `--no-warmup` | 8,652,372 | 37.44 | 13.74 |
| P2 lazy-clear + `--no-warmup` | 8,163,896 | ≈ baseline | ≈ baseline |

P2 统计：skipped_bytes=960 MiB，init_bytes=64 MiB，grow_bytes=0。

**P2 peak 降幅 = 8,652,372 − 8,163,896 = 488,476 KiB ≈ 477.0 MiB。**

> 即在 `-fa on` 短序列 + 大 ctx 预留场景下，P2 真实降低 peak RSS ≈ 477 MiB，且 prompt/eval 性能与 baseline 接近（无明显回退）。grow_bytes=0 说明本用例 n_kv 未超过初始 256 frontier，tail 全程未 commit。

---

## 7. P1+P2 同开解释

| 组（+`--no-warmup`）| Maximum RSS (KiB) | lazy-tail rss_before | lazy-tail rss_after | failures | skipped_bytes |
|---|---:|---:|---:|---:|---:|
| P1+P2 | 8,163,860 | 8,157,224 | 8,161,404 | 0 | 960 MiB |

- P1+P2 的 peak（8,163,860）与 P2 单开（8,163,896）**基本一致**；
- P1 lazy-tail 的 `rss_after` 反而略高于 `rss_before`（+≈4 MiB），即 **P1 不再能释放大块 current RSS**——因为 P2 已在构造期阻止 tail commit，tail 本就不在物理内存里，P1 的 madvise 无页可回收；
- 二者不冲突、可同开，但**收益不叠加**：P2 已覆盖 P1 在该场景的内存收益。P1 的价值回到「P2 不启用 / 非 P2 边界」时的 current RSS 兜底。

---

## 8. 结论

| 命题 | P2 结论 | 依据 |
|---|---|---|
| P2 绕过构造期 KV tail full clear | ✅ 成立（clear 增量 512→32 MiB，skipped 960 MiB）| §5.1 |
| 默认 warmup 下 process peak 下降 | ❌ 未降（warmup 引入 ~491 MiB 额外峰值覆盖收益）| §4、§5.2 |
| 排除 warmup 后 peak 真实下降 | ✅ 降 ≈ 477 MiB（`--no-warmup`）| §6 |
| 正确性（`-fa on` 边界）| ✅ 输出正常、无 NaN、性能接近 baseline | §3、§6 |
| P1+P2 收益叠加 | ❌ 不叠加（P2 已阻止 tail commit，P1 无页可回收）| §7 |

**一句话**：P2 clear-frontier 在 `-fa on / v_trans=false / n_stream=1` 边界下**真实降低 peak RSS ≈ 477 MiB**，但该收益**仅在排除默认 warmup 干扰（`--no-warmup`）时可见**——默认 warmup 引入的 ~491 MiB compute 峰值会掩盖它。

---

## 9. 后续方向

1. **warmup / compute buffer 峰值是独立瓶颈，归为 G 阶段独立方向**：RSS trace 已定位默认 warmup→prompt 之间的 ~491 MiB jump 来自 compute buffer 首次大批实算，与 KV 正交。这是比 KV tail 更大的 peak 杠杆，但属 ggml graph allocator / warmup 策略范畴，需独立立项（G0）。
2. **P2 本身不再加码**：clear-frontier 已达成设计目标（构造期绕过 tail clear、`--no-warmup` 下降 peak 477 MiB），动态 frontier 推进逻辑已就位（本用例未触发 grow，需更长序列验证 grow 路径）。
3. **不进入完整 PagedAttention、不扩展 backend**：维持 P2-read 边界，非 FA / 多 stream / 转置 V 退回 full clear。

---

## 10. 回滚

P2 代码回滚：

```bash
git checkout -- src/llama-kv-cache.cpp src/llama-kv-cache.h
```

本文档回滚：

```bash
rm docs/kv_lazy_block_stage_p2_results.md
```
