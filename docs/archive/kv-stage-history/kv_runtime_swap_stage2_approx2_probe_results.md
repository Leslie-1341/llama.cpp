# Stage 2 Approx2 Probe Results

本文记录 Stage 2 approximate-window 路线中 approx1 mask-only、approx2 view-offset probe 以及 fix-C 的阶段结果。该文档只总结当前验证结论，不声称 approximate-window 已完成，也不声称当前已经获得 RSS 收益。

## 1. approx1 mask-only 结果

approx1 在 `LLAMA_KV_SWAP_MODE=approx` 下只修改 attention mask，不改变物理 K/V read window，不触发 swap、madvise 或 backing store。

已观察结果：

```text
window=512: 输出与 baseline sha256 一致
window=64: 输出允许不同
approx_masked=2485
```

解释：

- `window=512` 覆盖本轮短上下文，因此 mask-only 路径不改变可见历史，输出与 baseline 一致。
- `window=64` 会屏蔽窗口外历史 token，输出不同属于预期。
- `approx_masked=2485` 表明 mask-only 滑窗语义确实生效。
- approx1 不触发 swap / madvise / backing store，也不缩小 K/V physical view，因此不能降低 RSS。

## 2. approx2-probe 初始失败

approx2-probe 尝试让 K/V view 从 `visible_lo` 开始，只覆盖最近窗口对应的物理 KV 行。但初始 probe 发现 mask 侧和 K/V view 侧没有使用同一套窗口参数。

已观察现象：

```text
mask side: visible_lo > 0
get_k/get_v side: visible_lo = 0
approx_debug_get_k_visible_gt0_calls=0
approx_debug_get_v_visible_gt0_calls=0
```

根因：

- K/V view 在 graph reserve / graph 构造期创建。
- reserve 期使用 dummy `sinfo`，并且原先 `n_kv = get_size()`，例如 `kv_size=512`。
- mask 在 apply 期使用真实 `sinfo`，并使用真实 `used_max_p1` / `visible_lo` / effective `n_kv`。
- 因此 K/V physical view 与 mask 使用了两套不同的 `n_kv` / window 参数。

结论：问题不在 approx1 的 mask 语义本身，而在 graph reserve / graph reuse 机制下，K/V view 的 shape 和 byte offset 没有随真实 apply-time window 动态更新。

## 3. fix-C 结果

fix-C 不解决动态 `visible_lo > 0`。它只验证一个更小的问题：当窗口覆盖当前真实历史、`visible_lo=0` 时，reserve / graph 构造期能否把 K/V physical view 的 `n_kv` 从 full `kv_size` 缩短到固定窗口上界。

实验条件：

```text
ctx=512
n=128
window=256
visible_lo=0
```

结果：

```text
base_vs_fixc_w256_equal=0
sha256(base.out)      = 0ba3f3616ab41d8ced16526831ebc1f8bf3fa140738a9c6ee309d28f32d11763
sha256(fixc_w256.out) = 0ba3f3616ab41d8ced16526831ebc1f8bf3fa140738a9c6ee309d28f32d11763

get_k: visible_lo=0 row_size=2048 byte_offset=0 n_kv=256 kv_size=512 visible_hi=256
backend_failures=0
approx_debug_get_k_visible_gt0_calls=0
approx_debug_get_v_visible_gt0_calls=0
```

解释：

- 在 `visible_lo=0` 且窗口覆盖真实历史时，将 reserve / graph K/V view 从 `512` 缩到 `256` 不破坏输出。
- 这证明“缩短 physical view length”本身在 no-offset 场景下可行。
- RoPE / mask / graph shape 在 no-offset 场景下成立。
- `approx_debug_get_k_visible_gt0_calls=0` 和 `approx_debug_get_v_visible_gt0_calls=0` 说明本次 fix-C 没有验证动态 offset。

## 4. route-A dynamic-view 结果

route-A 通过在 `visible_lo` 变化时禁止复用旧 graph，使 K/V physical view 的 byte offset 跟随 apply-time window。该路线只验证 correctness，不尝试降低 RSS。

已观察结果：

```text
window=256: 输出与 baseline sha256 一致
window=64: 输出允许不同，且与 approx1 mask-only 结果一致
approx_debug_get_k_visible_gt0_calls=2240
approx_debug_get_v_visible_gt0_calls=2240
backend_failures=0
default/no-env sha256 不变
```

解释：

- `visible_lo` 由 `llama_kv_cache_context::apply()` 与 `n_kv` 同步刷新，并作为 K/V view、mask 和 graph reuse 判断的单一来源。
- approx 下 reserve / dummy graph 的 physical read window 长度使用 `min(PAD(window, 256), kv_size)`。因此 `window=64` 时实际 K/V view 长度仍为 `256`，不是 `64`。
- physical read window 内包含有效历史和 padded empty rows。empty rows 依赖 KV cache 零初始化，并由 attention mask 屏蔽；correctness probe 已覆盖该行为。
- 对 approx3 / RSS 目标而言，`256` 是当前连续 physical read window 的最小粒度。要进一步降低粒度，需要后续 paged-read / block-table / gather 类设计，本阶段不进入。

## 5. 阶段结论

- approx1 证明 mask-only approximate-window 语义生效，但由于物理 K/V view 仍读连续 `[0,n_kv)`，不降低 RSS。
- approx2-probe 暴露出 graph reserve / graph reuse 问题：K/V view 在 reserve 期固定，mask 在 apply 期动态变化，两者窗口参数不一致。
- fix-C 证明 no-offset 场景下缩短 physical view length 可行，输出保持 bit-exact。
- route-A 证明 dynamic `visible_lo > 0` 可以通过 graph rebuild 进入 K/V physical view，并保持已测 correctness。
- 当前仍不能声称获得 RSS 收益，也不进入 approx3 / `MADV_DONTNEED` / `RELEASED`。

## 6. 后续方向

后续需要先解决 graph 与动态窗口的一致性问题，再讨论释放物理页或 RSS 收益。可选方向包括：

- 每步重建 graph，使 K/V view shape 和 byte offset 跟随真实 `visible_lo`。
- 让 graph 支持动态 K/V offset，并保证 mask、RoPE 和 attention view 使用同一套窗口参数。
- 转向 idx-gather / paged-read / block-table 方向，避免依赖连续 K/V view 的动态 offset。

当前禁止夸大结论：

- 不声称 approx2 已经降低 RSS。
- 不声称 dynamic-view route-A 是性能路径。
- 不声称当前连续 view 粒度已经满足 approx3 RSS 目标。
