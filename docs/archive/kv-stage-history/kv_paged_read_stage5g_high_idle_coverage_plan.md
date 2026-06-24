# Stage 5G-0：high-idle-coverage workload 验证计划

> 本文档为 Stage 5G-0 计划，仅新增 docs，不改源码、examples、CMake、common、include、ggml、scripts。
> 文件路径：`docs/kv_paged_read_stage5g_high_idle_coverage_plan.md`
> 关联文档：
> - `docs/kv_paged_read_stage5f_releasable_coverage_plan.md`（Stage 5F-0 coverage funnel 计划）
> - `docs/kv_paged_read_stage5f_releasable_coverage_results.md`（Stage 5F-1 coverage matrix 结果）

---

## 1. 背景

Stage 5F-1 coverage matrix 已定位当前约 12% release ratio 的主瓶颈：

```text
small/mid/large 的 idle-owned coverage ≈ 12.5%
最终 release ratio ≈ 11.7%
read-window = 0%
resident-safe gate = 100%
nonidentity-remap gate = 100%
KV drop / idle-owned ≈ 93.75%
```

结论：

```text
当前瓶颈不是 madvise 机制失效，也不是 read-window / remap gate 过严，
而是当前 workload 中可释放 idle-owned KV 覆盖范围太小。
```

---

## 2. Stage 5G 目标

验证路线 A：

```text
通过设计 high-idle-coverage workload，提高 idle-owned KV coverage；
观察 release ratio 是否随 idle-owned coverage 同步提高；
证明当前释放机制的收益上限主要由 workload 中 idle KV 占比决定。
```

本阶段不修改释放策略。

---

## 3. 实验思想

Stage 5F-1 矩阵中，三档 idle-owned coverage 几乎相同：

```text
ctx=512,  n=64  -> idle coverage ≈ 12.5%
ctx=1024, n=128 -> idle coverage ≈ 12.5%
ctx=2048, n=256 -> idle coverage ≈ 12.5%
```

三档 `n / ctx` 比例固定（64/512 = 128/1024 = 256/2048 = 0.125），idle coverage 也固定在约 12.5%，提示二者强相关。因此可以先通过增大 `n / ctx` 来提高 idle-owned coverage。

假设：

```text
idle-owned coverage 近似随 n / ctx 增大；
如果 idle-owned coverage 提高到 25% / 31% / 37%，release ratio 也应接近相应比例。
```

---

## 4. 建议实验矩阵

先跑 small / mid，不直接跑 large。

small（ctx=512）：

```text
small_12p: ctx=512, n=64    (n/ctx = 0.125)
small_25p: ctx=512, n=128   (n/ctx = 0.250)
small_31p: ctx=512, n=160   (n/ctx = 0.3125)
small_37p: ctx=512, n=192   (n/ctx = 0.375)
```

mid（ctx=1024）：

```text
mid_12p: ctx=1024, n=128    (n/ctx = 0.125)
mid_25p: ctx=1024, n=256    (n/ctx = 0.250)
mid_31p: ctx=1024, n=320    (n/ctx = 0.3125)
mid_37p: ctx=1024, n=384    (n/ctx = 0.375)
```

如果 small / mid 结果成立，再扩展 large（ctx=2048）：

```text
large_12p: ctx=2048, n=256  (n/ctx = 0.125)
large_25p: ctx=2048, n=512  (n/ctx = 0.250)
large_31p: ctx=2048, n=640  (n/ctx = 0.3125)
large_37p: ctx=2048, n=768  (n/ctx = 0.375)
```

其余参数沿用 Stage 5F-1 matrix（`--batch-size 128 --ubatch-size 128 --seed 1 --temp 0 --cache-type-k f32 --cache-type-v f32 --kv-unified --parallel 2`），并保持 base / madv 两组对比口径不变。

---

## 5. 需要观察的指标

继续使用 Stage 5F-1 coverage telemetry：

```text
idle_owned_bytes
kv_total_bytes
kv_drop_bytes
idle_coverage_pct
release_ratio_total_pct
read_window_block_pct
safe_candidate_pct
remap_gate_pct
```

同时记录性能代价：

```text
resume_first_ms
total_wall_ms
tokens_per_second
```

最关键三项：

```text
idle_coverage_pct 是否提高；
release_ratio_total_pct 是否随 idle_coverage_pct 提高；
resume first-token latency 是否明显增加。
```

---

## 6. 判断标准

如果结果表现为：

```text
idle_coverage_pct 从 12.5% 提高到 25% / 31% / 37%
release_ratio_total_pct 也随之接近提高
read_window / safe / remap gate 仍不是瓶颈
```

则说明：

```text
当前机制在 high-idle-coverage workload 下可以获得更高 release ratio；
此前 12% 低收益主要由 workload 覆盖不足造成。
```

如果 idle_coverage 提高但 release_ratio 不提高，则说明：

```text
后段还有新的瓶颈，需要进一步检查 page alignment / same-window madvise bytes / resident release efficiency。
```

---

## 7. 边界

```text
1. Stage 5G-0 只设计 workload，不改源码；
2. 不修改 swap/madvise 策略；
3. 不把 high-idle-coverage workload 直接等同于真实生产负载；
4. 本阶段目的是验证收益上限，而不是证明生产性能；
5. 若 release ratio 提升，后续仍需补 latency / perf tradeoff。
```

---

## 8. 一句话结论

Stage 5G 通过 high-idle-coverage workload 验证路线 A：如果提高 idle-owned KV coverage 后 release ratio 同步上升，则说明当前 idle KV swap + madvise 机制的收益上限主要由 workload 中 idle KV 占比决定，而不是由 read-window / remap / madvise gate 限制。
