# KV Paged Read — Stage 6C-2C Auto Delayed Prefetch Scaling Results

> 本文档记录 Stage 6C-2C（auto delayed prefetch scaling）的矩阵结果、稳定性分析、RSS 解释与后续方向。

---

## 1. Goal

Stage 6C-2B 已证明 auto delayed prefetch 可以在 warmup=64、n=128 下复现 fixed D。

Stage 6C-2C 的目标是验证该策略是否能适应不同 workload：

```text
warmup ∈ {32, 64, 128}
n ∈ {128, 256}
safety ∈ {0, 4}
```

核心问题：

```text
1. warmup 变化时，remaining_blocks 是否随之变化；
2. auto_start_token 是否随 remaining_blocks 自动提前；
3. n 增大时，active window 变长，auto_start_token 是否自动推后；
4. 所有组合是否能保持 remaining=0、fallback=0、ms_max<5 ms。
```

---

## 2. Experiment Setup

配置：

```text
ctx=1024
parallel=2
batch=128
ubatch=128
cache-type-k=f32
cache-type-v=f32
kv-unified
temp=0
3-run median
```

矩阵：

```text
n ∈ {128, 256}
warmup ∈ {32, 64, 128}
mode ∈ {auto_safety0, auto_safety4}
```

---

## 3. Median Results

结果表（核心字段）：

```text
n    warmup  mode           first_ms  seq1_active_ms  total_wall_ms  tps      calls  blocks  ms_total  ms_max  remaining  auto_active_total  auto_remaining  auto_need_steps  auto_start  expected_start  start_ok  auto_every  auto_step  auto_safety  auto_window_ok  auto_started  auto_completed  auto_fallback  active_rebound_mib  rss_before_active  rss_after_active  rss_before_resume  rss_after_resume
128  32      auto_safety0   98.212    10138.056       23927.796      12.222   3      3       9.650     3.257   0          128                3               3                120         120             True      4           1          0            1               1             1               0              11.086              8383300            8394652           8394652            8366816
128  32      auto_safety4   96.797    10093.714       23756.923      12.310   4      3       9.786     3.335   0          128                3               3                116         116             True      4           1          4            1               1             1               0              11.086              8383548            8394900           8394900            8366776
128  64      auto_safety0   96.245    10014.504       27229.843      11.821   5      5       16.561    3.428   0          128                5               5                112         112             True      4           1          0            1               1             1               0              18.820              8375644            8394916           8394916            8367004
128  64      auto_safety4   97.766    10028.262       27228.662      11.839   6      5       16.497    3.346   0          128                5               5                108         108             True      4           1          4            1               1             1               0              18.820              8376228            8395500           8395500            8367212
128  128     auto_safety0   114.943   10374.545       33139.154      11.410   9      9       31.005    3.785   0          128                9               9                96          96              True      4           1          0            1               1             1               0              33.766              8362800            8397376           8397376            8366880
128  128     auto_safety4   114.963   10364.967       32928.662      11.451   10     9       30.038    3.419   0          128                9               9                92          92              True      4           1          4            1               1             1               0              35.883              8360748            8397492           8397492            8366992
256  32      auto_safety0   131.813   21301.376       48061.783      11.354   3      3       9.883     3.385   0          256                3               3                248         248             True      4           1          0            1               1             1               0              11.336              8385972            8397580           8397580            8336772
256  32      auto_safety4   131.635   21199.581       48140.841      11.323   4      3       10.067    3.395   0          256                3               3                244         244             True      4           1          4            1               1             1               0              11.340              8386232            8397844           8397844            8336908
256  64      auto_safety0   131.427   21565.194       51435.299      11.166   5      5       16.599    3.382   0          256                5               5                240         240             True      4           1          0            1               1             1               0              18.820              8378060            8397332           8397332            8336932
256  64      auto_safety4   131.302   21771.507       51523.782      11.112   6      5       17.060    3.462   0          256                5               5                236         236             True      4           1          4            1               1             1               0              18.812              8378716            8397980           8397980            8337068
256  128     auto_safety0   129.959   22310.112       58313.201      10.730   9      9       30.567    3.991   0          256                9               9                224         224             True      4           1          0            1               1             1               0              33.766              8363148            8397724           8397724            8336804
256  128     auto_safety4   131.823   22459.912       58421.660      10.722   10     9       29.979    3.471   0          256                9               9                220         220             True      4           1          4            1               1             1               0              33.773              8362576            8397160           8397160            8336984
```

---

## 4. Decision Summary

```text
所有组合均满足：
start_ok=True
remaining0=True
blocks_ok=True
max_ms<5=True
auto_ok=True
fallback=0
```

---

## 5. Scaling Interpretation

### warmup 增大时

`n=128`：

```text
warmup  remaining  safety0_start  safety4_start
32      3          120           116
64      5          112           108
128     9          96            92
```

解释：

```text
warmup 越大，seq0 需要恢复的 KV blocks 越多，auto policy 会自动更早启动 prefetch。
```

### n 增大时

`warmup=64`：

```text
n    remaining  safety0_start  safety4_start
128  5          112           108
256  5          240           236
```

解释：

```text
同样恢复 5 个 blocks，active window 从 128 tokens 增加到 256 tokens 后，auto policy 会自动把 prefetch 推迟到更靠后的位置，从而延长低 RSS 窗口。
```

---

## 6. Prefetch Stability

```text
1. 所有组合 remaining=0；
2. 所有组合 fallback=0；
3. 所有组合 ms_max<5 ms；
4. warmup=128 时需要恢复 9 个 blocks，auto policy 仍能稳定完成；
5. safety4 比 safety0 多一次调用，但不增加恢复 blocks 数，只是更保守地提前启动。
```

---

## 7. RSS Interpretation

```text
active_rebound_mib 随 restored blocks 增加而增加。
```

例子：

```text
warmup  remaining  active_rebound_mib
32      3          ≈11.09 MiB
64      5          ≈18.82 MiB
128     9          ≈33.77 MiB
```

解释：

```text
需要恢复多少 blocks，RSS 就会回升多少。Stage 6C-2C 的收益是自动推迟和调度 RSS 回升，而不是消除 prefetch 后的 RSS rebound。
```

---

## 8. Limitations

必须写：

```text
Stage 6C-2C 仍未解决 prefetch 抵消净 RSS 收益的问题。
```

原因：

```text
1. 所有成功组合都是 resume 前 remaining=0；
2. 这意味着目标 blocks 已经恢复完；
3. RSS 在 resume 前已经回升；
4. 当前阶段证明的是 auto delayed prefetch 的自适应调度能力，不是最终内存收益保留能力。
```

---

## 9. Conclusion

```text
Stage 6C-2C 验证了 auto delayed prefetch policy 的 workload 自适应能力。在 warmup=32/64/128 和 n=128/256 的 3-run median 矩阵中，auto policy 均能正确 probe remaining swapped blocks，并根据 active_total_tokens 自动计算 start token。所有组合均满足 start_ok=True、remaining=0、fallback=0、ms_max<5 ms，说明该策略可以根据 KV footprint 和 active window 自动调整 prefetch 时机。
```

---

## 10. Next Direction

建议下一阶段：

```text
Stage 6C-3: scheduler-level resume-aware prefetch policy
```

方向：

```text
1. 把 example driver 中确定性的 auto prefetch 策略迁移到更接近真实多请求调度的策略；
2. 引入 resume prediction / resume-pending 状态；
3. 根据内存压力决定是否 prefetch；
4. 根据 active window 和 remaining blocks 动态决定 prefetch start；
5. 暂不直接做 true async thread。
```
