# Stage 4C-2B — multi-request / idle-request KV swap 仿真（Route B, exact）

前置：

- [docs/kv_paged_trace_sim_stage4c1_results.md](kv_paged_trace_sim_stage4c1_results.md) — 单请求 exact 仿真（结论：无安全冷块）
- [docs/kv_paged_routeb_sink_recent_sweep.md](kv_paged_routeb_sink_recent_sweep.md) — sink+recent approximate 扫描
- 脚本：[scripts/kv_paged_routeb_multi_request_sim.py](../scripts/kv_paged_routeb_multi_request_sim.py)
- 增长模板来源 trace：`/root/oscomp/kv_logs/paged_stage4c_trace_active_ctx4096_n512/trace.txt`
- 摘要 CSV：`/root/oscomp/kv_logs/paged_stage4c_routeb_multi_request/sim_summary.csv`

本文是 Stage 4C-2B 结果文档：新增独立仿真器评估 **multi-request / idle-request** 场景下 KV swap 的理论收益与恢复代价。**只改 Python 脚本和文档，不改 C++，不 build，不跑 llama，不 commit。**

结论先行：在多请求服务里，**idle 请求的 KV 是真正的冷数据**——idle 期间该请求一个 block 都不读，因此换出再换回是 **exact（保 sha256）** 的，与 Stage 4C-2A 的 approximate sink+recent 有本质区别。仿真显示按 idle 换出可把 **平均** resident KV 从 ~507 MiB 降到 ~233 MiB（节省 ~274 MiB），代价是 resume 时把整请求 KV 换回（单次最多 336 MiB）。但它**降不了绝对峰值**：峰值出现在所有请求同时 active 的时刻，那时没有 idle 可换。

---

## 1. synthetic workload 假设

没有现成多请求 trace，故构造 synthetic workload，每个请求的 KV 增长**复用单请求 trace 的实测模式**：`blocks_in_use` 从 prompt 起每 16 个 active step 增长 1 block（trace 实测 33 个增长点、步距 16）。请求带 `prompt_blocks`（到达时即驻留的 prompt KV）和 `cap`（上限）。

全局 512 步 timeline，每个请求每步处于 `active / idle / finished`（或尚未到达）：

| req | 到达 | 画像 | spans（active/idle/active …） | prompt_blocks | cap |
|---|---|---|---|---|---|
| A | step 0   | 长对话：解码→idle（用户思考/工具）→resume | active160, idle200, active152 | 24 | 80 |
| B | step 120 | A idle 期间到达并解码 | active240, idle80, active40 | 20 | 64 |
| C | step 60  | 短请求，两段突发后结束 | active48, idle24, active48 | 6 | 16 |
| D | step 20  | 长上下文：大 prompt，长 idle，低频 resume | active64, idle300, active64 | 80 | 120 |

设计意图：

- idle 窗口**错峰**（A 的 idle 被 B 的 active 填充；D 长期 idle），这样不同请求在不同 global step 变冷，才有 exact swap 的空间。
- D 是典型的“大 prompt + 长 idle + 低频 resume”——单请求最值得换出、最受益。
- 峰值并发 live = 4（四个请求同时未完成）。

block 默认 `block_bytes=4194304`（4 MiB）、`block_size=16`。

---

## 2. 为什么 multi-request / idle 更适合 exact swap

Stage 4C-1 已证明：单请求 exact full attention 下，`active_read_blocks` 每步覆盖几乎全部 `blocks_in_use`，没有空闲块；任何换出都会被下一步强制换回（thrashing），这也是 Stage 4B-RSS 强推 swap 时出现 `swap_out=swap_in=8000` 的根因。

多请求场景打破了这一点：

- 一个请求 **idle** 时，它本步**不读自己任何 KV block**（不在 decode）。
- 因此它的整组 block 在 idle 窗口内是真冷数据，换出期间无人访问。
- resume 时一次性换回全部 block，注意力看到的 KV 与从未换出时**逐位一致**——这是 exact，不丢任何历史。

换言之，冷的不是“某请求历史的中段”（那要靠近似才冷），而是“整个 idle 请求”（天然冷、且 exact）。

---

## 3. 仿真结果摘要

```
steps=512 requests=4 peak_concurrent_live=4 block_bytes=4194304
all_resident 基线: avg 506.7 MiB / peak 648.0 MiB
```

| policy | params | resident avg (MiB) | resident peak (MiB) | saved avg (MiB) | saved peak (MiB) | req swap_out/in | swap I/O (GiB) | resume w/ swap-in | avg / max resume swap-in (MiB) |
|---|---|---|---|---|---|---|---|---|---|
| all_resident | - | 506.7 | 648.0 | 0.0 | 0.0 | 0/0 | 0.0 | 0 | 0 / 0 |
| active_only | - | 233.1 | 648.0 | 273.6 | 0.0 | 4/4 | 1.266 | 4 | 162 / 336 |
| idle_swap_T | T=8  | 243.3 | 648.0 | 263.4 | 0.0 | 4/4 | 1.266 | 4 | 162 / 336 |
| idle_swap_T | T=16 | 253.4 | 648.0 | 253.3 | 0.0 | 4/4 | 1.266 | 4 | 162 / 336 |
| idle_swap_T | T=32 | 273.1 | 648.0 | 233.6 | 0.0 | 3/3 | 1.195 | 3 | 204 / 336 |
| idle_swap_T | T=64 | 311.3 | 648.0 | 195.4 | 0.0 | 3/3 | 1.195 | 3 | 204 / 336 |
| lru_request_swap_budget | budget=64  | 234.8 | 648.0 | 271.9 | 0.0 | 3/3 | 1.195 | 3 | 204 / 336 |
| lru_request_swap_budget | budget=96  | 294.5 | 648.0 | 212.2 | 0.0 | 2/2 | 0.930 | 2 | 238 / 336 |
| lru_request_swap_budget | budget=128 | 318.1 | 648.0 | 188.6 | 0.0 | 2/2 | 0.930 | 2 | 238 / 336 |
| lru_request_swap_budget | budget=192 | 506.7 | 648.0 | 0.0 | 0.0 | 0/0 | 0.0 | 0 | 0 / 0 |
| hybrid_idle_budget | T=32,budget=64  | 234.8 | 648.0 | 271.9 | 0.0 | 3/3 | 1.195 | 3 | 204 / 336 |
| hybrid_idle_budget | T=32,budget=96  | 249.9 | 648.0 | 256.8 | 0.0 | 3/3 | 1.195 | 3 | 204 / 336 |
| hybrid_idle_budget | T=32,budget=128 | 270.9 | 648.0 | 235.8 | 0.0 | 3/3 | 1.195 | 3 | 204 / 336 |
| hybrid_idle_budget | T=32,budget=192 | 273.1 | 648.0 | 233.6 | 0.0 | 3/3 | 1.195 | 3 | 204 / 336 |

完整指标（含 thrash 列，全为 0）见 CSV。所有策略 `thrash_in_4_steps=0`：idle 远长于换出/换回间隔，没有抖动。

### 3.1 内存收益

- **平均 resident 收益显著**：`active_only` 把平均从 507 MiB 压到 233 MiB（省 274 MiB ≈ 54%）。`idle_swap_T=8` 几乎一样（省 263 MiB），但更宽容（短暂 idle 不立刻换出）。
- **T 越大收益越小**：T=64 只省 195 MiB——长阈值让请求在换出前停留太久。
- **budget 策略**在预算 < 峰值并发占用时才生效：budget=192 ≥ 峰值附近 → 不触发、收益 0；budget=64 触发最积极、省 272 MiB。

### 3.2 swap-in 恢复代价

- resume 是**整请求换回**：avg 162–238 MiB / 次，**最大 336 MiB**（D 的大 idle footprint）。这是单步的 I/O 尖峰，会拉长该请求 resume 的首 token 延迟。
- swap I/O 总量 0.9–1.3 GiB（全程 512 步），由换出+换回的 block 数 ×4 MiB 得到。
- **关键**：本仿真所有 swap_in 都发生在 resume 那一步，thrash=0——不像 Stage 4B-RSS 那样每步换回。代价是“恢复延迟”，不是“持续抖动”。

### 3.3 一个诚实的局限：峰值降不了

所有策略 `saved_mb_peak=0`。峰值 resident（648 MiB）出现在**四个请求同时 active** 的 global step，此刻没有 idle 请求可换出。idle-swap 降的是**平均 / 持续** footprint，不是**绝对峰值**。要降峰值必须在 active 请求之间做取舍（牺牲并发或牺牲正确性），不在本 exact 路线内。

---

## 4. 与既有路线的区别

### 4.1 vs Stage 4C-2A sink+recent（approximate）

| 维度 | sink+recent (4C-2A) | multi-request idle swap (本文) |
|---|---|---|
| 冷数据来源 | 单请求历史的**中段** | **整个 idle 请求** |
| 正确性 | approximate，丢中间历史的读，**不保证 sha256** | **exact**，idle 期间无人读，换回逐位一致 |
| 降峰值 | 能（中段被丢） | 不能（峰值在全 active 时刻） |
| 降平均 | 能 | 能（~54%） |
| 落地前提 | sparse/windowed attention 等语义变更 | 多请求调度 + idle 检测，无语义变更 |

两者正交：sink+recent 用正确性换峰值，idle-swap 用恢复延迟换平均占用。

### 4.2 vs Stage 4B-RSS 强制 live-history swap

Stage 4B-RSS 在**单请求**里强制换出 live history，撞上“每步都读全历史”的访问模式，出现 `swap_out=swap_in=8000` 的持续 thrashing——换出立刻被换回，RSS 没真正降、I/O 爆炸。本文换出的是**不被读的 idle 请求**，thrash=0，swap-in 只在 resume 时发生一次。区别就是“换出的东西当下到底有没有人读”。

---

## 5. 推荐后续落地策略

**`idle_swap_T`，T 取 8–16（偏小），作为首个 exact 落地路线。**

理由：

- 收益逼近 `active_only`（省 ~253–263 MiB 平均）但更稳——短暂停顿不会立刻触发换出/换回。
- 实现简单：只需 per-request “上次 active step”计数 + idle 阈值，不需要全局内存预算这一额外旋钮。
- thrash=0，exact，对输出无影响，只引入 resume 首 token 延迟。

`hybrid_idle_budget`（idle 阈值 + 内存预算）作为**第二步**：在内存压力大时，用预算把更多 idle 请求逼出去；当前 workload 下它与 `idle_swap_T` 收益相当，价值要在更高并发 / 更紧内存场景才体现。

落地注意点：

- resume 的 swap-in 是单步 I/O 尖峰（最大 336 MiB），应预取 / 异步换入以摊薄首 token 延迟。
- 需要真实多请求 trace 校准 workload（当前 prompt/idle 时长是合成假设）；建议下一步在 server 上采集多并发 `LLAMA_KV_PAGED_TRACE`，用真实 idle 分布重跑本仿真。

---

## 6. 复现实验

```bash
python3 scripts/kv_paged_routeb_multi_request_sim.py \
  --csv /root/oscomp/kv_logs/paged_stage4c_routeb_multi_request/sim_summary.csv
```

可调：`--steps --block-bytes --block-size --idle-thresholds 8,16,32,64 --budgets 64,96,128,192`。workload 在脚本 `build_workload()` 中定义。
