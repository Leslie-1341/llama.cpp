# Stage 8F: Unified KV Reclaim Policy 设计文档

## 0. 文档性质与边界

本文件是 **设计文档**，不修改任何源码。Stage 8F 只新增本文档。

目标：把当前仓库中两条相互独立演进的 KV 内存优化线，统一成一个**单一设计口径**：

```text
KV cache lifecycle-aware resident memory management
```

即：根据 KV block 在生命周期中所处的区域（unused / active / idle / swapped / resume-needed），
选择合适的 resident memory 回收或恢复机制，而不是把它们当作两个割裂的功能点。

本文档严格区分两类内容：

```text
[已实现]  当前仓库中已有的 env gate / 函数 / 机制；
[未来设计] 尚未实现、Stage 8F 仅给出接口与决策框架的部分。
```

凡是标注 `[未来设计]` 的内容都不代表当前代码已有该行为。

---

## 1. 阅读到的源码与文档

本设计文档对齐以下实际存在的文件（均已逐一查看，路径真实存在）：

```text
src/llama-kv-cache.cpp
src/llama-kv-cache.h
examples/kv-idle-swap-resume/idle-swap-resume.cpp
scripts/kv_stage8e_kv_footprint_audit.sh
scripts/parse_stage8e_kv_footprint_audit.py
docs/kv_paged_read_stage8d_resume_latency_defer_results.md
docs/kv_paged_read_stage8e_kv_footprint_residency_audit.md
```

任务描述中所列文件路径全部存在，没有缺失，因此无需替换或编造路径。

从源码中确认的关键事实（用于保证本文档与代码一致）：

* 两条机制使用的是**不同的 env gate**，并非同一开关：

  * 已写入 KV 的 idle swap / madvise 线：

    ```text
    LLAMA_KV_PAGED_IDLE_SWAP            (src/llama-kv-cache.cpp:442)
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE    (src/llama-kv-cache.cpp:443)
    ```

    配套 resume 路径：

    ```text
    llama_kv_cache::set_seq_prefetch_protected()  (src/llama-kv-cache.cpp:1345)
    llama_kv_cache::defer_idle_swapout()          (src/llama-kv-cache.cpp:1352)
    llama_memory_prefetch_seq() / _seq_step()     (driver 调用)
    ```

  * 未写入有效 token 的 free / tail capacity 线：

    ```text
    LLAMA_KV_LAZY_CLEAR  -> kv_lazy_clear / clear_frontier_advance()  (src/llama-kv-cache.cpp:720, 4205)
    LLAMA_KV_LAZY_TAIL   -> kv_lazy_tail  / madvise_tail()            (src/llama-kv-cache.cpp:825, 4035)
    ```

* `madvise_tail()` 只处理 `[GGML_PAD(n_kv, 256), kv_size)` 这段**未使用尾部容量**，
  并显式保证该区间永远不与 `[0, n_kv)` 读窗口重叠（src/llama-kv-cache.cpp:4058-4064）。

* `clear_frontier_advance()` 只在 `kv_lazy_clear` 下，把尾部区域保持未提交，
  当 `n_kv` 增长时按 256 对齐推进 frontier，并在 graph 读 K/V 前把新可读行清零
  （src/llama-kv-cache.cpp:4205-4249）。

* `defer_idle_swapout()` 的计数单位是 **row_idx fill**，不是 `llama_decode` 次数
  （已在 Stage 8D 文档 §9.2 说明，源码 src/llama-kv-cache.cpp:4754-4773 一致）。

* 这两条线目前各自独立，**没有统一的 policy 入口**。Stage 8F 要补的正是这个统一口径。

---

## 2. 为什么需要 unified policy

### 2.1 两条机制不矛盾

很容易把这两条机制误解为竞争关系，实际不是：

```text
tail / lazy clear / free-block release：
    处理「尚未写入有效 token」的 unused / free / tail KV 容量。

idle KV swap / madvise：
    处理「已经写入有效 KV、但当前 request/seq idle」的 used-but-idle KV。
```

它们处理的是 **KV cache 生命周期中的不同区域**，因此可以同时存在、互不冲突。
矛盾感来自缺少一个统一的描述框架，而不是机制本身。

### 2.2 KV 区域划分

把一份 KV cache 的容量沿生命周期切分，可以得到下列互斥区域：

```text
unused / free / tail KV capacity   预留但尚未写入有效 token 的容量
used-but-active KV                 已写入、且当前 active seq 正在使用
used-but-idle KV                   已写入、但当前持有者 seq 处于 idle/paused
mixed / shared KV                  同一 block 被多个 seq 共享，含 active 成员
swapped / nonresident KV           已被 swap/madvise，物理页 nonresident
resume-needed KV                   idle/swapped，但即将 resume，需要恢复访问
```

Stage 8E 的 footprint 数据正好印证了这种划分的必要性：

```text
kv_total   = 511.750 MiB   (总容量，含 unused/free/tail)
kv_inuse   = 208.000 MiB   (实际写入有效 KV 的规模)
idle_owned = 136.000 MiB   (used-but-idle，当前机制的主要释放对象)
```

也就是说，「总容量」「实际 in-use」「idle-owned」是三个不同量级的概念，
不能用一条机制、一个分母去解释全部内存收益。这正是需要 unified policy 的根本原因。

---

## 3. 当前两条机制分别覆盖什么场景

### 3.1 低并发 / 中短上下文

特征与适配机制：

```text
KV 总容量可能较大（ctx 预留大）；
实际 in-use KV 较小；
主要浪费来源 = unused / free / tail capacity；
合适机制 = tail madvise / lazy clear / free-block release。
```

对应已实现机制：

```text
[已实现] LLAMA_KV_LAZY_CLEAR -> clear_frontier_advance()
[已实现] LLAMA_KV_LAZY_TAIL  -> madvise_tail()
```

这条线不依赖多 seq idle，只要 ctx 预留远大于实际使用，就有收益空间。

### 3.2 高并发 / 长上下文

特征与适配机制：

```text
KV in-use 较高；
多个 session 的历史 KV 已经写入；
部分 session idle；
主要可释放对象 = idle-owned KV；
合适机制 = idle swap / madvise + prefetch / resume defer。
```

对应已实现机制：

```text
[已实现] LLAMA_KV_PAGED_IDLE_SWAP / _MADVISE
[已实现] prefetch (llama_memory_prefetch_seq / _seq_step)
[已实现] LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME -> defer_idle_swapout()
[已实现] set_seq_prefetch_protected()
```

这条线依赖「已写入但当前不被 active 需要」的 KV，是 Stage 8C/8D/8E 的主线。

### 3.3 resume 阶段（两条线都会触及）

```text
swapped / nonresident KV 在 resume 前需要恢复访问；
合适机制 = prefetch + resume first-token defer；
目标 = 保护 resume first-token latency。
```

Stage 8D 已证明该路径有效（resume_first_ms 从 104.210 ms 降回 84.819 ms baseline 范围）。

---

## 4. 统一 KV block lifecycle

### 4.1 逻辑状态分类

设计一个**逻辑状态分类**，用当前实现已有概念对齐，不要求新增代码：

```text
UNUSED / FREE / TAIL     未写入有效 token 的预留容量
ACTIVE_OWNED             已写入，当前被 active seq 使用
IDLE_OWNED               已写入，持有者 seq 当前 idle
MIXED / SHARED           被多 seq 共享，且至少一个成员是 active
SWAPPED / NONRESIDENT    已 swap/madvise，物理页 nonresident
PREFETCHED / RESTORED    曾 swapped、已被 prefetch 拉回 resident
```

### 4.2 各状态的可释放性与处理机制

| 逻辑状态 | 是否可释放 | safety 条件 | 处理机制 | 实现状态 |
| --- | --- | --- | --- | --- |
| UNUSED / FREE / TAIL | 可释放（容量层面） | 区间必须在 `[0, n_kv)` 读窗口之外；未来写入前必须正确 clear/init | `madvise_tail()` / `clear_frontier_advance()` / free-block release | tail/clear [已实现]；free-block release [未来设计] |
| ACTIVE_OWNED | **不可释放** | 任意情况下都保护 | 无（始终常驻） | [已实现] 保护语义 |
| IDLE_OWNED | 可释放 | 不属于 active seq；不在 active read window；通过 visible 检查 | idle swap / madvise，resume 前 prefetch | [已实现] |
| MIXED / SHARED | 谨慎/默认不释放 | 只要含 active 成员就按 active 处理 | 默认保留；未来可做 per-seq 拆分 | [已实现] 保守保留 |
| SWAPPED / NONRESIDENT | 已释放（保持） | resume 前必须可恢复 | 保持 nonresident；按需 prefetch | [已实现] |
| PREFETCHED / RESTORED | 不再释放（窗口内） | resume first-token 期间不得被新一轮 swap-out 顶掉 | `set_seq_prefetch_protected()` + `defer_idle_swapout()` | [已实现] |

关键不变量（与源码一致）：

```text
active-owned KV 永不释放；
active read window [0, n_kv) 内的 KV 永不被 tail madvise 触碰（madvise_tail 显式保证）；
mixed/shared block 只要含 active 成员，就整体按 active 处理；
swapped KV 在 resume 前必须能被 prefetch 拉回；
tail/free release 区间必须保证未初始化内容不会被 attention 读取。
```

### 4.3 Tail/free release 的恢复语义

tail/free release 的「恢复」语义与 idle swap 完全不同，必须分开理解。

tail/free release **不是**重新 `malloc` / 重新分配 KV cache。更合理的机制是：

```text
保留原 KV buffer 的虚拟地址和逻辑容量不变；
通过 madvise(MADV_DONTNEED) 释放 unused/free/tail 区域的物理页；
未来 KV 增长写到这些地址时，由 OS page fault 自动重新分配物理页并重新变 resident。
```

也就是说，逻辑地址空间始终在，物理页按需消失/回来；这正是 `madvise_tail()`
(src/llama-kv-cache.cpp:4035) 的工作方式——它只对页对齐的尾部区间发 `MADV_DONTNEED`，
不改变 tensor 的逻辑布局。

与 idle-owned KV 的本质区别：

```text
unused / free / tail KV：
  没有有效历史内容需要保留；
  madvise 后内容丢失是可接受的；
  未来首次使用前必须重新完整写入 K/V，并保证 clear / init 语义正确
  （对应 clear_frontier_advance() 在 graph 读 K/V 前清零新可读行）。

idle-owned KV：
  已经写入有效历史 K/V；
  未来 resume 还要读取；
  因此不能只丢弃内容，必须通过 swap / prefetch / fallback 恢复原内容。
```

一句话总结：

```text
tail/free release = 丢弃 + 未来重写（内容无需保留）；
idle swap        = 保存 + 未来恢复（内容必须保留）。
```

---

## 5. 统一策略接口设计 `[未来设计]`

### 5.1 提议接口

提议一个未来可实现的统一入口：

```text
LLAMA_KV_RECLAIM_POLICY=off|tail_only|idle_only|auto
```

含义：

```text
off:
  不做额外 KV resident reclaim（等价于全部 reclaim gate 关闭）。

tail_only:
  只处理 unused/free/tail KV capacity。
  内部对应 kv_lazy_clear / kv_lazy_tail / free-block release。

idle_only:
  只处理 used-but-idle KV blocks。
  内部对应 idle swap/madvise + prefetch + resume defer。

auto:
  根据运行时信号动态选择 tail path / idle path / 两者，或保持常驻。
```

### 5.2 命名理由与兼容性

* 现有 env gate（`LLAMA_KV_LAZY_TAIL`、`LLAMA_KV_PAGED_IDLE_SWAP` 等）保留为**底层 gate**，
  `LLAMA_KV_RECLAIM_POLICY` 作为**上层 policy facade**，向下展开成已有 gate 组合。
  这样可避免一次性重写现有实验脚本，且保持 Stage 8C–8E 结果可复现。
* 之所以用 `tail_only` / `idle_only` 而非机制名（如 `lazy` / `swap`），
  是因为 policy 层应该按「处理哪类 KV 区域」命名，而不是按底层 syscall 命名，
  这样未来即使底层换实现（例如 free-block release 取代 tail madvise），policy 名仍成立。
* `off` 必须是默认值，保证不开启 policy 时行为与当前主线完全一致。

---

## 6. auto policy 的决策逻辑 `[未来设计]`

### 6.1 输入信号

auto 需要的运行时输入（多数已有 telemetry 对应，见 §9）：

```text
kv_total_capacity          总容量
kv_inuse                   已写入有效 KV
idle_owned_bytes           used-but-idle 规模
idle_age                   idle 持续时长 / step 数
unused_tail_or_free_bytes  unused/free/tail 容量
rss_pressure               进程/系统 RSS 压力
resume_pending             是否有 seq 即将 resume

logical_free_blocks        逻辑层面仍可写入的 block 数（容量信号，非物理）
kv_inuse_ratio             kv_inuse / kv_total
unused_tail_resident_bytes unused/tail 区域中当前仍 resident 的字节
growth_frontier_block      当前已写入 block frontier
growth_reserve_blocks      growth frontier 之上预留、不允许释放的 block 数
page_fault_or_refault_count tail release 后写回造成的 page fault / refault 计数
```

### 6.2 分层决策伪代码

```text
# 优先级从上到下，靠前的规则先生效

if resume_pending:
    prefetch needed swapped blocks            # 恢复 resume-needed KV
    defer new swap-out on resume first token  # 保护 resume first-token latency
    # 注意: 不在 resume 关键路径上做新的 idle swap-out

if logical_free_blocks low:
    trigger capacity policy, not reclaim-only policy
    # 逻辑容量将耗尽: 需 context shift / 释放 finished request / 降并发 / 排队
    # 这不是 resident reclaim 能解决的问题 (见 §6.4)

if memory_pressure_low:
    keep resident                             # 低压力时不主动回收，避免 page-fault 开销
    return

if unused_tail_or_free_resident_bytes large:
    reclaim only blocks beyond growth reserve # 只回收 growth frontier + reserve 之外的 block (见 §6.5)

if idle_owned_bytes large and idle_age >= threshold:
    swap / madvise idle-owned blocks          # 再回收 used-but-idle

# 任何分支下都恒成立的保护：
always protect active-needed blocks
```

### 6.3 必须强调的安全约束

```text
active-owned KV 不释放；
active read window 内 KV 不释放；
mixed / shared block 谨慎处理（含 active 成员即按 active 保留）；
resume-needed KV 优先 prefetch，不优先回收；
unused tail/free release 必须保证未初始化区域不会被 attention 读取；
未来写入这些区域前必须保证 clear / initialization 语义正确。
```

设计上「先 tail/free，后 idle」的顺序是有意的：

```text
tail/free release 处理的是从未写入有效内容的容量，安全成本最低；
idle swap/madvise 处理的是已写入、idle 的有效 KV，恢复时有 page-fault / prefetch 成本。
因此优先回收更廉价、风险更低的 unused/free/tail。
```

### 6.4 逻辑容量与物理 resident memory 的分离

auto policy 必须区分**两个互相独立的层面**，否则会用错机制：

```text
logical KV capacity:
  由 ctx-size / kv_size / parallel 等决定；
  tail/free madvise 不会扩大逻辑容量（地址空间和 kv_size 不变）；
  如果逻辑 KV 写满，正确做法是：
    context shift、释放 finished request、降低并发、
    拒绝/排队新请求，或未来做更完整的 paged KV allocator。

physical resident memory:
  由当前哪些 KV pages resident 决定；
  tail/free release 和 idle swap/madvise 主要降低这一层的 RSS。
```

关键结论：

```text
tail/free release 解决的是物理 RSS，不解决逻辑 KV 容量不足。
```

这就是 §6.2 中 `if logical_free_blocks low: trigger capacity policy` 与
`reclaim-only` 分支必须分开的原因：逻辑容量耗尽是 capacity policy 的职责，
不能靠 resident reclaim 缓解。

### 6.5 Growth frontier reserve / hot reserve `[未来设计]`

新增保护原则：

```text
不要释放马上可能写入的下一段 KV。
```

具体规则：

```text
current_used_blocks = 当前已写入 block frontier (growth_frontier_block)
reserve_blocks      = 4 或 8，具体后续实测确定 (growth_reserve_blocks)

只允许释放:
  block_id >= current_used_blocks + reserve_blocks

保护 (不释放):
  [current_used_blocks, current_used_blocks + reserve_blocks)
```

目的：

```text
避免刚 madvise 掉的 tail block 在下一轮 decode 立刻被写回；
减少 page fault 抖动；
降低 tail release 对 TPS / latency 的影响。
```

注意：现有 `madvise_tail()` 已用 `GGML_PAD(n_kv, 256)` 把释放下界推到读窗口之外，
这是一种隐式的对齐保护；growth reserve 是在此之上、更显式且可调的「热区预留」，
属于 `[未来设计]`。

### 6.6 融合策略的收益与代价预估

基于 Stage 8E footprint 的**粗略工程估算**（非实测）：

```text
kv_total            = 511.750 MiB
kv_inuse            = 208.000 MiB
idle_owned          = 136.000 MiB
unused/free capacity ≈ 511.750 - 208.000 = 303.750 MiB
当前 idle-only final RSS drop ≈ 86.422 MiB
当前 final KV nonresident   ≈ 90.000 MiB
```

收益侧：

```text
unused/free capacity 约 303.75 MiB 是 tail/free path 的潜在对象；
但不能直接认为全部都能变成 RSS drop；
需要扣除 growth reserve，并乘以实际 madvise -> RSS 转化率。
```

非实测粗估：

```text
在当前 Stage 8E workload 下，如果 tail/free release 也能安全启用，
额外 RSS drop 可能达到 150–240 MiB 级别；
融合策略总 final RSS drop 可能从 idle-only 的约 86 MiB
提升到 200–300 MiB 级别。
```

必须注明：

```text
这是基于 Stage 8E footprint 的粗略上界 / 工程估算；
不是已实测结果；
需要后续 tail/free release probe 验证。
```

代价侧：

```text
tail/free release 的主要代价是未来写入 released tail 时的
page fault / 清零 / 分配物理页成本；
保守策略 (带 growth reserve)  : 可能 0–3% TPS 下降；
较激进策略                   : 可能 3–8% TPS 下降；
无 growth reserve、频繁释放又写回: 可能超过 10%，或造成明显 latency 抖动。
```

idle swap 代价引用现有结论：

```text
idle swap 如果没有 prefetch / defer，会增加 resume latency；
Stage 8D 已证明 full prefetch + resume defer 可把
resume first-token latency 拉回 baseline 附近。
```

---

## 7. 与 Stage 8D / 8E 的关系

### 7.1 Stage 8D 提供 latency 结论

Stage 8D（clean perf, `LLAMA_KV_PAGED_MINCORE=0`）证明：

```text
idle swap/madvise + full prefetch + resume defer
能在保持 correctness 的前提下降低 RSS，
并把 resume first-token latency 从 104.210 ms 拉回 84.819 ms（baseline 范围）。
```

这为 unified policy 的 `idle_only` / `auto` 路径提供了 latency 可控性依据。

### 7.2 Stage 8E 提供 footprint 结论

Stage 8E（diagnostic, `LLAMA_KV_PAGED_MINCORE=1`）证明：

```text
RSS 下降确实对应 KV resident pages 释放（不是单纯 telemetry 计数）；
diagnostic workload 中 final KV nonresident ≈ 90 MiB，final process RSS drop ≈ 86.4 MiB，量级一致；
同时 KV total capacity 中还存在 unused/free capacity，可作为 tail/lazy clear 的适用对象。
```

这同时支撑了 unified policy 的两条线：

```text
final KV nonresident ≈ 90 MiB / RSS drop ≈ 86.4 MiB  ->  idle_only 收益证据
kv_total 511.75 MiB 远大于 kv_inuse 208 MiB           ->  tail_only 有空间
```

### 7.3 重要口径限制

```text
Stage 8E 的 MINCORE run 是 memory-accounting diagnostic，
不能当作 latency / TPS 性能结论。正式性能仍以 Stage 8D clean perf 为准。
```

---

## 8. 与后续 Stage 9 的关系

Stage 9 应当做：

```text
semi-real multi-session workload / simplified scheduler driver
```

目的：

```text
让 active / paused / resume 状态来自更自然的多会话请求流，
而不是 example driver 写死的固定时序；
验证 idle-owned KV 是否能在真实/半真实文本场景下稳定产生；
验证 unified policy 在不同 workload 下如何选择 tail path 或 idle path。
```

明确建议：

```text
Stage 9 暂不直接接入 llama-server，先做 simplified scheduler driver。
理由：保持实验可控、可复现、可归因，避免一次性引入 server scheduler 的复杂度。
```

这与 Stage 8D §11 / Stage 8E §18 的下一步建议一致。

---

## 9. 实现建议与非目标

### 9.1 实现性质

```text
Stage 8F 是设计文档，不做源码改动。
```

### 9.2 未来实现建议（均为 `[未来设计]`）

```text
policy wrapper        : LLAMA_KV_RECLAIM_POLICY facade，向下展开成现有 gate 组合
metric panel          : 把 §6.1 输入信号统一汇总到一处 telemetry
env gate              : policy 与底层 gate 的优先级 / 覆盖关系
tail/free release probe: 验证 unused/free/tail 回收的安全性与收益（先 diagnostic）
auto policy smoke     : 验证 auto 在不同信号组合下的选路正确
semi-real workload matrix: 配合 Stage 9，覆盖 tail-heavy 与 idle-heavy 两类场景
growth reserve probe      : 实测 reserve_blocks 取值对 page fault 抖动 / TPS 的影响
tail page-fault cost probe: 量化释放后写回 tail 的 page fault / 清零 / 分配成本
logical capacity vs physical residency telemetry: 把 §6.4 两层信号分开上报
```

建议实现顺序：先 metric panel（让 auto 有可靠输入），再 policy wrapper，
最后才是 tail/free release probe 与 auto smoke。

### 9.3 非目标

```text
不重写完整 llama-server scheduler；
不实现 vLLM-style full paged KV allocator；
不保证所有场景都有高 RSS drop（收益取决于 workload）；
不把 MINCORE diagnostic path 用作性能路径。
```

---

## 10. 结论

1. 当前仓库存在两条独立的 KV 内存优化线，且**使用不同的 env gate**：

   ```text
   tail/lazy clear : LLAMA_KV_LAZY_TAIL / LLAMA_KV_LAZY_CLEAR   (处理 unused/free/tail)
   idle swap       : LLAMA_KV_PAGED_IDLE_SWAP(_MADVISE) + prefetch + defer (处理 used-but-idle)
   ```

2. 二者不矛盾，分别覆盖 KV 生命周期中的不同区域，可统一为：

   ```text
   KV cache lifecycle-aware resident memory management
   ```

3. Stage 8E footprint（kv_total 511.75 / kv_inuse 208 / idle_owned 136 MiB）证明
   需要按区域、而非按单一分母解释内存收益，这是 unified policy 的根本动机。

4. 提议统一入口（`[未来设计]`）：

   ```text
   LLAMA_KV_RECLAIM_POLICY=off|tail_only|idle_only|auto
   ```

   作为上层 facade，向下复用已有 gate，默认 `off` 保证行为不变。

5. auto 决策遵循「resume 优先保护 → 低压力保持常驻 → 先回收 tail/free → 再回收 idle-owned →
   始终保护 active」的分层逻辑。

## 11. 限制

```text
本文档是设计口径，未实现 LLAMA_KV_RECLAIM_POLICY，auto 决策为伪代码；
当前 free-block release 尚未实现，tail 线仅有 lazy clear / tail madvise；
defer 计数单位仍是 row_idx fill，server scheduler 下需重新审视（见 Stage 8D §9.2）；
Stage 8E footprint 来自 controlled multi-seq workload，非真实线上请求流；
未接入 llama-server 真实 slot / continuous batching scheduler。
tail/free release 只能降低 resident RSS，不能扩大 logical KV capacity；
融合策略的收益 / 代价目前是估算，尚未实测；
auto policy 需要防止 tail release 与未来 KV growth 形成频繁 page fault。
```

## 12. 下一步

```text
1. Stage 9 semi-real multi-session workload / simplified scheduler driver；
2. 基于 §6.1 输入信号实现 metric panel；
3. 实现 LLAMA_KV_RECLAIM_POLICY facade（off/tail_only/idle_only 先行，auto 最后）；
4. tail/free release probe + auto policy smoke，在 tail-heavy 与 idle-heavy 两类 workload 上验证选路。
```
