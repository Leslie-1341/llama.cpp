# KV runtime swap / offloading Stage 2 设计阅读稿

> 操作系统功能赛技术报告素材 · Stage 2 read 篇
>
> **定位**：承接 Stage 1 `KV Lazy-Block + memory-aware warmup policy`，重新整理 runtime KV swap / offloading 的设计边界。本文只做文档设计，不恢复旧 swap 代码，不进入 prefetch，不进入 PagedAttention-style paged read。
>
> 文档日期：2026/06/09

---

## 1. 背景与阶段边界

Stage 1 已经解决的是 **未使用 KV tail capacity** 的物理页提交问题：

- `LLAMA_KV_LAZY_CLEAR`：构造期不再 full clear 整个 KV buffer，只清理 `[0, clear_frontier)` 前缀，未使用 tail 不触碰、不提交物理页，目标是降低 peak RSS。
- `LLAMA_KV_LAZY_TAIL`：运行期对 `[GGML_PAD(n_kv,256), kv_size)` 未使用 tail 做 `madvise(MADV_DONTNEED)`，目标是降低 current RSS。
- `LLAMA_LOW_MEM_WARMUP=minimal`：跳过冗余 warmup decode，使 Stage 1 的 peak RSS 收益不被默认 warmup 峰值覆盖。

Stage 2 的目标不同：它作用于 **`[0,n_kv)` 内已经写入过的 live-history KV cell / block**，尝试把暂时不活跃的内容换出到进程外或可回收的后备存储，从而降低运行期 KV cache 的常驻物理内存。Stage 1 与 Stage 2 的区间和目标必须分开：

| 阶段 | 作用区间 | 数据语义 | 主目标 |
|---|---|---|---|
| Stage 1 lazy-clear | 构造期 tail `[clear_frontier, kv_size)` | 未触碰、未写入 capacity | 降 peak RSS |
| Stage 1 lazy-tail | 运行期 tail `[GGML_PAD(n_kv,256), kv_size)` | 未使用 capacity | 降 current RSS |
| Stage 2 swap/offload | live 区间 `[0,n_kv)` 内已写入 cell/block | 有语义的 K/V 历史 | 降 active/current RSS |
| Stage 3 prefetch | Stage 2 已换出的 cold block | 提前换入 | 隐藏 swap-in 延迟 |
| Stage 4 paged read | 非连续物理 block pool | 改 attention 读取模型 | 解除连续 view 约束 |

Stage 2 不应简单恢复旧原型。旧原型证明了「换出 -> 毒化 -> 换入 -> 输出恢复」的正确性闭环，但没有证明真实 RSS 优化。Stage 2 要把“正确性 swap”和“内存优化 swap”拆开设计。

---

## 2. 旧 swap 原型代码地图

旧实现位于 `archive/kv-runtime-swap-e2-approx-bak`。以下表格只复盘设计要点，不要求在当前分支恢复代码。

| 文件 | 函数或字段 | 作用 | 是否复用 | 备注 |
|---|---|---|---|---|
| `common/arg.cpp` | `--kv-swap` / `LLAMA_ARG_KV_SWAP` | CLI 开关，启用旧 runtime swap demo | 不直接复用 | 可借鉴“默认关闭、env 桥接”的调试开关风格；Stage 2 应重新命名和分层，不沿用旧语义。 |
| `common/arg.cpp` | `--kv-swap-window` / `LLAMA_ARG_KV_SWAP_WINDOW` | 设置固定窗口，窗口外 cell 候选换出 | 部分借鉴 | 固定窗口适合 approximate demo，但不适合 exact RSS 优化的唯一策略。 |
| `src/llama-kv-cells.h` | `swapped` | per-cell 换出标记 | 概念复用 | Stage 2 应升级为明确状态机，而不是单个 bool。 |
| `src/llama-kv-cells.h` | `swap_offset` | per-cell 后备存储偏移 | 概念复用 | 需要抽象到 backend allocation handle / offset，不应绑定进程内 vector。 |
| `src/llama-kv-cells.h` | `last_access` | per-cell 冷热/窗口策略依据 | 概念复用 | Stage 2 可作为 LRU 或 window/sink 策略输入，但需要定义更新时机。 |
| `src/llama-kv-cache.h` | `kv_swap_enabled` / `kv_swap_window` | 旧 swap demo 全局配置 | 不复用 | Stage 2 应拆成 mode、backend、policy 三类配置。 |
| `src/llama-kv-cache.h` | `kv_swap_storage` | `std::vector<uint8_t>` 进程内后备存储 | 不复用 | 这是旧原型无法降低 RSS 的核心原因之一。 |
| `src/llama-kv-cache.h` | `kv_swap_slot_cap` | per-cell backing slot 容量，避免 vector 重复增长 | 不直接复用 | 思路可借鉴为 slot reuse，但 Stage 2 后备存储应在进程外或 file-backed。 |
| `src/llama-kv-cache.h` | `kv_swap_poison` / poison 统计 | 换出后用 `0xCC` 覆盖原 KV 字节，反证 swap-in 必需 | 只作 debug 复用 | 适合作为 correctness guard，不适合作为性能或 RSS 路径。 |
| `src/llama-kv-cache.h` | `kv_swap_madvise` / madvise 统计 | 对已换出连续 range 尝试 `MADV_DONTNEED` | 不作为主路径 | 可保留为 probe 思路，但不是稳定优化架构。 |
| `src/llama-kv-cache.cpp` | `apply_ubatch` 中初始化 metadata | 新写入 cell 置 `swapped=false`、`swap_offset=0`、`last_access=pos` | 概念复用 | Stage 2 仍需在写入边界初始化状态，但字段应在 state0 单独落地。 |
| `src/llama-kv-cache.cpp` | `swap_out_window()` | 按固定窗口选择较老 cell 并调用 `swap_out_cell` | 部分复用思路 | 更适合 Demo A：window/sink approximate swap；exact 模式下不能盲目丢弃长程历史。 |
| `src/llama-kv-cache.cpp` | `swap_out_cell(i)` | 逐层读出 cell `i` 的 K/V 字节，写入 `kv_swap_storage`，记录 offset，置 `swapped=true` | 复用数据搬运思想，不复用实现 | 旧实现绑定 `!v_trans && n_stream==1`，且后备存储在进程内；Stage 2 应改为 backend 抽象。 |
| `src/llama-kv-cache.cpp` | `ensure_resident(n_kv)` | graph 读取前遍历 `[0,n_kv)`，将 swapped cell 全部换回 | 只作 correctness 参考 | exact attention 的连续 view 会把它放大成全量换入，是旧原型 active RSS 难降的核心。 |
| `src/llama-kv-cache.cpp` | `swap_in_cell(i)` | 从 `swap_offset` 读回 K/V 字节，原位写回 cell `i`，清 `swapped` | 复用原位恢复思想 | 适合 exact correctness；Stage 2 需要配合 backend、状态机和读区间策略。 |
| `src/llama-kv-cache.cpp` | `madvise_swapped_range(lo, hi)` | 对已换出连续 cell range 的页对齐内部区间做 `MADV_DONTNEED` | 不作为主架构 | 只能说明部分页可回收；后续读路径会 fault back，收益不稳定。 |
| `src/llama-kv-cache.cpp` | destructor swap stats | 输出 swap-out/in/poison/madvise 统计 | 部分复用 | Stage 2 仍需统计，但指标要区分 backing bytes、resident bytes、RSS、swap latency。 |
| `src/llama-kv-cache.cpp` | `llama_kv_cache_context::apply()` 中 `ensure_resident(n_kv)` | 在 n_kv 已知、graph 读 K/V 前同步换入 | 插桩点可复用 | 这是 exact swap 的最小安全边界，也是 Stage 3 prefetch 的未来锚点。 |
| 旧文档 | `docs/kv_runtime_swap_stage_ab.md` | A/B：CLI + metadata 设计记录 | 参考 | 可作为 state0 的历史输入，但不要照抄旧字段。 |
| 旧文档 | `docs/kv_runtime_swap_stage_c.md` | C：固定窗口 swap-out | 参考 | 证明换出路径能跑通，不证明 RSS 降。 |
| 旧文档 | `docs/kv_runtime_swap_stage_d.md` / `stage_d_summary.md` | D：swap-in + ensure_resident + poison correctness | 强参考 | 正确性闭环的主要依据。 |
| 旧文档 | `docs/kv_runtime_swap_stage_e1_lite.md` | E1-lite：madvise probe | 参考但不作为主线 | Stage 2 需要更稳定的 offload backend，而不是只靠 madvise probe。 |

旧原型最有价值的部分是三点：per-cell 状态可表达换出态；`ggml_backend_tensor_get/set` 能做后端无关的数据搬运；原位 swap-in 能恢复正确输出。最不能复用的是进程内 heap vector 后备存储和 `[0,n_kv)` 全量 `ensure_resident` 作为 RSS 优化路径。

---

## 3. 旧 swap 失败原因

### 3.1 backing store 仍在进程内

旧实现把换出的 K/V 字节写入 `std::vector<uint8_t> kv_swap_storage`。这只能证明“字节被复制到了另一个地方”，不能证明物理内存减少：

- `kv_swap_storage` 本身属于同一进程 RSS；
- vector 扩容会额外提交 heap 页；
- 即使 Stage D4 加了 per-cell slot reuse，也只是限制 backing store 高水位，不会让原 KV tensor 的页消失。

结果是进程内可能同时持有 **原 KV buffer + backing store 副本**。在 RSS 口径下，这会不降反升。

### 3.2 原 KV buffer 从未释放

llama.cpp 当前 KV cache 架构会在构造期为每层 K/V tensor 按 `kv_size` 连续分配 buffer。旧 swap-out 只做：

1. 从原 tensor 读出字节；
2. 写入 `kv_swap_storage`；
3. 打 `swapped` 标记；
4. 可选 poison 覆盖原字节。

其中没有任何一步释放原 tensor 物理页。poison 甚至会写回原页，使它更确定地处于 resident 状态。旧 swap 因此是 correctness demo，不是 offload demo。

### 3.3 `ensure_resident([0,n_kv))` 导致全量换入放大

旧实现为了不改 attention kernel，在 `llama_kv_cache_context::apply()` 中拿到 `n_kv` 后执行 `ensure_resident(n_kv)`，遍历 `[0,n_kv)` 并同步换回所有 swapped cell。

这对正确性是保守安全的，但对 active RSS 是反方向的：

- exact attention 的 `get_k/get_v` view 是连续 `[0,n_kv)`；
- mask 只影响 attention score，不改变 view 的物理读范围；
- 只要 swapped cell 落在 `[0,n_kv)`，旧逻辑就会在 graph 读取前换回；
- 固定小窗口刚换出的老 cell，下一步仍可能在 `[0,n_kv)` 里，于是立即被换回。

因此旧路径容易形成“换出 -> 下一 step 全量检查 -> 换回”的抖动，swap I/O 增加，但净 resident 降不下来。

### 3.4 exact attention 连续 view 限制 active RSS

不改 attention kernel 时，llama.cpp 的 attention 消费的是连续 K/V view，而不是 block table 中的非连续物理页。只要仍要求 exact attention 与 baseline 等价，当前 step 被 view 覆盖的 `[0,n_kv)` 必须能读到正确字节。

这意味着：

- exact swap 可以证明 correctness；
- exact swap 可以在 idle 阶段或 decode 间隙把 cold block offload；
- 但如果每次 attention 前都要恢复完整 `[0,n_kv)`，active compute 期间 RSS 很难显著下降。

要真正解除这个约束，需要 Stage 4 的 PagedAttention-style paged read，而不是 Stage 2。

### 3.5 madvise probe 不是稳定优化路径

旧 E1-lite 尝试对 swapped range 的页对齐内部区间做 `MADV_DONTNEED`。它的价值是验证“某些页在 OS 层面可被回收”，但不能作为 Stage 2 主架构：

- page 粒度与 cell 行边界不完全一致，容易释放不干净或误伤相邻数据；
- 后续 `ensure_resident([0,n_kv))` 或 attention 读路径可能立即 fault back；
- 对 malloc/ggml backend buffer 的效果依赖分配器和页对齐；
- 它仍没有解决 in-process backing store 的 RSS 叠加问题。

Stage 2 应把 madvise 视为辅助 probe，而不是最终 offloading backend。

---

## 4. Stage 2 目标重新定义

Stage 2 需要同时保留 correctness 证据和 RSS 目标，但不能混淆不同模式。

| 模式 | 语义 | 是否 exact | RSS 目标 | Stage 2 定位 |
|---|---|---:|---|---|
| exact swap | 换出后仍保证 attention 读到完整历史，必要时同步换回 | 是 | 主要验证 correctness、idle/current RSS；active RSS 受连续 view 限制 | 必做，但不能夸大 active RSS 收益 |
| window/sink approximate swap | 只保留最近窗口 + sink token 等子集，冷历史不再参与 full attention | 否 | 可以展示 active RSS 下降 | 建议做成明确近似 demo |
| debug correctness swap | poison 原 KV 后强制依赖 swap-in 恢复 | 是 | 不用于 RSS | 作为回归护栏 |
| RSS optimization swap | backing store 不计入进程 RSS，原 resident 页可回收或不再 active | 视策略而定 | 核心目标 | Stage 2 主线 |
| PagedAttention-style paged read | attention 直接按 block table 读取非连续物理 block | 是，可 exact | 从架构上降低 active RSS / 碎片 | Stage 4，Stage 2 不进入 |

结论：

- exact swap 能保证正确性，但不改 attention kernel 时，active compute 阶段往往必须恢复连续 `[0,n_kv)`，active RSS 难以下降。
- window/sink approximate swap 更适合比赛展示 active RSS 下降，但必须明确它是近似策略，不能声称与 baseline exact attention 等价。
- debug poison 是必要的正确性回归护栏，但不是优化模式。
- PagedAttention-style paged read 是 Stage 4；Stage 2 只为它预留 block/page 状态，不直接改 attention kernel。

---

## 5. 推荐 Stage 2 架构

### 5.1 状态机

建议以 per-cell 起步，接口上预留 per-block 聚合。Stage 2 最小必须态如下：

| 状态 | 含义 | Stage 2 是否必须 | 备注 |
|---|---|---:|---|
| `UNTOUCHED` | 从未写入、属于未使用 capacity | 必须 | 与 Stage 1 lazy-clear/tail 的 tail 区间兼容。 |
| `RESIDENT` | 已写入，K/V 字节在原 KV tensor 中可读 | 必须 | 新 token 写入后进入该状态。 |
| `SWAPPED` | 已写入，权威副本在 backing store，原 tensor 字节不可被信任或可被释放 | 必须 | exact 读前必须换回；approx 模式可选择不换回。 |
| `DIRTY` | resident 副本被写过，换出前必须写回 backing store | 建议 Stage 2 预留，最小 exact 可简化 | decode 中新写入 cell 可视作 dirty；若 backing store 是换出时生成，可不单独暴露。 |
| `RELEASED` | 原 tensor 对应页已被 OS 回收或不保证 resident | 建议 Stage 2 实现 | RSS 优化需要区分“已复制”与“原页已释放”。 |
| `PREFETCHING` | 异步 swap-in 正在进行 | Stage 3 预留 | Stage 2 不实现异步。 |
| `READY` | 预取完成，等待切换为 resident | Stage 3 预留 | 用于 latency hiding。 |

Stage 2 第一版最小闭环可以只落地 `UNTOUCHED / RESIDENT / SWAPPED / RELEASED`，并把 `DIRTY` 作为字段或注释预留。`PREFETCHING / READY` 只写入设计，不实现。

### 5.2 状态转换

推荐的最小转换：

| 转换 | 触发点 | 动作 |
|---|---|---|
| `UNTOUCHED -> RESIDENT` | `apply_ubatch` / `cpy_k` / `cpy_v` 写入新 token | 初始化 metadata，记录 `last_access`，标记 resident。 |
| `RESIDENT -> SWAPPED` | swap-out 选择 cold cell/block | 把 K/V 字节写入 backing store，记录 handle/offset。 |
| `SWAPPED -> RELEASED` | backing store 写入完成后释放原页 | 对原 tensor 区间执行可回收动作，或在 file-backed/staging 架构中移出 resident 集合。 |
| `SWAPPED/RELEASED -> RESIDENT` | exact attention 读前需要该 cell/block | 从 backing store 原位恢复，保证 view 命中正确字节。 |
| `RESIDENT -> RESIDENT` | 被 attention 访问或被新写覆盖 | 更新 `last_access`，保持原状态。 |

注意：`swapped` 不应被 `find_slot` 简单当成 free。若换出 cell 的语义仍需保留，slot 被覆盖会造成数据丢失。Stage2-state0 必须先定义“可覆盖”和“已换出但仍 live”的区别。

### 5.3 与 Stage 1 的共存

Stage 1 与 Stage 2 的共存规则：

- Stage 1 lazy-clear 只保证未使用 tail 不被构造期 full clear 提交；Stage 2 不应把 `UNTOUCHED` tail 当作可换出 live KV。
- Stage 1 lazy-tail 只对 `[GGML_PAD(n_kv,256), kv_size)` 未使用 tail 做 `madvise`；Stage 2 只处理 `[0,n_kv)` 内已写入或曾写入的 live cell/block。
- Stage 2 exact swap 在换入前如果要触碰某段 `[0,n_kv)`，必须确保 Stage 1 clear-frontier 已覆盖这段或该段已经由真实 K/V 写入。
- `RELEASED` 与 lazy-tail 的区别是：`RELEASED` 曾有语义副本，权威数据在 backing store；lazy-tail 是未使用 capacity，没有语义副本。

这一区分很重要：Stage 1 的收益来自“不触碰未使用 tail”，Stage 2 的收益来自“已写入但暂时不活跃的 live KV 不常驻”。

---

## 6. backing store 设计

Stage 2 的 backing store 不能再默认使用进程内 heap vector。下面是候选方案对比。

| 方案 | RSS 特性 | 优点 | 缺点 | Stage 2 建议 |
|---|---|---|---|---|
| heap `std::vector<uint8_t>` | 计入同进程 RSS | 实现最简单，适合 correctness | 不是真 offload；RSS 可能变成原 KV + vector | 只用于 poison/debug，不用于 RSS demo |
| anonymous `mmap` | 仍计入进程 RSS，`MADV_DONTNEED` 后可回收 | 页粒度可控，比 vector 更接近 OS 管理 | 仍在进程地址空间；写入后仍是 RSS | 可作中间实验，不是第一推荐 |
| `memfd` | file-like，可 mmap，可被 OS page cache 管理 | 无路径文件，Linux 友好，可模拟 file-backed | 仍可能计入进程 RSS；行为需解释清楚 | 可作为 Linux demo 备选 |
| `tmpfile` + `pread/pwrite` | backing 数据主要进入 page cache / block layer，不作为 heap 常驻 | 语义清晰：进程内只保留小 buffer；易证明不是 heap 副本 | I/O 慢，需管理临时文件和 offset | **第一版最推荐** |
| file-backed `mmap` | 可把 backing store 映射到文件，OS 可回收 clean pages | 接近 offload，便于 page cache 行为观察 | mmap 后触碰页仍可能计入 RSS；需要正确 `msync/madvise` | 推荐作为 tmpfile 的 mmap 变体 |
| 压缩 cold KV | 降低 backing 大小，可能仍在内存或文件中 | 可展示额外内存收益 | 引入压缩耗时和误差/格式复杂度 | 后续优化，不作为 Stage 2 第一版 |

为什么旧 heap vector 不足以降低 RSS：

- 它把换出的字节保存在同一进程 heap；
- 原 KV buffer 未释放时，vector 是额外副本；
- poison/madvise 不能改变 vector 本身计入 RSS 的事实；
- 即使原页部分回收，vector 仍可能抵消收益。

第一版更适合考虑 `tmpfile + pread/pwrite` 或 file-backed mmap。比赛 demo 需要容易解释“KV 字节离开了进程匿名 heap”，并能用 RSS 指标观察区别。`tmpfile + pread/pwrite` 最直观：换出时写临时文件，换入时读回，进程内只保留固定小 I/O buffer 和 offset 表。它会牺牲速度，但 Stage 2 第一目标是证明 RSS 路径，而不是吞吐最优。

推荐第一版 backing store：

> `tmpfile` / unnamed temporary file + explicit `pread/pwrite`，按 cell/block 分配固定 slot，进程内只保存 metadata 和小 I/O buffer；后续再评估 file-backed mmap、memfd 和压缩。

本轮不实现任何 backing store。

---

## 7. 最小可行 demo 路线

建议把 demo 拆成三个互相独立的目标，不再用一个 `--kv-swap` 模式承载所有含义。

### Demo B：exact swap + file-backed backing store

目标：

- 用 file-backed backing store 替代 heap vector；
- 保证换出、释放、换入后输出与 baseline 一致；
- 观察 idle/current RSS 是否下降；
- 保留 `ensure_resident`，但明确它会限制 active RSS。

价值：

- 是 Stage 2 最该先做的 demo；
- 它证明“真实 offload backing store + correctness”；
- 即使 active compute RSS 不显著下降，也能把失败原因归因到连续 `[0,n_kv)` view，而不是 backing store 仍在 heap。

### Demo A：window/sink approximate swap

目标：

- 保留 sink token 和最近 window；
- 对窗口外 KV 进行 approximate discard/offload；
- 不再在每次 attention 前恢复完整 `[0,n_kv)`；
- 展示 active RSS 下降。

价值：

- 更适合比赛展示“运行期 active RSS 降低”；
- 但必须在文档、CLI、日志中明确 approximate，不与 exact baseline 等价。

### Demo C：poison correctness swap

目标：

- 换出后 poison 原 KV；
- 读前换入；
- 比对输出或关键统计，确保 swap-in 是 load-bearing。

价值：

- 作为回归护栏；
- 不用于 RSS 或性能展示；
- 可以复用旧 Stage D3 的思想，但不应污染主路径。

推荐顺序：

| 顺序 | demo | 原因 |
|---:|---|---|
| 1 | **Demo B** | 先把旧原型最大硬伤“heap backing store + 原 buffer 未释放”替换掉，建立 exact offload 的可信基础。 |
| 2 | Demo A | 在明确 exact 连续 view 上限后，再做近似策略展示 active RSS 下降，避免过早混淆正确性与近似收益。 |
| 3 | Demo C | 作为每次改 swap-in/out 后的 correctness guard，可在 B/A 之间穿插，但不作为主 demo。 |

一句话：第一步建议做 Demo B，因为它最直接修复旧 swap “不是真 offload”的根因，同时保留 exact correctness，能为后续 approximate 和 prefetch 提供可信基线。

---

## 8. 风险清单

| 风险 | 说明 | 影响 | 缓解策略 |
|---|---|---|---|
| 正确性风险 | 换出后 metadata、pos/seq/shift、K/V 字节不一致 | 输出错误、难定位 | 先做 poison correctness；默认关闭；限制单 seq / `!v_trans` / 无 shift 边界。 |
| 性能风险 | file-backed `pread/pwrite` 同步 I/O 慢 | decode latency 变差 | Stage 2 先接受慢；Stage 3 再做 prefetch latency hiding。 |
| RSS 不降反升 | 原 KV 未释放或 backing store 仍计入 RSS | 核心指标失败 | 第一版不用 heap vector；统计原 resident、backing bytes、RSS 三者。 |
| backing store 选择风险 | mmap/memfd/tmpfile 的 RSS 口径不同 | 结果解释复杂 | 首版选择 `tmpfile + pread/pwrite`，语义最清晰。 |
| attention kernel 连续 view 约束 | exact 模式读前要恢复 `[0,n_kv)` | active RSS 难降 | 文档中明确 exact 上限；active RSS 展示用 approximate 或 Stage 4。 |
| approximate 长程信息损失 | window/sink 丢弃或不读长历史 | 输出质量下降 | CLI 和报告明确 approximate；做质量/输出对照，不声称 exact。 |
| Stage 1 交互风险 | lazy-clear/tail 与 Stage 2 都会触碰 KV 页 | 误释放 live data 或重复 madvise | 严格区分 live `[0,n_kv)` 与 unused tail `[PAD(n_kv),kv_size)`。 |
| `find_slot` 覆盖风险 | swapped live cell 被当作 free 覆盖 | 数据永久丢失 | state0 明确定义 `SWAPPED` 不等于 free；覆盖前必须 discard 或 swap-in。 |
| V 转置风险 | `v_trans=true` 时 V 按 embedding 维碎片化 | I/O 放大、实现复杂 | Stage 2 demo 先限定 `-fa on` / `v_trans=false`。 |
| 多 stream / seq_cp / SWA / K-shift | 多引用和移动语义复杂 | 状态机不变量变多 | 第一版明确不支持；后续逐项解锁。 |

---

## 9. 后续 Codex 小步任务规划

以下只规划，不实现。

| 任务名 | 目标 | 产出 | 不做什么 |
|---|---|---|---|
| `Stage2-state0` | 在 clean Stage 1 基础上定义 per-cell/per-block 状态字段和访问器 | `UNTOUCHED/RESIDENT/SWAPPED/RELEASED` 最小状态，`DIRTY/PREFETCHING/READY` 预留说明 | 不写 swap I/O，不改 attention |
| `Stage2-backend0` | 设计 backing store 抽象 | backend interface、slot allocation、offset/size metadata | 不接入 decode 热路径 |
| `Stage2-debug-correctness` | 恢复 poison correctness guard | 可关闭的 poison + exact swap-in 验证路径 | 不用于 RSS 评估 |
| `Stage2-exact-offload` | file-backed exact swap demo | `tmpfile/pread/pwrite` backing store，exact 输出一致，idle/current RSS 观察 | 不做 prefetch，不声称 active RSS 必降 |
| `Stage2-window-approx` | window/sink approximate swap demo | 可配置 sink/window，展示 active RSS 下降 | 不声称 exact correctness |
| `Stage3-prefetch` | 在 Stage 2 backing store 上加入预取 | `PREFETCHING/READY` 状态，预取命中率和等待时间指标 | 不改 paged attention |
| `Stage4-paged-read` | 引入 block table / physical block pool / 非连续 paged read | PagedAttention-style 设计与最小 kernel 改造 | 不在 Stage 2 直接进入 |

建议最近一步进入 `Stage2-state0`，因为所有后续 backend、poison、exact offload、approx policy 都依赖清晰的状态语义。`state0` 不应该恢复旧 swap；它只定义状态字段、初始化/重置规则、与 Stage 1 lazy 区间的边界。

---

## 10. 结论

1. 旧 swap 原型的主要价值是正确性闭环：它证明了 per-cell 换出、poison、读前原位换入可以恢复正确输出。
2. 旧 swap 原型不能作为 Stage 2 直接恢复：heap vector backing store 计入进程 RSS，原 KV buffer 未释放，`ensure_resident([0,n_kv))` 又会把 exact attention 放大成全量换入。
3. Stage 2 必须把 exact correctness、approximate active RSS、debug poison、RSS optimization 四种模式分开，避免用一个开关混淆语义。
4. 第一版推荐 `Demo B: exact swap + file-backed backing store`，先修复“不是真 offload”的根因，再评估 exact 模式在连续 view 约束下的 RSS 上限。
5. Stage 4 的 PagedAttention-style paged read 是解除连续 view 的根本路径，但当前阶段只预留状态，不直接进入。
