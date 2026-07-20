# 按需加载优化方案设计（window 实测修订版）

> **本文状态**：原始版为愿景设计（lazy-v2 拷贝缓冲路线）。本版**根据实际落地的 `llama-window` 实现 + 本次完整测试结果**逐节修订，把已被实测推翻的指标和与实现不符的架构全部更正，并标注诚实边界。配套另有简版 [`WINDOW_DESIGN.md`](WINDOW_DESIGN.md)。
>
> **一句话结论**：window 在不受限内存下取得 **稳态 RSS −44%、速度与基线统计无差异（误差棒内相等）、输出与基线 `--no-repack` 逐位一致**。它是"和 page cache 合作、零拷贝、轻量省内存"的方案，适用于"内存差一点、想省 RSS 又不掉速"的场景。

---

## 方案归属与对比基线

本文所有"基线 / native"均指 **llama.cpp 原有的默认加载方式**，下表厘清哪些是 upstream 已有、哪些是本工作新增（经联网核实，2026-06）：

| 能力 | 归属 | 说明 |
|---|---|---|
| **mmap 默认加载 = 被动按需分页（lazy demand paging）** | **llama.cpp 原有** | 不传 `--no-mmap` 时，OS 按访问惰性 fault 页面；这是 upstream 文档化的默认行为。"lazy loading"这个词本身指的是它。 |
| `--no-mmap`（整模型一次性读入 RAM） | llama.cpp 原有 | 模型大于 RAM 时无法加载 |
| `--mlock`、`--no-repack`(extra_bufts) | llama.cpp 原有 | repack = 权重重排 kernel |
| **主动图感知预取 + 滑动窗口 + 可选回收（`llama-window`）** | **本工作新增** | upstream **没有**；社区仅有 PoC 讨论（[#23324](https://github.com/ggml-org/llama.cpp/discussions/23324)），未合入 |
| `node_callback`（ggml-cpu 每 node 回调）+ `MADV_POPULATE_READ` 预取 | 本工作新增 | 驱动 window 的时序与执行手段 |
| `LLAMA_LAZY_V2` / `LLAMA_LAZY_*` 开关 | 本工作新增 | 激活 window |
| lazy-v2（malloc+memcpy 拷贝路线）、v1 lazy | 本工作早期尝试 | 经确认为死代码，已移除 |

> **因此本文的对比是「llama.cpp 原有 mmap 默认」 vs 「本工作的 window」**。upstream 的"lazy loading"是**被动**的（OS 按需 fault、无预取、faults 砸在计算线程）；window 在其之上加了**主动预取 + 滑窗管理**。

---

## 设计目标（修订）

| 目标 | 原愿景 | 实测修订 |
|---|---|---|
| 内存 | 降峰值 30–50%（大模型 70–80%） | **不受限内存：稳态 RSS −44%**；但 dense 模型 RSS 有 **≈模型大小的硬地板**（见下），70–80% 仅在真实内存压力下（cgroup）成立，且那时速度塌陷 |
| 性能 | TPS 下降 ≤ 10% | **与 native 统计无差异**（噪声 ±1 t/s 内相等），优于原目标 |
| 正确性 | 与标准模式完全一致 | **与 `native --no-repack` 逐位一致**（PPL 7.5380）；与 native 默认差 0.5%，**纯粹来自 window 关闭了 repack kernel**，在 PPL 误差棒内，非 bug |
| 可配置 | 不同内存/性能策略 | 保留；但 dense 模型下 `memory_limit` 旋钮基本无效（见风险节） |

---

## 核心问题分析

### 推理访问模式特征（仍然成立——这是整套设计的基石）

```
Token 0: Layer 0 → 1 → 2 → ... → N-1
Token 1: Layer 0 → 1 → 2 → ... → N-1
Token t: Layer 0 → 1 → 2 → ... → N-1   （紧循环）
```
- 100% 顺序访问，下一层可预测（current+1）
- 每个 token 重复相同层序列，prefill 与 generation 模式一致

### 由此推出的硬约束（原文未点明，是本次最重要的认知）

> **dense 模型每个 token 都要跑全部 N 层、循环往复。被逐出的层下个 token 必然又要用。**

因此：
- **RSS 降到工作集以下 ⟺ 每 token 从存储重读被逐出的部分**，吞吐 ≤ 磁盘带宽 / 每token重读字节。
- **不受限内存下，被 `DONTNEED` 的层会被循环访问立刻重新 fault 回来** → 稳态 RSS 有个 **≈模型大小的地板**，"换出"换不掉。

### mmap 的两个失败模式（window 要解决的真正问题）

1. **开 prefetch（`MAP_POPULATE`/全文件 `WILLNEED`）**：加载即把整模型 fault 进来 → 峰值 RSS 高（native 实测稳态 ~7.68GB）。
2. **不开 prefetch（懒加载）**：页错误全砸在**计算线程** → 计算到某层同步 major fault、卡顿。

> **更正原文"瓶颈"**：原文批评的"粗粒度 mmap/munmap、malloc+memcpy 拷贝、简单 LRU"对应的是已被删除的 lazy-v2 拷贝路线。实际落地的 window 是**零拷贝**方案，不做拷贝、不 munmap，下文据此重写。

---

## 优化方案设计（实际架构：零拷贝 mmap + madvise + node_callback）

### 架构概览（修订——无拷贝缓冲、无后台 memcpy）

```
┌───────────────────────────────────────────────────────────┐
│  推理计算线程（ggml-cpu）   Layer 0 → 1 → … → N-1           │
│  权重张量 ->data 始终指向文件 mmap 地址（零拷贝）          │
└───────────────┬───────────────────────────────────────────┘
                │ 每个 node 完成后回调（node_callback，时序信号）
                ↓
┌───────────────────────────────────────────────────────────┐
│  window controller（llama-window.cpp，调度大脑）           │
│  - 据层号决定预取 [L+1, L+ahead] / 保护 / 回收远层          │
│  - 维护每层状态机、预算、自适应                            │
└───────────────┬───────────────────────────────────────────┘
                │ 任务队列
                ↓
┌───────────────────────────────────────────────────────────┐
│  预取/回收 worker 线程（对 mmap 地址做 madvise，非拷贝）    │
│  - prefetch → MADV_POPULATE_READ（真正 fault 进 RAM）       │
│  - reclaim  → MADV_DONTNEED（丢物理页，可选）               │
└───────────────────────────────────────────────────────────┘
```

**核心特征：权重全程零拷贝（`->data` = mmap 指针），residency 交给内核 page cache，window 只"提前喂热 + 可选丢冷"。**

### 1. 滑动窗口（Sliding Window）

- **加载期索引**（`llama_window_create`）：每个权重从名字 `blk.N.` 解析层号，地址区间**页对齐**后按层合并成 `ranges`。
- **每层状态机**：`cold → queued → advised → resident`。
- **推进**（`llama_window_advance_locked`）：换层到 L 时预取 `L+1..L+prefetch_ahead`、对**保护窗口外**层入队回收。
- **保护窗口**（`llama_window_is_protected`）：`[L-keep_behind, L+prefetch_ahead]` 永不回收。

### 2. 异步预取（节点驱动，而非主线程轮询）

时序由 **ggml-cpu 的 `node_callback`** 提供：每个计算 node 完成、跨过 barrier 后于 `ith==0` 触发 → window 精确知道"计算在第几层"。这比原愿景的"主线程 for 循环里手动 request/wait"更贴合 ggml 图执行模型，且不改 kernel。

> **更正原文线程模型**：原文的 `loader_worker_thread` 做 `mmap+memcpy+munmap` 是拷贝路线；实际 worker **只对已有 mmap 地址做 madvise**，无 memcpy。

### 3. 驱逐策略（实现现状 + 实测局限）

实现确有成本模型（`distance × 1MB + size/2` 打分，优先丢最远最大的层）+ 保护窗口 + 永不驱逐全局张量（embd/output）。
**但实测局限**：dense 模型循环访问下，**回收压不动 ~模型地板**（被丢的层下个 token 又 fault 回来）。回收相对"只预取"在不受限内存仅多省 ~5% RSS，代价是页表 churn。**智能驱逐的收益在 dense + 不受限场景有限**。

### 4. 自适应窗口（auto_tune）

`llama_window_auto_tune_locked` 用 EMA 跟踪"每层计算间隔 vs 预取延迟"，预取跟不上就增大 `prefetch_ahead`、内存压力大就减小。**实测**：不受限内存下其效果被 CPU 噪声淹没、不显著；真正发挥在受限/慢盘场景。

### 5. 关于"双缓冲"

原愿景的 primary/prefetch buffer swap 属于**拷贝缓冲**模型。window 是零拷贝，不做显式双缓冲——"重叠"由 page cache + 异步 `POPULATE_READ` 天然实现。显式双缓冲/ring buffer 属于"不 mmap、自管缓冲"的另一条路线，不在本方案范围内。

---

## 内存布局设计（修订）

```
┌───────────────────────────────┐
│ 永久驻留：embd / output / norm │  ~0.5–2GB，始终 mmap、不回收
├───────────────────────────────┤
│ 工作集：全部 N 层的权重        │  dense 下稳态 ≈ 模型大小（循环再 fault）
│  （window 管理其页的进/出）    │  不受限内存降不过此地板
├───────────────────────────────┤
│ 磁盘（mmap 文件背书）          │  受真实内存压力时，远层退回此处，下次重读
└───────────────────────────────┘
```

> **更正原文"节省 78%"**：原表基于"拷贝路线 + 整层换出常驻在外"的假设。实测 dense 模型：
> - **不受限内存**：window 稳态 ~4.3GB vs native ~7.68GB（**−44%**）——省的是 native 的**全文件 prefetch 冗余**，不是把工作集做小。
> - **真要 RSS < 模型**：只能靠真实内存压力（cgroup）逼内核回收 + 从盘重读，此时速度塌到 ~0.4 t/s（见性能节）。这是 mmap 路线的机制天花板：要把 RSS 稳定压到模型以下，需要"不 mmap、自管缓冲"的另一类方案，不在本方案范围内。

---

## 性能优化技术（修订为实际采用的）

1. **`MADV_POPULATE_READ`（关键）**：worker 线程一个 syscall 同步把整段 fault 进 RAM（带 readahead、不抛 SIGBUS），把"计算线程的同步 major fault"转成"后台预取、与计算重叠"。老内核回退 `WILLNEED + 逐页 touch`。
2. **页对齐区间 + 跨层共享页保护**：`create` 把每层 range 页对齐合并；检测不同层落同一页的区间标 `safe=false`、不回收，避免误丢另一层数据。
3. **保护窗口**：当前层 ± 窗口内不回收，杜绝"丢正在用/马上用的层"。
4. **零拷贝**：永久驻留 + 滑窗层都直接用 mmap 地址，无 memcpy（原文的 SIMD memcpy 优化在零拷贝路线下不需要）。

> **重要取舍（实测发现）**：window **必须关闭 `extra_bufts`（weight repack）**——因为 repack 会把权重**拷贝重排进独立 CPU 缓冲**，与 window 依赖的"张量指向文件 mmap、对文件页 madvise"零拷贝前提**冲突**。代价：用非重排 kernel，数值与 `native --no-repack` 严格一致、与 native 默认有 PPL-误差棒内的 0.5% 差异。

---

## 实现方案（实际代码）

### 关键文件与函数
- `src/llama-window.{h,cpp}`：`llama_window_create`（索引/区间合并/safe 检测/起 worker）、`advance_locked`（推进/预取/回收）、`is_protected`、`worker`（`llama_window_populate` = POPULATE_READ；DONTNEED）、`auto_tune_locked`、`graph_begin/node_done/graph_end`。
- `ggml/include/ggml-cpu.h` + `ggml/src/ggml-cpu/ggml-cpu.{c,cpp}`：`ggml_cplan.node_callback` + threadpool 每 node barrier 后于 ith0 触发；`ggml_backend_cpu_set_node_callback` 经 backend reg 暴露。
- `src/llama-model.cpp`：`LLAMA_LAZY_V2`/`LLAMA_LAZY_LOADING` 置位时（关 prefetch/mlock/extra_bufts）从 `weights_map` 建 `region_input` 并 `llama_window_create`。
- `src/llama-context.cpp`：decode 周围装 `llama_window_cpu_node_done` 回调 + `graph_begin(prefill)`，算完 `graph_end` 卸回调。

### 集成方式（修订——零拷贝，不重定向 data）
权重张量在加载时 `->data` 即指向 mmap 地址，**计算时直接读 mmap，window 不改 data**（与原文"get_tensor_data 返回 buffer 偏移"的拷贝模型不同——那是已废弃的拷贝路线做法）。window 仅通过 madvise 影响这些 mmap 页的物理驻留。

### 核心实现代码（实际 `llama-window.cpp` 摘录）

**(a) 加载期索引 + 跨层共享页安全检测**（`llama_window_create`）：把不同层落在同一页的区间标 `safe=false`、不纳入回收，杜绝误丢另一层数据。
```cpp
// 已按 addr 排序的 candidates：相邻重叠且属于不同层 → 都标记为不安全
for (size_t i = 0; i < candidates.size(); ++i) {
    const uintptr_t end_i = (uintptr_t) candidates[i].range.addr + candidates[i].range.size;
    for (size_t j = i + 1; j < candidates.size(); ++j) {
        if ((uintptr_t) candidates[j].range.addr >= end_i) break;
        if (candidates[i].layer != candidates[j].layer) {
            candidates[i].safe = false;
            candidates[j].safe = false;   // 共享页 → 不回收
        }
    }
}
for (const auto & c : candidates)
    if (c.safe) ctx->layers[c.layer].ranges.push_back(c.range);
```

**(b) 保护窗口**（`llama_window_is_protected`）：`[current-keep_behind, current+prefetch_ahead]` 内的层永不回收（环形）。
```cpp
static bool llama_window_is_protected(int layer, int current, int n_layers,
                                      int keep_behind, int prefetch_ahead) {
    if (layer < 0 || current < 0 || n_layers <= 0) return true;
    for (int delta = -keep_behind; delta <= prefetch_ahead; ++delta) {
        int p = (current + delta) % n_layers; if (p < 0) p += n_layers;
        if (layer == p) return true;
    }
    return false;
}
```

**(c) 推进：预取 + 回收**（`llama_window_advance_locked`，换层到 L 时触发）。
```cpp
// 预取 L+1 .. L+prefetch_ahead（受 prefetch_remaining 预算限）
for (int delta = 1; delta <= ctx.params.prefetch_ahead && ctx.prefetch_remaining > 0; ++delta)
    llama_window_request_prefetch_locked(ctx, (layer + delta) % ctx.n_layers, delta);

// 回收：仅当开启 dontneed 且（无 limit 或超 limit）；只丢保护窗口外的层
const bool memory_pressure = ctx.params.memory_limit > 0 &&
                             ctx.stats.current_rss > ctx.params.memory_limit;
if (ctx.params.use_dontneed && (ctx.params.memory_limit == 0 || memory_pressure)) {
    std::vector<int> candidates;
    for (int c = 0; c < ctx.n_layers; ++c)
        if (!llama_window_is_protected(c, layer, ctx.n_layers,
                                       ctx.params.keep_behind, ctx.params.prefetch_ahead))
            candidates.push_back(c);
    // 按"前向距离 + 层大小"打分，优先丢最远最大的
    std::sort(candidates.begin(), candidates.end(), /* score = distance*1MiB + bytes/2 */ ...);
    for (int c : candidates) llama_window_request_reclaim_locked(ctx, c, /*prio*/100);
}
```

**(d) worker：真正把页 fault 进 RAM（POPULATE_READ），回收用 DONTNEED**。
```cpp
static bool llama_window_populate(uint8_t * addr, size_t size) {
#if defined(__linux__) && defined(MADV_POPULATE_READ)
    if (madvise(addr, size, MADV_POPULATE_READ) == 0) return true;  // 5.14+：同步落实
#endif
#if defined(MADV_WILLNEED)
    madvise(addr, size, MADV_WILLNEED);                              // 老内核回退
#endif
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);              // + 逐页 touch 强制驻留
    volatile uint8_t sink = 0;
    for (size_t off = 0; off < size; off += page) sink ^= addr[off];
    return true;
}
// worker 主体：prefetch → populate(range)；reclaim → madvise(MADV_DONTNEED)
// reclaim 前再次检查 is_protected（异步期间窗口可能已移动 → 跳过 stale 任务）
```

**(e) 时序钩子：ggml-cpu 每 node 回调**（`ggml/src/ggml-cpu/ggml-cpu.c`，所有线程每 node）。
```c
// 计算完一个 node、跨过 barrier 后，由 ith==0 触发回调
if (node_n + 1 < cgraph->n_nodes || tp->sequence_node_callback != NULL)
    ggml_barrier(state->threadpool);
if (state->ith == 0 && tp->sequence_node_callback != NULL)
    tp->sequence_node_callback(node, tp->sequence_callback_data);
```
```cpp
// llama-window.cpp：从 node 名尾部 "-N" 解析层号 → 换层即推进窗口
void llama_window_node_done(llama_window_context & ctx, const ggml_tensor * node) {
    const int layer = llama_window_parse_layer(ggml_get_name(node));
    if (layer < 0 || layer >= ctx.n_layers) return;
    if (ctx.observed_layer.exchange(layer, std::memory_order_relaxed) == layer) return; // 同层早退
    std::lock_guard<std::mutex> lock(ctx.mutex);
    if (ctx.graph_active) llama_window_advance_locked(ctx, layer, true);
}
```

**(f) 接线：加载期建 region、decode 期装/卸回调**。
```cpp
// llama-model.cpp：weights_map → region_input（mmap 地址 + 偏移），零拷贝
for (const auto & it : ml.weights_map) {
    const auto & w = it.second;
    llama_window_region_input in;
    in.name = it.first;
    in.addr = (uint8_t *) pimpl->mappings[w.idx]->addr() + w.offs;  // 指向 mmap
    in.size = ggml_nbytes(w.tensor);
    in.file_idx = w.idx; in.file_offset = w.offs;
    inputs.emplace_back(std::move(in));
}
pimpl->window = llama_window_create(inputs, hparams.n_layer, window_params);
```
```cpp
// llama-context.cpp：decode 前装回调 + graph_begin，算完卸回调
static void llama_window_cpu_node_done(const ggml_tensor * node, void * ud) {
    if (auto * w = static_cast<llama_window_context *>(ud)) llama_window_node_done(*w, node);
}
if (backend_cpu && llama_window_enabled(window)) {
    set_node_callback_fn(backend_cpu, llama_window_cpu_node_done, window);
    llama_window_graph_begin(*window, /*prefill=*/batched);
}
ggml_backend_sched_graph_compute_async(sched.get(), gf);
if (set_node_callback_fn) {
    llama_window_graph_end(*window);
    set_node_callback_fn(backend_cpu, nullptr, nullptr);   // 卸回调
}
```

---

## 配置参数（实际环境变量）

```bash
export LLAMA_LAZY_V2=1                 # 启用 window（或 LLAMA_LAZY_LOADING=1）
export LLAMA_LAZY_WINDOW_SIZE=12       # 窗口层数
export LLAMA_LAZY_PREFETCH_AHEAD=2     # 提前预取层数
export LLAMA_LAZY_WORKER_THREADS=1     # 预取/回收 worker 数
export LLAMA_LAZY_DONTNEED=1           # 启用回收（dense 下基本无效，见风险）
export LLAMA_LAZY_MEMORY_LIMIT=4       # 回收触发预算 GB（dense 下基本无效）
export LLAMA_LAZY_PREFILL_AGGRESSIVE=1 # prefill 激进预取整窗
export LLAMA_LAZY_AUTO_TUNE=1          # 自适应（默认开）
export LLAMA_LAZY_DEBUG=1              # 调试日志
```
**推荐用法（dense）**：`LLAMA_LAZY_V2=1`（**只预取、不设 memory_limit**）即可拿 −44% RSS、≈native 速度的最佳折中。

---

## 测试验证（本次实际做了的）

### 0. 测试环境与对象

| 项 | 取值 |
|---|---|
| **模型** | `Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf`（4.58 GB，dense，32 层，GQA，Q4_K_M；`-aligned` = 经 gguf-align 工具张量对齐，利于后续 O_DIRECT，对 window 无影响） |
| **基线** | llama.cpp 原有：`llama-cli` / `llama-bench` / `llama-perplexity`，默认 mmap；`--no-repack` 用于隔离 repack kernel |
| **window 激活** | `LLAMA_LAZY_V2=1`（+ `LLAMA_LAZY_DONTNEED=1`/`LLAMA_LAZY_MEMORY_LIMIT` 测回收） |
| **硬件** | x86-64，8 核（计算用 `-t 6`），RAM 23 GB，单 SATA/NVMe（冷读 dd 实测 279 MB/s） |
| **OS/内核** | Linux（`MADV_POPULATE_READ` 可用，≥5.14；cgroup v2，memory controller + `memory.swap.max=0`） |
| **数据集** | [WikiText-2 raw](https://huggingface.co/datasets/ggml-org/ci)（`wikitext-2-raw/wiki.test.raw`，1.29 MB，标准 PPL 语料；用 `scripts/get-wikitext-2.sh` 下载） |
| **方法** | 速度 `llama-bench -r 3`（取均值±std，规避 CPU 噪声）；RSS 稳态采样 `/proc/<pid>/status:VmRSS`（峰值 RSS 对 window 不具代表性）；受限场景用 cgroup v2 `memory.max` + 冷启（`drop_caches`）；正确性用 `llama-perplexity --chunks 4` + `llama-cli --temp 0` 贪心对比 |

### 1. 单元测试 `tests/test-window.cpp`（不依赖真模型）
文件背书只读 mmap + `mincore()` 直接观测页驻留，验证：
- **线程稳定性**：8×(create + 4 趟 sweep + destroy)，worker 正常 join、不崩不 hang
- **预取 + 保护不变量**：推进到 K 后保护窗口 `[K-keep_behind, K+prefetch_ahead]` 驻留
- **边界**：超大参数、垃圾 node 名、0 层禁用 → 不崩
- **ASAN**：零内存错误/泄漏/UAF
- 结果：**ALL PASS（普通 + ASAN）**

```bash
g++ -O2 -std=c++17 -pthread -I ggml/include tests/test-window.cpp src/llama-window.cpp \
    build/bin/libggml-base.so* -o /tmp/test-window && LD_LIBRARY_PATH=build/bin /tmp/test-window
```

### 2. 端到端数值等价（perplexity，WikiText-2，--chunks 4）
| 配置 | PPL | chunk[1] |
|---|---|---|
| native（默认开 repack） | 7.5749 | 5.1096 |
| **native `--no-repack`** | **7.5380** | **5.1391** |
| **window** | **7.5380** | **5.1391** |
| **window + reclaim** | **7.5380** | **5.1391** |

**结论**：`native --no-repack` == window == window+reclaim **逐位一致** → ① window 数据处理字节精确；② 预取/回收/re-fault 零损坏；③ 与 native 默认的 0.5% 差异 100% 来自 repack kernel，误差棒（±0.66）内，非 bug。

### 3. 测量方法学教训
这台共享机 **CPU 吞吐波动 ±~1 t/s（17%）**。**单样本（`-r 1`）不可信**——会得出"window 慢 22%"或"快 9%"这类**纯噪声**结论。速度结论必须 `-r 3`+ 看误差棒。

---

## 实测性能指标（替换原愿景表）

### 不受限内存（误差棒校正，Llama-3-8B-Q4_K_M，8 核）
| 方案 | decode（带误差棒） | 稳态 RSS |
|---|---|---|
| native | 7.56 ± 0.14 t/s | 7.68 GB |
| **window** | **7.55 ± 0.08 t/s** | **~4.3 GB（−44%）** |

→ **速度与 native 统计无差异，RSS −44%**。`memory_limit` 4GB↔1GB 对 RSS（±80MB）和速度均无可测影响。

### 受限内存（cgroup memory.max=3GB < 4.58GB 模型）
| 方案 | 能跑 | RSS | decode |
|---|---|---|---|
| llama.cpp 原有 (mmap 默认) | ❌ **OOM killed** | — | — |
| **window** | ✅ | 3.06 GB | **0.40 t/s** |

→ 受限下 window 能跑（原有 mmap 因全文件 prefetch 撞上限被 OOM kill），但纯磁盘 IO 瓶颈（0.40 t/s）。这是 mmap 路线在"模型 > 内存"时的固有代价。

---

## 风险与缓解（修订）

| 风险 | 实测结论 / 缓解 |
|---|---|
| **memory_limit/reclaim 对 dense 名存实亡** | 不受限无效（旋钮无差异）、受限下真正封顶的是 cgroup。**倾向**：dense 模型 window 可退化为纯"graph-aware 预取"（无 reclaim/memory_limit）。待验：cgroup 下 reclaim 开/关隔离实验 |
| **与 repack 互斥** | window 关 extra_bufts → 用非重排 kernel（与 native 默认有误差棒内 0.5% PPL 差异）。要 native 逐位一致须 `--no-repack`。二者不可兼得 |
| **RSS 地板 ≈ 模型（dense）** | 不受限内存降不过模型大小（循环再 fault）。这是 mmap 路线的机制天花板，本方案不试图突破 |
| I/O 瓶颈（慢盘/HDD） | 受限下 IO-bound（0.40 t/s）；POPULATE_READ + 多 worker + prefill_aggressive 缓解有限 |
| 多线程竞态 | 单测 + ASAN 已验证 window 自身代码零内存错误；reclaim re-fault 数值字节精确 |
| 正确性 | 强同步：node barrier 后 ith0 操作；只读 mmap → DONTNEED 安全、re-fault 还原；跨层共享页 + 活跃窗口保护 |

---

## 适用区间（相对 llama.cpp 原有）

| 内存 vs 模型 | 推荐 | 说明 |
|---|---|---|
| 充裕 | llama.cpp 原有 (mmap 默认) | 全驻留，满速 |
| **差一点（想省 RSS 又不掉速）** | **window** | mmap 零拷贝 + 图感知预取（+可选回收）：**−44% RSS / ≈原有速度** |
| 装不下（原有 mmap 会 OOM 或 IO 瓶颈） | window 能跑但 IO-bound（0.40 t/s） | mmap 路线在此 regime 的固有上限 |

---

## 总结（修订）

### 实际达成
- **内存**：不受限 RSS −44%（来自不做全文件 prefetch）；dense 模型 RSS 有 ≈模型地板（mmap 路线的机制天花板）。
- **性能**：与 native 统计无差异（优于"≤10%"原目标）。
- **正确性**：与 `native --no-repack` 逐位一致；与 native 默认差 0.5% 纯 repack kernel、误差棒内。
- **可配置**：保留；但 `memory_limit` 对 dense 基本无效，应主推"只预取"。

### 核心修正（相对原愿景）
1. 架构从"malloc+memcpy 拷贝缓冲"改为**零拷贝 mmap + madvise**。
2. 驱动从"主线程轮询"改为 **ggml node_callback**。
3. "节省 70–80%"更正为**不受限 −44% + 模型地板**的诚实表述。
4. "TPS 降 ≤10%"更正为**与 native 噪声内相等**。
5. "输出完全一致"更正为**与 `native --no-repack` 逐位一致 + repack 取舍**。
6. 新增**测量噪声教训**（必须误差棒）与**方案归属**（厘清 upstream mmap vs 本工作的 window）。

### 适用场景
- ✅ 内存差一点、想省 RSS 又不掉速（消费级、SSD、单用户）
- ✅ 与原有数值等价要求可接受 `--no-repack` 口径
- ⚠️ 需要 RSS 远低于模型（dense）→ 超出本 mmap 方案能力上限
- ❌ 慢盘 + 受限内存（IO-bound，体验差）

---

**文档版本** v2.0（window 实测修订）· 数据来源：本会话 Llama-3-8B-Q4_K_M / 8 核 / WikiText-2 / cgroup 实验，速度均经 `-r 3` 误差棒校正。
