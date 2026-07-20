# 实现说明：对原生 llama.cpp 的更改、问题与算法（含代码）

本文档记录本框架（window / flex / moe-buffer / SRE）**对原生 llama.cpp 做的全部更改**、**会话中遇到的每个问题与解决方案**、以及**每个问题对应的算法及其实现代码**。配套：[`SRE_UNIFIED_DESIGN.md`](SRE_UNIFIED_DESIGN.md)（设计）、[`TEST_RESULTS.md`](TEST_RESULTS.md)（全部实测）。

---

## 1. 对原生 llama.cpp / ggml 的更改总览

### 1.1 新增模块（不动原生逻辑，纯新增）

| 文件 | 行数 | 职责 |
|---|---|---|
| `src/llama-window.{h,cpp}` | ~1437 | mmap 层滑窗 + MoE 专家 CLG 预测器 |
| `src/llama-flex.{h,cpp}` | ~612 | dense 匿名 ring 流式（FlexInfer 风格）|
| `src/llama-moe-buffer.{h,cpp}` | ~462 | MoE 专家匿名缓冲流式 + 真驱逐 + 多 worker |

### 1.2 修改的原生文件（最小侵入）

| 文件 | 改动 |
|---|---|
| `ggml/include/ggml-cpu.h`、`ggml/src/ggml-cpu/ggml-cpu.{c,cpp}` | **新增两个 CPU 后端钩子**（已并入历史 commit）：`ggml_backend_cpu_set_node_callback`（每节点完成后回调）+ `ggml_cpu_set_weight_stream_callback`（kernel 计算前回调）。这是整套流式的**唯一 ggml 层依赖**。 |
| `src/llama-model.{h,cpp}` | pimpl 加 `window/flex/moe_buffer` 成员 + getter；`load_tensors` 里创建/注册三模块；自适应 budget/ring；`llama_detect_available_memory()` helper |
| `src/llama-context.cpp` | 解码图前安装钩子：window→`node_callback`、flex/moe→`weight_stream_callback`；图后卸载 |
| `src/CMakeLists.txt` | 加 3 个新源文件 |

### 1.3 两个 ggml 钩子的语义（整套机制的地基）

```c
// 每个图节点计算完成后调用一次（仅 ith0），用于"看到刚算完的 hidden state"→ 预测/调度
typedef void (*ggml_graph_compute_sequence_node_callback)(const ggml_tensor * node, void * user_data);

// 每个算子 kernel 计算【前】调用（所有线程进入，ith0 做流式，然后 barrier），
// 返回 true 表示该 op 的权重被本回调接管 → CPU 后端在 kernel 前插入 barrier
typedef bool (*ggml_cpu_weight_stream_callback)(ggml_tensor * op, int ith, void * user_data);
```

**为什么需要两个**：`node_callback` 在节点**之后**、其他线程不等它 → 只能用于"预测下一步"（异步预取），**不能**保证"kernel 读权重前数据已就位"。`weight_stream_callback` 在 kernel **之前** + barrier → 是**唯一能保证正确性的同步点**。两者分工 = 异步预取拿 lead time + 同步兜底保正确。

---

## 2. 问题 → 解决方案 → 算法（贯穿会话）

| # | 问题 | 解决方案 | 算法/机制 |
|---|---|---|---|
| 1 | lazy v2 per-layer dontneed 过多 page fault、TPS 崩 | 清理死代码，转向显式缓冲流式 | — |
| 2 | **mmap+madvise 假降**：VmRSS 降但 cgroup `memory.current` 顶满 cap（cache 回填）| 权重不能 mmap-backed，改**匿名缓冲 + O_DIRECT** | §3.4 O_DIRECT 流式 |
| 3 | buffered pread 文件页被仍存在的 mmap 钉住（fadvise 丢不掉）| 必须 **O_DIRECT** 绕 page cache | §3.4 |
| 4 | moe-buffer `streams=0`：repoint 了 weights_map meta 张量、按指针匹配失败 | 改**按张量名匹配** + 在回调里 repoint **真计算张量** | §3.2 切片 repoint |
| 5 | 纯"用到才加载"同步 fault → 掉 TPS | **CLG 预测** + 异步预取拿 lead time | §3.1 CLG |
| 6 | node_callback 不能保证 kernel 读前就位 | **weight-stream 回调**同步兜底（ith0+barrier）| §3.3 状态机 |
| 7 | 单线程 O_DIRECT 欠载盘（488MB/s）| **多 worker 并行 pread**（4 打满 1.2GB/s）| §3.5 worker 池 |
| 8 | 热专家**绝对阈值**累计涨过 budget（resident 510→1077）| 改**相对频率判据**（有界）| §3.6 相对热 |
| 9 | **dense mmap 驱逐假降**（4560==4560）| dense 换**匿名 ring backing**（flex）| §3.7 ring LRU |
| 10 | 自适应读到根 cgroup 限额（=max）误判 | 解析 **`/proc/self/cgroup`** 找自己的子 cgroup | §3.8 自适应 |
| 11 | 量测噪声：VmRSS 假象、`-r1` ±1t/s | 用 **cgroup memory.current** + **evictions→0 拐点** + `-r3` | TEST_RESULTS §0 |
| 12 | native 比 mmap window 还高 2GB | **关 repack**（去掉不可回收的重排副本）| §3.9 |
| 13 | O_DIRECT 任意 offset/stride 不对齐 | **对齐 bounce buffer** + memcpy | §3.4 |

---

## 3. 核心算法及实现代码

### 3.1 CLG（Cross-Layer Gate）专家预测器

**问题**：MoE 一层的激活专家要等该层 router 跑完才知道，同步加载会 stall。
**算法**：在更早的 `ffn_inp[L]`（≈`ffn_inp[L+1]`，相邻层高余弦相似）上**复算 router 打分**，预测 L+1 的 top-(K+δ) 专家，提前异步预取。

```cpp
// src/llama-window.cpp  llama_window_clg_predict_and_schedule()
// 1. RMS norm
float sum_sq = 0.0f;
for (int d = 0; d < n_embd; ++d) sum_sq += hs[d] * hs[d];
const float inv_rms = 1.0f / std::sqrt(sum_sq / (float) n_embd + clg.norm_eps);
for (int d = 0; d < n_embd; ++d) tl_hs_norm[d] = hs[d] * inv_rms * clg.norm_w[d];

// 2. gate scores: scores[e] = dot(hs_norm, gate_w[e])   (gate_w 行主序 [n_expert × n_embd])
for (int e = 0; e < n_expert; ++e) {
    const float * gw = clg.gate_w.data() + (int64_t) e * n_embd;
    float score = 0.0f;
    for (int d = 0; d < n_embd; ++d) score += tl_hs_norm[d] * gw[d];
    tl_scores[e] = score;
}

// 3. top-(K+δ) 线性扫描，置位预测掩码
uint64_t local_mask = 0;
for (int k = 0; k < n_select; ++k) {        // n_select = n_expert_used + clg_delta
    int best_e = -1; float best = -1e38f;
    for (int e = 0; e < n_expert; ++e)
        if (!(local_mask & (1ull<<e)) && tl_scores[e] > best) { best = tl_scores[e]; best_e = e; }
    if (best_e >= 0) local_mask |= (1ull << best_e);
}
predicted_mask |= local_mask;

// 4. buffer 模式：把预测专家发给 moe-buffer 异步预取（拿 lead time）
if (has_buffer) {
    int ids[64]; int n = 0;
    for (int e = 0; e < n_expert && e < 64; ++e)
        if (predicted_mask & (1ull << e)) ids[n++] = e;
    llama_moe_buffer_prefetch(ctx.moe_buffer, next_layer, ids, n);
}
```

触发点（node_callback 拦截 `ffn_inp-{L}`）：
```cpp
// src/llama-window.cpp  llama_window_node_done()
if (std::sscanf(name, "ffn_inp-%d", &inp_layer) == 1 && inp_layer+1 < n_layers)
    llama_window_clg_predict_and_schedule(ctx, inp_layer + 1, node);
```

### 3.2 按专家切片 repoint（无需改 GEMM kernel）

**问题**：要让 `mul_mat_id` 读匿名缓冲而非 mmap，且 weights_map 的 meta 张量 ≠ 计算图张量。
**算法**：给每个 `*_exps` 分配**全尺寸匿名缓冲**，在 weight-stream 回调里**按名字**找到真计算张量并把 `->data` 指向缓冲；`mul_mat_id` 照常按 `id*stride` 索引，激活专家槽已流入、冷槽 zero-fill 不占物理。

```cpp
// src/llama-moe-buffer.cpp  llama_moe_buffer_stream_callback()
if (op->op != GGML_OP_MUL_MAT_ID) return false;
ggml_tensor * exps = op->src[0];
auto it = ctx->by_name.find(ggml_get_name(exps));   // 按【名字】匹配（不是指针）
if (it == ctx->by_name.end()) return false;
moe_managed & m = it->second;
if (ith == 0) {
    moe_ensure_buf(m);                  // 懒分配 ggml_nbytes 全尺寸匿名缓冲
    exps->data = m.buf;                 // repoint 真计算张量 → GEMM 读我们的缓冲
    ggml_tensor * ids = op->src[2];     // 选中专家 id (I32)
    const int32_t * idp = (const int32_t *) ids->data;
    for (int64_t i = 0; i < ggml_nelements(ids); ++i)
        moe_stream_slice(*ctx, m, (int) idp[i]);   // 确保每个选中专家驻留
}
return true;   // 接管 → CPU 后端在 kernel 前插 barrier
```

### 3.3 同步/异步协调：COLD/RESIDENT/INFLIGHT 状态机

**问题**：异步 worker 与同步回调可能同时请求同一专家；pread 慢不能在锁内做。
**算法**：每槽三态 + 条件变量。pread 在**锁外**执行（thread_local bounce 各线程独立）；第二个请求者见 INFLIGHT 则等待而非重复读。

```cpp
// src/llama-moe-buffer.cpp  moe_stream_slice()
std::unique_lock<std::mutex> lk(ctx.mtx);
for (;;) {
    const char s = m.resident[e];
    if (s == ST_RESIDENT) {                                   // 命中
        ctx.lru.splice(ctx.lru.begin(), ctx.lru, m.lru_pos[e]);   // 触达 MRU
        if (m.activation[e] != UINT32_MAX) { m.activation[e]++; m.total_act++; }
        ctx.hits++; return;
    }
    if (s == ST_INFLIGHT) { ctx.cv_done.wait(lk); continue; } // 别人在读 → 等
    break;                                                    // COLD → 我读
}
moe_ensure_buf(m);
m.resident[e] = ST_INFLIGHT;
if (ctx.params.budget_bytes > 0)                              // 驱逐到预算内
    while (ctx.resident_bytes + m.stride > ctx.params.budget_bytes && moe_evict_lru(ctx)) {}
ctx.resident_bytes += m.stride;
lk.unlock();
const bool ok = moe_pread_slice(m, e, ctx.align);            // 【锁外】O_DIRECT 读
lk.lock();
if (ok) { m.resident[e]=ST_RESIDENT; ctx.lru.push_front({&m,e}); m.lru_pos[e]=ctx.lru.begin();
          if (m.activation[e]!=UINT32_MAX){m.activation[e]++; m.total_act++;} ctx.streams++; }
else    { m.resident[e]=ST_COLD; ctx.resident_bytes -= std::min(ctx.resident_bytes, m.stride); }
ctx.cv_done.notify_all();
```

### 3.4 O_DIRECT 对齐 bounce 读

**问题**：O_DIRECT 要求偏移/长度/缓冲对齐到块大小，但专家切片 offset/stride 任意。
**算法**：把 `[foff, stride]` 向下/上对齐到页，读进**对齐 bounce**，再 memcpy 出有效区；非块设备回退 buffered + fadvise。

```cpp
// src/llama-moe-buffer.cpp  moe_pread_slice()  （O_DIRECT 分支）
const size_t aoff = foff & ~(A - 1);            // 向下对齐起点
const size_t head = foff - aoff;                // 有效数据在 bounce 内偏移
size_t want = (head + m.stride + A - 1) & ~(A - 1);   // 向上对齐长度
if (aoff + want > m.fsize) want = m.fsize - aoff;     // EOF 夹取
// tls_bounce 为 thread_local 对齐缓冲（各 worker 独立，pread 可并行）
ssize_t r = pread(m.fd, tls_bounce, want, (off_t) aoff);
if (r < 0 || (size_t) r < head + m.stride) return false;
std::memcpy(dst, tls_bounce + head, m.stride);  // 拷出有效切片到匿名缓冲槽
```

fd 以 O_DIRECT 重开（失败回退）：
```cpp
dfd = open(proc, O_RDONLY | O_DIRECT);  if (dfd >= 0) direct = true;
if (dfd < 0) dfd = open(proc, O_RDONLY);
```

### 3.5 多 worker 并行读（榨干盘带宽）

**问题**：单线程 O_DIRECT 488MB/s 欠载盘（盘可到 1.2GB/s）。
**算法**：worker 池 + 队列，多线程并发 `moe_stream_slice`（pread 锁外 → 真并行）。

```cpp
// src/llama-moe-buffer.cpp
for (int i = 0; i < n; ++i)                       // create 时起 N 个 worker
    ctx->workers.emplace_back(moe_worker_loop, ctx.get());

static void moe_worker_loop(llama_moe_buffer_context * ctx) {
    for (;;) {
        std::pair<moe_managed*, int> task;
        { std::unique_lock<std::mutex> lk(ctx->qmtx);
          ctx->cv_q.wait(lk, [ctx]{ return ctx->stop || !ctx->queue.empty(); });
          if (ctx->stop && ctx->queue.empty()) return;
          task = ctx->queue.front(); ctx->queue.pop_front(); }
        moe_stream_slice(*ctx, *task.first, task.second);   // 并行 pread
    }
}
// prefetch(layer) 把该层各 exps 的预测专家入队 → 唤醒 worker
```

### 3.6 相对频率热专家保护（有界 pin）

**问题**：绝对累计阈值 pin 集合随序列无界增长，涨过 budget。
**算法**：hot ⟺ 激活率 > `hot_ratio` × 本张量平均（分子分母同步增长 → 比例稳定 → 有界）。

```cpp
// src/llama-moe-buffer.cpp  moe_is_hot()
static bool moe_is_hot(const llama_moe_buffer_context & ctx, const moe_managed & m, int e) {
    if (ctx.params.hot_ratio <= 0.0f) return false;
    if (m.total_act < (uint64_t) m.n_expert * 2) return false;        // warmup
    return (double) m.activation[e] * m.n_expert >                    // activation[e]
           (double) ctx.params.hot_ratio * (double) m.total_act;      //  > ratio × mean
}
```

驱逐时跳过 hot：
```cpp
// moe_evict_lru()：从 LRU 尾向前找第一个非 hot 的槽
for (auto it = ctx.lru.end(); it != ctx.lru.begin(); ) {
    --it;
    if (!moe_is_hot(ctx, *it->first, it->second)) { victim = it; break; }  // 最冷非 hot
}
if (victim == ctx.lru.end()) return false;   // 全 hot → 允许暂超 budget
```

### 3.7 dense 匿名 ring LRU 驱逐（真降）

**问题**：mmap 驱逐假降。
**算法**：固定 k 个匿名 slot；请求层用空 slot，满则驱逐 **last_use 最旧 + 已 released（compute 用完）+ resident** 的层，复用其 slot（匿名页覆盖 = 物理释放）。dense 顺序访问下等价环形 FIFO。

```cpp
// src/llama-flex.cpp  flex_acquire_slot()
for (size_t s = 0; s < ctx.slots.size(); ++s)            // 1) 空 slot 优先
    if (ctx.slot_layer[s] < 0) { ctx.slot_layer[s] = layer; return (int) s; }
int victim = -1; uint64_t oldest = UINT64_MAX;           // 2) 驱逐 LRU+已释放
for (int l = 0; l < ctx.n_layers; ++l) {
    auto & L = ctx.layers[l];
    if (L.slot >= 0 && L.released && L.state == layer_state::resident && L.last_use < oldest)
        { oldest = L.last_use; victim = l; }
}
if (victim < 0) return -1;                                // 都在用 → backoff 重试
int slot = ctx.layers[victim].slot;
ctx.layers[victim].slot = -1; ctx.layers[victim].state = layer_state::not_resident;
ctx.slot_layer[slot] = layer; return slot;
```

### 3.8 自适应 budget / ring（按可用内存定档）

**问题**：手动调 budget 麻烦；读根 cgroup 误判。
**算法**：解析本进程 cgroup 拿真实可用内存；装得下不驱逐，装不下 `budget = available − 不可驱逐部分 − reserve`。

```cpp
// src/llama-model.cpp  llama_detect_available_memory()
//  从 /proc/self/cgroup 取 "0::<path>" → 读自己的 memory.max/current（不是根 cgroup）
if (std::strncmp(line, "0::", 3) == 0) { cg_path = line + 3; ... }
const std::string base = "/sys/fs/cgroup" + cg_path;
... cmax = read(base+"/memory.max"); cur = read(base+"/memory.current");
avail = std::min(avail, cmax > cur ? cmax - cur : 0);
... avail = std::min(avail, MemAvailable_from_/proc/meminfo);

// MoE 自适应 budget（load_tensors 内）
const size_t non_expert = model_total_bytes - expert_bytes;
if (avail >= non_expert + reserve + expert_bytes) budget = 0;          // 装得下 → 不驱逐
else { budget = avail > non_expert+reserve ? avail-non_expert-reserve : 0;
       if (budget == 0) budget = 64*MiB;                                // 至少一片
       if (budget > expert_bytes) budget = 0; }
llama_moe_buffer_set_budget(moe_buffer, budget);

// dense 自适应 ring（flex）
if (avail < non_layer + layer_total + reserve) {
    const size_t room = avail > non_layer+reserve ? avail-(non_layer+reserve) : 0;
    ring = clamp(room / max_layer, prefetch_ahead+2, n_layers);         // 受限缩 ring
} // 否则 ring = n_layers（全留，等价不流式）
```

### 3.9 关 repack（免费降 RSS）

**问题**：native 默认 repack 把量化权重重排成 CPU 友好布局，多存一份**不可回收**匿名副本（dense ~2GB / MoE ~2.1GB）。
**算法**：window/flex 路径关闭 `extra_bufts`，直接用 mmap 原始布局。实测不掉 TPS（消除的是冗余副本不是工作集）。

```cpp
// src/llama-model.cpp
pimpl->cpu_buft_list = make_cpu_buft_list(
        devices,
        (use_lazy_window || use_flex) ? false : params.use_extra_bufts,  // 关 repack
        ...);
```

---

## 4. 环境变量总表（当前已实现）

| 变量 | 模块 | 含义 |
|---|---|---|
| `LLAMA_LAZY_V2=1` | window | 启用 lazy/window 框架 |
| `LLAMA_LAZY_MOE_BUFFER=1` | moe-buffer | 启用 MoE 专家匿名缓冲流式 |
| `LLAMA_LAZY_MOE_BUFFER_MB=N` | moe-buffer | 固定专家驻留预算（LRU 驱逐到此）|
| `LLAMA_LAZY_MOE_BUFFER_AUTO=1` | moe-buffer | 自适应 budget（按可用内存）|
| `LLAMA_LAZY_MOE_BUFFER_WORKERS=N` | moe-buffer | 并行 pread worker（实测 4 最优）|
| `LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=R` | moe-buffer | 相对热专家阈值（激活率 > R× 平均则 pin；典型 2.0）|
| `LLAMA_LAZY_CLG=1` | window | 启用 CLG 预测预取 |
| `LLAMA_LAZY_CLG_DELTA=N` | window | CLG 过预测余量 δ（TPS 旋钮）|
| `LLAMA_FLEX=1` | flex | dense 匿名 ring 流式 |
| `LLAMA_FLEX_AUTO=1` | flex | 自适应 ring（按可用内存定层数）|
| `LLAMA_FLEX_RING=N` / `LLAMA_FLEX_THREADS=N` | flex | 固定 ring 层数 / 并行 IO 线程 |

---

## 5. 正确性保证

- **数学等价**：切片 repoint 不改 GEMM，`mul_mat_id` 仍按 `id*stride` 读，激活专家数据 bit-exact。
- **同步兜底**：weight-stream 回调（ith0 + barrier）保证 kernel 读前所有选中专家 RESIDENT。
- **实测**：moe-buffer+CLG vs window PPL **4 chunk 逐位一致**（47948.2078 / 80646.3078 / 95059.6850 / 104976.4199）；改热专家/自适应后默认路径复测仍一致。
- **退化安全**：不设新 env 时行为与改动前完全一致（`hot_ratio=0` 纯 LRU、`budget=0` 不驱逐、未设 AUTO 不触发检测）。
