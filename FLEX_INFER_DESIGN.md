# llama-flex: FlexInfer 式权重流式 offloading 设计文档

**目标**：在 dense LLM 的 CPU 推理中，把物理内存占用（RSS）降到模型大小**以下**且**可调**，同时在内存受限场景下保持可用的推理速度——尤其是让模型在**装不进 RAM** 时仍能运行。

**结论**：通过**抛弃 mmap、改用显式 ring 缓冲 + direct IO + balanced memory locking**（FlexInfer[^flexinfer] 路线），在 cgroup 限制到 3GB（模型 4.58GB 装不下）的条件下，相对最佳 mmap 方案取得 **prefill 4.6×、decode 8.6×** 提速，且 RSS 更低——native mmap 在此条件下直接 OOM。

[^flexinfer]: Du et al., *FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference*, arXiv:2503.03777.

---

## 1. 背景与问题

### 1.1 dense 模型的根本约束

transformer decode 阶段每生成一个 token 都要顺序跑全部 N 层：`layer 0→1→…→N-1`，然后立刻重复。这**不是**对序列的滑动窗口，而是一个几十毫秒一圈的紧循环——**每一层每个 token 都要用一次**。

由此推出一条硬约束：

> 任何被逐出物理内存的层，下一个 token 必然又要用。

因此对 dense 模型：
- **RSS 降到工作集以下 ⟺ 每个 token 都要从存储重读被逐出的部分**。
- 稳态吞吐有一个与算法无关的硬上限：`tok/s ≤ 磁盘带宽 / 每 token 重读字节数`。

降 RSS 的唯一出路是：(1) 把不可避免的 IO **藏到计算后面**（异步预取/双缓冲）；(2) 让发生的 IO **高效**（大块、顺序、不在计算线程上 fault）。

### 1.2 为什么 mmap 方案有天花板（已实证）

llama.cpp 默认用 mmap offloading。我们先尝试在 mmap 基础上做"滑动窗口"（`llama-window`：对页对齐的层区间做 `MADV_WILLNEED`/`MADV_POPULATE_READ` 预取 + `MADV_DONTNEED` 回收，由 ggml node 回调驱动）。实测结论：

| 测量 | 结果 |
|---|---|
| 稳态 RSS（生成 256 token 采样 VmRSS） | window ~4.26GB ≈ **93% 模型**，与 `LLAMA_LAZY_MEMORY_LIMIT`（1/1.5/2/3 GB）**无关** |
| 原因 | mmap 是**文件页**，`DONTNEED` 丢掉后被循环访问**立刻重新 fault**回来；内核控制驻留，应用只能"建议"不能"强制边界" |
| cgroup memory.max=3GB（< 模型） | native mmap+prefetch **OOM killed**；window 能跑但 **0.40 tok/s**（纯磁盘 IO 瓶颈） |

**mmap 的 RSS 有一个 ~模型大小的硬地板，且"内存越多越快"不成立**——这正是 FlexInfer 论文 §2.3 / §4.2 批判 mmap 的点。要突破，必须在架构上换掉 mmap。

---

## 2. 架构：llama-flex

核心思想（FlexInfer §3）：**不 mmap 流式权重**。只保留 k 层的显式堆缓冲（ring），用后台 IO 线程以大块顺序读把下面 k 层的张量流入缓冲，计算用完即释放槽位。权重的驻留量因此被 ring 硬绑定，与 page cache 无关：

```
RSS_weights ≈ (k / N) × model_size  +  locked_bytes
```

三个组件：

1. **异步流式预取**（§2.1）：k 层 ring buffer + IO worker 线程。
2. **balanced memory locking**（§2.2）：用富余内存把每层均匀的一部分永久锁驻，降低每 token 的 IO。
3. **flexible tensor preservation**（§2.3）：决定锁哪些张量（小张量/attention 优先）。

### 2.1 异步流式预取（ring buffer）

- **slot**：一块按页对齐的缓冲，大小 = 各层"被流式部分"的最大值。ring 有 k 个 slot。
- **状态机**：每层 `not_resident → loading → resident`；`released` 标记计算已消费、槽位可复用。
- **IO worker**：从队列取层 → 取一个空闲/可驱逐（已 released 的 LRU）slot → `pread` 该层未锁张量 → 标记 resident、通知等待者。
- **槽位驱逐**：只驱逐 `released && resident` 的层（当前正在算的层不会被驱逐）。
- **驱动**：计算推进到新层 L 时，预取 `L+1..L+ahead`、释放 `L-1`（见 §3 集成）。

理论模型（FlexInfer 公式 4）：`T_async = 1 / max(每 token 计算延迟, IO_size / 带宽)`。计算 ≥ IO 时 IO 被完全藏住。单测实证：注入每层 5ms 模拟计算后，`waits 56→2`（IO 几乎不再阻塞计算）。

### 2.2 balanced memory locking

仅靠流式，RSS 低但纯 IO 瓶颈，且"内存越多"无收益（预取不减少每 token 的 IO 总量）。balanced locking 用富余内存把权重的一部分**永久驻留**，直接减少每 token 的 `IO_size`。

**为什么"均衡"**：若整层整层地锁（锁前 5 层、流后面的层），各层 IO 不均 → 计算/IO 线程互相等待 → 流水线 stall（论文 Figure 3a）。正解：**给每层相同的锁预算**（`lock_bytes / N`），锁每层的一小部分张量 → 每层流式 IO 一致 → 流水线均衡（Figure 3b）。

实现：lock buffer（持久、开局 `pread` 填一次），每层选张量锁到预算为止；其余流式走 ring。`stream_per_token = (1 − lock_fraction) × model`。

### 2.3 flexible tensor preservation

锁哪些张量？FlexInfer Algorithm 1 的思想：**内存少时优先 attention 张量**（更小 → 能锁更多个 → 消掉更多 IO 操作数；FFN 大张量留给高效的大块流式读）；GQA 下尤其优先 K/V（最小）。

实现：每层张量**按大小升序**排序后贪心锁到预算。升序 = 小张量（attn/K/V）优先 = "attention-first"，且填充率最大化。各层结构相同 → 锁同一组 → 自动均衡。

> **填充率上限**：受 FFN 张量粒度限制（如每层预算 64 MiB，锁完 attn ~24 MiB + 1 个 FFN 33 MiB = 57 MiB 后，第 2 个 FFN 放不下）。85% 左右是均衡锁定的固有上限——再填就得破坏每层均衡（论文反对）或拆张量（不可能）。

### 2.4 O_DIRECT

流式读用 `O_DIRECT` **绕过 page cache**（FlexInfer 明确："use direct IO to bypass the page cache"）。这在内存受限下至关重要：buffered `pread` 会把页缓存留在 cgroup 预算内，与工作集**抢内存 → 抖动**；O_DIRECT 读直达对齐 bounce → ring，零 cache 压力。

**对齐处理**：O_DIRECT 要求文件偏移/缓冲地址/读长度都按块（4096）对齐，而张量偏移/大小未必对齐。解法：每 IO 线程一个对齐的 **bounce buffer**，读"向下对齐的超集范围"进 bounce，再 `memcpy` 精确字节到目标。对任意模型正确；非块设备（tmpfs 等）自动回退 buffered。

---

## 3. 与 ggml 的集成

**难点**：权重张量的 `->data` 在加载时指向 mmap 地址，而计算要读到 ring slot 的数据。

**方案**：在 ggml-cpu 计算每个 node 前插入一个 **weight-stream 回调**，由 ith==0 线程 wait+repoint，再 barrier 同步，正常 kernel 读到改后的 `->data`——**不重写任何 mul_mat kernel**。

### 3.1 ggml-cpu 钩子

`ggml/include/ggml-cpu.h` 新增：
```c
typedef bool (*ggml_cpu_weight_stream_callback)(struct ggml_tensor * op, int ith, void * user_data);
void ggml_cpu_set_weight_stream_callback(ggml_cpu_weight_stream_callback cb, void * user_data);
```

`ggml/src/ggml-cpu/ggml-cpu.c` 的 `ggml_compute_forward()` 顶部（所有线程每 node 都经过）：
```c
if (g_weight_stream_cb != NULL && g_weight_stream_cb(tensor, params->ith, g_weight_stream_ud)) {
    ggml_barrier(params->threadpool);   // 发布 ith0 的 repoint，所有线程随后读到
}
```

回调对每个 op 返回一致的 bool（由张量名确定，所有线程一致）：是 flex 管理的权重 op → 返回 true（触发 barrier）；否则 false（无开销）。

### 3.2 flex 回调（`llama_flex_stream_callback`）

对 op 的每个 src：若名字在 flex 注册表中 → 记下层号；ith==0 时 `wait_layer(L)` + `data = get_tensor(L, name)` 重定向。并在层切换时（ith0）驱动 `request_layer(L+1..L+ahead)` 预取、`release_layer(L-1)`。

> 正确性关键：只 ith0 wait+repoint，barrier 后所有线程才读 kernel；`wait_layer` 保证数据已流入；mmap 仍是 `->data` 的"家"（提供合法指针+寻址），但**计算前被改指向 ring，从不经 mmap 读 → mmap 页永不 fault → RSS 低**。

### 3.3 model/context 接线

- `llama-model.cpp`：`LLAMA_FLEX` 置位时（与 window 互斥、关 prefetch 让 mmap 页保持冷）创建 flex 上下文、注册所有 `blk.N.*` 权重、`finalize`。
- `llama-context.cpp`（decode 周围）：`llama_flex_graph_begin()` + `ggml_cpu_set_weight_stream_callback(llama_flex_stream_callback, flex)`，图算完清回调。

---

## 4. 结果

测试：Meta-Llama-3-8B-Instruct Q4_K_M（4.58GB），8 核（`-t 6`），冷盘（279 MB/s），llama-bench `-p 16 -n 8`。

### 4.1 受限 regime（cgroup memory.max = 3GB，模型装不下）

| 方案 | 能跑 | RSS | prefill pp16 | decode tg8 |
|---|---|---|---|---|
| native (mmap+prefetch) | ❌ **OOM killed** | — | — | — |
| window (mmap+madvise，最佳 mmap) | ✅ | 3.06 GB | 4.62 | 0.40 |
| flex buffered（无锁） | ✅ | 1.04 GB | 2.76 | 0.65 |
| flex **O_DIRECT**（无锁） | ✅ | 1.16 GB | 18.64 | 2.58 |
| flex **O_DIRECT + lock 2GB** | ✅ | 2.73 GB | **21.48** | **3.43** |

**关键发现**：
1. **flex 让模型能跑**（native OOM 跑不起来）。
2. **O_DIRECT 比 buffered 快 4–7×**（不是慢）——受限下 buffered 的 page cache 与工作集抢内存抖动，O_DIRECT 绕过之。
3. **balanced locking：内存换速度**（decode 2.58→3.43，RSS 1.16→2.73GB 随 lock 预算可调）——这是 window 的 `memory_limit` 旋钮**完全做不到**的（那是平的）。
4. **vs 最佳 mmap 方案（window）：prefill 4.6×、decode 8.6×，RSS 更低**——落在 FlexInfer 论文 5–12× 区间内。

### 4.2 正确性

`--temp 0` 贪心解码，flex 各配置（buffered/O_DIRECT/带锁）输出均与 native **逐字一致**。

### 4.3 非受限内存下的代价

模型放得进 RAM 时 flex **更慢**（gen 10.9→3.2 t/s）：它在不需要时仍流式、且 O_DIRECT 不吃 cache。**flex 是受限场景的工具**，应仅在内存 < 模型时启用。

---

## 5. 配置（环境变量）

| 变量 | 含义 | 默认 |
|---|---|---|
| `LLAMA_FLEX` | 启用 flex（>0） | 关 |
| `LLAMA_FLEX_RING` | ring 槽位数 k | 4 |
| `LLAMA_FLEX_AHEAD` | 初始/固定预取提前层数 | 2 |
| `LLAMA_FLEX_ADAPTIVE_AHEAD` | 运行时自适应调节预取深度（0=固定 `LLAMA_FLEX_AHEAD`） | 1 |
| `LLAMA_FLEX_MAX_AHEAD` | 自适应预取深度上限 | 8 |
| `LLAMA_FLEX_LOCK_GB` | balanced locking 预算（GB） | 0（关） |
| `LLAMA_FLEX_THREADS` | IO 线程数 | 2 |
| `LLAMA_FLEX_BUFFERED` | 强制 buffered（关 O_DIRECT） | 关（默认 O_DIRECT） |
| `LLAMA_FLEX_DEBUG` | 调试日志 | 关 |

调参指引：RSS ≈ `(k/N)×model + lock_bytes + embd/output(~1GB) + 激活`。先用 `LLAMA_FLEX_RING` 设最低可行 RSS，再用 `LLAMA_FLEX_LOCK_GB` 把剩余内存预算换成速度。

---

## 6. 代码清单

| 文件 | 内容 | 行数 |
|---|---|---|
| `src/llama-flex.h` | 公开接口 | 106 |
| `src/llama-flex.cpp` | ring 流式 + locking + O_DIRECT + preservation | 589 |
| `tests/test-flex.cpp` | 独立单测（correctness / RSS 绑定 / 重叠） | 128 |
| `ggml/include/ggml-cpu.h` | weight-stream 回调声明 | +9 |
| `ggml/src/ggml-cpu/ggml-cpu.c` | 钩子调用点 + barrier | +15 |
| `src/llama-model.{cpp,h}` | flex 创建/注册/getter | +~55 |
| `src/llama-context.cpp` | decode 周围装/卸回调 | +~15 |

`llama-flex.cpp` 零 llama 依赖、可独立编译单测：
```bash
g++ -O2 -std=c++17 -pthread -I ggml/include tests/test-flex.cpp src/llama-flex.cpp \
    build/bin/libggml-base.so* -o /tmp/test-flex && /tmp/test-flex
```

---

## 7. 已知限制与后续

- **仅 dense + CPU 后端**：weight-stream 钩子在 ggml-cpu；GPU/MoE 未覆盖。
- **embedding/output 仍 mmap**（~1GB）：未流式，构成 RSS 固定项；可后续纳入流式。
- **2d 在规整命名模型上收益边际**：本模型 `weights_map` 字母序已近似 attention-first；价值在命名不规整/MoE 模型上。
- **填充率 ~85%**：FFN 粒度的固有上限（见 §2.3）。
- **后续可选**：自动从 cgroup 探测可用内存设 lock 预算；auto-tune ring/ahead；把 ring/lock 计入 ggml backend buffer 以纳入官方内存统计；MoE 按 router 输出流式专家。

---

## 8. 演进历程（供回溯）

1. **清理整合**：移除已死的 lazy-v2 / v1 copy 路径（`lazy_v2_ctx` 从未赋值、v1 `use_lazy_loading` 恒 false，均为死代码），收敛到单一路线。
2. **window populate**：把 `MADV_WILLNEED` 改 `MADV_POPULATE_READ`，消热路径 major fault；但实证 mmap RSS 天花板（§1.2）。
3. **cgroup 实证**：native OOM / window 0.40 t/s，确认需换架构。
4. **2a–2d + O_DIRECT**：实现 llama-flex，复现 FlexInfer 核心结果（§4）。

**文档版本** v1.0 · 状态：原型完成、受限 regime 已验证
