# llama-window: 图感知的滑动窗口 mmap 预取/回收设计文档

**目标**：在权重保持 mmap **零拷贝**的前提下，降低 CPU 推理的物理内存峰值，同时**不增加**（统计意义上）推理时延——让"内存差一点 / 想腾出内存"的场景以接近 native 的速度运行。

**结论**：在不受限内存下，window 相对 native 取得 **稳态 RSS −44%（7.68GB → ~4.3GB）**，而 **decode 速度与 native 在测量噪声内相等**（native 7.56±0.14 vs window 7.55±0.08 t/s）。RSS 的降低来自**不做全文件 prefetch**，是真实且可复现的；速度上的任何"更快/更慢"都在这台共享机的算力波动（±~1 t/s）之内。

> 适用定位：window 是"**和 page cache 合作、轻量省内存、近乎不掉速**"的中间档。模型装不下、native 直接 OOM 的极端受限场景，应改用 [llama-flex](FLEX_INFER_DESIGN.md)（不 mmap、显式缓冲 + direct IO）。

---

## 1. 背景与问题

llama.cpp 默认用 mmap 做权重 offloading。对 dense 模型的 CPU 推理，mmap 有两个互斥的失败模式：

1. **开 prefetch（`MAP_POPULATE` / 全文件 `MADV_WILLNEED`）**：加载时把整模型（4.58GB）全部 fault 进来 → 峰值 RSS 高（native 实测稳态 ~7.68GB）。
2. **不开 prefetch（懒加载）**：页错误全部砸在**计算线程**上 → 计算到某层时同步 major fault、卡顿。

dense 模型每个 token 顺序跑全部 N 层（`0→1→…→N-1` 循环），下一层高度可预测。window 利用这个**完全可预测的顺序访问**，把"按需加载"做成"**提前、重叠、零拷贝**"的预取，并可选地按预算回收远层。

---

## 2. 架构

三个组件协作：

| 组件 | 角色 |
|---|---|
| **node_callback** | ggml-cpu 在每个计算 node 完成后回调 → 提供"计算现在在第 L 层"的**时序信号**。 |
| **window controller** | 据层号决定预取/保护/回收哪些层。**调度大脑**（`llama-window.cpp`）。 |
| **POPULATE_READ** | 预取 worker 真正把页 fault 进 RAM（不只是 advise）。**让预取落实**。 |

核心特征：**权重张量 `->data` 始终指向 mmap 地址，全程零拷贝**；window 只对这些地址做 `madvise`，不复制、不换指针。residency 交给内核 page cache 管理，window 只是"提前喂热 + 可选丢冷"。

### 2.1 时序信号：node_callback

ggml-cpu 在 `ggml_cplan` 上新增 `node_callback`，由 threadpool 在**每个 node 完成、跨过 barrier 后**于 `ith==0` 触发。`llama_window_cpu_node_done` 把它转给 window。这是整套机制的驱动源——window 由此精确知道"计算跑到哪一层"，而不是靠访问时被动 fault。

### 2.2 调度大脑：window controller

- **加载期索引**（`llama_window_create`）：对每个权重张量，从名字 `blk.N.` 解析层号，把其 mmap 地址区间**页对齐**后按层合并成 `ranges`。
- **每层状态机**：`cold → queued → advised → resident`。
- **推进**（`llama_window_advance_locked`）：换层到 L 时
  - **预取** `L+1 .. L+prefetch_ahead`（入队 prefetch 任务，受 `prefetch_budget` 限）。
  - **回收**（当 `use_dontneed` 且 `memory_limit==0` 或超限）：对**保护窗口外**的层入队 reclaim 任务（受 `reclaim_budget` 限），按"距当前层距离 + 层大小"打分优先丢最远最大的。
- **保护窗口**（`llama_window_is_protected`）：`[L-keep_behind, L+prefetch_ahead]` 内的层永不回收。

### 2.3 执行手段：POPULATE_READ

预取 worker（`llama_window_worker`）对任务区间：
- **prefetch** → `llama_window_populate`：优先 `MADV_POPULATE_READ`（Linux 5.14+，一个 syscall 同步把整段 fault 进来、带 readahead、不抛 SIGBUS）；老内核回退 `MADV_WILLNEED` + 逐页 touch。
- **reclaim** → `MADV_DONTNEED`。

**POPULATE_READ 的意义**：之前只 `MADV_WILLNEED` 是非绑定建议，内核可能没真读进来 → 计算线程到该层仍踩**同步 major fault**。改成 POPULATE_READ 后，**worker 线程把页同步读入、与计算重叠**，热路径不再 fault。

### 2.4 自适应窗口

`llama_window_auto_tune_locked` 用 EMA 跟踪"每层计算间隔"与"预取延迟"：预取跟不上计算就增大 `prefetch_ahead`，内存压力大就减小；`keep_behind` 随之调整。

---

## 3. 正确性

`MADV_DONTNEED` 在此**安全**，因为映射是 **`PROT_READ | MAP_SHARED`（只读文件映射）**：

1. **没有任何代码写权重** → 无脏页 → 无回写 → 数据恒等于文件原始字节。
2. **DONTNEED 后重新 fault → 内核从文件/page cache 重读 → 拿回逐字节相同的数据**。（这与匿名/私有映射不同——那种 DONTNEED 会清零、是数据丢失；只读文件映射是纯缓存驱逐。）
3. **跨层共享页保护**：`llama_window_create` 检测不同层落在同一页的区间并标记 `safe=false`，不回收，避免误丢另一层数据。
4. **活跃窗口保护**：当前层 + `keep_behind/prefetch_ahead` 范围不回收。

实测：window 各配置（预取 / 回收 / prefill_aggressive）的贪心解码输出与 native **逐字一致**。

---

## 4. 结果（已用误差棒校正）

测试：Llama-3-8B Q4_K_M（4.58GB），8 核（`-t 6`），不受限内存，稳态 VmRSS 采样 + `llama-bench -r 3`。

| 方案 | decode（带误差棒） | 稳态 RSS（中位） |
|---|---|---|
| native | 7.56 ± 0.14 t/s | 7682 MB |
| **window** | **7.55 ± 0.08 t/s** | **~4300 MB** |

**两点稳健结论**：
1. **RSS −44%（7.68GB → ~4.3GB）**：真实、可复现。来源是**不做全文件 prefetch**——`prefetch_only`（不回收）就已降到 ~4506MB。
2. **速度与 native 统计无差异**：7.55 vs 7.56，误差棒重叠。

> **测量噪声警告**：这台共享机的 CPU 吞吐波动达 ±~1 t/s（17%）。本会话早期基于单样本（`-r 1`）得出的"window −22% 慢"和"+9% 快"**都是噪声**，已作废。在此环境必须 `-r 3` 以上看误差棒才能下速度结论。

### 4.1 memory_limit 在 dense 模型下基本无效

`memory_limit` 4GB vs 1GB：RSS 仅差 ~80MB（噪声内），速度无可测差异。原因（见 §6）：dense 模型循环访问，被 `DONTNEED` 的层下个 token 立刻被重新 fault 回来，**稳态 RSS 有个 ≈模型大小的硬地板**，回收压不动它（reclaim 相对 prefetch_only 只多省 ~250MB/5%，代价是页表 churn）。

---

## 5. 与 ggml / llama 的集成

- **ggml-cpu**：`ggml_cplan.node_callback` + threadpool 每 node barrier 后于 `ith==0` 触发；`ggml_backend_cpu_set_node_callback` 经 backend reg proc address 暴露。
- **llama-model.cpp**：`LLAMA_LAZY_V2`/`LLAMA_LAZY_LOADING` 置位时（关 prefetch、关 mlock、关 extra_bufts），从 `weights_map` 建 `region_input`（mmap 地址 + 偏移），按 `LLAMA_LAZY_*` env 配参，`llama_window_create`。
- **llama-context.cpp**（decode 周围）：装 `llama_window_cpu_node_done` 回调 + `llama_window_graph_begin(prefill)`，图算完 `graph_end` 并卸回调。

---

## 6. 设计评估与已知限制（诚实）

### 6.1 memory_limit / reclaim 对 dense 模型名存实亡

- **不受限内存**：reclaim 相对"只预取"只多省 ~5% RSS，换持续的页表 churn + madvise syscall；`memory_limit` 旋钮在 4GB↔1GB 间无可测效果。**作为 RSS 控制旋钮是误导**。
- **受限内存（cgroup < 模型）**：真正强制 RSS 上限的是 **cgroup 本身**（内核自行回收），不是 app 的 DONTNEED。app reclaim 唯一可能的价值是"保护预取窗口不被内核误逐"，但 dense 循环访问无局部性可利用，大概率多余。
- **待验实验**：cgroup 受限下 window **reclaim 开/关**对比。若关掉不变差 → reclaim 可彻底删；若变差 → 保留但**改名为"预取窗口保护"，而非 memory_limit RSS 旋钮**。
- **倾向**：dense 模型下 window 可退化为纯"**graph-aware 预取**"（无 reclaim、无 memory_limit），把 RSS 控制交给 cgroup / 交给 flex。

### 6.2 其它限制

- **dense + CPU 后端**：node_callback 在 ggml-cpu；GPU/MoE 未覆盖。
- **RSS 地板 ≈ 模型**：不受限内存下降不过模型大小（循环再 fault）。要把工作集做到模型以下且留得住，需 flex 的"不 mmap"路线。
- **每 node 回调 + 强制 barrier**：理论上有逐 node 开销（解析 + 原子 + barrier），但实测被 CPU 噪声淹没，不显著。

### 6.3 与 flex 的分工

| 内存 vs 模型 | 方案 | 机制 | RSS / 速度 |
|---|---|---|---|
| 装得下 / 差一点 | **window** | mmap 零拷贝 + 图感知预取（+可选回收） | −44% RSS / ≈native 速度 |
| 装不下（native OOM） | **flex** | 不 mmap，显式 ring 缓冲 + O_DIRECT + locking | RSS 可降到模型以下 / IO 瓶颈 |

两者共用 node_callback 时序机制；window 是谱线"够用端"，flex 是"远不够端"。

---

## 7. 配置（环境变量）

| 变量 | 含义 | 默认 |
|---|---|---|
| `LLAMA_LAZY_V2` / `LLAMA_LAZY_LOADING` | 启用 window（>0） | 关 |
| `LLAMA_LAZY_WINDOW_SIZE` | 窗口层数 | 12 |
| `LLAMA_LAZY_PREFETCH_AHEAD` | 提前预取层数 | 2 |
| `LLAMA_LAZY_WORKER_THREADS` | 预取/回收 worker 数 | 1 |
| `LLAMA_LAZY_DONTNEED` | 启用回收（>0） | 关 |
| `LLAMA_LAZY_MEMORY_LIMIT` | 回收触发预算（GB；dense 下基本无效，见 §6.1） | 0 |
| `LLAMA_LAZY_PREFILL_AGGRESSIVE` | prefill 阶段激进预取整窗 | 关 |
| `LLAMA_LAZY_AUTO_TUNE` | 自适应 prefetch_ahead | 开 |
| `LLAMA_LAZY_DEBUG` | 调试日志 | 关 |

**推荐用法（dense）**：`LLAMA_LAZY_V2=1`（**只预取、不设 memory_limit**）即可拿到 −44% RSS、≈native 速度的最佳折中。

---

## 8. 代码清单

| 文件 | 内容 |
|---|---|
| `src/llama-window.{h,cpp}` | 窗口控制器：索引 / 状态机 / 预取 / 回收 / POPULATE_READ / auto-tune |
| `ggml/include/ggml-cpu.h`、`ggml/src/ggml-cpu/ggml-cpu.{c,cpp}` | node_callback 基础设施 + 每 node barrier 触发 |
| `ggml/src/ggml-backend*.{cpp,h}` | `ggml_backend_cpu_set_node_callback` 暴露 |
| `src/llama-model.cpp` | region 构建 + 参数 + 创建 |
| `src/llama-context.cpp` | decode 周围装/卸回调、graph_begin/end |

---

**文档版本** v1.0 · 状态：可用，dense regime 已用误差棒校正测量。speed=native（噪声内）、RSS −44% 为稳健结论；memory_limit/reclaim 对 dense 的去留待 §6.1 隔离实验定。
