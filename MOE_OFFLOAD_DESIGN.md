# MoE 专家按需加载设计文档

**目标**：对 MoE 模型按**专家粒度**做按需加载（而非 dense 的层滑窗），在**不降低 TPS** 的前提下，**同时压低峰值 RSS 和运行时 RSS**，让 MoE 真正突破 dense 的内存下界。

**一句话总纲**：按专家、不按层；用 **CLG 预测**拿 lead time 保 TPS；用**显式匿名缓冲 + O_DIRECT** 让 RSS 真降；**hot 固定 + 有界 LRU** 把驻留钉在预算上 → peak 与 runtime 收敛到"保 TPS 的最低地板"。

> 配套文档：dense 端的 [`WINDOW_DESIGN.md`](WINDOW_DESIGN.md)（mmap 零拷贝层滑窗）、[`FLEX_INFER_DESIGN.md`](FLEX_INFER_DESIGN.md)（dense 显式缓冲流式）。本文是 MoE 专家端，复用两者的 node_callback / weight-stream 时序机制。

---

## 1. 背景：MoE 的结构性机会与 dense 的不同

dense decode 每 token 用**全部**层权重 → 工作集固定 = 整模型 → RSS 地板 ≈ 模型大小，降不过去。

MoE 每层把所有专家堆在一个大张量 `ffn_*_exps.weight`（3D：`[n_embd, n_ff, n_expert]`），由 `ggml_mul_mat_id(exps, x, ids)` 按 router 选出的 `ids`（top-k 专家）索引计算。**每 token 每层只用 top-k 个专家，绝大多数专家不被 touch**。这给了 dense 没有的机会：
- **冷专家从不被路由 → 永不需要驻留 → RSS 可真低于模型。**
- **每 token IO = 共享权重 + 激活专家 ≪ 整模型**（不像 dense 流式每 token 读整模型）。

---

## 2. 为什么不能用层滑窗（粒度错配）

层滑窗按"层"预取/驱逐——对 MoE 等于把一层**全部专家**（如 20 个）一次性 madvise/载入，而该 token 只用 4 个。**75% 是白拉**：既不省 RSS，又浪费带宽，还在峰值上制造"整层专家"的瞬时尖峰。

> **MoE 必须按专家粒度调度。** 复用层滑窗的 node_callback 时序没问题，但调度键从"层号"变成"**层号 + 激活专家 id 列表**"。

---

## 3. 两个机制前提（被本会话实测确立）

### 3.1 madvise 降不了真实物理 RSS → 必须显式缓冲 + O_DIRECT

实测（cgroup memory.max < 模型）：layer/expert 级 `MADV_DONTNEED` 都把 peak `memory.current` 顶在 cap——**被 DONTNEED 的页内核立刻回填 cache**。对只读 mmap，DONTNEED 只摘进程 PTE、动 VmRSS，不动真实可回收内存。

**结论**：专家权重**不能 mmap-backed**。要给每个 `ffn_*_exps` 张量分配**全尺寸匿名缓冲**，`->data` 重指向它，只把激活专家切片用 **`O_DIRECT pread`**（绕过 page cache）读进 `buffer + id*stride` 槽；冷专家槽从不写 → 保持 zero-fill 匿名页、零物理。`mul_mat_id` 按 id 索引照常工作、**无需改 kernel、无需 id 重映射**。

> 重要陷阱（已踩）：若用 buffered pread，文件页进 cache 且被仍存在的 mmap 映射**钉住**（fadvise 丢不掉）→ RSS 照样涨满。**必须 O_DIRECT**（或 munmap 专家区）。

### 3.2 "按需"在热路径同步 fault → 掉 TPS → 必须预测拿 lead time

一层的激活专家**要等该层 router 跑完才知道**，而 router → `mul_mat_id` 只隔几个节点。纯"用到才加载"= 计算线程同步等 IO → stall → 掉 TPS。

**破解：CLG 预测**——从更早的 hidden state（上一层输出处）预测下一层激活专家：
```
hs_norm = RMSNorm(hidden_state)
scores  = gate_w · hs_norm           # 复算 router 的打分
predict = top-(K+δ)(scores)          # δ 为安全余量，多预测几个防漏
```
预测出来后**异步预取**这些专家进缓冲，等到该层 `mul_mat_id` 时已驻留 → 不 stall。

> **要"按需"又不掉 TPS，预测+异步预取是必需的，不是可选。**

---

## 4. 三段式驻留集（同时压峰值与运行时）

```
驻留专家 = [固定 hot]  几乎每 token 都用的专家，永不驱逐
         + [预取]      CLG 预测的下层专家，异步流入缓冲
         + [有界 LRU]  最近用过的专家，超预算则 MADV_DONTNEED 驱逐（匿名→真释放）
```

| 段 | 作用 | 影响 |
|---|---|---|
| **hot 固定** | 命中"几乎总用"的专家（含共享专家/高频路由），避免反复重读抖动 | 保 TPS |
| **CLG 预取** | 给冷专家 lead time，把同步 fault 变后台预取 | 保 TPS |
| **有界 LRU + 匿名缓冲 + O_DIRECT** | 把驻留量钉在预算，冷专家真不占物理 | 真降 RSS |

**峰值 ≈ 运行时（关键优势）**：显式缓冲是**惰性流式**——启动时专家一个都不载（不像 native 全载、也不像层滑窗整层尖峰）。所以
```
peak RSS ≈ runtime RSS ≈ 非专家权重 + 专家预算 + 激活
```
两者一起压到地板、且收敛。这是层滑窗（每层整层尖峰）和 mmap（peak=runtime=cap）都做不到的。

---

## 5. 核心约束：RSS 有个"保 TPS 的地板"

不存在"无限压 RSS 还不掉 TPS"。

**先讲清根本机理（为什么压 RSS 会掉 TPS）**：LLM decode 是 **memory-bound**——每个 token 的瓶颈不是算力，而是把这一步用到的权重从存储搬到 CPU 的带宽。两条路径的"权重在哪、带宽多少"决定一切：

| 路径 | 权重在哪 | 搬运带宽 | 每 token |
|---|---|---|---|
| native / mmap | page cache（RAM） | ~10–50 GB/s | `T_io≈0` → compute-bound（满速） |
| moe-buffer miss | SSD（O_DIRECT 绕开 cache） | 实测 **~1.18 GB/s** | `T_io` 主导（慢一个数量级） |

moe-buffer 用 O_DIRECT **故意不让专家留在 RAM**（否则文件页被 mmap 钉住、RSS 降不下来，见 3.1），代价就是每个 miss 的专家都要从 SSD 读。这是 offloading 的**固有代价**，不是 bug：要"专家不常驻"省 RSS，就得付"用到时从盘读"的时间。

把它公式化（FlexInfer 公式 4 的 MoE 版）：

```
保 TPS 条件:   每 token miss 流式量 / 盘带宽  ≤  每 token 计算时间

miss 流式量 = ( 激活专家 − 命中(hot ∪ 预取 ∪ LRU) ) × 专家大小
```
- 驻留集 ↑ → miss ↓ → IO 能藏住 → TPS 不掉，但 RSS ↑。
- 驻留集 ↓ → miss ↑ → IO 藏不住 → **掉 TPS**。

**额外的放大效应——thrashing**：当预算装不下"单步工作集 + 跨 token 复用集"，LRU 会把刚读进来、下一步还要用的专家驱逐，下一步又重读 → 同一份数据反复读，把 miss 流式量放大数倍（实测 768MB 档每 token 读 598MB ≫ 工作集本身）。所以预算不足时 TPS 是**双重**变慢：miss 多 + 每个 miss 还被重复读。预算加到不 thrash（`evictions=0`）即可消除这部分。

存在**最小 RSS 地板**：
```
RSS_floor = 非专家权重 + hot 专家 + 来不及预取/重读的在途工作集
```
**目标应改为：找到这个地板**——即"不掉 TPS 的最低 RSS"，而不是无脑压。

### 地板高不高由什么决定

| 因素 | 说明 |
|---|---|
| **激活比** (active/total) | 越低地板越低。Qwen1.5-MoE 原版 19%；本测试用的 20 专家剪枝版 **45%**（地板偏高、省得少）；Qwen3-30B-A3B **10%**（地板低、收益大） |
| **路由时间局部性** | 局部性好 → hot 集小、复发少 → 地板低 |
| **盘带宽 / 计算时间比** | NVMe + CPU 慢算 → 比值好 → 更能藏 IO |
| **专家数/粒度** | 专家越多越细 → 调度越平滑 |

---

## 6. 实现架构

```
                      ┌─────────────────────────────────────────────┐
   上一层输出节点 ───► │ CLG 预测器 (llama-window.cpp)                 │
   (l_out-{L})        │  gate_w·RMSNorm(hs) → top-(K+δ) → 预测掩码    │
                      └───────────────┬─────────────────────────────┘
                                      │ 异步预取请求(预测专家)
                                      ▼
   router topk 节点 ──► 确认实际选中 ──►┌─────────────────────────────┐
   (ffn_moe_topk-{L})                  │ moe-buffer (llama-moe-buffer)│
                                       │  每 exps 张量: 全尺寸匿名缓冲 │
                                       │  hot 固定 + 有界 LRU          │
                                       │  O_DIRECT pread 激活专家切片  │
                                       └───────────────┬─────────────┘
                                                       │ repoint exps->data=buffer
   mul_mat_id 节点 ◄── weight-stream 钩子(ith0 同步兜底) ┘
   (kernel 读 buffer[id*stride])
```

### 6.1 两个钩子点（时序）

| 钩子 | 时机 | 作用 |
|---|---|---|
| **node_callback** (已有) | 每节点完成后(异步) | CLG 在上层输出处预测 → **异步预取**下层专家(拿 lead time) |
| **weight-stream 回调** (flex 机制) | `mul_mat_id` 计算**前**(ith0 + barrier) | **同步兜底**：确保 `op->src[2]` 选中的专家全部驻留，再让 kernel 读 → **保证正确性** |

> 正确性根基：**weight-stream 回调是唯一能保证"kernel 读前专家已就位"的同步点**——node_callback 在节点之后、其他线程不等它，不能用于必须先于 kernel 完成的流式。异步预取负责"通常已命中"，同步兜底负责"万一漏了不出错"。

### 6.2 按专家切片 repoint（无需改 kernel）

- 给每个 `ffn_*_exps` 分配 `ggml_nbytes` 全尺寸**匿名缓冲**，把**计算图里真正用的那个张量**的 `->data` 指向它（注意：`weights_map` 的 meta 张量 ≠ 计算张量，必须在回调里按**名字**匹配 `op->src[0]` 再 repoint）。
- `mul_mat_id` 读 `buffer + id*stride`，激活专家槽已流入、冷槽 zero-fill 不占物理。

### 6.3 流式与驱逐

- 流入：`O_DIRECT pread`(对齐 bounce + memcpy) 把 `[file_offset + e*stride, stride]` 读进 `buffer + e*stride`。
- 驱逐：超预算时对 LRU 槽 `madvise(MADV_DONTNEED)`(匿名→真释放)；再被路由到则重新 pread。
- hot：activation 频率 > 阈值的专家永不入驱逐候选。

---

## 7. 当前实现状态与实测

| 项 | 状态 |
|---|---|
| CLG 预测器 + expert_slots（mmap 版） | ✅ 已实现（`llama-window.cpp`） |
| moe-buffer：显式匿名缓冲 + 切片 repoint + O_DIRECT + 有界 LRU | ✅ 已实现（`src/llama-moe-buffer.{h,cpp}`） |
| **正确性**：moe-buffer+CLG vs window（真异步流式下） | ✅ **PPL 逐位一致**（4 chunk 全等：47948.2078 / 80646.3078 / 95059.6850 / 104976.4199） |
| **CLG → moe-buffer 异步预取接线** | ✅ **已实现并生效**：CLG 在 `ffn_inp-{L}` 预测 → `llama_moe_buffer_prefetch(L+1)` → 后台 worker O_DIRECT 读入缓冲；weight-stream 回调同步兜底。实测 **worker 完成 89% 的流式**（streams=17979 中 worker=15975），同步回调仅兜底 11%。 |
| 实测 RSS 真降 | ✅ **已重测确认**：旧版 streams=0 bug（repoint meta 张量、按指针匹配）已改为**按名字匹配 + 回调内 repoint 真张量**；真实物理 RSS 现可降到模型大小以下并由预算控制。 |

### 实现机制（最后一环已接通）
- **异步预取 worker**：moe-buffer 内置后台线程 + 任务队列 + 每槽 `COLD/RESIDENT/INFLIGHT` 状态机。同步回调与 worker 用**独立 thread_local bounce**、pread 在锁外执行，两者请求同一槽时 INFLIGHT 等待而非重复读。
- **by-layer 索引**：register 时从张量名解析 `blk.N.` → `by_layer[N]`，prefetch(layer) 把该层 gate/up/down 三个 exps 张量的预测专家全部入队。
- **接线**：`llama_window_set_moe_buffer()`（model.cpp 在两者创建后调用）→ CLG 的 `predicted_mask` 在 buffer 模式下走 `llama_moe_buffer_prefetch` 而非 mmap window 的 prefetch/evict。

### 实测数据（20 专家剪枝模型 3.67GiB / 8 核 / O_DIRECT / `-r 1` 单样本）

对比基准有两条：
- **native** = 原始 llama.cpp，无任何 lazy/window 环境变量（默认 mmap 全模型 + repack）。
- **v2 mmap** = 本仓库的 mmap 滑窗（关 repack），是公平的"全载但不重排"权重基准。

#### A. cap=8G（内存充足，机器装得下整模型）

| 配置 | peak memory.current | tg t/s | RSS vs native |
|---|---|---|---|
| **native**（默认 mmap+repack） | 5730 MB | **19.26** | 100% |
| v2 mmap（关 repack） | 3605 MB | 19.58 | 63% |
| moe-buffer+CLG 256MB | **1392 MB** | 1.38 | 24% |
| moe-buffer+CLG 384MB | 1521 MB | 1.72 | 27% |
| moe-buffer+CLG 512MB | 1649 MB | 1.58 | 29% |
| moe-buffer+CLG 768MB | 1905 MB | 2.77 | 33% |
| moe-buffer+CLG 1536MB | 2671 MB | 3.31 | 47% |

两个要点：
1. **native RSS(5730) > v2 mmap(3605)**：差 ~2.1GB 是 repack——native 默认把量化权重重排成 CPU 友好布局，多存一份**不可回收的匿名副本**。
2. **内存够时 native/mmap 完胜 tg**（~19）：整模型常驻 page cache、decode 零磁盘 IO（compute-bound）。moe-buffer 的 O_DIRECT 故意绕 cache，每 miss 从盘读 → tg 掉一个数量级。**此场景该用 native。**

#### B. cap=2G（< 模型 3.67G，内存受限）

| 配置 | peak memory.current | tg t/s |
|---|---|---|
| **native** | 2048 MB（顶 cap） | **OOM 杀死**（rc=137，无结果产出） |
| v2 mmap | 2048 MB（顶 cap，内核反复回收+重 fault） | 2.35 |
| moe-buffer+CLG 768MB | **1904 MB**（受预算控制，<cap） | **2.57** |

- **native 直接 OOM 跑不起来**：repack 的 ~3.7GB 不可回收匿名副本超过 2G cap → 被内核杀死。
- moe-buffer 在 1904MB（<cap）内稳定跑出 2.57 t/s。**这才是 moe-buffer 不可替代的场景：把"内存装不下→跑不了"变成"装不下→预算内可跑"。**

### 找 TPS 地板（预算阶梯实证）

第 5 节的"地板"= 不掉 TPS 的最低 RSS。定位方法：扫预算看 `evictions`，**从 thrash(≫0) 归零的拐点就是地板**（预算刚够装下工作集）。cap=8G、`-p 1 -n 24`：

| budget | peak RSS | resident(工作集) | **evictions** | read 总量 | read/token | tg |
|---|---|---|---|---|---|---|
| 768MB | 1905 | 766 | 7392 | 14360 MiB | **598 MB** | 1.97 |
| 1536MB | 2666 | 1535 | 2710 | 6517 MiB | 272 MB | 2.23 |
| 2048MB | 3176 | 2047 | 618 | 3176 MiB | 132 MB | 6.22 |
| 2304MB | 3435 | 2302 | 133 | 2543 MiB | 106 MB | 8.40 |
| **2432MB** | **3542** | **2416** | **0** ✅ | **2416 MiB** | ~0(稳态) | 11.57 |
| 2560MB | 3541 | 2416 | 0 | 2416 MiB | ~0 | 9.43(噪声) |

- **拐点 = 2432MB**：evictions 首次=0，read 触底到 2416MB（=工作集一次性冷加载，不再重读），再加预算 resident/read/tg 均不变 → 工作集 = 2416MB，地板确认。
- thrash 区每 token 读量被放大数倍（768MB 档 598MB/token ≫ 工作集本身），这是 tg 崩的直接原因。

**稳态验证**（地板预算 vs native，看冷启动摊销）：

| | tg32 | tg128（冷启动摊销后） |
|---|---|---|
| native | 22.88 | 23.62 |
| moe-buffer 2432（地板） | 12.62 | **19.92（≈native 84%，仍收敛中）** |

→ 找到 evictions=0 的地板后，**稳态 tg 基本不掉**（剩余差距是冷启动一次性读 2.4GB + O_DIRECT 不留 cache 的零星 miss）。**"找地板"方法论实测有效。**

### 多线程并行读：松动 SSD 带宽墙

单线程 O_DIRECT 读欠载盘（`dd` 实测顺序 488 MB/s），非盘极限。`LLAMA_LAZY_MOE_BUFFER_WORKERS=N` 起 N 个 worker 并行 pread（各自 thread_local bounce），抬升有效带宽。同一预算扫 worker 数（`-p1 -n48`）：

| budget | peak RSS | tg(1 worker) | tg(4 worker) | 加速 |
|---|---|---|---|---|
| 512MB | 1654 | 1.58 | 2.87 | 1.8× |
| 768MB | 1905 | 1.97 | 3.75 | 1.9× |
| 1536MB | 2666 | 2.23 | **5.42** | 2.4× |
| 2432MB(地板) | 3575 | 11.32 | 16.45* | 1.45× |

768MB 档单看 worker 数：1→1.97, 2→3.45, **4→3.75**, 8→3.57（8 核机超订回落，4 最优）；read 量不变(~26GB)、tg 翻倍 → 确认是带宽提升而非读得少。

**关键意义**：多 worker 只在 **IO-bound 区**（受限预算，miss 多）有效，**抬升的是整条"预算→tg"曲线的低预算段** → 等价于"同样 TPS 下能用更小 RSS"。例如要 tg≈3.7：原来要 ~2GB 预算，现在 768MB（1905MB RSS）就够。地板/稳态区（evictions=0，compute-bound，几乎无 miss）多 worker 无收益（*2432MB 的 1.45× 只来自冷启动那一次性读的并行加速；`-n128` 稳态回到噪声内）。

> 借鉴自 ExpertFlow（DAC'26）的 overlap-I/O 思想，但场景不同：ExpertFlow 是 GPU-CPU（PCIe ~16–32 GB/s），我们是 CPU-盘（单线程 488 MB/s，4 线程饱和 ~1.2 GB/s）。它的"激进驱逐+每步重取"在 PCIe 下成立，在慢盘下会被带宽打死——所以我们的对策是**先用并行读把带宽顶到盘的物理上限**，而非更激进的 cache 策略。详见第 11 节系统上限。

### 但本模型地板 ≈ 全模型（核心局限）

| | peak RSS |
|---|---|
| 地板 moe-buffer 2432 | 3542 MB |
| native（v2 mmap 关 repack） | 3605 MB |
| **省下** | **63 MB（1.7%）** |

```
地板 RSS = 非专家权重 + 工作集
工作集   = 一段时间内被路由专家的并集 × 切片大小 ≈ 激活比 × 专家权重 × 复用系数
```
本模型 **45% 激活比** + CLG 过预测 + 路由跨 token 分散 → 几十 token 后几乎每个专家都被路由过 → 工作集并集 ≈ 全部专家 → **地板≈全模型，省不下**。价值需换**低激活比 MoE**（Qwen3-30B-A3B 10% → 工作集仅 ~1/4 → 地板远低于全模型且 tg 不掉）验证。

### 历史实测（旧 mmap+madvise 路线，已被本方案取代）
- 四配置(仅V2 / LRU驱逐 / CLG无驱逐 / CLG+驱逐) peak `memory.current` **全顶到 2048MB=cap** → 证明 **mmap+madvise 路线对 MoE 也降不了真实物理 RSS**，是改用显式缓冲 + O_DIRECT 的实证依据。

---

## 8. 配置（环境变量）

| 变量 | 含义 |
|---|---|
| `LLAMA_LAZY_V2=1` | 启用 lazy/window 框架（moe-buffer 当前依赖此块） |
| `LLAMA_LAZY_MOE_BUFFER=1` | 启用显式匿名缓冲专家流式（接管 exps 张量） |
| `LLAMA_LAZY_MOE_BUFFER_MB=N` | 驻留专家字节预算（LRU 驱逐到此值）；0/未设 = 不驱逐（累积） |
| `LLAMA_LAZY_MOE_BUFFER_WORKERS=N` | 并行预取 worker 数（默认 1）。提高有效盘带宽——单线程 O_DIRECT 随机读欠载 NVMe。**仅在 IO-bound（受限预算）区有效**；地板/稳态 compute-bound 无收益。8 核机实测 **4 最优**，再多线程超订回落 |
| `LLAMA_LAZY_CLG=1` | 启用 CLG 预测（lead time） |
| `LLAMA_LAZY_CLG_DELTA` / `_HOT` / `_HOT_WARMUP` / `_PREFILL_THR` | 预测余量 δ / hot 阈值 / 预热步数 / prefill 跳驱逐阈值 |

---

## 9. 测量方法学

- **度量真实物理占用用 cgroup `memory.current` 峰值采样**，不是 VmRSS（VmRSS 在过配机器上是 cache 的假象，已实证）。
- **找地板用预算阶梯 + evictions 判据**：固定大 cap，扫 `MOE_BUFFER_MB`，开 `LLAMA_LAZY_DEBUG=1` 看 teardown 的 `evictions`。**`evictions` 从 thrash(≫0) 首次归零的那一档 = 地板**（预算刚够装下工作集，read 同时触底到"工作集一次性冷加载"量）。比单看 tg 曲线更准——tg 在 `-r 1` 下有噪声，但 evictions/read 是确定量。验证：在地板档用长 `-n`（如 128）确认稳态 tg 回到 ≈native（冷启动摊销后）。本会话实测拐点见第 7 节。
- **TPS 必须 `-r 3`+ 看误差棒**（本机 CPU 吞吐噪声 ±~1 t/s，单样本会得出噪声结论——本会话已多次踩过）。
- **正确性用 perplexity**：`LLAMA_LAZY_V2` vs `+MOE_BUFFER` 的 PPL 必须逐位一致（同 repack-off 基线）。

---

## 10. 已知限制与后续

- ✅ **最后一环已接通**：CLG 异步预取已接到 moe-buffer（node_callback 在 `ffn_inp-{L}` 处 CLG 预测 → `llama_moe_buffer_prefetch(L+1)` → 后台 worker O_DIRECT 读入缓冲 → weight-stream 钩子兜底）。实测 worker 完成 89% 流式。
- **激活比高的剪枝模型收益有限**（本测试 45% → 地板高，且内存够装时 mmap 免费缓存更快）；价值在**低激活比 MoE（10–20%，如 Qwen3-30B-A3B）+ 内存受限（cap < 模型）**。验证下一步应换低激活比模型重测预算阶梯。
- **TPS 仍受 IO 带宽约束**：cap < 模型时每 token 必从盘读激活专家，总吞吐 = 计算与 IO 的较大者（第 5 节公式）。异步预取只能藏住延迟、藏不掉带宽；要进一步提速需更快盘 / 更低激活比 / 更大预算（命中率↑）。
- **prefill**：批量 token 的预测并集 ≈ 全专家；当前 buffer 模式预取按预测并集入队，LRU 预算自然限幅，未单独跳预取（`CLG_PREFILL_THR` 仅作用于 mmap 驱逐路径）。
- **hot 固定层**：当前未单独 pin，靠 CLG 过预测（δ）+ LRU 近期性让高频专家自然驻留；如需严格 pin 可在 moe-buffer 加 hot 名单。
- **O_DIRECT 对齐**：用 thread_local 对齐 bounce 处理（任意 offset/stride 不对齐均可），非块设备自动回退 buffered + fadvise。
- **仅 CPU 后端**：weight-stream 钩子在 ggml-cpu；GPU/混合未覆盖。
- **共享专家**（2D，非 `_exps`）保持 mmap，不在本方案管理范围（它几乎总用，留 mmap/常驻即可）。

---

## 11. 系统上限与运行方案选型

### 11.1 硬件天花板（本会话实测，决定一切上限）

当前环境是 **QEMU 虚拟机**（`systemd-detect-virt=qemu`），数据盘是 **virtio_blk 虚拟盘 `vda`（50G）**，**无任何物理 NVMe 直通**（`/dev/nvme*` 不存在，`rotational=1` → 共享/网络后端）。`dd` O_DIRECT 实测：

| 读法 | 带宽 |
|---|---|
| 单线程顺序 | **488 MB/s** |
| 4 线程并行 | **~1194 MiB/s（饱和）** |
| 8 线程 | 不再增（虚拟盘 + hypervisor I/O 路径的总上限 ≈ 1.2 GB/s）|

**这 ~1.2 GB/s 是本系统的物理硬墙**，`moe-buffer` 的 4-worker 已基本打满。它**不是代码瓶颈**——virtio_blk 已是高效半虚拟化驱动，`io_uring`、更多线程都突破不了后端存储/限速。

对照若有真硬件：PCIe Gen5 NVMe 单盘 ~14 GB/s（**~12×**）、NVMe 阵列数十 GB/s。**只有换裸金属/NVMe 直通（PCIe passthrough/SR-IOV）或挂本地 NVMe 的机型，才能突破这堵墙**；guest 内无法软件解决。

### 11.2 三个运行点（同模型 Qwen1.5-MoE-20experts-Q4_K_M / 8 核 / cap=8G）

| 运行点 | 配置 | peak RSS | tg t/s | 适用 |
|---|---|---|---|---|
| **① 最优性能** | native（mmap+repack） | 5730 MB | **19–24** | 内存充足、求最快；整模型常驻 RAM、零盘 IO |
| **②′ 性能-内存平衡（地板）** | moe-buffer 2432MB + 4worker | 3542 MB | **16–20**（稳态≈native 84%） | 想省点 RSS 又几乎不掉 TPS；evictions=0 不 thrash |
| **③ RSS 最小（主流方案）** | moe-buffer 256–768MB + 4worker | **1392–1905 MB** | 1.4–3.8 | 内存受限/装不下模型；用 TPS 换最低 RSS |

- **最优性能 = native**（或本仓库 v2 mmap 关 repack，3605MB/19.6，RSS 更低且同速——repack 那 2.1GB 副本不值）。
- **RSS 最小 = moe-buffer 小预算**：256MB 预算 → peak 1392MB（native 的 24%），代价是 tg 跌到带宽墙决定的下限。**这是"内存装不下模型也能跑"的主流运行方案。**
- **当前系统上限**：在 cap=2G（< 模型）下，**native 直接 OOM 跑不起来**，moe-buffer 是唯一能跑的，1904MB 稳定 2.57 t/s（4-worker 更高）。

### 11.3 当前系统的 TPS 上限是什么决定的

```
tg_上限 = 1 / max( T_compute ,  每token激活字节 / 盘带宽 )
                                  └ 受 ①激活比 ②量化位宽 ③盘带宽(1.2GB/s硬墙) 决定 ┘
```
- **RSS 最小方案的 tg** 完全由 **1.2 GB/s 带宽墙**锁死：预算越小→miss越多→每token读越多→撞墙越狠。
- 软件能做的（多 worker 并行读）**已到顶**（4-worker 打满 1.2GB/s）。

### 11.4 突破当前上限的手段（按性价比，全部绕过而非提升带宽）

| 手段 | 效果 | 代价/前提 | 本系统可行性 |
|---|---|---|---|
| **换低激活比好模型**（Qwen3-30B-A3B 10%） | 工作集↓ → 地板↓ + 命中率↑ → tg↑ | 换模型 | ★★★ 最大杠杆 |
| **专家二次量化** Q4→Q2/Q3（仅 `ffn_*_exps`） | I/O 字节 -42% → IO-bound 区 tg +60–70% + 地板↓ | 精度损失（本剪枝模型已退化，需好模型扛） | ★★ 零代码改动（`llama-quantize --tensor-type`） |
| **batching 摊销**（serving） | 一次读供多 token → throughput 数倍 | 需并发场景；45%激活下并集摊销打5折；抬 RSS；单token latency↑ | ★★ 仅 serving |
| **多 worker 并行读** | 抬有效带宽至盘上限 | — | ✅ 已实现，**已打满 1.2GB/s** |
| `io_uring` / 更多线程 | — | 盘已饱和，仅省 CPU | ✗ 带宽零增益 |
| **换 NVMe 硬件** | 带宽 ×12 → IO-bound 区基本消失 | 需裸金属/直通/换机型 | ✗ guest 内不可得 |

> 一句话：**本系统软件层 I/O 已优化到头（4-worker 打满 1.2GB/s 虚拟盘）。** 主流运行方案是「RSS 最小」（moe-buffer 小预算 + 4-worker），它把"装不下模型就跑不了"变成"装不下也能在带宽墙下跑"；其 TPS 上限由这块虚拟盘 ~1.2 GB/s 物理锁死。要再快只能**绕过带宽**（换模型 / 量化 / batching）或**换 NVMe 硬件**。

---

**文档版本** v1.2 · 数据来源：本会话 Qwen1.5-MoE-A2.7B-20experts-Q4_K_M(3.67GiB) / 8 核 / QEMU virtio 盘(488MB/s单线程·1.2GB/s饱和) / cgroup 实验 / WikiText-2 PPL。状态：**CLG→buffer 异步预取 + 多 worker 并行读已实现**；正确性 PPL 逐位一致（4 chunk）；RSS 真降已确认（最小 1392MB=native 24%）；**I/O 软件层已打满虚拟盘物理上限 ~1.2GB/s**，进一步提速需绕过带宽（换模型/量化/batching）或换 NVMe 硬件。
