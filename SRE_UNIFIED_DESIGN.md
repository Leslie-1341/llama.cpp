# SRE 统一流式驻留引擎设计文档
### Streaming Residency Engine — 统一 dense / MoE 的"匿名缓冲 + 真驱逐"权重换入换出

**目标**：把当前**两套独立**的权重流式实现——`llama-flex`（dense 层粒度 ring 流式）与 `llama-moe-buffer`（MoE 专家粒度 LRU 流式）——统一为**一个引擎**：**回收/驱逐机制共用**（匿名缓冲 + O_DIRECT + 真驱逐 + 多 worker），**预取策略分离可插拔**（dense 顺序滑窗 / MoE CLG 预测）。

**一句话总纲**：驱逐对物理内存是否生效，取决于 backing 是 **mmap（假降）** 还是**匿名缓冲（真降）**，不取决于驱逐算法。统一 = 抽出共用的「匿名缓冲真驱逐 Residency Manager」，让 dense 与 MoE 作为两个 backing/policy 组合接上去。

> 配套文档：
> - [`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md)：**对原生 llama.cpp 的全部更改 + 问题/解决方案 + 算法及实现代码**。
> - [`TEST_RESULTS.md`](TEST_RESULTS.md)：**全部详细测试结果**（本文第 6 节为摘要，权威数据以该文件为准）。
> - [`WINDOW_DESIGN.md`](WINDOW_DESIGN.md)（dense mmap 滑窗）、[`FLEX_INFER_DESIGN.md`](FLEX_INFER_DESIGN.md)（dense 显式缓冲流式）、[`MOE_OFFLOAD_DESIGN.md`](MOE_OFFLOAD_DESIGN.md)（MoE 专家流式）。

---

## 1. 背景：两条独立路线，回收本质相同

| | `llama-flex`（dense） | `llama-moe-buffer`（MoE） |
|---|---|---|
| 调度单元 | 整层（layer） | 单专家切片（expert slice） |
| backing | 匿名 **ring**（k 层槽循环） | 匿名 **full-size** per `*_exps`（按 `id*stride` 索引） |
| 预取 | **顺序** L+ahead（确定） | **CLG 预测** L+1 激活专家（输入相关） |
| 驱逐 | 用完即释放 slot（FIFO/ring） | 超预算 **LRU** + `MADV_DONTNEED` |
| 读取 | O_DIRECT 大块顺序 | O_DIRECT 切片 + thread_local bounce |
| pin | balanced-lock 高频层 | hot 高频专家 |
| 现状 | 已实现（[llama-flex.h](src/llama-flex.h)） | 已实现 + 实测（[llama-moe-buffer.h](src/llama-moe-buffer.h)） |

两者**回收机制本质相同**：匿名 buffer 流入 → 用完 → `MADV_DONTNEED` 真释放。差异只在 **slot 寻址**（ring vs id 索引）、**驱逐结构**（FIFO vs LRU）、**预取来源**（顺序 vs 预测）。当前是两套代码、各写一遍 O_DIRECT / worker / 状态机，且在 [llama-context.cpp](src/llama-context.cpp) 里**互斥二选一**。统一可消除重复。

---

## 2. 核心根因：为什么 mmap 驱逐"假降"、匿名缓冲"真降"

这是整个设计的出发点，已被本会话实测确立（见第 6 节）：

| backing | 驱逐动作（`MADV_DONTNEED`） | cgroup `memory.current`（真实物理） |
|---|---|---|
| **mmap 文件页** | 只摘进程 PTE、动 VmRSS | 内核**立刻 page cache 回填** → **顶满 cap（假降）** |
| **匿名缓冲** | 真释放物理页、归还系统 | **压到预算（真降）** |

**推论（关键）**：**不能"只改驱逐算法"让 dense mmap 真降**——无论驱逐多激进，mmap 文件页的 cache 会用满到 cap。要真降必须把 backing 从 mmap 换成匿名缓冲。**所以"让 dense 驱逐对物理生效" ⟺ "dense 走匿名缓冲流式" ⟺ flex 路线。** 统一驱逐必然意味着 dense 放弃 mmap。

---

## 3. 三种内存模式（贯穿全文的对照维度）

任何模型都有三个运行点，对应"是否减 repack / 是否开驱逐"：

| 模式 | backing | repack | 驱逐 | 运行时物理 RSS | TPS | 何时用 |
|---|---|---|---|---|---|---|
| **① native** | mmap | **开** | 无 | **全模型 + repack 副本**（最高）| 满速 | 内存充足、求最快 |
| **② 不开驱逐（关 repack）** | mmap | **关** | 无 | **≈ 模型大小**（去掉 repack 副本）| 满速（不掉）| 内存够装下模型；**免费降一截** |
| **③ 开驱逐（SRE）** | **匿名缓冲** | 关 | **真驱逐** | **预算可控（< 模型）** | 受带宽墙 | 内存装不下模型 |

- **① → ②**：纯靠**减 repack**，去掉那份不可回收的 CPU 友好布局副本，**不掉 TPS**（消除的是冗余副本不是工作集）。这是"不开驱逐就能降物理内存"的部分。
- **② → ③**：靠**匿名缓冲 + 真驱逐**把 RSS 压到模型以下，代价是每次 miss 从盘读 → 受 I/O 带宽锁死。
- dense 与 MoE 在 ③ 的差别 = **稀疏红利**：MoE 每 token 只读激活专家，dense 每 token 重读全模型层。

---

## 4. 统一架构：Streaming Residency Engine (SRE)

```
┌─────────────────────────────────────────────────────────────┐
│ 层3  Prefetch Policy（预取，可插拔，dense/MoE 不同）            │
│   - DensePolicy:  node=l_out-{L}   → 预取 group L+1..L+ahead   │ 顺序确定
│   - MoEPolicy:    node=ffn_inp-{L} → CLG 预测 L+1 专家          │ 输入相关
│        ↓ 都产出 prefetch(slice_ids[]) 入同一队列                │
├─────────────────────────────────────────────────────────────┤
│ 层2  Residency Manager（回收，dense/MoE 完全共用）  ← 统一点    │
│   匿名 buffer + COLD/RESIDENT/INFLIGHT 状态机                  │
│   + O_DIRECT pread(thread_local bounce) + N worker 并行        │
│   + 真驱逐(MADV_DONTNEED 匿名页) + pin/lock(hot 常驻)           │
│   驱逐结构(可选): ┌ ring/FIFO     dense 顺序访问，k 槽循环       │
│                  └ indexed-LRU   MoE 随机 id 索引，full-size   │
├─────────────────────────────────────────────────────────────┤
│ 层1  Slice Registry（注册）                                    │
│   slice = {name, file_idx, file_off, size/stride, group=layer}│
│   dense slice = 整层张量 ; MoE slice = 单专家切片               │
└─────────────────────────────────────────────────────────────┘
统一钩子:
  node_callback     → Prefetch Policy（驱动异步预取，拿 lead time）
  weight-stream cb  → ResidencyMgr.ensure_resident(op的slice) + repoint ->data
                      （同步兜底，唯一能保证 kernel 读前已就位的点）
```

### 4.1 关键设计决策

| 决策点 | dense | MoE | 统一处理 |
|---|---|---|---|
| **backing** | 匿名 ring（k 层槽） | 匿名 full-size per exps | ResidencyMgr 抽象 `slot_addr(slice)`：dense 返回 ring 槽、MoE 返回 `buf+id*stride` |
| **slice 粒度** | 整层 | 单专家切片 | `group_id=layer`；一个 group 含 1..N slice |
| **为何布局不同** | 顺序访问 → k 槽循环够 | `mul_mat_id` 按 `id*stride` 索引 → 必须全尺寸 | 由 backing policy 决定，对上层透明 |
| **驱逐** | 用完即释放（ring/FIFO，零 LRU 开销） | 超预算 LRU | 统一 `evict()` 接口，策略可换 |
| **预取** | 顺序 L+ahead | CLG 预测 | 统一 `prefetch(slice_ids)` 入同一 worker 队列 |
| **同步兜底** | 当前层张量必驻留 | mul_mat_id 选中专家必驻留 | 同一 weight-stream 回调，INFLIGHT 则等 cv |
| **pin/hot** | balanced-locked 高频层 | hot 高频专家 | 统一 `pin(slice)` 永不驱逐 |
| **多 worker** | io_threads | MOE_BUFFER_WORKERS | 共用 worker 池（实测 4 打满本系统盘）|

### 4.2 正确性根基（两个钩子的分工，dense/MoE 一致）

- **node_callback**（节点完成后，异步）：驱动 Prefetch Policy → 异步预取，拿 lead time，负责"通常已命中"。
- **weight-stream callback**（kernel 计算前，ith0 + barrier）：同步兜底，确保该 op 用到的 slice 全部 RESIDENT 再放行，负责"万一漏了不出错"。**这是唯一能保证 kernel 读前 slice 已就位的同步点**（node_callback 在节点之后、其他线程不等它）。

### 4.3 自适应驱逐策略（按可用内存自动定档）

**目标**：在可运行前提下尽量少驱逐 → 尽量少 IO → 尽量不掉 TPS。启动时检测一次可用内存，自动定档（运行中内存不剧变，无需实时调）：

```
available = min(cgroup memory.max − 基线 ,  /proc MemAvailable)      # 别用 MemFree
overhead  = 非专家/非层权重(mmap 常驻) + compute buffer + KV + 余量   # 不可驱逐部分 = OOM 下界

if available ≥ 模型(no-repack):        # 装得下
    模式②  只关 repack，不驱逐         # 省 ~33%，满速，免费
else:                                  # 装不下
    budget = clamp(available − overhead,  最小流水线,  专家总字节)
    模式③  驱逐量 = 模型 − budget，budget 自动贴可用内存上限（尽量少驱逐）
```

**MoE 与 dense 的关键差异（必须分开对待）**：

| | MoE | dense |
|---|---|---|
| budget↔IO 关系 | **连续可调**：budget 越大 → 越接近"工作集装得下" → evictions↓ → IO↓ → tg↑ | **无连续可调**：每 token 必过全部层，ring<全模型时每 token IO = 全模型（常数）|
| 自适应价值 | ✅ 高：budget=available−overhead 自动找最优；热专家 pin 进一步缩小有效工作集 | ⚠️ 退化为二态：装得下满速 / 装不下 tg 锁死 ~0.16（budget 多大都一样）|
| 实测佐证 | budget 768→2432MB：evictions 7392→0，tg 1.97→11.57 | cap 3072→128M：tg 恒 0.16，IO 不变 |

→ **自适应的真正连续价值只在 MoE**；dense 只做"够就不驱逐、不够就用最小 ring 保证能跑"的二选一。

**热专家保护（MoE，进一步降 IO）— 相对频率判据（已实现）**：一个专家是否 hot 用**相对判据**——`activation[e] > hot_ratio × (本张量平均激活 = total_act / n_expert)`，即"被路由频率高于本张量平均的 hot_ratio 倍"才 pin（驱逐时跳过）。配 warmup（每专家平均被看过 ~2 次才启用）。退化保证：`hot_ratio=0` 纯 LRU。

> **为何相对而非绝对**：绝对累计阈值（旧 `HOT_N`）只增不减，长序列下每个专家迟早都超阈值 → 全 pin → 退化成不驱逐、pin 集合无界（旧实测 resident 510→**1077** MiB，涨过 budget）。相对判据分子分母同步增长，比例稳定 → **pin 集合天然有界**，不会随序列变长而全 pin。

**实测（budget=512MB thrash，`HOT_RATIO=0` vs `2.0`，`-n32`）**：streams 17601→15336(-13%)、evictions 17322→15058(-13%)、read 32.3→28.1 GiB(-13%)、tg 3.06→3.21；**`resident=510.6→511.2 MiB（不超 budget！）`**——相对判据既减 IO 又把 pin 集合钉在预算内，解决了绝对阈值涨过 budget 的问题。

---

## 5. 迁移路径（从现有两套抽取）

```
现状（两套独立，O_DIRECT/worker/状态机各写一遍，context.cpp 互斥二选一）:
  llama-flex       = DensePolicy + ring backing
  llama-moe-buffer = MoEPolicy   + indexed-LRU backing

统一:
  llama-sre = 共用 ResidencyManager(worker池 / O_DIRECT / 状态机 / 真驱逐 / pin)
            + 可插拔 backing(ring | indexed-LRU)
            + 可插拔 PrefetchPolicy(Dense | MoE)
  flex / moe-buffer → 退化为 SRE 的两个配置组合
```

渐进步骤：
1. 把 moe-buffer 已验证的 ResidencyManager（COLD/RESIDENT/INFLIGHT 状态机 + LRU + worker 池 + O_DIRECT bounce + 真驱逐）抽成独立类。
2. 抽象 slot 寻址 `slot_addr(slice)`：MoE = `buf+id*stride`，dense = ring 槽。
3. flex 改为复用 ResidencyManager + ring 分配器 + DensePolicy。
4. context.cpp 从"二选一"改为"按模型类型挂对应 Policy"，引擎唯一。

**风险点**：ring（k 槽循环、顺序覆盖）与 full-size（id 索引、LRU）是两种寻址，需在 ResidencyManager 内抽象干净，避免 dense 退化成"每层一个 full-size buffer"（浪费）。

---

## 6. 全部测试数据

> **本节为摘要，完整数据（含命令、环境、所有阶梯）见 [`TEST_RESULTS.md`](TEST_RESULTS.md)。**

环境：**QEMU 虚拟机**，virtio_blk 虚拟盘 `vda`（**无 NVMe 直通**），8 核。盘带宽 `dd` O_DIRECT 实测：**单线程 488 MB/s，4 线程饱和 ~1194 MiB/s**（虚拟盘总上限，4-worker 打满）。RSS 一律用 cgroup `memory.current` 峰值（真实物理），非 VmRSS。

### 6.1 Dense — Llama-3-8B-Instruct-Q4_K_M（4.9 GB），三模式对照

**模式①/② — cap=8G 不约束（native vs 不开驱逐）+ 验证 mmap 驱逐假降**

| 配置 | 模式 | cgroup 真实 RSS | tg | 说明 |
|---|---|---|---|---|
| native（mmap+repack） | ① | **7770 MB** | 8.15 | 含 repack ~2GB 不可回收副本 + 全模型 cache |
| window mmap（关 repack，无驱逐） | ② | **4560 MB** | 8.60 | **-3.2GB 全来自关 repack，不掉 TPS** |
| window mmap + **dontneed**(limit 2G) | — | **4560 MB** | 8.93 | **和不开驱逐一模一样 → mmap 驱逐对物理零效果（假降）** |

→ ② 相比 ① 降的 3.2GB **全是关 repack**；mmap 开 dontneed 对 cgroup 真实物理**零效果**（4560==4560），证实第 2 节根因。

**模式③ — 开驱逐（window=4 + dontneed，cgroup cap 制造压力，`-n4`）**

| cap | cgroup 真实 RSS | tg | 状态 |
|---|---|---|---|
| native cap=3G | — | — | **OOM 杀死**（repack 副本超 cap）|
| 3072M | 3072 | 0.16 | 顶满 cap |
| 2048M | 2048 | 0.16 | 顶满 cap |
| 1024M | 1024 | 0.18 | 顶满 cap |
| 512M | 512 | 0.17 | 顶满 cap |
| 256M | 256 | 0.16 | 顶满 cap |
| **128M** | **128** | **0.17** | ✅ **最低可跑点** |
| 96M | — | — | OOM（连单层+compute 瞬时峰值都放不下）|

**模式③′ — 匿名 backing（flex，已落地）：dense 驱逐真对物理生效**

mmap window 的驱逐是假降（上表 4560==4560）。换成 flex 的匿名 ring backing 后，cgroup 真降：

| 配置 | cgroup 真实 RSS | tg | 说明 |
|---|---|---|---|
| mmap window 关 repack（对照）| 4560 MB（顶满，假降）| 8.60 | madvise 对物理无效 |
| flex ring=4，cap=8G | **1252 MB** | 1.52 | 匿名 ring 真降（非层权重+4层+KV+compute）|
| **flex AUTO**，cap=8G（充足）| 4277 MB | 7.45 | ring 自动定满 ≈ 不流式，速度近 native |
| **flex AUTO**，cap=2560M（<模型）| **1472 MB（<cap，真降）** | 1.77 | ring 自动缩，**不 OOM**（mmap 会 OOM/顶 cap）|

→ **dense 匿名 backing 真生效**：cgroup 压到 cap 以下且可控；`FLEX_AUTO` 按可用内存自动定 ring（充足全留满速 / 受限缩 ring 真降）。这就是"把 window backing 从 mmap 换成匿名缓冲"的落地（复用 flex，不在 window 内重写）。代价仍是 dense 无稀疏 → 受限下 tg 受带宽墙（1.5–1.8）。

**Dense 三结论**：
1. **mmap → cgroup 永远顶满 cap**，window_size 调不动真实 RSS；真正的旋钮是 cap 本身。**换 flex 匿名 backing 后驱逐才真生效**（1472<cap，对照 mmap 4560 顶满）。
2. **带宽墙下 RSS 与 tg 解耦**：tg 从 3G 到 128M 始终 ~0.16（每 token 重读全模型 / 带宽，早已触底），再压 RSS 不影响 tg。
3. **RSS 硬下界 = 128 MB（native 的 1/60）**，4.9GB 模型可在 128MB 物理内存跑，但 tg 0.17（≈6 秒/token，能跑不能用）——**dense 无稀疏，省 RSS 与保 TPS 彻底互斥**。

### 6.2 MoE — Qwen1.5-MoE-A2.7B-20experts-Q4_K_M（3.67 GB），三模式对照

**模式①/②/③ — cap=8G 不约束**

| 配置 | 模式 | cgroup 真实 RSS | tg | RSS vs native |
|---|---|---|---|---|
| native（mmap+repack） | ① | 5730 MB | 19.26 | 100% |
| v2 mmap（关 repack，无驱逐） | ② | 3605 MB | 19.58 | 63%（-2.1GB 全是 repack，不掉 TPS）|
| moe-buffer+CLG 1536MB | ③ | 2671 | 3.31 | 47% |
| moe-buffer+CLG 768MB | ③ | 1905 | 2.77 | 33% |
| moe-buffer+CLG 512MB | ③ | 1649 | 1.58 | 29% |
| moe-buffer+CLG 384MB | ③ | 1521 | 1.72 | 27% |
| moe-buffer+CLG 256MB | ③ | **1392** | 1.38 | **24%** |

**模式③ 找 TPS 地板（预算阶梯，`evictions→0` 拐点，`-p1 -n24`）**

| budget | 真实 RSS | 工作集 | evictions | read/token | tg |
|---|---|---|---|---|---|
| 768MB | 1905 | 766 | 7392 | 598 MB | 1.97 |
| 1536MB | 2666 | 1535 | 2710 | 272 MB | 2.23 |
| 2048MB | 3176 | 2047 | 618 | 132 MB | 6.22 |
| 2304MB | 3435 | 2302 | 133 | 106 MB | 8.40 |
| **2432MB** | **3542** | **2416** | **0** ✅ | ~0 | 11.57 |
| 2560MB | 3541 | 2416 | 0 | ~0 | 9.43(噪声) |

→ 地板 = evictions 首次归零的 2432MB；thrash 区每 token 读量被放大数倍（768MB 档 598MB/token ≫ 工作集），是 tg 崩的直接原因。

**模式③ 稳态验证（地板 vs native，看冷启动摊销）**

| | tg32 | tg128 |
|---|---|---|
| native | 22.88 | 23.62 |
| moe-buffer 2432（地板，1 worker） | 12.62 | 19.92（≈native 84%）|

**模式③ 多 worker 并行读（松动带宽墙，`-p1 -n48`）**

| budget | tg(1 worker) | tg(4 worker) | 加速 |
|---|---|---|---|
| 512MB | 1.58 | 2.87 | 1.8× |
| 768MB | 1.97 | 3.75 | 1.9× |
| 1536MB | 2.23 | **5.42** | 2.4× |
| 2432MB(地板) | 11.32 | 16.45 | 1.45× |

768MB 单看 worker 数：1→1.97, 2→3.45, **4→3.75**, 8→3.57（8 核超订回落，4 最优）；read 不变、tg 翻倍 → 带宽提升而非读得少。多 worker 只在 IO-bound 区有效。

**模式③ δ 旋钮（CLG 过预测余量，累积无驱逐，`-n64`）**

| δ | 工作集(RSS) | tg | 说明 |
|---|---|---|---|
| 0 | 2411 MB | 12.52 | δ↓ 省 RSS 但 tg 也↓（预取覆盖不足、同步 stall 增多）|
| 2 | 2487 | 13.60 | |
| 8 | 2634 | 17.27 | δ↑ tg↑ RSS↑；δ 实为 TPS 旋钮非 RSS 旋钮 |

**MoE 受限场景 — cap=2G（< 模型）**

| 配置 | 真实 RSS | tg |
|---|---|---|
| native | — | **OOM 杀死**（repack 副本超 cap）|
| v2 mmap | 2048（顶 cap，内核反复回收+重 fault）| 2.35 |
| moe-buffer+CLG 768 | **1904（<cap，受预算控制）** | **2.57** |

### 6.3 正确性

moe-buffer+CLG vs window：**PPL 4 chunk 逐位一致**（47948.2078 / 80646.3078 / 95059.6850 / 104976.4199）——真异步 O_DIRECT 流式下 bit-exact。

---

## 7. 三模式横向总结（native vs 不开驱逐 vs 开驱逐）

| 维度 | ① native | ② 不开驱逐（关 repack） | ③ 开驱逐（SRE） |
|---|---|---|---|
| **dense RSS** | 7770 MB | 4560 MB（-repack） | **128 MB–cap（真降可控）** |
| **dense tg** | 8.15 | 8.60（不掉） | 0.16（带宽墙，无稀疏）|
| **MoE RSS** | 5730 MB | 3605 MB（-repack） | **1392–2432 MB（24–62%）** |
| **MoE tg** | 19.26 | 19.58（不掉） | 1.4–16（预算/worker 可调）|
| **cap<模型** | **OOM** | **OOM**（repack 副本超 cap）| ✅ **唯一能跑** |
| 降 RSS 来源 | — | 去 repack 冗余副本（免费）| 匿名缓冲 + 真驱逐（付带宽）|

**核心结论**：
1. **不开驱逐（关 repack）是免费午餐**：dense -3.2GB / MoE -2.1GB，TPS 不掉——任何场景都该先开。
2. **开驱逐才能降到模型以下**，但驱逐对物理生效**必须匿名 backing**（mmap 是假降，dense 实测 4560==4560 验证）。
3. **统一引擎 SRE** 把 dense（ring+顺序）与 MoE（full-size+CLG）的回收合一，回收统一、预取分离。
4. **带宽墙是本系统硬约束**（虚拟盘 ~1.2GB/s，4-worker 打满）：dense 因无稀疏被锁死在 ~0.16 t/s（能跑不能用）；MoE 靠稀疏红利可调到 1.4–16 t/s。要再快只能绕过带宽（换低激活比模型 / 专家量化 / batching）或换 NVMe 硬件。

---

## 8. 配置（统一后设想，保持向后兼容）

| 变量 | 含义 |
|---|---|
| `LLAMA_SRE=1` | 启用统一引擎（自动按模型类型选 dense/MoE policy）|
| `LLAMA_SRE_BUDGET_MB=N` | 驻留预算（真驱逐到此值）；dense 折算为 ring 层数 |
| `LLAMA_SRE_WORKERS=N` | 并行 O_DIRECT worker（本系统实测 4 打满盘）|
| `LLAMA_SRE_PIN_MB=N` | pin/hot 常驻预算（dense balanced-lock / MoE hot 专家）|
| `LLAMA_SRE_CLG=1` | MoE 启用 CLG 预测预取（dense 用顺序预取，无需）|
| `LLAMA_SRE_CLG_DELTA` | CLG 过预测余量 δ（TPS/RSS 旋钮）|
| `LLAMA_LAZY_MOE_BUFFER_AUTO=1` | **MoE 自适应 budget**：按可用内存自动定（available−overhead），装得下则不驱逐 |
| `LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=R` | **热专家保护（相对）**：激活率 > R× 本张量平均的专家不被驱逐（0=关，纯 LRU；典型 2.0）|
| `LLAMA_FLEX=1` / `LLAMA_FLEX_AUTO=1` | **dense 匿名 ring 流式**（真降）/ **自适应 ring**：按可用内存定 ring 层数，装得下则全留 |
| `LLAMA_FLEX_RING=N` / `LLAMA_FLEX_THREADS=N` | 固定 ring 层数 / 并行 IO 线程（实测 4 打满盘）|

兼容映射：现有 `LLAMA_LAZY_MOE_BUFFER_*` / `LLAMA_FLEX_*` 保留为 SRE 的别名。

---

## 9. 已知限制

- **仅 CPU 后端**：weight-stream / node 钩子在 ggml-cpu；GPU/混合未覆盖。
- **dense 无出路**：带宽墙下 dense 省 RSS 与保 TPS 互斥，统一引擎只让其 RSS 真降可控，不改 ~0.16 t/s 的物理规律。价值在 MoE（稀疏红利）。
- **高激活比模型收益有限**：本测试 MoE 为 45% 激活剪枝版，地板≈全模型；价值在低激活比（Qwen3-30B-A3B 10%）。
- **共享专家 / 非层张量**：保持 mmap 常驻，不入 SRE 管理。
- **ring vs indexed 抽象**：迁移时 slot 寻址需抽象干净，避免 dense 退化为每层 full-size buffer。

---

**文档版本** v1.0 · 数据来源：本会话实测 · dense Llama-3-8B-Q4_K_M(4.9GB) + MoE Qwen1.5-MoE-20experts-Q4_K_M(3.67GiB) / 8 核 / QEMU virtio 盘(488MB/s 单线程·1.2GB/s 饱和) / cgroup memory.current / WikiText-2 PPL。状态：dense(flex) 与 MoE(moe-buffer) 流式均已实现并实测；SRE 统一为设计方案（未实现）。
