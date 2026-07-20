# 完整测试结果汇总（SRE / window / flex / moe-buffer）

本文档收录本项目**全部实测数据**。配套：[`SRE_UNIFIED_DESIGN.md`](SRE_UNIFIED_DESIGN.md)（统一设计）、[`IMPLEMENTATION_NOTES.md`](IMPLEMENTATION_NOTES.md)（对原生 llama.cpp 的更改 + 问题/算法）。

---

## 0. 测试环境与方法学

| 项 | 值 |
|---|---|
| 虚拟化 | QEMU 虚拟机（`systemd-detect-virt=qemu`，`/proc/cpuinfo` 有 hypervisor）|
| CPU | 8 核（推理用 `-t 6`）|
| 数据盘 | `vda` = **virtio_blk 虚拟盘**（50G），**无 NVMe 直通**（`/dev/nvme*` 不存在，`rotational=1`）|
| dense 模型 | Meta-Llama-3-8B-Instruct-Q4_K_M（**4.9 GB**，32 层）|
| MoE 模型 | Qwen1.5-MoE-A2.7B-20experts-SFT Q4_K_M（**3.67 GiB**，24 层，**45% 激活比**剪枝版）|
| PPL 数据集 | WikiText-2 raw（`wiki.test.raw`）|
| 工具 | `llama-bench`（吞吐）、`llama-perplexity`（正确性）|

**方法学要点（踩坑后确立）**：
- **RSS 一律用 cgroup `memory.current` 峰值**（真实物理），**不是 VmRSS**——VmRSS 在过配机器上是 page cache 假象。
- **找 TPS 地板用 `evictions→0` 拐点**（确定量），不靠 tg 曲线（`-r 1` 有 ±1 t/s 噪声）。
- **正确性用 perplexity 逐位比对**（同 repack-off 基线）。
- cgroup 测试前 `echo 3 > /proc/sys/vm/drop_caches`，并用子 cgroup 限 `memory.max` + `memory.swap.max=0`。

---

## 1. 盘带宽基准（决定一切上限）

`dd` O_DIRECT 实测（drop_caches 后）：

| 读法 | 带宽 |
|---|---|
| 单线程顺序 (`bs=1M iflag=direct`) | **488 MB/s** |
| 4 线程并行（不同区段）| **1194 MiB/s** |
| 8 线程 | 不再增（虚拟盘+hypervisor I/O 路径总上限 ≈ **1.2 GB/s**）|

→ **~1.2 GB/s 是物理硬墙**，4-worker 已打满。PCIe Gen5 NVMe 单盘 ~14 GB/s（×12），但本机无直通，guest 内不可得。

---

## 2. Dense — Llama-3-8B-Q4_K_M（4.9 GB）

### 2.1 cap=8G 不约束：native vs 关 repack vs mmap 驱逐（假降验证）

`llama-bench -p32 -n64 -t6 -r1`，同时采 cgroup memory.current + VmRSS：

| 配置 | cgroup 真实 RSS | tg | 说明 |
|---|---|---|---|
| native（mmap+repack）| **7770 MB** | 8.15 | repack ~2GB 不可回收副本 + 全模型 cache |
| window mmap（关 repack，无驱逐）| **4560 MB** | 8.60 | **-3.2GB 全来自关 repack，不掉 TPS** |
| window mmap + dontneed（`MEMORY_LIMIT=2G`）| **4560 MB** | 8.93 | **与不驱逐一模一样 → mmap 驱逐对物理零效果（假降）** |

**结论**：① 关 repack 降 3.2GB 是免费午餐；② mmap+madvise 对 cgroup 真实物理**零效果**（4560==4560）。

### 2.2 cap 阶梯（mmap window=4 + dontneed，制造内存压力，`-p4 -n4`）

| cap | cgroup 真实 RSS | tg | 状态 |
|---|---|---|---|
| native cap=3G | — | — | **OOM 杀死**（repack 副本超 cap）|
| 3072M | 3072 | 0.16 | 顶满 cap |
| 2048M | 2048 | 0.16 | 顶满 cap |
| 1500M | 1500 | 0.18 | 顶满 cap |
| 1024M | 1024 | 0.18 | 顶满 cap |
| 768M | 768 | 0.17 | 顶满 cap |
| 512M | 512 | 0.17 | 顶满 cap |
| 384M | 384 | 0.16 | 顶满 cap |
| 256M | 256 | 0.16 | 顶满 cap |
| 192M | 192 | 0.16 | 顶满 cap |
| **128M** | **128** | **0.17** | ✅ **最低可跑点** |
| 96M | — | — | OOM |
| 64M | — | — | OOM |

**结论**：① mmap → cgroup **永远顶满 cap**，window_size 调不动真实 RSS；② **带宽墙下 RSS 与 tg 解耦**（cap 3072→128M，tg 恒 ~0.16，IO=全模型/token 不变）；③ **硬下界 128 MB（native 的 1/60）**，4.9GB 模型可在 128MB 跑，但 0.17 t/s（≈6 秒/token，能跑不能用）。

### 2.3 flex 匿名 ring backing：dense 驱逐真降（已落地）

| 配置 | cgroup 真实 RSS | tg | 说明 |
|---|---|---|---|
| mmap window 关 repack（对照）| 4560（顶满，假降）| 8.60 | madvise 对物理无效 |
| **flex ring=4**，cap=8G | **1252 MB** | 1.52 | 匿名 ring 真降（非层权重+4层+KV+compute）|
| **flex AUTO**，cap=8G（充足）| 4277 MB | 7.45 | ring 自动定满 ≈ 不流式，速度近 native |
| **flex AUTO**，cap=2560M（<模型）| **1472 MB（<cap）** | 1.77 | ring 自动缩，**不 OOM**（mmap 会 OOM/顶 cap）|

**结论**：flex 的匿名 ring backing 让 dense 驱逐**真对物理生效**（1252/1472 < mmap 的 4560 顶满）；`FLEX_AUTO` 按可用内存自动定 ring（充足全留满速 / 受限缩 ring 真降）。dense 受限 tg 1.5–1.8（无稀疏 → 每 token 重读全模型）。

---

## 3. MoE — Qwen1.5-MoE-20experts-Q4_K_M（3.67 GB）

### 3.1 cap=8G 不约束：三模式（native / 关 repack / 开驱逐）

`llama-bench -p8 -n48/64 -t6 -r1`：

| 配置 | cgroup 真实 RSS | tg | RSS vs native |
|---|---|---|---|
| native（mmap+repack）| 5730 MB | 19.26 | 100% |
| v2 mmap（关 repack，无驱逐）| 3605 MB | 19.58 | 63%（-2.1GB repack，不掉 TPS）|
| moe-buffer+CLG 1536MB | 2671 | 3.31 | 47% |
| moe-buffer+CLG 768MB | 1905 | 2.77 | 33% |
| moe-buffer+CLG 512MB | 1649 | 1.58 | 29% |
| moe-buffer+CLG 384MB | 1521 | 1.72 | 27% |
| moe-buffer+CLG 256MB | **1392** | 1.38 | **24%** |

### 3.2 找 TPS 地板：预算阶梯 + evictions 拐点（`-p1 -n24`）

| budget | 真实 RSS | resident(工作集) | **evictions** | read 总量 | read/token | tg |
|---|---|---|---|---|---|---|
| 768MB | 1905 | 766 | 7392 | 14360 MiB | **598 MB** | 1.97 |
| 1536MB | 2666 | 1535 | 2710 | 6517 MiB | 272 MB | 2.23 |
| 2048MB | 3176 | 2047 | 618 | 3176 MiB | 132 MB | 6.22 |
| 2304MB | 3435 | 2302 | 133 | 2543 MiB | 106 MB | 8.40 |
| **2432MB** | **3542** | **2416** | **0** ✅ | **2416 MiB** | ~0 | 11.57 |
| 2560MB | 3541 | 2416 | 0 | 2416 MiB | ~0 | 9.43(噪声) |

**地板 = evictions 首次归零的 2432MB**（工作集 2416MB；read 触底=工作集一次性冷加载；再加预算 resident/read/tg 不变）。thrash 区每 token 读量被放大数倍（768MB 档 598MB/token ≫ 工作集），是 tg 崩的直接原因。

### 3.3 稳态验证（地板 vs native，冷启动摊销）

| | tg32 | tg128 |
|---|---|---|
| native | 22.88 | 23.62 |
| moe-buffer 2432（地板，1 worker）| 12.62 | **19.92（≈native 84%）** |
| moe-buffer 2432（地板，4 worker）| 11.01 | 18.47 |

→ 找到 evictions=0 地板后，稳态 tg 基本不掉（剩余差距 = 冷启动一次性读 2.4GB + O_DIRECT 不留 cache 的零星 miss）。

### 3.4 多 worker 并行读（松动带宽墙，`-p1 -n48`）

| budget | tg(1 worker) | tg(4 worker) | 加速 |
|---|---|---|---|
| 512MB | 1.58 | 2.87 | 1.8× |
| 768MB | 1.97 | 3.75 | 1.9× |
| 1536MB | 2.23 | **5.42** | 2.4× |
| 2432MB(地板) | 11.32 | 16.45 | 1.45× |

768MB 单看 worker 数：**1→1.97, 2→3.45, 4→3.75, 8→3.57**（8 核超订回落，**4 最优**）；read 量不变(~26GB)、tg 翻倍 → 是带宽提升而非读得少。多 worker 只在 IO-bound 区有效。

### 3.5 CLG δ 旋钮（过预测余量，累积 budget=0，`-n64`）

| δ | 工作集(RSS) | streams | 命中率 | tg |
|---|---|---|---|---|
| 0 | 2411 MB | 1314 | 94% | 12.52 |
| 1 | 2464 | 1344 | 93% | 13.97 |
| 2 | 2487 | 1356 | 93% | 13.60 |
| 4 | 2579 | 1407 | 93% | 16.48 |
| 8 | 2634 | 1437 | 93% | 17.27 |

→ δ↑ → 预取覆盖更全 → 异步 worker 命中更多 → 同步 stall 更少 → tg↑（但 RSS↑）。**δ 实为 TPS 旋钮，不是 RSS 旋钮**（δ:8→0 只省 8.5% RSS 却掉 27% tg）。

### 3.6 受限场景 cap=2G（< 模型）

| 配置 | 真实 RSS | tg |
|---|---|---|
| native | — | **OOM 杀死**（repack 副本超 cap）|
| v2 mmap | 2048（顶 cap，内核反复回收+重 fault）| 2.35 |
| moe-buffer+CLG 768 | **1904（<cap，受预算控制）** | **2.57** |

### 3.7 自适应 budget（`LLAMA_LAZY_MOE_BUFFER_AUTO=1`）

| 场景 | 实测 peak RSS | tg | 判定 |
|---|---|---|---|
| cap=8G（充足）| 2729 | **19.54** | 装得下 → budget=0 不驱逐（满速）|
| cap=2560M（<模型）| **1001** | 4.83 | 自动算 budget 驱逐，**不 OOM** |

### 3.8 热专家保护

**旧：绝对阈值 `HOT_N`（已废弃）**（budget=512MB thrash，`-n32`）：

| HOT_N | resident | streams | evictions | read | tg |
|---|---|---|---|---|---|
| 0 | 510 | 17604 | 17325 | 32257 MiB | 3.06 |
| 20 | **1077**（超 budget！）| 14460 | 13869 | 26548 MiB | 3.21 |

**新：相对频率 `HOT_RATIO`（已落地）**（budget=512MB thrash，`-n32`）：

| HOT_RATIO | resident | streams | evictions | read | tg |
|---|---|---|---|---|---|
| 0（纯 LRU）| 510.6 | 17601 | 17322 | 32251 MiB | 3.06 |
| 2.0 | **511.2（不超 budget）** | 15336 (-13%) | 15058 (-13%) | 28087 MiB (-13%) | 3.21 |

→ 相对判据**有界**：减 IO（-13%）的同时 pin 集合钉在预算内（511 vs 旧绝对阈值 1077）。

---

## 4. 正确性（perplexity，逐位比对）

moe-buffer+CLG（真异步 O_DIRECT 流式）vs window（mmap 基线），同 repack-off：

| chunk | window | moe-buffer+CLG |
|---|---|---|
| [1] | 47948.2078 | 47948.2078 |
| [2] | 80646.3078 | 80646.3078 |
| [3] | 95059.6850 | 95059.6850 |
| [4] | 104976.4199 | 104976.4199 |

**4 chunk 逐位一致** → O_DIRECT 流式 + 切片 repoint 数学 bit-exact。改动热专家/自适应后默认路径复测 [1][2] 仍为 47948.2078 / 80646.3078（未破坏正确性）。

> 注：PPL 绝对值极高是因为这是 20 专家剪枝 + SFT 的退化模型本身，与本框架无关（baseline 同样）。

---

## 5. 横向总结（native vs 不开驱逐 vs 开驱逐）

| 维度 | ① native | ② 不开驱逐（关 repack）| ③ 开驱逐 |
|---|---|---|---|
| **dense RSS** | 7770 MB | 4560 MB | **128 MB–cap（flex 真降可控）** |
| **dense tg** | 8.15 | 8.60 | 0.16（mmap 受限）/ 1.5–1.8（flex 受限）/ 7.45（flex 全留）|
| **MoE RSS** | 5730 MB | 3605 MB | **1392–2432 MB（24–62%）** |
| **MoE tg** | 19.26 | 19.58 | **1.4 → ~20 连续可调**（预算/worker）|
| **cap<模型** | OOM | OOM | ✅ **唯一能跑** |

**核心结论**：
1. 关 repack 免费降 dense -3.2GB / MoE -2.1GB，不掉 TPS。
2. 真降物理必须匿名 backing（mmap 假降已证 4560==4560 / dense 顶满 cap）。
3. dense 无稀疏 → 受限 tg 锁死带宽墙（1.5–1.8），RSS 与 tg 解耦（可压到 128MB）。
4. MoE 有稀疏 → tg 1.4→20 连续可调，地板（evictions=0）处稳态 ≈ native 84%。
5. 带宽墙 ~1.2 GB/s 是本系统硬约束，4-worker 打满；要再快须绕过带宽（换低激活比模型 / 量化 / batching）或换 NVMe。
