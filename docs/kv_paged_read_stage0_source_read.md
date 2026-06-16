# Stage 0 — Paged KV cache / PagedAttention source-read

## 阶段定位

本阶段是 paged-read / block-table 路线的**起点调研**，只做源码与参考分支阅读，输出判断。

- 只读当前 llama.cpp 源码与参考分支 `llama.cpp-paged-ref`；
- **不改源码**；
- **不 build**；
- **不实验**；
- 目标：判断 paged-read / block-table 是否适合作为下一阶段主线，以及第一版应做到哪条边界。

参考分支：`/root/oscomp/llama.cpp-paged-ref`，commit `0b0f7bd7e`（upstream 之上**单一提交**，+4029 −70 行）。

**一句话结论**：参考分支证明了 block-table 间接寻址的价值，但它选择"另起一条 paged path + 融合自定义算子 + 每步重建图 + 多序列 scheduler"的重型设计；我们第一版只借用 **block-table 元数据 + 物理块池 + free-list** 的思想，**不照搬算子、不破坏 graph reuse、不引入 scheduler**。

---

## 1. 当前 llama.cpp KV cache 读写路径

针对 `-fa on` / CPU / single-sequence / `n_stream=1`。

### 物理布局：连续 K/V tensor

每层一对连续张量（`llama-kv-cache.cpp:520`）：

```
k = ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream);
v = ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream);  // FA 下不转置
```

- 形状 `[n_embd_*_gqa, kv_size, 1]`，按位置 row-major。
- **整段 `kv_size` 连续预分配** —— 这是我们一直想打破的 RSS 来源。

### 写路径：find_slot → slot_info.idxs → cpy_k/cpy_v

1. `find_slot()`（`llama-kv-cache.cpp:1246`）在 `llama_kv_cells` 中挑空闲 cell；
2. 生成 `slot_info.idxs[stream][i]` = 本 ubatch 中 token `i` 的 cell 行号；
3. `cpy_k/cpy_v`（`llama-kv-cache.cpp:2239`）把 `k_cur` reshape 成 `[n_embd_gqa, n_tokens]`，再 `ggml_set_rows(k, k_cur, k_idxs)` 把每个 token 散写到对应行。

**关键**：token position → cell index 是**唯一映射**，且 cell index 在物理张量里**就是行号**（线性、连续）。`k_idxs/v_idxs` 是 runtime input（I64 张量），写动作是图中独立的 `set_rows` 节点。

### 读路径：get_k/get_v → [0,n_kv) 连续 view

`get_k/get_v`（`llama-kv-cache.cpp:2162`）返回一个 `ggml_view_4d`：

```
shape  = [n_embd_head_k, n_head_kv, n_kv, ns]
offset = stream_block + (approx ? visible_lo*row_size : 0)
```

- `n_kv = GGML_PAD(cells.used_max_p1(), n_pad)`；
- 这个 view 物理上要求 `[0, n_kv)` **连续驻留**，喂给标准 `build_attn_mha` → flash attention。

### set_input_kq_mask

`set_input_kq_mask`（`llama-kv-cache.cpp:2656`）建 `[n_kv, n_tokens, 1, n_stream]` F32 张量，按 `(q_pos, kv_pos)` 填 `0/-inf`，因果性在此体现。

### graph reserve / graph reuse 如何限制 dynamic view-offset

**这是我们 approx2 被卡的根因**：

- `n_kv` 和 mask 的 `ne[0]` 是**图拓扑常量**，在 `graph_reserve`（`llama-context.cpp:2222`）时烘进图；
- 只有 K/V 数据、`k_idxs/v_idxs`、mask 数值、`visible_lo` 是 runtime input；
- `can_reuse_kq_mask`（`llama-graph.cpp:39`）要求 `kq_mask->ne[0] == n_kv`，approx 模式下还要求 `visible_lo` 一致。

**推论**：任何试图缩短 physical view length 的做法都等于改 `n_kv`、等于改拓扑，立刻触发 reserve/reuse 冲突。**第一版的硬约束就是：不许动 `n_kv` 的拓扑常量地位。**

---

## 2. paged_attention 参考分支核心设计

参考分支**绕开**了 `get_k/get_v/set_input_kq_mask/build_attn_mha` 整条路径，并行新增了一套 paged 子系统，并用一个融合自定义算子完成 attention。

### 2.1 并行新增 paged 子系统（不是改造旧路）

通过 `cparams.kv_paged` 开关，在 graph 层做分流（`llama-graph.cpp:2309`）：

```
if (cparams.kv_paged) return build_attn_inp_kv_paged();
```

关闭时走原 KV path，开启时走全新 paged path。**这是它最值得借鉴的工程姿态：另起一条路 + 默认关闭 + 可回退。**

### 2.2 关键组件

| 组件 | 文件 | 作用 |
|---|---|---|
| `GGML_OP_PAGED_ATTN` + `ggml_paged_attn()` | `ggml.h/.c`, `ops.cpp` | 融合 write+read 算子；11 张量输入 + scale/block_size/max_blocks |
| `llama_kv_cache_paged` | `llama-kv-cache-paged.h/.cpp` (443+186) | paged 物理池管理，实现 `llama_memory_i` |
| `llama_block_manager` | `llama-block-manager.h/.cpp` (109+45) | 物理块池，free-list，ref_count，watermark 安全余量 |
| `llama_sequence_group` | `llama-sequence-group.h` (31) | per-seq：`block_table`、`logical_seq`、status、n_past/n_decoded |
| `llama_paged_scheduler(_impl)` | 3 文件 (701 行) | waiting/running/swapped 三队列、FCFS 准入、抢占、swap、死锁检测 |
| `llm_graph_input_attn_kv_paged` | `llama-graph.h/.cpp` | 持有 5 input 张量：write_slots / block_table / context_lens / batch_offsets / batch_lens |

### 2.3 physical block pool / K/V pool layout

物理池布局（`llama-kv-cache-paged.cpp:54-68`）：

```
逻辑 5D: [num_blocks, 2, n_heads_kv, block_size, head_dim]
因 GGML_MAX_DIMS=4 压平为 4D: [head_dim, block_size, 2*n_heads_kv, num_blocks]
```

- **K 与 V 交错存在同一张量**：前 `n_heads_kv` 个 head 段是 K，后 `n_heads_kv` 段是 V；
- `block_size` 个 token 一块（默认 16）。

### 2.4 block table 与 write_slots

- `llama_sequence_group.block_table`：逻辑块 → 物理块号；
- 映射公式 `calculate_global_slot_index`：

```
block_id      = token_pos / block_size
offset        = token_pos % block_size
global_slot   = block_table[block_id] * block_size + offset
```

- 本 batch 所有 token 的 `global_slot` 打包成 `write_slots[n_tokens]` 张量，作为算子输入。

### 2.5 block_table read path

算子 decode 段对每个 q：遍历 `num_blocks = q_pos/block_size + 1` 个逻辑块，`physical_block = block_table[seq*max_blocks + bid]`，逐块逐 token 用 `ggml_backend_tensor_get` 取 K/V 做 online-softmax flash attention。**物理块散布在池中任意位置，读取全靠 block_table 间接寻址** —— 这是 paged 的核心收益。无 kq_mask，因果性由 `end_token = min(start+block_size, q_pos+1)` 隐式实现。

### 2.6 cparams 新增

`kv_paged`（默认 false）、`block_size`（默认 16）、`n_gpu_blocks`、`n_cpu_blocks`、`kv_paged_watermark`（0.05）。

---

## 3. 明确判断

1. **它选择"另起一条 paged path"，不是改造旧 KV path** —— 旧路完全保留，新路由 `kv_paged` 分流。
2. **它修改了 ggml op / attention kernel** —— 新增 `GGML_OP_PAGED_ATTN`，CPU 参考实现 189 行（显式标注单线程、未优化、仅供正确性验证），CUDA 实现 208 行。融合算子在**单个 op 内同时完成 write + read**。
3. **它依赖/深度服务于 continuous batching** —— scheduler 三队列、抢占、swap、死锁检测都是多序列 serving 专用；算子的 `batch_offsets/batch_lens/context_lens` 都为多序列 batch 准备。
4. **graph 完全不复用** —— `llm_graph_input_attn_kv_paged::can_reuse` 直接 `return false`（`llama-graph.cpp:567`），注释："we can 'never' re-use the graph. Because we have write_slots"。它用每步重建图换任意可变 block table，**与我们想保住 graph reuse 的方向相反**。
5. **第一版不建议照搬完整设计** —— 重、维护 CPU+CUDA 两份算子、绕开成熟 FA、引入大量多序列机制，对单序列 CPU 零收益。

---

## 4. 对我们比赛目标可借鉴的部分

1. **block-table 间接寻址** —— 物理块不连续、按需分配，直击预分配 RSS。核心可借鉴点。
2. **physical block pool** —— 干净的物理池抽象，适配 lazy / 可释放路线。
3. **free-list** —— `free_*_ids` LIFO free list，块粒度分配/归还，简单可靠。
4. **logical position → physical block 的映射公式** —— `block_id = pos/block_size`、`offset = pos%block_size`、`global_slot = block_table[block_id]*block_size + offset`，直接可复用。
5. **默认关闭 + graph 分流的工程模式** —— `kv_paged` 开关 + `build_attn_inp_kv_paged()` 分流，与我们"默认关闭、可回退"边界完全契合。
6. **block 级 release / swap / prefetch 的基础** —— 块粒度是后续"可释放 / 可换出 / 可预取"的天然单位（本阶段只立 metadata，不做这些）。

---

## 5. 第一版不建议照搬的部分

1. **不新增 `GGML_OP_PAGED_ATTN`** —— 重、需维护 CPU+CUDA 两份、绕开成熟 FA、参考 CPU 实现单线程慢。
2. **不改 attention kernel** —— 复用现有 `build_attn_mha` + flash attention。
3. **不引入 continuous batching scheduler** —— 三队列 / FCFS 准入 / 死锁检测纯多序列 serving，单序列零收益。
4. **不做多序列抢占** —— preempt / evict / recompute 多序列专用。
5. **不做 GPU/CPU 双池** —— 我们 CPU-only，双池 + PCIe swap 无意义。
6. **不做 ref_count / CoW 前缀共享** —— 多序列特性，单序列用不上。

---

## 6. 我们的最小实现方向

边界：**CPU-only / single-sequence / `-fa on` / no shift / no seq_cp / no SWA / 默认关闭**。

核心策略：**借 block-table 的"物理不连续"思想，但不引入新算子、不破坏 graph reuse。**

- **Stage 1 — block-table metadata scaffold（先做这个）**：在 `llama_kv_cache` 旁加可选 `block_table`（逻辑块→物理块号）与块粒度物理池抽象，但**池仍是 ggml 张量、cell index 仍是池内行号**。`kv_paged=false` 时退化为恒等映射，字节级等价原行为。本阶段**只立元数据，不改读写语义**。
- **Stage 2 — write path block mapping**：`find_slot` 按块惰性分配物理块并填 block_table；`cpy_k/cpy_v` 的 `k_idxs/v_idxs` 改为经 block_table 解析的物理行号。先用恒等映射跑通回归再切真块式。
- **Stage 3 — staging paged-read**：读时按 block_table 用 `ggml_get_rows`（或等价 cpy）把所需物理块 gather 到**固定长度连续 staging 张量**（长度 = `GGML_PAD(n_kv, n_pad)`，保持拓扑常量），再交给现有 `get_k/get_v` view + FA。复用现有算子，不动 kernel、不破 reuse。
- **Stage 4 — block release / swap**：序列结束 / 超窗块归还 free-list；可选 CPU 内换出（非 PCIe）。
- **Stage 5 — prefetch**：按访问模式预取块。

---

## 7. 风险表（按严重度排序）

| # | 风险 | 严重度 | 触发条件 | 缓解 |
|---|---|---|---|---|
| R1 | staging gather 破坏 graph reuse | 高 | gather view 长度/结构随 batch 变 | 固定 staging 长度 = `GGML_PAD(n_kv, n_pad)`，只变数据不变形状（复用 fix-C 经验） |
| R2 | block_table 映射错误 → 静默 correctness bug | 高 | `pos/block_size` 映射与 find_slot 不一致 | 先 metadata-only + 恒等映射回归，逐 token 比对 master 再切块式 |
| R3 | gather 拷贝开销吃掉收益 | 中 | n_kv 大、block 小 | block_size 取 16/32，profile 拷贝 vs RSS |
| R4 | 与 lazy-clear/lazy-tail 状态交叉 | 中 | 两套"可释放"语义并存 | 明确块池为唯一 truth，lazy 标志映射到块状态 |
| R5 | `-fa on` view stride 假设被破坏 | 中 | get_k 连续 row 假设 | staging 后仍是连续张量，get_k 不变 |
| R6 | 默认关闭路径回归 | 低 | 开关分流写错 | 开关包住所有新路径，关闭 = 字节级原行为 |
| R7 | 误把 scheduler/swap 引入第一版 | 低 | 照搬冲动 | 本文档明确禁止 |

**关于 RSS 的诚实判断**：full-history exact 单序列下，paged-read **只能先减少预分配 / 碎片**（按需分配物理块替代 `kv_size` 整段预分配）。若要降低 **active RSS**，还需 window / release / swap（Stage 4+）。**不应过早声称 RSS 一定下降** —— Stage 1-3 的收益主要在"按需分配"与"为后续释放铺路"，active 工作集本身不变。

---

## 8. 下一步建议（阶段路线）

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Stage 0** | 本 source-read 文档 | ✅ 本文 |
| **Stage 1** | block-table metadata scaffold（恒等映射，默认关闭，字节级等价） | 下一步 |
| **Stage 2** | write path block mapping（find_slot 块式分配 + idxs 查表） | 待定 |
| **Stage 3** | staging paged-read（固定长度 gather + 复用 FA，不破 reuse） | 待定 |
| **Stage 4** | block release / swap（归还 free-list，降 active RSS） | 待定 |
| **Stage 5** | prefetch | 待定 |

**建议**：先做 Stage 1 metadata scaffold，因为它风险最低（默认关闭、恒等映射、字节级可验证），且能验证"block-table 抽象嵌入现有 `llama_kv_cache` 而不破 graph reuse"这个最关键的工程假设。Stage 1 通过后再评估 Stage 3 是否值得（取决于 gather 开销 profile）。
