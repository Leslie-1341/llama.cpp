# Stage 3 — staging paged-read timing analysis

前置：

- [docs/kv_paged_read_stage3_staging_read_plan.md](kv_paged_read_stage3_staging_read_plan.md)
- [docs/kv_paged_read_stage2a_write_identity_plan.md](kv_paged_read_stage2a_write_identity_plan.md)

本阶段**只做 source-read，不改源码、不 build、不跑实验、不 commit**。目的：核实 Stage 3 原设计文档推荐的「方案 B：graph 外 CPU staging copy 作为真实读路径」时序是否成立。

结论先行：**方案 B 作为真实读路径时序不成立**，必须降级为 shadow validation；真正读端接管走 Stage 3B graph 内 gather。

---

## 1. 关键源码结论

### 1.1 `get_k/get_v` 调用时机

`get_k/get_v` 是**构图期（graph build）函数**，返回 `ggml_view_4d` 节点，不搬数据。它们在 `model.build_graph()`（[llama-context.cpp:1282](../src/llama-context.cpp#L1282)）内、`build_attn`（[llama-graph.cpp:2247-2248](../src/llama-graph.cpp#L2247-L2248)）中被调。**不是每 decode step 都调** —— 仅当图不能 reuse 时 `build_graph` 才执行。

### 1.2 graph reuse 后是否重调

**不重调**。[llama-context.cpp:1263](../src/llama-context.cpp#L1263) `can_reuse(gparams)` 为真时 `n_reused++` 并整段跳过 `build_graph`，`get_k/get_v`/`cpy_k/cpy_v` 都不再执行。每 step 只跑 `set_inputs`（[:1304](../src/llama-context.cpp#L1304)）→ `graph_compute`（[:1309](../src/llama-context.cpp#L1309)）。view 节点沿用上一图，base 指针/形状在构图时已固定。

### 1.3 `cpy_k/cpy_v` 写入时机

`cpy_k/cpy_v`（[llama-graph.cpp:2240-2241](../src/llama-graph.cpp#L2240-L2241)）也是**构图期**函数，emit `ggml_set_rows` 节点。**真实 K/V 写入发生在 `graph_compute` 内**，作为图节点执行 —— 不是 compute 前。

### 1.4 `graph_compute` 内 cpy 与 attention 读的顺序

二者**在同一次 `graph_compute` 调用内**按 DAG 定序：① [llama-graph.cpp:2226-2231](../src/llama-graph.cpp#L2226-L2231) 注释明确「these nodes are added together so they are not reordered」，`cpy_k/cpy_v` 先于 `get_k/get_v` 加入 `gf`；② `cpy_k` 写的 cache 张量与 `get_k` view 的 base 是同一张量，数据依赖强制 set_rows 先于 view 求值。即 **写（cpy）→ 读（get→FA）顺序由图内数据依赖保证**，无需外部干预。

### 1.5 为什么 graph 外 CPU staging copy 会读旧 KV

唯一能放 CPU gather 的时机是 compute **前**（`set_inputs` 阶段，[:1304](../src/llama-context.cpp#L1304)）。但本 step 新 token 的 K/V 由 `cpy_k/cpy_v` 节点在 **compute 内**才写入 cache。故 compute 前的 CPU copy gather 到的是**上一 step 的旧状态**，缺少本 step 刚写入的行 —— 喂给 FA 会少一行 K/V，输出错误。

### 1.6 为什么不存在干净 CPU hook

`cpy` 写节点与 attention 读节点都在**同一次不透明的 `graph_compute`** 内顺序执行，二者之间**没有回到 CPU 的控制点**。`cb_eval`（[:1278](../src/llama-context.cpp#L1278) `set_eval_callback`）虽是 compute 期逐节点回调，但定位是观察/中止，tensor data 可能在 device 上，借它在节点间 mutate buffer 并让下游节点感知既非设计支持也极脆弱，**不能算可用 hook**。

---

## 2. 结论

1. **原方案 B（graph 外 CPU staging copy 作为真实读路径）不能成立** —— compute 前 gather 漏本 step 写入，cpy 与 read 之间无 CPU hook。
2. **方案 B 降级为 Stage 3A-shadow**：compute 后做影子 gather + reference 比对，**绝不接管 attention**。
3. **真正读路径接管应走 Stage 3B graph 内 gather**（`ggml_get_rows` 等价节点），或更重的自定义 op / split graph —— 这是唯一时序成立的真实路径。

---

## 3. 修正后路线

| 阶段 | 内容 | 是否接管读 |
|---|---|---|
| **Stage 3A-shadow** | compute 后，按 `block_table` 把物理行 gather 到独立 shadow buffer，与直接读 cache 连续 `[0,n_kv)` 逐字节比对；shadow 绝不喂 FA | 否（纯验证） |
| **Stage 3B** | `build_attn` 内、`cpy_k/cpy_v` 之后 `get_k/get_v` 之前插入 graph 内 gather 节点，`get_k/get_v` 改在 staging 节点上开 view | 是（真实读路径） |
| **Stage 2B** | 真正非连续块分配 —— **必须在 Stage 3B 之后**，读写寻址同时切换才安全 | — |

---

## 4. Stage 3A-shadow 通过标准（全部满足）

1. ✅ 输出与 baseline 完全一致（未改读路径，`LLAMA_KV_PAGED=0/1` 与 baseline diff 全空、三 sha256 一致）；
2. ✅ `shadow_gather_calls > 0`（影子 gather 确实跑）；
3. ✅ `shadow_changed == 0`（identity 下 `phys==r`）；
4. ✅ `shadow_mismatch == 0`（gather 行与连续 reference 逐字节相等）；
5. ✅ `write_resolve_changed == 0`（Stage 2A 写端自证仍成立）；
6. ✅ `identity_fail == 0`（Stage 1 block-table 自证仍成立）；
7. ✅ `graphs reused` 不下降（持平 62，未进图）。

判定核心：**diff 空 且 `shadow_gather_calls>0` 且 `mismatch==0`** —— 证明 gather 公式与整行字节拷贝逻辑在 identity 下正确，零正确性风险地为 Stage 3B 攒信心。

---

## 5. Stage 3B 主要风险

| # | 风险 | 说明 |
|---|---|---|
| 1 | **graph 内 gather 是否破坏 reuse** | `can_reuse` 比对图拓扑；gather 节点数/形状必须逐 step 恒定，否则 reuse 下降甚至失效 |
| 2 | **行号张量如何作为 runtime input** | 需像 `k_idxs` 一样走 `set_input` 链路填值（identity 下 `[0,n_kv)`），多一条输入链路 |
| 3 | **`ggml_get_rows` 对 K/V layout 是否适配** | `get_rows` 按 ne[1] 行 gather，4D K/V 需 reshape 成 2D 行集再 gather 再 reshape，构图复杂度上升 |
| 4 | **量化 / v_trans / offload 兼容性** | 量化 K 的 get_rows 类型支持、转置 V 的行列语义、device buffer 上的 gather 都需逐一核对 |

---

## 6. 下一步建议

1. **优先 Stage 3A-shadow**：零风险打通并验证 gather 公式，纠正设计文档把方案 B 当真实读路径的错误。
2. **不要直接做 Stage 2B**（非连续写）：读端尚未真实 block-table 化，会读错块。
3. **不要直接做真实 staging read**（原方案 B）：时序已证伪。
4. Stage 3A-shadow 通过后，再投入 Stage 3B graph 内 gather 作为真实读路径接管，最后才 Stage 2B。
