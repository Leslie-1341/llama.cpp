# Stage 3B — graph 内 gather 真实读路径设计计划

前置：

- [docs/kv_paged_read_stage3_timing_analysis.md](kv_paged_read_stage3_timing_analysis.md)
- [docs/kv_paged_read_stage3a_shadow_results.md](kv_paged_read_stage3a_shadow_results.md)

本文是**设计文档**：只新增文档，不改源码、不 build、不跑实验、不 commit。

背景：Stage 3A-shadow 已通过（`shadow_gather_calls=65`、`changed/mismatch/fail` 全 0），证明 identity 下 gather 公式与整行拷贝逻辑正确。但 shadow 在 compute 后、只做比对、不接管 attention。timing analysis 已证明 graph 外 CPU copy 不能作真实读路径。Stage 3B 把 gather 搬进图，成为唯一时序成立的真实读端接管。

---

## 1. Stage 3B 目标

1. 在 `build_attn` 内、`cpy_k/cpy_v` 之后、`get_k/get_v`/FA 读取之前，插入 **graph 内 gather 节点**，使读路径真正经 `block_table` 取物理 K/V 行；
2. gather 与 cpy 在**同一次 `graph_compute`** 内，由图数据依赖自动定序，无需 CPU hook（解决 timing 问题）；
3. 第一版仍 **identity**（行号 `[0,n_kv)`），输出与 baseline 逐位一致；
4. gather 输出形状/`n_kv` 拓扑常量不变，FA kernel 无感；
5. `LLAMA_KV_PAGED=0` 默认路径字节级不变。

---

## 2. 非目标（硬约束）

| # | 不做 | 理由 |
|---|---|---|
| 1 | 不做非 identity block allocation | Stage 2B 才动，本阶段 `block_table[i]=i` |
| 2 | 不做 release / swap / prefetch | Stage 4/5 |
| 3 | 不改 attention kernel | 复用现有 FA |
| 4 | 不新增自定义 ggml op | 仅用现有 `ggml_get_rows`/`reshape`/`view` |
| 5 | 第一版保持 identity | 先证进图 gather 链路恒等 |
| 6 | `LLAMA_KV_PAGED=0` 路径完全不变 | 零回归底线 |
| 7 | 不做 Stage 2B 非连续写 | 必须等 Stage 3B 读路径通过后 |
| 8 | 首版跳过 `v_trans` / `n_stream>1` / 量化 K/V | 见 §11 限制 |

---

## 3. 当前图内 K/V 写读顺序

源码核实（[llama-graph.cpp:2225-2250](../src/llama-graph.cpp#L2225-L2250)，单次 `graph_compute` 内的节点 DAG）：

```
ggml_build_forward_expand(gf, q_cur / v_cur / k_cur)
// store to KV cache（写）
cpy_k = ggml_set_rows(k, k_cur, k_idxs)   // 本 step K 写入物理行
cpy_v = ggml_set_rows(v, v_cur, v_idxs)   // 本 step V 写入物理行
// 读
k = get_k(ctx, il, ...)   // ggml_view_4d 连续 [0,n_kv)
v = get_v(ctx, il, ...)
cur = build_attn_mha(q, k, v, ...)        // FA 读
```

cpy（写）先于 get（读）由两点保证：① [llama-graph.cpp:2226](../src/llama-graph.cpp#L2226) 注释「added together so not reordered」；② cpy 写的 cache 张量与 get view 的 base 同张量，数据依赖强制 set_rows 先求值。**Stage 3B 的 gather 节点要插在 cpy 之后、get 之前**，天然继承同一数据依赖链。

`get_k` view 形状（[llama-kv-cache.cpp:2434-2440](../src/llama-kv-cache.cpp#L2434-L2440)）：`ne=[n_embd_head_k, n_head_kv, n_kv, ns]`，第 3 维 `n_kv` 步长 `row_size=ggml_row_size(type, n_embd_k_gqa)` —— 每个 cell 行是 `n_embd_k_gqa = n_embd_head_k*n_head_kv` 个连续元素。

---

## 4. graph 内 gather 技术方案

用 `ggml_get_rows(a, b)`：按 `b`（I32 行号）从 `a` 的 ne[1] 维 gather 行。源码约束（[ggml.c:3853](../ggml/src/ggml.c#L3853)）：

- `a->ne[2]==b->ne[1]`、`a->ne[3]==b->ne[2]`、`b->ne[3]==1`、`b` 必须 **I32**；
- **输出 dtype 被强制 F32**（除非 `a` 本身 I32）—— 见 `// TODO: implement non F32 return`。

**关键后果**：对 F16 / 量化 K/V 做 get_rows 会**升 F32**，改变喂给 FA 的 dtype。故首版限定 **K/V cache 为 F32**（或接受 gather 后 F32 这一路径，需确认 FA 对 F32 K/V 的支持与数值等价）。量化/F16 留待后续（或改用 cpy-based gather）。

流程（每 layer，paged 启用时）：

```
k_cont   = reshape_2d(k_cache, n_embd_k_gqa, kv_size)   // [row, kv_size]
k_rows   = ggml_get_rows(k_cont, row_idx)               // [row, n_kv] F32
k_gather = reshape_3d(k_rows, n_embd_head_k, n_head_kv, n_kv)
// get_k 改在 k_gather 上开 4D view（或直接返回 reshape_4d）
```

---

## 5. K gather 设计

1. `k = layers[ikv].k`，row = `n_embd_k_gqa`，`kv_size=get_size()`；
2. `k2d = ggml_reshape_2d(ctx, k, n_embd_k_gqa, kv_size)`（a->ne[2]=ne[3]=1）；
3. `row_idx`：I32 张量，`ne=[n_kv]`（→ ne[1]=ne[2]=1，满足 `a->ne[2]==b->ne[1]==1`）；
4. `k_rows = ggml_get_rows(ctx, k2d, row_idx)` → `[n_embd_k_gqa, n_kv]` F32；
5. `k_g = ggml_reshape_3d(ctx, k_rows, n_embd_head_k, n_head_kv, n_kv)`；
6. `get_k` 在 `k_g` 上返回 `ggml_view_4d`（ns 维=1，首版单序列），ne/nb 与原 view 对齐。

identity 下 `row_idx=[0,n_kv)`，get_rows 逐行复制 = 原连续 view，输出不变。

---

## 6. V gather 设计

- **非转置（`!v_trans`，首版唯一支持）**：V row = `n_embd_v_gqa`，与 K 同构 —— `reshape_2d → get_rows(row_idx) → reshape` → get_v 在其上开 view。
- **转置（`v_trans`）**：V 物理 layout 把 `n_kv` 放 ne[0]（列），gather「列」无法直接用按 ne[1] 行的 `get_rows`，需先 transpose 或换 cpy-based gather。**首版跳过 `v_trans`**，告警一次走原路径。
- K、V **分别处理**：row 维（`n_embd_k_gqa` vs `n_embd_v_gqa`）、转置语义不同，但**共用同一个 `row_idx`**（identity 下都是 `[0,n_kv)`，物理寻址一致）。

---

## 7. runtime input 行号张量设计

- 像 `k_idxs`（[llama-graph.cpp:2237](../src/llama-graph.cpp#L2237)）一样：建图期建 `row_idx` 输入张量（I32, `ne=[n_kv]`），`set_input` 期由 kv-cache 按 `block_table` 填 `paged_resolve(r)`；
- identity 下填 `[0,n_kv)`；Stage 2B 起填非连续物理行号 —— **读端寻址自动 block-table 化**，正是 Stage 3B 的价值；
- 填值时机在 `set_inputs`（compute 前），但 gather 在 compute 内执行、读的是 cpy 写完的 cache，**时序成立**（与 shadow 必须在 compute 后的原因一致：这里 gather 节点本身在 compute 内 cpy 之后）；
- 长度固定 `n_kv`（拓扑常量），逐 step 不变。

---

## 8. graph reuse 风险

- `can_reuse` 比对图拓扑：gather 节点数 / 形状 / `n_kv` 必须**逐 step 恒定**，否则 reuse 下降；
- gather 节点对每个 attn layer 固定新增同样数量，拓扑稳定 → 预期 reuse 仍可成立，但**首版必须以 `graphs reused` 持平 62 守门**，若下降需排查；
- `row_idx` 作为 input 与 `k_idxs` 同类，已有 input 链路模式可循，不改变节点序只增固定节点；
- `n_kv` 仍由 `get_n_kv`/`get_reserve_n_kv`（[:2360](../src/llama-kv-cache.cpp#L2360)/[:2398](../src/llama-kv-cache.cpp#L2398)）给出，gather 不触碰它 —— 拓扑常量不变。

---

## 9. 最小实现路线

**定义**：identity 下，paged 模式 `get_k/get_v`（`!v_trans && n_stream==1 && F32 cache`）改为返回「`cpy` 之后、经 `ggml_get_rows(row_idx)` gather 的 staging 节点」上的 view，输出与 baseline 字节级一致。

1. 建 `row_idx` 输入张量（I32, `n_kv`），入 input 集合，`set_input` 填 `paged_resolve(r)`；
2. `get_k`：paged 启用 → `reshape_2d → get_rows(row_idx) → reshape_3d →` 在其上开 4D view；否则原 view；
3. `get_v` 非转置同构；`v_trans` / 量化 / 多序列 → 告警一次走原路径；
4. 统计 `paged_ingraph_gather_layers` / `row_idx_changed`（identity 应 =0）；
5. 全程不碰 attention kernel / ggml / `n_kv` 计算。

---

## 10. 验证命令与通过标准

> 均 `-fa on --temp 0 --seed 42`，确定性贪心；本设计阶段不跑，Stage 3B 编码后执行。

```bash
./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on --seed 42 --temp 0 > /tmp/base.txt
LLAMA_KV_PAGED=0 ... > /tmp/off.txt ; diff /tmp/base.txt /tmp/off.txt   # 空
LLAMA_KV_PAGED=1 ... > /tmp/on.txt  ; diff /tmp/base.txt /tmp/on.txt    # identity 下空
sha256sum /tmp/base.txt /tmp/off.txt /tmp/on.txt                        # 三者一致
```

**通过标准（全部满足）**：
1. ✅ `base_vs_off` diff 空（默认关闭字节级等价）；
2. ✅ `base_vs_on` diff 空、三 sha256 一致（**进图 gather 真实接管读路径但仍恒等**）；
3. ✅ `row_idx_changed==0`、`identity_fail==0`、`write_resolve_changed==0`；
4. ✅ `paged_ingraph_gather_layers > 0`（读路径确实经图内 gather）；
5. ✅ `graphs reused` 持平 62（reuse 未破坏）；
6. ✅ 与 lazy-tail 同开仍 diff 空。

判定核心：**读路径真正走图内 `get_rows` 且 diff 空** —— 区别于 Stage 3A-shadow（不接管）。

---

## 11. 风险表

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| S3B-R1 | `get_rows` 强制 F32 输出，F16/量化 K/V 被升精度，改变 FA 输入 dtype | 高 | 首版限 F32 cache；F16/量化留后续或改 cpy-based gather；以 sha256 守门 |
| S3B-R2 | gather 节点破坏 graph reuse | 高 | 节点数/形状逐 step 恒定；`graphs reused` 持平 62 守门 |
| S3B-R3 | reshape/view 维度算错致输出乱 | 高 | 严格对齐原 `get_k` 的 ne/nb；逐 token diff |
| S3B-R4 | `v_trans` 列 gather 无法用 get_rows | 中 | 首版跳过 `v_trans`，告警走原路径 |
| S3B-R5 | `row_idx` input 链路与 `k_idxs` 交互 | 中 | 复用现有 input 模式；identity 下 `[0,n_kv)` |
| S3B-R6 | offload backend 上 get_rows / input 行为 | 中 | 首版限 CPU/host；后端覆盖留后续 |
| S3B-R7 | 多序列 ns>1 的 row_idx/形状 | 低 | 首版 `n_stream==1` |

**RSS 立场**：仍 identity，物理布局连续、分配不变，**不期待 RSS 下降**；get_rows 输出 staging 张量由图分配。收益是「读端真实经 block-table 寻址」，为 Stage 2B 非连续写解锁安全前置。

---

## 12. Codex 实现提示词草案（暂不改代码）

> **任务：在 `src/llama-graph.cpp` + `src/llama-kv-cache.{h,cpp}` 实现 Stage 3B graph 内 gather 真实读路径。paged 模式（`!v_trans && n_stream==1 && F32 K/V cache`）下，`get_k/get_v` 改为返回「`cpy_k/cpy_v` 之后经 `ggml_get_rows(row_idx)` gather」的节点上的 view，identity 下输出与 baseline 逐 token 一致。**
>
> **背景时序（已源码核实，勿违反）**：`build_attn`（[llama-graph.cpp:2225-2250](../src/llama-graph.cpp#L2225-L2250)）内 `cpy_k/cpy_v`（set_rows 写）先于 `get_k/get_v`（read）；gather 节点必须插在两者之间，靠图数据依赖定序，**勿用 CPU hook**。`ggml_get_rows`（[ggml.c:3853](../ggml/src/ggml.c#L3853)）：`b` 须 I32、`a->ne[2]==b->ne[1]`、**输出强制 F32** —— 故首版限 F32 cache。
>
> **硬约束**：① 只改 `llama-graph.cpp`/`llama-kv-cache.{h,cpp}`，禁改 attention kernel / ggml / context，不新增 ggml op；② identity（`block_table[i]=i`，`row_idx=[0,n_kv)`）；③ 不做 release/swap/prefetch、不做 Stage 2B 非连续写；④ `n_kv` 拓扑常量不变；⑤ `LLAMA_KV_PAGED=0` 字节级等价；⑥ 仅 `!v_trans && n_stream==1 && F32 cache` 启用，否则告警一次走原路径。
>
> **实现**：
> 1. 建 `row_idx` 输入张量（I32, `ne=[n_kv]`），入 input 集合，`set_input` 期按 `paged_resolve(r)` 填值（identity `[0,n_kv)`）；统计 `row_idx_changed`（`phys!=r`，identity 应 0）。
> 2. `get_k`（[:2434](../src/llama-kv-cache.cpp#L2434)）：paged 启用 → `k2d=reshape_2d(k, n_embd_k_gqa, kv_size)` → `k_rows=get_rows(k2d, row_idx)` → `reshape_3d(n_embd_head_k, n_head_kv, n_kv)` → 在其上 `ggml_view_4d`（ne/nb 对齐原 view，ns=1）；否则原样。
> 3. `get_v` 非转置（[:2447](../src/llama-kv-cache.cpp#L2447)）同构；`v_trans`/量化/多序列告警走原路径。
> 4. 统计 `paged_ingraph_gather_layers`；`paged_log_stats` 追加，走 `LLAMA_LOG_*`（stderr）。
>
> **验证**（见 §10）：`base_vs_off`/`base_vs_on` diff 全空、三 sha256 一致；`row_idx_changed==0`、`identity_fail==0`、`write_resolve_changed==0`；`paged_ingraph_gather_layers>0`；`graphs reused` 持平 62；lazy-tail 同开 diff 空。
>
> **对照（只读，勿照搬）**：`/root/oscomp/llama.cpp-paged-ref` 的 `calculate_global_slot_index`（`block_table[bid]*block_size+offset`）作行号公式参考，但不引入其 fused `GGML_OP_PAGED_ATTN`。
