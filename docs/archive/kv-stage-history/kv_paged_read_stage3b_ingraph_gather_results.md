# Stage 3B — graph 内 gather 真实读路径 results

前置：

- [docs/kv_paged_read_stage3b_ingraph_gather_plan.md](kv_paged_read_stage3b_ingraph_gather_plan.md)
- [docs/kv_paged_read_stage3a_shadow_results.md](kv_paged_read_stage3a_shadow_results.md)
- [docs/kv_paged_read_stage3_timing_analysis.md](kv_paged_read_stage3_timing_analysis.md)

本文记录 Stage 3B 的实现与验证结果。

---

## 1. Stage 3B 目标

在 `build_attn` 内、`cpy_k/cpy_v`（写）之后、attention 读取 K/V 之前，插入 **graph 内 gather 节点**（`ggml_get_rows(row_idx)`），使 attention **真正读取经 `block_table` gather 后的 K/V**。gather 与 cpy 同在一次 `graph_compute` 内，由图数据依赖自动定序，无需 CPU hook —— 这是 timing analysis 证伪 graph 外 staging copy 后，唯一时序成立的真实读端接管。

第一版限定：

- `LLAMA_KV_PAGED=1`；
- `-ctk f32 -ctv f32 -nkvo`（F32 K/V cache，no kv offload）；
- `n_stream==1`；
- `!v_trans`；
- identity `block_table[i]=i`，`row_idx=[0,n_kv)`。

---

## 2. 与 Stage 3A-shadow 的区别

| 维度 | Stage 3A-shadow | Stage 3B |
|---|---|---|
| gather 位置 | graph **外**，`graph_compute` 之后（`next()` 头部） | graph **内**，`cpy_k/cpy_v` 之后、attention 读之前 |
| 是否接管读 | **否** —— shadow buffer 只做逐字节比对，绝不喂 FA | **是** —— attention 真实读取 gather 后的 K/V |
| 时序依据 | compute 后比对，gather 到最新 cache 仅用于验证 | 图内数据依赖：set_rows 写先于 get_rows 读求值 |
| 价值 | 验证 gather 公式与整行拷贝逻辑正确 | 读路径真正经 block-table 寻址 |

Stage 3A-shadow 是零风险前置（证 gather 公式恒等）；Stage 3B 才是真实 paged-read。

---

## 3. 代码改动范围

修改三个文件：

1. **`src/llama-kv-cache.h`**：`get_k/get_v` 签名增加 `row_idx` 入参；新增 `mutable` 统计 `ingraph_gather_layers` / `row_idx_changed` / `row_idx_fail`。
2. **`src/llama-kv-cache.cpp`**：
   - `get_k`：paged + F32 + `n_stream==1` + `!v_trans` 时，`reshape_2d(k, n_embd_k_gqa, kv_size) → ggml_get_rows(k2d, row_idx) → reshape_3d(n_embd_head_k, n_head_kv, n_kv) →` 在其上开 `ggml_view_4d`（ne/nb 对齐原 view）；否则回退原连续 view。
   - `get_v` 非转置同构；`v_trans` / 量化 / 多序列 → 回退原路径。
   - `ingraph_gather_layers++`（每启用 gather 的 layer）；`row_idx` 填值时 `phys!=r` 计 `row_idx_changed`，INVALID 计 `row_idx_fail`。
   - `paged_log_stats` 追加打印三项统计（走 `LLAMA_LOG_*` / stderr）。
3. **`src/llama-graph.cpp`**：`build_attn` 内建 `row_idx` 输入张量，在 `cpy_k/cpy_v` 之后将其传入 `get_k/get_v`。

未改：attention kernel、ggml、context、`n_kv` 计算、`find_slot`/`sinfo` 生成。未新增 ggml op。

---

## 4. row_idx runtime input 创建与填值

- **创建**：与 `k_idxs` 同模式，在 `build_attn` 构图期建 `row_idx` 输入张量（I32, `ne=[n_kv]`），加入 input 集合；长度固定 `n_kv`（拓扑常量），逐 step 不变。
- **填值**：`set_input` 期由 kv-cache 按 `block_table` 填 `paged_resolve(r)`。identity 下填 `[0,n_kv)`；Stage 2B 起填非连续物理行号 —— 读端寻址自动 block-table 化。
- **时序**：填值在 `set_inputs`（compute 前），gather 节点在 compute 内 `cpy` 之后执行，读的是本 step 已写入的 cache —— 时序成立（与 shadow 必须在 compute 后同理）。

---

## 5. gather 节点位置

`build_attn` 内单次 `graph_compute` 的节点序：

```
cpy_k = ggml_set_rows(k, k_cur, k_idxs)   // 写本 step K
cpy_v = ggml_set_rows(v, v_cur, v_idxs)   // 写本 step V
// ↓ Stage 3B 插入：get_k/get_v 内部经 row_idx gather
k = get_k(ctx, il, ..., row_idx)          // reshape→get_rows(row_idx)→reshape→view
v = get_v(ctx, il, ..., row_idx)
cur = build_attn_mha(q, k, v, ...)        // FA 读 gather 后的 K/V
```

cpy（写）先于 get_rows（读）由数据依赖强制：get_rows 的源 `k2d` 与 cpy 写的 cache 同张量，set_rows 必先求值。

---

## 6. F32 / 单序列 / 非 v_trans 限制

- **F32**：`ggml_get_rows` 输出强制 F32（源码 `// TODO: implement non F32 return`）。F16/量化 K/V 会被升精度、改变 FA 输入 dtype，故首版限 F32 cache（`-ctk f32 -ctv f32`），以 sha256 守门。
- **单序列**：`n_stream==1`，避免 row_idx 多序列形状/寻址复杂度。
- **非 v_trans**：转置 V 把 `n_kv` 放 ne[0]（列），无法直接用按 ne[1] 行的 `get_rows`，首版告警一次走原路径。
- 不满足任一条件 → 回退原连续 view，行为不变。

---

## 7. 验证结果

> 均 `-fa on --temp 0 --seed 42`，确定性贪心；`-ctk f32 -ctv f32 -nkvo`。

### 7.1 Stage 3B F32 smoke

- `base_vs_off_equal=0`（默认关闭与 baseline 字节级等价）
- `base_vs_on_equal=0`（`LLAMA_KV_PAGED=1` 与 baseline 字节级等价）
- 三输出 sha256 完全一致：`5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006`

| 字段 | 值 |
|---|---|
| `ingraph_gather_layers` | 512 |
| `row_idx_changed` | 0 |
| `row_idx_fail` | 0 |
| `identity_fail` | 0 |
| `write_resolve_changed` | 0 |
| `shadow_gather_mismatch` | 0 |
| `graphs reused` | 62 |

### 7.2 lazy-tail 正交

- `base_vs_on_lazy_tail_equal=0`
- `ingraph_gather_layers=512`、`row_idx_changed=0`、`row_idx_fail=0`
- `lazy-tail failures=0`
- `graphs reused=62`
- lazy-tail `bytes=4345036800`（约 `4143.75 MiB`）—— F32 KV cache 下属正常（F32 行宽是 F16 两倍）。

---

## 8. 通过标准（全部满足）

1. ✅ `base_vs_off_equal=0`（默认关闭字节级等价）
2. ✅ `base_vs_on_equal=0` 且三 sha256 一致（paged 开启输出不变）
3. ✅ `ingraph_gather_layers=512 > 0`（读路径**确实经图内 gather 接管**）
4. ✅ `row_idx_changed=0`、`row_idx_fail=0`（identity 下 `phys==r`、无 INVALID）
5. ✅ `identity_fail=0`、`write_resolve_changed=0`、`shadow_gather_mismatch=0`（Stage 1/2A/3A 自证仍成立）
6. ✅ `graphs reused=62`（gather 节点逐 step 恒定，reuse 未破坏）
7. ✅ lazy-tail 同开仍 `base_vs_on_lazy_tail_equal=0`（正交）

判定核心：**`ingraph_gather_layers>0` 且 diff 空** —— 读路径真正走图内 `get_rows` 接管，区别于 Stage 3A-shadow（不接管）。

---

## 9. 结论

1. **默认关闭不改变行为**：`base_vs_off_equal=0`，paged 分支早退，零回归。
2. **paged 开启输出不变**：`base_vs_on_equal=0`，三 sha256 一致。
3. **graph 内 gather 确实启用**：`ingraph_gather_layers=512`，attention 真实读取 gather 后的 K/V。
4. **row_idx identity 下正确**：`row_idx_changed=0`、`row_idx_fail=0`，`phys==r`。
5. **graph reuse 未破坏**：`graphs reused=62`，gather 节点数/形状逐 step 恒定。
6. **与 lazy-tail 正交**：同开仍逐位一致，lazy-tail 统计正常。
7. **不期待 RSS 下降**：仍 identity，物理布局连续、分配不变；get_rows 输出 staging 张量由图分配。收益是「读端真实经 block-table 寻址」。
8. **解锁 Stage 2B**：读路径已 block-table 化，为 Stage 2B 非连续块式分配提供安全前置 —— 读写寻址可同时切换。

---

## 10. 下一步建议

1. **不要直接做 release/swap**（Stage 4/5），读写寻址刚通,先稳固分配语义。
2. **下一步进入 Stage 2B 非 identity block mapping 设计**：读端已 block-table 化，可安全引入非连续物理块分配。
3. **Stage 2B 仍先做最小非连续 / 受控映射**：从可控的少量非连续块起步，继续以 diff（`base_vs_on_equal=0`）+ `row_idx_changed>0` 守门 —— 此时 `row_idx_changed` 应转为非零（证非连续寻址生效），但输出仍须逐位一致。
