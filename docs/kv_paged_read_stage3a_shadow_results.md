# Stage 3A-shadow — shadow gather validation results

前置：

- [docs/kv_paged_read_stage3_staging_read_plan.md](kv_paged_read_stage3_staging_read_plan.md)（含 Timing update）
- [docs/kv_paged_read_stage3_timing_analysis.md](kv_paged_read_stage3_timing_analysis.md)
- [docs/kv_paged_read_stage2a_write_identity_results.md](kv_paged_read_stage2a_write_identity_results.md)

本文记录 Stage 3A-shadow 的实现与验证结果。

---

## 1. 目标

在 identity（`block_table[i]=i`）下，于 `graph_compute` **之后**按 `block_table` 把物理 K/V 行 gather 到一块独立 shadow buffer，并与直接读 cache 连续 `[0,n_kv)` 的 reference **逐字节比对**，证明 gather 公式与整行字节拷贝逻辑正确。shadow buffer **只用于比较，绝不喂给 attention**。这是为 Stage 3B graph 内 gather 真实读路径攒信心的零风险前置。

---

## 2. 为什么不是真实 staging read

Stage 3 timing analysis 已证伪「graph 外 CPU staging copy 作为真实读路径」：

- `get_k/get_v`、`cpy_k/cpy_v` 均为构图期函数，只 emit ggml 节点；
- 真实 K/V 写入（`ggml_set_rows`）与 attention 读（view→FA）**同在一次 `graph_compute` 内**，由图内数据依赖定序；
- graph 外 CPU copy 只能在 compute **前**执行，会漏掉本 step 刚写入的 K/V（读到旧 KV）；
- cpy 写节点与 attention 读节点之间**没有干净的 CPU hook**。

因此本阶段不接管读路径：真实读仍走原 `get_k/get_v` 连续 view，shadow gather 仅做 compute 后的离线比对。

---

## 3. 代码改动范围

**只修改** `src/llama-kv-cache.cpp` / `src/llama-kv-cache.h`：

1. 新增函数 `paged_shadow_validate(const slot_info & sinfo, uint32_t n_kv) const`：for `r in [0,n_kv)`，`phys=paged_resolve(r)`（INVALID→`r`，计 `fail`），整行 `memcpy` 物理行到 shadow buffer，`phys!=r` 计 `changed`；再 `memcmp(shadow_row, cache 连续行 r)`，不等计 `mismatch`；`calls++`。
2. 新增统计字段：`paged_shadow_gather_calls` / `paged_shadow_gather_changed` / `paged_shadow_gather_mismatch` / `paged_shadow_gather_fail`（`mutable`）。

**未修改**：`get_k/get_v`、attention kernel、graph/context、ggml、真实 K/V 读路径。shadow buffer 不进入 graph、不传给 FA。

---

## 4. shadow validate 挂载点与时序依据

| 位置 | 动作 |
|---|---|
| `llama_kv_cache_context::apply()` | 记录当前 ubatch 的 `n_kv`，置 `paged_shadow_pending=true` |
| `llama_kv_cache_context::next()` 开头 | 调用 `kv->paged_shadow_validate(...)` |

**时序依据**：主 decode 循环在 `process_ubatch()` 成功返回（即 `graph_compute` 已执行、本 step K/V 已由 `cpy_k/cpy_v` 节点写入 cache）**之后**才调用 `next()`。故 `next()` 开头的 shadow validate **必定位于 `graph_compute` 之后**，gather 到的是含本 step 写入的最新 cache 状态 —— 这正是 timing analysis 指出 compute 前 gather 会漏写入、必须放到 compute 后的修正。

---

## 5. 验证结果

> 均 `-fa on --temp 0 --seed 42`，确定性贪心。

### 5.1 correctness smoke

- `base_vs_off_equal=0`（默认关闭与 baseline 字节级等价）
- `base_vs_on_equal=0`（`LLAMA_KV_PAGED=1` 与 baseline 字节级等价）
- 三输出 sha256 完全一致：`5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006`

### 5.2 paged stats

| 字段 | 值 |
|---|---|
| `block_size` | 16 |
| `n_blocks` | 32 |
| `blocks_in_use` | 5 |
| `free_blocks` | 27 |
| `alloc_calls` | 6 |
| `identity_checks` | 72 |
| `identity_fail` | 0 |
| `write_resolve_checks` | 144 |
| `write_resolve_fail` | 0 |
| `write_resolve_changed` | 0 |
| `shadow_gather_calls` | 65 |
| `shadow_gather_changed` | 0 |
| `shadow_gather_mismatch` | 0 |
| `shadow_gather_fail` | 0 |

### 5.3 graph reuse

- `graphs reused=62`（与 Stage 2A 持平，未进图、view 拓扑未变）

### 5.4 lazy-tail 正交

- `base_vs_on_lazy_tail_equal=0`
- lazy-tail `calls=4160`、`bytes=2163998720`、`failures=0`
- `shadow_gather_calls=65`、`shadow_gather_changed=0`、`shadow_gather_mismatch=0`、`shadow_gather_fail=0`

---

## 6. 通过标准（全部满足）

1. ✅ `base_vs_off_equal=0`（默认关闭字节级等价）
2. ✅ `base_vs_on_equal=0` 且三 sha256 一致（paged 开启输出不变）
3. ✅ `shadow_gather_calls=65 > 0`（影子 gather 确实执行）
4. ✅ `shadow_gather_changed=0`（identity 下 `phys==r`）
5. ✅ `shadow_gather_mismatch=0`（gather 行与连续 reference 逐字节相等）
6. ✅ `shadow_gather_fail=0`、`write_resolve_changed=0`、`identity_fail=0`（Stage 1/2A 自证仍成立）
7. ✅ `graphs reused=62`（graph reuse 未下降）
8. ✅ lazy-tail 同开仍 `base_vs_on_lazy_tail_equal=0`（正交）

---

## 7. 结论

1. **默认关闭不改变行为**：`base_vs_off_equal=0`，paged 分支早退，零回归。
2. **`LLAMA_KV_PAGED=1` 输出不变**：`base_vs_on_equal=0`，三 sha256 一致。
3. **shadow gather 确实执行**：`shadow_gather_calls=65`。
4. **identity 下 gather 与 reference 逐字节一致**：`changed=0`、`mismatch=0`、`fail=0`，gather 公式与整行字节拷贝逻辑正确。
5. **graph reuse 未破坏**：`graphs reused=62`，shadow 不进图。
6. **与 lazy-tail 正交**：同开仍逐位一致，lazy-tail 统计正常。
7. **不期待 RSS 下降**：仍是 identity，物理布局连续、分配不变，且 shadow buffer 略增固定内存；收益是「gather 公式已被验证」。
8. **不是最终 paged-read**：shadow buffer 不接管真实读，只为 Stage 3B graph 内 gather 验证 gather 公式。

---

## 8. 下一步建议

1. **不做 Stage 2B 非连续写**：读端尚未真实 block-table 化，会读错块。
2. **不做 graph 外真实 staging read**：时序已证伪（见 timing analysis）。
3. **下一步进入 Stage 3B graph 内 gather 设计**：在 `build_attn` 内 `cpy_k/cpy_v` 之后、`get_k/get_v` 之前插入 gather 节点，作为唯一时序成立的真实读路径接管。重点攻克：gather 节点是否破坏 reuse、行号张量走 `set_input`、`ggml_get_rows` 对量化/转置 K/V layout 适配。
4. **Stage 2B 必须在 Stage 3B 之后**：读写寻址同时 block-table 化才安全。
