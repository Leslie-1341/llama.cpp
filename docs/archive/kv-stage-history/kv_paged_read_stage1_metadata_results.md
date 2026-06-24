# Stage 1 — paged KV metadata scaffold results

前置：

- [docs/kv_paged_read_stage0_source_read.md](kv_paged_read_stage0_source_read.md)
- [docs/kv_paged_read_stage1_metadata_plan.md](kv_paged_read_stage1_metadata_plan.md)

本文记录 Stage 1（block-table metadata scaffold）的实现内容与验证结果。

---

## 1. Stage 1 目标

建立**纯元数据（metadata-only）**的 block-table scaffold，采用 **identity mapping**（`block_table[i]=i`），只构建 / 维护 / 统计 / 自证恒等，**不接管任何实际寻址**：

- 不改变真实 K/V 读写路径；
- 不改 attention kernel、不改 graph、不破坏 graph reuse；
- 不引入 release / swap / prefetch；
- **不期待 RSS 下降**。

通过标准：默认关闭字节级等价当前分支；`LLAMA_KV_PAGED=1`（identity）输出与 baseline 逐 token 一致；`identity_fail=0`；graph reuse 不被破坏；与 lazy-tail 正交。

---

## 2. 本轮实现内容

### 2.1 修改文件（仅两个）

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`

未触碰任何 `ggml/**`、`llama-graph.*`、`llama-context.*`、`llama-cparams.h`、`llama-kv-cells.h`、`llama-kv-cache-iswa.*`，符合 Stage 1 禁改边界。

### 2.2 新增 env

- `LLAMA_KV_PAGED`（未设 / =0 → 关闭）
- `LLAMA_KV_PAGED_BLOCK_SIZE`（默认 16）

### 2.3 新增 metadata 字段

`kv_paged_enabled`、`kv_paged_warned`、`paged_block_size`、`paged_n_blocks`、`paged_block_table`、`paged_block_used`、`paged_free_list`、`paged_alloc_calls`、`paged_blocks_in_use`、`paged_identity_checks`、`paged_identity_fail`。

### 2.4 新增 helper

| 函数 | 位置 | 作用 |
|---|---|---|
| `paged_init` | `llama-kv-cache.cpp:1548` | 构造期建 identity `block_table[i]=i`、初始化块池元数据 |
| `paged_reset` | `:1569` | clear 时恢复 identity 初值 |
| `paged_note_cells` | `:1587` | ubatch 提交后纯记账（标记块 used、更新统计） |
| `paged_resolve` | （声明 `.h:443`） | 按 `block_table[cell/bs]*bs + cell%bs` 解析；identity 下返回 `cell` |
| `paged_assert_identity` | `:1636` | 校验 `paged_resolve(cell)==cell`，累加 `identity_checks/fail` |
| `paged_log_stats` | `:1654` | 打印统计 |

### 2.5 挂载点

- **构造函数** 解析 env，启用则 `paged_init(kv_size)`（`:431`）；
- **`clear(bool data)`** 先 `paged_log_stats()`（`:784`）后 `paged_reset()`（`:792`）重置 metadata；
- **`llama_kv_cache_context::apply()`** 在真实提交 ubatch 后 `paged_note_cells` + `paged_assert_identity`（`:3333-3334`）；
- **`state_read_meta()`** restore 后同步 metadata（`:3631-3632`）；
- **析构函数** 打印统计。

**关键**：所有挂载点只读 `sinfo`、纯记账与 assert，**不修改任何写入行为或 sinfo 本身**。

---

## 3. 验证命令与结果

> 均 `-fa on --temp 0 --seed 42`，确定性贪心解码。

### 3.1 correctness smoke

| 指标 | 结果 |
|---|---|
| `base_vs_off_equal` | `0`（baseline vs `LLAMA_KV_PAGED=0`，diff 空） |
| `base_vs_on_equal` | `0`（baseline vs `LLAMA_KV_PAGED=1`，diff 空） |
| 三输出 sha256 | `5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006`（完全一致） |
| `identity_fail` | `0` |
| `graphs reused` | `62`（graph reuse 未被破坏） |

### 3.2 paged metadata 日志

```
block_size=16  n_blocks=32  blocks_in_use=5  free_blocks=27
alloc_calls=6  identity_checks=72  identity_fail=0
```

### 3.3 lazy-tail 正交验证

| 指标 | 结果 |
|---|---|
| `base_vs_on_lazy_tail_equal` | `0`（diff 空） |
| `identity_fail` | `0` |
| lazy-tail `calls` | `4160` |
| lazy-tail `bytes` | `2163998720` |
| lazy-tail `failures` | `0` |

`LLAMA_KV_PAGED=1` 与 `LLAMA_KV_LAZY_TAIL=1` 同开，输出仍与 baseline 一致，二者正交、互不破坏。

---

## 4. 通过标准（全部满足）

1. ✅ 默认关闭（`LLAMA_KV_PAGED=0`）与 baseline diff 空；
2. ✅ `LLAMA_KV_PAGED=1` identity 输出与 baseline diff 空，三输出 sha256 一致；
3. ✅ `identity_fail=0`（block-table 映射公式成立）；
4. ✅ `graphs reused=62`（graph reuse 未被破坏）；
5. ✅ 与 lazy-tail 同开仍 diff 空、`identity_fail=0`（正交）。

---

## 5. 结论

1. **默认关闭不改变行为** —— 关闭路径与当前分支字节级等价。
2. **`LLAMA_KV_PAGED=1` identity metadata 不改变输出** —— 三输出 sha256 完全一致。
3. **block-table 公式当前成立** —— `phys = block_table[cell/bs]*bs + cell%bs` 在 identity 下逐 cell 校验 `identity_fail=0`。
4. **graph reuse 没被破坏** —— `graphs reused=62`，metadata 记账不进图、不动 `n_kv` 拓扑常量。
5. **与 lazy-tail 正交** —— 同开不破坏输出，两套机制独立。

**RSS 立场**：Stage 1 是元数据骨架，物理寻址与分配完全不变，**不期待也未声称 RSS 下降**。

---

## 6. 注意点

- **`blocks_in_use=5` 合理**：`block_size=16`，约 70 token 的运行覆盖约 `⌈70/16⌉≈5` 个 block，与预期一致。
- **`alloc_calls=6` 不作为失败项**：可能包含生命周期内累计记账，或 warmup / context 初始化过程中的额外一次。当前核心通过指标是 **输出一致 + `identity_fail=0` + graph reuse 保持**，`alloc_calls` 仅作观测量，不参与判定。

---

## 7. 下一步：Stage 2A — write path identity-consumed mapping

让写路径**首次真正消费** block_table，但仍停在 identity，确保零行为变化：

- 只让 `cpy_k/cpy_v` 的 `k_idxs/v_idxs` 经 `paged_resolve(cell)` 解析后再写入；
- **block_table 仍保持 identity**（`[i]=i`），故解析结果恒等于原 cell index；
- **不做非 identity 物理搬移**（块仍线性、连续）；
- **不做 staging read**（读路径继续冻结，留给 Stage 3）；
- **不做 release / swap / prefetch**。

Stage 2A 的意义：把"block_table 被实际寻址消费"这条链路在 identity 下先打通并回归验证，为 Stage 2B（真正的非连续块式分配）和 Stage 3（staging paged-read）铺平接口，且每一步都用 `diff` + `identity_fail=0` 守住正确性。
