# Stage 2A — write path identity-consumed mapping results

前置：

- [docs/kv_paged_read_stage0_source_read.md](kv_paged_read_stage0_source_read.md)
- [docs/kv_paged_read_stage1_metadata_plan.md](kv_paged_read_stage1_metadata_plan.md)
- [docs/kv_paged_read_stage1_metadata_results.md](kv_paged_read_stage1_metadata_results.md)
- [docs/kv_paged_read_stage2a_write_identity_plan.md](kv_paged_read_stage2a_write_identity_plan.md)

本文记录 Stage 2A（write path identity-consumed mapping）的实现内容与验证结果。

---

## 1. Stage 2A 目标

让写路径**第一次真正消费** `block_table`：写入 `k_idxs/v_idxs` 前先经 `paged_write_resolve(cell)` 解析。因 `block_table` 仍是 **identity**（`[i]=i`），`phys_cell == cell`，真实写入位置不变，**输出必须与 baseline 完全一致**。

通过标准：默认关闭字节级等价；`LLAMA_KV_PAGED=1` 输出与 baseline 逐 token 一致、sha256 相同；`write_resolve_checks>0`（链路确实接通）；`write_resolve_fail=0`、`write_resolve_changed=0`（identity 下解析恒等）；`identity_fail=0`；graph reuse 不被破坏；与 lazy-tail 正交。

---

## 2. 本轮实现内容

### 2.1 修改文件（仅两个）

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`

未触碰 `get_k/get_v`、attention kernel、`llama-graph.*`、`llama-context.*`、任何 `ggml/**`，未引入 release/swap/prefetch/madvise，未改 `sinfo` 本身。

### 2.2 新增统计字段（`mutable`，紧邻 Stage 1 的 `paged_identity_*`）

| 字段 | 位置 | 含义 |
|---|---|---|
| `paged_write_resolve_checks` | `llama-kv-cache.h:461` | 写路径解析次数（应 > 0） |
| `paged_write_resolve_fail` | `:462` | 解析返回 INVALID 次数（应 = 0） |
| `paged_write_resolve_changed` | `:463` | `phys != cell` 次数（identity 下应 = 0） |

采用 `mutable` 方案，`paged_write_resolve` 声明为 `const`（`llama-kv-cache.h:444`），故可在 `set_input_*` 的 const 上下文调用，**不动 `set_input_*` 与调用方的 const 签名**，改动面最小。

### 2.3 新增函数

`paged_write_resolve(uint32_t cell) const`（`llama-kv-cache.cpp:1636`）：关闭时早退返回 `cell`；否则 `paged_resolve(cell)`，`checks++`；INVALID 则 `fail++` 回退 `cell`（防御，不破坏写入）；`phys!=cell` 则 `changed++`；返回 `phys`。

`paged_log_stats`（`:1679`）追加打印三个 write_resolve 计数。

### 2.4 三处写路径接入点

唯一的 cell index → runtime-input 张量数值出口，只替换最内层 cell 项，`offs`/`j*kv_size`/索引步进全部保持原样：

| 接入点 | 位置 | 改动 |
|---|---|---|
| `set_input_k_idxs` | `llama-kv-cache.cpp:2582` | `cell = sinfo.idxs[s][i]` → `phys = paged_write_resolve(cell)` → `data[...] = offs + phys` |
| `set_input_v_idxs` 非转置 | `:2601` | 同式替换 |
| `set_input_v_idxs` 转置分支 | `:2616` | `offs + j*kv_size + phys`，仅替换 cell 项 |

**关键**：三处全程只读 `sinfo`，解析结果只写 `dst->data`，绝不回写 `sinfo.idxs`，不改张量形状、不动 graph。

---

## 3. 验证命令与结果

> 均 `-fa on --temp 0 --seed 42`，确定性贪心解码。

### 3.1 correctness smoke

| 指标 | 结果 |
|---|---|
| `base_vs_off_equal` | `0`（baseline vs `LLAMA_KV_PAGED=0`，diff 空） |
| `base_vs_on_equal` | `0`（baseline vs `LLAMA_KV_PAGED=1`，diff 空） |
| 三输出 sha256 | `5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006`（完全一致） |

### 3.2 paged metadata 统计

```
block_size=16  n_blocks=32  blocks_in_use=5  free_blocks=27
alloc_calls=6  identity_checks=72  identity_fail=0
write_resolve_checks=144  write_resolve_fail=0  write_resolve_changed=0
```

`write_resolve_checks=144 > 0` → 写路径**确实消费了** `block_table`（链路接通）；`fail=0` 且 `changed=0` → identity 下解析恒等。

### 3.3 graph reuse

`graphs reused=62`（与 Stage 1 持平，未被破坏）。

### 3.4 lazy-tail 正交验证

| 指标 | 结果 |
|---|---|
| `base_vs_on_lazy_tail_equal` | `0`（diff 空） |
| `identity_fail` | `0` |
| `write_resolve_checks` | `144` |
| `write_resolve_fail` | `0` |
| `write_resolve_changed` | `0` |
| lazy-tail `calls` | `4160` |
| lazy-tail `bytes` | `2163998720` |
| lazy-tail `failures` | `0` |

`LLAMA_KV_PAGED=1` 与 `LLAMA_KV_LAZY_TAIL=1` 同开，输出仍与 baseline 一致，写路径解析正常、二者正交。

---

## 4. 通过标准（全部满足）

1. ✅ 默认关闭（`=0`）与 baseline diff 空；
2. ✅ `=1` 输出与 baseline diff 空，三输出 sha256 一致；
3. ✅ `write_resolve_checks=144 > 0`（写路径实际消费 block_table）；
4. ✅ `write_resolve_fail=0`（解析无 INVALID）；
5. ✅ `write_resolve_changed=0`（identity 下 `phys==cell`）；
6. ✅ `identity_fail=0`（Stage 1 自证仍成立）；
7. ✅ `graphs reused=62`（graph reuse 未被破坏）；
8. ✅ 与 lazy-tail 同开仍 diff 空（正交）。

---

## 5. 结论

1. **默认关闭不改变行为** —— `LLAMA_KV_PAGED=0` 与当前分支字节级等价。
2. **`LLAMA_KV_PAGED=1` 输出不变** —— 三输出 sha256 完全一致。
3. **写路径已实际消费 block-table** —— `write_resolve_checks=144`，`k_idxs/v_idxs` 的值经 `paged_write_resolve` 解析后落入张量，链路首次接通。
4. **identity 下解析结果仍不变** —— `write_resolve_changed=0`、`write_resolve_fail=0`，`phys==cell` 逐次成立。
5. **graph reuse 未破坏 + 与 lazy-tail 正交** —— `graphs reused=62`，记账与解析不进图、不动 `n_kv` 拓扑常量；lazy-tail 同开仍 diff 空。

**RSS 立场**：Stage 2A 仍是 identity，物理寻址与分配完全不变，**不期待也未声称 RSS 下降**。收益是"写路径消费 block_table"的链路接通，为后续阶段铺路。

---

## 6. 下一阶段选择讨论

**不建议直接做真正非连续的 Stage 2B**。原因：

- 读路径仍是连续 `get_k/get_v` 返回 `[0, n_kv)` 的 `ggml_view_4d`（[llama-kv-cache.cpp:2162 附近](../src/llama-kv-cache.cpp)），物理上要求该区间**连续驻留**；
- 若 Stage 2B 让写路径把 token 写到非连续物理块（非 identity `block_table`），而读路径仍按连续 `[0,n_kv)` 取行，**读到的就是错块** —— 静默 correctness bug；
- 写、读两端的寻址必须**同时**切到 block-table 才安全。单独切写路径会破坏一致性。

**建议下一步二选一**：

1. **优先 Stage 3 — staging paged-read 设计**（推荐）：先把读路径按 `block_table` 用 `ggml_get_rows`（或等价 cpy）gather 到**固定长度连续 staging 张量**（长度 = `GGML_PAD(n_kv, n_pad)`，保持拓扑常量），再交给现有 `get_k/get_v` view + FA。读路径具备 block-table 寻址能力后，Stage 2B 的非连续写才安全。先出设计文档（不改代码）。
2. **或 Stage 2B-dryrun**：构造非 identity `block_table`（模拟非连续），让 `write_resolve_changed>0` 被实际触发，但**只在 `paged_write_resolve` 内部 assert/统计验证映射公式**，**不接管真实写入**（写入仍用原 cell）。纯验证"非 identity 解析公式正确"，零行为变化，为 Stage 2B/3 的真接管攒信心。

**结论**：先做 Stage 3 staging paged-read 设计，把读端补齐到 block-table 寻址,再回头让 Stage 2B 真正接管非连续写。Stage 2B-dryrun 可作为低风险的并行验证项。
