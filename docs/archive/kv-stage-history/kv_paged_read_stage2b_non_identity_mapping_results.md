# Stage 2B — non-identity block mapping results

前置：

- [docs/kv_paged_read_stage2a_write_identity_results.md](kv_paged_read_stage2a_write_identity_results.md)
- [docs/kv_paged_read_stage3a_shadow_results.md](kv_paged_read_stage3a_shadow_results.md)
- [docs/kv_paged_read_stage3b_ingraph_gather_results.md](kv_paged_read_stage3b_ingraph_gather_results.md)
- [docs/kv_paged_read_stage2b_non_identity_mapping_plan.md](kv_paged_read_stage2b_non_identity_mapping_plan.md)

本文记录 Stage 2B（受控 non-identity block mapping）的实现与验证结果。

---

## 1. Stage 2B 目标

把 `paged_block_table` 从 **identity** 切到一个**受控的 non-identity permutation**，让写路径真正写到非 identity 物理 cell、读路径真正从非 identity 物理行 gather，但**输出仍与 baseline 完全一致**。

这是 paged KV 首次让 `phys != cell`、`phys != r` **同时**成立的阶段：寻址被真正打散，正确性只能靠读写共用同一张映射表来保证。Stage 2B 只证「非连续寻址端到端成立且无回归」，不触碰物理分配/释放。

---

## 2. 为什么 Stage 2B 必须在 Stage 3B 之后做

Stage 2A 的结论已写明：单独切写路径会留下静默 correctness bug —— 写端把 token 落到非连续物理块，而读端仍按连续 `[0,n_kv)` 取行，就会读到错块。

Stage 3B 之前，读路径还没有 block-table 寻址能力（attention 直接 view 连续区间）。**只有 Stage 3B 把读端改成 graph 内 `ggml_get_rows(row_idx)` 之后**，读写两端才能由同一张 `block_table` 同时寻址。因此 non-identity mapping（Stage 2B）必须排在 in-graph gather（Stage 3B）之后 —— 读端先具备寻址能力，写端的非连续写才安全。

---

## 3. non-identity mapping 策略

- 新增环境变量 `LLAMA_KV_PAGED_SHIFT`，默认 `0`（identity，`block_table[i]=i`）。
- `LLAMA_KV_PAGED_SHIFT=1` 时启用受控 permutation：**保留 block 0 不动，其余 block 做环形平移**（ring shift）。block 0 固定可避免动到含前缀/特殊 token 的首块，平移其余块即可制造可控的非连续映射。
- permutation 是 block 级而非 cell 级：一个 block 内的 cell 仍连续，块与块之间被打散，足以让 `phys != cell` 大量发生而不引入越界。

---

## 4. 读写共用同一张 block_table

读、写路径解析的是**同一张** `paged_block_table`，这是 non-identity 下正确性的根基：

- 写路径：`paged_write_resolve(cell)` → 写入 `k_idxs/v_idxs` 时落到物理 cell。
- 读路径：`row_idx[r] = paged_resolve(r)` → graph 内 `ggml_get_rows(row_idx)` 从同一物理行 gather。

写到哪个物理 cell，读就从哪个物理行取。两端共用同一 permutation 是「打散后仍逐位等价」的唯一保证 —— 任何一端用了不同映射都会立刻被 sha256 守门发现。

---

## 5. 本轮实现内容

### 5.1 修改文件（仅两个）

- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`

### 5.2 新增环境变量

- `LLAMA_KV_PAGED_SHIFT`：默认 `0`（identity）；`1` 启用 non-identity ring-shift permutation。

### 5.3 新增统计字段

| 字段 | 含义 |
|---|---|
| `non_identity_enabled` | non-identity mapping 是否启用（shift=1 时为 1） |
| `block_mapping_changed` | block_table 中 `[i]!=i` 的条目数 |
| `mapping_oob_fail` | permutation 解析越界次数（应 = 0） |
| `logical_to_physical_checks` | 逻辑→物理映射校验次数 |
| `logical_to_physical_fail` | 映射校验失败次数（应 = 0） |
| `shadow_skipped_non_identity` | shadow validate 因 non-identity 跳过 memcmp 的次数 |

### 5.4 non-identity 下的三处行为修正

1. **`identity_fail` 语义收窄**：non-identity 下 `phys != cell` 是**预期**结果，不再计为失败。`identity_fail` 仅在 identity 模式下仍校验恒等。
2. **shadow validate 跳过连续物理行 memcmp**：shadow 比对原假设物理行与逻辑行连续对应；non-identity 下该假设不成立，对这类比对跳过 memcmp（计入 `shadow_skipped_non_identity`），避免误报 mismatch。
3. **lazy-tail 禁用**：non-identity mapping 破坏「物理连续前缀」假设，lazy-tail 被强制禁用（见 §8）。

---

## 6. 验证结果

> 均 `-fa on --temp 0 --seed 42`，确定性贪心；`-ctk f32 -ctv f32 -nkvo`。

参考 baseline sha256：`5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006`

### 6.1 identity 回归（`LLAMA_KV_PAGED_SHIFT=0`）

| 字段 | 值 |
|---|---|
| `base_vs_identity_equal` | `0`（diff 空） |
| sha256 | `5c16d81e…931e4006`（一致） |
| `non_identity_enabled` | `0` |
| `block_mapping_changed` | `0` |
| `write_resolve_changed` | `0` |
| `row_idx_changed` | `0` |
| `identity_fail` | `0` |
| `ingraph_gather_layers` | `512` |
| `graphs reused` | `62` |

非 identity 关闭时，行为与 Stage 3B identity 完全一致 —— 新代码默认零回归。

### 6.2 non-identity smoke（`LLAMA_KV_PAGED_SHIFT=1`）

| 字段 | 值 |
|---|---|
| `base_vs_off_equal` | `0`（diff 空） |
| `base_vs_shift1_equal` | `0`（diff 空） |
| 三输出 sha256 | `5c16d81e…931e4006`（完全一致） |
| `non_identity_enabled` | `1` |
| `block_mapping_changed` | `31` |
| `write_resolve_changed` | `108` |
| `row_idx_changed` | `15600` |
| `ingraph_gather_layers` | `512` |
| `mapping_oob_fail` | `0` |
| `logical_to_physical_fail` | `0` |
| `identity_fail` | `0` |
| `graphs reused` | `62` |
| `shadow_gather_mismatch` | `0` |
| `shadow_skipped_non_identity` | `1064960` |

`block_mapping_changed=31`、`write_resolve_changed=108`、`row_idx_changed=15600` **同时 > 0** → 写端确实写到非 identity 物理 cell、读端确实从非 identity 物理行 gather；而 `base_vs_shift1_equal=0` 且 sha256 一致 → 寻址被打散后输出仍逐位等价。`mapping_oob_fail=0`、`logical_to_physical_fail=0` → permutation 无越界、映射校验全过。

### 6.3 lazy-tail 禁用保护（`LLAMA_KV_PAGED_SHIFT=1` + lazy-tail 同开）

| 字段 | 值 |
|---|---|
| `base_vs_shift1_lazy_tail_equal` | `0`（diff 空） |
| `non_identity_enabled` | `1` |
| `write_resolve_changed` | `108` |
| `row_idx_changed` | `15600` |
| `shadow_gather_mismatch` | `0` |
| `mapping_oob_fail` | `0` |
| `logical_to_physical_fail` | `0` |
| `graphs reused` | `62` |

日志出现：

```
KV lazy-tail disabled because paged non-identity mapping breaks physical-prefix assumption
```

即便请求开启 lazy-tail，non-identity 下它被自动禁用并打印原因，输出仍逐位等价。

---

## 7. shadow validate 修正

Stage 3A-shadow 的逐字节比对假设：物理行 `r` 与逻辑行 `r` 连续一一对应，可与连续物理 row 直接 memcmp。non-identity 下 `phys = paged_resolve(r) != r`，该假设被打破 —— 若仍按连续行比对，会把「正确的非连续映射」误报成 mismatch。

修正：non-identity 下，对依赖连续物理行假设的那部分 memcmp **整体跳过**，并计入 `shadow_skipped_non_identity`（本轮 `1064960`）。真正经 block-table gather 的比对仍保留，故 `shadow_gather_mismatch` 仍为 `0` —— 证明 gather 公式本身没问题，只是跳过了对 non-identity 已不成立的连续行假设的比对。

---

## 8. lazy-tail 禁用保护

lazy-tail 的优化建立在「物理连续前缀」假设上：cache 的有效内容驻留在连续物理区间的前缀，尾部可惰性处理。non-identity mapping 把物理块环形平移，有效内容散布在非连续物理块中，连续前缀假设不再成立。

保护策略：检测到 non-identity（`non_identity_enabled=1`）时**强制禁用 lazy-tail**，并打印：

```
KV lazy-tail disabled because paged non-identity mapping breaks physical-prefix assumption
```

§6.3 验证：即使同时请求 lazy-tail，禁用保护生效后输出仍与 baseline 逐位一致，且 `write_resolve_changed`/`row_idx_changed` 仍为非连续值 —— non-identity 寻址正常工作，lazy-tail 被安全旁路而非破坏正确性。

---

## 9. 通过标准（全部满足）

1. ✅ identity 回归 `base_vs_identity_equal=0`，`non_identity_enabled=0`、`block_mapping_changed=0`，sha256 一致（默认零回归）；
2. ✅ non-identity `base_vs_shift1_equal=0` 且三 sha256 一致（打散后输出不变）；
3. ✅ `block_mapping_changed=31`、`write_resolve_changed=108`、`row_idx_changed=15600` 同时 > 0（读写非连续寻址确实生效）；
4. ✅ `mapping_oob_fail=0`、`logical_to_physical_fail=0`（permutation 无越界、映射校验全过）；
5. ✅ `identity_fail=0`（语义收窄后，non-identity 的 `phys!=cell` 不再误判）；
6. ✅ `shadow_gather_mismatch=0`、`shadow_skipped_non_identity=1064960`（shadow 修正生效，gather 公式仍正确）；
7. ✅ lazy-tail 同开时被禁用并 `base_vs_shift1_lazy_tail_equal=0`（禁用保护生效）；
8. ✅ `graphs reused=62`（graph reuse 未破坏）。

判定核心：**`block_mapping_changed` / `write_resolve_changed` / `row_idx_changed` 同时非零，而 diff 仍空** —— 寻址真被打散，输出仍逐位等价。

---

## 10. 结论

1. **non-identity mapping 生效**：`non_identity_enabled=1`、`block_mapping_changed=31`，block_table 已是非 identity permutation。
2. **写路径确实写到非 identity 物理 cell**：`write_resolve_changed=108`。
3. **读路径确实从非 identity 物理行 gather**：`row_idx_changed=15600`，`ingraph_gather_layers=512`。
4. **输出仍与 baseline 完全一致**：`base_vs_shift1_equal=0`，三 sha256 同为 `5c16d81e…931e4006` —— 读写共用同一 block_table 保证了打散后的逐位等价。
5. **graph reuse 未破坏、映射无越界、不期待 RSS 下降**：`graphs reused=62`，`mapping_oob_fail=0`；仍未触碰物理分配/释放，物理块只是被重排而非回收，故**不期待也未声称 RSS 下降** —— block-aware release / swap 留待下一阶段。

---

## 11. 下一步建议

1. **进入 Stage 4 设计**：Stage 2B 已端到端证明非连续寻址成立，Stage 4 的目标**不再是继续证明寻址**，而是开始设计 **block-aware release / madvise / swap** —— 真正回收/换出不再使用的物理块，这才是 RSS 收益的来源。
2. **彻底放弃旧 lazy-tail 的物理连续前缀假设**：Stage 2B 已证明该假设在 non-identity 下不成立。Stage 4 的释放/换出语义必须建立在 block-table 寻址之上，按块粒度判断驻留/回收，而非连续前缀。
