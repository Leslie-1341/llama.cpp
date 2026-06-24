# Stage 2B — non-identity block mapping 设计计划

前置：

- [docs/kv_paged_read_stage3b_ingraph_gather_results.md](kv_paged_read_stage3b_ingraph_gather_results.md)
- [docs/kv_paged_read_stage2a_write_identity_results.md](kv_paged_read_stage2a_write_identity_results.md)
- [docs/kv_paged_read_stage1_metadata_results.md](kv_paged_read_stage1_metadata_results.md)

本文是**设计文档**：只新增文档，不改源码、不 build、不跑实验、不 commit。

背景：Stage 1 完成 block-table metadata scaffold；Stage 2A 完成写路径 identity-consumed mapping（`paged_write_resolve(cell)`）；Stage 3A-shadow 验证 gather 公式；Stage 3B 完成 graph 内 gather 真实读路径接管（`ingraph_gather_layers=512`、`row_idx_changed=0`、输出与 baseline 逐位一致）。读写路径现已**双双经 block-table 的 identity 版本**：

- 写：`paged_write_resolve(cell)`；
- 读：`row_idx[r]=paged_resolve(r)`，graph 内 `ggml_get_rows` gather。

**关键源码事实（[llama-kv-cache.cpp:1617-1648](../src/llama-kv-cache.cpp#L1617-L1648)，已核实）**：`paged_resolve` 与 `paged_write_resolve` 都经**同一张 `paged_block_table[logical_block]`** 解析为 `physical_block * paged_block_size + offset`。因此**只要把 `paged_block_table` 从 identity 改成一个 permutation，读写寻址自动同时切换且天然一致**，无需分别改读写两条路径 —— 这是 Stage 2B 风险最小的支点。

---

## 1. Stage 2B 目标

设计一个**最小、受控、可回滚**的 non-identity block mapping，让逻辑 block 映射到不同物理 block，使：

1. `write_resolve_changed > 0`（写路径确实写到非 identity 物理 cell）；
2. `row_idx_changed > 0`（读路径 gather 确实从非 identity 物理行取）；
3. **输出仍与 baseline 完全一致**（读写映射用同一张表，gather 把分散物理块拉直回逻辑顺序，FA 输入逐位不变）。

判定核心：**`changed>0` 且 diff 空** —— 证明非连续寻址生效但恒等，区别于 Stage 3B（`changed==0`）。

---

## 2. 非目标（硬约束）

| # | 不做 | 理由 |
|---|---|---|
| 1 | release / madvise | Stage 4 |
| 2 | swap | Stage 4 |
| 3 | prefetch | Stage 5 |
| 4 | 复杂 free-list 复用 | 本版用静态 permutation，不动分配器 |
| 5 | 真正动态回收 | Stage 4 |
| 6 | 多序列 | `n_stream==1` |
| 7 | 量化 / `v_trans` / 多流 | 仍限 F32 KV、`!v_trans`、`n_stream==1` |
| 8 | 改 attention kernel | 复用现有 FA |
| 9 | 新增 ggml op | 复用 `ggml_get_rows` |
| 10 | 改 `n_kv` 拓扑常量 | gather 输出长度仍 `n_kv` |
| 11 | 期待 RSS 下降 | 仍全分配，只换物理块次序；本版只验证寻址正确性 |

---

## 3. 为什么现在可以做 Stage 2B

Stage 3B 之前读端按连续 `[0,n_kv)` 取物理行，若写到非连续块则**读错块（静默 correctness bug）**。Stage 3B 已让读端经 `row_idx=paged_resolve(r)` 的 graph 内 gather 取物理行，**读写寻址同时 block-table 化**。此时把 `paged_block_table` 改成 permutation：

- 写：`paged_write_resolve(cell)` 经表写到物理 cell；
- 读：`row_idx[r]=paged_resolve(r)` 经**同一张表** gather 回逻辑序；

二者用同一映射，gather 自动「拉直」，输出恒等。这正是 timing analysis 与 Stage 3B 反复强调「Stage 2B 必须在读端 block-table 化之后」的兑现点。

---

## 4. non-identity mapping 候选策略

| 策略 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| A 固定 permutation | `block_table` 为一个固定置换（如逆序） | changed 最大化、稳定 | 与 prompt 写入次序无直接关系，需确保覆盖 |
| B block offset 平移 | `block_table[i] = (i + k) % n_blocks` | 实现极简、易回滚、稳定 | 偏移跨界需取模，验证 wrap |
| C 只交换两个 block | `swap(table[a], table[b])` | 改动面最小、影响可定位 | changed 计数偏小，覆盖弱 |
| D 保留 block 0 的平移 | block 0 不动，`block_table[i]=1+((i-1+k)%(n_blocks-1))`（i≥1） | 隔离 prompt 首块、降低初始写冲突风险 | 公式略复杂 |

---

## 5. 推荐最小策略

**推荐策略 D：保留 block 0 的环形平移**（首版 `k=1`，即「block 0 固定，block 1..n-1 整体环移一位」）。

```
phys_block(i) = (i == 0) ? 0
                         : 1 + ((i - 1 + K_SHIFT) % (paged_n_blocks - 1))   // K_SHIFT=1
```

理由：

1. **保留 block 0** → prompt 最初写入的逻辑 block 0 仍落物理 block 0，降低与 reserve/初始写入交互的风险（对应重点 §3）；
2. **环形平移** → 天然是 permutation（双射），物理块两两不重不漏，**不越界**（值域恒在 `[0, n_blocks)`，对应重点 §2）；
3. 对 `i≥1` 的逻辑 block 全部 `phys!=i` → `write_resolve_changed>0`、`row_idx_changed>0` 稳定触发（对应重点 §7/§8）；
4. **静态、可回滚**：由环境变量（如 `LLAMA_KV_PAGED_SHIFT`）控制，默认 0（=identity，退回 Stage 3B 行为），设为非 0 才启用 Stage 2B 映射；
5. 逐 step 不变 → 映射稳定，不破坏 graph reuse（对应重点 §13 风险）。

退路：若 D 在边界（`n_blocks==1`）异常，回退策略 C（交换两块）或直接 `K_SHIFT=0`（identity）。

---

## 6. 写路径设计

- `find_slot` / `sinfo.idxs` **仍保留逻辑 cell**（不改分配，对应重点 §4）—— Stage 2B 不动 slot 生成。
- `set_input_k_idxs/v_idxs` 经 `paged_write_resolve(logical_cell)` → `paged_resolve(logical_cell)` → `block_table[logical_cell/bs]*bs + logical_cell%bs` 写到 physical cell（对应重点 §5）。
- block_table 改为策略 D 的 permutation 后，`logical_block>=1` 的 cell 自动写到非 identity 物理 cell → `paged_write_resolve_changed`（`phys!=cell`）从 0 变 >0（对应重点 §7）。
- **无需改写路径代码**：写路径已消费 block_table，只是表内容变了。

---

## 7. 读路径设计

- `row_idx[r] = paged_resolve(r)`（graph 内 gather 行号），经**同一张** block_table 解析。
- block_table permutation 后，`r` 所属 `logical_block>=1` 时 `phys!=r` → `row_idx_changed`（填值时计 `phys!=r`）从 0 变 >0（对应重点 §8）。
- gather：`ggml_get_rows(k2d, row_idx)` 把物理行 `phys` 取到逻辑位置 `r` → **拉直回逻辑顺序** → FA 读到的逻辑序与 baseline 逐位一致。
- **无需改读路径代码**：Stage 3B 的 gather 链路已就位，只是 `row_idx` 值变了。

读写共用 `paged_resolve`/同一 block_table 是输出恒等的根本保证（对应重点 §6 与风险「读写映射不一致」）。

---

## 8. metadata / stats 设计

`identity_fail` 处理（对应重点 §9）：Stage 1 的 `paged_identity_check` 断言 `phys==cell`。Stage 2B 下非 identity 是**预期**，故：

- **不要让 identity_check 误报失败**：将其改为**条件断言** —— `K_SHIFT==0` 时仍校验 `phys==cell`（identity 自证）；`K_SHIFT!=0` 时改为校验「映射是合法 permutation 且不越界」，统计 `logical_to_physical_checks` / `logical_to_physical_fail`，**不再要求 `phys==cell`**。
- 保留 `identity_fail` 字段语义仅在 `K_SHIFT==0` 生效，避免破坏既有 Stage 1/3B 守门。

新增统计字段（对应重点 §10）：

| 字段 | 含义 |
|---|---|
| `non_identity_enabled` | `K_SHIFT!=0` 时 1，否则 0 |
| `block_mapping_changed` | block_table 中 `table[i]!=i` 的 block 数 |
| `mapping_oob_fail` | 映射结果 `>=n_blocks` 或重复（非法 permutation）次数，**必须 0** |
| `logical_to_physical_checks` | 每次 resolve 经表解析的计数 |
| `logical_to_physical_fail` | resolve 返回 INVALID / OOB 次数，**必须 0** |

`row_idx_changed` / `write_resolve_changed`（已存在）在 Stage 2B 下应 >0。

---

## 9. correctness 验证命令

> 均 `-fa on --temp 0 --seed 42`，**必须 `-ctk f32 -ctv f32 -nkvo`**（F32 KV，no kv offload）。本设计阶段不跑，编码后执行。

```bash
MODEL=...; PROMPT="..."
# baseline
./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on --seed 42 --temp 0 \
    -ctk f32 -ctv f32 -nkvo > /tmp/base.txt
# 默认关闭
LLAMA_KV_PAGED=0 ./build/bin/llama-cli ... -ctk f32 -ctv f32 -nkvo > /tmp/off.txt
diff /tmp/base.txt /tmp/off.txt                       # 必须空
# paged 开启 + 非 identity 映射
LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1 ./build/bin/llama-cli ... \
    -ctk f32 -ctv f32 -nkvo > /tmp/on.txt 2>/tmp/on.log
diff /tmp/base.txt /tmp/on.txt                        # 仍必须空
sha256sum /tmp/base.txt /tmp/off.txt /tmp/on.txt      # 三者一致
grep -E 'paged|resolve|gather|mapping|row_idx' /tmp/on.log
```

---

## 10. 通过标准（全部满足）

1. ✅ `base_vs_off_equal=0`（默认关闭字节级等价）；
2. ✅ `base_vs_on_equal=0` 且三 sha256 一致（**非 identity 映射下输出仍逐位不变**）；
3. ✅ `write_resolve_changed > 0`（写路径确实写非 identity 物理 cell）；
4. ✅ `row_idx_changed > 0`（读路径确实从非 identity 物理行 gather）；
5. ✅ `ingraph_gather_layers > 0`（读路径仍经图内 gather）；
6. ✅ `mapping_oob_fail == 0` 且 `logical_to_physical_fail == 0`（映射合法、无越界）；
7. ✅ `graphs reused == 62`（映射逐 step 稳定，reuse 未破坏）；
8. ✅ `non_identity_enabled == 1`、`block_mapping_changed > 0`（映射确实非 identity）。

判定核心：**`write_resolve_changed>0` 且 `row_idx_changed>0` 且 diff 空** —— 非连续寻址生效但恒等。

---

## 11. 风险表

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| S2B-R1 | 写读映射不一致 → 读错块、输出乱 | 高 | 读写**共用** `paged_resolve`/同一 `block_table`，源码层面单一映射源；diff 守门 |
| S2B-R2 | `row_idx` 与 `write_resolve` 用了不同 mapping | 高 | 二者都只调 `paged_resolve`，不各自实现映射；代码审查确认无第二处映射 |
| S2B-R3 | 物理块越界 / 非法 permutation | 高 | 策略 D 是环形双射，值域恒 `[0,n_blocks)`；`mapping_oob_fail` 守门必 0 |
| S2B-R4 | prompt/decode 阶段 `n_kv` 增长致 mapping 不稳定 | 中 | block_table 长度=`n_blocks`（按 kv_size 一次性建），与 `n_kv` 增长无关；映射静态，逐 step 不变 |
| S2B-R5 | gather 节点 / 映射变动破坏 graph reuse | 中 | 映射静态（建图前定），`row_idx` 仍 I32 长度 `n_kv` input；拓扑不变，守门 reused=62 |
| S2B-R6 | F32 限制（get_rows 强制 F32） | 中 | 沿用 Stage 3B 限制，`-ctk f32 -ctv f32`；F16/量化留后续 |
| S2B-R7 | 保留 block 0 仍与 reserve 初始写交互 | 低 | 策略 D 固定 block 0；若仍异常，回退 `K_SHIFT=0` |

**RSS 立场**：仍全量分配，只是物理块次序变 permutation，**不期待 RSS 下降**；本版只验证「非连续寻址正确」。RSS 收益要等 Stage 4 release/swap。

---

## 12. Codex 实现提示词草案（暂不改代码）

> **任务：在 `src/llama-kv-cache.{h,cpp}` 实现 Stage 2B non-identity block mapping。把 `paged_block_table` 从 identity 改为受控环形平移（策略 D），使读写寻址同时非 identity 但输出与 baseline 逐位一致。**
>
> **关键事实（已核实）**：`paged_resolve`（[:1617](../src/llama-kv-cache.cpp#L1617)）与 `paged_write_resolve`（[:1636](../src/llama-kv-cache.cpp#L1636)）**共用同一张 `paged_block_table`**，解析为 `physical_block*paged_block_size+offset`。故只改 block_table 内容即让读写同时切换且一致，**勿新增第二处映射**。
>
> **硬约束**：① 只改 `llama-kv-cache.{h,cpp}`，禁改 attention kernel / ggml / graph / context，不新增 ggml op；② 不做 release/swap/prefetch/动态回收/多序列；③ 仍限 F32 KV、`!v_trans`、`n_stream==1`；④ `n_kv` 拓扑常量不变；⑤ `LLAMA_KV_PAGED=0` 字节级等价；⑥ 映射由 `LLAMA_KV_PAGED_SHIFT` 控制，默认 0 = identity（退回 Stage 3B）。
>
> **实现**：
> 1. 读 `LLAMA_KV_PAGED_SHIFT`（默认 0）。建表（[:1558](../src/llama-kv-cache.cpp#L1558) 附近）时：`SHIFT==0` → `block_table[i]=i`（identity）；`SHIFT!=0` → `block_table[0]=0`，`i>=1` → `1+((i-1+SHIFT)%(n_blocks-1))`。建表后校验是合法 permutation（无重复、无越界），违例 `mapping_oob_fail++`。
> 2. 新增统计 `non_identity_enabled` / `block_mapping_changed`（`table[i]!=i` 计数）/ `mapping_oob_fail` / `logical_to_physical_checks` / `logical_to_physical_fail`（`mutable`）。
> 3. `paged_identity_check`（[:1660](../src/llama-kv-cache.cpp#L1660)）改条件：`SHIFT==0` 仍断言 `phys==cell`（identity_fail 语义保留）；`SHIFT!=0` 改为校验 permutation 合法性 + 统计 `logical_to_physical_*`，**不再要求 phys==cell**。
> 4. `paged_log_stats`（[:1775](../src/llama-kv-cache.cpp#L1775)）追加打印新字段（走 `LLAMA_LOG_*` / stderr）。
> 5. 不改 `get_k/get_v`、`set_input_k_idxs/v_idxs`、`row_idx` 填值逻辑 —— 它们已消费 block_table，自动随表变化。
>
> **验证**（见 §9/§10）：`-ctk f32 -ctv f32 -nkvo`；`base_vs_off`/`base_vs_on` diff 全空、三 sha256 一致；`write_resolve_changed>0`、`row_idx_changed>0`、`ingraph_gather_layers>0`、`mapping_oob_fail=0`、`logical_to_physical_fail=0`、`graphs reused=62`、`non_identity_enabled=1`。
>
> **对照（只读，勿照搬）**：`/root/oscomp/llama.cpp-paged-ref` 的 `calculate_global_slot_index`（`block_table[bid]*block_size+offset`）作映射公式参考，不引入其 fused op。
