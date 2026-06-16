# Stage 3 — staging paged-read 设计计划

前置：

- [docs/kv_paged_read_stage0_source_read.md](kv_paged_read_stage0_source_read.md)
- [docs/kv_paged_read_stage1_metadata_plan.md](kv_paged_read_stage1_metadata_plan.md)
- [docs/kv_paged_read_stage1_metadata_results.md](kv_paged_read_stage1_metadata_results.md)
- [docs/kv_paged_read_stage2a_write_identity_plan.md](kv_paged_read_stage2a_write_identity_plan.md)
- [docs/kv_paged_read_stage2a_write_identity_results.md](kv_paged_read_stage2a_write_identity_results.md)

Stage 1 已完成 block-table metadata scaffold（identity `block_table[i]=i`、physical block pool、free-list、`paged_resolve` 公式、`identity_fail=0`）。Stage 2A 已完成写路径 identity-consumed mapping（`set_input_k_idxs/v_idxs` 经 `paged_write_resolve(cell)` 消费 block_table，`write_resolve_checks=144>0`、`changed=0`、输出与 baseline sha256 一致）。

**当前缺口**：写路径已能按 block_table 解析 cell，但**读路径 `get_k/get_v` 仍连续读取 `[0,n_kv)`**。因此无法直接进入真正非 identity 的 Stage 2B —— 若 K/V 写到非连续物理块、而读路径仍按连续行号取，**会读错块（静默 correctness bug）**。Stage 3 补齐读端到 block-table 寻址，是 Stage 2B 之前的必要前置。

本文是**设计文档**：只新增文档，不改源码、不 build、不 commit。

---

## 1. Stage 3 目标

1. 设计 **staging paged-read**：按 `block_table` 把需要读取的物理 K/V block **gather 到一个连续 staging K/V buffer**，再把这个连续 buffer 喂给现有 attention / FA 路径；
2. 读端因此**具备 block-table 寻址能力**：物理块可以非连续，但喂给 kernel 的始终是连续 staging；
3. 第一版仍以 **identity mapping** 验证 —— gather 后 staging 与原 `[0,n_kv)` 逐位相等，输出与 baseline 完全一致；
4. **staging buffer 形状固定**（长度 = 现有 `n_kv` 拓扑常量），不破坏 graph reserve / reuse；
5. 不污染默认路径，`LLAMA_KV_PAGED=0` 行为字节级不变；
6. 为 Stage 2B（真正非连续块分配）铺好"读端已 block-table 化"的安全接口。

---

## 2. 非目标（硬约束）

| # | 不做 | 理由 |
|---|---|---|
| 1 | 不新增 ggml op | 复用现有算子（view / cpy / get_rows） |
| 2 | 不改 attention kernel | 复用现有 FA |
| 3 | 不引入 full PagedAttention | 参考分支的重型 `GGML_OP_PAGED_ATTN` 不做 |
| 4 | 不引入 scheduler / continuous batching | 多序列专用，超范围 |
| 5 | 不做 release / swap / prefetch | Stage 4/5，且不与 lazy madvise 抢占 |
| 6 | 不做真正非 identity block allocation | Stage 2B 才动；本阶段 `block_table[i]=i` |
| 7 | 第一版仅 identity 验证 | 先证 gather 链路恒等，再谈非连续 |
| 8 | 不改变 `n_kv` 拓扑常量 | staging 长度 = `n_kv`，否则破坏 reserve/reuse |
| 9 | `LLAMA_KV_PAGED=0` 行为不变 | 零回归底线，paged 关闭时 `get_k/get_v` 原样返回 |
| 10 | 不改 `find_slot`/`sinfo` 生成 | 只读 sinfo |

**关键边界**：Stage 3 只在 `kv_paged_enabled` 且单序列 `!v_trans` 下，于读端插入"gather 到 staging → 用 staging 出 view"这一层；关闭时 `get_k/get_v` 走原路径，零差异。

---

## 3. 当前读路径问题

已读源码确认（[llama-kv-cache.cpp:2335](../src/llama-kv-cache.cpp#L2335) / [:2367](../src/llama-kv-cache.cpp#L2367)）：

- `get_k` 返回 `ggml_view_4d(ctx, k, n_embd_head_k, n_head_kv, n_kv, ns, …)`：在 layer 的 K 张量上开一个 **从 cell 行 0 起、连续 `n_kv` 行**的窗口（stream 基址偏移 `…*sinfo.s0`，approx-swap 时再加 `byte_offset`）。
- `get_v` 非转置分支同构；转置分支把 `n_kv` 放到 ne[0]，行列互换。
- **核心**：第 3 维（K）/ 对应维（V）就是 `n_kv` 个**物理连续 cell 行**。view 的本质是"物理行号 r ↔ 逻辑读位置 r"，**identity**。
- `n_kv` 是拓扑常量：graph reserve（`get_reserve_n_kv` [:2322](../src/llama-kv-cache.cpp#L2322)）与 reuse 都依赖它不随 step 变。

**问题**：view 假设逻辑读位置 r 的数据**就在物理行 r**。一旦 Stage 2B 让 `block_table` 非 identity，cell `c` 的数据被写到物理行 `paged_resolve(c) != c`，而 view 仍读物理行 `c` —— 读错块。读端必须改成"逻辑位置 r → 经 block_table 找到物理行 → 取该行"。

---

## 4. staging read 核心思想

在读端与 kernel 之间插入一层**固定长度连续 staging K/V buffer**：

```
逻辑读窗口 [0, n_kv)
   │  for each logical row r:
   │     phys = block_table[r/bs]*bs + r%bs     // identity 下 phys==r
   ▼
staging_k[r] = K_physical[phys]                 // gather（identity 下逐位等于原 view）
staging_v[r] = V_physical[phys]
   │
   ▼
get_k/get_v 返回 staging 上的 view → FA kernel
```

- staging 长度固定 = `n_kv`（拓扑常量），形状与原 view 完全一致，**kernel 无感**；
- identity 下 `phys==r`，staging 内容逐位等于"直接读物理 `[0,n_kv)`"，故输出与 baseline 一致；
- 非 identity（Stage 2B）下，gather 自动把分散物理块"拉直"成连续逻辑序，kernel 仍读连续 staging —— **读端寻址正确性由 gather 保证，kernel 不必懂分页**。

这正是"用一次 gather 把 block-table 寻址限制在读端局部，换取 kernel / graph 拓扑零改动"的取舍。

---

## 5. 方案 A：graph 内 staging gather

在 build_attn 构图时，用 `ggml_get_rows`（或等价 cpy）在图里加一个 gather 节点：以一个 I32/I64 行号张量（值 = `paged_resolve(r)`）从 layer K/V gather 出固定 `n_kv` 行到 staging 张量，`get_k/get_v` 改为在 staging 上开 view。

- **优点**：gather 进入图，享受后端调度 / offload，无显式 CPU copy；与现有"图内一切"的风格一致。
- **风险**：
  1. **拓扑敏感**：新增 gather 节点改变 graph 结构，可能破坏 graph reuse（reuse 比对节点序列）；需确认节点数 / 形状逐 step 恒定。
  2. **行号张量是 runtime input**：和 `k_idxs` 一样要 `set_input` 填值，多一条输入链路；identity 下值 = `[0,n_kv)`。
  3. **`ggml_get_rows` 语义/类型限制**：对量化 K、转置 V 的支持需逐一核对，可能不覆盖所有 layout。
  4. **4D view 与 get_rows 维度**：`get_rows` 按 ne[1] 行 gather，需把 K/V 先 reshape 成 2D 行集再 gather 再 reshape，构图复杂度上升。
  5. 与 approx-swap 的 `visible_lo`/`byte_offset` 路径叠加时交互复杂。

graph 内方案是 Stage 3 的"终态目标"，但首版引入它会同时动"图结构 + runtime input + 量化/转置兼容"三处，回归面大。

---

## 6. 方案 B：graph 外 CPU staging copy

在 graph compute **之前**（set_inputs 阶段，CPU 侧），按 `block_table` 把 layer K/V 的物理行**逐行 memcpy 到一块固定 staging K/V buffer**；paged 模式下 `get_k/get_v` 改为在 **staging buffer** 上开与原来形状完全相同的 view，喂给 FA。

- **优点**：
  1. **图拓扑零改动**：不新增任何节点，`get_k/get_v` 仍返回同形状 view，只是底层 buffer 换成 staging → **graph reuse 天然不受影响**；
  2. **最直观可控**：gather 是普通 CPU 循环 + memcpy，identity 下 `phys==r`，逐行拷贝结果逐位等于原物理行；
  3. **量化 / 转置无感**：按 `row_size` 字节整行拷贝，不关心 dtype；
  4. **改动面最小**：集中在 kv-cache 一处，关闭时早退。
- **风险**：
  1. **多一次 CPU copy**：每 step 拷 `n_kv * row_size` 字节 K + V；identity 下是纯开销（性能下降），但**正确性优先**，且第一版只验证正确性；
  2. staging buffer 需在 kv-cache 内额外分配 `n_kv * row_size`（K、V 各一），内存上升固定常量；
  3. 若后端非 CPU（offload），staging 在 host、layer K/V 在 device，需注意 backing buffer 位置 —— 目标场景是 CPU 推理，首版限定 CPU/host buffer。

方案 B 是"以一次确定性 CPU copy 换取图拓扑与 kernel 的零改动"，把 Stage 3 的风险压到最小。

---

## 7. 推荐方案

**首版（Stage 3A）采用方案 B —— graph 外 CPU staging copy。**

判断依据（对应任务 6 问）：

1. **哪个更适合比赛最小原型**：方案 B。它不动图、不动 kernel、不引入 runtime input 链路，单点可控，回归面最小，最契合"每阶段 diff 守门"的推进方式。
2. **第一版是否先做 graph 外 CPU copy**：是。先用 CPU copy 把"读端经 block_table 取物理行"这条链路在 identity 下打通并回归，证明 gather 公式正确、输出不变；性能优化（进图）留待方案 A 作为 Stage 3B。
3. **graph 外 copy 最小改哪些文件**：仅 `src/llama-kv-cache.cpp` / `src/llama-kv-cache.h`（加 staging buffer 字段、gather 函数、`get_k/get_v` 的 paged 分支）。不碰 graph/context/ggml/kernel。
4. **如何保证 identity 下输出一致**：`block_table[i]=i` → `phys==r` → staging 逐行拷贝结果逐位等于原 `[0,n_kv)` 物理行 → view 形状不变 → FA 输入逐位相同 → 输出 sha256 一致。加 `paged_read_gather_changed`（`phys!=r` 次数，identity 应 =0）自证。
5. **如何验证 graph reuse**：`get_k/get_v` 返回的 view 形状 / `n_kv` / 节点序列完全不变，故 `graphs reused` 应与 Stage 2A 持平（62）；以日志比对守门。
6. **哪些必须推迟**：进图 gather（方案 A，Stage 3B）；真正非 identity 块分配（Stage 2B，需在 Stage 3 之后）；release/swap/prefetch（Stage 4/5）；量化/转置/offload 的完整覆盖（首版限 CPU、`!v_trans`、单序列）。

---

## 8. 最小实现路线 Stage 3A

**定义**：identity 下，paged 模式 `get_k/get_v` 改读"经 CPU gather 填好的固定 staging buffer"，输出与 baseline 字节级一致。

1. **新增 staging 字段**（`llama-kv-cache.h`，紧邻 Stage 2A 字段）：每 layer 一块 `staging_k` / `staging_v`（或单块按 layer 复用），大小 `n_kv * row_size`，固定不随 step 变；统计 `paged_read_gather_calls` / `paged_read_gather_changed`（`mutable`）。
2. **新增 `paged_gather_read(il, n_kv, sinfo)`**：paged 关闭早退；否则 for `r in [0,n_kv)`：`phys = paged_resolve(r)`（INVALID 回退 `r` + 计 fail），`memcpy(staging + r*row_size, K_layer + phys*row_size, row_size)`，K/V 各一；`changed` 计 `phys!=r`。
3. **`get_k/get_v` 接入**（[:2335](../src/llama-kv-cache.cpp#L2335)/[:2367](../src/llama-kv-cache.cpp#L2367)）：`if (kv_paged_enabled && !v_trans && n_stream==1)` → 先 `paged_gather_read(...)`，再在 **staging buffer** 上开**与原 view 完全相同形状**的 `ggml_view_4d`（base 换成 staging，偏移归零，`nb` 不变）；否则原样返回。
4. **gather 时机**：必须在 graph compute 前、staging 内容就绪后建 view。首版可在 `get_k/get_v` 被调用时同步 gather（构图即填），保证 view 指向的 staging 已是最新；需确认 `get_k/get_v` 的调用时序在 compute 之前（build_attn 构图阶段）。
5. **`paged_log_stats` 追加** `read_gather_calls/changed/fail` 打印（stderr）。
6. **关闭/不支持布局**：`v_trans` 或 `n_stream>1` 时 paged-read 不启用、告警一次、走原路径（与 Stage 1 支持性检查一致）。

> **时序关键点**：`get_k/get_v` 在构图阶段被调，FA 在 compute 阶段读 staging。若 staging 在构图时 gather、compute 时才被 kernel 读，需保证两阶段之间 staging 不被覆写（单序列、每 step 重新 gather 即可）。这是方案 B 唯一需要在实现期重点验证的时序假设，设计阶段先记录、Stage 3A 编码时以 diff 实测确认。

---

## 9. 验证命令和通过标准

> 均 `-fa on --temp 0 --seed 42`，确定性贪心；本设计阶段不跑，Stage 3A 编码后执行。

### 9.1 默认关闭零回归

```bash
./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on --seed 42 --temp 0 \
    > /tmp/base.txt 2>/tmp/base.log
LLAMA_KV_PAGED=0 ./build/bin/llama-cli ... --temp 0 > /tmp/off.txt 2>/tmp/off.log
diff /tmp/base.txt /tmp/off.txt          # 必须空
```

### 9.2 开启 staging-read 输出一致

```bash
LLAMA_KV_PAGED=1 ./build/bin/llama-cli ... --temp 0 > /tmp/on.txt 2>/tmp/on.log
diff /tmp/base.txt /tmp/on.txt           # identity 下必须空
sha256sum /tmp/base.txt /tmp/off.txt /tmp/on.txt   # 三者一致
grep -E 'paged|identity|resolve|gather' /tmp/on.log
```

### 9.3 与 lazy-tail 正交

```bash
LLAMA_KV_PAGED=1 LLAMA_KV_LAZY_TAIL=1 ./build/bin/llama-cli ... --temp 0 \
    > /tmp/on_lazy.txt; diff /tmp/base.txt /tmp/on_lazy.txt   # 应空
```

### 通过标准（全部满足）

1. ✅ 9.1 diff 空（默认关闭字节级等价）；
2. ✅ 9.2 diff 空、三输出 sha256 一致（staging-read 不改输出）；
3. ✅ `read_gather_calls > 0`（读路径**确实经 staging gather**）；
4. ✅ `read_gather_changed == 0`、`read_gather_fail == 0`（identity 下 `phys==r`）；
5. ✅ `write_resolve_changed == 0`、`identity_fail == 0`（Stage 1/2A 自证仍成立）；
6. ✅ `graphs reused` 与 Stage 2A 持平（62，view 形状/拓扑未变）；
7. ✅ 9.3 与 lazy-tail 同开仍 diff 空。

判定核心：**`read_gather_calls > 0` 且 `changed == 0` 且 diff 空** —— 证明"读端经 block_table gather 但仍恒等"。

---

## 10. 风险表

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| S3-R1 | staging view 形状/`nb` 与原 view 不符致输出错 | 高 | 严格复制原 `ggml_view_4d` 的 ne/nb，仅换 base buffer、偏移归零；9.2 逐 token diff |
| S3-R2 | gather 时序晚于 compute，kernel 读到旧 staging | 高 | 构图阶段同步 gather；单序列每 step 重 gather；Stage 3A 以 diff 实测确认时序假设 |
| S3-R3 | 新增 staging buffer / view 改变图节点序破坏 reuse | 中 | 方案 B 不进图，view 形状不变；以 `graphs reused` 持平守门 |
| S3-R4 | `v_trans` / `n_stream>1` / offload 布局未覆盖 | 中 | 首版仅 `!v_trans && n_stream==1 && CPU/host`，否则告警一次走原路径 |
| S3-R5 | 每 step CPU copy 带来性能下降 | 低 | 首版正确性优先；性能留 Stage 3B（方案 A 进图）；关闭时零开销 |
| S3-R6 | 量化 K 整行字节拷贝边界 | 低 | 按 `ggml_row_size` 整行拷贝，不拆 block；与现有 cpy 语义一致 |
| S3-R7 | 统计日志污染 stdout 影响 diff | 低 | 全走 `LLAMA_LOG_*`（stderr） |

**RSS 立场**：Stage 3A 仍是 identity，物理布局连续、分配不变，**不期待 RSS 下降**；反而因 staging buffer 略增固定内存。收益是"读端具备 block-table 寻址能力"，为 Stage 2B 非连续写的安全接管铺路。

---

## 11. Codex 实现提示词草案（暂不改代码）

> **任务：在 `src/llama-kv-cache.{h,cpp}` 实现 Stage 3A graph-外 CPU staging paged-read。paged 模式下 `get_k/get_v` 改读经 `block_table` gather 的固定 staging buffer，identity 下输出与 baseline 逐 token 一致。**
>
> **硬约束**：① 只改 `llama-kv-cache.h`/`llama-kv-cache.cpp`，禁止改 attention kernel、graph/context、ggml，不新增 ggml op；② 不引入 full PagedAttention / scheduler / release / swap / prefetch；③ 不做非 identity block allocation（`block_table[i]=i`）；④ staging 长度固定 = `n_kv`，view 形状/`nb` 与原 `get_k/get_v` 完全一致，仅换 base buffer；⑤ `LLAMA_KV_PAGED=0` 字节级等价（paged 分支早退）；⑥ 仅 `!v_trans && n_stream==1`（且 host buffer）启用，否则告警一次走原路径。
>
> **实现**：
> 1. `.h` 加 staging buffer 字段（每 layer `n_kv*row_size` 的 K/V staging）与 `mutable` 统计 `paged_read_gather_calls/changed/fail`；声明 `paged_gather_read(int32_t il, uint32_t n_kv, const slot_info & sinfo) const`。
> 2. `.cpp` 实现 `paged_gather_read`：关闭早退；否则 for `r in [0,n_kv)`：`phys=paged_resolve(r)`（INVALID→`r`+fail++），`memcpy(staging+r*row_size, layer_K+phys*row_size, row_size)`，K/V 各一；`phys!=r`→changed++；`calls++`。
> 3. `get_k`（:2335）/`get_v` 非转置（:2367）：paged 启用时先 `paged_gather_read(...)`，再在 staging 上开与原 view 同形状的 `ggml_view_4d`（偏移归零）；否则原样。
> 4. `paged_log_stats`（:1679）追加 `read_gather_calls/changed/fail`。
> 5. 所有日志走 `LLAMA_LOG_*`（stderr）。
>
> **验证**（编码后，见 §9）：9.1/9.2/9.3 diff 全空、三输出 sha256 一致；`read_gather_calls>0`、`read_gather_changed==0`、`read_gather_fail==0`、`write_resolve_changed==0`、`identity_fail==0`；`graphs reused` 持平 62。
>
> **对照（只读，勿照搬）**：`/root/oscomp/llama.cpp-paged-ref` 的 `calculate_global_slot_index`（`block_table[bid]*block_size + offset`）—— 用作 gather 行号公式参考，但不引入其 fused op。
