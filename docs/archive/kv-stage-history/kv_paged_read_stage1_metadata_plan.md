# Stage 1 — block-table metadata scaffold 实现计划

前置：[docs/kv_paged_read_stage0_source_read.md](kv_paged_read_stage0_source_read.md)。

Stage 0 已确认参考 `paged_attention` 分支是重型设计（新增 paged path、自定义 `GGML_OP_PAGED_ATTN`、每步重建 graph、多序列 scheduler）。Stage 1 **只提取 block-table / physical block pool / free-list / logical→physical mapping 这四个思想的元数据骨架**，不触碰任何读写或图逻辑。

---

## 1. Stage 1 目标

建立一套**纯元数据（metadata-only）**的 block-table scaffold，作为后续 paged-read 的承载层：

1. 一个可选的 `block_table`（逻辑块 → 物理块号），单序列、单 stream；
2. 一个 physical block pool 的**元数据**表示（free-list + 块状态），**不新增任何 ggml 张量、不动现有 K/V 张量**；
3. **identity mapping**：开启时 `physical_block == logical_block`，使经由 block_table 解析出的 cell index 与现状逐位相等；
4. 在构造 / reset / clear 时正确维护这套元数据；
5. 统计日志，验证元数据随 decode 正确演进；
6. env 开关 `LLAMA_KV_PAGED`，**默认关闭**；
7. 关闭时**字节级等价于当前分支**，开启时（identity 下）**输出与 baseline 逐 token 一致**。

本阶段是"立骨架 + 自证恒等"，**不追求 RSS 下降，不改变任何实际寻址**。

---

## 2. 非目标（硬约束）

| # | 不做 | 理由 |
|---|---|---|
| 1 | 不默认开启 | 沿用 stage 惯例，env 显式开启 |
| 2 | 不新增 ggml op | Stage 0 判定：重、需双实现 |
| 3 | 不改 attention kernel | 复用现有 FA |
| 4 | 不改 `get_k/get_v` | 读路径冻结 |
| 5 | 不改 `cpy_k/cpy_v` 写入语义 | 写路径冻结（Stage 2 才动 idxs 解析） |
| 6 | 不引入 scheduler | 多序列专用 |
| 7 | 不引入 swap / prefetch | Stage 4/5 |
| 8 | 不引入 release / madvise | Stage 4；且不与现有 lazy madvise 抢占 |
| 9 | 不影响 exact swap / lazy-clear / lazy-tail | 三者继续可独立工作 |
| 10 | `LLAMA_KV_PAGED` 未设/=0 时行为与当前分支完全一致 | 零回归底线 |

**关键边界**：Stage 1 的 block_table **不被任何寻址实际消费**。它只被构建、维护、统计、并在内部 assert"identity 下解析结果 == 原 cell index"。真正接管寻址是 Stage 2（写）/ Stage 3（读）。

---

## 3. 数据结构设计

### 3.1 block_size 默认值

`paged_block_size = 16`（与参考分支一致，2 的幂便于 `>>`/`&`）。可经 `LLAMA_KV_PAGED_BLOCK_SIZE` 覆盖，要求是 2 的幂且 ≤ kv_size。

### 3.2 block_table 表示

单序列、单 stream，所以是一维：

```cpp
// 逻辑块号 -> 物理块号。size = ceil(kv_size / paged_block_size)。
// identity scaffold 阶段：block_table[i] == i（或 INVALID 表示尚未分配）。
std::vector<uint32_t> paged_block_table;
static constexpr uint32_t PAGED_BLOCK_INVALID = UINT32_MAX;
```

逻辑映射公式（与参考分支对齐，**本阶段仅用于 assert，不用于真实寻址**）：

```
logical_block = cell_index / paged_block_size
offset        = cell_index % paged_block_size
phys_cell     = paged_block_table[logical_block] * paged_block_size + offset
// identity 下必有 phys_cell == cell_index
```

### 3.3 physical block pool metadata

**只存元数据，不分配张量**。物理块就是现有 K/V 张量里 `[blk*block_size, (blk+1)*block_size)` 这段行的"名义所有权"：

```cpp
uint32_t paged_n_blocks = 0;                 // ceil(kv_size / paged_block_size)
std::vector<uint8_t>  paged_block_used;      // 每个物理块是否已被逻辑块占用（0/1）
std::vector<uint32_t> paged_free_list;       // 空闲物理块号，LIFO（参考分支 free-list 风格）
```

identity scaffold 下 `paged_free_list` 仅用于"按需点亮"统计，不改变 `block_table[i]=i` 的恒等关系。

### 3.4 开关与统计字段

挂在 `llama_kv_cache` 私有段（紧邻现有 `kv_lazy_tail` 等 stage 字段，风格一致）：

```cpp
bool     kv_paged_enabled   = false;  // LLAMA_KV_PAGED=1
bool     kv_paged_warned    = false;  // 不支持布局时只告警一次
uint32_t paged_block_size   = 16;
// + 3.2/3.3 的容器
// 统计
uint64_t paged_alloc_calls    = 0;    // 触发过块分配的次数
uint64_t paged_blocks_in_use  = 0;    // 当前占用块数
uint64_t paged_identity_checks = 0;   // 执行过的 identity assert 次数
uint64_t paged_identity_fail   = 0;   // identity 不成立次数（应恒为 0）
```

---

## 4. 代码改动范围

### 4.1 新增字段
见 §3.4 / §3.2 / §3.3，全部加在 `src/llama-kv-cache.h` 私有段。

### 4.2 新增小函数（均私有，`llama-kv-cache.cpp`）

| 函数 | 签名 | 作用 |
|---|---|---|
| `paged_init` | `void paged_init(uint32_t kv_size)` | 构造期算 `paged_n_blocks`，建 identity `block_table[i]=i`，填 `paged_block_used`/`free_list` |
| `paged_reset` | `void paged_reset()` | clear/reset 时恢复 identity 初值 |
| `paged_note_cells` | `void paged_note_cells(const slot_info & sinfo)` | 每个 ubatch 后被调一次：把本批 cell 所属块标记 used、更新统计（纯记账） |
| `paged_resolve` | `uint32_t paged_resolve(uint32_t cell) const` | §3.2 公式解析；identity 下返回 `cell` |
| `paged_assert_identity` | `void paged_assert_identity(const slot_info & sinfo)` | 对本批每个 cell 校验 `paged_resolve(cell)==cell`，累加 `paged_identity_*` |
| `paged_log_stats` | `void paged_log_stats() const` | 打印统计（构造期一次 + 析构/末次） |

### 4.3 挂载点（只加调用，不改既有逻辑）

- **构造函数** `llama_kv_cache::llama_kv_cache(...)`（`llama-kv-cache.cpp:336`）：读 `LLAMA_KV_PAGED` env（紧跟现有 `LLAMA_KV_SWAP*` 解析块，~357 行附近），支持性检查（要求 `n_stream==1 && !v_trans`，否则告警一次并保持关闭），随后 `paged_init(kv_size)` + 一行 INFO 日志。
- **`clear(bool data)`**（`llama-kv-cache.cpp:758`）：在 `v_cells[s].reset()` 之后调 `paged_reset()`。
- **`find_slot` 返回后 / `apply_ubatch` 处**（写入已确定 cell 的位置）：调 `paged_note_cells(sinfo)` + `paged_assert_identity(sinfo)`。**注意：只在 sinfo 已就绪后记账，绝不改 sinfo 本身或写入行为。**

### 4.4 可以改的文件

- `src/llama-kv-cache.h`（加字段、声明）
- `src/llama-kv-cache.cpp`（加函数、挂调用、env 解析、日志）
- `docs/kv_paged_read_stage1_metadata_plan.md`（本文）
- 可选：`docs/` 下的 Stage 1 结果文档（验证后）

### 4.5 禁止改的文件

- 任何 `ggml/**`（尤其 `ops.cpp` / `ggml.c` / `ggml.h` / `pagedattn.*`）
- `src/llama-graph.*`（不碰 build_attn / reserve / reuse）
- `src/llama-context.*`
- `src/llama-cparams.h`（Stage 1 用 env，不进 cparams）
- `src/llama-kv-cache-iswa.*`、`src/llama-kv-cells.h`（不改 cells 语义）
- 参考分支的任何文件（只读对照）

---

## 5. 默认关闭策略

1. 唯一开关 `LLAMA_KV_PAGED`（未设或 `=0` → `kv_paged_enabled=false`）；
2. 关闭时：`paged_init` 仍可建表但**所有挂载点的调用都包在 `if (kv_paged_enabled)` 内**——更稳妥的做法是关闭时连容器都不分配，挂载点直接早退，确保零额外内存与零 CPU；
3. 所有新增调用对既有读写/图/swap/lazy 路径**无副作用**（纯记账 + assert）；
4. 不进 cparams、不改默认构造参数 → 关闭路径与当前分支**字节级等价**。

---

## 6. correctness 验证命令

> 仅记录命令，Stage 1 编码完成后才执行；本计划阶段不跑。

### 6.1 默认关闭零回归（与当前分支 diff 为空）

```bash
# baseline（master 或本分支未开 paged）
./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on --seed 42 \
    --temp 0 > /tmp/base.txt 2>/tmp/base.log
# Stage 1 代码，开关关闭
LLAMA_KV_PAGED=0 ./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on \
    --seed 42 --temp 0 > /tmp/off.txt 2>/tmp/off.log
diff /tmp/base.txt /tmp/off.txt        # 必须为空
```

### 6.2 开启 metadata-only 输出一致

```bash
LLAMA_KV_PAGED=1 ./build/bin/llama-cli -m $MODEL -p "$PROMPT" -n 64 -fa on \
    --seed 42 --temp 0 > /tmp/on.txt 2>/tmp/on.log
diff /tmp/base.txt /tmp/on.txt          # identity 下必须为空
grep -E 'paged|identity' /tmp/on.log    # 看统计：paged_identity_fail 必须为 0
```

### 6.3 与既有路径正交

```bash
# 同时开 lazy-tail / exact swap，确认互不破坏
LLAMA_KV_PAGED=1 LLAMA_KV_LAZY_TAIL=1 ./build/bin/llama-cli ... --temp 0 \
    > /tmp/on_lazy.txt; diff /tmp/base.txt /tmp/on_lazy.txt   # 应为空
```

判定标准：6.1 / 6.2 / 6.3 的 `diff` 全空，且 `paged_identity_fail == 0`。

---

## 7. 风险点

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| S1-R1 | 误改 sinfo / 写入行为导致 6.2 不一致 | 高 | 挂载点严格只读 sinfo；先跑 6.2 逐 token diff |
| S1-R2 | identity 公式与 find_slot 真实 cell 不符（非连续 slot） | 高 | `paged_assert_identity` 对每批校验，fail 立即可见；非连续场景先告警关闭 |
| S1-R3 | 关闭路径仍分配容器 / 有开销 → 破坏字节级等价 | 中 | 关闭时挂载点早退，不分配容器 |
| S1-R4 | 与 lazy-tail 的 madvise 误交叉 | 中 | Stage 1 不碰 madvise；6.3 验证正交 |
| S1-R5 | `n_stream>1` / `v_trans` 布局下表义不清 | 低 | 支持性检查：仅 `n_stream==1 && !v_trans` 启用，否则告警一次并保持关闭 |
| S1-R6 | 统计日志噪声污染 stdout 影响 diff | 低 | 全部走 `LLAMA_LOG_*`（stderr），不进 stdout |

**RSS 立场**：Stage 1 **不期待也不声称** RSS 变化。这是元数据骨架，物理寻址与分配完全不变。

---

## 8. 给 Codex 的实现提示词草案（暂不改代码）

> **任务：在 `src/llama-kv-cache.{h,cpp}` 实现 Stage 1 block-table metadata scaffold。纯元数据 + identity 自证，默认关闭，开启时输出与 baseline 逐 token 一致。**
>
> **硬约束**：① 只改 `llama-kv-cache.h` / `llama-kv-cache.cpp`，禁止改 `ggml/**`、`llama-graph.*`、`llama-context.*`、`llama-cparams.h`、`llama-kv-cells.h`、`llama-kv-cache-iswa.*`；② 不新增 ggml op、不改 attention kernel、不改 `get_k/get_v`、不改 `cpy_k/cpy_v` 写入语义；③ 不引入 scheduler/swap/prefetch/release/madvise；④ 不影响 exact swap / lazy-clear / lazy-tail；⑤ `LLAMA_KV_PAGED` 未设或 =0 时字节级等价当前分支（挂载点早退、不分配容器）。
>
> **实现**：
> 1. 在 `llama-kv-cache.h` 私有段加字段：`kv_paged_enabled`、`kv_paged_warned`、`paged_block_size(=16)`、`paged_n_blocks`、`paged_block_table`、`paged_block_used`、`paged_free_list`、统计计数器（见计划 §3.4）；声明 `paged_init/paged_reset/paged_note_cells/paged_resolve/paged_assert_identity/paged_log_stats`。
> 2. 构造函数（`llama-kv-cache.cpp:336`，紧跟 `LLAMA_KV_SWAP*` 解析块）读 `LLAMA_KV_PAGED`（及可选 `LLAMA_KV_PAGED_BLOCK_SIZE`）；支持性检查仅 `n_stream==1 && !v_trans` 启用，否则 `LLAMA_LOG_WARN` 一次并保持关闭；启用则 `paged_init(kv_size)` 建 identity 表 + INFO 日志。
> 3. `clear(bool)`（:758，`v_cells[s].reset()` 后）调 `paged_reset()`。
> 4. 在 sinfo 已确定写入位置处调 `paged_note_cells(sinfo)` + `paged_assert_identity(sinfo)`——只读 sinfo，纯记账与 assert，绝不修改写入行为。
> 5. `paged_resolve(cell)` 按 `block_table[cell/block_size]*block_size + cell%block_size` 解析；identity 下返回 `cell`。`paged_assert_identity` 校验 `paged_resolve(cell)==cell`，不等则累加 `paged_identity_fail` 并 `LLAMA_LOG_WARN`。
> 6. 所有日志走 `LLAMA_LOG_*`（stderr），不污染 stdout。
>
> **验证**（编码后执行，见计划 §6）：`diff` baseline vs `LLAMA_KV_PAGED=0` vs `LLAMA_KV_PAGED=1`（`--temp 0 --seed 42`）三者全空，且日志中 `paged_identity_fail==0`；叠加 `LLAMA_KV_LAZY_TAIL=1` 仍 diff 空。
>
> **对照（只读，勿照搬代码）**：`/root/oscomp/llama.cpp-paged-ref` 的 `llama_block_manager` free-list 与 `calculate_global_slot_index` 映射公式。
