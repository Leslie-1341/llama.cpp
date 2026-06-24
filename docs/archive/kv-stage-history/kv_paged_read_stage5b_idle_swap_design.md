# Stage 5B：基于 safe-candidate 的 idle block 真实 swap 设计文档

> 本文档仅为源码定位与设计分析，不包含任何源码修改。
> 文件路径：`docs/kv_paged_read_stage5b_idle_swap_design.md`

---

## 1. 背景

Stage 5A 系列已全部完成并提交：

- **Stage 5A-1（identity in-graph gather）** 通过：
  - 新增 `LLAMA_KV_PAGED_INGRAPH`；
  - identity in-graph gather 输出正确；
  - view path 与 gather path 的输出 sha256 一致；
  - `ingraph_gather_layers > 0`；
  - 不做真实 swap / madvise。

- **Stage 5A-2（non-identity row_idx remap）** 通过：
  - 新增 `LLAMA_KV_PAGED_GATHER_NONIDENTITY=1`；
  - 在 `set_input_paged_row_idx()` 中对 idle-only cold cell 做 row_idx remap（指向 resident dummy cell）；
  - 不改变 `n_kv`、mask shape、graph shape；
  - 不改 `get_k/get_v`/attention；
  - 不做真实 swap / release / madvise / prefetch；
  - smoke 证明：`cold_in_read_window 1→0`、`cold_not_in_read_window 0→1`、`safe_swap_candidates 0→1`、输出 sha256 一致、`swapped_blocks=0`。

- **Stage 5A-3（robustness validation）** 通过：
  - 18/18 case（3 prompts × 2 seeds × n=32/64/128）；
  - 所有 case：`cmp=0`、`sha_base==sha_remap`、`remap_rows>0`、`remap_blocks>0`、各 skip 计数为 0、`cold_before=1`、`cold_after=0`、`safe_after=1`、`swapped_blocks=0`、`warnings=0`。

**当前状态**：系统已经能够让 idle-only cold block 通过 row_idx remap 退出 physical read window，并产生 `safe_swap_candidate`（计数），但尚未对这些候选做任何真实的物理 swap。

**Stage 5B 的目标**：对 `safe_swap_candidate` 执行真实的 block-level swap-out（active request 运行时），并在 idle request resume 时正确 swap-in，全过程输出必须与 no-swap baseline 完全一致。

> 说明：`paged_swap_out_window()` 在 `LLAMA_KV_PAGED_SWAP=1` 时其实已经执行了真实 swap-out，因此 Stage 5B 的真正新意不是"第一次做真实 swap"，而是把 swap-out 的**策略**从 positional window 换成 ownership-based safe candidate，并证明 remap + swap 的交互是输出一致的。

---

## 2. 源码定位

所有引用基于当前分支 `kv-paged-stage4c-idle-telemetry`。

### 2.1 函数

| 符号 | 位置 | const | 说明 |
|---|---|---|---|
| `paged_swap_out_block(uint32_t)` | `src/llama-kv-cache.cpp:1843` | **非 const** | 把单个 physical block 的 K/V 写入 backing store，`RESIDENT → SWAPPED` |
| `paged_swap_in_block(uint32_t) const` | `src/llama-kv-cache.cpp:1966` | const | 从 backing store 读回单个 block，`SWAPPED → RESIDENT` |
| `paged_check_read_resident(phys, active) const` | `src/llama-kv-cache.cpp:1805` | const | 读路径：遇 SWAPPED block 触发 swap-in |
| `paged_ensure_write_resident(phys) const` | `src/llama-kv-cache.cpp:1776` | const | 写路径：遇 SWAPPED block 触发 swap-in（也处理 RELEASED→RESIDENT） |
| `paged_swap_out_window(uint32_t n_kv)` | `src/llama-kv-cache.cpp:2657` | 非 const | 旧 positional swap policy（sink + window 启发式） |
| `paged_madvise_block(...) const` | `src/llama-kv-cache.cpp:2110` | const | `MADV_DONTNEED`，Stage 5B-1 禁止 |
| `set_input_paged_row_idx(dst, ubatch) const` | `src/llama-kv-cache.cpp:3608` | **const** | Stage 5A remap + safe-candidate 计算所在处 |

### 2.2 状态与数据成员

| 符号 | 位置 | 说明 |
|---|---|---|
| `kv_swap_store` | `src/llama-kv-cache.h:410` | file backing store（按 cell 粒度 `write_cell/read_cell`，block 由调用方循环组织） |
| `paged_block_states` | `src/llama-kv-cache.h:486` | `mutable std::vector<paged_block_state>`，per-block 状态 |
| `paged_swap_offsets` | `src/llama-kv-cache.h:487` | per-cell backing offset（**当前非 mutable**） |
| `paged_swap_sizes` | `src/llama-kv-cache.h:488` | per-cell backing size（**当前非 mutable**） |
| `paged_swap_enabled` | `src/llama-kv-cache.h:532` | `LLAMA_KV_PAGED_SWAP=1` 时为 true |
| `paged_idle_seq_seen` | `src/llama-kv-cache.h:568` | 已见过的 seq 位图（用于判定 idle-only） |

### 2.3 状态枚举与流转

`paged_block_state`（`src/llama-kv-cache.h:470`）：

```
UNUSED   = 0   // 未分配
RESIDENT = 1   // 在显存/内存中
RELEASED = 2   // 已 madvise release（Stage 4A，Stage 5B 禁用）
SWAPPED  = 3   // 已写入 backing store，内存内容不可信
```

当前流转：

- `UNUSED → RESIDENT`：block 分配/写入。
- `RESIDENT → SWAPPED`：`paged_swap_out_block`（仅当状态为 RESIDENT）。
- `SWAPPED → RESIDENT`：`paged_swap_in_block`（仅当状态为 SWAPPED），由 `paged_check_read_resident` / `paged_ensure_write_resident` 触发。
- `RESIDENT → RELEASED` / `RELEASED → RESIDENT`：Stage 4A 的 madvise 路线，**与 swap 互斥**（见 §3）。

### 2.4 swap-out 现有触发链

`apply()`（`src/llama-kv-cache.cpp:5095`）中：

- 上一步遗留的 `paged_swap_pending` 在 `apply()` 开头触发 `paged_swap_out_window(pending_n_kv)`（`:5105`）；
- 步末重新挂起 `paged_swap_pending = paged_swap_enabled`、`paged_swap_pending_n_kv = n_kv`（`:5137`）。

读路径接线：`set_input_paged_row_idx` 在写出每个 `phys` 后调用 `paged_check_read_resident(phys, active)`（`src/llama-kv-cache.cpp:3770`）。

### 2.5 safe-candidate 计算位置

在 `set_input_paged_row_idx` 的 idle-trace 区块内（`src/llama-kv-cache.cpp:3866-3915`）：

- 逐 block 统计 owner 位图、`has_active_seq`、`mixed`、`only_seen_idle_seq`；
- `cold_not_in_read_window` 且 `paged_block_states[block]==RESIDENT` 时 `safe_swap_candidates += 1`（`:3911-3914`）。

Stage 5A remap 的结果记录在同函数内的局部 `std::set<uint32_t> nonidentity_remapped_blocks`（remap 发生处 `:3763`）。

---

## 3. 现有 swap 机制能否复用

**结论：可复用，需两处小改。**

1. **`paged_swap_out_block` / `paged_swap_in_block` 已经是 block-level swap**：二者按 `begin..end` 遍历 block 内 cell，逐层 stage K/V 行，校验状态转换与 backing size，更新 offsets/sizes 与计数器。Stage 5B 不需要重写它们。

2. **`paged_swap_out_window` 是旧的 positional policy**：它用 `sink_blocks=1 / window_blocks=1` 的位置启发式选择 block。Stage 5B 不应改它，而是用 ownership-based safe candidate policy 作为新的选择策略。

3. **swap-in read/write path 已有基本接线**：读路径 `paged_check_read_resident`、写路径 `paged_ensure_write_resident` 都已能在遇到 SWAPPED block 时触发 swap-in。Stage 5B-2 的 resume **无需新增 swap-in 代码**。

4. **需要的两处小改**（详见 §6、§7）：
   - `paged_swap_out_block` 当前**无条件**调用 `paged_madvise_block`（`:1942-1963`），Stage 5B-1 禁止 madvise，需加 `do_madvise` 开关；
   - `set_input_paged_row_idx` 是 const，而 `paged_swap_out_block` 非 const 且写非 mutable 的 offsets/sizes，需调整 const-ness。

5. **RELEASED 路线与 swap 互斥**：初始化中 `paged_block_release_enabled = release_env && !paged_swap_enabled`（`src/llama-kv-cache.cpp:452`）。Stage 5B 开启 swap 时 release 自动关闭，无需额外处理。

---

## 4. Stage 5B-1 最小设计

- 新增环境变量 `LLAMA_KV_PAGED_IDLE_SWAP=1`，**默认关闭**；
- 关闭时 Stage 5A 行为完全不变（bit-identical）；
- 依赖前置开关：
  - `LLAMA_KV_PAGED_SWAP=1`：提供 `paged_swap_enabled` 与 `kv_swap_store`（backing store）；
  - `LLAMA_KV_PAGED_GATHER_NONIDENTITY=1`：提供 remap 与 `nonidentity_remapped_blocks`；
  - `LLAMA_KV_PAGED_IDLE_TRACE=1`：提供 `paged_idle_seq_seen` 与 safe-candidate 计算；
- 若 `IDLE_SWAP=1` 但前置开关缺失：warn once 并 no-op；
- 只对 **remapped safe candidate** 执行 swap-out；
- 禁止 `MADV_DONTNEED`；
- 禁止 release；
- 禁止 prefetch；
- 不做 RSS benchmark。

**接入点**：safe-candidate 循环内、`safe_swap_candidates += 1` 之处（`src/llama-kv-cache.cpp:3911-3914`）。在该 block 被判定为 safe candidate 后，若 `paged_idle_swap_enabled` 且满足 §5 全部条件，则调用 `paged_swap_out_block(block, /*do_madvise=*/false)`。

复用已存在的计数器（`src/llama-kv-cache.h:533-538`，已在 `paged_log_stats` 中输出）：`paged_swap_out_calls`、`paged_swap_in_calls`、`paged_blocks_swapped_out`、`paged_blocks_swapped_in`、`paged_swap_bytes_out`、`paged_swap_bytes_in`、`paged_swap_backend_failures`。

**修改文件清单**：

- `src/llama-kv-cache.h`：新增 `bool paged_idle_swap_enabled = false;`；把 `paged_swap_offsets` / `paged_swap_sizes` 改为 `mutable`；`paged_swap_out_block` 加 `bool do_madvise = true` 参数并改为 const。
- `src/llama-kv-cache.cpp`：在 `:416` 附近解析 `LLAMA_KV_PAGED_IDLE_SWAP`；`paged_swap_out_block` 中把 madvise 尾段用 `do_madvise` 包裹；在 `:3911` 附近加 swap-out 调用。

---

## 5. swap-out 条件

block 被 swap-out **必须同时满足**：

1. block 是 idle-only cold block（`only_seen_idle_seq == true`，即 `(owner & paged_idle_seq_seen) == owner` 且无 active seq）；
2. block 不是 mixed block（`owner.count() == 1`）；
3. block 不含 active seq（`(owner & active_seq).none()`）；
4. block 不在 remap 后的 physical read window（`!in_read_window`，即不在 `trace_read_blocks` 中）；
5. `block ∈ nonidentity_remapped_blocks`（本 step 其所有引用行已被 remap 到 resident dummy cell）；
6. `paged_block_states[block] == RESIDENT`；
7. backing store 可用（`kv_swap_store != nullptr`，由 `paged_swap_out_block` 内部再次校验）；
8. 当前 step attention 不会读该 block。

条件 5 是核心安全不变量：只有当一个 block 的全部引用行都已被重定向到 resident dummy cell 之后，才允许把它 swap-out，从而保证本 step 的 attention 永远不会读到 SWAPPED block（与条件 4、8 等价/互证）。

### 5.1 SWAPPED idle block 必须继续 remap（跨 step 不变量）

实现中 `set_input_paged_row_idx()` 的 non-identity remap 逻辑在 `idle_swap_ready` 时，**允许已经处于 `SWAPPED` 状态的 idle block 继续被 remap 到 dummy cell**（不只 remap RESIDENT block）。这**不是 bug，而是正确性必要条件**：

- step N：idle cold block 被 remap 后执行 swap-out，状态 `RESIDENT → SWAPPED`；
- step N+1：该 idle block 仍不属于 active seq；
- 若**不**继续 remap，它的 row_idx 会重新指向真实的 SWAPPED physical block；
- 于是 `paged_check_read_resident()` 检测到 SWAPPED 并触发 swap-in——**每一步都 swap-in**，idle swap 失去意义，且引入对刚 swap-out 的 block 的读取；
- 因此 SWAPPED idle block **必须持续 remap 到 resident dummy cell**，直到该 seq resume；
- seq resume 后该 block 重新含 active seq，remap 前提（`only_seen_idle_seq && !has_active_seq`）失效，remap 自动撤销，row_idx 指回真实 physical block，再由 `paged_check_read_resident()` 触发**一次** swap-in（见 §8）。

换言之，「block 是否被 remap 到 dummy」与「block 是否保持 swapped」是同一个开关：只要它仍是 idle-only，就持续 remap、持续 swapped；一旦 resume 变为 active，remap 撤销、swap-in 复活。swap-out 本身（§5 条件 6）只在 RESIDENT 时触发一次，SWAPPED 状态下不会重复 swap-out。

---

## 6. madvise gate 设计

- 现状：`paged_swap_out_block` 在 swap-out 成功后**无条件**调用 `paged_madvise_block(...)`（`src/llama-kv-cache.cpp:1942-1963`），并采样 RSS。
- Stage 5B-1 **禁止 madvise**（不降 RSS、不做 benchmark）。
- 设计：把签名改为 `paged_swap_out_block(uint32_t physical_block, bool do_madvise = true)`：
  - 原有调用路径（`paged_swap_out_window`）保持默认 `do_madvise=true`，行为不变；
  - Stage 5B-1 的新调用传 `do_madvise=false`，跳过整段 madvise + RSS 采样。
- 这样既不影响旧 positional 路线，又保证 5B-1 不触发任何 `MADV_DONTNEED`（验证时 `paged_swap_madvise_calls=0`）。

---

## 7. const-ness 设计

- `set_input_paged_row_idx` 是 **const** 成员函数（`src/llama-kv-cache.cpp:3608`）。
- `paged_swap_out_block` 当前是**非 const**，且写入**非 mutable** 的 `paged_swap_offsets` / `paged_swap_sizes`。
- 因为 swap-out 决策依赖 remap 结果（`nonidentity_remapped_blocks`），而 remap 只存在于 `set_input` 的 const 上下文内，所以 swap-out 必须能在 const 上下文中调用。
- **推荐方案**：
  - 把 `paged_swap_offsets` / `paged_swap_sizes` 改为 `mutable`；
  - 把 `paged_swap_out_block` 改为 **const**。
- **为什么可接受**：
  - `paged_block_states` 已经是 `mutable`，`paged_swap_in_block` 已经是 const——这两个事实说明"swap 相关元数据在逻辑上属于 cache 的可变内部账本，而非对外可观察状态"这一约定已被项目接受；
  - offsets/sizes 是 backing-store 记账数据，与 `paged_block_states` 同类，把它们一并 `mutable` 是一致的、最小的改动；
  - swap-out 不改变任何对外可观察的语义（KV 内容、`n_kv`、graph）——它只是把内容在内存与 backing store 之间搬动并记录位置，因此 const 化不破坏 const 语义契约。
- 备选方案（不推荐）：在 const 上下文中只把候选 block-id 记入某个 mutable 成员，留到下一次非 const 的 `apply()` 经由 `paged_swap_pending` 执行 swap-out。缺点是把决策与执行拆到两个时刻、引入额外状态、与现有 `paged_swap_in_block` 已 const 的设计不一致。

---

## 8. swap-in / resume 设计

- idle 阶段（resume 之前）：只要 seq0 仍是 idle-only，它的 block 即便已是 SWAPPED 也会**持续被 remap 到 resident dummy cell**（见 §5.1），从而保持 swapped、不触发逐步 swap-in。
- seq0 resume 后成为 **active**（出现在 `ubatch` 中，`active_seq.test(0)==true`）。
- Stage 5A remap 对含 active seq 的 block 不做 remap（remap 的前提是 `only_seen_idle_seq && !has_active_seq`），因此 **seq0 的 block 不再被 remap**——这是 §5.1 跨 step remap 不变量的自然终止：resume 这一步 remap 前提失效，remap 被自动撤销。
- row_idx 因此解析回真实 physical block（而非 dummy cell）。
- 读路径：`paged_check_read_resident(phys, active)`（`src/llama-kv-cache.cpp:3770`）检测到该 block 为 SWAPPED，调用 `paged_swap_in_block` 完成 `SWAPPED → RESIDENT`。
- 写路径：seq0 resume 后向曾被 swap-out 的 block 追加新 token 时，写解析路径上的 `paged_ensure_write_resident`（`src/llama-kv-cache.cpp:1776`）同样会在 SWAPPED 时触发 swap-in。
- swap-in 发生在 host 侧的 input-set 阶段，**在 graph compute 之前**，因此 attention 读到的一定是已 resident 的数据。
- **不需要额外的 pre-resume ensure**：读、写两条路径都能自愈。

> 待实现时需确认：写解析路径（`cpy_k` / `cpy_v` 经过的 resolve）确实在每个写入点调用了 `paged_ensure_write_resident`，而不仅仅是读路径。

---

## 9. Stage 5B-1 smoke 设计（swap-out only）

- 拓扑：A-idle（seq0）/ B-active（seq1）；
- seq0 idle 且**不 resume**；
- seq1 active decode；
- Stage 5A remap 打开，safe candidate 被真实 swap-out；
- 比较 **seq1** 输出；
- 可直接复用现有 `examples/kv-idle-telemetry/idle-telemetry.cpp`（它本就 seq0 prefill 后不 resume、只 decode seq1）。

**验证命令**：

```bash
# baseline（IDLE_SWAP 关闭）
LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_TRACE=1 \
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 LLAMA_KV_PAGED_IDLE_SWAP=0 \
  ./build/bin/llama-kv-idle-telemetry -m "$MODEL" -n 64 > base.txt 2> base.log

# swap（IDLE_SWAP 打开）
LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_TRACE=1 \
LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 LLAMA_KV_PAGED_IDLE_SWAP=1 \
  ./build/bin/llama-kv-idle-telemetry -m "$MODEL" -n 64 > swap.txt 2> swap.log

# 输出比对（active_text 段必须完全一致）
diff <(sed -n '/active_text_begin/,/active_text_end/p' base.txt) \
     <(sed -n '/active_text_begin/,/active_text_end/p' swap.txt)
```

**通过标准**：

- `base_vs_swap_equal=0`（diff 为空）；
- 两侧 seq1 输出 sha256 一致；
- `paged_swap_out_calls > 0`；
- `paged_swap_in_calls = 0`；
- `paged_blocks_swapped_out > 0`（swapped_blocks > 0）；
- `paged_swap_backend_failures = 0`；
- `paged_swap_madvise_calls = 0`；
- 无 release、无 madvise。

---

## 10. Stage 5B-2 resume correctness 设计（swap-in）

- **建议新增** `examples/kv-idle-swap-resume/`，不改旧 driver（保留 5A robustness 回归基线）。
- 流程：
  1. seq0 prefill；
  2. seq1 active prefill + decode（此阶段 seq0 的 idle-only block 被 swap-out）；
  3. **seq0 resume**（active decode）；
  4. resume 时触发 swap-in；
  5. 比较 **seq0** 输出。
- determinism：greedy 采样、`temp=0`、固定 seed、`kv_unified=true`（与现有 driver 一致）。
- baseline 组：同 driver、`LLAMA_KV_PAGED_IDLE_SWAP=0`；swap 组：`=1`。

**通过标准**：

- baseline vs swap 的 **seq0** 输出 sha256 一致；
- `paged_swap_out_calls > 0`；
- `paged_swap_in_calls > 0`（或 `paged_swap_read_swap_in_calls > 0`）；
- `bytes_written == bytes_read`（往返 block 的 `paged_swap_bytes_out == paged_swap_bytes_in`）；
- `paged_swap_backend_failures = 0`。

---

## 11. 风险与不变量

| 风险 | 防护 |
|---|---|
| mixed block 被误 swap | 条件 `owner.count()==1` + `!has_active_seq`（§5.2、5.3） |
| active seq 仍引用 SWAPPED block | 只 swap 本 step 已 remap-to-dummy 的 block（§5.5）；active 行永不指向它 |
| remap 前就 swap | 强制 `block ∈ nonidentity_remapped_blocks`（§5.5） |
| resume 时未撤销 remap | seq active 后 remap 前提失效，row_idx 自动指回真实 block（§8） |
| write path 写入 SWAPPED block | `paged_ensure_write_resident` 在写解析时先 swap-in（§8，待确认覆盖全部写入点） |
| resume 时未及时 swap-in | swap-in 在 graph compute 前的 host 侧 input-set 阶段完成（§8） |
| backing store 缺失 / 失效 | `paged_swap_out_block` / `paged_swap_in_block` 内部校验，失败计入 `paged_swap_backend_failures`（断言 =0） |
| bytes_written != bytes_read | `paged_swap_in_block` 逐 cell 校验 `swap_size == total_size`；验证断言 `bytes_out == bytes_in` |
| 同一 block 多次 swap-out/in 状态不一致 | 状态守卫：out 仅 RESIDENT、in 仅 SWAPPED，幂等；offsets 每次 swap-out 重写 |
| dummy remap 与 SWAPPED 状态不一致 | `paged_check_read_resident` 作用于 **post-remap** 的 `phys`（=dummy，resident），不会误触发 swap-in，SWAPPED block 本 step 无 live reader |
| SWAPPED idle block 逐步 swap-in（抖动） | SWAPPED idle block 持续 remap 到 dummy，row_idx 不指向真实 SWAPPED block，避免每步 swap-in（§5.1）；swap-out 仅在 RESIDENT 时发生一次，不重复写盘 |

> 不变量小结：对一个 idle-only block，「remap 到 dummy」与「保持 swapped」同生同灭——idle 期间两者同时成立，resume 时两者同时撤销。真实物理 SWAPPED block 在整个 idle 期间没有任何 live row_idx 指向它。

---

## 11.1 telemetry 解释：safe candidate vs swapped block

Stage 5B-1 末态出现以下组合时，**不是失败**：

```text
safe_swap_candidates=0
paged_nonidentity_safe_candidates_after=0
swapped_blocks=1
paged_idle_swap_candidates=1
paged_idle_swap_out_calls=1
```

原因——两个概念必须区分：

- **safe candidate**：当前仍为 `RESIDENT`、且本 step 满足全部安全条件、**可以被 swap** 的 block。计数 `safe_swap_candidates` / `paged_nonidentity_safe_candidates_after` 只在 block 处于 `RESIDENT` 分支时累加。
- **swapped block**：已被 safe-candidate policy **消费**、写入 backing store、状态变为 `SWAPPED` 的 block。

Stage 5A-2 的 safe candidate 表示"仍 RESIDENT、可 swap"；Stage 5B-1 中该 candidate 已被 swap-out **消费**，状态 `RESIDENT → SWAPPED`，于是它不再是"待 swap 的 safe candidate"，而是"已 swap 的 block"。因此末态 `safe_swap_candidates=0` 与 `swapped_blocks=1` 同时出现，恰是成功签名，而非回归。

**Stage 5B-1 成功指标**应看：

- `paged_idle_swap_candidates > 0`（本 step 识别出 idle swap 候选）；
- `paged_idle_swap_out_calls > 0`（候选确实被成功 swap-out，该计数以 `paged_swap_out_calls` 实际增长为门槛）；
- `swapped_blocks > 0`（`paged_blocks_swapped_out`）；
- `swap_in` / `madvise` / `backend_failures` **不出现**（`paged_swap_in_calls=0`、`paged_swap_madvise_calls=0`、`paged_swap_backend_failures=0`）。

> 注意：由于"消费"语义，`safe_swap_candidates` 在 swap 打开后会被 swap-out 拉低甚至归零，**不能**再用它作为 swap-out 成功指标；应改用上面三个 `paged_idle_swap_*` / `swapped_blocks` 计数。

---

## 12. 禁止内容（Stage 5B 第一版）

- 不做 `MADV_DONTNEED`；
- 不做 release；
- 不做 RSS benchmark；
- 不做 prefetch；
- 不改 `get_k` / `get_v`；
- 不改 attention；
- 不改 mask shape；
- 不改 graph shape；
- 不支持 server；
- 不做完整 PagedAttention；
- 不支持非 F32 KV；
- 不支持 `v_trans`。

---

## 13. 给 Codex 的最小实现任务草案

> 只做 Stage 5B-1，不做 resume driver，不做 RSS，不 commit。

1. `src/llama-kv-cache.h`：
   - 新增 `bool paged_idle_swap_enabled = false;`；
   - 把 `paged_swap_offsets` / `paged_swap_sizes` 改为 `mutable`；
   - `paged_swap_out_block` 签名改为 `void paged_swap_out_block(uint32_t physical_block, bool do_madvise = true) const;`。
2. `src/llama-kv-cache.cpp`：
   - 在 `:416` 附近解析 `LLAMA_KV_PAGED_IDLE_SWAP`，赋值 `paged_idle_swap_enabled = idle_swap_env && paged_swap_enabled`（缺前置开关时 warn once + no-op）；
   - 在 `paged_swap_out_block` 中，把 madvise + RSS 采样尾段（`:1942-1963`）用 `if (do_madvise) { ... }` 包裹；
   - 在 safe-candidate 循环 `:3911-3914` 处，当 `paged_idle_swap_enabled` 且该 block 满足 §5 全部条件（含 `block ∈ nonidentity_remapped_blocks` 且 `RESIDENT`）时，调用 `paged_swap_out_block(block, /*do_madvise=*/false)`。
3. **不**写任何 swap-in 代码（读/写路径已接线）。
4. 编译后跑 §9 smoke，确认通过标准全部满足。
5. **不 commit**，把 diff 与 smoke 结果交回人工审阅。
