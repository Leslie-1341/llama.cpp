# Stage 5C：safe idle swapped block madvise / release 计划

> 本文档仅为源码阅读、设计与实现计划，不包含任何源码修改。
> 文件路径：`docs/kv_paged_read_stage5c_idle_swap_release_plan.md`
> 关联文档：
> - `docs/kv_paged_read_stage5b_idle_swap_design.md`（Stage 5B 总设计）
> - `docs/kv_paged_read_stage5b1_idle_swap_out_results.md`（5B-1 swap-out 结果）
> - `docs/kv_paged_read_stage5b2_resume_swapin_plan.md`（5B-2 resume 计划）
> - `docs/kv_paged_read_stage5b2_resume_swapin_results.md`（5B-2 resume 结果）

---

## 1. Stage 5C 目标

在 Stage 5B correctness 已闭环（swap-out → 保持 SWAPPED → resume swap-in，逐字节一致）的基础上，验证：

- **safe idle swapped block 是否可以执行 `madvise(MADV_DONTNEED)` 释放物理页**，从而真正降低进程的 **current RSS**；
- 释放后该 block 仍能在 seq resume 时被 swap-in，输出与 no-swap baseline 逐字节一致。

明确的目标边界：

- 目标是降低 **current RSS**（madvise 后立即可观测的常驻内存下降），**不先承诺 peak RSS**；
- 不引入新的 release 语义状态混乱（见 §4）；
- 仍然只验证 correctness + current RSS，不做生产级 PagedAttention。

> 关键约束：本阶段 madvise **不改变 block 的逻辑状态**。block 在 swap-out 后状态已是 `SWAPPED`，madvise 只是丢弃其物理页内容（数据已写入 backing store），状态保持 `SWAPPED`。不进入 `RELEASED`（见 §4.5）。

---

## 2. 为什么现在可以考虑 madvise / release

逐级支撑：

| 前置 | 已证明 | 给 5C 的支撑 |
|---|---|---|
| 5A（non-identity gather remap） | idle cold block 已通过 row_idx remap **退出 active read window** | madvise 的物理页不在任何 active request 的读窗口内 |
| 5B-1（safe-candidate swap-out） | swap-out 不影响 active seq，输出逐字节一致；block 内容已落 backing store | madvise 丢弃的是“已持久化且不再被读”的页 |
| 5B-2（resume swap-in） | resume 后 remap 自动撤销，read/write path 在 compute 前触发 swap-in，输出一致 | madvise 后即使物理页被丢弃，resume 也能从 backing store 重新填回 |

因此“swap-out 成功 → 物理页已可安全丢弃 → resume 时再 swap-in 填回”这条链路的每一环都已被前序 stage 验证，5C 只是在 swap-out 与 resume 之间插入一次 `madvise`。

此外，madvise 的底层能力 `paged_madvise_block()`（`src/llama-kv-cache.cpp:2121`）与对应 telemetry（`paged_swap_madvise_calls / _bytes / _failures`、`paged_swap_rss_before/after/drop`）已经存在，并已在 `paged_swap_out_block(block, true)` 的 `do_madvise` 分支中接线（`src/llama-kv-cache.cpp:1945-1971`）。5C 不新增 madvise 实现，只新增一个 gate 决定 idle 路径是否启用它。

---

## 3. 最小实现路径

### 3.1 现状定位

- idle swap-out 调用点：`src/llama-kv-cache.cpp:3962`，当前硬编码 `paged_swap_out_block(block, false)`；
- `do_madvise=true` 分支：`src/llama-kv-cache.cpp:1945-1971`，已完成 madvise + RSS 采样 + telemetry；
- env 解析与 gate：`src/llama-kv-cache.cpp:410-424`（解析）与 `:3662-3676`（`idle_swap_ready` / warning）。

### 3.2 推荐路径（二选一，倾向方案 A）

**方案 A（首选，改动最小）**：新增 debug-only env `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1`，把 idle 调用点的第二参数从常量 `false` 改为一个布尔 gate：

```cpp
// 伪代码，仅示意
paged_swap_out_block(block, idle_swap_madvise_ready);
```

- `idle_swap_madvise_ready` 仅在以下**全部**成立时为 true：
  - `LLAMA_KV_PAGED_IDLE_SWAP=1`（即 `idle_swap_ready==true`，已含 SWAP/NONIDENTITY/IDLE_TRACE/backing store 全部前置）；
  - `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1`；
- 复用 `do_madvise=true` 分支已有的 madvise + RSS 统计，无需新增统计字段。

**方案 B（备选，解耦更清晰）**：保持调用点 `paged_swap_out_block(block, false)`，在其返回 swap-out 成功后，于 idle 路径**单独**调用一次 `paged_madvise_block(block, ...)`：

- 优点：madvise 与 swap-out 解耦，便于将来对 madvise 单独加保护（如 active bitmap）；
- 缺点：需要在 idle 路径自己累计 `paged_swap_madvise_calls/_bytes/_failures` 与 RSS 采样，重复方案 A 已有逻辑；
- 仅当方案 A 的 RSS 采样位置不合适（例如想在整轮 idle swap 结束后只采一次 RSS）时才选 B。

> 建议落地方案 A：改动最小、复用已验证的 madvise+telemetry 分支，风险最低。

### 3.3 统计字段

复用现有字段（无需新增，方案 A 下自动累计）：

- `paged_swap_madvise_calls`（= 文案中的 `madvise_calls`）；
- `paged_swap_madvise_bytes`（= `madvise_bytes`）；
- `paged_swap_madvise_failures`（= `madvise_failures`）；
- `paged_swap_madvise_skipped` / `paged_swap_madvise_skip_no_full_page` / `paged_swap_madvise_skip_neighbor`（部分页 / 邻居保护跳过）；
- `paged_swap_rss_before_last_kb` / `paged_swap_rss_after_last_kb` / `paged_swap_rss_drop_last_kb` / `paged_swap_rss_drop_max_kb`（= `rss_before` / `rss_after` / RSS 下降）。

若 idle 段需要独立计数（区分 idle-driven madvise 与其它 madvise），可新增一个 `paged_idle_swap_madvise_calls`，但属可选项，非必需。

---

## 4. 正确性风险

| # | 不变量 | 保障 |
|---|---|---|
| 1 | madvise 后 resume 必须触发 swap-in 并恢复输出一致 | block 状态仍是 `SWAPPED`；resume 时 remap 撤销 → row_idx 指回真实 phys → `paged_check_read_resident` / `paged_ensure_write_resident` 触发 `paged_swap_in_block`（5B-2 §3.4 已验证） |
| 2 | 不能对 active-visible block madvise | madvise 只发生在 idle safe-candidate 路径（`only_seen_idle_seq && !has_active_seq`，`src/llama-kv-cache.cpp:3708-3719`），且仅当该 block 已 remap 出读窗口 |
| 3 | 不能对 mixed active block madvise | 5B-1 funnel 已要求 `owner.count()==1`；madvise 走同一 idle 路径，继承该限制 |
| 4 | 不能对未成功写入 backing store 的 block madvise | madvise 在 `paged_swap_out_block` 内部、**仅在 swap-out 成功（状态置 SWAPPED）之后**执行（`src/llama-kv-cache.cpp:1944-1945`）；写盘失败会 `clear_block_entries()` 并提前 return，永不到达 madvise |
| 5 | 不能引入 release state 与 swapped state 混乱 | madvise **不**写 `RELEASED`；`RELEASED` 仅由 `paged_release_blocks()`（`src/llama-kv-cache.cpp:3050-3097`）设置，与本路径无关。madvise 后 block 保持 `SWAPPED`，resume swap-in 走 SWAPPED 分支（`:3807` 读 / `:1797-1808` 写），不走 RELEASED 分支。两套状态机互不交叉 |

补充风险（来自 `paged_madvise_block` 实现）：

- **邻居页保护**：block 边界未对齐页时，`paged_madvise_block` 用 `protected_neighbor()` 跳过会波及相邻 RESIDENT/active block 的部分页（`src/llama-kv-cache.cpp:2151,2180-2188`）。这意味着小 block / 大页时 `advised_bytes` 可能为 0、RSS 不降——**这是安全行为而非 bug**，但实验需用足够大的 ctx/block 让对齐后的整页区间非空，否则验收 §6 的“RSS 下降”可能观测不到。

---

## 5. 实验矩阵

复用 Stage 5B-2 的 `examples/kv-idle-swap-resume/` driver（seq0 prefill → seq1 prefill → seq1 decode N → seq0 resume decode M），不新增 driver。

| 组 | 环境变量（在 5B-2 ENV_COMMON 基础上） | 验证点 |
|---|---|---|
| base | `…IDLE_SWAP=0`（不启用 idle swap，开 backing store） | 输出基线 |
| idle swap no-madvise | `…IDLE_SWAP=1`（不设 MADVISE） | = 5B-2 行为，swap-out + resume swap-in 一致，RSS 不变 |
| idle swap + madvise | `…IDLE_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1` | swap-out + madvise + resume swap-in 一致，**current RSS 下降** |

`ENV_COMMON`（同 5B-2）：

```bash
LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_IDLE_TRACE=1 LLAMA_KV_PAGED_SWAP=1
```

矩阵覆盖：base / idle-swap-no-madvise / idle-swap+madvise / resume correctness（每组都打印 seq0 resume 段）/ current RSS before-after（取 `paged_swap_rss_before/after_last_kb` 或 driver 自采 `/proc/self/status` VmRSS）。

> 三组同 seed / prompt / ctx / n，除目标 env 外完全同构。RSS 对比以 “idle swap + madvise” 组的 `rss_after < rss_before` 且相对 no-madvise 组 RSS 更低为准。

---

## 6. 验收标准

- 三组完整 stdout 中 **seq1 active 段**与 **seq0 resume 段**均与 base 逐字节一致（diff 为空，full stdout sha256 一致）；
- `paged_idle_swap_out_calls > 0`；
- `paged_swap_in_calls > 0`（或 `paged_swap_read_swap_in_calls > 0`）——证明 madvise 后 resume 仍能 swap-in；
- `madvise_calls`（`paged_swap_madvise_calls`）`> 0`（仅 madvise 组）；
- `madvise_failures`（`paged_swap_madvise_failures`）`= 0`；
- **current RSS 下降**：madvise 组 `paged_swap_rss_drop_last_kb > 0`（或 `rss_after_last_kb < rss_before_last_kb`），且低于 no-madvise 组同期 RSS；
- `paged_release_violation = 0`、`paged_active_release_violation = 0`（无 release 状态误用）；
- 无 warning / error / NaN / backend failure（`paged_swap_backend_failures = 0`）。

> 若因 §4 邻居页保护导致 `advised_bytes=0`、RSS 不降，先增大 ctx / block_size 使整页区间非空，再判定，不直接判失败。

---

## 7. 不做事项

- 不做 prefetch；
- 不做 async swap-in；
- 不做 server 集成；
- 不做完整 PagedAttention；
- 不改 attention kernel / `get_k` / `get_v` / mask / graph shape；
- 不追求 peak RSS（只看 current RSS）；
- 不引入 `RELEASED` 状态到 idle swap 路径（保持 `SWAPPED`）；
- 不支持 `v_trans` / 非 F32 KV / 多 stream。

---

## 8. 下一轮 P5C-code 的最小修改建议

**修改文件**（仅 `src/llama-kv-cache.cpp` + `.h`，符合“P5C-code 才动 src”）：

- `src/llama-kv-cache.cpp:410-424`：新增 `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE` 解析，得到 `idle_swap_madvise_env`；
- `src/llama-kv-cache.cpp:3662-3676` 附近：计算 `idle_swap_madvise_ready = idle_swap_ready && idle_swap_madvise_env`；缺前置时 warning 一次（沿用 5B-1 misconfig no-op 风格）；
- `src/llama-kv-cache.cpp:3962`：`paged_swap_out_block(block, false)` → `paged_swap_out_block(block, idle_swap_madvise_ready)`；
- stats 打印行（`:3993` / `:2371` 附近）：补充输出 `madvise_calls / madvise_bytes / madvise_failures / rss_before / rss_after / rss_drop`（多数字段已存在，确认 idle 段也打印即可）。

**新增字段**（可选）：

- 若需区分 idle-driven madvise：`mutable uint64_t paged_idle_swap_madvise_calls = 0;`（`src/llama-kv-cache.h` 邻近 `paged_idle_swap_out_calls`）。否则复用现有 `paged_swap_madvise_*`，**无需新增**。

**新增 env**：

- `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1`（debug-only，默认关闭；关闭时 5C 行为完全等于 5B-2）。

**build / smoke 命令**：

```bash
# build（沿用既有 example 构建）
cmake --build build --target llama-kv-idle-swap-resume -j

MODEL=/path/to/model.gguf
ENV_COMMON="LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_IDLE_TRACE=1 LLAMA_KV_PAGED_SWAP=1"

# base
env $ENV_COMMON LLAMA_KV_PAGED_IDLE_SWAP=0 \
  ./build/bin/llama-kv-idle-swap-resume -m "$MODEL" -n 64 > base.txt 2> base.log

# idle swap, no madvise
env $ENV_COMMON LLAMA_KV_PAGED_IDLE_SWAP=1 \
  ./build/bin/llama-kv-idle-swap-resume -m "$MODEL" -n 64 > nomadv.txt 2> nomadv.log

# idle swap + madvise
env $ENV_COMMON LLAMA_KV_PAGED_IDLE_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
  ./build/bin/llama-kv-idle-swap-resume -m "$MODEL" -n 64 > madv.txt 2> madv.log

# resume 段逐字节比对（核心）
diff <(sed -n '/resume_text_begin/,/resume_text_end/p' base.txt) \
     <(sed -n '/resume_text_begin/,/resume_text_end/p' madv.txt)
sha256sum base.txt nomadv.txt madv.txt

# 统计 + RSS 提取
grep -oE "paged_idle_swap_out_calls=[0-9]+|paged_swap_in_calls=[0-9]+|\
paged_swap_read_swap_in_calls=[0-9]+|paged_swap_madvise_calls=[0-9]+|\
paged_swap_madvise_bytes=[0-9]+|paged_swap_madvise_failures=[0-9]+|\
paged_swap_rss_before_last_kb=[0-9]+|paged_swap_rss_after_last_kb=[0-9]+|\
paged_swap_rss_drop_last_kb=[0-9]+|paged_swap_backend_failures=[0-9]+|\
paged_release_violation=[0-9]+" madv.log
```

**回滚方式**：

- 单 env 回滚：不设 `LLAMA_KV_PAGED_IDLE_SWAP_MADVISE`（或设 `=0`）→ idle 路径 `do_madvise=false`，行为完全退回 Stage 5B-2，无需改码；
- 代码回滚：把 `:3962` 改回常量 `false` 并移除 env 解析 / gate / 新增字段，即恢复 5B-1/5B-2 源码状态（diff 极小，集中在 3 处）。

---

## 9. 是否建议先写文档再编码

**是**（本文件即该文档）。5C 的源码改动极小（3 处、单 env、复用已有 madvise+telemetry），风险全部集中在**正确性不变量**（§4：madvise 不得波及 active/mixed/未持久化 block，且不得污染 SWAPPED/RELEASED 状态机）与**RSS 可观测性**（§4 邻居页保护可能导致 advised_bytes=0）。先把 gate 条件、状态机边界、实验矩阵与“RSS 不降时先放大 ctx 再判定”的判据固定下来，再让 P5C-code 落地，可避免“madvise 误伤 active 页”或“误判 RSS 实验失败”两类最易犯的错误。
