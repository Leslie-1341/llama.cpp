# Stage 5B-2：idle request resume / swap-in correctness 计划

> 本文档仅为源码阅读、设计与实现计划，不包含任何源码修改。
> 文件路径：`docs/kv_paged_read_stage5b2_resume_swapin_plan.md`
> 关联文档：`docs/kv_paged_read_stage5b_idle_swap_design.md`（Stage 5B 总设计）

---

## 1. 背景

Stage 5B-1 已完成并提交（`dc4dcef95 feat: add safe-candidate idle kv swap-out probe`）。

它证明了：

- safe-candidate based idle KV block 可以被**真实 swap-out**（写入 backing store，状态 `RESIDENT → SWAPPED`）；
- active request（seq1）输出与 no-swap baseline **逐字节一致**（sha256 相同）；
- swapped idle block 在 active-only 阶段能**持续保持 SWAPPED**（依赖 §5.1 的"SWAPPED 仍继续 remap 到 dummy"不变量）；
- 未触发 swap-in、madvise、release、backend failure；
- misconfig no-op smoke 与 positional-regression smoke 已通过。

**当前限制**：

- 只验证了 swap-out 这半程；
- 尚未验证 idle request resume 后能正确 swap-in；
- 尚未验证 swap-out / swap-in roundtrip 的内容一致性；
- 不做 RSS、MADV_DONTNEED、release、prefetch、server、生产级 PagedAttention。

---

## 2. Stage 5B-2 要证明什么

设计一个最小 correctness demo：一个 idle sequence 被 swap-out 后重新 resume，验证 swap-in 正确。

期望场景：

1. seq0 / request A 先写入 KV；
2. seq0 进入 idle；
3. seq1 / request B active decode；
4. Stage 5B-1 逻辑把 seq0 的 idle-only cold block swap-out；
5. 随后让 seq0 **resume**；
6. resume 时该 block 重新成为 active-visible；
7. non-identity remap 前提（`only_seen_idle_seq && !has_active_seq`）失效，row_idx 指回真实 physical block；
8. read path（或 write path）触发 `SWAPPED → RESIDENT` swap-in；
9. seq0 resume 段输出与 no-swap baseline 逐字节一致；
10. 统计中出现 `swap_out > 0`、`swap_in > 0`、`backend_failures = 0`。

---

## 3. 源码定位

### 3.1 现有 driver 能否表达 "A idle → B active → A resume"

**不能。** `examples/kv-idle-telemetry/idle-telemetry.cpp` 的流程是：

- seq0 prefill（`{ 0 }`，`logits=false`，`idle-telemetry.cpp:106-117`）；
- seq1 prefill（`{ 1 }`，`idle-telemetry.cpp:119-131`）；
- **只对 seq1 decode**（`idle-telemetry.cpp:138-158`，token 始终 `{ 1 }`）；
- seq0 永不 resume。

它正是 Stage 5B-1 的 swap-out-only 拓扑。Stage 5B-2 需要在 seq1 decode 之后再对 **seq0** 续跑若干 decode step，触发 remap 撤销与 swap-in。

**建议新增** `examples/kv-idle-swap-resume/`（不改旧 driver，保留 5A/5B-1 回归基线）。

### 3.2 request / seq_id / ubatch / active_seq 的真实结构

- 每个 token 通过 `common_batch_add(batch, token, pos, { seq_id }, logits)` 绑定到一个或多个 `seq_id`；
- `set_input_paged_row_idx()` 从 `ubatch->seq_id[i][s]` 收集本 ubatch 的 `active_seq` 位图（`src/llama-kv-cache.cpp:3628-3636`）：

```cpp
for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
    for (int32_t s = 0; s < ubatch->n_seq_id[i]; ++s) {
        const llama_seq_id seq_id = ubatch->seq_id[i][s];
        if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) active_seq.set(seq_id);
    }
}
```

- 因此"哪个 seq 是 active"完全由**当前 ubatch 携带的 token 的 seq_id** 决定，没有独立的 request 对象；
- `paged_idle_seq_seen`（`src/llama-kv-cache.h:568`）累积"历史上出现过的 seq"，`paged_idle_seq_last_active_step` 记录每 seq 末次 active step（`src/llama-kv-cache.cpp:3789` 附近）——这两者用于判定 idle-only，不用于 resume 识别。

**resume 的识别完全是隐式的**：只要 seq0 的 token 重新出现在某个 ubatch，该 step 的 `active_seq.test(0)` 即为 true，无需任何显式 resume 标志。

### 3.3 SWAPPED-remap 分支何时撤销

remap 发生在 `set_input_paged_row_idx()`（`src/llama-kv-cache.cpp:3781-3793`）。`nonidentity_cold_blocks[block]` 仅在以下成立时置位（`:3708-3719`）：

```cpp
const bool has_active_seq    = (owner & active_seq).any();
const bool only_seen_idle_seq = ((owner & paged_idle_seq_seen) == owner) && !has_active_seq;
if (only_seen_idle_seq) nonidentity_cold_blocks[block] = 1;
```

resume step 中 seq0 进入 `active_seq` → seq0 拥有的 block `has_active_seq=true` → `only_seen_idle_seq=false` → **该 block 不再被标记为 cold → 不再 remap**。这就是 remap 的自动撤销（无显式代码，由前提失效驱动）。

### 3.4 swap-in 的两条触发路径

| 路径 | 函数 | 调用点 | 触发条件 |
|---|---|---|---|
| **读** | `paged_check_read_resident(phys, active)` | `src/llama-kv-cache.cpp:3807`（`set_input_paged_row_idx` 写出每个 row 后） | post-remap 的 `phys` 落在 SWAPPED block |
| **写** | `paged_ensure_write_resident(phys)` | `set_input_k_idxs:3584`、`set_input_v_idxs:3605`、v_trans 分支 `:3621` | 本 step 写入的 cell 解析到 SWAPPED block |

两者都调用 `paged_swap_in_block()`（`src/llama-kv-cache.cpp:1972`）完成 `SWAPPED → RESIDENT`，且都在 **host 侧 input-set 阶段、graph compute 之前**执行。

对 resume 场景：

- seq0 的**历史 KV**（已 swap-out 的 block）在 read path 被 swap-in——这是主路径，因为 resume 的 decode 需要 attention 读全部历史 KV；
- 若 resume 的新 token 恰好落入某个 SWAPPED block 的空 cell（部分填充 block 续写），write path 的 `paged_ensure_write_resident` 兜底。
- graph 输入顺序为 `k_idxs → v_idxs → paged_row_idx`（见 `src/llama-graph.cpp:461` 一线），因此写路径 ensure 与读路径 check 都先于 compute。

---

## 4. 正确性不变量

| # | 不变量 | 保障 |
|---|---|---|
| 1 | 只有 idle-only cold block 能 swap-out | `only_seen_idle_seq`（§3.3） |
| 2 | mixed block 不能 swap-out | `owner.count()==1`（5B-1 safe-candidate funnel） |
| 3 | active seq block 不能 swap-out | `!has_active_seq` |
| 4 | resume 时 block 不再 remap 到 dummy | resume step `has_active_seq=true` → cold 标记失效（§3.3） |
| 5 | row_idx 重新指回真实 physical block | 不再 remap，则 `paged_resolve(cell)` 返回真实 phys |
| 6 | SWAPPED block 必须在 compute/write 前 swap-in | read/write 两路径在 input-set 阶段触发（§3.4） |
| 7 | swap-in 后 state 恢复 RESIDENT | `paged_swap_in_block` 末尾 `state=RESIDENT`（`src/llama-kv-cache.cpp:2109`） |
| 8 | output 与 no-swap baseline 一致 | sha256 比对 |

---

## 5. 最小实验设计

### 5.1 driver（`examples/kv-idle-swap-resume/`）

在现有 idle-telemetry 流程后追加 seq0 resume 段：

1. seq0 prefill（`{ 0 }`，prompt A）；
2. seq1 prefill（`{ 1 }`，prompt B）；
3. seq1 decode N 步（idle 阶段，触发 seq0 swap-out）；
4. **seq0 resume**：以 seq0 prefill 末位 logits 采样首 token，再 decode M 步（token 始终 `{ 0 }`，pos 接续 seq0）；
5. 分别记录并打印 seq1 段与 **seq0 resume 段**输出。

determinism：greedy、`temp=0`、固定 seed、`kv_unified=true`、`backend_sampling=false`（与现有 driver 一致）。

> 关键：必须单独打印 seq0 resume 段输出（如 `resume_text_begin/end`）。若只比 seq1，则 B active 输出会掩盖 seq0 swap-in 的任何错误——swap-in 缺陷恰恰只反映在 seq0 输出上。

### 5.2 两组对照

| 组 | 环境变量 |
|---|---|
| base | `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 LLAMA_KV_PAGED_IDLE_TRACE=1 LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=0` |
| swap | `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 LLAMA_KV_PAGED_IDLE_TRACE=1 LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=1` |

两组同 seed / prompt / ctx / n。base 组开 backing store 但不启用 idle swap，确保两组除 swap 外完全同构。

---

## 6. 测试命令草案

```bash
MODEL=/path/to/model.gguf
ENV_COMMON="LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
LLAMA_KV_PAGED_IDLE_TRACE=1 LLAMA_KV_PAGED_SWAP=1"

# base：不启用 idle swap
env $ENV_COMMON LLAMA_KV_PAGED_IDLE_SWAP=0 \
  ./build/bin/llama-kv-idle-swap-resume -m "$MODEL" -n 64 > base.txt 2> base.log

# swap：启用 idle swap
env $ENV_COMMON LLAMA_KV_PAGED_IDLE_SWAP=1 \
  ./build/bin/llama-kv-idle-swap-resume -m "$MODEL" -n 64 > swap.txt 2> swap.log

# seq0 resume 段逐字节比对（核心）
diff <(sed -n '/resume_text_begin/,/resume_text_end/p' base.txt) \
     <(sed -n '/resume_text_begin/,/resume_text_end/p' swap.txt)

# 完整 stdout sha256
sha256sum base.txt swap.txt

# 统计提取
grep -oE "paged_idle_swap_out_calls=[0-9]+|paged_swap_out_calls=[0-9]+|\
paged_swap_in_calls=[0-9]+|paged_swap_read_swap_in_calls=[0-9]+|\
paged_swap_bytes_out=[0-9]+|paged_swap_bytes_in=[0-9]+|\
paged_swap_backend_failures=[0-9]+|paged_blocks_swapped_out=[0-9]+|\
paged_blocks_swapped_in=[0-9]+" swap.log
```

> 若 driver 未稳定输出 final stats 行，沿用 5B-1 做法：从 `KV_PAGED_IDLE_TRACE` / `KV_PAGED_TRACE` 的每步 fallback 字段取末值，并显式打印 swap 计数。

---

## 7. 预期通过标准

- `base_vs_swap_equal=0`（resume 段 diff 为空）；
- base.txt 与 swap.txt 完整 sha256 一致；
- `paged_idle_swap_out_calls > 0`；
- `paged_swap_out_calls > 0`；
- `paged_swap_in_calls > 0`（或 `paged_swap_read_swap_in_calls > 0`）；
- `paged_swap_bytes_in == paged_swap_bytes_out`（或至少同一 block 上 swap-in bytes 等于 swap-out bytes）；
- `paged_swap_backend_failures = 0`；
- resume 后 `swapped_blocks` 下降，或目标 block 变回 RESIDENT（`paged_blocks_swapped_in > 0`）；
- 无 NaN、无 crash、无 warning/error。

---

## 8. 风险分析

| 风险 | 症状 | 排查方向 |
|---|---|---|
| resume 时仍被 remap 到 dummy | 无 swap-in，`paged_swap_in_calls=0`，resume 段可能仍"正确"但走的是 dummy KV → 输出可能错 | 确认 resume step `active_seq.test(0)==true`；确认 cold 标记失效（§3.3） |
| row_idx 指回真实 SWAPPED block 但 swap-in 太晚 | active read 到 SWAPPED（未初始化/陈旧）数据 → 输出乱、NaN | 确认 `paged_check_read_resident` 在 compute 前；确认输入顺序 k/v_idxs→row_idx |
| write path 先写 SWAPPED block | 新 token KV 写入未 swap-in 的 block，污染 | 依赖 `paged_ensure_write_resident`（§3.4 已确认 K/V 两路覆盖） |
| block 内 mixed seq 被误 swap | swap 了仍被 active 引用的数据 | 5B-1 funnel 已排除；resume 场景同样依赖 `owner.count()==1` |
| baseline / swap 输出不一致 | sha256 不同 | 按优先级排查：① active_seq 识别；② remap 撤销；③ swap-in 时序；④ bytes roundtrip |

---

## 9. 非目标（Stage 5B-2 仍不做）

- 不做 MADV_DONTNEED；
- 不做 release；
- 不做 RSS benchmark；
- 不做 prefetch；
- 不改 `get_k` / `get_v` / attention / mask shape / graph shape；
- 不支持 server；
- 不做完整 PagedAttention；
- 不支持非 F32 KV；
- 不支持 `v_trans`；
- 不做多 idle seq / 多轮 resume 的压力场景（留待后续 stage）。

---

## 10. 给 Codex 的最小实现任务拆分

> 只新增一个 example driver + 验证脚本，**不改** `src/`，**不 commit**。

1. **新增 `examples/kv-idle-swap-resume/`**：
   - 复制 `examples/kv-idle-telemetry/` 的 CMakeLists + 源文件骨架；
   - 流程：seq0 prefill → seq1 prefill → seq1 decode N 步 → **seq0 resume decode M 步**；
   - seq0 resume：从 seq0 prefill 末位（需 `logits=true`）采样首 token，pos 接续 seq0 长度，token 绑定 `{ 0 }`；
   - stdout 分段打印：`active_text_begin/end`（seq1）与 `resume_text_begin/end`（seq0 resume）；
   - 同时打印 idle/active prompt token 数、decoded 数。
2. **注册到构建**：在 examples 的 CMake 聚合处加入新目录（参照 kv-idle-telemetry 注册方式）。
3. **不改任何 `src/` 文件**：swap-out / swap-in 逻辑已在 5B-1 完成并接线。
4. **跑 §6 两组命令**，核对 §7 全部通过标准。
5. **产出**：把 driver diff、两组 stdout、sha256、swap 计数提取结果交回人工审阅；**不 commit**。

---

## 11. 是否建议先写文档再编码

**是**（本文件即该文档）。Stage 5B-2 不改 `src/`，风险集中在 driver 的 resume 时序与"单独比对 seq0 resume 段"这一点；先把场景、对照组、通过标准、排查优先级固定下来，再让 Codex 写 driver，能避免"只比 seq1 输出而漏掉 swap-in 缺陷"这一最易犯的错误。
