# KV runtime swap 最小 demo 阶段 D 实现记录

> 阶段：读前同步 swap-in / `ensure_resident` 原型（D1+D2）｜ 分支：`kv-runtime-swap-stage-d`
> 前置：阶段 A+B（提交 `f83c7d4`，开关 + cell metadata）+ 阶段 C（固定窗口同步 swap-out）

## 1. 阶段 D 目标与边界

阶段 C 已把窗口外旧 cell 的 K/V 字节复制到 `kv_swap_storage` 并打 `swapped` 标记，但**从不写回**——因为原字节未被破坏，输出天然与 baseline 一致。阶段 D 闭合 swap-in 通路：

1. 实现 `swap_in_cell(i)`：从 `kv_swap_storage` 按 `swap_offset(i)` 读回 K/V 字节，**原位**恢复到物理 cell `i`；
2. 实现 `ensure_resident(n_kv)`：在 attention 读 K/V 之前，确保 `[0, n_kv)` 内所有 `swapped` cell 已恢复；
3. 增加 swap-in / ensure_resident 统计；
4. 默认不传 `--kv-swap` 时，行为与 baseline 完全一致；
5. 开启 `--kv-swap` 时，输出与 baseline 逐 token 一致。

**明确不做**（留待 D3 / 后续）：

- ❌ 不 poison / 不清零 / 不破坏原 KV buffer（**D3 才做**，用于反证 swap-in 必需性）；
- ❌ 不实现异步 prefetch；
- ❌ 不改 `get_k` / `get_v` 视图语义；
- ❌ 不改 mask、不改 attention kernel、不改 `src/llama-graph.cpp`；
- ❌ 不改 `state_read_data` / `state_write_data` 语义、不调用 `state_read_meta`；
- ❌ 不触发 `find_slot`、不重新分配 cell、不改 `pos` / `seq` / `shift`；
- ❌ 不改 `include/llama.h` 公共 ABI、不改 CMake、不新增源码文件；
- ❌ 不支持多 sequence / SWA / iSWA / K-shift / seq_cp；
- ❌ 不运行完整 baseline 矩阵、不直接追求 RSS 下降。

⚠️ **本阶段输出与 baseline 一致并不能反证 swap-in 的必需性**：因为阶段 C/D 都不破坏原字节，即使关掉 `ensure_resident`，原张量数据仍在原位，结果依旧正确。真正的必需性反证需要 D3 的 poison。本阶段只验证 swap-in **数据通路正确**（写回字节 == 换出字节，round-trip 无损）。

## 2. 修改文件总览

仅 2 个文件，**未新增文件、未改 CMake、未改公共 ABI、未改 `src/llama-context.cpp`**：

| 文件 | 修改内容 |
|------|----------|
| [src/llama-kv-cache.h](../src/llama-kv-cache.h) | 扩展 stage C 注释提及 stage D；新增 6 个 swap-in 统计字段 + `kv_swap_warned_in`；公有区声明 `ensure_resident(uint32_t)`；私有区声明 `swap_in_cell(uint32_t)` |
| [src/llama-kv-cache.cpp](../src/llama-kv-cache.cpp) | 析构函数追加 swap-in 统计打印；实现 `ensure_resident()` 与 `swap_in_cell()`；`llama_kv_cache_context::apply()` 末尾调用 `ensure_resident(n_kv)` |

`git diff --stat`：`src/llama-kv-cache.cpp | 103 +++`，`src/llama-kv-cache.h | 29 +-`，共 +127 / -5。

## 3. swap-in 恢复路径（`swap_in_cell`）

`swap_in_cell(i)` 是 `swap_out_cell(i)` 的逆操作，复用 stage C 的逐 cell / 单 stream / `!v_trans` 行布局假设：

1. 取 `offset = cells.get_swap_offset(i)`（换出时记录的后备存储起始偏移）；
2. 按 `layers` 顺序、每层先 K 后 V（**与 `swap_out_cell` append 的顺序严格一致**，格式是隐式 / positional 的）：
   - K：行宽 `k_size_row = ggml_row_size(k->type, hparams.n_embd_k_gqa(il))`，用 `ggml_backend_tensor_set(k, kv_swap_storage.data() + offset, i * k_size_row, k_size_row)` **原位**写回 cell `i`（其在张量中占 `i * k_size_row` 处一整行）；
   - V（`!v_trans`）：行宽 `v_size_row = ggml_row_size(v->type, hparams.n_embd_v_gqa(il))`，同理 `ggml_backend_tensor_set` 写回 `i * v_size_row` 处；
   - 每层把 `offset` 前移、累加 `cell_bytes`；
3. `cells.set_swapped(i, false)`——**只清换出标记，不动 `pos` / `seq` / `shift`**；
4. 累加 `kv_swap_in_count += 1`、`kv_swap_in_bytes += cell_bytes`，返回 1。

**关键不变量**：
- 写回位置 `i * size_row` 与换出读取位置完全相同 → **原位恢复**，不重新分配 cell、不调用 `find_slot`、不调用 `state_read_meta`；
- 后备存储按换出顺序顺序读出，与 `swap_offset(i)` 起点对齐，无需显式 per-layer 偏移表；
- `ggml_backend_tensor_set` 后端无关，与 `state_read_data` 写回原语同源（但不复用其 scatter / 类型校验逻辑，路径更窄）。

## 4. ensure_resident 插桩位置

**插桩点 = `llama_kv_cache_context::apply()` 末尾**（`src/llama-kv-cache.cpp`，`get_n_kv` 计算之后）：

```cpp
kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);   // stage C swap-out 在此末尾触发
n_kv = kv->get_n_kv(sinfos[i_cur]);                 // 计算本 ubatch 读区间

// stage D: restore any swapped cell in [0, n_kv) before the graph reads K/V.
kv->ensure_resident(n_kv);

return true;
```

**为什么选这里而非 `src/llama-context.cpp` 的 decode 主循环**：

- `apply()` 是 decode 循环里 `mctx->apply()` 的唯一落点，**既跑 swap-out（`apply_ubatch`）又算 `n_kv`**，两者都已在手边，且都在 `process_ubatch` 构图（`get_k` / `get_v`）之前——满足「graph 外、attention 读 K/V 前、低侵入」三要件；
- 在 `context.cpp` 插桩需通过抽象基类 `llama_memory_context_i` 拿到具体 `llama_kv_cache` 与私有 `n_kv`，要加 downcast + 访问器，**严格更多代码**；
- 因此 stage D **完全不改 `src/llama-context.cpp`**。

`ensure_resident(n_kv)` 逻辑：
1. `!kv_swap_enabled` → 直接返回（默认零开销）；
2. `v_trans || n_stream != 1` → 打印一次 warning 后返回（与 swap-out 同一组 demo guard）；
3. 取 `v_cells[0]`，`n = min(n_kv, cells.size())`；
4. 累加 `ensure_resident_calls += 1`、`ensure_resident_cells_checked += n`；
5. 遍历 `[0, n)`，对 `is_swapped(i)` 的 cell 调 `swap_in_cell(i)`，累加 `ensure_resident_cells_restored`；
6. 累加 `swap_in_us`（整段耗时）。

## 5. n_kv 获取方式

**复用既有 `get_n_kv(sinfo)`，不重写**。`apply()` 中已有 `n_kv = kv->get_n_kv(sinfos[i_cur])`，`ensure_resident` 直接接收该值。

`get_n_kv` 公式：`n_kv = min(size, max(256, PAD(used_max_p1, 256)))`——**向上 pad 到 ≥256**。这意味着即便 `--kv-swap-window 8`，position < `pos_max - 7` 的换出 cell 仍落在 `[0, n_kv)` 内，故每步都会被 `ensure_resident` 换回（解释了下文 `swap_in_count` 远大于窗口大小）。

## 6. v_trans / n_stream 处理

`ensure_resident` 与 `swap_in_cell` 沿用 stage C swap-out 的 demo 边界：

- **`v_trans == false`（`-fa on` / flash_attn=true）**：V 按 cell 连续行存储，可整行 `tensor_set` 写回；
- **`v_trans == true`**：`ensure_resident` 打印一次 warning（`kv_swap_warned_in`）并跳过 swap-in，不尝试转置布局的 scatter，不产生错误数据；
- **`n_stream != 1`**：同样 warning 跳过（demo 仅支持单 sequence / unified cache，stream 0）。

实测 CPU-only + `-fa on` 下 `v_trans = 0`、`n_stream = 1`，两个 guard 均通过、走真实 swap-in 路径。

## 7. 统计指标

阶段 D 在阶段 C 的 swap-out 统计行之后，于自定义析构函数追加第二行 swap-in 统计：

| 字段 | 含义 |
|------|------|
| `swap_in_count` | 从后备存储恢复的 cell 总数 |
| `restored_bytes` | 写回 KV 张量的总字节数（应等于换出字节数） |
| `swap_in_us` | swap-in 累计耗时（μs） |
| `ensure_resident_calls` | `ensure_resident()` 被调用次数（kv_swap_enabled 下） |
| `ensure_resident_cells_checked` | 跨所有调用累计扫描的 cell 数 |
| `ensure_resident_cells_restored` | 累计发现并恢复的 swapped cell 数 |

打印经 `LLAMA_LOG_INFO`，默认 verbosity 下需 `-v` 显现。

## 8. 验证结果

| 验证项 | 结果 |
|--------|------|
| `cmake --build build -j` | ✅ `[100%] Built`，无错误 |
| 默认关闭短烟雾（未传 `--kv-swap`，ctx=512, N_PREDICT=16） | ✅ RSS ≈ 8000.9 MB、decode ≈ 15.37 t/s，落在 baseline 包络内 |
| 开启 `--kv-swap --kv-swap-window 8 -fa on -v`（`-n 64`）真实推理 pass | ✅ swap-out：`swap_out_count=62 swapped_cells=1953 swapped_bytes=255983616 (244.12 MiB)`；swap-in：`swap_in_count=1953 restored_bytes=255983616 (244.12 MiB) swap_in_us=54096 (54.10 ms) ensure_resident_calls=65 ensure_resident_cells_checked=16640 ensure_resident_cells_restored=1953` |
| round-trip 无损 | ✅ `restored_bytes (255983616) == swapped_bytes (255983616)`，逐字节回写 |
| 输出一致性（off vs on, `-n 64`, seed 42） | ✅ 生成文本 IDENTICAL（294 bytes 对 294 bytes） |
| v_trans / 单流约束 | ✅ `-fa on` 下 `v_trans=0`、`n_stream=1`，两个 guard 均通过、走真实 swap-in |

注：开关开启时存在两个 kv_cache 实例日志——首个是 context-reserve 预热 pass（无真实 token，全零统计），第二个是真实推理 pass（计数非零）。

**`swap_in_count` 远大于窗口（1953 vs 8）的原因**：`get_n_kv` 向上 pad 到 ≥256，换出的旧 cell 仍落在 `[0, n_kv)` 读区间内，故每个 decode step 都被 `ensure_resident` 重新换回（`ensure_resident_calls=65` 步 × 每步约 30 个 swapped cell ≈ `cells_restored=1953`）。这暴露了**固定小窗口 + 全区间 resident 检查**的换入放大问题——窗口外 cell 换出后立刻又被读区间要求换回，是后续 prefetch / 非连续 paged read 才能根治的结构性问题（超出 demo 边界）。

## 9. 未实现内容（明确边界）

本阶段**严格未实现 / 未触及**：

- ❌ prefetch（无任何异步 / 预测性换入）；
- ❌ poison / 清零 / 破坏原 KV buffer（**D3 才做**）；
- ❌ attention kernel 修改；
- ❌ `src/llama-graph.cpp`、`get_k` / `get_v` 视图、mask、`state_read/write_data` 语义；
- ❌ `include/llama.h` 公共 ABI、CMake、新增源码文件；
- ❌ `src/llama-context.cpp`（插桩落在 `kv-cache.cpp` 的 `apply()`，context.cpp 零改动）；
- ❌ 多 sequence / SWA / iSWA / K-shift / seq_cp；
- ❌ 完整 baseline 矩阵、RSS 下降的直接验证。

## 10. D3 poison 验证计划（下一轮，非本轮）

阶段 D 输出与 baseline 一致，**但因原字节从未被破坏，无法反证 swap-in 是必需的**。D3 的目标正是补上这一反证：

1. **swap-out 后真正破坏原字节**：`swap_out_cell` 写完后对原张量对应 cell 区域填 0 / 填毒值（如 `0xCC`），使「不换回就读到脏数据」成立；
2. **预期**：关掉 `ensure_resident` 时输出**错乱**（反证换入必需）；开启时输出**仍与 baseline 逐 token 一致**（证明换入正确闭环）；
3. **RSS 收益探索（仍为核心未决问题）**：poison 仍不释放预分配 buffer 物理内存。要兑现 RSS 下降，需让原区域可被 madvise(MADV_DONTNEED) / 复用，或让后备存储承担常驻而原区域不回填——这一步是 demo 内存收益能否成立的关键，留作 D3 之后的专项实验。

## 11. 回滚方式

阶段 D 改动均为未提交工作区改动（2 文件）+ 本文档：

```bash
git checkout -- src/llama-kv-cache.cpp src/llama-kv-cache.h   # 撤销阶段 D 代码
rm docs/kv_runtime_swap_stage_d.md                            # 删除本文档
```

回滚后回到阶段 C 状态（swap-out 存在、无 swap-in / ensure_resident）。
