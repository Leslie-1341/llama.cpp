# KV runtime swap 最小 demo 阶段 C 实现记录

> 阶段：固定窗口同步 swap-out 原型 ｜ 分支：`kv-runtime-swap-demo`
> 前置：阶段 A+B（提交 `f83c7d4`，开关 + cell metadata）

## 1. 阶段 C 目标与边界

本阶段只实现**固定窗口同步 swap-out 原型**，验证以下数据通路与状态管理：

1. `--kv-swap` 开关能真正进入 KV cache 层；
2. 能根据 `--kv-swap-window` 选择窗口外旧 cell；
3. 能把旧 cell 的 K/V 字节复制到后备存储；
4. 能设置 cell metadata：`swapped` / `swap_offset`；
5. 能打印最小 swap-out 统计；
6. 默认不开启 `--kv-swap` 时，行为与 baseline 完全一致。

**明确不做**（留待 D/E/F）：

- ❌ 没有 swap-in；
- ❌ 没有 `ensure_resident`；
- ❌ 没有读前 resident 检查；
- ❌ **不清空 / 不破坏 / 不释放原 KV buffer**（只复制字节）；
- ❌ 不改 `get_k` / `get_v`、attention kernel、`src/llama-graph.cpp`；
- ❌ 不改 `state_read_data` / `state_write_data` 语义；
- ❌ 不实现异步 prefetch；
- ❌ 不改 `include/llama.h` 公共 ABI；不新增文件；不改 CMake。

⚠️ **本阶段不追求 RSS 下降**：llama.cpp 的 KV buffer 是构造期一次性预分配的连续 buffer（`ctxs_bufs`）。仅把 K/V 字节复制到后备存储并打标记，**不会**释放原 buffer 物理内存。该风险已在 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md) §7 记录。阶段 C 只验证 swap-out 数据通路与状态管理。

## 2. 修改文件总览

仅 3 个文件，未新增文件、未改 CMake、未改公共 ABI：

| 文件 | 修改内容 |
|------|----------|
| [common/arg.cpp](../common/arg.cpp) | `--kv-swap` / `--kv-swap-window` 的 lambda 内 `setenv` 导出环境变量（桥接到库层） |
| [src/llama-kv-cache.h](../src/llama-kv-cache.h) | `llama_kv_cache` 新增 swap 配置 / 统计 / 后备存储字段 + `swap_out_window()` / `swap_out_cell()` 声明；析构函数由 `=default` 改为自定义 |
| [src/llama-kv-cache.cpp](../src/llama-kv-cache.cpp) | 构造函数读 env 桥接；析构函数打印统计；实现 `swap_out_window()` 与 `swap_out_cell()`；`apply_ubatch` 末尾调用 swap-out |

## 3. 参数桥接：kv_swap 如何从 CLI 到达 KV cache

阶段 A+B 已在 `common_params` 加入 `kv_swap` / `kv_swap_window`，但它们止步于应用层。完整桥接链为：

```
CLI --kv-swap / --kv-swap-window
  → common/arg.cpp lambda: params.kv_swap=true; setenv("LLAMA_ARG_KV_SWAP","1",1)
  → [进程环境变量]
  → llama_kv_cache 构造函数: getenv("LLAMA_ARG_KV_SWAP") → kv_swap_enabled
```

**为什么用 getenv 桥接**：`common_params` → `llama_context_params`（**公共 ABI**，`include/llama.h`）→ 内部 `llama_cparams`（`src/llama-cparams.h`）→ `create_memory()`（`src/llama-model.cpp`）→ `llama_kv_cache` 构造函数。要把开关传进来，"正统"做法需改 `llama-cparams.h` 与 `llama-model.cpp`（超出允许文件范围），或改公共 ABI（明令禁止）。

而 `llama-kv-cache.cpp` 构造函数**已有 getenv 先例**（`LLAMA_KV_CACHE_DEBUG`、`LLAMA_ATTN_ROT_DISABLE`），所以最小侵入方案是：构造函数直接 `getenv`，CLI lambda 用 `setenv` 把标志导出。`.set_env(...)` 本身只做 env→params（读取），不会在传入 CLI 标志时回写环境变量，故需在 lambda 内显式 `setenv` 补上 params→env 方向。此方案只动 `arg.cpp` + `kv-cache.cpp/.h`，零 ABI、零 CMake。

（注：直接 `LLAMA_ARG_KV_SWAP=1 LLAMA_ARG_KV_SWAP_WINDOW=16 llama-completion ...` 也可启用，因为构造函数读的就是这两个环境变量。）

## 4. swap-out 插桩位置

在 `llama_kv_cache::apply_ubatch()` **末尾**（移动 head 之后）调用 `swap_out_window()`：

- 此处新 token cell 的 metadata 已由阶段 B 初始化完毕（`swapped=false` / `last_access=pos`）；
- 不改 `find_slot` 的 `can_use`、不改读路径、不改 mask、不改 graph；
- `swap_out_window()` 首行即 `if (!kv_swap_enabled) return;`，**默认关闭时零开销、零行为变化**。

详见下一节实现说明。

## 5. swap-out 实现

### 5.1 固定窗口策略 `swap_out_window()`

1. `kv_swap_enabled` 为假 → 直接返回（默认路径）；
2. **v_trans fail-fast**：`v_trans==true` → 打印一次 warning 后返回（阶段 C 只支持 V 非转置 / flash_attn=true）；
3. **单流约束**：`n_stream != 1` → 打印一次 warning 后返回（demo 仅支持单 sequence / unified cache，stream 0）；
4. 取 `v_cells[0]`，扫描 `[0, used_max_p1())` 求 `pos_max`（最新位置，单序列下即 recency 锚点，与阶段 B 的 `last_access = token position` 一致）；
5. 窗口边界 `pos_keep_from = pos_max - kv_swap_window + 1`：position ≥ 该值的 cell 保持驻留，更旧的 cell 若 `!is_empty && !is_swapped` 则换出；
6. 累加统计：`swap_out_count`（本次移动 >0 cells 才 +1）、`swapped_cells`、`swap_out_us`。

### 5.2 单 cell 字节搬运 `swap_out_cell(i)`

参考 `state_write_data` 的 K/V 序列化思路，但范围更小（按物理 cell idx、单 stream）：

- 记录当前后备存储尾部为该 cell 的 `swap_offset`；
- 遍历 `layers`，对每层 `k_stream[0]` / `v_stream[0]`：
  - K：行宽 `k_size_row = ggml_row_size(k->type, n_embd_k_gqa(il))`，cell `i` 在 `i * k_size_row` 处连续；
  - V（`!v_trans`）：行宽 `v_size_row = ggml_row_size(v->type, n_embd_v_gqa(il))`，cell `i` 在 `i * v_size_row` 处连续；
  - 用 `ggml_backend_tensor_get(tensor, dst, offset, size)` 读字节、append 到 `kv_swap_storage`；
- 写完置 `set_swap_offset(i, offset)`、`set_swapped(i, true)`，累加 `swap_bytes`。

**关键：只读取（`tensor_get`），从不写回或清零原张量** → 原 KV buffer 完全不变，故开启后输出与 baseline 逐 token 一致。

## 6. 后备存储实现方式

第一版按确认方案放在 `llama_kv_cache` 内，**不新增 `src/llama-kv-swap.h/.cpp`、不改 CMake**：

- `std::vector<uint8_t> kv_swap_storage;`：每次换出把 K/V 字节 append，起始 offset 记入 cell 的 `swap_offset`；
- 本阶段**只复制字节、不复用 offset、不实现 swap-in、不释放原张量**，仅验证数据通路。

## 7. v_trans 处理

- **支持 `v_trans=false`（flash_attn=true / `-fa on`）**：V 按 cell 连续行存储，可整行读取；
- **遇到 `v_trans=true`**：`swap_out_window()` 打印一次 warning 并跳过 swap-out（fail-fast），不尝试 gather 转置布局、不产生错误数据。CPU-only 下用 `-fa on` 即走非转置路径（实测 `v_trans = 0`）。

## 8. 验证结果

| 验证项 | 结果 |
|--------|------|
| `cmake --build build -j` | ✅ `[100%] Built`，无错误 |
| `llama-completion --help` | ✅ 可见 `--kv-swap` 与 `--kv-swap-window N` |
| 默认关闭短烟雾（未传 `--kv-swap`） | ✅ ctx=512：RSS ≈ 8000.9 MB、decode ≈ 15.2 t/s，落在 baseline 包络内 |
| 开启 `--kv-swap`（window 8, `-n 64`） | ✅ 真实推理 pass 统计：`swap_out_count=62 swapped_cells=62 swapped_bytes=8126464 (7.75 MiB) swap_out_us=8347 (8.35 ms)` |
| 输出一致性（off vs on, `-n 48`） | ✅ `diff` 结果 IDENTICAL —— 原 KV buffer 未受扰动 |
| v_trans / 单流约束 | ✅ `-fa on` 下 `v_trans=0`、`n_stream=1`，两个 guard 均通过 |

注：开关开启时存在两个 kv_cache 实例日志——首个是 context-reserve 预热 pass（无真实 token，`swap_out_count=0`），第二个是真实推理 pass（计数非零）。统计经自定义析构函数在 teardown 打印（`LLAMA_LOG_INFO`，默认 verbosity 下需 `-v` 显现）。

## 9. 回滚方式

阶段 C 改动均为未提交工作区改动（3 文件）+ 本文档：

```bash
git checkout -- common/arg.cpp src/llama-kv-cache.cpp src/llama-kv-cache.h   # 撤销阶段 C 代码
rm docs/kv_runtime_swap_stage_c.md                                            # 删除本文档
```

阶段 C 全部增量集中、与阶段 A+B 提交解耦，回滚后回到 A+B 状态（开关 + metadata 存在但无 swap 行为）。

## 10. 后续阶段 D 前置提醒

阶段 D 才进入 swap-in / 读前 resident 检查：

- 实现 `ensure_resident([0, n_kv))`：读路径前把窗口外被换出的 cell 字节写回原张量；
- 插桩在 decode 主循环 `process_ubatch` 之前；
- 为真正验证换入正确性，届时 swap-out 需**真正破坏原字节**（当前阶段 C 不破坏，故输出已一致，无法反证换入必需性）；
- **RSS 收益验证仍是核心未决问题**（见 §1 警告）。
