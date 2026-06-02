# KV runtime swap 最小 demo 阶段 A+B 实现记录

> 提交：`f83c7d4` (`f83c7d41ac5034be146b7d953caf97872293c56f`) — "add kv swap flags and cell metadata"
> 分支：`kv-runtime-swap-demo` ｜ 范围：仅阶段 A + 阶段 B ｜ 默认行为：与 baseline 一致

## 1. 阶段目标

阶段 A+B 是 runtime KV swap demo 的**低风险地基**，定位如下：

- 只加入：配置开关、CLI 参数、cell 级 metadata 字段与访问器；
- **默认关闭**：未传 `--kv-swap` 时，代码路径不触达任何 swap 逻辑；
- **不改变 baseline 行为**：新增字段在关闭态下完全惰性，输出 / RSS / 时序与 baseline 一致；
- **不实现真正的 swap-out / swap-in**：本阶段没有任何 KV 字节搬运；
- 作为阶段 C/D 的承载层：先把"开关 + 元数据载体"就位，后续搬运逻辑只需填充，不必再动这些结构定义。

设计依据见 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md)；baseline 对照见 [kv_baseline_experiment.md](kv_baseline_experiment.md) 与 [kv_baseline_results.md](kv_baseline_results.md)。

## 2. 修改文件总览

提交 `f83c7d4` 共改动 4 个文件，+76 行、0 删除：

| 文件 | 修改内容 | 作用 | 影响默认行为？ |
|------|----------|------|----------------|
| [common/common.h](../common/common.h) | `common_params` 新增 `kv_swap`(bool, 默认 false) 与 `kv_swap_window`(int32, 默认 256) | 提供 demo 配置载体 | 否（仅新增默认关闭字段） |
| [common/arg.cpp](../common/arg.cpp) | 注册 `--kv-swap` 与 `--kv-swap-window N` 两个 CLI 选项（仿 `--kv-unified`） | 暴露 CLI / 环境变量开关 | 否（不传则保持默认值） |
| [src/llama-kv-cells.h](../src/llama-kv-cells.h) | `llama_kv_cells` 新增并行数组 `swapped`/`swap_offset`/`last_access`，在 `resize()`/`reset()` 中同步维护，新增 6 个访问器 | cell 级 swap 元数据载体 | 否（关闭态下纯惰性记录） |
| [src/llama-kv-cache.cpp](../src/llama-kv-cache.cpp) | `apply_ubatch` 写入新 token cell 时初始化 metadata（swapped=false / offset=0 / last_access=pos） | 保证新 cell 元数据有确定初值 | 否（仅写入并行数组，不改计算 / 控制流） |

**未触及**：CMake、`include/llama.h`、`tools/`、`scripts/`、attention kernel、`src/llama-graph.cpp`。

## 3. 阶段 A：配置开关

在 [common/common.h](../common/common.h) 的 `common_params`（紧随 `kv_unified` 字段）新增：

```cpp
// runtime KV swap demo (off by default; see docs/kv_runtime_swap_minimal_demo_plan.md)
bool    kv_swap        = false; // enable runtime KV cache swap / offloading demo
int32_t kv_swap_window = 256;   // number of most-recent cells kept resident (fixed-window policy)
```

在 [common/arg.cpp](../common/arg.cpp) 仿 `--kv-unified` 注册模式新增两个选项：

| 选项 | 类型 | 含义 | 环境变量 | 默认值 |
|------|------|------|----------|--------|
| `--kv-swap` | 单 flag（置 true） | 启用 runtime KV swap / offloading demo | `LLAMA_ARG_KV_SWAP` | false |
| `--kv-swap-window N` | int | 固定窗口策略下保持驻留的最近 cell 数 | `LLAMA_ARG_KV_SWAP_WINDOW` | 256 |

两个选项均 `.set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI})`，使其出现在 llama-completion 的 `--help` 中并适配该工具。

**为什么不修改 `include/llama.h` 公共 ABI**：该头是 llama.cpp 对外稳定的 C API。本 demo 是单工具（llama-completion）内部实验，开关只需经 `common_params` 在应用层流转，无需进入 `llama_context_params`。不动公共 ABI 可避免破坏下游兼容、缩小改动面、降低回滚成本；阶段 C 把开关下沉到 `llama_kv_cache` 时，可走 common → context 的内部桥接（或环境变量）而不必扩展公共结构体。

## 4. 阶段 B：cell metadata

在 [src/llama-kv-cells.h](../src/llama-kv-cells.h) 的 `llama_kv_cells` 新增 3 个并行数组（紧随 `seq` 声明）：

```cpp
std::vector<uint8_t>  swapped;     // 该 cell 的 K/V 字节是否已在后备存储
std::vector<uint64_t> swap_offset; // 后备存储中的偏移
std::vector<uint64_t> last_access; // 粗粒度 recency 计数（固定窗口淘汰用）
```

**为什么放在 `llama_kv_cells` 而非 `llama_kv_cache`**：这三项是**每个 cell 一份**的逐 cell 状态，与已有的 `pos` / `ext` / `shift` / `seq` 完全同构。`llama_kv_cells` 已经是这些并行数组的统一宿主，且 per-stream 随 cells 一起被 `resize`/`reset`/`cp` 维护，天然保证生命周期与索引对齐。全局句柄（后备存储、统计计数器）才属于 `llama_kv_cache`——那部分留到阶段 C。

**为什么 `swapped` 用 `uint8_t` 而非 `std::vector<bool>`**：`std::vector<bool>` 是位压缩特化，不能取元素地址、不能按字节寻址、迭代器语义特殊。后续阶段 C/D 的搬运逻辑可能需要按字节读写或传指针，`uint8_t` 更安全清晰，且每 cell 仅 1 字节的开销可忽略。

**resize/reset 中如何维护**：
- `resize(n)`：`swapped`/`swap_offset`/`last_access` 与 `pos`/`ext`/`shift`/`seq` 一并 `resize(n)`，长度恒等于 cell 数，随后调用 `reset()`。
- `reset()`：逐 cell 置 `swapped[i]=0`、`swap_offset[i]=0`、`last_access[i]=0`，与现有 `pos[i]=-1` 等清零循环合并，不破坏 `pos/ext/shift/seq/used/seq_pos` 的既有语义。
- `cp(i,n)` / `cp(idxs)`：内部经 `resize()` 构造目标 cells，新数组自动按默认值（全 0 = 未换出）初始化，save/restore 路径无需额外处理。

**新增访问器**（均带 `assert(i < size)` 边界检查）：

| 访问器 | 作用 |
|--------|------|
| `is_swapped(i)` / `set_swapped(i, bool)` | 读写换出标记 |
| `get_swap_offset(i)` / `set_swap_offset(i, offset)` | 读写后备存储偏移 |
| `get_last_access(i)` / `set_last_access(i, value)` | 读写 recency 计数 |

封装为访问器而非暴露裸数组，保持 `llama_kv_cells` 现有的封装风格，便于阶段 C 在单一入口加断言或副作用。

**`apply_ubatch` 中对新写 cell 的初始化**（[src/llama-kv-cache.cpp](../src/llama-kv-cache.cpp)，紧随 `cells.pos_set(idx, ...)`）：

```cpp
cells.set_swapped(idx, false);
cells.set_swap_offset(idx, 0);
cells.set_last_access(idx, (uint64_t) ubatch.pos[i]);
```

确保每个新写入 cell 的 swap 元数据有确定初值（未换出、偏移 0、记录访问位置），不依赖 `resize` 时的默认清零（cell 可能被复用而未经过 resize）。

**为什么 `last_access` 暂用 `ubatch.pos[i]`**：在单 sequence demo 下，token position 本身单调递增，是天然的 recency 代理。直接复用它可避免为此引入一个新的 decode-step 计数器、避免把计数器穿过 `decode → apply_ubatch` 的调用链。多 sequence / prefix sharing 场景下 position 不再单调，届时（超出本 demo 边界）再替换为显式 step 计数器。

## 5. 明确未实现内容

本阶段**严格未实现 / 未触及**以下内容（全部留到阶段 C 及以后）：

- ❌ 没有 swap-out（无任何 KV 字节换出）；
- ❌ 没有 swap-in（无任何 KV 字节换入）；
- ❌ 没有 `ensure_resident`（无读前驻留检查）；
- ❌ 没有后备存储模块（无 host buffer / 偏移分配器）；
- ❌ 没有新增 `src/llama-kv-swap.h` / `src/llama-kv-swap.cpp`；
- ❌ 没有修改 CMake（`src/CMakeLists.txt` 等未动）；
- ❌ 没有修改 `include/llama.h`（公共 ABI 未动）；
- ❌ 没有修改 attention kernel；
- ❌ 没有修改 `src/llama-graph.cpp`；
- ❌ 没有修改 decode 主循环（`src/llama-context.cpp::decode` 未动）；
- ❌ 没有改变 `find_slot` 的 `can_use` 判定、`get_k` / `get_v`、`state_read_data` / `state_write_data` 的任何语义。

唯一进入 `.cpp` 的改动是 `apply_ubatch` 中对新增并行数组的初始化，**不影响计算结果与控制流**。

**统计结构亦未加入**：`kv_swap_stats` 的自然宿主是 `llama_kv_cache`（声明于 `src/llama-kv-cache.h`，不在本批允许文件内），且此刻无 swap 逻辑作为消费者，孤儿结构无意义。按计划留到阶段 C（swap-out 落地、给它消费者与头文件归属时）一并加入。

## 6. 验证结果

| 验证项 | 结果 |
|--------|------|
| `cmake --build build -j` | ✅ `[100%] Built`，无错误、无新增警告 |
| `llama-completion --help` | ✅ 可见 `--kv-swap` 与 `--kv-swap-window N`，默认值显示正确 |
| 短烟雾测试（未传 `--kv-swap`） | ✅ `CTX_LENGTHS=512 REPEATS=1 N_PREDICT=16` 正常完成 |
| `parse_kv_baseline.py` | ✅ 正常解析结果表 |
| RSS / decode 速度 | ✅ ctx=512 实测 RSS ≈ 8001 MB、decode ≈ 15.3 t/s，落在既有 baseline 包络内 |
| 默认关闭行为 | ✅ 与 baseline 一致（未传开关时不触达任何新逻辑） |

注：仅运行短烟雾，未跑完整 baseline 矩阵（依本轮约束）。

## 7. 回滚方式

本阶段全部改动集中在单个提交，回滚直接：

```bash
# 方式一：生成一个反向提交（保留历史，推荐）
git revert f83c7d4

# 方式二：分支回退到 A+B 之前（丢弃该提交，需确认无后续依赖）
git reset --hard e02e2d8   # e02e2d8 = A+B 之前的 "add kv baseline experiment results"
```

提交 `f83c7d4` 仅含本阶段 4 个文件的 +76 行新增、0 删除，无与其它改动交织，revert 无冲突风险。

## 8. 后续阶段 C 的前置提醒

阶段 C 才会真正进入有行为的实现，预期涉及：

- **固定窗口同步 swap-out**：按 `kv_swap_window` 把超窗的旧 cell range 字节搬到后备存储，置 `swapped=true`、记 `swap_offset`；
- **后备存储模块**：新增 host buffer + 偏移分配，预计新增 `src/llama-kv-swap.h` / `.cpp`；
- **swap 统计**：在 `llama_kv_cache` 引入 `kv_swap_stats`（swap_out/in 次数、字节数、耗时）；
- **可能修改 CMake**：一旦新增源文件，需在 `src/CMakeLists.txt` 注册；
- **可能下沉开关**：把 `common_params.kv_swap` 经 context 桥接到 `llama_kv_cache`。

⚠️ **阶段 C 的核心验证风险——RSS 是否真的下降**：llama.cpp 的 KV cache 是在 `llama_kv_cache` 构造期一次性预分配的连续 buffer（`ctxs_bufs`）。**仅把 KV 字节复制到后备存储并打 `swapped` 标记，并不会释放这块预分配 buffer 的物理内存**，RSS 不会自动下降。要兑现内存收益，必须让原 buffer 的对应区域可被实际释放 / 复用（例如不回填原 buffer、或让后备存储承担常驻、原区域可被 madvise/复用），这一点必须在阶段 C/D 用小实验显式验证，是 demo 能否成立的关键。详见 [kv_runtime_swap_minimal_demo_plan.md](kv_runtime_swap_minimal_demo_plan.md) §7。
