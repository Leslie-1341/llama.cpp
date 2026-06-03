# KV runtime swap 阶段 E1-lite：debug madvise 探针实现记录

> 阶段：E1-lite `madvise(MADV_DONTNEED)` 最小探针（debug-only）｜分支：`kv-runtime-swap-stage-d`
> 前置：阶段 C（swap-out）+ D1+D2（swap-in / ensure_resident）+ D3（poison）+ D4（slot 复用，修复无界增长）

## 1. 目标与边界

D4 修复了 `kv_swap_storage` 无界增长后，smaps 采样确认进程存在 GB 级可写匿名 `[anon]` 映射（KV 张量预分配 buffer）。E1-lite 用一个 **debug-only** 探针验证：换出后对原 KV buffer 对应页执行 `madvise(MADV_DONTNEED)`，**RSS 是否可能下降**。

**明确边界**（沿用 demo 约束）：

- ✅ 只加 `LLAMA_KV_SWAP_MADVISE=1` 环境开关，**仅当 `kv_swap_enabled && LLAMA_KV_SWAP_MADVISE=1` 生效**；
- ✅ 只支持 `v_trans=false`（`-fa on`）与 `n_stream=1`；
- ✅ page-aligned 内部区间，首尾不满页跳过，**不误伤相邻 live cell**；
- ❌ 不做 prefetch、不做架构重构、不改 CMake / `include/llama.h` / `common` / `get_k` / `get_v` / attention kernel / graph；
- ❌ 不改 swap-in 逻辑、不改 `pos` / `seq` / `shift`；
- ⚠️ **这是探针，不是最终内存优化方案**：换出页在下一 decode step 被 `ensure_resident` 立刻 fault 回来。

## 2. 修改文件

| 文件 | 修改 |
|------|------|
| [src/llama-kv-cache.h](../src/llama-kv-cache.h) | 新增 `kv_swap_madvise` 开关 + 4 个统计字段；声明 `madvise_swapped_range(lo, hi)` |
| [src/llama-kv-cache.cpp](../src/llama-kv-cache.cpp) | `<sys/mman.h>`/`<unistd.h>` 条件包含；构造函数读 env + 横幅；`swap_out_window` 末尾找连续 swapped run 并调用探针；实现 `madvise_swapped_range`；析构追加 madvise 统计行 |

## 3. madvise 插入位置

插桩点 = `swap_out_window()` 末尾（swap-out 循环之后）。理由：此时本步要换出的 cell 都已 `swapped=true`，可一次性扫描 `[0, n)` 找**极大连续 swapped run**，对整段而非单 cell 做 page-aligned 裁剪——因为单 cell 行（如 K 一层 2KB）远小于一页（4KB），只有按连续 range 聚合才能凑出完整页。

```cpp
// swap_out_window() 末尾
if (kv_swap_madvise && moved_cells > 0) {
    // 扫描 [0,n)，对每段极大连续 swapped run 调 madvise_swapped_range(lo, hi)
}
```

## 4. page-aligned range 计算方式

对每层 K / V 张量，连续 cell 区间 `[lo, hi)` 占绝对字节区间 `[base + lo*row, base + hi*row)`（`row = ggml_row_size(type, n_embd_*_gqa)`，`!v_trans` 下每 cell 一整行连续）：

```cpp
lo_a = (uintptr_t) base + lo*row;
hi_a = (uintptr_t) base + hi*row;
a_start = (lo_a + pg-1) & ~(pg-1);   // 绝对地址向上取整到页
a_end   =  hi_a        & ~(pg-1);    // 绝对地址向下取整到页
if (a_end > a_start) madvise((void*)a_start, a_end-a_start, MADV_DONTNEED);
```

**关键修正**：必须对**绝对地址**对齐，而非相对 `base` 的偏移。首版误对 offset 对齐导致 `base+offset` 落在页中间 → 全部 `EINVAL`（`madvise_failures` == calls，`bytes=0`）。改对绝对地址对齐后 `failures=0`。首尾不满页部分被裁掉，保证只 advise 完全落在 swapped range 内的整页，不触碰与 live cell 共享的页。

## 5. 统计指标

析构追加第五行 swap 统计：

| 字段 | 含义 |
|------|------|
| `madvise_calls` | `madvise()` 调用次数 |
| `madvise_bytes` | 成功 advise 的 page-aligned 总字节 |
| `madvise_failures` | 返回非零的调用数 |
| `madvise_us` | 探针累计耗时（μs） |

`LLAMA_LOG_INFO` 级，需 `-v` 显现。

## 6. 验证结果

| 验证项 | 结果 |
|--------|------|
| `cmake --build build -j` | ✅ Built，无错误 |
| 短测正确性（`madvise+poison`，`-c512 -n64 -w8`） | ✅ 文本连贯且正确（poison 字节经 swap-in 还原）；`madvise_calls=3776 madvise_bytes=225.00 MiB failures=0`；`swap_out=62/244.12 MiB`、`swap_in=1953/244.12 MiB`、`backing_store=7.75 MiB/62 slots` |
| 长测正常结束（`madvise`，`-c2048 -n256 -w8`） | ✅ **exit 0，无 `bad_alloc`**，256 token 生成完整；`madvise_calls=16448 madvise_bytes=4610.00 MiB failures=0 madvise_us=311 ms` |
| 长测 swap 统计 | `swap_out_count=257 swapped_bytes=4690.38 MiB`、`swap_in_count=37504 restored=4688.00 MiB`、`backing_store=34.25 MiB/274 slots` |
| 峰值 RSS（长测 `/usr/bin/time -v`） | **8476352 KB ≈ 8278 MiB** |

### 6.1 RSS 结论：未下降（探针的预期结果）

baseline ctx-2048 峰值 RSS ≈ 8202 MiB（见 [kv_baseline_results.md](kv_baseline_results.md)）。开 madvise 后峰值 RSS **8278 MiB，不降反略升**。原因结构性：

1. **换出即换回**：`get_n_kv` 向上 pad 到 ≥256，换出的旧 cell 下一 step 仍落在 `[0,n_kv)` 读区间 → `ensure_resident` 立刻 `swap_in_cell` 写回原页 → `MADV_DONTNEED` 释放的物理页**马上缺页 fault 回来**。`ensure_resident_cells_restored=37504` 与 `madvise_bytes` 同量级，证实换出/释放/换回抖动。
2. **峰值口径**：`MAX resident set size` 是整个进程生命周期的峰值；即便某瞬间页被丢弃，prompt 预热 + 任一 step 的回填都会把峰值顶到与 baseline 相当。
3. **探针多一次系统调用 + fault 开销**：故 RSS 略高于 baseline、且 `madvise_us`/`swap_in_us` 显著上升（吞吐回退）。

**这正是探针要回答的问题**：在「固定小窗口 + 全区间 resident 检查」的当前结构下，`MADV_DONTNEED` 不能兑现 RSS 下降——换入放大把释放的页立刻拉回。要真正降 RSS，需先消除换入放大（非连续 paged read / 不在 `[0,n_kv)` 全区间强制 resident），这超出 E1-lite 边界。

## 7. 边界与后续

- **debug-only**：默认关闭，仅验证 RSS 可变性，不是最终方案；
- madvise 对**匿名页**有效（KV buffer 是匿名 mmap），探针 `failures=0` 证明调用本身被内核接受、页确被标记可回收，但**换入放大**使净 RSS 不降；
- 后续若要兑现内存收益，方向是消除 `[0,n_kv)` 全区间 resident 约束（block table / 非连续 gather read），或引入真实分层存储让换出页不立即回填——均超出当前 demo / E1-lite 边界。

## 8. 回滚

```bash
git checkout -- src/llama-kv-cache.cpp src/llama-kv-cache.h
rm docs/kv_runtime_swap_stage_e1_lite.md
```
