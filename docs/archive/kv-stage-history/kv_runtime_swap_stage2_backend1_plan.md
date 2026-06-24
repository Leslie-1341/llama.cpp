# KV runtime swap Stage2-backend1：file-backed backing store 设计计划

> 操作系统功能赛技术报告素材 · Stage2-backend1 plan
>
> **定位**：承接 Stage2-state0 / Stage2-backend0，只设计 file-backed backing store 的第一版实现边界。本文只写文档，不改源码，不 build，不跑实验，不进入真实 swap-out / swap-in / ensure_resident / prefetch。
>
> 文档日期：2026/06/09

---

## 1. 阶段目标

Stage2-backend1 的目标不是把 KV cache 接入 swap 热路径，而是把后续 Demo B 所需的 backing store 设计清楚：

- Demo B 目标是 **exact swap + file-backed backing store**。
- exact swap 先保证换出 / 换入后语义不损坏。
- file-backed backing store 先解决旧 heap vector backing store 的结构性问题：换出的 cold KV 字节不能继续常驻在同进程匿名 heap 中。
- backend1-code 只应实现一个可独立 selftest 的 file backend 类，不读写真实 KV tensor，不接入 decode。

本阶段仍不解决 active RSS 的全部问题。只要 attention 仍要求连续 `[0,n_kv)` view，exact 模式在 graph 读取前仍可能需要恢复大量 cell；这是 Demo B 之后再评估的问题，不应在 backend1 中提前混入 PagedAttention 或 approximate 策略。

---

## 2. backing store 选择结论

第一版推荐：

> `tmpfile + pwrite/pread`，append-only，非持久化，不使用 mmap 作为第一版，不使用 memfd / anonymous mmap 作为第一版。

### 2.1 方案对比

| 方案 | 第一版结论 | 主要原因 |
|---|---|---|
| `tmpfile + pwrite/pread` | **推荐** | 显式 I/O，语义清晰；cold KV 字节离开进程匿名 heap；offset/size 可直接对应 `swap_offset/swap_size`；便于解释 RSS 结果。 |
| file-backed mmap | 不作为第一版 | mmap dirty page、fault-in、page cache 与进程 RSS 的关系更难解释；访问时机隐式，容易和“是否真正 offload”混淆。 |
| memfd | 不作为第一版 | 虽然 file-like，但仍更像内存对象；在无 swap 或内存受限环境下，不一定能体现真实 offload，RSS 解释复杂。 |
| anonymous mmap | 不作为第一版 | 仍在进程匿名映射内；没有明确 file-backed offload 语义；依赖 OS swap，嵌入式/边缘环境未必可用。 |

### 2.2 为什么不是 heap vector

旧 swap 原型使用进程内 `std::vector<uint8_t>` 保存换出的 K/V 字节。该方案可以证明数据搬运正确，但不能作为 RSS 优化路径：

- vector 属于同进程匿名 heap，计入 RSS。
- 原 KV tensor 未释放时，vector 是额外副本。
- 即使后续对原 KV 页做 madvise，vector 本身仍会抵消收益。
- 旧原型容易变成“原 KV buffer + backing store 副本”双份常驻。

Stage2-backend1 的第一原则是避免再把 cold KV 复制到同进程 heap 作为主要 backing store。

### 2.3 为什么第一版不用 mmap

file-backed mmap 是后续可选方向，但第一版不应使用它：

- mmap 的 dirty page 是否计入 RSS、何时被回收、何时 fault-in，不如 `pwrite/pread` 显式。
- mmap 读写容易让“backing store 是文件”与“进程映射页仍 resident”混在一起。
- 对比赛 demo 来说，`tmpfile + pwrite/pread` 更容易说明：换出时显式写文件，换入时显式读文件，进程内只保留小 I/O buffer 和 metadata。

### 2.4 为什么第一版不用 memfd / anonymous mmap

memfd 和 anonymous mmap 都更接近内存型 backing：

- memfd 虽是 fd，但常见解释仍是内存文件，RSS/page-cache 口径不直观。
- anonymous mmap 依赖 OS swap 或 `MADV_DONTNEED` 行为，不等价于 file-backed offload。
- 在内存受限且可能无 swap 的环境中，二者不一定能体现“冷 KV 离开匿名 heap”的 demo 目标。

因此 backend1-code 应先选择普通临时文件路径，后续 backend2/backend3 再比较 mmap/memfd 变体。

---

## 3. 接口调整建议

当前 `llama_kv_backing_store_i` 已有空壳接口：

- `write_cell(...)`
- `read_cell(...)`
- `release(...)`
- `reset()`

backend1-code 可以在不接入热路径的前提下调整接口，使 file backend 的错误与统计更清楚。

### 3.1 错误通道

建议增加类似：

```cpp
enum class kv_swap_status {
    ok,
    io_error,
    disabled,
};
```

或等价错误通道。原因：

- `bool` 只能表达成功/失败，不能区分 backend 未启用和真实 I/O 错误。
- Demo B 需要日志和统计明确指出失败原因。
- 后续 exact swap-in 前必须能判断 read 失败是 fatal correctness error。

第一版不需要复杂异常体系，返回 status 即可。

### 3.2 统计字段

实现类内部建议记录：

| 字段 | 含义 |
|---|---|
| `bytes_written` | 累计写入文件的字节数 |
| `bytes_read` | 累计从文件读回的字节数 |
| `write_calls` | `write_cell` 调用次数 |
| `read_calls` | `read_cell` 调用次数 |
| `released_bytes` | 逻辑 release 的累计字节数 |

这些统计不要求进入全局日志，backend1-code 可以只提供 getter 或 selftest 输出。后续 Demo B 再把它们接入 swap 统计。

### 3.3 不需要的接口

第一版不需要：

- `fsync` / `flush`：backend1 是非持久化临时文件，不保证崩溃恢复。
- 显式 `open` / `close`：用 RAII 管理 fd，构造创建，析构关闭。
- slot allocator：第一版 append-only，不复用空间。
- mmap 指针暴露：第一版不用 mmap。
- 多文件管理：单进程单临时文件足够。

### 3.4 建议保留的接口语义

| 接口 | 第一版语义 |
|---|---|
| `write_cell` | 将传入 buffer 显式追加写入文件尾，返回 offset。 |
| `read_cell` | 从 offset 读指定 size 到调用者提供的 buffer。 |
| `release` | 第一版只做记账，不回收文件空间。 |
| `reset` | 清空统计和文件逻辑长度；实现上可 `ftruncate(fd,0)`。 |

---

## 4. slot / offset 策略

第一版采用 append-only。

### 4.1 写入策略

`write_cell` 总是追加写文件尾：

- `offset_out = current_file_len`
- `pwrite(fd, data, size, offset_out)`
- 成功后 `current_file_len += size`
- 成功后调用方后续设置 cell 的 `swap_offset / swap_size`

注意：

- `offset == 0` 是合法文件偏移，不能当 sentinel。
- `swap_size == 0` 才是“没有 backing slot”的 sentinel。
- 空写入不应生成有效 slot。

### 4.2 release 策略

`release(offset, size)` 第一版只记账：

- 不回收文件空间。
- 不维护 freelist。
- 不做 hole punching。
- 不缩短文件，除非 `reset()` 清空整个 backend。

这样做的理由：

- append-only 最容易保证 correctness。
- slot reuse 容易引入 offset 生命周期错误。
- punch-hole 依赖文件系统支持，第一版不应把风险扩散到平台差异。

### 4.3 后续空间回收

以下能力留到 backend2 或更后续：

- slot reuse。
- freelist。
- block-level allocation。
- `fallocate(FALLOC_FL_PUNCH_HOLE)` / sparse file。
- 文件大小高水位压缩。

backend1-code 的目标是先把显式 file-backed I/O 跑通，而不是优化文件空间。

---

## 5. 生命周期设计

### 5.1 cell 新写入

现有 Stage2-backend0 语义：

- `pos_set` 将新写入 cell 标记为 `RESIDENT`。
- `pos_set` 调用 `clear_swap_metadata()`。
- 新写入 cell 不应继承旧 `swap_offset/swap_size`。

这点必须保持。被新 token 复用的 cell 如果曾经有 backing slot，必须先在覆盖安全点 release，然后清 metadata，再进入 `RESIDENT`。

### 5.2 swap-out 后

后续 exact swap-out 的预期顺序：

1. 从原 KV tensor 读出该 cell/block 的 K/V 字节。
2. 调用 `write_cell`。
3. `write_cell` 成功后设置 `swap_offset / swap_size`。
4. 设置 state 为 `SWAPPED`。
5. 后续阶段再考虑是否对原 KV 页做 madvise 或标记 `RELEASED`。

backend1 不实现上述接入，只定义 backing store 的可用语义。

### 5.3 swap-in 后

后续 exact swap-in 的预期顺序：

1. 根据 `swap_offset / swap_size` 调用 `read_cell`。
2. 将数据写回原 KV buffer。
3. state 设回 `RESIDENT`。
4. 第一版建议保留 `offset/size`，不要立刻 release。

为什么 swap-in 后不立刻 release：

- exact 模式下同一 cell 可能反复换出 / 换入。
- 保留 slot 能简化 debug 和统计。
- append-only 第一版没有空间复用，立刻 release 也不能回收文件空间。

真正 release 应发生在 cell 离开 live history 或被新 token 覆盖时。

### 5.4 release 安全点

release 只能发生在 cell 不再代表 live-history KV 时：

| 安全点 | 说明 |
|---|---|
| `rm` | cell 被明确移除。 |
| `seq_rm` | 最后一个 seq 被移除，cell 变空。 |
| `seq_keep` | cell 不再保留目标 seq，变空。 |
| `pos_set` 覆盖写前 | 新 token 占用旧槽位前，旧 backing slot 必须被逻辑释放。 |

Stage2-backend0 已有 `clear_swap_metadata()`，backend1/exact swap 后续接入时应在清 metadata 之前完成 release 记账。当前 `clear_swap_metadata()` 只清 offset/size，不应偷偷调用 backend。

### 5.5 关键不变量

- `SWAPPED` 不等于 free。
- `find_slot` 不能把 live swapped cell 当空槽。
- `release` 只能发生在 cell 真正离开 live history 或被新 token 覆盖时。
- `swap_size == 0` 表示无 backing slot。
- `offset == 0` 是合法 offset。

特别注意：backend0 中 `cp/set` 会复制 `swap_offset/swap_size`。这对保存 metadata 完整性有帮助，但在多序列 / copy / shared cell 场景可能导致 double-release 风险。因此 backend1 / exact swap 第一版必须明确不支持 multi-seq / `seq_cp`，release 逻辑只在单 seq、单 ownership 边界内启用。

---

## 6. 第一版边界

Stage2-backend1-code 的第一版边界必须保守：

| 项 | 第一版边界 |
|---|---|
| 默认行为 | 默认关闭，不触发任何 backing I/O。 |
| decode 热路径 | 不接入。 |
| swap-out / swap-in | 不实现。 |
| `ensure_resident` | 不实现。 |
| madvise | 不做。 |
| prefetch | 不做。 |
| attention kernel | 不改。 |
| backend 类型 | 只支持 CPU host buffer 语义的 dummy/selftest。 |
| layout | 后续真实 demo 限定 `!v_trans && n_stream==1`。 |
| K-shift | 第一版不支持。 |
| seq_cp / multi-seq | 第一版不支持。 |
| 文件模型 | 单进程临时文件。 |
| 分配策略 | append-only。 |
| 持久化 | 非持久化，无 fsync。 |

这意味着 backend1-code 即使实现 `tmpfile + pwrite/pread`，也只允许用 dummy buffer 做 selftest，不能读取或写回真实 KV tensor。

---

## 7. Stage2-backend1-code Codex 任务草案

下一轮代码任务建议如下；本轮不执行。

> 当前任务：Stage2-backend1-code。只实现 file-backed backing store 类和可选 selftest，不接入真实 KV tensor，不接入 decode 热路径，不实现 swap-out / swap-in / ensure_resident。
>
> 允许修改 `src/llama-kv-cache.h` / `src/llama-kv-cache.cpp`，如需要可新增很小的内部 helper。默认行为必须完全不变。
>
> 实现内容：
>
> - 基于 `llama_kv_backing_store_i` 新增 file-backed 实现类。
> - 使用临时文件和 `pwrite/pread`。
> - append-only，`swap_size == 0` 作为无 slot sentinel。
> - `release()` 第一版只做统计，不回收空间。
> - `reset()` 清空逻辑文件和统计。
> - 记录 `bytes_written / bytes_read / write_calls / read_calls / released_bytes`。
> - 可通过 env 或编译期不可触发路径做 dummy buffer selftest，但默认不运行。
>
> 禁止内容：
>
> - 不创建默认 backend 实例。
> - 不接入真实 KV tensor。
> - 不接入 decode。
> - 不实现 `swap_out_cell` / `swap_in_cell` / `ensure_resident`。
> - 不做 madvise。
> - 不改 attention kernel。

如果需要 selftest，建议只写固定小 buffer：

- 写入 `"hello-kv"`。
- 读取回 dummy buffer。
- 比对字节。
- 输出统计。

该 selftest 不应在默认推理中触发。

---

## 8. 后续路线

| 阶段 | 目标 |
|---|---|
| `backend1-code` | 实现 file backend 类 + env 门控 selftest，不接入真实 KV。 |
| `backend2` | 接入真实 `swap_out` / `swap_in` 数据通道，仍限定保守边界。 |
| exact Demo B | file-backed exact swap，验证输出一致性和 idle/current RSS。 |
| debug correctness | poison 回归护栏，证明 swap-in load-bearing。 |
| window/sink approximate | 明确近似策略，展示 active RSS 下降。 |
| Stage 3 prefetch | 引入 `PREFETCHING/READY`，隐藏 swap-in 延迟。 |
| Stage 4 paged read | block table / physical block pool / PagedAttention-style 非连续读取。 |

推荐顺序仍然是先 backend1-code，再 backend2，再 exact Demo B。不要在 backend1-code 中提前进入 approximate、prefetch 或 paged read。

---

## 9. 结论

1. Stage2-backend1 第一版应选 `tmpfile + pwrite/pread`，append-only，非持久化。
2. 第一版不用 mmap/memfd/anonymous mmap，避免 RSS 与 fault-in 语义解释复杂化。
3. `swap_size == 0` 是无 backing slot sentinel；`offset == 0` 是合法文件偏移。
4. `release()` 第一版只记账，不回收文件空间；slot reuse / punch-hole / freelist 留到 backend2。
5. backend1-code 只能实现 file backend 类和 dummy selftest，不能接入真实 KV 或 decode 热路径。
