# llama.cpp KV cache 机制知识清单

> 操作系统功能赛技术报告素材 · 现状分析篇
>
> **范围说明**:本文档只总结 llama.cpp 现有 KV cache 机制、已有优化方法,以及它与 PagedAttention/vLLM 的差异和缺口,不含优化方案、实现计划或代码。所有结论附源码文件与函数名;未读到源码确证处标注"仍需验证"。源码核对基于当前工作区 CodeGraph 索引(1664 文件、34089 节点、108780 边)。
>
> 核对日期:2026/05/31

---

## 一、总体结论

### 1. llama.cpp KV cache 总体设计

llama.cpp 采用 **"预分配大连续张量 + cell 级元数据管理"** 的设计。每层 K/V 各是一个在构造期一次性预分配的连续 `ggml_tensor`(构造函数见 `src/llama-kv-cache.cpp:80`,`kv_layer` 结构定义于 `src/llama-kv-cache.h:215`)。最小管理单位是 **cell**,约等于 1 个 token 的 KV 槽位;cell 的元数据(pos、seq 归属位图、shift、used、seq_pos)由 `llama_kv_cells`(`src/llama-kv-cells.h`)维护,而不是张量本身。

数据通路上存在一个**关键不对称**:

- **写侧** 经 `cpy_k` / `cpy_v`(`src/llama-kv-cache.cpp:1197` 起)通过 `ggml_set_rows` 按 `slot_info.idxs` **散列写入**——具备 paged-style scatter write 特征;
- **读侧** `get_k` / `get_v`(`src/llama-kv-cache.cpp:1145` 起)返回 `[0, n_kv)` 的**连续 `ggml_view_4d`**,attention 读连续区间,再由 `set_input_kq_mask_impl`(`src/llama-kv-cache.cpp:1434`)在 mask 上屏蔽空洞 cell、非本序列 cell、未来 token、SWA 窗外 token。

### 2. 与 PagedAttention/vLLM 的本质差异

vLLM 的核心是把 KV 切成固定大小 block,用 **logical block → physical block 的 block table** 把逻辑上连续的序列映射到**物理上非连续**的 block 池,PagedAttention kernel 按 block table 做**非连续 gather read**。llama.cpp **没有** block/page 抽象、没有 physical block pool、没有 block table、没有 runtime swap/recompute,attention 也不能直接读非连续 KV block——它靠"连续读 + mask"达到逻辑等效,而非物理分页。

### 3. 定性结论

llama.cpp **并非没有 KV cache 优化**:它已有 cell 级元数据管理、cell 级序列共享、环形复用、SWA 驱逐、K-shift、KV 量化、state 序列化等机制。它**缺少的是 block/page 级的虚拟内存式管理**——即固定块抽象、块表间接寻址、物理块池、按需换入换出。这正是本赛题"虚拟内存 / 按需加载 / 换出 / 预取"可切入的语义缺口。

---

## 二、核心数据结构

| 数据结构 | 所在文件 | 核心字段 | 作用 | 生命周期位置 |
|---|---|---|---|---|
| `llama_kv_cache` | `src/llama-kv-cache.h:20` / 实现 `src/llama-kv-cache.cpp` | `layers`、`v_cells`、`v_heads`、`seq_to_stream`、`v_trans`、`type_k`/`type_v` | KV cache 顶层对象,持有各层张量、cells 元数据、流映射;实现 `llama_memory_i` 接口 | 全程,由 `create_memory` 构造 |
| `kv_layer` | `src/llama-kv-cache.h:215` | `il`(模型层号)、`k`、`v`、`k_stream[]`、`v_stream[]` | 单层 K/V 张量(及多 stream 视图);KV 物理内存载体 | 构造期分配,推理期读写 |
| `llama_kv_cells` | `src/llama-kv-cells.h` | `pos[]`、`seq` 位图、`shift[]`、`used`、`seq_pos`(**现无 resident / swapped / dirty / swap_offset 等换出态字段**) | cell 级元数据;记录每个 token 槽位的位置、序列归属、RoPE 偏移、占用状态 | find_slot / apply_ubatch / mask 全程读写 |
| `slot_info` | `src/llama-kv-cache.h:34` | `s0`、`s1`、`strm[ns]`、`idxs[ns]`、`is_contiguous()`(:77) | 描述"本 ubatch 的每个 token 写到哪些 cell";`idxs` 直接喂 `ggml_set_rows` | 每个 ubatch 由 prepare / find_slot 生成,供 cpy / mask 使用 |
| `v_heads` | `src/llama-kv-cache.cpp`(成员) | 每 stream 一个环形写入头指针 | find_slot 的搜索起点,实现环形复用 | 推理期随写入推进 |
| `seq_to_stream` | `src/llama-kv-cache.cpp`(成员) | seq_id → stream 索引映射 | 决定每个序列用哪个 stream(unified 时 n_stream=1,否则 = n_seq_max) | 构造期确定,seq 操作时引用 |
| `sc_info`(`stream_copy_info`) | `src/llama-kv-cache.h:22` | `ssrc[]`、`sdst[]`(源 / 目标 stream 列表) | 描述 update 阶段需要的跨 stream 整块拷贝 | seq_cp 跨 stream 时生成,update 时执行 |
| `type_k` / `type_v` | 成员,来自 cparams | ggml 类型(F16 默认,可 q8_0 / q4_0) | KV 量化类型;决定每 cell 字节数与 kernel 内 dequant 路径 | 构造期固定,kernel 读写时引用 |

> 补充:`type_k`/`type_v` 的访问器为 `src/llama-kv-cache.h:155-156`;`v_trans` 默认 true、FA 开启时为 false(`src/llama-kv-cache.h:227`)。

---

## 三、KV cache 生命周期

| 阶段 | 关键函数 | 数据结构变化 | 作用 |
|---|---|---|---|
| 1. context 创建与工厂分派 | `llama_model::create_memory` `src/llama-model.cpp:1935`,唯一调用方 `llama_context` 构造 `src/llama-context.cpp:33` | 按 arch 选择 recurrent / hybrid / iswa / 标准 KV;Llama3 无 SWA → `new llama_kv_cache` | 决定用哪种 memory 实现 |
| 2. 构造与物理内存分配 | `llama_kv_cache` 构造 `src/llama-kv-cache.cpp:80` | 每层 `ggml_new_tensor`(K/V),buffer 一次性分配;`n_stream` 由 unified 决定 | **全量预分配** KV 物理内存 |
| 3. ubatch 进入,找 cell | `find_slot` `src/llama-kv-cache.cpp:818`(经 `prepare` / `init_batch` 调用) | 从 `v_heads[stream]` 起搜可用 cell,生成 `slot_info.idxs` | 环形定位写入位置 |
| 4. 写 cell 元数据 | `apply_ubatch` `src/llama-kv-cache.cpp:1017` | 在 `llama_kv_cells` 写入 pos / seq / used,推进 head,清理 SWA 窗外 cell | 提交 cell 占用 |
| 5. 写入当前 token K/V | `cpy_k` / `cpy_v` `src/llama-kv-cache.cpp:1197` 起,内部 `ggml_set_rows` | K/V 张量按 `idxs` 散列写入对应 cell | scatter write |
| 6. 读取历史 K/V | `get_k` / `get_v` `src/llama-kv-cache.cpp:1145` 起 | 返回 `[0, n_kv)` 连续 `ggml_view_4d` | 提供 attention 读视图 |
| 7. mask | `set_input_kq_mask_impl` `src/llama-kv-cache.cpp:1434`,外层 `set_input_kq_mask` :1609 | 在 kq_mask 上对空洞 / 异序列 / 未来 / SWA 窗外 cell 置 -INFINITY | 用 mask 修正连续读的"逻辑非连续" |
| 8. update:shift 与跨 stream 拷贝 | `update` `src/llama-kv-cache.cpp:742`,K-shift `build_rope_shift`(:1721 区域),跨 stream 拷贝据 `sc_info` 用 `ggml_backend_tensor_copy` | 重算 RoPE 偏移、执行跨 stream 整块复制 | 位置编码修正 + 流间数据迁移 |
| 9. 清理与持久化 | `seq_rm`(:343)、`clear`(:330)、`state_write_data` `src/llama-kv-cache.cpp:1969` / `state_read_data`(:2187) | 释放 / 清空 cell;按 cell range 序列化或还原 KV 字节 | 序列清理与 state save / load |

> 说明:`state_write_data` / `state_read_data` 在 `llama_kv_cache` 内分别位于 :1969 / :2187,顶层封装在 `src/llama-context.cpp:2994`。

---

## 四、已有 KV cache 优化机制

### 1. find_slot / ring buffer
- **解决**:在固定大小 KV 区内为新 ubatch 快速定位空闲槽位。
- **源码**:`find_slot` `src/llama-kv-cache.cpp:818`;搜索起点 `v_heads[stream]`。
- **逻辑**:从 head 起环形扫描;SWA 窗外 cell 可被复用;含 head reset 启发式。
- **局限**:仍是连续区间内的槽位管理,不是物理块池;碎片靠环形覆盖而非块重映射缓解。

### 2. 散列写入 cpy_k / cpy_v + ggml_set_rows
- **解决**:把当前 ubatch 的 K/V 写到 `idxs` 指定的离散 cell。
- **源码**:`cpy_k` / `cpy_v` `src/llama-kv-cache.cpp:1197` 起,内部 `ggml_set_rows`。
- **逻辑**:写侧已是 scatter——这是 llama.cpp 最接近 paged 行为的部分。
- **局限**:scatter 仅作用于"写",索引仍指向同一连续张量内的行,不是跨物理块。

### 3. 连续读取 get_k / get_v + mask
- **解决**:为 attention 提供可直接矩阵乘的连续 K/V 视图。
- **源码**:`get_k` / `get_v` `src/llama-kv-cache.cpp:1145`;mask `set_input_kq_mask_impl` `src/llama-kv-cache.cpp:1434`;读区间长度 `get_n_kv` `:1129`;消费处 `build_attn_mha` `src/llama-graph.cpp:1953`(取 K/V 于 `:2240-2241`)。
- **逻辑**:读 `[0, n_kv)` 连续视图(`ggml_view_4d`),用 -INFINITY mask 屏蔽无效 cell,逻辑上等效"只读有效 token"。`n_kv` 由 `used_max_p1()` 向上 pad 到 ≥256(`:1134`)。
- **局限**:**读侧未分页**;n_kv 增大时即便有效 token 稀疏,读取与 mask 计算仍覆盖整个连续区间。这是与 PagedAttention 的核心差距点。
- **对 runtime swap 的约束(本轮补充)**:mask 仅置 -INFINITY **屏蔽 softmax 数值,不避免物理读取**——被 mask 的 cell 物理上仍被内核按连续步长读到。因此 runtime swap 前**必须保证 `[0, n_kv)` 读区间内需要访问的 cell 已 resident**;落在该区间的换出 cell 须先换回。详见 [kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md) 第三节。

### 4. seq_cp / prefix reuse / cell bitset 共享
- **解决**:序列间共享相同前缀的 KV,避免重复计算 / 拷贝。
- **源码**:`seq_cp` `src/llama-kv-cache.cpp:406`。
- **逻辑**:**同 stream** 内通过 cell 的 seq 位图标记多序列归属,实现零拷贝共享;**跨 stream** 需经 `sc_info` 整块拷贝。
- **局限**:共享是"位图打标",非引用计数 + COW;跨 stream 必须整块复制;无块级细粒度分裂。

### 5. context shift / K-shift
- **解决**:cell 位置变化后修正 RoPE 位置编码(非内存优化本身)。
- **源码**:`update` 中的 K-shift,`build_rope_shift`(`src/llama-kv-cache.cpp:1721` 区域),输入 `llm_graph_input_k_shift`。
- **逻辑**:对受影响 cell 重新施加 RoPE 旋转。
- **局限**:解决正确性而非容量;不减少常驻内存。

### 6. SWA / iSWA
- **解决**:滑动窗口注意力下驱逐窗外 KV,降低有效占用。
- **源码**:`llama_kv_cache_iswa`(`src/llama-kv-cache-iswa.h:14`),`kv_base` + `kv_swa` 双实例;窗判定 `is_masked_swa`。
- **逻辑**:窗外 cell 标记可复用,find_slot 可覆盖。
- **局限**:仅适用于 SWA 类架构;**Llama3 标准 attention 不走此路径**(无 SWA 时是普通 `llama_kv_cache`)。

### 7. KV cache quantization(type_k / type_v + kernel 内 dequant)
- **解决**:用 q8_0 / q4_0 等降低 KV 字节占用。
- **源码**:类型来自 cparams(经 `create_memory` 传入构造);dequant 不在 llama-graph 图层显式插算子(`build_attn_mha` `src/llama-graph.cpp:1953` 区域),而在 ggml kernel 内经 type traits 完成。CPU FlashAttention 参考实现 `ggml_compute_forward_flash_attn_ext_f16_one_chunk`(`ggml/src/ggml-cpu/ops.cpp:8214`):K 侧把 Q 转成与 K 匹配的 `vec_dot_type`(`q_to_vec_dot`)在**量化域**做点积(`kq_vec_dot`),**K 不还原为 F32**;V 侧 `to_float` 还原 F32 后 `ggml_vec_mad_f32` 累加(F16 走 `ggml_vec_mad_f16` 快路径)。
- **局限**:量化降容量但引入精度损失(量化 K 配 Hadamard 旋转补偿);**非 CPU 后端(CUDA / Metal / SYCL)的 fattn kernel 内部 dequant 细节仍需验证**。

### 8. state_write_data / state_read_data
- **解决**:KV 状态的序列化与还原(session save / load),并具备按精确物理位置写回的底层能力。
- **源码**:`state_write_data` `src/llama-kv-cache.cpp:1969` / `state_read_data` `:2187`(结束于 `:2353`);入口 `state_write` `:1845` / `state_read` `:1898`;`cell_ranges_t` `src/llama-kv-cache.h:296`;IO 实现 `src/llama-context.cpp:2497-2545`。
- **逻辑**:
  - 写侧按 `cell_ranges_t`(per-stream 的多个连续 cell range)遍历**全部 layer** 写 K / V 字节;K 与 V(非转置)按行连续,V 转置按元素维 `j` × range 双重循环(`:2055-2062`)。
  - 读侧 `state_read_data` **已内置非连续 scatter 路径**:`sinfo.is_contiguous()` 为真走单次 memcpy 快路径,为假则**逐 cell 按 `sinfo.idxs[i]` 写回精确物理槽位**(K `:2241-2247`、V 非转置 `:2284-2290`、V 转置 `:2339-2347`)——即"原位写回"能力已存在。
  - 底层搬运 `write_tensor` / `read_tensor` 经 `ggml_backend_tensor_get` / `ggml_backend_tensor_set`(`src/llama-context.cpp:2506-2535`),**后端无关**(CPU / CUDA / Metal 统一)。
- **限定(为何仍非 runtime swap)**:尽管数据搬运层已具 runtime swap 基础,当前机制仍是**持久化语义**而非热路径 swap,因为 meta 编排层(`state_read_meta` `:2068`)在全量恢复时 `clear(true)`(`:2141`)、单序列恢复走 `find_slot` **重新分配**(`:2113`),且**无 `resident` / `dirty` / `swap_offset` 状态、无内存压力触发、无异步搬运**。可作为未来 runtime swap 的**数据通道**复用,但编排与触发机制需重写。


### 9. backend offload / KQV offload
- **解决**:把 KV 与 KQV 计算放到指定后端(含 CPU 回退)。
- **源码**:`offload` = `cparams.offload_kqv`(`create_memory` 传入);非 offload 时 `build_attn_mha` 内 `ggml_backend_sched_set_tensor_backend(...backend_cpu)` 回退 CPU。
- **逻辑**:控制 KV 张量与 attention 计算的设备归属。
- **局限**:是"设备放置"开关,非按需 swap;粒度是整层 / 整缓存,不是块级迁移。

### 10. defrag 废弃情况
- **现状**:碎片整理路径**已废弃**——`llama_kv_cells::mv()` 在 `src/llama-kv-cells.h` 中**被注释掉**(原 :100-117 区域)。
- **含义**:llama.cpp 当前**不做** cell 物理迁移式碎片整理,依赖环形复用 + mask 容忍空洞。
- **局限**:无主动 compaction,长程多序列场景下连续区间可能稀疏占用。

---

## 五、与 PagedAttention/vLLM 对照表

| PagedAttention 机制 | llama.cpp 是否具备 | 最接近的机制 | 相似点 | 差异 | 当前缺口 |
|---|---|---|---|---|---|
| logical block | 否 | cell(单 token 槽) | 都有逻辑位置抽象 | cell 是 1 token,非固定大小多 token block | 无 block 粒度逻辑单元 |
| physical block | 否 | 连续张量内的行 | 都最终落到物理存储 | 无独立物理块,只有一整块连续张量 | 无 physical block pool |
| block table | 否 | `slot_info.idxs`(仅写侧) | idxs 类似"行映射" | idxs 一次性用于 scatter write,无持久化逻辑→物理映射表供读侧 gather | 无持久 block table |
| on-demand allocation | 部分 | find_slot 环形占用 | 都按需占用槽位 | llama.cpp 在预分配大张量内占用,不新增物理内存 | 无真正按需物理分配 / 增长 |
| non-contiguous KV read | 否 | get_k / get_v 连续 view + mask | 都能"逻辑跳过"无效 token | llama.cpp 物理上连续读再 mask,vLLM 物理非连续 gather | **读侧无分页 gather** |
| block-level sharing | 部分 | seq_cp + cell seq 位图 | 都支持前缀共享 | llama.cpp 是位图打标(同 stream 零拷贝),非块级 refcount | 无块级引用计数 |
| copy-on-write | 否 | 跨 stream 整块拷贝(`sc_info`) | 都处理共享后分叉 | llama.cpp 写时整块复制,非块级 COW | 无块级 COW |
| CPU/GPU block allocator | 否 | backend offload 开关 | 都涉及设备放置 | offload 是整缓存设备归属,无块分配器 | 无双层块分配器 |
| swap / recompute | 否 | state_write / read(离线) | 都能序列化 KV | state 是离线 save / load,非运行时 swap / 重算 | **无 runtime swap / recompute** |
| preemption | 否 | 无 | — | vLLM 可抢占低优先序列释放块 | 无抢占机制 |
| block-level eviction | 部分 | SWA 窗外驱逐 + 环形覆盖 | 都能回收空间 | llama.cpp 驱逐是 SWA / 环形,非通用块级 LRU 驱逐 | 无通用块级驱逐策略 |

> "部分"项均为逻辑等效但缺乏物理分页语义;"否"项为机制层面不存在。

---

## 六、当前缺口总结(仅缺口,不含方案)

1. **抽象缺口**:无 block/page 抽象、无 block table 间接寻址、无 physical block pool。逻辑单元停留在 cell(1 token)+ 一整块预分配连续张量。

2. **读写路径缺口**:写侧已具 scatter 特征(`cpy_k` / `cpy_v` + `ggml_set_rows`),但**读侧仍是连续 `ggml_view_4d` + mask**(`get_k` / `get_v` + `set_input_kq_mask_impl`),未实现按块表的非连续 gather read——这是读写不对称的核心缺口。

3. **内存管理缺口**:无 runtime swap-in / swap-out 热路径,无 resident / swapped / dirty 状态机。`state_write_data` / `state_read_data` 仅服务离线 save / load。KV 在构造期全量预分配,运行时不增不减。

4. **共享机制缺口**:前缀共享是 cell seq 位图打标(同 stream 零拷贝、跨 stream 整块拷贝),缺少 vLLM 的块级引用计数与 COW,无法在块粒度上安全分叉共享。

5. **调度缺口**:无与内存压力联动的 preemption / recompute 策略,无优先级驱逐;空间回收仅靠 SWA 窗外驱逐与环形覆盖,defrag 已废弃。

6. **工程难点**:实现真正的 PagedAttention 非连续 paged read,会触及 attention kernel(各后端 fattn / matmul)与 ggml 计算图改造——非 CPU 后端 kernel 内部细节(CUDA / Metal / SYCL)**仍需验证**后才能评估改造面。

---

## 七、可用于报告的精炼表述

1. llama.cpp 的 KV cache 采用"构造期全量预分配的逐层连续张量 + cell 级元数据"设计,最小管理单位为约等于单 token 的 cell,元数据(pos / seq / shift / used / seq_pos)由 `llama_kv_cells` 维护,与张量存储分离。

2. llama.cpp KV 数据通路存在结构性不对称:写侧经 `cpy_k` / `cpy_v` 调用 `ggml_set_rows` 按 `slot_info.idxs` 散列写入,具 paged-style scatter 特征;读侧经 `get_k` / `get_v` 返回 `[0, n_kv)` 连续视图,再以 `set_input_kq_mask_impl` 置 -INFINITY 屏蔽无效 cell。

3. 与 vLLM 相比,llama.cpp 不具备 block/page 抽象、physical block pool、logical→physical block table 以及 PagedAttention 的非连续 gather read;其"连续读 + mask"在逻辑上等效但在物理上不分页。

4. llama.cpp 已具备多项 KV 优化:环形复用(`find_slot` / `v_heads`)、前缀共享(`seq_cp` + cell seq 位图)、SWA 驱逐(`llama_kv_cache_iswa`)、K-shift 位置修正、KV 量化(`type_k` / `type_v`)及 state 序列化;缺口集中在 block/page 级虚拟内存式管理。

5. KV 量化的解量化不在计算图层显式插入算子,而在 ggml kernel 内经 `ggml_type_traits` 完成;CPU FlashAttention 路径中 K 侧将 Q 量化到 K 的 `vec_dot_type` 在量化域点积(K 不还原 F32),V 侧经 `to_float` 还原后累加。

6. llama.cpp 的序列前缀共享为 cell seq 位图打标(同 stream 零拷贝、跨 stream 整块拷贝),不具备 vLLM 的块级引用计数与写时复制语义。

7. llama.cpp 当前无 runtime swap / recompute、无内存压力驱动的抢占与块级驱逐,碎片整理路径(`llama_kv_cells::mv()`)已废弃,空间回收依赖环形覆盖与 SWA 窗外驱逐。

8. KV cache 的实例化由模型架构在 `llama_model::create_memory`(`src/llama-model.cpp:1935`)统一裁决,唯一调用方为 `llama_context` 构造函数;Llama3(标准 attention、无 SWA)走 `new llama_kv_cache`,其 `type_k / type_v / kv_size / unified / offload` 等参数均来自 `cparams`。

---

## 附录:待验证项汇总

- **(a)** 非 CPU 后端(CUDA / Metal / SYCL)fattn / matmul kernel 内 KV dequant 细节;以及各后端**异步拷贝能力**(影响 runtime swap 预取重叠)。
- **(b)** ~~`state_read_data` 精确行号区间~~ —— **已确认**:`state_read_data` 止于 `src/llama-kv-cache.cpp:2353`。本项消除。
- **(c)** 跨 stream 拷贝在 `update` 中的完整执行序(`sc_info` → `ggml_backend_tensor_copy`)本轮未重读全程。
- **(d)** runtime swap 落地相关(详见 [kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md) 第五节):V 转置路径的换出 / 换入搬运代价;连续 view 在稀疏占用下的换入放大效应;绕开 meta 重分配实现"纯增量原位换入"的可行性;与 K-shift / SWA / seq_cp 现有不变量的协同。

其余结论均已对当前工作区源码核对。

