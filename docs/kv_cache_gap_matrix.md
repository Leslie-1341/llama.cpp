# llama.cpp KV cache 缺口矩阵与技术路线初筛

> 操作系统功能赛技术报告素材 · 缺口分析与方向初筛篇
>
> **定位**:本文档基于 [docs/llama_kv_cache_analysis.md](llama_kv_cache_analysis.md) 的现状分析结论,进一步整理"现有能力—缺口—参考技术—插桩点—验证指标"矩阵,为后续判断比赛优化方向做准备。
>
> **约束**:不重复解释基础机制;不直接确定最终方案;不展开实现计划;不写代码。所有源码相关判断保留文件名 / 函数名,不确定处标注"仍需验证"。
>
> 生成日期:2026/05/31

---

## 一、分析目标

本文档的目标边界:

1. **不重复**解释 llama.cpp KV cache 基础机制(见 [llama_kv_cache_analysis.md](llama_kv_cache_analysis.md) 第二、三、四节)。
2. **不直接确定**最终优化方案,也不锁定唯一技术路线。
3. **只从源码现状出发**,识别可进一步优化的缺口,并评估每个缺口与赛题的契合度。
4. 将缺口与成熟技术思路建立对应关系:**PagedAttention、vLLM、FlexInfer、KV cache offloading、KV cache quantization、StreamingLLM / H2O** 等。对应关系仅用于"参考定位",不代表采纳。

> 说明:本阶段属于"理解现有机制 + 方向初筛",仍为探索性分析,不涉及任何代码改动或提交。

---

## 二、核心缺口总览表

> 列含义:契合度 / 难度 / 深入程度均为**初筛主观评级**(高 / 中 / 低),用于排序,非最终结论。

| # | 缺口类别 | llama.cpp 当前机制 | 当前局限 | 对应技术 / 论文思路 | 契合赛题 | 可能插桩文件 / 函数 | 工程难度 | 风险 | 建议深入 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 缺 block/page 抽象 | cell(单 token)+ 每层连续大张量 `kv_layer.k/v`(`llama-kv-cache.h:215`) | 逻辑单元过细,无固定块抽象 | PagedAttention block | 中 | `llama-kv-cache.cpp` 构造(:80)、`find_slot`(:818) | 高 | 改动面大,牵动读写两侧 | B |
| 2 | 缺 block table(逻辑→物理映射) | `slot_info.idxs` 仅用于写侧 scatter(`llama-kv-cache.h:34`) | 无持久逻辑→物理映射供读侧 gather | vLLM block table | 中 | `slot_info`、`cpy_k/cpy_v`(:1197)、`get_k/get_v`(:1145) | 高 | 需新数据结构贯穿生命周期 | B |
| 3 | 缺 physical block pool | 构造期全量预分配单块连续 buffer(:80) | 运行时不增不减,无独立物理块池 | vLLM block allocator | 高 | `llama_kv_cache` 构造、buffer 分配段(:254 区域,仍需验证行号) | 高 | 预分配模型改动深 | B |
| 4 | 缺 runtime swap-in/out | `state_write_data`(:1969)/`state_read_data`(:2187):**数据搬运层已可复用**(后端无关 + scatter 原位写回),缺的是 runtime 编排 | 无运行时换入换出热路径(meta 编排 + 触发机制需重写) | KV offloading、FlexInfer、vLLM swap | **高** | `state_write_data`/`state_read_data`、`update`(:742) | 中(搬运高复用,编排需重写;纯增量换入路径已澄清) | **find_slot 覆盖语义与 swapped 态冲突=高**;K-shift=高;seq_cp/SWA=中高;V 转置碎片化=中(可规避);连续 view 换入放大(仍需验证) | **A**(值得继续验证) |
| 5 | 缺 resident/swapped/dirty 状态 | `llama_kv_cells` 有 pos/seq/used,无换出态 | 无法表达"已换出/需回写" | 操作系统页表 present/dirty 位 | **高** | `llama-kv-cells.h` 字段(:458 区域) | 中 | 状态机需与 find_slot/mask 协同 | **A** |
| 6 | 缺异步预取机制 | 无 | 无预取,无 I/O 与计算重叠 | 预取 / double buffering、FlexInfer | **高** | `update`(:742)、解码主循环(`llama-context.cpp` decode) | 中 | 预取命中率与时序难调 | **A** |
| 7 | 缺非连续 paged read | `get_k/get_v` 连续 `ggml_view_4d`(:1145) | 读侧不能按块表 gather | PagedAttention kernel | 低 | `get_k/get_v`、`build_attn_mha`(`llama-graph.cpp:1953`)、各后端 fattn kernel | **很高** | 须改 attention kernel,跨后端 | C |
| 8 | 缺 block-level refcount/COW | `seq_cp` cell 位图打标(:406) | 同 stream 零拷贝、跨 stream 整块拷贝,无块级 refcount/COW | vLLM COW、prefix caching | 中 | `seq_cp`(:406)、cell `seq` 位图(`llama-kv-cells.h`) | 中 | 引用计数正确性要求高 | B |
| 9 | 缺内存压力联动 preemption/recompute | 无 | 无抢占、无重算恢复 | vLLM preemption + recompute | 中 | `init_batch`/`prepare`(:676 区域)、`find_slot` 失败路径 | 高 | 调度逻辑复杂,易引入正确性 bug | C |
| 10 | 读侧连续 view+mask 的无效计算 | `set_input_kq_mask_impl` 置 -INFINITY(:1434) | n_kv 大且稀疏时仍全区间读 + mask | 稀疏 attention、块级跳过 | 中 | `get_n_kv`、`set_input_kq_mask_impl`(:1434) | 高 | 与 kernel 耦合 | C |
| 11 | KV 量化仅统一 type_k/type_v | 全局 `type_k/type_v`(来自 cparams) | 无冷热混合精度、无逐块精度 | KV quant、H2O(冷热区分) | 中 | `create_memory`(`llama-model.cpp:1935`)、cpy/get、ggml kernel type traits | 高 | 混合精度须 kernel 支持 | C |
| 12 | state save/load 未作 runtime swap 通道 | `state_read_data` **已具非连续 scatter 原位写回**(:2241-2247)+ 后端无关搬运 `ggml_backend_tensor_get/set`(`llama-context.cpp:2506-2535`) | 数据通路具备 swap 基础;但 meta 编排仍 `clear(true)`/`find_slot` 重分配,无 resident/dirty/swap_offset、无压力触发与异步 | mmap/分页换出、增量 checkpoint | **高** | `state_write_data`/`state_read_data`、`state_read_meta`(:2068)、按 cell range 序列化逻辑 | 中(搬运可复用,编排+触发需重写;纯增量换入路径已澄清=绕开 meta) | find_slot 覆盖语义冲突=高;V 转置碎片化=中(可规避);连续 view 换入放大(仍需验证) | **A**(值得继续验证) |

> 评级口径:契合赛题"高"= 直接对应"减少物理内存 / 换入换出 / 预取";难度按是否需改 ggml kernel 与跨后端判断;深入程度 A>B>C>D(见第五节)。

---

## 三、按技术方向归类

> 每个方向回答五问:解决什么问题 / llama.cpp 已有基础 / 缺少关键机制 / demo 最小切入点 / 是否需改 attention kernel / 偏系统还是偏算法。

### 1. PagedAttention / block 化 KV 管理
- **解决问题**:用固定块 + 块表把逻辑连续序列映射到物理非连续块,提升内存利用、支持灵活分配与共享。
- **已有基础**:写侧 scatter(`cpy_k/cpy_v` + `ggml_set_rows`,:1197)、`slot_info.idxs` 行映射、cell 级元数据。
- **缺少机制**:block 抽象、持久 block table、physical block pool、读侧非连续 gather(`get_k/get_v` 仍连续,:1145)。
- **demo 最小切入点**:在不改 kernel 前提下先做"块级元数据 + 块表"管理层(逻辑层),物理仍连续;非连续读作为后续可选。
- **是否改 attention kernel**:做到真正非连续读**需要**;仅做逻辑块管理层**不需要**。
- **偏系统 / 算法**:偏**系统**(内存管理 + 间接寻址)。

### 2. KV cache runtime swap / offloading
- **解决问题**:把不活跃 KV 换出到更慢更大的存储(host RAM / 磁盘),降低运行时常驻物理内存。
- **已有基础(经第一轮验证强化)**:`state_write_data`/`state_read_data`(:1969/:2187)可按 cell range 序列化 KV 字节;**`state_read_data` 已内置非连续 scatter 路径**(`:2241-2247`),可按 `sinfo.idxs` 将 cell 写回精确物理槽位("原位换入"已具底层能力);**搬运经 `ggml_backend_tensor_get/set` 后端无关**(`llama-context.cpp:2506-2535`);`update`(:742)为状态变更统一入口;backend offload 提供设备放置开关。
- **缺少机制**:runtime swap-in/out 热路径、resident/swapped/dirty/swap_offset 状态、压力触发与回写策略、异步搬运;meta 编排层(`clear(true)`/`find_slot` 重分配)不适合直接复用,需重写。
- **demo 最小切入点**:复用 state 搬运原语 + scatter 原位写回做"按 cell range 的换出/换入",在 `update` 或解码循环挂载触发点;**绕开 `state_read_meta` 的重分配**做纯增量换入(**第二轮验证:数据路径已澄清,可绕开 meta,保持 cell pos/seq/shift 不动只搬字节**)。
- **是否改 attention kernel**:**不需要**(原位换回到原张量偏移与连续 view 天然兼容,staging buffer 非必需)。
- **偏系统 / 算法**:偏**系统**(贴近虚拟内存/换页),与赛题最契合。
- **结论状态**:**值得继续验证**(非最终采用);第二轮风险验证(find_slot 覆盖=高、K-shift=高、seq_cp/SWA=中高、V 转置=中可规避)均已定位并有 demo 规避路径。详见 [kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md)。

### 3. KV cache prefetch / I/O latency hiding
- **解决问题**:用异步预取把换入 I/O 与计算重叠,掩盖 swap 延迟。
- **已有基础**:解码循环逐层顺序访问 KV,访问模式可预测(顺序性强,利于预取);ggml backend 已有异步拷贝原语(仍需验证具体 API)。
- **缺少机制**:预取调度、双缓冲、预取命中/未命中度量。
- **demo 最小切入点**:在方向 2 的换入路径上加"提前一层/几步预取"。**依赖方向 2 先成立**。
- **是否改 attention kernel**:**不需要**。
- **偏系统 / 算法**:偏**系统**(I/O 调度 + 重叠),直接命中赛题"预取掩盖 I/O"。

### 4. KV cache quantization / mixed precision
- **解决问题**:降低 KV 每 token 字节占用。
- **已有基础**:已支持统一 `type_k/type_v`(q8_0/q4_0),dequant 在 ggml kernel 内经 type traits 完成(`ggml-cpu/ops.cpp:8214`)。
- **缺少机制**:冷热混合精度、逐块/逐序列精度差异化。
- **demo 最小切入点**:难——混合精度要 kernel 支持多类型并存。
- **是否改 attention kernel**:**很可能需要**(混合精度读取)。
- **偏系统 / 算法**:偏**算法**(精度-容量权衡),与"虚拟内存"主题关联弱。

### 5. Sliding window / attention sink / KV eviction
- **解决问题**:只保留近窗 + 少量 sink token 的 KV,限制 KV 总量。
- **已有基础**:`llama_kv_cache_iswa`(`llama-kv-cache-iswa.h:14`)已实现 SWA 驱逐;`is_masked_swa` 窗判定。
- **缺少机制**:通用(非架构绑定)的重要性驱逐(如 H2O 的注意力分数驱逐)、attention sink 显式保留策略。
- **demo 最小切入点**:基于现有 SWA 框架扩展驱逐策略,但 **Llama3 标准 attention 默认不走 iSWA 路径**(局限明确)。
- **是否改 attention kernel**:基础 SWA **不需要**;基于分数的驱逐需采集注意力分数,可能涉及 kernel/图层。
- **偏系统 / 算法**:**算法为主**(驱逐策略),系统为辅。

### 6. Prefix reuse / block sharing / copy-on-write
- **解决问题**:多请求共享公共前缀 KV,降低重复存储与计算。
- **已有基础**:`seq_cp`(:406)同 stream 通过 cell seq 位图零拷贝共享。
- **缺少机制**:块级引用计数、COW、跨 stream 零拷贝共享(当前跨 stream 整块拷贝)。
- **demo 最小切入点**:在 cell/块元数据上加 refcount,改造 `seq_cp` 跨 stream 分叉。
- **是否改 attention kernel**:**不需要**(共享是元数据层)。
- **偏系统 / 算法**:偏**系统**(引用计数 + 写时复制),但与"减少物理内存"主线相关性中等。

### 7. Recompute-based recovery
- **解决问题**:内存压力下丢弃部分 KV,需要时重算而非换入。
- **已有基础**:几乎无(无抢占、无重算路径)。
- **缺少机制**:抢占触发、被弃序列的重算调度。
- **demo 最小切入点**:风险高,通常作为 swap 的替代/补充策略,**不建议作为主线**。
- **是否改 attention kernel**:**不需要**,但需改调度与批处理逻辑。
- **偏系统 / 算法**:偏**系统调度**,工程复杂度高。

---

## 四、与赛题目标的匹配度

> 赛题目标:分析 LLM 推理访存行为;用虚拟内存 / 按需加载 / 数据换出减少运行时物理内存;用预取掩盖 I/O 延迟;面向嵌入式 / 边缘设备;强调系统完成度与可演示性。
>
> 下表"✓/✗/部分"为初筛判断;实验指标为建议采集项。

| 技术方向 | 减少物理内存 | 涉及虚拟内存思想 | 涉及换入换出 | 体现预取 / I/O 掩盖 | 适合 llama.cpp | 适合比赛 demo | 需采集的实验指标 |
|---|---|---|---|---|---|---|---|
| 1. PagedAttention / block 化 | 部分(利用率) | ✓(块表间接寻址) | 间接支持 | ✗(本身不预取) | 中(读侧改动大) | 中 | 内存利用率、碎片率、块表开销 |
| 2. runtime swap / offloading | **✓** | **✓**(换页语义) | **✓** | 间接(为预取铺路) | **高**(可复用 state 通道) | **高** | 峰值常驻内存、换出量、换入延迟、命中率 |
| 3. prefetch / I/O 掩盖 | ✗(不直接降内存) | ✓ | 依赖换出 | **✓** | **高** | **高** | I/O 与计算重叠率、预取命中率、端到端时延 |
| 4. quantization / 混合精度 | **✓** | ✗ | ✗ | ✗ | 中 | 中 | KV 字节占用、困惑度/精度损失 |
| 5. SWA / attention sink / 驱逐 | ✓(限总量) | 部分 | ✗ | ✗ | 中(Llama3 默认不走 iSWA) | 中 | KV 占用上界、长文精度 |
| 6. prefix reuse / 共享 / COW | ✓(去重) | 部分(共享映射) | ✗ | ✗ | 中 | 中 | 共享命中率、内存节省、拷贝次数 |
| 7. recompute 恢复 | ✓(以算换存) | 部分 | 替代换入 | ✗ | 低 | 低 | 重算开销、吞吐损失 |

**匹配度小结**:方向 2(runtime swap)、3(prefetch)与赛题主线(虚拟内存 / 换出 / 预取 / 降低物理内存 / I/O 掩盖)**最直接对应**,且都**不需改 attention kernel**、可复用现有 state 搬运原语(后端无关 + scatter 原位写回)与 update 入口,系统完成度与可演示性好。**第一轮源码验证已为方向 2 提供正向证据,但结论仍为"值得继续验证",非最终采用**(详见 [kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md))。方向 1(block 化)是它们的"理论母体",但完整实现牵动读写两侧、性价比偏低。方向 4/5/6 偏算法或与主线相关性中等,适合作为辅助指标或对照项。

---

## 五、初步优先级排序

> 仅"初步优先级",非最终方案。等级:A 必须深入 / B 有价值备选 / C 理论相关但工程风险大 / D 暂不深入。

### A 类(强相关,后续必须深入研究)
- **方向 2 — KV runtime swap / offloading**:直击赛题"换出降低物理内存";**两轮验证确认**可复用 `state_write_data`/`state_read_data`(:1969/:2187)的搬运原语与 scatter 原位写回(:2241-2247)、`update`(:742)触发点;纯增量换入数据路径已澄清(绕开 `state_read_meta`);不改 kernel;可演示峰值内存下降。**结论:值得继续验证(非最终采用)。**
  - **最小 demo 边界(第二轮收敛)**:单 sequence、flash_attn=true、V 非转置、标准 attention、无 SWA/iSWA、无 K-shift、单 backend、同步 swap、固定窗口——恰好规避全部高/中风险项。
- **方向 3 — prefetch / I/O 掩盖**:直击赛题"预取掩盖 I/O";KV 逐层顺序访问利于预取;预取主轴建议按 layer + `[0,n_kv)` 连续区间;与方向 2 天然组合成完整"换出 + 预取"故事。**依赖方向 2 先成立,且依赖后端异步拷贝能力(仍需验证)。**
- **配套缺口 5(resident/swapped/dirty 状态)**:是方向 2/3 的前置数据结构基础,须随之深入(`llama-kv-cells.h`,现确认无任何换出态字段)。

### B 类(有价值,可作辅助或备选)
- **方向 1 — block 化 KV 管理(仅逻辑层)**:作为 swap 的管理粒度载体(按块换出比按 cell 更高效),可作为方向 2 的增强,但不必追求非连续读。
- **方向 6 — prefix reuse / 共享 / COW**:可作为"内存节省"的辅助展示项,与主线协同但非核心。

### C 类(理论相关,工程风险大)
- **方向 7 — non-contiguous paged read(缺口 7)**:须改各后端 attention kernel,跨后端工作量大,**仍需验证**非 CPU kernel 改造面,demo 周期内风险高。
- **方向 4 — 混合精度 KV**:需 kernel 支持多类型并存,偏算法,与主线弱相关。
- **方向 9/10 — preemption、读侧无效计算消除**:调度复杂或与 kernel 深耦合。

### D 类(暂不建议深入)
- **方向 7 — recompute 恢复**:工程复杂、与"虚拟内存换出"主线重叠且更难演示,暂不作为主线或备选主力。

---

## 六、下一步源码阅读建议

> 依据缺口矩阵,下一轮聚焦"A 类方向能否落地"的源码验证。每项给出:读哪些文件、验证什么、回填到 analysis 文档何处。

### 1. state 序列化能否复用为 runtime swap 通道(对应 A 类方向 2)—— ✅ 已完成第一轮验证
- **状态**:已完成,结论见 [kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md) 第二节。
- **已确认**:数据搬运层可复用度高(后端无关 `ggml_backend_tensor_get/set`);`state_read_data` 已具非连续 scatter 原位写回(:2241-2247);`state_read_data` 止于 :2353;最小粒度为 per-stream 多 cell range,但层维度被绑定为"全层一次"(无单层入口);V 转置路径搬运碎片化。
- **结论**:数据通道可复用,meta 编排与触发机制需重写;方向"值得继续验证"。

### 1b. state 通道的纯增量换入与代价 —— ✅ 已完成第二轮风险验证
- **状态**:已完成,结论见 [kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md)。
- **已确认**:纯增量原位换入数据路径可绕开 `state_read_meta`、复用 `state_read_data` scatter 写回(须保留物理映射、保持 pos/seq/shift 不动);V 转置=中风险(可由 flash_attn=true 规避);find_slot 覆盖语义=高风险、K-shift=高风险、seq_cp/SWA=中高风险;prefetch 插桩点首选 decode 主循环(`llama-context.cpp:1712`)。
- **遗留(实验类)**:V 转置实测搬运耗时、`[0,n_kv)` 换入放大比例、后端异步能力、输出一致性、find_slot 协同正确性——见该文件第八节。

### 2. get_k/get_v 与 build_attn_mha 的连续读限制(对应缺口 7 边界)—— ✅ 已完成第一轮验证
- **状态**:已完成,结论见 [kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md) 第三节。
- **已确认**:`get_k`/`get_v`(:1145)强制返回 `[0,n_kv)` 连续 view;`n_kv` 由 `used_max_p1()` pad≥256(:1129/:1134);mask 仅屏蔽数值不避免读取;落在读区间内的换出 cell **必须先换回**;**不改 attention kernel 即可做最小 demo**(原位换回与连续 view 兼容,staging 非必需)。
- **遗留**:不改 kernel 时换入是否可只覆盖当前 ubatch 实际 attend 的 stream 子集(仍需验证)。

### 3. type_k/type_v 与 KV 量化的可扩展性(对应 C 类方向 4)
- **读**:`create_memory`(`llama-model.cpp:1935`)中 type 传入链;ggml type traits 注册(`ggml-cpu/ops.cpp:8214` 周边)。
- **验证**:单一 cache 内能否并存多种 KV 类型?per-layer 是否已有差异化先例?
- **回填**:第四节"机制 7"补充"混合精度可行性"。

### 4. SWA / iSWA 的驱逐逻辑(对应 B 类方向 5)
- **读**:`src/llama-kv-cache-iswa.cpp`(`init_batch`、`get_can_shift`)、`is_masked_swa`、`apply_ubatch`(:1017)中 SWA 窗外清理。
- **验证**:驱逐触发条件与 cell 回收时机;能否借其"窗外回收"钩子挂载通用换出?Llama3 路径确认不启用 iSWA。
- **回填**:第四节"机制 6"补充驱逐触发细节。

### 5. backend offload / KQV offload 的边界(对应方向 2 设备维度)
- **读**:`create_memory` 中 `offload=cparams.offload_kqv` 传入;`build_attn_mha` 内 `ggml_backend_sched_set_tensor_backend(...backend_cpu)` 回退;buffer 分配段(:254 区域,**行号仍需验证**)。
- **验证**:KV 张量 buffer 的设备归属是整缓存还是可分层/分块?offload 与潜在 swap 的协同/冲突边界。
- **回填**:第四节"机制 9"补充粒度边界;第二节数据结构表 buffer 项补注。

---

## 七、更新记录

- 本次基于 [docs/llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)(现状分析篇)生成。
- 本次**新增**内容为"缺口矩阵与技术路线初筛"(缺口总览表、技术方向归类、赛题匹配度、初步优先级、下一步源码阅读建议),用于方向判断准备。
- **未做任何代码修改**,仅新增本分析文档。
- **未确定最终优化方案**,优先级为初筛,A/B/C/D 分级可随后续源码验证调整。
- 源码点沿用 analysis 文档已核对结论;新引入但未逐行复核的行号(buffer 分配段 :254 区域)已标注"仍需验证"。

**回填修订(基于 [kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md) 第一轮验证)**:
- 缺口 4/12:旧表述"仅离线全量"改为"数据搬运层可复用(后端无关 + scatter 原位写回),meta 编排与触发机制需重写";难度补注、风险列新增 V 转置碎片化与连续 view 换入放大。
- 方向 2:补充正向证据(scatter 原位写回 `:2241-2247`、后端无关搬运 `llama-context.cpp:2506-2535`),结论标注"值得继续验证(非最终采用)"。
- A 类优先级:维持 A,标注结论性质为"值得继续验证"。
- 第六节任务 1、2:标记"已完成第一轮验证",新增任务 1b(纯增量换入与搬运代价的第二轮验证)。
- `state_read_data` 结束行号确认 :2353。

**回填修订二(基于 [kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md) 第二轮风险验证)**:
- 缺口 4/12 风险列:补 find_slot 覆盖语义冲突=高、K-shift=高、seq_cp/SWA=中高、V 转置=中(可规避);难度列补"纯增量换入路径已澄清(绕开 meta)"。
- 方向 2、A 类优先级:补两轮验证结论与最小 demo 边界(单 seq + flash_attn=true + V 非转置 + 无 SWA/K-shift + 单 backend + 同步 swap + 固定窗口);维持 A、维持"值得继续验证"。
- 第六节任务 1b:标记"已完成第二轮风险验证"。
- 未修改源码,未确定最终方案,**下一步可进入 minimal demo plan 设计**。

---

> **下一轮触发条件**:可进入"最小 demo plan 设计"(在第二轮收敛的边界条件内),或先执行 [kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md) 第八节的实验类验证。

