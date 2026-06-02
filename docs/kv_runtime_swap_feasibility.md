# KV runtime swap / offloading 可落地性验证

> 操作系统功能赛技术报告素材 · A 类方向落地性验证篇
>
> **定位**:基于 [docs/llama_kv_cache_analysis.md](llama_kv_cache_analysis.md) 与 [docs/kv_cache_gap_matrix.md](kv_cache_gap_matrix.md) 第六节,只验证 A 类方向("KV runtime swap / offloading + prefetch")能否基于 llama.cpp 当前 KV cache 架构做最小 demo。
>
> 验证日期:2026/05/31

---

## 一、验证目标

- 本次**只验证 A 类方向**的可落地性,不扩展到其他方向。
- **不写代码、不修改源码、不确定最终优化方案、不展开实现计划。**
- 只回答两个问题:(一)`state_write_data`/`state_read_data` 能否复用为 runtime swap 数据通道;(二)`get_k`/`get_v` 与 `build_attn_mha` 的连续读限制是否阻碍 runtime swap。
- 所有结论给出源码文件与函数,不确定处标注"仍需验证"。

> **本轮核对修正**:先前 analysis 文档把 state 路径描述为"仅离线全量 save/load"。逐行读源码后发现 `state_read_data` **已内置非连续 scatter 路径**(`src/llama-kv-cache.cpp:2241-2247`、`:2284-2290`、`:2339-2347`),可按 `sinfo.idxs` 把单个 cell 写回**精确物理位置**;`write_tensor`/`read_tensor` 经 `ggml_backend_tensor_get/set` 实现(`src/llama-context.cpp:2506-2535`),**后端无关**。这两点对 swap 可行性是正面证据,下文据实修正。

---

## 二、state_write_data / state_read_data 可复用性分析

> 源码:`state_write_data` `src/llama-kv-cache.cpp:1969`、`state_read_data` `src/llama-kv-cache.cpp:2187`、`state_read_meta` `:2068`、`cell_ranges_t` `src/llama-kv-cache.h:296`、入口 `state_write` `:1845` / `state_read` `:1898`、IO 实现 `src/llama-context.cpp:2497-2545`。

| 问题 | 源码位置 | 当前机制 | 对 runtime swap 的意义 | 结论 | 仍需验证 |
|---|---|---|---|---|---|
| Q1 最小读写粒度 | `state_write` `:1850-1895`;`cell_ranges_t` `:296` | 单位是 **per-stream 的多个连续 cell range**(`cr.data` = `vector<pair<begin,end>>`);按 seq_id 过滤构建 range;一次调用**遍历全部 layer** | swap 天然可按"cell range"换出/换入,粒度足够细 | 支持 cell range 级,**但层维度被绑定为"全层一次"** | 否 |
| Q2 能否按任意 cell 子集序列化 | 写 `:1996-2000`;读 scatter `:2241-2247` | 写侧按任意 range 集合 `io.write_tensor`;读侧 `is_contiguous()` 假则走 scatter,逐 cell 写回 `idxs[i]*row` | 可换出任意 cell 子集、换入时精确归位 | **可以**(写按 range,读按 idxs scatter) | 否 |
| Q3 能否按单层 KV 序列化 | 写 `:1980` / `:2004` / `:2033`;读 `:2212` / `:2252` / `:2295` | 三段循环均 `for (const auto & layer : layers)`,**无 layer 参数** | 想做"逐层 swap"须改造函数签名或抽出单层逻辑 | **当前不暴露**单层入口(全层一次) | 否 |
| Q4 K/V 序列化路径是否一致 | K `:1980-2001`;V 非转置 `:2003-2028`;V 转置 `:2029-2065` | K 恒按行(1 行=1 cell)连续写;V 非转置同 K;V 转置按元素维 `j` × range 双重循环 | K 与 V(非转置)搬运高效;V 转置搬运碎片化 | **不完全一致**:V 转置路径显著更碎 | 否 |
| Q5 V 转置 vs 非转置差异 | `v_trans` 来源 `create_memory`(`!cparams.flash_attn`);转置写 `:2055-2062`、读 `:2331-2347` | 非转置:每 cell 一段连续 `v_size_row`;转置:每 cell 每个 embedding 维一个 `v_size_el` 元素,循环 `n_embd_v_gqa × cell_count` | flash_attn 关→V 转置→swap I/O 退化为大量小粒度搬运 | **转置路径换入换出开销显著更高** | 转置下单 cell 实测搬运次数/带宽 **仍需验证** |
| Q6 改造为 swap 需补的状态 | `llama_kv_cells` 字段(`src/llama-kv-cells.h`,现有 pos/seq/shift/used/seq_pos) | 现无任何"驻留/换出"语义字段 | 必须新增页式元数据 | 需补 `resident`、`swapped`、`dirty`、`swap_offset`(后备存储偏移)、`last_access`(驱逐依据)、`block_id/page_id`(分页粒度);均为新增,**不破坏现有字段** | 新增字段与 find_slot/mask 协同的正确性 **仍需验证** |
| Q7 为何还不是 runtime swap | `state_read_meta` `:2068`(`seq_rm`/`clear(true)`/`find_slot`)、入口经 `llama_file` | ① 全层一次、无增量;② 由用户 save/load API 触发,非内存压力驱动;③ 全量恢复先 `clear(true)`(`:2141`)、单序列恢复走 `find_slot` 重新分配,**不是"原位换回"**;④ 走磁盘/设备 buffer,无内存分层与异步;⑤ 无 resident/dirty 追踪 | 缺的是"原位、增量、压力触发、异步"四点 | 当前是**离线持久化语义**,非热路径 swap | 否 |
| Q8 作为 swap 通道可复用程度 | 数据搬运 `write_tensor`/`read_tensor`(`src/llama-context.cpp:2506-2535`,`ggml_backend_tensor_get/set`);scatter 读 `:2241-2247` | 字节级搬运后端无关;range+scatter 寻址已具备 | 搬运原语可直接复用,编排逻辑需重写 | **中**:数据搬运层=高复用(后端无关、支持精确归位);meta/clear/find_slot 编排层=不适合复用,需新增压力触发+原位换入编排 | 否 |

---

## 三、get_k / get_v 连续读限制分析

> 源码:`get_k`/`get_v` `src/llama-kv-cache.cpp:1145-1195`、`get_n_kv` `:1129`、`set_input_kq_mask_impl` `:1434`、`build_attn_mha` `src/llama-graph.cpp:1953`、读 K/V 调用点 `src/llama-graph.cpp:2240-2241`、上层 `kv_cache_context::get_k/get_v` `src/llama-kv-cache.cpp:2444-2450`、`n_kv` 来源 `get_n_kv()` `src/llama-graph.cpp:1926`。

| 问题 | 源码位置 | 当前机制 | 对 runtime swap 的影响 | 结论 | 仍需验证 |
|---|---|---|---|---|---|
| Q1 是否强制连续 `[0, n_kv)` view | `get_k` `:1157`、`get_v` `:1180`/`:1189` | 用 `ggml_view_4d` 在整层张量上取连续切片,第三维长度 = `n_kv`;`n_kv` 由 `get_n_kv()`(`:1129`)按 `used_max_p1()` 向上 pad 到 ≥256 | attention 读的物理范围是 `[0, n_kv)` 连续区间,非按需逐 cell | **是**,强制连续 view | 否 |
| Q2 mask 是屏蔽还是避免读取 | `set_input_kq_mask_impl` `:1434`,置 `-INFINITY` | mask 只影响 softmax 数值(无效位贡献为 0),**不改变**读取/计算的物理范围 | 被 mask 的 cell 物理上**仍被读取** | **仅屏蔽,不避免读取** | 否 |
| Q3 swap-out 的 cell 在 attention 前是否必须换回 | `get_k/get_v` + `build_attn_mha`(`:2240-2243`)+ FA/mul_mat 内核 | 内核按连续 view 步长访问 `[0, n_kv)` 全部行;换出 cell 若落在该区间,其物理内存会被直接读到 | 若读区间含已换出 cell,**必须先换回**,否则读到脏/空数据 | **必须换回**落在 `[0, n_kv)` 内的 cell | 否 |
| Q4 不改 kernel 的最小可行策略 | `get_n_kv` `:1129`、`apply_ubatch` head 推进 `:1017`、`find_slot` `:818` | `n_kv` 取决于 `used_max_p1()`,即"最高被占用 cell + 1";读区间随高水位增长 | 不改 kernel 时,只能保证"读区间内全 resident" | ① **只能保证被读连续范围全部 resident**;② 不能只换入"有效 cell"而跳过区间内其他 cell(view 连续);③ 故换入须覆盖 `[0, n_kv)` 内涉及的 cell。staging buffer **非必需**(可原位换回到同一张量偏移) | 不改 kernel 时换入是否可只覆盖"当前 ubatch 实际 attend 的 stream 子集" **仍需验证** |
| Q5 "逻辑分页 + 连续 staging buffer"插桩点 | `get_k/get_v` `:1145`(返回 view 处)、`build_attn_mha` `:2240`(消费处)、`update` `:742` | 若引入 staging:在 get_k/get_v 之前把分散的 resident 块 gather 进一段连续 staging,再让 view 指向 staging | staging 可解耦"物理存储布局"与"读连续性" | 候选插桩:`get_k/get_v` 内部(改 view 目标)或其上游 `update`/解码循环(预先 gather);**改 get_k/get_v 返回目标=较集中的插桩点** | staging 与 cpy_k 写路径、K-shift 的相互作用 **仍需验证** |
| Q6 不做 staging、原 buffer 上换入换出是否可行 | `read_tensor` scatter `:2241-2247`、原位偏移 `idxs[i]*row` | 现有 scatter 读已能把 cell 写回其原物理槽位(`idxs[i]`),即"原位换入" | 原位换入与连续 view 天然兼容(换回后 view 自然命中) | **可行**:原 buffer 原位换入是最贴合现有架构的路径,无需 staging | 换入与正在进行的图计算的时序/同步边界 **仍需验证** |
| Q7 对 prefetch 的粒度影响 | `get_k/get_v` 逐层调用(`build_attn` 每层)、`n_kv` 连续区间、`seq_to_stream` | 读是"逐层 × `[0,n_kv)` 连续区间";解码按层顺序推进,访问模式可预测 | 预取应顺应该访问顺序 | 建议预取粒度:**按 layer + `[0, n_kv)` 连续区间**为主轴(贴合逐层连续读);可叠加"按 stream/sequence"做跨请求隔离;**按单 cell range** 预取收益受限于连续读 | 逐层预取与 ggml graph 的层执行实际重叠窗口 **仍需验证** |

---

## 四、最小 runtime swap demo 的必要条件

> 只列条件,不设计完整方案。每条标注最贴近的现有源码锚点。

### 1. 元数据条件
- cell/页级新增状态:`resident`、`swapped`、`dirty`、`swap_offset`、`last_access`、(可选)`block_id/page_id`。锚点:`llama_kv_cells`(`src/llama-kv-cells.h`)。
- 状态须与 `used`/`seq` 现有语义并存,且能被 `find_slot`(`:818`)、`apply_ubatch`(`:1017`)、`set_input_kq_mask_impl`(`:1434`)正确读取。

### 2. 数据搬运条件
- 复用后端无关搬运原语 `ggml_backend_tensor_get/set`(经 `write_tensor`/`read_tensor`,`src/llama-context.cpp:2506-2535`)。
- 复用 scatter 寻址(`state_read_data` `:2241-2247`)实现"原位换入"。
- 需要一块后备存储(host RAM / 文件 / 设备),持有换出字节与 `swap_offset` 映射。

### 3. 换入触发条件
- 在读取前保证 `[0, n_kv)`(`get_n_kv` `:1129`)区间涉及的 cell 全部 resident。
- 触发点候选:`update`(`:742`)阶段或解码循环逐层进入 `build_attn`(`src/llama-graph.cpp:2240`)之前。

### 4. 换出触发条件
- 由内存压力 / 容量水位驱动(当前完全缺失,需新增判断点)。
- 候选驱逐依据:`last_access` + 是否在当前 `[0, n_kv)` 读区间外;SWA 架构可借窗外回收钩子(`apply_ubatch` 内 SWA 清理)。

### 5. 预取条件
- 预取粒度以 **layer + `[0, n_kv)` 连续区间**为主轴(见三-Q7)。
- 需异步搬运 + 双缓冲,使换入 I/O 与上一层计算重叠;依赖 ggml backend 是否提供异步拷贝(**仍需验证**)。

### 6. 正确性检查条件
- 换入后必须保证 view 命中的物理槽位字节正确(类型/行宽校验已存在:`state_read_data` `:2223`/`:2274`/`:2317`)。
- 不得破坏 K-shift(`build_rope_shift`)、SWA(`is_masked_swa`)、`seq_cp`(`:406`)的现有不变量(见第五节风险)。
- 建议保留可关闭开关,默认行为与原版逐位一致。

### 7. 实验指标条件
- 峰值常驻物理内存(主指标,对应赛题"减少运行时物理内存")。
- 换出字节量 / 换入字节量 / 换入延迟 / 预取命中率。
- I/O 与计算重叠率(对应赛题"预取掩盖 I/O")。
- 端到端时延 / 吞吐相对基线的回退幅度。
- (精度类对照)输出与基线的一致性,确认无正确性破坏。

---

## 五、风险与不确定性

> 以下均标注"仍需验证",是继续深入前应优先解决的问题。

1. **state_read_data 是否适合增量换入**:**数据路径已澄清(第二轮验证)**——`state_read_data`(`:2187`)内部不调用 meta,可复用其 scatter 写回(`:2241-2247`),**需绕开 `state_read_meta`(`:2068`)的 `clear(true)`/`find_slot` 重分配编排,并在换出时保留物理槽位映射、换入时保持 cell pos/seq/shift 不动**;具体实现仍需验证。
2. **V 转置路径的换出/换入开销**:`v_trans = !flash_attn`;转置下搬运按 `n_embd_v_gqa × cell_count` 元素级碎片化(`:2055-2062`、`:2331-2347`),小粒度 I/O 可能吞噬 swap 收益。**风险等级:中——可通过限定 flash_attn=true / V 非转置规避**(该配置下 V 按 cell 连续搬运)。两种配置下的实测搬运代价仍需验证。
3. **连续 view 是否迫使大范围换入**:`n_kv` 按 `used_max_p1()` pad 到 ≥256(`:1129`/`:1134`),读区间随高水位单调增长;高水位下即便有效 cell 稀疏,换入也须覆盖整个区间——**对稀疏占用场景的换入放大效应仍需验证**。
4. **多 backend(CPU/Metal/CUDA)额外限制**:搬运层 `ggml_backend_tensor_get/set` 后端无关(正面),但 attention 内核与异步拷贝能力各后端不同;**非 CPU 后端的原位换入时序与异步预取支持仍需验证**(沿用 analysis 附录(a))。
5. **是否破坏现有机制**(第二轮已细化风险等级):
   - K-shift:换出 cell 被 shift 时其 `shift` 元数据与物理字节须一致,换入后需重放或保序——**高风险**,demo 应避开 context shift 场景;仍需验证;
   - SWA/iSWA:`kv_base`/`kv_swa` 双实例 + 窗外回收与换出驱逐可能语义重叠/冲突——**中~高风险**,demo 用标准 attention 避开;仍需验证;
   - seq_cp:同 stream 位图共享的 cell 被换出时,多 seq 引用的回写一致性——**中~高风险**,demo 暂不支持 seq_cp;仍需验证;
   - find_slot:换出 cell 若被 `can_use`(`:962`)判为可覆盖会被新 token 覆盖、丢失换出数据——**高风险,必须处理**(将 `swapped` 态纳入 `can_use` 判定);仍需验证。

---

## 六、对现有文档的回填建议

### 回填 [docs/llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)
- **第四节 机制 8(state save/load)**:修正"仅离线全量"表述——补充 `state_read_data` 已具非连续 scatter 原位写回路径(`:2241-2247`/`:2284-2290`/`:2339-2347`),搬运经后端无关的 `ggml_backend_tensor_get/set`;但 meta 编排(`clear(true)`/`find_slot`)仍是全量/重分配语义。
- **第四节 机制 3(连续读)**:补充"mask 仅屏蔽不避免读取""读区间 `[0,n_kv)` 由 `used_max_p1` pad≥256 决定"两点,作为 swap 换入范围约束依据。
- **第二节 数据结构表**:`llama_kv_cells` 行补注"现无 resident/swapped/dirty 等换出态字段"。
- **附录待验证项**:(b)`state_read_data` 结束行号消项(已确认止于 `:2353`);新增 V 转置搬运代价、连续 view 换入放大两项。

### 回填 [docs/kv_cache_gap_matrix.md](kv_cache_gap_matrix.md)
- **第二节缺口总览表 缺口 4/12**:难度由"中"维持但补注"数据搬运层已后端无关且支持原位 scatter,复用度上调为高;编排层需重写"。
- **第二节 缺口 5(resident/swapped/dirty)**:确认为方向 2/3 的前置必备项,深入程度维持 A。
- **第五节优先级**:A 类(方向 2 swap + 方向 3 prefetch)结论**得到正向支持**(搬运原语可复用、原位换入与连续读兼容、不需改 attention kernel),维持 A;C 类(非连续 paged read)维持——本验证未触及 kernel 改造,结论不变。
- **第六节任务 1**:标记为"已完成本轮",结论回填至本文件。

---

## 七、结论

**runtime swap / offloading 方向:值得继续验证。**

依据:① 数据搬运原语 `ggml_backend_tensor_get/set` 后端无关、可直接复用(`src/llama-context.cpp:2506-2535`);② `state_read_data` 已具非连续 scatter 原位写回(`src/llama-kv-cache.cpp:2241-2247`),"原位换入"与连续 view 天然兼容,**最小 demo 不需改 attention kernel**;③ 触发点(`update` `:742` / 解码循环)与元数据载体(`llama_kv_cells`)清晰。

保留项(决定可行性上界,须在深入阶段解决):V 转置碎片化搬运代价(中风险,可由 flash_attn=true 规避)、连续 view 在稀疏占用下的换入放大、与 K-shift(高)/find_slot 覆盖(高)/SWA/seq_cp 的不变量协同——均标注"仍需验证"。

**第二轮风险验证(2026/05/31)增强了该方向的可信度**:纯增量换入数据路径已澄清(绕开 meta、复用 scatter 写回),全部主要风险均已定位到具体源码并有 demo 规避路径,prefetch 已识别出 decode 主循环这一可重叠插桩点。详见 [kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md)。结论维持"**值得继续验证**"。

> 本结论为"值得继续验证",**非**确定采用;不含实现方案与代码。

---

## 八、更新记录

- **2026/05/31 第一轮验证**:本文件初稿,确认 state 搬运通道可复用、连续读约束、最小条件。
- **2026/05/31 回填(基于 [kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md) 第二轮风险验证)**:
  - 第五节风险 1:纯增量原位换入由"仍需验证"改为"数据路径已澄清,需绕开 meta,具体实现仍需验证";
  - 第五节风险 2:V 转置标注"中风险,可由 flash_attn=true / V 非转置规避";
  - 第五节风险 5:补 find_slot 覆盖语义冲突=高、K-shift=高、seq_cp/SWA=中高;
  - 第七节结论:维持"值得继续验证",说明第二轮验证增强可信度;
  - 未修改源码;未确定最终方案;**下一步可进入 minimal demo plan 设计**。

