# KV runtime swap / offloading 最小 demo 设计文档

> 操作系统功能赛技术报告素材 · 最小 demo 设计篇(仅设计,不实现)
>
> **定位**:基于 [docs/llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)(现状)、[docs/kv_cache_gap_matrix.md](kv_cache_gap_matrix.md)(缺口矩阵)、[docs/kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md)(可行性)、[docs/kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md)(第二轮风险验证),设计一个不改 attention kernel、风险可控、能体现"减少运行时物理内存 + 换出/换入 + 后续可接预取"的最小 demo。
>
> **约束**:只做设计,不写代码、不修改源码、不进入实现。所有设计基于已有源码分析,不确定处标注"仍需验证"。
>
> 设计日期:2026/05/31

---

## 一、设计目标

1. **不完整复刻 vLLM PagedAttention**:不实现 physical block pool、block table、non-contiguous paged read。
2. **借鉴 PagedAttention 的 KV 分页 / 换页思想**:逻辑位置与物理驻留分离、按需换入、冷数据换出。
3. **基于 llama.cpp 现有机制**:复用 cell 级 KV cache(`llama_kv_cells`)与 state 数据搬运路径(`state_read_data` 的 scatter 原位写回,`src/llama-kv-cache.cpp:2241-2247`;后端无关搬运 `ggml_backend_tensor_get/set`,`src/llama-context.cpp:2506-2535`)。
4. **先验证同步 runtime swap 的正确性与内存收益**:确认"换出降低峰值物理内存"且输出与 baseline 一致。
5. **后续再扩展异步 prefetch / I/O latency hiding**:预取作为第二阶段,插桩点初判为 decode 主循环(`src/llama-context.cpp:1712`),**异步能力仍需验证**。

> 该 demo 的价值在于**用最小改动证明"llama.cpp 可承载 runtime KV 换出/换入"这一系统命题**,而非追求性能最优或机制完整。

---

## 二、demo 边界条件

> 边界来自 [kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md) 第七节的收敛结果,每条均为规避特定风险而设。

| 限制条件 | 理由(对应风险/源码) |
|---|---|
| 单 sequence | 规避 `seq_cp` 共享 cell 的引用一致性(`:406`,中~高风险) |
| flash_attn=true / V 非转置 | 规避 V 转置碎片化搬运(`v_trans=!flash_attn`,`:94`;转置慢路径 `:2331-2348`,中风险) |
| 标准 attention | 保持读路径为单一 `llama_kv_cache`,不引入 hybrid/recurrent 分支 |
| 不支持 SWA / iSWA | 规避窗外回收与换出驱逐语义重叠(`is_masked_swa` `:977`);Llama3 默认即非 iSWA |
| 不支持 K-shift | 规避 shift 元数据与换出字节一致性(高风险);即不做 context shift |
| 不支持 seq_cp / prefix sharing | 同"单 sequence",消除共享 cell refcount 难题 |
| 单 backend(建议 CPU 或单一指定后端) | 规避多后端异步能力差异(仍需验证);搬运 `ggml_backend_tensor_get/set` 单后端行为确定 |
| 同步 swap | 先验证正确性与内存收益,异步预取后置 |
| 固定 cell range / 固定窗口换出 | 降低 `[0,n_kv)` 换入放大与 swap_offset 映射复杂度,便于度量 |
| 不改 attention kernel | 原位换回与连续 view 天然兼容,staging buffer 非必需 |
| 不追求性能最优 | 只验证正确性、内存变化、I/O 开销 |

**边界小结**:首个 demo 收敛到"单 seq + flash_attn=true(V 非转置)+ 标准 attention(无 SWA/无 K-shift)+ 单 backend + 同步 swap + 固定窗口",恰好规避第二轮风险验证中全部高/中风险项,同时仍能演示赛题核心"峰值物理内存下降"。

---

## 三、核心设计思路

> 高层逻辑描述,不含代码;源码锚点均来自已验证结论。

- **给 cell 增加换出态**:在 `llama_kv_cells`(`src/llama-kv-cells.h`,现仅有 pos/seq/shift/used/seq_pos)旁挂一组与现有字段并存的状态——`resident`(是否在张量中)、`swapped`(是否已换出)、`dirty`(换入后是否被写,决定是否需回写)、`swap_offset`(后备存储偏移)、`last_access`(冷热判据)。**不改动现有 pos/seq/shift 语义**,只新增并行状态。
- **选择换出的 cell range**:demo 用**固定窗口策略**——例如保留最近 N 个 token 对应 cell 为 resident,把更早且确定不再进入近期读区间的 cell range 选为换出候选(依据 `last_access` 与是否落在预期 `[0,n_kv)` 内)。冷选择策略可后续替换为 LRU/重要性,demo 阶段从简。
- **保存换出 cell 的 K/V 字节**:对选中 cell range,按其物理 idx 逐层读出 K/V 字节(类比 `state_write_data` 的 `write_tensor`,但按物理 idx 而非 seq 过滤),写入后备存储并记录 `swap_offset`;随后该物理区域可视为"可被内存管理回收/复用"的目标(demo 阶段是否真正释放物理内存,取决于后备存储与 buffer 关系,**仍需验证**)。
- **读取前确保 resident**:每轮 decode 进入 attention 前,计算本轮 `[0,n_kv)`(`get_n_kv` `:1129`)内**实际需要访问的 cell**,若其中有 `swapped` 的,先同步换入。因 mask 仅屏蔽数值不避免物理读取(`set_input_kq_mask_impl` `:1434`),**落在读区间且会被内核连续读到的 cell 必须 resident**。
- **复用 scatter 原位写回思想**:换入时构造指向**原物理 idx** 的 `sinfo`,复用 `state_read_data`(`:2187`)的 scatter 写回逻辑(`:2241-2247`)把字节写回原槽位——即"原位换入"。**须绕开 `state_read_meta`(`:2068`)的 `clear(true)`/`find_slot` 重分配**,并保持该 cell 的 pos/seq/shift 不动。
- **为何不需要 staging buffer**:原位换入直接写回 KV 张量原偏移,换回后 `get_k/get_v`(`:1145`)的连续 view 自然命中,无需把分散块先 gather 进连续暂存区。
- **为何不需要改 attention kernel**:读路径(连续 view + mask)与内核完全不变;swap 只在"读之前"保证字节就位,对内核透明。
- **为何先用同步 swap**:同步路径时序简单、正确性易验证(换入完成才进入计算),先确立"内存收益 + 输出一致"两个结论;异步预取(掩盖 I/O)作为第二阶段,在此基础上叠加。

---

## 四、最小代码修改范围评估

> 仅评估,不写代码。复杂度/风险为初筛(低/中/高)。

| 模块 | 可能修改文件 | 可能新增结构 / 函数 | 修改目的 | 复杂度 | 风险 | 是否必须 |
|---|---|---|---|---|---|---|
| cell 元数据扩展 | `src/llama-kv-cells.h` | `resident`/`swapped`/`dirty`/`swap_offset`/`last_access` 并行数组 | 表达换出态 | 中 | 中(须与 used/seq 协同) | **是** |
| swap 后备存储管理 | 新增独立模块(建议) | 后备存储句柄 + `swap_offset` 映射表 | 持有换出字节 | 中 | 中 | **是** |
| swap-out 数据搬运 | `src/llama-kv-cache.cpp` | `swap_out(cell_range)`(参考 `state_write_data` 按物理 idx) | 读出并保存 K/V 字节 | 中 | 中(V 转置已被边界排除) | **是** |
| swap-in 数据搬运 | `src/llama-kv-cache.cpp` | `swap_in(cell_range)`(复用 `state_read_data` scatter,绕开 meta) | 原位写回字节 | 中 | 中 | **是** |
| find_slot 与 swapped 协同 | `src/llama-kv-cache.cpp:962` | `can_use` 判定纳入 `swapped` | 防止换出 cell 被覆盖丢数据 | 中 | **高**(正确性关键) | **是** |
| decode 读前 resident 检查 | `src/llama-context.cpp:1712`/`update` `:742` | 读前 `ensure_resident([0,n_kv) 内需访问 cell)` | 保证读区间字节就位 | 中 | 中 | **是** |
| 统计指标 | 上述模块 + 日志 | 计数器/计时器 | 采集 RSS/swap bytes/latency | 低 | 低 | **是** |
| 配置开关 | cparams / 启动参数 | `enable_kv_swap` + 策略参数 | 可关闭、对照 baseline | 低 | 低 | **是** |

> 评估口径:V 转置搬运、SWA、K-shift、seq_cp 相关改动**因边界条件而排除在最小 demo 之外**,故不列入必须项。具体新增函数签名与文件落点**仍需验证**(实现阶段确定)。

---

## 五、执行流程草图

> 一次 decode step 中可能发生的流程(同步 swap,demo 边界内)。

1. **写入当前 token KV**:`apply_ubatch`(`:1017`)写 cell 元数据,`cpy_k/cpy_v`(`:1197`)散列写入当前 token 的 K/V;新写入 cell 标记 `resident=true`、更新 `last_access`。
2. **选择冷 cell range**:按固定窗口策略,挑出 `last_access` 最旧且预期不在近期 `[0,n_kv)` 读区间内的 cell range 作为换出候选。
3. **换出冷 cell range**:逐层读出该 range 的 K/V 字节(`ggml_backend_tensor_get`),写入后备存储,记录 `swap_offset`。
4. **标记状态**:换出 cell 置 `swapped=true`、`resident=false`;其 pos/seq/shift **保持不变**。
5. **下一轮 decode 前计算读区间**:由 `get_n_kv`(`:1129`)得 `[0,n_kv)`,确定本轮实际需访问的 cell 集合。
6. **发现 swapped cell → 同步 swap-in**:若读区间内有 `swapped` cell,先同步换入(阻塞至完成)。
7. **scatter 原位写回**:复用 `state_read_data` 思想(`:2241-2247`),构造指向原物理 idx 的 `sinfo`,把字节写回原槽位;置 `resident=true`、`swapped=false`。
8. **进入连续 view**:`get_k/get_v`(`:1145`)返回 `[0,n_kv)` 连续 view——此时区间内需访问 cell 均已 resident。
9. **attention 计算**:内核(`ggml_flash_attn_ext`)正常执行,对 swap 完全透明。
10. **记录统计指标**:RSS、swapped bytes、swap-in/out 次数与延迟、有效占用率等(见第七节)。

> 说明:步骤 2~4(换出)与步骤 5~7(换入)是否在同一 step 内成对发生,取决于策略;demo 可先做"达到水位才换出、读前才换入"的惰性同步路径。**换出真正释放物理内存的时机与 ggml buffer 管理的关系仍需验证。**

---

## 六、正确性约束

必须保证(否则 demo 无效):

1. **字节一致**:原位换入后,cell 的 K/V 字节与未换出的 baseline 逐字节一致(类型/行宽校验沿用 `state_read_data` `:2222`/`:2231`/`:2274`)。
2. **元数据不破坏**:换出/换入全程,cell 的 `pos`/`seq`/`shift` 不被修改(纯增量换入的核心前提)。
3. **不覆盖 swapped cell**:`find_slot` 的 `can_use`(`:962`)必须把 `swapped` 视为不可随意覆盖,否则换出数据在换回前被新 token 覆盖丢失(**高风险项,必须处理**)。
4. **读区间 resident**:`[0,n_kv)` 内会被内核读到的 cell 必须 resident(因 mask 不避免物理读取)。
5. **配置一致**:`type_k`/`type_v` 与 `v_trans` 在换出/换入两侧一致(demo 固定 flash_attn=true → V 非转置)。
6. **输出一致**:开启 swap 后输出 token 与 baseline 尽量一致(理想为逐 token 相同,数值误差范围内)。
7. **可关闭等价**:关闭 `enable_kv_swap` 后行为与原版**完全一致**(零侵入回退)。

---

## 七、实验指标

demo 阶段采集:

| 指标 | 含义 / 用途 |
|---|---|
| RSS / peak memory | 进程峰值物理内存(赛题主指标) |
| KV resident bytes | 当前驻留 KV 字节数 |
| swapped bytes | 已换出字节总量 |
| swap-out 次数 | 换出操作计数 |
| swap-in 次数 | 换入操作计数 |
| swap-in latency | 单次/累计换入延迟(同步路径直接计入 decode) |
| decode latency | 每 token 解码延迟(对比 baseline 的回退) |
| tokens/s | 吞吐(对比 baseline) |
| 输出一致性 | 与 baseline 的逐 token 一致率 |
| `[0,n_kv)` 有效占用率 | used cell 数 / `[0,n_kv)` 长度(决定换入放大) |
| 换入放大比例 | 实际换入字节 / 理论必需字节 |

> 主结论应能展示:**开启 swap 后 peak memory 下降,且输出一致**;同步路径下 decode latency 可能上升(预取阶段再优化)。

---

## 八、风险与规避

| 风险 | 来源 | 风险等级 | demo 阶段规避方式 | 后续扩展方向 |
|---|---|---|---|---|
| find_slot 覆盖 swapped cell | `can_use` 判定(`:962`)不识别换出态 | **高** | `can_use` 纳入 `swapped`,换出 cell 不可被覆盖(必须实现) | 引入 swapped 优先级/回收策略 |
| K-shift 破坏换出字节一致性 | `update` 中 K-shift(`:742`/`:1721`) | **高** | demo 不支持 K-shift / context shift | 换入后重放 shift 或保序 |
| SWA/iSWA 与换出语义重叠 | `is_masked_swa`(`:977`)、双实例 | 中~高 | demo 用标准 attention(无 SWA) | 协调窗外回收与 swap 驱逐 |
| seq_cp 共享 cell 一致性 | cell seq 位图共享(`:406`) | 中~高 | demo 单 sequence、不支持 seq_cp | 块级 refcount / COW |
| V 转置路径搬运碎片化 | `v_trans=!flash_attn`(`:94`)、转置慢路径(`:2331-2348`) | 中 | demo 固定 flash_attn=true(V 非转置) | 支持转置或换出前转布局 |
| 连续 view 换入放大 | `n_kv=used_max_p1 pad≥256`(`:1134`)+ mask 不避免读取 | 中 | 固定窗口 + 单 seq 顺序填充,降低稀疏度 | 统计有效占用率指导策略 |
| 后端异步能力不确定 | `ggml_backend_tensor_get/set` 异步支持未确认 | 中 | demo 用同步 swap | 验证后端异步后接 prefetch |
| 性能回退(decode 变慢) | 同步换入阻塞计算 | 中 | demo 接受回退,只验证正确性+内存 | 异步预取掩盖 I/O |

> 全部高/中风险项在 demo 边界内均有规避路径;唯一**必须在 demo 内实现**的是"find_slot 不覆盖 swapped cell"(否则正确性破坏)。

---

## 九、与 PagedAttention 的关系

- **本 demo 不是完整 PagedAttention**:不实现 physical block pool、block table、non-contiguous paged read,attention kernel 不变。
- **借鉴的思想**:逻辑位置与物理驻留分离、冷数据换出、按需换入——即 PagedAttention 的"KV 分页 / 换页 / 逻辑到物理管理"内核理念,但落在 **cell 粒度 + 原位换入**而非 block 粒度 + 间接寻址。
- **与 vLLM 的差异**:vLLM 用固定 block + block table 把序列映射到**物理非连续** block 池,kernel 按表 gather;本 demo 仍是**单块连续张量 + 原位换回**,读路径连续不变。
- **为何先做 runtime swap + 原位换入而非 non-contiguous paged read**:后者须改各后端 attention kernel(跨后端、高风险,见 [kv_cache_gap_matrix.md](kv_cache_gap_matrix.md) 缺口 7,C 类);前者**不改 kernel**、可复用现有 state 搬运路径,风险可控、可演示,契合赛题"系统完成度"。
- **后续如何向 block/page 靠近**:① cell range 升级为固定大小 block 作为换出/换入单位;② `swap_offset` 映射升级为 block table 雏形;③ 在确有收益且 kernel 改造可行时,再评估 non-contiguous paged read——**均为后续方向,非本 demo 范围**。

---

## 十、是否进入实现阶段的判断标准

进入实现前需满足的条件(逐项核对):

| 条件 | 当前状态 |
|---|---|
| 设计边界清楚 | ✅ 第二节边界已收敛 |
| 最小修改范围明确 | ✅ 第四节评估完成(具体函数落点仍需验证) |
| 正确性检查可执行 | ✅ 第六节 7 条约束 + 可关闭等价 |
| 实验指标明确 | ✅ 第七节指标清单 |
| 风险可控 | ✅ 第八节高/中风险均有规避,唯 find_slot 协同必须实现 |
| 不需立即改 attention kernel | ✅ 已确认 |

**阶段性判断:可以进入最小 demo 原型设计。**

依据:设计边界、修改范围、正确性约束、实验指标、风险规避五项齐备,且不需改 attention kernel。**进入原型后应优先用小实验闭合的"仍需验证"项**:① 换出能否真正释放物理内存(与 ggml buffer 管理关系);② 原位换入输出逐 token 一致性;③ `[0,n_kv)` 有效占用率与换入放大实测;④ 后端是否支持高效异步(决定 prefetch 阶段可行性)。这些属"实现中验证",不构成进入原型的阻塞项。

> 本判断为"可进入最小 demo 原型设计",**仍不含代码、不修改源码**;原型设计阶段再细化函数签名与数据结构布局。

---

## 十一、更新记录

- **2026/05/31 初稿**:基于现状分析、缺口矩阵、可行性验证、第二轮风险验证四份文档生成最小 demo 设计。
- 未写代码;未修改源码;未进入实现。
- 阶段性判断:**可进入最小 demo 原型设计**(同步 swap 优先,异步 prefetch 第二阶段)。
- 关联文档:[llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)、[kv_cache_gap_matrix.md](kv_cache_gap_matrix.md)、[kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md)、[kv_runtime_swap_risk_validation.md](kv_runtime_swap_risk_validation.md)。

