# KV runtime swap / offloading 第二轮风险验证

> 操作系统功能赛技术报告素材 · A 类方向第二轮风险验证篇
>
> **定位**:基于 [docs/llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)、[docs/kv_cache_gap_matrix.md](kv_cache_gap_matrix.md)、[docs/kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md),对第一轮标注的关键不确定性做第二轮源码级风险验证。
>
> 验证日期:2026/05/31

---

## 一、验证目标

- 本轮**只针对 A 类方向**中的 runtime KV swap / offloading + prefetch,不扩展到其他方向。
- **只做源码分析与风险验证,不写代码、不修改源码、不确定最终方案、不展开实现计划。**
- 目标:验证第一轮遗留的关键不确定性,判断该方向是否**仍然"值得继续验证"**,而非进入实现。
- 所有结论给源码文件 / 函数,不确定处标注"仍需验证"。

> 本轮新读源码:`find_slot` 完整体(`src/llama-kv-cache.cpp:818-1015`)、`apply_ubatch` 头部(`:1017-1046`)、`v_trans` 来源(`:84`/`:94`/`:155`/`:187`)、decode 循环与 memory context 交互(`src/llama-context.cpp:1249`/`:1712-1942`/`:3214-3277`)。

---

## 二、纯增量原位换入可行性

> 源码:`state_read_data` `src/llama-kv-cache.cpp:2187-2353`、`state_read_meta` `:2068-2185`、`read_tensor` `src/llama-context.cpp:2466`/`2531`、scatter 写回 `:2241-2247`。

**背景**:第一轮确认 `state_read_data` 具非连续 scatter 原位写回;但 `state_read_meta` 含 `seq_rm`(`:2074`)、`find_slot`(`:2113`)、`apply_ubatch`(`:2121`)、全量分支 `clear(true)`(`:2141`)等"重新分配"语义。

1. **能否绕开 `state_read_meta` 只复用 `state_read_data` 的数据路径?**
   可行性较高。`state_read_data`(`:2187`)的输入只有 `(io, strm, cell_count, sinfo)`,其内部**完全不调用 meta**,仅:读类型/行宽校验头(`:2219-2235`)→ 按 `sinfo.is_contiguous()` 选 memcpy 快路径或 `idxs` scatter 慢路径(`:2237-2248`)。只要**外部提供一个已知 `sinfo`(目标物理 cell 索引)**,即可直接把换出的字节写回原位,无需 meta 的分配逻辑。**结论:数据路径可独立复用,前提是换出时保留好物理槽位映射。**

2. **`state_read_data` 的依赖项**:
   - `sinfo.idxs[0]`(scatter 目标物理位置,`:2244`)或 `sinfo.is_contiguous()/head()`(快路径,`:2240`);
   - `cell_count`(循环次数);
   - `layers` 遍历(全层,`:2212`)与每层 `n_embd_k_gqa`/`n_embd_v_gqa`;
   - K/V 行宽校验 `k_size_row`/`v_size_row`(`:2231`/`:2274`)、类型校验 `k->type`(`:2222`)、V 转置时 `v_size_el`+`n_embd_v_gqa`(`:2317`/`:2325`);
   - `this->v_trans`(决定走哪条 V 分支,`:2251`)。

3. **纯增量原位换入须提前保留的 metadata**:cell index(物理槽位)、stream id、layer id(或确认全层一致)、K/V row offset(可由 `n_embd_*_gqa` + type 推出)、swap_offset(后备存储偏移)、type_k/type_v、v_trans。**这些都是"换出时已知、换入时复现"的量,无需重走分配。**

4. **不能重走 `state_read_meta` 的 metadata**:cell 的 `pos`、`seq` 归属、`shift` 当前状态——因为 meta 会 `seq_rm`/`clear`/`find_slot` **重新决定 cell 落点**,破坏"原位"前提。纯增量换入必须**保持 cell 的 pos/seq/shift 元数据不动**,只搬运 K/V 字节。

5. **最小改造边界**:换出 = 读出指定 cell 的 K/V 字节(类比 `state_write_data` 的 `write_tensor`,但按物理 idx 而非 seq 过滤)+ 记录 swap_offset;换入 = 构造指向原物理 idx 的 `sinfo` + 调用 `state_read_data` 等价的 scatter 写回。**边界落在"数据搬运 + 物理槽位映射",不触碰 meta/find_slot/mask。**

6. **复用判定**:**可参考但需拆分**。`state_read_data` 的"按 sinfo scatter 写字节"逻辑可直接参考甚至复用,但需从 `state_read_meta` 的分配编排中剥离,并新增"换出时保存物理映射"的对偶路径。

---

## 三、V 转置路径风险

> 源码:`v_trans` 构造参数 `src/llama-kv-cache.cpp:84`/赋值 `:94`;布局相关 `:155`/`:187`;非转置写 `:2003-2028`/读 `:2251-2292`;转置写 `:2029-2065`/读 `:2293-2350`。

1. **`v_trans` 如何确定?** 它是构造函数入参(`:84`),在 `create_memory` 传入 `!cparams.flash_attn`(见 analysis 第三节)。即 **flash_attn 开 → `v_trans=false`(V 非转置);flash_attn 关 → `v_trans=true`(V 转置)**。存于成员(`:94`)。

2. **两种布局**:
   - 非转置(`v_trans=false`):V 每行 = 一个 cell 的 `n_embd_v_gqa` 连续元素;每 cell 一段连续 `v_size_row`(`:2019`)。
   - 转置(`v_trans=true`):V 按 `[kv_size, n_embd_v_gqa]` 存储,**同一 cell 的各 embedding 维分散在不同行**;构造期按 `n_embd_v_gqa_max()` 预留(`:187`)。

3. **搬运逻辑差异**:
   - 非转置读(`:2280-2291`):每 cell 一次 `read_tensor`(连续 `v_size_row`),或快路径整段 memcpy。
   - 转置读(`:2331-2348`):**双重循环 `for j in n_embd_v_gqa { for i in cell_count }`**,每次仅搬运 `v_size_el`(单元素)。`dst_offset = (idx + j*kv_size)*v_size_el`(`:2343`)。

4. **是否导致大量小粒度 I/O?** **是**。转置慢路径下,搬运次数 = `n_embd_v_gqa × cell_count`,每次仅一个元素宽(`v_size_el`)。即便快路径(连续 cell),也是 `n_embd_v_gqa` 次、每次 `cell_count*v_size_el`,仍按 embedding 维切成 `n_embd_v_gqa` 段。**相比非转置(每 cell 一段),转置搬运粒度显著更碎。**

5. **最小 demo 是否应限定 `flash_attn=true` / V 非转置?** **建议是**。flash_attn=true 时 V 非转置,换出/换入按 cell 连续搬运,I/O 粒度大、寻址简单,最贴合 swap。这也是现代部署常用配置。

6. **若无法限定 flash_attn=true 的工程风险**:转置路径换出/换入退化为大量小粒度 `read_tensor/write_tensor`(经 `ggml_backend_tensor_get/set`),每次调用都有固定开销,**I/O 放大可能吞噬 swap 的内存收益**;且转置下"单 cell"在物理上不连续,swap_offset 映射更复杂。

**风险等级:中**。理由:风险真实存在(转置碎片化明确),但**可通过限定 flash_attn=true / V 非转置规避**,而该配置是合理且常见的 demo 前提;故非不可控的"高"。转置路径的实际搬运耗时 **仍需验证**(实测)。

---

## 四、连续 view 导致的换入放大

> 源码:`get_n_kv` `src/llama-kv-cache.cpp:1129-1143`、`get_k/get_v` `:1145-1195`、`find_slot` head 回绕 `:925-927`/`:941-944`、SWA 复用判定 `:977`、`apply_ubatch` cell 覆盖 `:1035-1046`。

1. **`n_kv` 怎么来?** `get_n_kv`(`:1129`):`result = min(cells.size(), max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur)))`,其中 `n_pad_cur = max(n_pad, 256)`(`:1134`)。即 **n_kv = 把"最高占用 cell+1"向上 pad 到 ≥256 的对齐值**。

2. **`[0,n_kv)` 是否含大量无效 cell?** **可能**。`used_max_p1()` 是高水位线而非有效计数。当出现:head 回绕(`:925-927` 把 head 重置到 0 重新填充低位)、seq 结束被 `rm`、SWA 窗外 cell 被标记可复用(`:977`)——这些 cell 物理上落在 `[0, used_max_p1)` 内但已无效,**`[0,n_kv)` 会包含它们**。稀疏度取决于访问模式。

3. **无效 cell 被 swap-out 后是否必须换回?** 若它们落在 `[0,n_kv)` 内:attention 内核按连续步长读整个区间(mask 仅置 -INFINITY 不跳过物理读),**物理读取仍会触及这些槽位**。若已 swap-out 且物理内存被复用/置空,会读到脏数据。**因此要么换回,要么保证其物理槽位仍 resident 且内容无害**。这是连续 view 的核心代价。

4. **最小 demo 是否应限定场景?** 是,建议(降低稀疏度与无效区间占比):单 sequence、无复杂 seq_cp、固定 n_ctx / 较小 n_kv、关闭/避开 SWA(避免窗外无效 cell)、尽量顺序填充避免回绕碎片。

5. **不改 kernel 能否减少换入放大?** 部分手段:
   - 保证换出只针对 `[0,n_kv)` **之外**或确定不再进入读区间的 cell(如已结束的 seq、SWA 永久窗外);
   - 控制 n_kv 上界(小 n_ctx);
   - 顺序填充以让有效 cell 集中在低位,使高水位贴近有效计数。
   均为"避开放大",非消除——根因(连续读)不改 kernel 无法去除。**是否可只换入当前 ubatch 实际 attend 的 stream 子集,仍需验证。**

6. **是否需统计 used vs `[0,n_kv)` 比例?** **需要**。该比例(有效占用率)直接决定换入放大倍数,是判断 swap 收益的关键实验指标。

### 换入放大风险分析表

| 场景 | `[0,n_kv)` 含无效 cell 倾向 | 换入放大风险 | 缓解(不改 kernel) | 仍需验证 |
|---|---|---|---|---|
| 单 seq、顺序填充、无回绕 | 低 | 低 | 有效 cell 集中低位 | 实测有效占用率 |
| head 回绕后(`:925-927`) | 中 | 中 | 重置后重新致密 | 回绕频率 |
| seq 结束 `rm` 留空洞 | 中~高 | 中 | 及时压缩 / 限定单 seq | 空洞分布 |
| SWA 窗外大量失效(`:977`) | 高 | 高 | demo 避开 SWA | iSWA 下 n_kv 行为 |
| 多 seq + seq_cp 碎片 | 高 | 高 | demo 暂不支持 | 碎片度量 |

---

## 五、与现有机制的不变量协同

> 每个机制回答:依赖的不变量 / swap-out 后是否成立 / swap-in 前须恢复什么 / 是否需新增 resident 等状态 / demo 是否先禁用 / 风险等级。

| 机制 | 依赖的 KV 不变量 | swap-out 后是否成立 | swap-in 前须恢复 | 需新增状态判断 | demo 先禁用? | 风险 |
|---|---|---|---|---|---|---|
| K-shift / RoPE shift(`build_rope_shift`,`update` `:742`) | cell 的 `shift` 元数据与物理 K 字节对应;shift 作用于驻留张量 | **可能不成立**:若 cell 已换出,K-shift 无法对其物理字节施加旋转;换入后 shift 状态须与字节一致 | K 字节 + `shift` 元数据;或换入后补做 shift | 需 `resident`(shift 时跳过非驻留或先换回) | **是**(demo 避开 context shift 场景) | 高 |
| SWA / iSWA(`is_masked_swa` `:977`,`kv_base`/`kv_swa`) | 窗外 cell 可被 find_slot 复用;双实例独立 | 部分:SWA 窗外回收与 swap 驱逐语义重叠,易冲突 | 若换入窗内 cell:K/V 字节 + pos | 需协调 `swapped` 与"窗外可复用"优先级 | **是**(demo 用标准 attention,Llama3 默认不走 iSWA) | 中~高 |
| seq_cp / prefix reuse(`:406`,cell seq 位图) | 同 stream 多 seq 共享同一物理 cell(位图打标) | **风险**:共享 cell 换出后,多个 seq 的引用同时失效;换入需保证所有引用方可见 | 共享 cell 的 K/V 字节 + seq 位图 | 需 refcount 或"共享 cell 不换出"规则 | **是**(demo 暂不支持 seq_cp) | 中~高 |
| find_slot / ring buffer(`:818`,`can_use` `:962`) | 空/可复用 cell 可被覆盖写入 | **冲突**:换出 cell 若被判 `can_use` 会被新 token 覆盖,丢失换出数据 | —(预防而非恢复) | 需 `swapped` 态纳入 `can_use` 判定(换出 cell 不可被随意覆盖,或覆盖前丢弃其 swap 副本) | 否(但须改判定逻辑,仍需验证) | 高 |
| KV quantization(type_k/type_v) | 字节按量化格式布局;行宽 = `ggml_row_size(type,...)` | 成立(swap 搬运的是原始字节,与量化格式无关) | 按相同 type 的行宽搬运(校验已存在 `:2231`/`:2274`) | 否 | 否(可保留) | 低 |
| backend offload / KQV offload(`offload=cparams.offload_kqv`) | KV 张量驻留某 backend buffer | 取决于 swap 目标与 backend buffer 的关系 | 字节搬运回原 backend buffer(`ggml_backend_tensor_set` 后端无关) | 视多 backend 而定 | demo 建议先固定单 backend | 中 |

**小结**:与 swap 冲突最强的是 **find_slot 覆盖语义**(高,必须处理,否则换出数据被覆盖)、**K-shift**(高,demo 应避开)、**seq_cp / SWA**(中~高,demo 先禁用)。量化(低)与 swap 正交,可保留。多数冲突可由"最小 demo 边界条件"(第七节)规避。

---

## 六、prefetch 的真实可插桩点

> 源码:decode 主循环 `src/llama-context.cpp:1712-1942`(`mctx->apply()` `:762`/`:1250`、`mctx->next()` `:1942`)、`process_ubatch` `:1249`、graph 取 K/V `src/llama-graph.cpp:2240-2241`、`update` `src/llama-kv-cache.cpp:742`、`get_k/get_v` `:1145`、backend get/set `src/llama-context.cpp:2506-2535`。

| 候选插桩点 | 能否预知下一步 KV 范围 | 能否与计算重叠 | 易拿到 layer/stream/cell/n_kv? | 是否破坏 graph 构建/执行序 | 适合 demo? | 优点 / 缺点 / 风险 |
|---|---|---|---|---|---|---|
| 1. decode 主循环(`:1712`-`:1942`) | **能**:循环外可知整批要 attend 的 seq/范围 | **能**:在进入 graph 前发起异步换入 | n_kv 需经 mctx;stream/seq 已知 | 否(在 graph 之外) | **适合** | 优:粒度粗、与 graph 解耦、最易实现;缺:预知精度到 ubatch 级;风险低 |
| 2. update 阶段(`:742`) | 部分:update 处理 shift/跨 stream,非主读路径 | 部分 | 可拿到 cells/stream | 否 | 中 | 优:已是状态变更入口;缺:不是每步必经的读前点;风险中 |
| 3. build_attn 前(`graph.cpp:2240` 之前) | **能**:逐层、马上要读该层 K/V | 弱:已在 graph 内,重叠窗口小 | layer/n_kv 直接可得 | **可能**:在图内插同步换入会打断执行序 | 否 | 优:粒度精确到层;缺:太晚,重叠空间小,易破坏 graph;风险高 |
| 4. get_k/get_v 内部(`:1145`) | 能(该层) | 几乎不能(返回 view 即被消费) | 直接 | **可能**:这里只构 view,换入须在执行期 | 否 | 优:集中;缺:时机太晚=同步阻塞;风险高 |
| 5. backend get/set 附近(`:2506`) | 否(纯搬运原语层) | N/A | 否 | 否 | 否(作搬运实现,不作触发) | 仅适合作为"搬运实现",非"预取决策点" |

**阶段性判断**:prefetch 最值得继续验证的插桩点是 **(1) decode 主循环**(主预取触发,可与 graph 执行重叠、解耦、易拿信息)与 **(2) update 阶段**(作为状态变更钩子辅助)。**build_attn / get_k 内部(3/4)太晚,重叠窗口小且易破坏 graph,不适合**。不在此处给实现方案。逐层预取与 ggml graph 实际执行重叠窗口 **仍需验证**;后端异步拷贝 API 是否存在 **仍需验证**。

---

## 七、最小 demo 的边界条件建议

> 只给"边界条件",不设计实现。"必要性"指对正确性/可行性的影响。

| 限制 | 必要性 | 收益 | 代价 |
|---|---|---|---|
| 只支持 CPU / 指定单 backend | 高 | 规避多 backend 异步能力差异(仍需验证);搬运经 `ggml_backend_tensor_get/set` 单后端行为确定 | 不能立刻展示跨设备 offload |
| 只支持 flash_attn=true / V 非转置 | 高 | 避开 V 转置碎片化(第三节),换入换出按 cell 连续、寻址简单 | 不覆盖 flash_attn=false 配置 |
| 只支持单 sequence | 高 | 避开 seq_cp 共享 cell 的引用一致性(第五节) | 不展示多请求并发 |
| 暂不支持 seq_cp / prefix sharing | 高 | 消除共享 cell 换出的 refcount 难题 | 无前缀复用场景 |
| 暂不支持 SWA / iSWA | 中~高 | 避开窗外回收与 swap 驱逐语义重叠;Llama3 默认即非 iSWA | 不覆盖 SWA 模型 |
| 暂不支持 K-shift | 高 | 避开 shift 与换出字节一致性(第五节) | 不支持 context shift / 长上下文滚动 |
| 只做固定 cell range / 固定窗口的 swap | 中 | 降低 `[0,n_kv)` 换入放大与映射复杂度;便于度量 | 非自适应,内存收益受限 |
| 只做同步 swap,异步 prefetch 后置 | 高 | 先验证正确性与内存下降,再验证延迟掩盖 | 初版无 I/O 掩盖,延迟可能升高 |
| 只统计 RSS / swap bytes / latency | 中 | 聚焦赛题核心指标,降低工程量 | 非完整性能最优 |

**边界小结**:上述限制共同把首个 demo 收敛到"**单 seq + flash_attn=true(V 非转置)+ 标准 attention(无 SWA/无 K-shift)+ 单 backend + 同步 swap + 固定窗口**"这一最可控子集——恰好规避第三~五节的全部高/中风险项,且仍能演示"峰值物理内存下降"这一赛题核心。异步预取作为第二阶段验证。

---

## 八、仍需实验验证的问题

以下须通过实验或小型原型(非本阶段)验证:

1. V 转置路径单 cell 换出/换入的实际搬运耗时与调用次数(对比非转置)。
2. `[0,n_kv)` 连续 view 的换入放大比例(= `[0,n_kv)` 长度 / 有效 used cell 数),分场景实测。
3. `ggml_backend_tensor_get/set` 在目标 backend 上是否支持高效异步(决定 prefetch 重叠可行性)。
4. 原位换入后模型输出是否与 baseline 逐 token 一致(正确性回归)。
5. swap 对 tokens/s、TTFT、decode latency 的影响幅度。
6. 预取能否真正掩盖换入 I/O(重叠率)。
7. find_slot 覆盖语义与 `swapped` 态协同的正确性(防止换出 cell 被覆盖)。
8. 不改 kernel 时换入是否可只覆盖当前 ubatch 实际 attend 的 stream 子集。

---

## 九、对已有文档的回填建议

### 回填 [docs/llama_kv_cache_analysis.md](llama_kv_cache_analysis.md)
- 第四节机制 8:补注"纯增量原位换入须保持 cell 的 pos/seq/shift 元数据不动、只搬 K/V 字节,且须绕开 `state_read_meta` 的 `find_slot`/`clear` 分配"。
- 附录(d):可细化为"V 转置碎片化=中风险(可由 flash_attn=true 规避)""连续 view 换入放大与有效占用率相关"。

### 回填 [docs/kv_cache_gap_matrix.md](kv_cache_gap_matrix.md)
- 缺口 4/12:风险列补"find_slot 覆盖语义与 swapped 态冲突(高,须处理)"。
- 第五节 A 类:维持 A;补注"首个 demo 边界已收敛(单 seq + flash_attn=true + 无 SWA/K-shift + 单 backend + 同步 swap)"。
- 第六节任务 1b:标记"已完成第二轮验证",结论回填本文件。

### 回填 [docs/kv_runtime_swap_feasibility.md](kv_runtime_swap_feasibility.md)
- 第五节风险:
  - "纯增量原位换入"→ 由"仍需验证"降级为"**路径已澄清**(绕开 meta、复用 scatter 数据路径、保留物理映射),实现细节仍需验证";
  - "V 转置代价"→ 标注**中风险、可由 flash_attn=true 规避**;
  - 新增"find_slot 覆盖语义冲突(高)""K-shift 与换出字节一致性(高)"两项风险。
- 第七节结论:与本文件一致,维持"值得继续验证"。

### 可消除 / 须保留
- **可消除的"仍需验证"**:纯增量换入是否可绕开 meta(已确认数据路径可独立复用)、`state_read_data` 结束行号(已确认 :2353)。
- **须继续保留**:V 转置实测代价、连续 view 换入放大比例、后端异步能力、输出一致性、find_slot 协同正确性(均为实验类,见第八节)。
- **优先级**:A 类**维持不变**,且边界条件已收敛、主要风险均有规避路径,信心增强。

---

## 十、阶段性判断

**值得继续验证。**

理由:① 纯增量原位换入的**数据路径已澄清**——可复用 `state_read_data` 的 scatter 写回(`src/llama-kv-cache.cpp:2241-2247`),绕开 `state_read_meta` 的分配语义,只需换出时保留物理槽位映射;② 主要风险(V 转置碎片化、连续 view 换入放大、find_slot 覆盖、K-shift/SWA/seq_cp 协同)**均已定位到具体源码,且都有明确的 demo 规避路径**(第七节边界条件);③ prefetch 已识别出 decode 主循环这一**解耦、可重叠、不破坏 graph** 的插桩点;④ 全程**不需改 attention kernel**。

保留的实验类不确定性(第八节)属于"实现前需用原型量化"的范畴,不构成"证据不足"或"暂不建议"——它们不改变方向可行性的定性结论。

> 本判断为"值得继续验证",**非**确定采用,**不含**实现方案与代码。

