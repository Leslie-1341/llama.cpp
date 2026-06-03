# KV runtime swap 阶段 D 汇总与阶段 E 前评估

> 操作系统功能赛技术报告素材 · 阶段 D 收口篇
>
> **定位**:汇总 [阶段 C](kv_runtime_swap_stage_c.md)(固定窗口同步 swap-out)、[阶段 D](kv_runtime_swap_stage_d.md)(D1+D2 swap-in / ensure_resident、D3 debug poison)的实现与验证结论,评估进入阶段 E 前的方向选择。
>
> **约束**:本文档【只写文档、不改源码、不改已有文档、不跑实验、不进入阶段 E】。所有数据引自已提交的阶段 C/D 记录文档与其中实测结果;新增推断处标注「仍需验证」。
>
> 分支:`kv-runtime-swap-stage-d`(最近提交 `dce31e3 add kv swap poison validation mode`)
> 文档日期:2026/06/02

---

## 1. 当前实现进度

| 阶段 | 内容 | 状态 | 提交 |
|---|---|---|---|
| A+B | CLI 开关(`--kv-swap` / `--kv-swap-window`)+ cell metadata(`swapped` / `swap_offset` / `last_access`)| ✅ 已完成 | `f83c7d4` |
| C | 固定窗口同步 swap-out(冷 cell 字节复制到 `kv_swap_storage` + 打标记,**不破坏原字节**)| ✅ 已完成 | `28aa28b` |
| D1+D2 | swap-in(`swap_in_cell` 原位恢复)+ `ensure_resident([0,n_kv))` 读前换回 | ✅ 已完成 | `1ea60c2` |
| D3 | debug poison(换出后用 `0xCC` 破坏原 KV buffer,反证 swap-in 必需)| ✅ 已完成 | `dce31e3` |

**尚未实现**(明确边界外):

- ❌ prefetch / 异步换入 / I/O latency hiding;
- ❌ 真实 RSS 下降(原连续 buffer 未释放,见 §4);
- ❌ 多 sequence / SWA / iSWA / K-shift / seq_cp;
- ❌ 非连续 paged read / block table / attention kernel 改造。

至此 demo 已闭合「换出 → 标记 → 读前换回 → 正确性反证」整条数据通路,处于**正确性闭环完成、内存收益未兑现**的节点。

---

## 2. 阶段 D 的核心验证结论

整合 [阶段 C](kv_runtime_swap_stage_c.md) §8、[阶段 D](kv_runtime_swap_stage_d.md) §8 / §10.2 的实测:

1. **swap-out 数据通路正确**:窗口外冷 cell 的 K/V 字节经 `ggml_backend_tensor_get` 逐层(先 K 后 V)append 进 `kv_swap_storage`,起始偏移记入 `cells.swap_offset(i)`,并置 `swapped=true`。
2. **swap-in 原位恢复**:`swap_in_cell(i)` 按 `swap_offset(i)` 顺序读回,用 `ggml_backend_tensor_set` 写回**同一物理行** `i * size_row`,只清 `swapped` 标记,不动 `pos` / `seq` / `shift`。
3. **读前 resident 保证**:`ensure_resident(n_kv)` 在 `llama_kv_cache_context::apply()` 末尾(graph 构建前)遍历 `[0, n_kv)`,把其中 `swapped` 的 cell 全部换回。
4. **round-trip 无损(D1+D2)**:`restored_bytes == swapped_bytes`(`-n 64` 实测 255983616 字节双向相等),开关 off/on 生成文本逐 token IDENTICAL。
5. **swap-in 必需性反证(D3)**:在 poison 模式把原字节真实写成 `0xCC` 后,输出仍与无 poison **IDENTICAL**;`poison_bytes == swapped_bytes == restored_bytes`(`-n 48` 实测 141688832 字节三者相等,poison_cells=1081)。

**关键差异**:D1+D2 因不破坏原字节,off/on 一致只证「数据通路无损」,**不能反证 swap-in 必需**;D3 用 poison 补上这一反证——毒化的 1081 个 cell 与换回的 1081 个 cell 字节完全吻合,说明 `ensure_resident` + `swap_in_cell` 确实在读路径前承担了「还原正确 K/V」的职责,**不是摆设**。

---

## 3. 当前实现的系统现象

> 重点分析阶段 D 暴露的结构性现象,引自 [阶段 D](kv_runtime_swap_stage_d.md) §5 / §8。

1. **`swap_in_count` 远大于窗口大小**:`--kv-swap-window 8` 但 `-n 64` 实测 `swap_in_count=1953`、`-n 48` 实测 1081——换入次数与窗口无关,被读区间长度主导。
2. **`get_n_kv` 至少 pad 到 256**:公式 `n_kv = min(size, max(256, PAD(used_max_p1, 256)))`,即便实际有效 token 远少于 256,读区间 `[0, n_kv)` 仍被向上撑到 ≥256。
3. **连续 view + mask 强制换回**:mask 仅置 -INFINITY 屏蔽 softmax 数值,**不避免物理读取**;落在 `[0, n_kv)` 内会被内核连续读到的 swapped cell 必须先换回。于是窗口外刚换出的 cell,下一 step 立刻又落进读区间被 `ensure_resident` 换回。
4. **换入放大是当前最大性能问题**:`ensure_resident_calls=65` 步 × 每步约 30 个 swapped cell ≈ `cells_restored=1953`(D1+D2 `-n 64`),形成「换出→立即换回」的抖动,swap I/O 被反复触发却无净驻留下降。
5. **当前做法更像「正确性闭环 demo」而非最终内存优化方案**:它证明了 llama.cpp 能承载 runtime KV 换出/换入这一系统命题,但固定小窗口 + 全区间 resident 检查的组合,本身不是能兑现内存收益的策略。

---

## 4. 当前实现的局限

1. **RSS 不下降**:开启 swap 后峰值物理内存与 baseline 无实质差异。
2. **根因是 KV tensor 构造期一次性预分配**:每层 K/V 是构造函数里按 `n_ctx` 全量分配的连续 `ggml_tensor`(`ctxs_bufs` 持有),运行时不增不减。[baseline 结果](kv_baseline_results.md) §4 已实测 RSS 随 ctx 线性增长(+64MB / +128MB),正源于此预分配。
3. **swap-out 只是复制字节 + 打标记**:`ggml_backend_tensor_get` 把字节读进 `kv_swap_storage`,原张量区域**既不清零也不释放**,物理页仍常驻。
4. **poison 只是 debug 正确性验证**:`ggml_backend_tensor_set` 把原区域写成 `0xCC`,仍是**写回原张量**,不释放任何物理内存,反而多一次写——D3 明确不可用于性能 / RSS 评估。
5. **`kv_swap_storage` 额外占内存**:后备存储是进程内 `std::vector<uint8_t>`,换出字节在它里面**再存一份**;在原 buffer 未释放的前提下,总占用不降反升。
6. **因此当前阶段不能声称已减少峰值 RSS**:demo 的内存收益是「未决问题」,需 §5/§7 的专项实验才能判断能否兑现。

---

## 5. 阶段 E 的候选方向

> 两个候选方向对照,均不含实现,仅评估。

### 方向 E1:真实内存释放 / 减少 RSS

- **目标**:让换出后原 KV buffer 对应的常驻物理页真正减少,兑现赛题「减少运行时物理内存」。
- **可选思路**:`madvise(MADV_DONTNEED)` 对换出 cell 行所在页;mmap-backed buffer 替代匿名分配;分层后备存储(host RAM / 磁盘);分块(chunked)buffer 让 cell range 能整块释放。
- **优点**:直接对准赛题主指标(peak memory / RSS),与 [baseline 结果](kv_baseline_results.md) 的 RSS-vs-ctx 曲线可直接对比。
- **风险**:ggml tensor buffer 管理复杂(连续大张量、可能非 page 对齐),可能侵入 ggml backend buffer 层,改造面比 demo 边界深;cell 粒度太小,page 级释放效果可能差。
- **验证指标**:RSS / peak memory、KV resident bytes、swapped bytes、输出一致性。

### 方向 E2:prefetch / latency hiding

- **目标**:在已有 swap-in/out 数据通路上,用预取掩盖换入 I/O 延迟,贴赛题「预取掩盖 IO 延迟」。
- **可选思路**:提前一轮 / 提前一层预取下一 step 读区间的 swapped cell;同步搬运改异步(依赖后端异步拷贝能力,仍需验证);统计等待时间与命中率。
- **优点**:复用阶段 C/D 已闭合的搬运通路,不必触碰 buffer 释放难题。
- **风险**:当前**没有真实慢存储 I/O**(后备存储就在进程内 RAM),prefetch 收益可能不明显甚至难以观测;异步拷贝能力跨后端不确定(仍需验证)。
- **验证指标**:`swap_in_us`、decode t/s、换入等待时间、prefetch 命中率。

---

## 6. 阶段 E 推荐判断

1. **不建议马上做复杂 prefetch**:E2 的前提(慢 I/O、异步拷贝)在当前 demo 环境下都不成立,收益难以观测,容易投入产出失衡。
2. **优先做 E1-lite**:用最小方式验证「能否让 RSS 有可观变化」——这是赛题主指标,也是当前 demo 最大的未决问题(§4)。
3. **若 E1-lite 证明连续 buffer 难以释放 RSS**,再转向 E2 做 prefetch / latency hiding(届时可考虑引入真实慢后备存储以制造可观测 I/O)。
4. **阶段 E 应先做设计 + 小实验,不应直接大改架构**:与前几阶段一致,保持「可回滚、默认零侵入、先验证后扩展」的节奏。

---

## 7. 阶段 E 最小下一步建议

> 可执行路线,按风险递增排列,每步均可独立停下评估。

- **E0 — 补齐对比测试**:对当前 swap demo 在 [baseline](kv_baseline_results.md) 同矩阵(ctx 512/1024/2048、seed 42、固定 prompt、重复 3)下采集 RSS / decode t/s / swap 统计,产出与 baseline 的逐点对照(确认「关闭等价、开启正确、RSS 是否真未降」)。
- **E1-lite 研究**:调研 ggml buffer / tensor 对应的物理内存能否 `madvise`——确认 buffer 由哪种分配器持有(匿名 mmap / malloc)、是否 page 对齐、release 是否经 backend 接口。
- **E1-lite 小实验**:对 poison 后(已确知可安全覆盖)的 cell 行范围尝试 `madvise(MADV_DONTNEED)` 或等价机制,观察 RSS 是否下降;换入时再触发缺页回填。
- **若 RSS 无变化**:记录原因(分配器 / 对齐 / 粒度),作为「连续 buffer 难释放」的实证结论。
- **再决定**:是否进入分块 buffer(让 cell range 整块释放)或转向 E2 prefetch。

---

## 8. 风险清单

| 风险 | 说明 | 影响阶段 |
|---|---|---|
| `madvise` 对 malloc / ggml backend buffer 是否有效 | 取决于 buffer 分配方式;malloc arena 内的页未必能 `DONTNEED` 真正还给 OS | E1-lite,仍需验证 |
| KV 行不一定 page 对齐 | `madvise` 以页为粒度,行跨页或不对齐会导致释放不干净或误伤相邻行 | E1-lite,仍需验证 |
| cell 粒度太小 | 单 cell 行字节远小于一页,page 级释放效果差,需聚合到 block / range | E1 |
| `[0,n_kv)` 连续读导致换入放大 | §3 现象:释放的页会被下一 step 读区间立刻要求回填 | E1 / E2 |
| 释放后立即换回抵消 RSS 下降 | 即便能释放,换入放大会让常驻量反复回弹,净收益可能被吃掉 | E1,核心未决 |
| 多 sequence / SWA / K-shift 扩展后状态一致性更复杂 | 当前单 seq 边界规避了 `find_slot` 覆盖、shift 重放、seq 共享一致性 | 超出 demo,后续 |
| 后端异步拷贝能力不确定 | E2 prefetch 前提,跨后端差异未确认 | E2,仍需验证 |
