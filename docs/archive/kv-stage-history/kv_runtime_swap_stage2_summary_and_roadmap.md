# Stage 2 KV Runtime Swap Summary and Roadmap

本文总结 Stage 2 KV runtime swap / approximate-window 的阶段成果，并给出下一阶段路线评估。本文只用于项目汇报和后续开发决策，不表示 Stage 2 已经完成 RSS 优化，也不进入 Stage 4 代码实现。

## 1. 项目主线

主线目标：

```text
llama.cpp KV cache 从连续预分配走向块式 / 惰性 / 可换出 / 可释放的内存管理机制。
```

Stage 1 和 Stage 2 的关系：

- Stage 1：lazy clear / lazy tail / low-mem warmup，主要解决初始化和 warmup 阶段的峰值内存问题。它关注的是 KV cache 尚未被运行期 attention 持续读取之前，如何避免提前提交或清零未使用的大块内存。
- Stage 2：runtime swap / approximate window，主要探索运行期 KV cache 能否被换出、释放，或缩小物理读窗口。它关注的是生成过程中 KV history 已经增长后，attention graph 仍持续读取 `[0,n_kv)` 这一结构性约束。

两者共同服务同一方向：把 KV cache 从“启动时连续分配并长期常驻”推进到“按需提交、按需读取、按需释放”的内存管理模型。

## 2. Stage2 已完成工作

### 2.1 exact runtime swap

做了什么：

- 实现 file-backed backing store。
- 实现 `swap_out_cell` / `swap_in_cell`。
- 实现 `ensure_resident`。
- 实现 `swap_out_window`。
- 在 graph 读取前通过 `ensure_resident([0,n_kv))` 保证 exact attention 语义。

证明了什么：

- runtime KV swap 的无损 correctness 闭环成立。
- window=1 激进 swap 配置下，输出与 baseline sha256 一致。
- backing store I/O、cell metadata、逐层 K/V 字节搬运、swap-in 恢复路径没有破坏输出。

没证明什么：

- 没证明 RSS 能下降。
- 没证明 exact swap 是性能路径。
- 没证明可以跳过 `[0,n_kv)` 中任何历史 token。

为什么继续或停止：

- exact correctness 是必要里程碑，证明 KV cell 可以安全离开原 tensor 再恢复。
- 但 exact 模式为了无损语义必须在 attention 前全量 `ensure_resident([0,n_kv))`，因此结构上无法降低 active RSS。exact 路线作为 correctness 基座保留，不继续作为 RSS 主线推进。

### 2.2 exact-rss / madvise

做了什么：

- 在 exact swap 基础上实现 safe page-aligned `MADV_DONTNEED` probe。
- 对可释放区间做页对齐、边界内缩和布局守卫，避免误释放仍在读写范围内的数据。
- 在 exact correctness 配置下验证 madvise 路径不会破坏输出。

证明了什么：

- 当前 safe `MADV_DONTNEED` 实现没有造成数据损坏。
- page-aligned release probe 可以作为后续释放路径的基础工具。

没证明什么：

- 没证明 RSS 下降。
- 没证明 exact mode + madvise 能带来实际内存收益。

为什么继续或停止：

- 停止作为 exact RSS 优化方向。根因是 exact 模式每步都会 `ensure_resident([0,n_kv))`，随后 graph 仍读取 full prefix。即使窗口外页被 madvise，下一步也会被换回并重新提交。
- madvise 工具本身保留，供 approximate / paged-read 等真正缩小物理读范围的路线复用。

### 2.3 approx0

做了什么：

- 新增 `LLAMA_KV_SWAP_MODE=approx` no-op scaffold。
- 默认路径不变。
- approx mode 不触发 swap I/O，不改变 attention，不改变输出。

证明了什么：

- approx mode 的 env / mode 框架可以安全加入。
- 不带 env 的 default 路径保持不变。

没证明什么：

- 没证明 approximate-window 语义。
- 没证明 RSS 收益。

为什么继续：

- approx0 是低风险脚手架，为后续 mask-only 和 physical-view probe 提供明确门控。

### 2.4 approx1 mask-only sliding window

做了什么：

- 在 approx mode 下只修改 attention mask。
- window=512 时覆盖短上下文历史，输出与 baseline 一致。
- window=64 时屏蔽窗口外历史，输出允许不同，符合 approximate-window 预期。

证明了什么：

- mask-only sliding-window 语义成立。
- approximate-window 的有损语义可以被可控触发。

没证明什么：

- 没证明 RSS 下降。
- 没证明物理 K/V read window 缩小。

为什么继续：

- mask-only 是必要语义基线，但不可能降低 RSS。attention matmul 仍物理读取连续 `[0,n_kv)`，mask 只影响 softmax，不减少 K/V tensor 读范围。
- 因此必须继续探索 physical view / read window 的缩小。

### 2.5 approx2 initial view-offset probe

做了什么：

- 尝试让 K/V view 从动态 `visible_lo` 开始，只覆盖近似窗口内的物理 KV rows。
- 对比 mask 侧和 K/V view 侧的窗口参数生命周期。

证明了什么：

- 初始 view-offset probe 失败的根因不是 mask 语义，而是 graph 生命周期。
- K/V view 在 graph reserve / graph build 阶段固定。
- mask 在 apply 阶段使用真实 `sinfo` 和动态窗口参数。
- `can_reuse_kq_mask` 只比较 shape，不比较 `visible_lo`。
- window 固定时 `n_kv` shape 可保持不变，graph 会复用旧 K/V view，导致 dynamic `visible_lo` 无法进入 physical view。

没证明什么：

- 没证明 dynamic physical view correctness。
- 没证明 RSS 收益。

为什么继续：

- 该 probe 定位了 route-A 的必要条件：`visible_lo` 变化时必须禁止 graph reuse，或改用 idx-gather / paged-read 等动态读机制。

### 2.6 fix-C no-offset physical-view probe

做了什么：

- 在 `visible_lo=0` 且 window 覆盖真实历史时，将 reserve / graph K/V view 从 `512` 缩到 `256`。

证明了什么：

- no-offset 场景下缩短 physical view length 可行。
- 输出与 baseline bit-exact。
- RoPE / mask / graph shape 在 `visible_lo=0` 的缩读窗口下没有破坏 correctness。

没证明什么：

- 没证明 dynamic `visible_lo > 0`。
- 没证明 offset view correctness。
- 没证明 RSS 收益。

为什么继续：

- fix-C 证明“缩短连续 physical view length”本身不是问题，但没有解决动态窗口低端。下一步必须让 dynamic `visible_lo` 真正进入 K/V view。

### 2.7 route-A dynamic-view correctness probe

做了什么：

- 在 approx dynamic view 下保存当前 `visible_lo`。
- 当当前 `visible_lo` 与旧 graph 的 `visible_lo` 不同时，让 graph reuse 失败并重建。
- K/V physical view 使用 `byte_offset = row_size * visible_lo`。
- mask 列基准同步到同一 `visible_lo`。

证明了什么：

- dynamic `visible_lo` 可以通过 graph rebuild 进入 K/V physical view。
- window=256 输出与 baseline bit-exact。
- window=64 输出与 approx1 mask-only 一致。
- no NaN / no backend failure。
- default no-env 路径不变。

没证明什么：

- 没证明 RSS 下降。
- 没证明 route-A 是性能路径。
- 没证明当前方案接近 PagedAttention 终态。

为什么继续或停止：

- route-A 作为 correctness probe 已经达成目标。
- 但它依赖 graph rebuild，性能不是终态；连续 view 下读窗仍有 `PAD(window,256)` 粒度限制。它适合作为 approx3 feasibility 的基础，不适合作为长期架构。

## 3. 当前核心结论

1. exact swap correctness 成立，但无法降低 RSS。原因是 exact 语义要求每步在 graph 前 `ensure_resident([0,n_kv))`，graph 仍读 full prefix。
2. mask-only approximate-window 语义成立，但无法降低 RSS。mask 不减少 K/V tensor 的物理读范围。
3. dynamic physical view correctness 已由 route-A 初步验证。`visible_lo` 变化时重建 graph，可以让 K/V view 的 byte offset 跟随 apply-time window。
4. 当前仍没有 RSS 收益。Stage 2 到目前为止完成的是 correctness / feasibility probe，不是内存优化完成态。
5. 连续 view 下物理读窗最小粒度是 `PAD(window,256)`。例如 `window=64` 时实际 K/V view 长度仍为 `256`，窗口内包含有效历史和 padded empty rows；empty rows 依赖零初始化并由 attention mask 屏蔽。

对 approx3 的直接影响：

- 即使在 route-A 上叠加 madvise / release，收益也会受 `256` 连续读窗粒度限制。
- route-A 通过 graph rebuild 让 dynamic offset 生效，但 graph rebuild 性能不是终态。
- 真正解决 dynamic offset 与 graph reuse 矛盾，更可能需要 idx-gather / paged-read / block-table。

## 4. 下一阶段路线评估

### A. 继续 approx3：连续 dynamic view + MADV_DONTNEED / RELEASED

路线描述：

- 以 route-A 为基础。
- 保持连续 K/V physical view。
- 对窗口外 KV rows 尝试 `MADV_DONTNEED` 或标记 `RELEASED`。
- 只验证 current RSS 是否能在 `PAD(window,256)` 粒度下降。

优点：

- 沿当前代码最小递进。
- 可复用 exact-rss 中已经验证过的 safe page-aligned madvise 思路。
- 可快速回答“连续 dynamic view 是否具备 RSS feasibility”。

缺点：

- graph rebuild 性能差，不适合作为终态。
- 物理读窗最小长度仍是 `PAD(window,256)`，例如 window=64 仍读 256 rows。
- RSS 收益可能有限，尤其在小 ctx / 小模型 / backend 分配粒度较粗时可能被噪声淹没。

风险：

- 释放窗口外页后，graph rebuild、mask、empty rows、page alignment 需要重新验证。
- padded empty rows 依赖零初始化和 mask 屏蔽；release 后必须确保不会读到未定义数据。
- 连续 view 的页对齐边界可能导致可释放区间小于理论窗口外范围。

适合：

- 小规模 RSS feasibility probe。
- 不适合作为长期性能路径或终态架构。

### B. idx-gather / paged-read / block-table

路线描述：

- 不再依赖连续 `[visible_lo, visible_hi)` physical view 表示所有可见 KV。
- 使用 input idx 张量、block table 或 paged read 机制动态选择 K/V rows。
- 让 graph shape 更稳定，同时让运行期窗口选择可以动态变化。

优点：

- 从结构上解决 dynamic offset 与 graph reuse 的矛盾。
- 更接近 PagedAttention 终态。
- 可以表达非连续窗口，例如 sink + recent。
- 有机会把物理 residency / page release 与实际读取集合对齐。

缺点：

- 工程量明显更大。
- 可能引入 gather buffer 或中间 materialization，不一定直接降低 cache RSS。
- 需要重新评估 backend 支持、kernel 性能、graph shape、mask 语义和数据布局。

需要配合：

- madvise / page release。
- block residency metadata。
- 更明确的 cell lifecycle，例如 resident / released / swapped 的状态转移。

适合：

- 作为下一阶段主线。
- 适合作为通往 PagedAttention-style KV 管理的架构方向。

### C. 暂停 runtime swap，整理成果

路线描述：

- 不继续推进 approx3 或 paged-read 实现。
- 整理 Stage 1 + Stage 2 的 correctness / feasibility / negative results。
- 将当前成果用于赛题汇报、阶段报告或后续设计输入。

优点：

- 当前已有完整阶段性成果：lazy warmup、exact correctness、exact-rss negative result、approx mask semantics、dynamic view correctness。
- 风险低，适合时间有限时收束。

缺点：

- 没有进一步 RSS 收益。
- runtime KV cache 的终态方案仍未实现。

适合：

- 时间有限、需要稳定汇报材料时。
- 作为后续 paged-read / block-table 设计前的阶段冻结点。

## 5. 推荐路线

推荐决策：

1. 不直接进入大规模 approx3。
2. 先做一个很小的 approx3 feasibility probe，只验证连续 view + `MADV_DONTNEED` 是否能在 `window=256` 粒度下降 current RSS。
3. 如果收益很小或性能太差，停止 approx3。
4. 将下一阶段主线转向 idx-gather / paged-read / block-table。

推荐表述：

```text
route-A 已完成 correctness 验证；approx3 只适合做 RSS feasibility probe；
真正终态应转向 paged-read / block-table。
```

该推荐的依据：

- exact 路线证明无损 swap 正确，但 full-prefix read 结构阻断 RSS 收益。
- approx1 证明 mask 语义，但 mask-only 不改变物理读。
- route-A 证明 dynamic offset correctness，但 graph rebuild 不是性能路径。
- 连续 view 的 `PAD(window,256)` 粒度限制意味着 approx3 的收益上限有限。
- paged-read / block-table 才能同时处理 graph reuse、非连续窗口和 residency 管理。

## 6. 禁止夸大

后续汇报和 commit message 中应避免以下表述：

- 不声称 Stage 2 已经完成 RSS 优化。
- 不声称 route-A 是性能路径。
- 不声称 approx2 已经等价于 paged attention。
- 不声称 madvise 一定能带来 RSS 收益。
- 不声称连续 dynamic view 已经解决 sink + recent。
- 不声称当前实现已经进入 Stage 4。

当前可以诚实声称：

- exact runtime swap correctness 已验证。
- exact-rss madvise safety 已验证，但 RSS negative result 明确。
- approx mask-only semantics 已验证。
- dynamic physical view correctness 已由 route-A 初步验证。
- 下一阶段需要在 small approx3 RSS probe 和 paged-read / block-table 主线之间做取舍。

