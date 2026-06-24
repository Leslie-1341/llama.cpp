# Stage 2 — Approximate Sliding-Window KV Plan

状态：设计稿（design only）。本文不改源码、不 build、不跑实验、不 commit。

## 0. TL;DR

- exact 路线已收束：correctness 成立、`MADV_DONTNEED` 安全，但**全量 `ensure_resident([0,n_kv))` 架构下 RSS 不降**（实测噪声级）。
- approximate-window 是**有损滑窗近似**，不再追求 sha256 bit-exact，目标是通过**缩小物理 read window** 降低 active RSS。
- **硬约束**：mask-only 不能降 RSS——attention matmul 仍物理读连续 `[0,n_kv)`。真正降 RSS 必须缩小 effective `n_kv` / 物理 view。
- sink + recent 在连续 view 下无法直接表示，**第一版不做 sink**。
- 评价口径从 bit-exact 切到 **perplexity / 多 prompt smoke / 输出连贯性 / token/s / RSS**。

---

## 1. exact 路线已收束

Stage 2 exact 路线到此为止，作为已验证的 correctness 里程碑保留：

1. **exact correctness 成立**：`swap_out_window → swap_out_cell → file-backed store → ensure_resident → swap_in_cell → graph`。window=1 激进触发下，输出与 baseline sha256 完全一致
   （`5c16d81eabb17f8de95d02e32f359b8f7d2b028ec11de92587870586931e4006`）。
2. **exact-rss2 验证 `MADV_DONTNEED` 安全**：per-run page-aligned inward clipping，绝对地址 align_up 起点 / align_down 终点，`row==nb[1]` 守卫，逐层 K/V 独立。三组（baseline / swap-no-madvise / swap-madvise）输出 sha256 完全一致，证明释放路径零数据损坏。
3. **但 RSS 不降**：全量 `ensure_resident([0,n_kv))` 每步把窗口外 cell 全部换回，被 madvise 释放的页被随后的 `ensure_resident` write-fault 立即重新提交。graph 执行时 `[0,n_kv)` 全部 resident。实测 peak/current 均在 8161–8163 MiB 噪声带内，madvise 组 peak 甚至最高。

**根因**：active/peak RSS 由 apply() 内 graph 读取 `[0,n_kv)` 决定。只要每步全量 ensure，swap_out / madvise 触发再多也无法降低 active RSS。这是结构性的，不是实现 bug。

---

## 2. approximate-window 的定位

| 维度 | exact swap | approximate-window |
|---|---|---|
| 语义 | 无损：窗口外换出再换回，attend 全历史 | 有损：窗口外不参与 attention |
| 正确性口径 | sha256 bit-exact == baseline | perplexity / 质量 / smoke（不要求 bit-exact） |
| RSS 目标 | 不降（已实证） | 降低 active RSS（缩小物理 read window） |
| backing store | 需要（换回） | 第一版可不需要（窗口外纯丢弃 RELEASED） |
| 长程依赖 | 完整保留 | 牺牲：超窗口历史不可见 |

定位一句话：**用可量化的轻度质量损失，换 exact 路线拿不到的 active RSS 真实下降。** 这是 StreamingLLM 式有损滑窗，必须在文档与日志中诚实标注"approximate = 牺牲长程依赖换内存"，不得宣称无损。

---

## 3. 硬约束：mask-only 不能降 RSS

关键代码机制（已确认）：

- `get_n_kv()`（`src/llama-kv-cache.cpp:2087`）返回 `GGML_PAD(used_max_p1, 256)`，是从 cell 0 起的连续 `[0,n_kv)`。
- `get_k()` / `get_v()`（`:2103` / `:2123`）用该 `n_kv` 构造**连续 view**，attention matmul **物理读全部 `[0,n_kv)` 行**。
- `set_input_kq_mask`（`:2567`）只决定哪些位置参与 softmax，**不减少物理读**。

推论：

1. **masking ≠ 不读**。即便 mask 屏蔽窗口外 cell，matmul 仍读那些行，触发被 madvise 释放页的 read-fault，active RSS 当场弹回。
2. 真正降 active RSS **必须缩小 effective `n_kv` / 物理 view**，让 graph 根本不覆盖窗口外行。
3. **sink + recent 在连续 view 下无法同时表示**：recent 在高位 cell、sink 在低位，连续 view 不能跳过中间。第一版**放弃 sink**，只保留最近连续窗口。强行做 sink 需要非连续 paged read（Stage 4）或物理 compaction。

---

## 4. 分阶段路线

- **approx0**：新增 `kv_swap_mode::approx` + env，默认关闭，纯 no-op（走 exact 路径）。build + 默认 sha256 不变。
- **approx1（mask-only，建立近似语义，不期待 RSS 下降）**：仅在 `set_input_kq_mask` 中对 approx mode 屏蔽 `[sink, n_kv-window)`，不动 `n_kv`。**预期 RSS 不降**（matmul 仍全读），但可量化输出质量随 window 变化，建立质量评估脚本与近似正确性基线。
- **approx2（source-read / feasibility）**：专门分析 `get_n_kv` / `get_k` / `get_v` 的 row offset、pos / mask / RoPE 重映射，设计物理缩小 read window 的方案。这一步是架构难点，建议单独细化设计。
- **approx3（真正缩小物理 read window）**：effective `n_kv` 物理缩到窗口尺寸，view 起点偏移到 `n_kv-window`，窗口外 cell 标 `RELEASED`、approx 下 `ensure_resident` 只 ensure 窗口内，窗口外原 KV 页 madvise（复用 exact-rss2 safe-range 算法）。**这一步才可能降 active RSS。**
- **approx-rss-measure**：复用 `LLAMA_KV_SWAP_RSS_SAMPLE=1`，对比 baseline / exact-swap / exact-rss / approx 的 peak / current / token/s。建议 ctx=2048/4096，window=64/128/256。
- **approx-quality**：多 prompt smoke + perplexity + 输出连贯性，对比 baseline 与各 window。第一版只做 demo 级质量检查。
- **approx-results**：结果文档。

架构鸿沟提示：approx1（mask-only）→ approx2/3（改 n_kv / view / pos）之间有实质架构跨度。approx2 的 get_k/get_v offset 与 pos 重映射是真正难点与主要风险来源。

---

## 5. 风险清单

1. **mask-only 不降 RSS（最易踩）**：approx1 必然不降 active RSS，必须事先标注它只是语义/质量步。
2. **view offset / pos / RoPE 错位（最高危）**：approx3 改窗口起点后，RoPE 用的 pos、mask causal 对齐、cell↔pos 映射必须一致重映射，错则乱码或崩溃。
3. **连续 view 与 sink 冲突**：第一版放弃 sink；强行保留会与连续 view 冲突。
4. **输出质量劣化**：滑窗丢长程依赖，长程引用任务质量下降，需 perplexity 量化、demo 选合适任务。
5. **multi-seq / shift / defrag**：approx 改 n_kv 语义，与 seq_rm/seq_cp/defrag/shift 交互未知。第一版**仅支持 single-seq、causal、!v_trans、n_stream==1**，其余强制回退 exact 并 warn。
6. **与 Stage 1 lazy-tail / lazy-clear 交互**：lazy-tail 基于 `GGML_PAD(n_kv,256)`，approx 缩小 n_kv 后 tail 起点变化，需确认不与窗口区重叠。
7. **与 exact swap 状态机交互**：approx 引入 RELEASED 永久态，需明确 approx 下不复用 ensure/swap_in，避免误换回已 RELEASED cell。

---

## 6. Codex approx0 最小任务提示词

```
任务 approx0（只加 mode 脚手架，默认关闭，no-op，不改任何 attention/n_kv 行为）：
1. 在 kv_swap_mode 枚举新增 approx（与现有 exact 并列）。
2. 解析 LLAMA_KV_SWAP_MODE=approx 时设 kv_swap_mode_=approx；
   复用现有 LLAMA_KV_SWAP_WINDOW / LLAMA_KV_SWAP_SINK 解析。
3. approx mode 仅在 single-seq、causal、!v_trans、n_stream==1 时允许；
   否则 LOG_WARN 一次并强制回退 exact。
4. 本轮 approx 为纯 no-op：get_n_kv / get_k / get_v / set_input_kq_mask /
   ensure_resident / swap_out_window / apply 顺序全部不变，approx 走与 exact
   相同路径（或直接 early-return 到 exact 行为）。
5. 新增统计占位 kv_approx_* 计数器（先全 0），在析构统计处打印 mode 名。
验收：build 通过；不带 env 时输出 sha256 与之前完全一致；
LLAMA_KV_SWAP_MODE=approx 时也不改变输出（因为 no-op），sha256 仍一致；
日志正确打印 mode=approx 及 window/sink。
```

---

## 7. 建议

- **建议提交本设计文档**：它把 exact 收束结论、approx 有损定位、mask-only 硬约束、分阶段路线钉死，是后续工作的锚点。
- **建议进入 approx0**：零风险纯脚手架，固定 mode 门控 / single-seq 限制 / env 解析 / 统计占位，不碰 attention 行为。
- 进入前对齐预期：真正的 RSS 收益在 approx3（物理缩 n_kv + view offset），approx1 mask-only 不降 RSS；评估口径切换到 perplexity / 质量。
