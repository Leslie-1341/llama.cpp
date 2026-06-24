# Stage 4A — block-aware release 完整 benchmark results

前置：

- [docs/kv_paged_stage4a_release_benchmark_plan.md](kv_paged_stage4a_release_benchmark_plan.md)
- [docs/kv_paged_read_stage4a_block_release_results.md](kv_paged_read_stage4a_block_release_results.md)

数据来源：`/root/oscomp/kv_logs/paged_stage4a_release_benchmark/summary.csv`（27 个 cell，其中 ctx=512,n=512 三组因 `n_tokens >= ctx` 被跳过，实际成功运行 24 个）。

---

## 1. 实验目标

验证 llama.cpp paged KV 的 **block-aware release** 能否通过 Linux `MADV_DONTNEED` 降低**运行期 current RSS**，并严格区分 current RSS 与 peak RSS：

- 证明 release 真实发生（advised bytes > 0、blocks released > 0）且不破坏输出（三组 sha256 一致）；
- 量化 current RSS 实际降幅，并对比 advised bytes（验证 advise ≠ 实际回收）；
- 诚实报告 peak RSS —— 不声称 peak 优化，除非数据支持；
- 用 `paged_active_release_violation=0` 证明安全性不变量（注意力可见的有效 KV 块从不被释放后读取）。

---

## 2. 实验设置

| 项 | 值 |
|---|---|
| 对比组 | `baseline`（不启 paged）/ `paged_no_release`（`LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1`）/ `paged_release`（再加 `LLAMA_KV_PAGED_RELEASE=1`） |
| ctx | 512 / 2048 / 4096 |
| n_tokens | 16 / 128 / 512（ctx=512,n=512 跳过） |
| prompt | `"Hello, how are you?"` |
| seed | 42 |
| 采样参数 | `-t 12 -ngl 0 -fa on -no-cnv -ctk f32 -ctv f32 -nkvo`（贪心、CPU、F32 KV、no KV offload） |
| current RSS 来源 | stats `paged_block_release_rss_drop_*_kb`（release 前后 `/proc/self/statm` 采样） |
| peak RSS 来源 | `/usr/bin/time -v` `Maximum resident set size` |

---

## 3. benchmark 表格（paged_release 组关键指标）

| ctx | n | sha256 一致 | advised (MiB) | current RSS drop (MiB) | drop/advised | blocks released | active_vio | padded_vio | peak RSS (KB) |
|---|---|---|---|---|---|---|---|---|---|
| 512 | 16 | ✅ | 120.00 | 59.86 | 0.499 | 32 | 0 | 0 | 8260804 |
| 512 | 128 | ✅ | 120.00 | 59.86 | 0.499 | 32 | 0 | 0 | 8261368 |
| 2048 | 16 | ✅ | 840.00 | 420.00 | 0.500 | 224 | 0 | 0 | 8655268 |
| 2048 | 128 | ✅ | 840.00 | 420.00 | 0.500 | 224 | 0 | 0 | 8655000 |
| 2048 | 512 | ✅ | 840.00 | 420.00 | 0.500 | 224 | **0** | 32160 | 8655124 |
| 4096 | 16 | ✅ | 1800.00 | 899.75 | 0.500 | 480 | 0 | 0 | 9178292 |
| 4096 | 128 | ✅ | 1800.00 | 899.75 | 0.500 | 480 | 0 | 0 | 9178140 |
| 4096 | 512 | ✅ | 1800.00 | 899.75 | 0.500 | 480 | **0** | 32160 | 9178332 |

全部 cell 的 `paged_block_release_fail=0`、`paged_row_idx_fail=0`、`paged_logical_to_physical_fail=0`、`paged_swap_backend_failures=0`、`paged_blocks_released_unused == paged_blocks_released`。

---

## 4. correctness 结果

所有实际运行的 cell，`baseline` / `paged_no_release` / `paged_release` 三组输出 sha256 **完全一致**（按 n_tokens 分组的基准 sha256：n=16 与 n=128 短输出、n=512 长输出各自一致，三组同值）。这说明无论是否启用 paged 寻址、是否启用 block release，模型输出逐位等价 —— release 没有破坏任何参与计算的 KV 数据。

寻址与翻译守卫全 0（`row_idx_fail` / `logical_to_physical_fail` / `block_release_fail` / `backend_failures`），release 路径本身无失败。

---

## 5. current RSS 结果

current RSS 随 ctx 稳定增长，降幅与 advised 规模成正比：

- **ctx=512**：advised 120 MiB，current RSS drop ≈ 59.86 MiB，比例 ≈ 0.499；
- **ctx=2048**：advised 840 MiB，current RSS drop ≈ 420 MiB，比例 ≈ 0.500；
- **ctx=4096**：advised 1800 MiB，current RSS drop ≈ 899.75 MiB，比例 ≈ 0.500。

降幅在三组 ctx 下都稳定落在 advised 的约 50%。这是本 benchmark 的核心收益证据：block-aware `MADV_DONTNEED` 确实压低了运行期常驻内存，且随上下文规模线性放大。

**重要：advised bytes ≠ 实际 RSS drop。** `MADV_DONTNEED` 只是向内核建议丢弃页，实际回收受页驻留状态影响。这里稳定回收约 50%（K/V 两路中约一路的页在采样窗口内被实际收回，另一路尚未触发回收或仍驻留），所以不能用 advised 充当收益，必须以实测 drop 为准。

---

## 6. peak RSS 结果

peak RSS（`Maximum resident set size`）在三组之间**基本持平**，差异均在 MiB 量级噪声内。例如 ctx=4096,n=512：baseline 9179248 KB、paged_no_release 9187268 KB、paged_release 9178332 KB —— release 组甚至略低于 no_release，但差值远小于 1%，属噪声。

原因：release 发生在 peak **之后**。物理页先被引用、提交、达到峰值，advise 只能压低此后的 current RSS，无法回溯压低已发生的峰值；peak 由构造期 buffer 提交钉住。

**结论：Stage 4A 只优化 current RSS，不声称 peak RSS 优化。** 数据不支持 peak 收益，故不报告 peak 优化。

---

## 7. 性能 / 输出延迟结果

- **release_vs_baseline 平均 ≈ 0.906**：release 组平均速度约为 baseline 的 90.6%，即启用 paged + release 整体有约 9% 的吞吐损失。
- **release_vs_no_release 平均 ≈ 1.006**：release 相对 paged_no_release 几乎无额外开销（差异在噪声内）。

这说明性能开销**主要来自 paged read path（非连续寻址 / row_idx gather）本身，而非 block release 动作**。换言之，一旦付出了 paged 寻址的成本，开启 release 来回收 current RSS 几乎是“免费”的 —— 这对边缘设备内存优化是有利的权衡：用约 9% 的吞吐换取随 ctx 线性增长、最高约 900 MiB 的运行期内存回收。

prompt eval 时间在 paged_release 组略高（如 ctx=4096 的 ~290–320 ms vs baseline ~182 ms），主要落在 release 首次 madvise 与寻址建立阶段；eval 阶段 tok/s 三组接近。

---

## 8. padded violation 解释

n=512 的两个 cell（ctx=2048 与 ctx=4096）出现 `paged_release_violation=32160`，但诊断拆分显示：

- `paged_active_release_violation = 0`；
- `paged_padded_release_violation = 32160`。

含义：`set_input_paged_row_idx` 填充的是 **padded row_idx 宽度**（`GGML_PAD(used_max_p1, n_pad)`），它超出真实活跃 KV 范围，包含 padding/masked row。这些 padding row 解析到的物理块可能是已被释放的空闲块（被释放的全是 UNUSED 块，`blocks_released_unused == blocks_released` 佐证），于是计入 violation。但这些 row 会被注意力 mask 掉，**不参与任何有效计算**。

**这不是真实错误**，证据三重锁定：

1. `paged_active_release_violation = 0` —— 没有任何活跃、非空、注意力可见的 row 命中已释放块。这是安全性不变量（INV-1）成立的**关键直接证据**；
2. 三组 sha256 一致 —— 若有效数据被破坏，输出必变；
3. `paged_blocks_released_unused == paged_blocks_released` 且 `paged_block_ensure_released=0` —— 释放的全是从未被引用的空闲容量块，写路径也从未需要把任何 RELEASED 块复活。

两个 ctx 的 padded violation 数完全相同（均 32160），因为它只取决于 n=512 的步数与 padding 几何，与总 ctx 无关 —— 更大的 ctx 只是多出更多 UNUSED 空闲容量，不改变“哪些 padding row 命中已释放块”。

因此 n=512 在 correctness 上**判定通过**：输出逐位一致 + `active_release_violation=0`，padded violation 仅作为良性可观测项如实记录。

---

## 9. 结论与限制

**可诚实声称：**

1. block-aware `MADV_DONTNEED` release 在 paged KV 下 correctness 成立 —— 24 个 cell 三组 sha256 全一致，`paged_active_release_violation=0`，守卫字段全 0；
2. current RSS 实测下降，随 ctx 线性放大：ctx=512/2048/4096 分别约 59.86 / 420 / 899.75 MiB，稳定回收 advised 的约 50%；
3. release 本身几乎无额外性能开销（vs no_release ≈ 1.006），整体约 9% 吞吐损失主要来自 paged read path。

**明确限制（不夸大）：**

1. **不声称 peak RSS 优化** —— peak 三组持平（噪声量级），release 在 peak 之后发生；
2. **advised bytes ≠ RSS drop** —— 实际仅回收约 50%，须以实测 drop 为准；
3. **padded violation 非错误** —— 来自 padded/masked row_idx，`active_vio=0` 才是安全保证；不可将 32160 解读为数据损坏；
4. 释放的全是 UNUSED 空闲块，**未回收 active history**，也**未实现 swap**（RELEASED 不可恢复）；
5. 单模型、单 prompt、F32 KV、CPU 路径实测，收益不外推到所有模型 / dtype / backend；
6. ctx=512,n=512 因 `n_tokens >= ctx` 跳过，非失败。

下一步方向（Stage 4B）：在预留的 SWAPPED 状态上实现 backing store，回收 evicted history 块并支持 swap-in，需重评 I/O 时序与读前 ensure-resident 不变量。
