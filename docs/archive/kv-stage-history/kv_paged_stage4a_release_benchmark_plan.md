# Stage 4A — block-aware release 完整 benchmark plan

前置：

- [docs/kv_paged_read_stage4a_block_release_plan.md](kv_paged_read_stage4a_block_release_plan.md)
- [docs/kv_paged_read_stage4a_block_release_results.md](kv_paged_read_stage4a_block_release_results.md)
- [docs/kv_paged_read_stage4b_rss_plan.md](kv_paged_read_stage4b_rss_plan.md)

本文是 **benchmark 设计文档**，只规划实验，不改源码、不 build、不跑实验。Stage 4A（block-aware madvise-only release）已在单点上证明：ctx=4096 下 advised bytes ≈ 1785 MiB、current RSS 实测下降 ≈ 892 MiB、sha256 与 baseline 一致、peak RSS 基本不降。本计划把该单点扩成完整对比 benchmark，让结论可复现、可量化、不夸大。

---

## 1. 实验目标

1. **证明 block-aware release 通过 `MADV_DONTNEED` 降低 current RSS**：在 paged_release 组观察到 `paged_block_release_rss_drop_*_kb > 0`，且 advised bytes（`paged_block_release_bytes`）> 0。
2. **严格区分 current RSS 与 peak RSS**：current RSS 来自 release 前后采样（stats / 可选外部 sampler），peak RSS 来自 `/usr/bin/time -v` 的 `Maximum resident set size`。两者分别报告，不混用。
3. **不声称 peak RSS 优化，除非数据支持**：Stage 4A 的 release 发生在 peak 之后，预期 peak 基本不降；若某配置 peak 确实下降，必须有数据支撑才可报告，否则只报告“peak 不降”这一诚实结论。
4. **correctness 守门**：三组输出 sha256 必须一致，否则 RSS 数据无意义。

非目标：不证明 swap / backing store（Stage 4B 方向），不回收 active history，不优化 peak 构造期内存。

---

## 2. 对比组

| 组名 | 环境变量 | 说明 |
|---|---|---|
| `baseline` | （不设 paged 相关 env） | 不启用 paged KV，作为 correctness 与 RSS 基准 |
| `paged_no_release` | `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1` | 启用 paged 寻址但不释放物理块，隔离“寻址开销”与“释放收益” |
| `paged_release` | `LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_SHIFT=1 LLAMA_KV_PAGED_RELEASE=1` | 启用 block-aware madvise release，被测主体 |

三组共用同一可执行文件、同一模型、同一采样参数，仅 env 不同。

---

## 3. 实验矩阵

固定项：

- prompt 固定为 `"Hello, how are you?"`；
- seed 固定为 `42`；
- 采样参数固定为 `-t 12 -ngl 0 -fa on -no-cnv -ctk f32 -ctv f32 -nkvo`（贪心、CPU、F32 KV、no KV offload）；
- 模型固定为 Stage 4A results 所用模型（runner 阶段以 `$MODEL` 变量锁定，保证三组一致）。

变量项（笛卡尔积）：

| 维度 | 取值 |
|---|---|
| `ctx`（`-c`） | 512 / 2048 / 4096 |
| `n_tokens`（`-n`） | 16 / 128 / 512 |

组合数：3 组 × 3 ctx × 3 n_tokens = **27 次运行**。

注意：`n_tokens=512` 配合 `ctx=512` 时生成可能受 ctx 限制，runner 阶段需确认 `n_tokens <= ctx - prompt_tokens`，否则该 cell 标记为 N/A 并在 results 中说明，而非静默截断。

---

## 4. 指标

每次运行采集以下指标，逐一落 CSV：

| 指标 | 来源 | 含义 |
|---|---|---|
| sha256 | 对 `.out` 内容做 sha256 | correctness 守门 |
| current RSS | stats `paged_block_release_rss_*`（见 §5）/ 可选 sampler | 运行期常驻内存 |
| peak RSS | `/usr/bin/time -v` `Maximum resident set size` | 峰值常驻内存 |
| tokens/s | 程序 timing 输出（eval tok/s） | 吞吐 |
| eval time | 程序 timing 输出（eval time ms） | 评测耗时 |
| release bytes | `paged_block_release_bytes` | 累计 advised 字节 |
| blocks released | `paged_blocks_released` | 累计释放物理块数 |
| RSS drop / advised bytes | `paged_block_release_rss_drop_max_kb*1024 / paged_block_release_bytes` | 实际 RSS 降幅占 advised 的比例（≤1，量化“advise≠实际回收”） |
| release_violation | `paged_release_violation` | INV-1，必须 0 |
| row_idx_fail | stats `row_idx_fail`（字段 `paged_row_idx_fail`） | 寻址守卫，必须 0 |
| logical_to_physical_fail | stats `logical_to_physical_fail` | 翻译守卫，必须 0 |
| backend_failures | `paged_swap_backend_failures`（及 `kv_swap_backend_failures`） | 后端失败，必须 0 |

辅助字段（建议一并落 CSV，便于解读）：`paged_blocks_released_unused`、`paged_block_release_fail`、`paged_block_release_rss_before_max_kb`、`paged_block_release_rss_after_min_kb`、`paged_block_release_rss_drop_last_kb`、`graphs reused`。

`baseline` / `paged_no_release` 组的 paged release 指标为 0 或 N/A，CSV 中如实填 `0` / 空。

---

## 5. current RSS 采样方案

1. **`/usr/bin/time -v` 只能拿 peak RSS**：它报告的 `Maximum resident set size` 是进程生命周期峰值，无法反映 release 之后 current RSS 的回落。因此 `/usr/bin/time` 只用于 peak，不用于 current。
2. **current RSS 主路径来自 stats**：从程序退出时打印的 `paged_block_release_rss_*` 字段提取，关键为
   - `paged_block_release_rss_before_last_kb` / `paged_block_release_rss_after_last_kb`（最后一次 release 前后）；
   - `paged_block_release_rss_drop_last_kb`（最后一次降幅，应等于 before−after，自校验）；
   - `paged_block_release_rss_drop_max_kb`（单次最大降幅，主报告值）；
   - `paged_block_release_rss_before_max_kb` / `paged_block_release_rss_after_min_kb`（包络）。
   这些字段由 release 路径内 `/proc/self/statm` 采样得到，反映“release 这一动作”引起的 current RSS 变化，是 Stage 4A 的核心收益证据。
3. **可选增强：外部 `/proc/$pid/status` VmRSS sampler**：runner 可后台起一个轮询线程（如每 50–100 ms 读 `/proc/$pid/status` 的 `VmRSS`），把时间序列写入 `.rss.csv`，用于画 current RSS 随时间的曲线（看到 release 后的回落台阶）。外部 sampler 有调度噪声，仅作可视化与交叉验证，不作为主报告口径——主口径仍以 stats 字段为准。

---

## 6. 输出文件规划

- log 根目录：`/root/oscomp/kv_logs/paged_stage4a_release_benchmark/`
- 汇总：`summary.csv`（每行一次运行，列为 §4 全部指标 + group/ctx/n_tokens）
- 每次运行（命名建议 `{group}_ctx{ctx}_n{n}`）保存：
  - `{name}.out`：模型文本输出（用于 sha256）；
  - `{name}.log`：stderr + stats 全文；
  - `{name}.time`：`/usr/bin/time -v` 输出（peak RSS 等）；
  - `{name}.rss.csv`：可选，外部 VmRSS sampler 时间序列。

---

## 7. 判断标准

通过当且仅当：

1. **correctness**：同一 (ctx, n_tokens) 下 `baseline` / `paged_no_release` / `paged_release` 三组 `.out` 的 sha256 完全一致；
2. `paged_release` 组 `paged_blocks_released > 0`；
3. `paged_release` 组 `paged_block_release_bytes > 0`；
4. `paged_release_violation = 0`；
5. `paged_block_release_fail = 0`；
6. `paged_release` 组 current RSS drop（`paged_block_release_rss_drop_max_kb`）> 0；
7. peak RSS **可以不下降**，但必须如实报告（不得因 peak 不降而判失败，也不得把 peak 不降包装成优化）。

附加守卫（任一非 0 视为异常，需在 results 中解释）：`row_idx_fail`、`logical_to_physical_fail`、`backend_failures`。

---

## 8. 风险

1. **release 发生在 peak 之后，peak 不降**：advise 只能压低此后的 current RSS，无法回溯峰值，peak 由构造期 buffer 提交钉住——这是预期，不是 bug。
2. **`/usr/bin/time` 不能反映 current RSS drop**：必须靠 stats / sampler，不能用 peak 代替 current。
3. **系统 RSS 采样有噪声**：外部 sampler 受调度、page reclaim 时机影响，曲线有抖动；以 stats 字段为主口径降低误读。
4. **advised bytes ≠ RSS drop**：`MADV_DONTNEED` 是建议，实际回收受内核与页驻留状态影响，故须单列 `RSS drop / advised bytes` 比例，避免用 advised 充当收益。
5. **运行时间波动**：tokens/s、eval time 单次有抖动；如需稳定值，runner 可对关键 cell 重复 N 次取中位数（本计划默认单次，重复策略留给实施）。
6. **ctx / n 越大，compute buffer 可能主导 peak**：大配置下 peak 更多由计算缓冲而非 KV 决定，进一步说明 4A 收益限于 current RSS，不应外推到 peak。

---

## 9. 后续 Codex 实施任务拆分

1. **写 runner 脚本**：遍历 3×3×3 矩阵，按组设 env，统一调用 `llama-cli`，用 `/usr/bin/time -v` 包裹，落 `.out`/`.log`/`.time`（可选 `.rss.csv`），从 stats 与 time 提取指标，追加 `summary.csv`。
2. **跑 benchmark**：执行 runner，校验三组 sha256 一致，确认守卫字段全 0。
3. **写 results 文档**：`docs/kv_paged_stage4a_release_benchmark_results.md`，按 §7 判断标准逐条对照，诚实区分 current/peak。
4. **生成结果表**：从 `summary.csv` 生成对比表（current RSS drop、peak、tokens/s、advised bytes、RSS drop/advised 比例），并对 §8 风险逐条用数据回应。

---

## 10. 本轮边界

本轮仅产出本 benchmark plan 文档。不改 C++、不 build、不跑实验、不 commit。runner 脚本、benchmark 执行、results 文档为下一轮（§9）任务。
