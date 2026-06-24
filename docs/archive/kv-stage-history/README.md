# KV Cache 阶段性文档归档

本目录仅保留 KV Cache 运行时内存优化过程中最关键的阶段性文档，用于说明技术路线的演进、关键转向原因和实验依据。

最终提交材料位于：

- `README_KV_OPT.md`
- `docs/final_technical_report.md`
- `docs/reproduce_kv_cache_optimization.md`
- `docs/kv_trace_replay_stage12c_real_sharegpt_results.md`

本目录保留的 8 个阶段文档对应如下技术链路：

| 序号 | 文档 | 作用 |
|---:|---|---|
| 1 | `kv_paged_read_stage0_source_read.md` | 原始 llama.cpp KV Cache 机制与读侧连续窗口问题定位 |
| 2 | `kv_runtime_swap_stage2_summary_and_roadmap.md` | exact swap 正确性成立，但无法稳定降低 RSS 的根因分析 |
| 3 | `kv_lazy_block_stage_p2_results.md` | lazy tail / lazy clear 阶段对 current RSS 与 peak RSS 的结论 |
| 4 | `kv_paged_read_stage5a2_nonidentity_gather_results.md` | paged row index / non-identity gather，使 idle KV 退出 active read window |
| 5 | `kv_paged_read_stage5_summary_and_roadmap.md` | idle KV block swap-out、madvise、resume swap-in 的阶段收口 |
| 6 | `kv_paged_read_stage5e_mincore_resident_matrix_results.md` | mincore resident page 诊断，证明 RSS 下降来自 KV resident pages |
| 7 | `kv_paged_read_stage8d_resume_latency_defer_results.md` | prefetch / defer / fast maintenance 对性能回退的控制 |
| 8 | `kv_paged_read_stage9c_semi_real_multisession_3run_results.md` | semi-real multi-session 组合策略验证 |

其他计划文档、重复 sweep、中间 smoke/debug 结果和已被最终报告吸收的过程性材料不再保留在当前仓库树中。如需追溯完整历史，可通过 Git 历史或备份 tag 查看。
