# MoE FFN Pipeline Debug Report

- Summary: `/root/llama.cpp/rss-stage-results/moe-ffn-pipeline-debug-20260719-153344/summary.tsv`
- Static probes: `/root/llama.cpp/rss-stage-results/moe-ffn-pipeline-debug-20260719-153344/static-probes.txt`
- Baseline PPL: `46648.5435`

## Key Runs

- `baseline_fuse0_t6` rc=0 ppl=46648.5435 delta=NA check_abs=NA
- `fused_nonpipe_t6_rt128` rc=0 ppl=46648.5435 delta=0 check_abs=NA
- `pipeline_check_t6_rt16_r3` rc=0 ppl=46648.5435 delta=0 check_abs=NA

## Findings

- Non-pipeline fused FFN matches baseline for this run, so gate/up/down math is probably not the first suspect.
- Pipeline default probe delta vs baseline: 0.
- Static probe: run_index is a per-thread local counter but is used as part of the shared pipe key. If threads skip or group work differently, they can rendezvous on different entries for the same op.

## Suggested Next Instrumentation

- Log a graph-stable FFN run id from the op/context instead of a thread-local `run_index`, then print `(ith, run_index, op, group, mb, batch, n_blocks)` for the first mismatching op.
- Temporarily compare the complete `dsts[j][0:n_hidden]` after each fused down op, not only one row tile, because current `ffn-check` can miss rows owned by other consumers or later graph writes.
- Force `ROW_TILE=1` and `BLOCK_RING=1`; if drift remains, focus on op lifecycle/dst accumulation rather than ring aliasing.
