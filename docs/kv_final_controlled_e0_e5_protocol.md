# KV cache controlled E0-E5 protocol

This protocol fixes a six-case comparison around the compiled
`llama-kv-idle-swap-resume` workload. It is intentionally limited to scripts,
parsing, and audit artifacts. It does not build llama.cpp and does not change
the workload implementation.

## Fixed workload and source audit

The binary and argument contract comes from
`examples/kv-idle-swap-resume/idle-swap-resume.cpp` and the common batched
argument parser. The P0 regression/stability scripts fix the same model-side
arguments. The most recent local prefetch matrix and confirmation runners
(`/root/oscomp/kv_tools/run-kv-prefetch-matrix.sh` and
`/root/oscomp/kv_tools/run-kv-prefetch-confirmation.sh`) establish the delayed
active-prefetch controls and the round/order recording pattern. Historical
commands in `docs/reproduce_kv_cache_optimization.md` were not treated as proof
that a field is still consumed by this driver.

Every case uses:

```text
-m MODEL
--ctx-size 2048
--n-predict 128
--batch-size 128
--ubatch-size 128
--seed 1
--temp 0
--cache-type-k f32
--cache-type-v f32
--kv-unified
--parallel 4
--log-verbosity 4
--no-log-prefix
--no-log-timestamps
```

The common logger maps lifecycle `LLAMA_LOG_INFO` messages to verbosity 4.
That threshold is required to retain the paged/lazy destructor statistics used
for mechanism verification. It is applied identically to all cases. It does
not enable the feature-specific per-step trace, refault trace, debug probes, or
`--verbose`; those heavy paths remain off.

The requested terminology maps to the driver as follows:

| Protocol term | Actual argument/environment |
|---|---|
| warmup tokens 256 | `LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256` |
| idle sequences 2 | `LLAMA_KV_IDLE_NUM_IDLE_SEQS=2` |
| resume pending token 96 | `LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96` |
| active token statistics | `LLAMA_KV_ACTIVE_TOKEN_STATS=1` |
| paged I/O statistics | `LLAMA_KV_PAGED_IO_STATS=1` |
| debug probes off | `LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0` |
| shadow validation off | `LLAMA_KV_PAGED_SHADOW_VALIDATE=0` |
| heavy trace/timing off | `LLAMA_KV_PAGED_TRACE=0`, `LLAMA_KV_PAGED_IDLE_TRACE=0`, `LLAMA_KV_PAGED_TIMING=0`, resume/refault timing and trace controls `=0` |

The prompts are compiled into the driver: the idle prompt begins `Idle
request: alpha beta ...`, and the default active prompt is `Active request:
list three colors.` No external trace is used.

## Effective E0-E5 mapping

The runner explicitly sets every functional switch for every process and
removes all known stability and fault-injection variables from the child
environment. Common values are:

```text
LLAMA_KV_PAGED_BLOCK_SIZE=16
LLAMA_KV_PAGED_SHIFT=0
LLAMA_KV_PAGED_RELEASE=0
LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=1
LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=0
LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=0
LLAMA_KV_PAGED_MINCORE=0
LLAMA_KV_PAGED_RESUME_PREFETCH=0
LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off
LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0
LLAMA_KV_SWAP=0
LLAMA_KV_SWAP_MODE=exact
LLAMA_KV_SWAP_MADVISE=0
```

Case-specific effective values (later assignments override common values) are:

| Case | lazy clear/tail | paged/ingraph/nonidentity gather | paged swap/idle swap/madvise | active prefetch/auto delayed | every/blocks | defer |
|---|---|---|---|---|---|---|
| E0 | `0/0` | `0/0/0` | `0/0/0` | `0/0` | `1/1` (inactive) | `0` |
| E1 | `1/1` | `0/0/0` | `0/0/0` | `0/0` | `1/1` (inactive) | `0` |
| E2 | `0/0` | `1/1/1` | `0/0/0` | `0/0` | `1/1` (inactive) | `0` |
| E3 | `0/0` | `1/1/1` | `1/1/0` | `0/0` | `1/1` (inactive) | `0` |
| E4 | `0/0` | `1/1/1` | `1/1/1` | `0/0` | `1/1` (inactive) | `0` |
| E5 | `0/0` | `1/1/1` | `1/1/1` | `1/1` | `1/1` | `1` |

The nonidentity switch here is
`LLAMA_KV_PAGED_GATHER_NONIDENTITY=1`. It enables the workload's safe masked
row remapping; `LLAMA_KV_PAGED_SHIFT` remains zero because static shifted
mapping is a separate probe and is not used by the current P0/prefetch
controlled workload. `LLAMA_KV_PAGED_INGRAPH=1` keeps the row-index gather in
the graph.

E2 is **paged row-index + in-graph gather infrastructure only**. Nonidentity
support is allowed and the gather must execute, but idle swap, release, and
reclaim are all off. Actual masked nonidentity remap is driven by idle reclaim
candidates, so `paged_nonidentity_remap_rows=0` is a normal E2 observation.
E3-E5, where idle reclaim candidates exist, verify that actual remap rows are
greater than zero.

`1 block / token` is only this controlled baseline. It is not a core default
or a global optimum. E1 is independent from E2-E5: the source disables lazy
tail when nonidentity paged mapping is active, so their effects must not be
added.

## Execution order

`RUNS=1` uses E0, E1, E2, E3, E4, E5 once for a functional smoke. `RUNS=3`
uses these interleaved orders:

```text
round 1: E0 E3 E1 E4 E2 E5
round 2: E5 E2 E4 E1 E3 E0
round 3: E2 E5 E0 E3 E1 E4
```

Every round contains every case exactly once. `round` and `run_order` are in
the run metadata, manifest, and `runs.tsv`. Formal `RUNS=3` refuses a dirty
worktree unless `ALLOW_DIRTY=1`; an override produces a prominent warning in
`summary.md`.

## Output and manifest

The default root is
`/root/oscomp/kv_logs/kv_final_controlled_e0_e5_<UTC timestamp>/` and is never
reused. It contains `manifest.txt`, `manifest.json`, `runs.tsv`, `summary.tsv`,
`summary.md`, and one directory per invocation. Each run directory contains
the effective command and environment, stdout, stderr, exit code, run metadata,
Seq0/Seq1 text and SHA256, process/cgroup/backing-file samples, extracted
metrics, and the final result.

The manifest records repository path, branch, HEAD, upstream, porcelain
status, diff stat, dirty override, binary path/size/SHA256, model
path/size/SHA256, runner path/size/SHA256, parser path/size/SHA256, protocol
path/size/SHA256, prompt/trace source, every planned command and KV
environment, hostname/date/kernel/OS, discoverable CMake compiler/build data,
CPU model/logical CPUs/NUMA, total memory, cgroup version/path/current/max,
swap directory/filesystem/free space, run count, timeout, sample interval, and
the complete round plan. Unavailable host fields are `NA`.

## Observable acceptance rules

All cases first require exit 0 and exact Seq0/Seq1 equality to E0 in the same
round. The safety fields `paged_swapped_active_visible_violation`,
`paged_swapped_active_visible_violation_rows`,
`paged_swapped_active_visible_violation_blocks`,
`paged_active_row_nonresident_fatal`, `paged_row_mapping_invalid_fatal`,
`paged_write_mapping_invalid_fatal`, `paged_input_setup_fatal`, and
`paged_write_to_swapped_block` must all be zero, as must backend/I/O failure
counters. A missing required field is `UNVERIFIED`, not a presumed pass. A log
saying that a requested path was disabled or fell back makes its mechanism
check fail.

| Case | Additional mechanism checks |
|---|---|
| E0 | no paged stats emitted; lazy and swap disabled; active prefetch and madvise zero |
| E1 | lazy-clear and lazy-tail enabled stats both emitted; paged/swap/prefetch zero |
| E2 | paged stats emitted; in-graph gather layers greater than zero; nonidentity support enabled; row mapping/input fatal fields, swap-in/out, madvise, and active prefetch zero; remap rows may be zero |
| E3 | paged swap and idle swap report enabled; nonidentity remap rows, swap-out blocks/calls, and backing writes greater than zero; madvise and active prefetch zero; synchronous resume swap-in is allowed |
| E4 | E3 remap/swap checks plus madvise calls/bytes greater than zero; active prefetch zero |
| E5 | E4 checks plus active prefetch calls/restored blocks greater than zero, auto delayed started, fallback zero, and defer requested |

The driver always makes zero-block `llama_memory_prefetch_seq_step(..., 0)`
probes in paged cases, even when prefetch restoration is off. Therefore E2-E4
require zero active-prefetch calls and zero restored blocks, not a zero
low-level `paged_prefetch_seq_calls` probe counter. Both values are retained in
`runs.tsv` for audit.

`runs.tsv` keeps safety and restoration work separate:

- `active_visible_safety_violations` and its row/block detail map only to the
  three `paged_swapped_active_visible_violation*` fields and are correctness
  failures when nonzero;
- `active_restore_required_rows` and `active_restore_required_blocks` preserve
  the raw sources `paged_swapped_active_violation_rows/blocks`; they count
  resume-time active-required rows/blocks that remain SWAPPED and need
  synchronous restoration;
- `fallback_blocks` is the driver's resume-pending fallback workload.

For this workload E3/E4 have no prefetch, so 272 restore-required rows, 17
restore-required blocks, and 17 fallback blocks are expected. E5 prefetches 17
blocks and reaches zero restore-required/fallback blocks. These restore-work
counters are not data-safety errors.

## Metrics and reliable `NA` values

The parser retains every invocation, then reports per-case raw values,
median/min/max, and relative-to-E0 percentage where the E0 denominator exists
and is nonzero. The Markdown report emphasizes medians.

Current source output supplies active token avg/p50/p95/p99/max and decode
average, active/resume/total wall times, TPS, driver RSS phase points, paged
state/correctness counters, prefetch/fallback counters, and paged I/O totals and
phase timings. `/proc` sampling supplies VmRSS and VmHWM. The sampler attempts
to stat the backing descriptor through `/proc/<pid>/fd`, including O_TMPFILE
files, to obtain logical size and allocated 512-byte blocks.

The following are deliberately `NA` when running this protocol:

- KV resident/nonresident bytes, because mincore is disabled for the clean
  controlled comparison;
- a reliable per-run cgroup `memory.peak`, because the script does not reset a
  possibly shared cgroup peak; sampled `memory.current` is retained separately;
- `pending`, because that aggregate is emitted by stability mode, which this
  workload does not enable;
- backing logical/allocated size if the short-lived anonymous backing fd is
  not observed between samples;
- compiler/build fields when no readable `CMakeCache.txt` accompanies the
  selected binary.

Process VmRSS, process VmHWM, and cgroup `memory.current` have different
accounting scopes. mmap-backed shared file pages can appear in process RSS even
when their page-cache charge is not charged to the current cgroup. Process RSS
and cgroup current must not be directly compared or subtracted; E0-E5
comparisons are made only within the same metric. The script never substitutes
a single max RSS for current-RSS savings. `rss_before_*`/`rss_after_*`, sampled
VmRSS maximum, VmHWM, and cgroup samples remain separate fields.

The performance protocol keeps mincore off. KV resident-page attribution is
deferred to a separate diagnostic pass rather than inferred from these memory
accounting scopes.

`LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0` is recorded to keep the intended
configuration explicit, but the current `idle-swap-resume.cpp` does not read
that variable. Final-sync state is consequently not claimed as independently
verified.

## Commands

Functional smoke:

```bash
RUNS=1 scripts/kv-final-controlled-e0-e5.sh
```

Formal interleaved run (requires a clean worktree):

```bash
RUNS=3 scripts/kv-final-controlled-e0-e5.sh
```

Overrides:

```bash
RUNS=3 OUTPUT_ROOT=/path/to/new/output MODEL=/path/model.gguf \
  BINARY=/path/to/llama-kv-idle-swap-resume \
  scripts/kv-final-controlled-e0-e5.sh
```

Planning-only validation that does not start the model:

```bash
DRY_RUN=1 RUNS=1 OUTPUT_ROOT=/path/to/new/dry-run \
  scripts/kv-final-controlled-e0-e5.sh
```

Reparse an existing output directory without starting the model or modifying
its raw run artifacts:

```bash
python3 scripts/parse-kv-final-controlled-e0-e5.py \
  /root/oscomp/kv_logs/kv_final_controlled_e0_e5_smoke_20260715_044948
```

The runner never builds or creates a build directory. Without `BINARY`, it
tries `build/bin`, `build-release/bin`, then `build-Release/bin` and fails with
those expected paths if none is executable.
