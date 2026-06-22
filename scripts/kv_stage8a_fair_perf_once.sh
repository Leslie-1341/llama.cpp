#!/usr/bin/env bash
#
# Stage 8A-3a: fair (low-overhead) performance matrix.
#
# Purpose
# -------
# Stages 8A-1 / 8A-2 measured swap / prefetch tradeoffs with measurement
# machinery ON (LLAMA_KV_PAGED_MINCORE=1, LLAMA_KV_PAGED_TRACE=1, refault
# tracing). Full-prefetch resume_first_ms (~116ms) was still well above the
# paged-off baseline (~81ms). This runner re-measures the SAME policies with
# measurement/logging machinery OFF so we can separate true scheme cost from
# measurement overhead. It only reports performance fields; mincore / KV
# nonresident are intentionally not trusted here (mincore is disabled).
#
# Env var classification (READ THIS before editing cases)
# -------------------------------------------------------
# FUNCTIONAL (changing these changes behavior; required for the feature):
#   LLAMA_KV_PAGED                       - master paged-KV switch
#   LLAMA_KV_PAGED_INGRAPH               - in-graph gather path
#   LLAMA_KV_PAGED_GATHER_NONIDENTITY    - non-identity remap (swap-out needs it)
#   LLAMA_KV_PAGED_SWAP                  - enable swap backing store
#   LLAMA_KV_PAGED_IDLE_SWAP             - enable idle swap-out gate
#   LLAMA_KV_PAGED_IDLE_SWAP_MADVISE     - actually MADV_DONTNEED swapped pages
#   LLAMA_KV_PAGED_IDLE_TRACE            - *** FUNCTIONAL, not logging ***
#                                          The whole idle swap-out maintenance
#                                          loop (incl. paged_swap_out_block) is
#                                          gated by `if (paged_idle_trace_enabled)`
#                                          in src/llama-kv-cache.cpp (~L5078).
#                                          With it unset, NO swap-out happens, so
#                                          P2/P3/P4 must keep it =1. It does emit a
#                                          per-maintenance-step stderr line (small
#                                          unavoidable overhead until the gate is
#                                          decoupled from the trace flag).
#   LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE  - interleave prefetch during active decode
#   LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED   - auto-delayed prefetch scheduling
#   LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS / _BLOCKS_PER_STEP / _SAFETY_TOKENS
#                                          - prefetch granularity knobs
#   LLAMA_KV_PAGED_RESUME_PENDING_TOKEN    - when to start resume-pending prefetch
#   LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE  - prefetch target clamp (off/low/med/high)
#   LLAMA_KV_PAGED_RESUME_PREFETCH         - one-shot full prefetch right before resume
#   LLAMA_KV_IDLE_NUM_IDLE_SEQS            - number of idle seqs constructed
#   LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS       - seq0 warmup decode length
#
# MEASUREMENT / LOGGING ONLY (off in perf mode; off does NOT break swap/prefetch):
#   LLAMA_KV_PAGED_MINCORE        - mincore residency sampling (KV resident bytes)
#   LLAMA_KV_PAGED_TRACE          - verbose per-op paged trace
#   LLAMA_KV_PAGED_REFAULT_TRACE  - SIGSEGV-based refault source tracing
#   LLAMA_KV_CACHE_DEBUG          - end-of-run library counter dump
#
# This runner does NOT use `set -e`: a single failing case must not abort the
# rest of the matrix. Each case writes ${case}.out / ${case}.err / ${case}.exit.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-idle-swap-resume}"
OUT_DIR="${OUT_DIR:-/root/oscomp/kv_logs/stage8a_fair_perf_once}"
PARSER="${PARSER:-scripts/parse_stage8a_policy_matrix.py}"

mkdir -p "$OUT_DIR"

if [[ ! -f "$MODEL" ]]; then
    echo "error: model not found: $MODEL" >&2
    exit 2
fi
if [[ ! -x "$BIN" ]]; then
    echo "error: binary not found or not executable: $BIN" >&2
    echo "hint: cmake --build build --target llama-kv-idle-swap-resume -j\$(nproc)" >&2
    exit 2
fi

# Fixed benchmark config (Stage 8A controlled workload).
COMMON_ARGS=(
    -m "$MODEL"
    -n 128
    --ctx-size 2048
    --batch-size 128
    --ubatch-size 128
    --seed 1
    --temp 0
    --cache-type-k f32
    --cache-type-v f32
    --kv-unified
    --parallel 4
)

# Workload shape (functional): construct 2 idle seqs + 256-token seq0 warmup.
WORKLOAD_ENV=(
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
)

# Measurement/logging machinery: OFF for every case in this fair-perf runner.
PERF_OFF_ENV=(
    LLAMA_KV_PAGED_MINCORE=0
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0
)

run_one() {
    local case_name="$1"; shift
    local case_env=("$@")

    local out="$OUT_DIR/${case_name}.out"
    local err="$OUT_DIR/${case_name}.err"
    local exit_file="$OUT_DIR/${case_name}.exit"

    echo "===== ${case_name} ====="

    env "${WORKLOAD_ENV[@]}" "${PERF_OFF_ENV[@]}" "${case_env[@]}" \
        "$BIN" "${COMMON_ARGS[@]}" >"$out" 2>"$err"
    local rc=$?
    echo "$rc" >"$exit_file"
    echo "${case_name} exit=${rc}"
}

# ---------------------------------------------------------------------------
# P0: original paged-off baseline (latency lower bound, no paged framework).
# ---------------------------------------------------------------------------
run_one P0_paged_off \
    LLAMA_KV_PAGED=0 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0

# ---------------------------------------------------------------------------
# P1: paged-on, swap-off, minimal overhead (paged framework cost only).
# ---------------------------------------------------------------------------
run_one P1_paged_on_swap_off_min \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0

# ---------------------------------------------------------------------------
# P2: swap-only, minimal overhead. IDLE_TRACE=1 is FUNCTIONAL here (gates the
# swap-out loop); it is not enabled for logging.
# ---------------------------------------------------------------------------
run_one P2_swap_only_min \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0

# ---------------------------------------------------------------------------
# P3: full-prefetch granularity G2 (every=2, blocks_per_step=1), pressure=off.
# Based on P2 + interleaved auto-delayed prefetch.
# ---------------------------------------------------------------------------
run_one P3_prefetch_g2_min \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=2 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=1 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0 \
    LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96 \
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0

# ---------------------------------------------------------------------------
# P4: full-prefetch granularity G3 (every=4, blocks_per_step=4), pressure=off.
# Based on P2 + more aggressive interleaved auto-delayed prefetch.
# ---------------------------------------------------------------------------
run_one P4_prefetch_g3_min \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=4 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=4 \
    LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0 \
    LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96 \
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0

# ---------------------------------------------------------------------------
# Parse: reuse the Stage 8A policy-matrix parser. It already emits 0 for
# missing mincore fields and does not flag missing mincore as abnormal, so it
# is compatible with this performance (mincore-off) run.
# ---------------------------------------------------------------------------
echo
echo "===== parse ====="
python3 "$PARSER" "$OUT_DIR"

echo
echo "summary.tsv: $OUT_DIR/summary.tsv"
