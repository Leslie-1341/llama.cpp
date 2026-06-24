#!/usr/bin/env bash
#
# Stage 8A-3b: fair (low-overhead) performance matrix, 3-run median.
#
# This runner repeats the Stage 8A-3a fair performance P0-P4 policies three
# times and summarizes both per-run values and per-case medians. It does not use
# `set -e`: a single failing run must not abort the rest of the matrix.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-idle-swap-resume}"
OUT_DIR="${OUT_DIR:-/root/oscomp/kv_logs/stage8a_fair_perf_3run}"
PARSER="${PARSER:-scripts/parse_stage8a_fair_perf_3run.py}"
RUNS="${RUNS:-3}"

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
    local case_name="$1"
    local run_id="$2"
    shift 2
    local case_env=("$@")

    local out="$OUT_DIR/${case_name}_run${run_id}.out"
    local err="$OUT_DIR/${case_name}_run${run_id}.err"
    local exit_file="$OUT_DIR/${case_name}_run${run_id}.exit"

    echo "===== ${case_name} run${run_id} ====="

    env "${WORKLOAD_ENV[@]}" "${PERF_OFF_ENV[@]}" "${case_env[@]}" \
        "$BIN" "${COMMON_ARGS[@]}" >"$out" 2>"$err"
    local rc=$?
    echo "$rc" >"$exit_file"
    echo "${case_name} run${run_id} exit=${rc}"
}

run_case() {
    local case_name="$1"
    shift
    local case_env=("$@")

    local run_id
    for run_id in $(seq 1 "$RUNS"); do
        run_one "$case_name" "$run_id" "${case_env[@]}"
    done
}

# ---------------------------------------------------------------------------
# P0: original paged-off baseline (latency lower bound, no paged framework).
# ---------------------------------------------------------------------------
run_case P0_paged_off \
    LLAMA_KV_PAGED=0 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0

# ---------------------------------------------------------------------------
# P1: paged-on, swap-off, minimal overhead (paged framework cost only).
# ---------------------------------------------------------------------------
run_case P1_paged_on_swap_off_min \
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
run_case P2_swap_only_min \
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
run_case P3_prefetch_g2_min \
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
run_case P4_prefetch_g3_min \
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

echo
echo "===== parse ====="
python3 "$PARSER" "$OUT_DIR"

echo
echo "summary_all.tsv: $OUT_DIR/summary_all.tsv"
echo "summary_median.tsv: $OUT_DIR/summary_median.tsv"
