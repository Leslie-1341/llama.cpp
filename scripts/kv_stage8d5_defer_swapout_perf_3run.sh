#!/usr/bin/env bash
#
# Stage 8D-5: defer idle swap-out on resume, formal 3-run perf matrix.
#
# This runner intentionally disables timing probes and heavy measurement logs.
# It keeps idle trace enabled only for swap/madvise cases because the idle
# swap-out maintenance loop depends on that functional switch.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-idle-swap-resume}"
OUT_DIR="${OUT_DIR:-/root/oscomp/kv_logs/stage8d5_defer_swapout_perf_3run}"
PARSER="${PARSER:-scripts/parse_stage8d5_defer_swapout_perf_3run.py}"
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

COMMON_ARGS=(
    -m "$MODEL"
    --n-predict 128
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

WORKLOAD_ENV=(
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
)

PERF_ENV=(
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0
    LLAMA_KV_PAGED_MINCORE=0
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0
    LLAMA_KV_CACHE_DEBUG=0
    LLAMA_KV_PAGED_RESUME_TIMING=0
    LLAMA_KV_PAGED_RESUME_TIMING_STEP=0
)

PAGED_ENV=(
    LLAMA_KV_PAGED=1
    LLAMA_KV_PAGED_INGRAPH=1
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0
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

    env "${WORKLOAD_ENV[@]}" "${PERF_ENV[@]}" "${PAGED_ENV[@]}" "${case_env[@]}" \
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

# F0 mirrors E0 without timing probes.
run_case F0_paged_swap_off \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0 \
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0

# F1 mirrors E5 without timing probes.
run_case F1_full_prefetch_no_defer \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=1 \
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off \
    LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=4 \
    LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=4 \
    LLAMA_KV_PAGED_PREFETCH_SAFETY_TOKENS=0 \
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0

# F2 mirrors E6 without timing probes.
run_case F2_full_prefetch_defer \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=1 \
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off \
    LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=4 \
    LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=4 \
    LLAMA_KV_PAGED_PREFETCH_SAFETY_TOKENS=0 \
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1

# F3 mirrors E7 without timing probes.
run_case F3_full_prefetch_earlier_defer \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=1 \
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off \
    LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS=0 \
    LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=4 \
    LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=4 \
    LLAMA_KV_PAGED_PREFETCH_SAFETY_TOKENS=0 \
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1

echo
echo "===== parse ====="
python3 "$PARSER" "$OUT_DIR"

echo
echo "summary_all.tsv: $OUT_DIR/summary_all.tsv"
echo "summary_median.tsv: $OUT_DIR/summary_median.tsv"
