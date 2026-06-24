#!/usr/bin/env bash
#
# Stage 8B-1: paged overhead ablation, 3-run median.
#
# This runner isolates paged bookkeeping, in-graph gather, non-identity row
# remap, idle maintenance, and full swap-only path overhead. It intentionally
# avoids `set -e` so one failing run does not stop the remaining matrix.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-idle-swap-resume}"
OUT_DIR="${OUT_DIR:-/root/oscomp/kv_logs/stage8b1_overhead_3run}"
PARSER="${PARSER:-scripts/parse_stage8b1_overhead_3run.py}"
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

# Fixed benchmark config (same controlled workload as Stage 8A fair perf).
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

WORKLOAD_ENV=(
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
)

# Common low-overhead defaults. Case env below overrides only the route being
# measured.
BASE_ENV=(
    LLAMA_KV_PAGED_MINCORE=0
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0
    LLAMA_KV_CACHE_DEBUG=0
    LLAMA_KV_PAGED_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_RESUME_PREFETCH=0
    LLAMA_KV_PAGED_IDLE_TRACE=0
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

    env "${WORKLOAD_ENV[@]}" "${BASE_ENV[@]}" "${case_env[@]}" \
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

# Q0: original paged-off baseline.
run_case Q0_paged_off \
    LLAMA_KV_PAGED=0 \
    LLAMA_KV_PAGED_INGRAPH=0 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0

# Q1: paged bookkeeping / shadow_validate / note/assert overhead only.
run_case Q1_paged_bookkeeping_only \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=0 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0

# Q2: Q1 + in-graph gather route.
run_case Q2_paged_ingraph_only \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0

# Q4: Q2 + non-identity row remap / cold-block ownership scan.
run_case Q4_paged_ingraph_gather_nonidentity \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0

# Q5: Q4 + idle maintenance loop, but no real swap-out.
run_case Q5_paged_idle_trace_no_swap \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0

# Q6: real swap-only path with idle swap and MADV_DONTNEED.
run_case Q6_swap_only_full_path \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0

echo
echo "===== parse ====="
python3 "$PARSER" "$OUT_DIR"

echo
echo "summary_all.tsv: $OUT_DIR/summary_all.tsv"
echo "summary_median.tsv: $OUT_DIR/summary_median.tsv"
