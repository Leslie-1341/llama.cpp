#!/usr/bin/env bash
#
# Stage 9-B2: semi-real multi-session S0-S5 single-run matrix.
#
# This runner intentionally does not build. It only checks that the requested
# binary exists and preserves raw logs even if a case or the parser fails.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOGDIR="${1:-/root/oscomp/kv_logs/stage9b2_semi_real_matrix}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-semi-real-multisession}"
PARSER="scripts/parse_stage9b2_semi_real_matrix.py"

mkdir -p "$LOGDIR"

if [[ ! -x "$BIN" ]]; then
    echo "error: binary not found or not executable: $BIN" >&2
    exit 2
fi

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

DEBUG_ENV=(
    LLAMA_KV_PAGED_MINCORE=0
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0
)

RESET_ENV_NAMES=(
    LLAMA_KV_PAGED
    LLAMA_KV_PAGED_INGRAPH
    LLAMA_KV_PAGED_GATHER_NONIDENTITY
    LLAMA_KV_PAGED_SWAP
    LLAMA_KV_PAGED_IDLE_SWAP
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE
    LLAMA_KV_PAGED_IDLE_TRACE
    LLAMA_KV_PAGED_MINCORE
    LLAMA_KV_PAGED_TRACE
    LLAMA_KV_PAGED_REFAULT_TRACE
    LLAMA_KV_PAGED_SHADOW_VALIDATE
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED
    LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP
    LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS
    LLAMA_KV_PAGED_RESUME_PENDING_TOKEN
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME
    LLAMA_KV_LAZY_TAIL
    LLAMA_KV_LAZY_CLEAR
)

run_case() {
    local case_name="$1"
    shift
    local case_env=("$@")

    local out="$LOGDIR/${case_name}.out"
    local err="$LOGDIR/${case_name}.err"
    local exit_file="$LOGDIR/${case_name}.exit"

    echo "===== ${case_name} ====="

    (
        for name in "${RESET_ENV_NAMES[@]}"; do
            unset "$name"
        done
        env "${DEBUG_ENV[@]}" "${case_env[@]}" "$BIN" "${COMMON_ARGS[@]}"
    ) >"$out" 2>"$err"
    local rc=$?
    printf "%s\n" "$rc" >"$exit_file"
    echo "${case_name} exit=${rc}"
}

PAGED_OFF_ENV=(
    LLAMA_KV_PAGED=0
    LLAMA_KV_PAGED_INGRAPH=0
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0
    LLAMA_KV_PAGED_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
    LLAMA_KV_LAZY_TAIL=0
    LLAMA_KV_LAZY_CLEAR=0
)

PAGED_ON_ENV=(
    LLAMA_KV_PAGED=1
    LLAMA_KV_PAGED_INGRAPH=1
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
)

PAGED_RECLAIM_OFF_ENV=(
    "${PAGED_ON_ENV[@]}"
    LLAMA_KV_PAGED_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
    LLAMA_KV_LAZY_TAIL=0
    LLAMA_KV_LAZY_CLEAR=0
)

TAIL_LAZY_ENV=(
    LLAMA_KV_PAGED=0
    LLAMA_KV_PAGED_INGRAPH=0
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0
    LLAMA_KV_PAGED_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
    LLAMA_KV_LAZY_TAIL=1
    LLAMA_KV_LAZY_CLEAR=1
)

IDLE_SWAP_ENV=(
    "${PAGED_ON_ENV[@]}"
    LLAMA_KV_PAGED_SWAP=1
    LLAMA_KV_PAGED_IDLE_SWAP=1
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
    LLAMA_KV_LAZY_TAIL=0
    LLAMA_KV_LAZY_CLEAR=0
)

IDLE_PREFETCH_DEFER_ENV=(
    "${PAGED_ON_ENV[@]}"
    LLAMA_KV_PAGED_SWAP=1
    LLAMA_KV_PAGED_IDLE_SWAP=1
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=4
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=4
    LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
    LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
    LLAMA_KV_LAZY_TAIL=0
    LLAMA_KV_LAZY_CLEAR=0
)

run_case S0_baseline_paged_off \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    "${PAGED_OFF_ENV[@]}"

run_case S1_paged_on_reclaim_off \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    "${PAGED_RECLAIM_OFF_ENV[@]}"

run_case S2_tail_lazy_only \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    "${TAIL_LAZY_ENV[@]}"

run_case S3_idle_swap_only \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    "${IDLE_SWAP_ENV[@]}"

run_case S4_idle_swap_prefetch_defer \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    "${IDLE_PREFETCH_DEFER_ENV[@]}"

run_case S5_tail_idle_prefetch_defer \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    "${IDLE_PREFETCH_DEFER_ENV[@]}" \
    LLAMA_KV_LAZY_TAIL=1 \
    LLAMA_KV_LAZY_CLEAR=1

if [[ -f "$PARSER" ]]; then
    if ! python3 "$PARSER" "$LOGDIR"; then
        echo "warning: parser failed; raw logs kept in $LOGDIR" >&2
    fi
else
    echo "warning: parser not found: $PARSER; raw logs kept in $LOGDIR" >&2
fi
