#!/usr/bin/env bash
#
# Stage 8E-B: KV footprint / residency audit.
#
# Diagnostic memory-accounting probe. Reuses the Stage 8D-5 workload, but forces
# LLAMA_KV_PAGED_MINCORE=1 so the full KV capacity / residency accounting is
# emitted (Stage 8D-5 clean perf ran with MINCORE=0 to avoid the sampling cost).
# This adds NO new optimization; it only measures.
#
# Case matrix (both single-run smoke, no 3-run):
#   B0_paged_swap_off_mincore       <-> Stage 8D-5 F0 (paged on, swap off)
#   B1_full_prefetch_defer_mincore  <-> Stage 8D-5 F2 (swap+madvise, full
#                                       prefetch, resume defer)
#
# No `set -e`: a failing case must not stop the remaining probe or the parser.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-idle-swap-resume}"
OUT_DIR="${OUT_DIR:-/root/oscomp/kv_logs/stage8e_kv_footprint_audit}"
PARSER="${PARSER:-$ROOT_DIR/scripts/parse_stage8e_kv_footprint_audit.py}"

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

# Forced diagnostic accounting: MINCORE on, all timing / trace / shadow
# validation OFF so the audit reflects the same code path as Stage 8D-5 clean
# perf, plus residency sampling only.
DIAG_ENV=(
    LLAMA_KV_PAGED_MINCORE=1
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0
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
)

run_case() {
    local case_name="$1"
    shift
    local case_env=("$@")

    local out="$OUT_DIR/${case_name}.out"
    local err="$OUT_DIR/${case_name}.err"
    local exit_file="$OUT_DIR/${case_name}.exit"

    echo "===== ${case_name} ====="

    env "${WORKLOAD_ENV[@]}" "${DIAG_ENV[@]}" "${PAGED_ENV[@]}" "${case_env[@]}" \
        "$BIN" "${COMMON_ARGS[@]}" >"$out" 2>"$err"
    local rc=$?
    echo "$rc" >"$exit_file"
    echo "${case_name} exit=${rc}"
}

# B0 <-> Stage 8D-5 F0: paged on, swap off, no madvise / prefetch / defer.
run_case B0_paged_swap_off_mincore \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0

# B1 <-> Stage 8D-5 F2: swap + idle madvise, full prefetch, resume defer.
run_case B1_full_prefetch_defer_mincore \
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
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1

echo
echo "===== running parser ====="
python3 "$PARSER" "$OUT_DIR"
