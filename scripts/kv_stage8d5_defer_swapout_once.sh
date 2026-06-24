#!/usr/bin/env bash
#
# Stage 8D-5: resume first-token idle swap-out defer probe.
#
# Runs a once-only matrix with resume step timing enabled. No `set -e`: failed
# cases should not stop the remaining probes.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-idle-swap-resume}"
OUT_DIR="${OUT_DIR:-/root/oscomp/kv_logs/stage8d5_defer_swapout_once}"

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

TIMING_ENV=(
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0
    LLAMA_KV_PAGED_MINCORE=0
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0
    LLAMA_KV_CACHE_DEBUG=0
    LLAMA_KV_PAGED_RESUME_TIMING=1
    LLAMA_KV_PAGED_RESUME_TIMING_STEP=1
)

PAGED_ENV=(
    LLAMA_KV_PAGED=1
    LLAMA_KV_PAGED_INGRAPH=1
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0
)

run_case() {
    local case_name="$1"
    shift
    local case_env=("$@")

    local out="$OUT_DIR/${case_name}.out"
    local err="$OUT_DIR/${case_name}.err"
    local exit_file="$OUT_DIR/${case_name}.exit"

    echo "===== ${case_name} ====="

    env "${WORKLOAD_ENV[@]}" "${TIMING_ENV[@]}" "${PAGED_ENV[@]}" "${case_env[@]}" \
        "$BIN" "${COMMON_ARGS[@]}" >"$out" 2>"$err"
    local rc=$?
    echo "$rc" >"$exit_file"
    echo "${case_name} exit=${rc}"
}

run_case E0_paged_swap_off \
    LLAMA_KV_PAGED_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_TRACE=0 \
    LLAMA_KV_PAGED_IDLE_SWAP=0 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0 \
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 \
    LLAMA_KV_PAGED_RESUME_PREFETCH=0 \
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0

run_case E5_full_prefetch_no_defer \
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

run_case E6_full_prefetch_defer \
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

run_case E7_full_prefetch_earlier_defer \
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
echo "===== resume markers ====="
grep -a "KV_RESUME_FIRST" "$OUT_DIR"/*.err || true

echo
echo "===== step timing around resume ====="
for f in "$OUT_DIR"/*.err; do
    echo
    echo "===== $(basename "$f") ====="
    awk '
        /KV_RESUME_FIRST_BEGIN/ {p=1; c=0; print; next}
        p && c < 8 {print; c++}
        /KV_RESUME_FIRST_END/ {p=0}
    ' "$f"
done

echo
echo "===== final timing ====="
grep -a "KV_PAGED_TIMING" "$OUT_DIR"/*.err || true
