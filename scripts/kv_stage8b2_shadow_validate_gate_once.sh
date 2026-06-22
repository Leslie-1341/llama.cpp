#!/usr/bin/env bash
#
# Stage 8B-2: one-run shadow validation gate probe.
#
# This runner compares paged bookkeeping / gather paths with
# LLAMA_KV_PAGED_SHADOW_VALIDATE on vs off. It is intentionally a runner only:
# build and benchmark execution are left to the caller.

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BIN="${BIN:-build/bin/llama-kv-idle-swap-resume}"
OUT_DIR="${OUT_DIR:-/root/oscomp/kv_logs/stage8b2_shadow_validate_gate_once}"
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

PERF_OFF_ENV=(
    LLAMA_KV_PAGED_MINCORE=0
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0
    LLAMA_KV_CACHE_DEBUG=0
)

BASE_OFF_ENV=(
    LLAMA_KV_PAGED_SWAP=0
    LLAMA_KV_PAGED_IDLE_TRACE=0
    LLAMA_KV_PAGED_IDLE_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_RESUME_PREFETCH=0
)

run_one() {
    local case_name="$1"
    shift
    local case_env=("$@")

    local out="$OUT_DIR/${case_name}.out"
    local err="$OUT_DIR/${case_name}.err"
    local exit_file="$OUT_DIR/${case_name}.exit"

    echo "===== ${case_name} ====="

    env "${WORKLOAD_ENV[@]}" "${PERF_OFF_ENV[@]}" "${BASE_OFF_ENV[@]}" "${case_env[@]}" \
        "$BIN" "${COMMON_ARGS[@]}" >"$out" 2>"$err"
    local rc=$?
    echo "$rc" >"$exit_file"
    echo "${case_name} exit=${rc}"
}

run_one S0_paged_off \
    LLAMA_KV_PAGED=0 \
    LLAMA_KV_PAGED_INGRAPH=0 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0

run_one S1_paged_bookkeeping_shadow_on \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=0 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
    LLAMA_KV_PAGED_SHADOW_VALIDATE=1

run_one S2_paged_bookkeeping_shadow_off \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=0 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0

run_one S3_paged_ingraph_shadow_off \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0

run_one S4_paged_ingraph_gather_shadow_off \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0

run_one S5_prefetch_full_shadow_off \
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0 \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
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
echo "summary.tsv: $OUT_DIR/summary.tsv"
