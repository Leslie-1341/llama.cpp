#!/usr/bin/env bash

DIR=/root/oscomp/kv_logs/stage8a_prefetch_granularity_once
MODEL=/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
BIN=build/bin/llama-kv-idle-swap-resume

mkdir -p "$DIR"

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

run_case() {
  local case_name="$1"
  local every_tokens="$2"
  local blocks_per_step="$3"

  local out="$DIR/${case_name}.out"
  local err="$DIR/${case_name}.err"
  local exit_file="$DIR/${case_name}.exit"

  echo "===== ${case_name} ====="

  (
    unset LLAMA_KV_PAGED
    unset LLAMA_KV_PAGED_INGRAPH
    unset LLAMA_KV_PAGED_GATHER_NONIDENTITY
    unset LLAMA_KV_PAGED_IDLE_TRACE
    unset LLAMA_KV_PAGED_TRACE
    unset LLAMA_KV_PAGED_SWAP
    unset LLAMA_KV_PAGED_IDLE_SWAP
    unset LLAMA_KV_PAGED_IDLE_SWAP_MADVISE
    unset LLAMA_KV_PAGED_MINCORE
    unset LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE
    unset LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED
    unset LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS
    unset LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP
    unset LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS
    unset LLAMA_KV_PAGED_RESUME_PENDING_TOKEN
    unset LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE
    unset LLAMA_KV_PAGED_RESUME_PREFETCH
    unset LLAMA_KV_IDLE_NUM_IDLE_SEQS
    unset LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS

    env \
      LLAMA_KV_IDLE_NUM_IDLE_SEQS=2 \
      LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256 \
      LLAMA_KV_PAGED=1 \
      LLAMA_KV_PAGED_INGRAPH=1 \
      LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
      LLAMA_KV_PAGED_IDLE_TRACE=1 \
      LLAMA_KV_PAGED_TRACE=1 \
      LLAMA_KV_PAGED_SWAP=1 \
      LLAMA_KV_PAGED_IDLE_SWAP=1 \
      LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
      LLAMA_KV_PAGED_MINCORE=1 \
      LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 \
      LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1 \
      LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0 \
      LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96 \
      LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off \
      LLAMA_KV_PAGED_RESUME_PREFETCH=0 \
      LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS="$every_tokens" \
      LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP="$blocks_per_step" \
      "$BIN" "${COMMON_ARGS[@]}"
  ) > "$out" 2> "$err"

  local exit_code=$?
  printf "%s\n" "$exit_code" > "$exit_file"
  echo "${case_name} exit=${exit_code}"
}

run_case G0_off_e4_b1_s0 4 1
run_case G1_off_e4_b2_s0 4 2
run_case G2_off_e2_b1_s0 2 1
run_case G3_off_e4_b4_s0 4 4

python3 scripts/parse_stage8a_policy_matrix.py "$DIR"
