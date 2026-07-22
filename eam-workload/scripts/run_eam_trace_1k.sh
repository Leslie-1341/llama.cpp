#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/llama.cpp}
MODEL=${MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}
SIDECAR=${SIDECAR:-/tmp/qwen-moe-mwq-v2-hier.sidecar}
INDEX=${INDEX:-$ROOT/eam-workload/processed/eam_workload_1k.index.tsv}
TRACE=${TRACE:-$ROOT/eam-workload/traces/sharegpt_eam_1k.jsonl}
LOG_DIR=${LOG_DIR:-$ROOT/eam-workload/traces/logs}
GEN_TOKENS=${GEN_TOKENS:-64}
THREADS=${THREADS:-8}
BUFFER_MB=${BUFFER_MB:-1024}
WORKERS=${WORKERS:-4}
START=${START:-0}
LIMIT=${LIMIT:-1000}
APPEND=${APPEND:-0}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-300}

mkdir -p "$(dirname "$TRACE")" "$LOG_DIR"
if [[ "$APPEND" != "1" ]]; then
    : > "$TRACE"
fi

cd "$ROOT"

seen=0
ran=0
while IFS=$'\t' read -r workload_index request_id prompt_file bucket chars; do
    if (( workload_index < START )); then
        continue
    fi
    if (( seen >= LIMIT )); then
        break
    fi
    seen=$((seen + 1))
    ran=$((ran + 1))

    log_file="$LOG_DIR/$(printf "%04d" "$workload_index")_${request_id}.log"
    echo "[$ran/$LIMIT] workload=$workload_index request=$request_id bucket=$bucket chars=$chars"

    timeout "$REQUEST_TIMEOUT" env \
        LLAMA_LAZY_V2=1 \
        LLAMA_LAZY_MOE_BUFFER=1 \
        LLAMA_LAZY_MOE_BUFFER_MB="$BUFFER_MB" \
        LLAMA_LAZY_MOE_BUFFER_WORKERS="$WORKERS" \
        LLAMA_LAZY_MOE_SIDECAR="$SIDECAR" \
        LLAMA_LAZY_MOE_DYNBITS=1 \
        LLAMA_LAZY_MOE_DYNBITS_REAL=1 \
        LLAMA_LAZY_MOE_HOT_BITS=2 \
        LLAMA_LAZY_MOE_WARM_BITS=2 \
        LLAMA_LAZY_MOE_COLD_BITS=2 \
        LLAMA_LAZY_MOE_GATE_BITS=2 \
        LLAMA_LAZY_MOE_UP_BITS=2 \
        LLAMA_LAZY_MOE_DOWN_BITS=2 \
        LLAMA_LAZY_MOE_AVX512_Q2=1 \
        LLAMA_LAZY_MOE_AVX512_Q2_DOT=1 \
        LLAMA_LAZY_MOE_EAM_TRACE="$TRACE" \
        LLAMA_LAZY_MOE_EAM_TRACE_REQUEST_ID="$request_id" \
        LLAMA_LAZY_MOE_EAM_TRACE_WORKLOAD_INDEX="$workload_index" \
        LLAMA_LAZY_MOE_EAM_TRACE_PROMPT_FILE="$prompt_file" \
        LLAMA_LAZY_MOE_EAM_TRACE_BUCKET="$bucket" \
        LLAMA_LAZY_MOE_EAM_TRACE_CHARS="$chars" \
        LLAMA_LAZY_MOE_EAM_TRACE_DECODE_TOKENS="$GEN_TOKENS" \
        ./build/bin/llama-cli \
            -m "$MODEL" \
            -f "$prompt_file" \
            -n "$GEN_TOKENS" \
            -t "$THREADS" \
            -ctk q4_0 \
            -ctv q4_0 \
            -fa 1 \
            -no-cnv \
            -st \
            --no-display-prompt \
            --simple-io > "$log_file" 2>&1 || {
                echo "warning: request $request_id failed, see $log_file" >&2
            }
done < <(awk 'NR > 1' "$INDEX")

echo "trace written to $TRACE"
