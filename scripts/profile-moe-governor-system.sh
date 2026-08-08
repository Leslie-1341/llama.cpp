#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="${SERVER:-$ROOT/build/bin/llama-server}"
MODEL="${MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}"
SIDECAR="${SIDECAR:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.mwq-v2-hier-legacy.sidecar}"
OUT_DIR="${OUT_DIR:-$ROOT/rss-stage-results/moe-governor-system-$(date +%Y%m%d-%H%M%S)}"
LIMIT="${LIMIT:-2560M}"
MEMORY_HIGH="${MEMORY_HIGH:-2300M}"
THREADS="${THREADS:-6}"
CTX="${CTX:-2048}"
PARALLEL="${PARALLEL:-4}"
REQUESTS="${REQUESTS:-4}"
MAX_TOKENS="${MAX_TOKENS:-48}"
SERVER_TIMEOUT_SEC="${SERVER_TIMEOUT_SEC:-900}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-180}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-900}"
BASE_PORT="${BASE_PORT:-39400}"
CASES="${CASES:-static768,gov_static768,gov_optimal}"
EXTRA_ENV="${EXTRA_ENV:-}"

usage() {
    cat <<EOF
Usage:
  scripts/profile-moe-governor-system.sh

Environment:
  SERVER=FILE      default: $SERVER
  MODEL=FILE       default: $MODEL
  SIDECAR=FILE     default: $SIDECAR
  OUT_DIR=DIR      output directory
  LIMIT=2560M      cgroup memory.max
  MEMORY_HIGH=2300M
  THREADS=6
  CTX=2048
  PARALLEL=4       llama-server -np
  REQUESTS=4       concurrent /v1/completions requests per case
  MAX_TOKENS=48
  SERVER_TIMEOUT_SEC=900
  BASE_PORT=39400
  CASES=static768,gov_static768,gov_optimal,gov_global
  EXTRA_ENV='A=1 B=2'  appended after case env; can override duplicated vars

Outputs:
  summary.tsv
  report.md
  logs/<case>.server.log
  responses/<case>-<n>.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

[[ -x "$SERVER" ]] || { echo "missing server binary: $SERVER" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }
[[ -f "$SIDECAR" ]] || { echo "missing sidecar: $SIDECAR" >&2; exit 2; }
[[ -d /sys/fs/cgroup ]] || { echo "missing cgroup v2 mount: /sys/fs/cgroup" >&2; exit 2; }

mkdir -p "$OUT_DIR/logs" "$OUT_DIR/responses"
SUMMARY="$OUT_DIR/summary.tsv"
REPORT="$OUT_DIR/report.md"
printf 'case\tlimit\tmemory_high\tport\trc\telapsed_ms\trequest_count\tcompletion_tokens\ttok_s\tpeak_mb\tfinal_mb\tmoe_budget_mb\tmoe_action_grow\tmoe_action_shrink\tprefetch_runtime_auto\tkv_soft_enabled\tclean_ranked_enabled\tkv_release_attempts\tkv_offload_attempts\tkv_slot_offload_attempts\treallocation_moe_grants\treallocation_moe_grant_mb\tserver_log\n' > "$SUMMARY"

split_csv() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$value"
}

cgroup_bytes_to_mb() {
    local bytes="$1"
    if [[ ! "$bytes" =~ ^[0-9]+$ ]]; then
        echo 0
        return
    fi
    echo $((bytes / 1024 / 1024))
}

wait_for_server() {
    local port="$1"
    local pid="$2"
    local log="$3"
    local deadline=$((SECONDS + STARTUP_TIMEOUT_SEC))
    while (( SECONDS < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            return 2
        fi
        if grep -q "couldn't bind HTTP server socket\\|exiting due to HTTP server error" "$log" 2>/dev/null; then
            return 3
        fi
        if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    return 1
}

case_env() {
    local name="$1"
    case "$name" in
        static768)
            printf '%s\n' \
                LLAMA_LAZY_V2=1 \
                LLAMA_LAZY_CLG=1 \
                LLAMA_LAZY_MOE_BUFFER=1 \
                LLAMA_LAZY_MOE_BUFFER_MB=768 \
                LLAMA_LAZY_MOE_BUFFER_WORKERS=4 \
                LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0 \
                LLAMA_LAZY_MOE_SIDECAR="$SIDECAR"
            ;;
        gov_static768)
            printf '%s\n' \
                LLAMA_MEMORY_GOVERNOR=1 \
                LLAMA_MEMORY_GOVERNOR_OBSERVE_MS=500 \
                LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM=1 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MB_PER_TICK=64 \
                LLAMA_LAZY_V2=1 \
                LLAMA_LAZY_CLG=1 \
                LLAMA_LAZY_MOE_BUFFER=1 \
                LLAMA_LAZY_MOE_BUFFER_AUTO=0 \
                LLAMA_LAZY_MOE_BUFFER_MB=768 \
                LLAMA_LAZY_MOE_BUFFER_WORKERS=4 \
                LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0 \
                LLAMA_LAZY_MOE_SIDECAR="$SIDECAR"
            ;;
        gov_optimal)
            printf '%s\n' \
                LLAMA_MEMORY_GOVERNOR=1 \
                LLAMA_MEMORY_GOVERNOR_OBSERVE_MS=500 \
                LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM=1 \
                LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_RANKED=1 \
                LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_MAX_PASSES=2 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=1 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MIN_MB_PER_TICK=4 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MAX_MB_PER_TICK=64 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_HEADROOM_MB=128 \
                LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC=1 \
                LLAMA_MEMORY_GOVERNOR_MOE_FAST_START=1 \
                LLAMA_MEMORY_GOVERNOR_MOE_MIN_MB=256 \
                LLAMA_MEMORY_GOVERNOR_MOE_WARM_MB=768 \
                LLAMA_MEMORY_GOVERNOR_MOE_MAX_MB=1536 \
                LLAMA_MEMORY_GOVERNOR_MOE_GROW_MB=64 \
                LLAMA_MEMORY_GOVERNOR_MOE_HEADROOM_MB=256 \
                LLAMA_MEMORY_GOVERNOR_MOE_PRESSURE_SHRINK_PCT=25 \
                LLAMA_MEMORY_GOVERNOR_MOE_PRESSURE_SHRINK_MAX_MB=128 \
                LLAMA_MEMORY_GOVERNOR_MOE_GROW_SAMPLES=2 \
                LLAMA_MEMORY_GOVERNOR_MOE_COOLDOWN_SAMPLES=2 \
                LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=1 \
                LLAMA_MEMORY_GOVERNOR_KV_SOFT_TARGET_MB=0 \
                LLAMA_MEMORY_GOVERNOR_KV_SOFT_IDLE_MB=64 \
                LLAMA_LAZY_V2=1 \
                LLAMA_LAZY_CLG=1 \
                LLAMA_LAZY_MOE_BUFFER=1 \
                LLAMA_LAZY_MOE_BUFFER_MB=768 \
                LLAMA_LAZY_MOE_BUFFER_WORKERS=4 \
                LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0 \
                LLAMA_LAZY_MOE_SIDECAR="$SIDECAR"
            ;;
        gov_global)
            printf '%s\n' \
                LLAMA_MEMORY_GOVERNOR=1 \
                LLAMA_MEMORY_GOVERNOR_OBSERVE_MS=500 \
                LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM=1 \
                LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_RANKED=1 \
                LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_MAX_PASSES=2 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=1 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MIN_MB_PER_TICK=4 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_MAX_MB_PER_TICK=64 \
                LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_HEADROOM_MB=128 \
                LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER=1 \
                LLAMA_MEMORY_GOVERNOR_GLOBAL_MIN_CATCHUP=1 \
                LLAMA_MEMORY_GOVERNOR_HARD_HEADROOM_MB=64 \
                LLAMA_MEMORY_GOVERNOR_GLOBAL_ROI_THRESHOLD=0.05 \
                LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC=1 \
                LLAMA_MEMORY_GOVERNOR_MOE_FAST_START=1 \
                LLAMA_MEMORY_GOVERNOR_MOE_MIN_MB=768 \
                LLAMA_MEMORY_GOVERNOR_MOE_WARM_MB=768 \
                LLAMA_MEMORY_GOVERNOR_MOE_MAX_MB=1536 \
                LLAMA_MEMORY_GOVERNOR_MOE_GROW_MB=64 \
                LLAMA_MEMORY_GOVERNOR_MOE_HEADROOM_MB=128 \
                LLAMA_MEMORY_GOVERNOR_MOE_PRESSURE_SHRINK_PCT=25 \
                LLAMA_MEMORY_GOVERNOR_MOE_PRESSURE_SHRINK_MAX_MB=128 \
                LLAMA_MEMORY_GOVERNOR_MOE_GROW_SAMPLES=2 \
                LLAMA_MEMORY_GOVERNOR_MOE_COOLDOWN_SAMPLES=2 \
                LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=1 \
                LLAMA_MEMORY_GOVERNOR_KV_SOFT_TARGET_MB=0 \
                LLAMA_MEMORY_GOVERNOR_KV_SOFT_IDLE_MB=64 \
                LLAMA_LAZY_V2=1 \
                LLAMA_LAZY_CLG=1 \
                LLAMA_LAZY_MOE_BUFFER=1 \
                LLAMA_LAZY_MOE_BUFFER_MB=768 \
                LLAMA_LAZY_MOE_BUFFER_WORKERS=4 \
                LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0 \
                LLAMA_LAZY_MOE_SIDECAR="$SIDECAR"
            ;;
        *)
            echo "unknown case: $name" >&2
            return 2
            ;;
    esac
}

summarize_case() {
    local name="$1"
    local log="$2"
    local elapsed_ms="$3"
    local rc="$4"
    local port="$5"
    local peak_mb="$6"
    local final_mb="$7"

    local response_stats
    response_stats="$(python3 - "$OUT_DIR/responses" "$name" "$REQUESTS" "$elapsed_ms" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
case = sys.argv[2]
requests = int(sys.argv[3])
elapsed_ms = int(sys.argv[4])
tokens = 0
ok = 0
for i in range(1, requests + 1):
    path = root / f"{case}-{i}.json"
    try:
        data = json.loads(path.read_text())
    except Exception:
        continue
    if isinstance(data, dict) and "error" not in data:
        ok += 1
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    tokens += int(usage.get("completion_tokens") or 0)
tok_s = (tokens * 1000.0 / elapsed_ms) if elapsed_ms > 0 else 0.0
print(f"{ok}\t{tokens}\t{tok_s:.3f}")
PY
)"

    local moe_budget_mb grow shrink runtime_auto kv_soft clean_ranked kv_release kv_offload
    moe_budget_mb="$(grep 'memory_governor_observe' "$log" 2>/dev/null | sed -n 's/.* moe_budget_bytes=\([0-9]*\).*/\1/p' | tail -n 1 || true)"
    moe_budget_mb="$(cgroup_bytes_to_mb "${moe_budget_mb:-0}")"
    grow="$(grep -c 'moe_budget_action=grow' "$log" 2>/dev/null || true)"
    shrink="$(grep -c 'moe_budget_action=shrink' "$log" 2>/dev/null || true)"
    runtime_auto="$(grep 'memory_governor_observe' "$log" 2>/dev/null | sed -n 's/.* prefetch_budget_runtime_auto=\([0-9]*\).*/\1/p' | tail -n 1 || true)"
    kv_soft="$(grep 'memory_governor_observe' "$log" 2>/dev/null | sed -n 's/.* kv_soft_budget_enabled=\([0-9]*\).*/\1/p' | tail -n 1 || true)"
    clean_ranked="$(grep 'memory_governor_observe' "$log" 2>/dev/null | sed -n 's/.* clean_reclaim_ranked_enabled=\([0-9]*\).*/\1/p' | tail -n 1 || true)"
    kv_release="$(grep -c 'kv_release_attempted=1' "$log" 2>/dev/null || true)"
    kv_offload="$(grep -c 'kv_offload_attempted=1' "$log" 2>/dev/null || true)"
    kv_slot_offload="$(grep -c 'kv_offload_attempted=1.*kv_offload_backend=slot_state' "$log" 2>/dev/null || true)"
    realloc_grants="$(grep -c 'reallocation_reason=moe_credit_grant' "$log" 2>/dev/null || true)"
    realloc_grant_mb="$(( ( $(grep 'reallocation_reason=moe_credit_grant' "$log" 2>/dev/null | sed -n 's/.* reallocation_moe_grant_bytes=\([0-9]*\).*/\1/p' | awk '{s+=$1} END{printf "%.0f", s+0}' || true) ) / 1048576 ))"

    printf '%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$LIMIT" "$MEMORY_HIGH" "$port" "$rc" "$elapsed_ms" \
        "$response_stats" "$peak_mb" "$final_mb" "$moe_budget_mb" "$grow" "$shrink" \
        "${runtime_auto:-0}" "${kv_soft:-0}" "${clean_ranked:-0}" \
        "$kv_release" "$kv_offload" "$kv_slot_offload" \
        "$realloc_grants" "${realloc_grant_mb:-0}" "$log" >> "$SUMMARY"
}

run_case() {
    local name="$1"
    local index="$2"
    local port=$((BASE_PORT + index))
    local cg="/sys/fs/cgroup/moe_gov_${name}_$$"
    local log="$OUT_DIR/logs/${name}.server.log"
    local stdout="$OUT_DIR/logs/${name}.stdout"
    local peak_file="$OUT_DIR/logs/${name}.peak"
    local sample_stop="$OUT_DIR/logs/${name}.sample.stop"
    local pid=""
    local curl_pids=()
    local sampler_pid=""
    local peak=0
    local final=0
    local rc=0

    rm -f "$OUT_DIR/responses/${name}-"*.json
    : > "$log"
    : > "$stdout"

    mkdir -p "$cg"
    printf '%s\n' "$LIMIT" > "$cg/memory.max"
    printf '%s\n' "$MEMORY_HIGH" > "$cg/memory.high"
    printf '0\n' > "$cg/memory.swap.max" 2>/dev/null || true

    mapfile -t env_args < <(case_env "$name")
    if [[ -n "$EXTRA_ENV" ]]; then
        # shellcheck disable=SC2206
        extra_env_args=($EXTRA_ENV)
        env_args+=("${extra_env_args[@]}")
    fi
    (
        printf '%s\n' "$BASHPID" > "$cg/cgroup.procs"
        exec env "${env_args[@]}" "$SERVER" \
            -m "$MODEL" \
            -t "$THREADS" -ngl 0 -c "$CTX" -np "$PARALLEL" \
            --host 127.0.0.1 \
            --port "$port" \
            --metrics \
            --no-webui \
            --timeout "$SERVER_TIMEOUT_SEC" \
            --log-file "$log"
    ) >"$stdout" 2>&1 &
    pid=$!
    cleanup_case() {
        if [[ -n "${sampler_pid:-}" ]] && kill -0 "$sampler_pid" 2>/dev/null; then
            printf 'stop\n' > "$sample_stop"
            wait "$sampler_pid" 2>/dev/null || true
        fi
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
        rmdir "$cg" 2>/dev/null || true
    }
    trap cleanup_case RETURN

    set +e
    wait_for_server "$port" "$pid" "$log"
    local ready=$?
    set -e
    if (( ready != 0 )); then
        rc=$((120 + ready))
    else
        local start end
        rm -f "$sample_stop"
        printf '0\n' > "$peak_file"
        (
            local local_peak=0
            while [[ ! -f "$sample_stop" ]]; do
                local cur
                cur="$(cat "$cg/memory.current" 2>/dev/null || echo 0)"
                if [[ "$cur" =~ ^[0-9]+$ && "$cur" -gt "$local_peak" ]]; then
                    local_peak="$cur"
                    printf '%s\n' "$local_peak" > "$peak_file"
                fi
                sleep 0.2
            done
        ) &
        sampler_pid=$!
        start="$(date +%s%3N)"
        for i in $(seq 1 "$REQUESTS"); do
            curl --noproxy '*' -sS --max-time "$REQUEST_TIMEOUT_SEC" \
                "http://127.0.0.1:${port}/v1/completions" \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"default\",\"prompt\":\"Request $i: Explain briefly why global memory allocation changes MoE expert scheduling under cgroup pressure.\",\"max_tokens\":$MAX_TOKENS,\"temperature\":0}" \
                > "$OUT_DIR/responses/${name}-${i}.json" &
            curl_pids+=("$!")
        done
        for curl_pid in "${curl_pids[@]}"; do
            wait "$curl_pid" || rc=$?
        done
        end="$(date +%s%3N)"
        elapsed_ms=$((end - start))
        printf 'stop\n' > "$sample_stop"
        wait "$sampler_pid" 2>/dev/null || true
        sampler_pid=""
        peak="$(cat "$peak_file" 2>/dev/null || echo 0)"
    fi

    local cur
    cur="$(cat "$cg/memory.current" 2>/dev/null || echo 0)"
    if [[ "$cur" =~ ^[0-9]+$ ]]; then
        final="$(cgroup_bytes_to_mb "$cur")"
        if (( cur > peak )); then
            peak="$cur"
        fi
    fi
    peak="$(cgroup_bytes_to_mb "$peak")"

    summarize_case "$name" "$log" "${elapsed_ms:-0}" "$rc" "$port" "$peak" "$final"
    trap - RETURN
    cleanup_case
}

cases=()
split_csv "$CASES" cases
index=0
for case_name in "${cases[@]}"; do
    echo "==> case=$case_name limit=$LIMIT high=$MEMORY_HIGH"
    run_case "$case_name" "$index"
    index=$((index + 1))
done

{
    echo "# MoE Governor System Profile"
    echo
    echo "- model: \`$MODEL\`"
    echo "- sidecar: \`$SIDECAR\`"
    echo "- limit: \`$LIMIT\`, memory.high: \`$MEMORY_HIGH\`"
    echo "- requests: \`$REQUESTS\`, max_tokens: \`$MAX_TOKENS\`, parallel slots: \`$PARALLEL\`"
    echo
    echo '```tsv'
    column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
    echo '```'
} > "$REPORT"

echo "summary: $SUMMARY"
echo "report:  $REPORT"
