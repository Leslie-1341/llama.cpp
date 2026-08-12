#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="${SERVER:-$ROOT/build/bin/llama-server}"
DENSE_MODEL="${DENSE_MODEL:-/root/models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.ffn-bplus-b.gguf}"
MOE_MODEL="${MOE_MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}"
MOE_SIDECAR="${MOE_SIDECAR:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.mwq-v2-hier-legacy.sidecar}"
OUT_DIR="${OUT_DIR:-$ROOT/rss-stage-results/global-memory-sweep-$(date +%Y%m%d-%H%M%S)}"
MEMORY_MBS="${MEMORY_MBS:-1000 1250 1500 1750 2000 2250 2500 2750 3000}"
HIGH_MARGIN_MB="${HIGH_MARGIN_MB:-0}"
TOKENS="${TOKENS:-16}"
CTX="${CTX:-1024}"
BATCH="${BATCH:-16}"
THREADS="${THREADS:-8}"
BASE_PORT="${BASE_PORT:-41000}"
STARTUP_TIMEOUT_SEC="${STARTUP_TIMEOUT_SEC:-240}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-300}"

mkdir -p "$OUT_DIR"
SUMMARY="$OUT_DIR/summary.tsv"
printf 'model\tmem_mb\thigh_mb\tstatus\tfinish\terror\tpred_tok_s\tpred_ms\tprompt_tok_s\tserver_eval_tok_s\tserver_prompt_tok_s\tlast_state\tpressure_source\tmem_current\tmem_peak\tmax_events\toom\toom_kill\tdense_planner\tdense_ring\tdense_locked\tdense_stream_per_token\tdense_ahead\tdense_decision\tdense_repin_reason\tmoe_enabled\tmoe_budget\tmoe_resident\tmoe_evictions\tmoe_bytes_read\tmoe_prefetch_dropped\tmoe_action\tmoe_reason\tglobal_decision\treallocation_reason\tkv_offload_reason\trun_dir\n' > "$SUMMARY"

need_file() {
    local path="$1"
    [[ -e "$path" ]] || {
        echo "missing: $path" >&2
        exit 2
    }
}

kv() {
    local key="$1"
    local file="$2"
    grep -o "${key}=[^ ]*" "$file" 2>/dev/null | tail -1 | cut -d= -f2-
}

json_num() {
    local key="$1"
    local file="$2"
    grep -o "\"${key}\":[0-9.]*" "$file" 2>/dev/null | tail -1 | cut -d: -f2
}

json_str() {
    local key="$1"
    local file="$2"
    grep -o "\"${key}\":\"[^\"]*\"" "$file" 2>/dev/null | tail -1 | cut -d: -f2- | tr -d '"'
}

server_eval_tok_s() {
    local file="$1"
    grep 'eval time' "$file" 2>/dev/null | grep 'tokens per second' | tail -1 | sed -n 's/.* \([0-9.][0-9.]*\) tokens per second).*/\1/p'
}

server_prompt_tok_s() {
    local file="$1"
    grep 'prompt eval time' "$file" 2>/dev/null | tail -1 | sed -n 's/.* \([0-9.][0-9.]*\) tokens per second).*/\1/p'
}

wait_health() {
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

summarize() {
    local model="$1"
    local mem_mb="$2"
    local high_mb="$3"
    local status="$4"
    local run_dir="$5"
    local log="$run_dir/server.log"
    local out="$run_dir/completion.json"
    local events="$run_dir/memory.events"
    local current="$run_dir/memory.current"
    local flex_line="$run_dir/flex.line"

    grep 'llama_flex: layers=' "$log" 2>/dev/null | tail -1 > "$flex_line"
    local finish error pred_tok_s pred_ms prompt_tok_s eval_tok_s prompt_srv
    finish="$(json_str finish_reason "$out")"; finish="${finish:-NA}"
    error="$(json_str error "$out")"; error="${error:-0}"
    pred_tok_s="$(json_num predicted_per_second "$out")"; pred_tok_s="${pred_tok_s:-NA}"
    pred_ms="$(json_num predicted_ms "$out")"; pred_ms="${pred_ms:-NA}"
    prompt_tok_s="$(json_num prompt_per_second "$out")"; prompt_tok_s="${prompt_tok_s:-NA}"
    eval_tok_s="$(server_eval_tok_s "$log")"; eval_tok_s="${eval_tok_s:-NA}"
    prompt_srv="$(server_prompt_tok_s "$log")"; prompt_srv="${prompt_srv:-NA}"

    local max_events oom oom_kill mem_current mem_peak
    max_events="$(awk '$1=="max"{print $2}' "$events" 2>/dev/null | tail -1)"; max_events="${max_events:-NA}"
    oom="$(awk '$1=="oom"{print $2}' "$events" 2>/dev/null | tail -1)"; oom="${oom:-NA}"
    oom_kill="$(awk '$1=="oom_kill"{print $2}' "$events" 2>/dev/null | tail -1)"; oom_kill="${oom_kill:-NA}"
    mem_current="$(cat "$current" 2>/dev/null)"; mem_current="${mem_current:-NA}"
    mem_peak="$(awk '/memory_current/ {print}' "$run_dir/cgroup-samples.tsv" 2>/dev/null | tail -1 | cut -f2)"; mem_peak="${mem_peak:-NA}"

    local last_obs="$run_dir/last-observe.log"
    grep 'memory_governor_observe' "$log" 2>/dev/null | tail -1 > "$last_obs"

    local dense_planner dense_ring dense_locked dense_stream dense_ahead
    dense_planner="$(grep -q 'planner=1' "$flex_line" && echo 1 || echo 0)"
    dense_ring="$(grep -o 'ring=[0-9]*' "$flex_line" | tail -1 | cut -d= -f2)"; dense_ring="${dense_ring:-NA}"
    dense_locked="$(kv dense_locked_bytes "$last_obs")"; dense_locked="${dense_locked:-NA}"
    dense_stream="$(kv dense_stream_per_token_bytes "$last_obs")"; dense_stream="${dense_stream:-NA}"
    dense_ahead="$(kv dense_effective_ahead "$last_obs")"; dense_ahead="${dense_ahead:-NA}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$model" "$mem_mb" "$high_mb" "$status" "$finish" "$error" "$pred_tok_s" "$pred_ms" "$prompt_tok_s" \
        "$eval_tok_s" "$prompt_srv" \
        "$(kv effective_pressure_state "$last_obs")" \
        "$(kv pressure_source "$last_obs")" \
        "$mem_current" "$mem_peak" "$max_events" "$oom" "$oom_kill" \
        "$dense_planner" "$dense_ring" "$dense_locked" "$dense_stream" "$dense_ahead" \
        "$(kv global_optimizer_decision "$last_obs")" \
        "$(kv dense_repin_reason "$last_obs")" \
        "$(kv moe_enabled "$last_obs")" \
        "$(kv moe_budget_bytes "$last_obs")" \
        "$(kv moe_resident_bytes "$last_obs")" \
        "$(kv moe_evictions "$last_obs")" \
        "$(kv moe_bytes_read "$last_obs")" \
        "$(kv moe_prefetch_budget_dropped "$last_obs")" \
        "$(kv moe_budget_action "$last_obs")" \
        "$(kv moe_budget_reason "$last_obs")" \
        "$(kv global_optimizer_decision "$last_obs")" \
        "$(kv reallocation_reason "$last_obs")" \
        "$(kv kv_offload_reason "$last_obs")" \
        "$run_dir" >> "$SUMMARY"
}

run_case() {
    local model_kind="$1"
    local mem_mb="$2"
    local idx="$3"
    local port=$((BASE_PORT + idx))
    local high_mb="$mem_mb"
    if (( HIGH_MARGIN_MB > 0 && mem_mb > HIGH_MARGIN_MB )); then
        high_mb=$((mem_mb - HIGH_MARGIN_MB))
    fi
    local run_dir="$OUT_DIR/${model_kind}-${mem_mb}M"
    local cg="/sys/fs/cgroup/global-sweep-${model_kind}-${mem_mb}M-$$"
    local log="$run_dir/server.log"
    local out="$run_dir/completion.json"
    local samples="$run_dir/cgroup-samples.tsv"

    mkdir -p "$run_dir" "$cg" 2>/dev/null
    echo $((mem_mb * 1024 * 1024)) > "$cg/memory.max"
    if (( HIGH_MARGIN_MB > 0 )); then
        echo $((high_mb * 1024 * 1024)) > "$cg/memory.high" 2>/dev/null || true
    else
        echo max > "$cg/memory.high" 2>/dev/null || true
    fi
    echo 0 > "$cg/memory.oom.group" 2>/dev/null || true

    local model="$DENSE_MODEL"
    local prompt="Explain memory scheduling briefly."
    local -a env_cmd
    env_cmd=(
        env
        -u LLAMA_FLEX_LOCK_GB
        -u LLAMA_FLEX_RING
        -u LLAMA_FLEX_AHEAD
        -u LLAMA_FLEX_MAX_AHEAD
        -u LLAMA_FLEX_SCHED
        -u LLAMA_FLEX_AUTO
        LLAMA_MEMORY_GOVERNOR=1
        LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER=1
        LLAMA_MEMORY_GOVERNOR_REALLOCATION=1
        LLAMA_MEMORY_GOVERNOR_DENSE_REPIN=1
        LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_ASYNC=1
        LLAMA_MEMORY_GOVERNOR_OBSERVE_MS=500
        LLAMA_FLEX_DEBUG=1
    )

    if [[ "$model_kind" == "moe" ]]; then
        model="$MOE_MODEL"
        prompt="Explain why expert caching matters."
        env_cmd+=(
            LLAMA_LAZY_V2=1
            LLAMA_LAZY_CLG=1
            LLAMA_LAZY_MOE_BUFFER=1
            LLAMA_LAZY_MOE_BUFFER_AUTO=1
            LLAMA_LAZY_MOE_BUFFER_WORKERS=4
            LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0
            LLAMA_LAZY_MOE_SIDECAR="$MOE_SIDECAR"
            LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM=1
            LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_RANKED=1
            LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=1
            LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC=1
            LLAMA_MEMORY_GOVERNOR_MOE_FAST_START=1
            LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=1
        )
    fi

    echo -e "memory_current\tbytes" > "$samples"
    (
        echo "$BASHPID" > "$cg/cgroup.procs"
        exec "${env_cmd[@]}" "$SERVER" \
            -m "$model" \
            --host 127.0.0.1 \
            --port "$port" \
            -c "$CTX" \
            -b "$BATCH" \
            -ub "$BATCH" \
            -t "$THREADS" \
            -np 1 \
            --no-webui
    ) > "$log" 2>&1 &
    local pid=$!

    local status="ok"
    if ! wait_health "$port" "$pid" "$log"; then
        status="startup_failed"
    else
        local mon_pid=""
        (
            while kill -0 "$pid" 2>/dev/null; do
                printf 'memory_current\t%s\n' "$(cat "$cg/memory.current" 2>/dev/null || echo 0)" >> "$samples"
                sleep 1
            done
        ) &
        mon_pid=$!

        curl --noproxy '*' -sS --max-time "$REQUEST_TIMEOUT_SEC" "http://127.0.0.1:${port}/v1/completions" \
            -H 'Content-Type: application/json' \
            -d "{\"model\":\"default\",\"prompt\":\"${prompt}\",\"max_tokens\":${TOKENS},\"temperature\":0}" > "$out" 2>"$run_dir/curl.err" || status="request_failed"

        kill "$mon_pid" 2>/dev/null || true
    fi

    cat "$cg/memory.events" > "$run_dir/memory.events" 2>/dev/null || true
    cat "$cg/memory.current" > "$run_dir/memory.current" 2>/dev/null || true

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    summarize "$model_kind" "$mem_mb" "$high_mb" "$status" "$run_dir"
    echo -e "${model_kind}\t${mem_mb}M\t${status}\t${run_dir}"
}

main() {
    need_file "$SERVER"
    need_file "$DENSE_MODEL"
    need_file "$MOE_MODEL"
    need_file "$MOE_SIDECAR"

    pkill -x llama-server 2>/dev/null || true
    sleep 2

    local idx=0
    local mem
    for mem in $MEMORY_MBS; do
        run_case dense "$mem" "$idx"
        idx=$((idx + 1))
        run_case moe "$mem" "$idx"
        idx=$((idx + 1))
    done

    echo "summary: $SUMMARY"
}

main "$@"
