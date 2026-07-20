#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
MODEL="${MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}"
OUT_DIR="${OUT_DIR:-$ROOT/rss-stage-results/moe-token-profile-$(date +%Y%m%d-%H%M%S)}"
THREADS="${THREADS:-6}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1}"
GEN_TOKENS="${GEN_TOKENS:-48}"
REPEAT="${REPEAT:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
LIMIT="${LIMIT:-8G}"
BUDGETS="${BUDGETS:-512,768,1536,2432}"
WORKERS="${WORKERS:-1,4}"
EXTRA_ENV="${EXTRA_ENV:-}"
EXTRA_ARGS=()

usage() {
    cat <<EOF
Usage:
  scripts/profile-moe-token-tg.sh [-- extra llama-bench args]

Environment:
  MODEL=FILE             default: $MODEL
  BENCH=FILE             default: $BENCH
  OUT_DIR=DIR            output directory
  LIMIT=8G               cgroup memory.max
  BUDGETS=512,768,1536   LLAMA_LAZY_MOE_BUFFER_MB sweep
  WORKERS=1,4            LLAMA_LAZY_MOE_BUFFER_WORKERS sweep
  THREADS=6
  PROMPT_TOKENS=1
  GEN_TOKENS=48
  REPEAT=1
  TIMEOUT_SEC=900
  EXTRA_ENV='A=1 B=2'    appended to env for every run

Outputs:
  summary.tsv
  report.md
  logs/*.out, logs/*.err
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

[[ -x "$BENCH" ]] || { echo "missing bench binary: $BENCH" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }

mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.tsv"
REPORT="$OUT_DIR/report.md"

printf 'tag\trc\tbudget_mb\tworkers\tpeak_mb\tpp_tps\ttg_tps\tstreams\thits\tevictions\tresident_mib\tbytes_read_mib\tsidecar_read_mib\tsidecar_read_count\tmoe_total_us\tcache_lookup_us\tvictim_select_us\tsidecar_submit_us\tsidecar_wait_us\tsidecar_read_us\tq2_unpack_us\tcache_hit\tcache_miss\tprefetch_hit\tprefetch_late\tprefetch_unused\tevict_clean\tevict_active_window\treload_1tok\treload_4tok\tthrash_4tok_per_evict\n' > "$SUMMARY"

split_csv() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$value"
}

kv() {
    local key="$1"
    local file="$2"
    sed -n "s/.*[[:space:]]$key=\\([^[:space:]]*\\).*/\\1/p" "$file" | tail -n 1
}

extract_bench_tps() {
    local file="$1"
    local kind="$2"
    awk -F'|' -v kind="$kind" '
        $0 ~ ("[|][[:space:]]*" kind "[[:space:]]*[|]") {
            v = $8
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
            sub(/[[:space:]]+±.*/, "", v)
            print v
        }
    ' "$file" | tail -n 1
}

extract_stat_value() {
    local key="$1"
    local file="$2"
    sed -n "s/.*$key=\\([^[:space:]]*\\).*/\\1/p" "$file" | tail -n 1
}

run_case() {
    local budget="$1"
    local workers="$2"
    local tag="b${budget}_w${workers}"
    local cg="/sys/fs/cgroup/moe_profile_$tag"
    local out="$OUT_DIR/logs/$tag.out"
    local err="$OUT_DIR/logs/$tag.err"
    local rc=0
    local peak=0

    mkdir -p "$cg"
    echo "$LIMIT" > "$cg/memory.max"
    echo 0 > "$cg/memory.swap.max" 2>/dev/null || true

    set +e
    (
        echo "$BASHPID" > "$cg/cgroup.procs"
        exec timeout "$TIMEOUT_SEC" env \
            LLAMA_LAZY_V2=1 \
            LLAMA_LAZY_CLG=1 \
            LLAMA_LAZY_MOE_BUFFER=1 \
            LLAMA_MOE_PROFILE=1 \
            LLAMA_LAZY_DEBUG=1 \
            LLAMA_LAZY_MOE_BUFFER_MB="$budget" \
            LLAMA_LAZY_MOE_BUFFER_WORKERS="$workers" \
            $EXTRA_ENV \
            "$BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -t "$THREADS" -r "$REPEAT" "${EXTRA_ARGS[@]}"
    ) >"$out" 2>"$err" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        local cur
        cur="$(cat "$cg/memory.current" 2>/dev/null || echo 0)"
        if [[ "$cur" =~ ^[0-9]+$ && "$cur" -gt "$peak" ]]; then
            peak="$cur"
        fi
        sleep 0.2
    done
    wait "$pid"
    rc=$?
    set -e

    local pp_tps tg_tps streams hits evictions resident bytes_read_mib
    local sidecar_bytes sidecar_read_mib sidecar_read_count moe_total cache_lookup victim submit wait read unpack
    local cache_hit cache_miss prefetch_hit prefetch_late prefetch_unused evict_clean evict_active reload1 reload4 thrash

    pp_tps="$(extract_bench_tps "$out" "pp${PROMPT_TOKENS}" || true)"
    tg_tps="$(extract_bench_tps "$out" "tg${GEN_TOKENS}" || true)"
    streams="$(extract_stat_value 'streams' "$err")"
    hits="$(extract_stat_value 'hits' "$err")"
    evictions="$(extract_stat_value 'evictions' "$err")"
    resident="$(sed -n 's/.*resident=\([0-9.]*\) MiB.*/\1/p' "$err" | tail -n 1)"
    bytes_read_mib="$(sed -n 's/.*mwq_kernel=[^ ]*\/[^ ]*\/\([0-9.]*\) MiB.*/\1/p' "$err" | tail -n 1)"

    sidecar_bytes="$(kv sidecar_bytes "$err")"
    sidecar_read_count="$(kv sidecar_read_count "$err")"
    sidecar_read_mib="$(awk -v b="${sidecar_bytes:-0}" 'BEGIN { printf "%.3f", b / 1048576.0 }')"
    moe_total="$(kv moe_total_us "$err")"
    cache_lookup="$(kv cache_lookup_us "$err")"
    victim="$(kv victim_select_us "$err")"
    submit="$(kv sidecar_submit_us "$err")"
    wait="$(kv sidecar_wait_us "$err")"
    read="$(kv sidecar_read_us "$err")"
    unpack="$(kv q2_unpack_us "$err")"
    cache_hit="$(kv cache_hit "$err")"
    cache_miss="$(kv cache_miss "$err")"
    prefetch_hit="$(kv prefetch_hit "$err")"
    prefetch_late="$(kv prefetch_late "$err")"
    prefetch_unused="$(kv prefetch_unused "$err")"
    evict_clean="$(kv evict_clean "$err")"
    evict_active="$(kv evict_active_window "$err")"
    reload1="$(kv evict_reloaded_within_1_token "$err")"
    reload4="$(kv evict_reloaded_within_4_tokens "$err")"
    thrash="$(awk -v r="${reload4:-0}" -v e="${evict_clean:-0}" 'BEGIN { if (e > 0) printf "%.6f", r / e; else print "0" }')"

    printf '%s\t%d\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$tag" "$rc" "$budget" "$workers" "$((peak/1024/1024))" \
        "${pp_tps:-}" "${tg_tps:-}" "${streams:-}" "${hits:-}" "${evictions:-}" "${resident:-}" "${bytes_read_mib:-}" \
        "$sidecar_read_mib" "${sidecar_read_count:-}" "${moe_total:-}" "${cache_lookup:-}" "${victim:-}" \
        "${submit:-}" "${wait:-}" "${read:-}" "${unpack:-}" "${cache_hit:-}" "${cache_miss:-}" \
        "${prefetch_hit:-}" "${prefetch_late:-}" "${prefetch_unused:-}" "${evict_clean:-}" "${evict_active:-}" \
        "${reload1:-}" "${reload4:-}" "$thrash" >> "$SUMMARY"
}

budgets=()
workers_list=()
split_csv "$BUDGETS" budgets
split_csv "$WORKERS" workers_list

for budget in "${budgets[@]}"; do
    for workers in "${workers_list[@]}"; do
        echo "==> budget=${budget}MB workers=${workers}"
        run_case "$budget" "$workers"
    done
done

{
    echo "# MoE TG Profile"
    echo
    echo "- model: \`$MODEL\`"
    echo "- bench: \`$BENCH\`"
    echo "- output: \`$OUT_DIR\`"
    echo "- columns: \`summary.tsv\` includes timing counters and reload-within-1/4-token thrash indicators."
    echo
    echo '```tsv'
    column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
    echo '```'
    echo
    awk -F'\t' 'NR > 1 && $31+0 > 0.25 {
        printf "- %s: reload_4tok/evict_clean = %.3f, clear cache thrashing signal.\n", $1, $31
    }' "$SUMMARY"
} > "$REPORT"

echo "summary: $SUMMARY"
echo "report:  $REPORT"
