#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
MODEL="${MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}"
SIDECAR="${SIDECAR:-/tmp/qwen-moe-mwq-v2-hier.sidecar}"
OUT_DIR="${OUT_DIR:-$ROOT/rss-stage-results/moe-ablations-$(date +%Y%m%d-%H%M%S)}"
LIMIT="${LIMIT:-8G}"
PROMPT_TOKENS="${PROMPT_TOKENS:-1}"
GEN_TOKENS="${GEN_TOKENS:-48}"
REPEAT="${REPEAT:-1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
BASE_THREADS="${BASE_THREADS:-6}"
BASE_BUDGET="${BASE_BUDGET:-1536}"
BASE_WORKERS="${BASE_WORKERS:-1}"
BUDGETS="${BUDGETS:-512,768,1024,1280,1536,2048}"
THREAD_SCAN="${THREAD_SCAN:-1,2,4,6,8,12}"
TASKSET_CPUS="${TASKSET_CPUS:-0-7}"
RUN_GROUPS="${RUN_GROUPS:-full,storage,fixedq2,threads,budget,swiglu}"
EXTRA_ARGS=()

if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

[[ -x "$BENCH" ]] || { echo "missing bench binary: $BENCH" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }
[[ -f "$SIDECAR" ]] || { echo "missing sidecar: $SIDECAR" >&2; exit 2; }

mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.tsv"
REPORT="$OUT_DIR/report.md"
MOUNTS="$OUT_DIR/mounts.txt"

{
    findmnt /tmp || true
    findmnt /dev/shm || true
    df -h /tmp /dev/shm || true
    ls -lh "$SIDECAR" || true
} > "$MOUNTS"

printf 'group\ttag\trc\tbudget_mb\tworkers\tthreads\tsidecar\tclg\thebf\tfixed_bits\tfuse_gate_up\tfuse_swiglu\tdirect_swiglu\tpeak_mb\tpp_tps\ttg_tps\tstreams\thits\tevictions\tresident_mib\tsidecar_read_mib\tsidecar_read_count\tmoe_total_us\tcache_lookup_us\tvictim_select_us\tsidecar_submit_us\tsidecar_wait_us\tsidecar_read_us\tq2_unpack_us\tcache_hit\tcache_miss\tprefetch_hit\tprefetch_late\tprefetch_unused\treload_1tok\treload_4tok\tthrash_4tok_per_evict\n' > "$SUMMARY"

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

sanitize() {
    echo "$1" | tr '/: ,=' '______'
}

make_shm_sidecar() {
    local dst="/dev/shm/$(basename "$SIDECAR")"
    if cp "$SIDECAR" "$dst" 2>/dev/null; then
        echo "$dst"
    else
        echo ""
    fi
}

prewarm_file() {
    local path="$1"
    dd if="$path" of=/dev/null bs=64M status=none || true
}

run_case() {
    local group="$1"
    local tag="$2"
    local budget="$3"
    local workers="$4"
    local threads="$5"
    local sidecar="$6"
    local clg="$7"
    local hebf="$8"
    local fixed_bits="$9"
    local fuse_gate_up="${10}"
    local fuse_swiglu="${11}"
    local direct_swiglu="${12}"
    local prewarm="${13}"

    local safe_tag
    safe_tag="$(sanitize "${group}_${tag}")"
    local cg="/sys/fs/cgroup/moe_ablation_$safe_tag"
    local out="$OUT_DIR/logs/$safe_tag.out"
    local err="$OUT_DIR/logs/$safe_tag.err"
    local rc=0
    local peak=0

    mkdir -p "$cg"
    echo "$LIMIT" > "$cg/memory.max"
    echo 0 > "$cg/memory.swap.max" 2>/dev/null || true

    if [[ "$prewarm" == "1" ]]; then
        prewarm_file "$sidecar"
    fi

    set +e
    (
        echo "$BASHPID" > "$cg/cgroup.procs"
        exec timeout "$TIMEOUT_SEC" taskset -c "$TASKSET_CPUS" env \
            LLAMA_LAZY_V2=1 \
            LLAMA_LAZY_MOE_BUFFER=1 \
            LLAMA_MOE_PROFILE=1 \
            LLAMA_LAZY_DEBUG=1 \
            LLAMA_LAZY_MOE_DYNBITS_REAL=1 \
            LLAMA_LAZY_MOE_SIDECAR="$sidecar" \
            LLAMA_LAZY_MOE_BUFFER_MB="$budget" \
            LLAMA_LAZY_MOE_BUFFER_WORKERS="$workers" \
            LLAMA_LAZY_CLG="$clg" \
            LLAMA_LAZY_MOE_HEBF="$hebf" \
            LLAMA_LAZY_MOE_FIXED_BITS="$fixed_bits" \
            LLAMA_LAZY_MOE_GATE_MIN_BITS=0 \
            LLAMA_LAZY_MOE_UP_MIN_BITS=0 \
            LLAMA_LAZY_MOE_DOWN_MIN_BITS=0 \
            LLAMA_LAZY_MOE_FUSE_GATE_UP="$fuse_gate_up" \
            LLAMA_LAZY_MOE_FUSE_SWIGLU="$fuse_swiglu" \
            LLAMA_LAZY_MOE_FUSE_DIRECT_SWIGLU="$direct_swiglu" \
            "$BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -t "$threads" -r "$REPEAT" "${EXTRA_ARGS[@]}"
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

    local pp_tps tg_tps streams hits evictions resident sidecar_bytes sidecar_read_mib sidecar_read_count
    local moe_total cache_lookup victim submit wait read unpack cache_hit cache_miss prefetch_hit prefetch_late prefetch_unused reload1 reload4 evict_clean thrash

    pp_tps="$(extract_bench_tps "$out" "pp${PROMPT_TOKENS}" || true)"
    tg_tps="$(extract_bench_tps "$out" "tg${GEN_TOKENS}" || true)"
    streams="$(extract_stat_value 'streams' "$err")"
    hits="$(extract_stat_value 'hits' "$err")"
    evictions="$(extract_stat_value 'evictions' "$err")"
    resident="$(sed -n 's/.*resident=\([0-9.]*\) MiB.*/\1/p' "$err" | tail -n 1)"
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
    reload1="$(kv evict_reloaded_within_1_token "$err")"
    reload4="$(kv evict_reloaded_within_4_tokens "$err")"
    evict_clean="$(kv evict_clean "$err")"
    thrash="$(awk -v r="${reload4:-0}" -v e="${evict_clean:-0}" 'BEGIN { if (e > 0) printf "%.6f", r / e; else print "0" }')"

    printf '%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$group" "$tag" "$rc" "$budget" "$workers" "$threads" "$sidecar" "$clg" "$hebf" "$fixed_bits" \
        "$fuse_gate_up" "$fuse_swiglu" "$direct_swiglu" "$((peak/1024/1024))" \
        "${pp_tps:-}" "${tg_tps:-}" "${streams:-}" "${hits:-}" "${evictions:-}" "${resident:-}" \
        "$sidecar_read_mib" "${sidecar_read_count:-}" "${moe_total:-}" "${cache_lookup:-}" "${victim:-}" \
        "${submit:-}" "${wait:-}" "${read:-}" "${unpack:-}" "${cache_hit:-}" "${cache_miss:-}" \
        "${prefetch_hit:-}" "${prefetch_late:-}" "${prefetch_unused:-}" "${reload1:-}" "${reload4:-}" "$thrash" >> "$SUMMARY"
}

groups=()
split_csv "$RUN_GROUPS" groups
has_group() {
    local want="$1"
    local g
    for g in "${groups[@]}"; do
        [[ "$g" == "$want" ]] && return 0
    done
    return 1
}

SHM_SIDECAR=""
if has_group storage; then
    SHM_SIDECAR="$(make_shm_sidecar || true)"
fi

if has_group full; then
    run_case full resident_unbounded 0 "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 2 1 1 1 1
    run_case full current_budget "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 2 1 1 1 1
fi

if has_group storage; then
    run_case storage disk "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 2 1 1 1 0
    run_case storage disk_prewarm "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 2 1 1 1 1
    if [[ -n "$SHM_SIDECAR" ]]; then
        run_case storage shm "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SHM_SIDECAR" 1 1 2 1 1 1 0
    else
        printf 'storage\tshm_unavailable\t125\t%s\t%s\t%s\t/dev/shm-unavailable\t1\t1\t2\t1\t1\t1\t0\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\n' \
            "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" >> "$SUMMARY"
    fi
fi

if has_group fixedq2; then
    run_case fixedq2 fixed_q2_no_clg_hebf "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 0 0 2 1 1 1 1
    run_case fixedq2 current_dyn_clg_hebf "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 0 1 1 1 1
fi

if has_group threads; then
    thread_values=()
    split_csv "$THREAD_SCAN" thread_values
    for t in "${thread_values[@]}"; do
        run_case threads "t${t}" "$BASE_BUDGET" "$BASE_WORKERS" "$t" "$SIDECAR" 1 1 2 1 1 1 1
    done
fi

if has_group budget; then
    budget_values=()
    split_csv "$BUDGETS" budget_values
    for b in "${budget_values[@]}"; do
        run_case budget "b${b}" "$b" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 2 1 1 1 1
    done
fi

if has_group swiglu; then
    run_case swiglu normal_swiglu "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 2 1 0 0 1
    run_case swiglu direct_swiglu "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 2 1 1 1 1
    run_case swiglu current "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$SIDECAR" 1 1 0 1 1 1 1
fi

{
    echo "# MoE Ablation Report"
    echo
    echo "- model: \`$MODEL\`"
    echo "- sidecar: \`$SIDECAR\`"
    echo "- output: \`$OUT_DIR\`"
    echo "- mount info: [mounts.txt]($MOUNTS)"
    echo
    echo '```tsv'
    column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
    echo '```'
    echo
    awk -F'\t' 'NR > 1 && $37+0 > 0.25 {
        printf "- %s/%s: reload_4tok/evict_clean = %.3f, clear cache thrashing signal.\n", $1, $2, $37
    }' "$SUMMARY"
} > "$REPORT"

echo "summary: $SUMMARY"
echo "report:  $REPORT"
