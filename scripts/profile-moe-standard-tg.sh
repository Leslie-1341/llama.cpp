#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
MODEL="${MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}"
SIDECAR="${SIDECAR:-/tmp/qwen-moe-mwq-v2-hier.sidecar}"
OUT_DIR="${OUT_DIR:-$ROOT/rss-stage-results/moe-standard-tg-$(date +%Y%m%d-%H%M%S)}"
LIMIT="${LIMIT:-8G}"
PROMPT_TOKENS="${PROMPT_TOKENS:-16}"
GEN_TOKENS="${GEN_TOKENS:-64}"
REPEATS="${REPEATS:-5}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"
BASE_THREADS="${BASE_THREADS:-6}"
BASE_BUDGET="${BASE_BUDGET:-1024}"
BASE_WORKERS="${BASE_WORKERS:-1}"
TASKSET_CPUS="${TASKSET_CPUS:-0-7}"
RUN_GROUPS="${RUN_GROUPS:-standard,cooldown,pinned,workers}"
COOLDOWNS="${COOLDOWNS:-0,1,2,4}"
PINNED_MBS="${PINNED_MBS:-0,128,256,356,512}"
COMPUTE_THREADS="${COMPUTE_THREADS:-4,6,8}"
BUFFER_WORKERS="${BUFFER_WORKERS:-1,2,4}"
EXTRA_ARGS=()

if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

[[ -x "$BENCH" ]] || { echo "missing bench binary: $BENCH" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "missing model: $MODEL" >&2; exit 2; }
[[ -f "$SIDECAR" ]] || { echo "missing sidecar: $SIDECAR" >&2; exit 2; }

mkdir -p "$OUT_DIR/logs"
RAW="$OUT_DIR/raw.tsv"
SUMMARY="$OUT_DIR/summary.tsv"
REPORT="$OUT_DIR/report.md"
MOUNTS="$OUT_DIR/mounts.txt"

{
    date -u
    findmnt /tmp || true
    findmnt /dev/shm || true
    df -h /tmp /dev/shm || true
    grep -E 'model name|cpu MHz|scaling_governor' /proc/cpuinfo /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null | head -n 64 || true
    taskset -pc "$$" || true
    ls -lh "$SIDECAR" || true
} > "$MOUNTS"

printf 'group\ttag\trun\trc\tbudget_mb\tworkers\tthreads\tcooldown_tokens\tpinned_fraction\tpeak_mb\telapsed_s\tmax_rss_kb\tvol_cs\tinvol_cs\tpp_tps\ttg_tps\tstreams\thits\tevictions\tresident_mib\tsidecar_read_mib\tsidecar_read_count\tmoe_total_us\tcache_lookup_us\tvictim_select_us\tsidecar_submit_us\tsidecar_wait_us\tsidecar_read_us\tq2_unpack_us\tcache_hit\tcache_miss\tprefetch_hit\tprefetch_late\tprefetch_unused\treload_1tok\treload_4tok\tthrash_4tok_per_evict\n' > "$RAW"

split_csv() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$value"
}

sanitize() {
    echo "$1" | tr '/: ,=' '______'
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

time_value() {
    local label="$1"
    local file="$2"
    sed -n "s/[[:space:]]*$label: //p" "$file" | tail -n 1
}

run_once() {
    local group="$1"
    local tag="$2"
    local run_id="$3"
    local budget="$4"
    local workers="$5"
    local threads="$6"
    local cooldown="$7"
    local pinned_fraction="$8"

    local safe_tag
    safe_tag="$(sanitize "${group}_${tag}_r${run_id}")"
    local cg="/sys/fs/cgroup/moe_standard_$safe_tag"
    local out="$OUT_DIR/logs/$safe_tag.out"
    local err="$OUT_DIR/logs/$safe_tag.err"
    local time_file="$OUT_DIR/logs/$safe_tag.time"
    local rc=0
    local peak=0

    mkdir -p "$cg"
    echo "$LIMIT" > "$cg/memory.max"
    echo 0 > "$cg/memory.swap.max" 2>/dev/null || true

    set +e
    (
        echo "$BASHPID" > "$cg/cgroup.procs"
        exec /usr/bin/time -v -o "$time_file" timeout "$TIMEOUT_SEC" taskset -c "$TASKSET_CPUS" env \
            LLAMA_LAZY_V2=1 \
            LLAMA_LAZY_MOE_BUFFER=1 \
            LLAMA_MOE_PROFILE=1 \
            LLAMA_LAZY_DEBUG=1 \
            LLAMA_LAZY_CLG=0 \
            LLAMA_LAZY_MOE_HEBF=0 \
            LLAMA_LAZY_MOE_DYNBITS_REAL=1 \
            LLAMA_LAZY_MOE_STRICT_SIDECAR=1 \
            LLAMA_LAZY_MOE_SIDECAR="$SIDECAR" \
            LLAMA_LAZY_MOE_FIXED_BITS=2 \
            LLAMA_LAZY_MOE_BUFFER_MB="$budget" \
            LLAMA_LAZY_MOE_BUFFER_WORKERS="$workers" \
            LLAMA_LAZY_MOE_GROUP_COOLDOWN_TOKENS="$cooldown" \
            LLAMA_LAZY_MOE_PINNED_FRACTION="$pinned_fraction" \
            LLAMA_LAZY_MOE_FUSE_GATE_UP=1 \
            LLAMA_LAZY_MOE_FUSE_SWIGLU=0 \
            LLAMA_LAZY_MOE_FUSE_DIRECT_SWIGLU=0 \
            LLAMA_LAZY_MOE_FUSE_FFN=0 \
            LLAMA_LAZY_MOE_FFN_PIPELINE=0 \
            "$BENCH" -m "$MODEL" -p "$PROMPT_TOKENS" -n "$GEN_TOKENS" -t "$threads" -r 1 "${EXTRA_ARGS[@]}"
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
    local elapsed max_rss vol_cs invol_cs

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
    elapsed="$(time_value 'Elapsed (wall clock) time (h:mm:ss or m:ss)' "$time_file")"
    max_rss="$(time_value 'Maximum resident set size (kbytes)' "$time_file")"
    vol_cs="$(time_value 'Voluntary context switches' "$time_file")"
    invol_cs="$(time_value 'Involuntary context switches' "$time_file")"

    printf '%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$group" "$tag" "$run_id" "$rc" "$budget" "$workers" "$threads" "$cooldown" "$pinned_fraction" "$((peak/1024/1024))" \
        "${elapsed:-}" "${max_rss:-}" "${vol_cs:-}" "${invol_cs:-}" "${pp_tps:-}" "${tg_tps:-}" \
        "${streams:-}" "${hits:-}" "${evictions:-}" "${resident:-}" "$sidecar_read_mib" "${sidecar_read_count:-}" \
        "${moe_total:-}" "${cache_lookup:-}" "${victim:-}" "${submit:-}" "${wait:-}" "${read:-}" "${unpack:-}" \
        "${cache_hit:-}" "${cache_miss:-}" "${prefetch_hit:-}" "${prefetch_late:-}" "${prefetch_unused:-}" \
        "${reload1:-}" "${reload4:-}" "$thrash" >> "$RAW"
}

run_case() {
    local group="$1"
    local tag="$2"
    local budget="$3"
    local workers="$4"
    local threads="$5"
    local cooldown="$6"
    local pinned_fraction="$7"
    local i
    for ((i = 1; i <= REPEATS; ++i)); do
        echo "==> $group/$tag repeat=$i budget=${budget}MB workers=$workers threads=$threads cooldown=$cooldown pinned=$pinned_fraction"
        run_once "$group" "$tag" "$i" "$budget" "$workers" "$threads" "$cooldown" "$pinned_fraction"
    done
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

if has_group standard; then
    run_case standard baseline "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" 0 0
fi

if has_group cooldown; then
    cooldown_values=()
    split_csv "$COOLDOWNS" cooldown_values
    for c in "${cooldown_values[@]}"; do
        run_case cooldown "c${c}" "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" "$c" 0
    done
fi

if has_group pinned; then
    pinned_values=()
    split_csv "$PINNED_MBS" pinned_values
    for mb in "${pinned_values[@]}"; do
        frac="$(awk -v p="$mb" -v b="$BASE_BUDGET" 'BEGIN { if (b > 0) printf "%.6f", p / b; else print "0" }')"
        run_case pinned "p${mb}mb" "$BASE_BUDGET" "$BASE_WORKERS" "$BASE_THREADS" 0 "$frac"
    done
fi

if has_group workers; then
    thread_values=()
    worker_values=()
    split_csv "$COMPUTE_THREADS" thread_values
    split_csv "$BUFFER_WORKERS" worker_values
    for t in "${thread_values[@]}"; do
        for w in "${worker_values[@]}"; do
            run_case workers "t${t}_w${w}" "$BASE_BUDGET" "$w" "$t" 0 0
        done
    done
fi

awk -F'\t' '
BEGIN {
    OFS = "\t";
    print "group", "tag", "runs", "rc_fail", "budget_mb", "workers", "threads", "cooldown_tokens", "pinned_fraction",
          "tg_median", "tg_min", "tg_max", "tg_avg", "tg_cv", "peak_mb_max", "evictions_avg",
          "sidecar_read_mib_avg", "reload_4tok_avg", "thrash_4tok_avg", "cache_hit_avg", "cache_miss_avg";
}
NR == 1 { next }
{
    key = $1 SUBSEP $2;
    n[key]++;
    group[key] = $1; tag[key] = $2; budget[key] = $5; workers[key] = $6; threads[key] = $7;
    cooldown[key] = $8; pinned[key] = $9;
    if ($4 != 0) rc_fail[key]++;
    tg = $16 + 0;
    vals[key, n[key]] = tg;
    sum[key] += tg;
    sumsq[key] += tg * tg;
    if (n[key] == 1 || tg < min[key]) min[key] = tg;
    if (n[key] == 1 || tg > max[key]) max[key] = tg;
    if (($10 + 0) > peak[key]) peak[key] = $10 + 0;
    evict_sum[key] += $19 + 0;
    read_sum[key] += $21 + 0;
    reload4_sum[key] += $36 + 0;
    thrash_sum[key] += $37 + 0;
    hit_sum[key] += $30 + 0;
    miss_sum[key] += $31 + 0;
}
function sort_vals(k,    i,j,tmp) {
    for (i = 1; i <= n[k]; ++i) {
        for (j = i + 1; j <= n[k]; ++j) {
            if (vals[k, j] < vals[k, i]) {
                tmp = vals[k, i]; vals[k, i] = vals[k, j]; vals[k, j] = tmp;
            }
        }
    }
}
END {
    for (k in n) {
        sort_vals(k);
        mid = int((n[k] + 1) / 2);
        if (n[k] % 2) median = vals[k, mid];
        else median = (vals[k, mid] + vals[k, mid + 1]) / 2.0;
        avg = sum[k] / n[k];
        var = sumsq[k] / n[k] - avg * avg;
        if (var < 0) var = 0;
        cv = avg > 0 ? sqrt(var) / avg : 0;
        print group[k], tag[k], n[k], rc_fail[k] + 0, budget[k], workers[k], threads[k], cooldown[k], pinned[k],
              sprintf("%.4f", median), sprintf("%.4f", min[k]), sprintf("%.4f", max[k]), sprintf("%.4f", avg),
              sprintf("%.6f", cv), peak[k], sprintf("%.2f", evict_sum[k] / n[k]),
              sprintf("%.3f", read_sum[k] / n[k]), sprintf("%.2f", reload4_sum[k] / n[k]),
              sprintf("%.6f", thrash_sum[k] / n[k]), sprintf("%.2f", hit_sum[k] / n[k]),
              sprintf("%.2f", miss_sum[k] / n[k]);
    }
}' "$RAW" > "$SUMMARY"

{
    echo "# MoE Standard TG Report"
    echo
    echo "- model: \`$MODEL\`"
    echo "- sidecar: \`$SIDECAR\`"
    echo "- config: \`-p $PROMPT_TOKENS -n $GEN_TOKENS -r 1\`, external repeats=\`$REPEATS\`, taskset=\`$TASKSET_CPUS\`"
    echo "- baseline env: fixed q2, CLG=0, HEBF=0, FUSE_FFN=0, FFN_PIPELINE=0, DIRECT_SWIGLU=0, STRICT_SIDECAR=1"
    echo "- mount/cpu info: [mounts.txt]($MOUNTS)"
    echo
    echo '```tsv'
    column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
    echo '```'
} > "$REPORT"

echo "raw:     $RAW"
echo "summary: $SUMMARY"
echo "report:  $REPORT"
