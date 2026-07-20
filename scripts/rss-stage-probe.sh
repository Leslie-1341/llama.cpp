#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT_DIR/build/bin/llama-cli}"
MODEL="${MODEL:?set MODEL=/path/to/model.gguf}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/rss-stage-results/$(date +%Y%m%d-%H%M%S)}"

PROMPT="${PROMPT:-Hello world}"
N_PREDICT="${N_PREDICT:-1}"
THREADS="${THREADS:-2}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.05}"
TIMEOUT_SEC="${TIMEOUT_SEC:-300}"
VM_PATH="${VM_PATH:-1}"

mkdir -p "$OUT_DIR/smaps"
touch "$OUT_DIR/llama.log"

stdout_file="$OUT_DIR/stdout.txt"
stderr_file="$OUT_DIR/stderr.txt"
timeline_file="$OUT_DIR/timeline.samples.tsv"
stage_file="$OUT_DIR/stage.samples.tsv"
summary_file="$OUT_DIR/summary.tsv"

sample_rollup() {
    local pid="$1"
    local snap="$2"
    local dst="$OUT_DIR/smaps/$snap.smaps_rollup"
    local target_pid

    target_pid="$(main_sample_pid "$pid")"

    if [[ -r "/proc/$target_pid/smaps_rollup" ]]; then
        cp "/proc/$target_pid/smaps_rollup" "$dst"
    else
        : > "$dst"
    fi
}

descendant_pids() {
    local root="$1"
    local queue="$root"
    local all="$root"

    while [[ -n "$queue" ]]; do
        local next=""
        local pid
        for pid in $queue; do
            local children
            children="$(pgrep -P "$pid" 2>/dev/null || true)"
            if [[ -n "$children" ]]; then
                all="$all $children"
                next="$next $children"
            fi
        done
        queue="$next"
    done

    printf "%s\n" "$all"
}

main_sample_pid() {
    local root="$1"
    local best="$root"
    local best_rss=-1
    local pid

    for pid in $(descendant_pids "$root"); do
        if [[ ! -r "/proc/$pid/status" || ! -r "/proc/$pid/smaps_rollup" ]]; then
            continue
        fi

        local rss
        rss="$(awk '/VmRSS/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)"
        if (( ${rss:-0} > best_rss )); then
            best="$pid"
            best_rss="${rss:-0}"
        fi
    done

    printf "%s\n" "$best"
}

extract_rollup() {
    local file="$1"
    awk '
        /^Rss:/            { rss=$2 }
        /^Pss:/            { pss=$2 }
        /^Private_Clean:/  { pc=$2 }
        /^Private_Dirty:/  { pd=$2 }
        /^Shared_Clean:/   { sc=$2 }
        /^Shared_Dirty:/   { sd=$2 }
        /^Anonymous:/      { anon=$2 }
        /^File:/           { file=$2 }
        /^Swap:/           { swap=$2 }
        END {
            printf "%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n", rss, pss, pc, pd, sc, sd, anon, file, swap
        }
    ' "$file"
}

timeline_sampler() {
    local pid="$1"

    printf "ts\trss_kb\thwm_kb\tpss_kb\tminor\tmajor\n" > "$timeline_file"
    while kill -0 "$pid" 2>/dev/null; do
        local ts rss hwm pss minor major cur
        ts="$(date +%s.%N)"
        rss=0
        hwm=0
        pss=0
        minor=0
        major=0

        for cur in $(descendant_pids "$pid"); do
            if [[ ! -r "/proc/$cur/status" || ! -r "/proc/$cur/stat" ]]; then
                continue
            fi

            local cur_rss cur_hwm cur_pss cur_minor cur_major
            cur_rss="$(awk '/VmRSS/ {print $2}' "/proc/$cur/status" 2>/dev/null || echo 0)"
            cur_hwm="$(awk '/VmHWM/ {print $2}' "/proc/$cur/status" 2>/dev/null || echo 0)"
            cur_pss="$(awk '/^Pss:/ {print $2}' "/proc/$cur/smaps_rollup" 2>/dev/null || echo 0)"
            read -r cur_minor cur_major < <(awk '{print $10, $12}' "/proc/$cur/stat" 2>/dev/null || echo "0 0")

            rss=$((rss + ${cur_rss:-0}))
            hwm=$((hwm + ${cur_hwm:-0}))
            pss=$((pss + ${cur_pss:-0}))
            minor=$((minor + ${cur_minor:-0}))
            major=$((major + ${cur_major:-0}))
        done

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ts" "$rss" "$hwm" "$pss" "$minor" "$major" >> "$timeline_file"
        sleep "$SAMPLE_INTERVAL"
    done
}

stage_watcher() {
    local pid="$1"
    local seq=0

    printf "seq\tts\tstage\tlabel\trss_kb\tpss_kb\tprivate_clean_kb\tprivate_dirty_kb\tshared_clean_kb\tshared_dirty_kb\tanonymous_kb\tfile_kb\tswap_kb\tsmaps_file\textra\n" > "$stage_file"

    tail -n +1 -F "$OUT_DIR/llama.log" 2>/dev/null | while IFS= read -r line; do
        [[ "$line" == *"RSS_STAGE "* ]] || continue

        local payload stage label extra ts snap values
        payload="${line#*RSS_STAGE }"
        stage="${payload%% *}"
        payload="${payload#* }"
        label="${payload%% *}"
        if [[ "$payload" == "$label" ]]; then
            extra=""
        else
            extra="${payload#* }"
        fi

        ts="$(date +%s.%N)"
        snap="$(printf "%03d_%s_%s" "$seq" "$stage" "$label")"
        sample_rollup "$pid" "$snap"
        values="$(extract_rollup "$OUT_DIR/smaps/$snap.smaps_rollup")"

        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$seq" "$ts" "$stage" "$label" "$values" "$snap.smaps_rollup" "$extra" >> "$stage_file"

        seq=$((seq + 1))
    done
}

write_summary() {
    local status="$1"

    {
        printf "status\t%s\n" "$status"
        printf "out_dir\t%s\n" "$OUT_DIR"
        printf "timeline_peak\t"
        awk '
            NR == 2 || (NR > 1 && $2 > max) { max=$2; ts=$1; hwm=$3; pss=$4 }
            END { printf "ts=%s rss_kb=%d hwm_kb=%d pss_kb=%d\n", ts, max, hwm, pss }
        ' "$timeline_file"
        printf "stage_peak\t"
        awk -F '\t' '
            NR == 2 || (NR > 1 && $5 > max) { max=$5; seq=$1; ts=$2; stage=$3; label=$4; pss=$6; snap=$14; extra=$15 }
            END { printf "seq=%s ts=%s stage=%s label=%s rss_kb=%d pss_kb=%d smaps=%s extra=%s\n", seq, ts, stage, label, max, pss, snap, extra }
        ' "$stage_file"
    } > "$summary_file"
}

echo "rss stage output: $OUT_DIR"
echo "model: $MODEL"
echo "n_predict: $N_PREDICT"
echo "sample_interval: $SAMPLE_INTERVAL"

vm_args=()
if [[ "$VM_PATH" != "0" ]]; then
    vm_args=(
        --vm-debug-log
        --vm-subgraph
        --vm-layer-schedule
    )
fi

LLAMA_RSS_STAGE_LOG=1 timeout "$TIMEOUT_SEC" "$BIN" \
    -m "$MODEL" \
    -p "$PROMPT" \
    -n "$N_PREDICT" \
    -t "$THREADS" \
    --no-warmup \
    --single-turn \
    --no-display-prompt \
    "${vm_args[@]}" \
    --log-file "$OUT_DIR/llama.log" \
    "$@" \
    > "$stdout_file" \
    2> "$stderr_file" &

pid="$!"
timeline_sampler "$pid" &
sampler="$!"
stage_watcher "$pid" &
watcher="$!"

status="ok"
if ! wait "$pid"; then
    status="fail"
fi

wait "$sampler" 2>/dev/null || true
kill "$watcher" 2>/dev/null || true
wait "$watcher" 2>/dev/null || true

write_summary "$status"

cat "$summary_file"
