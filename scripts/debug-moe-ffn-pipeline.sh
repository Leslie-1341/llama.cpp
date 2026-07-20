#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PPL_BIN="$ROOT/build/bin/llama-perplexity"
MODEL=""
PROMPT_FILE=""
OUT_DIR="$ROOT/rss-stage-results/moe-ffn-pipeline-debug-$(date +%Y%m%d-%H%M%S)"
CHUNKS=4
THREADS_LIST="1,2,3,6"
ROW_TILES="1,4,16,32,64"
RINGS="1,2,3,4"
TIMEOUT=1200
KEEP_GOING=1
EXTRA_ARGS=()
COMMON_ENV=("LLAMA_LAZY_V2=1" "LLAMA_LAZY_MOE_BUFFER=1")

usage() {
    cat <<'EOF'
Usage:
  scripts/debug-moe-ffn-pipeline.sh -m MODEL.gguf -f wiki.test.raw [options] [-- extra llama-perplexity args]

Purpose:
  Reproduce and localize LLAMA_LAZY_MOE_FFN_PIPELINE correctness drift.

Required:
  -m, --model FILE          model path
  -f, --file FILE           perplexity input file

Options:
  -o, --out DIR             output directory
      --ppl-bin FILE        llama-perplexity binary
      --chunks N            default: 4
      --threads LIST        comma list, default: 1,2,3,6
      --row-tiles LIST      comma list, default: 1,4,16,32,64
      --rings LIST          comma list, default: 1,2,3,4
      --timeout SECONDS     default: 1200
      --env KEY=VALUE       extra env for every run; may be repeated
      --no-lazy-env         do not add default LLAMA_LAZY_V2/MOE_BUFFER env
      --fail-fast           stop at first failed command
  -h, --help

Examples:
  scripts/debug-moe-ffn-pipeline.sh \
    -m models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf \
    -f wikitext-2-raw/wiki.test.raw \
    --chunks 4 --threads 1,2,6

  scripts/debug-moe-ffn-pipeline.sh -m model.gguf -f wiki.test.raw -- --ctx-size 512

Outputs:
  summary.tsv   one line per run: tag, rc, ppl, delta, ffn-check max_abs, timings
  report.md     interpretation and likely failure class
  logs/*.out    stdout for each run
  logs/*.err    stderr for each run
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

csv_to_array() {
    local value="$1"
    local -n out_ref="$2"
    IFS=',' read -r -a out_ref <<< "$value"
}

while (($#)); do
    case "$1" in
        -m|--model)
            MODEL=${2:-}; shift 2 ;;
        -f|--file)
            PROMPT_FILE=${2:-}; shift 2 ;;
        -o|--out)
            OUT_DIR=${2:-}; shift 2 ;;
        --ppl-bin)
            PPL_BIN=${2:-}; shift 2 ;;
        --chunks)
            CHUNKS=${2:-}; shift 2 ;;
        --threads)
            THREADS_LIST=${2:-}; shift 2 ;;
        --row-tiles)
            ROW_TILES=${2:-}; shift 2 ;;
        --rings)
            RINGS=${2:-}; shift 2 ;;
        --timeout)
            TIMEOUT=${2:-}; shift 2 ;;
        --env)
            COMMON_ENV+=("${2:-}"); shift 2 ;;
        --no-lazy-env)
            COMMON_ENV=(); shift ;;
        --fail-fast)
            KEEP_GOING=0; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift
            EXTRA_ARGS=("$@")
            break ;;
        *)
            die "unknown argument: $1" ;;
    esac
done

[[ -x "$PPL_BIN" ]] || die "llama-perplexity not executable: $PPL_BIN"
[[ -n "$MODEL" && -f "$MODEL" ]] || die "model file not found: $MODEL"
[[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]] || die "perplexity input file not found: $PROMPT_FILE"
[[ "$CHUNKS" =~ ^[0-9]+$ && "$CHUNKS" -gt 0 ]] || die "--chunks must be positive"
[[ "$TIMEOUT" =~ ^[0-9]+$ && "$TIMEOUT" -gt 0 ]] || die "--timeout must be positive"

mkdir -p "$OUT_DIR/logs"
SUMMARY="$OUT_DIR/summary.tsv"
REPORT="$OUT_DIR/report.md"
STATIC="$OUT_DIR/static-probes.txt"

printf 'tag\trc\tthreads\tfuse\tpipeline\trow_tile\tring\tcheck\tppl\tdelta_vs_baseline\tffn_check_max_abs\tffn_check_max_rel\tchunk_tail\ttokens_per_second\n' > "$SUMMARY"

BASELINE_PPL=""

extract_ppl() {
    local out="$1"
    local err="$2"
    awk '
        /Final estimate/ {
            for (i = 1; i <= NF; ++i) {
                if ($i == "PPL" && (i + 2) <= NF && $(i + 1) == "=") {
                    v = $(i + 2)
                    gsub(/[^0-9.eE+-].*/, "", v)
                    print v
                }
            }
        }
    ' "$out" "$err" 2>/dev/null | tail -n 1
}

extract_chunk_tail() {
    local out="$1"
    local err="$2"
    grep -E '^\[[[:space:]]*[0-9]+/[[:space:]]*[0-9]+\]|Final estimate|PPL' "$out" "$err" 2>/dev/null |
        tail -n 3 |
        tr '\t' ' ' |
        sed 's/[[:space:]][[:space:]]*/ /g' |
        paste -sd '|' -
}

extract_ffn_check_abs() {
    local err="$1"
    sed -n 's/.*ffn-check.*max_abs=\([0-9.eE+-]*\).*/\1/p' "$err" | sort -g | tail -n 1
}

extract_ffn_check_rel() {
    local err="$1"
    sed -n 's/.*ffn-check.*max_rel=\([0-9.eE+-]*\).*/\1/p' "$err" | sort -g | tail -n 1
}

extract_tps() {
    local out="$1"
    local err="$2"
    grep -E 'tokens per second|tok/s|t/s' "$out" "$err" 2>/dev/null |
        tail -n 1 |
        sed 's/.*[^0-9.]\([0-9][0-9.]*\)[[:space:]]*\(tokens per second\|tok\/s\|t\/s\).*/\1/'
}

ppl_delta() {
    local value="$1"
    if [[ -z "$BASELINE_PPL" || -z "$value" ]]; then
        echo ""
        return
    fi
    awk -v a="$value" -v b="$BASELINE_PPL" 'BEGIN { printf "%.10g", a - b }'
}

run_case() {
    local tag="$1"
    local threads="$2"
    local fuse="$3"
    local pipeline="$4"
    local row_tile="$5"
    local ring="$6"
    local check="$7"
    shift 7
    local out="$OUT_DIR/logs/$tag.out"
    local err="$OUT_DIR/logs/$tag.err"
    local rc=0

    echo "==> $tag"
    set +e
    timeout "$TIMEOUT" \
        env -u LLAMA_LAZY_MOE_FUSE_FFN \
            -u LLAMA_LAZY_MOE_FFN_PIPELINE \
            -u LLAMA_LAZY_MOE_FFN_ROW_TILE \
            -u LLAMA_LAZY_MOE_FFN_BLOCK_RING \
            -u LLAMA_LAZY_MOE_FFN_CHECK \
            -u LLAMA_LAZY_V2 \
            -u LLAMA_LAZY_MOE_BUFFER \
            -u LLAMA_LAZY_MOE_BUFFER_MB \
            -u LLAMA_LAZY_MOE_BUFFER_AUTO \
            -u LLAMA_LAZY_CLG \
            -u LLAMA_LAZY_CLG_DELTA \
            "$@" \
            "$PPL_BIN" -m "$MODEL" -f "$PROMPT_FILE" -t "$threads" --chunks "$CHUNKS" "${EXTRA_ARGS[@]}" \
            >"$out" 2>"$err" </dev/null
    rc=$?
    set -e

    local ppl delta max_abs max_rel chunk_tail tps
    ppl=$(extract_ppl "$out" "$err" || true)
    delta=$(ppl_delta "$ppl")
    max_abs=$(extract_ffn_check_abs "$err" || true)
    max_rel=$(extract_ffn_check_rel "$err" || true)
    chunk_tail=$(extract_chunk_tail "$out" "$err" || true)
    tps=$(extract_tps "$out" "$err" || true)

    printf '%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$tag" "$rc" "$threads" "$fuse" "$pipeline" "$row_tile" "$ring" "$check" \
        "$ppl" "$delta" "$max_abs" "$max_rel" "$chunk_tail" "$tps" >> "$SUMMARY"

    if [[ "$rc" -ne 0 && "$KEEP_GOING" -eq 0 ]]; then
        die "$tag failed with rc=$rc; see $err"
    fi
}

env_args=()
run_with_env_values() {
    local tag="$1"
    local threads="$2"
    local fuse="$3"
    local pipeline="$4"
    local row_tile="$5"
    local ring="$6"
    local check="$7"
    env_args=()
    [[ "$fuse" != "-" ]] && env_args+=("LLAMA_LAZY_MOE_FUSE_FFN=$fuse")
    [[ "$pipeline" != "-" ]] && env_args+=("LLAMA_LAZY_MOE_FFN_PIPELINE=$pipeline")
    [[ "$row_tile" != "-" ]] && env_args+=("LLAMA_LAZY_MOE_FFN_ROW_TILE=$row_tile")
    [[ "$ring" != "-" ]] && env_args+=("LLAMA_LAZY_MOE_FFN_BLOCK_RING=$ring")
    [[ "$check" != "-" ]] && env_args+=("LLAMA_LAZY_MOE_FFN_CHECK=$check")
    run_case "$tag" "$threads" "$fuse" "$pipeline" "$row_tile" "$ring" "$check" "${COMMON_ENV[@]}" "${env_args[@]}"
}

csv_to_array "$THREADS_LIST" THREAD_VALUES
csv_to_array "$ROW_TILES" ROW_TILE_VALUES
csv_to_array "$RINGS" RING_VALUES

BASE_THREADS="${THREAD_VALUES[-1]}"

run_with_env_values "baseline_fuse0_t${BASE_THREADS}" "$BASE_THREADS" "0" "-" "-" "-" "-"
BASELINE_PPL=$(awk -F'\t' 'NR == 2 { print $9 }' "$SUMMARY")

run_with_env_values "fused_nonpipe_t${BASE_THREADS}_rt128" "$BASE_THREADS" "1" "0" "128" "-" "-"
run_with_env_values "pipeline_check_t${BASE_THREADS}_rt16_r3" "$BASE_THREADS" "1" "1" "16" "3" "1"

for t in "${THREAD_VALUES[@]}"; do
    run_with_env_values "pipeline_threads_t${t}_rt16_r3" "$t" "1" "1" "16" "3" "1"
done

for rt in "${ROW_TILE_VALUES[@]}"; do
    run_with_env_values "pipeline_rowtile_t${BASE_THREADS}_rt${rt}_r3" "$BASE_THREADS" "1" "1" "$rt" "3" "1"
done

for ring in "${RING_VALUES[@]}"; do
    run_with_env_values "pipeline_ring_t${BASE_THREADS}_rt16_r${ring}" "$BASE_THREADS" "1" "1" "16" "$ring" "1"
done

{
    echo "Static probes for likely pipeline drift sites"
    echo
    echo "[run_index]"
    grep -n 'run_index' "$ROOT/src/llama-moe-buffer.cpp" || true
    echo
    echo "[pipeline shared maps]"
    grep -n 'ffn_pipe_map\|ffn_barrier_map\|moe_ffn_pipe_acquire\|moe_ffn_pipe_release' "$ROOT/src/llama-moe-buffer.cpp" || true
    echo
    echo "[dst writes]"
    grep -n 'dsts\[.*row\|std::fill(dsts' "$ROOT/src/llama-moe-buffer.cpp" || true
    echo
    echo "[pipeline slot state]"
    grep -n 'slots.*state\|consumers_left\|ps.block\|ps.state' "$ROOT/src/llama-moe-buffer.cpp" || true
} > "$STATIC"

python3 - "$SUMMARY" "$STATIC" "$REPORT" <<'PY'
import csv
import math
import pathlib
import sys

summary = pathlib.Path(sys.argv[1])
static = pathlib.Path(sys.argv[2])
report = pathlib.Path(sys.argv[3])

rows = list(csv.DictReader(summary.open(), delimiter="\t"))

def fnum(v):
    try:
        if v == "":
            return None
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    except ValueError:
        return None

baseline = next((r for r in rows if r["tag"].startswith("baseline_")), None)
base_ppl = fnum(baseline["ppl"]) if baseline else None
nonpipe = next((r for r in rows if r["tag"].startswith("fused_nonpipe_")), None)
pipe = next((r for r in rows if r["tag"].startswith("pipeline_check_")), None)

bad = []
for r in rows:
    if r["rc"] != "0":
        bad.append(f'{r["tag"]}: command failed rc={r["rc"]}')
    p = fnum(r["ppl"])
    if base_ppl is not None and p is not None and abs(p - base_ppl) > 1e-5 and r["pipeline"] == "1":
        bad.append(f'{r["tag"]}: pipeline PPL delta={p - base_ppl:.10g}')

thread_deltas = [
    (int(r["threads"]), fnum(r["delta_vs_baseline"]), r["tag"])
    for r in rows
    if r["tag"].startswith("pipeline_threads_") and fnum(r["delta_vs_baseline"]) is not None
]
row_deltas = [
    (int(r["row_tile"]), fnum(r["delta_vs_baseline"]), r["tag"])
    for r in rows
    if r["tag"].startswith("pipeline_rowtile_") and fnum(r["delta_vs_baseline"]) is not None
]
ring_deltas = [
    (int(r["ring"]), fnum(r["delta_vs_baseline"]), r["tag"])
    for r in rows
    if r["tag"].startswith("pipeline_ring_") and fnum(r["delta_vs_baseline"]) is not None
]

ffn_checks = [
    fnum(r["ffn_check_max_abs"])
    for r in rows
    if r["pipeline"] == "1" and fnum(r["ffn_check_max_abs"]) is not None
]
max_check = max(ffn_checks) if ffn_checks else None

diagnosis = []
if base_ppl is None:
    diagnosis.append("Baseline PPL was not parsed; inspect baseline logs first.")
elif nonpipe and fnum(nonpipe["ppl"]) is not None and abs(fnum(nonpipe["ppl"]) - base_ppl) <= 1e-5:
    diagnosis.append("Non-pipeline fused FFN matches baseline for this run, so gate/up/down math is probably not the first suspect.")
elif nonpipe:
    diagnosis.append("Non-pipeline fused FFN also differs from baseline; debug the fused math before pipeline synchronization.")

if pipe and fnum(pipe["ppl"]) is not None and base_ppl is not None:
    delta = fnum(pipe["ppl"]) - base_ppl
    diagnosis.append(f"Pipeline default probe delta vs baseline: {delta:.10g}.")

if max_check is not None and max_check < 1e-5 and pipe and fnum(pipe["ppl"]) is not None and base_ppl is not None and abs(fnum(pipe["ppl"]) - base_ppl) > 1e-5:
    diagnosis.append("Local ffn-check is tight while full PPL drifts; this points away from a single tile arithmetic error and toward graph-level ordering/lifetime/coverage.")

if thread_deltas:
    nz = [(t, d, tag) for t, d, tag in thread_deltas if d is not None and abs(d) > 1e-5]
    if nz and any(t == 1 for t, _, _ in nz):
        diagnosis.append("Pipeline differs even with -t 1, which suggests enabling the pipeline changes semantics even when there are no consumers.")
    elif nz:
        diagnosis.append("Pipeline drift appears only with multiple threads, making shared state, barriers, dst zeroing, or ring slot ownership the most likely class.")

if row_deltas:
    unique = {round(d or 0.0, 8) for _, d, _ in row_deltas}
    if len(unique) > 1:
        diagnosis.append("Delta changes with ROW_TILE; investigate row-span coverage and dst zero/final write ownership.")

if ring_deltas:
    unique = {round(d or 0.0, 8) for _, d, _ in ring_deltas}
    if len(unique) > 1:
        diagnosis.append("Delta changes with BLOCK_RING; investigate slot reuse, state transitions, and producer/consumer visibility.")
    elif any(d is not None and abs(d) > 1e-5 for _, d, _ in ring_deltas):
        diagnosis.append("Delta is stable across ring sizes; slot aliasing is less likely than op/run identity or accumulation lifecycle.")

static_text = static.read_text(errors="replace")
if "size_t run_index = 0;" in static_text and "moe_ffn_pipe_acquire(*ctx, down_op, run_index" in static_text:
    diagnosis.append("Static probe: run_index is a per-thread local counter but is used as part of the shared pipe key. If threads skip or group work differently, they can rendezvous on different entries for the same op.")

with report.open("w") as fh:
    fh.write("# MoE FFN Pipeline Debug Report\n\n")
    fh.write(f"- Summary: `{summary}`\n")
    fh.write(f"- Static probes: `{static}`\n")
    fh.write(f"- Baseline PPL: `{base_ppl if base_ppl is not None else 'unparsed'}`\n\n")
    fh.write("## Key Runs\n\n")
    for r in rows[:3]:
        fh.write(f"- `{r['tag']}` rc={r['rc']} ppl={r['ppl'] or 'NA'} delta={r['delta_vs_baseline'] or 'NA'} check_abs={r['ffn_check_max_abs'] or 'NA'}\n")
    fh.write("\n## Findings\n\n")
    for item in diagnosis or ["No automatic diagnosis was produced; inspect summary.tsv and logs manually."]:
        fh.write(f"- {item}\n")
    if bad:
        fh.write("\n## Failing Or Drifting Runs\n\n")
        for item in bad[:40]:
            fh.write(f"- {item}\n")
    fh.write("\n## Suggested Next Instrumentation\n\n")
    fh.write("- Log a graph-stable FFN run id from the op/context instead of a thread-local `run_index`, then print `(ith, run_index, op, group, mb, batch, n_blocks)` for the first mismatching op.\n")
    fh.write("- Temporarily compare the complete `dsts[j][0:n_hidden]` after each fused down op, not only one row tile, because current `ffn-check` can miss rows owned by other consumers or later graph writes.\n")
    fh.write("- Force `ROW_TILE=1` and `BLOCK_RING=1`; if drift remains, focus on op lifecycle/dst accumulation rather than ring aliasing.\n")
PY

echo
echo "Wrote:"
echo "  $SUMMARY"
echo "  $REPORT"
echo "  $STATIC"
