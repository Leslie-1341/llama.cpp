#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BINARY="${BINARY:-$ROOT/build/bin/llama-kv-idle-swap-resume}"
DRY_RUN="${DRY_RUN:-0}"
USE_PERF="${USE_PERF:-auto}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-900}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/oscomp/kv_logs/kv_e0_e2_e5_single_turn_${TIMESTAMP}_$$}"
RUNNER="$ROOT/scripts/kv-e0-e2-e5-single-turn-diagnose.sh"
PARSER="$ROOT/scripts/parse-kv-e0-e2-e5-single-turn.py"


die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "$USE_PERF" == "auto" || "$USE_PERF" == "0" || "$USE_PERF" == "1" ]] || die "USE_PERF must be auto, 0, or 1"
[[ "$CASE_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] || die "CASE_TIMEOUT_SEC must be a positive integer"
[[ -d "$ROOT/.git" ]] || die "repository not found: $ROOT"
[[ -x "$BINARY" ]] || die "binary not executable: $BINARY"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"
[[ -f "$RUNNER" ]] || die "runner not found: $RUNNER"
[[ -f "$PARSER" ]] || die "parser not found: $PARSER"
[[ ! -e "$OUTPUT_ROOT" ]] || die "output path already exists: $OUTPUT_ROOT"
for dependency in python3 timeout sha256sum git; do
    command -v "$dependency" >/dev/null || die "required command not found: $dependency"
done

mkdir -p "$OUTPUT_ROOT/runs"

COMMON_ARGS=(
    -m "$MODEL" --ctx-size 2048 --n-predict 128 --batch-size 128 --ubatch-size 128
    --seed 1 --temp 0 --cache-type-k f32 --cache-type-v f32 --kv-unified --parallel 4
    --log-verbosity 4 --no-log-prefix --no-log-timestamps
)

COMMON_ENV=(
    LLAMA_KV_ACTIVE_TOKEN_STATS=1 LLAMA_KV_PAGED_IO_STATS=1 LLAMA_KV_PAGED_TIMING=1
    LLAMA_KV_E2_GET_ROWS_PROFILE=0
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2 LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256 LLAMA_KV_CACHE_DEBUG=0
    LLAMA_KV_PAGED_BLOCK_SIZE=16 LLAMA_KV_PAGED_SHIFT=0 LLAMA_KV_PAGED_RELEASE=0
    LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=1 LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=0 LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0 LLAMA_KV_PAGED_MINCORE=0 LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_IDLE_TRACE=0 LLAMA_KV_PAGED_REFAULT_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE=0 LLAMA_KV_PAGED_REFAULT_TRACE_MAX=0
    LLAMA_KV_PAGED_REFAULT_TRACE_ONCE=0 LLAMA_KV_PAGED_RESUME_TIMING=0
    LLAMA_KV_PAGED_RESUME_TIMING_STEP=0 LLAMA_KV_PAGED_RESUME_PREFETCH=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS=0 LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=1
    LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=1 LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=1 LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
    LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0 LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0 LLAMA_KV_SWAP=0 LLAMA_KV_SWAP_MODE=exact
    LLAMA_KV_SWAP_WINDOW=0 LLAMA_KV_SWAP_SINK=0 LLAMA_KV_SWAP_RSS_SAMPLE=0
    LLAMA_KV_SWAP_MADVISE=0 LLAMA_KV_SWAP_BACKEND_SELFTEST=0 LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST=0
)

# Remove inherited experiment, fault and stability controls before applying the explicit case environment.
POLLUTION_ENV=(
    LLAMA_FLEX LLAMA_FLEX_AHEAD LLAMA_FLEX_AUTO LLAMA_FLEX_BUFFERED LLAMA_FLEX_DEBUG
    LLAMA_FLEX_LOCK_GB LLAMA_FLEX_RING LLAMA_FLEX_THREADS LLAMA_FLEX_TYPE LLAMA_FLEX_FACTOR
    LLAMA_FLEX_ORIG_CTX LLAMA_KV_SWAP_DIR
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE
    LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE LLAMA_KV_PAGED_TEST_IO_FAIL_KIND
    LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE
    LLAMA_KV_TEST_EXPECT_SWAP_OUT_IO_FAILURE LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE
    LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE LLAMA_KV_TEST_RETRY_ACTIVE_DECODE
    LLAMA_KV_E2_GET_ROWS_PROFILE
    LLAMA_KV_STABILITY_CYCLES LLAMA_KV_STABILITY_DURATION_SEC LLAMA_KV_STABILITY_WARMUP_SEC
    LLAMA_KV_STABILITY_SAMPLE_EVERY_SEC LLAMA_KV_STABILITY_PROGRESS_EVERY
    LLAMA_KV_STABILITY_RSS_LIMIT_MB LLAMA_KV_STABILITY_VERIFY_TOKENS
)

case_env() {
    case "$1" in
        E0)
            printf '%s\n' LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_E2_GET_ROWS_PROFILE=0 \
                LLAMA_KV_PAGED=0 LLAMA_KV_PAGED_INGRAPH=0 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
                LLAMA_KV_PAGED_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
            ;;
        E2)
            printf '%s\n' LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_E2_GET_ROWS_PROFILE=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
                LLAMA_KV_PAGED_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
            ;;
        E5)
            printf '%s\n' LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_E2_GET_ROWS_PROFILE=0 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
                LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
                LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1 \
                LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=1 LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
            ;;
        *) die "unknown case $1" ;;
    esac
}

extract_section() {
    local input="$1" begin="$2" end="$3" output="$4"
    awk -v begin="$begin" -v end="$end" '
        $0 == begin { inside = 1; next }
        $0 == end { inside = 0; next }
        inside { print }
    ' "$input" > "$output"
}

write_command() {
    local destination="$1" use_perf="$2"
    shift 2
    local env_values=("$@") name
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$ROOT"
        if [[ "$use_perf" == "1" ]]; then
            printf 'perf record -q -F 99 -g --call-graph dwarf -o %q -- ' "$(dirname "$destination")/perf.data"
        fi
        printf 'env'
        for name in "${POLLUTION_ENV[@]}"; do
            printf ' -u %q' "$name"
        done
        printf ' %q' "${env_values[@]}" "$BINARY" "${COMMON_ARGS[@]}"
        printf '\n'
    } > "$destination"
    chmod +x "$destination"
}

sha256_file() {
    sha256sum "$1" | awk '{ print $1 }'
}

git_status="$(git -C "$ROOT" status --porcelain)"
head_sha="$(git -C "$ROOT" rev-parse HEAD)"
branch="$(git -C "$ROOT" branch --show-current 2>/dev/null || true)"
upstream="$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
binary_sha="$(sha256_file "$BINARY")"
model_sha="$(sha256_file "$MODEL")"
runner_sha="$(sha256_file "$RUNNER")"
parser_sha="$(sha256_file "$PARSER")"
binary_size="$(stat -Lc %s "$BINARY")"
model_size="$(stat -Lc %s "$MODEL")"
runner_size="$(stat -Lc %s "$RUNNER")"
parser_size="$(stat -Lc %s "$PARSER")"
compiler_info="NA"
cmake_cache="$(dirname "$(dirname "$BINARY")")/CMakeCache.txt"
if [[ -r "$cmake_cache" ]]; then
    compiler_info="$(awk -F= '/^CMAKE_(C|CXX)_COMPILER(:FILEPATH)?=|^CMAKE_BUILD_TYPE(:STRING)?=|^CMAKE_(C|CXX)_COMPILER_VERSION(:STRING)?=/ { print }' "$cmake_cache" | paste -sd ';' -)"
    [[ -n "$compiler_info" ]] || compiler_info="NA"
fi

PERF_AVAILABLE=0
PERF_REASON="not_requested"
if [[ "$DRY_RUN" == "1" ]]; then
    PERF_REASON="dry_run_not_attempted"
elif [[ "$USE_PERF" == "0" ]]; then
    PERF_REASON="disabled_by_user"
elif ! command -v perf >/dev/null; then
    PERF_REASON="perf_not_found"
    [[ "$USE_PERF" != "1" ]] || die "USE_PERF=1 but perf was not found"
else
    set +e
    perf record -q -o "$OUTPUT_ROOT/perf_probe.data" -- true > "$OUTPUT_ROOT/perf_probe.stdout" 2> "$OUTPUT_ROOT/perf_probe.stderr"
    probe_rc=$?
    set -e
    printf '%s\n' "$probe_rc" > "$OUTPUT_ROOT/perf_probe.exit_code"
    if [[ "$probe_rc" == "0" ]]; then
        PERF_AVAILABLE=1
        PERF_REASON="probe_passed"
    else
        PERF_REASON="permission_or_event_unavailable"
        [[ "$USE_PERF" != "1" ]] || die "USE_PERF=1 but the perf probe failed"
    fi
fi

python3 - "$OUTPUT_ROOT/manifest.json" "$ROOT" "$branch" "$head_sha" "$upstream" "$git_status" \
        "$BINARY" "$binary_sha" "$binary_size" "$MODEL" "$model_sha" "$model_size" \
        "$RUNNER" "$runner_sha" "$runner_size" "$PARSER" "$parser_sha" "$parser_size" \
        "$DRY_RUN" "$USE_PERF" "$PERF_AVAILABLE" "$PERF_REASON" "$CASE_TIMEOUT_SEC" "$compiler_info" <<'PY'
import json, pathlib, platform, socket, sys
(path, repo, branch, head, upstream, status, binary, binary_sha, binary_size, model, model_sha,
 model_size, runner, runner_sha, runner_size, parser, parser_sha, parser_size, dry_run, use_perf,
 perf_available, perf_reason, timeout_sec, compiler_info) = sys.argv[1:]
manifest = {
    "protocol": "kv_e0_e2_e5_single_turn_diagnostic",
    "version": 2,
    "scope": "informal_diagnostic_not_a_controlled_performance_conclusion",
    "repo": {"path": repo, "branch": branch or "NA", "head": head, "upstream": upstream or "NA",
             "dirty": bool(status.strip()), "status_porcelain": status.splitlines()},
    "binary": {"path": binary, "sha256": binary_sha, "size": int(binary_size)},
    "model": {"path": model, "sha256": model_sha, "size": int(model_size)},
    "framework": {
        "runner": {"path": runner, "sha256": runner_sha, "size": int(runner_size)},
        "parser": {"path": parser, "sha256": parser_sha, "size": int(parser_size)},
    },
    "host": {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
             "compiler_build": compiler_info},
    "execution": {"dry_run": dry_run == "1", "use_perf": use_perf,
                  "perf_available": perf_available == "1", "perf_reason": perf_reason,
                  "timeout_sec": int(timeout_sec)},
    "planned_runs": [{"order": i, "case": case} for i, case in enumerate(("E0", "E2", "E5"), 1)],
    "runs": [],
}
pathlib.Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

{
    printf 'protocol=kv_e0_e2_e5_single_turn_diagnostic\nversion=2\n'
    printf 'scope=informal_diagnostic_not_a_controlled_performance_conclusion\ncreated_utc=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'repo=%s\nbranch=%s\nhead=%s\nupstream=%s\ndirty=%s\n' "$ROOT" "${branch:-NA}" "$head_sha" "${upstream:-NA}" "$([[ -n "$git_status" ]] && printf 1 || printf 0)"
    printf 'binary=%s\nbinary_size=%s\nbinary_sha256=%s\nmodel=%s\nmodel_size=%s\nmodel_sha256=%s\n' "$BINARY" "$binary_size" "$binary_sha" "$MODEL" "$model_size" "$model_sha"
    printf 'runner=%s\nrunner_size=%s\nrunner_sha256=%s\nparser=%s\nparser_size=%s\nparser_sha256=%s\n' "$RUNNER" "$runner_size" "$runner_sha" "$PARSER" "$parser_size" "$parser_sha"
    printf 'dry_run=%s\nuse_perf=%s\nperf_available=%s\nperf_reason=%s\ntimeout_sec=%s\nrun_plan=E0,E2,E5\n' "$DRY_RUN" "$USE_PERF" "$PERF_AVAILABLE" "$PERF_REASON" "$CASE_TIMEOUT_SEC"
} > "$OUTPUT_ROOT/manifest.txt"

run_case() {
    local case_id="$1" order="$2" run_dir="$OUTPUT_ROOT/runs/$1" swap_dir="$OUTPUT_ROOT/swap/$1"
    local overrides=() env_values=() profile=0 rc report_rc
    mkdir -p "$run_dir" "$swap_dir"
    mapfile -t overrides < <(case_env "$case_id")
    mapfile -t env_values < <(
        printf '%s\n' "${COMMON_ENV[@]}" "${overrides[@]}" "LLAMA_KV_SWAP_DIR=$swap_dir" |
            python3 -c 'import sys
values = {}
for line in sys.stdin:
    line = line.rstrip("\n")
    key = line.split("=", 1)[0]
    values[key] = line
print("\n".join(values.values()))'
    )
    printf '%s\n' "${env_values[@]}" > "$run_dir/environment"
    if [[ "$PERF_AVAILABLE" == "1" && ( "$case_id" == "E0" || "$case_id" == "E2" ) ]]; then
        profile=1
    fi
    write_command "$run_dir/command" "$profile" "${env_values[@]}"
    python3 - "$run_dir/run.json" "$order" "$case_id" <<'PY'
import json, pathlib, sys
path, order, case_id = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({"order": int(order), "case": case_id}, sort_keys=True) + "\n")
PY
    : > "$run_dir/stdout"
    : > "$run_dir/stderr"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN\n' > "$run_dir/exit_code"
        if [[ "$case_id" == "E5" ]]; then
            printf 'state=not_applicable reason=e5_uses_swapin_phase_telemetry\n' > "$run_dir/perf_status"
        else
            printf 'state=unresolved reason=%s\n' "$PERF_REASON" > "$run_dir/perf_status"
        fi
        : > "$run_dir/perf_report.txt"
        : > "$run_dir/perf_report.stderr"
        : > "$run_dir/seq0"
        : > "$run_dir/seq1"
        return
    fi

    printf 'running case=%s perf=%s\n' "$case_id" "$profile"
    set +e
    timeout --signal=TERM --kill-after=10 "$CASE_TIMEOUT_SEC" "$run_dir/command" \
        > "$run_dir/stdout" 2> "$run_dir/stderr"
    rc=$?
    set -e
    printf '%s\n' "$rc" > "$run_dir/exit_code"
    extract_section "$run_dir/stdout" '===SEQ0_RESUME_BEGIN===' '===SEQ0_RESUME_END===' "$run_dir/seq0"
    extract_section "$run_dir/stdout" '===SEQ1_ACTIVE_BEGIN===' '===SEQ1_ACTIVE_END===' "$run_dir/seq1"

    if [[ "$case_id" == "E5" ]]; then
        printf 'state=not_applicable reason=e5_uses_swapin_phase_telemetry\n' > "$run_dir/perf_status"
        : > "$run_dir/perf_report.txt"
        : > "$run_dir/perf_report.stderr"
    elif [[ "$profile" == "1" && -s "$run_dir/perf.data" ]]; then
        set +e
        perf report --stdio --no-children --sort symbol -i "$run_dir/perf.data" \
            > "$run_dir/perf_report.txt" 2> "$run_dir/perf_report.stderr"
        report_rc=$?
        set -e
        if [[ "$report_rc" == "0" ]]; then
            printf 'state=captured reason=report_available\n' > "$run_dir/perf_status"
        else
            printf 'state=unresolved reason=perf_report_failed report_exit=%s\n' "$report_rc" > "$run_dir/perf_status"
        fi
    else
        printf 'state=unresolved reason=%s\n' "$PERF_REASON" > "$run_dir/perf_status"
        : > "$run_dir/perf_report.txt"
        : > "$run_dir/perf_report.stderr"
    fi
}

run_case E0 1
run_case E2 2
run_case E5 3

python3 - "$OUTPUT_ROOT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest_path = root / "manifest.json"
manifest = json.loads(manifest_path.read_text())
required = ("run.json", "command", "environment", "stdout", "stderr", "exit_code", "perf_status", "perf_report.txt", "perf_report.stderr", "seq0", "seq1")
runs = []
for planned in manifest["planned_runs"]:
    run_dir = root / "runs" / planned["case"]
    artifacts = {}
    for name in required:
        path = run_dir / name
        data = path.read_bytes()
        artifacts[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    perf_data = run_dir / "perf.data"
    if perf_data.is_file():
        data = perf_data.read_bytes()
        artifacts["perf.data"] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    runs.append({"order": planned["order"], "case": planned["case"], "directory": str(run_dir.relative_to(root)), "artifacts": artifacts})
manifest["runs"] = runs
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

parser_args=("$PARSER")
if [[ "$DRY_RUN" == "1" ]]; then
    parser_args+=(--dry-run)
fi
parser_args+=("$OUTPUT_ROOT")
printf '%q ' python3 "${parser_args[@]}" > "$OUTPUT_ROOT/parser_command"
printf '\n' >> "$OUTPUT_ROOT/parser_command"
set +e
python3 "${parser_args[@]}" > "$OUTPUT_ROOT/parser.stdout" 2> "$OUTPUT_ROOT/parser.stderr"
parser_rc=$?
set -e
printf '%s\n' "$parser_rc" > "$OUTPUT_ROOT/parser.exit_code"
case "$parser_rc" in
    0)
        if [[ "$DRY_RUN" == "1" ]]; then
            printf 'dry-run PASS: model was not started; artifacts: %s\n' "$OUTPUT_ROOT"
        else
            printf 'diagnostic resolved: %s\n' "$OUTPUT_ROOT"
        fi
        ;;
    1)
        printf 'diagnostic unresolved or failed validation: %s\n' "$OUTPUT_ROOT" >&2
        exit 1
        ;;
    *)
        if [[ ! -f "$OUTPUT_ROOT/summary.md" ]]; then
            printf '# E0/E2/E5 single-turn diagnostic\n\nFAIL: parser rejected the artifact. See `parser.stderr`.\n' > "$OUTPUT_ROOT/summary.md"
        fi
        printf 'error: parser rejected diagnostic artifacts: %s\n' "$OUTPUT_ROOT" >&2
        exit "$parser_rc"
        ;;
esac
