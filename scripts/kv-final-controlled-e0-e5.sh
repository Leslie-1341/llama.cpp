#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
RUNS="${RUNS:-1}"
ALLOW_DIRTY="${ALLOW_DIRTY:-0}"
DRY_RUN="${DRY_RUN:-0}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-900}"
SAMPLE_INTERVAL_SEC="${SAMPLE_INTERVAL_SEC:-0.10}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/oscomp/kv_logs/kv_final_controlled_e0_e5_${TIMESTAMP}}"
RUNNER_FILE="$ROOT/scripts/kv-final-controlled-e0-e5.sh"
PARSER="$ROOT/scripts/parse-kv-final-controlled-e0-e5.py"
PROTOCOL_FILE="$ROOT/docs/kv_final_controlled_e0_e5_protocol.md"
MEMORY_SAMPLER="$ROOT/scripts/kv-controlled-memory-sampler.sh"

DEFAULT_BINARY_CANDIDATES=(
    "$ROOT/build/bin/llama-kv-idle-swap-resume"
    "$ROOT/build-release/bin/llama-kv-idle-swap-resume"
    "$ROOT/build-Release/bin/llama-kv-idle-swap-resume"
)

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

[[ "$RUNS" == "1" || "$RUNS" == "3" ]] || die "RUNS must be 1 or 3 (got $RUNS)"
[[ "$ALLOW_DIRTY" == "0" || "$ALLOW_DIRTY" == "1" ]] || die "ALLOW_DIRTY must be 0 or 1"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ -d "$ROOT/.git" ]] || die "repository not found: $ROOT"
[[ -f "$RUNNER_FILE" ]] || die "runner not found: $RUNNER_FILE"
[[ -f "$PARSER" ]] || die "parser not found: $PARSER"
[[ -f "$PROTOCOL_FILE" ]] || die "protocol not found: $PROTOCOL_FILE"
[[ -f "$MEMORY_SAMPLER" ]] || die "memory sampler not found: $MEMORY_SAMPLER"
[[ ! -e "$OUTPUT_ROOT" ]] || die "output path already exists: $OUTPUT_ROOT"

# shellcheck source=scripts/kv-controlled-memory-sampler.sh
source "$MEMORY_SAMPLER"

if [[ -n "${BINARY:-}" ]]; then
    BINARY="$BINARY"
else
    BINARY=""
    for candidate in "${DEFAULT_BINARY_CANDIDATES[@]}"; do
        if [[ -x "$candidate" ]]; then
            BINARY="$candidate"
            break
        fi
    done
fi

if [[ -z "$BINARY" || ! -x "$BINARY" ]]; then
    {
        printf 'error: Release binary not found or not executable. Set BINARY explicitly.\n'
        printf 'expected paths:\n'
        printf '  %s\n' "${DEFAULT_BINARY_CANDIDATES[@]}"
    } >&2
    exit 2
fi
[[ -f "$MODEL" ]] || die "model not found: $MODEL"

git_status="$(git -C "$ROOT" status --porcelain)"
if [[ "$RUNS" == "3" && -n "$git_status" && "$ALLOW_DIRTY" != "1" ]]; then
    die "formal RUNS=3 requires a clean worktree; use ALLOW_DIRTY=1 to override (recorded as a warning)"
fi

mkdir -p "$OUTPUT_ROOT/runs"

COMMON_ARGS=(
    -m "$MODEL"
    --ctx-size 2048
    --n-predict 128
    --batch-size 128
    --ubatch-size 128
    --seed 1
    --temp 0
    --cache-type-k f32
    --cache-type-v f32
    --kv-unified
    --parallel 4
    --log-verbosity 4
    --no-log-prefix
    --no-log-timestamps
)

# Every functional switch is assigned for every case. Fault-injection variables are removed
# separately so an inherited test environment cannot contaminate a run.
COMMON_ENV=(
    LLAMA_KV_ACTIVE_TOKEN_STATS=1
    LLAMA_KV_PAGED_IO_STATS=1
    LLAMA_KV_PAGED_IDENTITY_FAST_PATH=0
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
    LLAMA_KV_CACHE_DEBUG=0
    LLAMA_KV_PAGED_BLOCK_SIZE=16
    LLAMA_KV_PAGED_SHIFT=0
    LLAMA_KV_PAGED_RELEASE=0
    LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=1
    LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=0
    LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
    LLAMA_KV_PAGED_SHADOW_VALIDATE=0
    LLAMA_KV_PAGED_MINCORE=0
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_IDLE_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE_MAX=0
    LLAMA_KV_PAGED_REFAULT_TRACE_ONCE=0
    LLAMA_KV_PAGED_TIMING=0
    LLAMA_KV_PAGED_RESUME_TIMING=0
    LLAMA_KV_PAGED_RESUME_TIMING_STEP=0
    LLAMA_KV_PAGED_RESUME_PREFETCH=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS=0
    LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=1
    LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off
    LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
    LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
    LLAMA_KV_SWAP=0
    LLAMA_KV_SWAP_MODE=exact
    LLAMA_KV_SWAP_WINDOW=0
    LLAMA_KV_SWAP_SINK=0
    LLAMA_KV_SWAP_RSS_SAMPLE=0
    LLAMA_KV_SWAP_MADVISE=0
    LLAMA_KV_SWAP_BACKEND_SELFTEST=0
    LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST=0
)

FAULT_ENV=(
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE
    LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_IO_FAIL_KIND
    LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID
    LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE
    LLAMA_KV_TEST_EXPECT_SWAP_OUT_IO_FAILURE
    LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE
    LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE
    LLAMA_KV_TEST_RETRY_ACTIVE_DECODE
    LLAMA_KV_STABILITY_CYCLES
    LLAMA_KV_STABILITY_DURATION_SEC
    LLAMA_KV_STABILITY_WARMUP_SEC
    LLAMA_KV_STABILITY_SAMPLE_EVERY_SEC
    LLAMA_KV_STABILITY_PROGRESS_EVERY
    LLAMA_KV_STABILITY_RSS_LIMIT_MB
    LLAMA_KV_STABILITY_VERIFY_TOKENS
)

case_env() {
    local case_id="$1"
    case "$case_id" in
        E0)
            printf '%s\n' \
                LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_PAGED=0 LLAMA_KV_PAGED_INGRAPH=0 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
                LLAMA_KV_PAGED_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
            ;;
        E1)
            printf '%s\n' \
                LLAMA_KV_LAZY_CLEAR=1 LLAMA_KV_LAZY_TAIL=1 \
                LLAMA_KV_PAGED=0 LLAMA_KV_PAGED_INGRAPH=0 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
                LLAMA_KV_PAGED_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
            ;;
        E2)
            printf '%s\n' \
                LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
                LLAMA_KV_PAGED_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
            ;;
        E3)
            printf '%s\n' \
                LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
                LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
            ;;
        E4)
            printf '%s\n' \
                LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
                LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
            ;;
        E5)
            printf '%s\n' \
                LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
                LLAMA_KV_PAGED_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP=1 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
                LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1 LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1 \
                LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=1 LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=1 \
                LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=1 LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=1 \
                LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
            ;;
        *) die "unknown case: $case_id" ;;
    esac
}

CASE_LABEL_E0="optimization off baseline"
CASE_LABEL_E1="lazy-only"
CASE_LABEL_E2="paged row-index + in-graph gather infrastructure only"
CASE_LABEL_E3="idle swap, no madvise, no prefetch"
CASE_LABEL_E4="idle swap + madvise, no prefetch"
CASE_LABEL_E5="idle swap + madvise + incremental prefetch + defer"

case_label() {
    local name="CASE_LABEL_$1"
    printf '%s' "${!name}"
}

if [[ "$RUNS" == "1" ]]; then
    RUN_PLAN=(
        1 1 E0  1 2 E1  1 3 E2  1 4 E3  1 5 E4  1 6 E5
    )
else
    RUN_PLAN=(
        1 1 E0  1 2 E3  1 3 E1  1 4 E4  1 5 E2  1 6 E5
        2 1 E5  2 2 E2  2 3 E4  2 4 E1  2 5 E3  2 6 E0
        3 1 E2  3 2 E5  3 3 E0  3 4 E3  3 5 E1  3 6 E4
    )
fi

sha256_or_na() {
    sha256sum "$1" 2>/dev/null | awk '{ print $1 }' || printf 'NA'
}

CGROUP_VERSION="none"
CGROUP_PATH="NA"
CGROUP_CURRENT_FILE=""
CGROUP_MAX_FILE=""
CGROUP_PEAK_FILE=""
if [[ -f /sys/fs/cgroup/cgroup.controllers ]]; then
    CGROUP_VERSION="v2"
    cgroup_rel="$(awk -F: '$1 == "0" { print $3; exit }' /proc/self/cgroup)"
    CGROUP_PATH="/sys/fs/cgroup${cgroup_rel}"
    CGROUP_CURRENT_FILE="$CGROUP_PATH/memory.current"
    CGROUP_MAX_FILE="$CGROUP_PATH/memory.max"
    CGROUP_PEAK_FILE="$CGROUP_PATH/memory.peak"
else
    cgroup_rel="$(awk -F: '$2 ~ /(^|,)memory(,|$)/ { print $3; exit }' /proc/self/cgroup)"
    if [[ -n "$cgroup_rel" ]]; then
        CGROUP_VERSION="v1"
        CGROUP_PATH="/sys/fs/cgroup/memory${cgroup_rel}"
        CGROUP_CURRENT_FILE="$CGROUP_PATH/memory.usage_in_bytes"
        CGROUP_MAX_FILE="$CGROUP_PATH/memory.limit_in_bytes"
        CGROUP_PEAK_FILE="$CGROUP_PATH/memory.max_usage_in_bytes"
    fi
fi

SWAP_BASE="${SWAP_DIR:-$OUTPUT_ROOT/swap}"
mkdir -p "$SWAP_BASE"
swap_fs_type="$(findmnt -n -o FSTYPE --target "$SWAP_BASE" 2>/dev/null || true)"
swap_available_bytes="$(df -PB1 "$SWAP_BASE" 2>/dev/null | awk 'NR == 2 { print $4 }')"

printf '%s\n' "$git_status" > "$OUTPUT_ROOT/git_status.txt"
git -C "$ROOT" diff --stat > "$OUTPUT_ROOT/dirty_diff_stat.txt"

branch="$(git -C "$ROOT" branch --show-current 2>/dev/null || true)"
head_sha="$(git -C "$ROOT" rev-parse HEAD)"
upstream="$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
binary_sha="$(sha256_or_na "$BINARY")"
model_sha="$(sha256_or_na "$MODEL")"
binary_size="$(stat -Lc %s "$BINARY")"
model_size="$(stat -Lc %s "$MODEL")"
runner_sha="$(sha256_or_na "$RUNNER_FILE")"
parser_sha="$(sha256_or_na "$PARSER")"
protocol_sha="$(sha256_or_na "$PROTOCOL_FILE")"
sampler_sha="$(sha256_or_na "$MEMORY_SAMPLER")"
runner_size="$(stat -Lc %s "$RUNNER_FILE")"
parser_size="$(stat -Lc %s "$PARSER")"
protocol_size="$(stat -Lc %s "$PROTOCOL_FILE")"
sampler_size="$(stat -Lc %s "$MEMORY_SAMPLER")"
compiler_info="NA"
cmake_cache="$(dirname "$(dirname "$BINARY")")/CMakeCache.txt"
if [[ -r "$cmake_cache" ]]; then
    compiler_info="$(awk -F= '/^CMAKE_(C|CXX)_COMPILER(:FILEPATH)?=|^CMAKE_BUILD_TYPE(:STRING)?=|^CMAKE_(C|CXX)_COMPILER_VERSION(:STRING)?=/ { print }' "$cmake_cache" | paste -sd ';' -)"
    [[ -n "$compiler_info" ]] || compiler_info="NA"
fi
cpu_model="$(awk -F: '/^model name[[:space:]]*:/ { sub(/^[[:space:]]+/, "", $2); print $2; exit }' /proc/cpuinfo)"
logical_cpus="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
numa_info="$(lscpu 2>/dev/null | awk -F: '/^NUMA node\(s\)|^NUMA node[0-9]+ CPU\(s\)/ { gsub(/^[[:space:]]+/, "", $2); printf "%s=%s;", $1, $2 }')"
memory_total_kb="$(awk '/^MemTotal:/ { print $2 }' /proc/meminfo)"
os_info="$(tr '\n' ' ' < /etc/os-release 2>/dev/null || true)"

{
    printf 'protocol=kv_final_controlled_e0_e5\n'
    printf 'created_utc=%s\n' "$(date -u --iso-8601=seconds)"
    printf 'repo=%s\nbranch=%s\nhead=%s\nupstream=%s\n' "$ROOT" "${branch:-NA}" "$head_sha" "${upstream:-NA}"
    printf 'dirty=%s\nallow_dirty=%s\n' "$([[ -n "$git_status" ]] && printf 1 || printf 0)" "$ALLOW_DIRTY"
    printf 'git_status_porcelain_begin\n%s\ngit_status_porcelain_end\n' "${git_status:-<clean>}"
    printf 'dirty_diff_stat_begin\n%s\ndirty_diff_stat_end\n' "$(<"$OUTPUT_ROOT/dirty_diff_stat.txt")"
    printf 'binary=%s\nbinary_size=%s\nbinary_sha256=%s\n' "$BINARY" "$binary_size" "$binary_sha"
    printf 'model=%s\nmodel_size=%s\nmodel_sha256=%s\n' "$MODEL" "$model_size" "$model_sha"
    printf 'runner=%s\nrunner_size=%s\nrunner_sha256=%s\n' "$RUNNER_FILE" "$runner_size" "$runner_sha"
    printf 'parser=%s\nparser_size=%s\nparser_sha256=%s\n' "$PARSER" "$parser_size" "$parser_sha"
    printf 'protocol_file=%s\nprotocol_size=%s\nprotocol_sha256=%s\n' "$PROTOCOL_FILE" "$protocol_size" "$protocol_sha"
    printf 'memory_sampler=%s\nmemory_sampler_size=%s\nmemory_sampler_sha256=%s\n' "$MEMORY_SAMPLER" "$sampler_size" "$sampler_sha"
    printf 'prompt_source=compiled driver prompts in examples/kv-idle-swap-resume/idle-swap-resume.cpp\n'
    printf 'trace_source=none (compiled controlled workload)\n'
    printf 'runs=%s\ndry_run=%s\ntimeout_sec=%s\nsample_interval_sec=%s\n' "$RUNS" "$DRY_RUN" "$CASE_TIMEOUT_SEC" "$SAMPLE_INTERVAL_SEC"
    printf 'hostname=%s\ndate=%s\nkernel=%s\nos=%s\n' "$(hostname)" "$(date --iso-8601=seconds)" "$(uname -srvm)" "${os_info:-NA}"
    printf 'compiler_build=%s\ncpu_model=%s\nlogical_cpus=%s\nnuma=%s\nmemory_total_kb=%s\n' "${compiler_info:-NA}" "${cpu_model:-NA}" "${logical_cpus:-NA}" "${numa_info:-NA}" "${memory_total_kb:-NA}"
    printf 'cgroup_version=%s\ncgroup_path=%s\nmemory_current=%s\nmemory_max=%s\nmemory_peak_snapshot=%s\n' \
        "$CGROUP_VERSION" "$CGROUP_PATH" "$(kv_controlled_read_first "$CGROUP_CURRENT_FILE")" "$(kv_controlled_read_first "$CGROUP_MAX_FILE")" "$(kv_controlled_read_first "$CGROUP_PEAK_FILE")"
    printf 'swap_dir=%s\nswap_fs_type=%s\nswap_available_bytes=%s\n' "$SWAP_BASE" "${swap_fs_type:-NA}" "${swap_available_bytes:-NA}"
    printf 'run_plan_begin\n'
    for ((i = 0; i < ${#RUN_PLAN[@]}; i += 3)); do
        printf 'round=%s run_order=%s case=%s label=%s\n' "${RUN_PLAN[i]}" "${RUN_PLAN[i+1]}" "${RUN_PLAN[i+2]}" "$(case_label "${RUN_PLAN[i+2]}")"
    done
    printf 'run_plan_end\n'
} > "$OUTPUT_ROOT/manifest.txt"

python3 - "$OUTPUT_ROOT" "$ROOT" "$branch" "$head_sha" "$upstream" "$BINARY" "$binary_sha" "$binary_size" \
        "$MODEL" "$model_sha" "$model_size" "$RUNS" "$DRY_RUN" "$ALLOW_DIRTY" "$CGROUP_VERSION" "$CGROUP_PATH" \
        "$(kv_controlled_read_first "$CGROUP_CURRENT_FILE")" "$(kv_controlled_read_first "$CGROUP_MAX_FILE")" "$SWAP_BASE" "${swap_fs_type:-NA}" \
        "${swap_available_bytes:-NA}" "$RUNNER_FILE" "$runner_sha" "$runner_size" \
        "$PARSER" "$parser_sha" "$parser_size" "$PROTOCOL_FILE" "$protocol_sha" "$protocol_size" \
        "$MEMORY_SAMPLER" "$sampler_sha" "$sampler_size" <<'PY'
import json
import pathlib
import sys

(out, repo, branch, head, upstream, binary, binary_sha, binary_size, model, model_sha,
 model_size, runs, dry_run, allow_dirty, cgroup_version, cgroup_path, memory_current,
 memory_max, swap_dir, swap_fs, swap_available, runner, runner_sha, runner_size,
 parser, parser_sha, parser_size, protocol_file, protocol_sha, protocol_size,
 sampler, sampler_sha, sampler_size) = sys.argv[1:]
root = pathlib.Path(out)
status = (root / "git_status.txt").read_text()
diff_stat = (root / "dirty_diff_stat.txt").read_text()
manifest = {
    "protocol": "kv_final_controlled_e0_e5",
    "repo": {"path": repo, "branch": branch or "NA", "head": head, "upstream": upstream or "NA",
             "dirty": bool(status.strip()), "status_porcelain": status.splitlines(), "dirty_diff_stat": diff_stat.splitlines()},
    "binary": {"path": binary, "sha256": binary_sha, "size": int(binary_size)},
    "model": {"path": model, "sha256": model_sha, "size": int(model_size)},
    "framework": {
        "runner": {"path": runner, "sha256": runner_sha, "size": int(runner_size)},
        "parser": {"path": parser, "sha256": parser_sha, "size": int(parser_size)},
        "protocol": {"path": protocol_file, "sha256": protocol_sha, "size": int(protocol_size)},
        "memory_sampler": {"path": sampler, "sha256": sampler_sha, "size": int(sampler_size)},
    },
    "workload": {"prompt_source": "compiled driver prompts", "trace_source": "none", "runs": int(runs)},
    "execution": {"dry_run": dry_run == "1", "allow_dirty": allow_dirty == "1"},
    "cgroup": {"version": cgroup_version, "path": cgroup_path, "memory_current": memory_current, "memory_max": memory_max},
    "swap": {"directory": swap_dir, "filesystem": swap_fs, "available_bytes": swap_available},
    "host_manifest": (root / "manifest.txt").read_text().splitlines(),
    "planned_runs": [],
}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

rm "$OUTPUT_ROOT/git_status.txt" "$OUTPUT_ROOT/dirty_diff_stat.txt"

write_command() {
    local destination="$1"
    shift
    local env_values=("$@")
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\ncd %q\n' "$ROOT"
        printf 'env'
        for name in "${FAULT_ENV[@]}"; do
            printf ' -u %q' "$name"
        done
        printf ' %q' "${env_values[@]}"
        printf ' %q' "$BINARY" "${COMMON_ARGS[@]}"
        printf '\n'
    } > "$destination"
    chmod +x "$destination"
}

append_planned_run_json() {
    local run_dir="$1"
    python3 - "$OUTPUT_ROOT/manifest.json" "$run_dir/run.json" "$run_dir/command" "$run_dir/environment" <<'PY'
import json
import pathlib
import sys
manifest_path, run_path, command_path, environment_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
run = json.loads(run_path.read_text())
run["command"] = command_path.read_text().strip()
run["environment"] = environment_path.read_text().splitlines()
manifest["planned_runs"].append(run)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

extract_section() {
    local input="$1" begin="$2" end="$3" output="$4"
    awk -v begin="$begin" -v end="$end" '
        $0 == begin { inside = 1; next }
        $0 == end { inside = 0; next }
        inside { print }
    ' "$input" > "$output"
}

run_once() {
    local round="$1" order="$2" case_id="$3"
    local run_dir="$OUTPUT_ROOT/runs/round_${round}_order_$(printf '%02d' "$order")_${case_id}"
    local swap_dir="$SWAP_BASE/${case_id}_round_${round}_order_${order}"
    local overrides=() env_values=() unset_args=() rc wrapper_pid sampler_pid
    mkdir -p "$run_dir" "$swap_dir"
    mapfile -t overrides < <(case_env "$case_id")
    env_values=("${COMMON_ENV[@]}" "${overrides[@]}" "LLAMA_KV_SWAP_DIR=$swap_dir")
    for name in "${FAULT_ENV[@]}"; do
        unset_args+=( -u "$name" )
    done

    python3 - "$run_dir/run.json" "$round" "$order" "$case_id" "$(case_label "$case_id")" <<'PY'
import json, pathlib, sys
path, round_no, order, case_id, label = sys.argv[1:]
pathlib.Path(path).write_text(json.dumps({"round": int(round_no), "run_order": int(order), "case": case_id, "label": label}, sort_keys=True) + "\n")
PY
    printf '%s\n' "${env_values[@]}" > "$run_dir/environment"
    write_command "$run_dir/command" "${env_values[@]}"
    append_planned_run_json "$run_dir"

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN\n' > "$run_dir/exit_code"
        : > "$run_dir/stdout"
        : > "$run_dir/stderr"
        : > "$run_dir/seq0"
        : > "$run_dir/seq1"
        printf 'NA  %s\n' "$run_dir/seq0" > "$run_dir/seq0.sha256"
        printf 'NA  %s\n' "$run_dir/seq1" > "$run_dir/seq1.sha256"
        printf 'elapsed_ms\tpid\tstarttime_ticks\tvmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes\n' > "$run_dir/memory_samples.tsv"
        return
    fi

    printf 'running round=%s order=%s case=%s (%s)\n' "$round" "$order" "$case_id" "$(case_label "$case_id")"
    set +e
    timeout --signal=TERM --kill-after=10 "$CASE_TIMEOUT_SEC" \
        env "${unset_args[@]}" "${env_values[@]}" "$BINARY" "${COMMON_ARGS[@]}" \
        > "$run_dir/stdout" 2> "$run_dir/stderr" &
    wrapper_pid=$!
    kv_controlled_sample_process "$wrapper_pid" "$run_dir/memory_samples.tsv" "$swap_dir" "$SAMPLE_INTERVAL_SEC" "$CGROUP_CURRENT_FILE" &
    sampler_pid=$!
    wait "$wrapper_pid"
    rc=$?
    wait "$sampler_pid" 2>/dev/null || true
    set -e
    printf '%s\n' "$rc" > "$run_dir/exit_code"
    extract_section "$run_dir/stdout" '===SEQ0_RESUME_BEGIN===' '===SEQ0_RESUME_END===' "$run_dir/seq0"
    extract_section "$run_dir/stdout" '===SEQ1_ACTIVE_BEGIN===' '===SEQ1_ACTIVE_END===' "$run_dir/seq1"
    sha256sum "$run_dir/seq0" > "$run_dir/seq0.sha256"
    sha256sum "$run_dir/seq1" > "$run_dir/seq1.sha256"
}

for ((i = 0; i < ${#RUN_PLAN[@]}; i += 3)); do
    run_once "${RUN_PLAN[i]}" "${RUN_PLAN[i+1]}" "${RUN_PLAN[i+2]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
    python3 "$PARSER" --dry-run "$OUTPUT_ROOT"
    printf 'dry-run prepared %s planned runs in %s; model was not started\n' "$(( ${#RUN_PLAN[@]} / 3 ))" "$OUTPUT_ROOT"
else
    python3 "$PARSER" "$OUTPUT_ROOT"
    printf 'results: %s\n' "$OUTPUT_ROOT"
fi
