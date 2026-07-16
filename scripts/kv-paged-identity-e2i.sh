#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BINARY="${BINARY:-$ROOT/build/bin/llama-kv-idle-swap-resume}"
DRY_RUN="${DRY_RUN:-0}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-900}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/oscomp/kv_logs/kv_paged_identity_e2i_${TIMESTAMP}_$$}"
RUNNER="$ROOT/scripts/kv-paged-identity-e2i.sh"
PARSER="$ROOT/scripts/parse-kv-paged-identity-e2i.py"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || die "DRY_RUN must be 0 or 1"
[[ "$CASE_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] || die "CASE_TIMEOUT_SEC must be a positive integer"
[[ -d "$ROOT/.git" ]] || die "repository not found: $ROOT"
[[ -x "$BINARY" ]] || die "binary not executable: $BINARY"
[[ -f "$MODEL" ]] || die "model not found: $MODEL"
[[ -f "$RUNNER" && -f "$PARSER" ]] || die "runner or parser missing"
[[ ! -e "$OUTPUT_ROOT" ]] || die "output path already exists: $OUTPUT_ROOT"
for dependency in python3 timeout sha256sum git; do
    command -v "$dependency" >/dev/null || die "required command not found: $dependency"
done

CASES=(E0 E2 E2I E2I_NOREUSE SHIFT NONIDENTITY SWAP RELEASE MADVISE)
mkdir -p "$OUTPUT_ROOT/runs" "$OUTPUT_ROOT/swap"

COMMON_ARGS=(
    -m "$MODEL" --ctx-size 2048 --n-predict 128 --batch-size 128 --ubatch-size 128
    --seed 1 --temp 0 --cache-type-k f32 --cache-type-v f32 --kv-unified --parallel 4
    --log-verbosity 4 --no-log-prefix --no-log-timestamps
)

COMMON_ENV=(
    LLAMA_KV_ACTIVE_TOKEN_STATS=1 LLAMA_KV_PAGED_IO_STATS=1
    LLAMA_KV_E2_GET_ROWS_PROFILE=0
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2 LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256 LLAMA_KV_CACHE_DEBUG=0
    LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0
    LLAMA_KV_PAGED_BLOCK_SIZE=16 LLAMA_KV_PAGED_SHIFT=0 LLAMA_KV_PAGED_RELEASE=0
    LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS=1 LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS=0
    LLAMA_KV_PAGED_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0
    LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0 LLAMA_KV_PAGED_SHADOW_VALIDATE=0
    LLAMA_KV_PAGED_MINCORE=0 LLAMA_KV_PAGED_TRACE=0 LLAMA_KV_PAGED_IDLE_TRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE=0 LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE=0
    LLAMA_KV_PAGED_REFAULT_TRACE_MAX=0 LLAMA_KV_PAGED_REFAULT_TRACE_ONCE=0
    LLAMA_KV_PAGED_TIMING=0 LLAMA_KV_PAGED_RESUME_TIMING=0
    LLAMA_KV_PAGED_RESUME_TIMING_STEP=0 LLAMA_KV_PAGED_RESUME_PREFETCH=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0 LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS=0 LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS=1
    LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP=1 LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=1 LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
    LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0 LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
    LLAMA_KV_SWAP=0 LLAMA_KV_SWAP_MODE=exact LLAMA_KV_SWAP_WINDOW=0
    LLAMA_KV_SWAP_SINK=0 LLAMA_KV_SWAP_RSS_SAMPLE=0 LLAMA_KV_SWAP_MADVISE=0
    LLAMA_KV_SWAP_BACKEND_SELFTEST=0 LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST=0
)

POLLUTION_ENV=(
    LLAMA_FLEX LLAMA_FLEX_AHEAD LLAMA_FLEX_AUTO LLAMA_FLEX_BUFFERED LLAMA_FLEX_DEBUG
    LLAMA_FLEX_LOCK_GB LLAMA_FLEX_RING LLAMA_FLEX_THREADS LLAMA_FLEX_TYPE LLAMA_FLEX_FACTOR
    LLAMA_FLEX_ORIG_CTX
    LLAMA_GRAPH_REUSE_DISABLE LLAMA_KV_PAGED_IDENTITY_FAST_PATH LLAMA_KV_PAGED
    LLAMA_KV_PAGED_INGRAPH LLAMA_KV_PAGED_GATHER_NONIDENTITY LLAMA_KV_PAGED_SHIFT
    LLAMA_KV_PAGED_RELEASE LLAMA_KV_PAGED_SWAP LLAMA_KV_PAGED_IDLE_SWAP
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE LLAMA_KV_SWAP LLAMA_KV_SWAP_MADVISE
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE
    LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE LLAMA_KV_PAGED_TEST_IO_FAIL_KIND
    LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE
    LLAMA_KV_TEST_EXPECT_SWAP_OUT_IO_FAILURE
    LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE
    LLAMA_KV_TEST_RETRY_ACTIVE_DECODE
)

# Fail forward-closed: inherited experimental variables must not reach the driver,
# including variables added after this runner was written.
while IFS= read -r name; do
    case "$name" in
        LLAMA_*|GGML_*)
            POLLUTION_ENV+=("$name")
            ;;
    esac
done < <(compgen -e)

case_env() {
    case "$1" in
        E0)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=0 \
                LLAMA_KV_PAGED=0 LLAMA_KV_PAGED_INGRAPH=0 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0
            ;;
        E2)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=0 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
            ;;
        E2I)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0
            ;;
        E2I_NOREUSE)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=1 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0
            ;;
        SHIFT)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
                LLAMA_KV_PAGED_SHIFT=1
            ;;
        NONIDENTITY)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
            ;;
        SWAP)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
                LLAMA_KV_PAGED_SWAP=1
            ;;
        RELEASE)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
                LLAMA_KV_PAGED_RELEASE=1
            ;;
        MADVISE)
            printf '%s\n' LLAMA_GRAPH_REUSE_DISABLE=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=1 \
                LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0 \
                LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
            ;;
        *) die "unknown case: $1" ;;
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

write_execution() {
    local destination="$1"
    shift
    local values=("$@")
    python3 - "$destination" "$ROOT" "$BINARY" "$CASE_TIMEOUT_SEC" "${COMMON_ARGS[@]}" \
        -- "${values[@]}" -- "${POLLUTION_ENV[@]}" <<'PY'
import json, pathlib, sys

destination, cwd, binary, timeout, *rest = sys.argv[1:]
first_sep = rest.index("--")
common_args = rest[:first_sep]
tail = rest[first_sep + 1:]
second_sep = tail.index("--")
env_lines = tail[:second_sep]
env_unset = tail[second_sep + 1:]
env_set = {}
for line in env_lines:
    key, value = line.split("=", 1)
    env_set[key] = value
document = {
    "cwd": cwd,
    "binary": binary,
    "argv": [binary, *common_args],
    "env_set": dict(sorted(env_set.items())),
    "env_unset": sorted(set(env_unset)),
    "timeout": {"seconds": int(timeout), "term_signal": "TERM", "kill_after_seconds": 10},
}
pathlib.Path(destination).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
PY
}

materialize_command() {
    local execution="$1" destination="$2"
    python3 - "$execution" "$destination" <<'PY'
import json, pathlib, shlex, sys

execution_path, destination = map(pathlib.Path, sys.argv[1:])
execution = json.loads(execution_path.read_text())
tokens = [
    "cd", shlex.quote(execution["cwd"]), "&&",
    "timeout", "--signal=TERM", "--kill-after=10", str(execution["timeout"]["seconds"]),
    "env",
]
tokens.extend(f"-u {shlex.quote(name)}" for name in execution["env_unset"])
tokens.extend(
    f"{shlex.quote(name)}={shlex.quote(value)}"
    for name, value in sorted(execution["env_set"].items())
)
tokens.extend(shlex.quote(value) for value in execution["argv"])
destination.write_text(" ".join(tokens) + "\n")
PY
}

run_execution() {
    local execution="$1" run_dir="$2"
    python3 - "$execution" "$run_dir/stdout" "$run_dir/stderr" "$run_dir/exit_code" <<'PY'
import json, os, pathlib, signal, subprocess, sys, time

execution_path, stdout_path, stderr_path, exit_path = sys.argv[1:]
execution = json.loads(pathlib.Path(execution_path).read_text())
env = os.environ.copy()
for name in execution["env_unset"]:
    env.pop(name, None)
env.update(execution["env_set"])
timeout = float(execution["timeout"]["seconds"])
kill_after = float(execution["timeout"]["kill_after_seconds"])
stdout = open(stdout_path, "wb")
stderr = open(stderr_path, "wb")
start = time.monotonic()
terminated_at = None
proc = subprocess.Popen(
    execution["argv"], cwd=execution["cwd"], env=env, stdout=stdout, stderr=stderr,
    start_new_session=True,
)
while True:
    rc = proc.poll()
    if rc is not None:
        break
    now = time.monotonic()
    if terminated_at is None and now - start > timeout:
        os.killpg(proc.pid, signal.SIGTERM)
        terminated_at = time.monotonic()
    if terminated_at is not None and now - terminated_at > kill_after:
        os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(0.05)
stdout.close()
stderr.close()
pathlib.Path(exit_path).write_text(f"{rc}\n")
PY
}

git_status="$(git -C "$ROOT" status --porcelain)"
if [[ "$DRY_RUN" == "0" && -n "$git_status" ]]; then
    die "non-dry-run E2I validation requires a clean worktree"
fi
python3 - "$OUTPUT_ROOT/manifest.json" "$ROOT" "$BINARY" "$MODEL" "$RUNNER" "$PARSER" \
        "$DRY_RUN" "$CASE_TIMEOUT_SEC" "$git_status" "${COMMON_ARGS[@]}" -- "${CASES[@]}" <<'PY'
import hashlib, json, pathlib, subprocess, sys
path, repo, binary, model, runner, parser, dry_run, timeout, status, *rest = sys.argv[1:]
sep = rest.index("--")
common_args = rest[:sep]
cases = rest[sep + 1:]
sha = lambda name: hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()
manifest = {
    "protocol": "kv_paged_identity_e2i", "version": 1,
    "repo": {"path": repo, "head": subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip(),
             "dirty": bool(status.strip()), "status_porcelain": status.splitlines()},
    "binary": {"path": binary, "size": pathlib.Path(binary).stat().st_size, "sha256": sha(binary)},
    "model": {"path": model, "size": pathlib.Path(model).stat().st_size, "sha256": sha(model)},
    "framework": {
        "runner": {"path": runner, "size": pathlib.Path(runner).stat().st_size, "sha256": sha(runner)},
        "parser": {"path": parser, "size": pathlib.Path(parser).stat().st_size, "sha256": sha(parser)},
    },
    "workload": {"common_args": common_args},
    "execution": {"dry_run": dry_run == "1", "timeout_sec": int(timeout)},
    "planned_runs": [{"order": index, "case": case} for index, case in enumerate(cases, 1)],
    "completed_runs": [],
}
pathlib.Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

record_completed() {
    local run_dir="$1"
    python3 - "$OUTPUT_ROOT/manifest.json" "$run_dir" <<'PY'
import hashlib, json, pathlib, sys
manifest_path, run_dir = map(pathlib.Path, sys.argv[1:])
names = ("run.json", "execution.json", "command", "environment", "stdout", "stderr", "exit_code",
         "seq0", "seq1", "sequence.sha256")
manifest = json.loads(manifest_path.read_text())
run = json.loads((run_dir / "run.json").read_text())
def identity(path):
    return {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
run["artifacts"] = {name: identity(run_dir / name) for name in names}
manifest["completed_runs"].append(run)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
}

run_case() {
    local case_id="$1" order="$2" run_dir="$OUTPUT_ROOT/runs/$1" swap_dir="$OUTPUT_ROOT/swap/$1"
    local overrides=() env_values=() rc
    mkdir -p "$run_dir" "$swap_dir"
    mapfile -t overrides < <(case_env "$case_id")
    mapfile -t env_values < <(
        printf '%s\n' "${COMMON_ENV[@]}" "${overrides[@]}" "LLAMA_KV_SWAP_DIR=$swap_dir" |
            python3 -c 'import sys
values = {}
for line in sys.stdin:
    line = line.rstrip("\n")
    values[line.split("=", 1)[0]] = line
print("\n".join(values.values()))'
    )
    printf '%s\n' "${env_values[@]}" > "$run_dir/environment"
    write_execution "$run_dir/execution.json" "${env_values[@]}"
    materialize_command "$run_dir/execution.json" "$run_dir/command"
    printf '{"case":"%s","order":%s}\n' "$case_id" "$order" > "$run_dir/run.json"
    : > "$run_dir/stdout"
    : > "$run_dir/stderr"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN\n' > "$run_dir/exit_code"
        : > "$run_dir/seq0"
        : > "$run_dir/seq1"
        sha256sum "$run_dir/seq0" "$run_dir/seq1" > "$run_dir/sequence.sha256"
        record_completed "$run_dir"
        return
    fi
    printf 'running order=%s case=%s\n' "$order" "$case_id"
    set +e
    run_execution "$run_dir/execution.json" "$run_dir"
    rc=$?
    set -e
    if [[ "$rc" != "0" && ! -s "$run_dir/exit_code" ]]; then
        printf '%s\n' "$rc" > "$run_dir/exit_code"
    fi
    extract_section "$run_dir/stdout" '===SEQ0_RESUME_BEGIN===' '===SEQ0_RESUME_END===' "$run_dir/seq0"
    extract_section "$run_dir/stdout" '===SEQ1_ACTIVE_BEGIN===' '===SEQ1_ACTIVE_END===' "$run_dir/seq1"
    sha256sum "$run_dir/seq0" "$run_dir/seq1" > "$run_dir/sequence.sha256"
    record_completed "$run_dir"
}

for index in "${!CASES[@]}"; do
    run_case "${CASES[index]}" "$((index + 1))"
done

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

if [[ "$parser_rc" == "0" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'dry-run PASS: %s planned cases; model was not started; artifacts: %s\n' "${#CASES[@]}" "$OUTPUT_ROOT"
    else
        printf 'E2I validation PASS: %s\n' "$OUTPUT_ROOT"
    fi
else
    printf 'E2I parser rejected artifact (exit %s): %s\n' "$parser_rc" "$OUTPUT_ROOT" >&2
fi
exit "$parser_rc"
