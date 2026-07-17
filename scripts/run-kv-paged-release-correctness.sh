#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
BINARY="${BINARY:-$ROOT/build/bin/llama-kv-idle-swap-resume}"
DRY_RUN="${DRY_RUN:-0}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-900}"
N_PREDICT="${N_PREDICT:-16}"
WARMUP_TOKENS="${WARMUP_TOKENS:-64}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/oscomp/kv_logs/kv_paged_release_${TIMESTAMP}_$$}"
PARSER="$ROOT/scripts/parse-kv-paged-release-correctness.py"
RUNNER="$(realpath "${BASH_SOURCE[0]}")"
BINARY="$(realpath "$BINARY")"
MODEL="$(realpath "$MODEL")"
PARSER="$(realpath "$PARSER")"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || die "DRY_RUN must be 0 or 1"
[[ "$CASE_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] || die "CASE_TIMEOUT_SEC must be positive"
[[ "$N_PREDICT" =~ ^[1-9][0-9]*$ ]] || die "N_PREDICT must be positive"
[[ "$WARMUP_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "WARMUP_TOKENS must be positive"
[[ -x "$BINARY" ]] || die "binary not executable: $BINARY"
[[ -f "$MODEL" ]] || die "model missing: $MODEL"
[[ -f "$PARSER" ]] || die "parser missing: $PARSER"
[[ ! -e "$OUTPUT_ROOT" ]] || die "output already exists: $OUTPUT_ROOT"
for dep in python3 timeout sha256sum git; do command -v "$dep" >/dev/null || die "missing $dep"; done

CASES=(R0 R1 R2 R3 R4 R5 N0 N1 N2)
if [[ "$DRY_RUN" == 0 && -n "$(git -C "$ROOT" status --porcelain)" ]]; then
    die "formal gate requires a clean worktree"
fi
mkdir -p "$OUTPUT_ROOT/runs"

COMMON_ARGS=(
    -m "$MODEL" --ctx-size 1024 --n-predict "$N_PREDICT" --batch-size 128 --ubatch-size 128
    --seed 1 --temp 0 --kv-unified --parallel 4
    --log-verbosity 4 --no-log-prefix --no-log-timestamps
)

COMMON_ENV=(
    LLAMA_KV_TEST_MODE=1 LLAMA_KV_ACTIVE_TOKEN_STATS=0
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2 LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS="$WARMUP_TOKENS"
    LLAMA_KV_PAGED=1 LLAMA_KV_PAGED_INGRAPH=1 LLAMA_KV_PAGED_GATHER_NONIDENTITY=0
    LLAMA_KV_PAGED_BLOCK_SIZE=16 LLAMA_KV_PAGED_SHIFT=0 LLAMA_KV_PAGED_RELEASE=0
    LLAMA_KV_PAGED_MINCORE=0 LLAMA_KV_PAGED_SWAP=0 LLAMA_KV_PAGED_IDLE_SWAP=0
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=0 LLAMA_KV_PAGED_SHADOW_VALIDATE=0
    LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE=0 LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ=-1
    LLAMA_KV_RELEASE_TEST_SHARE_SEQ0=0 LLAMA_KV_PAGED_IDENTITY_FAST_PATH=0
    LLAMA_KV_PAGED_RELEASE_TEST_FRESH_VERIFY=1 LLAMA_KV_PAGED_RELEASE_TEST_REPEAT=0
    LLAMA_KV_PAGED_RELEASE_TEST_REUSE=0
    LLAMA_KV_SWAP=0 LLAMA_KV_LAZY_CLEAR=0 LLAMA_KV_LAZY_TAIL=0
)

case_env() {
    case "$1" in
        R0) printf '%s\n' LLAMA_KV_PAGED_RELEASE=0 LLAMA_KV_PAGED_MINCORE=0 ;;
        R1) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=1 ;;
        R2) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=1 \
            LLAMA_KV_PAGED_RELEASE_TEST_REPEAT=1 ;;
        R3) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=1 \
            LLAMA_KV_PAGED_RELEASE_TEST_REUSE=1 ;;
        R4) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=1 LLAMA_KV_RELEASE_TEST_SHARE_SEQ0=1 ;;
        R5) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=0 \
            LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE=1 LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ=1 ;;
        N0) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=0 LLAMA_KV_PAGED_INGRAPH=0 ;;
        N1) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=0 ;;
        N2) printf '%s\n' LLAMA_KV_PAGED_RELEASE=1 LLAMA_KV_PAGED_MINCORE=0 LLAMA_KV_PAGED=0 ;;
        *) die "unknown case $1" ;;
    esac
}

case_args() {
    if [[ "$1" == N1 ]]; then
        printf '%s\n' --cache-type-k f16 --cache-type-v f16
    else
        printf '%s\n' --cache-type-k f32 --cache-type-v f32
    fi
}

git -C "$ROOT" rev-parse HEAD > "$OUTPUT_ROOT/head"
git -C "$ROOT" status --porcelain > "$OUTPUT_ROOT/status"
printf '%s\n' "${CASES[@]}" > "$OUTPUT_ROOT/plan"
for kind in binary model runner parser; do
    case "$kind" in
        binary) path="$BINARY" ;;
        model) path="$MODEL" ;;
        runner) path="$RUNNER" ;;
        parser) path="$PARSER" ;;
    esac
    printf '%s\n' "$path" > "$OUTPUT_ROOT/identity.$kind.path"
    sha256sum "$path" | awk '{print $1}' > "$OUTPUT_ROOT/identity.$kind.sha256"
done
printf 'protocol=kv_paged_release_correctness\nversion=2\ndry_run=%s\n' "$DRY_RUN" > "$OUTPUT_ROOT/manifest"
printf '%q ' "$BINARY" "${COMMON_ARGS[@]}" > "$OUTPUT_ROOT/command.base"
printf '\n' >> "$OUTPUT_ROOT/command.base"

for case_id in "${CASES[@]}"; do
    run="$OUTPUT_ROOT/runs/$case_id"
    mkdir -p "$run"
    mapfile -t overrides < <(case_env "$case_id")
    mapfile -t cache_args < <(case_args "$case_id")
    printf '%s\n' "${COMMON_ENV[@]}" "${overrides[@]}" | awk -F= '{v[$1]=$0} END {for (k in v) print v[k]}' | sort > "$run/environment"
    printf '%q ' env -i PATH="$PATH" HOME="$HOME" "${COMMON_ENV[@]}" "${overrides[@]}" "$BINARY" "${COMMON_ARGS[@]}" "${cache_args[@]}" > "$run/command"
    printf '\n' >> "$run/command"
    if [[ "$DRY_RUN" == 1 ]]; then
        : > "$run/stdout"; : > "$run/stderr"; printf 'DRY_RUN\n' > "$run/exit_code"
        (cd "$run" && sha256sum environment command stdout stderr exit_code > artifacts.sha256)
        continue
    fi
    set +e
    env -i PATH="$PATH" HOME="$HOME" "${COMMON_ENV[@]}" "${overrides[@]}" \
        timeout --signal=TERM --kill-after=10 "$CASE_TIMEOUT_SEC" \
        "$BINARY" "${COMMON_ARGS[@]}" "${cache_args[@]}" > "$run/stdout" 2> "$run/stderr"
    rc=$?
    set -e
    printf '%d\n' "$rc" > "$run/exit_code"
    (cd "$run" && sha256sum environment command stdout stderr exit_code > artifacts.sha256)
done

parser_args=()
if [[ "$DRY_RUN" == 1 ]]; then
    parser_args+=(--dry-run)
fi
python3 "$PARSER" "${parser_args[@]}" "$OUTPUT_ROOT"
printf 'artifact=%s\n' "$OUTPUT_ROOT"
