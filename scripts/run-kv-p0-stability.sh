#!/usr/bin/env bash

set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-kv-p0-stability}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/oscomp/kv_logs/kv_p0_stability}"
CYCLES="${CYCLES:-1000}"
VERIFY_TOKENS="${VERIFY_TOKENS:-128}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-1800}"
RSS_LIMIT_MB="${RSS_LIMIT_MB:-256}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
SKIP_BUILD="${SKIP_BUILD:-0}"

RUNNER="$BUILD_DIR/bin/llama-kv-idle-swap-resume"

COMMON_ARGS=(
    -m "$MODEL"
    --ctx-size 2048
    --n-predict "$VERIFY_TOKENS"
    --batch-size 128
    --ubatch-size 128
    --seed 1
    --temp 0
    --cache-type-k f32
    --cache-type-v f32
    --kv-unified
    --parallel 4
)

KV_ENV_COMMON=(
    LLAMA_KV_PAGED=1
    LLAMA_KV_PAGED_RELEASE=0
    LLAMA_KV_PAGED_SWAP=1
    LLAMA_KV_PAGED_IDLE_SWAP=1
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1
    LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES=0
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1
    LLAMA_KV_PAGED_INGRAPH=1
    LLAMA_KV_PAGED_TRACE=0
    LLAMA_KV_PAGED_IDLE_TRACE=0
    LLAMA_KV_PAGED_RESUME_TIMING=0
    LLAMA_KV_PAGED_RESUME_TIMING_STEP=0
    LLAMA_KV_LAZY_TAIL=0
    LLAMA_KV_LAZY_CLEAR=0
    LLAMA_KV_IDLE_NUM_IDLE_SEQS=2
    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS=256
    LLAMA_KV_PAGED_RESUME_PREFETCH=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
    LLAMA_KV_STABILITY_VERIFY_TOKENS="$VERIFY_TOKENS"
    LLAMA_KV_STABILITY_PROGRESS_EVERY="$PROGRESS_EVERY"
    LLAMA_KV_STABILITY_RSS_LIMIT_MB="$RSS_LIMIT_MB"
)

fail() {
    printf '[FAIL] %s\n' "$*" >&2
    exit 1
}

validate_u64_strict() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || fail "invalid $name=$value"
    local max="18446744073709551615"
    if [[ "${#value}" -gt "${#max}" || ( "${#value}" -eq "${#max}" && "$value" > "$max" ) ]]; then
        fail "overflow $name=$value"
    fi
}

validate_i32_strict() {
    local name="$1"
    local value="$2"
    validate_u64_strict "$name" "$value"
    local max="2147483647"
    if [[ "${#value}" -gt "${#max}" || ( "${#value}" -eq "${#max}" && "$value" > "$max" ) ]]; then
        fail "$name=$value exceeds int32_t max"
    fi
}

normalize_digits() {
    local value="$1"
    value="${value#"${value%%[!0]*}"}"
    printf '%s\n' "${value:-0}"
}

extract_section() {
    local src="$1"
    local begin="$2"
    local end="$3"
    local dst="$4"
    awk -v begin="$begin" -v end="$end" '
        $0 == begin { in_section = 1; next }
        $0 == end { in_section = 0; next }
        in_section { print }
    ' "$src" > "$dst"
}

assert_marker_count() {
    local file="$1"
    local marker="$2"
    local actual
    actual="$(awk -v marker="$marker" '$0 == marker { count += 1 } END { print count + 0 }' "$file")"
    [[ "$actual" == "1" ]] || fail "expected marker $marker exactly once in $file, got $actual"
}

assert_nonempty_file() {
    local file="$1"
    [[ -s "$file" ]] || fail "empty extracted sequence: $file"
}

summary_value() {
    local file="$1"
    local key="$2"
    awk -v key="$key" '
        /KV_STABILITY_SUMMARY/ {
            for (i = 1; i <= NF; ++i) {
                split($i, kv, "=")
                if (kv[1] == key) {
                    print kv[2]
                    exit
                }
            }
        }
    ' "$file"
}

POLLUTION_ENV=(
    LLAMA_FLEX
    LLAMA_FLEX_AHEAD
    LLAMA_FLEX_AUTO
    LLAMA_FLEX_BUFFERED
    LLAMA_FLEX_DEBUG
    LLAMA_FLEX_LOCK_GB
    LLAMA_FLEX_RING
    LLAMA_FLEX_THREADS
    LLAMA_FLEX_TYPE
    LLAMA_FLEX_FACTOR
    LLAMA_FLEX_ORIG_CTX
    LLAMA_KV_IDLE_NUM_IDLE_SEQS
    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS
    LLAMA_KV_CACHE_DEBUG
    LLAMA_KV_LAZY_CLEAR
    LLAMA_KV_LAZY_TAIL
    LLAMA_KV_PAGED
    LLAMA_KV_PAGED_BLOCK_SIZE
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME
    LLAMA_KV_PAGED_GATHER_NONIDENTITY
    LLAMA_KV_PAGED_IDLE_SWAP
    LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES
    LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE
    LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP
    LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS
    LLAMA_KV_PAGED_IDLE_TRACE
    LLAMA_KV_PAGED_INGRAPH
    LLAMA_KV_PAGED_MINCORE
    LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED
    LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS
    LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS
    LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE
    LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS
    LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE
    LLAMA_KV_PAGED_REFAULT_TRACE
    LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE
    LLAMA_KV_PAGED_REFAULT_TRACE_MAX
    LLAMA_KV_PAGED_REFAULT_TRACE_ONCE
    LLAMA_KV_PAGED_RELEASE
    LLAMA_KV_PAGED_RESUME_PENDING_TOKEN
    LLAMA_KV_PAGED_RESUME_PREFETCH
    LLAMA_KV_PAGED_RESUME_TIMING
    LLAMA_KV_PAGED_RESUME_TIMING_STEP
    LLAMA_KV_PAGED_SHADOW_VALIDATE
    LLAMA_KV_PAGED_SHIFT
    LLAMA_KV_PAGED_SWAP
    LLAMA_KV_PAGED_TIMING
    LLAMA_KV_PAGED_TRACE
    LLAMA_KV_SWAP
    LLAMA_KV_SWAP_BACKEND_SELFTEST
    LLAMA_KV_SWAP_DIR
    LLAMA_KV_SWAP_MODE
    LLAMA_KV_SWAP_WINDOW
    LLAMA_KV_SWAP_SINK
    LLAMA_KV_SWAP_RSS_SAMPLE
    LLAMA_KV_SWAP_MADVISE
    LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE
    LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE
    LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE
    LLAMA_KV_TEST_RETRY_ACTIVE_DECODE
    LLAMA_KV_STABILITY_CYCLES
    LLAMA_KV_STABILITY_VERIFY_TOKENS
    LLAMA_KV_STABILITY_PROGRESS_EVERY
    LLAMA_KV_STABILITY_RSS_LIMIT_MB
)

run_case() {
    local name="$1"
    local cycles="$2"
    local dir="$OUTPUT_ROOT/$name"
    rm -rf "$dir"
    mkdir -p "$dir"

    (
        for key in "${POLLUTION_ENV[@]}"; do
            unset "$key"
        done
        export "${KV_ENV_COMMON[@]}"
        export LLAMA_KV_STABILITY_CYCLES="$cycles"
        timeout "$CASE_TIMEOUT_SEC" "$RUNNER" "${COMMON_ARGS[@]}"
    ) > "$dir/run.out" 2> "$dir/run.err"
    local status=$?
    printf '%s\n' "$status" > "$dir/run.exit"
}

assert_exit_zero() {
    local dir="$1"
    local status
    status="$(cat "$dir/run.exit")"
    [[ "$status" == "0" ]] || fail "$dir exited with $status"
}

assert_token_count() {
    local file="$1"
    local key="$2"
    local expected="$3"
    local actual
    actual="$(awk -F= -v key="$key" '$1 == key { print $2; exit }' "$file")"
    [[ "$actual" == "$expected" ]] || fail "expected $key=$expected in $file, got ${actual:-missing}"
}

main() {
    validate_u64_strict CYCLES "$CYCLES"
    validate_i32_strict VERIFY_TOKENS "$VERIFY_TOKENS"
    validate_u64_strict CASE_TIMEOUT_SEC "$CASE_TIMEOUT_SEC"
    validate_u64_strict RSS_LIMIT_MB "$RSS_LIMIT_MB"
    validate_u64_strict PROGRESS_EVERY "$PROGRESS_EVERY"
    validate_u64_strict SKIP_BUILD "$SKIP_BUILD"
    CYCLES="$(normalize_digits "$CYCLES")"
    VERIFY_TOKENS="$(normalize_digits "$VERIFY_TOKENS")"
    CASE_TIMEOUT_SEC="$(normalize_digits "$CASE_TIMEOUT_SEC")"
    RSS_LIMIT_MB="$(normalize_digits "$RSS_LIMIT_MB")"
    PROGRESS_EVERY="$(normalize_digits "$PROGRESS_EVERY")"
    SKIP_BUILD="$(normalize_digits "$SKIP_BUILD")"
    [[ "$CYCLES" != "0" ]] || fail "CYCLES must be > 0"

    [[ -f "$MODEL" ]] || fail "model not found: $MODEL"

    if [[ "$SKIP_BUILD" != "1" ]]; then
        cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release || fail "cmake configure failed"
        cmake --build "$BUILD_DIR" -j"$(nproc)" --target llama llama-kv-idle-swap-resume test-kv-backing-store || \
            fail "build failed"
    fi
    [[ -x "$RUNNER" ]] || fail "runner not found: $RUNNER"

    rm -rf "$OUTPUT_ROOT"
    mkdir -p "$OUTPUT_ROOT"

    local strings_file libllama
    strings_file="$(mktemp "$OUTPUT_ROOT/runner.strings.XXXXXX")" || fail "mktemp failed"
    strings "$RUNNER" > "$strings_file" || fail "strings failed for runner"
    for marker in KV_STABILITY_SUMMARY llama_kv_cache_paged_stability_cycle target_blocks backing_stat_valid; do
        grep -F -q -- "$marker" "$strings_file" || fail "runner missing string marker: $marker"
    done
    rm -f "$strings_file"

    libllama="$BUILD_DIR/bin/libllama.so"
    [[ -f "$libllama" ]] || fail "libllama not found: $libllama"
    strings_file="$(mktemp "$OUTPUT_ROOT/libllama.strings.XXXXXX")" || fail "mktemp failed"
    strings "$libllama" > "$strings_file" || fail "strings failed for libllama"
    for marker in KV_STABILITY_SUMMARY llama_kv_cache_paged_stability_cycle target_blocks backing_stat_valid; do
        grep -F -q -- "$marker" "$strings_file" || fail "libllama missing string marker: $marker"
    done
    rm -f "$strings_file"

    run_case control 0
    run_case stress "$CYCLES"

    assert_exit_zero "$OUTPUT_ROOT/control"
    assert_exit_zero "$OUTPUT_ROOT/stress"
    assert_token_count "$OUTPUT_ROOT/control/run.out" seq1_decoded_tokens "$VERIFY_TOKENS"
    assert_token_count "$OUTPUT_ROOT/control/run.out" seq0_resume_decoded_tokens "$VERIFY_TOKENS"
    assert_token_count "$OUTPUT_ROOT/stress/run.out" seq1_decoded_tokens "$VERIFY_TOKENS"
    assert_token_count "$OUTPUT_ROOT/stress/run.out" seq0_resume_decoded_tokens "$VERIFY_TOKENS"

    grep -q 'KV_STABILITY_SUMMARY' "$OUTPUT_ROOT/stress/run.err" || fail "missing KV_STABILITY_SUMMARY"

    local cycles_completed backing_changed fatal_delta pending swap_out_delta swap_in_delta rss_growth rss_limit
    local target_blocks backing_stat_valid backing_capacity backing_size_first backing_size_final
    cycles_completed="$(summary_value "$OUTPUT_ROOT/stress/run.err" cycles_completed)"
    backing_changed="$(summary_value "$OUTPUT_ROOT/stress/run.err" backing_size_changed)"
    fatal_delta="$(summary_value "$OUTPUT_ROOT/stress/run.err" fatal_counter_delta)"
    pending="$(summary_value "$OUTPUT_ROOT/stress/run.err" pending_error_count)"
    swap_out_delta="$(summary_value "$OUTPUT_ROOT/stress/run.err" swap_out_delta)"
    swap_in_delta="$(summary_value "$OUTPUT_ROOT/stress/run.err" swap_in_delta)"
    rss_growth="$(summary_value "$OUTPUT_ROOT/stress/run.err" rss_growth_kb)"
    rss_limit="$(summary_value "$OUTPUT_ROOT/stress/run.err" rss_limit_kb)"
    target_blocks="$(summary_value "$OUTPUT_ROOT/stress/run.err" target_blocks)"
    backing_stat_valid="$(summary_value "$OUTPUT_ROOT/stress/run.err" backing_stat_valid)"
    backing_capacity="$(summary_value "$OUTPUT_ROOT/stress/run.err" backing_capacity)"
    backing_size_first="$(summary_value "$OUTPUT_ROOT/stress/run.err" backing_size_first)"
    backing_size_final="$(summary_value "$OUTPUT_ROOT/stress/run.err" backing_size_final)"

    [[ "$cycles_completed" == "$CYCLES" ]] || fail "cycles_completed=$cycles_completed expected $CYCLES"
    [[ "${target_blocks:-0}" -gt 0 ]] || fail "target_blocks=$target_blocks expected > 0"
    [[ "$backing_stat_valid" == "1" ]] || fail "backing_stat_valid=$backing_stat_valid"
    [[ "${backing_capacity:-0}" -gt 0 ]] || fail "backing_capacity=$backing_capacity expected > 0"
    [[ "$backing_size_first" == "$backing_capacity" ]] || fail "backing_size_first=$backing_size_first backing_capacity=$backing_capacity"
    [[ "$backing_size_final" == "$backing_capacity" ]] || fail "backing_size_final=$backing_size_final backing_capacity=$backing_capacity"
    [[ "$backing_changed" == "0" ]] || fail "backing_size_changed=$backing_changed"
    [[ "$fatal_delta" == "0" ]] || fail "fatal_counter_delta=$fatal_delta"
    [[ "$pending" == "0" ]] || fail "pending_error_count=$pending"
    [[ "${swap_out_delta:-0}" -ge "$CYCLES" ]] || fail "swap_out_delta=$swap_out_delta expected >= $CYCLES"
    [[ "${swap_in_delta:-0}" -ge "$CYCLES" ]] || fail "swap_in_delta=$swap_in_delta expected >= $CYCLES"
    [[ "${rss_growth:-0}" -le "${rss_limit:-0}" ]] || fail "rss_growth_kb=$rss_growth rss_limit_kb=$rss_limit"

    local file marker
    for file in "$OUTPUT_ROOT/control/run.out" "$OUTPUT_ROOT/stress/run.out"; do
        for marker in \
            "===SEQ1_ACTIVE_BEGIN===" \
            "===SEQ1_ACTIVE_END===" \
            "===SEQ0_RESUME_BEGIN===" \
            "===SEQ0_RESUME_END==="; do
            assert_marker_count "$file" "$marker"
        done
    done

    extract_section "$OUTPUT_ROOT/control/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$OUTPUT_ROOT/control/seq1.txt"
    extract_section "$OUTPUT_ROOT/control/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$OUTPUT_ROOT/control/seq0.txt"
    extract_section "$OUTPUT_ROOT/stress/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$OUTPUT_ROOT/stress/seq1.txt"
    extract_section "$OUTPUT_ROOT/stress/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$OUTPUT_ROOT/stress/seq0.txt"
    assert_nonempty_file "$OUTPUT_ROOT/control/seq1.txt"
    assert_nonempty_file "$OUTPUT_ROOT/stress/seq1.txt"
    assert_nonempty_file "$OUTPUT_ROOT/control/seq0.txt"
    assert_nonempty_file "$OUTPUT_ROOT/stress/seq0.txt"

    local exact_seq1_match=0
    local exact_seq0_match=0
    if cmp -s "$OUTPUT_ROOT/control/seq1.txt" "$OUTPUT_ROOT/stress/seq1.txt"; then
        exact_seq1_match=1
    else
        diff -u "$OUTPUT_ROOT/control/seq1.txt" "$OUTPUT_ROOT/stress/seq1.txt" | head -n 100 >&2 || true
    fi
    if cmp -s "$OUTPUT_ROOT/control/seq0.txt" "$OUTPUT_ROOT/stress/seq0.txt"; then
        exact_seq0_match=1
    else
        diff -u "$OUTPUT_ROOT/control/seq0.txt" "$OUTPUT_ROOT/stress/seq0.txt" | head -n 100 >&2 || true
    fi

    [[ "$exact_seq1_match" == "1" ]] || fail "SEQ1_EXACT_MATCH=0"
    [[ "$exact_seq0_match" == "1" ]] || fail "SEQ0_EXACT_MATCH=0"

    printf 'KV_P0_STABILITY_PASS\n'
    printf 'cycles_completed=%s\n' "$cycles_completed"
    printf 'target_blocks=%s\n' "$target_blocks"
    printf 'backing_size_bounded=1\n'
    printf 'SEQ1_EXACT_MATCH=1\n'
    printf 'SEQ0_EXACT_MATCH=1\n'
    printf 'exact_seq1_match=1\n'
    printf 'exact_seq0_match=1\n'
}

main "$@"
