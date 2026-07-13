#!/usr/bin/env bash

set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-kv-p0-b1}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/oscomp/kv_logs/kv_p0_b2b_regression}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-600}"

RUNNER="$BUILD_DIR/bin/llama-kv-idle-swap-resume"
CASES_PASSED=0
EXACT_SEQ1_MATCH=0
EXACT_SEQ0_MATCH=0

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
)

KV_ENV_ACTIVE_PREFETCH_ON=(
    LLAMA_KV_PAGED_RESUME_PREFETCH=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS=4
    LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP=1
    LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS=0
    LLAMA_KV_PAGED_RESUME_PENDING_TOKEN=96
    LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0
    LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=off
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=1
)

KV_ENV_ACTIVE_PREFETCH_OFF=(
    LLAMA_KV_PAGED_RESUME_PREFETCH=0
    LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE=0
    LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED=0
    LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS=0
    LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME=0
)

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
)

FAULT_ENV=(
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE
    LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE
    LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE
    LLAMA_KV_TEST_RETRY_ACTIVE_DECODE
)

fail_global() {
    printf '[FAIL] setup: %s\n' "$*" >&2
    exit 1
}

contains_file() {
    local file="$1"
    local needle="$2"
    grep -F -q -- "$needle" "$file" 2>/dev/null
}

count_file() {
    local file="$1"
    local needle="$2"
    grep -F -c -- "$needle" "$file" 2>/dev/null || true
}

summary_line_contains() {
    local file="$1"
    shift
    local summary
    summary="$(grep -F "KV_TEST_SUMMARY" "$file" 2>/dev/null || true)"
    [[ -n "$summary" ]] || return 1

    local line
    while IFS= read -r line; do
        local ok=1
        local needle
        for needle in "$@"; do
            if [[ "$line" != *"$needle"* ]]; then
                ok=0
                break
            fi
        done
        if [[ "$ok" -eq 1 ]]; then
            return 0
        fi
    done <<< "$summary"

    return 1
}

fault_line_contains() {
    local file="$1"
    shift
    local faults
    faults="$(grep -F "TEST FAULT INJECTION attempt_id=" "$file" 2>/dev/null || true)"
    [[ -n "$faults" ]] || return 1

    local line
    while IFS= read -r line; do
        local ok=1
        local needle
        for needle in "$@"; do
            if [[ "$line" != *"$needle"* ]]; then
                ok=0
                break
            fi
        done
        if [[ "$ok" -eq 1 ]]; then
            return 0
        fi
    done <<< "$faults"

    return 1
}

case_file() {
    local dir="$1"
    cat "$dir/run.out" "$dir/run.err" > "$dir/run.all"
    printf '%s\n' "$dir/run.all"
}

assert_exit() {
    local dir="$1"
    local expected="$2"
    local actual
    actual="$(cat "$dir/run.exit" 2>/dev/null || printf 'missing')"
    if [[ "$actual" != "$expected" ]]; then
        printf 'expected exit %s, got %s' "$expected" "$actual"
        return 1
    fi
}

assert_contains() {
    local file="$1"
    local needle="$2"
    if ! contains_file "$file" "$needle"; then
        printf 'missing "%s"' "$needle"
        return 1
    fi
}

assert_not_contains() {
    local file="$1"
    local needle="$2"
    if contains_file "$file" "$needle"; then
        printf 'unexpected "%s"' "$needle"
        return 1
    fi
}

assert_count() {
    local file="$1"
    local needle="$2"
    local expected="$3"
    local actual
    actual="$(count_file "$file" "$needle")"
    if [[ "$actual" != "$expected" ]]; then
        printf 'expected "%s" count %s, got %s' "$needle" "$expected" "$actual"
        return 1
    fi
}

assert_summary_line() {
    local file="$1"
    shift
    if ! summary_line_contains "$file" "$@"; then
        printf 'missing single KV_TEST_SUMMARY line containing: %s' "$*"
        return 1
    fi
}

assert_fault_line() {
    local file="$1"
    shift
    if ! fault_line_contains "$file" "$@"; then
        printf 'missing single TEST FAULT INJECTION line containing: %s' "$*"
        return 1
    fi
}

assert_tokens() {
    local file="$1"
    local key="$2"
    local expected="$3"
    local actual
    actual="$(awk -F= -v key="$key" '$1 == key { value = $2 } END { print value }' "$file")"
    if [[ "$actual" != "$expected" ]]; then
        printf 'expected %s=%s, got %s' "$key" "$expected" "${actual:-missing}"
        return 1
    fi
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
    if [[ "$actual" != "1" ]]; then
        printf 'expected marker "%s" exactly once in %s, got %s' "$marker" "$file" "$actual"
        return 1
    fi
}

assert_nonempty_file() {
    local file="$1"
    if [[ ! -s "$file" ]]; then
        printf 'expected non-empty extracted sequence: %s' "$file"
        return 1
    fi
}

run_case() {
    local name="$1"
    shift
    local dir="$OUTPUT_ROOT/$name"
    rm -rf "$dir"
    mkdir -p "$dir"

    (
        for key in "${POLLUTION_ENV[@]}" "${FAULT_ENV[@]}"; do
            unset "$key"
        done
        export "${KV_ENV_COMMON[@]}"
        "$@"
        timeout "$CASE_TIMEOUT_SEC" "$RUNNER" "${COMMON_ARGS[@]}"
    ) > "$dir/run.out" 2> "$dir/run.err"
    local status=$?
    printf '%s\n' "$status" > "$dir/run.exit"
}

report_pass() {
    local name="$1"
    printf '[PASS] %s\n' "$name"
    CASES_PASSED=$((CASES_PASSED + 1))
}

report_control_pass() {
    local name="$1"
    printf '[PASS] %s\n' "$name"
}

report_fail() {
    local name="$1"
    local reason="$2"
    printf '[FAIL] %s: %s\n' "$name" "$reason" >&2
}

validate_case1() {
    local name="case1_scope_off_normal"
    local dir="$OUTPUT_ROOT/$name"
    local all
    all="$(case_file "$dir")"
    local reason

    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    for needle in "TEST FAULT" "KV_TEST_" "ret = -3" "failed before graph_compute"; do
        reason="$(assert_not_contains "$all" "$needle")" || { report_fail "$name" "$reason"; return 1; }
    done

    report_pass "$name"
}

validate_case2() {
    local name="case2_prefetch_first_cell_fail_once"
    local dir="$OUTPUT_ROOT/$name"
    local all
    all="$(case_file "$dir")"
    local reason

    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "TEST FAULT INJECTION attempt_id=" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_fault_line "$all" \
        "scope=prefetch" \
        "successful_cells_before_failure=0" \
        "backend_status=io_error" \
        "backend_errno=EIO(5)" \
        "failure_reason=SWAP_IN_IO_FAILURE" \
        "block_state=3" \
        "metadata_present_cells=16")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_EXPECTED_PREFETCH_FAILURE" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_summary_line "$all" \
        "prefetch_failures_observed=1" \
        "active_decode_failures_observed=0" \
        "active_decode_retries=0" \
        "active_decode_retry_successes=0")" || { report_fail "$name" "$reason"; return 1; }
    for needle in "ret = -3" "failed before graph_compute"; do
        reason="$(assert_not_contains "$all" "$needle")" || { report_fail "$name" "$reason"; return 1; }
    done

    report_pass "$name"
}

validate_case3() {
    local name="case3_active_first_cell_fail"
    local dir="$OUTPUT_ROOT/$name"
    local all
    all="$(case_file "$dir")"
    local reason

    reason="$(assert_exit "$dir" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "TEST FAULT INJECTION attempt_id=" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_fault_line "$all" \
        "scope=active" \
        "successful_cells_before_failure=0" \
        "backend_status=io_error" \
        "backend_errno=EIO(5)" \
        "failure_reason=ACTIVE_VISIBLE_RESTORE_FAILURE" \
        "block_state=3" \
        "metadata_present_cells=16")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "failed before graph_compute")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "llama_decode: failed to decode, ret = -3" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_EXPECTED_ACTIVE_DECODE_FAILURE" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_TERMINATING_AFTER_EXPECTED_FAILURE" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_summary_line "$all" \
        "active_decode_failures_observed=1" \
        "active_decode_retries=0" \
        "active_decode_retry_successes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_not_contains "$all" "KV_TEST_RETRY_SAME_BATCH")" || { report_fail "$name" "$reason"; return 1; }

    report_pass "$name"
}

validate_case4() {
    local name="case4_active_partial_retry_same_batch"
    local dir="$OUTPUT_ROOT/$name"
    local all
    all="$(case_file "$dir")"
    local reason

    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "TEST FAULT INJECTION attempt_id=" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "llama_decode: failed to decode, ret = -3" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_fault_line "$all" \
        "scope=active" \
        "successful_cells_before_failure=1" \
        "backend_status=io_error" \
        "backend_errno=EIO(5)" \
        "failure_reason=ACTIVE_VISIBLE_RESTORE_FAILURE" \
        "block_state=3" \
        "metadata_present_cells=16")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SAME_BATCH" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SUCCEEDED" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_summary_line "$all" \
        "active_decode_failures_observed=1" \
        "active_decode_retries=1" \
        "active_decode_retry_successes=1")" || { report_fail "$name" "$reason"; return 1; }

    report_pass "$name"
}

validate_case4_control() {
    local name="case4_control_no_fault"
    local dir="$OUTPUT_ROOT/$name"
    local all
    all="$(case_file "$dir")"
    local reason

    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    for needle in "TEST FAULT" "KV_TEST_"; do
        reason="$(assert_not_contains "$all" "$needle")" || { report_fail "$name" "$reason"; return 1; }
    done

    report_control_pass "$name"
}

validate_case4_exact_match() {
    local control="$OUTPUT_ROOT/case4_control_no_fault"
    local retry="$OUTPUT_ROOT/case4_active_partial_retry_same_batch"
    local reason

    for file in "$control/run.out" "$retry/run.out"; do
        for marker in \
            "===SEQ1_ACTIVE_BEGIN===" \
            "===SEQ1_ACTIVE_END===" \
            "===SEQ0_RESUME_BEGIN===" \
            "===SEQ0_RESUME_END==="; do
            reason="$(assert_marker_count "$file" "$marker")" || {
                printf '[FAIL] case4_exact_match: %s\n' "$reason" >&2
                return 1
            }
        done
    done

    extract_section "$control/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$control/seq1.txt"
    extract_section "$control/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$control/seq0.txt"
    extract_section "$retry/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$retry/seq1.txt"
    extract_section "$retry/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$retry/seq0.txt"

    for file in "$control/seq1.txt" "$control/seq0.txt" "$retry/seq1.txt" "$retry/seq0.txt"; do
        reason="$(assert_nonempty_file "$file")" || {
            printf '[FAIL] case4_exact_match: %s\n' "$reason" >&2
            return 1
        }
    done

    if cmp -s "$control/seq1.txt" "$retry/seq1.txt"; then
        EXACT_SEQ1_MATCH=1
    else
        printf '[FAIL] case4_exact_match: SEQ1_EXACT_MATCH=0\n' >&2
        diff -u "$control/seq1.txt" "$retry/seq1.txt" | head -n 100 >&2 || true
        return 1
    fi

    if cmp -s "$control/seq0.txt" "$retry/seq0.txt"; then
        EXACT_SEQ0_MATCH=1
    else
        printf '[FAIL] case4_exact_match: SEQ0_EXACT_MATCH=0\n' >&2
        diff -u "$control/seq0.txt" "$retry/seq0.txt" | head -n 100 >&2 || true
        return 1
    fi
}

check_prereqs() {
    [[ -f "$MODEL" ]] || fail_global "model not found: $MODEL"
    [[ -d "$BUILD_DIR" ]] || fail_global "build dir not found: $BUILD_DIR"

    cmake --build "$BUILD_DIR" -j"$(nproc)" --target llama-kv-idle-swap-resume || \
        fail_global "build failed"

    [[ -x "$RUNNER" ]] || fail_global "runner not found or not executable: $RUNNER"

    local strings_file
    strings_file="$(mktemp "$OUTPUT_ROOT/runner.strings.XXXXXX")" || \
        fail_global "failed to create temporary strings file"
    strings "$RUNNER" > "$strings_file" || \
        fail_global "strings failed for runner: $RUNNER"

    for marker in \
        KV_TEST_EXPECTED_PREFETCH_FAILURE \
        KV_TEST_RETRY_SAME_BATCH \
        KV_TEST_SUMMARY; do
        grep -F -q -- "$marker" "$strings_file" || \
            fail_global "runner missing string marker: $marker"
    done
    rm -f "$strings_file"
}

main() {
    mkdir -p "$OUTPUT_ROOT"
    check_prereqs

    local failed=0

    run_case case1_scope_off_normal \
        export "${KV_ENV_ACTIVE_PREFETCH_ON[@]}"
    validate_case1 || failed=1

    run_case case2_prefetch_first_cell_fail_once \
        export "${KV_ENV_ACTIVE_PREFETCH_ON[@]}" \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE=prefetch \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS=0 \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE=1
    validate_case2 || failed=1

    run_case case3_active_first_cell_fail \
        export "${KV_ENV_ACTIVE_PREFETCH_OFF[@]}" \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE=active \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS=0 \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1 \
            LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=0
    validate_case3 || failed=1

    run_case case4_control_no_fault \
        export "${KV_ENV_ACTIVE_PREFETCH_OFF[@]}"
    validate_case4_control || failed=1

    run_case case4_active_partial_retry_same_batch \
        export "${KV_ENV_ACTIVE_PREFETCH_OFF[@]}" \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE=active \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS=1 \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1 \
            LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=1
    validate_case4 || failed=1

    validate_case4_exact_match || failed=1

    if [[ "$failed" -ne 0 ]]; then
        exit 1
    fi

    printf 'KV_P0_B2B_REGRESSION_PASS\n'
    printf 'cases_passed=%d\n' "$CASES_PASSED"
    printf 'exact_seq1_match=%d\n' "$EXACT_SEQ1_MATCH"
    printf 'exact_seq0_match=%d\n' "$EXACT_SEQ0_MATCH"
    printf 'log_root=%s\n' "$OUTPUT_ROOT"
}

main "$@"
