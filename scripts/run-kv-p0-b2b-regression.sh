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
MODEL_RUNS=0

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
    LLAMA_KV_PAGED_REFAULT_TRACE
    LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE
    LLAMA_KV_PAGED_REFAULT_TRACE_MAX
    LLAMA_KV_PAGED_REFAULT_TRACE_ONCE
    LLAMA_KV_PAGED_RELEASE
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
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE
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

mapping_fault_line_contains() {
    local file="$1"
    shift
    local faults
    faults="$(grep -F "TEST MAPPING FAULT INJECTION" "$file" 2>/dev/null || true)"
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

claimant_value() {
    local file="$1"
    local prefix="$2"
    local key="$3"
    grep -F "$prefix" "$file" 2>/dev/null | awk -v key="$key" '
        {
            for (i = 1; i <= NF; ++i) {
                split($i, kv, "=")
                if (kv[1] == key) {
                    print kv[2]
                    exit
                }
            }
        }
    '
}

assert_claimant_before_offload() {
    local file="$1"
    local prefix="KV_UNIFIED_STATE phase=seq0_before_offload"
    local valid target eligible swapped shared blocked
    valid="$(claimant_value "$file" "$prefix" valid)"
    target="$(claimant_value "$file" "$prefix" target_blocks)"
    eligible="$(claimant_value "$file" "$prefix" eligible_resident_blocks)"
    swapped="$(claimant_value "$file" "$prefix" swapped_blocks)"
    shared="$(claimant_value "$file" "$prefix" shared_blocks)"
    blocked="$(claimant_value "$file" "$prefix" blocked_blocks)"
    if ! [[ "$valid" == 1 && "$target" =~ ^[1-9][0-9]*$ &&
            "$eligible" == "$target" && "$swapped" == 0 &&
            "$shared" == 0 && "$blocked" == 0 ]]; then
        printf 'invalid pre-OFFLOAD claimant: valid=%s target=%s eligible=%s swapped=%s shared=%s blocked=%s' \
            "${valid:-missing}" "${target:-missing}" "${eligible:-missing}" "${swapped:-missing}" \
            "${shared:-missing}" "${blocked:-missing}"
        return 1
    fi
}

assert_claimant_after_offload() {
    local file="$1"
    local before="KV_UNIFIED_STATE phase=seq0_before_offload"
    local after="KV_UNIFIED_STATE phase=seq0_after_offload"
    local target_before eligible_before swapped_before shared_before blocked_before
    local valid_after target_after eligible_after swapped_after shared_after blocked_after
    target_before="$(claimant_value "$file" "$before" target_blocks)"
    eligible_before="$(claimant_value "$file" "$before" eligible_resident_blocks)"
    swapped_before="$(claimant_value "$file" "$before" swapped_blocks)"
    shared_before="$(claimant_value "$file" "$before" shared_blocks)"
    blocked_before="$(claimant_value "$file" "$before" blocked_blocks)"
    valid_after="$(claimant_value "$file" "$after" valid)"
    target_after="$(claimant_value "$file" "$after" target_blocks)"
    eligible_after="$(claimant_value "$file" "$after" eligible_resident_blocks)"
    swapped_after="$(claimant_value "$file" "$after" swapped_blocks)"
    shared_after="$(claimant_value "$file" "$after" shared_blocks)"
    blocked_after="$(claimant_value "$file" "$after" blocked_blocks)"
    if ! [[ "$valid_after" == 1 && "$target_before" =~ ^[1-9][0-9]*$ &&
            "$eligible_before" =~ ^[0-9]+$ && "$swapped_before" =~ ^[0-9]+$ &&
            "$target_after" == "$target_before" && "$eligible_after" =~ ^[0-9]+$ &&
            "$swapped_after" =~ ^[0-9]+$ && "$shared_after" == "$shared_before" &&
            "$blocked_after" == "$blocked_before" &&
            $((eligible_after + 1)) -eq "$eligible_before" &&
            $((swapped_after)) -eq $((swapped_before + 1)) ]]; then
        printf 'invalid OFFLOAD claimant transition'
        return 1
    fi
}

assert_claimant_equal() {
    local file="$1"
    local prefix_a="$2"
    local prefix_b="$3"
    local key a b
    for key in valid target_blocks eligible_resident_blocks swapped_blocks shared_blocks blocked_blocks; do
        a="$(claimant_value "$file" "$prefix_a" "$key")"
        b="$(claimant_value "$file" "$prefix_b" "$key")"
        if [[ -z "$a" || -z "$b" || "$a" != "$b" ]]; then
            printf 'claimant field %s changed between "%s" and "%s": %s vs %s' \
                "$key" "$prefix_a" "$prefix_b" "${a:-missing}" "${b:-missing}"
            return 1
        fi
    done
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

assert_mapping_fault_line() {
    local file="$1"
    shift
    if ! mapping_fault_line_contains "$file" "$@"; then
        printf 'missing single TEST MAPPING FAULT INJECTION line containing: %s' "$*"
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
    MODEL_RUNS=$((MODEL_RUNS + 1))
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
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=full seq=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_equal "$all" \
        "KV_UNIFIED_STATE phase=seq0_before_offload" \
        "KV_UNIFIED_STATE phase=seq0_after_prefetch")" || { report_fail "$name" "$reason"; return 1; }
    for needle in "TEST FAULT" "TEST MAPPING FAULT INJECTION" "KV_TEST_" "ret = -3"; do
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
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=full seq=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_UNIFIED_STATE phase=seq0_after_prefetch_failure valid=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_equal "$all" \
        "KV_UNIFIED_STATE phase=seq0_after_offload" \
        "KV_UNIFIED_STATE phase=seq0_after_prefetch_failure")" || { report_fail "$name" "$reason"; return 1; }
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
        "result=PASS" \
        "prefetch_failures_observed=1" \
        "active_decode_failures_observed=0" \
        "active_decode_retries=0" \
        "active_decode_retry_successes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_not_contains "$all" "TEST MAPPING FAULT INJECTION")" || { report_fail "$name" "$reason"; return 1; }
    for needle in "ret = -3"; do
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
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=skipped seq=0 reason=active_decode_failure")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "TEST FAULT INJECTION attempt_id=" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_fault_line "$all" \
        "scope=active" \
        "successful_cells_before_failure=0" \
        "backend_status=io_error" \
        "backend_errno=EIO(5)" \
        "failure_reason=SWAP_IN_IO_FAILURE" \
        "block_state=3" \
        "metadata_present_cells=16")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "llama_decode: failed to decode, ret = -3" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_EXPECTED_ACTIVE_DECODE_FAILURE" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_TERMINATING_AFTER_EXPECTED_FAILURE" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_summary_line "$all" \
        "active_decode_failures_observed=1" \
        "active_decode_retries=0" \
        "active_decode_retry_successes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_not_contains "$all" "TEST MAPPING FAULT INJECTION")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_not_contains "$all" "KV_TEST_RETRY_SAME_BATCH")" || { report_fail "$name" "$reason"; return 1; }

    report_pass "$name"
}

validate_case4() {
    local name="case4_active_atomic_retry_same_batch"
    local dir="$OUTPUT_ROOT/$name"
    local all
    all="$(case_file "$dir")"
    local reason

    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=skipped seq=0 reason=active_decode_failure")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "TEST FAULT INJECTION attempt_id=" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "llama_decode: failed to decode, ret = -3" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_fault_line "$all" \
        "scope=active" \
        "successful_cells_before_failure=0" \
        "backend_status=io_error" \
        "backend_errno=EIO(5)" \
        "failure_reason=SWAP_IN_IO_FAILURE" \
        "block_state=3" \
        "metadata_present_cells=16")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SAME_BATCH" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SUCCEEDED" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_summary_line "$all" \
        "result=PASS" \
        "active_decode_failures_observed=1" \
        "active_decode_retries=1" \
        "active_decode_retry_successes=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_not_contains "$all" "TEST MAPPING FAULT INJECTION")" || { report_fail "$name" "$reason"; return 1; }

    report_pass "$name"
}

validate_mapping_retry_case() {
    local name="$1"
    local scope="$2"
    local failure_reason="$3"
    local fatal_counter="$4"
    local dir="$OUTPUT_ROOT/$name"
    local all
    all="$(case_file "$dir")"
    local reason

    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=skipped seq=0 reason=active_decode_failure")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "TEST MAPPING FAULT INJECTION scope=$scope" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_mapping_fault_line "$all" \
        "scope=$scope" \
        "target_seq=0" \
        "failure_reason=$failure_reason" \
        "fatal_counter=$fatal_counter" \
        "fatal_counter_next=1" \
        "trigger_count=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "failure_reason=$failure_reason" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "llama_decode: failed to decode, ret = -3" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_EXPECTED_ACTIVE_DECODE_FAILURE" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SAME_BATCH" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SUCCEEDED" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_summary_line "$all" \
        "result=PASS" \
        "active_decode_failures_observed=1" \
        "active_decode_retries=1" \
        "active_decode_retry_successes=1")" || { report_fail "$name" "$reason"; return 1; }

    report_pass "$name"
}

validate_case5() {
    validate_mapping_retry_case \
        case5_read_mapping_fail_retry_same_batch \
        read \
        PAGED_ROW_MAPPING_INVALID \
        paged_row_mapping_invalid_fatal
}

validate_case6() {
    validate_mapping_retry_case \
        case6_write_mapping_fail_retry_same_batch \
        write \
        PAGED_WRITE_MAPPING_INVALID \
        paged_write_mapping_invalid_fatal
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
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=full seq=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_equal "$all" \
        "KV_UNIFIED_STATE phase=seq0_before_offload" \
        "KV_UNIFIED_STATE phase=seq0_after_prefetch")" || { report_fail "$name" "$reason"; return 1; }
    for needle in "TEST FAULT" "TEST MAPPING FAULT INJECTION" "KV_TEST_"; do
        reason="$(assert_not_contains "$all" "$needle")" || { report_fail "$name" "$reason"; return 1; }
    done

    report_control_pass "$name"
}

validate_exact_match_against_control() {
    local label="$1"
    local target_name="$2"
    local control="$OUTPUT_ROOT/case4_control_no_fault"
    local target="$OUTPUT_ROOT/$target_name"
    local reason

    for file in "$control/run.out" "$target/run.out"; do
        for marker in \
            "===SEQ1_ACTIVE_BEGIN===" \
            "===SEQ1_ACTIVE_END===" \
            "===SEQ0_RESUME_BEGIN===" \
            "===SEQ0_RESUME_END==="; do
            reason="$(assert_marker_count "$file" "$marker")" || {
                printf '[FAIL] %s: %s\n' "$label" "$reason" >&2
                return 1
            }
        done
    done

    extract_section "$control/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$control/seq1.txt"
    extract_section "$control/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$control/seq0.txt"
    extract_section "$target/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$target/seq1.txt"
    extract_section "$target/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$target/seq0.txt"

    for file in "$control/seq1.txt" "$control/seq0.txt" "$target/seq1.txt" "$target/seq0.txt"; do
        reason="$(assert_nonempty_file "$file")" || {
            printf '[FAIL] %s: %s\n' "$label" "$reason" >&2
            return 1
        }
    done

    if cmp -s "$control/seq1.txt" "$target/seq1.txt"; then
        EXACT_SEQ1_MATCH=1
    else
        printf '[FAIL] %s: SEQ1_EXACT_MATCH=0\n' "$label" >&2
        diff -u "$control/seq1.txt" "$target/seq1.txt" | head -n 100 >&2 || true
        return 1
    fi

    if cmp -s "$control/seq0.txt" "$target/seq0.txt"; then
        EXACT_SEQ0_MATCH=1
    else
        printf '[FAIL] %s: SEQ0_EXACT_MATCH=0\n' "$label" >&2
        diff -u "$control/seq0.txt" "$target/seq0.txt" | head -n 100 >&2 || true
        return 1
    fi
}

validate_case4_exact_match() {
    validate_exact_match_against_control case4_exact_match case4_active_atomic_retry_same_batch
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

    local libllama="$BUILD_DIR/bin/libllama.so"
    [[ -f "$libllama" ]] || fail_global "libllama not found: $libllama"

    strings_file="$(mktemp "$OUTPUT_ROOT/libllama.strings.XXXXXX")" || \
        fail_global "failed to create temporary strings file"
    strings "$libllama" > "$strings_file" || \
        fail_global "strings failed for libllama: $libllama"

    for marker in \
        "TEST MAPPING FAULT INJECTION" \
        "PAGED_ROW_MAPPING_INVALID" \
        "PAGED_WRITE_MAPPING_INVALID"; do
        grep -F -q -- "$marker" "$strings_file" || \
            fail_global "libllama missing string marker: $marker"
    done
    rm -f "$strings_file"
}

main() {
    mkdir -p "$OUTPUT_ROOT"
    check_prereqs

    local failed=0

    run_case case1_scope_off_normal \
        export "${KV_ENV_COMMON[@]}"
    validate_case1 || failed=1

    run_case case2_prefetch_first_cell_fail_once \
        export "${KV_ENV_COMMON[@]}" \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE=prefetch \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS=0 \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE=1
    validate_case2 || failed=1

    run_case case3_active_first_cell_fail \
        export "${KV_ENV_COMMON[@]}" \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE=active \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS=0 \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1 \
            LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=0
    validate_case3 || failed=1

    run_case case4_control_no_fault \
        export "${KV_ENV_COMMON[@]}"
    validate_case4_control || failed=1

    run_case case4_active_atomic_retry_same_batch \
        export "${KV_ENV_COMMON[@]}" \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE=active \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS=0 \
            LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1 \
            LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=1
    validate_case4 || failed=1

    validate_case4_exact_match || failed=1

    run_case case5_read_mapping_fail_retry_same_batch \
        export "${KV_ENV_COMMON[@]}" \
            LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE=read \
            LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID=0 \
            LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1 \
            LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=1
    validate_case5 || failed=1
    validate_exact_match_against_control case5_exact_match case5_read_mapping_fail_retry_same_batch || failed=1

    run_case case6_write_mapping_fail_retry_same_batch \
        export "${KV_ENV_COMMON[@]}" \
            LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE=write \
            LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID=0 \
            LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1 \
            LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=1
    validate_case6 || failed=1
    validate_exact_match_against_control case6_exact_match case6_write_mapping_fail_retry_same_batch || failed=1

    if [[ "$MODEL_RUNS" -ne 7 ]]; then
        report_fail final "expected MODEL_RUNS=7, got $MODEL_RUNS"
        failed=1
    fi
    if [[ "$CASES_PASSED" -ne 6 ]]; then
        report_fail final "expected CASES_PASSED=6, got $CASES_PASSED"
        failed=1
    fi

    if [[ "$failed" -ne 0 ]]; then
        exit 1
    fi

    printf 'KV_P0_B2B_REGRESSION_PASS\n'
    printf 'cases_passed=%d\n' "$CASES_PASSED"
    printf 'model_runs=%d\n' "$MODEL_RUNS"
    printf 'exact_seq1_match=%d\n' "$EXACT_SEQ1_MATCH"
    printf 'exact_seq0_match=%d\n' "$EXACT_SEQ0_MATCH"
    printf 'log_root=%s\n' "$OUTPUT_ROOT"
}

main "$@"
