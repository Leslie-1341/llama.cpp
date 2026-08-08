#!/usr/bin/env bash

set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-kv-p0-b1}"
MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/oscomp/kv_logs/kv_p0_io_fault_regression}"
CASE_TIMEOUT_SEC="${CASE_TIMEOUT_SEC:-600}"

RUNNER="$BUILD_DIR/bin/llama-kv-idle-swap-resume"
CASES_PASSED=0
MODEL_RUNS=0
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
    LLAMA_KV_IDLE_NUM_IDLE_SEQS
    LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS
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
    LLAMA_KV_PAGED_RELEASE
    LLAMA_KV_PAGED_RESUME_TIMING
    LLAMA_KV_PAGED_RESUME_TIMING_STEP
    LLAMA_KV_PAGED_SWAP
    LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_IO_FAIL_KIND
    LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID
    LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK
    LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID
    LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS
    LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE
    LLAMA_KV_STABILITY_CYCLES
    LLAMA_KV_STABILITY_PROGRESS_EVERY
    LLAMA_KV_STABILITY_RSS_LIMIT_MB
    LLAMA_KV_STABILITY_VERIFY_TOKENS
    LLAMA_KV_SWAP
    LLAMA_KV_TEST_EXPECT_SWAP_OUT_IO_FAILURE
    LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE
    LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE
    LLAMA_KV_TEST_RETRY_ACTIVE_DECODE
    LLAMA_KV_SWAP_DIR
    LLAMA_KV_SWAP_MADVISE
    LLAMA_KV_SWAP_MODE
    LLAMA_KV_SWAP_RSS_SAMPLE
    LLAMA_KV_SWAP_SINK
    LLAMA_KV_SWAP_WINDOW
)

fail_global() {
    printf '[FAIL] setup: %s\n' "$*" >&2
    exit 1
}

contains_file() {
    grep -F -q -- "$2" "$1" 2>/dev/null
}

count_file() {
    grep -F -c -- "$2" "$1" 2>/dev/null || true
}

case_file() {
    local dir="$1"
    cat "$dir/run.out" "$dir/run.err" > "$dir/run.all"
    printf '%s\n' "$dir/run.all"
}

assert_exit() {
    local actual
    actual="$(cat "$1/run.exit" 2>/dev/null || printf 'missing')"
    [[ "$actual" == "$2" ]] || { printf 'expected exit %s, got %s' "$2" "$actual"; return 1; }
}

assert_contains() {
    contains_file "$1" "$2" || { printf 'missing "%s"' "$2"; return 1; }
}

assert_not_contains() {
    ! contains_file "$1" "$2" || { printf 'unexpected "%s"' "$2"; return 1; }
}

assert_count() {
    local actual
    actual="$(count_file "$1" "$2")"
    [[ "$actual" == "$3" ]] || { printf 'expected "%s" count %s, got %s' "$2" "$3" "$actual"; return 1; }
}

assert_line_contains() {
    local file="$1"
    local prefix="$2"
    shift 2
    local lines
    lines="$(grep -F "$prefix" "$file" 2>/dev/null || true)"
    [[ -n "$lines" ]] || { printf 'missing line prefix "%s"' "$prefix"; return 1; }
    local line
    while IFS= read -r line; do
        local ok=1
        local needle
        for needle in "$@"; do
            [[ "$line" == *"$needle"* ]] || { ok=0; break; }
        done
        [[ "$ok" -eq 1 ]] && return 0
    done <<< "$lines"
    printf 'missing "%s" line containing: %s' "$prefix" "$*"
    return 1
}

line_value() {
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

assert_equal_fields() {
    local file="$1"
    local prefix="$2"
    local key_a="$3"
    local key_b="$4"
    local a
    local b
    a="$(line_value "$file" "$prefix" "$key_a")"
    b="$(line_value "$file" "$prefix" "$key_b")"
    [[ -n "$a" && -n "$b" && "$a" == "$b" ]] || {
        printf 'expected %s == %s on "%s", got %s vs %s' "$key_a" "$key_b" "$prefix" "${a:-missing}" "${b:-missing}"
        return 1
    }
}

assert_increased_fields() {
    local file="$1"
    local prefix="$2"
    local key_before="$3"
    local key_after="$4"
    local before
    local after
    before="$(line_value "$file" "$prefix" "$key_before")"
    after="$(line_value "$file" "$prefix" "$key_after")"
    [[ -n "$before" && -n "$after" && "$after" =~ ^[0-9]+$ && "$before" =~ ^[0-9]+$ && "$after" -gt "$before" ]] || {
        printf 'expected %s > %s on "%s", got %s vs %s' "$key_after" "$key_before" "${prefix}" "${after:-missing}" "${before:-missing}"
        return 1
    }
}

assert_equal_field_between() {
    local file="$1"
    local prefix_a="$2"
    local key_a="$3"
    local prefix_b="$4"
    local key_b="$5"
    local a
    local b
    a="$(line_value "$file" "$prefix_a" "$key_a")"
    b="$(line_value "$file" "$prefix_b" "$key_b")"
    [[ -n "$a" && -n "$b" && "$a" == "$b" ]] || {
        printf 'expected %s/%s == %s/%s, got %s vs %s' "$prefix_a" "$key_a" "$prefix_b" "$key_b" "${a:-missing}" "${b:-missing}"
        return 1
    }
}

assert_claimant_before_offload() {
    local file="$1"
    local prefix="KV_UNIFIED_STATE phase=seq0_before_offload"
    local valid target eligible swapped shared blocked
    valid="$(line_value "$file" "$prefix" valid)"
    target="$(line_value "$file" "$prefix" target_blocks)"
    eligible="$(line_value "$file" "$prefix" eligible_resident_blocks)"
    swapped="$(line_value "$file" "$prefix" swapped_blocks)"
    shared="$(line_value "$file" "$prefix" shared_blocks)"
    blocked="$(line_value "$file" "$prefix" blocked_blocks)"
    [[ "$valid" == 1 && "$target" =~ ^[1-9][0-9]*$ &&
        "$eligible" == "$target" && "$swapped" == 0 &&
        "$shared" == 0 && "$blocked" == 0 ]] || {
        printf 'invalid pre-OFFLOAD claimant: valid=%s target=%s eligible=%s swapped=%s shared=%s blocked=%s' \
            "${valid:-missing}" "${target:-missing}" "${eligible:-missing}" "${swapped:-missing}" \
            "${shared:-missing}" "${blocked:-missing}"
        return 1
    }
}

assert_claimant_after_offload() {
    local file="$1"
    local before="KV_UNIFIED_STATE phase=seq0_before_offload"
    local after="KV_UNIFIED_STATE phase=seq0_after_offload"
    local target_before eligible_before swapped_before shared_before blocked_before
    local valid_after target_after eligible_after swapped_after shared_after blocked_after
    target_before="$(line_value "$file" "$before" target_blocks)"
    eligible_before="$(line_value "$file" "$before" eligible_resident_blocks)"
    swapped_before="$(line_value "$file" "$before" swapped_blocks)"
    shared_before="$(line_value "$file" "$before" shared_blocks)"
    blocked_before="$(line_value "$file" "$before" blocked_blocks)"
    valid_after="$(line_value "$file" "$after" valid)"
    target_after="$(line_value "$file" "$after" target_blocks)"
    eligible_after="$(line_value "$file" "$after" eligible_resident_blocks)"
    swapped_after="$(line_value "$file" "$after" swapped_blocks)"
    shared_after="$(line_value "$file" "$after" shared_blocks)"
    blocked_after="$(line_value "$file" "$after" blocked_blocks)"
    [[ "$valid_after" == 1 && "$target_before" =~ ^[1-9][0-9]*$ &&
        "$eligible_before" =~ ^[0-9]+$ && "$swapped_before" =~ ^[0-9]+$ &&
        "$target_after" == "$target_before" && "$eligible_after" =~ ^[0-9]+$ &&
        "$swapped_after" =~ ^[0-9]+$ && "$shared_after" == "$shared_before" &&
        "$blocked_after" == "$blocked_before" &&
        $((eligible_after + 1)) -eq "$eligible_before" &&
        $((swapped_after)) -eq $((swapped_before + 1)) ]] || {
        printf 'invalid OFFLOAD claimant transition'
        return 1
    }
}

assert_claimant_equal() {
    local file="$1"
    local prefix_a="$2"
    local prefix_b="$3"
    local key a b
    for key in valid target_blocks eligible_resident_blocks swapped_blocks shared_blocks blocked_blocks; do
        a="$(line_value "$file" "$prefix_a" "$key")"
        b="$(line_value "$file" "$prefix_b" "$key")"
        [[ -n "$a" && -n "$b" && "$a" == "$b" ]] || {
            printf 'claimant field %s changed between "%s" and "%s": %s vs %s' \
                "$key" "$prefix_a" "$prefix_b" "${a:-missing}" "${b:-missing}"
            return 1
        }
    done
}

assert_tokens() {
    local actual
    actual="$(awk -F= -v key="$2" '$1 == key { value = $2 } END { print value }' "$1")"
    [[ "$actual" == "$3" ]] || { printf 'expected %s=%s, got %s' "$2" "$3" "${actual:-missing}"; return 1; }
}

assert_marker_once() {
    local file="$1"
    local marker="$2"
    local actual
    actual="$(grep -F -x -c -- "$marker" "$file" 2>/dev/null || true)"
    [[ "$actual" == 1 ]] || { printf 'expected marker "%s" count 1 in %s, got %s' "$marker" "$file" "$actual"; return 1; }
}

extract_section() {
    awk -v begin="$2" -v end="$3" '
        $0 == begin { in_section = 1; next }
        $0 == end { in_section = 0; next }
        in_section { print }
    ' "$1" > "$4"
}

run_case() {
    local name="$1"
    shift
    local dir="$OUTPUT_ROOT/$name"
    MODEL_RUNS=$((MODEL_RUNS + 1))
    rm -rf "$dir"
    mkdir -p "$dir"
    (
        for key in "${POLLUTION_ENV[@]}"; do
            unset "$key"
        done
        export "${KV_ENV_COMMON[@]}"
        "$@"
        timeout "$CASE_TIMEOUT_SEC" "$RUNNER" "${COMMON_ARGS[@]}"
    ) > "$dir/run.out" 2> "$dir/run.err"
    printf '%s\n' "$?" > "$dir/run.exit"
}

report_pass() {
    printf '[PASS] %s\n' "$1"
    CASES_PASSED=$((CASES_PASSED + 1))
}

report_fail() {
    printf '[FAIL] %s: %s\n' "$1" "$2" >&2
}

validate_control() {
    local name="case1_control_no_fault"
    local dir="$OUTPUT_ROOT/$name"
    local all
    local reason
    all="$(case_file "$dir")"
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
    reason="$(assert_not_contains "$all" "KV_PAGED_IO_FAULT")" || { report_fail "$name" "$reason"; return 1; }
    report_pass "$name"
}

validate_swap_out_enospc() {
    local name="case2_swap_out_write_enospc_once"
    local dir="$OUTPUT_ROOT/$name"
    local all
    local reason
    all="$(case_file "$dir")"
    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=failed reason=io_failure state_changed=0 fail_stop=0 io_failure=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_equal "$all" \
        "KV_UNIFIED_STATE phase=seq0_before_offload" \
        "KV_UNIFIED_STATE phase=seq0_after_failed_offload")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload_retry action=offload decision_id=2 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=full seq=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_equal "$all" \
        "KV_UNIFIED_STATE phase=seq0_before_offload" \
        "KV_UNIFIED_STATE phase=seq0_after_prefetch")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_PAGED_IO_FAULT scope=swap_out kind=write_enospc_once" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_line_contains "$all" "KV_PAGED_IO_FAULT scope=swap_out" \
        "target_seq=0" \
        "state_before=RESIDENT" \
        "state_after=RESIDENT" \
        "swap_out_counter_before=" \
        "madvise_counter_before=" \
        "backend_status=io_error" \
        "backend_errno=28" \
        "trigger_count=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_equal_fields "$all" "KV_PAGED_IO_FAULT scope=swap_out" swap_out_counter_before swap_out_counter_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_equal_fields "$all" "KV_PAGED_IO_FAULT scope=swap_out" madvise_counter_before madvise_counter_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_increased_fields "$all" "KV_PAGED_IO_FAULT scope=swap_out" pwrite_attempts_before pwrite_attempts_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=swap_out kind=write_enospc_once" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_line_contains "$all" "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=swap_out" \
        "state_before=RESIDENT" \
        "state_after=SWAPPED" \
        "fault_attempt_id=1" \
        "retry_success_count=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_equal_field_between "$all" \
        "KV_PAGED_IO_FAULT scope=swap_out" physical_block \
        "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=swap_out" physical_block)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_equal_field_between "$all" \
        "KV_PAGED_IO_FAULT scope=swap_out" attempt_id \
        "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=swap_out" fault_attempt_id)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_increased_fields "$all" "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=swap_out" swap_out_counter_before swap_out_counter_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_increased_fields "$all" "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=swap_out" madvise_counter_before madvise_counter_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_not_contains "$all" "llama_decode: failed to decode, ret = -3")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_line_contains "$all" "KV_TEST_SUMMARY" "expected_swap_out_io_failure=1" "active_decode_failures_observed=0")" || { report_fail "$name" "$reason"; return 1; }
    report_pass "$name"
}

validate_active_swap_in_eof() {
    local name="case3_active_swap_in_read_eof_once"
    local dir="$OUTPUT_ROOT/$name"
    local all
    local reason
    all="$(case_file "$dir")"
    reason="$(assert_exit "$dir" 0)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq1_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_tokens "$dir/run.out" seq0_resume_decoded_tokens 128)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_UNIFIED_ACTION phase=seq0_offload action=offload decision_id=1 outcome=completed reason=target_satisfied state_changed=1 fail_stop=0 io_failure=0 io_errno=0 blocks=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_before_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_claimant_after_offload "$all")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "shortfall_bytes=0")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_contains "$all" "KV_RESUME_PREFETCH mode=skipped seq=0 reason=active_decode_failure")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_PAGED_IO_FAULT scope=active_swap_in kind=read_eof_once" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_line_contains "$all" "KV_PAGED_IO_FAULT scope=active_swap_in" \
        "target_seq=0" \
        "state_before=SWAPPED" \
        "state_after=SWAPPED" \
        "backend_status=io_error" \
        "backend_errno=5" \
        "metadata_present_cells=16" \
        "failure_reason=PAGED_SWAP_IN_IO_ERROR" \
        "trigger_count=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_increased_fields "$all" "KV_PAGED_IO_FAULT scope=active_swap_in" pread_attempts_before pread_attempts_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_equal_fields "$all" "KV_PAGED_IO_FAULT scope=active_swap_in" swap_in_counter_before swap_in_counter_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "llama_decode: failed to decode, ret = -3" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SAME_BATCH" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_TEST_RETRY_SUCCEEDED" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_count "$all" "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=active_swap_in kind=read_eof_once" 1)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_line_contains "$all" "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=active_swap_in" \
        "state_before=SWAPPED" \
        "state_after=RESIDENT" \
        "fault_attempt_id=1" \
        "retry_success_count=1")" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_equal_field_between "$all" \
        "KV_PAGED_IO_FAULT scope=active_swap_in" physical_block \
        "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=active_swap_in" physical_block)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_equal_field_between "$all" \
        "KV_PAGED_IO_FAULT scope=active_swap_in" attempt_id \
        "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=active_swap_in" fault_attempt_id)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_increased_fields "$all" "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=active_swap_in" swap_in_counter_before swap_in_counter_after)" || { report_fail "$name" "$reason"; return 1; }
    reason="$(assert_line_contains "$all" "KV_TEST_SUMMARY" \
        "result=PASS" \
        "active_decode_failures_observed=1" \
        "active_decode_retries=1" \
        "active_decode_retry_successes=1")" || { report_fail "$name" "$reason"; return 1; }
    report_pass "$name"
}

compare_exact() {
    local target_name="$1"
    local control="$OUTPUT_ROOT/case1_control_no_fault"
    local target="$OUTPUT_ROOT/$target_name"
    local reason
    for marker in \
        "===SEQ1_ACTIVE_BEGIN===" \
        "===SEQ1_ACTIVE_END===" \
        "===SEQ0_RESUME_BEGIN===" \
        "===SEQ0_RESUME_END==="; do
        reason="$(assert_marker_once "$control/run.out" "$marker")" || { report_fail "exact_match_$target_name" "$reason"; return 1; }
        reason="$(assert_marker_once "$target/run.out" "$marker")" || { report_fail "exact_match_$target_name" "$reason"; return 1; }
    done
    extract_section "$control/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$control/seq1.txt"
    extract_section "$control/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$control/seq0.txt"
    extract_section "$target/run.out" "===SEQ1_ACTIVE_BEGIN===" "===SEQ1_ACTIVE_END===" "$target/seq1.txt"
    extract_section "$target/run.out" "===SEQ0_RESUME_BEGIN===" "===SEQ0_RESUME_END===" "$target/seq0.txt"
    [[ -s "$control/seq1.txt" && -s "$control/seq0.txt" && -s "$target/seq1.txt" && -s "$target/seq0.txt" ]] || {
        printf '[FAIL] exact_match_%s: extracted sequence is empty\n' "$target_name" >&2
        return 1
    }
    if cmp -s "$control/seq1.txt" "$target/seq1.txt"; then
        EXACT_SEQ1_MATCH=1
    else
        reason="SEQ1_EXACT_MATCH=0"
        report_fail "exact_match_$target_name" "$reason"
        return 1
    fi
    if cmp -s "$control/seq0.txt" "$target/seq0.txt"; then
        EXACT_SEQ0_MATCH=1
    else
        reason="SEQ0_EXACT_MATCH=0"
        report_fail "exact_match_$target_name" "$reason"
        return 1
    fi
}

check_prereqs() {
    [[ -f "$MODEL" ]] || fail_global "model not found: $MODEL"
    [[ -d "$BUILD_DIR" ]] || fail_global "build dir not found: $BUILD_DIR"
    cmake --build "$BUILD_DIR" -j"$(nproc)" --target llama-kv-idle-swap-resume || fail_global "build failed"
    [[ -x "$RUNNER" ]] || fail_global "runner not executable: $RUNNER"
}

main() {
    mkdir -p "$OUTPUT_ROOT"
    check_prereqs
    local failed=0

    run_case case1_control_no_fault true
    validate_control || failed=1

    run_case case2_swap_out_write_enospc_once \
        export LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE=swap_out \
            LLAMA_KV_PAGED_TEST_IO_FAIL_KIND=write_enospc_once \
            LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID=0 \
            LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_SWAP_OUT_IO_FAILURE=1
    validate_swap_out_enospc || failed=1
    compare_exact case2_swap_out_write_enospc_once || failed=1

    run_case case3_active_swap_in_read_eof_once \
        export LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE=active_swap_in \
            LLAMA_KV_PAGED_TEST_IO_FAIL_KIND=read_eof_once \
            LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID=0 \
            LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE=1 \
            LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1 \
            LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=1
    validate_active_swap_in_eof || failed=1
    compare_exact case3_active_swap_in_read_eof_once || failed=1

    [[ "$MODEL_RUNS" -eq 3 ]] || { report_fail final "expected MODEL_RUNS=3, got $MODEL_RUNS"; failed=1; }
    [[ "$CASES_PASSED" -eq 3 ]] || { report_fail final "expected CASES_PASSED=3, got $CASES_PASSED"; failed=1; }

    [[ "$failed" -eq 0 ]] || exit 1

    printf 'KV_P0_IO_FAULT_REGRESSION_PASS\n'
    printf 'cases_passed=%d\n' "$CASES_PASSED"
    printf 'model_runs=%d\n' "$MODEL_RUNS"
    printf 'exact_seq1_match=%d\n' "$EXACT_SEQ1_MATCH"
    printf 'exact_seq0_match=%d\n' "$EXACT_SEQ0_MATCH"
    printf 'log_root=%s\n' "$OUTPUT_ROOT"
}

main "$@"
