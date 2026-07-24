#!/usr/bin/env bash
# Shared verdict, logging, command-capture and summary helpers.

set -euo pipefail

readonly EXIT_PASS=0
readonly EXIT_FAIL=1
readonly EXIT_UNRESOLVED=2
readonly EXIT_INCOMPLETE=3
readonly EXIT_NO_CHANGES=4
readonly EXIT_UNVERIFIED=5

declare GATE_LOG_LEVEL="${GATE_LOG_LEVEL:-1}"   # 0 quiet, 1 compact, 2 verbose
declare GATE_MODE="${GATE_MODE:-}"
declare GATE_PROFILE="${GATE_PROFILE:-}"
declare GATE_BUILD_DIR="${GATE_BUILD_DIR:-build}"
declare GATE_ARTIFACT_DIR=""
declare GATE_FULL_LOG=""
declare GATE_SUMMARY_FILE=""
declare GATE_START_EPOCH=0
declare GATE_FINGERPRINT=""
declare GATE_REUSED_FROM=""
declare GATE_HARNESS_VERSION="unknown"

declare -A CHECK_RESULTS=()
declare -A CHECK_DETAILS=()
declare -a CHECK_ORDER=()

gate_log() {
    local level="${1:-1}"
    local msg="${2:-}"
    if (( GATE_LOG_LEVEL >= level )); then
        printf '[gate] %s\n' "$msg" >&2
    fi
    if [[ -n "${GATE_FULL_LOG:-}" && -f "$GATE_FULL_LOG" ]]; then
        printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$msg" >> "$GATE_FULL_LOG"
    fi
}

gate_record_check() {
    local name="${1:-}"
    local verdict="${2:-}"
    local detail="${3:-}"

    if [[ -z "$name" ]]; then
        name="harness-invalid-check-$(( ${#CHECK_ORDER[@]} + 1 ))"
        verdict="INCOMPLETE"
        detail="check recorded without a name${detail:+; $detail}"
    fi
    case "$verdict" in
        PASS|FAIL|SKIP|UNRESOLVED|UNVERIFIED|INCOMPLETE|NO_CHANGES) ;;
        *)
            detail="invalid check verdict '${verdict:-<empty>}'${detail:+; $detail}"
            verdict="INCOMPLETE"
            ;;
    esac

    if [[ -n "${CHECK_RESULTS[$name]+present}" ]]; then
        local prev="${CHECK_RESULTS[$name]}"
        case "$prev" in
            FAIL|UNRESOLVED|INCOMPLETE) ;;
            *) CHECK_RESULTS["$name"]="INCOMPLETE" ;;
        esac
        CHECK_DETAILS["$name"]="duplicate record: previous=$prev new=$verdict"
        gate_log 1 "$name INCOMPLETE — duplicate check record"
        return 0
    fi

    CHECK_RESULTS["$name"]="$verdict"
    CHECK_DETAILS["$name"]="$detail"
    CHECK_ORDER+=("$name")
    gate_log 1 "$name $verdict${detail:+ — $detail}"
}

gate_verdict_name() {
    case "${1:-3}" in
        0) printf 'PASS' ;;
        1) printf 'FAIL' ;;
        2) printf 'UNRESOLVED' ;;
        3) printf 'INCOMPLETE' ;;
        4) printf 'NO_CHANGES' ;;
        5) printf 'UNVERIFIED' ;;
        *) printf 'UNKNOWN(%s)' "${1:-}" ;;
    esac
}

# Priority is deliberately fail-closed.
gate_final_verdict() {
    local has_pass=0 has_fail=0 has_unresolved=0 has_incomplete=0 has_unverified=0 has_any=0
    local name verdict
    for name in "${CHECK_ORDER[@]}"; do
        has_any=1
        verdict="${CHECK_RESULTS[$name]:-INCOMPLETE}"
        case "$verdict" in
            FAIL) has_fail=1 ;;
            UNRESOLVED) has_unresolved=1 ;;
            INCOMPLETE) has_incomplete=1 ;;
            UNVERIFIED) has_unverified=1 ;;
            PASS) has_pass=1 ;;
            SKIP|NO_CHANGES) ;;
            *) has_incomplete=1 ;;
        esac
    done
    (( has_fail )) && return "$EXIT_FAIL"
    (( has_unresolved )) && return "$EXIT_UNRESOLVED"
    (( has_incomplete )) && return "$EXIT_INCOMPLETE"
    (( has_unverified )) && return "$EXIT_UNVERIFIED"
    (( has_pass )) && return "$EXIT_PASS"
    (( has_any )) && return "$EXIT_NO_CHANGES"
    return "$EXIT_NO_CHANGES"
}

gate_count_result() {
    local target="$1" count=0 name
    for name in "${CHECK_ORDER[@]}"; do
        [[ "${CHECK_RESULTS[$name]:-}" == "$target" ]] && count=$((count + 1))
    done
    printf '%d' "$count"
}

gate_failure_excerpt() {
    local log_file="$1"
    [[ -s "$log_file" ]] || return 0
    (( GATE_LOG_LEVEL == 0 )) && return 0

    printf '[gate] failure excerpt: %s\n' "$log_file" >&2
    local first
    first=$(grep -nEm1 '(^|[^[:alpha:]])(fatal error|error:|FAILED|FAIL:|Traceback|AssertionError|undefined reference)' "$log_file" 2>/dev/null | cut -d: -f1 || true)
    if [[ "$first" =~ ^[0-9]+$ ]]; then
        local start=$(( first > 8 ? first - 8 : 1 ))
        local end=$(( first + 20 ))
        sed -n "${start},${end}p" "$log_file" >&2
    else
        tail -n 24 "$log_file" >&2
    fi
}

# Capture full command output in an artifact. PASS is silent unless --verbose.
gate_run_logged() {
    local log_file="$1"; shift
    mkdir -p "$(dirname "$log_file")"
    local rc=0
    if (( GATE_LOG_LEVEL >= 2 )); then
        set +e
        "$@" 2>&1 | tee "$log_file"
        rc=${PIPESTATUS[0]}
        set -e
    else
        set +e
        "$@" >"$log_file" 2>&1
        rc=$?
        set -e
    fi
    if (( rc != 0 )); then
        gate_failure_excerpt "$log_file"
    fi
    return "$rc"
}

gate_repo_root() {
    git rev-parse --show-toplevel 2>/dev/null || pwd
}

gate_emit_summary() {
    local final_code="$1" final_name duration
    final_name="$(gate_verdict_name "$final_code")"
    duration=$(( $(date +%s) - GATE_START_EPOCH ))

    local count_pass count_fail count_skip count_unresolved count_incomplete count_unverified
    count_pass="$(gate_count_result PASS)"
    count_fail="$(gate_count_result FAIL)"
    count_skip="$(gate_count_result SKIP)"
    count_unresolved="$(gate_count_result UNRESOLVED)"
    count_incomplete="$(gate_count_result INCOMPLETE)"
    count_unverified="$(gate_count_result UNVERIFIED)"

    cat > "$GATE_SUMMARY_FILE" <<SUMMARY
OS_AGENT_GATE_SUMMARY
harness_version=${GATE_HARNESS_VERSION}
mode=${GATE_MODE}
profile=${GATE_PROFILE}
verdict=${final_name}
exit_code=${final_code}
fingerprint=${GATE_FINGERPRINT:-none}
reused_from=${GATE_REUSED_FROM:-none}
artifact_dir=${GATE_ARTIFACT_DIR}
duration_seconds=${duration}
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
head=$(git rev-parse HEAD 2>/dev/null || echo unknown)
branch=$(git branch --show-current 2>/dev/null || echo unknown)
checks_total=${#CHECK_ORDER[@]}
checks_pass=${count_pass}
checks_fail=${count_fail}
checks_skip=${count_skip}
checks_unresolved=${count_unresolved}
checks_incomplete=${count_incomplete}
checks_unverified=${count_unverified}
---
SUMMARY
    local name
    for name in "${CHECK_ORDER[@]}"; do
        printf '%-28s %-12s %s\n' "$name" "${CHECK_RESULTS[$name]}" "${CHECK_DETAILS[$name]:-}" >> "$GATE_SUMMARY_FILE"
    done

    printf 'OS_AGENT_GATE_RESULT mode=%s profile=%s verdict=%s code=%d checks=%d pass=%d fail=%d skip=%d unresolved=%d incomplete=%d unverified=%d fingerprint=%s artifact=%s\n' \
        "$GATE_MODE" "$GATE_PROFILE" "$final_name" "$final_code" "${#CHECK_ORDER[@]}" \
        "$count_pass" "$count_fail" "$count_skip" "$count_unresolved" "$count_incomplete" \
        "$count_unverified" "${GATE_FINGERPRINT:-none}" "$GATE_ARTIFACT_DIR"
}
