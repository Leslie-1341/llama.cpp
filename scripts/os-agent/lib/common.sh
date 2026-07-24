#!/usr/bin/env bash
# common.sh — shared constants, exit codes, logging, and utility functions
# Sourced by gate-runner and all check/gate modules.

set -euo pipefail

# --- exit codes (fail-closed: only 0 is PASS) ---
readonly EXIT_PASS=0
readonly EXIT_FAIL=1
readonly EXIT_UNRESOLVED=2
readonly EXIT_INCOMPLETE=3
readonly EXIT_NO_CHANGES=4
readonly EXIT_UNVERIFIED=5

# --- log level ---
declare GATE_LOG_LEVEL="${GATE_LOG_LEVEL:-1}"  # 0=quiet 1=normal 2=verbose

# --- artifact state ---
declare GATE_ARTIFACT_DIR=""
declare GATE_FULL_LOG=""
declare GATE_SUMMARY_FILE=""
declare GATE_START_TIME=""
declare GATE_MODE=""

# --- per-check tracking ---
declare -A CHECK_RESULTS
declare -a CHECK_ORDER

gate_log() {
    local level="${1:-1}"
    local msg="${2:-}"
    if (( GATE_LOG_LEVEL >= level )); then
        printf '[gate] %s\n' "$msg" >&2
    fi
    if [[ -n "${GATE_FULL_LOG:-}" ]] && [[ -f "$GATE_FULL_LOG" ]]; then
        printf '[%(%Y-%m-%dT%H:%M:%S)T] %s\n' -1 "$msg" >> "$GATE_FULL_LOG"
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

    # Duplicate names make counts and authority ambiguous. Fail closed without
    # overwriting an earlier FAIL.
    if [[ -n "${CHECK_RESULTS[$name]+present}" ]]; then
        local previous="${CHECK_RESULTS[$name]}"
        case "$previous" in
            FAIL|UNRESOLVED|INCOMPLETE) ;;
            *) CHECK_RESULTS["$name"]="INCOMPLETE" ;;
        esac
        if [[ -n "${GATE_FULL_LOG:-}" ]] && [[ -f "$GATE_FULL_LOG" ]]; then
            printf '[%s] %s: %s — duplicate check record (previous=%s, new=%s)\n' \
                "$(date +%H:%M:%S)" "$name" "${CHECK_RESULTS[$name]}" "$previous" "$verdict" >> "$GATE_FULL_LOG"
        fi
        return 0
    fi

    CHECK_RESULTS["$name"]="$verdict"
    CHECK_ORDER+=("$name")
    if [[ -n "${GATE_FULL_LOG:-}" ]] && [[ -f "$GATE_FULL_LOG" ]]; then
        printf '[%s] %s: %s%s\n' "$(date +%H:%M:%S)" "$name" "$verdict" "${detail:+ — $detail}" >> "$GATE_FULL_LOG"
    fi
}

gate_verdict_name() {
    local code="$1"
    case "$code" in
        0) printf 'PASS' ;;
        1) printf 'FAIL' ;;
        2) printf 'UNRESOLVED' ;;
        3) printf 'INCOMPLETE' ;;
        4) printf 'NO_CHANGES' ;;
        5) printf 'UNVERIFIED' ;;
        *) printf 'UNKNOWN(%d)' "$code" ;;
    esac
}

# Priority: FAIL > UNRESOLVED > INCOMPLETE > UNVERIFIED > PASS > NO_CHANGES.
# SKIP means not applicable and is neutral. UNVERIFIED means a required
# evidence layer was not executed or could not be established.
gate_final_verdict() {
    local has_pass=0 has_fail=0 has_unresolved=0 has_incomplete=0 has_unverified=0 has_any=0
    local name verdict

    for name in "${CHECK_ORDER[@]}"; do
        has_any=1
        verdict="${CHECK_RESULTS[$name]:-INCOMPLETE}"
        case "$verdict" in
            FAIL)        has_fail=1 ;;
            UNRESOLVED)  has_unresolved=1 ;;
            INCOMPLETE)  has_incomplete=1 ;;
            UNVERIFIED)  has_unverified=1 ;;
            PASS)        has_pass=1 ;;
            SKIP|NO_CHANGES) ;;
            *)           has_incomplete=1 ;;
        esac
    done

    if (( has_fail )); then return "$EXIT_FAIL"; fi
    if (( has_unresolved )); then return "$EXIT_UNRESOLVED"; fi
    if (( has_incomplete )); then return "$EXIT_INCOMPLETE"; fi
    if (( has_unverified )); then return "$EXIT_UNVERIFIED"; fi
    if (( has_pass )); then return "$EXIT_PASS"; fi
    if (( has_any )); then return "$EXIT_NO_CHANGES"; fi
    return "$EXIT_NO_CHANGES"
}

gate_count_result() {
    local target="$1"
    local count=0 name
    for name in "${CHECK_ORDER[@]}"; do
        [[ "${CHECK_RESULTS[$name]:-}" == "$target" ]] && count=$((count + 1))
    done
    printf '%d' "$count"
}

# Single source of truth: writes summary file and emits the only compact marker.
gate_emit_summary() {
    local final_code="$1"
    local final_name
    final_name="$(gate_verdict_name "$final_code")"

    local count_pass count_fail count_skip count_unresolved count_incomplete count_unverified
    count_pass="$(gate_count_result PASS)"
    count_fail="$(gate_count_result FAIL)"
    count_skip="$(gate_count_result SKIP)"
    count_unresolved="$(gate_count_result UNRESOLVED)"
    count_incomplete="$(gate_count_result INCOMPLETE)"
    count_unverified="$(gate_count_result UNVERIFIED)"

    cat > "$GATE_SUMMARY_FILE" <<SUMMARY
OS_AGENT_GATE_SUMMARY
mode=${GATE_MODE}
verdict=${final_name}
exit_code=${final_code}
artifact_dir=${GATE_ARTIFACT_DIR}
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
        printf '%-30s %s\n' "$name" "${CHECK_RESULTS[$name]}" >> "$GATE_SUMMARY_FILE"
    done

    printf 'OS_AGENT_GATE_RESULT mode=%s verdict=%s code=%d checks=%d pass=%d fail=%d skip=%d unresolved=%d incomplete=%d unverified=%d artifact=%s\n' \
        "$GATE_MODE" "$final_name" "$final_code" "${#CHECK_ORDER[@]}" \
        "$count_pass" "$count_fail" "$count_skip" "$count_unresolved" \
        "$count_incomplete" "$count_unverified" "$GATE_ARTIFACT_DIR"
}

gate_repo_root() {
    git rev-parse --show-toplevel 2>/dev/null || pwd
}
