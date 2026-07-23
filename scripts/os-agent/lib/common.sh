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
readonly EXIT_UNVERIFIED=5  # necessary check was skipped — cannot determine PASS/FAIL

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
    local name="$1"
    local verdict="$2"   # PASS | FAIL | SKIP | UNRESOLVED
    local detail="${3:-}"
    CHECK_RESULTS["$name"]="$verdict"
    CHECK_ORDER+=("$name")
    printf '[%s] %s: %s%s\n' "$(date +%H:%M:%S)" "$name" "$verdict" "${detail:+ — $detail}" >> "$GATE_FULL_LOG"
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

# Summarize all check results and determine final exit code.
# Priority: FAIL > UNRESOLVED > UNVERIFIED > INCOMPLETE > PASS > NO_CHANGES
# UNVERIFIED triggers when a necessary check was SKIPped — cannot determine
# PASS/FAIL without running the check.  Also triggers when knownfail tests
# were excluded without confirmation they're unrelated to current diff.
gate_final_verdict() {
    local worst=4  # start at NO_CHANGES
    local has_unverified=0
    for name in "${CHECK_ORDER[@]}"; do
        local v="${CHECK_RESULTS[$name]}"
        case "$v" in
            FAIL)        worst=1 ;;  # FAIL beats everything, unconditionally
            UNRESOLVED)  [[ $worst -gt 2 || $worst -eq 0 ]] && worst=2 ;;  # overrides PASS/UNVERIFIED/INCOMPLETE/NO_CHANGES
            UNVERIFIED)  has_unverified=1
                         [[ $worst -gt 3 ]] && worst=5 ;;  # overrides PASS/NO_CHANGES only
            INCOMPLETE)  [[ $worst -gt 3 ]] && worst=3 ;;  # only overrides NO_CHANGES
            PASS)        [[ $worst -eq 4 ]] && worst=0 ;;  # only if nothing worse (was NO_CHANGES)
            SKIP)        ;;
            NO_CHANGES)  ;;
        esac
    done
    # If a necessary check was SKIPped with an UNVERIFIED record, and
    # nothing worse than INCOMPLETE happened, the verdict is UNVERIFIED.
    if [[ $has_unverified -eq 1 ]] && [[ $worst -eq 0 || $worst -eq 3 || $worst -eq 4 ]]; then
        worst=5
    fi
    return $worst
}

# Single source of truth: writes summary file AND outputs compact marker to stdout.
# The caller (gate-runner) should NOT emit a second marker.
gate_emit_summary() {
    local final_code="$1"
    local final_name
    final_name="$(gate_verdict_name "$final_code")"

    # Write summary file
    cat > "$GATE_SUMMARY_FILE" <<EOF
OS_AGENT_GATE_SUMMARY
mode=${GATE_MODE}
verdict=${final_name}
exit_code=${final_code}
artifact_dir=${GATE_ARTIFACT_DIR}
timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
head=$(git rev-parse HEAD 2>/dev/null || echo unknown)
branch=$(git branch --show-current 2>/dev/null || echo unknown)
checks_total=${#CHECK_ORDER[@]}
checks_pass=$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "PASS" ]] && echo x; done | wc -l)
checks_fail=$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "FAIL" ]] && echo x; done | wc -l)
checks_skip=$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "SKIP" ]] && echo x; done | wc -l)
checks_unresolved=$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "UNRESOLVED" ]] && echo x; done | wc -l)
checks_unverified=$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "UNVERIFIED" ]] && echo x; done | wc -l)
---
EOF
    for name in "${CHECK_ORDER[@]}"; do
        printf '%-30s %s\n' "$name" "${CHECK_RESULTS[$name]}" >> "$GATE_SUMMARY_FILE"
    done

    # Single compact summary marker to stdout — this is the ONLY place it's emitted
    printf 'OS_AGENT_GATE_RESULT mode=%s verdict=%s code=%d checks=%d pass=%d fail=%d skip=%d unresolved=%d unverified=%d artifact=%s\n' \
        "$GATE_MODE" "$final_name" "$final_code" \
        "${#CHECK_ORDER[@]}" \
        "$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "PASS" ]] && echo x; done | wc -l)" \
        "$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "FAIL" ]] && echo x; done | wc -l)" \
        "$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "SKIP" ]] && echo x; done | wc -l)" \
        "$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "UNRESOLVED" ]] && echo x; done | wc -l)" \
        "$(for n in "${CHECK_ORDER[@]}"; do [[ "${CHECK_RESULTS[$n]:-}" == "UNVERIFIED" ]] && echo x; done | wc -l)" \
        "$GATE_ARTIFACT_DIR"
}

gate_repo_root() {
    git rev-parse --show-toplevel 2>/dev/null || pwd
}
