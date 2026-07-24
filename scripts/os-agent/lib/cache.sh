#!/usr/bin/env bash
# PASS evidence cache. Only review reuses a compatible unchanged fingerprint by default.

set -euo pipefail

gate_cache_root() { printf '%s' "${GATE_CACHE_DIR:-${GATE_ARTIFACT_PREFIX:-/tmp/os-agent-gate}/cache}"; }

cache_find_pass() {
    local fp="$1" root record max_age now ts artifact profile mode
    root="$(gate_cache_root)"; record="$root/$fp.env"
    [[ -f "$record" ]] || return 1
    # shellcheck disable=SC1090
    source "$record"
    max_age="${GATE_CACHE_MAX_AGE_SECONDS:-604800}"; now=$(date +%s)
    [[ "${timestamp_epoch:-}" =~ ^[0-9]+$ ]] || return 1
    (( now - timestamp_epoch <= max_age )) || return 1
    [[ -d "${artifact:-}" && -f "${artifact:-}/summary.txt" ]] || return 1
    grep -q '^verdict=PASS$' "$artifact/summary.txt" || return 1
    case "${profile:-}" in quick|verify|full) ;; *) return 1 ;; esac
    printf '%s' "$artifact"
}

cache_save_pass() {
    local fp="$1" root tmp record
    root="$(gate_cache_root)"; mkdir -p "$root"
    record="$root/$fp.env"; tmp="$record.tmp.$$"
    cat > "$tmp" <<CACHE
artifact=$(printf '%q' "$GATE_ARTIFACT_DIR")
mode=$(printf '%q' "$GATE_MODE")
profile=$(printf '%q' "$GATE_PROFILE")
timestamp_epoch=$(date +%s)
harness_version=$(printf '%q' "$GATE_HARNESS_VERSION")
CACHE
    mv "$tmp" "$record"
}

component_cache_find() {
    local kind="$1" fp="$2" root record max_age now
    root="$(gate_cache_root)/components/$kind"; record="$root/$fp.env"
    [[ -f "$record" ]] || return 1
    # shellcheck disable=SC1090
    source "$record"
    max_age="${GATE_CACHE_MAX_AGE_SECONDS:-604800}"; now=$(date +%s)
    [[ "${timestamp_epoch:-}" =~ ^[0-9]+$ ]] || return 1
    (( now - timestamp_epoch <= max_age )) || return 1
    [[ -d "${artifact:-}" && -f "${artifact:-}/component-${kind}.pass" ]] || return 1
    printf '%s' "$artifact"
}

component_cache_save() {
    local kind="$1" fp="$2" root record tmp
    root="$(gate_cache_root)/components/$kind"; mkdir -p "$root"
    record="$root/$fp.env"; tmp="$record.tmp.$$"
    printf 'kind=%s\nfingerprint=%s\ntimestamp=%s\n' "$kind" "$fp" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$GATE_ARTIFACT_DIR/component-${kind}.pass"
    cat > "$tmp" <<CACHE
artifact=$(printf '%q' "$GATE_ARTIFACT_DIR")
timestamp_epoch=$(date +%s)
harness_version=$(printf '%q' "$GATE_HARNESS_VERSION")
CACHE
    mv "$tmp" "$record"
}
