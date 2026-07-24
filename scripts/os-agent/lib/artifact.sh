#!/usr/bin/env bash
# Artifact lifecycle. Full command logs stay outside the repository.

set -euo pipefail

declare GATE_ARTIFACT_PREFIX_USED=""

artifact_prune() {
    local prefix="$1" keep="${GATE_ARTIFACT_KEEP:-24}"
    [[ "$keep" =~ ^[0-9]+$ ]] || keep=24
    (( keep > 0 )) || return 0
    mapfile -t old < <(find "$prefix" -mindepth 1 -maxdepth 1 -type d -name 'gate-*' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR>'"$keep"'{print $2}')
    ((${#old[@]} == 0)) || rm -rf -- "${old[@]}"
}

artifact_init() {
    local repo_root timestamp prefix
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    prefix="${GATE_ARTIFACT_PREFIX:-/tmp/os-agent-gate}"
    GATE_ARTIFACT_PREFIX_USED="$prefix"
    mkdir -p "$prefix"
    GATE_ARTIFACT_DIR="$prefix/gate-${GATE_MODE}-${timestamp}-$$"
    mkdir -p "$GATE_ARTIFACT_DIR/logs"
    GATE_FULL_LOG="$GATE_ARTIFACT_DIR/full.log"
    GATE_SUMMARY_FILE="$GATE_ARTIFACT_DIR/summary.txt"
    GATE_START_EPOCH=$(date +%s)

    cat > "$GATE_FULL_LOG" <<LOG
# OS Agent Gate compact execution log
mode=${GATE_MODE}
profile=${GATE_PROFILE}
repo=${repo_root}
head=$(git rev-parse HEAD 2>/dev/null || echo unknown)
branch=$(git branch --show-current 2>/dev/null || echo unknown)
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
---
LOG
    artifact_prune "$prefix"
    gate_log 1 "artifact=$GATE_ARTIFACT_DIR"
}

artifact_dir() { printf '%s' "$GATE_ARTIFACT_DIR"; }
