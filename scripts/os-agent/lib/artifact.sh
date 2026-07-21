#!/usr/bin/env bash
# artifact.sh — artifact directory creation and management
# Default artifact root: /tmp/os-agent-gate/ (outside repo to prevent self-contamination)
# Customizable via GATE_ARTIFACT_PREFIX env var.

set -euo pipefail

# After artifact_init, this variable contains the actual artifact prefix
# so diff-analyzer can exclude it.
declare GATE_ARTIFACT_PREFIX_USED=""

artifact_init() {
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    local timestamp
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"

    # Default to outside-repo path to avoid self-contamination
    local prefix="${GATE_ARTIFACT_PREFIX:-/tmp/os-agent-gate}"
    GATE_ARTIFACT_PREFIX_USED="$prefix"
    GATE_ARTIFACT_DIR="${prefix}/gate-${GATE_MODE}-${timestamp}"
    mkdir -p "$GATE_ARTIFACT_DIR"

    GATE_FULL_LOG="$GATE_ARTIFACT_DIR/full.log"
    GATE_SUMMARY_FILE="$GATE_ARTIFACT_DIR/summary.txt"
    GATE_START_TIME="$timestamp"

    # Start the full log
    cat > "$GATE_FULL_LOG" <<EOF
# OS Agent Gate — full log
# mode=${GATE_MODE}
# artifact_dir=${GATE_ARTIFACT_DIR}
# timestamp=${timestamp}
# repo=${repo_root}
# head=$(git rev-parse HEAD 2>/dev/null || echo unknown)
# branch=$(git branch --show-current 2>/dev/null || echo unknown)
# worktree_status=$(git status --porcelain 2>/dev/null | wc -l) untracked/working items
---
EOF

    gate_log 1 "artifact_init: $GATE_ARTIFACT_DIR"
}

artifact_save_file() {
    local src="$1"
    local name="${2:-$(basename "$src")}"
    if [[ -f "$src" ]]; then
        cp "$src" "$GATE_ARTIFACT_DIR/$name"
    fi
}

artifact_dir() {
    printf '%s' "$GATE_ARTIFACT_DIR"
}

# Returns a grep -E compatible pattern for paths to exclude from diff analysis
artifact_exclude_pattern() {
    # Always exclude the actual artifact directory being used
    printf '%s' "${GATE_ARTIFACT_PREFIX_USED:-/tmp/os-agent-gate}"
}
