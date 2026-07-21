#!/usr/bin/env bash
# Thin wrapper — delegates to scripts/os-agent/gate-runner
# Do NOT maintain gate logic here; see scripts/os-agent/ for the implementation.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GATE_RUNNER="$REPO_ROOT/scripts/os-agent/gate-runner"

if [[ ! -f "$GATE_RUNNER" ]]; then
    printf 'ERROR: gate-runner not found at %s\n' "$GATE_RUNNER" >&2
    printf 'Expected: scripts/os-agent/gate-runner in the repo root\n' >&2
    exit 3
fi

exec "$GATE_RUNNER" "$@"
