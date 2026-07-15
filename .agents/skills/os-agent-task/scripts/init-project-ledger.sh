#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
MODE="${1:-init}"

files=(PROJECT_STATE.md ARCHITECTURE.md DECISIONS.md EXPERIMENTS.md)

case "$MODE" in
    init)
        cd "$ROOT"
        created=0
        for f in "${files[@]}"; do
            if [[ -e "$f" ]]; then
                printf 'SKIP existing %s\n' "$f"
                continue
            fi
            cp "$SKILL_DIR/templates/$f" "$f"
            printf 'CREATE %s\n' "$f"
            created=$((created + 1))
        done
        printf 'OS_AGENT_LEDGER_READY created=%d\n' "$created"
        ;;
    check|--check)
        cd "$ROOT"
        missing=0
        for f in "${files[@]}"; do
            if [[ -s "$f" ]]; then
                printf 'OK %s sha256=%s lines=%s\n' \
                    "$f" \
                    "$(sha256sum "$f" | awk '{print $1}')" \
                    "$(wc -l < "$f")"
            else
                printf 'MISSING %s\n' "$f" >&2
                missing=1
            fi
        done
        (( missing == 0 )) || exit 1
        printf 'OS_AGENT_LEDGER_VALID\n'
        ;;
    *)
        printf 'usage: %s [init|check]\n' "$0" >&2
        exit 2
        ;;
esac
