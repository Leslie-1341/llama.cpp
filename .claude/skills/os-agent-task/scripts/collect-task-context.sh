#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

printf 'OS_AGENT_CONTEXT_V2\n'
printf 'repo=%s\n' "$ROOT"
printf 'branch=%s\n' "$(git branch --show-current 2>/dev/null || true)"
printf 'head=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
printf 'worktree_dirty=%s\n' "$([[ -n "$(git status --porcelain 2>/dev/null)" ]] && printf 1 || printf 0)"

printf '%s\n' '--- status --short ---'
git status --short 2>/dev/null || true

printf '%s\n' '--- diff --stat ---'
git diff --stat 2>/dev/null || true

printf '%s\n' '--- staged diff --stat ---'
git diff --cached --stat 2>/dev/null || true

printf '%s\n' '--- recent commits ---'
git log --oneline --decorate -5 2>/dev/null || true

printf '%s\n' '--- task rules ---'
for f in AGENTS.md CLAUDE.md; do
    if [[ -f "$f" ]]; then
        printf '%s sha256=%s lines=%s\n' \
            "$f" \
            "$(sha256sum "$f" | awk '{print $1}')" \
            "$(wc -l < "$f")"
    else
        printf '%s missing\n' "$f"
    fi
done

printf '%s\n' '--- project ledger ---'
for f in PROJECT_STATE.md ARCHITECTURE.md DECISIONS.md EXPERIMENTS.md; do
    if [[ -f "$f" ]]; then
        printf '%s sha256=%s lines=%s\n' \
            "$f" \
            "$(sha256sum "$f" | awk '{print $1}')" \
            "$(wc -l < "$f")"
    else
        printf '%s missing\n' "$f"
    fi
done
