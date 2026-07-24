#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

status="$(git status --porcelain=v1 2>/dev/null || true)"
tracked_diff="$(git diff --binary --no-ext-diff 2>/dev/null || true)"
staged_diff="$(git diff --cached --binary --no-ext-diff 2>/dev/null || true)"
untracked_list="$(git ls-files --others --exclude-standard 2>/dev/null | LC_ALL=C sort || true)"

fingerprint="$({
    printf '%s' "$tracked_diff"
    printf '%s' "$staged_diff"
    while IFS= read -r f; do
        [[ -n "$f" && -f "$f" ]] || continue
        printf '\nUNTRACKED %s\n' "$f"
        sha256sum "$f" 2>/dev/null || true
    done <<< "$untracked_list"
} | sha256sum | awk '{print $1}')"

printf 'OS_AGENT_CONTEXT_V3\n'
printf 'repo=%s\n' "$ROOT"
printf 'branch=%s\n' "$(git branch --show-current 2>/dev/null || true)"
printf 'head=%s\n' "$(git rev-parse HEAD 2>/dev/null || true)"
printf 'worktree_dirty=%s\n' "$([[ -n "$status" ]] && printf 1 || printf 0)"
printf 'diff_fingerprint=%s\n' "$fingerprint"

printf '%s\n' '--- status --short ---'
printf '%s\n' "$status"
printf '%s\n' '--- diff --stat ---'
git diff --stat 2>/dev/null || true
printf '%s\n' '--- staged diff --stat ---'
git diff --cached --stat 2>/dev/null || true
printf '%s\n' '--- recent commits ---'
git log --oneline --decorate -3 2>/dev/null || true

printf '%s\n' '--- authority files (metadata only) ---'
for f in AGENTS.md CLAUDE.md PROJECT_STATE.md ARCHITECTURE.md DECISIONS.md EXPERIMENTS.md; do
    if [[ -f "$f" ]]; then
        printf '%s sha256=%s lines=%s bytes=%s\n' \
            "$f" "$(sha256sum "$f" | awk '{print $1}')" \
            "$(wc -l < "$f")" "$(wc -c < "$f")"
    else
        printf '%s missing\n' "$f"
    fi
done
