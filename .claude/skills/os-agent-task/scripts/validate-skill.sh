#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_FILE="$SKILL_DIR/SKILL.md"

fail() {
    printf 'OS_AGENT_SKILL_INVALID: %s\n' "$*" >&2
    exit 1
}

[[ -r "$SKILL_FILE" ]] || fail 'missing or unreadable SKILL.md'
first_line=""
IFS= read -r first_line < "$SKILL_FILE"
[[ "$first_line" == '---' ]] || fail 'missing frontmatter opener'
[[ "$(grep -c '^---$' "$SKILL_FILE")" -ge 2 ]] || fail 'missing frontmatter closer'
grep -q '^name: os-agent-task$' "$SKILL_FILE" || fail 'wrong skill name'
grep -q '^argument-hint: ".*目标价值.*关键边界.*最小验收.*停止条件.*"$' "$SKILL_FILE" || fail 'task contract fields missing'

for marker in \
    'implement（默认）' \
    'audit（例外）' \
    'contract（例外）' \
    'review（例外）' \
    'memory update（提交后）' \
    '生产路径可达' \
    '影响正确性或核心结论' \
    '存在真实证据' \
    '45 分钟停止条件'; do
    grep -Fq "$marker" "$SKILL_FILE" || fail "missing policy marker: $marker"
done

expected="$(printf '%s\n' \
    './SKILL.md' \
    './agents/openai.yaml' \
    './references/high-risk.md' \
    './references/project-contract.md' \
    './scripts/validate-skill.sh' | LC_ALL=C sort)"
actual="$(cd "$SKILL_DIR" && find . -type f -print | LC_ALL=C sort)"
[[ "$actual" == "$expected" ]] || fail 'unexpected or missing files in lightweight skill'

for file in \
    "$SKILL_DIR/agents/openai.yaml" \
    "$SKILL_DIR/references/high-risk.md" \
    "$SKILL_DIR/references/project-contract.md"; do
    [[ -r "$file" && -s "$file" ]] || fail "missing or unreadable ${file#$SKILL_DIR/}"
done

bash -n "$SKILL_DIR/scripts/validate-skill.sh" || fail 'validator shell syntax'

ROOT="$(git -C "$SKILL_DIR" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -n "$ROOT" ]]; then
    other=""
    case "$SKILL_DIR" in
        "$ROOT/.claude/skills/os-agent-task") other="$ROOT/.agents/skills/os-agent-task" ;;
        "$ROOT/.agents/skills/os-agent-task") other="$ROOT/.claude/skills/os-agent-task" ;;
    esac
    if [[ -n "$other" ]]; then
        [[ -d "$other" ]] || fail 'mirror skill directory missing'
        diff -qr "$SKILL_DIR" "$other" >/dev/null || fail 'skill mirrors differ'
    fi
fi

printf 'OS_AGENT_SKILL_VALID\n'
