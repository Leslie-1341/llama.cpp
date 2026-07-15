#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_FILE="$SKILL_DIR/SKILL.md"

fail() {
    printf 'OS_AGENT_SKILL_INVALID: %s\n' "$*" >&2
    exit 1
}

[[ -f "$SKILL_FILE" ]] || fail "missing SKILL.md"
[[ "$(sed -n '1p' "$SKILL_FILE")" == '---' ]] || fail "missing opening frontmatter marker"
[[ "$(grep -c '^---$' "$SKILL_FILE")" -ge 2 ]] || fail "missing closing frontmatter marker"
grep -q '^name: os-agent-task$' "$SKILL_FILE" || fail "wrong or missing name"
grep -q '^description:' "$SKILL_FILE" || fail "missing description"

for f in \
    references/project-contract.md \
    references/ledger-policy.md \
    references/modes.md \
    references/validation-policy.md \
    references/output-templates.md \
    references/examples.md \
    scripts/collect-task-context.sh \
    scripts/init-project-ledger.sh \
    templates/PROJECT_STATE.md \
    templates/ARCHITECTURE.md \
    templates/DECISIONS.md \
    templates/EXPERIMENTS.md; do
    [[ -s "$SKILL_DIR/$f" ]] || fail "missing or empty $f"
done

bash -n "$SKILL_DIR/scripts/collect-task-context.sh"
bash -n "$SKILL_DIR/scripts/init-project-ledger.sh"
bash -n "$SKILL_DIR/scripts/validate-skill.sh"

for mode in contract audit implement review review-fix script memory; do
    grep -q "\`$mode\`" "$SKILL_DIR/references/modes.md" || fail "mode not documented: $mode"
done

printf 'OS_AGENT_SKILL_VALID\n'
