#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_FILE="$SKILL_DIR/SKILL.md"
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"

fail() { printf 'OS_AGENT_SKILL_INVALID: %s\n' "$*" >&2; exit 1; }

[[ -f "$SKILL_FILE" ]] || fail 'missing SKILL.md'
[[ "$(sed -n '1p' "$SKILL_FILE")" == '---' ]] || fail 'missing frontmatter'
grep -q '^name: os-agent-task$' "$SKILL_FILE" || fail 'wrong name'
grep -q '最小上下文策略' "$SKILL_FILE" || fail 'missing selective context policy'
grep -q '源码可达' "$SKILL_FILE" || fail 'missing blocker admission rule'
grep -q '禁止默认整本加载' "$SKILL_FILE" || fail 'missing full-ledger prohibition'

required=(
  references/project-contract.md
  references/context-policy.md
  references/high-risk.md
  references/model-selection.md
  references/modes/contract.md
  references/modes/audit.md
  references/modes/implement.md
  references/modes/review.md
  references/modes/review-fix.md
  references/modes/script.md
  references/modes/memory.md
  scripts/collect-task-context.sh
  scripts/read-ledger-sections.sh
  scripts/gate-runner.sh
  scripts/init-project-ledger.sh
  templates/PROJECT_STATE.md
  templates/ARCHITECTURE.md
  templates/DECISIONS.md
  templates/EXPERIMENTS.md
)
for f in "${required[@]}"; do [[ -s "$SKILL_DIR/$f" ]] || fail "missing or empty $f"; done

legacy=(
  references/examples.md references/ledger-policy.md references/systemic-workflow.md
  references/validation-policy.md references/output-templates.md references/modes.md
)
for f in "${legacy[@]}"; do [[ ! -e "$SKILL_DIR/$f" ]] || fail "legacy redundant file remains: $f"; done

for f in "$SKILL_DIR"/scripts/*.sh; do bash -n "$f" || fail "shell syntax: $f"; done

for mode in contract audit implement review review-fix script memory; do
    grep -q "# $mode" "$SKILL_DIR/references/modes/$mode.md" || fail "mode header missing: $mode"
done

grep -q 'Runner 只执行并记录原始事实' "$SKILL_DIR/references/modes/script.md" || fail 'runner/parser authority missing'
grep -q '证据不足，退回 audit' "$SKILL_DIR/references/modes/review-fix.md" || fail 'review-fix fallback missing'
grep -q 'release/offload→reuse' "$SKILL_DIR/references/high-risk.md" || fail 'lifecycle reuse matrix missing'

if [[ -n "$ROOT" ]]; then
    other=""
    case "$SKILL_DIR" in
      "$ROOT/.claude/skills/os-agent-task") other="$ROOT/.agents/skills/os-agent-task" ;;
      "$ROOT/.agents/skills/os-agent-task") other="$ROOT/.claude/skills/os-agent-task" ;;
    esac
    if [[ -n "$other" && -d "$other" ]]; then
        diff -qr "$SKILL_DIR" "$other" >/dev/null || fail '.claude and .agents skill copies differ'
    fi
    [[ -f "$ROOT/scripts/os-agent/gate-runner" ]] || fail 'missing scripts/os-agent/gate-runner'
    grep -q 'UNVERIFIED' "$ROOT/scripts/os-agent/lib/common.sh" || fail 'Harness lacks UNVERIFIED verdict'
    grep -q 'check_fixture_binding' "$ROOT/scripts/os-agent/checks/run-checks.sh" || fail 'Harness lacks fixture binding check'
fi

printf 'OS_AGENT_SKILL_V3_VALID\n'
