#!/usr/bin/env bash
set -euo pipefail

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_FILE="$SKILL_DIR/SKILL.md"
REPO_ROOT="$(git -C "$SKILL_DIR" rev-parse --show-toplevel 2>/dev/null || true)"

fail() {
    printf 'OS_AGENT_SKILL_INVALID: %s\n' "$*" >&2
    exit 1
}

[[ -n "$REPO_ROOT" ]] || fail "skill is not inside a git repository"

case "$SKILL_DIR" in
    "$REPO_ROOT/.claude/skills/os-agent-task")
        MIRROR_DIR="$REPO_ROOT/.agents/skills/os-agent-task"
        ;;
    "$REPO_ROOT/.agents/skills/os-agent-task")
        MIRROR_DIR="$REPO_ROOT/.claude/skills/os-agent-task"
        ;;
    *)
        fail "unexpected skill path: $SKILL_DIR"
        ;;
esac

HARNESS_CHECKS="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
HARNESS_COMMON="$REPO_ROOT/scripts/os-agent/lib/common.sh"
GATE_RUNNER="$REPO_ROOT/scripts/os-agent/gate-runner"

[[ -f "$SKILL_FILE" ]] || fail "missing SKILL.md"
[[ "$(sed -n '1p' "$SKILL_FILE")" == '---' ]] || fail "missing opening frontmatter marker"
[[ "$(grep -c '^---$' "$SKILL_FILE")" -ge 2 ]] || fail "missing closing frontmatter marker"
grep -q '^name: os-agent-task$' "$SKILL_FILE" || fail "wrong or missing name"
grep -q '^description:' "$SKILL_FILE" || fail "missing description"

for f in \
    references/project-contract.md \
    references/ledger-policy.md \
    references/systemic-workflow.md \
    references/model-selection.md \
    references/modes.md \
    references/validation-policy.md \
    references/output-templates.md \
    references/examples.md \
    scripts/collect-task-context.sh \
    scripts/init-project-ledger.sh \
    scripts/validate-skill.sh \
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

# System-level workflow invariants.
grep -q '系统级视角门禁' "$SKILL_FILE" || fail "missing system-level checkpoint"
grep -q '同一问题连续两轮未关闭' "$SKILL_FILE" || fail "missing repeated-failure escalation rule"
grep -q '证据不足，退回 audit' "$SKILL_DIR/references/modes.md" || fail "review-fix fallback not documented"
grep -q 'release 后 reuse' "$SKILL_DIR/references/validation-policy.md" || fail "lifecycle reuse test not documented"
grep -q '推荐模型：<model>' "$SKILL_DIR/references/output-templates.md" || fail "model recommendation missing from contract template"
grep -q 'runner 只执行并记录原始事实' "$SKILL_DIR/references/modes.md" || fail "runner authority boundary missing"

# Post-convergence invariants: prevent regression to ungrounded local review-fix loops.
grep -q '源码可达性' "$SKILL_DIR/references/modes.md" || fail "review-fix reachability gate missing"
grep -q '禁止将 agent 输出遗漏当作源码缺陷' "$SKILL_FILE" || fail "anti-pattern: agent omission as defect missing"
grep -q '假通过、错误归因或漏报' "$SKILL_FILE" || fail "tooling false-pass exception missing"
grep -q '反模式' "$SKILL_DIR/references/modes.md" || fail "anti-pattern section missing from modes.md"
grep -q '阻塞项准入门禁' "$SKILL_DIR/references/modes.md" || fail "blocking-item admission gate missing from modes.md"

[[ -s "$HARNESS_CHECKS" ]] || fail "missing harness checks: $HARNESS_CHECKS"
[[ -s "$HARNESS_COMMON" ]] || fail "missing harness common library: $HARNESS_COMMON"
[[ -s "$GATE_RUNNER" ]] || fail "missing gate runner: $GATE_RUNNER"
grep -q '^check_fixture_binding()' "$HARNESS_CHECKS" || fail "fixture-binding check not found"
grep -q 'EXIT_UNVERIFIED=5' "$HARNESS_COMMON" || fail "UNVERIFIED exit code not defined"
grep -q '5 UNVERIFIED' "$GATE_RUNNER" || fail "UNVERIFIED help text missing"

# Claude and Codex skill trees are mirrors. A stale mirror changes agent behavior.
[[ -d "$MIRROR_DIR" ]] || fail "missing mirrored skill directory: $MIRROR_DIR"
if ! diff -qr "$SKILL_DIR" "$MIRROR_DIR" >/dev/null; then
    fail "skill mirrors differ: $SKILL_DIR vs $MIRROR_DIR"
fi

printf 'OS_AGENT_SKILL_VALID\n'
