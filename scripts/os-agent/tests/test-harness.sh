#!/usr/bin/env bash
# Isolated Harness tests. Never edits the caller's real worktree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REAL_REPO="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
FIX_ROOT="${TMPDIR:-/tmp}/os-agent-harness-v2-test-$$"
MODE=fast
PASS=0 FAIL=0 TOTAL=0

case "${1:---fast}" in --fast) MODE=fast ;; --full) MODE=full ;; *) printf 'usage: %s [--fast|--full]\n' "$0" >&2; exit 2 ;; esac

before_status="$(git -C "$REAL_REPO" status --porcelain 2>/dev/null || true)"
cleanup() { rm -rf "$FIX_ROOT"; }
trap cleanup EXIT

ok() { PASS=$((PASS+1)); TOTAL=$((TOTAL+1)); printf 'PASS %s\n' "$1"; }
bad() { FAIL=$((FAIL+1)); TOTAL=$((TOTAL+1)); printf 'FAIL %s — %s\n' "$1" "$2" >&2; }

setup_repo() {
    local name="$1"
    local dir="$FIX_ROOT/$name"
    mkdir -p "$dir/scripts" "$dir/tests" "$dir/.claude/skills/os-agent-task/scripts" "$dir/.agents/skills/os-agent-task/scripts"
    cp -a "$HARNESS_ROOT" "$dir/scripts/os-agent"
    cat > "$dir/.claude/skills/os-agent-task/SKILL.md" <<'S'
---
name: os-agent-task
---
minimal fixture
S
    cat > "$dir/.claude/skills/os-agent-task/scripts/validate-skill.sh" <<'S'
#!/usr/bin/env bash
set -euo pipefail
echo OS_AGENT_SKILL_TEST_VALID
S
    chmod +x "$dir/.claude/skills/os-agent-task/scripts/validate-skill.sh"
    cat > "$dir/.claude/skills/os-agent-task/scripts/init-project-ledger.sh" <<'S'
#!/usr/bin/env bash
set -euo pipefail
[[ "${1:-}" == check ]] && exit 0
exit 2
S
    chmod +x "$dir/.claude/skills/os-agent-task/scripts/init-project-ledger.sh"
    cp -a "$dir/.claude/skills/os-agent-task/." "$dir/.agents/skills/os-agent-task/"
    cat > "$dir/scripts/demo-parser.py" <<'PY'
#!/usr/bin/env python3
print("parser")
PY
    cat > "$dir/tests/test-demo-parser.py" <<'PY'
#!/usr/bin/env python3
import subprocess, sys
subprocess.run([sys.executable, "scripts/demo-parser.py"], check=True)
PY
    cat > "$dir/scripts/os-agent/config/parser-tests.tsv" <<'TSV'
test-demo-parser	tests/test-demo-parser.py	module:scripts/demo-parser.py	tests/test-demo-parser.py;scripts/demo-parser.py	demo
TSV
    cat > "$dir/.gitignore" <<'G'
build/
__pycache__/
*.pyc
G
    echo '# fixture' > "$dir/README.md"
    git -C "$dir" init -q
    git -C "$dir" config user.email test@example.invalid
    git -C "$dir" config user.name harness-test
    git -C "$dir" add .
    git -C "$dir" commit -qm baseline
    printf '%s' "$dir"
}

run_gate() {
    local repo="$1"; shift
    (cd "$repo" && GATE_ARTIFACT_PREFIX="$FIX_ROOT/_artifacts" GATE_CACHE_DIR="$FIX_ROOT/_cache" GATE_SKIP_HARNESS_SELFTEST=1 bash scripts/os-agent/gate-runner "$@")
}

test_verdict_guards() {
    local rc=0
    set +e
    bash -c "source '$HARNESS_ROOT/lib/common.sh'; gate_record_check x MAYBE; gate_final_verdict" >/dev/null 2>&1; rc=$?
    set -e
    [[ $rc == 3 ]] || { bad verdict-invalid "rc=$rc"; return; }
    set +e
    bash -c "source '$HARNESS_ROOT/lib/common.sh'; gate_record_check x PASS; gate_record_check x PASS; gate_final_verdict" >/dev/null 2>&1; rc=$?
    set -e
    [[ $rc == 3 ]] && ok verdict-guards || bad verdict-duplicate "rc=$rc"
}


test_no_changes_marker() {
    local repo out rc=0
    repo="$(setup_repo no-changes)"
    set +e; out="$(run_gate "$repo" audit --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 4 && $(grep -c '^OS_AGENT_GATE_RESULT ' <<<"$out") == 1 && "$out" == *'verdict=NO_CHANGES'* ]] && ok no-changes-marker || bad no-changes-marker "rc=$rc output=$out"
}

test_quiet_single_marker() {
    local repo out rc=0
    repo="$(setup_repo quiet)"; echo '# change' >> "$repo/README.md"
    set +e; out="$(run_gate "$repo" audit --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 0 && $(grep -c '^OS_AGENT_GATE_RESULT ' <<<"$out") == 1 && $(wc -l <<<"$out") == 1 ]] && ok quiet-single-marker || bad quiet-single-marker "rc=$rc output=$out"
}

test_syntax_fail_closed() {
    local repo out rc=0
    repo="$(setup_repo syntax)"; cat > "$repo/bad.sh" <<'S'
#!/usr/bin/env bash
if [[ x == x ]; then
S
    set +e; out="$(run_gate "$repo" audit --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 1 && "$out" == *'verdict=FAIL'* ]] && ok syntax-fail-closed || bad syntax-fail-closed "rc=$rc $out"
}

test_diff_deleted_untracked() {
    local repo out
    repo="$(setup_repo diff)"; echo x > "$repo/new.py"; rm "$repo/README.md"
    out="$(cd "$repo" && bash -c "source scripts/os-agent/lib/common.sh; source scripts/os-agent/lib/diff-analyzer.sh; diff_analyze; echo \$DIFF_TOTAL \$HAS_UNTRACKED \$HAS_DELETED_ANY")"
    [[ "$out" == '2 1 1' ]] && ok diff-deleted-untracked || bad diff-deleted-untracked "$out"
}

test_parser_selected_only() {
    local repo out rc=0
    repo="$(setup_repo parser-select)"
    cat > "$repo/scripts/other-parser.py" <<'PY'
#!/usr/bin/env python3
print("other")
PY
    cat > "$repo/tests/test-other-parser.py" <<'PY'
#!/usr/bin/env python3
import subprocess, sys
subprocess.run([sys.executable, "scripts/other-parser.py"], check=True)
PY
    cat >> "$repo/scripts/os-agent/config/parser-tests.tsv" <<'TSV'
test-other-parser	tests/test-other-parser.py	module:scripts/other-parser.py	tests/test-other-parser.py;scripts/other-parser.py	other
TSV
    git -C "$repo" add . && git -C "$repo" commit -qm add-other
    cat > "$repo/tests/test-demo-parser.py" <<PY
#!/usr/bin/env python3
from pathlib import Path
import subprocess, sys
Path("$FIX_ROOT/demo-ran").write_text("1")
subprocess.run([sys.executable, "scripts/demo-parser.py"], check=True)
PY
    set +e; out="$(run_gate "$repo" implement --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 0 && -f "$FIX_ROOT/demo-ran" && "$out" == *'verdict=PASS'* ]] && ok parser-selected-only || bad parser-selected-only "rc=$rc $out"
}

test_unregistered_changed_parser() {
    local repo out rc=0
    repo="$(setup_repo unregistered)"; echo 'print(1)' > "$repo/tests/test-new-parser.py"
    set +e; out="$(run_gate "$repo" implement --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 2 && "$out" == *'unresolved='* ]] && ok unregistered-parser-unresolved || bad unregistered-parser-unresolved "rc=$rc $out"
}

test_missing_binding_fails() {
    local repo out rc=0
    repo="$(setup_repo missing-binding)"
    cat > "$repo/scripts/os-agent/config/parser-tests.tsv" <<'TSV'
test-demo-parser	tests/test-demo-parser.py	module:scripts/missing.py	tests/test-demo-parser.py	bad
TSV
    git -C "$repo" add . && git -C "$repo" commit -qm bad-registry
    echo change >> "$repo/README.md"
    set +e; out="$(run_gate "$repo" implement --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 1 && "$out" == *'verdict=FAIL'* ]] && ok missing-binding-fails || bad missing-binding-fails "rc=$rc $out"
}

test_skill_parity_once() {
    local repo out rc=0
    repo="$(setup_repo skill-parity)"; echo drift >> "$repo/.agents/skills/os-agent-task/SKILL.md"
    set +e; out="$(run_gate "$repo" implement --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 1 && "$out" == *'verdict=FAIL'* ]] && ok skill-parity-fails || bad skill-parity-fails "rc=$rc $out"
}

test_cache_review_reuse() {
    local repo out1 out2 rc1=0 rc2=0 artifact summary
    repo="$(setup_repo cache)"; echo change >> "$repo/README.md"
    set +e; out1="$(run_gate "$repo" implement --quiet 2>&1)"; rc1=$?; out2="$(run_gate "$repo" review --quiet 2>&1)"; rc2=$?; set -e
    artifact="${out2##*artifact=}"; summary="$artifact/summary.txt"
    [[ $rc1 == 0 && $rc2 == 0 && -f "$summary" ]] || { bad cache-review-reuse "rc1=$rc1 rc2=$rc2"; return; }
    grep -q '^reused_from=/tmp/' "$summary" && grep -q '^reused-validation[[:space:]]*PASS' "$summary" && ok cache-review-reuse || bad cache-review-reuse "summary did not prove reuse"
}

test_audit_readonly_cmake() {
    local repo out rc=0 before after
    repo="$(setup_repo audit-cmake)"; echo 'cmake_minimum_required(VERSION 3.16)' > "$repo/CMakeLists.txt"
    before="$(find "$repo" -maxdepth 1 -type d -printf '%f\n' | sort)"
    set +e; out="$(run_gate "$repo" audit --quiet 2>&1)"; rc=$?; set -e
    after="$(find "$repo" -maxdepth 1 -type d -printf '%f\n' | sort)"
    [[ $rc == 5 && "$before" == "$after" && ! -d "$repo/build" ]] && ok audit-cmake-readonly || bad audit-cmake-readonly "rc=$rc"
}

setup_cmake_repo() {
    local name="$1" repo
    repo="$(setup_repo "$name")"
    cat > "$repo/CMakeLists.txt" <<'C'
cmake_minimum_required(VERSION 3.16)
project(mini LANGUAGES CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_executable(mini main.cpp)
C
    echo 'int main(){return 0;}' > "$repo/main.cpp"
    git -C "$repo" add CMakeLists.txt main.cpp && git -C "$repo" commit -qm cmake
    cmake -S "$repo" -B "$repo/build" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON >/dev/null
    printf '%s' "$repo"
}

test_cmake_build_pass() {
    local repo out rc=0
    repo="$(setup_cmake_repo cmake-pass)"; echo '// change' >> "$repo/main.cpp"
    set +e; out="$(run_gate "$repo" implement --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 0 && "$out" == *'verdict=PASS'* && -x "$repo/build/mini" ]] && ok cmake-build-pass || bad cmake-build-pass "rc=$rc $out"
}

test_cmake_build_failure_not_pass() {
    local repo out rc=0
    repo="$(setup_cmake_repo cmake-fail)"; echo 'this is invalid C++' >> "$repo/main.cpp"
    set +e; out="$(run_gate "$repo" implement --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 1 && "$out" == *'verdict=FAIL'* ]] && ok cmake-build-failure || bad cmake-build-failure "rc=$rc $out"
}


test_review_fix_reuses_unchanged_build() {
    local repo out1 out2 rc1=0 rc2=0 artifact summary
    repo="$(setup_cmake_repo component-reuse)"; echo '// valid change' >> "$repo/main.cpp"
    set +e; out1="$(run_gate "$repo" implement --quiet 2>&1)"; rc1=$?; set -e
    echo '# parser-only follow-up' >> "$repo/scripts/demo-parser.py"
    set +e; out2="$(run_gate "$repo" review-fix --quiet 2>&1)"; rc2=$?; set -e
    artifact="${out2##*artifact=}"; summary="$artifact/summary.txt"
    [[ $rc1 == 0 && $rc2 == 0 && -f "$summary" ]] || { bad component-build-reuse "rc1=$rc1 rc2=$rc2"; return; }
    grep -Eq '^incremental-build[[:space:]]+PASS[[:space:]]+reused unchanged C/C\+\+ surface' "$summary" && ok component-build-reuse || bad component-build-reuse "build was rerun or reuse was not recorded"
}

test_full_parser_runs_all() {
    local repo out rc=0
    repo="$(setup_repo full-parser)"; echo change >> "$repo/README.md"
    set +e; out="$(run_gate "$repo" implement --full --quiet 2>&1)"; rc=$?; set -e
    [[ $rc == 0 && "$out" == *'profile=full'* ]] && ok full-parser || bad full-parser "rc=$rc $out"
}

mkdir -p "$FIX_ROOT"
printf 'Harness v2 isolated tests (%s)\n' "$MODE"
test_verdict_guards
test_no_changes_marker
test_quiet_single_marker
test_syntax_fail_closed
test_diff_deleted_untracked
test_parser_selected_only
test_unregistered_changed_parser
test_missing_binding_fails
test_skill_parity_once
test_cache_review_reuse
test_audit_readonly_cmake
if [[ "$MODE" == full ]]; then
    test_cmake_build_pass
    test_cmake_build_failure_not_pass
    test_review_fix_reuses_unchanged_build
    test_full_parser_runs_all
fi

after_status="$(git -C "$REAL_REPO" status --porcelain 2>/dev/null || true)"
if [[ "$before_status" == "$after_status" ]]; then ok real-worktree-unchanged; else bad real-worktree-unchanged "status changed"; fi

printf 'RESULT %d/%d passed\n' "$PASS" "$TOTAL"
(( FAIL == 0 ))
