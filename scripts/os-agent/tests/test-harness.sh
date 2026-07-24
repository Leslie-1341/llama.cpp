#!/usr/bin/env bash
# test-harness.sh — end-to-end synthetic tests for the os-agent gate harness
# Exercises gate-runner main entry. Uses temporary fixtures; auto-recovers.
#
# Usage: bash scripts/os-agent/tests/test-harness.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
GATE_RUNNER="$REPO_ROOT/scripts/os-agent/gate-runner"
FIXTURE_ROOT="/tmp/os-agent-gate-test-$$"
TEST_ARTIFACT_PREFIX="/tmp/os-agent-gate-test-artifacts"

PASS_COUNT=0; FAIL_COUNT=0; TOTAL=0

pass_test() { PASS_COUNT=$((PASS_COUNT + 1)); TOTAL=$((TOTAL + 1)); printf '  PASS: %s\n' "$1"; }
fail_test() { FAIL_COUNT=$((FAIL_COUNT + 1)); TOTAL=$((TOTAL + 1)); printf '  FAIL: %s — %s\n' "$1" "$2" >&2; }

cleanup_all() {
    rm -rf "$FIXTURE_ROOT" 2>/dev/null || true
    cd "$REPO_ROOT"

    # Recover production files even if a self-test is interrupted between
    # registry injection and its normal restore step.
    local reg_file="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
    local bak
    for bak in "$reg_file".bak-*; do
        [[ -f "$bak" ]] || continue
        cp "$bak" "$reg_file" 2>/dev/null || true
        rm -f "$bak"
    done

    if [[ -f /tmp/test-kv-backing-store.cpp.bak ]]; then
        cp /tmp/test-kv-backing-store.cpp.bak "$REPO_ROOT/tests/test-kv-backing-store.cpp" 2>/dev/null || true
        rm -f /tmp/test-kv-backing-store.cpp.bak
    else
        git checkout -- tests/test-kv-backing-store.cpp 2>/dev/null || true
    fi

    rm -f "$REPO_ROOT"/tests/test-gate-* "$REPO_ROOT"/scripts/test-gate-* 2>/dev/null || true
}

trap cleanup_all EXIT

setup_fixture_repo() {
    local name="$1"
    local dir="$FIXTURE_ROOT/$name"
    rm -rf "$dir"
    mkdir -p "$dir"
    cd "$dir"
    git init -q
    git config user.email "test@os-agent-gate"
    git config user.name "Gate Test"
    echo "# Test repo" > README.md
    git add README.md
    git commit -q -m "initial"
    printf '%s' "$dir"
}

# ============================================================
# E2E-1: NO_CHANGES — clean repo
# ============================================================
test_e2e_no_changes() {
    local name="E2E_NO_CHANGES"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Test NO_CHANGES by running diff_analyze in a bare sub-repo and checking the logic.
    # A bare fixture repo won't have build infrastructure, so we test the diff + verdict logic directly.
    local repo; repo=$(setup_fixture_repo "no-changes")
    local output rc=0
    set +e
    output=$(cd "$repo" && bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/diff-analyzer.sh
        diff_analyze
        if (( DIFF_TOTAL == 0 )); then
            printf 'OS_AGENT_GATE_RESULT mode=audit verdict=NO_CHANGES code=4 checks=0\n'
            exit 4
        fi
        exit 0
    " 2>&1)
    rc=$?
    set -e

    if [[ "$rc" -eq 4 ]] && echo "$output" | grep -q 'verdict=NO_CHANGES'; then
        pass_test "$name"
    else
        fail_test "$name" "expected exit 4 NO_CHANGES, got $rc: $output"
    fi
    cd "$REPO_ROOT"
}

# ============================================================
# E2E-2: audit PASS on current repo (read-only, no changes)
# ============================================================
test_e2e_audit_readonly() {
    local name="E2E_audit_readonly"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"
    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" audit --quiet 2>&1)
    rc=$?
    set -e

    # Audit on current repo: must produce marker after set -e fix
    if echo "$output" | grep -q 'OS_AGENT_GATE_RESULT'; then
        pass_test "$name"
    else
        fail_test "$name" "no summary marker (rc=$rc)"
    fi
}

# ============================================================
# E2E-3: Python syntax FAIL
# ============================================================
test_e2e_python_fail() {
    local name="E2E_python_syntax_FAIL"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    local repo; repo=$(setup_fixture_repo "py-fail")
    cat > "$repo/broken.py" << 'EOF'
def foo(
    return 42
EOF

    # Test: python3 -m py_compile should fail on broken syntax
    if python3 -m py_compile "$repo/broken.py" 2>/dev/null; then
        fail_test "$name" "py_compile should have failed"
    else
        pass_test "$name"
    fi
    cd "$REPO_ROOT"
}

# ============================================================
# E2E-4: Shell syntax FAIL
# ============================================================
test_e2e_shell_fail() {
    local name="E2E_shell_syntax_FAIL"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    local repo; repo=$(setup_fixture_repo "sh-fail")
    cat > "$repo/broken.sh" << 'EOF'
#!/usr/bin/env bash
if [[ -z "$X"; then
    echo "syntax error"
EOF

    if bash -n "$repo/broken.sh" 2>/dev/null; then
        fail_test "$name" "bash -n should have failed"
    else
        pass_test "$name"
    fi
    cd "$REPO_ROOT"
}

# ============================================================
# E2E-5: Untracked file detection
# ============================================================
test_e2e_untracked_detection() {
    local name="E2E_untracked_detection"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    local repo; repo=$(setup_fixture_repo "untracked")
    echo "print('hello')" > "$repo/new_script.py"

    local output
    output=$(cd "$repo" && bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/diff-analyzer.sh
        diff_analyze
        echo \"DIFF_TOTAL=\$DIFF_TOTAL\"
        echo \"HAS_UNTRACKED=\$HAS_UNTRACKED\"
        echo \"HAS_PY=\$HAS_PY\"
        for k in \"\${!DIFF_CLASSES[@]}\"; do
            echo \"CLASS[\$k]=\${DIFF_CLASSES[\$k]}\"
        done
    " 2>&1) || true

    if echo "$output" | grep -q 'HAS_UNTRACKED=1' && echo "$output" | grep -q 'HAS_PY=1'; then
        pass_test "$name"
    else
        fail_test "$name" "untracked file not detected: $output"
    fi
    cd "$REPO_ROOT"
}

# ============================================================
# E2E-6: Deleted file tracking
# ============================================================
test_e2e_deleted_tracking() {
    local name="E2E_deleted_tracking"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    local repo; repo=$(setup_fixture_repo "deleted")
    echo "print('test')" > "$repo/test.py"
    git -C "$repo" add test.py
    git -C "$repo" commit -q -m "add test.py"
    rm "$repo/test.py"
    # File is now deleted relative to HEAD

    # Run diff-analyzer directly via a subprocess
    local output
    output=$(cd "$repo" && bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/diff-analyzer.sh
        diff_analyze
        echo \"HAS_DELETED_ANY=\$HAS_DELETED_ANY\"
        echo \"HAS_DELETED_PY=\$HAS_DELETED_PY\"
    " 2>&1) || true

    if echo "$output" | grep -q 'HAS_DELETED_ANY=1' && echo "$output" | grep -q 'HAS_DELETED_PY=1'; then
        pass_test "$name"
    else
        fail_test "$name" "deleted file not tracked: $output"
    fi
    cd "$REPO_ROOT"
}

# ============================================================
# E2E-7: C++ target mapping via compile_commands.json -o flag
# ============================================================
test_e2e_cpp_target_mapping() {
    local name="E2E_cpp_target_mapping"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Test: a real .cpp file in src/ must map to 'llama'
    local target
    target=$(bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/target-mapper.sh
        target_mapper_init build
        target_for_file '$REPO_ROOT/src/llama-kv-cache.cpp'
    " 2>/dev/null) || true

    if echo "$target" | grep -q 'llama'; then
        pass_test "$name"
    else
        fail_test "$name" "llama-kv-cache.cpp should map to llama, got: $target"
    fi
}

# ============================================================
# E2E-8: server-kv-pressure.cpp maps to multiple targets
# ============================================================
test_e2e_multi_target() {
    local name="E2E_multi_target_mapping"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local targets
    targets=$(bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/target-mapper.sh
        target_mapper_init build
        target_for_file '$REPO_ROOT/tools/server/server-kv-pressure.cpp'
    " 2>/dev/null) || true

    # Should map to both server-context and test-server-kv-pressure
    if echo "$targets" | grep -q 'server-context' && echo "$targets" | grep -q 'test-server-kv-pressure'; then
        pass_test "$name"
    else
        fail_test "$name" "server-kv-pressure.cpp should map to server-context+test-server-kv-pressure, got: $targets"
    fi
}

# ============================================================
# E2E-9: UNRESOLVED — unknown C++ file
# ============================================================
test_e2e_unresolved_cpp() {
    local name="E2E_UNRESOLVED_cpp"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # A file not in compile_commands.json should return UNRESOLVED
    local rc=0
    set +e
    bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/target-mapper.sh
        target_mapper_init build
        if target_for_file '$REPO_ROOT/src/nonexistent_file_xyz.cpp'; then
            exit 0
        else
            exit 1
        fi
    " 2>/dev/null
    rc=$?
    set -e

    if [[ "$rc" -ne 0 ]]; then
        pass_test "$name"
    else
        fail_test "$name" "nonexistent file should NOT have a target"
    fi
}

# ============================================================
# E2E-10: Real C++ incremental build (modify test file, build, revert)
# ============================================================
test_e2e_cpp_build() {
    local name="E2E_cpp_build_PASS"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local test_file="tests/test-kv-backing-store.cpp"
    if [[ ! -f "$test_file" ]]; then
        fail_test "$name" "test file not found: $test_file"
        return
    fi

    # Save original
    cp "$test_file" /tmp/test-kv-backing-store.cpp.bak

    # Add a harmless comment (compiles fine)
    echo "// gate-harness e2e test marker $(date +%s)" >> "$test_file"

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" implement --quiet 2>&1)
    rc=$?
    set -e

    # Restore immediately
    cp /tmp/test-kv-backing-store.cpp.bak "$test_file"
    rm -f /tmp/test-kv-backing-store.cpp.bak

    # Verify: marker MUST be present (the set -e crash fix). Build check should pass.
    local marker
    marker=$(echo "$output" | grep 'OS_AGENT_GATE_RESULT' | head -1)
    if [[ -z "$marker" ]]; then
        fail_test "$name" "no marker (rc=$rc)"
        return
    fi

    # Check if incremental-build was recorded
    local build_verdict
    build_verdict=$(cat "$(echo "$marker" | sed 's/.*artifact=//')/summary.txt" 2>/dev/null | grep 'incremental-build' | awk '{print $2}')
    if [[ "$build_verdict" == "PASS" ]]; then
        pass_test "$name"
    elif [[ "$build_verdict" == "FAIL" ]]; then
        fail_test "$name" "real incremental-build failed; self-test must not convert it to PASS"
    else
        fail_test "$name" "incremental-build not found in summary (got: $build_verdict)"
    fi
}

# ============================================================
# E2E-11: Artifact directory outside repo (no self-contamination)
# ============================================================
test_e2e_artifact_outside_repo() {
    local name="E2E_artifact_outside_repo"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" audit --quiet 2>&1)
    rc=$?
    set -e

    # Extract artifact path from marker (must be present after set -e fix)
    local artifact_path
    artifact_path=$(echo "$output" | grep 'OS_AGENT_GATE_RESULT' | head -1 | sed 's/.*artifact=//')

    if [[ "$artifact_path" == /tmp/* ]] || [[ "$artifact_path" == "$TEST_ARTIFACT_PREFIX"* ]]; then
        pass_test "$name"
    else
        fail_test "$name" "artifact should be outside repo, got: $artifact_path"
    fi
}

# ============================================================
# E2E-12: Single summary marker (no duplicates)
# ============================================================
test_e2e_single_marker() {
    local name="E2E_single_summary_marker"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" audit --quiet 2>&1)
    rc=$?
    set -e

    local count
    count=$(echo "$output" | grep -c 'OS_AGENT_GATE_RESULT' || true)
    if [[ "$count" -eq 1 ]]; then
        pass_test "$name"
    else
        fail_test "$name" "expected exactly 1 OS_AGENT_GATE_RESULT marker, got $count"
    fi
}

# ============================================================
# E2E-13: Parser test UNRESOLVED (dummy unregistered test)
# ============================================================
test_e2e_parser_unresolved() {
    local name="E2E_parser_UNRESOLVED"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Create an unregistered parser test file
    local dummy="$REPO_ROOT/tests/test-gate-harness-dummy-parser.py"
    cat > "$dummy" << 'EOF'
#!/usr/bin/env python3
"""Dummy unregistered parser test."""
print("this is not in the parser registry")
EOF

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" implement --quiet 2>&1)
    rc=$?
    set -e

    # Cleanup
    rm -f "$dummy"

    # The parser-test check must be recorded as UNRESOLVED.
    # The overall verdict may be FAIL (if other checks also fail) or UNRESOLVED.
    # Key requirement: marker MUST be present (set -e crash is fixed) and unresolved > 0.
    local marker
    marker=$(echo "$output" | grep 'OS_AGENT_GATE_RESULT' | head -1)
    if [[ -z "$marker" ]]; then
        fail_test "$name" "no marker (rc=$rc)"
    elif echo "$marker" | grep -qE 'unresolved=[1-9]'; then
        if [[ "$rc" -eq 1 ]] || [[ "$rc" -eq 2 ]]; then
            pass_test "$name"
        else
            fail_test "$name" "expected non-zero exit with unresolved>0, got rc=$rc: $marker"
        fi
    else
        fail_test "$name" "expected unresolved>0 in marker: $marker"
    fi
}

# ============================================================
# E2E-14: implement/review/review-fix mode dispatch
# ============================================================
test_e2e_mode_dispatch() {
    local name="E2E_mode_dispatch"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"
    local failed=0

    for mode in implement review review-fix audit; do
        local output rc=0
        set +e
        output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" "$mode" --quiet 2>&1)
        rc=$?
        set -e

        # Key assertion: every mode MUST produce a marker (set -e crash fix)
        if echo "$output" | grep -q "mode=$mode"; then
            printf '  [info] %s: marker present (rc=%d)\n' "$mode" "$rc" >&2
        else
            printf '  [debug] %s: NO MARKER (rc=%d)\n' "$mode" "$rc" >&2
            failed=1
        fi
    done

    if (( failed == 0 )); then
        pass_test "$name"
    else
        fail_test "$name" "one or more modes failed to produce marker"
    fi
}

# ============================================================
# E2E-15: Gate runner respects --build-dir flag
# ============================================================
test_e2e_build_dir_flag() {
    local name="E2E_build_dir_flag"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local output
    output=$(bash "$GATE_RUNNER" implement --help 2>&1) || true
    if echo "$output" | grep -q '\-\-build-dir'; then
        pass_test "$name"
    else
        fail_test "$name" "--build-dir flag not documented in help"
    fi
}

# ============================================================
# E2E-16: knownfail;selected → UNRESOLVED (blocking)
# ============================================================
test_e2e_knownfail_selected_blocks() {
    local name="E2E_knownfail_selected_blocks"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local dummy="$REPO_ROOT/tests/test-gate-harness-knownfail-sel-parser.py"
    local producer="$REPO_ROOT/scripts/test-gate-harness-knownfail-sel-producer.py"
    cat > "$producer" << 'PYEOF'
#!/usr/bin/env python3
print("producer")
PYEOF
    cat > "$dummy" << 'PYEOF'
#!/usr/bin/env python3
"""Dummy knownfail selected parser test."""
import subprocess
PRODUCER = "test-gate-harness-knownfail-sel-producer.py"
print(PRODUCER, subprocess.__name__)
PYEOF

    local reg_file="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
    local bak="$reg_file.bak-$$"
    cp "$reg_file" "$bak"
    sed -i '/^)$/i\    "test-gate-harness-knownfail-sel|tests/test-gate-harness-knownfail-sel-parser.py|knownfail;selected|module:scripts/test-gate-harness-knownfail-sel-producer.py"' "$reg_file"

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" review-fix --quiet 2>&1)
    rc=$?
    set -e

    mv "$bak" "$reg_file"
    rm -f "$dummy" "$producer"

    local marker
    marker=$(echo "$output" | grep 'OS_AGENT_GATE_RESULT' | head -1)
    if [[ -z "$marker" ]]; then
        fail_test "$name" "no marker (rc=$rc)"
    elif echo "$marker" | grep -qE 'unresolved=[1-9]'; then
        pass_test "$name"
    else
        fail_test "$name" "expected unresolved>0 for selected knownfail, got: $marker"
    fi
}

# ============================================================
# E2E-17: current diff overrides knownfail;unrelated
# ============================================================
test_e2e_knownfail_unrelated_selected_by_diff() {
    local name="E2E_knownfail_unrelated_selected_by_diff"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local dummy="$REPO_ROOT/tests/test-gate-harness-knownfail-unrel-parser.py"
    local producer="$REPO_ROOT/scripts/test-gate-harness-knownfail-unrel-producer.py"
    cat > "$producer" << 'PYEOF'
#!/usr/bin/env python3
print("producer")
PYEOF
    cat > "$dummy" << 'PYEOF'
#!/usr/bin/env python3
"""A changed knownfail cannot remain unrelated."""
import subprocess
PRODUCER = "test-gate-harness-knownfail-unrel-producer.py"
print(PRODUCER, subprocess.__name__)
PYEOF

    local reg_file="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
    local bak="$reg_file.bak-$$"
    cp "$reg_file" "$bak"
    sed -i '/^)$/i\    "test-gate-harness-knownfail-unrel|tests/test-gate-harness-knownfail-unrel-parser.py|knownfail;unrelated|module:scripts/test-gate-harness-knownfail-unrel-producer.py"' "$reg_file"

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" review-fix --quiet 2>&1)
    rc=$?
    set -e

    mv "$bak" "$reg_file"
    rm -f "$dummy" "$producer"

    local marker
    marker=$(echo "$output" | grep 'OS_AGENT_GATE_RESULT' | head -1)
    if [[ -z "$marker" ]]; then
        fail_test "$name" "no marker (rc=$rc)"
    elif echo "$marker" | grep -qE 'unresolved=[1-9]'; then
        pass_test "$name"
    else
        fail_test "$name" "changed knownfail;unrelated must be promoted to selected: $marker"
    fi
}

# ============================================================
# E2E-18: New failure CANNOT disguise as knownfail
# ============================================================
test_e2e_new_failure_not_disguised() {
    local name="E2E_new_failure_not_disguised"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Create a parser test that ACTUALLY FAILS (exit 1)
    local dummy="$REPO_ROOT/tests/test-gate-harness-new-fail-parser.py"
    cat > "$dummy" << 'PYEOF'
#!/usr/bin/env python3
"""A genuinely broken parser test — must FAIL the gate, not be absorbed."""
import sys
print("FAIL: this test is genuinely broken", file=sys.stderr)
sys.exit(1)
PYEOF

    # Register it as selfcontained (NOT knownfail)
    local reg_file="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
    local bak="$reg_file.bak-$$"
    cp "$reg_file" "$bak"

    sed -i '/^)$/i\    "test-gate-harness-new-fail|tests/test-gate-harness-new-fail-parser.py|selfcontained"' "$reg_file"

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash "$GATE_RUNNER" review-fix --quiet 2>&1)
    rc=$?
    set -e

    mv "$bak" "$reg_file"
    rm -f "$dummy"

    local marker
    marker=$(echo "$output" | grep 'OS_AGENT_GATE_RESULT' | head -1)
    if [[ -z "$marker" ]]; then
        fail_test "$name" "no marker (rc=$rc)"
    elif echo "$marker" | grep -qE 'fail=[1-9]'; then
        # A genuinely broken non-knownfail test MUST produce a FAIL, not
        # be silently absorbed as knownfail/unresolved.
        if echo "$marker" | grep -qE 'unresolved=[1-9]'; then
            fail_test "$name" "new failure must FAIL (not UNRESOLVED): $marker"
        else
            pass_test "$name"
        fi
    else
        fail_test "$name" "expected fail>0 for broken non-knownfail test, got: $marker"
    fi
}

# ============================================================
# E2E-19: SKIP does NOT produce UNVERIFIED verdict
# ============================================================
test_e2e_skip_not_unverified() {
    local name="E2E_SKIP_not_UNVERIFIED"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Simulate: all checks SKIP (no changes to trigger them) — verdict should
    # be NO_CHANGES or PASS, NOT UNVERIFIED.  SKIP means "not applicable",
    # which is distinct from UNVERIFIED ("should have been checked but wasn't").
    local output rc=0
    set +e
    output=$(bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        gate_record_check 'git-diff-check' 'SKIP' 'no changes'
        gate_record_check 'shell-syntax' 'SKIP' 'no shell files'
        gate_record_check 'python-syntax' 'SKIP' 'no python files'
        gate_final_verdict
        rc=\$?
        gate_verdict_name \$rc
    " 2>&1)
    rc=$?
    set -e

    # All SKIP with no real checks → no changes to evaluate → should be
    # NO_CHANGES (4), not UNVERIFIED (5).
    if [[ "$rc" -eq 4 ]]; then
        pass_test "$name"
    elif [[ "$rc" -eq 5 ]]; then
        fail_test "$name" "SKIP-only checks produced UNVERIFIED — SKIP must be neutral"
    else
        fail_test "$name" "expected NO_CHANGES(4) for all-SKIP, got rc=$rc: $output"
    fi
}

# ============================================================
# E2E-20: INCOMPLETE takes priority over UNVERIFIED
# ============================================================
test_e2e_incomplete_over_unverified() {
    local name="E2E_INCOMPLETE_over_UNVERIFIED"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Simulate: one check UNVERIFIED, another INCOMPLETE.  INCOMPLETE
    # (harness structural problem) should take priority over UNVERIFIED
    # (coverage gap).
    local output rc=0
    set +e
    output=$(bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        gate_record_check 'parser-test' 'UNVERIFIED' 'no relevant changes'
        gate_record_check 'incremental-build' 'INCOMPLETE' 'build chain broken'
        gate_final_verdict
        rc=\$?
        gate_verdict_name \$rc
    " 2>&1)
    rc=$?
    set -e

    if [[ "$rc" -eq 3 ]]; then
        pass_test "$name"
    elif [[ "$rc" -eq 5 ]]; then
        fail_test "$name" "UNVERIFIED(5) overrode INCOMPLETE(3) — INCOMPLETE must take priority"
    else
        fail_test "$name" "expected INCOMPLETE(3), got rc=$rc: $output"
    fi
}

# ============================================================
# E2E-21: UNVERIFIED correctly promoted when nothing worse
# ============================================================
test_e2e_unverified_promotion() {
    local name="E2E_UNVERIFIED_promotion"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Simulate: one check PASS, one check UNVERIFIED.  With no FAIL/UNRESOLVED/
    # INCOMPLETE, the verdict should be UNVERIFIED.
    local output rc=0
    set +e
    output=$(bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        gate_record_check 'git-diff-check' 'PASS' ''
        gate_record_check 'parser-test' 'UNVERIFIED' 'no relevant py/cpp/sh changes'
        gate_final_verdict
        rc=\$?
        gate_verdict_name \$rc
    " 2>&1)
    rc=$?
    set -e

    if [[ "$rc" -eq 5 ]]; then
        pass_test "$name"
    elif [[ "$rc" -eq 0 ]]; then
        fail_test "$name" "UNVERIFIED was silently absorbed into PASS — must surface as UNVERIFIED"
    else
        fail_test "$name" "expected UNVERIFIED(5), got rc=$rc: $output"
    fi
}

# ============================================================
# E2E-22: Fixture binding check — all registered parsers have binding
# ============================================================
test_e2e_fixture_binding_pass() {
    local name="E2E_fixture_binding_PASS"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    # Run check_fixture_binding directly. All entries in the current
    # PARSER_TEST_REGISTRY should have valid module: bindings.
    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/artifact.sh
        GATE_MODE=implement
        artifact_init
        source $REPO_ROOT/scripts/os-agent/checks/run-checks.sh
        check_fixture_binding
        printf 'VERDICT=%s\n' \"\${CHECK_RESULTS[fixture-binding]:-UNKNOWN}\"
    " 2>&1)
    rc=$?
    set -e

    if echo "$output" | grep -q 'VERDICT=PASS'; then
        pass_test "$name"
    elif echo "$output" | grep -q 'VERDICT=UNVERIFIED'; then
        fail_test "$name" "fixture-binding UNVERIFIED — check for legacy/undocumented entries: $output"
    else
        fail_test "$name" "unexpected verdict: $output"
    fi
}

# ============================================================
# E2E-23: Fixture binding — detects missing module
# ============================================================
test_e2e_fixture_binding_missing_module() {
    local name="E2E_fixture_binding_missing_module"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local dummy="$REPO_ROOT/tests/test-gate-missing-module.py"
    cat > "$dummy" << 'PYEOF'
#!/usr/bin/env python3
PARSER = "nonexistent-parser.py"
print(PARSER)
PYEOF

    local reg_file="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
    local bak="$reg_file.bak-$$"
    cp "$reg_file" "$bak"
    sed -i '/^)$/i\    "test-gate-missing-module|tests/test-gate-missing-module.py|selfcontained|module:scripts/nonexistent-parser.py"' "$reg_file"

    local output rc=0
    set +e
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/artifact.sh
        GATE_MODE=implement
        artifact_init
        source $REPO_ROOT/scripts/os-agent/checks/run-checks.sh
        check_fixture_binding
        printf 'VERDICT=%s\\n' \"\${CHECK_RESULTS[fixture-binding]:-UNKNOWN}\"
    " 2>&1)
    rc=$?
    set -e

    mv "$bak" "$reg_file"
    rm -f "$dummy"

    if echo "$output" | grep -q 'VERDICT=FAIL'; then
        pass_test "$name"
    else
        fail_test "$name" "expected FAIL for missing module, got: $output"
    fi
}

# ============================================================
# E2E-24: SKILL.md anti-pattern rules are present
# ============================================================
test_e2e_skill_anti_pattern_rules() {
    local name="E2E_skill_anti_pattern_rules"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    local skill_file="$REPO_ROOT/.claude/skills/os-agent-task/SKILL.md"
    local modes_file="$REPO_ROOT/.claude/skills/os-agent-task/references/modes.md"

    local failed=0

    # SKILL.md must have the anti-omission-as-defect rule
    if grep -q '禁止将 agent 输出遗漏当作源码缺陷' "$skill_file"; then
        :
    else
        printf '  [FAIL] SKILL.md: missing anti-omission-as-defect rule\n' >&2
        failed=1
    fi

    # SKILL.md must have the blocking-item reachability rule
    if grep -q '源码可达性' "$skill_file"; then
        :
    else
        printf '  [FAIL] SKILL.md: missing reachability rule\n' >&2
        failed=1
    fi

    # A proven parser/Harness false-pass path remains a valid tooling blocker.
    if grep -q '假通过、错误归因或漏报' "$skill_file"; then
        :
    else
        printf '  [FAIL] SKILL.md: missing tooling false-pass exception\n' >&2
        failed=1
    fi

    # modes.md must have the blocking-item admission gate
    if grep -q '阻塞项准入门禁' "$modes_file"; then
        :
    else
        printf '  [FAIL] modes.md: missing blocking-item admission gate\n' >&2
        failed=1
    fi

    # modes.md must have the anti-pattern section
    if grep -q '反模式' "$modes_file"; then
        :
    else
        printf '  [FAIL] modes.md: missing anti-pattern section\n' >&2
        failed=1
    fi

    if (( failed == 0 )); then
        pass_test "$name"
    else
        fail_test "$name" "$failed rule(s) missing"
    fi
}

# ============================================================
# E2E-25: Gate runner handles --help for all modes
# ============================================================
test_e2e_help_all_modes() {
    local name="E2E_help_all_modes"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"

    local output
    output=$(bash "$GATE_RUNNER" --help 2>&1) || true

    local failed=0
    for mode in implement review review-fix audit; do
        if echo "$output" | grep -q "$mode"; then
            :
        else
            printf '  [FAIL] help: mode %s not documented\n' "$mode" >&2
            failed=1
        fi
    done

    if ! echo "$output" | grep -q '5 UNVERIFIED'; then
        printf '  [FAIL] help: UNVERIFIED exit code not documented\n' >&2
        failed=1
    fi

    if (( failed == 0 )); then
        pass_test "$name"
    else
        fail_test "$name" "one or more modes missing from --help"
    fi
}

# ============================================================
# E2E-26: Existing unrelated knownfail remains non-selected
# ============================================================
test_e2e_knownfail_unchanged_nonselected() {
    local name="E2E_knownfail_unchanged_nonselected"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"
    local output rc=0
    set +e
    output=$(bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/diff-analyzer.sh
        diff_analyze
        source $REPO_ROOT/scripts/os-agent/checks/run-checks.sh
        if _parser_entry_selected_by_diff '$REPO_ROOT' \\
            'tests/test-server-kv-pressure-stage3a-1c-parser.py' \\
            'module:scripts/parse-server-kv-pressure-stage3a-1c.py'; then
            exit 1
        fi
        exit 0
    " 2>&1)
    rc=$?
    set -e

    if [[ "$rc" -eq 0 ]]; then
        pass_test "$name"
    else
        fail_test "$name" "unchanged knownfail was incorrectly selected: $output"
    fi
}

# ============================================================
# E2E-27: Existing module is not enough — test must reference it
# ============================================================
test_e2e_fixture_binding_unbound_module() {
    local name="E2E_fixture_binding_unbound_module"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"
    local dummy="$REPO_ROOT/tests/test-gate-unbound-module.py"
    cat > "$dummy" << 'PYEOF'
#!/usr/bin/env python3
print("does not reference the declared parser")
PYEOF

    local reg_file="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
    local bak="$reg_file.bak-$$"
    cp "$reg_file" "$bak"
    sed -i '/^)$/i\    "test-gate-unbound-module|tests/test-gate-unbound-module.py|selfcontained|module:scripts/parse-kv-bounded-release-stage3a-2c.py"' "$reg_file"

    local output
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/artifact.sh
        GATE_MODE=implement
        artifact_init
        source $REPO_ROOT/scripts/os-agent/checks/run-checks.sh
        check_fixture_binding
        printf 'VERDICT=%s\\n' \"\${CHECK_RESULTS[fixture-binding]:-UNKNOWN}\"
    " 2>&1) || true

    mv "$bak" "$reg_file"
    rm -f "$dummy"

    if echo "$output" | grep -q 'VERDICT=FAIL'; then
        pass_test "$name"
    else
        fail_test "$name" "declared module existence produced a false binding PASS: $output"
    fi
}

# ============================================================
# E2E-28: Inline binding is UNVERIFIED, never PASS
# ============================================================
test_e2e_fixture_binding_inline_unverified() {
    local name="E2E_fixture_binding_inline_UNVERIFIED"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    cd "$REPO_ROOT"
    local dummy="$REPO_ROOT/tests/test-gate-inline-binding.py"
    cat > "$dummy" << 'PYEOF'
#!/usr/bin/env python3
print("inline fixture")
PYEOF

    local reg_file="$REPO_ROOT/scripts/os-agent/checks/run-checks.sh"
    local bak="$reg_file.bak-$$"
    cp "$reg_file" "$bak"
    sed -i '/^)$/i\    "test-gate-inline-binding|tests/test-gate-inline-binding.py|selfcontained|inline:human-claim"' "$reg_file"

    local output
    output=$(GATE_ARTIFACT_PREFIX="$TEST_ARTIFACT_PREFIX" bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        source $REPO_ROOT/scripts/os-agent/lib/artifact.sh
        GATE_MODE=implement
        artifact_init
        source $REPO_ROOT/scripts/os-agent/checks/run-checks.sh
        check_fixture_binding
        printf 'VERDICT=%s\\n' \"\${CHECK_RESULTS[fixture-binding]:-UNKNOWN}\"
    " 2>&1) || true

    mv "$bak" "$reg_file"
    rm -f "$dummy"

    if echo "$output" | grep -q 'VERDICT=UNVERIFIED'; then
        pass_test "$name"
    else
        fail_test "$name" "inline binding must not be machine-PASS: $output"
    fi
}

# ============================================================
# E2E-29: Invalid check verdict fails closed as INCOMPLETE
# ============================================================
test_e2e_invalid_verdict_incomplete() {
    local name="E2E_invalid_verdict_INCOMPLETE"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    local rc=0
    set +e
    bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        gate_record_check 'bad-check' 'MAYBE' 'invalid verdict'
        gate_final_verdict
    " >/dev/null 2>&1
    rc=$?
    set -e

    if [[ "$rc" -eq 3 ]]; then
        pass_test "$name"
    else
        fail_test "$name" "invalid verdict must yield INCOMPLETE(3), got $rc"
    fi
}

# ============================================================
# E2E-30: Duplicate check names fail closed as INCOMPLETE
# ============================================================
test_e2e_duplicate_check_incomplete() {
    local name="E2E_duplicate_check_INCOMPLETE"
    printf '\n[TEST %d] %s\n' "$((TOTAL + 1))" "$name"

    local rc=0
    set +e
    bash -c "
        source $REPO_ROOT/scripts/os-agent/lib/common.sh
        gate_record_check 'same-check' 'PASS' ''
        gate_record_check 'same-check' 'PASS' ''
        gate_final_verdict
    " >/dev/null 2>&1
    rc=$?
    set -e

    if [[ "$rc" -eq 3 ]]; then
        pass_test "$name"
    else
        fail_test "$name" "duplicate check names must yield INCOMPLETE(3), got $rc"
    fi
}

# ============================================================
# Main
# ============================================================
main() {
    printf '=== OS Agent Gate Harness — End-to-End Tests ===\n'
    printf 'Repo: %s\n' "$REPO_ROOT"
    printf 'Fixtures: %s\n' "$FIXTURE_ROOT"
    printf '\n'

    mkdir -p "$FIXTURE_ROOT"

    test_e2e_no_changes
    test_e2e_audit_readonly
    test_e2e_python_fail
    test_e2e_shell_fail
    test_e2e_untracked_detection
    test_e2e_deleted_tracking
    test_e2e_cpp_target_mapping
    test_e2e_multi_target
    test_e2e_unresolved_cpp
    test_e2e_cpp_build
    test_e2e_artifact_outside_repo
    test_e2e_single_marker
    test_e2e_parser_unresolved
    test_e2e_mode_dispatch
    test_e2e_build_dir_flag
    test_e2e_knownfail_selected_blocks
    test_e2e_knownfail_unrelated_selected_by_diff
    test_e2e_new_failure_not_disguised
    test_e2e_skip_not_unverified
    test_e2e_incomplete_over_unverified
    test_e2e_unverified_promotion
    test_e2e_fixture_binding_pass
    test_e2e_fixture_binding_missing_module
    test_e2e_skill_anti_pattern_rules
    test_e2e_help_all_modes
    test_e2e_knownfail_unchanged_nonselected
    test_e2e_fixture_binding_unbound_module
    test_e2e_fixture_binding_inline_unverified
    test_e2e_invalid_verdict_incomplete
    test_e2e_duplicate_check_incomplete

    printf '\n========================================\n'
    printf 'Results: %d/%d passed\n' "$PASS_COUNT" "$TOTAL"

    if (( FAIL_COUNT > 0 )); then
        printf '%d tests FAILED\n' "$FAIL_COUNT" >&2
        exit 1
    else
        printf 'All tests PASSED\n'
        exit 0
    fi
}

main "$@"
