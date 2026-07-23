#!/usr/bin/env bash
# run-checks.sh — check dispatch engine
# Uses DIFF_CLASSES (from diff-analyzer) to determine which checks to run.
# Each check function records its result via gate_record_check().

set -euo pipefail

OS_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- individual checks ----

check_git_diff_check() {
    local name="git-diff-check"
    gate_log 1 "Running $name..."

    local failed=0

    # Check tracked diff
    local output
    if output=$(git diff --check HEAD 2>&1); then
        gate_log 2 "$name: tracked diff clean"
    else
        gate_log 1 "$name: tracked diff FAIL — whitespace errors"
        printf '%s\n' "$output" >> "$GATE_FULL_LOG"
        failed=1
    fi

    # Check untracked text files for trailing whitespace / conflict markers
    local untracked_text
    untracked_text=$(git ls-files --others --exclude-standard 2>/dev/null | grep -vE '\.(o|so|a|bin|exe|jpg|png|gif|pdf|gz|zip|tar)$' || true)
    if [[ -n "$untracked_text" ]]; then
        while IFS= read -r f; do
            [[ -z "$f" ]] && continue
            [[ ! -f "$f" ]] && continue
            # Check only text files (avoid binary)
            if file "$f" 2>/dev/null | grep -q 'text'; then
                # Create a diff of the file against /dev/null and check it
                if git diff --check /dev/null "$f" 2>&1 | grep -q .; then
                    gate_log 1 "$name: $f has whitespace issues"
                    git diff --check /dev/null "$f" 2>&1 >> "$GATE_FULL_LOG"
                    failed=1
                fi
            fi
        done <<< "$untracked_text"
    fi

    if (( failed > 0 )); then
        gate_record_check "$name" "FAIL" "whitespace errors detected"
    else
        gate_record_check "$name" "PASS" ""
    fi
}

check_python_syntax() {
    local name="python-syntax"
    gate_log 1 "Running $name..."
    local py_files
    py_files="$(printf '%s\n' "${DIFF_CLASSES[py]:-}" | sed '/^$/d' || true)"
    if [[ -z "$py_files" ]]; then
        gate_record_check "$name" "SKIP" "no Python changes"
        return
    fi

    local failed=0 total=0
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        total=$((total + 1))
        if python3 -m py_compile "$f" 2>&1; then
            gate_log 2 "$name: $f OK"
        else
            gate_log 1 "$name: $f FAIL"
            failed=$((failed + 1))
        fi
    done <<< "$py_files"

    if (( failed > 0 )); then
        gate_record_check "$name" "FAIL" "$failed/$total files failed"
    else
        gate_record_check "$name" "PASS" "$total files OK"
    fi
}

check_shell_syntax() {
    local name="shell-syntax"
    gate_log 1 "Running $name..."
    local sh_files
    sh_files="$(printf '%s\n' "${DIFF_CLASSES[sh]:-}" | sed '/^$/d' || true)"
    if [[ -z "$sh_files" ]]; then
        gate_record_check "$name" "SKIP" "no shell changes"
        return
    fi

    local failed=0 total=0
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        total=$((total + 1))
        if bash -n "$f" 2>&1; then
            gate_log 2 "$name: $f OK"
        else
            gate_log 1 "$name: $f FAIL (bash -n)"
            failed=$((failed + 1))
        fi
    done <<< "$sh_files"

    if (( failed > 0 )); then
        gate_record_check "$name" "FAIL" "$failed/$total files failed"
    else
        gate_record_check "$name" "PASS" "$total files OK"
    fi
}

check_cmake_configure() {
    local name="cmake-configure"
    gate_log 1 "Running $name..."

    local cmake_files
    cmake_files="$(printf '%s\n' "${DIFF_CLASSES[cmake]:-}" | sed '/^$/d' || true)"
    if [[ -z "$cmake_files" ]]; then
        gate_record_check "$name" "SKIP" "no CMake changes"
        return
    fi

    local build_dir="${GATE_BUILD_DIR:-build}"
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

    local log_file="$GATE_ARTIFACT_DIR/cmake-configure.log"
    # Reconfigure: cmake automatically detects if nothing has changed and is a no-op
    if cmake -S "$repo_root" -B "$repo_root/$build_dir" 2>&1 | tee "$log_file"; then
        gate_record_check "$name" "PASS" "configure OK"
    else
        gate_record_check "$name" "FAIL" "cmake configure failed"
    fi
}

check_compile_commands() {
    local name="compile-commands"
    gate_log 1 "Running $name..."

    local build_dir="${GATE_BUILD_DIR:-build}"
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    local cc_json="$repo_root/$build_dir/compile_commands.json"

    if [[ ! -f "$cc_json" ]]; then
        gate_record_check "$name" "FAIL" "compile_commands.json missing"
        return
    fi

    local entries
    entries=$(python3 -c "import json; data=json.load(open('$cc_json')); print(len(data))" 2>&1)
    if [[ "$entries" =~ ^[0-9]+$ ]] && (( entries > 0 )); then
        local cpp_files
        cpp_files="$(printf '%s\n' "${DIFF_CLASSES[cpp]:-}" | sed '/^$/d' || true)"
        local unresolved=0 total=0

        if [[ -n "$cpp_files" ]]; then
            while IFS= read -r f; do
                [[ -z "$f" ]] && continue
                total=$((total + 1))
                local abs="$f"
                [[ "$abs" = /* ]] || abs="$repo_root/$f"
                if ! python3 -c "
import json
with open('$cc_json') as fh:
    data = json.load(fh)
found = any(e.get('file','') == '$abs' for e in data)
exit(0 if found else 1)
" 2>/dev/null; then
                    gate_log 1 "$name: $f not in compile_commands.json"
                    unresolved=$((unresolved + 1))
                fi
            done <<< "$cpp_files"
        fi

        if (( unresolved > 0 )); then
            gate_record_check "$name" "FAIL" "$unresolved/$total C++ files not in compile_commands.json"
        else
            gate_record_check "$name" "PASS" "$entries entries, $total changed C++ files all present"
        fi
    else
        gate_record_check "$name" "FAIL" "invalid compile_commands.json"
    fi
}

check_incremental_build() {
    local name="incremental-build"
    gate_log 1 "Running $name..."

    local cpp_files h_files
    cpp_files="$(printf '%s\n' "${DIFF_CLASSES[cpp]:-}" | sed '/^$/d' || true)"
    h_files="$(printf '%s\n' "${DIFF_CLASSES[cpp_headers]:-}" | sed '/^$/d' || true)"
    local all_cpp="$cpp_files"$'\n'"$h_files"

    if [[ -z "$(echo "$all_cpp" | sed '/^$/d' || true)" ]]; then
        gate_record_check "$name" "SKIP" "no C/C++ changes"
        return
    fi

    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    local build_dir="${GATE_BUILD_DIR:-build}"

    # Resolve targets using verified CMake target names
    local resolved
    if ! resolved=$(targets_for_files 1 $cpp_files $h_files 2>/dev/null); then
        gate_record_check "$name" "UNRESOLVED" "some C/C++ files have no verified build target"
        return
    fi

    local -a targets
    mapfile -t targets <<< "$resolved"

    if (( ${#targets[@]} == 0 )); then
        gate_record_check "$name" "UNRESOLVED" "no build targets resolved"
        return
    fi

    gate_log 1 "$name: building targets: ${targets[*]}"

    local failed=0 built=0
    local log_file="$GATE_ARTIFACT_DIR/build.log"
    for target in "${targets[@]}"; do
        gate_log 1 "$name: cmake --build $build_dir --target $target"
        if cmake --build "$repo_root/$build_dir" --target "$target" -j"$(nproc)" 2>&1 | tee -a "$log_file"; then
            gate_log 2 "$name: $target OK"
            built=$((built + 1))
        else
            gate_log 1 "$name: $target FAIL"
            failed=$((failed + 1))
        fi
    done

    if (( failed > 0 )); then
        gate_record_check "$name" "FAIL" "$failed/${#targets[@]} targets failed ($built built)"
    else
        gate_record_check "$name" "PASS" "${#targets[@]} targets built OK"
    fi
}

check_clang_tidy() {
    local name="clang-tidy"
    gate_log 1 "Running $name..."

    if ! command -v clang-tidy &> /dev/null; then
        gate_record_check "$name" "SKIP" "clang-tidy not found"
        return
    fi

    local cpp_files
    cpp_files="$(printf '%s\n' "${DIFF_CLASSES[cpp]:-}" | sed '/^$/d' || true)"
    if [[ -z "$cpp_files" ]]; then
        gate_record_check "$name" "SKIP" "no C++ changes"
        return
    fi

    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    local build_dir="${GATE_BUILD_DIR:-build}"
    local cc_json="$repo_root/$build_dir/compile_commands.json"

    if [[ ! -f "$cc_json" ]]; then
        gate_record_check "$name" "SKIP" "no compile_commands.json"
        return
    fi

    # Only analyze changed TRANSLATION UNITS (.cpp/.cc/.cxx), not standalone headers
    local -a tu_files=()
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        case "${f##*.}" in
            cpp|cc|cxx|C|c) tu_files+=("$f") ;;
            *) continue ;;
        esac
    done <<< "$cpp_files"

    if (( ${#tu_files[@]} == 0 )); then
        gate_record_check "$name" "SKIP" "no changed translation units (headers only)"
        return
    fi

    local failed=0 warnings=0 passed=0 total=${#tu_files[@]}
    local tidylog_dir="$GATE_ARTIFACT_DIR/clang-tidy"
    mkdir -p "$tidylog_dir"

    for f in "${tu_files[@]}"; do
        local abs="$f"
        [[ "$abs" = /* ]] || abs="$repo_root/$f"
        local bn; bn="$(basename "$f")"
        local tu_log="$tidylog_dir/${bn}.log"

        gate_log 2 "$name: analyzing $f"

        # Per-file independent log; use compile_commands.json for flags + explicit C++17
        local rc=0
        clang-tidy -p "$repo_root/$build_dir" "$abs" --quiet --extra-arg=-std=c++17 > "$tu_log" 2>&1 || rc=$?

        if (( rc == 0 )); then
            passed=$((passed + 1))
            gate_log 2 "$name: $f OK"
        else
            # Check THIS file's log only (NOT cumulative log) for error/fatal error
            if grep -qE 'error:|fatal error:' "$tu_log" 2>/dev/null; then
                gate_log 1 "$name: $f FAIL (compiler error)"
                failed=$((failed + 1))
            else
                gate_log 1 "$name: $f advisory warnings (exit=$rc)"
                warnings=$((warnings + 1))
            fi
        fi
    done

    if (( failed > 0 )); then
        gate_record_check "$name" "FAIL" "$failed/$total TUs have errors ($warnings warnings advisory, $passed clean)"
    elif (( warnings > 0 )); then
        gate_record_check "$name" "PASS" "$total TUs OK ($warnings warnings advisory, $passed clean)"
    else
        gate_record_check "$name" "PASS" "$total TUs clean"
    fi
}

# ---- parser test check: explicit registration, no heuristic filtering ----

# Registry: "name|path|flags"
# Each entry is a parser test that can run without external artifacts.
# Flags: selfcontained (runs without models/artifacts)
#        knownfail (pre-existing failure — see below)
#
# knownfail handling:
#   A knownfail entry is a parser test that fails for reasons the operator has
#   confirmed are PRE-EXISTING and unrelated to the current diff/stage.  The
#   gate distinguishes two cases:
#
#     knownfail;unrelated (default)  — failure is NOT selected by the current
#       diff/stage: RECORD it (logged + emitted in marker unresolved tally
#       NOTE) but do NOT block the verdict.  A separate baseline-vs-worktree
#       failure-signature comparison (see PROJECT_STATE / EXPERIMENTS) is the
#       evidence that the failure is pre-existing.
#     knownfail;selected             — failure IS selected by the current
#       diff/stage (e.g. the stage explicitly targets this module) → UNRESOLVED
#       and blocks until the operator resolves it.
#
#   Design intent (validation-policy): a newly-broken parser may NEVER disguise
#   itself as a pre-existing knownfail.  An unregistered new test resolves to
#   `unregistered` → UNRESOLVED.  A registered, non-knownfail test that fails
#   → FAIL.  Only an entry EXPLICITLY marked knownfail;unrelated can be a
#   non-blocking record, and that marking is human-curated against the diff.
declare -a PARSER_TEST_REGISTRY=(
    "test-kv-paged-identity-e2i-parser|tests/test-kv-paged-identity-e2i-parser.py|selfcontained"
    "test-kv-e0-e2-e5-single-turn-parser|tests/test-kv-e0-e2-e5-single-turn-parser.py|selfcontained"
    "test-kv-paged-release-correctness-parser|tests/test-kv-paged-release-correctness-parser.py|selfcontained"
    "test-kv-dry-run-stage3a-2b-parser|tests/test-kv-dry-run-stage3a-2b-parser.py|selfcontained"
    "test-kv-final-controlled-e0-e5-parser|tests/test-kv-final-controlled-e0-e5-parser.py|selfcontained"
    "test-kv-bounded-release-stage3a-2c-parser|tests/test-kv-bounded-release-stage3a-2c-parser.py|selfcontained"
    # knownfail — pre-existing failures unrelated to Stage 3A-2C bounded-release diff
    # (baseline-vs-worktree failure-signature comparison confirms both fail identically
    # at HEAD=a2da80f52 without the working-tree diff; neither test file is in the diff.)
    "test-server-kv-pressure-stage3a-1c-parser|tests/test-server-kv-pressure-stage3a-1c-parser.py|knownfail;unrelated"
    "test-kv-paged-identity-controlled-ab-parser|tests/test-kv-paged-identity-controlled-ab-parser.py|knownfail;unrelated"
)

check_parser_test() {
    local name="parser-test"
    gate_log 1 "Running $name..."

    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

    # Phase 1: discover all parser test candidates on disk
    local -a candidates=()
    while IFS= read -r -d '' f; do
        candidates+=("$f")
    done < <(find "$repo_root/tests" -maxdepth 1 -name '*parser*.py' -print0 2>/dev/null || true)

    if (( ${#candidates[@]} == 0 )); then
        gate_record_check "$name" "SKIP" "no parser tests found on disk"
        return
    fi

    # Phase 2: match candidates against registry
    local -a run_list=()
    local -a knownfail_selected=()
    local -a knownfail_unrelated=()
    local -a unregistered=()
    local -a excluded_reasons=()

    for candidate in "${candidates[@]}"; do
        local bn; bn="$(basename "$candidate")"
        local matched=0
        for entry in "${PARSER_TEST_REGISTRY[@]}"; do
            IFS='|' read -r reg_name reg_path reg_flags <<< "$entry"
            if [[ "$bn" == "$(basename "$reg_path")" ]]; then
                matched=1
                if [[ "$reg_flags" == *"knownfail"* ]]; then
                    if [[ "$reg_flags" == *"selected"* ]]; then
                        knownfail_selected+=("$bn")
                        excluded_reasons+=("$bn: knownfail;selected — this stage explicitly targets its module")
                    else
                        knownfail_unrelated+=("$bn")
                        excluded_reasons+=("$bn: knownfail;unrelated — pre-existing failure, baseline-vs-worktree confirmed unrelated")
                    fi
                else
                    run_list+=("$candidate")
                fi
                break
            fi
        done
        if (( matched == 0 )); then
            unregistered+=("$bn")
            excluded_reasons+=("$bn: not registered — add to PARSER_TEST_REGISTRY in run-checks.sh if self-contained")
        fi
    done

    # Phase 3a: selected knownfail — BLOCKING UNRESOLVED.
    # These are knownfail tests explicitly marked as relevant to the current
    # diff/stage.  The operator must resolve them before the gate can PASS.
    if (( ${#knownfail_selected[@]} > 0 )); then
        gate_log 1 "$name: UNRESOLVED — ${#knownfail_selected[@]} selected knownfail tests"
        for reason in "${excluded_reasons[@]}"; do
            if [[ "$reason" == *"knownfail;selected"* ]]; then
                gate_log 2 "$name: selected: $reason"
            fi
        done
        gate_record_check "$name" "UNRESOLVED" \
            "${#knownfail_selected[@]} selected knownfail tests: ${knownfail_selected[*]}"
        return
    fi

    # Phase 3b: unrelated knownfail — NON-BLOCKING record.
    # These are pre-existing failures confirmed (via baseline-vs-worktree
    # failure-signature comparison) to be unrelated to the current diff.
    # They are LOGGED but do NOT block the gate verdict.
    if (( ${#knownfail_unrelated[@]} > 0 )); then
        gate_log 1 "$name: NOTED — ${#knownfail_unrelated[@]} unrelated knownfail tests (non-blocking)"
        for reason in "${excluded_reasons[@]}"; do
            if [[ "$reason" == *"knownfail;unrelated"* ]]; then
                gate_log 2 "$name: unrelated: $reason"
            fi
        done
        # Do NOT return — continue through remaining phases.
    fi

    # Phase 4: if unregistered files exist, we have an UNRESOLVED situation
    if (( ${#unregistered[@]} > 0 )); then
        gate_log 1 "$name: UNRESOLVED — ${#unregistered[@]} parser tests not in registry"
        for reason in "${excluded_reasons[@]}"; do
            gate_log 2 "$name: excluded: $reason"
        done
        gate_record_check "$name" "UNRESOLVED" \
            "${#unregistered[@]} unregistered parser tests: ${unregistered[*]}"
        return
    fi

    # Phase 5: skip if no relevant changes — but mark as UNVERIFIED
    # (not SKIP) since registered parser tests could validate the diff.
    if [[ -z "${DIFF_CLASSES[py]:-}" ]] && [[ -z "${DIFF_CLASSES[cpp]:-}" ]] && [[ -z "${DIFF_CLASSES[sh]:-}" ]]; then
        gate_record_check "$name" "UNVERIFIED" \
            "no relevant py/cpp/sh changes — cannot verify parser tests are unrelated"
        return
    fi

    # Phase 6: run registered (non-knownfail) tests
    local kf_note=""
    if (( ${#knownfail_unrelated[@]} > 0 )); then
        kf_note="; ${#knownfail_unrelated[@]} unrelated knownfail noted (non-blocking): ${knownfail_unrelated[*]}"
    fi
    local failed=0 passed=0 total=${#run_list[@]}
    local log_file="$GATE_ARTIFACT_DIR/parser-test.log"

    if (( total == 0 )); then
        # All matched tests are unrelated knownfail — run nothing, note them.
        gate_record_check "$name" "PASS" "no runnable tests (all unrelated knownfail)${kf_note}"
        return
    fi

    for pt in "${run_list[@]}"; do
        gate_log 2 "$name: $(basename "$pt")"
        if python3 "$pt" >> "$log_file" 2>&1; then
            passed=$((passed + 1))
        else
            gate_log 1 "$name: $(basename "$pt") FAIL"
            failed=$((failed + 1))
        fi
    done

    if (( failed > 0 )); then
        gate_record_check "$name" "FAIL" "$failed/$total parser tests failed${kf_note}"
    else
        gate_record_check "$name" "PASS" "$total parser tests OK${kf_note}"
    fi
}

check_skill_validation() {
    local name="skill-validation"
    gate_log 1 "Running $name..."

    local skill_files
    skill_files="$(printf '%s\n' "${DIFF_CLASSES[skill]:-}" | sed '/^$/d' || true)"

    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

    # Determine if we should run: always if skill files changed; also if gate harness changed
    local should_run=0
    if [[ -n "$skill_files" ]]; then
        should_run=1
    else
        local gate_files
        gate_files=$(printf '%s\n' "${DIFF_CLASSES[sh]:-}" | sed '/^$/d' | { grep 'scripts/os-agent/' || true; })
        [[ -n "$gate_files" ]] && should_run=1
    fi

    if (( ! should_run )); then
        gate_record_check "$name" "SKIP" "no skill changes"
        return
    fi

    local failed=0
    for skill_dir in "$repo_root/.claude/skills/os-agent-task" "$repo_root/.agents/skills/os-agent-task"; do
        if [[ -f "$skill_dir/scripts/validate-skill.sh" ]]; then
            gate_log 2 "$name: validating $skill_dir"
            if bash "$skill_dir/scripts/validate-skill.sh" 2>&1 | tee -a "$GATE_FULL_LOG"; then
                :
            else
                gate_log 1 "$name: $skill_dir FAIL"
                failed=$((failed + 1))
            fi
        fi
    done

    # bash -n on all gate harness scripts
    if [[ -d "$repo_root/scripts/os-agent" ]]; then
        while IFS= read -r -d '' script; do
            local first_line
            first_line=$(head -1 "$script" 2>/dev/null || true)
            if [[ "$first_line" == '#!/usr/bin/env bash' ]] || [[ "$first_line" == '#!/bin/bash' ]]; then
                bash -n "$script" 2>&1 || failed=$((failed + 1))
            fi
        done < <(find "$repo_root/scripts/os-agent" -type f -print0 2>/dev/null)
    fi

    if (( failed > 0 )); then
        gate_record_check "$name" "FAIL" "$failed validations failed"
    else
        gate_record_check "$name" "PASS" "all skill validations OK"
    fi
}

check_memory_check() {
    local name="memory-check"
    gate_log 1 "Running $name..."

    local ledger_files
    ledger_files="$(printf '%s\n' "${DIFF_CLASSES[ledger]:-}" | sed '/^$/d' || true)"

    if [[ -z "$ledger_files" ]]; then
        gate_record_check "$name" "SKIP" "no ledger changes"
        return
    fi

    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

    local checker="$repo_root/.claude/skills/os-agent-task/scripts/init-project-ledger.sh"
    if [[ -f "$checker" ]]; then
        if bash "$checker" check 2>&1 | tee -a "$GATE_FULL_LOG"; then
            gate_record_check "$name" "PASS" "ledgers present"
        else
            gate_record_check "$name" "FAIL" "ledger check failed"
        fi
    else
        gate_record_check "$name" "SKIP" "no ledger checker"
    fi
}

# ---- main dispatch ----

run_checks_for_mode() {
    local mode="$1"

    case "$mode" in
        implement|review|review-fix|audit)
            ;;
        *)
            gate_log 0 "Unknown mode: $mode"
            return 1
            ;;
    esac

    local gate_script="$OS_AGENT_DIR/gates/define-gates.sh"
    if [[ -f "$gate_script" ]]; then
        source "$gate_script"
    else
        gate_log 0 "Gate definitions not found: $gate_script"
        return 1
    fi

    case "$mode" in
        implement)  gate_implement ;;
        review)     gate_review ;;
        review-fix) gate_review_fix ;;
        audit)      gate_audit ;;
    esac
}
