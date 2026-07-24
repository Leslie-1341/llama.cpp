#!/usr/bin/env bash
# Diff-selected checks. Expensive commands write full logs to the artifact.

set -euo pipefail

OS_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
declare TARGET_MAPPER_READY=0

affected_paths() {
    printf '%s\n' "${DIFF_FILES[@]}"
}

check_git_diff_check() {
    local name=git-diff-check log="$GATE_ARTIFACT_DIR/logs/git-diff-check.log" failed=0 f
    : > "$log"
    if ! git diff --check HEAD >>"$log" 2>&1; then failed=1; fi
    while IFS= read -r f; do
        [[ -n "$f" && -f "$f" ]] || continue
        if file "$f" 2>/dev/null | grep -q text; then
            if grep -nE '[[:blank:]]+$|^(<<<<<<<|=======|>>>>>>>)' "$f" >>"$log" 2>&1; then failed=1; fi
        fi
    done < <(git ls-files --others --exclude-standard)
    if (( failed )); then gate_failure_excerpt "$log"; gate_record_check "$name" FAIL "whitespace/conflict-marker errors"; else gate_record_check "$name" PASS; fi
}

check_python_syntax() {
    local name=python-syntax f failed=0 total=0 log="$GATE_ARTIFACT_DIR/logs/python-syntax.log"
    : > "$log"
    while IFS= read -r f; do
        [[ -n "$f" && -f "$f" ]] || continue
        total=$((total + 1))
        python3 -m py_compile "$f" >>"$log" 2>&1 || failed=$((failed + 1))
    done < <(diff_paths_for py)
    if (( total == 0 )); then gate_record_check "$name" SKIP "no changed Python files"
    elif (( failed )); then gate_failure_excerpt "$log"; gate_record_check "$name" FAIL "$failed/$total files failed"
    else gate_record_check "$name" PASS "$total files"; fi
}

check_shell_syntax() {
    local name=shell-syntax f failed=0 total=0 log="$GATE_ARTIFACT_DIR/logs/shell-syntax.log"
    : > "$log"
    while IFS= read -r f; do
        [[ -n "$f" && -f "$f" ]] || continue
        total=$((total + 1))
        bash -n "$f" >>"$log" 2>&1 || failed=$((failed + 1))
    done < <(diff_paths_for sh)
    if (( total == 0 )); then gate_record_check "$name" SKIP "no changed shell files"
    elif (( failed )); then gate_failure_excerpt "$log"; gate_record_check "$name" FAIL "$failed/$total files failed"
    else gate_record_check "$name" PASS "$total files"; fi
}

check_cmake_configure() {
    local name=cmake-configure repo_root build_dir cc
    repo_root="$(gate_repo_root)"; build_dir="${GATE_BUILD_DIR:-build}"; cc="$repo_root/$build_dir/compile_commands.json"
    if (( ! HAS_CMAKE )) && [[ -f "$cc" ]]; then gate_record_check "$name" SKIP "configuration unchanged"; return; fi
    if gate_run_logged "$GATE_ARTIFACT_DIR/logs/cmake-configure.log" cmake -S "$repo_root" -B "$repo_root/$build_dir" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON; then
        gate_record_check "$name" PASS
    else gate_record_check "$name" FAIL "see logs/cmake-configure.log"; fi
}

ensure_target_mapper() {
    (( TARGET_MAPPER_READY )) && return 0
    target_mapper_init "${GATE_BUILD_DIR:-build}" "${TARGET_MAPPER_READ_ONLY:-0}"
    TARGET_MAPPER_READY=1
}

cpp_changed_files() {
    printf '%s\n' "$(diff_paths_for cpp)" "$(diff_paths_for c)" "$(diff_paths_for h)" | sed '/^$/d' | sort -u
}

check_compile_commands() {
    local name=compile-commands f unresolved=0 total=0
    if (( !(HAS_CPP || HAS_C || HAS_H) )); then gate_record_check "$name" SKIP "no C/C++ changes"; return; fi
    ensure_target_mapper
    if (( TARGET_MAP_FAILED )); then gate_record_check "$name" FAIL "$TARGET_MAP_ERROR"; return; fi
    while IFS= read -r f; do
        [[ -n "$f" ]] || continue
        total=$((total + 1))
        target_for_file "$f" >/dev/null 2>&1 || unresolved=$((unresolved + 1))
    done < <(cpp_changed_files)
    if (( unresolved )); then gate_record_check "$name" UNRESOLVED "$unresolved/$total files lack a verified target"
    else gate_record_check "$name" PASS "$TARGET_MAP_ENTRIES entries; $total changed files mapped"; fi
}

check_incremental_build() {
    local name=incremental-build repo_root build_dir resolved target failed=0 built=0
    if (( !(HAS_CPP || HAS_C || HAS_H) )); then gate_record_check "$name" SKIP "no C/C++ changes"; return; fi
    ensure_target_mapper
    if (( TARGET_MAP_FAILED )); then gate_record_check "$name" FAIL "$TARGET_MAP_ERROR"; return; fi
    if ! resolved="$(targets_for_files 1 $(cpp_changed_files) 2>/dev/null)"; then gate_record_check "$name" UNRESOLVED "target mapping incomplete"; return; fi
    local -a targets=(); mapfile -t targets <<< "$resolved"
    if (( ${#targets[@]} == 0 )); then gate_record_check "$name" UNRESOLVED "no build targets"; return; fi
    local component_fp cached_artifact=""
    component_fp="$(fingerprint_build_surface)"
    if (( ${GATE_COMPONENT_REUSE:-1} )); then cached_artifact="$(component_cache_find build "$component_fp" 2>/dev/null || true)"; fi
    if [[ -n "$cached_artifact" ]]; then gate_record_check "$name" PASS "reused unchanged C/C++ surface from $cached_artifact"; return; fi
    repo_root="$(gate_repo_root)"; build_dir="${GATE_BUILD_DIR:-build}"
    : > "$GATE_ARTIFACT_DIR/logs/build.log"
    for target in "${targets[@]}"; do
        if gate_run_logged "$GATE_ARTIFACT_DIR/logs/build-${target//\//_}.log" cmake --build "$repo_root/$build_dir" --target "$target" -j"$(nproc)"; then
            built=$((built + 1))
        else failed=$((failed + 1)); fi
    done
    if (( failed )); then gate_record_check "$name" FAIL "$failed/${#targets[@]} targets failed"
    else component_cache_save build "$component_fp"; gate_record_check "$name" PASS "$built targets"; fi
}

check_clang_tidy() {
    local name=clang-tidy repo_root build_dir f rc failed=0 advisory=0 total=0
    if ! command -v clang-tidy >/dev/null 2>&1; then gate_record_check "$name" SKIP "clang-tidy unavailable"; return; fi
    repo_root="$(gate_repo_root)"; build_dir="${GATE_BUILD_DIR:-build}"
    local component_fp cached_artifact=""
    component_fp="$(fingerprint_tidy_surface)"
    if (( ${GATE_COMPONENT_REUSE:-1} )); then cached_artifact="$(component_cache_find tidy "$component_fp" 2>/dev/null || true)"; fi
    if [[ -n "$cached_artifact" ]]; then gate_record_check "$name" PASS "reused unchanged tidy surface from $cached_artifact"; return; fi
    while IFS= read -r f; do
        [[ -n "$f" && -f "$f" ]] || continue
        case "$f" in *.cpp|*.cc|*.cxx|*.c|*.C) ;; *) continue ;; esac
        total=$((total + 1)); rc=0
        gate_run_logged "$GATE_ARTIFACT_DIR/logs/clang-tidy-$(basename "$f").log" clang-tidy -p "$repo_root/$build_dir" "$repo_root/$f" --quiet --extra-arg=-std=c++17 || rc=$?
        if (( rc != 0 )); then
            if grep -qE 'error:|fatal error:' "$GATE_ARTIFACT_DIR/logs/clang-tidy-$(basename "$f").log"; then failed=$((failed + 1)); else advisory=$((advisory + 1)); fi
        fi
    done < <(printf '%s\n' "$(diff_paths_for cpp)" "$(diff_paths_for c)" | sed '/^$/d' | sort -u)
    if (( total == 0 )); then gate_record_check "$name" SKIP "no changed translation units"
    elif (( failed )); then gate_record_check "$name" FAIL "$failed/$total translation units"
    else component_cache_save tidy "$component_fp"; gate_record_check "$name" PASS "$total translation units; $advisory advisory"; fi
}

# ---------- parser registry ----------
declare -a PARSER_NAMES=()
declare -A PARSER_TEST=() PARSER_BINDING=() PARSER_SELECTORS=() PARSER_TAGS=()

load_parser_registry() {
    ((${#PARSER_NAMES[@]} > 0)) && return 0
    local cfg="$OS_AGENT_DIR/config/parser-tests.tsv" name test binding selectors tags extra
    [[ -f "$cfg" ]] || return 1
    while IFS=$'\t' read -r name test binding selectors tags extra; do
        [[ -n "$name" && "$name" != \#* ]] || continue
        [[ -z "$extra" ]] || { gate_log 1 "malformed parser registry row: $name"; return 1; }
        [[ -n "$test" && -n "$binding" && -n "$selectors" ]] || return 1
        [[ -z "${PARSER_TEST[$name]+present}" ]] || return 1
        PARSER_NAMES+=("$name"); PARSER_TEST["$name"]="$test"; PARSER_BINDING["$name"]="$binding"; PARSER_SELECTORS["$name"]="$selectors"; PARSER_TAGS["$name"]="$tags"
    done < "$cfg"
    ((${#PARSER_NAMES[@]} > 0))
}

binding_path() { case "$1" in module:*|fixture:*) printf '%s' "${1#*:}" ;; *) return 1 ;; esac; }

parser_selected() {
    local name="$1" full="${2:-0}" path pattern selectors
    (( full )) && return 0
    diff_path_changed "${PARSER_TEST[$name]}" && return 0
    path="$(binding_path "${PARSER_BINDING[$name]}" 2>/dev/null || true)"
    [[ -n "$path" ]] && diff_path_changed "$path" && return 0
    selectors="${PARSER_SELECTORS[$name]}"
    local IFS=';'; read -ra pats <<< "$selectors"
    for path in "${DIFF_FILES[@]}"; do
        for pattern in "${pats[@]}"; do
            [[ -n "$pattern" ]] || continue
            [[ "$pattern" == @all || "$path" == $pattern ]] && return 0
        done
    done
    return 1
}

parser_registry_validate() {
    local full="${1:-0}" repo_root name test binding path base unregistered=0 malformed=0
    repo_root="$(gate_repo_root)"
    if ! load_parser_registry; then gate_record_check parser-registry INCOMPLETE "missing or malformed config/parser-tests.tsv"; return 1; fi
    for name in "${PARSER_NAMES[@]}"; do
        test="${PARSER_TEST[$name]}"; binding="${PARSER_BINDING[$name]}"; path="$(binding_path "$binding" 2>/dev/null || true)"
        [[ -f "$repo_root/$test" && -n "$path" && -f "$repo_root/$path" ]] || malformed=$((malformed + 1))
    done
    local f bn registered
    while IFS= read -r -d '' f; do
        bn="${f#$repo_root/}"; registered=0
        for name in "${PARSER_NAMES[@]}"; do [[ "${PARSER_TEST[$name]}" == "$bn" ]] && { registered=1; break; }; done
        if (( ! registered )) && { (( full )) || diff_path_changed "$bn"; }; then unregistered=$((unregistered + 1)); fi
    done < <(find "$repo_root/tests" -maxdepth 1 -type f -name '*parser*.py' -print0 2>/dev/null || true)
    if (( malformed )); then gate_record_check parser-registry FAIL "$malformed missing test/binding paths"; return 1
    elif (( unregistered )); then gate_record_check parser-registry UNRESOLVED "$unregistered changed/full parser tests unregistered"; return 1
    else gate_record_check parser-registry PASS "${#PARSER_NAMES[@]} registered"; fi
}

parser_waiver_valid() {
    local name="$1" test="$2" binding="$3" log="$4" repo_root cfg row_name baseline test_sha bind_sha fail_sha expires reason path
    repo_root="$(gate_repo_root)"; cfg="$OS_AGENT_DIR/config/parser-waivers.tsv"
    [[ -f "$cfg" ]] || return 1
    path="$(binding_path "$binding" 2>/dev/null)" || return 1
    diff_path_changed "$test" && return 1
    diff_path_changed "$path" && return 1
    while IFS=$'\t' read -r row_name baseline test_sha bind_sha fail_sha expires reason; do
        [[ -n "$row_name" && "$row_name" != \#* ]] || continue
        [[ "$row_name" == "$name" ]] || continue
        [[ "$(sha256sum "$repo_root/$test" | awk '{print $1}')" == "$test_sha" ]] || return 1
        [[ "$(sha256sum "$repo_root/$path" | awk '{print $1}')" == "$bind_sha" ]] || return 1
        [[ "$(sha256sum "$log" | awk '{print $1}')" == "$fail_sha" ]] || return 1
        [[ $(date -u +%s) -lt $(date -u -d "$expires" +%s 2>/dev/null || echo 0) ]] || return 1
        gate_log 1 "parser waiver accepted: $name baseline=$baseline expires=$expires"
        return 0
    done < "$cfg"
    return 1
}

check_parser_plan() {
    local full="${1:-0}" selected=0 name
    parser_registry_validate "$full" || return 0
    for name in "${PARSER_NAMES[@]}"; do parser_selected "$name" "$full" && selected=$((selected + 1)); done
    if (( selected )); then gate_record_check parser-plan PASS "$selected selected; execution omitted in audit"
    else gate_record_check parser-plan SKIP "no parser tests selected"; fi
}

check_parser_test() {
    local full="${1:-0}" repo_root name test log failed=0 passed=0 waived=0 selected=0 component_fp cached_artifact=""
    parser_registry_validate "$full" || return 0
    repo_root="$(gate_repo_root)"
    local -a selected_names=()
    for name in "${PARSER_NAMES[@]}"; do parser_selected "$name" "$full" && selected_names+=("$name"); done
    selected=${#selected_names[@]}
    if (( selected == 0 )); then gate_record_check parser-test SKIP "no tests selected by diff"; return; fi
    component_fp="$(fingerprint_parser_surface "${selected_names[@]}")"
    if (( ${GATE_COMPONENT_REUSE:-1} && ! full )); then cached_artifact="$(component_cache_find parser "$component_fp" 2>/dev/null || true)"; fi
    if [[ -n "$cached_artifact" ]]; then gate_record_check parser-test PASS "reused $selected selected tests from $cached_artifact"; return; fi
    for name in "${selected_names[@]}"; do
        test="${PARSER_TEST[$name]}"; log="$GATE_ARTIFACT_DIR/logs/parser-$name.log"
        if gate_run_logged "$log" python3 "$repo_root/$test"; then passed=$((passed + 1))
        elif (( ! full )) && parser_waiver_valid "$name" "$test" "${PARSER_BINDING[$name]}" "$log"; then waived=$((waived + 1))
        else failed=$((failed + 1)); fi
    done
    if (( failed )); then gate_record_check parser-test FAIL "$failed/$selected failed; $passed passed; $waived waived"
    else component_cache_save parser "$component_fp"; gate_record_check parser-test PASS "$passed passed; $waived identity-bound waivers"; fi
}

check_fixture_binding() {
    local full="${1:-0}" repo_root name test binding path base selected=0 failed=0
    load_parser_registry || { gate_record_check fixture-binding INCOMPLETE "registry unavailable"; return; }
    repo_root="$(gate_repo_root)"
    for name in "${PARSER_NAMES[@]}"; do
        parser_selected "$name" "$full" || continue
        selected=$((selected + 1)); test="${PARSER_TEST[$name]}"; binding="${PARSER_BINDING[$name]}"; path="$(binding_path "$binding" 2>/dev/null || true)"; base="$(basename "$path")"
        case "$binding" in
            module:*) grep -Fq "$base" "$repo_root/$test" && grep -Eq 'subprocess|importlib|runpy' "$repo_root/$test" || failed=$((failed + 1)) ;;
            fixture:*) grep -Fq "$base" "$repo_root/$test" || failed=$((failed + 1)) ;;
            *) failed=$((failed + 1)) ;;
        esac
    done
    if (( selected == 0 )); then gate_record_check fixture-binding SKIP "no parser tests selected"
    elif (( failed )); then gate_record_check fixture-binding FAIL "$failed/$selected bindings unproven"
    else gate_record_check fixture-binding PASS "$selected bindings"; fi
}

check_skill_validation() {
    local full="${1:-0}" repo_root a b validator
    if (( ! full && ! HAS_SKILL && ! HAS_HARNESS )); then gate_record_check skill-validation SKIP "skill/Harness unchanged"; return; fi
    repo_root="$(gate_repo_root)"; a="$repo_root/.claude/skills/os-agent-task"; b="$repo_root/.agents/skills/os-agent-task"
    [[ -d "$a" && -d "$b" ]] || { gate_record_check skill-validation FAIL "both Skill mirrors are required"; return; }
    if ! diff -qr "$a" "$b" >"$GATE_ARTIFACT_DIR/logs/skill-parity.log" 2>&1; then gate_failure_excerpt "$GATE_ARTIFACT_DIR/logs/skill-parity.log"; gate_record_check skill-validation FAIL "Skill mirrors differ"; return; fi
    validator="$a/scripts/validate-skill.sh"
    [[ -x "$validator" || -f "$validator" ]] || { gate_record_check skill-validation FAIL "canonical validator missing"; return; }
    if gate_run_logged "$GATE_ARTIFACT_DIR/logs/skill-validation.log" bash "$validator"; then gate_record_check skill-validation PASS "parity + canonical validation"
    else gate_record_check skill-validation FAIL "canonical validation failed"; fi
}

check_memory_check() {
    local full="${1:-0}" repo_root checker
    if (( ! full && ! HAS_LEDGER )); then gate_record_check memory-check SKIP "ledgers unchanged"; return; fi
    repo_root="$(gate_repo_root)"; checker="$repo_root/.claude/skills/os-agent-task/scripts/init-project-ledger.sh"
    [[ -f "$checker" ]] || { gate_record_check memory-check UNVERIFIED "ledger checker missing"; return; }
    if gate_run_logged "$GATE_ARTIFACT_DIR/logs/memory-check.log" bash "$checker" check; then gate_record_check memory-check PASS
    else gate_record_check memory-check FAIL "ledger check failed"; fi
}

check_harness_selftest() {
    local full="${1:-0}" test="$OS_AGENT_DIR/tests/test-harness.sh"
    if [[ "${GATE_SKIP_HARNESS_SELFTEST:-0}" == 1 ]]; then gate_record_check harness-selftest SKIP "disabled by environment"; return; fi
    if (( ! HAS_HARNESS )) && [[ "${GATE_FULL_HARNESS:-0}" != 1 ]]; then gate_record_check harness-selftest SKIP "Harness unchanged"; return; fi
    [[ -f "$test" ]] || { gate_record_check harness-selftest FAIL "test-harness.sh missing"; return; }
    local -a args=(--fast); (( full )) && args=(--full)
    if gate_run_logged "$GATE_ARTIFACT_DIR/logs/harness-selftest.log" bash "$test" "${args[@]}"; then gate_record_check harness-selftest PASS "isolated ${args[*]}"
    else gate_record_check harness-selftest FAIL "isolated self-test failed"; fi
}

check_reused_validation() {
    local src="${GATE_REUSED_FROM:-}"
    if [[ -n "$src" && -f "$src/summary.txt" ]] && grep -q '^verdict=PASS$' "$src/summary.txt"; then gate_record_check reused-validation PASS "$src"
    else gate_record_check reused-validation INCOMPLETE "cache record invalid"; fi
}
