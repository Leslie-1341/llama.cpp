#!/usr/bin/env bash
# define-gates.sh — gate definitions per mode
# Each gate_* function runs the appropriate checks in order.

set -euo pipefail

# --- implement gate: full checks ---
gate_implement() {
    gate_log 1 "Gate IMPLEMENT: running full check suite"

    check_git_diff_check

    if (( HAS_CPP || HAS_C || HAS_H )); then
        target_mapper_init "${GATE_BUILD_DIR:-build}"
        check_cmake_configure
        check_compile_commands
        check_clang_tidy
        check_incremental_build
    else
        gate_record_check "cmake-configure" "SKIP" "no C/C++ changes"
        gate_record_check "compile-commands" "SKIP" "no C/C++ changes"
        gate_record_check "clang-tidy" "SKIP" "no C/C++ changes"
        gate_record_check "incremental-build" "SKIP" "no C/C++ changes"
    fi

    # If C++ files were deleted, flag it (but don't block — deletion may be intentional)
    if (( HAS_DELETED_CPP || HAS_DELETED_C || HAS_DELETED_H )); then
        local deleted_count
        deleted_count=$(printf '%s\n' "${DIFF_CLASSES[deleted_cpp]:-}${DIFF_CLASSES[deleted_c]:-}${DIFF_CLASSES[deleted_h]:-}" | sed '/^$/d' | wc -l)
        gate_record_check "deleted-cpp-files" "PASS" "$deleted_count C/C++ files deleted — verify dependencies"
    fi

    check_shell_syntax
    check_python_syntax
    check_parser_test
    check_skill_validation
    check_memory_check
}

# --- review gate: diff-focused checks ---
gate_review() {
    gate_log 1 "Gate REVIEW: running review checks"

    check_git_diff_check
    check_shell_syntax
    check_python_syntax

    if (( HAS_CPP || HAS_C || HAS_H )); then
        target_mapper_init "${GATE_BUILD_DIR:-build}"
        check_compile_commands
        check_clang_tidy
        check_incremental_build
    else
        gate_record_check "compile-commands" "SKIP" "no C/C++ changes"
        gate_record_check "clang-tidy" "SKIP" "no C/C++ changes"
        gate_record_check "incremental-build" "SKIP" "no C/C++ changes"
    fi

    check_cmake_configure
    check_parser_test
    check_skill_validation

    if (( HAS_DELETED_ANY )); then
        gate_record_check "deleted-files" "PASS" "deleted files noted — review impact"
    fi
}

# --- review-fix gate: targeted fix verification ---
gate_review_fix() {
    gate_log 1 "Gate REVIEW-FIX: running fix verification"

    check_git_diff_check
    check_shell_syntax
    check_python_syntax

    if (( HAS_CPP || HAS_C || HAS_H )); then
        target_mapper_init "${GATE_BUILD_DIR:-build}"
        check_compile_commands
        check_incremental_build
    else
        gate_record_check "compile-commands" "SKIP" "no C/C++ changes"
        gate_record_check "incremental-build" "SKIP" "no C/C++ changes"
    fi

    check_parser_test
    check_skill_validation
}

# --- audit gate: classification and read-only checks only ---
gate_audit() {
    gate_log 1 "Gate AUDIT: read-only classification"

    check_git_diff_check
    check_shell_syntax
    check_python_syntax

    if (( HAS_CPP || HAS_C || HAS_H )); then
        target_mapper_init "${GATE_BUILD_DIR:-build}"
        check_compile_commands

        # Audit: only check mapping, don't build
        local cpp_files
        cpp_files="$(printf '%s\n' "${DIFF_CLASSES[cpp]:-}" | sed '/^$/d' || true)"
        if [[ -n "$cpp_files" ]]; then
            local resolved
            if resolved=$(targets_for_files 0 $cpp_files 2>/dev/null); then
                gate_record_check "target-resolution" "PASS" "all C++ files mapped to verified targets"
            else
                gate_record_check "target-resolution" "UNRESOLVED" "some C++ files have no verified target"
            fi
        else
            gate_record_check "target-resolution" "SKIP" "no C++ files"
        fi
    else
        gate_record_check "compile-commands" "SKIP" "no C/C++ changes"
        gate_record_check "target-resolution" "SKIP" "no C++ files"
    fi

    if (( HAS_DELETED_ANY )); then
        gate_record_check "deleted-files" "PASS" "deleted files noted"
    fi

    check_parser_test
    check_skill_validation
    check_memory_check
}
