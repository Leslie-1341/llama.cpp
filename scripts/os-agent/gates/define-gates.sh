#!/usr/bin/env bash
# Mode/profile composition. Expensive checks are not repeated after compatible PASS reuse.

set -euo pipefail

gate_static_checks() {
    check_git_diff_check
    check_shell_syntax
    check_python_syntax
}

gate_build_checks() {
    if (( HAS_CPP || HAS_C || HAS_H || HAS_CMAKE )); then
        check_cmake_configure
        check_compile_commands
        check_incremental_build
    else
        gate_record_check cmake-configure SKIP "no C/C++ or CMake changes"
        gate_record_check compile-commands SKIP "no C/C++ changes"
        gate_record_check incremental-build SKIP "no C/C++ changes"
    fi
}

gate_quick() {
    gate_static_checks
    gate_build_checks
    check_parser_test 0
    check_fixture_binding 0
    check_skill_validation 0
    check_memory_check 0
    check_harness_selftest 0
}

gate_verify() {
    gate_static_checks
    if [[ -n "${GATE_REUSED_FROM:-}" ]]; then
        check_reused_validation
        gate_record_check cmake-configure SKIP "reused unchanged fingerprint"
        gate_record_check compile-commands SKIP "reused unchanged fingerprint"
        gate_record_check incremental-build SKIP "reused unchanged fingerprint"
        gate_record_check parser-registry SKIP "reused unchanged fingerprint"
        gate_record_check parser-test SKIP "reused unchanged fingerprint"
        gate_record_check fixture-binding SKIP "reused unchanged fingerprint"
    else
        gate_build_checks
        check_parser_test 0
        check_fixture_binding 0
    fi
    if (( HAS_CPP || HAS_C )); then check_clang_tidy; else gate_record_check clang-tidy SKIP "no changed translation units"; fi
    check_skill_validation 0
    check_memory_check 0
    check_harness_selftest 0
}

gate_full() {
    gate_static_checks
    gate_build_checks
    check_clang_tidy
    check_parser_test 1
    check_fixture_binding 1
    check_skill_validation 1
    check_memory_check 1
    check_harness_selftest 1
}

gate_audit_profile() {
    TARGET_MAPPER_READ_ONLY=1
    gate_static_checks
    if (( HAS_CMAKE )); then
        gate_record_check cmake-configure UNVERIFIED "audit is read-only; CMake changes require implement/verify"
    else
        gate_record_check cmake-configure SKIP "audit is read-only"
    fi
    if (( HAS_CPP || HAS_C || HAS_H )); then
        check_compile_commands
    else
        gate_record_check compile-commands SKIP "no C/C++ changes"
    fi
    check_parser_plan 0
    check_fixture_binding 0
    check_skill_validation 0
    check_memory_check 0
    gate_record_check incremental-build SKIP "audit is read-only"
    gate_record_check clang-tidy SKIP "audit is read-only"
    gate_record_check harness-selftest SKIP "audit is read-only"
}

run_gate_profile() {
    case "$GATE_PROFILE" in
        audit) gate_audit_profile ;;
        quick) gate_quick ;;
        verify) gate_verify ;;
        full) gate_full ;;
        *) gate_record_check harness-profile INCOMPLETE "unknown profile: $GATE_PROFILE" ;;
    esac
}
