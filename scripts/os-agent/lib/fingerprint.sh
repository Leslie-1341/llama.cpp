#!/usr/bin/env bash
# Stable identity for HEAD + worktree bytes + build configuration + harness version.

set -euo pipefail

fingerprint_compute() {
    local repo_root build_dir version_file
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    build_dir="${GATE_BUILD_DIR:-build}"
    version_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/VERSION"
    {
        printf 'harness=%s\n' "$(cat "$version_file" 2>/dev/null || echo unknown)"
        printf 'head=%s\n' "$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || echo unknown)"
        printf 'build_dir=%s\n' "$build_dir"
        printf 'cmake=%s\n' "$(cmake --version 2>/dev/null | head -1 || true)"
        printf 'python=%s\n' "$(python3 --version 2>&1 || true)"
        git -C "$repo_root" diff --binary HEAD -- 2>/dev/null || true
        local f
        while IFS= read -r f; do
            [[ -f "$repo_root/$f" ]] || continue
            printf 'untracked:%s:' "$f"
            sha256sum "$repo_root/$f" | awk '{print $1}'
        done < <(git -C "$repo_root" ls-files --others --exclude-standard | sort)
        for f in "$repo_root/$build_dir/CMakeCache.txt" "$repo_root/$build_dir/compile_commands.json"; do
            if [[ -f "$f" ]]; then printf '%s:' "$(basename "$f")"; sha256sum "$f" | awk '{print $1}'; else printf '%s:MISSING\n' "$(basename "$f")"; fi
        done
    } | sha256sum | awk '{print $1}'
}

fingerprint_hash_paths() {
    local label="$1"; shift
    local repo_root f
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    {
        printf 'label=%s\nharness=%s\nhead=%s\nbuild=%s\n' "$label" "$GATE_HARNESS_VERSION" "$(git -C "$repo_root" rev-parse HEAD)" "${GATE_BUILD_DIR:-build}"
        for f in "$@"; do
            [[ -n "$f" ]] || continue
            printf 'path=%s\n' "$f"
            if [[ -f "$repo_root/$f" ]]; then sha256sum "$repo_root/$f" | awk '{print $1}'; else printf 'DELETED\n'; fi
        done
        for f in "$repo_root/${GATE_BUILD_DIR:-build}/CMakeCache.txt" "$repo_root/${GATE_BUILD_DIR:-build}/compile_commands.json"; do
            [[ -f "$f" ]] && sha256sum "$f" | awk '{print $1}' || printf 'MISSING:%s\n' "$(basename "$f")"
        done
    } | sha256sum | awk '{print $1}'
}

fingerprint_build_surface() {
    local -a paths=()
    mapfile -t paths < <(printf '%s\n' "$(cpp_changed_files)" "$(diff_paths_for cmake)" | sed '/^$/d' | sort -u)
    fingerprint_hash_paths build "${paths[@]}"
}

fingerprint_tidy_surface() {
    local -a paths=()
    mapfile -t paths < <(printf '%s\n' "$(diff_paths_for cpp)" "$(diff_paths_for c)" | sed '/^$/d' | sort -u)
    fingerprint_hash_paths tidy "${paths[@]}"
}

fingerprint_parser_surface() {
    local -a names=("$@") paths=("scripts/os-agent/config/parser-tests.tsv" "scripts/os-agent/config/parser-waivers.tsv")
    local name binding path changed pattern selectors
    for name in "${names[@]}"; do
        paths+=("${PARSER_TEST[$name]}")
        binding="${PARSER_BINDING[$name]}"; path="$(binding_path "$binding" 2>/dev/null || true)"; [[ -n "$path" ]] && paths+=("$path")
        selectors="${PARSER_SELECTORS[$name]}"; local IFS=';'; read -ra pats <<< "$selectors"
        for changed in "${DIFF_FILES[@]}"; do
            for pattern in "${pats[@]}"; do [[ "$changed" == $pattern ]] && paths+=("$changed"); done
        done
    done
    mapfile -t paths < <(printf '%s\n' "${paths[@]}" | sed '/^$/d' | sort -u)
    fingerprint_hash_paths parser "${paths[@]}"
}
