#!/usr/bin/env bash
# Classify tracked, deleted and untracked changes once.

set -euo pipefail

declare -A DIFF_CLASSES=()
declare -A DIFF_CHANGED=()
declare -a DIFF_FILES=()
declare DIFF_TOTAL=0
declare HAS_CPP=0 HAS_C=0 HAS_H=0 HAS_PY=0 HAS_SH=0 HAS_CMAKE=0 HAS_SKILL=0 HAS_LEDGER=0 HAS_HARNESS=0
declare HAS_TRACKED=0 HAS_UNTRACKED=0 HAS_DELETED_ANY=0

diff_add_class() {
    local key="$1" path="$2"
    DIFF_CLASSES["$key"]+="$path"$'\n'
}

diff_classify_one() {
    local kind="$1" f="$2" ext="${2##*.}"
    DIFF_FILES+=("$f")
    DIFF_CHANGED["$f"]="$kind"
    case "$ext" in
        cpp|cc|cxx|C) diff_add_class "${kind}_cpp" "$f"; [[ "$kind" == deleted ]] || HAS_CPP=1 ;;
        c) diff_add_class "${kind}_c" "$f"; [[ "$kind" == deleted ]] || HAS_C=1 ;;
        h|hpp|hxx|hh|H) diff_add_class "${kind}_h" "$f"; [[ "$kind" == deleted ]] || HAS_H=1 ;;
        py) diff_add_class "${kind}_py" "$f"; [[ "$kind" == deleted ]] || HAS_PY=1 ;;
        sh|bash) diff_add_class "${kind}_sh" "$f"; [[ "$kind" == deleted ]] || HAS_SH=1 ;;
    esac
    if [[ "$f" == *CMakeLists.txt || "$f" == *.cmake || "$f" == CMakePresets.json ]]; then
        diff_add_class "${kind}_cmake" "$f"
        [[ "$kind" == deleted ]] || HAS_CMAKE=1
    fi
    if [[ "$f" == .claude/skills/* || "$f" == .agents/skills/* ]]; then
        diff_add_class "${kind}_skill" "$f"
        [[ "$kind" == deleted ]] || HAS_SKILL=1
    fi
    if [[ "$f" == scripts/os-agent/* ]]; then
        diff_add_class "${kind}_harness" "$f"
        [[ "$kind" == deleted ]] || HAS_HARNESS=1
    fi
    case "$f" in
        PROJECT_STATE.md|ARCHITECTURE.md|DECISIONS.md|EXPERIMENTS.md)
            diff_add_class "${kind}_ledger" "$f"
            [[ "$kind" == deleted ]] || HAS_LEDGER=1
            ;;
    esac
    if [[ "$kind" == deleted ]]; then HAS_DELETED_ANY=1; fi
}

diff_analyze() {
    local repo_root f
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    cd "$repo_root"
    DIFF_CLASSES=(); DIFF_CHANGED=(); DIFF_FILES=(); DIFF_TOTAL=0
    HAS_CPP=0; HAS_C=0; HAS_H=0; HAS_PY=0; HAS_SH=0; HAS_CMAKE=0; HAS_SKILL=0; HAS_LEDGER=0; HAS_HARNESS=0
    HAS_TRACKED=0; HAS_UNTRACKED=0; HAS_DELETED_ANY=0

    while IFS= read -r f; do
        [[ -n "$f" ]] || continue
        diff_classify_one tracked "$f"
        HAS_TRACKED=$((HAS_TRACKED + 1))
    done < <(git diff --name-only --diff-filter=ACMRTUXB HEAD 2>/dev/null || true)

    while IFS= read -r f; do
        [[ -n "$f" ]] || continue
        diff_classify_one deleted "$f"
    done < <(git diff --name-only --diff-filter=D HEAD 2>/dev/null || true)

    while IFS= read -r f; do
        [[ -n "$f" ]] || continue
        [[ "$f" == scripts/os-agent/tests/fixtures/* ]] && continue
        diff_classify_one untracked "$f"
        HAS_UNTRACKED=$((HAS_UNTRACKED + 1))
    done < <(git ls-files --others --exclude-standard 2>/dev/null || true)

    DIFF_TOTAL=${#DIFF_FILES[@]}
    local k
    for k in "${!DIFF_CLASSES[@]}"; do
        DIFF_CLASSES["$k"]="$(printf '%s\n' "${DIFF_CLASSES[$k]}" | sed '/^$/d' | sort -u)"
    done
}

diff_paths_for() {
    local key="$1"
    printf '%s\n' "${DIFF_CLASSES[tracked_$key]:-}" "${DIFF_CLASSES[untracked_$key]:-}" | sed '/^$/d' | sort -u
}

diff_deleted_for() {
    printf '%s\n' "${DIFF_CLASSES[deleted_$1]:-}" | sed '/^$/d' | sort -u
}

diff_path_changed() { [[ -n "${DIFF_CHANGED[$1]+present}" ]]; }

diff_report() {
    local deleted=0 kind
    for kind in cpp c h py sh cmake skill harness ledger; do
        deleted=$((deleted + $(printf '%s\n' "${DIFF_CLASSES[deleted_$kind]:-}" | sed '/^$/d' | wc -l)))
    done
    printf 'DIFF_TOTAL=%d TRACKED=%d UNTRACKED=%d DELETED=%d\n' "$DIFF_TOTAL" "$HAS_TRACKED" "$HAS_UNTRACKED" "$deleted"
    for kind in cpp c h py sh cmake skill harness ledger; do
        local count
        count=$(diff_paths_for "$kind" | wc -l)
        (( count == 0 )) || printf 'DIFF_%s=%d\n' "$kind" "$count"
    done
}
