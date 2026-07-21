#!/usr/bin/env bash
# diff-analyzer.sh — auto-detect changed files relative to HEAD
#
# Coverage: Modified (M), Added (A), Renamed (R), Copied (C), Deleted (D), Untracked.
# Deleted files trigger HAS_DELETED flags so gates can react (e.g., deleted test/CMake
# files need attention).
#
# Sets DIFF_CLASSES: associative array  ext -> newline-separated paths
# Sets HAS_* flags: HAS_CPP, HAS_C, HAS_H, HAS_PY, HAS_SH, HAS_CMAKE,
#                    HAS_SKILL, HAS_LEDGER, HAS_DELETED_CPP, HAS_DELETED_CMAKE, etc.
# Sets DIFF_TOTAL:   total changed files count

set -euo pipefail

declare -A DIFF_CLASSES
declare DIFF_TOTAL=0
declare HAS_CPP=0 HAS_C=0 HAS_H=0 HAS_PY=0 HAS_SH=0 HAS_CMAKE=0 HAS_SKILL=0 HAS_LEDGER=0
declare HAS_TRACKED=0 HAS_UNTRACKED=0
declare HAS_DELETED_CPP=0 HAS_DELETED_C=0 HAS_DELETED_H=0 HAS_DELETED_PY=0
declare HAS_DELETED_SH=0 HAS_DELETED_CMAKE=0 HAS_DELETED_ANY=0

# Build exclusion pattern from artifact (so .gate-artifacts or custom prefix is never scanned)
_artifact_exclude_pattern() {
    if [[ -n "${GATE_ARTIFACT_PREFIX_USED:-}" ]]; then
        printf '%s' "${GATE_ARTIFACT_PREFIX_USED}"
    else
        printf '%s' "${GATE_ARTIFACT_PREFIX:-/tmp/os-agent-gate}"
    fi
}

diff_analyze() {
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    cd "$repo_root"

    # Reset state
    DIFF_CLASSES=()
    DIFF_TOTAL=0
    HAS_CPP=0; HAS_C=0; HAS_H=0; HAS_PY=0; HAS_SH=0; HAS_CMAKE=0; HAS_SKILL=0; HAS_LEDGER=0
    HAS_TRACKED=0; HAS_UNTRACKED=0
    HAS_DELETED_CPP=0; HAS_DELETED_C=0; HAS_DELETED_H=0; HAS_DELETED_PY=0
    HAS_DELETED_SH=0; HAS_DELETED_CMAKE=0; HAS_DELETED_ANY=0

    local exclude_re
    exclude_re="$(_artifact_exclude_pattern)"

    # --- Tracked changes: MADRC (Modified, Added, Deleted, Renamed, Copied) ---
    local tracked_files
    tracked_files=$(git diff --name-only --diff-filter=MADRC HEAD 2>/dev/null || true)

    # --- Deleted files (separate tracking to set HAS_DELETED_*) ---
    local deleted_files
    deleted_files=$(git diff --name-only --diff-filter=D HEAD 2>/dev/null || true)

    # --- Untracked files (not in .gitignore), excluding artifacts + test fixtures ---
    local untracked_files
    untracked_files=$(git ls-files --others --exclude-standard 2>/dev/null | grep -vE "scripts/os-agent/tests/fixtures/|${exclude_re}" || true)

    _classify_files() {
        local source_label="$1"  # "tracked" or "untracked" or "deleted"
        shift
        local files=("$@")
        local f ext
        for f in "${files[@]}"; do
            [[ -z "$f" ]] && continue
            ext="${f##*.}"

            # --- extension classification ---
            case "$ext" in
                cpp|cc|cxx|C)
                    if [[ "$source_label" == "deleted" ]]; then
                        DIFF_CLASSES["deleted_cpp"]+="$f"$'\n'; HAS_DELETED_CPP=1; HAS_DELETED_ANY=1
                    else
                        DIFF_CLASSES["cpp"]+="$f"$'\n'; HAS_CPP=1
                    fi ;;
                c)
                    if [[ "$source_label" == "deleted" ]]; then
                        DIFF_CLASSES["deleted_c"]+="$f"$'\n'; HAS_DELETED_C=1; HAS_DELETED_ANY=1
                    else
                        DIFF_CLASSES["c"]+="$f"$'\n'; HAS_C=1
                    fi ;;
                h|hpp|hxx|hh|H)
                    if [[ "$source_label" == "deleted" ]]; then
                        DIFF_CLASSES["deleted_h"]+="$f"$'\n'; HAS_DELETED_H=1; HAS_DELETED_ANY=1
                    else
                        DIFF_CLASSES["h"]+="$f"$'\n'; HAS_H=1
                        # Headers in C/C++ source dirs -> cpp_headers
                        if [[ "$f" == ggml/* ]] || [[ "$f" == src/* ]] || [[ "$f" == common/* ]] || \
                           [[ "$f" == tools/* ]] || [[ "$f" == examples/* ]] || [[ "$f" == tests/* ]] || \
                           [[ "$f" == pocs/* ]]; then
                            DIFF_CLASSES["cpp_headers"]+="$f"$'\n'
                        fi
                    fi ;;
                py)
                    if [[ "$source_label" == "deleted" ]]; then
                        DIFF_CLASSES["deleted_py"]+="$f"$'\n'; HAS_DELETED_PY=1; HAS_DELETED_ANY=1
                    else
                        DIFF_CLASSES["py"]+="$f"$'\n'; HAS_PY=1
                    fi ;;
                sh|bash)
                    if [[ "$source_label" == "deleted" ]]; then
                        DIFF_CLASSES["deleted_sh"]+="$f"$'\n'; HAS_DELETED_SH=1; HAS_DELETED_ANY=1
                    else
                        DIFF_CLASSES["sh"]+="$f"$'\n'; HAS_SH=1
                    fi ;;
                cmake|txt)
                    if [[ "$f" == *CMakeLists.txt ]] || [[ "$f" == *.cmake ]]; then
                        if [[ "$source_label" == "deleted" ]]; then
                            DIFF_CLASSES["deleted_cmake"]+="$f"$'\n'; HAS_DELETED_CMAKE=1; HAS_DELETED_ANY=1
                        else
                            DIFF_CLASSES["cmake"]+="$f"$'\n'; HAS_CMAKE=1
                        fi
                    fi ;;
            esac

            # Skill files: under .claude/skills/ or .agents/skills/
            if [[ "$source_label" != "deleted" ]]; then
                if [[ "$f" == .claude/skills/* ]] || [[ "$f" == .agents/skills/* ]]; then
                    DIFF_CLASSES["skill"]+="$f"$'\n'; HAS_SKILL=1
                fi
                # Ledger files
                case "$f" in
                    PROJECT_STATE.md|ARCHITECTURE.md|DECISIONS.md|EXPERIMENTS.md)
                        DIFF_CLASSES["ledger"]+="$f"$'\n'; HAS_LEDGER=1 ;;
                esac
            else
                # Deleted skill or ledger files
                if [[ "$f" == .claude/skills/* ]] || [[ "$f" == .agents/skills/* ]]; then
                    DIFF_CLASSES["deleted_skill"]+="$f"$'\n'; HAS_DELETED_ANY=1
                fi
                case "$f" in
                    PROJECT_STATE.md|ARCHITECTURE.md|DECISIONS.md|EXPERIMENTS.md)
                        DIFF_CLASSES["deleted_ledger"]+="$f"$'\n'; HAS_DELETED_ANY=1 ;;
                esac
            fi

            DIFF_TOTAL=$((DIFF_TOTAL + 1))
        done
    }

    # Process tracked (non-deleted) files
    local -a tracked_arr=()
    while IFS= read -r f; do [[ -n "$f" ]] && tracked_arr+=("$f"); done <<< "$tracked_files"
    _classify_files "tracked" "${tracked_arr[@]}"
    HAS_TRACKED=${#tracked_arr[@]}

    # Process deleted files
    local -a deleted_arr=()
    while IFS= read -r f; do [[ -n "$f" ]] && deleted_arr+=("$f"); done <<< "$deleted_files"
    _classify_files "deleted" "${deleted_arr[@]}"

    # Process untracked files
    local -a untracked_arr=()
    while IFS= read -r f; do [[ -n "$f" ]] && untracked_arr+=("$f"); done <<< "$untracked_files"
    _classify_files "untracked" "${untracked_arr[@]}"
    HAS_UNTRACKED=${#untracked_arr[@]}

    # Recalculate DIFF_TOTAL (tracked + untracked + deleted)
    DIFF_TOTAL=$((HAS_TRACKED + HAS_UNTRACKED + ${#deleted_arr[@]}))

    # Deduplicate
    for k in "${!DIFF_CLASSES[@]}"; do
        DIFF_CLASSES["$k"]="$(printf '%s\n' "${DIFF_CLASSES[$k]}" | sed '/^$/d' | sort -u)"
    done
}

diff_report() {
    local del_total=0
    local del_cpp; del_cpp=$(printf '%s\n' "${DIFF_CLASSES[deleted_cpp]:-}" | sed '/^$/d' | wc -l)
    local del_c;   del_c=$(printf '%s\n' "${DIFF_CLASSES[deleted_c]:-}" | sed '/^$/d' | wc -l)
    local del_h;   del_h=$(printf '%s\n' "${DIFF_CLASSES[deleted_h]:-}" | sed '/^$/d' | wc -l)
    local del_py;  del_py=$(printf '%s\n' "${DIFF_CLASSES[deleted_py]:-}" | sed '/^$/d' | wc -l)
    local del_sh;  del_sh=$(printf '%s\n' "${DIFF_CLASSES[deleted_sh]:-}" | sed '/^$/d' | wc -l)
    local del_cmake; del_cmake=$(printf '%s\n' "${DIFF_CLASSES[deleted_cmake]:-}" | sed '/^$/d' | wc -l)
    del_total=$((del_cpp + del_c + del_h + del_py + del_sh + del_cmake))

    printf 'DIFF_TOTAL=%d TRACKED=%d UNTRACKED=%d DELETED=%d\n' \
        "$DIFF_TOTAL" "$HAS_TRACKED" "$HAS_UNTRACKED" "$del_total"
    for k in cpp c py sh cmake skill ledger cpp_headers h \
             deleted_cpp deleted_c deleted_h deleted_py deleted_sh deleted_cmake; do
        local count
        count=$(printf '%s\n' "${DIFF_CLASSES[$k]:-}" | sed '/^$/d' | wc -l)
        if (( count > 0 )); then
            printf 'DIFF_%s=%d\n' "$k" "$count"
        fi
    done
}
