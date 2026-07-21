#!/usr/bin/env bash
# target-mapper.sh — map C/C++ source files to verified CMake build targets
#
# Extraction strategy:
#   1. For .cpp/.c/.cc files: parse -o CMakeFiles/<target>.dir/ from compile_commands.json command
#   2. For .h/.hpp headers: derive umbrella target from source directory
#   3. Every returned target is verified against `cmake --build <dir> --target help`
#
# Sets:
#   SOURCE_TARGETS — associative array: abs_source_path -> space-separated target names
#   TARGET_MAP_FAILED — 1 if compile_commands.json missing or unparseable

set -euo pipefail

declare -A SOURCE_TARGETS
declare TARGET_MAP_FAILED=0

# Verified CMake target names, populated by target_mapper_init
declare -A _VALID_TARGETS

# Umbrella target per source directory (used for headers not in compile_commands)
declare -A UMBRELLA_MAP
UMBRELLA_MAP["src"]="llama"
UMBRELLA_MAP["common"]="llama-common"
UMBRELLA_MAP["ggml/src"]="ggml"
UMBRELLA_MAP["tools/server"]="llama-server"
UMBRELLA_MAP["tools/cli"]="llama-cli"
UMBRELLA_MAP["examples/kv-idle-swap-resume"]="llama-kv-idle-swap-resume"
UMBRELLA_MAP["examples/kv-trace-replay"]="llama-kv-trace-replay"
UMBRELLA_MAP["examples/kv-semi-real-multisession"]="llama-kv-semi-real-multisession"
UMBRELLA_MAP["examples/kv-idle-telemetry"]="llama-kv-idle-telemetry"

target_mapper_init() {
    local build_dir="${1:-${GATE_BUILD_DIR:-build}}"
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

    SOURCE_TARGETS=()
    _VALID_TARGETS=()
    TARGET_MAP_FAILED=0

    # Step 1: collect verified target names from cmake
    local cc_json="$repo_root/$build_dir/compile_commands.json"
    if [[ ! -f "$cc_json" ]]; then
        gate_log 1 "target_mapper: compile_commands.json not found at $cc_json"
        TARGET_MAP_FAILED=1
        return
    fi

    # Collect valid targets from cmake --target help
    while IFS= read -r line; do
        line="${line#... }"
        line="${line%% *}"
        [[ -n "$line" ]] && _VALID_TARGETS["$line"]=1
    done < <(cmake --build "$repo_root/$build_dir" --target help 2>&1 | grep '^\.\.\. ' || true)

    if (( ${#_VALID_TARGETS[@]} == 0 )); then
        gate_log 1 "target_mapper: could not enumerate CMake targets"
        TARGET_MAP_FAILED=1
        return
    fi

    gate_log 2 "target_mapper: ${#_VALID_TARGETS[@]} verified CMake targets"

    # Step 2: extract source->target from compile_commands.json -o flags
    # Use process substitution to avoid subshell (pipe loses associative array state)
    local tmp_map
    tmp_map=$(python3 -c "
import json, re

with open('$cc_json') as f:
    data = json.load(f)

for entry in data:
    src = entry.get('file', '')
    cmd = entry.get('command', '')
    m = re.search(r'-o\s+\S*CMakeFiles/([^/]+)\.dir/', cmd)
    if m:
        target = m.group(1)
        print(f'{src}\t{target}')
" 2>/dev/null)

    while IFS=$'\t' read -r src target; do
        [[ -z "$src" ]] && continue
        if [[ -n "${_VALID_TARGETS[$target]:-}" ]]; then
            if [[ -z "${SOURCE_TARGETS[$src]:-}" ]]; then
                SOURCE_TARGETS["$src"]="$target"
            else
                SOURCE_TARGETS["$src"]+=" $target"
            fi
        fi
    done <<< "$tmp_map"

    gate_log 2 "target_mapper: mapped ${#SOURCE_TARGETS[@]} source files to verified targets"
}

# Resolve a header file to an umbrella target, based on the source directory.
# Returns empty string if no umbrella mapping exists.
_umbrella_for_header() {
    local f="$1"
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    local rel="${f#$repo_root/}"

    # Try longest prefix match against UMBRELLA_MAP
    local best=""
    for prefix in "${!UMBRELLA_MAP[@]}"; do
        if [[ "$rel" == "$prefix"/* ]] || [[ "$rel" == "$prefix" ]]; then
            if (( ${#prefix} > ${#best} )); then
                best="$prefix"
            fi
        fi
    done

    if [[ -n "$best" ]]; then
        local t="${UMBRELLA_MAP[$best]}"
        if [[ -n "${_VALID_TARGETS[$t]:-}" ]]; then
            printf '%s' "$t"
            return 0
        fi
    fi
    return 1
}

# Given an absolute source file path, find its CMake build target(s).
# For .cpp/.c/.cc: returns targets from compile_commands.json
# For .h/.hpp:  returns umbrella target if mapping exists
# Returns 1 if no target found.
target_for_file() {
    local src_abs="$1"
    [[ "$src_abs" = /* ]] || src_abs="$(git rev-parse --show-toplevel 2>/dev/null || pwd)/$src_abs"

    # Check SOURCE_TARGETS (populated from compile_commands.json)
    if [[ -n "${SOURCE_TARGETS[$src_abs]:-}" ]]; then
        printf '%s' "${SOURCE_TARGETS[$src_abs]}"
        return 0
    fi

    # For headers, try umbrella mapping
    local ext="${src_abs##*.}"
    case "$ext" in
        h|hpp|hxx|hh|H)
            local t
            if t="$(_umbrella_for_header "$src_abs")"; then
                printf '%s' "$t"
                return 0
            fi
            ;;
    esac

    return 1
}

# Given a list of source files (relative or absolute), return unique verified build targets.
# If strict=1 and any file cannot be mapped, returns non-zero.
targets_for_files() {
    local strict="${1:-1}"
    shift || true
    local files=("$@")
    local repo_root
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

    local unresolved=0
    local -A seen
    local targets=()

    for f in "${files[@]}"; do
        [[ -z "$f" ]] && continue
        local abs="$f"
        [[ "$abs" = /* ]] || abs="$repo_root/$f"

        local found=0
        local tlist
        if tlist=$(target_for_file "$abs" 2>/dev/null); then
            for t in $tlist; do
                if [[ -z "${seen[$t]:-}" ]]; then
                    seen["$t"]=1
                    targets+=("$t")
                fi
            done
            found=1
        fi

        if (( found == 0 )); then
            gate_log 1 "target_mapper: UNRESOLVED — $f has no verified build target"
            unresolved=1
        fi
    done

    if (( unresolved && strict )); then
        return 1
    fi
    printf '%s\n' "${targets[@]}"
}

# Verify a target name exists in the current build system.
target_is_valid() {
    local t="$1"
    [[ -n "${_VALID_TARGETS[$t]:-}" ]]
}
