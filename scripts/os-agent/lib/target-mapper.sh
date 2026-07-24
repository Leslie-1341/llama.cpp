#!/usr/bin/env bash
# Parse compile_commands.json and CMake targets once per gate.

set -euo pipefail

declare -A SOURCE_TARGETS=()
declare -A VALID_TARGETS=()
declare TARGET_MAP_FAILED=0 TARGET_MAP_ERROR="" TARGET_MAP_ENTRIES=0

declare -A UMBRELLA_MAP=(
    [src]=llama
    [common]=llama-common
    [ggml/src]=ggml
    [tools/server]=llama-server
    [tools/cli]=llama-cli
    [examples/kv-idle-swap-resume]=llama-kv-idle-swap-resume
    [examples/kv-trace-replay]=llama-kv-trace-replay
    [examples/kv-semi-real-multisession]=llama-kv-semi-real-multisession
    [examples/kv-idle-telemetry]=llama-kv-idle-telemetry
)

target_mapper_init() {
    local build_dir="${1:-${GATE_BUILD_DIR:-build}}" read_only="${2:-${TARGET_MAPPER_READ_ONLY:-0}}" repo_root cc_json map_file targets_file
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    cc_json="$repo_root/$build_dir/compile_commands.json"
    SOURCE_TARGETS=(); VALID_TARGETS=(); TARGET_MAP_FAILED=0; TARGET_MAP_ERROR=""; TARGET_MAP_ENTRIES=0

    if [[ ! -f "$cc_json" ]]; then
        TARGET_MAP_FAILED=1; TARGET_MAP_ERROR="compile_commands.json missing: $cc_json"; return 0
    fi

    map_file="$GATE_ARTIFACT_DIR/compile-map.tsv"
    targets_file="$GATE_ARTIFACT_DIR/cmake-targets.txt"
    if (( ! read_only )); then
        if ! gate_run_logged "$GATE_ARTIFACT_DIR/logs/cmake-target-help.log" \
            cmake --build "$repo_root/$build_dir" --target help; then
            TARGET_MAP_FAILED=1; TARGET_MAP_ERROR="cannot enumerate CMake targets"; return 0
        fi
        sed -n 's/^\.\.\. \([^ ]*\).*/\1/p' "$GATE_ARTIFACT_DIR/logs/cmake-target-help.log" | sort -u > "$targets_file"
        while IFS= read -r t; do [[ -n "$t" ]] && VALID_TARGETS["$t"]=1; done < "$targets_file"
        if (( ${#VALID_TARGETS[@]} == 0 )); then
            TARGET_MAP_FAILED=1; TARGET_MAP_ERROR="CMake target list is empty"; return 0
        fi
    fi

    if ! python3 - "$cc_json" "$map_file" <<'PY'
import json, os, re, shlex, sys
src, out = sys.argv[1:]
with open(src, encoding="utf-8") as f:
    data = json.load(f)
rows = []
for e in data:
    p = e.get("file", "")
    if not p:
        continue
    if not os.path.isabs(p):
        p = os.path.join(e.get("directory", "."), p)
    p = os.path.realpath(p)
    cmd = e.get("command")
    if not cmd and e.get("arguments"):
        cmd = " ".join(shlex.quote(x) for x in e["arguments"])
    m = re.search(r"(?:^|\s)-o\s+\S*CMakeFiles/([^/]+)\.dir/", cmd or "")
    if m:
        rows.append((p, m.group(1)))
with open(out, "w", encoding="utf-8") as f:
    for p, t in rows:
        f.write(f"{p}\t{t}\n")
print(len(data))
PY
    then
        TARGET_MAP_FAILED=1; TARGET_MAP_ERROR="invalid compile_commands.json"; return 0
    fi > "$GATE_ARTIFACT_DIR/compile-map-count.txt"
    TARGET_MAP_ENTRIES=$(cat "$GATE_ARTIFACT_DIR/compile-map-count.txt")

    local src target
    if (( read_only )); then
        while IFS=$'\t' read -r src target; do [[ -n "$target" ]] && VALID_TARGETS["$target"]=1; done < "$map_file"
    fi
    while IFS=$'\t' read -r src target; do
        [[ -n "$src" && -n "${VALID_TARGETS[$target]:-}" ]] || continue
        if [[ -z "${SOURCE_TARGETS[$src]:-}" ]]; then SOURCE_TARGETS["$src"]="$target"; else SOURCE_TARGETS["$src"]+=" $target"; fi
    done < "$map_file"
}

umbrella_for_header() {
    local src="$1" repo_root rel best="" prefix t
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    [[ "$src" = /* ]] || src="$repo_root/$src"
    rel="${src#$repo_root/}"
    for prefix in "${!UMBRELLA_MAP[@]}"; do
        if [[ "$rel" == "$prefix" || "$rel" == "$prefix"/* ]]; then
            (( ${#prefix} > ${#best} )) && best="$prefix"
        fi
    done
    [[ -n "$best" ]] || return 1
    t="${UMBRELLA_MAP[$best]}"
    [[ -n "${VALID_TARGETS[$t]:-}" ]] || return 1
    printf '%s' "$t"
}

target_for_file() {
    local src="$1" repo_root ext t
    repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    [[ "$src" = /* ]] || src="$repo_root/$src"
    src="$(realpath -m "$src")"
    if [[ -n "${SOURCE_TARGETS[$src]:-}" ]]; then printf '%s' "${SOURCE_TARGETS[$src]}"; return 0; fi
    ext="${src##*.}"
    case "$ext" in h|hpp|hxx|hh|H) t="$(umbrella_for_header "$src")" && { printf '%s' "$t"; return 0; } ;; esac
    return 1
}

targets_for_files() {
    local strict="${1:-1}"; shift || true
    local -A seen=(); local -a out=(); local unresolved=0 f list t
    for f in "$@"; do
        [[ -n "$f" ]] || continue
        if list="$(target_for_file "$f" 2>/dev/null)"; then
            for t in $list; do [[ -n "${seen[$t]:-}" ]] || { seen["$t"]=1; out+=("$t"); }; done
        else
            gate_log 1 "target unresolved: $f"
            unresolved=1
        fi
    done
    (( unresolved && strict )) && return 1
    printf '%s\n' "${out[@]}"
}
