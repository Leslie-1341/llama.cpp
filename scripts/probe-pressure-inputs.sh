#!/usr/bin/env bash
# Stage 3A-1A: read-only pressure input probe
# Confirms cgroup, RSS, PSI data sources and their read cost on the current server.
# No model, no source modification, no ledger modification.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
N_READS="${N_READS:-100}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/oscomp/kv_logs/probe_pressure_inputs_${TIMESTAMP}_$$}"
RUNNER="$(realpath "${BASH_SOURCE[0]}")"

# Mock overrides for testing — when set, use these paths instead of real /proc /sys
PROC_SELF_CGROUP="${PROC_SELF_CGROUP:-/proc/self/cgroup}"
PROC_SELF_MOUNTINFO="${PROC_SELF_MOUNTINFO:-/proc/self/mountinfo}"
PROC_SELF_STATM="${PROC_SELF_STATM:-/proc/self/statm}"
PROC_SELF_STATUS="${PROC_SELF_STATUS:-/proc/self/status}"
PROC_PRESSURE_MEMORY="${PROC_PRESSURE_MEMORY:-/proc/pressure/memory}"
SYS_FS_CGROUP="${SYS_FS_CGROUP:-/sys/fs/cgroup}"

# ── helpers ──────────────────────────────────────────────────────────────
die()   { printf 'error: %s\n' "$*" >&2; exit 2; }
warn()  { printf 'WARN: %s\n' "$*" >&2; }
info()  { printf 'INFO: %s\n' "$*" >&2; }

# ── parameter validation ─────────────────────────────────────────────────
[[ "$N_READS" =~ ^[1-9][0-9]*$ ]] || die "N_READS must be a positive integer"
[[ "$N_READS" -ge 10 ]] || warn "N_READS=$N_READS is low; percentiles may be unreliable"
command -v git   >/dev/null || die "missing git"
command -v awk   >/dev/null || die "missing awk"
command -v sort  >/dev/null || die "missing sort"
command -v uname >/dev/null || die "missing uname"
[[ -d "$ROOT" ]] || die "ROOT not a directory: $ROOT"

# ── output directory ─────────────────────────────────────────────────────
if [[ -e "$OUTPUT_DIR" ]]; then
    die "output directory already exists: $OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# ── metadata ─────────────────────────────────────────────────────────────
info "collecting environment metadata"
git -C "$ROOT" rev-parse HEAD          > "$OUTPUT_DIR/head"
git -C "$ROOT" status --porcelain      > "$OUTPUT_DIR/worktree_status"
uname -r                               > "$OUTPUT_DIR/kernel_release"
uname -m                               > "$OUTPUT_DIR/arch"
cat /proc/version 2>/dev/null          > "$OUTPUT_DIR/proc_version" || true
printf 'bash %s\n' "$BASH_VERSION"     > "$OUTPUT_DIR/shell_version"
printf '%s\n' "$(realpath "${BASH_SOURCE[0]}")" > "$OUTPUT_DIR/runner_path"
sha256sum "$RUNNER" | awk '{print $1}'  > "$OUTPUT_DIR/runner_sha256"
printf '%q' "$0" > "$OUTPUT_DIR/cmd_raw"
printf '\n'      >> "$OUTPUT_DIR/cmd_raw"
{
    printf 'N_READS=%s\n' "$N_READS"
    printf 'TIMESTAMP=%s\n' "$TIMESTAMP"
    printf 'OUTPUT_DIR=%s\n' "$OUTPUT_DIR"
    printf 'ROOT=%s\n' "$ROOT"
} > "$OUTPUT_DIR/parameters"

WORKTREE_DIRTY=0
[[ -s "$OUTPUT_DIR/worktree_status" ]] && WORKTREE_DIRTY=1

HEAD="$(cat "$OUTPUT_DIR/head")"
KERNEL="$(cat "$OUTPUT_DIR/kernel_release")"

# ── cgroup version & memory path resolution ──────────────────────────────
info "resolving cgroup memory path"

CGROUP_V1_PATH=""
CGROUP_V2_PATH=""
CGROUP_VERSION="none"
CGROUP_MEM_PATH=""
CGROUP_PATH_AMBIGUOUS=0
CGROUP_RESOLVE_NOTES=""

# --- cgroup v2 (unified hierarchy) ---
CGROUP_SELF=$(cat "$PROC_SELF_CGROUP" 2>/dev/null) || true
MOUNTINFO=$(cat "$PROC_SELF_MOUNTINFO" 2>/dev/null) || true
printf '%s\n' "$CGROUP_SELF" > "$OUTPUT_DIR/raw_proc_self_cgroup"
printf '%s\n' "$MOUNTINFO"   > "$OUTPUT_DIR/raw_proc_self_mountinfo"

V2_REL_PATH=""
V2_CANDIDATES=0
while IFS= read -r line; do
    [[ "$line" == "0::"* ]] || continue
    V2_CANDIDATES=$((V2_CANDIDATES + 1))
    V2_REL_PATH="${line#0::}"
done <<< "$CGROUP_SELF"

# Find cgroup2 mount point
V2_MOUNT=""
V2_MOUNT_CANDIDATES=0
while IFS= read -r line; do
    # mountinfo fields: mnt_id parent_id major:minor root mount_point options ... - fstype opts...
    # cgroup2 line looks like: ... /sys/fs/cgroup cgroup2 ...
    [[ "$line" == *" - cgroup2 "* ]] || continue
    V2_MOUNT_CANDIDATES=$((V2_MOUNT_CANDIDATES + 1))
    # mount_point is field 5
    V2_MOUNT=$(echo "$line" | awk '{print $5}')
done <<< "$MOUNTINFO"

if [[ "$V2_CANDIDATES" -gt 0 && -n "$V2_MOUNT" ]]; then
    CGROUP_VERSION="v2"
    if [[ "$V2_CANDIDATES" -gt 1 ]]; then
        CGROUP_PATH_AMBIGUOUS=1
        CGROUP_RESOLVE_NOTES="v2: $V2_CANDIDATES cgroup lines; using first (0::)$V2_REL_PATH; "
    fi
    if [[ "$V2_MOUNT_CANDIDATES" -gt 1 ]]; then
        CGROUP_PATH_AMBIGUOUS=1
        CGROUP_RESOLVE_NOTES+="v2: $V2_MOUNT_CANDIDATES cgroup2 mounts; using $V2_MOUNT; "
    fi
    # Normalise: root "/" path should not produce double-slash
    if [[ "$V2_REL_PATH" == "/" ]]; then
        CGROUP_V2_PATH="$V2_MOUNT"
    else
        CGROUP_V2_PATH="$V2_MOUNT/${V2_REL_PATH#/}"
    fi
    CGROUP_MEM_PATH="$CGROUP_V2_PATH"
fi

# --- cgroup v1 memory controller ---
V1_MEM_LINES=""
V1_MEM_CANDIDATES=0
V1_MEM_REL_PATH=""
while IFS= read -r line; do
    # v1 format: controller_id:controller_list:relative_path
    # memory controller: either "memory" or listed with others like "cpu,memory"
    if echo "$line" | grep -qE '^[0-9]+:.*memory.*:'; then
        V1_MEM_CANDIDATES=$((V1_MEM_CANDIDATES + 1))
        V1_MEM_LINES+="$line"$'\n'
        V1_MEM_REL_PATH=$(echo "$line" | awk -F: '{print $NF}')
    fi
done <<< "$CGROUP_SELF"

V1_MOUNT=""
V1_MOUNT_CANDIDATES=0
while IFS= read -r line; do
    # cgroup v1 memory mount: fstype is "cgroup" and superblock options include "memory"
    [[ "$line" == *" - cgroup "* ]] || continue
    echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="-") break; for(j=i+1;j<=NF;j++) print $j}' | grep -q 'memory' || continue
    V1_MOUNT_CANDIDATES=$((V1_MOUNT_CANDIDATES + 1))
    V1_MOUNT=$(echo "$line" | awk '{print $5}')
done <<< "$MOUNTINFO"

if [[ "$V1_MEM_CANDIDATES" -gt 0 && -n "$V1_MOUNT" && "$CGROUP_VERSION" == "none" ]]; then
    CGROUP_VERSION="v1"
    if [[ "$V1_MEM_CANDIDATES" -gt 1 ]]; then
        CGROUP_PATH_AMBIGUOUS=1
        CGROUP_RESOLVE_NOTES+="v1: $V1_MEM_CANDIDATES memory controller lines; "
    fi
    if [[ "$V1_MOUNT_CANDIDATES" -gt 1 ]]; then
        CGROUP_PATH_AMBIGUOUS=1
        CGROUP_RESOLVE_NOTES+="v1: $V1_MOUNT_CANDIDATES memory mounts; "
    fi
    CGROUP_V1_PATH="$V1_MOUNT/${V1_MEM_REL_PATH#/}"
    CGROUP_MEM_PATH="$CGROUP_V1_PATH"
    if [[ "$CGROUP_VERSION" == "v2" ]]; then
        CGROUP_PATH_AMBIGUOUS=1
        CGROUP_RESOLVE_NOTES+="both v1 and v2 detected; preferring v2; "
    fi
elif [[ "$V1_MEM_CANDIDATES" -gt 0 && "$CGROUP_VERSION" == "v2" ]]; then
    CGROUP_V1_PATH="$V1_MOUNT/${V1_MEM_REL_PATH#/}"
    CGROUP_PATH_AMBIGUOUS=1
    CGROUP_RESOLVE_NOTES+="v1 memory also present at $CGROUP_V1_PATH; "
fi

# Record resolution results
{
    printf 'cgroup_version=%s\n' "$CGROUP_VERSION"
    printf 'cgroup_mem_path=%s\n' "$CGROUP_MEM_PATH"
    printf 'cgroup_v1_path=%s\n' "${CGROUP_V1_PATH:-unavailable}"
    printf 'cgroup_v2_path=%s\n' "${CGROUP_V2_PATH:-unavailable}"
    printf 'cgroup_path_ambiguous=%d\n' "$CGROUP_PATH_AMBIGUOUS"
    printf 'cgroup_resolve_notes=%s\n' "${CGROUP_RESOLVE_NOTES:-none}"
    printf 'v1_candidates=%d\n' "$V1_MEM_CANDIDATES"
    printf 'v2_candidates=%d\n' "$V2_CANDIDATES"
    printf 'v1_mount=%s\n' "${V1_MOUNT:-unavailable}"
    printf 'v2_mount=%s\n' "${V2_MOUNT:-unavailable}"
} > "$OUTPUT_DIR/cgroup_resolution"

if [[ "$CGROUP_VERSION" == "none" ]]; then
    warn "no cgroup memory controller found; cgroup sources will be unavailable"
fi

# ── data source registration ─────────────────────────────────────────────
info "registering data sources"

# Each source is registered as: label|path|parser_type|is_multi_line
# parser_type: raw, first_field, key_value_some_full, key_value_vmrss, statm_fields
# is_multi_line: 0 or 1
SOURCES=()

register_source() {
    local label="$1" path="$2" parser="$3" multi="$4"
    SOURCES+=("${label}|${path}|${parser}|${multi}")
}

# cgroup v2 sources
if [[ "$CGROUP_VERSION" == "v2" && -n "$CGROUP_MEM_PATH" ]]; then
    register_source "cgroup_memory_current"  "$CGROUP_MEM_PATH/memory.current"  "first_field" 0
    register_source "cgroup_memory_max"      "$CGROUP_MEM_PATH/memory.max"      "first_field" 0
    register_source "cgroup_memory_high"     "$CGROUP_MEM_PATH/memory.high"     "first_field" 0
    register_source "cgroup_memory_pressure" "$CGROUP_MEM_PATH/memory.pressure" "psi"         1
fi

# cgroup v1 sources (only if v2 not primary; otherwise informational)
if [[ "$CGROUP_VERSION" == "v1" && -n "$CGROUP_MEM_PATH" ]]; then
    register_source "cgroup_memory_usage"     "$CGROUP_MEM_PATH/memory.usage_in_bytes"  "first_field" 0
    register_source "cgroup_memory_limit"     "$CGROUP_MEM_PATH/memory.limit_in_bytes"  "first_field" 0
    register_source "cgroup_memory_softlimit" "$CGROUP_MEM_PATH/memory.soft_limit_in_bytes" "first_field" 0
    register_source "cgroup_memory_pressure"  "$CGROUP_MEM_PATH/memory.pressure_level"  "first_field" 0
fi

# If v2 is primary but v1 also exists, register v1 sources as supplementary
if [[ "$CGROUP_VERSION" == "v2" && -n "${CGROUP_V1_PATH:-}" && "$CGROUP_V1_PATH" != "unavailable" ]]; then
    register_source "cgroup_v1_memory_usage"  "$CGROUP_V1_PATH/memory.usage_in_bytes"  "first_field" 0
    register_source "cgroup_v1_memory_limit"  "$CGROUP_V1_PATH/memory.limit_in_bytes"  "first_field" 0
fi

# /proc sources (always registered)
register_source "proc_statm"          "$PROC_SELF_STATM"       "statm"         0
register_source "proc_status"         "$PROC_SELF_STATUS"      "status"        1
register_source "proc_pressure_mem"   "$PROC_PRESSURE_MEMORY"  "psi"           1

# ── read loop ────────────────────────────────────────────────────────────
info "starting read loop: N_READS=$N_READS across ${#SOURCES[@]} sources"

# Parse a value from raw content according to parser type
parse_value() {
    local raw="$1" parser="$2"
    case "$parser" in
        first_field)
            # Return first whitespace-delimited token
            printf '%s' "${raw%%[[:space:]]*}"
            ;;
        psi)
            # PSI format: "some avg10=... avg60=... avg300=... total=..."
            # Return "some_avg10=X some_total=Y full_avg10=Z full_total=W"
            local s_avg10 s_total f_avg10 f_total
            s_avg10=$(echo "$raw" | awk '/^some /{for(i=1;i<=NF;i++){if($i~/avg10=/){v=$i;sub(/avg10=/,"",v);print v;exit}}}')
            s_total=$(echo "$raw" | awk '/^some /{for(i=1;i<=NF;i++){if($i~/total=/){v=$i;sub(/total=/,"",v);print v;exit}}}')
            f_avg10=$(echo "$raw" | awk '/^full /{for(i=1;i<=NF;i++){if($i~/avg10=/){v=$i;sub(/avg10=/,"",v);print v;exit}}}')
            f_total=$(echo "$raw" | awk '/^full /{for(i=1;i<=NF;i++){if($i~/total=/){v=$i;sub(/total=/,"",v);print v;exit}}}')
            printf 'some_avg10=%s some_total=%s full_avg10=%s full_total=%s' \
                "${s_avg10:-unavailable}" "${s_total:-unavailable}" \
                "${f_avg10:-unavailable}" "${f_total:-unavailable}"
            ;;
        status)
            # Extract RSS-related fields from /proc/self/status
            local vm_rss vm_size vm_hwm rss_anon rss_file rss_shmem
            vm_rss=$(echo "$raw"    | awk '/^VmRSS:/{print $2$3}')
            vm_size=$(echo "$raw"   | awk '/^VmSize:/{print $2$3}')
            vm_hwm=$(echo "$raw"    | awk '/^VmHWM:/{print $2$3}')
            rss_anon=$(echo "$raw"  | awk '/^RssAnon:/{print $2$3}')
            rss_file=$(echo "$raw"  | awk '/^RssFile:/{print $2$3}')
            rss_shmem=$(echo "$raw" | awk '/^RssShmem:/{print $2$3}')
            printf 'VmRSS=%s VmSize=%s VmHWM=%s RssAnon=%s RssFile=%s RssShmem=%s' \
                "${vm_rss:-unavailable}" "${vm_size:-unavailable}" "${vm_hwm:-unavailable}" \
                "${rss_anon:-unavailable}" "${rss_file:-unavailable}" "${rss_shmem:-unavailable}"
            ;;
        statm)
            # /proc/self/statm: size resident shared text lib data dt (all in pages)
            printf '%s' "$raw"
            ;;
        *)
            printf '%s' "$raw"
            ;;
    esac
}

# Special markers for non-numeric values
is_special_value() {
    local val="$1"
    case "$val" in
        max|MAX|unavailable|FAIL) return 0 ;;
        *) return 1 ;;
    esac
}

# Compute percentiles from sorted newline-separated numeric values
compute_percentiles() {
    local data="$1" n total
    [[ -z "$data" ]] && { printf 'median=unavailable p95=unavailable max=unavailable'; return; }
    total=$(echo "$data" | wc -l)
    [[ "$total" -eq 0 ]] && { printf 'median=unavailable p95=unavailable max=unavailable'; return; }
    local mid=$(( (total + 1) / 2 ))
    local p95_idx=$(( (total * 95 + 99) / 100 ))  # ceiling of total * 0.95
    [[ "$p95_idx" -gt "$total" ]] && p95_idx="$total"
    [[ "$p95_idx" -lt 1 ]] && p95_idx=1
    local median p95 max
    median=$(echo "$data" | sed -n "${mid}p")
    p95=$(echo "$data" | sed -n "${p95_idx}p")
    max=$(echo "$data" | tail -1)
    printf 'median=%s p95=%s max=%s' "$median" "$p95" "$max"
}

TOTAL_FAILURES=0
SOURCES_AVAILABLE=0
SOURCES_UNAVAILABLE=0

for src_entry in "${SOURCES[@]}"; do
    IFS='|' read -r label path parser is_multi <<< "$src_entry"
    src_dir="$OUTPUT_DIR/sources/$label"
    mkdir -p "$src_dir"

    printf 'label=%s\n' "$label"     > "$src_dir/meta"
    printf 'path=%s\n' "$path"       >> "$src_dir/meta"
    printf 'parser=%s\n' "$parser"   >> "$src_dir/meta"

    # Check availability
    if [[ ! -e "$path" ]]; then
        printf 'status=unavailable\n'            > "$src_dir/result"
        printf 'reason=path_does_not_exist\n'    >> "$src_dir/result"
        printf 'read_count=%d\n' 0               >> "$src_dir/result"
        printf 'failures=%d\n' 0                 >> "$src_dir/result"
        printf 'value_median=unavailable\n'      >> "$src_dir/result"
        printf 'value_p95=unavailable\n'         >> "$src_dir/result"
        printf 'value_max=unavailable\n'         >> "$src_dir/result"
        printf 'latency_ns_median=unavailable\n' >> "$src_dir/result"
        printf 'latency_ns_p95=unavailable\n'    >> "$src_dir/result"
        printf 'latency_ns_max=unavailable\n'    >> "$src_dir/result"
        printf 'special_value=unavailable\n'     >> "$src_dir/result"
        SOURCES_UNAVAILABLE=$((SOURCES_UNAVAILABLE + 1))
        info "  $label: UNAVAILABLE (path does not exist)"
        continue
    fi
    if [[ ! -r "$path" ]]; then
        printf 'status=unavailable\n'            > "$src_dir/result"
        printf 'reason=path_not_readable\n'      >> "$src_dir/result"
        printf 'read_count=%d\n' 0               >> "$src_dir/result"
        printf 'failures=%d\n' 0                 >> "$src_dir/result"
        printf 'value_median=unavailable\n'      >> "$src_dir/result"
        printf 'value_p95=unavailable\n'         >> "$src_dir/result"
        printf 'value_max=unavailable\n'         >> "$src_dir/result"
        printf 'latency_ns_median=unavailable\n' >> "$src_dir/result"
        printf 'latency_ns_p95=unavailable\n'    >> "$src_dir/result"
        printf 'latency_ns_max=unavailable\n'    >> "$src_dir/result"
        printf 'special_value=unavailable\n'     >> "$src_dir/result"
        SOURCES_UNAVAILABLE=$((SOURCES_UNAVAILABLE + 1))
        info "  $label: UNAVAILABLE (not readable)"
        continue
    fi

    SOURCES_AVAILABLE=$((SOURCES_AVAILABLE + 1))
    info "  $label: probing ($N_READS reads)"

    # Run N reads, collect raw values + latencies
    raw_vals_file="$src_dir/raw_values.txt"
    lat_file="$src_dir/raw_latencies_ns.txt"
    failures=0
    :> "$raw_vals_file"
    :> "$lat_file"

    for _ in $(seq 1 "$N_READS"); do
        start_ns=$(date +%s%N) || { failures=$((failures + 1)); continue; }
        if content=$(cat "$path" 2>/dev/null); then
            end_ns=$(date +%s%N) || { failures=$((failures + 1)); continue; }
            latency=$(( end_ns - start_ns ))
            # Normalise content: collapse newlines to spaces, strip trailing whitespace
            content_flat=$(echo "$content" | tr '\n' ' ' | sed 's/[[:space:]]*$//')
            parsed=$(parse_value "$content_flat" "$parser")
            printf '%s\n' "$parsed" >> "$raw_vals_file"
            printf '%d\n' "$latency" >> "$lat_file"
        else
            end_ns=$(date +%s%N) || true
            latency=$(( end_ns - start_ns ))
            printf 'FAIL\n'   >> "$raw_vals_file"
            printf '%d\n' "$latency" >> "$lat_file"
            failures=$((failures + 1))
        fi
    done

    TOTAL_FAILURES=$((TOTAL_FAILURES + failures))

    # Store the raw content of the first successful read for reference
    if [[ "$failures" -lt "$N_READS" ]]; then
        cat "$path" 2>/dev/null > "$src_dir/raw_sample" || true
    else
        printf 'all reads failed\n' > "$src_dir/raw_sample"
    fi

    # Value analysis
    # Extract numeric values for statistical computation; exclude FAIL and special markers
    numeric_vals=$(grep -vE '^(FAIL|max|MAX|unavailable)$' "$raw_vals_file" | grep -E '^[0-9]+$' || true)
    numeric_count=$(echo "$numeric_vals" | grep -c . 2>/dev/null || true)
    numeric_count=${numeric_count:-0}
    numeric_count=$(printf '%d' "$numeric_count" 2>/dev/null || printf '0')
    sorted_numeric=$(echo "$numeric_vals" | sort -n)

    # Detect special values
    special_value="none"
    first_val=$(head -1 "$raw_vals_file")
    if [[ "$first_val" == "max" || "$first_val" == "MAX" ]]; then
        special_value="max_no_limit"
    fi

    # Detect value stability: all values identical?
    value_stable=0
    if [[ "$failures" -eq 0 && "$numeric_count" -eq "$N_READS" ]]; then
        unique_count=$(sort -u "$raw_vals_file" | wc -l)
        if [[ "$unique_count" -eq 1 ]]; then
            value_stable=1
        fi
    fi

    # Compute value statistics (only on pure-numeric values)
    if [[ "$special_value" == "max_no_limit" ]]; then
        # Don't compute numeric stats on "max" — record the special marker
        v_median="max_no_limit" v_p95="max_no_limit" v_max="max_no_limit"
    elif [[ "$numeric_count" -gt 0 ]]; then
        pct_output=$(compute_percentiles "$sorted_numeric")
        v_median=$(echo "$pct_output" | awk '{print $1}' | cut -d= -f2)
        v_p95=$(echo "$pct_output"   | awk '{print $2}' | cut -d= -f2)
        v_max=$(echo "$pct_output"   | awk '{print $3}' | cut -d= -f2)
        v_median=${v_median:-unavailable}
        v_p95=${v_p95:-unavailable}
        v_max=${v_max:-unavailable}
    else
        v_median="unavailable" v_p95="unavailable" v_max="unavailable"
    fi

    # Latency statistics (always numeric)
    sorted_lat=$(sort -n "$lat_file")
    pct_output=$(compute_percentiles "$sorted_lat")
    l_median=$(echo "$pct_output" | awk '{print $1}' | cut -d= -f2)
    l_p95=$(echo "$pct_output"   | awk '{print $2}' | cut -d= -f2)
    l_max=$(echo "$pct_output"   | awk '{print $3}' | cut -d= -f2)
    l_median=${l_median:-unavailable}
    l_p95=${l_p95:-unavailable}
    l_max=${l_max:-unavailable}

    # Check for duplicate field names in parsed values
    dup_detected=0
    dup_note=""
    if [[ "$parser" == "status" || "$parser" == "psi" ]]; then
        first_parsed=$(head -1 "$raw_vals_file")
        field_names=$(echo "$first_parsed" | grep -oE '[A-Za-z_]+=' | sort)
        unique_field_names=$(echo "$field_names" | sort -u)
        if [[ "$field_names" != "$unique_field_names" ]]; then
            dup_detected=1
            dup_note="DUPLICATE_FIELD_NAMES_in_parsed_output"
        fi
    fi

    # Format error detection: any line in raw_vals that doesn't match expected pattern
    fmt_errors=0
    case "$parser" in
        first_field)
            # Should be numeric, "max", or "FAIL"
            fmt_errors=$(grep -cvE '^(max|MAX|[0-9]+|FAIL)$' "$raw_vals_file" 2>/dev/null || true)
            fmt_errors=${fmt_errors:-0}
            fmt_errors=$(printf '%d' "$fmt_errors" 2>/dev/null || printf '0')
            ;;
        statm)
            # Should be 7 space-separated decimal numbers
            fmt_errors=$(grep -cvE '^[0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+ [0-9]+$' "$raw_vals_file" 2>/dev/null || true)
            fmt_errors=${fmt_errors:-0}
            fmt_errors=$(printf '%d' "$fmt_errors" 2>/dev/null || printf '0')
            ;;
    esac

    # Write result
    {
        printf 'status=available\n'
        printf 'read_count=%d\n' "$N_READS"
        printf 'failures=%d\n' "$failures"
        printf 'value_median=%s\n' "$v_median"
        printf 'value_p95=%s\n' "$v_p95"
        printf 'value_max=%s\n' "$v_max"
        printf 'value_stable=%d\n' "$value_stable"
        printf 'special_value=%s\n' "$special_value"
        printf 'format_errors=%d\n' "${fmt_errors:-0}"
        printf 'duplicate_fields=%d\n' "$dup_detected"
        printf 'duplicate_note=%s\n' "${dup_note:-none}"
        printf 'latency_ns_median=%s\n' "$l_median"
        printf 'latency_ns_p95=%s\n' "$l_p95"
        printf 'latency_ns_max=%s\n' "$l_max"
        printf 'numeric_count=%d\n' "$numeric_count"
    } > "$src_dir/result"

    info "    done: failures=$failures value_median=$v_median latency_median_ns=$l_median"
done

# ── summary ───────────────────────────────────────────────────────────────
info "writing summary"

SUMMARY_FILE="$OUTPUT_DIR/summary"
{
    printf '=== Stage 3A-1A Pressure Input Probe Summary ===\n'
    printf '\n'
    printf 'execution:\n'
    printf '  command: %s\n' "$(printf '%q' "$0"; printf ' N_READS=%s' "$N_READS")"
    printf '  timestamp_utc: %s\n' "$TIMESTAMP"
    printf '  pid: %s\n' "$$"
    printf '\n'
    printf 'environment:\n'
    printf '  head: %s\n' "$HEAD"
    printf '  worktree_dirty: %d\n' "$WORKTREE_DIRTY"
    printf '  kernel: %s\n' "$KERNEL"
    printf '  arch: %s\n' "$(cat "$OUTPUT_DIR/arch")"
    printf '  cgroup_version: %s\n' "$CGROUP_VERSION"
    printf '  cgroup_mem_path: %s\n' "$CGROUP_MEM_PATH"
    printf '  cgroup_path_ambiguous: %d\n' "$CGROUP_PATH_AMBIGUOUS"
    printf '  cgroup_resolve_notes: %s\n' "${CGROUP_RESOLVE_NOTES:-none}"
    printf '\n'
    printf 'sources:\n'
    printf '  total_registered: %d\n' "${#SOURCES[@]}"
    printf '  available: %d\n' "$SOURCES_AVAILABLE"
    printf '  unavailable: %d\n' "$SOURCES_UNAVAILABLE"
    printf '  read_attempts_per_source: %d\n' "$N_READS"
    printf '  total_failures: %d\n' "$TOTAL_FAILURES"
    printf '\n'
    printf 'per_source:\n'

    for src_entry in "${SOURCES[@]}"; do
        IFS='|' read -r label path parser is_multi <<< "$src_entry"
        src_dir="$OUTPUT_DIR/sources/$label"
        result_file="$src_dir/result"
        printf '  %s:\n' "$label"
        printf '    path: %s\n' "$path"
        if [[ -f "$result_file" ]]; then
            while IFS='=' read -r k v; do
                printf '    %s: %s\n' "$k" "$v"
            done < "$result_file"
        fi
    done
} > "$SUMMARY_FILE"

# ── exit code & final report ─────────────────────────────────────────────
rc=0
if [[ "$CGROUP_VERSION" == "none" ]]; then
    warn "no cgroup memory controller detected"
    # Not a hard error: /proc sources may still work
fi
if [[ "$CGROUP_PATH_AMBIGUOUS" -eq 1 ]]; then
    warn "cgroup path ambiguity detected: ${CGROUP_RESOLVE_NOTES}"
fi
if [[ "$SOURCES_AVAILABLE" -eq 0 ]]; then
    die "no data sources available — nothing to probe"
fi
if [[ "$TOTAL_FAILURES" -gt $(( N_READS * SOURCES_AVAILABLE / 2 )) ]]; then
    warn "high failure rate: $TOTAL_FAILURES / $(( N_READS * SOURCES_AVAILABLE ))"
fi

printf '%d\n' "$rc" > "$OUTPUT_DIR/exit_code"

# Create a machine-parseable final status
{
    printf 'exit_code=%d\n' "$rc"
    printf 'cgroup_version=%s\n' "$CGROUP_VERSION"
    printf 'cgroup_path_ambiguous=%d\n' "$CGROUP_PATH_AMBIGUOUS"
    printf 'sources_available=%d\n' "$SOURCES_AVAILABLE"
    printf 'sources_unavailable=%d\n' "$SOURCES_UNAVAILABLE"
    printf 'total_failures=%d\n' "$TOTAL_FAILURES"
} > "$OUTPUT_DIR/final_status"

# Artifact hash
(cd "$OUTPUT_DIR" && find . -type f | sort | xargs sha256sum > artifacts.sha256)

# Output locations
printf 'artifact=%s\n' "$OUTPUT_DIR"
printf 'summary=%s\n' "$SUMMARY_FILE"
printf 'exit_code=%d\n' "$rc"

exit "$rc"
