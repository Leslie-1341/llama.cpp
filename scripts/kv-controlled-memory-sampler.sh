#!/usr/bin/env bash
# Shared process/cgroup sampler for controlled KV workloads.

set -u

kv_controlled_read_first() {
    local file="${1:-}" value=""
    if [[ -n "$file" && -r "$file" ]]; then
        IFS= read -r value < "$file" || true
        printf '%s' "$value"
    else
        printf 'NA'
    fi
}

kv_controlled_descendant_pid() {
    local pid="$1" child
    while [[ -r "/proc/$pid/task/$pid/children" ]]; do
        read -r child _ < "/proc/$pid/task/$pid/children" || true
        [[ -n "${child:-}" ]] || break
        pid="$child"
    done
    printf '%s' "$pid"
}

kv_controlled_starttime_ticks() {
    local pid="$1" stat_text tail fields
    [[ -r "/proc/$pid/stat" ]] || {
        printf 'NA'
        return
    }
    stat_text="$(<"/proc/$pid/stat")"
    tail="${stat_text##*) }"
    read -r -a fields <<< "$tail"
    if [[ "${#fields[@]}" -gt 19 && "${fields[19]}" =~ ^[0-9]+$ ]]; then
        printf '%s' "${fields[19]}"
    else
        printf 'NA'
    fi
}

kv_controlled_sample_process() {
    local wrapper_pid="$1" output="$2" backing_dir="$3" sample_interval_sec="$4" cgroup_current_file="$5"
    local start now pid starttime_ticks vmrss vmhwm cgroup_current logical allocated fd target stat_values blocks

    start="$(date +%s%N)"
    printf 'elapsed_ms\tpid\tstarttime_ticks\tvmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes\n' > "$output"
    while kill -0 "$wrapper_pid" 2>/dev/null; do
        pid="$(kv_controlled_descendant_pid "$wrapper_pid")"
        starttime_ticks="$(kv_controlled_starttime_ticks "$pid")"
        now="$(date +%s%N)"
        vmrss="NA"
        vmhwm="NA"
        if [[ -r "/proc/$pid/status" ]]; then
            vmrss="$(awk '/^VmRSS:/ { print $2 }' "/proc/$pid/status")"
            vmhwm="$(awk '/^VmHWM:/ { print $2 }' "/proc/$pid/status")"
        fi
        cgroup_current="$(kv_controlled_read_first "$cgroup_current_file")"
        logical="NA"
        allocated="NA"
        if [[ -n "$backing_dir" && -d "/proc/$pid/fd" ]]; then
            for fd in /proc/"$pid"/fd/*; do
                target="$(readlink "$fd" 2>/dev/null || true)"
                [[ "$target" == "$backing_dir"/* || "$target" == *"/$(basename "$backing_dir")/"* ]] || continue
                stat_values="$(stat -Lc '%s %b' "$fd" 2>/dev/null || true)"
                if [[ -n "$stat_values" ]]; then
                    read -r logical blocks <<< "$stat_values"
                    allocated=$((blocks * 512))
                    break
                fi
            done
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(( (now - start) / 1000000 ))" "$pid" "${starttime_ticks:-NA}" \
            "${vmrss:-NA}" "${vmhwm:-NA}" "${cgroup_current:-NA}" "$logical" "$allocated" >> "$output"
        sleep "$sample_interval_sec"
    done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    case "${1:-}" in
        --sample-process)
            shift
            [[ "$#" == "5" ]] || {
                printf 'usage: %s --sample-process PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n' "$0" >&2
                exit 2
            }
            kv_controlled_sample_process "$@"
            ;;
        *)
            printf 'usage: %s --sample-process PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n' "$0" >&2
            exit 2
            ;;
    esac
fi
