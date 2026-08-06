#!/usr/bin/env bash
# Shared process/cgroup sampler for controlled KV workloads.
# Exit race guard: binds target PID + starttime at startup; only writes complete
# rows with full numeric identity; on any read failure re-checks the target
# process — exits cleanly if gone, errors loudly if alive.

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

kv_controlled_target_alive() {
    local wrapper_pid="$1" bound_starttime="$2"
    local pid starttime
    kill -0 "$wrapper_pid" 2>/dev/null || return 1
    pid="$(kv_controlled_descendant_pid "$wrapper_pid")"
    starttime="$(kv_controlled_starttime_ticks "$pid")"
    if [[ "$pid" != "$wrapper_pid" && "$starttime" =~ ^[0-9]+$ && "$starttime" == "$bound_starttime" ]]; then
        return 0
    fi
    return 1
}

kv_controlled_sample_process() {
    local wrapper_pid="$1" output="$2" backing_dir="$3" sample_interval_sec="$4" cgroup_current_file="$5"
    local start now pid starttime_ticks vmrss vmhwm cgroup_current logical allocated fd target stat_values blocks

    # --- bind identity at startup ---
    local bound_pid bound_starttime
    bound_pid="$(kv_controlled_descendant_pid "$wrapper_pid")"
    bound_starttime="$(kv_controlled_starttime_ticks "$bound_pid")"
    if [[ -z "$bound_pid" || "$bound_pid" == "NA" || "$bound_starttime" == "NA" ]]; then
        printf 'sampler: cannot bind target process identity (pid=%s starttime=%s)\n' \
            "$bound_pid" "$bound_starttime" >&2
        exit 7
    fi

    start="$(date +%s%N)"
    printf 'elapsed_ms\tpid\tstarttime_ticks\tvmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes\n' > "$output"
    while kill -0 "$wrapper_pid" 2>/dev/null; do
        pid="$(kv_controlled_descendant_pid "$wrapper_pid")"
        starttime_ticks="$(kv_controlled_starttime_ticks "$pid")"
        now="$(date +%s%N)"

        # Read VmRSS / VmHWM from /proc/PID/status
        vmrss="NA"
        vmhwm="NA"
        local status_read_ok=true
        if [[ -r "/proc/$pid/status" ]]; then
            vmrss="$(awk '/^VmRSS:/ { print $2 }' "/proc/$pid/status" 2>/dev/null)"
            vmhwm="$(awk '/^VmHWM:/ { print $2 }' "/proc/$pid/status" 2>/dev/null)"
            # Re-readability check: if awk produced empty output, the file may have
            # vanished between -r and open — treat as unreadable
            if [[ -z "$vmrss" ]]; then
                vmrss="NA"
                vmhwm="NA"
                status_read_ok=false
            fi
        else
            status_read_ok=false
        fi

        # Validate all numeric identity fields before writing the row
        local identity_valid=true
        if [[ ! "$pid" =~ ^[0-9]+$ || ! "$starttime_ticks" =~ ^[0-9]+$ ]]; then
            identity_valid=false
        fi
        if [[ "$pid" != "$bound_pid" || "$starttime_ticks" != "$bound_starttime" ]]; then
            identity_valid=false
        fi
        if [[ ! "$vmrss" =~ ^[0-9]+$ || ! "$vmhwm" =~ ^[0-9]+$ ]]; then
            identity_valid=false
        fi

        if ! $identity_valid; then
            # Re-check whether the target process has exited
            if ! kill -0 "$wrapper_pid" 2>/dev/null; then
                # Target is gone — exit silently, do NOT write a partial row
                exit 0
            fi
            # Target is still alive but we failed to read — fatal error
            printf 'sampler: failed to read process identity/memory for pid=%s starttime=%s vmrss=%s vmhwm=%s\n' \
                "$pid" "$starttime_ticks" "$vmrss" "$vmhwm" >&2
            exit 8
        fi

        # Read cgroup and backing (non-fatal: NA sentinel is valid)
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
            "$(( (now - start) / 1000000 ))" "$pid" "${starttime_ticks}" \
            "${vmrss}" "${vmhwm}" "${cgroup_current}" "$logical" "$allocated" >> "$output"
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
            printf "Usage: %s --sample-process PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n" "$0" >&2
            exit 2
            ;;
    esac
fi