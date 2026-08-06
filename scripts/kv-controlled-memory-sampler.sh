#!/usr/bin/env bash
# Shared process/cgroup sampler for controlled KV workloads.

set -u

KV_CONTROLLED_SAMPLE_HEADER=$'elapsed_ms\tpid\tstarttime_ticks\tvmrss_kb\tvmhwm_kb\tcgroup_memory_current_bytes\tbacking_logical_size\tbacking_allocated_bytes'
KV_CONTROLLED_SAMPLE_NA='NA'
KV_CONTROLLED_BIND_TIMEOUT_SEC="${KV_CONTROLLED_BIND_TIMEOUT_SEC:-5}"
KV_CONTROLLED_STOP_REQUESTED=0

kv_controlled_read_first() {
    local file="${1:-}" value=""
    if [[ -n "$file" && -r "$file" ]]; then
        IFS= read -r value < "$file" || true
        if [[ -n "$value" ]]; then
            printf '%s' "$value"
            return 0
        fi
    fi
    printf '%s' "$KV_CONTROLLED_SAMPLE_NA"
}

kv_controlled_read_proc_stat() {
    local pid="${1:-}" stat_text tail
    local -a fields=()
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    [[ -r "/proc/$pid/stat" ]] || return 1
    if ! stat_text="$(<"/proc/$pid/stat")"; then
        return 1
    fi
    tail="${stat_text##*) }"
    read -r -a fields <<< "$tail"
    if [[ "${#fields[@]}" -le 19 || ! "${fields[0]}" =~ ^[A-Za-z]$ ||
          ! "${fields[19]}" =~ ^[0-9]+$ ]]; then
        return 1
    fi
    KV_CONTROLLED_PROC_STATE="${fields[0]}"
    KV_CONTROLLED_PROC_STARTTIME="${fields[19]}"
    return 0
}

kv_controlled_descendant_pid() {
    local pid="${1:-}" child=""
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    while [[ -r "/proc/$pid/task/$pid/children" ]]; do
        child=""
        read -r child _ < "/proc/$pid/task/$pid/children" || true
        [[ "$child" =~ ^[1-9][0-9]*$ ]] || break
        pid="$child"
    done
    printf '%s' "$pid"
}

kv_controlled_validate_sample_arguments() {
    [[ "$#" == "5" ]] || {
        printf 'sampler: expected PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n' >&2
        return 2
    }
    local pid="$1" output="$2" backing_dir="$3" interval="$4" cgroup_file="$5"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || {
        printf 'sampler: invalid PID: %s\n' "$pid" >&2
        return 2
    }
    [[ -n "$output" ]] || {
        printf 'sampler: output path is empty\n' >&2
        return 2
    }
    [[ "$interval" =~ ^[0-9]+([.][0-9]+)?$ && "$interval" == *[1-9]* ]] || {
        printf 'sampler: interval must be a positive decimal: %s\n' "$interval" >&2
        return 2
    }
    [[ -z "$backing_dir" || -d "$backing_dir" ]] || {
        printf 'sampler: backing directory is not a directory: %s\n' "$backing_dir" >&2
        return 2
    }
    if [[ -n "$cgroup_file" && ( "$cgroup_file" == *$'\n'* || "$cgroup_file" == *$'\t'* ) ]]; then
        printf 'sampler: cgroup path contains an invalid tab/newline: %s\n' "$cgroup_file" >&2
        return 2
    fi
    return 0
}

kv_controlled_bind_direct_pid() {
    local pid="${1:-}"
    if ! kv_controlled_read_proc_stat "$pid"; then
        printf 'sampler: cannot bind direct process identity (pid=%s)\n' "$pid" >&2
        return 7
    fi
    KV_CONTROLLED_BOUND_PID="$pid"
    KV_CONTROLLED_BOUND_STARTTIME="$KV_CONTROLLED_PROC_STARTTIME"
    return 0
}

kv_controlled_bind_wrapper_child() {
    local wrapper_pid="${1:-}" candidate="" deadline_ns now_ns
    [[ "$wrapper_pid" =~ ^[1-9][0-9]*$ ]] || {
        printf 'sampler: invalid wrapper PID: %s\n' "$wrapper_pid" >&2
        return 2
    }
    if ! kv_controlled_read_proc_stat "$wrapper_pid"; then
        printf 'sampler: cannot bind wrapper identity (pid=%s)\n' "$wrapper_pid" >&2
        return 7
    fi
    if [[ "$KV_CONTROLLED_PROC_STATE" == "Z" || "$KV_CONTROLLED_PROC_STATE" == "X" ]]; then
        printf 'sampler: wrapper is already terminated (pid=%s state=%s)\n' \
            "$wrapper_pid" "$KV_CONTROLLED_PROC_STATE" >&2
        return 7
    fi
    if ! now_ns="$(date +%s%N)" || ! [[ "$now_ns" =~ ^[0-9]+$ ]]; then
        printf 'sampler: cannot read startup clock\n' >&2
        return 12
    fi
    deadline_ns=$((now_ns + KV_CONTROLLED_BIND_TIMEOUT_SEC * 1000000000))
    while :; do
        if ! kv_controlled_read_proc_stat "$wrapper_pid"; then
            printf 'sampler: wrapper disappeared before child binding (pid=%s)\n' "$wrapper_pid" >&2
            return 7
        fi
        if [[ "$KV_CONTROLLED_PROC_STATE" == "Z" || "$KV_CONTROLLED_PROC_STATE" == "X" ]]; then
            printf 'sampler: wrapper terminated before child binding (pid=%s)\n' "$wrapper_pid" >&2
            return 7
        fi
        candidate="$(kv_controlled_descendant_pid "$wrapper_pid" 2>/dev/null || true)"
        if [[ "$candidate" =~ ^[1-9][0-9]*$ && "$candidate" != "$wrapper_pid" ]] &&
           kv_controlled_read_proc_stat "$candidate"; then
            KV_CONTROLLED_BOUND_PID="$candidate"
            KV_CONTROLLED_BOUND_STARTTIME="$KV_CONTROLLED_PROC_STARTTIME"
            return 0
        fi
        if ! now_ns="$(date +%s%N)" || ! [[ "$now_ns" =~ ^[0-9]+$ ]]; then
            printf 'sampler: cannot read startup clock\n' >&2
            return 12
        fi
        if (( now_ns >= deadline_ns )); then
            printf 'sampler: wrapper child did not become bindable (wrapper_pid=%s)\n' "$wrapper_pid" >&2
            return 7
        fi
        if ! sleep 0.01; then
            printf 'sampler: startup child wait failed\n' >&2
            return 12
        fi
    done
}

kv_controlled_restore_signal_traps() {
    local term_trap="$1" int_trap="$2"
    if [[ -n "$term_trap" ]]; then
        eval "$term_trap"
    else
        trap - TERM
    fi
    if [[ -n "$int_trap" ]]; then
        eval "$int_trap"
    else
        trap - INT
    fi
}

kv_controlled_recheck_after_read_failure() {
    local bound_pid="$1" bound_starttime="$2"
    if ! kv_controlled_read_proc_stat "$bound_pid"; then
        KV_CONTROLLED_READ_FAILURE_STATE='normal_exit'
    elif [[ "$KV_CONTROLLED_PROC_STARTTIME" != "$bound_starttime" ]]; then
        KV_CONTROLLED_READ_FAILURE_STATE='starttime_changed'
    elif [[ "$KV_CONTROLLED_PROC_STATE" == "Z" || "$KV_CONTROLLED_PROC_STATE" == "X" ]]; then
        KV_CONTROLLED_READ_FAILURE_STATE='normal_exit'
    else
        KV_CONTROLLED_READ_FAILURE_STATE='live_read_failure'
    fi
}

kv_controlled_sample_bound_process() {
    local bound_pid="${1:-}" bound_starttime="${2:-}" output="${3:-}" backing_dir="${4:-}"
    local sample_interval_sec="${5:-}" cgroup_current_file="${6:-}"
    local start_ns now_ns elapsed_ms state starttime_ticks status_values vmrss vmhwm
    local cgroup_current logical allocated fd target stat_values blocks extra
    local rc=0 term_trap int_trap

    kv_controlled_validate_sample_arguments "$bound_pid" "$output" "$backing_dir" \
        "$sample_interval_sec" "$cgroup_current_file" || return $?
    [[ "$bound_starttime" =~ ^[0-9]+$ ]] || {
        printf 'sampler: invalid bound starttime: %s\n' "$bound_starttime" >&2
        return 2
    }

    term_trap="$(trap -p TERM)"
    int_trap="$(trap -p INT)"
    KV_CONTROLLED_STOP_REQUESTED=0
    trap 'KV_CONTROLLED_STOP_REQUESTED=1' TERM INT

    if ! start_ns="$(date +%s%N)" || ! [[ "$start_ns" =~ ^[0-9]+$ ]]; then
        printf 'sampler: cannot read sampling clock\n' >&2
        rc=12
    elif ! printf '%s\n' "$KV_CONTROLLED_SAMPLE_HEADER" > "$output"; then
        printf 'sampler: cannot write output header: %s\n' "$output" >&2
        rc=10
    else
        while (( KV_CONTROLLED_STOP_REQUESTED == 0 )); do
            if ! kv_controlled_read_proc_stat "$bound_pid"; then
                rc=0
                break
            fi
            state="$KV_CONTROLLED_PROC_STATE"
            starttime_ticks="$KV_CONTROLLED_PROC_STARTTIME"
            if [[ "$starttime_ticks" != "$bound_starttime" ]]; then
                printf 'sampler: bound process starttime changed (pid=%s expected=%s actual=%s)\n' \
                    "$bound_pid" "$bound_starttime" "$starttime_ticks" >&2
                rc=9
                break
            fi
            if [[ "$state" == "Z" || "$state" == "X" ]]; then
                rc=0
                break
            fi
            if ! now_ns="$(date +%s%N)" || ! [[ "$now_ns" =~ ^[0-9]+$ ]]; then
                printf 'sampler: cannot read sampling clock\n' >&2
                rc=12
                break
            fi

            status_values=""
            if ! status_values="$(awk '
                /^VmRSS:/ { rss = $2 }
                /^VmHWM:/ { hwm = $2 }
                END {
                    if (rss ~ /^[0-9]+$/ && hwm ~ /^[0-9]+$/) {
                        print rss, hwm
                    } else {
                        exit 1
                    }
                }
            ' "/proc/$bound_pid/status" 2>/dev/null)"; then
                if (( KV_CONTROLLED_STOP_REQUESTED != 0 )); then
                    rc=0
                    break
                fi
                kv_controlled_recheck_after_read_failure "$bound_pid" "$bound_starttime"
                case "$KV_CONTROLLED_READ_FAILURE_STATE" in
                    normal_exit) rc=0 ;;
                    starttime_changed)
                        printf 'sampler: bound process starttime changed during RSS read (pid=%s)\n' \
                            "$bound_pid" >&2
                        rc=9
                        ;;
                    live_read_failure)
                        printf 'sampler: live process RSS read failed (pid=%s)\n' "$bound_pid" >&2
                        rc=8
                        ;;
                esac
                break
            fi
            read -r vmrss vmhwm extra <<< "$status_values"
            if [[ -n "${extra:-}" || ! "$vmrss" =~ ^[0-9]+$ || ! "$vmhwm" =~ ^[0-9]+$ ]]; then
                kv_controlled_recheck_after_read_failure "$bound_pid" "$bound_starttime"
                case "$KV_CONTROLLED_READ_FAILURE_STATE" in
                    normal_exit) rc=0 ;;
                    starttime_changed)
                        printf 'sampler: bound process starttime changed during RSS read (pid=%s)\n' \
                            "$bound_pid" >&2
                        rc=9
                        ;;
                    live_read_failure)
                        printf 'sampler: live process RSS read was malformed (pid=%s)\n' "$bound_pid" >&2
                        rc=8
                        ;;
                esac
                break
            fi

            cgroup_current="$(kv_controlled_read_first "$cgroup_current_file")"
            logical="$KV_CONTROLLED_SAMPLE_NA"
            allocated="$KV_CONTROLLED_SAMPLE_NA"
            if [[ -n "$backing_dir" && -d "/proc/$bound_pid/fd" ]]; then
                for fd in /proc/"$bound_pid"/fd/*; do
                    target="$(readlink "$fd" 2>/dev/null || true)"
                    [[ "$target" == "$backing_dir"/* || "$target" == *"/$(basename "$backing_dir")/"* ]] || continue
                    stat_values="$(stat -Lc '%s %b' "$fd" 2>/dev/null || true)"
                    if [[ "$stat_values" =~ ^[0-9]+[[:space:]][0-9]+$ ]]; then
                        read -r logical blocks <<< "$stat_values"
                        allocated=$((blocks * 512))
                        break
                    fi
                done
            fi
            if [[ ! "$cgroup_current" =~ ^[0-9]+$ && "$cgroup_current" != "$KV_CONTROLLED_SAMPLE_NA" ]] ||
               [[ ! "$logical" =~ ^[0-9]+$ && "$logical" != "$KV_CONTROLLED_SAMPLE_NA" ]] ||
               [[ ! "$allocated" =~ ^[0-9]+$ && "$allocated" != "$KV_CONTROLLED_SAMPLE_NA" ]]; then
                printf 'sampler: optional sample field is malformed\n' >&2
                rc=13
                break
            fi
            if ! elapsed_ms=$(( (now_ns - start_ns) / 1000000 )); then
                printf 'sampler: elapsed time calculation failed\n' >&2
                rc=12
                break
            fi
            if ! printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$elapsed_ms" "$bound_pid" "$starttime_ticks" "$vmrss" "$vmhwm" \
                "$cgroup_current" "$logical" "$allocated" >> "$output"; then
                printf 'sampler: cannot write sample row: %s\n' "$output" >&2
                rc=11
                break
            fi
            if (( KV_CONTROLLED_STOP_REQUESTED != 0 )); then
                rc=0
                break
            fi
            if ! sleep "$sample_interval_sec"; then
                if (( KV_CONTROLLED_STOP_REQUESTED != 0 )); then
                    rc=0
                else
                    printf 'sampler: sleep failed\n' >&2
                    rc=12
                fi
                break
            fi
        done
    fi

    kv_controlled_restore_signal_traps "$term_trap" "$int_trap"
    return "$rc"
}

kv_controlled_sample_process() {
    local pid="${1:-}" output="${2:-}" backing_dir="${3:-}" sample_interval_sec="${4:-}" cgroup_current_file="${5:-}"
    kv_controlled_validate_sample_arguments "$pid" "$output" "$backing_dir" \
        "$sample_interval_sec" "$cgroup_current_file" || return $?
    local rc
    kv_controlled_bind_direct_pid "$pid"
    rc=$?
    if (( rc != 0 )); then
        return "$rc"
    fi
    kv_controlled_sample_bound_process "$KV_CONTROLLED_BOUND_PID" \
        "$KV_CONTROLLED_BOUND_STARTTIME" "$output" "$backing_dir" "$sample_interval_sec" "$cgroup_current_file"
}

kv_controlled_sample_wrapper() {
    local wrapper_pid="${1:-}" output="${2:-}" backing_dir="${3:-}" sample_interval_sec="${4:-}" cgroup_current_file="${5:-}"
    kv_controlled_validate_sample_arguments "$wrapper_pid" "$output" "$backing_dir" \
        "$sample_interval_sec" "$cgroup_current_file" || return $?
    local rc
    kv_controlled_bind_wrapper_child "$wrapper_pid"
    rc=$?
    if (( rc != 0 )); then
        return "$rc"
    fi
    kv_controlled_sample_bound_process "$KV_CONTROLLED_BOUND_PID" \
        "$KV_CONTROLLED_BOUND_STARTTIME" "$output" "$backing_dir" "$sample_interval_sec" "$cgroup_current_file"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    rc=2
    case "${1:-}" in
        --sample-process)
            shift
            if [[ "$#" == "5" ]]; then
                kv_controlled_sample_process "$@"
                rc=$?
            else
                printf 'usage: %s --sample-process PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n' "$0" >&2
            fi
            ;;
        --sample-wrapper)
            shift
            if [[ "$#" == "5" ]]; then
                kv_controlled_sample_wrapper "$@"
                rc=$?
            else
                printf 'usage: %s --sample-wrapper WRAPPER_PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n' "$0" >&2
            fi
            ;;
        *)
            printf 'usage: %s --sample-process PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n' "$0" >&2
            printf '       %s --sample-wrapper WRAPPER_PID OUTPUT BACKING_DIR INTERVAL_SEC CGROUP_CURRENT_FILE\n' "$0" >&2
            ;;
    esac
    exit "$rc"
fi
