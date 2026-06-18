#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN="${BIN:-${BUILD_DIR}/bin/llama-kv-idle-telemetry}"
OUT_DIR="${OUT_DIR:-/tmp/llama-kv-paged-idle-telemetry-smoke}"
MODEL="${1:-${MODEL:-}}"

if [[ -z "${MODEL}" ]]; then
    echo "error: provide model path as argv[1] or MODEL=/path/to/model.gguf" >&2
    exit 2
fi

if [[ ! -f "${MODEL}" ]]; then
    echo "error: model not found: ${MODEL}" >&2
    exit 2
fi

for swap_env in LLAMA_KV_PAGED_SWAP LLAMA_KV_PAGED_SWAP_MADVISE LLAMA_KV_PAGED_IDLE_SWAP; do
    if [[ -n "${!swap_env:-}" ]]; then
        echo "error: ${swap_env} must be unset for telemetry-only smoke" >&2
        exit 2
    fi
done

if [[ ! -x "${BIN}" ]]; then
    echo "error: binary not found or not executable: ${BIN}" >&2
    echo "hint: cmake --build ${BUILD_DIR} --target llama-kv-idle-telemetry -j\$(nproc)" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

BASE_OUT="${OUT_DIR}/base.out"
BASE_ERR="${OUT_DIR}/base.err"
TRACE_OUT="${OUT_DIR}/trace.out"
TRACE_ERR="${OUT_DIR}/trace.err"

COMMON_ARGS=(
    -m "${MODEL}"
    -p "Active request: list three colors."
    -n "${N_PREDICT:-32}"
    --ctx-size "${CTX_SIZE:-512}"
    --batch-size "${BATCH_SIZE:-128}"
    --ubatch-size "${UBATCH_SIZE:-128}"
    --seed "${SEED:-1}"
    --temp 0
    --cache-type-k f32
    --cache-type-v f32
    --kv-unified
    --parallel 2
)

BASE_ENV=(
    LLAMA_KV_PAGED=1
    LLAMA_KV_PAGED_SHIFT=1
)

env -u LLAMA_KV_PAGED_SWAP -u LLAMA_KV_PAGED_SWAP_MADVISE -u LLAMA_KV_PAGED_IDLE_SWAP \
    "${BASE_ENV[@]}" "${BIN}" "${COMMON_ARGS[@]}" >"${BASE_OUT}" 2>"${BASE_ERR}"

TRACE_ENV=(
    LLAMA_KV_PAGED=1
    LLAMA_KV_PAGED_SHIFT=1
    LLAMA_KV_PAGED_IDLE_TRACE=1
)

if [[ "${ENABLE_PAGED_TRACE:-1}" == "1" ]]; then
    TRACE_ENV+=(LLAMA_KV_PAGED_TRACE=1)
fi

env -u LLAMA_KV_PAGED_SWAP -u LLAMA_KV_PAGED_SWAP_MADVISE -u LLAMA_KV_PAGED_IDLE_SWAP \
    "${TRACE_ENV[@]}" "${BIN}" "${COMMON_ARGS[@]}" >"${TRACE_OUT}" 2>"${TRACE_ERR}"

if cmp -s "${BASE_OUT}" "${TRACE_OUT}"; then
    base_vs_trace_equal=0
else
    base_vs_trace_equal=1
fi

echo "out_dir=${OUT_DIR}"
echo "base_vs_trace_equal=${base_vs_trace_equal}"
sha256sum "${BASE_OUT}" "${TRACE_OUT}"

extract_field_from_line() {
    local line="$1"
    local name="$2"
    local matches
    matches="$(printf '%s\n' "${line}" | grep -Eo "${name}=[0-9]+" || true)"
    printf '%s\n' "${matches}" | tail -n 1 | cut -d= -f2
}

trace_lines="$(grep -c 'KV_PAGED_IDLE_TRACE' "${TRACE_ERR}" || true)"
echo "idle_trace_lines=${trace_lines}"

final_stats_line="$(grep 'KV paged metadata stats' "${TRACE_ERR}" | tail -n 1 || true)"
idle_trace_line="$(grep 'KV_PAGED_IDLE_TRACE' "${TRACE_ERR}" | tail -n 1 || true)"
paged_trace_line="$(grep 'KV_PAGED_TRACE' "${TRACE_ERR}" | tail -n 1 || true)"

if [[ -n "${final_stats_line}" ]]; then
    final_stats_present=1
else
    final_stats_present=0
fi

final_field() {
    extract_field_from_line "${final_stats_line}" "$1"
}

idle_trace_field() {
    extract_field_from_line "${idle_trace_line}" "$1"
}

paged_trace_field() {
    extract_field_from_line "${paged_trace_line}" "$1"
}

idle_value() {
    local final_name="$1"
    local trace_name="$2"
    local value
    value="$(final_field "${final_name}")"
    if [[ -z "${value}" ]]; then
        value="$(idle_trace_field "${trace_name}")"
    fi
    printf '%s\n' "${value}"
}

fail=0

declare -a idle_fields=(
    "paged_idle_cold_candidates:cold_candidates"
    "paged_idle_read_window_blocks:read_window_blocks"
    "paged_idle_cold_in_read_window:cold_in_read_window"
    "paged_idle_cold_not_in_read_window:cold_not_in_read_window"
    "paged_idle_skip_mixed_active:skip_mixed_active"
    "paged_idle_safe_swap_candidates:safe_swap_candidates"
)

for mapping in "${idle_fields[@]}"; do
    final_name="${mapping%%:*}"
    trace_name="${mapping##*:}"
    value="$(idle_value "${final_name}" "${trace_name}")"
    if [[ -z "${value}" ]]; then
        echo "${final_name}=MISSING"
        fail=1
    else
        echo "${final_name}=${value}"
    fi
done

safe="$(idle_value paged_idle_safe_swap_candidates safe_swap_candidates)"
if [[ -n "${safe}" && "${safe}" -gt 0 ]]; then
    echo "safe_swap_candidate_gate=>0"
else
    echo "safe_swap_candidate_gate=0"
fi

echo "final_stats_present=${final_stats_present}"

if [[ "${final_stats_present}" == "1" ]]; then
    for zero_field in \
        paged_swap_enabled \
        paged_swap_out_calls \
        paged_swap_in_calls \
        paged_swap_madvise_calls \
        paged_swap_backend_failures \
        row_idx_fail \
        logical_to_physical_fail; do
        value="$(final_field "${zero_field}")"
        if [[ -z "${value}" ]]; then
            echo "${zero_field}=MISSING"
            fail=1
        else
            echo "${zero_field}=${value}"
            if [[ "${value}" != "0" ]]; then
                fail=1
            fi
        fi
    done
else
    echo "paged_swap_enabled=not_available_without_final_stats"
    echo "paged_swap_out_calls=not_available_without_final_stats"
    echo "paged_swap_in_calls=not_available_without_final_stats"
    echo "paged_swap_madvise_calls=not_available_without_final_stats"
    echo "paged_swap_backend_failures=not_available_without_final_stats"
    echo "row_idx_fail=not_available_without_final_stats"
    echo "logical_to_physical_fail=not_available_without_final_stats"

    swapped_blocks="$(paged_trace_field swapped_blocks)"
    if [[ -z "${swapped_blocks}" ]]; then
        echo "swapped_blocks=MISSING"
        fail=1
    else
        echo "swapped_blocks=${swapped_blocks}"
        if [[ "${swapped_blocks}" != "0" ]]; then
            fail=1
        fi
    fi
fi

if [[ "${base_vs_trace_equal}" != "0" ]]; then
    fail=1
fi

if [[ "${trace_lines}" == "0" ]]; then
    fail=1
fi

exit "${fail}"
