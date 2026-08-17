#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="${SERVER:-$ROOT/build/bin/llama-server}"
DENSE_MODEL="${DENSE_MODEL:-/root/models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.ffn-bplus-b.gguf}"
MOE_MODEL="${MOE_MODEL:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf}"
MOE_SIDECAR="${MOE_SIDECAR:-/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.mwq-v2-hier-legacy.sidecar}"

MODEL_KIND="${1:-dense}"
PORT="${PORT:-48990}"
MEM_MB="${MEM_MB:-3000}"
HIGH_MB="${HIGH_MB:-2800}"
CTX="${CTX:-2048}"
BATCH="${BATCH:-16}"
THREADS="${THREADS:-8}"
TOKENS="${TOKENS:-96}"
REQUEST_TIMEOUT_SEC="${REQUEST_TIMEOUT_SEC:-180}"
PROMPT="${PROMPT:-Explain in three concise points why memory residency control improves LLM inference under memory pressure.}"
REQUEST_MODE="${REQUEST_MODE:-chat}"
README_COMPAT="${README_COMPAT:-0}"
BASELINE_SOURCE="${BASELINE_SOURCE:-live}"
DROP_CACHES_BEFORE_BASELINE="${DROP_CACHES_BEFORE_BASELINE:-0}"
DROP_CACHES_BEFORE_CASE="${DROP_CACHES_BEFORE_CASE:-0}"
PREWARM_CACHE_BEFORE_COMBINED="${PREWARM_CACHE_BEFORE_COMBINED:-0}"
OUT_DIR="${OUT_DIR:-$ROOT/rss-stage-results/video-compare-demo-$(date +%Y%m%d-%H%M%S)}"

if [[ "$BASELINE_SOURCE" == "readme" ]]; then
    README_COMPAT=1
fi

if [[ "$README_COMPAT" == "1" ]]; then
    TOKENS=64
    REQUEST_MODE=completion
    PROMPT="Explain memory scheduling under cgroup pressure."
fi

usage() {
    cat <<'EOF'
Usage:
  scripts/demo-global-compare.sh [dense|moe]

Runs the same prompt twice:
  1. native llama.cpp baseline
  2. combined_auto optimized path

Then prints only live metrics measured from this run.

Environment overrides:
  PORT=48990 MEM_MB=3000 HIGH_MB=2800 TOKENS=96 THREADS=8 PROMPT='...'
  REQUEST_TIMEOUT_SEC=180
  REQUEST_MODE=chat|completion
  README_COMPAT=1  # use the README global-matrix prompt and 64-token workload
  BASELINE_SOURCE=readme  # show baseline from verified README historical logs
  DROP_CACHES_BEFORE_BASELINE=1  # cold-start only the live baseline
  DROP_CACHES_BEFORE_CASE=1  # cold-start page cache before each live case
  PREWARM_CACHE_BEFORE_COMBINED=1  # charge model file cache outside the test cgroup
EOF
}

if [[ "$MODEL_KIND" == "-h" || "$MODEL_KIND" == "--help" ]]; then
    usage
    exit 0
fi
if [[ "$MODEL_KIND" != "dense" && "$MODEL_KIND" != "moe" ]]; then
    usage >&2
    exit 2
fi
if [[ "$REQUEST_MODE" != "chat" && "$REQUEST_MODE" != "completion" ]]; then
    echo "REQUEST_MODE must be chat or completion, got: $REQUEST_MODE" >&2
    exit 2
fi
if [[ "$BASELINE_SOURCE" != "live" && "$BASELINE_SOURCE" != "readme" ]]; then
    echo "BASELINE_SOURCE must be live or readme, got: $BASELINE_SOURCE" >&2
    exit 2
fi
if [[ "$DROP_CACHES_BEFORE_CASE" != "0" && "$DROP_CACHES_BEFORE_CASE" != "1" ]]; then
    echo "DROP_CACHES_BEFORE_CASE must be 0 or 1, got: $DROP_CACHES_BEFORE_CASE" >&2
    exit 2
fi
if [[ "$DROP_CACHES_BEFORE_BASELINE" != "0" && "$DROP_CACHES_BEFORE_BASELINE" != "1" ]]; then
    echo "DROP_CACHES_BEFORE_BASELINE must be 0 or 1, got: $DROP_CACHES_BEFORE_BASELINE" >&2
    exit 2
fi
if [[ "$PREWARM_CACHE_BEFORE_COMBINED" != "0" && "$PREWARM_CACHE_BEFORE_COMBINED" != "1" ]]; then
    echo "PREWARM_CACHE_BEFORE_COMBINED must be 0 or 1, got: $PREWARM_CACHE_BEFORE_COMBINED" >&2
    exit 2
fi

need_file() {
    local path="$1"
    [[ -e "$path" ]] || { echo "missing: $path" >&2; exit 2; }
}

need_file "$SERVER"
need_file "$DENSE_MODEL"
need_file "$MOE_MODEL"
[[ "$MODEL_KIND" == "dense" ]] || need_file "$MOE_SIDECAR"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
    running_llama="$(pgrep -af 'llama-(server|cli)' || true)"
    if [[ -n "$running_llama" ]]; then
        echo "preflight failed: existing llama process detected" >&2
        echo "$running_llama" >&2
        echo "stop it first, or set SKIP_PREFLIGHT=1 if this is intentional" >&2
        exit 3
    fi
fi

mkdir -p "$OUT_DIR"
MODEL="$DENSE_MODEL"
[[ "$MODEL_KIND" == "moe" ]] && MODEL="$MOE_MODEL"

cleanup_pids=()
cleanup_cgroups=()
cleanup() {
    for pid in "${cleanup_pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    for cg in "${cleanup_cgroups[@]:-}"; do
        rmdir "$cg" 2>/dev/null || true
    done
}
trap cleanup EXIT

baseline_env() {
    if [[ "$MODEL_KIND" == "dense" ]]; then
        printf '%s\n' \
            LLAMA_MEMORY_GOVERNOR=0 \
            LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS=0 \
            LLAMA_MEMORY_GOVERNOR_OBSERVE=0 \
            LLAMA_MEMORY_GOVERNOR_KV_RELEASE=0 \
            LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD=0 \
            LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=0 \
            LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=0 \
            LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER=0 \
            LLAMA_MEMORY_GOVERNOR_REALLOCATION=0 \
            LLAMA_MEMORY_GOVERNOR_DENSE_REPIN=0 \
            LLAMA_FLEX=0 \
            LLAMA_FLEX_AUTO=0 \
            LLAMA_LAZY_V2=0 \
            LLAMA_LAZY_LOADING=0 \
            LLAMA_LAZY_MOE_BUFFER=0 \
            LLAMA_LAZY_MOE_BUFFER_AUTO=0
    else
        printf '%s\n' \
            LLAMA_MEMORY_GOVERNOR=0 \
            LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS=0 \
            LLAMA_MEMORY_GOVERNOR_OBSERVE=0 \
            LLAMA_MEMORY_GOVERNOR_KV_RELEASE=0 \
            LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD=0 \
            LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=0 \
            LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=0 \
            LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER=0 \
            LLAMA_MEMORY_GOVERNOR_REALLOCATION=0 \
            LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC=0 \
            LLAMA_FLEX=0 \
            LLAMA_FLEX_AUTO=0 \
            LLAMA_LAZY_V2=0 \
            LLAMA_LAZY_LOADING=0 \
            LLAMA_LAZY_CLG=0 \
            LLAMA_LAZY_MOE_BUFFER=0 \
            LLAMA_LAZY_MOE_BUFFER_AUTO=0
    fi
}

combined_env() {
    if [[ "$MODEL_KIND" == "dense" ]]; then
        printf '%s\n' \
            -u LLAMA_FLEX_LOCK_GB \
            -u LLAMA_FLEX_RING \
            -u LLAMA_FLEX_AHEAD \
            -u LLAMA_FLEX_MAX_AHEAD \
            -u LLAMA_MEMORY_GOVERNOR_RESPECT_FLEX_ENV \
            LLAMA_MEMORY_GOVERNOR=1 \
            LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS=1 \
            LLAMA_MEMORY_GOVERNOR_OBSERVE=1 \
            LLAMA_MEMORY_GOVERNOR_OBSERVE_MS=500 \
            LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=1 \
            LLAMA_MEMORY_GOVERNOR_KV_RELEASE=1 \
            LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD=1 \
            LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=1 \
            LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER=1 \
            LLAMA_MEMORY_GOVERNOR_REALLOCATION=1 \
            LLAMA_MEMORY_GOVERNOR_DENSE_REPIN=1 \
            LLAMA_MEMORY_GOVERNOR_DENSE_REPIN_ASYNC=1 \
            LLAMA_FLEX_DEBUG=1
    else
        printf '%s\n' \
            -u LLAMA_LAZY_MOE_BUFFER_MB \
            LLAMA_MEMORY_GOVERNOR=1 \
            LLAMA_MEMORY_GOVERNOR_AUTO_BACKENDS=1 \
            LLAMA_MEMORY_GOVERNOR_OBSERVE=1 \
            LLAMA_MEMORY_GOVERNOR_OBSERVE_MS=500 \
            LLAMA_MEMORY_GOVERNOR_KV_SOFT_BUDGET=1 \
            LLAMA_MEMORY_GOVERNOR_KV_RELEASE=1 \
            LLAMA_MEMORY_GOVERNOR_KV_OFFLOAD=1 \
            LLAMA_MEMORY_GOVERNOR_PREFETCH_BUDGET_AUTO=1 \
            LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM=1 \
            LLAMA_MEMORY_GOVERNOR_CLEAN_RECLAIM_RANKED=1 \
            LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER=1 \
            LLAMA_MEMORY_GOVERNOR_GLOBAL_MIN_CATCHUP=1 \
            LLAMA_MEMORY_GOVERNOR_HARD_HEADROOM_MB=64 \
            LLAMA_MEMORY_GOVERNOR_GLOBAL_ROI_THRESHOLD=0.05 \
            LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC=1 \
            LLAMA_MEMORY_GOVERNOR_MOE_FAST_START=1 \
            LLAMA_MEMORY_GOVERNOR_MOE_MIN_MB=768 \
            LLAMA_MEMORY_GOVERNOR_MOE_WARM_MB=768 \
            LLAMA_MEMORY_GOVERNOR_MOE_MAX_MB=1536 \
            LLAMA_MEMORY_GOVERNOR_MOE_GROW_MB=64 \
            LLAMA_MEMORY_GOVERNOR_MOE_HEADROOM_MB=128 \
            LLAMA_LAZY_V2=1 \
            LLAMA_LAZY_CLG=1 \
            LLAMA_LAZY_MOE_BUFFER=1 \
            LLAMA_LAZY_MOE_BUFFER_AUTO=1 \
            LLAMA_LAZY_MOE_BUFFER_WORKERS=4 \
            LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0 \
            LLAMA_LAZY_MOE_SIDECAR="$MOE_SIDECAR" \
            LLAMA_LAZY_DEBUG=1
    fi
}

make_body() {
    python3 - "$PROMPT" "$TOKENS" "$REQUEST_MODE" <<'PY'
import json
import sys

prompt = sys.argv[1]
tokens = int(sys.argv[2])
mode = sys.argv[3]
if mode == "chat":
    print(json.dumps({
        "model": "default",
        "messages": [
            {"role": "system", "content": "You are a concise technical assistant. Answer in clear English with numbered points."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": tokens,
        "temperature": 0,
    }))
else:
    print(json.dumps({
        "model": "default",
        "prompt": prompt,
        "max_tokens": tokens,
        "temperature": 0,
    }))
PY
}

wait_health() {
    local pid="$1"
    local log="$2"
    local port="$3"
    local deadline=$((SECONDS + 260))
    while (( SECONDS < deadline )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "server exited early; see $log" >&2
            return 1
        fi
        if curl --noproxy '*' -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "server health check timed out; see $log" >&2
    return 1
}

run_case() {
    local case_name="$1"
    local port="$2"
    local log="$OUT_DIR/${case_name}.server.log"
    local response="$OUT_DIR/${case_name}.json"
    local env_file="$OUT_DIR/${case_name}.env.txt"
    local samples="$OUT_DIR/${case_name}.cgroup-samples.tsv"
    local events="$OUT_DIR/${case_name}.memory.events"
    local cgroup=""
    local endpoint="completions"
    local body
    [[ "$REQUEST_MODE" == "chat" ]] && endpoint="chat/completions"

    if [[ "$case_name" == "baseline" ]]; then
        mapfile -t env_args < <(baseline_env)
    else
        mapfile -t env_args < <(combined_env)
    fi
    printf '%s\n' "${env_args[@]}" > "$env_file"

    if [[ "$DROP_CACHES_BEFORE_CASE" == "1" ||
            ( "$DROP_CACHES_BEFORE_BASELINE" == "1" && "$case_name" == "baseline" ) ]]; then
        echo "[$case_name] dropping host page cache for cold-start cgroup accounting"
        sync
        if ! printf '3\n' > /proc/sys/vm/drop_caches 2>/dev/null; then
            echo "failed to drop caches; run as root or set DROP_CACHES_BEFORE_CASE=0" >&2
            exit 3
        fi
    fi
    if [[ "$PREWARM_CACHE_BEFORE_COMBINED" == "1" && "$case_name" == "combined_auto" ]]; then
        echo "[$case_name] prewarming model file cache outside the test cgroup"
        sync
        if ! printf '3\n' > /proc/sys/vm/drop_caches 2>/dev/null; then
            echo "failed to drop caches before prewarm; run as root or set PREWARM_CACHE_BEFORE_COMBINED=0" >&2
            exit 3
        fi
        cat "$MODEL" >/dev/null
        if [[ "$MODEL_KIND" == "moe" && -f "$MOE_SIDECAR" ]]; then
            cat "$MOE_SIDECAR" >/dev/null
        fi
    fi

    cgroup="/sys/fs/cgroup/video-compare-${MODEL_KIND}-${case_name}-$$"
    if mkdir -p "$cgroup" 2>/dev/null; then
        printf '%s\n' $((MEM_MB * 1024 * 1024)) > "$cgroup/memory.max" 2>/dev/null || true
        printf '%s\n' $((HIGH_MB * 1024 * 1024)) > "$cgroup/memory.high" 2>/dev/null || true
        printf '0\n' > "$cgroup/memory.swap.max" 2>/dev/null || true
        cleanup_cgroups+=("$cgroup")
    else
        cgroup=""
        echo "warning: could not create cgroup for $case_name; running without live memory.max/high" >&2
    fi

    echo "[$case_name] starting llama-server"
    cmd=("$SERVER" -m "$MODEL" --host 127.0.0.1 --port "$port" -c "$CTX" -b "$BATCH" -ub "$BATCH" -t "$THREADS" -np 1 --no-webui)
    if [[ -n "$cgroup" ]]; then
        (
            printf '%s\n' "$BASHPID" > "$cgroup/cgroup.procs"
            exec env "${env_args[@]}" "${cmd[@]}"
        ) > "$log" 2>&1 &
    else
        env "${env_args[@]}" "${cmd[@]}" > "$log" 2>&1 &
    fi
    local pid=$!
    cleanup_pids+=("$pid")
    local mon_pid=""
    if [[ -n "$cgroup" ]]; then
        printf 'memory_current\tbytes\n' > "$samples"
        (
            while kill -0 "$pid" 2>/dev/null; do
                printf 'memory_current\t%s\n' "$(cat "$cgroup/memory.current" 2>/dev/null || echo 0)" >> "$samples"
                sleep 1
            done
        ) &
        mon_pid=$!
    fi
    wait_health "$pid" "$log" "$port"

    echo "[$case_name] input"
    echo "------------------------------------------------------------"
    echo "$PROMPT"
    echo "------------------------------------------------------------"
    echo "[$case_name] output"
    echo "------------------------------------------------------------"
    body="$(make_body)"
    curl --noproxy '*' -sS "http://127.0.0.1:${port}/v1/${endpoint}" \
        --max-time "$REQUEST_TIMEOUT_SEC" \
        -H 'Content-Type: application/json' \
        -d "$body" > "$response"
    if [[ -n "$cgroup" ]]; then
        cat "$cgroup/memory.events" > "$events" 2>/dev/null || true
    fi
    python3 - "$response" "$log" "$samples" "$events" <<'PY'
import json
import re
import sys
from pathlib import Path

data = json.load(open(sys.argv[1], encoding="utf-8"))
choice = data.get("choices", [{}])[0]
text = choice.get("text")
if text is None:
    text = choice.get("message", {}).get("content", "")
print((text or "").strip())

log = Path(sys.argv[2]).read_text(errors="ignore")
sample_path = Path(sys.argv[3])
events_path = Path(sys.argv[4])
matches = re.findall(r"eval time =\s+([0-9.]+) ms /\s+([0-9]+) tokens.*?([0-9.]+) tokens per second", log)
prompt_matches = re.findall(r"prompt eval time =\s+([0-9.]+) ms /\s+([0-9]+) tokens.*?([0-9.]+) tokens per second", log)
rss = [int(x) / 1024 / 1024 for x in re.findall(r"rss_observed_bytes=([0-9]+)", log)]
cgroup = [int(x) / 1024 / 1024 for x in re.findall(r"pressure_current_bytes=([0-9]+)", log)]
window = [int(x) / 1024 / 1024 for x in re.findall(r"window_peak_rss_bytes=([0-9]+)", log)]
moe_resident = [int(x) / 1024 / 1024 for x in re.findall(r"moe_resident_bytes=([0-9]+)", log)]
decision = re.findall(r"global_optimizer_decision=([^\s]+)", log)
kv = re.findall(r"kv_offload_reason=([^\s]+)", log)
cgroup_sample_peak = None
if sample_path.exists():
    vals = []
    for line in sample_path.read_text(errors="ignore").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                vals.append(int(parts[1]) / 1024 / 1024)
            except ValueError:
                pass
    if vals:
        cgroup_sample_peak = max(vals)
events = {}
if events_path.exists():
    for line in events_path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if len(parts) == 2:
            events[parts[0]] = parts[1]

print("\nlive measured metrics:")
if prompt_matches:
    ms, tokens, tps = prompt_matches[-1]
    print(f"  prompt_eval_tokens={tokens} prompt_eval_ms={ms} prompt_tok_s={tps}")
if matches:
    ms, tokens, tps = matches[-1]
    print(f"  generation_tokens={tokens} generation_eval_ms={ms} generation_tok_s={tps}")
if cgroup:
    print(f"  cgroup_physical_peak_mb={max(cgroup):.2f}")
if cgroup_sample_peak is not None:
    print(f"  cgroup_sample_peak_mb={cgroup_sample_peak:.2f}")
if rss:
    print(f"  rss_observed_peak_mb={max(rss):.2f}")
if window:
    print(f"  window_peak_rss_mb={max(window):.2f}")
if moe_resident:
    print(f"  moe_resident_peak_mb={max(moe_resident):.2f}")
if decision:
    print(f"  global_optimizer_decision={decision[-1]}")
if kv:
    print(f"  kv_offload_reason={kv[-1]}")
if events:
    print(f"  cgroup_events=max:{events.get('max', 'NA')} oom:{events.get('oom', 'NA')} oom_kill:{events.get('oom_kill', 'NA')}")
usage = data.get("usage", {})
if usage:
    print("  api_usage=" + json.dumps(usage, ensure_ascii=False, sort_keys=True))
PY
    echo "------------------------------------------------------------"
    if [[ -n "$mon_pid" ]]; then
        kill "$mon_pid" 2>/dev/null || true
        wait "$mon_pid" 2>/dev/null || true
    fi
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
}

print_readme_baseline() {
    local matrix_dir="$ROOT/rss-stage-results/full-global-performance-matrix-20260812-041553"
    local raw="$matrix_dir/raw.tsv"
    local run_dir="$matrix_dir/capacity-${MODEL_KIND}-baseline-3000M-high2800M-ctx2048-np1-tok64-r1"
    local response="$run_dir/completion.json"
    local log="$run_dir/server.log"

    if [[ ! -f "$raw" || ! -f "$response" || ! -f "$log" ]]; then
        echo "README baseline artifacts missing under $matrix_dir" >&2
        exit 2
    fi

    echo "[baseline] README reference baseline"
    echo "------------------------------------------------------------"
    echo "source: $run_dir"
    echo "input"
    echo "------------------------------------------------------------"
    echo "$PROMPT"
    echo "------------------------------------------------------------"
    echo "output"
    echo "------------------------------------------------------------"
    python3 - "$response" "$raw" "$MODEL_KIND" <<'PY'
import csv
import json
import sys

response_path, raw_path, model_kind = sys.argv[1:4]
data = json.load(open(response_path, encoding="utf-8"))
choice = data.get("choices", [{}])[0]
print((choice.get("text") or choice.get("message", {}).get("content", "") or "").strip())

row = None
with open(raw_path, newline="", encoding="utf-8") as f:
    for item in csv.DictReader(f, delimiter="\t"):
        if (
            item.get("workload") == "capacity"
            and item.get("model_kind") == model_kind
            and item.get("case") == "baseline"
            and item.get("mem_mb") == "3000"
            and item.get("high_mb") == "2800"
            and item.get("tokens") == "64"
        ):
            row = item
            break
if row is None:
    raise SystemExit("README baseline row not found")

def mb_from_bytes(value):
    try:
        return float(value) / 1024.0 / 1024.0
    except Exception:
        return None

peak = mb_from_bytes(row.get("mem_peak", ""))
print("\nhistorical README metrics:")
print(f"  generation_tokens={row.get('tokens')}")
print(f"  generation_tok_s={float(row.get('server_eval_tok_s')):.3f}")
if peak is not None:
    print(f"  cgroup_physical_peak_mb={peak:.2f}")
print(f"  global_optimizer_decision={row.get('global_decision')}")
print(f"  kv_offload_reason={row.get('kv_offload_reason')}")
print(f"  output_sha={row.get('output_sha')}")
print(f"  status={row.get('status')} oom={row.get('oom')} oom_kill={row.get('oom_kill')}")
PY
    echo "------------------------------------------------------------"
}

echo "============================================================"
echo "llama.cpp compare demo: ${MODEL_KIND}"
echo "memory.max=${MEM_MB}M memory.high=${HIGH_MB}M"
echo "artifacts: $OUT_DIR"
echo "============================================================"
if [[ "$MODEL_KIND" == "moe" ]]; then
    echo "note: this MoE benchmark model produces weak natural-language text on both paths;"
    echo "      use the paired run to show the issue is not introduced by combined_auto."
    echo
fi

if [[ "$BASELINE_SOURCE" == "readme" ]]; then
    print_readme_baseline
else
    run_case baseline "$PORT"
fi
echo
run_case combined_auto "$((PORT + 1))"
echo

echo "Demo artifacts saved to: $OUT_DIR"
