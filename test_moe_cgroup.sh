#!/bin/bash
# MoE expert-offloading under real memory pressure (cgroup memory.max < model).
# Measures peak cgroup memory.current (true physical footprint) + tok/s, per config.
set -u
cd /root/llama.cpp
M=/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf
BENCH=./build/bin/llama-bench
LIMIT="${LIMIT:-2G}"

run() {
    local tag="$1"; shift
    local cg="/sys/fs/cgroup/moe_$tag"
    mkdir -p "$cg"; echo "$LIMIT" > "$cg/memory.max"; echo 0 > "$cg/memory.swap.max" 2>/dev/null
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
    ( echo $BASHPID > "$cg/cgroup.procs"
      exec timeout 600 env "$@" $BENCH -m "$M" -p 16 -n 64 -t 6 -r 1 ) \
        </dev/null 1>"/tmp/moe_${tag}.out" 2>"/tmp/moe_${tag}.err" &
    local pid=$!
    # sample memory.current for the peak
    local peak=0
    while kill -0 "$pid" 2>/dev/null; do
        local cur=$(cat "$cg/memory.current" 2>/dev/null)
        [ -n "$cur" ] && [ "$cur" -gt "$peak" ] && peak=$cur
        sleep 0.3
    done
    wait "$pid" 2>/dev/null; local rc=$?
    echo "================ [$tag] (memory.max=$LIMIT) rc=$rc ================"
    grep -E "pp16|tg64" "/tmp/moe_${tag}.out" 2>/dev/null
    printf "peak memory.current: %d MB\n" $((peak/1024/1024))
    echo "events: $(grep -E 'oom' $cg/memory.events 2>/dev/null | tr '\n' ' ')"
    echo
}

echo "model: $(basename $M)  limit=$LIMIT"
run A LLAMA_LAZY_V2=1
run B LLAMA_LAZY_V2=1 LLAMA_LAZY_MOE_WINDOW=1 LLAMA_LAZY_MOE_DONTNEED=1
run C LLAMA_LAZY_V2=1 LLAMA_LAZY_CLG=1
run D LLAMA_LAZY_V2=1 LLAMA_LAZY_CLG=1 LLAMA_LAZY_MOE_DONTNEED=1
echo "ALL_DONE"
