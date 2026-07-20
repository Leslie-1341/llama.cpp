#!/bin/bash
# Window RSS-vs-speed across memory_limit (unconstrained RAM, steady-state VmRSS).
set -u
cd /root/llama.cpp
M=models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf
BENCH=./build/bin/llama-bench

probe() {
    local tag="$1"; shift
    : > "/tmp/wl_${tag}.samples"
    env "$@" $BENCH -m "$M" -p 16 -n 96 -t 6 -r 1 </dev/null 1>"/tmp/wl_${tag}.out" 2>"/tmp/wl_${tag}.err" &
    local bpid=$!
    sleep 2
    local pid=$(pgrep -P "$bpid" -f llama-bench); [ -z "$pid" ] && pid=$bpid
    while kill -0 "$pid" 2>/dev/null; do
        local rss=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
        [ -n "$rss" ] && echo "$rss" >> "/tmp/wl_${tag}.samples"
        sleep 0.3
    done
    wait "$bpid" 2>/dev/null
    echo "================ $tag ================"
    grep -E "tg96" "/tmp/wl_${tag}.out" 2>/dev/null
    sort -n "/tmp/wl_${tag}.samples" | awk '{a[NR]=$1} END{if(NR)printf "steady RSS: median=%.0f MB  max=%.0f MB  (%d samples)\n", a[int(NR/2)]/1024, a[NR]/1024, NR}'
    echo
}

probe native
probe win_prefetch_only LLAMA_LAZY_V2=1
probe win_limit_4g      LLAMA_LAZY_V2=1 LLAMA_LAZY_DONTNEED=1 LLAMA_LAZY_MEMORY_LIMIT=4
probe win_limit_1g      LLAMA_LAZY_V2=1 LLAMA_LAZY_DONTNEED=1 LLAMA_LAZY_MEMORY_LIMIT=1
echo "ALL_DONE"
