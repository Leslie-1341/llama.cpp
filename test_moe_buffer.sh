#!/bin/bash
# MoE buffer真实物理footprint：cap=4GB(>模型)看自然占用，对比 mmap vs 匿名缓冲。
set -u
cd /root/llama.cpp
M=/root/models/Qwen1.5-MoE-A2.7B-20-experts-SFT-trained.Q4_K_M.gguf
BENCH=./build/bin/llama-bench
LIMIT="${LIMIT:-4G}"

run() {
    local tag="$1"; shift
    local cg="/sys/fs/cgroup/mb_$tag"
    mkdir -p "$cg"; echo "$LIMIT" > "$cg/memory.max"; echo 0 > "$cg/memory.swap.max" 2>/dev/null
    sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
    ( echo $BASHPID > "$cg/cgroup.procs"
      exec timeout 600 env "$@" $BENCH -m "$M" -p 16 -n 64 -t 6 -r 1 ) \
        </dev/null 1>"/tmp/mb_${tag}.out" 2>"/tmp/mb_${tag}.err" &
    local pid=$!; local peak=0
    while kill -0 "$pid" 2>/dev/null; do
        local cur=$(cat "$cg/memory.current" 2>/dev/null)
        [ -n "$cur" ] && [ "$cur" -gt "$peak" ] && peak=$cur
        sleep 0.3
    done
    wait "$pid" 2>/dev/null
    echo "================ [$tag] ================"
    grep -E "pp16|tg64" "/tmp/mb_${tag}.out" 2>/dev/null
    printf "peak memory.current: %d MB\n" $((peak/1024/1024))
    echo
}

echo "model=$(basename $M) cap=$LIMIT"
run v2_mmap        LLAMA_LAZY_V2=1
run moebuf_unbound LLAMA_LAZY_V2=1 LLAMA_LAZY_MOE_BUFFER=1
run moebuf_384     LLAMA_LAZY_V2=1 LLAMA_LAZY_MOE_BUFFER=1 LLAMA_LAZY_MOE_BUFFER_MB=384
echo "ALL_DONE"
