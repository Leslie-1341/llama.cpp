#!/bin/bash
# Numeric-equivalence test: perplexity must match between native and window.
set -u
cd /root/llama.cpp
M=models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf
F=wikitext-2-raw/wiki.test.raw
PPL=./build/bin/llama-perplexity

run() {
    local tag="$1"; shift
    echo "================ $tag ================"
    timeout 1200 env "$@" $PPL -m "$M" -f "$F" -t 6 --chunks 4 </dev/null 2>"/tmp/ppl_${tag}.err" 1>"/tmp/ppl_${tag}.out"
    echo "rc=$?"
    grep -E "Final estimate|^\[4\]|PPL" "/tmp/ppl_${tag}.err" "/tmp/ppl_${tag}.out" 2>/dev/null | tail -3
    echo
}

run native
run window      LLAMA_LAZY_V2=1
run window_dn   LLAMA_LAZY_V2=1 LLAMA_LAZY_DONTNEED=1 LLAMA_LAZY_MEMORY_LIMIT=1
echo "ALL_DONE"
