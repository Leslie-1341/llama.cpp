#!/bin/bash
# Comprehensive benchmark for Lazy Loading V2

set -e

MODEL="${1:-/root/llama.cpp/models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf}"
PROMPT="${2:-Hello, how are you today? I would like to discuss artificial intelligence and machine learning.}"
N_TOKENS="${3:-50}"

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found: $MODEL"
    exit 1
fi

echo "=========================================="
echo "Lazy Loading V2 - Performance Benchmark"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens to generate: $N_TOKENS"
echo ""

# Test 1: Baseline (without Lazy V2)
echo "=========================================="
echo "Test 1: Baseline (Lazy V2 disabled)"
echo "=========================================="
unset LLAMA_LAZY_V2
unset LLAMA_LAZY_V2_DEBUG

echo "Running baseline test..."
time ./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" --no-display-prompt 2>&1 | tee baseline_output.txt | grep -E "(Prompt:|Generation:|build time|load time)"

echo ""

# Test 2: Lazy V2 with default settings
echo "=========================================="
echo "Test 2: Lazy V2 (window=8, prefetch=3)"
echo "=========================================="
export LLAMA_LAZY_V2=1
export LLAMA_LAZY_V2_WINDOW=8
export LLAMA_LAZY_V2_PREFETCH=3
export LLAMA_LAZY_V2_WORKERS=2

echo "Running Lazy V2 test..."
time ./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" --no-display-prompt 2>&1 | tee lazy_v2_output.txt | grep -E "(Prompt:|Generation:|build time|load time|\[Lazy-V2\])"

echo ""

# Test 3: Lazy V2 with small window
echo "=========================================="
echo "Test 3: Lazy V2 (window=4, prefetch=2)"
echo "=========================================="
export LLAMA_LAZY_V2_WINDOW=4
export LLAMA_LAZY_V2_PREFETCH=2

echo "Running small window test..."
time ./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" --no-display-prompt 2>&1 | tee lazy_v2_small_output.txt | grep -E "(Prompt:|Generation:|build time|load time|\[Lazy-V2\])"

echo ""

# Test 4: Lazy V2 with large window
echo "=========================================="
echo "Test 4: Lazy V2 (window=16, prefetch=5)"
echo "=========================================="
export LLAMA_LAZY_V2_WINDOW=16
export LLAMA_LAZY_V2_PREFETCH=5

echo "Running large window test..."
time ./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n "$N_TOKENS" --no-display-prompt 2>&1 | tee lazy_v2_large_output.txt | grep -E "(Prompt:|Generation:|build time|load time|\[Lazy-V2\])"

echo ""
echo "=========================================="
echo "Benchmark Complete"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - baseline_output.txt"
echo "  - lazy_v2_output.txt"
echo "  - lazy_v2_small_output.txt"
echo "  - lazy_v2_large_output.txt"
