#!/bin/bash
# Test script for Lazy Loading V2 implementation

set -e

echo "=== Lazy Loading V2 Test Script ==="
echo ""

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_model.gguf> [prompt] [n_predict]"
    echo ""
    echo "Example:"
    echo "  $0 models/llama-7b-q4.gguf \"Hello\" 50"
    exit 1
fi

MODEL_PATH="$1"
PROMPT="${2:-Hello, how are you?}"
N_PREDICT="${3:-50}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Tokens to generate: $N_PREDICT"
echo ""

# Configuration
export LLAMA_LAZY_V2=1              # Enable Lazy V2
export LLAMA_LAZY_V2_WINDOW=12      # Keep 12 layers in memory
export LLAMA_LAZY_V2_PREFETCH=4     # Prefetch 4 layers ahead
export LLAMA_LAZY_V2_WORKERS=2      # 2 background loading threads
export LLAMA_LAZY_V2_MEMORY_GB=32   # Memory limit (optional)

echo "=== Lazy V2 Configuration ==="
echo "LLAMA_LAZY_V2=$LLAMA_LAZY_V2"
echo "LLAMA_LAZY_V2_WINDOW=$LLAMA_LAZY_V2_WINDOW"
echo "LLAMA_LAZY_V2_PREFETCH=$LLAMA_LAZY_V2_PREFETCH"
echo "LLAMA_LAZY_V2_WORKERS=$LLAMA_LAZY_V2_WORKERS"
if [ -n "$LLAMA_LAZY_V2_MEMORY_GB" ]; then
    echo "LLAMA_LAZY_V2_MEMORY_GB=$LLAMA_LAZY_V2_MEMORY_GB"
fi
echo ""

# Run inference
echo "=== Running Inference with Lazy V2 ==="
echo ""

./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "$PROMPT" \
    -n "$N_PREDICT" \
    --verbose-prompt

echo ""
echo "=== Test Complete ==="
