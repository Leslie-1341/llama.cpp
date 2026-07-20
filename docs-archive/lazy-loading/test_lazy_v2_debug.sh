#!/bin/bash
# Debug test for Lazy Loading V2 - checks if system initializes correctly

set -e

echo "=== Lazy Loading V2 Debug Test ==="
echo ""

# Check if model path is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_model.gguf>"
    echo ""
    echo "This will just load the model and generate a few tokens to verify Lazy V2 works"
    exit 1
fi

MODEL_PATH="$1"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

echo "Model: $MODEL_PATH"
echo ""

# Enable Lazy V2 with debug logging
export LLAMA_LAZY_V2=1
export LLAMA_LAZY_V2_DEBUG=1
export LLAMA_LAZY_V2_WINDOW=8
export LLAMA_LAZY_V2_PREFETCH=3
export LLAMA_LAZY_V2_WORKERS=2

echo "=== Lazy V2 Configuration (Debug Mode) ==="
echo "LLAMA_LAZY_V2=$LLAMA_LAZY_V2"
echo "LLAMA_LAZY_V2_DEBUG=$LLAMA_LAZY_V2_DEBUG"
echo "LLAMA_LAZY_V2_WINDOW=$LLAMA_LAZY_V2_WINDOW"
echo "LLAMA_LAZY_V2_PREFETCH=$LLAMA_LAZY_V2_PREFETCH"
echo "LLAMA_LAZY_V2_WORKERS=$LLAMA_LAZY_V2_WORKERS"
echo ""

echo "=== Loading Model (watch for Lazy V2 initialization messages) ==="
echo ""

# Run a minimal inference - just 5 tokens to verify it works
./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "Hello" \
    -n 5 \
    2>&1 | grep -E "\[Lazy-V2\]|error|Error" || true

echo ""
echo "=== Checking for Lazy V2 Activity ==="
echo ""

# Run again and capture all output
./build/bin/llama-cli \
    -m "$MODEL_PATH" \
    -p "Test" \
    -n 3 \
    2>&1 | head -50

echo ""
echo "=== Debug Test Complete ==="
echo ""
echo "Look for these messages above:"
echo "  - [Lazy-V2] Initializing Lazy V2 system..."
echo "  - [Lazy-V2] ✓ Initialization successful"
echo "  - [Lazy-V2] Loading tensor: ..."
echo "  - [Lazy-V2] Advanced to layer N"
