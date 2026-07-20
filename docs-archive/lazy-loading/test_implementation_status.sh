#!/bin/bash
# Quick comparison test: Standard vs Lazy V2

set -e

MODEL="models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf"
PROMPT="Hello, how are you today?"
N_PREDICT=50

echo "=== Lazy Loading V2 Performance Test ==="
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Tokens: $N_PREDICT"
echo ""

# Test 1: Standard mode (baseline)
echo "=== Test 1: Standard Mode (Baseline) ==="
export LLAMA_LAZY_V2=0
export LLAMA_LAZY_LOADING=0

echo "Running standard inference..."
./build/bin/llama-cli -m "$MODEL" -p "$PROMPT" -n $N_PREDICT --no-display-prompt 2>&1 | tee /tmp/standard_output.txt

# Extract TPS
STANDARD_TPS=$(grep -oP 'eval time.*?(\K[0-9.]+)(?= tokens per second)' /tmp/standard_output.txt || echo "N/A")
echo "Standard TPS: $STANDARD_TPS"
echo ""

# Get RSS
echo "Checking memory usage..."
sleep 2

echo ""
echo "=== Summary ==="
echo "Standard TPS: $STANDARD_TPS"
echo ""
echo "Note: Lazy V2 integration is not complete yet."
echo "The system has been implemented but needs full integration with model loader."
echo ""
echo "Implementation status:"
echo "  ✅ Core engine (llama-lazy-v2.cpp)"
echo "  ✅ Data structures and async prefetch"
echo "  ✅ Integration layer (llama-lazy-v2-integration.cpp)"
echo "  ⚠️  Model loader integration (partially done)"
echo "  ❌ Actual usage in inference loop (not connected)"
echo ""
echo "To complete:"
echo "  1. Call llama_lazy_v2_init_from_loader() in model loader"
echo "  2. Use llama_lazy_v2_load_tensor_data() in load_data_for()"
echo "  3. Call llama_lazy_v2_advance_to_layer() in inference loop"
