#!/usr/bin/env bash
# Build and benchmark experimental dense flex FFN-only requantization profiles.
#
# The input is expected to be an already-quantized dense GGUF. This script uses
# llama-quantize --allow-requantize and only overrides blk.N.ffn_{gate,up,down}
# tensors, leaving attention, embeddings and output on the normal target type.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANT="${QUANT:-$ROOT/build/bin/llama-quantize}"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/dense-flex-quant}"
PROFILE="${PROFILE:-balanced}"
THREADS="${THREADS:-6}"
NTHREADS_QUANT="${NTHREADS_QUANT:-$(nproc)}"
DO_QUANT="${DO_QUANT:-0}"
DO_BENCH="${DO_BENCH:-0}"
TYPE_FILE="${TYPE_FILE:-}"

usage() {
    cat <<EOF
usage: MODEL=/path/model.gguf PROFILE=safe|balanced|aggressive [DO_QUANT=1] [DO_BENCH=1] $0

profiles:
  safe        ffn_down=q4_K, ffn_gate/up=q3_K
  balanced    ffn_gate/up/down=q3_K
  aggressive  ffn_gate/up/down=q2_K

outputs:
  \$OUT_DIR/<model>.ffn-<profile>.types
  \$OUT_DIR/<model>.ffn-<profile>.gguf when DO_QUANT=1
  \$OUT_DIR/<model>.ffn-<profile>.bench.out when DO_BENCH=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$MODEL" ]]; then
    echo "missing MODEL: $MODEL" >&2
    exit 1
fi
if [[ ! -x "$QUANT" ]]; then
    echo "missing llama-quantize: $QUANT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"
base="$(basename "$MODEL" .gguf)"
name="$base.ffn-$PROFILE"
out_model="$OUT_DIR/$name.gguf"
type_file="${TYPE_FILE:-$OUT_DIR/$name.types}"
dry_log="$OUT_DIR/$name.dry-run.log"
quant_log="$OUT_DIR/$name.quant.log"
bench_log="$OUT_DIR/$name.bench.out"

case "$PROFILE" in
    safe)
        cat > "$type_file" <<'EOF'
blk\.\d+\.ffn_down\.weight=q4_K
blk\.\d+\.ffn_(gate|up)\.weight=q3_K
EOF
        target="Q4_K_M"
        ;;
    balanced)
        cat > "$type_file" <<'EOF'
blk\.\d+\.ffn_(gate|up|down)\.weight=q3_K
EOF
        target="Q4_K_M"
        ;;
    aggressive)
        cat > "$type_file" <<'EOF'
blk\.\d+\.ffn_(gate|up|down)\.weight=q2_K
EOF
        target="Q4_K_M"
        ;;
    *)
        echo "unknown PROFILE: $PROFILE" >&2
        usage >&2
        exit 1
        ;;
esac

echo "profile:   $PROFILE"
echo "model:     $MODEL"
echo "type file: $type_file"
echo "out model: $out_model"

"$QUANT" --dry-run --allow-requantize --tensor-type-file "$type_file" \
    "$MODEL" "$out_model" "$target" > "$dry_log" 2>&1

overrides="$(grep -c 'applying manual override' "$dry_log" || true)"
echo "dry-run:   $dry_log"
echo "overrides: $overrides"
grep -E 'applying manual override|model size|total size|size =' "$dry_log" | tail -20 || true

if [[ "$DO_QUANT" == "1" ]]; then
    "$QUANT" --allow-requantize --tensor-type-file "$type_file" \
        "$MODEL" "$out_model" "$target" "$NTHREADS_QUANT" > "$quant_log" 2>&1
    echo "quant:     $quant_log"
    ls -lh "$out_model"
fi

if [[ "$DO_BENCH" == "1" ]]; then
    if [[ ! -f "$out_model" ]]; then
        echo "cannot benchmark, quantized model missing: $out_model" >&2
        exit 1
    fi
    env LLAMA_FLEX=1 LLAMA_FLEX_DEBUG=1 LLAMA_FLEX_AHEAD="${LLAMA_FLEX_AHEAD:-3}" \
        LLAMA_FLEX_THREADS="${LLAMA_FLEX_THREADS:-4}" \
        LLAMA_FLEX_LOCK_GB="${LLAMA_FLEX_LOCK_GB:-1}" \
        LLAMA_FLEX_PIN_POLICY="${LLAMA_FLEX_PIN_POLICY:-attn-first}" \
        "$BENCH" -m "$out_model" -p "${PROMPT_TOKENS:-1}" -n "${GEN_TOKENS:-16}" \
        -t "$THREADS" -ngl 0 -r "${REPEAT:-1}" -o md > "$bench_log" 2>&1
    echo "bench:     $bench_log"
    grep -E 'llama_flex: layers=|tg[0-9]+|llama_flex IO:' "$bench_log" || true
fi
