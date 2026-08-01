#!/usr/bin/env bash
# Build and benchmark experimental dense flex FFN-only requantization profiles.
#
# The input is expected to be an already-quantized dense GGUF. This script uses
# llama-quantize --allow-requantize and only overrides blk.N.ffn_{gate,up,down}
# tensors, leaving attention, embeddings and output on the normal target type.
# When DO_IMATRIX=1 or IMATRIX points at an existing file, quantization uses
# llama.cpp's activation importance matrix. This is the first AWQ-like path for
# dense-flex experiments: collect activation-aware per-channel importance, then
# try lower-bit FFN profiles without adding mixed precision at runtime.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUANT="${QUANT:-$ROOT/build/bin/llama-quantize}"
BENCH="${BENCH:-$ROOT/build/bin/llama-bench}"
IMATRIX_TOOL="${IMATRIX_TOOL:-$ROOT/build/bin/llama-imatrix}"
PPL_TOOL="${PPL_TOOL:-$ROOT/build/bin/llama-perplexity}"

MODEL="${MODEL:-/root/models/Meta-Llama-3-8B-Instruct-Q4_K_M-aligned.gguf}"
OUT_DIR="${OUT_DIR:-/tmp/dense-flex-quant}"
PROFILE="${PROFILE:-balanced}"
THREADS="${THREADS:-6}"
NTHREADS_QUANT="${NTHREADS_QUANT:-$(nproc)}"
CALIB_FILE="${CALIB_FILE:-}"
CALIB_CHUNKS="${CALIB_CHUNKS:-8}"
DO_IMATRIX="${DO_IMATRIX:-0}"
DO_QUANT="${DO_QUANT:-0}"
DO_BENCH="${DO_BENCH:-0}"
DO_PPL="${DO_PPL:-0}"
PPL_FILE="${PPL_FILE:-}"
PPL_CHUNKS="${PPL_CHUNKS:-4}"
TYPE_FILE="${TYPE_FILE:-}"
IMATRIX="${IMATRIX:-}"

usage() {
    cat <<EOF
usage: MODEL=/path/model.gguf PROFILE=safe|balanced|bplus-a|bplus-b|bplus-c|aggressive [DO_IMATRIX=1] [DO_QUANT=1] [DO_PPL=1] [DO_BENCH=1] $0

profiles:
  safe        ffn_down=q4_K, ffn_gate/up=q3_K
  balanced    ffn_gate/up/down=q3_K
  bplus-a     ffn_gate=q2_K, ffn_up/down=q3_K
  bplus-b     ffn_gate/up=q2_K, ffn_down=q3_K
  bplus-c     ffn_up=q2_K, ffn_gate/down=q3_K
  aggressive  ffn_gate/up/down=q2_K

activation-aware path:
  DO_IMATRIX=1              collect an imatrix before quantization
  CALIB_FILE=/path.txt      calibration text; default builds one from eam-workload
  CALIB_CHUNKS=8            chunks for llama-imatrix
  IMATRIX=/path/imatrix.gguf use an existing imatrix

outputs:
  \$OUT_DIR/<model>.ffn-<profile>.types
  \$OUT_DIR/<model>.ffn-<profile>.imatrix.gguf when DO_IMATRIX=1
  \$OUT_DIR/<model>.ffn-<profile>.gguf when DO_QUANT=1
  \$OUT_DIR/<model>.ffn-<profile>.ppl.out when DO_PPL=1
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
imatrix_log="$OUT_DIR/$name.imatrix.log"
imatrix_file="${IMATRIX:-$OUT_DIR/$name.imatrix.gguf}"
quant_log="$OUT_DIR/$name.quant.log"
ppl_log="$OUT_DIR/$name.ppl.out"
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
    bplus-a)
        cat > "$type_file" <<'EOF'
blk\.\d+\.ffn_gate\.weight=q2_K
blk\.\d+\.ffn_(up|down)\.weight=q3_K
EOF
        target="Q4_K_M"
        ;;
    bplus-b)
        cat > "$type_file" <<'EOF'
blk\.\d+\.ffn_(gate|up)\.weight=q2_K
blk\.\d+\.ffn_down\.weight=q3_K
EOF
        target="Q4_K_M"
        ;;
    bplus-c)
        cat > "$type_file" <<'EOF'
blk\.\d+\.ffn_up\.weight=q2_K
blk\.\d+\.ffn_(gate|down)\.weight=q3_K
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
echo "imatrix:   $imatrix_file"

if [[ -z "$CALIB_FILE" ]]; then
    CALIB_FILE="$OUT_DIR/calib-sharegpt-128.txt"
    if [[ ! -f "$CALIB_FILE" ]]; then
        find "$ROOT/eam-workload/processed/prompts_1k" -maxdepth 1 -name '*.txt' | sort > "$OUT_DIR/calib-files.txt"
        head -n 128 "$OUT_DIR/calib-files.txt" | xargs sed -n '1,200p' > "$CALIB_FILE"
    fi
fi
if [[ -z "$PPL_FILE" ]]; then
    PPL_FILE="$CALIB_FILE"
fi

if [[ "$DO_IMATRIX" == "1" ]]; then
    if [[ ! -x "$IMATRIX_TOOL" ]]; then
        echo "missing llama-imatrix: $IMATRIX_TOOL" >&2
        exit 1
    fi
    "$IMATRIX_TOOL" -m "$MODEL" -f "$CALIB_FILE" -o "$imatrix_file" \
        --chunks "$CALIB_CHUNKS" --no-ppl -t "$THREADS" -ngl 0 > "$imatrix_log" 2>&1
    echo "imatrix log: $imatrix_log"
fi

quant_extra=()
if [[ -f "$imatrix_file" ]]; then
    quant_extra+=(--imatrix "$imatrix_file")
fi

"$QUANT" --dry-run --allow-requantize "${quant_extra[@]}" --tensor-type-file "$type_file" \
    "$MODEL" "$out_model" "$target" > "$dry_log" 2>&1

overrides="$(grep -c 'applying manual override' "$dry_log" || true)"
echo "dry-run:   $dry_log"
echo "overrides: $overrides"
grep -E 'applying manual override|model size|total size|size =' "$dry_log" | tail -20 || true

if [[ "$DO_QUANT" == "1" ]]; then
    "$QUANT" --allow-requantize "${quant_extra[@]}" --tensor-type-file "$type_file" \
        "$MODEL" "$out_model" "$target" "$NTHREADS_QUANT" > "$quant_log" 2>&1
    echo "quant:     $quant_log"
    ls -lh "$out_model"
fi

if [[ "$DO_PPL" == "1" ]]; then
    if [[ ! -f "$out_model" ]]; then
        echo "cannot run PPL, quantized model missing: $out_model" >&2
        exit 1
    fi
    if [[ ! -x "$PPL_TOOL" ]]; then
        echo "missing llama-perplexity: $PPL_TOOL" >&2
        exit 1
    fi
    "$PPL_TOOL" -m "$out_model" -f "$PPL_FILE" -c 512 -b 512 -ub 512 \
        -t "$THREADS" -ngl 0 --chunks "$PPL_CHUNKS" --no-warmup > "$ppl_log" 2>&1
    echo "ppl:       $ppl_log"
    grep -E 'Final estimate|seconds per pass' "$ppl_log" || true
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
