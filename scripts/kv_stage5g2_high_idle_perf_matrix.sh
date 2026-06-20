cd /root/oscomp/llama.cpp

DIR=/root/oscomp/kv_logs/stage5g2_high_idle_perf_matrix
mkdir -p "$DIR"

MODEL=/root/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
BIN=build/bin/llama-kv-idle-swap-resume
N_RUNS=3

RUNS_TSV="$DIR/runs.tsv"
MEDIAN_TSV="$DIR/median.tsv"

printf "case\tmode\trun\tctx\tn\tn_over_ctx\texit_code\tseq1_equal\tseq0_equal\treal_abnormal\ttotal_wall_ms\tseq1_active_ms\tseq0_resume_first_token_ms\tseq0_resume_total_ms\tseq1_active_tokens\tseq0_resume_tokens\ttotal_measured_tokens\ttokens_per_second\tseq0_prefill_ms\tseq1_prefill_ms\tpaged_swap_rss_total_drop_kb\tpaged_swap_madvise_bytes\n" > "$RUNS_TSV"

extract_num() {
  local key="$1"
  local file="$2"
  local val
  val=$(grep -a -oE "${key}=[0-9]+(\.[0-9]+)?" "$file" | tail -1 | cut -d= -f2)
  if [ -z "$val" ]; then
    echo "0"
  else
    echo "$val"
  fi
}

run_one() {
  local case_name="$1"
  local ctx="$2"
  local n="$3"
  local mode="$4"
  local run_id="$5"

  echo "===== $case_name mode=$mode run=$run_id ctx=$ctx n=$n ====="

  COMMON_ARGS=(
    -m "$MODEL"
    -n "$n"
    --ctx-size "$ctx"
    --batch-size 128
    --ubatch-size 128
    --seed 1
    --temp 0
    --cache-type-k f32
    --cache-type-v f32
    --kv-unified
    --parallel 2
  )

  OUT="$DIR/${case_name}.${mode}.r${run_id}.out"
  ERR="$DIR/${case_name}.${mode}.r${run_id}.err"

  if [ "$mode" = "base" ]; then
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    "$BIN" "${COMMON_ARGS[@]}" > "$OUT" 2> "$ERR"
    exit_code=$?
  elif [ "$mode" = "nomadv" ]; then
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    "$BIN" "${COMMON_ARGS[@]}" > "$OUT" 2> "$ERR"
    exit_code=$?
  elif [ "$mode" = "madv" ]; then
    LLAMA_KV_PAGED=1 \
    LLAMA_KV_PAGED_INGRAPH=1 \
    LLAMA_KV_PAGED_GATHER_NONIDENTITY=1 \
    LLAMA_KV_PAGED_IDLE_TRACE=1 \
    LLAMA_KV_PAGED_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP=1 \
    LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 \
    "$BIN" "${COMMON_ARGS[@]}" > "$OUT" 2> "$ERR"
    exit_code=$?
  else
    echo "unknown mode: $mode" >&2
    return 1
  fi

  if [ "$mode" = "base" ]; then
    seq1_equal=0
    seq0_equal=0
  else
    BASE_OUT="$DIR/${case_name}.base.r${run_id}.out"

    awk '/===SEQ1_ACTIVE_BEGIN===/{flag=1;next}/===SEQ1_ACTIVE_END===/{flag=0}flag' \
      "$BASE_OUT" > "$DIR/${case_name}.base.r${run_id}.seq1.txt"
    awk '/===SEQ1_ACTIVE_BEGIN===/{flag=1;next}/===SEQ1_ACTIVE_END===/{flag=0}flag' \
      "$OUT" > "$DIR/${case_name}.${mode}.r${run_id}.seq1.txt"
    cmp -s "$DIR/${case_name}.base.r${run_id}.seq1.txt" "$DIR/${case_name}.${mode}.r${run_id}.seq1.txt"
    seq1_equal=$?

    awk '/===SEQ0_RESUME_BEGIN===/{flag=1;next}/===SEQ0_RESUME_END===/{flag=0}flag' \
      "$BASE_OUT" > "$DIR/${case_name}.base.r${run_id}.seq0.txt"
    awk '/===SEQ0_RESUME_BEGIN===/{flag=1;next}/===SEQ0_RESUME_END===/{flag=0}flag' \
      "$OUT" > "$DIR/${case_name}.${mode}.r${run_id}.seq0.txt"
    cmp -s "$DIR/${case_name}.base.r${run_id}.seq0.txt" "$DIR/${case_name}.${mode}.r${run_id}.seq0.txt"
    seq0_equal=$?
  fi

  real_abnormal=$(grep -nEi "error|failed|failure|nan|backend_fail|violation|assert|abort|segmentation" "$ERR" \
    | grep -vE "failures=0|failure=0|violation=0|backend_failures=0|paged_swap_madvise_failures=0|paged_release_violation=0|paged_active_release_violation=0|kv_mincore_failures=0" \
    | wc -l)

  total_wall_ms=$(extract_num total_wall_ms "$ERR")
  seq1_active_ms=$(extract_num seq1_active_ms "$ERR")
  seq0_resume_first_token_ms=$(extract_num seq0_resume_first_token_ms "$ERR")
  seq0_resume_total_ms=$(extract_num seq0_resume_total_ms "$ERR")
  seq1_active_tokens=$(extract_num seq1_active_tokens "$ERR")
  seq0_resume_tokens=$(extract_num seq0_resume_tokens "$ERR")
  total_measured_tokens=$(extract_num total_measured_tokens "$ERR")
  tokens_per_second=$(extract_num tokens_per_second "$ERR")
  seq0_prefill_ms=$(extract_num seq0_prefill_ms "$ERR")
  seq1_prefill_ms=$(extract_num seq1_prefill_ms "$ERR")
  paged_swap_rss_total_drop_kb=$(extract_num paged_swap_rss_total_drop_kb "$ERR")
  paged_swap_madvise_bytes=$(extract_num paged_swap_madvise_bytes "$ERR")

  python3 - <<PY >> "$RUNS_TSV"
case_name="$case_name"
mode="$mode"
run_id=int("$run_id")
ctx=int("$ctx")
n=int("$n")
n_over_ctx=n/ctx if ctx else 0
exit_code=int("$exit_code")
seq1_equal=int("$seq1_equal")
seq0_equal=int("$seq0_equal")
real_abnormal=int("$real_abnormal")
vals = [
    "$total_wall_ms",
    "$seq1_active_ms",
    "$seq0_resume_first_token_ms",
    "$seq0_resume_total_ms",
    "$seq1_active_tokens",
    "$seq0_resume_tokens",
    "$total_measured_tokens",
    "$tokens_per_second",
    "$seq0_prefill_ms",
    "$seq1_prefill_ms",
    "$paged_swap_rss_total_drop_kb",
    "$paged_swap_madvise_bytes",
]
print(
    f"{case_name}\t{mode}\t{run_id}\t{ctx}\t{n}\t{n_over_ctx:.4f}\t"
    f"{exit_code}\t{seq1_equal}\t{seq0_equal}\t{real_abnormal}\t" +
    "\t".join(vals)
)
PY
}

run_case() {
  local case_name="$1"
  local ctx="$2"
  local n="$3"

  for r in $(seq 1 "$N_RUNS"); do
    run_one "$case_name" "$ctx" "$n" base "$r"
    run_one "$case_name" "$ctx" "$n" nomadv "$r"
    run_one "$case_name" "$ctx" "$n" madv "$r"
  done
}

run_case mid_12p 1024 128
run_case mid_31p 1024 320
run_case mid_37p 1024 384

python3 - <<'PY' "$RUNS_TSV" "$MEDIAN_TSV"
import sys
import pandas as pd

runs_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(runs_path, sep="\t")

numeric_cols = [
    "total_wall_ms",
    "seq1_active_ms",
    "seq0_resume_first_token_ms",
    "seq0_resume_total_ms",
    "tokens_per_second",
    "seq0_prefill_ms",
    "seq1_prefill_ms",
    "paged_swap_rss_total_drop_kb",
    "paged_swap_madvise_bytes",
]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

group_cols = ["case", "mode", "ctx", "n", "n_over_ctx"]
med = df.groupby(group_cols, as_index=False)[numeric_cols].median()

# correctness / abnormal summary
chk = df.groupby(["case", "mode"], as_index=False).agg(
    exit_code_max=("exit_code", "max"),
    seq1_equal_max=("seq1_equal", "max"),
    seq0_equal_max=("seq0_equal", "max"),
    real_abnormal_max=("real_abnormal", "max"),
)
med = med.merge(chk, on=["case", "mode"], how="left")

# pivot for deltas
rows = []
for case, sub in med.groupby("case"):
    row = {"case": case}
    for _, r in sub.iterrows():
        mode = r["mode"]
        row[f"{mode}_resume_first_ms"] = r["seq0_resume_first_token_ms"]
        row[f"{mode}_total_wall_ms"] = r["total_wall_ms"]
        row[f"{mode}_tokens_per_second"] = r["tokens_per_second"]
        row[f"{mode}_rss_drop_mib"] = r["paged_swap_rss_total_drop_kb"] / 1024.0
        row[f"{mode}_madvise_mib"] = r["paged_swap_madvise_bytes"] / 1024.0 / 1024.0
    base_rf = row.get("base_resume_first_ms")
    nomadv_rf = row.get("nomadv_resume_first_ms")
    madv_rf = row.get("madv_resume_first_ms")
    base_wall = row.get("base_total_wall_ms")
    madv_wall = row.get("madv_total_wall_ms")
    if base_rf is not None and madv_rf is not None:
        row["madv_resume_first_delta_vs_base_ms"] = madv_rf - base_rf
    if nomadv_rf is not None and madv_rf is not None:
        row["madv_resume_first_delta_vs_nomadv_ms"] = madv_rf - nomadv_rf
    if base_wall is not None and madv_wall is not None and base_wall:
        row["madv_total_slowdown_vs_base_pct"] = (madv_wall / base_wall - 1.0) * 100.0
    rows.append(row)

trade = pd.DataFrame(rows)

with open(out_path, "w", encoding="utf-8") as f:
    f.write("===== medians_by_mode =====\n")
    med.to_csv(f, sep="\t", index=False)
    f.write("\n===== tradeoff_summary =====\n")
    trade.to_csv(f, sep="\t", index=False)

print(open(out_path, encoding="utf-8").read())
PY

echo
echo "===== runs.tsv ====="
cat "$RUNS_TSV"

echo
echo "===== median.tsv ====="
cat "$MEDIAN_TSV"

echo
echo "===== git status ====="
git status --short