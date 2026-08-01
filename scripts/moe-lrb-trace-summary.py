#!/usr/bin/env python3

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict


def parse_float(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # C++ writes UINT64_MAX for unknown future/gap fields. Treat large sentinels
    # as missing so simple summaries do not get dominated by "unknown".
    if math.isfinite(v) and abs(v) < 1.0e18:
        return v
    return None


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values):
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
    }


def analyze(path, topn):
    counts = Counter()
    feature_deltas = defaultdict(lambda: defaultdict(list))
    numeric_suffixes = (
        "last_touch_gap",
        "last_evict_gap",
        "next_use_dist",
        "layer_dist",
        "resident_mib",
        "cache_score",
        "reuse_ema",
        "inter_token_gap_ema",
        "reuse_observed",
        "reuse_within_1",
        "reuse_within_4",
        "reuse_within_16",
        "bad_reload_score",
        "bad_reload_effective",
        "bad_reload_age",
        "bad_reload_1",
        "bad_reload_4",
        "bad_reload_16",
        "ghost_reload_risk_ema",
        "ghost_outcomes",
        "admission_feedback_ema",
        "admission_feedback_samples",
        "admission_regret_ema",
        "admission_regret_samples",
        "seq_access",
        "seq_rank0_access",
        "seq_cache_hits",
        "seq_cache_misses",
        "seq_prefetch_hits",
        "seq_prefetch_late",
        "seq_prefetch_unused",
        "seq_future_hints",
        "seq_future_rank0_hints",
        "seq_predicted",
        "seq_pred_enqueued",
        "eamc_prior",
        "cct_conf",
        "next_token_conf",
        "pinned",
        "active",
        "demand_pending",
    )

    with open(path, "r", encoding="utf-8", newline="") as trace:
        for row in csv.DictReader(trace, delimiter="\t"):
            event = row["event"]
            counts[f"event:{event}"] += 1
            if event == "decision":
                continue
            if event == "unfinished":
                continue
            preference = row["preference"]
            counts[f"preference:{preference}"] += 1
            if row["correct"] in {"0", "1"}:
                counts["classified"] += 1
                counts["correct" if row["correct"] == "1" else "wrong"] += 1
            if preference not in {"admit", "bypass"}:
                continue
            for suffix in numeric_suffixes:
                incoming = parse_float(row.get(f"incoming_{suffix}"))
                victim = parse_float(row.get(f"victim_{suffix}"))
                if incoming is None or victim is None:
                    continue
                # Positive means incoming has a larger feature value than victim.
                feature_deltas[suffix][preference].append(incoming - victim)

    classified = counts["classified"]
    features = []
    for suffix, by_pref in feature_deltas.items():
        admit = by_pref.get("admit", [])
        bypass = by_pref.get("bypass", [])
        if not admit or not bypass:
            continue
        admit_mean = statistics.fmean(admit)
        bypass_mean = statistics.fmean(bypass)
        features.append({
            "feature": suffix,
            "admit_delta_mean": admit_mean,
            "bypass_delta_mean": bypass_mean,
            "separation": abs(admit_mean - bypass_mean),
            "admit": summarize(admit),
            "bypass": summarize(bypass),
        })
    features.sort(key=lambda x: x["separation"], reverse=True)

    return {
        "trace": path,
        "counts": dict(counts),
        "accuracy": counts["correct"] / classified if classified else 0.0,
        "top_features": features[:topn],
    }


def print_report(result):
    counts = result["counts"]
    print(f"trace: {result['trace']}")
    print(
        "events: "
        f"decision={counts.get('event:decision', 0)} "
        f"resolve={counts.get('event:resolve', 0)} "
        f"expire={counts.get('event:expire', 0)} "
        f"unfinished={counts.get('event:unfinished', 0)}"
    )
    print(
        "preference: "
        f"admit={counts.get('preference:admit', 0)} "
        f"bypass={counts.get('preference:bypass', 0)} "
        f"tie={counts.get('preference:tie', 0)}"
    )
    print(
        "current_action: "
        f"classified={counts.get('classified', 0)} "
        f"correct={counts.get('correct', 0)} "
        f"wrong={counts.get('wrong', 0)} "
        f"accuracy={result['accuracy']:.3%}"
    )
    print("top_feature_delta_separation:")
    for item in result["top_features"]:
        print(
            f"  {item['feature']}: sep={item['separation']:.6f} "
            f"admit_mean={item['admit_delta_mean']:.6f} "
            f"bypass_mean={item['bypass_delta_mean']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize LLAMA_LAZY_MOE_LRB_TRACE TSV files.")
    parser.add_argument("trace")
    parser.add_argument("--topn", type=int, default=12)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = analyze(args.trace, args.topn)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
