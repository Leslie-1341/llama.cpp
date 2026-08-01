#!/usr/bin/env python3

import argparse
import csv
import math
import random
import statistics
from collections import Counter


DEFAULT_SUFFIXES = [
    "next_use_dist",
    "seq_access",
    "seq_cache_hits",
    "cache_score",
    "bad_reload_age",
    "seq_future_hints",
    "seq_rank0_access",
    "last_touch_gap",
    "seq_future_rank0_hints",
    "seq_prefetch_hits",
    "seq_cache_misses",
    "bad_reload_score",
    "admission_feedback_samples",
    "bad_reload_effective",
    "admission_regret_samples",
    "resident_mib",
    "admission_feedback_ema",
    "last_evict_gap",
    "admission_regret_ema",
    "bad_reload_1",
    "reuse_ema",
    "reuse_observed",
    "reuse_within_4",
    "inter_token_gap_ema",
    "eamc_prior",
    "cct_conf",
    "next_token_conf",
    "pinned",
    "active",
    "demand_pending",
]


def parse_number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    if abs(v) >= 1.0e18:
        return None
    return v


def sigmoid_margin(z):
    # Return log(1 + exp(-z))'s derivative in a stable form for y*z.
    if z >= 40.0:
        return 0.0
    if z <= -40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def infer_suffixes(path):
    with open(path, "r", encoding="utf-8", newline="") as trace:
        reader = csv.reader(trace, delimiter="\t")
        header = next(reader)
    incoming = {name[len("incoming_"):] for name in header if name.startswith("incoming_")}
    victim = {name[len("victim_"):] for name in header if name.startswith("victim_")}
    # Exclude future labels and object identity. The goal is a reusable score
    # rule, not memorizing this trace's layer/expert or reading its outcome gap.
    excluded = {"gap", "layer", "expert"}
    return sorted((incoming & victim) - excluded)


def load_samples(path, suffixes):
    raw = []
    counts = Counter()
    with open(path, "r", encoding="utf-8", newline="") as trace:
        reader = csv.DictReader(trace, delimiter="\t")
        for row in reader:
            event = row["event"]
            counts[f"event:{event}"] += 1
            if event == "decision" or event == "unfinished":
                continue
            preference = row["preference"]
            counts[f"preference:{preference}"] += 1
            if preference not in {"admit", "bypass"}:
                continue
            y = 1 if preference == "admit" else -1
            action = row["action"]
            current_correct = (
                action == "admit" and y > 0) or (action == "bypass" and y < 0)
            counts["classified"] += 1
            counts["current_correct" if current_correct else "current_wrong"] += 1
            values = []
            for suffix in suffixes:
                incoming = parse_number(row.get(f"incoming_{suffix}"))
                victim = parse_number(row.get(f"victim_{suffix}"))
                if incoming is None or victim is None:
                    values.append(None)
                else:
                    values.append(incoming - victim)
                # Positive means incoming is missing while victim is present.
                values.append((1.0 if incoming is None else 0.0) -
                              (1.0 if victim is None else 0.0))
            raw.append({
                "pair_id": int(row["pair_id"]),
                "y": y,
                "values": values,
            })
    feature_names = []
    for suffix in suffixes:
        feature_names.append(f"delta:{suffix}")
        feature_names.append(f"missing_delta:{suffix}")
    return raw, feature_names, counts


def split_samples(samples, test_fraction, seed):
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction))) if shuffled else 0
    return shuffled[n_test:], shuffled[:n_test]


def prepare_matrix(train, test, feature_names):
    n_features = len(feature_names)
    medians = []
    means = []
    stds = []

    for j in range(n_features):
        values = [s["values"][j] for s in train if s["values"][j] is not None]
        med = statistics.median(values) if values else 0.0
        medians.append(med)
        filled = [s["values"][j] if s["values"][j] is not None else med for s in train]
        mean = statistics.fmean(filled) if filled else 0.0
        var = statistics.fmean([(v - mean) * (v - mean) for v in filled]) if filled else 0.0
        std = math.sqrt(var) if var > 1.0e-24 else 1.0
        means.append(mean)
        stds.append(std)

    def convert(rows):
        x = []
        y = []
        for s in rows:
            vec = []
            for j, value in enumerate(s["values"]):
                v = medians[j] if value is None else value
                vec.append((v - means[j]) / stds[j])
            x.append(vec)
            y.append(s["y"])
        return x, y

    return convert(train), convert(test), medians, means, stds


def train_logistic(x, y, epochs, lr, l2):
    if not x:
        return [], 0.0
    n = len(x)
    d = len(x[0])
    w = [0.0] * d
    b = 0.0
    for epoch in range(epochs):
        rate = lr / math.sqrt(1.0 + epoch * 0.05)
        grad_w = [l2 * wi for wi in w]
        grad_b = 0.0
        for xi, yi in zip(x, y):
            z = yi * (sum(wj * xij for wj, xij in zip(w, xi)) + b)
            g = -yi * sigmoid_margin(z)
            grad_b += g
            for j, xij in enumerate(xi):
                grad_w[j] += g * xij
        inv_n = 1.0 / n
        b -= rate * grad_b * inv_n
        for j in range(d):
            w[j] -= rate * grad_w[j] * inv_n
    return w, b


def evaluate(x, y, w, b):
    counts = Counter()
    margins = []
    for xi, yi in zip(x, y):
        score = sum(wj * xij for wj, xij in zip(w, xi)) + b
        pred = 1 if score >= 0.0 else -1
        counts["correct" if pred == yi else "wrong"] += 1
        margins.append(yi * score)
    total = counts["correct"] + counts["wrong"]
    return {
        "n": total,
        "correct": counts["correct"],
        "wrong": counts["wrong"],
        "accuracy": counts["correct"] / total if total else 0.0,
        "mean_margin": statistics.fmean(margins) if margins else 0.0,
    }


def print_report(args, counts, train_eval, test_eval, feature_names, w, b):
    classified = counts["classified"]
    current_acc = counts["current_correct"] / classified if classified else 0.0
    print(f"trace: {args.trace}")
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
        f"classified={classified} "
        f"correct={counts.get('current_correct', 0)} "
        f"wrong={counts.get('current_wrong', 0)} "
        f"accuracy={current_acc:.3%}"
    )
    print(
        "linear_model: "
        f"train_n={train_eval['n']} train_acc={train_eval['accuracy']:.3%} "
        f"test_n={test_eval['n']} test_acc={test_eval['accuracy']:.3%} "
        f"test_correct={test_eval['correct']} test_wrong={test_eval['wrong']} "
        f"bias={b:.6f}"
    )
    print("top_weights:")
    ranked = sorted(zip(feature_names, w), key=lambda item: abs(item[1]), reverse=True)
    for name, weight in ranked[:args.topn]:
        print(f"  {name}: {weight:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit a small pairwise linear model from LLAMA_LAZY_MOE_LRB_TRACE."
    )
    parser.add_argument("trace")
    parser.add_argument(
        "--features",
        default=",".join(DEFAULT_SUFFIXES),
        help="comma-separated feature suffixes, or 'all' for every incoming/victim column",
    )
    parser.add_argument("--test-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--topn", type=int, default=24)
    args = parser.parse_args()

    suffixes = infer_suffixes(args.trace) if args.features == "all" else [
        s.strip() for s in args.features.split(",") if s.strip()
    ]
    samples, feature_names, counts = load_samples(args.trace, suffixes)
    if len(samples) < 4:
        raise SystemExit("not enough classified samples")

    train, test = split_samples(samples, args.test_fraction, args.seed)
    (x_train, y_train), (x_test, y_test), _, _, _ = prepare_matrix(
        train, test, feature_names)
    w, b = train_logistic(x_train, y_train, args.epochs, args.lr, args.l2)
    train_eval = evaluate(x_train, y_train, w, b)
    test_eval = evaluate(x_test, y_test, w, b)
    print_report(args, counts, train_eval, test_eval, feature_names, w, b)


if __name__ == "__main__":
    main()
