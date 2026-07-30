#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def value_summary(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
    }


def analyze(path):
    counts = Counter()
    targets = defaultdict(list)
    seen_decisions = set()
    seen_terminal = set()

    with open(path, "r", encoding="utf-8", newline="") as trace:
        for row in csv.DictReader(trace, delimiter="\t"):
            pair_id = int(row["pair_id"])
            event = row["event"]
            action = row["action"]
            if event == "decision":
                seen_decisions.add(pair_id)
                counts[f"decision:{action}"] += 1
                continue

            seen_terminal.add(pair_id)
            counts[f"event:{event}"] += 1
            if event == "unfinished":
                counts[f"unfinished:{action}"] += 1
                continue

            target = float(row["regret_target"])
            incoming_gap = int(row["incoming_gap"])
            victim_gap = int(row["victim_gap"])
            preference = "admit" if target > 0 else ("bypass" if target < 0 else "tie")
            counts[f"prefer:{preference}"] += 1
            counts[f"action:{action}:prefer:{preference}"] += 1
            targets[f"{action}:{preference}"].append(target)

            if target != 0:
                correct = action == preference
                counts["classified"] += 1
                counts["correct" if correct else "wrong"] += 1
                counts[f"{action}:{'correct' if correct else 'wrong'}"] += 1
            if incoming_gap <= 1:
                counts["incoming_gap<=1"] += 1
            if incoming_gap <= 4:
                counts["incoming_gap<=4"] += 1
            if victim_gap <= 1:
                counts["victim_gap<=1"] += 1
            if victim_gap <= 4:
                counts["victim_gap<=4"] += 1

    classified = counts["classified"]
    evaluated = counts["event:resolve"] + counts["event:expire"]
    return {
        "trace": path,
        "pairs": len(seen_decisions),
        "terminal": len(seen_terminal),
        "terminal_fraction": len(seen_terminal) / max(1, len(seen_decisions)),
        "evaluated": evaluated,
        "evaluated_fraction": evaluated / max(1, len(seen_decisions)),
        "classified_accuracy": counts["correct"] / max(1, classified),
        "counts": dict(counts),
        "target": {key: value_summary(values) for key, values in targets.items()},
        "integrity": {
            "terminal_without_decision": len(seen_terminal - seen_decisions),
            "decision_without_terminal": len(seen_decisions - seen_terminal),
        },
    }


def print_report(result):
    counts = result["counts"]
    print(f"trace: {result['trace']}")
    print(
        f"pairs={result['pairs']} terminal={result['terminal']} "
        f"terminal_fraction={result['terminal_fraction']:.3%}"
    )
    print(
        f"evaluated={result['evaluated']} "
        f"evaluated_fraction={result['evaluated_fraction']:.3%} "
        f"resolved={counts.get('event:resolve', 0)} "
        f"expired={counts.get('event:expire', 0)} "
        f"unfinished={counts.get('event:unfinished', 0)}"
    )
    print(
        "preference: "
        f"admit={counts.get('prefer:admit', 0)} "
        f"bypass={counts.get('prefer:bypass', 0)} "
        f"tie={counts.get('prefer:tie', 0)}"
    )
    print(
        "current_action: "
        f"classified={counts.get('classified', 0)} "
        f"correct={counts.get('correct', 0)} "
        f"wrong={counts.get('wrong', 0)} "
        f"accuracy={result['classified_accuracy']:.3%}"
    )
    print(
        "admit: "
        f"correct={counts.get('admit:correct', 0)} "
        f"wrong={counts.get('admit:wrong', 0)}"
    )
    print(
        "bypass: "
        f"correct={counts.get('bypass:correct', 0)} "
        f"wrong={counts.get('bypass:wrong', 0)}"
    )
    print(
        "short_gap: "
        f"incoming<=1={counts.get('incoming_gap<=1', 0)} "
        f"incoming<=4={counts.get('incoming_gap<=4', 0)} "
        f"victim<=1={counts.get('victim_gap<=1', 0)} "
        f"victim<=4={counts.get('victim_gap<=4', 0)}"
    )
    integrity = result["integrity"]
    print(
        "integrity: "
        f"terminal_without_decision={integrity['terminal_without_decision']} "
        f"decision_without_terminal={integrity['decision_without_terminal']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze paired incoming-victim admission regret traces."
    )
    parser.add_argument(
        "trace", help="TSV produced by LLAMA_LAZY_MOE_ADMISSION_REGRET_TRACE"
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()

    result = analyze(args.trace)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
