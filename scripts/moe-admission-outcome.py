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


def summarize_values(values):
    if not values:
        return {"count": 0, "mean": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
    }


def load_trace(path):
    decisions = {}
    outcomes = {}
    duplicate_decisions = 0
    duplicate_outcomes = 0

    with open(path, "r", encoding="utf-8", newline="") as trace:
        for row in csv.DictReader(trace, delimiter="\t"):
            key = (
                int(row["layer"]),
                int(row["expert"]),
                int(row["generation"]),
            )
            event = row["event"]
            if event == "decision":
                if key in decisions:
                    duplicate_decisions += 1
                decisions[key] = row
            else:
                if key in outcomes:
                    duplicate_outcomes += 1
                outcomes[key] = row

    return decisions, outcomes, duplicate_decisions, duplicate_outcomes


def analyze(path):
    decisions, outcomes, duplicate_decisions, duplicate_outcomes = load_trace(path)
    action_counts = Counter()
    outcome_counts = Counter()
    deltas = defaultdict(list)
    gaps = defaultdict(list)
    unmatched_outcomes = 0

    for key, decision in decisions.items():
        action = decision["action"]
        action_counts[action] += 1
        outcome = outcomes.get(key)
        candidate = float(decision["candidate_value"])
        victim = float(decision["victim_value"])
        delta = candidate - victim

        if outcome is None:
            label = f"{action}_missing_outcome"
            outcome_counts[label] += 1
            deltas[label].append(delta)
            continue

        event = outcome["event"]
        gap = int(outcome["gap"])
        reason = outcome["reason"]
        if action == "bypass":
            if event == "reload":
                label = "bypass_bad_reload"
            elif event == "expire":
                label = "bypass_good_expire"
            else:
                label = "bypass_unfinished"
        elif action == "admit":
            if event == "reuse":
                label = "admit_good_reuse"
            elif event in ("evict_unused", "expire"):
                label = "admit_bad_unused"
            else:
                label = "admit_unfinished"
        else:
            label = f"unknown_{event}"

        outcome_counts[label] += 1
        outcome_counts[f"event:{event}"] += 1
        outcome_counts[f"reason:{reason}"] += 1
        deltas[label].append(delta)
        gaps[label].append(gap)

        if gap <= 1:
            outcome_counts[f"{label}:gap<=1"] += 1
        if gap <= 4:
            outcome_counts[f"{label}:gap<=4"] += 1
        if gap <= 16:
            outcome_counts[f"{label}:gap<=16"] += 1

    for key in outcomes:
        if key not in decisions:
            unmatched_outcomes += 1

    bypass_total = action_counts["bypass"]
    admit_total = action_counts["admit"]
    bypass_bad = outcome_counts["bypass_bad_reload"]
    bypass_good = outcome_counts["bypass_good_expire"]
    admit_good = outcome_counts["admit_good_reuse"]
    admit_bad = outcome_counts["admit_bad_unused"]

    return {
        "trace": path,
        "decisions": len(decisions),
        "actions": dict(action_counts),
        "outcomes": dict(outcome_counts),
        "rates": {
            "false_bypass": bypass_bad / max(1, bypass_bad + bypass_good),
            "false_admit": admit_bad / max(1, admit_good + admit_bad),
            "resolved_bypass_fraction": (bypass_bad + bypass_good) / max(1, bypass_total),
            "resolved_admit_fraction": (admit_good + admit_bad) / max(1, admit_total),
        },
        "delta": {key: summarize_values(value) for key, value in deltas.items()},
        "gap": {key: summarize_values(value) for key, value in gaps.items()},
        "integrity": {
            "duplicate_decisions": duplicate_decisions,
            "duplicate_outcomes": duplicate_outcomes,
            "unmatched_outcomes": unmatched_outcomes,
        },
    }


def print_report(result):
    actions = result["actions"]
    outcomes = result["outcomes"]
    rates = result["rates"]
    print(f"trace: {result['trace']}")
    print(f"decisions: {result['decisions']}")
    print(
        "actions: "
        f"admit={actions.get('admit', 0)} "
        f"bypass={actions.get('bypass', 0)}"
    )
    print(
        "bypass: "
        f"bad_reload={outcomes.get('bypass_bad_reload', 0)} "
        f"good_expire={outcomes.get('bypass_good_expire', 0)} "
        f"false_rate={rates['false_bypass']:.3%} "
        f"resolved={rates['resolved_bypass_fraction']:.3%}"
    )
    print(
        "admit: "
        f"good_reuse={outcomes.get('admit_good_reuse', 0)} "
        f"bad_unused={outcomes.get('admit_bad_unused', 0)} "
        f"false_rate={rates['false_admit']:.3%} "
        f"resolved={rates['resolved_admit_fraction']:.3%}"
    )

    print("candidate_minus_victim:")
    for label in (
        "bypass_bad_reload",
        "bypass_good_expire",
        "admit_good_reuse",
        "admit_bad_unused",
        "bypass_unfinished",
        "admit_unfinished",
    ):
        item = result["delta"].get(label)
        if not item:
            continue
        print(
            f"  {label}: n={item['count']} mean={item['mean']:.3f} "
            f"p25={item['p25']:.3f} p50={item['p50']:.3f} p75={item['p75']:.3f}"
        )

    integrity = result["integrity"]
    print(
        "integrity: "
        f"duplicate_decisions={integrity['duplicate_decisions']} "
        f"duplicate_outcomes={integrity['duplicate_outcomes']} "
        f"unmatched_outcomes={integrity['unmatched_outcomes']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze demand admission decisions and their short-window outcomes."
    )
    parser.add_argument("trace", help="TSV produced by LLAMA_LAZY_MOE_ADMISSION_TRACE")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()

    result = analyze(args.trace)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
