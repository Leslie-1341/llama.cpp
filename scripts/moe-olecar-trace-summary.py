#!/usr/bin/env python3

import argparse
import csv
import statistics
from collections import Counter, defaultdict


def mean(values):
    return statistics.fmean(values) if values else 0.0


def analyze(path, kind_filter=None):
    counts = Counter()
    by_policy = defaultdict(lambda: Counter())
    costs = defaultdict(list)
    selected_costs = defaultdict(list)
    policy_probs = defaultdict(list)
    support_probs = defaultdict(list)
    last_weight = {}
    last_update = {}

    with open(path, "r", encoding="utf-8", newline="") as trace:
        for row in csv.DictReader(trace, delimiter="\t"):
            kind = row.get("kind") or "legacy"
            if kind_filter is not None and kind != kind_filter:
                continue
            event = row["event"]
            policy = row["policy"]
            selected = row["selected"] == "1"
            counts[f"event:{event}"] += 1
            counts[f"kind:{kind}"] += 1
            by_policy[policy][f"event:{event}"] += 1
            if "policy_prob" in row and row["policy_prob"]:
                policy_probs[policy].append(float(row["policy_prob"]))
            if "support_prob" in row and row["support_prob"]:
                support_probs[policy].append(float(row["support_prob"]))
            if "weight_after" in row and row["weight_after"]:
                update = int(row.get("update_count") or 0)
                if update >= last_update.get(policy, -1):
                    last_update[policy] = update
                    last_weight[policy] = float(row["weight_after"])
            if event == "decision" and selected:
                by_policy[policy]["selected"] += 1
            if event not in {"resolve", "expire"}:
                continue
            cost = float(row["cost"])
            costs[policy].append(cost)
            if selected:
                selected_costs[policy].append(cost)
            if cost > 0:
                by_policy[policy]["cost_positive"] += 1
            if cost >= 1.0:
                by_policy[policy]["cost_1tok"] += 1
            elif cost >= 0.5:
                by_policy[policy]["cost_4tok"] += 1
            elif cost > 0:
                by_policy[policy]["cost_16tok"] += 1

    return counts, by_policy, costs, selected_costs, policy_probs, support_probs, last_weight


def print_report(path, title, counts, by_policy, costs, selected_costs, policy_probs, support_probs, last_weight):
    print(f"{title}: {path}")
    print(
        "events: "
        f"decision={counts.get('event:decision', 0)} "
        f"resolve={counts.get('event:resolve', 0)} "
        f"expire={counts.get('event:expire', 0)} "
        f"unfinished={counts.get('event:unfinished', 0)} "
        f"counterfactual={counts.get('kind:counterfactual', 0)} "
        f"exp4={counts.get('kind:exp4', 0)} "
        f"legacy={counts.get('kind:legacy', 0)}"
    )
    print("policy_summary:")
    rows = []
    for policy, c in by_policy.items():
        terminal = c.get("event:resolve", 0) + c.get("event:expire", 0)
        rows.append((
            mean(costs[policy]),
            policy,
            c,
            terminal,
            mean(selected_costs[policy]),
        ))
    for avg_cost, policy, c, terminal, selected_avg in sorted(rows):
        print(
            f"  {policy}: terminal={terminal} "
            f"decision={c.get('event:decision', 0)} "
            f"selected={c.get('selected', 0)} "
            f"avg_cost={avg_cost:.6f} "
            f"selected_avg_cost={selected_avg:.6f} "
            f"avg_policy_prob={mean(policy_probs[policy]):.6f} "
            f"avg_support_prob={mean(support_probs[policy]):.6f} "
            f"last_weight={last_weight.get(policy, 0.0):.6f} "
            f"positive={c.get('cost_positive', 0)} "
            f"cost1={c.get('cost_1tok', 0)} "
            f"cost4={c.get('cost_4tok', 0)} "
            f"cost16={c.get('cost_16tok', 0)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize LLAMA_LAZY_MOE_OLECAR_TRACE.")
    parser.add_argument("trace")
    parser.add_argument(
        "--kind",
        choices=["all", "counterfactual", "exp4", "legacy"],
        default="all",
        help="Summarize all rows or only one trace kind.",
    )
    args = parser.parse_args()
    if args.kind == "all":
        for title, kind in [
            ("trace", None),
            ("counterfactual_trace", "counterfactual"),
            ("exp4_trace", "exp4"),
        ]:
            counts, by_policy, costs, selected_costs, policy_probs, support_probs, last_weight = analyze(args.trace, kind)
            print_report(args.trace, title, counts, by_policy, costs, selected_costs, policy_probs, support_probs, last_weight)
    else:
        counts, by_policy, costs, selected_costs, policy_probs, support_probs, last_weight = analyze(args.trace, args.kind)
        print_report(args.trace, f"{args.kind}_trace", counts, by_policy, costs, selected_costs, policy_probs, support_probs, last_weight)


if __name__ == "__main__":
    main()
