#!/usr/bin/env python3
import argparse
import bisect
import collections
import csv
import heapq
import os


def as_int(row, name, default=0):
    try:
        return int(row.get(name, default))
    except (TypeError, ValueError):
        return default


def as_bytes(row):
    for name in ("exact_group_resident_bytes", "bytes"):
        if name in row:
            return as_int(row, name, 0)
    return 0


def is_demand_action(action):
    return action.startswith(("cache_hit", "compat_hit", "prefetch_hit", "miss_load"))


def load_demands(cache_path):
    rows = []
    demands = []
    demand_tokens = collections.defaultdict(list)
    loads = []
    with open(cache_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if "layer" not in row or "expert" not in row:
                continue
            token = as_int(row, "token")
            layer = as_int(row, "layer", -1)
            expert = as_int(row, "expert", -1)
            if layer < 0 or expert < 0:
                continue
            action = row.get("action", "")
            if not is_demand_action(action):
                continue
            key = (layer, expert)
            rows.append((token, key, row))
            demands.append(key)
            if not demand_tokens[key] or demand_tokens[key][-1] != token:
                demand_tokens[key].append(token)
            if action.startswith("miss_load"):
                kind = row.get("kind", "")
                loads.append((token, key, kind, row))
    return rows, demands, demand_tokens, loads


def simulate_belady(demands, capacity):
    if capacity < 1:
        raise ValueError("capacity must be at least one group")

    n = len(demands)
    never = n + 1
    next_use = [never] * n
    next_position = {}
    for i in range(n - 1, -1, -1):
        key = demands[i]
        next_use[i] = next_position.get(key, never)
        next_position[key] = i

    resident = set()
    resident_next = {}
    future_heap = []
    seen = set()
    hits = 0
    misses = 0
    compulsory_misses = 0
    evictions = 0

    for i, key in enumerate(demands):
        key_next = next_use[i]
        if key in resident:
            hits += 1
        else:
            misses += 1
            if key not in seen:
                compulsory_misses += 1
            if len(resident) >= capacity:
                while future_heap:
                    neg_next, victim = heapq.heappop(future_heap)
                    victim_next = -neg_next
                    if victim in resident and resident_next.get(victim) == victim_next:
                        resident.remove(victim)
                        resident_next.pop(victim, None)
                        evictions += 1
                        break
            resident.add(key)

        seen.add(key)
        resident_next[key] = key_next
        heapq.heappush(future_heap, (-key_next, key))

    return {
        "capacity_groups": capacity,
        "demand_references": n,
        "unique_groups": len(seen),
        "hits": hits,
        "misses": misses,
        "compulsory_misses": compulsory_misses,
        "capacity_misses": misses - compulsory_misses,
        "evictions": evictions,
    }


def summarize_oracle(demand_tokens, loads):
    group_loads = []
    seen_token_group = set()
    for token, key, kind, row in loads:
        tg = (token, key)
        if tg in seen_token_group:
            continue
        seen_token_group.add(tg)
        group_loads.append((token, key))

    out = {
        "slice_miss_loads": len(loads),
        "group_loads": len(group_loads),
        "unique_groups": len(demand_tokens),
    }
    for h in (1, 2, 4, 8, 16):
        avoid = 0
        for token, key in group_loads:
            prevs = demand_tokens[key]
            i = bisect.bisect_left(prevs, token)
            if i > 0 and 0 < token - prevs[i - 1] <= h:
                avoid += 1
        out[f"avoidable_h{h}"] = avoid
        out[f"oracle_loads_h{h}"] = len(group_loads) - avoid
    return out


def summarize_resident(resident_path, demand_tokens):
    if not resident_path or not os.path.exists(resident_path):
        return None
    snapshots = collections.defaultdict(lambda: collections.Counter())
    with open(resident_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            token = as_int(row, "token")
            current_layer = as_int(row, "current_layer", -1)
            layer = as_int(row, "resident_layer", -1)
            expert = as_int(row, "expert", -1)
            b = as_bytes(row)
            if current_layer < 0 or layer < 0 or expert < 0 or b <= 0:
                continue
            key = (layer, expert)
            future = demand_tokens.get(key, [])
            i = bisect.bisect_left(future, token)
            gap = None if i >= len(future) else future[i] - token
            snap = snapshots[(token, current_layer)]
            snap["bytes_total"] += b
            snap["groups_total"] += 1
            if layer == current_layer:
                snap["bytes_h0"] += b
                snap["groups_h0"] += 1
            if gap == 0:
                snap["bytes_future0"] += b
                snap["groups_future0"] += 1
            elif gap == 1:
                snap["bytes_h1"] += b
                snap["groups_h1"] += 1
            elif gap is not None and gap <= 4:
                snap["bytes_h4"] += b
                snap["groups_h4"] += 1
            elif gap is None:
                snap["bytes_never"] += b
                snap["groups_never"] += 1
            else:
                snap["bytes_far"] += b
                snap["groups_far"] += 1

    if not snapshots:
        return None
    n = len(snapshots)
    totals = collections.Counter()
    for snap in snapshots.values():
        totals.update(snap)
    return {k: v / n for k, v in totals.items()} | {"snapshots": n}


def print_mib(label, value):
    print(f"{label}: {value / 1048576.0:.1f} MiB")


def print_belady(belady, actual_group_loads):
    misses = belady["misses"]
    references = belady["demand_references"]
    gap = max(0, actual_group_loads - misses)
    reduction = 100.0 * gap / actual_group_loads if actual_group_loads else 0.0
    opt_miss_rate = 100.0 * misses / references if references else 0.0
    actual_miss_rate = 100.0 * actual_group_loads / references if references else 0.0

    print("\n== fixed-capacity Belady OPT ==")
    for key in (
        "capacity_groups",
        "demand_references",
        "unique_groups",
        "hits",
        "misses",
        "compulsory_misses",
        "capacity_misses",
        "evictions",
    ):
        print(f"opt_{key}: {belady[key]}")
    print(f"actual_group_loads: {actual_group_loads}")
    print(f"opt_miss_rate_pct: {opt_miss_rate:.2f}")
    print(f"actual_miss_rate_pct: {actual_miss_rate:.2f}")
    print(f"actual_minus_opt_loads: {actual_group_loads - misses}")
    print(f"avoidable_vs_actual_loads: {gap}")
    print(f"avoidable_vs_actual_pct: {reduction:.2f}")


def main():
    ap = argparse.ArgumentParser(
        description="Replay MoE cache/resident traces and estimate short-horizon and fixed-capacity Belady bounds."
    )
    ap.add_argument("--cache", required=True, help="LLAMA_LAZY_MOE_CACHE_TRACE tsv")
    ap.add_argument("--resident", help="LLAMA_LAZY_MOE_RESIDENT_TRACE tsv")
    ap.add_argument(
        "--capacity-groups",
        type=int,
        default=164,
        help="ExpertGroup slots for the fixed-capacity Belady simulation (default: 164)",
    )
    args = ap.parse_args()

    _, demands, demand_tokens, loads = load_demands(args.cache)
    oracle = summarize_oracle(demand_tokens, loads)
    print("== demand / miss oracle ==")
    for k in ("unique_groups", "slice_miss_loads", "group_loads"):
        print(f"{k}: {oracle[k]}")
    for h in (1, 2, 4, 8, 16):
        print(f"h{h}_avoidable_group_loads: {oracle[f'avoidable_h{h}']}")
        print(f"h{h}_oracle_group_loads: {oracle[f'oracle_loads_h{h}']}")

    belady = simulate_belady(demands, args.capacity_groups)
    print_belady(belady, len(loads))

    resident = summarize_resident(args.resident, demand_tokens)
    if resident is None:
        return
    print("\n== resident oracle classes ==")
    print(f"snapshots: {int(resident['snapshots'])}")
    print(f"avg_groups_total: {resident.get('groups_total', 0.0):.1f}")
    print_mib("avg_bytes_total", resident.get("bytes_total", 0.0))
    for name in ("h0", "future0", "h1", "h4", "far", "never"):
        print(f"avg_groups_{name}: {resident.get('groups_' + name, 0.0):.1f}")
        print_mib(f"avg_bytes_{name}", resident.get("bytes_" + name, 0.0))


if __name__ == "__main__":
    main()
