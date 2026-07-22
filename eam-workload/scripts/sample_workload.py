#!/usr/bin/env python3

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    buckets = defaultdict(list)

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            bucket = row.get("prompt_length_bucket", "unknown")
            buckets[bucket].append(row)

    targets = {
        "short": int(args.total * 0.4),
        "medium": int(args.total * 0.4),
        "long": args.total - int(args.total * 0.4) * 2,
    }

    selected = []

    for bucket, target in targets.items():
        rows = buckets.get(bucket, [])
        random.shuffle(rows)
        take = min(len(rows), target)
        selected.extend(rows[:take])
        print(f"{bucket}: available={len(rows)}, selected={take}")

    if len(selected) < args.total:
        selected_ids = {row["request_id"] for row in selected}
        remaining = [
            row
            for rows in buckets.values()
            for row in rows
            if row["request_id"] not in selected_ids
        ]
        random.shuffle(remaining)
        need = args.total - len(selected)
        selected.extend(remaining[:need])

    random.shuffle(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for index, row in enumerate(selected):
            row["workload_index"] = index
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("written:", len(selected))
    print("output:", output)


if __name__ == "__main__":
    main()
