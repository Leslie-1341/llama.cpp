#!/usr/bin/env python3

import json
import sys
from collections import Counter

if len(sys.argv) != 2:
    print("usage: check_workload.py <jsonl>")
    sys.exit(1)

path = sys.argv[1]

length_buckets = Counter()
languages = Counter()
sources = Counter()
char_lengths = []
records = 0

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        row = json.loads(line)
        records += 1

        text = row.get("prompt", "")
        char_lengths.append(len(text))

        length_buckets[row.get("prompt_length_bucket", "unknown")] += 1
        languages[row.get("language", "unknown")] += 1
        sources[row.get("source", "unknown")] += 1

print("records:", records)

if char_lengths:
    char_lengths.sort()
    print("min chars:", char_lengths[0])
    print("median chars:", char_lengths[len(char_lengths) // 2])
    print("max chars:", char_lengths[-1])

print("\nlength buckets:")
for key, value in length_buckets.most_common():
    print(f"  {key}: {value}")

print("\nlanguages:")
for key, value in languages.most_common(20):
    print(f"  {key}: {value}")

print("\nsources:")
for key, value in sources.most_common():
    print(f"  {key}: {value}")
