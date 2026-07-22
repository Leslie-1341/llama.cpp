#!/usr/bin/env python3

import argparse
import json
import math
import struct
from collections import defaultdict


def key(layer, expert):
    return (int(layer) << 32) | int(expert)


def cosine(a, b):
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = 0.0
    for k, v in a.items():
        dot += v * b.get(k, 0.0)
    return dot


def normalize(counts):
    norm = math.sqrt(sum(float(v) * float(v) for v in counts.values()))
    if norm <= 0.0:
        return {}, 0.0
    return {k: float(v) / norm for k, v in counts.items()}, norm


def load_trace(path, min_total, min_count):
    rows = []
    max_layer = 0
    max_expert = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            counts = {}
            rank0 = {}
            total = 0
            for item in row.get("experts", []):
                layer = int(item.get("layer", -1))
                expert = int(item.get("expert", -1))
                count = int(item.get("count", 0))
                if layer < 0 or expert < 0 or count < min_count:
                    continue
                k = key(layer, expert)
                counts[k] = counts.get(k, 0) + count
                rank0[k] = rank0.get(k, 0) + int(item.get("rank0", 0))
                total += count
                max_layer = max(max_layer, layer)
                max_expert = max(max_expert, expert)
            if total < min_total or not counts:
                continue
            vec, norm = normalize(counts)
            rows.append({
                "request_id": row.get("request_id", ""),
                "counts": counts,
                "rank0": rank0,
                "total": total,
                "vec": vec,
                "norm": norm,
            })
    return rows, max_layer + 1, max_expert + 1


def build_clusters(rows, threshold, max_clusters):
    clusters = []
    for row in rows:
        best_i = -1
        best_sim = 0.0
        for i, cluster in enumerate(clusters):
            sim = cosine(row["vec"], cluster["vec"])
            if sim > best_sim:
                best_i = i
                best_sim = sim
        if best_i >= 0 and best_sim >= threshold:
            cluster = clusters[best_i]
            cluster["weight"] += 1
            cluster["total"] += row["total"]
            for k, v in row["counts"].items():
                cluster["counts"][k] += v
            for k, v in row["rank0"].items():
                cluster["rank0"][k] += v
            cluster["vec"], cluster["norm"] = normalize(cluster["counts"])
        else:
            clusters.append({
                "weight": 1,
                "total": row["total"],
                "counts": defaultdict(int, row["counts"]),
                "rank0": defaultdict(int, row["rank0"]),
                "vec": dict(row["vec"]),
                "norm": row["norm"],
                "example": row["request_id"],
            })
    clusters.sort(key=lambda c: (c["weight"], c["total"]), reverse=True)
    return clusters[:max_clusters]


def write_sidecar(path, clusters, n_layer, n_expert, item_topk):
    with open(path, "wb") as f:
        f.write(struct.pack("<8sIIII", b"EAMCV1\0\0", 1, n_layer, n_expert, len(clusters)))
        for i, cluster in enumerate(clusters, start=1):
            items = sorted(cluster["counts"].items(), key=lambda kv: kv[1], reverse=True)
            if item_topk > 0:
                items = items[:item_topk]
            total = sum(v for _, v in items)
            norm = math.sqrt(sum(float(v) * float(v) for _, v in items))
            f.write(struct.pack("<QQQdI", i, int(cluster["weight"]), int(total), float(norm), len(items)))
            for k, count in items:
                layer = (k >> 32) & 0xFFFF
                expert = k & 0xFFFF
                rank0 = int(cluster["rank0"].get(k, 0))
                f.write(struct.pack("<HHII", layer, expert, int(count), rank0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--max-snapshots", type=int, default=64)
    ap.add_argument("--min-total", type=int, default=64)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--item-topk", type=int, default=0)
    args = ap.parse_args()

    rows, n_layer, n_expert = load_trace(args.trace, args.min_total, args.min_count)
    clusters = build_clusters(rows, args.threshold, args.max_snapshots)
    write_sidecar(args.out, clusters, n_layer, n_expert, args.item_topk)
    print("trace rows:", len(rows))
    print("clusters:", len(clusters))
    print("layers:", n_layer)
    print("experts:", n_expert)
    print("sidecar:", args.out)


if __name__ == "__main__":
    main()
