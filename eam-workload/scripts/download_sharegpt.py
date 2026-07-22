#!/usr/bin/env python3

from pathlib import Path
from datasets import load_dataset

OUT = Path("/root/llama.cpp/eam-workload/raw/sharegpt")
OUT.mkdir(parents=True, exist_ok=True)

print("Downloading RyokoAI/ShareGPT52K ...")

dataset = load_dataset(
    "RyokoAI/ShareGPT52K",
    split="train",
)

print(dataset)
print("rows:", len(dataset))
print("columns:", dataset.column_names)

dataset.save_to_disk(str(OUT))

print(f"Saved to: {OUT}")
