#!/usr/bin/env python3

import json
import re
from pathlib import Path
from datasets import load_dataset

OUT = Path("/root/llama.cpp/eam-workload/processed/sharegpt_user_2k.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

USER_ROLES = {"human", "user", "prompter"}

def clean_text(x):
    x = str(x or "")
    x = x.replace("\x00", " ")
    x = re.sub(r"\r\n?", "\n", x)
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{4,}", "\n\n\n", x)
    return x.strip()

def get_role(m):
    return str(m.get("from") or m.get("role") or m.get("speaker") or "").strip().lower()

def get_text(m):
    return clean_text(m.get("value") or m.get("content") or m.get("text") or "")

def bucket(n):
    if n <= 200:
        return "short"
    if n <= 1000:
        return "medium"
    return "long"

print("Loading ShareGPT dataset from HF cache / hub ...")

ds = load_dataset(
    "RyokoAI/ShareGPT52K",
    split="train",
)

print("rows:", len(ds))
print("columns:", ds.column_names)

written = 0
seen = set()

with OUT.open("w", encoding="utf-8") as f:
    for row_i, row in enumerate(ds):
        conversations = row.get("conversations", [])
        if not isinstance(conversations, list):
            continue

        for turn_i, msg in enumerate(conversations):
            if not isinstance(msg, dict):
                continue

            if get_role(msg) not in USER_ROLES:
                continue

            prompt = get_text(msg)

            if len(prompt) < 20 or len(prompt) > 4000:
                continue

            key = re.sub(r"\s+", " ", prompt).lower()
            if key in seen:
                continue

            seen.add(key)

            rec = {
                "request_id": f"sharegpt_{written:06d}",
                "conversation_index": row_i,
                "turn_index": turn_i,
                "category": "unclassified",
                "language": "unknown",
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "prompt_length_bucket": bucket(len(prompt)),
                "max_new_tokens": 128,
                "temperature": 0.0,
                "source": "RyokoAI/ShareGPT52K",
            }

            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

            # 每个对话只取第一个用户请求
            break

        if written >= 2000:
            break

print("written:", written)
print("output:", OUT)
