#!/usr/bin/env python3
"""Convert ShareGPT-like conversations into kv-trace-replay traces."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Any


COUNT_KEYS = [
    "records_seen",
    "records_used",
    "records_dropped",
    "dropped_empty_conversation",
    "dropped_no_pairs",
    "dropped_bad_structure",
    "dropped_missing_text",
    "dropped_empty_text",
    "dropped_system_messages",
    "dropped_trailing_user",
    "truncated_user_prompts",
]


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def clean_text(value: Any) -> str:
    text = str(value)
    text = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    cleaned = []
    for ch in text:
        if ch in ("\n", "\t"):
            cleaned.append(ch)
            continue
        if unicodedata.category(ch)[0] == "C":
            continue
        cleaned.append(ch)

    return "".join(cleaned).strip()


def normalize_role(value: Any) -> str | None:
    role = str(value).strip().lower()
    if role in ("human", "user"):
        return "user"
    if role in ("gpt", "assistant"):
        return "assistant"
    if role == "system":
        return "system"
    return None


def extract_records(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        data = obj.get("data")
        if isinstance(data, list):
            return data
        if isinstance(obj.get("conversations"), list) or isinstance(obj.get("messages"), list):
            return [obj]
    fail("JSON input must be a list, a {'data': [...]} object, or one conversation object")


def load_input(path: Path) -> list[Any]:
    if not path.exists():
        fail(f"input does not exist: {path}")
    if not path.is_file():
        fail(f"input is not a file: {path}")

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        fail(f"failed to read input: {exc}")

    try:
        return extract_records(json.loads(text))
    except json.JSONDecodeError:
        pass

    records = []
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            fail(f"input is not valid JSON or JSONL at line {line_no}: {exc}")

    if not records:
        fail("input is not valid JSON or JSONL")
    return records


def message_list(record: Any) -> list[Any] | None:
    if not isinstance(record, dict):
        return None
    messages = record.get("conversations")
    if messages is None:
        messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    return messages


def text_field(message: dict[str, Any]) -> tuple[bool, Any]:
    if "value" in message:
        return True, message["value"]
    if "content" in message:
        return True, message["content"]
    return False, None


def clean_conversation(
    record: Any,
    counts: dict[str, int],
    max_user_chars: int,
) -> list[tuple[str, str]] | None:
    messages = message_list(record)
    if not messages:
        counts["dropped_empty_conversation"] += 1
        return None

    normalized: list[tuple[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            counts["dropped_bad_structure"] += 1
            return None

        raw_role = message.get("from", message.get("role"))
        if raw_role is None:
            counts["dropped_bad_structure"] += 1
            return None

        role = normalize_role(raw_role)
        if role is None:
            counts["dropped_bad_structure"] += 1
            return None

        if role == "system":
            counts["dropped_system_messages"] += 1
            continue

        has_text, raw_text = text_field(message)
        if not has_text or raw_text is None:
            counts["dropped_missing_text"] += 1
            return None

        normalized.append((role, clean_text(raw_text)))

    if not normalized:
        counts["dropped_no_pairs"] += 1
        return None

    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(normalized):
        role, user_text = normalized[i]
        if role != "user":
            counts["dropped_bad_structure"] += 1
            return None

        if i + 1 >= len(normalized):
            counts["dropped_trailing_user"] += 1
            break

        next_role, assistant_text = normalized[i + 1]
        if next_role != "assistant":
            counts["dropped_bad_structure"] += 1
            return None

        if not user_text or not assistant_text:
            counts["dropped_empty_text"] += 1
            i += 2
            continue

        if len(user_text) > max_user_chars:
            user_text = user_text[:max_user_chars].strip()
            counts["truncated_user_prompts"] += 1
            if not user_text:
                counts["dropped_empty_text"] += 1
                i += 2
                continue

        pairs.append((user_text, assistant_text))
        i += 2

    if not pairs:
        counts["dropped_no_pairs"] += 1
        return None

    return pairs


def estimate_decode_tokens(
    assistant_text: str,
    chars_per_token: float,
    min_decode_tokens: int,
    max_decode_tokens: int,
) -> int:
    estimated = int(math.ceil(len(assistant_text) / chars_per_token))
    return max(min_decode_tokens, min(max_decode_tokens, estimated))


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            fail(f"output-dir exists and is not empty: {output_dir}")
        try:
            shutil.rmtree(output_dir)
        except OSError as exc:
            fail(f"failed to clear output-dir: {exc}")

    try:
        (output_dir / "prompts").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fail(f"failed to create output directories: {exc}")


def write_outputs(
    args: argparse.Namespace,
    conversations: list[list[tuple[str, str]]],
    counts: dict[str, int],
) -> None:
    output_dir = Path(args.output_dir)
    prompt_dir = output_dir / "prompts"
    trace_file = output_dir / "trace.tsv"
    summary_file = output_dir / "summary.json"

    prepare_output_dir(output_dir, args.overwrite)

    rng = random.Random(args.seed)
    trace_rows: list[tuple[int, int, int, str, int, int]] = []
    prompt_files = 0

    try:
        for session_id, pairs in enumerate(conversations):
            kept_pairs = pairs[: args.max_turns_per_session]
            arrival_ms = session_id * args.start_gap_ms
            if args.jitter_ms > 0:
                arrival_ms += rng.randint(0, args.jitter_ms)

            for turn_id, (user_text, assistant_text) in enumerate(kept_pairs):
                name = f"s{session_id:06d}_t{turn_id:06d}.txt"
                rel_prompt = f"prompts/{name}"
                prompt_path = prompt_dir / name
                prompt_path.write_text(user_text + "\n", encoding="utf-8", errors="replace")
                prompt_files += 1

                if turn_id + 1 < len(kept_pairs):
                    idle_ms = rng.randint(args.idle_ms_low, args.idle_ms_high)
                else:
                    idle_ms = 0

                target_decode_tokens = estimate_decode_tokens(
                    assistant_text,
                    args.chars_per_token,
                    args.min_decode_tokens,
                    args.max_decode_tokens,
                )
                trace_rows.append((
                    session_id,
                    turn_id,
                    arrival_ms,
                    f"file:{rel_prompt}",
                    target_decode_tokens,
                    idle_ms,
                ))
                arrival_ms += args.turn_gap_ms + idle_ms

        trace_rows.sort(key=lambda row: (row[2], row[0], row[1]))
        with trace_file.open("w", encoding="utf-8", newline="") as trace:
            trace.write("# session_id\tturn_id\tarrival_ms\tprompt_source\ttarget_decode_tokens\tidle_ms\n")
            for row in trace_rows:
                trace.write("%d\t%d\t%d\t%s\t%d\t%d\n" % row)

        summary = {
            "input": str(Path(args.input)),
            "output_dir": str(output_dir),
            "trace_file": str(trace_file),
            "prompt_dir": str(prompt_dir),
            "decode_estimator": "char_div",
            "chars_per_token": args.chars_per_token,
            "min_decode_tokens": args.min_decode_tokens,
            "max_decode_tokens": args.max_decode_tokens,
            "seed": args.seed,
            "requested_num_sessions": args.num_sessions,
            "requested_max_turns_per_session": args.max_turns_per_session,
            "actual_sessions": len(conversations),
            "actual_turns": len(trace_rows),
            "prompt_files": prompt_files,
            "counts": counts,
            "timing": {
                "start_gap_ms": args.start_gap_ms,
                "turn_gap_ms": args.turn_gap_ms,
                "jitter_ms": args.jitter_ms,
                "idle_ms_low": args.idle_ms_low,
                "idle_ms_high": args.idle_ms_high,
            },
        }
        summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        fail(f"failed to write output: {exc}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-sessions", type=positive_int, default=8)
    parser.add_argument("--max-turns-per-session", type=positive_int, default=4)
    parser.add_argument("--start-gap-ms", type=nonnegative_int, default=40)
    parser.add_argument("--turn-gap-ms", type=nonnegative_int, default=400)
    parser.add_argument("--jitter-ms", type=nonnegative_int, default=50)
    parser.add_argument("--idle-ms-low", type=nonnegative_int, default=200)
    parser.add_argument("--idle-ms-high", type=nonnegative_int, default=800)
    parser.add_argument("--min-decode-tokens", type=positive_int, default=8)
    parser.add_argument("--max-decode-tokens", type=positive_int, default=128)
    parser.add_argument("--chars-per-token", type=positive_float, default=4)
    parser.add_argument("--max-user-chars", type=positive_int, default=2048)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.idle_ms_high < args.idle_ms_low:
        parser.error("--idle-ms-high must be >= --idle-ms-low")
    if args.max_decode_tokens < args.min_decode_tokens:
        parser.error("--max-decode-tokens must be >= --min-decode-tokens")

    return args


def main() -> int:
    args = parse_args()
    records = load_input(Path(args.input))
    counts = {key: 0 for key in COUNT_KEYS}

    conversations: list[list[tuple[str, str]]] = []
    for record in records:
        counts["records_seen"] += 1
        pairs = clean_conversation(record, counts, args.max_user_chars)
        if pairs is None:
            counts["records_dropped"] += 1
            continue
        conversations.append(pairs)

    if len(conversations) < args.num_sessions:
        fail(f"not enough usable conversations: requested {args.num_sessions}, found {len(conversations)}")

    selected = conversations[: args.num_sessions]
    counts["records_used"] = len(selected)
    write_outputs(args, selected, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
