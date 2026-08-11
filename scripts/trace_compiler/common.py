#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Alibaba/Qwen-Bailian trace workload compiler.

Streaming trace -> frozen canonical workload manifests for the V3 KV
offload work. This package ONLY compiles raw anonymized production traces
into versioned manifests; it is NOT a second benchmark runner/parser and
does not modify the canonical runner used by the V3-1A work
(scripts/run-kv-offload-benchmark.py et al.).

Pinned dataset: /root/oscomp/qwen-bailian-usagetraces-anon @
5f7439c51ec248a0c585f7d90a41a6f57773b912
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Fixed provenance
# ---------------------------------------------------------------------------
TRACE_REPO_DIR = "/root/oscomp/qwen-bailian-usagetraces-anon"
TRACE_REPO_HEAD = "5f7439c51ec248a0c585f7d90a41a6f57773b912"

TRACE_FILES: Dict[str, str] = {
    "traceA": "qwen_traceA_blksz_16.jsonl",
    "traceB": "qwen_traceB_blksz_16.jsonl",
    "coder": "qwen_coder_blksz_16.jsonl",
    "thinking": "qwen_thinking_blksz_16.jsonl",
}

BLOCK_SIZE_TOKENS = 16

# Manifest schema version for every frozen output emitted by this gate.
# Bump only on a backward-incompatible change to manifest shape.
MANIFEST_SCHEMA_VERSION = "gt-trace-1a/v2"

# prefix_mode is always session_namespaced in 1A; cross-session prefix reuse
# is recorded only as an offline ground-truth statistic, never exposed to the
# runtime event stream.
PREFIX_MODE = "session_namespaced"

# Gate 1A freezes only arrival metadata and the per-turn workload plan. TTL
# lifecycle transitions require a real completion signal and are deferred to
# GT-trace-1B; any TTL replay in this gate is offline projected evidence.
RUNTIME_EVENT_SCOPE = "arrival_and_request_plan_only"
OFFLINE_TTL_REPLAY_SCOPE = "offline_projected_characterization_only"
FROZEN_REQUEST_PLAN_FIELDS: Tuple[str, ...] = (
    "target_output_length",
    "n_predict",
    "required_context_tokens",
)


# ---------------------------------------------------------------------------
# Record model (validated per-record; bounded)
# ---------------------------------------------------------------------------
REQUIRED_RECORD_FIELDS = (
    "chat_id",
    "parent_chat_id",
    "timestamp",
    "input_length",
    "output_length",
    "type",
    "turn",
    "hash_ids",
)


class TraceRecordError(ValueError):
    """Raised when a raw trace record is malformed (fail-closed)."""


@dataclass(frozen=True)
class TraceRecord:
    """One request-unit record from a Bailian trace file.

    A record corresponds to exactly one request (one turn of one session),
    NOT a whole session. Each record has its own unique chat_id. A session
    lineage is reconstructed by chaining turn_k.parent_chat_id == turn_{k-1}.chat_id
    from a root (parent_chat_id == -1, turn == 1) downward.
    """

    chat_id: int
    parent_chat_id: int
    timestamp: float
    input_length: int
    output_length: int
    type: str
    turn: int
    # Tuple, not list: immutable and append-friendly for bounded memory use.
    hash_ids: Tuple[int, ...]

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "TraceRecord":
        miss = [f for f in REQUIRED_RECORD_FIELDS if f not in raw]
        if miss:
            raise TraceRecordError(f"record missing fields: {miss}")
        try:
            chat_id = int(raw["chat_id"])
            parent_chat_id = int(raw["parent_chat_id"])
            timestamp = float(raw["timestamp"])
            input_length = int(raw["input_length"])
            output_length = int(raw["output_length"])
            rtype = str(raw["type"])
            turn = int(raw["turn"])
            hash_ids = raw["hash_ids"]
        except (TypeError, ValueError) as exc:
            raise TraceRecordError(f"record field type error: {exc}") from exc
        if not isinstance(hash_ids, list):
            raise TraceRecordError("hash_ids must be a list")
        if chat_id < 0:
            raise TraceRecordError(f"chat_id must be >= 0 (got {chat_id})")
        if parent_chat_id == -1 and turn != 1:
            raise TraceRecordError(
                f"root record (parent=-1) must have turn==1 "
                f"(got chat_id={chat_id} turn={turn})"
            )
        if turn < 1:
            raise TraceRecordError(f"turn must be >= 1 (got {turn})")
        if input_length < 0 or output_length < 0:
            raise TraceRecordError("input/output_length must be >= 0")
        if timestamp < 0:
            raise TraceRecordError("timestamp must be >= 0")
        # Materialize as tuple; caller may drop it immediately after use.
        return cls(
            chat_id=chat_id,
            parent_chat_id=parent_chat_id,
            timestamp=timestamp,
            input_length=input_length,
            output_length=output_length,
            type=rtype,
            turn=turn,
            hash_ids=tuple(int(h) for h in hash_ids),
        )

    @property
    def is_root(self) -> bool:
        return self.parent_chat_id == -1

    @property
    def block_count(self) -> int:
        """Number of 16-token blocks in this request's prompt context.

        The Bailian trace remaps hashed blocks to sequential ids; hash_ids
        length is the prompt-context block count (NOT input_length/16 in
        general, since ids are deduplicated sequential integers, but the
        count of entries reflects the carried context). Used as the
        context-footprint proxy.
        """
        return len(self.hash_ids)


# ---------------------------------------------------------------------------
# Distinction (core contract: runtime-visible vs offline ground truth)
# ---------------------------------------------------------------------------
# Fields below are the ONLY items a future canonical runner may consume at
# runtime (i.e. while deciding KV eviction/offload for turn k, it may only
# know these about turn k's past). ANY field prefixed OFFLINE_ is ground
# truth derived from the whole trace and MUST NOT enter runtime events.
RUNTIME_VISIBLE_EVENT_FEATURES: Tuple[str, ...] = (
    "lineage_id",
    "turn",
    "relative_arrival_offset",  # seconds since lineage first turn (preserved)
    "timestamp_abs",  # only for ordering/emission, not for "is this the final turn"
    "input_length",
    "type",
    "block_count",
    "session_namespaced_block_ids",  # materialized, namespaced (materialize.py)
)

# Explicit deny-list: scanner to assert these never leak into runtime events.
OFFLINE_GROUND_TRUTH_FIELDS: Tuple[str, ...] = (
    "output_length",  # NOT runtime-visible at TURN_START (arrival): the answer
                      # has not been generated; target_output_length lives in
                      # the workload plan (n_predict), never on the policy event.
                      # A future Bidaw-style answer-length feature may only use
                      # the actual output tokens of PREVIOUS completed turns
                      # (offline inter-arrival / revisit gap statistic).
    "is_final_turn_of_lineage",  # <= MUST NOT be used to pre-release
    "next_revisit_time",  # <= MUST NOT be used to pre-release
    "total_lineage_turn_count",  # <= ground-truth; runtime discovers incrementally
    "global_hash_reuse_count",  # <= cross-session ground truth
    "raw_global_hash_ids",  # <= non-namespaced, the confound being isolated
)


def assert_no_future_fields(event_feature_dict: Dict[str, Any]) -> None:
    """Fail-closed guard: a runtime event feature dict must NOT carry any
    offline ground-truth / future-looking field. Call before emitting any
    runtime-visible event record.
    """
    for f in OFFLINE_GROUND_TRUTH_FIELDS:
        if f in event_feature_dict:
            raise AssertionError(
                f"runtime event feature leaked offline/future field '{f}': "
                f"online TTL contract forbids pre-release with future info"
            )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (never reads whole file into memory)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_head_or_pinned() -> str:
    """Return the pinned trace repo HEAD constant (this gate's dataset pin).

    We deliberately do NOT shell out to read the live repo's HEAD here: the
    contract pins the dataset at the documented commit, and a stale live
    checkout would otherwise silently drift the frozen manifests.
    """
    return TRACE_REPO_HEAD


def trace_file_path(trace_key: str) -> str:
    if trace_key not in TRACE_FILES:
        raise KeyError(f"unknown trace key {trace_key!r}; known={list(TRACE_FILES)}")
    return os.path.join(TRACE_REPO_DIR, TRACE_FILES[trace_key])
