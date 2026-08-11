#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streaming trace reader.

NEVER loads the whole trace into memory. Iterates JSONL line by line,
validates each record (common.TraceRecord.from_raw), and computes a
streaming SHA-256 over the raw byte stream plus a per-line SHA-256 index
so downstream stages can detect tampering / reordering without holding all
records.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Tuple

from .common import TraceRecord, TraceRecordError, sha256_bytes


@dataclass
class StreamProvenance:
    """Captured during a single streaming pass. Memory-bounded."""

    trace_key: str
    file_path: str
    record_count: int
    raw_byte_sha256: str
    # Streaming SHA-256 over the concatenation of per-record line SHA-256s.
    # This is an integrity/index fingerpring, stable regardless of the
    # dataset pin, and cheaper to reason about than re-reading the file.
    line_fingerprint_sha256: str
    first_timestamp: Optional[float]
    last_timestamp: Optional[float]


class TraceReadStream:
    """Iterate a trace file as validated TraceRecord objects.

    Memory footprint: O(1) beyond one in-flight record and the rolling
    hash state. Callers may pull hash_ids off each record and discard it.
    """

    def __init__(self, trace_key: str, file_path: str) -> None:
        self.trace_key = trace_key
        self.file_path = file_path

    def __iter__(self) -> Iterator[TraceRecord]:
        raw_h = hashlib.sha256()
        line_fp = hashlib.sha256()
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None
        count = 0
        with open(self.file_path, "rb") as f:
            for raw in f:
                if not raw.strip():
                    continue
                raw_h.update(raw)
                line_sha = hashlib.sha256(raw).hexdigest()
                line_fp.update(line_sha.encode("ascii"))
                try:
                    obj = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise TraceRecordError(
                        f"line {count + 1} is not valid UTF-8 JSON: {exc}"
                    ) from exc
                rec = TraceRecord.from_raw(obj)
                count += 1
                if first_ts is None:
                    first_ts = rec.timestamp
                last_ts = rec.timestamp
                yield rec
        # Stash provenance on the iterator after exhaustion.
        self.provenance = StreamProvenance(
            trace_key=self.trace_key,
            file_path=self.file_path,
            record_count=count,
            raw_byte_sha256=raw_h.hexdigest(),
            line_fingerprint_sha256=line_fp.hexdigest(),
            first_timestamp=first_ts,
            last_timestamp=last_ts,
        )


def read_with_provenance(trace_key: str, file_path: str) -> Tuple[list, StreamProvenance]:
    """Convenience collector: drains the stream and returns (records, provenance).

    WARNING: this materializes all records. Prefer iterating TraceReadStream
    directly for the compiler hot paths; this helper exists only for tests and
    bounded one-shot materialization (e.g. a small slice).
    """
    rs = TraceReadStream(trace_key, file_path)
    out = list(rs)
    return out, rs.provenance
