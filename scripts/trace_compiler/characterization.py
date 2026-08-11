#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline characterization.

Two outputs:
  1. Runtime characterization derived from CALIBRATION (cache metrics that
     the frozen calibration justifies). Driven by TTLReactor replay over cal
     lineages.
  2. Ground-truth characterization over selected evaluation lineages
     (offline only!). Whole-trace accounting is scalar-only.

Section 2 is the deliberately-isolated cross-session prefix confound:
for each selected global hash_id we count how many DISTINCT selected lineages
reference it. This statistic is recorded so a later gate can prove the V3
runtime's "session_namespaced" mode actually neutralizes this real
cross-session overlap; it is OFFLINE and MUST NOT enter any runtime event
feature
(see common.OFFLINE_GROUND_TRUTH_FIELDS: global_hash_reuse_count,
raw_global_hash_ids).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

from .common import OFFLINE_TTL_REPLAY_SCOPE
from .lineage import Lineage
from .ttl_reactor import TTLReactor, WorkloadEvent


# ---------------------------------------------------------------------------
# Calibration-driven cache characterization (runtime metrics)
# ---------------------------------------------------------------------------
@dataclass
class CacheCharacterization:
    """Runtime-revisit/idle context from a TTL replay over CAL lineages."""

    replayed_ttl_seconds: float
    replayed_turns: int
    semantics: str = OFFLINE_TTL_REPLAY_SCOPE
    revisit_events: int = 0
    dead_events: int = 0
    cold_restart_events: int = 0
    # REVISIT / (REVISIT + DEAD-rounded-up) == reuse-rate under this TTL.
    # Reuse is ONLY intra-session here: cross-session reuse is suppressed by
    # the session_namespaced prefix_mode and is reported separately below.
    intra_session_reuse_rate: Optional[float] = None
    # Sum of input_lengths over all turn-complete events: a context-footprint
    # proxy measured in tokens.
    context_footprint_tokens: int = 0
    max_context_footprint_tokens: int = 0
    max_lineage_turn_count: int = 0
    max_lineage_lifespan_seconds: float = 0.0


def replay_calibration(
    cal_lineages: List[Lineage], frozen_ttl: float
) -> CacheCharacterization:
    """Replay cal lineages through the TTL reactor under the frozen TTL.

    This replay is OFFLINE projected characterization ONLY (it estimates
    intra-session reuse under a frozen TTL). The reactor itself has no oracle
    and is not used to emit the persisted gate-1A runtime stream.

    Gate 1A emits only TURN_START (arrivals); TURN_COMPLETE is gated to 1B.
    The replay therefore reads input_length off TURN_START events. The
    revisit gap it computes is the inter-arrival gap (offline statistic name
    per P1-4); true revisit-after-completion will be a 1B statistic.
    """
    arrivals = TTLReactor.arrivals_from_lineages(cal_lineages)
    reactor = TTLReactor(frozen_ttl)
    cc = CacheCharacterization(
        replayed_ttl_seconds=frozen_ttl,
        replayed_turns=len(arrivals),
    )
    ctx_max = 0
    ctx_sum = 0
    max_turns = 0
    max_life = 0.0
    for ev in reactor.feed(arrivals):
        _tally(ev, cc)
        if ev.event_type.value == "TURN_START":
            il = int(ev.feature.get("input_length", 0))
            ctx_sum += il
            ctx_max = max(ctx_max, il)
    for ev in reactor.drain():
        _tally(ev, cc)
    by_lin = reactor._state
    for st in by_lin.values():
        max_turns = max(max_turns, st.last_turn)
    for lin in cal_lineages:
        max_life = max(max_life, lin.lifespan)
    denom = cc.revisit_events + cc.dead_events
    cc.intra_session_reuse_rate = (
        cc.revisit_events / denom if denom else None
    )
    cc.context_footprint_tokens = ctx_sum
    cc.max_context_footprint_tokens = ctx_max
    cc.max_lineage_turn_count = max_turns
    cc.max_lineage_lifespan_seconds = max_life
    return cc


def _tally(ev: WorkloadEvent, cc: CacheCharacterization) -> None:
    t = ev.event_type.value
    if t == "REVISIT":
        cc.revisit_events += 1
    elif t == "DEAD":
        cc.dead_events += 1
    elif t == "COLD_RESTART":
        cc.cold_restart_events += 1


# ---------------------------------------------------------------------------
# Ground-truth cross-session prefix reuse (OFFLINE, runtime-suppressed)
# ---------------------------------------------------------------------------
@dataclass
class GlobalHashReuse:
    """Per-global-hash reuse across the trace. OFFLINE ground truth.

    prefix_mode=session_namespaced in every emitted runtime event, so the
    runtime is structurally unable to consume this statistic. It is recorded
    to PROVE the confound exists and to let a later gate verify the
    namespacing neutralizes it.

    Memory model (P1-7):
      * Whole-trace accounting is a scalar streaming summary collected during
        the first validated reader pass; no hash->count map is retained.
      * The selected-prefix reuse CDF is computed only over selected lineages.
        A temporary per-lineage set is folded into a compact hash->count map;
        no hash->set(lineages) structure is retained.
    """

    total_records: int = 0
    total_distinct_hash_ids: Optional[int] = None
    # Whole-trace hash accounting is intentionally scalar-only. The old
    # hash->count map was another large resident structure and is no longer
    # retained; this empty field remains only for schema readers of v1.
    hash_to_global_use_count: Dict[int, int] = field(default_factory=dict)
    global_hash_use_summary: Dict[str, int] = field(default_factory=dict)
    # hash_id -> number of selected lineages referencing it. This is a compact
    # count map, not hash->set(lineages), and is built only from selected turns.
    hash_to_lineage_count: Dict[int, int] = field(default_factory=dict)
    total_hash_references: int = 0
    records_with_hash_ids: int = 0
    max_hash_ids_per_record: int = 0
    reuse_cdf: Dict[str, int] = field(default_factory=dict)
    max_reuse: int = 0
    median_reuse: Optional[float] = None
    p90_reuse: Optional[float] = None
    intra_session_prefix_shared_blocks_avg: Optional[float] = None
    # Scope of the accurate reuse statistic: which selected lineages the CDF
    # summarizes; report readers MUST scope their interpretation to this set.
    accurate_reuse_scope: str = "selected_lineages"
    accurate_reuse_lineage_count: int = 0


def _percentiles(xs: List[float], qs=(0.5, 0.9)):
    if not xs:
        return tuple(None for _ in qs)
    s = sorted(xs)
    import math as _m
    out = []
    for q in qs:
        idx = max(0, min(len(s) - 1, int(_m.ceil(q * len(s))) - 1))
        out.append(s[idx])
    return tuple(out)


@dataclass
class HashUseSummary:
    """Scalar whole-trace hash accounting collected during the first pass."""

    total_hash_references: int = 0
    records_with_hash_ids: int = 0
    max_hash_ids_per_record: int = 0

    def observe(self, hash_ids: Iterable[int]) -> None:
        n = 0
        for _ in hash_ids:
            n += 1
        self.total_hash_references += n
        if n:
            self.records_with_hash_ids += 1
        self.max_hash_ids_per_record = max(self.max_hash_ids_per_record, n)

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_hash_references": self.total_hash_references,
            "records_with_hash_ids": self.records_with_hash_ids,
            "max_hash_ids_per_record": self.max_hash_ids_per_record,
        }


def characterize_global_hash_reuse(
    selected_hash_records: Iterable[Tuple[int, List[Tuple[int, Tuple[int, ...]]]]],
    total_records: int,
    global_hash_use_summary=None,
    global_hash_use_count=None,
    accurate_reuse_scope: str = "selected_lineages",
) -> GlobalHashReuse:
    """Compute OFFLINE cross-session hash reuse without whole-trace maps.

    The selected lineages are the only input that retains hash sequences. For
    each selected lineage we deduplicate its hashes in a temporary set and
    increment a compact hash->lineage-count map. Whole-trace accounting is a
    scalar streaming summary collected during the reader's first pass.
    """
    reuse = GlobalHashReuse(total_records=total_records)
    reuse.accurate_reuse_scope = accurate_reuse_scope
    if global_hash_use_summary is not None:
        if hasattr(global_hash_use_summary, "to_dict"):
            summary = global_hash_use_summary.to_dict()
        else:
            summary = dict(global_hash_use_summary)
        reuse.global_hash_use_summary = {
            str(k): int(v) for k, v in summary.items()
        }
        reuse.total_hash_references = int(summary.get("total_hash_references", 0))
        reuse.records_with_hash_ids = int(summary.get("records_with_hash_ids", 0))
        reuse.max_hash_ids_per_record = int(summary.get("max_hash_ids_per_record", 0))
        reuse.total_distinct_hash_ids = summary.get("distinct_hash_ids")
    elif global_hash_use_count is not None:
        # Compatibility for older callers: aggregate the supplied mapping but
        # never copy it into the result object.
        reuse.total_distinct_hash_ids = len(global_hash_use_count)
        reuse.total_hash_references = int(sum(global_hash_use_count.values()))
        reuse.global_hash_use_summary = {
            "total_hash_references": reuse.total_hash_references,
            "distinct_hash_ids": reuse.total_distinct_hash_ids,
        }

    hash_to_lineage_count: Dict[int, int] = {}
    intra_shared_blocks: List[float] = []
    lineages_iterated = 0
    for lineage_id, turns in selected_hash_records:
        lineages_iterated += 1
        if not turns:
            continue
        common_prefix = list(turns[0][1])
        lineage_hashes = set()
        for _turn, hids in turns:
            lineage_hashes.update(hids)
            if _turn == turns[0][0]:
                continue
            m = 0
            for a, b in zip(common_prefix, hids):
                if a != b:
                    break
                m += 1
            common_prefix = common_prefix[:m]
        for h in lineage_hashes:
            hash_to_lineage_count[h] = hash_to_lineage_count.get(h, 0) + 1
        if len(turns) > 1:
            intra_shared_blocks.append(float(len(common_prefix)))
    reuse.accurate_reuse_lineage_count = lineages_iterated
    reuse.hash_to_lineage_count = hash_to_lineage_count
    counts = list(hash_to_lineage_count.values())
    if counts:
        reuse.max_reuse = max(counts)
        cdf: Dict[str, int] = {}
        for c in counts:
            key = str(c) if c <= 16 else ">16"
            cdf[key] = cdf.get(key, 0) + 1
        reuse.reuse_cdf = dict(sorted(cdf.items(), key=lambda kv: int(kv[0]) if kv[0] != ">16" else 99))
        p50, p90 = _percentiles([float(c) for c in counts], (0.5, 0.9))
        reuse.median_reuse = p50
        reuse.p90_reuse = p90
    if intra_shared_blocks:
        from statistics import fmean
        reuse.intra_session_prefix_shared_blocks_avg = fmean(intra_shared_blocks)
    return reuse


def whole_trace_hash_use_summary(file_path: str) -> HashUseSummary:
    """Read a trace once and return scalar hash accounting only.

    This compatibility helper is intentionally not used by the production CLI:
    the CLI observes the same summary during its first validated pass, then
    uses exactly one bounded second pass for selected hashes.
    """
    summary = HashUseSummary()
    import json as _json
    with open(file_path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            obj = _json.loads(raw.decode("utf-8"))
            hash_ids = obj.get("hash_ids")
            if not isinstance(hash_ids, list):
                raise ValueError("hash_ids must be a list")
            summary.observe(hash_ids)
    return summary


def characterize_lineages_accessory(lineages: List[Lineage]) -> dict:
    """Accessory descriptive statistics computed over all accepted lineages
    of a trace. Used to describe the trace; not consumed by the runtime.
    """
    by_type: Counter = Counter()
    life = []
    turns = []
    foot = []
    for lin in lineages:
        if not lin.turns:
            continue
        turns.append(lin.turn_count)
        life.append(lin.lifespan)
        for mt in lin.turns:
            by_type[mt.type] += 1
            foot.append(mt.input_length)
    return {
        "lineage_count": len(lineages),
        "turn_count_distribution": {str(k): v for k, v in Counter(turns).most_common()},
        "lifespan_p50_p90": list(_percentiles(life)) if life else [None, None],
        "lifespan_max": max(life) if life else 0.0,
        "input_length_p50_p90": list(_percentiles([float(x) for x in foot])) if foot else [None, None],
        "type_distribution": dict(by_type),
    }
