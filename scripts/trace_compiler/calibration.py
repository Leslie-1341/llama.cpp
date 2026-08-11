#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration: time-ordered split and frozen statistics.

Rule: the trace is split CHRONOLOGICALLY (by timestamp) into a calibration
prefix and a held-out evaluation suffix. Calibration computes and FREEZES
TTL and slice rules; evaluation MUST NOT retroactively retune.

Two decisions are explicit and pinned here:

  TTL calibration algorithm
  ------------------------
  Option A (default, implemented): "coverage quantile". Compute the
    inter-turn idle gap distribution over multi-turn lineages in the
    calibration window; pick TTL as the idle-gap quantile that captures a
    target fraction (default 0.9) of actual revisits (i.e. P(revisit<=TTL)
    >= 0.9). This is the simplest non-oracle calibration: TTL depends only
    on the observed idle-gap distribution of the CAL past, never on any
    held-out session's future.

  Option B (NOT implemented, kept as a flagged alternative for a later
    gate): "Pareto-knee on reuse-rate vs memory-residency". Sweep a grid of
    TTL candidates, replay calibration through the reactor, and pick the
    knee of (revisit_hit_rate vs mean_resident_lifespan). More principled
    but heavier and more easily mis-tuned; deferred.

  We pick Option A here because it is the cheapest calibration that still
    is provably future-free, and we do not perform any budget-vs-TTL sweep
    (that is explicitly out of scope for gate 1A).
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .lineage import Lineage  # noqa: F401  (re-export for old callers)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimeSplit:
    """A chronological calibration/evaluation split by ABSOLUTE timestamp cut.

    Boundary semantics (P0-3):
      * calibration_window = [first_ts, split_ts]  (inclusive on split_ts)
      * evaluation_window  = (split_ts, last_ts]   (exclusive on split_ts)
      * A lineage whose turns all fall on ONE side of split_ts is wholly
        calibration or wholly evaluation.
      * A lineage that STRADDLES split_ts (some turns <= split_ts, others
        > split_ts) is `boundary_excluded`: it is allowed in NEITHER side.
        It is recorded as excluded (cross-cut lineage) and NEVER enters
        calibration replay, idle-gap stats, or evaluation slice selection.

    This is an ABSOLUTE timestamp cut isolation: no calibration turn may
    land past split_ts, and no evaluation turn may land at-or-before split_ts.
    Cross-cut lineage is explicitly excluded (not silently dropped).
    """

    split_ts: float  # the absolute boundary; cal <= split_ts, eval > split_ts
    calibration_until_ts: float  # == split_ts (inclusive bound for cal)
    evaluation_from_ts: float    # exclusive lower bound for eval (== split_ts)
    calibration_session_count: int
    evaluation_session_count: int
    boundary_excluded_session_count: int  # straddling lineages
    calibration_turn_count: int
    evaluation_turn_count: int
    boundary_excluded_turn_count: int
    total_session_count: int  # accepted lineages considered
    total_turn_count: int   # cal + eval + boundary_excluded turns
    boundary_excluded_lineage_ids: Tuple[int, ...]
    first_ts: float
    last_ts: float

    @property
    def calibration_window(self) -> Tuple[float, float]:
        return (self.first_ts, self.calibration_until_ts)

    @property
    def evaluation_window(self) -> Tuple[float, float]:
        return (self.evaluation_from_ts, self.last_ts)

    @property
    def boundary_excluded_window(self) -> Tuple[float, float]:
        """The window spanned by boundary-excluded lineages (informational)."""
        return (self.calibration_until_ts, self.evaluation_from_ts)


def make_time_split(
    lineages: List[Lineage],
    calibration_fraction: float = 0.5,
    split_ts_override: Optional[float] = None,
) -> TimeSplit:
    """Compute the absolute-timestamp calibration/evaluation cut.

    select split_ts as the calibration_fraction quantile of ROOT timestamps
    (or use split_ts_override when provided for reproducibility). A lineage is:
      * calibration  iff ALL its turns' timestamps <= split_ts
      * evaluation   iff its ROOT ts > split_ts AND all its inner turns also
                      > split_ts (no inner turn lands in the cal window)
      * boundary_excluded otherwise (straddles the cut)

    Cross-cut lineages are returned in TimeSplit.boundary_excluded_lineage_ids
    and MUST be excluded from BOTH calibration replay and eval sampling.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0,1)")
    if not lineages:
        raise ValueError("no lineages to split")
    rooted = [lin for lin in lineages if lin.turns]
    if not rooted:
        raise ValueError("no lineages with turns to split")
    if split_ts_override is not None:
        split_ts = float(split_ts_override)
    else:
        roots = sorted(
            ((lin.turns[0].timestamp, lin.lineage_id) for lin in rooted),
            key=lambda x: (x[0], x[1]),
        )
        n = len(roots)
        cal_n = max(1, min(n - 1, int(round(n * calibration_fraction))))
        split_ts = roots[cal_n - 1][0]
    cal_sessions = 0
    cal_turns = 0
    eval_sessions = 0
    eval_turns = 0
    bnd_sessions = 0
    bnd_turns = 0
    bnd_ids: List[int] = []
    for lin in rooted:
        max_ts = max(t.timestamp for t in lin.turns)
        min_ts = lin.turns[0].timestamp
        if max_ts <= split_ts:
            cal_sessions += 1
            cal_turns += lin.turn_count
        elif min_ts > split_ts:
            eval_sessions += 1
            eval_turns += lin.turn_count
        else:
            bnd_sessions += 1
            bnd_turns += lin.turn_count
            bnd_ids.append(lin.lineage_id)
    first_ts = min(lin.turns[0].timestamp for lin in rooted)
    last_ts = max(max(t.timestamp for t in lin.turns) for lin in rooted)
    return TimeSplit(
        split_ts=split_ts,
        calibration_until_ts=split_ts,
        evaluation_from_ts=split_ts,
        calibration_session_count=cal_sessions,
        evaluation_session_count=eval_sessions,
        boundary_excluded_session_count=bnd_sessions,
        calibration_turn_count=cal_turns,
        evaluation_turn_count=eval_turns,
        boundary_excluded_turn_count=bnd_turns,
        total_session_count=len(rooted),
        total_turn_count=cal_turns + eval_turns + bnd_turns,
        boundary_excluded_lineage_ids=tuple(sorted(bnd_ids)),
        first_ts=first_ts,
        last_ts=last_ts,
    )


def assign_lineage_to_split(lin: Lineage, split: TimeSplit) -> str:
    """Classify a lineage w.r.t. the absolute-timestamp cut.

    Returns one of:
      "calibration"      -- all turns <= split_ts (root ts decides nothing
                            here; the WHOLE chain is past-cut-checked)
      "evaluation"       -- all turns > split_ts
      "boundary_excluded"-- straddles the cut (must NOT enter cal or eval)
    """
    if not lin.turns:
        return "boundary_excluded"
    t0 = lin.turns[0].timestamp
    tmax = max(t.timestamp for t in lin.turns)
    if tmax <= split.calibration_until_ts:
        return "calibration"
    if t0 > split.evaluation_from_ts:
        return "evaluation"
    return "boundary_excluded"


def _linenoop(*a, **k):
    """Placeholder retained for any old callers expecting the bool return."""
    raise DeprecationWarning(
        "assign_lineage_to_split now returns a str classification. Use assign_lineage_to_split(lin, split) in {'calibration','evaluation','boundary_excluded'}."
    )


# ---------------------------------------------------------------------------
# Idle-gap + revisit calibration
# ---------------------------------------------------------------------------
@dataclass
class IdleGapStats:
    revisits_observed: int = 0  # multi-turn lineages' inter-turn arrivals
    # Inter-arrival / revisit gaps (seconds). Gate-1A name is idle_gaps for
    # schema continuity, but the semantics are the INTER-ARRIVAL gap between
    # consecutive turns of a lineage (an offline calibration statistic per
    # P1-4; the true revisit-after-COMPLETION gap is a 1B statistic after
    # real completion is observed). Past-derived only.
    idle_gaps: List[float] = field(default_factory=list)  # seconds (inter-arrival)
    turn_counts: List[int] = field(default_factory=list)
    input_lengths: List[int] = field(default_factory=list)
    # output_lengths are workload-plan data only (n_predict / target_output_
    # length); never enter runtime TURN_START features (P1-5). Retained
    # purely to inform 1B generation-budget sizing from past observations.
    output_lengths: List[int] = field(default_factory=list)
    block_counts: List[int] = field(default_factory=list)
    session_lifespans: List[float] = field(default_factory=list)
    arrival_burst: List[int] = field(default_factory=list)  # arrivals per 60s bucket


def collect_calibration_stats(
    cal_lineages: List[Lineage],
    bin_seconds: int = 60,
) -> IdleGapStats:
    """Compute inter-arrival / turn / length / burst / lifespan / footprint
    distributions over the CALIBRATION lineages only (absolute-timestamp
    cut: only lineages wholly in the calibration window reach here; the
    caller must have excluded boundary-straddling lineages).

    idle_gap (inter-arrival gap, seconds) for a multi-turn lineage = the
    difference ts_{turn+1} - ts_turn. These are OBSERVED inter-arrival gaps
    used to fit the TTL coverage quantile below. They are PAST-derived
    (each gap is between two turns that already arrived), never future
    looking.
    """
    st = IdleGapStats()
    bucket_to_count: Dict[int, int] = {}
    for lin in cal_lineages:
        if not lin.turns:
            continue
        st.turn_counts.append(lin.turn_count)
        st.session_lifespans.append(lin.lifespan)
        for i, mt in enumerate(lin.turns):
            st.input_lengths.append(mt.input_length)
            st.output_lengths.append(mt.output_length)
            st.block_counts.append(mt.block_count)
            b = int(mt.timestamp // bin_seconds)
            bucket_to_count[b] = bucket_to_count.get(b, 0) + 1
            if i > 0:
                # inter-arrival gap (turn_{i+1} arrival - turn_i arrival).
                gap = mt.timestamp - lin.turns[i - 1].timestamp
                st.idle_gaps.append(gap)
                st.revisits_observed += 1
    st.arrival_burst = sorted(bucket_to_count.values(), reverse=True)
    return st


def fit_ttl_coverage_quantile(
    idle_gaps: List[float], target_fraction: float = 0.9
) -> Tuple[float, float]:
    """Pick TTL so P(revisit<=TTL) >= target_fraction against the calibration
    idle gaps.

    Returns (ttl_seconds, achieved_fraction). This is Option A from the
    module docstring. Assumes idle_gaps are observed revisit intervals
    (seconds between consecutive turns of a lineage) within calibration.
    """
    if not idle_gaps:
        # No multi-turn in calibration: defer to a gate-default minimal TTL.
        # (Trace B has no multi-turn; its workload doesn't drive TTL either.)
        return 60.0, 0.0
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0,1]")
    gs = sorted(idle_gaps)
    idx = max(0, min(len(gs) - 1, int(math.ceil(target_fraction * len(gs))) - 1))
    ttl = gs[idx]
    # Achieved fraction actually observed:
    achieved = sum(1 for g in gs if g <= ttl) / len(gs)
    return float(ttl), achieved


# ---------------------------------------------------------------------------
# Frozen calibration artifact
# ---------------------------------------------------------------------------
@dataclass
class FrozenCalibration:
    """Immutable result of the calibration phase. Stitched into every
    manifest as the TTL calibration identity, so evaluation can be audited
    against the exact calibration that produced TTL."""

    algorithm: str  # "coverage_quantile"
    target_fraction: float
    frozen_ttl_seconds: float
    achieved_fraction: float
    calibration_until_ts: float
    calibration_session_count: int
    calibration_turn_count: int
    idle_gap_count: int
    idle_gap_mean: Optional[float]
    idle_gap_p50: Optional[float]
    idle_gap_p90: Optional[float]
    idle_gap_max: Optional[float]
    turn_count_dist: Dict[str, int]  # "1","2",... -> count
    input_length_avg: Optional[float]
    output_length_avg: Optional[float]
    block_count_avg: Optional[float]
    session_lifespan_avg: Optional[float]
    arrival_burst_p50: Optional[float]
    arrival_burst_p90: Optional[float]
    split: TimeSplit

    def to_manifest_dict(self) -> dict:
        d = asdict(self)
        d["split"] = asdict(self.split)
        return d


def percentiles(xs: List[float], qs=(0.5, 0.9)) -> Tuple[Optional[float], ...]:
    if not xs:
        return tuple(None for _ in qs)
    s = sorted(xs)
    out = []
    for q in qs:
        idx = max(0, min(len(s) - 1, int(math.ceil(q * len(s))) - 1))
        out.append(s[idx])
    return tuple(out)


def build_frozen_calibration(
    cal_lineages: List[Lineage],
    split: TimeSplit,
    target_fraction: float = 0.9,
) -> FrozenCalibration:
    stats = collect_calibration_stats(cal_lineages)
    ttl, achieved = fit_ttl_coverage_quantile(stats.idle_gaps, target_fraction)
    (p50, p90) = percentiles(stats.idle_gaps, (0.5, 0.9))
    imax = max(stats.idle_gaps) if stats.idle_gaps else None
    imean = statistics.fmean(stats.idle_gaps) if stats.idle_gaps else None
    tc_dist: Dict[str, int] = {}
    for tc in stats.turn_counts:
        tc_dist[str(tc)] = tc_dist.get(str(tc), 0) + 1
    in_avg = statistics.fmean(stats.input_lengths) if stats.input_lengths else None
    out_avg = statistics.fmean(stats.output_lengths) if stats.output_lengths else None
    blk_avg = statistics.fmean(stats.block_counts) if stats.block_counts else None
    life_avg = statistics.fmean(stats.session_lifespans) if stats.session_lifespans else None
    b_p50, b_p90 = percentiles([float(x) for x in stats.arrival_burst], (0.5, 0.9))
    return FrozenCalibration(
        algorithm="coverage_quantile",
        target_fraction=target_fraction,
        frozen_ttl_seconds=ttl,
        achieved_fraction=achieved,
        calibration_until_ts=split.calibration_until_ts,
        calibration_session_count=split.calibration_session_count,
        calibration_turn_count=split.calibration_turn_count,
        idle_gap_count=len(stats.idle_gaps),
        idle_gap_mean=imean,
        idle_gap_p50=p50,
        idle_gap_p90=p90,
        idle_gap_max=imax,
        turn_count_dist=tc_dist,
        input_length_avg=in_avg,
        output_length_avg=out_avg,
        block_count_avg=blk_avg,
        session_lifespan_avg=life_avg,
        arrival_burst_p50=b_p50,
        arrival_burst_p90=b_p90,
        split=split,
    )
