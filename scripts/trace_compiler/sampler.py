#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen slice sampler + versioned manifest emitter.

Three held-out slice classes:
  * typical            - a deterministic whole-session sample of sessions
  * revisit-heavy      - sessions with turn_count >= the revisit-heaviness
                         threshold (frozen by calibration), rich in REVISIT
  * burst-long-context - sessions whose root falls in a high-arrival burst
                         window AND whose max context is large; exercises both
                         pressure (burst) and footprint (long ctx) together

Sampling unit = WHOLE SESSION (complete accepted lineage). Turn order and
relative arrival/idle within a session are preserved exactly; no turn is
randomly dropped. Only whole sessions are selected.

Scaling: if a future canonical runner needs to dilate time, only a single
time_dilation parameter is recorded per manifest. Gate 1A does NOT pick a
device-tuned value (the default is 1.0, meaning original timestamps); the
contract says the user/later gate chooses it from measured device perf.

Every manifest is versioned (common.MANIFEST_SCHEMA_VERSION) and binds the
full frozen identity required for reproducibility:
  trace repo HEAD, file SHA256, source window, selection rule, seed,
  session count, turn count, time_dilation, event-stream SHA256,
  max context footprint, prefix_mode=session_namespaced, TTL calibration
  identity, tokenization-calibration requirement.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from .calibration import FrozenCalibration, TimeSplit
from .common import (
    MANIFEST_SCHEMA_VERSION,
    OFFLINE_TTL_REPLAY_SCOPE,
    PREFIX_MODE,
    RUNTIME_EVENT_SCOPE,
    TRACE_FILES,
    TRACE_REPO_HEAD,
)
from .lineage import Lineage, MinimalTurn
from .materialize import Materializer
from .ttl_reactor import EventType, TTLReactor, WorkloadEvent


SLICE_TYPICAL = "typical"
SLICE_REVISIT_HEAVY = "revisit-heavy"
SLICE_BURST_LONG_CTX = "burst-long-context"


@dataclass
class FrozenManifest:
    """The versioned frozen workload manifest emitted for one slice (P1-6).

    Window/context fields are split explicitly (not a single source_window):
      * trace_span            : [first_ts, last_ts] of the whole trace file
      * calibration_window   : [first_ts, calibration_until_ts] (cal cut)
      * evaluation_window    : (split_ts, last_ts] (eval cut)
      * selected_event_span  : [min_ts, max_ts] across the slice's events
                                (only the selected sessions/turns). This is
                                the bound a future canonical runner may
                                replay from.

    Context fields:
      * max_prompt_tokens            : max input_length across the slice's
                                       turns (the prompt-side budget).
      * max_required_context_tokens  : max(input_length + target_output_length)
                                       over individual turns. The per-turn
                                       request plan is persisted beside the
                                       event feature and is not policy-visible.
    """

    schema_version: str
    slice_class: str
    trace_key: str
    trace_repo_head: str
    trace_file: str
    trace_file_sha256: str
    source_window_ts: Tuple[float, float]  # legacy alias of trace_span
    trace_span: Tuple[float, float]
    calibration_window: Tuple[float, float]
    evaluation_window: Tuple[float, float]
    selected_event_span: Tuple[float, float]
    selection_rule: str
    seed: int
    session_count: int
    turn_count: int
    time_dilation: float
    prefix_mode: str
    materialize_mode: str
    runtime_event_scope: str
    ttl_replay_scope: str
    tokenization_calibration_requirement: dict
    # Identity of the offline TTL projection; it does not govern persisted
    # gate-1A runtime events, which contain arrivals and request plans only.
    ttl_calibration_identity: dict
    max_prompt_tokens: int  # max input_length across slice turns (P1-6)
    max_required_context_tokens: int  # max(input + target_output) per turn
    max_context_footprint_tokens: int  # kept = max_prompt_tokens (legacy alias)
    event_stream_sha256: str
    event_count: int
    session_ids: List[int]
    descriptive_stats: dict

    def to_canonical_json(self) -> str:
        # Stable, sorted-key JSON for fingerprint stability across tools.
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


def _deterministic_sample(items: List[Lineage], seed: int, k: int) -> List[Lineage]:
    """Deterministic whole-session sample without RNG side effects.

    We avoid random.* (non-deterministic across interpreter seeds by
    default); a simple deterministic hash-based selection is used so the
    same (items, seed, k) always yields the same selection.
    """
    if k >= len(items):
        return list(items)
    # Build a deterministic ordering via SHA-256 of (seed, lineage_id).
    keyed = sorted(
        ((int(hashlib.sha256(f"{seed}:{it.lineage_id}".encode()).hexdigest(), 16), it) for it in items),
        key=lambda kv: kv[0],
    )
    return [it for _, it in keyed[:k]]


def select_typical(
    eval_lineages: List[Lineage], seed: int, n: int
) -> List[Lineage]:
    """Select a deterministic whole-session sample from the eval prefix."""
    return _deterministic_sample(eval_lineages, seed + 1, n)


def select_revisit_heavy(
    eval_lineages: List[Lineage], revisit_heavy_turn_floor: int, seed: int, n: int
) -> List[Lineage]:
    pool = [l for l in eval_lineages if l.turn_count >= revisit_heavy_turn_floor]
    return _deterministic_sample(pool, seed + 2, n)


def select_burst_long_context(
    eval_lineages: List[Lineage],
    burst_window_q: float,
    long_ctx_q: float,
    seed: int,
    n: int,
) -> List[Lineage]:
    """Select sessions in a high-arrival burst window with large context.

    A lineage is a candidate iff its root timestamp falls in the top
    `burst_window_q` fraction of per-bucket arrival counts AND its max
    in-turn input_length is in the top `long_ctx_q` fraction of all lineages'
    maxima. The selection respects both burst (pressure) and long-context
    (footprint) at once, for the slice class expected to stress the
    runtime most.
    """
    if not eval_lineages:
        return []
    # root-ts buckets by 60s to find burst windows
    from collections import Counter
    bins = Counter(int(l.turns[0].timestamp // 60) for l in eval_lineages if l.turns)
    arr_counts = sorted(bins.values())
    def qtile(xs, q):
        idx = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
        return xs[idx]
    burst_cut = qtile(arr_counts, burst_window_q)
    burst_bins = {b for b, c in bins.items() if c >= burst_cut}
    max_ctx = [max((t.input_length for t in l.turns), default=0) for l in eval_lineages]
    ctx_cut = qtile(sorted(max_ctx), long_ctx_q)
    pool = [
        l for l in eval_lineages
        if l.turns and int(l.turns[0].timestamp // 60) in burst_bins
        and max((t.input_length for t in l.turns), default=0) >= ctx_cut
    ]
    return _deterministic_sample(pool, seed + 3, n)


def _request_plan_for_turn(mt: MinimalTurn) -> dict:
    target = int(mt.output_length)
    return {
        "target_output_length": target,
        "n_predict": target,
        "required_context_tokens": int(mt.input_length) + target,
    }


def _validate_persisted_event_dict(
    event: dict,
    expected_turns: Optional[set] = None,
) -> None:
    if event.get("event_type") != EventType.TURN_START.value:
        raise ValueError(
            "GT-trace-1A runtime stream may persist only TURN_START arrivals; "
            f"got {event.get('event_type')!r}"
        )
    feature = event.get("feature")
    plan = event.get("request_plan")
    if not isinstance(feature, dict):
        raise ValueError("TURN_START feature must be an object")
    if not isinstance(plan, dict):
        raise ValueError("TURN_START request_plan is missing")
    forbidden_plan_fields = {
        "output_length", "target_output_length", "n_predict",
        "required_context_tokens",
    }
    leaked = forbidden_plan_fields.intersection(feature)
    if leaked:
        raise ValueError(
            "request-plan field leaked into policy-visible feature: "
            + ",".join(sorted(leaked))
        )
    required_plan = {
        "target_output_length", "n_predict", "required_context_tokens",
    }
    if set(plan) != required_plan:
        raise ValueError(
            "request_plan must contain exactly target_output_length, n_predict, "
            "required_context_tokens"
        )
    target = plan["target_output_length"]
    n_predict = plan["n_predict"]
    required_context = plan["required_context_tokens"]
    if not all(isinstance(v, int) and v >= 0 for v in (target, n_predict, required_context)):
        raise ValueError("request_plan values must be non-negative integers")
    if n_predict != target:
        raise ValueError("request_plan n_predict must equal target_output_length")
    input_length = feature.get("input_length")
    if not isinstance(input_length, int) or input_length < 0:
        raise ValueError("TURN_START input_length must be a non-negative integer")
    if required_context != input_length + target:
        raise ValueError(
            "request_plan.required_context_tokens must equal input_length + "
            "target_output_length"
        )
    block_count = feature.get("block_count")
    block_ids = feature.get("session_namespaced_block_ids")
    if not isinstance(block_count, int) or block_count < 0:
        raise ValueError("TURN_START block_count must be a non-negative integer")
    if not isinstance(block_ids, list) or len(block_ids) != block_count:
        raise ValueError(
            "TURN_START materialization incomplete: expected "
            f"{block_count} block ids, got "
            f"{len(block_ids) if isinstance(block_ids, list) else 'missing'}"
        )
    if not all(isinstance(block_id, str) for block_id in block_ids):
        raise ValueError("session_namespaced_block_ids must contain strings")
    if expected_turns is not None:
        key = (int(event["lineage_id"]), int(event["turn"]))
        if key not in expected_turns:
            raise ValueError(f"unexpected TURN_START in persisted stream: {key}")


def validate_persisted_event_stream(
    events: List[dict], expected_turns: Optional[set] = None
) -> None:
    """Fail closed on the exact persisted GT-trace-1A runtime contract."""
    seen = set()
    for index, event in enumerate(events, start=1):
        if event.get("line_msg_index") != index:
            raise ValueError("persisted event line_msg_index is not contiguous")
        _validate_persisted_event_dict(event, expected_turns)
        key = (int(event["lineage_id"]), int(event["turn"]))
        if key in seen:
            raise ValueError(f"duplicate TURN_START in persisted stream: {key}")
        seen.add(key)
    if expected_turns is not None and seen != expected_turns:
        missing = sorted(expected_turns - seen)
        extra = sorted(seen - expected_turns)
        raise ValueError(
            f"TURN_START materialization set mismatch: missing={missing[:3]} "
            f"extra={extra[:3]}"
        )


def validate_persistable_event_stream(
    events: List[WorkloadEvent], expected_turns: Optional[set] = None
) -> None:
    """Validate in-memory events before they are written to disk."""
    validate_persisted_event_stream(
        [event.to_runtime_contract_dict() for event in events], expected_turns
    )


def load_event_stream(path: str) -> Tuple[List[dict], str]:
    """Reload a persisted stream without consulting the original trace."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    validate_persisted_event_stream(events)
    return events, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _emit_event_stream_full(
    slice_lineages: List[Lineage],
    frozen_ttl: float,
    materializer: Materializer,
    hash_provider,
):
    """Emit the frozen gate-1A arrival/request-plan stream.

    `frozen_ttl` is accepted to keep the assembly interface stable, but TTL
    lifecycle replay is deliberately not used here. Without real generation
    completions, DEAD/REVISIT/COLD_RESTART are offline projections only and
    must not be persisted as runtime events.
    """
    del frozen_ttl
    arrivals = TTLReactor.arrivals_from_lineages(slice_lineages)
    first_ts = {
        lineage.lineage_id: lineage.turns[0].timestamp
        for lineage in slice_lineages if lineage.turns
    }
    events: List[WorkloadEvent] = []
    for index, (ts, lineage_id, turn, mt) in enumerate(arrivals, start=1):
        feature = {
            "lineage_id": lineage_id,
            "turn": turn,
            "relative_arrival_offset": ts - first_ts[lineage_id],
            "timestamp_abs": ts,
            "input_length": mt.input_length,
            "type": mt.type,
            "block_count": mt.block_count,
        }
        if hash_provider is not None:
            global_hash_ids = hash_provider(lineage_id, turn)
            if global_hash_ids is None:
                raise ValueError(
                    f"missing selected hash_ids for lineage={lineage_id} turn={turn}"
                )
            blocks = materializer.materialize_record(lineage_id, global_hash_ids)
            feature["session_namespaced_block_ids"] = [
                block.namespaced_block_id for block in blocks
            ]
        events.append(WorkloadEvent(
            line_msg_index=index,
            lineage_id=lineage_id,
            lifecycle_generation=1,
            event_type=EventType.TURN_START,
            ts=ts,
            turn=turn,
            request_plan=_request_plan_for_turn(mt),
            feature=feature,
        ))
    h = hashlib.sha256()
    parts: List[str] = []
    for event in events:
        line = json.dumps(
            event.to_runtime_contract_dict(), sort_keys=True, separators=(",", ":")
        )
        h.update(line.encode("utf-8"))
        h.update(b"\n")
        parts.append(line)
    stream_text = "\n".join(parts) + ("\n" if parts else "")
    return events, h.hexdigest(), len(events), stream_text


def assemble_slice(
    slice_class: str,
    slice_lineages: List[Lineage],
    trace_key: str,
    file_sha256: str,
    source_window_ts: Tuple[float, float],
    selection_rule: str,
    seed: int,
    time_dilation: float,
    frozen_cal: FrozenCalibration,
    calibration_window: Optional[Tuple[float, float]] = None,
    evaluation_window: Optional[Tuple[float, float]] = None,
    hash_provider=None,
    return_event_stream: bool = False,
):
    """Build one frozen manifest and, optionally, its persisted event stream.

    The persisted stream is fail-closed: every TURN_START must carry exactly
    its trace-derived block_count worth of session-namespaced block IDs and a
    complete per-turn request_plan. TTL lifecycle projections are not emitted
    here because gate 1A has no real completion signal.
    """
    if time_dilation != 1.0:
        raise ValueError("time_dilation must be 1.0 in gate 1A (recording-only)")
    trace_span = source_window_ts
    if calibration_window is None:
        calibration_window = trace_span
    if evaluation_window is None:
        evaluation_window = trace_span
    mat = Materializer()
    events, stream_sha, n_events, stream_text = _emit_event_stream_full(
        slice_lineages, frozen_cal.frozen_ttl_seconds, mat, hash_provider
    )
    if return_event_stream:
        if slice_lineages and hash_provider is None:
            raise ValueError(
                "persisted GT-trace-1A slices require selected-turn hash materialization"
            )
        expected_turns = {
            (lineage.lineage_id, mt.turn)
            for lineage in slice_lineages for mt in lineage.turns
        }
        validate_persistable_event_stream(events, expected_turns)

    max_prompt = max(
        (mt.input_length for lineage in slice_lineages for mt in lineage.turns),
        default=0,
    )
    max_target_output = max(
        (mt.output_length for lineage in slice_lineages for mt in lineage.turns),
        default=0,
    )
    max_required_context = max(
        (
            mt.input_length + mt.output_length
            for lineage in slice_lineages for mt in lineage.turns
        ),
        default=0,
    )
    if events:
        ts_vals = [event.ts for event in events]
        event_span = [min(ts_vals), max(ts_vals)]
    elif slice_lineages:
        all_ts = [mt.timestamp for lineage in slice_lineages for mt in lineage.turns]
        event_span = [min(all_ts), max(all_ts)] if all_ts else [0.0, 0.0]
    else:
        event_span = [0.0, 0.0]
    turn_count = sum(lineage.turn_count for lineage in slice_lineages)
    desc = {
        "slice_lineages": len(slice_lineages),
        "slice_turns": turn_count,
        "slice_lifespan_seconds": (
            max(lineage.lifespan for lineage in slice_lineages)
            if slice_lineages else 0.0
        ),
        "max_target_output_tokens_per_turn": max_target_output,
        "runtime_event_scope": RUNTIME_EVENT_SCOPE,
        "ttl_replay_scope": OFFLINE_TTL_REPLAY_SCOPE,
    }
    manifest = FrozenManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        slice_class=slice_class,
        trace_key=trace_key,
        trace_repo_head=TRACE_REPO_HEAD,
        trace_file=TRACE_FILES[trace_key],
        trace_file_sha256=file_sha256,
        source_window_ts=list(trace_span),
        trace_span=list(trace_span),
        calibration_window=list(calibration_window),
        evaluation_window=list(evaluation_window),
        selected_event_span=list(event_span),
        selection_rule=selection_rule,
        seed=seed,
        session_count=len(slice_lineages),
        turn_count=turn_count,
        time_dilation=time_dilation,
        prefix_mode=PREFIX_MODE,
        materialize_mode=mat.mode,
        runtime_event_scope=RUNTIME_EVENT_SCOPE,
        ttl_replay_scope=OFFLINE_TTL_REPLAY_SCOPE,
        tokenization_calibration_requirement=mat.calibration_requirement.to_dict(),
        ttl_calibration_identity={
            "algorithm": frozen_cal.algorithm,
            "target_fraction": frozen_cal.target_fraction,
            "frozen_ttl_seconds": frozen_cal.frozen_ttl_seconds,
            "achieved_fraction": frozen_cal.achieved_fraction,
            "calibration_until_ts": frozen_cal.calibration_until_ts,
            "calibration_session_count": frozen_cal.calibration_session_count,
            "calibration_turn_count": frozen_cal.calibration_turn_count,
            "evaluation_from_ts": frozen_cal.split.evaluation_from_ts,
            "evaluation_session_count": frozen_cal.split.evaluation_session_count,
            "boundary_excluded_session_count": frozen_cal.split.boundary_excluded_session_count,
            "replay_scope": OFFLINE_TTL_REPLAY_SCOPE,
        },
        max_prompt_tokens=max_prompt,
        max_required_context_tokens=max_required_context,
        max_context_footprint_tokens=max_prompt,
        event_stream_sha256=stream_sha,
        event_count=n_events,
        session_ids=sorted(lineage.lineage_id for lineage in slice_lineages),
        descriptive_stats=desc,
    )
    if return_event_stream:
        return manifest, stream_text
    return manifest
