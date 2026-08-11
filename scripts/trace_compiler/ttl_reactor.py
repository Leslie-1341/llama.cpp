#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Online TTL DEAD reactor.

THE CENTRAL CONTRACT of this module. TTL lifecycle transitions are projected
from arrivals for offline characterization. This reactor is NOT the persisted
GT-trace-1A runtime stream; that stream is assembled by sampler.py and contains
only TURN_START arrivals plus request plans.

The reactor itself decides transitions PURELY ONLINE from the event arrival
stream. It never
inspects "is this the final turn of the lineage" nor "when does the next
revisit happen". It only knows the past: turns it has already seen and
TTL timers it has already armed.

GT-trace-1A semantic scope (P1-4 / P1-5):
  * Gate 1A emits only Alibaba REQUEST ARRIVAL events: TURN_START. A
    request arrives at the runtime; lineages open a KV lifecycle at that
    moment; the TTL timer that governs DEAD/REVISIT is armed AT arrival
    (the KV is populated by the prefill of that turn; TTL residency
    starts when the request arrives, not when its hypothetical output
    completes).
  * Gate 1A does NOT emit TURN_COMPLETE. A real generation completion is
    a runtime event that gate 1A cannot observe (it has no model, no
    decode trace yet): emitting a synthetic TURN_COMPLETE at arrival
    would be "arrival masquerading as completion" and is forbidden. GT-
    trace-1B will emit the real TURN_COMPLETE once a real request
    finishes producing its output, and re-arm TTL at completion time.
  * The runtime-policy-visible feature set on TURN_START thus excludes
    output_length: at arrival the answer has not been generated, so the
    targeted output length is workload-plan data (n_predict /
    target_output_length), NEVER a policy-visible runtime feature. A
    future device that estimates answer length from the prompt may only
    do so from PREVIOUS completed turns' actual outputs (offline
    statistic named inter-arrival / revisit gap, never forward).

Lifecycle (online, no oracle):
  * At a lineage's first TURN_START: a KV lifecycle begins (generation 1)
    and a TTL timer is armed with expiry = arrival_ts + frozen_ttl.
  * If the SAME lineage's next TURN_START arrives strictly BEFORE the
    armed expiry -> REVISIT (KV state still resident and reused); the
    timer is cancelled and a fresh one armed from the new arrival ts.
  * If the armed TTL elapses with no revisiting TURN_START -> DEAD event
    is emitted; the lineage is tombstoned.
  * After DEAD, the next TURN_START of that lineage -> COLD_RESTART +
    a new lifecycle_generation, treated as a fresh (cold) lifecycle.

DEAD events come ONLY from time passing (the end-of-stream sweep or a
later arrival that crosses the Armed expiry), never from any "final
turn" oracle.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .lineage import Lineage, MinimalTurn


class EventType(str, Enum):
    TURN_START = "TURN_START"
    # TURN_COMPLETE is NOT emitted by gate 1A (see module docstring). The
    # enum member is retained so existing tests that reference it can be
    # migrated; reactor.feed() no longer emits it.
    TURN_COMPLETE = "TURN_COMPLETE"
    REVISIT = "REVISIT"
    TTL_EXPIRY = "TTL_EXPIRY"
    DEAD = "DEAD"
    COLD_RESTART = "COLD_RESTART"


@dataclass
class WorkloadEvent:
    """One emitted runtime-visible event.

    `feature` carries ONLY runtime-visible fields (the future-field guard
    is run on it). Offline ground truth (final-turn flag, next-revisit
    time, etc.) must NEVER appear here.
    """

    line_msg_index: int  # monotonic emission order
    lineage_id: int
    lifecycle_generation: int  # 1 for the first life, 2 after first cold restart, ...
    event_type: EventType
    ts: float  # absolute timestamp this event occurs at
    turn: int
    # Frozen workload data is carried beside, never inside, policy features.
    request_plan: Dict[str, object] = field(default_factory=dict)
    feature: Dict[str, object] = field(default_factory=dict)

    def to_runtime_contract_dict(self) -> Dict[str, object]:
        """The exact shape a future runtime consumes. Asserts no future
        fields are present (defensive)."""
        from .common import assert_no_future_fields  # local import to avoid cycle
        assert_no_future_fields(self.feature)
        return {
            "line_msg_index": self.line_msg_index,
            "lineage_id": self.lineage_id,
            "lifecycle_generation": self.lifecycle_generation,
            "event_type": self.event_type.value,
            "ts": self.ts,
            "turn": self.turn,
            "request_plan": dict(self.request_plan),
            "feature": dict(self.feature),
        }


@dataclass
class _LineageState:
    lifecycle_generation: int = 1
    alive: bool = False  # True once a turn has started and not yet DEAD
    armed_expiry: Optional[float] = None  # armed TTL timer, or None
    last_turn: int = 0
    # First-turn timestamp of the CURRENT lifecycle generation. Used only to
    # emit a *past-derived* relative_arrival_offset; never a future look.
    lifecycle_start_ts: Optional[float] = None


@dataclass
class TTLReactionStats:
    turn_start: int = 0
    # gate 1A does not emit TURN_COMPLETE; field kept for schema continuity
    # and populated only by callers that later migrate to 1B events, never
    # by reactor.feed() in 1A.
    turn_complete: int = 0
    revisit: int = 0
    ttl_expiry: int = 0
    dead: int = 0
    cold_restart: int = 0
    distinct_lineages: int = 0
    lineages_with_revisit: int = 0
    lineages_that_died: int = 0
    lineages_that_cold_restarted: int = 0


class TTLReactor:
    """Process a globally time-ordered arrival stream into lifecycle events.

    Inputs must be ordered by (ts, lineage_id, turn); the reactor does not
    re-sort. It maintains, per lineage_id, the live/tombstoned lifecycle
    state and at most one armed TTL timer.
    """

    def __init__(self, frozen_ttl_seconds: float) -> None:
        if frozen_ttl_seconds <= 0:
            raise ValueError("frozen TTL must be > 0 seconds")
        self.ttl = float(frozen_ttl_seconds)
        # Per lineage state.
        self._state: Dict[int, _LineageState] = {}
        # Min-heap of (expiry_ts, lineage_id) for timers currently armed.
        self._timers: List[Tuple[float, int]] = []
        self._heap_dirty = False  # timers may be cancelled by revisit
        self._msg_index = 0
        self.stats = TTLReactionStats()

    # -- emit helpers --------------------------------------------------------
    def _next_index(self) -> int:
        self._msg_index += 1
        return self._msg_index

    def _emit(
        self,
        lineage_id: int,
        gen: int,
        event_type: EventType,
        ts: float,
        turn: int,
        feature: Dict[str, object],
    ) -> WorkloadEvent:
        ev = WorkloadEvent(
            line_msg_index=self._next_index(),
            lineage_id=lineage_id,
            lifecycle_generation=gen,
            event_type=event_type,
            ts=ts,
            turn=turn,
            feature=feature,
        )
        # bookkeeping
        s = self.stats
        if event_type is EventType.TURN_START:
            s.turn_start += 1
        elif event_type is EventType.TURN_COMPLETE:
            s.turn_complete += 1
        elif event_type is EventType.REVISIT:
            s.revisit += 1
        elif event_type is EventType.TTL_EXPIRY:
            s.ttl_expiry += 1
        elif event_type is EventType.DEAD:
            s.dead += 1
        elif event_type is EventType.COLD_RESTART:
            s.cold_restart += 1
        return ev

    def _get_state(self, lineage_id: int) -> _LineageState:
        st = self._state.get(lineage_id)
        if st is None:
            st = _LineageState()
            self._state[lineage_id] = st
            self.stats.distinct_lineages += 1
        return st

    def _cancel_timer(self, lineage_id: int) -> None:
        """Mark the armed timer for a lineage as cancelled.

        We don't remove from the heap (O(n)); instead we lazy-delete by
        ignoring stale entries at sweep time (checked against st.armed_expiry).
        """
        st = self._state[lineage_id]
        # The canonical marker: stash sentinel so stale heap entry is rejected.
        st.armed_expiry = None

    def _sweep_to(self, now_ts: float) -> Iterator[WorkloadEvent]:
        """Fire any armed timers whose expiry <= now_ts, ignoring stale ones."""
        while self._timers:
            expiry, lid = self._timers[0]
            if expiry > now_ts:
                break
            heapq.heappop(self._timers)
            st = self._state.get(lid)
            if st is None or st.armed_expiry is not expiry:
                # stale (cancelled by a revisit or superseded) -> drop silently
                continue
            # Fire TTL expiry -> DEAD.
            st.armed_expiry = None
            yield self._emit(
                lineage_id=lid,
                gen=st.lifecycle_generation,
                event_type=EventType.TTL_EXPIRY,
                ts=expiry,
                turn=st.last_turn,
                feature={
                    "lineage_id": lid,
                    "expired_after_seconds": self.ttl,
                    "last_turn": st.last_turn,
                },
            )
            yield self._emit(
                lineage_id=lid,
                gen=st.lifecycle_generation,
                event_type=EventType.DEAD,
                ts=expiry,
                turn=st.last_turn,
                feature={
                    "lineage_id": lid,
                    "ttl_seconds": self.ttl,
                    "last_turn": st.last_turn,
                },
            )
            st.alive = False
            self.stats.lineages_that_died += 1

    # -- public driver --------------------------------------------------------
    def feed(
        self,
        arrivals: Iterable[Tuple[float, int, int, MinimalTurn]],
    ) -> Iterator[WorkloadEvent]:
        """Drive the reactor with an ordered arrival stream.

        Each arrival = (ts, lineage_id, turn, minimal_turn). The reactor
        first sweeps armed timers up to ts (emitting TTL_EXPIRY/DEAD for
        lineages whose TTL elapsed with no revisit), then processes the
        new arrival as TURN_START (with REVISIT/COLD_RESTART annotation per
        the lifecycle state). This reactor is used for offline projected
        characterization in gate 1A; the persisted runtime stream is
        arrival-only and is assembled separately by sampler.py.
        """
        for ts, lineage_id, turn, mt in arrivals:
            # 1) expire any timers <= ts (time has passed; no future info)
            yield from self._sweep_to(ts)
            st = self._get_state(lineage_id)
            gen = st.lifecycle_generation
            if not st.alive:
                # either first turn ever, or revival after DEAD.
                if st.last_turn > 0:
                    # reviving a tombstoned lineage -> cold restart
                    gen = st.lifecycle_generation + 1
                    st.lifecycle_generation = gen
                    self.stats.lineages_that_cold_restarted += 1
                    yield self._emit(
                        lineage_id=lineage_id,
                        gen=gen,
                        event_type=EventType.COLD_RESTART,
                        ts=ts,
                        turn=turn,
                        feature={
                            "lineage_id": lineage_id,
                            "prev_generation": gen - 1,
                            "prev_last_turn": st.last_turn,
                        },
                    )
                # Fresh lifecycle generation begins at this turn: reset the
                # lifecycle clock that relative_arrival_offset is measured from.
                st.lifecycle_start_ts = ts
            else:
                # lineage is alive: this arrival revisits a live KV state.
                # Cancel the armed TTL timer.
                self._cancel_timer(lineage_id)
                self.stats.lineages_with_revisit += 1
                yield self._emit(
                    lineage_id=lineage_id,
                    gen=gen,
                    event_type=EventType.REVISIT,
                    ts=ts,
                    turn=turn,
                    feature={
                        "lineage_id": lineage_id,
                        "revisits_after_turn": st.last_turn,
                    },
                )
            # relative_arrival_offset is past-derived: seconds since the
            # current lifecycle's first turn. Available without any future look.
            rel = ts - (st.lifecycle_start_ts if st.lifecycle_start_ts is not None else ts)
            # Gate 1A emits ONLY the REQUEST ARRIVAL (TURN_START); no
            # TURN_COMPLETE (real generation completion is invisible to 1A;
            # GT-trace-1B will emit it after real decode produces output and
            # re-arm TTL at completion time). At arrival the runtime prefill
            # populates the KV cache and a TTL timer is armed from this ts.
            yield self._emit(
                lineage_id=lineage_id,
                gen=gen,
                event_type=EventType.TURN_START,
                ts=ts,
                turn=turn,
                feature={
                    "lineage_id": lineage_id,
                    "turn": turn,
                    "relative_arrival_offset": rel,
                    "timestamp_abs": ts,
                    "input_length": mt.input_length,
                    "type": mt.type,
                    "block_count": mt.block_count,
                    # NO output_length here: at arrival the answer has not
                    # been generated. target_output_length / n_predict lives
                    # in the workload plan, never as a TURN_START policy-
                    # visible feature (P1-5).
                    # session_namespaced_block_ids is added by the
                    # materializer pass, not this reactor (reactor stays
                    # free of materialization concerns).
                },
            )
            st.alive = True
            st.last_turn = turn
            # Arm a fresh TTL timer AT ARRIVAL. The KV is resident from the
            # prefill of this turn; residency clock starts at arrival_ts and
            # a revisit before expiry re-uses that KV.
            st.armed_expiry = ts + self.ttl
            heapq.heappush(self._timers, (st.armed_expiry, lineage_id))

    def drain(self) -> Iterator[WorkloadEvent]:
        """At end-of-stream, sweep all remaining timers at +infinity.

        This is the ONLY way the final turn of a lineage transitions to
        DEAD: by time passing beyond its TTL. No oracle is consulted.
        """
        yield from self._sweep_to(float("inf"))

    # -- structured helpers ---------------------------------------------------
    @staticmethod
    def arrivals_from_lineages(
        lineages: Iterable[Lineage],
    ) -> List[Tuple[float, int, int, MinimalTurn]]:
        """Flatten accepted lineages into the globally time-ordered arrival
        stream the reactor consumes. Sorting by ts (then lineage_id, turn
        for determinism) is allowed: it is offline ordering, not a future
        look into any single lineage's intent.
        """
        out: List[Tuple[float, int, int, MinimalTurn]] = []
        for lin in lineages:
            for mt in lin.turns:
                out.append((mt.timestamp, lin.lineage_id, mt.turn, mt))
        out.sort(key=lambda x: (x[0], x[1], x[2]))
        return out
