#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic timestamp-faithful multi-session replay contract.

This module is deliberately independent of Alibaba field names.  It accepts the
model-bound GT-trace-1B-A transcript contract or a small controlled fixture with
the same session/turn/request shape, then provides deterministic scheduling,
explicit slot ownership, and fail-closed workload-fidelity checks.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


class ReplayError(ValueError):
    pass


class ReplayFidelityError(ReplayError):
    pass


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _strict_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ReplayError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ReplayError(f"{label} must be lowercase SHA-256")
    return value


@dataclass(frozen=True)
class ReplayEvent:
    event_seq: int
    logical_session_id: str
    lineage_id: int
    turn: int
    planned_ts_us: int
    prompt_tokens: tuple[int, ...]
    prompt_sha256: str
    n_predict: int
    source: str

    @property
    def request_id(self) -> str:
        return f"replay_{self.logical_session_id}_{self.turn}"


@dataclass(frozen=True)
class ReplaySession:
    logical_session_id: str
    lineage_id: int
    turns: tuple[ReplayEvent, ...]
    context_eligible: bool = True
    eligibility_reason: str | None = None


@dataclass(frozen=True)
class ReplayPlan:
    schema: str
    source_path: str
    source_sha256: str
    time_dilation: float
    n_parallel: int
    sessions: tuple[ReplaySession, ...]
    events: tuple[ReplayEvent, ...]
    excluded_sessions: tuple[dict[str, Any], ...]
    planned_arrival_origin_us: int

    @property
    def planned_session_count(self) -> int:
        return len(self.sessions)

    @property
    def planned_turn_count(self) -> int:
        return len(self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "time_dilation": self.time_dilation,
            "n_parallel": self.n_parallel,
            "planned_arrival_origin_us": self.planned_arrival_origin_us,
            "planned_session_count": self.planned_session_count,
            "planned_turn_count": self.planned_turn_count,
            "session_ids": [s.logical_session_id for s in self.sessions],
            "excluded_sessions": list(self.excluded_sessions),
            "events": [
                {
                    "event_seq": e.event_seq,
                    "logical_session_id": e.logical_session_id,
                    "lineage_id": e.lineage_id,
                    "turn": e.turn,
                    "planned_ts_us": e.planned_ts_us,
                    "prompt_tokens": list(e.prompt_tokens),
                    "prompt_sha256": e.prompt_sha256,
                    "n_predict": e.n_predict,
                    "request_id": e.request_id,
                }
                for e in self.events
            ],
        }


def _prompt_sha(tokens: Iterable[int]) -> str:
    return _sha256_json(list(tokens))


def _normalize_turn(raw: dict[str, Any], source: str, event_seq: int, *, session_id: str, lineage_id: int) -> ReplayEvent:
    turn = _strict_int(raw.get("turn"), f"{source}.turn", 1)
    timestamp = raw.get("timestamp", raw.get("planned_ts_us"))
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(float(timestamp)):
        raise ReplayError(f"{source}.timestamp must be finite")
    planned_ts_us = int(round(float(timestamp) * (1_000_000 if isinstance(timestamp, float) else 1)))
    if planned_ts_us < 0:
        raise ReplayError(f"{source}.timestamp must be non-negative")
    tokens = raw.get("prompt_tokens")
    if not isinstance(tokens, list) or not tokens or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in tokens):
        raise ReplayError(f"{source}.prompt_tokens must be a non-empty integer array")
    prompt_sha = raw.get("prompt_sha256")
    computed = _prompt_sha(tokens)
    if prompt_sha is not None and prompt_sha != computed:
        raise ReplayError(f"{source}.prompt_sha256 mismatch")
    n_predict = raw.get("n_predict", raw.get("request_plan", {}).get("n_predict"))
    n_predict = _strict_int(n_predict, f"{source}.n_predict", 0)
    lineage = _strict_int(raw.get("lineage_id", lineage_id), f"{source}.lineage_id", 0)
    return ReplayEvent(event_seq=event_seq, logical_session_id=session_id, lineage_id=lineage,
                       turn=turn, planned_ts_us=planned_ts_us, prompt_tokens=tuple(tokens),
                       prompt_sha256=computed, n_predict=n_predict, source=source)


def _load_json(path: Path) -> tuple[Any, str]:
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot load replay JSON {path}: {exc}") from exc
    return value, _sha256_bytes(data)


def _validate_session_turns(session_id: str, turns: tuple[ReplayEvent, ...]) -> None:
    expected_turns = list(range(1, len(turns) + 1))
    actual_turns = [event.turn for event in turns]
    if actual_turns != expected_turns:
        raise ReplayError(
            f"session {session_id} turns must be contiguous from one: {actual_turns}"
        )
    timestamps = [event.planned_ts_us for event in turns]
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ReplayError(f"session {session_id} timestamps move backwards")


def _load_fixture(path: Path, n_parallel: int, selected: set[str] | None) -> ReplayPlan:
    raw, source_sha = _load_json(path)
    if not isinstance(raw, dict) or raw.get("schema") not in {"generic-replay/v1", "gt-trace-1b-b/v1"}:
        raise ReplayError("controlled replay fixture schema is unsupported")
    sessions_raw = raw.get("sessions")
    if not isinstance(sessions_raw, list) or not sessions_raw:
        raise ReplayError("controlled replay fixture has no sessions")
    sessions: list[ReplaySession] = []
    excluded: list[dict[str, Any]] = []
    event_seq = 0
    for index, item in enumerate(sessions_raw):
        if not isinstance(item, dict):
            raise ReplayError(f"fixture.sessions[{index}] is not an object")
        sid = item.get("logical_session_id")
        if not isinstance(sid, str) or not sid:
            raise ReplayError(f"fixture.sessions[{index}].logical_session_id is invalid")
        if selected is not None and sid not in selected:
            continue
        eligible = item.get("context_eligible", True)
        if not isinstance(eligible, bool):
            raise ReplayError(f"fixture session {sid} context_eligible is invalid")
        if not eligible:
            excluded.append({"logical_session_id": sid, "reason": item.get("eligibility_reason", "context_ineligible")})
            continue
        lineage = _strict_int(item.get("lineage_id", index), f"fixture session {sid}.lineage_id", 0)
        turns_raw = item.get("turns")
        if not isinstance(turns_raw, list) or not turns_raw:
            raise ReplayError(f"fixture session {sid} has no turns")
        turns = tuple(_normalize_turn(turn, f"fixture.sessions[{index}].turns", event_seq + i,
                                      session_id=sid, lineage_id=lineage)
                      for i, turn in enumerate(turns_raw))
        _validate_session_turns(sid, turns)
        event_seq += len(turns)
        sessions.append(ReplaySession(sid, lineage, turns, True, None))
    if not sessions:
        raise ReplayError("controlled replay has no context-eligible selected session")
    events = tuple(sorted((event for session in sessions for event in session.turns),
                          key=lambda e: (e.planned_ts_us, e.logical_session_id, e.turn)))
    events = tuple(ReplayEvent(i, e.logical_session_id, e.lineage_id, e.turn, e.planned_ts_us,
                               e.prompt_tokens, e.prompt_sha256, e.n_predict, e.source)
                   for i, e in enumerate(events))
    origin = min(e.planned_ts_us for e in events)
    return ReplayPlan(raw["schema"], str(path), source_sha, 1.0, n_parallel,
                      tuple(sessions), events, tuple(excluded), origin)


def _load_transcript(path: Path, n_parallel: int, selected: set[str] | None) -> ReplayPlan:
    try:
        from trace_compiler.runtime_materialize import load_transcript
        load_transcript(path)
    except Exception as exc:
        raise ReplayError(f"frozen transcript validation failed: {exc}") from exc
    raw, source_sha = _load_json(path)
    if not isinstance(raw, dict) or raw.get("schema_version") != "gt-trace-1b-a/v1":
        raise ReplayError("replay transcript schema is unsupported")
    if raw.get("materialization_status") != "MODEL_BOUND_REAL" or raw.get("materialize_mode") != "direct_token_ids":
        raise ReplayError("replay requires MODEL_BOUND_REAL direct_token_ids transcript")
    _strict_sha(raw.get("transcript_sha256"), "transcript_sha256")
    turns_raw = raw.get("turns")
    if not isinstance(turns_raw, list) or not turns_raw:
        raise ReplayError("transcript has no turns")
    grouped: dict[str, list[ReplayEvent]] = {}
    lineage_by_sid: dict[str, int] = {}
    excluded: list[dict[str, Any]] = []
    effective_ctx = raw.get("effective_n_ctx_authority", {}).get("effective_n_ctx")
    for index, item in enumerate(turns_raw):
        if not isinstance(item, dict):
            raise ReplayError(f"transcript.turns[{index}] is not an object")
        lineage = _strict_int(item.get("lineage_id"), f"transcript.turns[{index}].lineage_id", 0)
        sid = f"lineage_{lineage}"
        if selected is not None and sid not in selected and str(lineage) not in selected:
            continue
        event = _normalize_turn({**item, "prompt_tokens": item.get("prompt_tokens")},
                                f"transcript.turns[{index}]", index,
                                session_id=sid, lineage_id=lineage)
        grouped.setdefault(sid, []).append(event)
        lineage_by_sid[sid] = lineage
    for sid in sorted(grouped):
        events = tuple(sorted(grouped[sid], key=lambda event: event.turn))
        _validate_session_turns(sid, events)
        max_required = max(len(event.prompt_tokens) + event.n_predict for event in events)
        if isinstance(effective_ctx, int) and max_required > effective_ctx:
            excluded.append({
                "logical_session_id": sid,
                "lineage_id": lineage_by_sid[sid],
                "reason": "context_ineligible",
                "max_required_context_tokens": max_required,
                "effective_n_ctx": effective_ctx,
                "whole_session": True,
            })
            del grouped[sid]
        else:
            grouped[sid] = list(events)
    if not grouped:
        raise ReplayError("replay transcript has no context-eligible selected session")
    sessions = tuple(ReplaySession(sid, lineage_by_sid[sid], tuple(sorted(events, key=lambda e: e.turn)), True, None)
                     for sid, events in sorted(grouped.items()))
    events = tuple(sorted((e for session in sessions for e in session.turns),
                          key=lambda e: (e.planned_ts_us, e.event_seq)))
    events = tuple(ReplayEvent(i, e.logical_session_id, e.lineage_id, e.turn, e.planned_ts_us,
                               e.prompt_tokens, e.prompt_sha256, e.n_predict, e.source)
                   for i, e in enumerate(events))
    origin = min(e.planned_ts_us for e in events)
    return ReplayPlan("gt-trace-1b-a/v1", str(path), source_sha, 1.0,
                      n_parallel, sessions, events, tuple(excluded), origin)


def load_replay(source: str, path: str | Path, *, n_parallel: int, time_dilation: float = 1.0,
                selected_session_ids: Iterable[str] | None = None) -> ReplayPlan:
    if isinstance(n_parallel, bool) or not isinstance(n_parallel, int) or n_parallel <= 0:
        raise ReplayError("n_parallel must be positive")
    if isinstance(time_dilation, bool) or not isinstance(time_dilation, (int, float)) or not math.isfinite(float(time_dilation)) or time_dilation < 0:
        raise ReplayError("time_dilation must be finite and non-negative")
    selected = set(selected_session_ids) if selected_session_ids is not None else None
    source_path = Path(path).resolve()
    plan = _load_transcript(source_path, n_parallel, selected) if source == "transcript" else _load_fixture(source_path, n_parallel, selected)
    return ReplayPlan(plan.schema, plan.source_path, plan.source_sha256, float(time_dilation),
                      n_parallel, plan.sessions, plan.events, plan.excluded_sessions, plan.planned_arrival_origin_us)


class SlotAdmission:
    """Explicit slot/seq binding; no modulo, eviction, or idle overwrite."""

    def __init__(self, n_parallel: int):
        if isinstance(n_parallel, bool) or not isinstance(n_parallel, int) or n_parallel <= 0:
            raise ReplayError("n_parallel must be positive")
        self.n_parallel = n_parallel
        self.free_slots = list(range(n_parallel))
        self.bindings: dict[str, dict[str, int]] = {}
        self.slot_generations = [0 for _ in range(n_parallel)]
        self.waiting: list[str] = []

    def admit(self, session_id: str, now_us: int) -> dict[str, Any]:
        if session_id in self.bindings:
            binding = self.bindings[session_id]
            return {"status": "already_live", "admitted_us": now_us, **binding}
        if not self.free_slots:
            if session_id not in self.waiting:
                self.waiting.append(session_id)
            return {"status": "queued", "admitted_us": None, "slot_id": None, "seq_id": None,
                    "runner_generation": None, "queue_position": self.waiting.index(session_id)}
        slot_id = self.free_slots.pop(0)
        self.slot_generations[slot_id] += 1
        generation = self.slot_generations[slot_id]
        binding = {"slot_id": slot_id, "seq_id": slot_id, "runner_generation": generation}
        self.bindings[session_id] = binding
        if session_id in self.waiting:
            self.waiting.remove(session_id)
        return {"status": "admitted", "admitted_us": now_us, **binding}

    def finish(self, session_id: str, now_us: int) -> dict[str, Any]:
        binding = self.bindings.pop(session_id, None)
        if binding is None:
            raise ReplayFidelityError(f"finish for non-live session {session_id}")
        slot_id = binding["slot_id"]
        if not 0 <= slot_id < self.n_parallel:
            raise ReplayFidelityError(f"slot binding out of range for {session_id}")
        self.free_slots.append(slot_id)
        self.free_slots.sort()
        return {"status": "finished", "completed_us": now_us, **binding}

    def binding(self, session_id: str) -> dict[str, int]:
        if session_id not in self.bindings:
            raise ReplayFidelityError(f"session {session_id} is not live")
        return dict(self.bindings[session_id])


def expand_schedule(plan: ReplayPlan) -> list[dict[str, Any]]:
    events = sorted(plan.events, key=lambda e: (e.planned_ts_us, e.event_seq))
    output: list[dict[str, Any]] = []
    for ordinal, event in enumerate(events):
        planned_delta = max(0, event.planned_ts_us - plan.planned_arrival_origin_us)
        planned_arrival_us = int(round(planned_delta * plan.time_dilation))
        output.append({
            "event_seq": ordinal,
            "logical_session_id": event.logical_session_id,
            "lineage_id": event.lineage_id,
            "turn": event.turn,
            "planned_ts_us": event.planned_ts_us,
            "planned_arrival_us": planned_arrival_us,
            "prompt_tokens": list(event.prompt_tokens),
            "prompt_sha256": event.prompt_sha256,
            "n_predict": event.n_predict,
            "request_id": event.request_id,
        })
    return output


def check_fidelity(plan: ReplayPlan, executed: list[dict[str, Any]], *, n_parallel: int) -> dict[str, Any]:
    planned = expand_schedule(plan)
    errors: list[str] = []
    if len(executed) != len(planned):
        errors.append(f"turn conservation mismatch planned={len(planned)} executed={len(executed)}")
    planned_sessions = {item["logical_session_id"] for item in planned}
    executed_sessions = {item.get("logical_session_id") for item in executed}
    if planned_sessions != executed_sessions:
        errors.append("planned sessions differ from executed sessions")
    seen_requests: set[str] = set()
    live_slots: dict[str, int] = {}
    actual_by_seq = sorted(executed, key=lambda item: item.get("event_seq", -1))
    expected_seqs = [item["event_seq"] for item in planned]
    actual_seqs = [item.get("event_seq") for item in actual_by_seq]
    if actual_seqs != expected_seqs:
        errors.append("event order or event sequence is not a permutation of the frozen schedule")
    dispatch_orders = [item.get("dispatch_order") for item in executed]
    if any(order is not None for order in dispatch_orders) and dispatch_orders != list(range(len(executed))):
        errors.append("dispatch order is not contiguous")
    last_turn_by_session: dict[str, int] = {}
    for item in executed:
        sid = item.get("logical_session_id")
        turn = item.get("turn")
        if not isinstance(turn, int):
            errors.append(f"session {sid} turn is invalid")
            continue
        if sid in last_turn_by_session and turn <= last_turn_by_session[sid]:
            errors.append(f"session {sid} turn dispatch order is invalid")
        last_turn_by_session[sid] = turn
    for index, (expected, actual) in enumerate(zip(planned, actual_by_seq)):
        for key in ("event_seq", "logical_session_id", "lineage_id", "turn", "planned_ts_us", "planned_arrival_us", "n_predict", "prompt_sha256", "request_id"):
            if actual.get(key) != expected[key]:
                errors.append(f"event {index} {key} drift")
        if actual.get("prompt_token_count", len(actual.get("prompt_tokens", []))) != len(expected["prompt_tokens"]):
            errors.append(f"event {index} prompt token count drift")
        request_id = actual.get("request_id")
        if not isinstance(request_id, str) or request_id in seen_requests:
            errors.append(f"duplicate or missing request at event {index}")
        else:
            seen_requests.add(request_id)
        slot_id = actual.get("slot_id")
        if isinstance(slot_id, bool) or not isinstance(slot_id, int) or not 0 <= slot_id < n_parallel:
            errors.append(f"event {index} has invalid slot_id")
        sid = actual.get("logical_session_id")
        if sid in live_slots and live_slots[sid] != slot_id:
            errors.append(f"session {sid} changed slot while live")
        live_slots.setdefault(sid, slot_id)
        if not isinstance(actual.get("http_status"), int) or actual["http_status"] != 200:
            errors.append(f"event {index} HTTP status is not 200")
        if actual.get("response_token_sha256") is None:
            errors.append(f"event {index} response token SHA is missing")
        admitted_us = actual.get("admitted_us")
        dispatched_us = actual.get("dispatched_us")
        completed_us = actual.get("completed_us")
        planned_arrival_us = expected["planned_arrival_us"]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
               for value in (admitted_us, dispatched_us, completed_us)):
            errors.append(f"event {index} timing field is invalid")
        else:
            if dispatched_us < admitted_us:
                errors.append(f"event {index} dispatch precedes admission")
            if completed_us < dispatched_us:
                errors.append(f"event {index} completion precedes dispatch")
            if actual.get("arrival_lag_us") != max(0, dispatched_us - planned_arrival_us):
                errors.append(f"event {index} arrival lag is inconsistent")
            if actual.get("admission_wait_us") != max(0, admitted_us - planned_arrival_us):
                errors.append(f"event {index} admission wait is inconsistent")
            if actual.get("service_us") != completed_us - dispatched_us:
                errors.append(f"event {index} service time is inconsistent")
    if len(seen_requests) != len(planned):
        errors.append("request loss or duplicate")
    status = "PASS" if not errors else "FAIL"
    return {
        "status": status,
        "errors": errors,
        "planned_sessions": len(planned_sessions),
        "executed_sessions": len(executed_sessions),
        "planned_turns": len(planned),
        "completed_turns": sum(1 for item in executed if item.get("completed_us") is not None),
        "request_count": len(executed),
        "duplicate_or_missing": len(seen_requests) != len(planned),
        "event_order_valid": not any("event_seq" in error for error in errors),
        "slot_binding_valid": not any("slot" in error or "session" in error for error in errors),
    }


def run_fake_schedule(plan: ReplayPlan, *, now_us: Callable[[], int] | None = None) -> list[dict[str, Any]]:
    """Deterministic dry-run protocol used by short tests; no HTTP or model evidence."""
    del now_us
    admission = SlotAdmission(plan.n_parallel)
    pending = list(expand_schedule(plan))
    next_turn = {session.logical_session_id: 1 for session in plan.sessions}
    last_turn = {session.logical_session_id: session.turns[-1].turn for session in plan.sessions}
    executed: list[dict[str, Any]] = []
    fake_clock = 0
    while pending:
        first = pending[0]
        sid = first["logical_session_id"]
        if sid in admission.bindings:
            if first["turn"] != next_turn[sid]:
                raise ReplayFidelityError(f"session {sid} turn order is not contiguous")
            decision = {"admitted_us": fake_clock}
            binding = admission.binding(sid)
        else:
            decision = admission.admit(sid, max(fake_clock, first["planned_arrival_us"]))
            if decision["status"] == "queued":
                candidates = [item for item in pending if item["logical_session_id"] in admission.bindings]
                if not candidates:
                    raise ReplayFidelityError("queue cannot drain without a live-session completion")
                first = min(candidates, key=lambda item: (item["planned_arrival_us"], item["event_seq"]))
                sid = first["logical_session_id"]
                binding = admission.binding(sid)
                decision = {"admitted_us": fake_clock}
            else:
                binding = {"slot_id": decision["slot_id"], "seq_id": decision["seq_id"],
                           "runner_generation": decision["runner_generation"]}
        dispatched = max(fake_clock, first["planned_arrival_us"])
        completed = dispatched + 1
        executed.append({**first, **binding, "dispatch_order": len(executed),
                         "cache_prompt": first["turn"] > 1, "prompt_token_count": len(first["prompt_tokens"]),
                         "claimant_epoch": "UNAVAILABLE", "physical_object_id": "UNAVAILABLE",
                         "physical_generation": "UNAVAILABLE", "admitted_us": decision["admitted_us"],
                         "dispatched_us": dispatched, "completed_us": completed,
                         "arrival_lag_us": max(0, decision["admitted_us"] - first["planned_arrival_us"]),
                         "admission_wait_us": max(0, decision["admitted_us"] - first["planned_arrival_us"]),
                         "service_us": 1, "http_status": 200, "prompt_token_count": len(first["prompt_tokens"]),
                         "response_token_sha256": _sha256_json([])})
        pending.remove(first)
        next_turn[sid] += 1
        fake_clock = completed
        if first["turn"] == last_turn[sid]:
            admission.finish(sid, completed)
    return executed
