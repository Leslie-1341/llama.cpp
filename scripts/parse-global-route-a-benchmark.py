#!/usr/bin/env python3
"""Fail-closed parser for the Global Fast-A1 Route-A harness."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from collections import Counter
from typing import Any

CASES = ("static_low", "global_route_a", "static_high")
PROTOCOL = "global_route_a_fast_a1"
PHASES = {
    "P0_preflight_identity", "P0_health", "P1_fixed_warmup", "P2_A_turn1_fill",
    "P3_A_completion_idle_release", "P4_route_a_credit_observation", "P5_B_moe_demand",
    "P6_A_turn2_exact_restore", "P7_short_steady_measurement",
}
REQUIRED_CASE_ARTIFACTS = (
    "server.stdout", "server.stderr", "requests.jsonl", "phase_events.jsonl",
    "memory_timeline.jsonl", "cleanup.json", "exit_codes.json", "result.json",
    "execution.json", "process.json", "environment.json",
)
ACTION_MARKER = "kv_pressure_unified_action"
OBSERVE_MARKER = "memory_governor_observe"
RESUME_MARKER = "kv_resume_order_event"
PAGED_MARKERS = ("KV_PAGED_PREFETCH_PHASE_CALL", "KV_PAGED_PREFETCH_BLOCK_PHASE", "KV_PAGED_PREFETCH_PIPELINE")


class InvalidArtifact(Exception):
    pass


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidArtifact(f"invalid JSON {path.name}: {exc}") from exc


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise InvalidArtifact(f"missing artifact: {path}")
    result: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InvalidArtifact(f"cannot read {path}: {exc}") from exc
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InvalidArtifact(f"invalid JSONL {path.name}:{index}: {exc}") from exc
        if not isinstance(value, dict):
            raise InvalidArtifact(f"non-object JSONL {path.name}:{index}")
        value["_line"] = index
        result.append(value)
    return result


def bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        if value.lower() in {"1", "true", "yes", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "off"}:
            return False
    return None


def int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            return None
    return None


def marker_fields(line: str, marker: str) -> dict[str, str] | None:
    occurrences = [match.start() for match in re.finditer(rf"(?<!\S){re.escape(marker)}(?!\S)", line)]
    if len(occurrences) != 1:
        return None
    words = line[occurrences[0] + len(marker):].strip().split()
    fields: dict[str, str] = {}
    for word in words:
        if "=" not in word:
            return None
        key, value = word.split("=", 1)
        if not key or not value or key in fields:
            return None
        fields[key] = value
    return fields


def marker_events(case_dir: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = case_dir / "server.stderr"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InvalidArtifact(f"cannot read {path}: {exc}") from exc
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    offset = 0
    for number, line_bytes in enumerate(raw.splitlines(keepends=True), 1):
        line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
        for marker in (OBSERVE_MARKER, ACTION_MARKER, RESUME_MARKER, *PAGED_MARKERS):
            if marker not in line:
                continue
            fields = marker_fields(line, marker)
            if fields is None:
                errors.append(f"malformed_or_duplicate_marker:{marker}:line={number}")
                continue
            records.append({"marker": marker, "fields": fields, "line": number,
                            "byte_start": offset, "byte_end": offset + len(line_bytes), "text": line})
        offset += len(line_bytes)
    return records, errors


def phase_windows(case_dir: pathlib.Path) -> list[dict[str, Any]]:
    events = read_jsonl(case_dir / "phase_events.jsonl")
    windows: list[dict[str, Any]] = []
    for event in events:
        name = event.get("name")
        if name not in PHASES:
            raise InvalidArtifact(f"unknown phase {name!r}")
        start = int_value(event.get("stderr_start"))
        end = int_value(event.get("stderr_end"))
        if start is None or end is None or end < start or start < 0:
            raise InvalidArtifact(f"invalid stderr phase window {name}")
        windows.append({"name": name, "start": start, "end": end, "status": event.get("status")})
    required_order = ["P0_preflight_identity", "P0_health", "P1_fixed_warmup", "P2_A_turn1_fill",
                      "P3_A_completion_idle_release", "P4_route_a_credit_observation", "P5_B_moe_demand",
                      "P6_A_turn2_exact_restore", "P7_short_steady_measurement"]
    names = [item["name"] for item in windows]
    if names != required_order:
        raise InvalidArtifact("phase windows are not exactly P0-P7 in order")
    previous_end = 0
    for item in windows:
        if item["start"] < previous_end:
            raise InvalidArtifact("phase windows overlap")
        previous_end = item["end"]
    if any(item["status"] != "PASS" for item in windows):
        raise InvalidArtifact("phase failure recorded")
    return windows


def phase_for_offset(windows: list[dict[str, Any]], offset: int) -> str:
    matches = [item["name"] for item in windows if item["start"] <= offset <= item["end"]]
    return matches[-1] if matches else "unknown"


def field(record: dict[str, Any], key: str, default: Any = None) -> Any:
    return record.get("fields", {}).get(key, default)


def action_ok(record: dict[str, Any]) -> bool:
    action = field(record, "action")
    attempted = bool_value(field(record, "offload_attempted"))
    return (action in {"release", "offload"} or attempted is True)


def parse_actions(case_dir: pathlib.Path, windows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    actions = []
    seen_identity: set[tuple[int, int]] = set()
    errors: list[str] = []
    for record in records:
        if record["marker"] != ACTION_MARKER:
            continue
        item = dict(record)
        item["phase"] = phase_for_offset(windows, record["byte_start"])
        if action_ok(record) and item["phase"] == "P2_A_turn1_fill":
            errors.append(f"destructive_action_during_active_fill:line={record['line']}")
        decision = int_value(field(record, "decision_id"))
        transaction = int_value(field(record, "transaction_id"))
        if decision is None or transaction is None:
            errors.append(f"action_missing_identity:line={record['line']}")
        else:
            identity = (decision, transaction)
            if identity in seen_identity:
                errors.append(f"duplicate_action_identity:{identity}")
            seen_identity.add(identity)
        if action_ok(record):
            actions.append(item)
    destructive = [item for item in actions if item["phase"] in {"P3_A_completion_idle_release", "P4_route_a_credit_observation"}]
    physical = []
    for item in destructive:
        available = bool_value(field(item, "physical_relief_available"))
        relief = int_value(field(item, "physical_relief_bytes"))
        changed = bool_value(field(item, "state_changed"))
        outcome = field(item, "outcome")
        io_failure = bool_value(field(item, "io_failure"))
        object_id = int_value(field(item, "physical_object_id"))
        generation = int_value(field(item, "physical_generation"))
        if available is True and relief is not None and relief > 0 and changed is True and outcome == "completed" and io_failure is False:
            if object_id is None or object_id <= 0 or generation is None or generation <= 0:
                errors.append(f"physical_relief_missing_identity:line={item['line']}")
            else:
                item["physical_relief_bytes_int"] = relief
                item["physical_object_id_int"] = object_id
                item["physical_generation_int"] = generation
                physical.append(item)
    identities = {(item["physical_object_id_int"], item["physical_generation_int"]) for item in physical}
    if len(identities) > 1:
        errors.append("physical_object_generation_mismatch")
    return {"actions": actions, "destructive_actions": destructive, "physical_relief": physical,
            "errors": errors, "unique_physical_identity": list(next(iter(identities))) if len(identities) == 1 else None}


def parse_observations(records: list[dict[str, Any]], windows: list[dict[str, Any]]) -> dict[str, Any]:
    observations = [item for item in records if item["marker"] == OBSERVE_MARKER]
    ledger = []
    for item in observations:
        fields = item["fields"]
        interesting = {
            key: int_value(fields[key]) for key in (
                "reallocation_confirmed_kv_physical_credit_bytes", "reallocation_confirmed_kv_physical_object_id",
                "reallocation_confirmed_kv_physical_generation", "reallocation_kv_physical_credit_earned_bytes",
                "reallocation_credit_earned_bytes", "reallocation_credit_available_bytes",
                "reallocation_credit_remaining_bytes", "reallocation_credit_spent_bytes",
                "reallocation_moe_grant_bytes", "reallocation_moe_old_budget_bytes",
                "reallocation_moe_new_budget_bytes", "reallocation_dense_grant_bytes",
                "moe_budget_bytes", "moe_resident_bytes", "moe_cache_hits", "moe_cache_misses",
                "moe_hits", "moe_evictions",
                "moe_bytes_read", "moe_prefetch_hits", "moe_prefetch_late", "moe_prefetch_unused",
            ) if key in fields}
        interesting.update({key: fields[key] for key in ("reallocation_reason", "moe_budget_action", "moe_budget_reason") if key in fields})
        interesting["line"] = item["line"]
        interesting["phase"] = phase_for_offset(windows, item["byte_start"])
        ledger.append(interesting)
    return {"observations": observations, "ledger": ledger}


def max_or_none(values: list[int]) -> int | None:
    return max(values) if values else None


def validate_ledger(case_name: str, parsed_actions: dict[str, Any], parsed_obs: dict[str, Any]) -> dict[str, Any]:
    ledger = parsed_obs["ledger"]
    errors: list[str] = list(parsed_actions["errors"])
    credit_records = [item for item in ledger if item.get("reallocation_reason") == "moe_credit_grant"]
    emergency = [item for item in ledger if item.get("reallocation_reason") == "moe_emergency_working_set"]
    startup = [item for item in ledger if item.get("moe_budget_reason") == "fast_start"]
    positive_grants = [item for item in ledger if (item.get("reallocation_moe_grant_bytes") or 0) > 0]
    dense_grants = [item for item in ledger if (item.get("reallocation_dense_grant_bytes") or 0) > 0]
    if dense_grants:
        errors.append("dense_grant_present")
    if case_name == "static_low":
        if credit_records:
            errors.append("static_low_has_route_a_credit_grant")
        if positive_grants:
            errors.append("static_low_has_moe_grant_bytes")
    if case_name == "static_high":
        if credit_records:
            errors.append("static_high_has_route_a_credit_grant")
        if positive_grants:
            errors.append("static_high_has_moe_grant_bytes")
    for item in startup + emergency:
        if (item.get("reallocation_credit_spent_bytes") or 0) > 0 or (item.get("reallocation_moe_grant_bytes") or 0) > 0:
            errors.append(f"non_live_growth_attributed_to_route_a:line={item['line']}")
    physical_confirmed = max_or_none([item.get("reallocation_confirmed_kv_physical_credit_bytes", 0) or 0 for item in ledger]) or 0
    earned = max_or_none([item.get("reallocation_credit_earned_bytes", 0) or 0 for item in ledger]) or 0
    spent = max_or_none([item.get("reallocation_credit_spent_bytes", 0) or 0 for item in ledger]) or 0
    grant = max_or_none([item.get("reallocation_moe_grant_bytes", 0) or 0 for item in ledger]) or 0
    remaining = max_or_none([item.get("reallocation_credit_remaining_bytes", 0) or 0 for item in ledger])
    available = max_or_none([item.get("reallocation_credit_available_bytes", 0) or 0 for item in ledger])
    if spent < 0 or grant < 0 or earned < 0:
        errors.append("negative_credit_ledger")
    if spent != grant:
        errors.append(f"credit_spent_grant_not_closed:{spent}!={grant}")
    if earned < spent:
        errors.append("credit_spent_exceeds_earned")
    if remaining is not None and remaining > earned:
        errors.append("credit_remaining_exceeds_earned")
    if available is not None and available > earned:
        errors.append("credit_available_exceeds_earned")
    if earned > physical_confirmed:
        errors.append("credit_earned_exceeds_confirmed_physical_credit")
    action_identity = parsed_actions.get("unique_physical_identity")
    physical_relief_actions = parsed_actions.get("physical_relief", [])
    action_identities = {(act["physical_object_id_int"], act["physical_generation_int"])
                         for act in physical_relief_actions}
    if len(action_identities) > 1:
        errors.append("physical_object_generation_mismatch")
    # Confirmation anchor: exactly one observe frame carrying a positive confirmed
    # credit bytes together with a valid confirmed object/generation identity emitted by
    # the C++ runtime at the moment it validates pending vs. live physical view identity.
    confirmation_anchors = [
        item for item in ledger
        if (item.get("reallocation_confirmed_kv_physical_credit_bytes") or 0) > 0
        and isinstance(item.get("reallocation_confirmed_kv_physical_object_id"), int)
        and item.get("reallocation_confirmed_kv_physical_object_id", 0) > 0
        and isinstance(item.get("reallocation_confirmed_kv_physical_generation"), int)
        and item.get("reallocation_confirmed_kv_physical_generation", 0) > 0
    ]
    if len(confirmation_anchors) > 1:
        errors.append("duplicate_confirmation_identity")
    confirmation_identity: tuple[int, int] | None = None
    confirmation_line: int | None = None
    if confirmation_anchors:
        anchor = confirmation_anchors[0]
        confirmation_identity = (anchor["reallocation_confirmed_kv_physical_object_id"],
                                 anchor["reallocation_confirmed_kv_physical_generation"])
        confirmation_line = anchor["line"]
    elif physical_confirmed > 0 or earned > 0 or spent > 0 or grant > 0:
        errors.append("credit_missing_physical_object_generation")
    # Contract 4: action physical identity must exactly match the confirmation identity.
    if confirmation_identity is not None and action_identities and confirmation_identity not in action_identities:
        errors.append("physical_object_generation_mismatch")
    # Source-action correlation anchors (decision_id, transaction_id) of the route-a
    # physical relief action whose identity the confirmation validated. These delimit the
    # sole attribution window; an unrelated state-changing KV action inside the window
    # breaks single-attribution and must fail closed (contract 5). No new runtime
    # transaction state is introduced: the parser reuses the action marker's existing
    # decision_id/transaction_id fields.
    if confirmation_line is not None:
        source_actions = [act for act in physical_relief_actions
                          if act["line"] <= confirmation_line
                          and (act["physical_object_id_int"], act["physical_generation_int"]) == confirmation_identity]
    else:
        source_actions = list(physical_relief_actions)
    source_action_keys = {(int_value(field(act, "decision_id")), int_value(field(act, "transaction_id")))
                          for act in source_actions}
    # Attribution window: opens at the confirmation anchor line; closes at the first
    # unrelated state-changing KV action that appears after it, or at the ledger end.
    state_changing_actions = [
        act for act in parsed_actions["actions"]
        if (field(act, "action") == "release" or field(act, "action") == "offload")
        and bool_value(field(act, "state_changed")) is True
    ]
    first_physical_line = min((act["line"] for act in physical_relief_actions), default=None)
    if confirmation_line is not None:
        window_lower = confirmation_line
        window_upper: int | float = float("inf")
        for rec in state_changing_actions:
            rec_line = rec["line"]
            if rec_line <= window_lower:
                continue
            rec_key = (int_value(field(rec, "decision_id")), int_value(field(rec, "transaction_id")))
            if rec_key not in source_action_keys:
                if rec_line < window_upper:
                    errors.append(f"duplicate_intervening_action:line={rec_line}")
                    window_upper = rec_line
        for item in ledger:
            positive = any((item.get(key) or 0) > 0 for key in (
                "reallocation_credit_earned_bytes", "reallocation_credit_spent_bytes",
                "reallocation_moe_grant_bytes"))
            if positive and not (window_lower <= item["line"] < window_upper):
                errors.append(f"credit_outside_attribution_window:line={item['line']}")
        for item in ledger:
            positive = any((item.get(key) or 0) > 0 for key in (
                "reallocation_confirmed_kv_physical_credit_bytes", "reallocation_credit_earned_bytes",
                "reallocation_credit_spent_bytes", "reallocation_moe_grant_bytes"))
            if positive and first_physical_line is not None and item["line"] <= first_physical_line:
                errors.append(f"credit_before_physical_relief:line={item['line']}")
    for item in credit_records:
        old = item.get("reallocation_moe_old_budget_bytes")
        new = item.get("reallocation_moe_new_budget_bytes")
        item_grant = item.get("reallocation_moe_grant_bytes") or 0
        item_spent = item.get("reallocation_credit_spent_bytes") or 0
        if not isinstance(old, int) or not isinstance(new, int) or not new > old:
            errors.append(f"credit_grant_budget_not_strictly_growing:line={item['line']}")
        if item_grant <= 0 or item_spent <= 0:
            errors.append(f"credit_grant_without_spend:line={item['line']}")
    return {
        "credit_records": credit_records, "emergency_records": emergency, "startup_records": startup,
        "confirmed_physical_credit_bytes": physical_confirmed, "earned_bytes": earned,
        "spent_bytes": spent, "grant_bytes": grant, "remaining_bytes": remaining,
        "available_bytes": available, "errors": errors,
    }


def validate_moe_utilization(parsed_obs: dict[str, Any], credit: dict[str, Any]) -> dict[str, Any]:
    ledger = parsed_obs["ledger"]
    demand_window = [item for item in ledger if item.get("phase") == "P5_B_moe_demand"]
    last = ledger[-1] if ledger else {}
    values = [item.get("moe_resident_bytes") for item in demand_window if isinstance(item.get("moe_resident_bytes"), int)]
    resident_growth = bool(values and max(values) > values[0])
    evidence_keys = ("moe_cache_hits", "moe_cache_misses", "moe_hits", "moe_bytes_read", "moe_prefetch_hits")
    demand_evidence = any(isinstance(item.get(key), int) and item.get(key, 0) > 0 for item in demand_window for key in evidence_keys)
    budget_growth = any(isinstance(item.get("reallocation_moe_old_budget_bytes"), int) and isinstance(item.get("reallocation_moe_new_budget_bytes"), int)
                        and item["reallocation_moe_new_budget_bytes"] > item["reallocation_moe_old_budget_bytes"] for item in credit["credit_records"])
    status = "PASS" if budget_growth and (resident_growth or demand_evidence) else "UNVERIFIED"
    return {"status": status, "budget_growth": budget_growth, "resident_growth": resident_growth,
            "demand_evidence": demand_evidence, "last_observation": last,
            "reason": "GLOBAL_ROUTE_A_UTILIZATION_UNVERIFIED" if status == "UNVERIFIED" else None}


def validate_restore(case_dir: pathlib.Path, windows: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    resumes = [item for item in records if item["marker"] == RESUME_MARKER]
    errors: list[str] = []
    state: dict[tuple[str, str], dict[str, Any]] = {}
    prefetch_count = 0
    graph_count = 0
    for item in resumes:
        phase = phase_for_offset(windows, item["byte_start"])
        action = item["fields"].get("action")
        seq = item["fields"].get("seq_id", "")
        decision = item["fields"].get("decision_id", "")
        key = (str(seq), str(decision))
        if action != "prefetch":
            errors.append(f"restore_non_prefetch_action:line={item['line']}")
            continue
        marker_phase = item["fields"].get("phase")
        transaction = item["fields"].get("transaction_id")
        epoch = item["fields"].get("claimant_epoch")
        if marker_phase == "prefetch":
            prefetch_count += 1
            if phase != "P6_A_turn2_exact_restore":
                errors.append(f"prefetch_outside_exact_restore_window:line={item['line']}")
            if item["fields"].get("outcome") != "completed":
                errors.append(f"prefetch_not_completed:line={item['line']}")
            state[key] = {"prefetch": item, "graph": False, "transaction": transaction, "epoch": epoch}
        elif marker_phase == "graph_gate":
            graph_count += 1
            if phase != "P6_A_turn2_exact_restore":
                errors.append(f"graph_gate_outside_exact_restore_window:line={item['line']}")
            allowed = bool_value(item["fields"].get("graph_allowed"))
            if allowed is not True:
                errors.append(f"graph_gate_not_allowed:line={item['line']}")
            if key not in state:
                errors.append(f"graph_gate_without_prefetch:line={item['line']}")
            else:
                prior = state[key]
                if transaction != prior.get("transaction") or epoch != prior.get("epoch"):
                    errors.append(f"restore_identity_mismatch:line={item['line']}")
                if item["fields"].get("outcome") != "completed":
                    errors.append(f"graph_gate_not_completed:line={item['line']}")
                prior["graph"] = allowed is True
        else:
            errors.append(f"unknown_restore_phase:line={item['line']}")
    for key, item in state.items():
        if not item.get("graph"):
            errors.append(f"prefetch_without_graph_gate:{key}")
    paged_calls = [item for item in records if item["marker"] == "KV_PAGED_PREFETCH_PHASE_CALL"
                   and phase_for_offset(windows, item["byte_start"]) == "P6_A_turn2_exact_restore"]
    restored_blocks = sum(int_value(item["fields"].get("restored_blocks")) or 0 for item in paged_calls)
    requested_blocks = sum(int_value(item["fields"].get("requested_blocks")) or 0 for item in paged_calls)
    if not paged_calls or requested_blocks <= 0 or restored_blocks <= 0 or requested_blocks != restored_blocks:
        errors.append("invalid_paged_restore_block_counts")
    offloaded_blocks = sum(int_value(item["fields"].get("blocks")) or 0 for item in records
                           if item["marker"] == ACTION_MARKER
                           and phase_for_offset(windows, item["byte_start"]) in {"P3_A_completion_idle_release", "P4_route_a_credit_observation"}
                           and item["fields"].get("action") in {"release", "offload"})
    if offloaded_blocks > 0 and restored_blocks != offloaded_blocks:
        errors.append(f"restore_blocks_do_not_match_offload:{restored_blocks}!={offloaded_blocks}")
    if prefetch_count == 0 or graph_count == 0:
        errors.append("missing_exact_restore_markers")
    return {"prefetch_count": prefetch_count, "graph_gate_count": graph_count,
            "paged_calls": len(paged_calls), "requested_blocks": requested_blocks,
            "restored_blocks": restored_blocks, "offloaded_blocks": offloaded_blocks, "errors": errors}


def validate_requests(case_dir: pathlib.Path) -> dict[str, Any]:
    requests = read_jsonl(case_dir / "requests.jsonl")
    errors: list[str] = []
    for item in requests:
        if item.get("http_status") != 200 or item.get("transport_error"):
            errors.append(f"request_failed:{item.get('label')}:{item.get('http_status')}")
        request = item.get("request")
        if isinstance(request, dict) and item.get("request_sha256"):
            expected = sha_bytes(json.dumps(request, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8"))
            if item.get("request_sha256") != expected:
                errors.append(f"request_hash_mismatch:{item.get('label')}")
        tokens = item.get("response_tokens")
        if isinstance(tokens, list) and all(isinstance(value, int) and not isinstance(value, bool) for value in tokens):
            expected = sha_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))
            if item.get("response_tokens_sha256") != expected:
                errors.append(f"response_token_hash_mismatch:{item.get('label')}")
    labels = Counter(str(item.get("label")) for item in requests)
    for required in ("A_turn1", "A_turn2"):
        if labels[required] != 1:
            errors.append(f"request_count_{required}={labels[required]}")
    return {"requests": requests, "errors": errors,
            "turn2": next((item for item in requests if item.get("label") == "A_turn2"), None),
            "turn1": next((item for item in requests if item.get("label") == "A_turn1"), None)}


def validate_case(root: pathlib.Path, case_name: str) -> dict[str, Any]:
    case = root / case_name
    if not case.is_dir():
        raise InvalidArtifact(f"missing case directory: {case_name}")
    missing = [name for name in REQUIRED_CASE_ARTIFACTS if not (case / name).is_file()]
    if missing:
        raise InvalidArtifact(f"{case_name} missing artifacts: {', '.join(missing)}")
    manifest_result = read_json(case / "result.json")
    if manifest_result.get("status") != "complete":
        raise InvalidArtifact(f"{case_name} runner status is not complete")
    cleanup = read_json(case / "cleanup.json")
    exits = read_json(case / "exit_codes.json")
    if cleanup.get("completed") is not True or exits.get("server", {}).get("residual_process"):
        raise InvalidArtifact(f"{case_name} cleanup incomplete")
    timeline = read_jsonl(case / "memory_timeline.jsonl")
    timeline_errors: list[str] = []
    for item in timeline:
        if int_value(item.get("monotonic_ns")) is None or not isinstance(item.get("phase"), str):
            timeline_errors.append(f"invalid_timeline_sample:{item.get('_line')}")
        if not isinstance(item.get("process"), dict) or not isinstance(item.get("cgroup"), dict):
            timeline_errors.append(f"incomplete_timeline_sample:{item.get('_line')}")
    if not timeline:
        timeline_errors.append("empty_memory_timeline")
    execution = read_json(case / "execution.json")
    process = read_json(case / "process.json")
    identity_errors: list[str] = []
    pid = int_value(process.get("pid"))
    starttime = int_value(process.get("starttime_ticks"))
    cmdline = process.get("cmdline")
    argv = execution.get("argv")
    if pid is None or pid <= 0 or starttime is None or starttime <= 0:
        identity_errors.append("invalid_process_identity")
    if not isinstance(cmdline, list) or not cmdline or not all(isinstance(item, str) for item in cmdline):
        identity_errors.append("invalid_process_cmdline")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        identity_errors.append("missing_execution_argv")
    elif isinstance(cmdline, list) and cmdline and cmdline[0] != argv[0]:
        identity_errors.append("process_binary_mismatch")
    windows = phase_windows(case)
    records, marker_errors = marker_events(case)
    actions = parse_actions(case, windows, records)
    obs = parse_observations(records, windows)
    credit = validate_ledger(case_name, actions, obs)
    restore = validate_restore(case, windows, records)
    requests = validate_requests(case)
    errors = marker_errors + timeline_errors + identity_errors + credit["errors"] + restore["errors"] + requests["errors"]
    return {"case": case_name, "windows": windows, "records": records, "actions": actions,
            "observations": obs, "credit": credit, "restore": restore, "requests": requests,
            "timeline_samples": len(timeline), "moe_utilization": validate_moe_utilization(obs, credit), "errors": errors,
            "environment": read_json(case / "environment.json"), "execution": execution,
            "process": process, "result": manifest_result}


def compare_common_config(root: pathlib.Path, cases: dict[str, dict[str, Any]], spec: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    execution_ids = {}
    for name, parsed in cases.items():
        execution = parsed["execution"]
        execution_ids[name] = {key: execution.get(key) for key in ("binary", "model", "cgroup_path")}
    if len({json.dumps(value, sort_keys=True) for value in execution_ids.values()}) != 1:
        errors.append("common_execution_identity_mismatch")
    envs = {name: parsed["environment"].get("effective", {}) for name, parsed in cases.items()}
    all_keys = set().union(*(set(value) for value in envs.values()))
    required_common = {
        "LLAMA_KV_PAGED", "LLAMA_KV_PAGED_INGRAPH", "LLAMA_KV_PAGED_SWAP",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY", "LLAMA_KV_PAGED_MINCORE",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES",
        "LLAMA_KV_RESIDENT_TARGET_BYTES", "LLAMA_KV_RESIDENT_TARGET_SOURCE",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_CONFIRM",
    }
    for key in sorted(required_common):
        observed = {name: envs[name].get(key) for name in CASES}
        if any(value is None for value in observed.values()):
            errors.append(f"missing_common_environment_key:{key}:{observed}")
    target = str(spec.get("kv_resident_target_bytes"))
    for key, expected in (("LLAMA_KV_RESIDENT_TARGET_BYTES", target), ("LLAMA_KV_RESIDENT_TARGET_SOURCE", "env_static"),
                          ("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", target),
                          ("LLAMA_MEMORY_GOVERNOR_REALLOCATION_CONFIRM", "1")):
        observed = {name: envs[name].get(key) for name in CASES}
        if any(value != expected for value in observed.values()):
            errors.append(f"target_contract_mismatch:{key}:{observed}")
    allowed = {
        "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER", "LLAMA_MEMORY_GOVERNOR_REALLOCATION",
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_APPLY_MOE", "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC",
        "LLAMA_MEMORY_GOVERNOR_MOE_WARM_MB", "LLAMA_MEMORY_GOVERNOR_MOE_MAX_MB",
    }
    config_diffs: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        values = {name: envs[name].get(key) for name in CASES}
        if len(set(values.values())) > 1:
            config_diffs[key] = values
            if key not in allowed:
                errors.append(f"unexpected_case_environment_diff:{key}")
    expected = {
        "LLAMA_MEMORY_GOVERNOR_GLOBAL_OPTIMIZER": {"static_low": "0", "global_route_a": "1", "static_high": "0"},
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION": {"static_low": "0", "global_route_a": "1", "static_high": "0"},
        "LLAMA_MEMORY_GOVERNOR_REALLOCATION_APPLY_MOE": {"static_low": "0", "global_route_a": "1", "static_high": "0"},
        "LLAMA_MEMORY_GOVERNOR_MOE_BUDGET_DYNAMIC": {"static_low": "0", "global_route_a": "1", "static_high": "0"},
    }
    for key, values in expected.items():
        observed = {name: envs[name].get(key) for name in CASES}
        if observed != values:
            errors.append(f"unexpected_mode_contract:{key}:{observed}")
    params = manifest.get("parameters", {})
    for key in ("kv_resident_target_bytes", "initial_moe_budget_bytes", "static_high_moe_budget_bytes", "memory_cap_bytes"):
        if params.get(key) != spec.get(key):
            errors.append(f"manifest_spec_mismatch:{key}")
    return {"errors": errors, "execution_ids": execution_ids, "config_diffs": config_diffs}


def compare_correctness(cases: dict[str, dict[str, Any]], spec: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    turn2 = {name: cases[name]["requests"]["turn2"] for name in CASES}
    if any(item is None for item in turn2.values()):
        errors.append("missing_turn2_response")
        return {"status": "INVALID", "errors": errors}
    expected_request = spec.get("workload", {}).get("session_a_turn2")
    expected_hash = None
    if isinstance(expected_request, dict):
        expected_hash = sha_bytes(json.dumps(expected_request, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8"))
    request_hashes = {name: item.get("request_sha256") for name, item in turn2.items()}
    if len(set(request_hashes.values())) != 1:
        errors.append("matched_continuation_request_identity_mismatch")
    for name, item in turn2.items():
        if expected_hash is None or item.get("request_sha256") != expected_hash or item.get("request") != expected_request:
            errors.append(f"turn2_request_not_canonical:{name}")
    token_hashes = {}
    for name, item in turn2.items():
        tokens = item.get("response_tokens")
        if isinstance(tokens, list) and all(isinstance(value, int) and not isinstance(value, bool) for value in tokens):
            token_hashes[name] = sha_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))
        else:
            token_hashes[name] = None
            errors.append(f"missing_response_tokens:{name}")
    if len(set(token_hashes.values())) != 1:
        errors.append("exact_token_output_drift")
    return {"status": "PASS" if not errors else "INVALID", "errors": errors,
            "request_sha256": request_hashes, "token_sha256": token_hashes}


def parse_artifact(root: pathlib.Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    spec = read_json(root / "case_spec.json")
    if manifest.get("runner_status") != "run_complete":
        raise InvalidArtifact("runner status is not run_complete")
    if manifest.get("protocol") != PROTOCOL:
        raise InvalidArtifact("wrong protocol")
    if manifest.get("case_names") != list(CASES):
        raise InvalidArtifact("case names are not exactly static_low/global_route_a/static_high")
    cases: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for name in CASES:
        try:
            cases[name] = validate_case(root, name)
        except InvalidArtifact as exc:
            errors.append(str(exc))
    if len(cases) != len(CASES):
        verdict = "INVALID"
        summary = {"protocol": PROTOCOL, "errors": errors, "cases": list(cases)}
        dump_outputs(root, summary, {"verdict": verdict, "errors": errors})
        return {"summary": summary, "verdict": {"verdict": verdict, "errors": errors}}
    common = compare_common_config(root, cases, spec, manifest)
    correctness = compare_correctness(cases, spec)
    errors.extend(common["errors"])
    for name, parsed in cases.items():
        if parsed["errors"]:
            errors.extend([f"{name}:{item}" for item in parsed["errors"]])
    errors.extend(correctness["errors"])
    global_case = cases["global_route_a"]
    global_credit = global_case["credit"]
    mechanism_pass = (
        not global_case["errors"] and global_credit["confirmed_physical_credit_bytes"] > 0
        and bool(global_case["actions"]["physical_relief"])
        and global_credit["earned_bytes"] > 0 and global_credit["spent_bytes"] > 0
        and global_credit["grant_bytes"] > 0 and global_credit["spent_bytes"] == global_credit["grant_bytes"]
        and global_credit["credit_records"]
    )
    utilization = global_case["moe_utilization"]
    static_low_valid = not cases["static_low"]["errors"]
    static_high_valid = not cases["static_high"]["errors"]
    utilization_status = "GLOBAL_ROUTE_A_UTILIZATION_PASS" if utilization["status"] == "PASS" else "GLOBAL_ROUTE_A_UTILIZATION_UNVERIFIED"
    dirty = manifest.get("git", {}).get("capture_mode") == "DIRTY_DEV_ONLY"
    final_ready = static_low_valid and static_high_valid and mechanism_pass and correctness["status"] == "PASS" and utilization["status"] == "PASS"
    verdict = "GLOBAL_FAST_A1_READY_FOR_INTEGRATION" if final_ready else "NOT_READY"
    summary = {
        "protocol": PROTOCOL, "protocol_version": manifest.get("protocol_version"), "manifest": manifest,
        "case_spec": spec, "cases": {name: summarize_case(parsed) for name, parsed in cases.items()},
        "common_config": common, "correctness": correctness, "dirty_dev_only": dirty,
        "gates": {
            "static_low": "STATIC_LOW_VALID" if static_low_valid else "INVALID",
            "route_a_mechanism": "GLOBAL_ROUTE_A_MECHANISM_PASS" if mechanism_pass else "INVALID",
            "route_a_utilization": utilization_status,
            "static_high": "STATIC_HIGH_VALID" if static_high_valid else "INVALID",
            "cross_case_correctness": "CROSS_CASE_CORRECTNESS_PASS" if correctness["status"] == "PASS" else "INVALID",
        }, "errors": errors,
    }
    verdict_obj = {"verdict": verdict, "errors": errors, "dirty_dev_only": dirty,
                   "performance_claims_allowed": not dirty and final_ready,
                   "route_a_mechanism": "PASS" if mechanism_pass else "FAIL",
                   "route_a_utilization": utilization_status,
                   "correctness": correctness["status"]}
    dump_outputs(root, summary, verdict_obj)
    return {"summary": summary, "verdict": verdict_obj}


def summarize_case(parsed: dict[str, Any]) -> dict[str, Any]:
    credit = parsed["credit"]
    return {
        "case": parsed["case"], "errors": parsed["errors"],
        "action_count": len(parsed["actions"]["actions"]),
        "physical_relief_action_count": len(parsed["actions"]["physical_relief"]),
        "physical_relief_bytes": sum(item.get("physical_relief_bytes_int", 0) for item in parsed["actions"]["physical_relief"]),
        "physical_identity": parsed["actions"]["unique_physical_identity"],
        "confirmed_physical_credit_bytes": credit["confirmed_physical_credit_bytes"],
        "credit_earned_bytes": credit["earned_bytes"], "credit_spent_bytes": credit["spent_bytes"],
        "moe_grant_bytes": credit["grant_bytes"], "moe_old_budget_bytes": max_or_none([item.get("reallocation_moe_old_budget_bytes", 0) or 0 for item in parsed["observations"]["ledger"]]),
        "moe_new_budget_bytes": max_or_none([item.get("reallocation_moe_new_budget_bytes", 0) or 0 for item in parsed["observations"]["ledger"]]),
        "moe_utilization": parsed["moe_utilization"], "restore": parsed["restore"],
        "request_count": len(parsed["requests"]["requests"]),
    }


def dump_outputs(root: pathlib.Path, summary: dict[str, Any], verdict: dict[str, Any]) -> None:
    (root / "parsed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "verdict.json").write_text(json.dumps(verdict, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir")
    args = parser.parse_args()
    try:
        result = parse_artifact(pathlib.Path(args.artifact_dir))
    except InvalidArtifact as exc:
        root = pathlib.Path(args.artifact_dir)
        payload = {"verdict": "INVALID", "errors": [str(exc)]}
        try:
            dump_outputs(root, {"protocol": PROTOCOL, "errors": [str(exc)]}, payload)
        except OSError:
            pass
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result["verdict"], sort_keys=True))
    return 0 if result["verdict"].get("verdict") == "GLOBAL_FAST_A1_READY_FOR_INTEGRATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
