#!/usr/bin/env python3
"""Fail-closed verdict authority for the Stage 3C unified multi-slot smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "kv_governor_stage3c_1c_2b_1r"
PROTOCOL_VERSION = 6
MARKER = "kv_pressure_unified_action"
RESUME_MARKER = "kv_resume_order_event"
CAPABILITY_MARKER = "KV_GOVERNOR_CAPABILITY"
CASES = (
    "OFF", "GOVERNOR_ON", "INVALID_UNIFIED", "CONFLICT_UNIFIED_LEGACY",
    "CONFLICT_UNIFIED_DRY_RUN", "CONFLICT_UNIFIED_BOUNDED",
)
UINT = re.compile(r"[0-9]+$")
REQUIRED = {"state", "source", "stale", "decision_id", "episode", "target_bytes", "max_blocks", "observed_excess_bytes", "debt_before_bytes", "debt_after_bytes", "offload_armed_before", "offload_armed_after", "next_action_sample", "evaluate_attempted", "evaluate_outcome", "evaluate_reason", "release_attempted", "offload_attempted", "selected_seq_id", "selected_claimant_epoch", "transaction_id", "outcome", "reason", "blocks", "bytes", "relieved_bytes", "shortfall_bytes", "io_failure", "io_errno", "state_changed", "decision_reason", "sample_count", "idle", "claimants", "scores"}
RESUME_REQUIRED = {"phase", "decision_id", "seq_id", "claimant_epoch", "transaction_id", "action", "outcome", "reason", "graph_allowed"}
CAPABILITY_REQUIRED = {"n_slots", "n_seq_max", "n_stream", "kv_unified", "paged_metadata", "ingraph_gather", "release_supported", "offload_supported", "prefetch_supported", "backing_ready", "swap_explicit_only"}
GOVERNOR_ENV = {"LLAMA_KV_PRESSURE_UNIFIED_ACTION", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"}
REQUIRED_ENABLED_CAPABILITY = {"kv_unified", "paged_metadata", "ingraph_gather", "release_supported", "offload_supported", "prefetch_supported", "backing_ready", "swap_explicit_only"}
NEGATIVE_ENV = {
    "INVALID_UNIFIED": {"LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "invalid"},
    "CONFLICT_UNIFIED_LEGACY": {"LLAMA_KV_PAGED_RELEASE": "1"},
    "CONFLICT_UNIFIED_DRY_RUN": {"LLAMA_KV_PRESSURE_DRY_RUN": "1"},
    "CONFLICT_UNIFIED_BOUNDED": {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"},
}


class Error(Exception):
    pass


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: pathlib.Path) -> Any:
    if not path.is_file():
        raise Error(f"missing {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        raise Error(f"invalid JSON {path}: {exc}") from exc


def parse_fields(line: str, token: str, required: set[str]) -> dict[str, str]:
    words = line.split()
    if words.count(token) != 1:
        raise Error(f"duplicate or malformed {token} token")
    values: dict[str, str] = {}
    for word in words[words.index(token) + 1:]:
        if word.count("=") != 1:
            raise Error(f"malformed {token} field {word!r}")
        key, value = word.split("=", 1)
        if not key or not value or key in values:
            raise Error(f"duplicate or malformed {token} key {key!r}")
        values[key] = value
    if set(values) != required:
        raise Error(f"{token} schema mismatch missing={sorted(required-set(values))} extra={sorted(set(values)-required)}")
    return values


def parse_marker(line: str) -> dict[str, str]:
    values = parse_fields(line, MARKER, REQUIRED)
    symbolic = {"state", "source", "evaluate_outcome", "evaluate_reason", "outcome", "reason", "decision_reason", "claimants", "scores"}
    for key, value in values.items():
        if key in symbolic:
            continue
        if key == "selected_seq_id":
            if not re.fullmatch(r"-?[0-9]+", value):
                raise Error("selected_seq_id is not integer")
        elif not UINT.fullmatch(value):
            raise Error(f"{key} must be unsigned integer")
    if values["state"] not in {"PRESSURE", "CRITICAL", "NORMAL", "RECOVERY"}:
        raise Error("bad pressure state")
    for key in ("stale", "offload_armed_before", "offload_armed_after", "evaluate_attempted", "release_attempted", "offload_attempted", "io_failure", "state_changed", "idle"):
        if values[key] not in {"0", "1"}:
            raise Error(f"{key} must be boolean")
    return values


def parse_resume(line: str) -> dict[str, str]:
    values = parse_fields(line, RESUME_MARKER, RESUME_REQUIRED)
    for key in ("decision_id", "seq_id", "claimant_epoch", "transaction_id"):
        if not UINT.fullmatch(values[key]):
            raise Error(f"resume {key} must be unsigned integer")
    if values["phase"] not in {"prefetch", "graph_gate"} or values["action"] != "prefetch" or values["graph_allowed"] not in {"0", "1"}:
        raise Error("invalid resume event")
    return values


def parse_capability(line: str) -> dict[str, str]:
    values = parse_fields(line, CAPABILITY_MARKER, CAPABILITY_REQUIRED)
    for key in ("n_slots", "n_seq_max", "n_stream"):
        if not UINT.fullmatch(values[key]):
            raise Error(f"capability {key} must be unsigned integer")
    for key in CAPABILITY_REQUIRED - {"n_slots", "n_seq_max", "n_stream"}:
        if values[key] not in {"0", "1"}:
            raise Error(f"capability {key} must be boolean")
    return values


def token_lines(data: bytes, token: str, parser: Callable[[str], dict[str, str]], origin: str, errors: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    offset = 0
    for number, raw_line in enumerate(data.splitlines(keepends=True), 1):
        line = raw_line.decode(errors="replace")
        if token in line:
            try:
                record = parser(line)
                record["_offset"] = str(offset)
                record["_end"] = str(offset + len(raw_line))
                result.append(record)
            except Error as exc:
                errors.append(f"{origin}:{number}: {exc}")
        offset += len(raw_line)
    return result


def expected_labels(parallel: int) -> set[str]:
    labels = {f"seed_s{slot}" for slot in range(parallel - 1)}
    labels |= {f"active_s{parallel - 1}", "reaccess_a"}
    if parallel == 3:
        labels |= {"reaccess_b", "reuse_a"}
    return labels


def expected_slot(label: str, parallel: int) -> int | None:
    if label.startswith("seed_s") or label.startswith("active_s"):
        match = re.fullmatch(r"(?:seed|active)_s([0-9]+)", label)
        return int(match.group(1)) if match else None
    if label in {"reaccess_a", "reuse_a"}:
        return 0
    if label == "reaccess_b":
        return 1
    return None


def rows(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Error(f"missing {path}")
    result = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("not object")
            result.append(item)
        except Exception as exc:
            raise Error(f"malformed request record {path}:{number}: {exc}") from exc
    return result


def keyed(items: list[dict[str, Any]], name: str, parallel: int, errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        label = item.get("label")
        if not isinstance(label, str) or not label or label in result:
            errors.append(f"{name}: missing or duplicate HTTP label")
            continue
        result[label] = item
    if set(result) != expected_labels(parallel):
        errors.append(f"{name}: incomplete or unexpected parallel={parallel} request layout")
    for label, item in result.items():
        slot = expected_slot(label, parallel)
        request = item.get("request", {})
        if slot is None or not isinstance(request, dict) or request.get("id_slot") != slot:
            errors.append(f"{name}: {label} does not bind its recorded slot")
        if label == f"active_s{parallel - 1}":
            if (request.get("n_predict") != -1 or request.get("stream") is not True or
                    request.get("ignore_eos") is not True):
                errors.append(f"{name}: active request is not an explicit endless stream")
            if (item.get("http_status") != 200 or item.get("cancelled_by_runner") is not True or
                    item.get("response", {}).get("completion") != "runner_cancelled" or
                    not isinstance(item.get("stop_requested_monotonic_ns"), int)):
                errors.append(f"{name}: active request was not explicitly cancelled by the runner")
    return result


def validate_layout_ready(
        result: dict[str, Any],
        request_rows: dict[str, dict[str, Any]],
        parallel: int,
        raw_size: int,
        markers: list[dict[str, str]],
        enabled: bool,
        errors: list[str],
        name: str) -> int | None:
    layout = result.get("layout_ready")
    required = {
        "stderr_end", "evidence_marker_end", "decision_id", "source",
        "observed_monotonic_ns", "slots", "claimants",
    }
    if not isinstance(layout, dict) or set(layout) != required:
        errors.append(f"{name}: missing or malformed layout_ready boundary")
        return None
    stderr_end = layout.get("stderr_end")
    marker_end = layout.get("evidence_marker_end")
    decision_id = layout.get("decision_id")
    observed_ns = layout.get("observed_monotonic_ns")
    if (not isinstance(stderr_end, int) or isinstance(stderr_end, bool) or
            not isinstance(marker_end, int) or isinstance(marker_end, bool) or
            not isinstance(decision_id, int) or isinstance(decision_id, bool) or
            stderr_end < 0 or marker_end < 0 or stderr_end > raw_size or marker_end > raw_size):
        errors.append(f"{name}: layout_ready stderr boundary is invalid")
        return None
    if not isinstance(observed_ns, int) or isinstance(observed_ns, bool) or observed_ns <= 0:
        errors.append(f"{name}: layout_ready timestamp is invalid")
        return None
    active_slot = parallel - 1
    expected_slots = [
        {"id": slot, "is_processing": slot == active_slot}
        for slot in range(parallel)
    ]
    if layout.get("slots") != expected_slots:
        errors.append(f"{name}: layout_ready does not preserve exact idle/active slot roles")

    claimants = layout.get("claimants")
    claimant_map: dict[int, dict[str, Any]] = {}
    claimant_fields = {
        "seq_id", "epoch", "active", "exhausted", "valid", "target_blocks",
        "eligible_resident_blocks", "swapped_blocks", "shared_blocks", "blocked_blocks",
    }
    if not isinstance(claimants, list) or len(claimants) != parallel:
        errors.append(f"{name}: layout_ready lacks exact runtime claimant capacity")
    else:
        for claimant in claimants:
            if not isinstance(claimant, dict) or set(claimant) != claimant_fields:
                errors.append(f"{name}: malformed runtime claimant capacity")
                continue
            seq_id = claimant.get("seq_id")
            counts = [claimant.get(key) for key in (
                "target_blocks", "eligible_resident_blocks", "swapped_blocks",
                "shared_blocks", "blocked_blocks")]
            if (not isinstance(seq_id, int) or isinstance(seq_id, bool) or seq_id in claimant_map or
                    not isinstance(claimant.get("epoch"), int) or claimant["epoch"] < 1 or
                    not isinstance(claimant.get("active"), bool) or
                    not isinstance(claimant.get("exhausted"), bool) or
                    not isinstance(claimant.get("valid"), bool) or
                    any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts)):
                errors.append(f"{name}: invalid runtime claimant capacity")
                continue
            if counts[0] != sum(counts[1:]):
                errors.append(f"{name}: runtime claimant capacity accounting mismatch")
            claimant_map[seq_id] = claimant
        if set(claimant_map) != set(range(parallel)):
            errors.append(f"{name}: runtime claimant capacity does not cover exact slots")
        for slot in range(parallel):
            claimant = claimant_map.get(slot)
            if not claimant:
                continue
            if claimant["active"] != (slot == active_slot):
                errors.append(f"{name}: runtime claimant active role mismatch for slot {slot}")
            if slot < active_slot and (claimant["exhausted"] or not claimant["valid"] or
                    claimant["eligible_resident_blocks"] < 2):
                errors.append(f"{name}: idle claimant {slot} lacks two eligible resident blocks")

    active = request_rows.get(f"active_s{active_slot}")
    started = active.get("started_monotonic_ns") if isinstance(active, dict) else None
    stop_requested = active.get("stop_requested_monotonic_ns") if isinstance(active, dict) else None
    finished = active.get("finished_monotonic_ns") if isinstance(active, dict) else None
    if (not isinstance(started, int) or isinstance(started, bool) or
            not isinstance(stop_requested, int) or isinstance(stop_requested, bool) or
            not isinstance(finished, int) or isinstance(finished, bool) or
            not started <= observed_ns < stop_requested <= finished):
        errors.append(f"{name}: active request lifecycle does not contain layout_ready")

    if enabled:
        if layout.get("source") != "governor_pre_action" or decision_id < 1 or marker_end <= stderr_end:
            errors.append(f"{name}: layout_ready lacks a pre-action Governor marker")
        evidence = next((marker for marker in markers
                         if int(marker["_offset"]) == stderr_end and int(marker["_end"]) == marker_end and
                         int(marker["decision_id"]) == decision_id), None)
        if evidence is None:
            errors.append(f"{name}: layout_ready marker evidence is missing")
        else:
            observed = validate_runtime_claimants(evidence, parallel, errors)
            if observed is not None and claimant_map and any(observed.get(slot) != claimant_map.get(slot) for slot in range(parallel)):
                errors.append(f"{name}: layout_ready claimant state differs from its pre-action marker")
    elif layout.get("source") != "slots" or decision_id != 0 or marker_end != 0:
        errors.append(f"{name}: OFF layout_ready is not a live /slots snapshot")
    return stderr_end


def validate_active_stop(
        result: dict[str, Any],
        request_rows: dict[str, dict[str, Any]],
        parallel: int,
        errors: list[str],
        name: str) -> None:
    active_label = f"active_s{parallel - 1}"
    active = request_rows.get(active_label)
    stop = result.get("active_stop")
    if not isinstance(active, dict) or not isinstance(stop, dict):
        errors.append(f"{name}: missing active cancellation evidence")
        return
    requested = stop.get("requested_monotonic_ns")
    finished = stop.get("finished_monotonic_ns")
    stderr_start = stop.get("stderr_start")
    if (stop.get("reason") != "gate_complete" or stop.get("thread_joined") is not True or
            stop.get("http_status") != 200 or stop.get("completion") != "runner_cancelled"):
        errors.append(f"{name}: active producer was not explicitly cancelled after gate completion")
    if (not isinstance(requested, int) or isinstance(requested, bool) or
            not isinstance(finished, int) or isinstance(finished, bool) or
            not isinstance(stderr_start, int) or isinstance(stderr_start, bool) or
            requested <= 0 or finished < requested or stderr_start < 0):
        errors.append(f"{name}: malformed active cancellation boundary")
        return
    if (active.get("stop_requested_monotonic_ns") != requested or
            active.get("finished_monotonic_ns") != finished or
            active.get("stop_reason") != "gate_complete"):
        errors.append(f"{name}: active request record differs from cancellation record")
    for label, row in request_rows.items():
        if label == active_label:
            continue
        row_finished = row.get("finished_monotonic_ns")
        if not isinstance(row_finished, int) or isinstance(row_finished, bool) or row_finished > requested:
            errors.append(f"{name}: active producer ended before {label} completed")
    scopes = result.get("resume_scopes")
    if isinstance(scopes, dict):
        for scope_name, scope in scopes.items():
            if not isinstance(scope, dict) or not isinstance(scope.get("end"), int) or scope["end"] > stderr_start:
                errors.append(f"{name}: active producer ended before resume scope {scope_name} completed")


def normalize_argv(argv: Any) -> list[str] | None:
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return None
    result = list(argv)
    try:
        result[result.index("--port") + 1] = "<port>"
    except (ValueError, IndexError):
        return None
    return result


def capability_from_log(data: bytes, parallel: int, errors: list[str], missing: list[str], name: str) -> dict[str, str] | None:
    records = token_lines(data, CAPABILITY_MARKER, parse_capability, name, errors)
    if not records:
        missing.append(f"{name}: missing production capability record")
        return None
    if len(records) != 1:
        errors.append(f"{name}: duplicate production capability record")
        return None
    capability = records[0]
    expected_numbers = {"n_slots": parallel, "n_seq_max": parallel, "n_stream": 1}
    for key, expected in expected_numbers.items():
        if int(capability[key]) != expected:
            errors.append(f"{name}: capability {key}={capability[key]}, expected {expected}")
    for key in REQUIRED_ENABLED_CAPABILITY:
        if capability[key] != "1":
            missing.append(f"{name}: production capability {key}=0")
    return capability


def validate_backing(case: pathlib.Path, env: dict[str, Any], errors: list[str]) -> None:
    backing = read_json(case / "backing.json")
    expected_path = (case / "backing").resolve()
    if (env.get("LLAMA_KV_SWAP_DIR") != "backing" or
            env.get("LLAMA_KV_PAGED_SWAP") != "1" or
            env.get("LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY") != "1" or
            env.get("LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE") != "1"):
        errors.append(f"{case}: missing exact explicit-only paged swap/claimant-trace environment")
    if backing.get("environment_value") != "backing" or backing.get("path") != str(expected_path):
        errors.append(f"{case}: backing record does not identify its own artifact directory")
    if backing.get("created") is not True or backing.get("cleanup_attempted") is not True:
        errors.append(f"{case}: backing lifecycle was not recorded")
    if backing.get("exists_after_cleanup") is not False or backing.get("cleanup_error") is not None:
        errors.append(f"{case}: backing cleanup did not complete")
    if (case / "backing").exists():
        errors.append(f"{case}: backing directory remains after cleanup")


def validate_process(case: pathlib.Path, errors: list[str]) -> None:
    process = read_json(case / "process.json")
    for key in ("term_timed_out", "kill_timed_out", "residual_process"):
        if not isinstance(process.get(key), bool):
            errors.append(f"{case}: missing bounded cleanup outcome {key}")
    if process.get("kill_timed_out"):
        errors.append(f"{case}: server process exceeded SIGKILL timeout")
    if process.get("residual_process"):
        errors.append(f"{case}: server process remained after cleanup")


def validate_action_startup(data: bytes, env: dict[str, Any], enabled: bool, errors: list[str], missing: list[str], name: str) -> None:
    text = data.decode(errors="replace")
    action = re.findall(r"KV pressure unified action enabled: target_bytes=([0-9]+) max_blocks=([0-9]+)", text)
    if not enabled:
        if action:
            errors.append(f"{name}: OFF baseline reports unified action enabled")
        return
    if not action:
        missing.append(f"{name}: missing unified action enabled startup evidence")
    elif len(action) != 1:
        errors.append(f"{name}: duplicate unified action enabled startup evidence")
    elif action[0] != (str(env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES")), str(env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"))):
        errors.append(f"{name}: unified action startup values differ from environment")


def validate_runtime_claimants(
        marker: dict[str, str],
        parallel: int,
        errors: list[str]) -> dict[int, dict[str, Any]] | None:
    encoded = marker["claimants"]
    if encoded == "none":
        errors.append(f"parallel={parallel}: marker lacks runtime claimant observations")
        return None
    result: dict[int, dict[str, Any]] = {}
    for part in encoded.split(";"):
        fields = part.split(":")
        if len(fields) != 10 or any(not UINT.fullmatch(field) for field in fields):
            errors.append("malformed runtime claimant observation")
            continue
        seq_id, epoch, active, exhausted, valid, target, eligible, swapped, shared, blocked = map(int, fields)
        if seq_id in result:
            errors.append("duplicate runtime claimant observation")
        if epoch < 1 or active not in (0, 1) or exhausted not in (0, 1) or valid not in (0, 1):
            errors.append("invalid runtime claimant state")
        if target != eligible + swapped + shared + blocked:
            errors.append("runtime claimant block accounting mismatch")
        result[seq_id] = {
            "seq_id": seq_id,
            "epoch": epoch,
            "active": bool(active),
            "exhausted": bool(exhausted),
            "valid": bool(valid),
            "target_blocks": target,
            "eligible_resident_blocks": eligible,
            "swapped_blocks": swapped,
            "shared_blocks": shared,
            "blocked_blocks": blocked,
        }
    if set(result) != set(range(parallel)):
        errors.append(f"runtime claimants do not cover exact parallel={parallel} slots")
    return result


def validate_scores(marker: dict[str, str], parallel: int, errors: list[str]) -> dict[int, tuple[bool, str]] | None:
    scores = marker["scores"]
    if scores == "none":
        return None
    result: dict[int, tuple[bool, str]] = {}
    for part in scores.split(";"):
        fields = part.split(":")
        if len(fields) != 10 or not re.fullmatch(r"-?[0-9]+", fields[0]) or fields[1] not in {"0", "1"}:
            errors.append("malformed claimant score")
            continue
        seq = int(fields[0])
        if seq in result:
            errors.append("duplicate claimant score")
        result[seq] = (fields[1] == "1", fields[2])
    if set(result) != set(range(parallel)):
        errors.append(f"claimant scores do not cover exact parallel={parallel} slots")
    return result


def validate_action_transaction(
        marker: dict[str, str],
        parallel: int,
        index: int,
        errors: list[str]) -> None:
    if marker["release_attempted"] != "1" and marker["offload_attempted"] != "1":
        return
    transaction = int(marker["transaction_id"])
    blocks = int(marker["blocks"])
    byte_count = int(marker["bytes"])
    relief = int(marker["relieved_bytes"])
    state_changed = marker["state_changed"] == "1"
    partial = marker["outcome"] == "partial_failure"
    no_candidate_noop = marker["outcome"] == "no_op" and marker["reason"] == "no_candidate"

    if no_candidate_noop:
        if transaction != 0 or state_changed or blocks != 0 or byte_count != 0 or relief != 0:
            errors.append(
                f"parallel={parallel} marker[{index}]: no_op/no_candidate has transaction or completed work")
        return
    if partial and (transaction == 0 or not state_changed):
        errors.append(
            f"parallel={parallel} marker[{index}]: partial action lacks state-changing transaction")
        return
    if state_changed and transaction == 0:
        errors.append(f"parallel={parallel} marker[{index}]: state-changing action lacks transaction")
    elif not state_changed and transaction != 0:
        errors.append(f"parallel={parallel} marker[{index}]: transaction exists without state change")


def validate_markers(
        markers: list[dict[str, str]],
        parallel: int,
        capability: dict[str, str] | None,
        errors: list[str],
        missing: list[str]) -> tuple[list[dict[str, str]], set[str]]:
    if not markers:
        missing.append(f"parallel={parallel}: no real production governor markers")
        return [], set()
    last_sample = last_decision = -1
    release_at: list[int] = []
    changed_offloads: list[dict[str, str]] = []
    for index, marker in enumerate(markers):
        sample, decision = int(marker["sample_count"]), int(marker["decision_id"])
        if sample <= last_sample or decision <= last_decision:
            errors.append(f"parallel={parallel} marker[{index}]: duplicate/out-of-order sample or decision")
        last_sample, last_decision = sample, decision
        if marker["state"] not in {"PRESSURE", "CRITICAL"} or marker["stale"] != "0" or marker["evaluate_attempted"] != "1":
            errors.append(f"parallel={parallel} marker[{index}]: invalid pressure/EVALUATE")
        if int(marker["release_attempted"]) + int(marker["offload_attempted"]) > 1:
            errors.append(f"parallel={parallel} marker[{index}]: multiple actions in one decision")
        before, after, relief = int(marker["debt_before_bytes"]), int(marker["debt_after_bytes"]), int(marker["relieved_bytes"])
        action = marker["release_attempted"] == "1" or marker["offload_attempted"] == "1"
        if action and (relief > before or after != before - relief):
            errors.append(f"parallel={parallel} marker[{index}]: debt is not reduced solely by relieved_bytes")
        if action and relief == 0 and after != before:
            errors.append(f"parallel={parallel} marker[{index}]: zero-relief action changed debt")
        validate_action_transaction(marker, parallel, index, errors)
        marker["_runtime_claimants"] = validate_runtime_claimants(marker, parallel, errors)
        marker["_scores"] = validate_scores(marker, parallel, errors)
        if capability is not None:
            if marker["release_attempted"] == "1" and capability["release_supported"] != "1":
                errors.append(f"parallel={parallel} marker[{index}]: runtime RELEASE contradicts release_supported=0")
            if marker["offload_attempted"] == "1" and capability["offload_supported"] != "1":
                errors.append(f"parallel={parallel} marker[{index}]: runtime OFFLOAD contradicts offload_supported=0")
            if marker["decision_reason"] == "offload_unsupported":
                if capability["offload_supported"] == "1":
                    errors.append(f"parallel={parallel} marker[{index}]: offload_unsupported contradicts production capability")
                else:
                    missing.append(f"parallel={parallel}: runtime OFFLOAD unavailable")
        if marker["decision_reason"] == "release_unsupported":
            if capability is not None and capability["release_supported"] == "1":
                errors.append(f"parallel={parallel} marker[{index}]: release_unsupported contradicts production capability")
            else:
                missing.append(f"parallel={parallel}: UNIFIED_ACTION_UNSUPPORTED release_unsupported")
        if marker["release_attempted"] == "1":
            release_at.append(index)
            if marker["offload_armed_before"] != "0":
                errors.append(f"parallel={parallel} marker[{index}]: RELEASE after OFFLOAD arm")
        if marker["offload_attempted"] == "1":
            seq, epoch = int(marker["selected_seq_id"]), int(marker["selected_claimant_epoch"])
            if marker["offload_armed_before"] != "1" or seq < 0 or epoch < 1:
                errors.append(f"parallel={parallel} marker[{index}]: invalid OFFLOAD claimant")
            if marker["state_changed"] == "1":
                if int(marker["blocks"]) < 2 or relief == 0:
                    errors.append(f"parallel={parallel} marker[{index}]: state-changing OFFLOAD is not multi-block/relieving")
                changed_offloads.append(marker)
    if not release_at:
        missing.append(f"parallel={parallel}: missing RELEASE-first decision")
    if not changed_offloads:
        missing.append(f"parallel={parallel}: missing state-changing multi-block OFFLOAD")
    elif release_at and min(markers.index(marker) for marker in changed_offloads) <= release_at[0]:
        errors.append(f"parallel={parallel}: OFFLOAD preceded RELEASE")
    return changed_offloads, set()


def find_after(markers: list[dict[str, str]], offset: int, predicate: Callable[[dict[str, str]], bool]) -> dict[str, str] | None:
    return next((marker for marker in markers if int(marker["_offset"]) >= offset and predicate(marker)), None)


def require_score(
        marker: dict[str, str],
        slot: int,
        *,
        eligible: bool | None = None,
        exclusion: str | None = None,
        phase: str,
        errors: list[str]) -> None:
    scores = marker.get("_scores")
    if not isinstance(scores, dict) or slot not in scores:
        errors.append(f"{phase}: missing claimant score for slot {slot}")
        return
    actual_eligible, actual_exclusion = scores[slot]
    if eligible is not None and actual_eligible != eligible:
        errors.append(f"{phase}: slot {slot} eligibility does not match workload role")
    if exclusion is not None and actual_exclusion != exclusion:
        errors.append(f"{phase}: slot {slot} exclusion={actual_exclusion}, expected {exclusion}")


def validate_workload(
        markers: list[dict[str, str]],
        parallel: int,
        layout_start: int,
        reaccess_start: int,
        errors: list[str],
        missing: list[str]) -> dict[str, dict[str, str]]:
    arm = find_after(
        markers,
        0,
        lambda marker: marker["release_attempted"] == "1" and marker["reason"] == "no_candidate" and marker["offload_armed_after"] == "1")
    if arm is None:
        missing.append(f"parallel={parallel}: missing full RELEASE tour no_candidate arm")
        return {}
    offload_start = max(layout_start, int(arm["_offset"]))
    first_offload = find_after(
        markers,
        offload_start,
        lambda marker: marker["offload_attempted"] == "1")
    no_idle = find_after(
        markers,
        offload_start,
        lambda marker: marker["decision_reason"] == "claimant_no_candidate")
    if no_idle is not None and (first_offload is None or int(no_idle["_offset"]) < int(first_offload["_offset"])):
        errors.append(f"parallel={parallel}: arm was followed by no idle claimant before OFFLOAD A")
    if first_offload is None:
        missing.append(f"parallel={parallel}: missing state-changing OFFLOAD A after arm")
        return {}
    if first_offload["selected_seq_id"] != "0":
        errors.append(f"parallel={parallel}: wrong claimant OFFLOAD occurred before claimant A")
        return {}
    if first_offload["outcome"] == "no_op" and first_offload["reason"] == "no_candidate":
        errors.append(f"parallel={parallel}: claimant A exhausted before its required multi-block OFFLOAD")
        return {}
    if first_offload["state_changed"] != "1":
        errors.append(f"parallel={parallel}: first claimant A OFFLOAD did not change state")
        return {}
    if int(first_offload["blocks"]) < 2:
        errors.append(f"parallel={parallel}: first claimant A OFFLOAD completed fewer than 2 blocks")
        return {}
    first_a = first_offload
    if int(first_a["_offset"]) >= reaccess_start:
        errors.append(f"parallel={parallel}: A was reaccessed before its required OFFLOAD")
    require_score(first_a, 0, eligible=True, phase=f"parallel={parallel} OFFLOAD A", errors=errors)
    if parallel == 2:
        require_score(first_a, 1, eligible=False, exclusion="active", phase="parallel=2 OFFLOAD A", errors=errors)
        return {"a_initial": first_a}

    require_score(first_a, 1, eligible=True, phase="parallel=3 OFFLOAD A", errors=errors)
    require_score(first_a, 2, eligible=False, exclusion="active", phase="parallel=3 OFFLOAD A", errors=errors)
    exhausted_a = find_after(
        markers,
        int(first_a["_offset"]),
        lambda marker: marker["offload_attempted"] == "1" and marker["selected_seq_id"] == "0" and marker["outcome"] == "no_op" and marker["reason"] == "no_candidate")
    if exhausted_a is None:
        missing.append("parallel=3: missing claimant A no_candidate exhaustion")
        return {"a_initial": first_a}
    offload_b = find_after(
        markers,
        int(exhausted_a["_offset"]),
        lambda marker: marker["offload_attempted"] == "1" and marker["selected_seq_id"] == "1" and marker["state_changed"] == "1")
    if offload_b is None:
        missing.append("parallel=3: exhausted claimant A did not advance to B")
        return {"a_initial": first_a}
    if int(offload_b["_offset"]) >= reaccess_start:
        errors.append("parallel=3: A was reaccessed before B advanced after exhaustion")
    require_score(offload_b, 0, eligible=False, exclusion="exhausted", phase="parallel=3 OFFLOAD B", errors=errors)
    require_score(offload_b, 1, eligible=True, phase="parallel=3 OFFLOAD B", errors=errors)
    require_score(offload_b, 2, eligible=False, exclusion="active", phase="parallel=3 OFFLOAD B", errors=errors)
    initial_epoch = int(first_a["selected_claimant_epoch"])
    reused_a = find_after(
        markers,
        reaccess_start,
        lambda marker: marker["offload_attempted"] == "1" and marker["selected_seq_id"] == "0" and marker["state_changed"] == "1" and int(marker["selected_claimant_epoch"]) > initial_epoch)
    if reused_a is None:
        missing.append(f"parallel=3: missing reused claimant A epoch after {initial_epoch}")
        return {"a_initial": first_a, "b": offload_b}
    require_score(reused_a, 0, eligible=True, phase="parallel=3 reused OFFLOAD A", errors=errors)
    require_score(reused_a, 2, eligible=False, exclusion="active", phase="parallel=3 reused OFFLOAD A", errors=errors)
    return {"a_initial": first_a, "b": offload_b, "a_reused": reused_a}


def validate_resume(
        case: pathlib.Path,
        offloads: dict[str, dict[str, str]],
        capability: dict[str, str] | None,
        errors: list[str],
        missing: list[str],
        parallel: int) -> None:
    result, raw = read_json(case / "result.json"), (case / "server.stderr").read_bytes()
    scopes = result.get("resume_scopes")
    expected = {"a_initial"} if parallel == 2 else {"a_initial", "b", "a_reused"}
    if not isinstance(scopes, dict) or set(scopes) != expected:
        missing.append(f"parallel={parallel}: missing exact scoped PREFETCH evidence")
        return
    if capability is not None and capability["prefetch_supported"] != "1":
        missing.append(f"parallel={parallel}: production prefetch capability unavailable")
        return
    for scope_name in expected:
        scope = scopes.get(scope_name)
        offload = offloads.get(scope_name)
        if offload is None:
            missing.append(f"parallel={parallel}: missing OFFLOAD for resume scope {scope_name}")
            continue
        if not isinstance(scope, dict):
            missing.append(f"parallel={parallel}: malformed resume scope {scope_name}")
            continue
        start, end = scope.get("start"), scope.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start < int(offload["_offset"]) or end < start or end > len(raw):
            errors.append(f"parallel={parallel}: invalid ordering/bounds for resume scope {scope_name}")
            continue
        events = token_lines(raw[start:end], RESUME_MARKER, parse_resume, str(case / "server.stderr"), errors)
        key = (offload["selected_seq_id"], offload["selected_claimant_epoch"])
        found = False
        for index, event in enumerate(events):
            if event["phase"] != "prefetch" or (event["seq_id"], event["claimant_epoch"]) != key:
                continue
            if event["outcome"] != "completed" or event["graph_allowed"] != "1":
                errors.append(f"parallel={parallel}: PREFETCH {key} in {scope_name} did not complete before graph")
                continue
            if any(later["phase"] == "graph_gate" and all(later[field] == event[field] for field in RESUME_REQUIRED - {"phase"}) for later in events[index + 1:]):
                found = True
        if not found:
            missing.append(f"parallel={parallel}: OFFLOAD claimant {key} lacks ordered PREFETCH→graph gate in {scope_name}")


def validate_unit(root: pathlib.Path, parallel: int, errors: list[str], missing: list[str]) -> None:
    unit = root / f"parallel_{parallel}"
    if not unit.is_dir():
        errors.append(f"missing parallel={parallel} unit")
        return
    cases = {name: unit / name for name in CASES}
    for name, case in cases.items():
        if not case.is_dir():
            errors.append(f"parallel={parallel}: missing {name} case")
    if any(not case.is_dir() for case in cases.values()):
        return

    all_envs: dict[str, dict[str, Any]] = {}
    for name, case in cases.items():
        env = read_json(case / "environment.json")
        if not isinstance(env, dict):
            errors.append(f"parallel={parallel}: malformed {name} environment record")
            continue
        all_envs[name] = env
        validate_backing(case, env, errors)
        validate_process(case, errors)
    for name, expected in NEGATIVE_ENV.items():
        env = all_envs.get(name)
        if not isinstance(env, dict):
            continue
        for key, value in expected.items():
            if env.get(key) != value:
                errors.append(f"parallel={parallel}: {name} lacks its required invalid/conflicting environment")
        for other_name, other_expected in NEGATIVE_ENV.items():
            if other_name == name:
                continue
            for key, value in other_expected.items():
                if env.get(key) == value:
                    errors.append(f"parallel={parallel}: {name} contains another negative-case trigger")
        result = read_json(cases[name] / "result.json")
        if not result.get("rejected_before_request_loop") or result.get("health_reached") or result.get("exit_code", 0) == 0 or result.get("request_loop_started"):
            errors.append(f"parallel={parallel}: {name} did not reject before listener/request loop")

    executions = {name: read_json(cases[name] / "execution.json") for name in CASES}
    envs = {name: all_envs.get(name) for name in ("OFF", "GOVERNOR_ON")}
    off_argv, on_argv = normalize_argv(executions["OFF"].get("argv")), normalize_argv(executions["GOVERNOR_ON"].get("argv"))
    if off_argv is None or on_argv is None or off_argv != on_argv:
        errors.append(f"parallel={parallel}: OFF/GOVERNOR_ON argv differ")
    else:
        required_argv = {
            "--kv-unified", "--no-cache-idle-slots", "--cache-type-k", "f32",
            "--cache-type-v", "f32", "--parallel", str(parallel), "--timeout", "300",
        }
        if not required_argv.issubset(set(off_argv)):
            errors.append(f"parallel={parallel}: argv lacks exact unified/no-idle/F32 configuration")
    for name, execution in executions.items():
        if execution.get("cwd") != str(cases[name].resolve()):
            errors.append(f"parallel={parallel}: {name} did not run in its artifact directory")
        if execution.get("environment") != all_envs.get(name):
            errors.append(f"parallel={parallel}: {name} execution environment differs from its recorded environment")
    if not isinstance(envs["OFF"], dict) or not isinstance(envs["GOVERNOR_ON"], dict):
        errors.append(f"parallel={parallel}: malformed paired environment record")
        return
    changed = {key for key in set(envs["OFF"]) | set(envs["GOVERNOR_ON"]) if envs["OFF"].get(key) != envs["GOVERNOR_ON"].get(key)}
    if changed != GOVERNOR_ENV:
        errors.append(f"parallel={parallel}: OFF/GOVERNOR_ON environment differs outside unified action keys")

    off_result, on_result = read_json(cases["OFF"] / "result.json"), read_json(cases["GOVERNOR_ON"] / "result.json")
    for name, result in (("OFF", off_result), ("GOVERNOR_ON", on_result)):
        if result.get("status") != "complete" or result.get("request_loop_started") is not True:
            errors.append(f"parallel={parallel}: {name} did not complete its request loop")
    off_rows = keyed(rows(cases["OFF"] / "requests.jsonl"), f"parallel={parallel} OFF", parallel, errors)
    on_rows = keyed(rows(cases["GOVERNOR_ON"] / "requests.jsonl"), f"parallel={parallel} GOVERNOR_ON", parallel, errors)
    active_label = f"active_s{parallel - 1}"
    for label in set(off_rows) | set(on_rows):
        if label not in off_rows or label not in on_rows:
            continue
        if off_rows[label].get("request") != on_rows[label].get("request"):
            errors.append(f"parallel={parallel}: {label} request differs between OFF and GOVERNOR_ON")
            continue
        if off_rows[label].get("http_status") != 200 or on_rows[label].get("http_status") != 200:
            errors.append(f"parallel={parallel}: {label} did not receive HTTP 200 in both cases")
        if label != active_label and off_rows[label].get("response_sha256") != on_rows[label].get("response_sha256"):
            errors.append(f"parallel={parallel}: {label} violates paired HTTP identity/correctness")

    off_raw = (cases["OFF"] / "server.stderr").read_bytes()
    on_raw = (cases["GOVERNOR_ON"] / "server.stderr").read_bytes()
    off_markers = token_lines(
        off_raw, MARKER, parse_marker, str(cases["OFF"] / "server.stderr"), errors)
    markers = token_lines(
        on_raw, MARKER, parse_marker, str(cases["GOVERNOR_ON"] / "server.stderr"), errors)
    off_layout = validate_layout_ready(
        off_result, off_rows, parallel, len(off_raw), off_markers, False,
        errors, f"parallel={parallel} OFF")
    on_layout = validate_layout_ready(
        on_result, on_rows, parallel, len(on_raw), markers, True,
        errors, f"parallel={parallel} GOVERNOR_ON")
    validate_active_stop(off_result, off_rows, parallel, errors, f"parallel={parallel} OFF")
    validate_active_stop(on_result, on_rows, parallel, errors, f"parallel={parallel} GOVERNOR_ON")
    off_capability = capability_from_log(off_raw, parallel, errors, missing, f"parallel={parallel} OFF")
    on_capability = capability_from_log(on_raw, parallel, errors, missing, f"parallel={parallel} GOVERNOR_ON")
    if off_capability is not None and on_capability is not None:
        if {key: value for key, value in off_capability.items() if not key.startswith("_")} != {key: value for key, value in on_capability.items() if not key.startswith("_")}:
            errors.append(f"parallel={parallel}: OFF/GOVERNOR_ON production capability differs")
    if off_markers:
        errors.append(f"parallel={parallel}: OFF baseline contains Governor marker")
    validate_action_startup(off_raw, envs["OFF"], False, errors, missing, f"parallel={parallel} OFF")
    validate_action_startup(on_raw, envs["GOVERNOR_ON"], True, errors, missing, f"parallel={parallel} GOVERNOR_ON")
    active_start = on_result.get("active_stderr_start")
    reaccess_start = on_result.get("reaccess_stderr_start")
    if not isinstance(active_start, int) or active_start < 0:
        missing.append(f"parallel={parallel}: missing active-workload boundary")
        return
    if on_layout is None:
        return
    if not isinstance(reaccess_start, int) or reaccess_start < 0:
        missing.append(f"parallel={parallel}: missing reaccess boundary")
        return
    if on_layout < active_start or reaccess_start < on_layout:
        errors.append(f"parallel={parallel}: layout_ready is outside the active pre-reaccess lifecycle")
        return
    if off_layout is not None:
        off_active_start = off_result.get("active_stderr_start")
        off_reaccess_start = off_result.get("reaccess_stderr_start")
        if (not isinstance(off_active_start, int) or not isinstance(off_reaccess_start, int) or
                off_layout < off_active_start or off_reaccess_start < off_layout):
            errors.append(f"parallel={parallel}: OFF layout_ready is outside the active pre-reaccess lifecycle")
    for marker in markers:
        if int(marker["_offset"]) < on_layout and marker["offload_attempted"] == "1":
            errors.append(f"parallel={parallel}: OFFLOAD occurred before layout_ready")
    validate_markers(markers, parallel, on_capability, errors, missing)
    closure_offloads = validate_workload(markers, parallel, on_layout, reaccess_start, errors, missing)
    validate_resume(cases["GOVERNOR_ON"], closure_offloads, on_capability, errors, missing, parallel)


def main_parse(root: pathlib.Path) -> tuple[str, list[str]]:
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL or manifest.get("protocol_version") != PROTOCOL_VERSION:
        raise Error("protocol mismatch")
    if manifest.get("runner_status") == "UNSUPPORTED":
        return "UNSUPPORTED", [str(manifest.get("unsupported_reason", "unspecified"))]
    errors: list[str] = []
    missing: list[str] = []
    for field in ("branch", "head", "dirty_status", "runner", "parser", "binary", "model", "parameters", "runs"):
        if field not in manifest:
            errors.append(f"manifest missing {field}")
    for field in ("runner", "parser", "binary", "model"):
        item = manifest.get(field)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or item.get("size", 0) <= 0 or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            errors.append(f"manifest invalid identity {field}")
    parameters = manifest.get("parameters", {})
    if (parameters.get("parallels") != [2, 3] or parameters.get("kv_unified") is not True or
            parameters.get("governor_max_blocks", 0) < 2 or
            parameters.get("active_n_predict") != -1):
        errors.append("manifest does not declare exact parallel=2/3 unified multi-block matrix with endless active producer")
    # Seed request and active socket timeouts must cover parallel=3 worst-case prefill
    # (~30.74s measured) with margin; require at least 90s so completed requests are
    # not reported as transport_error. They remain configurable for slower hosts.
    min_prefill_timeout = 90.0
    for key in ("request_timeout_seconds", "active_socket_timeout_seconds"):
        value = parameters.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < min_prefill_timeout:
            errors.append(f"manifest {key}={value!r} does not cover parallel=3 worst-case prefill (>= {min_prefill_timeout}s)")
    exact_timeouts = {
        "active_cancel_timeout_seconds": 8.0,
        "marker_timeout_seconds": 20.0,
        "layout_timeout_seconds": 20.0,
        "server_timeout_seconds": 300,
        "server_term_timeout_seconds": 10.0,
        "server_kill_timeout_seconds": 5.0,
    }
    if any(parameters.get(key) != value for key, value in exact_timeouts.items()):
        errors.append("manifest does not declare the bounded workload and cleanup timeout contract")
    runs = manifest.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"2", "3"}:
        errors.append("manifest missing parallel matrix")
    else:
        for parallel in (2, 3):
            if runs[str(parallel)].get("parallel") != parallel or set(runs[str(parallel)].get("cases", {})) != set(CASES):
                errors.append(f"manifest malformed parallel={parallel} run")
            validate_unit(root, parallel, errors, missing)
    if manifest.get("runner_status") != "run_complete":
        return "FAIL", [f"runner status is {manifest.get('runner_status')!r}", *errors, *missing]
    if errors:
        return "FAIL", errors
    if missing:
        return "UNSUPPORTED", missing
    return "PASS", []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=pathlib.Path)
    ap.add_argument("--result-path", type=pathlib.Path)
    ap.add_argument("--verify-result", type=pathlib.Path)
    args = ap.parse_args()
    if args.verify_result:
        old = read_json(args.verify_result)
        if old.get("status") != "PASS" or old.get("parser_sha256") != digest(pathlib.Path(__file__)):
            raise SystemExit("FAIL: saved result is not a verified PASS")
        print("PASS (verified)")
        return
    try:
        status, messages = main_parse(args.artifact)
    except (Error, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        status, messages = "FAIL", [str(exc)]
    for message in messages:
        print(f"{status}: {message}", file=sys.stderr)
    if args.result_path:
        args.result_path.write_text(json.dumps({"status": status, "errors": messages, "parser_sha256": digest(pathlib.Path(__file__)), "manifest_sha256": digest(args.artifact / "manifest.json") if (args.artifact / "manifest.json").is_file() else None}, indent=2, sort_keys=True) + "\n")
    print(status)
    raise SystemExit(0 if status == "PASS" else 3 if status == "UNSUPPORTED" else 1)


if __name__ == "__main__":
    main()
