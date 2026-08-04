#!/usr/bin/env python3
"""Fail-closed verdict authority for the G0-S1 single-session KV roundtrip gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "kv_governor_g0_s1"
PROTOCOL_VERSION = 1
SOURCE_MARKER_SCHEMA = "kv_governor_stage3c_1c_2b_1r/v6"
MARKER = "kv_pressure_unified_action"
RESUME_MARKER = "kv_resume_order_event"
RESIDENT_OBSERVATION_MARKER = "kv_g0_s1_resident_observation"
CAPABILITY_MARKER = "KV_GOVERNOR_CAPABILITY"
CASES = ("OFF", "GOVERNOR_ON")
UINT = re.compile(r"[0-9]+$")
SHA256 = re.compile(r"[0-9a-f]{64}$")

MARKER_REQUIRED = {
    "state", "source", "stale", "decision_id", "episode", "target_bytes",
    "max_blocks", "observed_excess_bytes", "debt_before_bytes", "debt_after_bytes",
    "offload_armed_before", "offload_armed_after", "next_action_sample",
    "evaluate_attempted", "evaluate_outcome", "evaluate_reason", "release_attempted",
    "offload_attempted", "selected_seq_id", "selected_claimant_epoch",
    "transaction_id", "outcome", "reason", "blocks", "bytes", "relieved_bytes",
    "shortfall_bytes", "io_failure", "io_errno", "state_changed", "decision_reason",
    "sample_count", "idle", "claimants", "scores",
}
RESUME_REQUIRED = {
    "phase", "decision_id", "seq_id", "claimant_epoch", "transaction_id", "action",
    "outcome", "reason", "graph_allowed",
}
CAPABILITY_REQUIRED = {
    "n_slots", "n_seq_max", "n_stream", "kv_unified", "paged_metadata",
    "ingraph_gather", "release_supported", "offload_supported", "prefetch_supported",
    "backing_ready", "swap_explicit_only",
}
ENABLED_CAPABILITY = {
    "kv_unified", "paged_metadata", "ingraph_gather", "release_supported",
    "offload_supported", "prefetch_supported", "backing_ready", "swap_explicit_only",
}
CLAIMANT_FIELDS = {
    "seq_id", "epoch", "active", "exhausted", "valid", "target_blocks",
    "eligible_resident_blocks", "swapped_blocks", "shared_blocks", "blocked_blocks",
}
RAW_CLAIMANT_FIELDS = CLAIMANT_FIELDS - {"seq_id", "active"}
RESIDENT_OBSERVATION_REQUIRED = {
    "source", "action", "decision_id", "seq_id", "transaction_id", "server_pid",
    "before_available", "before_object_id", "before_generation", "before_page_size",
    "before_total_bytes", "before_resident_bytes", "before_total_pages", "before_resident_pages",
    "after_available", "after_object_id", "after_generation", "after_page_size",
    "after_total_bytes", "after_resident_bytes", "after_total_pages", "after_resident_pages",
}


class Error(Exception):
    pass


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: pathlib.Path) -> Any:
    if not path.is_file():
        raise Error(f"missing {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
        raise Error(
            f"{token} schema mismatch missing={sorted(required-set(values))} "
            f"extra={sorted(set(values)-required)}")
    return values


def parse_marker(line: str) -> dict[str, str]:
    values = parse_fields(line, MARKER, MARKER_REQUIRED)
    symbolic = {
        "state", "source", "evaluate_outcome", "evaluate_reason", "outcome", "reason",
        "decision_reason", "claimants", "scores",
    }
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
    for key in (
            "stale", "offload_armed_before", "offload_armed_after", "evaluate_attempted",
            "release_attempted", "offload_attempted", "io_failure", "state_changed", "idle"):
        if values[key] not in {"0", "1"}:
            raise Error(f"{key} must be boolean")
    if int(values["release_attempted"]) + int(values["offload_attempted"]) > 1:
        raise Error("marker attempts multiple actions")
    return values


def parse_resume(line: str) -> dict[str, str]:
    values = parse_fields(line, RESUME_MARKER, RESUME_REQUIRED)
    for key in ("decision_id", "seq_id", "claimant_epoch", "transaction_id"):
        if not UINT.fullmatch(values[key]):
            raise Error(f"resume {key} must be unsigned integer")
    if values["phase"] not in {"prefetch", "graph_gate"}:
        raise Error("invalid resume phase")
    if values["action"] != "prefetch" or values["graph_allowed"] not in {"0", "1"}:
        raise Error("invalid resume event")
    return values


def parse_resident_observation(line: str) -> dict[str, str]:
    values = parse_fields(line, RESIDENT_OBSERVATION_MARKER, RESIDENT_OBSERVATION_REQUIRED)
    if values["source"] != "paged_sample_mincore" or values["action"] != "offload":
        raise Error("invalid resident observation source or action")
    for key in RESIDENT_OBSERVATION_REQUIRED - {"source", "action"}:
        if not UINT.fullmatch(values[key]):
            raise Error(f"resident observation {key} must be unsigned integer")
    for key in ("before_available", "after_available"):
        if values[key] not in {"0", "1"}:
            raise Error(f"resident observation {key} must be boolean")
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


def token_lines(
        data: bytes,
        token: str,
        parser: Callable[[str], dict[str, str]],
        origin: str,
        errors: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    offset = 0
    for number, raw_line in enumerate(data.splitlines(keepends=True), 1):
        line = raw_line.decode("utf-8", errors="replace")
        if token in line:
            try:
                record: dict[str, Any] = parser(line)
                record["_offset"] = offset
                record["_end"] = offset + len(raw_line)
                result.append(record)
            except Error as exc:
                errors.append(f"{origin}:{number}: {exc}")
        offset += len(raw_line)
    return result


def identity_valid(item: Any, label: str, errors: list[str]) -> None:
    if not isinstance(item, dict):
        errors.append(f"manifest invalid identity {label}")
        return
    if (not isinstance(item.get("path"), str) or not item["path"] or
            not isinstance(item.get("size"), int) or isinstance(item["size"], bool) or
            item["size"] <= 0 or not SHA256.fullmatch(str(item.get("sha256", "")))):
        errors.append(f"manifest invalid identity {label}")


def validate_manifest(manifest: Any, errors: list[str]) -> bool:
    if not isinstance(manifest, dict):
        errors.append("manifest is not an object")
        return False
    if manifest.get("protocol") != PROTOCOL or manifest.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("protocol mismatch")
    if manifest.get("source_marker_schema") != SOURCE_MARKER_SCHEMA:
        errors.append("source marker schema mismatch")
    required = {
        "timestamp_utc", "finished_timestamp_utc", "branch", "head", "dirty_status",
        "tracked_diff_fingerprint", "capture_mode", "runner", "parser", "parameters",
        "case_names", "runner_status",
    }
    if manifest.get("runner_status") != "UNSUPPORTED":
        required |= {"binary", "model"}
    for field in required:
        if field not in manifest:
            errors.append(f"manifest missing {field}")
    if not isinstance(manifest.get("branch"), str) or not manifest.get("branch"):
        errors.append("manifest invalid branch")
    if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("head", ""))):
        errors.append("manifest invalid HEAD")
    if not isinstance(manifest.get("dirty_status"), list) or not all(isinstance(x, str) for x in manifest.get("dirty_status", [])):
        errors.append("manifest invalid dirty_status")
    if not SHA256.fullmatch(str(manifest.get("tracked_diff_fingerprint", ""))):
        errors.append("manifest invalid tracked_diff_fingerprint")
    if manifest.get("capture_mode") not in {"archival_clean", "diagnostic_dirty"}:
        errors.append("manifest invalid capture_mode")
    for label in ("runner", "parser"):
        identity_valid(manifest.get(label), label, errors)
    for label in ("binary", "model"):
        if manifest.get("runner_status") != "UNSUPPORTED" or label in manifest:
            identity_valid(manifest.get(label), label, errors)
    if manifest.get("case_names") != list(CASES):
        errors.append("manifest missing, duplicate, or unexpected case names")
    return not errors


def validate_parameters(parameters: Any, errors: list[str]) -> None:
    if not isinstance(parameters, dict):
        errors.append("manifest parameters are invalid")
        return
    exact = {
        "parallel": 1,
        "n_stream": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "temperature": 0.0,
        "seed": 1,
        "step1_n_predict": 0,
        "step2_n_predict": 32,
        "mincore_requested": True,
        "source_marker_schema": SOURCE_MARKER_SCHEMA,
    }
    for key, expected in exact.items():
        if parameters.get(key) != expected:
            errors.append(f"manifest parameter {key} must be {expected!r}")
    for key in ("paged_block_size", "ctx_size", "governor_target_bytes", "governor_max_blocks"):
        value = parameters.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"manifest parameter {key} is invalid")
    for key in ("paged_block_size", "ctx_size", "governor_target_bytes"):
        if isinstance(parameters.get(key), int) and parameters[key] <= 0:
            errors.append(f"manifest parameter {key} must be positive")
    if isinstance(parameters.get("governor_max_blocks"), int) and parameters["governor_max_blocks"] < 2:
        errors.append("manifest governor_max_blocks is below two")
    for key in ("prefix_text_sha256", "query_text_sha256"):
        if not SHA256.fullmatch(str(parameters.get(key, ""))):
            errors.append(f"manifest parameter {key} is invalid")


def read_rows(path: pathlib.Path, errors: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing {path}")
        return []
    result: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("not object")
            result.append(item)
        except Exception as exc:
            errors.append(f"malformed request record {path}:{number}: {exc}")
    return result


def rows_by_label(rows: list[dict[str, Any]], name: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = row.get("label")
        if not isinstance(label, str) or label in result:
            errors.append(f"{name}: missing or duplicate request label")
            continue
        result[label] = row
    if set(result) != {"step1", "step2"}:
        errors.append(f"{name}: missing or unexpected request labels")
    return result


def validate_request(row: Any, label: str, name: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        errors.append(f"{name}: missing {label} request")
        return
    request = row.get("request")
    required = {"prompt", "n_predict", "temperature", "seed", "cache_prompt", "id_slot", "stream"}
    if not isinstance(request, dict) or set(request) != required:
        errors.append(f"{name}: {label} request schema mismatch")
        return
    prompt = request.get("prompt")
    if (not isinstance(prompt, list) or not prompt or
            any(not isinstance(token, int) or isinstance(token, bool) for token in prompt)):
        errors.append(f"{name}: {label} request prompt is invalid")
    expected_predict = 0 if label == "step1" else 32
    if (request.get("n_predict") != expected_predict or request.get("temperature") != 0.0 or
            request.get("seed") != 1 or request.get("cache_prompt") is not True or
            request.get("id_slot") != 0 or request.get("stream") is not False):
        errors.append(f"{name}: {label} request parameters are not fixed")
    raw = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if row.get("request_sha256") != sha256_bytes(raw):
        errors.append(f"{name}: {label} request hash mismatch")
    text = row.get("response_text")
    if not isinstance(text, str):
        errors.append(f"{name}: {label} response text is invalid")
    else:
        if row.get("response_sha256") != sha256_bytes(text.encode("utf-8")):
            errors.append(f"{name}: {label} response hash mismatch")
        if label == "step2" and not text:
            errors.append(f"{name}: step2 continuation is empty")
    if row.get("http_status") != 200:
        errors.append(f"{name}: {label} did not receive HTTP 200")
    started, finished = row.get("started_monotonic_ns"), row.get("finished_monotonic_ns")
    if (not isinstance(started, int) or isinstance(started, bool) or
            not isinstance(finished, int) or isinstance(finished, bool) or started <= 0 or finished < started):
        errors.append(f"{name}: {label} request timestamps are invalid")


def validate_request_pair(rows: dict[str, dict[str, Any]], name: str, errors: list[str]) -> None:
    validate_request(rows.get("step1"), "step1", name, errors)
    validate_request(rows.get("step2"), "step2", name, errors)
    first = rows.get("step1", {}).get("request", {})
    second = rows.get("step2", {}).get("request", {})
    if isinstance(first, dict) and isinstance(second, dict):
        first_prompt, second_prompt = first.get("prompt"), second.get("prompt")
        if (not isinstance(first_prompt, list) or not isinstance(second_prompt, list) or
                len(first_prompt) >= len(second_prompt) or second_prompt[:len(first_prompt)] != first_prompt):
            errors.append(f"{name}: tokens(P) is not a strict prefix of tokens(P+Q)")
        first_row, second_row = rows.get("step1", {}), rows.get("step2", {})
        first_finished, second_started = first_row.get("finished_monotonic_ns"), second_row.get("started_monotonic_ns")
        if (not isinstance(first_finished, int) or isinstance(first_finished, bool) or
                not isinstance(second_started, int) or isinstance(second_started, bool) or
                first_finished >= second_started):
            errors.append(f"{name}: step2 does not begin after completed step1")


def normalize_argv(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return None
    result = list(value)
    try:
        result[result.index("--port") + 1] = "<port>"
    except (ValueError, IndexError):
        return None
    return result


def validate_server_identity(
        value: Any,
        expected_argv: list[str] | None,
        name: str,
        errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != {"pid", "starttime_ticks", "cmdline", "cmdline_sha256"}:
        errors.append(f"{name}: invalid server identity")
        return None
    pid, starttime, cmdline = value.get("pid"), value.get("starttime_ticks"), value.get("cmdline")
    if (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or
            not isinstance(starttime, int) or isinstance(starttime, bool) or starttime <= 0 or
            not isinstance(cmdline, list) or not cmdline or not all(isinstance(arg, str) for arg in cmdline)):
        errors.append(f"{name}: invalid server identity")
        return None
    computed = sha256_bytes(b"\0".join(arg.encode("utf-8") for arg in cmdline))
    if value.get("cmdline_sha256") != computed:
        errors.append(f"{name}: server command identity hash mismatch")
    if expected_argv is None or cmdline != expected_argv:
        errors.append(f"{name}: server cmdline differs from recorded execution argv")
    return value


def validate_cleanup(case: pathlib.Path, identity: dict[str, Any] | None, errors: list[str]) -> None:
    try:
        value = read_json(case / "cleanup.json")
    except Error as exc:
        errors.append(str(exc))
        return
    if not isinstance(value, dict) or set(value) != {"server", "backing"}:
        errors.append(f"{case}: malformed cleanup record")
        return
    server, backing = value.get("server"), value.get("backing")
    if not isinstance(server, dict):
        errors.append(f"{case}: missing server cleanup")
    else:
        for key in ("pid", "pgid", "term_timed_out", "kill_timed_out", "residual_process", "exit_code"):
            if key not in server:
                errors.append(f"{case}: server cleanup missing {key}")
        if identity is not None and server.get("pid") != identity.get("pid"):
            errors.append(f"{case}: cleanup PID differs from execution identity")
        if server.get("term_timed_out") is not False or server.get("kill_timed_out") is not False or server.get("residual_process") is not False:
            errors.append(f"{case}: server cleanup is incomplete")
    if not isinstance(backing, dict):
        errors.append(f"{case}: missing backing cleanup")
    else:
        expected = (case / "backing").resolve()
        if (backing.get("path") != str(expected) or backing.get("environment_value") != "backing" or
                backing.get("created") is not True or backing.get("cleanup_attempted") is not True or
                backing.get("exists_after_cleanup") is not False or backing.get("cleanup_error") is not None):
            errors.append(f"{case}: backing cleanup is incomplete")
        if (case / "backing").exists():
            errors.append(f"{case}: backing directory remains after cleanup")


def validate_claimant(value: Any, name: str, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != CLAIMANT_FIELDS:
        errors.append(f"{name}: claimant schema mismatch")
        return None
    seq_id = value.get("seq_id")
    bools = ("active", "exhausted", "valid")
    counts = ("target_blocks", "eligible_resident_blocks", "swapped_blocks", "shared_blocks", "blocked_blocks")
    if (not isinstance(seq_id, int) or isinstance(seq_id, bool) or seq_id != 0 or
            not isinstance(value.get("epoch"), int) or isinstance(value.get("epoch"), bool) or
            value["epoch"] < 1 or any(not isinstance(value.get(key), bool) for key in bools) or
            any(not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < 0 for key in counts)):
        errors.append(f"{name}: claimant values are invalid")
        return None
    if value["target_blocks"] != sum(value[key] for key in counts if key != "target_blocks"):
        errors.append(f"{name}: claimant block accounting mismatch")
    return value


def validate_raw_claimant(slot: Any, name: str, errors: list[str]) -> dict[str, Any] | None:
    value = slot.get("kv_claimant") if isinstance(slot, dict) else None
    if not isinstance(value, dict) or set(value) != RAW_CLAIMANT_FIELDS:
        errors.append(f"{name}: claimant schema mismatch")
        return None
    return validate_claimant({
        "seq_id": slot.get("id"),
        "active": slot.get("is_processing"),
        **value,
    }, name, errors)


def safe_raw_path(case: pathlib.Path, raw_path: Any, name: str, errors: list[str]) -> pathlib.Path | None:
    if not isinstance(raw_path, str) or not raw_path:
        errors.append(f"{name}: raw path is invalid")
        return None
    path = (case / raw_path).resolve()
    try:
        path.relative_to(case.resolve())
    except ValueError:
        errors.append(f"{name}: raw path escapes case directory")
        return None
    if not path.is_file():
        errors.append(f"{name}: raw file is missing")
        return None
    return path


def validate_snapshot(
        case: pathlib.Path,
        filename: str,
        name: str,
        errors: list[str]) -> tuple[dict[str, Any] | None, int | None, int | None]:
    try:
        value = read_json(case / filename)
    except Error as exc:
        errors.append(str(exc))
        return None, None, None
    required = {"raw_path", "raw_sha256", "observed_monotonic_ns", "stderr_end", "claimant"}
    if not isinstance(value, dict) or set(value) != required:
        errors.append(f"{name}: {filename} schema mismatch")
        return None, None, None
    path = safe_raw_path(case, value.get("raw_path"), f"{name}: {filename}", errors)
    if path is not None and value.get("raw_sha256") != digest(path):
        errors.append(f"{name}: {filename} raw hash mismatch")
    observed, stderr_end = value.get("observed_monotonic_ns"), value.get("stderr_end")
    if (not isinstance(observed, int) or isinstance(observed, bool) or observed <= 0 or
            not isinstance(stderr_end, int) or isinstance(stderr_end, bool) or stderr_end < 0):
        errors.append(f"{name}: {filename} observation boundary is invalid")
    claimant = validate_claimant(value.get("claimant"), f"{name}: {filename}", errors)
    if path is not None:
        try:
            slots = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{name}: {filename} raw slots JSON is invalid: {exc}")
        else:
            if not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict):
                errors.append(f"{name}: {filename} raw slots snapshot is invalid")
            else:
                slot = slots[0]
                if slot.get("id") != 0 or slot.get("is_processing") is not False:
                    errors.append(f"{name}: {filename} does not bind idle slot 0")
                raw_claimant = validate_raw_claimant(slot, f"{name}: {filename} raw", errors)
                if claimant is not None and raw_claimant is not None and claimant != raw_claimant:
                    errors.append(f"{name}: {filename} derived claimant differs from raw slots")
    valid_stderr_end = stderr_end if isinstance(stderr_end, int) and not isinstance(stderr_end, bool) else None
    valid_observed = observed if isinstance(observed, int) and not isinstance(observed, bool) else None
    return claimant, valid_stderr_end, valid_observed


def validate_capability(case: pathlib.Path, raw: bytes, name: str, errors: list[str]) -> dict[str, str] | None:
    records = token_lines(raw, CAPABILITY_MARKER, parse_capability, f"{case}/server.stderr", errors)
    if len(records) != 1:
        errors.append(f"{name}: capability marker is missing or duplicated")
        return None
    try:
        recorded = read_json(case / "capability.json")
    except Error as exc:
        errors.append(str(exc))
        return None
    if not isinstance(recorded, dict) or set(recorded) != {"offset", "end", "fields"}:
        errors.append(f"{name}: capability evidence schema mismatch")
    else:
        record = records[0]
        if (recorded.get("offset") != record["_offset"] or recorded.get("end") != record["_end"] or
                recorded.get("fields") != {key: record[key] for key in CAPABILITY_REQUIRED}):
            errors.append(f"{name}: capability evidence does not bind raw stderr")
    capability = {key: records[0][key] for key in CAPABILITY_REQUIRED}
    expected = {"n_slots": "1", "n_seq_max": "1", "n_stream": "1"}
    if any(capability[key] != value for key, value in expected.items()):
        errors.append(f"{name}: single-session capability dimensions are wrong")
    for key in ENABLED_CAPABILITY:
        if capability[key] != "1":
            errors.append(f"{name}: capability {key}=0")
    return capability


def marker_claimant(marker: dict[str, Any], name: str, errors: list[str]) -> dict[str, Any] | None:
    encoded = marker.get("claimants")
    if not isinstance(encoded, str) or encoded == "none":
        errors.append(f"{name}: marker lacks runtime claimant observation")
        return None
    entries = encoded.split(";")
    if len(entries) != 1:
        errors.append(f"{name}: marker does not cover exactly seq 0")
        return None
    fields = entries[0].split(":")
    if len(fields) != 10 or any(not UINT.fullmatch(field) for field in fields):
        errors.append(f"{name}: malformed runtime claimant observation")
        return None
    seq, epoch, active, exhausted, valid, target, eligible, swapped, shared, blocked = map(int, fields)
    if active not in (0, 1) or exhausted not in (0, 1) or valid not in (0, 1):
        errors.append(f"{name}: runtime claimant booleans are invalid")
        return None
    return validate_claimant({
        "seq_id": seq,
        "epoch": epoch,
        "active": bool(active),
        "exhausted": bool(exhausted),
        "valid": bool(valid),
        "target_blocks": target,
        "eligible_resident_blocks": eligible,
        "swapped_blocks": swapped,
        "shared_blocks": shared,
        "blocked_blocks": blocked,
    }, name, errors)


def marker_selected_score(
        marker: dict[str, Any],
        selected_seq_id: int,
        name: str,
        errors: list[str]) -> dict[str, Any] | None:
    encoded = marker.get("scores")
    if not isinstance(encoded, str) or encoded == "none":
        errors.append(f"{name}: marker lacks claimant scores")
        return None
    matches: list[dict[str, Any]] = []
    for entry in encoded.split(";"):
        fields = entry.split(":")
        if (len(fields) != 10 or not UINT.fullmatch(fields[0]) or
                fields[1] not in {"0", "1"} or not fields[2] or
                any(not re.fullmatch(r"-?[0-9]+", value) for value in fields[3:])):
            errors.append(f"{name}: malformed claimant score")
            return None
        score = {
            "seq_id": int(fields[0]),
            "eligible": fields[1] == "1",
            "exclusion": fields[2],
        }
        if score["seq_id"] == selected_seq_id:
            matches.append(score)
    if len(matches) != 1:
        errors.append(f"{name}: selected claimant score is missing or duplicated")
        return None
    return matches[0]


def validate_marker_reference(
        case: pathlib.Path,
        filename: str,
        markers: list[dict[str, Any]],
        name: str,
        errors: list[str]) -> dict[str, Any] | None:
    try:
        evidence = read_json(case / filename)
    except Error as exc:
        errors.append(str(exc))
        return None
    if not isinstance(evidence, dict) or set(evidence) != {"offset", "end", "fields"}:
        errors.append(f"{name}: {filename} schema mismatch")
        return None
    for marker in markers:
        fields = {key: marker[key] for key in MARKER_REQUIRED}
        if evidence.get("offset") == marker["_offset"] and evidence.get("end") == marker["_end"] and evidence.get("fields") == fields:
            return marker
    errors.append(f"{name}: {filename} does not bind an exact raw marker")
    return None


def validate_resident_evidence(
        case: pathlib.Path,
        raw: bytes,
        identity: dict[str, Any] | None,
        offload: dict[str, Any] | None,
        name: str,
        errors: list[str]) -> None:
    if offload is None:
        return
    records = token_lines(
        raw, RESIDENT_OBSERVATION_MARKER, parse_resident_observation,
        f"{case}/server.stderr", errors)
    try:
        value = read_json(case / "resident.json")
    except Error as exc:
        errors.append(str(exc))
        return
    if not isinstance(value, dict) or set(value) != {"offset", "end", "fields"}:
        errors.append(f"{name}: resident evidence schema mismatch")
        return
    record = next((item for item in records if
        value.get("offset") == item["_offset"] and value.get("end") == item["_end"] and
        value.get("fields") == {key: item[key] for key in RESIDENT_OBSERVATION_REQUIRED}), None)
    if record is None:
        errors.append(f"{name}: resident evidence does not bind an exact raw observation")
        return

    matching = [item for item in records if item["transaction_id"] == offload["transaction_id"]]
    if len(matching) != 1 or matching[0] is not record:
        errors.append(f"{name}: resident observation is missing or duplicated for the OFFLOAD transaction")
    fields = record
    if (fields["decision_id"] != offload["decision_id"] or
            fields["seq_id"] != offload["selected_seq_id"] or
            fields["transaction_id"] != offload["transaction_id"]):
        errors.append(f"{name}: resident observation does not bind the OFFLOAD transaction")
    if identity is None or int(fields["server_pid"]) != identity.get("pid"):
        errors.append(f"{name}: resident observation does not bind the server PID")
    if fields["before_available"] != "1" or fields["after_available"] != "1":
        errors.append(f"{name}: resident observation sampling is unavailable")
        return

    def sample(prefix: str) -> dict[str, int] | None:
        result = {key: int(fields[f"{prefix}_{key}"]) for key in (
            "object_id", "generation", "page_size", "total_bytes", "resident_bytes",
            "total_pages", "resident_pages")}
        if (result["object_id"] == 0 or result["generation"] == 0 or result["page_size"] == 0 or
                result["total_bytes"] == 0 or result["total_pages"] == 0 or
                result["resident_bytes"] > result["total_bytes"] or
                result["resident_pages"] > result["total_pages"] or
                result["total_bytes"] != result["total_pages"] * result["page_size"] or
                result["resident_bytes"] != result["resident_pages"] * result["page_size"]):
            errors.append(f"{name}: {prefix} resident sample accounting is invalid")
            return None
        return result

    before, after = sample("before"), sample("after")
    if before is None or after is None:
        return
    for key in ("object_id", "generation", "page_size", "total_bytes", "total_pages"):
        if before[key] != after[key]:
            errors.append(f"{name}: resident samples do not bind one KV object/generation")
            break
    if before["resident_bytes"] <= after["resident_bytes"]:
        errors.append(f"{name}: resident bytes did not decline across OFFLOAD")
    if before["resident_pages"] <= after["resident_pages"]:
        errors.append(f"{name}: resident pages did not decline across OFFLOAD")


def validate_resume(
        case: pathlib.Path,
        raw: bytes,
        offload: dict[str, Any] | None,
        epoch: int | None,
        step2: dict[str, Any] | None,
        name: str,
        errors: list[str]) -> tuple[int | None, int | None]:
    try:
        scope = read_json(case / "resume_scope.json")
    except Error as exc:
        errors.append(str(exc))
        return None, None
    required = {"start", "end", "request_label", "request_started_monotonic_ns", "request_finished_monotonic_ns"}
    if not isinstance(scope, dict) or set(scope) != required:
        errors.append(f"{name}: resume scope schema mismatch")
        return None, None
    start, end = scope.get("start"), scope.get("end")
    if (not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or
            start < 0 or end < start or end > len(raw) or scope.get("request_label") != "step2"):
        errors.append(f"{name}: resume scope bounds are invalid")
        return None, None
    if not isinstance(step2, dict) or (
            scope.get("request_started_monotonic_ns") != step2.get("started_monotonic_ns") or
            scope.get("request_finished_monotonic_ns") != step2.get("finished_monotonic_ns")):
        errors.append(f"{name}: resume scope is not bound to the recorded step2 request")
    if offload is not None and start < offload["_end"]:
        errors.append(f"{name}: reaccess begins before OFFLOAD evidence completed")
    events = token_lines(raw[start:end], RESUME_MARKER, parse_resume, f"{case}/server.stderr", errors)
    for event in events:
        event["_offset"] += start
        event["_end"] += start
    if len(events) != 2:
        errors.append(f"{name}: PREFETCH/graph_gate pair is missing or duplicated")
        return start, end
    prefetch, graph_gate = events
    if prefetch["phase"] != "prefetch" or graph_gate["phase"] != "graph_gate":
        errors.append(f"{name}: resume events are out of order")
        return start, end
    expected = {key: prefetch[key] for key in RESUME_REQUIRED - {"phase"}}
    if {key: graph_gate[key] for key in RESUME_REQUIRED - {"phase"}} != expected:
        errors.append(f"{name}: graph_gate does not close the PREFETCH event")
    if (prefetch["seq_id"] != "0" or epoch is None or prefetch["claimant_epoch"] != str(epoch) or
            int(prefetch["transaction_id"]) <= 0 or prefetch["action"] != "prefetch" or
            prefetch["outcome"] != "completed" or prefetch["graph_allowed"] != "1"):
        errors.append(f"{name}: PREFETCH is not a completed same-session restore")
    return start, end


def validate_on_markers(
        case: pathlib.Path,
        raw: bytes,
        post: dict[str, Any] | None,
        expected_blocks: int | None,
        name: str,
        errors: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    markers = token_lines(raw, MARKER, parse_marker, f"{case}/server.stderr", errors)
    if not markers:
        errors.append(f"{name}: missing governor markers")
        return None, None
    offload = validate_marker_reference(case, "offload.json", markers, name, errors)
    if offload is None:
        return None, None

    selected_seq_id = int(offload["selected_seq_id"])
    candidate = marker_claimant(offload, f"{name}: OFFLOAD marker", errors)
    selected_score = marker_selected_score(
        offload, selected_seq_id, f"{name}: OFFLOAD marker", errors)
    blocks = int(offload["blocks"])
    if not (
            offload["state"] in {"PRESSURE", "CRITICAL"} and offload["stale"] == "0" and
            offload["evaluate_attempted"] == "1" and offload["offload_attempted"] == "1" and
            offload["release_attempted"] == "0" and offload["offload_armed_before"] == "1" and
            selected_seq_id == 0 and offload["outcome"] == "completed" and
            offload["state_changed"] == "1" and int(offload["transaction_id"]) > 0 and
            blocks >= 2 and int(offload["bytes"]) > 0 and int(offload["relieved_bytes"]) > 0 and
            offload["io_failure"] == "0" and offload["idle"] == "1"):
        errors.append(f"{name}: OFFLOAD does not satisfy the state-changing transaction contract")
    if expected_blocks is not None and blocks != expected_blocks:
        errors.append(f"{name}: OFFLOAD does not cover the complete step1 KV workload")
    if selected_score is not None and (
            not selected_score["eligible"] or selected_score["exclusion"] != "none"):
        errors.append(f"{name}: selected claimant score is not eligible")
    if candidate is not None:
        if (candidate["seq_id"] != selected_seq_id or
                offload["selected_claimant_epoch"] != str(candidate["epoch"]) or
                candidate["active"] or candidate["exhausted"] or not candidate["valid"] or
                candidate["eligible_resident_blocks"] != blocks or
                candidate["swapped_blocks"] != 0 or candidate["shared_blocks"] != 0 or
                candidate["blocked_blocks"] != 0):
            errors.append(f"{name}: OFFLOAD marker does not bind one eligible pre-transaction claimant")
        if expected_blocks is not None and candidate["target_blocks"] != expected_blocks:
            errors.append(f"{name}: OFFLOAD claimant does not match the complete step1 KV workload")
        if post is not None and (
                post["seq_id"] != candidate["seq_id"] or post["epoch"] != candidate["epoch"] or
                post["target_blocks"] != candidate["target_blocks"] or
                candidate["eligible_resident_blocks"] - post["eligible_resident_blocks"] != blocks or
                post["swapped_blocks"] - candidate["swapped_blocks"] != blocks or
                post["shared_blocks"] != candidate["shared_blocks"] or
                post["blocked_blocks"] != candidate["blocked_blocks"]):
            errors.append(f"{name}: post claimant does not close the OFFLOAD block transition")
    return offload, candidate


def load_case(case: pathlib.Path, name: str, manifest: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    try:
        result = read_json(case / "result.json")
    except Error as exc:
        errors.append(str(exc))
        return {"upstream_failure": True}
    if not isinstance(result, dict):
        errors.append(f"{name}: result schema is invalid")
        return {"upstream_failure": True}
    if result.get("status") != "complete" or result.get("request_loop_started") is not True:
        reason = result.get("workload_error")
        if not isinstance(reason, str) or not reason:
            reason = result.get("cleanup_error")
        if not isinstance(reason, str) or not reason:
            reason = "workload did not complete"
        errors.append(f"{name}: {reason}")
        return {"result": result, "upstream_failure": True}

    required = (
        "execution.json", "environment.json", "capability.json", "requests.jsonl",
        "server.stdout", "server.stderr", "cleanup.json")
    for filename in required:
        if not (case / filename).is_file():
            errors.append(f"{name}: missing {filename}")
    try:
        execution = read_json(case / "execution.json")
        environment = read_json(case / "environment.json")
    except Error as exc:
        errors.append(str(exc))
        return {}
    if not isinstance(execution, dict) or not isinstance(environment, dict):
        errors.append(f"{name}: execution/environment schema is invalid")
        return {}
    if execution.get("environment") != environment:
        errors.append(f"{name}: execution environment differs from environment.json")
    raw_argv = execution.get("argv")
    argv = normalize_argv(raw_argv)
    if argv is None or not isinstance(raw_argv, list):
        errors.append(f"{name}: execution argv is invalid")
        raw_argv = None
    for label in ("binary", "model"):
        if execution.get(label) != manifest.get(label):
            errors.append(f"{name}: execution {label} identity differs from manifest")
    if isinstance(raw_argv, list):
        binary_path = manifest.get("binary", {}).get("path") if isinstance(manifest.get("binary"), dict) else None
        model_path = manifest.get("model", {}).get("path") if isinstance(manifest.get("model"), dict) else None
        if not raw_argv or raw_argv[0] != binary_path:
            errors.append(f"{name}: execution argv is not bound to binary identity")
        model_positions = [index for index, value in enumerate(raw_argv) if value == "--model"]
        if len(model_positions) != 1 or model_positions[0] + 1 >= len(raw_argv) or raw_argv[model_positions[0] + 1] != model_path:
            errors.append(f"{name}: execution argv is not bound to model identity")
    identity = validate_server_identity(execution.get("server_identity"), raw_argv, name, errors)
    raw = (case / "server.stderr").read_bytes() if (case / "server.stderr").is_file() else b""
    capability = validate_capability(case, raw, name, errors)
    rows = rows_by_label(read_rows(case / "requests.jsonl", errors), name, errors)
    validate_request_pair(rows, name, errors)
    validate_cleanup(case, identity, errors)
    return {
        "argv": argv,
        "environment": environment,
        "identity": identity,
        "raw": raw,
        "capability": capability,
        "rows": rows,
        "result": result,
    }


def validate_off(case: pathlib.Path, data: dict[str, Any], errors: list[str]) -> None:
    raw = data.get("raw", b"")
    markers = token_lines(raw, MARKER, parse_marker, f"{case}/server.stderr", errors)
    if markers:
        errors.append("OFF: baseline contains governor marker")
    environment = data.get("environment")
    if not isinstance(environment, dict):
        return
    for key in (
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"):
        if key in environment:
            errors.append(f"OFF: unified governor environment key {key} is present")


def validate_on(case: pathlib.Path, data: dict[str, Any], manifest: dict[str, Any], errors: list[str]) -> None:
    environment, raw, identity = data.get("environment"), data.get("raw", b""), data.get("identity")
    if not isinstance(environment, dict):
        return
    expected = {
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_INGRAPH": "1",
        "LLAMA_KV_PAGED_SWAP": "1",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
        "LLAMA_KV_PAGED_MINCORE": "1",
        "LLAMA_KV_SWAP_DIR": "backing",
        "LLAMA_KV_PRESSURE_SAMPLER": "1",
        "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "1",
        "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "1",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
    }
    for key, expected_value in expected.items():
        if environment.get(key) != expected_value:
            errors.append(f"GOVERNOR_ON: environment {key} must be {expected_value}")
    for key in ("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"):
        value = environment.get(key)
        if not isinstance(value, str) or not UINT.fullmatch(value) or int(value) == 0:
            errors.append(f"GOVERNOR_ON: environment {key} is invalid")
    if isinstance(environment.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"), str) and int(environment["LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"]) < 2:
        errors.append("GOVERNOR_ON: max blocks is below two")
    parameters = manifest.get("parameters") if isinstance(manifest.get("parameters"), dict) else {}
    expected_block_size = parameters.get("paged_block_size")
    if environment.get("LLAMA_KV_PAGED_BLOCK_SIZE") != str(expected_block_size):
        errors.append("GOVERNOR_ON: paged block-size environment differs from manifest")
    rows = data.get("rows", {})
    step1 = rows.get("step1") if isinstance(rows, dict) else None
    step2 = rows.get("step2") if isinstance(rows, dict) else None
    expected_blocks: int | None = None
    step1_request = step1.get("request") if isinstance(step1, dict) else None
    step1_prompt = step1_request.get("prompt") if isinstance(step1_request, dict) else None
    if (isinstance(expected_block_size, int) and expected_block_size > 0 and
            isinstance(step1_prompt, list)):
        if len(step1_prompt) < 2 * expected_block_size or len(step1_prompt) % expected_block_size != 0:
            errors.append("GOVERNOR_ON: step1 prompt does not cover whole multi-block KV pages")
        else:
            expected_blocks = len(step1_prompt) // expected_block_size

    post, post_stderr_end, post_observed = validate_snapshot(
        case, "post_claimant.json", "GOVERNOR_ON", errors)
    offload, candidate = validate_on_markers(
        case, raw, post, expected_blocks, "GOVERNOR_ON", errors)
    epoch = candidate.get("epoch") if candidate is not None else None
    resume_start, _resume_end = validate_resume(
        case, raw, offload, epoch, step2, "GOVERNOR_ON", errors)
    if offload is not None and post_stderr_end is not None and post_stderr_end < offload["_end"]:
        errors.append("GOVERNOR_ON: post-OFFLOAD snapshot predates completed OFFLOAD")
    if resume_start is not None and post_stderr_end is not None and post_stderr_end > resume_start:
        errors.append("GOVERNOR_ON: post-OFFLOAD snapshot crosses reaccess")
    if isinstance(step2, dict) and post_observed is not None and post_observed >= step2.get("started_monotonic_ns", 0):
        errors.append("GOVERNOR_ON: post-OFFLOAD snapshot is not before step2")
    if post is not None and (post["active"] or not post["valid"]):
        errors.append("GOVERNOR_ON: post-OFFLOAD claimant is not an idle valid same-session slot")
    validate_resident_evidence(case, raw, identity, offload, "GOVERNOR_ON", errors)


def validate_pair(off: dict[str, Any], on: dict[str, Any], errors: list[str]) -> None:
    off_argv, on_argv = off.get("argv"), on.get("argv")
    if off_argv is None or on_argv is None or off_argv != on_argv:
        errors.append("OFF/GOVERNOR_ON argv differ")
    off_env, on_env = off.get("environment"), on.get("environment")
    governor_keys = {
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES",
        "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS",
    }
    if not isinstance(off_env, dict) or not isinstance(on_env, dict):
        errors.append("OFF/GOVERNOR_ON environment is invalid")
    else:
        changed = {key for key in set(off_env) | set(on_env) if off_env.get(key) != on_env.get(key)}
        if changed != governor_keys:
            errors.append("OFF/GOVERNOR_ON environment differs outside unified governor keys")
    off_capability, on_capability = off.get("capability"), on.get("capability")
    if off_capability is None or on_capability is None or off_capability != on_capability:
        errors.append("OFF/GOVERNOR_ON capability differs")
    off_rows, on_rows = off.get("rows", {}), on.get("rows", {})
    for label in ("step1", "step2"):
        off_row, on_row = off_rows.get(label), on_rows.get(label)
        if not isinstance(off_row, dict) or not isinstance(on_row, dict):
            continue
        if off_row.get("request") != on_row.get("request") or off_row.get("request_sha256") != on_row.get("request_sha256"):
            errors.append(f"OFF/GOVERNOR_ON {label} request differs")
    off_step2, on_step2 = off_rows.get("step2"), on_rows.get("step2")
    if isinstance(off_step2, dict) and isinstance(on_step2, dict):
        if (off_step2.get("response_text") != on_step2.get("response_text") or
                off_step2.get("response_sha256") != on_step2.get("response_sha256")):
            errors.append("OFF/GOVERNOR_ON step2 continuation output differs")


def validate_preworkload_unsupported(root: pathlib.Path, errors: list[str]) -> None:
    for case in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        name = case.name
        requests = case / "requests.jsonl"
        if requests.is_file() and requests.stat().st_size > 0:
            errors.append(f"UNSUPPORTED artifact contains {name} request evidence")
        result = case / "result.json"
        if result.is_file():
            try:
                value = read_json(result)
            except Error as exc:
                errors.append(str(exc))
            else:
                if not isinstance(value, dict) or value.get("request_loop_started") is not False:
                    errors.append(f"UNSUPPORTED artifact contains an executed {name} workload")
        stderr = case / "server.stderr"
        if stderr.is_file():
            markers = token_lines(stderr.read_bytes(), MARKER, parse_marker, f"{stderr}", errors)
            if markers:
                errors.append(f"UNSUPPORTED artifact contains {name} governor action markers")


def main_parse(root: pathlib.Path) -> tuple[str, list[str]]:
    try:
        manifest = read_json(root / "manifest.json")
    except Error as exc:
        return "FAIL", [str(exc)]
    errors: list[str] = []
    validate_manifest(manifest, errors)
    if not isinstance(manifest, dict):
        return "FAIL", errors
    validate_parameters(manifest.get("parameters"), errors)
    if manifest.get("runner_status") == "UNSUPPORTED":
        reason = manifest.get("unsupported_reason")
        if not isinstance(reason, str) or not reason:
            errors.append("UNSUPPORTED manifest lacks reason")
        if manifest.get("unsupported_stage") not in {"pre_workload_binary_model", "pre_workload_physical_probe"}:
            errors.append("UNSUPPORTED manifest is not a permitted pre-workload condition")
        validate_preworkload_unsupported(root, errors)
        return ("FAIL", errors) if errors else ("UNSUPPORTED", [reason])
    if manifest.get("runner_status") != "run_complete":
        reason = manifest.get("runner_error")
        if not isinstance(reason, str) or not reason:
            reason = f"runner status is {manifest.get('runner_status')!r}"
        errors.append(reason)
    if errors:
        return "FAIL", errors

    off = load_case(root / "OFF", "OFF", manifest, errors)
    if off.get("upstream_failure"):
        return "FAIL", errors
    validate_off(root / "OFF", off, errors)
    if errors:
        return "FAIL", errors

    on = load_case(root / "GOVERNOR_ON", "GOVERNOR_ON", manifest, errors)
    if on.get("upstream_failure"):
        return "FAIL", errors
    validate_on(root / "GOVERNOR_ON", on, manifest, errors)
    if errors:
        return "FAIL", errors
    validate_pair(off, on, errors)
    return ("FAIL", errors) if errors else ("PASS", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact", type=pathlib.Path)
    ap.add_argument("--result-path", type=pathlib.Path)
    ap.add_argument("--verify-result", type=pathlib.Path)
    args = ap.parse_args()
    if args.verify_result:
        status, details = main_parse(args.artifact)
        if status != "PASS":
            raise SystemExit(f"FAIL: artifact no longer verifies as PASS: {details}")
        try:
            old = read_json(args.verify_result)
        except Error as exc:
            raise SystemExit(f"FAIL: {exc}") from exc
        if old.get("status") != "PASS" or old.get("parser_sha256") != digest(pathlib.Path(__file__)):
            raise SystemExit("FAIL: saved result is not a verified PASS")
        print("PASS (verified)")
        return
    status, details = main_parse(args.artifact)
    result = {
        "status": status,
        "details": details,
        "parser": str(pathlib.Path(__file__)),
        "parser_sha256": digest(pathlib.Path(__file__)),
    }
    if args.result_path:
        args.result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(status)
    for detail in details:
        print(f"- {detail}")
    raise SystemExit(0 if status == "PASS" else 3 if status == "UNSUPPORTED" else 1)


if __name__ == "__main__":
    main()
