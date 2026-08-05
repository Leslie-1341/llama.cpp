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
SUPPORTED_CTX_SIZES = {1024, 2048, 4096, 8064}
PAGED_BLOCK_SIZE = 64
MIN_PREFIX_TOKENS = 2 * PAGED_BLOCK_SIZE
N_PREDICT = 32
WORKLOAD_RESULT_FIELDS = {
    "target_prefix_tokens", "actual_p_token_count", "actual_pq_token_count",
    "actual_p_block_count", "target_prefix_token_delta",
}
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


def token_ids_sha256(tokens: list[int]) -> str:
    return sha256_bytes(json.dumps(tokens, separators=(",", ":")).encode("utf-8"))


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
        "step2_n_predict": N_PREDICT,
        "mincore_requested": True,
        "source_marker_schema": SOURCE_MARKER_SCHEMA,
        "paged_block_size": PAGED_BLOCK_SIZE,
    }
    for key, expected in exact.items():
        if parameters.get(key) != expected:
            errors.append(f"manifest parameter {key} must be {expected!r}")
    ctx_size = parameters.get("ctx_size")
    if (not isinstance(ctx_size, int) or isinstance(ctx_size, bool) or
            ctx_size not in SUPPORTED_CTX_SIZES):
        errors.append("manifest parameter ctx_size is unsupported")
    target_prefix_tokens = parameters.get("target_prefix_tokens")
    if (not isinstance(target_prefix_tokens, int) or isinstance(target_prefix_tokens, bool) or
            target_prefix_tokens < MIN_PREFIX_TOKENS):
        errors.append("manifest parameter target_prefix_tokens is invalid")
    elif isinstance(ctx_size, int):
        aligned_target = target_prefix_tokens // PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE
        if aligned_target + 1 + N_PREDICT > ctx_size:
            errors.append("manifest target_prefix_tokens cannot fit a strict P+Q workload in ctx_size")
    for key in ("governor_target_bytes", "governor_max_blocks"):
        value = parameters.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            errors.append(f"manifest parameter {key} is invalid")
    if isinstance(parameters.get("governor_target_bytes"), int) and parameters["governor_target_bytes"] <= 0:
        errors.append("manifest parameter governor_target_bytes must be positive")
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


def argv_option(
        argv: list[str] | None,
        option: str,
        name: str,
        errors: list[str]) -> str | None:
    if argv is None:
        return None
    positions = [index for index, value in enumerate(argv) if value == option]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        errors.append(f"{name}: execution argv must contain exactly one {option}")
        return None
    return argv[positions[0] + 1]


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


def validate_workload(
        case: pathlib.Path,
        name: str,
        result: dict[str, Any],
        rows: dict[str, dict[str, Any]],
        manifest: dict[str, Any],
        errors: list[str]) -> dict[str, Any] | None:
    try:
        workload = read_json(case / "workload.json")
    except Error as exc:
        errors.append(str(exc))
        return None
    if not isinstance(workload, dict):
        errors.append(f"{name}: workload schema is invalid")
        return None

    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        return workload
    ctx_size = parameters.get("ctx_size")
    target_prefix_tokens = parameters.get("target_prefix_tokens")
    block_size = parameters.get("paged_block_size")
    step1 = rows.get("step1")
    step2 = rows.get("step2")
    step1_request = step1.get("request") if isinstance(step1, dict) else None
    step2_request = step2.get("request") if isinstance(step2, dict) else None
    prompt_p = step1_request.get("prompt") if isinstance(step1_request, dict) else None
    prompt_pq = step2_request.get("prompt") if isinstance(step2_request, dict) else None
    if not isinstance(prompt_p, list) or not isinstance(prompt_pq, list):
        return workload
    if (not isinstance(ctx_size, int) or isinstance(ctx_size, bool) or
            not isinstance(target_prefix_tokens, int) or isinstance(target_prefix_tokens, bool) or
            not isinstance(block_size, int) or isinstance(block_size, bool) or block_size <= 0):
        return workload

    actual_p_tokens = len(prompt_p)
    actual_pq_tokens = len(prompt_pq)
    actual_p_blocks = actual_p_tokens // block_size
    target_delta = target_prefix_tokens - actual_p_tokens
    expected_result = {
        "target_prefix_tokens": target_prefix_tokens,
        "actual_p_token_count": actual_p_tokens,
        "actual_pq_token_count": actual_pq_tokens,
        "actual_p_block_count": actual_p_blocks,
        "target_prefix_token_delta": target_delta,
    }
    if result.get("ctx_size") != ctx_size:
        errors.append(f"{name}: result ctx_size differs from manifest")
    if result.get("target_prefix_tokens") != target_prefix_tokens:
        errors.append(f"{name}: result target_prefix_tokens differs from manifest")
    result_workload = result.get("workload")
    if not isinstance(result_workload, dict) or set(result_workload) != WORKLOAD_RESULT_FIELDS:
        errors.append(f"{name}: result workload summary schema mismatch")
    elif result_workload != expected_result:
        errors.append(f"{name}: result workload summary differs from actual requests")

    expected_workload = {
        "ctx_size": ctx_size,
        "target_prefix_tokens": target_prefix_tokens,
        "actual_p_token_count": actual_p_tokens,
        "actual_pq_token_count": actual_pq_tokens,
        "actual_p_block_count": actual_p_blocks,
        "target_prefix_token_delta": target_delta,
        "prefix_token_count": actual_p_tokens,
        "prefix_block_count": actual_p_blocks,
        "continuation_prompt_token_count": actual_pq_tokens,
        "n_predict": N_PREDICT,
        "context_token_count": actual_pq_tokens + N_PREDICT,
        "prefix_tokens_sha256": token_ids_sha256(prompt_p),
        "continuation_prompt_tokens_sha256": token_ids_sha256(prompt_pq),
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            errors.append(f"{name}: workload {key} differs from actual request/manifest evidence")
    if workload.get("status") != "ready" or workload.get("failure_reasons") != []:
        errors.append(f"{name}: workload is not a successful fail-closed construction")
    if workload.get("token_boundary_policy") != "largest_paged_block_boundary_within_tokenized_common_prefix":
        errors.append(f"{name}: workload token-boundary policy mismatch")
    if workload.get("strict_prefix") is not True or prompt_pq[:actual_p_tokens] != prompt_p:
        errors.append(f"{name}: workload does not bind a strict P/P+Q token prefix")
    if workload.get("target_prefix_satisfied") is not True:
        errors.append(f"{name}: workload does not declare target satisfaction")
    if actual_p_tokens % block_size != 0 or actual_p_blocks < 2:
        errors.append(f"{name}: actual P is not a complete multi-block prefix")
    if target_delta < 0 or target_delta >= block_size:
        errors.append(f"{name}: target and actual P differ by at least one full block")
    if actual_pq_tokens + N_PREDICT > ctx_size or workload.get("context_fits") is not True:
        errors.append(f"{name}: actual P+Q workload exceeds ctx_size")
    unit_count = workload.get("prefix_text_unit_count")
    max_units = workload.get("prefix_text_max_units")
    if (not isinstance(unit_count, int) or isinstance(unit_count, bool) or unit_count <= 0 or
            not isinstance(max_units, int) or isinstance(max_units, bool) or max_units < unit_count or
            not SHA256.fullmatch(str(workload.get("selected_prefix_text_sha256", "")))):
        errors.append(f"{name}: scalable prefix construction evidence is invalid")
    return workload


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


def validate_resident_sample(
        fields: dict[str, Any],
        prefix: str,
        name: str,
        errors: list[str]) -> dict[str, int] | None:
    result = {key: int(fields[f"{prefix}_{key}"]) for key in (
        "object_id", "generation", "page_size", "total_bytes", "resident_bytes",
        "total_pages", "resident_pages")}
    if (
            result["object_id"] == 0 or result["generation"] == 0 or
            result["page_size"] == 0 or result["total_bytes"] == 0 or
            result["total_pages"] == 0 or
            result["resident_bytes"] > result["total_bytes"] or
            result["resident_pages"] > result["total_pages"] or
            result["total_bytes"] != result["total_pages"] * result["page_size"] or
            result["resident_bytes"] != result["resident_pages"] * result["page_size"]):
        errors.append(f"{name}: {prefix} resident sample accounting is invalid")
        return None
    return result


def validate_resume_scope(
        case: pathlib.Path,
        raw: bytes,
        step2: dict[str, Any] | None,
        name: str,
        errors: list[str]) -> tuple[int | None, int | None]:
    try:
        scope = read_json(case / "resume_scope.json")
    except Error as exc:
        errors.append(str(exc))
        return None, None
    required = {
        "start", "end", "request_label", "request_started_monotonic_ns",
        "request_finished_monotonic_ns",
    }
    if not isinstance(scope, dict) or set(scope) != required:
        errors.append(f"{name}: resume scope schema mismatch")
        return None, None
    start, end = scope.get("start"), scope.get("end")
    if (
            not isinstance(start, int) or isinstance(start, bool) or
            not isinstance(end, int) or isinstance(end, bool) or
            start < 0 or end < start or end > len(raw) or
            scope.get("request_label") != "step2"):
        errors.append(f"{name}: resume scope bounds are invalid")
        return None, None
    if not isinstance(step2, dict) or (
            scope.get("request_started_monotonic_ns") != step2.get("started_monotonic_ns") or
            scope.get("request_finished_monotonic_ns") != step2.get("finished_monotonic_ns")):
        errors.append(f"{name}: resume scope is not bound to the recorded step2 request")
    return start, end


def validate_resume(
        case: pathlib.Path,
        raw: bytes,
        start: int | None,
        end: int | None,
        last_offload_end: int | None,
        epoch: int | None,
        name: str,
        errors: list[str]) -> None:
    if start is None or end is None:
        return
    if last_offload_end is not None and start < last_offload_end:
        errors.append(f"{name}: reaccess begins before cumulative OFFLOAD evidence completed")
    events = token_lines(
        raw[start:end], RESUME_MARKER, parse_resume,
        f"{case}/server.stderr", errors)
    for event in events:
        event["_offset"] += start
        event["_end"] += start
    if len(events) != 2:
        errors.append(f"{name}: PREFETCH/graph_gate pair is missing or duplicated")
        return
    prefetch, graph_gate = events
    if prefetch["phase"] != "prefetch" or graph_gate["phase"] != "graph_gate":
        errors.append(f"{name}: resume events are out of order")
        return
    expected = {key: prefetch[key] for key in RESUME_REQUIRED - {"phase"}}
    if {key: graph_gate[key] for key in RESUME_REQUIRED - {"phase"}} != expected:
        errors.append(f"{name}: graph_gate does not close the PREFETCH event")
    if (
            prefetch["seq_id"] != "0" or epoch is None or
            prefetch["claimant_epoch"] != str(epoch) or
            int(prefetch["transaction_id"]) <= 0 or
            prefetch["action"] != "prefetch" or prefetch["outcome"] != "completed" or
            prefetch["graph_allowed"] != "1"):
        errors.append(f"{name}: PREFETCH is not a completed same-session restore")


def validate_on_transactions(
        case: pathlib.Path,
        raw: bytes,
        identity: dict[str, Any] | None,
        post: dict[str, Any] | None,
        expected_blocks: int | None,
        resume_start: int | None,
        name: str,
        errors: list[str]) -> tuple[int | None, int | None]:
    markers = token_lines(raw, MARKER, parse_marker, f"{case}/server.stderr", errors)
    residents = token_lines(
        raw, RESIDENT_OBSERVATION_MARKER, parse_resident_observation,
        f"{case}/server.stderr", errors)
    if not markers:
        errors.append(f"{name}: missing governor markers")
        return None, None
    try:
        evidence = read_json(case / "offload.json")
    except Error as exc:
        errors.append(str(exc))
        return None, None
    required = {
        "status", "scope_start", "scope_end", "expected_blocks", "selected_seq_id",
        "selected_claimant_epoch", "transactions", "cumulative",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        errors.append(f"{name}: cumulative OFFLOAD evidence schema mismatch")
        return None, None

    scope_start = evidence.get("scope_start")
    scope_end = evidence.get("scope_end")
    selected_seq_id = evidence.get("selected_seq_id")
    selected_epoch = evidence.get("selected_claimant_epoch")
    transactions = evidence.get("transactions")
    cumulative = evidence.get("cumulative")
    boundary = resume_start if resume_start is not None else len(raw)
    if evidence.get("status") != "complete":
        errors.append(f"{name}: cumulative OFFLOAD evidence is not complete")
    if (
            not isinstance(scope_start, int) or isinstance(scope_start, bool) or
            not isinstance(scope_end, int) or isinstance(scope_end, bool) or
            scope_start < 0 or scope_end < scope_start or scope_end > boundary):
        errors.append(f"{name}: cumulative OFFLOAD scope is invalid")
        return None, None
    if (
            not isinstance(selected_seq_id, int) or isinstance(selected_seq_id, bool) or
            not isinstance(selected_epoch, int) or isinstance(selected_epoch, bool) or
            selected_seq_id != 0 or selected_epoch <= 0):
        errors.append(f"{name}: cumulative OFFLOAD seq/epoch identity is invalid")
        return None, None
    recorded_expected_blocks = evidence.get("expected_blocks")
    if (
            not isinstance(recorded_expected_blocks, int) or
            isinstance(recorded_expected_blocks, bool) or recorded_expected_blocks <= 0 or
            recorded_expected_blocks != expected_blocks):
        errors.append(f"{name}: cumulative OFFLOAD expected_blocks differs from step1 workload")
    if not isinstance(transactions, list) or not transactions:
        errors.append(f"{name}: cumulative OFFLOAD transactions are missing")
        return selected_epoch, None

    pre_step2_markers = [
        marker for marker in markers if marker["_end"] <= boundary
    ]
    for marker in pre_step2_markers:
        if marker["offload_attempted"] == "1" and int(marker["transaction_id"]) == 0:
            if (
                    marker["state_changed"] != "0" or marker["outcome"] != "no_op" or
                    any(int(marker[key]) != 0 for key in (
                        "blocks", "bytes", "relieved_bytes"))):
                errors.append(f"{name}: transaction=0 OFFLOAD marker is not a no-op")
    changed_markers = [
        marker for marker in pre_step2_markers
        if marker["offload_attempted"] == "1" and
        marker["state_changed"] == "1" and int(marker["transaction_id"]) > 0
    ]

    marker_spans: list[tuple[int, int]] = []
    transaction_ids: set[int] = set()
    resident_spans: set[tuple[int, int]] = set()
    cumulative_blocks = 0
    cumulative_bytes = 0
    cumulative_relief = 0
    first_resident_bytes: int | None = None
    last_resident_bytes: int | None = None
    previous_after: dict[str, int] | None = None
    previous_marker_end = scope_start

    transaction_required = {
        "index", "offset", "end", "fields", "resident", "resident_drop_bytes",
    }
    resident_required = {"offset", "end", "fields"}
    changed_spans = [(marker["_offset"], marker["_end"]) for marker in changed_markers]
    for index, item in enumerate(transactions):
        label = f"{name}: OFFLOAD transaction[{index}]"
        if not isinstance(item, dict) or set(item) != transaction_required:
            errors.append(f"{label} evidence schema mismatch")
            continue
        item_index = item.get("index")
        if (
                not isinstance(item_index, int) or isinstance(item_index, bool) or
                item_index != index):
            errors.append(f"{label} index is not ordered")
        marker = next((record for record in markers if
            item.get("offset") == record["_offset"] and item.get("end") == record["_end"] and
            item.get("fields") == {key: record[key] for key in MARKER_REQUIRED}), None)
        if marker is None:
            errors.append(f"{label} does not bind an exact raw marker")
            continue
        span = (marker["_offset"], marker["_end"])
        marker_spans.append(span)
        if span not in changed_spans:
            errors.append(f"{label} is not a pre-step2 state-changing OFFLOAD")
        if marker["_offset"] < previous_marker_end:
            errors.append(f"{label} marker order overlaps a prior transaction")
        previous_marker_end = marker["_end"]

        seq_id = int(marker["selected_seq_id"])
        epoch = int(marker["selected_claimant_epoch"])
        transaction_id = int(marker["transaction_id"])
        blocks = int(marker["blocks"])
        byte_count = int(marker["bytes"])
        relieved_bytes = int(marker["relieved_bytes"])
        if not (
                marker["state"] in {"PRESSURE", "CRITICAL"} and marker["stale"] == "0" and
                marker["evaluate_attempted"] == "1" and marker["offload_attempted"] == "1" and
                marker["release_attempted"] == "0" and marker["offload_armed_before"] == "1" and
                seq_id == selected_seq_id and epoch == selected_epoch and
                marker["outcome"] == "completed" and marker["state_changed"] == "1" and
                transaction_id > 0 and blocks > 0 and byte_count > 0 and relieved_bytes > 0 and
                marker["io_failure"] == "0" and marker["idle"] == "1"):
            errors.append(f"{label} violates the state-changing transaction contract")
        if transaction_id in transaction_ids:
            errors.append(f"{label} duplicates transaction_id")
        transaction_ids.add(transaction_id)

        candidate = marker_claimant(marker, f"{label} marker", errors)
        selected_score = marker_selected_score(
            marker, seq_id, f"{label} marker", errors)
        if selected_score is not None and (
                not selected_score["eligible"] or selected_score["exclusion"] != "none"):
            errors.append(f"{label} selected claimant score is not eligible")
        if candidate is not None and expected_blocks is not None:
            if (
                    candidate["seq_id"] != selected_seq_id or
                    candidate["epoch"] != selected_epoch or candidate["active"] or
                    candidate["exhausted"] or not candidate["valid"] or
                    candidate["target_blocks"] != expected_blocks or
                    candidate["eligible_resident_blocks"] != expected_blocks - cumulative_blocks or
                    candidate["swapped_blocks"] != cumulative_blocks or
                    candidate["shared_blocks"] != 0 or candidate["blocked_blocks"] != 0 or
                    blocks > candidate["eligible_resident_blocks"]):
                errors.append(f"{label} claimant does not bind the cumulative pre-transaction state")

        resident_evidence = item.get("resident")
        if not isinstance(resident_evidence, dict) or set(resident_evidence) != resident_required:
            errors.append(f"{label} resident evidence schema mismatch")
        else:
            resident_record = next((record for record in residents if
                resident_evidence.get("offset") == record["_offset"] and
                resident_evidence.get("end") == record["_end"] and
                resident_evidence.get("fields") == {
                    key: record[key] for key in RESIDENT_OBSERVATION_REQUIRED}), None)
            if resident_record is None:
                errors.append(f"{label} resident evidence does not bind an exact raw observation")
            else:
                resident_span = (resident_record["_offset"], resident_record["_end"])
                if resident_span in resident_spans:
                    errors.append(f"{label} reuses a resident observation")
                resident_spans.add(resident_span)
                matching = [record for record in residents if
                    record["decision_id"] == marker["decision_id"] and
                    record["seq_id"] == marker["selected_seq_id"] and
                    record["transaction_id"] == marker["transaction_id"]]
                if len(matching) != 1 or matching[0] is not resident_record:
                    errors.append(f"{label} resident observation is missing or duplicated")
                if (
                        resident_record["_offset"] < scope_start or
                        resident_record["_end"] > marker["_offset"] or
                        resident_record["_offset"] < (
                            marker_spans[-2][1] if len(marker_spans) > 1 else scope_start)):
                    errors.append(f"{label} resident observation is out of order")
                if identity is None or int(resident_record["server_pid"]) != identity.get("pid"):
                    errors.append(f"{label} resident observation does not bind the server PID")
                if (
                        resident_record["before_available"] != "1" or
                        resident_record["after_available"] != "1"):
                    errors.append(f"{label} resident observation sampling is unavailable")
                else:
                    before = validate_resident_sample(
                        resident_record, "before", label, errors)
                    after = validate_resident_sample(
                        resident_record, "after", label, errors)
                    if before is not None and after is not None:
                        identity_keys = (
                            "object_id", "generation", "page_size", "total_bytes", "total_pages")
                        if any(before[key] != after[key] for key in identity_keys):
                            errors.append(f"{label} resident samples do not bind one KV object/generation")
                        if previous_after is not None and before != previous_after:
                            errors.append(f"{label} resident observations are not contiguous")
                        resident_drop = before["resident_bytes"] - after["resident_bytes"]
                        if resident_drop <= 0 or before["resident_pages"] <= after["resident_pages"]:
                            errors.append(f"{label} resident bytes/pages did not decline")
                        if resident_drop != relieved_bytes:
                            errors.append(f"{label} resident drop differs from relieved_bytes")
                        recorded_drop = item.get("resident_drop_bytes")
                        if (
                                not isinstance(recorded_drop, int) or
                                isinstance(recorded_drop, bool) or
                                recorded_drop != resident_drop):
                            errors.append(f"{label} recorded resident_drop_bytes is inconsistent")
                        if first_resident_bytes is None:
                            first_resident_bytes = before["resident_bytes"]
                        last_resident_bytes = after["resident_bytes"]
                        previous_after = after

        cumulative_blocks += blocks
        cumulative_bytes += byte_count
        cumulative_relief += relieved_bytes
        if expected_blocks is not None and cumulative_blocks > expected_blocks:
            errors.append(f"{name}: cumulative OFFLOAD blocks exceed the step1 workload")

    if marker_spans != changed_spans:
        errors.append(
            f"{name}: ordered transactions do not cover every pre-step2 state-changing OFFLOAD")
    if marker_spans and scope_end != marker_spans[-1][1]:
        errors.append(f"{name}: cumulative OFFLOAD scope_end differs from the final transaction")
    if expected_blocks is not None and cumulative_blocks != expected_blocks:
        errors.append(f"{name}: cumulative OFFLOAD blocks do not cover the complete step1 workload")

    cumulative_required = {
        "transaction_count", "blocks", "bytes", "relieved_bytes",
        "first_resident_bytes", "last_resident_bytes", "resident_drop_bytes",
    }
    if not isinstance(cumulative, dict) or set(cumulative) != cumulative_required:
        errors.append(f"{name}: cumulative OFFLOAD summary schema mismatch")
    elif any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in cumulative.values()):
        errors.append(f"{name}: cumulative OFFLOAD summary contains invalid counts")
    elif first_resident_bytes is not None and last_resident_bytes is not None:
        expected_cumulative = {
            "transaction_count": len(transactions),
            "blocks": cumulative_blocks,
            "bytes": cumulative_bytes,
            "relieved_bytes": cumulative_relief,
            "first_resident_bytes": first_resident_bytes,
            "last_resident_bytes": last_resident_bytes,
            "resident_drop_bytes": first_resident_bytes - last_resident_bytes,
        }
        if cumulative != expected_cumulative:
            errors.append(f"{name}: cumulative OFFLOAD summary differs from raw transactions")
        if expected_cumulative["resident_drop_bytes"] != cumulative_relief:
            errors.append(f"{name}: cumulative first-to-last resident drop differs from total relief")

    if post is not None and expected_blocks is not None:
        if (
                post["seq_id"] != selected_seq_id or post["epoch"] != selected_epoch or
                post["active"] or post["exhausted"] or not post["valid"] or
                post["target_blocks"] != expected_blocks or
                post["eligible_resident_blocks"] != 0 or
                post["swapped_blocks"] != expected_blocks or
                post["shared_blocks"] != 0 or post["blocked_blocks"] != 0):
            errors.append(f"{name}: final claimant does not close the complete OFFLOAD workload")
    return selected_epoch, marker_spans[-1][1] if marker_spans else None


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
        "execution.json", "environment.json", "capability.json", "workload.json",
        "requests.jsonl", "server.stdout", "server.stderr", "cleanup.json")
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
        parameters = manifest.get("parameters")
        expected_ctx_size = parameters.get("ctx_size") if isinstance(parameters, dict) else None
        argv_ctx_size = argv_option(raw_argv, "--ctx-size", name, errors)
        if argv_ctx_size is not None and argv_ctx_size != str(expected_ctx_size):
            errors.append(f"{name}: execution --ctx-size differs from manifest and CLI evidence")
    identity = validate_server_identity(execution.get("server_identity"), raw_argv, name, errors)
    raw = (case / "server.stderr").read_bytes() if (case / "server.stderr").is_file() else b""
    capability = validate_capability(case, raw, name, errors)
    rows = rows_by_label(read_rows(case / "requests.jsonl", errors), name, errors)
    validate_request_pair(rows, name, errors)
    workload = validate_workload(case, name, result, rows, manifest, errors)
    validate_cleanup(case, identity, errors)
    return {
        "argv": argv,
        "environment": environment,
        "identity": identity,
        "raw": raw,
        "capability": capability,
        "rows": rows,
        "result": result,
        "workload": workload,
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

    resume_start, resume_end = validate_resume_scope(
        case, raw, step2, "GOVERNOR_ON", errors)
    post, post_stderr_end, post_observed = validate_snapshot(
        case, "post_claimant.json", "GOVERNOR_ON", errors)
    epoch, last_offload_end = validate_on_transactions(
        case, raw, identity, post, expected_blocks, resume_start,
        "GOVERNOR_ON", errors)
    validate_resume(
        case, raw, resume_start, resume_end, last_offload_end, epoch,
        "GOVERNOR_ON", errors)
    if (
            last_offload_end is not None and post_stderr_end is not None and
            post_stderr_end < last_offload_end):
        errors.append("GOVERNOR_ON: final claimant snapshot predates cumulative OFFLOAD")
    if (
            resume_start is not None and post_stderr_end is not None and
            post_stderr_end != resume_start):
        errors.append("GOVERNOR_ON: final claimant snapshot does not bind the step2 byte boundary")
    if (
            isinstance(step2, dict) and post_observed is not None and
            post_observed >= step2.get("started_monotonic_ns", 0)):
        errors.append("GOVERNOR_ON: final claimant snapshot is not before step2")


def validate_pair(off: dict[str, Any], on: dict[str, Any], errors: list[str]) -> None:
    off_argv, on_argv = off.get("argv"), on.get("argv")
    if off_argv is None or on_argv is None or off_argv != on_argv:
        errors.append("OFF/GOVERNOR_ON argv differ")
    if off.get("workload") is None or off.get("workload") != on.get("workload"):
        errors.append("OFF/GOVERNOR_ON workload construction evidence differs")
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
