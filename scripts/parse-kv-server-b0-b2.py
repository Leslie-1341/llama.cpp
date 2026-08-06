#!/usr/bin/env python3
"""Fail-closed verdict authority for B0-B2 server baseline artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[1]
BASELINE_PROTOCOL = "kv_server_b0_b2_baseline"
BASELINE_PROTOCOL_VERSION = 1
NOT_APPLICABLE = "NOT_APPLICABLE"
SOURCE_MARKER_SCHEMA = "kv_governor_stage3c_1c_2b_1r/v6"
MARKER = "kv_pressure_unified_action"
RESUME_MARKER = "kv_resume_order_event"
RESIDENT_OBSERVATION_MARKER = "kv_g0_s1_resident_observation"
CAPABILITY_MARKER = "KV_GOVERNOR_CAPABILITY"
SUPPORTED_CTX_SIZES = {1024, 2048, 4096, 8064}
PAGED_BLOCK_SIZE = 64
MIN_PREFIX_TOKENS = 2 * PAGED_BLOCK_SIZE
N_PREDICT = 32
BASELINE_CASES = (
    "CURRENT_E0",
    "FLEXKV_RESIDENT",
    "FLEXKV_K1_SYNC",
)
BASELINE_CASE_LABELS = {
    "CURRENT_E0": "current server with all experimental KV mechanisms explicitly disabled",
    "FLEXKV_RESIDENT": "current server with paged runtime resident and Governor actions disabled",
    "FLEXKV_K1_SYNC": "current server with Governor OFFLOAD and synchronous K1 restore",
}
BASELINE_RUN_PLAN = (
    (1, 1, "CURRENT_E0"),
    (1, 2, "FLEXKV_RESIDENT"),
    (1, 3, "FLEXKV_K1_SYNC"),
    (2, 1, "FLEXKV_K1_SYNC"),
    (2, 2, "FLEXKV_RESIDENT"),
    (2, 3, "CURRENT_E0"),
    (3, 1, "FLEXKV_RESIDENT"),
    (3, 2, "FLEXKV_K1_SYNC"),
    (3, 3, "CURRENT_E0"),
)
BASELINE_COMPARISON_EDGES = {
    "B0": {
        "left": "CURRENT_E0",
        "right": "FLEXKV_RESIDENT",
        "purpose": "current E0 versus resident FlexKV runtime baseline",
    },
    "B1": {
        "left": "FLEXKV_RESIDENT",
        "right": "FLEXKV_K1_SYNC",
        "purpose": "resident FlexKV versus synchronous K1 offload/restore baseline",
    },
}
BASELINE_BASE_ENV = {"HOME": "/tmp", "LANG": "C", "LC_ALL": "C"}
BASELINE_E0_ENV = {
    "LLAMA_KV_ACTIVE_TOKEN_STATS": "1",
    "LLAMA_KV_PAGED_IO_STATS": "1",
    "LLAMA_KV_PAGED_IDENTITY_FAST_PATH": "0",
    "LLAMA_KV_IDLE_NUM_IDLE_SEQS": "2",
    "LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS": "256",
    "LLAMA_KV_CACHE_DEBUG": "0",
    "LLAMA_KV_PAGED_BLOCK_SIZE": "16",
    "LLAMA_KV_PAGED_SHIFT": "0",
    "LLAMA_KV_LAZY_CLEAR": "0",
    "LLAMA_KV_LAZY_TAIL": "0",
    "LLAMA_KV_PAGED": "0",
    "LLAMA_KV_PAGED_INGRAPH": "0",
    "LLAMA_KV_PAGED_GATHER_NONIDENTITY": "0",
    "LLAMA_KV_PAGED_SWAP": "0",
    "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "0",
    "LLAMA_KV_PAGED_MINCORE": "0",
    "LLAMA_KV_PAGED_RELEASE": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MADVISE": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS": "0",
    "LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES": "0",
    "LLAMA_KV_PAGED_SHADOW_VALIDATE": "0",
    "LLAMA_KV_PAGED_TRACE": "0",
    "LLAMA_KV_PAGED_IDLE_TRACE": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE": "0",
    "LLAMA_KV_PAGED_REFAULT_TRACE_MAX": "0",
    "LLAMA_KV_PAGED_TIMING": "0",
    "LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED": "0",
    "LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP": "0",
    "LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS": "0",
    "LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS": "0",
    "LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE": "off",
    "LLAMA_KV_PAGED_RESUME_PENDING_TOKEN": "96",
    "LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME": "0",
    "LLAMA_KV_PAGED_RESUME_PREFETCH": "0",
    "LLAMA_KV_PAGED_RESUME_TIMING": "0",
    "LLAMA_KV_PAGED_RESUME_TIMING_STEP": "0",
    "LLAMA_KV_PRESSURE_SAMPLER": "0",
    "LLAMA_KV_PRESSURE_DRY_RUN": "0",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "0",
    "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "0",
    "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "0",
    "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": "0",
    "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "0",
    "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": "0",
    "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "0",
    "LLAMA_KV_RESUME_STAGE_TIMING": "0",
    "LLAMA_KV_SWAP": "0",
    "LLAMA_KV_SWAP_MODE": "exact",
    "LLAMA_KV_SWAP_WINDOW": "0",
    "LLAMA_KV_SWAP_SINK": "0",
    "LLAMA_KV_SWAP_RSS_SAMPLE": "0",
    "LLAMA_KV_SWAP_MADVISE": "0",
    "LLAMA_KV_SWAP_BACKEND_SELFTEST": "0",
    "LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST": "0",
}
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


def validate_request(
        row: Any,
        label: str,
        name: str,
        errors: list[str],
        step2_stream: bool = False) -> None:
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
    expected_stream = step2_stream if label == "step2" else False
    if (request.get("n_predict") != expected_predict or request.get("temperature") != 0.0 or
            request.get("seed") != 1 or request.get("cache_prompt") is not True or
            request.get("id_slot") != 0 or request.get("stream") is not expected_stream):
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


def validate_request_pair(
        rows: dict[str, dict[str, Any]],
        name: str,
        errors: list[str],
        step2_stream: bool = False) -> None:
    validate_request(rows.get("step1"), "step1", name, errors, step2_stream)
    validate_request(rows.get("step2"), "step2", name, errors, step2_stream)
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


def baseline_validate_k1_transactions(
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
def baseline_case_flags(name: str) -> dict[str, bool]:
    return {
        "flex_runtime": name in {"FLEXKV_RESIDENT", "FLEXKV_K1_SYNC"},
        "backing": name in {"FLEXKV_RESIDENT", "FLEXKV_K1_SYNC"},
        "resident": name in {"FLEXKV_RESIDENT", "FLEXKV_K1_SYNC"},
        "k1_sync": name == "FLEXKV_K1_SYNC",
    }


def baseline_evidence_paths(name: str) -> dict[str, str]:
    flags = baseline_case_flags(name)
    resident = (
        "offload.json" if flags["k1_sync"] else
        "resident_after_step1.json" if flags["resident"] else NOT_APPLICABLE)
    timing = "timing.json" if flags["k1_sync"] else NOT_APPLICABLE
    return {
        "capability": "capability.json",
        "resident": resident,
        "backing": "memory_phases.json" if flags["backing"] else NOT_APPLICABLE,
        "staging": NOT_APPLICABLE,
        "offload": "offload.json" if flags["k1_sync"] else NOT_APPLICABLE,
        "restore": timing,
        "read": timing,
        "gate": timing,
        "io": timing,
    }


def baseline_expected_environment(name: str) -> dict[str, str | None]:
    env: dict[str, str | None] = {**BASELINE_BASE_ENV, "PATH": None}
    env.update(BASELINE_E0_ENV)
    if name == "CURRENT_E0":
        return env
    env.update({
        "LLAMA_KV_PAGED": "1",
        "LLAMA_KV_PAGED_INGRAPH": "1",
        "LLAMA_KV_PAGED_SWAP": "1",
        "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
        "LLAMA_KV_PAGED_MINCORE": "1",
        "LLAMA_KV_PAGED_BLOCK_SIZE": str(PAGED_BLOCK_SIZE),
        "LLAMA_KV_SWAP_DIR": "backing",
        "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": (
            "1" if name == "FLEXKV_K1_SYNC" else "preflight"),
        "LLAMA_KV_PAGED_IO_STATS": "1",
        "LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE": "1",
        "LLAMA_KV_RESUME_STAGE_TIMING": "1",
        "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "100",
        "LLAMA_KV_PRESSURE_LOG_INTERVAL_MS": "1000",
        "LLAMA_KV_LOW_WATER_RSS_KB": "1",
        "LLAMA_KV_PRESSURE_RSS_KB": "2",
        "LLAMA_KV_CRITICAL_RSS_KB": "3",
    })
    if name == "FLEXKV_K1_SYNC":
        env.update({
            "LLAMA_KV_PRESSURE_SAMPLER": "1",
            "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "1",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": str(1 << 30),
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": "64",
        })
    return env


def baseline_metadata(round_no: int, run_order: int, name: str) -> dict[str, Any]:
    return {
        "round": round_no,
        "run_order": run_order,
        "case": name,
        "label": BASELINE_CASE_LABELS[name],
    }


def baseline_plan() -> list[dict[str, Any]]:
    return [
        baseline_metadata(round_no, run_order, name)
        for round_no, run_order, name in BASELINE_RUN_PLAN
    ]


def is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def is_positive_int(value: Any) -> bool:
    return is_nonnegative_int(value) and value > 0


def is_finite_number(value: Any, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and
        math.isfinite(float(value)) and (float(value) > 0 if positive else float(value) >= 0))


def baseline_cgroup_valid(value: Any, label: str, errors: list[str]) -> bool:
    required = {
        "version", "path", "memory_current_file", "memory_max_file", "memory_peak_file",
        "memory_current", "memory_max", "memory_peak_snapshot",
    }
    if not isinstance(value, dict) or set(value) != required:
        errors.append(f"{label}: cgroup schema mismatch")
        return False
    if value.get("version") not in {"v1", "v2", "none"}:
        errors.append(f"{label}: cgroup version is invalid")
        return False
    if not all(isinstance(value.get(key), str) and value[key] for key in required):
        errors.append(f"{label}: cgroup values are invalid")
        return False
    if value["version"] == "none":
        if any(value[key] != NOT_APPLICABLE for key in required - {"version"}):
            errors.append(f"{label}: unavailable cgroup must use NOT_APPLICABLE")
    elif value["path"] == NOT_APPLICABLE or value["memory_current_file"] == NOT_APPLICABLE:
        errors.append(f"{label}: available cgroup lacks path/current identity")
    return True


def baseline_validate_server_cgroup(
        cgroup: dict[str, Any], value: Any, label: str, errors: list[str]) -> None:
    if cgroup.get("version") == "none":
        if value != NOT_APPLICABLE:
            errors.append(f"{label}: unavailable cgroup must not claim child membership")
        return
    if not isinstance(value, list) or not value or not all(isinstance(line, str) for line in value):
        errors.append(f"{label}: server cgroup membership is missing")
        return
    path = cgroup.get("path")
    if not isinstance(path, str):
        errors.append(f"{label}: cgroup path is invalid")
        return
    try:
        if cgroup["version"] == "v2":
            rel = pathlib.PurePosixPath(path).relative_to("/sys/fs/cgroup")
            expected = f"0::/{rel}" if str(rel) != "." else "0::/"
            if expected not in value:
                errors.append(f"{label}: server does not belong to the recorded v2 cgroup")
        else:
            rel = pathlib.PurePosixPath(path).relative_to("/sys/fs/cgroup/memory")
            expected = f"/{rel}" if str(rel) != "." else "/"
            matches = [line for line in value if len(line.split(":", 2)) == 3 and
                       "memory" in line.split(":", 2)[1].split(",") and
                       line.split(":", 2)[2] == expected]
            if len(matches) != 1:
                errors.append(f"{label}: server does not belong to the recorded v1 memory cgroup")
    except ValueError:
        errors.append(f"{label}: cgroup path is outside the expected hierarchy")


def baseline_validate_manifest(manifest: Any, errors: list[str]) -> bool:
    if not isinstance(manifest, dict):
        errors.append("baseline manifest is not an object")
        return False
    required = {
        "protocol", "protocol_version", "source_marker_schema", "timestamp_utc",
        "finished_timestamp_utc", "branch", "head", "dirty_status",
        "tracked_diff_fingerprint", "capture_mode", "runner", "parser", "memory_sampler",
        "binary_requested", "model_requested",
        "binary", "model", "host", "cgroup", "execution", "parameters",
        "case_names", "comparison_edges", "planned_runs", "run_results", "runner_status",
    }
    missing = required - set(manifest)
    if missing:
        errors.append(f"baseline manifest missing {sorted(missing)}")
    if manifest.get("protocol") != BASELINE_PROTOCOL or manifest.get("protocol_version") != BASELINE_PROTOCOL_VERSION:
        errors.append("baseline protocol mismatch")
    if manifest.get("source_marker_schema") != SOURCE_MARKER_SCHEMA:
        errors.append("baseline source marker schema mismatch")
    if not is_nonempty_string(manifest.get("branch")) or not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("head", ""))):
        errors.append("baseline manifest Git identity is invalid")
    if (not isinstance(manifest.get("dirty_status"), list) or
            not all(isinstance(item, str) for item in manifest.get("dirty_status", [])) or
            not SHA256.fullmatch(str(manifest.get("tracked_diff_fingerprint", ""))) or
            manifest.get("capture_mode") not in {"archival_clean", "diagnostic_dirty", "diagnostic_smoke"}):
        errors.append("baseline manifest dirty-tree identity is invalid")
    elif not (bool(manifest.get("dirty_status")) and manifest.get("capture_mode") in {"diagnostic_dirty", "diagnostic_smoke"} or
              not bool(manifest.get("dirty_status")) and manifest.get("capture_mode") in {"archival_clean"} or
              manifest.get("capture_mode") == "diagnostic_smoke"):
        errors.append("baseline capture_mode contradicts the recorded dirty status")
    smoke = manifest.get("execution", {}).get("smoke") is True
    if smoke and manifest.get("capture_mode") != "diagnostic_smoke":
        errors.append("baseline smoke run must have capture_mode diagnostic_smoke")
    for key in ("runner", "parser", "memory_sampler"):
        identity_valid(manifest.get(key), f"baseline {key}", errors)
    for key in ("binary_requested", "model_requested"):
        if not is_nonempty_string(manifest.get(key)):
            errors.append(f"baseline manifest {key} is invalid")
    if manifest.get("case_names") != list(BASELINE_CASES):
        errors.append("baseline case_names mismatch")
    if manifest.get("comparison_edges") != BASELINE_COMPARISON_EDGES:
        errors.append("baseline comparison edge plan mismatch")
    expected_plan = baseline_plan() if not smoke else [
        baseline_metadata(1, idx + 1, name) for idx, name in enumerate(BASELINE_CASES)
    ]
    if manifest.get("planned_runs") != expected_plan:
        errors.append("baseline fixed three-round run plan mismatch")
    if not isinstance(manifest.get("run_results"), list):
        errors.append("baseline run_results is invalid")
    status = manifest.get("runner_status")
    if status not in {"DRY_RUN", "run_complete", "run_incomplete", "run_interrupted"}:
        errors.append("baseline runner status is invalid")
    if not isinstance(manifest.get("host"), dict) or set(manifest["host"]) != {"hostname", "kernel", "os_release"} or not all(
            is_nonempty_string(value) for value in manifest["host"].values()):
        errors.append("baseline host identity is invalid")
    baseline_cgroup_valid(manifest.get("cgroup"), "baseline manifest", errors)
    execution = manifest.get("execution")
    if (not isinstance(execution, dict) or set(execution) != {"dry_run", "allow_dirty", "smoke"} or
            not isinstance(execution.get("dry_run"), bool) or not isinstance(execution.get("allow_dirty"), bool) or
            not isinstance(execution.get("smoke"), bool)):
        errors.append("baseline execution capture mode is invalid")
    elif (manifest.get("runner_status") == "DRY_RUN") != execution["dry_run"]:
        errors.append("baseline execution dry-run state differs from runner status")
    elif (manifest.get("runner_status") == "run_complete" and manifest.get("dirty_status") and not execution["allow_dirty"]):
        errors.append("baseline formal run was captured dirty without explicit allow_dirty")
    parameters = manifest.get("parameters")
    expected_parameters = {
        "parallel": 1, "n_stream": 1, "id_slot": 0, "cache_prompt": True,
        "temperature": 0.0, "seed": 1, "step1_n_predict": 0,
        "step2_n_predict": N_PREDICT, "step2_stream": True, "mincore_requested": True,
        "source_marker_schema": SOURCE_MARKER_SCHEMA, "paged_block_size": PAGED_BLOCK_SIZE,
        "sample_interval_seconds": 0.10,
    }
    if not isinstance(parameters, dict):
        errors.append("baseline parameters are invalid")
    else:
        for key, expected in expected_parameters.items():
            actual = parameters.get(key)
            if isinstance(expected, float):
                if not is_finite_number(actual) or not math.isclose(float(actual), expected, abs_tol=1e-12):
                    errors.append(f"baseline parameter {key} mismatch")
            elif actual != expected:
                errors.append(f"baseline parameter {key} mismatch")
        ctx_size = parameters.get("ctx_size")
        target = parameters.get("target_prefix_tokens")
        if (not isinstance(ctx_size, int) or isinstance(ctx_size, bool) or ctx_size not in SUPPORTED_CTX_SIZES or
                not isinstance(target, int) or isinstance(target, bool) or target < MIN_PREFIX_TOKENS or
                target // PAGED_BLOCK_SIZE * PAGED_BLOCK_SIZE + 1 + N_PREDICT > ctx_size):
            errors.append("baseline ctx/prefix parameters are invalid")
        for key in ("prefix_text_sha256", "query_text_sha256"):
            if not SHA256.fullmatch(str(parameters.get(key, ""))):
                errors.append(f"baseline parameter {key} is invalid")
        for key in ("governor_target_bytes", "governor_max_blocks"):
            if not is_positive_int(parameters.get(key)):
                errors.append(f"baseline parameter {key} is invalid")
        rounds = parameters.get("rounds")
        if not is_positive_int(rounds) or rounds not in {1, 3}:
            errors.append("baseline parameter rounds is invalid")
        if smoke and rounds != 1:
            errors.append("baseline smoke run must have rounds=1")
    if status != "DRY_RUN":
        for key in ("binary", "model"):
            identity_valid(manifest.get(key), f"baseline {key}", errors)
        expected_results = [
            {**item, "status": manifest["run_results"][index].get("status")}
            for index, item in enumerate(expected_plan)
            if index < len(manifest.get("run_results", [])) and isinstance(manifest["run_results"][index], dict)
        ]
        if len(expected_results) != len(expected_plan) or [
                {key: value for key, value in result.items() if key != "status"}
                for result in manifest.get("run_results", []) if isinstance(result, dict)
        ] != expected_plan or any(not is_nonempty_string(result.get("status")) for result in expected_results):
            errors.append("baseline run_results do not bind the fixed plan")
    return not errors


def baseline_validate_matrix(root: pathlib.Path, dry_run: bool, smoke: bool, errors: list[str]) -> list[tuple[dict[str, Any], pathlib.Path]]:
    runs_root = root / "runs"
    if not runs_root.is_dir():
        errors.append("baseline runs directory is missing")
        return []
    expected = baseline_plan() if not smoke else [
        baseline_metadata(1, idx + 1, name) for idx, name in enumerate(BASELINE_CASES)
    ]
    pairs: list[tuple[dict[str, Any], pathlib.Path]] = []
    for path in runs_root.iterdir():
        if not path.is_dir():
            errors.append(f"baseline runs contains a non-directory entry: {path.name}")
            continue
        try:
            metadata = read_json(path / "run.json")
        except Error as exc:
            errors.append(str(exc))
            continue
        if metadata not in expected:
            errors.append(f"baseline run metadata is unexpected: {path}")
            continue
        pairs.append((metadata, path))
    pairs.sort(key=lambda item: (item[0]["round"], item[0]["run_order"]))
    actual = [item[0] for item in pairs]
    if actual != expected:
        errors.append("baseline artifact run matrix does not match the fixed plan")
    if dry_run:
        for metadata, case in pairs:
            try:
                plan = read_json(case / "plan.json")
            except Error as exc:
                errors.append(str(exc))
                continue
            required = {"round", "run_order", "case", "label", "dry_run", "evidence", "environment"}
            if (not isinstance(plan, dict) or set(plan) != required or
                    any(plan.get(key) != metadata[key] for key in metadata) or
                    plan.get("dry_run") is not True or
                    plan.get("evidence") != baseline_evidence_paths(metadata["case"])):
                errors.append(f"{metadata['case']}: dry-run plan differs from the fixed case contract")
            else:
                baseline_validate_environment(metadata["case"], plan.get("environment"), errors)
            for filename in ("requests.jsonl", "server.stdout", "server.stderr", "result.json"):
                if (case / filename).exists():
                    errors.append(f"{metadata['case']}: dry-run artifact contains workload evidence")
    return pairs


def baseline_validate_environment(name: str, value: Any, errors: list[str]) -> dict[str, str] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        errors.append(f"{name}: environment is invalid")
        return None
    expected = baseline_expected_environment(name)
    if set(value) != set(expected):
        errors.append(f"{name}: environment keys differ from the closed case contract")
        return value
    for key, expected_value in expected.items():
        actual = value.get(key)
        if key == "PATH":
            if not actual:
                errors.append(f"{name}: PATH is empty")
        elif actual != expected_value:
            errors.append(f"{name}: environment {key} differs from the closed case contract")
    return value


def baseline_expected_argv(
        binary_path: str,
        model_path: str,
        ctx_size: int,
        name: str) -> list[str]:
    argv = [
        binary_path, "--host", "127.0.0.1", "--port", "<port>", "--model", model_path,
        "--ctx-size", str(ctx_size), "--parallel", "1", "--timeout", "300",
        "--threads", "4", "--n-gpu-layers", "0", "--cache-type-k", "f32",
        "--cache-type-v", "f32", "--no-warmup",
    ]
    argv.extend(("--kv-unified", "--no-cache-idle-slots"))
    return argv


def baseline_validate_execution(
        case: pathlib.Path,
        metadata: dict[str, Any],
        manifest: dict[str, Any],
        errors: list[str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str] | None]:
    name = metadata["case"]
    try:
        execution = read_json(case / "execution.json")
        environment = read_json(case / "environment.json")
    except Error as exc:
        errors.append(str(exc))
        return None, None, None
    required = {
        "case", "round", "run_order", "host", "argv", "cwd", "environment",
        "environment_closure", "binary", "model", "server_identity", "server_cgroup",
        "cgroup", "memory_sampler", "memory_sampler_argv",
    }
    if not isinstance(execution, dict) or set(execution) != required:
        errors.append(f"{name}: execution schema mismatch")
        return None, None, None
    if any(execution.get(key) != metadata[key] for key in ("case", "round", "run_order")):
        errors.append(f"{name}: execution matrix identity differs from run.json")
    if execution.get("host") != "127.0.0.1" or execution.get("cwd") != str(case.resolve()):
        errors.append(f"{name}: execution endpoint/cwd differs from contract")
    env = baseline_validate_environment(name, environment, errors)
    if execution.get("environment") != environment:
        errors.append(f"{name}: execution environment does not bind environment.json")
    expected_closure = {"inherits_parent_environment": False, "base_environment_keys": ["HOME", "LANG", "LC_ALL", "PATH"]}
    if execution.get("environment_closure") != expected_closure:
        errors.append(f"{name}: environment closure metadata is invalid")
    if execution.get("binary") != manifest.get("binary"):
        errors.append(f"{name}: execution binary identity does not match manifest")
    if execution.get("model") != manifest.get("model"):
        errors.append(f"{name}: execution model identity differs from manifest")
    raw_argv = execution.get("argv")
    argv = normalize_argv(raw_argv)
    if argv is None or not isinstance(raw_argv, list):
        errors.append(f"{name}: execution argv is invalid")
    else:
        parameters = manifest.get("parameters")
        ctx_size = parameters.get("ctx_size") if isinstance(parameters, dict) else None
        binary_path = manifest.get("binary", {}).get("path") if isinstance(manifest.get("binary"), dict) else ""
        model_path = manifest.get("model", {}).get("path") if isinstance(manifest.get("model"), dict) else ""
        if not isinstance(ctx_size, int) or argv != baseline_expected_argv(binary_path, model_path, ctx_size, name):
            errors.append(f"{name}: execution argv differs from the fixed case argv")
    identity = validate_server_identity(execution.get("server_identity"), raw_argv if isinstance(raw_argv, list) else None, name, errors)
    cgroup = execution.get("cgroup")
    if baseline_cgroup_valid(cgroup, name, errors) and isinstance(cgroup, dict):
        baseline_validate_server_cgroup(cgroup, execution.get("server_cgroup"), name, errors)
    identity_valid(execution.get("memory_sampler"), f"{name} memory sampler", errors)
    if execution.get("memory_sampler") != manifest.get("memory_sampler"):
        errors.append(f"{name}: execution memory sampler identity differs from manifest")
    if identity is not None and isinstance(cgroup, dict):
        backing_arg = str(case / "backing") if baseline_case_flags(name)["backing"] else ""
        current_file = cgroup.get("memory_current_file")
        current_file = "" if current_file == NOT_APPLICABLE else current_file
        expected_sampler_argv = [
            "bash", str(manifest.get("memory_sampler", {}).get("path", "")), "--sample-process",
            str(identity["pid"]), str(case / "memory_samples.tsv"), backing_arg, "0.10", current_file,
        ]
        if execution.get("memory_sampler_argv") != expected_sampler_argv:
            errors.append(f"{name}: memory sampler argv differs from execution identity")
    return execution, env, argv


def baseline_validate_capability(
        case: pathlib.Path,
        raw: bytes,
        name: str,
        errors: list[str]) -> dict[str, str] | None:
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
        expected = {key: record[key] for key in CAPABILITY_REQUIRED}
        if recorded.get("offset") != record["_offset"] or recorded.get("end") != record["_end"] or recorded.get("fields") != expected:
            errors.append(f"{name}: capability evidence does not bind raw stderr")
    capability = {key: records[0][key] for key in CAPABILITY_REQUIRED}
    if any(capability[key] != "1" for key in ("n_slots", "n_seq_max", "n_stream")):
        errors.append(f"{name}: capability does not bind one server slot/stream")
    if name == "CURRENT_E0":
        if capability.get("kv_unified") != "1" or any(
                capability[key] != "0" for key in ENABLED_CAPABILITY - {"kv_unified"}):
            errors.append("CURRENT_E0: experimental KV capability is not disabled")
    elif any(capability[key] != "1" for key in ENABLED_CAPABILITY):
        errors.append(f"{name}: required FlexKV capability is unavailable")
    return capability


def baseline_validate_memory(
        case: pathlib.Path,
        name: str,
        execution: dict[str, Any] | None,
        result: dict[str, Any],
        errors: list[str]) -> None:
    if execution is None:
        return
    identity = execution.get("server_identity")
    cgroup = execution.get("cgroup")
    if not isinstance(identity, dict) or not isinstance(cgroup, dict):
        return
    try:
        memory = read_json(case / "memory_phases.json")
    except Error as exc:
        errors.append(str(exc))
        return
    expected_phases = ["server_ready", "after_step1"]
    if name == "FLEXKV_K1_SYNC":
        expected_phases.append("after_offload")
    expected_phases.append("after_step2")
    if (not isinstance(memory, dict) or set(memory) != {"schema_version", "sample_interval_seconds", "cgroup", "phases"} or
            memory.get("schema_version") != 1 or not is_finite_number(memory.get("sample_interval_seconds")) or
            not math.isclose(float(memory["sample_interval_seconds"]), 0.10, abs_tol=1e-12) or
            memory.get("cgroup") != cgroup or not isinstance(memory.get("phases"), list)):
        errors.append(f"{name}: memory phase schema/cgroup binding is invalid")
    else:
        phases = memory["phases"]
        if [item.get("phase") for item in phases if isinstance(item, dict)] != expected_phases or len(phases) != len(expected_phases):
            errors.append(f"{name}: required memory phase set is incomplete or reordered")
        previous = -1
        for index, phase in enumerate(phases):
            label = f"{name}: memory phase[{index}]"
            required = {"phase", "monotonic_ns", "server_pid", "vmrss_kb", "vmhwm_kb", "cgroup_memory_current_bytes", "backing"}
            if not isinstance(phase, dict) or set(phase) != required:
                errors.append(f"{label} schema mismatch")
                continue
            if (not is_positive_int(phase.get("monotonic_ns")) or phase["monotonic_ns"] < previous or
                    phase.get("server_pid") != identity.get("pid") or not is_positive_int(phase.get("vmrss_kb")) or
                    not is_nonnegative_int(phase.get("vmhwm_kb")) or phase["vmhwm_kb"] < phase["vmrss_kb"]):
                errors.append(f"{label} process RSS identity is invalid")
            previous = phase.get("monotonic_ns") if is_nonnegative_int(phase.get("monotonic_ns")) else previous
            current = phase.get("cgroup_memory_current_bytes")
            if cgroup.get("version") == "none":
                if current != NOT_APPLICABLE:
                    errors.append(f"{label} unavailable cgroup value is invalid")
            elif not (isinstance(current, str) and current.isdigit()):
                errors.append(f"{label} cgroup memory.current is unavailable")
            backing = phase.get("backing")
            if baseline_case_flags(name)["backing"]:
                if not isinstance(backing, dict) or backing.get("status") not in {"open_fd_observed", "not_observed", "unavailable"}:
                    errors.append(f"{label} backing observation is invalid")
            elif backing != NOT_APPLICABLE:
                errors.append(f"{label} non-backing case reports backing evidence")
    sampler = result.get("memory_sampler")
    if (not isinstance(sampler, dict) or sampler.get("started") is not True or
            not is_positive_int(sampler.get("pid")) or sampler.get("exit_code") != 0 or sampler.get("timed_out") is not False):
        errors.append(f"{name}: memory sampler did not cleanly complete")
    stderr = case / "memory_sampler.stderr"
    if not stderr.is_file() or stderr.read_bytes():
        errors.append(f"{name}: memory sampler stderr is missing or nonempty")
    samples = case / "memory_samples.tsv"
    expected_header = [
        "elapsed_ms", "pid", "starttime_ticks", "vmrss_kb", "vmhwm_kb",
        "cgroup_memory_current_bytes", "backing_logical_size", "backing_allocated_bytes",
    ]
    if not samples.is_file():
        errors.append(f"{name}: memory sampler output is missing")
        return
    try:
        with samples.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rows = list(reader)
            if reader.fieldnames != expected_header:
                errors.append(f"{name}: memory sampler header differs from the shared schema")
    except OSError as exc:
        errors.append(f"{name}: cannot read memory sampler output: {exc}")
        return
    if not rows:
        errors.append(f"{name}: memory sampler captured zero samples")
        return
    for index, row in enumerate(rows):
        label = f"{name}: memory sample[{index}]"
        try:
            elapsed = int(row.get("elapsed_ms", ""))
            pid = int(row.get("pid", ""))
            starttime = int(row.get("starttime_ticks", ""))
            vmrss = int(row.get("vmrss_kb", ""))
            vmhwm = int(row.get("vmhwm_kb", ""))
        except ValueError:
            errors.append(f"{label} lacks numeric process identity/RSS")
            continue
        if (elapsed < 0 or pid != identity.get("pid") or starttime != identity.get("starttime_ticks") or
                vmrss <= 0 or vmhwm < vmrss):
            errors.append(f"{label} does not bind the server process identity")
        current = row.get("cgroup_memory_current_bytes")
        if cgroup.get("version") == "none":
            if current != NOT_APPLICABLE:
                errors.append(f"{label} has invalid unavailable cgroup value")
        elif current is None or not current.isdigit():
            errors.append(f"{label} lacks cgroup memory.current")
        for key in ("backing_logical_size", "backing_allocated_bytes"):
            value = row.get(key)
            if baseline_case_flags(name)["backing"]:
                if value not in {NOT_APPLICABLE, "NA", None} and not str(value).isdigit():
                    errors.append(f"{label} {key} is invalid")
            elif value not in {NOT_APPLICABLE, "NA"}:
                errors.append(f"{label} non-backing case has {key}")


def baseline_validate_step2_metrics(
        case: pathlib.Path,
        rows: dict[str, dict[str, Any]],
        errors: list[str]) -> None:
    try:
        metrics = read_json(case / "step2_metrics.json")
    except Error as exc:
        errors.append(str(exc))
        return
    step2 = rows.get("step2")
    required = {
        "schema_version", "request_label", "ttft_source", "tpot_source", "total_duration_source",
        "tps_source", "ttft_ms", "tpot_ms", "total_duration_ms", "tps", "response_timings",
    }
    if not isinstance(metrics, dict) or set(metrics) != required:
        errors.append(f"{case.name}: step2 metric schema mismatch")
        return
    expected_sources = {
        "schema_version": 1,
        "request_label": "step2",
        "ttft_source": "http_stream_first_nonempty_content",
        "tpot_source": "response.timings.predicted_per_token_ms",
        "total_duration_source": "http_stream_wall",
        "tps_source": "n_predict/http_stream_wall",
    }
    if any(metrics.get(key) != value for key, value in expected_sources.items()):
        errors.append(f"{case.name}: step2 metric source is invalid")
    if not isinstance(step2, dict):
        return
    timings = step2.get("response_timings")
    if metrics.get("response_timings") != timings or not isinstance(timings, dict) or timings.get("predicted_n") != N_PREDICT:
        errors.append(f"{case.name}: step2 metric timings are not bound to the response")
        return
    start, first, finish = step2.get("started_monotonic_ns"), step2.get("first_content_monotonic_ns"), step2.get("finished_monotonic_ns")
    if not all(is_positive_int(value) for value in (start, first, finish)) or first < start or finish < first:
        errors.append(f"{case.name}: step2 stream timestamps are invalid")
        return
    expected_ttft = (first - start) / 1_000_000.0
    expected_total = (finish - start) / 1_000_000.0
    tpot = timings.get("predicted_per_token_ms")
    if (not is_finite_number(tpot, positive=True) or expected_total <= 0 or
            not is_finite_number(metrics.get("ttft_ms")) or not is_finite_number(metrics.get("tpot_ms"), positive=True) or
            not is_finite_number(metrics.get("total_duration_ms"), positive=True) or not is_finite_number(metrics.get("tps"), positive=True) or
            not math.isclose(float(metrics["ttft_ms"]), expected_ttft, abs_tol=1e-9) or
            not math.isclose(float(metrics["tpot_ms"]), float(tpot), abs_tol=1e-9) or
            not math.isclose(float(metrics["total_duration_ms"]), expected_total, abs_tol=1e-9) or
            not math.isclose(float(metrics["tps"]), N_PREDICT * 1000.0 / expected_total, abs_tol=1e-9)):
        errors.append(f"{case.name}: step2 TTFT/TPOT/duration/TPS metrics are invalid")


def baseline_validate_resident(
        case: pathlib.Path,
        execution: dict[str, Any] | None,
        errors: list[str]) -> None:
    if execution is None:
        return
    try:
        value = read_json(case / "resident_after_step1.json")
    except Error as exc:
        errors.append(str(exc))
        return
    required = {
        "raw_path", "raw_sha256", "observed_monotonic_ns", "stderr_end", "slots_http_status",
        "claimant", "physical_probe", "resident_scope",
    }
    if not isinstance(value, dict) or set(value) != required:
        errors.append("FLEXKV_RESIDENT: resident evidence schema mismatch")
        return
    path = safe_raw_path(case, value.get("raw_path"), "FLEXKV_RESIDENT resident", errors)
    if path is None:
        return
    if value.get("raw_sha256") != digest(path) or value.get("slots_http_status") != 200 or not is_positive_int(value.get("observed_monotonic_ns")) or not is_nonnegative_int(value.get("stderr_end")):
        errors.append("FLEXKV_RESIDENT: resident evidence binding is invalid")
    try:
        slots = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"FLEXKV_RESIDENT: invalid resident raw slots JSON: {exc}")
        return
    if not isinstance(slots, list) or len(slots) != 1 or not isinstance(slots[0], dict):
        errors.append("FLEXKV_RESIDENT: resident raw slots shape is invalid")
        return
    slot = slots[0]
    claimant = validate_raw_claimant(slot, "FLEXKV_RESIDENT resident raw", errors)
    if claimant is not None and claimant != value.get("claimant"):
        errors.append("FLEXKV_RESIDENT: resident claimant differs from raw slots")
    if claimant is not None and (
            claimant["active"] or claimant["exhausted"] or not claimant["valid"] or
            claimant["target_blocks"] <= 0 or claimant["eligible_resident_blocks"] != claimant["target_blocks"] or
            claimant["swapped_blocks"] != 0 or claimant["shared_blocks"] != 0 or claimant["blocked_blocks"] != 0):
        errors.append("FLEXKV_RESIDENT: P blocks are not logically resident")
    raw_resident = slot.get("kv_resident")
    resident_fields = {
        "status", "source", "object_id", "generation", "page_size", "total_bytes",
        "resident_bytes", "total_pages", "resident_pages",
    }
    if not isinstance(raw_resident, dict) or set(raw_resident) != resident_fields or raw_resident.get("status") != "available" or raw_resident.get("source") != "paged_sample_mincore":
        errors.append("FLEXKV_RESIDENT: raw physical resident sample is invalid")
        return
    numeric = resident_fields - {"status", "source"}
    if any(not is_nonnegative_int(raw_resident.get(key)) for key in numeric) or (
            raw_resident["object_id"] == 0 or raw_resident["generation"] == 0 or raw_resident["page_size"] == 0 or
            raw_resident["total_bytes"] == 0 or raw_resident["total_pages"] == 0 or
            raw_resident["resident_bytes"] <= 0 or raw_resident["resident_pages"] <= 0 or
            raw_resident["resident_bytes"] > raw_resident["total_bytes"] or raw_resident["resident_pages"] > raw_resident["total_pages"] or
            raw_resident["total_bytes"] != raw_resident["total_pages"] * raw_resident["page_size"] or
            raw_resident["resident_bytes"] != raw_resident["resident_pages"] * raw_resident["page_size"]):
        errors.append("FLEXKV_RESIDENT: physical mincore resident accounting is invalid")
    probe = value.get("physical_probe")
    expected_probe_keys = {
        "source", "server_pid", "slots_raw_sha256", "slots_shape", "claimant_keys",
        "logical_claimant_fields_present", "physical_resident_sample_available", "resident", "reason",
    }
    identity = execution.get("server_identity")
    if (not isinstance(probe, dict) or set(probe) != expected_probe_keys or
            probe.get("source") != "GET /slots.kv_resident" or probe.get("server_pid") != (identity or {}).get("pid") or
            probe.get("slots_raw_sha256") != value.get("raw_sha256") or probe.get("slots_shape") != 1 or
            probe.get("physical_resident_sample_available") is not True or probe.get("resident") != raw_resident or
            probe.get("logical_claimant_fields_present") is not True or
            value.get("resident_scope") != "whole_kv_mincore_sample_not_per_prefix_proof"):
        errors.append("FLEXKV_RESIDENT: physical resident probe is not bound to raw server evidence")


def raw_marker_records(
        raw: bytes,
        token: str,
        start: int,
        end: int,
        required: set[str],
        label: str,
        errors: list[str],
        allow_extra: bool = False) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        line_end = offset + len(line)
        if offset >= start and line_end <= end and token.encode("utf-8") in line:
            try:
                words = line.decode("utf-8", errors="replace").split()
                if words.count(token) != 1:
                    raise Error(f"duplicate or malformed {token} token")
                fields: dict[str, str] = {}
                for word in words[words.index(token) + 1:]:
                    if word.count("=") != 1:
                        raise Error(f"malformed {token} field {word!r}")
                    key, value = word.split("=", 1)
                    if not key or not value or key in fields:
                        raise Error(f"duplicate or malformed {token} key {key!r}")
                    fields[key] = value
                if not required <= set(fields) or (not allow_extra and set(fields) != required):
                    raise Error(f"{token} schema mismatch")
                result.append(fields)
            except Error as exc:
                errors.append(f"{label}: {exc}")
        offset = line_end
    return result


def baseline_validate_k1_timing(
        case: pathlib.Path,
        raw: bytes,
        scope: tuple[int | None, int | None],
        rows: dict[str, dict[str, Any]],
        errors: list[str]) -> None:
    try:
        timing = read_json(case / "timing.json")
        saved_scope = read_json(case / "resume_scope.json")
    except Error as exc:
        errors.append(str(exc))
        return
    required = {
        "schema_version", "scope", "stage", "prefetch_call", "block_phases", "io_stats_record_count",
        "io_stats", "response_timings", "derived",
    }
    if not isinstance(timing, dict) or set(timing) != required or timing.get("schema_version") != 1 or timing.get("scope") != saved_scope:
        errors.append("FLEXKV_K1_SYNC: timing schema/scope binding is invalid")
        return
    start, end = scope
    if start is None or end is None:
        return
    stage_fields = {
        "decision_id", "seq_id", "transaction_id", "restored_blocks", "restored_bytes",
        "queue_us", "gate_us", "graph_us", "total_us",
    }
    call_fields = {"call", "seq_id", "requested_blocks", "restored_blocks", "phase_events"}
    block_fields = {"call", "block_index", "physical_block", "validate_us", "read_us", "unpack_us", "commit_us", "phase_sum_us"}
    stage_records = raw_marker_records(raw, "kv_resume_stage_timing", start, end, stage_fields, "FLEXKV_K1_SYNC timing", errors)
    call_records = raw_marker_records(raw, "KV_PAGED_PREFETCH_PHASE_CALL", start, end, call_fields, "FLEXKV_K1_SYNC timing", errors)
    block_records = raw_marker_records(raw, "KV_PAGED_PREFETCH_BLOCK_PHASE", start, end, block_fields, "FLEXKV_K1_SYNC timing", errors)
    if len(stage_records) != 1 or len(call_records) != 1:
        errors.append("FLEXKV_K1_SYNC: raw K1 timing markers are missing or duplicated")
        return
    def integer_fields(value: Any, keys: set[str], label: str) -> dict[str, int] | None:
        if not isinstance(value, dict) or set(value) != keys:
            errors.append(f"{label}: field schema mismatch")
            return None
        parsed: dict[str, int] = {}
        for key in keys:
            candidate = value.get(key)
            if not isinstance(candidate, int) or isinstance(candidate, bool) or candidate < 0:
                errors.append(f"{label}: {key} is invalid")
                return None
            parsed[key] = candidate
        return parsed
    stage = integer_fields(timing.get("stage"), stage_fields, "FLEXKV_K1_SYNC timing stage")
    call = integer_fields(timing.get("prefetch_call"), call_fields, "FLEXKV_K1_SYNC timing prefetch call")
    block_phases = timing.get("block_phases")
    if stage is None or call is None or not isinstance(block_phases, list):
        return
    raw_stage = {key: int(value) for key, value in stage_records[0].items()} if all(value.isdigit() for value in stage_records[0].values()) else None
    raw_call = {key: int(value) for key, value in call_records[0].items()} if all(value.isdigit() for value in call_records[0].values()) else None
    if raw_stage != stage or raw_call != call or stage["restored_blocks"] <= 0 or stage["restored_bytes"] <= 0:
        errors.append("FLEXKV_K1_SYNC: saved K1 timing does not bind raw resume markers")
    parsed_blocks: list[dict[str, int]] = []
    for index, value in enumerate(block_phases):
        block = integer_fields(value, block_fields, f"FLEXKV_K1_SYNC timing block[{index}]")
        if block is not None:
            parsed_blocks.append(block)
    parsed_raw_blocks = [
        {key: int(value) for key, value in record.items()}
        for record in block_records if all(value.isdigit() for value in record.values())
    ]
    if parsed_blocks != parsed_raw_blocks or len(parsed_blocks) != call["phase_events"] or call["restored_blocks"] != stage["restored_blocks"]:
        errors.append("FLEXKV_K1_SYNC: block phase timing does not close the raw restore call")
    for index, block in enumerate(parsed_blocks):
        if (block["call"] != call["call"] or block["block_index"] != index or
                block["phase_sum_us"] != block["validate_us"] + block["read_us"] + block["unpack_us"] + block["commit_us"]):
            errors.append("FLEXKV_K1_SYNC: block phase timing accounting is invalid")
    io_fields = {
        "block_swap_out_calls", "block_swap_in_calls", "backing_read_syscalls", "backing_write_syscalls",
        "bytes_read", "bytes_written", "avg_block_swap_in_latency_us", "max_block_swap_in_latency_us",
        "block_in_validate_us", "block_in_read_us", "block_in_unpack_us", "block_in_commit_us",
    }
    io_records = raw_marker_records(
        raw, "KV_PAGED_IO_STATS", 0, len(raw), io_fields,
        "FLEXKV_K1_SYNC timing", errors, allow_extra=True)
    io_stats = integer_fields(timing.get("io_stats"), io_fields, "FLEXKV_K1_SYNC IO stats")
    if timing.get("io_stats_record_count") != len(io_records) or io_stats is None or not any(
            all(record.get(key, "").isdigit() and int(record[key]) == io_stats[key] for key in io_fields)
            for record in io_records):
        errors.append("FLEXKV_K1_SYNC: I/O stats are not bound to raw stderr")
    step2 = rows.get("step2")
    timing_keys = {"prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "predicted_per_token_ms"}
    if (not isinstance(step2, dict) or not isinstance(step2.get("response_timings"), dict) or
            timing.get("response_timings") != {key: step2["response_timings"].get(key) for key in sorted(timing_keys)}):
        errors.append("FLEXKV_K1_SYNC: timing response metrics are not bound to step2")
    derived_keys = {
        "server_ttft_ms", "server_prompt_ms", "tpot_ms", "queue_us", "read_us", "restore_us",
        "commit_us", "graph_us", "gate_us", "gate_overhead_us", "residual_us", "io_bytes", "io_syscalls", "io_service_us",
    }
    derived = timing.get("derived")
    if not isinstance(derived, dict) or set(derived) != derived_keys:
        errors.append("FLEXKV_K1_SYNC: derived timing schema mismatch")
        return
    if not all(is_finite_number(value) for value in derived.values()):
        errors.append("FLEXKV_K1_SYNC: derived timing contains invalid values")
        return
    read_us = sum(block["read_us"] for block in parsed_blocks)
    restore_us = sum(block["validate_us"] + block["unpack_us"] for block in parsed_blocks)
    commit_us = sum(block["commit_us"] for block in parsed_blocks)
    if (derived["read_us"] != read_us or derived["restore_us"] != restore_us or derived["commit_us"] != commit_us or
            derived["gate_us"] != stage["gate_us"] or derived["io_bytes"] != stage["restored_bytes"] or
            derived["io_service_us"] != read_us or derived["io_syscalls"] <= 0 or derived["read_us"] <= 0 or
            derived["restore_us"] <= 0 or derived["gate_us"] <= 0):
        errors.append("FLEXKV_K1_SYNC: derived read/restore/gate/I/O timing is inconsistent")


def baseline_validate_stream_step2(row: Any, name: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        return
    if (row.get("streaming") is not True or row.get("stream_terminal_received") is not True or
            row.get("stream_error") is not None or not is_positive_int(row.get("first_content_monotonic_ns")) or
            not is_positive_int(row.get("stream_event_count")) or
            not isinstance(row.get("response_timings"), dict)):
        errors.append(f"{name}: step2 stream evidence is incomplete")
    content_type = row.get("content_type")
    if not isinstance(content_type, str) or "text/event-stream" not in content_type.lower():
        errors.append(f"{name}: step2 is not an SSE event-stream response")


def baseline_validate_cleanup(
        case: pathlib.Path,
        name: str,
        execution: dict[str, Any] | None,
        result: dict[str, Any],
        errors: list[str]) -> None:
    try:
        cleanup = read_json(case / "cleanup.json")
    except Error as exc:
        errors.append(str(exc))
        return
    if not isinstance(cleanup, dict) or set(cleanup) != {"server", "memory_sampler", "backing"}:
        errors.append(f"{name}: cleanup schema mismatch")
        return
    result_keys = {"server": "process", "memory_sampler": "memory_sampler", "backing": "backing"}
    for cleanup_key, result_key in result_keys.items():
        if result.get(result_key) != cleanup.get(cleanup_key):
            errors.append(f"{name}: result {result_key} does not bind cleanup {cleanup_key} evidence")
    server = cleanup.get("server")
    identity = execution.get("server_identity") if isinstance(execution, dict) else None
    server_required = {"pid", "pgid", "exit_code", "term_timed_out", "kill_timed_out", "residual_process"}
    if not isinstance(server, dict) or set(server) != server_required:
        errors.append(f"{name}: server cleanup schema mismatch")
    elif (not isinstance(identity, dict) or server.get("pid") != identity.get("pid") or
            server.get("term_timed_out") is not False or server.get("kill_timed_out") is not False or
            server.get("residual_process") is not False):
        errors.append(f"{name}: server cleanup is incomplete or unbound")
    backing = cleanup.get("backing")
    if baseline_case_flags(name)["backing"]:
        expected_path = str((case / "backing").resolve())
        expected = {
            "path": expected_path,
            "environment_value": "backing",
            "created": True,
            "cleanup_attempted": True,
            "exists_after_cleanup": False,
            "cleanup_error": None,
        }
        if backing != expected or (case / "backing").exists():
            errors.append(f"{name}: backing cleanup is incomplete")
    elif backing != {"status": NOT_APPLICABLE}:
        errors.append(f"{name}: non-backing cleanup must use NOT_APPLICABLE")


def load_baseline_case(
        case: pathlib.Path,
        metadata: dict[str, Any],
        manifest: dict[str, Any],
        errors: list[str]) -> dict[str, Any]:
    name = metadata["case"]
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
        if not is_nonempty_string(reason):
            reason = result.get("cleanup_error")
        errors.append(f"{name}: {reason if is_nonempty_string(reason) else 'workload did not complete'}")
        return {"upstream_failure": True, "result": result}
    required_files = {
        "execution.json", "environment.json", "requests.jsonl", "server.stdout", "server.stderr",
        "memory_sampler.stderr", "memory_samples.tsv", "memory_phases.json", "resume_scope.json",
        "step2_metrics.json", "cleanup.json", "result.json",
    }
    required_files |= {path for path in baseline_evidence_paths(name).values() if path != NOT_APPLICABLE}
    for filename in sorted(required_files):
        if not (case / filename).is_file():
            errors.append(f"{name}: missing {filename}")
    expected_result = {
        "status": "complete",
        "case": metadata["case"],
        "round": metadata["round"],
        "run_order": metadata["run_order"],
        "request_loop_started": True,
        "ctx_size": manifest.get("parameters", {}).get("ctx_size"),
        "target_prefix_tokens": manifest.get("parameters", {}).get("target_prefix_tokens"),
        "evidence": baseline_evidence_paths(name),
    }
    for key, value in expected_result.items():
        if result.get(key) != value:
            errors.append(f"{name}: result {key} differs from the fixed case contract")
    execution, _environment, argv = baseline_validate_execution(case, metadata, manifest, errors)
    raw = (case / "server.stderr").read_bytes() if (case / "server.stderr").is_file() else b""
    rows = rows_by_label(read_rows(case / "requests.jsonl", errors), name, errors)
    validate_request_pair(rows, name, errors, step2_stream=True)
    baseline_validate_stream_step2(rows.get("step2"), name, errors)
    workload = validate_workload(case, name, result, rows, manifest, errors)
    baseline_validate_step2_metrics(case, rows, errors)
    baseline_validate_memory(case, name, execution, result, errors)
    baseline_validate_cleanup(case, name, execution, result, errors)
    scope = validate_resume_scope(case, raw, rows.get("step2"), name, errors)

    capability = baseline_validate_capability(case, raw, name, errors)
    if capability is not None and result.get("capability") != capability:
        errors.append(f"{name}: result capability differs from raw marker")
    markers = token_lines(raw, MARKER, parse_marker, f"{case}/server.stderr", errors)
    if name in {"CURRENT_E0", "FLEXKV_RESIDENT"} and markers:
        errors.append(f"{name}: Governor action marker is present while actions are disabled")
    if name == "CURRENT_E0":
        if result.get("offload") != NOT_APPLICABLE or result.get("timing") != NOT_APPLICABLE:
            errors.append("CURRENT_E0: offload/restore evidence must be NOT_APPLICABLE")
    elif name == "FLEXKV_RESIDENT":
        if result.get("offload") != NOT_APPLICABLE or result.get("timing") != NOT_APPLICABLE:
            errors.append("FLEXKV_RESIDENT: offload/restore evidence must be NOT_APPLICABLE")
        baseline_validate_resident(case, execution, errors)
    else:
        pre, _pre_stderr_end, _pre_observed = validate_snapshot(
            case, "pre_offload_claimant.json", "FLEXKV_K1_SYNC", errors)
        step1 = rows.get("step1", {})
        prompt = step1.get("request", {}).get("prompt") if isinstance(step1, dict) else None
        expected_blocks = len(prompt) // PAGED_BLOCK_SIZE if isinstance(prompt, list) else None
        if pre is not None and expected_blocks is not None and (
                pre["target_blocks"] != expected_blocks or pre["eligible_resident_blocks"] != expected_blocks or
                pre["swapped_blocks"] != 0 or pre["active"] or pre["exhausted"] or not pre["valid"]):
            errors.append("FLEXKV_K1_SYNC: pre-offload claimant is not a complete resident prefix")
        post, post_stderr_end, post_observed = validate_snapshot(
            case, "post_claimant.json", "FLEXKV_K1_SYNC", errors)
        epoch, last_offload_end = baseline_validate_k1_transactions(
            case, raw, execution.get("server_identity") if isinstance(execution, dict) else None,
            post, expected_blocks, scope[0], "FLEXKV_K1_SYNC", errors)
        validate_resume(case, raw, scope[0], scope[1], last_offload_end, epoch, "FLEXKV_K1_SYNC", errors)
        if last_offload_end is not None and post_stderr_end is not None and post_stderr_end != scope[0]:
            errors.append("FLEXKV_K1_SYNC: post-offload claimant does not bind the step2 boundary")
        if isinstance(rows.get("step2"), dict) and post_observed is not None and post_observed >= rows["step2"].get("started_monotonic_ns", 0):
            errors.append("FLEXKV_K1_SYNC: post-offload claimant is not before step2")
        try:
            offload = read_json(case / "offload.json")
        except Error as exc:
            errors.append(str(exc))
        else:
            if isinstance(offload, dict) and result.get("offload") != offload.get("cumulative"):
                errors.append("FLEXKV_K1_SYNC: result offload summary differs from raw transactions")
        baseline_validate_k1_timing(case, raw, scope, rows, errors)
    return {
        "metadata": metadata,
        "case": case,
        "result": result,
        "execution": execution,
        "argv": argv,
        "rows": rows,
        "workload": workload,
    }


def baseline_validate_comparisons(
        runs: list[dict[str, Any]],
        errors: list[str]) -> list[dict[str, Any]]:
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for item in runs:
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            by_round.setdefault(metadata["round"], {})[metadata["case"]] = item
    records: list[dict[str, Any]] = []
    available_rounds = sorted(by_round)
    for round_no in available_rounds:
        cases = by_round.get(round_no, {})
        if set(cases) != set(BASELINE_CASES):
            errors.append(f"round {round_no}: baseline comparison cases are incomplete")
            continue
        for edge, relation in BASELINE_COMPARISON_EDGES.items():
            left = cases[relation["left"]]
            right = cases[relation["right"]]
            left_rows, right_rows = left.get("rows", {}), right.get("rows", {})
            matched_requests = True
            for label in ("step1", "step2"):
                left_row, right_row = left_rows.get(label), right_rows.get(label)
                if (not isinstance(left_row, dict) or not isinstance(right_row, dict) or
                        left_row.get("request") != right_row.get("request") or
                        left_row.get("request_sha256") != right_row.get("request_sha256")):
                    matched_requests = False
            left_step2, right_step2 = left_rows.get("step2"), right_rows.get("step2")
            matched_output = (
                isinstance(left_step2, dict) and isinstance(right_step2, dict) and
                left_step2.get("response_text") == right_step2.get("response_text") and
                left_step2.get("response_sha256") == right_step2.get("response_sha256"))
            if not matched_requests:
                errors.append(f"round {round_no} {edge}: P/P+Q request workload differs across the comparison edge")
            if not matched_output:
                errors.append(f"round {round_no} {edge}: step2 output is not Exact across the comparison edge")
            records.append({
                "round": round_no,
                "edge": edge,
                "left": relation["left"],
                "right": relation["right"],
                "requests_exact": matched_requests,
                "output_exact": matched_output,
            })
    return records


def write_baseline_summary(
        root: pathlib.Path,
        runs: list[dict[str, Any]],
        comparisons: list[dict[str, Any]],
        diagnostic_dirty: bool) -> None:
    columns = [
        "round", "run_order", "case", "ttft_ms", "tpot_ms", "total_duration_ms", "tps",
        "vmrss_server_ready_kb", "vmrss_after_step1_kb", "vmrss_after_offload_kb",
        "vmrss_after_step2_kb", "memory_current_after_step2_bytes", "evidence",
    ]
    rows: list[dict[str, Any]] = []
    for item in sorted(runs, key=lambda value: (value["metadata"]["round"], value["metadata"]["run_order"])):
        metadata, case, result = item["metadata"], item["case"], item["result"]
        metrics = read_json(case / "step2_metrics.json")
        memory = read_json(case / "memory_phases.json")
        phases = {phase["phase"]: phase for phase in memory["phases"]}
        rows.append({
            "round": metadata["round"],
            "run_order": metadata["run_order"],
            "case": metadata["case"],
            "ttft_ms": metrics["ttft_ms"],
            "tpot_ms": metrics["tpot_ms"],
            "total_duration_ms": metrics["total_duration_ms"],
            "tps": metrics["tps"],
            "vmrss_server_ready_kb": phases["server_ready"]["vmrss_kb"],
            "vmrss_after_step1_kb": phases["after_step1"]["vmrss_kb"],
            "vmrss_after_offload_kb": phases.get("after_offload", {}).get("vmrss_kb", NOT_APPLICABLE),
            "vmrss_after_step2_kb": phases["after_step2"]["vmrss_kb"],
            "memory_current_after_step2_bytes": phases["after_step2"]["cgroup_memory_current_bytes"],
            "evidence": json.dumps(result["evidence"], sort_keys=True, separators=(",", ":")),
        })
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (root / "comparison.json").write_text(
        json.dumps({"comparison_edges": BASELINE_COMPARISON_EDGES, "rounds": comparisons}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    capture_message = (
        "WARNING: this is a dirty diagnostic capture. It is not archival evidence and cannot be reported as formal PASS."
        if diagnostic_dirty else
        "All fixed comparison edges passed request and step2 output Exact checks in every round.")
    lines = [
        "# B0-B2 server baseline summary", "", capture_message, "",
        "| Edge | Left | Right | Meaning |", "|---|---|---|---|",
    ]
    lines.extend(
        f"| {edge} | {item['left']} | {item['right']} | {item['purpose']} |"
        for edge, item in BASELINE_COMPARISON_EDGES.items())
    lines.extend([
        "", "The table in `summary.tsv` records the streaming second-request TTFT, server TPOT, request-wall duration, TPS, and phase RSS/cgroup observations. These measurements are controlled diagnostic evidence, not a performance conclusion.",
        "", "`FLEXKV_RESIDENT` records a whole-KV mincore sample plus logical claimant state; it does not claim per-prefix physical residency. `FLEXKV_K1_SYNC` is the synchronous restore gate path and does not enable E5 delayed active prefetch.",
    ])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main_parse_baseline(root: pathlib.Path, manifest: dict[str, Any]) -> tuple[str, list[str]]:
    errors: list[str] = []
    baseline_validate_manifest(manifest, errors)
    status = manifest.get("runner_status")
    smoke = manifest.get("execution", {}).get("smoke") is True
    pairs = baseline_validate_matrix(root, status == "DRY_RUN", smoke, errors)
    if status == "DRY_RUN":
        if manifest.get("run_results") != []:
            errors.append("baseline dry-run must not contain run results")
        return ("FAIL", errors) if errors else ("DRY_RUN", ["B0-B1 plan materialized; no model process was started"])
    if status != "run_complete":
        reason = manifest.get("runner_error")
        errors.append(reason if is_nonempty_string(reason) else f"baseline runner status is {status!r}")
    if errors:
        return "FAIL", errors
    runs: list[dict[str, Any]] = []
    for metadata, case in pairs:
        item = load_baseline_case(case, metadata, manifest, errors)
        if item.get("upstream_failure"):
            return "FAIL", errors
        runs.append(item)
    comparisons = baseline_validate_comparisons(runs, errors)
    if errors:
        return "FAIL", errors
    try:
        write_baseline_summary(root, runs, comparisons, bool(manifest.get("dirty_status")))
    except (Error, OSError, TypeError, KeyError, ValueError) as exc:
        return "FAIL", [f"could not write baseline summary: {type(exc).__name__}: {exc}"]
    if manifest.get("dirty_status") and manifest.get("capture_mode") != "diagnostic_smoke":
        return "DIAGNOSTIC", ["dirty capture completed; diagnostic evidence cannot be verified as formal PASS"]
    if manifest.get("capture_mode") == "diagnostic_smoke":
        return "DIAGNOSTIC", ["diagnostic smoke completed; not a formal PASS"]
    return "PASS", []


def main_parse(root: pathlib.Path) -> tuple[str, list[str]]:
    try:
        manifest = read_json(root / "manifest.json")
    except Error as exc:
        return "FAIL", [str(exc)]
    if not isinstance(manifest, dict):
        return "FAIL", ["baseline manifest is not an object"]
    return main_parse_baseline(root, manifest)


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
    raise SystemExit(0 if status in {"PASS", "DRY_RUN"} else 4 if status == "DIAGNOSTIC" else 1)


if __name__ == "__main__":
    main()
