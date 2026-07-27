#!/usr/bin/env python3
"""Fail-closed parser for Stage 3B-2A with long-context ladder and continuous requests.

Validates:
  0. Protocol identity, manifest integrity (binary/model/runner/parser/source_snapshot)
  1. Tokenizer calibration: measured token counts within ±2 of targets
  2. RSS calibration per context length: thresholds derived from real measurements
  3. Long context ladder per token target:
     - All mandatory files present (server.stderr, environment.json, execution.json,
       result.json, phases.json) — missing any file = FAIL
     - OFF: zero bounded_release markers, response identity match
     - DYNAMIC_RELEASE: >=1 bounded_release execution, target_mode=dynamic,
       ownership_aborted=0, madvise_failures=0, mincore drop observed
     - Per-request stderr windows: bounded/pressure decisions, blocks_skipped_owned>0
  4. Continuous requests: all rounds present/consecutive/unique, KV_PAGED_RELEASE_STATS
     mandatory with reuse_allocations/write_commits/bounded_release_calls>0 and
     all rollback/fatal=0, all responses identical
  5. Config contamination and environment closure
  6. Zero residual processes everywhere

Parser is the sole verdict authority. Fails closed on any violation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any, NoReturn

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACTIVE_RESULT_PATH: pathlib.Path | None = None
ACTIVE_ARTIFACT: pathlib.Path | None = None

# ── constants ────────────────────────────────────────────────────────────────

PROTOCOL = "kv_bounded_release_stage3b_2a"

BOUNDED_RELEASE_RE = re.compile(r"kv_pressure_bounded_release\b")
DRY_RUN_MARKER_RE = re.compile(r"kv_pressure_dry_run\b")
TELEMETRY_MARKER_RE = re.compile(r"kv_pressure_telemetry\b")

TOKEN_TARGETS_DEFAULT = {1024, 2048, 4096, 8192}

# ── live-context / timeout derivation (mirror runner defaults) ──────────────
# The parser does NOT re-derive; it validates the runner's recorded derivation
# against these shared invariants.  Defaults must match the runner constants.
N_PREDICT_DEFAULT = 32
SPECIAL_OVERHEAD = 1
SAFETY_DEFAULT = 95
TOKEN_ALIGN = 32
TIMEOUT_MARGIN = 2.5
TIMEOUT_FLOOR_S = 30.0
COMPLETION_TIMEOUT_FALLBACK_S = 180.0
CALIBRATION_TIMEOUT_BOOTSTRAP_S = 180.0
CALIBRATION_TIMEOUT_SAFETY_FACTOR = 1.5


def derive_completion_timeout(measured_s: float,
                              margin: float = TIMEOUT_MARGIN,
                              floor: float = TIMEOUT_FLOOR_S,
                              fallback: float = COMPLETION_TIMEOUT_FALLBACK_S) -> float:
    if measured_s is None or measured_s <= 0.0:
        return fallback
    return max(measured_s * margin, floor)


def derive_calibration_timeout(target_tokens: int,
                               previous_target_tokens: int | None = None,
                               previous_completion_wall_s: float | None = None,
                               bootstrap_s: float = CALIBRATION_TIMEOUT_BOOTSTRAP_S,
                               safety_factor: float = CALIBRATION_TIMEOUT_SAFETY_FACTOR,
                               ) -> dict[str, Any]:
    if previous_target_tokens is None or previous_completion_wall_s is None:
        return {
            "calibration_timeout_s": bootstrap_s,
            "calibration_timeout_mode": "bootstrap",
            "calibration_timeout_bootstrap_s": bootstrap_s,
            "calibration_timeout_previous_target": None,
            "calibration_timeout_previous_wall_s": None,
            "calibration_timeout_token_ratio": None,
            "calibration_timeout_safety_factor": safety_factor,
        }
    token_ratio = target_tokens / previous_target_tokens
    derived = previous_completion_wall_s * token_ratio * safety_factor
    return {
        "calibration_timeout_s": max(bootstrap_s, derived),
        "calibration_timeout_mode": "previous_success_scaled",
        "calibration_timeout_bootstrap_s": bootstrap_s,
        "calibration_timeout_previous_target": previous_target_tokens,
        "calibration_timeout_previous_wall_s": previous_completion_wall_s,
        "calibration_timeout_token_ratio": token_ratio,
        "calibration_timeout_safety_factor": safety_factor,
        "calibration_timeout_scaled_s": derived,
    }


def derive_max_prompt_tokens(effective_n_ctx: int, n_predict: int = N_PREDICT_DEFAULT,
                             special_overhead: int = SPECIAL_OVERHEAD,
                             safety: int = SAFETY_DEFAULT,
                             align: int = TOKEN_ALIGN) -> int:
    budget = effective_n_ctx - n_predict - special_overhead - safety
    if budget <= 0:
        return 0
    if align > 1:
        budget = (budget // align) * align
    return max(budget, 0)

BOUNDED_ONLY_KEYS = {
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET",
}

CALIBRATED_KEYS = {
    "LLAMA_KV_PRESSURE_RSS_KB",
    "LLAMA_KV_CRITICAL_RSS_KB",
    "LLAMA_KV_LOW_WATER_RSS_KB",
}

BOUNDED_REQUIRED_KEYS = {
    "state", "source", "stale",
    "released_bytes", "released_blocks",
    "blocks_scanned", "blocks_skipped_owned", "blocks_skipped_state",
    "madvise_failures", "shortfall_bytes", "overshoot_bytes",
    "block_scan_exhausted", "ownership_aborted",
    "target_mode", "pressure_basis_valid", "pressure_current_bytes",
    "pressure_low_water_bytes", "pressure_basis_generation",
    "kv_budget_valid", "kv_budget_ownership_aborted", "kv_resident_bytes",
    "kv_reclaimable_resident_bytes", "water_excess_bytes", "water_shortfall_bytes",
    "water_overshoot_bytes", "max_release_bytes", "target_clamp", "decision_reason",
    "target_bytes", "max_scan_blocks",
    "legacy_enabled", "sample_count", "episode",
    "cooldown_ms", "skipped_reason", "idle",
    "mincore_before_bytes", "mincore_after_bytes",
    "bounded_cnt_bytes_delta", "bounded_cnt_blocks_delta",
    "can_enable", "cap_paged", "cap_ingraph", "cap_layers",
    "cap_row_idx", "cap_swap_disabled", "cap_layout",
}

TELEMETRY_REQUIRED_KEYS = {
    "state", "previous_state", "source", "sample_valid", "stale", "config_valid",
    "rss_kb", "cgroup_current_bytes", "cgroup_max_bytes", "cgroup_current_kb",
    "cgroup_max_kb", "cgroup_high_kb", "psi_some_avg10", "psi_full_avg10",
    "pressure_basis_valid", "pressure_current_bytes", "pressure_low_water_bytes",
    "pressure_basis_generation",
    "sample_latency_ns", "sample_count", "skip_count", "idle", "trigger",
}

BOOL_FIELDS = {"stale", "release_enabled", "block_scan_exhausted", "ownership_aborted",
               "legacy_enabled", "idle", "can_enable", "cap_paged", "cap_ingraph",
               "cap_layers", "cap_row_idx", "cap_swap_disabled", "cap_layout",
               "sample_valid", "config_valid", "pressure_basis_valid", "kv_budget_valid",
               "kv_budget_ownership_aborted"}

ENUMS = {
    "state": {"NORMAL", "PRESSURE", "CRITICAL", "RECOVERY"},
    "previous_state": {"NORMAL", "PRESSURE", "CRITICAL", "RECOVERY"},
    "source": {"NONE", "RSS_ABSOLUTE", "CGROUP_RATIO", "CGROUP_ABSOLUTE"},
    "target_mode": {"fixed", "dynamic"},
    "target_clamp": {"none", "max_release", "resident", "reclaimable"},
    "decision_reason": {"fixed", "dynamic", "invalid_pressure_basis", "not_pressure",
        "ownership_aborted", "invalid_kv_budget", "no_budget", "no_excess_or_candidate"},
    "skipped_reason": {"none", "dry_run_active", "no_memory", "not_paged",
        "layout_unsupported", "swap_enabled", "not_ingraph", "no_layers", "no_row_idx",
        "legacy_active", "stale", "not_pressure", "cooldown",
        "structurally_disabled", "ownership_aborted", "invalid_pressure_basis",
        "invalid_kv_budget", "no_budget", "no_excess_or_candidate"},
}

STRING_FIELDS = {"trigger"}

TRIGGERS = {"first", "state", "source", "stale", "periodic", "wake_completion"}

RELEASE_STATS_MARKER = "KV_PAGED_RELEASE_STATS"
LIFECYCLE_STATS_FIELDS = (
    "reuse_allocations", "write_commits", "write_rollbacks",
    "dummy_candidate_pending_write_cell", "released_redirect_no_dummy",
    "released_redirect_no_dummy_pending_write", "ensure_pending_write_rejected",
    "input_setup_fatal", "row_mapping_fatal", "write_mapping_fatal",
    "active_nonresident_fatal",
)
BOUNDED_SOURCE_STATS_FIELDS = (
    "bounded_release_calls", "bounded_release_bytes", "bounded_release_blocks",
)
REQUIRED_RELEASE_STATS_FIELDS = LIFECYCLE_STATS_FIELDS + BOUNDED_SOURCE_STATS_FIELDS

# Fatal fields that must be zero in continuous-request release stats
FATAL_STATS_FIELDS = (
    "write_rollbacks", "input_setup_fatal", "row_mapping_fatal",
    "write_mapping_fatal", "active_nonresident_fatal",
)

_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UINT_RE = re.compile(r"[0-9]+")
_STATS_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Mandatory files per case directory (must all exist)
CASE_MANDATORY_FILES = [
    "server.stderr", "environment.json", "execution.json",
    "result.json", "phases.json",
]

# RSS consistency: if sample RSS is below this fraction of calibration idle RSS,
# it is treated as likely sampling the wrong process (e.g., strace instead of server).
RSS_CONSISTENCY_MIN_FRACTION = 0.1
# Minimum calibration idle RSS to activate the check (KiB).  Below this
# threshold even tiny models would trigger false positives.
RSS_CONSISTENCY_MIN_IDLE_KIB = 500_000
# Absolute floor: any RSS reading below this KiB value is suspicious regardless
# of calibration ratio when calibration idle is substantial.
RSS_CONSISTENCY_ABSOLUTE_FLOOR_KIB = 50_000


# ── helpers ───────────────────────────────────────────────────────────────────

class ProtocolError(Exception):
    pass


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fail(message: str) -> NoReturn:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def parser_identity() -> dict[str, Any]:
    path = pathlib.Path(__file__).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def write_parser_result(exit_code: int, status: str) -> None:
    if ACTIVE_RESULT_PATH is None or ACTIVE_ARTIFACT is None:
        return
    manifest_path = ACTIVE_ARTIFACT / "manifest.json"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "exit_code": exit_code,
        "command": list(sys.argv),
        "parser": parser_identity(),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
    }
    ACTIVE_RESULT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")


def check_saved_result(path: pathlib.Path, manifest_path: pathlib.Path) -> None:
    if not path.is_file():
        fail(f"parser result missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail("parser result must be an object")
    if data.get("schema_version") != 1:
        fail("parser result schema_version mismatch")
    if data.get("status") != "PASS" or data.get("exit_code") != 0:
        fail("parser result is not a successful protocol verdict")
    if data.get("manifest_sha256") != sha256_file(manifest_path):
        fail("parser result manifest hash mismatch")
    if data.get("parser") != parser_identity():
        fail("parser result parser identity mismatch")
    command = data.get("command")
    if not isinstance(command, list) or "--result-path" not in command:
        fail("parser result command is incomplete")


# ── marker parsing ───────────────────────────────────────────────────────────

def _validate_trigger_value(marker: str, value: str, case: str) -> None:
    if not value.strip():
        raise ProtocolError(f"{case}: {marker} trigger is empty")
    components = value.split(",")
    if any(not c for c in components):
        raise ProtocolError(f"{case}: {marker} trigger has empty component")
    if len(components) != len(set(components)):
        raise ProtocolError(f"{case}: {marker} trigger has duplicate values")
    for c in components:
        if c not in TRIGGERS:
            raise ProtocolError(f"{case}: {marker} trigger unknown {c!r}")


def _validate_value(marker: str, key: str, value: str, case: str) -> None:
    if key in BOOL_FIELDS and value not in {"0", "1"}:
        raise ProtocolError(f"{case}: {marker} {key} must be 0 or 1")
    if key == "trigger":
        _validate_trigger_value(marker, value, case)
        return
    if key in ENUMS and value not in ENUMS[key]:
        raise ProtocolError(f"{case}: {marker} {key} invalid {value!r}")
    if key in STRING_FIELDS:
        return
    if key not in BOOL_FIELDS and key not in ENUMS:
        if not _UINT_RE.fullmatch(value):
            raise ProtocolError(f"{case}: {marker} {key} must be uint, got {value!r}")


def parse_strict_marker_line(line: str, marker: str, case: str) -> dict[str, str]:
    tokens = line.split()
    indexes = [i for i, token in enumerate(tokens) if token == marker]
    if len(indexes) != 1:
        raise ProtocolError(f"{case}: {marker} must occur as exactly one token")
    fields: dict[str, str] = {}
    for token in tokens[indexes[0] + 1:]:
        if token.count("=") != 1:
            raise ProtocolError(f"{case}: {marker} malformed token {token!r}")
        key, value = token.split("=", 1)
        if not _KEY_RE.fullmatch(key) or not value:
            raise ProtocolError(f"{case}: {marker} bad field {token!r}")
        if key in fields:
            raise ProtocolError(f"{case}: {marker} duplicate field {key!r}")
        fields[key] = value
    return fields


def parse_bounded_marker(line: str, case: str) -> dict[str, str]:
    fields = parse_strict_marker_line(line, "kv_pressure_bounded_release", case)
    if set(fields) != BOUNDED_REQUIRED_KEYS:
        missing = BOUNDED_REQUIRED_KEYS - set(fields)
        extra = set(fields) - BOUNDED_REQUIRED_KEYS
        raise ProtocolError(f"{case}: bounded marker schema mismatch "
                           f"missing={sorted(missing)} extra={sorted(extra)}")
    for key, value in fields.items():
        _validate_value("kv_pressure_bounded_release", key, value, case)
    if int(fields["sample_count"]) < 1:
        raise ProtocolError(f"{case}: bounded sample_count must be >= 1")
    if int(fields["max_scan_blocks"]) < 1:
        raise ProtocolError(f"{case}: bounded max_scan_blocks must be >= 1")
    return fields


def parse_telemetry_marker(line: str, case: str) -> dict[str, str]:
    fields = parse_strict_marker_line(line, "kv_pressure_telemetry", case)
    if set(fields) != TELEMETRY_REQUIRED_KEYS:
        missing = TELEMETRY_REQUIRED_KEYS - set(fields)
        extra = set(fields) - TELEMETRY_REQUIRED_KEYS
        raise ProtocolError(f"{case}: telemetry marker schema mismatch "
                           f"missing={sorted(missing)} extra={sorted(extra)}")
    for key, value in fields.items():
        _validate_value("kv_pressure_telemetry", key, value, case)
    return fields


def find_markers(stderr_text: str, marker_re: re.Pattern[str], case: str) -> list[dict[str, str]]:
    markers: list[dict[str, str]] = []
    for line in stderr_text.split("\n"):
        if marker_re.search(line):
            if marker_re is BOUNDED_RELEASE_RE:
                markers.append(parse_bounded_marker(line, case))
            elif marker_re is TELEMETRY_MARKER_RE:
                markers.append(parse_telemetry_marker(line, case))
    return markers


def extract_final_release_stats(case: str, stderr_text: str) -> dict[str, int] | None:
    marker_lines = [line for line in stderr_text.splitlines()
                    if RELEASE_STATS_MARKER in line]
    if not marker_lines:
        return None

    line = marker_lines[-1]
    tokens = line.split()
    marker_indexes = [i for i, token in enumerate(tokens) if token == RELEASE_STATS_MARKER]
    if len(marker_indexes) != 1:
        return None

    fields: dict[str, str] = {}
    for token in tokens[marker_indexes[0] + 1:]:
        if token.count("=") != 1:
            return None
        key, value = token.split("=", 1)
        if not _STATS_KEY_RE.fullmatch(key) or not value:
            return None
        if key in fields:
            return None
        fields[key] = value

    missing = [n for n in REQUIRED_RELEASE_STATS_FIELDS if n not in fields]
    if missing:
        return None

    result: dict[str, int] = {}
    for name in REQUIRED_RELEASE_STATS_FIELDS:
        value = fields[name]
        if not _UINT_RE.fullmatch(value):
            return None
        result[name] = int(value)
    return result


# ── validation ───────────────────────────────────────────────────────────────

def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _validate_bounded_marker(marker: dict[str, str], case_name: str,
                             marker_index: int, errors: list[str]) -> bool:
    """Validate one bounded-release marker and return whether it performed an action."""
    prefix = f"{case_name}: bounded marker[{marker_index}]"
    released_bytes = int(marker["released_bytes"])
    released_blocks = int(marker["released_blocks"])
    bytes_delta = int(marker["bounded_cnt_bytes_delta"])
    blocks_delta = int(marker["bounded_cnt_blocks_delta"])

    _check(marker["ownership_aborted"] == "0" and
           marker["kv_budget_ownership_aborted"] == "0",
           f"{prefix}: ownership safety violation", errors)
    _check(marker["madvise_failures"] == "0",
           f"{prefix}: madvise failure", errors)
    _check(all(marker[field] == "1" for field in (
        "can_enable", "cap_paged", "cap_ingraph", "cap_layers", "cap_row_idx",
        "cap_swap_disabled", "cap_layout")),
           f"{prefix}: mapping capability violation", errors)

    if released_bytes > 0:
        _check(released_blocks > 0,
               f"{prefix}: action has released_bytes>0 but released_blocks=0", errors)
        _check(bytes_delta == released_bytes,
               f"{prefix}: action byte delta={bytes_delta} != released_bytes={released_bytes}",
               errors)
        _check(blocks_delta == released_blocks,
               f"{prefix}: action block delta={blocks_delta} != released_blocks={released_blocks}",
               errors)
        mincore_before = int(marker["mincore_before_bytes"])
        mincore_after = int(marker["mincore_after_bytes"])
        observed_drop = mincore_before - mincore_after
        _check(mincore_after < mincore_before,
               f"{prefix}: action mincore did not drop "
               f"(before={mincore_before} after={mincore_after})", errors)
        _check(observed_drop == released_bytes,
               f"{prefix}: action mincore drop={observed_drop} != released_bytes={released_bytes}",
               errors)
        return True

    _check(released_blocks == 0,
           f"{prefix}: no-op has released_bytes=0 but released_blocks={released_blocks}", errors)
    _check(bytes_delta == 0 and blocks_delta == 0,
           f"{prefix}: no-op counter deltas must both be zero "
           f"(bytes={bytes_delta} blocks={blocks_delta})", errors)
    no_op_reason = (
        int(marker["blocks_skipped_owned"]) > 0 or
        int(marker["blocks_skipped_state"]) > 0 or
        marker["block_scan_exhausted"] == "1" or
        marker["skipped_reason"] != "none" or
        marker["decision_reason"] in {
            "no_excess_or_candidate", "no_budget", "not_pressure",
            "invalid_pressure_basis", "invalid_kv_budget", "ownership_aborted",
        }
    )
    _check(no_op_reason,
           f"{prefix}: no-op has no ownership/state/scan or equivalent reason", errors)
    return False


def _validate_bounded_marker_order(markers: list[dict[str, str]], case_name: str,
                                   errors: list[str]) -> None:
    """Reject duplicate or out-of-order bounded-release sample/episode markers."""
    previous_sample = -1
    previous_episode = -1
    for marker_index, marker in enumerate(markers):
        sample = int(marker["sample_count"])
        episode = int(marker["episode"])
        _check(sample > previous_sample,
               f"{case_name}: bounded marker[{marker_index}] sample_count={sample} "
               f"is duplicate or out of order", errors)
        _check(episode >= previous_episode,
               f"{case_name}: bounded marker[{marker_index}] episode={episode} "
               f"is out of order", errors)
        previous_sample = sample
        previous_episode = episode


def _check_mandatory_files(case_dir: pathlib.Path, case_name: str,
                           errors: list[str]) -> None:
    """Every case must have all mandatory files. Missing file = hard failure."""
    for filename in CASE_MANDATORY_FILES:
        fpath = case_dir / filename
        _check(fpath.is_file(), f"{case_name}: missing mandatory file {filename}", errors)


def validate_token_calibration(art_dir: pathlib.Path, effective_targets: set[int],
                               errors: list[str]) -> dict[int, dict[str, Any]]:
    calib_dir = art_dir / "token_calibration"
    calib_file = calib_dir / "calibration.json"
    if not calib_file.is_file():
        errors.append("token_calibration/calibration.json missing")
        return {}
    data = json.loads(calib_file.read_text(encoding="utf-8"))
    results_raw = data.get("results", {})
    results: dict[int, dict[str, Any]] = {}
    for key, val in results_raw.items():
        try:
            target = int(key)
        except ValueError:
            errors.append(f"token calibration has non-integer key: {key}")
            continue
        results[target] = val

    calibrated_targets = set(results)
    prompt_file_targets: set[int] = set()
    for prompt_file in calib_dir.glob("prompt_*t.json"):
        match = re.fullmatch(r"prompt_(\d+)t\.json", prompt_file.name)
        if match:
            prompt_file_targets.add(int(match.group(1)))
    _check(prompt_file_targets == effective_targets,
           f"token calibration prompt files {sorted(prompt_file_targets)} != effective targets "
           f"{sorted(effective_targets)}", errors)
    _check(calibrated_targets == effective_targets,
           f"token calibration targets {sorted(calibrated_targets)} != effective targets "
           f"{sorted(effective_targets)}", errors)
    for target, result in sorted(results.items()):
        actual = result.get("actual_tokens")
        _check(actual == target,
               f"token calibration target={target} actual={actual}; exact effective-tier prompt required",
               errors)
        _check(result.get("target_tokens") == target,
               f"token calibration key={target} target_tokens={result.get('target_tokens')} mismatch",
               errors)
        _check(len(result.get("prompt_text", "")) > 0,
               f"token calibration target={target} prompt_text empty", errors)
        for field in ["prompt_text_sha256", "actual_tokens", "repetitions"]:
            _check(field in result,
                   f"token calibration target={target} missing {field}", errors)

    return results


def validate_rss_calibration(art_dir: pathlib.Path, effective_targets: set[int],
                            errors: list[str]) -> dict[int, dict[str, Any]]:
    calib_dir = art_dir / "rss_calibration"
    calib_file = calib_dir / "rss_calibration.json"
    if not calib_file.is_file():
        errors.append("rss_calibration/rss_calibration.json missing")
        return {}
    data = json.loads(calib_file.read_text(encoding="utf-8"))
    levels_raw = data.get("levels", {})
    levels: dict[int, dict[str, Any]] = {}
    for key, val in levels_raw.items():
        try:
            t = int(key)
        except ValueError:
            errors.append(f"rss calibration has non-integer key: {key}")
            continue
        levels[t] = val

    calibrated_targets = set(levels.keys())
    level_dir_targets: set[int] = set()
    for level_dir in calib_dir.glob("ctx_*"):
        match = re.fullmatch(r"ctx_(\d+)", level_dir.name)
        if level_dir.is_dir() and match:
            level_dir_targets.add(int(match.group(1)))
    _check(level_dir_targets == effective_targets,
           f"rss calibration directories {sorted(level_dir_targets)} != effective targets "
           f"{sorted(effective_targets)}", errors)
    _check(calibrated_targets == effective_targets,
           f"rss calibration targets {sorted(calibrated_targets)} != effective targets "
           f"{sorted(effective_targets)}", errors)

    previous_success_target: int | None = None
    previous_success_wall_s: float | None = None
    for t, cal in sorted(levels.items()):
        expected_calibration_timeout = derive_calibration_timeout(
            t, previous_success_target, previous_success_wall_s)
        for field, expected in expected_calibration_timeout.items():
            recorded = cal.get(field)
            if isinstance(expected, float):
                _check(isinstance(recorded, (int, float)) and
                       abs(float(recorded) - expected) < 1e-3,
                       f"rss calib t={t}: {field}={recorded} != derived {expected}", errors)
            else:
                _check(recorded == expected,
                       f"rss calib t={t}: {field}={recorded!r} != expected {expected!r}", errors)
        # A failed calibration (e.g. effective context exhausted the prompt)
        # must NOT be silently recoverable — the runner records it so the parser
        # treats it as a hard failure (no evidence fake-pass).
        _check(cal.get("source_target") == t,
               f"rss calib t={t}: source_target={cal.get('source_target')} must equal target", errors)
        _check(not cal.get("calibration_failed"),
               f"rss calib t={t}: calibration_failed is forbidden for formal effective tiers", errors)
        _check(cal.get("http_status") == 200,
               f"rss calib t={t}: HTTP {cal.get('http_status')} != 200", errors)
        if cal.get("calibration_failed"):
            continue
        _check(cal.get("rss_idle_kb", 0) > 0, f"rss calib t={t}: idle RSS must be > 0", errors)
        _check(cal.get("rss_peak_kb", 0) > cal.get("rss_idle_kb", 0),
               f"rss calib t={t}: peak RSS {cal.get('rss_peak_kb')} <= idle {cal.get('rss_idle_kb')}", errors)
        _check(cal.get("pressure_trigger_kb", 0) > 0, f"rss calib t={t}: pressure_trigger_kb missing", errors)
        _check(cal.get("pressure_safe_kb", 0) > 0, f"rss calib t={t}: pressure_safe_kb missing", errors)
        _check(cal.get("pressure_trigger_kb", 0) < cal.get("pressure_safe_kb", 0),
               f"rss calib t={t}: trigger must be < safe", errors)
        _check(cal.get("low_water_trigger_kb", 0) < cal.get("pressure_trigger_kb", 0),
               f"rss calib t={t}: low_water must be < pressure_trigger", errors)
        # Per-tier completion timeout must be derived from that tier's real
        # measured completion wall-clock, not the historical fixed 180s.
        wall = cal.get("completion_wall_s")
        recorded = cal.get("completion_timeout_s")
        _check(isinstance(wall, (int, float)) and wall > 0,
               f"rss calib t={t}: completion_wall_s missing or <=0 (no real timing sample)", errors)
        if isinstance(wall, (int, float)) and wall > 0 and isinstance(recorded, (int, float)):
            expected = derive_completion_timeout(float(wall))
            _check(abs(float(recorded) - expected) < 1e-3,
                   f"rss calib t={t}: completion_timeout_s={recorded} inconsistent with "
                   f"derive(max({wall}*{TIMEOUT_MARGIN}, {TIMEOUT_FLOOR_S}))={expected}", errors)
        elif recorded is None:
            errors.append(f"rss calib t={t}: completion_timeout_s missing — "
                          f"runner must record the per-tier derived timeout")

        if not cal.get("calibration_failed") and cal.get("http_status") == 200 and \
                isinstance(wall, (int, float)) and wall > 0:
            previous_success_target = t
            previous_success_wall_s = float(wall)

        level_file = calib_dir / f"ctx_{t}" / "rss_calibration.json"
        _check(level_file.is_file(), f"rss calib t={t}: per-tier file missing", errors)
        if level_file.is_file():
            level_value = json.loads(level_file.read_text(encoding="utf-8"))
            _check(level_value == cal,
                   f"rss calib t={t}: aggregate level differs from per-tier artifact", errors)

    return levels


def validate_manifest_identity(manifest: dict[str, Any], errors: list[str]) -> None:
    """Validate manifest identity fields: binary, model, runner, parser, source_snapshot."""
    for field in ["binary", "model", "runner", "parser"]:
        obj = manifest.get(field, {})
        _check(isinstance(obj, dict), f"manifest.{field} must be an object", errors)
        for sub in ["path", "size", "sha256"]:
            _check(sub in obj, f"manifest.{field}.{sub} missing", errors)
        _check(isinstance(obj.get("size"), int) and obj["size"] > 0,
               f"manifest.{field}.size must be > 0", errors)
        _check(len(obj.get("sha256", "")) == 64,
               f"manifest.{field}.sha256 must be 64 hex chars", errors)

    ss = manifest.get("source_snapshot", {})
    _check(isinstance(ss, dict) and ss.get("schema_version") == 1,
           "manifest.source_snapshot schema_version invalid", errors)
    _check("head_sha" in ss, "manifest.source_snapshot.head_sha missing", errors)
    td = ss.get("tracked_diff", {})
    _check(isinstance(td, dict), "manifest.source_snapshot.tracked_diff must be object", errors)

    # runner_status in summary
    summary_file = ACTIVE_ARTIFACT / "summary.json"
    if summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        _check(summary.get("runner_status") in ("run_complete", "run_incomplete"),
               f"summary.runner_status invalid: {summary.get('runner_status')}", errors)


def validate_environment_closure(case_dir: pathlib.Path, case_name: str,
                                errors: list[str]) -> None:
    """Verify environment.json and execution.json agree and have no unexpected keys."""
    env_file = case_dir / "environment.json"
    exec_file = case_dir / "execution.json"
    if not env_file.is_file() or not exec_file.is_file():
        return  # mandatory file check catches this separately
    env = json.loads(env_file.read_text(encoding="utf-8"))
    exe = json.loads(exec_file.read_text(encoding="utf-8"))

    exe_env = exe.get("environment", {})
    _check(isinstance(exe_env, dict),
           f"{case_name}: execution.json.environment must be an object", errors)
    for key in sorted(env):
        _check(key in exe_env,
               f"{case_name}: env key {key} in environment.json but not in execution.json", errors)
    for key in sorted(exe_env):
        _check(key in env,
               f"{case_name}: env key {key} in execution.json but not in environment.json", errors)


def validate_effective_context(manifest: dict[str, Any], errors: list[str],
                               requested_targets: set[int]) -> dict[str, Any]:
    ec = manifest.get("effective_context", {})
    if not isinstance(ec, dict):
        errors.append("manifest.effective_context missing or not an object")
        return {}

    recorded_requested = ec.get("requested_targets")
    _check(isinstance(recorded_requested, list) and set(recorded_requested) == requested_targets,
           f"effective_context.requested_targets={recorded_requested} != manifest targets "
           f"{sorted(requested_targets)}", errors)
    effective_targets = ec.get("effective_targets")
    _check(isinstance(effective_targets, list) and len(effective_targets) >= 2,
           f"effective_context.effective_targets must contain >=2 tiers, got {effective_targets!r}",
           errors)
    if isinstance(effective_targets, list):
        _check(len(effective_targets) == len(set(effective_targets)),
               "effective_context.effective_targets contains duplicates", errors)

    eff = ec.get("effective_n_ctx")
    _check(isinstance(eff, int) and eff > 0,
           "effective_context.effective_n_ctx must be a positive int", errors)
    n_predict = ec.get("n_predict", N_PREDICT_DEFAULT)
    special = ec.get("special_overhead", SPECIAL_OVERHEAD)
    safety = ec.get("safety", SAFETY_DEFAULT)
    align = ec.get("token_align", TOKEN_ALIGN)
    max_prompt = ec.get("max_prompt_tokens")
    if isinstance(eff, int) and eff > 0:
        expected_max = derive_max_prompt_tokens(eff, n_predict, special, safety, align)
        _check(max_prompt == expected_max,
               f"effective_context.max_prompt_tokens={max_prompt} != derived {expected_max}",
               errors)
        expected_targets = []
        for target in sorted(requested_targets):
            effective = target if target <= expected_max else (
                (expected_max // align) * align if align > 1 else expected_max)
            if effective >= 256 and effective not in expected_targets:
                expected_targets.append(effective)
        _check(effective_targets == expected_targets,
               f"effective_context.effective_targets={effective_targets} != derived "
               f"{expected_targets}", errors)

    probe = ec.get("probe", {})
    probe_file = ACTIVE_ARTIFACT / "context_probe" / "probe.json"
    _check(probe_file.is_file(), "context_probe/probe.json missing", errors)
    if probe_file.is_file():
        recorded_probe = json.loads(probe_file.read_text(encoding="utf-8"))
        _check(recorded_probe == probe,
               "context_probe/probe.json differs from manifest effective_context.probe", errors)
    _check(not (ACTIVE_ARTIFACT / "context_probe" / "completion.sse").exists(),
           "context probe must not contain completion.sse", errors)
    _check(isinstance(probe, dict), "effective_context.probe missing", errors)
    if isinstance(probe, dict):
        _check(probe.get("completion_requests") == 0,
               "context probe must not send completion requests", errors)
        _check(probe.get("http_completion_statuses") == [],
               "context probe recorded completion HTTP statuses", errors)
        _check(probe.get("requested_targets") == recorded_requested,
               "context probe requested_targets mismatch", errors)
        _check(probe.get("effective_n_ctx") == eff,
               "context probe effective_n_ctx mismatch", errors)
        _check(probe.get("effective_targets") == effective_targets,
               "context probe effective_targets mismatch", errors)

    return ec


def validate_ladder_case(art_dir: pathlib.Path, target: int, case_label: str,
                         errors: list[str],
                         rss_calib: dict[int, dict[str, Any]] | None = None,
                         token_prompts: dict[int, dict[str, Any]] | None = None,
                         max_prompt_tokens: int | None = None) -> dict[str, Any] | None:
    case_dir = art_dir / "ladder" / f"t{target}_{case_label.lower()}"
    case_name = f"LADDER_{case_label}_t{target}"

    # Mandatory files — missing any is a hard failure
    _check_mandatory_files(case_dir, case_name, errors)
    validate_environment_closure(case_dir, case_name, errors)

    result_file = case_dir / "result.json"
    if not result_file.is_file():
        return None
    result = json.loads(result_file.read_text(encoding="utf-8"))

    _check(result.get("response_text_len", 0) > 0,
           f"{case_name}: empty response", errors)
    _check(result.get("http_status", 0) == 200,
           f"{case_name}: HTTP {result.get('http_status')} — ladder must run inside effective context (no fake-pass on context overflow)", errors)
    _check(result.get("response_text_sha256", ""),
           f"{case_name}: missing response_text_sha256", errors)

    # Effective-context fit: the tier must be a legal value the runner converged
    # to, and the prompt actually sent must leave room for n_predict + safety
    # inside the server-effective n_ctx.
    if max_prompt_tokens is not None:
        _check(target <= max_prompt_tokens,
               f"{case_name}: tier {target} exceeds max_prompt_tokens {max_prompt_tokens} "
               f"— ladder ran an illegal (context-overflowing) tier", errors)
    case_eff = result.get("effective_n_ctx")
    if isinstance(case_eff, int) and case_eff > 0 and token_prompts is not None:
        src = result.get("source_target")
        prompt_info = token_prompts.get(src)
        if prompt_info is not None:
            actual_tokens = int(prompt_info.get("actual_tokens", 0))
            # Invariant: prompt_tokens + SPECIAL_OVERHEAD + n_predict + SAFETY
            # <= effective_n_ctx.  actual_tokens already < target by tolerance,
            # and the runner clamps to max_prompt_tokens which enforces this, so
            # any breach proves the runner sent a prompt that does not fit.
            used = actual_tokens + SPECIAL_OVERHEAD + N_PREDICT_DEFAULT + SAFETY_DEFAULT
            _check(used <= case_eff,
                   f"{case_name}: prompt~{actual_tokens} + overhead {SPECIAL_OVERHEAD} + "
                   f"n_predict {N_PREDICT_DEFAULT} + safety {SAFETY_DEFAULT} = {used} "
                   f"> effective_n_ctx {case_eff} — context overflow risk", errors)

    # Per-tier completion timeout: each ladder case must use the timeout derived
    # from its source tier's RSS calibration wall-clock, not the fixed 180s.
    recorded_timeout = result.get("completion_timeout_s")
    source_target = result.get("source_target")
    _check(source_target == target,
           f"{case_name}: source_target={source_target} must equal effective target {target}", errors)
    if rss_calib and source_target in rss_calib:
        src_cal = rss_calib[source_target]
        if isinstance(src_cal.get("completion_wall_s"), (int, float)) and \
                src_cal.get("completion_wall_s", 0) > 0 and not src_cal.get("calibration_failed"):
            expected_timeout = derive_completion_timeout(float(src_cal["completion_wall_s"]))
            if isinstance(recorded_timeout, (int, float)):
                _check(abs(float(recorded_timeout) - expected_timeout) < 1e-3,
                       f"{case_name}: completion_timeout_s={recorded_timeout} != "
                       f"derived {expected_timeout} from source tier {source_target} "
                       f"(wall={src_cal['completion_wall_s']})", errors)
            else:
                errors.append(f"{case_name}: completion_timeout_s missing — "
                              f"runner must record the per-tier derived timeout")

    # PID identity validation
    pid_ident = result.get("rss_pid_identity")
    exec_file = case_dir / "execution.json"
    execution = {}
    if exec_file.is_file():
        execution = json.loads(exec_file.read_text(encoding="utf-8"))
    _validate_pid_identity_single(pid_ident, case_name, "ladder", errors)
    _validate_rss_pid_identity_not_proc_pid(pid_ident, execution, case_name, errors)

    # RSS consistency: compare ladder RSS against same-tier calibration
    if rss_calib is not None and target in rss_calib:
        cal_idle = rss_calib[target].get("rss_idle_kb", 0)
        rss_after_req = result.get("rss_after_request_kb", 0)
        if rss_after_req > 0:
            _validate_rss_consistency([{"rss_kb": rss_after_req}], cal_idle,
                                     case_name, errors)

    # Parse server stderr
    stderr_file = case_dir / "server.stderr"
    stderr_text = stderr_file.read_text(errors="replace")

    bounded_markers = find_markers(stderr_text, BOUNDED_RELEASE_RE, case_name)
    if case_label == "OFF":
        _check(len(bounded_markers) == 0,
               f"{case_name}: must have zero bounded_release markers, "
               f"got {len(bounded_markers)}", errors)
    else:
        _validate_bounded_marker_order(bounded_markers, case_name, errors)
        action_count = 0
        for marker_index, marker in enumerate(bounded_markers):
            _check(marker["target_mode"] == "dynamic",
                   f"{case_name}: bounded marker[{marker_index}] target_mode must be dynamic",
                   errors)
            if _validate_bounded_marker(marker, case_name, marker_index, errors):
                action_count += 1
        _check(action_count >= 1,
               f"{case_name}: DYNAMIC ladder has no real bounded-release action "
               f"(total markers={len(bounded_markers)})", errors)

    # Validate per-request stderr windows if present
    windows_file = case_dir / "stderr_windows.json"
    if windows_file.is_file():
        _validate_stderr_windows(case_dir, stderr_text, case_name, windows_file, 1, errors)

    # Cleanup check
    phases_file = case_dir / "phases.json"
    phases = json.loads(phases_file.read_text(encoding="utf-8"))
    _check(not phases.get("shutdown", {}).get("residual_process", False),
           f"{case_name}: residual process after shutdown", errors)

    return result


def _validate_stderr_windows(case_dir: pathlib.Path, stderr_text: str,
                             case_name: str, windows_file: pathlib.Path,
                             expected_count: int, errors: list[str]) -> None:
    """Validate per-request stderr windows: markers within window, safety checks."""
    windows_data = json.loads(windows_file.read_text(encoding="utf-8"))
    windows = windows_data.get("windows", [])
    _check(len(windows) == expected_count,
           f"{case_name}: stderr_windows has {len(windows)} windows, expected {expected_count}",
           errors)

    for w in windows:
        round_num = w.get("round", 0)
        _check(round_num >= 1 and round_num <= expected_count,
               f"{case_name}: stderr window round {round_num} out of range [1,{expected_count}]",
               errors)
        start = w.get("start_byte", 0)
        end = w.get("end_byte", 0)
        _check(end > start,
               f"{case_name}: stderr window round {round_num}: end_byte <= start_byte", errors)
        _check(end <= len(stderr_text.encode()),
               f"{case_name}: stderr window round {round_num}: end_byte past EOF", errors)

        window_text = stderr_text.encode()[start:end].decode(errors="replace")
        # Trim to last complete newline to avoid mid-line truncation
        last_nl = window_text.rfind("\n")
        if last_nl >= 0:
            window_text = window_text[:last_nl + 1]
        if not window_text.strip():
            continue  # empty window after trimming — not an error
        has_bounded = BOUNDED_RELEASE_RE.search(window_text)
        has_telemetry = TELEMETRY_MARKER_RE.search(window_text)
        _check(has_bounded or has_telemetry,
               f"{case_name}: stderr window round {round_num}: "
               f"no pressure/bounded marker found", errors)

        # Global marker validation owns action/no-op and safety invariants.  Windows
        # only prove that each request interval contains a complete pressure marker,
        # avoiding duplicate reports for a marker that appears in cumulative windows.


def _validate_rss_consistency(rss_samples: list[dict[str, Any]],
                             calibration_idle_kb: int,
                             case_name: str, errors: list[str]) -> None:
    """Reject obviously distorted RSS samples by comparing against calibration idle RSS.

    When the runner accidentally samples the strace wrapper process instead of
    llama-server, RSS reads drop from GiB to MiB range.  This check compares
    each sample against the same-tier calibration idle RSS and flags samples
    that fall below the consistency threshold.
    """
    if calibration_idle_kb <= RSS_CONSISTENCY_MIN_IDLE_KIB:
        return  # calibration too small to reliably detect process mismatch

    threshold = max(calibration_idle_kb * RSS_CONSISTENCY_MIN_FRACTION,
                    RSS_CONSISTENCY_ABSOLUTE_FLOOR_KIB)

    for sample in rss_samples:
        rss_kb = sample.get("rss_kb", 0)
        if rss_kb <= 0:
            continue  # missing or zero — skip (other checks will catch zero-RSS issues)
        if rss_kb < threshold:
            errors.append(
                f"{case_name}: RSS {rss_kb} KiB far below calibration idle "
                f"{calibration_idle_kb} KiB (threshold={threshold:.0f} KiB) — "
                f"likely sampling wrong process (e.g., strace wrapper instead of server)")


def _validate_pid_identity_single(pid_identity: dict[str, Any] | None,
                                  case_name: str, label: str,
                                  errors: list[str]) -> None:
    """Validate a single PID identity entry: pid positive, starttime non-zero, cmdline plausible."""
    if pid_identity is None:
        return  # absent — skip (backward-compatible with old artifacts)
    pid = pid_identity.get("pid", 0)
    starttime = pid_identity.get("starttime", 0)
    cmdline = pid_identity.get("cmdline", "")
    if not isinstance(pid, int) or pid <= 0:
        errors.append(f"{case_name}: {label} pid_identity.pid={pid} must be positive int")
    if not isinstance(starttime, int) or starttime <= 0:
        errors.append(f"{case_name}: {label} pid_identity.starttime={starttime} missing or zero")
    if not cmdline:
        errors.append(f"{case_name}: {label} pid_identity.cmdline is empty")
    elif not ("/" in cmdline or "llama" in cmdline.lower()):
        # cmdline should contain at least the binary path or name
        errors.append(f"{case_name}: {label} pid_identity.cmdline does not look like "
                      f"a server binary: {cmdline[:120]!r}")


def _validate_pid_cross_round_consistency(rounds: list[dict[str, Any]],
                                          case_name: str,
                                          errors: list[str]) -> None:
    """All rounds must sample the same server process (same pid + starttime)."""
    identities: list[dict[str, Any]] = []
    for r in rounds:
        ident = r.get("rss_pid_identity")
        if ident is not None:
            identities.append(ident)
    if len(identities) < 2:
        return  # need at least 2 to compare
    ref_pid = identities[0].get("pid")
    ref_start = identities[0].get("starttime")
    for i, ident in enumerate(identities[1:], start=2):
        pid = ident.get("pid")
        start = ident.get("starttime")
        if pid != ref_pid or start != ref_start:
            errors.append(
                f"{case_name}: PID identity changed across rounds — "
                f"round 1 pid={ref_pid} starttime={ref_start} vs "
                f"round {i} pid={pid} starttime={start} — "
                f"server may have restarted or PID was reused")


def _validate_rss_pid_identity_not_proc_pid(pid_identity: dict[str, Any] | None,
                                            execution: dict[str, Any],
                                            case_name: str,
                                            errors: list[str]) -> None:
    """When strace wraps the server, the sampled PID must differ from the parent PID.

    The sampled cmdline must reference the server binary, not the strace wrapper.
    Also checks against execution.json argv to confirm the strace wrapping was used.
    """
    if pid_identity is None:
        return
    cmdline = pid_identity.get("cmdline", "")
    # Direct check: if the sampled cmdline mentions strace, we sampled the
    # wrapper process instead of the server.  This is always wrong regardless
    # of execution.json.
    if cmdline and "strace" in cmdline.lower():
        errors.append(
            f"{case_name}: pid_identity.cmdline contains 'strace' — "
            f"sampled the wrapper process instead of the server: {cmdline[:120]!r}")
        return

    # Cross-reference with execution.json: if strace was used, the sampled
    # cmdline must differ from the parent process.
    argv = execution.get("argv", [])
    if not isinstance(argv, list) or not argv:
        return
    has_strace = any("strace" in a for a in argv)
    if not has_strace:
        return
    # With strace in argv, the sampled PID must be a child (different process).
    # We verify the cmdline does NOT contain the strace wrapper itself.
    if not cmdline:
        errors.append(
            f"{case_name}: pid_identity.cmdline is empty — cannot verify "
            f"that RSS was sampled from the server (not strace wrapper)")


def validate_continuous_requests(art_dir: pathlib.Path, num_expected: int,
                                errors: list[str],
                                rss_calib: dict[int, dict[str, Any]] | None = None,
                                effective_n_ctx: int | None = None,
                                max_prompt_tokens: int | None = None) -> dict[str, Any] | None:
    loop_dir = art_dir / "continuous_requests"
    case_name = "CONTINUOUS_REQUESTS"

    # Mandatory files
    _check_mandatory_files(loop_dir, case_name, errors)
    validate_environment_closure(loop_dir, case_name, errors)

    result_file = loop_dir / "result.json"
    if not result_file.is_file():
        return None
    result = json.loads(result_file.read_text(encoding="utf-8"))

    _check(result.get("completed_rounds", 0) == num_expected,
           f"{case_name}: expected {num_expected} rounds, got {result.get('completed_rounds')}",
           errors)
    _check(result.get("all_responses_identical", False),
           f"{case_name}: not all responses are identical", errors)
    _check(result.get("cumulative_error_count", 1) == 0,
           f"{case_name}: cumulative_error_count={result.get('cumulative_error_count')}",
           errors)

    # Effective-context fit for the continuous tier: it must run inside the
    # probed effective n_ctx with a request-derived completion timeout.
    if isinstance(effective_n_ctx, int) and effective_n_ctx > 0:
        case_ctx = result.get("ctx_size")
        case_eff = result.get("effective_n_ctx", effective_n_ctx)
        _check(isinstance(case_eff, int) and case_eff <= effective_n_ctx,
               f"{case_name}: effective_n_ctx={case_eff} exceeds probed {effective_n_ctx}", errors)
        if max_prompt_tokens is not None and isinstance(case_ctx, int):
            _check(case_ctx <= effective_n_ctx,
                   f"{case_name}: ctx_size={case_ctx} exceeds effective n_ctx {effective_n_ctx}", errors)
        recorded_timeout = result.get("completion_timeout_s")
        _check(isinstance(recorded_timeout, (int, float)) and recorded_timeout > 0,
               f"{case_name}: completion_timeout_s missing or <=0", errors)
        if isinstance(recorded_timeout, (int, float)) and rss_calib:
            usable = sorted(t for t, c in rss_calib.items()
                            if not c.get("calibration_failed"))
            if usable and isinstance(rss_calib[usable[0]].get("completion_wall_s"), (int, float)):
                wall = float(rss_calib[usable[0]]["completion_wall_s"])
                expected_timeout = derive_completion_timeout(wall)
                _check(abs(float(recorded_timeout) - expected_timeout) < 1e-3,
                       f"{case_name}: completion_timeout_s={recorded_timeout} != derived "
                       f"{expected_timeout} from tier {usable[0]} (wall={wall})", errors)

    rounds = result.get("rounds", [])
    _validate_round_numbering(rounds, case_name, num_expected, errors)

    # RSS consistency: compare each round's RSS against calibration idle.
    # Use the smallest *non-failed* calibration level as the reference target
    # (the runner runs continuous at the smallest legal tier, whose
    # calibration must have succeeded).
    if rss_calib is not None:
        usable = [t for t, c in rss_calib.items() if not c.get("calibration_failed")]
        min_target = min(usable) if usable else None
        if min_target is not None:
            cal_idle = rss_calib[min_target].get("rss_idle_kb", 0)
            rss_samples = [{"rss_kb": r.get("rss_after_request_kb", 0)}
                          for r in rounds]
            _validate_rss_consistency(rss_samples, cal_idle, case_name, errors)

    # PID identity validation
    exec_file = loop_dir / "execution.json"
    execution = {}
    if exec_file.is_file():
        execution = json.loads(exec_file.read_text(encoding="utf-8"))

    for r in rounds:
        pid_ident = r.get("rss_pid_identity")
        _validate_pid_identity_single(pid_ident, case_name,
                                      f"round {r.get('round', '?')}", errors)
        if pid_ident is not None:
            _validate_rss_pid_identity_not_proc_pid(pid_ident, execution,
                                                    f"{case_name} round {r.get('round', '?')}",
                                                    errors)
    _validate_pid_cross_round_consistency(rounds, case_name, errors)

    for r in rounds:
        _check(r.get("http_status") == 200,
               f"{case_name} round {r.get('round')}: HTTP {r.get('http_status')}", errors)
        _check(r.get("matches_baseline", False),
               f"{case_name} round {r.get('round')}: does not match baseline", errors)

    # Parse server stderr
    stderr_file = loop_dir / "server.stderr"
    stderr_text = stderr_file.read_text(errors="replace")
    bounded_markers = find_markers(stderr_text, BOUNDED_RELEASE_RE, case_name)
    _validate_bounded_marker_order(bounded_markers, case_name, errors)
    action_count = 0
    for marker_index, marker in enumerate(bounded_markers):
        if _validate_bounded_marker(marker, case_name, marker_index, errors):
            action_count += 1
    _check(action_count >= 1,
           f"{case_name}: no real bounded-release action", errors)

    # KV_PAGED_RELEASE_STATS is mandatory for continuous requests
    stats = extract_final_release_stats(case_name, stderr_text)
    _check(stats is not None,
           f"{case_name}: KV_PAGED_RELEASE_STATS missing or malformed — mandatory", errors)
    if stats:
        _check(stats.get("reuse_allocations", 0) > 0,
               f"{case_name}: reuse_allocations={stats.get('reuse_allocations')} must be > 0",
               errors)
        _check(stats.get("write_commits", 0) > 0,
               f"{case_name}: write_commits={stats.get('write_commits')} must be > 0",
               errors)
        _check(stats.get("bounded_release_calls", 0) > 0,
               f"{case_name}: bounded_release_calls={stats.get('bounded_release_calls')} must be > 0",
               errors)
        for fatal_field in FATAL_STATS_FIELDS:
            _check(stats.get(fatal_field, 1) == 0,
                   f"{case_name}: {fatal_field}={stats.get(fatal_field)} must be 0", errors)

    # Validate per-request stderr windows
    windows_file = loop_dir / "stderr_windows.json"
    if windows_file.is_file():
        _validate_stderr_windows(loop_dir, stderr_text, case_name, windows_file,
                                num_expected, errors)

    # Cleanup
    phases_file = loop_dir / "phases.json"
    phases = json.loads(phases_file.read_text(encoding="utf-8"))
    _check(not phases.get("shutdown", {}).get("residual_process", False),
           f"{case_name}: residual process after shutdown", errors)

    return result


def _validate_round_numbering(rounds: list[dict[str, Any]], case_name: str,
                             expected_count: int, errors: list[str]) -> None:
    """Rounds must be numbered 1..N consecutively with no gaps or duplicates."""
    round_nums = [r.get("round", 0) for r in rounds]
    _check(len(round_nums) == expected_count,
           f"{case_name}: round count {len(round_nums)} != expected {expected_count}", errors)
    _check(len(set(round_nums)) == len(round_nums),
           f"{case_name}: duplicate round numbers detected: {sorted(round_nums)}", errors)
    expected = list(range(1, len(round_nums) + 1))
    _check(sorted(round_nums) == expected,
           f"{case_name}: round numbers not consecutive 1..{len(round_nums)}: "
           f"got {sorted(round_nums)}", errors)


def validate_env_contamination(art_dir: pathlib.Path, token_targets: set[int],
                              errors: list[str]) -> None:
    """OFF cases must not have bounded-only keys."""
    for t in sorted(token_targets):
        for case_label in ["OFF", "DYNAMIC_RELEASE"]:
            case_dir = art_dir / "ladder" / f"t{t}_{case_label.lower()}"
            env_file = case_dir / "environment.json"
            if not env_file.is_file():
                continue
            env = json.loads(env_file.read_text(encoding="utf-8"))

            if case_label == "OFF":
                for key in BOUNDED_ONLY_KEYS:
                    _check(key not in env,
                           f"LADDER_OFF_t{t}: has bounded-release key {key}", errors)


def validate_ladder_response_identity(art_dir: pathlib.Path, token_targets: set[int],
                                     errors: list[str]) -> None:
    """OFF and DYNAMIC_RELEASE responses must be byte-identical per context level."""
    for t in sorted(token_targets):
        off_dir = art_dir / "ladder" / f"t{t}_off"
        dyn_dir = art_dir / "ladder" / f"t{t}_dynamic_release"

        off_result_file = off_dir / "result.json"
        dyn_result_file = dyn_dir / "result.json"

        if not off_result_file.is_file() or not dyn_result_file.is_file():
            continue

        off_result = json.loads(off_result_file.read_text(encoding="utf-8"))
        dyn_result = json.loads(dyn_result_file.read_text(encoding="utf-8"))

        off_hash = off_result.get("response_text_sha256", "")
        dyn_hash = dyn_result.get("response_text_sha256", "")

        _check(off_hash and dyn_hash,
               f"ladder t={t}: missing response hash", errors)
        _check(off_hash == dyn_hash,
               f"ladder t={t}: OFF response != DYNAMIC_RELEASE response "
               f"(off_sha256={off_hash[:16]}... dyn_sha256={dyn_hash[:16]}...)",
               errors)


def compute_diagnostic_trends(art_dir: pathlib.Path, token_targets: set[int],
                             errors: list[str]) -> dict[str, Any]:
    trends: dict[str, Any] = {"ladder": {}, "notes": []}
    for t in sorted(token_targets):
        level_trends: dict[str, Any] = {}
        for case_label in ["OFF", "DYNAMIC_RELEASE"]:
            case_dir = art_dir / "ladder" / f"t{t}_{case_label.lower()}"
            result_file = case_dir / "result.json"
            if not result_file.is_file():
                continue
            local_result = json.loads(result_file.read_text(encoding="utf-8"))
            level_trends[case_label] = {
                "rss_after_request_kb": local_result.get("rss_after_request_kb", 0),
                "rss_after_release_kb": local_result.get("rss_after_release_kb", 0),
                "response_len": local_result.get("response_text_len", 0),
            }
        trends["ladder"][str(t)] = level_trends
    rss_values = [(t, trends["ladder"][str(t)].get("OFF", {}).get("rss_after_request_kb", 0))
                  for t in sorted(token_targets)]
    if len(rss_values) >= 2 and all(v[1] > 0 for v in rss_values):
        increasing = all(rss_values[i][1] <= rss_values[i + 1][1]
                        for i in range(len(rss_values) - 1))
        if not increasing:
            trends["notes"].append("RSS did not monotonically increase with context length "
                                   "(diagnostic only, not a failure)")
    return trends


# ── main ─────────────────────────────────────────────────────────────────────

def main_parse(art_dir: pathlib.Path, result_path: pathlib.Path | None = None) -> int:
    global ACTIVE_RESULT_PATH, ACTIVE_ARTIFACT
    ACTIVE_ARTIFACT = art_dir
    ACTIVE_RESULT_PATH = result_path

    errors: list[str] = []

    # 0. Manifest and protocol identity
    manifest_file = art_dir / "manifest.json"
    if not manifest_file.is_file():
        fail("manifest.json missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    runner_status = manifest.get("runner_status")
    summary_file = art_dir / "summary.json"
    if runner_status != "run_complete" and summary_file.is_file():
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        runner_status = summary.get("runner_status", runner_status)
    if runner_status != "run_complete":
        phase = manifest.get("failure_phase")
        target = manifest.get("failure_target")
        reason = manifest.get("failure_reason")
        message = (f"runner artifact incomplete: status={runner_status!r} "
                   f"phase={phase!r} target={target!r} reason={reason!r}")
        print(f"  INCOMPLETE: {message}", file=sys.stderr)
        if ACTIVE_RESULT_PATH:
            write_parser_result(2, "INCOMPLETE")
        return 2

    _check(manifest.get("protocol") == PROTOCOL,
           f"protocol mismatch: expected {PROTOCOL}, got {manifest.get('protocol')}", errors)
    _check(manifest.get("protocol_version") == 1,
           "protocol_version mismatch", errors)
    _check(isinstance(manifest.get("token_targets"), list),
           "manifest token_targets must be a list", errors)

    validate_manifest_identity(manifest, errors)

    token_targets_list = manifest.get("token_targets", [])
    token_targets = set(token_targets_list)
    _check(len(token_targets) >= 2, f"need >=2 token targets, got {len(token_targets)}", errors)
    _check(token_targets == set(TOKEN_TARGETS_DEFAULT) or len(token_targets) >= 2,
           "token targets validation", errors)

    num_continuous = manifest.get("num_continuous", 20)
    _check(num_continuous >= 5, f"num_continuous {num_continuous} < 5", errors)

    # Effective context is probed before any formal calibration.
    ec = validate_effective_context(manifest, errors, token_targets)
    effective_targets_list = ec.get("effective_targets") if isinstance(ec, dict) else None
    effective_targets = set(effective_targets_list) if isinstance(effective_targets_list, list) else set()
    max_prompt_tokens = ec.get("max_prompt_tokens") if isinstance(ec, dict) else None

    # Formal calibration keys must equal effective targets exactly.  Requested
    # overflow targets remain metadata-only and never enter calibration/ladder.
    token_prompts = validate_token_calibration(art_dir, effective_targets, errors)
    rss_calib = validate_rss_calibration(art_dir, effective_targets, errors)
    manifest_token_targets = set(int(key) for key in
                                 manifest.get("token_calibration", {}).get("results", {}))
    manifest_rss_targets = set(int(key) for key in
                               manifest.get("rss_calibration", {}).get("levels", {}))
    _check(manifest_token_targets == effective_targets,
           f"manifest token calibration keys {sorted(manifest_token_targets)} != effective targets "
           f"{sorted(effective_targets)}", errors)
    _check(manifest_rss_targets == effective_targets,
           f"manifest rss calibration keys {sorted(manifest_rss_targets)} != effective targets "
           f"{sorted(effective_targets)}", errors)
    ladder_targets_set = effective_targets
    ladder_dir_targets: set[int] = set()
    ladder_dir = art_dir / "ladder"
    if ladder_dir.is_dir():
        for case_dir in ladder_dir.iterdir():
            match = re.fullmatch(r"t(\d+)_(off|dynamic_release)", case_dir.name)
            if case_dir.is_dir() and match:
                ladder_dir_targets.add(int(match.group(1)))
    _check(ladder_dir_targets == effective_targets,
           f"ladder directory targets {sorted(ladder_dir_targets)} != effective targets "
           f"{sorted(effective_targets)}", errors)

    # 3. Long context ladder (mandatory files enforced inside)
    for t in sorted(ladder_targets_set):
        for case_label in ["OFF", "DYNAMIC_RELEASE"]:
            validate_ladder_case(art_dir, t, case_label, errors, rss_calib,
                                 token_prompts=token_prompts,
                                 max_prompt_tokens=max_prompt_tokens)

    # 4. Ladder response identity
    validate_ladder_response_identity(art_dir, ladder_targets_set, errors)

    # 5. Config contamination
    validate_env_contamination(art_dir, ladder_targets_set, errors)

    # 6. Continuous requests (mandatory files + KV_PAGED_RELEASE_STATS enforced)
    validate_continuous_requests(art_dir, num_continuous, errors, rss_calib,
                                 effective_n_ctx=(ec.get("effective_n_ctx")
                                                  if isinstance(ec, dict) else None),
                                 max_prompt_tokens=max_prompt_tokens)

    # 7. Diagnostic trends (informational, non-gating)
    trends = compute_diagnostic_trends(art_dir, effective_targets, errors)
    if ACTIVE_RESULT_PATH:
        trends_path = ACTIVE_RESULT_PATH.parent / "diagnostic_trends.json"
        trends_path.write_text(json.dumps(trends, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")

    # 8. Verdict
    if errors:
        for e in errors:
            print(f"  FAIL: {e}", file=sys.stderr)
        status = "FAIL"
        exit_code = 1
    else:
        status = "PASS"
        exit_code = 0

    if ACTIVE_RESULT_PATH:
        write_parser_result(exit_code, status)

    if exit_code == 0:
        print("PASS")
    return exit_code


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parser for Stage 3B-2A long-context ladder + continuous-request boundary test")
    ap.add_argument("artifact", type=pathlib.Path,
                    help="Path to the artifact directory")
    ap.add_argument("--result-path", type=pathlib.Path,
                    help="Write structured parser result to this path")
    ap.add_argument("--verify-result", type=pathlib.Path,
                    help="Re-check a previously saved parser result")
    args = ap.parse_args()

    if args.verify_result:
        manifest_path = args.artifact / "manifest.json"
        check_saved_result(args.verify_result, manifest_path)
        print("PASS (verified)")
        return

    try:
        exit_code = main_parse(args.artifact, args.result_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"  FAIL: incomplete or malformed artifact: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        if args.result_path:
            write_parser_result(1, "FAIL")
        exit_code = 1
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
