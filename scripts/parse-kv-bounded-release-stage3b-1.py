#!/usr/bin/env python3
"""Fail-closed parser for Stage 3B-1 v2 with RSS calibration and reuse verification.

Validates:
  0. RSS calibration exists and thresholds are derived from it (not hardcoded
     1/2/3 KiB, 1 KiB, 100 GB)
  1. OFF: zero bounded_release markers, zero destructive release, single request,
     response matches calibration
  2. FIXED: at least 1 bounded_release execution, target_mode=fixed,
     two requests, both match OFF, reuse_allocations>0, write_commits>0
  3. DYNAMIC_NOOP: bounded_release + dynamic_target configured, telemetry state
     is NORMAL (PRESSURE/CRITICAL thresholds above actual RSS), zero bounded
     release markers, zero destructive release, single request matches OFF
  4. DYNAMIC_RELEASE: target_mode=dynamic, real release (released_bytes>0),
     target == min(water_excess, hard_cap, KV_resident, KV_reclaimable),
     two requests both match OFF, reuse_allocations>0, write_commits>0
  5. All four responses are byte-identical to OFF (both requests where applicable)
  6. No config contamination
  7. Zero residual processes
  8. KV mincore shows physical page drop for FIXED and DYNAMIC_RELEASE
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

def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

# --- marker patterns ---
BOUNDED_RELEASE_RE = re.compile(r"kv_pressure_bounded_release\b")
DRY_RUN_MARKER_RE = re.compile(r"kv_pressure_dry_run\b")
TELEMETRY_MARKER_RE = re.compile(r"kv_pressure_telemetry\b")
DESTRUCTIVE_RELEASE_RE = re.compile(
    r"(?i)(?:paged_release_blocks_bounded\s*\(|"
    r"paged_block_release_bytes=(?:0*[1-9][0-9]*)|"
    r"paged_blocks_released=(?:0*[1-9][0-9]*)"
    r")"
)
MADV_DONTNEED_RE = re.compile(r"(?i)^\s*\d+\s+madvise\s*\([^)]*MADV_DONTNEED", re.MULTILINE)

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

DRY_REQUIRED_KEYS = {
    "state", "source", "stale", "release_enabled", "would_release_bytes",
    "would_release_blocks", "blocks_scanned", "blocks_skipped_owned",
    "blocks_skipped_state", "shortfall_bytes", "overshoot_bytes",
    "block_scan_exhausted", "ownership_aborted", "target_bytes",
    "max_scan_blocks", "skipped_reason", "cooldown_ms", "sample_count", "idle",
}

MARKER_SCHEMAS = {
    "kv_pressure_bounded_release": BOUNDED_REQUIRED_KEYS,
    "kv_pressure_dry_run": DRY_REQUIRED_KEYS,
    "kv_pressure_telemetry": TELEMETRY_REQUIRED_KEYS,
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
    "skipped_reason": {"none", "dry_run_active", "no_memory", "not_paged",
        "layout_unsupported", "swap_enabled", "not_ingraph", "no_layers", "no_row_idx",
        "legacy_active", "stale", "not_pressure", "cooldown",
        "structurally_disabled", "ownership_aborted", "invalid_pressure_basis",
        "invalid_kv_budget", "no_budget", "no_excess_or_candidate"},
    "target_mode": {"fixed", "dynamic"},
    "target_clamp": {"none", "max_release", "resident", "reclaimable"},
    "decision_reason": {"fixed", "dynamic", "invalid_pressure_basis", "not_pressure",
        "ownership_aborted", "invalid_kv_budget", "no_budget", "no_excess_or_candidate"},
}
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
_STATS_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UINT_RE = re.compile(r"[0-9]+")
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class ProtocolError(Exception):
    pass


BOUNDED_ONLY_KEYS = {
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET",
}

# Keys derived from calibration — allowed to differ per case but must
# come from calibration (not hardcoded artificial values)
CALIBRATED_KEYS = {
    "LLAMA_KV_PRESSURE_RSS_KB",
    "LLAMA_KV_CRITICAL_RSS_KB",
    "LLAMA_KV_LOW_WATER_RSS_KB",
}

DYNAMIC_ONLY_KEYS = {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET"}

FORBIDDEN_HARDCODED_THRESHOLDS = {
    # Must not appear as literal 1, 2, 3 KiB or 100,000,000 KiB in env values
}


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


def _validate_trigger_value(marker: str, value: str, case: str) -> None:
    if not value.strip():
        raise ProtocolError(f"{case}: {marker} trigger is empty")
    components = value.split(",")
    if any(not component for component in components):
        raise ProtocolError(f"{case}: {marker} trigger has empty component in {value!r}")
    if len(components) != len(set(components)):
        raise ProtocolError(f"{case}: {marker} trigger has duplicate values in {value!r}")
    for component in components:
        if component not in TRIGGERS:
            raise ProtocolError(
                f"{case}: {marker} trigger has unknown value {component!r} in {value!r}")


def _validate_value(marker: str, key: str, value: str, case: str) -> None:
    if key in BOOL_FIELDS and value not in {"0", "1"}:
        raise ProtocolError(f"{case}: {marker} {key} must be 0 or 1")
    if key == "trigger":
        _validate_trigger_value(marker, value, case)
        return
    if key in ENUMS and value not in ENUMS[key]:
        raise ProtocolError(f"{case}: {marker} {key} has invalid value {value!r}")
    if key not in BOOL_FIELDS and key not in ENUMS and key != "contract":
        if not _UINT_RE.fullmatch(value):
            raise ProtocolError(f"{case}: {marker} {key} must be an unsigned decimal integer")


def parse_strict_marker_line(line: str, marker: str, case: str) -> dict[str, str]:
    tokens = line.split()
    indexes = [i for i, token in enumerate(tokens) if token == marker]
    if len(indexes) != 1:
        raise ProtocolError(f"{case}: {marker} must occur as exactly one standalone token")
    fields: dict[str, str] = {}
    for token in tokens[indexes[0] + 1:]:
        if token.count("=") != 1:
            raise ProtocolError(f"{case}: {marker} has malformed token {token!r}")
        key, value = token.split("=", 1)
        if not _KEY_RE.fullmatch(key) or not value:
            raise ProtocolError(f"{case}: {marker} has malformed field {token!r}")
        if key in fields:
            raise ProtocolError(f"{case}: {marker} duplicates field {key!r}")
        fields[key] = value
    expected = MARKER_SCHEMAS[marker]
    if set(fields) != expected:
        missing, extra = expected - set(fields), set(fields) - expected
        raise ProtocolError(f"{case}: {marker} schema mismatch missing={sorted(missing)} extra={sorted(extra)}")
    for key, value in fields.items():
        _validate_value(marker, key, value, case)
    if int(fields["sample_count"]) < 1:
        raise ProtocolError(f"{case}: {marker} sample_count must be >= 1")
    if marker == "kv_pressure_dry_run" and int(fields["max_scan_blocks"]) < 1:
        raise ProtocolError(f"{case}: dry-run max_scan_blocks must be >= 1")
    if marker == "kv_pressure_bounded_release" and int(fields["max_scan_blocks"]) < 1:
        raise ProtocolError(f"{case}: bounded max_scan_blocks must be >= 1")
    return fields


def find_markers(stderr_text: str, marker_re: re.Pattern[str], case: str) -> list[dict[str, str]]:
    markers: list[dict[str, str]] = []
    marker = marker_re.pattern.replace(r"\b", "")
    for line in stderr_text.split("\n"):
        if marker_re.search(line):
            markers.append(parse_strict_marker_line(line, marker, case))
    return markers


def extract_final_release_stats(case: str, stderr_text: str, check: Any) -> dict[str, int] | None:
    marker_lines = [line for line in stderr_text.splitlines()
                    if RELEASE_STATS_MARKER in line]
    if not marker_lines:
        check(False, f"{case}: {RELEASE_STATS_MARKER} missing")
        return None

    line = marker_lines[-1]
    tokens = line.split()
    marker_indexes = [i for i, token in enumerate(tokens) if token == RELEASE_STATS_MARKER]
    if len(marker_indexes) != 1:
        check(False, f"{case}: final {RELEASE_STATS_MARKER} has ambiguous marker count={len(marker_indexes)}")
        return None

    fields: dict[str, str] = {}
    malformed = False
    for token in tokens[marker_indexes[0] + 1:]:
        if token.count("=") != 1:
            malformed = True
            continue
        key, value = token.split("=", 1)
        if not _STATS_KEY_RE.fullmatch(key) or not value:
            malformed = True
            continue
        if key in fields:
            check(False, f"{case}: final {RELEASE_STATS_MARKER} duplicates field {key!r}")
            return None
        fields[key] = value
    if malformed:
        check(False, f"{case}: final {RELEASE_STATS_MARKER} has malformed field syntax")
        return None

    missing = [name for name in REQUIRED_RELEASE_STATS_FIELDS if name not in fields]
    if missing:
        check(False, f"{case}: final {RELEASE_STATS_MARKER} missing lifecycle fields: {missing}")
        return None

    result: dict[str, int] = {}
    for name in REQUIRED_RELEASE_STATS_FIELDS:
        value = fields[name]
        if not _UINT_RE.fullmatch(value):
            check(False, f"{case}: final {RELEASE_STATS_MARKER} field {name} invalid: {value!r}")
            return None
        result[name] = int(value)
    return result


def check_env_isolation(envs: dict[str, dict[str, str]], check) -> None:
    """Stage 3B-1 v2 env isolation with calibrated thresholds."""
    labels = ["OFF", "FIXED", "DYNAMIC_NOOP", "DYNAMIC_RELEASE"]

    # OFF must not have any bounded-only keys
    off_env = envs["OFF"]
    for key in BOUNDED_ONLY_KEYS:
        if key in off_env:
            check(False, f"OFF env has bounded-release key: {key}")

    # All bounded variants must have LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1
    for label in ["FIXED", "DYNAMIC_NOOP", "DYNAMIC_RELEASE"]:
        check(envs[label].get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE") == "1",
              f"{label} must have LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1")

    # FIXED must NOT have DYNAMIC_TARGET
    check("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET" not in envs["FIXED"],
          "FIXED must not have DYNAMIC_TARGET")

    # DYNAMIC variants must have DYNAMIC_TARGET=1
    for label in ["DYNAMIC_NOOP", "DYNAMIC_RELEASE"]:
        check(envs[label].get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET") == "1",
              f"{label} must have DYNAMIC_TARGET=1")

    # Shared env keys must be identical across all four cases
    # (exclude bounded-only keys and calibrated keys)
    exclusive_keys = BOUNDED_ONLY_KEYS | CALIBRATED_KEYS
    shared_keys = set(off_env.keys())
    for label in labels[1:]:
        shared_keys &= set(envs[label].keys())
    shared_keys -= exclusive_keys

    for key in shared_keys:
        ref = envs["OFF"].get(key)
        for label in labels[1:]:
            val = envs[label].get(key)
            if val != ref:
                check(False,
                      f"Shared key '{key}' differs in {label}: OFF={ref!r} {label}={val!r}")

    # Calibrated keys: OFF and DYNAMIC_NOOP share SAFE values;
    # FIXED and DYNAMIC_RELEASE share TRIGGER values
    for key in CALIBRATED_KEYS:
        off_val = envs["OFF"].get(key)
        noop_val = envs["DYNAMIC_NOOP"].get(key)
        fixed_val = envs["FIXED"].get(key)
        dynrel_val = envs["DYNAMIC_RELEASE"].get(key)
        check(off_val == noop_val,
              f"Calibrated key '{key}' differs between OFF={off_val!r} and "
              f"DYNAMIC_NOOP={noop_val!r} (both must use SAFE thresholds)")
        check(fixed_val == dynrel_val,
              f"Calibrated key '{key}' differs between FIXED={fixed_val!r} and "
              f"DYNAMIC_RELEASE={dynrel_val!r} (both must use TRIGGER thresholds)")
        # SAFE must be > TRIGGER (by at least the noise margin from calibration)
        check(int(off_val) > int(fixed_val),
              f"Calibrated key '{key}' SAFE={off_val} must be > TRIGGER={fixed_val}")


def count_strace_madvise_bytes(strace_files: list[pathlib.Path]) -> tuple[int, list[str]]:
    total = 0
    addrs: list[str] = []
    for sf in strace_files:
        content = sf.read_text(errors="replace")
        for match in MADV_DONTNEED_RE.finditer(content):
            line = match.group(0)
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    total += int(parts[1].strip())
                except ValueError:
                    pass
                addr_part = line.split("(")[1] if "(" in line else ""
                addr = addr_part.split(",")[0].strip() if addr_part else ""
                if addr:
                    addrs.append(addr)
    return total, addrs


def _section_pass(label: str, section_errors_before: int, errors: list[str]) -> None:
    section_errors = len(errors) - section_errors_before
    if section_errors == 0:
        print(f"  PASS")
    else:
        print(f"  FAIL ({section_errors} check(s) failed)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse Stage 3B-1 v2 calibrated dynamic target artifact")
    ap.add_argument("artifact_dir", help="Path to the artifact directory")
    ap.add_argument("--result-path", help="Write parser-owned final result JSON here")
    ap.add_argument("--verify-result", help="Verify a previously saved parser result")
    args = ap.parse_args()

    art = pathlib.Path(args.artifact_dir)
    global ACTIVE_RESULT_PATH, ACTIVE_ARTIFACT
    ACTIVE_ARTIFACT = art
    ACTIVE_RESULT_PATH = pathlib.Path(args.result_path) if args.result_path else None
    if not art.is_dir():
        fail(f"artifact directory not found: {art}")

    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)
            print(f"FAIL: {msg}", file=sys.stderr)

    # =====================================================================
    # Artifact identity
    # =====================================================================
    manifest_path = art / "manifest.json"
    check(manifest_path.is_file(), "manifest.json missing")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("protocol", "protocol_version", "head_sha", "worktree_dirty",
                    "worktree_status", "capture_mode", "source_snapshot", "diff",
                    "diff_sha256", "binary", "model", "runner", "parser", "prompt",
                    "seed", "timestamp_utc", "completed_at_utc", "case_configs",
                    "calibration"):
            check(key in manifest, f"manifest missing {key}")
        check(manifest.get("protocol") == "kv_bounded_release_stage3b_1",
              "manifest protocol mismatch")
        check(manifest.get("protocol_version") == 2, "manifest protocol_version mismatch")
        check(isinstance(manifest.get("head_sha"), str) and
              re.fullmatch(r"[0-9a-f]{40}", manifest.get("head_sha", "")) is not None,
              "manifest head_sha is invalid")
        mode = manifest.get("capture_mode")
        check(mode in {"diagnostic_dirty", "archival_clean"}, "manifest capture_mode is invalid")
        check(isinstance(manifest.get("worktree_dirty"), bool),
              "manifest worktree_dirty must be boolean")
        if mode == "diagnostic_dirty":
            check(manifest.get("worktree_dirty") is True,
                  "diagnostic_dirty artifact must record a dirty worktree")
        if mode == "archival_clean":
            check(manifest.get("worktree_dirty") is False,
                  "archival_clean artifact must record a clean worktree")
        source_snapshot = manifest.get("source_snapshot")
        check(isinstance(source_snapshot, dict), "manifest source_snapshot is incomplete")
        if isinstance(source_snapshot, dict):
            check(source_snapshot.get("schema_version") == 1,
                  "source_snapshot schema_version mismatch")
            check(source_snapshot.get("head_sha") == manifest.get("head_sha"),
                  "source_snapshot HEAD disagrees with manifest")
            tracked = source_snapshot.get("tracked_diff")
            check(isinstance(tracked, dict), "source_snapshot tracked_diff is incomplete")
            expected_diff = art / "source_snapshot" / "tracked.diff"
            if isinstance(tracked, dict):
                tracked_path = pathlib.Path(tracked.get("path", ""))
                check(tracked_path == expected_diff,
                      "source_snapshot tracked_diff path is not canonical")
                check(tracked_path.is_file() and
                      tracked_path.stat().st_size == tracked.get("size") and
                      sha256_file(tracked_path) == tracked.get("sha256"),
                      "source_snapshot tracked_diff identity drift")
                check(manifest.get("diff") == tracked,
                      "manifest diff differs from source_snapshot tracked_diff")
            untracked = source_snapshot.get("untracked_files")
            check(isinstance(untracked, list), "source_snapshot untracked_files must be a list")
        diff_entry = manifest.get("diff")
        check(isinstance(diff_entry, dict), "manifest diff identity is incomplete")
        if isinstance(diff_entry, dict):
            diff_path = pathlib.Path(diff_entry.get("path", ""))
            check(diff_path.is_file() and
                  sha256_file(diff_path) == diff_entry.get("sha256"),
                  "manifest diff hash drift")
            check(diff_entry.get("sha256") == manifest.get("diff_sha256"),
                  "manifest diff_sha256 disagrees with diff")
        for name in ("binary", "model", "runner", "parser"):
            entry = manifest.get(name)
            check(isinstance(entry, dict) and isinstance(entry.get("path"), str) and
                  isinstance(entry.get("size"), int) and entry.get("size") >= 0 and
                  isinstance(entry.get("sha256"), str) and
                  re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is not None,
                  f"manifest {name} identity is incomplete")
        for name in ("binary", "model", "runner", "parser"):
            entry = manifest.get(name)
            if isinstance(entry, dict):
                item_path = pathlib.Path(entry.get("path", ""))
                check(item_path.is_file() and
                      item_path.stat().st_size == entry.get("size") and
                      sha256_file(item_path) == entry.get("sha256"),
                      f"{name} identity drift")
        check(manifest.get("parser") == parser_identity(),
              "manifest parser is not the executing parser")
        if args.verify_result:
            check_saved_result(pathlib.Path(args.verify_result), manifest_path)

    # =====================================================================
    # 0. RSS Calibration
    # =====================================================================
    print("--- RSS Calibration ---")
    e0 = len(errors)

    calib = manifest.get("calibration")
    check(isinstance(calib, dict), "manifest calibration missing or invalid")
    if isinstance(calib, dict):
        for key in ("rss_idle_kb", "rss_peak_kb", "delta_kb", "margin_kb",
                    "pressure_trigger_kb", "critical_trigger_kb",
                    "pressure_safe_kb", "critical_safe_kb",
                    "low_water_trigger_kb", "low_water_safe_kb"):
            check(isinstance(calib.get(key), int) and calib.get(key, 0) > 0,
                  f"calibration missing or invalid {key}={calib.get(key)}")
        check(calib.get("rss_peak_kb") >= calib.get("rss_idle_kb"),
              "calibration rss_peak < rss_idle")
        check(calib.get("delta_kb") >= 1024,
              "calibration delta too small")
        check(calib.get("pressure_trigger_kb") < calib.get("pressure_safe_kb"),
              "calibration pressure_trigger must be < pressure_safe")

        # Forbid hardcoded artificial thresholds: 1, 2, 3, 100M KiB
        hardcoded_artificial = {1, 2, 3, 100000000}
        threshold_fields = ("pressure_trigger_kb", "critical_trigger_kb",
                           "pressure_safe_kb", "critical_safe_kb",
                           "low_water_trigger_kb", "low_water_safe_kb")
        for key in threshold_fields:
            val = calib.get(key, 0)
            check(val not in hardcoded_artificial,
                  f"calibration {key}={val} is a hardcoded artificial threshold — "
                  f"must be derived from actual RSS measurement")

        # Derivation check: pressure_trigger must be between idle and peak
        check(calib["pressure_trigger_kb"] >= calib["rss_idle_kb"],
              f"pressure_trigger_kb={calib['pressure_trigger_kb']} < "
              f"rss_idle_kb={calib['rss_idle_kb']}")

        print(f"  idle={calib['rss_idle_kb']} KiB  peak={calib['rss_peak_kb']} KiB  "
              f"delta={calib['delta_kb']} KiB  margin={calib['margin_kb']} KiB")
        print(f"  trigger: pressure={calib['pressure_trigger_kb']} "
              f"critical={calib['critical_trigger_kb']}")
        print(f"  safe:    pressure={calib['pressure_safe_kb']} "
              f"critical={calib['critical_safe_kb']}")

    # Verify case envs use calibration values
    case_dirs = {
        "OFF": art / "bounded_off",
        "FIXED": art / "bounded_fixed",
        "DYNAMIC_NOOP": art / "bounded_dynamic_noop",
        "DYNAMIC_RELEASE": art / "bounded_dynamic_release",
    }
    for label in case_dirs:
        check(case_dirs[label].is_dir(), f"case directory missing: {label}")

    envs = {label: json.loads((case_dirs[label] / "environment.json").read_text())
            for label in case_dirs}

    # Each case's calibrated key values must match calibration
    if isinstance(calib, dict):
        for label in ["OFF", "DYNAMIC_NOOP"]:
            env = envs[label]
            pres = int(env.get("LLAMA_KV_PRESSURE_RSS_KB", "0"))
            crit = int(env.get("LLAMA_KV_CRITICAL_RSS_KB", "0"))
            check(pres == calib.get("pressure_safe_kb"),
                  f"{label} PRESSURE_RSS_KB={pres} != calibrate pressure_safe_kb={calib.get('pressure_safe_kb')}")
            check(crit == calib.get("critical_safe_kb"),
                  f"{label} CRITICAL_RSS_KB={crit} != calibrate critical_safe_kb={calib.get('critical_safe_kb')}")
        for label in ["FIXED", "DYNAMIC_RELEASE"]:
            env = envs[label]
            pres = int(env.get("LLAMA_KV_PRESSURE_RSS_KB", "0"))
            crit = int(env.get("LLAMA_KV_CRITICAL_RSS_KB", "0"))
            check(pres == calib.get("pressure_trigger_kb"),
                  f"{label} PRESSURE_RSS_KB={pres} != calibrate pressure_trigger_kb={calib.get('pressure_trigger_kb')}")
            check(crit == calib.get("critical_trigger_kb"),
                  f"{label} CRITICAL_RSS_KB={crit} != calibrate critical_trigger_kb={calib.get('critical_trigger_kb')}")

    _section_pass("RSS Calibration", e0, errors)

    # --- Load case data ---
    stderrs = {label: (case_dirs[label] / "server.stderr").read_text(errors="replace")
               for label in case_dirs}
    results = {label: json.loads((case_dirs[label] / "result.json").read_text())
               for label in case_dirs}

    # Verify manifest/case closure
    for label, case_dir in case_dirs.items():
        cfg = manifest.get("case_configs", {}).get(label) \
            if isinstance(manifest.get("case_configs"), dict) else None
        check(isinstance(cfg, dict), f"manifest missing case config for {label}")
        if isinstance(cfg, dict):
            check(cfg.get("environment") == envs[label],
                  f"manifest environment differs from {label} environment")
            execution = json.loads((case_dir / "execution.json").read_text())
            check(execution.get("environment") == envs[label],
                  f"{label}: execution.json environment differs from environment.json")
            check(cfg.get("execution") == execution,
                  f"manifest execution differs from {label} execution")

    # --- Summary integrity ---
    summary_path = art / "summary.json"
    if summary_path.exists():
        summary_data = json.loads(summary_path.read_text())
        runner_status = summary_data.get("runner_status", "")
        runner_verdict = summary_data.get("verdict", "")
        check(runner_verdict != "PASS",
              "summary.json contains verdict=PASS")
        check(runner_status == "run_complete",
              f"summary.json runner_status={runner_status!r}")
        check(summary_data.get("case_failures") == [],
              "summary.json case_failures must be empty")
        if runner_verdict and runner_verdict != "PASS":
            check(False,
                  f"summary.json contains verdict={runner_verdict!r}")
    else:
        check(False, "summary.json missing")

    # =====================================================================
    # 1. Configuration isolation
    # =====================================================================
    print("--- Configuration isolation ---")
    e0 = len(errors)
    check_env_isolation(envs, check)

    for label, env in envs.items():
        legacy = env.get("LLAMA_KV_PAGED_RELEASE", "0")
        check(legacy == "0" or legacy == "",
              f"{label}: LLAMA_KV_PAGED_RELEASE must be 0 or unset, got {legacy!r}")
        id_fp = env.get("LLAMA_KV_PAGED_IDENTITY_FAST_PATH", "1")
        check(id_fp == "0",
              f"{label}: LLAMA_KV_PAGED_IDENTITY_FAST_PATH must be 0, got {id_fp!r}")
        mc = env.get("LLAMA_KV_PAGED_MINCORE", "0")
        check(mc == "1", f"{label}: LLAMA_KV_PAGED_MINCORE must be 1, got {mc!r}")

    _section_pass("Config isolation", e0, errors)

    # =====================================================================
    # 1b. Server topology
    # =====================================================================
    print("--- Server topology ---")
    e0 = len(errors)
    for label, case_dir in case_dirs.items():
        exec_path = case_dir / "execution.json"
        check(exec_path.exists(), f"{label}: execution.json missing")
        if exec_path.exists():
            exec_data = json.loads(exec_path.read_text())
            argv = exec_data.get("argv", [])
            check("--parallel" in argv, f"{label}: --parallel not in argv")
            try:
                par_idx = argv.index("--parallel")
                check(int(argv[par_idx + 1]) == 1, f"{label}: --parallel must be 1")
            except (ValueError, IndexError) as e:
                check(False, f"{label}: --parallel: {e}")
            check("--cache-ram" in argv, f"{label}: --cache-ram not in argv")
            try:
                cr_idx = argv.index("--cache-ram")
                check(int(argv[cr_idx + 1]) == 0, f"{label}: --cache-ram must be 0")
            except (ValueError, IndexError) as e:
                check(False, f"{label}: --cache-ram: {e}")
            check("--no-warmup" in argv, f"{label}: --no-warmup not in argv")
            try:
                ctk = argv.index("--cache-type-k")
                check(argv[ctk + 1] == "f32", f"{label}: --cache-type-k must be f32")
            except (ValueError, IndexError):
                check(False, f"{label}: --cache-type-k not in argv")
            try:
                ctv = argv.index("--cache-type-v")
                check(argv[ctv + 1] == "f32", f"{label}: --cache-type-v must be f32")
            except (ValueError, IndexError):
                check(False, f"{label}: --cache-type-v not in argv")

    def _strip_strace(a: list[str]) -> list[str]:
        if not a or pathlib.Path(a[0]).name != "strace":
            return list(a)
        if "--" not in a:
            raise ProtocolError("strace argv missing --")
        return list(a[a.index("--") + 1:])

    def _normalize_port(a: list[str]) -> list[str]:
        out = list(a)
        for i, token in enumerate(out):
            if token == "--port" and i + 1 < len(out):
                out[i + 1] = "<PORT>"
        return out

    argv_ref = _normalize_port(_strip_strace(
        json.loads((case_dirs["OFF"] / "execution.json").read_text())["argv"]))
    for label in ["FIXED", "DYNAMIC_NOOP", "DYNAMIC_RELEASE"]:
        other = _normalize_port(_strip_strace(
            json.loads((case_dirs[label] / "execution.json").read_text())["argv"]))
        check(other == argv_ref, f"OFF/{label} argv must be identical")
    _section_pass("Server topology", e0, errors)

    # =====================================================================
    # 1c. Startup capability marker
    # =====================================================================
    print("--- Startup capability marker ---")
    e0 = len(errors)
    for label in ["FIXED", "DYNAMIC_NOOP", "DYNAMIC_RELEASE"]:
        stderr = stderrs[label]
        cap_match = re.search(
            r"kv_pressure_bounded_release_capability\s+"
            r"can_enable=(\d+)\s+paged=(\d+)\s+ingraph=(\d+)\s+"
            r"layers_supported=(\d+)\s+row_idx=(\d+)\s+"
            r"swap_disabled=(\d+)\s+layout_supported=(\d+)",
            stderr)
        check(cap_match is not None,
              f"{label}: startup capability marker not found in server stderr")
        if cap_match:
            fields = ("can_enable", "paged", "ingraph", "layers_supported",
                       "row_idx", "swap_disabled", "layout_supported")
            cap = {k: int(v) for k, v in zip(fields, cap_match.groups())}
            STARTUP_HARD = ("paged", "ingraph", "layers_supported",
                           "swap_disabled", "layout_supported")
            for k in STARTUP_HARD:
                check(cap.get(k, -1) == 1,
                      f"{label}: startup capability {k}={cap.get(k)}, expected 1")
            print(f"  {label}: capability ✓")
    _section_pass("Startup capability marker", e0, errors)

    # =====================================================================
    # 2. OFF case
    # =====================================================================
    print("--- OFF case ---")
    e0 = len(errors)
    off_bounded = find_markers(stderrs["OFF"], BOUNDED_RELEASE_RE, "OFF")
    check(len(off_bounded) == 0,
          f"OFF must have zero bounded markers, found {len(off_bounded)}")
    off_dry = find_markers(stderrs["OFF"], DRY_RUN_MARKER_RE, "OFF")
    check(len(off_dry) == 0, f"OFF must have zero dry_run markers, found {len(off_dry)}")
    off_telemetry = find_markers(stderrs["OFF"], TELEMETRY_MARKER_RE, "OFF")
    check(len(off_telemetry) >= 1,
          f"OFF must have telemetry markers, found {len(off_telemetry)}")
    destructive = DESTRUCTIVE_RELEASE_RE.findall(stderrs["OFF"])
    check(len(destructive) == 0,
          f"OFF must have zero destructive release evidence: {destructive[:5]!r}")
    check(results["OFF"].get("requests_count", 1) == 1,
          "OFF must have single request")
    print(f"  bounded: 0  dry_run: 0  telemetry: {len(off_telemetry)}")
    _section_pass("OFF case", e0, errors)

    # =====================================================================
    # 3. FIXED case (two requests)
    # =====================================================================
    print("--- FIXED case ---")
    e0 = len(errors)
    fixed_markers = find_markers(stderrs["FIXED"], BOUNDED_RELEASE_RE, "FIXED")
    check(len(fixed_markers) >= 1,
          f"FIXED must have at least 1 bounded marker, found {len(fixed_markers)}")

    for i, m in enumerate(fixed_markers):
        check(m.get("target_mode") == "fixed",
              f"FIXED marker[{i}] target_mode={m.get('target_mode')}, expected fixed")
        check(m.get("decision_reason") == "fixed",
              f"FIXED marker[{i}] decision_reason={m.get('decision_reason')}, expected fixed")

    fixed_execs = [m for m in fixed_markers
                   if m.get("skipped_reason") == "none"
                   and int(m.get("released_bytes", "0")) > 0]
    check(len(fixed_execs) == 1,
          f"FIXED: expected 1 execution, found {len(fixed_execs)}")

    for i, m in enumerate(fixed_markers):
        skipped = m.get("skipped_reason", "none")
        released_bytes = int(m.get("released_bytes", "0"))
        ownership_aborted = int(m.get("ownership_aborted", "0"))
        madvise_failures = int(m.get("madvise_failures", "0"))
        if skipped == "none" and released_bytes > 0:
            check(madvise_failures == 0,
                  f"FIXED execution[{i}] madvise_failures={madvise_failures}")
            check(ownership_aborted == 0,
                  f"FIXED execution[{i}] ownership_aborted=1")
            cnt_delta = int(m.get("bounded_cnt_bytes_delta", "0"))
            check(cnt_delta == released_bytes,
                  f"FIXED execution[{i}] cnt_delta={cnt_delta} != released={released_bytes}")
            mc_before = int(m.get("mincore_before_bytes", "0"))
            mc_after = int(m.get("mincore_after_bytes", "0"))
            check(mc_before > 0 and mc_after > 0,
                  f"FIXED execution[{i}] mincore must be >0")
            check(mc_after < mc_before,
                  f"FIXED execution[{i}] mincore must show physical drop")

    # Two requests
    check(results["FIXED"].get("requests_count", 1) == 2,
          "FIXED must have 2 requests (reuse verification)")

    print(f"  FIXED markers: {len(fixed_markers)}  executions: {len(fixed_execs)}")
    _section_pass("FIXED case", e0, errors)

    # =====================================================================
    # 4. DYNAMIC_NOOP case (NORMAL, zero bounded markers)
    # =====================================================================
    print("--- DYNAMIC_NOOP case ---")
    e0 = len(errors)
    noop_bounded = find_markers(stderrs["DYNAMIC_NOOP"], BOUNDED_RELEASE_RE, "DYNAMIC_NOOP")
    check(len(noop_bounded) == 0,
          f"DYNAMIC_NOOP must have zero bounded markers (NORMAL state, "
          f"bounded_release_due never fires), found {len(noop_bounded)}")

    noop_dry = find_markers(stderrs["DYNAMIC_NOOP"], DRY_RUN_MARKER_RE, "DYNAMIC_NOOP")
    check(len(noop_dry) == 0,
          f"DYNAMIC_NOOP must have zero dry_run markers, found {len(noop_dry)}")

    noop_destructive = DESTRUCTIVE_RELEASE_RE.findall(stderrs["DYNAMIC_NOOP"])
    check(len(noop_destructive) == 0,
          f"DYNAMIC_NOOP must have zero destructive release: {noop_destructive[:5]!r}")

    # Telemetry must show NORMAL state (thresholds above RSS)
    noop_telemetry = find_markers(stderrs["DYNAMIC_NOOP"], TELEMETRY_MARKER_RE, "DYNAMIC_NOOP")
    check(len(noop_telemetry) >= 1,
          f"DYNAMIC_NOOP must have telemetry markers, found {len(noop_telemetry)}")
    # All telemetry states must be NORMAL
    non_normal = [t for t in noop_telemetry if t.get("state") != "NORMAL"]
    check(len(non_normal) == 0,
          f"DYNAMIC_NOOP must have ALL telemetry states=NORMAL (thresholds above RSS), "
          f"found {len(non_normal)} non-NORMAL: "
          f"{[(t.get('state'), t.get('rss_kb')) for t in non_normal[:5]]}")

    check(results["DYNAMIC_NOOP"].get("requests_count", 1) == 1,
          "DYNAMIC_NOOP must have single request")

    # Must have bounded release + dynamic target configured
    noop_env = envs["DYNAMIC_NOOP"]
    check(noop_env.get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE") == "1",
          "DYNAMIC_NOOP must have BOUNDED_RELEASE=1")
    check(noop_env.get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE_DYNAMIC_TARGET") == "1",
          "DYNAMIC_NOOP must have DYNAMIC_TARGET=1")

    print(f"  bounded: 0  dry_run: 0  telemetry: {len(noop_telemetry)} "
          f"(all NORMAL) ✓")
    _section_pass("DYNAMIC_NOOP case", e0, errors)

    # =====================================================================
    # 5. DYNAMIC_RELEASE case (two requests)
    # =====================================================================
    print("--- DYNAMIC_RELEASE case ---")
    e0 = len(errors)
    dynrel_markers = find_markers(stderrs["DYNAMIC_RELEASE"], BOUNDED_RELEASE_RE, "DYNAMIC_RELEASE")
    check(len(dynrel_markers) >= 1,
          f"DYNAMIC_RELEASE must have at least 1 bounded marker, found {len(dynrel_markers)}")

    dynrel_execs = [m for m in dynrel_markers
                    if m.get("skipped_reason") == "none"
                    and int(m.get("released_bytes", "0")) > 0]
    check(len(dynrel_execs) == 1,
          f"DYNAMIC_RELEASE: expected 1 execution, found {len(dynrel_execs)}")

    valid_dynrel = dynrel_execs[0] if dynrel_execs else None

    for i, m in enumerate(dynrel_markers):
        check(m.get("target_mode") == "dynamic",
              f"DYNAMIC_RELEASE marker[{i}] target_mode must be dynamic")
        skipped = m.get("skipped_reason", "none")
        released_bytes = int(m.get("released_bytes", "0"))
        ownership_aborted = int(m.get("ownership_aborted", "0"))
        madvise_failures = int(m.get("madvise_failures", "0"))

        if skipped == "none" and released_bytes > 0:
            check(madvise_failures == 0,
                  f"DYNAMIC_RELEASE exec[{i}] madvise_failures={madvise_failures}")
            check(ownership_aborted == 0,
                  f"DYNAMIC_RELEASE exec[{i}] ownership_aborted=1")
            check(m.get("decision_reason") == "dynamic",
                  f"DYNAMIC_RELEASE exec[{i}] decision_reason must be dynamic")

            cnt_delta = int(m.get("bounded_cnt_bytes_delta", "0"))
            check(cnt_delta == released_bytes,
                  f"DYNAMIC_RELEASE exec[{i}] cnt_delta={cnt_delta} != released={released_bytes}")

            # Dynamic target formula
            target_bytes = int(m.get("target_bytes", "0"))
            water_excess = int(m.get("water_excess_bytes", "0"))
            max_release = int(m.get("max_release_bytes", "0"))
            kv_resident = int(m.get("kv_resident_bytes", "0"))
            kv_reclaimable = int(m.get("kv_reclaimable_resident_bytes", "0"))

            check(target_bytes > 0, f"DYNAMIC_RELEASE exec[{i}] target_bytes must be >0")
            check(water_excess > 0, f"DYNAMIC_RELEASE exec[{i}] water_excess must be >0")

            expected_target = min(water_excess, max_release, kv_resident, kv_reclaimable)
            check(expected_target > 0,
                  f"DYNAMIC_RELEASE exec[{i}] expected_target=0")
            check(target_bytes == expected_target,
                  f"DYNAMIC_RELEASE exec[{i}] target={target_bytes} != "
                  f"expected={expected_target} = min(we={water_excess}, "
                  f"max={max_release}, res={kv_resident}, recl={kv_reclaimable})")

            # Clamp check
            clamp = m.get("target_clamp", "none")
            if expected_target == kv_reclaimable and kv_reclaimable < kv_resident:
                check(clamp == "reclaimable",
                      f"DYNAMIC_RELEASE clamp should be reclaimable, got {clamp}")
            elif expected_target == max_release and max_release < water_excess:
                check(clamp == "max_release",
                      f"DYNAMIC_RELEASE clamp should be max_release, got {clamp}")

            mc_before = int(m.get("mincore_before_bytes", "0"))
            mc_after = int(m.get("mincore_after_bytes", "0"))
            check(mc_before > 0 and mc_after > 0,
                  f"DYNAMIC_RELEASE exec[{i}] mincore must be >0")
            check(mc_after < mc_before,
                  f"DYNAMIC_RELEASE exec[{i}] mincore must show physical drop")

            print(f"  exec: target={target_bytes} released={released_bytes} "
                  f"water_excess={water_excess} clamp={clamp} ✓")

    if valid_dynrel:
        check(valid_dynrel.get("kv_budget_valid") == "1",
              "DYNAMIC_RELEASE exec must have kv_budget_valid=1")
        check(valid_dynrel.get("kv_budget_ownership_aborted") == "0",
              "DYNAMIC_RELEASE exec must have kv_budget_ownership_aborted=0")
        check(valid_dynrel.get("pressure_basis_valid") == "1",
              "DYNAMIC_RELEASE exec must have pressure_basis_valid=1")

    # Two requests
    check(results["DYNAMIC_RELEASE"].get("requests_count", 1) == 2,
          "DYNAMIC_RELEASE must have 2 requests (reuse verification)")

    print(f"  DYNAMIC_RELEASE markers: {len(dynrel_markers)}  executions: {len(dynrel_execs)}")
    _section_pass("DYNAMIC_RELEASE case", e0, errors)

    # =====================================================================
    # 6. Final KV lifecycle counters (reuse_allocations, write_commits)
    # =====================================================================
    print("--- Final KV lifecycle counters ---")
    e0 = len(errors)
    lifecycle_stats = {}
    for label in case_dirs:
        lifecycle_stats[label] = extract_final_release_stats(label, stderrs[label], check)

    # OFF: zero bounded source stats, zero lifecycle errors
    off_ls = lifecycle_stats.get("OFF")
    if off_ls is not None:
        for name in LIFECYCLE_STATS_FIELDS:
            check(off_ls[name] == 0,
                  f"OFF: final lifecycle {name}={off_ls[name]}, must be 0")
        for name in BOUNDED_SOURCE_STATS_FIELDS:
            check(off_ls[name] == 0,
                  f"OFF: final bounded source {name}={off_ls[name]}, must be 0")

    # FIXED and DYNAMIC_RELEASE: positive reuse and commits
    for label in ["FIXED", "DYNAMIC_RELEASE"]:
        ls = lifecycle_stats.get(label)
        if ls is not None:
            check(ls["bounded_release_calls"] >= 1,
                  f"{label}: bounded_release_calls must be >=1")
            check(ls["bounded_release_bytes"] > 0,
                  f"{label}: bounded_release_bytes must be >0")
            # reuse_allocations > 0: released blocks were reused by req2
            check(ls["reuse_allocations"] > 0,
                  f"{label}: reuse_allocations={ls['reuse_allocations']} must be >0 "
                  f"(req2 must reuse blocks released after req1)")
            # write_commits > 0: write operations committed on reused blocks
            check(ls["write_commits"] > 0,
                  f"{label}: write_commits={ls['write_commits']} must be >0 "
                  f"(write mapping on reused blocks)")
            for name in ("write_rollbacks", "released_redirect_no_dummy",
                         "released_redirect_no_dummy_pending_write",
                         "ensure_pending_write_rejected", "input_setup_fatal",
                         "row_mapping_fatal", "write_mapping_fatal",
                         "active_nonresident_fatal"):
                check(ls[name] == 0,
                      f"{label}: final lifecycle {name}={ls[name]}, must be 0")

    # DYNAMIC_NOOP: zero bounded source counters, zero lifecycle
    noop_ls = lifecycle_stats.get("DYNAMIC_NOOP")
    if noop_ls is not None:
        for name in LIFECYCLE_STATS_FIELDS:
            check(noop_ls[name] == 0,
                  f"DYNAMIC_NOOP: final lifecycle {name}={noop_ls[name]}, must be 0")
        for name in BOUNDED_SOURCE_STATS_FIELDS:
            check(noop_ls[name] == 0,
                  f"DYNAMIC_NOOP: final bounded source {name}={noop_ls[name]}, must be 0")

    for label, ls in lifecycle_stats.items():
        if ls is not None:
            print("  " + label + ": " + " ".join(
                f"{name}={ls[name]}" for name in
                ["bounded_release_calls", "bounded_release_bytes", "reuse_allocations",
                 "write_commits", "write_rollbacks"]))
    _section_pass("Final KV lifecycle counters", e0, errors)

    # =====================================================================
    # 7. Response byte-identity
    # =====================================================================
    print("--- Response identity ---")
    e0 = len(errors)
    off_text = results["OFF"].get("response_text_1", "")
    check(len(off_text) > 0, "OFF response must be non-empty")

    for label in ["FIXED", "DYNAMIC_NOOP", "DYNAMIC_RELEASE"]:
        r = results[label]
        text1 = r.get("response_text_1", "")
        check(text1 == off_text,
              f"OFF/{label} req1 responses must match\n"
              f"  OFF: {off_text[:60]!r}\n  {label}: {text1[:60]!r}")
        if r.get("requests_count", 1) >= 2:
            text2 = r.get("response_text_2", "")
            check(text2 == off_text,
                  f"OFF/{label} req2 responses must match\n"
                  f"  OFF: {off_text[:60]!r}\n  {label}: {text2[:60]!r}")
        check(r.get("http_status_1") == 200,
              f"{label} req1 HTTP status must be 200")
        if r.get("requests_count", 1) >= 2:
            check(r.get("http_status_2") == 200,
                  f"{label} req2 HTTP status must be 200")

    # Also check against calibration response
    if isinstance(calib, dict):
        calib_text = calib.get("calibration_response_text", "")
        if calib_text:
            check(calib_text == off_text,
                  f"calibration response must match OFF")
    _section_pass("Response identity", e0, errors)

    # =====================================================================
    # 8. Strace audit
    # =====================================================================
    print("--- Strace audit ---")
    e0 = len(errors)
    for label, case_dir in case_dirs.items():
        strace_files = sorted(case_dir.glob("strace.log"))
        check(len(strace_files) > 0,
              f"{label}: strace.log missing")
        if strace_files:
            madvise_total, _ = count_strace_madvise_bytes(strace_files)
            print(f"  {label}: {madvise_total} bytes MADV_DONTNEED")

    for label in ["FIXED", "DYNAMIC_RELEASE"]:
        markers = find_markers(stderrs[label], BOUNDED_RELEASE_RE, label)
        execs = [m for m in markers
                 if m.get("skipped_reason") == "none"
                 and int(m.get("released_bytes", "0")) > 0]
        if execs:
            cnt_delta = int(execs[0].get("bounded_cnt_bytes_delta", "0"))
            strace_files = sorted(case_dirs[label].glob("strace.log"))
            madvise_total, _ = count_strace_madvise_bytes(strace_files)
            check(madvise_total >= cnt_delta,
                  f"{label}: strace MADV_DONTNEED ({madvise_total}) < cnt_delta ({cnt_delta})")
    _section_pass("Strace audit", e0, errors)

    # =====================================================================
    # 9. KV mincore observational
    # =====================================================================
    print("--- KV mincore ---")
    e0 = len(errors)
    for label in ["FIXED", "DYNAMIC_RELEASE"]:
        markers = find_markers(stderrs[label], BOUNDED_RELEASE_RE, label)
        execs = [m for m in markers
                 if m.get("skipped_reason") == "none"
                 and int(m.get("released_bytes", "0")) > 0]
        if execs:
            m = execs[0]
            mc_before = int(m["mincore_before_bytes"])
            mc_after = int(m["mincore_after_bytes"])
            check(mc_before > 0 and mc_after > 0,
                  f"{label}: mincore must be >0")
            check(mc_after < mc_before,
                  f"{label}: mincore must show drop: before={mc_before} after={mc_after}")
            print(f"  {label}: before={mc_before} after={mc_after} drop={mc_before - mc_after}")
    _section_pass("KV mincore", e0, errors)

    # =====================================================================
    # 10. Process cleanup
    # =====================================================================
    print("--- Process cleanup ---")
    e0 = len(errors)
    for label, case_dir in case_dirs.items():
        phases_path = case_dir / "phases.json"
        check(phases_path.is_file(), f"{label}: phases.json missing")
        if phases_path.is_file():
            phases = json.loads(phases_path.read_text())
            check(isinstance(phases, dict), f"{label}: phases.json must be object")
            shutdown = phases.get("shutdown", {})
            check(isinstance(shutdown, dict), f"{label}: shutdown missing")
            if isinstance(shutdown, dict):
                required = {"pgid", "exit_code", "pgid_check_complete",
                            "cleanup_kill_attempted", "residual_process"}
                check(required <= set(shutdown),
                      f"{label}: shutdown missing fields: {sorted(required - set(shutdown))}")
                check(isinstance(shutdown.get("pgid"), int) and shutdown.get("pgid", 0) > 0,
                      f"{label}: shutdown PGID invalid")
                check(shutdown.get("pgid_check_complete") is True,
                      f"{label}: shutdown PGID check incomplete")
                check(shutdown.get("residual_process") is False,
                      f"{label}: residual process after shutdown")
    _section_pass("Process cleanup", e0, errors)

    # =====================================================================
    # Final verdict
    # =====================================================================
    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        fail(f"{len(errors)} parser check(s) failed")

    print("\nPASS: Stage 3B-1 v2 calibrated dynamic target with reuse verification")
    print(f"  artifact: {art}")
    off_t = results["OFF"].get("response_text_1", "")
    all_ok = True
    for label in ["FIXED", "DYNAMIC_NOOP", "DYNAMIC_RELEASE"]:
        if results[label].get("response_text_1", "") != off_t:
            all_ok = False
        if results[label].get("requests_count", 1) >= 2:
            if results[label].get("response_text_2", "") != off_t:
                all_ok = False
    print(f"  response match: {'YES' if all_ok else 'NO'}")
    for label in ["FIXED", "DYNAMIC_RELEASE"]:
        ls = lifecycle_stats.get(label)
        if ls:
            print(f"  {label}: reuse_allocations={ls['reuse_allocations']} "
                  f"write_commits={ls['write_commits']}")
    write_parser_result(0, "PASS")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        write_parser_result(code, "FAIL")
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ProtocolError) as exc:
        print(f"FAIL: protocol input invalid: {exc}", file=sys.stderr)
        write_parser_result(2, "FAIL")
        raise SystemExit(2)
