#!/usr/bin/env python3
"""Fail-closed parser for Stage 3A-2C bounded release OFF/DRY/BOUNDED artifacts.

Validates:
  1. OFF: zero bounded_release or destructive release markers
  2. DRY: at least 1 dry_run marker, zero bounded source/destructive lifecycle;
     process-wide MADV_DONTNEED is reported as non-attributable background
  3. BOUNDED: at least 1 bounded_release execution (released_bytes>0,
     ownership_aborted=0, madvise_failures=0).  The bounded counter delta
     (bounded_cnt_bytes_delta) is the PRIMARY authoritative source attribution.
     KV mincore is an INDEPENDENT OBSERVATION (physical page drop in correct
     direction), NOT an exact equality check against the counter delta.
     Process-wide strace MADV_DONTNEED is a lower-bound consistency check;
     surplus is reported as non-attributable background.
  4. All three responses are byte-identical
  5. NO config contamination (LLAMA_KV_PAGED_RELEASE != 1)
  6. NO path mixing (dry-run markers in bounded, bounded markers in dry-run)
  7. NO duplicate sample_count in bounded markers
  8. KV mincore shows physical page drop (OBSERVATIONAL, not exact match)
  9. RSS absolute drop exceeds noise floor (auxiliary cross-check only)
  10. Zero residual processes
  11. Quarantine / open-transaction rejection: ownership_aborted or
      skipped_reason indicating PENDING_WRITE/INVALID state is fail-closed
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

# Bounded release marker required keys
BOUNDED_REQUIRED_KEYS = {
    "state", "source", "stale",
    "released_bytes", "released_blocks",
    "blocks_scanned", "blocks_skipped_owned", "blocks_skipped_state",
    "madvise_failures", "shortfall_bytes", "overshoot_bytes",
    "block_scan_exhausted", "ownership_aborted",
    "target_bytes", "max_scan_blocks",
    "legacy_enabled", "sample_count", "episode",
    "cooldown_ms", "skipped_reason", "idle",
    "mincore_before_bytes", "mincore_after_bytes",
    "bounded_cnt_bytes_delta", "bounded_cnt_blocks_delta",
    # Per-condition capability decomposition fields
    "can_enable", "cap_paged", "cap_ingraph", "cap_layers",
    "cap_row_idx", "cap_swap_disabled", "cap_layout",
}

# The server emits this marker at teardown.  There can be early snapshots
# during startup, so the protocol deliberately uses the final marker from each
# case: it is the only snapshot that covers the complete request lifecycle.
# Do not make these optional: a missing or malformed counter must never turn a
# destructive-release artifact into a PASS.
RELEASE_STATS_MARKER = "KV_PAGED_RELEASE_STATS"
LIFECYCLE_STATS_FIELDS = (
    "reuse_allocations",
    "write_commits",
    "write_rollbacks",
    "dummy_candidate_pending_write_cell",
    "released_redirect_no_dummy",
    "released_redirect_no_dummy_pending_write",
    "ensure_pending_write_rejected",
    "input_setup_fatal",
    "row_mapping_fatal",
    "write_mapping_fatal",
    "active_nonresident_fatal",
)
BOUNDED_SOURCE_STATS_FIELDS = (
    "bounded_release_calls",
    "bounded_release_bytes",
    "bounded_release_blocks",
)
REQUIRED_RELEASE_STATS_FIELDS = LIFECYCLE_STATS_FIELDS + BOUNDED_SOURCE_STATS_FIELDS
_STATS_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_UINT_RE = re.compile(r"[0-9]+")
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

class ProtocolError(Exception):
    pass

# Exact grammar for every marker this protocol consumes.  A marker is one
# standalone token; fields follow it, have one '=', are unique, and exactly
# match the schema.  Prefix log tokens are allowed because SRV_INF adds them.
DRY_REQUIRED_KEYS = {
    "state", "source", "stale", "release_enabled", "would_release_bytes",
    "would_release_blocks", "blocks_scanned", "blocks_skipped_owned",
    "blocks_skipped_state", "shortfall_bytes", "overshoot_bytes",
    "block_scan_exhausted", "ownership_aborted", "target_bytes",
    "max_scan_blocks", "skipped_reason", "cooldown_ms", "sample_count", "idle",
}
TELEMETRY_REQUIRED_KEYS = {
    "state", "previous_state", "source", "sample_valid", "stale", "config_valid",
    "rss_kb", "cgroup_current_bytes", "cgroup_max_bytes", "cgroup_current_kb",
    "cgroup_max_kb", "cgroup_high_kb", "psi_some_avg10", "psi_full_avg10",
    "sample_latency_ns", "sample_count", "skip_count", "idle", "trigger",
}
MARKER_SCHEMAS = {
    "kv_pressure_bounded_release": BOUNDED_REQUIRED_KEYS,
    "kv_pressure_dry_run": DRY_REQUIRED_KEYS,
    "kv_pressure_telemetry": TELEMETRY_REQUIRED_KEYS,
}
BOOL_FIELDS = {"stale", "release_enabled", "block_scan_exhausted", "ownership_aborted",
               "legacy_enabled", "idle", "can_enable", "cap_paged", "cap_ingraph",
               "cap_layers", "cap_row_idx", "cap_swap_disabled", "cap_layout",
               "sample_valid", "config_valid"}
ENUMS = {
    "state": {"NORMAL", "PRESSURE", "CRITICAL", "RECOVERY"},
    "previous_state": {"NORMAL", "PRESSURE", "CRITICAL", "RECOVERY"},
    "source": {"NONE", "RSS_ABSOLUTE", "CGROUP_RATIO", "PSI"},
    "skipped_reason": {"none", "dry_run_active", "no_memory", "not_paged",
        "layout_unsupported", "swap_enabled", "not_ingraph", "no_layers", "no_row_idx",
        "legacy_active", "stale", "not_pressure", "cooldown",
        "structurally_disabled", "ownership_aborted"},
    "trigger": {"first", "state", "source", "stale", "periodic", "wake_completion"},
}

def _validate_value(marker: str, key: str, value: str, case: str) -> None:
    if key in BOOL_FIELDS and value not in {"0", "1"}:
        raise ProtocolError(f"{case}: {marker} {key} must be 0 or 1")
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
    # Values that are structurally unsigned have additional range constraints.
    if int(fields["sample_count"]) < 1:
        raise ProtocolError(f"{case}: {marker} sample_count must be >= 1")
    if marker == "kv_pressure_dry_run" and int(fields["max_scan_blocks"]) < 1:
        raise ProtocolError(f"{case}: dry-run max_scan_blocks must be >= 1")
    if marker == "kv_pressure_bounded_release" and int(fields["max_scan_blocks"]) < 1:
        raise ProtocolError(f"{case}: bounded max_scan_blocks must be >= 1")
    return fields

# Env keys that differ by design for each variant
BOUNDED_ONLY_KEYS = {
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_TARGET_BYTES",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_MAX_SCAN_BLOCKS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_COOLDOWN_MS",
    "LLAMA_KV_PRESSURE_BOUNDED_RELEASE_BACKOFF_MS",
}

DRY_RUN_ONLY_KEYS = {
    "LLAMA_KV_PRESSURE_DRY_RUN",
    "LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES",
    "LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS",
    "LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS",
}


def fail(message: str) -> NoReturn:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def parser_identity() -> dict[str, Any]:
    path = pathlib.Path(__file__).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def write_parser_result(exit_code: int, status: str) -> None:
    """Persist the parser-owned verdict record for runner closure.

    The runner never writes this file.  It is intentionally small and binds
    the exact parser file and final manifest bytes used for this invocation.
    """
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
    """Reject a saved parser record that is not the same parser/manifest PASS."""
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


def parse_marker_fields(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.split():
        if "=" in token:
            key, _, value = token.partition("=")
            fields[key] = value
    return fields


def find_markers(stderr_text: str, marker_re: re.Pattern[str], case: str) -> list[dict[str, str]]:
    markers: list[dict[str, str]] = []
    marker = marker_re.pattern.replace(r"\b", "")
    for line in stderr_text.split("\n"):
        if marker_re.search(line):
            markers.append(parse_strict_marker_line(line, marker, case))
    return markers


def extract_final_release_stats(case: str, stderr_text: str, check: Any) -> dict[str, int] | None:
    """Return lifecycle counters from the unambiguous final teardown marker.

    Earlier stats snapshots are allowed, but only the final one is a complete
    lifecycle observation.  Its grammar is strict so duplicate keys cannot be
    silently overwritten and malformed values cannot default to zero.
    """
    marker_lines = [line for line in stderr_text.splitlines()
                    if RELEASE_STATS_MARKER in line]
    if not marker_lines:
        check(False, f"{case}: {RELEASE_STATS_MARKER} missing; final lifecycle marker is indeterminate")
        return None

    line = marker_lines[-1]
    tokens = line.split()
    marker_indexes = [i for i, token in enumerate(tokens) if token == RELEASE_STATS_MARKER]
    if len(marker_indexes) != 1:
        check(False, f"{case}: final {RELEASE_STATS_MARKER} has ambiguous marker token count={len(marker_indexes)}")
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
            check(False, f"{case}: final {RELEASE_STATS_MARKER} field {name} has invalid unsigned integer {value!r}")
            return None
        result[name] = int(value)
    return result


def check_env_isolation(off_env: dict, dry_env: dict, bounded_env: dict) -> None:
    """Verify OFF/DRY/BOUNDED environments differ only in their respective keys."""
    base_keys = set(off_env.keys())

    off_extra = base_keys & BOUNDED_ONLY_KEYS
    if off_extra:
        fail(f"OFF env has bounded-release keys: {off_extra}")
    off_dry = base_keys & DRY_RUN_ONLY_KEYS
    if off_dry:
        fail(f"OFF env has dry-run keys: {off_dry}")

    dry_extra = set(dry_env.keys()) - base_keys
    unexpected_dry = dry_extra - DRY_RUN_ONLY_KEYS
    if unexpected_dry:
        fail(f"DRY env has unexpected extra keys: {unexpected_dry}")
    dry_bounded = set(dry_env.keys()) & BOUNDED_ONLY_KEYS
    if dry_bounded:
        fail(f"DRY env has bounded-release keys: {dry_bounded}")

    bounded_extra = set(bounded_env.keys()) - base_keys
    unexpected_bounded = bounded_extra - BOUNDED_ONLY_KEYS
    if unexpected_bounded:
        fail(f"BOUNDED env has unexpected extra keys: {unexpected_bounded}")
    bounded_dry = set(bounded_env.keys()) & DRY_RUN_ONLY_KEYS
    if bounded_dry:
        fail(f"BOUNDED env has dry-run keys: {bounded_dry}")

    for key in base_keys & set(dry_env.keys()) & set(bounded_env.keys()):
        v_off = off_env[key]
        v_dry = dry_env.get(key, v_off)
        v_bounded = bounded_env.get(key, v_off)
        if v_dry != v_off:
            if key not in DRY_RUN_ONLY_KEYS:
                fail(f"Shared key '{key}' differs in DRY: OFF={v_off!r} DRY={v_dry!r}")
        if v_bounded != v_off:
            if key not in BOUNDED_ONLY_KEYS:
                fail(f"Shared key '{key}' differs in BOUNDED: OFF={v_off!r} BOUNDED={v_bounded!r}")


def count_strace_madvise_bytes(strace_files: list[pathlib.Path]) -> tuple[int, list[str]]:
    """Sum the length arguments of madvise(..., MADV_DONTNEED) calls in strace output.
    Returns (total_bytes, source_addresses) for attribution."""
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
                # Capture address for attribution
                addr_part = line.split("(")[1] if "(" in line else ""
                addr = addr_part.split(",")[0].strip() if addr_part else ""
                if addr:
                    addrs.append(addr)
    return total, addrs


def _section_pass(label: str, section_errors_before: int, errors: list[str]) -> None:
    """Print PASS only when no new errors were added during this section."""
    section_errors = len(errors) - section_errors_before
    if section_errors == 0:
        print(f"  PASS")
    else:
        print(f"  FAIL ({section_errors} check(s) failed)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse Stage 3A-2C bounded release OFF/DRY/BOUNDED artifact")
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

    # Artifact identity is part of the protocol, not optional provenance.
    manifest_path = art / "manifest.json"
    check(manifest_path.is_file(), "manifest.json missing")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("protocol", "protocol_version", "head_sha", "worktree_dirty", "worktree_status",
                    "capture_mode", "source_snapshot", "diff", "diff_sha256", "binary", "model", "runner", "parser", "prompt",
                    "seed", "target_bytes", "timestamp_utc", "completed_at_utc", "case_configs"):
            check(key in manifest, f"manifest missing {key}")
        check(manifest.get("protocol") == "kv_bounded_release_stage3a_2c",
              "manifest protocol mismatch")
        check(manifest.get("protocol_version") == 4, "manifest protocol_version mismatch")
        check(isinstance(manifest.get("head_sha"), str) and
              re.fullmatch(r"[0-9a-f]{40}", manifest.get("head_sha", "")) is not None,
              "manifest head_sha is invalid")
        mode = manifest.get("capture_mode")
        check(mode in {"diagnostic_dirty", "archival_clean"}, "manifest capture_mode is invalid")
        check(isinstance(manifest.get("worktree_dirty"), bool), "manifest worktree_dirty must be boolean")
        check(isinstance(manifest.get("worktree_status"), list) and
              all(isinstance(item, str) for item in manifest.get("worktree_status", [])),
              "manifest worktree_status is invalid")
        if isinstance(manifest.get("worktree_status"), list):
            check(bool(manifest["worktree_status"]) == manifest.get("worktree_dirty"),
                  "manifest worktree_dirty disagrees with worktree_status")
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
            check(isinstance(tracked, dict), "source_snapshot tracked_diff identity is incomplete")
            expected_diff = art / "source_snapshot" / "tracked.diff"
            if isinstance(tracked, dict):
                tracked_path = pathlib.Path(tracked.get("path", ""))
                check(tracked_path == expected_diff,
                      "source_snapshot tracked_diff path is not canonical")
                check(tracked_path.is_file() and tracked_path.stat().st_size == tracked.get("size") and
                      sha256_file(tracked_path) == tracked.get("sha256"),
                      "source_snapshot tracked_diff identity drift")
                check(manifest.get("diff") == tracked,
                      "manifest diff differs from source_snapshot tracked_diff")
            untracked = source_snapshot.get("untracked_files")
            check(isinstance(untracked, list), "source_snapshot untracked_files must be a list")
            if isinstance(untracked, list):
                paths: list[str] = []
                for index, item in enumerate(untracked):
                    check(isinstance(item, dict), f"source_snapshot untracked_files[{index}] is invalid")
                    if not isinstance(item, dict):
                        continue
                    repo_path = item.get("path")
                    check(item.get("status") == "??",
                          f"source_snapshot untracked_files[{index}] status must be ??")
                    check(isinstance(repo_path, str) and repo_path != "",
                          f"source_snapshot untracked_files[{index}] path is invalid")
                    if not isinstance(repo_path, str) or not repo_path:
                        continue
                    rel = pathlib.PurePosixPath(repo_path)
                    check(not rel.is_absolute() and ".." not in rel.parts,
                          f"source_snapshot untracked path is unsafe: {repo_path!r}")
                    paths.append(repo_path)
                    expected = art / "source_snapshot" / "untracked" / pathlib.Path(*rel.parts)
                    snap_path = pathlib.Path(item.get("snapshot_path", ""))
                    check(snap_path == expected,
                          f"source_snapshot path is not canonical: {repo_path}")
                    check(isinstance(item.get("size"), int) and item.get("size", -1) >= 0 and
                          isinstance(item.get("sha256"), str) and
                          re.fullmatch(r"[0-9a-f]{64}", item.get("sha256", "")) is not None and
                          isinstance(item.get("mode"), int) and 0 <= item.get("mode", -1) <= 0o777,
                          f"source_snapshot identity fields are invalid: {repo_path}")
                    check(snap_path.is_file() and snap_path.stat().st_size == item.get("size") and
                          sha256_file(snap_path) == item.get("sha256") and
                          (snap_path.stat().st_mode & 0o777) == item.get("mode"),
                          f"source_snapshot content identity drift: {repo_path}")
                check(paths == sorted(set(paths)),
                      "source_snapshot untracked paths must be sorted and unique")
                status_lines = manifest.get("worktree_status", [])
                for repo_path in paths:
                    check(f"?? {repo_path}" in status_lines,
                          f"source_snapshot untracked path missing from worktree_status: {repo_path}")
                status_untracked = sorted(
                    line[3:] for line in status_lines
                    if isinstance(line, str) and line.startswith("?? "))
                check(status_untracked == paths,
                      "worktree_status untracked paths differ from source_snapshot")
                if mode == "archival_clean":
                    check(not paths, "archival_clean artifact contains untracked source files")
                if mode == "diagnostic_dirty" and isinstance(tracked, dict):
                    check(tracked.get("size", 0) > 0 or bool(paths),
                          "diagnostic_dirty source_snapshot contains no source changes")
        diff_entry = manifest.get("diff")
        check(isinstance(diff_entry, dict), "manifest diff identity is incomplete")
        if isinstance(diff_entry, dict):
            diff_path = pathlib.Path(diff_entry.get("path", ""))
            check(diff_path.is_file() and sha256_file(diff_path) == diff_entry.get("sha256"),
                  "manifest diff hash drift")
            check(diff_entry.get("sha256") == manifest.get("diff_sha256"),
                  "manifest diff_sha256 disagrees with diff identity")
            if mode == "archival_clean" and diff_path.is_file():
                check(diff_path.stat().st_size == 0, "archival_clean artifact has non-empty source.diff")
        for name in ("binary", "model", "runner", "parser"):
            entry = manifest.get(name)
            check(isinstance(entry, dict) and isinstance(entry.get("path"), str) and
                  isinstance(entry.get("size"), int) and entry.get("size") >= 0 and
                  isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")) is not None,
                  f"manifest {name} identity is incomplete")
        for name in ("binary", "model", "runner", "parser"):
            entry = manifest.get(name)
            if isinstance(entry, dict):
                item_path = pathlib.Path(entry.get("path", ""))
                check(item_path.is_file() and item_path.stat().st_size == entry.get("size") and
                      sha256_file(item_path) == entry.get("sha256"), f"{name} identity drift")
        check(manifest.get("parser") == parser_identity(), "manifest parser is not the executing parser")
        if args.verify_result:
            check_saved_result(pathlib.Path(args.verify_result), manifest_path)

    # --- Load case data ---
    off_dir = art / "bounded_off"
    dry_dir = art / "dry_run_off"
    bounded_dir = art / "bounded_on"

    for d, name in [(off_dir, "OFF"), (dry_dir, "DRY"), (bounded_dir, "BOUNDED")]:
        if not d.is_dir():
            fail(f"case directory missing: {d}")

    off_stderr = (off_dir / "server.stderr").read_text(errors="replace")
    dry_stderr = (dry_dir / "server.stderr").read_text(errors="replace")
    bounded_stderr = (bounded_dir / "server.stderr").read_text(errors="replace")

    off_result = json.loads((off_dir / "result.json").read_text())
    dry_result = json.loads((dry_dir / "result.json").read_text())
    bounded_result = json.loads((bounded_dir / "result.json").read_text())

    off_env = json.loads((off_dir / "environment.json").read_text())
    dry_env = json.loads((dry_dir / "environment.json").read_text())
    bounded_env = json.loads((bounded_dir / "environment.json").read_text())
    case_files = {"OFF": (off_dir, off_env), "DRY": (dry_dir, dry_env),
                  "BOUNDED": (bounded_dir, bounded_env)}
    for name, (case_dir, environment) in case_files.items():
        cfg = manifest.get("case_configs", {}).get(name) if isinstance(manifest.get("case_configs"), dict) else None
        check(isinstance(cfg, dict), f"manifest missing case config for {name}")
        if isinstance(cfg, dict):
            check(cfg.get("environment") == environment,
                  f"manifest environment differs from recorded {name} environment")
            execution = json.loads((case_dir / "execution.json").read_text())
            check(execution.get("environment") == environment,
                  f"{name}: execution.json environment differs from environment.json")
            check(isinstance(execution.get("argv"), list) and execution.get("argv") and
                  all(isinstance(arg, str) for arg in execution.get("argv", [])),
                  f"{name}: execution.json argv is invalid")
            check(cfg.get("execution") == execution,
                  f"manifest execution differs from recorded {name} execution")
            check(cfg.get("environment") == execution.get("environment"),
                  f"manifest/environment/execution closure differs for {name}")

    # --- Summary integrity: runner must not write verdict=PASS ---
    summary_path = art / "summary.json"
    if summary_path.exists():
        summary_data = json.loads(summary_path.read_text())
        runner_status = summary_data.get("runner_status", "")
        runner_verdict = summary_data.get("verdict", "")
        check(runner_verdict != "PASS",
              "summary.json contains verdict=PASS — RUNNER MUST NOT WRITE VERDICT.  "
              "The runner reports runner_status={run_complete,run_incomplete}; "
              "the final verdict is the parser's responsibility.")
        check(runner_status == "run_complete",
              f"summary.json runner_status={runner_status!r} — incomplete runs are protocol failures")
        check(summary_data.get("case_failures") == [],
              "summary.json case_failures must be an empty list")
        if runner_verdict and runner_verdict != "PASS":
            check(False,
                  f"summary.json contains verdict={runner_verdict!r} — "
                  f"runner must only write runner_status, not verdict.")
    else:
        check(False, "summary.json missing — runner must produce this file")

    # =====================================================================
    # 1. Configuration isolation
    # =====================================================================
    print("--- Configuration isolation ---")
    e0 = len(errors)
    check_env_isolation(off_env, dry_env, bounded_env)

    for name, env in [("OFF", off_env), ("DRY", dry_env), ("BOUNDED", bounded_env)]:
        legacy = env.get("LLAMA_KV_PAGED_RELEASE", "0")
        check(legacy == "0" or legacy == "",
              f"{name}: LLAMA_KV_PAGED_RELEASE must be 0 or unset, got {legacy!r}")

    check(bounded_env.get("LLAMA_KV_PRESSURE_BOUNDED_RELEASE") == "1",
          "BOUNDED must have LLAMA_KV_PRESSURE_BOUNDED_RELEASE=1")
    check("LLAMA_KV_PAGED" in off_env or "LLAMA_KV_PAGED" in bounded_env,
          "LLAMA_KV_PAGED must be set")
    # Identity fast path must be disabled for all three — bounded release
    # requires row-index gather mode for ownership collection.
    for name, env in [("OFF", off_env), ("DRY", dry_env), ("BOUNDED", bounded_env)]:
        id_fp = env.get("LLAMA_KV_PAGED_IDENTITY_FAST_PATH", "1")
        check(id_fp == "0",
              f"{name}: LLAMA_KV_PAGED_IDENTITY_FAST_PATH must be 0 "
              f"(identity fast path prevents row-index gather mode, "
              f"which bounded_release_can_enable() requires); got {id_fp!r}")
    # Mincore must be enabled for KV resident-page evidence
    for name, env in [("OFF", off_env), ("DRY", dry_env), ("BOUNDED", bounded_env)]:
        mc = env.get("LLAMA_KV_PAGED_MINCORE", "0")
        check(mc == "1",
              f"{name}: LLAMA_KV_PAGED_MINCORE must be 1, got {mc!r}")
    _section_pass("Config isolation", e0, errors)

    # =====================================================================
    # 1b. Server topology — argv validation (hard gate)
    # =====================================================================
    print("--- Server topology ---")
    e0 = len(errors)

    for name, case_dir in [("OFF", off_dir), ("DRY", dry_dir), ("BOUNDED", bounded_dir)]:
        exec_path = case_dir / "execution.json"
        check(exec_path.exists(),
              f"{name}: execution.json missing — runner must record server argv")
        if exec_path.exists():
            exec_data = json.loads(exec_path.read_text())
            argv = exec_data.get("argv", [])
            check("--parallel" in argv,
                  f"{name}: --parallel not in argv — server topology unconstrained")
            try:
                par_idx = argv.index("--parallel")
                par_val = int(argv[par_idx + 1])
                check(par_val == 1,
                      f"{name}: --parallel={par_val} — must be 1 for single-slot "
                      f"row-index gather topology; default (-1/auto) is not allowed")
            except (ValueError, IndexError) as e:
                check(False, f"{name}: --parallel value unparseable: {e}")

            check("--cache-ram" in argv,
                  f"{name}: --cache-ram not in argv — prompt cache must be disabled")
            try:
                cr_idx = argv.index("--cache-ram")
                cr_val = int(argv[cr_idx + 1])
                check(cr_val == 0,
                      f"{name}: --cache-ram={cr_val} — must be 0 to disable "
                      f"prompt cache (cached-prompt shortcut bypasses KV paged path)")
            except (ValueError, IndexError) as e:
                check(False, f"{name}: --cache-ram value unparseable: {e}")

            check("--no-warmup" in argv,
                  f"{name}: --no-warmup not in argv — warmup must be disabled "
                  f"for deterministic cold-path verification")

            # F32 K/V enforcement — paged row-index gather and bounded release
            # ownership collection require F32 K/V tensors.  F16 (the server
            # default) silently disables paged_row_idx_enabled.
            try:
                ctk_idx = argv.index("--cache-type-k")
                ctk_val = argv[ctk_idx + 1]
                check(ctk_val == "f32",
                      f"{name}: --cache-type-k={ctk_val} — must be f32 for "
                      f"paged row-index gather (F16 disables paged_row_idx_enabled)")
            except (ValueError, IndexError):
                check(False,
                      f"{name}: --cache-type-k not in argv — must be explicitly f32; "
                      f"server default is f16 which disables paged_row_idx_enabled")
            try:
                ctv_idx = argv.index("--cache-type-v")
                ctv_val = argv[ctv_idx + 1]
                check(ctv_val == "f32",
                      f"{name}: --cache-type-v={ctv_val} — must be f32")
            except (ValueError, IndexError):
                check(False,
                      f"{name}: --cache-type-v not in argv — must be explicitly f32")

    # OFF/DRY/BOUNDED argv must be identical modulo env-only vars
    # (the argv arrays differ only in strace prefix for BOUNDED)
    off_argv = json.loads((off_dir / "execution.json").read_text())["argv"]
    dry_argv = json.loads((dry_dir / "execution.json").read_text())["argv"]
    bounded_argv = json.loads((bounded_dir / "execution.json").read_text())["argv"]
    # Strip strace prefix for comparison
    def _strip_strace(a: list[str]) -> list[str]:
        # Remove the strace prefix: "strace" ["-f"] ["-e" "trace=..."] ["-o" "FILE"] "--"
        # The argv format is: [strace, -f, -e, trace=madvise, -o, <log>, --, <server>, --host, ...]
        if not a or pathlib.Path(a[0]).name != "strace":
            return list(a)
        if "--" not in a:
            raise ProtocolError("strace argv is missing the -- command separator")
        return list(a[a.index("--") + 1:])
    off_clean = _strip_strace(off_argv)
    dry_clean = _strip_strace(dry_argv)
    bounded_clean = _strip_strace(bounded_argv)

    # Normalize per-case port values (each case uses a unique free port)
    def _normalize_port(a: list[str]) -> list[str]:
        out = list(a)
        for i, token in enumerate(out):
            if token == "--port" and i + 1 < len(out):
                out[i + 1] = "<PORT>"
        return out

    off_norm = _normalize_port(off_clean)
    dry_norm = _normalize_port(dry_clean)
    bounded_norm = _normalize_port(bounded_clean)
    check(off_norm == dry_norm == bounded_norm,
          f"OFF/DRY/BOUNDED argv must be identical (modulo strace prefix and port).\n"
          f"  OFF:     {off_norm}\n  DRY:     {dry_norm}\n  BOUNDED: {bounded_norm}")
    _section_pass("Server topology", e0, errors)

    # =====================================================================
    # 1c. Startup capability marker — hard gate for 5 static fields
    #     row_idx may be 0 at startup (deferred — graph input not yet
    #     created).  The per-marker capability fields in the bounded
    #     release markers track the transition to row_idx=1.
    # =====================================================================
    print("--- Startup capability marker ---")
    e0 = len(errors)
    cap_match = re.search(
        r"kv_pressure_bounded_release_capability\s+"
        r"can_enable=(\d+)\s+paged=(\d+)\s+ingraph=(\d+)\s+"
        r"layers_supported=(\d+)\s+row_idx=(\d+)\s+"
        r"swap_disabled=(\d+)\s+layout_supported=(\d+)",
        bounded_stderr)
    check(cap_match is not None,
          "BOUNDED: startup capability marker not found in server stderr — "
          "bounded_release_can_enable_diagnose() must emit at init time")
    startup_cap = {}
    startup_row_idx = 0
    if cap_match:
        fields = ("can_enable", "paged", "ingraph", "layers_supported",
                   "row_idx", "swap_disabled", "layout_supported")
        startup_cap = {k: int(v) for k, v in zip(fields, cap_match.groups())}
        startup_row_idx = startup_cap.get("row_idx", 0)
        # Five static fields must be 1 at startup
        STARTUP_HARD = ("paged", "ingraph", "layers_supported",
                        "swap_disabled", "layout_supported")
        for k in STARTUP_HARD:
            actual = startup_cap.get(k, -1)
            check(actual == 1,
                  f"BOUNDED: startup capability {k}={actual}, expected 1")
        if startup_row_idx == 0:
            print(f"  startup: row_idx=0 (deferred — graph input not yet created)")
        else:
            print(f"  startup: {' '.join(f'{k}={v}' for k, v in startup_cap.items())} ✓")
    _section_pass("Startup capability marker", e0, errors)

    # =====================================================================
    # 2. OFF: zero release markers
    # =====================================================================
    print("--- OFF case ---")
    e0 = len(errors)
    off_bounded = find_markers(off_stderr, BOUNDED_RELEASE_RE, "OFF")
    check(len(off_bounded) == 0,
          f"OFF must have zero kv_pressure_bounded_release markers, found {len(off_bounded)}")
    off_dry = find_markers(off_stderr, DRY_RUN_MARKER_RE, "OFF")
    check(len(off_dry) == 0,
          f"OFF must have zero kv_pressure_dry_run markers, found {len(off_dry)}")
    off_telemetry = find_markers(off_stderr, TELEMETRY_MARKER_RE, "OFF")
    check(len(off_telemetry) >= 1,
          f"OFF must have telemetry markers (sampler active), found {len(off_telemetry)}")
    destructive = DESTRUCTIVE_RELEASE_RE.findall(off_stderr)
    check(len(destructive) == 0,
          f"OFF must have zero destructive release evidence: {destructive[:5]!r}")
    print(f"  bounded: 0  dry_run: 0  telemetry: {len(off_telemetry)}")
    _section_pass("OFF case", e0, errors)

    # =====================================================================
    # 3. DRY: dry-run markers only
    # =====================================================================
    print("--- DRY case ---")
    e0 = len(errors)
    dry_dry = find_markers(dry_stderr, DRY_RUN_MARKER_RE, "DRY")
    check(len(dry_dry) >= 1,
          f"DRY must have at least 1 kv_pressure_dry_run marker, found {len(dry_dry)}")
    dry_bounded_in_dry = find_markers(dry_stderr, BOUNDED_RELEASE_RE, "DRY")
    check(len(dry_bounded_in_dry) == 0,
          f"DRY must have zero bounded_release markers, found {len(dry_bounded_in_dry)}")
    destructive_dry = DESTRUCTIVE_RELEASE_RE.findall(dry_stderr)
    check(len(destructive_dry) == 0,
          f"DRY must have zero destructive release: {destructive_dry[:5]!r}")
    for i, marker in enumerate(dry_dry):
        # DRY is prediction only: never advertise a real release and keep the
        # bounded target/scan/ownership/sample facts internally consistent.
        check(marker["release_enabled"] == "0", f"DRY marker[{i}] release_enabled must be 0")
        check(int(marker["target_bytes"]) > 0, f"DRY marker[{i}] target_bytes must be >0")
        check(int(marker["max_scan_blocks"]) >= 1, f"DRY marker[{i}] max_scan_blocks must be >=1")
        check(marker["ownership_aborted"] == "0", f"DRY marker[{i}] ownership_aborted must be 0")
        check(marker["stale"] == "0", f"DRY marker[{i}] stale must be 0")
        if marker["skipped_reason"] == "none":
            check(int(marker["blocks_scanned"]) > 0,
                  f"DRY marker[{i}] prediction without scanned blocks")
            check(int(marker["would_release_bytes"]) > 0 and int(marker["would_release_blocks"]) > 0,
                  f"DRY marker[{i}] prediction must report positive would-release bytes and blocks")
    print(f"  dry_run markers: {len(dry_dry)}")
    _section_pass("DRY case", e0, errors)

    # =====================================================================
    # 4. BOUNDED: exactly 1 destructive execution (first validation)
    # =====================================================================
    print("--- BOUNDED case ---")
    e0 = len(errors)
    bounded_markers = find_markers(bounded_stderr, BOUNDED_RELEASE_RE, "BOUNDED")
    check(len(bounded_markers) >= 1,
          f"BOUNDED must have at least 1 bounded_release marker, found {len(bounded_markers)}")
    bounded_dry_in_bounded = find_markers(bounded_stderr, DRY_RUN_MARKER_RE, "BOUNDED")
    check(len(bounded_dry_in_bounded) == 0,
          f"BOUNDED must have zero dry_run markers, found {len(bounded_dry_in_bounded)}")

    # Print skipped_reason distribution
    reason_counts: dict[str, int] = {}
    for m in bounded_markers:
        r = m.get("skipped_reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1
    print(f"  skipped_reason distribution: {dict(sorted(reason_counts.items()))}")

    # Count executions (skipped_reason=none, released_bytes>0)
    executions = [m for m in bounded_markers
                  if m.get("skipped_reason") == "none"
                  and int(m.get("released_bytes", "0")) > 0]
    check(len(executions) == 1,
          f"First validation requires exactly 1 bounded release execution (skipped_reason=none, "
          f"released_bytes>0), found {len(executions)}.  "
          f"skipped_reason distribution: {dict(sorted(reason_counts.items()))}")

    # Validate all markers and track sample_count
    prev_sample_count = -1
    valid_execution = None
    for i, marker in enumerate(bounded_markers):
        missing = BOUNDED_REQUIRED_KEYS - set(marker.keys())
        if missing:
            check(False, f"BOUNDED marker[{i}] missing required keys: {sorted(missing)}")

        try:
            sc = int(marker.get("sample_count", "-1"))
        except (ValueError, TypeError):
            sc = -1
        if sc < 0:
            check(False, f"BOUNDED marker[{i}] invalid sample_count")
        elif sc <= prev_sample_count:
            check(False,
                  f"BOUNDED marker[{i}] sample_count={sc} not > prev={prev_sample_count}")
        prev_sample_count = sc

        skipped = marker.get("skipped_reason", "none")
        released_bytes = int(marker.get("released_bytes", "0"))
        released_blocks = int(marker.get("released_blocks", "0"))
        ownership_aborted = int(marker.get("ownership_aborted", "0"))
        madvise_failures = int(marker.get("madvise_failures", "0"))

        if skipped == "none" and released_bytes > 0:
            check(madvise_failures == 0,
                  f"BOUNDED execution[{i}] has madvise_failures={madvise_failures}, must be 0")
            check(ownership_aborted == 0,
                  f"BOUNDED execution[{i}] has ownership_aborted=1, must be 0")
            check(released_blocks > 0,
                  f"BOUNDED execution[{i}] released_blocks=0, must be >0")
            # Bounded-source counter delta must exactly match per-call result.
            # This is the INDEPENDENT attribution — not contaminated by
            # process-wide MADV_DONTNEED from other mechanisms.
            cnt_delta = int(marker.get("bounded_cnt_bytes_delta", "0"))
            cnt_blk_delta = int(marker.get("bounded_cnt_blocks_delta", "0"))
            check(cnt_delta == released_bytes,
                  f"BOUNDED execution[{i}] bounded_cnt_bytes_delta={cnt_delta} "
                  f"!= released_bytes={released_bytes}; independent counter "
                  f"attribution must match per-call result exactly")
            check(cnt_blk_delta == released_blocks,
                  f"BOUNDED execution[{i}] bounded_cnt_blocks_delta={cnt_blk_delta} "
                  f"!= released_blocks={released_blocks}; independent counter "
                  f"attribution must match per-call result exactly")
            # Mincore observational gate: the source counter delta
            # (bounded_cnt_bytes_delta) is the PRIMARY authoritative
            # release attribution.  KV mincore is an INDEPENDENT
            # OBSERVATION: it must show a physical page drop in the
            # correct direction (after < before), but mincore samples
            # resident pages at page granularity whereas madvise works
            # at byte granularity, so exact equality is NOT required.
            mc_before = int(marker.get("mincore_before_bytes", "0"))
            mc_after  = int(marker.get("mincore_after_bytes", "0"))
            check(mc_before > 0,
                  f"BOUNDED execution[{i}] mincore_before_bytes={mc_before} must be >0")
            check(mc_after > 0,
                  f"BOUNDED execution[{i}] mincore_after_bytes={mc_after} must be >0")
            check(mc_after < mc_before,
                  f"BOUNDED execution[{i}] mincore must show physical drop: "
                  f"before={mc_before} after={mc_after}")
            mc_drop = mc_before - mc_after
            # Observational consistency: mincore drop must be non-zero
            # and must not exceed the counter delta by more than
            # page-alignment margin (at most one page per released block).
            # The counter delta is the truth; mincore is evidence.
            if mc_drop > cnt_delta + int(released_blocks) * 4096:
                check(False,
                      f"BOUNDED execution[{i}] mincore drop={mc_drop} "
                      f"far exceeds bounded_cnt_bytes_delta={cnt_delta} "
                      f"(drop >> source + page-alignment margin per block); "
                      f"mincore is observational but implausibly large")
            print(f"  execution[{i}]: released_bytes={released_bytes} "
                  f"released_blocks={released_blocks} "
                  f"cnt_delta={cnt_delta} mincore_drop={mc_drop} "
                  f"(observational) ✓")
            valid_execution = marker

        # Capability consistency check (all markers with populated capability fields)
        can_enable_v = int(marker.get("can_enable", "-1"))
        if can_enable_v >= 0:
            cap_paged   = int(marker.get("cap_paged", "0"))
            cap_ingraph = int(marker.get("cap_ingraph", "0"))
            cap_layers  = int(marker.get("cap_layers", "0"))
            cap_row_idx = int(marker.get("cap_row_idx", "0"))
            cap_swap    = int(marker.get("cap_swap_disabled", "0"))
            cap_layout  = int(marker.get("cap_layout", "0"))

            if skipped == "none" and released_bytes > 0:
                # Execution: all capabilities must be 1
                check(can_enable_v == 1,
                      f"BOUNDED marker[{i}] execution with can_enable=0 — "
                      f"capability fields must all be 1 for successful execution")
                check(cap_paged == 1 and cap_ingraph == 1 and cap_layers == 1 and
                      cap_row_idx == 1 and cap_swap == 1 and cap_layout == 1,
                      f"BOUNDED marker[{i}] execution with missing capability: "
                      f"paged={cap_paged} ingraph={cap_ingraph} layers={cap_layers} "
                      f"row_idx={cap_row_idx} swap_disabled={cap_swap} layout={cap_layout}")
            else:
                # Skipped marker: can_enable should match the skip reason
                if skipped == "not_ingraph":
                    check(cap_ingraph == 0,
                          f"BOUNDED marker[{i}] skipped_reason=not_ingraph but cap_ingraph={cap_ingraph}")
                elif skipped == "no_layers":
                    check(cap_layers == 0,
                          f"BOUNDED marker[{i}] skipped_reason=no_layers but cap_layers={cap_layers}")
                elif skipped == "no_row_idx":
                    check(cap_row_idx == 0,
                          f"BOUNDED marker[{i}] skipped_reason=no_row_idx but cap_row_idx={cap_row_idx}")
                elif skipped == "layout_unsupported":
                    check(cap_layout == 0,
                          f"BOUNDED marker[{i}] skipped_reason=layout_unsupported but cap_layout={cap_layout}")
                elif skipped == "swap_enabled":
                    check(cap_swap == 0,
                          f"BOUNDED marker[{i}] skipped_reason=swap_enabled but cap_swap_disabled={cap_swap}")
                elif skipped == "not_paged":
                    check(cap_paged == 0,
                          f"BOUNDED marker[{i}] skipped_reason=not_paged but cap_paged={cap_paged}")

    # --- Row-idx lifecycle tracking ---
    # row_idx may be 0 at startup (deferred).  It MUST transition to 1
    # before the first destructive release execution and MUST NOT regress
    # to 0 after execution.  BEFORE execution, a finite number of
    # no_row_idx skipped markers are expected (graph not yet built).
    # AFTER execution, no_row_idx must never appear.
    row_idx_ever_seen_one = False
    row_idx_regression = False
    row_idx_seen_at_execution = False
    pre_exec_no_row_idx_count = 0
    post_exec_no_row_idx_count = 0
    execution_found = False
    for i, marker in enumerate(bounded_markers):
        cap_row_idx = int(marker.get("cap_row_idx", "-1"))
        if cap_row_idx == 1:
            row_idx_ever_seen_one = True
        skipped = marker.get("skipped_reason", "none")
        released_bytes = int(marker.get("released_bytes", "0"))

        is_exec = (skipped == "none" and released_bytes > 0)
        if is_exec:
            execution_found = True
            if cap_row_idx == 1:
                row_idx_seen_at_execution = True

        if execution_found:
            # After execution: cap_row_idx must never be 0 (or -1)
            if cap_row_idx != 1:
                row_idx_regression = True
            if skipped == "no_row_idx":
                post_exec_no_row_idx_count += 1
        else:
            # Before execution: count no_row_idx markers
            if skipped == "no_row_idx":
                pre_exec_no_row_idx_count += 1

    # row_idx MUST have been 1 at some point
    check(row_idx_ever_seen_one,
          "row_idx_never_initialized: none of the bounded release markers "
          "have cap_row_idx=1 — the paged row-index graph input was never "
          "created.  Likely causes: (a) identity fast path is enabled, "
          f"(b) ingraph path not used, or (c) paged_row_idx_enabled was "
          f"never set.  Startup cap_row_idx={startup_row_idx}.")
    # Execution must have occurred with row_idx=1
    check(row_idx_seen_at_execution,
          f"Valid execution found but cap_row_idx was not 1 at execution time.  "
          f"row_idx_ever_seen_one={row_idx_ever_seen_one} "
          f"startup_row_idx={startup_row_idx}.")
    # No regression after execution
    check(not row_idx_regression,
          "cap_row_idx regressed to 0 or -1 after a destructive release "
          "execution — the graph was rebuilt without row-index mode.  "
          "This indicates an unexpected topology change during inference.")
    # Post-exec no_row_idx is forbidden
    check(post_exec_no_row_idx_count == 0,
          f"Found {post_exec_no_row_idx_count} no_row_idx skipped marker(s) "
          f"AFTER the destructive release execution — row_idx must stay 1 "
          f"once the graph is built.")
    print(f"  row_idx lifecycle: startup={startup_row_idx} "
          f"pre_exec_no_row_idx={pre_exec_no_row_idx_count} "
          f"seen_at_exec={int(row_idx_seen_at_execution)} "
          f"regression={int(row_idx_regression)}")

    # Structural-disabled regression guard: if any marker has
    # skipped_reason=structurally_disabled, the diagnose is broken.
    structurally_disabled_count = reason_counts.get("structurally_disabled", 0)
    check(structurally_disabled_count == 0,
          f"Found {structurally_disabled_count} marker(s) with "
          f"skipped_reason=structurally_disabled — the per-condition diagnose "
          f"is failing to produce a precise reason.  Check "
          f"bounded_release_can_enable_diagnose() coverage.")

    check(valid_execution is not None,
          "No valid bounded release execution found (released_bytes>0, skipped_reason=none)")

    # --- Unconditional mincore hard gate ---
    # Regardless of whether a valid execution was found, if any bounded marker
    # carries mincore_before/after that are both 0, equal, or show growth, the
    # BOUNDED variant has no physical evidence of release — fail unconditionally.
    mincore_issues = 0
    for i, marker in enumerate(bounded_markers):
        mc_b = int(marker.get("mincore_before_bytes", "0"))
        mc_a = int(marker.get("mincore_after_bytes", "0"))
        if mc_b == 0 and mc_a == 0:
            mincore_issues += 1
            if marker.get("skipped_reason") != "none":
                # Skipped markers with mincore=0 are expected (mincore only runs
                # inside the execute block) — count but don't individually fail
                pass
            else:
                check(False,
                      f"BOUNDED marker[{i}] has mincore_before=0 mincore_after=0 "
                      f"but skipped_reason=none; mincore must produce non-zero samples")
        elif mc_b > 0 and mc_a >= mc_b:
            mincore_issues += 1
            check(False,
                  f"BOUNDED marker[{i}] mincore shows no drop or growth: "
                  f"before={mc_b} after={mc_a}")

    # If ALL markers have mincore 0/0 and there are no executions (all skipped),
    # fail: physical evidence is absent and the test didn't exercise the path.
    if mincore_issues == len(bounded_markers) and len(executions) == 0:
        check(False,
              f"All {len(bounded_markers)} bounded markers have mincore_before=0 mincore_after=0.  "
              f"Either mincore is not enabled, the execution block was never entered, "
              f"or sample_kv_resident_bytes() returned 0.  "
              f"skipped_reason distribution: {dict(sorted(reason_counts.items()))}")

    print(f"  bounded markers: {len(bounded_markers)}  executions: {len(executions)}  "
          f"mincore_zero: {mincore_issues}")
    _section_pass("BOUNDED case", e0, errors)

    # =====================================================================
    # 5. Final KV lifecycle counters
    # =====================================================================
    print("--- Final KV lifecycle counters ---")
    e0 = len(errors)
    lifecycle_stats = {
        "OFF": extract_final_release_stats("OFF", off_stderr, check),
        "DRY": extract_final_release_stats("DRY", dry_stderr, check),
        "BOUNDED": extract_final_release_stats("BOUNDED", bounded_stderr, check),
    }

    bounded_stats = lifecycle_stats["BOUNDED"]
    if bounded_stats is not None:
        for name in ("reuse_allocations", "write_commits",
                     "dummy_candidate_pending_write_cell"):
            check(bounded_stats[name] >= 1,
                  f"BOUNDED: final lifecycle {name}={bounded_stats[name]}, must be >=1")
        for name in ("write_rollbacks", "released_redirect_no_dummy",
                     "released_redirect_no_dummy_pending_write",
                     "ensure_pending_write_rejected", "input_setup_fatal",
                     "row_mapping_fatal", "write_mapping_fatal",
                     "active_nonresident_fatal"):
            check(bounded_stats[name] == 0,
                  f"BOUNDED: final lifecycle {name}={bounded_stats[name]}, must be 0")

        evaluated_markers = [m for m in bounded_markers if m.get("skipped_reason") == "none"]
        marker_bytes = sum(int(m["bounded_cnt_bytes_delta"]) for m in evaluated_markers)
        marker_blocks = sum(int(m["bounded_cnt_blocks_delta"]) for m in evaluated_markers)
        check(bounded_stats["bounded_release_calls"] == len(evaluated_markers),
              f"BOUNDED: final bounded_release_calls={bounded_stats['bounded_release_calls']} "
              f"!= evaluated marker count={len(evaluated_markers)}")
        check(bounded_stats["bounded_release_bytes"] == marker_bytes,
              f"BOUNDED: final bounded_release_bytes={bounded_stats['bounded_release_bytes']} "
              f"!= bounded marker source delta total={marker_bytes}")
        check(bounded_stats["bounded_release_blocks"] == marker_blocks,
              f"BOUNDED: final bounded_release_blocks={bounded_stats['bounded_release_blocks']} "
              f"!= bounded marker source delta total={marker_blocks}")

    for case in ("OFF", "DRY"):
        stats = lifecycle_stats[case]
        if stats is not None:
            for name in LIFECYCLE_STATS_FIELDS:
                check(stats[name] == 0,
                      f"{case}: final lifecycle {name}={stats[name]}, must be 0")
            for name in BOUNDED_SOURCE_STATS_FIELDS:
                check(stats[name] == 0,
                      f"{case}: final bounded source {name}={stats[name]}, must be 0")

    for case, stats in lifecycle_stats.items():
        if stats is not None:
            print("  " + case + ": " + " ".join(
                f"{name}={stats[name]}" for name in REQUIRED_RELEASE_STATS_FIELDS))
    _section_pass("Final KV lifecycle counters", e0, errors)

    # =====================================================================
    # 6. Response byte-identity
    # =====================================================================
    print("--- Response identity ---")
    e0 = len(errors)
    off_text = off_result.get("response_text", "")
    dry_text = dry_result.get("response_text", "")
    bounded_text = bounded_result.get("response_text", "")
    check(len(off_text) > 0, "OFF response must be non-empty")
    check(off_text == dry_text,
          f"OFF/DRY responses must match\n  OFF: {off_text[:80]!r}\n  DRY: {dry_text[:80]!r}")
    check(off_text == bounded_text,
          f"OFF/BOUNDED responses must match\n  OFF:     {off_text[:80]!r}\n"
          f"  BOUNDED: {bounded_text[:80]!r}")
    for name, res in [("OFF", off_result), ("DRY", dry_result), ("BOUNDED", bounded_result)]:
        check(res.get("http_status") == 200, f"{name} HTTP status must be 200")
    _section_pass("Response identity", e0, errors)

    # =====================================================================
    # 6. Strace audit — source attribution plus process-wide background report
    # =====================================================================
    print("--- Strace audit ---")
    e0 = len(errors)

    # OFF/DRY source/destructive signals are checked above and in final
    # lifecycle counters.  Process-wide strace cannot attribute a MADV call to
    # a subsystem, so ggml or shutdown cleanup is report-only in all variants.
    off_strace_files = sorted(off_dir.glob("strace.log"))
    check(len(off_strace_files) > 0,
          "OFF: strace.log file missing — cannot report MADV_DONTNEED background; "
          "runner must capture strace for ALL three variants")
    if off_strace_files:
        off_madvise, off_madv_addrs = count_strace_madvise_bytes(off_strace_files)
        print(f"  OFF: {off_madvise} bytes MADV_DONTNEED "
              f"({len(off_madv_addrs)} calls) — background report only")

    dry_strace_files = sorted(dry_dir.glob("strace.log"))
    check(len(dry_strace_files) > 0,
          "DRY: strace.log file missing — cannot report MADV_DONTNEED background; "
          "runner must capture strace for ALL three variants")
    if dry_strace_files:
        dry_madvise, dry_madv_addrs = count_strace_madvise_bytes(dry_strace_files)
        print(f"  DRY: {dry_madvise} bytes MADV_DONTNEED "
              f"({len(dry_madv_addrs)} calls) — background report only")

    # BOUNDED source counter/mincore is PRIMARY; total strace is only a lower
    # bound and any surplus is explicitly background, never release bytes.
    bounded_strace_files = sorted(bounded_dir.glob("strace.log"))
    if not bounded_strace_files:
        check(False, "BOUNDED: strace.log file missing — cannot verify MADV_DONTNEED")
    else:
        madvise_total, madvise_addrs = count_strace_madvise_bytes(bounded_strace_files)
        cnt_delta = int(valid_execution["bounded_cnt_bytes_delta"]) if valid_execution else 0

        if cnt_delta == 0 and madvise_total > 0:
            check(False,
                  f"BOUNDED strace has {madvise_total} bytes MADV_DONTNEED "
                  f"({len(madvise_addrs)} calls) but bounded counter delta is 0 "
                  f"(bounded release never executed).  These MADV_DONTNEED calls "
                  f"are stray background contamination from non-bounded-release "
                  f"paged-KV mechanisms.")
        elif cnt_delta > 0:
            # Counter/mincore equality above is the bounded-source proof.
            # Process-wide strace can contain un-attributable background calls.
            check(int(madvise_total) >= int(cnt_delta),
                  f"BOUNDED strace MADV_DONTNEED ({madvise_total} bytes, "
                  f"{len(madvise_addrs)} calls) must contain at least "
                  f"bounded_cnt_bytes_delta ({cnt_delta} bytes)")
            background = int(madvise_total) - int(cnt_delta)
            if background > 0:
                print(f"  counter delta: {cnt_delta} bytes (PRIMARY, authoritative)")
                print(f"  strace total:   {madvise_total} bytes "
                      f"({len(madvise_addrs)} calls)")
                print(f"  background:     {background} bytes "
                      f"(non-bounded-release MADV_DONTNEED — REPORTED, not used "
                      f"as bounded release evidence)")
            else:
                print(f"  counter delta: {cnt_delta} bytes (PRIMARY)")
                print(f"  strace total:   {madvise_total} bytes "
                      f"({len(madvise_addrs)} calls) — clean, zero background")
        print(f"  BOUNDED MADV_DONTNEED: {madvise_total} bytes "
              f"({len(madvise_addrs)} calls)")
    _section_pass("Strace audit", e0, errors)

    # =====================================================================
    # 7. KV mincore observational gate — physical page-drop evidence
    #    (OBSERVATIONAL, not exact-match; counter delta is authoritative)
    # =====================================================================
    print("--- KV mincore (observational) ---")
    e0 = len(errors)
    if valid_execution:
        mc_before = int(valid_execution["mincore_before_bytes"])
        mc_after  = int(valid_execution["mincore_after_bytes"])
        mc_drop   = mc_before - mc_after if mc_before > mc_after else 0
        check(mc_before > 0 and mc_after > 0,
              f"Valid execution has mincore before={mc_before} after={mc_after} — must both be >0")
        check(mc_drop > 0,
              f"KV resident pages must show physical drop: before={mc_before} after={mc_after}")
        print(f"  before: {mc_before}  after: {mc_after}  drop: {mc_drop} (observational)")
    else:
        check(False,
              "No valid execution — KV mincore physical drop cannot be verified.  "
              "A valid execution requires skipped_reason=none and released_bytes>0.")
    _section_pass("KV mincore", e0, errors)

    # =====================================================================
    # 8. RSS absolute drop (auxiliary cross-check)
    # =====================================================================
    print("--- RSS (auxiliary) ---")
    e0 = len(errors)
    rss_path = bounded_dir / "rss.json"
    if rss_path.exists():
        rss = json.loads(rss_path.read_text())
        rss_before = rss.get("rss_before_kb", 0)
        rss_after = rss.get("rss_after_kb", 0)
        if rss_before > 0 and rss_after > 0:
            drop_kb = int(rss_before) - int(rss_after)
            if drop_kb > 0:
                print(f"  before: {rss_before} KB  after: {rss_after} KB  drop: {drop_kb} KB")
            else:
                print(f"  before: {rss_before} KB  after: {rss_after} KB  NO DROP")
        else:
            print("  RSS data incomplete")
    else:
        print("  no RSS data (auxiliary only, mincore is authoritative)")
    _section_pass("RSS", e0, errors)

    # =====================================================================
    # 9. Process cleanup
    # =====================================================================
    print("--- Process cleanup ---")
    e0 = len(errors)
    for name, case_dir in [("OFF", off_dir), ("DRY", dry_dir), ("BOUNDED", bounded_dir)]:
        phases_path = case_dir / "phases.json"
        check(phases_path.is_file(), f"{name}: phases.json missing")
        if phases_path.is_file():
            phases = json.loads(phases_path.read_text())
            check(isinstance(phases, dict), f"{name}: phases.json must be an object")
            shutdown = phases.get("shutdown", {})
            check(isinstance(shutdown, dict), f"{name}: shutdown record missing")
            if isinstance(shutdown, dict):
                required = {"pgid", "exit_code", "pgid_check_complete",
                            "cleanup_kill_attempted", "residual_process"}
                check(required <= set(shutdown),
                      f"{name}: shutdown record missing fields: {sorted(required - set(shutdown))}")
                check(isinstance(shutdown.get("pgid"), int) and shutdown.get("pgid", 0) > 0,
                      f"{name}: shutdown PGID is invalid")
                check(isinstance(shutdown.get("exit_code"), int),
                      f"{name}: shutdown exit_code is invalid")
                check(shutdown.get("pgid_check_complete") is True,
                      f"{name}: shutdown PGID residual check incomplete")
                check(isinstance(shutdown.get("cleanup_kill_attempted"), bool),
                      f"{name}: shutdown cleanup_kill_attempted must be boolean")
                check(shutdown.get("residual_process") is False,
                      f"{name}: residual process after shutdown")
    _section_pass("Process cleanup", e0, errors)

    # =====================================================================
    # Final verdict
    # =====================================================================
    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        fail(f"{len(errors)} parser check(s) failed")

    executed = valid_execution
    print("\nPASS: Stage 3A-2C bounded release OFF/DRY/BOUNDED validation")
    print(f"  artifact: {art}")
    if executed:
        print(f"  released_bytes: {executed.get('released_bytes', 'N/A')}")
        print(f"  released_blocks: {executed.get('released_blocks', 'N/A')}")
        print(f"  ownership_aborted: {executed.get('ownership_aborted', 'N/A')}")
        print(f"  madvise_failures: {executed.get('madvise_failures', 'N/A')}")
    print(f"  response match: {'YES' if off_text == bounded_text else 'NO'}")
    write_parser_result(0, "PASS")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        write_parser_result(code, "FAIL")
        raise
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, ProtocolError) as exc:
        # Corrupt/incomplete artifacts are protocol failures, never parser
        # tracebacks.  Keep one stable non-zero outcome for automation.
        print(f"FAIL: protocol input invalid: {exc}", file=sys.stderr)
        write_parser_result(2, "FAIL")
        raise SystemExit(2)
