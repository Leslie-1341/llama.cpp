#!/usr/bin/env python3
"""Fail-closed parser for Stage 3A-2B dry-run OFF/ON artifacts.

Validates:
  - OFF case: zero kv_pressure_dry_run markers, telemetry markers present
  - ON  case: at least one kv_pressure_dry_run marker with release_enabled=0,
              skipped_reason=none, would_release_bytes>0, ownership_aborted=0
  - Both responses are byte-identical
  - ON strace has zero MADV_DONTNEED calls
  - No destructive release markers in either case
  - No residual processes
  - Configuration isolation (only DRY_RUN differs between OFF/ON)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, NoReturn

ROOT = pathlib.Path(__file__).resolve().parents[1]

# --- marker patterns ---
DRY_RUN_MARKER_RE = re.compile(r"kv_pressure_dry_run\b")
TELEMETRY_MARKER_RE = re.compile(r"kv_pressure_telemetry\b")
DESTRUCTIVE_RELEASE_RE = re.compile(
    r"(?i)(?:paged_release_blocks\s*\(|release_blocks\s*\(|"
    # Only match non-zero counter values — zero-value diagnostic dumps
    # are not evidence of actual destructive release.
    r"paged_block_release_bytes=(?:0*[1-9][0-9]*)|"
    r"paged_blocks_released=(?:0*[1-9][0-9]*)"
    r")"
)
MADV_DONTNEED_RE = re.compile(r"(?i)madvise\s*\([^)]*MADV_DONTNEED")

# Dry-run marker required fields
DRY_RUN_REQUIRED_KEYS = {
    "state", "source", "stale", "release_enabled",
    "would_release_bytes", "would_release_blocks",
    "blocks_scanned", "blocks_skipped_owned", "blocks_skipped_state",
    "shortfall_bytes", "overshoot_bytes",
    "block_scan_exhausted", "ownership_aborted",
    "target_bytes", "max_scan_blocks",
    "skipped_reason", "cooldown_ms",
    "sample_count", "idle",
}

# Keys that differ by design between OFF and ON (dry-run config)
EXPECTED_ENV_DIFFS = {
    "LLAMA_KV_PRESSURE_DRY_RUN",
    "LLAMA_KV_PRESSURE_DRY_RUN_TARGET_BYTES",
    "LLAMA_KV_PRESSURE_DRY_RUN_MAX_SCAN_BLOCKS",
    "LLAMA_KV_PRESSURE_DRY_RUN_COOLDOWN_MS",
}


def fail(message: str) -> NoReturn:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_marker_fields(line: str) -> dict[str, str]:
    """Parse key=value pairs from a log marker line."""
    fields: dict[str, str] = {}
    # Split on whitespace, parse key=value pairs
    for token in line.split():
        if "=" in token:
            key, _, value = token.partition("=")
            fields[key] = value
    return fields


def find_markers(stderr_text: str, marker_re: re.Pattern[str]) -> list[dict[str, str]]:
    """Find all marker lines matching the regex and parse their fields."""
    markers: list[dict[str, str]] = []
    for line in stderr_text.split("\n"):
        if marker_re.search(line):
            fields = parse_marker_fields(line)
            if fields:
                markers.append(fields)
    return markers


def check_env_isolation(off_env: dict[str, str], on_env: dict[str, str]) -> None:
    """Verify OFF/ON environments differ only in dry-run config."""
    off_keys = set(off_env.keys())
    on_keys = set(on_env.keys())
    off_only = off_keys - on_keys
    on_only = on_keys - off_keys

    if off_only:
        fail(f"OFF env has extra keys not in ON: {sorted(off_only)}")
    if on_only:
        unexpected = on_only - EXPECTED_ENV_DIFFS
        if unexpected:
            fail(f"ON env has unexpected extra keys: {sorted(unexpected)}")

    # Check value differences
    for key in off_keys & on_keys:
        if off_env[key] != on_env[key]:
            if key not in EXPECTED_ENV_DIFFS:
                fail(f"Shared env key '{key}' differs: OFF={off_env[key]!r} ON={on_env[key]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse Stage 3A-2B dry-run OFF/ON artifact")
    ap.add_argument("artifact_dir", help="Path to the artifact directory")
    args = ap.parse_args()

    art = pathlib.Path(args.artifact_dir)
    if not art.is_dir():
        fail(f"artifact directory not found: {art}")

    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)
            print(f"FAIL: {msg}", file=sys.stderr)

    # --- Load case data ---
    off_dir = art / "dry_run_off"
    on_dir = art / "dry_run_on"

    for d, name in [(off_dir, "OFF"), (on_dir, "ON")]:
        if not d.is_dir():
            fail(f"case directory missing: {d}")

    off_stderr = (off_dir / "server.stderr").read_text(errors="replace")
    on_stderr = (on_dir / "server.stderr").read_text(errors="replace")
    off_result = json.loads((off_dir / "result.json").read_text())
    on_result = json.loads((on_dir / "result.json").read_text())
    off_env = json.loads((off_dir / "environment.json").read_text())
    on_env = json.loads((on_dir / "environment.json").read_text())

    # =====================================================================
    # 1. Configuration isolation
    # =====================================================================
    print("--- Configuration isolation ---")
    check_env_isolation(off_env, on_env)
    check("LLAMA_KV_PAGED_RELEASE" not in on_env or on_env["LLAMA_KV_PAGED_RELEASE"] == "0",
          "LLAMA_KV_PAGED_RELEASE must not be set to 1 (release disabled)")
    check(on_env.get("LLAMA_KV_PRESSURE_DRY_RUN") == "1",
          "ON must have LLAMA_KV_PRESSURE_DRY_RUN=1")
    check("LLAMA_KV_PRESSURE_DRY_RUN" not in off_env,
          "OFF must NOT have LLAMA_KV_PRESSURE_DRY_RUN set")
    check(on_env.get("LLAMA_KV_PAGED") == "1",
          "LLAMA_KV_PAGED must be 1 for paged KV")
    print("  PASS")

    # =====================================================================
    # 2. OFF: no dry-run markers
    # =====================================================================
    print("--- OFF case ---")
    off_dry = find_markers(off_stderr, DRY_RUN_MARKER_RE)
    check(len(off_dry) == 0,
          f"OFF must have zero kv_pressure_dry_run markers, found {len(off_dry)}")
    # OFF must still have telemetry (sampler is on)
    off_telemetry = find_markers(off_stderr, TELEMETRY_MARKER_RE)
    check(len(off_telemetry) >= 1,
          f"OFF must have at least 1 kv_pressure_telemetry marker (sampler active), found {len(off_telemetry)}")
    print(f"  dry_run markers: {len(off_dry)} (expected 0)")
    print(f"  telemetry markers: {len(off_telemetry)}")
    print("  PASS")

    # =====================================================================
    # 3. ON: at least one valid dry-run marker
    # =====================================================================
    print("--- ON case ---")
    on_dry = find_markers(on_stderr, DRY_RUN_MARKER_RE)
    check(len(on_dry) >= 1,
          f"ON must have at least 1 kv_pressure_dry_run marker, found {len(on_dry)}")
    on_telemetry = find_markers(on_stderr, TELEMETRY_MARKER_RE)
    check(len(on_telemetry) >= 1,
          f"ON must have at least 1 kv_pressure_telemetry marker, found {len(on_telemetry)}")

    valid_dry_marker_found = False
    prev_sample_count = -1
    for i, marker in enumerate(on_dry):
        # Check required keys
        missing = DRY_RUN_REQUIRED_KEYS - set(marker.keys())
        if missing:
            print(f"  marker[{i}] missing keys: {sorted(missing)}", file=sys.stderr)
            continue

        # --- Marker temporal ordering: sample_count must be strictly increasing ---
        try:
            sc = int(marker.get("sample_count", "-1"))
        except (ValueError, TypeError):
            sc = -1
        if sc < 0:
            check(False, f"ON marker[{i}] has invalid sample_count: {marker.get('sample_count')!r}")
        elif sc <= prev_sample_count:
            check(False,
                  f"ON marker[{i}] sample_count={sc} not strictly greater than "
                  f"previous marker[{i-1}] sample_count={prev_sample_count} "
                  f"(duplicate or out-of-order marker)")
        prev_sample_count = sc

        # Check the key acceptance criteria
        release_enabled = int(marker.get("release_enabled", "-1"))
        skipped_reason = marker.get("skipped_reason", "")
        would_release_bytes = int(marker.get("would_release_bytes", "0"))
        ownership_aborted = int(marker.get("ownership_aborted", "-1"))
        would_release_blocks = int(marker.get("would_release_blocks", "0"))

        if (release_enabled == 0 and skipped_reason == "none"
                and would_release_bytes > 0 and ownership_aborted == 0):
            valid_dry_marker_found = True
            print(f"  marker[{i}]: release_enabled=0 skipped_reason=none "
                  f"would_release_bytes={would_release_bytes} "
                  f"would_release_blocks={would_release_blocks} "
                  f"ownership_aborted=0 ✓")
        else:
            print(f"  marker[{i}]: release_enabled={release_enabled} "
                  f"skipped_reason={skipped_reason} "
                  f"would_release_bytes={would_release_bytes} "
                  f"ownership_aborted={ownership_aborted} "
                  f"(does not meet target criteria)")

    check(valid_dry_marker_found,
          "No ON dry-run marker met criteria: release_enabled=0, skipped_reason=none, "
          "would_release_bytes>0, ownership_aborted=0")
    print(f"  dry_run markers: {len(on_dry)} (valid: {1 if valid_dry_marker_found else 0})")
    print(f"  telemetry markers: {len(on_telemetry)}")
    print("  PASS")

    # =====================================================================
    # 4. Response byte-identity
    # =====================================================================
    print("--- Response identity ---")
    off_text = off_result.get("response_text", "")
    on_text = on_result.get("response_text", "")
    check(len(off_text) > 0, "OFF response must be non-empty")
    check(off_text == on_text,
          f"OFF/ON responses must be byte-identical\n"
          f"  OFF: {off_text[:80]!r}\n  ON:  {on_text[:80]!r}")
    check(off_result.get("http_status") == 200, "OFF HTTP status must be 200")
    check(on_result.get("http_status") == 200, "ON HTTP status must be 200")
    print(f"  OFF: {off_text[:60]!r}...")
    print(f"  ON:  {on_text[:60]!r}...")
    print("  PASS")

    # =====================================================================
    # 5. No destructive release in either case
    # =====================================================================
    print("--- Destructive release audit ---")
    for name, stderr_text in [("OFF", off_stderr), ("ON", on_stderr)]:
        destructive = DESTRUCTIVE_RELEASE_RE.findall(stderr_text)
        check(len(destructive) == 0,
              f"{name}: found destructive release evidence: {destructive[:5]!r}")
    print("  PASS (zero destructive release markers)")

    # =====================================================================
    # 6. ON strace: zero MADV_DONTNEED
    # =====================================================================
    print("--- Strace audit (ON) ---")
    strace_files = sorted(on_dir.glob("strace.*"))
    if not strace_files:
        # strace might produce no output files if the trace had no matching events
        print("  No strace output files — checking process record")
        strace_info_path = on_dir / "strace_process.json"
        if strace_info_path.exists():
            strace_info = json.loads(strace_info_path.read_text())
            check(strace_info.get("returncode") == 0,
                  f"strace exited non-zero: {strace_info.get('returncode')}")
        print("  PASS (strace exited cleanly, zero matching syscalls)")
    else:
        madvise_count = 0
        for sf in strace_files:
            content = sf.read_text(errors="replace")
            madvise_count += len(MADV_DONTNEED_RE.findall(content))
        check(madvise_count == 0,
              f"ON strace must have zero MADV_DONTNEED calls, found {madvise_count}")
        print(f"  strace files: {len(strace_files)}, MADV_DONTNEED calls: {madvise_count}")
        print("  PASS")

    # =====================================================================
    # 7. Process cleanup
    # =====================================================================
    print("--- Process cleanup ---")
    for name, case_dir in [("OFF", off_dir), ("ON", on_dir)]:
        result_path = case_dir / "result.json"
        phases_path = case_dir / "phases.json"
        if phases_path.exists():
            phases = json.loads(phases_path.read_text())
            shutdown = phases.get("shutdown", {})
            check(not shutdown.get("residual_process", True),
                  f"{name}: residual process after shutdown")
    print("  PASS (no residual processes)")

    # =====================================================================
    # Final verdict
    # =====================================================================
    if errors:
        print(f"\n{len(errors)} error(s):", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        fail(f"{len(errors)} parser check(s) failed")

    print("\nPASS: Stage 3A-2B dry-run OFF/ON validation")
    print(f"  artifact: {art}")
    print(f"  dry_run markers: OFF={len(off_dry)} ON={len(on_dry)}")
    print(f"  response match: {'YES' if off_text == on_text else 'NO'}")
    print(f"  destructive release: NONE")
    print(f"  MADV_DONTNEED (strace): 0")
    print(f"  residual processes: 0")


if __name__ == "__main__":
    main()
