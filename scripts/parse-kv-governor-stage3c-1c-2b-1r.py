#!/usr/bin/env python3
"""Fail-closed verdict authority for the Stage 3C unified multi-slot smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTOCOL = "kv_governor_stage3c_1c_2b_1r"
MARKER = "kv_pressure_unified_action"
RESUME_MARKER = "kv_resume_order_event"
CASES = ("OFF", "GOVERNOR_ON", "INVALID_UNIFIED", "CONFLICT_UNIFIED_LEGACY")
UINT = re.compile(r"[0-9]+$")
REQUIRED = {"state", "source", "stale", "decision_id", "episode", "target_bytes", "max_blocks", "observed_excess_bytes", "debt_before_bytes", "debt_after_bytes", "offload_armed_before", "offload_armed_after", "next_action_sample", "evaluate_attempted", "evaluate_outcome", "evaluate_reason", "release_attempted", "offload_attempted", "selected_seq_id", "selected_claimant_epoch", "transaction_id", "outcome", "reason", "blocks", "bytes", "relieved_bytes", "shortfall_bytes", "io_failure", "io_errno", "state_changed", "decision_reason", "sample_count", "idle", "scores"}
RESUME_REQUIRED = {"phase", "decision_id", "seq_id", "claimant_epoch", "transaction_id", "action", "outcome", "reason", "graph_allowed"}
GOVERNOR_ENV = {"LLAMA_KV_PRESSURE_UNIFIED_ACTION", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"}


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
    symbolic = {"state", "source", "evaluate_outcome", "evaluate_reason", "outcome", "reason", "decision_reason", "scores"}
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


def token_lines(text: str, token: str, parser, origin: str, errors: list[str]) -> list[dict[str, str]]:
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if token in line:
            try:
                result.append(parser(line))
            except Error as exc:
                errors.append(f"{origin}:{number}: {exc}")
    return result


def expected_labels(parallel: int) -> set[str]:
    return ({f"seed_c{cycle}_s{slot}" for cycle in range(2) for slot in range(parallel)} |
            {f"reaccess_s{slot}" for slot in range(parallel)} |
            {f"reuse_c2_s{slot}" for slot in range(parallel)} | {f"active_s{parallel - 1}"})


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
        match = re.search(r"(?:_s)([0-9]+)$", label)
        if not match or item.get("request", {}).get("id_slot") != int(match.group(1)):
            errors.append(f"{name}: {label} does not bind its recorded slot")
    return result


def normalize_argv(argv: Any) -> list[str] | None:
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return None
    result = list(argv)
    try:
        result[result.index("--port") + 1] = "<port>"
    except (ValueError, IndexError):
        return None
    return result


def startup(text: str, parallel: int, env: dict[str, Any], enabled: bool, errors: list[str], missing: list[str], name: str) -> None:
    def one(pattern: str, label: str) -> str | None:
        matches = re.findall(pattern, text, re.MULTILINE)
        if not matches:
            missing.append(f"{name}: missing startup {label}")
            return None
        if len(matches) != 1:
            errors.append(f"{name}: duplicate startup {label}")
            return None
        return matches[0] if isinstance(matches[0], str) else matches[0][0]
    slots = one(r"initializing slots, n_slots = ([0-9]+)", "n_slots")
    nseq = one(r"llama_context: n_seq_max\s*=\s*([0-9]+)", "n_seq_max")
    unified = one(r"llama_context: kv_unified\s*=\s*(true|false)", "kv_unified")
    if slots is not None and int(slots) != parallel:
        errors.append(f"{name}: n_slots={slots}, expected {parallel}")
    if nseq is not None and int(nseq) != parallel:
        errors.append(f"{name}: n_seq_max={nseq}, expected {parallel}")
    if unified is not None and unified != "true":
        errors.append(f"{name}: --kv-unified did not produce kv_unified=true")
    disabled = re.search(r"KV paged metadata requires .*\(n_stream=([^,]+), v_trans=([^\)]+)\) - disabled", text)
    if disabled:
        missing.append(f"{name}: PAGED_LAYOUT_UNSUPPORTED n_stream={disabled.group(1)} v_trans={disabled.group(2)}")
    elif "KV paged metadata enabled (" not in text:
        missing.append(f"{name}: missing paged metadata enabled startup evidence")
    fallback = re.search(r"KV paged in-graph gather requires .*\(n_stream=([^,]+), v_trans=([^,]+), k_type=([^,]+), v_type=([^\)]+)\)", text)
    if fallback:
        missing.append(f"{name}: PAGED_INGRAPH_GATHER_UNSUPPORTED n_stream={fallback.group(1)} v_trans={fallback.group(2)} k_type={fallback.group(3)} v_type={fallback.group(4)}")
    stats = re.findall(r"KV paged metadata stats:.*ingraph_gather_layers=([0-9]+)", text)
    if not stats:
        missing.append(f"{name}: missing paged gather final stats")
    elif len(stats) != 1:
        errors.append(f"{name}: duplicate paged gather final stats")
    elif int(stats[0]) == 0:
        missing.append(f"{name}: paged gather never used")
    if enabled:
        action = re.findall(r"KV pressure unified action enabled: target_bytes=([0-9]+) max_blocks=([0-9]+)", text)
        if not action:
            missing.append(f"{name}: missing unified action enabled startup evidence")
        elif len(action) != 1:
            errors.append(f"{name}: duplicate unified action enabled startup evidence")
        elif action[0] != (str(env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES")), str(env.get("LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS"))):
            errors.append(f"{name}: unified action startup values differ from environment")


def validate_scores(marker: dict[str, str], errors: list[str]) -> set[str]:
    scores = marker["scores"]
    if scores == "none":
        return set()
    exclusions: set[str] = set()
    for part in scores.split(";"):
        fields = part.split(":")
        if len(fields) != 10 or not re.fullmatch(r"-?[0-9]+", fields[0]) or fields[1] not in {"0", "1"}:
            errors.append("malformed claimant score")
            continue
        exclusions.add(fields[2])
    return exclusions


def validate_markers(markers: list[dict[str, str]], parallel: int, errors: list[str], missing: list[str]) -> list[dict[str, str]]:
    if not markers:
        missing.append(f"parallel={parallel}: no real production governor markers")
        return []
    last_sample = last_decision = -1
    release_at, changed_offloads, exhausted, exclusions = [], [], [], set()
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
        exclusions |= validate_scores(marker, errors)
        if marker["decision_reason"] == "release_unsupported":
            missing.append(f"parallel={parallel}: UNIFIED_ACTION_UNSUPPORTED release_unsupported")
        if marker["release_attempted"] == "1":
            release_at.append(index)
            if marker["offload_armed_before"] != "0":
                errors.append(f"parallel={parallel} marker[{index}]: RELEASE after OFFLOAD arm")
            no_transaction_noop = marker["outcome"] == "no_op" and marker["state_changed"] == "0" and relief == 0
            if int(marker["transaction_id"]) == 0 and not no_transaction_noop:
                errors.append(f"parallel={parallel} marker[{index}]: RELEASE lacks transaction")
        if marker["offload_attempted"] == "1":
            seq, epoch = int(marker["selected_seq_id"]), int(marker["selected_claimant_epoch"])
            if marker["offload_armed_before"] != "1" or seq < 0 or epoch < 1 or int(marker["transaction_id"]) == 0:
                errors.append(f"parallel={parallel} marker[{index}]: invalid OFFLOAD claimant/transaction")
            if marker["state_changed"] == "1":
                if int(marker["blocks"]) < 2 or relief == 0:
                    errors.append(f"parallel={parallel} marker[{index}]: state-changing OFFLOAD is not multi-block/relieving")
                changed_offloads.append(marker)
            if marker["outcome"] == "no_op" and marker["reason"] == "no_candidate" and relief == 0:
                exhausted.append((index, seq, epoch))
    if not release_at:
        missing.append(f"parallel={parallel}: missing RELEASE-first decision")
    if not changed_offloads:
        missing.append(f"parallel={parallel}: missing state-changing multi-block OFFLOAD")
    elif release_at and min(index for index, marker in enumerate(markers) if marker in changed_offloads) <= release_at[0]:
        errors.append(f"parallel={parallel}: OFFLOAD preceded RELEASE")
    if "active" not in exclusions or "protected_sequence" not in exclusions:
        missing.append(f"parallel={parallel}: production scores lack active/protected exclusion evidence")
    if parallel == 3:
        if not exhausted:
            missing.append("parallel=3: missing claimant A no_candidate exhaustion")
        for index, seq, epoch in exhausted:
            later = [m for m in markers[index + 1:] if m["offload_attempted"] == "1"]
            if not any(int(m["selected_seq_id"]) != seq for m in later):
                errors.append(f"parallel=3: exhausted claimant {seq}/{epoch} did not advance to B")
            if not any(int(m["selected_seq_id"]) == seq and int(m["selected_claimant_epoch"]) > epoch for m in later):
                missing.append(f"parallel=3: missing reused claimant epoch after {seq}/{epoch}")
    return changed_offloads


def validate_resume(case: pathlib.Path, offloads: list[dict[str, str]], errors: list[str], missing: list[str], parallel: int) -> None:
    result, raw = read_json(case / "result.json"), (case / "server.stderr").read_bytes()
    start, end = result.get("reaccess_stderr_start"), result.get("reaccess_stderr_end")
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start or end > len(raw):
        missing.append(f"parallel={parallel}: missing valid scoped PREFETCH evidence")
        return
    events = token_lines(raw[start:end].decode(errors="replace"), RESUME_MARKER, parse_resume, str(case / "server.stderr"), errors)
    if not events:
        missing.append(f"parallel={parallel}: no scoped PREFETCH events")
        return
    for offload in offloads:
        key = (offload["selected_seq_id"], offload["selected_claimant_epoch"])
        found = False
        for index, event in enumerate(events):
            if event["phase"] != "prefetch" or (event["seq_id"], event["claimant_epoch"]) != key:
                continue
            if event["graph_allowed"] != "1":
                errors.append(f"parallel={parallel}: PREFETCH {key} blocked graph")
                continue
            if any(later["phase"] == "graph_gate" and all(later[field] == event[field] for field in RESUME_REQUIRED - {"phase"}) for later in events[index + 1:]):
                found = True
        if not found:
            missing.append(f"parallel={parallel}: OFFLOAD claimant {key} lacks ordered PREFETCH→graph gate")


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
    for name in ("INVALID_UNIFIED", "CONFLICT_UNIFIED_LEGACY"):
        result = read_json(cases[name] / "result.json")
        if not result.get("rejected_before_request_loop") or result.get("health_reached") or result.get("exit_code", 0) == 0:
            errors.append(f"parallel={parallel}: {name} did not reject before listener/request loop")
    executions = {name: read_json(cases[name] / "execution.json") for name in ("OFF", "GOVERNOR_ON")}
    envs = {name: read_json(cases[name] / "environment.json") for name in ("OFF", "GOVERNOR_ON")}
    off_argv, on_argv = normalize_argv(executions["OFF"].get("argv")), normalize_argv(executions["GOVERNOR_ON"].get("argv"))
    if off_argv is None or on_argv is None or off_argv != on_argv:
        errors.append(f"parallel={parallel}: OFF/GOVERNOR_ON argv differ")
    elif "--kv-unified" not in off_argv or "--parallel" not in off_argv or off_argv[off_argv.index("--parallel") + 1] != str(parallel):
        errors.append(f"parallel={parallel}: argv lacks exact --parallel/--kv-unified")
    if not isinstance(envs["OFF"], dict) or not isinstance(envs["GOVERNOR_ON"], dict):
        errors.append(f"parallel={parallel}: malformed environment record")
    else:
        changed = {key for key in set(envs["OFF"]) | set(envs["GOVERNOR_ON"]) if envs["OFF"].get(key) != envs["GOVERNOR_ON"].get(key)}
        if changed != GOVERNOR_ENV:
            errors.append(f"parallel={parallel}: OFF/GOVERNOR_ON environment differs outside unified action keys")
    off_rows, on_rows = keyed(rows(cases["OFF"] / "requests.jsonl"), f"parallel={parallel} OFF", parallel, errors), keyed(rows(cases["GOVERNOR_ON"] / "requests.jsonl"), f"parallel={parallel} GOVERNOR_ON", parallel, errors)
    for label in set(off_rows) | set(on_rows):
        if label not in off_rows or label not in on_rows:
            continue
        if off_rows[label].get("request") != on_rows[label].get("request") or off_rows[label].get("http_status") != 200 or on_rows[label].get("http_status") != 200 or off_rows[label].get("response_sha256") != on_rows[label].get("response_sha256"):
            errors.append(f"parallel={parallel}: {label} violates paired HTTP identity/correctness")
    off_text = (cases["OFF"] / "server.stderr").read_text(errors="replace")
    on_text = (cases["GOVERNOR_ON"] / "server.stderr").read_text(errors="replace")
    if token_lines(off_text, MARKER, parse_marker, str(cases["OFF"] / "server.stderr"), errors):
        errors.append(f"parallel={parallel}: OFF baseline contains Governor marker")
    startup(off_text, parallel, envs["OFF"], False, errors, missing, f"parallel={parallel} OFF")
    startup(on_text, parallel, envs["GOVERNOR_ON"], True, errors, missing, f"parallel={parallel} GOVERNOR_ON")
    offloads = validate_markers(token_lines(on_text, MARKER, parse_marker, str(cases["GOVERNOR_ON"] / "server.stderr"), errors), parallel, errors, missing)
    if parallel == 2:
        validate_resume(cases["GOVERNOR_ON"], offloads, errors, missing, parallel)


def main_parse(root: pathlib.Path) -> tuple[str, list[str]]:
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("protocol") != PROTOCOL or manifest.get("protocol_version") != 2:
        raise Error("protocol mismatch")
    errors: list[str] = []
    missing: list[str] = []
    for field in ("branch", "head", "dirty_status", "runner", "parser", "binary", "model", "parameters", "runs"):
        if field not in manifest:
            errors.append(f"manifest missing {field}")
    for field in ("runner", "parser", "binary", "model"):
        item = manifest.get(field)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or item.get("size", 0) <= 0 or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
            errors.append(f"manifest invalid identity {field}")
    if manifest.get("parameters", {}).get("parallels") != [2, 3] or manifest.get("parameters", {}).get("kv_unified") is not True:
        errors.append("manifest does not declare exact parallel=2/3 unified matrix")
    runs = manifest.get("runs")
    if not isinstance(runs, dict) or set(runs) != {"2", "3"}:
        errors.append("manifest missing parallel matrix")
    else:
        for parallel in (2, 3):
            if runs[str(parallel)].get("parallel") != parallel or set(runs[str(parallel)].get("cases", {})) != set(CASES):
                errors.append(f"manifest malformed parallel={parallel} run")
            validate_unit(root, parallel, errors, missing)
    if manifest.get("runner_status") == "UNSUPPORTED":
        missing.append(str(manifest.get("unsupported_reason", "unspecified")))
    elif manifest.get("runner_status") != "run_complete":
        return "UNVERIFIED", [f"runner status is {manifest.get('runner_status')!r}", *errors, *missing]
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
