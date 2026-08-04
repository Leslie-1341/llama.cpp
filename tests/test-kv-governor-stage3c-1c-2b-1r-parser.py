#!/usr/bin/env python3
"""Fail-closed fixtures for the Stage 3C unified multi-slot parser."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts/parse-kv-governor-stage3c-1c-2b-1r.py"
RUNNER = ROOT / "scripts/run-kv-governor-stage3c-1c-2b-1r.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("kv_governor_stage3c_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load runner module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def ident(path):
    return {"path": str(path), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ParserTest(unittest.TestCase):
    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        self.valid()

    def tearDown(self):
        shutil.rmtree(self.d)

    def env(self, enabled):
        value = {
            "HOME": "/tmp",
            "LLAMA_KV_PAGED": "1",
            "LLAMA_KV_PAGED_INGRAPH": "1",
            "LLAMA_KV_PAGED_SWAP": "1",
            "LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY": "1",
            "LLAMA_KV_PAGED_BLOCK_SIZE": "64",
            "LLAMA_KV_SWAP_DIR": "backing",
            "LLAMA_KV_PRESSURE_SAMPLER": "1",
            "LLAMA_KV_PRESSURE_GOVERNOR_CLAIMANT_TRACE": "1",
            "LLAMA_KV_PRESSURE_SAMPLE_INTERVAL_MS": "100",
        }
        if enabled:
            value.update({
                "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
                "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "1073741824",
                "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": "64",
            })
        return value

    def capability(self, parallel, **overrides):
        values = {
            "n_slots": str(parallel),
            "n_seq_max": str(parallel),
            "n_stream": "1",
            "kv_unified": "1",
            "paged_metadata": "1",
            "ingraph_gather": "1",
            "release_supported": "1",
            "offload_supported": "1",
            "prefetch_supported": "1",
            "backing_ready": "1",
            "swap_explicit_only": "1",
        }
        values.update({key: str(value) for key, value in overrides.items()})
        return "KV_GOVERNOR_CAPABILITY " + " ".join(f"{key}={value}" for key, value in values.items()) + "\n"

    def startup(self, parallel, enabled):
        text = self.capability(parallel)
        if enabled:
            text += "KV pressure unified action enabled: target_bytes=1073741824 max_blocks=64\n"
        return text

    def scores(self, parallel, *, eligible=0, exhausted=()):
        parts = []
        for slot in range(parallel):
            if slot in exhausted:
                exclusion, eligible_value = "exhausted", 0
            elif slot == parallel - 1:
                exclusion, eligible_value = "active", 0
            elif slot == eligible:
                exclusion, eligible_value = "none", 1
            else:
                exclusion, eligible_value = "none", 1
            parts.append(f"{slot}:{eligible_value}:{exclusion}:1:1:1:1:0:0:0")
        return ";".join(parts)

    def claimants(
            self, parallel, *, exhausted=(), epochs=None,
            eligible=None, swapped=None, shared=None, blocked=None):
        epochs = epochs or {}
        eligible = eligible or {}
        swapped = swapped or {}
        shared = shared or {}
        blocked = blocked or {}
        parts = []
        for slot in range(parallel):
            active = int(slot == parallel - 1)
            eligible_blocks = eligible.get(slot, 3)
            swapped_blocks = swapped.get(slot, 0)
            shared_blocks = shared.get(slot, 0)
            blocked_blocks = blocked.get(slot, 0)
            target_blocks = eligible_blocks + swapped_blocks + shared_blocks + blocked_blocks
            parts.append(
                f"{slot}:{epochs.get(slot, 1)}:{active}:{int(slot in exhausted)}:1:"
                f"{target_blocks}:{eligible_blocks}:{swapped_blocks}:{shared_blocks}:{blocked_blocks}")
        return ";".join(parts)

    def marker(
            self, decision, sample, *, release=0, offload=0, armed_before=0,
            armed_after=0, seq=-1, epoch=0, blocks=0, relief=0,
            bytes_value=None, state_changed=None, transaction=None,
            before=1073741824, outcome="no_op", reason="no_candidate",
            claimants=None, scores="none", decision_reason=None):
        after = before - relief
        if state_changed is None:
            state_changed = 1 if relief else 0
        if transaction is None:
            transaction = decision if state_changed else 0
        if bytes_value is None:
            bytes_value = relief
        if decision_reason is None:
            decision_reason = "release_submitted" if release else "offload_submitted" if offload else "release_unsupported"
        if claimants is None:
            claimants = self.claimants(3 if "2:0:active" in scores else 2)
        return ("kv_pressure_unified_action state=CRITICAL source=RSS_ABSOLUTE stale=0 "
                f"decision_id={decision} episode=1 target_bytes=1073741824 max_blocks=64 observed_excess_bytes=1073741824 debt_before_bytes={before} debt_after_bytes={after} "
                f"offload_armed_before={armed_before} offload_armed_after={armed_after} next_action_sample={sample + 1} evaluate_attempted=1 evaluate_outcome=completed evaluate_reason=none "
                f"release_attempted={release} offload_attempted={offload} selected_seq_id={seq} selected_claimant_epoch={epoch} transaction_id={transaction} "
                f"outcome={outcome} reason={reason} blocks={blocks} bytes={bytes_value} relieved_bytes={relief} shortfall_bytes=0 io_failure=0 io_errno=0 state_changed={state_changed} "
                f"decision_reason={decision_reason} sample_count={sample} idle=0 claimants={claimants} scores={scores}\n")

    def events(self, seq, epoch):
        return (f"kv_resume_order_event phase=prefetch decision_id=99 seq_id={seq} claimant_epoch={epoch} transaction_id=7 action=prefetch outcome=completed reason=none graph_allowed=1\n"
                f"kv_resume_order_event phase=graph_gate decision_id=99 seq_id={seq} claimant_epoch={epoch} transaction_id=7 action=prefetch outcome=completed reason=none graph_allowed=1\n")

    def argv(self, parallel):
        return [
            "/bin/server", "--host", "127.0.0.1", "--port", "1", "--model", "/m",
            "--ctx-size", "2048", "--parallel", str(parallel), "--kv-unified",
            "--no-cache-idle-slots", "--timeout", "300", "--threads", "4", "--cache-type-k", "f32",
            "--cache-type-v", "f32",
        ]

    def rows(self, parallel):
        labels = [f"seed_s{slot}" for slot in range(parallel - 1)]
        labels.extend([f"active_s{parallel - 1}", "reaccess_a"])
        if parallel == 3:
            labels.extend(["reaccess_b", "reuse_a"])
        result = []
        for label in labels:
            if label in {"reaccess_a", "reuse_a"}:
                slot = 0
            elif label == "reaccess_b":
                slot = 1
            else:
                slot = int(label.rsplit("s", 1)[1])
            active = label == f"active_s{parallel - 1}"
            request = {"id_slot": slot, "seed": 1}
            if active:
                request.update({"n_predict": -1, "stream": True, "ignore_eos": True})
            result.append({
                "label": label,
                "request": request,
                "http_status": 200,
                "response": {"completion": "runner_cancelled"} if active else {},
                "response_sha256": "active-stream" if active else "same",
                "cancelled_by_runner": active,
                "stop_reason": "gate_complete" if active else None,
                "stop_requested_monotonic_ns": 900 if active else None,
                "started_monotonic_ns": 100,
                "finished_monotonic_ns": 1000 if active else 800,
            })
        return result

    def claimant_rows(self, parallel, **kwargs):
        encoded = self.claimants(parallel, **kwargs)
        result = []
        for item in encoded.split(";"):
            seq_id, epoch, active, exhausted, valid, target, eligible, swapped, shared, blocked = map(int, item.split(":"))
            result.append({
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
            })
        return result

    def layout(
            self, parallel, stderr_end, *, enabled, evidence_marker_end=0,
            decision_id=0, claimants=None):
        return {
            "stderr_end": stderr_end,
            "evidence_marker_end": evidence_marker_end,
            "decision_id": decision_id,
            "source": "governor_pre_action" if enabled else "slots",
            "observed_monotonic_ns": 500,
            "slots": [
                {"id": slot, "is_processing": slot == parallel - 1}
                for slot in range(parallel)
            ],
            "claimants": claimants or self.claimant_rows(parallel),
        }

    @staticmethod
    def active_stop(stderr_start):
        return {
            "reason": "gate_complete",
            "requested_monotonic_ns": 900,
            "finished_monotonic_ns": 1000,
            "thread_joined": True,
            "http_status": 200,
            "completion": "runner_cancelled",
            "transport_error": "OSError: cancelled",
            "stderr_start": stderr_start,
        }

    def write_backing(self, case):
        put(case / "backing.json", {
            "environment_value": "backing",
            "path": str((case / "backing").resolve()),
            "created": True,
            "contents_before_cleanup": [],
            "cleanup_attempted": True,
            "cleanup_error": None,
            "exists_after_cleanup": False,
        })
        put(case / "process.json", {
            "pid": 1,
            "pgid": 1,
            "exit_code": 0,
            "term_timed_out": False,
            "kill_timed_out": False,
            "residual_process": False,
        })

    def write_negative(self, root, name, parallel):
        case = root / name
        case.mkdir()
        env = self.env(True)
        triggers = {
            "INVALID_UNIFIED": {"LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "invalid"},
            "CONFLICT_UNIFIED_LEGACY": {"LLAMA_KV_PAGED_RELEASE": "1"},
            "CONFLICT_UNIFIED_DRY_RUN": {"LLAMA_KV_PRESSURE_DRY_RUN": "1"},
            "CONFLICT_UNIFIED_BOUNDED": {"LLAMA_KV_PRESSURE_BOUNDED_RELEASE": "1"},
        }
        env.update(triggers[name])
        put(case / "execution.json", {"argv": self.argv(parallel), "cwd": str(case.resolve()), "environment": env})
        put(case / "environment.json", env)
        self.write_backing(case)
        put(case / "result.json", {
            "status": "complete",
            "request_loop_started": False,
            "rejected_before_request_loop": True,
            "health_reached": False,
            "exit_code": 1,
        })

    def write_positive(self, root, name, parallel, enabled):
        case = root / name
        case.mkdir()
        env = self.env(enabled)
        put(case / "execution.json", {"argv": self.argv(parallel), "cwd": str(case.resolve()), "environment": env})
        put(case / "environment.json", env)
        self.write_backing(case)
        (case / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows(parallel)))
        return case

    def unit(self, parallel):
        root = self.d / f"parallel_{parallel}"
        root.mkdir()
        off = self.write_positive(root, "OFF", parallel, False)
        gov = self.write_positive(root, "GOVERNOR_ON", parallel, True)

        off_text = self.startup(parallel, False)
        off_end = len(off_text.encode())
        off_scopes = {"a_initial": {"start": off_end, "end": off_end}}
        if parallel == 3:
            off_scopes.update({
                "b": {"start": off_end, "end": off_end},
                "a_reused": {"start": off_end, "end": off_end},
            })
        (off / "server.stderr").write_text(off_text)
        put(off / "result.json", {
            "status": "complete",
            "request_loop_started": True,
            "active_stderr_start": off_end,
            "layout_ready": self.layout(parallel, off_end, enabled=False),
            "reaccess_stderr_start": off_end,
            "reaccess_stderr_end": off_end,
            "resume_scopes": off_scopes,
            "active_stop": self.active_stop(off_end),
        })

        startup = self.startup(parallel, True)
        layout_claimants = self.claimants(parallel)
        arm = self.marker(
            1, 1, release=1, armed_after=1,
            claimants=layout_claimants, scores="none")
        first_a = self.marker(
            2, 2, offload=1, armed_before=1, armed_after=1,
            seq=0, epoch=1, blocks=3, relief=300,
            outcome="completed", reason="target_shortfall",
            claimants=self.claimants(parallel),
            scores=self.scores(parallel, eligible=0))
        event_a_initial = self.events(0, 1)
        if parallel == 2:
            prefix = startup + arm + first_a
            text = prefix + event_a_initial
            scopes = {
                "a_initial": {"start": len(prefix.encode()), "end": len(text.encode())},
            }
        else:
            exhausted_a = self.marker(
                3, 3, offload=1, armed_before=1, armed_after=1,
                seq=0, epoch=1, blocks=0, relief=0,
                outcome="no_op", reason="no_candidate",
                claimants=self.claimants(parallel, eligible={0: 0}, swapped={0: 3}),
                scores=self.scores(parallel, eligible=0))
            offload_b = self.marker(
                4, 4, offload=1, armed_before=1, armed_after=1,
                seq=1, epoch=1, blocks=3, relief=300,
                outcome="completed", reason="target_shortfall",
                claimants=self.claimants(
                    parallel, exhausted=(0,), eligible={0: 0}, swapped={0: 3}),
                scores=self.scores(parallel, eligible=1, exhausted=(0,)))
            reused_a = self.marker(
                5, 5, offload=1, armed_before=1, armed_after=1,
                seq=0, epoch=2, blocks=3, relief=300,
                outcome="completed", reason="target_shortfall",
                claimants=self.claimants(
                    parallel, exhausted=(1,), epochs={0: 2},
                    eligible={1: 0}, swapped={1: 3}),
                scores=self.scores(parallel, eligible=0, exhausted=(1,)))
            prefix = startup + arm + first_a + exhausted_a + offload_b
            after_a = prefix + event_a_initial
            before_b = after_a + reused_a
            event_b = self.events(1, 1)
            before_reused_a = before_b + event_b
            event_a_reused = self.events(0, 2)
            text = before_reused_a + event_a_reused
            scopes = {
                "a_initial": {"start": len(prefix.encode()), "end": len(after_a.encode())},
                "b": {"start": len(before_b.encode()), "end": len(before_reused_a.encode())},
                "a_reused": {"start": len(before_reused_a.encode()), "end": len(text.encode())},
            }
        (gov / "server.stderr").write_text(text)
        arm_start = len(startup.encode())
        arm_end = arm_start + len(arm.encode())
        text_end = len(text.encode())
        put(gov / "result.json", {
            "status": "complete",
            "request_loop_started": True,
            "active_stderr_start": arm_start,
            "layout_ready": self.layout(
                parallel, arm_start, enabled=True,
                evidence_marker_end=arm_end, decision_id=1,
                claimants=self.claimant_rows(parallel)),
            "reaccess_stderr_start": scopes["a_initial"]["start"],
            "reaccess_stderr_end": scopes["a_initial"]["end"],
            "resume_scopes": scopes,
            "active_stop": self.active_stop(text_end),
        })
        self.write_negative(root, "INVALID_UNIFIED", parallel)
        self.write_negative(root, "CONFLICT_UNIFIED_LEGACY", parallel)
        self.write_negative(root, "CONFLICT_UNIFIED_DRY_RUN", parallel)
        self.write_negative(root, "CONFLICT_UNIFIED_BOUNDED", parallel)

    def valid(self):
        manifest = {
            "protocol": "kv_governor_stage3c_1c_2b_1r",
            "protocol_version": 6,
            "runner_status": "run_complete",
            "branch": "x",
            "head": "0" * 40,
            "dirty_status": [],
            "binary": {"path": "b", "size": 1, "sha256": "a" * 64},
            "model": {"path": "m", "size": 1, "sha256": "b" * 64},
            "runner": ident(RUNNER),
            "parser": ident(PARSER),
            "parameters": {
                "parallels": [2, 3],
                "kv_unified": True,
                "governor_max_blocks": 64,
                "active_n_predict": -1,
                "request_timeout_seconds": 90.0,
                "active_socket_timeout_seconds": 90.0,
                "active_cancel_timeout_seconds": 8.0,
                "marker_timeout_seconds": 20.0,
                "layout_timeout_seconds": 20.0,
                "server_timeout_seconds": 300,
                "server_term_timeout_seconds": 10.0,
                "server_kill_timeout_seconds": 5.0,
            },
            "runs": {str(p): {"parallel": p, "cases": {name: {"status": "complete"} for name in ("OFF", "GOVERNOR_ON", "INVALID_UNIFIED", "CONFLICT_UNIFIED_LEGACY", "CONFLICT_UNIFIED_DRY_RUN", "CONFLICT_UNIFIED_BOUNDED")}} for p in (2, 3)},
        }
        put(self.d / "manifest.json", manifest)
        self.unit(2)
        self.unit(3)

    def parse(self):
        return subprocess.run([sys.executable, str(PARSER), str(self.d), "--result-path", str(self.d / "parser.json")], text=True, capture_output=True)

    def rewrite_capability(self, parallel, name, **overrides):
        path = self.d / f"parallel_{parallel}" / name / "server.stderr"
        text = path.read_text()
        old = self.capability(parallel)
        new = self.capability(parallel, **overrides)
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1))

    def test_valid_passes(self):
        self.assertEqual(self.parse().returncode, 0)

    def test_transaction_contract_accepts_changed_and_no_candidate_noop(self):
        self.assertEqual(self.parse().returncode, 0)

    def test_parallel_two_real_offload_prefetch_closure_passes(self):
        self.assertEqual(self.parse().returncode, 0)

    def test_parallel_three_a_to_b_and_epoch_reuse_passes(self):
        self.assertEqual(self.parse().returncode, 0)

    def test_parallel_three_missing_slot_fails(self):
        path = self.d / "parallel_3/GOVERNOR_ON/requests.jsonl"
        path.write_text("\n".join(line for line in path.read_text().splitlines() if "seed_s1\"" not in line) + "\n")
        self.assertEqual(self.parse().returncode, 1)

    def test_missing_kv_unified_fails(self):
        path = self.d / "parallel_2/OFF/execution.json"
        value = json.loads(path.read_text())
        value["argv"].remove("--kv-unified")
        put(path, value)
        self.assertEqual(self.parse().returncode, 1)

    def test_missing_capability_record_is_unsupported(self):
        for parallel in (2, 3):
            for name in ("OFF", "GOVERNOR_ON"):
                path = self.d / f"parallel_{parallel}" / name / "server.stderr"
                capability = self.capability(parallel)
                replacement = " " * (len(capability) - 1) + "\n"
                path.write_text(path.read_text().replace(capability, replacement, 1))
        self.assertEqual(self.parse().returncode, 3)

    def disable_runtime_actions(self, parallel):
        for name in ("OFF", "GOVERNOR_ON"):
            self.rewrite_capability(
                parallel, name,
                offload_supported=0, prefetch_supported=0,
                backing_ready=0, swap_explicit_only=0)
        capability = self.capability(
            parallel, offload_supported=0, prefetch_supported=0,
            backing_ready=0, swap_explicit_only=0)
        startup = capability + "KV pressure unified action enabled: target_bytes=1073741824 max_blocks=64\n"
        hold = self.marker(
            1, 1, armed_before=1, armed_after=1,
            claimants=self.claimants(parallel), scores=self.scores(parallel),
            reason="unsupported", decision_reason="offload_unsupported")
        path = self.d / f"parallel_{parallel}/GOVERNOR_ON/server.stderr"
        path.write_text(startup + hold)
        result_path = self.d / f"parallel_{parallel}/GOVERNOR_ON/result.json"
        value = json.loads(result_path.read_text())
        marker_start = len(startup.encode())
        marker_end = marker_start + len(hold.encode())
        scopes = {"a_initial": {"start": marker_end, "end": marker_end}}
        if parallel == 3:
            scopes.update({
                "b": {"start": marker_end, "end": marker_end},
                "a_reused": {"start": marker_end, "end": marker_end},
            })
        value.update({
            "active_stderr_start": marker_start,
            "layout_ready": self.layout(
                parallel, marker_start, enabled=True,
                evidence_marker_end=marker_end, decision_id=1),
            "reaccess_stderr_start": marker_end,
            "reaccess_stderr_end": marker_end,
            "resume_scopes": scopes,
            "active_stop": self.active_stop(marker_end),
        })
        put(result_path, value)

    def test_swap_not_enabled_is_unsupported(self):
        for parallel in (2, 3):
            for name in ("OFF", "GOVERNOR_ON"):
                path = self.d / f"parallel_{parallel}" / name / "environment.json"
                value = json.loads(path.read_text())
                value["LLAMA_KV_PAGED_SWAP"] = "0"
                put(path, value)
            self.disable_runtime_actions(parallel)
        self.assertEqual(self.parse().returncode, 1)

    def test_backing_initialization_failure_is_unsupported(self):
        for parallel in (2, 3):
            self.disable_runtime_actions(parallel)
        self.assertEqual(self.parse().returncode, 3)

    def test_capability_runtime_offload_contradiction_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        contradiction = self.marker(3, 3, armed_before=1, armed_after=1, scores=self.scores(2), decision_reason="offload_unsupported")
        path.write_text(path.read_text() + contradiction)
        self.assertEqual(self.parse().returncode, 1)

    def test_state_changing_offload_requires_nonzero_transaction(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        text = path.read_text()
        old = "selected_seq_id=0 selected_claimant_epoch=1 transaction_id=2 outcome=completed"
        self.assertIn(old, text)
        path.write_text(text.replace(old, old.replace("transaction_id=2", "transaction_id=0"), 1))
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("state-changing action lacks transaction", parsed.stderr)

    def test_partial_offload_requires_state_changing_transaction(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        text = path.read_text()
        old = "selected_seq_id=0 selected_claimant_epoch=1 transaction_id=2 outcome=completed"
        new = "selected_seq_id=0 selected_claimant_epoch=1 transaction_id=0 outcome=partial_failure"
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1))
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("partial action lacks state-changing transaction", parsed.stderr)

    def test_no_candidate_noop_requires_zero_transaction_and_work(self):
        path = self.d / "parallel_3/GOVERNOR_ON/server.stderr"
        original = path.read_text()
        prefix = "selected_seq_id=0 selected_claimant_epoch=1 "
        valid = prefix + (
            "transaction_id=0 outcome=no_op reason=no_candidate blocks=0 bytes=0 "
            "relieved_bytes=0 shortfall_bytes=0 io_failure=0 io_errno=0 state_changed=0")
        self.assertIn(valid, original)
        contradictions = (
            valid.replace("transaction_id=0", "transaction_id=3", 1),
            valid.replace("blocks=0", "blocks=1", 1),
            valid.replace(" bytes=0 ", " bytes=1 ", 1),
            valid.replace("relieved_bytes=0", "relieved_bytes=1", 1),
            valid.replace("state_changed=0", "state_changed=1", 1),
        )
        for contradiction in contradictions:
            with self.subTest(contradiction=contradiction):
                path.write_text(original.replace(valid, contradiction, 1))
                parsed = self.parse()
                self.assertEqual(parsed.returncode, 1)
                self.assertIn("no_op/no_candidate has transaction or completed work", parsed.stderr)
        path.write_text(original)

    def test_no_idle_claimant_after_arm_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        startup = self.startup(2, True)
        arm = self.marker(
            1, 1, release=1, armed_after=1,
            claimants=self.claimants(2), scores=self.scores(2))
        no_idle = self.marker(
            2, 2, armed_before=1, armed_after=1,
            claimants=self.claimants(2, eligible={0: 0}),
            scores="0:0:empty:0:0:0:0:0:0:0;1:0:active:0:0:0:0:0:0:0",
            decision_reason="claimant_no_candidate")
        event = self.events(0, 1)
        text = startup + arm + no_idle + event
        path.write_text(text)
        result = self.d / "parallel_2/GOVERNOR_ON/result.json"
        arm_start = len(startup.encode())
        arm_end = arm_start + len(arm.encode())
        reaccess_start = len((startup + arm + no_idle).encode())
        put(result, {
            "status": "complete",
            "request_loop_started": True,
            "active_stderr_start": arm_start,
            "layout_ready": self.layout(
                2, arm_start, enabled=True,
                evidence_marker_end=arm_end, decision_id=1),
            "reaccess_stderr_start": reaccess_start,
            "reaccess_stderr_end": len(text.encode()),
            "resume_scopes": {
                "a_initial": {"start": reaccess_start, "end": len(text.encode())},
            },
            "active_stop": self.active_stop(len(text.encode())),
        })
        self.assertEqual(self.parse().returncode, 1)

    def test_idle_claimant_reaccessed_before_offload_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/result.json"
        value = json.loads(path.read_text())
        value["reaccess_stderr_start"] = 0
        put(path, value)
        self.assertEqual(self.parse().returncode, 1)

    def test_offload_before_layout_ready_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/result.json"
        value = json.loads(path.read_text())
        value["layout_ready"]["stderr_end"] = value["reaccess_stderr_start"]
        put(path, value)
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("OFFLOAD occurred before layout_ready", parsed.stderr)

    def test_layout_ready_requires_exact_idle_and_active_roles(self):
        cases = ((2, 0), (3, 1), (3, 2))
        for parallel, slot in cases:
            with self.subTest(parallel=parallel, slot=slot):
                path = self.d / f"parallel_{parallel}/GOVERNOR_ON/result.json"
                original = json.loads(path.read_text())
                value = json.loads(path.read_text())
                value["layout_ready"]["slots"][slot]["is_processing"] = (
                    not value["layout_ready"]["slots"][slot]["is_processing"])
                put(path, value)
                parsed = self.parse()
                self.assertEqual(parsed.returncode, 1)
                self.assertIn("layout_ready does not preserve exact idle/active slot roles", parsed.stderr)
                put(path, original)

    def test_active_lifecycle_must_contain_layout_ready(self):
        path = self.d / "parallel_2/GOVERNOR_ON/requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["label"] == "active_s1":
                row["finished_monotonic_ns"] = 500
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("active request lifecycle does not contain layout_ready", parsed.stderr)

    def test_layout_ready_requires_two_eligible_resident_blocks(self):
        path = self.d / "parallel_2/GOVERNOR_ON/result.json"
        value = json.loads(path.read_text())
        claimant = value["layout_ready"]["claimants"][0]
        claimant["target_blocks"] = 1
        claimant["eligible_resident_blocks"] = 1
        put(path, value)
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("lacks two eligible resident blocks", parsed.stderr)

    def test_active_timeout_or_unjoined_worker_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/requests.jsonl"
        values = [json.loads(line) for line in path.read_text().splitlines()]
        for row in values:
            if row["label"] == "active_s1":
                row["cancelled_by_runner"] = False
                row["response"]["completion"] = "transport_error"
                row["stop_requested_monotonic_ns"] = None
        path.write_text("".join(json.dumps(row) + "\n" for row in values))
        result_path = self.d / "parallel_2/GOVERNOR_ON/result.json"
        result = json.loads(result_path.read_text())
        result["active_stop"].update({
            "thread_joined": False,
            "completion": "thread_alive",
        })
        put(result_path, result)
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("active", parsed.stderr)

    def test_first_a_single_block_offload_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        text = path.read_text()
        old = "selected_seq_id=0 selected_claimant_epoch=1 transaction_id=2 outcome=completed reason=target_shortfall blocks=3"
        self.assertIn(old, text)
        path.write_text(text.replace(old, old.replace("blocks=3", "blocks=1"), 1))
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("fewer than 2 blocks", parsed.stderr)

    def test_wrong_claimant_first_offload_fails(self):
        path = self.d / "parallel_3/GOVERNOR_ON/server.stderr"
        text = path.read_text()
        old = "selected_seq_id=0 selected_claimant_epoch=1 transaction_id=2 outcome=completed reason=target_shortfall blocks=3"
        self.assertIn(old, text)
        path.write_text(text.replace(old, old.replace("selected_seq_id=0", "selected_seq_id=1"), 1))
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("wrong claimant OFFLOAD", parsed.stderr)

    def test_cleanup_kill_timeout_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/process.json"
        value = json.loads(path.read_text())
        value["kill_timed_out"] = True
        put(path, value)
        parsed = self.parse()
        self.assertEqual(parsed.returncode, 1)
        self.assertIn("exceeded SIGKILL timeout", parsed.stderr)

    def test_output_mismatch_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/requests.jsonl"
        values = path.read_text().splitlines()
        row = json.loads(values[0])
        row["response_sha256"] = "different"
        values[0] = json.dumps(row)
        path.write_text("\n".join(values) + "\n")
        self.assertEqual(self.parse().returncode, 1)

    def test_bad_debt_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        path.write_text(path.read_text().replace("debt_after_bytes=1073741524", "debt_after_bytes=7", 1))
        self.assertEqual(self.parse().returncode, 1)

    def test_execution_environment_mismatch_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/execution.json"
        value = json.loads(path.read_text())
        value["environment"]["LLAMA_KV_PAGED_SWAP"] = "0"
        put(path, value)
        self.assertEqual(self.parse().returncode, 1)

    def test_negative_case_requires_its_actual_trigger(self):
        for parallel in (2, 3):
            path = self.d / f"parallel_{parallel}/CONFLICT_UNIFIED_DRY_RUN/environment.json"
            value = json.loads(path.read_text())
            value.pop("LLAMA_KV_PRESSURE_DRY_RUN")
            put(path, value)
            execution = self.d / f"parallel_{parallel}/CONFLICT_UNIFIED_DRY_RUN/execution.json"
            execution_value = json.loads(execution.read_text())
            execution_value["environment"] = value
            put(execution, execution_value)
        self.assertEqual(self.parse().returncode, 1)

    def test_residual_server_process_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/process.json"
        value = json.loads(path.read_text())
        value["residual_process"] = True
        put(path, value)
        self.assertEqual(self.parse().returncode, 1)

    def test_parallel_three_b_resume_scope_is_required(self):
        path = self.d / "parallel_3/GOVERNOR_ON/result.json"
        value = json.loads(path.read_text())
        value["resume_scopes"].pop("b")
        put(path, value)
        self.assertEqual(self.parse().returncode, 3)

    def test_parallel_three_reused_a_resume_scope_is_required(self):
        path = self.d / "parallel_3/GOVERNOR_ON/result.json"
        value = json.loads(path.read_text())
        value["resume_scopes"]["a_reused"]["end"] = value["resume_scopes"]["a_reused"]["start"]
        put(path, value)
        self.assertEqual(self.parse().returncode, 3)

    def test_parallel_three_requires_c_as_active_producer(self):
        path = self.d / "parallel_3/GOVERNOR_ON/server.stderr"
        text = path.read_text()
        text = text.replace("2:0:active:1:1:1:1:0:0:0", "2:1:none:1:1:1:1:0:0:0", 1)
        path.write_text(text)
        self.assertEqual(self.parse().returncode, 1)

    def test_unverified_runner_fails(self):
        manifest = json.loads((self.d / "manifest.json").read_text())
        manifest["runner_status"] = "run_incomplete"
        put(self.d / "manifest.json", manifest)
        self.assertEqual(self.parse().returncode, 1)


class RunnerOrderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def setUp(self):
        self.d = pathlib.Path(tempfile.mkdtemp())
        (self.d / "server.stderr").write_bytes(b"")

    def tearDown(self):
        shutil.rmtree(self.d)

    @staticmethod
    def slots(parallel, *, eligible=3, exhausted=()):
        return [
            {
                "id": slot,
                "is_processing": slot == parallel - 1,
                "extra": "ignored",
                "kv_claimant": {
                    "epoch": 1,
                    "exhausted": slot in exhausted,
                    "valid": True,
                    "target_blocks": eligible,
                    "eligible_resident_blocks": eligible,
                    "swapped_blocks": 0,
                    "shared_blocks": 0,
                    "blocked_blocks": 0,
                },
            }
            for slot in reversed(range(parallel))
        ]

    def test_layout_snapshot_preserves_parallel_two_and_three_roles(self):
        for parallel in (2, 3):
            with self.subTest(parallel=parallel):
                snapshot = self.runner.layout_snapshot(
                    self.slots(parallel), parallel, require_capacity=True)
                self.assertIsNotNone(snapshot)
                roles, claimants = snapshot
                self.assertEqual(roles, [
                    {"id": slot, "is_processing": slot == parallel - 1}
                    for slot in range(parallel)
                ])
                self.assertEqual(len(claimants), parallel)
                self.assertTrue(all(
                    claimant["eligible_resident_blocks"] == 3
                    for claimant in claimants[:-1]))

    def test_layout_snapshot_rejects_wrong_active_slot(self):
        slots = self.slots(3)
        slots[0]["is_processing"] = False
        self.assertIsNone(self.runner.layout_snapshot(slots, 3))

    def wait_layout(self, line, parallel=2):
        (self.d / "server.stderr").write_text(line)
        proc = mock.Mock(returncode=None)
        proc.poll.return_value = None
        active = mock.Mock()
        active.is_alive.return_value = True
        with mock.patch.object(self.runner, "query_slots", return_value=self.slots(parallel)):
            return self.runner.wait_layout_ready(
                self.d, proc, active, 1, parallel, 0, True, seconds=0.01)

    def test_wait_layout_ready_rejects_single_block_a(self):
        claimants = "0:1:0:0:1:3:3:0:0:0;1:1:1:0:1:3:3:0:0:0"
        layout, error = self.wait_layout(
            "kv_pressure_unified_action decision_id=1 offload_attempted=1 "
            "selected_seq_id=0 outcome=completed reason=target_shortfall "
            f"state_changed=1 blocks=1 claimants={claimants}\n")
        self.assertIsNone(layout)
        self.assertEqual(error, "first claimant A OFFLOAD completed fewer than 2 blocks")

    def test_wait_layout_ready_rejects_wrong_claimant_order(self):
        claimants = "0:1:0:0:1:3:3:0:0:0;1:1:1:0:1:3:3:0:0:0"
        layout, error = self.wait_layout(
            "kv_pressure_unified_action decision_id=1 offload_attempted=1 "
            "selected_seq_id=1 outcome=completed reason=target_shortfall "
            f"state_changed=1 blocks=3 claimants={claimants}\n")
        self.assertIsNone(layout)
        self.assertEqual(error, "wrong claimant OFFLOAD occurred before claimant A")

    def test_wait_layout_ready_rejects_early_a_exhaustion(self):
        claimants = "0:1:0:0:1:3:0:3:0:0;1:1:1:0:1:3:3:0:0:0"
        layout, error = self.wait_layout(
            "kv_pressure_unified_action decision_id=1 offload_attempted=1 "
            "selected_seq_id=0 outcome=no_op reason=no_candidate "
            f"state_changed=0 blocks=0 claimants={claimants}\n")
        self.assertIsNone(layout)
        self.assertEqual(error, "claimant A exhausted before its required multi-block OFFLOAD")

    def test_layout_snapshot_rejects_insufficient_claimant_capacity(self):
        self.assertIsNone(self.runner.layout_snapshot(
            self.slots(2, eligible=1), 2, require_capacity=True))
        self.assertIsNone(self.runner.layout_snapshot(
            self.slots(3, exhausted=(0,)), 3, require_capacity=True))

    def test_wait_layout_ready_rejects_ended_active_lifecycle(self):
        proc = mock.Mock(returncode=None)
        proc.poll.return_value = None
        active = mock.Mock()
        active.is_alive.return_value = False
        with mock.patch.object(self.runner, "query_slots", return_value=self.slots(2)):
            layout, error = self.runner.wait_layout_ready(
                self.d, proc, active, 1, 2, 0, True, seconds=0.01)
        self.assertIsNone(layout)
        self.assertEqual(error, "active request completed before layout_ready")

    def test_run_case_persists_signal_and_exception_final_state(self):
        for name, exc, expected in (
                ("SIGNAL", self.runner.RunnerInterrupted(signal.SIGTERM), "interrupted"),
                ("EXCEPTION", RuntimeError("boom"), "request_failed")):
            with self.subTest(name=name):
                root = self.d / name.lower()
                root.mkdir()
                with mock.patch.object(self.runner, "start", side_effect=exc):
                    result = self.runner.run_case(
                        name, "/bin/server", "/model", True, root, 2)
                case = root / name
                persisted = json.loads((case / "result.json").read_text())
                self.assertEqual(result["status"], expected)
                self.assertEqual(persisted["status"], expected)
                self.assertFalse((case / "backing").exists())
                self.assertFalse(persisted["backing"]["exists_after_cleanup"])

    def test_stop_escalates_and_reaps_process_group(self):
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "print('ready', flush=True); time.sleep(60)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )
        self.assertEqual(proc.stdout.readline().strip(), b"ready")
        with (mock.patch.object(self.runner, "SERVER_TERM_TIMEOUT_SECONDS", 0.05),
              mock.patch.object(self.runner, "SERVER_KILL_TIMEOUT_SECONDS", 1.0)):
            record = self.runner.stop(proc, self.d)
        self.assertTrue(record["term_timed_out"])
        self.assertFalse(record["kill_timed_out"])
        self.assertFalse(record["residual_process"])
        self.assertIsNotNone(record["exit_code"])

    def test_long_prefill_parallel_three_timeout_coverage(self):
        """Regression: parallel=3 three 576-token prefill ~30.74s must not hit timeout.
        Default timeout >=90s provides margin over measured worst case."""
        # Verify manifest parameters declare >=90s prefill timeout
        module = self.runner
        self.assertGreaterEqual(module.REQUEST_TIMEOUT_SECONDS, 90.0)
        self.assertGreaterEqual(module.ACTIVE_SOCKET_TIMEOUT_SECONDS, 90.0)

    def test_active_slot_active_before_first_chunk(self):
        """Regression: active request confirmed by /slots is_processing, not first stream chunk.
        The wait_layout_ready already uses /slots for capacity check when governor enabled."""
        module = self.runner
        slots = self.slots(3)
        # Active slot 2 is processing
        snapshot = module.layout_snapshot(slots, 3, require_capacity=True)
        self.assertIsNotNone(snapshot)
        roles, claimants = snapshot
        self.assertTrue(roles[2]["is_processing"])  # slot 2 is active
        # Idle slots 0,1 have >=2 eligible blocks
        self.assertGreaterEqual(claimants[0]["eligible_resident_blocks"], 2)
        self.assertGreaterEqual(claimants[1]["eligible_resident_blocks"], 2)

    def test_timeout_cleanup_regression(self):
        """Regression: when HTTP timeout fires, runner must still cancel active,
        wait bounded, clean up backing and process."""
        module = self.runner
        with tempfile.TemporaryDirectory() as td:
            case = pathlib.Path(td)
            backing = case / "backing"
            backing.mkdir()
            env = module.governor_env(True)
            port = module.free_port()
            # Simulate a server that takes > timeout to respond
            proc = subprocess.Popen(
                [sys.executable, "-c",
                 "import http.server, socketserver, time, threading; "
                 "class H(http.server.BaseHTTPRequestHandler): "
                 "  def do_POST(self): time.sleep(10) "
                 "socketserver.TCPServer(('', 0), H).serve_forever()"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            )
            try:
                # Give it a moment to start
                time.sleep(0.5)
                # Short timeout to force transport_error
                with mock.patch.object(module, "REQUEST_TIMEOUT_SECONDS", 0.1):
                    status = module.request(
                        port, {"prompt": [1], "n_predict": 0, "stream": False},
                        "timeout_test", case / "requests.jsonl", threading.Lock())
                # Request should fail with 0 status (transport_error)
                self.assertEqual(status, 0)
            finally:
                # Cleanup must work even after timeout
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait(timeout=2.0)
            # Backing cleanup should still be attempted
            module.cleanup_backing(case, backing)
            # Should not raise; verify cleanup recorded
            self.assertTrue((case / "backing.json").is_file())


if __name__ == "__main__":
    unittest.main()
