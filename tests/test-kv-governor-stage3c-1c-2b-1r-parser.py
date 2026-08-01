#!/usr/bin/env python3
"""Fail-closed fixtures for the Stage 3C unified multi-slot parser."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PARSER = ROOT / "scripts/parse-kv-governor-stage3c-1c-2b-1r.py"
RUNNER = ROOT / "scripts/run-kv-governor-stage3c-1c-2b-1r.py"


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
        value = {"HOME": "/tmp", "LLAMA_KV_PAGED": "1", "LLAMA_KV_PAGED_INGRAPH": "1"}
        if enabled:
            value.update({"LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES": "1024", "LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS": "64"})
        return value

    def startup(self, parallel, enabled):
        text = (f"initializing slots, n_slots = {parallel}\n"
                f"llama_context: n_seq_max     = {parallel}\n"
                "llama_context: kv_unified    = true\n"
                "llama_kv_cache: KV paged metadata enabled (block_size=32, n_blocks=8, shift=0, non_identity=0, mapping_changed=0)\n")
        if enabled:
            text += "KV pressure unified action enabled: target_bytes=1024 max_blocks=64\n"
        return text

    def marker(self, decision, sample, *, release=0, offload=0, armed_before=0, armed_after=0, seq=-1, epoch=0, blocks=0, relief=0, before=1024, outcome="no_op", reason="no_candidate", scores="0:1:none:1:1:1:1:0:0:0;1:0:active:0:0:0:0:0:0:0;2:0:protected_sequence:0:0:0:0:0:0:0"):
        after = before - relief
        action = release or offload
        decision_reason = "release_submitted" if release else "offload_submitted" if offload else "release_unsupported"
        return ("kv_pressure_unified_action state=CRITICAL source=RSS_ABSOLUTE stale=0 "
                f"decision_id={decision} episode=1 target_bytes=1024 max_blocks=64 observed_excess_bytes=1024 debt_before_bytes={before} debt_after_bytes={after} "
                f"offload_armed_before={armed_before} offload_armed_after={armed_after} next_action_sample={sample + 1} evaluate_attempted=1 evaluate_outcome=completed evaluate_reason=none "
                f"release_attempted={release} offload_attempted={offload} selected_seq_id={seq} selected_claimant_epoch={epoch} transaction_id={decision if action else 0} "
                f"outcome={outcome} reason={reason} blocks={blocks} bytes={relief} relieved_bytes={relief} shortfall_bytes=0 io_failure=0 io_errno=0 state_changed={1 if relief else 0} "
                f"decision_reason={decision_reason} sample_count={sample} idle=1 scores={scores}\n")

    def events(self, seq, epoch):
        return (f"kv_resume_order_event phase=prefetch decision_id=99 seq_id={seq} claimant_epoch={epoch} transaction_id=7 action=prefetch outcome=completed reason=none graph_allowed=1\n"
                f"kv_resume_order_event phase=graph_gate decision_id=99 seq_id={seq} claimant_epoch={epoch} transaction_id=7 action=prefetch outcome=completed reason=none graph_allowed=1\n")

    def argv(self, parallel):
        return ["/bin/server", "--host", "127.0.0.1", "--port", "1", "--model", "/m", "--ctx-size", "2048", "--parallel", str(parallel), "--kv-unified", "--threads", "4", "--cache-type-k", "f32", "--cache-type-v", "f32"]

    def rows(self, parallel):
        result = []
        for cycle in range(2):
            for slot in range(parallel):
                result.append({"label": f"seed_c{cycle}_s{slot}", "request": {"id_slot": slot, "seed": 1}, "http_status": 200, "response_sha256": "same"})
        for slot in range(parallel):
            result.append({"label": f"reaccess_s{slot}", "request": {"id_slot": slot, "seed": 1}, "http_status": 200, "response_sha256": "same"})
            result.append({"label": f"reuse_c2_s{slot}", "request": {"id_slot": slot, "seed": 1}, "http_status": 200, "response_sha256": "same"})
        result.append({"label": f"active_s{parallel - 1}", "request": {"id_slot": parallel - 1, "seed": 1}, "http_status": 200, "response_sha256": "same"})
        return result

    def unit(self, parallel):
        root = self.d / f"parallel_{parallel}"
        root.mkdir()
        values = self.rows(parallel)
        for name, enabled in (("OFF", False), ("GOVERNOR_ON", True)):
            case = root / name
            case.mkdir()
            put(case / "execution.json", {"argv": self.argv(parallel)})
            put(case / "environment.json", self.env(enabled))
            (case / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
            (case / "server.stderr").write_text(self.startup(parallel, enabled) + "KV paged metadata stats: ingraph_gather_layers=4\n")
            put(case / "result.json", {"reaccess_stderr_start": 0, "reaccess_stderr_end": 0})
        gov = root / "GOVERNOR_ON"
        if parallel == 2:
            markers = (self.marker(1, 1, release=1, armed_after=1) +
                       self.marker(2, 2, offload=1, armed_before=1, armed_after=1, seq=0, epoch=1, blocks=2, relief=1024, outcome="completed", reason="none"))
            event = self.events(0, 1)
        else:
            markers = (self.marker(1, 1, release=1, armed_after=1) +
                       self.marker(2, 2, offload=1, armed_before=1, armed_after=1, seq=0, epoch=1) +
                       self.marker(3, 3, offload=1, armed_before=1, armed_after=1, seq=1, epoch=1, blocks=2, relief=1024, outcome="completed", reason="none") +
                       self.marker(4, 4, offload=1, armed_before=1, armed_after=1, seq=0, epoch=2, blocks=2, relief=1024, outcome="completed", reason="none"))
            event = ""
        text = self.startup(parallel, True) + markers + event + "KV paged metadata stats: ingraph_gather_layers=4\n"
        (gov / "server.stderr").write_text(text)
        start = len((self.startup(parallel, True) + markers).encode())
        put(gov / "result.json", {"reaccess_stderr_start": start, "reaccess_stderr_end": start + len(event.encode())})
        for name in ("INVALID_UNIFIED", "CONFLICT_UNIFIED_LEGACY"):
            case = root / name
            case.mkdir()
            put(case / "result.json", {"rejected_before_request_loop": True, "health_reached": False, "exit_code": 1})

    def valid(self):
        manifest = {"protocol": "kv_governor_stage3c_1c_2b_1r", "protocol_version": 2, "runner_status": "run_complete", "branch": "x", "head": "0" * 40, "dirty_status": [], "binary": {"path": "b", "size": 1, "sha256": "a" * 64}, "model": {"path": "m", "size": 1, "sha256": "b" * 64}, "runner": ident(RUNNER), "parser": ident(PARSER), "parameters": {"parallels": [2, 3], "kv_unified": True}, "runs": {str(p): {"parallel": p, "cases": {name: {"status": "complete"} for name in ("OFF", "GOVERNOR_ON", "INVALID_UNIFIED", "CONFLICT_UNIFIED_LEGACY")}} for p in (2, 3)}}
        put(self.d / "manifest.json", manifest)
        self.unit(2)
        self.unit(3)

    def parse(self):
        return subprocess.run([sys.executable, str(PARSER), str(self.d), "--result-path", str(self.d / "parser.json")], text=True, capture_output=True)

    def test_valid_passes(self):
        self.assertEqual(self.parse().returncode, 0)

    def test_parallel_three_missing_slot_fails(self):
        path = self.d / "parallel_3/GOVERNOR_ON/requests.jsonl"
        path.write_text("\n".join(line for line in path.read_text().splitlines() if "_s2\"" not in line) + "\n")
        self.assertEqual(self.parse().returncode, 1)

    def test_missing_kv_unified_fails(self):
        path = self.d / "parallel_2/OFF/execution.json"
        value = json.loads(path.read_text())
        value["argv"].remove("--kv-unified")
        put(path, value)
        self.assertEqual(self.parse().returncode, 1)

    def test_missing_capability_log_is_unsupported(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        path.write_text(path.read_text().replace("KV paged metadata stats: ingraph_gather_layers=4\n", ""))
        self.assertEqual(self.parse().returncode, 3)

    def test_n_stream_guard_is_unsupported(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        path.write_text(path.read_text().replace("KV paged metadata enabled (block_size=32, n_blocks=8, shift=0, non_identity=0, mapping_changed=0)", "KV paged metadata requires n_stream==1 && !v_trans (n_stream=2, v_trans=0) - disabled"))
        self.assertEqual(self.parse().returncode, 3)

    def test_output_mismatch_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/requests.jsonl"
        values = path.read_text().splitlines()
        row = json.loads(values[0]); row["response_sha256"] = "different"; values[0] = json.dumps(row)
        path.write_text("\n".join(values) + "\n")
        self.assertEqual(self.parse().returncode, 1)

    def test_bad_debt_fails(self):
        path = self.d / "parallel_2/GOVERNOR_ON/server.stderr"
        path.write_text(path.read_text().replace("debt_after_bytes=0", "debt_after_bytes=7", 1))
        self.assertEqual(self.parse().returncode, 1)

    def test_unverified_runner_fails(self):
        manifest = json.loads((self.d / "manifest.json").read_text())
        manifest["runner_status"] = "run_incomplete"
        put(self.d / "manifest.json", manifest)
        self.assertEqual(self.parse().returncode, 1)


if __name__ == "__main__":
    unittest.main()
