#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit/fixture tests for GT-trace-1A Alibaba Trace Workload Compiler.

Covers the 10 required cases from the gate contract:
  1. lineage reconstruction (chain parent_chat_id == prev chat_id)
  2. parent/turn anomalies fail-closed
  3. no-oracle TTL: timer / cancel / cold-restart (no final-turn pre-release)
  4. calibration/evaluation no-leak (time-ordered, whole-session)
  5. complete-session slice (no drop-turn; whole sessions only)
  6. deterministic output hash (event-stream SHA stable across re-runs)
  7. Trace B produces NO pseudo multi-turn reuse
  8. session-namespaced hash mapping (intra same, cross different)
  9. future-field guard (offline fields never enter runtime event feature)
 10. streaming/bounded reader provenance (SHA + count)

Runs from the repo root or the scripts dir; no model, no network.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
import sys
import tempfile
import unittest

# Make `trace_compiler` importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.normpath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from trace_compiler import common, materialize, ttl_reactor, lineage, calibration
from trace_compiler.sampler import (
    SLICE_BURST_LONG_CTX,
    SLICE_REVISIT_HEAVY,
    SLICE_TYPICAL,
    assemble_slice,
    load_event_stream,
    select_revisit_heavy,
    select_typical,
    validate_persisted_event_stream,
)
from trace_compiler.cli import compile_trace
from trace_compiler.common import assert_no_future_fields


def _record(rec_id, parent, ts, turn, inl=40, outl=8, t="text", hids=None):
    return {
        "chat_id": rec_id,
        "parent_chat_id": parent,
        "timestamp": ts,
        "input_length": inl,
        "output_length": outl,
        "type": t,
        "turn": turn,
        "hash_ids": hids if hids is not None else list(range(inl // 16 or 1)),
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _records_to_lineages(records):
    """Build Lineage list from raw dicts (unit-fixture convenience)."""
    trs = [common.TraceRecord.from_raw(r) for r in records]
    accepted, rejected, stats = lineage.reconstruct_lineages(trs)
    return accepted, rejected, stats


# ---------------------------------------------------------------------------
class LineageReconstructionTest(unittest.TestCase):
    def test_chain_lineage_grouped(self):
        # chain: root 100 -> 101 -> 102 ; plus single 200 ; plus orphan with bad parent
        recs = [
            _record(100, -1, 0.0, 1, hids=[0, 1, 2, 3]),
            _record(101, 100, 5.0, 2, hids=[0, 1, 2, 7]),
            _record(102, 101, 9.0, 3, hids=[0, 1, 2, 8]),
            _record(200, -1, 1.0, 1, hids=[50, 51]),
        ]
        acc, rej, stats = _records_to_lineages(recs)
        self.assertEqual(len(acc), 2)
        by_id = {l.lineage_id: l for l in acc}
        self.assertEqual(by_id[100].turn_count, 3)
        self.assertEqual([t.chat_id for t in by_id[100].turns], [100, 101, 102])
        self.assertEqual([t.turn for t in by_id[100].turns], [1, 2, 3])
        self.assertTrue(by_id[100].is_accepted)
        self.assertEqual(by_id[200].turn_count, 1)


class FailClosedParentTurnTest(unittest.TestCase):
    def test_orphan_parent_fail_closed(self):
        # turn-2 record whose parent is absent from the file never forms a
        # bad accepted lineage; its lineage-building path only roots parents=-1.
        recs = [
            _record(100, -1, 0.0, 1),
            # turn2 with parent 999 (absent)
            _record(101, 999, 1.0, 2),
        ]
        acc, rej, stats = _records_to_lineages(recs)
        # 101 should not be glued onto 100 (different parent chain).
        self.assertEqual(len(acc), 1)
        self.assertEqual(acc[0].turn_count, 1)
        # Orphan interface: count_orphans on the minimal index.
        receiver = lineage.reconstruct_lineages([common.TraceRecord.from_raw(r) for r in recs])
        minimal = {}
        # rebuild a minimal index to feed count_orphans.
        trs = [common.TraceRecord.from_raw(r) for r in recs]
        # reconstruct_lineages doesn't return the index; we test orphan count
        # via a direct helper on a hand-built index below.
        idx = {
            r.chat_id: lineage.MinimalTurn(
                r.chat_id, r.parent_chat_id, r.timestamp, r.turn,
                r.input_length, r.output_length, r.type, r.block_count,
            )
            for r in trs
        }
        self.assertEqual(lineage.count_orphans(idx), 1)

    def test_broken_continuity_rejected(self):
        # root + turn3 (turn 2 missing) -> broken continuity
        recs = [
            _record(100, -1, 0.0, 1),
            _record(102, 100, 9.0, 3),  # gap: turn 2 absent
        ]
        acc, rej, _ = _records_to_lineages(recs)
        self.assertEqual(len(acc), 0)
        self.assertEqual(len(rej), 1)
        self.assertIn("continuity", rej[0].rejected_reason)

    def test_nonmonotonic_ts_rejected(self):
        recs = [
            _record(100, -1, 5.0, 1),
            _record(101, 100, 0.0, 2),  # ts < parent (but >= 0)
        ]
        acc, rej, _ = _records_to_lineages(recs)
        self.assertEqual(len(acc), 0)
        self.assertEqual(len(rej), 1)
        self.assertIn("non-monotonic timestamp", rej[0].rejected_reason)

    def test_root_with_turn_not_1_rejected_at_parse(self):
        with self.assertRaises(common.TraceRecordError):
            common.TraceRecord.from_raw(_record(100, -1, 0.0, 2))


class NoOracleTTLTest(unittest.TestCase):
    def _mk(self, turn, ts):
        return lineage.MinimalTurn(
            chat_id=500 + turn, parent_chat_id=-1, timestamp=ts, turn=turn,
            input_length=10, output_length=5, type="text", block_count=1,
        )

    def test_revisit_within_ttl_cancels_timer(self):
        arrivals = [(0.0, 1, 1, self._mk(1, 0.0)),
                    (5.0, 1, 2, self._mk(2, 5.0))]
        r = ttl_reactor.TTLReactor(15.0)
        fed_types = [e.event_type.value for e in r.feed(arrivals)]
        # gate-1A emits ONLY TURN_START (no synthetic TURN_COMPLETE).
        # REVISIT must fire when turn2 arrives before the armed TTL (15) elapses.
        self.assertEqual(fed_types, ["TURN_START", "REVISIT", "TURN_START"])
        # NO DEAD appeared while the lineage was still being fed: the final
        # turn's DEAD comes only from the TTL elapsing during drain, never
        # from any "final turn" oracle.
        self.assertNotIn("DEAD", fed_types)
        self.assertNotIn("TURN_COMPLETE", fed_types,
                         "gate 1A emits ONLY TURN_START; TURN_COMPLETE is deferred to 1B")
        drained = [e.event_type.value for e in r.drain()]
        # Drain fires the final turn's TTL expiry -> exactly one DEAD.
        self.assertEqual(drained, ["TTL_EXPIRY", "DEAD"])

    def test_no_final_turn_oracle(self):
        # single turn: must not be DEAD instantly; only via TTL elapse.
        arrivals = [(0.0, 1, 1, self._mk(1, 0.0))]
        r = ttl_reactor.TTLReactor(10.0)
        fed = [e.event_type.value for e in r.feed(arrivals)]
        self.assertEqual(fed, ["TURN_START"],
                         "no DEAD without time passing -> no oracle on 'final turn'")
        self.assertNotIn("TURN_COMPLETE", fed,
                         "gate 1A emits ONLY TURN_START; TURN_COMPLETE is deferred to 1B")
        drained = [e.event_type.value for e in r.drain()]
        self.assertEqual(drained, ["TTL_EXPIRY", "DEAD"])

    def test_cold_restart_after_dead(self):
        arrivals = [
            (0.0, 1, 1, self._mk(1, 0.0)),
            (20.0, 1, 2, self._mk(2, 20.0)),  # 20 > TTL=10 -> DEAD before turn2
        ]
        r = ttl_reactor.TTLReactor(10.0)
        events = list(r.feed(arrivals))  # feed ONCE; repeated feed is invalid API use
        types = [e.event_type.value for e in events]
        self.assertIn("DEAD", types)
        self.assertIn("COLD_RESTART", types)
        # generation incremented exactly once on the cold restart (gen 1 -> 2)
        gens = [e.lifecycle_generation for e in events if e.event_type.value == "COLD_RESTART"]
        self.assertEqual(gens, [2])


class CalEvalNoLeakTest(unittest.TestCase):
    def test_time_ordered_whole_session_split(self):
        acc = []
        # 6 lineages, root ts in [0..5]; split 0.5 -> roots 0..2 cal, 3..5 eval
        for i in range(6):
            mt1 = lineage.MinimalTurn(i, -1, float(i), 1, 10, 5, "text", 1)
            acc.append(lineage.Lineage(lineage_id=i, turns=[mt1]))
        sp = calibration.make_time_split(acc, 0.5)
        self.assertEqual(sp.calibration_session_count, 3)
        self.assertEqual(sp.evaluation_session_count, 3)
        self.assertTrue(acc[0].turns[0].timestamp <= sp.calibration_until_ts)
        self.assertTrue(acc[5].turns[0].timestamp > sp.calibration_until_ts)
        # whole-session bucket: a lineage straddling the cut (some turns in
        # the cal window, others in the eval window) is boundary_excluded and
        # MUST NOT enter either calibration or evaluation.
        multi = lineage.Lineage(lineage_id=99, turns=[
            lineage.MinimalTurn(99, -1, 1.0, 1, 10, 5, "text", 1),
            lineage.MinimalTurn(100, 99, 999999.0, 2, 20, 5, "text", 1),
        ])
        sp2 = calibration.make_time_split(acc + [multi], 0.5)
        self.assertEqual(
            calibration.assign_lineage_to_split(multi, sp2),
            "boundary_excluded",
            "straddling lineage must be excluded from both cal and eval",
        )


class CompleteSessionSliceTest(unittest.TestCase):
    def test_whole_session_no_turn_drop(self):
        # 5 multi-turn lineages, each 4 turns
        acc = []
        for lid in range(5):
            turns = []
            for k in range(4):
                turns.append(lineage.MinimalTurn(
                    lid * 100 + k, -1 if k == 0 else lid * 100 + k - 1,
                    float(k), k + 1, 40 + 10 * k, 5, "text", 1
                ))
            acc.append(lineage.Lineage(lineage_id=lid, turns=turns))
        sel = select_typical(acc, seed=7, n=3)
        # every selected session keeps all 4 turns; none dropped
        for lin in sel:
            self.assertEqual(lin.turn_count, 4)
        self.assertLessEqual(len(sel), 3)


class DeterministicOutputHashTest(unittest.TestCase):
    def test_event_stream_sha_stable(self):
        acc = []
        for lid in range(4):
            turns = [lineage.MinimalTurn(
                lid * 100 + k, -1 if k == 0 else lid * 100 + k - 1,
                float(k) + lid, k + 1, 40, 5, "text", 1)
                for k in range(3)]
            acc.append(lineage.Lineage(lineage_id=lid, turns=turns))
        frozen = calibration.build_frozen_calibration(acc, calibration.make_time_split(acc, 0.5), 0.9)
        m1 = assemble_slice(
            slice_class=SLICE_TYPICAL, slice_lineages=acc[:2],
            trace_key="traceA", file_sha256="deadbeef",
            source_window_ts=(0.0, 10.0), selection_rule="r", seed=1,
            time_dilation=1.0, frozen_cal=frozen,
        )
        m2 = assemble_slice(
            slice_class=SLICE_TYPICAL, slice_lineages=acc[:2],
            trace_key="traceA", file_sha256="deadbeef",
            source_window_ts=(0.0, 10.0), selection_rule="r", seed=1,
            time_dilation=1.0, frozen_cal=frozen,
        )
        self.assertEqual(m1.event_stream_sha256, m2.event_stream_sha256)
        self.assertTrue(m1.session_count >= 1)


class TraceBNoPseudoReuseTest(unittest.TestCase):
    def test_traceB_synthetic_all_single_turn(self):
        recs = [_record(i, -1, float(i), 1, inl=100, outl=10, t="api", hids=list(range(i, i + 6))) for i in range(50)]
        with tempfile.TemporaryDirectory() as d:
            _write_jsonl(os.path.join(d, "qwen_traceB_blksz_16.jsonl"), recs)
            old_trace_dir = common.TRACE_REPO_DIR
            common.TRACE_REPO_DIR = d
            base = tempfile.mkdtemp()
            try:
                r = compile_trace("traceB", base, cal_fraction=0.5, nslice_typical=5, nslice_revisit=5, nslice_burst=3, seed=2)
            finally:
                common.TRACE_REPO_DIR = old_trace_dir
        # Trace B all turn=1 -> no revisit, dead==all cal turns, revisit-heavy empty.
        self.assertEqual(r["cache_characterization"]["revisit_events"], 0)
        self.assertEqual(r["cache_characterization"]["cold_restart_events"], 0)
        self.assertEqual(r["manifests"]["revisit-heavy"]["session_count"], 0,
                         "Trace B MUST NOT be packaged as multi-turn reuse evidence")


class SessionNamespacedHashTest(unittest.TestCase):
    def test_intra_same_cross_different(self):
        m = materialize.Materializer()
        b1 = m.materialize_block(1, 42)
        b1b = m.materialize_block(1, 42)
        b2 = m.materialize_block(2, 42)
        self.assertEqual(b1.token_ids, b1b.token_ids, "same session same hash -> same block")
        self.assertNotEqual(b1.token_ids, b2.token_ids, "different session -> namespaced different")
        self.assertEqual(materialize.BLOCK_SIZE_TOKENS, len(b1.token_ids))


class FutureFieldGuardTest(unittest.TestCase):
    def test_no_future_fields_in_events(self):
        acc = [lineage.Lineage(lineage_id=1, turns=[
            lineage.MinimalTurn(1, -1, 0.0, 1, 40, 5, "text", 1),
            lineage.MinimalTurn(2, 1, 5.0, 2, 45, 5, "text", 2),
        ])]
        frozen = calibration.build_frozen_calibration(acc, calibration.make_time_split(acc, 0.5), 0.9)
        m = assemble_slice(
            slice_class=SLICE_TYPICAL, slice_lineages=acc,
            trace_key="traceA", file_sha256="x",
            source_window_ts=(0.0, 5.0), selection_rule="r", seed=1,
            time_dilation=1.0, frozen_cal=frozen,
            hash_provider=lambda lid, turn: tuple([0, 1, 2, 3]),
        )
        # assert_no_future_fields already invoked in to_runtime_contract_dict;
        # double-check no offline field name in the manifest dump.
        blob = m.to_canonical_json()
        for f in common.OFFLINE_GROUND_TRUTH_FIELDS:
            self.assertNotIn(f, blob)

    def test_guard_raises(self):
        with self.assertRaises(AssertionError):
            assert_no_future_fields({"is_final_turn_of_lineage": True})


class StreamingReaderProvenanceTest(unittest.TestCase):
    def test_sha_and_count(self):
        recs = [_record(i, -1, float(i), 1) for i in range(10)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "qwen_traceA_blksz_16.jsonl")
            _write_jsonl(p, recs)
            rs = lineage_replay(p)
        self.assertEqual(rs.provenance.record_count, 10)
        self.assertEqual(len(rs.provenance.raw_byte_sha256), 64)
        self.assertEqual(rs.provenance.first_timestamp, 0.0)

    def test_record_count_not_all_loaded(self):
        # We cannot easily assert memory from a unittest, but we assert that
        # the iterator yields lazily (next() works one at a time without
        # exhausting). At minimum, the reader exposes a generator interface.
        recs = [_record(i, -1, float(i), 1) for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "qwen_traceA_blksz_16.jsonl")
            _write_jsonl(p, recs)
            from trace_compiler.reader import TraceReadStream
            it = iter(TraceReadStream("traceA", p))
            first = next(it)
            self.assertEqual(first.chat_id, 0)
            # remaining not yet materialized
            rest = list(it)
            self.assertEqual(len(rest), 3)


class PersistedWorkloadIntegrationTest(unittest.TestCase):
    @staticmethod
    def _mini_records():
        records = []
        turn_specs = ((160, 1), (16, 100), (32, 2))
        for session in range(6):
            root = 1000 + session * 10
            for turn, (input_length, output_length) in enumerate(turn_specs, start=1):
                chat_id = root + turn - 1
                parent = -1 if turn == 1 else chat_id - 1
                records.append(_record(
                    chat_id,
                    parent,
                    float(root + turn - 1),
                    turn,
                    inl=input_length,
                    outl=output_length,
                    hids=list(range(session * 100 + turn, session * 100 + turn + input_length // 16)),
                ))
        return records

    def test_compile_persist_reload_contract(self):
        records = self._mini_records()
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as out_dir:
            _write_jsonl(
                os.path.join(source_dir, "qwen_traceA_blksz_16.jsonl"), records
            )
            old_trace_dir = common.TRACE_REPO_DIR
            common.TRACE_REPO_DIR = source_dir
            try:
                report = compile_trace(
                    "traceA",
                    out_dir,
                    cal_fraction=0.5,
                    nslice_typical=2,
                    nslice_revisit=2,
                    nslice_burst=1,
                    seed=9,
                )
            finally:
                common.TRACE_REPO_DIR = old_trace_dir

            report_path = os.path.join(out_dir, "traceA", "report.json")
            with open(report_path, "r", encoding="utf-8") as f:
                persisted_report = json.load(f)
            self.assertEqual(
                persisted_report["runtime_event_scope"],
                "arrival_and_request_plan_only",
            )
            self.assertEqual(
                persisted_report["offline_ttl_replay_scope"],
                "offline_projected_characterization_only",
            )
            self.assertEqual(
                persisted_report["cache_characterization"]["semantics"],
                "offline_projected_characterization_only",
            )

            for slice_name, summary in report["manifests"].items():
                manifest_path = os.path.join(
                    out_dir, "traceA", f"manifest_{slice_name}.json"
                )
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                event_path = os.path.join(
                    out_dir,
                    "traceA",
                    {
                        "typical": "events_typical.jsonl",
                        "revisit-heavy": "events_revisit-heavy.jsonl",
                        "burst-long-context": "events_burst-long-context.jsonl",
                    }[slice_name],
                )
                with open(event_path, "r", encoding="utf-8") as f:
                    raw_stream = f.read()
                events, reload_sha = load_event_stream(event_path)
                self.assertEqual(
                    hashlib.sha256(raw_stream.encode("utf-8")).hexdigest(),
                    manifest["event_stream_sha256"],
                )
                self.assertEqual(reload_sha, manifest["event_stream_sha256"])
                self.assertEqual(len(events), manifest["event_count"])
                self.assertEqual(len(events), manifest["turn_count"])
                validate_persisted_event_stream(events)
                self.assertNotIn("stratified", manifest["selection_rule"])
                self.assertEqual(
                    manifest["runtime_event_scope"],
                    "arrival_and_request_plan_only",
                )
                self.assertEqual(
                    manifest["ttl_replay_scope"],
                    "offline_projected_characterization_only",
                )

                required_contexts = []
                target_lengths = []
                seen_turns = set()
                for event in events:
                    self.assertEqual(event["event_type"], "TURN_START")
                    self.assertNotIn("TURN_COMPLETE", event)
                    self.assertNotIn("hash_ids", event)
                    self.assertNotIn("output_length", event["feature"])
                    self.assertNotIn("target_output_length", event["feature"])
                    self.assertNotIn("n_predict", event["feature"])
                    self.assertNotIn("required_context_tokens", event["feature"])
                    key = (event["lineage_id"], event["turn"])
                    self.assertNotIn(key, seen_turns)
                    seen_turns.add(key)
                    feature = event["feature"]
                    plan = event["request_plan"]
                    self.assertEqual(
                        len(feature["session_namespaced_block_ids"]),
                        feature["block_count"],
                    )
                    self.assertEqual(
                        plan["n_predict"], plan["target_output_length"]
                    )
                    self.assertEqual(
                        plan["required_context_tokens"],
                        feature["input_length"] + plan["target_output_length"],
                    )
                    required_contexts.append(plan["required_context_tokens"])
                    target_lengths.append(plan["target_output_length"])
                self.assertEqual(
                    manifest["max_required_context_tokens"], max(required_contexts, default=0)
                )
                self.assertGreater(len(set(target_lengths)), 1)
                self.assertEqual(summary["event_count"], summary["turn_count"])

    def test_frozen_reload_does_not_need_source_trace(self):
        records = self._mini_records()
        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as out_dir:
            _write_jsonl(
                os.path.join(source_dir, "qwen_traceA_blksz_16.jsonl"), records
            )
            old_trace_dir = common.TRACE_REPO_DIR
            common.TRACE_REPO_DIR = source_dir
            try:
                compile_trace(
                    "traceA", out_dir, cal_fraction=0.5,
                    nslice_typical=1, nslice_revisit=1, nslice_burst=1,
                )
            finally:
                common.TRACE_REPO_DIR = os.path.join(source_dir, "does-not-exist")
            try:
                load_event_stream(os.path.join(out_dir, "traceA", "events_typical.jsonl"))
            finally:
                common.TRACE_REPO_DIR = old_trace_dir


class MaterializationBlockingTest(unittest.TestCase):
    def test_persist_rejects_incomplete_materialization(self):
        lin = lineage.Lineage(
            lineage_id=1,
            turns=[lineage.MinimalTurn(1, -1, 0.0, 1, 32, 5, "text", 2)],
        )
        frozen = calibration.build_frozen_calibration(
            [lin], calibration.make_time_split([lin], 0.5), 0.9
        )
        with self.assertRaises(ValueError):
            assemble_slice(
                slice_class=SLICE_TYPICAL,
                slice_lineages=[lin],
                trace_key="traceA",
                file_sha256="x",
                source_window_ts=(0.0, 0.0),
                selection_rule="test",
                seed=1,
                time_dilation=1.0,
                frozen_cal=frozen,
                hash_provider=lambda _lid, _turn: (42,),
                return_event_stream=True,
            )


class CLIMemoryPathStructureTest(unittest.TestCase):
    def test_production_cli_uses_selected_second_pass(self):
        source = inspect.getsource(compile_trace)
        self.assertIn("hash_summary.observe", source)
        self.assertIn("selected_pairs", source)
        self.assertIn("selected_hashes = _second_pass_hash_provider", source)
        self.assertNotIn("per_chat_hash", source)
        self.assertNotIn("lineages_with_hash", source)
        self.assertNotIn("whole_trace_hash_use_count", source)


def lineage_replay(path):
    from trace_compiler.reader import TraceReadStream
    rs = TraceReadStream("traceA", path)
    list(rs)  # drain to populate provenance
    return rs


if __name__ == "__main__":
    unittest.main(verbosity=2)
