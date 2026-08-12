#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run-memory-behavior-profile.py"

spec = importlib.util.spec_from_file_location("memory_behavior_profile", SCRIPT)
assert spec is not None and spec.loader is not None
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class MemoryBehaviorProfileParserTest(unittest.TestCase):
    def parse_text(self, text: str):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "server.stderr"
            path.write_text(text, encoding="utf-8")
            return profile.parse_log_file(path)

    def test_parse_prefixed_memory_governor_marker(self):
        events, warnings = self.parse_text(
            "srv update_slots: memory_governor_observe sample_count=1 "
            "effective_pressure_state=NORMAL dense_resident_bytes=100 "
            "moe_resident_bytes=0 kv_effective_resident_bytes=200\n"
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["marker"], "memory_governor_observe")
        self.assertEqual(events[0]["fields"]["dense_resident_bytes"], 100)
        self.assertEqual(events[0]["fields"]["kv_effective_resident_bytes"], 200)
        self.assertFalse(any("no memory_governor_observe" in item for item in warnings))

    def test_aggregate_dense_moe_kv_deltas(self):
        events, _ = self.parse_text(
            "memory_governor_observe sample_count=1 effective_pressure_state=NORMAL "
            "dense_flex_enabled=1 dense_resident_bytes=100 dense_bytes_read_phys=10 dense_read_ops=1 "
            "moe_enabled=1 moe_resident_bytes=50 moe_bytes_read=20 moe_streams=2 moe_evictions=0 "
            "moe_cache_hits=3 moe_cache_misses=1 kv_memory_present=1 kv_resident_bytes=70 "
            "kv_reclaimable_resident_bytes=7 kv_slot_resident_bytes=30 kv_effective_resident_bytes=70\n"
            "memory_governor_observe sample_count=2 effective_pressure_state=PRESSURE "
            "dense_flex_enabled=1 dense_resident_bytes=150 dense_bytes_read_phys=40 dense_read_ops=4 "
            "moe_enabled=1 moe_resident_bytes=80 moe_bytes_read=50 moe_streams=5 moe_evictions=2 "
            "moe_cache_hits=7 moe_cache_misses=3 kv_memory_present=1 kv_resident_bytes=90 "
            "kv_reclaimable_resident_bytes=9 kv_slot_resident_bytes=45 kv_effective_resident_bytes=90\n"
        )
        obs = profile.marker_events(events, "memory_governor_observe")
        dense = profile.aggregate_dense(obs)
        moe = profile.aggregate_moe(obs)
        kv = profile.aggregate_kv(obs)
        pressure = profile.aggregate_pressure(obs)
        self.assertEqual(dense["resident_bytes_max"], 150)
        self.assertEqual(dense["bytes_read_phys_delta"], 30)
        self.assertEqual(dense["read_ops_delta"], 3)
        self.assertEqual(moe["resident_bytes_max"], 80)
        self.assertEqual(moe["bytes_read_delta"], 30)
        self.assertEqual(moe["streams_delta"], 3)
        self.assertEqual(moe["evictions_delta"], 2)
        self.assertAlmostEqual(moe["cache_hit_rate"], 0.7)
        self.assertEqual(kv["resident_bytes_max"], 90)
        self.assertEqual(kv["slot_resident_bytes_max"], 45)
        self.assertEqual(pressure["states_seen"], ["NORMAL", "PRESSURE"])

    def test_unknown_field_is_preserved(self):
        events, _ = self.parse_text(
            "memory_governor_observe sample_count=1 effective_pressure_state=NORMAL "
            "dense_resident_bytes=1 moe_resident_bytes=2 kv_effective_resident_bytes=3 future_new_field=abc\n"
        )
        self.assertEqual(events[0]["fields"]["future_new_field"], "abc")

    def test_duplicate_key_records_warning(self):
        events, warnings = self.parse_text(
            "memory_governor_observe sample_count=1 effective_pressure_state=NORMAL "
            "dense_resident_bytes=1 dense_resident_bytes=2 moe_resident_bytes=0 kv_effective_resident_bytes=0\n"
        )
        self.assertEqual(events[0]["fields"]["dense_resident_bytes"], 2)
        self.assertIn("dense_resident_bytes", events[0]["duplicate_keys"])
        self.assertTrue(any("duplicate keys" in item for item in warnings))

    def test_no_marker_warns_but_parses(self):
        events, warnings = self.parse_text("ordinary log line\n")
        self.assertEqual(events, [])
        self.assertTrue(any("no memory_governor_observe markers found" in item for item in warnings))


if __name__ == "__main__":
    unittest.main()
