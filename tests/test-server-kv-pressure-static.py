#!/usr/bin/env python3

import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "tools/server/server-context.cpp"
RUNTIME_H = ROOT / "tools/server/server-kv-pressure.h"
RUNTIME_CPP = ROOT / "tools/server/server-kv-pressure.cpp"


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace:index + 1]
    raise AssertionError(f"unterminated function: {signature}")


class ServerKvPressureStaticTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = CONTEXT.read_text(encoding="utf-8")
        cls.runtime = RUNTIME_H.read_text(encoding="utf-8") + RUNTIME_CPP.read_text(encoding="utf-8")

    def test_single_owner_and_update_slots_only(self):
        self.assertIn("std::unique_ptr<kv_pressure_sampler> kv_pressure_sampler_owner", self.context)
        self.assertEqual(self.context.count("kv_pressure_sampler_owner->sample()"), 1)

        maybe_sample = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)")
        self.assertIn("kv_pressure_sampler_owner->sample()", maybe_sample)

        update_slots = function_body(self.context, "void update_slots()")
        self.assertIn("maybe_sample_kv_pressure(all_idle);", update_slots)
        self.assertLess(
            update_slots.index("maybe_sample_kv_pressure(all_idle);"),
            update_slots.index("if (all_idle)"),
        )

    def test_no_kv_mutation_or_reclaim_calls(self):
        integration = function_body(self.context, "void maybe_sample_kv_pressure(bool idle)") + self.runtime
        forbidden = (
            "paged_release_blocks(",
            "release_blocks(",
            "swap_out(",
            "swap_in(",
            "prefetch_seq(",
            "prefetch_seq_step(",
            "madvise(",
            "MADV_DONTNEED",
        )
        for token in forbidden:
            self.assertNotIn(token, integration)

    def test_no_background_thread_or_lock(self):
        forbidden = ("std::thread", "std::mutex", "std::shared_mutex", "condition_variable")
        for token in forbidden:
            self.assertNotIn(token, self.runtime)

    def test_structured_marker_fields(self):
        required = (
            "state=", "previous_state=", "source=", "sample_valid=", "stale=",
            "rss_kb=", "cgroup_current_bytes=", "cgroup_max_bytes=",
            "psi_some_avg10=", "psi_full_avg10=", "sample_latency_ns=",
            "sample_count=", "skip_count=",
        )
        for field in required:
            self.assertIn(field, self.runtime)

    def test_master_switch_preflight_precedes_sampler_init(self):
        init_body = function_body(self.context, "void init_kv_pressure_sampler()")
        self.assertLess(
            init_body.index("kv_pressure_sampler_environment_enablement()"),
            init_body.index("std::make_unique<kv_pressure_sampler>()"),
        )
        self.assertRegex(init_body, re.compile(r"DISABLED\) \{\s*return;", re.MULTILINE))

    def test_sleep_resume_resets_sampler_lifecycle(self):
        lifecycle = function_body(self.context, "void handle_sleeping_state(bool new_state)")
        self.assertIn("kv_pressure_sampler_owner.reset();", lifecycle)
        self.assertIn("kv_pressure_runtime.disable();", lifecycle)
        self.assertIn("init_kv_pressure_sampler();", lifecycle)


if __name__ == "__main__":
    unittest.main()
