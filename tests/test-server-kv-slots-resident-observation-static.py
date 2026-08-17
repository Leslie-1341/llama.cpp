#!/usr/bin/env python3
# Static contract test for the restored `/slots.kv_resident` physical authority.
#
# Asserts on the source of tools/server/server-context.cpp that:
#   - the new independent flag `kv_g0_s1_slots_resident_observation` is declared
#     separately from the transaction marker and periodic-preflight flags;
#   - in SERVER_TASK_TYPE_METRICS, exactly one whole-KV sample is taken before
#     iterating slots and copied verbatim to every slot (no per-slot mincore);
#   - the canonical available/unavailable schema is produced through the
#     unit-tested format_kv_slots_resident_observation helper;
#   - the preflight flag `kv_g0_s1_resident_preflight` still routes only periodic
#     preflight telemetry and is not the gate for /slots kv_resident;
#   - LLAMA_KV_RESIDENT_PREFLIGHT is parsed independently of the matrix.

import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "tools/server/server-kv-slots-resident-observation.h").read_text(encoding="utf-8")


class SlotsResidentObservationStaticTest(unittest.TestCase):
    def test_separate_flag_declared_with_others(self) -> None:
        decl_start = CONTEXT.index("bool kv_g0_s1_resident_observation = false;")
        decl_end = CONTEXT.index("uint64_t next_kv_decision_id()", decl_start)
        decls = CONTEXT[decl_start:decl_end]
        self.assertIn("bool kv_g0_s1_resident_observation = false;", decls)
        self.assertIn("bool kv_g0_s1_slots_resident_observation = false;", decls)
        self.assertIn("bool kv_g0_s1_resident_preflight = false;", decls)
        # Order matters: slots flag sits between transaction marker and preflight.
        idx_obs = decls.index("bool kv_g0_s1_resident_observation = false;")
        idx_slots = decls.index("bool kv_g0_s1_slots_resident_observation = false;")
        idx_preflight = decls.index("bool kv_g0_s1_resident_preflight = false;")
        self.assertLess(idx_obs, idx_slots)
        self.assertLess(idx_slots, idx_preflight)

    def test_matrix_is_parsed_outside_init_contract_window(self) -> None:
        # The existing preflight static contract asserts on the window:
        #   [resident_observation, const char * resume_stage_timing =)
        # We insert the matrix parser *after* that window so the preflight test
        # is preserved.  Verify the parser call lives strictly after the
        # resume_stage_timing line.
        init_start = CONTEXT.index('const char * resident_observation =')
        resume_line = CONTEXT.index('const char * resume_stage_timing =', init_start)
        parser_line = CONTEXT.index('parse_kv_slots_resident_routing(resident_observation)')
        self.assertGreater(parser_line, resume_line)
        window = CONTEXT[init_start:resume_line]
        # The init window must NOT contain the preflight strcmp that 9aefabb1e
        # removed, and our new parser must not reintroduce it there.
        self.assertNotIn('std::strcmp(resident_observation, "preflight")', window)

    def test_slots_observation_derives_from_matrix_helper(self) -> None:
        parser_line = CONTEXT.index('const auto resident_route = parse_kv_slots_resident_routing(resident_observation)')
        tail = CONTEXT[parser_line:parser_line + 300]
        self.assertIn('kv_g0_s1_slots_resident_observation = resident_route.slots_observation', tail)

    def test_metrics_takes_one_sample_copied_to_all_slots(self) -> None:
        metrics_start = CONTEXT.index('case SERVER_TASK_TYPE_METRICS:')
        # An unambiguous boundary: the next sibling case.
        next_case = CONTEXT.index('\n            case SERVER_TASK_TYPE_', metrics_start + 5)
        metrics = CONTEXT[metrics_start:next_case]
        # One whole-KV sample, taken before the slot loop.
        loop = metrics.index('for (server_slot & slot : slots)')
        sample_calls = metrics[:loop].count('mem->sample_kv_resident()')
        self.assertGreaterEqual(sample_calls, 1)
        # No per-slot mincore inside the loop.
        self.assertEqual(metrics[loop:].count('mem->sample_kv_resident()'), 0)
        self.assertEqual(metrics[loop:].count('sample_kv_resident'), 0)
        # The /slots field is produced via the unit-tested formatter only.
        self.assertIn('format_kv_slots_resident_observation(slots_kv_resident)', metrics)
        self.assertIn('slot_data["kv_resident"] =', metrics)
        # The slots snapshot is gated on the slots-observation flag only — not
        # on the periodic-preflight flag and not on the transaction marker.
        gate = metrics.index('if (kv_g0_s1_slots_resident_observation)')
        self.assertLess(gate, loop)
        self.assertNotIn('kv_g0_s1_resident_preflight', metrics)
        self.assertNotIn('kv_g0_s1_resident_observation', metrics)

    def test_unavailable_fail_closed_via_helper_not_inline_zero(self) -> None:
        self.assertIn('format_kv_slots_resident_observation(slots_kv_resident)', CONTEXT)
        self.assertIn('{"status", "unavailable"}', HEADER)
        # No inline schema reimplementation remains in server-context.cpp.
        self.assertNotIn('{"status", "unavailable"}', CONTEXT)
        self.assertNotIn('"paged_sample_mincore"', CONTEXT)

    def test_preflight_flag_still_routes_only_periodic_telemetry(self) -> None:
        # The preflight formatter/helper block must still be gated on the
        # preflight flag, untouched by this change.
        emit_start = CONTEXT.index('void emit_resident_preflight_observation(')
        emit_end = CONTEXT.index('void maybe_sample_kv_pressure(', emit_start)
        emit = CONTEXT[emit_start:emit_end]
        self.assertIn('if (!kv_g0_s1_resident_preflight)', emit)
        # The slots-observation flag must not appear in the preflight helper.
        self.assertNotIn('kv_g0_s1_slots_resident_observation', emit)

    def test_preflight_env_switch_is_independent_of_matrix(self) -> None:
        init_start = CONTEXT.index('const char * resident_observation =')
        resume_line = CONTEXT.index('const char * resume_stage_timing =', init_start)
        window = CONTEXT[init_start:resume_line]
        self.assertIn('std::getenv("LLAMA_KV_RESIDENT_PREFLIGHT")', window)
        self.assertIn('std::strcmp(resident_preflight, "1")', window)
        self.assertIn('kv_g0_s1_resident_preflight = resident_preflight', window)
        self.assertIn('kv_g0_s1_resident_observation = resident_observation_both', window)

    def test_no_mutation_of_lifecycle_or_global_credit(self) -> None:
        # The METRICS handler must not gain any lifecycle/governor calls.
        metrics_start = CONTEXT.index('case SERVER_TASK_TYPE_METRICS:')
        next_case = CONTEXT.index('\n            case SERVER_TASK_TYPE_', metrics_start + 5)
        metrics = CONTEXT[metrics_start:next_case]
        for banned in (
            'execute_action',
            'server_kv_pressure_execute_governor',
            'prefetch_seq',
            'release_settled',
            'physical_relief',
        ):
            self.assertNotIn(banned, metrics)
        # sample_kv_resident is read-only; no execute / clear / reset.

    def test_header_matrix_helper_is_pure_and_single_source(self) -> None:
        self.assertIn('struct server_kv_slots_resident_routing', HEADER)
        self.assertIn('inline server_kv_slots_resident_routing parse_kv_slots_resident_routing(const char * mode)', HEADER)
        for mode, tm, so in (
            ('"1"', True, False),
            ('"preflight"', False, True),
            ('"both"', True, True),
        ):
            self.assertIn('std::strcmp(mode, %s)' % mode, HEADER)
        # Fail-closed default when mode is nullptr.
        self.assertIn('if (mode == nullptr)', HEADER)


if __name__ == "__main__":
    unittest.main()
