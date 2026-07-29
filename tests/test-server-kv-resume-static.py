#!/usr/bin/env python3
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTEXT = (ROOT / "tools/server/server-context.cpp").read_text(encoding="utf-8")
RESUME = (ROOT / "tools/server/server-kv-resume.cpp").read_text(encoding="utf-8")
ACTION = (ROOT / "src/llama-kv-cache-action.h").read_text(encoding="utf-8")


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


class ServerKvResumeStaticTest(unittest.TestCase):
    def test_action_has_explicit_all_required_semantics(self):
        self.assertIn("bool all_required = false", ACTION)
        self.assertIn("true,\n            true", RESUME)

    def test_resume_gate_is_after_n_past_and_before_batch_setup(self):
        update_slots = function_body(CONTEXT, "void update_slots()")
        gate = "server_kv_resume_gate("
        self.assertIn(gate, update_slots)
        self.assertLess(update_slots.index("slot.n_prompt_tokens_cache = n_past;"), update_slots.index(gate))
        self.assertLess(update_slots.index(gate), update_slots.index("common_context_seq_rm(ctx_tgt, slot.id, p0, -1);"))
        self.assertLess(update_slots.index(gate), update_slots.index("const int ret = llama_decode(ctx_tgt, batch_view);"))

    def test_server_only_passes_logical_intent(self):
        gate = function_body(RESUME, "server_kv_resume_gate_result server_kv_resume_gate(")
        self.assertIn("llama_kv_action::prefetch", gate)
        self.assertIn("decision_id", gate)
        self.assertIn("seq_id", gate)
        self.assertIn("true,\n            true", gate)
        for forbidden in ("physical", "free_list", "backing", "paged_block"):
            self.assertNotIn(forbidden, gate)

    def test_strict_failure_gate_prevents_graph(self):
        gate = function_body(RESUME, "server_kv_resume_gate_result server_kv_resume_gate(")
        for required in (
                "result.action.decision_id == decision_id",
                "outcome_ok",
                "!result.action.io_failure",
                "!result.action.fail_stop",
                "!result.action.capability.context_invalid",
                "result.action.shortfall_bytes == 0",
        ):
            self.assertIn(required, gate)
        update_slots = function_body(CONTEXT, "void update_slots()")
        self.assertIn("if (!result.graph_allowed)", update_slots)
        self.assertIn("slot.release();", update_slots)
        self.assertIn("continue;", update_slots)

    def test_protection_lifecycle_is_not_pressure_gated(self):
        self.assertIn("server_kv_resume_trigger::active_access", CONTEXT)
        self.assertNotIn("kv_pressure", RESUME)
        self.assertIn("clear_kv_resume_protection();", function_body(CONTEXT, "void prompt_clear(bool allow_processing)"))
        self.assertIn("clear_kv_resume_protection();", function_body(CONTEXT, "void release()"))
        self.assertNotIn("clear_kv_resume_protection", RESUME)


if __name__ == "__main__":
    unittest.main()
