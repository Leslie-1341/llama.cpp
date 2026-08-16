#!/usr/bin/env python3
"""Synthetic tests for the canonical Formal OFFLOAD Benchmark evidence spine."""
from __future__ import annotations

import hashlib
import http.client
import io
import importlib.util
import json
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import textwrap
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run-kv-offload-benchmark.py"
PARSER = ROOT / "scripts" / "parse-kv-offload-benchmark.py"


def load_runner_module():
    module_spec = importlib.util.spec_from_file_location("kv_offload_runner_test_module", RUNNER)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load benchmark runner")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_parser_module():
    module_spec = importlib.util.spec_from_file_location("kv_offload_parser_test_module", PARSER)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("cannot load benchmark parser")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


ACTION_FIELDS = {
    "state": "NORMAL", "source": "RSS_ABSOLUTE", "sample_valid": "1", "stale": "0", "pressure_basis_valid": "1",
    "decision_id": "1", "episode": "0",
    "target_bytes": "8192", "max_blocks": "64", "observed_excess_bytes": "4096", "debt_before_bytes": "4096",
    "debt_after_bytes": "0", "budget_active": "1", "budget_target_enabled": "1", "budget_source": "env_static",
    "budget_target_bytes": "4096", "budget_basis_generation": "1", "budget_view_valid": "1",
    "budget_resident_available": "1", "budget_reclaimable_available": "1", "budget_resident_bytes": "8192",
    "budget_dead_resident_reclaimable_bytes": "0", "budget_transient_staging_bound_bytes": "0",
    "budget_observed_excess_bytes": "4096", "budget_debt_before_bytes": "4096", "budget_debt_after_bytes": "0",
    "soft_offload_armed_before": "1", "soft_offload_armed_after": "0", "budget_next_action_sample": "2",
    "unmet_budget_bytes_after": "0", "offload_armed_before": "1", "offload_armed_after": "0",
    "next_action_sample": "2", "evaluate_attempted": "1", "evaluate_outcome": "completed",
    "evaluate_reason": "none", "release_attempted": "0", "offload_attempted": "1", "selected_seq_id": "0",
    "selected_claimant_epoch": "1", "transaction_id": "7", "outcome": "completed", "reason": "none",
    "blocks": "1", "bytes": "4096", "relieved_bytes": "4096", "shortfall_bytes": "0", "io_failure": "0",
    "io_errno": "0", "state_changed": "1", "decision_reason": "budget_excess", "sample_count": "1", "idle": "1",
    "claimants": "0:1:0:0:1:1:1:0:0:0", "scores": "none",
}
IO_FIELDS = {
    "block_swap_out_calls": "1", "block_swap_in_calls": "1", "backing_read_syscalls": "1", "backing_write_syscalls": "1",
    "bytes_read": "4096", "bytes_written": "4096",
    "avg_block_swap_out_latency_us": "1", "max_block_swap_out_latency_us": "1",
    "avg_block_swap_in_latency_us": "1", "max_block_swap_in_latency_us": "1",
    "staging_buffer_bytes": "4096", "k2_enabled": "0", "k2_group_byte_cap": "0",
    "k2_staging_bound_bytes": "0", "k2_peak_staging_groups": "0", "k2_peak_staging_bytes": "0", "k2_pipeline_wall_us": "0",
    "k2_exposed_read_wait_us": "0", "k2_pipeline_stall_us": "0", "k2_read_completed_ahead": "0",
    "block_out_validate_us": "1", "avg_block_out_validate_us": "1",
    "block_out_pack_us": "1", "avg_block_out_pack_us": "1",
    "block_out_write_us": "1", "avg_block_out_write_us": "1",
    "block_out_metadata_us": "1", "avg_block_out_metadata_us": "1",
    "block_out_madvise_us": "1", "avg_block_out_madvise_us": "1",
    "block_in_validate_us": "1", "avg_block_in_validate_us": "1",
    "block_in_read_us": "2", "avg_block_in_read_us": "2",
    "block_in_unpack_us": "3", "avg_block_in_unpack_us": "3",
    "block_in_commit_us": "1", "avg_block_in_commit_us": "1",
    "restore_prefault_enabled": "0", "restore_prefault_groups": "0", "restore_prefault_calls": "0", "restore_prefault_us": "0",
    "restore_prefault_minor_faults": "0", "restore_prefault_major_faults": "0", "restore_scatter_groups": "1",
    "restore_scatter_us": "10", "restore_scatter_fault_groups": "0", "restore_scatter_minor_faults": "0",
    "restore_scatter_major_faults": "0",
}


def marker(token: str, fields: dict[str, str]) -> str:
    return token + " " + " ".join(f"{key}={value}" for key, value in fields.items())


class CanonicalBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="kv_offload_benchmark_")
        self.root = pathlib.Path(self.temp.name)
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"synthetic-model")
        self.fake_server = self.root / "fake-server.py"
        self.fake_server.write_text(textwrap.dedent(f"""
            #!/usr/bin/env python3
            import argparse, json, os, signal, sys, threading, time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            offload_enabled = os.environ.get('LLAMA_KV_PAGED_SWAP') == '1'
            unified_enabled = os.environ.get('LLAMA_KV_PRESSURE_UNIFIED_ACTION') == '1'
            release_only_enabled = unified_enabled and not offload_enabled
            mode = os.environ.get('KV_SYNTHETIC_MODE', 'complete')
            observation_mode = os.environ.get('LLAMA_KV_G0_S1_RESIDENT_OBSERVATION', '')
            transaction_local_observation = observation_mode in ('1', 'both')
            slots_physical_observation = observation_mode in ('preflight', 'both')
            omit_v2_resident = os.environ.get('KV_SYNTHETIC_OMIT_V2_SLOTS_RESIDENT') == '1'
            resident_target = int(os.environ.get('LLAMA_KV_RESIDENT_TARGET_BYTES', '4096') or '4096')
            action_target = int(os.environ.get('LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES', '8192') or '8192')
            action_max_blocks = int(os.environ.get('LLAMA_KV_PRESSURE_UNIFIED_ACTION_MAX_BLOCKS', '64') or '64')
            action_fields = {ACTION_FIELDS!r}
            io_fields = {IO_FIELDS!r}
            state = {{
                'requests': 0, 'offload_emitted': False, 'prefetch_completed': False,
                'release_emitted': False, 'release_count': 0, 'released_bytes': 0,
                'offload_count': 0, 'offloaded_bytes': 0,
                'resident_bytes': 12288 if mode.startswith('characterization_') else 8192,
            }}
            characterization_mode = mode.startswith('characterization_')
            k2_enabled = os.environ.get('LLAMA_KV_PAGED_RESTORE_K2') == '1'
            output_lock = threading.Lock()

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--port', type=int, required=True)
            parser.add_argument('--host', default='127.0.0.1')
            args, _ = parser.parse_known_args()

            def emit_marker(token, fields):
                with output_lock:
                    print(token + " " + " ".join(f"{{key}}={{value}}" for key, value in fields.items()), file=sys.stderr, flush=True)

            def emit_transaction_observation(fields):
                if transaction_local_observation:
                    emit_marker('kv_g0_s1_resident_observation', fields)

            def resident_drop(decision_id, transaction_id, before_bytes, after_bytes):
                total_bytes = 12288 if characterization_mode else 8192
                return {{
                    'source': 'paged_sample_mincore', 'action': 'offload',
                    'decision_id': str(decision_id), 'seq_id': '0',
                    'transaction_id': str(transaction_id), 'server_pid': str(os.getpid()),
                    'before_available': '1', 'before_object_id': '1', 'before_generation': '1',
                    'before_page_size': '4096', 'before_total_bytes': str(total_bytes),
                    'before_resident_bytes': str(before_bytes),
                    'before_total_pages': str(total_bytes // 4096),
                    'before_resident_pages': str(before_bytes // 4096),
                    'after_available': '1', 'after_object_id': '1', 'after_generation': '1',
                    'after_page_size': '4096', 'after_total_bytes': str(total_bytes),
                    'after_resident_bytes': str(after_bytes),
                    'after_total_pages': str(total_bytes // 4096),
                    'after_resident_pages': str(after_bytes // 4096),
                }}

            def characterization_action(decision_id, resident_bytes, debt_after, *, terminal=None):
                values = dict(action_fields)
                observed_excess = max(0, resident_bytes - resident_target)
                values.update({{
                    'decision_id': str(decision_id), 'target_bytes': str(action_target),
                    'max_blocks': str(action_max_blocks),
                    'observed_excess_bytes': str(observed_excess),
                    'debt_before_bytes': str(observed_excess), 'debt_after_bytes': str(debt_after),
                    'budget_target_bytes': str(resident_target), 'budget_resident_bytes': str(resident_bytes),
                    'budget_observed_excess_bytes': str(observed_excess),
                    'budget_debt_before_bytes': str(observed_excess),
                    'budget_debt_after_bytes': str(debt_after),
                    'budget_transient_staging_bound_bytes': '2048' if k2_enabled else '0',
                    'sample_count': str(decision_id), 'idle': '1',
                }})
                if terminal is None:
                    values.update({{
                        'transaction_id': str(6 + decision_id), 'selected_seq_id': '0',
                        'selected_claimant_epoch': '1', 'offload_attempted': '1',
                        'release_attempted': '0', 'outcome': 'completed',
                        'reason': 'target_shortfall' if debt_after else 'none',
                        'blocks': '1', 'bytes': '4096', 'relieved_bytes': '4096',
                        'shortfall_bytes': str(debt_after), 'io_failure': '0',
                        'state_changed': '1', 'decision_reason': 'budget_offload_submitted',
                        'unmet_budget_bytes_after': '0', 'soft_offload_armed_before': '1',
                        'soft_offload_armed_after': '1' if debt_after else '0',
                    }})
                else:
                    unmet = observed_excess if terminal == 'budget_unmet_terminal' else 0
                    values.update({{
                        'transaction_id': '0', 'selected_seq_id': '-1',
                        'selected_claimant_epoch': '0', 'offload_attempted': '0',
                        'release_attempted': '0', 'outcome': 'no_op', 'reason': 'none',
                        'blocks': '0', 'bytes': '0', 'relieved_bytes': '0',
                        'shortfall_bytes': '0', 'io_failure': '0', 'state_changed': '0',
                        'decision_reason': terminal, 'unmet_budget_bytes_after': str(unmet),
                        'budget_debt_after_bytes': str(unmet), 'debt_after_bytes': str(unmet),
                        'evaluate_attempted': '0', 'evaluate_outcome': 'no_op',
                        'soft_offload_armed_before': '0', 'soft_offload_armed_after': '0',
                        'offload_armed_before': '0', 'offload_armed_after': '0',
                    }})
                return values

            def characterization_release_action(
                    decision_id, resident_bytes, *, no_candidate=False, released_bytes=None):
                values = dict(action_fields)
                observed_excess = max(0, resident_bytes - resident_target)
                values.update({{
                    'decision_id': str(decision_id), 'target_bytes': str(action_target),
                    'max_blocks': str(action_max_blocks),
                    'observed_excess_bytes': str(observed_excess),
                    'debt_before_bytes': str(observed_excess),
                    'budget_target_bytes': str(resident_target),
                    'budget_resident_bytes': str(resident_bytes),
                    'budget_observed_excess_bytes': str(observed_excess),
                    'budget_debt_before_bytes': str(observed_excess),
                    'sample_count': str(decision_id), 'idle': '1',
                    'release_attempted': '1', 'offload_attempted': '0',
                    'offload_armed_before': '0', 'offload_armed_after': '0',
                }})
                if no_candidate:
                    values.update({{
                        'transaction_id': '0', 'selected_seq_id': '-1',
                        'selected_claimant_epoch': '0', 'outcome': 'no_op',
                        'reason': 'no_candidate', 'blocks': '0', 'bytes': '0',
                        'relieved_bytes': '0', 'shortfall_bytes': str(observed_excess),
                        'io_failure': '0', 'state_changed': '0',
                        'decision_reason': 'budget_release_submitted',
                        'debt_after_bytes': str(observed_excess),
                        'budget_debt_after_bytes': str(observed_excess),
                        'unmet_budget_bytes_after': '0',
                        'soft_offload_armed_before': '0', 'soft_offload_armed_after': '1',
                        'offload_armed_before': '0', 'offload_armed_after': '1',
                    }})
                else:
                    released = (
                        max(0, resident_bytes - resident_target)
                        if released_bytes is None else released_bytes)
                    remaining = max(0, resident_bytes - released - resident_target)
                    values.update({{
                        'transaction_id': str(20 + decision_id), 'selected_seq_id': '-1',
                        'selected_claimant_epoch': '0', 'outcome': 'completed',
                        'reason': 'target_satisfied' if remaining == 0 else 'target_shortfall',
                        'blocks': str(released // 4096), 'bytes': str(released),
                        'relieved_bytes': str(released),
                        'shortfall_bytes': str(remaining), 'io_failure': '0',
                        'state_changed': '1', 'decision_reason': 'budget_release_submitted',
                        'debt_after_bytes': str(remaining),
                        'budget_debt_after_bytes': str(remaining),
                        'unmet_budget_bytes_after': '0',
                        'soft_offload_armed_before': '0', 'soft_offload_armed_after': '0',
                    }})
                return values

            def characterization_soft_offload_unsupported(decision_id, resident_bytes):
                values = dict(action_fields)
                observed_excess = max(0, resident_bytes - resident_target)
                values.update({{
                    'decision_id': str(decision_id), 'target_bytes': str(action_target),
                    'max_blocks': str(action_max_blocks),
                    'observed_excess_bytes': str(observed_excess),
                    'debt_before_bytes': str(observed_excess),
                    'debt_after_bytes': str(observed_excess),
                    'budget_target_bytes': str(resident_target),
                    'budget_resident_bytes': str(resident_bytes),
                    'budget_observed_excess_bytes': str(observed_excess),
                    'budget_debt_before_bytes': str(observed_excess),
                    'budget_debt_after_bytes': str(observed_excess),
                    'budget_transient_staging_bound_bytes': '2048' if k2_enabled else '0',
                    'sample_count': str(decision_id), 'idle': '1',
                    'transaction_id': '0', 'selected_seq_id': '-1',
                    'selected_claimant_epoch': '0', 'offload_attempted': '0',
                    'release_attempted': '0', 'outcome': 'no_op',
                    'reason': 'offload_unsupported', 'blocks': '0', 'bytes': '0',
                    'relieved_bytes': '0', 'shortfall_bytes': str(observed_excess),
                    'io_failure': '0', 'state_changed': '0',
                    'decision_reason': 'budget_offload_unsupported',
                    'unmet_budget_bytes_after': '0',
                    'soft_offload_armed_before': '1', 'soft_offload_armed_after': '1',
                    'offload_armed_before': '0', 'offload_armed_after': '0',
                    'evaluate_attempted': '1', 'evaluate_outcome': 'no_op',
                    'evaluate_reason': 'offload_unsupported',
                }})
                return values

            def emit_qualification_offload_after_fill():
                time.sleep(0.08)
                if not offload_enabled or mode == 'no_offload':
                    return
                observation = resident_drop(1, 7, 8192, 4096)
                if mode == 'mismatch_decision':
                    observation['decision_id'] = '2'
                elif mode == 'mismatch_transaction':
                    observation['transaction_id'] = '8'
                elif mode == 'mismatch_seq':
                    observation['seq_id'] = '1'
                elif mode == 'drop_zero':
                    observation['after_resident_bytes'] = observation['before_resident_bytes']
                    observation['after_resident_pages'] = observation['before_resident_pages']
                if mode != 'action_only':
                    emit_transaction_observation(observation)
                emit_marker('kv_pressure_unified_action', action_fields)
                state['offload_emitted'] = True
                state['offload_count'] = 1
                state['offloaded_bytes'] = 4096
                state['resident_bytes'] = 4096

            def emit_characterization_after_fill():
                time.sleep(0.08)
                if release_only_enabled:
                    if mode == 'characterization_release_no_candidate':
                        emit_marker(
                            'kv_pressure_unified_action',
                            characterization_release_action(1, state['resident_bytes'], no_candidate=True))
                        state['release_emitted'] = True
                        state['release_count'] = 1
                        time.sleep(0.02)
                        emit_marker(
                            'kv_pressure_unified_action',
                            characterization_soft_offload_unsupported(2, state['resident_bytes']))
                        return
                    before = state['resident_bytes']
                    after = resident_target
                    emit_marker(
                        'kv_pressure_unified_action',
                        characterization_release_action(1, before))
                    state['resident_bytes'] = after
                    state['release_emitted'] = True
                    state['release_count'] = max(1, (before - after) // 4096)
                    state['released_bytes'] = before - after
                    return
                if not offload_enabled:
                    return
                if mode != 'characterization_release_then_offload':
                    emit_marker(
                        'kv_pressure_unified_action',
                        characterization_release_action(1, state['resident_bytes'], no_candidate=True))
                    state['release_emitted'] = True
                    state['release_count'] = 1
                if mode == 'characterization_release_then_offload':
                    before = state['resident_bytes']
                    emit_marker(
                        'kv_pressure_unified_action',
                        characterization_release_action(1, before, released_bytes=4096))
                    state['resident_bytes'] = before - 4096
                    state['release_count'] = 1
                    state['released_bytes'] = 4096
                    time.sleep(0.02)
                    emit_marker(
                        'kv_pressure_unified_action',
                        characterization_release_action(2, state['resident_bytes'], no_candidate=True))
                    time.sleep(0.02)
                    before = state['resident_bytes']
                    after = before - 4096
                    emit_transaction_observation(resident_drop(3, 9, before, after))
                    emit_marker('kv_pressure_unified_action', characterization_action(3, before, 0))
                    state['resident_bytes'] = after
                    state['offload_emitted'] = True
                    state['offload_count'] = 1
                    state['offloaded_bytes'] = 4096
                    return
                time.sleep(0.02)
                before = state['resident_bytes']
                after = before - 4096
                emit_transaction_observation(resident_drop(2, 8, before, after))
                emit_marker('kv_pressure_unified_action', characterization_action(2, before, after - 4096))
                state['resident_bytes'] = after
                state['offload_emitted'] = True
                state['offload_count'] += 1
                state['offloaded_bytes'] += 4096
                if mode in ('characterization_timeout', 'characterization_first_only'):
                    return
                time.sleep(0.05)
                if mode == 'characterization_target':
                    before = state['resident_bytes']
                    after = before - 4096
                    emit_transaction_observation(resident_drop(3, 9, before, after))
                    emit_marker('kv_pressure_unified_action', characterization_action(3, before, 0))
                    state['resident_bytes'] = after
                    state['offload_count'] += 1
                    state['offloaded_bytes'] += 4096
                elif mode == 'characterization_unmet':
                    emit_marker('kv_pressure_unified_action', characterization_action(
                        3, state['resident_bytes'], state['resident_bytes'] - 4096,
                        terminal='budget_unmet_terminal'))

            def emit_resume_evidence():
                decision_id = '4' if characterization_mode else '2'
                transaction_id = '10' if characterization_mode else '8'
                if mode in ('prefetch_noop', 'characterization_release_noop_prefetch'):
                    outcome = 'no_op'
                    restored_blocks = '0'
                    restored_bytes = '0'
                    total_us = '6'
                else:
                    outcome = 'completed'
                    restored = state['offloaded_bytes'] or 4096
                    restored_blocks = str(restored // 4096)
                    restored_bytes = str(restored)
                    total_us = '6'
                    state['prefetch_completed'] = True
                    state['resident_bytes'] += restored
                base = {{
                    'decision_id': decision_id, 'seq_id': '0', 'claimant_epoch': '1',
                    'transaction_id': transaction_id, 'action': 'prefetch',
                    'outcome': outcome, 'reason': 'none', 'graph_allowed': '1',
                }}
                emit_marker('kv_resume_order_event', {{'phase': 'prefetch', **base}})
                emit_marker('kv_resume_order_event', {{'phase': 'graph_gate', **base}})
                emit_marker('kv_resume_stage_timing', {{
                    'decision_id': decision_id, 'seq_id': '0', 'transaction_id': transaction_id,
                    'restored_blocks': restored_blocks, 'restored_bytes': restored_bytes,
                    'queue_us': '1', 'gate_us': '2', 'graph_us': '3', 'total_us': total_us,
                }})

            def stop(_signum, _frame):
                values = dict(io_fields)
                for key in ('block_swap_out_calls', 'block_swap_in_calls', 'backing_read_syscalls', 'backing_write_syscalls', 'bytes_read', 'bytes_written'):
                    values[key] = '0'
                if offload_enabled and state['offload_emitted']:
                    values['block_swap_out_calls'] = str(state['offload_count'])
                    values['backing_write_syscalls'] = str(state['offload_count'])
                    values['bytes_written'] = str(state['offloaded_bytes'])
                if offload_enabled and state['prefetch_completed'] and mode != 'swap_in_zero':
                    values['block_swap_in_calls'] = str(max(1, state['offload_count']))
                    values['backing_read_syscalls'] = str(max(1, state['offload_count']))
                    values['bytes_read'] = str(state['offloaded_bytes'] or 4096)
                if not (offload_enabled and state['prefetch_completed']):
                    for key in (
                            'block_in_validate_us', 'avg_block_in_validate_us',
                            'block_in_read_us', 'avg_block_in_read_us',
                            'block_in_unpack_us', 'avg_block_in_unpack_us',
                            'block_in_commit_us', 'avg_block_in_commit_us',
                            'restore_prefault_groups', 'restore_prefault_calls',
                            'restore_prefault_us', 'restore_prefault_minor_faults',
                            'restore_prefault_major_faults', 'restore_scatter_groups',
                            'restore_scatter_us', 'restore_scatter_fault_groups',
                            'restore_scatter_minor_faults', 'restore_scatter_major_faults',
                    ):
                        values[key] = '0'
                if k2_enabled:
                    values.update({{
                        'k2_enabled': '1', 'k2_group_byte_cap': '2048',
                        'k2_staging_bound_bytes': '2048',
                        'k2_peak_staging_groups': '1' if state['prefetch_completed'] else '0',
                        'k2_peak_staging_bytes': '2048' if state['prefetch_completed'] else '0',
                        'k2_pipeline_wall_us': '11' if state['prefetch_completed'] else '0',
                        'k2_exposed_read_wait_us': '4' if state['prefetch_completed'] else '0',
                        'k2_pipeline_stall_us': '2' if state['prefetch_completed'] else '0',
                        'k2_read_completed_ahead': '1' if state['prefetch_completed'] else '0',
                    }})
                emit_marker('KV_PAGED_IO_STATS', values)
                raise SystemExit(0)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def do_GET(self):
                    if self.path == '/health':
                        self.send_response(200); self.end_headers(); self.wfile.write(b'{{"status":"ok"}}'); return
                    if self.path == '/slots':
                        slot = {{'id': 0, 'is_processing': False}}
                        if slots_physical_observation and not omit_v2_resident:
                            total_bytes = 12288 if characterization_mode else 8192
                            slot['kv_resident'] = {{
                                'status': 'available', 'source': 'synthetic', 'object_id': 1, 'generation': 1,
                                'page_size': 4096, 'total_bytes': total_bytes,
                                'resident_bytes': state['resident_bytes'],
                                'total_pages': total_bytes // 4096,
                                'resident_pages': state['resident_bytes'] // 4096,
                            }}
                        encoded = json.dumps([slot]).encode()
                        self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length', str(len(encoded))); self.end_headers(); self.wfile.write(encoded); return
                    self.send_response(404); self.end_headers()

                def do_POST(self):
                    length = int(self.headers.get('Content-Length', '0'))
                    self.rfile.read(length)
                    state['requests'] += 1
                    ordinal = state['requests']
                    print(f"synthetic_request ordinal={{ordinal}}", flush=True)
                    if (offload_enabled or release_only_enabled) and characterization_mode and ordinal == 1:
                        threading.Thread(target=emit_characterization_after_fill, daemon=True).start()
                    elif release_only_enabled and characterization_mode and ordinal == 2 and mode == 'characterization_release_noop_prefetch':
                        emit_resume_evidence()
                    elif offload_enabled and not characterization_mode and ordinal == 2:
                        threading.Thread(target=emit_qualification_offload_after_fill, daemon=True).start()
                    elif offload_enabled and (
                            (characterization_mode and ordinal == 2)
                            or (not characterization_mode and ordinal == 3)):
                        emit_resume_evidence()
                    time.sleep(0.02)
                    payload = {{
                        "content":"ok",
                        "timings":{{"predicted_n":2,"predicted_ms":4,"predicted_per_second":500.0}},
                    }}
                    encoded = json.dumps(payload).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(encoded)))
                    self.end_headers(); self.wfile.write(encoded)

            signal.signal(signal.SIGTERM, stop)
            server = ThreadingHTTPServer((args.host, args.port), Handler)
            server.daemon_threads = True
            server.serve_forever()
        """).strip() + "\n", encoding="utf-8")
        self.fake_server.chmod(self.fake_server.stat().st_mode | stat.S_IXUSR)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def spec(self, *, policy: str = "v2") -> dict[str, object]:
        target = 4096 if policy in {"release_only", "v2", "idle_age", "v3"} else None
        return {
            "schema_version": 2,
            "protocol": "kv_offload_benchmark",
            "phase": "coarse_target",
            "run_kind": "qualification",
            "run_mode": "qualification",
            "binary": str(self.fake_server),
            "model": str(self.model),
            "model_quantization": "synthetic",
            "server_args": [],
            "environment": {
                "KV_SYNTHETIC_MODE": "complete",
            },
            "pressure_basis": {
                "authority": "rss_absolute",
                "low_water_kb": 64 * 1024 * 1024,
                "pressure_kb": 128 * 1024 * 1024,
                "critical_kb": 256 * 1024 * 1024,
            },
            "workload": {
                "warmup": [{"request_id": "seed", "prompt": "seed", "n_predict": 1, "stream": False}],
                "requests": [{"request_id": "request", "prompt": "hello", "n_predict": 2, "stream": False}],
                "repeat": 1,
                "qualification": {
                    "idle_seconds": 0.01,
                    "offload_timeout_seconds": 1.0,
                    "resume_request_id": "request",
                },
                "characterization": None,
            },
            "cases": [{
                "case_id": "case",
                "policy": policy,
                "kv_representation": "paged",
                "loading_mode": "exact",
                "restore": "k1_sync",
                "prefault": "off",
                "kv_target_bytes": target,
                "action_target_bytes": 8192 if policy in {"release_only", "v2", "idle_age", "v3"} else None,
            }],
            "run_order": [{"round": 1, "run_order": 1, "case_id": "case"}],
            "sampler": {"interval_seconds": 1.0},
            "cgroup": {"expected_memory_max": None},
            "max_blocks": 64,
            "health_timeout_seconds": 10,
            "request_timeout_seconds": 10,
        }

    def characterization_spec(
            self,
            *,
            policy: str = "v2",
            mode: str = "characterization_target",
            restore: str = "k2_pipeline",
    ) -> dict[str, object]:
        value = self.spec(policy=policy)
        value["run_mode"] = "characterization"
        value["environment"]["KV_SYNTHETIC_MODE"] = mode
        value["workload"]["qualification"] = None
        value["workload"]["characterization"] = {
            "idle_seconds": 0.01,
            "settle_timeout_seconds": 1.0,
            "target_tolerance_bytes": 0,
            "resume_request_id": "request",
        }
        value["cases"][0]["restore"] = restore
        return value

    def write_spec(self, value: dict[str, object], name: str = "spec.json") -> pathlib.Path:
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def run_runner(self, spec: pathlib.Path, artifact: pathlib.Path, dry_run: bool = False) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, str(RUNNER), "--spec", str(spec), "--output", str(artifact)]
        if dry_run:
            command.append("--dry-run")
        return subprocess.run(command, text=True, capture_output=True, env={**os.environ, "PYTHONUTF8": "1"})

    def run_parser(self, artifact: pathlib.Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(PARSER), str(artifact)],
            text=True,
            capture_output=True,
            env={**os.environ, "PYTHONUTF8": "1"},
        )

    def run_real_artifact(
            self,
            *,
            policy: str = "v2",
            mode: str = "complete",
    ) -> pathlib.Path:
        value = self.spec(policy=policy)
        value["environment"]["KV_SYNTHETIC_MODE"] = mode
        spec = self.write_spec(value)
        artifact = self.root / f"real-{policy}-{mode}-{len(list(self.root.glob('real-*')))}"
        runner = self.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        return artifact

    def run_characterization_artifact(
            self,
            *,
            policy: str = "v2",
            mode: str = "characterization_target",
            restore: str = "k2_pipeline",
    ) -> pathlib.Path:
        value = self.characterization_spec(policy=policy, mode=mode, restore=restore)
        spec = self.write_spec(value, f"{mode}-{policy}.json")
        artifact = self.root / f"characterization-{policy}-{mode}-{len(list(self.root.glob('characterization-*')))}"
        runner = self.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        return artifact

    def probe_synthetic_observation_mode(self, mode: str) -> tuple[list[dict[str, object]], str]:
        runner = load_runner_module()
        port = runner.free_port()
        environment = {
            **os.environ,
            "LLAMA_KV_PAGED_SWAP": "1",
            "LLAMA_KV_PRESSURE_UNIFIED_ACTION": "1",
            "LLAMA_KV_G0_S1_RESIDENT_OBSERVATION": mode,
            "KV_SYNTHETIC_MODE": "complete",
        }
        process = subprocess.Popen(
            [sys.executable, str(self.fake_server), "--host", "127.0.0.1", "--port", str(port)],
            cwd=self.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        slots: list[dict[str, object]] | None = None
        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(f"synthetic server exited before observation probe: {process.returncode}")
                try:
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=0.2)
                    connection.request("GET", "/health")
                    response = connection.getresponse()
                    response.read()
                    connection.close()
                    if response.status == 200:
                        break
                except (OSError, http.client.HTTPException):
                    time.sleep(0.01)
            else:
                self.fail("synthetic server observation probe health timeout")

            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
            connection.request("GET", "/slots")
            response = connection.getresponse()
            slots = json.loads(response.read().decode("utf-8"))
            connection.close()
            for _ in range(2):
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1.0)
                connection.request("POST", "/completion", body=b"{}")
                response = connection.getresponse()
                response.read()
                connection.close()
            time.sleep(0.15)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                _, stderr = process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                _, stderr = process.communicate(timeout=2.0)
        self.assertIsNotNone(slots)
        return slots, stderr.decode("utf-8", errors="replace")

    def remove_slot_resident(self, artifact: pathlib.Path, filename: str) -> None:
        snapshot_path = next(artifact.glob(f"runs/*/{filename}"))
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        body_path = snapshot_path.parent / snapshot["body_path"]
        body = json.loads(body_path.read_text(encoding="utf-8"))
        for slot in body:
            if isinstance(slot, dict):
                slot.pop("kv_resident", None)
        raw = json.dumps(body).encode("utf-8")
        body_path.write_bytes(raw)
        snapshot["body_json"] = body
        snapshot["body_bytes"] = len(raw)
        snapshot["body_sha256"] = hashlib.sha256(raw).hexdigest()
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

    def run_incomplete_artifact(self, mode: str) -> tuple[pathlib.Path, subprocess.CompletedProcess[str]]:
        value = self.spec()
        value["environment"]["KV_SYNTHETIC_MODE"] = mode
        value["workload"]["qualification"]["offload_timeout_seconds"] = 0.2
        spec = self.write_spec(value, f"{mode}.json")
        artifact = self.root / f"incomplete-{mode}"
        return artifact, self.run_runner(spec, artifact)

    def test_dry_run_parser_and_runner_do_not_write_verdict(self) -> None:
        spec = self.write_spec(self.spec())
        artifact = self.root / "dry-run"
        runner = self.run_runner(spec, artifact, dry_run=True)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("verdict", json.dumps(manifest))
        self.assertFalse((artifact / "result.json").exists())
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "DRY_RUN")

    def test_v0_rejects_slo_facade_and_canonical_overrides(self) -> None:
        for mutation in (
            lambda value: value["workload"].update({"slo": {"ttft_ms": 1, "tpot_ms": 1, "attainment": 1}}),
            lambda value: value.update({"server_args": ["--port=9"]}),
            lambda value: value.update({"environment": {"LLAMA_KV_PAGED_SWAP": "0"}}),
        ):
            value = self.spec()
            mutation(value)
            spec = self.write_spec(value)
            artifact = self.root / "rejected"
            result = self.run_runner(spec, artifact, dry_run=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(artifact.exists())

    def test_nonfinite_timeout_and_interval_are_rejected(self) -> None:
        for field in ("health_timeout_seconds", "request_timeout_seconds"):
            value = self.spec()
            value[field] = float("nan")
            result = self.run_runner(self.write_spec(value), self.root / field, dry_run=True)
            self.assertEqual(result.returncode, 2, result.stderr)
        value = self.spec()
        value["sampler"]["interval_seconds"] = float("inf")
        result = self.run_runner(self.write_spec(value), self.root / "interval", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        value = self.spec()
        value["workload"]["qualification"]["offload_timeout_seconds"] = float("nan")
        result = self.run_runner(self.write_spec(value), self.root / "offload-timeout", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_resume_request_id_must_reference_measurement_request(self) -> None:
        value = self.spec()
        value["workload"]["qualification"]["resume_request_id"] = "seed"
        result = self.run_runner(self.write_spec(value), self.root / "bad-resume-id", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_v2_requires_positive_successful_offload_evidence(self) -> None:
        artifact = self.run_real_artifact()
        stderr = next(artifact.glob("runs/*/server.stderr"))
        stderr.write_text(stderr.read_text(encoding="utf-8").replace("offload_attempted=1", "offload_attempted=0"), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("matching successful action", result["errors"][0])

    def test_action_selection_identity_uses_contextual_seq_sentinel(self) -> None:
        parser = load_parser_module()

        def validate(action: dict[str, str]) -> None:
            fields = parser.marker_records(
                marker("kv_pressure_unified_action", action),
                "kv_pressure_unified_action",
                parser.ACTION_REQUIRED,
                "action",
            )[0]
            parser.validate_action_fields(fields, "action")

        for selected_seq_id in ("0", "7"):
            action = dict(ACTION_FIELDS)
            action["selected_seq_id"] = selected_seq_id
            validate(action)

        invalid = dict(ACTION_FIELDS)
        invalid["selected_seq_id"] = "-1"
        with self.assertRaisesRegex(parser.ParseError, "state-changing OFFLOAD"):
            validate(invalid)

        for field in ("selected_claimant_epoch", "transaction_id"):
            invalid = dict(ACTION_FIELDS)
            invalid[field] = "0"
            with self.subTest(field=field):
                with self.assertRaisesRegex(parser.ParseError, "positive"):
                    validate(invalid)

        noop = dict(ACTION_FIELDS)
        noop.update({
            "offload_attempted": "0", "release_attempted": "0", "outcome": "no_op",
            "state_changed": "0", "selected_seq_id": "-1", "selected_claimant_epoch": "0",
            "transaction_id": "0", "blocks": "0", "bytes": "0", "relieved_bytes": "0",
            "shortfall_bytes": "0",
        })
        validate(noop)

    def test_v3_audit_fail_closed_on_missing_marker_physical_and_score_evidence(self) -> None:
        # Canonical V3 tightening regression guard: a V3 decision marker must
        # carry (a) policy/decision_fallback marker fields, (b) physical
        # feedback fields, (c) a per-claimant score schema with cost-aware /
        # fallback tagging that agrees with the marker-level authority, and
        # (d) a selected victim that is auditable.  Each branch must fail-closed
        # so a mixed or evidence-less decision can never be certified valid.
        parser = load_parser_module()
        v3_case = {"policy": "v3"}

        def v3_score(seq_id: str, eligible: str, rank: str, *, cost_aware: bool,
                     exclusion: str = "none", fallback_reason: str = "none",
                     physical_bytes: str = "4096", expected_cost: str = "1125",
                     physical_object_id: str = "11", physical_generation: str = "3") -> str:
            cells = {f: "0" for f in parser.SCORE_FIELDS}
            cells.update({
                "seq_id": seq_id, "eligible": eligible, "exclusion": exclusion,
                "total": (str((int(expected_cost) * 1_000_000) // max(int(physical_bytes), 1))
                          if cost_aware else "1000"),
                "idle_age_score": "1000", "logical_kv_score": "1",
                "reclaimable_score": "0", "lcp_n_past_penalty": "0",
                "io_cost_penalty": "0", "failure_penalty": "0", "rank": rank,
                "cost_aware": "1" if cost_aware else "0",
                "physical_estimate_available": "1" if cost_aware else "0",
                "physical_estimate_authoritative": "1" if cost_aware else "0",
                "estimated_physical_bytes": physical_bytes,
                "reuse_probability_ppm": "500000" if cost_aware else "0",
                "expected_offload_write_cost_us": "100" if cost_aware else "0",
                "expected_restore_gate_cost_us": "50" if cost_aware else "0",
                "expected_cost_us": expected_cost if cost_aware else "0",
                "churn_penalty_us": "1000" if cost_aware else "0",
                "raw_idle_age_us": "1000", "raw_answer_tokens": "64",
                "raw_lcp_hint_tokens": "64",
                "physical_object_id": physical_object_id,
                "physical_generation": physical_generation,
                "actual_relief_bytes": "4096" if cost_aware else "0",
                "resident_lease_until_sample": "6" if cost_aware else "0",
                "round_trip_count": "1" if cost_aware else "0",
                "last_offload_bytes": "4096" if cost_aware else "0",
                "last_restore_bytes": "4096" if cost_aware else "0",
                "fallback_reason": fallback_reason,
            })
            return ":".join(cells[f] for f in parser.SCORE_FIELDS)

        def base_action(scores: str, *, decision_fallback: str = "none") -> dict[str, str]:
            action = dict(ACTION_FIELDS)
            action.update({
                "policy": "v3", "decision_fallback": decision_fallback,
                "action_elapsed_us": "120",
                "physical_relief_available": "1", "physical_relief_bytes": "4096",
                "physical_object_id": "11", "physical_generation": "3",
                "scores": scores,
            })
            return action

        def audit(action: dict[str, str]) -> None:
            fields = parser.marker_records(
                marker("kv_pressure_unified_action", action),
                "kv_pressure_unified_action", parser.ACTION_REQUIRED, "act")[0]
            parser.validate_action_fields(fields, "act")
            parser.validate_action_v3_audit(fields, v3_case, "act")

        # Happy path: fully cost-aware decision.
        victim = v3_score("0", "1", "0", cost_aware=True)
        other = v3_score("1", "0", "1", cost_aware=False, exclusion="empty", fallback_reason="none")
        audit(base_action(f"{victim};{other}", decision_fallback="none"))

        # Happy path: mixed decision falls back to idle-age; the fallback victim
        # is selected and the marker documents the fallback reason.
        fb_victim = v3_score("0", "1", "0", cost_aware=False,
                             fallback_reason="physical_unavailable", physical_bytes="0",
                             expected_cost="0", physical_object_id="0", physical_generation="0")
        audit(base_action(f"{fb_victim};{other}", decision_fallback="physical_unavailable"))

        # (a) Missing marker field fails closed.
        for missing in ("policy", "decision_fallback"):
            with self.subTest(missing=missing):
                action = base_action(victim, decision_fallback="none")
                action.pop(missing)
                with self.assertRaisesRegex(parser.ParseError, f"missing V3 marker field {missing!r}"):
                    audit(action)

        # Missing physical feedback field fails closed.
        for missing in ("action_elapsed_us", "physical_relief_available", "physical_object_id"):
            with self.subTest(missing=missing):
                action = base_action(victim, decision_fallback="none")
                action.pop(missing)
                with self.assertRaisesRegex(parser.ParseError, "missing V3 physical feedback field"):
                    audit(action)

        # (c1) Score schema with the wrong field count fails closed.
        bad_count = base_action("0:1:none", decision_fallback="none")
        with self.assertRaisesRegex(parser.ParseError, "score\\[0\\] has 3 fields"):
            audit(bad_count)

        # (c2) cost-aware claimant must not carry a fallback reason.
        mixed = v3_score("0", "1", "0", cost_aware=True, fallback_reason="physical_unavailable")
        with self.assertRaisesRegex(parser.ParseError, "cost-aware claimant must carry no fallback reason"):
            audit(base_action(f"{mixed};{other}", decision_fallback="physical_unavailable"))

        # (c3) eligible non-cost-aware claimant must record a fallback reason.
        bare = v3_score("0", "1", "0", cost_aware=False, fallback_reason="none")
        with self.assertRaisesRegex(parser.ParseError, "eligible non-cost-aware claimant must record a fallback reason"):
            audit(base_action(f"{bare};{other}", decision_fallback="none"))

        # (d) A claimant with history must NOT win just because it carries cost
        # evidence when the decision falls back to idle-age: an historical but
        # warm claimant is selected over a colder fallback claimant — fail
        # closed because the marker claims "none" but a fallback score exists.
        historical = v3_score("0", "1", "0", cost_aware=True)
        cold_fallback = v3_score("1", "1", "1", cost_aware=False,
                                 fallback_reason="physical_unavailable")
        with self.assertRaisesRegex(parser.ParseError, "carries a fallback reason"):
            audit(base_action(f"{historical};{cold_fallback}", decision_fallback="none"))

        # Conversely, marker fallback is spurious when no eligible non-cost-aware
        # claimant carries it — the decision authority is not auditable.
        with self.assertRaisesRegex(parser.ParseError, "not carried by any eligible non-cost-aware"):
            audit(base_action(f"{victim};{other}", decision_fallback="physical_unavailable"))

        # (e) selected_seq_id without any eligible score fails closed.
        lone_excluded = v3_score("2", "0", "0", cost_aware=False, exclusion="empty", fallback_reason="none")
        no_eligible = base_action(lone_excluded, decision_fallback="none")
        no_eligible["selected_seq_id"] = "0"
        with self.assertRaisesRegex(parser.ParseError, "no eligible claimant score was emitted"):
            audit(no_eligible)

        # (f) state-changing OFFLOAD missing physical lineage fails closed.
        no_lineage = base_action(victim, decision_fallback="none")
        no_lineage["physical_object_id"] = "0"
        with self.assertRaisesRegex(parser.ParseError, "missing physical lineage"):
            audit(no_lineage)

    def test_v2_requires_normal_valid_basis_and_budget_excess(self) -> None:
        artifact = self.run_real_artifact()
        stderr = next(artifact.glob("runs/*/server.stderr"))
        text = stderr.read_text(encoding="utf-8")
        self.assertIn("state=NORMAL", text)
        self.assertIn("sample_valid=1", text)
        self.assertIn("pressure_basis_valid=1", text)
        self.assertIn("budget_observed_excess_bytes=4096", text)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)

        for field, replacement in (
            ("pressure_basis_valid=1", "pressure_basis_valid=0"),
            ("budget_observed_excess_bytes=4096", "budget_observed_excess_bytes=0"),
            ("state=NORMAL", "state=PRESSURE"),
        ):
            with self.subTest(field=field):
                invalid = self.run_real_artifact()
                invalid_stderr = next(invalid.glob("runs/*/server.stderr"))
                invalid_text = invalid_stderr.read_text(encoding="utf-8")
                self.assertIn(field, invalid_text)
                invalid_stderr.write_text(invalid_text.replace(field, replacement, 1), encoding="utf-8")
                invalid_result = self.run_parser(invalid)
                self.assertNotEqual(invalid_result.returncode, 0)
                verdict = json.loads((invalid / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(verdict["verdict"], "INVALID_ARTIFACT")

    def test_offload_barrier_enforces_production_action_contract(self) -> None:
        runner = load_runner_module()
        parser = load_parser_module()
        observation = {
            "source": "paged_sample_mincore", "action": "offload", "decision_id": "1",
            "seq_id": "0", "transaction_id": "7", "server_pid": "123",
            "before_available": "1", "before_object_id": "1", "before_generation": "1",
            "before_page_size": "4096", "before_total_bytes": "8192",
            "before_resident_bytes": "8192", "before_total_pages": "2", "before_resident_pages": "2",
            "after_available": "1", "after_object_id": "1", "after_generation": "1",
            "after_page_size": "4096", "after_total_bytes": "8192",
            "after_resident_bytes": "4096", "after_total_pages": "2", "after_resident_pages": "1",
        }
        text = marker("kv_g0_s1_resident_observation", observation) + "\n" \
            + marker("kv_pressure_unified_action", ACTION_FIELDS) + "\n"
        self.assertIsNotNone(runner.find_offload_barrier_pair(text, "RSS_ABSOLUTE"))
        self.assertEqual(
            len(parser.qualified_offload_pairs([ACTION_FIELDS], [observation], "RSS_ABSOLUTE")), 1)
        for field, invalid in (
            ("state", "PRESSURE"),
            ("source", "CGROUP_RATIO"),
            ("sample_valid", "0"),
            ("stale", "1"),
            ("pressure_basis_valid", "0"),
            ("budget_active", "0"),
            ("budget_observed_excess_bytes", "0"),
            ("offload_attempted", "0"),
            ("outcome", "no_op"),
            ("state_changed", "0"),
            ("blocks", "0"),
            ("bytes", "0"),
            ("relieved_bytes", "0"),
            ("shortfall_bytes", "1"),
            ("io_failure", "1"),
        ):
            with self.subTest(field=field):
                action = dict(ACTION_FIELDS)
                action[field] = invalid
                invalid_text = marker("kv_g0_s1_resident_observation", observation) + "\n" \
                    + marker("kv_pressure_unified_action", action) + "\n"
                self.assertIsNone(runner.find_offload_barrier_pair(invalid_text, "RSS_ABSOLUTE"))
                self.assertEqual(
                    parser.qualified_offload_pairs([action], [observation], "RSS_ABSOLUTE"), [])
        for field, invalid in (
            ("before_available", "0"),
            ("after_available", "0"),
            ("after_object_id", "2"),
            ("after_generation", "2"),
            ("after_resident_bytes", "8192"),
        ):
            with self.subTest(resident_field=field):
                resident = dict(observation)
                resident[field] = invalid
                invalid_text = marker("kv_g0_s1_resident_observation", resident) + "\n" \
                    + marker("kv_pressure_unified_action", ACTION_FIELDS) + "\n"
                self.assertIsNone(runner.find_offload_barrier_pair(invalid_text, "RSS_ABSOLUTE"))
                self.assertEqual(
                    parser.qualified_offload_pairs([ACTION_FIELDS], [resident], "RSS_ABSOLUTE"), [])

    def test_offload_barrier_timeout_fails_closed_without_resume(self) -> None:
        artifact, runner = self.run_incomplete_artifact("no_offload")
        self.assertEqual(runner.returncode, 1, runner.stderr)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
        self.assertIsNone(execution["qualification"]["resume"])
        responses = next(artifact.glob("runs/*/responses.jsonl")).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(responses), 2)
        stdout = next(artifact.glob("runs/*/server.stdout")).read_text(encoding="utf-8")
        self.assertNotIn("ordinal=3", stdout)

    def test_offload_barrier_requires_matching_resident_drop(self) -> None:
        artifact, runner = self.run_incomplete_artifact("action_only")
        self.assertEqual(runner.returncode, 1, runner.stderr)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
        self.assertIsNone(execution["qualification"]["resume"])

    def test_offload_barrier_rejects_mismatched_correlation_key(self) -> None:
        for mode in ("mismatch_decision", "mismatch_transaction", "mismatch_seq"):
            with self.subTest(mode=mode):
                artifact, runner = self.run_incomplete_artifact(mode)
                self.assertEqual(runner.returncode, 1, runner.stderr)
                execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
                self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
                self.assertIsNone(execution["qualification"]["resume"])

    def test_offload_barrier_rejects_zero_physical_drop(self) -> None:
        artifact, runner = self.run_incomplete_artifact("drop_zero")
        self.assertEqual(runner.returncode, 1, runner.stderr)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["qualification"]["offload_barrier"]["status"], "timeout")
        self.assertIsNone(execution["qualification"]["resume"])

    def test_resume_is_real_measurement_replay_after_barrier(self) -> None:
        artifact = self.run_real_artifact()
        run = json.loads(next(artifact.glob("runs/*/run.json")).read_text(encoding="utf-8"))
        self.assertEqual(
            [item["request_id"] for item in run["request_plan"]],
            ["seed", "request", "request"],
        )
        self.assertEqual(
            [item["measurement"] for item in run["request_plan"]],
            [False, True, False],
        )
        self.assertNotIn("__qualification", json.dumps(run))
        runner_source = RUNNER.read_text(encoding="utf-8")
        self.assertNotIn("X-KV-Benchmark-Phase", runner_source)
        self.assertNotIn("__qualification_offload__", runner_source)
        self.assertNotIn("__qualification_resume__", runner_source)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        barrier = execution["qualification"]["offload_barrier"]
        resume = execution["qualification"]["resume"]
        self.assertEqual(barrier["status"], "passed")
        self.assertGreater(resume["started_mono_ns"], barrier["completed_mono_ns"])
        self.assertEqual(resume["request_id"], "request")

    def test_synthetic_observation_modes_match_server_semantics(self) -> None:
        for mode, slots_expected, marker_expected in (
            ("1", False, True),
            ("preflight", True, False),
            ("both", True, True),
        ):
            with self.subTest(mode=mode):
                slots, stderr = self.probe_synthetic_observation_mode(mode)
                has_slots_resident = "kv_resident" in slots[0]
                self.assertEqual(has_slots_resident, slots_expected)
                self.assertEqual("kv_g0_s1_resident_observation" in stderr, marker_expected)

    def test_v2_swap_out_without_swap_in_is_invalid(self) -> None:
        artifact = self.run_real_artifact(mode="swap_in_zero")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("no swap-in/read", result["errors"][0])

    def test_v2_prefetch_noop_and_zero_restore_are_invalid(self) -> None:
        artifact = self.run_real_artifact(mode="prefetch_noop")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("completed graph-allowed PREFETCH", result["errors"][0])

    def test_v2_qualification_uses_transaction_only_without_slots_authority(self) -> None:
        artifact = self.run_real_artifact()
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(
            execution["environment"]["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"], "1")
        slots_before = json.loads(next(artifact.glob("runs/*/slots_before.json")).read_text(encoding="utf-8"))
        self.assertNotIn("kv_resident", slots_before["body_json"][0])
        stderr = next(artifact.glob("runs/*/server.stderr")).read_text(encoding="utf-8")
        self.assertIn("kv_g0_s1_resident_observation", stderr)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "QUALIFICATION_PASS")
        observation = result["physical_observations"][0]
        self.assertEqual(observation["authority"], "transaction_local_mincore_only")
        # V2 qualification has a single OFFLOAD barrier with one transaction-local
        # resident drop (8192 -> 4096 = 4096 B). There is no RELEASE phase, so the
        # OFFLOAD relief must be attributed to transaction_local_mincore and the
        # total must be release(0) + offload(4096), never 0 (the F16 regression).
        self.assertEqual(observation["offload_authority"], "transaction_local_mincore")
        self.assertEqual(observation["offload_physical_relief_bytes"], 4096)
        self.assertEqual(observation["release_authority"], "not_applicable")
        self.assertEqual(observation["release_physical_relief_bytes"], 0)
        self.assertEqual(len(observation["transaction_local_offload"]), 1)
        self.assertEqual(
            observation["transaction_local_offload"][0]["resident_drop_bytes"], 4096)
        summary = result["action_summary"]
        self.assertEqual(summary["offload_physical_relief_bytes"], 4096)
        self.assertEqual(summary["release_physical_relief_bytes"], 0)
        self.assertEqual(summary["total_physical_relief_bytes"],
                         summary["release_physical_relief_bytes"]
                         + summary["offload_physical_relief_bytes"])
        self.assertEqual(summary["total_physical_relief_bytes"], 4096)
        self.assertEqual(summary["physical_relief_authority"]["offload"],
                         "transaction_local_mincore")

    def test_v2_characterization_offload_relief_sums_two_transaction_local_drops(self) -> None:
        # Regression for the F16 OFFLOAD physical-relief attribution defect. The V2
        # characterization target fixture emits two positive OFFLOAD actions (decision
        # ids 2 and 3), each backed by a transaction_local_mincore resident drop of
        # 4096 B. Their resident drops must sum to 8192, and the action_summary total
        # must equal release + offload (conservation). The per-run offload relief must
        # carry the transaction_local_mincore authority, never substitute action.bytes
        # or relieved_bytes for the mincore drop.
        artifact = self.run_characterization_artifact()
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        observation = result["physical_observations"][0]
        pairs = observation["transaction_local_offload"]
        self.assertEqual(len(pairs), 2)
        self.assertEqual(
            [pair["resident_drop_bytes"] for pair in pairs], [4096, 4096])
        self.assertEqual(
            sum(pair["resident_drop_bytes"] for pair in pairs), 8192)
        self.assertEqual(observation["offload_authority"],
                         "transaction_local_mincore")
        self.assertEqual(
            sum(int(pair["resident_observation"]["before_resident_bytes"])
                for pair in pairs)
            - sum(int(pair["resident_observation"]["after_resident_bytes"])
                  for pair in pairs), 8192)
        summary = result["action_summary"]
        self.assertEqual(summary["offload_physical_relief_bytes"], 8192)
        self.assertEqual(summary["total_physical_relief_bytes"],
                         summary["release_physical_relief_bytes"]
                         + summary["offload_physical_relief_bytes"])
        # The single physical barrier above checks that logical action.bytes or
        # relieved_bytes never substituted for the mincore drop: action.bytes is
        # also 4096-per-action here, but relief must equal the resident drop, not
        # some other field; if the parser had substituted action["relieved_bytes"]
        # the value would still be 8192 by coincidence — so fail-closed the
        # substitution by also asserting the per-pair resident drop matches the
        # mincore before-after delta exactly, proving the source is mincore.
        for pair in pairs:
            observation_record = pair["resident_observation"]
            self.assertEqual(
                pair["resident_drop_bytes"],
                int(observation_record["before_resident_bytes"])
                - int(observation_record["after_resident_bytes"]))
            self.assertEqual(
                observation_record["source"], "paged_sample_mincore")

    def test_v2_qualification_relief_fail_closed_on_tampered_resident_drop(self) -> None:
        # Boundary: with the barrier transaction-local mincore pair removed (no
        # matching resident drop), the qualifier rejects the artifact; the parser
        # MUST never backfill offload_physical_relief_bytes from action.relieved_bytes
        # or action.bytes. The existing barrier contract already rejects this run,
        # but here we additionally assert that an INVALID ARTIFACT verdict leaves no
        # plausible relief attribution at all.
        artifact = self.run_real_artifact()
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        stderr_path.write_text(
            "\n".join(
                line for line in stderr_path.read_text(encoding="utf-8").splitlines()
                if "kv_g0_s1_resident_observation" not in line) + "\n",
            encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        # No successful parse -> no physical relief is reported; the artifact is
        # rejected, and action_summary/physical_observations are not emitted.
        self.assertNotIn("physical_observations", result)
        self.assertNotIn("action_summary", result)

    def test_v2_qualification_zero_relief_stays_zero_for_resident_policy(self) -> None:
        # Boundary: resident policy issues no offload, so qualified_offload_pairs is
        # empty and offload_physical_relief_bytes must remain exactly 0 (never
        # synthesized from action.bytes/io), with release relief also 0.
        artifact = self.run_real_artifact(policy="resident")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        observation = result["physical_observations"][0]
        self.assertEqual(observation["authority"], "slots_pre_post")
        self.assertEqual(observation["offload_physical_relief_bytes"], 0)
        self.assertEqual(observation["release_physical_relief_bytes"], 0)
        self.assertEqual(observation["offload_authority"], "not_applicable")
        self.assertEqual(observation["transaction_local_offload"], [])
        summary = result["action_summary"]
        self.assertEqual(summary["offload_physical_relief_bytes"], 0)
        self.assertEqual(summary["release_physical_relief_bytes"], 0)
        self.assertEqual(summary["total_physical_relief_bytes"], 0)

    def test_characterization_requires_combined_observation_mode(self) -> None:
        runner = load_runner_module()
        parser = load_parser_module()
        value = self.characterization_spec()
        spec, cases, _, _ = runner.validate_spec(value)
        environment = runner.runtime_environment(
            spec, cases["case"], self.root / "combined-observation")
        execution = {"environment": environment}
        parser.validate_execution_environment(execution, cases["case"], "combined", spec)
        self.assertEqual(
            parser.qualified_offload_pairs([ACTION_FIELDS], [], "RSS_ABSOLUTE"), [])
        for mode in ("1", "preflight"):
            invalid = {"environment": dict(environment)}
            invalid["environment"]["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"] = mode
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(parser.ParseError, "physical resident observation mode"):
                    parser.validate_execution_environment(
                        invalid, cases["case"], mode, spec)

    def test_resident_no_swap_in_does_not_require_resume(self) -> None:
        artifact = self.run_real_artifact(policy="resident")
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertIsNone(execution["qualification"]["offload_barrier"])
        self.assertIsNone(execution["qualification"]["resume"])
        self.assertEqual(execution["request_count"], 2)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "QUALIFICATION_PASS")
        self.assertEqual(result["physical_observations"][0]["authority"], "slots_pre_post")

        stderr = next(artifact.glob("runs/*/server.stderr"))
        text = stderr.read_text(encoding="utf-8")
        self.assertIn("bytes_written=0", text)
        stderr.write_text(text.replace("bytes_written=0", "bytes_written=1", 1), encoding="utf-8")
        invalid = self.run_parser(artifact)
        self.assertNotEqual(invalid.returncode, 0)
        invalid_result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("migration IO", invalid_result["errors"][0])

    def test_resident_characterization_records_b_full_without_migration(self) -> None:
        artifact = self.run_characterization_artifact(
            policy="resident", mode="characterization_resident", restore="k1_sync")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "TARGET_REACHED")
        characterization = result["characterization"]
        self.assertEqual(characterization["b_full"]["status"], "AVAILABLE")
        self.assertEqual(characterization["b_full"]["observations"][0]["bytes"], 12288)
        run = characterization["runs"][0]
        self.assertEqual(run["status"], "RESIDENT_BASELINE")
        self.assertEqual(run["resident_after_fill"], 12288)
        self.assertIsNone(run["resident_after_release_settle"])
        self.assertIsNone(run["resident_after_offload_settle"])
        self.assertEqual(run["release_physical_relief_bytes"], 0)
        self.assertEqual(run["offload_physical_relief_bytes"], 0)
        self.assertEqual(run["total_physical_relief_bytes"], 0)
        self.assertEqual(run["physical_relief_bytes"], 0)
        self.assertEqual(run["offload"]["actions"], 0)
        self.assertEqual(run["offload"]["write_syscalls"], 0)
        self.assertEqual(run["resume"]["read_syscalls"], 0)

    def test_release_only_characterization_records_release_without_swap_io(self) -> None:
        artifact = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release", restore="k1_sync")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "RELEASE_SETTLED")
        run = result["characterization"]["runs"][0]
        self.assertEqual(run["release_terminal"], "release_settled")
        self.assertEqual(run["resident_after_fill"], 12288)
        self.assertEqual(run["resident_after_release_settle"], 4096)
        self.assertIsNone(run["resident_after_offload_settle"])
        self.assertEqual(run["release_physical_relief_bytes"], 8192)
        self.assertEqual(run["offload_physical_relief_bytes"], 0)
        self.assertEqual(run["total_physical_relief_bytes"], 8192)
        self.assertEqual(run["release"]["actions"], 1)
        self.assertEqual(run["release"]["blocks"], 2)
        self.assertEqual(run["offload"]["actions"], 0)
        self.assertEqual(run["resume"]["restored_bytes"], 0)
        self.assertTrue(run["performance_eligible"])
        self.assertEqual(run["performance"]["e2e_ms"]["n"], 1)
        self.assertEqual(result["characterization"]["b_release_floor"]["status"], "UNAVAILABLE")
        self.assertEqual(len(result["characterization"]["release_target_points"]), 1)
        execution = json.loads(next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        env = execution["environment"]
        self.assertEqual(env["LLAMA_KV_PAGED_SWAP"], "0")
        self.assertEqual(env["LLAMA_KV_PRESSURE_UNIFIED_ACTION"], "1")
        self.assertNotIn("kv_g0_s1_resident_observation", next(artifact.glob("runs/*/server.stderr")).read_text(encoding="utf-8"))
        io = result["restore_observations"][0]["io"]
        for field in (
            "block_swap_out_calls", "block_swap_in_calls", "backing_write_syscalls",
            "backing_read_syscalls", "bytes_written", "bytes_read",
        ):
            self.assertEqual(int(io[field]), 0)

    def test_release_only_no_candidate_is_valid_zero_relief(self) -> None:
        artifact = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release_no_candidate", restore="k1_sync")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        run = result["characterization"]["runs"][0]
        self.assertEqual(result["verdict"], "RELEASE_FLOOR_PROBE")
        self.assertEqual(run["status"], "RELEASE_FLOOR_PROBE")
        self.assertEqual(run["release_terminal"], "release_no_candidate")
        self.assertFalse(run["performance_eligible"])
        self.assertEqual(run["performance"]["e2e_ms"]["status"], "UNAVAILABLE")
        self.assertEqual(run["resident_after_release_settle"], 12288)
        self.assertEqual(run["release_physical_relief_bytes"], 0)
        self.assertEqual(run["total_physical_relief_bytes"], 0)
        self.assertEqual(result["characterization"]["b_release_floor"]["p50_bytes"], 12288)
        self.assertEqual(result["characterization"]["b_reachable_floor"]["status"], "UNAVAILABLE")
        stderr = next(artifact.glob("runs/*/server.stderr")).read_text(encoding="utf-8")
        self.assertIn("decision_reason=budget_offload_unsupported", stderr)
        self.assertIn("soft_offload_armed_before=1", stderr)

    def test_release_floor_probe_performance_delta_is_unavailable(self) -> None:
        value = self.characterization_spec(
            policy="resident", mode="characterization_release_no_candidate", restore="k1_sync")
        value["cases"] = [
            dict(value["cases"][0], case_id="resident", policy="resident", kv_target_bytes=None, action_target_bytes=None),
            dict(value["cases"][0], case_id="release", policy="release_only", kv_target_bytes=4096, action_target_bytes=8192),
        ]
        value["run_order"] = [
            {"round": 1, "run_order": 1, "case_id": "resident"},
            {"round": 1, "run_order": 2, "case_id": "release"},
        ]
        artifact = self.root / "release-floor-performance"
        runner = self.run_runner(self.write_spec(value, "release-floor-performance.json"), artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        comparison = next(
            item for item in result["characterization"]["comparisons"]
            if item["policy"] == "release_only")
        self.assertEqual(comparison["performance_delta"]["status"], "UNAVAILABLE")
        self.assertEqual(comparison["performance_delta"]["candidate_phase"], "release_only_steady")
        self.assertEqual(result["characterization"]["comparison_aggregate"]["release_only"]["performance_delta_p50"], {})

    def test_release_only_multi_round_aggregate_is_observation_based(self) -> None:
        value = self.characterization_spec(
            policy="release_only", mode="characterization_release_no_candidate", restore="k1_sync")
        value["run_order"] = [
            {"round": 1, "run_order": 1, "case_id": "case"},
            {"round": 2, "run_order": 1, "case_id": "case"},
        ]
        artifact = self.root / "release-only-multi-round"
        runner = self.run_runner(self.write_spec(value, "release-only-multi-round.json"), artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        characterization = result["characterization"]
        self.assertEqual(result["statistics"]["runs"], 2)
        self.assertEqual(len(characterization["b_release_floor"]["observations"]), 2)
        self.assertEqual(characterization["b_release_floor"]["min_bytes"], 12288)
        self.assertEqual(characterization["b_release_floor"]["max_bytes"], 12288)
        self.assertEqual(characterization["b_release_floor"]["p50_bytes"], 12288)
        self.assertEqual(len(characterization["runs"]), 2)
        parser = load_parser_module()
        wrapped = [
            {
                "run_id": item["run_id"], "case_id": item["case_id"],
                "round": item["round"], "policy": item["policy"],
                "characterization": item,
            }
            for item in characterization["runs"]
        ]
        self.assertEqual(
            parser.summarize_characterization(wrapped, "formal")["b_release_floor"]["status"],
            "AVAILABLE")
        wrapped[1]["characterization"]["release_terminal"] = "release_settled"
        self.assertEqual(
            parser.summarize_characterization(wrapped, "formal")["b_release_floor"]["status"],
            "UNAVAILABLE")

    def test_v2_splits_release_and_offload_physical_relief(self) -> None:
        artifact = self.run_characterization_artifact(
            mode="characterization_release_then_offload", restore="k1_sync")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        run = result["characterization"]["runs"][0]
        self.assertEqual(run["resident_after_release_settle"], 8192)
        self.assertEqual(run["resident_after_offload_settle"], 4096)
        self.assertEqual(run["release_physical_relief_bytes"], 4096)
        self.assertEqual(run["offload_physical_relief_bytes"], 4096)
        self.assertEqual(run["total_physical_relief_bytes"], 8192)
        self.assertEqual(run["total_physical_relief_bytes"], run["release_physical_relief_bytes"] + run["offload_physical_relief_bytes"])
        self.assertEqual(run["offload"]["physical_relieved_bytes"], 4096)
        self.assertEqual(run["offload"]["action_relieved_bytes"], 4096)
        self.assertEqual(run["release_physical_relief_authority"], "phase_boundary_slots")
        self.assertEqual(run["offload_physical_relief_authority"], "transaction_local_mincore")
        self.assertEqual(
            run["total_physical_relief_bytes"],
            run["resident_after_fill"] - run["resident_after_offload_settle"],
        )

        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        lines = stderr_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "offload_attempted=1" in line:
                tokens = line.split()
                tokens[tokens.index("bytes=4096")] = "bytes=9999"
                lines[index] = " ".join(tokens)
                break
        stderr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        tampered = json.loads((artifact / "result.json").read_text(encoding="utf-8"))["characterization"]["runs"][0]
        self.assertEqual(tampered["release_physical_relief_bytes"], 4096)
        self.assertEqual(tampered["offload_physical_relief_bytes"], 4096)

    def test_v2_total_physical_relief_requires_conservation(self) -> None:
        artifact = self.run_characterization_artifact(
            mode="characterization_release_then_offload", restore="k1_sync")
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        lines = stderr_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "kv_g0_s1_resident_observation" in line and "decision_id=3" in line:
                lines[index] = line.replace("after_resident_bytes=4096", "after_resident_bytes=2048") \
                    .replace("after_resident_pages=1", "after_resident_pages=0")
                break
        stderr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("conservation", result["errors"][0])

    def test_v2_offload_before_release_boundary_is_invalid(self) -> None:
        artifact = self.run_characterization_artifact(
            mode="characterization_release_then_offload", restore="k1_sync")
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        lines = stderr_path.read_text(encoding="utf-8").splitlines()
        release_index = next(
            index for index, line in enumerate(lines)
            if "kv_pressure_unified_action" in line
            and "decision_id=2" in line
            and "release_attempted=1" in line)
        release_line = lines.pop(release_index)
        offload_action_index = next(
            index for index, line in enumerate(lines)
            if "kv_pressure_unified_action" in line
            and "decision_id=3" in line
            and "offload_attempted=1" in line)
        lines.insert(offload_action_index + 1, release_line)
        stderr_text = "\n".join(lines) + "\n"
        stderr_path.write_text(stderr_text, encoding="utf-8")
        execution_path = next(artifact.glob("runs/*/execution.json"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        settle = execution["characterization"]["settle"]
        end_offset = len(stderr_text.encode("utf-8"))
        settle["stderr_end_offset"] = end_offset
        settle["decision_ids"] = [1, 3]
        settle["offload_decision_ids"] = [3]
        execution["characterization"]["resume"]["stderr_start_offset"] = end_offset
        execution["characterization"]["resume"]["stderr_end_offset"] = end_offset
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("overlaps", result["errors"][0])

    def test_release_only_swap_io_or_prefetch_is_invalid(self) -> None:
        artifact = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release", restore="k1_sync")
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        stderr_path.write_text(
            stderr_path.read_text(encoding="utf-8").replace("bytes_written=0", "bytes_written=1", 1),
            encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")

        artifact = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release", restore="k1_sync")
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        with stderr_path.open("a", encoding="utf-8") as stream:
            stream.write("kv_resume_order_event phase=prefetch decision_id=2 seq_id=0 claimant_epoch=1 transaction_id=1 action=prefetch outcome=completed reason=none graph_allowed=1\\n")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")

    def test_release_only_noop_prefetch_passes_but_positive_restore_fails(self) -> None:
        artifact = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release_noop_prefetch", restore="k1_sync")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "RELEASE_SETTLED")
        self.assertEqual(result["characterization"]["runs"][0]["resume"]["restored_bytes"], 0)
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        stderr = stderr_path.read_text(encoding="utf-8")
        self.assertIn("phase=prefetch", stderr)
        self.assertIn("outcome=no_op", stderr)
        self.assertIn("restored_blocks=0", stderr)
        self.assertIn("restored_bytes=0", stderr)

        positive = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release_noop_prefetch", restore="k1_sync")
        positive_stderr = next(positive.glob("runs/*/server.stderr"))
        positive_text = positive_stderr.read_text(encoding="utf-8")
        positive_stderr.write_text(
            positive_text.replace("restored_blocks=0", "restored_blocks=1", 1)
            .replace("restored_bytes=0", "restored_bytes=4096", 1),
            encoding="utf-8",
        )
        invalid = self.run_parser(positive)
        self.assertNotEqual(invalid.returncode, 0)
        invalid_result = json.loads((positive / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(invalid_result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("positive restore", invalid_result["errors"][0])

        activity = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release_noop_prefetch", restore="k2_pipeline")
        activity_stderr = next(activity.glob("runs/*/server.stderr"))
        activity_text = activity_stderr.read_text(encoding="utf-8")
        activity_stderr.write_text(
            activity_text.replace("restore_scatter_groups=0", "restore_scatter_groups=1", 1),
            encoding="utf-8",
        )
        activity_invalid = self.run_parser(activity)
        self.assertNotEqual(activity_invalid.returncode, 0)
        activity_result = json.loads((activity / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(activity_result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("positive restore activity", activity_result["errors"][0])

    def test_characterization_aggregate_uses_three_layer_baseline_and_phase_delta(self) -> None:
        value = self.characterization_spec(policy="resident", mode="characterization_release_then_offload", restore="k1_sync")
        value["cases"] = [
            dict(value["cases"][0], case_id="resident", policy="resident", kv_target_bytes=None, action_target_bytes=None),
            dict(value["cases"][0], case_id="release", policy="release_only", kv_target_bytes=4096, action_target_bytes=8192),
            dict(value["cases"][0], case_id="v2", policy="v2", kv_target_bytes=4096, action_target_bytes=8192),
        ]
        value["run_order"] = [
            {"round": 1, "run_order": 1, "case_id": "resident"},
            {"round": 1, "run_order": 2, "case_id": "release"},
            {"round": 1, "run_order": 3, "case_id": "v2"},
        ]
        value["workload"]["repeat"] = 2
        artifact = self.root / "characterization-three-layer"
        runner = self.run_runner(self.write_spec(value, "three-layer.json"), artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        characterization = result["characterization"]
        self.assertEqual(characterization["b_full"]["p50_bytes"], 12288)
        self.assertEqual(characterization["b_release_floor"]["status"], "UNAVAILABLE")
        self.assertEqual(len(characterization["release_target_points"]), 1)
        self.assertEqual(characterization["release_target_points"][0]["resident_after_release_settle"], 4096)
        self.assertEqual(characterization["b_reachable_floor"]["status"], "UNAVAILABLE")
        self.assertEqual(characterization["baseline_ladder"]["resident_b_full"]["p50_bytes"], 12288)
        comparisons = {item["policy"]: item for item in characterization["comparisons"]}
        self.assertEqual(comparisons["release_only"]["performance_delta"]["candidate_phase"], "release_only_steady")
        self.assertEqual(comparisons["v2"]["performance_delta"]["candidate_phase"], "post_resume_steady")
        self.assertEqual(result["action_summary"]["total_physical_relief_bytes"], result["action_summary"]["release_physical_relief_bytes"] + result["action_summary"]["offload_physical_relief_bytes"])

    def test_resident_target_change_does_not_change_action_target(self) -> None:
        runner = load_runner_module()
        first_value = self.characterization_spec()
        first_spec, first_cases, _, _ = runner.validate_spec(first_value)
        first_env = runner.runtime_environment(
            first_spec, first_cases["case"], self.root / "first-backing")

        second_value = json.loads(json.dumps(first_value))
        second_value["cases"][0]["kv_target_bytes"] = 8192
        second_spec, second_cases, _, _ = runner.validate_spec(second_value)
        second_env = runner.runtime_environment(
            second_spec, second_cases["case"], self.root / "second-backing")

        self.assertEqual(first_env["LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES"], "8192")
        self.assertEqual(second_env["LLAMA_KV_PRESSURE_UNIFIED_ACTION_TARGET_BYTES"], "8192")
        self.assertEqual(first_env["LLAMA_KV_RESIDENT_TARGET_BYTES"], "4096")
        self.assertEqual(second_env["LLAMA_KV_RESIDENT_TARGET_BYTES"], "8192")

    def test_characterization_marks_resume_once_and_excludes_steady_repeats(self) -> None:
        value = self.characterization_spec()
        value["workload"]["repeat"] = 3
        spec = self.write_spec(value, "characterization-repeats.json")
        artifact = self.root / "characterization-repeats"
        runner = self.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)

        responses = [
            json.loads(line) for line in next(artifact.glob("runs/*/responses.jsonl"))
            .read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [record["measurement_phase"] for record in responses],
            ["fill", "resume", "post_resume_steady", "post_resume_steady"],
        )
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        run = result["characterization"]["runs"][0]
        self.assertEqual(run["resume_measurement"]["measurement_phase"], "resume")
        self.assertEqual(len(run["post_resume_steady_measurements"]), 2)
        self.assertEqual(run["performance"]["e2e_ms"]["n"], 1)
        self.assertEqual(run["performance"]["tpot_ms_per_token"]["n"], 1)
        self.assertEqual(run["performance"]["throughput_tokens_per_second"]["n"], 1)
        self.assertEqual(run["post_resume_steady_performance"]["e2e_ms"]["n"], 2)
        self.assertEqual(result["statistics"]["by_case"]["case"]["runs"][0]["e2e_ms"]["n"], 1)

    def test_formal_characterization_uses_independent_rounds_not_repeat(self) -> None:
        runner = load_runner_module()
        parser = load_parser_module()
        value = self.characterization_spec()
        value["run_kind"] = "formal"
        value["workload"]["repeat"] = 1
        value["run_order"] = [
            {"round": 1, "run_order": 1, "case_id": "case"},
            {"round": 2, "run_order": 1, "case_id": "case"},
        ]
        runner.validate_spec(value)
        parser.validate_spec(value)
        value["run_order"] = [{"round": 1, "run_order": 1, "case_id": "case"}]
        with self.assertRaisesRegex(runner.RunnerError, "at least two independent rounds"):
            runner.validate_spec(value)
        with self.assertRaisesRegex(parser.ParseError, "at least two independent rounds"):
            parser.validate_spec(value)

    def test_characterization_waits_for_multiple_offloads_before_target_and_resume(self) -> None:
        artifact = self.run_characterization_artifact()
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["spec"]["cases"][0]["kv_target_bytes"], 4096)
        self.assertEqual(manifest["spec"]["cases"][0]["action_target_bytes"], 8192)
        self.assertEqual(manifest["planned_runs"][0]["kv_target_bytes"], 4096)
        self.assertEqual(manifest["planned_runs"][0]["action_target_bytes"], 8192)
        execution = json.loads(
            next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        settle = execution["characterization"]["settle"]
        resume = execution["characterization"]["resume"]
        self.assertEqual(
            execution["environment"]["LLAMA_KV_G0_S1_RESIDENT_OBSERVATION"], "both")
        slots_after_fill = json.loads(
            next(artifact.glob("runs/*/slots_after_fill.json")).read_text(encoding="utf-8"))
        self.assertIn("kv_resident", slots_after_fill["body_json"][0])
        self.assertIn(
            "kv_g0_s1_resident_observation",
            next(artifact.glob("runs/*/server.stderr")).read_text(encoding="utf-8"),
        )
        self.assertEqual(settle["status"], "target_reached")
        self.assertEqual(settle["decision_ids"], [1, 2, 3])
        self.assertEqual(settle["offload_decision_ids"], [2, 3])
        self.assertEqual(settle["physical_resident_bytes"], 4096)
        self.assertGreater(resume["started_mono_ns"], settle["completed_mono_ns"])

        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "TARGET_REACHED")
        run = result["characterization"]["runs"][0]
        self.assertEqual(run["requested_target_bytes"], 4096)
        self.assertEqual(run["action_target_bytes"], 8192)
        self.assertEqual(run["resident_after_fill"], 12288)
        self.assertEqual(run["resident_settled"], 4096)
        self.assertEqual(run["physical_relief_bytes"], 8192)
        self.assertEqual(run["budget_debt_after"], 0)
        self.assertEqual(run["offload"]["actions"], 2)
        self.assertEqual(run["offload"]["blocks"], 2)
        self.assertEqual(run["offload"]["bytes"], 8192)
        self.assertEqual(run["offload"]["write_syscalls"], 2)
        self.assertEqual(run["resume"]["restored_blocks"], 2)
        self.assertEqual(run["resume"]["restored_bytes"], 8192)
        self.assertEqual(run["resume"]["read_syscalls"], 2)
        self.assertEqual(run["resume"]["gate_us"], 2)
        self.assertEqual(run["resume"]["total_us"], 6)
        self.assertEqual(run["resident_after_resume"], 12288)
        self.assertEqual(run["k2"]["pipeline_wall_us"], 11)
        self.assertEqual(run["k2"]["read_us"], 2)
        self.assertEqual(run["k2"]["unpack_us"], 3)
        self.assertEqual(run["k2"]["prefault"]["us"], 0)
        self.assertEqual(run["k2"]["scatter"]["us"], 10)
        self.assertEqual(run["performance"]["ttft_ms"]["status"], "UNAVAILABLE")
        self.assertEqual(run["performance"]["e2e_ms"]["status"], "AVAILABLE")
        self.assertEqual(
            run["performance"]["throughput_tokens_per_second"]["status"], "AVAILABLE")

    def test_first_offload_is_not_a_characterization_terminal(self) -> None:
        runner = load_runner_module()
        first = ACTION_FIELDS | {
            "decision_reason": "budget_offload_submitted",
            "budget_resident_bytes": "12288",
            "budget_observed_excess_bytes": "8192",
            "budget_debt_after_bytes": "4096",
            "debt_after_bytes": "4096",
            "shortfall_bytes": "4096",
        }
        observations, terminal = runner.budget_settle_observations(
            marker("kv_pressure_unified_action", first), "RSS_ABSOLUTE", 4096, 8192, 64, 0)
        self.assertEqual([item["decision_id"] for item in observations], [1])
        self.assertIsNone(terminal)

    def test_unmet_terminal_is_valid_floor_with_actual_resident(self) -> None:
        artifact = self.run_characterization_artifact(mode="characterization_unmet")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "UNMET_FLOOR")
        run = result["characterization"]["runs"][0]
        self.assertEqual(run["requested_target_bytes"], 4096)
        self.assertEqual(run["resident_settled"], 8192)
        self.assertEqual(run["unmet_budget_bytes"], 4096)
        self.assertNotEqual(run["requested_target_bytes"], run["resident_settled"])
        self.assertGreater(run["resume"]["restored_bytes"], 0)
        io = result["restore_observations"][0]["io"]
        for field in ("block_swap_in_calls", "backing_read_syscalls", "bytes_read"):
            self.assertGreater(int(io[field]), 0)
        execution = json.loads(
            next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertIsNotNone(execution["characterization"]["resume"])
        self.assertEqual(execution["request_count"], 2)
        floor = result["characterization"]["b_reachable_floor"]
        self.assertEqual(floor["status"], "AVAILABLE")
        self.assertEqual(floor["requested_target_bytes"], 4096)
        self.assertEqual(floor["observations"][0]["resident_settled"], 8192)

    def test_unmet_floor_requires_positive_swap_in_read_and_restore(self) -> None:
        for field, replacement in (
            ("block_swap_in_calls", "0"),
            ("backing_read_syscalls", "0"),
            ("bytes_read", "0"),
            ("restored_blocks", "0"),
            ("restored_bytes", "0"),
        ):
            with self.subTest(field=field):
                artifact = self.run_characterization_artifact(mode="characterization_unmet")
                stderr_path = next(artifact.glob("runs/*/server.stderr"))
                text = stderr_path.read_text(encoding="utf-8")
                old_value = {
                    "restored_blocks": "1", "restored_bytes": "4096", "bytes_read": "4096",
                }.get(field, "1")
                stderr_path.write_text(
                    text.replace(f"{field}={old_value}", f"{field}={replacement}", 1),
                    encoding="utf-8",
                )
                parsed = self.run_parser(artifact)
                self.assertNotEqual(parsed.returncode, 0)
                result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["verdict"], "INVALID_ARTIFACT")

    def test_unmet_floor_resumes_without_slot_resident_snapshot(self) -> None:
        value = self.characterization_spec(mode="characterization_unmet")
        value["environment"]["KV_SYNTHETIC_OMIT_V2_SLOTS_RESIDENT"] = "1"
        spec = self.write_spec(value, "characterization-unmet-no-slot-resident.json")
        artifact = self.root / "characterization-unmet-no-slot-resident"
        runner = self.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        execution = json.loads(
            next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["request_count"], 2)
        self.assertIsNone(execution["characterization"]["settle"]["physical_resident_bytes"])
        self.assertIsNotNone(execution["characterization"]["resume"])
        self.assertEqual(
            len(next(artifact.glob("runs/*/responses.jsonl")).read_text(encoding="utf-8").splitlines()),
            2,
        )
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "UNMET_FLOOR")
        run = result["characterization"]["runs"][0]
        self.assertIsNone(run["resident_after_fill"])
        self.assertIsNone(run["resident_after_release_settle"])
        self.assertIsNone(run["resident_settled"])
        self.assertIsNone(run["resident_after_resume"])
        self.assertEqual(
            run["resident_views"],
            {"after_fill": None, "after_release_settle": None, "settled": None, "after_resume": None},
        )
        self.assertEqual(
            run["budget_resident_views"]["after_fill"]["authority"], "budget_view_marker")
        self.assertEqual(
            run["budget_resident_views"]["settled"]["authority"], "budget_view_marker")
        self.assertIsNone(run["release_physical_relief_bytes"])
        self.assertEqual(run["offload_physical_relief_bytes"], 4096)
        self.assertIsNone(run["total_physical_relief_bytes"])
        self.assertEqual(
            run["release_physical_relief_authority"],
            "unavailable_no_independent_physical_authority",
        )
        self.assertEqual(run["offload_physical_relief_authority"], "transaction_local_mincore")
        self.assertIsNone(run["physical_relief_bytes"])
        self.assertGreater(run["resume"]["restored_bytes"], 0)
        self.assertEqual(
            result["characterization"]["b_reachable_floor"]["status"], "UNAVAILABLE")
        self.assertEqual(
            result["action_summary"]["total_physical_relief_bytes"], None)

    def test_v2_marker_fallback_cannot_create_formal_physical_relief(self) -> None:
        value = self.characterization_spec(mode="characterization_unmet")
        value["environment"]["KV_SYNTHETIC_OMIT_V2_SLOTS_RESIDENT"] = "1"
        artifact = self.root / "characterization-unmet-marker-authority"
        runner = self.run_runner(self.write_spec(value, "marker-authority.json"), artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        run = result["characterization"]["runs"][0]
        self.assertIsNone(run["release_physical_relief_bytes"])
        self.assertIsNone(run["total_physical_relief_bytes"])
        self.assertEqual(run["offload_physical_relief_authority"], "transaction_local_mincore")
        self.assertEqual(
            result["physical_observations"][0]["authority"],
            "transaction_local_mincore_only",
        )
        self.assertEqual(
            run["budget_resident_views"]["after_fill"],
            {"authority": "budget_view_marker", "resident_bytes": 12288},
        )
        self.assertEqual(
            run["budget_resident_views"]["settled"],
            {"authority": "budget_view_marker", "resident_bytes": 8192},
        )

    def test_v2_partial_resident_missing_is_invalid(self) -> None:
        artifact = self.run_characterization_artifact(mode="characterization_unmet")
        self.remove_slot_resident(artifact, "slots_release_settled.json")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("partially missing", result["errors"][0])

    def test_v2_target_reached_without_transaction_observation_is_invalid(self) -> None:
        artifact = self.run_characterization_artifact(mode="characterization_target")
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        stderr_path.write_text(
            "\n".join(
                line for line in stderr_path.read_text(encoding="utf-8").splitlines()
                if "kv_g0_s1_resident_observation" not in line
            ) + "\n",
            encoding="utf-8",
        )
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("transaction-local", result["errors"][0])

    def test_v2_target_reached_resident_missing_is_invalid(self) -> None:
        artifact = self.run_characterization_artifact(mode="characterization_target")
        self.remove_slot_resident(artifact, "slots_before.json")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("no valid physical resident observation", result["errors"][0])

    def test_v2_unrelated_positive_prefetch_cannot_satisfy_unmet_floor(self) -> None:
        artifact = self.run_characterization_artifact(mode="characterization_unmet")
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        lines = stderr_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "kv_resume_order_event" in line or "kv_resume_stage_timing" in line:
                lines[index] = line.replace("seq_id=0", "seq_id=7", 1)
        stderr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("not selected", result["errors"][0])

    def test_release_only_claimant_epoch_mismatch_is_invalid(self) -> None:
        artifact = self.run_characterization_artifact(
            policy="release_only", mode="characterization_release_noop_prefetch", restore="k1_sync")
        stderr_path = next(artifact.glob("runs/*/server.stderr"))
        lines = stderr_path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if "kv_resume_order_event" in line and "phase=prefetch" in line:
                lines[index] = line.replace("claimant_epoch=1", "claimant_epoch=2", 1)
                break
        stderr_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("paired graph gate", result["errors"][0])

    def test_characterization_timeout_is_invalid_and_never_resumes(self) -> None:
        value = self.characterization_spec(mode="characterization_timeout")
        value["workload"]["characterization"]["settle_timeout_seconds"] = 0.25
        artifact = self.root / "characterization-timeout"
        runner = self.run_runner(self.write_spec(value, "characterization-timeout.json"), artifact)
        self.assertEqual(runner.returncode, 1, runner.stderr)
        execution = json.loads(
            next(artifact.glob("runs/*/execution.json")).read_text(encoding="utf-8"))
        self.assertEqual(execution["characterization"]["settle"]["status"], "timeout")
        self.assertEqual(execution["characterization"]["settle"]["offload_decision_ids"], [2])
        self.assertIsNone(execution["characterization"]["resume"])
        responses = next(artifact.glob("runs/*/responses.jsonl")).read_text(
            encoding="utf-8").splitlines()
        self.assertEqual(len(responses), 1)
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")

    def test_transient_staging_is_not_subtracted_from_steady_target(self) -> None:
        artifact = self.run_characterization_artifact()
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        run = json.loads((artifact / "result.json").read_text(
            encoding="utf-8"))["characterization"]["runs"][0]
        self.assertEqual(run["requested_target_bytes"], 4096)
        self.assertEqual(run["resident_settled"], 4096)
        self.assertEqual(run["transient_staging_peak_bytes"], 2048)
        self.assertEqual(run["transient_staging_bound_bytes"], 2048)
        self.assertEqual(run["status"], "TARGET_REACHED")

    def test_requested_target_cannot_be_replaced_by_actual_resident(self) -> None:
        artifact = self.run_characterization_artifact(mode="characterization_unmet")
        execution_path = next(artifact.glob("runs/*/execution.json"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        execution["characterization"]["requested_target_bytes"] = 8192
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("requested target", result["errors"][0])

    def test_nonstreaming_first_byte_is_rejected_as_ttft(self) -> None:
        artifact = self.run_real_artifact()
        response = next(artifact.glob("runs/*/responses.jsonl"))
        record = json.loads(response.read_text(encoding="utf-8").splitlines()[0])
        record["first_byte_mono_ns"] = record["started_mono_ns"]
        response.write_text(json.dumps(record) + "\n" + "\n".join(response.read_text(encoding="utf-8").splitlines()[1:]) + "\n", encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("non-streaming", result["errors"][0])

    def test_slots_and_cleanup_are_fail_closed(self) -> None:
        artifact = self.run_real_artifact()
        slots = next(artifact.glob("runs/*/slots_before.json"))
        value = json.loads(slots.read_text(encoding="utf-8"))
        value["http_status"] = 404
        slots.write_text(json.dumps(value), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("HTTP 200", result["errors"][0])

        artifact = self.run_real_artifact()
        cleanup = next(artifact.glob("runs/*/cleanup.json"))
        value = json.loads(cleanup.read_text(encoding="utf-8"))
        value["server"]["residual_process"] = True
        cleanup.write_text(json.dumps(value), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("residual", result["errors"][0])

    def test_expected_memory_max_mismatch_is_rejected(self) -> None:
        value = self.spec()
        value["cgroup"]["expected_memory_max"] = "not-the-current-limit"
        result = self.run_runner(self.write_spec(value), self.root / "bad-cgroup")
        self.assertNotEqual(result.returncode, 0)

    def test_missing_finite_cgroup_authority_is_unsupported_before_workload(self) -> None:
        value = self.spec()
        value["pressure_basis"] = {
            "authority": "cgroup_finite",
            "low_water_kb": None,
            "pressure_kb": None,
            "critical_kb": None,
        }
        value["cgroup"]["expected_memory_max"] = None
        artifact = self.root / "missing-pressure-authority"
        result = self.run_runner(self.write_spec(value), artifact)
        self.assertEqual(result.returncode, 3, result.stderr)
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runner_status"], "UNSUPPORTED")
        self.assertEqual(manifest["unsupported"]["stage"], "pre_workload_pressure_authority")
        self.assertFalse(any((artifact / "runs").iterdir()))
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 3, parsed.stderr)
        verdict = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(verdict["verdict"], "UNSUPPORTED")

    def test_formal_release_only_requires_finite_cgroup_authority(self) -> None:
        value = self.characterization_spec(
            policy="release_only", mode="characterization_release_no_candidate", restore="k1_sync")
        value["run_kind"] = "formal"
        value["run_order"] = [
            {"round": 1, "run_order": 1, "case_id": "case"},
            {"round": 2, "run_order": 1, "case_id": "case"},
        ]
        runner = load_runner_module()
        spec, _, plan, _ = runner.validate_spec(value)
        with self.assertRaisesRegex(runner.UnsupportedPlan, "formal budget"):
            runner.validate_pressure_authority(spec, plan)
        parser = load_parser_module()
        with self.assertRaisesRegex(parser.ParseError, "formal budget"):
            parser.validate_formal_pressure_authority(
                spec, plan, "run_complete")
        parser.validate_formal_pressure_authority(spec, plan, "UNSUPPORTED")

    def test_budget_sweep_plan_requires_canonical_targets_and_reverse_rounds(self) -> None:
        runner = load_runner_module()
        parser = load_parser_module()
        value = self.characterization_spec(
            policy="resident", mode="characterization_resident", restore="k1_sync")
        release_targets = list(runner.CANONICAL_BUDGET_RELEASE_TARGETS)
        offload_targets = list(runner.CANONICAL_BUDGET_OFFLOAD_TARGETS)
        cases = [{
            "case_id": "resident",
            "policy": "resident",
            "kv_representation": "paged",
            "loading_mode": "exact",
            "restore": "k1_sync",
            "prefault": "off",
            "kv_target_bytes": None,
            "action_target_bytes": None,
        }]
        case_by_signature = {("resident", None): "resident"}
        for index, target in enumerate(release_targets):
            case_id = f"release-{index}"
            cases.append({
                "case_id": case_id, "policy": "release_only", "kv_representation": "paged",
                "loading_mode": "exact", "restore": "k1_sync", "prefault": "off",
                "kv_target_bytes": target,
                "action_target_bytes": runner.CANONICAL_BUDGET_ACTION_TARGET_BYTES,
            })
            case_by_signature[("release_only", target)] = case_id
        for index, target in enumerate(offload_targets):
            case_id = f"offload-{index}"
            cases.append({
                "case_id": case_id, "policy": "v2", "kv_representation": "paged",
                "loading_mode": "exact", "restore": "k2_pipeline", "prefault": "off",
                "kv_target_bytes": target,
                "action_target_bytes": runner.CANONICAL_BUDGET_ACTION_TARGET_BYTES,
            })
            case_by_signature[("v2", target)] = case_id
        first_order = [
            ("resident", None),
            ("v2", offload_targets[0]),
            ("release_only", release_targets[0]),
            ("v2", offload_targets[1]),
            ("release_only", release_targets[1]),
            ("v2", offload_targets[2]),
            ("release_only", release_targets[2]),
            ("v2", offload_targets[3]),
        ]
        second_order = list(reversed(first_order))
        value.update({
            "run_kind": "formal",
            "workload": dict(value["workload"], repeat=2),
            "cases": cases,
            "run_order": [
                {"round": round_id, "run_order": order, "case_id": case_by_signature[signature]}
                for round_id, order_list in ((1, first_order), (2, second_order))
                for order, signature in enumerate(order_list, start=1)
            ],
            "budget_sweep": {
                "release_targets_bytes": release_targets,
                "offload_targets_bytes": offload_targets,
                "action_target_bytes": runner.CANONICAL_BUDGET_ACTION_TARGET_BYTES,
                "max_blocks": runner.CANONICAL_BUDGET_MAX_BLOCKS,
                "rounds": 2,
                "order_mode": runner.BUDGET_SWEEP_ORDER_MODE,
            },
        })
        normalized, _, plan, _ = runner.validate_spec(value)
        self.assertEqual(normalized["budget_sweep"]["rounds"], 2)
        parser.validate_spec(value)
        self.assertEqual(len(plan), 16)

        invalid = json.loads(json.dumps(value))
        invalid["budget_sweep"]["offload_targets_bytes"][0] = 1.0
        with self.assertRaisesRegex(runner.RunnerError, "positive absolute byte"):
            runner.validate_spec(invalid)
        with self.assertRaisesRegex(parser.ParseError, "positive absolute byte"):
            parser.validate_spec(invalid)

    def test_budget_curve_uses_actual_resident_and_keeps_round_aggregates(self) -> None:
        parser = load_parser_module()
        sweep = {
            "release_targets_bytes": list(parser.CANONICAL_BUDGET_RELEASE_TARGETS),
            "offload_targets_bytes": list(parser.CANONICAL_BUDGET_OFFLOAD_TARGETS),
            "action_target_bytes": parser.CANONICAL_BUDGET_ACTION_TARGET_BYTES,
            "max_blocks": parser.CANONICAL_BUDGET_MAX_BLOCKS,
            "rounds": 2,
            "order_mode": parser.BUDGET_SWEEP_ORDER_MODE,
        }
        metric = lambda unit, value: {
            "status": "AVAILABLE", "unit": unit, "n": 1,
            "p50": value, "p95": value, "p99": value,
        }
        runs = []
        baseline_bytes = parser.REFERENCE_B_FULL_BYTES
        for round_id in (1, 2):
            runs.append({
                "run_id": f"resident-{round_id}", "case_id": "resident", "round": round_id,
                "policy": "resident", "status": "RESIDENT_BASELINE",
                "resident_after_fill": baseline_bytes,
            })
            for segment, policy, targets in (
                ("release_segment", "release_only", sweep["release_targets_bytes"]),
                ("offload_segment", "v2", sweep["offload_targets_bytes"]),
            ):
                for target in targets:
                    release_relief = 4096 if policy == "v2" else baseline_bytes - target
                    offload_relief = 4096 if policy == "v2" else 0
                    total_relief = release_relief + offload_relief
                    performance = {
                        "e2e_ms": metric("ms", float(target) / 1024),
                        "tpot_ms_per_token": metric("ms/token", 2.0),
                        "throughput_tokens_per_second": metric("tokens/s", 500.0),
                        "ttft_ms": {
                            "status": "UNAVAILABLE", "unit": "ms", "n": 0,
                            "p50": None, "p95": None, "p99": None,
                        },
                    }
                    runs.append({
                        "run_id": f"{policy}-{round_id}-{target}",
                        "case_id": f"{policy}-{target}", "round": round_id,
                        "policy": policy,
                        "status": "RELEASE_SETTLED" if policy == "release_only" else "TARGET_REACHED",
                        "performance_eligible": True,
                        "release_terminal": "release_settled",
                        "requested_target_bytes": target,
                        "resident_after_fill": baseline_bytes,
                        "resident_after_release_settle": target if policy == "release_only" else baseline_bytes - 4096,
                        "resident_after_release_settle_authority": "slots_physical",
                        "resident_settled": target,
                        "resident_settled_authority": "slots_physical",
                        "release_physical_relief_bytes": release_relief,
                        "release_physical_relief_authority": "phase_boundary_slots",
                        "offload_physical_relief_bytes": offload_relief,
                        "offload_physical_relief_authority": "transaction_local_mincore",
                        "total_physical_relief_bytes": total_relief,
                        "total_physical_relief_authority": "sum_of_independent_physical_authorities",
                        "memory_saved_bytes": total_relief,
                        "memory_saved_ratio": total_relief / baseline_bytes,
                        "release": {"actions": 1, "blocks": 1, "bytes": release_relief},
                        "offload": {"actions": 1 if policy == "v2" else 0, "blocks": 1 if policy == "v2" else 0},
                        "io": {}, "resume": {"restored_bytes": 4096, "gate_us": 2, "total_us": 6},
                        "k2": {"read_us": 2, "unpack_us": 3, "prefault": {}, "scatter": {}},
                        "transient_staging_peak_bytes": 2048,
                        "transient_staging_bound_bytes": 2048,
                        "performance": performance,
                        "post_resume_steady_performance": performance,
                    })
        curve = parser.build_budget_curve(runs, sweep, "formal")
        self.assertEqual(curve["status"], "READY")
        self.assertEqual(curve["reference_anchors"]["b_full_bytes"], baseline_bytes)
        self.assertEqual(len(curve["release_segment"]["points"]), 6)
        self.assertEqual(len(curve["offload_segment"]["points"]), 8)
        self.assertEqual(
            [point["actual_settled_resident_bytes"] for point in curve["release_segment"]["points"]],
            sorted(parser.CANONICAL_BUDGET_RELEASE_TARGETS * 2),
        )
        self.assertEqual(len(curve["offload_segment"]["per_round"]), 2)
        self.assertEqual(
            curve["offload_segment"]["target_aggregates"][0]["actual_settled_resident_bytes"]["p50"],
            float(parser.CANONICAL_BUDGET_OFFLOAD_TARGETS[0]),
        )
        point = curve["offload_segment"]["points"][0]
        self.assertIn("resume_restored_bytes", point)
        self.assertIn("k2_prefault", point)
        self.assertEqual(
            point["total_physical_relief_bytes"],
            point["release_physical_relief_bytes"] + point["offload_physical_relief_bytes"],
        )
        self.assertIsNone(curve["knee"])

    def test_budget_curve_excludes_no_candidate_and_unmet_without_physical_authority(self) -> None:
        parser = load_parser_module()
        sweep = {
            "release_targets_bytes": [4096], "offload_targets_bytes": [2048],
            "action_target_bytes": 256, "max_blocks": 64, "rounds": 1,
            "order_mode": parser.BUDGET_SWEEP_ORDER_MODE,
        }
        baseline = {
            "run_id": "resident", "case_id": "resident", "round": 1,
            "policy": "resident", "status": "RESIDENT_BASELINE",
            "resident_after_fill": 12288,
        }
        release = {
            "run_id": "release-floor", "case_id": "release-floor", "round": 1,
            "policy": "release_only", "status": "RELEASE_FLOOR_PROBE",
            "release_terminal": "release_no_candidate", "performance_eligible": False,
            "requested_target_bytes": 4096, "resident_after_release_settle": 12288,
            "resident_after_release_settle_authority": "slots_physical",
            "release_physical_relief_bytes": 0,
            "release_physical_relief_authority": "phase_boundary_slots",
            "offload_physical_relief_bytes": 0,
            "offload_physical_relief_authority": "not_applicable",
            "total_physical_relief_bytes": 0,
            "total_physical_relief_authority": "sum_of_independent_physical_authorities",
            "memory_saved_bytes": 0, "memory_saved_ratio": 0.0,
            "release": {"actions": 1, "blocks": 0}, "offload": {"actions": 0, "blocks": 0},
            "io": {}, "resume": {}, "k2": {}, "transient_staging_peak_bytes": 0,
            "transient_staging_bound_bytes": 0, "performance": {},
        }
        unmet = {
            "run_id": "offload-unmet", "case_id": "offload-unmet", "round": 1,
            "policy": "v2", "status": "UNMET_FLOOR", "performance_eligible": True,
            "requested_target_bytes": 2048, "resident_settled": None,
            "resident_settled_authority": "budget_view_marker",
            "release_physical_relief_bytes": None,
            "release_physical_relief_authority": "unavailable_no_independent_physical_authority",
            "offload_physical_relief_bytes": 4096,
            "offload_physical_relief_authority": "transaction_local_mincore",
            "total_physical_relief_bytes": None,
            "total_physical_relief_authority": "unavailable_no_independent_physical_authority",
            "memory_saved_bytes": None, "memory_saved_ratio": None,
            "release": {"actions": 1, "blocks": 0}, "offload": {"actions": 1, "blocks": 1},
            "io": {}, "resume": {"restored_bytes": 4096}, "k2": {},
            "transient_staging_peak_bytes": 0, "transient_staging_bound_bytes": 0,
            "performance": {}, "post_resume_steady_performance": {},
        }
        curve = parser.build_budget_curve([baseline, release, unmet], sweep, None)
        self.assertEqual(curve["status"], "INCOMPLETE")
        self.assertEqual(curve["release_segment"]["points"], [])
        self.assertEqual(curve["offload_segment"]["points"], [])
        self.assertIsNone(curve["offload_segment"]["aggregate"]["actual_settled_resident_bytes"]["p50"])
        diagnostic = curve["offload_segment"]["diagnostics"][0]
        self.assertIsNone(diagnostic["actual_settled_resident_bytes"])
        self.assertEqual(diagnostic["actual_resident_authority"], "UNAVAILABLE")
        self.assertIn("UNMET_FLOOR", diagnostic["reason"])
        with self.assertRaisesRegex(parser.ParseError, "formal budget_sweep physical curve gate"):
            parser.build_budget_curve([baseline, release, unmet], sweep, "formal")

    def test_process_identity_startup_transient_retries_and_timeout_fails_closed(self) -> None:
        runner = load_runner_module()
        expected = ["bash", "sampler.sh", "--sample-process"]
        with mock.patch.object(
                runner, "process_starttime", side_effect=["10", "10", "10", "10", "10"]), \
             mock.patch.object(
                runner, "process_cmdline", side_effect=[
                    runner.RunnerError("process cmdline is malformed for pid 123"),
                    expected,
                    expected,
                ]):
            identity = runner.process_identity(123, expected, timeout_seconds=0.1)
        self.assertEqual(identity["pid"], 123)
        self.assertEqual(identity["starttime_ticks"], 10)
        self.assertEqual(identity["cmdline"], expected)

        with mock.patch.object(runner, "process_starttime", return_value="10"), \
             mock.patch.object(
                runner, "process_cmdline",
                side_effect=runner.RunnerError("process cmdline is malformed for pid 123")):
            with self.assertRaisesRegex(runner.RunnerError, "process cmdline is malformed"):
                runner.process_identity(123, expected, timeout_seconds=0.01)

    def test_wrapper_binding_uses_one_monotonic_deadline_domain(self) -> None:
        script = ROOT / "scripts/kv-controlled-memory-sampler.sh"
        shell = f"""
            source {str(script)!r}
            KV_CONTROLLED_BIND_TIMEOUT_NS=5
            clock_file=$(mktemp)
            date_called=0
            trap 'rm -f "$clock_file"' EXIT
            printf '0' >"$clock_file"
            kv_controlled_read_proc_stat() {{ KV_CONTROLLED_PROC_STATE=S; KV_CONTROLLED_PROC_STARTTIME=1; return 0; }}
            kv_controlled_descendant_pid() {{ printf '123'; }}
            kv_controlled_read_monotonic_ns() {{
                clock_calls=$(<"$clock_file")
                clock_calls=$((clock_calls + 1))
                printf '%s' "$clock_calls" >"$clock_file"
                if (( clock_calls == 1 )); then printf '100'; else printf '106'; fi
            }}
            date() {{ date_called=1; return 99; }}
            rc=0
            kv_controlled_bind_wrapper_child 123 >/dev/null || rc=$?
            printf '%s:%s' "$rc" "$date_called"
        """
        result = subprocess.run(["bash", "-c", shell], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "7:0")

    def test_cleanup_order_is_sampler_then_server(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        sampler_stop = source.index('if sampler is not None:\n            sampler_cleanup = terminate_process')
        server_stop = source.index('if server is not None:\n            server_cleanup = terminate_process')
        self.assertLess(sampler_stop, server_stop)

    def test_process_identity_hash_and_execution_order_are_checked(self) -> None:
        artifact = self.run_real_artifact()
        execution = next(artifact.glob("runs/*/execution.json"))
        value = json.loads(execution.read_text(encoding="utf-8"))
        value["server_identity"]["cmdline_sha256"] = "0" * 64
        execution.write_text(json.dumps(value), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertIn("cmdline hash", result["errors"][0])

    def test_formal_requires_clean_worktree_and_measurements(self) -> None:
        # The dirty-worktree and missing-measurement guards must each fail
        # closed with runner error 2.  Injecting dirty provenance keeps the
        # dirty-guard assertion hermetic: without it the spec's rss_absolute
        # authority falls through to the formal cgroup_finite gate and returns
        # UNSUPPORTED=3, and the result depended on the host repo being dirty.
        runner = load_runner_module()
        value = self.spec()
        value["run_kind"] = "formal"
        value["workload"]["repeat"] = 2
        value["run_order"] = [
            {"round": 1, "run_order": 1, "case_id": "case"},
            {"round": 2, "run_order": 1, "case_id": "case"},
        ]
        spec_path = self.write_spec(value)
        artifact = self.root / "formal-dirty"
        dirty_provenance = {
            "head": "deadbeef", "branch": "test", "diff_sha256": "0" * 64,
            "dirty_status": [" M tracked/file"], "capture_mode": "diagnostic_dirty",
        }
        saved_argv = sys.argv
        saved_stdout, saved_stderr = sys.stdout, sys.stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        try:
            sys.argv = [str(RUNNER), "--spec", str(spec_path), "--output", str(artifact), "--dry-run"]
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            with mock.patch.object(runner, "git_provenance", return_value=dirty_provenance):
                returncode = runner.main()
        finally:
            sys.argv = saved_argv
            sys.stdout, sys.stderr = saved_stdout, saved_stderr
        self.assertEqual(returncode, 2, stderr_buf.getvalue())
        self.assertFalse(artifact.exists(), stderr_buf.getvalue())

        # The "no measurement request" guard is independent of git state: an
        # empty workload.requests fails closed at spec validation before any
        # formal git-provenance gate runs, so it stays here unchanged.
        value["workload"]["requests"] = []
        value["run_kind"] = "qualification"
        value["run_order"] = [{"round": 1, "run_order": 1, "case_id": "case"}]
        result = self.run_runner(self.write_spec(value), self.root / "no-measurement", dry_run=True)
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_v3_policy_is_supported_before_workload(self) -> None:
        value = self.spec(policy="v3")
        spec = self.write_spec(value, "v3.json")
        artifact = self.root / "v3"
        runner = self.run_runner(spec, artifact, dry_run=True)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["runner_status"], "DRY_RUN")
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "DRY_RUN")

    def test_unimplemented_factor_is_unsupported_before_workload(self) -> None:
        for field, value in (("kv_representation", "quantized"), ("loading_mode", "selective"), ("restore", "k3"), ("prefault", "r3")):
            with self.subTest(field=field):
                spec_value = self.spec()
                spec_value["cases"][0][field] = value
                spec = self.write_spec(spec_value, f"unsupported-{field}.json")
                artifact = self.root / f"unsupported-{field}"
                runner = self.run_runner(spec, artifact, dry_run=True)
                self.assertEqual(runner.returncode, 3, runner.stderr)
                self.assertFalse(any((artifact / "runs").iterdir()))
                parsed = self.run_parser(artifact)
                self.assertEqual(parsed.returncode, 3, parsed.stderr)
                result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
                self.assertEqual(result["verdict"], "UNSUPPORTED")

    def test_plan_mutations_fail_closed(self) -> None:
        spec_value = self.spec()
        spec_value["cases"].append({
            "case_id": "case-2", "policy": "resident", "kv_representation": "paged",
            "loading_mode": "exact", "restore": "k1_sync", "prefault": "off", "kv_target_bytes": None,
            "action_target_bytes": None,
        })
        spec_value["run_order"].append({"round": 1, "run_order": 2, "case_id": "case-2"})
        spec = self.write_spec(spec_value, "plan.json")
        artifact = self.root / "plan"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 0)
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["planned_runs"] = list(reversed(manifest["planned_runs"]))
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")
        self.assertIn("planned_runs", result["errors"][0])

    def test_formal_requires_explicit_interleaved_repeats(self) -> None:
        spec_value = self.spec()
        spec_value["run_kind"] = "formal"
        spec = self.write_spec(spec_value, "formal.json")
        artifact = self.root / "formal"
        runner = self.run_runner(spec, artifact, dry_run=True)
        self.assertEqual(runner.returncode, 2, runner.stderr)
        self.assertFalse(artifact.exists())

    def test_unknown_case_fails_closed(self) -> None:
        spec_value = self.spec()
        spec_value["run_order"][0]["case_id"] = "missing-case"
        spec = self.write_spec(spec_value, "unknown-case.json")
        artifact = self.root / "unknown-case"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 2)
        self.assertFalse(artifact.exists())

    def test_duplicate_case_fails_closed(self) -> None:
        spec_value = self.spec()
        spec_value["cases"].append(dict(spec_value["cases"][0]))
        spec = self.write_spec(spec_value, "duplicate-case.json")
        artifact = self.root / "duplicate-case"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 2)
        self.assertFalse(artifact.exists())

    def test_missing_case_in_plan_fails_closed(self) -> None:
        spec_value = self.spec()
        spec_value["cases"].append({
            "case_id": "case-2", "policy": "resident", "kv_representation": "paged",
            "loading_mode": "exact", "restore": "k1_sync", "prefault": "off", "kv_target_bytes": None,
            "action_target_bytes": None,
        })
        spec_value["run_order"].append({"round": 1, "run_order": 2, "case_id": "case-2"})
        spec = self.write_spec(spec_value, "missing-case.json")
        artifact = self.root / "missing-case"
        self.assertEqual(self.run_runner(spec, artifact, dry_run=True).returncode, 0)
        manifest_path = artifact / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["spec"]["cases"] = manifest["spec"]["cases"][:1]
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        parsed = self.run_parser(artifact)
        self.assertNotEqual(parsed.returncode, 0)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "INVALID_ARTIFACT")

    def test_sampler_monotonic_fraction_conversion_regression(self) -> None:
        shell = f"source {str(ROOT / 'scripts/kv-controlled-memory-sampler.sh')!r}; " \
            "for value in 7.01 7.45 7.123456; do " \
            "kv_controlled_monotonic_ns_from_uptime \"$value\"; printf '\\n'; done"
        result = subprocess.run(["bash", "-c", shell], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["7010000000", "7450000000", "7123456000"])

    def test_sampler_active_stop_with_interrupted_clock_reads_exit_zero(self) -> None:
        # SIGTERM arrives while the trap is installed and interrupts the
        # monotonic sampling-clock read inside the initial $() subshell.
        # The trap runs in the parent context and sets KV_CONTROLLED_STOP_REQUESTED=1;
        # the interrupted subshell read returns failure with empty output. The sampler
        # must treat this as a normal stop (rc=0) and must NOT print the clock-failure
        # diagnostic. We drive the read through a long-running child so the real
        # SIGTERM has a stable window to interrupt it.
        import shlex
        script = ROOT / "scripts/kv-controlled-memory-sampler.sh"
        out = self.root / "active-stop-clock.tsv"
        err = self.root / "active-stop-clock.err"
        log = self.root / "active-stop-script.log"
        shell = "\n".join([
            f"source {str(script)!r}",
            "KV_CONTROLLED_SAMPLE_SCHEMA=legacy_v1",
            "date() { printf '%s' '1000000000'; return 0; }",
            "kv_controlled_read_monotonic_ns() { sleep 3; return 1; }",
            f"kv_controlled_sample_bound_process 123 1 {str(out)!r} '' 1 '' "
            f">/dev/null 2>{str(err)!r} &",
            "sp=$!",
            "sleep 0.5",
            "kill -TERM $sp 2>/dev/null || true",
            "wait $sp; echo $?",
        ])
        result = subprocess.run(
            ["bash", "-c", shell], text=True, capture_output=True, check=False)
        log.write_text(result.stdout, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0",
                         f"expected rc=0 for active-stop interrupt, got: {result.stdout!r}; "
                         f"stderr={result.stderr!r}; errfile={err.read_text(encoding='utf-8') if err.exists() else '<empty>'}")
        stderr_text = err.read_text(encoding="utf-8") if err.exists() else ""
        self.assertNotIn("cannot read monotonic sampling clock", stderr_text)
        self.assertNotIn("cannot read realtime sampling clock", stderr_text)

    def test_sampler_real_clock_failure_without_stop_request_exits_twelve(self) -> None:
        # A genuine monotonic clock failure with no stop requested must stay
        # fail-closed at rc=12 and must surface the diagnostic on stderr.
        script = ROOT / "scripts/kv-controlled-memory-sampler.sh"
        out = (self.root / "real-clock-fail.tsv").as_posix()
        err = (self.root / "real-clock-fail.err").as_posix()
        shell = f"""
            source {str(script)!r}
            KV_CONTROLLED_SAMPLE_SCHEMA=legacy_v1
            date() {{ printf '%s' '1000000000'; return 0; }}
            kv_controlled_read_monotonic_ns() {{ return 1; }}
            kv_controlled_sample_bound_process 123 1 {out!r} '' 0.1 '' >/dev/null 2>{err!r}
            rc=$?
            printf '%s' "$rc"
        """
        result = subprocess.run(["bash", "-c", shell], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "12")
        stderr_text = (self.root / "real-clock-fail.err").read_text(encoding="utf-8")
        self.assertIn("cannot read monotonic sampling clock", stderr_text)

    def test_real_short_synthetic_http_and_sampler_path(self) -> None:
        spec = self.write_spec(self.spec(), "real.json")
        artifact = self.root / "real"
        runner = self.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        self.assertFalse((artifact / "result.json").exists())
        samples = (artifact / "runs/r001_o001_case/memory_samples.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(samples[0].split("\t")[:5], [
            "elapsed_ms", "timestamp_mono_ns", "timestamp_realtime_ns", "pid", "starttime_ticks",
        ])
        self.assertGreaterEqual(len(samples), 2)
        parsed = self.run_parser(artifact)
        self.assertEqual(parsed.returncode, 0, parsed.stderr)
        result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(result["verdict"], "QUALIFICATION_PASS")
        self.assertEqual(result["statistics"]["runs"], 1)
        self.assertEqual(result["statistics"]["by_case"]["case"]["runs"][0]["request_count"], 1)
        self.assertEqual(result["statistics"]["by_case"]["case"]["runs"][0]["ttft_ms"]["status"], "UNAVAILABLE")
        self.assertEqual(result["statistics"]["by_case"]["case"]["runs"][0]["e2e_ms"]["status"], "AVAILABLE")
        self.assertEqual(result["action_summary"]["offload_bytes"], 4096)
        round_trip = result["restore_observations"][0]["qualification_round_trip"]
        self.assertEqual(round_trip["offload"]["resident_drop_bytes"], 4096)
        self.assertEqual(round_trip["prefetch"]["outcome"], "completed")
        self.assertEqual(round_trip["prefetch"]["graph_allowed"], "1")
        self.assertEqual(round_trip["timing"]["restored_blocks"], "1")
        self.assertEqual(round_trip["timing"]["restored_bytes"], "4096")
        self.assertGreater(int(result["restore_observations"][0]["io"]["block_swap_in_calls"]), 0)
        self.assertGreater(int(result["restore_observations"][0]["io"]["backing_read_syscalls"]), 0)
        self.assertGreater(int(result["restore_observations"][0]["io"]["bytes_read"]), 0)

        execution_path = next(artifact.glob("runs/*/execution.json"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        self.assertEqual(execution["environment"]["LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE"], "0")
        execution["server_identity"]["starttime_ticks"] += 1
        execution_path.write_text(json.dumps(execution), encoding="utf-8")
        tampered = self.run_parser(artifact)
        self.assertNotEqual(tampered.returncode, 0)
        tampered_result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(tampered_result["verdict"], "INVALID_ARTIFACT")


class LifecycleReplayIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.helper = CanonicalBenchmarkTest("runTest")
        self.helper.setUp()
        self.root = self.helper.root
        self.fake_server = self.helper.fake_server
        self.fake_server.write_text(textwrap.dedent("""
            #!/usr/bin/env python3
            import argparse, json, os, signal, sys
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--port', type=int, required=True)
            parser.add_argument('--host', default='127.0.0.1')
            parser.add_argument('--slot-save-path', required=True)
            args, _ = parser.parse_known_args()
            if not os.path.isdir(args.slot_save_path):
                raise SystemExit('slot-save-path is not a directory')
            erase_log = os.environ.get('ERASE_LOG')
            erase_no_clear = os.environ.get('ERASE_NO_CLEAR') == '1'
            slot_tokens = {0: 0, 1: 0}

            def slot_record(slot):
                n_tokens = slot_tokens[slot]
                return {
                    'id': slot, 'n_ctx': 1024, 'is_processing': False,
                    'n_prompt_tokens': n_tokens,
                    'kv_claimant': {
                        'epoch': 1, 'exhausted': False, 'valid': True,
                        'target_blocks': 1 if n_tokens else 0,
                        'eligible_resident_blocks': 1 if n_tokens else 0,
                        'swapped_blocks': 0, 'shared_blocks': 0, 'blocked_blocks': 0,
                    },
                }

            def stop(_signum, _frame):
                raise SystemExit(0)

            signal.signal(signal.SIGTERM, stop)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def do_GET(self):
                    if self.path == '/health':
                        body = b'{"status":"ok"}'
                    elif self.path == '/slots':
                        body = json.dumps([slot_record(0), slot_record(1)]).encode()
                    else:
                        self.send_response(404); self.end_headers(); return
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers(); self.wfile.write(body)

                def do_POST(self):
                    length = int(self.headers.get('Content-Length', '0'))
                    raw = self.rfile.read(length)
                    if self.path.startswith('/slots/') and self.path.endswith('?action=erase'):
                        slot = int(self.path.split('/')[2].split('?')[0])
                        if erase_log:
                            with open(erase_log, 'a', encoding='utf-8') as stream:
                                stream.write(json.dumps({'path': self.path, 'slot': slot}) + '\\n')
                        n_erased = slot_tokens[slot]
                        if not erase_no_clear:
                            slot_tokens[slot] = 0
                        payload = {'id_slot': slot, 'n_erased': n_erased}
                    elif self.path == '/completion':
                        request = json.loads(raw.decode('utf-8'))
                        prompt = request.get('prompt', [])
                        n_predict = int(request.get('n_predict', 0))
                        slot = int(request['id_slot'])
                        slot_tokens[slot] = len(prompt) + n_predict
                        payload = {
                            'id_slot': slot,
                            'tokens': list(range(n_predict)),
                            'tokens_evaluated': len(prompt),
                            'tokens_predicted': n_predict,
                        }
                    else:
                        self.send_response(404); self.end_headers(); return
                    body = json.dumps(payload, separators=(',', ':')).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(body)))
                    self.end_headers(); self.wfile.write(body)

            ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
        """).strip() + "\n", encoding="utf-8")
        self.fake_server.chmod(self.fake_server.stat().st_mode | stat.S_IXUSR)

    def tearDown(self) -> None:
        self.helper.tearDown()

    def test_completion_driven_revisit_expiry_dead_and_restart(self) -> None:
        fixture = self.root / 'lifecycle-fixture.json'
        fixture.write_text(json.dumps({
            'schema': 'generic-replay/v1',
            'ttl_seconds': 0.05,
            'sessions': [
                {
                    'logical_session_id': 'A', 'lineage_id': 1,
                    'turns': [
                        {'turn': 1, 'timestamp': 0, 'prompt_tokens': [1], 'n_predict': 1},
                        {'turn': 2, 'timestamp': 10000, 'prompt_tokens': [1, 2], 'n_predict': 1},
                        {'turn': 3, 'timestamp': 100000, 'prompt_tokens': [1, 2, 3], 'n_predict': 1},
                    ],
                },
                {
                    'logical_session_id': 'B', 'lineage_id': 2,
                    'turns': [
                        {'turn': 1, 'timestamp': 0, 'prompt_tokens': [4], 'n_predict': 1},
                        {'turn': 2, 'timestamp': 0, 'prompt_tokens': [4, 5], 'n_predict': 1},
                    ],
                },
                {
                    'logical_session_id': 'C', 'lineage_id': 3,
                    'turns': [
                        {'turn': 1, 'timestamp': 20000, 'prompt_tokens': [6], 'n_predict': 1},
                    ],
                },
            ],
        }), encoding='utf-8')
        erase_log = self.root / 'erase.jsonl'
        value = self.helper.spec()
        value['environment']['ERASE_LOG'] = str(erase_log)
        value['workload'] = {
            'warmup': [], 'requests': [], 'repeat': 1,
            'qualification': None, 'characterization': None,
            'replay': {
                'source': 'fixture', 'path': str(fixture), 'time_dilation': 1.0,
                'n_parallel': 2, 'session_ids': None,
                'admission_timeout_seconds': 2.0,
                'lifecycle': {'enabled': True, 'drain_after_last_arrival': True},
            },
        }
        spec = self.helper.write_spec(value, 'lifecycle.json')
        artifact = self.root / 'lifecycle-artifact'
        runner = self.helper.run_runner(spec, artifact)
        self.assertEqual(runner.returncode, 0, runner.stderr)
        parsed = self.helper.run_parser(artifact)
        self.assertEqual(parsed.returncode, 1, parsed.stderr)
        result = json.loads((artifact / 'result.json').read_text(encoding='utf-8'))
        self.assertEqual(result['verdict'], 'INVALID_ARTIFACT')
        self.assertTrue(any(
            'generic replay lacks derived four-phase phase evidence' in error
            for error in result['errors']))
        replay = json.loads(next(artifact.glob('runs/*/replay.json')).read_text(encoding='utf-8'))
        events = replay['lifecycle']
        self.assertTrue(any(item.get('event') == 'REVISIT' for item in events))
        self.assertTrue(any(item.get('event') == 'TTL_EXPIRY' for item in events))
        self.assertTrue(any(item.get('event') == 'DEAD' for item in events))
        self.assertTrue(any(item.get('event') == 'TURN_START' and item.get('trigger') == 'COLD_RESTART' for item in events))
        self.assertTrue(any(item.get('event') == 'TURN_START' and item.get('trigger') == 'CONTINUATION' and item.get('logical_session_id') == 'B' for item in events))
        self.assertGreater(next(item for item in replay['events'] if item.get('logical_session_id') == 'C')['admission_wait_us'], 0)
        erase_rows = erase_log.read_text(encoding='utf-8').splitlines()
        self.assertGreaterEqual(len(erase_rows), 2)
        execution = json.loads(next(artifact.glob('runs/*/execution.json')).read_text(encoding='utf-8'))
        self.assertIn('--slot-save-path', execution['argv'])
        slot_save_path = pathlib.Path(execution['argv'][execution['argv'].index('--slot-save-path') + 1])
        self.assertTrue(slot_save_path.is_dir())
        erase_snapshots = list(next(artifact.glob('runs/*')).glob('slots_after_erase_*.json'))
        self.assertGreaterEqual(len(erase_snapshots), 2)
        for snapshot_path in erase_snapshots:
            snapshot = json.loads(snapshot_path.read_text(encoding='utf-8'))
            slot_id = next(
                item['erase_id_slot'] for item in events
                if item.get('event') == 'DEAD' and item.get('erase_verify_path') == snapshot_path.name)
            load_runner_module().validate_replay_erased_slot_snapshot(snapshot, slot_id)

    def test_lifecycle_expiry_tie_and_erase_response_are_fail_closed(self) -> None:
        runner_module = load_runner_module()
        self.assertTrue(runner_module.replay_arrival_precedes_expiry(99, 100))
        self.assertFalse(runner_module.replay_arrival_precedes_expiry(100, 100))
        self.assertFalse(runner_module.replay_arrival_precedes_expiry(101, 100))
        good = b'{"id_slot":0,"n_erased":3}'
        self.assertEqual(3, runner_module.validate_replay_erase_response(200, good, None, 0)['n_erased'])
        bad_cases = [
            (500, good, None, 0),
            (200, b'not-json', None, 0),
            (200, b'{"id_slot":1,"n_erased":3}', None, 0),
            (200, b'{"id_slot":0,"n_erased":3,"extra":1}', None, 0),
            (200, b'{"id_slot":0,"n_erased":0}', None, 0),
            (200, good, 'transport-error', 0),
        ]
        for status, body, error, slot_id in bad_cases:
            with self.subTest(status=status, body=body, error=error):
                with self.assertRaises(runner_module.RunnerError):
                    runner_module.validate_replay_erase_response(status, body, error, slot_id)

    def test_lifecycle_parser_rejects_premature_slot_reuse_and_generation_drift(self) -> None:
        parser_module = load_parser_module()
        fixture = self.root / 'lifecycle-parser-slot-negative.json'
        fixture.write_text(json.dumps({
            'schema': 'generic-replay/v1', 'ttl_seconds': 0.1,
            'sessions': [
                {'logical_session_id': 'A', 'lineage_id': 1,
                 'turns': [{'turn': 1, 'timestamp': 0, 'prompt_tokens': [1], 'n_predict': 1}]},
                {'logical_session_id': 'B', 'lineage_id': 2,
                 'turns': [{'turn': 1, 'timestamp': 0, 'prompt_tokens': [2], 'n_predict': 1}]},
            ],
        }), encoding='utf-8')
        plan = parser_module.load_replay('fixture', fixture, n_parallel=2, lifecycle=True)
        cfg = {'time_dilation': 1.0, 'n_parallel': 2,
               'lifecycle': {'enabled': True, 'drain_after_last_arrival': True}}
        actual = [
            {'request_id': 'replay_A_1', 'turn': 1, 'cache_prompt': False,
             'planned_arrival_us': 0, 'completed_us': 1},
            {'request_id': 'replay_B_1', 'turn': 1, 'cache_prompt': False,
             'planned_arrival_us': 0, 'completed_us': 1},
        ]
        def start(sid: str, lineage: int, slot: int, generation: int) -> dict[str, object]:
            return {
                'event': 'TURN_START', 'logical_session_id': sid, 'lineage_id': lineage,
                'turn': 1, 'slot_id': slot, 'seq_id': slot, 'runner_generation': generation,
                'lifecycle_generation': 1, 'arrival_us': 0, 'completion_us': None,
                'expiry_us': None, 'request_id': f'replay_{sid}_1', 'trigger': 'COLD_START',
            }
        with self.assertRaises(parser_module.ParseError):
            parser_module.validate_replay_lifecycle(
                'slot-reuse', plan, cfg, actual, [start('A', 1, 0, 1), start('B', 2, 0, 1)])
        with self.assertRaises(parser_module.ParseError):
            parser_module.validate_replay_lifecycle(
                'generation-drift', plan, cfg, actual, [start('A', 1, 0, 2)])

    def test_post_erase_verification_rejects_uncleared_prompt_or_claimant(self) -> None:
        good = {
            'http_status': 200, 'error': None,
            'body_json': [{
                'id': 0, 'is_processing': False, 'n_prompt_tokens': 0,
                'kv_claimant': {
                    'target_blocks': 0, 'eligible_resident_blocks': 0,
                    'swapped_blocks': 0, 'shared_blocks': 0, 'blocked_blocks': 0,
                },
            }],
        }
        runner_module = load_runner_module()
        self.assertEqual(0, runner_module.validate_replay_erased_slot_snapshot(good, 0)['n_prompt_tokens'])
        bad_prompt = json.loads(json.dumps(good))
        bad_prompt['body_json'][0]['n_prompt_tokens'] = 1
        with self.assertRaises(runner_module.RunnerError):
            runner_module.validate_replay_erased_slot_snapshot(bad_prompt, 0)
        bad_claimant = json.loads(json.dumps(good))
        bad_claimant['body_json'][0]['kv_claimant']['swapped_blocks'] = 1
        with self.assertRaises(runner_module.RunnerError):
            runner_module.validate_replay_erased_slot_snapshot(bad_claimant, 0)


class PhaseEvidenceContractTest(unittest.TestCase):
    """Directed tests for the V3-1B-Q1 four-phase evidence contract.

    These exercise the extracted phase-evidence predicates directly against
    field dicts, without standing up the full fake-server artifact pipeline.
    """

    def _offload_action(self, seq_id: int, decision_id: int = 1,
                        transaction_id: int = 7, epoch: int = 1) -> dict[str, str]:
        action = dict(ACTION_FIELDS)
        action["selected_seq_id"] = str(seq_id)
        action["decision_id"] = str(decision_id)
        action["transaction_id"] = str(transaction_id)
        action["selected_claimant_epoch"] = str(epoch)
        return action

    def _resident_drop(self, seq_id: int, decision_id: int = 1,
                       transaction_id: int = 7) -> dict[str, str]:
        return {
            "source": "paged_sample_mincore", "action": "offload",
            "decision_id": str(decision_id), "seq_id": str(seq_id),
            "transaction_id": str(transaction_id), "server_pid": "123",
            "before_available": "1", "before_object_id": "1", "before_generation": "1",
            "before_page_size": "4096", "before_total_bytes": "8192",
            "before_resident_bytes": "8192", "before_total_pages": "2",
            "before_resident_pages": "2",
            "after_available": "1", "after_object_id": "1", "after_generation": "1",
            "after_page_size": "4096", "after_total_bytes": "8192",
            "after_resident_bytes": "4096", "after_total_pages": "2",
            "after_resident_pages": "1",
        }

    def _resume_event(self, seq_id: int, decision_id: int = 1,
                      transaction_id: int = 7, epoch: int = 1,
                      phase: str = "prefetch", outcome: str = "completed",
                      graph_allowed: str = "1") -> dict[str, str]:
        return {
            "phase": phase, "decision_id": str(decision_id), "seq_id": str(seq_id),
            "claimant_epoch": str(epoch), "transaction_id": str(transaction_id),
            "action": "offload-reuse", "outcome": outcome, "reason": "none",
            "graph_allowed": graph_allowed,
        }

    def test_final_competition_single_victim_passes(self) -> None:
        parser = load_parser_module()
        ab = {0, 1}
        action = self._offload_action(0)
        observation = self._resident_drop(0)
        victim = parser.final_competition_selected_victim(
            [action], [observation], "RSS_ABSOLUTE", ab)
        self.assertIsNotNone(victim)
        self.assertEqual(victim["seq_id"], 0)
        self.assertIs(victim["action"], action)

    def test_final_competition_two_victims_fail_closed(self) -> None:
        parser = load_parser_module()
        ab = {0, 1}
        actions = [self._offload_action(0, decision_id=1, transaction_id=7),
                   self._offload_action(1, decision_id=2, transaction_id=8)]
        observations = [self._resident_drop(0, decision_id=1, transaction_id=7),
                        self._resident_drop(1, decision_id=2, transaction_id=8)]
        # Two concurrent state-changing OFFLOAD victims in the same decision set
        # is ambiguous and must fail closed.
        self.assertIsNone(parser.final_competition_selected_victim(
            actions, observations, "RSS_ABSOLUTE", ab))

    def test_final_competition_no_victim_fail_closed(self) -> None:
        parser = load_parser_module()
        ab = {0, 1}
        # An OFFLOAD action with no transaction-local resident drop is not a
        # qualifying victim pair; the final window must fail closed.
        actions = [self._offload_action(0)]
        self.assertIsNone(parser.final_competition_selected_victim(
            actions, [], "RSS_ABSOLUTE", ab))
        # A victim outside the A/B candidate set is not selectable.
        victim_outside = self._offload_action(9, decision_id=3, transaction_id=9)
        obs_outside = self._resident_drop(9, decision_id=3, transaction_id=9)
        self.assertIsNone(parser.final_competition_selected_victim(
            [victim_outside], [obs_outside], "RSS_ABSOLUTE", ab))

    def test_final_competition_trailing_noop_marker_does_not_shadow_victim(self) -> None:
        parser = load_parser_module()
        ab = {0, 1}
        victim_action = self._offload_action(0, decision_id=1, transaction_id=7)
        victim_obs = self._resident_drop(0, decision_id=1, transaction_id=7)
        # A later no-op marker (state_changed=0, no resident drop) that names a
        # different selected_seq_id must not redefine the qualifying victim.
        noop = dict(ACTION_FIELDS)
        noop["decision_id"] = "5"
        noop["transaction_id"] = "11"
        noop["selected_seq_id"] = "1"
        noop["state_changed"] = "0"
        noop["outcome"] = "no_op"
        noop["offload_attempted"] = "0"
        actions = [victim_action, noop]
        observations = [victim_obs]
        victim = parser.final_competition_selected_victim(
            actions, observations, "RSS_ABSOLUTE", ab)
        self.assertIsNotNone(victim)
        self.assertEqual(victim["seq_id"], 0)
        self.assertIs(victim["action"], victim_action)

    def test_phase_evidence_seq_set_separates_ab_binding_from_evidence_count(self) -> None:
        parser = load_parser_module()
        ab_seqs = {0, 1}
        # prepare-offload: both A and B must each carry positive evidence.
        prepare_off_window = {"seq_ids": [0, 1]}
        prepare_off_actions = [self._offload_action(0, 1, 7),
                               self._offload_action(1, 2, 8)]
        prepare_off_obs = [self._resident_drop(0, 1, 7),
                           self._resident_drop(1, 2, 8)]
        self.assertEqual(parser.phase_evidence_seq_set(
            "prepare-offload", prepare_off_window, prepare_off_actions,
            prepare_off_obs, [], "RSS_ABSOLUTE"), ab_seqs)
        # final-competition: the A/B candidate set binds two seqs, but only one
        # state-changing OFFLOAD victim appears in positive evidence.
        final_window = {"seq_ids": [0, 1]}
        final_actions = [self._offload_action(0, 1, 7)]
        final_obs = [self._resident_drop(0, 1, 7)]
        self.assertEqual(parser.phase_evidence_seq_set(
            "final-competition", final_window, final_actions,
            final_obs, [], "RSS_ABSOLUTE"), {0})
        # post-final-restore: the window binds only the single selected victim,
        # and its positive restore evidence set is a singleton, not the A/B pair.
        post_window = {"seq_ids": [0]}
        post_resumes = [self._resume_event(0)]
        self.assertEqual(parser.phase_evidence_seq_set(
            "post-final-restore", post_window, [], [],
            post_resumes, "RSS_ABSOLUTE"), {0})

    def test_post_final_only_selected_restore_passes(self) -> None:
        parser = load_parser_module()
        post_window = {"seq_ids": [0]}
        post_resumes = [self._resume_event(0)]
        evidence = parser.phase_evidence_seq_set(
            "post-final-restore", post_window, [], [], post_resumes, "RSS_ABSOLUTE")
        self.assertEqual(evidence, set(post_window["seq_ids"]))

    def test_post_final_non_selected_restore_fails(self) -> None:
        parser = load_parser_module()
        # The window binds the single selected victim (seq 0), but a non-selected
        # claimant (seq 1) performing a positive restore must surface as evidence
        # outside the selected set.
        post_window = {"seq_ids": [0]}
        post_resumes = [self._resume_event(0), self._resume_event(1, decision_id=4, transaction_id=9)]
        evidence = parser.phase_evidence_seq_set(
            "post-final-restore", post_window, [], [], post_resumes, "RSS_ABSOLUTE")
        self.assertEqual(evidence, {0, 1})
        self.assertNotEqual(evidence, set(post_window["seq_ids"]))

    def test_runner_victim_trailing_noop_marker_does_not_shadow_victim(self) -> None:
        # Runner and parser must share one authority over the final victim: the
        # unique qualifying state-changing OFFLOAD + transaction-local resident
        # drop pair within A/B. A trailing no-op action marker after the real
        # OFFLOAD must not redefine the selection, and runner must select the same
        # victim as the parser so the post-final restore barrier advances to the
        # correct claimant.
        runner = load_runner_module()
        parser = load_parser_module()
        ab = {0, 1}
        victim_action = self._offload_action(0, decision_id=1, transaction_id=7)
        victim_obs = self._resident_drop(0, decision_id=1, transaction_id=7)
        noop = dict(ACTION_FIELDS)
        noop["decision_id"] = "5"
        noop["transaction_id"] = "11"
        noop["selected_seq_id"] = "1"
        noop["state_changed"] = "0"
        noop["outcome"] = "no_op"
        noop["offload_attempted"] = "0"
        text = (marker("kv_g0_s1_resident_observation", victim_obs) + "\n"
                + marker("kv_pressure_unified_action", victim_action) + "\n"
                + marker("kv_pressure_unified_action", noop) + "\n")
        runner_victim = runner.final_competition_victim(text, "RSS_ABSOLUTE", ab)
        parser_victim = parser.final_competition_selected_victim(
            [victim_action, noop], [victim_obs], "RSS_ABSOLUTE", ab)
        self.assertIsNotNone(runner_victim)
        self.assertIsNotNone(parser_victim)
        self.assertEqual(runner_victim["seq_id"], 0)
        self.assertEqual(runner_victim["seq_id"], parser_victim["seq_id"])
        self.assertEqual(
            (runner_victim["decision_id"], runner_victim["transaction_id"]),
            (parser_victim["decision_id"], parser_victim["transaction_id"]))
        # The trailing no-op (selected_seq_id=1) did not become the victim.
        self.assertNotEqual(runner_victim["seq_id"], 1)

    def test_runner_victim_multiple_victims_fail_closed(self) -> None:
        # Two concurrent state-changing OFFLOAD victims in A/B within the same
        # final window is ambiguous; both runner and parser must fail closed
        # (return None) rather than committing to one selection.
        runner = load_runner_module()
        parser = load_parser_module()
        ab = {0, 1}
        a_action = self._offload_action(0, decision_id=1, transaction_id=7)
        a_obs = self._resident_drop(0, decision_id=1, transaction_id=7)
        b_action = self._offload_action(1, decision_id=2, transaction_id=8)
        b_obs = self._resident_drop(1, decision_id=2, transaction_id=8)
        text = (marker("kv_g0_s1_resident_observation", a_obs) + "\n"
                + marker("kv_pressure_unified_action", a_action) + "\n"
                + marker("kv_g0_s1_resident_observation", b_obs) + "\n"
                + marker("kv_pressure_unified_action", b_action) + "\n")
        self.assertIsNone(runner.final_competition_victim(text, "RSS_ABSOLUTE", ab))
        self.assertIsNone(parser.final_competition_selected_victim(
            [a_action, b_action], [a_obs, b_obs], "RSS_ABSOLUTE", ab))


    def test_runner_final_victim_uses_frozen_barrier_window(self) -> None:
        runner = load_runner_module()
        with tempfile.TemporaryDirectory() as tmp:
            stderr_path = pathlib.Path(tmp) / "server.stderr"
            a_action = self._offload_action(0, decision_id=1, transaction_id=7)
            a_observation = self._resident_drop(0, decision_id=1, transaction_id=7)
            stderr_path.write_text(
                marker("kv_g0_s1_resident_observation", a_observation) + "\n"
                + marker("kv_pressure_unified_action", a_action) + "\n",
                encoding="utf-8",
            )
            barrier_end = stderr_path.stat().st_size
            b_action = self._offload_action(1, decision_id=2, transaction_id=8)
            b_observation = self._resident_drop(1, decision_id=2, transaction_id=8)
            with stderr_path.open("a", encoding="utf-8") as stream:
                stream.write(marker("kv_g0_s1_resident_observation", b_observation) + "\n")
                stream.write(marker("kv_pressure_unified_action", b_action) + "\n")
            frozen = runner.complete_stderr_window(stderr_path, 0, barrier_end)
            victim = runner.final_competition_victim(frozen, "RSS_ABSOLUTE", {0, 1})
            self.assertIsNotNone(victim)
            self.assertEqual(victim["seq_id"], 0)
            self.assertNotIn("transaction_id=8", frozen)

    def test_replay_admission_seq_authority_is_independent_and_fail_closed(self) -> None:
        parser = load_parser_module()
        admission = [
            {"logical_session_id": "A", "status": "admitted", "slot_id": 4, "seq_id": 4},
            {"logical_session_id": "B", "status": "admitted", "slot_id": 9, "seq_id": 9},
            {"logical_session_id": "A", "status": "lineage_live", "slot_id": 4, "seq_id": 4},
            {"logical_session_id": "B", "status": "lineage_live", "slot_id": 9, "seq_id": 9},
        ]
        actual = [
            {"logical_session_id": "A", "slot_id": 4, "seq_id": 4},
            {"logical_session_id": "B", "slot_id": 9, "seq_id": 9},
        ]
        self.assertEqual(
            {"A": 4, "B": 9},
            parser.replay_admission_seq_authority(admission, actual, "valid"),
        )
        invalid = {
            "missing": [item for item in admission if item["logical_session_id"] != "B"],
            "drift": admission + [{"logical_session_id": "A", "status": "lineage_live", "slot_id": 5, "seq_id": 5}],
            "duplicate": [
                *[item for item in admission if item["logical_session_id"] != "B"],
                {"logical_session_id": "B", "status": "lineage_live", "slot_id": 4, "seq_id": 4},
            ],
            "forged": [
                {**item, "slot_id": 5} if item["logical_session_id"] == "A" and item["status"] == "admitted" else item
                for item in admission
            ],
        }
        for name, candidate in invalid.items():
            with self.subTest(name=name):
                with self.assertRaises(parser.ParseError):
                    parser.replay_admission_seq_authority(candidate, actual, name)

    def test_replay_phase_windows_match_admission_authority_and_selected_victim(self) -> None:
        parser = load_parser_module()

        def window(seq_ids: list[int], expected: list[int] | None = None) -> dict[str, object]:
            return {
                "session_ids": ["A", "B"],
                "seq_ids": list(seq_ids),
                "expected_seq_ids": list(seq_ids if expected is None else expected),
            }

        self.assertEqual(
            {4, 9},
            parser.validate_replay_phase_window_bindings(
                "final-competition", window([4, 9]), {4, 9}, None, "valid"),
        )
        for name, candidate in {
            "window_drift": window([4, 5]),
            "window_duplicate": window([4, 4]),
            "window_forged_expected": window([4, 9], [4, 5]),
            "window_wrong_session": {**window([4, 9]), "session_ids": ["A", "C"]},
        }.items():
            with self.subTest(name=name):
                with self.assertRaises(parser.ParseError):
                    parser.validate_replay_phase_window_bindings(
                        "prepare-offload", candidate, {4, 9}, None, name)
        self.assertEqual(
            {4},
            parser.validate_replay_phase_window_bindings(
                "post-final-restore", window([4]), {4, 9}, 4, "post-valid"),
        )
        for selected in (9, None):
            with self.subTest(selected=selected):
                candidate = window([4]) if selected is not None else window([3])
                with self.assertRaises(parser.ParseError):
                    parser.validate_replay_phase_window_bindings(
                        "post-final-restore", candidate, {4, 9}, selected, "post-invalid")


    def _feedback_offload_action(
            self, seq_id: int, decision_id: int, transaction_id: int,
            epoch: int = 1, elapsed_us: int = 13, offload_bytes: int = 4096,
            relief_bytes: int = 4096, object_id: int = 1, generation: int = 1) -> dict[str, str]:
        action = self._offload_action(seq_id, decision_id, transaction_id, epoch)
        action.update({
            "action_elapsed_us": str(elapsed_us),
            "bytes": str(offload_bytes),
            "physical_relief_available": "1",
            "physical_relief_bytes": str(relief_bytes),
            "physical_object_id": str(object_id),
            "physical_generation": str(generation),
        })
        return action

    def _feedback_resident_drop(
            self, seq_id: int, decision_id: int, transaction_id: int,
            object_id: int = 1, generation: int = 1) -> dict[str, str]:
        observation = self._resident_drop(seq_id, decision_id, transaction_id)
        observation.update({
            "before_object_id": str(object_id),
            "after_object_id": str(object_id),
            "before_generation": str(generation),
            "after_generation": str(generation),
        })
        return observation

    def _feedback_restore(
            self, seq_id: int, decision_id: int, transaction_id: int,
            epoch: int = 1, restored_bytes: int = 4096, gate_us: int = 7,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        prefetch = self._resume_event(
            seq_id, decision_id, transaction_id, epoch,
            phase="prefetch", outcome="completed", graph_allowed="1")
        prefetch["action"] = "prefetch"
        graph_gate = dict(prefetch)
        graph_gate["phase"] = "graph_gate"
        timing = {
            "decision_id": str(decision_id), "seq_id": str(seq_id),
            "transaction_id": str(transaction_id), "restored_blocks": "1",
            "restored_bytes": str(restored_bytes), "queue_us": "1",
            "gate_us": str(gate_us), "graph_us": "1", "total_us": "9",
        }
        return [prefetch, graph_gate], [timing]

    def _prepare_feedback_fixture(self):
        parser = load_parser_module()
        offload_actions = [
            self._feedback_offload_action(0, 1, 7),
            self._feedback_offload_action(1, 2, 8),
        ]
        offload_observations = [
            self._feedback_resident_drop(0, 1, 7),
            self._feedback_resident_drop(1, 2, 8),
        ]
        offload_feedback = parser.replay_prepare_offload_feedback(
            offload_actions, offload_observations, "RSS_ABSOLUTE", {0, 1}, "fixture.offload")
        resumes: list[dict[str, str]] = []
        timings: list[dict[str, str]] = []
        for seq_id, decision_id, transaction_id in ((0, 11, 17), (1, 12, 18)):
            seq_resumes, seq_timings = self._feedback_restore(
                seq_id, decision_id, transaction_id)
            resumes.extend(seq_resumes)
            timings.extend(seq_timings)
        restore_feedback = parser.replay_prepare_restore_feedback(
            resumes, timings, offload_feedback, "fixture.restore")
        final_action = self._feedback_offload_action(0, 30, 70)
        final_action["claimants"] = (
            "0:1:1:0:1:1:1:0:0:0;1:1:1:0:1:1:1:0:0:0")
        scores = []
        for seq_id in (0, 1):
            offload = offload_feedback[seq_id]
            restore = restore_feedback[seq_id]
            scores.append({
                "seq_id": str(seq_id), "eligible": "1", "cost_aware": "1",
                "fallback_reason": "none",
                "expected_offload_write_cost_us": str(offload["offload_elapsed_us"]),
                "expected_restore_gate_cost_us": str(restore["restore_gate_us"]),
                "last_offload_bytes": str(offload["offload_bytes"]),
                "last_restore_bytes": str(restore["restore_bytes"]),
                "actual_relief_bytes": str(offload["physical_relief_bytes"]),
                "round_trip_count": "1",
                "physical_object_id": str(offload["object_id"]),
                "physical_generation": str(offload["generation"]),
            })
        return parser, offload_feedback, restore_feedback, final_action, scores, resumes, timings

    def test_prepare_feedback_history_propagates_for_both_claimants(self) -> None:
        parser, offload, restore, final_action, scores, _, _ = self._prepare_feedback_fixture()
        history = parser.validate_prepare_feedback_history(
            final_action, scores, offload, restore, {0, 1}, 0, "fixture.history")
        self.assertEqual(13, history[0]["offload_elapsed_us"])
        self.assertEqual(7, history[1]["restore_gate_us"])
        self.assertEqual(4096, history[0]["physical_relief_bytes"])
        self.assertEqual(1, history[1]["round_trip_count"])

    def test_prepare_feedback_offload_cost_mismatch_fails_closed(self) -> None:
        parser, offload, restore, final_action, scores, _, _ = self._prepare_feedback_fixture()
        scores[0] = dict(scores[0])
        scores[0]["expected_offload_write_cost_us"] = "99"
        with self.assertRaises(parser.ParseError):
            parser.validate_prepare_feedback_history(
                final_action, scores, offload, restore, {0, 1}, 0, "offload-cost")

    def test_prepare_feedback_restore_gate_or_bytes_mismatch_fails_closed(self) -> None:
        parser, offload, restore, final_action, scores, _, _ = self._prepare_feedback_fixture()
        for field, value in {
            "expected_restore_gate_cost_us": "99",
            "last_restore_bytes": "9999",
        }.items():
            with self.subTest(field=field):
                candidate = [dict(score) for score in scores]
                candidate[0][field] = value
                with self.assertRaises(parser.ParseError):
                    parser.validate_prepare_feedback_history(
                        final_action, candidate, offload, restore, {0, 1}, 0, f"restore-{field}")

    def test_prepare_feedback_missing_positive_restore_fails_closed(self) -> None:
        parser, offload, _, _, _, resumes, timings = self._prepare_feedback_fixture()
        resumes = [event for event in resumes if event["seq_id"] != "1"]
        timings = [timing for timing in timings if timing["seq_id"] != "1"]
        with self.assertRaises(parser.ParseError):
            parser.replay_prepare_restore_feedback(
                resumes, timings, offload, "restore-missing")

    def test_prepare_feedback_stale_object_generation_or_epoch_fails_closed(self) -> None:
        parser, offload, restore, final_action, scores, _, _ = self._prepare_feedback_fixture()
        for field, value in {
            "physical_object_id": "2",
            "physical_generation": "2",
        }.items():
            with self.subTest(field=field):
                candidate = [dict(score) for score in scores]
                candidate[0][field] = value
                with self.assertRaises(parser.ParseError):
                    parser.validate_prepare_feedback_history(
                        final_action, candidate, offload, restore, {0, 1}, 0, f"stale-{field}")
        stale_epoch_action = dict(final_action)
        stale_epoch_action["claimants"] = stale_epoch_action["claimants"].replace(
            "0:1:", "0:2:", 1)
        with self.assertRaises(parser.ParseError):
            parser.validate_prepare_feedback_history(
                stale_epoch_action, scores, offload, restore, {0, 1}, 0, "stale-epoch")

    def test_prepare_feedback_missing_history_or_round_trip_fails_closed(self) -> None:
        parser, offload, restore, final_action, scores, _, _ = self._prepare_feedback_fixture()
        no_round_trip = [dict(score) for score in scores]
        no_round_trip[0]["round_trip_count"] = "0"
        with self.assertRaises(parser.ParseError):
            parser.validate_prepare_feedback_history(
                final_action, no_round_trip, offload, restore, {0, 1}, 0, "round-trip")
        missing_history = [dict(score) for score in scores]
        missing_history[0].pop("last_restore_bytes")
        with self.assertRaises(parser.ParseError):
            parser.validate_prepare_feedback_history(
                final_action, missing_history, offload, restore, {0, 1}, 0, "history-missing")


if __name__ == "__main__":
    unittest.main()
