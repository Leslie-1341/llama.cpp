#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GT-trace-1A CLI: compile a Bailian trace into frozen workload manifests.

Usage:
  python3 -m trace_compiler.cli compile --trace traceA \
      --out /root/oscomp/kv_logs/gt-trace-1a/traceA \
      [--cal-fraction 0.5] [--target-fraction 0.9] \
      [--nslice-typical 100] [--nslice-revisit 100] [--nslice-burst 50]

Pipeline (all streaming / bounded):
  1. Stream and validate the trace while reconstructing lineages; accumulate
     only scalar whole-trace hash accounting.
  2. Chronological split and frozen calibration on the calibration prefix.
  3. Select three held-out whole-session slices from the evaluation prefix.
  4. Run one bounded second pass over the raw trace, reading hash_ids only for
     the union of selected (chat_id, turn) pairs.
  5. Use that selected mapping for selected-prefix characterization and all
     session-namespaced materialization; persistence is fail-closed.
  6. Write arrival/request-plan event streams, manifests, and offline reports.

This is read/write to the output dir only; it touches NONE of the V3-1A
canonical runner/parser files and none of src/tools/server.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from .calibration import (
    build_frozen_calibration,
    make_time_split,
    assign_lineage_to_split,
)
from .characterization import (
    HashUseSummary,
    characterize_global_hash_reuse,
    characterize_lineages_accessory,
    replay_calibration,
)
from .common import (
    OFFLINE_TTL_REPLAY_SCOPE,
    RUNTIME_EVENT_SCOPE,
    TRACE_FILES,
    TraceRecord,
    trace_file_path,
    MANIFEST_SCHEMA_VERSION,
    TRACE_REPO_HEAD,
)
from .contract import BudgetQualificationContract
from .lineage import Lineage, reconstruct_lineages
from .reader import TraceReadStream
from .sampler import (
    SLICE_BURST_LONG_CTX,
    SLICE_REVISIT_HEAVY,
    SLICE_TYPICAL,
    assemble_slice,
    select_burst_long_context,
    select_revisit_heavy,
    select_typical,
)


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, sort_keys=True, indent=2, ensure_ascii=False)
        f.write("\n")


def _second_pass_hash_provider(
    file_path: str,
    wanted: set,
    expected_block_counts: Optional[Dict[Tuple[int, int], int]] = None,
) -> Dict[Tuple[int, int], Tuple[int, ...]]:
    """Read selected hash_ids in one bounded second pass.

    The first pass deliberately drops every record's hash_ids. This pass
    touches the hash list only when the raw record's (chat_id, turn) key is in
    `wanted`; missing or duplicate selected records fail closed.
    """
    out: Dict[Tuple[int, int], Tuple[int, ...]] = {}
    if not wanted:
        return out
    with open(file_path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            obj = json.loads(raw.decode("utf-8"))
            key = (int(obj["chat_id"]), int(obj["turn"]))
            if key not in wanted:
                continue
            if key in out:
                raise ValueError(f"duplicate selected raw record: {key}")
            record = TraceRecord.from_raw(obj)
            out[key] = record.hash_ids
    missing = wanted.difference(out)
    if missing:
        raise ValueError(
            "selected second-pass missing hash_ids for "
            f"{len(missing)} turns: {sorted(missing)[:3]}"
        )
    if expected_block_counts is not None:
        mismatches = [
            (key, expected, len(out[key]))
            for key, expected in expected_block_counts.items()
            if len(out[key]) != expected
        ]
        if mismatches:
            raise ValueError(
                "selected second-pass block_count mismatch: "
                f"{mismatches[:3]}"
            )
    return out


def compile_trace(
    trace_key: str,
    out_dir: str,
    cal_fraction: float = 0.5,
    target_fraction: float = 0.9,
    nslice_typical: int = 100,
    nslice_revisit: int = 100,
    nslice_burst: int = 50,
    seed: int = 1337,
) -> dict:
    file_path = trace_file_path(trace_key)
    out_dir = os.path.join(out_dir, trace_key)
    os.makedirs(out_dir, exist_ok=True)

    # 1-2. One validated streaming pass. Lineage reconstruction retains only
    # MinimalTurn metadata; hash accounting is scalar and never stores a
    # per-record or per-lineage hash sequence.
    rs = TraceReadStream(trace_key, file_path)
    hash_summary = HashUseSummary()

    def records_with_summary():
        for record in rs:
            hash_summary.observe(record.hash_ids)
            yield record

    accepted, rejected, stats = reconstruct_lineages(records_with_summary())
    prov = rs.provenance

    # 3-4. Split + calibrate on complete calibration lineages.
    split = make_time_split(accepted, cal_fraction)
    cal_lineages = [
        lineage for lineage in accepted
        if assign_lineage_to_split(lineage, split) == "calibration"
    ]
    eval_lineages = [
        lineage for lineage in accepted
        if assign_lineage_to_split(lineage, split) == "evaluation"
    ]
    frozen_cal = build_frozen_calibration(cal_lineages, split, target_fraction)

    # 5. Runtime metrics are explicitly offline projected characterization;
    # they are not used to populate the persisted runtime stream.
    cache_char = replay_calibration(cal_lineages, frozen_cal.frozen_ttl_seconds)
    acc = characterize_lineages_accessory(accepted)

    # 6. Select held-out complete sessions before touching any raw hash_ids.
    revisit_floor = max(
        3,
        int(sorted(
            (lineage.turn_count for lineage in eval_lineages), reverse=True
        )[max(0, len(eval_lineages) // 4)] if eval_lineages else 3),
    )
    typical_sel = select_typical(eval_lineages, seed, nslice_typical)
    revisit_sel = select_revisit_heavy(
        eval_lineages, revisit_floor, seed, nslice_revisit
    )
    burst_sel = select_burst_long_context(
        eval_lineages, 0.9, 0.9, seed, nslice_burst
    )
    selected_lineages_by_id: Dict[int, Lineage] = {}
    for selection in (typical_sel, revisit_sel, burst_sel):
        for lineage in selection:
            selected_lineages_by_id[lineage.lineage_id] = lineage
    selected_pairs = {
        (mt.chat_id, mt.turn)
        for lineage in selected_lineages_by_id.values()
        for mt in lineage.turns
    }
    expected_block_counts = {
        (mt.chat_id, mt.turn): mt.block_count
        for lineage in selected_lineages_by_id.values()
        for mt in lineage.turns
    }

    # 7. Exactly one bounded second pass for the selected turns. The same
    # mapping feeds selected-prefix offline characterization and all slices.
    selected_hashes = _second_pass_hash_provider(
        file_path,
        selected_pairs,
        expected_block_counts=expected_block_counts,
    )

    def selected_hash_records():
        for lineage in selected_lineages_by_id.values():
            yield (
                lineage.lineage_id,
                [
                    (mt.turn, selected_hashes[(mt.chat_id, mt.turn)])
                    for mt in lineage.turns
                ],
            )

    global_reuse = characterize_global_hash_reuse(
        selected_hash_records(),
        total_records=prov.record_count,
        global_hash_use_summary=hash_summary,
        accurate_reuse_scope="selected_eval_slice_lineages",
    )

    # Translate the runtime-facing (lineage_id, turn) key to the raw record's
    # (chat_id, turn) key without retaining any unselected hashes.
    def make_hash_provider(selection: List[Lineage]):
        pair_to_raw_key = {
            (lineage.lineage_id, mt.turn): (mt.chat_id, mt.turn)
            for lineage in selection for mt in lineage.turns
        }

        def provider(lineage_id: int, turn: int):
            raw_key = pair_to_raw_key.get((lineage_id, turn))
            if raw_key is None:
                return None
            return selected_hashes.get(raw_key)

        return provider

    def assemble(slice_class: str, selection: List[Lineage], rule: str):
        manifest, stream_text = assemble_slice(
            slice_class=slice_class,
            slice_lineages=selection,
            trace_key=trace_key,
            file_sha256=prov.raw_byte_sha256,
            source_window_ts=(prov.first_timestamp or 0.0, prov.last_timestamp or 0.0),
            calibration_window=frozen_cal.split.calibration_window,
            evaluation_window=frozen_cal.split.evaluation_window,
            selection_rule=rule,
            seed=seed,
            time_dilation=1.0,
            frozen_cal=frozen_cal,
            hash_provider=make_hash_provider(selection),
            return_event_stream=True,
        )
        fname = {
            SLICE_TYPICAL: "events_typical.jsonl",
            SLICE_REVISIT_HEAVY: "events_revisit-heavy.jsonl",
            SLICE_BURST_LONG_CTX: "events_burst-long-context.jsonl",
        }[slice_class]
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
            f.write(stream_text)
        return manifest

    m_typ = assemble(
        SLICE_TYPICAL,
        typical_sel,
        "deterministic whole-session sample from eval prefix",
    )
    m_rev = assemble(
        SLICE_REVISIT_HEAVY,
        revisit_sel,
        f"eval sessions with turn_count>={revisit_floor}",
    )
    m_brs = assemble(
        SLICE_BURST_LONG_CTX,
        burst_sel,
        "eval sessions in top burst bin AND top long-ctx quantile",
    )
    manifests = {
        SLICE_TYPICAL: m_typ,
        SLICE_REVISIT_HEAVY: m_rev,
        SLICE_BURST_LONG_CTX: m_brs,
    }

    # 8-9. Persist only after assemble_slice's in-memory contract gate passes.
    _write_json(
        os.path.join(out_dir, "manifest_typical.json"),
        json.loads(m_typ.to_canonical_json()),
    )
    _write_json(
        os.path.join(out_dir, "manifest_revisit-heavy.json"),
        json.loads(m_rev.to_canonical_json()),
    )
    _write_json(
        os.path.join(out_dir, "manifest_burst-long-context.json"),
        json.loads(m_brs.to_canonical_json()),
    )
    _write_json(os.path.join(out_dir, "calibration.json"), frozen_cal.to_manifest_dict())
    _write_json(
        os.path.join(out_dir, "qualification_contract.json"),
        BudgetQualificationContract().to_dict(),
    )

    report = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "trace_key": trace_key,
        "trace_repo_head": TRACE_REPO_HEAD,
        "trace_file": TRACE_FILES[trace_key],
        "trace_file_sha256": prov.raw_byte_sha256,
        "record_count": prov.record_count,
        "line_fingerprint_sha256": prov.line_fingerprint_sha256,
        "ts_span": [prov.first_timestamp, prov.last_timestamp],
        "runtime_event_scope": RUNTIME_EVENT_SCOPE,
        "offline_ttl_replay_scope": OFFLINE_TTL_REPLAY_SCOPE,
        "assembly_stats": {
            "total_records": stats.total_records,
            "roots": stats.roots,
            "non_root": stats.non_root,
            "lineages_assembled": stats.lineages_assembled,
            "lineages_rejected": stats.lineages_rejected,
            "broken_continuity": stats.broken_continuity,
            "non_monotonic_timestamp": stats.non_monotonic_timestamp,
            "duplicate_turn": stats.duplicate_turn,
        },
        "split": {
            "calibration_until_ts": split.calibration_until_ts,
            "calibration_session_count": split.calibration_session_count,
            "evaluation_session_count": split.evaluation_session_count,
            "calibration_turn_count": split.calibration_turn_count,
            "evaluation_turn_count": split.evaluation_turn_count,
        },
        "frozen_calibration": frozen_cal.to_manifest_dict(),
        "cache_characterization": asdict_cache(cache_char),
        "offline_global_hash_reuse": asdict_global(global_reuse),
        "accessory_characterization": acc,
        "rejected_lineage_count": len(rejected),
        "manifests": {
            key: {
                "session_count": manifest.session_count,
                "turn_count": manifest.turn_count,
                "event_stream_sha256": manifest.event_stream_sha256,
                "event_count": manifest.event_count,
                "source_window_ts": manifest.source_window_ts,
                "selection_rule": manifest.selection_rule,
                "seed": manifest.seed,
                "time_dilation": manifest.time_dilation,
                "prefix_mode": manifest.prefix_mode,
                "materialize_mode": manifest.materialize_mode,
                "runtime_event_scope": manifest.runtime_event_scope,
                "ttl_replay_scope": manifest.ttl_replay_scope,
                "max_context_footprint_tokens": manifest.max_context_footprint_tokens,
                "max_required_context_tokens": manifest.max_required_context_tokens,
            }
            for key, manifest in manifests.items()
        },
        "slices_emitted": list(manifests.keys()),
    }
    _write_json(os.path.join(out_dir, "report.json"), report)
    return report


def asdict_cache(cc):
    return {
        "semantics": cc.semantics,
        "replayed_ttl_seconds": cc.replayed_ttl_seconds,
        "replayed_turns": cc.replayed_turns,
        "revisit_events": cc.revisit_events,
        "dead_events": cc.dead_events,
        "cold_restart_events": cc.cold_restart_events,
        "intra_session_reuse_rate": cc.intra_session_reuse_rate,
        "context_footprint_tokens": cc.context_footprint_tokens,
        "max_context_footprint_tokens": cc.max_context_footprint_tokens,
        "max_lineage_turn_count": cc.max_lineage_turn_count,
        "max_lineage_lifespan_seconds": cc.max_lineage_lifespan_seconds,
    }


def asdict_global(g):
    return dataclasses.asdict(g)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="trace_compiler.cli")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compile", help="compile a trace into frozen manifests")
    c.add_argument("--trace", required=True, choices=list(TRACE_FILES))
    c.add_argument("--out", default="/root/oscomp/kv_logs/gt-trace-1a")
    c.add_argument("--cal-fraction", type=float, default=0.5)
    c.add_argument("--target-fraction", type=float, default=0.9)
    c.add_argument("--nslice-typical", type=int, default=100)
    c.add_argument("--nslice-revisit", type=int, default=100)
    c.add_argument("--nslice-burst", type=int, default=50)
    c.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    report = compile_trace(
        trace_key=args.trace,
        out_dir=args.out,
        cal_fraction=args.cal_fraction,
        target_fraction=args.target_fraction,
        nslice_typical=args.nslice_typical,
        nslice_revisit=args.nslice_revisit,
        nslice_burst=args.nslice_burst,
        seed=args.seed,
    )
    print(json.dumps({
        "trace_key": report["trace_key"],
        "record_count": report["record_count"],
        "manifests": report["manifests"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
