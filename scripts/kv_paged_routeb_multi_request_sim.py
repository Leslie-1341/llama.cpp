#!/usr/bin/env python3
"""Stage 4C-2B: multi-request / idle-request KV swap simulator (Route B, exact).

Stage 4C-1 showed that a *single* request under exact full attention reads
essentially every live KV block every decode step, so there is no cold block to
swap without breaking correctness. Stage 4C-2A (sink+recent) only saves RSS by
*approximating* attention, which is not sha256-exact.

This simulator explores the exact-correct opportunity instead: with *multiple*
concurrent requests, a request that goes idle stops touching its KV entirely.
Those blocks are genuinely cold and can be swapped out and faithfully swapped
back in on resume -- no attention is dropped, so output / sha256 is unchanged.

The workload is synthetic: each request's KV growth reuses the measured growth
pattern from the single-request trace (~1 block per 16 active steps), and a
global step timeline marks each request active / idle / finished. Policies
decide which non-finished requests' blocks stay resident.

Pure offline analysis. Does not touch the C++ engine, does not run llama.
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# KV growth template (derived from the single-request trace)
#
# The Stage 4C trace (ctx=4096, n=512) grows blocks_in_use linearly: it starts
# at 1 block and gains 1 block every ~16 active decode steps, peaking at 33.
# We reuse that as a per-request template: a request that has been active for
# `a` steps holds blocks_for_active_steps(a) KV blocks.
# ---------------------------------------------------------------------------

STEPS_PER_BLOCK = 16  # measured: +1 block every 16 active steps


def blocks_for_active_steps(active_steps, prompt_blocks=1, cap=None):
    """KV blocks held by a request after `active_steps` decode steps.

    prompt_blocks models a request that arrives with a non-trivial prompt
    already resident (long-context requests start large)."""
    grown = prompt_blocks + active_steps // STEPS_PER_BLOCK
    if cap is not None:
        grown = min(grown, cap)
    return grown


# ---------------------------------------------------------------------------
# workload: a timeline of per-request states
# ---------------------------------------------------------------------------

ACTIVE = "active"
IDLE = "idle"
FINISHED = "finished"
NOT_ARRIVED = "not_arrived"


@dataclass
class RequestSpec:
    name: str
    arrival: int          # global step the request first becomes active
    # spans: list of (state, duration) starting at arrival; state in {active, idle}
    spans: list
    prompt_blocks: int = 1
    cap: int = None       # max blocks (long-context requests cap higher)

    def timeline(self, n_steps):
        """Return per-global-step (state, active_steps_so_far) for this request."""
        out = [(NOT_ARRIVED, 0)] * n_steps
        active_accum = 0
        g = self.arrival
        for state, dur in self.spans:
            for _ in range(dur):
                if g >= n_steps:
                    break
                if state == ACTIVE:
                    active_accum += 1
                out[g] = (state, active_accum)
                g += 1
        # after the last span the request is finished
        for k in range(g, n_steps):
            out[k] = (FINISHED, active_accum)
        return out


def build_workload(n_steps):
    """Synthetic multi-tenant workload (A long+idle+resume, B fills A's idle,
    C short intermittent, D long-context low-frequency resume).

    Footprints are sized like a small multi-tenant server: prompts dominate the
    initial KV (long-context requests arrive large), so the concurrent live set
    can exceed the 64..192-block budget range and the budget policies actually
    bind. The idle windows are staggered so different requests are cold at
    different global steps -- that overlap is what makes exact swap pay off."""
    reqs = [
        # A: long chat request -- decodes, goes idle (user thinking), resumes.
        RequestSpec("A", arrival=0, prompt_blocks=24, cap=80, spans=[
            (ACTIVE, 160),   # warm prompt + decode
            (IDLE, 200),     # idle while the user reads / a tool runs
            (ACTIVE, 152),   # resumes and keeps decoding
        ]),
        # B: arrives mid-stream (while A is still warm), runs across A's idle gap.
        RequestSpec("B", arrival=120, prompt_blocks=20, cap=64, spans=[
            (ACTIVE, 240),   # decodes across A's idle window
            (IDLE, 80),
            (ACTIVE, 40),
        ]),
        # C: short, intermittent request -- two bursts then finishes early.
        RequestSpec("C", arrival=60, prompt_blocks=6, cap=16, spans=[
            (ACTIVE, 48),    # short burst
            (IDLE, 24),
            (ACTIVE, 48),    # second short burst, then finishes
        ]),
        # D: long-context request -- arrives with a huge prompt, mostly idle,
        #    resumes infrequently. The prime exact-swap candidate.
        RequestSpec("D", arrival=20, prompt_blocks=80, cap=120, spans=[
            (ACTIVE, 64),    # arrives with a big prompt (80 blocks), grows
            (IDLE, 300),     # long idle -- huge cold footprint
            (ACTIVE, 64),    # infrequent resume
        ]),
    ]
    return reqs


# ---------------------------------------------------------------------------
# per-step world state
# ---------------------------------------------------------------------------

@dataclass
class StepView:
    """Resolved state of every request at one global step."""
    step: int
    states: dict          # name -> state
    blocks: dict          # name -> blocks_in_use this step (0 if not_arrived/finished)
    last_active: dict      # name -> last global step the request was active (or -1)


def resolve_timeline(reqs, n_steps):
    tls = {r.name: r.timeline(n_steps) for r in reqs}
    spec = {r.name: r for r in reqs}
    last_active = {r.name: -1 for r in reqs}
    views = []
    for g in range(n_steps):
        states, blocks = {}, {}
        for r in reqs:
            state, accum = tls[r.name][g]
            states[r.name] = state
            if state in (ACTIVE, IDLE):
                blocks[r.name] = blocks_for_active_steps(
                    accum, prompt_blocks=spec[r.name].prompt_blocks, cap=spec[r.name].cap)
            else:  # not_arrived / finished hold no live KV
                blocks[r.name] = 0
            if state == ACTIVE:
                last_active[r.name] = g
        views.append(StepView(g, states, blocks, dict(last_active)))
    return views


# ---------------------------------------------------------------------------
# result accounting
# ---------------------------------------------------------------------------

@dataclass
class Result:
    policy: str
    params: str = ""
    resident_series: list = field(default_factory=list)  # resident blocks per step
    request_swap_out_count: int = 0   # request-level swap-out events
    request_swap_in_count: int = 0
    blocks_swapped_out: int = 0       # block-granular I/O
    blocks_swapped_in: int = 0
    resume_events: int = 0            # idle/not->active transitions (excl. first arrival? see below)
    resume_with_swap_in: int = 0      # resumes that required swapping blocks back
    resume_swap_in_blocks: list = field(default_factory=list)  # blocks paged in per resume-with-swap
    # thrash: request swapped in within K steps of being swapped out
    thrash_in_1_step: int = 0
    thrash_in_4_steps: int = 0
    notes: str = ""


def summarize(res, block_bytes, baseline_avg_mb, baseline_peak_mb):
    series = res.resident_series or [0]
    avg = sum(series) / len(series)
    peak = max(series)
    mb = block_bytes / (1024 * 1024)
    avg_mb = avg * mb
    peak_mb = peak * mb
    swap_blocks = res.blocks_swapped_out + res.blocks_swapped_in
    swap_gib = (swap_blocks * block_bytes) / (1024 ** 3)
    resume_mb = [b * mb for b in res.resume_swap_in_blocks]
    avg_resume_mb = (sum(resume_mb) / len(resume_mb)) if resume_mb else 0.0
    max_resume_mb = max(resume_mb) if resume_mb else 0.0
    thrash_ratio = (res.thrash_in_4_steps / res.request_swap_in_count
                    if res.request_swap_in_count else 0.0)
    return {
        "policy": res.policy,
        "parameters": res.params,
        "avg_resident_blocks": round(avg, 2),
        "peak_resident_blocks": peak,
        "estimated_resident_mb_avg": round(avg_mb, 1),
        "estimated_resident_mb_peak": round(peak_mb, 1),
        "saved_mb_avg_vs_all_resident": round(baseline_avg_mb - avg_mb, 1),
        "saved_mb_peak_vs_all_resident": round(baseline_peak_mb - peak_mb, 1),
        "request_swap_out_count": res.request_swap_out_count,
        "request_swap_in_count": res.request_swap_in_count,
        "blocks_swapped_out": res.blocks_swapped_out,
        "blocks_swapped_in": res.blocks_swapped_in,
        "swap_io_gib": round(swap_gib, 3),
        "resume_events": res.resume_events,
        "resume_with_swap_in": res.resume_with_swap_in,
        "avg_resume_swap_in_mb": round(avg_resume_mb, 1),
        "max_resume_swap_in_mb": round(max_resume_mb, 1),
        "thrash_in_1_step": res.thrash_in_1_step,
        "thrash_in_4_steps": res.thrash_in_4_steps,
        "thrash_ratio": round(thrash_ratio, 3),
        "notes": res.notes,
    }


# ---------------------------------------------------------------------------
# simulation core
#
# State per request: resident (bool) -- whether its KV blocks are in RAM.
# A request that is ACTIVE this step MUST be resident (it is reading/writing its
# own KV). A request that is IDLE may be swapped out by policy. On the step a
# request transitions idle->active (resume), any swapped-out blocks are paged in
# (the resume cost). All swaps here are EXACT: an idle request reads nothing, so
# evicting/restoring its blocks does not change any attention output.
# ---------------------------------------------------------------------------

def _run(views, n_steps, decide_swapped_out, policy_name, params, notes):
    """Generic driver. `decide_swapped_out(view, resident_now)` returns the set of
    request names that should be SWAPPED OUT this step (must never include an
    ACTIVE request)."""
    res = Result(policy_name, params=params, notes=notes)
    swapped = set()                 # request names currently swapped out
    last_swap_out_step = {}         # name -> step it was last swapped out
    prev_states = {}                # name -> previous step's state

    for v in views:
        # 1) resume detection: idle/not_arrived -> active this step
        for name, st in v.states.items():
            was = prev_states.get(name, NOT_ARRIVED)
            if st == ACTIVE and was in (IDLE,):
                res.resume_events += 1
                if name in swapped:
                    # page the whole request back in (exact restore)
                    blocks = v.blocks[name]
                    swapped.discard(name)
                    res.request_swap_in_count += 1
                    res.blocks_swapped_in += blocks
                    res.resume_with_swap_in += 1
                    res.resume_swap_in_blocks.append(blocks)
                    gap = v.step - last_swap_out_step.get(name, -10**9)
                    if gap <= 1:
                        res.thrash_in_1_step += 1
                    if gap <= 4:
                        res.thrash_in_4_steps += 1

        # 2) finished requests free their KV entirely (no swap accounting)
        for name, st in v.states.items():
            if st in (FINISHED, NOT_ARRIVED):
                swapped.discard(name)

        # 3) policy decides which idle requests to swap out
        resident_now = {n for n, st in v.states.items()
                        if st in (ACTIVE, IDLE) and n not in swapped}
        to_swap = decide_swapped_out(v, resident_now)
        for name in to_swap:
            if v.states[name] != ACTIVE and name not in swapped and v.blocks[name] > 0:
                swapped.add(name)
                res.request_swap_out_count += 1
                res.blocks_swapped_out += v.blocks[name]
                last_swap_out_step[name] = v.step

        # 4) record resident block total
        resident_blocks = sum(v.blocks[n] for n, st in v.states.items()
                              if st in (ACTIVE, IDLE) and n not in swapped)
        res.resident_series.append(resident_blocks)

        prev_states = dict(v.states)
    return res


def _live_names(v):
    return [n for n, st in v.states.items() if st in (ACTIVE, IDLE)]


def policy_all_resident(views, n_steps, **_):
    return _run(views, n_steps,
                lambda v, r: set(),
                "all_resident", "-",
                "upper bound: every non-finished request stays resident")


def policy_active_only(views, n_steps, **_):
    def decide(v, resident_now):
        return {n for n in _live_names(v) if v.states[n] == IDLE}
    return _run(views, n_steps, decide,
                "active_only", "-",
                "swap out every idle request immediately; max savings, max resume cost")


def policy_idle_swap_T(views, n_steps, idle_T, **_):
    def decide(v, resident_now):
        out = set()
        for n in _live_names(v):
            if v.states[n] == IDLE:
                idle_for = v.step - v.last_active.get(n, v.step)
                if idle_for > idle_T:
                    out.add(n)
        return out
    return _run(views, n_steps, decide,
                "idle_swap_T", f"T={idle_T}",
                f"swap out a request after it has been idle > {idle_T} global steps")


def policy_lru_request_swap_budget(views, n_steps, budget, **_):
    def decide(v, resident_now):
        # current resident block total
        total = sum(v.blocks[n] for n in resident_now)
        if total <= budget:
            return set()
        # evict least-recently-active IDLE requests until under budget
        idle_resident = [n for n in resident_now if v.states[n] == IDLE]
        idle_resident.sort(key=lambda n: v.last_active.get(n, -1))  # oldest active first
        out = set()
        for n in idle_resident:
            if total <= budget:
                break
            out.add(n)
            total -= v.blocks[n]
        return out
    return _run(views, n_steps, decide,
                "lru_request_swap_budget", f"budget={budget}",
                f"keep active resident; over {budget} blocks evict LRU idle requests")


def policy_hybrid_idle_budget(views, n_steps, idle_T, budget, **_):
    def decide(v, resident_now):
        out = set()
        # 1) idle past threshold -> swap out first
        for n in _live_names(v):
            if v.states[n] == IDLE:
                idle_for = v.step - v.last_active.get(n, v.step)
                if idle_for > idle_T:
                    out.add(n)
        # 2) if still over budget, LRU-evict remaining idle requests
        total = sum(v.blocks[n] for n in resident_now if n not in out)
        if total > budget:
            idle_rest = [n for n in resident_now
                         if v.states[n] == IDLE and n not in out]
            idle_rest.sort(key=lambda n: v.last_active.get(n, -1))
            for n in idle_rest:
                if total <= budget:
                    break
                out.add(n)
                total -= v.blocks[n]
        return out
    return _run(views, n_steps, decide,
                "hybrid_idle_budget", f"T={idle_T},budget={budget}",
                f"swap idle>{idle_T}; if still over {budget} blocks, LRU-evict idle")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

COLUMNS = [
    "policy", "parameters",
    "avg_resident_blocks", "peak_resident_blocks",
    "estimated_resident_mb_avg", "estimated_resident_mb_peak",
    "saved_mb_avg_vs_all_resident", "saved_mb_peak_vs_all_resident",
    "request_swap_out_count", "request_swap_in_count",
    "blocks_swapped_out", "blocks_swapped_in", "swap_io_gib",
    "resume_events", "resume_with_swap_in",
    "avg_resume_swap_in_mb", "max_resume_swap_in_mb",
    "thrash_in_1_step", "thrash_in_4_steps", "thrash_ratio",
    "notes",
]


def print_markdown(rows):
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=512, help="global timeline length")
    ap.add_argument("--block-bytes", type=int, default=4194304,
                    help="bytes per KV block (default 4 MiB)")
    ap.add_argument("--block-size", type=int, default=16,
                    help="cells per block (informational; default 16)")
    ap.add_argument("--idle-thresholds", default="8,16,32,64",
                    help="comma list of T for idle_swap_T / hybrid")
    ap.add_argument("--budgets", default="64,96,128,192",
                    help="comma list of memory_budget_blocks for budget policies")
    ap.add_argument("--csv", default=None, help="path to write CSV summary")
    args = ap.parse_args(argv)

    n_steps = args.steps
    reqs = build_workload(n_steps)
    views = resolve_timeline(reqs, n_steps)

    Ts = [int(x) for x in args.idle_thresholds.split(",") if x.strip()]
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]

    results = []
    results.append(policy_all_resident(views, n_steps))
    results.append(policy_active_only(views, n_steps))
    for T in Ts:
        results.append(policy_idle_swap_T(views, n_steps, idle_T=T))
    for b in budgets:
        results.append(policy_lru_request_swap_budget(views, n_steps, budget=b))
    # hybrid: pair a mid threshold with each budget
    hybrid_T = Ts[len(Ts) // 2] if Ts else 16
    for b in budgets:
        results.append(policy_hybrid_idle_budget(views, n_steps, idle_T=hybrid_T, budget=b))

    mb = args.block_bytes / (1024 * 1024)
    base = results[0]  # all_resident
    base_avg_mb = (sum(base.resident_series) / len(base.resident_series)) * mb
    base_peak_mb = max(base.resident_series) * mb

    rows = [summarize(r, args.block_bytes, base_avg_mb, base_peak_mb) for r in results]

    peak_concurrent = max(
        sum(1 for st in v.states.values() if st in (ACTIVE, IDLE)) for v in views)
    print("# KV paged multi-request / idle-request swap simulation (Route B, exact)")
    print(f"steps={n_steps} requests={len(reqs)} peak_concurrent_live={peak_concurrent} "
          f"block_bytes={args.block_bytes}\n")
    print_markdown(rows)

    if args.csv:
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
