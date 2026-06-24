#!/usr/bin/env python3
"""Stage 4C-1: offline cold/hot policy simulator for paged KV block traces.

Consumes a trace produced by ``LLAMA_KV_PAGED_TRACE=1`` (one ``KV_PAGED_TRACE``
line per decode step) and replays several block-eviction policies against it,
reporting the theoretical RSS benefit and swap I/O cost of each.

IMPORTANT: the simulator keys every policy off ``active_read_blocks`` (the real,
populated KV range), never ``read_blocks`` (the padded graph row_idx coverage,
which includes reserve/padding blocks and would inflate the working set).

This is a pure offline analysis tool. It does not touch the C++ engine, does not
change inference, and its approximate policies (``sink_recent_approx``) do NOT
guarantee output / sha256 equivalence -- they only estimate an RSS upper bound.
"""

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# trace parsing
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(r"(\w+)=(\S+)")


def _parse_blocks(value):
    """Parse a CSV block list field. '-' (or empty) means no blocks."""
    if value in ("-", ""):
        return []
    return [int(x) for x in value.split(",") if x != ""]


@dataclass
class Step:
    step: int
    n_kv: int
    active_n_kv: int
    read_blocks: list           # padded row_idx coverage (NOT used for policy)
    active_read_blocks: list    # real populated read set (used for policy)
    write_blocks: list
    blocks_in_use: int
    resident_blocks: int
    swapped_blocks: int
    released_blocks: int
    free_blocks: int

    @property
    def touched(self):
        """Blocks the engine actually reads or writes this step (the live set)."""
        return set(self.active_read_blocks) | set(self.write_blocks)


def parse_trace(path):
    steps = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if "KV_PAGED_TRACE" not in line:
                continue
            fields = dict(_LINE_RE.findall(line))
            if "step" not in fields:
                continue
            steps.append(Step(
                step=int(fields["step"]),
                n_kv=int(fields.get("n_kv", 0)),
                active_n_kv=int(fields.get("active_n_kv", 0)),
                read_blocks=_parse_blocks(fields.get("read_blocks", "-")),
                active_read_blocks=_parse_blocks(fields.get("active_read_blocks", "-")),
                write_blocks=_parse_blocks(fields.get("write_block", "-")),
                blocks_in_use=int(fields.get("blocks_in_use", 0)),
                resident_blocks=int(fields.get("resident_blocks", 0)),
                swapped_blocks=int(fields.get("swapped_blocks", 0)),
                released_blocks=int(fields.get("released_blocks", 0)),
                free_blocks=int(fields.get("free_blocks", 0)),
            ))
    steps.sort(key=lambda s: s.step)
    return steps


# ---------------------------------------------------------------------------
# simulation result accounting
# ---------------------------------------------------------------------------

@dataclass
class Result:
    policy: str
    resident_series: list = field(default_factory=list)  # resident block count per step
    swap_out_count: int = 0
    swap_in_count: int = 0
    # thrash: a block swapped in this step that was swapped out within the last
    #   K steps (it should not have been evicted). Measured at K=1 and K=4.
    thrash_in_1_step: int = 0
    thrash_in_4_steps: int = 0
    notes: str = ""


def summarize(res, block_bytes, n_tokens, baseline_avg_mb=None, baseline_peak_mb=None):
    series = res.resident_series or [0]
    avg = sum(series) / len(series)
    peak = max(series)
    low = min(series)
    mb = block_bytes / (1024 * 1024)
    avg_mb = avg * mb
    peak_mb = peak * mb
    swap_bytes = (res.swap_out_count + res.swap_in_count) * block_bytes
    swap_gib = swap_bytes / (1024 ** 3)
    swap_per_tok_mib = (swap_bytes / (1024 * 1024) / n_tokens) if n_tokens else 0.0
    thrash_ratio = (res.thrash_in_4_steps / res.swap_in_count) if res.swap_in_count else 0.0
    row = {
        "policy": res.policy,
        "avg_resident_blocks": round(avg, 2),
        "peak_resident_blocks": peak,
        "min_resident_blocks": low,
        "estimated_resident_mb_avg": round(avg_mb, 1),
        "estimated_resident_mb_peak": round(peak_mb, 1),
        "estimated_saved_mb_avg_vs_no_swap":
            round(baseline_avg_mb - avg_mb, 1) if baseline_avg_mb is not None else 0.0,
        "estimated_saved_mb_peak_vs_no_swap":
            round(baseline_peak_mb - peak_mb, 1) if baseline_peak_mb is not None else 0.0,
        "swap_out_count": res.swap_out_count,
        "swap_in_count": res.swap_in_count,
        "swap_io_gib": round(swap_gib, 3),
        "swap_io_per_token_mib": round(swap_per_tok_mib, 2),
        "thrash_in_1_step": res.thrash_in_1_step,
        "thrash_in_4_steps": res.thrash_in_4_steps,
        "thrash_ratio": round(thrash_ratio, 3),
        "notes": res.notes,
    }
    return row


# ---------------------------------------------------------------------------
# policies
#
# Common model: a block is "allocated" once it has ever been touched. A policy
# decides, per step, which allocated blocks are resident vs swapped-out. swap_out
# = resident->swapped transition; swap_in = swapped->resident transition. We
# count thrash when a block we swapped in was swapped out <= K steps earlier.
# ---------------------------------------------------------------------------

def _account_swaps(res, prev_swapped, new_swapped, last_swap_out_step, step_idx):
    """Count swap events from the swapped-set transition.

    A block entering `swapped` is a swap-out; a block leaving it is a swap-in.
    This deliberately ignores first-allocation (resident appearing for the first
    time): a brand-new block was never on disk, so writing it is not a swap-in.
    Thrash = a swap-in of a block that was swapped out <= K steps ago.
    """
    for b in new_swapped - prev_swapped:      # resident -> swapped
        res.swap_out_count += 1
        last_swap_out_step[b] = step_idx
    for b in prev_swapped - new_swapped:      # swapped -> resident
        res.swap_in_count += 1
        if b in last_swap_out_step:
            gap = step_idx - last_swap_out_step[b]
            if gap <= 1:
                res.thrash_in_1_step += 1
            if gap <= 4:
                res.thrash_in_4_steps += 1


def policy_no_swap(steps, **_):
    """All blocks ever touched stay resident forever. No swap at all."""
    res = Result("no_swap")
    resident = set()
    for s in steps:
        resident |= s.touched
        res.resident_series.append(len(resident))
    res.notes = "all live blocks resident; baseline RSS"
    return res


def policy_stage4a_release_only(steps, **_):
    """Only free/unused blocks are reclaimed; every live block stays resident.

    Under exact full attention this is the safe floor: resident == blocks_in_use.
    No live history is ever swapped, so swap_out=swap_in=0.
    """
    res = Result("stage4a_release_only")
    for s in steps:
        # live resident set == blocks currently in use (free blocks already released)
        res.resident_series.append(s.blocks_in_use)
    res.notes = "resident==blocks_in_use; reclaims only free blocks; zero swap (exact-safe)"
    return res


def policy_lru_idle_threshold(steps, idle_threshold, **_):
    """Swap out a block after it has been idle (untouched) for > idle_threshold
    steps; swap it back in the step it is touched again."""
    res = Result("lru_idle_threshold")
    resident = set()
    swapped = set()
    last_touch = {}
    last_swap_out_step = {}
    for i, s in enumerate(steps):
        touched = s.touched
        prev_swapped = set(swapped)
        # swap-in anything touched that is currently swapped
        for b in touched:
            if b in swapped:
                swapped.discard(b)
                resident.add(b)
        resident |= touched
        for b in touched:
            last_touch[b] = i
        # evict blocks idle beyond threshold
        for b in list(resident):
            if b in touched:
                continue
            if i - last_touch.get(b, i) > idle_threshold:
                resident.discard(b)
                swapped.add(b)
        _account_swaps(res, prev_swapped, swapped, last_swap_out_step, i)
        res.resident_series.append(len(resident))
    res.notes = f"evict after idle>{idle_threshold} steps; swap-in on touch"
    return res


def policy_anti_thrash_cooldown(steps, idle_threshold, cooldown, **_):
    """idle-threshold eviction, but a block swapped in is pinned for `cooldown`
    steps before it may be evicted again -- damps swap-in/swap-out oscillation."""
    res = Result("anti_thrash_cooldown")
    resident = set()
    swapped = set()
    last_touch = {}
    last_swap_in_step = {}
    last_swap_out_step = {}
    for i, s in enumerate(steps):
        touched = s.touched
        prev_swapped = set(swapped)
        for b in touched:
            if b in swapped:
                swapped.discard(b)
                resident.add(b)
                last_swap_in_step[b] = i
        resident |= touched
        for b in touched:
            last_touch[b] = i
        for b in list(resident):
            if b in touched:
                continue
            if i - last_swap_in_step.get(b, -10**9) < cooldown:
                continue  # cooled-down: pinned resident
            if i - last_touch.get(b, i) > idle_threshold:
                resident.discard(b)
                swapped.add(b)
        _account_swaps(res, prev_swapped, swapped, last_swap_out_step, i)
        res.resident_series.append(len(resident))
    res.notes = f"idle>{idle_threshold} evict + {cooldown}-step swap-in cooldown"
    return res


def policy_sink_recent_approx(steps, sink_blocks, recent_blocks, **_):
    """APPROXIMATE: keep first `sink_blocks` + most-recent `recent_blocks`
    resident; treat the middle history as cold (swapped). Does NOT preserve exact
    full attention -- middle blocks are dropped from the read set -- so it only
    estimates an RSS upper-bound benefit, not a correctness-preserving run."""
    res = Result("sink_recent_approx")
    resident = set()
    swapped = set()
    allocated = set()  # cumulative real block ids ever touched (handles non-contiguous ids)
    last_swap_out_step = {}
    for i, s in enumerate(steps):
        allocated |= s.touched
        if not allocated:
            res.resident_series.append(0)
            continue
        ordered = sorted(allocated)
        sink = set(ordered[:sink_blocks])                 # oldest (attention sink)
        recent = set(ordered[-recent_blocks:]) if recent_blocks > 0 else set()
        # NOTE: deliberately does NOT keep active_read_blocks resident -- that is the
        # whole point of the approximation: under full attention the middle history is
        # read every step, but this policy drops it anyway (so it is NOT sha256-exact).
        # Only the write block must be resident (you cannot write to a swapped block).
        hot = sink | recent | set(s.write_blocks)
        prev_swapped = set(swapped)
        resident = set(b for b in allocated if b in hot)
        swapped = allocated - resident
        _account_swaps(res, prev_swapped, swapped, last_swap_out_step, i)
        res.resident_series.append(len(resident))
    res.notes = (f"APPROX (not sha256-exact): keep sink={sink_blocks}+recent={recent_blocks}; "
                 "middle history dropped")
    return res


def policy_belady_oracle(steps, resident_budget, **_):
    """Belady's optimal (MIN): with a fixed resident budget, when a block must be
    evicted, evict the one whose next use is farthest in the future. Uses the full
    future trace as the oracle. Live correctness still requires swap-in on demand."""
    res = Result("belady_oracle")

    # Precompute, for each step, the per-block list of future touch step-indices.
    n = len(steps)
    # next_use[i][b] = smallest j >= i where block b is touched, or +inf
    touched_at = [s.touched for s in steps]
    # Build future occurrence lists lazily via a reverse scan.
    next_use_after = [dict() for _ in range(n + 1)]  # next_use_after[i][b] = next j>=i
    for i in range(n - 1, -1, -1):
        nxt = dict(next_use_after[i + 1]) if i + 1 <= n else {}
        for b in touched_at[i]:
            nxt[b] = i
        next_use_after[i] = nxt

    resident = set()
    swapped = set()
    last_swap_out_step = {}
    INF = 10 ** 12
    for i, s in enumerate(steps):
        touched = touched_at[i]
        prev_swapped = set(swapped)
        # demand swap-in
        for b in touched:
            if b in swapped:
                swapped.discard(b)
                resident.add(b)
        resident |= touched
        # if over budget, evict farthest-next-use among non-touched resident blocks
        if len(resident) > resident_budget:
            future = next_use_after[i + 1] if i + 1 < len(next_use_after) else {}
            candidates = [b for b in resident if b not in touched]
            # sort by next use descending (farthest first); never-again = INF
            candidates.sort(key=lambda b: future.get(b, INF), reverse=True)
            n_evict = len(resident) - resident_budget
            for b in candidates[:n_evict]:
                resident.discard(b)
                swapped.add(b)
        _account_swaps(res, prev_swapped, swapped, last_swap_out_step, i)
        res.resident_series.append(len(resident))
    res.notes = f"optimal MIN eviction at budget={resident_budget} blocks (oracle future)"
    return res


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

COLUMNS = [
    "policy",
    "avg_resident_blocks", "peak_resident_blocks", "min_resident_blocks",
    "estimated_resident_mb_avg", "estimated_resident_mb_peak",
    "estimated_saved_mb_avg_vs_no_swap", "estimated_saved_mb_peak_vs_no_swap",
    "swap_out_count", "swap_in_count", "swap_io_gib", "swap_io_per_token_mib",
    "thrash_in_1_step", "thrash_in_4_steps", "thrash_ratio", "notes",
]


def print_markdown(rows):
    print("| " + " | ".join(COLUMNS) + " |")
    print("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in COLUMNS) + " |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True, help="path to KV_PAGED_TRACE trace file")
    ap.add_argument("--block-bytes", type=int, default=4194304,
                    help="bytes per KV block (default 4 MiB)")
    ap.add_argument("--block-size", type=int, default=16,
                    help="cells per block (informational; default 16)")
    ap.add_argument("--recent-blocks", type=int, default=4,
                    help="recent window for sink_recent_approx")
    ap.add_argument("--sink-blocks", type=int, default=2,
                    help="sink (attention-sink) blocks kept resident")
    ap.add_argument("--idle-threshold", type=int, default=8,
                    help="idle steps before eviction (lru/anti-thrash)")
    ap.add_argument("--cooldown", type=int, default=8,
                    help="post-swap-in pin steps for anti_thrash_cooldown")
    ap.add_argument("--belady-budget", type=int, default=0,
                    help="resident block budget for belady_oracle (0 = auto: half of peak)")
    ap.add_argument("--csv", default=None, help="path to write CSV summary")
    args = ap.parse_args(argv)

    steps = parse_trace(args.trace)
    if not steps:
        print(f"error: no KV_PAGED_TRACE lines found in {args.trace}", file=sys.stderr)
        return 1

    n_tokens = len(steps)
    peak_in_use = max(s.blocks_in_use for s in steps)
    belady_budget = args.belady_budget or max(1, peak_in_use // 2)

    kwargs = dict(
        idle_threshold=args.idle_threshold,
        cooldown=args.cooldown,
        sink_blocks=args.sink_blocks,
        recent_blocks=args.recent_blocks,
        resident_budget=belady_budget,
    )

    policies = [
        policy_no_swap,
        policy_stage4a_release_only,
        policy_lru_idle_threshold,
        policy_anti_thrash_cooldown,
        policy_sink_recent_approx,
        policy_belady_oracle,
    ]

    results = [p(steps, **kwargs) for p in policies]

    mb = args.block_bytes / (1024 * 1024)
    base = results[0]
    base_avg_mb = (sum(base.resident_series) / len(base.resident_series)) * mb
    base_peak_mb = max(base.resident_series) * mb

    rows = [summarize(r, args.block_bytes, n_tokens, base_avg_mb, base_peak_mb)
            for r in results]

    # console
    print(f"# KV paged cold/hot policy simulation")
    print(f"trace={args.trace} steps={n_tokens} block_bytes={args.block_bytes} "
          f"peak_blocks_in_use={peak_in_use} belady_budget={belady_budget}\n")
    print_markdown(rows)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
