#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lineage reconstruction.

A session lineage = root record (parent_chat_id == -1, turn == 1) plus every
record reachable from the root by chaining turn_k.parent_chat_id ==
turn_{k-1}.chat_id downward. The root's chat_id is the lineage key.

Reconstruction is offline ground truth: it reads the WHOLE trace (a single
streaming pass) to assemble lineages, then validates per-lineage turn
CONTINUITY and timestamp MONOTONICITY. Any anomaly -> FAIL-CLOSED:
the offending lineage is routed to RejectedLineage and excluded from any
runtime-visible event stream.

Memory: O(total_records) for the chat_id -> record index, because lineage
linkage requires resolving a parent chain that may span the whole file
(verified: orphans reference parents that appear anywhere in the file).
We keep only the lightweight index (chat_id -> minimal turn record WITHOUT
hash_ids) plus the raw hash_ids consumed eagerly. hash_ids are never held
for more than the records currently being assembled into a lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .common import TraceRecord


class LineageError(Exception):
    """Raised for fatal, non-salvageable lineage anomalies."""


@dataclass(frozen=True)
class MinimalTurn:
    """The lightweight per-turnstile metadata we retain for index/assembly.

    hash_ids are deliberately omitted here: when a lineage is assembled we
    re-walk the chat_id chain and pull hash_ids only for the lineage's own
    turns, dropping everything else. This keeps steady-state memory near
    O(lineage-footprint) rather than O(total * blocks).
    """

    chat_id: int
    parent_chat_id: int
    timestamp: float
    turn: int
    input_length: int
    output_length: int
    type: str
    block_count: int


@dataclass
class Lineage:
    """A fully reconstructed session lineage.

    `turns` is ordered turn 1..N. lineage_id == root.chat_id.
    """

    lineage_id: int
    turns: List[MinimalTurn]
    # Rejected reason, or None if accepted. Accepted lineages are the only
    # ones eligible to appear in runtime event streams.
    rejected_reason: Optional[str] = None

    @property
    def is_accepted(self) -> bool:
        return self.rejected_reason is None

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def lifespan(self) -> float:
        if not self.turns:
            return 0.0
        return self.turns[-1].timestamp - self.turns[0].timestamp

    def timestamps(self) -> List[float]:
        return [t.timestamp for t in self.turns]


@dataclass
class LineageAssemblyStats:
    total_records: int = 0
    roots: int = 0
    non_root: int = 0
    accepted_records: int = 0  # records absorbed into an accepted lineage
    rejected_records: int = 0  # records routed into a rejected lineage
    orphan_records: int = 0  # explicitly rejected: parent chat_id absent
    broken_chain_records: int = 0  # explicitly rejected: turn[k].parent != turn[k-1].chat_id
    lineages_assembled: int = 0
    lineages_rejected: int = 0
    broken_continuity: int = 0  # accepted-lineage turn gap (defensive; rejected)
    non_monotonic_timestamp: int = 0  # rejected
    duplicate_turn: int = 0  # rejected

    @property
    def accounted_records(self) -> int:
        """Records we can account for via accepted or rejected lineages."""
        return self.accepted_records + self.rejected_records

    @property
    def explicitly_rejected_records(self) -> int:
        return self.rejected_records


# ---------------------------------------------------------------------------
# Two-pass streaming reconstruction.
# Pass 1: build chat_id -> MinimalTurn index (no hash_ids retained).
# Pass 2: resolve each record UPWARD by following parent_chat_id to its
#   predecessor, root (parent=-1,turn=1) at the top. A lineage is a STRICT
#   linear chain: turn[k].parent_chat_id == turn[k-1].chat_id for k=2..N.
#   Any deviation (orphan parent absent, broken link, wrong turn, gap,
#   duplicate, non-monotonic timestamp) -> FAIL-CLOSED: the whole offending
#   lineage (every record reachable from each offending record's chain) is
#   routed to a single RejectedLineage keyed by its root chat_id (or by a
#   synthetic id when no root exists), and accepted+rejected == total_records
#   is enforced. No record is silently dropped.
# ---------------------------------------------------------------------------
def build_child_index(
    minimal_by_chat: Dict[int, MinimalTurn],
) -> Dict[int, List[int]]:
    """Map parent_chat_id -> [child chat_id, ...] for non-root records.

    Kept for diagnostics; the strict resolver below does NOT use a downward
    BFS (which can mis-assemble a branching parent set into a fake linear
    chain). It walks UPWARD from each record instead.
    """
    idx: Dict[int, List[int]] = {}
    for mt in minimal_by_chat.values():
        if mt.parent_chat_id != -1:
            idx.setdefault(mt.parent_chat_id, []).append(mt.chat_id)
    return idx


def reconstruct_lineages(
    records: Iterable[TraceRecord],
) -> Tuple[List[Lineage], List[Lineage], LineageAssemblyStats]:
    """Stream a trace's records and return (accepted, rejected, stats).

    Acceptance rule (STRICT LINEAR CHAIN, fail-closed):
      A lineage is exactly { root (parent=-1, turn=1) } U { records whose
      upward parent chain terminates at that root }, AND every adjacent pair
      satisfies turn[k].parent_chat_id == turn[k-1].chat_id with turn k-1
      having turn == k-1, producing the exact sequence turn 1,2,...,N with
      strictly non-decreasing timestamps.

    Any anomaly -> the entire chain reachable from each offending record is
    rejected as one RejectedLineage. Records are partitioned: every record
    goes to exactly one accepted OR one rejected lineage. The caller MUST
    verify stats.accepted_records + stats.rejected_records ==
    stats.total_records; this function raises LineageError on violation
    (fail-closed rather than emitting a silently lossy manifest).
    """
    minimal_by_chat: Dict[int, MinimalTurn] = {}
    stats = LineageAssemblyStats()

    for rec in records:
        stats.total_records += 1
        if rec.is_root:
            stats.roots += 1
        else:
            stats.non_root += 1
        if rec.chat_id in minimal_by_chat:
            raise LineageError(
                f"duplicate chat_id {rec.chat_id}: trace records must be unique"
            )
        minimal_by_chat[rec.chat_id] = MinimalTurn(
            chat_id=rec.chat_id,
            parent_chat_id=rec.parent_chat_id,
            timestamp=rec.timestamp,
            turn=rec.turn,
            input_length=rec.input_length,
            output_length=rec.output_length,
            type=rec.type,
            block_count=rec.block_count,
        )
        # hash_ids dropped here for non-retained records; retained ones are
        # pulled later by the slice materializer, not here.

    accepted: List[Lineage] = []
    rejected: List[Lineage] = []
    # chat_id -> True once classified into an accepted OR rejected lineage.
    assigned: Dict[int, bool] = {}
    used_reject_keys: set = set()
    synth_seq: List[int] = [-1]  # synthetic reject keys (negative, unique)

    # child index: parent_chat_id -> [child chat_id, ...]. Built ONCE.
    children: Dict[int, List[int]] = {}
    for mt in minimal_by_chat.values():
        if mt.parent_chat_id != -1:
            children.setdefault(mt.parent_chat_id, []).append(mt.chat_id)

    def mint_reject_key(seed: int) -> int:
        if seed in minimal_by_chat and seed not in used_reject_keys:
            used_reject_keys.add(seed)
            return seed
        while synth_seq[0] in used_reject_keys or synth_seq[0] in minimal_by_chat:
            synth_seq[0] -= 1
        k = synth_seq[0]
        synth_seq[0] -= 1
        used_reject_keys.add(k)
        return k

    def reject_subtree(seed_chat: int, reason: str) -> None:
        """Reject seed_chat and everything below it (downward) that is still
        unassigned, as ONE RejectedLineage. The lineage_id is the seed chat_id
        when unique, else a synthetic key. Reads down via `children`.
        """
        subtree: List[int] = []
        seen: set = set()
        frontier = [seed_chat]
        while frontier:
            c = frontier.pop()
            if c in seen or c in assigned:
                continue
            if c not in minimal_by_chat:
                continue
            seen.add(c)
            subtree.append(c)
            for child in children.get(c, []):
                if child not in seen and child not in assigned:
                    frontier.append(child)
        subtree.sort(key=lambda x: (minimal_by_chat[x].turn, x))
        key = mint_reject_key(seed_chat)
        for c in subtree:
            assigned[c] = False
        stats.lineages_rejected += 1
        stats.rejected_records += len(subtree)
        if "orphan" in reason:
            stats.orphan_records += len(subtree)
        elif "broken chain" in reason:
            stats.broken_chain_records += len(subtree)
        elif "continuity" in reason:
            stats.broken_continuity += 1
        elif "non-monotonic timestamp" in reason:
            stats.non_monotonic_timestamp += 1
        elif "duplicate turn" in reason:
            stats.duplicate_turn += 1
        rejected.append(Lineage(
            lineage_id=key,
            turns=[minimal_by_chat[c] for c in subtree],
            rejected_reason=reason,
        ))

    def accept_chain(root_chat: int) -> None:
        """Walk strictly DOWNWARD from root. At each turn k, among the
        children of turn k there must be exactly ONE record whose
        parent_chat_id == cur.chat_id AND turn == cur.turn + 1. That is the
        sole linear continuation. Any OTHER child claiming this parent is a
        branch -> that branch's own subtree (the offending child + its
        descendants) is rejected fail-closed. If the valid continuation's
        timestamp is < its parent's timestamp, the entire remainder from
        that child down is rejected (non-monotonic). Continuity is the turn
        match, which the loop enforces step by step.

        Fail-closed on a DECLARED but malformed multi-turn lineage: when a
        root's claimed continuation(s) are rejected (any branch child,
        ambiguous continuation, or non-monotonic continuation) AND the root
        never reached a second turn, the single root record is NOT emitted
        as a phantom single-turn session that never existed in the trace.
        The whole declared lineage (root + rejected children) is routed to
        ONE RejectedLineage, so the manifest does not fabricate a session
        from a broken chain. A genuine single-turn root (no child claims it)
        is accepted unchanged.
        """
        cur_mt = minimal_by_chat[root_chat]
        chain_members: List[int] = [root_chat]
        assigned[root_chat] = True
        cur = root_chat
        root_first = root_chat
        # Anomaly bookkeeping: children that CLAIMED to belong to the chain
        # (parent == cur) but were rejected for breaking continuity,
        # ambiguous continuation, or monotonicity. We defer their rejection
        # so that if the root never grows past turn 1 we can MERGE them with
        # the root into ONE RejectedLineage rather than emitting a phantom
        # single-turn session AND a separate rejected child lineage.
        deferred_anomaly: List[int] = []
        deferred_reason: Optional[str] = None
        while True:
            kids = children.get(cur, [])
            # Partition: linear-continuation candidates vs everyone else.
            linear_kids = [
                k for k in kids
                if k not in assigned
                and minimal_by_chat[k].parent_chat_id == cur
                and minimal_by_chat[k].turn == cur_mt.turn + 1
            ]
            branch_kids = [
                k for k in kids
                if k not in assigned and k not in linear_kids
            ]
            # Any branch record claims this parent but is NOT the canonical
            # next turn -> malformed (branching parent set). Defer.
            for bk in branch_kids:
                if bk in assigned:
                    continue
                if deferred_reason is None:
                    deferred_reason = (
                        f"broken continuity: chat_id={bk} claims parent "
                        f"chat_id={cur} but is not the turn-{cur_mt.turn + 1} "
                        f"linear continuation"
                    )
                deferred_anomaly.append(bk)
            if not linear_kids:
                break
            if len(linear_kids) > 1:
                # Two records both claim to be turn+1 of cur -> ambiguous
                # chain; defer all, fail-closed.
                if deferred_reason is None:
                    deferred_reason = (
                        f"broken continuity: turn {cur_mt.turn + 1} has "
                        f"{len(linear_kids)} candidate children of "
                        f"chat_id={cur}"
                    )
                deferred_anomaly.extend(linear_kids)
                break
            child = linear_kids[0]
            child_mt = minimal_by_chat[child]
            # Timestamp strict monotonic non-decreasing.
            if child_mt.timestamp < cur_mt.timestamp:
                if deferred_reason is None:
                    deferred_reason = (
                        f"non-monotonic timestamp: chat_id={child} ts="
                        f"{child_mt.timestamp} < parent chat_id={cur} ts="
                        f"{cur_mt.timestamp}"
                    )
                deferred_anomaly.append(child)
                break
            # Flush prior deferred anomalies now that the chain has grown
            # past a single turn: the root is a genuine multi-turn session,
            # so the rejected siblings get their own RejectedLineage(s).
            for anom in deferred_anomaly:
                reject_subtree(
                    anom,
                    reason=(
                        f"branched/orphan: chat_id={anom} claims parent "
                        f"chat_id={root_first} but is not the canonical "
                        f"linear continuation (declared while lineage grew)"
                    ),
                )
            deferred_anomaly = []
            chain_members.append(child)
            assigned[child] = True
            cur_mt = child_mt
            cur = child
        # Commits the assembled linear chain. The walk above already
        # guarantees strict parent-link identity, exact turn continuity, and
        # non-decreasing timestamps; _validate_lineage is a defensive guard.
        turns = [minimal_by_chat[c] for c in chain_members]
        vreason = _validate_lineage(chain_members[0], turns)
        if vreason is not None:
            # Defensive fail-closed: reject the whole chain we just walked.
            for c in chain_members:
                assigned[c] = False
            key = mint_reject_key(chain_members[0])
            stats.lineages_rejected += 1
            stats.rejected_records += len(chain_members)
            rejected.append(Lineage(
                lineage_id=key, turns=turns, rejected_reason=vreason,
            ))
            return
        # Fail-closed merge: a root whose claimed continuation(s) were all
        # rejected and which never grew past a single turn did not exist as
        # a single-turn session in the trace. Roll the root AND its claimed
        # children's subtrees into ONE RejectedLineage so the manifest does
        # not fabricate a session from a broken chain and does not split the
        # declared lineage across two rejected records. A genuine single-turn
        # root (no child claims it -> no deferred anomaly) is accepted as-is.
        if deferred_anomaly and len(chain_members) == 1:
            for c in chain_members:
                assigned[c] = False
            # Collect the root plus every deferred child's full subtree
            # (reject_subtree expands downward via `children`).
            merge_ids: List[int] = [root_chat]
            seen_merge: set = {root_chat}
            frontier = list(deferred_anomaly)
            while frontier:
                c = frontier.pop()
                if c in seen_merge or c in assigned:
                    continue
                if c not in minimal_by_chat:
                    continue
                seen_merge.add(c)
                merge_ids.append(c)
                for ch in children.get(c, []):
                    if ch not in seen_merge and ch not in assigned:
                        frontier.append(ch)
            merge_ids.sort(key=lambda x: (minimal_by_chat[x].turn, x))
            for c in merge_ids:
                assigned[c] = False
            merge_turns = [minimal_by_chat[c] for c in merge_ids]
            key = mint_reject_key(root_chat)
            stats.lineages_rejected += 1
            stats.rejected_records += len(merge_ids)
            rejected.append(Lineage(
                lineage_id=key,
                turns=merge_turns,
                rejected_reason=deferred_reason or "broken continuity",
            ))
            return
        stats.lineages_assembled += 1
        stats.accepted_records += len(chain_members)
        accepted.append(Lineage(lineage_id=chain_members[0], turns=turns))

    # Pass 1: process every ROOT.
    for chat_id, mt in minimal_by_chat.items():
        if chat_id in assigned:
            continue
        if mt.parent_chat_id != -1:
            continue  # non-root handled in pass 2
        if mt.turn != 1:
            # Declared root but turn != 1 -> invalid; reject its subtree.
            reject_subtree(
                chat_id,
                reason=f"root chat_id={chat_id} has turn={mt.turn} != 1",
            )
            continue
        accept_chain(chat_id)

    # Pass 2: any still-unassigned record is an ORPHAN (its parent either is
    # absent from the trace, or was rejected, or sits in a rejected branch).
    for chat_id, mt in minimal_by_chat.items():
        if chat_id in assigned:
            continue
        # Determine the reason: parent absent vs parent in a rejected lineage.
        parent_chat = mt.parent_chat_id
        if parent_chat not in minimal_by_chat:
            reason = (
                f"orphan: parent chat_id {parent_chat} of "
                f"chat_id={chat_id} absent from trace"
            )
        else:
            reason = (
                f"orphan: parent chat_id {parent_chat} of "
                f"chat_id={chat_id} was rejected (in a bad lineage); child "
                f"is unreachable from any accepted root, fail-closed"
            )
        reject_subtree(chat_id, reason=reason)

    # Conservation check (fail-closed). Every record must be accounted for.
    if stats.accepted_records + stats.rejected_records != stats.total_records:
        raise LineageError(
            f"record conservation violated: accepted={stats.accepted_records} "
            f"+ rejected={stats.rejected_records} = "
            f"{stats.accepted_records + stats.rejected_records} but "
            f"total_records={stats.total_records}; "
            f"{stats.total_records - stats.accepted_records - stats.rejected_records} "
            f"record(s) silently lost or double-counted"
        )
    if len(assigned) != stats.total_records:
        raise LineageError(
            f"assignment conservation violated: assigned={len(assigned)} "
            f"!= total_records={stats.total_records}"
        )
    return accepted, rejected, stats


def _validate_lineage(root_id: int, turns: List[MinimalTurn]) -> Optional[str]:
    """Return a failure reason string, or None if the lineage is clean.

    Validation rules (all fail-closed):
      1. turn continuity: turns must be exactly 1, 2, ..., N (no gaps).
      2. no duplicate turn numbers.
      3. timestamps strictly monotonic non-decreasing within the lineage.
    Roots are assembled only from parent_chat_id == -1 records, so orphan
    parents never produce a root; the orphan children themselves are
    unreachable downward and thus naturally absent here. They are counted
    via stats.orphan_children in the higher-level caller if desired.
    """
    if not turns:
        return "empty lineage"
    ts = [t.timestamp for t in turns]
    turn_nums = [t.turn for t in turns]
    if len(set(turn_nums)) != len(turn_nums):
        return "duplicate turn"
    if turn_nums != list(range(1, len(turn_nums) + 1)):
        return (
            f"broken continuity: turns={turn_nums} not 1..{len(turn_nums)}"
        )
    if any(ts[i] < ts[i - 1] for i in range(1, len(ts))):
        return "non-monotonic timestamp"
    return None


def count_orphans(minimal_by_chat: Dict[int, MinimalTurn]) -> int:
    """Count records whose parent_chat_id points to a chat_id absent from
    the trace. Pass the index built during reconstruct_lineages.

    Diagnostic only: orphans are now FAIl-CLOSED into RejectedLineage during
    reconstruction (and counted in stats.orphan_records). This helper is
    kept for standalone audits and matches the strict definition.
    """
    n = 0
    for mt in minimal_by_chat.values():
        if mt.parent_chat_id != -1 and mt.parent_chat_id not in minimal_by_chat:
            n += 1
    return n
