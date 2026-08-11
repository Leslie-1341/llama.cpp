#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block -> text/token materialization contract (synthetic workload).

The Bailian trace stores hash_ids: salted SipHash of 16-token blocks,
remapped to sequential ints. These are IRREVERSIBLE. We CANNOT recover real
text or semantics, and we MUST NOT claim to.

This module defines the contract the synthetic workload uses to attach
deterministic block material to each hash_id so a future canonical runner
can produce inputs of the right length and intra-session prefix structure:

  prefix_mode = session_namespaced (the only mode emitted in gate 1A)
    * Within one session/lineage: the SAME global hash_id maps to the SAME
      deterministic block token sequence, so intra-session prefix carry-over
      is preserved (turn k+1's leading blocks match turn k's blocks).
    * ACROSS sessions: the material is NAMESPACED by lineage_id. The same
      global hash_id in two different lineages maps to two different block
      token sequences. This neutralizes the runtime's cross-session prefix
      cache (llama-server would not see a shared prefix across sessions even
      though the offline trace shows one), which is the confound gate 1A
      deliberately isolates.

Key decision (no guessing patch): we cannot reliably construct EXACT 16-token
text blocks that tokenize to a stable per-block token count without running
a real tokenizer. We therefore implement a PLACEHOLDER materializer that:
  * produces a deterministic, reproducible placeholder BLOCK_TOKENS-length
    integer-token-id sequence per (lineage_id, global_hash_id) pair; and
  * emits an explicit `tokenization_calibration_requirement` record stating
    that real text/token materialization is deferred to gate GT-trace-1B,
    which must close it by running the real target tokenizer.
The manifest binds a materialize_mode of "placeholder_token_ids" and the
calibration requirement; later gates flip the mode to "real_tokenized_text"
only after providing real tokenizer output.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

from .common import BLOCK_SIZE_TOKENS


# Materialization mode flags. "real_tokenized_text" is the target end-state
# that gate GT-trace-1B must enable; gate 1A only ever emits placeholder.
MATERIALIZE_MODE_PLACEHOLDER = "placeholder_token_ids"
MATERIALIZE_MODE_REAL = "real_tokenized_text"


@dataclass(frozen=True)
class NamespacedBlock:
    """A deterministic 16-token-id block materialized for one session.

    token_ids is a tuple of exactly BLOCK_SIZE_TOKENS ints. It is a
    placeholder: NOT real tokenizer output, NOT recoverable text. The
    contract is solely that (a) it is deterministic in (lineage_id,
    global_hash_id), and (b) distinct lineages get distinct material for
    the same global hash_id, so cross-session prefix reuse is impossible
    at the runtime layer.
    """

    lineage_id: int
    global_hash_id: int
    namespaced_block_id: str  # "L<lineage_id>:H<hash>"
    token_ids: Tuple[int, ...]


@dataclass
class TokenizationCalibrationRequirement:
    """Explicit gap record so GT-trace-1B can close it.

    Gate 1A emits this verbatim in every synthetic slice manifest. It MUST
    NOT be silently dropped or hand-waved; the synthetic workload is not
    valid for real byte-level inference until this is resolved.
    """

    status: str = "OPEN"  # OPEN -> RESOLVED by gate 1B
    required_by_gate: str = "GT-trace-1B"
    materialize_mode_now: str = MATERIALIZE_MODE_PLACEHOLDER
    reason: str = (
        "Bailian hash_ids are irreversible (salted SipHash, remapped). "
        "Gate 1A materializes deterministic placeholder token-id blocks "
        "of BLOCK_SIZE_TOKENS to preserve length + intra-session prefix "
        "structure, and namespaces them per lineage to isolate cross-"
        "session prefix confound. Real 16-token text that tokenizes "
        "deterministically under the target model's tokenizer must be "
        "supplied by gate GT-trace-1B (real/tokenize run); until then "
        "no claim of exact token count or text fidelity may be made."
    )
    block_size_tokens: int = BLOCK_SIZE_TOKENS
    required_action: str = (
        "Run the target model tokenizer to produce real 16-token text "
        "blocks; bind the resulting material and flip materialize_mode "
        "to real_tokenized_text. Re-validate deterministic event-stream "
        "hash and slice identity."
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Materializer:
    """Stateful namespaced block materializer.

    For one lineage, the SAME global hash_id must return the SAME block
    (deterministic + memoized). Across lineages, the mapping is namespaced,
    so the same global hash_id yields a DIFFERENT block. The offline
    cross-session overlap (characterization.global_hash_reuse) is preserved
    only in the offline statistic, never in the runtime material.

    The placeholder scheme: token_ids = deterministic PRF over
    (lineage_id, global_hash_id) -> 16 ints in a placeholder vocabulary
    range. NOT real tokens; deliberately looks like ints so no one
    mistakes them for text.
    """

    # Placeholder vocabulary range for token IDs (kept clear of common real
    # tokenizer ranges to avoid accidental collisions under real tokenization).
    PLACEHOLDER_TOKEN_BASE: int = 1_000_000
    PLACEHOLDER_TOKEN_RANGE: int = 65_536
    _memo: Dict[Tuple[int, int], NamespacedBlock] = field(default_factory=dict)

    def _materialize(self, lineage_id: int, ghash: int) -> NamespacedBlock:
        key = f"{lineage_id}:{ghash}".encode("ascii")
        dig = hashlib.sha256(key).digest()
        out: List[int] = []
        counter = 0
        buf = dig
        while len(out) < BLOCK_SIZE_TOKENS:
            buf = hashlib.sha256(buf + counter.to_bytes(4, "big")).digest()
            for b in buf:
                out.append(self.PLACEHOLDER_TOKEN_BASE + (b % self.PLACEHOLDER_TOKEN_RANGE))
                if len(out) >= BLOCK_SIZE_TOKENS:
                    break
            counter += 1
        return NamespacedBlock(
            lineage_id=lineage_id,
            global_hash_id=ghash,
            namespaced_block_id=f"L{lineage_id}:H{ghash}",
            token_ids=tuple(out),
        )

    def materialize_block(self, lineage_id: int, global_hash_id: int) -> NamespacedBlock:
        """Deterministic namespaced materialization. Same (lid,ghash)->same block."""
        return self._memo.setdefault(
            (lineage_id, global_hash_id),
            self._materialize(lineage_id, global_hash_id),
        )

    def materialize_record(
        self, lineage_id: int, global_hash_ids: Tuple[int, ...]
    ) -> List[NamespacedBlock]:
        return [self.materialize_block(lineage_id, h) for h in global_hash_ids]

    @property
    def mode(self) -> str:
        return MATERIALIZE_MODE_PLACEHOLDER

    @property
    def calibration_requirement(self) -> TokenizationCalibrationRequirement:
        return TokenizationCalibrationRequirement()
