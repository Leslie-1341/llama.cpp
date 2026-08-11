#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qualification contract (no budget chosen in this gate).

Gate 1A only DEFINES the contract that a future budget-qualification gate
must satisfy. It does NOT pick a moderate/tight memory budget and it does
NOT bake in any number derived from the synthetic workload sweep. The
contract explicitly forbids the previously-attempted wrong bounds:

  * The OFFLOAD-strategy comparison budget must be BELOW the release-only
    residual working-set AND ABOVE the real reachable floor.
  * It must NOT be `B_release_floor * 1.1` (that sits above the release-only
    residual and so never triggers OFFLOAD — a non-test).
  * It must NOT be the artificial 1.0 GiB / 1.5 GiB of the human-workload
    sweep (those are not trace-derived).
  * The exact number is to be FROZEN by a future gate that physically
    probes Resident -> RELEASE-only -> V2 capacity, in that dependency
    order, against the FROZEN workload manifest this gate emits.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class BudgetQualificationContract:
    """Frozen-by-later-gate budget contract for OFFLOAD strategy comparison.

    status = DEFINED in gate 1A; no value chosen. A future gate must run the
    ordered probe and flip status to FROZEN with a numeric budget.
    """

    scope: str = (
        "OFFLOAD-strategy comparison budget for the V3 KV offload canonical "
        "benchmark, replayed against a GT-trace-1A frozen workload manifest."
    )
    status: str = "DEFINED"  # DEFINED -> FROZEN by a later probe gate
    upper_bound_inclusive: Optional[str] = None  # to be filled: release-only residual WS
    lower_bound_inclusive: Optional[str] = None  # to be filled: real reachable floor
    forbidden_upper_bound_formula: str = "B_release_floor * 1.1"
    forbidden_budget_keys: List[str] = field(
        default_factory=lambda: [
            "B_release_floor * 1.1",  # would sit above residual WS; never triggers OFFLOAD
            "synthetic_workload 1.0 GiB",  # not trace-derived
            "synthetic_workload 1.5 GiB",  # not trace-derived
        ]
    )
    required_probe_order: List[str] = field(
        default_factory=lambda: [
            "Resident (no evict) baseline",
            "RELEASE-only residual working-set measurement",
            "real reachable floor measurement",
            "V2 elastic capacity probe",
        ]
    )
    invariant: str = (
        "chosen_budget < release_only_residual_working_set AND "
        "chosen_budget > real_reachable_floor, with both bounds measured "
        "physically against THIS gate's frozen workload manifest. A budget "
        "above the release-only residual or below the real reachable floor "
        "is a non-test and MUST NOT be accepted."
    )

    def to_dict(self) -> dict:
        return asdict(self)
