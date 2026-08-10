#pragma once

#include "llama.h"

#include <cstdint>
#include <vector>

enum class llama_kv_action : uint8_t {
    noop,
    evaluate,
    prefetch,
    release,
    offload,
};

enum class llama_kv_action_outcome : uint8_t {
    completed,
    no_op,
    unsupported,
    rejected,
    failed,
    partial_failure,
};

enum class llama_kv_memory_claimant : uint8_t {
    kv,
    dense_weight,
    moe_expert,
};

enum class llama_kv_io_class : uint8_t {
    correctness_read,
    latency_read,
    capacity_write,
    background_read,
    background_write,
};

enum class llama_kv_action_reason : uint8_t {
    none,
    zero_budget,
    invalid_sequence,
    context_invalid,
    write_transaction_open,
    unsupported,
    protected_sequence,
    shared_block,
    no_eligible_block,
    state_rejected,
    ownership_invalid,
    io_failure,
    prefetch_failed,
    no_candidate,
    scan_budget_exhausted,
    target_satisfied,
    target_shortfall,
    blocked,
    failed,
};

struct llama_kv_action_request {
    llama_kv_action action = llama_kv_action::noop;
    uint64_t decision_id = 0;
    llama_seq_id seq_id = -1;
    uint64_t target_bytes = 0;
    uint32_t max_blocks = 0;
    bool correctness_required = false;
    // Restore every currently SWAPPED block owned by seq_id; max_blocks is
    // intentionally ignored for this explicit correctness path.
    bool all_required = false;
    llama_kv_memory_claimant claimant = llama_kv_memory_claimant::kv;
    llama_kv_io_class io_class = llama_kv_io_class::capacity_write;
    int32_t priority = 0;
    uint64_t io_byte_budget = 0;
};

struct llama_kv_action_capability {
    bool context_invalid = false;
    bool write_transaction_open = false;
    bool can_prefetch = false;
    bool can_release = false;
    bool can_offload = false;
};

// Snapshot of the production KV object used by server integration checks.  These
// fields describe resolved runtime state, not command-line intent.
struct llama_kv_runtime_capability {
    uint32_t n_seq_max = 0;
    uint32_t n_stream = 0;
    bool kv_unified = false;
    bool paged_metadata = false;
    bool ingraph_gather = false;
    bool release_supported = false;
    bool offload_supported = false;
    bool prefetch_supported = false;
    bool backing_ready = false;
    bool swap_explicit_only = false;
};

struct llama_kv_runtime_claimant {
    bool valid = false;
    uint32_t target_blocks = 0;
    uint32_t eligible_resident_blocks = 0;
    uint32_t swapped_blocks = 0;
    uint32_t shared_blocks = 0;
    uint32_t blocked_blocks = 0;
};

// One batch physical view for server-side claimant ranking.  The core keeps
// ownership/state and mincore authoritative; only aggregate per-sequence
// quantities cross the interface.  No block table or cell mapping is exposed.
struct llama_kv_claimant_physical_view {
    llama_seq_id seq_id = -1;
    bool valid = false;
    bool available = false;
    bool authoritative = false;
    bool shared = false;
    uint64_t object_id = 0;
    uint64_t generation = 0;
    uint64_t estimated_exclusive_resident_bytes = 0;
    uint64_t estimated_swapped_bytes = 0;
    uint32_t exclusive_resident_blocks = 0;
    uint32_t swapped_blocks = 0;
};

// A read-only, whole-KV mincore sample.  `available` distinguishes a valid
// zero-resident result from an unsupported or failed platform probe.
struct llama_kv_resident_sample {
    bool available = false;
    uint64_t object_id = 0;
    uint64_t generation = 0;
    uint64_t page_size = 0;
    uint64_t total_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
};

// Read-only whole-KV physical budget view — single coherent snapshot of the
// physical state required by a unified scheduler to compute resident /
// transient reservations.  All fields are derived at snapshot time from
// authoritative core state (block-state enum + swap metadata + mincore
// resident sample + reclaimable dry-run + ownership collector + K2 staging
// bound).  No shadow counter is introduced; no server lifecycle vector
// (idle_age / active / reuse / hotness / score / reuse prediction / attention
// heat) is included.  Hot/Elastic/Cold categorization is the scheduler's
// concern — this view only exposes neutral ownership counts.
//
// Field semantics:
//   - object_id: stable identifier for the cache instance that produced this
//     snapshot.  Assigned once in the llama_kv_cache constructor
//     (paged_resident_object_id) and never reset by clear()/paged_reset().
//   - generation: monotonic counter incremented at every clear() boundary.
//     Binds this snapshot to the in-flight generation; readers must compare
//     generation across snapshots taken between actions.
//   - total_bytes: full logical K/V tensor capacity backing this object;
//     equals `paged_mincore_total_bytes` when mincore is available, else
//     derived from per-layer K/V row sizes × kv_size.
//   - resident_bytes: page-aligned mincore count across all K/V tensors,
//     identical to `sample_kv_resident().resident_bytes`.  `resident_available`
//     distinguishes a true zero from an unsupported / failed probe.
//   - dead_resident_reclaimable_bytes: bytes that a destructive RELEASE in
//     the current layout could actually free — ownership-gated and
//     restricted to RESIDENT/UNUSED unowned blocks.  Identical to
//     `sample_kv_release_budget().reclaimable_resident_bytes`.
//   - swapped_authoritative_bytes: live logical backing bytes for blocks
//     currently in `SWAPPED` state — the authoritative copy held by the
//     fixed-slot backing file.  Computed by summing `paged_swap_sizes[c]`
//     for cells belonging to `SWAPPED` blocks; not equal to filesystem
//     `st_blocks`, fixed-slot capacity, or cumulative `bytes_written`.
//   - swapped_metadata_consistent: false iff any of the following fail-closed
//     conditions are detected at snapshot time — `paged_swap_sizes.size()`
//     does not equal `paged_kv_size`; any non-zero `paged_swap_sizes[c]`
//     exceeds the per-cell logical byte budget; any non-zero
//     `paged_swap_sizes[c]` lives in a block whose state is one of
//     RELEASED / UNUSED / PENDING_WRITE / INVALID.  Non-zero metadata is
//     LEGITIMATELY retained in `SWAPPED` blocks (the authoritative backing
//     publication) AND in `RESIDENT` blocks immediately after a PREFETCH
//     restore (the swap_in path preserves swap_sizes until the next
//     release / reset boundary) — neither is corruption.  When false,
//     `swapped_authoritative_bytes` MUST NOT be used as a coherent bound;
//     `valid` is also cleared so callers cannot treat the snapshot as
//     authoritative.
//   - transient_staging_bound_bytes: K2 producer-consumer staging upper
//     bound (2 × max(group_byte_cap, max_single_block_bytes)).  Zero when
//     K2 restore pipeline is disabled OR no restore pipeline has run yet.
//     Orthogonal to resident_bytes.
//   - n_owned_blocks / n_shared_blocks: neutral block-count aggregates
//     computed by a single `llama_kv_release_collect_ownership()` pass
//     over `v_cells`.  `n_owned_blocks` = blocks touched by at least one
//     live sequence; `n_shared_blocks` = blocks touched by ≥2 live
//     sequences.  These are ownership counts only — no byte computation,
//     no policy assignment, no Hot/Elastic/Cold classification.
//
// Cost: O(paged_n_blocks) for the block-state walk + O(layers) for the
// per-layer row-size sum + one mincore pass + one release-budget dry-run
// + one ownership collector pass.  Read-only; safe to call from the same
// single owner that drives the unified Governor.
struct llama_kv_physical_budget_view {
    bool     valid = false;
    bool     resident_available = false;  // true iff mincore sampler is usable
    bool     reclaimable_available = false;  // true iff release-budget scanner usable
    bool     swapped_metadata_consistent = false;  // fail-closed flag for swap metadata integrity
    uint64_t object_id = 0;
    uint64_t generation = 0;
    uint64_t page_size = 0;
    uint64_t total_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t dead_resident_reclaimable_bytes = 0;
    uint64_t swapped_authoritative_bytes = 0;
    uint64_t transient_staging_bound_bytes = 0;
    uint32_t n_blocks = 0;
    uint32_t n_owned_blocks = 0;   // blocks with ≥1 live sequence owner (any state)
    uint32_t n_shared_blocks = 0;  // subset of n_owned_blocks with ≥2 live sequence owners
    uint32_t resident_block_count = 0;
    uint32_t swapped_block_count = 0;
    uint32_t released_block_count = 0;
    uint32_t pending_write_block_count = 0;
    uint32_t invalid_block_count = 0;
    uint32_t unused_block_count = 0;
};

struct llama_kv_action_result {
    llama_kv_action action = llama_kv_action::noop;
    uint64_t decision_id = 0;
    uint64_t core_transaction_id = 0;
    llama_kv_action_outcome outcome = llama_kv_action_outcome::no_op;
    llama_kv_action_reason reason = llama_kv_action_reason::none;
    bool state_changed = false;
    bool fail_stop = false;
    bool io_failure = false;
    int io_errno = 0;
    uint32_t blocks = 0;
    uint64_t bytes = 0;
    uint64_t relieved_bytes = 0;
    uint64_t shortfall_bytes = 0;
    // Action-local feedback.  `physical_relief_bytes` is populated only when
    // a before/after physical sample has the same object and generation;
    // advised/logical bytes are never silently promoted to physical relief.
    uint64_t action_elapsed_us = 0;
    uint64_t physical_relief_bytes = 0;
    bool physical_relief_available = false;
    uint64_t physical_object_id = 0;
    uint64_t physical_generation = 0;
    llama_kv_action_capability capability;
};
