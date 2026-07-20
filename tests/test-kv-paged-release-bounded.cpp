// Bounded KV release unit correctness test — validates paged_release_blocks_bounded()
// without connecting to server, pressure sampler, swap, or prefetch paths.
//
// Part A (no-model): synthetic ownership-collection ABORT detection.
// Part B (model-dependent): real bounded-release gates via test-only seams:
//   B1-B2:   basic edge cases (target=0, max_scan=0)
//   B3-B4:   overshoot + scan-budget
//   B5:      invalid ownership ABORT via real paged_release_blocks_bounded() path
//   B6:      PENDING_WRITE dynamic block state override → skip + zero-change
//   B7:      madvise failure injection → no state change, scan continues
//   B8:      active-owned skip + output consistency (dual context)
//   B9-B11:  shortfall, idempotent, corner cases
//
// Seam-dependent tests (B5-B7) run in fresh contexts to avoid the all-RELEASED
// block reuse edge case.
//
// Requires a GGUF model. Pass it via LLAMACPP_TEST_MODELFILE or argv[1].
// Without a model the test exits with code 77 (CTest SKIP_RETURN_CODE).

#include "llama.h"

#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-release.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s\n", msg); \
        failures++; \
    } \
} while(0)

static const int SKIP_EXIT_CODE = 77;

// ===========================================================================
// Part A: Synthetic ownership ABORT detection.
// ===========================================================================

struct fake_cells {
    std::vector<unsigned> owners;
    uint32_t used_max_p1() const { return (uint32_t) owners.size(); }
    bool is_empty(uint32_t cell) const { return owners[cell] == 0; }
    bool seq_has(uint32_t cell, int32_t seq) const { return (owners[cell] & (1u << seq)) != 0; }
};

static void test_ownership_fault_fixture() {
    {
        const std::vector<fake_cells> streams {{{1, 2, 0, 0}}};
        const auto r = llama_kv_release_collect_ownership(
                streams, 2, 4, UINT32_MAX, 4,
                [](uint32_t cell) { return cell; });
        CHECK(r.valid, "FAULT_1: identity valid");
    }
    {
        const std::vector<fake_cells> streams {{{1, 2, 0, 0}}};
        const auto r = llama_kv_release_collect_ownership(
                streams, 2, 4, UINT32_MAX, 4,
                [](uint32_t cell) { return cell == 1 ? UINT32_MAX : cell; });
        CHECK(!r.valid, "FAULT_2: invalid mapping detected");
        CHECK(r.invalid_mappings == 1, "FAULT_2: exactly 1 invalid");
    }
    {
        const std::vector<fake_cells> streams {
            fake_cells{{1, 1, 0, 0}}, fake_cells{{1, 0, 0, 0}}};
        const auto r = llama_kv_release_collect_ownership(
                streams, 2, 4, UINT32_MAX, 4,
                [](uint32_t) { return UINT32_MAX; });
        CHECK(!r.valid, "FAULT_3: all invalid");
        CHECK(r.invalid_mappings == 3, "FAULT_3: 3 cells invalid");
    }
    {
        const std::vector<fake_cells> streams {{{1, 0, 1, 0}}};
        const auto r = llama_kv_release_collect_ownership(
                streams, 1, 4, UINT32_MAX, 4,
                [](uint32_t cell) { return cell == 2 ? 4u : cell; });
        CHECK(!r.valid, "FAULT_4: OOB block detected");
    }
    std::fprintf(stderr, "Part A ownership fault fixture: 4 groups OK\n");
}

// ===========================================================================
// Helpers
// ===========================================================================

static int decode_prompt(llama_context * lctx, const std::vector<llama_token> & tokens) {
    std::vector<llama_token> full = { 128000 };
    full.insert(full.end(), tokens.begin(), tokens.end());
    llama_batch b = llama_batch_get_one(full.data(), (int32_t) full.size());
    return llama_decode(lctx, b);
}

struct ContextGuard {
    llama_context * ctx = nullptr;
    llama_kv_cache * kv = nullptr;
    llama_memory_t  mem = nullptr;

    bool init(llama_model * model, const llama_context_params & cparams) {
        ctx = llama_init_from_model(model, cparams);
        if (!ctx) return false;
        mem = llama_get_memory(ctx);
        kv = static_cast<llama_kv_cache *>(mem);
        return kv != nullptr;
    }

    ~ContextGuard() { if (ctx) llama_free(ctx); }
};

// Snapshot helpers
static void snapshot_states(llama_kv_cache * kv, std::vector<uint8_t> & out, uint32_t n) {
    out.resize(n);
    for (uint32_t b = 0; b < n; ++b)
        out[b] = kv->paged_release_bounded_test_read_block_state(b);
}
static void snapshot_free(llama_kv_cache * kv, std::vector<bool> & out, uint32_t n) {
    out.resize(n);
    for (uint32_t b = 0; b < n; ++b)
        out[b] = kv->paged_release_bounded_test_block_in_free_list(b);
}

// ===========================================================================
// Part B: Real-model tests.
// ===========================================================================

int main(int argc, char ** argv) {
    test_ownership_fault_fixture();

    const char * model_path = nullptr;
    if (argc > 1) model_path = argv[1];
    else model_path = getenv("LLAMACPP_TEST_MODELFILE");

    if (!model_path || strlen(model_path) == 0) {
        std::fprintf(stderr, "SKIP: no model file. "
                "Set LLAMACPP_TEST_MODELFILE=<gguf_path> to run Part B.\n");
        return SKIP_EXIT_CODE;
    }

    setenv("LLAMA_KV_PAGED", "1", 1);
    setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);
    setenv("LLAMA_KV_PAGED_BLOCK_SIZE", "16", 1);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    auto * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        std::fprintf(stderr, "FAIL: failed to load model\n");
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;

    // Primary context for non-seam tests
    ContextGuard main_ctx;
    if (!main_ctx.init(model, cparams)) {
        std::fprintf(stderr, "FAIL: failed to create main context\n");
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    // =========================================================================
    // B1: target=0 → immediate return
    // =========================================================================
    {
        const auto r = main_ctx.kv->paged_release_blocks_bounded(0, 100);
        CHECK(r.released_bytes == 0, "B1: released_bytes == 0");
        CHECK(r.released_blocks == 0, "B1: released_blocks == 0");
        CHECK(r.blocks_scanned == 0, "B1: blocks_scanned == 0");
        CHECK(!r.ownership_aborted, "B1: ownership_aborted == false");
        CHECK(!r.block_scan_exhausted, "B1: exhausted == false");
        CHECK(r.shortfall_bytes == 0, "B1: shortfall == 0");
        CHECK(r.overshoot_bytes == 0, "B1: overshoot == 0");
        std::fprintf(stderr, "B1 target=0: OK\n");
    }

    // =========================================================================
    // B2: max_scan_blocks=0 → exhausted + full shortfall
    // =========================================================================
    {
        const uint64_t t = 65536;
        const auto r = main_ctx.kv->paged_release_blocks_bounded(t, 0);
        CHECK(r.block_scan_exhausted, "B2: exhausted == true");
        CHECK(r.shortfall_bytes == t, "B2: shortfall == target");
        CHECK(r.released_bytes == 0, "B2: released_bytes == 0");
        CHECK(r.blocks_scanned == 0, "B2: blocks_scanned == 0");
        CHECK(!r.ownership_aborted, "B2: ownership_aborted == false");
        std::fprintf(stderr, "B2 max_scan_blocks=0: OK\n");
    }

    // =========================================================================
    // B3: overshoot — target=1, one block released → overshoot
    // =========================================================================
    {
        std::vector<llama_token> toks(20, 1);
        int rc = decode_prompt(main_ctx.ctx, toks);
        CHECK(rc == 0, "B3: decode must succeed");
        if (rc == 0) {
            llama_memory_seq_rm(main_ctx.mem, 0, -1, -1);
            const auto r = main_ctx.kv->paged_release_blocks_bounded(1, UINT32_MAX);
            CHECK(r.released_blocks > 0, "B3: released_blocks > 0");
            CHECK(r.released_bytes > 1, "B3: released_bytes > target");
            CHECK(r.overshoot_bytes == r.released_bytes - 1,
                    "B3: overshoot == released - target");
            CHECK(r.shortfall_bytes == 0, "B3: shortfall == 0");
            CHECK(!r.ownership_aborted, "B3: ownership_aborted == false");
            std::fprintf(stderr, "B3 overshoot: %" PRIu64 "B/%" PRIu32
                    " blocks overshoot=%" PRIu64 " OK\n",
                    r.released_bytes, r.released_blocks, r.overshoot_bytes);
        }
    }

    // =========================================================================
    // B4: scan budget — blocks_scanned == max_scan_blocks
    // =========================================================================
    {
        const auto r = main_ctx.kv->paged_release_blocks_bounded(UINT64_MAX, 2);
        CHECK(r.blocks_scanned == 2, "B4: blocks_scanned == budget");
        CHECK(r.block_scan_exhausted, "B4: exhausted == true");
        std::fprintf(stderr, "B4 scan budget: scanned=%" PRIu32 " OK\n",
                r.blocks_scanned);
    }

    // =========================================================================
    // B5: Invalid ownership ABORT — real paged_release_blocks_bounded() path.
    //   Uses test_force_ownership_abort seam in a FRESH context.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "B5: fresh context creation failed");
        } else {
            // Create RESIDENT blocks via decode
            std::vector<llama_token> toks = { 50, 51, 52, 53, 54, 55, 56, 57 };
            int rc = decode_prompt(g.ctx, toks);
            CHECK(rc == 0, "B5: decode must succeed");

            // Snapshot pre-ABORT state (first 4 blocks)
            std::vector<uint8_t> st_before;
            std::vector<bool> fl_before;
            snapshot_states(g.kv, st_before, 4);
            snapshot_free(g.kv, fl_before, 4);

            // Trigger ABORT via test-only seam
            g.kv->paged_release_bounded_test_force_ownership_abort = true;
            // Also set block_state_override — verify it is auto-cleared by the
            // ABORT path (single-shot: all test seams reset together).
            g.kv->paged_release_bounded_test_block_state_override.block = 0;
            g.kv->paged_release_bounded_test_block_state_override.state = 4;
            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);

            CHECK(r.ownership_aborted, "B5: ownership_aborted == true");
            CHECK(r.released_blocks == 0, "B5: ABORT → released_blocks == 0");
            CHECK(r.released_bytes == 0, "B5: ABORT → released_bytes == 0");
            CHECK(r.blocks_scanned == 0, "B5: ABORT → blocks_scanned == 0");
            CHECK(!r.block_scan_exhausted, "B5: ABORT → exhausted == false");
            CHECK(r.shortfall_bytes == 0, "B5: ABORT → shortfall == 0");
            CHECK(r.overshoot_bytes == 0, "B5: ABORT → overshoot == 0");
            CHECK(r.blocks_skipped_owned == 0, "B5: ABORT → skipped_owned == 0");
            CHECK(r.blocks_skipped_state == 0, "B5: ABORT → skipped_state == 0");
            CHECK(r.madvise_failures == 0, "B5: ABORT → madvise_failures == 0");

            // Single-shot flags were auto-reset by the ABORT path
            CHECK(!g.kv->paged_release_bounded_test_force_ownership_abort,
                    "B5: force_ownership_abort flag auto-reset after ABORT");
            CHECK(g.kv->paged_release_bounded_test_block_state_override.block == UINT32_MAX,
                    "B5: block_state_override auto-reset after ABORT");

            // Verify zero state changes
            std::vector<uint8_t> st_after;
            std::vector<bool> fl_after;
            snapshot_states(g.kv, st_after, 4);
            snapshot_free(g.kv, fl_after, 4);

            bool ok = true;
            for (uint32_t b = 0; b < 4; ++b) {
                if (st_before[b] != st_after[b]) { ok = false; break; }
                if (fl_before[b] != fl_after[b]) { ok = false; break; }
            }
            CHECK(ok, "B5: all block states + free list unchanged after ABORT");

            std::fprintf(stderr, "B5 ownership ABORT: released=%" PRIu32
                    " states_ok=%d OK\n", r.released_blocks, ok ? 1 : 0);
        }
    }

    // =========================================================================
    // B6: PENDING_WRITE dynamic override on a NON-OWNED block — verify that the
    //   state gate (not the earlier ownership gate) realiably skips the block.
    //   If the block were still owned, `owned[block]` would skip before the state
    //   override is even read — that would be fake coverage.
    //   Fresh context.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "B6: fresh context creation failed");
        } else {
            // Decode → create RESIDENT blocks owned by seq 0
            std::vector<llama_token> toks = { 60, 61, 62, 63, 64, 65, 66, 67, 68, 69 };
            int rc = decode_prompt(g.ctx, toks);
            CHECK(rc == 0, "B6: decode must succeed");

            // Remove seq 0 → blocks become dead (unowned).  This is critical:
            // the PENDING_WRITE state gate runs *after* the ownership check;
            // without this step the ownership check would skip the block first
            // and the state override would never be exercised.
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint32_t tb = 0; // target block
            uint8_t real_state = g.kv->paged_release_bounded_test_read_block_state(tb);
            bool real_free = g.kv->paged_release_bounded_test_block_in_free_list(tb);

            // Override: bounded release sees block 0 as PENDING_WRITE (4)
            g.kv->paged_release_bounded_test_block_state_override.block = tb;
            g.kv->paged_release_bounded_test_block_state_override.state = 4;

            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
            CHECK(!r.ownership_aborted, "B6: ownership valid");

            // Override was auto-reset
            CHECK(g.kv->paged_release_bounded_test_block_state_override.block == UINT32_MAX,
                    "B6: override auto-reset");

            // Block was skipped by the state gate (not the owned gate)
            CHECK(r.blocks_skipped_owned == 0,
                    "B6: zero blocks skipped by owned gate (block is dead)");
            CHECK(r.blocks_skipped_state >= 1,
                    "B6: at least one block skipped by state gate (PENDING_WRITE hit)");

            // Real state unchanged
            uint8_t real_after = g.kv->paged_release_bounded_test_read_block_state(tb);
            CHECK(real_after == real_state,
                    "B6: real block state unchanged after PENDING_WRITE skip");
            bool free_after = g.kv->paged_release_bounded_test_block_in_free_list(tb);
            CHECK(free_after == real_free,
                    "B6: block not added to free list by PENDING_WRITE skip");

            // madvise was never called on the skipped block
            CHECK(r.madvise_failures == 0,
                    "B6: no madvise failures (madvise was never called on skipped block)");

            std::fprintf(stderr, "B6 PENDING_WRITE: state=%u→%u free=%d→%d "
                    "skipped_owned=%" PRIu32 " skipped_state=%" PRIu32 " OK\n",
                    real_state, real_after, real_free ? 1 : 0, free_after ? 1 : 0,
                    r.blocks_skipped_owned, r.blocks_skipped_state);
        }
    }

    // =========================================================================
    // B7: madvise failure injection — block NOT released, NOT in free list,
    //   scan continues to subsequent candidates. Fresh context.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "B7: fresh context creation failed");
        } else {
            // Decode → create RESIDENT blocks
            std::vector<llama_token> toks = { 70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
                                              80, 81, 82, 83, 84, 85, 86, 87 };
            int rc = decode_prompt(g.ctx, toks);
            CHECK(rc == 0, "B7: decode must succeed");

            // Remove seq → all blocks dead
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint32_t tb = 0;
            uint8_t b0_before = g.kv->paged_release_bounded_test_read_block_state(tb);
            bool b0_free_before = g.kv->paged_release_bounded_test_block_in_free_list(tb);

            // Inject madvise failure on block 0
            g.kv->paged_release_bounded_test_madvise_fail_block = (int32_t) tb;

            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);

            // Test flag auto-reset
            CHECK(g.kv->paged_release_bounded_test_madvise_fail_block == -1,
                    "B7: test flag auto-reset");

            // Block 0 state unchanged
            uint8_t b0_after = g.kv->paged_release_bounded_test_read_block_state(tb);
            CHECK(b0_after == b0_before,
                    "B7: madvise-failed block state unchanged (not RELEASED)");

            bool b0_free_after = g.kv->paged_release_bounded_test_block_in_free_list(tb);
            CHECK(b0_free_after == b0_free_before,
                    "B7: madvise-failed block not added to free list");

            // Scan continued past the failed block
            CHECK(r.blocks_scanned > 1,
                    "B7: blocks_scanned > 1 (scan continued past failed block)");
            // Later blocks were released
            CHECK(r.released_blocks > 0,
                    "B7: subsequent blocks released (scan continued)");
            CHECK(r.released_bytes > 0,
                    "B7: released budget > 0 (from subsequent blocks)");
            CHECK(!r.ownership_aborted, "B7: ownership valid");

            // madvise failure counter reflects the injected failure
            CHECK(r.madvise_failures == 1,
                    "B7: exactly 1 madvise failure recorded");
            // The failed block was not owned → no owned-skip
            CHECK(r.blocks_skipped_owned == 0,
                    "B7: zero owned skips (all blocks dead)");

            std::fprintf(stderr, "B7 madvise fail: b0_state=%u→%u b0_free=%d→%d "
                    "released=%" PRIu32 " blocks scanned=%" PRIu32
                    " madvise_failures=%" PRIu32 " OK\n",
                    b0_before, b0_after, b0_free_before ? 1 : 0, b0_free_after ? 1 : 0,
                    r.released_blocks, r.blocks_scanned, r.madvise_failures);
        }
    }

    // =========================================================================
    // B8: active-owned blocks — decode MUST succeed, bounded release MUST
    //   release zero blocks, output MUST match no-release control.
    // =========================================================================
    {
        auto * ctx1 = llama_init_from_model(model, cparams);
        auto * ctx2 = llama_init_from_model(model, cparams);
        if (!ctx1 || !ctx2) {
            std::fprintf(stderr, "B8: dual context failed — skipped\n");
            if (ctx1) llama_free(ctx1);
            if (ctx2) llama_free(ctx2);
        } else {
            const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
            auto * kv2 = static_cast<llama_kv_cache *>(llama_get_memory(ctx2));

            std::vector<llama_token> prompt = { 100, 200, 300, 400, 500 };
            std::vector<llama_token> cont = { 600 };

            // ctx1: baseline
            llama_batch b1 = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
            CHECK(llama_decode(ctx1, b1) == 0, "B8: baseline prompt decode ok");
            llama_batch c1 = llama_batch_get_one(cont.data(), (int32_t) cont.size());
            CHECK(llama_decode(ctx1, c1) == 0, "B8: baseline continuation ok");
            float * logits1 = llama_get_logits(ctx1);
            CHECK(logits1 != nullptr, "B8: baseline logits not null");

            // ctx2: bounded release between prompt and continuation
            llama_batch b2 = llama_batch_get_one(prompt.data(), (int32_t) prompt.size());
            CHECK(llama_decode(ctx2, b2) == 0, "B8: variant prompt decode ok");

            const auto r = kv2->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
            CHECK(!r.ownership_aborted, "B8: ownership_aborted == false");
            CHECK(r.released_blocks == 0, "B8: released_blocks == 0");
            CHECK(r.released_bytes == 0, "B8: released_bytes == 0");
            CHECK(r.blocks_skipped_owned > 0,
                    "B8: blocks_skipped_owned > 0 (active seq owns all blocks)");

            llama_batch c2 = llama_batch_get_one(cont.data(), (int32_t) cont.size());
            CHECK(llama_decode(ctx2, c2) == 0, "B8: variant continuation ok");
            float * logits2 = llama_get_logits(ctx2);
            CHECK(logits2 != nullptr, "B8: variant logits not null");

            bool match = true;
            for (int i = 0; i < nv; ++i) {
                if (logits1[i] != logits2[i]) { match = false; break; }
            }
            CHECK(match, "B8: logits match — K/V cache not corrupted");

            std::fprintf(stderr, "B8 active-owned: released=%" PRIu32
                    " logits_match=%d OK\n", r.released_blocks, match ? 1 : 0);
            llama_free(ctx1);
            llama_free(ctx2);
        }
    }

    // =========================================================================
    // B9-B11: shortfall, idempotent, corner cases (use main_ctx)
    // =========================================================================
    {
        // B9: shortfall
        llama_memory_seq_rm(main_ctx.mem, 0, -1, -1);
        main_ctx.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
        const auto r = main_ctx.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
        CHECK(r.block_scan_exhausted, "B9: exhausted == true");
        CHECK(r.shortfall_bytes == UINT64_MAX - r.released_bytes,
                "B9: shortfall == target - released");
        CHECK(!r.ownership_aborted, "B9: ownership_aborted == false");
        std::fprintf(stderr, "B9 shortfall: released=%" PRIu32 " shortfall=%" PRIu64 " OK\n",
                r.released_blocks, r.shortfall_bytes);
    }
    {
        // B10: idempotent repeat — all blocks already RELEASED
        const auto r1 = main_ctx.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
        const auto r2 = main_ctx.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
        CHECK(r2.released_blocks == 0, "B10: repeat released_blocks == 0");
        CHECK(r2.released_bytes == 0, "B10: repeat released_bytes == 0");
        CHECK(!r2.ownership_aborted, "B10: ownership_aborted == false");
        CHECK(r2.blocks_skipped_state == r1.blocks_skipped_state,
                "B10: idempotent — same skipped_state count both calls");
        CHECK(r2.blocks_skipped_state > 0,
                "B10: repeat skipped_state > 0 (RELEASED gate hit)");
        std::fprintf(stderr, "B10 idempotent: first=%" PRIu32 " second=%" PRIu32
                " skipped_state=%" PRIu32 " OK\n",
                r1.released_blocks, r2.released_blocks, r2.blocks_skipped_state);
    }
    {
        // B11: scan precision + corner case
        const auto r = main_ctx.kv->paged_release_blocks_bounded(UINT64_MAX, 3);
        CHECK(r.blocks_scanned == 3, "B11a: blocks_scanned == 3");
        CHECK(r.block_scan_exhausted, "B11a: exhausted == true");

        const auto r2 = main_ctx.kv->paged_release_blocks_bounded(0, 0);
        CHECK(r2.released_bytes == 0, "B11b: target=0 max_scan=0 → released 0");
        CHECK(r2.blocks_scanned == 0, "B11b: blocks_scanned == 0");
        CHECK(!r2.block_scan_exhausted, "B11b: exhausted == false");
        CHECK(!r2.ownership_aborted, "B11b: ownership_aborted == false");
        std::fprintf(stderr, "B11 scan precision + corner: OK\n");
    }

    llama_model_free(model);
    llama_backend_free();

    if (failures > 0) {
        std::fprintf(stderr, "\n%d test(s) FAILED\n", failures);
        return 1;
    }
    std::printf("PASS: paged release bounded correctness\n");
    return 0;
}
