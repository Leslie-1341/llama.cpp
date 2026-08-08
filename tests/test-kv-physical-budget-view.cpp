// KV-Budget-V1 Physical Budget View — read-only whole-KV snapshot test.
//
// Validates that `sample_kv_physical_budget_view()` reconciles byte and
// identity accounting with the existing authoritative samplers and with
// the real OFFLOAD → SWAPPED → PREFETCH → RESIDENT state cycle, without
// touching the legacy `LLAMA_KV_PAGED_RELEASE` apply path or any other
// Cleanup-C2 deletion surface.
//
// Fixture: synthetic in-memory LLM_ARCH_LLAMA model built via
// `llama_model_init_from_user` (no external GGUF model file required, no
// CTest SKIP).  Reuses the same `ContextGuard` and `make_synthetic_model`
// pattern as `test-kv-paged-release-bounded.cpp`.
//
// Tests (read-only snapshot, no policy action):
//   BV0: structural precondition — invalid snapshot when no context is open
//   BV1: post-decode identity/byte reconciliation vs sample_kv_resident()
//   BV2: post-seq_rm dead_resident_reclaimable_bytes vs sample_kv_release_budget()
//   BV3: OFFLOAD → SWAPPED — swapped_authoritative_bytes == offload.bytes for
//        the clean single-block case (precise reconciliation)
//   BV4: PREFETCH → RESIDENT — swap bookkeeping zeroes out; resident_bytes recovers
//   BV5: unified RELEASE+clear() lifecycle (no LLAMA_KV_PAGED_RELEASE apply
//        path; survives Cleanup-C2 deletion of legacy gate)
//   BV6: per-state block counters reflect real state-machine output.
//   BV7: object_id stable across clear() (cache-instance id); generation bumps
//   BV8: transient_staging_bound_bytes = 0 when K2 disabled; equals
//        paged_restore_k2_staging_bound_bytes when K2 enabled
//   BV9: shared ownership via seq_cp(0,1) — n_owned_blocks / n_shared_blocks
//        reconcile with the existing llama_kv_release_collect_ownership helper

#include "common.h"
#include "llama.h"
#include "llama-cpp.h"

#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-batch.h"
#include "../src/llama-model-saver.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-action.h"
#include "../src/llama-kv-cache-release.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL: %s\n", msg); \
        failures++; \
    } \
} while(0)

// ===========================================================================
// Synthetic model fixture — same construction as test-kv-paged-release-bounded.
// ===========================================================================

static void synthetic_set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);
    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) tmp[i] = dis(gen);
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) tmp[i] = ggml_fp32_to_fp16(dis(gen));
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal: unsupported tensor type in synthetic model fixture");
    }
}

static bool synthetic_silent_load(float, void *) { return true; }

static gguf_context_ptr make_synthetic_llama_gguf_ctx() {
    gguf_context_ptr ret(gguf_init_empty());
    const llm_arch arch = LLM_ARCH_LLAMA;
    llama_model_saver ms(arch, ret.get());
    const uint32_t n_ctx   = 256;
    const uint32_t n_vocab = 128;
    const uint32_t n_embd  = 128;
    const uint32_t n_head  = 2;
    const uint32_t n_ff    = 192;
    const uint32_t n_layer = 2;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,      n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,               1.0f);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,     n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,  n_head);
    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS,  8.0f);
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,       1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,         1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,     1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,         1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,      uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,          uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,        n_ctx/8);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    uint32_t(1));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    uint32_t(64));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,         uint32_t(8));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");
    return ret;
}

static llama_model * make_synthetic_model() {
    gguf_context_ptr gguf = make_synthetic_llama_gguf_ctx();
    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = synthetic_silent_load;
    static std::vector<ggml_backend_dev_t> devs = { nullptr };
    mparams.devices = devs.data();
    size_t seed = 1234;
    return llama_model_init_from_user(gguf.get(), synthetic_set_tensor_data, &seed, mparams);
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

static int decode_prompt(llama_context * lctx, const std::vector<llama_token> & tokens) {
    std::vector<llama_token> toks = tokens;
    llama_batch b = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    return llama_decode(lctx, b);
}

// ===========================================================================
// BV0..BV9 — Physical budget view reconciliation and state-cycle tests.
// ===========================================================================

int main(int /*argc*/, char ** /*argv*/) {
    setenv("LLAMA_KV_PAGED", "1", 1);
    // LLAMA_KV_PAGED_RELEASE is intentionally never set in this test.
    // The budget view has zero dependency on the Cleanup-C2 legacy apply
    // path; the test stays green regardless of whether the env var is
    // "0", unset, or removed entirely by Cleanup-C2.
    setenv("LLAMA_KV_PAGED_BLOCK_SIZE", "16", 1);
    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);

    llama_backend_init();

    llama_model * model = make_synthetic_model();
    if (!model) {
        std::fprintf(stderr, "FAIL: synthetic model init failed\n");
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 256;
    cparams.n_seq_max = 2;
    cparams.kv_unified = true;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    // -------------------------------------------------------------------------
    // BV0: Structural precondition — before any context, no valid snapshot.
    // -------------------------------------------------------------------------
    {
        const llama_kv_physical_budget_view empty;
        CHECK(!empty.valid && empty.total_bytes == 0 && empty.resident_bytes == 0 &&
                empty.swapped_authoritative_bytes == 0 && empty.n_blocks == 0 &&
                !empty.swapped_metadata_consistent &&
                empty.n_owned_blocks == 0 && empty.n_shared_blocks == 0,
                "BV0: default-constructed snapshot is invalid with zero fields");
        std::fprintf(stderr, "BV0 default snapshot: valid=%d OK\n", empty.valid ? 1 : 0);
    }

    // -------------------------------------------------------------------------
    // BV1: After decode, identity/byte reconciliation with sample_kv_resident().
    // -------------------------------------------------------------------------
    uint64_t view_after_decode_object_id = 0;
    uint64_t view_after_decode_generation = 0;
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV1: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 30);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV1: decode ok");

            const auto view = g.kv->sample_kv_physical_budget_view();
            const auto resident = g.kv->sample_kv_resident();
            CHECK(view.valid, "BV1: view marked valid after decode");
            CHECK(view.object_id != 0 && view.generation != 0,
                    "BV1: object_id and generation bound to current KV object");
            CHECK(view.n_blocks > 0, "BV1: paged layout exposes non-zero block count");
            CHECK(view.total_bytes > 0,
                    "BV1: total_bytes derived from layer row sizes × kv_size");
            if (resident.available) {
                CHECK(view.resident_available && view.resident_bytes == resident.resident_bytes,
                        "BV1: resident_bytes reconciles with sample_kv_resident()");
            } else {
                CHECK(!view.resident_available && view.resident_bytes == 0,
                        "BV1: resident_bytes zero when sampler unavailable");
            }
            CHECK(view.swapped_block_count == 0 && view.swapped_authoritative_bytes == 0,
                    "BV1: no SWAPPED blocks after fresh decode");
            CHECK(view.released_block_count == 0,
                    "BV1: zero RELEASED blocks after fresh decode");
            CHECK(view.transient_staging_bound_bytes == 0,
                    "BV1: K2 staging bound zero when K2 pipeline disabled");
            CHECK(view.swapped_metadata_consistent,
                    "BV1: swapped_metadata_consistent true on fresh layout (no SWAPPED)");
            CHECK(view.n_owned_blocks > 0,
                    "BV1: post-decode blocks are owned by at least one live sequence");

            view_after_decode_object_id = view.object_id;
            view_after_decode_generation = view.generation;

            std::fprintf(stderr, "BV1 post-decode: obj=%llu gen=%llu resident=%llu total=%llu blocks=%u owned=%u shared=%u OK\n",
                    (unsigned long long) view.object_id,
                    (unsigned long long) view.generation,
                    (unsigned long long) view.resident_bytes,
                    (unsigned long long) view.total_bytes,
                    view.n_blocks,
                    view.n_owned_blocks,
                    view.n_shared_blocks);
        }
    }

    // -------------------------------------------------------------------------
    // BV2: After seq_rm (no owners), dead_resident_reclaimable_bytes
    //      reconciles with sample_kv_release_budget().
    // -------------------------------------------------------------------------
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV2: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 31);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV2: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const auto view = g.kv->sample_kv_physical_budget_view();
            const auto rb = g.kv->sample_kv_release_budget();
            CHECK(view.valid, "BV2: view valid after seq_rm");
            if (rb.valid && view.reclaimable_available) {
                CHECK(view.dead_resident_reclaimable_bytes == rb.reclaimable_resident_bytes,
                        "BV2: dead_resident_reclaimable_bytes reconciles with sample_kv_release_budget()");
            }
            CHECK(view.swapped_block_count == 0 && view.swapped_authoritative_bytes == 0,
                    "BV2: idle path has no SWAPPED authoritative bytes");
            CHECK(view.object_id != view_after_decode_object_id ||
                    view.generation != view_after_decode_generation,
                    "BV2: object identity distinct across context reopens");
            CHECK(view.n_owned_blocks == 0,
                    "BV2: after seq_rm with no surviving sequences, no owned blocks");
            std::fprintf(stderr, "BV2 idle: reclaim=%llu/%llu owned=%u shared=%u OK\n",
                    (unsigned long long) view.dead_resident_reclaimable_bytes,
                    (unsigned long long) rb.reclaimable_resident_bytes,
                    view.n_owned_blocks,
                    view.n_shared_blocks);
        }
    }

    // -------------------------------------------------------------------------
    // BV2A: explicit-only swap keeps read-only dead-resident observation
    // available even though destructive bounded-release capability is false.
    // This is the V2 P0 gate-closure regression test.
    // -------------------------------------------------------------------------
    {
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        setenv("LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY", "1", 1);
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV2A: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 35);
            CHECK(decode_prompt(g.ctx, prompt) == 0, "BV2A: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const auto release_cap = g.kv->bounded_release_can_enable();
            const auto rb = g.kv->sample_kv_release_budget();
            const auto view = g.kv->sample_kv_physical_budget_view();
            CHECK(!release_cap,
                    "BV2A: destructive bounded release remains disabled under explicit-only swap");
            CHECK(rb.valid && rb.reclaimable_resident_bytes > 0,
                    "BV2A: read-only reclaimable sampler remains valid under explicit-only swap");
            CHECK(view.valid && view.reclaimable_available &&
                    view.dead_resident_reclaimable_bytes == rb.reclaimable_resident_bytes,
                    "BV2A: physical budget view exposes dead resident bytes under explicit-only swap");
        }
        unsetenv("LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY");
        setenv("LLAMA_KV_PAGED_SWAP", "0", 1);
    }

    // -------------------------------------------------------------------------
    // BV2B: invalid layout and invalid swap metadata fail closed.
    // -------------------------------------------------------------------------
    {
        setenv("LLAMA_KV_PAGED_BLOCK_SIZE", "3", 1);
        ContextGuard invalid_layout;
        if (!invalid_layout.init(model, cparams)) {
            CHECK(false, "BV2B: invalid-layout context creation failed");
        } else {
            const auto view = invalid_layout.kv->sample_kv_physical_budget_view();
            CHECK(!view.valid && !view.reclaimable_available,
                    "BV2B: invalid paged layout returns unavailable view");
        }
        setenv("LLAMA_KV_PAGED_BLOCK_SIZE", "16", 1);

        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        ContextGuard invalid_metadata;
        if (!invalid_metadata.init(model, cparams)) {
            CHECK(false, "BV2B: invalid-metadata context creation failed");
        } else {
            CHECK(decode_prompt(invalid_metadata.ctx, std::vector<llama_token>(16, 36)) == 0,
                    "BV2B: metadata setup decode ok");
            const auto offload = invalid_metadata.kv->execute_action({
                llama_kv_action::offload, 7351, 0, UINT64_MAX, 1, false });
            CHECK(offload.state_changed, "BV2B: metadata setup offload ok");
            invalid_metadata.kv->paged_budget_view_test_corrupt_swap_metadata();
            const auto view = invalid_metadata.kv->sample_kv_physical_budget_view();
            CHECK(!view.valid && !view.swapped_metadata_consistent,
                    "BV2B: swap metadata drift invalidates physical budget view");
        }
        setenv("LLAMA_KV_PAGED_SWAP", "0", 1);
    }

    // -------------------------------------------------------------------------
    // BV3: OFFLOAD → SWAPPED — swapped_authoritative_bytes exactly equals
    //      offload.bytes on a clean single-block case (no other SWAPPED
    //      blocks from a prior cycle).
    // -------------------------------------------------------------------------
    {
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV3: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 32);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV3: decode ok");

            const auto before = g.kv->sample_kv_physical_budget_view();
            CHECK(before.valid && before.resident_block_count > 0,
                    "BV3: pre-offload has at least one RESIDENT block");
            CHECK(before.swapped_block_count == 0,
                    "BV3: pre-offload has zero SWAPPED blocks (clean state)");

            const auto offload = g.kv->execute_action({
                llama_kv_action::offload, 7301, 0, UINT64_MAX, 1, false });
            CHECK(offload.state_changed && offload.blocks == 1,
                    "BV3: unified offload swaps exactly one block (clean case)");

            const auto after = g.kv->sample_kv_physical_budget_view();
            CHECK(after.valid, "BV3: post-offload view valid");
            CHECK(after.swapped_metadata_consistent,
                    "BV3: SWAPPED metadata consistent after offload");
            CHECK(after.swapped_block_count == 1,
                    "BV3: exactly one SWAPPED block after single-block offload");
            CHECK(after.swapped_authoritative_bytes == offload.bytes,
                    "BV3: swapped_authoritative_bytes == offload.bytes (precise reconciliation)");
            if (after.resident_available && before.resident_available) {
                CHECK(after.resident_bytes <= before.resident_bytes,
                        "BV3: resident_bytes does not grow when blocks move to SWAPPED");
            }
            std::fprintf(stderr, "BV3 offload->SWAPPED: swapped=%u swapped_bytes=%llu offload_bytes=%llu OK\n",
                    after.swapped_block_count,
                    (unsigned long long) after.swapped_authoritative_bytes,
                    (unsigned long long) offload.bytes);
        }
        setenv("LLAMA_KV_PAGED_SWAP", "0", 1);
    }

    // -------------------------------------------------------------------------
    // BV4: PREFETCH → RESIDENT — swap bookkeeping zeroes, resident recovers.
    // -------------------------------------------------------------------------
    {
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV4: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 33);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV4: decode ok");

            const auto offload = g.kv->execute_action({
                llama_kv_action::offload, 7401, 0, UINT64_MAX, 1, false });
            CHECK(offload.state_changed, "BV4: setup offload succeeded");

            const auto mid = g.kv->sample_kv_physical_budget_view();
            CHECK(mid.swapped_block_count > 0 && mid.swapped_authoritative_bytes > 0,
                    "BV4: pre-prefetch has SWAPPED authoritative bytes");

            const auto prefetch = g.kv->execute_action({
                llama_kv_action::prefetch, 7402, 0, 0, 1, true });
            CHECK(prefetch.outcome == llama_kv_action_outcome::completed && prefetch.state_changed,
                    "BV4: prefetch restored at least one block");

            const auto after = g.kv->sample_kv_physical_budget_view();
            CHECK(after.valid, "BV4: post-prefetch view valid");
            CHECK(after.swapped_metadata_consistent,
                    "BV4: SWAPPED metadata consistent (zero SWAPPED blocks left)");
            CHECK(after.swapped_block_count == 0,
                    "BV4: swapped_block_count == 0 after full prefetch");
            CHECK(after.swapped_authoritative_bytes == 0,
                    "BV4: swapped_authoritative_bytes == 0 after full prefetch (closure)");
            if (after.resident_available) {
                CHECK(after.resident_bytes >= mid.resident_bytes,
                        "BV4: resident_bytes non-decreasing after restore");
            }
            std::fprintf(stderr, "BV4 prefetch->RESIDENT: swapped=0 res=%llu OK\n",
                    (unsigned long long) after.resident_bytes);
        }
        setenv("LLAMA_KV_PAGED_SWAP", "0", 1);
    }

    // -------------------------------------------------------------------------
    // BV5: Unified RELEASE + clear() lifecycle (no LLAMA_KV_PAGED_RELEASE apply
    //      path; survives Cleanup-C2 deletion of legacy gate).  Uses only
    //      execute_action({release}) and llama_memory_clear to drive state
    //      transitions — never sets LLAMA_KV_PAGED_RELEASE=1.
    // -------------------------------------------------------------------------
    {
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV5: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 34);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV5: decode ok");
            // Drive a block to SWAPPED through the unified offload path.
            const auto offload = g.kv->execute_action({
                llama_kv_action::offload, 7501, 0, UINT64_MAX, 1, false });
            CHECK(offload.state_changed, "BV5: setup offload ok");

            const auto before = g.kv->sample_kv_physical_budget_view();
            CHECK(before.swapped_authoritative_bytes > 0,
                    "BV5: pre-release has SWAPPED authoritative bytes");

            // Drop seq ownership so the block becomes eligible for destructive
            // release through the unified RELEASE action.  This uses the
            // current unified RELEASE gate, not the legacy LLAMA_KV_PAGED_RELEASE
            // apply path.
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            // Unified RELEASE: only affects RESIDENT/UNUSED unowned blocks.
            // SWAPPED blocks are not in the release state-machine's eligible
            // set, so the metadata remains intact after a unified RELEASE
            // call — the budget view must report the same authoritative bytes.
            const auto release_action = g.kv->execute_action({
                llama_kv_action::release, 7502, 0, UINT64_MAX, g.kv->paged_release_bounded_test_read_block_size(), false });
            CHECK(release_action.state_changed,
                    "BV5: unified RELEASE returns state_changed on eligible dead blocks");
            const auto view_after_release = g.kv->sample_kv_physical_budget_view();
            CHECK(view_after_release.swapped_authoritative_bytes == before.swapped_authoritative_bytes,
                    "BV5: unified RELEASE leaves SWAPPED authoritative bytes intact (state gate)");

            // Now reset the cache so SWAPPED metadata is cleared; the unified
            // clear() is the documented lifecycle boundary.
            llama_memory_clear(g.mem, true);
            const auto after_clear = g.kv->sample_kv_physical_budget_view();
            CHECK(after_clear.swapped_block_count == 0,
                    "BV5: cleared cache has no SWAPPED blocks");
            CHECK(after_clear.swapped_authoritative_bytes == 0,
                    "BV5: cleared cache has zero swapped_authoritative_bytes");
            CHECK(after_clear.swapped_metadata_consistent,
                    "BV5: post-clear metadata consistent (no SWAPPED blocks left)");
            // object_id is a per-instance identity — preserved by clear()
            // until the KV cache object itself is destroyed.
            CHECK(after_clear.object_id == before.object_id,
                    "BV5: clear() preserves object_id (same cache instance)");
            CHECK(after_clear.generation != view_after_release.generation,
                    "BV5: clear() bumps generation");
            std::fprintf(stderr, "BV5 unified lifecycle: pre=%llu post_clear=%llu gen=%llu->%llu OK\n",
                    (unsigned long long) before.swapped_authoritative_bytes,
                    (unsigned long long) after_clear.swapped_authoritative_bytes,
                    (unsigned long long) view_after_release.generation,
                    (unsigned long long) after_clear.generation);
        }
        setenv("LLAMA_KV_PAGED_SWAP", "0", 1);
    }

    // -------------------------------------------------------------------------
    // BV6: Per-state block counters reflect real state-machine output.
    // -------------------------------------------------------------------------
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV6: context creation failed");
        } else {
            const auto view_empty = g.kv->sample_kv_physical_budget_view();
            CHECK(view_empty.valid, "BV6: empty-cache view valid");
            CHECK(view_empty.unused_block_count == view_empty.n_blocks,
                    "BV6: empty cache reports all blocks UNUSED");

            std::vector<llama_token> prompt(16, 35);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV6: decode ok");
            const auto view_after = g.kv->sample_kv_physical_budget_view();
            CHECK(view_after.resident_block_count > 0,
                    "BV6: at least one RESIDENT after decode");
            CHECK(view_after.unused_block_count + view_after.resident_block_count +
                    view_after.released_block_count + view_after.swapped_block_count +
                    view_after.pending_write_block_count + view_after.invalid_block_count
                    == view_after.n_blocks,
                    "BV6: per-state counts sum to n_blocks");
            std::fprintf(stderr, "BV6 state counters: r=%u s=%u re=%u pw=%u in=%u un=%u total=%u OK\n",
                    view_after.resident_block_count,
                    view_after.swapped_block_count,
                    view_after.released_block_count,
                    view_after.pending_write_block_count,
                    view_after.invalid_block_count,
                    view_after.unused_block_count,
                    view_after.n_blocks);
        }
    }

    // -------------------------------------------------------------------------
    // BV7: object_id stable across clear() (cache-instance id); generation bumps.
    // -------------------------------------------------------------------------
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV7: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 36);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV7: decode ok");

            const auto v1 = g.kv->sample_kv_physical_budget_view();
            const auto v2 = g.kv->sample_kv_physical_budget_view();
            CHECK(v1.object_id == v2.object_id && v1.generation == v2.generation,
                    "BV7: identity stable across back-to-back snapshots");
            CHECK(v1.object_id != 0 && v1.generation != 0,
                    "BV7: identity non-zero while context is alive");

            llama_memory_clear(g.mem, true);
            const auto v3 = g.kv->sample_kv_physical_budget_view();
            // object_id is bound to the cache instance, not the
            // generation.  clear() bumps generation but keeps object_id.
            CHECK(v3.object_id == v1.object_id,
                    "BV7: clear() preserves object_id (same cache instance)");
            CHECK(v3.generation != v1.generation,
                    "BV7: clear() bumps generation");
            std::fprintf(stderr, "BV7 identity: id=%llu gen=%llu->%llu post_clear_id=%llu OK\n",
                    (unsigned long long) v1.object_id,
                    (unsigned long long) v1.generation,
                    (unsigned long long) v3.generation,
                    (unsigned long long) v3.object_id);
        }
    }

    // -------------------------------------------------------------------------
    // BV8a: K2 disabled — staging bound is zero regardless of any prior
    //       clear() state.
    // -------------------------------------------------------------------------
    {
        unsetenv("LLAMA_KV_PAGED_RESTORE_K2");
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV8a: K2-disabled context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 37);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV8a: decode ok");
            const auto v = g.kv->sample_kv_physical_budget_view();
            CHECK(!g.kv->paged_unified_action_test_k2_enabled(),
                    "BV8a: K2 disabled by default in test env");
            CHECK(v.transient_staging_bound_bytes == 0,
                    "BV8a: K2 staging bound is zero when K2 is disabled");
            std::fprintf(stderr, "BV8a staging bound (K2 disabled): %llu OK\n",
                    (unsigned long long) v.transient_staging_bound_bytes);
        }
    }

    // -------------------------------------------------------------------------
    // BV8b: K2 enabled — after a swap-offload + prefetch cycle, the budget
    //       view's transient_staging_bound_bytes equals the authoritative
    //       paged_restore_k2_staging_bound_bytes (read via test seam).
    // -------------------------------------------------------------------------
    {
        setenv("LLAMA_KV_PAGED_RESTORE_K2", "1", 1);
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV8b: K2-enabled context creation failed");
        } else {
            CHECK(g.kv->paged_unified_action_test_k2_enabled(),
                    "BV8b: K2 enabled via LLAMA_KV_PAGED_RESTORE_K2=1");

            std::vector<llama_token> prompt(48, 7);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV8b: decode ok");

            const auto before = g.kv->sample_kv_physical_budget_view();
            CHECK(before.transient_staging_bound_bytes == 0,
                    "BV8b: K2 staging bound zero before any restore pipeline runs");

            const auto offload = g.kv->execute_action({
                llama_kv_action::offload, 7601, 0, UINT64_MAX, 3, false });
            CHECK(offload.state_changed && offload.blocks >= 1,
                    "BV8b: offload succeeds with K2 enabled");

            // Mirror test-kv-paged-release-bounded WT30b: align the group
            // cap with the block size so the bound collapses to
            // 2 * block_bytes.  This makes the precise equality check
            // deterministic.
            const std::vector<uint8_t> block_bytes_v = g.kv->paged_unified_action_test_read_block_bytes(0);
            const uint64_t block_bytes = block_bytes_v.size();
            g.kv->paged_unified_action_test_set_restore_group_byte_cap(block_bytes);

            const auto prefetch = g.kv->execute_action({
                llama_kv_action::prefetch, 7602, 0, 0, 0, true, true });
            CHECK(prefetch.outcome == llama_kv_action_outcome::completed && prefetch.blocks >= 1,
                    "BV8b: K2 prefetch restored at least one block");

            const uint64_t seam_bound = g.kv->paged_unified_action_test_read_k2_staging_bound_bytes();
            const auto after = g.kv->sample_kv_physical_budget_view();
            CHECK(seam_bound > 0,
                    "BV8b: K2 staging bound non-zero after restore pipeline ran");
            CHECK(after.transient_staging_bound_bytes == seam_bound,
                    "BV8b: view.transient_staging_bound_bytes == K2 authoritative staging bound");
            std::fprintf(stderr, "BV8b K2 staging: seam=%llu view=%llu OK\n",
                    (unsigned long long) seam_bound,
                    (unsigned long long) after.transient_staging_bound_bytes);
        }
        unsetenv("LLAMA_KV_PAGED_RESTORE_K2");
        setenv("LLAMA_KV_PAGED_SWAP", "0", 1);
    }

    // -------------------------------------------------------------------------
    // BV9a: Shared ownership via in-range seq_cp(0,1).  Fresh context,
    //      cparams.n_seq_max=2, decode → seq_cp(0,1,-1,-1) → re-sample.
    //      Expect n_owned_blocks unchanged, n_shared_blocks == n_owned_blocks.
    // -------------------------------------------------------------------------
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV9a: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 38);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV9a: decode ok");

            const auto v_pre = g.kv->sample_kv_physical_budget_view();
            CHECK(v_pre.valid && v_pre.n_owned_blocks > 0 && v_pre.n_shared_blocks == 0,
                    "BV9a: pre-CP — owned > 0, shared == 0");

            llama_memory_seq_cp(g.mem, 0, 1, -1, -1);
            const auto v_a = g.kv->sample_kv_physical_budget_view();
            CHECK(v_a.valid, "BV9a: view valid after in-range seq_cp");
            CHECK(v_a.n_owned_blocks == v_pre.n_owned_blocks,
                    "BV9a: n_owned_blocks unchanged by seq_cp (CP adds owner, doesn't create new block)");
            CHECK(v_a.n_shared_blocks == v_pre.n_owned_blocks,
                    "BV9a: n_shared_blocks == n_owned_blocks after seq_cp(0,1)");
            CHECK(v_a.n_shared_blocks > 0,
                    "BV9a: shared ownership observed in-range");
            std::fprintf(stderr,
                    "BV9a in-range seq_cp(0,1): owned=%u shared=%u OK\n",
                    v_a.n_owned_blocks, v_a.n_shared_blocks);
        }
    }

    // -------------------------------------------------------------------------
    // BV9b: Shared ownership via out-of-range seq_cp(0,7).  Independent
    //      fresh context — must NOT inherit any state from BV9a.  With
    //      cparams.n_seq_max=2, seq_id=7 is outside the configured
    //      concurrent-sequence count but still a legal seq_id value
    //      (LLAMA_MAX_SEQ=256).  The only seq_cp applied in this context
    //      is (0,7,-1,-1); we then assert n_shared_blocks > 0 and equals
    //      n_owned_blocks.  If the ownership collector ever regresses to
    //      scanning only up to n_seq_max=2, seq 7 is silently dropped and
    //      n_shared_blocks stays at 0 — this case fails the assertion.
    // -------------------------------------------------------------------------
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "BV9b: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 39);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "BV9b: decode ok");

            const auto v_pre = g.kv->sample_kv_physical_budget_view();
            CHECK(v_pre.valid && v_pre.n_owned_blocks > 0 && v_pre.n_shared_blocks == 0,
                    "BV9b: pre-CP — owned > 0, shared == 0");

            llama_memory_seq_cp(g.mem, 0, 7, -1, -1);
            const auto v_b = g.kv->sample_kv_physical_budget_view();
            CHECK(v_b.valid, "BV9b: view valid after out-of-range seq_cp");
            CHECK(v_b.n_owned_blocks == v_pre.n_owned_blocks,
                    "BV9b: n_owned_blocks unchanged by seq_cp(0,7)");
            CHECK(v_b.n_shared_blocks == v_pre.n_owned_blocks,
                    "BV9b: n_shared_blocks == n_owned_blocks (seq 7 owner detected, no authority truncation)");
            CHECK(v_b.n_shared_blocks > 0,
                    "BV9b: shared ownership observed for out-of-range seq_id (LLAMA_MAX_SEQ authority)");
            std::fprintf(stderr,
                    "BV9b out-of-range seq_cp(0,7): owned=%u shared=%u OK\n",
                    v_b.n_owned_blocks, v_b.n_shared_blocks);
        }
    }

    llama_model_free(model);
    llama_backend_free();

    if (failures > 0) {
        std::fprintf(stderr, "FAIL: %d check(s) failed\n", failures);
        return 1;
    }
    std::fprintf(stderr, "PASS: KV-Budget-V1 physical budget view reconciliation\n");
    return 0;
}
