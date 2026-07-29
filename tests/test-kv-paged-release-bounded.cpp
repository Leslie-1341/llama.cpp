// Bounded KV release correctness test — validates the Stage 3A-2C
// release → reuse → commit / rollback lifecycle and the read/write/ensure
// safety predicates WITHOUT an on-disk GGUF model.
//
// Model fixture: a minimal in-memory LLM_ARCH_LLAMA model is built at runtime
// via llama_model_init_from_user (same construction pattern proven by
// tests/test-llama-archs.cpp).  No LLAMACPP_TEST_MODELFILE or argv model path
// is required; the test always runs (never CTest SKIP).
//
// Part A (no-model): synthetic ownership-collection ABORT detection.
// Part B (synthetic-model): real bounded-release gates via the genuine graph
//   compute (llama_decode) path, so PENDING_WRITE → RESIDENT (commit) and
//   PENDING_WRITE → RELEASED (rollback) are driven by the authoritative
//   state-machine triggers, not by overrides or stubs:
//   WT1-WT4: ensure_write_resident / read_resident / state-gate fail-closed
//            behaviors against RELEASED, fresh/stale PENDING_WRITE, SWAPPED.
//   WT6:     idle release → reuse decode → commit → RESIDENT
//   WT7:     undersized prompt: stale/padding rows trigger dummy candidate +
//            released_redirect_no_dummy == 0 + input_setup_fatal == 0.
//   WT8:     graph success: reuse_allocations >=1, write_commits >=1,
//            write_rollbacks == 0, block RESIDENT, free-list / pending-map /
//            transaction_open consistent.
//   WT9:     deterministic pre-graph failure after a real write transaction
//            opens → rollback → block back to RELEASED, pending bitmap cleared,
//            transaction_open false, free-list consistent.

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
#include "../src/llama-kv-cache-release.h"

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
// Synthetic in-memory LLM_ARCH_LLAMA model fixture (no GGUF file).
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

// Minimal LLM_ARCH_LLAMA gguf context.  Mirrors the llama subset of
// tests/test-llama-archs::get_gguf_ctx.
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
    // llama_model_init_from_user synchronously consumes the metadata needed to
    // construct the model, so the local gguf context can retain normal RAII.
    return llama_model_init_from_user(gguf.get(), synthetic_set_tensor_data, &seed, mparams);
}

// ===========================================================================
// Helpers
// ===========================================================================

static int decode_prompt(llama_context * lctx, const std::vector<llama_token> & tokens) {
    std::vector<llama_token> toks = tokens;  // mutable copy — llama_batch_get_one needs non-const
    llama_batch b = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    return llama_decode(lctx, b);
}

static llama_ubatch make_ubatch(
        const std::vector<llama_token> & tokens,
        const std::vector<llama_pos> & positions,
        llama_seq_id seq_id) {
    CHECK(tokens.size() == positions.size(), "make_ubatch: token/position size match");

    llama_ubatch ubatch = {};
    ubatch.data = std::make_shared<llama_ubatch::data_t>();
    auto & data = *ubatch.data;
    data.token = tokens;
    data.pos = positions;
    data.n_seq_id.assign(tokens.size(), 1);
    data.seq_id_data.assign(tokens.size(), seq_id);
    data.seq_id.resize(tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        data.seq_id[i] = &data.seq_id_data[i];
    }
    data.seq_id_unq = { seq_id };
    data.seq_idx.assign(LLAMA_MAX_SEQ, -1);
    data.seq_idx[seq_id] = 0;
    data.output.assign(tokens.size(), 1);

    ubatch.b_equal_seqs = 1;
    ubatch.n_tokens = tokens.size();
    ubatch.n_seq_tokens = tokens.size();
    ubatch.n_seqs = 1;
    ubatch.n_seqs_unq = 1;
    ubatch.n_pos = 1;
    ubatch.token = data.token.data();
    ubatch.pos = data.pos.data();
    ubatch.n_seq_id = data.n_seq_id.data();
    ubatch.seq_id = data.seq_id.data();
    ubatch.seq_id_unq = data.seq_id_unq.data();
    ubatch.seq_idx = data.seq_idx.data();
    ubatch.output = data.output.data();
    return ubatch;
}

static llama_kv_cache::slot_info make_slot(std::vector<uint32_t> cells) {
    llama_kv_cache::slot_info sinfo = {};
    sinfo.s0 = 0;
    sinfo.s1 = 0;
    sinfo.strm = { 0 };
    sinfo.idxs = { std::move(cells) };
    return sinfo;
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

// ===========================================================================
// Part B: real-model (synthetic) bounded-release lifecycle tests.
// ===========================================================================

int main(int /*argc*/, char ** /*argv*/) {
    test_ownership_fault_fixture();

    setenv("LLAMA_KV_PAGED", "1", 1);
    setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);
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
    cparams.n_seq_max = 2;          // one unified stream, two sequence ids for rollback coverage
    cparams.kv_unified = true;
    cparams.type_k = GGML_TYPE_F32;
    cparams.type_v = GGML_TYPE_F32;
    cparams.n_threads = 4;
    cparams.n_threads_batch = 4;

    // =========================================================================
    // Setup sanity: synthetic model drives the paged-KV block-state path.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "SETUP: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 1);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "SETUP: decode ok");
            uint8_t s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "SETUP: block 0 RESIDENT after decode");
            const auto owned_budget = g.kv->sample_kv_release_budget();
            CHECK(owned_budget.valid, "SETUP: release budget snapshot valid");
            CHECK(owned_budget.resident_bytes > 0, "SETUP: resident bytes observed");
            CHECK(owned_budget.reclaimable_resident_bytes == 0,
                    "SETUP: active-owned block is not reclaimable");
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            const auto idle_budget = g.kv->sample_kv_release_budget();
            CHECK(idle_budget.valid, "SETUP: idle release budget snapshot valid");
            CHECK(idle_budget.reclaimable_resident_bytes > 0,
                    "SETUP: unowned resident block is reclaimable");
            CHECK(idle_budget.reclaimable_resident_bytes <= idle_budget.resident_bytes,
                    "SETUP: reclaimable resident bytes bounded by resident bytes");
            std::fprintf(stderr, "SETUP synthetic-model paged: rc=%d block0=%u OK\n", rc, s0);
        }
    }

    // =========================================================================
    // WT0: ownership collection failure aborts before any destructive state
    //   mutation.  This uses the legacy test seam only to force the collector's
    //   ABORT branch; WT6-WT9 exercise the production bounded-only primitive.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT0: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 1);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT0: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint8_t state_before =
                g.kv->paged_release_bounded_test_read_block_state(0);
            const bool free_before =
                g.kv->paged_release_bounded_test_block_in_free_list(0);
            const uint64_t released_before =
                g.kv->paged_release_bounded_test_read_released_blocks();

            g.kv->paged_release_bounded_test_force_ownership_abort = true;
            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);

            CHECK(r.ownership_aborted, "WT0: ownership failure reported");
            CHECK(r.released_blocks == 0, "WT0: zero blocks released after ownership abort");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == state_before,
                    "WT0: block state unchanged after ownership abort");
            CHECK(g.kv->paged_release_bounded_test_block_in_free_list(0) == free_before,
                    "WT0: free-list unchanged after ownership abort");
            CHECK(g.kv->paged_release_bounded_test_read_released_blocks() == released_before,
                    "WT0: release counter unchanged after ownership abort");

            std::fprintf(stderr, "WT0 ownership abort: state=%u free=%d released=%llu OK\n",
                    state_before, free_before ? 1 : 0,
                    (unsigned long long)released_before);
        }
    }

    // =========================================================================
    // WT1: ensure_write_resident accepts fresh current-transaction PENDING_WRITE
    //   cells; bounded release skips PENDING_WRITE blocks (state gate preserves
    //   them) without touching real state.  Uses block_state_override to mark a
    //   dead block as PENDING_WRITE and verifies the state gate skip.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT1: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 1);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT1: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint32_t tb = 0;
            uint8_t real_state = g.kv->paged_release_bounded_test_read_block_state(tb);
            CHECK(real_state == 1, "WT1: block 0 is RESIDENT");

            g.kv->paged_release_bounded_test_block_state_override.block = tb;
            g.kv->paged_release_bounded_test_block_state_override.state = 4;

            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
            CHECK(!r.ownership_aborted, "WT1: ownership valid");
            CHECK(r.blocks_skipped_state >= 1,
                    "WT1: PENDING_WRITE block skipped by state gate");
            CHECK(r.blocks_skipped_owned == 0,
                    "WT1: zero blocks skipped by owned gate (block is dead)");
            CHECK(r.madvise_failures == 0,
                    "WT1: no madvise failures (never touched PENDING_WRITE block)");

            uint8_t real_after = g.kv->paged_release_bounded_test_read_block_state(tb);
            CHECK(real_after == real_state,
                    "WT1: real block state unchanged after PENDING_WRITE skip");
            CHECK(g.kv->paged_release_bounded_test_block_state_override.block == UINT32_MAX,
                    "WT1: override auto-reset");

            std::fprintf(stderr, "WT1 PENDING_WRITE bypass: state=%u->%u skipped_state=%" PRIu32 " OK\n",
                    real_state, real_after, r.blocks_skipped_state);
        }
    }

    // =========================================================================
    // WT2: graph success keeps block RESIDENT and owned by the active seq, so
    //   bounded release must skip it (no release while owned).
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT2: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 2);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT2: decode ok");

            uint8_t s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "WT2: block 0 is RESIDENT after decode");

            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
            CHECK(!r.ownership_aborted, "WT2: ownership valid");
            CHECK(r.released_blocks == 0,
                    "WT2: zero released (all blocks owned by active seq)");
            CHECK(r.blocks_skipped_owned > 0,
                    "WT2: blocks skipped by owned gate (active seq owns them)");

            s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "WT2: block 0 stays RESIDENT after skipped release");

            std::fprintf(stderr, "WT2 active-owned protection: released=%" PRIu32
                    " skipped_owned=%" PRIu32 " s0=%u OK\n",
                    r.released_blocks, r.blocks_skipped_owned, s0);
        }
    }

    // =========================================================================
    // WT3: active-visible RELEASED remains fail-closed.  This test exercises
    //   the legacy force-active seam only; deterministic transaction rollback
    //   is covered independently by WT9.
    // =========================================================================
    {
        setenv("LLAMA_KV_TEST_MODE", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ", "0", 1);

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT3: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 3);
            int rc = decode_prompt(g.ctx, prompt);

            const uint64_t triggers =
                g.kv->paged_release_bounded_test_read_force_active_triggers();
            CHECK(triggers > 0, "WT3: force-active-release path triggered");
            CHECK(rc != 0, "WT3: decode failed after force-active-release");
            std::fprintf(stderr, "WT3 active-visible RELEASED: triggers=%" PRIu64
                    " decode_rc=%d OK\n", triggers, rc);
        }

        unsetenv("LLAMA_KV_TEST_MODE");
        unsetenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE");
        unsetenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ");
        setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);
    }

    // =========================================================================
    // WT4: authoritative predicates reject PENDING_WRITE without an explicit
    //   transaction owner, even if its pending bitmap is set, plus stale PENDING_WRITE,
    //   active-visible RELEASED, and SWAPPED without backing.  The release
    //   state gate also skips PENDING_WRITE without changing real state.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT4: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 4);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT4: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint32_t tb = 0;
            uint8_t real_state = g.kv->paged_release_bounded_test_read_block_state(tb);
            CHECK(real_state == 1, "WT4: block 0 is RESIDENT");

            const uint64_t rejected_before =
                g.kv->paged_release_bounded_test_read_ensure_pending_write_rejected();
            CHECK(!g.kv->paged_release_bounded_test_probe_residency(
                        0, 4, true, false, false),
                    "WT4: pending bitmap without transaction owner rejected");
            CHECK(!g.kv->paged_release_bounded_test_probe_residency(
                        0, 4, false, false, false),
                    "WT4: stale/non-current PENDING_WRITE write rejected");
            CHECK(!g.kv->paged_release_bounded_test_probe_residency(
                        0, 4, false, true, true),
                    "WT4: stale/non-current PENDING_WRITE active read rejected");
            CHECK(!g.kv->paged_release_bounded_test_probe_residency(
                        0, 2, false, true, true),
                    "WT4: active-visible RELEASED read rejected");
            CHECK(!g.kv->paged_release_bounded_test_probe_residency(
                        0, 3, false, false, false),
                    "WT4: SWAPPED write without backing rejected");
            const uint64_t rejected_after =
                g.kv->paged_release_bounded_test_read_ensure_pending_write_rejected();
            CHECK(rejected_after >= rejected_before + 3,
                    "WT4: pending-owner and stale write/read rejections counted");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(tb) == real_state,
                    "WT4: predicate probes restore real block state");

            g.kv->paged_release_bounded_test_block_state_override.block = tb;
            g.kv->paged_release_bounded_test_block_state_override.state = 4;

            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
            CHECK(!r.ownership_aborted, "WT4: ownership valid");
            CHECK(r.blocks_skipped_state >= 1,
                    "WT4: non-current PENDING_WRITE block skipped by state gate");

            uint8_t real_after = g.kv->paged_release_bounded_test_read_block_state(tb);
            CHECK(real_after == real_state, "WT4: real block state unchanged");
            CHECK(g.kv->paged_release_bounded_test_block_state_override.block == UINT32_MAX,
                    "WT4: override auto-reset");

            std::fprintf(stderr, "WT4 non-current PENDING_WRITE: state=%u->%u"
                    " skipped_state=%" PRIu32 " OK\n",
                    real_state, real_after, r.blocks_skipped_state);
        }
    }

    // =========================================================================
    // WT5: active-visible RELEASED without backing fails (regression guard).
    // =========================================================================
    {
        setenv("LLAMA_KV_TEST_MODE", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ", "0", 1);

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT5: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 5);
            int rc = decode_prompt(g.ctx, prompt);

            const uint64_t triggers =
                g.kv->paged_release_bounded_test_read_force_active_triggers();
            CHECK(triggers > 0, "WT5: force-active-release path triggered");
            CHECK(rc != 0, "WT5: decode failed after active-visible RELEASED violation");
            std::fprintf(stderr, "WT5 active-visible RELEASED: triggers=%" PRIu64
                    " decode_rc=%d OK\n", triggers, rc);
        }

        unsetenv("LLAMA_KV_TEST_MODE");
        unsetenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE");
        unsetenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ");
        setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);
    }

    // WT6-WT9 are bounded-only: the legacy destructive-release switch is off,
    // while the structurally gated server primitive remains available.
    setenv("LLAMA_KV_PAGED_RELEASE", "0", 1);

    // =========================================================================
    // WT6: idle release → reuse → commit → RESIDENT (real lifecycle).
    //   Real assertions on every AC point — no log-only fake-greens.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT6: context creation failed");
        } else {
            CHECK(!g.kv->paged_release_bounded_test_legacy_release_enabled(),
                    "WT6: legacy release disabled");
            CHECK(g.kv->bounded_release_can_enable(),
                    "WT6: bounded release structurally enabled");
            std::vector<llama_token> prompt(16, 6);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT6: first decode ok");
            uint8_t s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "WT6: block 0 RESIDENT after first decode");

            // Clear all seq refs so block becomes dead.
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint64_t reuse_before =
                g.kv->paged_release_bounded_test_read_reuse_allocations();
            const uint64_t commits_before =
                g.kv->paged_release_bounded_test_read_write_commits();
            const uint64_t rollbacks_before =
                g.kv->paged_release_bounded_test_read_write_rollbacks();

            // Real destructive bounded release of block 0.
            const auto r1 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r1.ownership_aborted, "WT6: release ownership valid");
            CHECK(r1.released_blocks >= 1, "WT6: at least one real block released");

            s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 2, "WT6: block 0 RELEASED after release");

            // Reuse the released block with a new decode (PENDING_WRITE → commit).
            std::vector<llama_token> prompt2(16, 7);
            rc = decode_prompt(g.ctx, prompt2);
            CHECK(rc == 0, "WT6: reuse decode ok");

            const uint64_t reuse_after =
                g.kv->paged_release_bounded_test_read_reuse_allocations();
            const uint64_t commits_after =
                g.kv->paged_release_bounded_test_read_write_commits();
            const uint64_t rollbacks_after =
                g.kv->paged_release_bounded_test_read_write_rollbacks();

            // Real assertions (AC #3,4,6,8).
            CHECK(reuse_after > reuse_before,
                    "WT6: reuse_allocations >= 1 (real reuse of released block)");
            CHECK(commits_after > commits_before,
                    "WT6: write_commits >= 1 after successful graph");
            CHECK(rollbacks_after == rollbacks_before,
                    "WT6: write_rollbacks == 0 after successful graph");
            s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "WT6: block 0 RESIDENT after reuse commit");
            // AC #8 consistency: no open transaction after commit, pending map empty.
            CHECK(!g.kv->paged_release_bounded_test_transaction_open(),
                    "WT6: transaction closed after commit");
            CHECK(g.kv->paged_release_bounded_test_pending_write_blocks_count() == 0,
                    "WT6: pending write blocks cleared after commit");

            std::fprintf(stderr, "WT6 lifecycle: reuse %llu->%llu commits %llu->%llu"
                    " rollbacks %llu->%llu block0=%u OK\n",
                    (unsigned long long)reuse_before, (unsigned long long)reuse_after,
                    (unsigned long long)commits_before, (unsigned long long)commits_after,
                    (unsigned long long)rollbacks_before, (unsigned long long)rollbacks_after,
                    s0);
        }
    }

    // =========================================================================
    // WT7: prompt未占满block — stale/padding rows trigger the dummy candidate
    //   path and keep redirection fail-closed (no_dummy == 0, input_setup_fatal
    //   == 0).  A fresh PENDING_WRITE cell of the current transaction is the
    //   transaction-local dummy; stale/padding rows must be redirected without
    //   raising INPUT_SETUP_FAILURE.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT7: context creation failed");
        } else {
            CHECK(!g.kv->paged_release_bounded_test_legacy_release_enabled(),
                    "WT7: legacy release disabled");
            CHECK(g.kv->bounded_release_can_enable(),
                    "WT7: bounded release structurally enabled");
            const uint64_t dummy_pw_before =
                g.kv->paged_release_bounded_test_read_dummy_candidate_pending_write_cell();
            const uint64_t no_dummy_before =
                g.kv->paged_release_bounded_test_read_released_redirect_no_dummy();
            const uint64_t no_dummy_pw_before =
                g.kv->paged_release_bounded_test_read_released_redirect_no_dummy_pending_write();
            const uint64_t setup_fatal_before =
                g.kv->paged_release_bounded_test_read_input_setup_fatal();

            // First a full block so the second (undersized) decode reuses it.
            std::vector<llama_token> prompt(16, 8);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT7: full prompt decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            const auto r1 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r1.ownership_aborted, "WT7: release ownership valid after seq_rm");
            uint8_t s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 2, "WT7: block 0 RELEASED after release");

            // Undersized reuse: only 4 cells are fresh; the remaining 12 cells
            // in the block are stale/padding.  Row-index setup must redirect the
            // stale rows to a transaction-local dummy candidate (fresh
            // PENDING_WRITE cell) without firing INPUT_SETUP_FAILURE.
            std::vector<llama_token> prompt2(4, 9);
            rc = decode_prompt(g.ctx, prompt2);
            CHECK(rc == 0, "WT7: undersized reuse decode ok");

            const uint64_t dummy_pw_after =
                g.kv->paged_release_bounded_test_read_dummy_candidate_pending_write_cell();
            const uint64_t no_dummy_after =
                g.kv->paged_release_bounded_test_read_released_redirect_no_dummy();
            const uint64_t no_dummy_pw_after =
                g.kv->paged_release_bounded_test_read_released_redirect_no_dummy_pending_write();
            const uint64_t setup_fatal_after =
                g.kv->paged_release_bounded_test_read_input_setup_fatal();

            // AC: not fully filled => stale/padding rows drove the dummy-candidate
            // path (the block had 16 cells; only 4 became fresh, so 12 are stale).
            CHECK(dummy_pw_after > dummy_pw_before,
                    "WT7: dummy_candidate_pending_write_cell >= 1 (stale/padding rows)");
            CHECK(no_dummy_after == no_dummy_before,
                    "WT7: released_redirect_no_dummy == 0");
            CHECK(no_dummy_pw_after == no_dummy_pw_before,
                    "WT7: released_redirect_no_dummy_pending_write == 0");
            CHECK(setup_fatal_after == setup_fatal_before,
                    "WT7: input_setup_fatal == 0 (no setup failure)");

            s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "WT7: block 0 RESIDENT after reuse decode");

            std::fprintf(stderr, "WT7 undersized reuse: dummy_pw %llu->%llu"
                    " no_dummy %llu->%llu no_dummy_pw %llu->%llu setup_fatal %llu->%llu OK\n",
                    (unsigned long long)dummy_pw_before, (unsigned long long)dummy_pw_after,
                    (unsigned long long)no_dummy_before, (unsigned long long)no_dummy_after,
                    (unsigned long long)no_dummy_pw_before, (unsigned long long)no_dummy_pw_after,
                    (unsigned long long)setup_fatal_before, (unsigned long long)setup_fatal_after);
        }
    }

    // =========================================================================
    // WT8: graph success → state-gate skips a PENDING_WRITE block, AND a full
    //   successful reuse cycle confirms reuse_allocations >= 1, write_commits
    //   >= 1, write_rollbacks == 0, block RESIDENT, free-list / pending-map /
    //   transaction_open consistent.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT8: context creation failed");
        } else {
            CHECK(!g.kv->paged_release_bounded_test_legacy_release_enabled(),
                    "WT8: legacy release disabled");
            CHECK(g.kv->bounded_release_can_enable(),
                    "WT8: bounded release structurally enabled");
            // Real active-owned protection first (re-asserts WT2 invariant).
            std::vector<llama_token> prompt(16, 10);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT8: first decode ok");

            const uint64_t reuse_before =
                g.kv->paged_release_bounded_test_read_reuse_allocations();
            const uint64_t commits_before =
                g.kv->paged_release_bounded_test_read_write_commits();
            const uint64_t rollbacks_before =
                g.kv->paged_release_bounded_test_read_write_rollbacks();

            // Release after clearing ownership.
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            const auto r1 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r1.ownership_aborted, "WT8: release ownership valid");
            CHECK(r1.released_blocks >= 1, "WT8: real block released");
            uint8_t s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 2, "WT8: block 0 RELEASED");

            // Successful reuse decode → commit.
            std::vector<llama_token> prompt2(16, 11);
            rc = decode_prompt(g.ctx, prompt2);
            CHECK(rc == 0, "WT8: reuse decode ok");

            const uint64_t reuse_after =
                g.kv->paged_release_bounded_test_read_reuse_allocations();
            const uint64_t commits_after =
                g.kv->paged_release_bounded_test_read_write_commits();
            const uint64_t rollbacks_after =
                g.kv->paged_release_bounded_test_read_write_rollbacks();

            // Real assertions (AC #3,4,6,8).
            CHECK(reuse_after > reuse_before, "WT8: reuse_allocations >= 1");
            CHECK(commits_after > commits_before, "WT8: write_commits >= 1");
            CHECK(rollbacks_after == rollbacks_before, "WT8: write_rollbacks == 0");
            s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "WT8: block 0 RESIDENT after commit");
            // AC #8 consistency (free-list + ownership + transaction_open).
            CHECK(!g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT8: block 0 removed from free list after commit");
            CHECK(!g.kv->paged_release_bounded_test_transaction_open(),
                    "WT8: transaction closed after commit");
            CHECK(g.kv->paged_release_bounded_test_pending_write_blocks_count() == 0,
                    "WT8: pending write blocks cleared");

            std::fprintf(stderr, "WT8 graph success: reuse %llu->%llu commits %llu->%llu"
                    " rollbacks %llu->%llu block0=%u freelist=0 txn_open=0 OK\n",
                    (unsigned long long)reuse_before, (unsigned long long)reuse_after,
                    (unsigned long long)commits_before, (unsigned long long)commits_after,
                    (unsigned long long)rollbacks_before, (unsigned long long)rollbacks_after,
                    s0);
        }
    }

    // =========================================================================
    // WT9: exact metadata journal without RELEASED reuse. Overwriting a cell
    //   owned by a foreign sequence at an older incoming position purges another
    //   foreign cell; rollback restores both cells, old positions, and head.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT9: context creation failed");
        } else {
            const auto old_slot = make_slot({ 0, 1 });
            const auto old_ubatch = make_ubatch({ 20, 21 }, { 100, 120 }, 1);
            {
                llama_kv_cache_context old_ctx(g.kv, { old_slot }, { old_ubatch });
                CHECK(old_ctx.apply(), "WT9: seed transaction applies");
                CHECK(old_ctx.finish_paged_kv_write(llama_paged_kv_write_action::COMMIT),
                        "WT9: seed transaction commits");
            }

            const uint32_t head_before = g.kv->paged_release_bounded_test_read_head(0);
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 1,
                    "WT9: no RELEASED reuse; block is RESIDENT");
            CHECK(g.kv->paged_release_bounded_test_cell_has_seq(0, 0, 1),
                    "WT9: foreign cell 0 seeded");
            CHECK(g.kv->paged_release_bounded_test_cell_has_seq(0, 1, 1),
                    "WT9: selected foreign cell seeded");

            const auto incoming_slot = make_slot({ 1 });
            const auto incoming_ubatch = make_ubatch({ 22 }, { 5 }, 0);
            llama_kv_cache_context incoming_ctx(g.kv, { incoming_slot }, { incoming_ubatch });
            CHECK(incoming_ctx.apply(), "WT9: incoming transaction applies");
            CHECK(!g.kv->paged_release_bounded_test_cell_has_seq(0, 0, 1),
                    "WT9: foreign seq_rm indirect purge triggered");
            CHECK(g.kv->paged_release_bounded_test_cell_has_seq(0, 1, 0),
                    "WT9: selected cell overwritten before rollback");
            CHECK(incoming_ctx.finish_paged_kv_write(
                        llama_paged_kv_write_action::ROLLBACK_PRE_COMPUTE),
                    "WT9: pre-compute rollback succeeds");

            CHECK(g.kv->paged_release_bounded_test_cell_has_seq(0, 0, 1),
                    "WT9: indirect-purge cell restored");
            CHECK(g.kv->paged_release_bounded_test_cell_has_seq(0, 1, 1),
                    "WT9: selected foreign cell restored");
            CHECK(g.kv->paged_release_bounded_test_read_cell_pos(0, 0) == 100,
                    "WT9: foreign position 100 restored");
            CHECK(g.kv->paged_release_bounded_test_read_cell_pos(0, 1) == 120,
                    "WT9: old position greater than incoming restored");
            CHECK(g.kv->paged_release_bounded_test_read_head(0) == head_before,
                    "WT9: stream head restored");
            CHECK(!g.kv->paged_release_bounded_test_transaction_open(),
                    "WT9: transaction closed");
            CHECK(g.kv->paged_release_bounded_test_pending_write_blocks_count() == 0,
                    "WT9: no pending ownership remains");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 1 &&
                    g.kv->paged_release_bounded_test_block_used(0) &&
                    !g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT9: resident allocation metadata restored consistently");
            std::fprintf(stderr, "WT9 exact metadata journal rollback: OK\n");
        }
    }

    // =========================================================================
    // WT10: graph allocation lifecycle failure before compute. The transaction
    //   has no RELEASED reuse, rolls back completely, and the same context retries.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT10: context creation failed");
        } else {
            const uint64_t triggers_before =
                g.kv->paged_release_bounded_test_read_fail_graph_alloc_triggers();
            g.kv->paged_release_bounded_test_arm_fail_graph_alloc();
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 30)) != 0,
                    "WT10: graph allocation failure propagates");
            CHECK(g.kv->paged_release_bounded_test_read_fail_graph_alloc_triggers() ==
                    triggers_before + 1,
                    "WT10: graph allocation seam triggered exactly once");
            CHECK(g.kv->paged_release_bounded_test_error_reason() ==
                    llama_paged_swap_error_reason::PAGED_WRITE_GRAPH_ALLOC_FAILURE,
                    "WT10: exact graph allocation error reason");
            CHECK(g.kv->seq_pos_max(0) == -1,
                    "WT10: failed ubatch metadata fully rolled back");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 0 &&
                    !g.kv->paged_release_bounded_test_block_used(0) &&
                    g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT10: UNUSED block allocation rolled back");
            CHECK(!g.kv->paged_release_bounded_test_transaction_open() &&
                    g.kv->paged_release_bounded_test_pending_write_blocks_count() == 0,
                    "WT10: transaction and pending ownership closed");
            CHECK(!g.kv->paged_release_bounded_test_context_invalid(),
                    "WT10: pre-compute failure does not poison context");
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 31)) == 0,
                    "WT10: same context safely retries after rollback");
            std::fprintf(stderr, "WT10 graph alloc failure rollback: OK\n");
        }
    }

    // =========================================================================
    // WT11: the real graph computes, then a deterministic failure is reported.
    //   Metadata is not rolled back over potentially-written K/V bytes; write
    //   blocks and context are invalid until explicit memory clear/reset.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT11: context creation failed");
        } else {
            const uint64_t triggers_before =
                g.kv->paged_release_bounded_test_read_fail_after_compute_triggers();
            g.kv->paged_release_bounded_test_arm_fail_after_compute();
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 40)) != 0,
                    "WT11: post-compute failure propagates");
            CHECK(g.kv->paged_release_bounded_test_read_fail_after_compute_triggers() ==
                    triggers_before + 1,
                    "WT11: post-compute seam triggered exactly once");
            CHECK(g.kv->paged_release_bounded_test_error_reason() ==
                    llama_paged_swap_error_reason::PAGED_WRITE_COMPUTE_FAILURE,
                    "WT11: exact compute failure reason");
            CHECK(g.kv->paged_release_bounded_test_context_invalid(),
                    "WT11: context poisoned after compute-started failure");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 5 &&
                    g.kv->paged_release_bounded_test_block_used(0) &&
                    !g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT11: write block quarantined INVALID");
            CHECK(g.kv->seq_pos_max(0) == 15,
                    "WT11: metadata retained instead of unsafe rollback");
            CHECK(!g.kv->paged_release_bounded_test_transaction_open() &&
                    g.kv->paged_release_bounded_test_pending_write_blocks_count() == 0,
                    "WT11: transaction closed with no pending orphan");

            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(1, 41)) != 0,
                    "WT11: next decode rejected while poisoned");
            CHECK(g.kv->paged_release_bounded_test_error_reason() ==
                    llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID,
                    "WT11: next decode reports context invalid");

            llama_memory_clear(g.mem, true);
            CHECK(!g.kv->paged_release_bounded_test_context_invalid(),
                    "WT11: explicit reset clears poison");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 0 &&
                    !g.kv->paged_release_bounded_test_block_used(0) &&
                    g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT11: reset restores UNUSED/free allocation state");
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 42)) == 0,
                    "WT11: decode recovers after explicit reset");
            std::fprintf(stderr, "WT11 compute-started fail-stop + reset: OK\n");
        }
    }

    // =========================================================================
    // WT12: rollback discard failure never leaves a PENDING_WRITE orphan. The
    //   failed block becomes INVALID, next decode is rejected, and reset recovers.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT12: context creation failed");
        } else {
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 50)) == 0,
                    "WT12: seed decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            CHECK(g.kv->bounded_release(UINT64_MAX, UINT32_MAX).released_blocks >= 1,
                    "WT12: seed block released");

            const uint64_t alloc_before =
                g.kv->paged_release_bounded_test_read_fail_graph_alloc_triggers();
            const uint64_t discard_before =
                g.kv->paged_release_bounded_test_read_fail_rollback_madvise_triggers();
            g.kv->paged_release_bounded_test_arm_fail_graph_alloc();
            g.kv->paged_release_bounded_test_arm_fail_rollback_madvise();
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(15, 51)) != 0,
                    "WT12: rollback discard failure propagates");
            CHECK(g.kv->paged_release_bounded_test_read_fail_graph_alloc_triggers() == alloc_before + 1,
                    "WT12: graph allocation seam triggered");
            CHECK(g.kv->paged_release_bounded_test_read_fail_rollback_madvise_triggers() == discard_before + 1,
                    "WT12: rollback discard seam triggered");
            CHECK(g.kv->paged_release_bounded_test_error_reason() ==
                    llama_paged_swap_error_reason::PAGED_WRITE_ROLLBACK_DISCARD_FAILURE,
                    "WT12: exact rollback discard error reason");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 5 &&
                    g.kv->paged_release_bounded_test_block_used(0) &&
                    !g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT12: failed block quarantined INVALID");
            CHECK(!g.kv->paged_release_bounded_test_pending_write_cell_set(0) &&
                    g.kv->paged_release_bounded_test_pending_write_blocks_count() == 0 &&
                    !g.kv->paged_release_bounded_test_transaction_open(),
                    "WT12: no pending ownership or open transaction orphan");
            CHECK(g.kv->paged_release_bounded_test_block_owned_cells(0) == 0,
                    "WT12: metadata rolled back despite discard failure");
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(1, 52)) != 0,
                    "WT12: next decode rejected after discard failure");
            CHECK(g.kv->paged_release_bounded_test_error_reason() ==
                    llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID,
                    "WT12: next decode reports context invalid");

            llama_memory_clear(g.mem, true);
            CHECK(!g.kv->paged_release_bounded_test_context_invalid(),
                    "WT12: reset clears discard poison");
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 53)) == 0,
                    "WT12: decode recovers after reset");
            std::fprintf(stderr, "WT12 rollback discard failure quarantine: OK\n");
        }
    }

    // =========================================================================
    // WT13: multi-block rollback cleanup is partial but consistent: one injected
    //   discard failure quarantines its block while the other block completes
    //   rollback to RELEASED/free; all pending ownership is closed.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT13: context creation failed");
        } else {
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(32, 60)) == 0,
                    "WT13: two-block seed decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            CHECK(g.kv->bounded_release(UINT64_MAX, UINT32_MAX).released_blocks >= 2,
                    "WT13: two blocks released");

            g.kv->paged_release_bounded_test_arm_fail_graph_alloc();
            g.kv->paged_release_bounded_test_arm_fail_rollback_madvise();
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(31, 61)) != 0,
                    "WT13: multi-block rollback failure propagates");
            std::fprintf(stderr,
                    "WT13 states: b0=%u used0=%d free0=%d b1=%u used1=%d free1=%d pending=%llu\n",
                    g.kv->paged_release_bounded_test_read_block_state(0),
                    g.kv->paged_release_bounded_test_block_used(0),
                    g.kv->paged_release_bounded_test_block_in_free_list(0),
                    g.kv->paged_release_bounded_test_read_block_state(1),
                    g.kv->paged_release_bounded_test_block_used(1),
                    g.kv->paged_release_bounded_test_block_in_free_list(1),
                    (unsigned long long) g.kv->paged_release_bounded_test_pending_write_blocks_count());
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 5,
                    "WT13: first cleanup failure block INVALID");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(1) == 2,
                    "WT13: second block completes rollback to RELEASED");
            CHECK(g.kv->paged_release_bounded_test_block_used(0) &&
                    !g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT13: INVALID block quarantined");
            CHECK(!g.kv->paged_release_bounded_test_block_used(1) &&
                    g.kv->paged_release_bounded_test_block_in_free_list(1),
                    "WT13: successful cleanup block returned to free list");
            CHECK(g.kv->paged_release_bounded_test_block_owned_cells(0) == 0 &&
                    g.kv->paged_release_bounded_test_block_owned_cells(1) == 0,
                    "WT13: metadata rollback covers both blocks");
            CHECK(g.kv->paged_release_bounded_test_pending_write_blocks_count() == 0 &&
                    !g.kv->paged_release_bounded_test_transaction_open(),
                    "WT13: multi-block pending ownership fully closed");
            CHECK(g.kv->paged_release_bounded_test_context_invalid(),
                    "WT13: partial cleanup failure poisons context");
            std::fprintf(stderr, "WT13 multi-block partial cleanup: OK\n");
        }
    }

    // =========================================================================
    // WT14: default non-paged KV must not claim paged failure handling. A
    // compute-started failure therefore reaches llama_context::decode's original
    // outer seq_rm cleanup, and the next decode starts from a clean sequence.
    // =========================================================================
    {
        unsetenv("LLAMA_KV_PAGED");
        unsetenv("LLAMA_KV_PAGED_RELEASE");

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT14: non-paged context creation failed");
        } else {
            const uint64_t triggers_before =
                g.kv->paged_release_bounded_test_read_fail_after_compute_triggers();
            g.kv->paged_release_bounded_test_arm_fail_after_compute();
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 70)) != 0,
                    "WT14: non-paged compute failure propagates");
            CHECK(g.kv->paged_release_bounded_test_read_fail_after_compute_triggers() ==
                    triggers_before + 1,
                    "WT14: compute failure seam triggered exactly once");
            CHECK(g.kv->seq_pos_max(0) == -1,
                    "WT14: outer cleanup removed failed ubatch metadata");
            CHECK(!g.kv->paged_release_bounded_test_context_invalid(),
                    "WT14: non-paged failure does not invent paged poison");
            CHECK(decode_prompt(g.ctx, std::vector<llama_token>(16, 71)) == 0,
                    "WT14: next non-paged decode succeeds from clean state");
            std::fprintf(stderr, "WT14 non-paged failure isolation: OK\n");
        }
    }

    // =========================================================================
    // WT15: unsupported wrapper distinction — bounded_release_can_enable_diagnose()
    //   decomposes every structural precondition.  Each negative case maps to a
    //   specific field so server policy can attribute the skip reason precisely
    //   ("unsupported" wrapper vs "supported-but-no-candidate").
    // =========================================================================
    {
        // Create a context without paged KV so bounded_release_can_enable is false.
        unsetenv("LLAMA_KV_PAGED");
        cparams.kv_unified = false;

        ContextGuard g;
        if (g.init(model, cparams)) {
            // Non-paged KV: can_enable must be false.
            CHECK(!g.kv->bounded_release_can_enable(),
                    "WT15: non-paged can_enable is false");

            const auto cap = g.kv->bounded_release_can_enable_diagnose();
            CHECK(!cap.can_enable,
                    "WT15: diagnose can_enable agrees with can_enable()");
            CHECK(!cap.paged,
                    "WT15: cap_paged is false (not paged)");
            CHECK(cap.ingraph == false,
                    "WT15: cap_ingraph is false (no paged → no ingraph)");
            CHECK(cap.layers_supported == false,
                    "WT15: cap_layers is false (no paged)");
            CHECK(cap.row_idx == false,
                    "WT15: cap_row_idx is false (no paged)");
            CHECK(!g.mem->sample_kv_release_budget().valid,
                    "WT15: unsupported wrapper reports invalid release budget");

            // Each field independently reported for attribution.
            std::fprintf(stderr, "WT15 unsupported wrapper diagnose: paged=%d"
                    " ingraph=%d layers=%d row_idx=%d swap_ok=%d layout=%d OK\n",
                    cap.paged ? 1 : 0, cap.ingraph ? 1 : 0,
                    cap.layers_supported ? 1 : 0, cap.row_idx ? 1 : 0,
                    cap.swap_disabled ? 1 : 0, cap.layout_supported ? 1 : 0);
        }

        // Restore paged env for remaining tests.
        setenv("LLAMA_KV_PAGED", "1", 1);
        setenv("LLAMA_KV_PAGED_RELEASE", "0", 1);
        cparams.kv_unified = true;
    }

    // =========================================================================
    // WT16: repeated rejection does not drift — multiple bounded_release calls
    //   on the same dead-block state produce identical result fields.
    //   Round 1 releases blocks → round 2 is idempotent (all blocks already
    //   RELEASED), not releasing additional blocks and reporting zero counters.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT16: context creation failed");
        } else {
            CHECK(!g.kv->paged_release_bounded_test_legacy_release_enabled(),
                    "WT16: legacy release disabled");
            CHECK(g.kv->bounded_release_can_enable(),
                    "WT16: bounded release structurally enabled");
            std::vector<llama_token> prompt(16, 80);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT16: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            // Round 1 — real release
            const auto r1 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r1.ownership_aborted, "WT16: round1 ownership valid");
            CHECK(r1.released_blocks >= 1, "WT16: round1 released some blocks");
            const auto released_budget = g.kv->sample_kv_release_budget();
            CHECK(released_budget.valid, "WT16: zero-candidate snapshot remains valid");
            CHECK(released_budget.reclaimable_resident_bytes == 0,
                    "WT16: RELEASED blocks are not reclaimable candidates");

            // Round 2 — all candidates already RELEASED, idempotent
            const auto r2 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r2.ownership_aborted, "WT16: round2 ownership valid");
            CHECK(r2.released_blocks == 0,
                    "WT16: round2 releases zero additional blocks (all already RELEASED)");
            CHECK(r2.released_bytes == 0,
                    "WT16: round2 releases zero additional bytes");
            CHECK(r2.blocks_skipped_state >= 1,
                    "WT16: round2 blocks skipped by state gate (RELEASED)");

            // Round 3 — still idempotent (no drift)
            const auto r3 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r3.ownership_aborted, "WT16: round3 ownership valid");
            CHECK(r3.released_blocks == 0,
                    "WT16: round3 still releases zero");
            CHECK(r3.blocks_skipped_state == r2.blocks_skipped_state,
                    "WT16: round3 skipped_state equals round2 (no drift)");

            std::fprintf(stderr, "WT16 repeated rejection no-drift: r1=%" PRIu32
                    " r2=%" PRIu32 " r3=%" PRIu32 " skip2=%" PRIu32
                    " skip3=%" PRIu32 " OK\n",
                    r1.released_blocks, r2.released_blocks, r3.released_blocks,
                    r2.blocks_skipped_state, r3.blocks_skipped_state);
        }
    }

    // =========================================================================
    // WT17: budget edge cases for bounded_release() (target_bytes=0 and
    //   max_scan_blocks=0) — verify the zero-side-effect fast-return contracts.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT17: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 90);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT17: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            // Capture pre-call counter values
            const uint64_t cnt_bytes_before =
                g.kv->bounded_release_counter_bytes();
            const uint64_t cnt_blocks_before =
                g.kv->bounded_release_counter_blocks();

            // target_bytes=0 → immediate return
            {
                const auto r = g.kv->bounded_release(0, 100);
                CHECK(r.released_bytes == 0, "WT17a: target=0 → released_bytes == 0");
                CHECK(r.released_blocks == 0, "WT17a: target=0 → released_blocks == 0");
                CHECK(r.blocks_scanned == 0, "WT17a: target=0 → blocks_scanned == 0");
                CHECK(!r.ownership_aborted, "WT17a: target=0 → ownership_aborted == false");
                CHECK(!r.block_scan_exhausted, "WT17a: target=0 → exhausted == false");
                CHECK(r.shortfall_bytes == 0, "WT17a: target=0 → shortfall == 0");
                CHECK(r.overshoot_bytes == 0, "WT17a: target=0 → overshoot == 0");
                CHECK(r.madvise_failures == 0, "WT17a: target=0 → madvise_failures == 0");
            }

            // max_scan_blocks=0 → exhausted + full shortfall
            {
                const uint64_t t = 65536;
                const auto r = g.kv->bounded_release(t, 0);
                CHECK(r.block_scan_exhausted, "WT17b: max_scan=0 → exhausted == true");
                CHECK(r.shortfall_bytes == t, "WT17b: max_scan=0 → shortfall == target");
                CHECK(r.released_bytes == 0, "WT17b: max_scan=0 → released_bytes == 0");
                CHECK(r.blocks_scanned == 0, "WT17b: max_scan=0 → blocks_scanned == 0");
                CHECK(!r.ownership_aborted, "WT17b: max_scan=0 → ownership_aborted == false");
                CHECK(r.madvise_failures == 0, "WT17b: max_scan=0 → madvise_failures == 0");
            }

            // Independent counters unchanged by both zero-work calls
            CHECK(g.kv->bounded_release_counter_bytes() == cnt_bytes_before,
                    "WT17: bounded counter bytes unchanged by zero-work calls");
            CHECK(g.kv->bounded_release_counter_blocks() == cnt_blocks_before,
                    "WT17: bounded counter blocks unchanged by zero-work calls");

            std::fprintf(stderr, "WT17 budget edge cases: target=0 max_scan=0 OK\n");
        }
    }

    // =========================================================================
    // WT18: full idempotent lifecycle — release → reuse → commit → release
    //   again, verifying the independent counters correctly accumulate across
    //   cycles and each cycle's release result is self-consistent.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT18: context creation failed");
        } else {
            const uint64_t cnt_bytes_before =
                g.kv->bounded_release_counter_bytes();
            const uint64_t cnt_blocks_before =
                g.kv->bounded_release_counter_blocks();

            // Cycle 1: allocate → release
            std::vector<llama_token> prompt1(16, 100);
            int rc = decode_prompt(g.ctx, prompt1);
            CHECK(rc == 0, "WT18: cycle1 decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            const auto r1 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r1.ownership_aborted, "WT18: cycle1 ownership valid");
            CHECK(r1.released_blocks >= 1, "WT18: cycle1 released blocks");

            // Cycle 2: reuse released block → commit → release again
            std::vector<llama_token> prompt2(16, 101);
            rc = decode_prompt(g.ctx, prompt2);
            CHECK(rc == 0, "WT18: cycle2 reuse decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            const auto r2 = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(!r2.ownership_aborted, "WT18: cycle2 ownership valid");
            CHECK(r2.released_blocks >= 1, "WT18: cycle2 released blocks again");

            // Independent counters accumulate across cycles
            const uint64_t cnt_bytes_after =
                g.kv->bounded_release_counter_bytes();
            const uint64_t cnt_blocks_after =
                g.kv->bounded_release_counter_blocks();
            CHECK(cnt_bytes_after >= cnt_bytes_before + r1.released_bytes + r2.released_bytes,
                    "WT18: bounded counter bytes accumulate across cycles");
            CHECK(cnt_blocks_after >= cnt_blocks_before + r1.released_blocks + r2.released_blocks,
                    "WT18: bounded counter blocks accumulate across cycles");

            std::fprintf(stderr, "WT18 idempotent cycle: r1=%" PRIu32
                    " r2=%" PRIu32 " cnt_bytes=%llu->%llu cnt_blocks=%llu->%llu OK\n",
                    r1.released_blocks, r2.released_blocks,
                    (unsigned long long)cnt_bytes_before, (unsigned long long)cnt_bytes_after,
                    (unsigned long long)cnt_blocks_before, (unsigned long long)cnt_blocks_after);
        }
    }

    // =========================================================================
    // WT19: Unbounded legacy paged_release_blocks() skips PENDING_WRITE blocks.
    //   Re-enable legacy release (was set to 0 for WT6-WT18), create a block,
    //   override its state to PENDING_WRITE via test seam, and verify the
    //   unbounded path skips it without madvise.
    // =========================================================================
    {
        setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT19: context creation failed");
        } else {
            CHECK(g.kv->paged_release_bounded_test_legacy_release_enabled(),
                    "WT19: legacy release enabled");

            std::vector<llama_token> prompt(16, 110);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT19: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint64_t released_before =
                g.kv->paged_release_bounded_test_read_released_blocks();
            const uint64_t unused_before =
                g.kv->paged_release_bounded_test_read_released_unused();
            const uint64_t dead_before =
                g.kv->paged_release_bounded_test_read_released_dead();

            // Override block 0 to PENDING_WRITE before calling legacy release
            g.kv->paged_release_bounded_test_block_state_override.block = 0;
            g.kv->paged_release_bounded_test_block_state_override.state = 4;

            // Call the UNBOUNDED legacy release directly
            g.kv->paged_release_blocks(1);

            // Legacy path should skip PENDING_WRITE — no state change, no counter
            // advance beyond what the override absorbs.
            const uint8_t s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == 1, "WT19: legacy release skipped PENDING_WRITE block (still RESIDENT)");
            CHECK(g.kv->paged_release_bounded_test_read_released_blocks() == released_before,
                    "WT19: legacy release counter unchanged (PENDING_WRITE skipped)");
            CHECK(g.kv->paged_release_bounded_test_read_released_unused() == unused_before,
                    "WT19: legacy unused counter unchanged");
            CHECK(g.kv->paged_release_bounded_test_read_released_dead() == dead_before,
                    "WT19: legacy dead counter unchanged");
            CHECK(!g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT19: PENDING_WRITE block not added to free list");
            CHECK(g.kv->paged_release_bounded_test_block_state_override.block == UINT32_MAX,
                    "WT19: override auto-reset after legacy release");

            std::fprintf(stderr, "WT19 legacy PENDING_WRITE gate: state=%u released=%llu OK\n",
                    s0, (unsigned long long)released_before);
        }

        setenv("LLAMA_KV_PAGED_RELEASE", "0", 1);
    }

    // =========================================================================
    // WT20: Unbounded legacy release skips INVALID (quarantined) blocks.
    // =========================================================================
    {
        setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT20: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 111);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT20: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint64_t released_before =
                g.kv->paged_release_bounded_test_read_released_blocks();
            const uint8_t real_state =
                g.kv->paged_release_bounded_test_read_block_state(0);

            // Override to INVALID
            g.kv->paged_release_bounded_test_block_state_override.block = 0;
            g.kv->paged_release_bounded_test_block_state_override.state = 5;

            g.kv->paged_release_blocks(1);

            const uint8_t s0 = g.kv->paged_release_bounded_test_read_block_state(0);
            CHECK(s0 == real_state, "WT20: legacy release skipped INVALID block (state unchanged)");
            CHECK(g.kv->paged_release_bounded_test_read_released_blocks() == released_before,
                    "WT20: legacy release counter unchanged (INVALID skipped)");
            CHECK(!g.kv->paged_release_bounded_test_block_in_free_list(0),
                    "WT20: INVALID block not added to free list");

            std::fprintf(stderr, "WT20 legacy INVALID gate: state=%u released=%llu OK\n",
                    s0, (unsigned long long)released_before);
        }

        setenv("LLAMA_KV_PAGED_RELEASE", "0", 1);
    }

    // =========================================================================
    // WT21: Shared bounded-release impl skips INVALID (quarantined) blocks.
    //   Uses the legacy-gate paged_release_blocks_bounded() (which passes
    //   use_test_seams=true to the shared impl) since the server-path
    //   bounded_release() is designed to never access test seams.
    //   Temporarily enables LLAMA_KV_PAGED_RELEASE=1 for the legacy gate.
    // =========================================================================
    {
        setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT21: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 112);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT21: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint64_t released_before =
                g.kv->paged_release_bounded_test_read_released_blocks();

            // Override to INVALID — shared impl must skip it via state gate
            g.kv->paged_release_bounded_test_block_state_override.block = 0;
            g.kv->paged_release_bounded_test_block_state_override.state = 5;

            const auto r = g.kv->paged_release_blocks_bounded(UINT64_MAX, UINT32_MAX);
            CHECK(!r.ownership_aborted, "WT21: ownership valid");
            CHECK(r.released_blocks == 0,
                    "WT21: shared impl released zero (INVALID skipped)");
            CHECK(r.blocks_skipped_state >= 1,
                    "WT21: shared impl state gate skipped INVALID");
            CHECK(r.madvise_failures == 0,
                    "WT21: zero madvise calls (INVALID never reached madvise)");

            CHECK(g.kv->paged_release_bounded_test_read_released_blocks() == released_before,
                    "WT21: legacy release counter unchanged");

            std::fprintf(stderr, "WT21 shared impl INVALID gate: released=%" PRIu32
                    " skipped_state=%" PRIu32 " OK\n",
                    r.released_blocks, r.blocks_skipped_state);
        }

        setenv("LLAMA_KV_PAGED_RELEASE", "0", 1);
    }

    // =========================================================================
    // WT22: Fail-stop context_invalid prevents bounded release.  After
    //   compute-started failure poisons the context, bounded_release() must
    //   return with ownership_aborted=true and zero state mutation.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT22: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 113);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT22: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            // Poison via compute-started failure seam
            g.kv->paged_release_bounded_test_arm_fail_after_compute();
            rc = decode_prompt(g.ctx, std::vector<llama_token>(16, 114));
            CHECK(rc != 0, "WT22: compute failure triggered");
            CHECK(g.kv->paged_release_bounded_test_context_invalid(),
                    "WT22: context poisoned after compute failure");
            const auto evaluation = g.kv->execute_action({
                llama_kv_action::evaluate, 6601, -1, 0, 0, false });
            CHECK(evaluation.capability.context_invalid && !evaluation.capability.can_release &&
                    !evaluation.state_changed,
                    "WT22: EVALUATE reports context invalid without mutation");

            const uint64_t cnt_bytes_before = g.kv->bounded_release_counter_bytes();
            const uint64_t cnt_blocks_before = g.kv->bounded_release_counter_blocks();

            // Bounded release must refuse under context_invalid
            const auto r = g.kv->bounded_release(UINT64_MAX, UINT32_MAX);
            CHECK(r.ownership_aborted,
                    "WT22: bounded release aborted (context invalid)");
            CHECK(r.released_blocks == 0,
                    "WT22: zero blocks released under fail-stop");
            CHECK(g.kv->bounded_release_counter_bytes() == cnt_bytes_before,
                    "WT22: bounded counter bytes unchanged");
            CHECK(g.kv->bounded_release_counter_blocks() == cnt_blocks_before,
                    "WT22: bounded counter blocks unchanged");

            // Reset recovers
            llama_memory_clear(g.mem, true);
            CHECK(!g.kv->paged_release_bounded_test_context_invalid(),
                    "WT22: reset clears context poison");

            std::fprintf(stderr, "WT22 fail-stop bounded release: aborted=%d OK\n",
                    r.ownership_aborted ? 1 : 0);
        }
    }

    // =========================================================================
    // WT23: Fail-stop context_invalid prevents unbounded legacy release.
    //   Same poison → reject → reset cycle as WT22, but through the legacy
    //   paged_release_blocks() path (LLAMA_KV_PAGED_RELEASE=1).
    // =========================================================================
    {
        setenv("LLAMA_KV_PAGED_RELEASE", "1", 1);

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT23: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 115);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT23: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            g.kv->paged_release_bounded_test_arm_fail_after_compute();
            rc = decode_prompt(g.ctx, std::vector<llama_token>(16, 116));
            CHECK(rc != 0, "WT23: compute failure triggered");
            CHECK(g.kv->paged_release_bounded_test_context_invalid(),
                    "WT23: context poisoned");

            const uint64_t released_before =
                g.kv->paged_release_bounded_test_read_released_blocks();
            const uint8_t state_before =
                g.kv->paged_release_bounded_test_read_block_state(0);

            g.kv->paged_release_blocks(1);

            CHECK(g.kv->paged_release_bounded_test_read_released_blocks() == released_before,
                    "WT23: legacy release counter unchanged under fail-stop");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == state_before,
                    "WT23: block state unchanged under fail-stop");

            llama_memory_clear(g.mem, true);
            CHECK(!g.kv->paged_release_bounded_test_context_invalid(),
                    "WT23: reset clears context poison");

            std::fprintf(stderr, "WT23 fail-stop legacy release: released=%llu state=%u OK\n",
                    (unsigned long long)released_before, state_before);
        }

        setenv("LLAMA_KV_PAGED_RELEASE", "0", 1);
    }

    // =========================================================================
    // WT24: Stage 3C-1B unified core action API.  EVALUATE is read-only and
    // unified RELEASE remains available independently of the legacy switch.
    // =========================================================================
    {
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT24: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 120);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT24: decode ok");
            llama_memory_seq_rm(g.mem, 0, -1, -1);

            const uint8_t state_before = g.kv->paged_release_bounded_test_read_block_state(0);
            const auto evaluation = g.kv->execute_action({
                llama_kv_action::evaluate, 7001, -1, 0, 0, false });
            CHECK(evaluation.decision_id == 7001, "WT24: EVALUATE preserves decision id");
            CHECK(evaluation.outcome == llama_kv_action_outcome::completed,
                    "WT24: EVALUATE completed");
            CHECK(!evaluation.state_changed && evaluation.core_transaction_id == 0,
                    "WT24: EVALUATE creates no transaction");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == state_before,
                    "WT24: EVALUATE leaves block state unchanged");
            CHECK(evaluation.capability.can_release,
                    "WT24: EVALUATE reports unified release capability");

            llama_kv_cache_context transaction_owner(LLAMA_MEMORY_STATUS_SUCCESS);
            CHECK(g.kv->paged_unified_action_test_begin_transaction(&transaction_owner),
                    "WT24: open test write transaction");
            const auto transaction_evaluation = g.kv->execute_action({
                llama_kv_action::evaluate, 7004, -1, 0, 0, false });
            CHECK(transaction_evaluation.capability.write_transaction_open &&
                    !transaction_evaluation.capability.can_release &&
                    !transaction_evaluation.state_changed,
                    "WT24: EVALUATE reports open transaction without mutation");
            llama_memory_clear(g.mem, true);

            const auto zero = g.kv->execute_action({
                llama_kv_action::release, 7002, -1, 0, 1, false });
            CHECK(zero.outcome == llama_kv_action_outcome::no_op &&
                    zero.reason == llama_kv_action_reason::zero_budget &&
                    !zero.state_changed && zero.core_transaction_id == 0,
                    "WT24: zero release budget is a no-op");

            const auto zero_scan = g.kv->execute_action({
                llama_kv_action::release, 7005, -1, 4096, 0, false });
            CHECK(zero_scan.outcome == llama_kv_action_outcome::no_op &&
                    zero_scan.reason == llama_kv_action_reason::scan_budget_exhausted &&
                    zero_scan.shortfall_bytes == 4096 && !zero_scan.state_changed,
                    "WT24: zero scan budget is an unchanged scan-budget shortfall");

            const auto no_swap_prefetch = g.kv->execute_action({
                llama_kv_action::prefetch, 7006, 0, 0, 0, true, true });
            CHECK(no_swap_prefetch.outcome == llama_kv_action_outcome::no_op &&
                    !no_swap_prefetch.io_failure && !no_swap_prefetch.fail_stop &&
                    no_swap_prefetch.shortfall_bytes == 0,
                    "WT24: all-required prefetch without swap is a safe no-op");

            const auto scan_limited = g.kv->execute_action({
                llama_kv_action::release, 7003, -1, UINT64_MAX, 1, false });
            CHECK(scan_limited.decision_id == 7003 && scan_limited.state_changed &&
                    scan_limited.core_transaction_id > 0 &&
                    scan_limited.reason == llama_kv_action_reason::scan_budget_exhausted &&
                    scan_limited.shortfall_bytes > 0,
                    "WT24: limited scan reports a scan-budget shortfall with one transaction");

            const auto release = g.kv->execute_action({
                llama_kv_action::release, 7007, -1, UINT64_MAX, UINT32_MAX, false });
            CHECK(release.decision_id == 7007 && release.state_changed &&
                    release.core_transaction_id > scan_limited.core_transaction_id,
                    "WT24: release correlates decision and core transaction");
            CHECK(release.blocks > 0 && release.bytes > 0 &&
                    release.reason == llama_kv_action_reason::target_shortfall &&
                    release.shortfall_bytes > 0,
                    "WT24: full scan reports target shortfall after releasing candidates");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 2,
                    "WT24: unified release changes the block to RELEASED");

            const auto no_candidate = g.kv->execute_action({
                llama_kv_action::release, 7008, -1, 4096, UINT32_MAX, false });
            CHECK(no_candidate.outcome == llama_kv_action_outcome::no_op &&
                    no_candidate.reason == llama_kv_action_reason::no_candidate &&
                    no_candidate.blocks == 0 && no_candidate.bytes == 0 &&
                    no_candidate.shortfall_bytes == 4096 && !no_candidate.state_changed &&
                    no_candidate.core_transaction_id == 0,
                    "WT24: complete scan reports no candidate without mutation");
            std::fprintf(stderr, "WT24 unified evaluate/release: tx=%llu blocks=%u OK\n",
                    (unsigned long long) release.core_transaction_id, release.blocks);
        }
    }

    // =========================================================================
    // WT25: unified offload/prefetch roundtrip and swap-enabled unified release.
    // =========================================================================
    {
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        setenv("LLAMA_KV_PAGED_RELEASE", "0", 1);

        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT25: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 121);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT25: decode ok");

            const auto evaluation = g.kv->execute_action({
                llama_kv_action::evaluate, 7101, -1, 0, 0, false });
            CHECK(evaluation.capability.can_release && evaluation.capability.can_offload &&
                    evaluation.capability.can_prefetch,
                    "WT25: swap-enabled EVALUATE exposes unified capabilities");

            const auto zero = g.kv->execute_action({
                llama_kv_action::offload, 7102, 0, 0, 0, false });
            CHECK(zero.outcome == llama_kv_action_outcome::no_op && !zero.state_changed,
                    "WT25: zero offload budget is a no-op");

            const auto kv_before_offload = g.kv->paged_unified_action_test_read_block_bytes(0);
            CHECK(!kv_before_offload.empty(), "WT25: captures populated KV tensor bytes before offload");
            const auto offload = g.kv->execute_action({
                llama_kv_action::offload, 7103, 0, UINT64_MAX, 1, false });
            CHECK(offload.outcome == llama_kv_action_outcome::completed && offload.state_changed &&
                    offload.blocks == 1 && offload.bytes > 0,
                    "WT25: unified offload swaps one exclusive resident block");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 3,
                    "WT25: offload publishes SWAPPED only after backing write");

            const auto prefetch = g.kv->execute_action({
                llama_kv_action::prefetch, 7104, 0, 0, 1, true });
            CHECK(prefetch.outcome == llama_kv_action_outcome::completed && prefetch.state_changed &&
                    prefetch.core_transaction_id > offload.core_transaction_id,
                    "WT25: offload to prefetch roundtrip creates ordered transactions");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 1,
                    "WT25: prefetch restores RESIDENT");
            const auto kv_after_prefetch = g.kv->paged_unified_action_test_read_block_bytes(0);
            CHECK(kv_after_prefetch == kv_before_offload,
                    "WT25: offload/prefetch restores byte-exact KV tensor data");

            g.kv->set_seq_prefetch_protected(0, true);
            const auto protected_offload = g.kv->execute_action({
                llama_kv_action::offload, 7105, 0, UINT64_MAX, 1, false });
            CHECK(protected_offload.outcome == llama_kv_action_outcome::rejected &&
                    protected_offload.reason == llama_kv_action_reason::protected_sequence &&
                    !protected_offload.state_changed,
                    "WT25: protected sequence is rejected for offload");
            g.kv->set_seq_prefetch_protected(0, false);
            g.kv->seq_cp(0, 1, -1, -1);
            const auto shared_offload = g.kv->execute_action({
                llama_kv_action::offload, 7106, 0, UINT64_MAX, 1, false });
            CHECK(shared_offload.outcome == llama_kv_action_outcome::rejected &&
                    shared_offload.reason == llama_kv_action_reason::shared_block &&
                    !shared_offload.state_changed,
                    "WT25: shared block is rejected for offload");

            g.kv->seq_rm(1, -1, -1);
            const auto second_offload = g.kv->execute_action({
                llama_kv_action::offload, 7107, 0, UINT64_MAX, 1, false });
            CHECK(second_offload.state_changed, "WT25: second offload succeeds");
            llama_memory_seq_rm(g.mem, 0, -1, -1);
            const auto release = g.kv->execute_action({
                llama_kv_action::release, 7108, -1, UINT64_MAX, UINT32_MAX, false });
            CHECK(release.decision_id == 7108,
                    "WT25: unified release returns the request correlation id");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 3,
                    "WT25: unified release skips SWAPPED state with paging enabled");
            std::fprintf(stderr, "WT25 unified swap roundtrip: offload_tx=%llu prefetch_tx=%llu OK\n",
                    (unsigned long long) offload.core_transaction_id,
                    (unsigned long long) prefetch.core_transaction_id);
        }
        unsetenv("LLAMA_KV_PAGED_SWAP");
    }

    // =========================================================================
    // WT26: I/O failures are structured action failures and never publish a
    // failed swap-out or turn a failed correctness-required prefetch into NOOP.
    // =========================================================================
    {
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE", "swap_out", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_KIND", "write_enospc_once", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID", "0", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE", "1", 1);
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT26a: context creation failed");
        } else {
            std::vector<llama_token> prompt(32, 122);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT26a: decode ok");
            const auto failed = g.kv->execute_action({
                llama_kv_action::offload, 7201, 0, UINT64_MAX, 1, false });
            CHECK(failed.outcome == llama_kv_action_outcome::failed && failed.io_failure &&
                    !failed.state_changed && failed.core_transaction_id == 0,
                    "WT26a: swap-out I/O failure is structured and has no transaction");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(1) == 1,
                    "WT26a: failed swap-out keeps target block RESIDENT");
        }
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_KIND");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE");

        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE", "prefetch_swap_in", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_KIND", "read_eof_once", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID", "0", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE", "1", 1);
        ContextGuard p;
        if (!p.init(model, cparams)) {
            CHECK(false, "WT26b: context creation failed");
        } else {
            std::vector<llama_token> prompt(32, 123);
            int rc = decode_prompt(p.ctx, prompt);
            CHECK(rc == 0, "WT26b: decode two blocks");
            const auto offload = p.kv->execute_action({
                llama_kv_action::offload, 7202, 0, UINT64_MAX, 1, false });
            CHECK(offload.state_changed && p.kv->paged_release_bounded_test_read_block_state(1) == 3,
                    "WT26b: unified offload prepares the later SWAPPED block");
            CHECK(p.kv->paged_unified_action_test_swap_out_block(0),
                    "WT26b: setup swaps the first block through the backing store");
            CHECK(p.kv->paged_release_bounded_test_read_block_state(0) == 3 &&
                    p.kv->paged_release_bounded_test_read_block_state(1) == 3,
                    "WT26b: setup has two SWAPPED blocks");
            const auto backing_before = p.kv->paged_unified_action_test_read_backing_stats();
            const auto fault_before = p.kv->paged_unified_action_test_read_io_fault();

            const auto failed = p.kv->execute_action({
                llama_kv_action::prefetch, 7203, 0, 0, 0, true, true });
            CHECK(failed.outcome == llama_kv_action_outcome::partial_failure && failed.io_failure &&
                    failed.fail_stop && failed.state_changed && failed.core_transaction_id > offload.core_transaction_id,
                    "WT26b: partial prefetch failure has one transaction and fail-stop");
            CHECK(failed.blocks == 1 && failed.bytes > 0 && failed.shortfall_bytes == failed.bytes,
                    "WT26b: partial prefetch reports completed bytes and exact shortfall");
            const auto backing_after = p.kv->paged_unified_action_test_read_backing_stats();
            const auto fault_after = p.kv->paged_unified_action_test_read_io_fault();
            CHECK(backing_after.read_calls == backing_before.read_calls + 1 &&
                    backing_after.read_syscalls == backing_before.read_syscalls + 2 &&
                    backing_after.syscall_attempts >= backing_before.syscall_attempts + 2 &&
                    backing_after.bytes_read == backing_before.bytes_read + failed.bytes,
                    "WT26b: both blocks reach backing reads but only the first completes");
            CHECK(fault_after.matching_attempts == fault_before.matching_attempts + 1 &&
                    fault_after.trigger_count == fault_before.trigger_count + 1 &&
                    fault_after.failed_block == 1 && fault_after.failed_attempt_id > 0,
                    "WT26b: backend EOF fault targets the second block");
            CHECK(p.kv->paged_release_bounded_test_read_block_state(0) == 1 &&
                    p.kv->paged_release_bounded_test_read_block_state(1) == 3,
                    "WT26b: completed block stays RESIDENT and failed block stays SWAPPED");
        }
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_KIND");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE");

        setenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE", "prefetch", 1);
        setenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS", "0", 1);
        setenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE", "1", 1);
        ContextGuard q;
        if (!q.init(model, cparams)) {
            CHECK(false, "WT26c: context creation failed");
        } else {
            std::vector<llama_token> prompt(16, 124);
            int rc = decode_prompt(q.ctx, prompt);
            CHECK(rc == 0, "WT26c: decode ok");
            const auto offload = q.kv->execute_action({
                llama_kv_action::offload, 7204, 0, UINT64_MAX, 1, false });
            CHECK(offload.state_changed, "WT26c: setup offload succeeds");
            const auto failed = q.kv->execute_action({
                llama_kv_action::prefetch, 7205, 0, 0, 1, true });
            CHECK(failed.outcome == llama_kv_action_outcome::failed && failed.io_failure &&
                    failed.fail_stop && !failed.state_changed && failed.core_transaction_id == 0,
                    "WT26c: first swap-in failure is not a no-op or transaction");
            CHECK(failed.blocks == 0 && failed.bytes == 0 && failed.shortfall_bytes > 0,
                    "WT26c: first swap-in failure reports zero completion and shortfall");
            CHECK(q.kv->paged_release_bounded_test_read_block_state(0) == 3,
                    "WT26c: first failed block stays SWAPPED");
        }
        unsetenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE");
        unsetenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS");
        unsetenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE");
        unsetenv("LLAMA_KV_PAGED_SWAP");
    }

    // =========================================================================
    // WT27: bounded multi-block OFFLOAD honors byte/block budgets, publishes one
    // transaction, and preserves per-block authority across partial I/O failure.
    // =========================================================================
    {
        setenv("LLAMA_KV_PAGED_SWAP", "1", 1);
        ContextGuard g;
        if (!g.init(model, cparams)) {
            CHECK(false, "WT27a: context creation failed");
        } else {
            std::vector<llama_token> prompt(48, 125);
            int rc = decode_prompt(g.ctx, prompt);
            CHECK(rc == 0, "WT27a: decode three blocks");

            const auto target_one = g.kv->execute_action({
                llama_kv_action::offload, 7301, 0, 1, 3, false });
            CHECK(target_one.outcome == llama_kv_action_outcome::completed &&
                    target_one.reason == llama_kv_action_reason::target_satisfied &&
                    target_one.blocks == 1 && target_one.bytes > 1 &&
                    target_one.shortfall_bytes == 0 && target_one.state_changed,
                    "WT27a: byte target stops after one atomic block overshoot");

            const auto max_one = g.kv->execute_action({
                llama_kv_action::offload, 7302, 0, UINT64_MAX, 1, false });
            CHECK(max_one.outcome == llama_kv_action_outcome::completed &&
                    max_one.reason == llama_kv_action_reason::scan_budget_exhausted &&
                    max_one.blocks == 1 && max_one.shortfall_bytes == UINT64_MAX - max_one.bytes &&
                    max_one.core_transaction_id > target_one.core_transaction_id,
                    "WT27a: max_blocks bounds later offload and reports exact shortfall");

            const auto restore_two = g.kv->execute_action({
                llama_kv_action::prefetch, 7303, 0, 0, 0, true, true });
            CHECK(restore_two.outcome == llama_kv_action_outcome::completed &&
                    restore_two.blocks == 2 && restore_two.state_changed,
                    "WT27a: all-required prefetch restores both bounded offloads");

            const auto multi = g.kv->execute_action({
                llama_kv_action::offload, 7304, 0, UINT64_MAX, 3, false });
            CHECK(multi.outcome == llama_kv_action_outcome::completed &&
                    multi.blocks == 3 && multi.bytes > 0 && multi.state_changed &&
                    multi.core_transaction_id > restore_two.core_transaction_id,
                    "WT27a: one bounded action offloads three blocks under one transaction");
            CHECK(g.kv->paged_release_bounded_test_read_block_state(0) == 3 &&
                    g.kv->paged_release_bounded_test_read_block_state(1) == 3 &&
                    g.kv->paged_release_bounded_test_read_block_state(2) == 3,
                    "WT27a: every completed block publishes SWAPPED");
        }

        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE", "swap_out", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_KIND", "write_enospc_once", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID", "0", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK", "1", 1);
        setenv("LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE", "1", 1);
        ContextGuard p;
        if (!p.init(model, cparams)) {
            CHECK(false, "WT27b: context creation failed");
        } else {
            std::vector<llama_token> prompt(48, 126);
            int rc = decode_prompt(p.ctx, prompt);
            CHECK(rc == 0, "WT27b: decode three blocks");

            const auto partial = p.kv->execute_action({
                llama_kv_action::offload, 7311, 0, UINT64_MAX, 3, false });
            CHECK(partial.outcome == llama_kv_action_outcome::partial_failure &&
                    partial.reason == llama_kv_action_reason::io_failure &&
                    partial.io_failure && partial.io_errno == ENOSPC &&
                    partial.state_changed && partial.core_transaction_id > 0,
                    "WT27b: second-block ENOSPC returns partial failure and one transaction");
            CHECK(partial.blocks == 1 && partial.bytes > 0 &&
                    partial.shortfall_bytes == UINT64_MAX - partial.bytes,
                    "WT27b: partial offload reports exact completed bytes and shortfall");
            CHECK(p.kv->paged_release_bounded_test_read_block_state(2) == 3 &&
                    p.kv->paged_release_bounded_test_read_block_state(1) == 1 &&
                    p.kv->paged_release_bounded_test_read_block_state(0) == 1,
                    "WT27b: completed block stays SWAPPED and failed/later blocks stay RESIDENT");

            const auto retry = p.kv->execute_action({
                llama_kv_action::offload, 7312, 0, UINT64_MAX, 3, false });
            CHECK(retry.outcome == llama_kv_action_outcome::completed &&
                    retry.blocks == 2 && retry.state_changed &&
                    retry.core_transaction_id > partial.core_transaction_id,
                    "WT27b: new decision retries remaining resident blocks without redoing SWAPPED block");
            CHECK(p.kv->paged_release_bounded_test_read_block_state(0) == 3 &&
                    p.kv->paged_release_bounded_test_read_block_state(1) == 3 &&
                    p.kv->paged_release_bounded_test_read_block_state(2) == 3,
                    "WT27b: retry completes remaining block publications");
        }
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_KIND");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK");
        unsetenv("LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE");
        unsetenv("LLAMA_KV_PAGED_SWAP");
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
