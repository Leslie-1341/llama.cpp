#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

using perf_clock = std::chrono::steady_clock;

extern "C" bool llama_kv_cache_prefetch_seq_last_stats(
        llama_memory_t mem,
        uint64_t * owned_blocks,
        uint64_t * swapped_blocks,
        uint64_t * resident_blocks,
        uint64_t * released_blocks,
        uint64_t * invalid_cells,
        uint64_t * failures);

static double elapsed_ms(perf_clock::time_point t0, perf_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static uint64_t current_rss_kb() {
#if defined(__linux__)
    FILE * f = std::fopen("/proc/self/statm", "r");
    if (!f) {
        return 0;
    }

    long pages_total = 0;
    long pages_rss = 0;
    const int n = std::fscanf(f, "%ld %ld", &pages_total, &pages_rss);
    std::fclose(f);

    if (n != 2 || pages_rss < 0) {
        return 0;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return 0;
    }

    return (uint64_t) pages_rss * (uint64_t) page / 1024u;
#else
    return 0;
#endif
}

static void print_usage(int, char ** argv) {
    fprintf(stderr, "\nexample usage:\n");
    fprintf(stderr, "\n    %s -m model.gguf -p \"Active request prompt\" -n 64 --ctx-size 768\n", argv[0]);
    fprintf(stderr, "\n");
}

static bool decode_batch(llama_context * ctx, llama_batch & batch, const char * stage) {
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: llama_decode() failed during %s\n", __func__, stage);
        return false;
    }
    return true;
}

static void cleanup(
        llama_batch & batch,
        common_sampler * seq0_smpl,
        common_sampler * seq1_smpl,
        llama_context * ctx,
        llama_model * model) {
    llama_batch_free(batch);
    common_sampler_free(seq0_smpl);
    common_sampler_free(seq1_smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "Active request: list three colors.";
    params.n_predict = 32;
    params.n_parallel = 2;
    params.kv_unified = true;
    params.sampling.seed = 1;
    params.sampling.temp = 0.0f;
    params.sampling.backend_sampling = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BATCHED, print_usage)) {
        return 1;
    }

    params.n_parallel = std::max<int32_t>(params.n_parallel, 2);
    params.kv_unified = true;
    params.sampling.backend_sampling = false;

    const int n_decode = params.n_predict < 0 ? 32 : params.n_predict;

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (model == nullptr) {
        fprintf(stderr, "%s: unable to load model\n", __func__);
        llama_backend_free();
        return 1;
    }

    if (llama_model_has_encoder(model)) {
        fprintf(stderr, "%s: encoder-decoder models are not supported by this resume driver\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    const std::string idle_prompt =
        "Idle request: alpha beta gamma delta epsilon zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho sigma tau.";
    const std::string active_prompt = params.prompt.empty() ? "Active request: list three colors." : params.prompt;

    std::vector<llama_token> idle_tokens   = common_tokenize(vocab, idle_prompt,   true);
    std::vector<llama_token> active_tokens = common_tokenize(vocab, active_prompt, true);

    if (idle_tokens.empty() || active_tokens.empty()) {
        fprintf(stderr, "%s: tokenization produced an empty prompt\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const char * seq0_warmup_env = std::getenv("LLAMA_KV_IDLE_SEQ0_WARMUP_TOKENS");
    const int seq0_warmup = seq0_warmup_env ? std::max(0, std::atoi(seq0_warmup_env)) : 0;

    llama_context_params ctx_params = common_context_params_to_llama(params);
    const int32_t n_kv_req =
        (int32_t) idle_tokens.size() +
        (int32_t) active_tokens.size() +
        seq0_warmup +
        2 * n_decode +
        32;
    const size_t max_prompt_tokens = std::max(idle_tokens.size(), active_tokens.size());
    ctx_params.n_ctx     = std::max<int32_t>(ctx_params.n_ctx, n_kv_req);
    ctx_params.n_batch   = std::max<int32_t>(ctx_params.n_batch,  (int32_t) max_prompt_tokens);
    ctx_params.n_ubatch  = std::max<int32_t>(ctx_params.n_ubatch, (int32_t) max_prompt_tokens);
    ctx_params.n_seq_max = std::max<uint32_t>(ctx_params.n_seq_max, 2);
    ctx_params.kv_unified = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create llama_context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    common_sampler * seq0_smpl = common_sampler_init(model, params.sampling);
    common_sampler * seq1_smpl = common_sampler_init(model, params.sampling);
    llama_batch batch = llama_batch_init((int32_t) max_prompt_tokens, 0, 2);

    const auto total_t0 = perf_clock::now();
    double seq0_prefill_ms = 0.0;
    double seq1_active_ms = 0.0;
    double seq1_prefill_ms = 0.0;
    double seq0_resume_first_token_ms = 0.0;
    double seq0_resume_total_ms = 0.0;
    const char * resume_prefetch_env = std::getenv("LLAMA_KV_PAGED_RESUME_PREFETCH");
    const bool prefetch_enabled = resume_prefetch_env != nullptr && std::atoi(resume_prefetch_env) != 0;
    double prefetch_ms = 0.0;
    int32_t prefetch_blocks = 0;
    uint64_t rss_before_prefetch_kb = 0;
    uint64_t rss_after_prefetch_kb = 0;
    uint64_t rss_after_resume_kb = 0;
    uint64_t prefetch_owned_blocks = 0;
    uint64_t prefetch_swapped_blocks = 0;
    uint64_t prefetch_resident_blocks = 0;
    uint64_t prefetch_released_blocks = 0;
    uint64_t prefetch_invalid_cells = 0;
    uint64_t prefetch_failures = 0;

    common_batch_clear(batch);
    for (size_t i = 0; i < idle_tokens.size(); ++i) {
        common_batch_add(batch, idle_tokens[i], (llama_pos) i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    const auto seq0_prefill_t0 = perf_clock::now();
    if (!decode_batch(ctx, batch, "seq0-idle-prefill")) {
        cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
        return 1;
    }
    seq0_prefill_ms = elapsed_ms(seq0_prefill_t0, perf_clock::now());

    const llama_token seq0_resume_first = common_sampler_sample(seq0_smpl, ctx, batch.n_tokens - 1);
    common_sampler_accept(seq0_smpl, seq0_resume_first, true);
    if (llama_vocab_is_eog(vocab, seq0_resume_first)) {
        fprintf(stderr, "%s: seq0 prefill sampled EOG; choose a different prompt/model for this driver\n", __func__);
        cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
        return 1;
    }

    llama_token seq0_warm_token = seq0_resume_first;
    llama_pos seq0_warm_pos = (llama_pos) idle_tokens.size();
    int seq0_warmed = 0;
    for (int w = 0; w < seq0_warmup; ++w) {
        if (llama_vocab_is_eog(vocab, seq0_warm_token)) {
            break;
        }

        common_batch_clear(batch);
        common_batch_add(batch, seq0_warm_token, seq0_warm_pos++, { 0 }, true);

        if (!decode_batch(ctx, batch, "seq0-idle-warmup")) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }

        seq0_warmed += 1;
        seq0_warm_token = common_sampler_sample(seq0_smpl, ctx, batch.n_tokens - 1);
        common_sampler_accept(seq0_smpl, seq0_warm_token, true);
    }

    common_batch_clear(batch);
    for (size_t i = 0; i < active_tokens.size(); ++i) {
        common_batch_add(batch, active_tokens[i], (llama_pos) i, { 1 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    const auto seq1_active_t0 = perf_clock::now();
    const auto seq1_prefill_t0 = perf_clock::now();
    if (!decode_batch(ctx, batch, "seq1-active-prefill")) {
        cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
        return 1;
    }
    seq1_prefill_ms = elapsed_ms(seq1_prefill_t0, perf_clock::now());

    std::string seq1_generated;
    int32_t sample_idx = batch.n_tokens - 1;
    llama_pos seq1_pos = (llama_pos) active_tokens.size();
    int32_t seq1_decoded = 0;

    for (; seq1_decoded < n_decode; ++seq1_decoded) {
        const llama_token token = common_sampler_sample(seq1_smpl, ctx, sample_idx);
        common_sampler_accept(seq1_smpl, token, true);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        seq1_generated += common_token_to_piece(ctx, token);

        common_batch_clear(batch);
        common_batch_add(batch, token, seq1_pos++, { 1 }, true);
        sample_idx = batch.n_tokens - 1;

        if (!decode_batch(ctx, batch, "seq1-active-decode")) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }
    }
    seq1_active_ms = elapsed_ms(seq1_active_t0, perf_clock::now());

    std::string seq0_resume_generated;
    llama_token seq0_token = seq0_warmed > 0 ? seq0_warm_token : seq0_resume_first;
    llama_pos seq0_pos = (llama_pos) idle_tokens.size() + seq0_warmed;
    int32_t seq0_resume_decoded = 0;

    rss_before_prefetch_kb = current_rss_kb();
    if (prefetch_enabled) {
        const auto prefetch_t0 = perf_clock::now();
        prefetch_blocks = llama_memory_prefetch_seq(llama_get_memory(ctx), 0);
        prefetch_ms = elapsed_ms(prefetch_t0, perf_clock::now());
        llama_kv_cache_prefetch_seq_last_stats(
                llama_get_memory(ctx),
                &prefetch_owned_blocks,
                &prefetch_swapped_blocks,
                &prefetch_resident_blocks,
                &prefetch_released_blocks,
                &prefetch_invalid_cells,
                &prefetch_failures);
        rss_after_prefetch_kb = current_rss_kb();
        if (prefetch_blocks < 0) {
            fprintf(stderr, "%s: llama_memory_prefetch_seq() failed for seq0\n", __func__);
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }
    } else {
        rss_after_prefetch_kb = rss_before_prefetch_kb;
    }

    const auto seq0_resume_total_t0 = perf_clock::now();
    for (; seq0_resume_decoded < n_decode; ++seq0_resume_decoded) {
        if (llama_vocab_is_eog(vocab, seq0_token)) {
            break;
        }

        seq0_resume_generated += common_token_to_piece(ctx, seq0_token);

        const auto seq0_resume_step_t0 = seq0_resume_decoded == 0 ? perf_clock::now() : perf_clock::time_point{};

        common_batch_clear(batch);
        common_batch_add(batch, seq0_token, seq0_pos++, { 0 }, true);
        sample_idx = batch.n_tokens - 1;

        if (!decode_batch(ctx, batch, "seq0-resume-decode")) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }

        seq0_token = common_sampler_sample(seq0_smpl, ctx, sample_idx);
        common_sampler_accept(seq0_smpl, seq0_token, true);
        if (seq0_resume_decoded == 0) {
            seq0_resume_first_token_ms = elapsed_ms(seq0_resume_step_t0, perf_clock::now());
        }
    }
    seq0_resume_total_ms = elapsed_ms(seq0_resume_total_t0, perf_clock::now());
    rss_after_resume_kb = current_rss_kb();

    printf("idle_seq=0\n");
    printf("active_seq=1\n");
    printf("idle_prompt_tokens=%zu\n", idle_tokens.size());
    printf("active_prompt_tokens=%zu\n", active_tokens.size());
    printf("seq1_decoded_tokens=%d\n", seq1_decoded);
    printf("seq0_resume_decoded_tokens=%d\n", seq0_resume_decoded);
    printf("===SEQ1_ACTIVE_BEGIN===\n");
    printf("%s\n", seq1_generated.c_str());
    printf("===SEQ1_ACTIVE_END===\n");
    printf("===SEQ0_RESUME_BEGIN===\n");
    printf("%s\n", seq0_resume_generated.c_str());
    printf("===SEQ0_RESUME_END===\n");
    fflush(stdout);

    const double total_wall_ms = elapsed_ms(total_t0, perf_clock::now());
    const int32_t seq1_active_tokens = seq1_decoded;
    const int32_t seq0_resume_tokens = seq0_resume_decoded;
    const int32_t total_measured_tokens = seq1_active_tokens + seq0_resume_tokens;
    const double measured_generation_ms = seq1_active_ms + seq0_resume_total_ms;
    const double tokens_per_second = measured_generation_ms > 0.0 ?
        1000.0 * (double) total_measured_tokens / measured_generation_ms : 0.0;

    fprintf(stderr,
            "KV_IDLE_SWAP_RESUME_PERF total_wall_ms=%.3f seq1_active_ms=%.3f "
            "seq0_resume_first_token_ms=%.3f seq0_resume_total_ms=%.3f "
            "seq1_active_tokens=%d seq0_resume_tokens=%d total_measured_tokens=%d "
            "tokens_per_second=%.6f seq0_prefill_ms=%.3f seq1_prefill_ms=%.3f "
            "seq0_warmup_tokens=%d seq0_warmed_tokens=%d "
            "prefetch_enabled=%d prefetch_ms=%.3f prefetch_blocks=%d "
            "prefetch_owned_blocks=%llu prefetch_swapped_blocks=%llu "
            "prefetch_resident_blocks=%llu prefetch_released_blocks=%llu "
            "prefetch_invalid_cells=%llu prefetch_failures=%llu "
            "rss_before_prefetch_kb=%llu rss_after_prefetch_kb=%llu rss_after_resume_kb=%llu\n",
            total_wall_ms, seq1_active_ms,
            seq0_resume_first_token_ms, seq0_resume_total_ms,
            seq1_active_tokens, seq0_resume_tokens, total_measured_tokens,
            tokens_per_second, seq0_prefill_ms, seq1_prefill_ms,
            seq0_warmup, seq0_warmed,
            prefetch_enabled ? 1 : 0, prefetch_ms, prefetch_blocks,
            (unsigned long long) prefetch_owned_blocks,
            (unsigned long long) prefetch_swapped_blocks,
            (unsigned long long) prefetch_resident_blocks,
            (unsigned long long) prefetch_released_blocks,
            (unsigned long long) prefetch_invalid_cells,
            (unsigned long long) prefetch_failures,
            (unsigned long long) rss_before_prefetch_kb,
            (unsigned long long) rss_after_prefetch_kb,
            (unsigned long long) rss_after_resume_kb);

    cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
    return 0;
}
