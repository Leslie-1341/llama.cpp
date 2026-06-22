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
#include <cstring>
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

extern "C" bool llama_kv_cache_set_seq_prefetch_protected(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        bool           enabled);

extern "C" bool llama_kv_cache_defer_idle_swapout(
        llama_memory_t mem,
        int32_t        n_steps);

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

enum class prefetch_pressure_mode {
    off,
    low,
    medium,
    high,
};

static const char * prefetch_pressure_mode_name(prefetch_pressure_mode mode) {
    switch (mode) {
        case prefetch_pressure_mode::off:
            return "off";
        case prefetch_pressure_mode::low:
            return "low";
        case prefetch_pressure_mode::medium:
            return "medium";
        case prefetch_pressure_mode::high:
            return "high";
    }

    return "off";
}

static prefetch_pressure_mode parse_prefetch_pressure_mode(const char * env) {
    if (env == nullptr || std::strcmp(env, "off") == 0) {
        return prefetch_pressure_mode::off;
    }
    if (std::strcmp(env, "low") == 0) {
        return prefetch_pressure_mode::low;
    }
    if (std::strcmp(env, "medium") == 0) {
        return prefetch_pressure_mode::medium;
    }
    if (std::strcmp(env, "high") == 0) {
        return prefetch_pressure_mode::high;
    }

    fprintf(stderr,
            "%s: warning: invalid LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE=%s; using off\n",
            __func__, env);
    return prefetch_pressure_mode::off;
}

static uint64_t clamp_prefetch_target_by_pressure(
        uint64_t remaining_blocks,
        prefetch_pressure_mode mode) {
    switch (mode) {
        case prefetch_pressure_mode::off:
        case prefetch_pressure_mode::low:
            return remaining_blocks;
        case prefetch_pressure_mode::medium:
            return std::min<uint64_t>(remaining_blocks, 3);
        case prefetch_pressure_mode::high:
            return 0;
    }

    return remaining_blocks;
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
    const llama_seq_id active_seq = 1;

    const char * num_idle_seqs_env = std::getenv("LLAMA_KV_IDLE_NUM_IDLE_SEQS");
    int32_t num_idle_seqs_requested = num_idle_seqs_env ? std::atoi(num_idle_seqs_env) : 1;
    num_idle_seqs_requested = std::max<int32_t>(1, num_idle_seqs_requested);
    const int32_t max_idle_seqs = std::max<int32_t>(1, params.n_parallel - 1);
    const int32_t num_idle_seqs = std::min<int32_t>(num_idle_seqs_requested, max_idle_seqs);
    if (num_idle_seqs_requested != num_idle_seqs) {
        fprintf(stderr,
                "KV_IDLE_SWAP_RESUME_MULTI_IDLE_CLAMP requested_num_idle_seqs=%d "
                "num_idle_seqs=%d n_parallel=%d\n",
                num_idle_seqs_requested, num_idle_seqs, params.n_parallel);
    }

    std::vector<llama_seq_id> idle_seqs;
    idle_seqs.reserve(num_idle_seqs);
    idle_seqs.push_back(0);
    for (int32_t i = 1; i < num_idle_seqs; ++i) {
        idle_seqs.push_back((llama_seq_id) (i + 1));
    }

    std::string idle_seqs_csv;
    for (size_t i = 0; i < idle_seqs.size(); ++i) {
        if (!idle_seqs_csv.empty()) {
            idle_seqs_csv += ",";
        }
        idle_seqs_csv += std::to_string(idle_seqs[i]);
    }

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
        num_idle_seqs * ((int32_t) idle_tokens.size() + seq0_warmup) +
        (int32_t) active_tokens.size() +
        2 * n_decode +
        32;
    const size_t max_prompt_tokens = std::max(idle_tokens.size(), active_tokens.size());
    ctx_params.n_ctx     = std::max<int32_t>(ctx_params.n_ctx, n_kv_req);
    ctx_params.n_batch   = std::max<int32_t>(ctx_params.n_batch,  (int32_t) max_prompt_tokens);
    ctx_params.n_ubatch  = std::max<int32_t>(ctx_params.n_ubatch, (int32_t) max_prompt_tokens);
    ctx_params.n_seq_max = std::max<uint32_t>(ctx_params.n_seq_max, (uint32_t) params.n_parallel);
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
    llama_batch batch = llama_batch_init((int32_t) max_prompt_tokens, 0, (int32_t) ctx_params.n_seq_max);

    const auto total_t0 = perf_clock::now();
    double seq0_prefill_ms = 0.0;
    double seq1_active_ms = 0.0;
    double seq1_prefill_ms = 0.0;
    double seq0_resume_first_token_ms = 0.0;
    double seq0_resume_total_ms = 0.0;
    const char * resume_timing_step_env = std::getenv("LLAMA_KV_PAGED_RESUME_TIMING_STEP");
    const bool resume_timing_step_enabled =
        resume_timing_step_env != nullptr && std::strcmp(resume_timing_step_env, "1") == 0;
    const char * defer_swapout_on_resume_env = std::getenv("LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME");
    const bool defer_swapout_on_resume =
        defer_swapout_on_resume_env != nullptr && std::strcmp(defer_swapout_on_resume_env, "1") == 0;
    const char * resume_prefetch_env = std::getenv("LLAMA_KV_PAGED_RESUME_PREFETCH");
    const bool prefetch_enabled = resume_prefetch_env != nullptr && std::atoi(resume_prefetch_env) != 0;
    const char * prefetch_during_active_env = std::getenv("LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE");
    const bool prefetch_during_active =
        prefetch_during_active_env != nullptr && std::atoi(prefetch_during_active_env) != 0;
    int32_t prefetch_after_active_tokens = 64;
    int32_t prefetch_every_tokens = 8;
    int32_t prefetch_blocks_per_step = 1;
    const char * prefetch_auto_delayed_env = std::getenv("LLAMA_KV_PAGED_PREFETCH_AUTO_DELAYED");
    const bool prefetch_auto_delayed =
        prefetch_auto_delayed_env != nullptr && std::atoi(prefetch_auto_delayed_env) != 0;
    const char * prefetch_auto_every_tokens_env = std::getenv("LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS");
    const char * prefetch_auto_blocks_per_step_env = std::getenv("LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP");
    const char * prefetch_auto_safety_tokens_env = std::getenv("LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS");
    const char * resume_pending_token_env = std::getenv("LLAMA_KV_PAGED_RESUME_PENDING_TOKEN");
    const prefetch_pressure_mode pressure_mode =
        parse_prefetch_pressure_mode(std::getenv("LLAMA_KV_PAGED_PREFETCH_PRESSURE_MODE"));
    if (const char * env = std::getenv("LLAMA_KV_PAGED_PREFETCH_AFTER_ACTIVE_TOKENS")) {
        prefetch_after_active_tokens = std::atoi(env);
    }
    if (const char * env = std::getenv("LLAMA_KV_PAGED_PREFETCH_EVERY_TOKENS")) {
        prefetch_every_tokens = std::atoi(env);
    }
    if (const char * env = std::getenv("LLAMA_KV_PAGED_PREFETCH_BLOCKS_PER_STEP")) {
        prefetch_blocks_per_step = std::atoi(env);
    }
    prefetch_after_active_tokens = std::max<int32_t>(0, prefetch_after_active_tokens);
    prefetch_every_tokens = std::max<int32_t>(1, prefetch_every_tokens);
    prefetch_blocks_per_step = std::max<int32_t>(1, prefetch_blocks_per_step);
    int32_t prefetch_auto_every_tokens = prefetch_auto_every_tokens_env ?
        std::atoi(prefetch_auto_every_tokens_env) : prefetch_every_tokens;
    int32_t prefetch_auto_blocks_per_step = prefetch_auto_blocks_per_step_env ?
        std::atoi(prefetch_auto_blocks_per_step_env) : prefetch_blocks_per_step;
    int32_t prefetch_auto_safety_tokens = prefetch_auto_safety_tokens_env ?
        std::atoi(prefetch_auto_safety_tokens_env) : 0;
    prefetch_auto_every_tokens = std::max<int32_t>(1, prefetch_auto_every_tokens);
    prefetch_auto_blocks_per_step = std::max<int32_t>(1, prefetch_auto_blocks_per_step);
    prefetch_auto_safety_tokens = std::max<int32_t>(0, prefetch_auto_safety_tokens);
    const int32_t resume_pending_token = resume_pending_token_env ? std::atoi(resume_pending_token_env) : 0;
    int32_t prefetch_auto_active_total_tokens = 0;
    uint64_t prefetch_auto_remaining_blocks = 0;
    uint64_t target_restore_blocks = 0;
    uint64_t prefetch_auto_need_steps = 0;
    int32_t prefetch_auto_start_token = 0;
    int32_t prefetch_auto_window_ok = 1;
    int32_t resume_pending_started = 0;
    int32_t resume_pending_target_limited = 0;
    int32_t active_window_remaining = 0;
    int32_t effective_start_token = 0;
    int32_t resume_pending_window_ok = 1;
    double prefetch_ms = 0.0;
    int32_t prefetch_blocks = 0;
    int32_t prefetch_during_active_calls = 0;
    int32_t prefetch_during_active_blocks = 0;
    int32_t prefetch_protect_enabled = 0;
    double prefetch_during_active_ms_total = 0.0;
    double prefetch_during_active_ms_max = 0.0;
    uint64_t prefetch_remaining_blocks_before_resume = 0;
    uint64_t rss_before_prefetch_kb = 0;
    uint64_t rss_after_prefetch_kb = 0;
    uint64_t rss_before_active_prefetch_kb = 0;
    uint64_t rss_after_active_prefetch_kb = 0;
    uint64_t rss_before_resume_kb = 0;
    uint64_t rss_after_resume_kb = 0;
    uint64_t prefetch_owned_blocks = 0;
    uint64_t prefetch_swapped_blocks = 0;
    uint64_t prefetch_resident_blocks = 0;
    uint64_t prefetch_released_blocks = 0;
    uint64_t prefetch_invalid_cells = 0;
    uint64_t prefetch_failures = 0;
    llama_token seq0_resume_first = LLAMA_TOKEN_NULL;
    llama_token seq0_warm_token = LLAMA_TOKEN_NULL;
    int seq0_warmed = 0;

    const auto warmup_idle_seq = [&](llama_seq_id seq_id) -> bool {
        common_sampler * smpl = seq_id == 0 ? seq0_smpl : common_sampler_init(model, params.sampling);
        double prefill_ms = 0.0;

        common_batch_clear(batch);
        for (size_t i = 0; i < idle_tokens.size(); ++i) {
            common_batch_add(batch, idle_tokens[i], (llama_pos) i, { seq_id }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        const auto prefill_t0 = perf_clock::now();
        if (!decode_batch(ctx, batch, seq_id == 0 ? "seq0-idle-prefill" : "multi-idle-prefill")) {
            if (seq_id != 0) {
                common_sampler_free(smpl);
            }
            return false;
        }
        prefill_ms = elapsed_ms(prefill_t0, perf_clock::now());
        if (seq_id == 0) {
            seq0_prefill_ms = prefill_ms;
        }

        llama_token warm_token = common_sampler_sample(smpl, ctx, batch.n_tokens - 1);
        common_sampler_accept(smpl, warm_token, true);
        if (llama_vocab_is_eog(vocab, warm_token)) {
            fprintf(stderr,
                    "%s: seq%d prefill sampled EOG; choose a different prompt/model for this driver\n",
                    __func__, (int) seq_id);
            if (seq_id != 0) {
                common_sampler_free(smpl);
            }
            return false;
        }
        const llama_token first_token = warm_token;

        llama_pos warm_pos = (llama_pos) idle_tokens.size();
        int warmed = 0;
        for (int w = 0; w < seq0_warmup; ++w) {
            if (llama_vocab_is_eog(vocab, warm_token)) {
                break;
            }

            common_batch_clear(batch);
            common_batch_add(batch, warm_token, warm_pos++, { seq_id }, true);

            if (!decode_batch(ctx, batch, seq_id == 0 ? "seq0-idle-warmup" : "multi-idle-warmup")) {
                if (seq_id != 0) {
                    common_sampler_free(smpl);
                }
                return false;
            }

            warmed += 1;
            warm_token = common_sampler_sample(smpl, ctx, batch.n_tokens - 1);
            common_sampler_accept(smpl, warm_token, true);
        }

        if (seq_id == 0) {
            seq0_resume_first = first_token;
            seq0_warm_token = warm_token;
            seq0_warmed = warmed;
        } else {
            common_sampler_free(smpl);
        }

        return true;
    };

    for (llama_seq_id idle_seq : idle_seqs) {
        if (!warmup_idle_seq(idle_seq)) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }
    }

    common_batch_clear(batch);
    for (size_t i = 0; i < active_tokens.size(); ++i) {
        common_batch_add(batch, active_tokens[i], (llama_pos) i, { active_seq }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    const auto seq1_active_t0 = perf_clock::now();
    const auto seq1_prefill_t0 = perf_clock::now();
    if (!decode_batch(ctx, batch, "seq1-active-prefill")) {
        cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
        return 1;
    }
    seq1_prefill_ms = elapsed_ms(seq1_prefill_t0, perf_clock::now());

    bool prefetch_during_active_schedule_enabled = prefetch_during_active && !prefetch_auto_delayed;
    const auto start_resume_pending = [&]() -> bool {
        if (resume_pending_started) {
            return true;
        }

        resume_pending_started = 1;
        llama_kv_cache_set_seq_prefetch_protected(llama_get_memory(ctx), 0, true);
        prefetch_protect_enabled = 1;

        const int32_t prefetch_auto_probe_blocks = llama_memory_prefetch_seq_step(llama_get_memory(ctx), 0, 0);
        if (prefetch_auto_probe_blocks < 0) {
            fprintf(stderr, "%s: llama_memory_prefetch_seq_step() auto probe failed for seq0\n", __func__);
            return false;
        }

        uint64_t prefetch_auto_owned_blocks = 0;
        uint64_t prefetch_auto_resident_blocks = 0;
        uint64_t prefetch_auto_released_blocks = 0;
        uint64_t prefetch_auto_invalid_cells = 0;
        uint64_t prefetch_auto_failures = 0;
        llama_kv_cache_prefetch_seq_last_stats(
                llama_get_memory(ctx),
                &prefetch_auto_owned_blocks,
                &prefetch_auto_remaining_blocks,
                &prefetch_auto_resident_blocks,
                &prefetch_auto_released_blocks,
                &prefetch_auto_invalid_cells,
                &prefetch_auto_failures);

        prefetch_auto_active_total_tokens = n_decode;
        active_window_remaining = std::max<int32_t>(0, n_decode - resume_pending_token);
        target_restore_blocks = clamp_prefetch_target_by_pressure(prefetch_auto_remaining_blocks, pressure_mode);
        resume_pending_target_limited = 1;
        if (target_restore_blocks == 0) {
            prefetch_auto_need_steps = 0;
            effective_start_token = resume_pending_token;
            prefetch_auto_start_token = effective_start_token;
            prefetch_auto_window_ok = 1;
            resume_pending_window_ok = 1;
            prefetch_during_active_schedule_enabled = false;
        } else {
            prefetch_auto_need_steps =
                (target_restore_blocks + (uint64_t) prefetch_auto_blocks_per_step - 1) /
                (uint64_t) prefetch_auto_blocks_per_step;
            const int64_t need_span =
                (int64_t) (prefetch_auto_need_steps - 1) * (int64_t) prefetch_auto_every_tokens +
                (int64_t) prefetch_auto_safety_tokens;
            if (need_span <= (int64_t) active_window_remaining) {
                effective_start_token =
                    resume_pending_token + (active_window_remaining - (int32_t) need_span);
                prefetch_auto_window_ok = 1;
                resume_pending_window_ok = 1;
            } else {
                effective_start_token = resume_pending_token;
                prefetch_auto_window_ok = 0;
                resume_pending_window_ok = 0;
            }
            prefetch_auto_start_token = effective_start_token;
            prefetch_during_active_schedule_enabled = true;
        }

        prefetch_after_active_tokens = prefetch_auto_start_token;
        prefetch_every_tokens = prefetch_auto_every_tokens;
        prefetch_blocks_per_step = prefetch_auto_blocks_per_step;

        return true;
    };

    if (prefetch_during_active && prefetch_auto_delayed &&
            resume_pending_token == 0 &&
            !start_resume_pending()) {
        cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
        return 1;
    }

    // Stage 6C-1A: while interleaving prefetch for seq0 during seq1 active decode, mark seq0
    // resume-pending so the idle swap-out gate does not re-evict the blocks we just prefetched.
    if (prefetch_during_active && !prefetch_auto_delayed) {
        llama_kv_cache_set_seq_prefetch_protected(llama_get_memory(ctx), 0, true);
        prefetch_protect_enabled = 1;
    }

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
        common_batch_add(batch, token, seq1_pos++, { active_seq }, true);
        sample_idx = batch.n_tokens - 1;

        if (!decode_batch(ctx, batch, "seq1-active-decode")) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }

        const int32_t seq1_decoded_done = seq1_decoded + 1;
        if (prefetch_during_active && prefetch_auto_delayed &&
                resume_pending_token > 0 &&
                seq1_decoded_done >= resume_pending_token &&
                !start_resume_pending()) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }

        if (prefetch_during_active_schedule_enabled &&
                (!prefetch_auto_delayed || resume_pending_started) &&
                (!resume_pending_target_limited ||
                 (uint64_t) prefetch_during_active_blocks < target_restore_blocks) &&
                seq1_decoded_done >= prefetch_after_active_tokens &&
                ((seq1_decoded_done - prefetch_after_active_tokens) % prefetch_every_tokens) == 0) {
            if (prefetch_during_active_calls == 0) {
                rss_before_active_prefetch_kb = current_rss_kb();
            }

            uint32_t blocks_this_step = (uint32_t) prefetch_blocks_per_step;
            if (resume_pending_target_limited) {
                const uint64_t target_remaining =
                    target_restore_blocks - (uint64_t) prefetch_during_active_blocks;
                blocks_this_step = (uint32_t) std::min<uint64_t>(target_remaining, blocks_this_step);
            }

            const auto active_prefetch_t0 = perf_clock::now();
            const int32_t restored = llama_memory_prefetch_seq_step(
                    llama_get_memory(ctx), 0, blocks_this_step);
            const double active_prefetch_ms = elapsed_ms(active_prefetch_t0, perf_clock::now());

            prefetch_during_active_calls += 1;
            prefetch_during_active_ms_total += active_prefetch_ms;
            prefetch_during_active_ms_max = std::max(prefetch_during_active_ms_max, active_prefetch_ms);
            if (restored > 0) {
                int32_t restored_capped = restored;
                if (resume_pending_target_limited) {
                    const uint64_t target_remaining =
                        target_restore_blocks - (uint64_t) prefetch_during_active_blocks;
                    restored_capped = (int32_t) std::min<uint64_t>((uint64_t) restored, target_remaining);
                }
                prefetch_during_active_blocks += restored_capped;
            }
            rss_after_active_prefetch_kb = current_rss_kb();

            if (restored < 0) {
                fprintf(stderr, "%s: llama_memory_prefetch_seq_step() failed for seq0 during seq1 active decode\n", __func__);
                cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
                return 1;
            }
        }
    }
    seq1_active_ms = elapsed_ms(seq1_active_t0, perf_clock::now());

    std::string seq0_resume_generated;
    llama_token seq0_token = seq0_warmed > 0 ? seq0_warm_token : seq0_resume_first;
    llama_pos seq0_pos = (llama_pos) idle_tokens.size() + seq0_warmed;
    int32_t seq0_resume_decoded = 0;

    rss_before_prefetch_kb = current_rss_kb();
    const int32_t prefetch_probe_blocks = llama_memory_prefetch_seq_step(llama_get_memory(ctx), 0, 0);
    if (prefetch_probe_blocks < 0) {
        fprintf(stderr, "%s: llama_memory_prefetch_seq_step() probe failed for seq0\n", __func__);
        cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
        return 1;
    }
    uint64_t prefetch_probe_owned_blocks = 0;
    uint64_t prefetch_probe_resident_blocks = 0;
    uint64_t prefetch_probe_released_blocks = 0;
    uint64_t prefetch_probe_invalid_cells = 0;
    uint64_t prefetch_probe_failures = 0;
    llama_kv_cache_prefetch_seq_last_stats(
            llama_get_memory(ctx),
            &prefetch_probe_owned_blocks,
            &prefetch_remaining_blocks_before_resume,
            &prefetch_probe_resident_blocks,
            &prefetch_probe_released_blocks,
            &prefetch_probe_invalid_cells,
            &prefetch_probe_failures);
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
    const int32_t prefetch_auto_completed =
        prefetch_remaining_blocks_before_resume == 0 ? 1 : 0;
    const uint64_t prefetch_auto_fallback_blocks = prefetch_remaining_blocks_before_resume;
    const uint64_t resume_pending_fallback_blocks = prefetch_remaining_blocks_before_resume;

    rss_before_resume_kb = current_rss_kb();
    const auto seq0_resume_total_t0 = perf_clock::now();
    for (; seq0_resume_decoded < n_decode; ++seq0_resume_decoded) {
        if (llama_vocab_is_eog(vocab, seq0_token)) {
            break;
        }

        seq0_resume_generated += common_token_to_piece(ctx, seq0_token);

        const auto seq0_resume_step_t0 = seq0_resume_decoded == 0 ? perf_clock::now() : perf_clock::time_point{};
        if (seq0_resume_decoded == 0 && defer_swapout_on_resume) {
            llama_kv_cache_defer_idle_swapout(llama_get_memory(ctx), 1);
        }
        if (seq0_resume_decoded == 0 && resume_timing_step_enabled) {
            fprintf(stderr, "KV_RESUME_FIRST_BEGIN\n");
        }

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
            if (resume_timing_step_enabled) {
                fprintf(stderr, "KV_RESUME_FIRST_END\n");
            }
        }
    }
    seq0_resume_total_ms = elapsed_ms(seq0_resume_total_t0, perf_clock::now());
    rss_after_resume_kb = current_rss_kb();

    // Stage 6C-1A: seq0 has resumed; drop the resume-pending protection so the default idle
    // swap-out policy applies again to seq0-owned blocks.
    if (prefetch_protect_enabled) {
        llama_kv_cache_set_seq_prefetch_protected(llama_get_memory(ctx), 0, false);
    }

    printf("idle_seq=0\n");
    printf("active_seq=%d\n", (int) active_seq);
    printf("multi_idle_enabled=%d\n", num_idle_seqs > 1 ? 1 : 0);
    printf("num_idle_seqs=%d\n", num_idle_seqs);
    printf("idle_seqs=%s\n", idle_seqs_csv.c_str());
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
            "multi_idle_enabled=%d num_idle_seqs=%d active_seq=%d idle_seqs=%s "
            "seq0_resume_first_token_ms=%.3f seq0_resume_total_ms=%.3f "
            "seq1_active_tokens=%d seq0_resume_tokens=%d total_measured_tokens=%d "
            "tokens_per_second=%.6f seq0_prefill_ms=%.3f seq1_prefill_ms=%.3f "
            "seq0_warmup_tokens=%d seq0_warmed_tokens=%d "
            "prefetch_enabled=%d prefetch_ms=%.3f prefetch_blocks=%d "
            "prefetch_during_active_enabled=%d prefetch_during_active_calls=%d "
            "prefetch_protect_enabled=%d "
            "prefetch_during_active_blocks=%d prefetch_during_active_ms_total=%.3f "
            "prefetch_during_active_ms_max=%.3f prefetch_remaining_blocks_before_resume=%llu "
            "prefetch_owned_blocks=%llu prefetch_swapped_blocks=%llu "
            "prefetch_resident_blocks=%llu prefetch_released_blocks=%llu "
            "prefetch_invalid_cells=%llu prefetch_failures=%llu "
            "prefetch_auto_enabled=%d prefetch_auto_active_total_tokens=%d "
            "prefetch_auto_remaining_blocks=%llu prefetch_auto_need_steps=%llu "
            "pressure_mode=%s target_restore_blocks=%llu "
            "prefetch_auto_start_token=%d prefetch_auto_every_tokens=%d "
            "prefetch_auto_blocks_per_step=%d prefetch_auto_safety_tokens=%d "
            "prefetch_auto_window_ok=%d prefetch_auto_started=%d "
            "prefetch_auto_completed=%d prefetch_auto_fallback_blocks=%llu "
            "resume_pending_token=%d resume_pending_started=%d "
            "active_window_remaining=%d effective_start_token=%d "
            "resume_pending_window_ok=%d resume_pending_fallback_blocks=%llu "
            "rss_before_active_prefetch_kb=%llu rss_after_active_prefetch_kb=%llu "
            "rss_before_prefetch_kb=%llu rss_after_prefetch_kb=%llu "
            "rss_before_resume_kb=%llu rss_after_resume_kb=%llu\n",
            total_wall_ms, seq1_active_ms,
            num_idle_seqs > 1 ? 1 : 0,
            num_idle_seqs,
            (int) active_seq,
            idle_seqs_csv.c_str(),
            seq0_resume_first_token_ms, seq0_resume_total_ms,
            seq1_active_tokens, seq0_resume_tokens, total_measured_tokens,
            tokens_per_second, seq0_prefill_ms, seq1_prefill_ms,
            seq0_warmup, seq0_warmed,
            prefetch_enabled ? 1 : 0, prefetch_ms, prefetch_blocks,
            prefetch_during_active ? 1 : 0,
            prefetch_during_active_calls,
            prefetch_protect_enabled,
            prefetch_during_active_blocks,
            prefetch_during_active_ms_total,
            prefetch_during_active_ms_max,
            (unsigned long long) prefetch_remaining_blocks_before_resume,
            (unsigned long long) prefetch_owned_blocks,
            (unsigned long long) prefetch_swapped_blocks,
            (unsigned long long) prefetch_resident_blocks,
            (unsigned long long) prefetch_released_blocks,
            (unsigned long long) prefetch_invalid_cells,
            (unsigned long long) prefetch_failures,
            prefetch_auto_delayed ? 1 : 0,
            prefetch_auto_active_total_tokens,
            (unsigned long long) prefetch_auto_remaining_blocks,
            (unsigned long long) prefetch_auto_need_steps,
            prefetch_pressure_mode_name(pressure_mode),
            (unsigned long long) target_restore_blocks,
            prefetch_auto_start_token,
            prefetch_auto_every_tokens,
            prefetch_auto_blocks_per_step,
            prefetch_auto_safety_tokens,
            prefetch_auto_window_ok,
            prefetch_auto_delayed && prefetch_during_active_calls > 0 ? 1 : 0,
            prefetch_auto_completed,
            (unsigned long long) prefetch_auto_fallback_blocks,
            resume_pending_token,
            resume_pending_started,
            active_window_remaining,
            effective_start_token,
            resume_pending_window_ok,
            (unsigned long long) resume_pending_fallback_blocks,
            (unsigned long long) rss_before_active_prefetch_kb,
            (unsigned long long) rss_after_active_prefetch_kb,
            (unsigned long long) rss_before_prefetch_kb,
            (unsigned long long) rss_after_prefetch_kb,
            (unsigned long long) rss_before_resume_kb,
            (unsigned long long) rss_after_resume_kb);

    cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
    return 0;
}
