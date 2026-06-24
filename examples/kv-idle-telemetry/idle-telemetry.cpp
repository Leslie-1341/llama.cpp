#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <string>
#include <vector>

static void print_usage(int, char ** argv) {
    fprintf(stderr, "\nexample usage:\n");
    fprintf(stderr, "\n    %s -m model.gguf -p \"Active request prompt\" -n 32 --ctx-size 512\n", argv[0]);
    fprintf(stderr, "\n");
}

static bool decode_batch(llama_context * ctx, llama_batch & batch, const char * stage) {
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: llama_decode() failed during %s\n", __func__, stage);
        return false;
    }
    return true;
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
    params.sampling.temp = 0.0f;
    params.sampling.backend_sampling = false;

    const int n_predict = params.n_predict < 0 ? 32 : params.n_predict;

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
        fprintf(stderr, "%s: encoder-decoder models are not supported by this telemetry driver\n", __func__);
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

    llama_context_params ctx_params = common_context_params_to_llama(params);
    const int32_t n_kv_req = (int32_t) idle_tokens.size() + (int32_t) active_tokens.size() + n_predict + 16;
    ctx_params.n_ctx     = std::max<int32_t>(ctx_params.n_ctx, n_kv_req);
    ctx_params.n_batch   = std::max<int32_t>(ctx_params.n_batch, std::max(idle_tokens.size(), active_tokens.size()));
    ctx_params.n_ubatch  = std::max<int32_t>(ctx_params.n_ubatch, std::max(idle_tokens.size(), active_tokens.size()));
    ctx_params.n_seq_max = std::max<uint32_t>(ctx_params.n_seq_max, 2);
    ctx_params.kv_unified = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create llama_context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_sampler * smpl = llama_sampler_init_greedy();
    llama_batch batch = llama_batch_init(std::max(idle_tokens.size(), active_tokens.size()), 0, 2);

    common_batch_clear(batch);
    for (size_t i = 0; i < idle_tokens.size(); ++i) {
        common_batch_add(batch, idle_tokens[i], (llama_pos) i, { 0 }, false);
    }
    batch.logits[batch.n_tokens - 1] = false;
    if (!decode_batch(ctx, batch, "seq0-idle-prefill")) {
        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    common_batch_clear(batch);
    for (size_t i = 0; i < active_tokens.size(); ++i) {
        common_batch_add(batch, active_tokens[i], (llama_pos) i, { 1 }, false);
    }
    batch.logits[batch.n_tokens - 1] = true;
    if (!decode_batch(ctx, batch, "seq1-active-prefill")) {
        llama_batch_free(batch);
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::string generated;
    int32_t sample_idx = batch.n_tokens - 1;
    llama_pos pos = (llama_pos) active_tokens.size();
    int32_t n_decoded = 0;

    for (; n_decoded < n_predict; ++n_decoded) {
        const llama_token token = llama_sampler_sample(smpl, ctx, sample_idx);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        generated += common_token_to_piece(ctx, token);

        common_batch_clear(batch);
        common_batch_add(batch, token, pos++, { 1 }, true);
        sample_idx = batch.n_tokens - 1;

        if (!decode_batch(ctx, batch, "seq1-active-decode")) {
            llama_batch_free(batch);
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    printf("idle_seq=0\n");
    printf("active_seq=1\n");
    printf("idle_prompt_tokens=%zu\n", idle_tokens.size());
    printf("active_prompt_tokens=%zu\n", active_tokens.size());
    printf("decoded_tokens=%d\n", n_decoded);
    printf("active_text_begin\n");
    printf("%s%s\n", active_prompt.c_str(), generated.c_str());
    printf("active_text_end\n");

    llama_batch_free(batch);
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
