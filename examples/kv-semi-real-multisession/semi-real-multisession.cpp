#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <clocale>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

using perf_clock = std::chrono::steady_clock;

extern "C" bool llama_kv_cache_set_seq_prefetch_protected(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        bool           enabled);

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

enum class session_type {
    long_context_session,
    short_context_session,
    bursty_session,
};

enum class session_state {
    WAITING,
    PREFILL,
    ACTIVE_DECODE,
    PAUSED_IDLE,
    RESUME_PENDING,
    RESUMING,
    FINISHED,
};

static const char * session_type_name(session_type type) {
    switch (type) {
        case session_type::long_context_session:
            return "long_context_session";
        case session_type::short_context_session:
            return "short_context_session";
        case session_type::bursty_session:
            return "bursty_session";
    }

    return "unknown";
}

static const char * session_state_name(session_state state) {
    switch (state) {
        case session_state::WAITING:
            return "WAITING";
        case session_state::PREFILL:
            return "PREFILL";
        case session_state::ACTIVE_DECODE:
            return "ACTIVE_DECODE";
        case session_state::PAUSED_IDLE:
            return "PAUSED_IDLE";
        case session_state::RESUME_PENDING:
            return "RESUME_PENDING";
        case session_state::RESUMING:
            return "RESUMING";
        case session_state::FINISHED:
            return "FINISHED";
    }

    return "UNKNOWN";
}

struct semi_session {
    const char * name;
    llama_seq_id seq_id;
    session_type type;
    std::string prompt;
    int32_t target_decode_tokens;
    session_state state = session_state::WAITING;
    int32_t decoded_tokens = 0;
    bool resumed = false;
    bool resume_first_done = false;
    bool prefetch_protected = false;
    double resume_first_ms = 0.0;
    std::vector<llama_token> prompt_tokens = {};
    common_sampler * sampler = nullptr;
    llama_token next_token = LLAMA_TOKEN_NULL;
    llama_pos pos = 0;
};

static void print_usage(int, char ** argv) {
    fprintf(stderr, "\nexample usage:\n");
    fprintf(stderr, "\n    %s -m model.gguf -n 24 --ctx-size 2048 --parallel 4\n", argv[0]);
    fprintf(stderr, "\n");
}

static void log_session_event(const semi_session & s, int phase, const char * event) {
    fprintf(stderr,
            "KV_SEMI_SESSION event=%s phase=%d seq=%d name=%s type=%s state=%s decoded=%d target=%d rss_kb=%llu\n",
            event,
            phase,
            (int) s.seq_id,
            s.name,
            session_type_name(s.type),
            session_state_name(s.state),
            s.decoded_tokens,
            s.target_decode_tokens,
            (unsigned long long) current_rss_kb());
}

static bool decode_batch(llama_context * ctx, llama_batch & batch, const char * stage) {
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: llama_decode() failed during %s\n", __func__, stage);
        return false;
    }
    return true;
}

static int32_t parse_env_i32_or_default(const char * name, int32_t default_value, bool require_positive) {
    const char * value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }

    errno = 0;
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed > INT32_MAX || parsed < INT32_MIN ||
            (require_positive ? parsed <= 0 : parsed < 0)) {
        fprintf(stderr, "%s: warning: invalid %s=%s, using %d\n", __func__, name, value, default_value);
        return default_value;
    }

    return (int32_t) parsed;
}

static bool read_corpus_file(const char * path, std::string & out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fprintf(stderr, "%s: failed to open corpus file: %s\n", __func__, path);
        return false;
    }

    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (!file.good() && !file.eof()) {
        fprintf(stderr, "%s: failed to read corpus file: %s\n", __func__, path);
        return false;
    }

    return true;
}

static std::string clean_corpus_text(const std::string & input) {
    std::string cleaned;
    cleaned.reserve(input.size());

    bool in_space = true;
    for (unsigned char ch : input) {
        if (std::isspace(ch)) {
            if (!in_space) {
                cleaned.push_back(' ');
                in_space = true;
            }
        } else {
            cleaned.push_back((char) ch);
            in_space = false;
        }
    }

    if (!cleaned.empty() && cleaned.back() == ' ') {
        cleaned.pop_back();
    }

    return cleaned;
}

static size_t utf8_advance_to_boundary(const std::string & text, size_t pos) {
    while (pos < text.size() && (((unsigned char) text[pos] & 0xc0) == 0x80)) {
        ++pos;
    }
    return pos;
}

static size_t utf8_retreat_to_boundary(const std::string & text, size_t pos) {
    pos = std::min(pos, text.size());
    while (pos > 0 && pos < text.size() && (((unsigned char) text[pos] & 0xc0) == 0x80)) {
        --pos;
    }
    return pos;
}

static std::string corpus_chunk(const std::string & corpus, size_t raw_start, size_t chars) {
    if (raw_start >= corpus.size()) {
        return {};
    }

    size_t start = utf8_advance_to_boundary(corpus, raw_start);
    if (start > 0 && corpus[start - 1] != ' ') {
        const size_t next_space = corpus.find(' ', start);
        if (next_space == std::string::npos) {
            return {};
        }
        start = next_space + 1;
    }
    while (start < corpus.size() && corpus[start] == ' ') {
        ++start;
    }
    start = utf8_advance_to_boundary(corpus, start);
    if (start >= corpus.size()) {
        return {};
    }

    const size_t hard_end = utf8_retreat_to_boundary(corpus, std::min(corpus.size(), start + chars));
    size_t end = hard_end;
    if (hard_end < corpus.size()) {
        const size_t prev_space = corpus.rfind(' ', hard_end);
        if (prev_space != std::string::npos && prev_space > start) {
            end = prev_space;
        }
    }

    if (end <= start) {
        end = hard_end;
    }
    while (end > start && corpus[end - 1] == ' ') {
        --end;
    }

    if (end <= start) {
        return {};
    }

    return corpus.substr(start, end - start);
}

static void cleanup(std::vector<semi_session> & sessions, llama_batch & batch, llama_context * ctx, llama_model * model) {
    llama_batch_free(batch);
    for (semi_session & s : sessions) {
        common_sampler_free(s.sampler);
        s.sampler = nullptr;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_predict = 24;
    params.n_parallel = 4;
    params.kv_unified = true;
    params.sampling.seed = 1;
    params.sampling.temp = 0.0f;
    params.sampling.backend_sampling = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BATCHED, print_usage)) {
        return 1;
    }

    params.n_parallel = std::max<int32_t>(params.n_parallel, 4);
    params.kv_unified = true;
    params.sampling.backend_sampling = false;

    const int32_t default_n_decode = params.n_predict < 0 ? 24 : params.n_predict;

    std::vector<semi_session> sessions = {
        {
            "A",
            0,
            session_type::long_context_session,
            "Long context session. The request contains a stable synthetic incident report with many details. "
            "A team maintains a service that accepts chat traffic from several departments. The first department "
            "asks for a careful summary of deployment events, capacity notes, escalation messages, recovery steps, "
            "and verification items. The report repeats concrete but deterministic facts so the prompt creates a "
            "larger KV footprint. Deployment window one started at midnight, warmed caches, checked routing tables, "
            "confirmed queue depth, measured memory pressure, reviewed swap activity, and recorded latency notes. "
            "Deployment window two compared old and new workers, watched request fanout, checked rate limits, "
            "reviewed retry storms, and wrote a concise operational timeline. The answer should preserve the order "
            "of events and avoid inventing any external data.",
            std::max<int32_t>(default_n_decode, 20),
        },
        {
            "B",
            1,
            session_type::short_context_session,
            "Short context session. Answer briefly: name two practical checks before restarting a service.",
            std::max<int32_t>(default_n_decode / 2, 8),
        },
        {
            "C",
            2,
            session_type::bursty_session,
            "Bursty session. A user asks for a compact plan, pauses, then returns. Include setup, observation, "
            "decision, and cleanup steps in a deterministic order.",
            std::max<int32_t>(default_n_decode / 2, 12),
        },
        {
            "D",
            3,
            session_type::short_context_session,
            "Short context session. Give a terse checklist for validating logs after a change.",
            std::max<int32_t>(default_n_decode / 2, 8),
        },
    };

    const char * corpus_file_env = std::getenv("LLAMA_KV_SEMI_CORPUS_FILE");
    const bool corpus_enabled = corpus_file_env != nullptr && corpus_file_env[0] != '\0';
    int32_t corpus_chars = 1024;
    int32_t corpus_offset = 0;
    if (corpus_enabled) {
        corpus_chars = parse_env_i32_or_default("LLAMA_KV_SEMI_CORPUS_CHARS", 1024, true);
        corpus_offset = parse_env_i32_or_default("LLAMA_KV_SEMI_CORPUS_OFFSET", 0, false);

        std::string corpus_raw;
        if (!read_corpus_file(corpus_file_env, corpus_raw)) {
            return 1;
        }

        const std::string corpus = clean_corpus_text(corpus_raw);
        if (corpus.empty()) {
            fprintf(stderr, "%s: corpus file is empty after cleaning: %s\n", __func__, corpus_file_env);
            return 1;
        }

        for (size_t i = 0; i < sessions.size(); ++i) {
            const size_t raw_start = (size_t) corpus_offset + i * (size_t) corpus_chars;
            std::string chunk = corpus_chunk(corpus, raw_start, (size_t) corpus_chars);
            if (chunk.empty()) {
                fprintf(stderr,
                        "%s: corpus is too short to provide non-empty chunk for session %s "
                        "(file=%s chars=%d offset=%d)\n",
                        __func__, sessions[i].name, corpus_file_env, corpus_chars, corpus_offset);
                return 1;
            }
            sessions[i].prompt = std::move(chunk);
        }
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
        fprintf(stderr, "%s: encoder-decoder models are not supported by this driver\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    size_t max_prompt_tokens = 1;
    int32_t n_kv_req = 64;
    for (semi_session & s : sessions) {
        s.prompt_tokens = common_tokenize(vocab, s.prompt, true);
        if (s.prompt_tokens.empty()) {
            fprintf(stderr, "%s: tokenization produced an empty prompt for session %s\n", __func__, s.name);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        max_prompt_tokens = std::max(max_prompt_tokens, s.prompt_tokens.size());
        n_kv_req += (int32_t) s.prompt_tokens.size() + s.target_decode_tokens + 8;
    }

    if (corpus_enabled) {
        fprintf(stderr,
                "KV_SEMI_CORPUS enabled=1 file=%s chars=%d offset=%d "
                "A_chars=%zu B_chars=%zu C_chars=%zu D_chars=%zu "
                "A_tokens=%zu B_tokens=%zu C_tokens=%zu D_tokens=%zu\n",
                corpus_file_env,
                corpus_chars,
                corpus_offset,
                sessions[0].prompt.size(),
                sessions[1].prompt.size(),
                sessions[2].prompt.size(),
                sessions[3].prompt.size(),
                sessions[0].prompt_tokens.size(),
                sessions[1].prompt_tokens.size(),
                sessions[2].prompt_tokens.size(),
                sessions[3].prompt_tokens.size());
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.n_ctx     = std::max<int32_t>(ctx_params.n_ctx, n_kv_req);
    ctx_params.n_batch   = std::max<int32_t>(ctx_params.n_batch,  (int32_t) max_prompt_tokens);
    ctx_params.n_ubatch  = std::max<int32_t>(ctx_params.n_ubatch, (int32_t) max_prompt_tokens);
    ctx_params.n_seq_max = std::max<uint32_t>(ctx_params.n_seq_max, (uint32_t) sessions.size());
    ctx_params.kv_unified = true;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create llama_context\n", __func__);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    for (semi_session & s : sessions) {
        s.sampler = common_sampler_init(model, params.sampling);
        if (s.sampler == nullptr) {
            fprintf(stderr, "%s: failed to create sampler for session %s\n", __func__, s.name);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    llama_batch batch = llama_batch_init((int32_t) max_prompt_tokens, 0, (int32_t) ctx_params.n_seq_max);
    const auto total_t0 = perf_clock::now();
    double active_decode_ms = 0.0;
    int32_t active_decode_tokens = 0;
    const auto prefill_session = [&](semi_session & s, int phase) -> bool {
        s.state = session_state::PREFILL;
        log_session_event(s, phase, "prefill_begin");

        common_batch_clear(batch);
        for (size_t i = 0; i < s.prompt_tokens.size(); ++i) {
            common_batch_add(batch, s.prompt_tokens[i], (llama_pos) i, { s.seq_id }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (!decode_batch(ctx, batch, "semi-prefill")) {
            return false;
        }

        s.next_token = common_sampler_sample(s.sampler, ctx, batch.n_tokens - 1);
        common_sampler_accept(s.sampler, s.next_token, true);
        if (llama_vocab_is_eog(vocab, s.next_token)) {
            fprintf(stderr,
                    "%s: session %s prefill sampled EOG; choose a different prompt/model for this driver\n",
                    __func__, s.name);
            return false;
        }
        s.pos = (llama_pos) s.prompt_tokens.size();
        log_session_event(s, phase, "prefill_done");
        return true;
    };

    const auto decode_some = [&](semi_session & s, int phase, int32_t n_tokens, bool resume) -> bool {
        if (s.state == session_state::FINISHED || n_tokens <= 0) {
            return true;
        }

        s.state = resume ? session_state::RESUMING : session_state::ACTIVE_DECODE;
        log_session_event(s, phase, resume ? "resume_decode_begin" : "active_decode_begin");
        if (resume && !s.resumed) {
            s.resumed = true;
            fprintf(stderr, "KV_SEMI_RESUME_BEGIN seq=%d name=%s\n", (int) s.seq_id, s.name);
        }

        for (int32_t i = 0; i < n_tokens && s.decoded_tokens < s.target_decode_tokens; ++i) {
            if (llama_vocab_is_eog(vocab, s.next_token)) {
                s.state = session_state::FINISHED;
                log_session_event(s, phase, "finished_eog");
                return true;
            }

            const bool first_resume_token = resume && !s.resume_first_done;
            const auto step_t0 = first_resume_token ? perf_clock::now() : perf_clock::time_point{};

            common_batch_clear(batch);
            common_batch_add(batch, s.next_token, s.pos++, { s.seq_id }, true);
            batch.logits[batch.n_tokens - 1] = true;

            const auto decode_t0 = perf_clock::now();
            if (!decode_batch(ctx, batch, resume ? "semi-resume-decode" : "semi-active-decode")) {
                return false;
            }
            active_decode_ms += elapsed_ms(decode_t0, perf_clock::now());
            active_decode_tokens += 1;

            s.decoded_tokens += 1;
            s.next_token = common_sampler_sample(s.sampler, ctx, batch.n_tokens - 1);
            common_sampler_accept(s.sampler, s.next_token, true);

            if (first_resume_token) {
                s.resume_first_done = true;
                s.resume_first_ms = elapsed_ms(step_t0, perf_clock::now());
                fprintf(stderr,
                        "KV_SEMI_RESUME_FIRST_DONE seq=%d name=%s first_ms=%.3f\n",
                        (int) s.seq_id, s.name, s.resume_first_ms);
            }

        }

        if (s.decoded_tokens >= s.target_decode_tokens) {
            s.state = session_state::FINISHED;
            log_session_event(s, phase, "finished_target");
        } else {
            log_session_event(s, phase, resume ? "resume_decode_pause" : "active_decode_pause");
        }

        return true;
    };

    const auto pause_session = [&](semi_session & s, int phase) {
        if (s.state != session_state::FINISHED) {
            s.state = session_state::PAUSED_IDLE;
            log_session_event(s, phase, "paused_idle");
        }
    };

    const auto start_resume_pending = [&](semi_session & s, int phase) -> bool {
        s.state = session_state::RESUME_PENDING;
        s.prefetch_protected = llama_kv_cache_set_seq_prefetch_protected(llama_get_memory(ctx), s.seq_id, true);
        log_session_event(s, phase, "resume_pending");

        const auto prefetch_t0 = perf_clock::now();
        const int32_t prefetch_blocks = llama_memory_prefetch_seq(llama_get_memory(ctx), s.seq_id);
        const double prefetch_ms = elapsed_ms(prefetch_t0, perf_clock::now());
        if (prefetch_blocks < 0) {
            fprintf(stderr,
                    "%s: llama_memory_prefetch_seq() failed for seq=%d\n",
                    __func__, (int) s.seq_id);
            return false;
        }
        fprintf(stderr,
                "KV_SEMI_PREFETCH_EXACT phase=%d seq=%d restored=%d elapsed_ms=%.3f\n",
                phase,
                (int) s.seq_id,
                prefetch_blocks,
                prefetch_ms);
        log_session_event(s, phase, "prefetch");
        return true;
    };

    const auto clear_prefetch_protection = [&](semi_session & s, int phase) {
        if (s.prefetch_protected) {
            llama_kv_cache_set_seq_prefetch_protected(llama_get_memory(ctx), s.seq_id, false);
            s.prefetch_protected = false;
            log_session_event(s, phase, "prefetch_unprotect");
        }
    };

    // phase 0: A prefill + active decode.
    if (!prefill_session(sessions[0], 0) ||
            !decode_some(sessions[0], 0, std::min<int32_t>(10, sessions[0].target_decode_tokens), false)) {
        cleanup(sessions, batch, ctx, model);
        return 1;
    }

    // phase 1: A idle, B active, C burst starts then idles.
    pause_session(sessions[0], 1);
    if (!prefill_session(sessions[1], 1) ||
            !decode_some(sessions[1], 1, std::min<int32_t>(8, sessions[1].target_decode_tokens), false) ||
            !prefill_session(sessions[2], 1) ||
            !decode_some(sessions[2], 1, std::min<int32_t>(4, sessions[2].target_decode_tokens), false)) {
        cleanup(sessions, batch, ctx, model);
        return 1;
    }
    pause_session(sessions[2], 1);

    // phase 2: A becomes resume-pending while C resumes for a short burst; B goes idle.
    if (!start_resume_pending(sessions[0], 2)) {
        cleanup(sessions, batch, ctx, model);
        return 1;
    }
    pause_session(sessions[1], 2);
    if (!decode_some(sessions[2], 2, std::min<int32_t>(6, sessions[2].target_decode_tokens - sessions[2].decoded_tokens), true)) {
        cleanup(sessions, batch, ctx, model);
        return 1;
    }
    pause_session(sessions[2], 2);

    // phase 3: A resumes, B resumes after pending, D runs, and C drains to finished.
    if (!decode_some(sessions[0], 3, sessions[0].target_decode_tokens - sessions[0].decoded_tokens, true)) {
        cleanup(sessions, batch, ctx, model);
        return 1;
    }
    clear_prefetch_protection(sessions[0], 3);

    if (!start_resume_pending(sessions[1], 3) ||
            !prefill_session(sessions[3], 3) ||
            !decode_some(sessions[3], 3, sessions[3].target_decode_tokens, false) ||
            !decode_some(sessions[1], 3, sessions[1].target_decode_tokens - sessions[1].decoded_tokens, true)) {
        cleanup(sessions, batch, ctx, model);
        return 1;
    }
    clear_prefetch_protection(sessions[1], 3);

    if (!decode_some(sessions[2], 3, sessions[2].target_decode_tokens - sessions[2].decoded_tokens, true)) {
        cleanup(sessions, batch, ctx, model);
        return 1;
    }

    const double total_wall_ms = elapsed_ms(total_t0, perf_clock::now());
    const double active_tps = active_decode_ms > 0.0 ?
        1000.0 * (double) active_decode_tokens / active_decode_ms : 0.0;

    for (const semi_session & s : sessions) {
        fprintf(stderr,
                "KV_SEMI_SUMMARY seq=%d name=%s type=%s decoded=%d resumed=%d finished=%d resume_first_ms=%.3f\n",
                (int) s.seq_id,
                s.name,
                session_type_name(s.type),
                s.decoded_tokens,
                s.resumed ? 1 : 0,
                s.state == session_state::FINISHED ? 1 : 0,
                s.resume_first_ms);
    }

    fprintf(stderr,
            "KV_SEMI_PERF total_wall_ms=%.3f active_tps=%.6f active_decode_tokens=%d active_decode_ms=%.3f rss_kb=%llu\n",
            total_wall_ms,
            active_tps,
            active_decode_tokens,
            active_decode_ms,
            (unsigned long long) current_rss_kb());
    fprintf(stderr,
            "KV_SEMI_DRIVER role=lifecycle_driver prefetch_capability=exact_prefetch "
            "restore_validation=not_claimed_without_explicit_offload\n");

    cleanup(sessions, batch, ctx, model);
    return 0;
}
