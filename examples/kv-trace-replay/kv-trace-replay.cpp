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
#include <limits>
#include <map>
#include <sstream>
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

enum class session_state {
    WAITING,
    PREFILL,
    ACTIVE_DECODE,
    PAUSED_IDLE,
    RESUME_PENDING,
    RESUMING,
    FINISHED,
};

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

struct trace_turn {
    int32_t session_id = 0;
    int32_t turn_id = 0;
    int64_t arrival_ms = 0;
    std::string prompt_source;
    std::string prompt;
    int32_t target_decode_tokens = 0;
    int64_t idle_ms = 0;
    int line_no = 0;
    std::vector<llama_token> prompt_tokens = {};
};

struct trace_session {
    llama_seq_id seq_id = 0;
    std::vector<trace_turn> turns = {};
    int32_t cur_turn_idx = 0;
    session_state state = session_state::WAITING;
    int32_t decoded_this_turn = 0;
    int32_t decoded_total = 0;
    llama_pos pos = 0;
    int32_t resumed = 0;
    bool resume_first_pending = false;
    bool prefetch_protected = false;
    std::vector<double> resume_first_ms = {};
    int64_t idle_ms_total = 0;
    common_sampler * sampler = nullptr;
    llama_token next_token = LLAMA_TOKEN_NULL;
};

struct replay_config {
    std::string trace_file;
    std::string corpus_file;
    int64_t tick_ms = 10;
    bool prefetch_during_active = false;
    int32_t prefetch_auto_every_tokens = 4;
    int32_t prefetch_auto_blocks_per_step = 1;
    int32_t prefetch_auto_safety_tokens = 0;
    int32_t prefetch_final_sync_blocks = 0;
    bool defer_swapout_on_resume = false;
};

static void print_usage(int, char ** argv) {
    fprintf(stderr, "\nexample usage:\n");
    fprintf(stderr,
            "\n    LLAMA_KV_TRACE_FILE=examples/kv-trace-replay/traces/smoke_4s2t.tsv "
            "LLAMA_KV_TRACE_CORPUS_FILE=corpus.txt %s -m model.gguf --ctx-size 4096 --parallel 4\n",
            argv[0]);
    fprintf(stderr, "\n");
}

static void trace_error(int line, const char * reason) {
    fprintf(stderr, "KV_TRACE_ERROR line=%d reason=%s\n", line, reason);
}

static void trace_error_str(int line, const std::string & reason) {
    fprintf(stderr, "KV_TRACE_ERROR line=%d reason=%s\n", line, reason.c_str());
}

static std::string trim_ascii_space(const std::string & input) {
    size_t begin = 0;
    while (begin < input.size() && std::isspace((unsigned char) input[begin])) {
        ++begin;
    }

    size_t end = input.size();
    while (end > begin && std::isspace((unsigned char) input[end - 1])) {
        --end;
    }

    return input.substr(begin, end - begin);
}

static bool parse_i64_field(const std::string & text, int64_t min_value, int64_t max_value, int64_t & out) {
    if (text.empty()) {
        return false;
    }

    errno = 0;
    char * end = nullptr;
    const long long parsed = std::strtoll(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str() || *end != '\0' || parsed < min_value || parsed > max_value) {
        return false;
    }

    out = (int64_t) parsed;
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

static bool read_text_file(const char * path, std::string & out) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }

    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return file.good() || file.eof();
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

static std::vector<std::string> split_tsv_line(const std::string & line) {
    std::vector<std::string> fields;
    size_t begin = 0;
    while (true) {
        const size_t tab = line.find('\t', begin);
        if (tab == std::string::npos) {
            fields.push_back(line.substr(begin));
            break;
        }
        fields.push_back(line.substr(begin, tab - begin));
        begin = tab + 1;
    }
    return fields;
}

static bool prompt_source_needs_corpus(const std::string & source) {
    return source.rfind("corpus:", 0) == 0;
}

static bool resolve_prompt_source(
        const std::string & source,
        const std::string & corpus,
        std::string &       prompt,
        std::string &       reason) {
    if (source.rfind("text:", 0) == 0) {
        prompt = trim_ascii_space(source.substr(strlen("text:")));
        if (prompt.empty()) {
            reason = "empty text prompt";
            return false;
        }
        return true;
    }

    if (source.rfind("corpus:", 0) == 0) {
        const std::string rest = source.substr(strlen("corpus:"));
        const size_t sep = rest.find(':');
        if (sep == std::string::npos || rest.find(':', sep + 1) != std::string::npos) {
            reason = "invalid corpus prompt_source";
            return false;
        }

        int64_t offset = 0;
        int64_t chars = 0;
        if (!parse_i64_field(rest.substr(0, sep), 0, INT64_MAX, offset) ||
                !parse_i64_field(rest.substr(sep + 1), 1, INT64_MAX, chars)) {
            reason = "invalid corpus offset or chars";
            return false;
        }

        if ((uint64_t) offset >= (uint64_t) corpus.size()) {
            reason = "corpus slice out of range";
            return false;
        }

        prompt = corpus_chunk(corpus, (size_t) offset, (size_t) chars);
        if (prompt.empty()) {
            reason = "empty corpus slice";
            return false;
        }
        return true;
    }

    reason = "unsupported prompt_source";
    return false;
}

static bool load_trace_file(
        const std::string & trace_file,
        const std::string & corpus,
        std::vector<trace_session> & sessions,
        int32_t & turns_total) {
    std::ifstream file(trace_file);
    if (!file) {
        trace_error(0, "trace file unreadable");
        return false;
    }

    std::map<int32_t, trace_session> by_session;
    std::map<int32_t, int32_t> next_turn_by_session;
    std::map<int32_t, int64_t> last_arrival_by_session;
    std::string line;
    int line_no = 0;
    turns_total = 0;

    while (std::getline(file, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        const std::string trimmed = trim_ascii_space(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }

        const std::vector<std::string> fields = split_tsv_line(line);
        if (fields.size() != 6) {
            trace_error(line_no, "wrong column count");
            return false;
        }

        int64_t session_id64 = 0;
        int64_t turn_id64 = 0;
        int64_t arrival_ms = 0;
        int64_t target_decode_tokens64 = 0;
        int64_t idle_ms = 0;
        if (!parse_i64_field(fields[0], 0, INT32_MAX, session_id64) ||
                !parse_i64_field(fields[1], 0, INT32_MAX, turn_id64) ||
                !parse_i64_field(fields[2], 0, INT64_MAX, arrival_ms) ||
                !parse_i64_field(fields[4], 1, INT32_MAX, target_decode_tokens64) ||
                !parse_i64_field(fields[5], 0, INT64_MAX, idle_ms)) {
            trace_error(line_no, "invalid integer field");
            return false;
        }

        const int32_t session_id = (int32_t) session_id64;
        const int32_t turn_id = (int32_t) turn_id64;
        const auto next_it = next_turn_by_session.find(session_id);
        const int32_t expected_turn = next_it == next_turn_by_session.end() ? 0 : next_it->second;
        if (turn_id != expected_turn) {
            trace_error(line_no, "turn_id not contiguous");
            return false;
        }

        const auto last_arrival_it = last_arrival_by_session.find(session_id);
        if (last_arrival_it != last_arrival_by_session.end() && arrival_ms < last_arrival_it->second) {
            trace_error(line_no, "arrival_ms reversed within session");
            return false;
        }

        trace_turn turn;
        turn.session_id = session_id;
        turn.turn_id = turn_id;
        turn.arrival_ms = arrival_ms;
        turn.prompt_source = fields[3];
        turn.target_decode_tokens = (int32_t) target_decode_tokens64;
        turn.idle_ms = idle_ms;
        turn.line_no = line_no;

        std::string reason;
        if (!resolve_prompt_source(turn.prompt_source, corpus, turn.prompt, reason)) {
            trace_error_str(line_no, reason);
            return false;
        }

        trace_session & session = by_session[session_id];
        session.seq_id = session_id;
        session.turns.push_back(std::move(turn));
        next_turn_by_session[session_id] = turn_id + 1;
        last_arrival_by_session[session_id] = arrival_ms;
        turns_total += 1;
    }

    if (!file.good() && !file.eof()) {
        trace_error(0, "trace file read failed");
        return false;
    }
    if (turns_total == 0) {
        trace_error(0, "trace file has no turns");
        return false;
    }
    if (by_session.size() > (size_t) INT32_MAX) {
        trace_error(0, "session_id out of supported range");
        return false;
    }

    sessions.clear();
    sessions.reserve(by_session.size());
    for (auto & kv : by_session) {
        sessions.push_back(std::move(kv.second));
    }

    return true;
}

static bool trace_uses_corpus(const std::string & trace_file, bool & uses_corpus) {
    std::ifstream file(trace_file);
    if (!file) {
        trace_error(0, "trace file unreadable");
        return false;
    }

    std::string line;
    int line_no = 0;
    uses_corpus = false;
    while (std::getline(file, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string trimmed = trim_ascii_space(line);
        if (trimmed.empty() || trimmed[0] == '#') {
            continue;
        }
        const std::vector<std::string> fields = split_tsv_line(line);
        if (fields.size() != 6) {
            trace_error(line_no, "wrong column count");
            return false;
        }
        if (prompt_source_needs_corpus(fields[3])) {
            uses_corpus = true;
        }
    }

    if (!file.good() && !file.eof()) {
        trace_error(0, "trace file read failed");
        return false;
    }

    return true;
}

static void log_event(
        int64_t now_ms,
        const trace_session & s,
        const trace_turn & turn,
        const char * event) {
    fprintf(stderr,
            "KV_TRACE_EVENT now_ms=%lld seq=%d turn=%d state=%s event=%s decoded=%d target=%d rss_kb=%llu\n",
            (long long) now_ms,
            (int) s.seq_id,
            turn.turn_id,
            session_state_name(s.state),
            event,
            s.decoded_this_turn,
            turn.target_decode_tokens,
            (unsigned long long) current_rss_kb());
}

static void log_idle_status(int64_t now_ms, const std::vector<trace_session> & sessions) {
    int32_t idle_sessions = 0;
    int32_t active_sessions = 0;
    int32_t waiting_turns = 0;
    int32_t paused_sessions = 0;
    int32_t resume_pending = 0;

    for (const trace_session & s : sessions) {
        if (s.state == session_state::PAUSED_IDLE) {
            idle_sessions += 1;
            paused_sessions += 1;
        } else if (s.state == session_state::RESUME_PENDING) {
            resume_pending += 1;
        } else if (s.state == session_state::ACTIVE_DECODE || s.state == session_state::RESUMING ||
                s.state == session_state::PREFILL) {
            active_sessions += 1;
        }

        if (s.cur_turn_idx < (int32_t) s.turns.size()) {
            const trace_turn & turn = s.turns[s.cur_turn_idx];
            if ((turn.turn_id == 0 && s.state == session_state::WAITING) ||
                    (turn.turn_id > 0 && s.state == session_state::PAUSED_IDLE)) {
                waiting_turns += 1;
            }
        }
    }

    fprintf(stderr,
            "KV_TRACE_IDLE now_ms=%lld idle_sessions=%d active_sessions=%d waiting_turns=%d paused_sessions=%d resume_pending=%d\n",
            (long long) now_ms,
            idle_sessions,
            active_sessions,
            waiting_turns,
            paused_sessions,
            resume_pending);
}

static bool decode_batch(llama_context * ctx, llama_batch & batch, const char * stage) {
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "%s: llama_decode() failed during %s\n", __func__, stage);
        return false;
    }
    return true;
}

static void cleanup(std::vector<trace_session> & sessions, llama_batch & batch, llama_context * ctx, llama_model * model) {
    llama_batch_free(batch);
    for (trace_session & s : sessions) {
        common_sampler_free(s.sampler);
        s.sampler = nullptr;
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    const char * trace_file_env = std::getenv("LLAMA_KV_TRACE_FILE");
    if (trace_file_env == nullptr || trace_file_env[0] == '\0') {
        trace_error(0, "LLAMA_KV_TRACE_FILE is required");
        return 1;
    }

    replay_config config;
    config.trace_file = trace_file_env;
    const char * corpus_file_env = std::getenv("LLAMA_KV_TRACE_CORPUS_FILE");
    if (corpus_file_env != nullptr) {
        config.corpus_file = corpus_file_env;
    }

    bool uses_corpus = false;
    if (!trace_uses_corpus(config.trace_file, uses_corpus)) {
        return 1;
    }

    std::string corpus;
    if (uses_corpus) {
        if (config.corpus_file.empty()) {
            trace_error(0, "LLAMA_KV_TRACE_CORPUS_FILE is required");
            return 1;
        }

        std::string corpus_raw;
        if (!read_text_file(config.corpus_file.c_str(), corpus_raw)) {
            trace_error(0, "corpus file unreadable");
            return 1;
        }

        corpus = clean_corpus_text(corpus_raw);
        if (corpus.empty()) {
            trace_error(0, "corpus file empty after cleaning");
            return 1;
        }
    }

    std::vector<trace_session> sessions;
    int32_t turns_total = 0;
    if (!load_trace_file(config.trace_file, corpus, sessions, turns_total)) {
        return 1;
    }

    config.prefetch_during_active =
        std::getenv("LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE") != nullptr &&
        std::atoi(std::getenv("LLAMA_KV_PAGED_PREFETCH_DURING_ACTIVE")) != 0;
    config.prefetch_auto_every_tokens =
        parse_env_i32_or_default("LLAMA_KV_PAGED_PREFETCH_AUTO_EVERY_TOKENS", 4, true);
    config.prefetch_auto_blocks_per_step =
        parse_env_i32_or_default("LLAMA_KV_PAGED_PREFETCH_AUTO_BLOCKS_PER_STEP", 1, true);
    config.prefetch_auto_safety_tokens =
        parse_env_i32_or_default("LLAMA_KV_PAGED_PREFETCH_AUTO_SAFETY_TOKENS", 0, false);
    config.prefetch_final_sync_blocks =
        parse_env_i32_or_default("LLAMA_KV_PAGED_PREFETCH_FINAL_SYNC_BLOCKS", 0, false);
    config.defer_swapout_on_resume =
        std::getenv("LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME") != nullptr &&
        std::atoi(std::getenv("LLAMA_KV_PAGED_DEFER_SWAPOUT_ON_RESUME")) != 0;

    common_params params;
    params.n_predict = 24;
    params.n_parallel = std::max<int32_t>(4, (int32_t) sessions.size());
    params.kv_unified = true;
    params.sampling.seed = 1;
    params.sampling.temp = 0.0f;
    params.sampling.backend_sampling = false;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BATCHED, print_usage)) {
        return 1;
    }

    params.n_parallel = std::max<int32_t>(params.n_parallel, (int32_t) sessions.size());
    params.kv_unified = true;
    params.sampling.backend_sampling = false;

    fprintf(stderr,
            "KV_TRACE_CONFIG trace_file=%s corpus_file=%s sessions=%zu turns=%d tick_ms=%lld\n",
            config.trace_file.c_str(),
            config.corpus_file.c_str(),
            sessions.size(),
            turns_total,
            (long long) config.tick_ms);
    fprintf(stderr,
            "KV_TRACE_PREFETCH_CONFIG during_active=%d every=%d blocks_per_step=%d safety=%d defer=%d final_sync_blocks=%d\n",
            config.prefetch_during_active ? 1 : 0,
            config.prefetch_auto_every_tokens,
            config.prefetch_auto_blocks_per_step,
            config.prefetch_auto_safety_tokens,
            config.defer_swapout_on_resume ? 1 : 0,
            config.prefetch_final_sync_blocks);

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
    for (trace_session & s : sessions) {
        for (trace_turn & turn : s.turns) {
            turn.prompt_tokens = common_tokenize(vocab, turn.prompt, true);
            if (turn.prompt_tokens.empty()) {
                trace_error(turn.line_no, "tokenization produced empty prompt");
                llama_model_free(model);
                llama_backend_free();
                return 1;
            }
            max_prompt_tokens = std::max(max_prompt_tokens, turn.prompt_tokens.size());
            n_kv_req += (int32_t) turn.prompt_tokens.size() + turn.target_decode_tokens + 8;
        }
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

    for (trace_session & s : sessions) {
        s.sampler = common_sampler_init(model, params.sampling);
        if (s.sampler == nullptr) {
            fprintf(stderr, "%s: failed to create sampler for seq=%d\n", __func__, (int) s.seq_id);
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
    int64_t prefetch_step_calls = 0;
    int64_t prefetch_step_blocks = 0;
    int64_t final_prefetch_calls = 0;
    int64_t final_prefetch_blocks = 0;
    int64_t now_ms = 0;
    size_t rr_next = 0;

    const auto prefetch_pending_step = [&]() -> bool {
        if (!config.prefetch_during_active) {
            return true;
        }
        if (active_decode_tokens <= 0 || active_decode_tokens % config.prefetch_auto_every_tokens != 0) {
            return true;
        }

        for (trace_session & s : sessions) {
            if (s.state != session_state::RESUME_PENDING || !s.prefetch_protected) {
                continue;
            }

            const auto step_prefetch_t0 = perf_clock::now();
            const int32_t restored = llama_memory_prefetch_seq_step(
                    llama_get_memory(ctx), s.seq_id, config.prefetch_auto_blocks_per_step);
            const double step_prefetch_ms = elapsed_ms(step_prefetch_t0, perf_clock::now());
            if (restored < 0) {
                fprintf(stderr,
                        "%s: llama_memory_prefetch_seq_step() failed for seq=%d\n",
                        __func__, (int) s.seq_id);
                return false;
            }
            prefetch_step_calls += 1;
            if (restored > 0) {
                prefetch_step_blocks += restored;
            }
            fprintf(stderr,
                    "KV_TRACE_PREFETCH_STEP now_ms=%lld seq=%d restored=%d calls=%lld blocks=%lld elapsed_ms=%.3f\n",
                    (long long) now_ms,
                    (int) s.seq_id,
                    restored,
                    (long long) prefetch_step_calls,
                    (long long) prefetch_step_blocks,
                    step_prefetch_ms);
        }
        return true;
    };

    const auto run_final_prefetch_before_resume = [&](trace_session & s) -> bool {
        if (config.prefetch_final_sync_blocks <= 0) {
            return true;
        }

        int32_t restored_total = 0;
        int32_t calls = 0;
        const auto final_prefetch_t0 = perf_clock::now();
        while (restored_total < config.prefetch_final_sync_blocks) {
            const int32_t remaining = config.prefetch_final_sync_blocks - restored_total;
            const int32_t requested = std::min(config.prefetch_auto_blocks_per_step, remaining);
            const int32_t restored = llama_memory_prefetch_seq_step(llama_get_memory(ctx), s.seq_id, requested);
            if (restored < 0) {
                fprintf(stderr,
                        "%s: llama_memory_prefetch_seq_step() final prefetch failed for seq=%d\n",
                        __func__, (int) s.seq_id);
                return false;
            }

            calls += 1;
            final_prefetch_calls += 1;
            if (restored > 0) {
                restored_total += restored;
                final_prefetch_blocks += restored;
            }
            if (restored == 0) {
                break;
            }
        }

        fprintf(stderr,
                "KV_TRACE_PREFETCH_FINAL now_ms=%lld seq=%d requested=%d restored=%d calls=%d blocks=%lld rss_kb=%llu elapsed_ms=%.3f\n",
                (long long) now_ms,
                (int) s.seq_id,
                config.prefetch_final_sync_blocks,
                restored_total,
                calls,
                (long long) final_prefetch_blocks,
                (unsigned long long) current_rss_kb(),
                elapsed_ms(final_prefetch_t0, perf_clock::now()));
        return true;
    };

    const auto clear_prefetch_protection = [&](trace_session & s) {
        if (s.prefetch_protected) {
            llama_kv_cache_set_seq_prefetch_protected(llama_get_memory(ctx), s.seq_id, false);
            s.prefetch_protected = false;
        }
    };

    const auto prefill_turn = [&](trace_session & s, trace_turn & turn, bool resume) -> bool {
        if (resume && !run_final_prefetch_before_resume(s)) {
            return false;
        }

        s.state = resume ? session_state::RESUMING : session_state::PREFILL;
        log_event(now_ms, s, turn, resume ? "RESUME_PREFILL_BEGIN" : "PREFILL_BEGIN");

        const auto prefill_t0 = perf_clock::now();
        common_batch_clear(batch);
        for (llama_token token : turn.prompt_tokens) {
            common_batch_add(batch, token, s.pos++, { s.seq_id }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        if (!decode_batch(ctx, batch, resume ? "trace-resume-prefill" : "trace-prefill")) {
            return false;
        }

        s.next_token = common_sampler_sample(s.sampler, ctx, batch.n_tokens - 1);
        common_sampler_accept(s.sampler, s.next_token, true);
        if (llama_vocab_is_eog(vocab, s.next_token)) {
            fprintf(stderr,
                    "%s: seq=%d turn=%d prefill sampled EOG; choose a different prompt/model for this driver\n",
                    __func__, (int) s.seq_id, turn.turn_id);
            return false;
        }

        if (resume) {
            s.resume_first_pending = true;
        }
        fprintf(stderr,
                "KV_TRACE_EVENT now_ms=%lld seq=%d turn=%d state=%s event=%s decoded=%d target=%d rss_kb=%llu prefill_ms=%.3f\n",
                (long long) now_ms,
                (int) s.seq_id,
                turn.turn_id,
                session_state_name(s.state),
                resume ? "RESUME_PREFILL_END" : "PREFILL_END",
                s.decoded_this_turn,
                turn.target_decode_tokens,
                (unsigned long long) current_rss_kb(),
                elapsed_ms(prefill_t0, perf_clock::now()));
        s.state = resume ? session_state::RESUMING : session_state::ACTIVE_DECODE;
        log_event(now_ms, s, turn, "DECODE_BEGIN");
        return true;
    };

    const auto start_resume_pending = [&](trace_session & s, trace_turn & turn) -> bool {
        s.state = session_state::RESUME_PENDING;
        s.prefetch_protected = llama_kv_cache_set_seq_prefetch_protected(llama_get_memory(ctx), s.seq_id, true);
        s.resumed += 1;
        log_event(now_ms, s, turn, "RESUME_PENDING");

        const int32_t probe_blocks = llama_memory_prefetch_seq_step(llama_get_memory(ctx), s.seq_id, 0);
        if (probe_blocks < 0) {
            fprintf(stderr,
                    "%s: llama_memory_prefetch_seq_step() probe failed for seq=%d\n",
                    __func__, (int) s.seq_id);
            return false;
        }

        if (!config.prefetch_during_active && config.prefetch_final_sync_blocks > 0) {
            const auto sync_prefetch_t0 = perf_clock::now();
            const int32_t prefetch_blocks = llama_memory_prefetch_seq(llama_get_memory(ctx), s.seq_id);
            const double sync_prefetch_ms = elapsed_ms(sync_prefetch_t0, perf_clock::now());
            if (prefetch_blocks < 0) {
                fprintf(stderr,
                        "%s: llama_memory_prefetch_seq() failed for seq=%d\n",
                        __func__, (int) s.seq_id);
                return false;
            }
            fprintf(stderr,
                    "KV_TRACE_PREFETCH_FINAL now_ms=%lld seq=%d requested=-1 restored=%d calls=1 blocks=%d rss_kb=%llu elapsed_ms=%.3f\n",
                    (long long) now_ms,
                    (int) s.seq_id,
                    prefetch_blocks,
                    prefetch_blocks,
                    (unsigned long long) current_rss_kb(),
                    sync_prefetch_ms);
        }

        return true;
    };

    const auto decode_one = [&](trace_session & s, trace_turn & turn) -> bool {
        if (llama_vocab_is_eog(vocab, s.next_token)) {
            s.state = session_state::FINISHED;
            log_event(now_ms, s, turn, "FINISH");
            return true;
        }

        const bool first_resume_token = s.resume_first_pending;
        const auto step_t0 = first_resume_token ? perf_clock::now() : perf_clock::time_point{};
        if (first_resume_token && config.defer_swapout_on_resume) {
            llama_kv_cache_defer_idle_swapout(llama_get_memory(ctx), 1);
        }

        common_batch_clear(batch);
        common_batch_add(batch, s.next_token, s.pos++, { s.seq_id }, true);
        batch.logits[batch.n_tokens - 1] = true;

        const auto decode_t0 = perf_clock::now();
        if (!decode_batch(ctx, batch, s.state == session_state::RESUMING ? "trace-resume-decode" : "trace-active-decode")) {
            return false;
        }
        active_decode_ms += elapsed_ms(decode_t0, perf_clock::now());
        active_decode_tokens += 1;

        s.decoded_this_turn += 1;
        s.decoded_total += 1;

        if (s.decoded_this_turn < turn.target_decode_tokens) {
            s.next_token = common_sampler_sample(s.sampler, ctx, batch.n_tokens - 1);
            common_sampler_accept(s.sampler, s.next_token, true);
        }

        if (first_resume_token) {
            s.resume_first_pending = false;
            const double first_ms = elapsed_ms(step_t0, perf_clock::now());
            s.resume_first_ms.push_back(first_ms);
            fprintf(stderr,
                    "KV_TRACE_EVENT now_ms=%lld seq=%d turn=%d state=%s event=RESUME_FIRST_TOKEN decoded=%d target=%d rss_kb=%llu first_ms=%.3f\n",
                    (long long) now_ms,
                    (int) s.seq_id,
                    turn.turn_id,
                    session_state_name(s.state),
                    s.decoded_this_turn,
                    turn.target_decode_tokens,
                    (unsigned long long) current_rss_kb(),
                    first_ms);
            clear_prefetch_protection(s);
        }

        now_ms += config.tick_ms;

        if (!prefetch_pending_step()) {
            return false;
        }

        if (s.decoded_this_turn >= turn.target_decode_tokens) {
            log_event(now_ms, s, turn, "TURN_DONE");
            s.cur_turn_idx += 1;
            s.decoded_this_turn = 0;
            s.idle_ms_total += turn.idle_ms;
            if (s.cur_turn_idx >= (int32_t) s.turns.size()) {
                s.state = session_state::FINISHED;
                log_event(now_ms, s, turn, "FINISH");
            } else {
                s.state = session_state::PAUSED_IDLE;
                log_event(now_ms, s, turn, "PAUSE_IDLE");
            }
            log_idle_status(now_ms, sessions);
        }

        return true;
    };

    const auto all_finished = [&]() -> bool {
        for (const trace_session & s : sessions) {
            if (s.state != session_state::FINISHED) {
                return false;
            }
        }
        return true;
    };

    const auto next_arrival_ms = [&]() -> int64_t {
        int64_t next = INT64_MAX;
        for (const trace_session & s : sessions) {
            if (s.cur_turn_idx >= (int32_t) s.turns.size()) {
                continue;
            }
            const trace_turn & turn = s.turns[s.cur_turn_idx];
            if ((turn.turn_id == 0 && s.state == session_state::WAITING) ||
                    (turn.turn_id > 0 && s.state == session_state::PAUSED_IDLE)) {
                next = std::min(next, turn.arrival_ms);
            }
        }
        return next;
    };

    const auto dispatch_arrivals = [&]() -> bool {
        bool changed = false;
        for (trace_session & s : sessions) {
            if (s.cur_turn_idx >= (int32_t) s.turns.size()) {
                continue;
            }

            trace_turn & turn = s.turns[s.cur_turn_idx];
            if (turn.arrival_ms > now_ms) {
                continue;
            }

            if (turn.turn_id == 0 && s.state == session_state::WAITING) {
                log_event(now_ms, s, turn, "TURN_ARRIVE");
                if (!prefill_turn(s, turn, false)) {
                    return false;
                }
                changed = true;
            } else if (turn.turn_id > 0 && s.state == session_state::PAUSED_IDLE) {
                log_event(now_ms, s, turn, "TURN_ARRIVE");
                if (!start_resume_pending(s, turn)) {
                    return false;
                }
                changed = true;
            }
        }
        if (changed) {
            log_idle_status(now_ms, sessions);
        }
        return true;
    };

    const auto select_runnable = [&]() -> int32_t {
        if (sessions.empty()) {
            return -1;
        }
        for (size_t probe = 0; probe < sessions.size(); ++probe) {
            const size_t idx = (rr_next + probe) % sessions.size();
            trace_session & s = sessions[idx];
            if (s.cur_turn_idx >= (int32_t) s.turns.size()) {
                continue;
            }
            const session_state state = s.state;
            if (state == session_state::ACTIVE_DECODE || state == session_state::RESUMING ||
                    state == session_state::RESUME_PENDING) {
                rr_next = (idx + 1) % sessions.size();
                return (int32_t) idx;
            }
        }
        return -1;
    };

    log_idle_status(now_ms, sessions);
    while (!all_finished()) {
        if (!dispatch_arrivals()) {
            cleanup(sessions, batch, ctx, model);
            return 1;
        }

        const int32_t runnable_idx = select_runnable();
        if (runnable_idx < 0) {
            const int64_t next = next_arrival_ms();
            if (next == INT64_MAX) {
                break;
            }
            now_ms = std::max(now_ms, next);
            log_idle_status(now_ms, sessions);
            continue;
        }

        trace_session & s = sessions[(size_t) runnable_idx];
        trace_turn & turn = s.turns[s.cur_turn_idx];
        if (s.state == session_state::RESUME_PENDING) {
            if (!prefill_turn(s, turn, true)) {
                cleanup(sessions, batch, ctx, model);
                return 1;
            }
        }

        if (!decode_one(s, turn)) {
            cleanup(sessions, batch, ctx, model);
            return 1;
        }
    }

    const double total_wall_ms = elapsed_ms(total_t0, perf_clock::now());
    const double active_tps = active_decode_ms > 0.0 ?
        1000.0 * (double) active_decode_tokens / active_decode_ms : 0.0;

    for (const trace_session & s : sessions) {
        double resume_first_sum = 0.0;
        for (const double v : s.resume_first_ms) {
            resume_first_sum += v;
        }
        const double resume_first_avg_ms = s.resume_first_ms.empty() ? 0.0 :
            resume_first_sum / (double) s.resume_first_ms.size();
        fprintf(stderr,
                "KV_TRACE_SUMMARY seq=%d turns=%zu decoded=%d resumed=%d finished=%d idle_ms_total=%lld resume_first_avg_ms=%.3f\n",
                (int) s.seq_id,
                s.turns.size(),
                s.decoded_total,
                s.resumed,
                s.state == session_state::FINISHED ? 1 : 0,
                (long long) s.idle_ms_total,
                resume_first_avg_ms);
    }

    fprintf(stderr,
            "KV_TRACE_PERF total_wall_ms=%.3f active_tps=%.6f active_decode_tokens=%d active_decode_ms=%.3f rss_kb=%llu\n",
            total_wall_ms,
            active_tps,
            active_decode_tokens,
            active_decode_ms,
            (unsigned long long) current_rss_kb());

    cleanup(sessions, batch, ctx, model);
    return 0;
}
