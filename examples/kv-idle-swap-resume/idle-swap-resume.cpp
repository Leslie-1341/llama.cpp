#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include "../../src/llama-kv-cache-stability.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
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

static uint64_t elapsed_ms_u64(perf_clock::time_point t0, perf_clock::time_point t1) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

struct active_token_stat {
    int32_t token = 0;
    bool prefetched = false;
    uint32_t requested_blocks = 0;
    int32_t restored_blocks = 0;
    double decode_ms = 0.0;
    double prefetch_ms = 0.0;
    double total_ms = 0.0;
};

enum class kv_get_rows_profile_error {
    none,
    allocation,
    callback_pairing,
    invalid_kv_node,
    duplicate_kv_node,
    incomplete_kv_nodes,
    event_overflow,
};

struct kv_get_rows_profile_event {
    uint64_t step = 0;
    uint64_t n_kv = 0;
    uint64_t row_bytes = 0;
    uint64_t segment_ending_at_get_rows_wall_us = 0;
    int32_t layer = 0;
    char kv = '\0';
    char src[GGML_MAX_NAME] = {};
};

struct kv_get_rows_profiler {
    bool enabled = false;
    kv_get_rows_profile_error error = kv_get_rows_profile_error::none;
    uint64_t step = 0;
    int32_t n_layers = 0;
    int32_t n_kv_layers = 0;
    size_t event_count = 0;
    size_t step_event_begin = 0;
    bool expected_nodes_initialized = false;
    ggml_tensor * pending_node = nullptr;
    perf_clock::time_point segment_t0;
    std::vector<kv_get_rows_profile_event> events;
    std::vector<uint8_t> step_seen;
    std::vector<uint8_t> expected_seen;

    static bool checked_add(size_t a, size_t b, size_t & result) {
        if (a > std::numeric_limits<size_t>::max() - b) {
            return false;
        }
        result = a + b;
        return true;
    }

    static bool checked_mul(size_t a, size_t b, size_t & result) {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
            return false;
        }
        result = a * b;
        return true;
    }

    bool init(int32_t model_layers, int32_t num_idle_seqs, int32_t idle_warmup,
            int32_t n_decode, bool allow_retry) {
        n_layers = model_layers;
        if (n_layers <= 0 || num_idle_seqs <= 0 || idle_warmup < 0 || n_decode < 0) {
            error = kv_get_rows_profile_error::allocation;
            return false;
        }

        size_t idle_steps = 0;
        size_t max_steps = 0;
        size_t nodes_per_step = 0;
        size_t capacity = 0;
        if (!checked_mul((size_t) num_idle_seqs, (size_t) idle_warmup + 1, idle_steps) ||
                !checked_add(idle_steps, 1, max_steps) ||
                !checked_add(max_steps, (size_t) n_decode, max_steps) ||
                !checked_add(max_steps, (size_t) n_decode, max_steps) ||
                (allow_retry && !checked_add(max_steps, 1, max_steps)) ||
                !checked_mul((size_t) n_layers, 2, nodes_per_step) ||
                !checked_mul(max_steps, nodes_per_step, capacity)) {
            error = kv_get_rows_profile_error::allocation;
            return false;
        }

        try {
            events.resize(capacity);
            step_seen.resize(nodes_per_step);
            expected_seen.resize(nodes_per_step);
        } catch (const std::bad_alloc &) {
            error = kv_get_rows_profile_error::allocation;
            return false;
        } catch (const std::length_error &) {
            error = kv_get_rows_profile_error::allocation;
            return false;
        }

        enabled = true;
        return true;
    }

    bool begin_step() {
        if (!enabled || error != kv_get_rows_profile_error::none || pending_node != nullptr) {
            if (error == kv_get_rows_profile_error::none) {
                error = kv_get_rows_profile_error::callback_pairing;
            }
            return false;
        }

        if (step == std::numeric_limits<uint64_t>::max()) {
            error = kv_get_rows_profile_error::event_overflow;
            return false;
        }
        step += 1;
        step_event_begin = event_count;
        std::fill(step_seen.begin(), step_seen.end(), 0);
        return true;
    }

    static bool parse_cache_source(const ggml_tensor * node, int32_t & layer, char & kv, const char *& src_name) {
        if (node == nullptr || node->op != GGML_OP_GET_ROWS || node->src[0] == nullptr || node->src[1] == nullptr) {
            return false;
        }

        const ggml_tensor * src = node->src[0];
        while (src != nullptr && src->op == GGML_OP_RESHAPE) {
            src = src->src[0];
        }
        if (src == nullptr) {
            return false;
        }

        const char * digits = nullptr;
        if (std::strncmp(src->name, "cache_k_l", 9) == 0) {
            kv = 'K';
            digits = src->name + 9;
        } else if (std::strncmp(src->name, "cache_v_l", 9) == 0) {
            kv = 'V';
            digits = src->name + 9;
        } else {
            return false;
        }

        if (*digits < '0' || *digits > '9') {
            return false;
        }
        int32_t parsed = 0;
        for (const char * p = digits; *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') {
                return false;
            }
            const int32_t digit = *p - '0';
            if (parsed > (std::numeric_limits<int32_t>::max() - digit) / 10) {
                return false;
            }
            parsed = parsed * 10 + digit;
        }

        layer = parsed;
        src_name = src->name;
        return true;
    }

    bool observe(ggml_tensor * node, bool ask) {
        if (error != kv_get_rows_profile_error::none) {
            return false;
        }

        if (ask) {
            int32_t layer = 0;
            char kv = '\0';
            const char * src_name = nullptr;
            if (!parse_cache_source(node, layer, kv, src_name)) {
                return false;
            }
            if (pending_node != nullptr) {
                error = kv_get_rows_profile_error::callback_pairing;
                return true;
            }
            pending_node = node;
            segment_t0 = perf_clock::now();
            return true;
        }

        const auto t1 = perf_clock::now();
        if (pending_node == nullptr || pending_node != node) {
            error = kv_get_rows_profile_error::callback_pairing;
            pending_node = nullptr;
            return false;
        }
        pending_node = nullptr;

        int32_t layer = 0;
        char kv = '\0';
        const char * src_name = nullptr;
        if (!parse_cache_source(node, layer, kv, src_name)) {
            return true;
        }
        if (layer < 0 || layer >= n_layers || node->src[1]->ne[0] <= 0 || node->src[0]->ne[0] <= 0) {
            error = kv_get_rows_profile_error::invalid_kv_node;
            return false;
        }

        const size_t seen_index = (size_t) layer * 2 + (kv == 'V' ? 1 : 0);
        if (step_seen[seen_index] != 0) {
            error = kv_get_rows_profile_error::duplicate_kv_node;
            return false;
        }
        if (event_count >= events.size()) {
            error = kv_get_rows_profile_error::event_overflow;
            return false;
        }

        step_seen[seen_index] = 1;
        kv_get_rows_profile_event & event = events[event_count++];
        event.step = step;
        event.layer = layer;
        event.kv = kv;
        event.n_kv = (uint64_t) node->src[1]->ne[0];
        event.row_bytes = (uint64_t) ggml_row_size(node->src[0]->type, node->src[0]->ne[0]);
        event.segment_ending_at_get_rows_wall_us = (uint64_t)
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - segment_t0).count();
        std::snprintf(event.src, sizeof(event.src), "%s", src_name);
        return true;
    }

    bool end_step(bool compute_succeeded) {
        if (pending_node != nullptr) {
            pending_node = nullptr;
            if (compute_succeeded && error == kv_get_rows_profile_error::none) {
                error = kv_get_rows_profile_error::callback_pairing;
            }
        }
        if (!compute_succeeded) {
            event_count = step_event_begin;
            step_event_begin = event_count;
            step -= 1;
            std::fill(step_seen.begin(), step_seen.end(), 0);
            return error == kv_get_rows_profile_error::none;
        }
        if (error != kv_get_rows_profile_error::none) {
            return false;
        }

        if (!expected_nodes_initialized) {
            for (int32_t layer = 0; layer < n_layers; ++layer) {
                const uint8_t seen_k = step_seen[(size_t) layer * 2];
                const uint8_t seen_v = step_seen[(size_t) layer * 2 + 1];
                if (seen_k != seen_v) {
                    error = kv_get_rows_profile_error::incomplete_kv_nodes;
                    return false;
                }
                if (seen_k == 1) {
                    n_kv_layers += 1;
                }
            }
            if (n_kv_layers == 0) {
                error = kv_get_rows_profile_error::incomplete_kv_nodes;
                return false;
            }
            std::copy(step_seen.begin(), step_seen.end(), expected_seen.begin());
            expected_nodes_initialized = true;
        } else if (step_seen != expected_seen) {
            error = kv_get_rows_profile_error::incomplete_kv_nodes;
            return false;
        }
        if (event_count - step_event_begin != (size_t) n_kv_layers * 2) {
            error = kv_get_rows_profile_error::incomplete_kv_nodes;
            return false;
        }
        return true;
    }

    const char * error_name() const {
        switch (error) {
            case kv_get_rows_profile_error::none:                return "none";
            case kv_get_rows_profile_error::allocation:          return "allocation";
            case kv_get_rows_profile_error::callback_pairing:    return "callback_pairing";
            case kv_get_rows_profile_error::invalid_kv_node:     return "invalid_kv_node";
            case kv_get_rows_profile_error::duplicate_kv_node:   return "duplicate_kv_node";
            case kv_get_rows_profile_error::incomplete_kv_nodes: return "incomplete_kv_nodes";
            case kv_get_rows_profile_error::event_overflow:      return "event_overflow";
        }
        return "unknown";
    }

    void print() const {
        for (size_t i = 0; i < event_count; ++i) {
            const kv_get_rows_profile_event & event = events[i];
            fprintf(stderr,
                    "KV_E2_GET_ROWS_PROFILE step=%llu layer=%d kv=%c src=%s n_kv=%llu "
                    "row_bytes=%llu segment_ending_at_get_rows_wall_us=%llu\n",
                    (unsigned long long) event.step,
                    event.layer,
                    event.kv,
                    event.src,
                    (unsigned long long) event.n_kv,
                    (unsigned long long) event.row_bytes,
                    (unsigned long long) event.segment_ending_at_get_rows_wall_us);
        }
        fprintf(stderr,
                "KV_E2_GET_ROWS_PROFILE_SUMMARY steps=%llu model_layers=%d kv_layers=%d "
                "events=%zu capacity=%zu "
                "scope=scheduler_graph_segment_from_previous_callback_boundary_through_target_get_rows_completion\n",
                (unsigned long long) step, n_layers, n_kv_layers, event_count, events.size());
    }
};

struct kv_eval_callback_chain {
    ggml_backend_sched_eval_callback original = nullptr;
    void * original_user_data = nullptr;
    kv_get_rows_profiler * profiler = nullptr;
    ggml_tensor * pending_node = nullptr;
    bool pending_original = false;
    bool pending_profiler = false;

    bool begin_step() {
        if (pending_node != nullptr) {
            profiler->error = kv_get_rows_profile_error::callback_pairing;
            return false;
        }
        return profiler->begin_step();
    }

    bool end_step(bool compute_succeeded) {
        if (compute_succeeded && pending_node != nullptr) {
            profiler->error = kv_get_rows_profile_error::callback_pairing;
        }
        pending_node = nullptr;
        pending_original = false;
        pending_profiler = false;
        return profiler->end_step(compute_succeeded);
    }

    bool observe(ggml_tensor * node, bool ask) {
        if (ask) {
            if (pending_node != nullptr) {
                profiler->error = kv_get_rows_profile_error::callback_pairing;
                return true;
            }

            const bool original_requested = original != nullptr && original(node, true, original_user_data);
            const bool profiler_requested = profiler->observe(node, true);
            if (original_requested || profiler_requested) {
                pending_node = node;
                pending_original = original_requested;
                pending_profiler = profiler_requested;
            }
            return original_requested || profiler_requested;
        }

        if (pending_node == nullptr || pending_node != node) {
            profiler->error = kv_get_rows_profile_error::callback_pairing;
            pending_node = nullptr;
            pending_original = false;
            pending_profiler = false;
            return false;
        }

        const bool call_original = pending_original;
        const bool call_profiler = pending_profiler;
        pending_node = nullptr;
        pending_original = false;
        pending_profiler = false;

        bool original_ok = true;
        bool profiler_ok = true;
        if (call_original) {
            original_ok = original(node, false, original_user_data);
        }
        if (call_profiler) {
            profiler_ok = profiler->observe(node, false);
        }
        return original_ok && profiler_ok;
    }
};

static bool kv_eval_callback_chain_cb(ggml_tensor * node, bool ask, void * user_data) {
    return static_cast<kv_eval_callback_chain *>(user_data)->observe(node, ask);
}

static double active_token_percentile(const std::vector<double> & sorted_samples, size_t percentile) {
    // Nearest-rank: for N sorted samples, percentile P selects ceil(P * N / 100), using a 1-based rank.
    const size_t rank = (percentile * sorted_samples.size() + 99) / 100;
    return sorted_samples[rank - 1];
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

static bool parse_env_u64_strict(const char * name, uint64_t fallback, uint64_t & value) {
    const char * env = std::getenv(name);
    if (env == nullptr || env[0] == '\0') {
        value = fallback;
        return true;
    }

    uint64_t parsed = 0;
    for (const char * p = env; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            fprintf(stderr, "%s: invalid %s=%s\n", __func__, name, env);
            return false;
        }

        const uint64_t digit = (uint64_t) (*p - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            fprintf(stderr, "%s: overflow in %s=%s\n", __func__, name, env);
            return false;
        }
        parsed = parsed * 10 + digit;
    }

    value = parsed;
    return true;
}

static bool parse_env_i32_nonnegative_strict(const char * name, uint64_t fallback, int32_t & value) {
    uint64_t parsed = 0;
    if (!parse_env_u64_strict(name, fallback, parsed)) {
        return false;
    }
    if (parsed > (uint64_t) std::numeric_limits<int32_t>::max()) {
        fprintf(stderr, "%s: %s=%llu exceeds int32_t max\n",
                __func__, name, (unsigned long long) parsed);
        return false;
    }
    value = (int32_t) parsed;
    return true;
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

struct kv_test_state {
    bool test_mode_enabled = false;
    bool expect_swap_out_io_failure = false;
    bool expect_prefetch_failure = false;
    bool expect_active_decode_failure = false;
    bool retry_active_decode = false;
    bool expected_prefetch_failure_seen = false;
    bool expected_active_decode_failure_seen = false;
    uint64_t decode_calls = 0;
    uint64_t prefetch_failures_observed = 0;
    uint64_t active_decode_failures_observed = 0;
    uint64_t active_decode_retries = 0;
    uint64_t active_decode_retry_successes = 0;
};

static bool parse_test_bool(const char * name, bool & value) {
    const char * env = std::getenv(name);
    if (env == nullptr || std::strcmp(env, "0") == 0) {
        value = false;
        return true;
    }
    if (std::strcmp(env, "1") == 0) {
        value = true;
        return true;
    }

    fprintf(stderr, "%s: invalid %s=%s; expected 0 or 1\n", __func__, name, env);
    return false;
}

static int decode_batch(
        llama_context * ctx,
        llama_batch & batch,
        const char * stage,
        kv_test_state & test_state,
        kv_eval_callback_chain * callback_chain) {
    if (callback_chain != nullptr && !callback_chain->begin_step()) {
        fprintf(stderr, "%s: E2 GET_ROWS profiler failed before %s: %s\n",
                __func__, stage, callback_chain->profiler->error_name());
        return -1;
    }

    int ret = llama_decode(ctx, batch);
    test_state.decode_calls += 1;
    if (callback_chain != nullptr && !callback_chain->end_step(ret == 0)) {
        fprintf(stderr, "%s: E2 GET_ROWS profiler failed during %s at step=%llu: %s\n",
                __func__, stage, (unsigned long long) callback_chain->profiler->step, callback_chain->profiler->error_name());
        ret = -1;
    }
    if (test_state.test_mode_enabled) {
        fprintf(stderr, "KV_TEST_DECODE_RESULT call_index=%llu ret=%d\n",
                (unsigned long long) test_state.decode_calls, ret);
    }
    if (ret != 0) {
        fprintf(stderr, "%s: llama_decode() failed during %s\n", __func__, stage);
    }
    return ret;
}

static void print_test_summary(const kv_test_state & test_state, bool passed) {
    if (!test_state.test_mode_enabled) {
        return;
    }

    fprintf(stderr,
            "KV_TEST_SUMMARY result=%s decode_calls=%llu expected_swap_out_io_failure=%d "
            "prefetch_failures_observed=%llu "
            "active_decode_failures_observed=%llu active_decode_retries=%llu "
            "active_decode_retry_successes=%llu\n",
            passed ? "PASS" : "FAIL",
            (unsigned long long) test_state.decode_calls,
            test_state.expect_swap_out_io_failure ? 1 : 0,
            (unsigned long long) test_state.prefetch_failures_observed,
            (unsigned long long) test_state.active_decode_failures_observed,
            (unsigned long long) test_state.active_decode_retries,
            (unsigned long long) test_state.active_decode_retry_successes);
}

static bool handle_prefetch_result(int32_t ret, kv_test_state & test_state) {
    if (ret >= 0) {
        return true;
    }

    test_state.prefetch_failures_observed += 1;
    if (ret == -1 && test_state.expect_prefetch_failure && !test_state.expected_prefetch_failure_seen) {
        test_state.expected_prefetch_failure_seen = true;
        if (test_state.test_mode_enabled) {
            fprintf(stderr, "KV_TEST_EXPECTED_PREFETCH_FAILURE\n");
        }
        return true;
    }

    print_test_summary(test_state, false);
    return false;
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

    uint64_t stability_cycles = 0;
    uint64_t stability_duration_sec = 0;
    uint64_t stability_warmup_sec = 0;
    uint64_t stability_sample_every_sec = 0;
    uint64_t stability_progress_every = 0;
    uint64_t stability_rss_limit_mb = 0;
    int32_t stability_verify_tokens = 0;
    if (!parse_env_u64_strict("LLAMA_KV_STABILITY_CYCLES", 0, stability_cycles) ||
            !parse_env_u64_strict("LLAMA_KV_STABILITY_DURATION_SEC", 0, stability_duration_sec) ||
            !parse_env_u64_strict("LLAMA_KV_STABILITY_WARMUP_SEC", 60, stability_warmup_sec) ||
            !parse_env_u64_strict("LLAMA_KV_STABILITY_SAMPLE_EVERY_SEC", 60, stability_sample_every_sec) ||
            !parse_env_u64_strict("LLAMA_KV_STABILITY_PROGRESS_EVERY", 50, stability_progress_every) ||
            !parse_env_u64_strict("LLAMA_KV_STABILITY_RSS_LIMIT_MB", 256, stability_rss_limit_mb) ||
            !parse_env_i32_nonnegative_strict("LLAMA_KV_STABILITY_VERIFY_TOKENS", 128, stability_verify_tokens)) {
        return 1;
    }
    if (stability_cycles > 0 && stability_duration_sec > 0) {
        fprintf(stderr,
                "%s: LLAMA_KV_STABILITY_CYCLES and LLAMA_KV_STABILITY_DURATION_SEC "
                "cannot both be greater than zero\n",
                __func__);
        return 1;
    }
    if (stability_sample_every_sec == 0) {
        fprintf(stderr, "%s: LLAMA_KV_STABILITY_SAMPLE_EVERY_SEC must be greater than zero\n", __func__);
        return 1;
    }
    if (stability_duration_sec > 0 && stability_duration_sec <= stability_warmup_sec) {
        fprintf(stderr,
                "%s: LLAMA_KV_STABILITY_DURATION_SEC=%llu must be greater than "
                "LLAMA_KV_STABILITY_WARMUP_SEC=%llu\n",
                __func__,
                (unsigned long long) stability_duration_sec,
                (unsigned long long) stability_warmup_sec);
        return 1;
    }
    if ((stability_duration_sec > 0 && stability_duration_sec > std::numeric_limits<uint64_t>::max() / 1000ull) ||
            stability_warmup_sec > std::numeric_limits<uint64_t>::max() / 1000ull) {
        fprintf(stderr, "%s: stability duration or warmup overflows milliseconds\n", __func__);
        return 1;
    }
    if (stability_cycles > 0 || stability_duration_sec > 0) {
        params.n_predict = stability_verify_tokens;
    }

    params.n_parallel = std::max<int32_t>(params.n_parallel, 2);
    params.kv_unified = true;
    params.sampling.backend_sampling = false;

    kv_test_state test_state;
    bool test_mode_requested = false;
    bool get_rows_profile_enabled = false;
    if (!parse_test_bool("LLAMA_KV_TEST_MODE", test_mode_requested) ||
            !parse_test_bool("LLAMA_KV_TEST_EXPECT_SWAP_OUT_IO_FAILURE", test_state.expect_swap_out_io_failure) ||
            !parse_test_bool("LLAMA_KV_TEST_EXPECT_PREFETCH_FAILURE", test_state.expect_prefetch_failure) ||
            !parse_test_bool("LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE", test_state.expect_active_decode_failure) ||
            !parse_test_bool("LLAMA_KV_TEST_RETRY_ACTIVE_DECODE", test_state.retry_active_decode) ||
            !parse_test_bool("LLAMA_KV_E2_GET_ROWS_PROFILE", get_rows_profile_enabled)) {
        return 1;
    }
    test_state.test_mode_enabled =
        test_mode_requested || test_state.expect_swap_out_io_failure ||
        test_state.expect_prefetch_failure ||
        test_state.expect_active_decode_failure ||
        test_state.retry_active_decode;
    if (test_state.retry_active_decode && !test_state.expect_active_decode_failure) {
        fprintf(stderr,
                "%s: LLAMA_KV_TEST_RETRY_ACTIVE_DECODE=1 requires "
                "LLAMA_KV_TEST_EXPECT_ACTIVE_DECODE_FAILURE=1\n",
                __func__);
        return 1;
    }

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

    kv_get_rows_profiler get_rows_profiler;
    kv_eval_callback_chain eval_callback_chain;
    if (get_rows_profile_enabled) {
        if (!get_rows_profiler.init(
                    llama_model_n_layer(model), num_idle_seqs, seq0_warmup,
                    n_decode, test_state.retry_active_decode)) {
            fprintf(stderr, "%s: failed to initialize E2 GET_ROWS profiler: %s\n",
                    __func__, get_rows_profiler.error_name());
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        eval_callback_chain.original = params.cb_eval;
        eval_callback_chain.original_user_data = params.cb_eval_user_data;
        eval_callback_chain.profiler = &get_rows_profiler;
        params.cb_eval = kv_eval_callback_chain_cb;
        params.cb_eval_user_data = &eval_callback_chain;
    }

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
    const char * active_token_stats_env = std::getenv("LLAMA_KV_ACTIVE_TOKEN_STATS");
    const bool active_token_stats_enabled =
        active_token_stats_env != nullptr && std::strcmp(active_token_stats_env, "1") == 0;
    std::vector<active_token_stat> active_token_stats;
    if (active_token_stats_enabled) {
        active_token_stats.reserve(n_decode);
    }
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
        if (decode_batch(ctx, batch, seq_id == 0 ? "seq0-idle-prefill" : "multi-idle-prefill",
                    test_state, get_rows_profile_enabled ? &eval_callback_chain : nullptr) != 0) {
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

            if (decode_batch(ctx, batch, seq_id == 0 ? "seq0-idle-warmup" : "multi-idle-warmup",
                        test_state, get_rows_profile_enabled ? &eval_callback_chain : nullptr) != 0) {
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
    if (decode_batch(ctx, batch, "seq1-active-prefill", test_state,
                get_rows_profile_enabled ? &eval_callback_chain : nullptr) != 0) {
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
        if (!handle_prefetch_result(prefetch_auto_probe_blocks, test_state)) {
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

        perf_clock::time_point active_token_t0;
        if (active_token_stats_enabled) {
            active_token_t0 = perf_clock::now();
        }
        if (decode_batch(ctx, batch, "seq1-active-decode", test_state,
                    get_rows_profile_enabled ? &eval_callback_chain : nullptr) != 0) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }
        const auto active_token_decode_t1 = active_token_stats_enabled ? perf_clock::now() : perf_clock::time_point();
        auto active_token_total_t1 = active_token_decode_t1;

        const int32_t seq1_decoded_done = seq1_decoded + 1;
        bool active_token_prefetched = false;
        uint32_t active_token_requested_blocks = 0;
        int32_t active_token_restored_blocks = 0;
        double active_token_prefetch_ms = 0.0;
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
            const auto active_prefetch_t1 = perf_clock::now();
            const double active_prefetch_ms = elapsed_ms(active_prefetch_t0, active_prefetch_t1);

            active_token_prefetched = true;
            active_token_requested_blocks = blocks_this_step;
            active_token_restored_blocks = restored;
            active_token_prefetch_ms = active_prefetch_ms;
            if (active_token_stats_enabled) {
                active_token_total_t1 = active_prefetch_t1;
            }

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

            if (!handle_prefetch_result(restored, test_state)) {
                fprintf(stderr, "%s: llama_memory_prefetch_seq_step() failed for seq0 during seq1 active decode\n", __func__);
                cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
                return 1;
            }
        }

        if (active_token_stats_enabled) {
            const active_token_stat stat = {
                seq1_decoded_done,
                active_token_prefetched,
                active_token_requested_blocks,
                active_token_restored_blocks,
                elapsed_ms(active_token_t0, active_token_decode_t1),
                active_token_prefetch_ms,
                elapsed_ms(active_token_t0, active_token_total_t1),
            };
            active_token_stats.push_back(stat);
        }
    }
    seq1_active_ms = elapsed_ms(seq1_active_t0, perf_clock::now());

    uint64_t stability_cycles_completed = 0;
    uint64_t stability_swap_out_delta = 0;
    uint64_t stability_swap_in_delta = 0;
    uint64_t stability_target_blocks = 0;
    uint64_t stability_backing_stat_valid = 0;
    uint64_t stability_backing_capacity = 0;
    uint64_t stability_backing_size_first = 0;
    uint64_t stability_backing_size_final = 0;
    uint64_t stability_backing_size_changed = 0;
    uint64_t stability_backing_blocks_baseline = 0;
    uint64_t stability_backing_blocks_final = 0;
    uint64_t stability_backing_blocks_max = 0;
    uint64_t stability_rss_baseline_kb = 0;
    uint64_t stability_rss_final_kb = 0;
    uint64_t stability_rss_max_kb = 0;
    uint64_t stability_fatal_counter_delta = 0;
    uint64_t stability_pending_error_count = 0;
    uint64_t stability_duration_requested_ms = stability_duration_sec * 1000ull;
    uint64_t stability_duration_elapsed_ms = 0;
    uint64_t stability_warmup_ms = stability_warmup_sec * 1000ull;
    uint64_t stability_cycles_at_baseline = 0;
    uint64_t stability_cycles_after_baseline = 0;
    uint64_t stability_sample_count = 0;
    uint64_t stability_last_sample_cycle = 0;
    double stability_elapsed_ms_legacy = 0.0;

    const bool stability_cycles_mode = stability_cycles > 0;
    const bool stability_duration_mode = stability_duration_sec > 0;
    if (stability_cycles_mode || stability_duration_mode) {
        llama_kv_stability_stats initial_stats;
        if (!llama_kv_cache_paged_stability_stats(llama_get_memory(ctx), 0, &initial_stats)) {
            fprintf(stderr, "%s: failed to read initial KV stability stats\n", __func__);
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }

        const uint64_t initial_swap_out = initial_stats.swap_out_calls;
        const uint64_t initial_swap_in  = initial_stats.swap_in_calls;
        const uint64_t initial_fatal    = initial_stats.fatal_counters;
        stability_target_blocks = initial_stats.target_blocks;
        stability_backing_capacity = initial_stats.backing_capacity;
        const auto stability_t0 = perf_clock::now();
        auto stability_next_sample = stability_t0 + std::chrono::seconds(stability_warmup_sec);
        bool stability_ok = true;

        for (uint64_t cycle = 1; stability_cycles_mode ? cycle <= stability_cycles : true; ++cycle) {
            llama_kv_stability_stats after_swap_out;
            llama_kv_stability_stats after_swap_in;
            const uint64_t swap_out_before = cycle == 1 ? initial_swap_out : stability_swap_out_delta + initial_swap_out;
            const uint64_t swap_in_before  = cycle == 1 ? initial_swap_in  : stability_swap_in_delta  + initial_swap_in;

            if (!llama_kv_cache_paged_stability_cycle(
                        llama_get_memory(ctx), 0, &after_swap_out, &after_swap_in)) {
                fprintf(stderr, "%s: KV stability cycle failed at cycle=%llu\n",
                        __func__, (unsigned long long) cycle);
                stability_pending_error_count = after_swap_in.pending_error + after_swap_out.pending_error;
                stability_ok = false;
                break;
            }

            if (after_swap_out.swap_out_calls <= swap_out_before ||
                    after_swap_in.swap_in_calls <= swap_in_before ||
                    after_swap_out.target_blocks == 0 ||
                    after_swap_out.swapped_blocks != after_swap_out.target_blocks ||
                    after_swap_in.resident_blocks != after_swap_out.target_blocks ||
                    after_swap_in.swapped_blocks != 0 ||
                    after_swap_in.target_blocks != after_swap_out.target_blocks ||
                    after_swap_out.pending_error != 0 ||
                    after_swap_in.pending_error != 0 ||
                    after_swap_out.backing_stat_valid == 0 ||
                    after_swap_in.backing_stat_valid == 0 ||
                    after_swap_out.backing_capacity == 0 ||
                    after_swap_out.backing_size != after_swap_out.backing_capacity ||
                    after_swap_in.backing_size != after_swap_out.backing_capacity ||
                    after_swap_in.backing_capacity != after_swap_out.backing_capacity ||
                    after_swap_out.fatal_counters != initial_fatal ||
                    after_swap_in.fatal_counters != initial_fatal) {
                fprintf(stderr,
                        "%s: KV stability assertion failed at cycle=%llu "
                        "swap_out_before=%llu swap_out_after=%llu swap_in_before=%llu swap_in_after=%llu "
                        "target_blocks=%llu swapped_after_out=%llu resident_after_in=%llu swapped_after_in=%llu "
                        "backing_valid_out=%llu backing_valid_in=%llu backing_capacity_out=%llu "
                        "backing_size_out=%llu backing_size_in=%llu pending_out=%llu pending_in=%llu "
                        "fatal_initial=%llu fatal_out=%llu fatal_in=%llu\n",
                        __func__, (unsigned long long) cycle,
                        (unsigned long long) swap_out_before,
                        (unsigned long long) after_swap_out.swap_out_calls,
                        (unsigned long long) swap_in_before,
                        (unsigned long long) after_swap_in.swap_in_calls,
                        (unsigned long long) after_swap_out.target_blocks,
                        (unsigned long long) after_swap_out.swapped_blocks,
                        (unsigned long long) after_swap_in.resident_blocks,
                        (unsigned long long) after_swap_in.swapped_blocks,
                        (unsigned long long) after_swap_out.backing_stat_valid,
                        (unsigned long long) after_swap_in.backing_stat_valid,
                        (unsigned long long) after_swap_out.backing_capacity,
                        (unsigned long long) after_swap_out.backing_size,
                        (unsigned long long) after_swap_in.backing_size,
                        (unsigned long long) after_swap_out.pending_error,
                        (unsigned long long) after_swap_in.pending_error,
                        (unsigned long long) initial_fatal,
                        (unsigned long long) after_swap_out.fatal_counters,
                        (unsigned long long) after_swap_in.fatal_counters);
                stability_pending_error_count = after_swap_in.pending_error + after_swap_out.pending_error;
                stability_ok = false;
                break;
            }

            if (cycle == 1) {
                stability_target_blocks = after_swap_out.target_blocks;
                stability_backing_stat_valid = after_swap_in.backing_stat_valid;
                stability_backing_capacity = after_swap_in.backing_capacity;
                stability_backing_size_first = after_swap_in.backing_size;
            } else if (after_swap_out.backing_size != stability_backing_size_first ||
                    after_swap_in.backing_size != stability_backing_size_first ||
                    after_swap_out.backing_capacity != stability_backing_capacity ||
                    after_swap_in.backing_capacity != stability_backing_capacity) {
                stability_backing_size_changed = 1;
                fprintf(stderr,
                        "%s: backing store changed at cycle=%llu capacity=%llu/%llu/%llu size_first=%llu size_out=%llu size_in=%llu\n",
                        __func__, (unsigned long long) cycle,
                        (unsigned long long) stability_backing_capacity,
                        (unsigned long long) after_swap_out.backing_capacity,
                        (unsigned long long) after_swap_in.backing_capacity,
                        (unsigned long long) stability_backing_size_first,
                        (unsigned long long) after_swap_out.backing_size,
                        (unsigned long long) after_swap_in.backing_size);
                stability_ok = false;
                break;
            }

            stability_cycles_completed = cycle;
            stability_swap_out_delta = after_swap_in.swap_out_calls - initial_swap_out;
            stability_swap_in_delta  = after_swap_in.swap_in_calls  - initial_swap_in;
            stability_backing_size_final = after_swap_in.backing_size;
            stability_backing_blocks_final = after_swap_in.backing_blocks_512;
            stability_backing_blocks_max = std::max<uint64_t>(
                    stability_backing_blocks_max, stability_backing_blocks_final);
            stability_fatal_counter_delta = after_swap_in.fatal_counters - initial_fatal;
            stability_pending_error_count = after_swap_in.pending_error;

            const auto now = perf_clock::now();
            if (stability_cycles_mode) {
                const uint64_t rss_now = current_rss_kb();
                stability_rss_final_kb = rss_now;
                stability_rss_max_kb = std::max<uint64_t>(stability_rss_max_kb, rss_now);
                stability_sample_count += 1;
                stability_last_sample_cycle = cycle;
                if (cycle == 10 || (stability_cycles < 10 && cycle == stability_cycles)) {
                    stability_rss_baseline_kb = rss_now;
                    stability_backing_blocks_baseline = stability_backing_blocks_final;
                    stability_cycles_at_baseline = cycle;
                }
            } else if (now >= stability_t0 + std::chrono::seconds(stability_warmup_sec)) {
                if (stability_sample_count == 0) {
                    stability_cycles_at_baseline = cycle;
                    stability_rss_baseline_kb = current_rss_kb();
                    stability_rss_final_kb = stability_rss_baseline_kb;
                    stability_rss_max_kb = stability_rss_baseline_kb;
                    stability_backing_blocks_baseline = stability_backing_blocks_final;
                    stability_backing_blocks_max = std::max<uint64_t>(
                            stability_backing_blocks_max, stability_backing_blocks_final);
                    stability_sample_count = 1;
                    stability_last_sample_cycle = cycle;
                    stability_next_sample = now + std::chrono::seconds(stability_sample_every_sec);
                } else if (now >= stability_next_sample) {
                    const uint64_t rss_now = current_rss_kb();
                    stability_rss_final_kb = rss_now;
                    stability_rss_max_kb = std::max<uint64_t>(stability_rss_max_kb, rss_now);
                    stability_backing_blocks_max = std::max<uint64_t>(
                            stability_backing_blocks_max, stability_backing_blocks_final);
                    stability_sample_count += 1;
                    stability_last_sample_cycle = cycle;
                    fprintf(stderr,
                            "KV_STABILITY_PROGRESS mode=duration cycle=%llu elapsed_ms=%llu "
                            "swap_out_delta=%llu swap_in_delta=%llu target_blocks=%llu "
                            "backing_capacity=%llu backing_size=%llu backing_blocks_512=%llu "
                            "rss_kb=%llu sample_count=%llu\n",
                            (unsigned long long) cycle,
                            (unsigned long long) elapsed_ms_u64(stability_t0, now),
                            (unsigned long long) stability_swap_out_delta,
                            (unsigned long long) stability_swap_in_delta,
                            (unsigned long long) stability_target_blocks,
                            (unsigned long long) stability_backing_capacity,
                            (unsigned long long) stability_backing_size_final,
                            (unsigned long long) stability_backing_blocks_final,
                            (unsigned long long) rss_now,
                            (unsigned long long) stability_sample_count);
                    do {
                        stability_next_sample += std::chrono::seconds(stability_sample_every_sec);
                    } while (now >= stability_next_sample);
                }
            }
            if (stability_cycles_mode && stability_progress_every > 0 && cycle % stability_progress_every == 0) {
                fprintf(stderr,
                        "KV_STABILITY_PROGRESS mode=cycles cycle=%llu swap_out_delta=%llu swap_in_delta=%llu "
                        "target_blocks=%llu backing_capacity=%llu backing_size=%llu "
                        "backing_blocks_512=%llu rss_kb=%llu\n",
                        (unsigned long long) cycle,
                        (unsigned long long) stability_swap_out_delta,
                        (unsigned long long) stability_swap_in_delta,
                        (unsigned long long) stability_target_blocks,
                        (unsigned long long) stability_backing_capacity,
                        (unsigned long long) stability_backing_size_final,
                        (unsigned long long) stability_backing_blocks_final,
                        (unsigned long long) stability_rss_final_kb);
            }

            if (stability_duration_mode &&
                    now - stability_t0 >= std::chrono::seconds(stability_duration_sec)) {
                break;
            }
            if (cycle == std::numeric_limits<uint64_t>::max()) {
                fprintf(stderr, "%s: KV stability cycle counter reached uint64_t max\n", __func__);
                stability_ok = false;
                break;
            }
        }

        const auto stability_t1 = perf_clock::now();
        stability_duration_elapsed_ms = elapsed_ms_u64(stability_t0, stability_t1);
        stability_elapsed_ms_legacy = elapsed_ms(stability_t0, stability_t1);
        if (stability_duration_mode && stability_sample_count > 0 &&
                stability_last_sample_cycle != stability_cycles_completed) {
            const uint64_t rss_now = current_rss_kb();
            stability_rss_final_kb = rss_now;
            stability_rss_max_kb = std::max<uint64_t>(stability_rss_max_kb, rss_now);
            stability_backing_blocks_max = std::max<uint64_t>(
                    stability_backing_blocks_max, stability_backing_blocks_final);
            stability_sample_count += 1;
            stability_last_sample_cycle = stability_cycles_completed;
        }
        if (stability_rss_baseline_kb == 0) {
            stability_rss_baseline_kb = stability_rss_final_kb;
        }
        if (stability_cycles_at_baseline == 0 && stability_cycles_completed > 0) {
            stability_cycles_at_baseline = stability_cycles_completed;
        }
        if (stability_backing_blocks_baseline == 0) {
            stability_backing_blocks_baseline = stability_backing_blocks_final;
        }
        stability_cycles_after_baseline =
            stability_cycles_completed > stability_cycles_at_baseline ?
            stability_cycles_completed - stability_cycles_at_baseline : 0;
        const uint64_t rss_growth_kb =
            stability_rss_final_kb > stability_rss_baseline_kb ?
            stability_rss_final_kb - stability_rss_baseline_kb : 0;
        const uint64_t rss_peak_growth_kb =
            stability_rss_max_kb > stability_rss_baseline_kb ?
            stability_rss_max_kb - stability_rss_baseline_kb : 0;
        const uint64_t rss_limit_kb =
            std::max<uint64_t>(stability_rss_limit_mb * 1024ull, stability_rss_baseline_kb / 20ull);
        if (stability_ok && rss_peak_growth_kb > rss_limit_kb) {
            fprintf(stderr,
                    "%s: KV stability peak RSS growth too high baseline_kb=%llu max_kb=%llu "
                    "peak_growth_kb=%llu limit_kb=%llu final_kb=%llu final_growth_kb=%llu\n",
                    __func__,
                    (unsigned long long) stability_rss_baseline_kb,
                    (unsigned long long) stability_rss_max_kb,
                    (unsigned long long) rss_peak_growth_kb,
                    (unsigned long long) rss_limit_kb,
                    (unsigned long long) stability_rss_final_kb,
                    (unsigned long long) rss_growth_kb);
            stability_ok = false;
        }
        if (stability_duration_mode &&
                (stability_duration_elapsed_ms < stability_duration_requested_ms ||
                 stability_cycles_completed == 0 ||
                 stability_cycles_after_baseline == 0 ||
                 stability_sample_count < 2)) {
            fprintf(stderr,
                    "%s: KV duration stability requirements failed elapsed_ms=%llu requested_ms=%llu "
                    "cycles_completed=%llu cycles_after_baseline=%llu sample_count=%llu\n",
                    __func__,
                    (unsigned long long) stability_duration_elapsed_ms,
                    (unsigned long long) stability_duration_requested_ms,
                    (unsigned long long) stability_cycles_completed,
                    (unsigned long long) stability_cycles_after_baseline,
                    (unsigned long long) stability_sample_count);
            stability_ok = false;
        }

        fprintf(stderr,
                "KV_STABILITY_SUMMARY mode=%s cycles_requested=%llu cycles_completed=%llu "
                "duration_requested_ms=%llu duration_elapsed_ms=%llu warmup_ms=%llu "
                "cycles_at_baseline=%llu cycles_after_baseline=%llu sample_count=%llu "
                "swap_out_delta=%llu swap_in_delta=%llu target_blocks=%llu "
                "backing_stat_valid=%llu backing_capacity=%llu backing_size_first=%llu "
                "backing_size_final=%llu backing_size_changed=%llu backing_blocks_512_final=%llu "
                "backing_blocks_512_baseline=%llu backing_blocks_512_max=%llu "
                "rss_baseline_kb=%llu rss_final_kb=%llu rss_max_kb=%llu rss_growth_kb=%llu "
                "rss_peak_growth_kb=%llu rss_limit_kb=%llu fatal_counter_delta=%llu "
                "pending_error_count=%llu elapsed_ms=%.3f\n",
                stability_duration_mode ? "duration" : "cycles",
                (unsigned long long) stability_cycles,
                (unsigned long long) stability_cycles_completed,
                (unsigned long long) stability_duration_requested_ms,
                (unsigned long long) stability_duration_elapsed_ms,
                (unsigned long long) stability_warmup_ms,
                (unsigned long long) stability_cycles_at_baseline,
                (unsigned long long) stability_cycles_after_baseline,
                (unsigned long long) stability_sample_count,
                (unsigned long long) stability_swap_out_delta,
                (unsigned long long) stability_swap_in_delta,
                (unsigned long long) stability_target_blocks,
                (unsigned long long) stability_backing_stat_valid,
                (unsigned long long) stability_backing_capacity,
                (unsigned long long) stability_backing_size_first,
                (unsigned long long) stability_backing_size_final,
                (unsigned long long) stability_backing_size_changed,
                (unsigned long long) stability_backing_blocks_final,
                (unsigned long long) stability_backing_blocks_baseline,
                (unsigned long long) stability_backing_blocks_max,
                (unsigned long long) stability_rss_baseline_kb,
                (unsigned long long) stability_rss_final_kb,
                (unsigned long long) stability_rss_max_kb,
                (unsigned long long) rss_growth_kb,
                (unsigned long long) rss_peak_growth_kb,
                (unsigned long long) rss_limit_kb,
                (unsigned long long) stability_fatal_counter_delta,
                (unsigned long long) stability_pending_error_count,
                stability_elapsed_ms_legacy);

        if (!stability_ok) {
            cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
            return 1;
        }
    }

    std::string seq0_resume_generated;
    llama_token seq0_token = seq0_warmed > 0 ? seq0_warm_token : seq0_resume_first;
    llama_pos seq0_pos = (llama_pos) idle_tokens.size() + seq0_warmed;
    int32_t seq0_resume_decoded = 0;

    rss_before_prefetch_kb = current_rss_kb();
    const int32_t prefetch_probe_blocks = llama_memory_prefetch_seq_step(llama_get_memory(ctx), 0, 0);
    if (!handle_prefetch_result(prefetch_probe_blocks, test_state)) {
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
        if (!handle_prefetch_result(prefetch_blocks, test_state)) {
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

        int decode_ret = decode_batch(ctx, batch, "seq0-resume-decode", test_state,
                get_rows_profile_enabled ? &eval_callback_chain : nullptr);
        if (decode_ret == -3) {
            test_state.active_decode_failures_observed += 1;
        }
        if (decode_ret != 0) {
            if (decode_ret == -3 && test_state.expect_active_decode_failure &&
                    !test_state.expected_active_decode_failure_seen) {
                test_state.expected_active_decode_failure_seen = true;
                fprintf(stderr, "KV_TEST_EXPECTED_ACTIVE_DECODE_FAILURE call_index=%llu\n",
                        (unsigned long long) test_state.decode_calls);
                if (!test_state.retry_active_decode) {
                    fprintf(stderr, "KV_TEST_TERMINATING_AFTER_EXPECTED_FAILURE\n");
                    print_test_summary(test_state, false);
                    cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
                    return 1;
                }

                test_state.active_decode_retries += 1;
                fprintf(stderr, "KV_TEST_RETRY_SAME_BATCH\n");
                decode_ret = decode_batch(ctx, batch, "seq0-resume-decode-retry", test_state,
                        get_rows_profile_enabled ? &eval_callback_chain : nullptr);
                if (decode_ret == -3) {
                    test_state.active_decode_failures_observed += 1;
                }
                if (decode_ret != 0) {
                    print_test_summary(test_state, false);
                    cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
                    return 1;
                }
                test_state.active_decode_retry_successes += 1;
                fprintf(stderr, "KV_TEST_RETRY_SUCCEEDED\n");
            } else {
                print_test_summary(test_state, false);
                cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
                return 1;
            }
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

    bool test_expectations_met = true;
    if (test_state.expect_prefetch_failure && !test_state.expected_prefetch_failure_seen) {
        fprintf(stderr, "%s: expected prefetch failure -1 was not observed\n", __func__);
        test_expectations_met = false;
    }
    if (test_state.expect_active_decode_failure && !test_state.expected_active_decode_failure_seen) {
        fprintf(stderr, "%s: expected seq0 resume decode failure -3 was not observed\n", __func__);
        test_expectations_met = false;
    }
    if (test_state.retry_active_decode &&
            (test_state.active_decode_retries != 1 || test_state.active_decode_retry_successes != 1)) {
        fprintf(stderr,
                "%s: expected exactly one active decode retry and one retry success; "
                "observed retries=%llu successes=%llu\n",
                __func__,
                (unsigned long long) test_state.active_decode_retries,
                (unsigned long long) test_state.active_decode_retry_successes);
        test_expectations_met = false;
    }
    if (!test_expectations_met) {
        print_test_summary(test_state, false);
        cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
        return 1;
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

    if (active_token_stats_enabled && !active_token_stats.empty()) {
        std::vector<double> total_samples;
        total_samples.reserve(active_token_stats.size());
        double total_ms_sum = 0.0;
        double decode_ms_sum = 0.0;
        double active_prefetch_ms_total = 0.0;
        size_t active_prefetch_calls = 0;
        for (const active_token_stat & stat : active_token_stats) {
            total_samples.push_back(stat.total_ms);
            total_ms_sum += stat.total_ms;
            decode_ms_sum += stat.decode_ms;
            if (stat.prefetched) {
                active_prefetch_calls += 1;
                active_prefetch_ms_total += stat.prefetch_ms;
                fprintf(stderr,
                        "KV_ACTIVE_TOKEN_PREFETCH token=%d requested_blocks=%u restored_blocks=%d "
                        "decode_ms=%.3f prefetch_ms=%.3f total_ms=%.3f\n",
                        stat.token, stat.requested_blocks, stat.restored_blocks,
                        stat.decode_ms, stat.prefetch_ms, stat.total_ms);
            }
        }
        std::sort(total_samples.begin(), total_samples.end());

        fprintf(stderr,
                "KV_ACTIVE_TOKEN_STATS active_token_count=%zu avg_ms=%.3f p50_ms=%.3f "
                "p95_ms=%.3f p99_ms=%.3f max_ms=%.3f decode_avg_ms=%.3f "
                "prefetch_calls=%zu prefetch_total_ms=%.3f\n",
                active_token_stats.size(),
                total_ms_sum / (double) active_token_stats.size(),
                active_token_percentile(total_samples, 50),
                active_token_percentile(total_samples, 95),
                active_token_percentile(total_samples, 99),
                total_samples.back(),
                decode_ms_sum / (double) active_token_stats.size(),
                active_prefetch_calls,
                active_prefetch_ms_total);
    }

    if (get_rows_profile_enabled) {
        get_rows_profiler.print();
    }

    const llama_perf_context_data perf = llama_perf_context(ctx);
    fprintf(stderr, "KV_GRAPH_REUSE_STATS n_reused=%d\n", perf.n_reused);

    print_test_summary(test_state, true);
    cleanup(batch, seq0_smpl, seq1_smpl, ctx, model);
    return 0;
}
