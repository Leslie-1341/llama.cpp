#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

struct llama_window_region_input {
    std::string name;
    void * addr = nullptr;
    size_t size = 0;
    uint16_t file_idx = 0;
    size_t file_offset = 0;
};

struct llama_window_params {
    bool enabled = false;
    bool use_dontneed = false;
    bool auto_tune = true;
    bool prefill_aggressive = false;
    bool debug_log = false;
    int window_size = 12;
    int prefetch_ahead = 2;
    int keep_behind = 3;
    int worker_threads = 1;
    size_t prefetch_budget = 256ull * 1024ull * 1024ull;
    size_t reclaim_budget = 128ull * 1024ull * 1024ull;
    size_t memory_limit = 0;
};

struct llama_window_stats {
    uint64_t graph_seq = 0;
    uint64_t layer_events = 0;
    uint64_t prefetch_calls = 0;
    uint64_t reclaim_calls = 0;
    uint64_t prefetch_failures = 0;
    uint64_t reclaim_failures = 0;
    uint64_t prefetch_tasks = 0;
    uint64_t reclaim_tasks = 0;
    uint64_t stale_tasks = 0;
    size_t bytes_prefetched = 0;
    size_t bytes_reclaimed = 0;
    size_t current_rss = 0;
    size_t peak_rss = 0;
};

struct llama_window_context;

std::shared_ptr<llama_window_context> llama_window_create(
        const std::vector<llama_window_region_input> & inputs,
        int n_layers,
        const llama_window_params & params);

bool llama_window_enabled(const llama_window_context * ctx);

void llama_window_graph_begin(llama_window_context & ctx, bool prefill);
void llama_window_node_done(llama_window_context & ctx, const ggml_tensor * node);
void llama_window_graph_end(llama_window_context & ctx);

const llama_window_stats & llama_window_get_stats(const llama_window_context & ctx);
