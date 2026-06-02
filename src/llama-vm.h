#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_cgraph;

enum llama_vm_block_state {
    LLAMA_VM_BLOCK_COLD,
    LLAMA_VM_BLOCK_WILLNEED,
    LLAMA_VM_BLOCK_IN_WINDOW,
    LLAMA_VM_BLOCK_RECLAIMED,
};

enum llama_vm_tensor_kind {
    LLAMA_VM_TENSOR_ATTN_Q,
    LLAMA_VM_TENSOR_ATTN_K,
    LLAMA_VM_TENSOR_ATTN_V,
    LLAMA_VM_TENSOR_ATTN_O,
    LLAMA_VM_TENSOR_ATTN_QKV,
    LLAMA_VM_TENSOR_FFN_GATE,
    LLAMA_VM_TENSOR_FFN_UP,
    LLAMA_VM_TENSOR_FFN_DOWN,
    LLAMA_VM_TENSOR_FFN_EXPERT,
    LLAMA_VM_TENSOR_NORM,
    LLAMA_VM_TENSOR_EMBED,
    LLAMA_VM_TENSOR_OUTPUT,
    LLAMA_VM_TENSOR_OTHER,
};

enum llama_vm_residency_class {
    LLAMA_VM_RESIDENCY_HOT,
    LLAMA_VM_RESIDENCY_WARM,
    LLAMA_VM_RESIDENCY_COLD,
};

struct llama_vm_region_input {
    std::string name;

    void * addr = nullptr;
    size_t size = 0;

    uint16_t file_idx = 0;
    size_t file_offset = 0;
};

struct llama_vm_params {
    bool debug_log = false;

    size_t block_size = 4ull*1024ull*1024ull;
    size_t pin_small_bytes = 2ull*1024ull*1024ull;
    size_t pin_budget_bytes = 128ull*1024ull*1024ull;
    size_t prefetch_budget_bytes = 512ull*1024ull*1024ull;
    size_t reclaim_budget_bytes = 256ull*1024ull*1024ull;

    int window_steps = 2;
    int keep_behind_steps = 2;
    int max_plan_cache_entries = 8;

    bool use_dontneed = false;
};

struct llama_vm_block {
    size_t region_id = 0;

    int layer = -1;
    int block_id = 0;

    void * addr = nullptr;
    size_t size = 0;

    void * page_addr = nullptr;
    size_t page_size = 0;

    uint16_t file_idx = 0;
    size_t file_offset = 0;

    bool pinned = false;
    bool prefetched = false;

    llama_vm_block_state state = LLAMA_VM_BLOCK_COLD;
    uint64_t last_prefetch_epoch = 0;
    uint64_t last_window_epoch = 0;
    uint64_t last_dontneed_epoch = 0;
};

struct llama_vm_addr_range {
    uintptr_t begin = 0;
    uintptr_t end = 0;
    size_t region_id = 0;
};

struct llama_vm_region {
    std::string name;

    int layer = -1;
    llama_vm_tensor_kind kind = LLAMA_VM_TENSOR_OTHER;

    void * addr = nullptr;
    size_t size = 0;

    uint16_t file_idx = 0;
    size_t file_offset = 0;

    float kind_score = 0.0f;
    float layer_score = 0.0f;
    float io_score = 0.0f;
    float wait_score = 0.0f;
    float fault_score = 0.0f;

    float static_heat = 0.0f;
    float io_cost = 0.0f;
    float stall_cost = 0.0f;
    float size_penalty = 0.0f;
    float score = 0.0f;

    float pin_priority = 0.0f;
    float prefetch_priority = 0.0f;
    float evict_priority = 0.0f;

    llama_vm_residency_class cls = LLAMA_VM_RESIDENCY_COLD;

    bool small_tensor = false;
    bool pinned = false;
    bool protected_from_reclaim = false;

    std::vector<size_t> block_ids;
};

struct llama_vm_exec_step {
    int layer = -1;
    std::vector<size_t> block_ids;
};

struct llama_vm_exec_plan {
    std::vector<llama_vm_exec_step> steps;
    size_t bytes = 0;
};

struct llama_vm_plan_cache_entry {
    uint64_t key = 0;
    llama_vm_exec_plan plan;
    uint64_t hit_count = 0;
    uint64_t last_used_epoch = 0;
};

struct llama_vm_context {
    std::vector<llama_vm_region> regions;
    std::vector<llama_vm_block> blocks;
    std::vector<std::vector<size_t>> layer_regions;
    std::unordered_map<std::string, size_t> region_by_name;
    std::vector<llama_vm_addr_range> addr_ranges;
    std::unordered_map<uint64_t, llama_vm_plan_cache_entry> plan_cache;

    llama_vm_params params;

    size_t bytes_indexed = 0;
    size_t bytes_hot = 0;
    size_t bytes_warm = 0;
    size_t bytes_cold = 0;

    size_t n_hot = 0;
    size_t n_warm = 0;
    size_t n_cold = 0;

    size_t bytes_blocked = 0;
    size_t bytes_pinned = 0;
    size_t n_pinned = 0;

    size_t n_exec_plan_logs = 0;
    size_t n_prefetch_logs = 0;
    size_t n_plan_cache_logs = 0;
    size_t n_reclaim_logs = 0;

    uint64_t epoch = 0;
    uint64_t n_plan_cache_hits = 0;
    uint64_t n_plan_cache_misses = 0;
    uint64_t n_match_by_name = 0;
    uint64_t n_match_by_addr = 0;
    uint64_t n_match_miss = 0;
    uint64_t n_dontneed = 0;
    uint64_t n_dontneed_fail = 0;
    size_t bytes_dontneed = 0;
};

std::unique_ptr<llama_vm_context> llama_vm_build_index(
        const std::vector<llama_vm_region_input> & inputs,
        uint32_t n_layer,
        const llama_vm_params & params);

llama_vm_exec_plan llama_vm_build_exec_plan(
        llama_vm_context & vm,
        const ggml_cgraph * gf);

void llama_vm_prefetch_plan(
        llama_vm_context & vm,
        const llama_vm_exec_plan & plan);

void llama_vm_on_graph_compute_begin(
        llama_vm_context & vm,
        const ggml_cgraph * gf);

void llama_vm_discard_all(
        llama_vm_context & vm);
