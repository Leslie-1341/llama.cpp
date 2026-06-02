#include "llama-vm.h"

#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static constexpr size_t llama_vm_default_pin_budget = 512ull*1024ull*1024ull;
static constexpr int    llama_vm_default_warm_top_n_per_layer = 3;

static constexpr float llama_vm_lambda_io    = 0.20f;
static constexpr float llama_vm_lambda_stall = 0.30f;
static constexpr float llama_vm_gamma        = 0.05f;
static constexpr float llama_vm_alpha        = 0.15f;
static constexpr float llama_vm_beta         = 0.20f;

static int llama_vm_parse_layer(const char * name) {
    if (std::strncmp(name, "blk.", 4) != 0) {
        return -1;
    }

    const char * cur = name + 4;
    char * end = nullptr;
    const long layer = std::strtol(cur, &end, 10);
    if (end == cur || *end != '.' || layer < 0) {
        return -1;
    }

    return (int) layer;
}

static bool llama_vm_name_has(const char * name, const char * needle) {
    return std::strstr(name, needle) != nullptr;
}

static llama_vm_tensor_kind llama_vm_parse_kind(const char * name) {
    if (std::strcmp(name, "token_embd.weight") == 0 || llama_vm_name_has(name, "embed")) {
        return LLAMA_VM_TENSOR_EMBED;
    }
    if (std::strncmp(name, "output.", 7) == 0 || llama_vm_name_has(name, ".output.")) {
        return LLAMA_VM_TENSOR_OUTPUT;
    }

    if (llama_vm_name_has(name, ".attn_qkv.")) {
        return LLAMA_VM_TENSOR_ATTN_QKV;
    }
    if (llama_vm_name_has(name, ".attn_q.")) {
        return LLAMA_VM_TENSOR_ATTN_Q;
    }
    if (llama_vm_name_has(name, ".attn_k.")) {
        return LLAMA_VM_TENSOR_ATTN_K;
    }
    if (llama_vm_name_has(name, ".attn_v.")) {
        return LLAMA_VM_TENSOR_ATTN_V;
    }
    if (llama_vm_name_has(name, ".attn_output.") || llama_vm_name_has(name, ".attn_o.")) {
        return LLAMA_VM_TENSOR_ATTN_O;
    }

    if (llama_vm_name_has(name, ".ffn_gate_exps.") ||
            llama_vm_name_has(name, ".ffn_up_exps.") ||
            llama_vm_name_has(name, ".ffn_down_exps.") ||
            llama_vm_name_has(name, ".ffn_gate_up_exps.")) {
        return LLAMA_VM_TENSOR_FFN_EXPERT;
    }
    if (llama_vm_name_has(name, ".ffn_gate.") || llama_vm_name_has(name, ".ffn_gate_shexp.")) {
        return LLAMA_VM_TENSOR_FFN_GATE;
    }
    if (llama_vm_name_has(name, ".ffn_up.") || llama_vm_name_has(name, ".ffn_up_shexp.")) {
        return LLAMA_VM_TENSOR_FFN_UP;
    }
    if (llama_vm_name_has(name, ".ffn_down.") || llama_vm_name_has(name, ".ffn_down_shexp.")) {
        return LLAMA_VM_TENSOR_FFN_DOWN;
    }

    if (llama_vm_name_has(name, "norm")) {
        return LLAMA_VM_TENSOR_NORM;
    }

    return LLAMA_VM_TENSOR_OTHER;
}

static float llama_vm_kind_score(llama_vm_tensor_kind kind) {
    switch (kind) {
        case LLAMA_VM_TENSOR_EMBED:      return 1.15f;
        case LLAMA_VM_TENSOR_OUTPUT:     return 1.15f;
        case LLAMA_VM_TENSOR_NORM:       return 1.05f;
        case LLAMA_VM_TENSOR_ATTN_QKV:   return 1.00f;
        case LLAMA_VM_TENSOR_ATTN_Q:
        case LLAMA_VM_TENSOR_ATTN_K:
        case LLAMA_VM_TENSOR_ATTN_V:
        case LLAMA_VM_TENSOR_ATTN_O:     return 0.95f;
        case LLAMA_VM_TENSOR_FFN_GATE:
        case LLAMA_VM_TENSOR_FFN_UP:
        case LLAMA_VM_TENSOR_FFN_DOWN:   return 0.90f;
        case LLAMA_VM_TENSOR_FFN_EXPERT: return 0.80f;
        case LLAMA_VM_TENSOR_OTHER:      return 0.45f;
    }

    return 0.45f;
}

static float llama_vm_layer_score(int layer, uint32_t n_layer) {
    if (layer < 0 || n_layer == 0) {
        return 0.55f;
    }

    const float pos = n_layer > 1 ? float(layer) / float(n_layer - 1) : 0.0f;

    // Keep the middle layers hot, while slightly favoring early and final layers.
    const float middle = 1.0f - std::fabs(2.0f*pos - 1.0f);
    const float edges = std::max(1.0f - 4.0f*pos, 4.0f*pos - 3.0f);
    return 0.70f + 0.25f*middle + 0.15f*std::max(0.0f, edges);
}

static float llama_vm_io_score(size_t size) {
    if (size == 0) {
        return 0.0f;
    }

    constexpr float mib = 1024.0f*1024.0f;
    return std::min(1.0f, std::log2(1.0f + float(size)/mib) / 8.0f);
}

static float llama_vm_log_mib(size_t size) {
    constexpr float mib = 1024.0f*1024.0f;
    return std::log2(1.0f + float(size)/mib);
}

static float llama_vm_size_mib(size_t size) {
    constexpr float mib = 1024.0f*1024.0f;
    return float(size)/mib;
}

static size_t llama_vm_align_up(size_t value, size_t align) {
    if (align == 0) {
        return value;
    }
    return (value + align - 1) & ~(align - 1);
}

static size_t llama_vm_page_size() {
#if defined(__linux__) || defined(__APPLE__)
    const long page = sysconf(_SC_PAGESIZE);
    return page > 0 ? (size_t) page : 4096;
#else
    return 4096;
#endif
}

static void llama_vm_page_range(void * addr, size_t size, void ** page_addr, size_t * page_size) {
    const size_t page = llama_vm_page_size();
    const uintptr_t begin = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t aligned = begin & ~(uintptr_t(page) - 1);
    const size_t delta = begin - aligned;

    *page_addr = reinterpret_cast<void *>(aligned);
    *page_size = llama_vm_align_up(delta + size, page);
}

static int llama_vm_madvise(void * addr, size_t size, int advice) {
#if defined(__linux__) || defined(__APPLE__)
    return madvise(addr, size, advice);
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(size);
    GGML_UNUSED(advice);
    return 0;
#endif
}

static int llama_vm_mlock(void * addr, size_t size) {
#if defined(__linux__) || defined(__APPLE__)
    return mlock(addr, size);
#else
    GGML_UNUSED(addr);
    GGML_UNUSED(size);
    return -1;
#endif
}

static bool llama_vm_region_protected_from_reclaim(const llama_vm_region & region) {
    if (region.pinned || region.small_tensor || region.layer < 0) {
        return true;
    }

    switch (region.kind) {
        case LLAMA_VM_TENSOR_NORM:
        case LLAMA_VM_TENSOR_EMBED:
        case LLAMA_VM_TENSOR_OUTPUT:
            return true;
        default:
            return false;
    }
}

static bool llama_vm_is_small_tensor(const llama_vm_region & region, size_t pin_small_bytes) {
    if (region.size <= pin_small_bytes) {
        return true;
    }

    switch (region.kind) {
        case LLAMA_VM_TENSOR_NORM:
            return true;
        default:
            return false;
    }
}

static void llama_vm_compute_scores(llama_vm_region & region) {
    const float log_size = llama_vm_log_mib(region.size);
    const float size_mib = llama_vm_size_mib(region.size);

    region.static_heat = 0.65f*region.kind_score + 0.35f*region.layer_score;
    region.io_cost = log_size;
    region.stall_cost = region.wait_score + region.fault_score;
    region.size_penalty = llama_vm_gamma*log_size;

    region.score = region.static_heat +
            llama_vm_lambda_io*region.io_cost +
            llama_vm_lambda_stall*region.stall_cost -
            region.size_penalty;

    region.pin_priority = region.score / std::max(size_mib, 0.25f);
    region.prefetch_priority = region.score + llama_vm_alpha*log_size;
    region.evict_priority = -region.score + llama_vm_beta*log_size;
}

static void llama_vm_build_blocks(llama_vm_context & vm) {
    const size_t block_size = std::max<size_t>(vm.params.block_size, llama_vm_page_size());

    for (size_t region_id = 0; region_id < vm.regions.size(); ++region_id) {
        auto & region = vm.regions[region_id];
        region.small_tensor = llama_vm_is_small_tensor(region, vm.params.pin_small_bytes);

        const size_t n_blocks = region.small_tensor ? 1 : (region.size + block_size - 1) / block_size;
        region.block_ids.reserve(n_blocks);

        for (size_t i = 0; i < n_blocks; ++i) {
            const size_t offset = region.small_tensor ? 0 : i * block_size;
            const size_t size = region.small_tensor ? region.size : std::min(block_size, region.size - offset);

            llama_vm_block block;
            block.region_id = region_id;
            block.layer = region.layer;
            block.block_id = (int) i;
            block.addr = static_cast<uint8_t *>(region.addr) + offset;
            block.size = size;
            block.file_idx = region.file_idx;
            block.file_offset = region.file_offset + offset;
            llama_vm_page_range(block.addr, block.size, &block.page_addr, &block.page_size);

            const size_t block_id = vm.blocks.size();
            region.block_ids.push_back(block_id);
            vm.blocks.emplace_back(block);
            vm.bytes_blocked += size;
        }

        region.protected_from_reclaim = llama_vm_region_protected_from_reclaim(region);
    }
}

static void llama_vm_pin_small_blocks(llama_vm_context & vm) {
    if (vm.params.pin_budget_bytes == 0) {
        return;
    }

    std::vector<size_t> ids;
    ids.reserve(vm.regions.size());
    for (size_t i = 0; i < vm.regions.size(); ++i) {
        if (vm.regions[i].small_tensor) {
            ids.push_back(i);
        }
    }

    std::sort(ids.begin(), ids.end(), [&](size_t a, size_t b) {
        return vm.regions[a].pin_priority > vm.regions[b].pin_priority;
    });

    size_t used = 0;
    for (const size_t region_id : ids) {
        auto & region = vm.regions[region_id];
        if (region.block_ids.empty()) {
            continue;
        }

        const size_t block_id = region.block_ids.front();
        auto & block = vm.blocks[block_id];
        if (used + block.page_size > vm.params.pin_budget_bytes) {
            continue;
        }

        if (llama_vm_mlock(block.page_addr, block.page_size) == 0) {
            block.pinned = true;
            block.prefetched = true;
            block.state = LLAMA_VM_BLOCK_IN_WINDOW;
            region.pinned = true;
            region.protected_from_reclaim = true;
            used += block.page_size;
            vm.bytes_pinned += block.size;
            vm.n_pinned++;
        } else if (vm.params.debug_log) {
            LLAMA_LOG_DEBUG("llama_vm: mlock failed for %-56s %.2f MiB\n",
                    region.name.c_str(), llama_vm_size_mib(block.page_size));
        }
    }
}

static const char * llama_vm_class_name(llama_vm_residency_class cls) {
    switch (cls) {
        case LLAMA_VM_RESIDENCY_HOT:  return "HOT";
        case LLAMA_VM_RESIDENCY_WARM: return "WARM";
        case LLAMA_VM_RESIDENCY_COLD: return "COLD";
    }

    return "UNKNOWN";
}

static const char * llama_vm_kind_name(llama_vm_tensor_kind kind) {
    switch (kind) {
        case LLAMA_VM_TENSOR_ATTN_Q:     return "ATTN_Q";
        case LLAMA_VM_TENSOR_ATTN_K:     return "ATTN_K";
        case LLAMA_VM_TENSOR_ATTN_V:     return "ATTN_V";
        case LLAMA_VM_TENSOR_ATTN_O:     return "ATTN_O";
        case LLAMA_VM_TENSOR_ATTN_QKV:   return "ATTN_QKV";
        case LLAMA_VM_TENSOR_FFN_GATE:   return "FFN_GATE";
        case LLAMA_VM_TENSOR_FFN_UP:     return "FFN_UP";
        case LLAMA_VM_TENSOR_FFN_DOWN:   return "FFN_DOWN";
        case LLAMA_VM_TENSOR_FFN_EXPERT: return "FFN_EXPERT";
        case LLAMA_VM_TENSOR_NORM:       return "NORM";
        case LLAMA_VM_TENSOR_EMBED:      return "EMBED";
        case LLAMA_VM_TENSOR_OUTPUT:     return "OUTPUT";
        case LLAMA_VM_TENSOR_OTHER:      return "OTHER";
    }

    return "UNKNOWN";
}

static int llama_vm_class_index(llama_vm_residency_class cls) {
    switch (cls) {
        case LLAMA_VM_RESIDENCY_HOT:  return 0;
        case LLAMA_VM_RESIDENCY_WARM: return 1;
        case LLAMA_VM_RESIDENCY_COLD: return 2;
    }

    return 2;
}

static void llama_vm_mark_hot_by_pin_budget(llama_vm_context & vm, size_t pin_budget) {
    if (pin_budget == 0 || vm.regions.empty()) {
        return;
    }

    std::vector<size_t> ids(vm.regions.size());
    std::iota(ids.begin(), ids.end(), 0);

    std::sort(ids.begin(), ids.end(), [&](size_t a, size_t b) {
        return vm.regions[a].pin_priority > vm.regions[b].pin_priority;
    });

    size_t used = 0;
    std::vector<size_t> layer_used(vm.layer_regions.size(), 0);
    const size_t base_per_layer = pin_budget / std::max<size_t>(1, vm.layer_regions.size());
    const size_t layer_cap = std::max<size_t>(base_per_layer*2, 1);

    for (const size_t id : ids) {
        auto & region = vm.regions[id];
        if (used + region.size > pin_budget) {
            continue;
        }

        const bool has_layer = region.layer >= 0 && (size_t) region.layer < vm.layer_regions.size();
        if (has_layer && layer_used[region.layer] + region.size > layer_cap) {
            continue;
        }

        region.cls = LLAMA_VM_RESIDENCY_HOT;
        used += region.size;
        if (has_layer) {
            layer_used[region.layer] += region.size;
        }
    }
}

static void llama_vm_mark_warm_per_layer(llama_vm_context & vm, int top_n_per_layer) {
    if (top_n_per_layer <= 0) {
        return;
    }

    for (auto layer_ids : vm.layer_regions) {
        std::sort(layer_ids.begin(), layer_ids.end(), [&](size_t a, size_t b) {
            return vm.regions[a].prefetch_priority > vm.regions[b].prefetch_priority;
        });

        int marked = 0;
        for (const size_t id : layer_ids) {
            auto & region = vm.regions[id];
            if (region.cls == LLAMA_VM_RESIDENCY_HOT) {
                continue;
            }

            region.cls = LLAMA_VM_RESIDENCY_WARM;
            if (++marked >= top_n_per_layer) {
                break;
            }
        }
    }
}

static void llama_vm_update_class_stats(llama_vm_context & vm) {
    vm.bytes_hot = 0;
    vm.bytes_warm = 0;
    vm.bytes_cold = 0;
    vm.n_hot = 0;
    vm.n_warm = 0;
    vm.n_cold = 0;

    for (const auto & region : vm.regions) {
        switch (region.cls) {
            case LLAMA_VM_RESIDENCY_HOT:
                vm.n_hot++;
                vm.bytes_hot += region.size;
                break;
            case LLAMA_VM_RESIDENCY_WARM:
                vm.n_warm++;
                vm.bytes_warm += region.size;
                break;
            case LLAMA_VM_RESIDENCY_COLD:
                vm.n_cold++;
                vm.bytes_cold += region.size;
                break;
        }
    }
}

static void llama_vm_log_top(
        const llama_vm_context & vm,
        const char * title,
        llama_vm_residency_class cls,
        float llama_vm_region::* priority,
        int top_k) {
    std::vector<size_t> ids;
    ids.reserve(vm.regions.size());

    for (size_t i = 0; i < vm.regions.size(); ++i) {
        if (vm.regions[i].cls == cls) {
            ids.push_back(i);
        }
    }

    std::sort(ids.begin(), ids.end(), [&](size_t a, size_t b) {
        return vm.regions[a].*priority > vm.regions[b].*priority;
    });

    LLAMA_LOG_DEBUG("%s:\n", title);

    const int n = std::min<int>(top_k, ids.size());
    for (int i = 0; i < n; ++i) {
        const auto & region = vm.regions[ids[i]];
        LLAMA_LOG_DEBUG("  %-4s L%03d %-48s %8.2f MiB score=%7.4f pin=%7.4f prefetch=%7.4f evict=%7.4f\n",
                llama_vm_class_name(region.cls),
                region.layer,
                region.name.c_str(),
                llama_vm_size_mib(region.size),
                region.score,
                region.pin_priority,
                region.prefetch_priority,
                region.evict_priority);
    }
}

static void llama_vm_log_class_by_kind(const llama_vm_context & vm) {
    struct kind_stats {
        size_t count[3] = {};
        size_t bytes[3] = {};
    };

    constexpr int n_kinds = int(LLAMA_VM_TENSOR_OTHER) + 1;
    kind_stats stats[n_kinds];

    for (const auto & region : vm.regions) {
        const int kind = int(region.kind);
        const int cls = llama_vm_class_index(region.cls);
        if (kind < 0 || kind >= n_kinds) {
            continue;
        }

        stats[kind].count[cls]++;
        stats[kind].bytes[cls] += region.size;
    }

    LLAMA_LOG_DEBUG("llama_vm: class by kind:\n");
    for (int kind = 0; kind < n_kinds; ++kind) {
        const auto & s = stats[kind];
        if (s.count[0] + s.count[1] + s.count[2] == 0) {
            continue;
        }

        LLAMA_LOG_DEBUG("  %-11s hot=%4zu/%8.2f MiB warm=%4zu/%8.2f MiB cold=%4zu/%8.2f MiB\n",
                llama_vm_kind_name((llama_vm_tensor_kind) kind),
                s.count[0], s.bytes[0] / 1024.0 / 1024.0,
                s.count[1], s.bytes[1] / 1024.0 / 1024.0,
                s.count[2], s.bytes[2] / 1024.0 / 1024.0);
    }
}

static void llama_vm_log_class_by_layer(const llama_vm_context & vm) {
    LLAMA_LOG_DEBUG("llama_vm: layer bytes:\n");

    for (size_t layer = 0; layer < vm.layer_regions.size(); ++layer) {
        size_t bytes[3] = {};
        size_t count[3] = {};

        for (const size_t id : vm.layer_regions[layer]) {
            const auto & region = vm.regions[id];
            const int cls = llama_vm_class_index(region.cls);
            bytes[cls] += region.size;
            count[cls]++;
        }

        if (count[0] + count[1] + count[2] == 0) {
            continue;
        }

        LLAMA_LOG_DEBUG("  L%03zu hot=%8.2f MiB/%3zu warm=%8.2f MiB/%3zu cold=%8.2f MiB/%3zu\n",
                layer,
                bytes[0] / 1024.0 / 1024.0, count[0],
                bytes[1] / 1024.0 / 1024.0, count[1],
                bytes[2] / 1024.0 / 1024.0, count[2]);
    }
}

static void llama_vm_log_class_averages(const llama_vm_context & vm) {
    double score_sum[3] = {};
    double size_sum[3] = {};
    size_t count[3] = {};

    for (const auto & region : vm.regions) {
        const int cls = llama_vm_class_index(region.cls);
        score_sum[cls] += region.score;
        size_sum[cls] += llama_vm_size_mib(region.size);
        count[cls]++;
    }

    LLAMA_LOG_DEBUG("llama_vm: avg score by class:\n");
    for (int cls = 0; cls < 3; ++cls) {
        const double avg = count[cls] ? score_sum[cls] / double(count[cls]) : 0.0;
        LLAMA_LOG_DEBUG("  %-4s score=%.4f\n",
                llama_vm_class_name((llama_vm_residency_class) cls), avg);
    }

    LLAMA_LOG_DEBUG("llama_vm: avg size by class:\n");
    for (int cls = 0; cls < 3; ++cls) {
        const double avg = count[cls] ? size_sum[cls] / double(count[cls]) : 0.0;
        LLAMA_LOG_DEBUG("  %-4s size=%.2f MiB\n",
                llama_vm_class_name((llama_vm_residency_class) cls), avg);
    }
}

static void llama_vm_log_regions(const llama_vm_context & vm) {
    LLAMA_LOG_DEBUG("llama_vm: regions:\n");
    for (const auto & region : vm.regions) {
        LLAMA_LOG_DEBUG("  %-4s L%03d %-11s %-56s %8.2f MiB score=%7.4f pin=%7.4f prefetch=%7.4f evict=%7.4f file=%u offset=%zu addr=%p\n",
                llama_vm_class_name(region.cls),
                region.layer,
                llama_vm_kind_name(region.kind),
                region.name.c_str(),
                llama_vm_size_mib(region.size),
                region.score,
                region.pin_priority,
                region.prefetch_priority,
                region.evict_priority,
                unsigned(region.file_idx),
                region.file_offset,
                region.addr);
    }
}

static void llama_vm_log_blocks(const llama_vm_context & vm) {
    LLAMA_LOG_DEBUG("llama_vm: blocks: count=%zu bytes=%.2f MiB pinned=%zu/%.2f MiB block_size=%.2f MiB\n",
            vm.blocks.size(),
            vm.bytes_blocked / 1024.0 / 1024.0,
            vm.n_pinned,
            vm.bytes_pinned / 1024.0 / 1024.0,
            vm.params.block_size / 1024.0 / 1024.0);

    for (size_t region_id = 0; region_id < vm.regions.size(); ++region_id) {
        const auto & region = vm.regions[region_id];
        LLAMA_LOG_DEBUG("  region=%5zu L%03d %-11s blocks=%5zu pinned=%d small=%d %-56s %8.2f MiB\n",
                region_id,
                region.layer,
                llama_vm_kind_name(region.kind),
                region.block_ids.size(),
                region.pinned ? 1 : 0,
                region.small_tensor ? 1 : 0,
                region.name.c_str(),
                llama_vm_size_mib(region.size));

        for (const size_t block_id : region.block_ids) {
            const auto & block = vm.blocks[block_id];
            LLAMA_LOG_DEBUG("    block=%6zu id=%4d size=%8.2f MiB page=%8.2f MiB pinned=%d file=%u offset=%zu addr=%p\n",
                    block_id,
                    block.block_id,
                    llama_vm_size_mib(block.size),
                    llama_vm_size_mib(block.page_size),
                    block.pinned ? 1 : 0,
                    unsigned(block.file_idx),
                    block.file_offset,
                    block.addr);
        }
    }
}

static void llama_vm_classify_regions(llama_vm_context & vm) {
    for (auto & region : vm.regions) {
        region.cls = LLAMA_VM_RESIDENCY_COLD;
    }

    llama_vm_mark_hot_by_pin_budget(vm, llama_vm_default_pin_budget);
    llama_vm_mark_warm_per_layer(vm, llama_vm_default_warm_top_n_per_layer);
    llama_vm_update_class_stats(vm);
}

static void llama_vm_log_summary(const llama_vm_context & vm, bool debug_log) {
    LLAMA_LOG_DEBUG("%s: regions=%zu, mapped=%.2f MiB, layers=%zu\n",
            __func__, vm.regions.size(), vm.bytes_indexed / 1024.0 / 1024.0, vm.layer_regions.size());
    LLAMA_LOG_DEBUG("%s: classes: hot=%zu tensors / %.2f MiB, warm=%zu tensors / %.2f MiB, cold=%zu tensors / %.2f MiB\n",
            __func__,
            vm.n_hot, vm.bytes_hot / 1024.0 / 1024.0,
            vm.n_warm, vm.bytes_warm / 1024.0 / 1024.0,
            vm.n_cold, vm.bytes_cold / 1024.0 / 1024.0);

    llama_vm_log_top(vm, "llama_vm: top hot tensors by pin_priority", LLAMA_VM_RESIDENCY_HOT, &llama_vm_region::pin_priority, 8);
    llama_vm_log_top(vm, "llama_vm: top warm tensors by prefetch_priority", LLAMA_VM_RESIDENCY_WARM, &llama_vm_region::prefetch_priority, 8);
    llama_vm_log_top(vm, "llama_vm: top cold tensors by evict_priority", LLAMA_VM_RESIDENCY_COLD, &llama_vm_region::evict_priority, 8);
    llama_vm_log_class_by_kind(vm);
    llama_vm_log_class_by_layer(vm);
    llama_vm_log_class_averages(vm);
    if (debug_log) {
        llama_vm_log_regions(vm);
        llama_vm_log_blocks(vm);
    }
}

static void llama_vm_build_addr_ranges(llama_vm_context & vm) {
    vm.addr_ranges.clear();
    vm.addr_ranges.reserve(vm.regions.size());

    for (size_t region_id = 0; region_id < vm.regions.size(); ++region_id) {
        const auto & region = vm.regions[region_id];
        const uintptr_t begin = reinterpret_cast<uintptr_t>(region.addr);
        llama_vm_addr_range range;
        range.begin = begin;
        range.end = begin + region.size;
        range.region_id = region_id;
        vm.addr_ranges.push_back(range);
    }

    std::sort(vm.addr_ranges.begin(), vm.addr_ranges.end(), [](const llama_vm_addr_range & a, const llama_vm_addr_range & b) {
        return a.begin < b.begin;
    });
}

static size_t llama_vm_find_region_by_addr(const llama_vm_context & vm, const void * ptr) {
    if (ptr == nullptr || vm.addr_ranges.empty()) {
        return SIZE_MAX;
    }

    const uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    auto it = std::upper_bound(vm.addr_ranges.begin(), vm.addr_ranges.end(), p,
            [](uintptr_t value, const llama_vm_addr_range & range) {
                return value < range.begin;
            });

    if (it == vm.addr_ranges.begin()) {
        return SIZE_MAX;
    }

    --it;
    return p >= it->begin && p < it->end ? it->region_id : SIZE_MAX;
}

static size_t llama_vm_match_region(
        llama_vm_context & vm,
        const ggml_tensor * src,
        bool update_stats) {
    if (src == nullptr) {
        return SIZE_MAX;
    }

    const auto found = vm.region_by_name.find(ggml_get_name(src));
    if (found != vm.region_by_name.end()) {
        if (update_stats) {
            vm.n_match_by_name++;
        }
        return found->second;
    }

    const size_t region_id = llama_vm_find_region_by_addr(vm, src->data);
    if (region_id != SIZE_MAX) {
        if (update_stats) {
            vm.n_match_by_addr++;
        }
        return region_id;
    }

    if (update_stats) {
        vm.n_match_miss++;
    }
    return SIZE_MAX;
}

static bool llama_vm_ranges_overlap(uintptr_t a_begin, uintptr_t a_end, uintptr_t b_begin, uintptr_t b_end) {
    return a_begin < b_end && b_begin < a_end;
}

static void llama_vm_hash_mix(uint64_t & hash, uint64_t value) {
    hash ^= value;
    hash *= 1099511628211ull;
}

static uint64_t llama_vm_hash_graph(llama_vm_context & vm, const ggml_cgraph * gf) {
    if (gf == nullptr) {
        return 0;
    }

    uint64_t hash = 1469598103934665603ull;
    auto * gf_mut = const_cast<ggml_cgraph *>(gf);
    const int n_nodes = ggml_graph_n_nodes(gf_mut);
    llama_vm_hash_mix(hash, (uint64_t) n_nodes);

    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(gf_mut, i);
        if (node == nullptr) {
            continue;
        }

        llama_vm_hash_mix(hash, (uint64_t) node->op);
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            const size_t region_id = llama_vm_match_region(vm, src, false);
            if (region_id == SIZE_MAX) {
                continue;
            }

            llama_vm_hash_mix(hash, (uint64_t) region_id);
            llama_vm_hash_mix(hash, (uint64_t) reinterpret_cast<uintptr_t>(src->data));
            llama_vm_hash_mix(hash, (uint64_t) ggml_nbytes(src));
        }
    }

    return hash;
}

static void llama_vm_add_region_blocks_for_src(
        llama_vm_context & vm,
        llama_vm_exec_plan & plan,
        std::vector<std::unordered_set<size_t>> & seen_by_step,
        std::unordered_map<int, size_t> & step_by_layer,
        size_t region_id,
        const ggml_tensor * src) {
    if (region_id >= vm.regions.size()) {
        return;
    }

    const auto & region = vm.regions[region_id];
    size_t step_id;
    auto it = step_by_layer.find(region.layer);
    if (it == step_by_layer.end()) {
        step_id = plan.steps.size();
        step_by_layer.emplace(region.layer, step_id);
        seen_by_step.emplace_back();

        llama_vm_exec_step step;
        step.layer = region.layer;
        plan.steps.emplace_back(std::move(step));
    } else {
        step_id = it->second;
    }

    uintptr_t src_begin = 0;
    uintptr_t src_end = 0;
    bool use_overlap = false;
    if (src != nullptr && src->data != nullptr) {
        src_begin = reinterpret_cast<uintptr_t>(src->data);
        src_end = src_begin + ggml_nbytes(src);
        const uintptr_t region_begin = reinterpret_cast<uintptr_t>(region.addr);
        const uintptr_t region_end = region_begin + region.size;
        use_overlap = src_begin >= region_begin && src_begin < region_end &&
                !(src_begin == region_begin && src_end == region_end);
    }

    auto & seen = seen_by_step[step_id];
    auto & step = plan.steps[step_id];
    for (const size_t block_id : region.block_ids) {
        if (block_id >= vm.blocks.size()) {
            continue;
        }

        const auto & block = vm.blocks[block_id];
        if (use_overlap) {
            const uintptr_t block_begin = reinterpret_cast<uintptr_t>(block.addr);
            const uintptr_t block_end = block_begin + block.size;
            if (!llama_vm_ranges_overlap(src_begin, src_end, block_begin, block_end)) {
                continue;
            }
        }

        if (seen.insert(block_id).second) {
            step.block_ids.push_back(block_id);
            if (!block.pinned) {
                plan.bytes += block.size;
            }
        }
    }
}

static llama_vm_exec_plan llama_vm_build_exec_plan_uncached(
        llama_vm_context & vm,
        const ggml_cgraph * gf) {
    llama_vm_exec_plan plan;
    if (gf == nullptr) {
        return plan;
    }

    std::unordered_map<int, size_t> step_by_layer;
    std::vector<std::unordered_set<size_t>> seen_by_step;

    auto * gf_mut = const_cast<ggml_cgraph *>(gf);
    const int n_nodes = ggml_graph_n_nodes(gf_mut);
    for (int i = 0; i < n_nodes; ++i) {
        const ggml_tensor * node = ggml_graph_node(gf_mut, i);
        if (node == nullptr) {
            continue;
        }

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * src = node->src[j];
            const size_t region_id = llama_vm_match_region(vm, src, true);
            if (region_id != SIZE_MAX) {
                llama_vm_add_region_blocks_for_src(vm, plan, seen_by_step, step_by_layer, region_id, src);
            }
        }
    }

    return plan;
}

static void llama_vm_log_exec_plan_once(llama_vm_context & vm, const llama_vm_exec_plan & plan) {
    if (!vm.params.debug_log || plan.steps.empty() || vm.n_exec_plan_logs++ != 0) {
        return;
    }

    LLAMA_LOG_DEBUG("llama_vm: exec plan: steps=%zu bytes=%.2f MiB\n",
            plan.steps.size(), plan.bytes / 1024.0 / 1024.0);
    for (size_t i = 0; i < plan.steps.size(); ++i) {
        size_t bytes = 0;
        for (const size_t block_id : plan.steps[i].block_ids) {
            if (block_id < vm.blocks.size() && !vm.blocks[block_id].pinned) {
                bytes += vm.blocks[block_id].size;
            }
        }
        LLAMA_LOG_DEBUG("  step=%4zu L%03d blocks=%6zu bytes=%8.2f MiB\n",
                i,
                plan.steps[i].layer,
                plan.steps[i].block_ids.size(),
                bytes / 1024.0 / 1024.0);
    }

    LLAMA_LOG_DEBUG("llama_vm: graph match by_name=%" PRIu64 " by_addr=%" PRIu64 " miss=%" PRIu64 "\n",
            vm.n_match_by_name, vm.n_match_by_addr, vm.n_match_miss);
}

static const llama_vm_exec_plan * llama_vm_get_or_build_exec_plan(
        llama_vm_context & vm,
        const ggml_cgraph * gf) {
    const uint64_t key = llama_vm_hash_graph(vm, gf);
    auto found = vm.plan_cache.find(key);
    if (found != vm.plan_cache.end()) {
        found->second.hit_count++;
        found->second.last_used_epoch = vm.epoch;
        vm.n_plan_cache_hits++;
        if (vm.params.debug_log && vm.n_plan_cache_logs++ < 8) {
            LLAMA_LOG_DEBUG("llama_vm: plan cache hit key=%" PRIu64 " hits=%" PRIu64 " steps=%zu\n",
                    key, found->second.hit_count, found->second.plan.steps.size());
        }
        return &found->second.plan;
    }

    vm.n_plan_cache_misses++;
    llama_vm_exec_plan plan = llama_vm_build_exec_plan_uncached(vm, gf);
    llama_vm_log_exec_plan_once(vm, plan);

    if (vm.plan_cache.size() >= (size_t) vm.params.max_plan_cache_entries) {
        auto evict = std::min_element(vm.plan_cache.begin(), vm.plan_cache.end(),
                [](const auto & a, const auto & b) {
                    return a.second.last_used_epoch < b.second.last_used_epoch;
                });
        if (evict != vm.plan_cache.end()) {
            vm.plan_cache.erase(evict);
        }
    }

    llama_vm_plan_cache_entry entry;
    entry.key = key;
    entry.plan = std::move(plan);
    entry.last_used_epoch = vm.epoch;
    auto inserted = vm.plan_cache.emplace(key, std::move(entry));
    if (vm.params.debug_log && vm.n_plan_cache_logs++ < 8) {
        LLAMA_LOG_DEBUG("llama_vm: plan cache miss key=%" PRIu64 " steps=%zu bytes=%.2f MiB\n",
                key, inserted.first->second.plan.steps.size(), inserted.first->second.plan.bytes / 1024.0 / 1024.0);
    }
    return &inserted.first->second.plan;
}

std::unique_ptr<llama_vm_context> llama_vm_build_index(
        const std::vector<llama_vm_region_input> & inputs,
        uint32_t n_layer,
        const llama_vm_params & params) {
    auto vm = std::make_unique<llama_vm_context>();
    vm->params = params;

    vm->regions.reserve(inputs.size());
    vm->layer_regions.resize(n_layer);

    for (const auto & input : inputs) {
        if (input.addr == nullptr || input.size == 0) {
            continue;
        }

        llama_vm_region region;
        region.name = input.name;
        region.addr = input.addr;
        region.size = input.size;
        region.file_idx = input.file_idx;
        region.file_offset = input.file_offset;

        region.layer = llama_vm_parse_layer(region.name.c_str());
        region.kind = llama_vm_parse_kind(region.name.c_str());

        region.kind_score = llama_vm_kind_score(region.kind);
        region.layer_score = llama_vm_layer_score(region.layer, n_layer);
        region.io_score = llama_vm_io_score(region.size);
        region.wait_score = 0.0f;
        region.fault_score = 0.0f;
        llama_vm_compute_scores(region);

        const size_t region_idx = vm->regions.size();
        vm->bytes_indexed += region.size;
        vm->region_by_name.emplace(region.name, region_idx);
        vm->regions.emplace_back(std::move(region));

        if (vm->regions.back().layer >= 0 && (uint32_t) vm->regions.back().layer < n_layer) {
            vm->layer_regions[vm->regions.back().layer].push_back(region_idx);
        }
    }

    llama_vm_classify_regions(*vm);
    llama_vm_build_blocks(*vm);
    llama_vm_pin_small_blocks(*vm);
    llama_vm_build_addr_ranges(*vm);
    llama_vm_log_summary(*vm, params.debug_log);

    return vm;
}

llama_vm_exec_plan llama_vm_build_exec_plan(
        llama_vm_context & vm,
        const ggml_cgraph * gf) {
    llama_vm_exec_plan plan = llama_vm_build_exec_plan_uncached(vm, gf);
    llama_vm_log_exec_plan_once(vm, plan);
    return plan;
}

void llama_vm_prefetch_plan(
        llama_vm_context & vm,
        const llama_vm_exec_plan & plan) {
    if (plan.steps.empty() || vm.params.prefetch_budget_bytes == 0) {
        return;
    }

    size_t issued = 0;
    size_t n_blocks = 0;
    const int window_steps = std::max(1, vm.params.window_steps);
    const size_t n_steps = std::min<size_t>(plan.steps.size(), (size_t) window_steps);

    for (size_t step_id = 0; step_id < n_steps; ++step_id) {
        for (const size_t block_id : plan.steps[step_id].block_ids) {
            if (block_id >= vm.blocks.size()) {
                continue;
            }

            auto & block = vm.blocks[block_id];
            if (block.pinned || block.prefetched) {
                block.state = LLAMA_VM_BLOCK_IN_WINDOW;
                block.last_window_epoch = vm.epoch;
                continue;
            }
            if (issued + block.page_size > vm.params.prefetch_budget_bytes) {
                goto done;
            }

#if defined(MADV_WILLNEED)
            if (llama_vm_madvise(block.page_addr, block.page_size, MADV_WILLNEED) == 0) {
                block.prefetched = true;
                block.state = LLAMA_VM_BLOCK_WILLNEED;
                block.last_prefetch_epoch = vm.epoch;
                block.last_window_epoch = vm.epoch;
                issued += block.page_size;
                n_blocks++;
            }
#endif
        }
    }

done:
    if (vm.params.debug_log && (vm.n_prefetch_logs++ == 0 || n_blocks > 0)) {
        LLAMA_LOG_DEBUG("llama_vm: prefetch initial window steps=%zu issued=%zu blocks %.2f MiB\n",
                n_steps, n_blocks, issued / 1024.0 / 1024.0);
    }
}

static void llama_vm_reclaim_outside_current_plan(
        llama_vm_context & vm,
        const llama_vm_exec_plan & plan) {
    if (!vm.params.use_dontneed || vm.params.reclaim_budget_bytes == 0 || vm.blocks.empty()) {
        return;
    }

    std::vector<uint8_t> keep(vm.blocks.size(), 0);
    for (const auto & step : plan.steps) {
        for (const size_t block_id : step.block_ids) {
            if (block_id < keep.size()) {
                keep[block_id] = 1;
                vm.blocks[block_id].last_window_epoch = vm.epoch;
            }
        }
    }

    size_t issued = 0;
    size_t n_blocks = 0;

    for (auto & block : vm.blocks) {
        if (issued >= vm.params.reclaim_budget_bytes) {
            break;
        }
        if (block.pinned || !block.prefetched) {
            continue;
        }
        const size_t block_index = (size_t) (&block - vm.blocks.data());
        if (block_index < keep.size() && keep[block_index]) {
            continue;
        }
        if (block.region_id >= vm.regions.size()) {
            continue;
        }
        const auto & region = vm.regions[block.region_id];
        if (region.protected_from_reclaim) {
            continue;
        }
        if (block.state == LLAMA_VM_BLOCK_RECLAIMED) {
            continue;
        }
        if (vm.epoch - block.last_prefetch_epoch <= 1) {
            continue;
        }

#if defined(MADV_DONTNEED)
        if (llama_vm_madvise(block.page_addr, block.page_size, MADV_DONTNEED) == 0) {
            block.prefetched = false;
            block.state = LLAMA_VM_BLOCK_RECLAIMED;
            block.last_dontneed_epoch = vm.epoch;
            issued += block.page_size;
            n_blocks++;
            vm.n_dontneed++;
            vm.bytes_dontneed += block.page_size;
        } else {
            vm.n_dontneed_fail++;
        }
#endif
    }

    if (vm.params.debug_log && (n_blocks > 0 || vm.n_reclaim_logs++ == 0)) {
        LLAMA_LOG_DEBUG("llama_vm: dontneed current-plan-safe issued=%zu blocks %.2f MiB total=%" PRIu64 "/%.2f MiB fail=%" PRIu64 "\n",
                n_blocks,
                issued / 1024.0 / 1024.0,
                vm.n_dontneed,
                vm.bytes_dontneed / 1024.0 / 1024.0,
                vm.n_dontneed_fail);
    }
}

void llama_vm_on_graph_compute_begin(
        llama_vm_context & vm,
        const ggml_cgraph * gf) {
    vm.epoch++;

    if (vm.params.max_plan_cache_entries <= 0) {
        llama_vm_exec_plan plan = llama_vm_build_exec_plan_uncached(vm, gf);
        llama_vm_log_exec_plan_once(vm, plan);
        if (plan.steps.empty()) {
            return;
        }
        llama_vm_reclaim_outside_current_plan(vm, plan);
        llama_vm_prefetch_plan(vm, plan);
        return;
    }

    const llama_vm_exec_plan * plan = llama_vm_get_or_build_exec_plan(vm, gf);
    if (plan == nullptr || plan->steps.empty()) {
        return;
    }

    llama_vm_reclaim_outside_current_plan(vm, *plan);
    llama_vm_prefetch_plan(vm, *plan);
}

void llama_vm_discard_all(
        llama_vm_context & vm) {
    size_t issued = 0;
    size_t n_blocks = 0;
    for (auto & block : vm.blocks) {
        if (block.pinned || !block.prefetched) {
            continue;
        }

#if defined(MADV_DONTNEED)
        if (llama_vm_madvise(block.page_addr, block.page_size, MADV_DONTNEED) == 0) {
            block.prefetched = false;
            issued += block.page_size;
            n_blocks++;
        }
#endif
    }

    if (vm.params.debug_log) {
        LLAMA_LOG_DEBUG("llama_vm: discard all issued=%zu blocks %.2f MiB\n",
                n_blocks, issued / 1024.0 / 1024.0);
    }
}
