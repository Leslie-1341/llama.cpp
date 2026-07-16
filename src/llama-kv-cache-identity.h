#pragma once

#include <cstdint>

enum class llama_kv_paged_identity_fast_path_reject : uint8_t {
    NONE = 0,
    NOT_REQUESTED,
    PAGED_DISABLED,
    INGRAPH_DISABLED,
    MULTI_STREAM,
    V_TRANS,
    APPROX_DYNAMIC,
    NON_IDENTITY_MAPPING,
    DYNAMIC_REMAP,
    SWAP,
    RELEASE,
    MADVISE,
    TEST_FAULT,
    UNSUPPORTED_LAYER,
};

struct llama_kv_paged_identity_fast_path_config {
    bool requested        = false;
    bool paged_enabled    = false;
    bool ingraph_enabled  = false;
    bool single_stream    = false;
    bool v_trans          = false;
    bool approx_dynamic   = false;
    bool identity_mapping = false;
    bool dynamic_remap    = false;
    bool swap             = false;
    bool release          = false;
    bool madvise          = false;
    bool mapping_read_fault  = false;
    bool mapping_write_fault = false;
    bool swapin_fault        = false;
    bool backing_io_fault    = false;
    bool layers_supported = false;
};

struct llama_kv_paged_identity_fast_path_result {
    bool enabled = false;
    llama_kv_paged_identity_fast_path_reject reject =
        llama_kv_paged_identity_fast_path_reject::NOT_REQUESTED;
};

inline llama_kv_paged_identity_fast_path_result llama_kv_paged_identity_fast_path_resolve(
        const llama_kv_paged_identity_fast_path_config & config) {
    if (!config.requested) {
        return { false, llama_kv_paged_identity_fast_path_reject::NOT_REQUESTED };
    }
    if (config.mapping_read_fault || config.mapping_write_fault ||
            config.swapin_fault || config.backing_io_fault) {
        return { false, llama_kv_paged_identity_fast_path_reject::TEST_FAULT };
    }
    if (!config.paged_enabled) {
        return { false, llama_kv_paged_identity_fast_path_reject::PAGED_DISABLED };
    }
    if (!config.ingraph_enabled) {
        return { false, llama_kv_paged_identity_fast_path_reject::INGRAPH_DISABLED };
    }
    if (!config.single_stream) {
        return { false, llama_kv_paged_identity_fast_path_reject::MULTI_STREAM };
    }
    if (config.v_trans) {
        return { false, llama_kv_paged_identity_fast_path_reject::V_TRANS };
    }
    if (config.approx_dynamic) {
        return { false, llama_kv_paged_identity_fast_path_reject::APPROX_DYNAMIC };
    }
    if (!config.identity_mapping) {
        return { false, llama_kv_paged_identity_fast_path_reject::NON_IDENTITY_MAPPING };
    }
    if (config.dynamic_remap) {
        return { false, llama_kv_paged_identity_fast_path_reject::DYNAMIC_REMAP };
    }
    if (config.swap) {
        return { false, llama_kv_paged_identity_fast_path_reject::SWAP };
    }
    if (config.release) {
        return { false, llama_kv_paged_identity_fast_path_reject::RELEASE };
    }
    if (config.madvise) {
        return { false, llama_kv_paged_identity_fast_path_reject::MADVISE };
    }
    if (!config.layers_supported) {
        return { false, llama_kv_paged_identity_fast_path_reject::UNSUPPORTED_LAYER };
    }
    return { true, llama_kv_paged_identity_fast_path_reject::NONE };
}

inline const char * llama_kv_paged_identity_fast_path_reject_name(
        llama_kv_paged_identity_fast_path_reject reject) {
    switch (reject) {
        case llama_kv_paged_identity_fast_path_reject::NONE:                 return "none";
        case llama_kv_paged_identity_fast_path_reject::NOT_REQUESTED:        return "not_requested";
        case llama_kv_paged_identity_fast_path_reject::PAGED_DISABLED:       return "paged_disabled";
        case llama_kv_paged_identity_fast_path_reject::INGRAPH_DISABLED:     return "ingraph_disabled";
        case llama_kv_paged_identity_fast_path_reject::MULTI_STREAM:         return "multi_stream";
        case llama_kv_paged_identity_fast_path_reject::V_TRANS:              return "v_trans";
        case llama_kv_paged_identity_fast_path_reject::APPROX_DYNAMIC:       return "approx_dynamic";
        case llama_kv_paged_identity_fast_path_reject::NON_IDENTITY_MAPPING: return "non_identity_mapping";
        case llama_kv_paged_identity_fast_path_reject::DYNAMIC_REMAP:        return "dynamic_remap";
        case llama_kv_paged_identity_fast_path_reject::SWAP:                 return "swap";
        case llama_kv_paged_identity_fast_path_reject::RELEASE:              return "release";
        case llama_kv_paged_identity_fast_path_reject::MADVISE:              return "madvise";
        case llama_kv_paged_identity_fast_path_reject::TEST_FAULT:           return "test_fault";
        case llama_kv_paged_identity_fast_path_reject::UNSUPPORTED_LAYER:    return "unsupported_layer";
    }
    return "unknown";
}

inline bool llama_kv_paged_row_idx_topology_matches(bool graph_has_row_idx, bool context_uses_row_idx) {
    return graph_has_row_idx == context_uses_row_idx;
}
