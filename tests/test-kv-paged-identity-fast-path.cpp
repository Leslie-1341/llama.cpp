#include "../src/llama-kv-cache-identity.h"

#include <cstdio>
#include <cstdlib>

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

static llama_kv_paged_identity_fast_path_config eligible_config() {
    return {
        /* .requested        = */ true,
        /* .paged_enabled    = */ true,
        /* .ingraph_enabled  = */ true,
        /* .single_stream    = */ true,
        /* .v_trans          = */ false,
        /* .identity_mapping = */ true,
        /* .dynamic_remap    = */ false,
        /* .swap             = */ false,
        /* .release          = */ false,
        /* .madvise          = */ false,
        /* .mapping_read_fault  = */ false,
        /* .mapping_write_fault = */ false,
        /* .swapin_fault        = */ false,
        /* .backing_io_fault    = */ false,
        /* .layers_supported = */ true,
    };
}

static void expect_reject(
        const llama_kv_paged_identity_fast_path_config & config,
        llama_kv_paged_identity_fast_path_reject expected) {
    const auto result = llama_kv_paged_identity_fast_path_resolve(config);
    check(!result.enabled, "rejected configuration must not enable fast path");
    check(result.reject == expected, "unexpected reject reason");
    check(llama_kv_paged_identity_fast_path_reject_name(result.reject)[0] != '\0',
            "reject reason must have telemetry text");
}

int main() {
    auto config = eligible_config();
    auto result = llama_kv_paged_identity_fast_path_resolve(config);
    check(result.enabled, "strict identity configuration must be eligible");
    check(result.reject == llama_kv_paged_identity_fast_path_reject::NONE,
            "eligible configuration must report reject_reason=none");

    config = eligible_config(); config.requested = false;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::NOT_REQUESTED);
    config = eligible_config(); config.identity_mapping = false;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::NON_IDENTITY_MAPPING);
    config = eligible_config(); config.dynamic_remap = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::DYNAMIC_REMAP);
    config = eligible_config(); config.swap = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::SWAP);
    config = eligible_config(); config.release = true;
    // Cleanup-C2: identity fast-path `.release` is now driven by the
    // authoritative bounded_release_can_enable() capability.  The reject
    // enum value RELEASE and its telemetry token are preserved so external
    // tooling still distinguishes this fast-path disqualifier; only the
    // producer (what feeds .release) has migrated.
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::RELEASE);
    config = eligible_config(); config.madvise = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::MADVISE);
    config = eligible_config(); config.mapping_read_fault = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::TEST_FAULT);
    config = eligible_config(); config.mapping_write_fault = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::TEST_FAULT);
    config = eligible_config(); config.swapin_fault = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::TEST_FAULT);
    config = eligible_config(); config.backing_io_fault = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::TEST_FAULT);
    config = eligible_config(); config.single_stream = false;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::MULTI_STREAM);
    config = eligible_config(); config.v_trans = true;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::V_TRANS);
    config = eligible_config(); config.layers_supported = false;
    expect_reject(config, llama_kv_paged_identity_fast_path_reject::UNSUPPORTED_LAYER);

    check(llama_kv_paged_row_idx_topology_matches(false, false), "matching no-row topology must reuse");
    check(llama_kv_paged_row_idx_topology_matches(true, true), "matching row topology must reuse");
    check(!llama_kv_paged_row_idx_topology_matches(false, true), "missing row input must reject reuse");
    check(!llama_kv_paged_row_idx_topology_matches(true, false), "extra row input must reject reuse");

    std::puts("PASS: paged identity fast-path eligibility");
    return 0;
}
