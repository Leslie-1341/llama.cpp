#include "../src/llama-kv-cache-release.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

struct fake_cells {
    std::vector<unsigned> owners;
    uint32_t used_max_p1() const { return (uint32_t) owners.size(); }
    bool is_empty(uint32_t cell) const { return owners[cell] == 0; }
    bool seq_has(uint32_t cell, int32_t seq) const { return (owners[cell] & (1u << seq)) != 0; }
};

static void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    check(llama_kv_destructive_release_can_enable(true, true, true, true, false),
            "row-index gather permits destructive release");
    check(!llama_kv_destructive_release_can_enable(true, false, true, false, false),
            "INGRAPH=0 continuous view rejects destructive release");
    check(!llama_kv_destructive_release_can_enable(true, true, false, false, false),
            "non-F32/unsupported layers reject destructive release");
    check(!llama_kv_destructive_release_can_enable(true, true, true, false, false),
            "any other row-index-unavailable topology rejects destructive release");
    check(!llama_kv_destructive_release_can_enable(false, true, true, false, false),
            "non-paged topology rejects destructive release");
    check(!llama_kv_destructive_release_can_enable(true, true, true, true, true),
            "swap and destructive release remain mutually exclusive");

    const std::vector<fake_cells> streams {{{0, 1, 0, 0, 3, 0, 0, 0}}, {{0, 0, 0, 1}}};
    const auto identity = llama_kv_release_collect_ownership(
            streams, 3, 4, UINT32_MAX, 4, [](uint32_t cell) { return cell; });
    check(identity.valid, "identity ownership mapping is valid");
    check(identity.owned[0] == 1, "single-owner cell protects its block");
    check(identity.owned[1] == 1, "shared cell protects its block");
    check(identity.owned[2] == 0, "empty block remains releasable");
    check(identity.shared[0] == 0, "single-owner block is not shared");
    check(identity.shared[1] == 1, "shared owner is recorded");

    const auto remapped = llama_kv_release_collect_ownership(
            streams, 3, 4, UINT32_MAX, 4,
            [](uint32_t cell) { return cell < 4 ? cell + 8 : cell; });
    check(remapped.owned[2] == 1, "ownership follows logical-to-physical mapping");
    check(remapped.owned[0] == 0 && remapped.owned[1] == 1,
            "empty remapped block stays releasable and shared cells remain protected");

    const std::vector<fake_cells> distinct_owners {{{1, 2, 0, 0}}};
    const auto block_shared = llama_kv_release_collect_ownership(
            distinct_owners, 1, 4, UINT32_MAX, 4, [](uint32_t cell) { return cell; });
    check(block_shared.shared[0] == 1, "distinct single-owner cells make the block shared");

    const auto invalid = llama_kv_release_collect_ownership(
            streams, 3, 4, UINT32_MAX, 4,
            [](uint32_t cell) { return cell == 1 ? UINT32_MAX : cell; });
    check(!invalid.valid && invalid.invalid_mappings == 1,
            "a live cell with invalid mapping fails the whole ownership pass");

    std::puts("PASS: paged release ownership eligibility");
    return 0;
}
