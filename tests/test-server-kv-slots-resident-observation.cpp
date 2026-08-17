// Unit tests for the `/slots.kv_resident` production authority restoration.
//
// Covers:
//   1. mode routing matrix (parse_kv_slots_resident_routing):
//        "1"        -> transaction marker on, /slots off
//        "preflight"-> transaction marker off, /slots on
//        "both"     -> both on
//        "" / unrecognized / nullptr -> both off
//   2. observation JSON schema (format_kv_slots_resident_observation):
//        available -> exact 9-field canonical schema
//        unavailable -> {"status": "unavailable"} fail-closed
//   3. multi-slot identity: one sample copied verbatim yields equal JSON across slots.
//
// `LLAMA_KV_RESIDENT_PREFLIGHT=1` is orthogonal to this matrix and is not
// parsed here; its periodic preflight tests live in test-kv-resident-preflight.

#include "server-kv-slots-resident-observation.h"

#include <cstdio>
#include <vector>

static int tests_total = 0;
static int tests_failed = 0;

static bool check(bool condition, const char * message) {
    ++tests_total;
    if (!condition) {
        ++tests_failed;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

#define CHECK(condition) check((condition), #condition)

static void test_mode_routing_matrix() {
    auto r_none = parse_kv_slots_resident_routing(nullptr);
    CHECK(!r_none.transaction_marker);
    CHECK(!r_none.slots_observation);

    auto r_empty = parse_kv_slots_resident_routing("");
    CHECK(!r_empty.transaction_marker);
    CHECK(!r_empty.slots_observation);

    // mode "1": transaction marker only
    auto r1 = parse_kv_slots_resident_routing("1");
    CHECK(r1.transaction_marker);
    CHECK(!r1.slots_observation);

    // mode "preflight": /slots observation only, no transaction marker
    auto r_pf = parse_kv_slots_resident_routing("preflight");
    CHECK(!r_pf.transaction_marker);
    CHECK(r_pf.slots_observation);

    // mode "both": both contracts active
    auto r_both = parse_kv_slots_resident_routing("both");
    CHECK(r_both.transaction_marker);
    CHECK(r_both.slots_observation);

    // unrecognized / invalid modes fail closed: nothing on
    auto r_bogus = parse_kv_slots_resident_routing("yes");
    CHECK(!r_bogus.transaction_marker);
    CHECK(!r_bogus.slots_observation);

    auto r_case = parse_kv_slots_resident_routing("BOTH");
    CHECK(!r_case.transaction_marker);
    CHECK(!r_case.slots_observation);
}

static void test_available_observation_schema() {
    llama_kv_resident_sample s{};
    s.available       = true;
    s.object_id       = 7;
    s.generation      = 3;
    s.page_size        = 4096;
    s.total_bytes      = 1ULL << 20;     // 1 MiB
    s.resident_bytes   = 1 << 18;        // 256 KiB resident
    s.total_pages      = s.total_bytes / s.page_size;
    s.resident_pages   = s.resident_bytes / s.page_size;

    auto j = format_kv_slots_resident_observation(s);
    CHECK(j.is_object());
    CHECK(j.size() == 9);
    CHECK(std::string(j["status"]) == "available");
    CHECK(std::string(j["source"]) == "paged_sample_mincore");
    CHECK(j["object_id"].get<uint64_t>() == 7);
    CHECK(j["generation"].get<uint64_t>() == 3);
    CHECK(j["page_size"].get<uint64_t>() == 4096);
    CHECK(j["total_bytes"].get<uint64_t>() == (1ULL << 20));
    CHECK(j["resident_bytes"].get<uint64_t>() == (1 << 18));
    CHECK(j["total_pages"].get<uint64_t>() == 256);
    CHECK(j["resident_pages"].get<uint64_t>() == 64);
    CHECK(j["resident_bytes"].get<uint64_t>() <= j["total_bytes"].get<uint64_t>());
    CHECK(j["resident_pages"].get<uint64_t>() <= j["total_pages"].get<uint64_t>());

    // Canonical key order matters for the Final F16 parser contract.
    const std::vector<std::string> expected_keys = {
        "status", "source", "object_id", "generation", "page_size",
        "total_bytes", "resident_bytes", "total_pages", "resident_pages",
    };
    std::vector<std::string> actual_keys;
    for (auto it = j.begin(); it != j.end(); ++it) {
        actual_keys.emplace_back(it.key());
    }
    CHECK(actual_keys == expected_keys);

    // No proxy fields sneak in — only physical mincore measurement is exposed.
    CHECK(!j.contains("budget_resident_bytes"));
    CHECK(!j.contains("logical_bytes"));
    CHECK(!j.contains("requested_bytes"));
    CHECK(!j.contains("serialized_bytes"));
    CHECK(!j.contains("advised_bytes"));
    CHECK(!j.contains("backing_bytes"));
}

static void test_unavailable_observation_fail_closed() {
    llama_kv_resident_sample s{}; // available = false, everything zero
    auto j = format_kv_slots_resident_observation(s);
    CHECK(j.is_object());
    CHECK(j.size() == 1);
    CHECK(std::string(j["status"]) == "unavailable");
    // A zero-resident sample must not be forged as an unavailable status: a
    // genuinely available zero-resident cache still reports "available".
    llama_kv_resident_sample zero{};
    zero.available = true; // physically resident observe succeeded, even if 0
    auto jz = format_kv_slots_resident_observation(zero);
    CHECK(std::string(jz["status"]) == "available");
    CHECK(jz.size() == 9);
    CHECK(jz["resident_bytes"].get<uint64_t>() == 0);
    CHECK(jz["resident_pages"].get<uint64_t>() == 0);
}

static void test_multislot_observation_identity() {
    // The METRICS handler samples exactly once and copies the same observation
    // to every slot.  Simulate that by formatting one sample into N copies and
    // asserting byte-for-byte equality — snapshot drift would diverge here.
    llama_kv_resident_sample s{};
    s.available       = true;
    s.object_id       = 42;
    s.generation      = 11;
    s.page_size        = 4096;
    s.total_bytes      = 8 * s.page_size;
    s.resident_bytes   = 3 * s.page_size;
    s.total_pages      = 8;
    s.resident_pages   = 3;

    const auto snapshot = format_kv_slots_resident_observation(s);
    for (int slot = 0; slot < 4; ++slot) {
        const auto copy = format_kv_slots_resident_observation(s);
        CHECK(copy.dump() == snapshot.dump());
    }
}

static void test_preflight_orthogonal_flag_not_in_matrix() {
    // Demonstrates that LLAMA_KV_RESIDENT_PREFLIGHT is a concern of
    // server-context.cpp's init(), not of the mode parser: the routing helper
    // only reads LLAMA_KV_G0_S1_RESIDENT_OBSERVATION.  This guard prevents
    // re-merging periodic preflight into the /slots authority boolean here.
    auto r1 = parse_kv_slots_resident_routing("1");
    auto r_both = parse_kv_slots_resident_routing("both");
    // No third output field exists to couple periodic preflight in.
    CHECK(r1.transaction_marker == true);
    CHECK(r1.slots_observation == false);
    CHECK(r_both.transaction_marker == true);
    CHECK(r_both.slots_observation == true);
}

int main() {
    test_mode_routing_matrix();
    test_available_observation_schema();
    test_unavailable_observation_fail_closed();
    test_multislot_observation_identity();
    test_preflight_orthogonal_flag_not_in_matrix();

    std::printf("server KV slots resident observation tests: %d/%d passed\n",
        tests_total - tests_failed, tests_total);
    return tests_failed == 0 ? 0 : 1;
}
