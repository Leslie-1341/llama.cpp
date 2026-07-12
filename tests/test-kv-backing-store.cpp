// KV-P0-B1: unit tests for the fixed physical-cell slot backing store
// (llama_kv_backing_store_file). Does not load a model or exercise llama_kv_cache; it only
// drives the backing-store class directly to verify the bounded-capacity invariant:
//   - offset(cell_id) = cell_id * cell_stride is a pure function (no append-at-file_len growth)
//   - the file's on-disk size never exceeds n_slots * cell_stride, no matter how many times a
//     cell is repeatedly swapped out
//   - invalid cell_id / size are rejected rather than silently accepted
//   - reset() reclaims the store for reuse without changing its addressing

#include "../src/llama-kv-cache.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/types.h>

static void check(bool cond, const std::string & msg) {
    if (!cond) {
        throw std::runtime_error("FAILED: " + msg);
    }
    fprintf(stderr, "OK: %s\n", msg.c_str());
}

static std::vector<uint8_t> make_pattern(size_t size, uint32_t seed) {
    std::vector<uint8_t> buf(size);
    std::mt19937 rng(seed);
    for (auto & b : buf) {
        b = (uint8_t) (rng() & 0xFF);
    }
    return buf;
}

// 1) same slot, 1000x write/read, byte-identical each time; capacity never changes.
static void test_same_slot_repeated_roundtrip() {
    const uint32_t n_slots     = 4;
    const size_t   cell_stride = 256;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "same_slot_repeated_roundtrip: store enabled");
    check(store.get_capacity() == (uint64_t) n_slots * cell_stride, "capacity == n_slots*cell_stride");
    check(store.get_actual_file_size() == store.get_capacity(), "initial file size == capacity");

    const uint32_t cell = 2;

    for (uint32_t i = 0; i < 1000; ++i) {
        auto payload = make_pattern(cell_stride, i);
        uint64_t offset = 0;

        const auto ws = store.write_cell(0, cell, payload.data(), payload.size(), offset);
        check(ws == llama_kv_backing_store_status::ok, "write ok at iter " + std::to_string(i));
        check(offset == (uint64_t) cell * cell_stride, "offset deterministic at iter " + std::to_string(i));

        std::vector<uint8_t> restored(cell_stride, 0);
        const auto rs = store.read_cell(0, cell, offset, restored.data(), restored.size());
        check(rs == llama_kv_backing_store_status::ok, "read ok at iter " + std::to_string(i));
        check(std::memcmp(payload.data(), restored.data(), cell_stride) == 0,
                "byte-identical roundtrip at iter " + std::to_string(i));

        check(store.get_actual_file_size() == store.get_capacity(),
                "file size stays == capacity at iter " + std::to_string(i));
    }
}

// 2) repeated overwrite of the same slots must not keep allocating new disk blocks.
static void test_blocks_do_not_grow() {
    const uint32_t n_slots     = 8;
    const size_t   cell_stride = 4096;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "blocks_do_not_grow: store enabled");

    // establish a baseline: touch every slot once so its backing blocks are allocated.
    for (uint32_t c = 0; c < n_slots; ++c) {
        auto payload = make_pattern(cell_stride, 1000 + c);
        uint64_t offset = 0;
        const auto ws = store.write_cell(0, c, payload.data(), payload.size(), offset);
        check(ws == llama_kv_backing_store_status::ok, "baseline write ok for cell " + std::to_string(c));
    }

    const uint64_t blocks_after_baseline = store.get_actual_blocks_512();

    // hammer the same slots another 1000 times; no new offsets are ever touched, so block
    // usage must not increase.
    for (uint32_t i = 0; i < 1000; ++i) {
        const uint32_t cell = i % n_slots;
        auto payload = make_pattern(cell_stride, 2000 + i);
        uint64_t offset = 0;
        const auto ws = store.write_cell(0, cell, payload.data(), payload.size(), offset);
        check(ws == llama_kv_backing_store_status::ok, "repeat overwrite ok at iter " + std::to_string(i));
    }

    const uint64_t blocks_final = store.get_actual_blocks_512();
    check(store.get_actual_file_size() == store.get_capacity(), "file size still == capacity after hammering");
    check(blocks_final <= blocks_after_baseline,
            "st_blocks does not grow after repeated in-place overwrites (baseline=" +
            std::to_string(blocks_after_baseline) + " final=" + std::to_string(blocks_final) + ")");
}

// 3) multiple slots do not clobber each other.
static void test_slots_are_independent() {
    const uint32_t n_slots     = 6;
    const size_t   cell_stride = 128;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "slots_independent: store enabled");

    std::vector<std::vector<uint8_t>> payloads(n_slots);
    for (uint32_t c = 0; c < n_slots; ++c) {
        payloads[c] = make_pattern(cell_stride, 42 + c);
        uint64_t offset = 0;
        const auto ws = store.write_cell(0, c, payloads[c].data(), payloads[c].size(), offset);
        check(ws == llama_kv_backing_store_status::ok, "write ok for slot " + std::to_string(c));
    }

    for (uint32_t c = 0; c < n_slots; ++c) {
        std::vector<uint8_t> restored(cell_stride, 0);
        const uint64_t offset = (uint64_t) c * cell_stride;
        const auto rs = store.read_cell(0, c, offset, restored.data(), restored.size());
        check(rs == llama_kv_backing_store_status::ok, "read ok for slot " + std::to_string(c));
        check(std::memcmp(payloads[c].data(), restored.data(), cell_stride) == 0,
                "slot " + std::to_string(c) + " independent of other slots");
    }
}

// 4) out-of-range cell_id and size mismatch are rejected.
static void test_bad_slot_rejected() {
    const uint32_t n_slots     = 4;
    const size_t   cell_stride = 64;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "bad_slot_rejected: store enabled");

    auto payload = make_pattern(cell_stride, 7);
    uint64_t offset = 0;

    // out-of-range cell_id
    auto ws = store.write_cell(0, n_slots, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::bad_slot, "write rejects cell_id >= n_slots");

    ws = store.write_cell(0, n_slots + 100, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::bad_slot, "write rejects far out-of-range cell_id");

    // wrong size
    std::vector<uint8_t> short_payload(cell_stride - 1, 0xAB);
    ws = store.write_cell(0, 0, short_payload.data(), short_payload.size(), offset);
    check(ws == llama_kv_backing_store_status::bad_slot, "write rejects size != cell_stride (too small)");

    std::vector<uint8_t> long_payload(cell_stride + 1, 0xCD);
    ws = store.write_cell(0, 0, long_payload.data(), long_payload.size(), offset);
    check(ws == llama_kv_backing_store_status::bad_slot, "write rejects size != cell_stride (too large)");

    // a legitimate write first, then bad reads
    ws = store.write_cell(0, 1, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "setup write ok before bad-read checks");

    std::vector<uint8_t> restored(cell_stride, 0);
    auto rs = store.read_cell(0, n_slots, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::bad_slot, "read rejects cell_id >= n_slots");

    rs = store.read_cell(0, 1, offset, restored.data(), cell_stride - 1);
    check(rs == llama_kv_backing_store_status::bad_slot, "read rejects size != cell_stride");

    // stale/mismatched offset for an otherwise valid cell_id must also be refused.
    rs = store.read_cell(0, 1, offset + cell_stride, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::bad_slot, "read rejects offset that does not match cell_id");
}

// 5) reset() clears telemetry and re-zeroes the file but keeps it usable with the same geometry.
static void test_reset_then_reuse() {
    const uint32_t n_slots     = 4;
    const size_t   cell_stride = 128;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "reset_then_reuse: store enabled");

    auto payload_a = make_pattern(cell_stride, 11);
    uint64_t offset = 0;
    auto ws = store.write_cell(0, 0, payload_a.data(), payload_a.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "pre-reset write ok");

    const auto reset_status = store.reset();
    check(reset_status == llama_kv_backing_store_status::ok, "reset() succeeds");
    check(store.get_n_slots() == n_slots, "reset() preserves n_slots");
    check(store.get_cell_stride() == cell_stride, "reset() preserves cell_stride");
    check(store.get_capacity() == (uint64_t) n_slots * cell_stride, "reset() preserves capacity");
    check(store.get_actual_file_size() == store.get_capacity(), "reset() restores full-capacity file size");
    check(store.get_stats().write_calls == 0, "reset() clears cumulative write_calls telemetry");

    auto payload_b = make_pattern(cell_stride, 22);
    ws = store.write_cell(0, 0, payload_b.data(), payload_b.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "post-reset write ok");

    std::vector<uint8_t> restored(cell_stride, 0);
    const auto rs = store.read_cell(0, 0, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::ok, "post-reset read ok");
    check(std::memcmp(payload_b.data(), restored.data(), cell_stride) == 0, "post-reset data is the new payload");
}

// 6) 验证大块 payload 往返正确。
static void test_large_payload_roundtrip() {
    const uint32_t n_slots     = 2;
    const size_t   cell_stride = 4 * 1024 * 1024; // 4 MiB

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "large_payload_roundtrip: store enabled");

    auto payload = make_pattern(cell_stride, 99);
    uint64_t offset = 0;
    const auto ws = store.write_cell(0, 1, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "large write ok");

    std::vector<uint8_t> restored(cell_stride, 0);
    const auto rs = store.read_cell(0, 1, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::ok, "large read ok");
    check(std::memcmp(payload.data(), restored.data(), cell_stride) == 0, "large payload byte-identical roundtrip");
    check(store.get_actual_file_size() == store.get_capacity(), "large payload: file size == capacity");
}

// 7) construction rejects impossible geometries before opening/truncating a file.
static void test_capacity_overflow_rejected() {
    const uint32_t n_slots = 2;
    const size_t cell_stride = (size_t) (std::numeric_limits<uint64_t>::max() / n_slots + 1);

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(!store.is_enabled(), "capacity multiplication overflow disables store");
    check(store.get_stats().last_errno == EOVERFLOW, "capacity multiplication overflow reports EOVERFLOW");
}

static void test_off_t_overflow_rejected_when_applicable() {
    const uint64_t off_t_max = (uint64_t) std::numeric_limits<off_t>::max();
    if (off_t_max == std::numeric_limits<uint64_t>::max()) {
        fprintf(stderr, "OK: off_t max equals uint64_t max; off_t overflow test not applicable\n");
        return;
    }

    const uint32_t n_slots = 2;
    const uint64_t capacity = off_t_max + 1;
    const size_t cell_stride = (size_t) ((capacity + n_slots - 1) / n_slots);
    if ((uint64_t) cell_stride <= off_t_max / n_slots) {
        fprintf(stderr, "OK: off_t overflow geometry not representable on this platform; skipped\n");
        return;
    }

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(!store.is_enabled(), "off_t capacity overflow disables store");
    check(store.get_stats().last_errno == EOVERFLOW, "off_t capacity overflow reports EOVERFLOW");
}

int main() {
    try {
        test_same_slot_repeated_roundtrip();
        test_blocks_do_not_grow();
        test_slots_are_independent();
        test_bad_slot_rejected();
        test_reset_then_reuse();
        test_large_payload_roundtrip();
        test_capacity_overflow_rejected();
        test_off_t_overflow_rejected_when_applicable();
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    fprintf(stderr, "all KV backing-store tests passed\n");
    return 0;
}
