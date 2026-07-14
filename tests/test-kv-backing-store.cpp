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

static void check_bytes_equal(
        const std::vector<uint8_t> & expected,
        const std::vector<uint8_t> & actual,
        const std::string & msg) {
    check(expected.size() == actual.size(), msg + " size");
    check(std::memcmp(expected.data(), actual.data(), expected.size()) == 0, msg);
}

static bool all_bytes_are(const std::vector<uint8_t> & data, uint8_t value) {
    for (const uint8_t b : data) {
        if (b != value) {
            return false;
        }
    }
    return true;
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

static void test_fault_free_observability() {
    const uint32_t n_slots     = 2;
    const size_t   cell_stride = 96;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "fault_free_observability: store enabled");

    auto payload = make_pattern(cell_stride, 300);
    uint64_t offset = 0;
    const auto ws = store.write_cell(0, 1, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "fault-free write succeeds");

    std::vector<uint8_t> restored(cell_stride, 0);
    const auto rs = store.read_cell(0, 1, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::ok, "fault-free read succeeds");
    check_bytes_equal(payload, restored, "fault-free data roundtrip");

    const auto & stats = store.get_stats();
    check(stats.syscall_attempts == 2, "fault-free records one write and one read syscall attempt");
    check(stats.write_syscalls == 1, "fault-free records one write syscall");
    check(stats.read_syscalls == 1, "fault-free records one read syscall");
    check(stats.eintr_retries == 0, "fault-free records no EINTR retries");
    check(stats.short_io_events == 0, "fault-free records no short I/O events");
    check(stats.terminal_failures == 0, "fault-free records no terminal failures");
    check(stats.last_status == llama_kv_backing_store_status::ok, "fault-free last status ok");
    check(stats.last_errno == 0, "fault-free last errno zero");
    check(store.get_actual_file_size() == store.get_capacity(), "fault-free file size remains fixed");
}

static void test_eintr_once_retries_and_preserves_data() {
    const uint32_t n_slots     = 2;
    const size_t   cell_stride = 128;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "eintr_once: store enabled");

    auto payload = make_pattern(cell_stride, 301);
    uint64_t offset = 0;

    llama_kv_backing_store_faults faults;
    faults.write_eintr_once = true;
    store.set_test_faults(faults);
    const auto ws = store.write_cell(0, 0, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "write EINTR once retries and succeeds");
    check(!store.get_test_faults().write_eintr_once, "write EINTR fault consumed");

    std::vector<uint8_t> restored(cell_stride, 0);
    faults = {};
    faults.read_eintr_once = true;
    store.set_test_faults(faults);
    const auto rs = store.read_cell(0, 0, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::ok, "read EINTR once retries and succeeds");
    check(!store.get_test_faults().read_eintr_once, "read EINTR fault consumed");
    check_bytes_equal(payload, restored, "EINTR roundtrip data identical");

    const auto & stats = store.get_stats();
    check(stats.eintr_retries == 2, "records both read and write EINTR retries");
    check(stats.syscall_attempts == 4, "EINTR paths record retry attempts");
    check(stats.write_syscalls == 2, "EINTR write path records retry attempts");
    check(stats.read_syscalls == 2, "EINTR read path records retry attempts");
    check(stats.terminal_failures == 0, "EINTR paths have no terminal failures");
    check(store.get_actual_file_size() == store.get_capacity(), "EINTR file size remains fixed");
}

static void test_short_once_completes_remaining_io() {
    const uint32_t n_slots     = 2;
    const size_t   cell_stride = 256;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "short_once: store enabled");

    auto payload = make_pattern(cell_stride, 302);
    uint64_t offset = 0;

    llama_kv_backing_store_faults faults;
    faults.write_short_once = true;
    store.set_test_faults(faults);
    const auto ws = store.write_cell(0, 1, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "short write once completes remaining bytes");
    check(!store.get_test_faults().write_short_once, "write short fault consumed");

    std::vector<uint8_t> restored(cell_stride, 0);
    faults = {};
    faults.read_short_once = true;
    store.set_test_faults(faults);
    const auto rs = store.read_cell(0, 1, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::ok, "short read once completes remaining bytes");
    check(!store.get_test_faults().read_short_once, "read short fault consumed");
    check_bytes_equal(payload, restored, "short I/O roundtrip data identical");

    const auto & stats = store.get_stats();
    check(stats.short_io_events == 2, "records both read and write short I/O events");
    check(stats.syscall_attempts == 4, "short I/O paths record follow-up attempts");
    check(stats.write_syscalls == 2, "short write path records follow-up attempt");
    check(stats.read_syscalls == 2, "short read path records follow-up attempt");
    check(stats.terminal_failures == 0, "short I/O paths have no terminal failures");
    check(store.get_actual_file_size() == store.get_capacity(), "short I/O file size remains fixed");
}

static void test_read_eof_fails_without_exposing_partial_data() {
    const uint32_t n_slots     = 2;
    const size_t   cell_stride = 192;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "read_eof: store enabled");

    auto payload = make_pattern(cell_stride, 303);
    uint64_t offset = 0;
    const auto ws = store.write_cell(0, 0, payload.data(), payload.size(), offset);
    check(ws == llama_kv_backing_store_status::ok, "read EOF setup write succeeds");

    std::vector<uint8_t> restored(cell_stride, 0xA5);
    llama_kv_backing_store_faults faults;
    faults.read_short_once = true;
    faults.read_eof_once   = true;
    store.set_test_faults(faults);
    const auto rs = store.read_cell(0, 0, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::io_error, "read EOF returns io_error");
    check(store.get_stats().last_status == llama_kv_backing_store_status::io_error, "read EOF last status io_error");
    check(store.get_stats().last_errno == EIO, "read EOF maps incomplete read to EIO");
    check(store.get_stats().terminal_failures == 1, "read EOF records terminal failure");
    check(store.get_stats().read_calls == 0, "read EOF does not count as successful read");
    check(store.get_stats().short_io_events == 1, "read EOF after partial read records short event");
    check(all_bytes_are(restored, 0x00), "read EOF clears partial restored data");
    check(store.get_actual_file_size() == store.get_capacity(), "read EOF file size remains fixed");
}

static void test_write_enospc_fails_then_full_retry_overwrites_slot() {
    const uint32_t n_slots     = 2;
    const size_t   cell_stride = 256;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "write_enospc: store enabled");

    auto payload_a = make_pattern(cell_stride, 304);
    auto payload_b = make_pattern(cell_stride, 305);
    uint64_t offset = 0;

    llama_kv_backing_store_faults faults;
    faults.write_short_once  = true;
    faults.write_enospc_once = true;
    store.set_test_faults(faults);
    const auto failed = store.write_cell(0, 1, payload_a.data(), payload_a.size(), offset);
    check(failed == llama_kv_backing_store_status::io_error, "write ENOSPC returns io_error");
    check(store.get_stats().last_status == llama_kv_backing_store_status::io_error, "write ENOSPC last status io_error");
    check(store.get_stats().last_errno == ENOSPC, "write ENOSPC preserves errno");
    check(store.get_stats().terminal_failures == 1, "write ENOSPC records terminal failure");
    check(store.get_stats().write_calls == 0, "failed partial write is not counted as successful write");
    check(store.get_stats().bytes_written == 0, "failed partial write does not add bytes_written");
    check(store.get_stats().short_io_events == 1, "partial write before ENOSPC records short event");

    store.clear_test_faults();
    const auto retry = store.write_cell(0, 1, payload_b.data(), payload_b.size(), offset);
    check(retry == llama_kv_backing_store_status::ok, "write succeeds after one-shot ENOSPC fault");

    std::vector<uint8_t> restored(cell_stride, 0);
    const auto rs = store.read_cell(0, 1, offset, restored.data(), restored.size());
    check(rs == llama_kv_backing_store_status::ok, "read succeeds after ENOSPC retry");
    check_bytes_equal(payload_b, restored, "retry overwrites slot from the first byte");
    check(store.get_stats().syscall_attempts <= 5, "ENOSPC retry path has bounded syscall attempts");
    check(store.get_actual_file_size() == store.get_capacity(), "ENOSPC retry file size remains fixed");
}

static void test_last_errno_cleared_after_error_status_changes() {
    const uint32_t n_slots     = 2;
    const size_t   cell_stride = 128;

    llama_kv_backing_store_file store("", n_slots, cell_stride);
    check(store.is_enabled(), "last_errno_status_changes: store enabled");

    auto payload = make_pattern(cell_stride, 306);
    uint64_t offset = 0;

    llama_kv_backing_store_faults faults;
    faults.write_enospc_once = true;
    store.set_test_faults(faults);
    const auto failed = store.write_cell(0, 0, payload.data(), payload.size(), offset);
    check(failed == llama_kv_backing_store_status::io_error, "last_errno setup ENOSPC fails");
    check(store.get_stats().last_status == llama_kv_backing_store_status::io_error,
            "last_errno setup status io_error");
    check(store.get_stats().last_errno == ENOSPC, "last_errno setup stores ENOSPC");

    const auto bad = store.write_cell(0, n_slots, payload.data(), payload.size(), offset);
    check(bad == llama_kv_backing_store_status::bad_slot, "bad_slot after ENOSPC fails as bad_slot");
    check(store.get_stats().last_status == llama_kv_backing_store_status::bad_slot,
            "bad_slot after ENOSPC updates last_status");
    check(store.get_stats().last_errno == 0, "bad_slot after ENOSPC clears last_errno");

    const auto ok = store.write_cell(0, 0, payload.data(), payload.size(), offset);
    check(ok == llama_kv_backing_store_status::ok, "ok write after bad_slot succeeds");
    check(store.get_stats().last_status == llama_kv_backing_store_status::ok,
            "ok write after bad_slot updates last_status");
    check(store.get_stats().last_errno == 0, "ok write after bad_slot keeps last_errno clear");
}

static void test_contiguous_cell_range_roundtrip() {
    const uint32_t n_slots = 5;
    const size_t stride = 127;
    llama_kv_backing_store_file store("", n_slots, stride);
    check(store.is_enabled(), "range: store enabled");
    std::vector<uint8_t> payload(3 * stride);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t) i;
    uint64_t offset = 0;
    const uint64_t writes_before = store.get_stats().write_syscalls;
    check(store.write_cells(0, 1, 3, payload.data(), payload.size(), offset) == llama_kv_backing_store_status::ok,
            "range write succeeds");
    check(offset == stride, "range write fixed base offset");
    check(store.get_stats().write_syscalls == writes_before + 1, "range write uses one syscall normally");
    std::vector<uint8_t> restored(payload.size());
    const uint64_t reads_before = store.get_stats().read_syscalls;
    check(store.read_cells(0, 1, 3, offset, restored.data(), restored.size()) == llama_kv_backing_store_status::ok,
            "range read succeeds");
    check(store.get_stats().read_syscalls == reads_before + 1, "range read uses one syscall normally");
    check_bytes_equal(payload, restored, "range byte-identical");
    check(store.write_cells(0, n_slots, 1, payload.data(), stride, offset) == llama_kv_backing_store_status::bad_slot,
            "range rejects begin out of bounds");
    check(store.write_cells(0, 4, 2, payload.data(), 2 * stride, offset) == llama_kv_backing_store_status::bad_slot,
            "range rejects count out of bounds");
    check(store.write_cells(0, 1, 3, payload.data(), payload.size() - 1, offset) == llama_kv_backing_store_status::bad_slot,
            "range rejects bad total size");
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
        test_fault_free_observability();
        test_eintr_once_retries_and_preserves_data();
        test_short_once_completes_remaining_io();
        test_read_eof_fails_without_exposing_partial_data();
        test_write_enospc_fails_then_full_retry_overwrites_slot();
        test_last_errno_cleared_after_error_status_changes();
        test_contiguous_cell_range_roundtrip();
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }

    fprintf(stderr, "all KV backing-store tests passed\n");
    return 0;
}
