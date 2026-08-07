#include "llama-kv-cache.h"
#include "llama-kv-cache-release.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <set>
#include <stdexcept>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__linux__) && defined(__GLIBC__)
#include <execinfo.h>
#include <sys/resource.h>
#if defined(RUSAGE_THREAD)
#define LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED 1
#endif
#if defined(MADV_POPULATE_WRITE)
#define LLAMA_KV_RESTORE_PREFAULT_SUPPORTED 1
#endif
#define LLAMA_KV_REFAULT_TRACE_SUPPORTED 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LLAMA_KV_USED __attribute__((used))
#else
#define LLAMA_KV_USED
#endif

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

static uint64_t llama_paged_timing_now_us() {
    using clock = std::chrono::steady_clock;
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
            clock::now().time_since_epoch()).count();
}

#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
static bool llama_paged_restore_read_thread_faults(
        uint64_t & minor_faults,
        uint64_t & major_faults) {
    struct rusage usage = {};
    if (getrusage(RUSAGE_THREAD, &usage) != 0 || usage.ru_minflt < 0 || usage.ru_majflt < 0) {
        return false;
    }
    minor_faults = (uint64_t) usage.ru_minflt;
    major_faults = (uint64_t) usage.ru_majflt;
    return true;
}
#endif

static uint64_t llama_paged_restore_staging_bound_bytes(
        size_t group_byte_cap, size_t max_single_block_bytes) {
    const size_t max_slot_bytes = std::max(group_byte_cap, max_single_block_bytes);
    if (max_slot_bytes > std::numeric_limits<uint64_t>::max() / 2) {
        return std::numeric_limits<uint64_t>::max();
    }
    return (uint64_t) max_slot_bytes * 2;
}

static bool llama_kv_parse_u64_strict(const char * value, uint64_t & result) {
    if (!value || value[0] == '\0') {
        return false;
    }

    uint64_t parsed = 0;
    for (const char * p = value; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') {
            return false;
        }

        const uint64_t digit = (uint64_t) (*p - '0');
        if (parsed > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        parsed = parsed * 10 + digit;
    }

    result = parsed;
    return true;
}

static uint64_t llama_kv_next_resident_object_id() {
    static std::atomic<uint64_t> next { 1 };
    return next.fetch_add(1, std::memory_order_relaxed);
}

static const char llama_kv_stability_binary_markers[] LLAMA_KV_USED =
    "KV_STABILITY_SUMMARY llama_kv_cache_paged_stability_cycle target_blocks backing_stat_valid";

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

//
// llama_kv_backing_store_file
//

static const char * llama_kv_backing_store_status_name(llama_kv_backing_store_status status) {
    switch (status) {
        case llama_kv_backing_store_status::ok:       return "ok";
        case llama_kv_backing_store_status::disabled: return "disabled";
        case llama_kv_backing_store_status::io_error: return "io_error";
        case llama_kv_backing_store_status::bad_slot: return "bad_slot";
    }

    return "unknown";
}

llama_kv_backing_store_file::llama_kv_backing_store_file(
        const std::string & dir_req,
        uint32_t n_slots_,
        size_t cell_stride_)
    : dir(dir_req.empty() ? std::string("/tmp") : dir_req), n_slots(n_slots_), cell_stride(cell_stride_) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    if (n_slots == 0 || cell_stride == 0) {
        stats.last_errno = EOVERFLOW;
        return;
    }
    if ((uint64_t) cell_stride > std::numeric_limits<uint64_t>::max() / (uint64_t) n_slots) {
        stats.last_errno = EOVERFLOW;
        return;
    }

    const uint64_t capacity_ = (uint64_t) n_slots * (uint64_t) cell_stride;
    if (capacity_ > (uint64_t) std::numeric_limits<off_t>::max()) {
        stats.last_errno = EOVERFLOW;
        return;
    }

    capacity = capacity_;

    // LLAMA_KV_SWAP_DIR must be honored exactly: if the directory is not usable, fail rather
    // than silently falling back to some other location the caller did not ask for.
    struct stat st;
    if (::stat(dir.c_str(), &st) != 0) {
        stats.last_errno = errno;
        return;
    }
    if (!S_ISDIR(st.st_mode)) {
        stats.last_errno = ENOTDIR;
        return;
    }

#if defined(O_TMPFILE)
    fd = open(dir.c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd >= 0) {
        used_o_tmpfile_ = true;
    } else {
        stats.last_errno = errno;
    }
#endif

    if (fd < 0) {
        // O_TMPFILE unavailable/unsupported on this directory's filesystem: fall back to
        // mkstemp()+unlink() *within the same directory* so the file is still anonymous and
        // still under the caller-specified path. Never falls back to a different directory.
        std::string tmpl = dir + "/llama-kv-swap-XXXXXX";
        std::vector<char> buf(tmpl.begin(), tmpl.end());
        buf.push_back('\0');

        const int tmp_fd = mkstemp(buf.data());
        if (tmp_fd < 0) {
            stats.last_errno = errno;
            return;
        }

        if (unlink(buf.data()) != 0) {
            stats.last_errno = errno;
            close(tmp_fd);
            return;
        }
        const int fd_flags = fcntl(tmp_fd, F_GETFD);
        if (fd_flags < 0 || fcntl(tmp_fd, F_SETFD, fd_flags | FD_CLOEXEC) != 0) {
            stats.last_errno = errno;
            close(tmp_fd);
            return;
        }

        fd = tmp_fd;
        used_o_tmpfile_ = false;
    }

    if (ftruncate(fd, (off_t) capacity) != 0) {
        stats.last_errno = errno;
        close(fd);
        fd = -1;
        return;
    }

    stats.bytes_capacity = capacity;
    stats.last_errno = 0;
#else
    stats.last_errno = ENOSYS;
#endif
}

static bool llama_kv_fixed_slot_bounds(
        uint32_t   cell,
        uint32_t   n_slots,
        size_t     cell_stride,
        uint64_t   capacity,
        uint64_t & offset_out) {
    offset_out = 0;
    if (cell >= n_slots || cell_stride == 0) {
        return false;
    }
    if (cell != 0 && (uint64_t) cell_stride > std::numeric_limits<uint64_t>::max() / (uint64_t) cell) {
        return false;
    }

    const uint64_t offset = (uint64_t) cell * (uint64_t) cell_stride;
    if (offset > capacity || (uint64_t) cell_stride > capacity - offset) {
        return false;
    }

    offset_out = offset;
    return true;
}

static bool llama_kv_fixed_slot_range_bounds(
        uint32_t begin_cell, uint32_t cell_count, uint32_t n_slots, size_t cell_stride,
        uint64_t capacity, uint64_t & offset_out, size_t & total_size_out) {
    offset_out = 0;
    total_size_out = 0;
    if (cell_count == 0 || begin_cell >= n_slots || cell_count > n_slots - begin_cell || cell_stride == 0 ||
            cell_count > std::numeric_limits<size_t>::max() / cell_stride) {
        return false;
    }
    if (!llama_kv_fixed_slot_bounds(begin_cell, n_slots, cell_stride, capacity, offset_out)) {
        return false;
    }
    total_size_out = (size_t) cell_count * cell_stride;
    if ((uint64_t) total_size_out > capacity - offset_out ||
            offset_out > (uint64_t) std::numeric_limits<off_t>::max() ||
            (uint64_t) total_size_out > (uint64_t) std::numeric_limits<off_t>::max() - offset_out) {
        return false;
    }
    return true;
}

llama_kv_backing_store_file::~llama_kv_backing_store_file() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
#endif
}

bool llama_kv_backing_store_file::stat_actual(uint64_t & actual_file_size, uint64_t & actual_blocks_512) const {
    actual_file_size = 0;
    actual_blocks_512 = 0;
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    if (fd < 0) {
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0) {
        return false;
    }
    actual_file_size = (uint64_t) st.st_size;
    actual_blocks_512 = (uint64_t) st.st_blocks;
    return true;
#else
    return false;
#endif
}

uint64_t llama_kv_backing_store_file::get_actual_file_size() const {
    uint64_t actual_file_size = 0;
    uint64_t actual_blocks_512 = 0;
    (void) stat_actual(actual_file_size, actual_blocks_512);
    return actual_file_size;
}

uint64_t llama_kv_backing_store_file::get_actual_blocks_512() const {
    uint64_t actual_file_size = 0;
    uint64_t actual_blocks_512 = 0;
    (void) stat_actual(actual_file_size, actual_blocks_512);
    return actual_blocks_512;
}

void llama_kv_backing_store_file::set_test_faults(const llama_kv_backing_store_faults & faults_) {
    faults = faults_;
}

void llama_kv_backing_store_file::clear_test_faults() {
    faults = {};
    read_gate_armed.store(false, std::memory_order_release);
    read_gate_released.store(true, std::memory_order_release);
}

void llama_kv_backing_store_file::arm_test_read_gate() {
    read_gate_released.store(false, std::memory_order_release);
    read_gate_entered_flag.store(false, std::memory_order_release);
    read_gate_armed.store(true, std::memory_order_release);
}

bool llama_kv_backing_store_file::test_read_gate_entered() const {
    return read_gate_entered_flag.load(std::memory_order_acquire);
}

void llama_kv_backing_store_file::release_test_read_gate() {
    read_gate_released.store(true, std::memory_order_release);
}

llama_kv_backing_store_status llama_kv_backing_store_file::finish_status(
        llama_kv_backing_store_status status,
        int err) {
    stats.last_status = status;

    switch (status) {
        case llama_kv_backing_store_status::ok:
        case llama_kv_backing_store_status::bad_slot:
            stats.last_errno = 0;
            break;
        case llama_kv_backing_store_status::io_error:
            stats.last_errno = err != 0 ? err : EIO;
            stats.terminal_failures += 1;
            break;
        case llama_kv_backing_store_status::disabled:
            stats.last_errno = err;
            break;
    }

    return status;
}

int64_t llama_kv_backing_store_file::pwrite_once(const char * data, size_t size, uint64_t offset) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    stats.syscall_attempts += 1;
    stats.write_syscalls += 1;
    if (faults.write_eintr_once) {
        faults.write_eintr_once = false;
        errno = EINTR;
        return -1;
    }

    size_t request = size;
    if (faults.write_short_once && size > 1) {
        faults.write_short_once = false;
        request = std::max<size_t>(1, size / 2);
        if (request >= size) {
            request = size - 1;
        }
    }
    if (request == size && faults.write_enospc_once) {
        faults.write_enospc_once = false;
        errno = ENOSPC;
        return -1;
    }

    const ssize_t ret = pwrite(fd, data, request, (off_t) offset);
    if (ret > 0 && (size_t) ret < size) {
        stats.short_io_events += 1;
    }
    return (int64_t) ret;
#else
    (void) data;
    (void) size;
    (void) offset;
    errno = ENOSYS;
    return -1;
#endif
}

int64_t llama_kv_backing_store_file::pread_once(char * data, size_t size, uint64_t offset) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    stats.syscall_attempts += 1;
    stats.read_syscalls += 1;
    if (faults.read_eintr_once) {
        faults.read_eintr_once = false;
        errno = EINTR;
        return -1;
    }

    size_t request = size;
    if (faults.read_short_once && size > 1) {
        faults.read_short_once = false;
        request = std::max<size_t>(1, size / 2);
        if (request >= size) {
            request = size - 1;
        }
    }
    if (request == size && faults.read_eof_once) {
        faults.read_eof_once = false;
        return 0;
    }

    const ssize_t ret = pread(fd, data, request, (off_t) offset);
    if (ret > 0 && (size_t) ret < size) {
        stats.short_io_events += 1;
    }
    return (int64_t) ret;
#else
    (void) data;
    (void) size;
    (void) offset;
    errno = ENOSYS;
    return -1;
#endif
}

llama_kv_backing_store_status llama_kv_backing_store_file::write_cell(
        uint32_t   strm,
        uint32_t   cell,
        const void * data,
        size_t     size,
        uint64_t & offset_out) {
    return write_cells(strm, cell, 1, data, size, offset_out);
}

llama_kv_backing_store_status llama_kv_backing_store_file::write_cells(
        uint32_t strm, uint32_t begin_cell, uint32_t cell_count, const void * data,
        size_t total_size, uint64_t & offset_out) {
    (void) strm;
    offset_out = 0;
    if (fd < 0) return finish_status(llama_kv_backing_store_status::disabled, stats.last_errno);
    uint64_t offset = 0;
    size_t expected_size = 0;
    if (!data || !llama_kv_fixed_slot_range_bounds(
            begin_cell, cell_count, n_slots, cell_stride, capacity, offset, expected_size) ||
            total_size != expected_size) {
        return finish_status(llama_kv_backing_store_status::bad_slot, 0);
    }
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    const char * ptr = static_cast<const char *>(data);
    size_t written = 0;

    while (written < total_size) {
        if (offset + written > (uint64_t) std::numeric_limits<off_t>::max()) {
            return finish_status(llama_kv_backing_store_status::io_error, EOVERFLOW);
        }
        const int64_t ret = pwrite_once(ptr + written, total_size - written, offset + written);
        if (ret < 0) {
            if (errno == EINTR) {
                stats.eintr_retries += 1;
                continue;
            }
            return finish_status(llama_kv_backing_store_status::io_error, errno);
        }
        if (ret == 0) {
            return finish_status(llama_kv_backing_store_status::io_error, EIO);
        }
        written += (size_t) ret;
    }

    offset_out = offset;
    stats.bytes_written += total_size;
    stats.write_calls += 1;
    stats.last_errno = 0;

    return finish_status(llama_kv_backing_store_status::ok, 0);
#else
    return finish_status(llama_kv_backing_store_status::disabled, ENOSYS);
#endif
}

llama_kv_backing_store_status llama_kv_backing_store_file::read_cell(
        uint32_t strm,
        uint32_t cell,
        uint64_t offset,
        void *   data,
        size_t   size) {
    return read_cells(strm, cell, 1, offset, data, size);
}

llama_kv_backing_store_status llama_kv_backing_store_file::read_cells(
        uint32_t strm, uint32_t begin_cell, uint32_t cell_count, uint64_t offset,
        void * data, size_t total_size) {
    (void) strm;
    if (fd < 0) return finish_status(llama_kv_backing_store_status::disabled, stats.last_errno);
    uint64_t expected_offset = 0;
    size_t expected_size = 0;
    if (!data || !llama_kv_fixed_slot_range_bounds(
            begin_cell, cell_count, n_slots, cell_stride, capacity, expected_offset, expected_size) ||
            offset != expected_offset || total_size != expected_size) {
        return finish_status(llama_kv_backing_store_status::bad_slot, 0);
    }

    if (read_gate_armed.load(std::memory_order_acquire)) {
        read_gate_entered_flag.store(true, std::memory_order_release);
        while (!read_gate_released.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        read_gate_armed.store(false, std::memory_order_release);
    }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    char * ptr = static_cast<char *>(data);
    size_t read_bytes = 0;

    while (read_bytes < total_size) {
        if (expected_offset + read_bytes > (uint64_t) std::numeric_limits<off_t>::max()) {
            std::memset(data, 0, total_size);
            return finish_status(llama_kv_backing_store_status::io_error, EOVERFLOW);
        }
        const int64_t ret = pread_once(ptr + read_bytes, total_size - read_bytes, expected_offset + read_bytes);
        if (ret < 0) {
            if (errno == EINTR) {
                stats.eintr_retries += 1;
                continue;
            }
            std::memset(data, 0, total_size);
            return finish_status(llama_kv_backing_store_status::io_error, errno);
        }
        if (ret == 0) {
            std::memset(data, 0, total_size);
            return finish_status(llama_kv_backing_store_status::io_error, EIO);
        }
        read_bytes += (size_t) ret;
    }

    stats.bytes_read += total_size;
    stats.read_calls += 1;
    stats.last_errno = 0;

    return finish_status(llama_kv_backing_store_status::ok, 0);
#else
    return finish_status(llama_kv_backing_store_status::disabled, ENOSYS);
#endif
}

llama_kv_backing_store_status llama_kv_backing_store_file::release(uint64_t offset, size_t size) {
    if (fd < 0) {
        return finish_status(llama_kv_backing_store_status::disabled, stats.last_errno);
    }
    if (size == 0 || size != cell_stride || cell_stride == 0 ||
            offset % cell_stride != 0 || offset > capacity || size > capacity - offset) {
        return finish_status(llama_kv_backing_store_status::bad_slot, 0);
    }

    stats.bytes_released += size;
    stats.release_calls += 1;

    return finish_status(llama_kv_backing_store_status::ok, 0);
}

llama_kv_backing_store_status llama_kv_backing_store_file::reset() {
    // reset() only clears cumulative I/O telemetry and re-zeroes the fixed-capacity file; the
    // slot geometry (n_slots/cell_stride/capacity) is structural and survives reset so the
    // store can be reused immediately with the same addressing.
    if (fd < 0) {
        return finish_status(llama_kv_backing_store_status::io_error, stats.last_errno);
    }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    // truncate to 0 then back up to capacity to re-establish a zero-filled sparse file of the
    // same fixed size (portable across tmpfs/regular filesystems; avoids relying on
    // FALLOC_FL_PUNCH_HOLE support).
    if (ftruncate(fd, 0) != 0) {
        return finish_status(llama_kv_backing_store_status::io_error, errno);
    }
    if (ftruncate(fd, (off_t) capacity) != 0) {
        const int err = errno;
        close(fd);
        fd = -1;
        return finish_status(llama_kv_backing_store_status::io_error, err);
    }

    stats.bytes_written  = 0;
    stats.bytes_read     = 0;
    stats.bytes_released = 0;
    stats.write_calls    = 0;
    stats.read_calls     = 0;
    stats.release_calls  = 0;
    stats.syscall_attempts  = 0;
    stats.read_syscalls     = 0;
    stats.write_syscalls    = 0;
    stats.eintr_retries     = 0;
    stats.short_io_events   = 0;
    stats.terminal_failures = 0;
    stats.last_errno = 0;
    return finish_status(llama_kv_backing_store_status::ok, 0);
#else
    return finish_status(llama_kv_backing_store_status::disabled, ENOSYS);
#endif
}

static void llama_kv_backing_store_selftest_once() {
    static bool checked = false;
    if (checked) {
        return;
    }
    checked = true;

    const char * env = std::getenv("LLAMA_KV_SWAP_BACKEND_SELFTEST");
    if (!env || std::atoi(env) == 0) {
        return;
    }

    const char payload[] = "hello-kv";

    llama_kv_backing_store_file store("", /*n_slots=*/1, /*cell_stride=*/sizeof(payload));
    if (!store.is_enabled()) {
        const auto & stats = store.get_stats();
        LLAMA_LOG_ERROR("KV_SWAP_BACKEND_SELFTEST: backing store selftest fail status=disabled errno=%d\n",
                stats.last_errno);
        return;
    }

    char restored[sizeof(payload)] = {};
    uint64_t offset = 0;

    auto write_status = store.write_cell(0, 0, payload, sizeof(payload), offset);
    auto read_status  = store.read_cell (0, 0, offset, restored, sizeof(restored));
    const bool same = std::memcmp(payload, restored, sizeof(payload)) == 0;
    auto release_status = store.release(offset, sizeof(payload));

    const auto stats_before_reset = store.get_stats();
    auto reset_status = store.reset();

    const bool pass =
        write_status   == llama_kv_backing_store_status::ok &&
        read_status    == llama_kv_backing_store_status::ok &&
        release_status == llama_kv_backing_store_status::ok &&
        reset_status   == llama_kv_backing_store_status::ok &&
        same;

    LLAMA_LOG_INFO("KV_SWAP_BACKEND_SELFTEST: backing store selftest %s write=%s read=%s release=%s reset=%s "
            "bytes_written=%llu bytes_read=%llu released_bytes=%llu write_calls=%llu read_calls=%llu release_calls=%llu\n",
            pass ? "pass" : "fail",
            llama_kv_backing_store_status_name(write_status),
            llama_kv_backing_store_status_name(read_status),
            llama_kv_backing_store_status_name(release_status),
            llama_kv_backing_store_status_name(reset_status),
            (unsigned long long) stats_before_reset.bytes_written,
            (unsigned long long) stats_before_reset.bytes_read,
            (unsigned long long) stats_before_reset.bytes_released,
            (unsigned long long) stats_before_reset.write_calls,
            (unsigned long long) stats_before_reset.read_calls,
            (unsigned long long) stats_before_reset.release_calls);
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse) :
    model(model), hparams(model.hparams), v_trans(v_trans), kv_unified(unified),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    paged_resident_object_id = llama_kv_next_resident_object_id();

    llama_kv_backing_store_selftest_once();

    // KV-P0-B2B-1: deterministic paged swap-in read failure for dynamic tests. Parse once per
    // KV cache; the hot path only reads this context-local state and never calls getenv().
    const char * test_swapin_scope_env = std::getenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_SCOPE");
    const char * test_swapin_after_env = std::getenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_AFTER_CELLS");
    const char * test_swapin_once_env  = std::getenv("LLAMA_KV_PAGED_TEST_SWAPIN_FAIL_ONCE");

    bool test_swapin_config_valid = true;
    if (!test_swapin_scope_env || std::strcmp(test_swapin_scope_env, "off") == 0) {
        paged_test_swapin_fault_.scope = paged_test_swapin_fail_scope::OFF;
    } else if (std::strcmp(test_swapin_scope_env, "prefetch") == 0) {
        paged_test_swapin_fault_.scope = paged_test_swapin_fail_scope::PREFETCH;
    } else if (std::strcmp(test_swapin_scope_env, "active") == 0) {
        paged_test_swapin_fault_.scope = paged_test_swapin_fail_scope::ACTIVE;
    } else {
        test_swapin_config_valid = false;
    }

    if (paged_test_swapin_fault_.scope != paged_test_swapin_fail_scope::OFF) {
        if (test_swapin_after_env) {
            test_swapin_config_valid =
                llama_kv_parse_u64_strict(test_swapin_after_env, paged_test_swapin_fault_.fail_after_cells) &&
                test_swapin_config_valid;
        } else {
            test_swapin_config_valid = false;
        }

        if (test_swapin_once_env) {
            uint64_t fail_once = 0;
            if (!llama_kv_parse_u64_strict(test_swapin_once_env, fail_once) || fail_once > 1) {
                test_swapin_config_valid = false;
            } else {
                paged_test_swapin_fault_.fail_once = fail_once != 0;
            }
        }
    }

    if (!test_swapin_config_valid) {
        LLAMA_LOG_WARN(
                "%s: invalid paged swap-in test fault configuration "
                "(scope=%s after_cells=%s fail_once=%s); disabling TEST FAULT INJECTION\n",
                __func__,
                test_swapin_scope_env ? test_swapin_scope_env : "<unset>",
                test_swapin_after_env ? test_swapin_after_env : "<unset>",
                test_swapin_once_env  ? test_swapin_once_env  : "<unset>");
        paged_test_swapin_fault_ = {};
    } else if (paged_test_swapin_fault_.scope != paged_test_swapin_fail_scope::OFF) {
        LLAMA_LOG_INFO(
                "TEST FAULT INJECTION ENABLED scope=%s fail_after_cells=%llu fail_once=%d\n",
                test_swapin_scope_env,
                (unsigned long long) paged_test_swapin_fault_.fail_after_cells,
                paged_test_swapin_fault_.fail_once ? 1 : 0);
    }

    const char * test_mapping_scope_env = std::getenv("LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SCOPE");
    const char * test_mapping_seq_env   = std::getenv("LLAMA_KV_PAGED_TEST_MAPPING_FAIL_SEQ_ID");
    const char * test_mapping_once_env  = std::getenv("LLAMA_KV_PAGED_TEST_MAPPING_FAIL_ONCE");

    bool test_mapping_config_valid = true;
    if (!test_mapping_scope_env || std::strcmp(test_mapping_scope_env, "off") == 0) {
        paged_test_mapping_fault_.scope = paged_test_mapping_fail_scope::OFF;
    } else if (std::strcmp(test_mapping_scope_env, "read") == 0) {
        paged_test_mapping_fault_.scope = paged_test_mapping_fail_scope::READ;
    } else if (std::strcmp(test_mapping_scope_env, "write") == 0) {
        paged_test_mapping_fault_.scope = paged_test_mapping_fail_scope::WRITE;
    } else {
        test_mapping_config_valid = false;
    }

    if (paged_test_mapping_fault_.scope != paged_test_mapping_fail_scope::OFF) {
        uint64_t target_seq = 0;
        if (!test_mapping_seq_env ||
                !llama_kv_parse_u64_strict(test_mapping_seq_env, target_seq) ||
                target_seq >= LLAMA_MAX_SEQ ||
                target_seq > (uint64_t) std::numeric_limits<llama_seq_id>::max()) {
            test_mapping_config_valid = false;
        } else {
            paged_test_mapping_fault_.target_seq = (llama_seq_id) target_seq;
        }

        if (test_mapping_once_env) {
            uint64_t fail_once = 0;
            if (!llama_kv_parse_u64_strict(test_mapping_once_env, fail_once) || fail_once > 1) {
                test_mapping_config_valid = false;
            } else {
                paged_test_mapping_fault_.fail_once = fail_once != 0;
            }
        }
    }

    if (!test_mapping_config_valid) {
        LLAMA_LOG_WARN(
                "%s: invalid paged mapping test fault configuration "
                "(scope=%s seq_id=%s fail_once=%s); disabling TEST MAPPING FAULT INJECTION\n",
                __func__,
                test_mapping_scope_env ? test_mapping_scope_env : "<unset>",
                test_mapping_seq_env   ? test_mapping_seq_env   : "<unset>",
                test_mapping_once_env  ? test_mapping_once_env  : "<unset>");
        paged_test_mapping_fault_ = {};
    } else if (paged_test_mapping_fault_.scope != paged_test_mapping_fail_scope::OFF) {
        LLAMA_LOG_INFO(
                "TEST MAPPING FAULT INJECTION ENABLED scope=%s target_seq=%d fail_once=%d\n",
                test_mapping_scope_env,
                (int) paged_test_mapping_fault_.target_seq,
                paged_test_mapping_fault_.fail_once ? 1 : 0);
    }

    const char * test_io_scope_env = std::getenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SCOPE");
    const char * test_io_kind_env  = std::getenv("LLAMA_KV_PAGED_TEST_IO_FAIL_KIND");
    const char * test_io_seq_env   = std::getenv("LLAMA_KV_PAGED_TEST_IO_FAIL_SEQ_ID");
    const char * test_io_block_env = std::getenv("LLAMA_KV_PAGED_TEST_IO_FAIL_BLOCK");
    const char * test_io_once_env  = std::getenv("LLAMA_KV_PAGED_TEST_IO_FAIL_ONCE");

    bool test_io_config_valid = true;
    if (!test_io_scope_env || std::strcmp(test_io_scope_env, "off") == 0) {
        paged_test_io_fault_.scope = paged_test_io_fail_scope::OFF;
    } else if (std::strcmp(test_io_scope_env, "swap_out") == 0) {
        paged_test_io_fault_.scope = paged_test_io_fail_scope::SWAP_OUT;
    } else if (std::strcmp(test_io_scope_env, "active_swap_in") == 0) {
        paged_test_io_fault_.scope = paged_test_io_fail_scope::ACTIVE_SWAP_IN;
    } else if (std::strcmp(test_io_scope_env, "prefetch_swap_in") == 0) {
        paged_test_io_fault_.scope = paged_test_io_fail_scope::PREFETCH_SWAP_IN;
    } else {
        test_io_config_valid = false;
    }

    if (!test_io_kind_env || std::strcmp(test_io_kind_env, "none") == 0) {
        paged_test_io_fault_.kind = paged_test_io_fail_kind::NONE;
    } else if (std::strcmp(test_io_kind_env, "write_enospc_once") == 0) {
        paged_test_io_fault_.kind = paged_test_io_fail_kind::WRITE_ENOSPC_ONCE;
    } else if (std::strcmp(test_io_kind_env, "read_eof_once") == 0) {
        paged_test_io_fault_.kind = paged_test_io_fail_kind::READ_EOF_ONCE;
    } else {
        test_io_config_valid = false;
    }

    if (paged_test_io_fault_.scope != paged_test_io_fail_scope::OFF) {
        uint64_t target_seq = 0;
        if (!test_io_seq_env ||
                !llama_kv_parse_u64_strict(test_io_seq_env, target_seq) ||
                target_seq >= LLAMA_MAX_SEQ ||
                target_seq > (uint64_t) std::numeric_limits<llama_seq_id>::max()) {
            test_io_config_valid = false;
        } else {
            paged_test_io_fault_.target_seq = (llama_seq_id) target_seq;
        }

        if (paged_test_io_fault_.scope == paged_test_io_fail_scope::SWAP_OUT ||
                paged_test_io_fault_.scope == paged_test_io_fail_scope::PREFETCH_SWAP_IN) {
            uint64_t target_block = 0;
            if (!test_io_block_env ||
                    !llama_kv_parse_u64_strict(test_io_block_env, target_block) ||
                    target_block > (uint64_t) std::numeric_limits<uint32_t>::max()) {
                test_io_config_valid = false;
            } else {
                paged_test_io_fault_.target_block = (uint32_t) target_block;
            }
        }

        if (test_io_once_env) {
            uint64_t fail_once = 0;
            if (!llama_kv_parse_u64_strict(test_io_once_env, fail_once) || fail_once > 1) {
                test_io_config_valid = false;
            } else {
                paged_test_io_fault_.fail_once = fail_once != 0;
            }
        }

        if (paged_test_io_fault_.scope == paged_test_io_fail_scope::SWAP_OUT &&
                paged_test_io_fault_.kind != paged_test_io_fail_kind::WRITE_ENOSPC_ONCE) {
            test_io_config_valid = false;
        }
        if ((paged_test_io_fault_.scope == paged_test_io_fail_scope::ACTIVE_SWAP_IN ||
                paged_test_io_fault_.scope == paged_test_io_fail_scope::PREFETCH_SWAP_IN) &&
                paged_test_io_fault_.kind != paged_test_io_fail_kind::READ_EOF_ONCE) {
            test_io_config_valid = false;
        }
    }

    if (!test_io_config_valid) {
        LLAMA_LOG_WARN(
                "%s: invalid paged backing-store I/O test fault configuration "
                "(scope=%s kind=%s seq_id=%s block=%s fail_once=%s); disabling KV_PAGED_IO_FAULT\n",
                __func__,
                test_io_scope_env ? test_io_scope_env : "<unset>",
                test_io_kind_env  ? test_io_kind_env  : "<unset>",
                test_io_seq_env   ? test_io_seq_env   : "<unset>",
                test_io_block_env ? test_io_block_env : "<unset>",
                test_io_once_env  ? test_io_once_env  : "<unset>");
        paged_test_io_fault_ = {};
    } else if (paged_test_io_fault_.scope != paged_test_io_fail_scope::OFF) {
        LLAMA_LOG_INFO(
                "KV_PAGED_IO_FAULT_ENABLED scope=%s kind=%s target_seq=%d fail_once=%d\n",
                test_io_scope_env,
                test_io_kind_env ? test_io_kind_env : "none",
                (int) paged_test_io_fault_.target_seq,
                paged_test_io_fault_.fail_once ? 1 : 0);
    }

    // KV-P0-B1: shared backing-store directory for both exact and paged swap. Honored exactly
    // as given (see llama_kv_backing_store_file ctor) - no silent fallback to a different dir.
    const char * LLAMA_KV_SWAP_DIR = std::getenv("LLAMA_KV_SWAP_DIR");
    const std::string kv_swap_dir = LLAMA_KV_SWAP_DIR ? LLAMA_KV_SWAP_DIR : "";

    // Resolved to true below if exact swap passes its own validity checks. Actual backing-store
    // construction (and the exact/paged mutual-exclusion decision) is deferred until the KV
    // layer tensors exist, because the fixed per-cell slot stride is computed from their row
    // sizes (see the deferred construction block after the layer loop).
    bool kv_swap_want_exact = false;

    const char * LLAMA_KV_SWAP      = std::getenv("LLAMA_KV_SWAP");
    const char * LLAMA_KV_SWAP_MODE = std::getenv("LLAMA_KV_SWAP_MODE");
    const char * LLAMA_KV_SWAP_WINDOW = std::getenv("LLAMA_KV_SWAP_WINDOW");
    const char * LLAMA_KV_SWAP_SINK   = std::getenv("LLAMA_KV_SWAP_SINK");
    const char * LLAMA_KV_SWAP_RSS_SAMPLE = std::getenv("LLAMA_KV_SWAP_RSS_SAMPLE");
    const char * LLAMA_KV_SWAP_MADVISE = std::getenv("LLAMA_KV_SWAP_MADVISE");
    kv_swap_window = LLAMA_KV_SWAP_WINDOW ? std::max(0, std::atoi(LLAMA_KV_SWAP_WINDOW)) : 0;
    kv_swap_sink   = LLAMA_KV_SWAP_SINK   ? std::max(0, std::atoi(LLAMA_KV_SWAP_SINK))   : 0;
    kv_swap_rss_sample = LLAMA_KV_SWAP_RSS_SAMPLE ? (std::atoi(LLAMA_KV_SWAP_RSS_SAMPLE) != 0) : false;
    const bool kv_swap_madvise_requested = LLAMA_KV_SWAP_MADVISE ? (std::atoi(LLAMA_KV_SWAP_MADVISE) != 0) : false;

    const bool kv_swap_requested = LLAMA_KV_SWAP ? (std::atoi(LLAMA_KV_SWAP) != 0) : false;
    if (kv_swap_requested) {
        if (!LLAMA_KV_SWAP_MODE) {
            LLAMA_LOG_WARN("%s: KV swap requested but LLAMA_KV_SWAP_MODE=%s is unsupported "
                    "(expected exact or approx) - disabled\n", __func__, "<unset>");
        } else if (std::strcmp(LLAMA_KV_SWAP_MODE, "approx") == 0) {
            if (v_trans || n_stream != 1 || n_seq_max != 1) {
                LLAMA_LOG_WARN("%s: KV swap approx mode requires single-seq, !v_trans, and n_stream==1 "
                        "(n_seq_max=%u, v_trans=%d, n_stream=%u) - disabled\n",
                        __func__, n_seq_max, (int) v_trans, n_stream);
            } else {
                kv_swap_enabled = true;
                kv_swap_mode_   = kv_swap_mode::approx;
                kv_approx_window = kv_swap_window;
                LLAMA_LOG_INFO("%s: KV swap approx mode enabled (no-op scaffold, window=%u, sink=%u)\n",
                        __func__, kv_swap_window, kv_swap_sink);
            }
        } else if (std::strcmp(LLAMA_KV_SWAP_MODE, "exact") != 0) {
            LLAMA_LOG_WARN("%s: KV swap requested but LLAMA_KV_SWAP_MODE=%s is unsupported "
                    "(expected exact or approx) - disabled\n", __func__, LLAMA_KV_SWAP_MODE);
        } else if (v_trans || n_stream != 1) {
            LLAMA_LOG_WARN("%s: KV swap exact mode requires !v_trans && n_stream==1 "
                    "(v_trans=%d, n_stream=%u) - disabled\n", __func__, (int) v_trans, n_stream);
        } else {
            // backing-store construction deferred: see kv_swap_want_exact declaration above.
            kv_swap_want_exact = true;
        }
    }

    const char * LLAMA_KV_PAGED = std::getenv("LLAMA_KV_PAGED");
    const char * LLAMA_KV_PAGED_INGRAPH = std::getenv("LLAMA_KV_PAGED_INGRAPH");
    const char * LLAMA_KV_PAGED_GATHER_NONIDENTITY = std::getenv("LLAMA_KV_PAGED_GATHER_NONIDENTITY");
    const char * LLAMA_KV_PAGED_IDENTITY_FAST_PATH = std::getenv("LLAMA_KV_PAGED_IDENTITY_FAST_PATH");
    const char * LLAMA_KV_PAGED_RELEASE = std::getenv("LLAMA_KV_PAGED_RELEASE");
    const char * LLAMA_KV_PAGED_TIMING = std::getenv("LLAMA_KV_PAGED_TIMING");
    const char * LLAMA_KV_PAGED_RESUME_TIMING = std::getenv("LLAMA_KV_PAGED_RESUME_TIMING");
    const char * LLAMA_KV_PAGED_RESUME_TIMING_STEP = std::getenv("LLAMA_KV_PAGED_RESUME_TIMING_STEP");
    paged_base_timing_enabled =
        LLAMA_KV_PAGED_TIMING && std::strcmp(LLAMA_KV_PAGED_TIMING, "1") == 0;
    if (paged_base_timing_enabled) {
        paged_base_timing_getenv_calls += 1;
    }
    paged_resume_timing_enabled =
        LLAMA_KV_PAGED_RESUME_TIMING && std::strcmp(LLAMA_KV_PAGED_RESUME_TIMING, "1") == 0;
    paged_resume_timing_step_enabled =
        LLAMA_KV_PAGED_RESUME_TIMING_STEP && std::strcmp(LLAMA_KV_PAGED_RESUME_TIMING_STEP, "1") == 0;
    const bool kv_paged_requested = LLAMA_KV_PAGED && std::strcmp(LLAMA_KV_PAGED, "1") == 0;
    const bool paged_identity_fast_path_requested =
        LLAMA_KV_PAGED_IDENTITY_FAST_PATH && std::strcmp(LLAMA_KV_PAGED_IDENTITY_FAST_PATH, "1") == 0;
    const bool paged_dynamic_remap_requested =
        LLAMA_KV_PAGED_GATHER_NONIDENTITY && std::strcmp(LLAMA_KV_PAGED_GATHER_NONIDENTITY, "1") == 0;
    paged_nonidentity_probe_requested = paged_dynamic_remap_requested;
    paged_ingraph_enabled =
        !(LLAMA_KV_PAGED_INGRAPH && std::strcmp(LLAMA_KV_PAGED_INGRAPH, "0") == 0);
    bool paged_dynamic_swap_requested = false;
    const bool paged_dynamic_release_requested =
        LLAMA_KV_PAGED_RELEASE && std::strcmp(LLAMA_KV_PAGED_RELEASE, "1") == 0;
    paged_block_release_requested = paged_dynamic_release_requested;
    bool paged_dynamic_madvise_requested = false;
    if (kv_paged_requested) {
        const char * LLAMA_KV_PAGED_BLOCK_SIZE = std::getenv("LLAMA_KV_PAGED_BLOCK_SIZE");
        const char * LLAMA_KV_PAGED_SHIFT      = std::getenv("LLAMA_KV_PAGED_SHIFT");
        const char * LLAMA_KV_PAGED_SWAP       = std::getenv("LLAMA_KV_PAGED_SWAP");
        const char * LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY = std::getenv("LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY");
        const char * LLAMA_KV_PAGED_TRACE      = std::getenv("LLAMA_KV_PAGED_TRACE");
        const char * LLAMA_KV_PAGED_IDLE_TRACE = std::getenv("LLAMA_KV_PAGED_IDLE_TRACE");
        const char * LLAMA_KV_PAGED_IDLE_SWAP  = std::getenv("LLAMA_KV_PAGED_IDLE_SWAP");
        const char * LLAMA_KV_PAGED_IDLE_SWAP_MADVISE = std::getenv("LLAMA_KV_PAGED_IDLE_SWAP_MADVISE");
        const char * LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS = std::getenv("LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS");
        const char * LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP = std::getenv("LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP");
        const char * LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS = std::getenv("LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS");
        const char * LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES = std::getenv("LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES");
        const char * LLAMA_KV_PAGED_SHADOW_VALIDATE = std::getenv("LLAMA_KV_PAGED_SHADOW_VALIDATE");
        const char * LLAMA_KV_PAGED_MINCORE = std::getenv("LLAMA_KV_PAGED_MINCORE");
        const char * LLAMA_KV_PAGED_REFAULT_TRACE           = std::getenv("LLAMA_KV_PAGED_REFAULT_TRACE");
        const char * LLAMA_KV_PAGED_REFAULT_TRACE_MAX       = std::getenv("LLAMA_KV_PAGED_REFAULT_TRACE_MAX");
        const char * LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE = std::getenv("LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE");
        const char * LLAMA_KV_PAGED_REFAULT_TRACE_ONCE      = std::getenv("LLAMA_KV_PAGED_REFAULT_TRACE_ONCE");
        const char * LLAMA_KV_PAGED_IO_STATS                = std::getenv("LLAMA_KV_PAGED_IO_STATS");
#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
        const char * LLAMA_KV_PAGED_RESTORE_FAULT_STATS     = std::getenv("LLAMA_KV_PAGED_RESTORE_FAULT_STATS");
#endif
#if defined(LLAMA_KV_RESTORE_PREFAULT_SUPPORTED)
        const char * LLAMA_KV_PAGED_RESTORE_PREFAULT_PROBE  = std::getenv("LLAMA_KV_PAGED_RESTORE_PREFAULT_PROBE");
#endif
        const char * LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE    = std::getenv("LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE");
        const char * LLAMA_KV_PAGED_RESTORE_K2              = std::getenv("LLAMA_KV_PAGED_RESTORE_K2");
        const char * LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE =
            std::getenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE");
        const char * LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ =
            std::getenv("LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ");
        const char * LLAMA_KV_TEST_MODE = std::getenv("LLAMA_KV_TEST_MODE");
        const char * LLAMA_KV_PAGED_RELEASE_TEST_FRESH_VERIFY =
            std::getenv("LLAMA_KV_PAGED_RELEASE_TEST_FRESH_VERIFY");
        const char * LLAMA_KV_PAGED_RELEASE_TEST_REPEAT =
            std::getenv("LLAMA_KV_PAGED_RELEASE_TEST_REPEAT");
        const char * LLAMA_KV_PAGED_RELEASE_TEST_REUSE =
            std::getenv("LLAMA_KV_PAGED_RELEASE_TEST_REUSE");
        const int block_size_env = LLAMA_KV_PAGED_BLOCK_SIZE ? std::atoi(LLAMA_KV_PAGED_BLOCK_SIZE) : 16;
        const int shift_env      = LLAMA_KV_PAGED_SHIFT      ? std::atoi(LLAMA_KV_PAGED_SHIFT)      : 0;
        const bool paged_swap_env = LLAMA_KV_PAGED_SWAP && std::strcmp(LLAMA_KV_PAGED_SWAP, "1") == 0;
        const bool paged_swap_explicit_only_env = LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY &&
            std::strcmp(LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY, "1") == 0;
        const bool idle_swap_env  = LLAMA_KV_PAGED_IDLE_SWAP && std::strcmp(LLAMA_KV_PAGED_IDLE_SWAP, "1") == 0;
        const bool idle_swap_madvise_env = LLAMA_KV_PAGED_IDLE_SWAP_MADVISE && std::strcmp(LLAMA_KV_PAGED_IDLE_SWAP_MADVISE, "1") == 0;
        paged_dynamic_swap_requested = paged_swap_env || idle_swap_env;
        paged_dynamic_madvise_requested = idle_swap_madvise_env;
        const bool paged_test_mode = LLAMA_KV_TEST_MODE && std::strcmp(LLAMA_KV_TEST_MODE, "1") == 0;
        paged_test_force_active_release = paged_test_mode &&
            LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE &&
            std::strcmp(LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE, "1") == 0;
        paged_release_fresh_verify_enabled = paged_test_mode &&
            LLAMA_KV_PAGED_RELEASE_TEST_FRESH_VERIFY &&
            std::strcmp(LLAMA_KV_PAGED_RELEASE_TEST_FRESH_VERIFY, "1") == 0;
        paged_release_test_repeat = paged_test_mode && LLAMA_KV_PAGED_RELEASE_TEST_REPEAT &&
            std::strcmp(LLAMA_KV_PAGED_RELEASE_TEST_REPEAT, "1") == 0;
        paged_release_test_reuse = paged_test_mode && LLAMA_KV_PAGED_RELEASE_TEST_REUSE &&
            std::strcmp(LLAMA_KV_PAGED_RELEASE_TEST_REUSE, "1") == 0;
        paged_test_force_active_release_seq = LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ ?
            (llama_seq_id) std::atoi(LLAMA_KV_PAGED_TEST_FORCE_ACTIVE_RELEASE_SEQ) : -1;
        const long idle_swap_every_tokens_env =
            LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS ? std::atol(LLAMA_KV_PAGED_IDLE_SWAP_EVERY_TOKENS) : 1;
        const long idle_swap_max_blocks_env =
            LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP ? std::atol(LLAMA_KV_PAGED_IDLE_SWAP_MAX_BLOCKS_PER_STEP) : 0;
        const long idle_swap_min_idle_steps_env =
            LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS ? std::atol(LLAMA_KV_PAGED_IDLE_SWAP_MIN_IDLE_STEPS) : 0;

        if (block_size_env <= 0 ||
                !ggml_is_power_of_2(block_size_env) ||
                (uint32_t) block_size_env > kv_size) {
            LLAMA_LOG_WARN("%s: KV paged metadata requested but LLAMA_KV_PAGED_BLOCK_SIZE=%d is invalid "
                    "(expected power of 2 in [1, %u]) - disabled\n",
                    __func__, block_size_env, kv_size);
        } else if (n_stream != 1 || v_trans) {
            if (!kv_paged_warned) {
                LLAMA_LOG_WARN("%s: KV paged metadata requires n_stream==1 && !v_trans "
                        "(n_stream=%u, v_trans=%d) - disabled\n",
                        __func__, n_stream, (int) v_trans);
                kv_paged_warned = true;
            }
        } else {
            kv_paged_enabled = true;
            paged_block_size = (uint32_t) block_size_env;
            if (shift_env > 0 && (type_k != GGML_TYPE_F32 || type_v != GGML_TYPE_F32)) {
                LLAMA_LOG_WARN("%s: KV paged non-identity mapping requires F32 K/V cache "
                        "(type_k=%s, type_v=%s) - using identity mapping\n",
                        __func__, ggml_type_name(type_k), ggml_type_name(type_v));
                paged_shift = 0;
            } else {
                paged_shift = shift_env > 0 ? (uint32_t) shift_env : 0;
            }
            paged_init(kv_size);
            paged_swap_enabled = paged_swap_env;
            paged_swap_explicit_only = paged_swap_env && paged_swap_explicit_only_env && !idle_swap_env;
            paged_idle_swap_requested = idle_swap_env;
            paged_idle_swap_madvise_requested = idle_swap_madvise_env;
            paged_idle_swap_every_tokens = idle_swap_every_tokens_env > 0 ? (uint64_t) idle_swap_every_tokens_env : 1;
            paged_idle_swap_max_blocks_per_step = idle_swap_max_blocks_env > 0 ? (uint64_t) idle_swap_max_blocks_env : 0;
            paged_idle_swap_min_idle_steps = idle_swap_min_idle_steps_env > 0 ? (uint64_t) idle_swap_min_idle_steps_env : 0;
            paged_idle_swap_debug_probes =
                !(LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES &&
                  std::strcmp(LLAMA_KV_PAGED_IDLE_SWAP_DEBUG_PROBES, "0") == 0);
            paged_shadow_validate_enabled =
                LLAMA_KV_PAGED_SHADOW_VALIDATE &&
                std::strcmp(LLAMA_KV_PAGED_SHADOW_VALIDATE, "1") == 0;
            paged_io_stats_enabled =
                LLAMA_KV_PAGED_IO_STATS &&
                std::strcmp(LLAMA_KV_PAGED_IO_STATS, "1") == 0;
#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
            paged_restore_fault_stats_enabled =
                LLAMA_KV_PAGED_RESTORE_FAULT_STATS &&
                std::strcmp(LLAMA_KV_PAGED_RESTORE_FAULT_STATS, "1") == 0;
#endif
#if defined(LLAMA_KV_RESTORE_PREFAULT_SUPPORTED)
            paged_restore_prefault_probe_enabled =
                LLAMA_KV_PAGED_RESTORE_PREFAULT_PROBE &&
                std::strcmp(LLAMA_KV_PAGED_RESTORE_PREFAULT_PROBE, "1") == 0;
#endif
            paged_prefetch_phase_trace_enabled =
                LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE &&
                std::strcmp(LLAMA_KV_PAGED_PREFETCH_PHASE_TRACE, "1") == 0;
            paged_restore_k2_enabled =
                LLAMA_KV_PAGED_RESTORE_K2 &&
                std::strcmp(LLAMA_KV_PAGED_RESTORE_K2, "1") == 0;
            if (paged_prefetch_phase_trace_enabled) {
                paged_io_stats_enabled = true;
            }
            paged_mincore_requested = LLAMA_KV_PAGED_MINCORE && std::strcmp(LLAMA_KV_PAGED_MINCORE, "1") == 0;
#if defined(__linux__)
            // kv_paged_enabled already implies n_stream==1 && !v_trans (checked above). CPU host
            // pointers are mincore-able; non-CPU backends never reach this branch in this driver.
            paged_mincore_enabled = paged_mincore_requested;
#else
            paged_mincore_enabled = false;
#endif
            if (paged_mincore_requested && !paged_mincore_enabled && !paged_mincore_warned) {
                LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED_MINCORE=1 requires Linux + CPU KV path "
                        "(kv_paged_enabled, n_stream==1, !v_trans); KV mincore telemetry disabled\n",
                        __func__);
                paged_mincore_warned = true;
            }
            paged_trace_enabled = LLAMA_KV_PAGED_TRACE && std::strcmp(LLAMA_KV_PAGED_TRACE, "1") == 0;
            paged_idle_trace_enabled = LLAMA_KV_PAGED_IDLE_TRACE && std::strcmp(LLAMA_KV_PAGED_IDLE_TRACE, "1") == 0;
            paged_refault_trace_requested =
                LLAMA_KV_PAGED_REFAULT_TRACE && std::strcmp(LLAMA_KV_PAGED_REFAULT_TRACE, "1") == 0;
            if (paged_refault_trace_requested) {
                paged_refault_trace_backtrace =
                    LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE &&
                    std::strcmp(LLAMA_KV_PAGED_REFAULT_TRACE_BACKTRACE, "1") == 0;
                // ONCE defaults to on; only "0" turns it off.
                paged_refault_trace_once =
                    !(LLAMA_KV_PAGED_REFAULT_TRACE_ONCE &&
                      std::strcmp(LLAMA_KV_PAGED_REFAULT_TRACE_ONCE, "0") == 0);
                if (LLAMA_KV_PAGED_REFAULT_TRACE_MAX) {
                    const long m = std::atol(LLAMA_KV_PAGED_REFAULT_TRACE_MAX);
                    if (m > 0) {
                        paged_refault_trace_max = (uint64_t) m;
                    }
                }
            }
            // paged_swap_enabled here is still the tentative request flag; final resolution
            // (mutual exclusion with exact swap, then backing-store construction) happens after
            // the KV layer tensors are built - see the deferred construction block below.
            // Destructive release is resolved only after the KV tensors exist and the final
            // row-index topology is known.  Until then the request must not enable execution.
            paged_block_release_enabled = false;
            LLAMA_LOG_INFO("%s: KV paged metadata enabled (block_size=%u, n_blocks=%u, shift=%u, "
                    "non_identity=%d, mapping_changed=%llu)\n",
                    __func__, paged_block_size, paged_n_blocks, paged_shift,
                    paged_non_identity_enabled ? 1 : 0,
                    (unsigned long long) paged_block_mapping_changed);
            if (paged_trace_enabled) {
                LLAMA_LOG_INFO("%s: KV paged block access trace enabled (telemetry only, stderr)\n", __func__);
            }
            if (paged_idle_trace_enabled) {
                LLAMA_LOG_INFO("%s: KV paged idle trace enabled (telemetry only)\n", __func__);
            }
            if (paged_resume_timing_enabled) {
                LLAMA_LOG_INFO("%s: KV paged resume timing enabled (telemetry only)\n", __func__);
            }
            if (paged_base_timing_enabled) {
                LLAMA_LOG_INFO("%s: KV paged base timing enabled (telemetry only)\n", __func__);
            }
            if (paged_io_stats_enabled) {
                LLAMA_LOG_INFO("%s: KV paged block I/O stats enabled (telemetry only)\n", __func__);
            }
#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
            if (paged_restore_fault_stats_enabled) {
                LLAMA_LOG_INFO("%s: KV paged restore scatter page-fault stats enabled (owner thread only)\n",
                        __func__);
            }
#endif
#if defined(LLAMA_KV_RESTORE_PREFAULT_SUPPORTED)
            if (paged_restore_prefault_probe_enabled) {
                LLAMA_LOG_INFO("%s: KV paged restore prefault probe enabled (owner thread only)\n", __func__);
            }
#endif
            if (paged_prefetch_phase_trace_enabled) {
                LLAMA_LOG_INFO("%s: KV paged prefetch phase trace enabled (diagnostic only)\n", __func__);
            }
            if (paged_restore_k2_enabled) {
                LLAMA_LOG_INFO("%s: KV paged restore K2 bounded two-stage pipeline enabled\n", __func__);
            }
            if (paged_swap_explicit_only_env && idle_swap_env) {
                LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED_SWAP_EXPLICIT_ONLY=1 conflicts with "
                        "LLAMA_KV_PAGED_IDLE_SWAP=1; explicit-only mode disabled\n", __func__);
            } else if (paged_swap_explicit_only) {
                LLAMA_LOG_INFO("%s: KV paged swap explicit-only mode enabled\n", __func__);
            }
            if (paged_idle_swap_requested) {
                LLAMA_LOG_INFO("%s: KV paged idle swap requested (safe-candidate probe only)\n", __func__);
            }
            // Stage 7D-A: install the refault SIGSEGV handler + arm bookkeeping. The trap table
            // itself is built lazily on the first protect call, once KV tensor data is allocated.
            paged_refault_init();
        }
    }

    // KV-P0-B1: exact swap and paged block swap cannot share one fixed-slot backing store in
    // this stage (exact addresses by physical cell id, paged addresses the same physical-cell
    // id space one block at a time - reusing the same n_slots=kv_size layout would work
    // arithmetically, but the two paths have never been validated to swap-out/in the same cell
    // concurrently without racing each other's SWAPPED/RESIDENT bookkeeping). If both are
    // requested, keep paged swap and disable exact - never pick silently.
    if (kv_swap_want_exact && paged_swap_enabled) {
        LLAMA_LOG_WARN("%s: KV exact swap (LLAMA_KV_SWAP_MODE=exact) and KV paged block swap "
                "(LLAMA_KV_PAGED_SWAP=1) were both requested; a single fixed-slot backing store "
                "cannot safely serve both in this stage - keeping paged block swap, disabling "
                "exact swap\n", __func__);
        kv_swap_want_exact = false;
    }

    const uint32_t n_layer_kv = hparams.n_layer_kv();

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*(1 + n_stream)*n_layer_kv*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < hparams.n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (n_embd_head_v_all == 0) {
            n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
        } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
            n_embd_head_v_all = -1;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_stream, v_stream, });
    }

    // KV-P0-B1: deferred fixed-slot backing-store construction. cell_stride is the full K/V
    // byte width of one physical cell across all layers - the same formula swap_out_cell() /
    // paged_swap_out_block_impl() already use for their staging buffer, computed here from the
    // just-built layer tensors (nb[1] is set at tensor-creation time, independent of whether the
    // backing buffer has been allocated yet). n_slots = kv_size, i.e. one slot per physical
    // cell; paged block swap-out still writes/reads one cell at a time into this same fixed
    // layout (no block aggregation in this stage).
    if (kv_swap_want_exact || paged_swap_enabled) {
        size_t cell_stride = 0;
        bool cell_stride_overflow = false;
        for (const auto & layer : layers) {
            if (!layer.k_stream.empty() && layer.k_stream[0]) {
                const size_t row_size = layer.k_stream[0]->nb[1];
                if (row_size > std::numeric_limits<size_t>::max() - cell_stride) {
                    cell_stride_overflow = true;
                    break;
                }
                cell_stride += row_size;
            }
            if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
                const size_t row_size = layer.v_stream[0]->nb[1];
                if (row_size > std::numeric_limits<size_t>::max() - cell_stride) {
                    cell_stride_overflow = true;
                    break;
                }
                cell_stride += row_size;
            }
        }

        if (cell_stride_overflow) {
            LLAMA_LOG_ERROR("%s: KV swap disabled: per-cell K/V byte stride overflow (errno=%d)\n",
                    __func__, EOVERFLOW);
            if (kv_swap_want_exact) {
                kv_swap_backend_failures += 1;
            }
            if (paged_swap_enabled) {
                paged_swap_backend_failures += 1;
                paged_swap_enabled = false;
            }
            kv_swap_want_exact = false;
        } else if (cell_stride == 0) {
            LLAMA_LOG_WARN("%s: KV swap disabled: could not determine a non-zero per-cell K/V byte stride\n",
                    __func__);
            if (kv_swap_want_exact) {
                kv_swap_backend_failures += 1;
            }
            if (paged_swap_enabled) {
                paged_swap_backend_failures += 1;
                paged_swap_enabled = false;
            }
            kv_swap_want_exact = false;
        } else {
            auto store = std::make_unique<llama_kv_backing_store_file>(kv_swap_dir, kv_size, cell_stride);
            if (!store->is_enabled()) {
                const auto & st = store->get_stats();
                LLAMA_LOG_WARN("%s: KV swap backing store unavailable (dir=%s errno=%d) - swap disabled\n",
                        __func__, (kv_swap_dir.empty() ? "/tmp" : kv_swap_dir.c_str()), st.last_errno);
                if (kv_swap_want_exact) {
                    kv_swap_backend_failures += 1;
                }
                if (paged_swap_enabled) {
                    paged_swap_backend_failures += 1;
                    paged_swap_enabled = false;
                }
                kv_swap_want_exact = false;
            } else {
                LLAMA_LOG_INFO("%s: KV swap backing store ready (dir=%s o_tmpfile=%d n_slots=%u "
                        "cell_stride=%zu capacity=%.2f MiB)\n",
                        __func__, store->get_dir().c_str(), store->used_o_tmpfile() ? 1 : 0,
                        store->get_n_slots(), store->get_cell_stride(),
                        (double) store->get_capacity() / 1024.0 / 1024.0);

                kv_swap_cell_stride = cell_stride;
                kv_swap_store = std::move(store);

                if (kv_swap_want_exact) {
                    kv_swap_enabled = true;
                    kv_swap_mode_   = kv_swap_mode::exact;
                    kv_swap_madvise = kv_swap_madvise_requested;
                    LLAMA_LOG_INFO("%s: KV swap exact mode enabled (backend=file, window=%u, sink=%u)\n",
                            __func__, kv_swap_window, kv_swap_sink);
                }
                if (paged_swap_enabled) {
                    LLAMA_LOG_INFO("%s: KV paged block swap enabled (backend=file, release=disabled)\n", __func__);
                }
            }
        }
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < hparams.n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // stage P2: clear-frontier (debug-only, off by default). Read before the buffer clear so we
    // can replace the full memset with a prefix-only clear. Only the !v_trans / single-stream
    // layout is supported; otherwise warn once and fall back to the full clear.
    const char * LLAMA_KV_LAZY_CLEAR = getenv("LLAMA_KV_LAZY_CLEAR");
    kv_lazy_clear = LLAMA_KV_LAZY_CLEAR ? (atoi(LLAMA_KV_LAZY_CLEAR) != 0) : false;
    if (kv_lazy_clear && (v_trans || n_stream != 1)) {
        LLAMA_LOG_WARN("%s: KV lazy-clear requires !v_trans && n_stream==1 (v_trans=%d, n_stream=%u) "
                "- falling back to full clear\n", __func__, (int) v_trans, n_stream);
        kv_lazy_clear = false;
    }
    if (kv_lazy_clear) {
        clear_frontier = std::min<uint32_t>(kv_size, 256u);
        LLAMA_LOG_INFO("%s: KV lazy-clear enabled (clear_frontier = %u cells, kv_size = %u) -- "
                "tail left uncommitted to lower peak RSS\n", __func__, clear_frontier, kv_size);
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (model.hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        if (kv_lazy_clear && !model.hparams.no_alloc) {
            // P2: only zero the [0, clear_frontier) prefix of each K/V tensor; leave the tail
            // uncommitted. n_stream==1 so each tensor row stride is t->nb[1].
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                const size_t total  = ggml_nbytes(t);
                const size_t prefix = std::min<size_t>(total, (size_t) clear_frontier * t->nb[1]);
                if (prefix > 0) {
                    ggml_backend_tensor_memset(t, 0, /*offset=*/0, /*size=*/prefix);
                }
                lazy_clear_init_bytes    += prefix;
                lazy_clear_skipped_bytes += (total - prefix);
            }
        } else {
            ggml_backend_buffer_clear(buf, 0);
        }
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
    const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
    if (attn_rot_disable) {
        LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
    }

    attn_rot_k =
        !attn_rot_disable &&
        n_embd_head_k_all > 0 &&
        ggml_is_quantized(type_k) &&
        hparams.n_embd_head_k() % 64 == 0;

    attn_rot_v =
        !attn_rot_disable &&
        n_embd_head_v_all > 0 &&
        ggml_is_quantized(type_v) &&
        hparams.n_embd_head_v() % 64 == 0;

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;

    // stage F1 / P1: KV Lazy-Block tail madvise (debug-only, off by default).
    // See docs/kv_lazy_block_stage_f1_design.md.
    const char * LLAMA_KV_LAZY_TAIL = getenv("LLAMA_KV_LAZY_TAIL");
    kv_lazy_tail = LLAMA_KV_LAZY_TAIL ? (atoi(LLAMA_KV_LAZY_TAIL) != 0) : false;
    if (kv_lazy_tail && paged_non_identity_enabled) {
        LLAMA_LOG_WARN("%s: KV lazy-tail disabled because paged non-identity mapping breaks "
                "physical-prefix assumption\n", __func__);
        kv_lazy_tail = false;
    }
    if (kv_lazy_tail) {
        LLAMA_LOG_INFO("%s: KV lazy-tail madvise enabled (v_trans = %d, n_stream = %u) -- advises unused tail "
                "[PAD(n_kv,256), %u) per step\n", __func__, (int) v_trans, n_stream, kv_size);
    }

    bool identity_mapping = kv_paged_enabled && !paged_non_identity_enabled && paged_block_mapping_changed == 0;
    if (identity_mapping) {
        for (uint32_t logical = 0; logical < paged_block_table.size(); ++logical) {
            if (paged_block_table[logical] != logical) {
                identity_mapping = false;
                break;
            }
        }
    }

    paged_layers_supported = !layers.empty();
    for (const auto & layer : layers) {
        paged_layers_supported = paged_layers_supported && layer.k && layer.v &&
            layer.k->type == GGML_TYPE_F32 && layer.v->type == GGML_TYPE_F32;
    }

    const auto fast_path = llama_kv_paged_identity_fast_path_resolve({
        /* .requested        = */ paged_identity_fast_path_requested,
        /* .paged_enabled    = */ kv_paged_enabled,
        /* .ingraph_enabled  = */ paged_ingraph_enabled,
        /* .single_stream    = */ n_stream == 1,
        /* .v_trans          = */ v_trans,
        /* .approx_dynamic   = */ uses_approx_dynamic_view(),
        /* .identity_mapping = */ identity_mapping,
        /* .dynamic_remap    = */ paged_dynamic_remap_requested,
        /* .swap             = */ kv_swap_requested || paged_dynamic_swap_requested,
        /* .release          = */ paged_dynamic_release_requested,
        /* .madvise          = */ kv_swap_madvise_requested || paged_dynamic_madvise_requested || kv_lazy_tail,
        /* .mapping_read_fault  = */ paged_test_mapping_fault_.scope == paged_test_mapping_fail_scope::READ,
        /* .mapping_write_fault = */ paged_test_mapping_fault_.scope == paged_test_mapping_fail_scope::WRITE,
        /* .swapin_fault        = */ paged_test_swapin_fault_.scope != paged_test_swapin_fail_scope::OFF,
        /* .backing_io_fault    = */ paged_test_io_fault_.scope != paged_test_io_fail_scope::OFF,
        /* .layers_supported = */ paged_layers_supported,
    });
    paged_identity_fast_path_enabled = fast_path.enabled;
    paged_identity_fast_path_reject = fast_path.reject;
    paged_identity_fast_path_layers = fast_path.enabled ? (uint32_t) map_layer_ids.size() : 0;
    paged_row_idx_enabled = kv_paged_enabled && paged_ingraph_enabled && paged_layers_supported && !fast_path.enabled;

    if (paged_dynamic_release_requested) {
        paged_block_release_enabled = llama_kv_destructive_release_can_enable(
                kv_paged_enabled,
                paged_ingraph_enabled,
                paged_layers_supported,
                paged_row_idx_enabled,
                paged_swap_enabled);
        if (paged_block_release_enabled) {
            LLAMA_LOG_INFO(
                    "KV_PAGED_RELEASE_CONFIG requested=1 enabled=1 reason=ROW_INDEX_GATHER_ENABLED "
                    "paged=%d ingraph=%d layers_supported=%d row_idx=%d\n",
                    kv_paged_enabled ? 1 : 0,
                    paged_ingraph_enabled ? 1 : 0,
                    paged_layers_supported ? 1 : 0,
                    paged_row_idx_enabled ? 1 : 0);
        } else {
            paged_test_force_active_release = false;
            paged_release_test_repeat = false;
            paged_release_test_reuse = false;
            LLAMA_LOG_WARN(
                    "KV_PAGED_RELEASE_CONFIG requested=1 enabled=0 reason=ROW_INDEX_GATHER_UNAVAILABLE "
                    "paged=%d ingraph=%d layers_supported=%d row_idx=%d swap=%d\n",
                    kv_paged_enabled ? 1 : 0,
                    paged_ingraph_enabled ? 1 : 0,
                    paged_layers_supported ? 1 : 0,
                    paged_row_idx_enabled ? 1 : 0,
                    paged_swap_enabled ? 1 : 0);
        }
    }

    if (kv_paged_enabled || paged_identity_fast_path_requested) {
        LLAMA_LOG_INFO(
                "%s: KV paged identity fast path: enabled=%d layers=%u reject_reason=%s\n",
                __func__,
                paged_identity_fast_path_enabled ? 1 : 0,
                paged_identity_fast_path_layers,
                llama_kv_paged_identity_fast_path_reject_name(paged_identity_fast_path_reject));
    }
}

llama_kv_cache::~llama_kv_cache() {
    paged_restore_worker_stop_and_join();

    // Stage 7D-A: tear down refault protection BEFORE anything else touches KV memory during
    // teardown -- restore every protected page to PROT_READ|PROT_WRITE and disarm the handler so
    // no late access traps. Drain the atomic fault counters into the instance for the stats line.
    paged_refault_drain();
    paged_refault_unprotect_all();

    kv_swap_roundtrip_selftest();

    static const llama_kv_backing_store_stats kv_swap_empty_stats;
    const auto & kv_swap_stats = kv_swap_store ? kv_swap_store->get_stats() : kv_swap_empty_stats;
    const char * kv_swap_mode_name =
        kv_swap_mode_ == kv_swap_mode::exact  ? "exact"  :
        kv_swap_mode_ == kv_swap_mode::approx ? "approx" : "off";
    LLAMA_LOG_INFO("%s: KV swap stats: enabled=%d mode=%s window=%u sink=%u "
            "swap_out_calls=%llu swap_in_calls=%llu ensure_calls=%llu window_calls=%llu "
            "window_skipped=%llu backend_failures=%llu bytes_written=%llu bytes_read=%llu "
            "write_calls=%llu read_calls=%llu release_calls=%llu "
            "RSS peak_kb=%llu current_last_kb=%llu current_min_kb=%llu current_max_kb=%llu "
            "rss_samples=%llu madvise_enabled=%d madvise_calls=%llu madvise_candidate_runs=%llu "
            "madvise_advised_runs=%llu madvise_advised_bytes=%llu madvise_failures=%llu "
            "madvise_skipped_bytes=%llu approx_calls=%llu approx_window=%llu approx_masked=%llu "
            "approx_debug_get_k_visible_gt0_calls=%llu approx_debug_get_v_visible_gt0_calls=%llu\n",
            __func__, kv_swap_enabled ? 1 : 0,
            kv_swap_mode_name,
            kv_swap_window, kv_swap_sink,
            (unsigned long long) kv_swap_out_calls,
            (unsigned long long) kv_swap_in_calls,
            (unsigned long long) kv_swap_ensure_calls,
            (unsigned long long) kv_swap_window_calls,
            (unsigned long long) kv_swap_window_skipped,
            (unsigned long long) kv_swap_backend_failures,
            (unsigned long long) kv_swap_stats.bytes_written,
            (unsigned long long) kv_swap_stats.bytes_read,
            (unsigned long long) kv_swap_stats.write_calls,
            (unsigned long long) kv_swap_stats.read_calls,
            (unsigned long long) kv_swap_stats.release_calls,
            (unsigned long long) get_peak_rss_kb(),
            (unsigned long long) kv_swap_rss_last_kb,
            (unsigned long long) kv_swap_rss_min_kb,
            (unsigned long long) kv_swap_rss_max_kb,
            (unsigned long long) kv_swap_rss_samples,
            kv_swap_madvise ? 1 : 0,
            (unsigned long long) kv_swap_madvise_calls,
            (unsigned long long) kv_swap_madvise_candidate_runs,
            (unsigned long long) kv_swap_madvise_advised_runs,
            (unsigned long long) kv_swap_madvise_advised_bytes,
            (unsigned long long) kv_swap_madvise_failures,
            (unsigned long long) kv_swap_madvise_skipped_bytes,
            (unsigned long long) kv_approx_calls,
            (unsigned long long) kv_approx_window,
            (unsigned long long) kv_approx_masked,
            (unsigned long long) kv_approx_debug_get_k_visible_gt0_calls,
            (unsigned long long) kv_approx_debug_get_v_visible_gt0_calls);

    if (kv_lazy_tail) {
        // stage F1 / P1: lazy-tail madvise counters (debug-only, current-RSS check).
        LLAMA_LOG_INFO("%s: kv lazy-tail stats: enabled=1 calls=%llu bytes=%llu (%.2f MiB) failures=%llu "
                "us=%llu (%.2f ms) rss_before=%llu KiB rss_after=%llu KiB\n", __func__,
                (unsigned long long) lazy_tail_madvise_calls,
                (unsigned long long) lazy_tail_madvise_bytes,
                lazy_tail_madvise_bytes / (1024.0 * 1024.0),
                (unsigned long long) lazy_tail_madvise_failures,
                (unsigned long long) lazy_tail_madvise_us,
                lazy_tail_madvise_us / 1000.0,
                (unsigned long long) lazy_tail_rss_before_kb,
                (unsigned long long) lazy_tail_rss_after_kb);
    }
    if (kv_lazy_clear) {
        // stage P2: clear-frontier counters (debug-only, peak-RSS path).
        LLAMA_LOG_INFO("%s: kv lazy-clear stats: enabled=1 clear_frontier=%u init_bytes=%llu (%.2f MiB) "
                "grow_bytes=%llu (%.2f MiB) skipped_bytes=%llu (%.2f MiB) calls=%llu us=%llu (%.2f ms)\n",
                __func__, clear_frontier,
                (unsigned long long) lazy_clear_init_bytes,
                lazy_clear_init_bytes / (1024.0 * 1024.0),
                (unsigned long long) lazy_clear_grow_bytes,
                lazy_clear_grow_bytes / (1024.0 * 1024.0),
                (unsigned long long) lazy_clear_skipped_bytes,
                lazy_clear_skipped_bytes / (1024.0 * 1024.0),
                (unsigned long long) lazy_clear_calls,
                (unsigned long long) lazy_clear_us,
                lazy_clear_us / 1000.0);
    }
    paged_log_base_timing();
    paged_log_timing();
    paged_log_stats();
}

void llama_kv_cache::clear(bool data) {
    paged_restore_worker_stop_and_join();
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }
    paged_reset();
    paged_resident_generation = paged_resident_generation == std::numeric_limits<uint64_t>::max()
        ? 1 : paged_resident_generation + 1;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

int32_t llama_kv_cache::prefetch_seq(llama_seq_id seq_id) {
    return this->prefetch_seq_step(seq_id, std::numeric_limits<uint32_t>::max());
}

size_t llama_kv_cache::paged_restore_group_byte_cap() const {
    return paged_restore_group_byte_cap_override != 0
        ? paged_restore_group_byte_cap_override
        : PAGED_RESTORE_GROUP_BYTE_CAP;
}

bool llama_kv_cache::paged_restore_group_identity_valid(
        const paged_restore_group_task & task,
        bool require_swapped) const {
    if (!kv_paged_enabled || !paged_swap_enabled || !kv_swap_store || v_trans || n_stream != 1 ||
            paged_block_size == 0 || v_cells.empty() ||
            task.object_id == 0 || task.object_id != paged_resident_object_id ||
            task.generation == 0 || task.generation != paged_resident_generation ||
            task.mapping_generation == 0 || task.mapping_generation != paged_mapping_generation ||
            task.blocks.empty() || task.blocks.size() != task.block_bytes.size() ||
            task.begin_block == UINT32_MAX || task.end_block <= task.begin_block ||
            task.end_block > paged_n_blocks || task.blocks.size() != task.end_block - task.begin_block ||
            task.begin_cell == UINT32_MAX || task.bytes_per_cell == 0 || task.total_bytes == 0) {
        return false;
    }

    size_t expected_total = 0;
    uint32_t expected_cells = 0;
    for (size_t i = 0; i < task.blocks.size(); ++i) {
        const uint32_t block = task.blocks[i];
        if (block != task.begin_block + i || block >= paged_block_states.size() ||
                (require_swapped && paged_block_states[block] != paged_block_state::SWAPPED)) {
            return false;
        }

        const uint32_t begin = block * paged_block_size;
        const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
        const uint32_t cells = end - begin;
        if (cells == 0 || cells > std::numeric_limits<size_t>::max() / task.bytes_per_cell) {
            return false;
        }
        const size_t bytes = (size_t) cells * task.bytes_per_cell;
        if (task.block_bytes[i] != bytes || bytes > std::numeric_limits<size_t>::max() - expected_total ||
                cells > std::numeric_limits<uint32_t>::max() - expected_cells) {
            return false;
        }
        expected_total += bytes;
        expected_cells += cells;
    }

    if (task.begin_cell != task.begin_block * paged_block_size ||
            task.cell_count != expected_cells || task.total_bytes != expected_total ||
            task.bytes_per_cell != kv_swap_cell_stride) {
        return false;
    }

    uint64_t expected_offset = 0;
    size_t expected_size = 0;
    return llama_kv_fixed_slot_range_bounds(
            task.begin_cell, task.cell_count, paged_kv_size,
            task.bytes_per_cell, kv_swap_store->get_stats().bytes_capacity, expected_offset, expected_size) &&
        task.backing_offset == expected_offset && task.total_bytes == expected_size;
}

bool llama_kv_cache::paged_restore_group_plan(
        llama_seq_id seq_id,
        const std::vector<uint32_t> & candidates,
        std::vector<paged_restore_group_task> & tasks) const {
    tasks.clear();
    if (candidates.empty()) {
        return true;
    }
    if (!kv_swap_store || paged_block_size == 0) {
        return false;
    }

    const size_t bytes_per_cell = kv_swap_cell_stride;
    const size_t byte_cap = paged_restore_group_byte_cap();
    if (bytes_per_cell == 0 || byte_cap == 0) {
        return false;
    }

    for (const uint32_t block : candidates) {
        if (block >= paged_n_blocks || block >= paged_block_states.size() ||
                paged_block_states[block] != paged_block_state::SWAPPED) {
            return false;
        }

        const uint32_t begin = block * paged_block_size;
        const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
        const uint32_t cell_count = end - begin;
        if (cell_count == 0 || cell_count > std::numeric_limits<size_t>::max() / bytes_per_cell) {
            return false;
        }
        const size_t block_bytes = (size_t) cell_count * bytes_per_cell;

        const bool start_group = tasks.empty() ||
            block != tasks.back().end_block ||
            tasks.back().total_bytes > byte_cap ||
            block_bytes > byte_cap - tasks.back().total_bytes;
        if (start_group) {
            paged_restore_group_task task;
            task.seq_id = seq_id;
            task.object_id = paged_resident_object_id;
            task.generation = paged_resident_generation;
            task.mapping_generation = paged_mapping_generation;
            task.begin_block = block;
            task.end_block = block;
            task.begin_cell = begin;
            task.bytes_per_cell = bytes_per_cell;
            tasks.push_back(std::move(task));
        }

        auto & task = tasks.back();
        if (task.total_bytes > std::numeric_limits<size_t>::max() - block_bytes ||
                task.cell_count > std::numeric_limits<uint32_t>::max() - cell_count) {
            return false;
        }
        task.blocks.push_back(block);
        task.block_bytes.push_back(block_bytes);
        task.end_block = block + 1;
        task.cell_count += cell_count;
        task.total_bytes += block_bytes;
    }

    for (auto & task : tasks) {
        uint64_t offset = 0;
        size_t total_size = 0;
        if (!llama_kv_fixed_slot_range_bounds(
                    task.begin_cell, task.cell_count, paged_kv_size,
                    task.bytes_per_cell, kv_swap_store->get_stats().bytes_capacity, offset, total_size) ||
                total_size != task.total_bytes) {
            return false;
        }
        task.backing_offset = offset;
    }
    return true;
}

bool llama_kv_cache::paged_restore_group_prepare(paged_restore_group_task & task) const {
    task.started_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    if (!paged_restore_group_identity_valid(task, true)) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        return false;
    }

    if (paged_restore_layout_cache.object_id != paged_resident_object_id ||
            paged_restore_layout_cache.rows.empty()) {
        paged_restore_layout layout;
        layout.object_id = paged_resident_object_id;
        for (const auto & layer : layers) {
            for (ggml_tensor * tensor : {
                    layer.k_stream.empty() ? nullptr : layer.k_stream[0],
                    (!layer.v || layer.v_stream.empty()) ? nullptr : layer.v_stream[0] }) {
                if (!tensor) {
                    continue;
                }
                const size_t row_size = tensor->nb[1];
                if (row_size == 0 || row_size > std::numeric_limits<size_t>::max() - layout.bytes_per_cell) {
                    paged_swap_in_fail_bad_size += 1;
                    paged_swap_backend_failures += 1;
                    task.read_status = llama_kv_backing_store_status::bad_slot;
                    return false;
                }
                layout.rows.push_back({ tensor, row_size });
                layout.bytes_per_cell += row_size;
            }
        }
        paged_restore_layout_cache = std::move(layout);
    }

    size_t layout_bytes = 0;
    for (const auto & row : paged_restore_layout_cache.rows) {
        if (!row.tensor || row.row_size == 0 || row.tensor->nb[1] != row.row_size ||
                row.row_size > std::numeric_limits<size_t>::max() - layout_bytes) {
            paged_swap_in_fail_bad_size += 1;
            paged_swap_backend_failures += 1;
            task.read_status = llama_kv_backing_store_status::bad_slot;
            return false;
        }
        layout_bytes += row.row_size;
    }
    if (paged_restore_layout_cache.object_id != task.object_id ||
            layout_bytes != task.bytes_per_cell ||
            paged_restore_layout_cache.bytes_per_cell != task.bytes_per_cell) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        return false;
    }

    const auto & cells = v_cells[0];
    for (uint32_t cell = task.begin_cell; cell < task.begin_cell + task.cell_count; ++cell) {
        if (cell >= cells.size() || cell >= paged_swap_offsets.size() || cell >= paged_swap_sizes.size() ||
                paged_swap_sizes[cell] != task.bytes_per_cell) {
            paged_swap_in_fail_bad_size += 1;
            paged_swap_backend_failures += 1;
            task.read_status = llama_kv_backing_store_status::bad_slot;
            task.failed_cell = cell;
            task.failed_block = cell / paged_block_size;
            return false;
        }
        const uint64_t expected = task.backing_offset +
            (uint64_t) (cell - task.begin_cell) * task.bytes_per_cell;
        if (paged_swap_offsets[cell] != expected) {
            paged_swap_in_fail_no_offset += 1;
            paged_swap_backend_failures += 1;
            task.read_status = llama_kv_backing_store_status::bad_slot;
            task.failed_cell = cell;
            task.failed_block = cell / paged_block_size;
            return false;
        }
    }

    task.layout = paged_restore_layout_cache.rows;
    if (task.staging.size() < task.total_bytes) {
        task.staging.resize(task.total_bytes);
    }
    if (paged_io_stats_enabled) {
        task.validate_us = llama_paged_timing_now_us() - task.started_us;
        paged_io_block_in_validate_us += task.validate_us;
        paged_io_block_in_validate_calls += 1;
        if (task.total_bytes > paged_io_staging_buffer_bytes) {
            paged_io_staging_buffer_bytes = task.total_bytes;
        }
    }
    task.prepared = true;
    return true;
}

void llama_kv_cache::paged_restore_group_read_cells(
        llama_kv_backing_store_i * store,
        paged_restore_group_task & task) {
    const uint64_t read_start_us = task.read_submit_us;
    if (!store || task.staging.size() < task.total_bytes) {
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.backend_errno = EINVAL;
    } else {
        try {
            task.read_status = store->read_cells(
                    0, task.begin_cell, task.cell_count, task.backing_offset,
                    task.staging.data(), task.total_bytes);
        } catch (...) {
            task.read_status = llama_kv_backing_store_status::io_error;
            task.backend_errno = EIO;
        }
    }
    task.read_completed = true;
    task.read_complete_us = llama_paged_timing_now_us();
    if (read_start_us != 0) {
        task.read_us = task.read_complete_us - read_start_us;
    }
}

bool llama_kv_cache::paged_restore_group_read_prepare(paged_restore_group_task & task) const {
    if (!task.prepared || task.read_completed || task.restore_completed || task.committed ||
            task.staging.size() < task.total_bytes || !paged_restore_group_identity_valid(task, true)) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        return false;
    }

    task.read_submitted = false;
    task.read_result_accounted = false;
    task.read_submit_us = 0;
    task.read_complete_us = 0;
    task.restore_start_us = 0;
    task.restore_complete_us = 0;
    task.exposed_read_wait_us = 0;
    task.pipeline_stall_us = 0;
    task.prefault_us = 0;
    task.prefault_calls = 0;
    task.prefault_minor_faults = 0;
    task.prefault_major_faults = 0;
    task.scatter_us = 0;
    task.io_fault_armed = false;
    task.io_fault_block = task.begin_block;
    task.io_fault_attempt_id = 0;
    task.io_fault_swap_in_before = paged_blocks_swapped_in;
    task.io_fault_syscalls_before = kv_swap_store->get_stats().syscall_attempts;
    task.io_fault_state_before = static_cast<uint8_t>(paged_block_states[task.begin_block]);

    const bool test_fault_attempt =
        paged_test_swapin_fault_.scope == paged_test_swapin_fail_scope::PREFETCH;
    if (test_fault_attempt) {
        paged_test_swapin_fault_.matching_attempts += 1;
        LLAMA_LOG_INFO(
                "TEST FAULT SWAPIN ATTEMPT attempt_id=%llu scope=prefetch physical_block=%u "
                "block_begin=%u block_end=%u fail_after_cells=%llu fail_once=%d consumed=%d "
                "block_state=%d paged_swap_in_calls_before=%llu\n",
                (unsigned long long) paged_test_swapin_fault_.matching_attempts,
                task.begin_block,
                task.begin_cell,
                task.begin_cell + task.cell_count,
                (unsigned long long) paged_test_swapin_fault_.fail_after_cells,
                paged_test_swapin_fault_.fail_once ? 1 : 0,
                paged_test_swapin_fault_.consumed ? 1 : 0,
                (int) paged_block_states[task.begin_block],
                (unsigned long long) paged_swap_in_calls);
    }
    if (test_fault_attempt &&
            paged_test_swapin_fault_.successful_cells >= paged_test_swapin_fault_.fail_after_cells &&
            (!paged_test_swapin_fault_.fail_once || !paged_test_swapin_fault_.consumed)) {
        if (paged_test_swapin_fault_.fail_once) {
            paged_test_swapin_fault_.consumed = true;
        }
        paged_test_swapin_fault_.trigger_count += 1;
        paged_test_swapin_fault_.prefetch_trigger_count += 1;
        task.read_status = llama_kv_backing_store_status::io_error;
        task.backend_errno = EIO;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        task.read_completed = true;
        LLAMA_LOG_ERROR(
                "TEST FAULT INJECTION attempt_id=%llu scope=prefetch physical_block=%u physical_cell=%u "
                "successful_cells_before_failure=%llu backend_status=io_error backend_errno=EIO(%d) "
                "failure_reason=%s block_state=%d metadata_present_cells=%u "
                "paged_swap_in_calls_before=%llu\n",
                (unsigned long long) paged_test_swapin_fault_.matching_attempts,
                task.begin_block,
                task.begin_cell,
                (unsigned long long) paged_test_swapin_fault_.successful_cells,
                EIO,
                llama_paged_swap_error_reason_name(llama_paged_swap_error_reason::SWAP_IN_IO_FAILURE),
                (int) paged_block_states[task.begin_block],
                task.cell_count,
                (unsigned long long) paged_swap_in_calls);
        return true;
    }

    const bool io_fault_matches =
        paged_test_io_fault_.scope == paged_test_io_fail_scope::PREFETCH_SWAP_IN &&
        paged_test_io_fault_.kind == paged_test_io_fail_kind::READ_EOF_ONCE &&
        paged_test_io_fault_.target_seq == task.seq_id &&
        std::find(task.blocks.begin(), task.blocks.end(), paged_test_io_fault_.target_block) != task.blocks.end() &&
        (!paged_test_io_fault_.fail_once || !paged_test_io_fault_.consumed);
    if (io_fault_matches) {
        const uint32_t fault_block = paged_test_io_fault_.target_block;
        const uint32_t begin = fault_block * paged_block_size;
        const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
        bool target_owned = false;
        for (uint32_t cell = begin; cell < end; ++cell) {
            if (cell < v_cells[0].size() && v_cells[0].seq_has(cell, task.seq_id)) {
                target_owned = true;
                break;
            }
        }
        if (target_owned) {
            if (auto * store_file = dynamic_cast<llama_kv_backing_store_file *>(kv_swap_store.get())) {
                llama_kv_backing_store_faults faults = store_file->get_test_faults();
                faults.read_eof_once = true;
                store_file->set_test_faults(faults);
                if (paged_test_io_fault_.fail_once) {
                    paged_test_io_fault_.consumed = true;
                }
                paged_test_io_fault_.matching_attempts += 1;
                paged_test_io_fault_.trigger_count += 1;
                task.io_fault_attempt_id = paged_test_io_fault_.matching_attempts;
                task.io_fault_block = fault_block;
                task.io_fault_swap_in_before = paged_blocks_swapped_in;
                task.io_fault_syscalls_before = kv_swap_store->get_stats().syscall_attempts;
                task.io_fault_state_before = static_cast<uint8_t>(paged_block_states[fault_block]);
                task.io_fault_armed = true;
            }
        }
    }

    return true;
}

bool llama_kv_cache::paged_restore_group_read_finish(paged_restore_group_task & task) const {
    if (task.read_result_accounted) {
        return task.read_status == llama_kv_backing_store_status::ok;
    }
    task.read_result_accounted = true;

    if (task.read_submitted && paged_io_stats_enabled) {
        paged_io_block_in_read_us += task.read_us;
        paged_io_block_in_read_calls += 1;
    }
    if (!task.read_completed) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        return false;
    }
    if (task.read_status == llama_kv_backing_store_status::ok) {
        return true;
    }

    paged_swap_in_fail_read_cell += 1;
    paged_swap_backend_failures += 1;
    if ((task.read_status == llama_kv_backing_store_status::io_error ||
            task.read_status == llama_kv_backing_store_status::disabled) && kv_swap_store) {
        task.backend_errno = kv_swap_store->get_stats().last_errno;
    }
    task.failed_block = task.io_fault_armed ? task.io_fault_block : task.begin_block;
    task.failed_cell = task.failed_block * paged_block_size;
    if (task.io_fault_armed && kv_swap_store) {
        const auto & stats = kv_swap_store->get_stats();
        auto block_state_name = [](paged_block_state state) {
            switch (state) {
                case paged_block_state::UNUSED:        return "UNUSED";
                case paged_block_state::RESIDENT:      return "RESIDENT";
                case paged_block_state::RELEASED:      return "RELEASED";
                case paged_block_state::SWAPPED:       return "SWAPPED";
                case paged_block_state::PENDING_WRITE: return "PENDING_WRITE";
                case paged_block_state::INVALID:       return "INVALID";
            }
            return "UNKNOWN";
        };
        LLAMA_LOG_ERROR(
                "KV_PAGED_IO_FAULT scope=prefetch_swap_in kind=read_eof_once target_seq=%d "
                "physical_block=%u physical_cell=%u state_before=%s state_after=%s "
                "swap_in_counter_before=%llu swap_in_counter_after=%llu "
                "backend_status=%s backend_errno=%d pread_attempts_before=%llu "
                "pread_attempts_after=%llu metadata_present_cells=%u "
                "failure_reason=%s trigger_count=%llu attempt_id=%llu\n",
                (int) task.seq_id,
                task.io_fault_block,
                task.failed_cell,
                block_state_name(static_cast<paged_block_state>(task.io_fault_state_before)),
                block_state_name(paged_block_states[task.io_fault_block]),
                (unsigned long long) task.io_fault_swap_in_before,
                (unsigned long long) paged_blocks_swapped_in,
                llama_kv_backing_store_status_name(stats.last_status),
                task.backend_errno,
                (unsigned long long) task.io_fault_syscalls_before,
                (unsigned long long) stats.syscall_attempts,
                task.cell_count,
                llama_paged_swap_error_reason_name(llama_paged_swap_error_reason::PAGED_SWAP_IN_IO_ERROR),
                (unsigned long long) paged_test_io_fault_.trigger_count,
                (unsigned long long) task.io_fault_attempt_id);
        paged_test_io_fault_.failed_block = task.io_fault_block;
        paged_test_io_fault_.failed_attempt_id = task.io_fault_attempt_id;
    }
    return false;
}

bool llama_kv_cache::paged_restore_group_read(paged_restore_group_task & task) const {
    if (!paged_restore_group_read_prepare(task)) {
        return false;
    }
    if (!task.read_completed) {
        task.read_submitted = true;
        task.read_submit_us = llama_paged_timing_now_us();
        paged_restore_group_read_cells(kv_swap_store.get(), task);
    }
    return paged_restore_group_read_finish(task);
}

bool llama_kv_cache::paged_restore_group_prefault(paged_restore_group_task & task) const {
#if defined(LLAMA_KV_RESTORE_PREFAULT_SUPPORTED)
    if (!paged_restore_prefault_probe_enabled) {
        return true;
    }

    uint64_t fault_minor_before = 0;
    uint64_t fault_major_before = 0;
    const bool fault_stats_sampled =
#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
        llama_paged_restore_read_thread_faults(fault_minor_before, fault_major_before);
#else
        false;
#endif

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return false;
    }

    const uint64_t prefault_start_us = llama_paged_timing_now_us();
    bool ok = true;
    uint32_t failed_block = UINT32_MAX;
    int failed_errno = 0;
    for (const uint32_t block : task.blocks) {
        for (const auto & row : task.layout) {
            if (!row.tensor || !row.tensor->data || row.row_size == 0) {
                ok = false;
                failed_block = block;
                failed_errno = EINVAL;
                break;
            }

            paged_block_page_range page_range;
            if (!paged_compute_block_page_range(
                    row.tensor, block, row.row_size, (uintptr_t) page, page_range)) {
                continue;
            }

            task.prefault_calls += 1;
            if (madvise(
                    (void *) page_range.page_begin,
                    (size_t) (page_range.page_end - page_range.page_begin),
                    MADV_POPULATE_WRITE) != 0) {
                ok = false;
                failed_block = block;
                failed_errno = errno;
                break;
            }
        }
        if (!ok) {
            break;
        }
    }
    task.prefault_us = llama_paged_timing_now_us() - prefault_start_us;

#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
    if (fault_stats_sampled) {
        uint64_t fault_minor_after = 0;
        uint64_t fault_major_after = 0;
        if (llama_paged_restore_read_thread_faults(fault_minor_after, fault_major_after) &&
                fault_minor_after >= fault_minor_before && fault_major_after >= fault_major_before) {
            task.prefault_minor_faults = fault_minor_after - fault_minor_before;
            task.prefault_major_faults = fault_major_after - fault_major_before;
        }
    }
#endif

    paged_restore_prefault_calls += task.prefault_calls;
    paged_restore_prefault_us += task.prefault_us;
    paged_restore_prefault_minor += task.prefault_minor_faults;
    paged_restore_prefault_major += task.prefault_major_faults;
    if (!ok) {
        LLAMA_LOG_ERROR(
                "%s: MADV_POPULATE_WRITE failed for restore group blocks=[%u,%u) "
                "block=%u errno=%d (%s)\n",
                __func__, task.begin_block, task.end_block, failed_block, failed_errno, std::strerror(failed_errno));
        return false;
    }

    paged_restore_prefault_groups += 1;
    return true;
#else
    (void) task;
    return true;
#endif
}

bool llama_kv_cache::paged_restore_group_post_read(paged_restore_group_task & task) const {
    if (!task.prepared || !task.read_completed || task.restore_completed || task.committed ||
            task.read_status != llama_kv_backing_store_status::ok) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        return false;
    }

    // This is the last owner-side identity check before any tensor write. The worker never
    // reaches this function and cannot unprotect refault pages or publish block state.
    if (!paged_restore_group_identity_valid(task, true)) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        return false;
    }

    task.restore_start_us = llama_paged_timing_now_us();
    for (const uint32_t block : task.blocks) {
        paged_refault_unprotect_block(block);
    }

    if (!paged_restore_group_prefault(task)) {
        for (const uint32_t block : task.blocks) {
            paged_refault_protect_block(block);
        }
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        return false;
    }

    paged_restore_test_scatter_groups += 1;
    const uint64_t unpack_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
    uint64_t fault_minor_before = 0;
    uint64_t fault_major_before = 0;
    const bool fault_stats_sampled = paged_restore_fault_stats_enabled &&
        llama_paged_restore_read_thread_faults(fault_minor_before, fault_major_before);
#endif
    const uint64_t scatter_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    for (uint32_t cell = task.begin_cell; cell < task.begin_cell + task.cell_count; ++cell) {
        size_t cursor = (size_t) (cell - task.begin_cell) * task.bytes_per_cell;
        for (const auto & row : task.layout) {
            ggml_backend_tensor_set(
                    row.tensor, task.staging.data() + cursor, (size_t) cell * row.row_size, row.row_size);
            cursor += row.row_size;
        }
        GGML_ASSERT(cursor == (size_t) (cell - task.begin_cell + 1) * task.bytes_per_cell);
    }
    if (paged_io_stats_enabled) {
        task.scatter_us = llama_paged_timing_now_us() - scatter_start_us;
        paged_restore_scatter_us += task.scatter_us;
    }
#if defined(LLAMA_KV_RESTORE_FAULT_STATS_SUPPORTED)
    if (fault_stats_sampled) {
        uint64_t fault_minor_after = 0;
        uint64_t fault_major_after = 0;
        if (llama_paged_restore_read_thread_faults(fault_minor_after, fault_major_after) &&
                fault_minor_after >= fault_minor_before && fault_major_after >= fault_major_before) {
            paged_restore_fault_stats_groups += 1;
            paged_restore_fault_stats_minor += fault_minor_after - fault_minor_before;
            paged_restore_fault_stats_major += fault_major_after - fault_major_before;
        }
    }
#endif
    if (paged_io_stats_enabled) {
        task.unpack_us = llama_paged_timing_now_us() - unpack_start_us;
        paged_io_block_in_unpack_us += task.unpack_us;
        paged_io_block_in_unpack_calls += 1;
    }
    task.restore_complete_us = llama_paged_timing_now_us();
    task.restore_completed = true;
    return true;
}

bool llama_kv_cache::paged_restore_group_execute(paged_restore_group_task & task) const {
    return paged_restore_group_read(task) && paged_restore_group_post_read(task);
}

bool llama_kv_cache::paged_restore_group_start_async(
        const std::shared_ptr<paged_restore_group_task> & task) const {
    if (!task || paged_restore_worker_thread.joinable() ||
            !paged_restore_group_read_prepare(*task)) {
        return false;
    }
    if (task->read_completed) {
        return true;
    }

    task->read_submitted = true;
    task->read_submit_us = llama_paged_timing_now_us();
    paged_restore_worker_task = task;
    paged_restore_worker_stop_requested.store(false, std::memory_order_release);
    llama_kv_backing_store_i * store = kv_swap_store.get();
    try {
        paged_restore_worker_thread = std::thread([store, task]() {
            llama_kv_cache::paged_restore_group_read_cells(store, *task);
        });
    } catch (...) {
        paged_restore_worker_task.reset();
        task->read_submitted = false;
        task->read_status = llama_kv_backing_store_status::io_error;
        task->backend_errno = EAGAIN;
        task->read_completed = true;
    }
    return true;
}

bool llama_kv_cache::paged_restore_group_wait_async(paged_restore_group_task & task) const {
    paged_restore_worker_stop_and_join();
    return paged_restore_group_read_finish(task);
}

void llama_kv_cache::paged_restore_worker_stop_and_join() const {
    paged_restore_worker_stop_requested.store(true, std::memory_order_release);
    if (paged_restore_worker_thread.joinable()) {
        paged_restore_worker_thread.join();
    }
    paged_restore_worker_task.reset();
}

void llama_kv_cache::paged_restore_test_invalidate_before_scatter(
        paged_restore_group_task & task) const {
    if (paged_restore_test_stale_before_complete) {
        paged_restore_test_stale_before_complete = false;
        paged_restore_test_stale_triggers += 1;
        task.generation = task.generation == std::numeric_limits<uint64_t>::max()
            ? 1
            : task.generation + 1;
    }
    if (paged_restore_test_mapping_stale_before_complete) {
        paged_restore_test_mapping_stale_before_complete = false;
        paged_restore_test_mapping_stale_triggers += 1;
        paged_mapping_generation = paged_mapping_generation == std::numeric_limits<uint64_t>::max()
            ? 1
            : paged_mapping_generation + 1;
    }
}

bool llama_kv_cache::paged_restore_group_complete(paged_restore_group_task & task) const {
    if (!task.prepared || !task.read_completed || !task.restore_completed || task.committed ||
            task.read_status != llama_kv_backing_store_status::ok ||
            !paged_restore_group_identity_valid(task, true)) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        task.read_status = llama_kv_backing_store_status::bad_slot;
        task.failed_block = task.begin_block;
        task.failed_cell = task.begin_cell;
        return false;
    }

    const uint64_t commit_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    for (const uint32_t block : task.blocks) {
        paged_block_states[block] = paged_block_state::RESIDENT;
    }
    paged_swap_in_calls += task.blocks.size();
    paged_blocks_swapped_in += task.blocks.size();
    paged_swap_bytes_in += task.total_bytes;
    paged_swap_in_last_block = task.blocks.back();
    if (paged_io_stats_enabled) {
        task.commit_us = llama_paged_timing_now_us() - commit_start_us;
        paged_io_block_in_commit_us += task.commit_us;
        paged_io_block_in_commit_calls += 1;
        const uint64_t elapsed_us = llama_paged_timing_now_us() - task.started_us;
        paged_io_swap_in_latency_us += elapsed_us;
        paged_io_swap_in_timed_calls += 1;
        if (elapsed_us > paged_io_swap_in_latency_max_us) {
            paged_io_swap_in_latency_max_us = elapsed_us;
        }
    }
    task.committed = true;

    if (paged_test_swapin_fault_.scope == paged_test_swapin_fail_scope::PREFETCH) {
        paged_test_swapin_fault_.successful_cells += task.cell_count;
        LLAMA_LOG_INFO(
                "TEST FAULT SWAPIN COMPLETE attempt_id=%llu scope=prefetch physical_block=%u "
                "restored_cells=%u block_state=%d paged_swap_in_calls_after=%llu\n",
                (unsigned long long) paged_test_swapin_fault_.matching_attempts,
                task.begin_block,
                task.cell_count,
                (int) paged_block_states[task.begin_block],
                (unsigned long long) paged_swap_in_calls);
    }
    return true;
}

bool llama_kv_cache::paged_restore_group_sync(paged_restore_group_task & task) const {
    const bool cache_staging_reused = paged_io_staging.size() >= task.total_bytes;
    struct staging_swap_guard {
        std::vector<uint8_t> & task_staging;
        std::vector<uint8_t> & cache_staging;

        staging_swap_guard(
                std::vector<uint8_t> & task_staging_,
                std::vector<uint8_t> & cache_staging_)
            : task_staging(task_staging_), cache_staging(cache_staging_) {
            task_staging.swap(cache_staging);
        }

        ~staging_swap_guard() {
            task_staging.swap(cache_staging);
        }
    } staging_guard(task.staging, paged_io_staging);

    if (cache_staging_reused) {
        paged_restore_k1_sync_staging_reuses += 1;
    }

    bool ok = paged_restore_group_prepare(task) && paged_restore_group_read(task);
    if (ok) {
        paged_restore_test_invalidate_before_scatter(task);
        ok = paged_restore_group_post_read(task);
    }
    if (ok) {
        ok = paged_restore_group_complete(task);
    }
    return ok;
}

llama_kv_cache::paged_prefetch_step_result llama_kv_cache::prefetch_seq_step_impl(
        llama_seq_id seq_id,
        uint32_t max_blocks,
        bool all_required) {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    paged_prefetch_step_result result;
    paged_prefetch_seq_last_owned_blocks = 0;
    paged_prefetch_seq_last_swapped_blocks = 0;
    paged_prefetch_seq_last_resident_blocks = 0;
    paged_prefetch_seq_last_released_blocks = 0;
    paged_prefetch_seq_last_invalid_cells = 0;
    paged_prefetch_seq_last_failures = 0;
    paged_restore_k1_sync_staging_reuses = 0;

    if (!kv_paged_enabled) {
        return result;
    }

    paged_prefetch_seq_calls += 1;

    if (!paged_swap_enabled) {
        return result;
    }
    if (!kv_swap_store || v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0 ||
            v_cells.empty() || paged_block_states.size() != paged_n_blocks) {
        paged_prefetch_seq_last_failures += 1;
        paged_prefetch_seq_failures += 1;
        result.failed = true;
        return result;
    }

    const auto & cells = v_cells[0];
    std::set<uint32_t> blocks;
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (!cells.seq_has(cell, seq_id)) {
            continue;
        }

        const uint32_t phys_cell = paged_resolve(cell);
        if (phys_cell == PAGED_BLOCK_INVALID) {
            paged_prefetch_seq_last_invalid_cells += 1;
            continue;
        }

        const uint32_t physical_block = phys_cell / paged_block_size;
        if (physical_block < paged_n_blocks) {
            blocks.insert(physical_block);
        }
    }
    paged_prefetch_seq_last_owned_blocks = blocks.size();

    const uint64_t bytes_per_cell = kv_swap_cell_stride;
    if (bytes_per_cell == 0) {
        paged_prefetch_seq_last_failures += 1;
        paged_prefetch_seq_failures += 1;
        result.failed = true;
        return result;
    }

    for (const uint32_t physical_block : blocks) {
        const paged_block_state state = paged_block_states[physical_block];
        if (state == paged_block_state::SWAPPED) {
            paged_prefetch_seq_last_swapped_blocks += 1;
        } else if (state == paged_block_state::RESIDENT) {
            paged_prefetch_seq_last_resident_blocks += 1;
            paged_prefetch_seq_skip_resident += 1;
        } else if (state == paged_block_state::RELEASED) {
            paged_prefetch_seq_last_released_blocks += 1;
            paged_prefetch_seq_skip_released += 1;
        }
    }

    if (max_blocks == 0 && !all_required) {
        return result;
    }

    std::vector<uint32_t> candidates;
    for (const uint32_t physical_block : blocks) {
        if ((!all_required && candidates.size() >= max_blocks) ||
                paged_block_states[physical_block] != paged_block_state::SWAPPED) {
            continue;
        }
        const uint32_t begin = physical_block * paged_block_size;
        const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
        candidates.push_back(physical_block);
        result.candidate_blocks += 1;
        result.candidate_bytes += bytes_per_cell * (end - begin);
    }

    std::vector<paged_restore_group_task> restore_tasks;
    if (!paged_restore_group_plan(seq_id, candidates, restore_tasks)) {
        paged_prefetch_seq_last_failures += 1;
        paged_prefetch_seq_failures += 1;
        result.failed = true;
        LLAMA_LOG_ERROR("%s: paged KV prefetch transfer-group planning failed\n", __func__);
        return result;
    }

    size_t max_single_block_bytes = 0;
    for (const auto & task : restore_tasks) {
        for (const size_t block_bytes : task.block_bytes) {
            max_single_block_bytes = std::max(max_single_block_bytes, block_bytes);
        }
    }

    const bool phase_trace = paged_prefetch_phase_trace_enabled;
    const uint64_t phase_call = phase_trace ? ++paged_prefetch_phase_trace_calls : 0;
    uint32_t phase_events = 0;
    const auto phase_share = [](uint64_t total, size_t count, size_t index) {
        return count == 0 ? uint64_t(0) : total / count + (index < total % count ? 1 : 0);
    };
    const auto discard_staging = [](paged_restore_group_task & task) {
        std::vector<uint8_t>().swap(task.staging);
    };
    const auto record_success = [&](paged_restore_group_task & task) {
        result.restored_blocks += task.blocks.size();
        result.restored_bytes += task.total_bytes;
        paged_prefetch_seq_blocks += task.blocks.size();
        paged_prefetch_seq_bytes += task.total_bytes;
        if (phase_trace) {
            for (size_t i = 0; i < task.blocks.size(); ++i) {
                const uint64_t validate_us = phase_share(task.validate_us, task.blocks.size(), i);
                const uint64_t read_us = phase_share(task.read_us, task.blocks.size(), i);
                const uint64_t unpack_us = phase_share(task.unpack_us, task.blocks.size(), i);
                const uint64_t commit_us = phase_share(task.commit_us, task.blocks.size(), i);
                fprintf(stderr,
                        "KV_PAGED_PREFETCH_BLOCK_PHASE call=%llu block_index=%u physical_block=%u "
                        "validate_us=%llu read_us=%llu unpack_us=%llu commit_us=%llu phase_sum_us=%llu\n",
                        (unsigned long long) phase_call,
                        phase_events,
                        task.blocks[i],
                        (unsigned long long) validate_us,
                        (unsigned long long) read_us,
                        (unsigned long long) unpack_us,
                        (unsigned long long) commit_us,
                        (unsigned long long) (validate_us + read_us + unpack_us + commit_us));
                phase_events += 1;
            }
        }
        if (paged_restore_k2_enabled && paged_io_stats_enabled) {
            fprintf(stderr,
                    "KV_PAGED_PREFETCH_PIPELINE call=%llu group_index=%u "
                    "read_submit_us=%llu read_complete_us=%llu read_service_us=%llu "
                    "restore_start_us=%llu restore_complete_us=%llu restore_service_us=%llu "
                    "exposed_read_wait_us=%llu pipeline_stall_us=%llu\n",
                    (unsigned long long) phase_call,
                    task.pipeline_index,
                    (unsigned long long) task.read_submit_us,
                    (unsigned long long) task.read_complete_us,
                    (unsigned long long) (task.read_complete_us >= task.read_submit_us
                        ? task.read_complete_us - task.read_submit_us : 0),
                    (unsigned long long) task.restore_start_us,
                    (unsigned long long) task.restore_complete_us,
                    (unsigned long long) (task.restore_complete_us >= task.restore_start_us
                        ? task.restore_complete_us - task.restore_start_us : 0),
                    (unsigned long long) task.exposed_read_wait_us,
                    (unsigned long long) task.pipeline_stall_us);
        }
    };
    const auto fail_group = [&](const paged_restore_group_task & task) {
        paged_prefetch_seq_last_failures += 1;
        paged_prefetch_seq_failures += 1;
        result.failed = true;
        LLAMA_LOG_ERROR(
                "%s: paged KV prefetch transfer-group failed for blocks=[%u,%u) bytes=%zu\n",
                __func__, task.begin_block, task.end_block, task.total_bytes);
    };

    paged_restore_k2_read_ahead_observed = 0;
    paged_restore_k2_peak_staging_groups = 0;
    paged_restore_k2_peak_staging_bytes = 0;
    paged_restore_k2_staging_bound_bytes = paged_restore_k2_enabled
        ? llama_paged_restore_staging_bound_bytes(
                paged_restore_group_byte_cap(), max_single_block_bytes)
        : 0;
    paged_restore_k2_staging_reuses = 0;
    paged_restore_k2_exposed_read_wait_us = 0;
    paged_restore_k2_pipeline_stall_us = 0;
    paged_restore_k2_pipeline_wall_us = 0;
    paged_restore_k2_read_completed_before_restore_complete = 0;
    if (!paged_restore_k2_enabled) {
        for (auto & task : restore_tasks) {
            if (!paged_restore_group_sync(task)) {
                fail_group(task);
                discard_staging(task);
                return result;
            }
            record_success(task);
            discard_staging(task);
        }
    } else if (!restore_tasks.empty()) {
        const uint64_t pipeline_start_us = llama_paged_timing_now_us();
        const auto finish_pipeline = [&]() {
            paged_restore_k2_pipeline_wall_us = llama_paged_timing_now_us() - pipeline_start_us;
        };
        const auto load_slot = [](const std::shared_ptr<paged_restore_group_task> & slot,
                                  paged_restore_group_task && planned,
                                  uint32_t pipeline_index) {
            const bool staging_reused = planned.total_bytes != 0 &&
                slot->staging.size() >= planned.total_bytes;
            std::vector<uint8_t> staging = std::move(slot->staging);
            *slot = std::move(planned);
            slot->staging = std::move(staging);
            slot->pipeline_index = pipeline_index;
            return staging_reused;
        };

        auto current = std::make_shared<paged_restore_group_task>();
        auto next = std::make_shared<paged_restore_group_task>();
        const bool current_staging_reused =
            load_slot(current, std::move(restore_tasks.front()), 0);
        paged_restore_k2_peak_staging_groups = 1;
        if (!paged_restore_group_prepare(*current)) {
            fail_group(*current);
            paged_restore_worker_stop_and_join();
            finish_pipeline();
            return result;
        }
        if (current_staging_reused) {
            paged_restore_k2_staging_reuses += 1;
        }
        paged_restore_k2_peak_staging_bytes = current->staging.size();
        if (!paged_restore_group_start_async(current) ||
                !paged_restore_group_wait_async(*current)) {
            fail_group(*current);
            paged_restore_worker_stop_and_join();
            finish_pipeline();
            return result;
        }

        for (size_t group_index = 0; group_index < restore_tasks.size(); ++group_index) {
            bool next_ready = true;
            bool next_staging_reused = false;
            if (group_index + 1 < restore_tasks.size()) {
                next_staging_reused = load_slot(
                        next, std::move(restore_tasks[group_index + 1]), (uint32_t) group_index + 1);
                if (!paged_restore_group_prepare(*next)) {
                    next_ready = false;
                } else {
                    if (next_staging_reused) {
                        paged_restore_k2_staging_reuses += 1;
                    }
                    paged_restore_k2_peak_staging_bytes = std::max<uint64_t>(
                            paged_restore_k2_peak_staging_bytes,
                            current->staging.size() + next->staging.size());
                    paged_restore_k2_peak_staging_groups = std::max<uint32_t>(
                            paged_restore_k2_peak_staging_groups, 2);
                    if (!paged_restore_group_start_async(next)) {
                        next_ready = false;
                    } else if (next->read_submitted) {
                        paged_restore_k2_read_ahead_observed += 1;
                    }
                }
            }

            paged_restore_test_invalidate_before_scatter(*current);
            if (!paged_restore_group_post_read(*current) ||
                    !paged_restore_group_complete(*current)) {
                fail_group(*current);
                paged_restore_worker_stop_and_join();
                finish_pipeline();
                return result;
            }
            record_success(*current);
            const uint64_t restore_complete_us = current->restore_complete_us;

            if (!next_ready) {
                paged_restore_worker_stop_and_join();
                fail_group(*next);
                finish_pipeline();
                return result;
            }
            if (group_index + 1 < restore_tasks.size()) {
                if (!paged_restore_group_wait_async(*next)) {
                    fail_group(*next);
                    finish_pipeline();
                    return result;
                }
                if (next->read_complete_us > restore_complete_us) {
                    next->exposed_read_wait_us = next->read_complete_us - restore_complete_us;
                    next->pipeline_stall_us = next->exposed_read_wait_us;
                    paged_restore_k2_exposed_read_wait_us += next->exposed_read_wait_us;
                    paged_restore_k2_pipeline_stall_us += next->pipeline_stall_us;
                } else if (next->read_complete_us != 0) {
                    paged_restore_k2_read_completed_before_restore_complete += 1;
                }
                std::swap(current, next);
            }
        }
        finish_pipeline();
    }

    if (phase_trace) {
        fprintf(stderr,
                "KV_PAGED_PREFETCH_PHASE_CALL call=%llu seq_id=%d requested_blocks=%u "
                "restored_blocks=%u phase_events=%u\n",
                (unsigned long long) phase_call,
                (int) seq_id,
                max_blocks,
                result.restored_blocks,
                phase_events);
    }

    return result;
}

int32_t llama_kv_cache::prefetch_seq_step(llama_seq_id seq_id, uint32_t max_blocks) {
    const auto result = prefetch_seq_step_impl(seq_id, max_blocks);
    return result.failed ? -1 : (int32_t) result.restored_blocks;
}

void llama_kv_cache::prefetch_seq_last_stats(
        uint64_t & owned_blocks,
        uint64_t & swapped_blocks,
        uint64_t & resident_blocks,
        uint64_t & released_blocks,
        uint64_t & invalid_cells,
        uint64_t & failures) const {
    owned_blocks    = paged_prefetch_seq_last_owned_blocks;
    swapped_blocks  = paged_prefetch_seq_last_swapped_blocks;
    resident_blocks = paged_prefetch_seq_last_resident_blocks;
    released_blocks = paged_prefetch_seq_last_released_blocks;
    invalid_cells   = paged_prefetch_seq_last_invalid_cells;
    failures        = paged_prefetch_seq_last_failures;
}

extern "C" bool llama_kv_cache_prefetch_seq_last_stats(
        llama_memory_t mem,
        uint64_t * owned_blocks,
        uint64_t * swapped_blocks,
        uint64_t * resident_blocks,
        uint64_t * released_blocks,
        uint64_t * invalid_cells,
        uint64_t * failures) {
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv || !owned_blocks || !swapped_blocks || !resident_blocks ||
            !released_blocks || !invalid_cells || !failures) {
        return false;
    }

    kv->prefetch_seq_last_stats(
            *owned_blocks,
            *swapped_blocks,
            *resident_blocks,
            *released_blocks,
            *invalid_cells,
            *failures);
    return true;
}

std::vector<uint8_t> llama_kv_cache::paged_unified_action_test_read_block_bytes(uint32_t block) const {
    std::vector<uint8_t> bytes;
    if (block >= paged_n_blocks || paged_block_size == 0) {
        return bytes;
    }

    const uint32_t begin = block * paged_block_size;
    const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
    for (uint32_t cell = begin; cell < end; ++cell) {
        for (const auto & layer : layers) {
            for (ggml_tensor * tensor : {
                    layer.k_stream.empty() ? nullptr : layer.k_stream[0],
                    (!layer.v || layer.v_stream.empty()) ? nullptr : layer.v_stream[0] }) {
                if (!tensor) {
                    continue;
                }
                const size_t row_size = tensor->nb[1];
                const size_t offset = bytes.size();
                bytes.resize(offset + row_size);
                ggml_backend_tensor_get(tensor, bytes.data() + offset, (size_t) cell * row_size, row_size);
            }
        }
    }
    return bytes;
}

bool llama_kv_cache::paged_unified_action_test_swap_out_block(uint32_t block) {
    if (block >= paged_block_states.size() || paged_block_states[block] != paged_block_state::RESIDENT) {
        return false;
    }
    paged_swap_out_block(block, true, false);
    return paged_block_states[block] == paged_block_state::SWAPPED;
}

llama_kv_backing_store_stats llama_kv_cache::paged_unified_action_test_read_backing_stats() const {
    return kv_swap_store ? kv_swap_store->get_stats() : llama_kv_backing_store_stats {};
}

void llama_kv_cache::paged_unified_action_test_arm_read_gate() {
    if (auto * store = dynamic_cast<llama_kv_backing_store_file *>(kv_swap_store.get())) {
        store->arm_test_read_gate();
    }
}

bool llama_kv_cache::paged_unified_action_test_read_gate_entered() const {
    const auto * store = dynamic_cast<const llama_kv_backing_store_file *>(kv_swap_store.get());
    return store && store->test_read_gate_entered();
}

void llama_kv_cache::paged_unified_action_test_release_read_gate() {
    if (auto * store = dynamic_cast<llama_kv_backing_store_file *>(kv_swap_store.get())) {
        store->release_test_read_gate();
    }
}

bool llama_kv_cache::paged_unified_action_test_start_pending_restore_read(uint32_t block) {
    if (!paged_restore_k2_enabled || block >= paged_block_states.size() ||
            paged_block_states[block] != paged_block_state::SWAPPED || paged_restore_worker_thread.joinable()) {
        return false;
    }
    std::vector<uint32_t> candidates = { block };
    std::vector<paged_restore_group_task> tasks;
    if (!paged_restore_group_plan(0, candidates, tasks) || tasks.size() != 1) {
        return false;
    }
    auto task = std::make_shared<paged_restore_group_task>(std::move(tasks.front()));
    size_t max_single_block_bytes = 0;
    for (const size_t block_bytes : task->block_bytes) {
        max_single_block_bytes = std::max(max_single_block_bytes, block_bytes);
    }
    paged_restore_k2_staging_bound_bytes = llama_paged_restore_staging_bound_bytes(
            paged_restore_group_byte_cap(), max_single_block_bytes);
    paged_restore_k2_staging_reuses = 0;
    if (!paged_restore_group_prepare(*task) || !paged_restore_group_start_async(task)) {
        std::vector<uint8_t>().swap(task->staging);
        return false;
    }
    paged_restore_k2_peak_staging_groups = 1;
    paged_restore_k2_peak_staging_bytes = task->staging.size();
    return task->read_submitted;
}

llama_kv_cache::paged_unified_action_test_io_fault_stats
llama_kv_cache::paged_unified_action_test_read_io_fault() const {
    return {
        paged_test_io_fault_.matching_attempts,
        paged_test_io_fault_.trigger_count,
        paged_test_io_fault_.failed_block,
        paged_test_io_fault_.failed_attempt_id,
    };
}

llama_kv_runtime_capability llama_kv_cache::get_kv_runtime_capability() const {
    const bool paged_layout_valid = kv_paged_enabled && !v_trans && n_stream == 1 &&
        paged_block_size != 0 && paged_n_blocks != 0 &&
        paged_block_states.size() == paged_n_blocks && !v_cells.empty();
    const bool write_transaction_open = paged_write_transaction_owner != nullptr;
    const bool ingraph_gather = paged_layout_valid && paged_ingraph_enabled &&
        paged_layers_supported && paged_row_idx_enabled;
    const bool backing_ready = paged_layout_valid && paged_swap_enabled && kv_swap_store;
    const bool swap_ready = backing_ready;
    const bool action_ready = !paged_write_context_invalid && !write_transaction_open;

    return {
        n_seq_max,
        n_stream,
        kv_unified,
        kv_paged_enabled,
        ingraph_gather,
        ingraph_gather && action_ready,
        swap_ready && action_ready,
        swap_ready && action_ready,
        backing_ready,
        backing_ready && paged_swap_explicit_only && !paged_idle_swap_requested,
    };
}

llama_kv_runtime_claimant llama_kv_cache::get_kv_runtime_claimant(llama_seq_id seq_id) const {
    llama_kv_runtime_claimant result;
    const bool paged_layout_valid = kv_paged_enabled && !v_trans && n_stream == 1 &&
        paged_block_size != 0 && paged_n_blocks != 0 &&
        paged_block_states.size() == paged_n_blocks && !v_cells.empty();
    if (!paged_layout_valid || seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return result;
    }

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    std::set<uint32_t> target_blocks;
    std::set<uint32_t> shared_blocks;
    for (uint32_t logical_cell = 0; logical_cell < cells.size(); ++logical_cell) {
        if (cells.is_empty(logical_cell)) {
            continue;
        }
        const uint32_t physical_cell = paged_resolve(logical_cell);
        if (physical_cell == PAGED_BLOCK_INVALID || physical_cell / paged_block_size >= paged_n_blocks) {
            return result;
        }
        const uint32_t block = physical_cell / paged_block_size;
        if (cells.seq_has(logical_cell, seq_id)) {
            target_blocks.insert(block);
        }
        if (cells.seq_count(logical_cell) != 1 || !cells.seq_has(logical_cell, seq_id)) {
            shared_blocks.insert(block);
        }
    }

    result.valid = true;
    result.target_blocks = (uint32_t) target_blocks.size();
    for (uint32_t block : target_blocks) {
        if (shared_blocks.count(block)) {
            result.shared_blocks += 1;
        } else if (paged_block_states[block] == paged_block_state::RESIDENT) {
            result.eligible_resident_blocks += 1;
        } else if (paged_block_states[block] == paged_block_state::SWAPPED) {
            result.swapped_blocks += 1;
        } else {
            result.blocked_blocks += 1;
        }
    }
    return result;
}

llama_kv_action_result llama_kv_cache::execute_action(
        const llama_kv_action_request & request) {
    llama_kv_action_result result;
    result.action = request.action;
    result.decision_id = request.decision_id;

    const bool paged_layout_valid = kv_paged_enabled && !v_trans && n_stream == 1 &&
        paged_block_size != 0 && paged_n_blocks != 0 &&
        paged_block_states.size() == paged_n_blocks && !v_cells.empty();
    const bool write_transaction_open = paged_write_transaction_owner != nullptr;
    const bool swap_ready = paged_layout_valid && paged_swap_enabled && kv_swap_store;

    result.capability.context_invalid = paged_write_context_invalid;
    result.capability.write_transaction_open = write_transaction_open;
    result.capability.can_prefetch = swap_ready && !paged_write_context_invalid && !write_transaction_open;
    result.capability.can_release = paged_layout_valid && paged_ingraph_enabled &&
        paged_layers_supported && paged_row_idx_enabled && !paged_write_context_invalid &&
        !write_transaction_open;
    result.capability.can_offload = swap_ready && !paged_write_context_invalid && !write_transaction_open;

    if (request.action == llama_kv_action::evaluate) {
        result.outcome = llama_kv_action_outcome::completed;
        return result;
    }
    if (request.action == llama_kv_action::noop) {
        return result;
    }
    if (paged_write_context_invalid) {
        result.outcome = llama_kv_action_outcome::rejected;
        result.reason = llama_kv_action_reason::context_invalid;
        result.fail_stop = true;
        return result;
    }
    if (write_transaction_open) {
        result.outcome = llama_kv_action_outcome::rejected;
        result.reason = llama_kv_action_reason::write_transaction_open;
        return result;
    }

    const auto publish_transaction = [&]() {
        result.state_changed = true;
        result.core_transaction_id = ++paged_unified_action_transaction_next;
    };

    if (request.action == llama_kv_action::prefetch) {
        if (request.max_blocks == 0 && !request.all_required) {
            result.reason = llama_kv_action_reason::zero_budget;
            return result;
        }
        if (request.seq_id < 0 || (size_t) request.seq_id >= seq_to_stream.size()) {
            result.outcome = llama_kv_action_outcome::rejected;
            result.reason = llama_kv_action_reason::invalid_sequence;
            return result;
        }
        if (!result.capability.can_prefetch) {
            // With swap disabled, no sequence can own a SWAPPED block. This is
            // a successful no-op that preserves legacy cache-reuse behavior.
            if (!paged_swap_enabled) {
                return result;
            }
            result.outcome = llama_kv_action_outcome::unsupported;
            result.reason = llama_kv_action_reason::unsupported;
            return result;
        }

        const auto prefetch = prefetch_seq_step_impl(
                request.seq_id, request.max_blocks, request.all_required);
        result.blocks = prefetch.restored_blocks;
        result.bytes = prefetch.restored_bytes;
        result.shortfall_bytes = prefetch.candidate_bytes - prefetch.restored_bytes;
        if (prefetch.restored_blocks > 0) {
            publish_transaction();
        }
        if (prefetch.failed) {
            result.outcome = prefetch.restored_blocks > 0
                ? llama_kv_action_outcome::partial_failure
                : llama_kv_action_outcome::failed;
            result.reason = llama_kv_action_reason::prefetch_failed;
            result.io_failure = true;
            result.fail_stop = request.correctness_required;
            return result;
        }
        result.outcome = prefetch.restored_blocks > 0
            ? llama_kv_action_outcome::completed
            : llama_kv_action_outcome::no_op;
        return result;
    }

    if (request.action == llama_kv_action::release) {
        if (request.target_bytes == 0) {
            result.reason = llama_kv_action_reason::zero_budget;
            return result;
        }
        if (request.max_blocks == 0) {
            result.reason = llama_kv_action_reason::scan_budget_exhausted;
            result.shortfall_bytes = request.target_bytes;
            return result;
        }
        if (!result.capability.can_release) {
            result.outcome = llama_kv_action_outcome::unsupported;
            result.reason = llama_kv_action_reason::unsupported;
            return result;
        }

        paged_bounded_release_counters counters;
        counters.calls = &paged_unified_release_calls;
        counters.blocks = &paged_unified_release_blocks;
        counters.bytes = &paged_unified_release_bytes;
        const auto release = paged_release_blocks_bounded_impl(
                request.target_bytes, request.max_blocks, counters, false);
        paged_unified_release_calls += 1;
        result.blocks = release.released_blocks;
        result.bytes = release.released_bytes;
        result.relieved_bytes = release.released_bytes;
        result.shortfall_bytes = release.shortfall_bytes;
        if (release.ownership_aborted) {
            result.outcome = llama_kv_action_outcome::rejected;
            result.reason = llama_kv_action_reason::blocked;
            result.fail_stop = true;
            return result;
        }
        if (release.released_blocks > 0) {
            publish_transaction();
        }
        if (release.madvise_failures > 0) {
            result.outcome = release.released_blocks > 0
                ? llama_kv_action_outcome::partial_failure
                : llama_kv_action_outcome::failed;
            result.reason = llama_kv_action_reason::failed;
            result.io_failure = true;
            return result;
        }
        if (release.released_bytes >= request.target_bytes) {
            result.outcome = llama_kv_action_outcome::completed;
            result.reason = llama_kv_action_reason::target_satisfied;
            return result;
        }
        if (release.scan_budget_exhausted) {
            result.outcome = release.released_blocks > 0
                ? llama_kv_action_outcome::completed
                : llama_kv_action_outcome::no_op;
            result.reason = llama_kv_action_reason::scan_budget_exhausted;
            return result;
        }
        if (release.released_blocks == 0) {
            result.reason = llama_kv_action_reason::no_candidate;
            return result;
        }

        result.outcome = llama_kv_action_outcome::completed;
        result.reason = llama_kv_action_reason::target_shortfall;
        return result;
    }

    if (request.action != llama_kv_action::offload) {
        result.outcome = llama_kv_action_outcome::rejected;
        result.reason = llama_kv_action_reason::unsupported;
        return result;
    }
    if (request.target_bytes == 0) {
        result.reason = llama_kv_action_reason::zero_budget;
        return result;
    }
    if (request.max_blocks == 0) {
        result.reason = llama_kv_action_reason::scan_budget_exhausted;
        result.shortfall_bytes = request.target_bytes;
        return result;
    }
    if (!result.capability.can_offload) {
        result.outcome = llama_kv_action_outcome::unsupported;
        result.reason = llama_kv_action_reason::unsupported;
        return result;
    }
    if (request.seq_id < 0 || (size_t) request.seq_id >= seq_to_stream.size()) {
        result.outcome = llama_kv_action_outcome::rejected;
        result.reason = llama_kv_action_reason::invalid_sequence;
        return result;
    }
    if ((size_t) request.seq_id < paged_prefetch_protected_seq.size() &&
            paged_prefetch_protected_seq.test(request.seq_id)) {
        result.outcome = llama_kv_action_outcome::rejected;
        result.reason = llama_kv_action_reason::protected_sequence;
        return result;
    }

    const auto & cells = v_cells[seq_to_stream[request.seq_id]];
    std::set<uint32_t> target_blocks;
    std::set<uint32_t> shared_blocks;
    for (uint32_t logical_cell = 0; logical_cell < cells.size(); ++logical_cell) {
        if (cells.is_empty(logical_cell)) continue;
        const uint32_t physical_cell = paged_resolve(logical_cell);
        if (physical_cell == PAGED_BLOCK_INVALID || physical_cell / paged_block_size >= paged_n_blocks) {
            result.outcome = llama_kv_action_outcome::rejected;
            result.reason = llama_kv_action_reason::ownership_invalid;
            result.fail_stop = true;
            return result;
        }
        const uint32_t block = physical_cell / paged_block_size;
        if (cells.seq_has(logical_cell, request.seq_id)) {
            target_blocks.insert(block);
        }
        if (cells.seq_count(logical_cell) != 1 || !cells.seq_has(logical_cell, request.seq_id)) {
            shared_blocks.insert(block);
        }
    }
    if (target_blocks.empty()) {
        result.reason = llama_kv_action_reason::no_eligible_block;
        result.shortfall_bytes = request.target_bytes;
        return result;
    }
    for (uint32_t block : target_blocks) {
        if (shared_blocks.count(block)) {
            result.outcome = llama_kv_action_outcome::rejected;
            result.reason = llama_kv_action_reason::shared_block;
            return result;
        }
        if (paged_block_states[block] != paged_block_state::RESIDENT &&
                paged_block_states[block] != paged_block_state::SWAPPED) {
            result.outcome = llama_kv_action_outcome::rejected;
            result.reason = llama_kv_action_reason::state_rejected;
            return result;
        }
    }

    uint64_t bytes_per_cell = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) bytes_per_cell += layer.k_stream[0]->nb[1];
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) bytes_per_cell += layer.v_stream[0]->nb[1];
    }

    for (auto it = target_blocks.rbegin(); it != target_blocks.rend(); ++it) {
        if (result.blocks >= request.max_blocks || result.bytes >= request.target_bytes) {
            break;
        }

        const uint32_t physical_block = *it;
        if (paged_block_states[physical_block] == paged_block_state::SWAPPED) {
            continue;
        }
        if (paged_block_states[physical_block] != paged_block_state::RESIDENT) {
            result.outcome = result.blocks > 0
                ? llama_kv_action_outcome::partial_failure
                : llama_kv_action_outcome::rejected;
            result.reason = llama_kv_action_reason::state_rejected;
            result.shortfall_bytes = request.target_bytes > result.bytes
                ? request.target_bytes - result.bytes
                : 0;
            if (result.blocks > 0) publish_transaction();
            return result;
        }

        const uint32_t begin = physical_block * paged_block_size;
        const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
        const uint64_t block_bytes = bytes_per_cell * (end - begin);
        const uint64_t failures_before = paged_swap_backend_failures;
        uint64_t block_relieved_bytes = 0;
        bool block_io_failure = false;
        int block_io_errno = 0;
        paged_swap_out_block(
                physical_block, true, false,
                &block_relieved_bytes, &block_io_failure, &block_io_errno);
        if (paged_block_states[physical_block] != paged_block_state::SWAPPED) {
            result.outcome = result.blocks > 0
                ? llama_kv_action_outcome::partial_failure
                : llama_kv_action_outcome::failed;
            result.reason = llama_kv_action_reason::io_failure;
            result.io_failure = block_io_failure || paged_swap_backend_failures > failures_before;
            result.io_errno = result.io_failure ? block_io_errno : 0;
            result.shortfall_bytes = request.target_bytes > result.bytes
                ? request.target_bytes - result.bytes
                : 0;
            if (result.blocks > 0) publish_transaction();
            return result;
        }

        result.blocks += 1;
        result.bytes += block_bytes;
        result.relieved_bytes += block_relieved_bytes;
    }

    result.shortfall_bytes = request.target_bytes > result.bytes
        ? request.target_bytes - result.bytes
        : 0;
    if (result.blocks == 0) {
        result.reason = llama_kv_action_reason::no_candidate;
        return result;
    }

    publish_transaction();
    result.outcome = llama_kv_action_outcome::completed;
    if (result.shortfall_bytes == 0) {
        result.reason = llama_kv_action_reason::target_satisfied;
    } else if (result.blocks >= request.max_blocks) {
        result.reason = llama_kv_action_reason::scan_budget_exhausted;
    } else {
        result.reason = llama_kv_action_reason::target_shortfall;
    }
    return result;
}

void llama_kv_cache::set_seq_prefetch_protected(llama_seq_id seq_id, bool enabled) {
    if (seq_id < 0 || (size_t) seq_id >= LLAMA_MAX_SEQ) {
        return;
    }
    paged_prefetch_protected_seq.set(seq_id, enabled);
}

void llama_kv_cache::defer_idle_swapout(int32_t n_steps) {
    if (n_steps <= 0) {
        return;
    }
    paged_defer_idle_swapout_steps = std::max(paged_defer_idle_swapout_steps, n_steps);
}

extern "C" bool llama_kv_cache_set_seq_prefetch_protected(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        bool           enabled) {
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) {
        return false;
    }
    kv->set_seq_prefetch_protected(seq_id, enabled);
    return true;
}

extern "C" bool llama_kv_cache_defer_idle_swapout(
        llama_memory_t mem,
        int32_t        n_steps) {
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv) {
        return false;
    }
    kv->defer_idle_swapout(n_steps);
    return true;
}

void llama_kv_cache::paged_stability_stats_for_seq(
        llama_seq_id seq_id,
        llama_kv_stability_stats & stats) const {
    stats = {};
    stats.swap_out_calls = paged_swap_out_calls;
    stats.swap_in_calls  = paged_swap_in_calls;
    stats.fatal_counters =
        paged_swapped_active_visible_violation +
        paged_swapped_active_visible_violation_rows +
        paged_swapped_active_visible_violation_blocks +
        paged_row_mapping_invalid_fatal +
        paged_write_mapping_invalid_fatal +
        paged_active_row_nonresident_fatal +
        paged_input_setup_fatal +
        paged_write_to_swapped_block +
        paged_write_to_swapped_block_seq;
    stats.pending_error = has_paged_swap_error() ? 1 : 0;

    if (kv_swap_store) {
        if (const auto * store = dynamic_cast<const llama_kv_backing_store_file *>(kv_swap_store.get())) {
            stats.backing_capacity = store->get_capacity();
            stats.backing_stat_valid =
                store->stat_actual(stats.backing_size, stats.backing_blocks_512) ? 1 : 0;
        }
    }

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size() ||
            !kv_paged_enabled || paged_block_size == 0 || v_cells.empty()) {
        return;
    }

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    std::set<uint32_t> target_blocks;
    std::set<uint32_t> shared_blocks;
    for (uint32_t cell = 0; cell < cells.size(); ++cell) {
        if (cells.is_empty(cell)) {
            continue;
        }

        const uint32_t phys_cell = paged_resolve(cell);
        if (phys_cell == PAGED_BLOCK_INVALID) {
            continue;
        }

        const uint32_t physical_block = phys_cell / paged_block_size;
        if (physical_block < paged_block_states.size()) {
            if (cells.seq_has(cell, seq_id)) {
                target_blocks.insert(physical_block);
                if (cells.seq_count(cell) > 1) {
                    shared_blocks.insert(physical_block);
                }
            } else if (cells.seq_count(cell) > 0) {
                shared_blocks.insert(physical_block);
            }
        }
    }

    for (const uint32_t block : target_blocks) {
        if (shared_blocks.count(block)) {
            continue;
        }
        if (paged_block_states[block] == paged_block_state::SWAPPED) {
            stats.swapped_blocks += 1;
        } else if (paged_block_states[block] == paged_block_state::RESIDENT) {
            stats.resident_blocks += 1;
        }
    }
    stats.target_blocks = stats.swapped_blocks + stats.resident_blocks;
}

bool llama_kv_cache::paged_stability_cycle(
        llama_seq_id seq_id,
        llama_kv_stability_stats & after_swap_out,
        llama_kv_stability_stats & after_swap_in) {
    after_swap_out = {};
    after_swap_in  = {};

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size() ||
            !kv_paged_enabled || !paged_swap_enabled || !kv_swap_store ||
            v_trans || n_stream != 1 || paged_block_size == 0 ||
            v_cells.empty() || paged_block_states.size() != paged_n_blocks) {
        return false;
    }

    const auto & cells = v_cells[seq_to_stream[seq_id]];
    const auto collect_resident_blocks = [&]() {
        std::set<uint32_t> blocks;
        std::set<uint32_t> rejected;
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }

            const uint32_t phys_cell = paged_resolve(cell);
            if (phys_cell == PAGED_BLOCK_INVALID) {
                blocks.clear();
                return blocks;
            }

            const uint32_t physical_block = phys_cell / paged_block_size;
            if (physical_block >= paged_block_states.size() ||
                    paged_block_states[physical_block] != paged_block_state::RESIDENT) {
                continue;
            }

            if (cells.seq_has(cell, seq_id)) {
                if (cells.seq_count(cell) > 1) {
                    blocks.erase(physical_block);
                    rejected.insert(physical_block);
                } else if (!rejected.count(physical_block)) {
                    blocks.insert(physical_block);
                }
                continue;
            }

            if (cells.seq_count(cell) > 0) {
                blocks.erase(physical_block);
                rejected.insert(physical_block);
            }
        }
        return blocks;
    };

    std::set<uint32_t> blocks = collect_resident_blocks();
    if (blocks.empty()) {
        llama_kv_stability_stats current;
        paged_stability_stats_for_seq(seq_id, current);
        if (current.swapped_blocks > 0 && prefetch_seq(seq_id) > 0) {
            blocks = collect_resident_blocks();
        }
    }

    if (blocks.empty()) {
        return false;
    }

    const uint64_t swap_out_before = paged_swap_out_calls;
    for (const uint32_t physical_block : blocks) {
        paged_swap_out_block(physical_block);
    }
    paged_stability_stats_for_seq(seq_id, after_swap_out);
    if (after_swap_out.swap_out_calls <= swap_out_before ||
            after_swap_out.target_blocks != blocks.size() ||
            after_swap_out.swapped_blocks != blocks.size() ||
            after_swap_out.pending_error != 0) {
        return false;
    }

    const uint64_t swap_in_before = paged_swap_in_calls;
    const int32_t restored = prefetch_seq(seq_id);
    paged_stability_stats_for_seq(seq_id, after_swap_in);
    if (restored <= 0 ||
            after_swap_in.swap_in_calls <= swap_in_before ||
            after_swap_in.target_blocks != blocks.size() ||
            after_swap_in.resident_blocks != blocks.size() ||
            after_swap_in.swapped_blocks != 0 ||
            after_swap_in.pending_error != 0) {
        return false;
    }

    return true;
}

extern "C" bool llama_kv_cache_paged_stability_cycle(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        llama_kv_stability_stats * after_swap_out,
        llama_kv_stability_stats * after_swap_in) {
    if (llama_kv_stability_binary_markers[0] == '\0') {
        return false;
    }
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv || !after_swap_out || !after_swap_in) {
        return false;
    }
    return kv->paged_stability_cycle(seq_id, *after_swap_out, *after_swap_in);
}

extern "C" bool llama_kv_cache_paged_stability_stats(
        llama_memory_t mem,
        llama_seq_id   seq_id,
        llama_kv_stability_stats * stats) {
    if (llama_kv_stability_binary_markers[0] == '\0') {
        return false;
    }
    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (!kv || !stats) {
        return false;
    }
    kv->paged_stability_stats_for_seq(seq_id, *stats);
    return true;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

void llama_kv_cache::paged_init(uint32_t kv_size) {
    if (!kv_paged_enabled) {
        return;
    }

    paged_kv_size = kv_size;
    paged_n_blocks = (kv_size + paged_block_size - 1) / paged_block_size;
    paged_block_table.resize(paged_n_blocks);
    paged_block_used.assign(paged_n_blocks, 0);
    paged_block_states.assign(paged_n_blocks, paged_block_state::UNUSED);
    paged_swap_offsets.assign(kv_size, 0);
    paged_swap_sizes.assign(kv_size, 0);
    paged_pending_write_cells.assign(kv_size, 0);
    paged_pending_write_blocks.clear();
    paged_write_transaction_owner = nullptr;
    paged_write_context_invalid = false;
    paged_write_context_invalid_cause = llama_paged_swap_error_reason::NONE;
    paged_swap_error = {};
    paged_release_post_ranges.clear();
    paged_released_ranges_by_block.assign(paged_n_blocks, {});
    paged_free_list.resize(paged_n_blocks);

    paged_build_block_table();
    paged_release_scan_reset();

    paged_alloc_calls     = 0;
    paged_blocks_in_use   = 0;
    paged_identity_checks = 0;
    paged_identity_fail   = 0;
}

void llama_kv_cache::paged_reset() {
    if (!kv_paged_enabled) {
        return;
    }

    if (paged_n_blocks == 0) {
        return;
    }

    paged_block_used.assign(paged_n_blocks, 0);
    paged_block_states.assign(paged_n_blocks, paged_block_state::UNUSED);
    paged_swap_offsets.assign(paged_kv_size, 0);
    paged_swap_sizes.assign(paged_kv_size, 0);
    paged_pending_write_cells.assign(paged_kv_size, 0);
    paged_pending_write_blocks.clear();
    paged_write_transaction_owner = nullptr;
    paged_write_context_invalid = false;
    paged_write_context_invalid_cause = llama_paged_swap_error_reason::NONE;
    paged_swap_error = {};
    paged_release_post_ranges.clear();
    paged_released_ranges_by_block.assign(paged_n_blocks, {});
    paged_free_list.resize(paged_n_blocks);
    paged_build_block_table();
    paged_release_scan_reset();
    paged_blocks_in_use = 0;
    paged_swap_pending = false;
    paged_swap_pending_n_kv = 0;
}

void llama_kv_cache::paged_build_block_table() {
    if (!kv_paged_enabled) {
        return;
    }

    paged_mapping_generation = paged_mapping_generation == UINT64_MAX
        ? 1
        : paged_mapping_generation + 1;
    paged_block_mapping_changed = 0;
    paged_mapping_oob_fail = 0;
    paged_non_identity_enabled = false;

    if (paged_n_blocks == 0) {
        return;
    }

    for (uint32_t i = 0; i < paged_n_blocks; ++i) {
        paged_block_table[i] = i;
        paged_free_list[paged_n_blocks - 1 - i] = i;
    }

    if (paged_n_blocks <= 1 || paged_shift == 0 || paged_kv_size % paged_block_size != 0) {
        return;
    }

    const uint32_t span = paged_n_blocks - 1;
    const uint32_t shift = paged_shift % span;
    if (shift == 0) {
        return;
    }

    paged_block_table[0] = 0;
    for (uint32_t i = 1; i < paged_n_blocks; ++i) {
        paged_block_table[i] = 1 + ((i - 1 + shift) % span);
    }

    std::vector<uint8_t> seen(paged_n_blocks, 0);
    for (uint32_t i = 0; i < paged_n_blocks; ++i) {
        const uint32_t phys = paged_block_table[i];
        if (phys >= paged_n_blocks || seen[phys]) {
            paged_mapping_oob_fail += 1;
            paged_block_table[i] = i;
            continue;
        }
        seen[phys] = 1;
        if (phys != i) {
            paged_block_mapping_changed += 1;
        }
    }

    paged_non_identity_enabled = paged_mapping_oob_fail == 0 && paged_block_mapping_changed > 0;
}

void llama_kv_cache::paged_release_scan_reset() {
    paged_release_scan_cursor = 0;
    paged_release_scan_scanned_since_release = 0;
    paged_release_scan_block_size = paged_block_size;
    paged_release_scan_n_blocks = paged_n_blocks;
    paged_release_scan_kv_size = paged_kv_size;
    paged_release_scan_mapping_generation = paged_mapping_generation;
}

void llama_kv_cache::paged_release_scan_sync_layout() {
    if (paged_release_scan_block_size != paged_block_size ||
            paged_release_scan_n_blocks != paged_n_blocks ||
            paged_release_scan_kv_size != paged_kv_size ||
            paged_release_scan_mapping_generation != paged_mapping_generation ||
            paged_release_scan_cursor >= paged_n_blocks ||
            paged_release_scan_scanned_since_release >= paged_n_blocks) {
        paged_release_scan_reset();
    }
}

void llama_kv_cache::paged_note_cells(const slot_info & sinfo) {
    if (!kv_paged_enabled) {
        return;
    }

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (const uint32_t cell : sinfo.idxs[s]) {
            const uint32_t logical_block = cell / paged_block_size;
            if (logical_block >= paged_n_blocks) {
                continue;
            }

            const uint32_t physical_block = paged_block_table[logical_block];
            if (physical_block == PAGED_BLOCK_INVALID || physical_block >= paged_n_blocks) {
                continue;
            }

            if (!paged_block_used[physical_block]) {
                const bool reusing_released =
                    physical_block < paged_block_states.size() &&
                    paged_block_states[physical_block] == paged_block_state::RELEASED;
                paged_block_used[physical_block] = 1;
                if (physical_block < paged_block_states.size()) {
                    paged_block_states[physical_block] = reusing_released ?
                        paged_block_state::PENDING_WRITE : paged_block_state::RESIDENT;
                }
                if (reusing_released) {
                    paged_block_release_reuse_allocations += 1;
                    if (paged_release_test_reuse &&
                            paged_release_test_reuse_block == PAGED_BLOCK_INVALID) {
                        paged_release_test_reuse_block = physical_block;
                        paged_release_test_reuse_pending_seen =
                            paged_block_states[physical_block] == paged_block_state::PENDING_WRITE;
                        paged_release_test_reuse_fresh_bytes_before = paged_release_fresh_verify_bytes;
                        const uint32_t begin = physical_block * paged_block_size;
                        const uint32_t end = std::min<uint32_t>(
                                begin + paged_block_size, paged_swap_offsets.size());
                        paged_release_test_reuse_metadata_absent = true;
                        for (uint32_t cell = begin; cell < end; ++cell) {
                            if (paged_swap_offsets[cell] != 0 || paged_swap_sizes[cell] != 0) {
                                paged_release_test_reuse_metadata_absent = false;
                                break;
                            }
                        }
                    }
                    if (std::find(paged_pending_write_blocks.begin(), paged_pending_write_blocks.end(),
                                physical_block) == paged_pending_write_blocks.end()) {
                        paged_pending_write_blocks.push_back(physical_block);
                    }
                }
                paged_blocks_in_use += 1;
                paged_alloc_calls += 1;
                auto it = std::find(paged_free_list.begin(), paged_free_list.end(), physical_block);
                if (it != paged_free_list.end()) {
                    paged_free_list.erase(it);
                }
            } else if (physical_block < paged_block_states.size() &&
                    paged_block_states[physical_block] == paged_block_state::UNUSED) {
                paged_block_states[physical_block] = paged_block_state::RESIDENT;
            }
            if (physical_block < paged_block_states.size() &&
                    paged_block_states[physical_block] == paged_block_state::PENDING_WRITE) {
                const uint32_t phys = paged_resolve(cell);
                if (phys != PAGED_BLOCK_INVALID && phys < paged_pending_write_cells.size()) {
                    paged_pending_write_cells[phys] = 1;
                }
            }
        }
    }
}

uint32_t llama_kv_cache::paged_resolve(uint32_t cell) const {
    if (!kv_paged_enabled || paged_block_size == 0) {
        return cell;
    }

    if (paged_base_timing_enabled) {
        paged_base_timing_paged_resolve_calls += 1;
    }

    paged_logical_to_physical_checks += 1;

    const uint32_t logical_block = cell / paged_block_size;
    const uint32_t offset        = cell % paged_block_size;
    if (logical_block >= paged_block_table.size()) {
        paged_logical_to_physical_fail += 1;
        return PAGED_BLOCK_INVALID;
    }

    const uint32_t physical_block = paged_block_table[logical_block];
    if (physical_block == PAGED_BLOCK_INVALID || physical_block >= paged_n_blocks) {
        paged_logical_to_physical_fail += 1;
        return PAGED_BLOCK_INVALID;
    }

    const uint32_t phys_cell = physical_block * paged_block_size + offset;
    if (phys_cell >= paged_kv_size) {
        paged_logical_to_physical_fail += 1;
        return PAGED_BLOCK_INVALID;
    }

    return phys_cell;
}

uint32_t llama_kv_cache::paged_write_resolve(uint32_t cell) const {
    if (!kv_paged_enabled) {
        return cell;
    }

    const uint32_t phys = paged_resolve(cell);
    paged_write_resolve_checks += 1;
    if (phys == PAGED_BLOCK_INVALID) {
        paged_write_resolve_fail += 1;
        return PAGED_BLOCK_INVALID;
    }
    if (phys != cell) {
        paged_write_resolve_changed += 1;
    }

    return phys;
}

bool llama_kv_cache::paged_test_mapping_fault_token_matches(const llama_ubatch * ubatch, uint32_t token) const {
    if (!ubatch || token >= (uint32_t) ubatch->n_tokens) {
        return false;
    }

    for (int32_t sid = 0; sid < ubatch->n_seq_id[token]; ++sid) {
        if (ubatch->seq_id[token][sid] == paged_test_mapping_fault_.target_seq) {
            return true;
        }
    }

    return false;
}

bool llama_kv_cache::paged_test_mapping_fault_should_inject_read(uint32_t logical_cell) const {
    (void) logical_cell;

    return kv_paged_enabled &&
        paged_test_mapping_fault_.scope == paged_test_mapping_fail_scope::READ &&
        paged_blocks_swapped_out > 0 &&
        (!paged_test_mapping_fault_.fail_once || !paged_test_mapping_fault_.consumed);
}

bool llama_kv_cache::paged_test_mapping_fault_should_inject_write(
        const llama_ubatch * ubatch,
        uint32_t token) const {
    return kv_paged_enabled &&
        paged_test_mapping_fault_.scope == paged_test_mapping_fail_scope::WRITE &&
        paged_blocks_swapped_out > 0 &&
        (!paged_test_mapping_fault_.fail_once || !paged_test_mapping_fault_.consumed) &&
        paged_test_mapping_fault_token_matches(ubatch, token);
}

void llama_kv_cache::paged_test_mapping_fault_log_and_consume(
        bool read_scope,
        uint32_t logical_cell,
        llama_paged_swap_error_reason reason,
        uint64_t fatal_counter_next) const {
    const char * scope = read_scope ? "read" : "write";
    const char * counter = read_scope ? "paged_row_mapping_invalid_fatal" : "paged_write_mapping_invalid_fatal";
    fprintf(stderr,
            "TEST MAPPING FAULT INJECTION scope=%s target_seq=%d logical_cell=%u "
            "failure_reason=%s fatal_counter=%s fatal_counter_next=%llu trigger_count=%llu\n",
            scope,
            (int) paged_test_mapping_fault_.target_seq,
            logical_cell,
            llama_paged_swap_error_reason_name(reason),
            counter,
            (unsigned long long) fatal_counter_next,
            (unsigned long long) (paged_test_mapping_fault_triggers + 1));

    paged_test_mapping_fault_triggers += 1;
    if (read_scope) {
        paged_test_mapping_fault_read_triggers += 1;
    } else {
        paged_test_mapping_fault_write_triggers += 1;
    }
    if (paged_test_mapping_fault_.fail_once) {
        paged_test_mapping_fault_.consumed = true;
    }
}

uint32_t llama_kv_cache::paged_write_resolve_input(
        uint32_t cell,
        const llama_ubatch * ubatch,
        uint32_t token) const {
    const uint32_t phys = paged_write_resolve(cell);
    if (phys != PAGED_BLOCK_INVALID &&
            paged_test_mapping_fault_should_inject_write(ubatch, token)) {
        paged_test_mapping_fault_log_and_consume(
                false,
                cell,
                llama_paged_swap_error_reason::PAGED_WRITE_MAPPING_INVALID,
                paged_write_mapping_invalid_fatal + 1);
        return PAGED_BLOCK_INVALID;
    }

    return phys;
}

bool llama_kv_cache::paged_validate_write_mapping(uint32_t phys_cell) const {
    if (!kv_paged_enabled) {
        return true;
    }

    if (phys_cell == PAGED_BLOCK_INVALID ||
            phys_cell >= paged_kv_size ||
            paged_block_size == 0) {
        paged_write_mapping_invalid_fatal += 1;
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_MAPPING_INVALID,
                PAGED_BLOCK_INVALID, phys_cell);
        return false;
    }

    const uint32_t physical_block = phys_cell / paged_block_size;
    if (physical_block >= paged_n_blocks ||
            physical_block >= paged_block_states.size()) {
        paged_write_mapping_invalid_fatal += 1;
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_MAPPING_INVALID,
                physical_block, phys_cell);
        return false;
    }

    return true;
}

void llama_kv_cache::clear_paged_swap_error() {
    paged_swap_error = {};
    if (paged_write_context_invalid) {
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID,
                PAGED_BLOCK_INVALID,
                PAGED_BLOCK_INVALID,
                static_cast<int>(paged_write_context_invalid_cause));
        return;
    }
    if (paged_test_io_fault_.scope == paged_test_io_fail_scope::ACTIVE_SWAP_IN &&
            paged_test_io_fault_.failed_block != PAGED_BLOCK_INVALID &&
            paged_test_io_fault_.failed_block < paged_block_states.size() &&
            paged_block_states[paged_test_io_fault_.failed_block] == paged_block_state::SWAPPED) {
        auto block_state_name = [](paged_block_state state) {
            switch (state) {
                case paged_block_state::UNUSED:   return "UNUSED";
                case paged_block_state::RESIDENT: return "RESIDENT";
                case paged_block_state::RELEASED: return "RELEASED";
                case paged_block_state::SWAPPED:  return "SWAPPED";
                case paged_block_state::PENDING_WRITE: return "PENDING_WRITE";
                case paged_block_state::INVALID: return "INVALID";
            }
            return "UNKNOWN";
        };

        const uint32_t retry_block = paged_test_io_fault_.failed_block;
        const uint64_t retry_fault_attempt_id = paged_test_io_fault_.failed_attempt_id;
        const uint64_t retry_swap_in_before = paged_blocks_swapped_in;
        const paged_block_state retry_state_before = paged_block_states[retry_block];
        const bool restored = paged_swap_in_block(
                retry_block,
                true,
                llama_paged_swap_error_reason::PAGED_SWAP_IN_IO_ERROR);
        if (restored && paged_test_io_fault_.failed_block == retry_block &&
                paged_block_states[retry_block] == paged_block_state::RESIDENT &&
                paged_blocks_swapped_in > retry_swap_in_before) {
            paged_test_io_fault_.retry_success_count += 1;
            LLAMA_LOG_ERROR(
                    "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=active_swap_in kind=read_eof_once "
                    "target_seq=%d physical_block=%u state_before=%s state_after=%s "
                    "swap_in_counter_before=%llu swap_in_counter_after=%llu "
                    "fault_attempt_id=%llu retry_success_count=%llu\n",
                    (int) paged_test_io_fault_.target_seq,
                    retry_block,
                    block_state_name(retry_state_before),
                    block_state_name(paged_block_states[retry_block]),
                    (unsigned long long) retry_swap_in_before,
                    (unsigned long long) paged_blocks_swapped_in,
                    (unsigned long long) retry_fault_attempt_id,
                    (unsigned long long) paged_test_io_fault_.retry_success_count);
            paged_test_io_fault_.failed_block = PAGED_BLOCK_INVALID;
            paged_test_io_fault_.failed_attempt_id = 0;
        }
    }
}

void llama_kv_cache::set_paged_swap_error(
        llama_paged_swap_error_reason reason,
        uint32_t physical_block,
        uint32_t physical_cell,
        int backend_status,
        int backend_errno) const {
    if (paged_swap_error.pending) {
        return;
    }

    paged_swap_error.pending = true;
    paged_swap_error.reason = reason;
    paged_swap_error.physical_block = physical_block;
    paged_swap_error.physical_cell = physical_cell;
    paged_swap_error.backend_status = backend_status;
    paged_swap_error.backend_errno = backend_errno;
}

void llama_kv_cache_context::set_paged_input_setup_error() const {
    if (!kv) {
        return;
    }

    kv->paged_input_setup_fatal += 1;
    kv->set_paged_swap_error(
            llama_paged_swap_error_reason::INPUT_SETUP_FAILURE,
            llama_kv_cache::PAGED_BLOCK_INVALID,
            llama_kv_cache::PAGED_BLOCK_INVALID);
}

bool llama_kv_cache::has_paged_swap_error() const {
    return paged_swap_error.pending;
}

llama_paged_swap_error llama_kv_cache::get_paged_swap_error() const {
    return paged_swap_error;
}

bool llama_kv_cache::paged_ensure_write_resident(uint32_t phys_cell) const {
    // Safety is driven by actual block state, NOT by the legacy
    // paged_block_release_enabled / paged_swap_enabled gates.  In a bounded-only
    // configuration (LLAMA_KV_PRESSURE_BOUNDED_RELEASE set, LLAMA_KV_PAGED_RELEASE
    // and LLAMA_KV_SWAP_PAGED unset) both legacy flags are false, so gating on
    // them would short-circuit to `return true` here and bypass every
    // RELEASED / PENDING_WRITE / SWAPPED check — a fail-open hole.  Only the
    // non-paged case (no block state exists at all) may return true early.
    if (!kv_paged_enabled || phys_cell == PAGED_BLOCK_INVALID || paged_block_size == 0) {
        return true;
    }

    paged_block_ensure_calls += 1;

    const uint32_t physical_block = phys_cell / paged_block_size;
    if (physical_block >= paged_block_states.size()) {
        return true;
    }

    if (paged_block_states[physical_block] == paged_block_state::INVALID) {
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID,
                physical_block, phys_cell);
        return false;
    }

    if (paged_block_states[physical_block] == paged_block_state::RELEASED) {
        paged_block_ensure_released += 1;
        paged_write_mapping_invalid_fatal += 1;
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_MAPPING_INVALID,
                physical_block, phys_cell);
        return false;
    } else if (paged_block_states[physical_block] == paged_block_state::SWAPPED) {
        paged_swap_write_swapped_hits += 1;
        if (paged_swap_in_block(
                    physical_block,
                    true,
                    llama_paged_swap_error_reason::SWAP_IN_IO_FAILURE) &&
                paged_block_states[physical_block] == paged_block_state::RESIDENT) {
            paged_swap_write_swap_in_calls += 1;
        } else {
            paged_swap_write_swap_in_failures += 1;
            return false;
        }
    }

    if (paged_block_states[physical_block] == paged_block_state::PENDING_WRITE) {
        // PENDING_WRITE is only valid if the cell belongs to the current
        // write transaction. Reject stale PENDING_WRITE cells left over
        // from a different (aborted / concurrent) transaction.
        if (paged_write_transaction_owner != nullptr &&
                phys_cell < paged_pending_write_cells.size() &&
                paged_pending_write_cells[phys_cell]) {
            return true;
        }
        paged_block_ensure_pending_write_rejected += 1;
        paged_write_mapping_invalid_fatal += 1;
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_MAPPING_INVALID,
                physical_block, phys_cell);
        return false;
    }

    return paged_block_states[physical_block] == paged_block_state::RESIDENT;
}

bool llama_kv_cache::paged_check_read_resident(uint32_t phys_cell, bool required_by_active) const {
    if (!paged_resume_timing_enabled && !paged_resume_timing_step_enabled) {
        return paged_check_read_resident_impl(phys_cell, required_by_active);
    }

    const uint64_t start_us = llama_paged_timing_now_us();
    const bool ok = paged_check_read_resident_impl(phys_cell, required_by_active);
    paged_timing_check_read_us += llama_paged_timing_now_us() - start_us;
    paged_timing_check_read_calls += 1;
    return ok;
}

bool llama_kv_cache::paged_check_read_resident_impl(uint32_t phys_cell, bool required_by_active) const {
    // Safety is driven by actual block state, NOT by the legacy
    // paged_block_release_enabled / paged_swap_enabled gates.  In a bounded-only
    // configuration (LLAMA_KV_PRESSURE_BOUNDED_RELEASE set, LLAMA_KV_PAGED_RELEASE
    // and LLAMA_KV_SWAP_PAGED unset) both legacy flags are false, so gating on
    // them would short-circuit to `return true` here and silently allow reads of
    // RELEASED / stale-PENDING_WRITE / SWAPPED blocks — a fail-open hole.  Only
    // the non-paged case (no block state exists at all) may return true early.
    if (!kv_paged_enabled || phys_cell == PAGED_BLOCK_INVALID || paged_block_size == 0) {
        return true;
    }

    const uint32_t physical_block = phys_cell / paged_block_size;
    if (physical_block >= paged_block_states.size()) {
        return true;
    }

    if (paged_block_states[physical_block] == paged_block_state::INVALID) {
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID,
                physical_block, phys_cell);
        return false;
    }

    if (paged_block_states[physical_block] == paged_block_state::RELEASED) {
        paged_release_violation += 1;
        if (required_by_active) {
            paged_active_release_violation += 1;
            set_paged_swap_error(
                    llama_paged_swap_error_reason::ACTIVE_READ_RELEASED_BLOCK,
                    physical_block, phys_cell);
            return false;
        } else {
            paged_padded_release_violation += 1;
        }
    } else if (paged_block_states[physical_block] == paged_block_state::PENDING_WRITE) {
        // PENDING_WRITE cells only have valid backing memory when they belong to
        // the current write transaction (paged_pending_write_cells flag set).
        // A stale / non-current-transaction PENDING_WRITE cell was RELEASED then
        // MADV_DONTNEED'd before the PENDING_WRITE transition and contains
        // garbage — it must NOT be read as if resident.  Fresh cells of the
        // current transaction may be read (their backing memory is allocated and
        // being written this transaction).
        const bool pending_fresh_cell =
            paged_write_transaction_owner != nullptr &&
            phys_cell < paged_pending_write_cells.size() &&
            paged_pending_write_cells[phys_cell];
        if (!pending_fresh_cell) {
            paged_block_ensure_pending_write_rejected += 1;
            paged_release_violation += 1;
            if (required_by_active) {
                paged_active_release_violation += 1;
                set_paged_swap_error(
                        llama_paged_swap_error_reason::ACTIVE_READ_RELEASED_BLOCK,
                        physical_block, phys_cell);
                return false;
            } else {
                paged_padded_release_violation += 1;
            }
        }
        // else: fresh current-transaction cell — readable, fall through to
        // the resident check below.
    } else if (paged_block_states[physical_block] == paged_block_state::SWAPPED) {
        paged_swap_read_swapped_hits += 1;
        if (paged_swap_in_block(
                    physical_block,
                    true,
                    llama_paged_swap_error_reason::SWAP_IN_IO_FAILURE) &&
                paged_block_states[physical_block] == paged_block_state::RESIDENT) {
            paged_swap_read_swap_in_calls += 1;
            // Stage 5E-1: resume swap-in just made this block resident again; resample KV
            // resident so after_resume reflects the most recent swap-in.
            if (paged_mincore_enabled) {
                paged_mincore_after_resume_resident_bytes = paged_sample_mincore();
            }
        } else {
            paged_swap_read_swap_in_failures += 1;
            return false;
        }
    }

    return !has_paged_swap_error();
}

void llama_kv_cache::paged_swap_out_block(
        uint32_t physical_block,
        bool do_madvise,
        bool retry_on_io_failure,
        uint64_t * relieved_bytes,
        bool * io_failure,
        int * io_errno) const {
    if (relieved_bytes) *relieved_bytes = 0;
    if (io_failure) *io_failure = false;
    if (io_errno) *io_errno = 0;
    if (!paged_resume_timing_enabled && !paged_resume_timing_step_enabled) {
        paged_swap_out_block_impl(
                physical_block, do_madvise, retry_on_io_failure,
                relieved_bytes, io_failure, io_errno);
        return;
    }

    const uint64_t start_us = llama_paged_timing_now_us();
    paged_swap_out_block_impl(
            physical_block, do_madvise, retry_on_io_failure,
            relieved_bytes, io_failure, io_errno);
    paged_timing_swap_out_us += llama_paged_timing_now_us() - start_us;
    paged_timing_swap_out_calls += 1;
}

void llama_kv_cache::paged_swap_out_block_impl(
        uint32_t physical_block,
        bool do_madvise,
        bool retry_on_io_failure,
        uint64_t * relieved_bytes,
        bool * io_failure,
        int * io_errno) const {
    if (!kv_paged_enabled || !paged_swap_enabled || !kv_swap_store || v_trans || n_stream != 1 ||
            paged_block_size == 0 || v_cells.empty()) {
        return;
    }
    if (physical_block >= paged_block_states.size() ||
            paged_block_states[physical_block] != paged_block_state::RESIDENT) {
        return;
    }
    const uint64_t validate_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;

    size_t total_size = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            total_size += layer.k_stream[0]->nb[1];
        }
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
            total_size += layer.v_stream[0]->nb[1];
        }
    }
    if (total_size == 0) {
        return;
    }

    auto & cells = v_cells[0];
    const uint32_t begin = physical_block * paged_block_size;
    const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
    if (end > paged_swap_offsets.size() || end > paged_swap_sizes.size()) {
        paged_swap_backend_failures += 1;
        return;
    }

    auto clear_block_entries = [&]() {
        for (uint32_t cell = begin; cell < end; ++cell) {
            paged_swap_offsets[cell] = 0;
            paged_swap_sizes[cell] = 0;
        }
    };

    auto block_state_name = [](paged_block_state state) {
        switch (state) {
            case paged_block_state::UNUSED:   return "UNUSED";
            case paged_block_state::RESIDENT: return "RESIDENT";
            case paged_block_state::RELEASED: return "RELEASED";
            case paged_block_state::SWAPPED:  return "SWAPPED";
            case paged_block_state::PENDING_WRITE: return "PENDING_WRITE";
            case paged_block_state::INVALID: return "INVALID";
        }
        return "UNKNOWN";
    };

    bool io_fault_candidate = false;
    if (do_madvise &&
            physical_block > 0 &&
            paged_test_io_fault_.scope == paged_test_io_fail_scope::SWAP_OUT &&
            paged_test_io_fault_.kind == paged_test_io_fail_kind::WRITE_ENOSPC_ONCE &&
            paged_test_io_fault_.target_seq >= 0 &&
            paged_test_io_fault_.target_block == physical_block &&
            (!paged_test_io_fault_.fail_once || !paged_test_io_fault_.consumed)) {
        for (uint32_t cell = begin; cell < end; ++cell) {
            if (cell < cells.size() && cells.seq_has(cell, paged_test_io_fault_.target_seq)) {
                io_fault_candidate = true;
                break;
            }
        }
    }

    const uint32_t cell_count = end - begin;
    if (cell_count == 0 || cell_count > std::numeric_limits<size_t>::max() / total_size) {
        paged_swap_backend_failures += 1;
        return;
    }
    const size_t block_size = (size_t) cell_count * total_size;
    if (paged_io_staging.size() < block_size) paged_io_staging.resize(block_size);
    bool io_fault_armed = false;
    uint64_t io_fault_attempt_id = 0;
    uint64_t io_fault_swap_out_before = 0;
    uint64_t io_fault_madvise_before = 0;
    uint64_t io_fault_syscalls_before = 0;
    paged_block_state io_fault_state_before = paged_block_states[physical_block];
    const bool retry_success_candidate =
        paged_test_io_fault_.scope == paged_test_io_fail_scope::SWAP_OUT &&
        paged_test_io_fault_.failed_block == physical_block;
    const uint64_t retry_fault_attempt_id = paged_test_io_fault_.failed_attempt_id;
    const uint64_t retry_swap_out_before = paged_blocks_swapped_out;
    const uint64_t retry_madvise_before = paged_swap_madvise_calls;
    const paged_block_state retry_state_before = paged_block_states[physical_block];
    if (paged_io_stats_enabled && block_size > paged_io_staging_buffer_bytes) {
        paged_io_staging_buffer_bytes = block_size;
    }
    if (paged_io_stats_enabled) {
        paged_io_block_out_validate_us += llama_paged_timing_now_us() - validate_start_us;
        paged_io_block_out_validate_calls += 1;
    }

    const uint64_t pack_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    for (uint32_t cell = begin; cell < end; ++cell) {
        if (cell >= cells.size()) {
            paged_swap_backend_failures += 1;
            clear_block_entries();
            return;
        }
    }
    for (uint32_t cell = begin; cell < end; ++cell) {
        size_t cursor = (size_t) (cell - begin) * total_size;
        for (const auto & layer : layers) {
            for (ggml_tensor * tensor : {
                    layer.k_stream.empty() ? nullptr : layer.k_stream[0],
                    (!layer.v || layer.v_stream.empty()) ? nullptr : layer.v_stream[0] }) {
                if (!tensor) continue;
                const size_t row_size = tensor->nb[1];
                ggml_backend_tensor_get(
                        tensor, paged_io_staging.data() + cursor, (size_t) cell * row_size, row_size);
                cursor += row_size;
            }
        }
        GGML_ASSERT(cursor == (size_t) (cell - begin + 1) * total_size);
    }
    if (paged_io_stats_enabled) {
        paged_io_block_out_pack_us += llama_paged_timing_now_us() - pack_start_us;
        paged_io_block_out_pack_calls += 1;
    }

    if (io_fault_candidate) {
        if (auto * store_file = dynamic_cast<llama_kv_backing_store_file *>(kv_swap_store.get())) {
            llama_kv_backing_store_faults faults = store_file->get_test_faults();
            faults.write_enospc_once = true;
            store_file->set_test_faults(faults);
            if (paged_test_io_fault_.fail_once) paged_test_io_fault_.consumed = true;
            paged_test_io_fault_.matching_attempts += 1;
            paged_test_io_fault_.trigger_count += 1;
            io_fault_attempt_id = paged_test_io_fault_.matching_attempts;
            io_fault_swap_out_before = paged_blocks_swapped_out;
            io_fault_madvise_before = paged_swap_madvise_calls;
            io_fault_syscalls_before = kv_swap_store->get_stats().syscall_attempts;
            io_fault_state_before = paged_block_states[physical_block];
            io_fault_armed = true;
        }
    }
    uint64_t block_offset = 0;
    const uint64_t write_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    const auto status = kv_swap_store->write_cells(
            0, begin, cell_count, paged_io_staging.data(), block_size, block_offset);
    if (paged_io_stats_enabled) {
        paged_io_block_out_write_us += llama_paged_timing_now_us() - write_start_us;
        paged_io_block_out_write_calls += 1;
    }
    if (status != llama_kv_backing_store_status::ok) {
            paged_swap_backend_failures += 1;
            const auto & failure_stats = kv_swap_store->get_stats();
            if (io_failure) *io_failure = true;
            if (io_errno) *io_errno = failure_stats.last_errno;
            if (io_fault_armed) {
                const auto & stats = kv_swap_store->get_stats();
                LLAMA_LOG_ERROR(
                        "KV_PAGED_IO_FAULT scope=swap_out kind=write_enospc_once target_seq=%d "
                        "physical_block=%u physical_cell=%u state_before=%s state_after=%s "
                        "swap_out_counter_before=%llu swap_out_counter_after=%llu "
                        "madvise_counter_before=%llu madvise_counter_after=%llu "
                        "backend_status=%s backend_errno=%d pwrite_attempts_before=%llu "
                        "pwrite_attempts_after=%llu trigger_count=%llu attempt_id=%llu\n",
                        (int) paged_test_io_fault_.target_seq,
                        physical_block,
                        begin,
                        block_state_name(io_fault_state_before),
                        block_state_name(paged_block_states[physical_block]),
                        (unsigned long long) io_fault_swap_out_before,
                        (unsigned long long) paged_blocks_swapped_out,
                        (unsigned long long) io_fault_madvise_before,
                        (unsigned long long) paged_swap_madvise_calls,
                        llama_kv_backing_store_status_name(stats.last_status),
                        stats.last_errno,
                        (unsigned long long) io_fault_syscalls_before,
                        (unsigned long long) stats.syscall_attempts,
                        (unsigned long long) paged_test_io_fault_.trigger_count,
                        (unsigned long long) io_fault_attempt_id);
                paged_test_io_fault_.failed_block = physical_block;
                paged_test_io_fault_.failed_attempt_id = io_fault_attempt_id;
            }
            clear_block_entries();
            if (io_fault_armed && do_madvise && retry_on_io_failure) {
                if (io_failure) *io_failure = false;
                if (io_errno) *io_errno = 0;
                paged_swap_out_block_impl(
                        physical_block, do_madvise, true,
                        relieved_bytes, io_failure, io_errno);
            }
            return;
    }

    const uint64_t metadata_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    // Publication is all-or-nothing: no cell exposes a slot until the full contiguous write
    // completed. This also makes a failed range retry overwrite from block_offset.
    for (uint32_t cell = begin; cell < end; ++cell) {
        paged_swap_offsets[cell] = block_offset + (uint64_t) (cell - begin) * total_size;
        paged_swap_sizes[cell] = total_size;
    }

    paged_block_states[physical_block] = paged_block_state::SWAPPED;
    paged_swap_out_calls += 1;
    paged_blocks_swapped_out += 1;
    paged_swap_bytes_out += block_size;
    if (paged_io_stats_enabled) {
        paged_io_block_out_metadata_us += llama_paged_timing_now_us() - metadata_start_us;
        paged_io_block_out_metadata_calls += 1;
    }

    if (!do_madvise) {
        if (paged_io_stats_enabled) {
            const uint64_t elapsed_us = llama_paged_timing_now_us() - validate_start_us;
            paged_io_swap_out_latency_us += elapsed_us;
            paged_io_swap_out_timed_calls += 1;
            paged_io_swap_out_latency_max_us = std::max(paged_io_swap_out_latency_max_us, elapsed_us);
        }
        return;
    }

    const uint64_t madvise_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    const uint64_t rss_before_kb = get_current_rss_kb();
    const uint64_t skip_no_full_before = paged_swap_madvise_skip_no_full_page;
    const uint64_t skip_neighbor_before = paged_swap_madvise_skip_neighbor;
    const uint64_t advised_bytes = paged_madvise_block(
            physical_block, nullptr,
            paged_swap_madvise_failures,
            paged_swap_madvise_skip_no_full_page,
            paged_swap_madvise_skip_neighbor);
    if (relieved_bytes) *relieved_bytes = advised_bytes;
    paged_swap_madvise_skipped +=
        (paged_swap_madvise_skip_no_full_page - skip_no_full_before) +
        (paged_swap_madvise_skip_neighbor - skip_neighbor_before);
    if (advised_bytes > 0) {
        const uint64_t rss_after_kb = get_current_rss_kb();
        const uint64_t rss_drop_kb = rss_before_kb > rss_after_kb ? rss_before_kb - rss_after_kb : 0;

        paged_swap_madvise_calls += 1;
        paged_swap_madvise_bytes += advised_bytes;
        paged_swap_rss_samples += 1;
        paged_swap_rss_before_last_kb = rss_before_kb;
        paged_swap_rss_after_last_kb = rss_after_kb;
        paged_swap_rss_drop_last_kb = rss_drop_kb;
        if (rss_drop_kb > paged_swap_rss_drop_max_kb) {
            paged_swap_rss_drop_max_kb = rss_drop_kb;
        }
        // cumulative window telemetry (Stage 5C-scale-B)
        if (!paged_swap_rss_before_first_set) {
            paged_swap_rss_before_first_kb = rss_before_kb;
            paged_swap_rss_before_first_set = true;
        }
        paged_swap_rss_drop_sum_kb += rss_drop_kb;
        paged_swap_rss_total_drop_kb =
            paged_swap_rss_before_first_kb > paged_swap_rss_after_last_kb
                ? paged_swap_rss_before_first_kb - paged_swap_rss_after_last_kb
                : 0;

        if (retry_success_candidate &&
                retry_state_before == paged_block_state::RESIDENT &&
                paged_block_states[physical_block] == paged_block_state::SWAPPED &&
                paged_blocks_swapped_out > retry_swap_out_before &&
                paged_swap_madvise_calls > retry_madvise_before) {
            paged_test_io_fault_.retry_success_count += 1;
            LLAMA_LOG_ERROR(
                    "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=swap_out kind=write_enospc_once "
                    "target_seq=%d physical_block=%u state_before=%s state_after=%s "
                    "swap_out_counter_before=%llu swap_out_counter_after=%llu "
                    "madvise_counter_before=%llu madvise_counter_after=%llu "
                    "fault_attempt_id=%llu retry_success_count=%llu\n",
                    (int) paged_test_io_fault_.target_seq,
                    physical_block,
                    block_state_name(retry_state_before),
                    block_state_name(paged_block_states[physical_block]),
                    (unsigned long long) retry_swap_out_before,
                    (unsigned long long) paged_blocks_swapped_out,
                    (unsigned long long) retry_madvise_before,
                    (unsigned long long) paged_swap_madvise_calls,
                    (unsigned long long) retry_fault_attempt_id,
                    (unsigned long long) paged_test_io_fault_.retry_success_count);
            paged_test_io_fault_.failed_block = PAGED_BLOCK_INVALID;
            paged_test_io_fault_.failed_attempt_id = 0;
        }
    }

    // Stage 7D-A: arm refault protection AFTER madvise. The block is now SWAPPED and its pages
    // have been advised away; mprotect(PROT_NONE) so any subsequent read of those virtual
    // addresses traps. Must come after madvise (a fault would re-fault the page resident, which is
    // exactly the event we want to catch); doing it before madvise would also work but would not
    // reflect the post-madvise state we are measuring. No-op unless refault tracing is enabled.
    paged_refault_protect_block(physical_block);
    if (paged_io_stats_enabled) {
        paged_io_block_out_madvise_us += llama_paged_timing_now_us() - madvise_start_us;
        paged_io_block_out_madvise_calls += 1;
        const uint64_t elapsed_us = llama_paged_timing_now_us() - validate_start_us;
        paged_io_swap_out_latency_us += elapsed_us;
        paged_io_swap_out_timed_calls += 1;
        paged_io_swap_out_latency_max_us = std::max(paged_io_swap_out_latency_max_us, elapsed_us);
    }
}

bool llama_kv_cache::paged_swap_in_block(
        uint32_t physical_block,
        bool fatal_on_failure,
        llama_paged_swap_error_reason failure_reason) const {
    bool io_fault_armed = false;
    bool io_fault_failed = false;
    uint64_t io_fault_attempt_id = 0;

    const auto fail = [&](uint32_t physical_cell, llama_kv_backing_store_status backend_status, int backend_errno = 0) {
        if (fatal_on_failure) {
            if (backend_status == llama_kv_backing_store_status::io_error ||
                    backend_status == llama_kv_backing_store_status::disabled) {
                if (kv_swap_store) {
                    const int stats_errno = kv_swap_store->get_stats().last_errno;
                    if (stats_errno != 0) {
                        backend_errno = stats_errno;
                    }
                }
            } else {
                backend_errno = 0;
            }
            set_paged_swap_error(
                    io_fault_failed ? llama_paged_swap_error_reason::PAGED_SWAP_IN_IO_ERROR : failure_reason,
                    physical_block, physical_cell, (int) backend_status, backend_errno);
        }

        return false;
    };

    if (!kv_paged_enabled || !paged_swap_enabled || !kv_swap_store || v_trans || n_stream != 1 ||
            paged_block_size == 0 || v_cells.empty()) {
        return fail(UINT32_MAX, llama_kv_backing_store_status::disabled);
    }
    if (physical_block >= paged_block_states.size() ||
            paged_block_states[physical_block] != paged_block_state::SWAPPED) {
        return fail(UINT32_MAX, llama_kv_backing_store_status::bad_slot);
    }
    const uint64_t validate_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;

    size_t total_size = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            total_size += layer.k_stream[0]->nb[1];
        }
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
            total_size += layer.v_stream[0]->nb[1];
        }
    }
    if (total_size == 0) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        return fail(UINT32_MAX, llama_kv_backing_store_status::bad_slot);
    }

    const auto & cells = v_cells[0];
    const uint32_t begin = physical_block * paged_block_size;
    const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
    const uint32_t cell_count = end - begin;
    if (cell_count == 0 || cell_count > std::numeric_limits<size_t>::max() / total_size) {
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        return fail(UINT32_MAX, llama_kv_backing_store_status::bad_slot);
    }
    const size_t block_size = (size_t) cell_count * total_size;
    if (paged_io_stats_enabled && block_size > paged_io_staging_buffer_bytes) {
        paged_io_staging_buffer_bytes = block_size;
    }

    auto block_state_name = [](paged_block_state state) {
        switch (state) {
            case paged_block_state::UNUSED:   return "UNUSED";
            case paged_block_state::RESIDENT: return "RESIDENT";
            case paged_block_state::RELEASED: return "RELEASED";
            case paged_block_state::SWAPPED:  return "SWAPPED";
            case paged_block_state::PENDING_WRITE: return "PENDING_WRITE";
            case paged_block_state::INVALID: return "INVALID";
        }
        return "UNKNOWN";
    };

    const bool io_fault_scope_matches =
        (fatal_on_failure && paged_test_io_fault_.scope == paged_test_io_fail_scope::ACTIVE_SWAP_IN) ||
        (!fatal_on_failure && paged_test_io_fault_.scope == paged_test_io_fail_scope::PREFETCH_SWAP_IN);
    bool io_fault_candidate = false;
    if (io_fault_scope_matches &&
            paged_test_io_fault_.kind == paged_test_io_fail_kind::READ_EOF_ONCE &&
            paged_test_io_fault_.target_seq >= 0 &&
            (paged_test_io_fault_.scope != paged_test_io_fail_scope::PREFETCH_SWAP_IN ||
                    paged_test_io_fault_.target_block == physical_block) &&
            (!paged_test_io_fault_.fail_once || !paged_test_io_fault_.consumed)) {
        for (uint32_t cell = begin; cell < end; ++cell) {
            if (cell < cells.size() && cells.seq_has(cell, paged_test_io_fault_.target_seq)) {
                io_fault_candidate = true;
                break;
            }
        }
    }
    uint64_t io_fault_swap_in_before = paged_blocks_swapped_in;
    uint64_t io_fault_syscalls_before = kv_swap_store->get_stats().syscall_attempts;
    paged_block_state io_fault_state_before = paged_block_states[physical_block];
    const bool retry_success_candidate =
        paged_test_io_fault_.scope == paged_test_io_fail_scope::ACTIVE_SWAP_IN &&
        paged_test_io_fault_.failed_block == physical_block;
    const uint64_t retry_fault_attempt_id = paged_test_io_fault_.failed_attempt_id;
    const uint64_t retry_swap_in_before = paged_blocks_swapped_in;
    const paged_block_state retry_state_before = paged_block_states[physical_block];

    const paged_test_swapin_fail_scope call_scope = fatal_on_failure
        ? paged_test_swapin_fail_scope::ACTIVE
        : paged_test_swapin_fail_scope::PREFETCH;
    const bool test_fault_attempt = paged_test_swapin_fault_.scope == call_scope;
    uint64_t test_fault_attempt_id = 0;
    uint64_t test_fault_swap_in_calls_before = 0;
    const char * test_fault_scope_name = fatal_on_failure ? "active" : "prefetch";
    if (test_fault_attempt) {
        paged_test_swapin_fault_.matching_attempts += 1;
        test_fault_attempt_id = paged_test_swapin_fault_.matching_attempts;
        test_fault_swap_in_calls_before = paged_swap_in_calls;
        LLAMA_LOG_INFO(
                "TEST FAULT SWAPIN ATTEMPT attempt_id=%llu scope=%s physical_block=%u "
                "block_begin=%u block_end=%u fail_after_cells=%llu fail_once=%d consumed=%d "
                "block_state=%d paged_swap_in_calls_before=%llu\n",
                (unsigned long long) test_fault_attempt_id,
                test_fault_scope_name,
                physical_block,
                begin,
                end,
                (unsigned long long) paged_test_swapin_fault_.fail_after_cells,
                paged_test_swapin_fault_.fail_once ? 1 : 0,
                paged_test_swapin_fault_.consumed ? 1 : 0,
                (int) paged_block_states[physical_block],
                (unsigned long long) test_fault_swap_in_calls_before);
    }

    // Validate every published slot before issuing I/O or touching a tensor. A corrupt later
    // cell must not leave an earlier cell restored.
    uint64_t block_offset = 0;
    for (uint32_t cell = begin; cell < end; ++cell) {
        if (cell >= cells.size() || cell >= paged_swap_offsets.size() || cell >= paged_swap_sizes.size() ||
                paged_swap_sizes[cell] != total_size) {
            paged_swap_in_fail_bad_size += 1;
            paged_swap_backend_failures += 1;
            return fail(cell, llama_kv_backing_store_status::bad_slot);
        }
        const uint64_t expected = (uint64_t) cell * total_size;
        if (paged_swap_offsets[cell] != expected || (cell > begin && paged_swap_offsets[cell] != block_offset +
                (uint64_t) (cell - begin) * total_size)) {
            paged_swap_in_fail_no_offset += 1;
            paged_swap_backend_failures += 1;
            return fail(cell, llama_kv_backing_store_status::bad_slot);
        }
        if (cell == begin) block_offset = paged_swap_offsets[cell];
    }
    if (paged_io_staging.size() < block_size) paged_io_staging.resize(block_size);
    if (paged_io_stats_enabled) {
        paged_io_block_in_validate_us += llama_paged_timing_now_us() - validate_start_us;
        paged_io_block_in_validate_calls += 1;
    }

    if (test_fault_attempt &&
            paged_test_swapin_fault_.successful_cells >= paged_test_swapin_fault_.fail_after_cells &&
            (!paged_test_swapin_fault_.fail_once || !paged_test_swapin_fault_.consumed)) {
        if (paged_test_swapin_fault_.fail_once) paged_test_swapin_fault_.consumed = true;
        paged_test_swapin_fault_.trigger_count += 1;
        if (fatal_on_failure) paged_test_swapin_fault_.active_trigger_count += 1;
        else paged_test_swapin_fault_.prefetch_trigger_count += 1;
        uint64_t metadata_present_cells = 0;
        for (uint32_t metadata_cell = begin; metadata_cell < end; ++metadata_cell) {
            if (paged_swap_sizes[metadata_cell] != 0) metadata_present_cells += 1;
        }
        LLAMA_LOG_ERROR(
                "TEST FAULT INJECTION attempt_id=%llu scope=%s physical_block=%u physical_cell=%u "
                "successful_cells_before_failure=0 backend_status=io_error backend_errno=EIO(%d) "
                "failure_reason=%s block_state=%d metadata_present_cells=%llu "
                "paged_swap_in_calls_before=%llu\n",
                (unsigned long long) test_fault_attempt_id, test_fault_scope_name, physical_block, begin, EIO,
                llama_paged_swap_error_reason_name(failure_reason), (int) paged_block_states[physical_block],
                (unsigned long long) metadata_present_cells,
                (unsigned long long) test_fault_swap_in_calls_before);
        paged_swap_in_fail_read_cell += 1;
        paged_swap_backend_failures += 1;
        return fail(begin, llama_kv_backing_store_status::io_error, EIO);
    }

    if (io_fault_candidate) {
        if (auto * store_file = dynamic_cast<llama_kv_backing_store_file *>(kv_swap_store.get())) {
            llama_kv_backing_store_faults faults = store_file->get_test_faults();
            faults.read_eof_once = true;
            store_file->set_test_faults(faults);
            if (paged_test_io_fault_.fail_once) paged_test_io_fault_.consumed = true;
            paged_test_io_fault_.matching_attempts += 1;
            paged_test_io_fault_.trigger_count += 1;
            io_fault_attempt_id = paged_test_io_fault_.matching_attempts;
            io_fault_swap_in_before = paged_blocks_swapped_in;
            io_fault_syscalls_before = kv_swap_store->get_stats().syscall_attempts;
            io_fault_state_before = paged_block_states[physical_block];
            io_fault_armed = true;
        }
    }

    const uint64_t read_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    const auto status = kv_swap_store->read_cells(
            0, begin, cell_count, block_offset, paged_io_staging.data(), block_size);
    if (paged_io_stats_enabled) {
        paged_io_block_in_read_us += llama_paged_timing_now_us() - read_start_us;
        paged_io_block_in_read_calls += 1;
    }
    if (status != llama_kv_backing_store_status::ok) {
            paged_swap_in_fail_read_cell += 1;
            paged_swap_backend_failures += 1;
            int backend_errno = 0;
            if (status == llama_kv_backing_store_status::io_error ||
                    status == llama_kv_backing_store_status::disabled) {
                backend_errno = kv_swap_store->get_stats().last_errno;
            }
            if (io_fault_armed) {
                io_fault_failed = true;
                uint64_t metadata_present_cells = 0;
                for (uint32_t metadata_cell = begin; metadata_cell < end; ++metadata_cell) {
                    if (metadata_cell < paged_swap_sizes.size() && paged_swap_sizes[metadata_cell] != 0) {
                        metadata_present_cells += 1;
                    }
                }
                const auto & stats = kv_swap_store->get_stats();
                const char * io_fault_scope_name =
                    paged_test_io_fault_.scope == paged_test_io_fail_scope::PREFETCH_SWAP_IN
                    ? "prefetch_swap_in" : "active_swap_in";
                LLAMA_LOG_ERROR(
                        "KV_PAGED_IO_FAULT scope=%s kind=read_eof_once target_seq=%d "
                        "physical_block=%u physical_cell=%u state_before=%s state_after=%s "
                        "swap_in_counter_before=%llu swap_in_counter_after=%llu "
                        "backend_status=%s backend_errno=%d pread_attempts_before=%llu "
                        "pread_attempts_after=%llu metadata_present_cells=%llu "
                        "failure_reason=%s trigger_count=%llu attempt_id=%llu\n",
                        io_fault_scope_name,
                        (int) paged_test_io_fault_.target_seq,
                        physical_block,
                        begin,
                        block_state_name(io_fault_state_before),
                        block_state_name(paged_block_states[physical_block]),
                        (unsigned long long) io_fault_swap_in_before,
                        (unsigned long long) paged_blocks_swapped_in,
                        llama_kv_backing_store_status_name(stats.last_status),
                        backend_errno,
                        (unsigned long long) io_fault_syscalls_before,
                        (unsigned long long) stats.syscall_attempts,
                        (unsigned long long) metadata_present_cells,
                        llama_paged_swap_error_reason_name(llama_paged_swap_error_reason::PAGED_SWAP_IN_IO_ERROR),
                        (unsigned long long) paged_test_io_fault_.trigger_count,
                        (unsigned long long) io_fault_attempt_id);
                paged_test_io_fault_.failed_block = physical_block;
                paged_test_io_fault_.failed_attempt_id = io_fault_attempt_id;
            }
            return fail(begin, status, backend_errno);
    }

    // Keep SWAPPED pages protected until the complete cell-major file image has been read into
    // staging. A terminal read failure above therefore leaves refault protection intact.
    paged_refault_unprotect_block(physical_block);
    const uint64_t unpack_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    for (uint32_t cell = begin; cell < end; ++cell) {
        size_t cursor = (size_t) (cell - begin) * total_size;
        for (const auto & layer : layers) {
            for (ggml_tensor * tensor : {
                    layer.k_stream.empty() ? nullptr : layer.k_stream[0],
                    (!layer.v || layer.v_stream.empty()) ? nullptr : layer.v_stream[0] }) {
                if (!tensor) continue;
                const size_t row_size = tensor->nb[1];
                ggml_backend_tensor_set(
                        tensor, paged_io_staging.data() + cursor, (size_t) cell * row_size, row_size);
                cursor += row_size;
            }
        }
        GGML_ASSERT(cursor == (size_t) (cell - begin + 1) * total_size);
    }
    if (paged_io_stats_enabled) {
        paged_io_block_in_unpack_us += llama_paged_timing_now_us() - unpack_start_us;
        paged_io_block_in_unpack_calls += 1;
    }

    const uint64_t commit_start_us = paged_io_stats_enabled ? llama_paged_timing_now_us() : 0;
    paged_block_states[physical_block] = paged_block_state::RESIDENT;
    paged_swap_in_calls += 1;
    paged_blocks_swapped_in += 1;
    paged_swap_bytes_in += block_size;
    paged_swap_in_last_block = physical_block;
    if (paged_io_stats_enabled) {
        paged_io_block_in_commit_us += llama_paged_timing_now_us() - commit_start_us;
        paged_io_block_in_commit_calls += 1;
    }
    if (paged_io_stats_enabled) {
        const uint64_t elapsed_us = llama_paged_timing_now_us() - validate_start_us;
        paged_io_swap_in_latency_us += elapsed_us;
        paged_io_swap_in_timed_calls += 1;
        if (elapsed_us > paged_io_swap_in_latency_max_us) {
            paged_io_swap_in_latency_max_us = elapsed_us;
        }
    }
    if (retry_success_candidate) {
        paged_test_io_fault_.retry_success_count += 1;
        LLAMA_LOG_ERROR(
                "KV_PAGED_IO_FAULT_RETRY_SUCCESS scope=active_swap_in kind=read_eof_once "
                "target_seq=%d physical_block=%u state_before=%s state_after=%s "
                "swap_in_counter_before=%llu swap_in_counter_after=%llu "
                "fault_attempt_id=%llu retry_success_count=%llu\n",
                (int) paged_test_io_fault_.target_seq,
                physical_block,
                block_state_name(retry_state_before),
                block_state_name(paged_block_states[physical_block]),
                (unsigned long long) retry_swap_in_before,
                (unsigned long long) paged_blocks_swapped_in,
                (unsigned long long) retry_fault_attempt_id,
                (unsigned long long) paged_test_io_fault_.retry_success_count);
        paged_test_io_fault_.failed_block = PAGED_BLOCK_INVALID;
        paged_test_io_fault_.failed_attempt_id = 0;
    }
    if (test_fault_attempt) {
        paged_test_swapin_fault_.successful_cells += cell_count;
        LLAMA_LOG_INFO(
                "TEST FAULT SWAPIN COMPLETE attempt_id=%llu scope=%s physical_block=%u "
                "restored_cells=%llu block_state=%d paged_swap_in_calls_before=%llu "
                "paged_swap_in_calls_after=%llu\n",
                (unsigned long long) test_fault_attempt_id,
                test_fault_scope_name,
                physical_block,
                (unsigned long long) cell_count,
                (int) paged_block_states[physical_block],
                (unsigned long long) test_fault_swap_in_calls_before,
                (unsigned long long) paged_swap_in_calls);
    }
    return true;
}

bool llama_kv_cache::paged_compute_block_page_range(
        const ggml_tensor * tensor,
        uint32_t physical_block,
        uint64_t row_size,
        uintptr_t page_size,
        paged_block_page_range & range) const {
    range = {};
    if (!tensor || !tensor->data || row_size == 0 || page_size == 0 ||
            physical_block >= paged_n_blocks || paged_block_size == 0) {
        return false;
    }

    const uint64_t lo_cell = (uint64_t) physical_block * paged_block_size;
    const uint64_t hi_cell = std::min<uint64_t>(lo_cell + paged_block_size, paged_kv_size);
    const uintptr_t lo_a = (uintptr_t) tensor->data + (uintptr_t) lo_cell * row_size;
    const uintptr_t hi_a = (uintptr_t) tensor->data + (uintptr_t) hi_cell * row_size;
    if (hi_a <= lo_a) {
        return false;
    }

    const uintptr_t pg = page_size;
    const uintptr_t a_start = (lo_a + pg - 1) & ~(pg - 1);
    const uintptr_t a_end   = hi_a & ~(pg - 1);
    if (a_end <= a_start) {
        return false;
    }

    range.byte_begin = lo_a;
    range.byte_end = hi_a;
    range.page_begin = a_start;
    range.page_end = a_end;
    return true;
}

uint64_t llama_kv_cache::paged_madvise_block(
        uint32_t physical_block,
        const std::vector<uint8_t> * active,
        uint64_t & failures,
        uint64_t & skipped,
        uint64_t & skip_live,
        std::vector<paged_release_range> * advised_ranges) const {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    if (!kv_paged_enabled || v_trans || n_stream != 1 || paged_block_size == 0 ||
            physical_block >= paged_n_blocks || physical_block >= paged_block_states.size()) {
        return 0;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return 0;
    }
    const uint64_t pg = (uint64_t) page;

    auto protected_neighbor = [&](uint32_t block) {
        if (block >= paged_n_blocks) {
            return false;
        }
        if (active) {
            return block < active->size() && (*active)[block];
        }

        if (block >= paged_block_states.size()) {
            return true;
        }
        const paged_block_state state = paged_block_states[block];
        return state != paged_block_state::SWAPPED && state != paged_block_state::RELEASED;
    };

    uint64_t advised_bytes = 0;
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        auto advise_tensor = [&](ggml_tensor * t, uint64_t row) {
            if (!t || row == 0 || !t->data) {
                return;
            }

            paged_block_page_range page_range;
            if (!paged_compute_block_page_range(t, physical_block, row, pg, page_range)) {
                skipped += 1;
                return;
            }

            if ((page_range.byte_begin & (uintptr_t) (pg - 1)) != 0 && physical_block > 0 &&
                    protected_neighbor(physical_block - 1)) {
                skip_live += 1;
            }
            if ((page_range.byte_end & (uintptr_t) (pg - 1)) != 0 && physical_block + 1 < paged_n_blocks &&
                    protected_neighbor(physical_block + 1)) {
                skip_live += 1;
            }

            const size_t len = (size_t) (page_range.page_end - page_range.page_begin);
            paged_release_range range { (void *) page_range.page_begin, len, 0, physical_block };
            if (advised_ranges) {
                range.before_resident = paged_sample_release_ranges({ range });
            }
            const int rc = madvise((void *) page_range.page_begin, len, MADV_DONTNEED);
            if (rc != 0) {
                failures += 1;
            } else {
                advised_bytes += len;
                if (advised_ranges) {
                    advised_ranges->push_back(range);
                }
            }
        };

        ggml_tensor * k = layer.k_stream.empty() ? nullptr : layer.k_stream[0];
        ggml_tensor * v = layer.v_stream.empty() ? nullptr : layer.v_stream[0];
        advise_tensor(k, k ? ggml_row_size(k->type, hparams.n_embd_k_gqa(il)) : 0);
        advise_tensor(v, v ? ggml_row_size(v->type, hparams.n_embd_v_gqa(il)) : 0);
    }

    return advised_bytes;
#else
    (void) physical_block;
    (void) active;
    (void) failures;
    (void) skipped;
    (void) skip_live;
    (void) advised_ranges;
    return 0;
#endif
}

uint64_t llama_kv_cache::paged_sample_mincore() const {
#if defined(__linux__)
    if (!paged_mincore_enabled) {
        return 0;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return 0;
    }
    const uintptr_t pg = (uintptr_t) page;

    uint64_t total_bytes      = 0;
    uint64_t resident_bytes   = 0;
    uint64_t total_pages      = 0;
    uint64_t resident_pages   = 0;
    uint64_t k_total_bytes    = 0;
    uint64_t k_resident_bytes = 0;
    uint64_t v_total_bytes    = 0;
    uint64_t v_resident_bytes = 0;

    // Read-only residency probe for one tensor's host interval, page-aligned with the same
    // round-up-start / round-down-end rule as paged_madvise_block so the sampled region is a
    // page-aligned subset of what madvise could advise. Accumulates into the per-kind totals.
    auto sample_tensor = [&](ggml_tensor * t, uint64_t & kind_total, uint64_t & kind_resident) {
        if (!t || !t->data) {
            return;
        }
        const uintptr_t lo_a = (uintptr_t) t->data;
        const uintptr_t hi_a = lo_a + (uintptr_t) ggml_nbytes(t);
        const uintptr_t a_start = (lo_a + pg - 1) & ~(pg - 1);
        const uintptr_t a_end   = hi_a & ~(pg - 1);
        if (a_end <= a_start) {
            return;
        }

        const size_t len = (size_t) (a_end - a_start);
        const uint64_t n_pages = (uint64_t) (len / pg);
        std::vector<unsigned char> vec(n_pages);
        if (mincore((void *) a_start, len, vec.data()) != 0) {
            paged_mincore_failures += 1;
            return;
        }

        uint64_t res = 0;
        for (uint64_t i = 0; i < n_pages; ++i) {
            if (vec[i] & 1u) {
                res += 1;
            }
        }

        total_pages    += n_pages;
        resident_pages += res;
        const uint64_t tb = n_pages * (uint64_t) pg;
        const uint64_t rb = res * (uint64_t) pg;
        total_bytes    += tb;
        resident_bytes += rb;
        kind_total     += tb;
        kind_resident  += rb;
    };

    for (const auto & layer : layers) {
        ggml_tensor * k = layer.k_stream.empty() ? nullptr : layer.k_stream[0];
        ggml_tensor * v = layer.v_stream.empty() ? nullptr : layer.v_stream[0];
        sample_tensor(k, k_total_bytes, k_resident_bytes);
        sample_tensor(v, v_total_bytes, v_resident_bytes);
    }

    // Stage 7C-C: per-block residency of currently-SWAPPED blocks. For each block whose state is
    // SWAPPED, probe its K/V page range (same round-up-start / round-down-end alignment as
    // paged_madvise_block, applied per layer/tensor/block) and tally resident vs non-resident
    // pages. A block counts as "resident" if any of its probed pages is still resident — that is
    // the signal that an already-swapped block was re-touched back into core after swap-out.
    uint64_t swp_block_count    = 0;
    uint64_t swp_total_bytes    = 0;
    uint64_t swp_resident_bytes = 0;
    uint64_t swp_resident_blk   = 0;
    if (paged_block_size != 0 && !paged_block_states.empty()) {
        for (uint32_t block = 0; block < paged_block_states.size(); ++block) {
            if (paged_block_states[block] != paged_block_state::SWAPPED) {
                continue;
            }
            swp_block_count += 1;
            uint64_t blk_resident = 0;

            auto probe_block_tensor = [&](ggml_tensor * t) {
                if (!t || !t->data) {
                    return;
                }
                const uint64_t row    = (uint64_t) t->nb[1];
                const uint64_t lo_cell = (uint64_t) block * paged_block_size;
                const uint64_t hi_cell = std::min<uint64_t>(lo_cell + paged_block_size, paged_kv_size);
                const uintptr_t lo_a = (uintptr_t) t->data + (uintptr_t) lo_cell * row;
                const uintptr_t hi_a = (uintptr_t) t->data + (uintptr_t) hi_cell * row;
                if (hi_a <= lo_a) {
                    return;
                }
                const uintptr_t a_start = (lo_a + pg - 1) & ~(pg - 1);
                const uintptr_t a_end   = hi_a & ~(pg - 1);
                if (a_end <= a_start) {
                    return;
                }
                const size_t len = (size_t) (a_end - a_start);
                const uint64_t n_pages = (uint64_t) (len / pg);
                std::vector<unsigned char> vec(n_pages);
                if (mincore((void *) a_start, len, vec.data()) != 0) {
                    paged_mincore_failures += 1;
                    return;
                }
                uint64_t res = 0;
                for (uint64_t i = 0; i < n_pages; ++i) {
                    if (vec[i] & 1u) {
                        res += 1;
                    }
                }
                swp_total_bytes    += n_pages * (uint64_t) pg;
                swp_resident_bytes += res * (uint64_t) pg;
                blk_resident       += res;
            };

            for (const auto & layer : layers) {
                ggml_tensor * k = layer.k_stream.empty() ? nullptr : layer.k_stream[0];
                ggml_tensor * v = layer.v_stream.empty() ? nullptr : layer.v_stream[0];
                probe_block_tensor(k);
                probe_block_tensor(v);
            }
            if (blk_resident > 0) {
                swp_resident_blk += 1;
            }
        }
    }

    paged_mincore_sample_calls += 1;
    paged_mincore_total_bytes      = total_bytes;
    paged_mincore_resident_bytes   = resident_bytes;
    paged_mincore_total_pages      = total_pages;
    paged_mincore_resident_pages   = resident_pages;
    paged_mincore_k_total_bytes    = k_total_bytes;
    paged_mincore_k_resident_bytes = k_resident_bytes;
    paged_mincore_v_total_bytes    = v_total_bytes;
    paged_mincore_v_resident_bytes = v_resident_bytes;

    paged_mincore_swapped_block_count       = swp_block_count;
    paged_mincore_swapped_total_bytes       = swp_total_bytes;
    paged_mincore_swapped_resident_bytes    = swp_resident_bytes;
    paged_mincore_swapped_nonresident_bytes = swp_total_bytes > swp_resident_bytes
        ? swp_total_bytes - swp_resident_bytes : 0;
    paged_mincore_swapped_resident_blocks    = swp_resident_blk;
    paged_mincore_swapped_nonresident_blocks = swp_block_count > swp_resident_blk
        ? swp_block_count - swp_resident_blk : 0;

    return resident_bytes;
#else
    return 0;
#endif
}

uint64_t llama_kv_cache::paged_sample_release_ranges(
        const std::vector<paged_release_range> & ranges,
        uint64_t * total_bytes_out) const {
#if defined(__linux__)
    if (!paged_mincore_enabled) {
        if (total_bytes_out) {
            *total_bytes_out = 0;
        }
        return 0;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        if (total_bytes_out) {
            *total_bytes_out = 0;
        }
        return 0;
    }

    const uintptr_t pg = (uintptr_t) page;
    uint64_t total_bytes = 0;
    uint64_t resident_bytes = 0;
    for (const auto & range : ranges) {
        if (!range.addr || range.len == 0 ||
                ((uintptr_t) range.addr & (pg - 1)) != 0 || range.len % pg != 0) {
            continue;
        }
        std::vector<unsigned char> vec(range.len / pg);
        if (mincore(range.addr, range.len, vec.data()) != 0) {
            paged_mincore_failures += 1;
            continue;
        }
        total_bytes += range.len;
        for (unsigned char value : vec) {
            if (value & 1u) {
                resident_bytes += pg;
            }
        }
    }
    if (total_bytes_out) {
        *total_bytes_out = total_bytes;
    }
    return resident_bytes;
#else
    (void) ranges;
    if (total_bytes_out) {
        *total_bytes_out = 0;
    }
    return 0;
#endif
}

void llama_kv_cache::paged_verify_fresh_writes(const slot_info & sinfo) const {
    if (!paged_release_fresh_verify_enabled || sinfo.n_stream() != 1 || v_trans) {
        return;
    }
    auto hash_bytes = [&](const uint8_t * data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            paged_release_fresh_verify_hash ^= data[i];
            paged_release_fresh_verify_hash *= 1099511628211ULL;
        }
        paged_release_fresh_verify_bytes += size;
    };
    for (uint32_t logical : sinfo.idxs[0]) {
        const uint32_t phys = paged_resolve(logical);
        if (phys == PAGED_BLOCK_INVALID) {
            continue;
        }
        for (const auto & layer : layers) {
            for (ggml_tensor * tensor : {
                    layer.k_stream.empty() ? nullptr : layer.k_stream[0],
                    layer.v_stream.empty() ? nullptr : layer.v_stream[0] }) {
                if (!tensor) {
                    continue;
                }
                const size_t row_size = tensor->nb[1];
                std::vector<uint8_t> row(row_size);
                ggml_backend_tensor_get(tensor, row.data(), (size_t) phys * row_size, row_size);
                hash_bytes(row.data(), row.size());
            }
        }
    }
}

void llama_kv_cache::paged_invalidate_write_context(
        llama_paged_swap_error_reason reason,
        uint32_t physical_block,
        uint32_t physical_cell) {
    paged_write_context_invalid = true;
    paged_write_context_invalid_cause = reason;
    paged_swap_error = {};
    set_paged_swap_error(reason, physical_block, physical_cell);
}

bool llama_kv_cache::paged_begin_write_transaction(const llama_kv_cache_context * owner) {
    if (!kv_paged_enabled) {
        return true;
    }
    if (paged_write_context_invalid) {
        paged_swap_error = {};
        set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID,
                PAGED_BLOCK_INVALID,
                PAGED_BLOCK_INVALID,
                static_cast<int>(paged_write_context_invalid_cause));
        return false;
    }
    if (paged_write_transaction_owner && paged_write_transaction_owner != owner) {
        paged_invalidate_write_context(
                llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID);
        return false;
    }
    paged_write_transaction_owner = owner;
    return true;
}

bool llama_kv_cache::paged_finish_write_transaction(
        llama_paged_kv_write_action action,
        const slot_info & sinfo,
        const llama_kv_cache_context * owner) {
    if (!kv_paged_enabled) {
        return true;
    }

    if (paged_write_transaction_owner != owner) {
        paged_invalidate_write_context(
                llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID);
        return false;
    }

    bool finished = true;
    if (action == llama_paged_kv_write_action::COMMIT) {
        if (paged_mincore_enabled && !paged_release_post_ranges.empty()) {
            uint64_t total = 0;
            paged_release_mincore_post_graph_last = paged_sample_release_ranges(
                    paged_release_post_ranges, &total);
            paged_release_mincore_total_last = total;
        }
        paged_verify_fresh_writes(sinfo);
        for (uint32_t block : paged_pending_write_blocks) {
            if (block < paged_block_states.size() &&
                    paged_block_states[block] == paged_block_state::PENDING_WRITE) {
                paged_block_states[block] = paged_block_state::RESIDENT;
                paged_release_write_commits += 1;
                if (paged_release_test_reuse) {
                    paged_release_test_reuse_commits += 1;
                }
            }
        }
    } else if (action == llama_paged_kv_write_action::ROLLBACK_PRE_COMPUTE) {
        std::vector<uint8_t> rollback_protected(paged_n_blocks, 0);
        for (uint32_t block = 0; block < paged_n_blocks; ++block) {
            rollback_protected[block] =
                paged_block_states[block] != paged_block_state::PENDING_WRITE &&
                paged_block_states[block] != paged_block_state::RELEASED &&
                paged_block_states[block] != paged_block_state::SWAPPED;
        }
        for (uint32_t block : paged_pending_write_blocks) {
            if (block >= paged_block_states.size() ||
                    paged_block_states[block] != paged_block_state::PENDING_WRITE) {
                continue;
            }

            uint64_t failures = 0;
            uint64_t skipped = 0;
            uint64_t skip_live = 0;
            if (paged_release_bounded_test_fail_rollback_madvise) {
                paged_release_bounded_test_fail_rollback_madvise = false;
                paged_release_bounded_test_fail_rollback_madvise_triggers += 1;
                failures = 1;
            } else {
                (void) paged_madvise_block(
                        block, &rollback_protected, failures, skipped, skip_live);
            }

            if (failures != 0 || skipped != 0 || skip_live != 0) {
                paged_block_states[block] = paged_block_state::INVALID;
                if (block < paged_block_used.size() && !paged_block_used[block]) {
                    paged_block_used[block] = 1;
                    paged_blocks_in_use += 1;
                }
                paged_free_list.erase(
                        std::remove(paged_free_list.begin(), paged_free_list.end(), block),
                        paged_free_list.end());
                paged_invalidate_write_context(
                        llama_paged_swap_error_reason::PAGED_WRITE_ROLLBACK_DISCARD_FAILURE,
                        block,
                        PAGED_BLOCK_INVALID);
                finished = false;
                continue;
            }

            paged_block_states[block] = paged_block_state::RELEASED;
            if (block < paged_block_used.size() && paged_block_used[block]) {
                paged_block_used[block] = 0;
                if (paged_blocks_in_use > 0) {
                    paged_blocks_in_use -= 1;
                }
            }
            if (std::find(paged_free_list.begin(), paged_free_list.end(), block) ==
                    paged_free_list.end()) {
                paged_free_list.push_back(block);
            }
            paged_release_write_rollbacks += 1;
        }
    } else {
        std::vector<uint32_t> write_blocks;
        for (uint32_t stream = 0; stream < sinfo.n_stream(); ++stream) {
            for (uint32_t logical_cell : sinfo.idxs[stream]) {
                const uint32_t logical_block = logical_cell / paged_block_size;
                if (logical_block >= paged_block_table.size()) {
                    continue;
                }
                const uint32_t block = paged_block_table[logical_block];
                if (block < paged_block_states.size()) {
                    write_blocks.push_back(block);
                }
            }
        }
        std::sort(write_blocks.begin(), write_blocks.end());
        write_blocks.erase(std::unique(write_blocks.begin(), write_blocks.end()), write_blocks.end());
        for (uint32_t block : write_blocks) {
            paged_block_states[block] = paged_block_state::INVALID;
            if (block < paged_block_used.size() && !paged_block_used[block]) {
                paged_block_used[block] = 1;
                paged_blocks_in_use += 1;
            }
            paged_free_list.erase(
                    std::remove(paged_free_list.begin(), paged_free_list.end(), block),
                    paged_free_list.end());
        }
        paged_invalidate_write_context(
                llama_paged_swap_error_reason::PAGED_WRITE_COMPUTE_FAILURE,
                write_blocks.empty() ? PAGED_BLOCK_INVALID : write_blocks.front(),
                PAGED_BLOCK_INVALID);
        finished = false;
    }

    for (uint32_t block : paged_pending_write_blocks) {
        const uint32_t begin = block * paged_block_size;
        const uint32_t end = std::min<uint32_t>(
                begin + paged_block_size, paged_pending_write_cells.size());
        std::fill(
                paged_pending_write_cells.begin() + begin,
                paged_pending_write_cells.begin() + end,
                0);
    }
    paged_pending_write_blocks.clear();
    paged_write_transaction_owner = nullptr;

    return finished;
}

// ===========================================================================================
// Stage 7D-A: debug-only SWAPPED-page refault tracing.
//
// Goal: find out which read path touches an already-swapped-out + madvise'd KV page during graph
// compute (the "refault" that keeps whole-KV resident drop pinned at ~63.75 MiB). When enabled we
// mprotect(PROT_NONE) the exact page-aligned K/V interior we just advised away; any later access
// traps into the SIGSEGV handler below, which logs the fault site, restores the page, and returns
// so the faulting instruction retries. This is purely diagnostic -- it never changes swap/madvise/
// row_idx semantics and is a no-op unless LLAMA_KV_PAGED_REFAULT_TRACE=1.
//
// Signal-handler safety: the handler only does async-signal-safe work -- it scans a fixed,
// pre-built trap table (never reallocated while armed), calls mprotect(2), bumps lock-free atomic
// counters, and emits one line via write(2) using a hand-rolled integer formatter (no malloc, no
// locks, no stdio). Optional backtrace uses backtrace()+backtrace_symbols_fd() (the fd variant is
// the async-signal-safe one). All aggregate counters live in file-scope atomics and are copied
// into the instance for the stats line by paged_refault_drain().
// ===========================================================================================
#if defined(LLAMA_KV_REFAULT_TRACE_SUPPORTED)
namespace {

struct refault_trap {
    uintptr_t lo;        // page-aligned start of this block's K or V interval in one layer
    uintptr_t hi;        // page-aligned end (exclusive)
    uint32_t  block;     // owning physical block id
    uint32_t  layer_il;  // KV layer id
    uint8_t   is_v;      // 0 = K tensor, 1 = V tensor
    std::atomic<uint8_t> armed; // 1 => currently PROT_NONE and watched
};

// Fixed table, allocated once (covers every block x layer x {K,V}) and published before arming.
refault_trap *        g_refault_traps   = nullptr;
std::atomic<size_t>   g_refault_ntraps  {0};
std::atomic<bool>     g_refault_armed   {false};
std::atomic<long>     g_refault_pagesize{0};
std::atomic<int>      g_refault_backtrace{0};
std::atomic<int>      g_refault_once    {1};
std::atomic<uint64_t> g_refault_max     {64};
std::atomic<uint64_t> g_refault_step    {0};

std::atomic<uint64_t> g_refault_fault_count{0};
std::atomic<uint64_t> g_refault_fault_k   {0};
std::atomic<uint64_t> g_refault_fault_v   {0};
std::atomic<uint64_t> g_refault_unmapped  {0};

struct sigaction g_refault_old_sa;
std::atomic<bool> g_refault_installed{false};

// async-signal-safe unsigned -> decimal, appended at p, returns new end.
char * refault_u64(char * p, uint64_t v) {
    char tmp[20];
    int n = 0;
    if (v == 0) {
        tmp[n++] = '0';
    }
    while (v > 0) {
        tmp[n++] = (char) ('0' + (v % 10));
        v /= 10;
    }
    while (n > 0) {
        *p++ = tmp[--n];
    }
    return p;
}

// async-signal-safe uintptr -> 0x-prefixed hex.
char * refault_hex(char * p, uintptr_t v) {
    static const char hexd[] = "0123456789abcdef";
    *p++ = '0';
    *p++ = 'x';
    char tmp[16];
    int n = 0;
    if (v == 0) {
        tmp[n++] = '0';
    }
    while (v > 0) {
        tmp[n++] = hexd[v & 0xf];
        v >>= 4;
    }
    while (n > 0) {
        *p++ = tmp[--n];
    }
    return p;
}

char * refault_lit(char * p, const char * s) {
    while (*s) {
        *p++ = *s++;
    }
    return p;
}

void paged_refault_sigsegv_handler(int sig, siginfo_t * si, void * /*uc*/) {
    const uintptr_t addr = (uintptr_t) (si ? si->si_addr : nullptr);

    if (g_refault_armed.load(std::memory_order_acquire) && g_refault_traps) {
        const size_t n = g_refault_ntraps.load(std::memory_order_acquire);
        for (size_t i = 0; i < n; ++i) {
            refault_trap & t = g_refault_traps[i];
            if (t.armed.load(std::memory_order_acquire) == 0) {
                continue;
            }
            if (addr < t.lo || addr >= t.hi) {
                continue;
            }

            const long pg  = g_refault_pagesize.load(std::memory_order_relaxed);
            const int  once = g_refault_once.load(std::memory_order_relaxed);

            // Restore access so the faulting instruction can retry. With ONCE we drop protection
            // on this whole K/V-per-layer trap region (so we do not segv-storm on the same range);
            // without ONCE we restore just the single faulting page and keep the rest watched.
            void * raddr;
            size_t rlen;
            if (once) {
                raddr = (void *) t.lo;
                rlen  = (size_t) (t.hi - t.lo);
            } else {
                const uintptr_t page = addr & ~((uintptr_t) pg - 1);
                raddr = (void *) page;
                rlen  = (size_t) pg;
            }
            mprotect(raddr, rlen, PROT_READ | PROT_WRITE);
            if (once) {
                t.armed.store(0, std::memory_order_release);
            }

            const uint64_t cnt = g_refault_fault_count.fetch_add(1, std::memory_order_relaxed) + 1;
            if (t.is_v) {
                g_refault_fault_v.fetch_add(1, std::memory_order_relaxed);
            } else {
                g_refault_fault_k.fetch_add(1, std::memory_order_relaxed);
            }

            if (cnt <= g_refault_max.load(std::memory_order_relaxed)) {
                char buf[256];
                char * p = buf;
                p = refault_lit(p, "KV_REFAULT_TRACE step=");
                p = refault_u64(p, g_refault_step.load(std::memory_order_relaxed));
                p = refault_lit(p, " addr=");
                p = refault_hex(p, addr);
                p = refault_lit(p, " kind=");
                p = refault_lit(p, t.is_v ? "V" : "K");
                p = refault_lit(p, " layer=");
                p = refault_u64(p, t.layer_il);
                p = refault_lit(p, " block=");
                p = refault_u64(p, t.block);
                p = refault_lit(p, " state=SWAPPED page=");
                p = refault_hex(p, addr & ~((uintptr_t) pg - 1));
                p = refault_lit(p, " offset=");
                p = refault_u64(p, (uint64_t) (addr - t.lo));
                p = refault_lit(p, " fault_count=");
                p = refault_u64(p, cnt);
                *p++ = '\n';
                (void) !write(STDERR_FILENO, buf, (size_t) (p - buf));

                if (g_refault_backtrace.load(std::memory_order_relaxed)) {
                    void * bt[32];
                    const int nb = backtrace(bt, 32);
                    const char * hdr = "KV_REFAULT_TRACE_BT\n";
                    (void) !write(STDERR_FILENO, hdr, std::strlen(hdr));
                    backtrace_symbols_fd(bt, nb, STDERR_FILENO);
                }
            }
            return; // retry faulting instruction against the now-readable page
        }
    }

    // Not one of our (currently-armed) KV pages: this is a genuine fault. Count it, restore the
    // previous SIGSEGV disposition, and return so the faulting instruction re-runs and crashes
    // through the original handler instead of being silently swallowed.
    g_refault_unmapped.fetch_add(1, std::memory_order_relaxed);
    sigaction(SIGSEGV, &g_refault_old_sa, nullptr);
    (void) sig;
}

} // namespace
#endif // LLAMA_KV_REFAULT_TRACE_SUPPORTED

void llama_kv_cache::paged_refault_init() {
#if defined(LLAMA_KV_REFAULT_TRACE_SUPPORTED)
    if (!kv_paged_enabled || !paged_refault_trace_requested) {
        return;
    }
    // Refault tracing only makes sense on the same CPU/single-stream path that swap+madvise use.
    if (v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0) {
        LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED_REFAULT_TRACE=1 requires CPU KV path "
                "(n_stream==1, !v_trans, paged enabled); refault tracing disabled\n", __func__);
        return;
    }
    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return;
    }

    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = paged_refault_sigsegv_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    if (sigaction(SIGSEGV, &sa, &g_refault_old_sa) != 0) {
        LLAMA_LOG_WARN("%s: failed to install refault SIGSEGV handler (errno=%d); disabled\n",
                __func__, errno);
        return;
    }

    g_refault_pagesize.store(page, std::memory_order_relaxed);
    g_refault_backtrace.store(paged_refault_trace_backtrace ? 1 : 0, std::memory_order_relaxed);
    g_refault_once.store(paged_refault_trace_once ? 1 : 0, std::memory_order_relaxed);
    g_refault_max.store(paged_refault_trace_max, std::memory_order_relaxed);
    g_refault_installed.store(true, std::memory_order_release);

    paged_refault_trace_enabled = true;
    paged_refault_protected.assign(paged_n_blocks, 0);
    LLAMA_LOG_INFO("%s: KV paged refault tracing enabled (mprotect PROT_NONE on SWAPPED KV pages, "
            "backtrace=%d once=%d max=%llu)\n",
            __func__, paged_refault_trace_backtrace ? 1 : 0, paged_refault_trace_once ? 1 : 0,
            (unsigned long long) paged_refault_trace_max);
#else
    if (paged_refault_trace_requested) {
        LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED_REFAULT_TRACE=1 requires Linux+glibc; disabled\n", __func__);
    }
#endif
}

void llama_kv_cache::paged_refault_protect_block(uint32_t physical_block) const {
#if defined(LLAMA_KV_REFAULT_TRACE_SUPPORTED)
    if (!paged_refault_trace_enabled || physical_block >= paged_n_blocks) {
        return;
    }
    const long page = g_refault_pagesize.load(std::memory_order_relaxed);
    if (page <= 0) {
        return;
    }
    const uintptr_t pg = (uintptr_t) page;

    // Build the trap table once, covering every block x layer x {K,V}. Tensor base pointers are
    // fixed after buffer allocation, so this is built outside any signal context and never
    // reallocated while the handler can run.
    if (g_refault_traps == nullptr) {
        const size_t cap = (size_t) paged_n_blocks * layers.size() * 2 + 1;
        refault_trap * traps = new (std::nothrow) refault_trap[cap];
        if (!traps) {
            return;
        }
        size_t idx = 0;
        for (uint32_t b = 0; b < paged_n_blocks; ++b) {
            for (const auto & layer : layers) {
                ggml_tensor * k = layer.k_stream.empty() ? nullptr : layer.k_stream[0];
                ggml_tensor * v = layer.v_stream.empty() ? nullptr : layer.v_stream[0];
                auto add = [&](ggml_tensor * t, uint8_t is_v) {
                    if (!t || idx >= cap) {
                        return;
                    }
                    paged_block_page_range page_range;
                    if (!paged_compute_block_page_range(
                            t, b, (uint64_t) t->nb[1], pg, page_range)) {
                        return;
                    }
                    refault_trap & tr = traps[idx++];
                    tr.lo = page_range.page_begin;
                    tr.hi = page_range.page_end;
                    tr.block = b;
                    tr.layer_il = layer.il;
                    tr.is_v = is_v;
                    tr.armed.store(0, std::memory_order_relaxed);
                };
                add(k, 0);
                add(v, 1);
            }
        }
        g_refault_traps = traps;
        g_refault_ntraps.store(idx, std::memory_order_release);
        g_refault_armed.store(true, std::memory_order_release);
    }

    if (paged_refault_protected[physical_block]) {
        return;
    }

    const size_t n = g_refault_ntraps.load(std::memory_order_relaxed);
    uint64_t pages = 0;
    bool any_fail = false;
    for (size_t i = 0; i < n; ++i) {
        refault_trap & t = g_refault_traps[i];
        if (t.block != physical_block) {
            continue;
        }
        if (mprotect((void *) t.lo, (size_t) (t.hi - t.lo), PROT_NONE) != 0) {
            any_fail = true;
            continue;
        }
        t.armed.store(1, std::memory_order_release);
        pages += (uint64_t) ((t.hi - t.lo) / pg);
    }
    if (any_fail) {
        paged_refault_protect_failures += 1;
    }
    if (pages > 0) {
        paged_refault_protected[physical_block] = 1;
        paged_refault_protect_calls += 1;
        paged_refault_protected_pages += pages;
    }
#else
    (void) physical_block;
#endif
}

void llama_kv_cache::paged_refault_unprotect_block(uint32_t physical_block) const {
#if defined(LLAMA_KV_REFAULT_TRACE_SUPPORTED)
    if (!paged_refault_trace_enabled || physical_block >= paged_n_blocks ||
            g_refault_traps == nullptr) {
        return;
    }
    if (physical_block < paged_refault_protected.size() && !paged_refault_protected[physical_block]) {
        return;
    }
    const long page = g_refault_pagesize.load(std::memory_order_relaxed);
    const uintptr_t pg = page > 0 ? (uintptr_t) page : 4096;
    const size_t n = g_refault_ntraps.load(std::memory_order_relaxed);
    uint64_t pages = 0;
    bool any_fail = false;
    for (size_t i = 0; i < n; ++i) {
        refault_trap & t = g_refault_traps[i];
        if (t.block != physical_block) {
            continue;
        }
        if (t.armed.load(std::memory_order_acquire) == 0) {
            continue;
        }
        if (mprotect((void *) t.lo, (size_t) (t.hi - t.lo), PROT_READ | PROT_WRITE) != 0) {
            any_fail = true;
        } else {
            pages += (uint64_t) ((t.hi - t.lo) / pg);
        }
        t.armed.store(0, std::memory_order_release);
    }
    if (any_fail) {
        paged_refault_unprotect_failures += 1;
    }
    if (physical_block < paged_refault_protected.size()) {
        paged_refault_protected[physical_block] = 0;
    }
    if (pages > 0) {
        paged_refault_unprotect_calls += 1;
        paged_refault_unprotected_pages += pages;
    }
#else
    (void) physical_block;
#endif
}

void llama_kv_cache::paged_refault_unprotect_all() const {
#if defined(LLAMA_KV_REFAULT_TRACE_SUPPORTED)
    if (!paged_refault_trace_enabled) {
        return;
    }
    for (uint32_t b = 0; b < paged_n_blocks; ++b) {
        paged_refault_unprotect_block(b);
    }
    // Stop the handler from touching the table after this point.
    g_refault_armed.store(false, std::memory_order_release);
#endif
}

void llama_kv_cache::paged_refault_drain() const {
#if defined(LLAMA_KV_REFAULT_TRACE_SUPPORTED)
    if (!paged_refault_trace_enabled) {
        return;
    }
    paged_refault_fault_count        = g_refault_fault_count.load(std::memory_order_relaxed);
    paged_refault_fault_k_count      = g_refault_fault_k.load(std::memory_order_relaxed);
    paged_refault_fault_v_count      = g_refault_fault_v.load(std::memory_order_relaxed);
    paged_refault_unmapped_fault_count = g_refault_unmapped.load(std::memory_order_relaxed);
    paged_refault_trace_enabled_flag = 1;
    // Distinct faulted blocks: count blocks whose protection was dropped by a fault (armed flag
    // cleared while we still recorded it as protected). Coarse signal; the per-fault
    // KV_REFAULT_TRACE lines carry the authoritative block list.
    uint64_t faulted_blocks = 0;
    if (g_refault_traps) {
        const size_t n = g_refault_ntraps.load(std::memory_order_relaxed);
        for (uint32_t b = 0; b < paged_n_blocks; ++b) {
            if (b >= paged_refault_protected.size() || !paged_refault_protected[b]) {
                continue;
            }
            bool any_disarmed = false;
            for (size_t i = 0; i < n; ++i) {
                if (g_refault_traps[i].block == b &&
                        g_refault_traps[i].armed.load(std::memory_order_relaxed) == 0) {
                    any_disarmed = true;
                    break;
                }
            }
            if (any_disarmed) {
                faulted_blocks += 1;
            }
        }
    }
    paged_refault_fault_blocks = faulted_blocks;
#endif
}

void llama_kv_cache::paged_assert_identity(const slot_info & sinfo) {
    if (!kv_paged_enabled) {
        return;
    }

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (const uint32_t cell : sinfo.idxs[s]) {
            const uint32_t phys_cell = paged_resolve(cell);
            paged_identity_checks += 1;
            if (phys_cell == PAGED_BLOCK_INVALID) {
                paged_identity_fail += paged_non_identity_enabled ? 0 : 1;
                paged_mapping_oob_fail += paged_non_identity_enabled ? 1 : 0;
                LLAMA_LOG_WARN("%s: KV paged mapping check failed: cell=%u phys_cell=INVALID block_size=%u\n",
                        __func__, cell, paged_block_size);
            } else if (!paged_non_identity_enabled && phys_cell != cell) {
                paged_identity_fail += 1;
                LLAMA_LOG_WARN("%s: KV paged identity check failed: cell=%u phys_cell=%u block_size=%u\n",
                        __func__, cell, phys_cell, paged_block_size);
            }
        }
    }
}

void llama_kv_cache::paged_shadow_validate(const slot_info & sinfo, uint32_t n_kv) const {
    if (!kv_paged_enabled) {
        return;
    }

    if (n_stream != 1 || v_trans || sinfo.n_stream() != 1 || sinfo.s0 != 0 || sinfo.s1 != 0) {
        paged_shadow_gather_fail += 1;
        return;
    }

    paged_shadow_gather_calls += 1;
    paged_shadow_validate_calls += 1;

    // Raw validation reads are safe only for committed RESIDENT rows. SWAPPED, RELEASED and
    // PENDING_WRITE pages either have no readable authority or are not committed yet.
    const auto row_block_unreadable = [&](uint32_t row) -> bool {
        if (paged_block_size == 0) {
            return false;
        }
        const uint32_t block = row / paged_block_size;
        if (block >= paged_block_states.size()) {
            return true;
        }
        return paged_block_states[block] != paged_block_state::RESIDENT;
    };

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            const ggml_tensor * k = layer.k_stream[0];
            if (!k->data) {
                paged_shadow_gather_fail += 1;
            } else {
                const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
                const size_t row_size = ggml_row_size(k->type, n_embd_k_gqa);
                const uint8_t * base = static_cast<const uint8_t *>(k->data);
                std::vector<uint8_t> shadow(row_size);

                for (uint32_t r = 0; r < n_kv; ++r) {
                    uint32_t phys = paged_resolve(r);
                    if (phys == PAGED_BLOCK_INVALID) {
                        paged_shadow_gather_fail += 1;
                        phys = r;
                    }
                    if (phys != r) {
                        paged_shadow_gather_changed += 1;
                    }

                    // Skip rows whose resolved or logical backing block is SWAPPED so we never
                    // read madvise'd / PROT_NONE K/V pages from validation.
                    if (row_block_unreadable(phys) || row_block_unreadable(r)) {
                        paged_shadow_validate_swapped_blocks_skipped += 1;
                        paged_shadow_validate_fault_risk_skipped += 1;
                        paged_shadow_validate_bytes_skipped += row_size;
                        continue;
                    }

                    paged_shadow_validate_blocks_checked += 1;
                    std::memcpy(shadow.data(), base + (size_t) phys * row_size, row_size);
                    if (paged_non_identity_enabled) {
                        paged_shadow_skipped_non_identity += 1;
                    } else if (std::memcmp(shadow.data(), base + (size_t) r * row_size, row_size) != 0) {
                        paged_shadow_gather_mismatch += 1;
                    }
                }
            }
        }

        if (!layer.v_stream.empty() && layer.v_stream[0]) {
            const ggml_tensor * v = layer.v_stream[0];
            if (!v->data) {
                paged_shadow_gather_fail += 1;
            } else {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);
                const size_t row_size = ggml_row_size(v->type, n_embd_v_gqa);
                const uint8_t * base = static_cast<const uint8_t *>(v->data);
                std::vector<uint8_t> shadow(row_size);

                for (uint32_t r = 0; r < n_kv; ++r) {
                    uint32_t phys = paged_resolve(r);
                    if (phys == PAGED_BLOCK_INVALID) {
                        paged_shadow_gather_fail += 1;
                        phys = r;
                    }
                    if (phys != r) {
                        paged_shadow_gather_changed += 1;
                    }

                    if (row_block_unreadable(phys) || row_block_unreadable(r)) {
                        paged_shadow_validate_swapped_blocks_skipped += 1;
                        paged_shadow_validate_fault_risk_skipped += 1;
                        paged_shadow_validate_bytes_skipped += row_size;
                        continue;
                    }

                    paged_shadow_validate_blocks_checked += 1;
                    std::memcpy(shadow.data(), base + (size_t) phys * row_size, row_size);
                    if (paged_non_identity_enabled) {
                        paged_shadow_skipped_non_identity += 1;
                    } else if (std::memcmp(shadow.data(), base + (size_t) r * row_size, row_size) != 0) {
                        paged_shadow_gather_mismatch += 1;
                    }
                }
            }
        }
    }
}

bool llama_kv_cache::paged_ingraph_gather_supported(int32_t il) const {
    if (!kv_paged_enabled) {
        return false;
    }

    const int32_t ikv = map_layer_ids.at(il);
    const auto & layer = layers[ikv];
    const bool supported = n_stream == 1 && !v_trans &&
        layer.k && layer.v &&
        layer.k->type == GGML_TYPE_F32 &&
        layer.v->type == GGML_TYPE_F32;

    if (!supported && !paged_ingraph_warned) {
        LLAMA_LOG_WARN("%s: KV paged in-graph gather requires n_stream==1, !v_trans, and F32 K/V cache "
                "(n_stream=%u, v_trans=%d, k_type=%s, v_type=%s) - falling back to continuous K/V views\n",
                __func__, n_stream, (int) v_trans,
                layer.k ? ggml_type_name(layer.k->type) : "none",
                layer.v ? ggml_type_name(layer.v->type) : "none");
        paged_ingraph_warned = true;
    }

    return supported;
}

void llama_kv_cache::paged_log_timing() const {
    if (!paged_resume_timing_enabled) {
        return;
    }

    fprintf(stderr,
            "KV_PAGED_TIMING "
            "set_input_us=%llu set_input_calls=%llu "
            "idle_maintenance_us=%llu idle_maintenance_calls=%llu "
            "swap_out_us=%llu swap_out_calls=%llu "
            "check_read_us=%llu check_read_calls=%llu\n",
            (unsigned long long) paged_timing_set_input_us,
            (unsigned long long) paged_timing_set_input_calls,
            (unsigned long long) paged_timing_idle_maintenance_us,
            (unsigned long long) paged_timing_idle_maintenance_calls,
            (unsigned long long) paged_timing_swap_out_us,
            (unsigned long long) paged_timing_swap_out_calls,
            (unsigned long long) paged_timing_check_read_us,
            (unsigned long long) paged_timing_check_read_calls);
}

void llama_kv_cache::paged_log_base_timing() const {
    if (!paged_base_timing_enabled) {
        return;
    }

    fprintf(stderr,
            "KV_PAGED_TIMING_SUMMARY "
            "apply_calls=%llu apply_paged_total_us=%llu "
            "apply_ubatch_us=%llu note_cells_us=%llu assert_identity_us=%llu "
            "swap_out_window_us=%llu ensure_resident_us=%llu "
            "clear_frontier_us=%llu madvise_tail_us=%llu paged_release_blocks_us=%llu "
            "set_row_idx_calls=%llu set_row_idx_total_us=%llu "
            "row_idx_fill_us=%llu active_visible_us=%llu nonidentity_probe_us=%llu "
            "swapped_blocks_scan_us=%llu check_read_resident_us=%llu "
            "check_read_resident_calls=%llu paged_resolve_calls=%llu "
            "cells_scanned=%llu blocks_scanned=%llu row_idx_entries=%llu "
            "getenv_calls=%llu\n",
            (unsigned long long) paged_base_timing_apply_calls,
            (unsigned long long) paged_base_timing_apply_paged_total_us,
            (unsigned long long) paged_base_timing_apply_ubatch_us,
            (unsigned long long) paged_base_timing_note_cells_us,
            (unsigned long long) paged_base_timing_assert_identity_us,
            (unsigned long long) paged_base_timing_swap_out_window_us,
            (unsigned long long) paged_base_timing_ensure_resident_us,
            (unsigned long long) paged_base_timing_clear_frontier_us,
            (unsigned long long) paged_base_timing_madvise_tail_us,
            (unsigned long long) paged_base_timing_paged_release_blocks_us,
            (unsigned long long) paged_base_timing_set_row_idx_calls,
            (unsigned long long) paged_base_timing_set_row_idx_total_us,
            (unsigned long long) paged_base_timing_row_idx_fill_us,
            (unsigned long long) paged_base_timing_active_visible_us,
            (unsigned long long) paged_base_timing_nonidentity_probe_us,
            (unsigned long long) paged_base_timing_swapped_blocks_scan_us,
            (unsigned long long) paged_base_timing_check_read_resident_us,
            (unsigned long long) paged_base_timing_check_read_resident_calls,
            (unsigned long long) paged_base_timing_paged_resolve_calls,
            (unsigned long long) paged_base_timing_cells_scanned,
            (unsigned long long) paged_base_timing_blocks_scanned,
            (unsigned long long) paged_base_timing_row_idx_entries,
            (unsigned long long) paged_base_timing_getenv_calls);
}

void llama_kv_cache::paged_log_stats() const {
    if (!kv_paged_enabled && !paged_block_release_requested) {
        return;
    }

    // Stage 7D-A: copy live atomic refault counters into the instance before emitting.
    paged_refault_drain();

    LLAMA_LOG_INFO("%s: KV paged metadata stats: enabled=%d block_size=%u n_blocks=%u "
            "blocks_in_use=%llu free_blocks=%zu alloc_calls=%llu identity_checks=%llu identity_fail=%llu "
            "write_resolve_checks=%llu write_resolve_fail=%llu write_resolve_changed=%llu "
            "test_mapping_fault_triggers=%llu test_mapping_fault_read_triggers=%llu "
            "test_mapping_fault_write_triggers=%llu "
            "shadow_gather_calls=%llu shadow_gather_changed=%llu shadow_gather_mismatch=%llu shadow_gather_fail=%llu "
            "shadow_skipped_non_identity=%llu "
            "shadow_validate_calls=%llu shadow_validate_blocks_checked=%llu "
            "shadow_validate_swapped_blocks_skipped=%llu shadow_validate_fault_risk_skipped=%llu "
            "shadow_validate_bytes_skipped=%llu "
            "ingraph_gather_layers=%llu paged_identity_fast_path_enabled=%d "
            "paged_identity_fast_path_layers=%u paged_identity_fast_path_reject_reason=%s "
            "paged_row_idx_inputs_created=%llu paged_row_idx_set_calls=%llu "
            "row_idx_changed=%llu row_idx_fail=%llu "
            "non_identity_enabled=%d block_mapping_changed=%llu mapping_oob_fail=%llu "
            "logical_to_physical_checks=%llu logical_to_physical_fail=%llu "
            "paged_block_release_enabled=%d paged_block_release_calls=%llu paged_blocks_released=%llu "
            "paged_blocks_released_unused=%llu paged_block_release_bytes=%llu "
            "paged_block_release_blocks_last=%llu paged_block_release_bytes_last=%llu "
            "paged_block_release_skip_live=%llu paged_block_release_skip_unaligned=%llu paged_block_release_fail=%llu "
            "paged_block_release_rss_samples=%llu paged_block_release_rss_before_last_kb=%llu "
            "paged_block_release_rss_after_last_kb=%llu paged_block_release_rss_before_max_kb=%llu "
            "paged_block_release_rss_after_min_kb=%llu paged_block_release_rss_drop_last_kb=%llu "
            "paged_block_release_rss_drop_max_kb=%llu "
            "paged_block_ensure_calls=%llu paged_block_ensure_released=%llu paged_release_violation=%llu "
            "paged_active_release_violation=%llu paged_padded_release_violation=%llu "
            "paged_idle_trace_enabled=%d paged_idle_active_seq_steps=%llu "
            "paged_idle_active_seq_empty=%llu paged_idle_seq_seen_count=%llu "
            "paged_idle_active_seq_count_last=%llu paged_idle_idle_seq_count_last=%llu "
            "paged_idle_active_seq_count_max=%llu paged_idle_non_empty_blocks=%llu "
            "paged_idle_single_seq_blocks=%llu paged_idle_multi_seq_blocks=%llu "
            "paged_idle_blocks_with_active_seq=%llu paged_idle_blocks_without_active_seq=%llu "
            "paged_idle_cold_candidates=%llu "
            "paged_idle_read_window_blocks=%llu paged_idle_cold_in_read_window=%llu "
            "paged_idle_cold_not_in_read_window=%llu paged_idle_skip_mixed_active=%llu "
            "paged_idle_safe_swap_candidates=%llu "
            "paged_idle_swap_enabled=%d paged_idle_swap_candidates=%llu "
            "paged_idle_swap_out_calls=%llu paged_idle_swap_skip_not_remapped=%llu "
            "paged_idle_swap_skip_not_resident=%llu paged_idle_swap_skip_protected=%llu "
            "paged_idle_swap_skip_deferred=%llu "
            "paged_nonidentity_enabled=%d paged_nonidentity_remap_rows=%llu "
            "paged_nonidentity_remap_blocks=%llu paged_nonidentity_skip_no_dummy=%llu "
            "paged_nonidentity_skip_not_masked=%llu paged_nonidentity_skip_not_resident=%llu "
            "paged_nonidentity_cold_in_read_window_before=%llu "
            "paged_nonidentity_cold_in_read_window_after=%llu "
            "paged_nonidentity_safe_candidates_after=%llu "
            "paged_swapped_redirect_rows=%llu paged_swapped_redirect_blocks=%llu "
            "paged_swapped_redirect_skip_no_dummy=%llu paged_swapped_active_visible_violation=%llu paged_swapped_redirect_probe_rows=%llu paged_swapped_redirect_probe_swapped_rows=%llu paged_swapped_redirect_probe_resident_rows=%llu paged_swapped_redirect_probe_invalid_rows=%llu paged_swapped_redirect_probe_state_mismatch=%llu paged_swapped_redirect_probe_disabled=%llu paged_swapped_active_visible_violation_rows=%llu paged_swapped_active_visible_violation_blocks=%llu paged_swapped_active_visible_logical_seq_has=%llu paged_swapped_active_visible_phys_seq_has=%llu paged_swapped_active_visible_in_read_window=%llu paged_swapped_active_visible_not_in_read_window=%llu paged_swapped_active_visible_masked=%llu paged_swapped_active_visible_unmasked=%llu paged_row_mapping_invalid_fatal=%llu paged_write_mapping_invalid_fatal=%llu paged_active_row_nonresident_fatal=%llu paged_input_setup_fatal=%llu paged_no_dummy_restore_sync=%llu paged_active_row_dummy_redirect_blocked=%llu paged_swapped_active_violation_rows=%llu paged_swapped_active_violation_blocks=%llu paged_swapped_active_violation_block_had_active_owner_at_swapout=%llu paged_swapped_active_violation_after_swapout_write=%llu paged_swapped_active_violation_resolve_to_swapped=%llu paged_swapped_active_visible_restore_rows=%llu paged_swapped_active_visible_restore_blocks=%llu paged_swap_out_skip_active_visible_block=%llu paged_swap_out_skip_active_owned_block=%llu paged_swap_out_candidate_blocks=%llu paged_swap_out_allowed_blocks=%llu paged_swap_out_skip_fullprefix_read_window_only=%llu paged_swap_out_skip_true_active_owned=%llu paged_swap_out_skip_true_active_unmasked=%llu paged_active_restore_from_swapped_blocks=%llu paged_idle_only_swapped_blocks=%llu paged_write_to_swapped_block=%llu paged_write_to_swapped_block_seq=%llu "
            "paged_cov_idle_owned_blocks=%llu paged_cov_in_read_window_blocks=%llu "
            "paged_cov_not_in_read_window_blocks=%llu paged_cov_resident_safe_blocks=%llu "
            "paged_cov_nonidentity_remapped_blocks=%llu "
            "paged_cov_idle_owned_bytes=%llu paged_cov_in_read_window_bytes=%llu "
            "paged_cov_resident_safe_bytes=%llu paged_cov_nonidentity_remapped_bytes=%llu "
            "paged_swap_enabled=%d paged_swap_out_calls=%llu paged_swap_in_calls=%llu "
            "paged_blocks_swapped_out=%llu paged_blocks_swapped_in=%llu "
            "paged_swap_bytes_out=%llu paged_swap_bytes_in=%llu paged_swap_in_last_block=%u "
            "paged_swap_backend_failures=%llu paged_swap_window_skipped=%llu "
            "paged_swap_read_swapped_hits=%llu paged_swap_read_swap_in_calls=%llu paged_swap_read_swap_in_failures=%llu "
            "paged_swap_write_swapped_hits=%llu paged_swap_write_swap_in_calls=%llu paged_swap_write_swap_in_failures=%llu "
            "paged_swap_in_fail_no_offset=%llu paged_swap_in_fail_bad_size=%llu "
            "paged_swap_in_fail_read_cell=%llu paged_swap_in_fail_tensor_set=%llu "
            "paged_swap_madvise_calls=%llu paged_swap_madvise_bytes=%llu "
            "paged_swap_madvise_failures=%llu paged_swap_madvise_skipped=%llu "
            "paged_swap_madvise_skip_no_full_page=%llu paged_swap_madvise_skip_neighbor=%llu "
            "paged_swap_rss_samples=%llu paged_swap_rss_before_last_kb=%llu "
            "paged_swap_rss_after_last_kb=%llu paged_swap_rss_drop_last_kb=%llu "
            "paged_swap_rss_drop_max_kb=%llu paged_swap_rss_before_first_kb=%llu "
            "paged_swap_rss_total_drop_kb=%llu paged_swap_rss_drop_sum_kb=%llu "
            "paged_prefetch_seq_calls=%llu paged_prefetch_seq_blocks=%llu "
            "paged_prefetch_seq_bytes=%llu paged_prefetch_seq_skip_resident=%llu "
            "paged_prefetch_seq_skip_released=%llu paged_prefetch_seq_failures=%llu "
            "kv_mincore_enabled=%d kv_mincore_sample_calls=%llu kv_mincore_failures=%llu "
            "kv_mincore_total_bytes=%llu kv_mincore_resident_bytes=%llu "
            "kv_mincore_total_pages=%llu kv_mincore_resident_pages=%llu "
            "kv_mincore_resident_ratio_permille=%llu "
            "kv_mincore_k_total_bytes=%llu kv_mincore_k_resident_bytes=%llu "
            "kv_mincore_v_total_bytes=%llu kv_mincore_v_resident_bytes=%llu "
            "kv_mincore_prefill_resident_bytes=%llu kv_mincore_before_madvise_resident_bytes=%llu "
            "kv_mincore_after_madvise_resident_bytes=%llu kv_mincore_after_resume_resident_bytes=%llu "
            "kv_mincore_madvise_drop_bytes=%llu kv_mincore_resume_recover_bytes=%llu "
            "kv_mincore_swapped_block_count=%llu kv_mincore_swapped_total_bytes=%llu "
            "kv_mincore_swapped_resident_bytes=%llu kv_mincore_swapped_nonresident_bytes=%llu "
            "kv_mincore_swapped_resident_blocks=%llu kv_mincore_swapped_nonresident_blocks=%llu "
            "kv_mincore_swapped_resident_ratio_permille=%llu "
            "paged_refault_trace_enabled=%llu paged_refault_fault_count=%llu "
            "paged_refault_fault_k_count=%llu paged_refault_fault_v_count=%llu "
            "paged_refault_fault_blocks=%llu paged_refault_unmapped_fault_count=%llu "
            "paged_refault_protect_calls=%llu paged_refault_unprotect_calls=%llu "
            "paged_refault_protected_pages=%llu paged_refault_unprotected_pages=%llu "
            "paged_refault_protect_failures=%llu paged_refault_unprotect_failures=%llu\n",
            __func__, kv_paged_enabled ? 1 : 0, paged_block_size, paged_n_blocks,
            (unsigned long long) paged_blocks_in_use,
            paged_free_list.size(),
            (unsigned long long) paged_alloc_calls,
            (unsigned long long) paged_identity_checks,
            (unsigned long long) paged_identity_fail,
            (unsigned long long) paged_write_resolve_checks,
            (unsigned long long) paged_write_resolve_fail,
            (unsigned long long) paged_write_resolve_changed,
            (unsigned long long) paged_test_mapping_fault_triggers,
            (unsigned long long) paged_test_mapping_fault_read_triggers,
            (unsigned long long) paged_test_mapping_fault_write_triggers,
            (unsigned long long) paged_shadow_gather_calls,
            (unsigned long long) paged_shadow_gather_changed,
            (unsigned long long) paged_shadow_gather_mismatch,
            (unsigned long long) paged_shadow_gather_fail,
            (unsigned long long) paged_shadow_skipped_non_identity,
            (unsigned long long) paged_shadow_validate_calls,
            (unsigned long long) paged_shadow_validate_blocks_checked,
            (unsigned long long) paged_shadow_validate_swapped_blocks_skipped,
            (unsigned long long) paged_shadow_validate_fault_risk_skipped,
            (unsigned long long) paged_shadow_validate_bytes_skipped,
            (unsigned long long) paged_ingraph_gather_layers,
            paged_identity_fast_path_enabled ? 1 : 0,
            paged_identity_fast_path_layers,
            llama_kv_paged_identity_fast_path_reject_name(paged_identity_fast_path_reject),
            (unsigned long long) paged_row_idx_inputs_created,
            (unsigned long long) paged_row_idx_set_calls,
            (unsigned long long) paged_row_idx_changed,
            (unsigned long long) paged_row_idx_fail,
            paged_non_identity_enabled ? 1 : 0,
            (unsigned long long) paged_block_mapping_changed,
            (unsigned long long) paged_mapping_oob_fail,
            (unsigned long long) paged_logical_to_physical_checks,
            (unsigned long long) paged_logical_to_physical_fail,
            paged_block_release_enabled ? 1 : 0,
            (unsigned long long) paged_block_release_calls,
            (unsigned long long) paged_blocks_released,
            (unsigned long long) paged_blocks_released_unused,
            (unsigned long long) paged_block_release_bytes,
            (unsigned long long) paged_block_release_blocks_last,
            (unsigned long long) paged_block_release_bytes_last,
            (unsigned long long) paged_block_release_skip_live,
            (unsigned long long) paged_block_release_skip_unaligned,
            (unsigned long long) paged_block_release_fail,
            (unsigned long long) paged_block_release_rss_samples,
            (unsigned long long) paged_block_release_rss_before_last_kb,
            (unsigned long long) paged_block_release_rss_after_last_kb,
            (unsigned long long) paged_block_release_rss_before_max_kb,
            (unsigned long long) paged_block_release_rss_after_min_kb,
            (unsigned long long) paged_block_release_rss_drop_last_kb,
            (unsigned long long) paged_block_release_rss_drop_max_kb,
            (unsigned long long) paged_block_ensure_calls,
            (unsigned long long) paged_block_ensure_released,
            (unsigned long long) paged_release_violation,
            (unsigned long long) paged_active_release_violation,
            (unsigned long long) paged_padded_release_violation,
            paged_idle_trace_enabled ? 1 : 0,
            (unsigned long long) paged_idle_active_seq_steps,
            (unsigned long long) paged_idle_active_seq_empty,
            (unsigned long long) paged_idle_seq_seen_count,
            (unsigned long long) paged_idle_active_seq_count_last,
            (unsigned long long) paged_idle_idle_seq_count_last,
            (unsigned long long) paged_idle_active_seq_count_max,
            (unsigned long long) paged_idle_non_empty_blocks,
            (unsigned long long) paged_idle_single_seq_blocks,
            (unsigned long long) paged_idle_multi_seq_blocks,
            (unsigned long long) paged_idle_blocks_with_active_seq,
            (unsigned long long) paged_idle_blocks_without_active_seq,
            (unsigned long long) paged_idle_cold_candidates,
            (unsigned long long) paged_idle_read_window_blocks,
            (unsigned long long) paged_idle_cold_in_read_window,
            (unsigned long long) paged_idle_cold_not_in_read_window,
            (unsigned long long) paged_idle_skip_mixed_active,
            (unsigned long long) paged_idle_safe_swap_candidates,
            paged_idle_swap_enabled ? 1 : 0,
            (unsigned long long) paged_idle_swap_candidates,
            (unsigned long long) paged_idle_swap_out_calls,
            (unsigned long long) paged_idle_swap_skip_not_remapped,
            (unsigned long long) paged_idle_swap_skip_not_resident,
            (unsigned long long) paged_idle_swap_skip_protected,
            (unsigned long long) paged_idle_swap_skip_deferred,
            paged_nonidentity_probe_enabled ? 1 : 0,
            (unsigned long long) paged_nonidentity_remap_rows,
            (unsigned long long) paged_nonidentity_remap_blocks,
            (unsigned long long) paged_nonidentity_skip_no_dummy,
            (unsigned long long) paged_nonidentity_skip_not_masked,
            (unsigned long long) paged_nonidentity_skip_not_resident,
            (unsigned long long) paged_nonidentity_cold_in_read_window_before,
            (unsigned long long) paged_nonidentity_cold_in_read_window_after,
            (unsigned long long) paged_nonidentity_safe_candidates_after,
            (unsigned long long) paged_swapped_redirect_rows,
            (unsigned long long) paged_swapped_redirect_blocks,
            (unsigned long long) paged_swapped_redirect_skip_no_dummy,
            (unsigned long long) paged_swapped_active_visible_violation,
            (unsigned long long) paged_swapped_redirect_probe_rows,
            (unsigned long long) paged_swapped_redirect_probe_swapped_rows,
            (unsigned long long) paged_swapped_redirect_probe_resident_rows,
            (unsigned long long) paged_swapped_redirect_probe_invalid_rows,
            (unsigned long long) paged_swapped_redirect_probe_state_mismatch,
            (unsigned long long) paged_swapped_redirect_probe_disabled,
            (unsigned long long) paged_swapped_active_visible_violation_rows,
            (unsigned long long) paged_swapped_active_visible_violation_blocks,
            (unsigned long long) paged_swapped_active_visible_logical_seq_has,
            (unsigned long long) paged_swapped_active_visible_phys_seq_has,
            (unsigned long long) paged_swapped_active_visible_in_read_window,
            (unsigned long long) paged_swapped_active_visible_not_in_read_window,
            (unsigned long long) paged_swapped_active_visible_masked,
            (unsigned long long) paged_swapped_active_visible_unmasked,
            (unsigned long long) paged_row_mapping_invalid_fatal,
            (unsigned long long) paged_write_mapping_invalid_fatal,
            (unsigned long long) paged_active_row_nonresident_fatal,
            (unsigned long long) paged_input_setup_fatal,
            (unsigned long long) paged_no_dummy_restore_sync,
            (unsigned long long) paged_active_row_dummy_redirect_blocked,
            (unsigned long long) paged_swapped_active_violation_rows,
            (unsigned long long) paged_swapped_active_violation_blocks,
            (unsigned long long) paged_swapped_active_violation_block_had_active_owner_at_swapout,
            (unsigned long long) paged_swapped_active_violation_after_swapout_write,
            (unsigned long long) paged_swapped_active_violation_resolve_to_swapped,
            (unsigned long long) paged_swapped_active_visible_restore_rows,
            (unsigned long long) paged_swapped_active_visible_restore_blocks,
            (unsigned long long) paged_swap_out_skip_active_visible_block,
            (unsigned long long) paged_swap_out_skip_active_owned_block,
            (unsigned long long) paged_swap_out_candidate_blocks,
            (unsigned long long) paged_swap_out_allowed_blocks,
            (unsigned long long) paged_swap_out_skip_fullprefix_read_window_only,
            (unsigned long long) paged_swap_out_skip_true_active_owned,
            (unsigned long long) paged_swap_out_skip_true_active_unmasked,
            (unsigned long long) paged_active_restore_from_swapped_blocks,
            (unsigned long long) paged_idle_only_swapped_blocks,
            (unsigned long long) paged_write_to_swapped_block,
            (unsigned long long) paged_write_to_swapped_block_seq,
            (unsigned long long) paged_cov_idle_owned_blocks,
            (unsigned long long) paged_cov_in_read_window_blocks,
            (unsigned long long) paged_cov_not_in_read_window_blocks,
            (unsigned long long) paged_cov_resident_safe_blocks,
            (unsigned long long) paged_cov_nonidentity_remapped_blocks,
            (unsigned long long) paged_cov_idle_owned_bytes,
            (unsigned long long) paged_cov_in_read_window_bytes,
            (unsigned long long) paged_cov_resident_safe_bytes,
            (unsigned long long) paged_cov_nonidentity_remapped_bytes,
            paged_swap_enabled ? 1 : 0,
            (unsigned long long) paged_swap_out_calls,
            (unsigned long long) paged_swap_in_calls,
            (unsigned long long) paged_blocks_swapped_out,
            (unsigned long long) paged_blocks_swapped_in,
            (unsigned long long) paged_swap_bytes_out,
            (unsigned long long) paged_swap_bytes_in,
            paged_swap_in_last_block,
            (unsigned long long) paged_swap_backend_failures,
            (unsigned long long) paged_swap_window_skipped,
            (unsigned long long) paged_swap_read_swapped_hits,
            (unsigned long long) paged_swap_read_swap_in_calls,
            (unsigned long long) paged_swap_read_swap_in_failures,
            (unsigned long long) paged_swap_write_swapped_hits,
            (unsigned long long) paged_swap_write_swap_in_calls,
            (unsigned long long) paged_swap_write_swap_in_failures,
            (unsigned long long) paged_swap_in_fail_no_offset,
            (unsigned long long) paged_swap_in_fail_bad_size,
            (unsigned long long) paged_swap_in_fail_read_cell,
            (unsigned long long) paged_swap_in_fail_tensor_set,
            (unsigned long long) paged_swap_madvise_calls,
            (unsigned long long) paged_swap_madvise_bytes,
            (unsigned long long) paged_swap_madvise_failures,
            (unsigned long long) paged_swap_madvise_skipped,
            (unsigned long long) paged_swap_madvise_skip_no_full_page,
            (unsigned long long) paged_swap_madvise_skip_neighbor,
            (unsigned long long) paged_swap_rss_samples,
            (unsigned long long) paged_swap_rss_before_last_kb,
            (unsigned long long) paged_swap_rss_after_last_kb,
            (unsigned long long) paged_swap_rss_drop_last_kb,
            (unsigned long long) paged_swap_rss_drop_max_kb,
            (unsigned long long) paged_swap_rss_before_first_kb,
            (unsigned long long) paged_swap_rss_total_drop_kb,
            (unsigned long long) paged_swap_rss_drop_sum_kb,
            (unsigned long long) paged_prefetch_seq_calls,
            (unsigned long long) paged_prefetch_seq_blocks,
            (unsigned long long) paged_prefetch_seq_bytes,
            (unsigned long long) paged_prefetch_seq_skip_resident,
            (unsigned long long) paged_prefetch_seq_skip_released,
            (unsigned long long) paged_prefetch_seq_failures,
            paged_mincore_enabled ? 1 : 0,
            (unsigned long long) paged_mincore_sample_calls,
            (unsigned long long) paged_mincore_failures,
            (unsigned long long) paged_mincore_total_bytes,
            (unsigned long long) paged_mincore_resident_bytes,
            (unsigned long long) paged_mincore_total_pages,
            (unsigned long long) paged_mincore_resident_pages,
            (unsigned long long) (paged_mincore_total_pages > 0
                ? paged_mincore_resident_pages * 1000ull / paged_mincore_total_pages : 0),
            (unsigned long long) paged_mincore_k_total_bytes,
            (unsigned long long) paged_mincore_k_resident_bytes,
            (unsigned long long) paged_mincore_v_total_bytes,
            (unsigned long long) paged_mincore_v_resident_bytes,
            (unsigned long long) paged_mincore_prefill_resident_bytes,
            (unsigned long long) paged_mincore_before_madvise_resident_bytes,
            (unsigned long long) paged_mincore_after_madvise_resident_bytes,
            (unsigned long long) paged_mincore_after_resume_resident_bytes,
            (unsigned long long) (paged_mincore_before_madvise_resident_bytes > paged_mincore_after_madvise_resident_bytes
                ? paged_mincore_before_madvise_resident_bytes - paged_mincore_after_madvise_resident_bytes : 0),
            (unsigned long long) (paged_mincore_after_resume_resident_bytes > paged_mincore_after_madvise_resident_bytes
                ? paged_mincore_after_resume_resident_bytes - paged_mincore_after_madvise_resident_bytes : 0),
            (unsigned long long) paged_mincore_swapped_block_count,
            (unsigned long long) paged_mincore_swapped_total_bytes,
            (unsigned long long) paged_mincore_swapped_resident_bytes,
            (unsigned long long) paged_mincore_swapped_nonresident_bytes,
            (unsigned long long) paged_mincore_swapped_resident_blocks,
            (unsigned long long) paged_mincore_swapped_nonresident_blocks,
            (unsigned long long) (paged_mincore_swapped_total_bytes > 0
                ? paged_mincore_swapped_resident_bytes * 1000ull / paged_mincore_swapped_total_bytes : 0),
            (unsigned long long) paged_refault_trace_enabled_flag,
            (unsigned long long) paged_refault_fault_count,
            (unsigned long long) paged_refault_fault_k_count,
            (unsigned long long) paged_refault_fault_v_count,
            (unsigned long long) paged_refault_fault_blocks,
            (unsigned long long) paged_refault_unmapped_fault_count,
            (unsigned long long) paged_refault_protect_calls,
            (unsigned long long) paged_refault_unprotect_calls,
            (unsigned long long) paged_refault_protected_pages,
            (unsigned long long) paged_refault_unprotected_pages,
            (unsigned long long) paged_refault_protect_failures,
            (unsigned long long) paged_refault_unprotect_failures);

    fprintf(stderr,
            "KV_PAGED_RELEASE_STATS contract=no_backing_unrecoverable_dead_or_unused_only "
            "release_calls=%llu released_blocks=%llu released_unused=%llu released_dead=%llu "
            "bounded_release_calls=%llu bounded_release_bytes=%llu bounded_release_blocks=%llu "
            "skip_owned=%llu skip_shared=%llu ownership_invalid=%llu backing_metadata_cleared=%llu "
            "backing_metadata_stale=%llu "
            "idempotent_skips=%llu reuse_allocations=%llu write_commits=%llu write_rollbacks=%llu "
            "released_redirect_rows=%llu released_redirect_blocks=%llu released_redirect_no_dummy=%llu "
            "released_redirect_no_dummy_pending_write=%llu "
            "dummy_candidate_resident=%llu dummy_candidate_pending_write_cell=%llu "
            "ensure_released=%llu ensure_pending_write_rejected=%llu "
            "release_violation=%llu active_release_violation=%llu padded_release_violation=%llu "
            "identity_fail=%llu mapping_oob_fail=%llu logical_mapping_fail=%llu write_resolve_fail=%llu "
            "row_idx_fail=%llu row_mapping_fatal=%llu write_mapping_fatal=%llu "
            "active_nonresident_fatal=%llu input_setup_fatal=%llu shadow_mismatch=%llu shadow_fail=%llu "
            "release_madvise_fail=%llu mincore_failures=%llu "
            "mincore_enabled=%d mincore_samples=%llu mincore_before_last=%llu mincore_after_last=%llu "
            "mincore_post_graph_last=%llu mincore_released_total_last=%llu mincore_drop_max=%llu "
            "mincore_reaccess_last=%llu mincore_reaccess_total_last=%llu "
            "fresh_verify_bytes=%llu fresh_verify_hash=%llu repeat_test_passes=%llu "
            "reuse_test_commits=%llu force_active_release_triggers=%llu\n",
            (unsigned long long) paged_block_release_calls,
            (unsigned long long) paged_blocks_released,
            (unsigned long long) paged_blocks_released_unused,
            (unsigned long long) paged_blocks_released_dead,
            (unsigned long long) paged_bounded_release_calls,
            (unsigned long long) paged_bounded_release_bytes,
            (unsigned long long) paged_bounded_release_blocks,
            (unsigned long long) paged_block_release_skip_owned,
            (unsigned long long) paged_block_release_skip_shared,
            (unsigned long long) paged_block_release_ownership_invalid,
            (unsigned long long) paged_block_release_metadata_cleared,
            (unsigned long long) paged_block_release_metadata_stale,
            (unsigned long long) paged_block_release_idempotent,
            (unsigned long long) paged_block_release_reuse_allocations,
            (unsigned long long) paged_release_write_commits,
            (unsigned long long) paged_release_write_rollbacks,
            (unsigned long long) paged_released_redirect_rows,
            (unsigned long long) paged_released_redirect_blocks,
            (unsigned long long) paged_released_redirect_no_dummy,
            (unsigned long long) paged_released_redirect_no_dummy_pending_write,
            (unsigned long long) paged_dummy_candidate_resident,
            (unsigned long long) paged_dummy_candidate_pending_write_cell,
            (unsigned long long) paged_block_ensure_released,
            (unsigned long long) paged_block_ensure_pending_write_rejected,
            (unsigned long long) paged_release_violation,
            (unsigned long long) paged_active_release_violation,
            (unsigned long long) paged_padded_release_violation,
            (unsigned long long) paged_identity_fail,
            (unsigned long long) paged_mapping_oob_fail,
            (unsigned long long) paged_logical_to_physical_fail,
            (unsigned long long) paged_write_resolve_fail,
            (unsigned long long) paged_row_idx_fail,
            (unsigned long long) paged_row_mapping_invalid_fatal,
            (unsigned long long) paged_write_mapping_invalid_fatal,
            (unsigned long long) paged_active_row_nonresident_fatal,
            (unsigned long long) paged_input_setup_fatal,
            (unsigned long long) paged_shadow_gather_mismatch,
            (unsigned long long) paged_shadow_gather_fail,
            (unsigned long long) paged_block_release_fail,
            (unsigned long long) paged_mincore_failures,
            paged_mincore_enabled ? 1 : 0,
            (unsigned long long) paged_release_mincore_samples,
            (unsigned long long) paged_release_mincore_before_last,
            (unsigned long long) paged_release_mincore_after_last,
            (unsigned long long) paged_release_mincore_post_graph_last,
            (unsigned long long) paged_release_mincore_total_last,
            (unsigned long long) paged_release_mincore_drop_max,
            (unsigned long long) paged_release_mincore_reaccess_last,
            (unsigned long long) paged_release_mincore_reaccess_total_last,
            (unsigned long long) paged_release_fresh_verify_bytes,
            (unsigned long long) paged_release_fresh_verify_hash,
            (unsigned long long) paged_release_test_repeat_passes,
            (unsigned long long) paged_release_test_reuse_commits,
            (unsigned long long) paged_test_force_active_release_triggers);

    if (paged_io_stats_enabled) {
        static const llama_kv_backing_store_stats empty_stats;
        const auto & backing_stats = kv_swap_store ? kv_swap_store->get_stats() : empty_stats;
        const uint64_t avg_out_us = paged_io_swap_out_timed_calls > 0 ?
            paged_io_swap_out_latency_us / paged_io_swap_out_timed_calls : 0;
        const uint64_t avg_in_us = paged_io_swap_in_timed_calls > 0 ?
            paged_io_swap_in_latency_us / paged_io_swap_in_timed_calls : 0;
        const auto avg_phase = [](uint64_t total, uint64_t calls) { return calls ? total / calls : 0; };
        fprintf(stderr,
                "KV_PAGED_IO_STATS "
                "block_swap_out_calls=%llu block_swap_in_calls=%llu "
                "backing_read_syscalls=%llu backing_write_syscalls=%llu "
                "bytes_read=%llu bytes_written=%llu "
                "avg_block_swap_out_latency_us=%llu max_block_swap_out_latency_us=%llu "
                "avg_block_swap_in_latency_us=%llu max_block_swap_in_latency_us=%llu "
                "staging_buffer_bytes=%llu "
                "k2_enabled=%d k2_group_byte_cap=%llu k2_staging_bound_bytes=%llu "
                "k2_peak_staging_groups=%u k2_peak_staging_bytes=%llu k2_pipeline_wall_us=%llu "
                "k2_exposed_read_wait_us=%llu k2_pipeline_stall_us=%llu "
                "k2_read_completed_ahead=%llu "
                "block_out_validate_us=%llu avg_block_out_validate_us=%llu "
                "block_out_pack_us=%llu avg_block_out_pack_us=%llu "
                "block_out_write_us=%llu avg_block_out_write_us=%llu "
                "block_out_metadata_us=%llu avg_block_out_metadata_us=%llu "
                "block_out_madvise_us=%llu avg_block_out_madvise_us=%llu "
                "block_in_validate_us=%llu avg_block_in_validate_us=%llu "
                "block_in_read_us=%llu avg_block_in_read_us=%llu "
                "block_in_unpack_us=%llu avg_block_in_unpack_us=%llu "
                "block_in_commit_us=%llu avg_block_in_commit_us=%llu "
                "restore_prefault_enabled=%d restore_prefault_groups=%llu "
                "restore_prefault_calls=%llu restore_prefault_us=%llu "
                "restore_prefault_minor_faults=%llu restore_prefault_major_faults=%llu "
                "restore_scatter_groups=%llu restore_scatter_us=%llu "
                "restore_scatter_fault_groups=%llu restore_scatter_minor_faults=%llu "
                "restore_scatter_major_faults=%llu\n",
                (unsigned long long) paged_swap_out_calls,
                (unsigned long long) paged_swap_in_calls,
                (unsigned long long) backing_stats.read_syscalls,
                (unsigned long long) backing_stats.write_syscalls,
                (unsigned long long) backing_stats.bytes_read,
                (unsigned long long) backing_stats.bytes_written,
                (unsigned long long) avg_out_us,
                (unsigned long long) paged_io_swap_out_latency_max_us,
                (unsigned long long) avg_in_us,
                (unsigned long long) paged_io_swap_in_latency_max_us,
                (unsigned long long) paged_io_staging_buffer_bytes,
                paged_restore_k2_enabled ? 1 : 0,
                (unsigned long long) paged_restore_group_byte_cap(),
                (unsigned long long) paged_restore_k2_staging_bound_bytes,
                paged_restore_k2_peak_staging_groups,
                (unsigned long long) paged_restore_k2_peak_staging_bytes,
                (unsigned long long) paged_restore_k2_pipeline_wall_us,
                (unsigned long long) paged_restore_k2_exposed_read_wait_us,
                (unsigned long long) paged_restore_k2_pipeline_stall_us,
                (unsigned long long) paged_restore_k2_read_completed_before_restore_complete,
                (unsigned long long) paged_io_block_out_validate_us,
                (unsigned long long) avg_phase(paged_io_block_out_validate_us, paged_io_block_out_validate_calls),
                (unsigned long long) paged_io_block_out_pack_us,
                (unsigned long long) avg_phase(paged_io_block_out_pack_us, paged_io_block_out_pack_calls),
                (unsigned long long) paged_io_block_out_write_us,
                (unsigned long long) avg_phase(paged_io_block_out_write_us, paged_io_block_out_write_calls),
                (unsigned long long) paged_io_block_out_metadata_us,
                (unsigned long long) avg_phase(paged_io_block_out_metadata_us, paged_io_block_out_metadata_calls),
                (unsigned long long) paged_io_block_out_madvise_us,
                (unsigned long long) avg_phase(paged_io_block_out_madvise_us, paged_io_block_out_madvise_calls),
                (unsigned long long) paged_io_block_in_validate_us,
                (unsigned long long) avg_phase(paged_io_block_in_validate_us, paged_io_block_in_validate_calls),
                (unsigned long long) paged_io_block_in_read_us,
                (unsigned long long) avg_phase(paged_io_block_in_read_us, paged_io_block_in_read_calls),
                (unsigned long long) paged_io_block_in_unpack_us,
                (unsigned long long) avg_phase(paged_io_block_in_unpack_us, paged_io_block_in_unpack_calls),
                (unsigned long long) paged_io_block_in_commit_us,
                (unsigned long long) avg_phase(paged_io_block_in_commit_us, paged_io_block_in_commit_calls),
                paged_restore_prefault_probe_enabled ? 1 : 0,
                (unsigned long long) paged_restore_prefault_groups,
                (unsigned long long) paged_restore_prefault_calls,
                (unsigned long long) paged_restore_prefault_us,
                (unsigned long long) paged_restore_prefault_minor,
                (unsigned long long) paged_restore_prefault_major,
                (unsigned long long) paged_restore_test_scatter_groups,
                (unsigned long long) paged_restore_scatter_us,
                (unsigned long long) paged_restore_fault_stats_groups,
                (unsigned long long) paged_restore_fault_stats_minor,
                (unsigned long long) paged_restore_fault_stats_major);
    }
}

void llama_kv_cache::swap_out_cell(uint32_t cell) {
    if (!kv_swap_enabled) {
        return;
    }

    if (kv_swap_mode_ != kv_swap_mode::exact || !kv_swap_store || v_trans || n_stream != 1 || v_cells.empty()) {
        return;
    }

    auto & cells = v_cells[0];
    if (cell >= cells.size() || !cells.is_resident(cell)) {
        return;
    }

    size_t total_size = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            total_size += layer.k_stream[0]->nb[1];
        }
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
            total_size += layer.v_stream[0]->nb[1];
        }
    }
    if (total_size == 0) {
        return;
    }

    // Fixed staging layout: layer0 K, layer0 V, layer1 K, layer1 V, ...
    std::vector<uint8_t> staging(total_size);
    size_t cursor = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            auto * k = layer.k_stream[0];
            const size_t row_size = k->nb[1];
            ggml_backend_tensor_get(k, staging.data() + cursor, (size_t) cell * row_size, row_size);
            cursor += row_size;
        }
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
            auto * v = layer.v_stream[0];
            const size_t row_size = v->nb[1];
            ggml_backend_tensor_get(v, staging.data() + cursor, (size_t) cell * row_size, row_size);
            cursor += row_size;
        }
    }

    GGML_ASSERT(cursor == total_size);

    uint64_t offset = 0;
    const auto status = kv_swap_store->write_cell(0, cell, staging.data(), staging.size(), offset);
    if (status != llama_kv_backing_store_status::ok) {
        kv_swap_backend_failures += 1;
        return;
    }

    cells.set_swap_offset(cell, offset);
    cells.set_swap_size(cell, staging.size());
    cells.set_state(cell, llama_kv_cell_state::SWAPPED);
    kv_swap_out_calls += 1;
}

void llama_kv_cache::swap_in_cell(uint32_t cell) {
    if (!kv_swap_enabled) {
        return;
    }

    if (kv_swap_mode_ != kv_swap_mode::exact || !kv_swap_store || v_trans || n_stream != 1 || v_cells.empty()) {
        return;
    }

    auto & cells = v_cells[0];
    if (cell >= cells.size() || !cells.is_swapped(cell)) {
        return;
    }

    const uint64_t offset = cells.get_swap_offset(cell);
    const size_t swap_size = cells.get_swap_size(cell);
    if (swap_size == 0) {
        return;
    }

    size_t total_size = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            total_size += layer.k_stream[0]->nb[1];
        }
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
            total_size += layer.v_stream[0]->nb[1];
        }
    }
    if (total_size == 0 || total_size != swap_size) {
        kv_swap_backend_failures += 1;
        return;
    }

    std::vector<uint8_t> staging(total_size);
    const auto status = kv_swap_store->read_cell(0, cell, offset, staging.data(), staging.size());
    if (status != llama_kv_backing_store_status::ok) {
        kv_swap_backend_failures += 1;
        return;
    }

    // Fixed staging layout mirrors swap_out_cell(): layer0 K, layer0 V, layer1 K, layer1 V, ...
    size_t cursor = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            auto * k = layer.k_stream[0];
            const size_t row_size = k->nb[1];
            ggml_backend_tensor_set(k, staging.data() + cursor, (size_t) cell * row_size, row_size);
            cursor += row_size;
        }
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
            auto * v = layer.v_stream[0];
            const size_t row_size = v->nb[1];
            ggml_backend_tensor_set(v, staging.data() + cursor, (size_t) cell * row_size, row_size);
            cursor += row_size;
        }
    }

    GGML_ASSERT(cursor == total_size);

    cells.set_state(cell, llama_kv_cell_state::RESIDENT);
    kv_swap_in_calls += 1;
}

void llama_kv_cache::ensure_resident(uint32_t n_kv) {
    if (!kv_swap_enabled) {
        return;
    }

    if (kv_swap_mode_ != kv_swap_mode::exact || v_cells.empty()) {
        return;
    }

    auto & cells = v_cells[0];
    const uint32_t end = std::min<uint32_t>(n_kv, cells.size());
    for (uint32_t i = 0; i < end; ++i) {
        if (cells.is_swapped(i)) {
            swap_in_cell(i);
        }
    }
    kv_swap_ensure_calls += 1;
}

void llama_kv_cache::swap_out_window(uint32_t n_kv) {
    if (!kv_swap_enabled) {
        return;
    }

    kv_swap_window_calls += 1;

    if (kv_swap_mode_ != kv_swap_mode::exact || v_trans || n_stream != 1 || v_cells.empty()) {
        kv_swap_window_skipped += 1;
        return;
    }
    if (kv_swap_window == 0) {
        kv_swap_window_skipped += 1;
        return;
    }
    if ((uint64_t) n_kv <= (uint64_t) kv_swap_window + kv_swap_sink) {
        kv_swap_window_skipped += 1;
        return;
    }

    auto & cells = v_cells[0];
    const uint32_t begin = std::min<uint32_t>(kv_swap_sink, cells.size());
    const uint32_t end = std::min<uint32_t>(n_kv - kv_swap_window, cells.size());
    for (uint32_t cell = begin; cell < end; ++cell) {
        if (cells.is_resident(cell)) {
            swap_out_cell(cell);
        }
    }
    madvise_swapped_runs(n_kv);
}

void llama_kv_cache::paged_swap_out_window(uint32_t n_kv) {
    if (!kv_paged_enabled || !paged_swap_enabled) {
        return;
    }
    if (paged_swap_explicit_only || paged_idle_swap_requested) {
        paged_swap_window_skipped += 1;
        return;
    }

    if (!kv_swap_store || v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0 ||
            paged_block_states.size() != paged_n_blocks) {
        paged_swap_window_skipped += 1;
        return;
    }

    const uint32_t sink_blocks = 1;
    const uint32_t window_blocks = 1;
    const uint32_t n_kv_blocks = (n_kv + paged_block_size - 1) / paged_block_size;
    if (n_kv_blocks <= sink_blocks + window_blocks) {
        paged_swap_window_skipped += 1;
        return;
    }

    const uint32_t begin = std::min<uint32_t>(sink_blocks, paged_n_blocks);
    const uint32_t end = std::min<uint32_t>(n_kv_blocks - window_blocks, paged_n_blocks);
    for (uint32_t physical_block = begin; physical_block < end; ++physical_block) {
        if (paged_block_states[physical_block] == paged_block_state::RESIDENT) {
            paged_swap_out_block(physical_block);
        }
    }
}

void llama_kv_cache::madvise_swapped_runs(uint32_t n_kv) {
    if (!kv_swap_madvise) {
        return;
    }

    kv_swap_madvise_calls += 1;

    if (kv_swap_mode_ != kv_swap_mode::exact || v_trans || n_stream != 1 || v_cells.empty()) {
        return;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        kv_swap_madvise_failures += 1;
        return;
    }
    const uintptr_t page_size = (uintptr_t) page;
    const auto page_align_up = [page_size](uintptr_t p) {
        return (p + page_size - 1) & ~(page_size - 1);
    };
    const auto page_align_down = [page_size](uintptr_t p) {
        return p & ~(page_size - 1);
    };

    auto & cells = v_cells[0];
    const uint32_t end = std::min<uint32_t>(n_kv, cells.size());
    uint32_t c_lo = 0;
    while (c_lo < end) {
        while (c_lo < end && !cells.is_swapped(c_lo)) {
            ++c_lo;
        }
        if (c_lo >= end) {
            break;
        }

        uint32_t c_hi = c_lo + 1;
        while (c_hi < end && cells.is_swapped(c_hi)) {
            ++c_hi;
        }

        for (const auto & layer : layers) {
            const auto dry_run_tensor = [&](ggml_tensor * t, uint32_t n_embd_gqa) {
                if (!t || !t->data) {
                    return;
                }

                kv_swap_madvise_candidate_runs += 1;

                const size_t row = ggml_row_size(t->type, n_embd_gqa);
                const uint64_t run_bytes = (uint64_t) (c_hi - c_lo) * (uint64_t) t->nb[1];
                if (row != (size_t) t->nb[1]) {
                    kv_swap_madvise_failures += 1;
                    kv_swap_madvise_skipped_bytes += run_bytes;
                    return;
                }

                const uintptr_t base = (uintptr_t) t->data;
                const uintptr_t lo_byte = base + (uintptr_t) c_lo * (uintptr_t) row;
                const uintptr_t hi_byte = base + (uintptr_t) c_hi * (uintptr_t) row;
                const uintptr_t a_start = page_align_up(lo_byte);
                const uintptr_t a_end = page_align_down(hi_byte);
                if (a_end > a_start) {
                    const size_t len = (size_t) (a_end - a_start);
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
                    if (madvise((void *) a_start, len, MADV_DONTNEED) == 0) {
                        kv_swap_madvise_advised_runs += 1;
                        kv_swap_madvise_advised_bytes += (uint64_t) len;
                    } else {
                        kv_swap_madvise_failures += 1;
                    }
#else
                    (void) len;
                    kv_swap_madvise_failures += 1;
#endif
                } else {
                    kv_swap_madvise_skipped_bytes += (uint64_t) (hi_byte - lo_byte);
                }
            };

            if (!layer.k_stream.empty()) {
                dry_run_tensor(layer.k_stream[0], hparams.n_embd_k_gqa(layer.il));
            }
            if (layer.v && !layer.v_stream.empty()) {
                dry_run_tensor(layer.v_stream[0], hparams.n_embd_v_gqa(layer.il));
            }
        }

        c_lo = c_hi;
    }
}

void llama_kv_cache::kv_swap_roundtrip_selftest() {
    const char * LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST = std::getenv("LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST");
    if (!LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST || std::atoi(LLAMA_KV_SWAP_ROUNDTRIP_SELFTEST) == 0) {
        return;
    }

    static bool done = false;
    if (done || !kv_swap_enabled || kv_swap_mode_ != kv_swap_mode::exact || !kv_swap_store ||
            v_trans || n_stream != 1 || v_cells.empty()) {
        return;
    }

    auto & cells = v_cells[0];
    uint32_t cell = cells.size();
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_resident(i)) {
            cell = i;
            break;
        }
    }
    if (cell == cells.size()) {
        return;
    }

    size_t total_size = 0;
    for (const auto & layer : layers) {
        if (!layer.k_stream.empty() && layer.k_stream[0]) {
            total_size += layer.k_stream[0]->nb[1];
        }
        if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
            total_size += layer.v_stream[0]->nb[1];
        }
    }
    if (total_size == 0) {
        return;
    }

    auto read_cell_bytes = [&](std::vector<uint8_t> & out) {
        out.resize(total_size);
        size_t cursor = 0;
        for (const auto & layer : layers) {
            if (!layer.k_stream.empty() && layer.k_stream[0]) {
                auto * k = layer.k_stream[0];
                const size_t row_size = k->nb[1];
                ggml_backend_tensor_get(k, out.data() + cursor, (size_t) cell * row_size, row_size);
                cursor += row_size;
            }
            if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
                auto * v = layer.v_stream[0];
                const size_t row_size = v->nb[1];
                ggml_backend_tensor_get(v, out.data() + cursor, (size_t) cell * row_size, row_size);
                cursor += row_size;
            }
        }
        return cursor == total_size;
    };

    std::vector<uint8_t> before;
    if (!read_cell_bytes(before)) {
        done = true;
        LLAMA_LOG_ERROR("%s: KV swap roundtrip selftest fail: snapshot layout mismatch\n", __func__);
        return;
    }

    swap_out_cell(cell);
    if (!cells.is_swapped(cell)) {
        done = true;
        LLAMA_LOG_ERROR("%s: KV swap roundtrip selftest fail: swap_out did not mark cell %u swapped\n",
                __func__, cell);
        return;
    }

    swap_in_cell(cell);
    if (!cells.is_resident(cell)) {
        done = true;
        LLAMA_LOG_ERROR("%s: KV swap roundtrip selftest fail: swap_in did not restore cell %u resident\n",
                __func__, cell);
        return;
    }

    std::vector<uint8_t> after;
    if (!read_cell_bytes(after)) {
        done = true;
        LLAMA_LOG_ERROR("%s: KV swap roundtrip selftest fail: restore layout mismatch\n", __func__);
        return;
    }

    done = true;
    if (before == after) {
        LLAMA_LOG_INFO("%s: KV swap roundtrip selftest pass: cell=%u bytes=%zu\n",
                __func__, cell, total_size);
    } else {
        LLAMA_LOG_ERROR("%s: KV swap roundtrip selftest fail: byte mismatch cell=%u bytes=%zu\n",
                __func__, cell, total_size);
    }
}

uint64_t llama_kv_cache::get_current_rss_kb() const {
#if defined(__linux__)
    FILE * f = fopen("/proc/self/statm", "r");
    if (!f) {
        return 0;
    }
    long pages_total = 0;
    long pages_rss   = 0;
    const int n = fscanf(f, "%ld %ld", &pages_total, &pages_rss);
    fclose(f);
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

uint64_t llama_kv_cache::get_peak_rss_kb() const {
#if defined(__linux__)
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) {
        return 0;
    }
    char line[256];
    uint64_t peak_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            unsigned long long value = 0;
            if (sscanf(line + 6, "%llu", &value) == 1) {
                peak_kb = (uint64_t) value;
            }
            break;
        }
    }
    fclose(f);
    return peak_kb;
#else
    return 0;
#endif
}

void llama_kv_cache::sample_swap_rss() {
    if (!kv_swap_rss_sample) {
        return;
    }

    const uint64_t rss_kb = get_current_rss_kb();
    if (rss_kb == 0) {
        return;
    }

    if (kv_swap_rss_samples == 0 || rss_kb < kv_swap_rss_min_kb) {
        kv_swap_rss_min_kb = rss_kb;
    }
    if (rss_kb > kv_swap_rss_max_kb) {
        kv_swap_rss_max_kb = rss_kb;
    }
    kv_swap_rss_last_kb = rss_kb;
    kv_swap_rss_samples += 1;
}

void llama_kv_cache::madvise_tail(uint32_t n_kv) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    if (!kv_lazy_tail) {
        return;
    }
    if (paged_non_identity_enabled) {
        if (!kv_lazy_tail_warned) {
            LLAMA_LOG_WARN("%s: KV lazy-tail disabled because paged non-identity mapping breaks "
                    "physical-prefix assumption\n", __func__);
            kv_lazy_tail_warned = true;
        }
        return;
    }
    // boundary: only the !v_trans / single-stream layout has contiguous per-cell rows.
    if (v_trans || n_stream != 1) {
        if (!kv_lazy_tail_warned) {
            LLAMA_LOG_WARN("%s: lazy-tail madvise skipped: requires !v_trans && n_stream==1 "
                    "(v_trans=%d, n_stream=%u)\n", __func__, (int) v_trans, n_stream);
            kv_lazy_tail_warned = true;
        }
        return;
    }

    const uint32_t kv_size = get_size();
    // tail starts at the next 256-cell boundary at/above n_kv, so it can never overlap the
    // [0, n_kv) read window even after n_kv grows by one pad step next step.
    const uint32_t lo_cell = GGML_PAD(n_kv, 256);
    if (lo_cell >= kv_size) {
        return; // no unused tail capacity
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return;
    }
    const uint64_t pg = (uint64_t) page;

    const int64_t t_start = ggml_time_us();
    if (lazy_tail_madvise_calls == 0) {
        lazy_tail_rss_before_kb = get_current_rss_kb();
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        ggml_tensor * k = layer.k_stream[0];
        ggml_tensor * v = layer.v_stream[0];

        // advise only the page-aligned interior of the tail byte range [lo_cell*row, kv_size*row)
        // (absolute-address aligned, never a relative offset -- see E1-lite), so we never touch
        // a page shared with the last live cell below lo_cell.
        auto advise = [&](ggml_tensor * t, uint64_t row) {
            if (!t || row == 0) {
                return;
            }
            char * base = (char *) t->data; // CPU backend: tensor data is a host pointer
            if (!base) {
                return;
            }
            const uintptr_t lo_a = (uintptr_t) base + (uintptr_t) lo_cell * row;
            const uintptr_t hi_a = (uintptr_t) base + (uintptr_t) kv_size * row;
            const uintptr_t a_start = (lo_a + pg - 1) & ~(uintptr_t) (pg - 1);
            const uintptr_t a_end   = hi_a & ~(uintptr_t) (pg - 1);
            if (a_end <= a_start) {
                return; // tail smaller than a page after trimming
            }
            const size_t len = (size_t) (a_end - a_start);
            const int rc = madvise((void *) a_start, len, MADV_DONTNEED);
            lazy_tail_madvise_calls += 1;
            if (rc != 0) {
                lazy_tail_madvise_failures += 1;
            } else {
                lazy_tail_madvise_bytes += len;
            }
        };

        advise(k, k ? ggml_row_size(k->type, hparams.n_embd_k_gqa(il)) : 0);
        advise(v, v ? ggml_row_size(v->type, hparams.n_embd_v_gqa(il)) : 0);
    }

    lazy_tail_madvise_us += (uint64_t) (ggml_time_us() - t_start);
    lazy_tail_rss_after_kb = get_current_rss_kb();
#else
    (void) n_kv;
#endif
}

void llama_kv_cache::paged_release_blocks(uint32_t n_kv) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    (void) n_kv;
    if (!kv_paged_enabled || !paged_block_release_enabled) {
        return;
    }
    if (v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0 ||
            paged_block_states.size() != paged_n_blocks) {
        return;
    }

    // Fail-stop guard: if the write context is invalid (compute-started failure
    // has poisoned the cache), refuse all destructive release until explicit reset.
    if (paged_write_context_invalid) {
        LLAMA_LOG_WARN("KV_PAGED_RELEASE_SKIP reason=CONTEXT_INVALID cause=%d\n",
                static_cast<int>(paged_write_context_invalid_cause));
        return;
    }

    paged_block_release_calls += 1;

    // RELEASED is destructive: it has no backing and is never recoverable. Protect every
    // physical block containing any live cell, regardless of the current padded graph width.
    const auto ownership = llama_kv_release_collect_ownership(
            v_cells, paged_n_blocks, paged_block_size, PAGED_BLOCK_INVALID, LLAMA_MAX_SEQ,
            [&](uint32_t logical_cell) { return paged_resolve(logical_cell); });
    if (!ownership.valid) {
        paged_block_release_ownership_invalid += ownership.invalid_mappings;
        LLAMA_LOG_ERROR(
                "KV_PAGED_RELEASE_ABORT reason=INVALID_LIVE_MAPPING invalid_mappings=%llu destructive_release_skipped=1\n",
                (unsigned long long) ownership.invalid_mappings);
        return;
    }
    const auto & owned = ownership.owned;
    const auto & shared = ownership.shared;

    const uint64_t blocks_before = paged_blocks_released;
    const uint64_t bytes_before = paged_block_release_bytes;
    const uint64_t rss_before_kb = get_current_rss_kb();
    paged_release_post_ranges.clear();
    uint64_t mincore_before = 0;

    for (uint32_t physical_block = 0; physical_block < paged_n_blocks; ++physical_block) {
        if (owned[physical_block]) {
            paged_block_release_skip_owned += 1;
            if (shared[physical_block]) {
                paged_block_release_skip_shared += 1;
            }
            continue;
        }

        const paged_block_state state =
            (paged_release_bounded_test_block_state_override.block == physical_block)
            ? static_cast<paged_block_state>(
                    paged_release_bounded_test_block_state_override.state)
            : paged_block_states[physical_block];
        // Unified safety gate — shares the same state rejection set as the
        // bounded and dry-run paths:
        // - RELEASED: already destructively discarded, idempotent skip
        // - SWAPPED:   recoverable history, must not be destroyed
        // - PENDING_WRITE: open write transaction, must not madvise mid-tx
        // - INVALID:   quarantined after compute-started failure, must not touch
        if (state == paged_block_state::RELEASED) {
            paged_block_release_idempotent += 1;
            continue;
        }
        if (state == paged_block_state::SWAPPED) {
            continue;
        }
        if (state == paged_block_state::PENDING_WRITE) {
            continue;
        }
        if (state == paged_block_state::INVALID) {
            continue;
        }

        // Per-candidate recheck immediately before madvise: re-confirm the
        // block state hasn't changed since the initial gate check, the write
        // context is still valid, and quarantine hasn't been set.  This is the
        // last safety barrier before the destructive MADV_DONTNEED syscall.
        if (paged_write_context_invalid) {
            continue;
        }
        {
            const paged_block_state recheck_state =
                (paged_release_bounded_test_block_state_override.block == physical_block)
                ? static_cast<paged_block_state>(
                        paged_release_bounded_test_block_state_override.state)
                : paged_block_states[physical_block];
            if (recheck_state != paged_block_state::RESIDENT &&
                    recheck_state != paged_block_state::UNUSED) {
                continue;
            }
        }

        std::vector<paged_release_range> advised_ranges;
        const uint64_t advised_bytes = paged_madvise_block(
                physical_block, &owned,
                paged_block_release_fail,
                paged_block_release_skip_unaligned,
                paged_block_release_skip_live,
                paged_mincore_enabled ? &advised_ranges : nullptr);

        if (advised_bytes > 0) {
            for (const auto & range : advised_ranges) {
                mincore_before += range.before_resident;
            }
            paged_release_post_ranges.insert(
                    paged_release_post_ranges.end(), advised_ranges.begin(), advised_ranges.end());
            if (physical_block < paged_released_ranges_by_block.size()) {
                paged_released_ranges_by_block[physical_block] = advised_ranges;
            }
            const uint32_t begin = physical_block * paged_block_size;
            const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
            for (uint32_t cell = begin; cell < end; ++cell) {
                if (cell < paged_swap_offsets.size()) {
                    paged_swap_offsets[cell] = 0;
                }
                if (cell < paged_swap_sizes.size()) {
                    paged_swap_sizes[cell] = 0;
                }
                paged_block_release_metadata_cleared += 1;
                if ((cell < paged_swap_offsets.size() && paged_swap_offsets[cell] != 0) ||
                        (cell < paged_swap_sizes.size() && paged_swap_sizes[cell] != 0)) {
                    paged_block_release_metadata_stale += 1;
                }
            }
            paged_block_release_bytes += advised_bytes;
            paged_block_states[physical_block] = paged_block_state::RELEASED;
            paged_blocks_released += 1;
            if (state == paged_block_state::UNUSED) {
                paged_blocks_released_unused += 1;
            } else {
                paged_blocks_released_dead += 1;
            }
            if (physical_block < paged_block_used.size() && paged_block_used[physical_block]) {
                paged_block_used[physical_block] = 0;
                if (paged_blocks_in_use > 0) {
                    paged_blocks_in_use -= 1;
                }
                if (std::find(paged_free_list.begin(), paged_free_list.end(), physical_block) ==
                        paged_free_list.end()) {
                    paged_free_list.push_back(physical_block);
                }
            }
        }
    }

    const uint64_t blocks_delta = paged_blocks_released - blocks_before;
    if (blocks_delta > 0) {
        const uint64_t bytes_delta = paged_block_release_bytes - bytes_before;
        const uint64_t rss_after_kb = get_current_rss_kb();
        const uint64_t rss_drop_kb = rss_before_kb > rss_after_kb ? rss_before_kb - rss_after_kb : 0;

        paged_block_release_blocks_last = blocks_delta;
        paged_block_release_bytes_last = bytes_delta;
        paged_block_release_rss_before_last_kb = rss_before_kb;
        paged_block_release_rss_after_last_kb = rss_after_kb;
        paged_block_release_rss_drop_last_kb = rss_drop_kb;
        paged_block_release_rss_samples += 1;

        if (rss_before_kb > paged_block_release_rss_before_max_kb) {
            paged_block_release_rss_before_max_kb = rss_before_kb;
        }
        if (paged_block_release_rss_after_min_kb == 0 ||
                (rss_after_kb > 0 && rss_after_kb < paged_block_release_rss_after_min_kb)) {
            paged_block_release_rss_after_min_kb = rss_after_kb;
        }
        if (rss_drop_kb > paged_block_release_rss_drop_max_kb) {
            paged_block_release_rss_drop_max_kb = rss_drop_kb;
        }
        if (paged_mincore_enabled) {
            uint64_t released_total = 0;
            const uint64_t mincore_after = paged_sample_release_ranges(
                    paged_release_post_ranges, &released_total);
            paged_release_mincore_samples += 1;
            paged_release_mincore_before_last = mincore_before;
            paged_release_mincore_after_last = mincore_after;
            paged_release_mincore_total_last = released_total;
            const uint64_t drop = mincore_before > mincore_after ? mincore_before - mincore_after : 0;
            paged_release_mincore_drop_max = std::max(paged_release_mincore_drop_max, drop);
        }
    }
    if (paged_release_test_repeat && !paged_release_test_repeat_active &&
            !paged_release_test_repeat_marker_emitted && blocks_delta > 0) {
        const auto hash_u64 = [](uint64_t hash, uint64_t value) {
            for (uint32_t i = 0; i < 8; ++i) {
                hash ^= (value >> (i * 8)) & 0xffU;
                hash *= 1099511628211ULL;
            }
            return hash;
        };
        const auto hash_candidate = [&](const llama_kv_release_block_ownership & value) {
            uint64_t hash = 1469598103934665603ULL;
            for (uint8_t owned_value : value.owned) {
                hash = hash_u64(hash, owned_value);
            }
            for (uint8_t shared_value : value.shared) {
                hash = hash_u64(hash, shared_value);
            }
            return hash;
        };
        const auto hash_state = [&]() {
            uint64_t hash = 1469598103934665603ULL;
            for (paged_block_state state : paged_block_states) {
                hash = hash_u64(hash, (uint8_t) state);
            }
            return hash;
        };
        const auto hash_metadata = [&]() {
            uint64_t hash = 1469598103934665603ULL;
            for (uint64_t offset : paged_swap_offsets) {
                hash = hash_u64(hash, offset);
            }
            for (size_t size : paged_swap_sizes) {
                hash = hash_u64(hash, size);
            }
            return hash;
        };
        const auto hash_ranges = [&]() {
            uint64_t hash = 1469598103934665603ULL;
            for (const auto & range : paged_release_post_ranges) {
                hash = hash_u64(hash, range.block);
                hash = hash_u64(hash, range.len);
                hash = hash_u64(hash, range.before_resident);
            }
            return hash;
        };

        const auto post_ranges = paged_release_post_ranges;
        const uint64_t first_candidate_hash = hash_candidate(ownership);
        const uint64_t first_state_hash = hash_state();
        const uint64_t first_metadata_hash = hash_metadata();
        const uint64_t first_range_hash = hash_ranges();
        const auto second_ownership = llama_kv_release_collect_ownership(
                v_cells, paged_n_blocks, paged_block_size, PAGED_BLOCK_INVALID, LLAMA_MAX_SEQ,
                [&](uint32_t logical_cell) { return paged_resolve(logical_cell); });
        const uint64_t release_before_second = paged_blocks_released;
        paged_release_test_repeat_active = true;
        paged_release_blocks(n_kv);
        paged_release_test_repeat_active = false;
        paged_release_post_ranges = post_ranges;
        const uint64_t second_transition = paged_blocks_released - release_before_second;
        const uint64_t second_candidate_hash = hash_candidate(second_ownership);
        const uint64_t second_state_hash = hash_state();
        const uint64_t second_metadata_hash = hash_metadata();
        const uint64_t second_range_hash = hash_ranges();
        fprintf(stderr,
                "KV_PAGED_RELEASE_R2 first_transition=%llu second_transition=%llu "
                "first_abort=0 second_abort=%d candidate_hash_first=%llu candidate_hash_second=%llu "
                "state_hash_first=%llu state_hash_second=%llu metadata_hash_first=%llu "
                "metadata_hash_second=%llu range_hash_first=%llu range_hash_second=%llu\n",
                (unsigned long long) blocks_delta,
                (unsigned long long) second_transition,
                second_ownership.valid ? 0 : 1,
                (unsigned long long) first_candidate_hash,
                (unsigned long long) second_candidate_hash,
                (unsigned long long) first_state_hash,
                (unsigned long long) second_state_hash,
                (unsigned long long) first_metadata_hash,
                (unsigned long long) second_metadata_hash,
                (unsigned long long) first_range_hash,
                (unsigned long long) second_range_hash);
        if (second_ownership.valid && second_transition == 0 &&
                first_candidate_hash == second_candidate_hash &&
                first_state_hash == second_state_hash &&
                first_metadata_hash == second_metadata_hash &&
                first_range_hash == second_range_hash) {
            paged_release_test_repeat_passes += 1;
        }
        paged_release_test_repeat_marker_emitted = true;
    }

    // Auto-reset test-only state override so fault injections are single-shot
    // across the legacy path as well (matching the bounded impl's reset pattern).
    paged_release_bounded_test_block_state_override.block = UINT32_MAX;
#else
    (void) n_kv;
#endif
}

// --- Bounded-release common implementation ------------------------------------
// Shared by the legacy paged_release_blocks_bounded (gated on
// LLAMA_KV_PAGED_RELEASE → paged_block_release_enabled, legacy counters) and the
// server bounded_release path (gated on LLAMA_KV_PRESSURE_BOUNDED_RELEASE,
// independent counters).  Both paths enforce the identical block of safety gates
// (layout, ownership, PENDING_WRITE, SWAPPED, RELEASED, madvise result, neighbour
// page protection) — the only differences are the authorisation source and which
// counter set to update.

llama_kv_bounded_release_result llama_kv_cache::paged_release_blocks_bounded(
        uint64_t target_bytes,
        uint32_t max_scan_blocks) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    // Legacy gate: must have paged_block_release_enabled (LLAMA_KV_PAGED_RELEASE=1)
    if (!kv_paged_enabled || !paged_block_release_enabled) {
        return {};
    }

    paged_bounded_release_counters cnt;
    cnt.bytes  = &paged_block_release_bytes;
    cnt.blocks = &paged_blocks_released;
    cnt.unused = &paged_blocks_released_unused;
    cnt.dead   = &paged_blocks_released_dead;

    return paged_release_blocks_bounded_impl(
            target_bytes, max_scan_blocks, cnt,
            true /* use_test_seams */);
#else
    (void) target_bytes;
    (void) max_scan_blocks;
    return {};
#endif
}

llama_kv_bounded_release_result llama_kv_cache::bounded_release(
        uint64_t target_bytes,
        uint32_t max_scan_blocks) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    // Server gate: must satisfy structural preconditions independently of
    // LLAMA_KV_PAGED_RELEASE.  The server policy layer gates on
    // LLAMA_KV_PRESSURE_BOUNDED_RELEASE before calling this.
    if (!bounded_release_can_enable()) {
        return {};
    }

    paged_bounded_release_counters cnt;
    cnt.calls  = &paged_bounded_release_calls;
    cnt.bytes  = &paged_bounded_release_bytes;
    cnt.blocks = &paged_bounded_release_blocks;
    cnt.unused = &paged_bounded_release_unused;
    cnt.dead   = &paged_bounded_release_dead;

    llama_kv_bounded_release_result result = paged_release_blocks_bounded_impl(
            target_bytes, max_scan_blocks, cnt,
            false /* use_test_seams — server path never accesses test seams */);

    if (cnt.calls) { *cnt.calls += 1; }
    return result;
#else
    (void) target_bytes;
    (void) max_scan_blocks;
    return {};
#endif
}

llama_kv_bounded_release_result llama_kv_cache::paged_release_blocks_bounded_impl(
        uint64_t target_bytes,
        uint32_t max_scan_blocks,
        const paged_bounded_release_counters & cnt,
        bool use_test_seams) {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    llama_kv_bounded_release_result result;

    // Structural safety — enforced equally for both legacy and server paths.
    // Both callers have already verified their respective authorisation gates
    // (paged_block_release_enabled for legacy, bounded_release_can_enable()
    // for server) before entering the impl.
    if (!kv_paged_enabled) {
        return result;
    }
    if (v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0 ||
            paged_block_states.size() != paged_n_blocks) {
        return result;
    }

    // Fail-stop guard: if the write context is invalid (compute-started failure
    // has poisoned the cache), refuse all destructive release until explicit reset.
    if (paged_write_context_invalid) {
        result.ownership_aborted = true;
        return result;
    }

    // target=0: return immediately — no ownership collection or scan.
    if (target_bytes == 0) {
        if (use_test_seams) {
            paged_release_bounded_test_force_ownership_abort     = false;
            paged_release_bounded_test_madvise_fail_block        = -1;
            paged_release_bounded_test_block_state_override.block = UINT32_MAX;
        }
        return result;
    }

    // scan budget of zero: return immediately without scanning or ownership check.
    if (max_scan_blocks == 0) {
        result.block_scan_exhausted = true;
        result.scan_budget_exhausted = true;
        result.shortfall_bytes      = target_bytes;
        if (use_test_seams) {
            paged_release_bounded_test_force_ownership_abort     = false;
            paged_release_bounded_test_madvise_fail_block        = -1;
            paged_release_bounded_test_block_state_override.block = UINT32_MAX;
        }
        return result;
    }

    // Ownership collection
    const auto ownership = llama_kv_release_collect_ownership(
            v_cells, paged_n_blocks, paged_block_size, PAGED_BLOCK_INVALID, LLAMA_MAX_SEQ,
            [&](uint32_t logical_cell) { return paged_resolve(logical_cell); });
    const bool force_abort = use_test_seams &&
        paged_release_bounded_test_force_ownership_abort;
    if (!ownership.valid || force_abort) {
        result.ownership_aborted = true;
        if (!ownership.valid) {
            paged_block_release_ownership_invalid += ownership.invalid_mappings;
            LLAMA_LOG_ERROR(
                    "KV_PAGED_RELEASE_BOUNDED_ABORT reason=INVALID_LIVE_MAPPING invalid_mappings=%llu\n",
                    (unsigned long long) ownership.invalid_mappings);
        }
        if (use_test_seams) {
            paged_release_bounded_test_force_ownership_abort     = false;
            paged_release_bounded_test_madvise_fail_block        = -1;
            paged_release_bounded_test_block_state_override.block = UINT32_MAX;
        }
        return result;
    }
    const auto & owned = ownership.owned;

    paged_release_scan_sync_layout();

    uint32_t scanned = 0;
    uint32_t no_release_scanned = paged_release_scan_scanned_since_release;
    bool no_candidate = false;
    const uint32_t scan_limit = std::min(max_scan_blocks, paged_n_blocks);
    const uint32_t start_block = paged_release_scan_cursor;
    const auto note_no_release = [&]() {
        no_release_scanned += 1;
        no_candidate = no_release_scanned >= paged_n_blocks;
        return no_candidate;
    };

    for (uint32_t step = 0; step < scan_limit; ++step) {
        const uint32_t physical_block = (start_block + step) % paged_n_blocks;
        scanned = step + 1;

        // Skip owned blocks (contains live cells)
        if (owned[physical_block]) {
            result.blocks_skipped_owned += 1;
            if (note_no_release()) {
                break;
            }
            continue;
        }

        const paged_block_state state =
            (use_test_seams &&
             paged_release_bounded_test_block_state_override.block == physical_block)
            ? static_cast<paged_block_state>(
                    paged_release_bounded_test_block_state_override.state)
            : paged_block_states[physical_block];
        // Unified safety gate — skip PENDING_WRITE, SWAPPED, RELEASED, INVALID.
        // Same state rejection set as the unbounded legacy path and the dry-run
        // scanner: quarantined INVALID blocks must never be madvise'd.
        if (state == paged_block_state::PENDING_WRITE ||
                state == paged_block_state::SWAPPED ||
                state == paged_block_state::RELEASED ||
                state == paged_block_state::INVALID) {
            result.blocks_skipped_state += 1;
            if (note_no_release()) {
                break;
            }
            continue;
        }

        // Per-candidate recheck: re-confirm safety conditions immediately
        // before madvise — same barrier as the legacy unbounded path.
        if (paged_write_context_invalid) {
            if (note_no_release()) {
                break;
            }
            continue;
        }
        {
            const paged_block_state recheck_state = use_test_seams &&
                paged_release_bounded_test_block_state_override.block == physical_block
                ? static_cast<paged_block_state>(
                        paged_release_bounded_test_block_state_override.state)
                : paged_block_states[physical_block];
            if (recheck_state != paged_block_state::RESIDENT &&
                    recheck_state != paged_block_state::UNUSED) {
                if (note_no_release()) {
                    break;
                }
                continue;
            }
        }

        // Candidate block: try madvise
        uint64_t advised_bytes = 0;
        uint64_t local_failures = 0;
        uint64_t local_skipped = 0;
        uint64_t local_skip_live = 0;
        const bool inject_fail = use_test_seams &&
            (int32_t)physical_block == paged_release_bounded_test_madvise_fail_block;
        if (!inject_fail) {
            advised_bytes = paged_madvise_block(
                    physical_block, &owned,
                    local_failures, local_skipped, local_skip_live,
                    nullptr);
        } else {
            if (use_test_seams) {
                paged_release_bounded_test_madvise_fail_block = -1;
            }
            local_failures = 1;
        }

        if (advised_bytes == 0) {
            result.madvise_failures += local_failures;
            if (note_no_release()) {
                break;
            }
            continue;
        }

        // madvise succeeded: release the block
        // Clear backing metadata
        const uint32_t begin = physical_block * paged_block_size;
        const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
        for (uint32_t cell = begin; cell < end; ++cell) {
            if (cell < paged_swap_offsets.size()) {
                paged_swap_offsets[cell] = 0;
            }
            if (cell < paged_swap_sizes.size()) {
                paged_swap_sizes[cell] = 0;
            }
        }

        paged_block_states[physical_block] = paged_block_state::RELEASED;
        result.released_blocks += 1;
        result.released_bytes += advised_bytes;
        no_release_scanned = 0;

        // Update usage tracking and free list
        if (physical_block < paged_block_used.size() && paged_block_used[physical_block]) {
            paged_block_used[physical_block] = 0;
            if (paged_blocks_in_use > 0) {
                paged_blocks_in_use -= 1;
            }
            if (std::find(paged_free_list.begin(), paged_free_list.end(), physical_block) ==
                    paged_free_list.end()) {
                paged_free_list.push_back(physical_block);
            }
        }

        // Update counters (legacy or independent based on caller)
        if (cnt.bytes)  { *cnt.bytes  += advised_bytes; }
        if (cnt.blocks) { *cnt.blocks += 1; }
        if (state == paged_block_state::UNUSED) {
            if (cnt.unused) { *cnt.unused += 1; }
        } else {
            if (cnt.dead)   { *cnt.dead   += 1; }
        }

        // Budget check: stop once target is met (allow one-block overshoot)
        if (result.released_bytes >= target_bytes) {
            break;
        }
    }

    result.blocks_scanned = scanned;
    result.block_scan_exhausted = (scanned >= scan_limit);

    // Resume after the range visited by this call. A successful release starts
    // a new no-release tour but does not discard the rest of this call's scan
    // budget; later non-released blocks count toward that fresh tour.
    paged_release_scan_cursor = (start_block + scanned) % paged_n_blocks;
    paged_release_scan_scanned_since_release = no_candidate ? 0 : no_release_scanned;

    const bool target_unmet = result.released_bytes < target_bytes;

    // Preserve the existing bounded result split: a partial candidate-space
    // walk is scan_budget_exhausted, while a full walk since the last successful
    // release is the only path that falls through to no_candidate. Full-budget
    // scans that released blocks retain target_shortfall semantics.
    result.scan_budget_exhausted =
        target_unmet && !no_candidate && max_scan_blocks < paged_n_blocks;

    if (use_test_seams) {
        paged_release_bounded_test_force_ownership_abort     = false;
        paged_release_bounded_test_madvise_fail_block        = -1;
        paged_release_bounded_test_block_state_override.block = UINT32_MAX;
    }

    if (result.released_bytes < target_bytes) {
        result.shortfall_bytes = target_bytes - result.released_bytes;
    }
    if (result.released_bytes > target_bytes) {
        result.overshoot_bytes = result.released_bytes - target_bytes;
    }

    return result;
#else
    (void) target_bytes;
    (void) max_scan_blocks;
    (void) cnt;
    (void) use_test_seams;
    return {};
#endif
}

bool llama_kv_cache::bounded_release_can_enable() const {
    return bounded_release_can_enable_diagnose().can_enable;
}

llama_kv_bounded_release_capability llama_kv_cache::bounded_release_can_enable_diagnose() const {
    // Per-condition decomposition of bounded_release_can_enable() for precise
    // diagnostic skip-reason attribution.  Server policy can use each field
    // individually to produce an exact skip_reason instead of a catch-all
    // "structurally_disabled".
    llama_kv_bounded_release_capability cap;

    cap.paged = kv_paged_enabled;
    if (!cap.paged) return cap;

    cap.ingraph = paged_ingraph_enabled;
    cap.layers_supported = paged_layers_supported;
    cap.row_idx = paged_row_idx_enabled;
    cap.swap_disabled = !paged_swap_enabled;
    cap.layout_supported =
        !v_trans && n_stream == 1 && paged_block_size > 0 && paged_n_blocks > 0;

    cap.can_enable = cap.paged && cap.ingraph && cap.layers_supported &&
                     cap.row_idx && cap.swap_disabled && cap.layout_supported;
    return cap;
}

uint64_t llama_kv_cache::sample_kv_resident_bytes() const {
    return sample_kv_resident().resident_bytes;
}

llama_kv_resident_sample llama_kv_cache::sample_kv_resident() const {
    llama_kv_resident_sample result;
#if defined(__linux__)
    if (!paged_mincore_enabled) {
        return result;
    }

    const uint64_t failures_before = paged_mincore_failures;
    const uint64_t resident_bytes = paged_sample_mincore();
    const long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || paged_mincore_failures != failures_before ||
            paged_mincore_total_pages == 0 || paged_mincore_total_bytes == 0 ||
            paged_mincore_resident_pages > paged_mincore_total_pages ||
            resident_bytes != paged_mincore_resident_bytes) {
        return result;
    }

    const uint64_t page = (uint64_t) page_size;
    if (paged_mincore_total_bytes / page != paged_mincore_total_pages ||
            paged_mincore_resident_bytes / page != paged_mincore_resident_pages ||
            paged_mincore_resident_bytes > paged_mincore_total_bytes ||
            paged_resident_object_id == 0 || paged_resident_generation == 0) {
        return result;
    }

    result.available = true;
    result.object_id = paged_resident_object_id;
    result.generation = paged_resident_generation;
    result.page_size = page;
    result.total_bytes = paged_mincore_total_bytes;
    result.resident_bytes = resident_bytes;
    result.total_pages = paged_mincore_total_pages;
    result.resident_pages = paged_mincore_resident_pages;
#endif
    return result;
}

llama_kv_release_budget_snapshot llama_kv_cache::sample_kv_release_budget() const {
    llama_kv_release_budget_snapshot result;
#if defined(__linux__)
    if (!bounded_release_can_enable() || paged_write_context_invalid ||
            paged_block_states.size() != paged_n_blocks) {
        return result;
    }

    const auto ownership = llama_kv_release_collect_ownership(
            v_cells, paged_n_blocks, paged_block_size, PAGED_BLOCK_INVALID, LLAMA_MAX_SEQ,
            [&](uint32_t logical_cell) { return paged_resolve(logical_cell); });
    if (!ownership.valid) {
        result.ownership_aborted = true;
        return result;
    }

    const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) {
        return result;
    }
    const uintptr_t pg = (uintptr_t) page;
    std::vector<paged_release_range> resident_ranges;
    std::vector<paged_release_range> reclaimable_ranges;
    for (const auto & layer : layers) {
        auto add_tensor = [&](ggml_tensor * tensor) {
            if (!tensor || !tensor->data) {
                return;
            }
            const uintptr_t lo = (uintptr_t) tensor->data;
            const uintptr_t hi = lo + (uintptr_t) ggml_nbytes(tensor);
            const uintptr_t begin = (lo + pg - 1) & ~(pg - 1);
            const uintptr_t end = hi & ~(pg - 1);
            if (end > begin) {
                resident_ranges.push_back({ (void *) begin, (size_t) (end - begin), 0, UINT32_MAX });
            }
        };
        add_tensor(layer.k_stream.empty() ? nullptr : layer.k_stream[0]);
        add_tensor(layer.v_stream.empty() ? nullptr : layer.v_stream[0]);
    }
    for (uint32_t block = 0; block < paged_n_blocks; ++block) {
        if (ownership.owned[block]) {
            continue;
        }
        const paged_block_state state = paged_block_states[block];
        if (state != paged_block_state::RESIDENT && state != paged_block_state::UNUSED) {
            continue;
        }

        for (const auto & layer : layers) {
            auto add_tensor = [&](ggml_tensor * tensor, uint64_t row) {
                if (!tensor || !tensor->data || row == 0) {
                    return;
                }
                const uint64_t lo_cell = (uint64_t) block * paged_block_size;
                const uint64_t hi_cell = std::min<uint64_t>(lo_cell + paged_block_size, paged_kv_size);
                const uintptr_t lo = (uintptr_t) tensor->data + (uintptr_t) lo_cell * row;
                const uintptr_t hi = (uintptr_t) tensor->data + (uintptr_t) hi_cell * row;
                const uintptr_t begin = (lo + pg - 1) & ~(pg - 1);
                const uintptr_t end = hi & ~(pg - 1);
                if (end > begin) {
                    reclaimable_ranges.push_back({ (void *) begin, (size_t) (end - begin), 0, block });
                }
            };
            ggml_tensor * k = layer.k_stream.empty() ? nullptr : layer.k_stream[0];
            ggml_tensor * v = layer.v_stream.empty() ? nullptr : layer.v_stream[0];
            add_tensor(k, k ? ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il)) : 0);
            add_tensor(v, v ? ggml_row_size(v->type, hparams.n_embd_v_gqa(layer.il)) : 0);
        }
    }

    const auto sample_ranges = [&](const std::vector<paged_release_range> & ranges,
                                   uint64_t & resident_bytes) {
        resident_bytes = 0;
        for (const auto & range : ranges) {
            std::vector<unsigned char> vec(range.len / pg);
            if (mincore(range.addr, range.len, vec.data()) != 0) {
                paged_mincore_failures += 1;
                return false;
            }
            for (unsigned char value : vec) {
                if (value & 1u) {
                    resident_bytes += pg;
                }
            }
        }
        return true;
    };
    if (!sample_ranges(resident_ranges, result.resident_bytes) ||
            !sample_ranges(reclaimable_ranges, result.reclaimable_resident_bytes)) {
        result.resident_bytes = 0;
        result.reclaimable_resident_bytes = 0;
        return result;
    }
    result.valid = true;
#endif
    return result;
}

llama_kv_bounded_release_result llama_kv_cache::paged_release_blocks_bounded_dry_run(
        uint64_t target_bytes,
        uint32_t max_scan_blocks) const {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    llama_kv_bounded_release_result result;

    // Dry-run is decoupled from LLAMA_KV_PAGED_RELEASE: it only requires
    // paged KV to be enabled so it can walk the K/V tensor layout.  The
    // destructive release flag is irrelevant — dry-run never calls madvise.
    if (!kv_paged_enabled) {
        return result;
    }
    if (v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0 ||
            paged_block_states.size() != paged_n_blocks) {
        return result;
    }

    // Fail-stop guard: if the write context is invalid, dry-run has no safe
    // candidates — all write-transaction blocks are quarantined.
    if (paged_write_context_invalid) {
        return result;
    }

    // target=0: return immediately — zero side effects.
    if (target_bytes == 0) {
        return result;
    }

    // scan budget of zero: return immediately without scanning or ownership check.
    if (max_scan_blocks == 0) {
        result.block_scan_exhausted = true;
        result.scan_budget_exhausted = true;
        result.shortfall_bytes      = target_bytes;
        return result;
    }

    // Ownership collection — same intent as the destructive variant.
    const auto ownership = llama_kv_release_collect_ownership(
            v_cells, paged_n_blocks, paged_block_size, PAGED_BLOCK_INVALID, LLAMA_MAX_SEQ,
            [&](uint32_t logical_cell) { return paged_resolve(logical_cell); });
    if (!ownership.valid) {
        result.ownership_aborted = true;
        return result;
    }
    const auto & owned = ownership.owned;

    const long page = sysconf(_SC_PAGESIZE);
    const uint64_t pg = (page > 0) ? (uint64_t) page : 4096;

    uint32_t scanned = 0;
    const uint32_t scan_limit = std::min(max_scan_blocks, paged_n_blocks);

    for (uint32_t physical_block = 0; physical_block < scan_limit; ++physical_block) {
        scanned = physical_block + 1;

        // Skip owned blocks (contains live cells)
        if (physical_block < owned.size() && owned[physical_block]) {
            result.blocks_skipped_owned += 1;
            continue;
        }

        // State gate: skip PENDING_WRITE, SWAPPED, RELEASED, INVALID.
        // Does NOT access test-only seams — dry-run is a production path.
        // Same rejection set as the destructive paths.
        const paged_block_state state = paged_block_states[physical_block];
        if (state == paged_block_state::PENDING_WRITE ||
                state == paged_block_state::SWAPPED ||
                state == paged_block_state::RELEASED ||
                state == paged_block_state::INVALID) {
            result.blocks_skipped_state += 1;
            continue;
        }

        // Candidate block: compute would-release byte count by walking each
        // layer's K/V tensors with the same page-alignment + neighbor-protection
        // logic as the destructive release path, but without calling madvise().
        uint64_t block_bytes = 0;

        for (const auto & layer : layers) {
            auto count_tensor = [&](ggml_tensor * t, uint64_t row) {
                if (!t || row == 0 || !t->data) {
                    return;
                }

                char * base = (char *) t->data;
                const uint64_t lo_cell = (uint64_t) physical_block * paged_block_size;
                const uint64_t hi_cell = std::min<uint64_t>(
                        lo_cell + paged_block_size, paged_kv_size);
                const uintptr_t lo_a = (uintptr_t) base + (uintptr_t) lo_cell * row;
                const uintptr_t hi_a = (uintptr_t) base + (uintptr_t) hi_cell * row;
                if (hi_a <= lo_a) {
                    return;
                }

                const uintptr_t a_start = (lo_a + pg - 1) & ~(uintptr_t) (pg - 1);
                const uintptr_t a_end   = hi_a & ~(uintptr_t) (pg - 1);
                if (a_end <= a_start) {
                    return;
                }

                // Page-alignment math already excludes partial pages at block
                // boundaries (which would be shared with neighbors).  The
                // aligned interior [a_start, a_end) is entirely within this
                // candidate block's address range, matching what the
                // destructive path would actually advise.
                block_bytes += (uint64_t) (a_end - a_start);
            };

            ggml_tensor * k = layer.k_stream.empty() ? nullptr : layer.k_stream[0];
            ggml_tensor * v = layer.v_stream.empty() ? nullptr : layer.v_stream[0];
            count_tensor(k, k ? ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il)) : 0);
            count_tensor(v, v ? ggml_row_size(v->type, hparams.n_embd_v_gqa(layer.il)) : 0);
        }

        if (block_bytes == 0) {
            // Page-sharing prevented all pages from being counted.
            // Not a failure — just an uncountable candidate.
            continue;
        }

        // This block WOULD be released by the destructive path.
        result.released_blocks += 1;
        result.released_bytes += block_bytes;

        // Budget check: stop once target is met (allow one-block overshoot)
        if (result.released_bytes >= target_bytes) {
            break;
        }
    }

    result.blocks_scanned = scanned;
    result.block_scan_exhausted = (scanned >= scan_limit);
    result.scan_budget_exhausted = result.released_bytes < target_bytes &&
        scan_limit < paged_n_blocks && scanned >= scan_limit;

    // NO test-seam access, NO state mutation, NO free-list change,
    // NO backing-metadata clear, NO global-counter increment.

    if (result.released_bytes < target_bytes) {
        result.shortfall_bytes = target_bytes - result.released_bytes;
    }
    if (result.released_bytes > target_bytes) {
        result.overshoot_bytes = result.released_bytes - target_bytes;
    }

    return result;
#else
    (void) target_bytes;
    (void) max_scan_blocks;
    return {};
#endif
}

llama_kv_bounded_release_result llama_kv_cache::bounded_release_dry_run(
        uint64_t target_bytes, uint32_t max_scan_blocks) {
    return paged_release_blocks_bounded_dry_run(target_bytes, max_scan_blocks);
}

llama_kv_release_status llama_kv_cache::paged_release_status() const {
    // Fast path: the stored enabled flag already encodes the full can_enable check.
    if (paged_block_release_enabled) {
        return llama_kv_release_status::available;
    }

    // Dry-run scanner preconditions (also required by destructive release).
    // The scanner walks K/V tensor data using the same address math, so it
    // needs paged KV enabled, a valid layout, and swap disabled.
    // It does NOT need paged_row_idx_enabled or layers_supported — those are
    // specific to the ingraph gather path and destructive release.
    if (!kv_paged_enabled) {
        return llama_kv_release_status::not_paged;
    }
    if (v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0) {
        return llama_kv_release_status::layout_unsupported;
    }
    if (paged_swap_enabled) {
        return llama_kv_release_status::swap_enabled;
    }
    // All preconditions met but LLAMA_KV_PAGED_RELEASE != 1.
    // Dry-run can still proceed — disabled is observational, not a skip reason.
    return llama_kv_release_status::disabled;
}

void llama_kv_cache::clear_frontier_advance(uint32_t n_kv) {
    if (!kv_lazy_clear) {
        return;
    }
    // boundary guard (should already hold: kv_lazy_clear is only set for this layout).
    if (v_trans || n_stream != 1) {
        if (!kv_lazy_clear_warned) {
            LLAMA_LOG_WARN("%s: lazy-clear advance skipped: requires !v_trans && n_stream==1\n", __func__);
            kv_lazy_clear_warned = true;
        }
        return;
    }

    const uint32_t kv_size = get_size();
    const uint32_t target  = std::min<uint32_t>(kv_size, GGML_PAD(n_kv, 256));
    if (target <= clear_frontier) {
        return; // already zeroed up to here
    }

    const int64_t t_start = ggml_time_us();

    // zero the newly-readable rows [clear_frontier, target) of every layer's K/V before the
    // graph reads them, so any cell that enters the [0, n_kv) view has defined (zero) bytes.
    for (const auto & layer : layers) {
        ggml_tensor * k = layer.k_stream[0];
        ggml_tensor * v = layer.v_stream[0];
        auto zero_rows = [&](ggml_tensor * t) {
            if (!t) {
                return;
            }
            const size_t row    = t->nb[1];                       // n_stream==1: contiguous rows
            const size_t offset = (size_t) clear_frontier * row;
            const size_t size   = (size_t) (target - clear_frontier) * row;
            if (size > 0) {
                ggml_backend_tensor_memset(t, 0, offset, size);
                lazy_clear_grow_bytes += size;
            }
        };
        zero_rows(k);
        zero_rows(v);
    }

    lazy_clear_calls += 1;
    lazy_clear_us    += (uint64_t) (ggml_time_us() - t_start);
    clear_frontier   = target;
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    if (uses_approx_dynamic_view()) {
        return get_reserve_n_kv();
    }

    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

uint32_t llama_kv_cache::get_visible_lo(const slot_info & sinfo) const {
    if (!uses_approx_dynamic_view()) {
        return 0;
    }

    uint32_t used_max_p1 = 0;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        used_max_p1 = std::max(used_max_p1, v_cells[sinfo.strm[s]].used_max_p1());
    }

    const uint32_t n_kv = get_reserve_n_kv();
    const uint32_t kv_size = get_size();
    const uint32_t keep_from = used_max_p1 > kv_swap_window ? used_max_p1 - kv_swap_window : 0;
    const uint32_t max_visible_lo = kv_size > n_kv ? kv_size - n_kv : 0;

    return std::min(keep_from, max_visible_lo);
}

uint32_t llama_kv_cache::get_reserve_n_kv() const {
    if (!uses_approx_dynamic_view()) {
        return get_size();
    }

    return std::min<uint32_t>(GGML_PAD(kv_swap_window, 256), get_size());
}

bool llama_kv_cache::uses_approx_dynamic_view() const {
    return kv_swap_enabled && kv_swap_mode_ == kv_swap_mode::approx &&
        kv_swap_window > 0 && !v_trans && n_stream == 1 && n_seq_max == 1;
}

ggml_tensor * llama_kv_cache::get_k(
        ggml_context * ctx,
        int32_t il,
        uint32_t n_kv,
        uint32_t visible_lo,
        const slot_info & sinfo,
        bool causal_attn,
        ggml_tensor * row_idx) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;
    const bool approx_dynamic = uses_approx_dynamic_view() && causal_attn && ns == 1;
    const uint64_t row_size = ggml_row_size(k->type, n_embd_k_gqa);
    const uint64_t byte_offset = row_size*visible_lo;

    if (uses_approx_dynamic_view() && !approx_dynamic && !kv_approx_dynamic_warned) {
        LLAMA_LOG_WARN("%s: KV swap approx dynamic view requires single-seq causal attention "
                "with !v_trans and n_stream==1 - falling back to original K view\n", __func__);
        kv_approx_dynamic_warned = true;
    }
    if (approx_dynamic && visible_lo > 0) {
        ++kv_approx_debug_get_k_visible_gt0_calls;
    }

    if (!paged_identity_fast_path_enabled && row_idx &&
            paged_ingraph_gather_supported(il) && ns == 1 && !approx_dynamic) {
        ggml_tensor * k2d = ggml_reshape_2d(ctx, k, n_embd_k_gqa, kv_size);
        ggml_tensor * rows = ggml_get_rows(ctx, k2d, row_idx);
        paged_ingraph_gather_layers += 1;

        return ggml_reshape_4d(ctx, rows,
                hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, 1);
    }

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            row_size,
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0 + (approx_dynamic ? byte_offset : 0));
}

ggml_tensor * llama_kv_cache::get_v(
        ggml_context * ctx,
        int32_t il,
        uint32_t n_kv,
        uint32_t visible_lo,
        const slot_info & sinfo,
        bool causal_attn,
        ggml_tensor * row_idx) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        const bool approx_dynamic = uses_approx_dynamic_view() && causal_attn && ns == 1;
        const uint64_t row_size = ggml_row_size(v->type, n_embd_v_gqa);
        const uint64_t byte_offset = row_size*visible_lo;

        if (uses_approx_dynamic_view() && !approx_dynamic && !kv_approx_dynamic_warned) {
            LLAMA_LOG_WARN("%s: KV swap approx dynamic view requires single-seq causal attention "
                    "with !v_trans and n_stream==1 - falling back to original V view\n", __func__);
            kv_approx_dynamic_warned = true;
        }
        if (approx_dynamic && visible_lo > 0) {
            ++kv_approx_debug_get_v_visible_gt0_calls;
        }

        if (!paged_identity_fast_path_enabled && row_idx &&
                paged_ingraph_gather_supported(il) && ns == 1 && !approx_dynamic) {
            ggml_tensor * v2d = ggml_reshape_2d(ctx, v, n_embd_v_gqa, kv_size);
            ggml_tensor * rows = ggml_get_rows(ctx, v2d, row_idx);
            paged_ingraph_gather_layers += 1;

            return ggml_reshape_4d(ctx, rows,
                    hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, 1);
        }

        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                row_size,                                                // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0 + (approx_dynamic ? byte_offset : 0));
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_paged_row_idx(ggml_context * ctx, uint32_t n_kv) const {
    if (!uses_paged_row_idx()) {
        return nullptr;
    }

    ggml_tensor * row_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
    ggml_set_input(row_idx);
    paged_row_idx_inputs_created += 1;

    return row_idx;
}

bool llama_kv_cache::uses_paged_row_idx() const {
    return paged_row_idx_enabled;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

bool llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    if (has_paged_swap_error()) {
        return false;
    }

    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            const uint32_t token = s*sinfo.size() + i;
            const uint32_t cell = sinfo.idxs[s][i];
            const uint32_t phys = paged_write_resolve_input(cell, ubatch, token);
            if (!paged_validate_write_mapping(phys)) {
                return false;
            }
            if (kv_paged_enabled && paged_block_size != 0 && phys != PAGED_BLOCK_INVALID) {
                const uint32_t block = phys / paged_block_size;
                if (block < paged_block_states.size() &&
                        paged_block_states[block] == paged_block_state::SWAPPED) {
                    paged_write_to_swapped_block += 1;
                    paged_swapped_active_violation_after_swapout_write += 1;
                    for (int32_t sid = 0; sid < ubatch->n_seq_id[token]; ++sid) {
                        if (ubatch->seq_id[token][sid] >= 0) {
                            paged_write_to_swapped_block_seq += 1;
                        }
                    }
                }
            }
            if (!paged_ensure_write_resident(phys)) {
                if (!has_paged_swap_error()) {
                    paged_input_setup_fatal += 1;
                    set_paged_swap_error(
                            llama_paged_swap_error_reason::INPUT_SETUP_FAILURE,
                            PAGED_BLOCK_INVALID, phys);
                }
                return false;
            }
            paged_trace_note_write_block(phys);
            data[s*sinfo.size() + i] = offs + phys;
        }
    }

    return true;
}

bool llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    if (has_paged_swap_error()) {
        return false;
    }

    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                const uint32_t token = s*sinfo.size() + i;
                const uint32_t cell = sinfo.idxs[s][i];
                const uint32_t phys = paged_write_resolve_input(cell, ubatch, token);
                if (!paged_validate_write_mapping(phys)) {
                    return false;
                }
                if (kv_paged_enabled && paged_block_size != 0 && phys != PAGED_BLOCK_INVALID) {
                    const uint32_t block = phys / paged_block_size;
                    if (block < paged_block_states.size() &&
                            paged_block_states[block] == paged_block_state::SWAPPED) {
                        paged_write_to_swapped_block += 1;
                        paged_swapped_active_violation_after_swapout_write += 1;
                        for (int32_t sid = 0; sid < ubatch->n_seq_id[token]; ++sid) {
                            if (ubatch->seq_id[token][sid] >= 0) {
                                paged_write_to_swapped_block_seq += 1;
                            }
                        }
                    }
                }
                if (!paged_ensure_write_resident(phys)) {
                    if (!has_paged_swap_error()) {
                        paged_input_setup_fatal += 1;
                        set_paged_swap_error(
                                llama_paged_swap_error_reason::INPUT_SETUP_FAILURE,
                                PAGED_BLOCK_INVALID, phys);
                    }
                    return false;
                }
                data[s*sinfo.size() + i] = offs + phys;
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                const uint32_t token = s*sinfo.size() + i;
                const uint32_t cell = sinfo.idxs[s][i];
                const uint32_t phys = paged_write_resolve_input(cell, ubatch, token);
                if (!paged_validate_write_mapping(phys)) {
                    return false;
                }
                if (kv_paged_enabled && paged_block_size != 0 && phys != PAGED_BLOCK_INVALID) {
                    const uint32_t block = phys / paged_block_size;
                    if (block < paged_block_states.size() &&
                            paged_block_states[block] == paged_block_state::SWAPPED) {
                        paged_write_to_swapped_block += 1;
                        paged_swapped_active_violation_after_swapout_write += 1;
                        for (int32_t sid = 0; sid < ubatch->n_seq_id[token]; ++sid) {
                            if (ubatch->seq_id[token][sid] >= 0) {
                                paged_write_to_swapped_block_seq += 1;
                            }
                        }
                    }
                }
                if (!paged_ensure_write_resident(phys)) {
                    if (!has_paged_swap_error()) {
                        paged_input_setup_fatal += 1;
                        set_paged_swap_error(
                                llama_paged_swap_error_reason::INPUT_SETUP_FAILURE,
                                PAGED_BLOCK_INVALID, phys);
                    }
                    return false;
                }
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + phys;
                }
            }
        }
    }

    return true;
}

bool llama_kv_cache::set_input_paged_row_idx(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (has_paged_swap_error()) {
        return false;
    }

    const bool base_timing_enabled = paged_base_timing_enabled;
    const uint64_t base_set_input_start_us = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    if (base_timing_enabled) {
        paged_base_timing_set_row_idx_calls += 1;
    }
    const bool timing_enabled = paged_resume_timing_enabled || paged_resume_timing_step_enabled;
    const uint64_t step = paged_trace_step;
    const bool idle_swap_debug_probes = paged_idle_swap_debug_probes;
    const bool idle_swap_maintenance_due =
        paged_idle_swap_every_tokens <= 1 ||
        (paged_trace_step % paged_idle_swap_every_tokens) == 0;
    const uint64_t set_input_start_us = timing_enabled ? llama_paged_timing_now_us() : 0;
    const uint64_t idle_maintenance_us_before = paged_timing_idle_maintenance_us;
    const uint64_t swap_out_us_before = paged_timing_swap_out_us;
    const uint64_t swap_out_calls_before = paged_timing_swap_out_calls;
    const uint64_t check_read_us_before = paged_timing_check_read_us;
    const uint64_t check_read_calls_before = paged_timing_check_read_calls;
    const bool defer_idle_swapout_now = paged_defer_idle_swapout_steps > 0;
    if (!dst) {
        if (timing_enabled) {
            const uint64_t set_input_us = llama_paged_timing_now_us() - set_input_start_us;
            if (paged_resume_timing_step_enabled) {
                fprintf(stderr,
                        "KV_PAGED_STEP_TIMING step=%llu set_input_us=%llu "
                        "idle_maintenance_us=0 swap_out_us=0 swap_out_calls=0 "
                        "check_read_us=0 check_read_calls=0 swapped_blocks=0 "
                        "idle_owned_blocks=0 prefetch_protected_blocks=0 remaining_prefetch_blocks=0 "
                        "defer_idle_swapout=%d swapout_deferred_blocks=0\n",
                        (unsigned long long) step,
                        (unsigned long long) set_input_us,
                        defer_idle_swapout_now ? 1 : 0);
            }
            paged_timing_set_input_us += set_input_us;
            paged_timing_set_input_calls += 1;
        }
        if (base_timing_enabled) {
            paged_base_timing_set_row_idx_total_us += llama_paged_timing_now_us() - base_set_input_start_us;
        }
        if (defer_idle_swapout_now && paged_defer_idle_swapout_steps > 0) {
            paged_defer_idle_swapout_steps -= 1;
        }
        paged_trace_step += 1;
        return true;
    }

    paged_row_idx_set_calls += 1;

    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    int32_t * data = (int32_t *) dst->data;

    const auto fail_row_idx = [&](llama_paged_swap_error_reason reason, uint32_t physical_block, uint32_t physical_cell) {
        if (reason == llama_paged_swap_error_reason::PAGED_ROW_MAPPING_INVALID) {
            paged_row_mapping_invalid_fatal += 1;
        } else if (reason == llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT) {
            paged_active_row_nonresident_fatal += 1;
        } else if (reason == llama_paged_swap_error_reason::INPUT_SETUP_FAILURE) {
            paged_input_setup_fatal += 1;
        }
        set_paged_swap_error(reason, physical_block, physical_cell);
        return false;
    };

#if defined(LLAMA_KV_REFAULT_TRACE_SUPPORTED)
    // Stage 7D-A: publish the current decode step so a fault during the upcoming graph compute is
    // attributed to the right step in KV_REFAULT_TRACE lines. Cheap, only when tracing is armed.
    if (paged_refault_trace_enabled) {
        g_refault_step.store(paged_trace_step, std::memory_order_relaxed);
    }
#endif

    std::set<uint32_t> trace_read_blocks;
    std::set<uint32_t> trace_read_blocks_before_remap;

    uint32_t active_n_kv = 0;
    if (!v_cells.empty()) {
        active_n_kv = std::min<uint32_t>(v_cells[0].used_max_p1(), (uint32_t) dst->ne[0]);
    }

    std::bitset<LLAMA_MAX_SEQ> active_seq;
    std::array<llama_pos, LLAMA_MAX_SEQ> active_seq_pos_max;
    active_seq_pos_max.fill(std::numeric_limits<llama_pos>::min());
    const char * active_seq_source = "none";
    if (ubatch) {
        active_seq_source = "ubatch_seq_id";
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            for (int32_t s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch->seq_id[i][s];
                if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
                    active_seq.set(seq_id);
                    active_seq_pos_max[seq_id] = std::max(active_seq_pos_max[seq_id], ubatch->pos[i]);
                }
            }
        }
    }

    if (base_timing_enabled) {
        paged_base_timing_getenv_calls += 1;
    }
    const bool nonidentity_probe = paged_nonidentity_probe_requested;
    paged_nonidentity_probe_enabled = nonidentity_probe;

    // Stage 10-E-F: idle swap execution no longer depends on the idle-trace print
    // flag. LLAMA_KV_PAGED_IDLE_TRACE only gates the per-step KV_PAGED_IDLE_TRACE
    // diagnostic line below; whether idle swap-out / madvise actually run is decided
    // here from the functional prerequisites alone.
    const bool idle_swap_ready =
        paged_idle_swap_requested &&
        paged_swap_enabled &&
        kv_swap_store &&
        nonidentity_probe;
    paged_idle_swap_enabled = idle_swap_ready;
    if (paged_idle_swap_requested && !idle_swap_ready && !paged_idle_swap_warned) {
        LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED_IDLE_SWAP=1 requires LLAMA_KV_PAGED_SWAP=1, "
                "LLAMA_KV_PAGED_GATHER_NONIDENTITY=1, and a backing store; "
                "idle swap disabled for this run\n", __func__);
        paged_idle_swap_warned = true;
    }

    const bool idle_swap_madvise_ready = idle_swap_ready && paged_idle_swap_madvise_requested;
    paged_idle_swap_madvise_enabled = idle_swap_madvise_ready;
    if (paged_idle_swap_madvise_requested && !idle_swap_madvise_ready && !paged_idle_swap_madvise_warned) {
        LLAMA_LOG_WARN("%s: LLAMA_KV_PAGED_IDLE_SWAP_MADVISE=1 requires idle swap ready "
                "(LLAMA_KV_PAGED_IDLE_SWAP=1, LLAMA_KV_PAGED_SWAP=1, LLAMA_KV_PAGED_GATHER_NONIDENTITY=1, "
                "and a backing store); idle swap madvise disabled for this run\n",
                __func__);
        paged_idle_swap_madvise_warned = true;
    }

    const bool swapped_redirect_probe_ready =
        paged_block_size != 0 &&
        paged_n_blocks != 0 &&
        !v_cells.empty() &&
        paged_block_states.size() == paged_n_blocks;

    struct active_row_visibility {
        bool in_read_window = false;
        bool logical_seq_has = false;
        bool phys_seq_has = false;
        bool unmasked_for_active = false;
        bool required_by_active = false;
    };

    const auto get_active_row_visibility = [&](uint32_t logical_cell, uint32_t phys_cell, bool collect_phys_seq_has) {
        active_row_visibility result;
        if (!active_seq.any() || logical_cell >= active_n_kv || v_cells.empty()) {
            return result;
        }

        const auto & cells = v_cells[0];
        if (cells.is_empty(logical_cell)) {
            return result;
        }

        result.in_read_window = true;
        const llama_pos p0 = cells.pos_get(logical_cell);
        for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
            if (active_seq.test(seq_id) && cells.seq_has(logical_cell, seq_id)) {
                result.logical_seq_has = true;
                const llama_pos p1 = active_seq_pos_max[seq_id];
                if (p1 == std::numeric_limits<llama_pos>::min() ||
                        (p0 <= p1 && !llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1))) {
                    result.unmasked_for_active = true;
                }
            }
            if (collect_phys_seq_has &&
                    active_seq.test(seq_id) &&
                    phys_cell < cells.size() &&
                    !cells.is_empty(phys_cell) &&
                    cells.seq_has(phys_cell, seq_id)) {
                result.phys_seq_has = true;
            }
        }

        result.required_by_active = result.logical_seq_has && result.unmasked_for_active;
        return result;
    };

    const auto logical_row_required_by_mapping_fault_target = [&](uint32_t logical_cell) {
        if (paged_test_mapping_fault_.target_seq < 0 ||
                paged_test_mapping_fault_.target_seq >= LLAMA_MAX_SEQ ||
                !active_seq.test(paged_test_mapping_fault_.target_seq) ||
                logical_cell >= active_n_kv ||
                v_cells.empty()) {
            return false;
        }

        const auto & cells = v_cells[0];
        if (cells.is_empty(logical_cell) ||
                !cells.seq_has(logical_cell, paged_test_mapping_fault_.target_seq)) {
            return false;
        }

        const llama_pos p0 = cells.pos_get(logical_cell);
        const llama_pos p1 = active_seq_pos_max[paged_test_mapping_fault_.target_seq];
        return p1 == std::numeric_limits<llama_pos>::min() ||
            (p0 <= p1 && !llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1));
    };

    if (swapped_redirect_probe_ready && active_seq.any()) {
        const uint64_t t_active_visible = base_timing_enabled ? llama_paged_timing_now_us() : 0;
        if (base_timing_enabled) {
            paged_base_timing_cells_scanned += active_n_kv;
        }
        const auto & cells = v_cells[0];
        std::set<uint32_t> active_visible_swapped_blocks;
        uint64_t active_visible_swapped_rows = 0;
        for (uint32_t cell = 0; cell < active_n_kv; ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }

            const uint32_t phys = paged_resolve(cell);
            if (phys == PAGED_BLOCK_INVALID) {
                continue;
            }

            const uint32_t block = phys / paged_block_size;
            if (block < paged_block_states.size() &&
                    paged_block_states[block] == paged_block_state::SWAPPED &&
                    get_active_row_visibility(cell, phys, false).required_by_active) {
                active_visible_swapped_rows += 1;
                active_visible_swapped_blocks.insert(block);
            }
        }

        if (!active_visible_swapped_blocks.empty()) {
            paged_swapped_active_violation_rows += active_visible_swapped_rows;
            paged_swapped_active_violation_blocks += active_visible_swapped_blocks.size();
            paged_swapped_active_violation_resolve_to_swapped += active_visible_swapped_blocks.size();
            paged_swapped_active_visible_restore_rows += active_visible_swapped_rows;
            for (const uint32_t block : active_visible_swapped_blocks) {
                if (paged_swap_in_block(
                                block,
                                true,
                                llama_paged_swap_error_reason::ACTIVE_VISIBLE_RESTORE_FAILURE) &&
                    paged_block_states[block] == paged_block_state::RESIDENT) {
                paged_swapped_active_visible_restore_blocks += 1;
                paged_active_restore_from_swapped_blocks += 1;
            } else {
                    if (!has_paged_swap_error()) {
                        paged_active_row_nonresident_fatal += 1;
                        set_paged_swap_error(
                                llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT,
                                block, block * paged_block_size);
                    }
                    return false;
                }
            }
        }
        if (base_timing_enabled) {
            paged_base_timing_active_visible_us += llama_paged_timing_now_us() - t_active_visible;
        }
    }

    uint32_t dummy_phys = PAGED_BLOCK_INVALID;
    std::vector<uint8_t> nonidentity_cold_blocks;
    uint64_t t_nonidentity_probe = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    bool nonidentity_probe_timed = false;
    if (nonidentity_probe &&
            paged_block_size != 0 &&
            paged_n_blocks != 0 &&
            !v_cells.empty() &&
            paged_block_states.size() == paged_n_blocks &&
            active_seq.any()) {
        nonidentity_probe_timed = true;
        const auto & cells = v_cells[0];

        if (base_timing_enabled) {
            paged_base_timing_cells_scanned += active_n_kv;
        }
        for (uint32_t cell = 0; cell < active_n_kv; ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }

            bool visible_to_active_seq = false;
            for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                if (active_seq.test(seq_id) && cells.seq_has(cell, seq_id)) {
                    visible_to_active_seq = true;
                    break;
                }
            }
            if (!visible_to_active_seq) {
                continue;
            }

            const uint32_t phys = paged_resolve(cell);
            if (phys == PAGED_BLOCK_INVALID) {
                continue;
            }

            const uint32_t block = phys / paged_block_size;
            if (block < paged_block_states.size() &&
                    paged_block_states[block] == paged_block_state::RESIDENT) {
                dummy_phys = phys;
                paged_dummy_candidate_resident += 1;
                break;
            }
        }

        std::vector<std::bitset<LLAMA_MAX_SEQ>> block_seq((size_t) paged_n_blocks);
        if (base_timing_enabled) {
            paged_base_timing_cells_scanned += cells.used_max_p1();
        }
        for (uint32_t cell = 0; cell < cells.used_max_p1(); ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }

            const uint32_t phys = paged_resolve(cell);
            if (phys == PAGED_BLOCK_INVALID) {
                continue;
            }

            const uint32_t block = phys / paged_block_size;
            if (block >= paged_n_blocks) {
                continue;
            }

            auto & owner = block_seq[block];
            for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                if (cells.seq_has(cell, seq_id)) {
                    owner.set(seq_id);
                }
            }
        }

        nonidentity_cold_blocks.assign((size_t) paged_n_blocks, 0);
        if (base_timing_enabled) {
            paged_base_timing_blocks_scanned += block_seq.size();
        }
        for (uint32_t block = 0; block < block_seq.size(); ++block) {
            const auto & owner = block_seq[block];
            if (owner.none() || owner.count() != 1) {
                continue;
            }

            const bool has_active_seq = (owner & active_seq).any();
            const bool only_seen_idle_seq = ((owner & paged_idle_seq_seen) == owner) && !has_active_seq;
            if (only_seen_idle_seq) {
                nonidentity_cold_blocks[block] = 1;
            }
        }
    }

    if (dummy_phys == PAGED_BLOCK_INVALID && swapped_redirect_probe_ready && active_seq.any()) {
        nonidentity_probe_timed = true;
        if (base_timing_enabled) {
            paged_base_timing_cells_scanned += active_n_kv;
        }
        const auto & cells = v_cells[0];
        for (uint32_t cell = 0; cell < active_n_kv; ++cell) {
            if (cells.is_empty(cell)) {
                continue;
            }

            bool visible_to_active_seq = false;
            for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                if (active_seq.test(seq_id) && cells.seq_has(cell, seq_id)) {
                    visible_to_active_seq = true;
                    break;
                }
            }
            if (!visible_to_active_seq) {
                continue;
            }

            const uint32_t phys = paged_resolve(cell);
            if (phys == PAGED_BLOCK_INVALID ||
                    phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
                continue;
            }

            const uint32_t block = phys / paged_block_size;
            if (block < paged_block_states.size() &&
                    paged_block_states[block] == paged_block_state::RESIDENT) {
                dummy_phys = phys;
                paged_dummy_candidate_resident += 1;
                break;
            }
        }

        // Fallback: if no RESIDENT dummy was found among active-visible cells,
        // scan fresh PENDING_WRITE cells of the current write transaction.  Such
        // a cell's physical row is a transaction-local safe dummy.
        //
        // Safety basis (do NOT rely on tensor-content or graph-internal data
        // dependencies):
        //   1. dummy rows are ONLY ever used to redirect rows that are masked /
        //      active-invisible — the caller (released/stale-PENDING_WRITE
        //      redirect below) never substitutes an active-visible row with a
        //      dummy.  Because the redirected row contributes nothing to the
        //      attention result (its mask bit clears the read), the actual K/V
        //      values at the dummy row are irrelevant.
        //   2. the dummy row only needs an accessible backing address (no
        //      MADV_DONTNEED, within the allocated KV buffer) so the gather
        //      does not fault.  A fresh PENDING_WRITE cell's physical backing
        //      is allocated and has not been MADV_DONTNEED'd, so its address is
        //      accessible.
        //   3. the gather that reads through the row index honours the current
        //      SET_ROWS / GET_ROWS graph expansion ordering already established
        //      for this context — dummy substitution happens during row-index
        //      setup, before graph compute; the subsequent GET_ROWS simply reads
        //      whatever physical row the index points at.
        //
        // NOTE: paged_note_cells only updates block/cell *metadata* (state, the
        // pending_write_cells flag, free list); it does NOT write tensor data.
        // The freshness signal used here is the per-cell pending_write_cells
        // flag set by paged_note_cells, which is valid for cells of the
        // current transaction and is cleared by paged_finish_write_transaction
        // after this graph completes.  We deliberately do NOT use the post-
        // graph finish_write_transaction ordering as a proof of gather data
        // dependency — the redirect safety rests on mask + accessible address
        // + existing graph expansion order, not on what the gather reads.
        if (dummy_phys == PAGED_BLOCK_INVALID && paged_pending_write_cells.size() > 0) {
            const uint64_t t_pending_dummy = base_timing_enabled ? llama_paged_timing_now_us() : 0;
            for (uint32_t phys = 0; phys < paged_pending_write_cells.size(); ++phys) {
                if (!paged_pending_write_cells[phys]) {
                    continue;
                }
                if (phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
                    continue;
                }
                const uint32_t block = phys / paged_block_size;
                if (block >= paged_block_states.size()) {
                    continue;
                }
                if (paged_block_states[block] == paged_block_state::PENDING_WRITE) {
                    dummy_phys = phys;
                    paged_dummy_candidate_pending_write_cell += 1;
                    break;
                }
            }
            if (base_timing_enabled) {
                paged_base_timing_nonidentity_probe_us += llama_paged_timing_now_us() - t_pending_dummy;
            }
        }
    }
    if (base_timing_enabled && nonidentity_probe_timed) {
        paged_base_timing_nonidentity_probe_us += llama_paged_timing_now_us() - t_nonidentity_probe;
    }

    uint64_t swapped_probe_rows_this_call = 0;
    uint64_t swapped_probe_swapped_rows_this_call = 0;
    uint64_t swapped_probe_resident_rows_this_call = 0;
    uint64_t swapped_probe_invalid_rows_this_call = 0;
    uint64_t swapped_blocks_at_row_idx = 0;
    std::set<uint32_t> swapped_active_visible_violation_blocks_this_call;
    if (idle_swap_debug_probes && swapped_redirect_probe_ready) {
        const uint64_t t_swapped_scan = base_timing_enabled ? llama_paged_timing_now_us() : 0;
        if (base_timing_enabled) {
            paged_base_timing_blocks_scanned += paged_n_blocks;
        }
        for (uint32_t block = 0; block < paged_n_blocks; ++block) {
            if (paged_block_states[block] == paged_block_state::SWAPPED) {
                swapped_blocks_at_row_idx += 1;
            }
        }
        if (base_timing_enabled) {
            paged_base_timing_swapped_blocks_scan_us += llama_paged_timing_now_us() - t_swapped_scan;
        }
    } else if (!swapped_redirect_probe_ready) {
        paged_swapped_redirect_probe_disabled += 1;
    }

    std::set<uint32_t> nonidentity_remapped_blocks;
    std::set<uint32_t> swapped_redirected_blocks;
    std::set<uint32_t> released_redirected_blocks;

    const uint64_t t_row_idx_fill = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    if (base_timing_enabled) {
        paged_base_timing_row_idx_entries += dst->ne[0];
    }
    for (int64_t r = 0; r < dst->ne[0]; ++r) {
        uint32_t phys = paged_resolve((uint32_t) r);
        if (phys != PAGED_BLOCK_INVALID &&
                paged_test_mapping_fault_should_inject_read((uint32_t) r) &&
                logical_row_required_by_mapping_fault_target((uint32_t) r)) {
            paged_test_mapping_fault_log_and_consume(
                    true,
                    (uint32_t) r,
                    llama_paged_swap_error_reason::PAGED_ROW_MAPPING_INVALID,
                    paged_row_mapping_invalid_fatal + 1);
            phys = PAGED_BLOCK_INVALID;
        }
        const bool phys_orig_valid =
            phys != PAGED_BLOCK_INVALID &&
            phys <= (uint32_t) std::numeric_limits<int32_t>::max();
        if (!phys_orig_valid) {
            paged_row_idx_fail += 1;
            return fail_row_idx(
                    llama_paged_swap_error_reason::PAGED_ROW_MAPPING_INVALID,
                    PAGED_BLOCK_INVALID,
                    (uint32_t) r);
        }
        const uint32_t phys_orig = phys;
        if ((paged_trace_enabled || paged_idle_trace_enabled) && paged_block_size != 0) {
            trace_read_blocks_before_remap.insert(phys_orig / paged_block_size);
        }

        uint32_t block = PAGED_BLOCK_INVALID;
        paged_block_state state = paged_block_state::RESIDENT;
        bool block_state_valid = false;
        if (paged_block_size != 0) {
            block = phys_orig / paged_block_size;
            block_state_valid = block < paged_block_states.size();
            if (block_state_valid) {
                state = paged_block_states[block];
            }
        }

        active_row_visibility visibility {};
        bool visibility_known = false;
        bool required_by_active = false;
        const auto ensure_visibility = [&]() {
            if (!visibility_known && active_seq.any() && !v_cells.empty()) {
                visibility = get_active_row_visibility((uint32_t) r, phys_orig, idle_swap_debug_probes);
                required_by_active = visibility.required_by_active;
                visibility_known = true;
            }
        };

        if (!block_state_valid || state != paged_block_state::RESIDENT) {
            ensure_visibility();
        }

        if (paged_test_force_active_release &&
                !paged_test_force_active_release_consumed &&
                paged_test_force_active_release_seq >= 0 &&
                active_seq.test(paged_test_force_active_release_seq)) {
            ensure_visibility();
            if (required_by_active && block_state_valid && state == paged_block_state::RESIDENT) {
                const uint32_t begin = block * paged_block_size;
                const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
                for (uint32_t cell = begin; cell < end; ++cell) {
                    if (cell < paged_swap_offsets.size()) {
                        paged_swap_offsets[cell] = 0;
                    }
                    if (cell < paged_swap_sizes.size()) {
                        paged_swap_sizes[cell] = 0;
                    }
                }
                paged_block_states[block] = paged_block_state::RELEASED;
                state = paged_block_state::RELEASED;
                paged_test_force_active_release_consumed = true;
                paged_test_force_active_release_triggers += 1;
                fprintf(stderr,
                        "KV_PAGED_TEST_FORCE_ACTIVE_RELEASE target_seq=%d logical_row=%lld "
                        "physical_block=%u physical_cell=%u trigger_count=%llu\n",
                        (int) paged_test_force_active_release_seq,
                        (long long) r,
                        block,
                        phys_orig,
                        (unsigned long long) paged_test_force_active_release_triggers);
            }
        }

        if (required_by_active) {
            if (paged_block_size == 0 || paged_block_states.empty()) {
                return fail_row_idx(
                        llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT,
                        PAGED_BLOCK_INVALID, phys_orig);
            }
            if (!block_state_valid) {
                return fail_row_idx(
                        llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT,
                        block, phys_orig);
            }

            const bool pending_fresh_cell = state == paged_block_state::PENDING_WRITE &&
                phys_orig < paged_pending_write_cells.size() && paged_pending_write_cells[phys_orig];
            if (state == paged_block_state::RELEASED ||
                    (state == paged_block_state::PENDING_WRITE && !pending_fresh_cell)) {
                paged_release_violation += 1;
                paged_active_release_violation += 1;
                return fail_row_idx(
                        llama_paged_swap_error_reason::ACTIVE_READ_RELEASED_BLOCK,
                        block, phys_orig);
            } else if (state == paged_block_state::SWAPPED) {
                paged_swapped_active_visible_violation += 1;
                paged_swapped_active_visible_violation_rows += 1;
                swapped_active_visible_violation_blocks_this_call.insert(block);
                if (paged_swap_in_block(
                            block,
                            true,
                            llama_paged_swap_error_reason::ACTIVE_VISIBLE_RESTORE_FAILURE) &&
                    paged_block_states[block] == paged_block_state::RESIDENT) {
                paged_swapped_active_visible_restore_blocks += 1;
                paged_active_restore_from_swapped_blocks += 1;
                state = paged_block_state::RESIDENT;
            } else {
                    if (!has_paged_swap_error()) {
                        return fail_row_idx(
                                llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT,
                                block, phys_orig);
                    }
                    return false;
                }
            } else if (state != paged_block_state::RESIDENT &&
                    state != paged_block_state::PENDING_WRITE) {
                return fail_row_idx(
                        llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT,
                        block, phys_orig);
            }
        }

        // Stage 7C-E: SWAPPED-block redirect. Highest priority. A row whose physical block is
        // currently SWAPPED must not keep its real physical row index, otherwise the decode
        // graph's ggml_get_rows(k2d/v2d, row_idx) faults the madvise'd pages back resident
        // (the Stage 7C-D refault source). If the row is genuinely not visible to / needed by
        // the active seq, redirect it to a resident dummy row. This does NOT change block state
        // (no swap-in, stays SWAPPED) and runs independently of the cold/owner.count()==1 path.
        bool swapped_handled = false;
        bool released_handled = false;
        // RELEASED blocks: redirect to dummy row (no valid backing).
        // PENDING_WRITE blocks: cells that are being freshly written in the
        // current transaction (paged_pending_write_cells flag set) have valid
        // backing memory — use real physical row. Other PENDING_WRITE cells
        // (not in current write transaction) were RELEASED→MADV_DONTNEED'd
        // before the PENDING_WRITE transition and contain garbage — redirect
        // to dummy (same as RELEASED).
        if (block_state_valid && state == paged_block_state::RELEASED) {
            ensure_visibility();
            if (required_by_active) {
                paged_active_row_dummy_redirect_blocked += 1;
                return fail_row_idx(
                        llama_paged_swap_error_reason::ACTIVE_READ_RELEASED_BLOCK,
                        block, phys_orig);
            }
            if (dummy_phys == PAGED_BLOCK_INVALID ||
                    dummy_phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
                paged_released_redirect_no_dummy += 1;
                return fail_row_idx(
                        llama_paged_swap_error_reason::INPUT_SETUP_FAILURE,
                        block, phys_orig);
            }
            phys = dummy_phys;
            paged_released_redirect_rows += 1;
            released_redirected_blocks.insert(block);
            released_handled = true;
        } else if (block_state_valid && state == paged_block_state::PENDING_WRITE &&
                   !required_by_active) {
            // PENDING_WRITE rows not required by active seq: check whether
            // this cell is being freshly written in the current transaction.
            const bool pending_fresh_cell =
                phys_orig < paged_pending_write_cells.size() &&
                paged_pending_write_cells[phys_orig];
            if (!pending_fresh_cell) {
                // Cell was RELEASED→MADV_DONTNEED'd before PENDING_WRITE
                // transition and is NOT being rewritten in this transaction.
                // Must redirect to dummy (content is garbage).
                ensure_visibility();
                if (required_by_active) {
                    paged_active_row_dummy_redirect_blocked += 1;
                    return fail_row_idx(
                            llama_paged_swap_error_reason::ACTIVE_READ_RELEASED_BLOCK,
                            block, phys_orig);
                }
                if (dummy_phys == PAGED_BLOCK_INVALID ||
                        dummy_phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
                    paged_released_redirect_no_dummy_pending_write += 1;
                    return fail_row_idx(
                            llama_paged_swap_error_reason::INPUT_SETUP_FAILURE,
                            block, phys_orig);
                }
                phys = dummy_phys;
                paged_released_redirect_rows += 1;
                released_redirected_blocks.insert(block);
                released_handled = true;
            }
            // else: pending_fresh_cell — cell is being written in current
            // transaction; physical row has valid backing → keep real phys.
        }
        if (!released_handled && swapped_redirect_probe_ready) {
            if (idle_swap_debug_probes) {
                swapped_probe_rows_this_call += 1;
            }
            if (!phys_orig_valid || !block_state_valid) {
                if (idle_swap_debug_probes) {
                    swapped_probe_invalid_rows_this_call += 1;
                }
            } else if (state == paged_block_state::SWAPPED) {
                ensure_visibility();
                if (idle_swap_debug_probes) {
                    swapped_probe_swapped_rows_this_call += 1;
                }
                if (idle_swap_debug_probes && visibility.logical_seq_has) {
                    paged_swapped_active_visible_logical_seq_has += 1;
                }
                if (idle_swap_debug_probes && visibility.phys_seq_has) {
                    paged_swapped_active_visible_phys_seq_has += 1;
                }
                if (idle_swap_debug_probes) {
                    if (visibility.in_read_window) {
                        paged_swapped_active_visible_in_read_window += 1;
                    } else {
                        paged_swapped_active_visible_not_in_read_window += 1;
                    }
                }
                if (idle_swap_debug_probes && visibility.logical_seq_has && !visibility.unmasked_for_active) {
                    paged_swapped_active_visible_masked += 1;
                } else if (idle_swap_debug_probes && required_by_active) {
                    paged_swapped_active_visible_unmasked += 1;
                }

                if (required_by_active) {
                    paged_active_row_dummy_redirect_blocked += 1;
                    return fail_row_idx(
                            llama_paged_swap_error_reason::ACTIVE_ROW_NOT_RESIDENT,
                            block, phys_orig);
                } else if (dummy_phys == PAGED_BLOCK_INVALID ||
                        dummy_phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
                    paged_swapped_redirect_skip_no_dummy += 1;
                    paged_no_dummy_restore_sync += 1;
                    if (paged_swap_in_block(
                                block,
                                true,
                                llama_paged_swap_error_reason::NO_DUMMY_RESTORE_FAILURE) &&
                            paged_block_states[block] == paged_block_state::RESIDENT) {
                        paged_swapped_active_visible_restore_blocks += 1;
                        paged_active_restore_from_swapped_blocks += 1;
                    } else {
                        if (!has_paged_swap_error()) {
                            return fail_row_idx(
                                    llama_paged_swap_error_reason::NO_DUMMY_RESTORE_FAILURE,
                                    block, phys_orig);
                        }
                        return false;
                    }
                    swapped_handled = true;
                } else {
                    phys = dummy_phys;
                    swapped_redirected_blocks.insert(block);
                    paged_swapped_redirect_rows += 1;
                    swapped_handled = true;
                }
            } else if (idle_swap_debug_probes && state == paged_block_state::RESIDENT) {
                swapped_probe_resident_rows_this_call += 1;
            }
        }

        if (!swapped_handled &&
                nonidentity_probe && paged_block_size != 0 && !nonidentity_cold_blocks.empty()) {

            const uint32_t block = phys_orig / paged_block_size;
            if (block < nonidentity_cold_blocks.size() && nonidentity_cold_blocks[block]) {
                const auto & cells = v_cells[0];
                bool masked = false;

                if ((uint32_t) r >= active_n_kv || cells.is_empty((uint32_t) r)) {
                    masked = true;
                } else {
                    masked = true;
                    for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                        if (active_seq.test(seq_id) && cells.seq_has((uint32_t) r, seq_id)) {
                            masked = false;
                            break;
                        }
                    }
                }

                if (!masked) {
                    paged_nonidentity_skip_not_masked += 1;
                } else if (block >= paged_block_states.size() ||
                        (paged_block_states[block] != paged_block_state::RESIDENT &&
                         !(idle_swap_ready && paged_block_states[block] == paged_block_state::SWAPPED))) {
                    paged_nonidentity_skip_not_resident += 1;
                } else if (dummy_phys == PAGED_BLOCK_INVALID ||
                        dummy_phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
                    paged_nonidentity_skip_no_dummy += 1;
                    if (block < paged_block_states.size() &&
                            paged_block_states[block] == paged_block_state::SWAPPED) {
                        paged_no_dummy_restore_sync += 1;
                        if (paged_swap_in_block(
                                    block,
                                    true,
                                    llama_paged_swap_error_reason::NO_DUMMY_RESTORE_FAILURE) &&
                                paged_block_states[block] == paged_block_state::RESIDENT) {
                            paged_active_restore_from_swapped_blocks += 1;
                        } else {
                            if (!has_paged_swap_error()) {
                                return fail_row_idx(
                                        llama_paged_swap_error_reason::NO_DUMMY_RESTORE_FAILURE,
                                        block, phys_orig);
                            }
                            return false;
                        }
                    }
                } else {
                    phys = dummy_phys;
                    nonidentity_remapped_blocks.insert(block);
                    paged_nonidentity_remap_rows += 1;
                }
            }
        }

        const uint64_t t_check_read = base_timing_enabled ? llama_paged_timing_now_us() : 0;
        const bool read_resident_ok = paged_check_read_resident(phys, required_by_active);
        if (base_timing_enabled) {
            paged_base_timing_check_read_resident_us += llama_paged_timing_now_us() - t_check_read;
            paged_base_timing_check_read_resident_calls += 1;
        }
        if (!read_resident_ok) {
            if (!has_paged_swap_error()) {
                return fail_row_idx(
                        llama_paged_swap_error_reason::INPUT_SETUP_FAILURE,
                        PAGED_BLOCK_INVALID, phys);
            }
            return false;
        }
        if (phys != (uint32_t) r) {
            paged_row_idx_changed += 1;
        }
        if ((paged_trace_enabled || paged_idle_trace_enabled) && paged_block_size != 0) {
            trace_read_blocks.insert(phys / paged_block_size);
        }
        data[r] = (int32_t) phys;
    }
    if (base_timing_enabled) {
        paged_base_timing_row_idx_fill_us += llama_paged_timing_now_us() - t_row_idx_fill;
    }
    if (idle_swap_debug_probes) {
        paged_swapped_redirect_probe_rows += swapped_probe_rows_this_call;
        paged_swapped_redirect_probe_swapped_rows += swapped_probe_swapped_rows_this_call;
        paged_swapped_redirect_probe_resident_rows += swapped_probe_resident_rows_this_call;
        paged_swapped_redirect_probe_invalid_rows += swapped_probe_invalid_rows_this_call;
        if (swapped_blocks_at_row_idx > 0 && swapped_probe_swapped_rows_this_call == 0) {
            paged_swapped_redirect_probe_state_mismatch += 1;
        }
    }

    paged_nonidentity_remap_blocks += nonidentity_remapped_blocks.size();
    paged_swapped_redirect_blocks += swapped_redirected_blocks.size();
    paged_released_redirect_blocks += released_redirected_blocks.size();
    paged_swapped_active_visible_violation_blocks += swapped_active_visible_violation_blocks_this_call.size();

    uint64_t prefetch_protected_blocks_this_call = 0;
    uint64_t swapout_deferred_blocks_this_call = 0;
    // Stage 10-E-F: idle maintenance (block analysis, safe-swap-candidate computation,
    // prefetch-protection / defer checks, idle swap-out and madvise) is an execution path
    // and must run whenever idle swap is requested OR diagnostics are on. The per-step
    // KV_PAGED_IDLE_TRACE print is gated separately, inside this block.
    const bool idle_maintenance_active = paged_idle_trace_enabled || paged_idle_swap_requested;
    if (idle_maintenance_active) {
        const uint64_t idle_maintenance_start_us = timing_enabled ? llama_paged_timing_now_us() : 0;
        const uint64_t idle_step = paged_idle_active_seq_steps;
        const uint64_t active_seq_count = active_seq.count();
        for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
            if (!active_seq.test(seq_id)) {
                continue;
            }
            paged_idle_seq_seen.set(seq_id);
            paged_idle_seq_last_active_step[seq_id] = idle_step;
        }

        const uint64_t seen_seq_count = paged_idle_seq_seen.count();
        const uint64_t idle_seq_count = seen_seq_count >= active_seq_count ? seen_seq_count - active_seq_count : 0;
        paged_idle_active_seq_steps += 1;
        paged_idle_seq_seen_count = seen_seq_count;
        paged_idle_active_seq_count_last = active_seq_count;
        paged_idle_idle_seq_count_last = idle_seq_count;
        if (active_seq_count == 0) {
            paged_idle_active_seq_empty += 1;
        }
        if (active_seq_count > paged_idle_active_seq_count_max) {
            paged_idle_active_seq_count_max = active_seq_count;
        }

        std::string active_seq_csv;
        std::string last_active_csv;
        if (idle_swap_debug_probes && paged_idle_trace_enabled) {
            for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                if (!active_seq.test(seq_id)) {
                    continue;
                }
                if (!active_seq_csv.empty()) {
                    active_seq_csv += ',';
                }
                active_seq_csv += std::to_string(seq_id);
            }

            for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                if (!paged_idle_seq_seen.test(seq_id)) {
                    continue;
                }
                if (!last_active_csv.empty()) {
                    last_active_csv += ',';
                }
                last_active_csv += std::to_string(seq_id);
                last_active_csv += ':';
                last_active_csv += std::to_string(paged_idle_seq_last_active_step[seq_id]);
            }
        }

        uint64_t non_empty_blocks = 0;
        uint64_t single_seq_blocks = 0;
        uint64_t multi_seq_blocks = 0;
        uint64_t blocks_with_active_seq = 0;
        uint64_t blocks_without_active_seq = 0;
        uint64_t cold_candidates = 0;
        uint64_t cold_in_read_window_before = 0;
        uint64_t cold_in_read_window = 0;
        uint64_t cold_not_in_read_window = 0;
        uint64_t skip_mixed_active = 0;
        uint64_t safe_swap_candidates = 0;
        uint64_t nonidentity_remapped_this_round = 0;
        uint64_t per_block_bytes = 0;
        uint64_t idle_swap_out_attempts_this_call = 0;
        const uint64_t madvise_calls_before_cov = paged_swap_madvise_calls;
        // Stage 5E-1: mincore window locals, hoisted so the post-loop latch can read them.
        uint64_t mincore_resident_before_loop = 0;
        uint64_t mincore_madvise_calls_before = paged_swap_madvise_calls;
        if (idle_swap_maintenance_due && paged_block_size != 0 && paged_n_blocks != 0 && !v_cells.empty()) {
            for (const auto & layer : layers) {
                if (!layer.k_stream.empty() && layer.k_stream[0]) {
                    per_block_bytes += (uint64_t) layer.k_stream[0]->nb[1];
                }
                if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
                    per_block_bytes += (uint64_t) layer.v_stream[0]->nb[1];
                }
            }
            per_block_bytes *= paged_block_size;

            const auto & cells = v_cells[0];
            std::vector<std::bitset<LLAMA_MAX_SEQ>> block_seq((size_t) paged_n_blocks);
            // Stage 7C-G: per-block "true active-needed unmasked" bit, computed from the
            // physical cells that actually live in each block (not from the full-prefix read
            // window). A block is hard-protected from swap-out only if some physical cell in it
            // is owned by an active seq AND is unmasked vs that seq's active_seq_pos_max. This
            // is the real correctness invariant; full-prefix read-window membership is not.
            std::vector<uint8_t> block_active_unmasked((size_t) paged_n_blocks, 0);
            for (uint32_t cell = 0; cell < cells.used_max_p1(); ++cell) {
                if (cells.is_empty(cell)) {
                    continue;
                }

                const uint32_t phys = paged_resolve(cell);
                if (phys == PAGED_BLOCK_INVALID) {
                    continue;
                }

                const uint32_t block = phys / paged_block_size;
                if (block >= paged_n_blocks) {
                    continue;
                }

                auto & owner = block_seq[block];
                const llama_pos p0 = cells.pos_get(cell);
                for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                    if (cells.seq_has(cell, seq_id)) {
                        owner.set(seq_id);
                        if (active_seq.test(seq_id)) {
                            const llama_pos p1 = active_seq_pos_max[seq_id];
                            if (p1 == std::numeric_limits<llama_pos>::min() ||
                                    (p0 <= p1 && !llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1))) {
                                block_active_unmasked[block] = 1;
                            }
                        }
                    }
                }
            }

            // Stage 5E-1: read-only KV resident sampling. prefill snapshot is recorded once at
            // the first idle-trace step that reaches here. before_madvise/after_madvise latch the
            // resident level around the step that actually performs idle swap-out + madvise (gated
            // on paged_swap_madvise_calls growing), so the printed window reflects a real madvise
            // drop rather than a later step where no block was advised.
            mincore_madvise_calls_before = paged_swap_madvise_calls;
            if (paged_mincore_enabled) {
                mincore_resident_before_loop = paged_sample_mincore();
                if (!paged_mincore_prefill_set) {
                    paged_mincore_prefill_resident_bytes = mincore_resident_before_loop;
                    paged_mincore_prefill_set = true;
                }
            }

            bool idle_swap_budget_exhausted = false;
            for (uint32_t block = 0; block < block_seq.size(); ++block) {
                if (idle_swap_budget_exhausted) {
                    break;
                }
                const auto & owner = block_seq[block];
                if (owner.none()) {
                    continue;
                }

                non_empty_blocks += 1;
                const uint64_t owner_count = owner.count();
                const bool has_active_seq = (owner & active_seq).any();
                if (owner_count == 1) {
                    single_seq_blocks += 1;
                } else if (owner_count > 1) {
                    multi_seq_blocks += 1;
                }

                if (has_active_seq) {
                    blocks_with_active_seq += 1;
                } else {
                    blocks_without_active_seq += 1;
                }

                if (owner_count > 1 && has_active_seq) {
                    skip_mixed_active += 1;
                }

                if ((owner & paged_prefetch_protected_seq).any()) {
                    prefetch_protected_blocks_this_call += 1;
                }

                if (has_active_seq) {
                    paged_swap_out_skip_active_owned_block += 1;
                    if (trace_read_blocks_before_remap.find(block) != trace_read_blocks_before_remap.end() ||
                            trace_read_blocks.find(block) != trace_read_blocks.end()) {
                        // Stage 7C-G: diagnostic only. This counts active-owned blocks that
                        // also appear in the full-prefix read window. It does NOT gate
                        // swap-out (the only swap-out path is the idle-only branch below);
                        // it stays for continuity with earlier-stage trace comparisons.
                        paged_swap_out_skip_active_visible_block += 1;
                    }
                }

                const bool mixed = owner_count != 1;
                const bool only_seen_idle_seq = ((owner & paged_idle_seq_seen) == owner) && !has_active_seq;
                // Stage 7C-G: count idle-only blocks that are currently SWAPPED (kept
                // non-resident). This is the "benefit retained" signal: it should track
                // swapped_blocks once the gate is recovered.
                if (only_seen_idle_seq && !block_active_unmasked[block] &&
                        block < paged_block_states.size() &&
                        paged_block_states[block] == paged_block_state::SWAPPED) {
                    paged_idle_only_swapped_blocks += 1;
                }
                if (mixed || !only_seen_idle_seq) {
                    continue;
                }

                cold_candidates += 1;

                const bool in_read_window_before = trace_read_blocks_before_remap.find(block) != trace_read_blocks_before_remap.end();
                if (in_read_window_before) {
                    cold_in_read_window_before += 1;
                }

                const bool in_read_window = trace_read_blocks.find(block) != trace_read_blocks.end();
                if (in_read_window) {
                    cold_in_read_window += 1;
                    continue;
                }

                cold_not_in_read_window += 1;
                if (block < paged_block_states.size() &&
                        paged_block_states[block] == paged_block_state::RESIDENT) {
                    safe_swap_candidates += 1;
                    if (nonidentity_remapped_blocks.find(block) != nonidentity_remapped_blocks.end()) {
                        nonidentity_remapped_this_round += 1;
                    }
                    if (paged_idle_swap_requested) {
                        paged_idle_swap_candidates += 1;
                        paged_swap_out_candidate_blocks += 1;
                        const bool in_read_window =
                            trace_read_blocks_before_remap.find(block) != trace_read_blocks_before_remap.end() ||
                            trace_read_blocks.find(block) != trace_read_blocks.end();
                        // Stage 6C-1A: never swap out a block owned (even partially) by a
                        // prefetch-protected / resume-pending seq, else interleaved prefetch
                        // gets undone by this same idle gate within the active-decode window.
                        if ((owner & paged_prefetch_protected_seq).any()) {
                            paged_idle_swap_skip_protected += 1;
                        } else if (!idle_swap_ready ||
                                nonidentity_remapped_blocks.find(block) == nonidentity_remapped_blocks.end()) {
                            paged_idle_swap_skip_not_remapped += 1;
                        } else if (paged_block_states[block] != paged_block_state::RESIDENT) {
                            paged_idle_swap_skip_not_resident += 1;
                        } else if ((owner & active_seq).any()) {
                            // Defensive: owner aggregates physical-cell seq ownership for this
                            // block, so an active bit here means a physical cell really belongs
                            // to an active seq. (only_seen_idle_seq already excludes this, so it
                            // should be 0; kept as a true-active-owned hard skip + telemetry.)
                            paged_swap_out_skip_active_owned_block += 1;
                            paged_swap_out_skip_true_active_owned += 1;
                            paged_swapped_active_violation_block_had_active_owner_at_swapout += 1;
                        } else if (block_active_unmasked[block]) {
                            // Stage 7C-G: the only correctness-mandated hard skip. A physical
                            // cell in this block is owned by an active seq AND unmasked vs that
                            // seq's pos_max, so swapping it out would hide an active-needed row.
                            paged_swap_out_skip_true_active_unmasked += 1;
                            paged_swap_out_skip_active_visible_block += 1;
                        } else {
                            if (paged_idle_swap_min_idle_steps > 0) {
                                bool old_enough = true;
                                for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
                                    if (!owner.test(seq_id)) {
                                        continue;
                                    }
                                    if (!paged_idle_seq_seen.test(seq_id) ||
                                            idle_step < paged_idle_seq_last_active_step[seq_id] ||
                                            idle_step - paged_idle_seq_last_active_step[seq_id] < paged_idle_swap_min_idle_steps) {
                                        old_enough = false;
                                        break;
                                    }
                                }
                                if (!old_enough) {
                                    paged_idle_swap_skip_min_idle += 1;
                                    continue;
                                }
                            }
                            // Stage 7C-G: idle-only block with no true active-needed unmasked
                            // cell. 7C-F skipped this whenever `in_read_window` was set, which
                            // covered the entire full-prefix gather and killed all swap-out. The
                            // full-prefix read window is NOT a correctness signal: those idle
                            // rows are masked / redirected to the dummy row in the row_idx pass,
                            // so the SWAPPED pages are never faulted back. Swap it out.
                            if (in_read_window) {
                                paged_swap_out_skip_fullprefix_read_window_only += 1;
                            }
                            if (defer_idle_swapout_now) {
                                paged_idle_swap_skip_deferred += 1;
                                swapout_deferred_blocks_this_call += 1;
                            } else {
                                if (paged_idle_swap_max_blocks_per_step > 0 &&
                                        idle_swap_out_attempts_this_call >= paged_idle_swap_max_blocks_per_step) {
                                    idle_swap_budget_exhausted = true;
                                    break;
                                }
                                idle_swap_out_attempts_this_call += 1;
                                const uint64_t before = paged_swap_out_calls;
                                paged_swap_out_block(block, idle_swap_madvise_ready);
                                if (paged_swap_out_calls > before) {
                                    paged_idle_swap_out_calls += 1;
                                    paged_swap_out_allowed_blocks += 1;
                                }
                                if (paged_idle_swap_max_blocks_per_step > 0 &&
                                        idle_swap_out_attempts_this_call >= paged_idle_swap_max_blocks_per_step) {
                                    idle_swap_budget_exhausted = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Stage 5E-1: if this step actually advised any block away (madvise_calls grew), latch the
        // before/after resident snapshots for the madvise window. Steps that swapped nothing leave
        // the prior window intact, so the printed madvise_drop reflects a real release.
        if (paged_mincore_enabled && paged_swap_madvise_calls > mincore_madvise_calls_before) {
            paged_mincore_before_madvise_resident_bytes = mincore_resident_before_loop;
            paged_mincore_before_madvise_set = true;
            paged_mincore_after_madvise_resident_bytes = paged_sample_mincore();
        }
        paged_idle_non_empty_blocks = non_empty_blocks;
        paged_idle_single_seq_blocks = single_seq_blocks;
        paged_idle_multi_seq_blocks = multi_seq_blocks;
        paged_idle_blocks_with_active_seq = blocks_with_active_seq;
        paged_idle_blocks_without_active_seq = blocks_without_active_seq;
        paged_idle_cold_candidates = cold_candidates;
        paged_idle_read_window_blocks = trace_read_blocks.size();
        paged_idle_cold_in_read_window = cold_in_read_window;
        paged_idle_cold_not_in_read_window = cold_not_in_read_window;
        paged_idle_skip_mixed_active = skip_mixed_active;
        paged_idle_safe_swap_candidates = safe_swap_candidates;
        if (idle_swap_debug_probes && paged_swap_madvise_calls > madvise_calls_before_cov) {
            paged_cov_idle_owned_blocks = cold_candidates;
            paged_cov_in_read_window_blocks = cold_in_read_window;
            paged_cov_not_in_read_window_blocks = cold_not_in_read_window;
            paged_cov_resident_safe_blocks = safe_swap_candidates;
            paged_cov_nonidentity_remapped_blocks = nonidentity_remapped_this_round;
            paged_cov_idle_owned_bytes = cold_candidates * per_block_bytes;
            paged_cov_in_read_window_bytes = cold_in_read_window * per_block_bytes;
            paged_cov_resident_safe_bytes = safe_swap_candidates * per_block_bytes;
            paged_cov_nonidentity_remapped_bytes = nonidentity_remapped_this_round * per_block_bytes;
        }
        paged_nonidentity_cold_in_read_window_before = cold_in_read_window_before;
        paged_nonidentity_cold_in_read_window_after = cold_in_read_window;
        paged_nonidentity_safe_candidates_after = safe_swap_candidates;

        // Stage 10-E-F: per-step diagnostic line only. Execution (swap-out, madvise,
        // counter updates, mincore snapshots above) is unconditional within this block.
        if (paged_idle_trace_enabled) {
        fprintf(stderr,
                "KV_PAGED_IDLE_TRACE step=%llu active_seq_source=%s active_seq_count=%llu "
                "idle_seq_count=%llu seen_seq_count=%llu active_seq=%s seq_last_active=%s "
                "non_empty_blocks=%llu single_seq_blocks=%llu multi_seq_blocks=%llu "
                "blocks_with_active_seq=%llu blocks_without_active_seq=%llu "
                "cold_candidates=%llu read_window_blocks=%llu cold_in_read_window=%llu "
                "cold_not_in_read_window=%llu skip_mixed_active=%llu safe_swap_candidates=%llu "
                "paged_idle_swap_enabled=%d paged_idle_swap_candidates=%llu "
                "paged_idle_swap_out_calls=%llu paged_idle_swap_skip_not_remapped=%llu "
                "paged_idle_swap_skip_not_resident=%llu paged_idle_swap_skip_protected=%llu "
                "paged_idle_swap_skip_deferred=%llu "
                "paged_nonidentity_enabled=%d paged_nonidentity_remap_rows=%llu "
                "paged_nonidentity_remap_blocks=%llu paged_nonidentity_skip_no_dummy=%llu "
                "paged_nonidentity_skip_not_masked=%llu paged_nonidentity_skip_not_resident=%llu "
                "paged_nonidentity_cold_in_read_window_before=%llu "
                "paged_nonidentity_cold_in_read_window_after=%llu "
                "paged_nonidentity_safe_candidates_after=%llu "
                "paged_swapped_redirect_rows=%llu paged_swapped_redirect_blocks=%llu "
                "paged_swapped_redirect_skip_no_dummy=%llu paged_swapped_active_visible_violation=%llu paged_swapped_redirect_probe_rows=%llu paged_swapped_redirect_probe_swapped_rows=%llu paged_swapped_redirect_probe_resident_rows=%llu paged_swapped_redirect_probe_invalid_rows=%llu paged_swapped_redirect_probe_state_mismatch=%llu paged_swapped_redirect_probe_disabled=%llu paged_swapped_active_visible_violation_rows=%llu paged_swapped_active_visible_violation_blocks=%llu paged_swapped_active_visible_logical_seq_has=%llu paged_swapped_active_visible_phys_seq_has=%llu paged_swapped_active_visible_in_read_window=%llu paged_swapped_active_visible_not_in_read_window=%llu paged_swapped_active_visible_masked=%llu paged_swapped_active_visible_unmasked=%llu paged_swapped_active_violation_rows=%llu paged_swapped_active_violation_blocks=%llu paged_swapped_active_violation_block_had_active_owner_at_swapout=%llu paged_swapped_active_violation_after_swapout_write=%llu paged_swapped_active_violation_resolve_to_swapped=%llu paged_swapped_active_visible_restore_rows=%llu paged_swapped_active_visible_restore_blocks=%llu paged_swap_out_skip_active_visible_block=%llu paged_swap_out_skip_active_owned_block=%llu paged_swap_out_candidate_blocks=%llu paged_swap_out_allowed_blocks=%llu paged_swap_out_skip_fullprefix_read_window_only=%llu paged_swap_out_skip_true_active_owned=%llu paged_swap_out_skip_true_active_unmasked=%llu paged_active_restore_from_swapped_blocks=%llu paged_idle_only_swapped_blocks=%llu paged_write_to_swapped_block=%llu paged_write_to_swapped_block_seq=%llu "
                "paged_cov_idle_owned_blocks=%llu paged_cov_in_read_window_blocks=%llu "
                "paged_cov_not_in_read_window_blocks=%llu paged_cov_resident_safe_blocks=%llu "
                "paged_cov_nonidentity_remapped_blocks=%llu "
                "paged_cov_idle_owned_bytes=%llu paged_cov_in_read_window_bytes=%llu "
                "paged_cov_resident_safe_bytes=%llu paged_cov_nonidentity_remapped_bytes=%llu "
                "paged_idle_swap_madvise_enabled=%d paged_swap_madvise_calls=%llu "
                "paged_swap_madvise_bytes=%llu paged_swap_madvise_failures=%llu "
                "paged_swap_madvise_skip_no_full_page=%llu paged_swap_madvise_skip_neighbor=%llu "
                "paged_swap_rss_before_last_kb=%llu paged_swap_rss_after_last_kb=%llu "
                "paged_swap_rss_drop_last_kb=%llu paged_swap_rss_drop_max_kb=%llu "
                "paged_swap_rss_before_first_kb=%llu paged_swap_rss_total_drop_kb=%llu "
                "paged_swap_rss_drop_sum_kb=%llu "
                "kv_mincore_enabled=%d kv_mincore_sample_calls=%llu kv_mincore_failures=%llu "
                "kv_mincore_total_bytes=%llu kv_mincore_resident_bytes=%llu "
                "kv_mincore_total_pages=%llu kv_mincore_resident_pages=%llu "
                "kv_mincore_resident_ratio_permille=%llu "
                "kv_mincore_k_total_bytes=%llu kv_mincore_k_resident_bytes=%llu "
                "kv_mincore_v_total_bytes=%llu kv_mincore_v_resident_bytes=%llu "
                "kv_mincore_prefill_resident_bytes=%llu kv_mincore_before_madvise_resident_bytes=%llu "
                "kv_mincore_after_madvise_resident_bytes=%llu kv_mincore_after_resume_resident_bytes=%llu "
                "kv_mincore_madvise_drop_bytes=%llu kv_mincore_resume_recover_bytes=%llu "
                "kv_mincore_swapped_block_count=%llu kv_mincore_swapped_total_bytes=%llu "
                "kv_mincore_swapped_resident_bytes=%llu kv_mincore_swapped_nonresident_bytes=%llu "
                "kv_mincore_swapped_resident_blocks=%llu kv_mincore_swapped_nonresident_blocks=%llu "
                "kv_mincore_swapped_resident_ratio_permille=%llu\n",
                (unsigned long long) idle_step,
                active_seq_source,
                (unsigned long long) active_seq_count,
                (unsigned long long) idle_seq_count,
                (unsigned long long) seen_seq_count,
                active_seq_csv.empty() ? "-" : active_seq_csv.c_str(),
                last_active_csv.empty() ? "-" : last_active_csv.c_str(),
                (unsigned long long) non_empty_blocks,
                (unsigned long long) single_seq_blocks,
                (unsigned long long) multi_seq_blocks,
                (unsigned long long) blocks_with_active_seq,
                (unsigned long long) blocks_without_active_seq,
                (unsigned long long) paged_idle_cold_candidates,
                (unsigned long long) paged_idle_read_window_blocks,
                (unsigned long long) paged_idle_cold_in_read_window,
                (unsigned long long) paged_idle_cold_not_in_read_window,
                (unsigned long long) paged_idle_skip_mixed_active,
                (unsigned long long) paged_idle_safe_swap_candidates,
                paged_idle_swap_enabled ? 1 : 0,
                (unsigned long long) paged_idle_swap_candidates,
                (unsigned long long) paged_idle_swap_out_calls,
                (unsigned long long) paged_idle_swap_skip_not_remapped,
                (unsigned long long) paged_idle_swap_skip_not_resident,
                (unsigned long long) paged_idle_swap_skip_protected,
                (unsigned long long) paged_idle_swap_skip_deferred,
                paged_nonidentity_probe_enabled ? 1 : 0,
                (unsigned long long) paged_nonidentity_remap_rows,
                (unsigned long long) paged_nonidentity_remap_blocks,
                (unsigned long long) paged_nonidentity_skip_no_dummy,
                (unsigned long long) paged_nonidentity_skip_not_masked,
                (unsigned long long) paged_nonidentity_skip_not_resident,
                (unsigned long long) paged_nonidentity_cold_in_read_window_before,
                (unsigned long long) paged_nonidentity_cold_in_read_window_after,
                (unsigned long long) paged_nonidentity_safe_candidates_after,
                (unsigned long long) paged_swapped_redirect_rows,
                (unsigned long long) paged_swapped_redirect_blocks,
                (unsigned long long) paged_swapped_redirect_skip_no_dummy,
                (unsigned long long) paged_swapped_active_visible_violation,
                (unsigned long long) paged_swapped_redirect_probe_rows,
                (unsigned long long) paged_swapped_redirect_probe_swapped_rows,
                (unsigned long long) paged_swapped_redirect_probe_resident_rows,
                (unsigned long long) paged_swapped_redirect_probe_invalid_rows,
                (unsigned long long) paged_swapped_redirect_probe_state_mismatch,
                (unsigned long long) paged_swapped_redirect_probe_disabled,
                (unsigned long long) paged_swapped_active_visible_violation_rows,
                (unsigned long long) paged_swapped_active_visible_violation_blocks,
                (unsigned long long) paged_swapped_active_visible_logical_seq_has,
                (unsigned long long) paged_swapped_active_visible_phys_seq_has,
                (unsigned long long) paged_swapped_active_visible_in_read_window,
                (unsigned long long) paged_swapped_active_visible_not_in_read_window,
                (unsigned long long) paged_swapped_active_visible_masked,
                (unsigned long long) paged_swapped_active_visible_unmasked,
                (unsigned long long) paged_swapped_active_violation_rows,
                (unsigned long long) paged_swapped_active_violation_blocks,
                (unsigned long long) paged_swapped_active_violation_block_had_active_owner_at_swapout,
                (unsigned long long) paged_swapped_active_violation_after_swapout_write,
                (unsigned long long) paged_swapped_active_violation_resolve_to_swapped,
                (unsigned long long) paged_swapped_active_visible_restore_rows,
                (unsigned long long) paged_swapped_active_visible_restore_blocks,
                (unsigned long long) paged_swap_out_skip_active_visible_block,
                (unsigned long long) paged_swap_out_skip_active_owned_block,
                (unsigned long long) paged_swap_out_candidate_blocks,
                (unsigned long long) paged_swap_out_allowed_blocks,
                (unsigned long long) paged_swap_out_skip_fullprefix_read_window_only,
                (unsigned long long) paged_swap_out_skip_true_active_owned,
                (unsigned long long) paged_swap_out_skip_true_active_unmasked,
                (unsigned long long) paged_active_restore_from_swapped_blocks,
                (unsigned long long) paged_idle_only_swapped_blocks,
                (unsigned long long) paged_write_to_swapped_block,
                (unsigned long long) paged_write_to_swapped_block_seq,
                (unsigned long long) paged_cov_idle_owned_blocks,
                (unsigned long long) paged_cov_in_read_window_blocks,
                (unsigned long long) paged_cov_not_in_read_window_blocks,
                (unsigned long long) paged_cov_resident_safe_blocks,
                (unsigned long long) paged_cov_nonidentity_remapped_blocks,
                (unsigned long long) paged_cov_idle_owned_bytes,
                (unsigned long long) paged_cov_in_read_window_bytes,
                (unsigned long long) paged_cov_resident_safe_bytes,
                (unsigned long long) paged_cov_nonidentity_remapped_bytes,
                paged_idle_swap_madvise_enabled ? 1 : 0,
                (unsigned long long) paged_swap_madvise_calls,
                (unsigned long long) paged_swap_madvise_bytes,
                (unsigned long long) paged_swap_madvise_failures,
                (unsigned long long) paged_swap_madvise_skip_no_full_page,
                (unsigned long long) paged_swap_madvise_skip_neighbor,
                (unsigned long long) paged_swap_rss_before_last_kb,
                (unsigned long long) paged_swap_rss_after_last_kb,
                (unsigned long long) paged_swap_rss_drop_last_kb,
                (unsigned long long) paged_swap_rss_drop_max_kb,
                (unsigned long long) paged_swap_rss_before_first_kb,
                (unsigned long long) paged_swap_rss_total_drop_kb,
                (unsigned long long) paged_swap_rss_drop_sum_kb,
                paged_mincore_enabled ? 1 : 0,
                (unsigned long long) paged_mincore_sample_calls,
                (unsigned long long) paged_mincore_failures,
                (unsigned long long) paged_mincore_total_bytes,
                (unsigned long long) paged_mincore_resident_bytes,
                (unsigned long long) paged_mincore_total_pages,
                (unsigned long long) paged_mincore_resident_pages,
                (unsigned long long) (paged_mincore_total_pages > 0
                    ? paged_mincore_resident_pages * 1000ull / paged_mincore_total_pages : 0),
                (unsigned long long) paged_mincore_k_total_bytes,
                (unsigned long long) paged_mincore_k_resident_bytes,
                (unsigned long long) paged_mincore_v_total_bytes,
                (unsigned long long) paged_mincore_v_resident_bytes,
                (unsigned long long) paged_mincore_prefill_resident_bytes,
                (unsigned long long) paged_mincore_before_madvise_resident_bytes,
                (unsigned long long) paged_mincore_after_madvise_resident_bytes,
                (unsigned long long) paged_mincore_after_resume_resident_bytes,
                (unsigned long long) (paged_mincore_before_madvise_resident_bytes > paged_mincore_after_madvise_resident_bytes
                    ? paged_mincore_before_madvise_resident_bytes - paged_mincore_after_madvise_resident_bytes : 0),
                (unsigned long long) (paged_mincore_after_resume_resident_bytes > paged_mincore_after_madvise_resident_bytes
                    ? paged_mincore_after_resume_resident_bytes - paged_mincore_after_madvise_resident_bytes : 0),
                (unsigned long long) paged_mincore_swapped_block_count,
                (unsigned long long) paged_mincore_swapped_total_bytes,
                (unsigned long long) paged_mincore_swapped_resident_bytes,
                (unsigned long long) paged_mincore_swapped_nonresident_bytes,
                (unsigned long long) paged_mincore_swapped_resident_blocks,
                (unsigned long long) paged_mincore_swapped_nonresident_blocks,
                (unsigned long long) (paged_mincore_swapped_total_bytes > 0
                    ? paged_mincore_swapped_resident_bytes * 1000ull / paged_mincore_swapped_total_bytes : 0));
        }
        if (timing_enabled) {
            paged_timing_idle_maintenance_us += llama_paged_timing_now_us() - idle_maintenance_start_us;
            paged_timing_idle_maintenance_calls += 1;
        }
    }

    if (paged_trace_enabled) {
        // Build the *active* read set: only logical cells inside the real attention-visible
        // KV range that are actually populated. dst->ne[0] is the padded graph row_idx width
        // (GGML_PAD(used_max_p1, n_pad)), so it over-counts padding/reserve cells; the true
        // active length is v_cells[0].used_max_p1() (paged requires n_stream==1). Empty cells
        // inside that range are skipped so unwritten/free blocks are not counted.
        std::set<uint32_t> trace_active_read_blocks;
        if (paged_block_size != 0 && !v_cells.empty()) {
            const auto & cells = v_cells[0];
            for (uint32_t r = 0; r < active_n_kv; ++r) {
                if (cells.is_empty(r)) {
                    continue;
                }
                uint32_t phys = paged_resolve(r);
                if (phys == PAGED_BLOCK_INVALID) {
                    phys = r;
                }
                trace_active_read_blocks.insert(phys / paged_block_size);
            }
        }
        paged_trace_emit_step(step, trace_read_blocks, trace_active_read_blocks,
                (uint32_t) dst->ne[0], active_n_kv);
    }

    if (timing_enabled) {
        const uint64_t set_input_us = llama_paged_timing_now_us() - set_input_start_us;
        if (paged_resume_timing_step_enabled) {
            uint64_t swapped_blocks = 0;
            for (uint32_t b = 0; b < paged_n_blocks && b < paged_block_states.size(); ++b) {
                if (paged_block_states[b] == paged_block_state::SWAPPED) {
                    swapped_blocks += 1;
                }
            }

            fprintf(stderr,
                    "KV_PAGED_STEP_TIMING step=%llu set_input_us=%llu "
                    "idle_maintenance_us=%llu swap_out_us=%llu swap_out_calls=%llu "
                    "check_read_us=%llu check_read_calls=%llu swapped_blocks=%llu "
                    "idle_owned_blocks=%llu prefetch_protected_blocks=%llu remaining_prefetch_blocks=%llu "
                    "defer_idle_swapout=%d swapout_deferred_blocks=%llu\n",
                    (unsigned long long) step,
                    (unsigned long long) set_input_us,
                    (unsigned long long) (paged_timing_idle_maintenance_us - idle_maintenance_us_before),
                    (unsigned long long) (paged_timing_swap_out_us - swap_out_us_before),
                    (unsigned long long) (paged_timing_swap_out_calls - swap_out_calls_before),
                    (unsigned long long) (paged_timing_check_read_us - check_read_us_before),
                    (unsigned long long) (paged_timing_check_read_calls - check_read_calls_before),
                    (unsigned long long) swapped_blocks,
                    (unsigned long long) paged_idle_cold_candidates,
                    (unsigned long long) prefetch_protected_blocks_this_call,
                    (unsigned long long) paged_prefetch_seq_last_swapped_blocks,
                    defer_idle_swapout_now ? 1 : 0,
                    (unsigned long long) swapout_deferred_blocks_this_call);
        }
        paged_timing_set_input_us += set_input_us;
        paged_timing_set_input_calls += 1;
    }
    if (defer_idle_swapout_now && paged_defer_idle_swapout_steps > 0) {
        paged_defer_idle_swapout_steps -= 1;
    }
    if (base_timing_enabled) {
        paged_base_timing_set_row_idx_total_us += llama_paged_timing_now_us() - base_set_input_start_us;
    }
    paged_trace_step += 1;
    return true;
}

void llama_kv_cache::paged_trace_note_write_block(uint32_t physical_block) const {
    if (!paged_trace_enabled || physical_block == PAGED_BLOCK_INVALID || paged_block_size == 0) {
        return;
    }
    paged_trace_write_blocks.insert(physical_block / paged_block_size);
}

void llama_kv_cache::paged_trace_emit_step(
        uint64_t step,
        const std::set<uint32_t> & read_blocks,
        const std::set<uint32_t> & active_read_blocks,
        uint32_t n_kv,
        uint32_t active_n_kv) const {
    // Telemetry only. Emit a single grep/Python-friendly line per decode step describing the
    // physical blocks touched this step and the current block-state population. The write-block
    // set is populated by set_input_k/v_idxs earlier in the same step (k_idxs -> v_idxs ->
    // paged_row_idx ordering in llm_graph_input_attn_kv::set_input); it is cleared here so the
    // next step starts fresh. Step id == count of paged_row_idx fills (one per decode step on
    // the paged in-graph gather path).
    //
    // read_blocks       = physical blocks covered by the padded row_idx tensor (n_kv wide).
    // active_read_blocks = physical blocks of the real, populated KV range (active_n_kv wide),
    //                      with padding/free/unwritten cells excluded.
    uint32_t resident = 0;
    uint32_t swapped  = 0;
    uint32_t released = 0;
    uint32_t free_b   = 0;
    for (uint32_t b = 0; b < paged_n_blocks && b < paged_block_states.size(); ++b) {
        switch (paged_block_states[b]) {
            case paged_block_state::RESIDENT: resident += 1; break;
            case paged_block_state::SWAPPED:  swapped  += 1; break;
            case paged_block_state::RELEASED: released += 1; break;
            case paged_block_state::UNUSED:   free_b   += 1; break;
            case paged_block_state::PENDING_WRITE: resident += 1; break;
            case paged_block_state::INVALID: break;
        }
    }

    auto to_csv = [](const std::set<uint32_t> & blocks) {
        std::string csv;
        for (uint32_t b : blocks) {
            if (!csv.empty()) {
                csv += ',';
            }
            csv += std::to_string(b);
        }
        return csv;
    };

    const std::string read_csv        = to_csv(read_blocks);
    const std::string active_read_csv = to_csv(active_read_blocks);
    const std::string write_csv       = to_csv(paged_trace_write_blocks);

    fprintf(stderr,
            "KV_PAGED_TRACE step=%llu n_kv=%u active_n_kv=%u "
            "read_block_count=%zu read_blocks=%s "
            "active_read_block_count=%zu active_read_blocks=%s "
            "write_block_count=%zu write_block=%s blocks_in_use=%llu "
            "resident_blocks=%u swapped_blocks=%u released_blocks=%u free_blocks=%u "
            "paged_swap_in_calls=%llu paged_swap_bytes_in=%llu paged_swap_in_last_block=%u paged_swapped_redirect_probe_rows=%llu paged_swapped_redirect_probe_swapped_rows=%llu paged_swapped_redirect_probe_resident_rows=%llu paged_swapped_redirect_probe_invalid_rows=%llu paged_swapped_redirect_probe_state_mismatch=%llu paged_swapped_redirect_probe_disabled=%llu paged_swapped_active_visible_violation_rows=%llu paged_swapped_active_visible_violation_blocks=%llu paged_swapped_active_visible_logical_seq_has=%llu paged_swapped_active_visible_phys_seq_has=%llu paged_swapped_active_visible_in_read_window=%llu paged_swapped_active_visible_not_in_read_window=%llu paged_swapped_active_visible_masked=%llu paged_swapped_active_visible_unmasked=%llu paged_swapped_active_violation_rows=%llu paged_swapped_active_violation_blocks=%llu paged_swapped_active_violation_block_had_active_owner_at_swapout=%llu paged_swapped_active_violation_after_swapout_write=%llu paged_swapped_active_violation_resolve_to_swapped=%llu paged_swapped_active_visible_restore_rows=%llu paged_swapped_active_visible_restore_blocks=%llu paged_swap_out_skip_active_visible_block=%llu paged_swap_out_skip_active_owned_block=%llu paged_swap_out_candidate_blocks=%llu paged_swap_out_allowed_blocks=%llu paged_swap_out_skip_fullprefix_read_window_only=%llu paged_swap_out_skip_true_active_owned=%llu paged_swap_out_skip_true_active_unmasked=%llu paged_active_restore_from_swapped_blocks=%llu paged_idle_only_swapped_blocks=%llu paged_write_to_swapped_block=%llu paged_write_to_swapped_block_seq=%llu\n",
            (unsigned long long) step, n_kv, active_n_kv,
            read_blocks.size(), read_csv.empty() ? "-" : read_csv.c_str(),
            active_read_blocks.size(), active_read_csv.empty() ? "-" : active_read_csv.c_str(),
            paged_trace_write_blocks.size(), write_csv.empty() ? "-" : write_csv.c_str(),
            (unsigned long long) paged_blocks_in_use,
            resident, swapped, released, free_b,
            (unsigned long long) paged_swap_in_calls,
            (unsigned long long) paged_swap_bytes_in,
            paged_swap_in_last_block,
            (unsigned long long) paged_swapped_redirect_probe_rows,
            (unsigned long long) paged_swapped_redirect_probe_swapped_rows,
            (unsigned long long) paged_swapped_redirect_probe_resident_rows,
            (unsigned long long) paged_swapped_redirect_probe_invalid_rows,
            (unsigned long long) paged_swapped_redirect_probe_state_mismatch,
            (unsigned long long) paged_swapped_redirect_probe_disabled,
            (unsigned long long) paged_swapped_active_visible_violation_rows,
            (unsigned long long) paged_swapped_active_visible_violation_blocks,
            (unsigned long long) paged_swapped_active_visible_logical_seq_has,
            (unsigned long long) paged_swapped_active_visible_phys_seq_has,
            (unsigned long long) paged_swapped_active_visible_in_read_window,
            (unsigned long long) paged_swapped_active_visible_not_in_read_window,
            (unsigned long long) paged_swapped_active_visible_masked,
            (unsigned long long) paged_swapped_active_visible_unmasked,
            (unsigned long long) paged_swapped_active_violation_rows,
            (unsigned long long) paged_swapped_active_violation_blocks,
            (unsigned long long) paged_swapped_active_violation_block_had_active_owner_at_swapout,
            (unsigned long long) paged_swapped_active_violation_after_swapout_write,
            (unsigned long long) paged_swapped_active_violation_resolve_to_swapped,
            (unsigned long long) paged_swapped_active_visible_restore_rows,
            (unsigned long long) paged_swapped_active_visible_restore_blocks,
            (unsigned long long) paged_swap_out_skip_active_visible_block,
            (unsigned long long) paged_swap_out_skip_active_owned_block,
            (unsigned long long) paged_swap_out_candidate_blocks,
            (unsigned long long) paged_swap_out_allowed_blocks,
            (unsigned long long) paged_swap_out_skip_fullprefix_read_window_only,
            (unsigned long long) paged_swap_out_skip_true_active_owned,
            (unsigned long long) paged_swap_out_skip_true_active_unmasked,
            (unsigned long long) paged_active_restore_from_swapped_blocks,
            (unsigned long long) paged_idle_only_swapped_blocks,
            (unsigned long long) paged_write_to_swapped_block,
            (unsigned long long) paged_write_to_swapped_block_seq);

    paged_trace_write_blocks.clear();
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;

    bool     approx_enabled;
    uint32_t visible_lo;
};

template<bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                const uint32_t cell_idx = args.approx_enabled ? args.visible_lo + j : j;

                if (cells.is_empty(cell_idx)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(cell_idx, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(cell_idx);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(cell_idx);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = -std::abs(p0 - p1);
                } else {
                    data[idst + j] = 0.0f;
                }

                continue;
skip:
                data[idst + j] = -INFINITY;
            }
        }
    }
}

template<bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, swa, is_2d, false>(args, data);
    }
}

template<bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, swa, false>(args, data);
    }
}

template<bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, float * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<causal, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(
        ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn, uint32_t visible_lo, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    float * data = (float *) dst->data;

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    const bool approx_enabled = uses_approx_dynamic_view() && causal_attn && visible_lo > 0;

    if (kv_swap_enabled && kv_swap_mode_ == kv_swap_mode::approx) {
        ++kv_approx_calls;
    }
    if (approx_enabled) {
        // Estimate cells hidden by the shifted physical read window; the mask tensor itself only spans [visible_lo, visible_hi).
        kv_approx_masked += (uint64_t) visible_lo*n_tokens;
    }

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
        /*.approx_enabled   =*/ approx_enabled,
        /*.visible_lo       =*/ visible_lo,
    };

    if (causal_attn) {
        set_input_kq_mask_impl<true> (args, data);
    } else {
        set_input_kq_mask_impl<false>(args, data);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id))) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);
        paged_note_cells(sinfo);
        paged_assert_identity(sinfo);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_reserve_n_kv();
    visible_lo = 0;

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() {
    if (kv) {
        kv->paged_restore_worker_stop_and_join();
    }
    if (paged_write_transaction_state_ == paged_write_transaction_state::APPLIED) {
        finish_paged_kv_write(llama_paged_kv_write_action::ROLLBACK_PRE_COMPUTE);
    } else if (paged_write_transaction_state_ == paged_write_transaction_state::COMPUTE_STARTED) {
        finish_paged_kv_write(llama_paged_kv_write_action::INVALIDATE_COMPUTE_STARTED);
    }
}

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (paged_shadow_pending) {
        if (kv->paged_shadow_validate_enabled) {
            kv->paged_shadow_validate(sinfos[i_cur], paged_shadow_n_kv);
        }
        paged_shadow_pending = false;
    }

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    if (kv->paged_write_context_invalid) {
        kv->paged_swap_error = {};
        kv->set_paged_swap_error(
                llama_paged_swap_error_reason::PAGED_WRITE_CONTEXT_INVALID,
                llama_kv_cache::PAGED_BLOCK_INVALID,
                llama_kv_cache::PAGED_BLOCK_INVALID,
                static_cast<int>(kv->paged_write_context_invalid_cause));
        return false;
    }

    const bool base_timing_enabled = kv->paged_base_timing_enabled;
    const uint64_t apply_paged_start_us = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    if (base_timing_enabled) {
        kv->paged_base_timing_apply_calls += 1;
    }

    if (kv->paged_swap_pending) {
        const uint32_t pending_n_kv = kv->paged_swap_pending_n_kv;
        kv->paged_swap_pending = false;
        kv->paged_swap_pending_n_kv = 0;
        const uint64_t t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
        kv->paged_swap_out_window(pending_n_kv);
        if (base_timing_enabled) {
            kv->paged_base_timing_swap_out_window_us += llama_paged_timing_now_us() - t0;
        }
    }

    uint64_t t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    paged_write_failure_handled = false;
    paged_write_metadata_deltas.clear();
    paged_write_block_deltas.clear();

    if (paged_write_transaction_state_ != paged_write_transaction_state::CLOSED ||
            !kv->paged_begin_write_transaction(this)) {
        return false;
    }

    std::vector<std::vector<uint32_t>> transaction_cells(kv->v_cells.size());
    llama_pos overwritten_max[LLAMA_MAX_SEQ];
    std::fill(overwritten_max, overwritten_max + LLAMA_MAX_SEQ, -1);
    std::vector<uint32_t> transaction_blocks;

    for (uint32_t stream = 0; stream < sinfos[i_cur].n_stream(); ++stream) {
        const uint32_t stream_id = sinfos[i_cur].strm[stream];
        if (stream_id >= kv->v_cells.size()) {
            continue;
        }

        const auto & selected = sinfos[i_cur].idxs[stream];
        transaction_cells[stream_id].insert(
                transaction_cells[stream_id].end(), selected.begin(), selected.end());

        for (uint32_t cell : selected) {
            if (cell < kv->v_cells[stream_id].size() && !kv->v_cells[stream_id].is_empty(cell)) {
                GGML_ASSERT(kv->v_cells[stream_id].seq_count(cell) == 1);
                const llama_seq_id old_seq = kv->v_cells[stream_id].seq_get(cell);
                if (old_seq >= 0 && old_seq < LLAMA_MAX_SEQ) {
                    overwritten_max[old_seq] = std::max(
                            overwritten_max[old_seq], kv->v_cells[stream_id].pos_get(cell));
                }
            }

            if (kv->kv_paged_enabled && kv->paged_block_size != 0) {
                const uint32_t logical_block = cell / kv->paged_block_size;
                if (logical_block < kv->paged_block_table.size()) {
                    const uint32_t block = kv->paged_block_table[logical_block];
                    if (block < kv->paged_block_states.size()) {
                        transaction_blocks.push_back(block);
                    }
                }
            }
        }
    }

    for (uint32_t seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
        if (overwritten_max[seq_id] < 0 || seq_id >= kv->seq_to_stream.size()) {
            continue;
        }
        const uint32_t stream_id = kv->seq_to_stream[seq_id];
        if (stream_id >= kv->v_cells.size()) {
            continue;
        }
        const auto & cells = kv->v_cells[stream_id];
        for (uint32_t cell = 0; cell < cells.size(); ++cell) {
            if (!cells.is_empty(cell) && cells.seq_has(cell, seq_id) &&
                    cells.pos_get(cell) <= overwritten_max[seq_id]) {
                transaction_cells[stream_id].push_back(cell);
            }
        }
    }

    for (uint32_t stream_id = 0; stream_id < transaction_cells.size(); ++stream_id) {
        auto & cells = transaction_cells[stream_id];
        if (cells.empty() || stream_id >= kv->v_heads.size()) {
            continue;
        }
        std::sort(cells.begin(), cells.end());
        cells.erase(std::unique(cells.begin(), cells.end()), cells.end());
        paged_write_metadata_deltas.push_back({
                stream_id, cells, kv->v_cells[stream_id].cp(cells), kv->v_heads[stream_id] });
    }

    std::sort(transaction_blocks.begin(), transaction_blocks.end());
    transaction_blocks.erase(
            std::unique(transaction_blocks.begin(), transaction_blocks.end()),
            transaction_blocks.end());
    for (uint32_t block : transaction_blocks) {
        paged_write_block_deltas.push_back({
                block,
                kv->paged_block_states[block],
                kv->paged_block_used[block] != 0,
                std::find(kv->paged_free_list.begin(), kv->paged_free_list.end(), block) !=
                    kv->paged_free_list.end() });
    }

    paged_write_transaction_state_ = paged_write_transaction_state::APPLIED;
    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    if (base_timing_enabled) {
        kv->paged_base_timing_apply_ubatch_us += llama_paged_timing_now_us() - t0;
    }
    t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    kv->paged_note_cells(sinfos[i_cur]);
    if (base_timing_enabled) {
        kv->paged_base_timing_note_cells_us += llama_paged_timing_now_us() - t0;
    }
    t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    kv->paged_assert_identity(sinfos[i_cur]);
    if (base_timing_enabled) {
        kv->paged_base_timing_assert_identity_us += llama_paged_timing_now_us() - t0;
    }

    n_kv = kv->get_n_kv(sinfos[i_cur]);
    visible_lo = kv->get_visible_lo(sinfos[i_cur]);
    paged_shadow_n_kv = n_kv;
    paged_shadow_pending = true;

    t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    kv->swap_out_window(n_kv);
    if (base_timing_enabled) {
        kv->paged_base_timing_swap_out_window_us += llama_paged_timing_now_us() - t0;
    }
    t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    kv->ensure_resident(n_kv);
    if (base_timing_enabled) {
        kv->paged_base_timing_ensure_resident_us += llama_paged_timing_now_us() - t0;
    }

    // stage P2: zero any rows that just entered the [0, n_kv) read window but were left
    // uncommitted at construction. Must run before madvise_tail so the cleared range and the
    // advised tail never overlap. No-op unless LLAMA_KV_LAZY_CLEAR=1.
    t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    kv->clear_frontier_advance(n_kv);
    if (base_timing_enabled) {
        kv->paged_base_timing_clear_frontier_us += llama_paged_timing_now_us() - t0;
    }

    // stage F1 / P1: advise the unused tail capacity [PAD(n_kv,256), kv_size) away to lower
    // current RSS. Runs after n_kv is known but does not change it; targets only capacity
    // outside the [0, n_kv) read window. No-op unless LLAMA_KV_LAZY_TAIL=1.
    t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    kv->madvise_tail(n_kv);
    if (base_timing_enabled) {
        kv->paged_base_timing_madvise_tail_us += llama_paged_timing_now_us() - t0;
    }

    // Stage 4A: block-aware madvise-only release. This does not assume a physical tail; it
    // releases only RESIDENT physical blocks that are absent from the current row_idx mapping.
    t0 = base_timing_enabled ? llama_paged_timing_now_us() : 0;
    kv->paged_release_blocks(n_kv);
    if (base_timing_enabled) {
        kv->paged_base_timing_paged_release_blocks_us += llama_paged_timing_now_us() - t0;
        kv->paged_base_timing_apply_paged_total_us += llama_paged_timing_now_us() - apply_paged_start_us;
    }
    kv->paged_swap_pending = kv->paged_swap_enabled &&
        !kv->paged_swap_explicit_only && !kv->paged_idle_swap_requested;
    kv->paged_swap_pending_n_kv = n_kv;
    kv->sample_swap_rss();

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

void llama_kv_cache_context::clear_paged_swap_error() {
    kv->clear_paged_swap_error();
}

bool llama_kv_cache_context::has_paged_swap_error() const {
    return kv->has_paged_swap_error();
}

llama_paged_swap_error llama_kv_cache_context::get_paged_swap_error() const {
    return kv->get_paged_swap_error();
}

void llama_kv_cache_context::mark_paged_kv_compute_started() {
    if (paged_write_transaction_state_ == paged_write_transaction_state::APPLIED) {
        paged_write_transaction_state_ = paged_write_transaction_state::COMPUTE_STARTED;
    }
}

bool llama_kv_cache_context::finish_paged_kv_write(llama_paged_kv_write_action action) {
    paged_write_failure_handled = false;

    if (!kv || i_cur >= sinfos.size()) {
        paged_write_transaction_state_ = paged_write_transaction_state::CLOSED;
        paged_write_metadata_deltas.clear();
        paged_write_block_deltas.clear();
        return true;
    }

    if (paged_write_transaction_state_ == paged_write_transaction_state::CLOSED) {
        paged_write_failure_handled = action != llama_paged_kv_write_action::COMMIT &&
            kv->kv_paged_enabled && kv->paged_write_context_invalid;
        return true;
    }

    if (paged_write_transaction_state_ == paged_write_transaction_state::COMPUTE_STARTED &&
            action == llama_paged_kv_write_action::ROLLBACK_PRE_COMPUTE) {
        action = llama_paged_kv_write_action::INVALIDATE_COMPUTE_STARTED;
    }

    const bool finished = kv->paged_finish_write_transaction(action, sinfos[i_cur], this);

    if (action == llama_paged_kv_write_action::ROLLBACK_PRE_COMPUTE) {
        for (const auto & delta : paged_write_metadata_deltas) {
            if (delta.stream < kv->v_cells.size()) {
                kv->v_cells[delta.stream].set(delta.cells, delta.cells_before);
            }
            if (delta.stream < kv->v_heads.size()) {
                kv->v_heads[delta.stream] = delta.head_before;
            }
        }

        for (const auto & delta : paged_write_block_deltas) {
            if (delta.block >= kv->paged_block_states.size() ||
                    kv->paged_block_states[delta.block] == llama_kv_cache::paged_block_state::INVALID) {
                continue;
            }

            const bool used_now = kv->paged_block_used[delta.block] != 0;
            if (used_now != delta.used_before) {
                if (delta.used_before) {
                    kv->paged_blocks_in_use += 1;
                } else if (kv->paged_blocks_in_use > 0) {
                    kv->paged_blocks_in_use -= 1;
                }
            }
            kv->paged_block_used[delta.block] = delta.used_before ? 1 : 0;
            kv->paged_block_states[delta.block] = delta.state_before;

            kv->paged_free_list.erase(
                    std::remove(
                        kv->paged_free_list.begin(),
                        kv->paged_free_list.end(),
                        delta.block),
                    kv->paged_free_list.end());
            if (delta.in_free_list_before) {
                kv->paged_free_list.push_back(delta.block);
            }
        }
    }

    paged_write_failure_handled = action != llama_paged_kv_write_action::COMMIT &&
        kv->kv_paged_enabled && (finished || kv->paged_write_context_invalid);
    paged_write_metadata_deltas.clear();
    paged_write_block_deltas.clear();
    paged_write_transaction_state_ = paged_write_transaction_state::CLOSED;
    return finished;
}

bool llama_kv_cache_context::needs_paged_kv_post_graph_sync() const {
    return kv && (paged_write_transaction_state_ != paged_write_transaction_state::CLOSED ||
            !kv->paged_release_post_ranges.empty());
}

bool llama_kv_cache_context::paged_kv_failure_handled() const {
    return paged_write_failure_handled;
}

bool llama_kv_cache_context::test_paged_kv_fail_graph_alloc() {
    if (!kv || !kv->paged_release_bounded_test_fail_graph_alloc) {
        return false;
    }
    kv->paged_release_bounded_test_fail_graph_alloc = false;
    kv->paged_release_bounded_test_fail_graph_alloc_triggers += 1;
    kv->set_paged_swap_error(
            llama_paged_swap_error_reason::PAGED_WRITE_GRAPH_ALLOC_FAILURE,
            llama_kv_cache::PAGED_BLOCK_INVALID,
            llama_kv_cache::PAGED_BLOCK_INVALID);
    return true;
}

bool llama_kv_cache_context::test_paged_kv_fail_after_compute() {
    if (!kv || !kv->paged_release_bounded_test_fail_after_compute) {
        return false;
    }
    kv->paged_release_bounded_test_fail_after_compute = false;
    kv->paged_release_bounded_test_fail_after_compute_triggers += 1;
    return true;
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

uint32_t llama_kv_cache_context::get_visible_lo() const {
    return visible_lo;
}

bool llama_kv_cache_context::uses_approx_dynamic_view() const {
    return kv->uses_approx_dynamic_view();
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il, bool causal_attn, ggml_tensor * row_idx) const {
    return kv->get_k(ctx, il, n_kv, visible_lo, sinfos[i_cur], causal_attn, row_idx);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il, bool causal_attn, ggml_tensor * row_idx) const {
    return kv->get_v(ctx, il, n_kv, visible_lo, sinfos[i_cur], causal_attn, row_idx);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_paged_row_idx(ggml_context * ctx) const {
    return kv->build_input_paged_row_idx(ctx, n_kv);
}

bool llama_kv_cache_context::uses_paged_row_idx() const {
    return kv->uses_paged_row_idx();
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

bool llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    return kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

bool llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    return kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

bool llama_kv_cache_context::set_input_paged_row_idx(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    return kv->set_input_paged_row_idx(dst, ubatch);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn, visible_lo, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}
