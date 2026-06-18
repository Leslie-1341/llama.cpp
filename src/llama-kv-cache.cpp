#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

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

llama_kv_backing_store_file::llama_kv_backing_store_file() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#if defined(O_TMPFILE)
    fd = open("/tmp", O_TMPFILE | O_RDWR | O_CLOEXEC, S_IRUSR | S_IWUSR);
    if (fd >= 0) {
        return;
    }
    stats.last_errno = errno;
#endif

    file = std::tmpfile();
    if (!file) {
        stats.last_errno = errno;
        return;
    }

    fd = fileno(file);
    if (fd < 0) {
        stats.last_errno = errno;
        std::fclose(file);
        file = nullptr;
    }
#else
    stats.last_errno = ENOSYS;
#endif
}

llama_kv_backing_store_file::~llama_kv_backing_store_file() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    if (file) {
        std::fclose(file);
        file = nullptr;
        fd = -1;
        return;
    }

    if (fd >= 0) {
        close(fd);
        fd = -1;
    }
#endif
}

llama_kv_backing_store_status llama_kv_backing_store_file::write_cell(
        uint32_t   strm,
        uint32_t   cell,
        const void * data,
        size_t     size,
        uint64_t & offset_out) {
    (void) strm;
    (void) cell;

    offset_out = 0;

    if (fd < 0) {
        return llama_kv_backing_store_status::disabled;
    }
    if (!data || size == 0) {
        return llama_kv_backing_store_status::bad_slot;
    }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    const uint64_t offset = file_len;
    const char * ptr = static_cast<const char *>(data);
    size_t written = 0;

    while (written < size) {
        const ssize_t ret = pwrite(fd, ptr + written, size - written, (off_t) (offset + written));
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            stats.last_errno = errno;
            return llama_kv_backing_store_status::io_error;
        }
        if (ret == 0) {
            stats.last_errno = EIO;
            return llama_kv_backing_store_status::io_error;
        }
        written += (size_t) ret;
    }

    offset_out = offset;
    file_len += size;
    stats.bytes_written += size;
    stats.write_calls += 1;
    stats.last_errno = 0;

    return llama_kv_backing_store_status::ok;
#else
    stats.last_errno = ENOSYS;
    return llama_kv_backing_store_status::disabled;
#endif
}

llama_kv_backing_store_status llama_kv_backing_store_file::read_cell(
        uint32_t strm,
        uint32_t cell,
        uint64_t offset,
        void *   data,
        size_t   size) {
    (void) strm;
    (void) cell;

    if (fd < 0) {
        return llama_kv_backing_store_status::disabled;
    }
    if (!data || size == 0 || offset > file_len || size > file_len - offset) {
        return llama_kv_backing_store_status::bad_slot;
    }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    char * ptr = static_cast<char *>(data);
    size_t read = 0;

    while (read < size) {
        const ssize_t ret = pread(fd, ptr + read, size - read, (off_t) (offset + read));
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            stats.last_errno = errno;
            return llama_kv_backing_store_status::io_error;
        }
        if (ret == 0) {
            stats.last_errno = EIO;
            return llama_kv_backing_store_status::io_error;
        }
        read += (size_t) ret;
    }

    stats.bytes_read += size;
    stats.read_calls += 1;
    stats.last_errno = 0;

    return llama_kv_backing_store_status::ok;
#else
    stats.last_errno = ENOSYS;
    return llama_kv_backing_store_status::disabled;
#endif
}

llama_kv_backing_store_status llama_kv_backing_store_file::release(uint64_t offset, size_t size) {
    if (fd < 0) {
        return llama_kv_backing_store_status::disabled;
    }
    if (size == 0 || offset > file_len || size > file_len - offset) {
        return llama_kv_backing_store_status::bad_slot;
    }

    stats.bytes_released += size;
    stats.release_calls += 1;
    stats.last_errno = 0;

    return llama_kv_backing_store_status::ok;
}

llama_kv_backing_store_status llama_kv_backing_store_file::reset() {
    stats = {};

    if (fd < 0) {
        return llama_kv_backing_store_status::disabled;
    }

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    if (ftruncate(fd, 0) != 0) {
        stats.last_errno = errno;
        return llama_kv_backing_store_status::io_error;
    }

    file_len = 0;
    return llama_kv_backing_store_status::ok;
#else
    stats.last_errno = ENOSYS;
    return llama_kv_backing_store_status::disabled;
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

    llama_kv_backing_store_file store;
    if (!store.is_enabled()) {
        const auto & stats = store.get_stats();
        LLAMA_LOG_ERROR("KV_SWAP_BACKEND_SELFTEST: backing store selftest fail status=disabled errno=%d\n",
                stats.last_errno);
        return;
    }

    const char payload[] = "hello-kv";
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
    model(model), hparams(model.hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type) {

    GGML_ASSERT(kv_size % n_pad == 0);

    llama_kv_backing_store_selftest_once();

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
            auto store = std::make_unique<llama_kv_backing_store_file>();
            if (!store->is_enabled()) {
                const auto & stats = store->get_stats();
                kv_swap_backend_failures += 1;
                LLAMA_LOG_WARN("%s: KV swap exact mode disabled: file backing store unavailable "
                        "(errno=%d)\n", __func__, stats.last_errno);
            } else {
                kv_swap_store   = std::move(store);
                kv_swap_enabled = true;
                kv_swap_mode_   = kv_swap_mode::exact;
                kv_swap_madvise = kv_swap_madvise_requested;
                LLAMA_LOG_INFO("%s: KV swap exact mode enabled (backend=file, window=%u, sink=%u)\n",
                        __func__, kv_swap_window, kv_swap_sink);
            }
        }
    }

    const char * LLAMA_KV_PAGED = std::getenv("LLAMA_KV_PAGED");
    const bool kv_paged_requested = LLAMA_KV_PAGED && std::strcmp(LLAMA_KV_PAGED, "1") == 0;
    if (kv_paged_requested) {
        const char * LLAMA_KV_PAGED_BLOCK_SIZE = std::getenv("LLAMA_KV_PAGED_BLOCK_SIZE");
        const char * LLAMA_KV_PAGED_SHIFT      = std::getenv("LLAMA_KV_PAGED_SHIFT");
        const char * LLAMA_KV_PAGED_RELEASE    = std::getenv("LLAMA_KV_PAGED_RELEASE");
        const char * LLAMA_KV_PAGED_SWAP       = std::getenv("LLAMA_KV_PAGED_SWAP");
        const char * LLAMA_KV_PAGED_TRACE      = std::getenv("LLAMA_KV_PAGED_TRACE");
        const char * LLAMA_KV_PAGED_IDLE_TRACE = std::getenv("LLAMA_KV_PAGED_IDLE_TRACE");
        const int block_size_env = LLAMA_KV_PAGED_BLOCK_SIZE ? std::atoi(LLAMA_KV_PAGED_BLOCK_SIZE) : 16;
        const int shift_env      = LLAMA_KV_PAGED_SHIFT      ? std::atoi(LLAMA_KV_PAGED_SHIFT)      : 0;
        const bool release_env   = LLAMA_KV_PAGED_RELEASE && std::strcmp(LLAMA_KV_PAGED_RELEASE, "1") == 0;
        const bool paged_swap_env = LLAMA_KV_PAGED_SWAP && std::strcmp(LLAMA_KV_PAGED_SWAP, "1") == 0;

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
            paged_trace_enabled = LLAMA_KV_PAGED_TRACE && std::strcmp(LLAMA_KV_PAGED_TRACE, "1") == 0;
            paged_idle_trace_enabled = LLAMA_KV_PAGED_IDLE_TRACE && std::strcmp(LLAMA_KV_PAGED_IDLE_TRACE, "1") == 0;
            paged_block_release_enabled = release_env && !paged_swap_enabled;
            if (paged_swap_enabled && !kv_swap_store) {
                auto store = std::make_unique<llama_kv_backing_store_file>();
                if (!store->is_enabled()) {
                    const auto & stats = store->get_stats();
                    paged_swap_backend_failures += 1;
                    paged_swap_enabled = false;
                    LLAMA_LOG_WARN("%s: KV paged block swap disabled: file backing store unavailable "
                            "(errno=%d)\n", __func__, stats.last_errno);
                } else {
                    kv_swap_store = std::move(store);
                }
            }
            LLAMA_LOG_INFO("%s: KV paged metadata enabled (block_size=%u, n_blocks=%u, shift=%u, "
                    "non_identity=%d, mapping_changed=%llu)\n",
                    __func__, paged_block_size, paged_n_blocks, paged_shift,
                    paged_non_identity_enabled ? 1 : 0,
                    (unsigned long long) paged_block_mapping_changed);
            if (paged_swap_enabled) {
                LLAMA_LOG_INFO("%s: KV paged block swap enabled (backend=file, release=disabled)\n", __func__);
            }
            if (paged_block_release_enabled) {
                LLAMA_LOG_INFO("%s: KV paged block release enabled (madvise-only)\n", __func__);
            }
            if (paged_trace_enabled) {
                LLAMA_LOG_INFO("%s: KV paged block access trace enabled (telemetry only, stderr)\n", __func__);
            }
            if (paged_idle_trace_enabled) {
                LLAMA_LOG_INFO("%s: KV paged idle trace enabled (telemetry only)\n", __func__);
            }
        }
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
}

llama_kv_cache::~llama_kv_cache() {
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
    paged_log_stats();
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }
    paged_reset();

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
    paged_free_list.resize(paged_n_blocks);

    paged_build_block_table();

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
    paged_free_list.resize(paged_n_blocks);
    paged_build_block_table();
    paged_blocks_in_use = 0;
    paged_swap_pending = false;
    paged_swap_pending_n_kv = 0;
}

void llama_kv_cache::paged_build_block_table() {
    if (!kv_paged_enabled) {
        return;
    }

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
                paged_block_used[physical_block] = 1;
                if (physical_block < paged_block_states.size()) {
                    paged_block_states[physical_block] = paged_block_state::RESIDENT;
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
        }
    }
}

uint32_t llama_kv_cache::paged_resolve(uint32_t cell) const {
    if (!kv_paged_enabled || paged_block_size == 0) {
        return cell;
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
        return cell;
    }
    if (phys != cell) {
        paged_write_resolve_changed += 1;
    }

    return phys;
}

void llama_kv_cache::paged_ensure_write_resident(uint32_t phys_cell) const {
    if (!kv_paged_enabled || (!paged_block_release_enabled && !paged_swap_enabled) ||
            phys_cell == PAGED_BLOCK_INVALID || paged_block_size == 0) {
        return;
    }

    paged_block_ensure_calls += 1;

    const uint32_t physical_block = phys_cell / paged_block_size;
    if (physical_block >= paged_block_states.size()) {
        return;
    }

    if (paged_block_states[physical_block] == paged_block_state::RELEASED) {
        paged_block_states[physical_block] = paged_block_state::RESIDENT;
        paged_block_ensure_released += 1;
    } else if (paged_block_states[physical_block] == paged_block_state::SWAPPED) {
        paged_swap_write_swapped_hits += 1;
        if (paged_swap_in_block(physical_block) &&
                paged_block_states[physical_block] == paged_block_state::RESIDENT) {
            paged_swap_write_swap_in_calls += 1;
        } else {
            paged_swap_write_swap_in_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap write swap-in failed: block=%u cell=%u state=%d\n",
                    __func__, physical_block, phys_cell, (int) paged_block_states[physical_block]);
        }
    }
}

void llama_kv_cache::paged_check_read_resident(uint32_t phys_cell, bool active) const {
    if (!kv_paged_enabled || (!paged_block_release_enabled && !paged_swap_enabled) ||
            phys_cell == PAGED_BLOCK_INVALID || paged_block_size == 0) {
        return;
    }

    const uint32_t physical_block = phys_cell / paged_block_size;
    if (physical_block >= paged_block_states.size()) {
        return;
    }

    if (paged_block_states[physical_block] == paged_block_state::RELEASED) {
        paged_release_violation += 1;
        if (active) {
            paged_active_release_violation += 1;
        } else {
            paged_padded_release_violation += 1;
        }
    } else if (paged_block_states[physical_block] == paged_block_state::SWAPPED) {
        paged_swap_read_swapped_hits += 1;
        if (paged_swap_in_block(physical_block) &&
                paged_block_states[physical_block] == paged_block_state::RESIDENT) {
            paged_swap_read_swap_in_calls += 1;
        } else {
            paged_swap_read_swap_in_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap read swap-in failed: block=%u cell=%u state=%d "
                    "fail_no_offset=%llu fail_bad_size=%llu fail_read_cell=%llu "
                    "fail_tensor_set=%llu backend_failures=%llu\n",
                    __func__, physical_block, phys_cell, (int) paged_block_states[physical_block],
                    (unsigned long long) paged_swap_in_fail_no_offset,
                    (unsigned long long) paged_swap_in_fail_bad_size,
                    (unsigned long long) paged_swap_in_fail_read_cell,
                    (unsigned long long) paged_swap_in_fail_tensor_set,
                    (unsigned long long) paged_swap_backend_failures);
        }
    }
}

void llama_kv_cache::paged_swap_out_block(uint32_t physical_block) {
    if (!kv_paged_enabled || !paged_swap_enabled || !kv_swap_store || v_trans || n_stream != 1 ||
            paged_block_size == 0 || v_cells.empty()) {
        return;
    }
    if (physical_block >= paged_block_states.size() ||
            paged_block_states[physical_block] != paged_block_state::RESIDENT) {
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

    auto & cells = v_cells[0];
    const uint32_t begin = physical_block * paged_block_size;
    const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
    if (end > paged_swap_offsets.size() || end > paged_swap_sizes.size()) {
        paged_swap_backend_failures += 1;
        return;
    }

    for (uint32_t cell = begin; cell < end; ++cell) {
        paged_swap_offsets[cell] = 0;
        paged_swap_sizes[cell] = 0;
    }

    auto clear_block_entries = [&]() {
        for (uint32_t cell = begin; cell < end; ++cell) {
            paged_swap_offsets[cell] = 0;
            paged_swap_sizes[cell] = 0;
        }
    };

    std::vector<uint8_t> staging(total_size);
    uint64_t block_bytes = 0;

    for (uint32_t cell = begin; cell < end; ++cell) {
        if (cell >= cells.size()) {
            paged_swap_backend_failures += 1;
            clear_block_entries();
            return;
        }

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
            paged_swap_backend_failures += 1;
            clear_block_entries();
            return;
        }

        paged_swap_offsets[cell] = offset;
        paged_swap_sizes[cell] = staging.size();
        block_bytes += staging.size();
    }

    for (uint32_t cell = begin; cell < end; ++cell) {
        if (paged_swap_sizes[cell] != total_size) {
            paged_swap_backend_failures += 1;
            clear_block_entries();
            return;
        }
    }

    paged_block_states[physical_block] = paged_block_state::SWAPPED;
    paged_swap_out_calls += 1;
    paged_blocks_swapped_out += 1;
    paged_swap_bytes_out += block_bytes;

    const uint64_t rss_before_kb = get_current_rss_kb();
    const uint64_t skip_no_full_before = paged_swap_madvise_skip_no_full_page;
    const uint64_t skip_neighbor_before = paged_swap_madvise_skip_neighbor;
    const uint64_t advised_bytes = paged_madvise_block(
            physical_block, nullptr,
            paged_swap_madvise_failures,
            paged_swap_madvise_skip_no_full_page,
            paged_swap_madvise_skip_neighbor);
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
    }
}

bool llama_kv_cache::paged_swap_in_block(uint32_t physical_block) const {
    if (!kv_paged_enabled || !paged_swap_enabled || !kv_swap_store || v_trans || n_stream != 1 ||
            paged_block_size == 0 || v_cells.empty()) {
        LLAMA_LOG_ERROR("%s: KV paged swap-in unavailable: block=%u enabled=%d store=%d v_trans=%d n_stream=%u block_size=%u v_cells=%zu\n",
                __func__, physical_block, paged_swap_enabled ? 1 : 0, kv_swap_store ? 1 : 0,
                (int) v_trans, n_stream, paged_block_size, v_cells.size());
        return false;
    }
    if (physical_block >= paged_block_states.size() ||
            paged_block_states[physical_block] != paged_block_state::SWAPPED) {
        LLAMA_LOG_ERROR("%s: KV paged swap-in bad state: block=%u n_states=%zu state=%d\n",
                __func__, physical_block, paged_block_states.size(),
                physical_block < paged_block_states.size() ? (int) paged_block_states[physical_block] : -1);
        return false;
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
        paged_swap_in_fail_bad_size += 1;
        paged_swap_backend_failures += 1;
        LLAMA_LOG_ERROR("%s: KV paged swap-in bad expected size: block=%u expected=0 state=%d\n",
                __func__, physical_block, (int) paged_block_states[physical_block]);
        return false;
    }

    const auto & cells = v_cells[0];
    const uint32_t begin = physical_block * paged_block_size;
    const uint32_t end = std::min<uint32_t>(begin + paged_block_size, paged_kv_size);
    std::vector<uint8_t> staging(total_size);
    uint64_t block_bytes = 0;

    for (uint32_t cell = begin; cell < end; ++cell) {
        const uint32_t cell_in_block = cell - begin;
        if (cell >= cells.size()) {
            paged_swap_in_fail_bad_size += 1;
            paged_swap_backend_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap-in cell OOB: block=%u cell=%u cell_in_block=%u cells=%zu expected=%zu state=%d\n",
                    __func__, physical_block, cell, cell_in_block, cells.size(), total_size,
                    (int) paged_block_states[physical_block]);
            return false;
        }

        if (cell >= paged_swap_offsets.size() || cell >= paged_swap_sizes.size()) {
            paged_swap_in_fail_no_offset += 1;
            paged_swap_backend_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap-in missing backing entry: block=%u cell=%u cell_in_block=%u metadata_cells=%zu/%zu expected=%zu state=%d\n",
                    __func__, physical_block, cell, cell_in_block,
                    paged_swap_offsets.size(), paged_swap_sizes.size(), total_size,
                    (int) paged_block_states[physical_block]);
            return false;
        }

        const uint64_t offset = paged_swap_offsets[cell];
        const size_t swap_size = paged_swap_sizes[cell];
        if (swap_size == 0) {
            paged_swap_in_fail_no_offset += 1;
            paged_swap_backend_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap-in missing backing entry: block=%u cell=%u cell_in_block=%u offset=%llu swap_size=%zu expected=%zu state=%d\n",
                    __func__, physical_block, cell, cell_in_block,
                    (unsigned long long) offset, swap_size, total_size,
                    (int) paged_block_states[physical_block]);
            return false;
        }
        if (swap_size != total_size) {
            paged_swap_in_fail_bad_size += 1;
            paged_swap_backend_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap-in bad backing size: block=%u cell=%u cell_in_block=%u offset=%llu swap_size=%zu expected=%zu state=%d\n",
                    __func__, physical_block, cell, cell_in_block,
                    (unsigned long long) offset, swap_size, total_size,
                    (int) paged_block_states[physical_block]);
            return false;
        }

        const auto status = kv_swap_store->read_cell(0, cell, offset, staging.data(), staging.size());
        if (status != llama_kv_backing_store_status::ok) {
            paged_swap_in_fail_read_cell += 1;
            paged_swap_backend_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap-in read_cell failed: block=%u cell=%u cell_in_block=%u offset=%llu swap_size=%zu expected=%zu status=%d state=%d\n",
                    __func__, physical_block, cell, cell_in_block,
                    (unsigned long long) offset, swap_size, total_size, (int) status,
                    (int) paged_block_states[physical_block]);
            return false;
        }

        size_t cursor = 0;
        for (const auto & layer : layers) {
            if (!layer.k_stream.empty() && layer.k_stream[0]) {
                auto * k = layer.k_stream[0];
                const size_t row_size = k->nb[1];
                if (!k->data || cursor + row_size > staging.size()) {
                    paged_swap_in_fail_tensor_set += 1;
                    paged_swap_backend_failures += 1;
                    LLAMA_LOG_ERROR("%s: KV paged swap-in tensor_set precheck failed: block=%u cell=%u cell_in_block=%u layer=%u tensor=K offset=%llu swap_size=%zu expected=%zu row_size=%zu cursor=%zu state=%d\n",
                            __func__, physical_block, cell, cell_in_block, layer.il,
                            (unsigned long long) offset, swap_size, total_size, row_size, cursor,
                            (int) paged_block_states[physical_block]);
                    return false;
                }
                ggml_backend_tensor_set(k, staging.data() + cursor, (size_t) cell * row_size, row_size);
                cursor += row_size;
            }
            if (layer.v && !layer.v_stream.empty() && layer.v_stream[0]) {
                auto * v = layer.v_stream[0];
                const size_t row_size = v->nb[1];
                if (!v->data || cursor + row_size > staging.size()) {
                    paged_swap_in_fail_tensor_set += 1;
                    paged_swap_backend_failures += 1;
                    LLAMA_LOG_ERROR("%s: KV paged swap-in tensor_set precheck failed: block=%u cell=%u cell_in_block=%u layer=%u tensor=V offset=%llu swap_size=%zu expected=%zu row_size=%zu cursor=%zu state=%d\n",
                            __func__, physical_block, cell, cell_in_block, layer.il,
                            (unsigned long long) offset, swap_size, total_size, row_size, cursor,
                            (int) paged_block_states[physical_block]);
                    return false;
                }
                ggml_backend_tensor_set(v, staging.data() + cursor, (size_t) cell * row_size, row_size);
                cursor += row_size;
            }
        }
        if (cursor != total_size) {
            paged_swap_in_fail_tensor_set += 1;
            paged_swap_backend_failures += 1;
            LLAMA_LOG_ERROR("%s: KV paged swap-in layout mismatch: block=%u cell=%u cell_in_block=%u offset=%llu swap_size=%zu expected=%zu cursor=%zu state=%d\n",
                    __func__, physical_block, cell, cell_in_block,
                    (unsigned long long) offset, swap_size, total_size, cursor,
                    (int) paged_block_states[physical_block]);
            return false;
        }
        block_bytes += staging.size();
    }

    paged_block_states[physical_block] = paged_block_state::RESIDENT;
    paged_swap_in_calls += 1;
    paged_blocks_swapped_in += 1;
    paged_swap_bytes_in += block_bytes;
    return true;
}

uint64_t llama_kv_cache::paged_madvise_block(
        uint32_t physical_block,
        const std::vector<uint8_t> * active,
        uint64_t & failures,
        uint64_t & skipped,
        uint64_t & skip_live) const {
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

            char * base = (char *) t->data;
            const uint64_t lo_cell = (uint64_t) physical_block * paged_block_size;
            const uint64_t hi_cell = std::min<uint64_t>(lo_cell + paged_block_size, paged_kv_size);
            const uintptr_t lo_a = (uintptr_t) base + (uintptr_t) lo_cell * row;
            const uintptr_t hi_a = (uintptr_t) base + (uintptr_t) hi_cell * row;
            if (hi_a <= lo_a) {
                skipped += 1;
                return;
            }

            const uintptr_t a_start = (lo_a + pg - 1) & ~(uintptr_t) (pg - 1);
            const uintptr_t a_end   = hi_a & ~(uintptr_t) (pg - 1);
            if (a_end <= a_start) {
                skipped += 1;
                return;
            }

            if ((lo_a & (uintptr_t) (pg - 1)) != 0 && physical_block > 0 &&
                    protected_neighbor(physical_block - 1)) {
                skip_live += 1;
            }
            if ((hi_a & (uintptr_t) (pg - 1)) != 0 && physical_block + 1 < paged_n_blocks &&
                    protected_neighbor(physical_block + 1)) {
                skip_live += 1;
            }

            const size_t len = (size_t) (a_end - a_start);
            const int rc = madvise((void *) a_start, len, MADV_DONTNEED);
            if (rc != 0) {
                failures += 1;
            } else {
                advised_bytes += len;
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
    return 0;
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

void llama_kv_cache::paged_log_stats() const {
    if (!kv_paged_enabled) {
        return;
    }

    LLAMA_LOG_INFO("%s: KV paged metadata stats: enabled=1 block_size=%u n_blocks=%u "
            "blocks_in_use=%llu free_blocks=%zu alloc_calls=%llu identity_checks=%llu identity_fail=%llu "
            "write_resolve_checks=%llu write_resolve_fail=%llu write_resolve_changed=%llu "
            "shadow_gather_calls=%llu shadow_gather_changed=%llu shadow_gather_mismatch=%llu shadow_gather_fail=%llu "
            "shadow_skipped_non_identity=%llu ingraph_gather_layers=%llu row_idx_changed=%llu row_idx_fail=%llu "
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
            "paged_nonidentity_enabled=%d paged_nonidentity_remap_rows=%llu "
            "paged_nonidentity_remap_blocks=%llu paged_nonidentity_skip_no_dummy=%llu "
            "paged_nonidentity_skip_not_masked=%llu paged_nonidentity_skip_not_resident=%llu "
            "paged_nonidentity_cold_in_read_window_before=%llu "
            "paged_nonidentity_cold_in_read_window_after=%llu "
            "paged_nonidentity_safe_candidates_after=%llu "
            "paged_swap_enabled=%d paged_swap_out_calls=%llu paged_swap_in_calls=%llu "
            "paged_blocks_swapped_out=%llu paged_blocks_swapped_in=%llu "
            "paged_swap_bytes_out=%llu paged_swap_bytes_in=%llu "
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
            "paged_swap_rss_drop_max_kb=%llu\n",
            __func__, paged_block_size, paged_n_blocks,
            (unsigned long long) paged_blocks_in_use,
            paged_free_list.size(),
            (unsigned long long) paged_alloc_calls,
            (unsigned long long) paged_identity_checks,
            (unsigned long long) paged_identity_fail,
            (unsigned long long) paged_write_resolve_checks,
            (unsigned long long) paged_write_resolve_fail,
            (unsigned long long) paged_write_resolve_changed,
            (unsigned long long) paged_shadow_gather_calls,
            (unsigned long long) paged_shadow_gather_changed,
            (unsigned long long) paged_shadow_gather_mismatch,
            (unsigned long long) paged_shadow_gather_fail,
            (unsigned long long) paged_shadow_skipped_non_identity,
            (unsigned long long) paged_ingraph_gather_layers,
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
            paged_nonidentity_probe_enabled ? 1 : 0,
            (unsigned long long) paged_nonidentity_remap_rows,
            (unsigned long long) paged_nonidentity_remap_blocks,
            (unsigned long long) paged_nonidentity_skip_no_dummy,
            (unsigned long long) paged_nonidentity_skip_not_masked,
            (unsigned long long) paged_nonidentity_skip_not_resident,
            (unsigned long long) paged_nonidentity_cold_in_read_window_before,
            (unsigned long long) paged_nonidentity_cold_in_read_window_after,
            (unsigned long long) paged_nonidentity_safe_candidates_after,
            paged_swap_enabled ? 1 : 0,
            (unsigned long long) paged_swap_out_calls,
            (unsigned long long) paged_swap_in_calls,
            (unsigned long long) paged_blocks_swapped_out,
            (unsigned long long) paged_blocks_swapped_in,
            (unsigned long long) paged_swap_bytes_out,
            (unsigned long long) paged_swap_bytes_in,
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
            (unsigned long long) paged_swap_rss_drop_max_kb);
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
    if (!kv_paged_enabled || !paged_block_release_enabled) {
        return;
    }
    if (v_trans || n_stream != 1 || paged_block_size == 0 || paged_n_blocks == 0 ||
            paged_block_states.size() != paged_n_blocks) {
        return;
    }

    paged_block_release_calls += 1;

    std::vector<uint8_t> active(paged_n_blocks, 0);
    const uint32_t n_active = std::min(n_kv, paged_kv_size);
    for (uint32_t r = 0; r < n_active; ++r) {
        const uint32_t phys = paged_resolve(r);
        if (phys == PAGED_BLOCK_INVALID) {
            continue;
        }
        const uint32_t physical_block = phys / paged_block_size;
        if (physical_block < paged_n_blocks) {
            active[physical_block] = 1;
        }
    }

    const uint64_t blocks_before = paged_blocks_released;
    const uint64_t bytes_before = paged_block_release_bytes;
    const uint64_t rss_before_kb = get_current_rss_kb();

    for (uint32_t physical_block = 0; physical_block < paged_n_blocks; ++physical_block) {
        if (active[physical_block]) {
            continue;
        }

        const paged_block_state state = paged_block_states[physical_block];
        if (state == paged_block_state::RELEASED || state == paged_block_state::SWAPPED) {
            continue;
        }

        const uint64_t advised_bytes = paged_madvise_block(
                physical_block, &active,
                paged_block_release_fail,
                paged_block_release_skip_unaligned,
                paged_block_release_skip_live);

        if (advised_bytes > 0) {
            paged_block_release_bytes += advised_bytes;
            paged_block_states[physical_block] = paged_block_state::RELEASED;
            paged_blocks_released += 1;
            if (state == paged_block_state::UNUSED) {
                paged_blocks_released_unused += 1;
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
    }
#else
    (void) n_kv;
#endif
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

    if (row_idx && paged_ingraph_gather_supported(il) && ns == 1 && !approx_dynamic) {
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

        if (row_idx && paged_ingraph_gather_supported(il) && ns == 1 && !approx_dynamic) {
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
    if (!kv_paged_enabled) {
        return nullptr;
    }

    const char * LLAMA_KV_PAGED_INGRAPH = std::getenv("LLAMA_KV_PAGED_INGRAPH");
    if (LLAMA_KV_PAGED_INGRAPH && std::strcmp(LLAMA_KV_PAGED_INGRAPH, "0") == 0) {
        return nullptr;
    }

    bool supported = n_stream == 1 && !v_trans;
    for (const auto & layer : layers) {
        supported = supported &&
            layer.k &&
            layer.v &&
            layer.k->type == GGML_TYPE_F32 &&
            layer.v->type == GGML_TYPE_F32;
    }

    if (!supported) {
        if (!paged_ingraph_warned) {
            LLAMA_LOG_WARN("%s: KV paged in-graph gather requires n_stream==1, !v_trans, and F32 K/V cache "
                    "(n_stream=%u, v_trans=%d) - falling back to continuous K/V views\n",
                    __func__, n_stream, (int) v_trans);
            paged_ingraph_warned = true;
        }
        return nullptr;
    }

    ggml_tensor * row_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
    ggml_set_input(row_idx);

    return row_idx;
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

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            const uint32_t cell = sinfo.idxs[s][i];
            const uint32_t phys = paged_write_resolve(cell);
            paged_ensure_write_resident(phys);
            paged_trace_note_write_block(phys);
            data[s*sinfo.size() + i] = offs + phys;
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                const uint32_t cell = sinfo.idxs[s][i];
                const uint32_t phys = paged_write_resolve(cell);
                paged_ensure_write_resident(phys);
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
                const uint32_t cell = sinfo.idxs[s][i];
                const uint32_t phys = paged_write_resolve(cell);
                paged_ensure_write_resident(phys);
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + phys;
                }
            }
        }
    }
}

void llama_kv_cache::set_input_paged_row_idx(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    if (!dst) {
        return;
    }

    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    int32_t * data = (int32_t *) dst->data;

    std::set<uint32_t> trace_read_blocks;
    std::set<uint32_t> trace_read_blocks_before_remap;

    uint32_t active_n_kv = 0;
    if (!v_cells.empty()) {
        active_n_kv = std::min<uint32_t>(v_cells[0].used_max_p1(), (uint32_t) dst->ne[0]);
    }

    std::bitset<LLAMA_MAX_SEQ> active_seq;
    const char * active_seq_source = "none";
    if (ubatch) {
        active_seq_source = "ubatch_seq_id";
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            for (int32_t s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch->seq_id[i][s];
                if (seq_id >= 0 && seq_id < LLAMA_MAX_SEQ) {
                    active_seq.set(seq_id);
                }
            }
        }
    }

    const char * LLAMA_KV_PAGED_GATHER_NONIDENTITY = std::getenv("LLAMA_KV_PAGED_GATHER_NONIDENTITY");
    const bool nonidentity_probe =
        LLAMA_KV_PAGED_GATHER_NONIDENTITY &&
        std::strcmp(LLAMA_KV_PAGED_GATHER_NONIDENTITY, "1") == 0;
    paged_nonidentity_probe_enabled = nonidentity_probe;

    uint32_t dummy_phys = PAGED_BLOCK_INVALID;
    std::vector<uint8_t> nonidentity_cold_blocks;
    if (nonidentity_probe &&
            paged_block_size != 0 &&
            paged_n_blocks != 0 &&
            !v_cells.empty() &&
            paged_block_states.size() == paged_n_blocks &&
            active_seq.any()) {
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
            if (phys == PAGED_BLOCK_INVALID) {
                continue;
            }

            const uint32_t block = phys / paged_block_size;
            if (block < paged_block_states.size() &&
                    paged_block_states[block] == paged_block_state::RESIDENT) {
                dummy_phys = phys;
                break;
            }
        }

        std::vector<std::bitset<LLAMA_MAX_SEQ>> block_seq((size_t) paged_n_blocks);
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

    std::set<uint32_t> nonidentity_remapped_blocks;

    for (int64_t r = 0; r < dst->ne[0]; ++r) {
        uint32_t phys = paged_resolve((uint32_t) r);
        if (phys == PAGED_BLOCK_INVALID || phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
            paged_row_idx_fail += 1;
            phys = (uint32_t) r;
        }
        const uint32_t phys_orig = phys;
        if ((paged_trace_enabled || paged_idle_trace_enabled) && paged_block_size != 0) {
            trace_read_blocks_before_remap.insert(phys_orig / paged_block_size);
        }

        if (nonidentity_probe && paged_block_size != 0 && !nonidentity_cold_blocks.empty()) {
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
                        paged_block_states[block] != paged_block_state::RESIDENT) {
                    paged_nonidentity_skip_not_resident += 1;
                } else if (dummy_phys == PAGED_BLOCK_INVALID ||
                        dummy_phys > (uint32_t) std::numeric_limits<int32_t>::max()) {
                    paged_nonidentity_skip_no_dummy += 1;
                } else {
                    phys = dummy_phys;
                    nonidentity_remapped_blocks.insert(block);
                    paged_nonidentity_remap_rows += 1;
                }
            }
        }

        const bool active = (uint32_t) r < active_n_kv && !v_cells[0].is_empty((uint32_t) r);
        paged_check_read_resident(phys, active);
        if (phys != (uint32_t) r) {
            paged_row_idx_changed += 1;
        }
        if ((paged_trace_enabled || paged_idle_trace_enabled) && paged_block_size != 0) {
            trace_read_blocks.insert(phys / paged_block_size);
        }
        data[r] = (int32_t) phys;
    }
    paged_nonidentity_remap_blocks += nonidentity_remapped_blocks.size();

    if (paged_idle_trace_enabled) {
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
        for (llama_seq_id seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
            if (!active_seq.test(seq_id)) {
                continue;
            }
            if (!active_seq_csv.empty()) {
                active_seq_csv += ',';
            }
            active_seq_csv += std::to_string(seq_id);
        }

        std::string last_active_csv;
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
        if (paged_block_size != 0 && paged_n_blocks != 0 && !v_cells.empty()) {
            const auto & cells = v_cells[0];
            std::vector<std::bitset<LLAMA_MAX_SEQ>> block_seq((size_t) paged_n_blocks);
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

            for (uint32_t block = 0; block < block_seq.size(); ++block) {
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

                const bool mixed = owner_count != 1;
                const bool only_seen_idle_seq = ((owner & paged_idle_seq_seen) == owner) && !has_active_seq;
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
                }
            }
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
        paged_nonidentity_cold_in_read_window_before = cold_in_read_window_before;
        paged_nonidentity_cold_in_read_window_after = cold_in_read_window;
        paged_nonidentity_safe_candidates_after = safe_swap_candidates;

        fprintf(stderr,
                "KV_PAGED_IDLE_TRACE step=%llu active_seq_source=%s active_seq_count=%llu "
                "idle_seq_count=%llu seen_seq_count=%llu active_seq=%s seq_last_active=%s "
                "non_empty_blocks=%llu single_seq_blocks=%llu multi_seq_blocks=%llu "
                "blocks_with_active_seq=%llu blocks_without_active_seq=%llu "
                "cold_candidates=%llu read_window_blocks=%llu cold_in_read_window=%llu "
                "cold_not_in_read_window=%llu skip_mixed_active=%llu safe_swap_candidates=%llu "
                "paged_nonidentity_enabled=%d paged_nonidentity_remap_rows=%llu "
                "paged_nonidentity_remap_blocks=%llu paged_nonidentity_skip_no_dummy=%llu "
                "paged_nonidentity_skip_not_masked=%llu paged_nonidentity_skip_not_resident=%llu "
                "paged_nonidentity_cold_in_read_window_before=%llu "
                "paged_nonidentity_cold_in_read_window_after=%llu "
                "paged_nonidentity_safe_candidates_after=%llu\n",
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
                paged_nonidentity_probe_enabled ? 1 : 0,
                (unsigned long long) paged_nonidentity_remap_rows,
                (unsigned long long) paged_nonidentity_remap_blocks,
                (unsigned long long) paged_nonidentity_skip_no_dummy,
                (unsigned long long) paged_nonidentity_skip_not_masked,
                (unsigned long long) paged_nonidentity_skip_not_resident,
                (unsigned long long) paged_nonidentity_cold_in_read_window_before,
                (unsigned long long) paged_nonidentity_cold_in_read_window_after,
                (unsigned long long) paged_nonidentity_safe_candidates_after);
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
        paged_trace_emit_step(trace_read_blocks, trace_active_read_blocks,
                (uint32_t) dst->ne[0], active_n_kv);
    }
}

void llama_kv_cache::paged_trace_note_write_block(uint32_t physical_block) const {
    if (!paged_trace_enabled || physical_block == PAGED_BLOCK_INVALID || paged_block_size == 0) {
        return;
    }
    paged_trace_write_blocks.insert(physical_block / paged_block_size);
}

void llama_kv_cache::paged_trace_emit_step(
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
    const uint64_t step = paged_trace_step++;

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
            "resident_blocks=%u swapped_blocks=%u released_blocks=%u free_blocks=%u\n",
            (unsigned long long) step, n_kv, active_n_kv,
            read_blocks.size(), read_csv.empty() ? "-" : read_csv.c_str(),
            active_read_blocks.size(), active_read_csv.empty() ? "-" : active_read_csv.c_str(),
            paged_trace_write_blocks.size(), write_csv.empty() ? "-" : write_csv.c_str(),
            (unsigned long long) paged_blocks_in_use,
            resident, swapped, released, free_b);

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

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (paged_shadow_pending) {
        kv->paged_shadow_validate(sinfos[i_cur], paged_shadow_n_kv);
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

    if (kv->paged_swap_pending) {
        const uint32_t pending_n_kv = kv->paged_swap_pending_n_kv;
        kv->paged_swap_pending = false;
        kv->paged_swap_pending_n_kv = 0;
        kv->paged_swap_out_window(pending_n_kv);
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    kv->paged_note_cells(sinfos[i_cur]);
    kv->paged_assert_identity(sinfos[i_cur]);

    n_kv = kv->get_n_kv(sinfos[i_cur]);
    visible_lo = kv->get_visible_lo(sinfos[i_cur]);
    paged_shadow_n_kv = n_kv;
    paged_shadow_pending = true;

    kv->swap_out_window(n_kv);
    kv->ensure_resident(n_kv);

    // stage P2: zero any rows that just entered the [0, n_kv) read window but were left
    // uncommitted at construction. Must run before madvise_tail so the cleared range and the
    // advised tail never overlap. No-op unless LLAMA_KV_LAZY_CLEAR=1.
    kv->clear_frontier_advance(n_kv);

    // stage F1 / P1: advise the unused tail capacity [PAD(n_kv,256), kv_size) away to lower
    // current RSS. Runs after n_kv is known but does not change it; targets only capacity
    // outside the [0, n_kv) read window. No-op unless LLAMA_KV_LAZY_TAIL=1.
    kv->madvise_tail(n_kv);

    // Stage 4A: block-aware madvise-only release. This does not assume a physical tail; it
    // releases only RESIDENT physical blocks that are absent from the current row_idx mapping.
    kv->paged_release_blocks(n_kv);
    kv->paged_swap_pending = kv->paged_swap_enabled;
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

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_paged_row_idx(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_paged_row_idx(dst, ubatch);
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
