// Stability / invariant test for llama-window (the mmap sliding-window prefetch
// controller). Builds without a real model: it backs the "layer" regions with a
// real read-only file mmap (mirroring production), drives the controller through
// many graph passes, and uses mincore(2) to observe page residency directly.
//
// Checks:
//   (1) Threading stability  : many create/advance/destroy cycles, workers join,
//       no crash/hang/leak (run under ASAN/TSAN for races).
//   (2) Prefetch + protection: after advancing to layer K with reclaim active,
//       the protected window [K-keep_behind, K+prefetch_ahead] is resident
//       (prefetched and never reclaimed).
//   (3) Edge cases           : disabled ctx, oversized params, garbage node names.
//
// Build:
//   g++ -O2 -std=c++17 -pthread -I ggml/include tests/test-window.cpp \
//       src/llama-window.cpp build/bin/libggml-base.so* -o /tmp/test-window

#include "../src/llama-window.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

static int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); g_fail = 1; } } while (0)

static size_t PAGE = 0;

// Is every page of [addr,addr+len) resident? (mincore: low bit set == resident)
static bool all_resident(void * addr, size_t len) {
    const size_t n = (len + PAGE - 1) / PAGE;
    std::vector<unsigned char> vec(n, 0);
    if (mincore(addr, len, vec.data()) != 0) {
        return false;
    }
    for (unsigned char c : vec) {
        if ((c & 1) == 0) return false;
    }
    return true;
}

int main() {
    PAGE = (size_t) sysconf(_SC_PAGESIZE);

    const int    n_layers     = 16;
    const size_t region_bytes = 4 * PAGE;            // each layer: 4 distinct pages
    const size_t total        = (size_t) n_layers * region_bytes;

    // Build a temp file and map it read-only/shared, like the real loader.
    char path[] = "/tmp/wintestXXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp");
    if (fd < 0) return 1;
    std::vector<uint8_t> buf(region_bytes);
    for (int l = 0; l < n_layers; ++l) {
        std::memset(buf.data(), (uint8_t) (l + 1), region_bytes);
        CHECK(write(fd, buf.data(), region_bytes) == (ssize_t) region_bytes, "write");
    }
    fsync(fd);
    uint8_t * base = (uint8_t *) mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
    CHECK(base != MAP_FAILED, "mmap");
    if (base == MAP_FAILED) return 1;

    std::vector<llama_window_region_input> inputs;
    for (int l = 0; l < n_layers; ++l) {
        llama_window_region_input in;
        in.name        = "blk." + std::to_string(l) + ".weight";
        in.addr        = base + (size_t) l * region_bytes;
        in.size        = region_bytes;
        in.file_idx    = 0;
        in.file_offset = (size_t) l * region_bytes;
        inputs.push_back(in);
    }

    // ggml context just to mint node tensors with names like "n-<layer>".
    struct ggml_init_params gp { 1024 * 1024, nullptr, true };
    ggml_context * gctx = ggml_init(gp);
    ggml_tensor * node = ggml_new_tensor_1d(gctx, GGML_TYPE_F32, 1);

    auto drive_pass = [&](llama_window_context & ctx, int up_to) {
        llama_window_graph_begin(ctx, false);
        for (int l = 0; l <= up_to; ++l) {
            char nm[32]; std::snprintf(nm, sizeof(nm), "n-%d", l);
            ggml_set_name(node, nm);
            llama_window_node_done(ctx, node);
        }
        llama_window_graph_end(ctx);
    };

    // ---- (1) Threading stability: repeated create/advance/destroy -------------
    for (int rep = 0; rep < 8; ++rep) {
        llama_window_params p;
        p.enabled = true; p.use_dontneed = true; p.memory_limit = 0; // aggressive reclaim
        p.window_size = 8; p.keep_behind = 2; p.prefetch_ahead = 2; p.worker_threads = 2;
        p.prefetch_budget = total; p.reclaim_budget = total;
        auto ctx = llama_window_create(inputs, n_layers, p);
        CHECK(llama_window_enabled(ctx.get()), "enabled after create");
        for (int pass = 0; pass < 4; ++pass) {
            drive_pass(*ctx, n_layers - 1);
        }
        const auto & st = llama_window_get_stats(*ctx);
        CHECK(st.prefetch_failures == 0, "no prefetch failures on valid regions");
        CHECK(st.reclaim_failures  == 0, "no reclaim failures on valid regions");
        CHECK(st.reclaim_calls <= st.reclaim_tasks, "reclaim_calls bounded by tasks");
        // ctx destroyed here -> workers must join cleanly (no hang)
    }
    std::printf("stability: 8x(create + 4 passes + destroy) completed\n");

    // ---- (2) Prefetch + protection invariant via mincore ----------------------
    {
        llama_window_params p;
        p.enabled = true; p.use_dontneed = true; p.memory_limit = 0;
        p.window_size = 8; p.keep_behind = 2; p.prefetch_ahead = 2; p.worker_threads = 2;
        p.prefetch_budget = total; p.reclaim_budget = total;
        auto ctx = llama_window_create(inputs, n_layers, p);

        const int K = 8;
        llama_window_graph_begin(*ctx, false);
        for (int l = 0; l <= K; ++l) {
            char nm[32]; std::snprintf(nm, sizeof(nm), "n-%d", l);
            ggml_set_name(node, nm);
            llama_window_node_done(*ctx, node);
        }
        // The protected window must end up resident (prefetched, never reclaimed).
        // Retry to tolerate async worker latency.
        bool protected_resident = false;
        for (int tries = 0; tries < 50 && !protected_resident; ++tries) {
            protected_resident = true;
            for (int d = -p.keep_behind; d <= p.prefetch_ahead; ++d) {
                int L = (K + d) % n_layers; if (L < 0) L += n_layers;
                if (!all_resident(base + (size_t) L * region_bytes, region_bytes)) {
                    protected_resident = false; break;
                }
            }
            if (!protected_resident) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        CHECK(protected_resident, "protected window [K-keep_behind, K+prefetch_ahead] resident");
        llama_window_graph_end(*ctx);
        std::printf("protection: window around layer %d resident after advance\n", K);
    }

    // ---- (3) Edge cases: must not crash --------------------------------------
    {
        llama_window_params p; p.enabled = true; p.use_dontneed = true;
        // window/ahead larger than n_layers, tiny memory_limit
        p.window_size = 999; p.prefetch_ahead = 999; p.keep_behind = 999;
        p.memory_limit = 1; p.worker_threads = 4;
        p.prefetch_budget = total; p.reclaim_budget = total;
        auto ctx = llama_window_create(inputs, n_layers, p);
        drive_pass(*ctx, n_layers - 1);
        // garbage / non-layer node names -> parse returns -1 -> no-op, no crash
        for (const char * nm : { "", "no-layer", "blk.x", "n-", "-", "n-99999999999999999999" }) {
            ggml_set_name(node, nm);
            llama_window_node_done(*ctx, node);
        }
        std::printf("edge cases: oversized params + garbage node names handled\n");
    }
    {
        // disabled context (n_layers = 0) must be inert
        llama_window_params p; p.enabled = true;
        auto ctx = llama_window_create({}, 0, p);
        CHECK(!llama_window_enabled(ctx.get()), "0-layer context disabled");
        llama_window_graph_begin(*ctx, false);  // must be safe no-ops
        llama_window_graph_end(*ctx);
    }

    ggml_free(gctx);
    munmap(base, total);
    close(fd);
    unlink(path);

    std::printf("\n%s\n", g_fail ? "FAILED" : "ALL PASS");
    return g_fail;
}
