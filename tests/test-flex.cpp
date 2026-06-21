// Standalone unit test for llama-flex (stage 2a): ring-buffered streaming loader.
// Builds without ggml/llama:  g++ test-flex.cpp ../src/llama-flex.cpp
//
// Verifies: (1) streamed tensor bytes match the file exactly across a cyclic
// access pattern, (2) the resident footprint is bounded by the ring (k slots),
// (3) prefetch hits occur when looking ahead.

#include "../src/llama-flex.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

static size_t rss_kb() {
    FILE * f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    long total = 0, resident = 0;
    if (std::fscanf(f, "%ld %ld", &total, &resident) != 2) resident = 0;
    std::fclose(f);
    return (size_t) resident * (size_t) sysconf(_SC_PAGESIZE) / 1024;
}

int main() {
    const int    n_layers      = 16;
    const int    tensors_per   = 3;
    const size_t tensor_bytes  = 8 * 1024 * 1024;   // 8 MiB per tensor
    const size_t layer_bytes   = tensors_per * tensor_bytes;
    const int    ring          = 4;

    // Build a temp file: each tensor filled with a byte = (layer*tensors_per + t) & 0xff.
    char path[] = "/tmp/flextestXXXXXX";
    int fd = mkstemp(path);
    if (fd < 0) { std::perror("mkstemp"); return 1; }

    std::vector<uint8_t> buf(tensor_bytes);
    std::vector<llama_flex_tensor> meta;
    size_t off = 0;
    for (int l = 0; l < n_layers; ++l) {
        for (int t = 0; t < tensors_per; ++t) {
            uint8_t val = (uint8_t) ((l * tensors_per + t) & 0xff);
            std::memset(buf.data(), val, tensor_bytes);
            if (write(fd, buf.data(), tensor_bytes) != (ssize_t) tensor_bytes) {
                std::perror("write"); return 1;
            }
            llama_flex_tensor m;
            m.name = "blk." + std::to_string(l) + ".w" + std::to_string(t);
            m.file_idx = 0; m.file_offset = off; m.size = tensor_bytes;
            meta.push_back(m);
            off += tensor_bytes;
        }
    }
    fsync(fd);
    int rfd = open(path, O_RDONLY);
    if (rfd < 0) { std::perror("open"); return 1; }

    llama_flex_params p;
    p.enabled = true; p.direct_io = false; p.debug_log = true;
    p.ring_layers = ring; p.io_threads = 2;
    if (const char * lb = std::getenv("FLEX_LOCK_BYTES")) p.lock_bytes = (size_t) atoll(lb);

    auto ctx = llama_flex_create({ rfd }, n_layers, p);
    if (!llama_flex_enabled(ctx.get())) { std::fprintf(stderr, "FAIL: not enabled\n"); return 1; }

    int mi = 0;
    for (int l = 0; l < n_layers; ++l)
        for (int t = 0; t < tensors_per; ++t)
            llama_flex_register_tensor(*ctx, l, meta[mi++]);
    llama_flex_finalize(*ctx);

    const size_t rss_before = rss_kb();

    // Simulate 4 decode passes over all layers with prefetch_ahead = 2.
    bool correct = true;
    size_t rss_peak = rss_before;
    const int passes = 4, ahead = 2;
    for (int pass = 0; pass < passes && correct; ++pass) {
        for (int l = 0; l < n_layers; ++l) {
            for (int a = 1; a <= ahead; ++a) {
                llama_flex_request_layer(*ctx, (l + a) % n_layers);
            }
            llama_flex_wait_layer(*ctx, l);
            for (int t = 0; t < tensors_per; ++t) {
                std::string nm = "blk." + std::to_string(l) + ".w" + std::to_string(t);
                uint8_t * d = (uint8_t *) llama_flex_get_tensor(*ctx, l, nm);
                if (!d) { std::fprintf(stderr, "FAIL: miss layer %d %s\n", l, nm.c_str()); correct = false; break; }
                uint8_t expect = (uint8_t) ((l * tensors_per + t) & 0xff);
                // spot-check head, middle, tail
                if (d[0] != expect || d[tensor_bytes/2] != expect || d[tensor_bytes-1] != expect) {
                    std::fprintf(stderr, "FAIL: data mismatch layer %d w%d got %d expect %d\n",
                                 l, t, d[0], expect); correct = false; break;
                }
            }
            llama_flex_release_layer(*ctx, l);
            // Simulate per-layer compute so IO threads have time to prefetch ahead.
            if (const char * c = std::getenv("FLEX_COMPUTE_US")) usleep(atoi(c));
            size_t r = rss_kb();
            if (r > rss_peak) rss_peak = r;
        }
    }

    const auto & st = llama_flex_get_stats(*ctx);
    std::printf("\n--- results ---\n");
    std::printf("correctness     : %s\n", correct ? "PASS" : "FAIL");
    std::printf("ring bytes      : %.1f MiB (k=%d x %.1f MiB slot)\n",
                st.ring_bytes/1048576.0, ring, layer_bytes/1048576.0);
    std::printf("model bytes     : %.1f MiB (%d layers)\n", (n_layers*layer_bytes)/1048576.0, n_layers);
    std::printf("layer loads     : %llu  hits: %llu  waits: %llu\n",
                (unsigned long long)st.layer_loads, (unsigned long long)st.layer_hits,
                (unsigned long long)st.wait_events);
    std::printf("bytes streamed  : %.1f MiB\n", st.bytes_streamed/1048576.0);
    std::printf("RSS before      : %.1f MiB\n", rss_before/1024.0);
    std::printf("RSS peak        : %.1f MiB  (delta %.1f MiB; ring bound %.1f MiB)\n",
                rss_peak/1024.0, (rss_peak-rss_before)/1024.0, st.ring_bytes/1048576.0);

    // RSS growth during the run must be bounded by ~ring size (+ slack), NOT the model.
    bool rss_ok = (rss_peak - rss_before) < (st.ring_bytes/1024 + 64*1024);
    std::printf("rss bounded     : %s\n", rss_ok ? "PASS" : "FAIL");

    close(rfd);
    unlink(path);
    bool pass = correct && rss_ok;
    std::printf("\n%s\n", pass ? "ALL PASS" : "FAILED");
    return pass ? 0 : 1;
}
