#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static size_t parse_alignment(const char * value) {
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 || parsed > std::numeric_limits<uint32_t>::max() ||
            (parsed & (parsed - 1)) != 0) {
        throw std::invalid_argument("alignment must be a power of two that fits in uint32_t");
    }
    return (size_t) parsed;
}

static void copy_bytes(std::ifstream & input, std::ofstream & output, size_t input_offset, size_t size) {
    constexpr size_t chunk_size = 8ull * 1024ull * 1024ull;
    std::vector<char> buffer(std::min(chunk_size, size));

    input.seekg((std::streamoff) input_offset);
    size_t remaining = size;
    while (remaining > 0) {
        const size_t chunk = std::min(remaining, buffer.size());
        input.read(buffer.data(), (std::streamsize) chunk);
        output.write(buffer.data(), (std::streamsize) chunk);
        remaining -= chunk;
    }
}

static void write_zeros(std::ofstream & output, size_t size) {
    static const std::vector<char> zeros(64 * 1024, 0);
    while (size > 0) {
        const size_t chunk = std::min(size, zeros.size());
        output.write(zeros.data(), (std::streamsize) chunk);
        size -= chunk;
    }
}

static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

int main(int argc, char ** argv) {
    try {
        size_t alignment = 4096;
        int arg = 1;
        if (argc > 2 && std::string(argv[arg]) == "--alignment") {
            alignment = parse_alignment(argv[arg + 1]);
            arg += 2;
        }
        if (argc - arg != 2) {
            std::fprintf(stderr, "usage: %s [--alignment N] INPUT.gguf OUTPUT.gguf\n", argv[0]);
            return 1;
        }

        const std::string input_path = argv[arg];
        const std::string output_path = argv[arg + 1];
        if (input_path == output_path) {
            throw std::invalid_argument("input and output paths must differ");
        }

        ggml_context * ctx_meta = nullptr;
        gguf_init_params params = {
            /*.no_alloc =*/ true,
            /*.ctx      =*/ &ctx_meta,
        };
        gguf_context * ctx_in = gguf_init_from_file(input_path.c_str(), params);
        if (ctx_in == nullptr || ctx_meta == nullptr) {
            throw std::runtime_error("failed to open input GGUF");
        }

        gguf_context * ctx_out = gguf_init_empty();
        gguf_set_kv(ctx_out, ctx_in);
        gguf_set_val_u32(ctx_out, GGUF_KEY_GENERAL_ALIGNMENT, (uint32_t) alignment);

        const int64_t n_tensors = gguf_get_n_tensors(ctx_in);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(ctx_in, i);
            ggml_tensor * tensor = ggml_get_tensor(ctx_meta, name);
            if (tensor == nullptr) {
                throw std::runtime_error(std::string("missing tensor metadata: ") + name);
            }
            gguf_add_tensor(ctx_out, tensor);
        }

        std::vector<uint8_t> metadata(gguf_get_meta_size(ctx_out));
        gguf_get_meta_data(ctx_out, metadata.data());

        std::ifstream input(input_path, std::ios::binary);
        std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
        input.exceptions(std::ios::badbit | std::ios::failbit);
        output.exceptions(std::ios::badbit | std::ios::failbit);
        output.write((const char *) metadata.data(), (std::streamsize) metadata.size());

        size_t total_payload = 0;
        size_t total_padding = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            const size_t input_offset = gguf_get_data_offset(ctx_in) + gguf_get_tensor_offset(ctx_in, i);
            const size_t output_offset = metadata.size() + gguf_get_tensor_offset(ctx_out, i);
            const size_t tensor_size = gguf_get_tensor_size(ctx_in, i);
            const size_t current = (size_t) output.tellp();
            if (current > output_offset) {
                throw std::runtime_error("output metadata offsets are inconsistent");
            }
            write_zeros(output, output_offset - current);
            total_padding += output_offset - current;
            copy_bytes(input, output, input_offset, tensor_size);
            total_payload += tensor_size;

            const size_t padded_end = align_up(output_offset + tensor_size, alignment);
            const size_t after_tensor = (size_t) output.tellp();
            write_zeros(output, padded_end - after_tensor);
            total_padding += padded_end - after_tensor;
        }

        output.close();
        input.close();

        gguf_free(ctx_out);
        gguf_free(ctx_in);
        ggml_free(ctx_meta);

        std::printf("aligned %" PRId64 " tensors to %zu bytes\n", n_tensors, alignment);
        std::printf("payload: %.2f MiB, padding: %.2f MiB\n",
                total_payload / 1024.0 / 1024.0,
                total_padding / 1024.0 / 1024.0);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
