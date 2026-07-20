#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

static constexpr uint32_t GGUF_MAGIC = 0x46554747u;
static constexpr uint32_t GGUF_DEFAULT_ALIGNMENT = 32;
static constexpr uint16_t CODEC_RAW = 0;
static constexpr uint16_t CODEC_MWQ = 2;
static constexpr uint16_t CODEC_MWQ_HIER = 3;
static constexpr int MWQ_BLOCK_DEFAULT = 32;
static constexpr int MWQ_SCALE_GROUP_DEFAULT = 16;

enum gguf_value_type : uint32_t {
    VT_UINT8 = 0,
    VT_INT8 = 1,
    VT_UINT16 = 2,
    VT_INT16 = 3,
    VT_UINT32 = 4,
    VT_INT32 = 5,
    VT_FLOAT32 = 6,
    VT_BOOL = 7,
    VT_STRING = 8,
    VT_ARRAY = 9,
    VT_UINT64 = 10,
    VT_INT64 = 11,
    VT_FLOAT64 = 12,
};

struct source {
    int bits = 0;
    int block = MWQ_BLOCK_DEFAULT;
    bool v2 = false;
    int scale_group = MWQ_SCALE_GROUP_DEFAULT;
    int outlier_max = 0;
    float outlier_threshold = 6.0f;
    std::string path;
};

struct tensor_info {
    std::string name;
    std::vector<uint64_t> shape;
    ggml_type type = GGML_TYPE_COUNT;
    uint64_t offset = 0;
    uint64_t nbytes = 0;
};

struct sidecar_entry {
    std::string name;
    std::string src_path;
    ggml_type src_type = GGML_TYPE_COUNT;
    uint32_t expert = 0;
    uint16_t bits = 0;
    uint16_t block = MWQ_BLOCK_DEFAULT;
    uint16_t scale_group = MWQ_SCALE_GROUP_DEFAULT;
    uint16_t outlier_max = 0;
    float outlier_threshold = 6.0f;
    uint16_t codec = 0;
    uint64_t decoded_size = 0;
    uint64_t encoded_size = 0;
    uint64_t offset = 0;
    uint64_t src_offset = 0;
    uint64_t src_stride = 0;
    int64_t n_elem = 0;
};

static uint64_t align_to(uint64_t x, uint64_t a) {
    return ((x + a - 1) / a) * a;
}

static void read_exact(FILE * f, void * dst, size_t n) {
    if (std::fread(dst, 1, n, f) != n) {
        throw std::runtime_error("short read");
    }
}

template <typename T>
static T read_le(FILE * f) {
    T v;
    read_exact(f, &v, sizeof(v));
    return v;
}

static std::string read_string(FILE * f) {
    const uint64_t n = read_le<uint64_t>(f);
    std::string s(n, '\0');
    if (n > 0) {
        read_exact(f, s.data(), (size_t) n);
    }
    return s;
}

static void skip_value(FILE * f, uint32_t type) {
    switch (type) {
        case VT_UINT8:
        case VT_INT8:
        case VT_BOOL:
            std::fseek(f, 1, SEEK_CUR);
            return;
        case VT_UINT16:
        case VT_INT16:
            std::fseek(f, 2, SEEK_CUR);
            return;
        case VT_UINT32:
        case VT_INT32:
        case VT_FLOAT32:
            std::fseek(f, 4, SEEK_CUR);
            return;
        case VT_UINT64:
        case VT_INT64:
        case VT_FLOAT64:
            std::fseek(f, 8, SEEK_CUR);
            return;
        case VT_STRING: {
            const uint64_t n = read_le<uint64_t>(f);
            std::fseek(f, (long) n, SEEK_CUR);
            return;
        }
        case VT_ARRAY: {
            const uint32_t elem_type = read_le<uint32_t>(f);
            const uint64_t n = read_le<uint64_t>(f);
            for (uint64_t i = 0; i < n; ++i) {
                skip_value(f, elem_type);
            }
            return;
        }
        default:
            throw std::runtime_error("unknown GGUF metadata value type");
    }
}

static uint64_t tensor_nbytes(const std::vector<uint64_t> & shape, ggml_type type) {
    if (shape.empty()) {
        return 0;
    }
    uint64_t rest = 1;
    for (size_t i = 1; i < shape.size(); ++i) {
        rest *= shape[i];
    }
    return (uint64_t) ggml_row_size(type, (int64_t) shape[0]) * rest;
}

static std::vector<tensor_info> parse_gguf(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        throw std::runtime_error("failed to open " + path + ": " + std::strerror(errno));
    }

    try {
        if (read_le<uint32_t>(f) != GGUF_MAGIC) {
            throw std::runtime_error("not a GGUF file: " + path);
        }
        (void) read_le<uint32_t>(f); // version
        const uint64_t n_tensors = read_le<uint64_t>(f);
        const uint64_t n_kv = read_le<uint64_t>(f);

        uint32_t alignment = GGUF_DEFAULT_ALIGNMENT;
        for (uint64_t i = 0; i < n_kv; ++i) {
            const std::string key = read_string(f);
            const uint32_t type = read_le<uint32_t>(f);
            if (key == "general.alignment" && type == VT_UINT32) {
                alignment = read_le<uint32_t>(f);
            } else {
                skip_value(f, type);
            }
        }

        struct raw_tensor {
            std::string name;
            std::vector<uint64_t> shape;
            ggml_type type;
            uint64_t rel_offset;
        };
        std::vector<raw_tensor> raw;
        raw.reserve((size_t) n_tensors);
        for (uint64_t i = 0; i < n_tensors; ++i) {
            raw_tensor t;
            t.name = read_string(f);
            const uint32_t n_dims = read_le<uint32_t>(f);
            t.shape.resize(n_dims);
            for (uint32_t d = 0; d < n_dims; ++d) {
                t.shape[d] = read_le<uint64_t>(f);
            }
            t.type = (ggml_type) read_le<uint32_t>(f);
            t.rel_offset = read_le<uint64_t>(f);
            raw.push_back(std::move(t));
        }

        const uint64_t data_base = align_to((uint64_t) std::ftell(f), alignment);
        std::vector<tensor_info> infos;
        infos.reserve(raw.size());
        for (const auto & r : raw) {
            tensor_info t;
            t.name = r.name;
            t.shape = r.shape;
            t.type = r.type;
            t.offset = data_base + r.rel_offset;
            t.nbytes = tensor_nbytes(t.shape, t.type);
            infos.push_back(std::move(t));
        }
        std::fclose(f);
        return infos;
    } catch (...) {
        std::fclose(f);
        throw;
    }
}

static bool is_expert_tensor(const tensor_info & t) {
    return t.name.find("_exps") != std::string::npos && t.shape.size() == 3 && t.shape[2] > 1;
}

static std::vector<uint8_t> mwq_encode(const float * vals, int64_t n, int bits, int block_size) {
    if (bits < 1 || bits > 8 || block_size <= 0) {
        throw std::runtime_error("MWQ bits must be in the range 1..8 and block must be positive");
    }

    const int qmax = (1 << bits) - 1;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t block_bytes = 4 + qbytes;
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    std::vector<uint8_t> out(n_blocks * block_bytes);

    for (size_t ib = 0; ib < n_blocks; ++ib) {
        const int64_t off = (int64_t) ib * block_size;
        const int n_this = (int) std::min<int64_t>(block_size, n - off);
        float mn = vals[off];
        float mx = vals[off];
        for (int i = 1; i < n_this; ++i) {
            mn = std::min(mn, vals[off + i]);
            mx = std::max(mx, vals[off + i]);
        }
        const float scale = mx > mn ? (mx - mn) / (float) qmax : 0.0f;
        const float inv = scale != 0.0f ? 1.0f / scale : 0.0f;

        uint8_t * dst = out.data() + ib * block_bytes;
        const ggml_fp16_t mn_h = ggml_fp32_to_fp16(mn);
        const ggml_fp16_t sc_h = ggml_fp32_to_fp16(scale);
        std::memcpy(dst + 0, &mn_h, sizeof(mn_h));
        std::memcpy(dst + 2, &sc_h, sizeof(sc_h));
        std::memset(dst + 4, 0, qbytes);

        for (int i = 0; i < n_this; ++i) {
            int q = scale != 0.0f ? (int) std::lrintf((vals[off + i] - mn) * inv) : 0;
            q = std::max(0, std::min(qmax, q));
            const int bit = i * bits;
            uint8_t * p = dst + 4 + (bit >> 3);
            const int shift = bit & 7;
            p[0] |= (uint8_t) ((q << shift) & 0xff);
            if (shift + bits > 8) {
                p[1] |= (uint8_t) (q >> (8 - shift));
            }
        }
    }
    return out;
}

static uint16_t fp16_bits(float v) {
    const ggml_fp16_t h = ggml_fp32_to_fp16(v);
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

static void put_u16(uint8_t * p, uint16_t v) {
    p[0] = (uint8_t) (v & 0xffu);
    p[1] = (uint8_t) (v >> 8);
}

static size_t mwq_hier_encoded_size(int64_t n, int bits, int block_size, int scale_group, int outlier_max) {
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    const size_t n_groups = (n_blocks + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t outlier_stride = 1 + (size_t) std::max(0, outlier_max) * 3;
    return n_blocks * qbytes + n_blocks * 2 + n_groups * 2 + n_blocks + n_blocks * outlier_stride;
}

static std::vector<uint8_t> mwq_hier_encode(
        const float * vals,
        int64_t n,
        int bits,
        int block_size,
        int scale_group,
        int outlier_max,
        float outlier_threshold) {
    if (bits < 1 || bits > 8 || block_size <= 0 || scale_group <= 0 || outlier_max < 0) {
        throw std::runtime_error("invalid MWQ-HIER parameters");
    }
    const int qmax = (1 << bits) - 1;
    const size_t n_blocks = (size_t) ((n + block_size - 1) / block_size);
    const size_t n_groups = (n_blocks + (size_t) scale_group - 1) / (size_t) scale_group;
    const size_t qbytes = (size_t) ((block_size * bits + 7) / 8);
    const size_t weights_bytes = n_blocks * qbytes;
    const size_t mins_off = weights_bytes;
    const size_t gscale_off = mins_off + n_blocks * 2;
    const size_t scode_off = gscale_off + n_groups * 2;
    const size_t outlier_off = scode_off + n_blocks;
    const size_t outlier_stride = 1 + (size_t) outlier_max * 3;
    std::vector<uint8_t> out(outlier_off + n_blocks * outlier_stride);
    std::vector<float> scales(n_blocks, 0.0f);

    for (size_t ib = 0; ib < n_blocks; ++ib) {
        const int64_t off = (int64_t) ib * block_size;
        const int n_this = (int) std::min<int64_t>(block_size, n - off);
        float mean = 0.0f;
        for (int i = 0; i < n_this; ++i) {
            mean += vals[off + i];
        }
        mean /= (float) std::max(1, n_this);
        float var = 0.0f;
        for (int i = 0; i < n_this; ++i) {
            const float d = vals[off + i] - mean;
            var += d * d;
        }
        const float sigma = std::sqrt(var / (float) std::max(1, n_this));

        std::vector<int> outlier_idx;
        if (outlier_max > 0 && sigma > 0.0f) {
            std::vector<std::pair<float, int>> cand;
            for (int i = 0; i < n_this; ++i) {
                const float z = std::fabs(vals[off + i] - mean) / sigma;
                if (z >= outlier_threshold) {
                    cand.push_back({z, i});
                }
            }
            std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int i = 0; i < std::min<int>(outlier_max, (int) cand.size()); ++i) {
                outlier_idx.push_back(cand[i].second);
            }
            std::sort(outlier_idx.begin(), outlier_idx.end());
        }

        float mn = vals[off];
        float mx = vals[off];
        for (int i = 0; i < n_this; ++i) {
            if (std::binary_search(outlier_idx.begin(), outlier_idx.end(), i)) {
                continue;
            }
            mn = std::min(mn, vals[off + i]);
            mx = std::max(mx, vals[off + i]);
        }
        const float scale = mx > mn ? (mx - mn) / (float) qmax : 0.0f;
        const float inv = scale != 0.0f ? 1.0f / scale : 0.0f;
        scales[ib] = scale;
        put_u16(out.data() + mins_off + ib * 2, fp16_bits(mn));

        uint8_t * qdst = out.data() + ib * qbytes;
        for (int i = 0; i < n_this; ++i) {
            int q = scale != 0.0f ? (int) std::lrintf((vals[off + i] - mn) * inv) : 0;
            q = std::max(0, std::min(qmax, q));
            const int bit = i * bits;
            uint8_t * p = qdst + (bit >> 3);
            const int shift = bit & 7;
            p[0] |= (uint8_t) ((q << shift) & 0xff);
            if (shift + bits > 8) {
                p[1] |= (uint8_t) (q >> (8 - shift));
            }
        }

        uint8_t * ob = out.data() + outlier_off + ib * outlier_stride;
        ob[0] = (uint8_t) outlier_idx.size();
        for (int j = 0; j < (int) outlier_idx.size(); ++j) {
            const int idx = outlier_idx[j];
            const int bit = idx * bits;
            uint32_t packed = qdst[bit >> 3];
            if ((bit & 7) + bits > 8) {
                packed |= (uint32_t) qdst[(bit >> 3) + 1] << 8;
            }
            const int q = (packed >> (bit & 7)) & qmax;
            const float approx = mn + scale * (float) q;
            const float residual = vals[off + idx] - approx;
            ob[1 + j * 3] = (uint8_t) idx;
            put_u16(ob + 1 + j * 3 + 1, fp16_bits(residual));
        }
    }

    for (size_t g = 0; g < n_groups; ++g) {
        const size_t b0 = g * (size_t) scale_group;
        const size_t b1 = std::min(n_blocks, b0 + (size_t) scale_group);
        float gs = 0.0f;
        for (size_t b = b0; b < b1; ++b) {
            gs = std::max(gs, scales[b]);
        }
        put_u16(out.data() + gscale_off + g * 2, fp16_bits(gs));
        for (size_t b = b0; b < b1; ++b) {
            int code = gs > 0.0f ? (int) std::lrintf(255.0f * scales[b] / gs) : 0;
            out[scode_off + b] = (uint8_t) std::max(0, std::min(255, code));
        }
    }
    return out;
}

static uint64_t mwq_encoded_size(int64_t n, int bits, int block_size) {
    const uint64_t qbytes = (uint64_t) ((block_size * bits + 7) / 8);
    const uint64_t block_bytes = 4 + qbytes;
    const uint64_t n_blocks = (uint64_t) ((n + block_size - 1) / block_size);
    return n_blocks * block_bytes;
}

static std::vector<sidecar_entry> collect_entries_for_source(const source & src) {
    std::vector<tensor_info> tensors = parse_gguf(src.path);

    std::vector<sidecar_entry> entries;
    int tensor_idx = 0;
    int tensor_total = 0;
    for (const auto & t : tensors) {
        if (is_expert_tensor(t)) {
            ++tensor_total;
        }
    }

    for (const auto & t : tensors) {
        if (!is_expert_tensor(t)) {
            continue;
        }
        ++tensor_idx;
        const int n_expert = (int) t.shape[2];
        if (t.nbytes % (uint64_t) n_expert != 0) {
            throw std::runtime_error("non-uniform expert slices: " + t.name);
        }
        const uint64_t stride = t.nbytes / (uint64_t) n_expert;
        const int64_t n_elem = (int64_t) t.shape[0] * (int64_t) t.shape[1];
        const ggml_type_traits * traits = ggml_get_type_traits(t.type);
        if (traits == nullptr || traits->to_float == nullptr) {
            throw std::runtime_error("unsupported tensor type for MWQ: " + t.name);
        }

        std::fprintf(stderr, "moe-sidecar: index [%d/%d] %s bits=%d block=%d experts=%d\n",
                tensor_idx, tensor_total, t.name.c_str(), src.bits, src.block, n_expert);

        for (int expert = 0; expert < n_expert; ++expert) {
            sidecar_entry ent;
            ent.name = t.name;
            ent.src_path = src.path;
            ent.src_type = t.type;
            ent.expert = (uint32_t) expert;
            ent.bits = (uint16_t) src.bits;
            ent.block = (uint16_t) src.block;
            ent.scale_group = (uint16_t) src.scale_group;
            ent.outlier_max = (uint16_t) src.outlier_max;
            ent.outlier_threshold = src.outlier_threshold;
            ent.codec = src.v2 ? CODEC_MWQ_HIER : CODEC_MWQ;
            ent.decoded_size = stride;
            ent.encoded_size = src.v2 ?
                mwq_hier_encoded_size(n_elem, src.bits, src.block, src.scale_group, src.outlier_max) :
                mwq_encoded_size(n_elem, src.bits, src.block);
            ent.src_offset = t.offset + (uint64_t) expert * stride;
            ent.src_stride = stride;
            ent.n_elem = n_elem;
            entries.push_back(std::move(ent));
        }
    }
    return entries;
}

static void write_exact(FILE * f, const void * data, size_t size, const std::string & what) {
    if (size == 0) {
        return;
    }
    if (std::fwrite(data, 1, size, f) != size) {
        throw std::runtime_error("write failed while writing " + what + ": " + std::strerror(errno));
    }
}

static void close_checked(FILE * f, const std::string & what) {
    if (f != nullptr && std::fclose(f) != 0) {
        throw std::runtime_error("close failed for " + what + ": " + std::strerror(errno));
    }
}

static void write_u16(FILE * f, uint16_t v) {
    write_exact(f, &v, sizeof(v), "u16");
}

static void write_u8(FILE * f, uint8_t v) {
    write_exact(f, &v, sizeof(v), "u8");
}

static void write_u32(FILE * f, uint32_t v) {
    write_exact(f, &v, sizeof(v), "u32");
}

static void write_u64(FILE * f, uint64_t v) {
    write_exact(f, &v, sizeof(v), "u64");
}

static void write_sidecar(const std::string & path, std::vector<sidecar_entry> & entries) {
    bool v2 = false;
    for (const auto & ent : entries) {
        v2 = v2 || ent.codec == CODEC_MWQ_HIER;
    }
    uint64_t off = v2 ? 32 : 16;
    for (const auto & ent : entries) {
        off += (v2 ? 58 : 34) + ent.name.size();
    }
    for (auto & ent : entries) {
        ent.offset = off;
        off += ent.encoded_size;
    }

    FILE * out = std::fopen(path.c_str(), "wb");
    if (out == nullptr) {
        throw std::runtime_error("failed to open output " + path + ": " + std::strerror(errno));
    }
    write_exact(out, v2 ? "LMOEBV2\0" : "LMOEBC1\0", 8, "sidecar magic");
    write_u32(out, (uint32_t) entries.size());
    write_u32(out, 0);
    if (v2) {
        write_u64(out, 32);
        write_u64(out, 0);
    }
    for (const auto & ent : entries) {
        write_u16(out, (uint16_t) ent.name.size());
        write_u32(out, ent.expert);
        uint16_t block_log2 = 0;
        for (uint16_t b = ent.block; b > 1; b >>= 1) {
            ++block_log2;
        }
        if (v2) {
            write_u8(out, (uint8_t) ent.bits);
            write_u8(out, (uint8_t) block_log2);
            write_u16(out, ent.codec);
            write_u16(out, ent.scale_group);
            write_u16(out, ent.outlier_max);
            write_u32(out, 0);
            write_u64(out, ent.decoded_size);
            write_u64(out, ent.encoded_size);
            write_u64(out, ent.offset);
            write_u64(out, 0);
            write_u64(out, 0);
        } else {
            write_u16(out, (uint16_t) (ent.bits | (block_log2 << 8)));
            write_u16(out, ent.codec);
            write_u64(out, ent.decoded_size);
            write_u64(out, ent.encoded_size);
            write_u64(out, ent.offset);
        }
        write_exact(out, ent.name.data(), ent.name.size(), "entry name " + ent.name);
    }
    FILE * src_file = nullptr;
    std::string src_path;
    std::string last_label;
    std::vector<uint8_t> encoded_slice;
    std::vector<float> f32;
    for (const auto & ent : entries) {
        if (src_path != ent.src_path) {
            if (src_file != nullptr) {
                std::fclose(src_file);
            }
            src_path = ent.src_path;
            src_file = std::fopen(src_path.c_str(), "rb");
            if (src_file == nullptr) {
                std::fclose(out);
                throw std::runtime_error("failed to open " + src_path + ": " + std::strerror(errno));
            }
        }
        const std::string label = ent.name + "#" + std::to_string(ent.bits) + "b" + std::to_string(ent.block);
        if (label != last_label) {
            last_label = label;
            std::fprintf(stderr, "moe-sidecar: write %s bits=%u block=%u\n", ent.name.c_str(), ent.bits, ent.block);
        }

        const ggml_type_traits * traits = ggml_get_type_traits(ent.src_type);
        if (traits == nullptr || traits->to_float == nullptr) {
            std::fclose(out);
            if (src_file != nullptr) {
                std::fclose(src_file);
            }
            throw std::runtime_error("unsupported tensor type while writing " + ent.name);
        }
        encoded_slice.resize((size_t) ent.src_stride);
        f32.resize((size_t) ent.n_elem);
        if (std::fseek(src_file, (long) ent.src_offset, SEEK_SET) != 0) {
            std::fclose(out);
            if (src_file != nullptr) {
                std::fclose(src_file);
            }
            throw std::runtime_error("seek failed");
        }
        read_exact(src_file, encoded_slice.data(), encoded_slice.size());
        traits->to_float(encoded_slice.data(), f32.data(), ent.n_elem);
        std::vector<uint8_t> blob = ent.codec == CODEC_MWQ_HIER ?
            mwq_hier_encode(f32.data(), ent.n_elem, ent.bits, ent.block, ent.scale_group, ent.outlier_max, ent.outlier_threshold) :
            mwq_encode(f32.data(), ent.n_elem, ent.bits, ent.block);
        if (blob.size() != ent.encoded_size) {
            std::fclose(out);
            if (src_file != nullptr) {
                std::fclose(src_file);
            }
            throw std::runtime_error("internal encoded size mismatch");
        }
        write_exact(out, blob.data(), blob.size(), "blob " + label);
    }
    if (src_file != nullptr) {
        close_checked(src_file, src_path);
    }
    close_checked(out, path);
}

static void usage(const char * argv0) {
    std::fprintf(stderr,
            "usage: %s --bits N [--block B] [--v2] [--scale-group N] [--outliers N] [--outlier-threshold Z] [--add N[:B]:MODEL ...] MODEL OUTPUT\n"
            "\n"
            "Build an MWQ MoE expert sidecar directly from GGUF expert tensors.\n"
            "B is the MWQ block size, normally 32, 64, or 128. --v2 emits hierarchical-scale MWQ with sparse residual outliers.\n",
            argv0);
}

static bool is_power_of_two(int v) {
    return v > 0 && (v & (v - 1)) == 0;
}

}

int main(int argc, char ** argv) {
    try {
        int bits = 0;
        int block = MWQ_BLOCK_DEFAULT;
        bool v2 = false;
        int scale_group = MWQ_SCALE_GROUP_DEFAULT;
        int outlier_max = 0;
        float outlier_threshold = 6.0f;
        std::vector<source> sources;
        std::vector<std::string> positional;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                usage(argv[0]);
                return 0;
            } else if (arg == "--bits" && i + 1 < argc) {
                bits = std::atoi(argv[++i]);
            } else if (arg == "--block" && i + 1 < argc) {
                block = std::atoi(argv[++i]);
            } else if (arg == "--v2") {
                v2 = true;
            } else if (arg == "--scale-group" && i + 1 < argc) {
                scale_group = std::atoi(argv[++i]);
            } else if (arg == "--outliers" && i + 1 < argc) {
                outlier_max = std::atoi(argv[++i]);
            } else if (arg == "--outlier-threshold" && i + 1 < argc) {
                outlier_threshold = std::atof(argv[++i]);
            } else if (arg == "--add" && i + 1 < argc) {
                std::string spec = argv[++i];
                const size_t colon = spec.find(':');
                if (colon == std::string::npos) {
                    throw std::runtime_error("--add expects N[:B]:MODEL");
                }
                const std::string first = spec.substr(0, colon);
                const std::string rest = spec.substr(colon + 1);
                const size_t colon2 = rest.find(':');
                if (colon2 == std::string::npos) {
                    sources.push_back({std::atoi(first.c_str()), block, v2, scale_group, outlier_max, outlier_threshold, rest});
                } else {
                    sources.push_back({std::atoi(first.c_str()), std::atoi(rest.substr(0, colon2).c_str()),
                            v2, scale_group, outlier_max, outlier_threshold, rest.substr(colon2 + 1)});
                }
            } else if (arg == "--codec" && i + 1 < argc) {
                std::string codec = argv[++i];
                if (codec != "mwq") {
                    throw std::runtime_error("llama-moe-sidecar currently emits mwq sidecars only");
                }
            } else if (arg == "--target-model" && i + 1 < argc) {
                ++i; // MWQ decodes back into the source tensor layout.
            } else {
                positional.push_back(arg);
            }
        }
        if (bits <= 0 || positional.size() != 2 || !is_power_of_two(block)) {
            usage(argv[0]);
            return 1;
        }
        sources.insert(sources.begin(), {bits, block, v2, scale_group, outlier_max, outlier_threshold, positional[0]});

        std::vector<sidecar_entry> entries;
        for (const auto & src : sources) {
            if (src.bits < 1 || src.bits > 8) {
                throw std::runtime_error("bits must be in the range 1..8");
            }
            if (!is_power_of_two(src.block) || src.block < 16 || src.block > 256) {
                throw std::runtime_error("block must be a power of two in [16, 256]");
            }
            if (!is_power_of_two(src.scale_group) || src.scale_group < 1 || src.scale_group > 256) {
                throw std::runtime_error("scale-group must be a power of two in [1, 256]");
            }
            if (src.outlier_max < 0 || src.outlier_max > 8) {
                throw std::runtime_error("outliers must be in [0, 8]");
            }
            auto more = collect_entries_for_source(src);
            entries.insert(entries.end(),
                    std::make_move_iterator(more.begin()),
                    std::make_move_iterator(more.end()));
        }

        write_sidecar(positional[1], entries);
        uint64_t total = 0;
        for (const auto & ent : entries) {
            total += ent.encoded_size;
        }
        std::fprintf(stderr, "moe-sidecar: wrote %zu slices, %.1f MiB: %s\n",
                entries.size(), total / 1048576.0, positional[1].c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "moe-sidecar: error: %s\n", e.what());
        return 1;
    }
}
