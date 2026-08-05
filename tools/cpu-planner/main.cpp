#include "llama.h"
#include "../../src/llama-ext.h"

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr size_t MiB = 1024ull * 1024ull;
constexpr size_t GiB = 1024ull * MiB;

struct planner_params {
    std::string model;
    size_t ram_budget = 0;
    size_t margin = 512ull * MiB;
    uint32_t n_ctx = 4096;
    std::vector<int> parallels = {1, 2, 4};
    std::vector<int> batches = {1024, 2048};
    std::vector<int> ubatches = {256, 512};
    std::vector<ggml_type> kv_types = {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0};
};

struct memory_estimate {
    size_t model = 0;
    size_t context = 0;
    size_t compute = 0;

    size_t total() const {
        return model + context + compute;
    }
};

struct candidate {
    int parallel = 1;
    int batch = 512;
    int ubatch = 128;
    ggml_type kv_type = GGML_TYPE_F16;
    memory_estimate no_repack;
    memory_estimate native;
    bool native_fits = false;
    bool no_repack_fits = false;
    double score = -1.0;
    std::string mode;
    size_t weight_budget = 0;
    size_t flex_ring_layers = 0;
    double flex_lock_gb = 0.0;
    size_t moe_budget_mb = 0;
};

static void usage() {
    printf(
        "usage: llama-cpu-planner -m MODEL [options]\n"
        "\n"
        "CPU-only advisor for choosing KV, batch/ubatch/parallel and existing\n"
        "dense/MoE weight scheduling modes.\n"
        "\n"
        "options:\n"
        "  -m, --model FILE          GGUF model path (required)\n"
        "  --ram-budget SIZE        RAM budget, e.g. 12G, 8192M (default: cgroup/MemAvailable)\n"
        "  --margin SIZE            safety margin (default: 512M)\n"
        "  -c, --ctx-size N         context size used for estimates (default: 4096)\n"
        "  --parallel LIST          comma-separated n_seq/parallel candidates (default: 1,2,4)\n"
        "  -b, --batch-size LIST    comma-separated n_batch candidates (default: 1024,2048)\n"
        "  -ub, --ubatch-size LIST  comma-separated n_ubatch candidates (default: 256,512)\n"
        "  --kv-types LIST          comma-separated KV types (default: f16,q8_0,q4_0)\n"
        "  -h, --help               show this help\n");
}

static void planner_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static bool parse_size(const std::string & s, size_t & out) {
    if (s.empty()) {
        return false;
    }
    char * end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || v < 0.0) {
        return false;
    }
    size_t mul = 1;
    if (*end != '\0') {
        if ((end[0] == 'g' || end[0] == 'G') && end[1] == '\0') {
            mul = GiB;
        } else if ((end[0] == 'm' || end[0] == 'M') && end[1] == '\0') {
            mul = MiB;
        } else if ((end[0] == 'k' || end[0] == 'K') && end[1] == '\0') {
            mul = 1024;
        } else {
            return false;
        }
    }
    out = (size_t) (v * (double) mul);
    return true;
}

static std::vector<std::string> split_csv(const std::string & s) {
    std::vector<std::string> ret;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            ret.push_back(item);
        }
    }
    return ret;
}

static bool parse_int_list(const std::string & s, std::vector<int> & out) {
    std::vector<int> tmp;
    for (const auto & item : split_csv(s)) {
        char * end = nullptr;
        const long v = std::strtol(item.c_str(), &end, 10);
        if (end == item.c_str() || *end != '\0' || v <= 0 || v > std::numeric_limits<int>::max()) {
            return false;
        }
        tmp.push_back((int) v);
    }
    if (tmp.empty()) {
        return false;
    }
    out = std::move(tmp);
    return true;
}

static bool parse_kv_type(const std::string & s, ggml_type & out) {
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const ggml_type t = (ggml_type) i;
        const char * name = ggml_type_name(t);
        if (name != nullptr && s == name) {
            out = t;
            return true;
        }
    }
    return false;
}

static bool parse_kv_types(const std::string & s, std::vector<ggml_type> & out) {
    std::vector<ggml_type> tmp;
    for (const auto & item : split_csv(s)) {
        ggml_type t = GGML_TYPE_COUNT;
        if (!parse_kv_type(item, t)) {
            return false;
        }
        tmp.push_back(t);
    }
    if (tmp.empty()) {
        return false;
    }
    out = std::move(tmp);
    return true;
}

static bool parse_args(int argc, char ** argv, planner_params & params) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * opt) -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", opt);
                std::exit(1);
            }
            return argv[++i];
        };

        if (arg == "-h" || arg == "--help") {
            usage();
            std::exit(0);
        } else if (arg == "-m" || arg == "--model") {
            params.model = need_value(arg.c_str());
        } else if (arg == "--ram-budget") {
            if (!parse_size(need_value(arg.c_str()), params.ram_budget)) {
                fprintf(stderr, "invalid --ram-budget\n");
                return false;
            }
        } else if (arg == "--margin") {
            if (!parse_size(need_value(arg.c_str()), params.margin)) {
                fprintf(stderr, "invalid --margin\n");
                return false;
            }
        } else if (arg == "-c" || arg == "--ctx-size") {
            params.n_ctx = (uint32_t) std::max(1, std::atoi(need_value(arg.c_str())));
        } else if (arg == "--parallel") {
            if (!parse_int_list(need_value(arg.c_str()), params.parallels)) {
                fprintf(stderr, "invalid --parallel list\n");
                return false;
            }
        } else if (arg == "-b" || arg == "--batch-size") {
            if (!parse_int_list(need_value(arg.c_str()), params.batches)) {
                fprintf(stderr, "invalid --batch-size list\n");
                return false;
            }
        } else if (arg == "-ub" || arg == "--ubatch-size") {
            if (!parse_int_list(need_value(arg.c_str()), params.ubatches)) {
                fprintf(stderr, "invalid --ubatch-size list\n");
                return false;
            }
        } else if (arg == "--kv-types") {
            if (!parse_kv_types(need_value(arg.c_str()), params.kv_types)) {
                fprintf(stderr, "invalid --kv-types list\n");
                return false;
            }
        } else {
            fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return false;
        }
    }

    if (params.model.empty()) {
        fprintf(stderr, "missing required -m/--model\n");
        return false;
    }
    return true;
}

static size_t read_mem_available() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    size_t value_kb = 0;
    std::string unit;
    while (f >> key >> value_kb >> unit) {
        if (key == "MemAvailable:") {
            return value_kb * 1024ull;
        }
    }
    return 0;
}

static std::string current_cgroup_path() {
    std::ifstream f("/proc/self/cgroup");
    std::string line;
    while (std::getline(f, line)) {
        const size_t pos = line.find("::");
        if (pos != std::string::npos) {
            return line.substr(pos + 2);
        }
    }
    return "/";
}

static size_t read_size_file(const std::string & path) {
    std::ifstream f(path);
    std::string s;
    if (!(f >> s) || s == "max") {
        return 0;
    }
    size_t out = 0;
    parse_size(s, out);
    return out;
}

static size_t detect_ram_budget() {
    const size_t mem_avail = read_mem_available();
    const std::string cg = current_cgroup_path();
    const std::string base = "/sys/fs/cgroup" + (cg == "/" ? std::string() : cg);
    const size_t cg_max = read_size_file(base + "/memory.max");
    const size_t cg_cur = read_size_file(base + "/memory.current");

    size_t cg_avail = 0;
    if (cg_max > 0 && cg_max > cg_cur) {
        cg_avail = cg_max - cg_cur;
    }
    if (mem_avail > 0 && cg_avail > 0) {
        return std::min(mem_avail, cg_avail);
    }
    return mem_avail > 0 ? mem_avail : cg_avail;
}

static llama_model * load_model_for_estimate(
        const planner_params & params,
        bool use_extra_bufts) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = false;
    mparams.use_mlock = false;
    mparams.use_extra_bufts = use_extra_bufts;
    mparams.no_alloc = true;

    return llama_model_load_from_file(params.model.c_str(), mparams);
}

static memory_estimate estimate_memory(
        llama_model * model,
        const planner_params & params,
        const candidate & cand) {
    if (model == nullptr) {
        return {};
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = params.n_ctx;
    cparams.n_seq_max = (uint32_t) cand.parallel;
    cparams.n_batch = (uint32_t) cand.batch;
    cparams.n_ubatch = (uint32_t) cand.ubatch;
    cparams.type_k = cand.kv_type;
    cparams.type_v = cand.kv_type;
    cparams.offload_kqv = false;
    cparams.op_offload = false;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        return {};
    }

    memory_estimate ret;
    const llama_memory_breakdown breakdown = llama_get_memory_breakdown(ctx);
    for (const auto & it : breakdown) {
        ret.model += it.second.model;
        ret.context += it.second.context;
        ret.compute += it.second.compute;
    }

    llama_free(ctx);
    return ret;
}

static void score_candidate(candidate & cand, const planner_params & params, int n_expert) {
    const size_t budget = params.ram_budget > params.margin ? params.ram_budget - params.margin : 0;
    cand.native_fits = cand.native.total() > 0 && cand.native.total() <= budget;
    cand.no_repack_fits = cand.no_repack.total() > 0 && cand.no_repack.total() <= budget;

    const size_t non_weight = cand.no_repack.context + cand.no_repack.compute;
    const size_t weight = cand.no_repack.model;
    const size_t remaining = budget > non_weight ? budget - non_weight : 0;

    if (cand.native_fits) {
        cand.mode = "native";
        cand.score = 1000000.0 + cand.parallel * 1000.0 - cand.no_repack.total() / (double) GiB;
        return;
    }
    if (cand.no_repack_fits) {
        cand.mode = "mmap-no-repack";
        cand.score = 900000.0 + cand.parallel * 1000.0 - cand.no_repack.total() / (double) GiB;
        return;
    }
    if (remaining <= 0 || weight == 0) {
        cand.mode = "unfit";
        cand.score = -1.0;
        return;
    }

    cand.weight_budget = remaining;
    if (n_expert > 0) {
        cand.mode = "moe-buffer";
        cand.moe_budget_mb = remaining / MiB;
        const double kv_pressure = non_weight / (double) std::max<size_t>(1, params.ram_budget);
        cand.score = (double) cand.parallel * 1000.0
                + (double) cand.moe_budget_mb
                - kv_pressure * 250.0
                - (cand.kv_type == GGML_TYPE_F16 ? 50.0 : 0.0);
        return;
    }

    cand.mode = "flex";
    const size_t approx_layers = 32; // advisor-only fallback; runtime FLEX_AUTO computes exact ring.
    const size_t approx_layer = std::max<size_t>(1, weight / approx_layers);
    cand.flex_ring_layers = std::max<size_t>(2, std::min<size_t>(approx_layers, remaining / approx_layer));
    const size_t ring_bytes = cand.flex_ring_layers * approx_layer;
    const size_t lock_bytes = remaining > ring_bytes ? remaining - ring_bytes : 0;
    cand.flex_lock_gb = lock_bytes / (double) GiB;

    const double stream_bytes = weight > lock_bytes ? (double) (weight - lock_bytes) : 0.0;
    const double io_per_token = stream_bytes / (double) std::max(1, cand.parallel);
    cand.score = (double) cand.parallel * 1000.0
            - io_per_token / (double) MiB
            + cand.flex_lock_gb * 100.0
            - (cand.kv_type == GGML_TYPE_F16 ? 50.0 : 0.0);
}

static std::string shell_quote(const std::string & s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static void print_candidate(const candidate & c, const planner_params & params, int n_expert) {
    printf("mode: %s (%s)\n", c.mode.c_str(), n_expert > 0 ? "MoE" : "dense");
    printf("memory: native %.1f MiB, no-repack %.1f MiB, budget %.1f MiB, margin %.1f MiB\n",
            c.native.total() / (double) MiB,
            c.no_repack.total() / (double) MiB,
            params.ram_budget / (double) MiB,
            params.margin / (double) MiB);
    printf("breakdown(no-repack): model %.1f MiB, context/KV %.1f MiB, compute %.1f MiB\n",
            c.no_repack.model / (double) MiB,
            c.no_repack.context / (double) MiB,
            c.no_repack.compute / (double) MiB);
    printf("params: -c %" PRIu32 " -b %d -ub %d -ctk %s -ctv %s",
            params.n_ctx, c.batch, c.ubatch, ggml_type_name(c.kv_type), ggml_type_name(c.kv_type));
    if (c.parallel > 1) {
        printf(" --parallel %d", c.parallel);
    }
    printf("\n\n");

    printf("recommended environment:\n");
    if (c.mode == "native") {
        printf("  # no streaming env needed\n");
    } else if (c.mode == "mmap-no-repack") {
        printf("  # no streaming env needed; run with --no-repack\n");
    } else if (c.mode == "flex") {
        printf("  export LLAMA_FLEX=1\n");
        printf("  export LLAMA_FLEX_AUTO=1\n");
        printf("  export LLAMA_FLEX_THREADS=4\n");
        printf("  export LLAMA_FLEX_RING=%zu\n", c.flex_ring_layers);
        printf("  export LLAMA_FLEX_LOCK_GB=%.2f\n", std::max(0.0, c.flex_lock_gb));
        printf("  export LLAMA_FLEX_PIN_POLICY=cost-aware\n");
    } else if (c.mode == "moe-buffer") {
        printf("  export LLAMA_LAZY_V2=1\n");
        printf("  export LLAMA_LAZY_MOE_BUFFER=1\n");
        printf("  export LLAMA_LAZY_MOE_BUFFER_MB=%zu\n", c.moe_budget_mb);
        printf("  export LLAMA_LAZY_MOE_BUFFER_WORKERS=4\n");
        printf("  export LLAMA_LAZY_CLG=1\n");
        printf("  export LLAMA_LAZY_MOE_BUFFER_HOT_RATIO=2.0\n");
    } else {
        printf("  # no feasible candidate found under this RAM budget\n");
    }

    printf("\nexample command:\n");
    printf("  llama-cli -m %s -c %" PRIu32 " -b %d -ub %d -ctk %s -ctv %s",
            shell_quote(params.model).c_str(), params.n_ctx, c.batch, c.ubatch,
            ggml_type_name(c.kv_type), ggml_type_name(c.kv_type));
    if (c.parallel > 1) {
        printf(" -np %d", c.parallel);
    }
    if (c.mode == "mmap-no-repack") {
        printf(" --no-repack");
    }
    printf("\n");
}

} // namespace

int main(int argc, char ** argv) {
    planner_params params;
    if (!parse_args(argc, argv, params)) {
        usage();
        return 1;
    }

    llama_backend_init();
    llama_log_set(planner_log_callback, nullptr);

    if (params.ram_budget == 0) {
        params.ram_budget = detect_ram_budget();
    }
    if (params.ram_budget == 0) {
        fprintf(stderr, "failed to detect RAM budget; pass --ram-budget\n");
        return 1;
    }

    llama_model * model_no_repack = load_model_for_estimate(params, false);
    if (model_no_repack == nullptr) {
        fprintf(stderr, "failed to load model for no-repack estimate: %s\n", params.model.c_str());
        llama_backend_free();
        return 1;
    }
    llama_model * model_native = load_model_for_estimate(params, true);
    if (model_native == nullptr) {
        fprintf(stderr, "failed to load model for native estimate: %s\n", params.model.c_str());
        llama_model_free(model_no_repack);
        llama_backend_free();
        return 1;
    }

    const int n_expert = llama_model_n_expert(model_no_repack);
    std::vector<candidate> candidates;

    for (int parallel : params.parallels) {
        for (int batch : params.batches) {
            for (int ubatch : params.ubatches) {
                if (ubatch > batch) {
                    continue;
                }
                for (ggml_type kv_type : params.kv_types) {
                    candidate c;
                    c.parallel = parallel;
                    c.batch = batch;
                    c.ubatch = ubatch;
                    c.kv_type = kv_type;
                    c.no_repack = estimate_memory(model_no_repack, params, c);
                    c.native = estimate_memory(model_native, params, c);
                    score_candidate(c, params, n_expert);
                    if (c.score >= 0.0) {
                        candidates.push_back(c);
                    }
                }
            }
        }
    }

    if (candidates.empty()) {
        candidate c;
        c.mode = "unfit";
        print_candidate(c, params, n_expert);
        llama_model_free(model_native);
        llama_model_free(model_no_repack);
        llama_backend_free();
        return 2;
    }

    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        return a.score > b.score;
    });

    print_candidate(candidates.front(), params, n_expert);
    llama_model_free(model_native);
    llama_model_free(model_no_repack);
    llama_backend_free();
    return 0;
}
