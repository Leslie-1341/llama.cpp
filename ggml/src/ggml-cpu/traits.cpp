#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CPU_REPACK
#include "repack.h"
#endif

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
#ifdef GGML_USE_CPU_REPACK
    // Try JIT repack first: handles mmap-backed weight tensors (Lazy V2 mode)
    // by repacking into the work buffer on demand for the fast GEMM kernel.
    {
        auto * jit    = ggml_cpu_jit_repack_extra_buffer_type();
        auto * traits = jit->get_tensor_traits(op);
        if (traits && traits->compute_forward(params, op)) {
            return true;
        }
    }
#endif
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
#ifdef GGML_USE_CPU_REPACK
    // JIT repack needs extra work buffer space (repacked weights + kernel q8 + metadata)
    {
        auto * jit    = ggml_cpu_jit_repack_extra_buffer_type();
        auto * traits = jit->get_tensor_traits(op);
        if (traits && traits->work_size(n_threads, op, *size)) {
            return true;
        }
    }
#endif
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}

