// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/backend.hpp"

#include "prowl/errors.hpp"

namespace prowl {

void Backend::hermitian_eig(std::complex<double>*, int, double*) {
    fail(ErrorCode::BackendUnavailable, "backend '" + name() + "' has no dense eigensolver");
}

std::unique_ptr<RepairStepper> Backend::make_repair_stepper(const std::complex<double>*, int, int, int, int,
                                                          double, double) {
    return nullptr;
}

std::vector<std::string> available_backends() {
    std::vector<std::string> v{"cpu"};
#ifdef PROWL_HAVE_CUDA
    if (cuda_device_available()) v.emplace_back("cuda");
#endif
    return v;
}

std::unique_ptr<Backend> make_backend(const std::string& kind) {
    if (kind == "cpu") return make_cpu_backend();
    if (kind == "cuda" || kind == "auto") {
#ifdef PROWL_HAVE_CUDA
        if (cuda_device_available()) return make_cuda_backend(0);
        if (kind == "cuda") fail(ErrorCode::BackendUnavailable, "CUDA backend compiled in but no usable NVIDIA device found");
#else
        if (kind == "cuda") fail(ErrorCode::BackendUnavailable, "this build has no CUDA backend (configure with PROWL_ENABLE_CUDA=ON)");
#endif
        return make_cpu_backend();
    }
    fail(ErrorCode::InvalidArgument, "unknown backend '" + kind + "' (cpu|cuda|auto)");
}

} // namespace prowl
