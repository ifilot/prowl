// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <exception>

namespace prowl::detail {
// Exceptions cannot escape OpenMP regions. Capture the first failure and rethrow after the region joins.
class ExceptionGuard {
public:
    template <class F> void run(F&& f) noexcept {
        try {
            f();
        } catch (...) {
#pragma omp critical(prowl_analysis_exception)
            if (!ptr_) ptr_ = std::current_exception();
        }
    }
    void rethrow() const {
        if (ptr_) std::rethrow_exception(ptr_);
    }

private:
    std::exception_ptr ptr_;
};
} // namespace prowl::detail
