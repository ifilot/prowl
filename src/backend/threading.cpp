// SPDX-License-Identifier: GPL-3.0-or-later
#include "prowl/threading.hpp"

#include <condition_variable>
#include <mutex>

#include "prowl/errors.hpp"

extern "C" {
void openblas_set_num_threads(int);
int openblas_get_num_threads();
}

namespace prowl {
namespace {
std::mutex policy_mutex;
std::condition_variable policy_changed;
int users = 0, policy = 0, saved = 0;
thread_local int local_users = 0, local_policy = 0;
} // namespace

BlasThreads::BlasThreads(int threads) {
    require(threads >= 0, ErrorCode::InvalidArgument, "BLAS thread count must be nonnegative");
    const int requested = threads ? threads : (local_users ? local_policy : 1);
    require(local_users == 0 || requested == local_policy, ErrorCode::InvalidArgument,
            "nested BLAS thread scopes must use the same policy");
    std::unique_lock lock(policy_mutex);
    policy_changed.wait(lock, [&] { return users == 0 || policy == requested; });
    if (users == 0) {
        saved = openblas_get_num_threads();
        policy = requested;
        openblas_set_num_threads(policy);
    }
    ++users;
    ++local_users;
    local_policy = requested;
    active_ = true;
}

BlasThreads::~BlasThreads() {
    release();
}

void BlasThreads::release() noexcept {
    if (!active_) return;
    std::lock_guard lock(policy_mutex);
    active_ = false;
    --local_users;
    if (--users == 0) {
        openblas_set_num_threads(saved);
        policy_changed.notify_all();
    }
}
} // namespace prowl
