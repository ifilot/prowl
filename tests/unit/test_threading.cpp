// SPDX-License-Identifier: GPL-3.0-or-later
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <future>
#include <thread>

#include "prowl/errors.hpp"
#include "prowl/threading.hpp"

extern "C" int openblas_get_num_threads();

TEST_CASE("BLAS policy leases share compatible work and restore the previous setting", "[threading]") {
    using namespace std::chrono_literals;
    const int previous = openblas_get_num_threads();
    std::promise<void> release;
    auto released = release.get_future().share();
    std::future<int> same, different;
    bool shared = false, waited = false, nested_rejected = false;
    int observed = 0;
    {
        prowl::BlasThreads policy(1);
        same = std::async(std::launch::async, [&] {
            prowl::BlasThreads shared_policy(1);
            return openblas_get_num_threads();
        });
        shared = same.wait_for(2s) == std::future_status::ready;
        different = std::async(std::launch::async, [&] {
            prowl::BlasThreads another_policy(2);
            const int count = openblas_get_num_threads();
            released.wait();
            return count;
        });
        waited = different.wait_for(20ms) == std::future_status::timeout;
        observed = openblas_get_num_threads();
        try {
            prowl::BlasThreads nested(2);
        } catch (const prowl::Error&) {
            nested_rejected = true;
        }
    }
    // Release the waiter before assertions, so a failing assertion cannot strand an async future.
    release.set_value();
    REQUIRE(shared);
    REQUIRE(waited);
    REQUIRE(observed == 1);
    REQUIRE(nested_rejected);
    REQUIRE(same.get() == 1);
    REQUIRE(different.get() == 2);
    REQUIRE(openblas_get_num_threads() == previous);
}
