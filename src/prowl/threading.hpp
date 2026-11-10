// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

namespace prowl {

/**
 * @brief Holds a process-wide OpenBLAS thread policy for a native computation.
 *
 * Computations requesting the same policy can overlap. A different policy waits for the current users to
 * finish; the last user restores the previous OpenBLAS setting. External calls to OpenBLAS thread setters or
 * threadpoolctl must not overlap these scopes. Construct and release each scope on the same thread.
 */
class BlasThreads {
public:
    /**
     * @brief Acquires a shared lease on a BLAS thread count, waiting if a different count is active.
     * @param[in] threads Positive thread count, or zero to inherit an enclosing scope (one otherwise).
     * @throws Error If threads is negative or a nested scope requests a different policy.
     */
    explicit BlasThreads(int threads = 0);
    /** @brief Releases this scope's lease, restoring the prior setting when it was the last user. */
    ~BlasThreads();
    BlasThreads(const BlasThreads&) = delete;
    BlasThreads& operator=(const BlasThreads&) = delete;
    /** @brief Releases the lease early; repeated calls have no effect. */
    void release() noexcept;

private:
    bool active_{false};
};

} // namespace prowl
