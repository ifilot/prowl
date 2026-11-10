// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
/// \file errors.hpp
/// Typed conditions for checked input, numerical and backend failures. Error carries
/// a machine-readable code and an actionable message. Standard allocation or JSON
/// exceptions can also propagate from APIs where that behavior is documented.

#include <stdexcept>
#include <string>
#include <string_view>

namespace prowl {

enum class ErrorCode {
    InvalidArgument,      ///< caller passed inconsistent arguments
    FileNotFound,         ///< required input file missing
    CorruptFile,          ///< file layout violates the format contract (truncated, bad record)
    UnsupportedFeature,   ///< valid input whose physics is not (yet) supported, e.g. spinors
    AmbiguousFormat,      ///< layout cannot be determined without an explicit override
    InconsistentInput,    ///< inputs disagree (e.g. POSCAR vs WAVECAR lattice)
    NumericalFailure,     ///< a numerical invariant failed (norm, conditioning, ...)
    BackendUnavailable,   ///< requested compute backend not compiled in / no device
    DeviceError,          ///< CUDA / library error on the device
    OutOfMemory,          ///< allocation exceeds configured budget or device memory
    Internal              ///< bug: an internal invariant was violated
};

/**
 * @brief Names a typed error code without allocating.
 * @param[in] c Error classification.
 * @return Borrowed static name, or Unknown for an unrecognized value; valid for the process lifetime.
 */
[[nodiscard]] constexpr std::string_view to_string(ErrorCode c) noexcept {
    switch (c) {
    case ErrorCode::InvalidArgument: return "InvalidArgument";
    case ErrorCode::FileNotFound: return "FileNotFound";
    case ErrorCode::CorruptFile: return "CorruptFile";
    case ErrorCode::UnsupportedFeature: return "UnsupportedFeature";
    case ErrorCode::AmbiguousFormat: return "AmbiguousFormat";
    case ErrorCode::InconsistentInput: return "InconsistentInput";
    case ErrorCode::NumericalFailure: return "NumericalFailure";
    case ErrorCode::BackendUnavailable: return "BackendUnavailable";
    case ErrorCode::DeviceError: return "DeviceError";
    case ErrorCode::OutOfMemory: return "OutOfMemory";
    case ErrorCode::Internal: return "Internal";
    }
    return "Unknown";
}

class Error : public std::runtime_error {
public:
    /**
     * @brief Builds an owning exception message with a machine-readable classification.
     * @param[in] code Failure classification.
     * @param[in] msg Diagnostic copied into the runtime_error message.
     */
    Error(ErrorCode code, const std::string& msg)
        : std::runtime_error(std::string(to_string(code)) + ": " + msg), code_(code) {}
    /**
     * @brief Retrieves this exception classification.
     * @return The ErrorCode supplied at construction.
     */
    [[nodiscard]] ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

/**
 * @brief Throws a typed failure with an actionable diagnostic.
 * @param[in] code Failure classification.
 * @param[in] msg Message copied into the exception.
 * @throws Error Always; this function never returns.
 */
[[noreturn]] inline void fail(ErrorCode code, const std::string& msg) { throw Error(code, msg); }

/**
 * @brief Enforces a checked precondition or numerical invariant.
 * @param[in] cond Condition that must hold.
 * @param[in] code Classification used if cond is false.
 * @param[in] msg Failure diagnostic.
 * @throws Error When cond is false; otherwise has no side effects.
 */
inline void require(bool cond, ErrorCode code, const std::string& msg) {
    if (!cond) fail(code, msg);
}

} // namespace prowl
