// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__GNUC__)
#define XINFER_RPU_API __attribute__((visibility("default")))
#else
#define XINFER_RPU_API
#endif

namespace rhino_lkn { class Program_t; }

namespace xinfer::rpu {
// Requires the external, release-matched Rhino Launch Program_t C++ ABI.
// The caller owns Program_t, Kernel_t, all buffers and all launch queues.
XINFER_RPU_API std::uint32_t library_abi_version() noexcept;
XINFER_RPU_API std::size_t kernel_count() noexcept;
XINFER_RPU_API const char* kernel_name(std::size_t index) noexcept;
// Empty metadata means the catalog does not declare the corresponding field.
// Unspecified admission is not a production-readiness assertion.
XINFER_RPU_API const char* kernel_admission(std::size_t index) noexcept;
XINFER_RPU_API const char* kernel_execution_status(std::size_t index) noexcept;
// Catalog membership only; does not establish asset presence or execution support.
XINFER_RPU_API bool has_kernel(const char* name) noexcept;

// Load a catalog-listed external asset into a caller-owned, fresh Program_t.
// No board work is enqueued. Keep xinfer_rpu.so loaded while using the program.
// Candidate assets require explicit opt-in. Successful loading does not validate launch, shape,
// synchronization, or Graph replay requirements. In particular, qkv_expand_kv
// requires its extra atomic register initialization in the caller.
// Returns 0 on success; -1 unknown entry, -2 asset loading failure,
// -3 candidate disabled, -6 unrecognized admission/status
// metadata, or a nonzero Rhino Launch status.
// Errors never fall back silently.
XINFER_RPU_API std::int32_t load_program(rhino_lkn::Program_t& program,
                                         const char* name,
                                         bool allow_candidate = false) noexcept;
// Thread-local diagnostic; valid until the next load_program call on this thread.
XINFER_RPU_API const char* last_error() noexcept;
}  // namespace xinfer::rpu

// Unmangled exports of the unified xinfer library. Program_t still requires
// the release-matched external Launch C++ ABI. allow_candidate must be 0 or 1;
// an invalid flag returns -5. These exports do not enqueue device work.
extern "C" {
XINFER_RPU_API std::uint32_t xinfer_rpu_abi_version_v1() noexcept;
XINFER_RPU_API std::int32_t xinfer_rpu_load_program_v1(
    rhino_lkn::Program_t*, const char*, std::uint32_t allow_candidate) noexcept;
XINFER_RPU_API const char* xinfer_rpu_last_error_v1() noexcept;
}
