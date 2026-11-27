// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/library.hpp"
#include <cstddef>
#include <cstdint>

namespace xinfer::rpu::execution {
// Process-local checked handles, never raw SDK pointers. Releasing a handle
// invalidates it immediately; prepared executions retain their dependencies.
using Handle = std::int64_t;
XINFER_RPU_API Handle context_create(std::int64_t core, bool allow_candidate);
XINFER_RPU_API void release(Handle handle);
XINFER_RPU_API Handle buffer_empty(Handle context, std::int64_t rows,
                                    std::int64_t cols);
XINFER_RPU_API void buffer_upload(Handle buffer, const std::uint16_t* data,
                                   std::size_t rows, std::size_t cols);
XINFER_RPU_API void buffer_download(std::uint16_t* data, Handle buffer,
                                     std::size_t rows, std::size_t cols);
// Input is immutable ordinary row-major W[N,K]. Pack and flush once here.
XINFER_RPU_API Handle prepare_weight(Handle context, const std::uint16_t* data,
                                      std::size_t n, std::size_t k);
XINFER_RPU_API Handle prepare_gemm(Handle output, Handle input, Handle weight,
                                    Handle bias);
XINFER_RPU_API Handle prepare_layernorm(Handle output, Handle input,
                                         Handle weight, Handle bias,
                                         double epsilon);
XINFER_RPU_API Handle prepare_silu_mul(Handle output, Handle x, Handle y);
// Preparation owns Program/Kernel/built Queue. Execute only replays that queue.
// One outstanding execution per device; wait/close completes pending work.
// Host copies/preparation are rejected while any adapter queue is in flight.
XINFER_RPU_API void execute(Handle prepared, bool asynchronous);
XINFER_RPU_API void wait(Handle prepared);
}  // namespace xinfer::rpu::execution
