// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#include "common/rpu/execution.hpp"
#include <tvm/ffi/function.h>
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_prepare_silu_mul, xinfer::rpu::execution::prepare_silu_mul);
