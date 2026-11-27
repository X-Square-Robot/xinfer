// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/tp_contract.hpp"

#include <map>
#include <set>
#include <string>

// Family entry points of the TP8 operator API.
//
// RpuTpContext.prepare() accepts either a family op (gemm, qkv_proj, rmsnorm,
// ...) with the same semantic argument names as the CUDA family API, or a
// catalog entry name. A family request is rewritten here, before any
// validation or lowering, into the catalog entry and argument names that the
// per-family lowerers already implement; nothing downstream changes, so a
// family request and the equivalent catalog request lower to the same launch
// packet. Variant choices that change numerics stay explicit arguments
// (activation, mode, ...), never inferred.
namespace xinfer::rpu::tp {

// True when op names a family entry point (as opposed to a catalog entry).
XINFER_RPU_API bool is_family(const std::string& op);

// Rewrite a family request in place. Catalog requests are left unchanged and
// return false. renamed receives caller tensor name -> catalog tensor name for
// every renamed argument (unrenamed arguments keep their name). Throws
// std::invalid_argument with the family, argument and violated condition when
// the request matches no implementation; the error lists what is supported.
// selectors receives the family-only integer selectors (parallel, mode, ...)
// that chose the kernel; they are fixed at prepare time.
XINFER_RPU_API bool resolve_family(Request& request,
                                     std::map<std::string, std::string>* renamed = nullptr,
                                     std::set<std::string>* selectors = nullptr);

}  // namespace xinfer::rpu::tp
