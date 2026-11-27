// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
// Host-only family resolution for TP8 operations: no device, SDK or context.
//
// rpu_tp_resolve(op, names, shapes, layouts, distributions, ikeys, ints)
// returns [entry, n, catalog_name_0 .. catalog_name_n-1, m, key_0, value_0 ..
// key_m-1, value_m-1]: the catalog entry, each tensor's catalog name in input
// order, and the final integer arguments (selectors consumed, values the
// family supplied added). Values are decimal strings. Callers that plan operations without a device (for
// example, framework tests that record bindings) use it to normalize a family
// call to the catalog entry and argument names the device would receive.
#include "common/rpu/tp_family.hpp"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace xinfer::rpu::tp::ffi {
using tvm::ffi::Array;
using tvm::ffi::String;

Array<String> resolve(String op, Array<String> names, Array<Array<std::int64_t>> shapes,
                      Array<String> layouts, Array<String> distributions, Array<String> ikeys,
                      Array<std::int64_t> ints) {
  if (names.size() != shapes.size() || names.size() != layouts.size() ||
      names.size() != distributions.size() || ikeys.size() != ints.size())
    throw std::invalid_argument("rpu_tp_resolve: parameter count mismatch");
  Request request;
  request.op = std::string(op);
  for (std::size_t i = 0; i < names.size(); ++i) {
    Tensor t;
    for (const auto d : shapes[i]) t.shape.push_back(d);
    t.layout = std::string(layouts[i]);
    t.distribution = std::string(distributions[i]);
    t.storage = t.layout == "row_major" && t.distribution != "shared" ? "spm" : "ddr";
    if (!request.tensors.emplace(std::string(names[i]), t).second)
      throw std::invalid_argument("rpu_tp_resolve: duplicate tensor name " +
                                  std::string(names[i]));
  }
  for (std::size_t i = 0; i < ikeys.size(); ++i)
    if (!request.integers.emplace(std::string(ikeys[i]), ints[i]).second)
      throw std::invalid_argument("rpu_tp_resolve: duplicate integer name " +
                                  std::string(ikeys[i]));
  std::map<std::string, std::string> renamed;
  resolve_family(request, &renamed);
  Array<String> result;
  result.push_back(String(request.op));
  result.push_back(String(std::to_string(names.size())));
  for (std::size_t i = 0; i < names.size(); ++i) {
    const auto found = renamed.find(std::string(names[i]));
    result.push_back(String(found == renamed.end() ? std::string(names[i]) : found->second));
  }
  result.push_back(String(std::to_string(request.integers.size())));
  for (const auto& item : request.integers) {
    result.push_back(String(item.first));
    result.push_back(String(std::to_string(item.second)));
  }
  return result;
}

bool is_family_name(String op) { return is_family(std::string(op)); }
}  // namespace xinfer::rpu::tp::ffi

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_resolve, xinfer::rpu::tp::ffi::resolve);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(rpu_tp_is_family, xinfer::rpu::tp::ffi::is_family_name);
