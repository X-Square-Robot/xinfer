// Utilities for raw Ascend C-ABI launchers. Submissions use the caller's active torch_npu stream.
#pragma once

#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace xinfer::aot::ffi {

inline void Require(bool condition, const std::string& message) {
  if (!condition) throw std::invalid_argument(message);
}

inline void* CurrentStream(int32_t device) {
  int32_t current = -1;
  const auto error = aclrtGetDevice(&current);
  if (error != ACL_SUCCESS)
    throw std::runtime_error("aclrtGetDevice failed: " + std::to_string(error));
  Require(current == device, "tensor device must match the current Ascend device");
  const char* soc = aclrtGetSocName();
  Require(soc != nullptr && std::string_view(soc).rfind("Ascend950", 0) == 0,
          "AOT kernels require the Ascend950 architecture");
  auto* stream = c10_npu::getCurrentNPUStream(static_cast<c10::DeviceIndex>(device)).stream();
  Require(stream != nullptr, "Ascend current stream is unavailable");
  return stream;
}

}  // namespace xinfer::aot::ffi
