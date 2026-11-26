// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "common/rpu/tp_contract.hpp"

namespace xinfer::rpu::tp {
inline bool is_cache_layout(const std::string& layout) {
  return layout == "vision_k_cache" || layout == "vision_v_cache" ||
         layout == "causal_k_cache" || layout == "causal_v_cache";
}

// Element offset in the device's tiled DDR cache. Public host data remains
// [core, physical token capacity, local channel]. K tiles store token/channel;
// V tiles transpose those two inner axes. The separate H2/D64 vision axes
// collapse to eight channel tiles; causal H1/D256 has sixteen channel tiles.
// This codec is a production layout utility, not a numerical reference kernel.
inline std::uint64_t cache_offset(const Tensor& tensor, unsigned core,
                                  unsigned token, unsigned channel) {
  require(is_cache_layout(tensor.layout), "unsupported cache layout");
  require(tensor.storage == "ddr" && tensor.dtype == "float16",
          "cache codec requires FP16 DDR storage");
  const bool vision = tensor.layout == "vision_k_cache" || tensor.layout == "vision_v_cache";
  const unsigned width = vision ? 128 : 256;
  require(tensor.shape.size() == 3 && tensor.shape[0] == 8 &&
              tensor.shape[1] > 0 && tensor.shape[1] % 16 == 0 &&
              tensor.shape[1] <= 65520 && tensor.shape[2] == width,
          "cache codec expects [8,16-aligned physical capacity,128 or 256]");
  require(tensor.bytes >= 2 * elements(tensor), "cache codec allocation is too short");
  require(core < 8 && token < tensor.shape[1] && channel < width,
          "cache coordinate is out of bounds");
  const auto tile = (std::uint64_t(token / 16) * (width / 16) + channel / 16) * 8 + core;
  const bool value = tensor.layout == "vision_v_cache" || tensor.layout == "causal_v_cache";
  return tile * 256 + (value ? (channel % 16) * 16 + token % 16
                            : (token % 16) * 16 + channel % 16);
}
}  // namespace xinfer::rpu::tp
