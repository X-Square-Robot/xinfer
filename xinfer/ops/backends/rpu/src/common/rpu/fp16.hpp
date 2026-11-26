// Copyright 2026 X-Square Robot
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstring>

namespace xinfer::rpu {

// IEEE-754 binary32 -> binary16, round-to-nearest-even. Keeping this helper
// independent of PyTorch makes the host oracle usable on a clean board image.
inline std::uint16_t fp32_to_fp16(float value) noexcept {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t source_exponent = (bits >> 23U) & 0xffU;
  std::uint32_t fraction = bits & 0x7fffffU;

  if (source_exponent == 0xffU) {
    if (fraction == 0U) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    std::uint32_t payload = fraction >> 13U;
    if (payload == 0U) {
      payload = 1U;
    }
    return static_cast<std::uint16_t>(sign | 0x7c00U | payload);
  }

  int exponent = static_cast<int>(source_exponent) - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    fraction |= 0x800000U;
    const int shift = 14 - exponent;
    std::uint32_t rounded = fraction >> shift;
    const std::uint32_t halfway = 1U << (shift - 1);
    if ((fraction & halfway) != 0U &&
        (((fraction & (halfway - 1U)) != 0U) || ((rounded & 1U) != 0U))) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }

  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }

  std::uint32_t rounded = fraction >> 13U;
  const std::uint32_t discarded = fraction & 0x1fffU;
  if (discarded > 0x1000U ||
      (discarded == 0x1000U && (rounded & 1U) != 0U)) {
    ++rounded;
  }
  if (rounded == 0x400U) {
    rounded = 0U;
    ++exponent;
  }
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint32_t>(exponent) << 10U) | rounded);
}

inline float fp16_to_fp32(std::uint16_t value) noexcept {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t fraction = value & 0x3ffU;
  std::uint32_t bits = 0U;

  if (exponent == 0U) {
    if (fraction == 0U) {
      bits = sign;
    } else {
      int shift = -1;
      do {
        ++shift;
        fraction <<= 1U;
      } while ((fraction & 0x400U) == 0U);
      fraction &= 0x3ffU;
      bits = sign |
             (static_cast<std::uint32_t>(127 - 15 - shift) << 23U) |
             (fraction << 13U);
    }
  } else if (exponent == 31U) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 127U - 15U) << 23U) | (fraction << 13U);
  }

  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

inline float round_to_fp16(float value) noexcept {
  return fp16_to_fp32(fp32_to_fp16(value));
}

}  // namespace xinfer::rpu
