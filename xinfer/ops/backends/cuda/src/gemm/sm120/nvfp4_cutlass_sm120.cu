// Portions of this file are derived from NVIDIA CUTLASS (https://github.com/NVIDIA/cutlass).
// Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Redistribution and use are subject to the BSD 3-Clause license in third_party/cutlass/LICENSE.txt.
//
// TVM FFI entry of the SM120 CUTLASS NVFP4 GEMMs (nvfp4_cutlass_sm120.cuh), GEMM family backend
// cutlass_nvfp4_sm120:
//   nvfp4_gemm_cutlass_sm120(out, a, a_scale, w, w_scale, bias?, residual?, alpha, tile)
//     out = bf16(alpha * (A @ W^T) + bias), then with a residual out = bf16(out + residual);
//   nvfp4_gemm_act_quant_cutlass_sm120(q, scales, a, a_scale, w, w_scale, bias?, alpha, act, tile)
//     (q, scales) = quantize_nvfp4(bf16(act(bf16(alpha * (A @ W^T) + bias)))), act 1 gelu (tanh), 2 gelu (erf);
//   tile -1 picks the tile by TileFor(), 0..3 forces one (tests). TMA operands, the residual and the bias are
//   16-byte aligned.
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <atomic>
#include <cmath>
#include <string>

#include "gemm/sm120/cutlass_ffi.h"
#include "gemm/sm120/nvfp4_cutlass_sm120.cuh"
#include "gemm/sm120/nvfp4_cutlass_sm120_quant.cuh"

namespace xinfer::nvfp4_cutlass_sm120 {
namespace {

using cutlass_ffi::CheckAligned;
using cutlass_ffi::CheckBf16;
using cutlass_ffi::CheckCapability120;
using cutlass_ffi::CheckContiguous;
using cutlass_ffi::CheckCuda;
using cutlass_ffi::CheckedInt;
using cutlass_ffi::CheckRank;
using cutlass_ffi::CheckSameDevice;
using cutlass_ffi::CheckUInt8;
using cutlass_ffi::DeviceGuard;
using cutlass_ffi::HasDtype;
using cutlass_ffi::Numel;
using cutlass_ffi::Require;
using cutlass_ffi::RoundUp;
using cutlass_ffi::Stream;

constexpr const char* kOp = "nvfp4_gemm_cutlass_sm120";

void CheckE4m3Scales(const tvm::ffi::TensorView& t, int64_t rows, int64_t cols, const char* name,
                     const char* op = kOp) {
  Require(HasDtype(t, kDLFloat8_e4m3fn, 8) || HasDtype(t, kDLUInt, 8),
          std::string(op) + ": " + name + " must have float8_e4m3fn or uint8 storage");
  Require(Numel(t) >= RoundUp(rows, 128) * RoundUp(cols / 16, 4),
          std::string(op) + ": " + name + " storage too small (SWIZZLE_32_4_4, rows padded to 128, 1x16 blocks to 4)");
}

int SmCount(int device) {
  constexpr int kMaxDevices = 64;
  static std::atomic<int> cached[kMaxDevices];
  int sms = device >= 0 && device < kMaxDevices ? cached[device].load(std::memory_order_relaxed) : 0;
  if (sms == 0) {
    Require(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess && sms > 0,
            std::string(kOp) + ": cannot query the SM count");
    if (device >= 0 && device < kMaxDevices) cached[device].store(sms, std::memory_order_relaxed);
  }
  return sms;
}

}  // namespace

void Nvfp4GemmCutlassSm120(tvm::ffi::TensorView out, tvm::ffi::TensorView a, tvm::ffi::TensorView a_scale,
                           tvm::ffi::TensorView w, tvm::ffi::TensorView w_scale,
                           tvm::ffi::Optional<tvm::ffi::TensorView> bias,
                           tvm::ffi::Optional<tvm::ffi::TensorView> residual, double alpha, int64_t tile) {
  CheckCuda(a, "a");
  CheckCapability120(a, kOp);
  CheckRank(a, 2, "a");
  CheckRank(w, 2, "w");
  CheckRank(out, 2, "out");
  CheckUInt8(a, "a");
  CheckUInt8(w, "w");
  CheckBf16(out, "out");
  for (const auto& [t, name] : {std::pair{a_scale, "a_scale"}, std::pair{w, "w"}, std::pair{w_scale, "w_scale"},
                                std::pair{out, "out"}}) {
    CheckSameDevice(t, a, name);
  }
  for (const auto& [t, name] : {std::pair{a, "a"}, std::pair{w, "w"}, std::pair{a_scale, "a_scale"},
                                std::pair{w_scale, "w_scale"}, std::pair{out, "out"}}) {
    CheckContiguous(t, name);
    CheckAligned(t, 16, name);  // TMA operands
  }
  const int m = CheckedInt(a.size(0), "M"), k = CheckedInt(2 * a.size(1), "K"), n = CheckedInt(w.size(0), "N");
  Require(m > 0 && 2 * w.size(1) == k && k % 128 == 0 && n % 16 == 0 && out.size(0) == m && out.size(1) == n,
          std::string(kOp) + ": need a [M, K / 2], w [N, K / 2], out [M, N], K % 128 == 0, N % 16 == 0");
  CheckE4m3Scales(a_scale, m, k, "a_scale");
  CheckE4m3Scales(w_scale, n, k, "w_scale");
  const BF* bias_ptr = nullptr;
  if (bias.has_value()) {
    const tvm::ffi::TensorView& b = bias.value();
    CheckSameDevice(b, a, "bias");
    CheckBf16(b, "bias");
    CheckContiguous(b, "bias");
    CheckAligned(b, 16, "bias");  // read 8 elements at a time by the EVT's row broadcast
    Require(Numel(b) == n, std::string(kOp) + ": bias must have N elements");
    bias_ptr = reinterpret_cast<const BF*>(b.data_ptr());
  }
  const BF* residual_ptr = nullptr;
  if (residual.has_value()) {
    const tvm::ffi::TensorView& r = residual.value();
    CheckSameDevice(r, a, "residual");
    CheckBf16(r, "residual");
    CheckContiguous(r, "residual");
    CheckAligned(r, 16, "residual");
    Require(r.ndim() == 2 && r.size(0) == m && r.size(1) == n, std::string(kOp) + ": residual must be [M, N]");
    residual_ptr = reinterpret_cast<const BF*>(r.data_ptr());
  }
  const float alpha_f = static_cast<float>(alpha);
  Require(alpha_f > 0.0f && std::isfinite(alpha_f), std::string(kOp) + ": alpha must be finite and positive in FP32");
  Require(tile >= -1 && tile <= 3, std::string(kOp) + ": tile must be -1 (by shape) or 0..3");
  DeviceGuard guard(a.device().device_id);
  const int sms = SmCount(a.device().device_id);
  const Tile t = tile < 0 ? TileFor(m, n, sms) : static_cast<Tile>(tile);
  const cutlass::Status status =
      Run(t, a.data_ptr(), a_scale.data_ptr(), w.data_ptr(), w_scale.data_ptr(), reinterpret_cast<BF*>(out.data_ptr()),
          bias_ptr, residual_ptr, alpha_f, m, n, k, sms, Stream(a));
  Require(status == cutlass::Status::kSuccess, std::string(kOp) + ": " + cutlassGetStatusString(status));
  const cudaError_t err = cudaGetLastError();
  Require(err == cudaSuccess, std::string(kOp) + ": launch failed: " + cudaGetErrorString(err));
}

void Nvfp4GemmActQuantCutlassSm120(tvm::ffi::TensorView q, tvm::ffi::TensorView scales, tvm::ffi::TensorView a,
                                   tvm::ffi::TensorView a_scale, tvm::ffi::TensorView w, tvm::ffi::TensorView w_scale,
                                   tvm::ffi::Optional<tvm::ffi::TensorView> bias, double alpha, int64_t act,
                                   int64_t tile) {
  constexpr const char* op = "nvfp4_gemm_act_quant_cutlass_sm120";
  CheckCuda(a, "a");
  CheckCapability120(a, op);
  CheckRank(a, 2, "a");
  CheckRank(w, 2, "w");
  CheckRank(q, 2, "q");
  CheckUInt8(a, "a");
  CheckUInt8(w, "w");
  CheckUInt8(q, "q");
  for (const auto& [t, name] : {std::pair{a_scale, "a_scale"}, std::pair{w, "w"}, std::pair{w_scale, "w_scale"},
                                std::pair{q, "q"}, std::pair{scales, "scales"}}) {
    CheckSameDevice(t, a, name);
  }
  for (const auto& [t, name] : {std::pair{a, "a"}, std::pair{w, "w"}, std::pair{a_scale, "a_scale"},
                                std::pair{w_scale, "w_scale"}, std::pair{q, "q"}, std::pair{scales, "scales"}}) {
    CheckContiguous(t, name);
    CheckAligned(t, 16, name);  // TMA operands
  }
  const int m = CheckedInt(a.size(0), "M"), k = CheckedInt(2 * a.size(1), "K"), n = CheckedInt(w.size(0), "N");
  Require(m > 0 && 2 * w.size(1) == k && k % 128 == 0 && n % 64 == 0 && q.size(0) == m && 2 * q.size(1) == n,
          std::string(op) + ": need a [M, K / 2], w [N, K / 2], q [M, N / 2], K % 128 == 0, N % 64 == 0");
  CheckE4m3Scales(a_scale, m, k, "a_scale", op);
  CheckE4m3Scales(w_scale, n, k, "w_scale", op);
  CheckE4m3Scales(scales, m, n, "scales", op);
  const BF* bias_ptr = nullptr;
  if (bias.has_value()) {
    const tvm::ffi::TensorView& b = bias.value();
    CheckSameDevice(b, a, "bias");
    CheckBf16(b, "bias");
    CheckContiguous(b, "bias");
    CheckAligned(b, 16, "bias");  // read 8 elements at a time by the EVT's row broadcast
    Require(Numel(b) == n, std::string(op) + ": bias must have N elements");
    bias_ptr = reinterpret_cast<const BF*>(b.data_ptr());
  }
  const float alpha_f = static_cast<float>(alpha);
  Require(alpha_f > 0.0f && std::isfinite(alpha_f), std::string(op) + ": alpha must be finite and positive in FP32");
  Require(act == 1 || act == 2, std::string(op) + ": act must be 1 (gelu, tanh) or 2 (gelu, erf)");
  Require(tile >= -1 && tile <= 3, std::string(op) + ": tile must be -1 (by shape) or 0..3");
  DeviceGuard guard(a.device().device_id);
  const int sms = SmCount(a.device().device_id);
  const Tile t = tile < 0 ? TileFor(m, n, sms) : static_cast<Tile>(tile);
  const cutlass::Status status =
      RunActQuant(t, static_cast<int>(act), a.data_ptr(), a_scale.data_ptr(), w.data_ptr(), w_scale.data_ptr(),
                  q.data_ptr(), scales.data_ptr(), bias_ptr, alpha_f, m, n, k, sms, Stream(a));
  Require(status == cutlass::Status::kSuccess, std::string(op) + ": " + cutlassGetStatusString(status));
  const cudaError_t err = cudaGetLastError();
  Require(err == cudaSuccess, std::string(op) + ": launch failed: " + cudaGetErrorString(err));
}

// The tile TileFor() picks for an [m, n] output on the device's SMs: 0 128x128, 1 256x128, 2 128x64, 3 128x32.
int64_t Nvfp4CutlassSm120Tile(int64_t m, int64_t n, int64_t device) {
  return static_cast<int64_t>(TileFor(m, n, SmCount(static_cast<int>(device))));
}

}  // namespace xinfer::nvfp4_cutlass_sm120

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_gemm_cutlass_sm120, xinfer::nvfp4_cutlass_sm120::Nvfp4GemmCutlassSm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_gemm_act_quant_cutlass_sm120,
                              xinfer::nvfp4_cutlass_sm120::Nvfp4GemmActQuantCutlassSm120);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvfp4_cutlass_sm120_tile, xinfer::nvfp4_cutlass_sm120::Nvfp4CutlassSm120Tile);
