// Portions of this file are derived from NVIDIA CUTLASS (https://github.com/NVIDIA/cutlass).
// Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Redistribution and use are subject to the BSD 3-Clause license in third_party/cutlass/LICENSE.txt.
//
// SM120 CUTLASS block-scaled NVFP4 x NVFP4 -> BF16 GEMMs (GEMM family backend cutlass_nvfp4_sm120): the operands of
// gemm_generic_nvfp4 (packed E2M1 [rows, K / 2], E4M3 1x16 scales in the SWIZZLE_32_4_4 storage) and its result,
// out = bf16(alpha * acc (+ bias)), with the residual added after that rounding: out = bf16(out + residual).
//
// The mainloop is the cuBLASLt one's arithmetic (block-scaled MMA, K in order, no split), so the output is the
// generic path's bit for bit; what differs is the tile, picked per shape by TileFor().
#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include <cstdint>

namespace xinfer::nvfp4_cutlass_sm120 {

using namespace cute;
using NV = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using BF = cutlass::bfloat16_t;
namespace fusion = cutlass::epilogue::fusion;
constexpr auto kRn = cutlass::FloatRoundStyle::round_to_nearest;

// The tiles (M x N x K): wide outputs use the larger tiles, narrow ones the smaller tiles that fill more SMs.
enum class Tile : int { k128x128 = 0, k256x128 = 1, k128x64 = 2, k128x32 = 3 };

// The tile of an [m, n] output on `sms` SMs (tests/perf scan, RTX 5090): N <= 512 takes 128 x 32; else 128 x 64
// while 128 x 128 tiles would fill at most half the SMs; else 128 x 128 while they fit one wave; else 256 x 128
// while those fit one wave; else 128 x 128.
inline Tile TileFor(int64_t m, int64_t n, int sms) {
  const int64_t tiles128 = (m + 127) / 128 * ((n + 127) / 128);
  if (n <= 512) return Tile::k128x32;
  if (2 * tiles128 <= sms) return Tile::k128x64;
  if (tiles128 <= sms) return Tile::k128x128;
  if ((m + 255) / 256 * ((n + 127) / 128) <= sms) return Tile::k256x128;
  return Tile::k128x128;
}

template <Tile T>
struct TileTraits;
template <>
struct TileTraits<Tile::k128x128> {
  using Shape = cute::Shape<_128, _128, _128>;
};
template <>
struct TileTraits<Tile::k256x128> {
  using Shape = cute::Shape<_256, _128, _128>;
};
template <>
struct TileTraits<Tile::k128x64> {
  using Shape = cute::Shape<_128, _64, _256>;
};
template <>
struct TileTraits<Tile::k128x32> {
  using Shape = cute::Shape<_128, _32, _128>;
};

// bf16(alpha * acc), or bf16(alpha * acc + bias[n])
template <class TileShape, bool Bias>
struct Scaled {
  using Op = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, BF, float, kRn>,
                             fusion::Sm90ScalarBroadcast<float>, fusion::Sm90AccFetch>;
};
template <class TileShape>
struct Scaled<TileShape, true> {
  using Op = fusion::Sm90EVT<
      fusion::Sm90Compute<cutlass::plus, BF, float, kRn>,
      fusion::Sm90EVT<fusion::Sm90Compute<cutlass::multiplies, float, float, kRn>, fusion::Sm90ScalarBroadcast<float>,
                      fusion::Sm90AccFetch>,
      fusion::Sm90RowBroadcast<0, TileShape, BF, float, Stride<_0, _1, int64_t>, 8>>;
};

// ... then + residual (the C operand): bf16(float(that) + residual)
template <class TileShape, bool Bias, bool Residual>
struct Epilogue {
  using Op = typename Scaled<TileShape, Bias>::Op;
};
template <class TileShape, bool Bias>
struct Epilogue<TileShape, Bias, true> {
  using Op = fusion::Sm90EVT<fusion::Sm90Compute<cutlass::plus, BF, float, kRn>, typename Scaled<TileShape, Bias>::Op,
                             fusion::Sm90SrcFetch<BF>>;
};

template <Tile T, bool Bias, bool Residual>
struct Gemm {
  using TileShape = typename TileTraits<T>::Shape;
  using Cluster = Shape<_1, _1, _1>;
  using ElementC = cute::conditional_t<Residual, BF, void>;
  using Row = cutlass::layout::RowMajor;
  using Col = cutlass::layout::ColumnMajor;
  using Callbacks = typename Epilogue<TileShape, Bias, Residual>::Op;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, ElementC, Row, 8, BF, Row, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto, Callbacks>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, NV, Row, 32, NV, Col, 32, float, TileShape,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, CollectiveEpilogue>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static typename Callbacks::Arguments FusionArgs(float alpha, const BF* bias) {
    if constexpr (Bias) {
      typename Scaled<TileShape, true>::Op::Arguments scaled{{{alpha}, {}, {}}, {bias, BF(0), {}}, {}};
      if constexpr (Residual) {
        return {scaled, {}, {}};
      } else {
        return scaled;
      }
    } else {
      typename Scaled<TileShape, false>::Op::Arguments scaled{{{alpha}}, {}, {}};
      if constexpr (Residual) {
        return {scaled, {}, {}};
      } else {
        return scaled;
      }
    }
  }

  // Runs on `stream`; returns the CUTLASS status (the workspace is empty).
  static cutlass::Status Run(const void* a, const void* a_scale, const void* w, const void* w_scale, BF* out,
                             const BF* bias, const BF* residual, float alpha, int m, int n, int k, int sms,
                             cudaStream_t stream) {
    auto da = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1});
    auto db = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1});
    auto dc = cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m, n, 1});
    auto dd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1});
    using Config = typename Mainloop::Sm1xxBlkScaledConfig;
    auto la = Config::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto lb = Config::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    cutlass::KernelHardwareInfo hw;
    hw.sm_count = sms;
    typename Device::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {reinterpret_cast<NV::DataType const*>(a), da, reinterpret_cast<NV::DataType const*>(w), db,
         reinterpret_cast<NV::ScaleFactorType const*>(a_scale), la,
         reinterpret_cast<NV::ScaleFactorType const*>(w_scale), lb},
        {FusionArgs(alpha, bias), residual, dc, out, dd},
        hw};
    Device gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;
    if (Device::get_workspace_size(args) != 0) return cutlass::Status::kErrorWorkspaceNull;
    status = gemm.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return status;
    return gemm.run(stream);
  }
};

// The kernel of (tile, bias, residual).
inline cutlass::Status Run(Tile tile, const void* a, const void* a_scale, const void* w, const void* w_scale, BF* out,
                           const BF* bias, const BF* residual, float alpha, int m, int n, int k, int sms,
                           cudaStream_t stream) {
#define XC_NVFP4_RUN(T)                                                                                          \
  case T:                                                                                                         \
    if (bias && residual)                                                                                         \
      return Gemm<T, true, true>::Run(a, a_scale, w, w_scale, out, bias, residual, alpha, m, n, k, sms, stream);  \
    if (bias) return Gemm<T, true, false>::Run(a, a_scale, w, w_scale, out, bias, residual, alpha, m, n, k, sms, stream); \
    if (residual)                                                                                                 \
      return Gemm<T, false, true>::Run(a, a_scale, w, w_scale, out, bias, residual, alpha, m, n, k, sms, stream); \
    return Gemm<T, false, false>::Run(a, a_scale, w, w_scale, out, bias, residual, alpha, m, n, k, sms, stream);
  switch (tile) {
    XC_NVFP4_RUN(Tile::k128x128)
    XC_NVFP4_RUN(Tile::k256x128)
    XC_NVFP4_RUN(Tile::k128x64)
    XC_NVFP4_RUN(Tile::k128x32)
  }
#undef XC_NVFP4_RUN
  return cutlass::Status::kErrorInvalidProblem;
}

}  // namespace xinfer::nvfp4_cutlass_sm120
