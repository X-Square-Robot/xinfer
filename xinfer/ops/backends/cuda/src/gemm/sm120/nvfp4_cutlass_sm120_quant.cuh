// Portions of this file are derived from NVIDIA CUTLASS (https://github.com/NVIDIA/cutlass).
// Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// Redistribution and use are subject to the BSD 3-Clause license in third_party/cutlass/LICENSE.txt.
//
// NVFP4 output of the SM120 CUTLASS NVFP4 GEMMs (GEMM family backend cutlass_nvfp4_sm120, gemm with a GELU and
// out_quant="nvfp4"): out = quantize_nvfp4(bf16(act(float(bf16(alpha * acc + bias))))), bit for bit what the generic
// path writes (gemm_generic_nvfp4, then gemm_epilogue_quant_nvfp4). XcNvfp4RowStore is CUTLASS's Sm120BlockScaleFactorRowStore with the quantizer's arithmetic: the block
// scale e4m3(min(448, max(2^-6, amax * rcp.approx(6)))), the elements x * rcp.approx(scale) (the epilogue then
// converts them with cvt.rn.satfinite.e2m1x2), and scale 0 for blocks outside [M, N] up to the padded storage.
#pragma once

#include "gemm/sm120/nvfp4_cutlass_sm120.cuh"
#include "cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp"
#include "quant/nvfp4_format_core.cuh"

namespace cutlass::epilogue::fusion {

template <
  int SFVecSize,
  class EpilogueTile,
  class CtaTileShapeMNK,
  int FragmentSize,
  class ElementOutput,
  class ElementCompute,
  class ElementBlockScaleFactor,
  FloatRoundStyle RoundStyle = FloatRoundStyle::round_to_nearest
>
struct XcNvfp4RowStore {

  static_assert(size<1>(EpilogueTile{}) % SFVecSize == 0, "EpilogueTileN should be divisible by SFVecSize");
  static_assert(size<1>(EpilogueTile{}) / SFVecSize == 1 or
                size<1>(EpilogueTile{}) / SFVecSize == 2 or
                size<1>(EpilogueTile{}) / SFVecSize == 4 or
                size<1>(EpilogueTile{}) / SFVecSize == 8,
                "Possible store in interleaved 4B aligned format");

  static constexpr int NumWarpgroups = 2;
  static constexpr int NumSyncWarps = NumWarpsPerWarpGroup * NumWarpgroups;
  static constexpr int NumQuadsPerWarp = 8;
  static constexpr int NumSyncQuads = NumSyncWarps * NumQuadsPerWarp;
  struct SharedStorage {
    array_aligned<ElementCompute, NumSyncQuads> smem_aux;
  };
  using NormalConstStrideMNL = Stride<_0,_0,int64_t>;
  struct Arguments {
    ElementBlockScaleFactor* ptr_scale_factor = {};
    // A matrix wide constant value to scale the output matrix
    // Avoids generating small FP4 values.
    ElementCompute const* norm_constant_ptr = {};
    NormalConstStrideMNL norm_constant_stride = {};
  };

  using Params = Arguments;

  using UnderlyingElementBlockScaleFactor = cute::remove_pointer_t<ElementBlockScaleFactor>;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;
    bool implementable = (N % SFVecSize == 0);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: [EVT Sm120BlockScaleFactorRowStore] N-dim should be divisible by SFVecSize.\n");
    }
    return implementable;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XcNvfp4RowStore() { }

  CUTLASS_HOST_DEVICE
  XcNvfp4RowStore(Params const& params, SharedStorage const& shared_storage)
      : params_ptr(&params)
      , smem_aux(const_cast<ElementCompute*>(shared_storage.smem_aux.data())) { }

  Params const* params_ptr = nullptr;
  ElementCompute *smem_aux = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <
    class RTensor,
    class GTensor,
    class STensor,
    class CoordGTensor,
    class ThrResidue,
    class TileCoordMN,
    class PaddedResidue,
    class TiledCopy_
  >
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
          RTensor&& tC_rSFD_,
          GTensor&& tC_gSFD_,
          STensor&& sAmaxs_,
          CoordGTensor tC_cSFD_,
          ThrResidue residue_tC_cSFD_,
          Params const* params_ptr_,
          TileCoordMN tile_coord_mn_,
          PaddedResidue padded_residue_,
          int thread_idx_,
          TiledCopy_ const&)
      : tC_rSFD(cute::forward<RTensor>(tC_rSFD_))
      , tC_gSFD(cute::forward<GTensor>(tC_gSFD_))
      , sAmaxs(cute::forward<STensor>(sAmaxs_))
      , tC_cSFD(tC_cSFD_)
      , residue_tC_cSFD(residue_tC_cSFD_)
      , params_ptr(params_ptr_)
      , padded_residue(padded_residue_)
      , tile_coord_mn(tile_coord_mn_)
      , thread_idx(thread_idx_) {}

    RTensor tC_rSFD;
    GTensor tC_gSFD;
    STensor sAmaxs;
    CoordGTensor tC_cSFD;
    ThrResidue residue_tC_cSFD;
    Params const* params_ptr;
    PaddedResidue padded_residue;  // residue of the scale storage: rows padded to 128, columns to 64
    TileCoordMN tile_coord_mn;
    int thread_idx;
    static constexpr int NumCollaboratingThreads = decltype(size(TiledCopy_{}))::value;
    static_assert(NumCollaboratingThreads % NumThreadsPerWarpGroup == 0);
    static constexpr int NumCollaboratingWarpGroups = NumCollaboratingThreads / NumThreadsPerWarpGroup;
    static_assert(NumCollaboratingWarpGroups == 1 || NumCollaboratingWarpGroups == 2,
                  "SM120 epilogue currently only supports one or two warp groups collaborating.");

    template <class ElementAccumulator, class ElementInput>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v,
          int epi_m,
          int epi_n,
          Array<ElementInput, FragmentSize> const& frg_input) {
      return frg_input;
    }

    template <class SmemTensor, class SyncFn, class VTensor>
    CUTLASS_DEVICE void
    reduce(SmemTensor&& smem_buffer, SyncFn const& sync_fn, int epi_m, int epi_n, bool is_last_iteration, VTensor visit_results) {
      /*
      Accumulator fragments are distributed across quads in different warps.
      For SFVector = 16, we have:

         8 elements          8 elements       8 elements          8 elements
      <----------------><-----------------><-----------------><----------------->
        Warp 0 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7      Warp 4 Quad 7
        Warp 0 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7      Warp 4 Quad 7

        <same pattern for warps 1 and 5 for the next set of 16 rows>
        <same pattern for warps 2 and 6 for the next set of 16 rows>
        <same pattern for warps 3 and 7 for the next set of 16 rows>

      In this case, row-wise scale factors are cooperatively reduced across 4
      threads from 1 quad in 1 warp. Each quad computes its own, local absolute
      maximum without communicating with other warps through shared memory.

      For SFVector = 32, we have:
         8 elements        8 elements         8 elements         8 elements
      <----------------><-----------------><-----------------><----------------->
        Warp 0 Quad 0      Warp 4 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 4 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 4 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7
        Warp 0 Quad 0      Warp 4 Quad 0      Warp 0 Quad 0      Warp 4 Quad 0
        Warp 0 Quad 1      Warp 4 Quad 1      Warp 0 Quad 1      Warp 4 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 4 Quad 7      Warp 0 Quad 7      Warp 4 Quad 7

        <same pattern for warps 1 and 5 for the next set of 16 rows>
        <same pattern for warps 2 and 6 for the next set of 16 rows>
        <same pattern for warps 3 and 7 for the next set of 16 rows>

      For SFVector = 64, we have:
          8 elements        8 elements         8 elements         8 elements
      <----------------><-----------------><-----------------><----------------->
        Warp 0 Quad 0      Warp 2 Quad 0      Warp 4 Quad 0      Warp 6 Quad 0
        Warp 0 Quad 1      Warp 2 Quad 1      Warp 4 Quad 1      Warp 6 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 2 Quad 7      Warp 4 Quad 7      Warp 6 Quad 7
        Warp 0 Quad 0      Warp 2 Quad 0      Warp 4 Quad 0      Warp 6 Quad 0
        Warp 0 Quad 1      Warp 2 Quad 1      Warp 4 Quad 1      Warp 6 Quad 1
        ...                ...                ...                ...
        Warp 0 Quad 7      Warp 2 Quad 7      Warp 4 Quad 7      Warp 6 Quad 7

        <same pattern for warps 1, 3, 5 and 7 for the next set of 16 rows>

      Thus, rowwise scale factors are cooperatively reduced across 8 threads
      from two quads in two warps. Each quad first computes its own, local
      absolute maximum and then shares this with the corresponding quad in the
      other warp. In this case, a reduction through shared memory is needed.

      For a non-cooperative epilogue (in which each warpgroup computes a
      separate tile), the pattern is the same as that above, except that warps 0
      and 2 are in the same row, and 1 and 3 are in the same row, and warps 4-7
      are not included.
      */

      // Accumulator fragments consist of two elements from two different rows of a 16x8 MMA output
      static constexpr int ColsPerThreadAccFrag = 2;
      static constexpr int RowsPerThreadAccFrag = 2;
      static_assert(FragmentSize ==
                    (ColsPerThreadAccFrag * RowsPerThreadAccFrag));

      static constexpr int NumThreadsPerQuad = 4;
      static_assert(SFVecSize == 16 || SFVecSize == 32 || SFVecSize == 64, "SF vector size must be either 16, 32 or 64.");
      // A quad from two or four warps participate in computing each scale factor.
      constexpr int WarpsPerSF = SFVecSize / 16;
      static_assert(WarpsPerSF == 1 || WarpsPerSF == 2 || WarpsPerSF == 4, "Only one, two or four warps are allowed in reduction.");

      constexpr bool IsInterWarpReductionNeeded = (WarpsPerSF != 1);

      // Number of fragments for each thread that are needed for computing a scale factor
      static constexpr int AccFragsPerSF = SFVecSize / (ColsPerThreadAccFrag * NumThreadsPerQuad * WarpsPerSF);
      static_assert(size<2>(visit_results) % AccFragsPerSF == 0,
        "Fragments along N mode must be a multiple of the number of accumulator fragments needed per SF");

      auto warp_idx = thread_idx / NumThreadsPerWarp;
      auto warpgroup_idx = thread_idx / NumThreadsPerWarpGroup;
      auto quad_idx_in_warp = (thread_idx % NumThreadsPerWarp) / NumThreadsPerQuad;
      auto thread_idx_in_quad = thread_idx % NumThreadsPerQuad;

      cutlass::maximum_absolute_value_reduction<ElementCompute, true> amax_op;
      cutlass::multiplies<ElementCompute> mul;

      Tensor tC_rSFD_flt = filter_zeros(tC_rSFD);

      auto synchronize = [&] () {
        cutlass::arch::NamedBarrier::sync(NumCollaboratingThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      };

      CUTLASS_PRAGMA_UNROLL
      for (int sf_id = 0; sf_id < size(tC_rSFD_flt); ++sf_id) {

        auto coord = idx2crd(sf_id, tC_rSFD_flt.shape());
        auto row_in_acc = get<0,1,1>(coord);
        auto row = crd2idx(get<1>(coord), get<1>(tC_rSFD_flt.shape()));
        auto sf = crd2idx(get<2>(coord), get<2>(tC_rSFD_flt.shape()));

        //
        // Compute amax for this scale factor
        //
        ElementCompute amax{0};

        // Compute amax among vals owned by this thread for this vector
        auto acc_frag_row = row_in_acc * RowsPerThreadAccFrag;
        auto acc_frag_start_for_sf = sf * AccFragsPerSF;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccFragsPerSF; ++i) {
          auto acc_frg = visit_results(0, row, acc_frag_start_for_sf + i);
          amax = amax_op(amax, acc_frg[acc_frag_row]);
          amax = amax_op(amax, acc_frg[acc_frag_row + 1]);
        }

        // At this point, each thread has computed the amax of the values that it owns for this SF vector.
        // We now need to compute the amax across threads. Because the TiledMMA uses an MmaThrLayout of <4,1,1>,
        // we know that all fragments in this row will belong to threads in this warp. Furthermore, because
        // SM120 narrow-precision MMAs have 16x8 output size with a quad owning two rows, we know that a quad
        // will own all of the elements to be reduced via amax. Therefore, we can use warp shuffle intrinsics
        // among threads in one quad to compute the amax.
        CUTLASS_PRAGMA_UNROLL
        for (int i = 1; i < 3; ++i) {
          auto amax_other = __shfl_xor_sync(0xffffffff, amax, i);
          amax = amax_op(amax, amax_other);
        }

        if constexpr (IsInterWarpReductionNeeded) {
          // At this point, all threads in the quad have the amax for the elements of the accumulator owned by its quad
          // that should be used in computing the amax for this SF. Threads 0 in each quad of warps 0 and 2
          // (similarly, 1 and 3) now exchange amaxes to compute the final amax.
          if (thread_idx_in_quad == 0) {
            sAmaxs(quad_idx_in_warp, warp_idx) = amax;
          }
          synchronize();

          // Get the amax broadcasted by the warp with which we share.
          // Work on 4 warps per SFD generation
          if constexpr (WarpsPerSF == 4) {
            if constexpr (NumCollaboratingWarpGroups == 2) {
              // This implementation assumes warp layout 2 x 4.
              // For cooperative kernels (NumCollaboratingWarpGroups=2),
              // warp 0 shares with 2 / 4 / 6, warp 1 shares with 3 / 5/ 7.
              auto amax_other2 = sAmaxs(quad_idx_in_warp, warp_idx ^ 2);
              auto amax_other4 = sAmaxs(quad_idx_in_warp, warp_idx ^ 4);
              auto amax_other6 = sAmaxs(quad_idx_in_warp, warp_idx ^ 6);
              synchronize();
              amax = amax_op(amax, amax_other2);
              amax = amax_op(amax, amax_other4);
              amax = amax_op(amax, amax_other6);
            }
            else {
              static_assert(cutlass::detail::dependent_false<TiledCopy_>, "Unsupported warp layout.");
            }
          }
          // Work on 2 warps per SFD generation
          else if constexpr(WarpsPerSF == 2) {
            // For cooperative kernels (NumCollaboratingWarpGroups=2), 0 shares
            // with 4, 1 shares with 5, etc. For non-cooperative kernels
            // (NumCollaboratingWarpGroups=1), 0 shares with 2, 1 shares with 3.
            auto amax_other = sAmaxs(
                quad_idx_in_warp, warp_idx ^ (1 << NumCollaboratingWarpGroups));
            synchronize();
            amax = amax_op(amax, amax_other);
          }
        }

        // Wall-X / quantize_nvfp4 arithmetic: e4m3(min(448, max(2^-6, amax * rcp.approx(6)))); a block outside the
        // [M, N] output (a padding row or column of the scale storage) gets scale 0.
        const float scale = xinfer::nvfp4::AmaxScale(amax, 1.0f);
        const int v_idx = crd2idx(make_coord(0, row, acc_frag_start_for_sf), shape(visit_results));
        const bool live = elem_less(tC_cSFD(_, _, _, epi_m, epi_n)(v_idx * FragmentSize + acc_frag_row), residue_tC_cSFD);
        tC_rSFD_flt(coord) = UnderlyingElementBlockScaleFactor::bitcast(live ? xinfer::nvfp4::ScaleBits(scale) : uint8_t(0));

        //
        // Apply the scale factor to the output
        //
        // element: x * rcp.approx.ftz(scale), then the epilogue's cvt.rn.satfinite to E2M1
        float acc_scale;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(acc_scale) : "f"(scale));

        // Compute quantized output values
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < AccFragsPerSF; ++i) {
          auto acc_frag = visit_results(0, row, acc_frag_start_for_sf + i);
          visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row    ] = mul(acc_frag[acc_frag_row], acc_scale);
          visit_results(0, row, acc_frag_start_for_sf + i)[acc_frag_row + 1] = mul(acc_frag[acc_frag_row + 1], acc_scale);
        }
      } // sf

      // Since scale factors are computed cooperatively across two quads from two warps, we only need one thread from the
      // set of 8 cooperating threads to write out the data. We do this with thread 0 in each quad of the first warp that collaborates.
      bool write_sf = (thread_idx_in_quad == 0);
      if constexpr (NumCollaboratingWarpGroups == 2) {
        // For cooperative kernels (NumCollaboratingWarpGroups=2), 0 shares with 4, 1 shares with 5, etc.
        // Thus, only the warps in the first warpgroup need to write out scale factors.
        if constexpr (IsInterWarpReductionNeeded) {
          write_sf &= warp_idx < NumWarpsPerWarpGroup;
        }
      }
      else {
        if constexpr (IsInterWarpReductionNeeded) {
          // When non-cooperative kernels apply inter warp reduce, they are with
          // SF output rule as below :
          // 1. warp 0 shares with 2 and 1 shares with 3 within each warpgroup.
          // 2. warps 0 and 1 of the first warpgroup and 4 and 5 of the second
          //   warpgroup need to write output sf.
          write_sf &= ((warp_idx < 2) || (warpgroup_idx == 1 && warp_idx < 6));
        }
      }

      if (write_sf && elem_less(tC_cSFD(_0{}, _0{}, _0{}, epi_m, epi_n), padded_residue)) {
        copy_aligned(tC_rSFD, tC_gSFD(_, _, _, _0{}, _0{}, get<0>(tile_coord_mn) + epi_m, get<1>(tile_coord_mn) + epi_n));
      }
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using Sm1xxBlockScaledOutputConfig = cutlass::detail::Sm1xxBlockScaledOutputConfig<SFVecSize>;
    // Batch/group index for the norm constant lookup below. In the
    // ptr-array/grouped case l is zeroed after selecting the per-group
    // scale factor pointer (each group's SFD tensor has no batch mode),
    // but the norm constant tensor is still indexed per group.
    UnderlyingElementBlockScaleFactor* ptr_scale_factor = nullptr;
    // If Ptr-Array/Grouped GEMM with BlockScaleFactor per batch/group
    if constexpr (!cute::is_same_v<UnderlyingElementBlockScaleFactor, ElementBlockScaleFactor>) {
      ptr_scale_factor = params_ptr->ptr_scale_factor[l];
      l = 0;
    }
    else {
      ptr_scale_factor = params_ptr->ptr_scale_factor;
    }

    auto epi_tile_mn = shape<1>(zipped_divide(make_layout(take<0,2>(args.tile_shape_mnk)), args.epi_tile));
    Tensor mSFD = make_tensor(make_gmem_ptr(ptr_scale_factor), Sm1xxBlockScaledOutputConfig::tile_atom_to_shape_SFD(args.problem_shape_mnkl));

    static_assert(size<1>(EpilogueTile{}) && ((size<1>(EpilogueTile{}) & (size<1>(EpilogueTile{}) - 1)) == 0), "Epilogue Tile N should be pow of 2");
    Tensor gSFD = local_tile(mSFD, args.epi_tile, make_coord(_, _,l));                             // (EPI_M,EPI_N, #EPI_Ms, #EPI_Ns)
    Tensor tCgSFD = sm90_partition_for_epilogue<ReferenceSrc>(                                     // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,#EPI_Ms, #EPI_Ns)
                        gSFD, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrSFD = make_tensor_like<UnderlyingElementBlockScaleFactor>(take<0,3>(cute::layout(tCgSFD)));    // (CPY,CPY_M,CPY_N)

    auto tile_coord_mn = make_coord(m * size<0>(epi_tile_mn), n * size<1>(epi_tile_mn));

    // Fetch and compute these during initialization
    auto padded_residue = make_coord(get<0>(args.residue_tCcD) + ((M + 127) / 128 * 128 - M),
                                     get<1>(args.residue_tCcD) + ((N + 63) / 64 * 64 - N));

    Tensor sAmaxs = make_tensor(
      make_smem_ptr(smem_aux),
      make_layout(make_shape(Int<NumQuadsPerWarp>{}, Int<NumSyncWarps>{}))
    );

    return ConsumerStoreCallbacks(
      cute::move(tCrSFD),
      cute::move(tCgSFD),
      cute::move(sAmaxs),
      args.tCcD,
      args.residue_tCcD,
      params_ptr,
      tile_coord_mn,
      padded_residue,
      args.thread_idx,
      args.tiled_copy);

  }
};

// act 1: gelu (tanh approximation), 2: gelu (erf); gemm_epilogue.cu's Act, on each element
template <int Act>
struct XcAct {
  template <class T>
  struct Fn;
  template <int N>
  struct Fn<Array<float, N>> {
    CUTLASS_DEVICE Array<float, N> operator()(Array<float, N> const& x) const {
      Array<float, N> y;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N; ++i) {
        const float g = x[i];
        if constexpr (Act == 2) {
          y[i] = 0.5f * g * (1.0f + erff(g * 0.7071067811865476f));
        } else {
          const float u = 0.7978845608028654f * (g + 0.044715f * g * g * g);
          y[i] = 0.5f * g * (1.0f + tanhf(u));
        }
      }
      return y;
    }
  };
};

template <class T>
struct XcIdentity;
template <int N>
struct XcIdentity<Array<float, N>> {
  CUTLASS_DEVICE Array<float, N> operator()(Array<float, N> const& x) const { return x; }
};

// The operation tag (block-scaled NVFP4 output with E4M3 1x16 scales); its callbacks are XcNvfp4ActQuantTree.
template <int Act>
struct XcNvfp4ActQuant : LinCombBlockScaleFactor<16, cutlass::float_e2m1_t, float, cutlass::float_ue4m3_t,
                                                 cutlass::layout::RowMajor, cutlass::float_e2m1_t, float> {};

template <int Act, class EpilogueTile, class CtaTileShapeMNK, int FragmentSize>
using XcNvfp4ActQuantTree = Sm90EVT<
    XcNvfp4RowStore<16, EpilogueTile, CtaTileShapeMNK, FragmentSize, cutlass::float_e2m1_t, float,
                    cutlass::float_ue4m3_t>,
    Sm90EVT<Sm90Compute<XcIdentity, float, float, FloatRoundStyle::round_to_nearest>,                          // the BF16 value, exactly
            Sm90EVT<Sm90Compute<XcAct<Act>::template Fn, cutlass::bfloat16_t, float, FloatRoundStyle::round_to_nearest>,  // bf16(act(.))
                    Sm90EVT<Sm90Compute<cutlass::plus, cutlass::bfloat16_t, float, FloatRoundStyle::round_to_nearest>,     // bf16(alpha * acc + bias)
                            Sm90EVT<Sm90Compute<cutlass::multiplies, float, float, FloatRoundStyle::round_to_nearest>, Sm90ScalarBroadcast<float>,
                                    Sm90AccFetch>,
                            Sm90RowBroadcast<0, CtaTileShapeMNK, cutlass::bfloat16_t, float,
                                             Stride<_0, _1, int64_t>, 8>>>>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore, int Act,
          class CtaTileShapeMNK, class EpilogueTile>
struct FusionCallbacks<epilogue::Sm120TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
                       XcNvfp4ActQuant<Act>, CtaTileShapeMNK, EpilogueTile>
    : XcNvfp4ActQuantTree<Act, EpilogueTile, CtaTileShapeMNK, FragmentSize> {
  using Impl = XcNvfp4ActQuantTree<Act, EpilogueTile, CtaTileShapeMNK, FragmentSize>;
  using Operation = XcNvfp4ActQuant<Act>;
  struct Arguments {
    float alpha = 1.0f;
    cutlass::bfloat16_t const* bias = nullptr;
    cutlass::float_ue4m3_t* scales = nullptr;
    operator typename Impl::Arguments() const {
      return {                                        // (identity node, scale store)
          {                                           // identity node: (activation node, op)
              {                                       // activation node: (bias node, op)
                  {                                   // bias node: (alpha node, bias, op)
                      {{{alpha}}, {}, {}},            // alpha node: (alpha, accumulator, op)
                      {bias, cutlass::bfloat16_t(0), {}},
                      {}},
                  {}},
              {}},
          {scales}};
    }
  };
  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace xinfer::nvfp4_cutlass_sm120 {

template <Tile T, int Act>
struct GemmActQuant {
  using TileShape = typename TileTraits<T>::Shape;
  using Cluster = Shape<_1, _1, _1>;
  using Row = cutlass::layout::RowMajor;
  using Col = cutlass::layout::ColumnMajor;
  using E2M1 = cutlass::float_e2m1_t;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, TileShape, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void, Row, 32, E2M1, Row, 32,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      cutlass::epilogue::fusion::XcNvfp4ActQuant<Act>>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, NV, Row, 32, NV, Col, 32, float, TileShape,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedNvf4Sm120>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, CollectiveEpilogue>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static cutlass::Status Run(const void* a, const void* a_scale, const void* w, const void* w_scale, void* out,
                             void* out_scale, const BF* bias, float alpha, int m, int n, int k, int sms,
                             cudaStream_t stream) {
    auto da = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1});
    auto db = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1});
    auto dd = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1});
    using Config = typename Mainloop::Sm1xxBlkScaledConfig;
    auto la = Config::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    auto lb = Config::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    cutlass::KernelHardwareInfo hw;
    hw.sm_count = sms;
    typename CollectiveEpilogue::FusionCallbacks::Arguments fusion{
        alpha, bias, reinterpret_cast<cutlass::float_ue4m3_t*>(out_scale)};
    typename Device::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {reinterpret_cast<NV::DataType const*>(a), da, reinterpret_cast<NV::DataType const*>(w), db,
         reinterpret_cast<NV::ScaleFactorType const*>(a_scale), la,
         reinterpret_cast<NV::ScaleFactorType const*>(w_scale), lb},
        {fusion, nullptr, dd, reinterpret_cast<E2M1*>(out), dd},
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

// The kernel of (tile, act): act 1 gelu (tanh approximation), 2 gelu (erf).
inline cutlass::Status RunActQuant(Tile tile, int act, const void* a, const void* a_scale, const void* w,
                                   const void* w_scale, void* out, void* out_scale, const BF* bias, float alpha, int m,
                                   int n, int k, int sms, cudaStream_t stream) {
#define XC_NVFP4_ACT_QUANT(T)                                                                                       \
  case T:                                                                                                            \
    if (act == 1)                                                                                                    \
      return GemmActQuant<T, 1>::Run(a, a_scale, w, w_scale, out, out_scale, bias, alpha, m, n, k, sms, stream);     \
    if (act == 2)                                                                                                    \
      return GemmActQuant<T, 2>::Run(a, a_scale, w, w_scale, out, out_scale, bias, alpha, m, n, k, sms, stream);     \
    break;
  switch (tile) {
    XC_NVFP4_ACT_QUANT(Tile::k128x128)
    XC_NVFP4_ACT_QUANT(Tile::k256x128)
    XC_NVFP4_ACT_QUANT(Tile::k128x64)
    XC_NVFP4_ACT_QUANT(Tile::k128x32)
  }
#undef XC_NVFP4_ACT_QUANT
  return cutlass::Status::kErrorInvalidProblem;
}

}  // namespace xinfer::nvfp4_cutlass_sm120
