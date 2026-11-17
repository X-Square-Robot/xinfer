// Fused GEMM + SiLU + mul, hand-written tcgen05 (Thor sm_110a), CuTe atoms.
//   out[M,I] = silu(x@Wg^T+bg) * (x@Wu^T+bu)
//
// Drop-in replacement for bf16_gemm_silu_mul (was in bf16_fused_mlp.cu):
// identical signature (x, w_gate, w_up, bias_gate?, bias_up?, out_features) and
// semantics (returns act[:, :out_features] of an [M, I_pad] buffer; pad columns
// are exactly zero — padded weight+bias rows give silu(0)*0=0). Best measured
// 186µs on the ViT shape (M=1728, I=3420, K=1280) — 1.65× the 2-CUTLASS-kernel
// baseline (307µs), 162 TFLOPS / 63% of peak. See gate_up_fused_opt/ITERATIONS.md.
//
// THE full 2-SM overlapping-accumulator collective (KernelWiki pr-cutlass-2995
// pattern). 2-SM MMA (256×128) splits 256 M-rows across 2 CTAs → each CTA holds
// 128 rows = HALF the tmem → 2 double-buffered accumulator sets (2×256=512 cols)
// fit, while the weights multicast to the 2 M-peers (weight reuse kept).
// Persistent + warp-specialized: warp0 = leader-CTA MMA (fills tmem set s),
// warp1 = TMA producer (both CTAs), warps4-7 = epilogue (both CTAs drain their
// 128-row half of the OTHER set). The MMA fills set s+1 while the epilogue drains
// set s → the epilogue hides behind tensor work. Accumulator release is cross-CTA:
// both CTAs' epilogues arrive the leader's acc_empty (ClusterBarrier).
// COLUMN-MAJOR raster (mt=t%n_m_tiles): the ~10 co-resident clusters share the
// same nt = the bigger weight (B) tile → weights stay hot in L2 (−13% vs row-major).

#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include "common/tvm_ffi_cuda_tensor.h"

#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace gu_fused {

using namespace cute;

using TA = cutlass::bfloat16_t; using TB = cutlass::bfloat16_t;
using TAcc = float;             using TD = cutlass::bfloat16_t;

static constexpr int kMmaM  = 256;   // 2-SM MMA M (128 rows per CTA)
static constexpr int kMmaN  = 128;
static constexpr int kTileK = 64;
static constexpr int kEpiN  = 32;
static constexpr int kStages = 4;    // mainloop smem pipeline depth
static constexpr int kAccSets = 2;   // double-buffered tmem accumulators
static constexpr int kEpiBuf = 2;
static constexpr int kTmemCols = kAccSets * 2 * kMmaN;  // 512

static constexpr int kMmaWarp = 0;
static constexpr int kProdWarp = 1;
static constexpr int kEpiWarp0 = 4;  // warps 4-7 = epilogue (128 threads)
static constexpr int kNumWarps = 8;

CUTE_DEVICE void mbar_arrive(cute::uint64_t& bar) {   // local plain arrive
  uint32_t a = cute::cast_smem_ptr_to_uint(&bar);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(a));
}

template <class ASmemLayout, class BSmemLayout, class DSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TA, kStages * cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TB, kStages * cute::cosize_v<BSmemLayout>> Bg;
  alignas(128) cute::ArrayEngine<TB, kStages * cute::cosize_v<BSmemLayout>> Bu;
  alignas(128) cute::ArrayEngine<TD, kEpiBuf * cute::cosize_v<DSmemLayout>> D;
  alignas(16) float biasG[kAccSets][kMmaN];
  alignas(16) float biasU[kAccSets][kMmaN];
  alignas(16) cute::uint64_t full[kStages];       // producer -> leader MMA
  alignas(16) cute::uint64_t empty[kStages];      // leader MMA -> producer (2x1SM)
  alignas(16) cute::uint64_t acc_full[kAccSets];  // leader MMA -> both epilogues (2x1SM)
  alignas(16) cute::uint64_t acc_empty[kAccSets]; // both epilogues -> leader MMA (cluster)
  alignas(16) cute::uint32_t tmem_base_ptr;
};

template <class SmemT, class ASmemLayout, class BSmemLayout, class DSmemLayout,
          class ATensor, class BTensor, class DTensor,
          class MmaTiler, class TiledMMA, class ClusterShape, class EpiTiler,
          class TmaAtomA, class TmaAtomB, class TmaAtomD>
__global__ static void
gate_up_device(ATensor mA, BTensor mWg, BTensor mWu, DTensor mD,
               const TB* __restrict__ bias_g, const TB* __restrict__ bias_u,
               int M, int Ip, int n_m_tiles, int n_n_tiles,
               MmaTiler mma_tiler, TiledMMA tiled_mma, ClusterShape cluster_shape,
               EpiTiler epi_tiler_mn,
               ASmemLayout sA_layout, BSmemLayout sB_layout, DSmemLayout sD_layout,
               CUTE_GRID_CONSTANT TmaAtomA const tma_A,
               CUTE_GRID_CONSTANT TmaAtomB const tma_Bg,
               CUTE_GRID_CONSTANT TmaAtomB const tma_Bu,
               CUTE_GRID_CONSTANT TmaAtomD const tma_D)
{
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  extern __shared__ char smem_raw[];
  SmemT& smem = *reinterpret_cast<SmemT*>(smem_raw);

  const int warp_idx = threadIdx.x / 32;
  uint32_t elect = cute::elect_one_sync();
  const int total_tiles  = n_m_tiles * n_n_tiles;
  const int num_clusters = gridDim.x / size<0>(cluster_layout_vmnk);
  const int cluster_id   = blockIdx.x / size<0>(cluster_layout_vmnk);
  auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  auto elect_one_cta  = get<0>(cta_coord_vmnk) == Int<0>{};
  const int mma_v = get<0>(cta_coord_vmnk);
  ThrMMA cta_mma = tiled_mma.get_slice(mma_v);

  uint16_t mcast_a = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord_vmnk);
  uint16_t mcast_b = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_coord_vmnk);
  uint16_t mma_mcast_c = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_coord_vmnk) |
                         create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_coord_vmnk);

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (warp_idx == kMmaWarp) tmem_allocator.allocate(kTmemCols, &smem.tmem_base_ptr);
  if (warp_idx == kMmaWarp && elect) {
    int nparts = size<1>(cluster_layout_vmnk) + size<2>(cluster_layout_vmnk) - 1;
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(smem.full[s], 1);
      cute::initialize_barrier(smem.empty[s], nparts);
    }
    for (int a = 0; a < kAccSets; ++a) {
      cute::initialize_barrier(smem.acc_full[a], nparts);
      cute::initialize_barrier(smem.acc_empty[a], int(size<0>(cluster_layout_vmnk)));  // both CTAs
    }
  }
  cute::cluster_sync();
  uint32_t tmem_base = smem.tmem_base_ptr;

  // ---- coordinate / tensor helpers (per tile) ----------------------------
  auto gA_of = [&](int mt){ return local_tile(mA, mma_tiler, make_coord(mt, 0, _), Step<_1,X,_1>{}); };
  auto gWg_of= [&](int nt){ return local_tile(mWg, mma_tiler, make_coord(0, nt, _), Step<X,_1,_1>{}); };
  auto gWu_of= [&](int nt){ return local_tile(mWu, mma_tiler, make_coord(0, nt, _), Step<X,_1,_1>{}); };
  auto gD_of = [&](int mt, int nt){ return local_tile(mD, mma_tiler, make_coord(mt, nt, _), Step<_1,_1,X>{}); };
  auto cD_of = [&](int mt, int nt){ return local_tile(make_identity_tensor(make_shape(M, Ip)),
                                    mma_tiler, make_coord(mt, nt, _), Step<_1,_1,X>{}); };
  auto tCgA_of  = [&](int mt){ return cta_mma.partition_A(gA_of(mt)); };
  auto tCgWg_of = [&](int nt){ return cta_mma.partition_B(gWg_of(nt)); };
  auto tCgWu_of = [&](int nt){ return cta_mma.partition_B(gWu_of(nt)); };
  auto accg_of = [&](int set){ auto t = cta_mma.make_fragment_C(cta_mma.partition_C(gD_of(0,0)));
                               t.data() = tmem_base + uint32_t(set*2*kMmaN); return t; };
  auto accu_of = [&](int set){ auto t = cta_mma.make_fragment_C(cta_mma.partition_C(gD_of(0,0)));
                               t.data() = tmem_base + uint32_t(set*2*kMmaN + kMmaN); return t; };

  constexpr int coA = cute::cosize_v<ASmemLayout>;
  constexpr int coB = cute::cosize_v<BSmemLayout>;
  auto sA  = [&](int s){ return make_tensor(make_smem_ptr(smem.A.begin()  + s*coA), sA_layout); };
  auto sBg = [&](int s){ return make_tensor(make_smem_ptr(smem.Bg.begin() + s*coB), sB_layout); };
  auto sBu = [&](int s){ return make_tensor(make_smem_ptr(smem.Bu.begin() + s*coB), sB_layout); };
  auto tAsA = [&](int mt,int s){ return get<1>(tma_partition(tma_A, get<2>(cta_coord_vmnk),
                make_layout(size<2>(cluster_layout_vmnk)), group_modes<0,3>(sA(s)), group_modes<0,3>(tCgA_of(mt)))); };
  auto tBsg = [&](int s){ return get<1>(tma_partition(tma_Bg, get<1>(cta_coord_vmnk),
                make_layout(size<1>(cluster_layout_vmnk)), group_modes<0,3>(sBg(s)), group_modes<0,3>(tCgWg_of(0)))); };
  auto tBsu = [&](int s){ return get<1>(tma_partition(tma_Bu, get<1>(cta_coord_vmnk),
                make_layout(size<1>(cluster_layout_vmnk)), group_modes<0,3>(sBu(s)), group_modes<0,3>(tCgWu_of(0)))); };
  auto tAgA_of = [&](int mt){ return get<0>(tma_partition(tma_A, get<2>(cta_coord_vmnk),
                make_layout(size<2>(cluster_layout_vmnk)), group_modes<0,3>(sA(0)), group_modes<0,3>(tCgA_of(mt)))); };
  auto tBgg_of = [&](int nt){ return get<0>(tma_partition(tma_Bg, get<1>(cta_coord_vmnk),
                make_layout(size<1>(cluster_layout_vmnk)), group_modes<0,3>(sBg(0)), group_modes<0,3>(tCgWg_of(nt)))); };
  auto tBgu_of = [&](int nt){ return get<0>(tma_partition(tma_Bu, get<1>(cta_coord_vmnk),
                make_layout(size<1>(cluster_layout_vmnk)), group_modes<0,3>(sBu(0)), group_modes<0,3>(tCgWu_of(nt)))); };

  const int k_tiles = size<3>(tCgA_of(0));
  const int nkb     = size<2>(cta_mma.make_fragment_A(sA(0)));
  int tma_bytes = size<0>(cluster_layout_vmnk) *
                  (int(sizeof(make_tensor_like(tAsA(0,0)))) + int(sizeof(make_tensor_like(tBsg(0))))
                   + int(sizeof(make_tensor_like(tBsu(0)))));

  // ================= PRODUCER (warp 1, both CTAs) =========================
  if (warp_idx == kProdWarp) {
    int gstep = 0, p_empty[kStages]; for (int s=0;s<kStages;++s) p_empty[s]=0;
    for (int t = cluster_id; t < total_tiles; t += num_clusters) {
      int mt = t % n_m_tiles, nt = t / n_m_tiles;  // column-major raster (share B in L2)
      auto tAgA = tAgA_of(mt); auto tBgg = tBgg_of(nt); auto tBgu = tBgu_of(nt);
      for (int kt = 0; kt < k_tiles; ++kt) {
        int s = gstep % kStages;
        if (gstep >= kStages) { cute::wait_barrier(smem.empty[s], p_empty[s]); p_empty[s]^=1; }
        if (elect) {
          if (elect_one_cta) cute::set_barrier_transaction_bytes(smem.full[s], tma_bytes);
          copy(tma_A.with(smem.full[s], mcast_a),  tAgA(_, kt), tAsA(mt, s));
          copy(tma_Bg.with(smem.full[s], mcast_b), tBgg(_, kt), tBsg(s));
          copy(tma_Bu.with(smem.full[s], mcast_b), tBgu(_, kt), tBsu(s));
        }
        ++gstep;
      }
    }
  }
  // ================= LEADER MMA (warp 0, leader CTA) ======================
  else if (warp_idx == kMmaWarp) {
    int gstep = 0, p_full[kStages]; for (int s=0;s<kStages;++s) p_full[s]=0;
    int p_accempty[kAccSets]; for (int a=0;a<kAccSets;++a) p_accempty[a]=0;
    int it = 0;
    for (int t = cluster_id; t < total_tiles; t += num_clusters, ++it) {
      int set = it % kAccSets;
      auto acc_g = accg_of(set); auto acc_u = accu_of(set);
      if (elect_one_cta) {
        if (it >= kAccSets) { cute::wait_barrier(smem.acc_empty[set], p_accempty[set]); p_accempty[set]^=1; }
        for (int kt = 0; kt < k_tiles; ++kt) {
          int s = gstep % kStages;
          cute::wait_barrier(smem.full[s], p_full[s]); p_full[s]^=1;
          Tensor tCrA  = cta_mma.make_fragment_A(sA(s));
          Tensor tCrBg = cta_mma.make_fragment_B(sBg(s));
          Tensor tCrBu = cta_mma.make_fragment_B(sBu(s));
          tiled_mma.accumulate_ = (kt==0)?UMMA::ScaleOut::Zero:UMMA::ScaleOut::One;
          CUTE_UNROLL
          for (int kb=0;kb<nkb;++kb){ gemm(tiled_mma,tCrA(_,_,kb),tCrBg(_,_,kb),acc_g); tiled_mma.accumulate_=UMMA::ScaleOut::One; }
          tiled_mma.accumulate_ = (kt==0)?UMMA::ScaleOut::Zero:UMMA::ScaleOut::One;
          CUTE_UNROLL
          for (int kb=0;kb<nkb;++kb){ gemm(tiled_mma,tCrA(_,_,kb),tCrBu(_,_,kb),acc_u); tiled_mma.accumulate_=UMMA::ScaleOut::One; }
          cutlass::arch::umma_arrive_multicast_2x1SM(&smem.empty[s], mma_mcast_c);
          ++gstep;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&smem.acc_full[set], mma_mcast_c);
      } else {
        gstep += k_tiles;  // peer CTA keeps gstep in lockstep (issues no mma)
      }
    }
  }
  // ================= EPILOGUE (warps 4-7, both CTAs) ======================
  else if (warp_idx >= kEpiWarp0) {
    auto epi_tiler_v = make_tile(epi_tiler_mn);
    constexpr int coD = cute::cosize_v<DSmemLayout>;
    auto sD_of = [&](int b){ return make_tensor(make_smem_ptr(smem.D.begin() + b*coD), sD_layout); };
    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, zipped_divide(accg_of(0), epi_tiler_v)(_, Int<0>{}));
    const int etid = threadIdx.x - kEpiWarp0*32;
    const int enthr = (kNumWarps - kEpiWarp0)*32;
    ThrCopy thr = t2r.get_slice(etid);
    Tensor rg = make_tensor<TAcc>(shape(thr.partition_D(sD_of(0))));
    Tensor ru = make_tensor<TAcc>(shape(thr.partition_D(sD_of(0))));
    Tensor rd = make_tensor<TD>(shape(thr.partition_D(sD_of(0))));

    int p_accfull[kAccSets]; for (int a=0;a<kAccSets;++a) p_accfull[a]=0;
    int store_iter = 0, it = 0;
    for (int t = cluster_id; t < total_tiles; t += num_clusters, ++it) {
      int mt = t % n_m_tiles, nt = t / n_m_tiles, set = it % kAccSets;  // column-major
      cute::wait_barrier(smem.acc_full[set], p_accfull[set]); p_accfull[set]^=1;

      int n0 = nt * kMmaN;
      for (int j = etid; j < kMmaN; j += enthr) {
        int n = n0 + j; bool ok = n < Ip;
        smem.biasG[set][j] = (bias_g && ok) ? float(bias_g[n]) : 0.f;
        smem.biasU[set][j] = (bias_u && ok) ? float(bias_u[n]) : 0.f;
      }
      asm volatile("bar.sync 1, %0;" :: "r"(enthr));

      Tensor gAcc_g = zipped_divide(accg_of(set), epi_tiler_v);
      Tensor gAcc_u = zipped_divide(accu_of(set), epi_tiler_v);
      Tensor gD_epi = zipped_divide(cta_mma.partition_C(gD_of(mt,nt)), epi_tiler_v);
      Tensor cD_epi = zipped_divide(cta_mma.partition_C(cD_of(mt,nt)), epi_tiler_v);
      Tensor tTR_g = thr.partition_S(gAcc_g);
      Tensor tTR_u = thr.partition_S(gAcc_u);
      Tensor tTR_cD = thr.partition_D(cD_epi);
      int nEpi = size<2>(tTR_g);
      for (int e = 0; e < nEpi; ++e) {
        int buf = store_iter % kEpiBuf;
        Tensor sD_b = sD_of(buf);
        auto [tSG_gD, tSG_sD] = tma_partition(tma_D, sD_b, gD_epi);
        Tensor tTR_sD = thr.partition_D(sD_b);
        copy(t2r, tTR_g(_,_,e), rg);
        copy(t2r, tTR_u(_,_,e), ru);
        Tensor cD_e = tTR_cD(_,_,e);
        CUTE_UNROLL
        for (int v = 0; v < size(rg); ++v) {
          int n = get<1>(cD_e(v));
          float g = rg(v) + smem.biasG[set][n - n0];
          float u = ru(v) + smem.biasU[set][n - n0];
          rd(v) = TD((g / (1.f + __expf(-g))) * u);
        }
        if (etid == 0 && store_iter >= kEpiBuf) tma_store_wait<kEpiBuf-1>();
        asm volatile("bar.sync 1, %0;" :: "r"(enthr));
        copy(rd, tTR_sD);
        cutlass::arch::fence_view_async_shared();
        asm volatile("bar.sync 1, %0;" :: "r"(enthr));
        if (etid == 0) { copy(tma_D, tSG_sD, tSG_gD(_, e)); tma_store_arrive(); }
        ++store_iter;
      }
      // set drained -> signal leader's acc_empty (both CTAs)
      asm volatile("bar.sync 1, %0;" :: "r"(enthr));
      if (etid == 0) {
        if (elect_one_cta) mbar_arrive(smem.acc_empty[set]);
        else cutlass::arch::ClusterBarrier::arrive(&smem.acc_empty[set], 0u, 1u);  // arrive leader
      }
    }
    if (etid == 0) tma_store_wait<0>();
  }

  cute::cluster_sync();
  if (warp_idx == kMmaWarp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, kTmemCols);
  }
}

// ---------------------------------------------------------------------------
void run(tvm::ffi::TensorView out, tvm::ffi::TensorView x,
         tvm::ffi::TensorView w_g, tvm::ffi::TensorView w_u,
         const TB* bias_g, const TB* bias_u) {
  int M = x.size(0), K = x.size(1), Ip = w_g.size(0);
  cudaStream_t stream = xinfer::ffi::Stream(x);

  auto mA  = make_tensor(make_gmem_ptr(reinterpret_cast<const TA*>(x.data_ptr())),  make_layout(make_shape(M, K),  make_stride(K, Int<1>{})));
  auto mWg = make_tensor(make_gmem_ptr(reinterpret_cast<const TB*>(w_g.data_ptr())), make_layout(make_shape(Ip, K), make_stride(K, Int<1>{})));
  auto mWu = make_tensor(make_gmem_ptr(reinterpret_cast<const TB*>(w_u.data_ptr())), make_layout(make_shape(Ip, K), make_stride(K, Int<1>{})));
  auto mD0 = make_tensor(make_gmem_ptr(reinterpret_cast<TD*>(out.data_ptr())),       make_layout(make_shape(M, Ip), make_stride(Ip, Int<1>{})));

  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TA, TB, TAcc, kMmaM, kMmaN, UMMA::Major::K, UMMA::Major::K>{});
  auto cluster_shape = make_shape(Int<2>{}, Int<1>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));
  auto mma_tiler = make_shape(Int<kMmaM>{}, Int<kMmaN>{}, Int<kTileK>{});
  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(Int<kMmaM>{}, Int<kTileK>{}));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(Int<kMmaN>{}, Int<kTileK>{}));
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TB>{}, mma_shape_B);
  auto epi_tiler = make_tile(Int<128>{}, Int<kEpiN>{});
  auto sD_layout_mn = tile_to_shape(UMMA::Layout_K_SW64_Atom<TD>{}, make_shape(Int<128>{}, Int<kEpiN>{}));
  auto sD_layout = group<0,2>(sD_layout_mn);

  using SmemT = SharedStorage<decltype(sA_layout), decltype(sB_layout), decltype(sD_layout)>;

  Copy_Atom tma_A  = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mA,  sA_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_Bg = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mWg, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_Bu = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD_MULTICAST{}, mWu, sB_layout, mma_tiler, tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_D  = make_tma_atom(SM90_TMA_STORE{}, mD0, sD_layout, epi_tiler);
  auto mA_tma  = tma_A.get_tma_tensor(shape(mA));
  auto mWg_tma = tma_Bg.get_tma_tensor(shape(mWg));
  auto mWu_tma = tma_Bu.get_tma_tensor(shape(mWu));
  auto mD_tma  = tma_D.get_tma_tensor(shape(mD0));

  int n_m_tiles = (M + kMmaM - 1) / kMmaM;
  int n_n_tiles = (Ip + kMmaN - 1) / kMmaN;
  int total = n_m_tiles * n_n_tiles;
  int clusters = total < 10 ? total : 10;   // persistent: 10 clusters = 20 CTAs = 1/SM

  int smem_bytes = sizeof(SmemT);
  dim3 block(256);
  dim3 cluster(2, 1, 1);
  dim3 grid(clusters * 2, 1, 1);

  auto* kernel = &gate_up_device<SmemT, decltype(sA_layout), decltype(sB_layout), decltype(sD_layout),
                                 decltype(mA_tma), decltype(mWg_tma), decltype(mD_tma),
                                 decltype(mma_tiler), decltype(tiled_mma), decltype(cluster_shape),
                                 decltype(epi_tiler), decltype(tma_A), decltype(tma_Bg), decltype(tma_D)>;
  xinfer::ffi::CheckCudaError(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_bytes),
      "bf16_gemm_silu_mul_sm110 cudaFuncSetAttribute");
  cutlass::ClusterLaunchParams params{grid, block, cluster, smem_bytes, stream};
  cutlass::launch_kernel_on_cluster(params, (void const*)kernel,
                                    mA_tma, mWg_tma, mWu_tma, mD_tma, bias_g, bias_u, M, Ip,
                                    n_m_tiles, n_n_tiles, mma_tiler, tiled_mma, cluster_shape, epi_tiler,
                                    sA_layout, sB_layout, sD_layout, tma_A, tma_Bg, tma_Bu, tma_D);
}

}  // namespace gu_fused

static const gu_fused::TB* bias_ptr(
    const tvm::ffi::Optional<tvm::ffi::TensorView>& bias) {
  if (!bias.has_value()) return nullptr;
  return xinfer::ffi::ConstData<gu_fused::TB>(bias.value());
}
void Bf16GemmSiluMulSm110(
    tvm::ffi::TensorView out, tvm::ffi::TensorView x,
    tvm::ffi::TensorView w_g, tvm::ffi::TensorView w_u,
    tvm::ffi::Optional<tvm::ffi::TensorView> b_g,
    tvm::ffi::Optional<tvm::ffi::TensorView> b_u) {
  xinfer::ffi::CheckCuda(x, "x");
  xinfer::ffi::CheckBFloat16(x, "x");
  xinfer::ffi::CheckContiguous(x, "x");
  xinfer::ffi::CheckBFloat16(w_g, "gate weight");
  xinfer::ffi::CheckBFloat16(w_u, "up weight");
  xinfer::ffi::CheckContiguous(w_g, "gate weight");
  xinfer::ffi::CheckContiguous(w_u, "up weight");
  xinfer::ffi::CheckSameDevice(w_g, x, "gate weight");
  xinfer::ffi::CheckSameDevice(w_u, x, "up weight");
  if (x.ndim() != 2 || w_g.ndim() != 2 || w_u.ndim() != 2 ||
      w_g.size(1) != x.size(1) || w_u.size(0) != w_g.size(0) ||
      w_u.size(1) != w_g.size(1)) {
    throw std::runtime_error("BF16 fused GEMM weight shape mismatch");
  }
  if (x.size(1) % gu_fused::kTileK != 0) {
    throw std::runtime_error("K must be divisible by the kernel tile K");
  }
  xinfer::ffi::CheckBFloat16(out, "out");
  xinfer::ffi::CheckContiguous(out, "out");
  xinfer::ffi::CheckSameDevice(out, x, "out");
  if (out.ndim() != 2 || out.size(0) != x.size(0) ||
      out.size(1) != w_g.size(0)) {
    throw std::runtime_error("out must have shape [x_rows, gate_weight_rows]");
  }
  for (const auto* entry : {&b_g, &b_u}) {
    if (!entry->has_value()) continue;
    const tvm::ffi::TensorView value = entry->value();
    xinfer::ffi::CheckBFloat16(value, "bias");
    xinfer::ffi::CheckContiguous(value, "bias");
    xinfer::ffi::CheckSameDevice(value, x, "bias");
    if (xinfer::ffi::NumElements(value) != w_g.size(0)) {
      throw std::runtime_error("bias width must equal weight rows");
    }
  }
  gu_fused::run(out, x, w_g, w_u, bias_ptr(b_g), bias_ptr(b_u));
}

#else
void Bf16GemmSiluMulSm110(
    tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::TensorView,
    tvm::ffi::TensorView, tvm::ffi::Optional<tvm::ffi::TensorView>,
    tvm::ffi::Optional<tvm::ffi::TensorView>) {
  throw std::runtime_error(
      "BF16 fused GEMM + SiLU-multiply requires CUTLASS SM100 support");
}
#endif

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    bf16_gemm_silu_mul_sm110, Bf16GemmSiluMulSm110);
