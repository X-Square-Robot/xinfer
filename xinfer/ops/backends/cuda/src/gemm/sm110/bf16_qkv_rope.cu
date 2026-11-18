// Fused qkv GEMM + bias + rotate_half RoPE, hand-written tcgen05 (Thor sm_110a).
//
//   qkv[M,N] = rope( x @ W_perm^T + bias_perm )   (rope on cols [0,rot_cols))
//   x [M,K] bf16 K-major | W_perm [N,K] bf16 K-major | qkv [M,N] bf16
//
// Pair-adjacent weight permutation (offline): rotate_half partners land in
// ADJACENT output columns (2j,2j+1). Kernel emits pair-interleaved layout.
//   d = acc + bias;  for n<rot_cols, j=(n%hd)/2:
//     out[2j]   = d[2j]*cos[j] - d[2j+1]*sin[j]
//     out[2j+1] = d[2j+1]*cos[j] + d[2j]*sin[j]
//   cols [rot_cols,N) (V region): out = d (bias only).
// cs table: float4 (hd/4,M) = {cos_2j,sin_2j,cos_2j+1,sin_2j+1}, cs_f4[jj*M+m], jj=(n%hd)>>2.
//
// PERSISTENT + OVERLAP (2-SM MMA). A cluster of 2 CTAs collaborates on a 256x256
// 2-SM MMA (SM100_MMA_F16BF16_2x1SM_SS; leader owns rows 0-127, peer 128-255).
// Persistent: grid = one wave of clusters; each cluster loops over a strided set of
// output tiles. Warp-specialized WITHIN the CTA with NO phase-separating __syncthreads:
//   warp 5 = TMA producer, warp 4 = MMA (leader), warps 0-3 = epilogue drain.
// Two tmem accumulator buffers double-buffer the MMA<->epilogue handoff, so the
// epilogue of tile i overlaps the mainloop of tile i+1 (baseline's trick) => the
// ~93us mainloop hides under the epilogue instead of running serially. Pipelines:
//   full[]/empty[]  : producer<->MMA smem stages (continuous across tiles)
//   acc_full[2]     : MMA->epilogue  (leader umma_arrive_multicast_2x1SM)
//   acc_empty[2]    : epilogue->MMA tmem-free (umma_arrive_2x1SM_sm0, both CTAs -> leader)
// Epilogue: 2 column-groups (128 thr each) drain the 256-col tile. Per subtile: tmem->reg,
// bias+cos/sin+rope in registers (cs/bias loaded point-of-use from L2-resident gmem, no smem
// preload/barrier), rope result -> smem sD -> ASYNC SM90_TMA_STORE (coalesced, bounds-checked,
// drops the M-padding rows). The TMA store (vs a strided per-thread uint4 store) was the key win:
// it frees the epilogue warps and kills the uncoalesced-store lg_throttle. acc_empty is signalled
// right after the tmem reads (before the stores drain) so the MMA reuses tmem sooner.

#include <tvm/ffi/function.h>

#include "common/tvm_ffi_cuda_tensor.h"

#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>

#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace qkv_rope_ns {

using namespace cute;

using TA   = cutlass::bfloat16_t;
using TB   = cutlass::bfloat16_t;
using TAcc = float;
using TD   = cutlass::bfloat16_t;

// --- sweep-overridable via -D (defaults = the iter-43 best config) ---
#ifndef QKV_MMAN
#define QKV_MMAN 256
#endif
#ifndef QKV_STAGES
#define QKV_STAGES 4
#endif
#ifndef QKV_EPIN
#define QKV_EPIN 32
#endif
#ifndef QKV_EPIBUF
#define QKV_EPIBUF 3
#endif
#ifndef QKV_ACCBUF
#define QKV_ACCBUF 2
#endif
#ifndef QKV_EPIGROUPS
#define QKV_EPIGROUPS 2
#endif

static constexpr int kMmaM  = 256;                  // 2-SM MMA M (cluster tile); each CTA owns 128
static constexpr int kMmaN  = QKV_MMAN;
static constexpr int kTileK = 64;
static constexpr int kStages = QKV_STAGES;          // producer<->MMA smem pipeline depth
static constexpr int kEpiN  = QKV_EPIN;             // epilogue subtile width
static constexpr int kClusterM = 2;                 // 2-SM pair
static constexpr int kClusterN = 1;
static constexpr int kMmaMcta = kMmaM / kClusterM;  // 128 output rows / CTA
static constexpr int kHd = 80;                       // fixed head dim (rope)
static constexpr int kCsPerRow = kHd / 4;            // 20 cos/sin float4 per row
static constexpr int kAccBuf = QKV_ACCBUF;           // double-buffered tmem accumulators
static constexpr int kTmemCols = kAccBuf * kMmaN;   // <=512 tmem columns/CTA
static constexpr int kEpiBuf = QKV_EPIBUF;           // smem sD buffers per group (TMA-store pipeline)

static constexpr int kEpiGroups   = QKV_EPIGROUPS;   // # of 128-thread column groups draining the tile
static constexpr int kEpiWarps    = kEpiGroups * 4;  // 128-thread groups -> 4 warps each
static constexpr int kEpiThreads  = kEpiWarps * 32;  // kEpiGroups * 128
static constexpr int kMmaWarp     = kEpiWarps;       // MMA (leader) warp right after the epilogue warps
static constexpr int kProducerWarp = kEpiWarps + 1;  // TMA producer warp
static constexpr int kEpiBarId    = 1;               // named barrier base (per-group = kEpiBarId+grp)

CUTE_DEVICE void epi_named_barrier() {
  asm volatile("bar.sync %0, %1;" :: "r"(kEpiBarId), "r"(kEpiThreads) : "memory");
}
CUTE_DEVICE void epi_group_barrier(int grp) {   // sync one 128-thread column group only
  asm volatile("bar.sync %0, 128;" :: "r"(kEpiBarId + 1 + grp) : "memory");
}

// ---------------------------------------------------------------------------
template <class ASmemLayout, class BSmemLayout, class DSmemLayout>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<TA, kStages * cute::cosize_v<ASmemLayout>> A;
  alignas(128) cute::ArrayEngine<TB, kStages * cute::cosize_v<BSmemLayout>> B;
  alignas(128) cute::ArrayEngine<TD, kEpiGroups * kEpiBuf * cute::cosize_v<DSmemLayout>> D;  // TMA-store staging: per-group ping-pong
  alignas(16) cute::uint64_t full[kStages];   // TMA-fill (transaction-byte) barriers (leader waits)
  alignas(16) cute::uint64_t empty[kStages];  // MMA-done (smem-reuse) barriers
  alignas(16) cute::uint64_t acc_full[kAccBuf];   // MMA -> epilogue (tile acc ready)
  alignas(16) cute::uint64_t acc_empty[kAccBuf];  // epilogue -> MMA (tmem buffer free)
  alignas(16) cute::uint32_t tmem_base_ptr;
};

// ---------------------------------------------------------------------------
template <class SmemT, class ASmemLayout, class BSmemLayout, class DSmemLayout,
          class ATensor, class BTensor, class DTensor,
          class MmaTiler, class TiledMMA, class EpiTiler, class ClusterShape,
          class TmaAtomA, class TmaAtomB, class TmaAtomD>
__global__ static void
qkv_device(ATensor mA, BTensor mW, DTensor mD, TD* __restrict__ out_ptr,
           const TB* __restrict__ bias_ptr, const float* __restrict__ cs_ptr,
           int M, int N, int rot_cols, int hd,
           int num_tiles, int num_m_tiles, int num_clusters,
           MmaTiler mma_tiler, TiledMMA tiled_mma, EpiTiler epi_tiler_mn, ClusterShape cluster_shape,
           ASmemLayout sA_layout, BSmemLayout sB_layout, DSmemLayout sD_layout,
           CUTE_GRID_CONSTANT TmaAtomA const tma_A,
           CUTE_GRID_CONSTANT TmaAtomB const tma_W,
           CUTE_GRID_CONSTANT TmaAtomD const tma_D)
{
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename TiledMMA::AtomThrID{}));
  int  mma_v      = int(blockIdx.x) % size<0>(cluster_layout_vmnk);   // 0=leader,1=peer
  int  cluster_id = int(blockIdx.x) / size<0>(cluster_layout_vmnk);
  ThrMMA cta_mma  = tiled_mma.get_slice(mma_v);

  extern __shared__ char smem_raw[];
  SmemT& smem = *reinterpret_cast<SmemT*>(smem_raw);

  const int warp_idx = threadIdx.x / 32;
  const int etid     = threadIdx.x;               // epilogue thread id (warps 0-3 == 0..127)
  uint32_t elect = cute::elect_one_sync();

  auto cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(int(cute::block_rank_in_cluster()));
  bool elect_one_cta  = (get<0>(cta_coord_vmnk) == Int<0>{});   // leader CTA

  using TmemAllocator = cute::TMEM::Allocator2Sm;
  TmemAllocator tmem_allocator{};
  if (warp_idx == 0) tmem_allocator.allocate(kTmemCols, &smem.tmem_base_ptr);
  if (warp_idx == 0 && elect) {
    CUTE_UNROLL
    for (int s = 0; s < kStages; ++s) {
      cute::initialize_barrier(smem.full[s],  1);
      cute::initialize_barrier(smem.empty[s], 1);   // bare 2x1 => 1 mcast participant
    }
    CUTE_UNROLL
    for (int b = 0; b < kAccBuf; ++b) {
      cute::initialize_barrier(smem.acc_full[b],  1);   // leader multicast -> 1 per CTA
      cute::initialize_barrier(smem.acc_empty[b], 2);   // both CTAs' epilogue arrive on leader
    }
  }
  cute::cluster_sync();
  uint32_t tmem_base = smem.tmem_base_ptr;

  // reference partitions / layouts (tile-independent) for smem + fragment shapes
  auto gA0 = local_tile(mA, mma_tiler, make_coord(0,0,_), Step<_1,X,_1>{});
  auto gW0 = local_tile(mW, mma_tiler, make_coord(0,0,_), Step<X,_1,_1>{});
  auto gD0 = local_tile(mD, mma_tiler, make_coord(0,0,_), Step<_1,_1,X>{});
  auto tCgA0 = cta_mma.partition_A(gA0);
  auto tCgW0 = cta_mma.partition_B(gW0);
  auto acc_ref = cta_mma.partition_C(gD0);
  auto acc_of = [&](int buf) { auto t = cta_mma.make_fragment_C(acc_ref);
                               t.data() = tmem_base + uint32_t(buf * kMmaN); return t; };

  constexpr int coA = cute::cosize_v<ASmemLayout>;
  constexpr int coB = cute::cosize_v<BSmemLayout>;
  auto sA_of = [&](int s) { return make_tensor(make_smem_ptr(smem.A.begin() + s * coA), sA_layout); };
  auto sB_of = [&](int s) { return make_tensor(make_smem_ptr(smem.B.begin() + s * coB), sB_layout); };

  auto tAsA = [&](int s) { return get<1>(tma_partition(tma_A, get<2>(cta_coord_vmnk),
                 make_layout(size<2>(cluster_layout_vmnk)), group_modes<0,3>(sA_of(s)), group_modes<0,3>(tCgA0))); };
  auto tBsB = [&](int s) { return get<1>(tma_partition(tma_W, get<1>(cta_coord_vmnk),
                 make_layout(size<1>(cluster_layout_vmnk)), group_modes<0,3>(sB_of(s)), group_modes<0,3>(tCgW0))); };

  uint16_t peer_mask = create_tma_multicast_mask<0,1>(cluster_layout_vmnk, cta_coord_vmnk)
                     | create_tma_multicast_mask<0,2>(cluster_layout_vmnk, cta_coord_vmnk);

  int tma_bytes = int(size<0>(cluster_layout_vmnk)) *
                  (int(sizeof(make_tensor_like(tAsA(0)))) + int(sizeof(make_tensor_like(tBsB(0)))));

  const int k_tiles = size<3>(tCgA0);
  const int nkb     = size<2>(cta_mma.make_fragment_A(sA_of(0)));

  // =========================== PRODUCER (warp 5) ===========================
  if (warp_idx == kProducerWarp) {
    int p_empty[kStages];
    CUTE_UNROLL
    for (int s = 0; s < kStages; ++s) p_empty[s] = 0;
    int stage_iter = 0;
    for (int t = cluster_id; t < num_tiles; t += num_clusters) {
      int mt = t % num_m_tiles, nt = t / num_m_tiles;  // COLUMN-MAJOR raster: consecutive tiles
      auto gA = local_tile(mA, mma_tiler, make_coord(mt, nt, _), Step<_1,X,_1>{});
      auto gW = local_tile(mW, mma_tiler, make_coord(mt, nt, _), Step<X,_1,_1>{});
      auto tCgA = cta_mma.partition_A(gA);
      auto tCgW = cta_mma.partition_B(gW);
      auto tAgA = get<0>(tma_partition(tma_A, get<2>(cta_coord_vmnk),
                    make_layout(size<2>(cluster_layout_vmnk)), group_modes<0,3>(sA_of(0)), group_modes<0,3>(tCgA)));
      auto tBgB = get<0>(tma_partition(tma_W, get<1>(cta_coord_vmnk),
                    make_layout(size<1>(cluster_layout_vmnk)), group_modes<0,3>(sB_of(0)), group_modes<0,3>(tCgW)));
      for (int kt = 0; kt < k_tiles; ++kt) {
        int s = stage_iter % kStages;
        if (stage_iter >= kStages) { cute::wait_barrier(smem.empty[s], p_empty[s]); p_empty[s] ^= 1; }
        if (elect) {
          if (elect_one_cta) cute::set_barrier_transaction_bytes(smem.full[s], tma_bytes);
          copy(tma_A.with(smem.full[s]), tAgA(_, kt), tAsA(s));
          copy(tma_W.with(smem.full[s]), tBgB(_, kt), tBsB(s));
        }
        ++stage_iter;
      }
    }
  }
  // =========================== MMA (warp 4, leader) ========================
  else if (warp_idx == kMmaWarp && elect_one_cta) {
    int p_full[kStages];
    CUTE_UNROLL
    for (int s = 0; s < kStages; ++s) p_full[s] = 0;
    int p_acc_empty[kAccBuf];
    CUTE_UNROLL
    for (int b = 0; b < kAccBuf; ++b) p_acc_empty[b] = 0;
    int stage_iter = 0, tile_idx = 0;
    for (int t = cluster_id; t < num_tiles; t += num_clusters) {
      int buf = tile_idx % kAccBuf;
      if (tile_idx >= kAccBuf) { cute::wait_barrier(smem.acc_empty[buf], p_acc_empty[buf]); p_acc_empty[buf] ^= 1; }
      auto acc = acc_of(buf);
      for (int kt = 0; kt < k_tiles; ++kt) {
        int s = stage_iter % kStages;
        cute::wait_barrier(smem.full[s], p_full[s]); p_full[s] ^= 1;
        Tensor tCrB = cta_mma.make_fragment_B(sB_of(s));
        Tensor tCrA = cta_mma.make_fragment_A(sA_of(s));
        tiled_mma.accumulate_ = (kt == 0) ? UMMA::ScaleOut::Zero : UMMA::ScaleOut::One;
        CUTE_UNROLL
        for (int kb = 0; kb < nkb; ++kb) {
          gemm(tiled_mma, tCrA(_,_,kb), tCrB(_,_,kb), acc);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&smem.empty[s], peer_mask);
        ++stage_iter;
      }
      cutlass::arch::umma_arrive_multicast_2x1SM(&smem.acc_full[buf], peer_mask);
      ++tile_idx;
    }
  }
  // =========================== EPILOGUE (warps 0-3) ========================
  else if (warp_idx < kEpiWarps) {
    const float4* cs_f4 = reinterpret_cast<const float4*>(cs_ptr);
    auto epi_tiler_v = make_tile(epi_tiler_mn);
    constexpr int coD = cute::cosize_v<DSmemLayout>;
    auto sD_of = [&](int bf) { return make_tensor(make_smem_ptr(smem.D.begin() + bf * coD), sD_layout); };
    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{},
                                   zipped_divide(acc_of(0), epi_tiler_v)(_, Int<0>{}));
    const int grp   = etid >> 7;          // 0..kEpiGroups-1 -> column group of the 256-col accumulator
    const int lane  = etid & 127;         // lane within the 128-thread warp-group
    const bool grp_leader = (warp_idx % 4 == 0) && elect;  // one thread/group issues the TMA store
    ThrCopy thr = t2r.get_slice(lane);
    auto racc = make_tensor<TAcc>(shape(thr.partition_D(sD_of(0))));
    auto rd   = make_tensor<TD>  (shape(thr.partition_D(sD_of(0))));
    constexpr int NR = kEpiN;
    constexpr int NG = NR / 4;
    constexpr int kNEpiG = (kMmaN / kEpiN) / kEpiGroups;   // # column subtiles per group

    int p_acc_full[kAccBuf];
    CUTE_UNROLL
    for (int b = 0; b < kAccBuf; ++b) p_acc_full[b] = 0;
    int tile_idx = 0;
    int st = 0;                    // continuous store-buffer counter (flows across tile boundaries)
    for (int t = cluster_id; t < num_tiles; t += num_clusters) {
      int buf = tile_idx % kAccBuf;
      cute::wait_barrier(smem.acc_full[buf], p_acc_full[buf]); p_acc_full[buf] ^= 1;

      int mt = t % num_m_tiles, nt = t / num_m_tiles;  // COLUMN-MAJOR raster (share nt/B in L2)
      auto acc = acc_of(buf);
      auto cD  = local_tile(make_identity_tensor(make_shape(M, N)), mma_tiler, make_coord(mt, nt, _), Step<_1,_1,X>{});
      auto gD  = local_tile(mD, mma_tiler, make_coord(mt, nt, _), Step<_1,_1,X>{});
      auto tTRacc = thr.partition_S(zipped_divide(acc, epi_tiler_v));
      auto tTRcD  = thr.partition_D(zipped_divide(cta_mma.partition_C(cD), epi_tiler_v));
      auto tCgD   = cta_mma.partition_C(gD);
      // Each group drains kNEpiG column subtiles; each subtile: tmem->reg (rope) ->smem-> async
      // TMA store (coalesced + bounds-checked, so M-padding rows are dropped by the descriptor).
      for (int es = 0; es < kNEpiG; ++es) {
        int e   = grp * kNEpiG + es;
        int bf  = grp * kEpiBuf + (st % kEpiBuf);   // rotating smem buffer, continuous across tiles
        auto sD_b = sD_of(bf);
        auto tSG = tma_partition(tma_D, sD_b, zipped_divide(tCgD, epi_tiler_v));
        auto tSG_gD = get<0>(tSG);   // gmem-side partition (subtile mode -> tSG_gD(_,e))
        auto tSG_sD = get<1>(tSG);   // smem-side partition of sD_b
        auto tTR_sD = thr.partition_D(sD_b);
        copy(t2r, tTRacc(_,_,e), racc);            // tmem -> reg (all threads, no divergence)

        auto cD_e = tTRcD(_,_,e);
        int m = get<0>(cD_e(0));
        int nbase = get<1>(cD_e(0));               // GLOBAL column (multiple of 4)
        int mc = (m < M) ? m : (M - 1);            // clamp cs row (padding rows dropped by TMA)
        bool do_rope = (nbase < rot_cols);         // subtile uniform (rot_cols % kMmaN == 0)
        // point-of-use gmem loads issued up front so their L2-hit latency overlaps
        float4 bias_v[NG];
        if (bias_ptr) {
          CUTE_UNROLL
          for (int g = 0; g < NG; ++g) {
            const TB* bp = bias_ptr + (nbase + g * 4);
            bias_v[g] = make_float4(float(bp[0]), float(bp[1]), float(bp[2]), float(bp[3]));
          }
        } else {
          CUTE_UNROLL
          for (int g = 0; g < NG; ++g) bias_v[g] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        float4 cs_v[NG];
        if (do_rope) {
          CUTE_UNROLL
          for (int g = 0; g < NG; ++g) {
            int nh = (nbase + g * 4) % hd;
            cs_v[g] = cs_f4[(int64_t)(nh >> 2) * M + mc];
          }
        }
        CUTE_UNROLL
        for (int g = 0; g < NG; ++g) {
          int v = g * 4;
          float d0 = racc(v)   + bias_v[g].x;
          float d1 = racc(v+1) + bias_v[g].y;
          float d2 = racc(v+2) + bias_v[g].z;
          float d3 = racc(v+3) + bias_v[g].w;
          if (do_rope) {
            float4 f4 = cs_v[g];
            rd(v)   = TD(d0 * f4.x - d1 * f4.y);
            rd(v+1) = TD(d1 * f4.x + d0 * f4.y);
            rd(v+2) = TD(d2 * f4.z - d3 * f4.w);
            rd(v+3) = TD(d3 * f4.z + d2 * f4.w);
          } else {
            rd(v) = TD(d0); rd(v+1) = TD(d1); rd(v+2) = TD(d2); rd(v+3) = TD(d3);
          }
        }
        if (grp_leader && st >= kEpiBuf) tma_store_wait<kEpiBuf - 1>();
        epi_group_barrier(grp);                    // sD_b free to overwrite
        copy(rd, tTR_sD);                          // reg -> smem
        cutlass::arch::fence_view_async_shared();
        epi_group_barrier(grp);                    // sD_b visible to TMA
        if (grp_leader) { copy(tma_D, tSG_sD, tSG_gD(_, e)); tma_store_arrive(); }
        ++st;
      }
      // tmem[buf] is fully READ once every thread passed its last copy(t2r,...); free it for the
      // MMA NOW (before the TMA stores finish draining smem->gmem) so the next tile's MMA overlaps.
      // No tile-boundary store drain: buffers rotate continuously via `st`, so the TMA store pipe
      // flows across tiles (only per-reuse tma_store_wait<kEpiBuf-1> above; final drain after loop).
      epi_named_barrier();   // all epilogue threads done reading tmem[buf]
      if (etid == 0) cutlass::arch::umma_arrive_2x1SM_sm0(&smem.acc_empty[buf]);
      ++tile_idx;
    }
    if (grp_leader) tma_store_wait<0>();            // drain the last in-flight stores
  }

  cute::cluster_sync();   // keep cluster alive until all warps/peers finish
  if (warp_idx == 0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(smem.tmem_base_ptr, kTmemCols);
  }
}

// ---------------------------------------------------------------------------
void run(tvm::ffi::TensorView out, tvm::ffi::TensorView a,
         tvm::ffi::TensorView w_perm, const TB* bias_ptr,
         const float* cs_ptr, int M, int N, int K, int rot_cols, int hd) {
  cudaStream_t stream = xinfer::ffi::Stream(a);

  auto mA = make_tensor(make_gmem_ptr(reinterpret_cast<const TA*>(a.data_ptr())),
                        make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
  auto mW = make_tensor(make_gmem_ptr(reinterpret_cast<const TB*>(w_perm.data_ptr())),
                        make_layout(make_shape(N, K), make_stride(K, Int<1>{})));
  auto mD0 = make_tensor(make_gmem_ptr(reinterpret_cast<TD*>(out.data_ptr())),
                         make_layout(make_shape(M, N), make_stride(N, Int<1>{})));

  TiledMMA tiled_mma = make_tiled_mma(
      SM100_MMA_F16BF16_2x1SM_SS<TA, TB, TAcc, kMmaM, kMmaN,
                                 UMMA::Major::K, UMMA::Major::K>{});
  auto mma_tiler = make_shape(Int<kMmaM>{}, Int<kMmaN>{}, Int<kTileK>{});
  auto cluster_shape = make_shape(Int<kClusterM>{}, Int<kClusterN>{}, Int<1>{});
  Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                                            make_tile(typename decltype(tiled_mma)::AtomThrID{}));

  auto mma_shape_A = partition_shape_A(tiled_mma, make_shape(Int<kMmaM>{}, Int<kTileK>{}));
  auto mma_shape_B = partition_shape_B(tiled_mma, make_shape(Int<kMmaN>{}, Int<kTileK>{}));
  auto sA_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TA>{}, mma_shape_A);
  auto sB_layout = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<TB>{}, mma_shape_B);

  auto epi_tiler = make_tile(Int<kMmaM / kClusterM>{}, Int<kEpiN>{});   // per-CTA epilogue tile (128 x 32)
  auto sD_layout_mn = tile_to_shape(UMMA::Layout_K_SW64_Atom<TD>{},
                                    make_shape(Int<kMmaM / kClusterM>{}, Int<kEpiN>{}));
  auto sD_layout = group<0,2>(sD_layout_mn);

  using SmemT = SharedStorage<decltype(sA_layout), decltype(sB_layout), decltype(sD_layout)>;

  Copy_Atom tma_A = make_tma_atom_A_sm100(SM100_TMA_2SM_LOAD{}, mA, sA_layout, mma_tiler,
                                          tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_W = make_tma_atom_B_sm100(SM100_TMA_2SM_LOAD{}, mW, sB_layout, mma_tiler,
                                          tiled_mma, cluster_layout_vmnk);
  Copy_Atom tma_D = make_tma_atom(SM90_TMA_STORE{}, mD0, sD_layout, epi_tiler);   // async coalesced store
  auto mA_tma = tma_A.get_tma_tensor(shape(mA));
  auto mW_tma = tma_W.get_tma_tensor(shape(mW));
  auto mD_tma = tma_D.get_tma_tensor(shape(mD0));

  int num_m_tiles = ceil_div(M, kMmaM);
  int num_n_tiles = ceil_div(N, kMmaN);
  int num_tiles   = num_m_tiles * num_n_tiles;
  int num_sms = 20;
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount,
                         a.device().device_id);
  int num_clusters = num_sms / kClusterM;   // one wave: 10 clusters (2 CTAs each) on 20 SMs
  if (num_clusters < 1) num_clusters = 1;
  if (num_clusters > num_tiles) num_clusters = num_tiles;

  dim3 block((kEpiWarps + 2) * 32);   // epilogue warps + MMA warp + producer warp
  dim3 cluster(kClusterM, kClusterN, 1);
  dim3 grid(kClusterM * num_clusters, 1, 1);
  int smem_bytes = sizeof(SmemT);

  auto* kernel = &qkv_device<SmemT, decltype(sA_layout), decltype(sB_layout),
                             decltype(sD_layout), decltype(mA_tma), decltype(mW_tma),
                             decltype(mD_tma), decltype(mma_tiler), decltype(tiled_mma),
                             decltype(epi_tiler), decltype(cluster_shape), decltype(tma_A),
                             decltype(tma_W), decltype(tma_D)>;
  xinfer::ffi::CheckCudaError(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           smem_bytes),
      "bf16_qkv_rope_sm110 cudaFuncSetAttribute");

  cutlass::ClusterLaunchParams params{grid, block, cluster, smem_bytes, stream};
  cutlass::launch_kernel_on_cluster(params, (void const*)kernel,
                                    mA_tma, mW_tma, mD_tma, reinterpret_cast<TD*>(out.data_ptr()),
                                    bias_ptr, cs_ptr, M, N, rot_cols, hd,
                                    num_tiles, num_m_tiles, num_clusters,
                                    mma_tiler, tiled_mma, epi_tiler, cluster_shape,
                                    sA_layout, sB_layout, sD_layout,
                                    tma_A, tma_W, tma_D);
}

}  // namespace qkv_rope_ns

// Drop-in for the CUTLASS bf16_qkv_rope (see cutlass/qkv_rope_bf16.cu): same
// 7-arg ABI (a, w_perm, bias_perm, cs, q_dim, k_dim, hd), same pair-permuted
// weight + cs-table contract. ~1.05x faster on the ViT qkv shape.
void Bf16QkvRopeSm110(tvm::ffi::TensorView out,
                      tvm::ffi::TensorView a,
                      tvm::ffi::TensorView w_perm,
                      tvm::ffi::TensorView bias_perm,
                      tvm::ffi::TensorView cs, int64_t q_dim,
                      int64_t k_dim, int64_t hd) {
  xinfer::ffi::CheckCuda(a, "a");
  xinfer::ffi::CheckBFloat16(a, "a");
  xinfer::ffi::CheckContiguous(a, "a");
  for (const auto& entry : {w_perm, bias_perm, out}) {
    xinfer::ffi::CheckBFloat16(entry, "BF16 QKV tensor");
    xinfer::ffi::CheckContiguous(entry, "BF16 QKV tensor");
    xinfer::ffi::CheckSameDevice(entry, a, "BF16 QKV tensor");
  }
  xinfer::ffi::CheckFloat32(cs, "cosine/sine table");
  xinfer::ffi::CheckContiguous(cs, "cosine/sine table");
  xinfer::ffi::CheckSameDevice(cs, a, "cosine/sine table");
  if (a.ndim() != 2 || w_perm.ndim() != 2) {
    throw std::runtime_error("a and weight must be 2D");
  }
  int M = a.size(0), K = a.size(1), N = w_perm.size(0);
  if (w_perm.size(1) != K) {
    throw std::runtime_error("weight K mismatch");
  }
  if (K % qkv_rope_ns::kTileK != 0) {
    throw std::runtime_error("K must be divisible by the kernel tile K");
  }
  if (xinfer::ffi::NumElements(bias_perm) != N) {
    throw std::runtime_error("bias width must equal weight rows");
  }
  if (out.ndim() != 2 || out.size(0) != M || out.size(1) != N) {
    throw std::runtime_error("out must have shape [a_rows, weight_rows]");
  }
  int rot_cols = (int)(q_dim + k_dim);
  if ((int)hd != qkv_rope_ns::kHd) {
    throw std::runtime_error("BF16 QKV RoPE head dimension is unsupported");
  }
  qkv_rope_ns::run(
      out, a, w_perm, xinfer::ffi::ConstData<qkv_rope_ns::TB>(bias_perm),
      xinfer::ffi::ConstData<float>(cs), M, N, K, rot_cols, (int)hd);
}

#else

void Bf16QkvRopeSm110(tvm::ffi::TensorView, tvm::ffi::TensorView,
                      tvm::ffi::TensorView, tvm::ffi::TensorView,
                      tvm::ffi::TensorView, int64_t, int64_t, int64_t) {
  throw std::runtime_error(
      "BF16 QKV projection + RoPE requires CUTLASS SM100 support");
}

#endif

TVM_FFI_DLL_EXPORT_TYPED_FUNC(bf16_qkv_rope_sm110, Bf16QkvRopeSm110);
