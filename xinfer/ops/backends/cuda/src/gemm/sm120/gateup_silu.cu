// SM120 Gate/Up + SiLU multiply.
//
// The GEMM B matrix is interleaved per output pair:
//   [gate_0, up_0, gate_1, up_1, ...]
// The custom CUTLASS epilogue consumes the FP32 accumulator in those pairs,
// computes SiLU(gate) * up, and stores the compact BF16 output.  Thus the
// activation is in the GEMM epilogue, with no gate/up workspace and no second
// activation kernel.
#include <tvm/ffi/function.h>
#include "common/tvm_ffi_cuda_tensor.h"

#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/kernel/gemm.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace xinfer::gemm::sm120::gateup_silu {
using BF16 = cutlass::bfloat16_t;
using RM = cutlass::layout::RowMajor;

// Output iterator receives the interleaved physical GEMM tile, but writes the
// compact [M,N_OUT] result.  Each float4 contains two gate/up pairs.
template <class Base>
class SiluMulStoreIterator : public Base {
 public:
  using ThreadMap = typename Base::ThreadMap;
  using Fragment = typename Base::Fragment;
  using TensorCoord = typename Base::TensorCoord;
  static_assert(Base::kElementsPerAccess == 4,
                "GateUp epilogue expects float4 accumulator access");

  struct Params : Base::Params {
    BF16* out = nullptr;
    int output_cols = 0;
    CUTLASS_HOST_DEVICE Params() = default;
    CUTLASS_HOST_DEVICE Params(RM const& layout) : Base::Params(layout) {}
  };

  Params params;
  int rows;
  int physical_cols;

  CUTLASS_DEVICE SiluMulStoreIterator(
      Params const& p, float* pointer, TensorCoord extent, int tid,
      TensorCoord offset = TensorCoord(), int const* indices = nullptr)
      : Base(p, pointer, extent, tid, offset, indices),
        params(p), rows(extent.row()), physical_cols(extent.column()) {}

  CUTLASS_DEVICE static float silu(float x) {
    return x / (1.0f + __expf(-x));
  }

  CUTLASS_DEVICE void store(Fragment const& f) const {
    CUTLASS_PRAGMA_UNROLL
    for (int cl = 0; cl < ThreadMap::Iterations::kCluster; ++cl) {
      CUTLASS_PRAGMA_UNROLL
      for (int gr = 0; gr < ThreadMap::Iterations::kGroup; ++gr) {
        CUTLASS_PRAGMA_UNROLL
        for (int rr = 0; rr < ThreadMap::Iterations::kRow; ++rr) {
          int fr = rr + ThreadMap::Iterations::kRow *
              (gr + ThreadMap::Iterations::kGroup * cl);
          int row = this->thread_start_row() + rr * ThreadMap::Delta::kRow +
              gr * ThreadMap::Delta::kGroup + cl * ThreadMap::Delta::kCluster;
          CUTLASS_PRAGMA_UNROLL
          for (int cc = 0; cc < ThreadMap::Iterations::kColumn; ++cc) {
            int col = this->thread_start_column() + cc * ThreadMap::Delta::kColumn;
            int fi = (fr * ThreadMap::Iterations::kColumn + cc) * 4;
            if (row < rows && col < physical_cols) {
              // Fast path: a full float4 contains two complete gate/up pairs.
              if (col + 3 < physical_cols) {
                cutlass::Array<BF16, 2> result;
                result[0] = BF16(silu(f[fi]) * f[fi + 1]);
                result[1] = BF16(silu(f[fi + 2]) * f[fi + 3]);
                *reinterpret_cast<cutlass::Array<BF16, 2>*>(
                    params.out + row * params.output_cols + col / 2) = result;
              } else {
                for (int p = 0; p < 2; ++p) {
                  int packed_col = col + 2 * p;
                  if (packed_col + 1 < physical_cols) {
                    params.out[row * params.output_cols + packed_col / 2] =
                        BF16(silu(f[fi + 2 * p]) * f[fi + 2 * p + 1]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
};

template <int TM, int TN, int TK, int WM, int WN, int Stages>
struct Config {
  using ThreadblockShape = cutlass::gemm::GemmShape<TM, TN, TK>;
  using Device = cutlass::gemm::device::Gemm<
      BF16, RM, BF16, RM, float, RM, float,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<TM, TN, TK>,
      cutlass::gemm::GemmShape<WM, WN, TK>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Stages>;
  using BaseKernel = typename Device::GemmKernel;
  using E = typename BaseKernel::Epilogue;
  using Iterator = SiluMulStoreIterator<typename E::OutputTileIterator>;
  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      typename E::Shape, typename E::WarpMmaOperator, E::kPartitionsK,
      Iterator, typename E::AccumulatorFragmentIterator,
      typename E::WarpTileIterator, typename E::SharedLoadIterator,
      typename E::OutputOp, typename E::Padding, E::kFragmentsPerIteration>;
  using Kernel = cutlass::gemm::kernel::Gemm<
      typename BaseKernel::Mma, Epilogue,
      typename BaseKernel::ThreadblockSwizzle, false>;
  using ThreadblockSwizzle = typename BaseKernel::ThreadblockSwizzle;
};

// For K=1024, N_OUT=2048 and M=1..32 the 64-row tile leaves inactive rows,
// so qualify smaller row tiles while
// retaining the same pair-interleaved weight and fused SiLU*up epilogue.
using SmallM16x64 = Config<16, 64, 64, 16, 32, 3>;

// M=256..512: a wider N tile. The choice remains bounded because the 64x64
// tile wins once M is large.
using PrefillM64x128 = Config<64, 128, 64, 32, 32, 3>;

template <class C>
static void launch(BF16* out, const BF16* x, const BF16* packed_weight,
                   int m, int k, int physical_n, int output_n,
                   cudaStream_t stream) {
  using K = typename C::Kernel;
  typename K::ThreadblockSwizzle swizzle;
  cutlass::gemm::GemmCoord problem(m, physical_n, k);
  auto tiled = swizzle.get_tiled_shape(
      problem, {C::ThreadblockShape::kM, C::ThreadblockShape::kN,
                C::ThreadblockShape::kK}, 1);

  typename K::Mma::IteratorA::TensorRef ref_a(
      const_cast<BF16*>(x), k);
  typename K::Mma::IteratorB::TensorRef ref_b(
      const_cast<BF16*>(packed_weight), physical_n);
  typename K::Epilogue::OutputTileIterator::TensorRef ref_c(nullptr, physical_n);
  typename K::Epilogue::OutputTileIterator::TensorRef ref_d(nullptr, physical_n);
  typename K::Params params(problem, tiled, ref_a, ref_b, ref_c, ref_d,
                            typename K::OutputOp::Params(1.f, 0.f));
  params.params_D.out = out;
  params.params_D.output_cols = output_n;

  int smem = sizeof(typename K::SharedStorage);
  if (smem >= (48 << 10)) {
    xinfer::ffi::CheckCudaError(
        cudaFuncSetAttribute(cutlass::Kernel<K>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem),
        "GateUpSiLU cudaFuncSetAttribute");
  }
  dim3 block(K::kThreadCount, 1, 1);
  cutlass::Kernel<K><<<swizzle.get_grid_shape(tiled), block, smem, stream>>>(params);
  xinfer::ffi::CheckCudaError(cudaGetLastError(),
                                "GateUpSiLU launch");
}

static void validate_bf16(tvm::ffi::TensorView t, int rank, const char* name) {
  if (t.ndim() != rank)
    throw std::runtime_error(std::string(name) + " has invalid rank");
  xinfer::ffi::CheckCuda(t, name);
  xinfer::ffi::CheckBFloat16(t, name);
  xinfer::ffi::CheckContiguous(t, name);
  if (!t.data_ptr() || reinterpret_cast<uintptr_t>(t.data_ptr()) % 16)
    throw std::runtime_error(std::string(name) + " must be 16-byte aligned");
}

void Run(tvm::ffi::TensorView out, tvm::ffi::TensorView x,
         tvm::ffi::TensorView packed_weight, int variant) {
  validate_bf16(out, 2, "out");
  validate_bf16(x, 2, "x");
  validate_bf16(packed_weight, 2, "interleaved_weight");
  xinfer::ffi::CheckSameDevice(out, x, "out");
  xinfer::ffi::CheckSameDevice(packed_weight, x, "interleaved_weight");
  xinfer::ffi::CheckNoOverlap(out, "out", x, "x");
  xinfer::ffi::CheckNoOverlap(out, "out", packed_weight, "weight");
  xinfer::ffi::RequireExactComputeCapability(x, 12, 0,
                                               "SM120 GateUpSiLU");

  int64_t m64 = x.size(0);
  int64_t k64 = x.size(1);
  if ((k64 != 1024 && k64 != 2048) || m64 < (k64 == 1024 ? 1 : 256) ||
      m64 > 1024 || packed_weight.size(0) != k64 ||
      packed_weight.size(1) != (k64 == 1024 ? 4096 : 12288) ||
      out.size(0) != m64 || out.size(1) != packed_weight.size(1) / 2)
    throw std::runtime_error(
        "GateUpSiLU requires K=1024/2048, M<=1024, and interleaved [K,2*N_OUT]");

  int device = -1;
  xinfer::ffi::CheckCudaError(cudaGetDevice(&device), "cudaGetDevice");
  if (device != x.device().device_id)
    throw std::runtime_error("x must be on the current CUDA device");
  auto stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));

  int m = static_cast<int>(m64);
  int k = static_cast<int>(k64);
  int physical_n = static_cast<int>(packed_weight.size(1));
  int output_n = physical_n / 2;
  if (variant == -1) {
    // Variant 11 is the qualified small-M decode path. For the monitored
    // prefill shape, M<=512 uses the wider 64x128x64 tile; larger M retains
    // the existing 64x64x64 path because it wins once the grid is saturated.
    if (k == 1024 && physical_n == 4096 && m <= 32)
      variant = 11;
    else if (k == 2048 && physical_n == 12288 &&
             ((m >= 385 && m <= 448) || (m >= 511 && m <= 512)))
      variant = 12;
    else
      variant = 7;
  }

  auto* op = static_cast<BF16*>(out.data_ptr());
  auto* xp = static_cast<const BF16*>(x.data_ptr());
  auto* wp = static_cast<const BF16*>(packed_weight.data_ptr());
  switch (variant) {
    case 0:
      launch<Config<64, 64, 32, 32, 32, 2>>(
          op, xp, wp, m, k, physical_n, output_n, stream);
      break;
    case 7:
      launch<Config<64, 64, 64, 32, 32, 3>>(
          op, xp, wp, m, k, physical_n, output_n, stream);
      break;
    case 11:
      launch<SmallM16x64>(op, xp, wp, m, k, physical_n, output_n, stream);
      break;
    case 12:
      launch<PrefillM64x128>(op, xp, wp, m, k, physical_n, output_n, stream);
      break;
    default:
      throw std::runtime_error("GateUpSiLU variant must be -1,0,7,11,12");
  }
}

}  // namespace xinfer::gemm::sm120::gateup_silu

TVM_FFI_DLL_EXPORT_TYPED_FUNC(
    gateup_silu,
    xinfer::gemm::sm120::gateup_silu::Run);
