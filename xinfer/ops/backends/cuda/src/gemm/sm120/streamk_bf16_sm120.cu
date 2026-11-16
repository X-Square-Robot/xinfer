// Stream-K BF16 GEMM for SM120 (RTX 5090): out = (a @ w^T + bias) + residual.
//
// a [M, K] and w [N, K] are row-major BF16 (w is the original weight, no repack); out and residual are
// [M, N] BF16; bias [N] and residual are optional, and residual may be out itself (each output element is
// read and written by the same thread in the epilogue). Accumulation is FP32; bias and residual are added
// in FP32 and the result is rounded to BF16 once.
//
// Mainloop: CUTLASS 2.x SM80 tensor-op (mma.sync m16n8k16), threadblock 64x128x32, warp 32x64x32, 3 stages,
// compiled for sm_120a. CUTLASS's SM120 collective builder only covers F8F6F4, so dense BF16 uses this
// mainloop; one CTA of it runs the tensor pipe at ~96% on an SM.
//
// Schedule: CUTLASS stream-K (ThreadblockSwizzleStreamK, deterministic peer reduction "kMixed"), balanced as
// one CTA per SM. CUTLASS would balance over its occupancy (2 CTAs of this tile fit an SM), but two
// co-resident CTAs share one tensor pipe, so that balance leaves partial waves at some M (e.g. M 1792:
// 95.8 vs 67.8 us). The launch layer below therefore builds the kernel parameters itself with occupancy 1.
//
// Epilogue: a CUTLASS 2.x visitor tree. The residual load node has no null-pointer fallback, so the
// residual-free call uses a second instantiation without it; both share the tile and the schedule.
//
// Workspace: the partial-tile accumulators live in the caller's workspace (any contents); the reduction
// barriers live in a separate buffer that must be zero on entry. The kernel leaves the barriers zero
// (wait_eq_reset), so the caller keeps one zeroed barrier buffer per stream.

#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include "common/tvm_ffi_cuda_tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <cutlass/arch/mma_sm80.h>
#include <cutlass/cutlass.h>

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
#include <cutlass/bfloat16.h>
#include <cutlass/functional.h>
// The 2.x visitor headers rely on the GEMM kernel headers being included first.
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle_streamk.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

namespace xinfer::gemm::streamk_sm120 {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

using Element = cutlass::bfloat16_t;
using Tile = cutlass::gemm::GemmShape<64, 128, 32>;
using Warp = cutlass::gemm::GemmShape<32, 64, 32>;
constexpr int kStages = 3;
constexpr int kAlign = 8;
constexpr int kOccupancy = 1;  // balance as one CTA per SM (see the file comment)
// Epilogue software pipeline: 2 stages issue the residual loads of the next output step ahead (the 2.x visitor
// tree allows at most 2); with 1 stage every step waits a full L2 round trip (M 1568: 187.0k -> 182.8k cycles).
constexpr int kEpilogueStages = 2;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;

namespace ev = cutlass::epilogue::threadblock;
using ThreadMap = ev::OutputTileThreadLayout<Tile, Warp, Element, kAlign, kEpilogueStages>;
using RowStride = cute::Stride<cute::_0, cute::_1, int32_t>;
using MatStride = cute::Stride<int64_t, cute::_1, int64_t>;
using Plus = ev::VisitorCompute<cutlass::plus, float, float, kRound>;
using Store = ev::VisitorAuxStore<ThreadMap, Element, kRound, MatStride>;
// acc + row_broadcast(bias); a null bias pointer broadcasts zero.
using AccPlusBias = ev::Sm80EVT<Plus, ev::VisitorAccFetch, ev::VisitorRowBroadcast<ThreadMap, Element, RowStride>>;

template <bool kResidual>
struct Variant {
  // D = store((acc + bias) + residual), or store(acc + bias) without a residual.
  using Callbacks = std::conditional_t<
      kResidual,
      ev::Sm80EVT<Store, ev::Sm80EVT<Plus, AccPlusBias, ev::VisitorAuxLoad<ThreadMap, Element, MatStride>>>,
      ev::Sm80EVT<Store, AccPlusBias>>;
  using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
      Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, kAlign,
      Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, kAlign,
      Element, cutlass::layout::RowMajor, kAlign, float, float, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, Tile, Warp, cutlass::gemm::GemmShape<16, 8, 16>, Callbacks,
      cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, kStages, cutlass::arch::OpMultiplyAdd, kEpilogueStages>::GemmKernel;
  using KernelParams = typename Kernel::Params;
  // The visitor kernel keeps the two workspace sizes protected; expose them to place the buffers separately.
  struct Params : KernelParams {
    using KernelParams::KernelParams;
    using KernelParams::get_barrier_workspace_size;
    using KernelParams::get_partials_workspace_size;
  };

  static typename Kernel::Arguments Arguments(int m, int n, int k, const void* a, const void* w, const void* bias,
                                              const void* residual, void* out) {
    const int64_t mn = int64_t{m} * n;
    typename AccPlusBias::Arguments acc_bias{
        {}, {static_cast<const Element*>(bias), Element(0), {cute::_0{}, cute::_1{}, int32_t(n)}}, {}};
    typename Store::Arguments store{static_cast<Element*>(out), {int64_t(n), cute::_1{}, mn}};
    typename Callbacks::Arguments callbacks;
    if constexpr (kResidual) {
      callbacks = {{acc_bias,
                    {static_cast<Element*>(const_cast<void*>(residual)), Element(0), {int64_t(n), cute::_1{}, mn}},
                    {}},
                   store};
    } else {
      callbacks = {acc_bias, store};
    }
    return typename Kernel::Arguments(cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k}, 1, callbacks, a, w,
                                      nullptr, nullptr, int64_t{m} * k, int64_t{n} * k, 0, 0, k, k, 0, 0, -1);
  }

  static void Launch(Params& params, int device, cudaStream_t stream) {
    constexpr int smem = int(sizeof(typename Kernel::SharedStorage));
    if constexpr (smem >= (48 << 10)) {
      static thread_local int configured = -1;
      if (configured != device) {
        xinfer::ffi::CheckCudaError(
            cudaFuncSetAttribute(cutlass::Kernel2<Kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
            "cudaFuncSetAttribute(streamk_bf16_sm120)");
        configured = device;
      }
    }
    cutlass::Kernel2<Kernel><<<params.get_grid_dims(), dim3(Kernel::kThreadCount, 1, 1), smem, stream>>>(
        static_cast<KernelParams&>(params));
  }
};

int DeviceSms(int device) {
  int sms = 0;
  xinfer::ffi::CheckCudaError(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device),
                                "cudaDeviceGetAttribute(MultiProcessorCount)");
  return sms;
}

int CheckedDim(int64_t v, const char* what) {
  if (v < 1 || v > (int64_t{1} << 30)) {
    throw std::runtime_error(std::string("streamk_bf16_sm120: ") + what + " must be in [1, 2^30]");
  }
  return static_cast<int>(v);
}

int CurrentDevice() {
  int device = 0;
  xinfer::ffi::CheckCudaError(cudaGetDevice(&device), "cudaGetDevice");
  return device;
}

// Both variants share the tile and the stream-K mapping, so their workspaces have the same size.
using SizeVariant = Variant<true>;

int64_t PartialsWorkspaceSize(int64_t m, int64_t n, int64_t k) {
  const int mi = CheckedDim(m, "M"), ni = CheckedDim(n, "N"), ki = CheckedDim(k, "K");
  SizeVariant::Params params(SizeVariant::Arguments(mi, ni, ki, nullptr, nullptr, nullptr, nullptr, nullptr),
                             DeviceSms(CurrentDevice()), kOccupancy);
  return static_cast<int64_t>(params.get_partials_workspace_size());
}

int64_t BarrierWorkspaceSize(int64_t m, int64_t n, int64_t k) {
  const int mi = CheckedDim(m, "M"), ni = CheckedDim(n, "N"), ki = CheckedDim(k, "K");
  SizeVariant::Params params(SizeVariant::Arguments(mi, ni, ki, nullptr, nullptr, nullptr, nullptr, nullptr),
                             DeviceSms(CurrentDevice()), kOccupancy);
  return static_cast<int64_t>(params.get_barrier_workspace_size());
}

void CheckMatrix(const TensorView& t, const char* name, int64_t rows, int64_t cols) {
  xinfer::ffi::CheckCuda(t, name);
  xinfer::ffi::CheckBFloat16(t, name);
  xinfer::ffi::CheckContiguous(t, name);
  if (t.ndim() != 2 || t.size(0) != rows || t.size(1) != cols) {
    throw std::runtime_error(std::string("streamk_bf16_sm120: ") + name + " must be [" + std::to_string(rows) +
                             ", " + std::to_string(cols) + "]");
  }
  if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 != 0) {
    throw std::runtime_error(std::string("streamk_bf16_sm120: ") + name + " must be 16-byte aligned");
  }
}

void* CheckBuffer(const Optional<TensorView>& t, const char* name, size_t need, const TensorView& like) {
  if (need == 0) return nullptr;
  if (!t.has_value()) {
    throw std::runtime_error(std::string("streamk_bf16_sm120: this problem needs ") + name + " of " +
                             std::to_string(need) + " bytes");
  }
  const TensorView& b = t.value();
  xinfer::ffi::CheckCuda(b, name);
  xinfer::ffi::CheckUInt8(b, name);
  xinfer::ffi::CheckContiguous(b, name);
  xinfer::ffi::CheckSameDevice(b, like, name);
  if (b.ndim() != 1 || static_cast<size_t>(b.size(0)) < need) {
    throw std::runtime_error(std::string("streamk_bf16_sm120: ") + name + " must be rank-1 uint8 of at least " +
                             std::to_string(need) + " bytes");
  }
  if (reinterpret_cast<uintptr_t>(b.data_ptr()) % 128 != 0) {
    throw std::runtime_error(std::string("streamk_bf16_sm120: ") + name + " must be 128-byte aligned");
  }
  return b.data_ptr();
}

template <bool kResidual>
void Run(int m, int n, int k, const TensorView& out, const Optional<TensorView>& workspace,
         const Optional<TensorView>& barriers, const TensorView& a, const TensorView& w, const void* bias,
         const void* residual) {
  using V = Variant<kResidual>;
  const int device = a.device().device_id;
  typename V::Params params(V::Arguments(m, n, k, a.data_ptr(), w.data_ptr(), bias, residual, out.data_ptr()),
                            DeviceSms(device), kOccupancy);
  params.partials_workspace = CheckBuffer(workspace, "workspace", params.get_partials_workspace_size(), a);
  params.barrier_workspace = CheckBuffer(barriers, "barriers", params.get_barrier_workspace_size(), a);
  V::Launch(params, device, xinfer::ffi::Stream(a));
}

// out = (a @ w^T + bias) + residual.
// workspace: gemm_streamk_bf16_sm120_workspace_size(M, N, K) bytes, any contents.
// barriers: gemm_streamk_bf16_sm120_barrier_size(M, N, K) bytes, zero on entry; the kernel leaves it zero.
void Gemm(TensorView out, Optional<TensorView> workspace, Optional<TensorView> barriers, TensorView a,
          TensorView w, Optional<TensorView> bias, Optional<TensorView> residual) {
  const char* op = "streamk_bf16_sm120";
  xinfer::ffi::RequireArchitecture(a, 12, 0, op);
  if (a.ndim() != 2 || w.ndim() != 2) throw std::runtime_error("streamk_bf16_sm120: a and w must be rank-2");
  const int m = CheckedDim(a.size(0), "M"), k = CheckedDim(a.size(1), "K"), n = CheckedDim(w.size(0), "N");
  if (k % kAlign != 0 || n % kAlign != 0) {
    throw std::runtime_error("streamk_bf16_sm120: K and N must be multiples of 8");
  }
  CheckMatrix(a, "a", m, k);
  CheckMatrix(w, "w", n, k);
  CheckMatrix(out, "out", m, n);
  xinfer::ffi::CheckSameDevice(w, a, "w");
  xinfer::ffi::CheckSameDevice(out, a, "out");
  xinfer::ffi::CheckNoOverlap(out, "out", a, "a");
  xinfer::ffi::CheckNoOverlap(out, "out", w, "w");
  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    const TensorView& b = bias.value();
    xinfer::ffi::CheckCuda(b, "bias");
    xinfer::ffi::CheckBFloat16(b, "bias");
    xinfer::ffi::CheckContiguous(b, "bias");
    xinfer::ffi::CheckSameDevice(b, a, "bias");
    if (b.ndim() != 1 || b.size(0) != n) throw std::runtime_error("streamk_bf16_sm120: bias must be [N]");
    if (reinterpret_cast<uintptr_t>(b.data_ptr()) % 16 != 0) {
      throw std::runtime_error("streamk_bf16_sm120: bias must be 16-byte aligned");
    }
    bias_ptr = b.data_ptr();
  }
  if (residual.has_value()) {
    const TensorView& r = residual.value();
    CheckMatrix(r, "residual", m, n);
    xinfer::ffi::CheckSameDevice(r, a, "residual");
    xinfer::ffi::CheckExactAliasOrNoOverlap(out, "out", r, "residual");
    Run<true>(m, n, k, out, workspace, barriers, a, w, bias_ptr, r.data_ptr());
  } else {
    Run<false>(m, n, k, out, workspace, barriers, a, w, bias_ptr, nullptr);
  }
  xinfer::ffi::CheckLastCudaError(op);
}

}  // namespace xinfer::gemm::streamk_sm120

#else
namespace xinfer::gemm::streamk_sm120 {
[[noreturn]] void Unavailable() { throw std::runtime_error("streamk_bf16_sm120 is not available in this build"); }
void Gemm(tvm::ffi::TensorView, tvm::ffi::Optional<tvm::ffi::TensorView>, tvm::ffi::Optional<tvm::ffi::TensorView>,
          tvm::ffi::TensorView, tvm::ffi::TensorView, tvm::ffi::Optional<tvm::ffi::TensorView>,
          tvm::ffi::Optional<tvm::ffi::TensorView>) {
  Unavailable();
}
int64_t PartialsWorkspaceSize(int64_t, int64_t, int64_t) { Unavailable(); }
int64_t BarrierWorkspaceSize(int64_t, int64_t, int64_t) { Unavailable(); }
}  // namespace xinfer::gemm::streamk_sm120
#endif

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_streamk_bf16_sm120, xinfer::gemm::streamk_sm120::Gemm);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_streamk_bf16_sm120_workspace_size,
                              xinfer::gemm::streamk_sm120::PartialsWorkspaceSize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_streamk_bf16_sm120_barrier_size,
                              xinfer::gemm::streamk_sm120::BarrierWorkspaceSize);
