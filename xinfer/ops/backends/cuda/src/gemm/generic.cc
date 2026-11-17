// Generic GEMM: out[M, N] = alpha * a[M, K] @ w^T (+ bias[N]) (+ beta * residual[M, N]) through cuBLASLt,
// BF16 or FP32 (all operands one dtype, FP32 accumulation; FP32 without TF32), with the weight as w[N, K]
// or, w_kn, as w[K, N]; and the same product of MXFP8 operands with a BF16 result. It runs every call no
// specialized kernel takes; the epilogues cuBLASLt lacks run in epilogue.cu. NVFP4 operands (gemm_generic_nvfp4)
// take alpha on the FP32 accumulator.
//
// The algorithm is cuBLASLt's heuristic choice for a fixed workspace size (never timed), so it is the same
// in every process.
#include <cublasLt.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

#include "common/tvm_ffi_cuda_tensor.h"

namespace xinfer::gemm {
namespace {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

constexpr int64_t kWorkspaceBytes = int64_t{32} << 20;

void Check(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("gemm generic: ") + what + " failed with cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

// Largest power of two (<= 256) dividing the address; cuBLASLt otherwise assumes 256-byte alignment.
uint32_t Alignment(const void* p) {
  const uintptr_t v = reinterpret_cast<uintptr_t>(p);
  uint32_t a = 256;
  while (a > 1 && (v % a)) a >>= 1;
  return a;
}

struct Plan {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  cublasLtMatmulAlgo_t algo{};
};

// (device, m, n, k, bias, residual, algo index, alignments of w, a, residual/out, bias, fp32, w_kn)
using PlanKey = std::tuple<int, int64_t, int64_t, int64_t, bool, bool, int64_t, uint32_t, uint32_t, uint32_t,
                           uint32_t, bool, bool>;

struct State {
  std::mutex mu;
  std::map<int, cublasLtHandle_t> handles;
  std::map<PlanKey, Plan> plans;
};

State& GetState() {
  static State state;
  return state;
}

cublasLtHandle_t Handle(State& s, int device) {
  auto it = s.handles.find(device);
  if (it != s.handles.end()) return it->second;
  cublasLtHandle_t h = nullptr;
  Check(cublasLtCreate(&h), "cublasLtCreate");
  s.handles[device] = h;
  return h;
}

// Row-major out[M, N] = a[M, K] @ w[N, K]^T is column-major out^T[N, M] = w^T @ a: cuBLASLt "TN" with
// m' = N, n' = M, so the bias epilogue (one value per row of out^T) is the per-column bias of out.
Plan& GetPlan(State& s, const PlanKey& key, int64_t algo_index) {
  auto it = s.plans.find(key);
  if (it != s.plans.end()) return it->second;
  const int device = std::get<0>(key);
  const int64_t m = std::get<1>(key), n = std::get<2>(key), k = std::get<3>(key);
  const bool has_bias = std::get<4>(key);
  const bool fp32 = std::get<11>(key), w_kn = std::get<12>(key);
  const cudaDataType dtype = fp32 ? CUDA_R_32F : CUDA_R_16BF;
  Plan p;
  Check(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
  // w[N, K] row-major is column-major [K, N] (use transposed); w[K, N] row-major is column-major [N, K].
  const cublasOperation_t ta = w_kn ? CUBLAS_OP_N : CUBLAS_OP_T, tb = CUBLAS_OP_N;
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "TRANSA");
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "TRANSB");
  if (has_bias) {
    const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)), "EPILOGUE");
    const cudaDataType bias_type = dtype;
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                         sizeof(bias_type)),
          "BIAS_DATA_TYPE");
  }
  if (w_kn) {
    Check(cublasLtMatrixLayoutCreate(&p.a, dtype, n, k, n), "layout w");
  } else {
    Check(cublasLtMatrixLayoutCreate(&p.a, dtype, k, n, k), "layout w");
  }
  Check(cublasLtMatrixLayoutCreate(&p.b, dtype, k, m, k), "layout a");
  Check(cublasLtMatrixLayoutCreate(&p.c, dtype, n, m, n), "layout out");

  cublasLtMatmulPreference_t pref = nullptr;
  Check(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
  const uint64_t ws = kWorkspaceBytes;
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
        "MAX_WORKSPACE_BYTES");
  const uint32_t align_w = std::get<7>(key), align_a = std::get<8>(key), align_c = std::get<9>(key);
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_w,
                                             sizeof(align_w)),
        "MIN_ALIGNMENT_A");
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_a,
                                             sizeof(align_a)),
        "MIN_ALIGNMENT_B");
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align_c,
                                             sizeof(align_c)),
        "MIN_ALIGNMENT_C");
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align_c,
                                             sizeof(align_c)),
        "MIN_ALIGNMENT_D");
  // Split-K partials are reduced in FP32 only: the in-place and output-type schemes add BF16-rounded partials.
  const uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_mask,
                                             sizeof(reduction_mask)),
        "REDUCTION_SCHEME_MASK");
  constexpr int kMaxResults = 8;
  cublasLtMatmulHeuristicResult_t results[kMaxResults] = {};
  int found = 0;
  Check(cublasLtMatmulAlgoGetHeuristic(Handle(s, device), p.desc, p.a, p.b, p.c, p.c, pref, kMaxResults, results,
                                       &found),
        "cublasLtMatmulAlgoGetHeuristic");
  cublasLtMatmulPreferenceDestroy(pref);
  if (found == 0) {
    throw std::runtime_error("gemm generic: cuBLASLt has no algorithm for M=" + std::to_string(m) +
                             " N=" + std::to_string(n) + " K=" + std::to_string(k));
  }
  p.algo = results[algo_index < found ? algo_index : found - 1].algo;
  return s.plans.emplace(key, p).first->second;
}

}  // namespace

int64_t GenericWorkspaceSize() { return kWorkspaceBytes; }

void GemmGeneric(TensorView out, TensorView a, TensorView w, Optional<TensorView> bias, Optional<TensorView> residual,
                 double alpha, double beta, TensorView workspace, bool w_kn) {
  using namespace xinfer::ffi;
  const char* op = "gemm_generic";
  const bool fp32 = IsFloat32(a.dtype());
  auto check_dtype = [&](const TensorView& t, const char* name) {
    if (fp32 ? !IsFloat32(t.dtype()) : !IsBFloat16(t.dtype())) {
      throw std::runtime_error(std::string("gemm_generic: ") + name + " must have the dtype of a (BF16 or FP32)");
    }
  };
  for (const TensorView* t : {&out, &a, &w}) {
    CheckCuda(*t, op);
    check_dtype(*t, "out/a/w");
    CheckContiguous(*t, op);
  }
  CheckSameDevice(a, out, "gemm_generic: a");
  CheckSameDevice(w, out, "gemm_generic: w");
  if (a.ndim() != 2 || w.ndim() != 2 || out.ndim() != 2) throw std::runtime_error("gemm_generic: rank-2 only");
  const int64_t m = a.size(0), k = a.size(1), n = w_kn ? w.size(1) : w.size(0);
  if ((w_kn ? w.size(0) : w.size(1)) != k || out.size(0) != m || out.size(1) != n) {
    throw std::runtime_error("gemm_generic: expected a[M,K], w[N,K] (or w[K,N] with w_kn), out[M,N]");
  }
  if (bias.has_value()) {
    check_dtype(bias.value(), "bias");
    CheckContiguous(bias.value(), "bias");
    CheckSameDevice(bias.value(), out, "gemm_generic: bias");
    if (NumElements(bias.value()) != n) throw std::runtime_error("gemm_generic: bias must have N elements");
  }
  if (residual.has_value()) {
    check_dtype(residual.value(), "residual");
    CheckContiguous(residual.value(), "residual");
    CheckSameShape(residual.value(), out, "residual");
    CheckSameDevice(residual.value(), out, "gemm_generic: residual");
  }
  CheckUInt8(workspace, "workspace");
  CheckSameDevice(workspace, out, "gemm_generic: workspace");
  if (NumBytes(workspace) < kWorkspaceBytes) {
    throw std::runtime_error("gemm_generic: workspace must hold " + std::to_string(kWorkspaceBytes) +
                             " bytes");
  }
  if (m == 0 || n == 0) return;

  const int device = out.device().device_id;
  cudaStream_t stream = Stream(out);
  const int64_t algo_index = 0;
  const void* c_ptr = residual.has_value() ? residual.value().data_ptr() : out.data_ptr();
  uint32_t align_c = Alignment(out.data_ptr());
  if (residual.has_value()) align_c = std::min(align_c, Alignment(c_ptr));
  const uint32_t align_bias = bias.has_value() ? Alignment(bias.value().data_ptr()) : 0;
  const PlanKey key{device, m, n, k, bias.has_value(), residual.has_value(), algo_index,
                    Alignment(w.data_ptr()), Alignment(a.data_ptr()), align_c, align_bias, fp32, w_kn};

  State& s = GetState();
  std::lock_guard<std::mutex> lock(s.mu);
  Plan& p = GetPlan(s, key, algo_index);
  if (bias.has_value()) {
    const void* bias_ptr = bias.value().data_ptr();
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)),
          "BIAS_POINTER");
  }
  const float alpha_f = static_cast<float>(alpha);
  const float beta_f = residual.has_value() ? static_cast<float>(beta) : 0.0f;
  Check(cublasLtMatmul(Handle(s, device), p.desc, &alpha_f, w.data_ptr(), p.a, a.data_ptr(), p.b, &beta_f, c_ptr,
                       p.c, out.data_ptr(), p.c, &p.algo, workspace.data_ptr(), kWorkspaceBytes, stream),
        "cublasLtMatmul");
}

namespace {

// Block-scaled plans: MXFP8 (E4M3 data, E8M0 1x32 SWIZZLE_32_4_4 scales = cuBLASLt VEC32_UE8M0) or NVFP4 (E2M1
// data, E4M3 1x16 SWIZZLE_32_4_4 scales = VEC16_UE4M3):
// (device, m, n, k, bias, fast_accum, algo index, alignments of w, a, out, nvfp4)
using MxPlanKey =
    std::tuple<int, int64_t, int64_t, int64_t, bool, bool, int64_t, uint32_t, uint32_t, uint32_t, bool>;

std::map<MxPlanKey, Plan>& MxPlans() {
  static std::map<MxPlanKey, Plan> plans;
  return plans;
}

Plan& GetMxPlan(State& s, const MxPlanKey& key, int64_t algo_index, const void* w_scale, const void* a_scale) {
  auto& plans = MxPlans();
  auto it = plans.find(key);
  if (it != plans.end()) return it->second;
  const int device = std::get<0>(key);
  const int64_t m = std::get<1>(key), n = std::get<2>(key), k = std::get<3>(key);
  const bool has_bias = std::get<4>(key), fast_accum = std::get<5>(key), nvfp4 = std::get<10>(key);
  Plan p;
  Check(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
  const cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "TRANSA");
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "TRANSB");
  const int32_t scale_mode =
      nvfp4 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 : CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)),
        "A_SCALE_MODE");
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)),
        "B_SCALE_MODE");
  // The heuristic needs the scale pointers set (they are re-set before every launch).
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &w_scale, sizeof(w_scale)),
        "A_SCALE_POINTER");
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale, sizeof(a_scale)),
        "B_SCALE_POINTER");
  if (fast_accum) {
    const int8_t fa = 1;
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof(fa)), "FAST_ACCUM");
  }
  if (has_bias) {
    const cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)), "EPILOGUE");
    const cudaDataType bias_type = CUDA_R_16BF;
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
                                         sizeof(bias_type)),
          "BIAS_DATA_TYPE");
  }
  const cudaDataType element = nvfp4 ? CUDA_R_4F_E2M1 : CUDA_R_8F_E4M3;
  Check(cublasLtMatrixLayoutCreate(&p.a, element, k, n, k), "layout w");
  Check(cublasLtMatrixLayoutCreate(&p.b, element, k, m, k), "layout a");
  Check(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_16BF, n, m, n), "layout out");
  cublasLtMatmulPreference_t pref = nullptr;
  Check(cublasLtMatmulPreferenceCreate(&pref), "cublasLtMatmulPreferenceCreate");
  const uint64_t ws = kWorkspaceBytes;
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)),
        "MAX_WORKSPACE_BYTES");
  const uint32_t align_w = std::get<7>(key), align_a = std::get<8>(key), align_c = std::get<9>(key);
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_w,
                                             sizeof(align_w)),
        "MIN_ALIGNMENT_A");
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_a,
                                             sizeof(align_a)),
        "MIN_ALIGNMENT_B");
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align_c,
                                             sizeof(align_c)),
        "MIN_ALIGNMENT_D");
  const uint32_t reduction_mask = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE;
  Check(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK, &reduction_mask,
                                             sizeof(reduction_mask)),
        "REDUCTION_SCHEME_MASK");
  constexpr int kMaxResults = 8;
  cublasLtMatmulHeuristicResult_t results[kMaxResults] = {};
  int found = 0;
  Check(cublasLtMatmulAlgoGetHeuristic(Handle(s, device), p.desc, p.a, p.b, p.c, p.c, pref, kMaxResults, results,
                                       &found),
        "cublasLtMatmulAlgoGetHeuristic (block-scaled)");
  cublasLtMatmulPreferenceDestroy(pref);
  if (found == 0) {
    throw std::runtime_error(std::string("gemm generic ") + (nvfp4 ? "NVFP4" : "MXFP8") +
                             ": cuBLASLt has no algorithm for M=" + std::to_string(m) +
                             " N=" + std::to_string(n) + " K=" + std::to_string(k));
  }
  p.algo = results[algo_index < found ? algo_index : found - 1].algo;
  return plans.emplace(key, p).first->second;
}

}  // namespace

// out[M, N] (BF16) = a[M, K] @ w[N, K]^T (+ bias) for MXFP8 a / w (E4M3 + E8M0 1x32 SWIZZLE_32_4_4 scales).
void GemmGenericMxfp8(TensorView out, TensorView a, TensorView a_scale, TensorView w, TensorView w_scale,
                      Optional<TensorView> bias, TensorView workspace, bool fast_accum) {
  using namespace xinfer::ffi;
  CheckCuda(out, "out");
  CheckBFloat16(out, "out");
  CheckContiguous(out, "out");
  for (const TensorView* t : {&a, &w}) {
    CheckCuda(*t, "a/w");
    CheckFloat8E4M3Storage(*t, "a/w");
    CheckContiguous(*t, "a/w");
  }
  CheckFloat8E8M0Storage(a_scale, "a_scale");
  CheckFloat8E8M0Storage(w_scale, "w_scale");
  for (const TensorView* t : {&a, &a_scale, &w, &w_scale, &workspace}) {
    CheckSameDevice(*t, out, "gemm_generic_mxfp8: a/a_scale/w/w_scale/workspace");
  }
  if (bias.has_value()) CheckSameDevice(bias.value(), out, "gemm_generic_mxfp8: bias");
  if (a.ndim() != 2 || w.ndim() != 2 || out.ndim() != 2) {
    throw std::runtime_error("gemm_generic_mxfp8: rank-2 only");
  }
  const int64_t m = a.size(0), k = a.size(1), n = w.size(0);
  if (w.size(1) != k || out.size(0) != m || out.size(1) != n || k % 32) {
    throw std::runtime_error("gemm_generic_mxfp8: expected a[M,K], w[N,K], out[M,N], K % 32 == 0");
  }
  auto storage = [](int64_t rows, int64_t cols) { return ((rows + 127) / 128) * 128 * (((cols / 32) + 3) / 4) * 4; };
  if (NumBytes(a_scale) < storage(m, k) || NumBytes(w_scale) < storage(n, k)) {
    throw std::runtime_error("gemm_generic_mxfp8: scale storage too small (SWIZZLE_32_4_4)");
  }
  if (bias.has_value()) {
    CheckBFloat16(bias.value(), "bias");
    if (NumElements(bias.value()) != n) throw std::runtime_error("gemm_generic_mxfp8: bias must have N elements");
  }
  CheckUInt8(workspace, "workspace");
  if (NumBytes(workspace) < kWorkspaceBytes) {
    throw std::runtime_error("gemm_generic_mxfp8: workspace must hold " + std::to_string(kWorkspaceBytes) + " bytes");
  }
  if (m == 0 || n == 0) return;
  const int device = out.device().device_id;
  cudaStream_t stream = Stream(out);
  const int64_t algo_index = 0;
  const MxPlanKey key{device, m, n, k, bias.has_value(), fast_accum, algo_index,
                      Alignment(w.data_ptr()), Alignment(a.data_ptr()), Alignment(out.data_ptr()), false};
  State& s = GetState();
  std::lock_guard<std::mutex> lock(s.mu);
  Plan& p = GetMxPlan(s, key, algo_index, w_scale.data_ptr(), a_scale.data_ptr());
  // A' = weight, B' = activation (column-major trick, see GetPlan).
  const void* sa = w_scale.data_ptr();
  const void* sb = a_scale.data_ptr();
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)),
        "A_SCALE_POINTER");
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)),
        "B_SCALE_POINTER");
  if (bias.has_value()) {
    const void* bias_ptr = bias.value().data_ptr();
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)),
          "BIAS_POINTER");
  }
  const float alpha_f = 1.0f, beta_f = 0.0f;
  Check(cublasLtMatmul(Handle(s, device), p.desc, &alpha_f, w.data_ptr(), p.a, a.data_ptr(), p.b, &beta_f,
                       out.data_ptr(), p.c, out.data_ptr(), p.c, &p.algo, workspace.data_ptr(), kWorkspaceBytes,
                       stream),
        "cublasLtMatmul (MXFP8)");
}

// out[M, N] (BF16) = alpha * (a[M, K] @ w[N, K]^T) (+ bias) for NVFP4 a / w: packed E2M1 uint8 [rows, K / 2] (column
// 2j in the low nibble) with E4M3 1x16 SWIZZLE_32_4_4 scales. alpha (the weight's global scale) and the bias apply to
// the FP32 accumulator, which is rounded once.
void GemmGenericNvfp4(TensorView out, TensorView a, TensorView a_scale, TensorView w, TensorView w_scale,
                      Optional<TensorView> bias, TensorView workspace, double alpha) {
  using namespace xinfer::ffi;
  CheckCuda(out, "out");
  CheckBFloat16(out, "out");
  CheckContiguous(out, "out");
  for (const TensorView* t : {&a, &w}) {
    CheckCuda(*t, "a/w");
    CheckUInt8(*t, "a/w");
    CheckContiguous(*t, "a/w");
  }
  CheckFloat8E4M3Storage(a_scale, "a_scale");
  CheckFloat8E4M3Storage(w_scale, "w_scale");
  for (const TensorView* t : {&a, &a_scale, &w, &w_scale, &workspace}) {
    CheckSameDevice(*t, out, "gemm_generic_nvfp4: a/a_scale/w/w_scale/workspace");
  }
  if (bias.has_value()) CheckSameDevice(bias.value(), out, "gemm_generic_nvfp4: bias");
  if (a.ndim() != 2 || w.ndim() != 2 || out.ndim() != 2) {
    throw std::runtime_error("gemm_generic_nvfp4: rank-2 only");
  }
  const int64_t m = a.size(0), k = 2 * a.size(1), n = w.size(0);
  if (2 * w.size(1) != k || out.size(0) != m || out.size(1) != n || k % 32) {
    throw std::runtime_error("gemm_generic_nvfp4: expected a[M, K/2], w[N, K/2], out[M, N], K % 32 == 0");
  }
  auto storage = [](int64_t rows, int64_t cols) { return ((rows + 127) / 128) * 128 * (((cols / 16) + 3) / 4) * 4; };
  if (NumBytes(a_scale) < storage(m, k) || NumBytes(w_scale) < storage(n, k)) {
    throw std::runtime_error("gemm_generic_nvfp4: scale storage too small (SWIZZLE_32_4_4)");
  }
  if (bias.has_value()) {
    CheckBFloat16(bias.value(), "bias");
    if (NumElements(bias.value()) != n) throw std::runtime_error("gemm_generic_nvfp4: bias must have N elements");
  }
  const float alpha_f = static_cast<float>(alpha);
  if (!(alpha_f > 0.0f) || !std::isfinite(alpha_f)) {
    throw std::runtime_error("gemm_generic_nvfp4: alpha must be finite and positive in FP32");
  }
  CheckUInt8(workspace, "workspace");
  if (NumBytes(workspace) < kWorkspaceBytes) {
    throw std::runtime_error("gemm_generic_nvfp4: workspace must hold " + std::to_string(kWorkspaceBytes) + " bytes");
  }
  if (m == 0 || n == 0) return;
  const int device = out.device().device_id;
  cudaStream_t stream = Stream(out);
  const int64_t algo_index = 0;
  const MxPlanKey key{device, m, n, k, bias.has_value(), false, algo_index,
                      Alignment(w.data_ptr()), Alignment(a.data_ptr()), Alignment(out.data_ptr()), true};
  State& s = GetState();
  std::lock_guard<std::mutex> lock(s.mu);
  Plan& p = GetMxPlan(s, key, algo_index, w_scale.data_ptr(), a_scale.data_ptr());
  // A' = weight, B' = activation (column-major trick, see GetPlan).
  const void* sa = w_scale.data_ptr();
  const void* sb = a_scale.data_ptr();
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)),
        "A_SCALE_POINTER");
  Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)),
        "B_SCALE_POINTER");
  if (bias.has_value()) {
    const void* bias_ptr = bias.value().data_ptr();
    Check(cublasLtMatmulDescSetAttribute(p.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(bias_ptr)),
          "BIAS_POINTER");
  }
  const float beta_f = 0.0f;
  Check(cublasLtMatmul(Handle(s, device), p.desc, &alpha_f, w.data_ptr(), p.a, a.data_ptr(), p.b, &beta_f,
                       out.data_ptr(), p.c, out.data_ptr(), p.c, &p.algo, workspace.data_ptr(), kWorkspaceBytes,
                       stream),
        "cublasLtMatmul (NVFP4)");
}

}  // namespace xinfer::gemm

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_generic_mxfp8, xinfer::gemm::GemmGenericMxfp8);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_generic_workspace_size, xinfer::gemm::GenericWorkspaceSize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_generic, xinfer::gemm::GemmGeneric);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gemm_generic_nvfp4, xinfer::gemm::GemmGenericNvfp4);
