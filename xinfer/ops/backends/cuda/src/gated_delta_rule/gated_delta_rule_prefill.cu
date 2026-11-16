// FFI of gated_delta_rule_prefill. Direct callers get the full contract check; the shapes it is used on are
// listed in ../attn.py.
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "gated_delta_rule_prefill.cuh"

namespace xinfer::gated_delta_rule_prefill {

namespace {

constexpr size_t kAlign = 256;
using tvm::ffi::Optional;
using tvm::ffi::TensorView;
namespace ffi = xinfer::ffi;

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error("gated_delta_rule_prefill: " + what); }

void CheckBackendShape(int64_t backend, int64_t t, int64_t hk, int64_t hv) {
  if (backend != kChunk32) Fail("backend must be 0 (chunk32), got " + std::to_string(backend));
  if (t < 1 || t > kMaxSeq) Fail("sequence length must be in [1, " + std::to_string(kMaxSeq) + "], got " + std::to_string(t));
  if (hk < 1 || hk > 64) Fail("num_k_heads must be in [1, 64], got " + std::to_string(hk));
  if (hv != hk && hv != 2 * hk) Fail("num_v_heads must equal num_k_heads or twice it, got H_k=" + std::to_string(hk) + " H_v=" + std::to_string(hv));
}

void CheckRows(const TensorView& t, const TensorView& ref, int64_t rows, int64_t cols, int64_t align_elems, const char* name) {
  ffi::CheckSameDevice(t, ref, name);
  ffi::CheckBFloat16(t, name);
  if (t.ndim() != 2 || t.size(0) != rows || t.size(1) != cols) {
    Fail(std::string(name) + " must be BF16 [" + std::to_string(rows) + ", " + std::to_string(cols) + "]");
  }
  if (t.stride(1) != 1 || t.stride(0) < cols || t.stride(0) % align_elems ||
      reinterpret_cast<uintptr_t>(t.data_ptr()) % (align_elems * 2)) {
    Fail(std::string(name) + " needs a contiguous last dimension, a row stride that is a multiple of " + std::to_string(align_elems) +
         " elements and a " + std::to_string(align_elems * 2) + "-byte aligned base");
  }
  if ((rows - 1) * t.stride(0) + cols >= (int64_t{1} << 31)) Fail(std::string(name) + " spans 2^31 elements or more");
}

void CheckVector(const TensorView& t, const TensorView& ref, int64_t n, bool bf16, const char* name) {
  ffi::CheckSameDevice(t, ref, name);
  if (bf16) {
    ffi::CheckBFloat16(t, name);
  } else {
    ffi::CheckFloat32(t, name);
  }
  ffi::CheckContiguous(t, name);
  if (t.ndim() != 1 || t.size(0) != n) Fail(std::string(name) + " must be [" + std::to_string(n) + "]");
  if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16) Fail(std::string(name) + " must be 16-byte aligned");
}

}  // namespace

// out [T, H_v*128] BF16; workspace uint8 [>= workspace_size]; qkv [T, (2 H_k + H_v)*128] BF16; zba [T, H_v*130] BF16 ([z | b | a]);
// conv_weight [(2 H_k + H_v)*128, 4] or [.., 1, 4] BF16; A_log, dt_bias [H_v] FP32; norm_weight [128] BF16; conv_bias [(2 H_k + H_v)*128] BF16.
void gated_delta_rule_prefill(TensorView out, TensorView workspace, TensorView qkv, TensorView zba, TensorView conv_weight, TensorView A_log,
                              TensorView dt_bias, TensorView norm_weight, Optional<TensorView> conv_bias, int64_t num_k_heads, int64_t backend,
                              double eps, bool pdl) {
  ffi::CheckCuda(qkv, "qkv");
  {
    int major = 0;
    ffi::CheckCudaError(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, qkv.device().device_id), "cudaDeviceGetAttribute");
    if (major < 9) Fail("needs compute capability 9.0 or newer (griddepcontrol, mbarrier.try_wait)");
  }
  if (qkv.ndim() != 2 || zba.ndim() != 2) Fail("qkv and zba must be 2-D [T, columns]");
  const int64_t t = qkv.size(0), hk = num_k_heads, d = kHeadDim;
  if (zba.size(1) % (d + 2)) Fail("zba must be [T, H_v*130] ([z | beta logits | decay logits])");
  const int64_t hv = zba.size(1) / (d + 2), c = (2 * hk + hv) * d;
  CheckBackendShape(backend, t, hk, hv);
  CheckRows(qkv, qkv, t, c, 8, "qkv");
  CheckRows(zba, qkv, t, hv * (d + 2), 2, "zba");
  CheckRows(out, qkv, t, hv * d, 8, "out");
  ffi::CheckSameDevice(conv_weight, qkv, "conv_weight");
  ffi::CheckBFloat16(conv_weight, "conv_weight");
  ffi::CheckContiguous(conv_weight, "conv_weight");
  const bool cw2 = conv_weight.ndim() == 2 && conv_weight.size(0) == c && conv_weight.size(1) == 4;
  const bool cw3 = conv_weight.ndim() == 3 && conv_weight.size(0) == c && conv_weight.size(1) == 1 && conv_weight.size(2) == 4;
  if (!cw2 && !cw3) Fail("conv_weight must be [" + std::to_string(c) + ", 4] or [" + std::to_string(c) + ", 1, 4]");
  if (reinterpret_cast<uintptr_t>(conv_weight.data_ptr()) % 16) Fail("conv_weight must be 16-byte aligned");
  CheckVector(A_log, qkv, hv, false, "A_log");
  CheckVector(dt_bias, qkv, hv, false, "dt_bias");
  CheckVector(norm_weight, qkv, d, true, "norm_weight");
  if (!(eps > 0.0) || !std::isfinite(eps)) Fail("eps must be positive and finite");
  ffi::CheckSameDevice(workspace, qkv, "workspace");
  ffi::CheckUInt8(workspace, "workspace");
  ffi::CheckContiguous(workspace, "workspace");
  const size_t need = WorkspaceBytes(static_cast<int>(backend), static_cast<int>(t), static_cast<int>(hk), static_cast<int>(hv));
  if (workspace.ndim() != 1 || static_cast<size_t>(workspace.size(0)) < need) {
    Fail("workspace must be uint8 [>= " + std::to_string(need) + "] (gated_delta_rule_prefill_workspace_size)");
  }
  if (reinterpret_cast<uintptr_t>(workspace.data_ptr()) % kAlign) Fail("workspace must be 256-byte aligned");
  const __nv_bfloat16* cb = nullptr;
  if (conv_bias.has_value()) {
    CheckVector(conv_bias.value(), qkv, c, true, "conv_bias");
    cb = ffi::ConstData<__nv_bfloat16>(conv_bias.value());
  }
  ffi::CheckNoOverlap(out, "out", qkv, "qkv");
  ffi::CheckNoOverlap(out, "out", zba, "zba");
  ffi::CheckNoOverlap(out, "out", workspace, "workspace");
  ffi::CheckNoOverlap(workspace, "workspace", qkv, "qkv");
  ffi::CheckNoOverlap(workspace, "workspace", zba, "zba");
  ffi::CheckNoOverlap(workspace, "workspace", conv_weight, "conv_weight");
  ffi::CheckNoOverlap(workspace, "workspace", norm_weight, "norm_weight");

  Params p{};
  p.qkv = ffi::ConstData<__nv_bfloat16>(qkv);
  p.zba = ffi::ConstData<__nv_bfloat16>(zba);
  p.conv_w = ffi::ConstData<__nv_bfloat16>(conv_weight);
  p.conv_b = cb;
  p.A_log = ffi::ConstData<float>(A_log);
  p.dt_bias = ffi::ConstData<float>(dt_bias);
  p.norm_w = ffi::ConstData<__nv_bfloat16>(norm_weight);
  p.out = ffi::Data<__nv_bfloat16>(out);
  p.workspace = workspace.data_ptr();
  p.ldq = qkv.stride(0);
  p.ldz = zba.stride(0);
  p.ldo = out.stride(0);
  p.T = static_cast<int>(t);
  p.HK = static_cast<int>(hk);
  p.HV = static_cast<int>(hv);
  p.eps = static_cast<float>(eps);
  p.scale = 0.08838834764831845f;  // 128^-0.5
  p.pdl = pdl;
  p.preconv = false;
  const cudaStream_t stream = ffi::Stream(qkv);
  const cudaError_t e = LaunchChunk32(p, stream);
  if (e != cudaSuccess) Fail(std::string("launch failed: ") + cudaGetErrorString(e));
}

int64_t gated_delta_rule_prefill_workspace_size(int64_t backend, int64_t seq_len, int64_t num_k_heads, int64_t num_v_heads) {
  CheckBackendShape(backend, seq_len, num_k_heads, num_v_heads);
  return static_cast<int64_t>(WorkspaceBytes(static_cast<int>(backend), static_cast<int>(seq_len), static_cast<int>(num_k_heads),
                                             static_cast<int>(num_v_heads)));
}

}  // namespace xinfer::gated_delta_rule_prefill

TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_delta_rule_prefill, xinfer::gated_delta_rule_prefill::gated_delta_rule_prefill);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(gated_delta_rule_prefill_workspace_size, xinfer::gated_delta_rule_prefill::gated_delta_rule_prefill_workspace_size);
