// FFI of denoise_output_euler. Direct callers get the full contract check.
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "denoise_output_euler.cuh"

namespace xinfer::denoise_output {
namespace {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;
namespace ffi = xinfer::ffi;

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error("denoise_output_euler: " + what); }

void CheckBf16OrFp32(const TensorView& t, const char* name) {
  if (!ffi::IsBFloat16(t.dtype()) && !ffi::IsFloat32(t.dtype())) Fail(std::string(name) + " must be BF16 or FP32");
}

void CheckActionTensor(const TensorView& t, const TensorView& ref, int64_t b, int64_t m, int64_t a, const char* name) {
  ffi::CheckSameDevice(t, ref, name);
  ffi::CheckFloat32(t, name);
  ffi::CheckContiguous(t, name);
  if (t.ndim() != 3 || t.size(0) != b || t.size(1) != m || t.size(2) != a) {
    Fail(std::string(name) + " must be FP32 [" + std::to_string(b) + ", " + std::to_string(m) + ", " +
         std::to_string(a) + "]");
  }
}

void CheckVector(const TensorView& t, const TensorView& ref, int64_t n, const char* name) {
  ffi::CheckSameDevice(t, ref, name);
  ffi::CheckContiguous(t, name);
  if (t.ndim() != 1 || t.size(0) != n) Fail(std::string(name) + " must be [" + std::to_string(n) + "]");
}

}  // namespace

// out / x [B, M, A] FP32 (out may be x); h [B, M, K]; weight [A, K]; prenorm 0 none / 1 rmsnorm / 2 adarms
// (norm_weight = rmsnorm weight or adarms scale; shift = adarms shift); dt_tensor [B] FP32 overrides dt.
void denoise_output_euler(TensorView out, TensorView x, TensorView h, TensorView weight, Optional<TensorView> bias,
                          int64_t prenorm, Optional<TensorView> norm_weight, Optional<TensorView> shift, double eps,
                          Optional<TensorView> row_mask, Optional<TensorView> dof_mask,
                          Optional<TensorView> padding_velocity, Optional<TensorView> pinned,
                          Optional<TensorView> pinned_values, double dt, Optional<TensorView> dt_tensor, bool pdl) {
  ffi::CheckCuda(h, "h");
  CheckBf16OrFp32(h, "h");
  CheckBf16OrFp32(weight, "weight");
  if (h.ndim() != 3) Fail("h must be [B, M, K]");
  const int64_t b = h.size(0), m = h.size(1), k = h.size(2);
  if (weight.ndim() != 2 || weight.size(1) != k) Fail("weight must be [A, K] with K of h");
  const int64_t a = weight.size(0);
  if (k != 1024) Fail("K must be 1024, got " + std::to_string(k));
  if (a < 1 || a > 32) Fail("A must be in [1, 32], got " + std::to_string(a));
  if (b < 1 || m < 1 || b * m > (1 << 20)) Fail("B and M must be positive");
  ffi::CheckSameDevice(weight, h, "weight");
  ffi::CheckContiguous(weight, "weight");
  if (h.stride(2) != 1 || h.stride(1) % 4 || h.stride(0) % 4 ||
      reinterpret_cast<uintptr_t>(h.data_ptr()) % (ffi::IsBFloat16(h.dtype()) ? 8 : 16)) {
    Fail("h needs a contiguous last dimension, row and batch strides that are multiples of 4 elements and an aligned base");
  }
  if (reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16) Fail("weight must be 16-byte aligned");
  if ((b - 1) * h.stride(0) + (m - 1) * h.stride(1) + k >= (int64_t{1} << 31)) Fail("h spans 2^31 elements or more");
  CheckActionTensor(x, h, b, m, a, "x");
  CheckActionTensor(out, h, b, m, a, "out");
  ffi::CheckExactAliasOrNoOverlap(out, "out", x, "x");
  ffi::CheckNoOverlap(out, "out", h, "h");
  ffi::CheckNoOverlap(out, "out", weight, "weight");

  Params p{};
  p.h = h.data_ptr();
  p.h_sb = h.stride(0);
  p.h_st = h.stride(1);
  p.w = weight.data_ptr();
  p.B = static_cast<int>(b);
  p.M = static_cast<int>(m);
  p.A = static_cast<int>(a);
  p.K = static_cast<int>(k);
  p.x = ffi::ConstData<float>(x);
  p.out = ffi::Data<float>(out);
  p.eps = static_cast<float>(eps);
  p.prenorm = static_cast<int>(prenorm);
  p.dt = static_cast<float>(dt);
  if (prenorm < 0 || prenorm > 2) Fail("prenorm must be 0 (none), 1 (rmsnorm) or 2 (adarms)");
  if (!(eps > 0.0) || !std::isfinite(eps)) Fail("eps must be positive");
  if (bias.has_value()) {
    const TensorView t = bias.value();
    CheckVector(t, h, a, "bias");
    if (t.dtype().code != weight.dtype().code || t.dtype().bits != weight.dtype().bits) Fail("bias must have the dtype of weight");
    ffi::CheckNoOverlap(out, "out", t, "bias");
    p.bias = t.data_ptr();
  }
  if (prenorm != kNone) {
    if (!norm_weight.has_value()) Fail(prenorm == kAdaRms ? "adarms needs ada_scale" : "rmsnorm needs norm_weight");
    const TensorView f = norm_weight.value();
    CheckVector(f, h, k, prenorm == kAdaRms ? "ada_scale" : "norm_weight");
    CheckBf16OrFp32(f, "norm_weight");
    if (reinterpret_cast<uintptr_t>(f.data_ptr()) % 16) Fail("norm_weight / ada_scale must be 16-byte aligned");
    p.mod_bf16 = ffi::IsBFloat16(f.dtype());
    p.norm_w = f.data_ptr();
    if (prenorm == kAdaRms) {
      if (!shift.has_value()) Fail("adarms needs ada_shift");
      const TensorView s = shift.value();
      CheckVector(s, h, k, "ada_shift");
      if (s.dtype().code != f.dtype().code || s.dtype().bits != f.dtype().bits) Fail("ada_shift must have the dtype of ada_scale");
      if (reinterpret_cast<uintptr_t>(s.data_ptr()) % 16) Fail("ada_shift must be 16-byte aligned");
      p.shift = s.data_ptr();
    } else if (shift.has_value()) {
      Fail("ada_shift is only used with adarms");
    }
  } else if (norm_weight.has_value() || shift.has_value()) {
    Fail("norm_weight / ada_scale / ada_shift need a prenorm");
  }
  if (row_mask.has_value()) {
    const TensorView t = row_mask.value();
    ffi::CheckSameDevice(t, h, "row_mask");
    ffi::CheckUInt8(t, "row_mask");
    if (t.ndim() != 2 || t.size(0) != b || t.size(1) != m) Fail("row_mask must be [B, M]");
    p.row_mask = ffi::ConstData<uint8_t>(t);
    p.rm_sb = t.stride(0);
    p.rm_st = t.stride(1);
  }
  if (dof_mask.has_value()) {
    const TensorView t = dof_mask.value();
    ffi::CheckSameDevice(t, h, "dof_mask");
    if (!ffi::IsFloat32(t.dtype()) && !ffi::IsUInt8(t.dtype())) Fail("dof_mask must be FP32 or uint8 / bool");
    if (t.stride(t.ndim() - 1) != 1) Fail("dof_mask needs a contiguous last dimension");
    if (t.ndim() == 3 && t.size(0) == b && t.size(1) == m && t.size(2) == a) {
      p.dof_sb = t.stride(0);
      p.dof_st = t.stride(1);
    } else if (t.ndim() == 2 && t.size(0) == b && t.size(1) == a) {
      p.dof_sb = t.stride(0);
      p.dof_st = 0;
    } else {
      Fail("dof_mask must be [B, M, A] or [B, A]");
    }
    if (!padding_velocity.has_value()) Fail("dof_mask needs padding_velocity");
    CheckActionTensor(padding_velocity.value(), h, b, m, a, "padding_velocity");
    ffi::CheckNoOverlap(out, "out", padding_velocity.value(), "padding_velocity");
    ffi::CheckNoOverlap(out, "out", t, "dof_mask");
    p.dof = t.data_ptr();
    p.dof_f32 = ffi::IsFloat32(t.dtype());
    p.vpad = ffi::ConstData<float>(padding_velocity.value());
  } else if (padding_velocity.has_value()) {
    Fail("padding_velocity is only used with dof_mask");
  }
  if (pinned.has_value()) {
    const TensorView t = pinned.value();
    CheckVector(t, h, b, "pinned");
    ffi::CheckInt32(t, "pinned");
    p.pinned = ffi::ConstData<int>(t);
    if (pinned_values.has_value()) {
      CheckActionTensor(pinned_values.value(), h, b, m, a, "pinned_values");
      ffi::CheckNoOverlap(out, "out", pinned_values.value(), "pinned_values");
      p.pinned_values = ffi::ConstData<float>(pinned_values.value());
    }
  } else if (pinned_values.has_value()) {
    Fail("pinned_values needs pinned");
  }
  if (dt_tensor.has_value()) {
    const TensorView t = dt_tensor.value();
    CheckVector(t, h, b, "dt");
    ffi::CheckFloat32(t, "dt");
    p.dt_t = ffi::ConstData<float>(t);
  } else if (!std::isfinite(dt)) {
    Fail("dt must be finite");
  }
  const cudaError_t e = Launch(p, ffi::IsBFloat16(h.dtype()), ffi::IsBFloat16(weight.dtype()), pdl, ffi::Stream(h));
  if (e != cudaSuccess) Fail(std::string("launch failed: ") + cudaGetErrorString(e));
}

}  // namespace xinfer::denoise_output

TVM_FFI_DLL_EXPORT_TYPED_FUNC(denoise_output_euler, xinfer::denoise_output::denoise_output_euler);
