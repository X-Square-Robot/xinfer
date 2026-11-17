// The unified ACLNN Mm bridge: the raw C-ABI pair over aclnnMm (BF16 or FP32
// rows against an NK weight into an FP32 out), ported from the v22 bridge with
// the TensorView halves and the in-library BNSD routes dropped (those AOT
// kernels are served by the glue over their own libraries now).
#include "ffi.h"

#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_mm.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace xinfer::cann_mm_fp32 {
namespace {
using ::xinfer::aot::ffi::Require;

struct TensorDeleter {
  void operator()(aclTensor* ptr) const noexcept { aclDestroyTensor(ptr); }
};
struct ExecutorDeleter {
  void operator()(aclOpExecutor* ptr) const noexcept { aclDestroyAclOpExecutor(ptr); }
};
using Tensor = std::unique_ptr<aclTensor, TensorDeleter>;
using Executor = std::unique_ptr<aclOpExecutor, ExecutorDeleter>;

void Check(aclnnStatus status, const char* api) {
  if (status != 0) throw std::runtime_error(std::string(api) + " failed: " + std::to_string(status));
}

struct Range {
  uintptr_t begin;
  uintptr_t end;
  const char* name;
};

struct MatrixView {
  std::array<int64_t, 2> shape;
  aclDataType dtype;
  Range memory;
};

Tensor Describe(const MatrixView& view, bool transpose = false) {
  const auto& storage = view.shape;
  const std::array<int64_t, 2> shape = transpose ? std::array<int64_t, 2>{storage[1], storage[0]} : storage;
  const std::array<int64_t, 2> strides = transpose ? std::array<int64_t, 2>{1, storage[1]}
                                                  : std::array<int64_t, 2>{storage[1], 1};
  Tensor tensor(aclCreateTensor(shape.data(), 2, view.dtype, strides.data(), 0, ACL_FORMAT_ND,
                                storage.data(), 2, reinterpret_cast<void*>(view.memory.begin)));
  Require(tensor != nullptr, std::string("aclCreateTensor failed for ") + view.memory.name);
  return tensor;
}

struct Call {
  Tensor input, weight, output;
  Executor executor;
  uint64_t bytes = 0;
};

struct RawMm {
  Tensor input, weight, output;
  Executor executor;
  uint64_t bytes = 0;
};

RawMm PrepareRaw(int32_t device, aclDataType dtype, const int64_t* in_dims,
                 const int64_t* w_dims, const int64_t* out_dims, void* in, void* w, void* out,
                 bool weight_kn) {
  const MatrixView view_in{{in_dims[0], in_dims[1]}, dtype, {reinterpret_cast<uintptr_t>(in),
      reinterpret_cast<uintptr_t>(in) + in_dims[0] * in_dims[1] * (dtype == ACL_BF16 ? 2 : 4), "raw input"}};
  const auto w_elements = w_dims[0] * w_dims[1];
  const MatrixView view_w_nk{{w_dims[0], w_dims[1]}, dtype, {reinterpret_cast<uintptr_t>(w),
      reinterpret_cast<uintptr_t>(w) + w_elements * (dtype == ACL_BF16 ? 2 : 4), "raw weight"}};
  // weight_kn: the buffer is row-major [K, N]; describe it as the natural B of
  // out = A @ B without a physical transpose. Otherwise B is the transpose of
  // the row-major [N, K] weight.
  const MatrixView view_w_kn{{w_dims[1], w_dims[0]}, dtype, {reinterpret_cast<uintptr_t>(w),
      reinterpret_cast<uintptr_t>(w) + w_elements * (dtype == ACL_BF16 ? 2 : 4), "raw weight"}};
  const MatrixView view_out{{out_dims[0], out_dims[1]}, ACL_FLOAT, {reinterpret_cast<uintptr_t>(out),
      reinterpret_cast<uintptr_t>(out) + out_dims[0] * out_dims[1] * 4, "raw out"}};
  RawMm call{Describe(view_in), weight_kn ? Describe(view_w_kn) : Describe(view_w_nk, true),
             Describe(view_out), nullptr, 0};
  aclOpExecutor* executor = nullptr;
  Check(aclnnMmGetWorkspaceSize(call.input.get(), call.weight.get(), call.output.get(),
                               static_cast<int8_t>(0), &call.bytes, &executor),
        "aclnnMmGetWorkspaceSize");
  Require(executor != nullptr, "aclnnMm did not return an executor");
  Check(aclSetAclOpExecutorRepeatable(executor), "aclSetAclOpExecutorRepeatable");
  call.executor.reset(executor);
  return call;
}
}  // namespace
int64_t RawWorkspaceBytes(int32_t device, bool bf16, const int64_t* in_dims,
                          const int64_t* w_dims, const int64_t* out_dims, void* in, void* w, void* out,
                          bool weight_kn) {
  (void)device;
  auto call = PrepareRaw(device, bf16 ? ACL_BF16 : ACL_FLOAT, in_dims, w_dims, out_dims, in, w, out, weight_kn);
  Require(call.bytes <= INT64_MAX, "workspace size overflow");
  return static_cast<int64_t>(call.bytes);
}

void RawLaunch(int32_t device, bool bf16, const int64_t* in_dims, const int64_t* w_dims,
               const int64_t* out_dims, void* in, void* w, void* out, void* workspace,
               int64_t workspace_bytes, bool weight_kn) {
  auto call = PrepareRaw(device, bf16 ? ACL_BF16 : ACL_FLOAT, in_dims, w_dims, out_dims, in, w, out, weight_kn);
  Require(call.bytes <= static_cast<uint64_t>(workspace_bytes), "workspace is too small for ACLNN Mm");
  auto* stream = xinfer::aot::ffi::CurrentStream(device);
  auto ptr = call.bytes ? reinterpret_cast<uintptr_t>(workspace) : 0;
  Check(aclnnMm(reinterpret_cast<void*>(ptr), call.bytes, call.executor.get(), stream), "aclnnMm");
}


}  // namespace xinfer::cann_mm_fp32

// Raw C-ABI wrappers (xinfer_ names; the glue binds these through ctypes).
namespace {
std::string& MmError() {
  thread_local std::string buffer;
  return buffer;
}
}  // namespace

extern "C" const char* xinfer_cann_mm_fp32_get_last_error() {
  return MmError().c_str();
}

extern "C" int64_t xinfer_cann_mm_fp32_workspace_bytes(
    int32_t device, int64_t bf16, const int64_t* in_dims, const int64_t* w_dims,
    const int64_t* out_dims, void* in, void* w, void* out, int64_t weight_kn) {
  try {
    return xinfer::cann_mm_fp32::RawWorkspaceBytes(device, bf16 != 0, in_dims, w_dims, out_dims,
                                                   in, w, out, weight_kn != 0);
  } catch (const std::exception& error) {
    MmError() = std::string("cann_mm_fp32 workspace: ") + error.what();
  } catch (...) {
    MmError() = "cann_mm_fp32 workspace: unknown failure";
  }
  return -1;
}

extern "C" int xinfer_cann_mm_fp32_launch(
    int32_t device, int64_t bf16, const int64_t* in_dims, const int64_t* w_dims,
    const int64_t* out_dims, void* in, void* w, void* out, void* workspace,
    int64_t workspace_bytes, int64_t weight_kn) {
  try {
    xinfer::cann_mm_fp32::RawLaunch(device, bf16 != 0, in_dims, w_dims, out_dims, in, w, out,
                                    workspace, workspace_bytes, weight_kn != 0);
    return 0;
  } catch (const std::exception& error) {
    MmError() = std::string("cann_mm_fp32 launch: ") + error.what();
  } catch (...) {
    MmError() = "cann_mm_fp32 launch: unknown failure";
  }
  return 1;
}
