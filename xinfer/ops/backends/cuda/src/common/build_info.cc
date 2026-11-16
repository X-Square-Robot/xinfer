// Build information of xinfer_kernels.so.

#include <tvm/ffi/function.h>

#include <string>

namespace xinfer::build_info {

// Comma-separated compute capabilities this library was compiled for (e.g. "110" or "110,120"). Kernels exist
// only for these architectures; launching on another GPU fails with the message of
// xinfer::ffi::CheckCudaError.
std::string BuildTargetArchs() { return XINFER_TARGET_ARCHS; }

}  // namespace xinfer::build_info

TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_target_archs, xinfer::build_info::BuildTargetArchs);
