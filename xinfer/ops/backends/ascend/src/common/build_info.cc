// Build information of xinfer_kernels_ascend.so.

#include <tvm/ffi/function.h>

#include <string>

namespace xinfer::build_info {

// Comma-separated SoCs this library was compiled for (e.g. "ascend950dt9582"). The SoC string is the vendor's
// device name normalized to lowercase alphanumerics, the same string xinfer/platforms/npu.py derives from
// torch_npu's device properties and reports as the platform's arch, so a registered kernel matches its platform
// exactly. Kernels exist only for these SoCs; the Python loader (../_native.py) registers nothing for a library
// that reports none.
std::string BuildTargetSocs() { return XINFER_TARGET_SOCS; }

}  // namespace xinfer::build_info

TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_target_archs, xinfer::build_info::BuildTargetSocs);
