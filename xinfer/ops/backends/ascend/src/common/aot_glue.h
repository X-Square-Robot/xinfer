// The marshalling helpers of the per-kernel TVM-FFI faces (the aot_glue.cc files the
// generator writes beside each family's kernels): resolve the current NPU stream from
// the first tensor's device and drive one packed-launch C ABI call, raising the
// launcher's own error text on failure.
//
// The faces marshal only: the python envelopes that call them own the validation
// (shapes, dtypes, envelopes), exactly as the ctypes bindings these faces replace had
// no checks of their own either - the envelope logic is unchanged, only the FFI
// boundary moved from ctypes to TVM-FFI.
#pragma once

#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "ffi.h"

namespace xinfer::aot::glue {

using tvm::ffi::TensorView;

namespace details {
// A marshalled argument: a tensor becomes its device pointer, a scalar passes through.
inline void* ToC(const TensorView& tensor) { return tensor.data_ptr(); }
inline int64_t ToC(int64_t scalar) { return scalar; }
}  // namespace details

// The stream of the device the tensors of this launch live on, resolved per call: NPUGraph
// capture records the node (the first-call warmup rule the envelopes enforce stays theirs).
inline void* Stream(const TensorView& tensor) {
  return xinfer::aot::ffi::CurrentStream(static_cast<int32_t>(tensor.device().device_id));
}

// One packed-launch call: ``call(ToC(args)..., Stream(first))``, raising the kernel's own
// error text on a nonzero status. ``first`` is the launch's first tensor (every kernel's
// first parameter is a tensor pointer; the stream follows its device). ``call`` and
// ``error_fn`` are the kernel's ``xinfer_aot_<name>_{call,get_last_error}``, declared in
// each face as the packed-launch contract - void* per tensor, int64 per scalar, the
// stream last - which is the ABI the ctypes bindings always drove and the aarch64 C ABI
// passes type-agnostically.
template <typename Call, typename Error, typename... Args>
inline void Launch(const char* what, Call call, Error error_fn, const TensorView& first,
                   Args... args) {
  void* stream = Stream(first);
  int32_t status = call(details::ToC(args)..., stream);
  if (status != 0) {
    const char* message = error_fn();
    throw std::runtime_error(std::string(what) + ": " +
                             (message != nullptr ? message : "unknown failure"));
  }
}

}  // namespace xinfer::aot::glue
