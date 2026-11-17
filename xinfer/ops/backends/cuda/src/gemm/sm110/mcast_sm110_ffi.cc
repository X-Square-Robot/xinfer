// Entries of the SM110 weight-multicast BF16 GEMMs (mcast_bf16_sm110, mcast_ffn_sm110): persistent tcgen05 2-SM
// kernels, one 6-CTA cluster (3 CTA pairs sharing each weight tile by TMA multicast) for 512 < M <= 768, CTA pairs
// otherwise. Hidden 2048:
//   gemm x += a @ W^T in place, BF16 residual rounded once    (K 2048, or the intermediate size such as 11008 /
//        6144, -> N 2048; the down projection reads W row-major or K-blocked: [N/512][K/64][512][64] keeps each
//        64-column weight box contiguous in DRAM)
//   gemm out = act(row_scale * (a @ W^T) + bias), stored (any N a multiple of 8, K a multiple of 64)
//   glu  h = act(x @ Wg^T) * (x @ Wu^T), act GELU-tanh or SiLU, gate/up rows in blocks of 128
//        (K 2048 -> N = intermediate, a multiple of 256)
//   mlp  x += glu(x_n) @ Wd^T in one persistent kernel (GELU-tanh or SiLU, an intermediate in blocks of 256 that
//        splits into 2-8 equal chunks): the intermediate goes through an L2-resident ring of FFN chunks, three
//        co-resident 6-CTA clusters (18 SMs) synchronized by global flags, 512 < M <= 768; the down projection adds
//        BF16 partials into x by TMA reduce-add (order not fixed). Weights K-blocked.
// The kernels run with their default configuration: plain launches (no PDL), 5 smem stages for the cluster kernel
// (512 < M <= 768) and 4 otherwise, the split-K partials of the down projection added in arrival order, the fused
// MLP's tensor pipe paced at 1500 SM clocks per k-block MMA group.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/optional.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "common/tvm_ffi_cuda_tensor.h"
#include "gemm/sm110/mcast_bf16_sm110.h"
#include "gemm/sm110/mcast_ffn_sm110.h"

namespace xinfer::gemm::mcast_sm110 {
namespace {

using tvm::ffi::Optional;
using tvm::ffi::TensorView;

constexpr int kFusedMlpPace = 1500;

void CheckBf16(const TensorView& t, const char* name, int64_t rows, int64_t cols) {
  using namespace xinfer::ffi;
  CheckCuda(t, name);
  CheckBFloat16(t, name);
  CheckContiguous(t, name);
  if (NumElements(t) != rows * cols) {
    throw std::runtime_error(std::string("mcast_bf16_sm110: ") + name + " must be [" + std::to_string(rows) + ", " +
                             std::to_string(cols) + "]");
  }
}

int Rows(const TensorView& x, int64_t k) {
  xinfer::ffi::RequireArchitecture(x, 11, 0, "mcast_bf16_sm110");
  if (x.ndim() != 2 || x.size(1) != k || x.size(0) < 1) {
    throw std::runtime_error("mcast_bf16_sm110: x must be [M, " + std::to_string(k) + "]");
  }
  return static_cast<int>(x.size(0));
}

void Check(int rc, const char* what) {
  if (rc != 0) throw std::runtime_error(std::string("mcast_bf16_sm110 ") + what + " failed (" + std::to_string(rc) + ")");
}

void SetPdl() { mcast_bf16_sm110_set_config("pdl", 0); }

int Stages(int m) { return m > 512 && m <= 768 ? 5 : 4; }

}  // namespace

// K 2048: the output projection; otherwise the down projection, K the intermediate size (a multiple of 64, at most
// MCAST_MAX_INTER). w_kblk: w is K-blocked; down projection only.
void GemmResidual(TensorView out, TensorView a, TensorView w, Optional<TensorView> ws, bool w_kblk) {
  using namespace xinfer::ffi;
  const int64_t k = a.ndim() == 2 ? a.size(1) : 0;
  if (k <= 0 || k % 64 != 0 || k > MCAST_MAX_INTER) {
    throw std::runtime_error("mcast_bf16_sm110: gemm K must be a multiple of 64, at most 16384");
  }
  if (w_kblk && k == 2048) throw std::runtime_error("mcast_bf16_sm110: a K-blocked weight needs the down projection");
  const int m = Rows(a, k);
  CheckBf16(a, "a", m, k);
  CheckBf16(w, "w", 2048, k);
  CheckBf16(out, "out", m, 2048);
  CheckNoOverlap(out, "out", a, "a");
  void* wsp = nullptr;
  if (ws.has_value()) {
    CheckCuda(ws.value(), "ws");
    if (NumElements(ws.value()) < static_cast<int64_t>(mcast_bf16_sm110_workspace_bytes(m))) {
      throw std::runtime_error("mcast_bf16_sm110: workspace too small");
    }
    wsp = ws.value().data_ptr();
  }
  SetPdl();
  if (k == 2048) {
    mcast_bf16_sm110_set_config("p4_stages", Stages(m));
    Check(mcast_bf16_sm110_o_proj(ConstData<__nv_bfloat16>(a), ConstData<__nv_bfloat16>(w), Data<__nv_bfloat16>(out),
                                  m, wsp, Stream(a)),
          "gemm");
  } else {
    mcast_bf16_sm110_set_config("p7_exact", 0);
    Check(mcast_bf16_sm110_down_ex(ConstData<__nv_bfloat16>(a), ConstData<__nv_bfloat16>(w), Data<__nv_bfloat16>(out),
                                   m, static_cast<int>(k), wsp, w_kblk ? 1 : 0, Stream(a)),
          "gemm");
  }
}

// out = act(row_scale * (a @ w^T) + bias): act MCAST_OUT_ACT_NONE (-1) or MCAST_OUT_ACT_GELU_TANH (1), bias [N] BF16,
// row_scale [M] FP32, both optional; w [N, K] row-major, N a multiple of 8, K a multiple of 64.
void GemmOut(TensorView out, TensorView a, TensorView w, Optional<TensorView> bias, Optional<TensorView> row_scale,
             int64_t act) {
  using namespace xinfer::ffi;
  const int64_t k = a.ndim() == 2 ? a.size(1) : 0, n = w.ndim() == 2 ? w.size(0) : 0;
  if (k <= 0 || k % 64 != 0 || n <= 0 || n % 8 != 0 || (act != MCAST_OUT_ACT_NONE && act != MCAST_OUT_ACT_GELU_TANH)) {
    throw std::runtime_error("mcast_bf16_sm110: gemm out needs K a multiple of 64, N of 8, act -1 (none) or 1 (GELU-tanh)");
  }
  const int m = Rows(a, k);
  CheckBf16(a, "a", m, k);
  CheckBf16(w, "w", n, k);
  CheckBf16(out, "out", m, n);
  CheckNoOverlap(out, "out", a, "a");
  const __nv_bfloat16* b = nullptr;
  if (bias.has_value()) {
    CheckBf16(bias.value(), "bias", 1, n);
    b = ConstData<__nv_bfloat16>(bias.value());
  }
  const float* rs = nullptr;
  if (row_scale.has_value()) {
    CheckCuda(row_scale.value(), "row_scale");
    CheckFloat32(row_scale.value(), "row_scale");
    CheckContiguous(row_scale.value(), "row_scale");
    if (NumElements(row_scale.value()) != m) throw std::runtime_error("mcast_bf16_sm110: row_scale must have M entries");
    rs = ConstData<float>(row_scale.value());
  }
  SetPdl();
  Check(mcast_bf16_sm110_gemm_out(ConstData<__nv_bfloat16>(a), ConstData<__nv_bfloat16>(w), Data<__nv_bfloat16>(out),
                                  b, rs, static_cast<int>(act), m, static_cast<int>(n), static_cast<int>(k), Stream(a)),
        "gemm out");
}

// act: MCAST_ACT_SILU or MCAST_ACT_GELU_TANH; the intermediate size is h's width (a multiple of 256).
void Glu(TensorView h, TensorView x, TensorView w, int64_t act) {
  using namespace xinfer::ffi;
  const int m = Rows(x, 2048);
  const int64_t inter = h.ndim() == 2 ? h.size(1) : 0;
  if (inter <= 0 || inter % 256 != 0 || inter > MCAST_MAX_INTER || (act != MCAST_ACT_SILU && act != MCAST_ACT_GELU_TANH)) {
    throw std::runtime_error("mcast_glu_sm110: h must be [M, I] with I a multiple of 256, act SiLU (0) or GELU-tanh (1)");
  }
  CheckBf16(x, "x", m, 2048);
  CheckBf16(h, "h", m, inter);
  CheckBf16(w, "w", 2 * inter, 2048);
  CheckNoOverlap(h, "h", x, "x");
  SetPdl();
  Check(mcast_bf16_sm110_gate_up(ConstData<__nv_bfloat16>(x), ConstData<__nv_bfloat16>(w), Data<__nv_bfloat16>(h), m,
                                 static_cast<int>(inter), static_cast<int>(act), Stream(x)),
        "glu");
}

// act: MCAST_ACT_SILU or MCAST_ACT_GELU_TANH; the intermediate size is w_down's width (I / 256 blocks, at most 64,
// with a chunk count in [2, 8] dividing them: 16384, 11264, ...).
void FusedMlp(TensorView out, TensorView x, TensorView w_gate_up, TensorView w_down, TensorView hbuf, TensorView flags,
              int64_t act) {
  using namespace xinfer::ffi;
  const int m = Rows(x, 2048);
  if (m <= 512 || m > 768) throw std::runtime_error("mcast_bf16_sm110: fused_mlp needs 512 < M <= 768");
  const int64_t inter = w_down.ndim() == 2 ? w_down.size(1) : 0;
  if (inter <= 0 || inter > 16384 || mcast_ffn_sm110_hbuf_bytes_ex(m, static_cast<int>(inter)) == 0 ||
      (act != MCAST_ACT_SILU && act != MCAST_ACT_GELU_TANH)) {
    throw std::runtime_error("mcast_bf16_sm110: fused_mlp takes an intermediate of up to 64 blocks of 256 split into "
                             "2-8 equal chunks, and act SiLU (0) or GELU-tanh (1)");
  }
  CheckBf16(x, "x", m, 2048);
  CheckBf16(out, "out", m, 2048);
  CheckBf16(w_gate_up, "w_gate_up", 2 * inter, 2048);
  CheckBf16(w_down, "w_down", 2048, inter);
  CheckNoOverlap(out, "out", x, "x");
  CheckCuda(hbuf, "hbuf");
  CheckCuda(flags, "flags");
  if (NumElements(hbuf) < static_cast<int64_t>(mcast_ffn_sm110_hbuf_bytes_ex(m, static_cast<int>(inter))) ||
      NumElements(flags) * 4 < static_cast<int64_t>(mcast_ffn_sm110_flags_bytes())) {
    throw std::runtime_error("mcast_bf16_sm110: fused_mlp buffers too small");
  }
  mcast_ffn_sm110_set_config("pace", kFusedMlpPace);
  mcast_ffn_sm110_set_pace_ptr(nullptr);
  Check(mcast_ffn_sm110_ex(x.data_ptr(), w_gate_up.data_ptr(), w_down.data_ptr(), out.data_ptr(), hbuf.data_ptr(),
                           static_cast<int*>(flags.data_ptr()), m, static_cast<int>(inter), static_cast<int>(act),
                           Stream(x)),
        "fused_mlp");
}

int64_t FusedMlpHbufBytes(int64_t m, int64_t inter) {
  return static_cast<int64_t>(mcast_ffn_sm110_hbuf_bytes_ex(static_cast<int>(m), static_cast<int>(inter)));
}
int64_t FusedMlpFlagsBytes() { return static_cast<int64_t>(mcast_ffn_sm110_flags_bytes()); }

int64_t WorkspaceBytes(int64_t m) { return static_cast<int64_t>(mcast_bf16_sm110_workspace_bytes(static_cast<int>(m))); }

}  // namespace xinfer::gemm::mcast_sm110

TVM_FFI_DLL_EXPORT_TYPED_FUNC(mcast_gemm_residual_sm110, xinfer::gemm::mcast_sm110::GemmResidual);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mcast_gemm_out_sm110, xinfer::gemm::mcast_sm110::GemmOut);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mcast_glu_sm110, xinfer::gemm::mcast_sm110::Glu);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mcast_fused_mlp_sm110, xinfer::gemm::mcast_sm110::FusedMlp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mcast_fused_mlp_sm110_hbuf_bytes, xinfer::gemm::mcast_sm110::FusedMlpHbufBytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mcast_fused_mlp_sm110_flags_bytes, xinfer::gemm::mcast_sm110::FusedMlpFlagsBytes);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(mcast_sm110_workspace_bytes, xinfer::gemm::mcast_sm110::WorkspaceBytes);
