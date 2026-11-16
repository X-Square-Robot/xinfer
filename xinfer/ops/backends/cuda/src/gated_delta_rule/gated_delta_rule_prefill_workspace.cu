// Workspace layout of gated_delta_rule_prefill (shared by both backends; see gated_delta_rule_prefill.cuh).
#include "gated_delta_rule_prefill.cuh"

namespace xinfer::gated_delta_rule_prefill {

int ChunkLength(int backend) { return backend == kChunk64 ? 64 : 32; }

namespace {
constexpr size_t kAlign = 256;
size_t Up(size_t x) { return (x + kAlign - 1) / kAlign * kAlign; }
}  // namespace

size_t WorkspaceBytes(int backend, int T, int HK, int HV) {
  const size_t bt = ChunkLength(backend), nt = (T + bt - 1) / bt, tp = nt * bt, d = kHeadDim;
  return 2 * Up(HK * tp * d * 2) + 2 * Up(HV * tp * d * 2) + Up(HV * tp * 4) + Up(HV * nt * d * d * 2) + Up(HV * d * tp * 2);
}

Workspace CarveWorkspace(void* base, int backend, int T, int HK, int HV) {
  const size_t bt = ChunkLength(backend), nt = (T + bt - 1) / bt, tp = nt * bt, d = kHeadDim;
  auto* b = static_cast<uint8_t*>(base);
  Workspace w{};
  auto take = [&](size_t bytes) {
    uint8_t* r = b;
    b += Up(bytes);
    return r;
  };
  w.qh = reinterpret_cast<__nv_bfloat16*>(take(HK * tp * d * 2));
  w.kh = reinterpret_cast<__nv_bfloat16*>(take(HK * tp * d * 2));
  w.w = reinterpret_cast<__nv_bfloat16*>(take(HV * tp * d * 2));
  w.u = reinterpret_cast<__nv_bfloat16*>(take(HV * tp * d * 2));
  w.gc = reinterpret_cast<float*>(take(HV * tp * 4));
  w.hst = reinterpret_cast<__nv_bfloat16*>(take(HV * nt * d * d * 2));
  w.vnT = reinterpret_cast<__nv_bfloat16*>(take(HV * d * tp * 2));
  w.Tp = static_cast<int>(tp);
  w.NT = static_cast<int>(nt);
  return w;
}

}  // namespace xinfer::gated_delta_rule_prefill
