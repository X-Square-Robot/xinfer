// The unified GDN vendor-chunk bridge: the cached-plan LaunchCore composition
// over aclnnChunkGatedDeltaRule (ported from the v22 bridge; the TensorView
// wrappers are dropped -- the glue binds raw pointers). Sizing uses the same
// GetWorkspaceSize primitive the plan cache warms.
#include "ffi.h"

#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_chunk_gated_delta_rule.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <string>

namespace {
using ::xinfer::aot::ffi::Require;

std::string& GdrError() {
  thread_local std::string buffer;
  return buffer;
}

aclTensor* Shape(void* pointer, std::initializer_list<int64_t> shape, aclDataType dtype) {
  std::array<int64_t, 5> dims{};
  const auto rank = static_cast<int64_t>(shape.size());
  int axis = 0;
  for (const auto dim : shape) dims[axis++] = dim;
  return aclCreateTensor(dims.data(), rank, dtype, nullptr, 0, ACL_FORMAT_ND, dims.data(), rank,
                         reinterpret_cast<void*>(pointer));
}

uint64_t SizeWorkspace(void* q, void* k, void* v, void* beta, void* g, void* initial_state,
                       void* final_state, void* out, void* lengths, int64_t t, int64_t nk,
                       int64_t nv, float scale) {
  aclTensor* descriptors[9] = {
      Shape(q, {t, nk, 128}, ACL_BF16),
      Shape(k, {t, nk, 128}, ACL_BF16),
      Shape(v, {t, nv, 128}, ACL_BF16),
      Shape(beta, {t, nv}, ACL_BF16),
      g ? Shape(g, {t, nv}, ACL_FLOAT) : nullptr,
      Shape(initial_state, {1, nv, 128, 128}, ACL_BF16),
      Shape(final_state, {1, nv, 128, 128}, ACL_BF16),
      Shape(out, {t, nv, 128}, ACL_BF16),
      Shape(lengths, {1}, ACL_INT32),
  };
  uint64_t need = 0;
  aclOpExecutor* executor = nullptr;
  // API: (q, k, v, beta, initialState, actualSeqLengths, gOptional, scale,
  //      out, finalState, &workspaceSize, &executor)
  const auto status = aclnnChunkGatedDeltaRuleGetWorkspaceSize(
      descriptors[0], descriptors[1], descriptors[2], descriptors[3],
      descriptors[5], descriptors[8], descriptors[4], scale,
      descriptors[7], descriptors[6], &need, &executor);
  for (auto* tensor : descriptors)
    if (tensor) aclDestroyTensor(tensor);
  if (executor) aclDestroyAclOpExecutor(executor);
  if (status != 0)
    throw std::runtime_error("aclnnChunkGatedDeltaRuleGetWorkspaceSize failed: " + std::to_string(status));
  return need;
}
}  // namespace

// Raw-pointer launcher used by the production gated_delta_rule binding: the
// public call routes its fused TND epilogue kernels around this primitive.
namespace xinfer::gdr_cann_route {
namespace {

aclTensor* DescribeRaw(uintptr_t address, std::initializer_list<int64_t> shape, aclDataType dtype) {
  std::array<int64_t, 5> dims{};
  const auto rank = static_cast<int64_t>(shape.size());
  int axis = 0;
  for (const auto dim : shape) dims[axis++] = dim;
  return aclCreateTensor(dims.data(), rank, dtype, nullptr, 0, ACL_FORMAT_ND, dims.data(), rank,
                         reinterpret_cast<void*>(address));
}

// Eager callers rebuild the aclnn execution plan on every call, which costs
// ~100 us of host time; hot NPUGraph replay never re-enters this bridge. Cache
// the repeatable executor for the most recent (shape, scale, buffer addresses)
// configuration: graph capture is oblivious (it snapshots device commands),
// and any address change misses the single entry and rebuilds exactly as
// before. The nine aclTensor descriptors stay alive next to the executor -
// destroying an executor whose descriptors were freed crashes inside
// ~KernelGraph (aclStorage::GetExtend). Set GDR_CANN_EXECUTOR_CACHE=0 to
// disable (A/B measurements).
struct CachedPlan {
  int64_t t = -1, nk = -1, nv = -1;
  float scale = 0.0f;
  void* pointers[9] = {};
  uint64_t need = 0;
  aclOpExecutor* executor = nullptr;
  std::array<aclTensor*, 9> descriptors{};
};

std::mutex& PlanMutex() {
  static std::mutex mutex;
  return mutex;
}

CachedPlan& PlanCache() {
  static CachedPlan cache;
  return cache;
}

bool ExecutorCacheEnabled() {
  static const bool enabled = [] {
    const char* option = std::getenv("GDR_CANN_EXECUTOR_CACHE");
    return option == nullptr || std::strcmp(option, "0") != 0;
  }();
  return enabled;
}
}  // namespace

bool Enabled() { return true; }

void LaunchCore(void* q, void* k, void* v, void* beta, void* g, void* initial_state,
                void* final_state, void* out, void* lengths, int64_t t, int64_t nk, int64_t nv,
                float scale, void* stream, unsigned char* workspace, int64_t workspace_bytes) {
  void* const pointers[9] = {q, k, v, beta, g, initial_state, final_state, out, lengths};
  const std::lock_guard<std::mutex> guard(PlanMutex());
  CachedPlan& cache = PlanCache();
  const float scale_f32 = scale;
  bool hit = ExecutorCacheEnabled() && cache.executor != nullptr && cache.t == t &&
             cache.nk == nk && cache.nv == nv && cache.scale == scale_f32 &&
             std::equal(std::begin(cache.pointers), std::end(cache.pointers), std::begin(pointers));
  if (!hit) {
    std::array<aclTensor*, 9> raw{};
    raw[0] = DescribeRaw(reinterpret_cast<uintptr_t>(q), {t, nk, 128}, ACL_BF16);
    raw[1] = DescribeRaw(reinterpret_cast<uintptr_t>(k), {t, nk, 128}, ACL_BF16);
    raw[2] = DescribeRaw(reinterpret_cast<uintptr_t>(v), {t, nv, 128}, ACL_BF16);
    raw[3] = DescribeRaw(reinterpret_cast<uintptr_t>(beta), {t, nv}, ACL_BF16);
    raw[4] = DescribeRaw(reinterpret_cast<uintptr_t>(initial_state), {1, nv, 128, 128}, ACL_BF16);
    raw[5] = DescribeRaw(reinterpret_cast<uintptr_t>(lengths), {1}, ACL_INT32);
    raw[6] = DescribeRaw(reinterpret_cast<uintptr_t>(g), {t, nv}, ACL_FLOAT);
    raw[7] = DescribeRaw(reinterpret_cast<uintptr_t>(out), {t, nv, 128}, ACL_BF16);
    raw[8] = DescribeRaw(reinterpret_cast<uintptr_t>(final_state), {1, nv, 128, 128}, ACL_BF16);
    uint64_t need = 0;
    aclOpExecutor* executor = nullptr;
    const aclnnStatus prepared = aclnnChunkGatedDeltaRuleGetWorkspaceSize(
        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], scale_f32, raw[7], raw[8], &need,
        &executor);
    if (prepared != 0) {
      for (auto* tensor : raw) {
        if (tensor) aclDestroyTensor(tensor);
      }
      throw std::runtime_error("aclnnChunkGatedDeltaRuleGetWorkspaceSize failed: " + std::to_string(prepared));
    }
    if (executor == nullptr) {
      for (auto* tensor : raw) {
        if (tensor) aclDestroyTensor(tensor);
      }
      throw std::runtime_error("CANN chunk GDR returned no executor");
    }
    if (static_cast<uint64_t>(workspace_bytes) < need) {
      aclDestroyAclOpExecutor(executor);
      for (auto* tensor : raw) {
        if (tensor) aclDestroyTensor(tensor);
      }
      throw std::runtime_error("gated_delta_rule_prefill: CANN chunk core needs " + std::to_string(need) +
                               " workspace bytes");
    }
    aclSetAclOpExecutorRepeatable(executor);
    if (!ExecutorCacheEnabled()) {
      // Legacy single-shot flow: the descriptors outlive the launch only.
      for (auto* tensor : raw) {
        if (tensor) aclDestroyTensor(tensor);
      }
      const aclnnStatus single = aclnnChunkGatedDeltaRule(reinterpret_cast<void*>(workspace), need, executor, stream);
      aclDestroyAclOpExecutor(executor);
      if (single != 0) {
        throw std::runtime_error("aclnnChunkGatedDeltaRule failed: " + std::to_string(single));
      }
      return;
    }
    if (cache.executor) {
      aclDestroyAclOpExecutor(cache.executor);
      for (auto* tensor : cache.descriptors) {
        if (tensor) aclDestroyTensor(tensor);
      }
    }
    cache.t = t;
    cache.nk = nk;
    cache.nv = nv;
    cache.scale = scale_f32;
    std::copy(std::begin(pointers), std::end(pointers), std::begin(cache.pointers));
    cache.need = need;
    cache.executor = executor;
    cache.descriptors = raw;
  }
  if (static_cast<uint64_t>(workspace_bytes) < cache.need) {
    throw std::runtime_error("gated_delta_rule_prefill: CANN chunk core needs " + std::to_string(cache.need) +
                             " workspace bytes");
  }
  const aclnnStatus launched =
      aclnnChunkGatedDeltaRule(reinterpret_cast<void*>(workspace), cache.need, cache.executor, stream);
  if (launched != 0) {
    throw std::runtime_error("aclnnChunkGatedDeltaRule failed: " + std::to_string(launched));
  }
}
}  // namespace xinfer::gdr_cann_route

// ---------------------------------------------------------------------------
// Raw C-ABI wrappers (xinfer_ names; the glue binds these through ctypes).
extern "C" const char* xinfer_gdr_cann_chunk_get_last_error() {
  return GdrError().c_str();
}

extern "C" int64_t xinfer_gdr_cann_chunk_workspace_size(
    int32_t device, void* q, void* k, void* v, void* beta, void* g, void* initial_state,
    void* final_state, void* out, void* lengths, int64_t t, int64_t nk, int64_t nv, float scale) {
  (void)device;
  try {
    return static_cast<int64_t>(SizeWorkspace(q, k, v, beta, g, initial_state, final_state,
                                              out, lengths, t, nk, nv, scale));
  } catch (const std::exception& error) {
    GdrError() = std::string("gdr chunk workspace: ") + error.what();
  } catch (...) {
    GdrError() = "gdr chunk workspace: unknown failure";
  }
  return -1;
}

extern "C" int xinfer_gdr_cann_chunk_launch(
    int32_t device, void* q, void* k, void* v, void* beta, void* g, void* initial_state,
    void* final_state, void* out, void* lengths, int64_t t, int64_t nk, int64_t nv, float scale,
    void* stream, void* workspace, int64_t workspace_bytes) {
  try {
    xinfer::gdr_cann_route::LaunchCore(q, k, v, beta, g, initial_state, final_state, out,
                                       lengths, t, nk, nv, scale, stream,
                                       static_cast<unsigned char*>(workspace), workspace_bytes);
    return 0;
  } catch (const std::exception& error) {
    GdrError() = std::string("gdr chunk launch: ") + error.what();
  } catch (...) {
    GdrError() = "gdr chunk launch: unknown failure";
  }
  return 1;
}
