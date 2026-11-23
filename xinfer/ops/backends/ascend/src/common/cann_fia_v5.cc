// C ABI for aclnnFusedInferAttentionScoreV5 with BNSD inputs or packed TND vision inputs.
#include "ffi.h"

#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v5.h>

#include <array>
#include <cmath>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace xinfer::cann_fia_v5 {
namespace {
using ::xinfer::aot::ffi::Require;

struct AclDelete {
  void operator()(aclTensor* value) const noexcept { aclDestroyTensor(value); }
  void operator()(aclTensorList* value) const noexcept { aclDestroyTensorList(value); }
  void operator()(aclIntArray* value) const noexcept { aclDestroyIntArray(value); }
  void operator()(aclOpExecutor* value) const noexcept { aclDestroyAclOpExecutor(value); }
};
using Tensor = std::unique_ptr<aclTensor, AclDelete>;
using TensorList = std::unique_ptr<aclTensorList, AclDelete>;
using IntArray = std::unique_ptr<aclIntArray, AclDelete>;
using Executor = std::unique_ptr<aclOpExecutor, AclDelete>;

void Check(aclnnStatus status, const char* api) {
  if (status != 0) throw std::runtime_error(std::string(api) + " failed: " + std::to_string(status));
}

struct Call {
  std::vector<int64_t> endpoints;
  IntArray q_ends;
  IntArray kv_ends;
  Tensor q;
  TensorList k;
  TensorList v;
  Tensor mask;
  Tensor out;
  Executor executor;
  uint64_t bytes = 0;
};

void BuildExecutor(Call& call, char* layout, int64_t num_heads, int64_t num_kv_heads,
                   double scale, int64_t inner_precise, int64_t sparse_mode = 0,
                   int64_t q_depth = 0) {
  const auto scale_f32 = static_cast<float>(scale);
  Require(std::isfinite(scale_f32) && scale_f32 > 0, "FIA requires a positive finite FP32 scale");
  Require(inner_precise == 0 || inner_precise == 2, "FIA accepts inner_precise 0 or 2");
  Require(sparse_mode == 0 || sparse_mode == 1, "FIA accepts sparse_mode 0 or 1");
  Require(sparse_mode == 0 || call.mask.get() == nullptr, "FIA sparse_mode takes no dense mask");
  constexpr int64_t window = INT32_MAX;
  const int64_t pre_tokens = sparse_mode == 1 ? q_depth : window;
  const int64_t next_tokens = sparse_mode == 1 ? 0 : window;
  aclOpExecutor* executor = nullptr;
  Check(aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
            call.q.get(), call.k.get(), call.v.get(),
            /*pseShift=*/nullptr, /*attenMask=*/call.mask.get(),
            /*actualSeqLengths=*/call.q_ends.get(), /*actualSeqLengthsKv=*/call.kv_ends.get(),
            /*deqScale1=*/nullptr, /*quantScale1=*/nullptr, /*deqScale2=*/nullptr,
            /*quantScale2=*/nullptr, /*quantOffset2=*/nullptr,
            /*antiquantScale=*/nullptr, /*antiquantOffset=*/nullptr, /*blockTable=*/nullptr,
            /*queryPaddingSize=*/nullptr, /*kvPaddingSize=*/nullptr,
            /*keyAntiquantScale=*/nullptr, /*keyAntiquantOffset=*/nullptr,
            /*valueAntiquantScale=*/nullptr, /*valueAntiquantOffset=*/nullptr,
            /*keySharedPrefix=*/nullptr, /*valueSharedPrefix=*/nullptr, /*actualSharedPrefixLen=*/nullptr,
            /*queryRope=*/nullptr, /*keyRope=*/nullptr, /*keyRopeAntiquantScale=*/nullptr,
            /*dequantScaleQuery=*/nullptr, /*learnableSink=*/nullptr,
            /*qStartIdx=*/nullptr, /*kvStartIdx=*/nullptr,
            /*numHeads=*/num_heads, /*scaleValue=*/scale,
            /*preTokens=*/pre_tokens, /*nextTokens=*/next_tokens, /*inputLayout=*/layout,
            /*numKeyValueHeads=*/num_kv_heads, /*sparseMode=*/sparse_mode, /*innerPrecise=*/inner_precise,
            /*blockSize=*/0, /*antiquantMode=*/0, /*softmaxLseFlag=*/false,
            /*keyAntiquantMode=*/0, /*valueAntiquantMode=*/0, /*queryQuantMode=*/0, /*pseType=*/0,
            /*attentionOut=*/call.out.get(), /*softmaxLse=*/nullptr, &call.bytes, &executor),
        "aclnnFusedInferAttentionScoreV5GetWorkspaceSize");
  Require(executor != nullptr, "FIA did not return an executor");
  Check(aclSetAclOpExecutorRepeatable(executor), "aclSetAclOpExecutorRepeatable");
  call.executor.reset(executor);
}

// Raw-pointer Describe variants for the absorbed bridge route: the python
// caller has already validated shapes, dtypes and contiguity.

// Raw-pointer Describe variants for the absorbed bridge route: the python
// caller has already validated shapes, dtypes and contiguity.
Tensor DescribeRaw(void* data, const int64_t* shape, int64_t rank, aclDataType type) {
  Tensor tensor(aclCreateTensor(shape, rank, type, nullptr, 0, ACL_FORMAT_ND, shape, rank, data));
  Require(tensor != nullptr, "aclCreateTensor failed for a raw FIA operand");
  return tensor;
}

TensorList DescribeSingletonRaw(void* data, const int64_t* shape, int64_t rank, aclDataType type) {
  auto tensor = DescribeRaw(data, shape, rank, type);
  const aclTensor* element = tensor.get();
  TensorList list(aclCreateTensorList(&element, 1));
  Require(list != nullptr, "aclCreateTensorList failed for a raw FIA operand");
  // aclDestroyTensorList destroys the member descriptor as well.
  tensor.release();
  return list;
}


Call PrepareRawBnsd(void* out, void* q, void* k, void* v, void* drop, const int64_t* dims,
                    double scale, int64_t inner_precise, bool bsnd_output) {
  const int64_t bnsd[4] = {dims[0], dims[1], dims[2], dims[3]};
  const int64_t bsnd[4] = {dims[0], dims[2], dims[1], dims[3]};
  const int64_t kv[4] = {dims[0], dims[4], dims[5], dims[3]};
  const int64_t mask[4] = {dims[0], 1, dims[2], dims[5]};
  Call call;
  call.q = DescribeRaw(q, bnsd, 4, ACL_BF16);
  call.k = DescribeSingletonRaw(k, kv, 4, ACL_BF16);
  call.v = DescribeSingletonRaw(v, kv, 4, ACL_BF16);
  call.mask = DescribeRaw(drop, mask, 4, ACL_BOOL);
  call.out = DescribeRaw(out, bsnd_output ? bsnd : bnsd, 4, ACL_BF16);
  char bnsd_layout[] = "BNSD";
  char bsnd_output_layout[] = "BNSD_BSND";
  BuildExecutor(call, bsnd_output ? bsnd_output_layout : bnsd_layout, dims[1], dims[4], scale, inner_precise);
  return call;
}

// Raw TND composition for the absorbed bridge route: dims = [tokens, heads,
// dim] over dense BF16 q/k/v/out; endpoints are the positive cumulative
// segment endpoints omitting the leading zero.
Call PrepareRawTnd(void* out, void* q, void* k, void* v, const int64_t* endpoints,
                   int64_t endpoint_count, const int64_t* dims, double scale) {
  int64_t previous = 0;
  for (int64_t i = 0; i < endpoint_count; ++i) {
    Require(endpoints[i] > previous, "TND endpoints must be strictly increasing and omit the leading zero");
    previous = endpoints[i];
  }
  Require(previous == dims[0], "TND endpoints must end at the packed token count");
  Call call;
  call.endpoints.assign(endpoints, endpoints + endpoint_count);
  call.q_ends.reset(aclCreateIntArray(call.endpoints.data(), call.endpoints.size()));
  call.kv_ends.reset(aclCreateIntArray(call.endpoints.data(), call.endpoints.size()));
  Require(call.q_ends && call.kv_ends, "aclCreateIntArray failed for vision TND endpoints");
  call.q = DescribeRaw(q, dims, 3, ACL_BF16);
  call.k = DescribeSingletonRaw(k, dims, 3, ACL_BF16);
  call.v = DescribeSingletonRaw(v, dims, 3, ACL_BF16);
  call.out = DescribeRaw(out, dims, 3, ACL_BF16);
  char tnd_layout[] = "TND";
  BuildExecutor(call, tnd_layout, dims[1], dims[1], scale, 0);
  return call;
}

void EnqueueRaw(const Call& call, void* workspace, int64_t workspace_bytes, int32_t device) {
  Require(call.bytes <= static_cast<uint64_t>(workspace_bytes), "workspace is too small for FIA");
  auto* stream = xinfer::aot::ffi::CurrentStream(device);
  Check(aclnnFusedInferAttentionScoreV5(workspace, call.bytes, call.executor.get(), stream),
        "aclnnFusedInferAttentionScoreV5");
}
}  // namespace
int64_t WorkspaceBytes(const Call& call) {
  Require(call.bytes <= INT64_MAX, "FIA workspace size overflows int64");
  return static_cast<int64_t>(call.bytes);
}

int64_t RawBnsdWorkspaceBytes(void* out, void* q, void* k, void* v, void* drop, const int64_t* dims,
                              double scale, int64_t inner_precise, bool bsnd_output) {
  return WorkspaceBytes(PrepareRawBnsd(out, q, k, v, drop, dims, scale, inner_precise, bsnd_output));
}

void RawBnsdLaunch(int32_t device, void* out, void* workspace, int64_t workspace_bytes, void* q,
                   void* k, void* v, void* drop, const int64_t* dims, double scale,
                   int64_t inner_precise, bool bsnd_output) {
  auto call = PrepareRawBnsd(out, q, k, v, drop, dims, scale, inner_precise, bsnd_output);
  EnqueueRaw(call, workspace, workspace_bytes, device);
}

int64_t RawTndWorkspaceBytes(void* out, void* q, void* k, void* v, const int64_t* endpoints,
                             int64_t endpoint_count, const int64_t* dims, double scale) {
  return WorkspaceBytes(PrepareRawTnd(out, q, k, v, endpoints, endpoint_count, dims, scale));
}

void RawTndLaunch(int32_t device, void* out, void* workspace, int64_t workspace_bytes, void* q,
                  void* k, void* v, const int64_t* endpoints, int64_t endpoint_count,
                  const int64_t* dims, double scale) {
  auto call = PrepareRawTnd(out, q, k, v, endpoints, endpoint_count, dims, scale);
  EnqueueRaw(call, workspace, workspace_bytes, device);
}
}  // namespace xinfer::cann_fia_v5

// ---------------------------------------------------------------------------
// Raw C-ABI wrappers (xinfer_ names; the glue binds these through ctypes).
namespace {
std::string& FiaError() {
  thread_local std::string buffer;
  return buffer;
}
}  // namespace

extern "C" const char* xinfer_cann_fia_v5_get_last_error() {
  return FiaError().c_str();
}

static int64_t BnsdWorkspaceSize(void* out, void* q, void* k, void* v, void* drop,
                                 const int64_t* dims, double scale, int64_t inner_precise, bool bsnd_output) {
  try {
    return xinfer::cann_fia_v5::RawBnsdWorkspaceBytes(out, q, k, v, drop, dims, scale, inner_precise, bsnd_output);
  } catch (const std::exception& error) {
    FiaError() = std::string("fia v5 workspace: ") + error.what();
  } catch (...) {
    FiaError() = "fia v5 workspace: unknown failure";
  }
  return -1;
}

extern "C" int64_t xinfer_cann_fia_v5_probe_workspace_size(
    int32_t, void* out, void* q, void* k, void* v, void* drop,
    const int64_t* dims, double scale, int64_t inner_precise) {
  return BnsdWorkspaceSize(out, q, k, v, drop, dims, scale, inner_precise, false);
}

extern "C" int64_t xinfer_cann_fia_v5_bnsd_bsnd_workspace_size(
    int32_t, void* out, void* q, void* k, void* v, void* drop,
    const int64_t* dims, double scale, int64_t inner_precise) {
  return BnsdWorkspaceSize(out, q, k, v, drop, dims, scale, inner_precise, true);
}

extern "C" int64_t xinfer_cann_fia_vision_probe_workspace_size(
    int32_t device, void* out, void* q, void* k, void* v, const int64_t* endpoints,
    int64_t endpoint_count, const int64_t* dims, double scale) {
  (void)device;
  try {
    return xinfer::cann_fia_v5::RawTndWorkspaceBytes(out, q, k, v, endpoints, endpoint_count, dims, scale);
  } catch (const std::exception& error) {
    FiaError() = std::string("fia vision workspace: ") + error.what();
  } catch (...) {
    FiaError() = "fia vision workspace: unknown failure";
  }
  return -1;
}

static int BnsdLaunch(int32_t device, void* out, void* workspace, int64_t workspace_bytes, void* q, void* k,
                      void* v, void* drop, const int64_t* dims, double scale, int64_t inner_precise,
                      bool bsnd_output) {
  try {
    xinfer::cann_fia_v5::RawBnsdLaunch(device, out, workspace, workspace_bytes, q, k, v, drop,
                                       dims, scale, inner_precise, bsnd_output);
    return 0;
  } catch (const std::exception& error) {
    FiaError() = std::string("fia v5 launch: ") + error.what();
  } catch (...) {
    FiaError() = "fia v5 launch: unknown failure";
  }
  return 1;
}

extern "C" int xinfer_cann_fia_v5_probe_launch(
    int32_t device, void* out, void* workspace, int64_t workspace_bytes, void* q, void* k,
    void* v, void* drop, const int64_t* dims, double scale, int64_t inner_precise) {
  return BnsdLaunch(device, out, workspace, workspace_bytes, q, k, v, drop, dims, scale, inner_precise, false);
}

extern "C" int xinfer_cann_fia_v5_bnsd_bsnd_launch(
    int32_t device, void* out, void* workspace, int64_t workspace_bytes, void* q, void* k,
    void* v, void* drop, const int64_t* dims, double scale, int64_t inner_precise) {
  return BnsdLaunch(device, out, workspace, workspace_bytes, q, k, v, drop, dims, scale, inner_precise, true);
}

extern "C" int xinfer_cann_fia_vision_probe_launch(
    int32_t device, void* out, void* workspace, int64_t workspace_bytes, void* q, void* k,
    void* v, const int64_t* endpoints, int64_t endpoint_count, const int64_t* dims, double scale) {
  try {
    xinfer::cann_fia_v5::RawTndLaunch(device, out, workspace, workspace_bytes, q, k, v,
                                      endpoints, endpoint_count, dims, scale);
    return 0;
  } catch (const std::exception& error) {
    FiaError() = std::string("fia vision launch: ") + error.what();
  } catch (...) {
    FiaError() = "fia vision launch: unknown failure";
  }
  return 1;
}
