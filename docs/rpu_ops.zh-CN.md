# 原生 RPU 算子

[English](rpu_ops.md) | 简体中文

RPU 算子后端位于 `xinfer/ops/backends/rpu/`，单独构建为 `xinfer_rpu.so`。
其中包含主机运行时与 TVM-FFI 绑定、catalog 与族解析器、Python 适配器及
设备源码。公开 Python 入口是 `from xinfer.ops.backends import rpu`。

## 分发与依赖

| 组件 | 源码分发包 | 使用条件 |
|---|---|---|
| 主机 C++17 运行时、绑定、catalog 与 Python 适配器 | 包含 | 主机解析及设备运行时接入 |
| 设备 `.rc` 源码及其配套头文件 | 包含 | 使用另行提供的编译器构建算子资产包 |
| 生成的 `.ref` 算子资产 | 另行提供 | 实际算子执行 |
| Rhino Launch SDK 与 `hxcc` | 由 HUIXI-AI 另行提供 | 运行时构建需要 SDK；设备源码编译需要编译器 |
| 模型 checkpoint 与请求数据 | 另行提供 | 模型级评估 |

原生库从外部资产目录加载算子文件，不将算子二进制嵌入动态库。
仅主机构建不需要 SDK、设备源码编译器或板卡。运行时 SDK 配置、算子资产生成
及设备执行是独立步骤。第三方归属及组件条款见[第三方声明](../THIRD_PARTY_NOTICES.md)。

设备源码面向 R1 Layer0 接口。后续硬件代际的兼容性需要单独移植和验证。
源码归档包含原生 RPU 源码，Python wheel 包含 Python 适配层，
原生运行库和设备资产另行构建。

## 仅主机构建

使用 C++17 编译器、CMake 3.18 或更新版本，以及安装了 Apache TVM FFI 的
Python 环境。在仓库根目录执行：

```bash
python -m pip install -e ".[kernels]"
cmake -S xinfer/ops/backends/rpu/src -B build/rpu-host \
  -DCMAKE_BUILD_TYPE=Release \
  -DXINFER_RPU_HOST_ONLY=ON -DXINFER_RPU_BUILD_TESTS=ON
cmake --build build/rpu-host --parallel
ctest --test-dir build/rpu-host --output-on-failure
export XINFER_RPU_LIBRARY="$PWD/build/rpu-host/lib/xinfer_rpu.so"
```

CMake 使用选定的 Python 解释器定位 TVM FFI；需要时在配置阶段指定
`-DPython_EXECUTABLE=/path/to/python`。

```python
from types import SimpleNamespace

from xinfer.ops.backends import rpu

backend = rpu.load_backend()
print(backend.provenance)
tensor = SimpleNamespace(shape=(16, 16), layout="row_major", distribution="local")
print(rpu.resolve("activation", {"x": tensor, "out": tensor}, function="gelu_tanh"))
```

此解析示例使用元数据，不分配设备内存，返回
`('gelu_f16', {'x': 'src', 'out': 'out'}, {'mode': 1})`。

来源信息包含动态库逻辑名称及 SHA-256、编入的后端、ABI 版本、catalog SHA-256
及 `runtime_available`。仅主机构建报告 `runtime_available=False`：
`rpu.resolve(...)` 可解析族调用，但创建设备上下文会明确失败。主机解析不执行
算子，也不能验收其数值行为。成功创建的设备上下文提供 `asset_provenance` 副本，
其中记录分配前实际校验的 manifest 字节的 SHA-256。安装路径仅供本地使用。

## 运行时构建

另行安装兼容的 Rhino Launch SDK，并将 `RHINO_SDK_ROOT` 指向其根目录。
系统动态库加载器须能找到 SDK 动态库。

```bash
export RHINO_SDK_ROOT=/path/to/rhino-launch-sdk
cmake -S xinfer/ops/backends/rpu/src -B build/rpu-runtime \
  -DCMAKE_BUILD_TYPE=Release \
  -DXINFER_RPU_HOST_ONLY=OFF -DRHINO_SDK_ROOT="$RHINO_SDK_ROOT" \
  -DXINFER_RPU_BUILD_TESTS=ON
cmake --build build/rpu-runtime --parallel
export XINFER_RPU_LIBRARY="$PWD/build/rpu-runtime/lib/xinfer_rpu.so"
```

`rpu.create_backend(...)` 创建低层设备上下文。除运行时动态库外，它还需要
匹配的设备环境和算子资产。导入 Python 后端不会构建原生库或初始化设备。
第一次加载后端前应选定动态库与资产；缺库、ABI／catalog 不匹配及无效资产
都会报错，不会回退到 CPU 或 Torch 算子。

## 编译与选择算子资产

配置外部提供的 `hxcc` 可执行文件，使用源码树之外的新目录或空目录
存放输出：

```bash
export HXCC=/path/to/hxcc
python tools/build_rpu_assets.py --compiler "$HXCC" --output /path/to/rpu-assets
export XINFER_RPU_ASSETS=/path/to/rpu-assets
```

`--kernels name ...` 可将编译范围限制为指定的 catalog 入口。驱动使用标准
`hxcc -O2 -c <source>.rc -o <name>.ref` 接口和必要的 include 配置，在所选
资产编译完成后写出 `manifest.json`。生成文件不进入源码分发包或二进制包。

资产目录包含 `manifest.json` 及同目录的 `<catalog-entry>.ref` 文件。
manifest 结构如下，构建驱动会填入实际摘要：

```json
{
  "abi_version": 11,
  "catalog_sha256": "<native catalog SHA-256>",
  "assets": {
    "<catalog-entry>": {"sha256": "<entry.ref SHA-256>"}
  }
}
```

Python provider 检查 ABI 与原生 catalog 标识、校验已声明文件的摘要，并只准入
原生 catalog 中的入口。每个资产必须是同目录下非空的普通文件；符号链接和
自定义路径会被拒绝。原生加载器在加载算子时也检查所选文件。动态库、catalog、
manifest 和算子资产必须属于同一个受支持构建；编译成功本身不能证明它们匹配。

## 执行契约

RPU 与 CPU、CUDA、Ascend 使用同一 `OpRegistry` 和 `build_table`。
`prepare_<family>` 规格选择的调用接收原生上下文、常驻张量、整数、标量和
selector，返回原生预构建操作，签名区别于 Torch 张量调用。它们没有 Torch
参考实现或回退，报告 `Contract(numerics="unqualified")`。固定选择不受支持或
请求 `reference_only` 时显式报错。空的 RPU 选择在绑定时报错；模型实际准备
某个操作时缺少选择，也会显式报错。算子选择不构成数值验收。

`rpu.bind_ops(table)` 返回使用表中所选调用的算子工厂，保留表上的包装。
匹配的框架接入将它与 `load_backend().backend_factory` 一同注入：模型计划
调用 `context.prepare`，经表中选中的实现完成原生准备。准备不执行或等待；
批执行和完成等待保留原有生命周期。仅有算子包并不提供模型或运行时接入。

`rpu.resolve(op, tensors, integers=None, **selectors)` 使用动态库中的族解析器。
支持的形状、dtype、布局、别名规则和 selector 值由随附 catalog 及族绑定定义。
准备执行前先解析族调用；改变数值行为须使用对应的显式 selector。

TP 接口使用常驻缓冲区与准备操作。调用方负责张量分布、算子顺序及必要的集合
通信步骤。投影的 partial 输出在调用方连接规约之前仍是 partial。
候选入口要求 `allow_candidate=True`；候选准入和符号可用不代表板卡验收。

## 验证边界

仅主机 CTest 和 Python 测试不执行设备算子，用于检查契约、族解析及加载行为。
使用 SDK stub 的编译检查仅验证主机代码与测试接口的集成，不等于真实 SDK
构建或板卡执行。

GELU Repeat 路径使用标准缩放算术，在加法阶段应用 1.5957691216 系数，
再乘输入，从而保留原实现的 FP16 舍入顺序。加法的两个输入缩放均初始化为一。

HUIXI-AI 报告使用 hxcc 1.1.0 和 R1 仿真器验证了独立 tanh GELU 路径的全部
65,536 个 FP16 编码，以及 33、225、417 行的融合 up 投影用例，均与原实现逐位一致。
真实设备验证需使用选定的运行库与资产包。Torch 对照还会衡量相对于另一参考实现的
近似误差。设备数值测试通过 `XINFER_RPU_RUN_BOARD=1` 显式启用，
应在独占板卡上使用选定的 SDK 和资产执行。

真实 SDK 编译、设备源码编译和板卡执行须使用选定的编译器、SDK、资产包及设备
分别验证。算子数值对齐、准备操作回放、故障恢复及模型精度／性能门禁需要实际
硬件与输入。本次源码迁移不构成数值或性能验收。
