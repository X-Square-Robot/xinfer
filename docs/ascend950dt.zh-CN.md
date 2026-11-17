# 昇腾 950DT

[English](ascend950dt.md) | 简体中文

Wall-OSS 和 Wall-X 2B 在昇腾 950DT 上复用同一算子注册表和引擎。仓内 AOT 算子编译为 `xinfer_kernels_ascend.so`，无需外部 xcompute 包。Wall-OSS 还保留仓内 TileLang JIT 算子，在预热期间编译。不可用的实现使用已注册的回退路径。

## 编译

在 aarch64 上使用匹配的 PyTorch/TorchNPU、昇腾 TileLang 编译环境、CMake 3.24+ 和 TVM-FFI。工具链使用 CANN 9.1.1 运行和链接，使用 CANN 9.3.0 中的 bisheng 编译。导入 TorchNPU 前先加载运行时环境。AOT 编译使用仓内算子代码与模板头文件，无需预编译对象或额外算子仓库。AA-resample 直接编译原始生成后的设备和主机源码。

```bash
source /path/to/cann-9.1.1/set_env.sh
export BISHENG_HOME=/path/to/cann-9.3.0
export ASCEND_HOME_PATH="$BISHENG_HOME" ASCEND_TOOLKIT_HOME="$BISHENG_HOME"
export TILELANG_CACHE_DIR="$PWD/artifacts/tilelang-cache"
python -m pip install -e ".[kernels,models,serve]"
cmake -S xinfer/ops/backends/ascend/src -B build/kernels-ascend \
  -DPython_EXECUTABLE="$(command -v python)" \
  -DXINFER_ASCEND_SOC=ascend950dt9582 \
  -DXINFER_CANN_RUNTIME=/path/to/cann-9.1.1 \
  -DXINFER_BISHENG_HOME=/path/to/cann-9.3.0 \
  -DXINFER_TILELANG_PYTHON=/path/to/tilelang/bin/python
cmake --build build/kernels-ascend -j1
```

库生成在昇腾 Python 后端旁。可选的主机辅助模块使用与服务相同的 Python 和 Torch 编译：

```bash
cmake -S xinfer/src -B build/host \
  -DPython_EXECUTABLE="$(command -v python)" -DXINFER_HOST_ASCEND=ON
cmake --build build/host -j1
```

Wall-OSS 配方在运行时也需要支持昇腾的 TileLang 分支和 bisheng，不只是 AOT 编译时需要。原有 G2112/SAMU factory 位于 `src/gemm/g2112_tilelang.py`，后端原样转出。实测分支报告版本 `0.1.15+cpu.git1e8448fc`。`kernels` extra 安装 TVM-FFI，不安装该编译器。保留 CANN 9.1.1 的运行时环境，通过上面的 exports 选择编译器，不要改为加载编译器工具包的运行时环境。

## 运行

后端名为 `ascend`，PyTorch 设备名为 `npu:N`。设置 `ASCEND_RT_VISIBLE_DEVICES` 后，索引相对于可见设备集。两个模型继续使用原有 checkpoint 和线上请求格式。

```bash
source examples/ascend/wall_oss.env
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 --reference-ops \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --warmup flow.msgpack rtc.msgpack --port 33985
```

Wall-X 2B 先加载 [wallx2b.env](../examples/ascend/wallx2b.env)，再使用 `--model wallx2b` 及其 checkpoint 和请求。这些配方选择实测的 BF16 配置、显式原生算子与图像路径，不改变包默认值。使用没有其他 XInfer 覆盖设置的干净 shell。`Policy(device="npu:0")` 使用同一后端和引擎。

XInfer 默认加载包内库。`XINFER_ASCEND_KERNELS_LIBRARY=/path/to/xinfer_kernels_ascend.so` 选择同一源码的其他构建；`XINFER_ASCEND_KERNELS=0` 禁用原生注册。默认构建缺失时仍可使用回退；显式路径不存在或选中的库损坏时会报错。

## 图像和图执行

服务默认在 CPU 上处理图像。顶层选项 `image_preprocess_device="gpu"` 在可用时选择 DVPP；`Policy` 在支持的平台上默认使用此路径。不支持的 JPEG 类型在主机解码后到设备缩放。比较输出和性能时需记录图像路径。

平台负责 JPEG 资源、流顺序和解码后的图像图。已选中的 `image_resize` 算子提供缩放契约。进程内预处理可以把视图预取到其所属引擎，前端进程不初始化设备。

运行时支持时默认启用阶段图。服务开始前用代表性的 flow/RTC 形状预热，每个形状先以 eager 执行再捕获。`--enforce-eager` 或 `Policy(cudagraphs=False)` 禁用图。NPU 上使用 `--reference-ops` 且未固定原生注意力时，应配合 `--enforce-eager`，因为参考 SDPA 可能不支持捕获。配方固定了可捕获的注意力；原生布局不受支持时，使用已注册的、可捕获的昇腾 torch 组合。

流优先级使用公共运行时接口。无法核实优先级范围的运行时使用优先级 0，并拒绝非零请求。

## 检查

`Policy.describe()["ops"]`、启动日志及 `--resolved-config` 记录所选后端、AOT 库指纹和契约。仓内库显示为 `ascend:kernels`。基准环境元数据另外记录 TileLang 的可用性、版本及 JIT 源码 SHA；这些身份不证明逐调用算子覆盖率，也没有逐个标识生成的 JIT 二进制。不支持的调用可能走通用或参考路径。调用被准入后，原生执行错误会向上传递。

先运行 CPU 契约测试，再在独占设备上运行 NPU 运行时、图像和模型测试。用固定预处理输入和初始噪声比较 eager 与图模式，覆盖 RTC、新形状和重复请求。复用既有[精度门禁](precision.zh-CN.md)和[基准脚本](benchmark.zh-CN.md)，原始证据保留在 Git 外。

```bash
source examples/ascend/wall_oss.env
python benchmarks/performance.py --name "Wall-OSS · BF16 · Ascend 950DT" --reference-ops \
  --model wall_oss --checkpoint /path/to/checkpoint --device npu:0 \
  --model-options "$ASCEND_MODEL_OPTIONS" --op-backends "$ASCEND_OP_BACKENDS" \
  --latency-requests flow.msgpack rtc.msgpack --warmup flow.msgpack rtc.msgpack \
  --robots 1 2 3 4 5 6 --out artifacts/perf-wall-oss-ascend
```

Wall-X 2B 替换为对应模型、checkpoint 和代表性请求集。比较构建时保持输入、精度、图像路径和图设置一致。[性能表](performance.zh-CN.md)记录已验证结果及测量条件。
