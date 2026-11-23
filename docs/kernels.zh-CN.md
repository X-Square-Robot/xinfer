# Kernel

[English](kernels.md) | 简体中文

WALL-OSS-0.5 和 WALL-OSS-1.0-Nano 的默认组合在 CUDA 上使用融合算子。它们的 kernel 在本仓库中（`xinfer/ops/backends/cuda/src`），编译成一个库 `xinfer_kernels.so`。没有编译这个库时，XInfer 用 PyTorch 参考实现运行同样的组合。安装可选的 `flash-attn` 包可启用视觉和 prompt 注意力的优化实现。

内置的 `cuda` 后端模块（`xinfer/ops/backends/cuda/`）调用 kernel 库和 flash-attn，与 `xinfer.ops` 入口点中的扩展以同样方式加载（[注册](design.zh-CN.md#142-后端注册与选择)）。库加载失败时，相应算子运行 torch 参考实现，并由一条告警列出变慢的算子；设置 `XINFER_OPS_STRICT=1` 时这条告警变成错误。例如，设置 `XINFER_FLASH_ATTN=0` 会在加载时跳过 flash-attn。

## GPU 支持

kernel 库针对一种 GPU 架构编译，只在这种架构上注册算子。本仓库有 SM120（RTX 50 系列）和 SM110（Jetson Thor）的 kernel。

GEMM 的调用落在某个专用 kernel 调优过的范围内时由这个 kernel 执行，否则走 cuBLASLt，epilogue 由单独的 kernel 完成。每个 kernel 接受的形状写在发射它的代码旁边（`xinfer/ops/backends/cuda/gemm_sm120.py`、`gemm_sm110.py`）：在 RTX 5090 上是实测不慢于 cuBLASLt 路径的行数区间；在 Jetson AGX Thor 上是连续三个或更多扫描点都快至少 5% 的行数区间，与参考 SM110 算子构建的选路一致，本库在参考请求上的输出与该构建逐位相同。在 SM110 上本库覆盖 WALL-OSS-0.5 和 WALL-OSS-1.0-Nano 的 BF16 计算组合。注意力和 flow 内核各自的已测形状列在 `attn.py` 和 `flow.py` 中。

当原生 attention 内核不覆盖输入形状时，分段 attention 和带门控的因果 attention 可用 flash-attn，有无输出门控的前缀 KV 缓存 attention 可用编译的 FlexAttention，门控 delta 规则可用 FLA。首次以给定设备、形状、类型、步长和选项在 eager 模式调用时，会尝试可选库。CUDA Graph 录制可以使用已验证可运行的路径，首次就在录制中遇到的新形状则使用参考实现。启动前可设 `XINFER_FLASH_ATTN=0`、`XINFER_FLEX_ATTENTION=0` 或 `XINFER_FLA=0` 关闭对应库。`xinfer-profile check` 的 `routes` 字段按观测到的 attention 调用形状列出实际运行的库。

对于已测范围以外的输入，如果 split-KV 内核和 FlexAttention 都可用，带或不带输出门控的前缀 KV 缓存注意力默认会比较这两种实现。在启动前设置 `XINFER_OP_TUNING=0` 可跳过计时，按固定顺序选择实现。eager 预热时，两种实现读取相同的缓存，分别写入独立输出。原生结果相对 FlexAttention 的 L2 相对误差须小于 1%。同一进程内，相同设备、布局和选项会复用合格且较快的实现。CUDA Graph 录制时复用已有选择，未命中则使用静态顺序。

在 Jetson Thor 上，FlexAttention 需要能识别 `sm_110a` 的 PTX 汇编器。如果 Triton 3.6 提示自带的 `ptxas-blackwell` 不支持该目标，请将 `CUDA_HOME` 指向支持 Thor 的 CUDA 工具包，并在启动 XInfer 前设置 `TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas"`。

## 编译

编译需要 CUDA 13 工具链、CMake 3.24 或更高版本（SM110 需要 3.28）、`apache-tvm-ffi` Python 包和 CUTLASS 头文件（子模块）：

```bash
git submodule update --init third_party/cutlass
pip install -e ".[kernels]"
cmake -S xinfer/ops/backends/cuda/src -B build/kernels
cmake --build build/kernels -j
```

产物是 `xinfer/ops/backends/cuda/xinfer_kernels.so`。目标架构取编译机器的第一块 GPU；在没有该 GPU 的机器上用 `-DXINFER_CUDA_ARCH=120`（或 `110`）指定。

## 运行

```bash
xinfer-serve --model wall-oss-0.5 --checkpoint /path/to/checkpoint --device cuda \
  --warmup observation.msgpack --port 33985
```

XInfer 从自己的包目录加载 `xinfer_kernels.so`。设置 `XINFER_KERNELS_LIBRARY=/path/to/xinfer_kernels.so` 时改为加载在别处用同一份源码编译的库。`XINFER_KERNELS=0` 关闭这个后端。同样的环境变量也适用于 `Policy`、测试和 benchmark。

## Flow 内核覆盖范围

在受支持的 GPU 上，只要原生内核支持输入的形状、类型、布局和可选参数，`flow_input_embed` 和 `flow_velocity_euler` 就直接运行原生内核，包括实测范围之外的兼容形状。Flow 没有需要比较耗时的外部库，因此设置 `XINFER_OP_TUNING=0` 时也按这条规则选路。不支持的输入使用 torch 参考实现，并沿用严格模式的诊断。`xinfer-profile check` 会显示所选档位和内核入口，例如 `measured (denoise_input_embed)` 或 `can_run (denoise_output_euler)`。

## 检查

`xinfer-serve` 在模型加载后会记录每个算子的后端：

```
xinfer.serve INFO model loaded in 7.2s on cuda-sm120; ops {..., 'gemm_epilogue': 'kernels', 'gemm_glu_epilogue': 'kernels', ...}
```

`--resolved-config FILE` 会把同样的列表写入 `FILE`；在 Python 中，`Policy.describe()["ops"]` 也会列出它。库已加载时，融合算子（`gemm_epilogue`、`gemm_glu_epilogue`、`qkv_rope_kvcache`，以及模型用到的 `xinfer/ops/reference/` 中的其他算子）显示为 `kernels`。它们显示为 `reference` 说明 XInfer 没有加载到这个库：

- 没有编译 `xinfer/ops/backends/cuda/xinfer_kernels.so`，或 `XINFER_KERNELS_LIBRARY` 指向的文件不存在；
- 没有安装 `apache-tvm-ffi` 包；
- 库是为另一种 GPU 架构编译的。

此时 `Policy` 和 `xinfer-serve` 在 GPU 上加载 WALL-OSS-0.5 时会记录一条警告：

```
wall-oss-0.5: 7 operators have no fast kernel loaded and fall back to slow torch code (add_rms_norm, ...). Build the
CUDA kernel library for this GPU: docs/kernels.md
```

`pytest tests/kernels` 在每个 kernel 接受的形状上把它与 FP32 torch 对比；需要 GPU 和已编译的库。

## 昇腾

`ascend` 后端通过 `xinfer/ops/backends/_xcompute.py` 绑定外部的 xcompute 构建，使用
`XINFER_XCOMPUTE_LIBRARY` / `XINFER_XCOMPUTE=0` 设置。可用实现会自动选中；不支持的调用使用 torch 回退。
运行环境、图像处理和 NPU 图执行见[昇腾指南](ascend950dt.zh-CN.md)。

## 通用 GEMM 调优

通用 GEMM 默认采用 cuBLASLt 的第一个启发式算法。使用本版本构建的库时，可在启动前设 `XINFER_GEMM_TUNING=1`，在每种形状和内存对齐方式的首次 eager 调用中比较候选算法。候选算法要在预热输入上与默认算法的结果一致，且耗时中位数至少快 5%。试跑使用独立输出缓冲区。原地残差更新沿用默认算法。将 `XINFER_GEMM_TUNING_FILE` 指向可写的 JSON 路径，可以跨进程共享选择。可复现回放时设 `XINFER_GEMM_TUNING=frozen`，并指向同一文件。在 CUDA Graph 录制前先对每种形状做 eager 预热。冻结模式下缺失或不兼容的选择会报错，普通调优模式在冷启动录制时使用默认算法。
