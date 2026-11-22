# Kernel

[English](kernels.md) | 简体中文

WALL-OSS-0.5 和 WALL-OSS-1.0-Nano 的默认组合在 CUDA 上使用融合算子。它们的 kernel 在本仓库中（`xinfer/ops/backends/cuda/src`），编译成一个库 `xinfer_kernels.so`。没有编译这个库时，XInfer 用 PyTorch 参考实现运行同样的组合。安装可选的 `flash-attn` 包可启用视觉和 prompt 注意力的优化实现。

内置的 `cuda` 后端模块（`xinfer/ops/backends/cuda/`）调用 kernel 库和 flash-attn，与 `xinfer.ops` 入口点中的扩展以同样方式加载（[注册](design.zh-CN.md#142-后端注册与选择)）。库加载失败时，相应算子运行 torch 参考实现，并由一条告警列出变慢的算子；设置 `XINFER_OPS_STRICT=1` 时这条告警变成错误。例如，设置 `XINFER_FLASH_ATTN=0` 会在加载时跳过 flash-attn。

## GPU 支持

kernel 库针对一种 GPU 架构编译，只在这种架构上注册算子。本仓库有 SM120（RTX 50 系列）和 SM110（Jetson Thor）的 kernel。

GEMM 的调用落在某个专用 kernel 调优过的范围内时由这个 kernel 执行，否则走 cuBLASLt，epilogue 由单独的 kernel 完成。每个 kernel 接受的形状写在发射它的代码旁边（`xinfer/ops/backends/cuda/gemm_sm120.py`、`gemm_sm110.py`）：在 RTX 5090 上是实测不慢于 cuBLASLt 路径的行数区间；在 Jetson AGX Thor 上是连续三个或更多扫描点都快至少 5% 的行数区间，与参考 SM110 算子构建的选路一致，本库在参考请求上的输出与该构建逐位相同。在 SM110 上本库覆盖 WALL-OSS-0.5 和 WALL-OSS-1.0-Nano 的 BF16 计算组合。注意力和 flow 的 kernel 以同样方式列出各自的形状（`attn.py`、`flow.py`），范围之外的调用运行 torch 参考实现。

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
