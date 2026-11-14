# Kernel

[English](kernels.md) | 简体中文

Wall-OSS 和 Wall-X 2B 默认的 BF16 组合在 CUDA 上使用 xcompute 融合算子。没有兼容的 xcompute 构建时，XInfer 用 PyTorch 参考实现运行同一组合。安装可选的 `flash-attn` 包可启用视觉和 prompt 注意力的优化实现。

内置的 `cuda` 后端模块（`xinfer/ops/backends/cuda/`）调用 xcompute 和 flash-attn 库，与 `xinfer.ops` 入口点中的扩展以同样方式加载（[注册](design.zh-CN.md#142-后端注册与选择)）。库 import 失败时，相应算子运行 torch 参考实现，并由一条告警列出变慢的算子；设置 `XINFER_OPS_STRICT=1` 时这条告警变成错误。例如，设置 `XINFER_FLASH_ATTN=0` 会在加载时跳过 flash-attn。

## GPU 支持

xcompute 构建必须支持服务所用 GPU 的架构。可通过 `xcompute.build_target_archs()` 检查。[Wall-OSS 和 Wall-X 2B 的性能测试](performance.zh-CN.md)均使用 RTX 5090 和 xcompute `7336b0d`。为 Jetson Thor 构建（`--arch 110`）时，xcompute `c0d03ca` 及之后的版本还带有 Wall-OSS optimized 组合在 Thor 上选用的 SM110 kernel：融合的 prefill MLP、权重组播 GEMM，以及按旋转对顺序输出 Q、K 的 QKV 投影（[Wall-OSS 设计](models/wall_oss_design.zh-CN.md)）。

## 安装与选择构建

从提供方获得相互匹配的 xcompute Python 包与动态库并安装；XInfer 发行包不包含它们。BF16 需要统一的 GEMM、注意力和归一化接口。支持 `prepack_weight(layout="kn")` 的构建可以直接使用转置权重；旧版构建会把这些权重复制到连续张量中。Wall-X 2B 会逐项检查当前构建是否提供所需入口，选择可用的 CUDA 算子。若从源码构建 CUDA 13 版本，请在同一环境安装 `apache-tvm-ffi`，并针对部署 GPU 编译。

## 运行

把这份源码的 Python 包加入路径，它会找到这份源码编译出的库：

```bash
export PYTHONPATH=/path/to/xcompute/python:$PYTHONPATH
xinfer-serve --model wall_oss --checkpoint /path/to/checkpoint --device cuda \
  --warmup observation.msgpack --port 33985
```

XInfer 加载可导入的 `xcompute` 包所用的库，并使用其统一的 GEMM、注意力和归一化入口，同时关闭包在导入时的自动重新编译。设置 `XINFER_XCOMPUTE_LIBRARY=/path/to/xcompute.so` 时改为加载这个库（例如另一个 BF16 构建），这时包也要来自同一个构建。`XINFER_XCOMPUTE=0` 关闭这个后端。同样的环境变量也适用于 `Policy`、测试和 benchmark。

## 检查

`xinfer-serve` 在模型加载后会记录每个算子的后端：

```
xinfer.serve INFO model loaded in 7.2s on cuda-sm120; ops {..., 'gemm_epilogue': 'xcompute', 'gemm_glu_epilogue': 'xcompute', ...}
```

`--resolved-config FILE` 会把同样的列表写入 `FILE`；在 Python 中，`Policy.describe()["ops"]` 也会列出它。构建可用时，融合算子（`gemm_epilogue`、`gemm_glu_epilogue`、`qkv_rope_kvcache`，以及模型用到的 `xinfer/ops/reference/` 中的其他算子）显示为 `xcompute`。它们显示为 `reference` 说明 XInfer 没有找到可用的构建：

- 没有可导入的 `xcompute` 包且没有设置 `XINFER_XCOMPUTE_LIBRARY`，或它指向的文件不存在；
- 构建没有针对当前 GPU 的架构（`python -c "import xcompute; print(xcompute.build_target_archs())"`）；
- `xcompute` 包没有统一入口；
- 库缺少这些入口调用的函数：库与包来自不同的构建。

此时 `Policy` 和 `xinfer-serve` 在 GPU 上加载 Wall-OSS 时会记录一条警告：

```
wall_oss: 7 operators have no fast kernel loaded and fall back to slow torch code (add_rms_norm, ...). Install
xcompute built for this GPU, or set XINFER_XCOMPUTE_LIBRARY to such a build: docs/kernels.md
```

对于没有专用 kernel 的权重形状，xcompute 可以选择通用的 cuBLASLt GEMM。

## 昇腾

`ascend` 后端使用相同的共享 xcompute 加载器与 `XINFER_XCOMPUTE_LIBRARY` / `XINFER_XCOMPUTE=0`
设置。可用实现会自动选中；不支持的调用使用 torch 回退。运行环境、图像处理和 NPU 图执行见
[昇腾指南](ascend950dt.zh-CN.md)。
