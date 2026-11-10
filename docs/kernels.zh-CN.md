# Kernel

[English](kernels.md) | 简体中文

Wall-OSS 默认的 BF16 组合在 CUDA 上使用 xcompute 融合算子。没有兼容的 xcompute 构建时，XInfer 用 PyTorch 参考实现运行同一组合。安装可选的 `flash-attn` 包可启用视觉和 prompt 注意力的优化实现。

内置的 xcompute 和 flash-attn 后端与通过 `xinfer.ops` 入口点加载的扩展使用相同的[注册协议](design.zh-CN.md#142-后端注册与选择)。例如，设置 `XINFER_FLASH_ATTN=0` 会在加载时跳过 flash-attn。

## GPU 支持

xcompute 构建必须支持服务所用 GPU 的架构。可通过 `xcompute.build_target_archs()` 检查。[Wall-OSS 性能测量](performance.zh-CN.md)使用 RTX 5090 和 xcompute `7336b0d`。

## 安装与选择构建

从提供方获得相互匹配的 xcompute Python 包与动态库并安装；XInfer 发行包不包含它们。BF16 需要统一的 GEMM、注意力和归一化接口。权重预打包需要提供 `prepack_weight(layout="kn")` 的构建。若从源码构建 CUDA 13 版本，请在同一环境安装 `apache-tvm-ffi`，并针对部署 GPU 编译。

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

`--resolved-config FILE` 会把同样的列表写入 `FILE`；在 Python 中，`Policy.describe()["ops"]` 也会列出它。构建可用时，融合算子（`gemm_epilogue`、`gemm_glu_epilogue`、`qkv_rope_kvcache`，以及模型用到的 `xinfer/ops/fused.py` 中的其他算子）显示为 `xcompute`。它们显示为 `reference` 说明 XInfer 没有找到可用的构建：

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
