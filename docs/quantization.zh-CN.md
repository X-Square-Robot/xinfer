# 量化

[English](quantization.md) | 简体中文

线性层的量化方法由 `QuantConfig` 决定（docs/design.md §12）。模型为每个线性层指定一个*组*（group）；配置把组映射到方法，组会继承其以点分隔的各级父组的设置。

```python
from xinfer.models.layers.quantization import QuantConfig

QuantConfig(groups={"vision.blocks": "mxfp8", "vision.merger": "mxfp8", "prefill": "mxfp8"})
```

checkpoint 加载完成后，`process_weights(model)` 在权重被共享或 graph 被捕获之前，把每一层转换一次。模型可以定义预设（`quantization="<preset>"`），也可以接受一个 `QuantConfig`，或一个包含 `groups`、`default` 和 `options` 的 dict；Wall-X 2B 提供 `"bf16"`（默认）和 `"mxfp8"` 两种预设。MXFP8 对视觉块和提示词路径使用固定的 1×128 缩放。单独启用其中一个阶段可用 `{"groups":{"vision":"mxfp8"}}` 或 `{"groups":{"prefill":"mxfp8"}}`。详见 [Wall-X 2B 设计](models/wallx2b_design.zh-CN.md)。Wall-OSS 使用 BF16。目前没有模型预设使用 `nvfp4` 或 `nvfp4_mse`。

## 方法

| 方法 | 权重 | 前向 |
|---|---|---|
| `bf16`（默认） | 加载时的 dtype | `linear` 算子 |
| `mxfp8` | E4M3 + E8M0 块缩放因子，由 BF16 权重量化得到；丢弃 BF16 权重 | 逐行量化 BF16 输入，块缩放 GEMM，输出 BF16（融合形式见下文） |
| `nvfp4` | E2M1 + E4M3 块缩放因子（由块 amax 得到），由 BF16 权重量化得到；丢弃 BF16 权重 | 逐行量化 BF16 输入，块缩放 GEMM，bias 在 epilogue 中相加，输出 BF16 |
| `nvfp4_mse` | 同 `nvfp4`，但每个块缩放因子是在逐张量全局缩放因子下通过 MSE 搜索选出的 | 同 `nvfp4`；全局缩放因子乘到 FP32 累加器上，bias 加到 BF16 输出上 |

`mxfp8` 的选项通过 `QuantConfig(options={"mxfp8": {...}})` 传入：
- `share_scale_128`（默认 `True`）：每 128 个输入列一个缩放因子，即 Wall-X 的“1x128”方案。
- `fast_accum`：`"auto"`、`True` 或 `False`。`"auto"` 在 `rows <= 16 or out_features >= 8192` 时开启 FP8 快速累加，这是 Wall-X 服务的规则。它只作用于普通 GEMM。

输入宽度必须是 32 的倍数。若配置把不支持的层映射到 `mxfp8`，构建模型时会失败。

`nvfp4` 和 `nvfp4_mse` 是 Wall-X NVFP4 服务的两种方案（ViT 用 `nvfp4_1d1d_16bs`，expert 0 用 `nvfp4_mse_1d1d_16bs`）。它们是两个方法，而不是一个带选项的方法，因为选项按方法生效，而该方案在一个模型中同时使用两者。它们没有选项。激活始终使用 amax 缩放因子。输入和输出宽度必须是 16 的倍数。

## 可融合的层形式

层为每个投影请求一种形式；由方法决定融合什么。未量化的方法运行普通组合，即 eager 数值，因此这种请求不会改变 BF16 模型。

| 层调用 | 计算 | `mxfp8` 激活量化 | `mxfp8` GEMM |
|---|---|---|---|
| `linear(x)` | `x W^T + b` | `mxfp8_quantize` | `mxfp8_gemm`（bias 在 epilogue 中） |
| `linear(x, norm=n)` | `n(x) W^T + b` | `mxfp8_rmsnorm_quantize`：RMSNorm 以 FP32 计算，量化前不经 BF16 舍入 | 同上 |
| `linear(gu, input_act="silu_and_mul")` | `(silu(g) * u) W^T + b` | `mxfp8_silu_mul_quantize`：以 FP32 计算，不经 BF16 舍入 | 同上 |
| `linear(x, residual=r)` | `r + x W^T + b` | 同上 | `mxfp8_gemm_residual`：`r` 加到 FP32 累加器上，只舍入一次 |
| `QKVLinear(x, rope, norm=n)` | 旋转后的 q、k 和 v | 同上 | `mxfp8_gemm_rope`：bias 和旋转位置编码作用于 FP32 累加器，使用 FP32 cos/sin |
| `GatedLinear(x, norm=n)` | `[g \| u]` 的 `silu(g) * u` | 同上 | `mxfp8_gemm_silu_mul`：在 FP32 累加器上计算，只舍入一次 |

- `QKVLinear` 和 `GatedLinear` 的权重按融合 epilogue 读取的顺序存储各行（`rope_pair_permutation`、`gate_up_interleave`）。以普通形式调用这类层会抛出异常。
- `GatedMLP(act_fused_into="gate_up" | "down")` 把门控激活放在 gate/up 的 epilogue 中，或放在 down 投影的输入量化中。
- `RopeTables`（来自 `MRope.tables`）保存一次前向的 FP32 cos/sin。它们既用于 eager 旋转（cos/sin 舍入到激活的 dtype），也用于融合 kernel 的打包表（使用未舍入的值）。

这些形式与 Wall-X 4B MXFP8 服务的融合方式一致（ViT 的 down 投影经由 `silu_mul_quantize`，expert-0 整层融合）。目前没有模型预设使用它们；由 `tests/unit/test_mxfp8.py` 覆盖。

`nvfp4` 和 `nvfp4_mse` 只融合进激活量化，与 Wall-X NVFP4 服务的做法相同：

| 层调用 | `nvfp4` / `nvfp4_mse` 激活量化 |
|---|---|
| `linear(x)` | `nvfp4_quantize` |
| `linear(x, norm=n)` | `nvfp4_rmsnorm_quantize`：先把 `n(x)` 舍入到 BF16，与 Wall-X kernel 的做法一致 |
| `linear(gu, input_act="silu_and_mul")` | `nvfp4_silu_mul_quantize`：以 FP32 计算，不经 BF16 舍入 |

GEMM 始终是输出 BF16 的 `nvfp4_gemm`。其余部分都在该输出上运行普通组合：残差加法、`QKVLinear` 的 eager rotary，以及 `GatedLinear` 的 `silu_and_mul`。权重行保持 checkpoint 中的顺序。该服务在 down 投影的输入量化中计算 expert-0 的门控激活，对应 `GatedMLP(act_fused_into="down")`。

## MXFP8 格式

格式由 `xinfer/ops/reference/mxfp8.py` 中的 torch 参考实现定义，scale 布局和权重行顺序在 `xinfer/ops/layout/mxfp8.py` 中，与 xcompute 保持一致：
- **元素**：E4M3。
- **缩放因子**：每 32 列一个 E8M0 字节；`share_scale_128` 时一个缩放因子复制到 128 列。
- **缩放值**：能把块的 amax 映射到 ±448 以内的最小的 2 的幂。
- **舍入**：钳位后按最近偶数舍入。
- **缩放因子存储**：SWIZZLE_32_4_4 布局（行填充到 128，块填充到 4）。

融合算子的参考实现在 kernel 运算有明确规定之处遵循其运算方式：
- `rmsnorm_quantize` 按 kernel 的顺序求平方和（先求 128 个逐线程部分和，再树形归约），并计算 `(x * rsqrt(mean + eps)) * w`。
- GEMM 参考实现以 FP32 乘反量化后的操作数，并以 FP32 执行 epilogue。

## NVFP4 格式

格式由 `xinfer/ops/reference/nvfp4.py` 中的 torch 参考实现定义，scale 布局在 `xinfer/ops/layout/nvfp4.py` 中，它遵循生产服务中的 Wall-X NVFP4 kernel 及其下游的 GEMM：
- **元素**：E2M1（0、0.5、1、1.5、2、3、4、6 及其负值），每字节两个，存为 uint8 `[M, K/2]`：第 `2j` 列在低半字节，第 `2j + 1` 列在高半字节，bit 3 为符号位。
- **缩放因子**：每 16 列一个 FP8 E4M3 缩放因子，以 `float8_e4m3fn` 按 SWIZZLE_32_4_4 布局存储（行填充到 128，16 列的块填充到 4，填充值为零）。
- **全局缩放因子**：`nvfp4_mse` 权重带有一个逐张量的 FP32 缩放因子 `g = amax / (448 · 6)`，按整个权重计算（以双精度计算，舍入到 FP32；全零权重为 1）。一个值等于 `e2m1 · scale · g`；激活和 `nvfp4` 权重的 `g = 1`。
- **缩放值（amax）**：`e4m3(clamp(amax / (6g), 2^-6, 448))`，按最近偶数舍入。amax 低于 `6 · 2^-6` 的块取下限缩放值 `2^-6`。
- **缩放值（MSE，`nvfp4_mse` 权重）**：按从小到大的顺序，用 126 个正的有限 E4M3 值 `v` 中的每一个把块重建为 `rn(x / (v · g)) · (v · g)`；平方误差按 kernel 的 warp 归约顺序求和（第 `i` 列加第 `i + 8` 列，再把这 8 个和对半相加，依此类推），取第一个出现的最小和。
- **舍入**：元素为以 FP32 计算的 `x / (scale · g)`，舍入到最近的 E2M1 值，平局时取偶数编码，饱和到 ±6；舍入为零的负值保留其符号位。

融合算子的参考实现遵循 kernel 的运算方式：
- `rmsnorm_quantize` 按 kernel 的顺序求平方和（先求 256 个逐线程部分和，再树形归约），用舍入到 BF16 的 norm 权重计算 `x · (rsqrt(mean + eps) · w)`，并在量化前把结果舍入到 BF16。
- `silu_mul_quantize` 以 FP32 计算 `silu(gate) · up`，不经 BF16 舍入直接量化。
- `gemm` 以 FP32 乘反量化后的操作数，然后在累加器上应用 `alpha`（全局缩放因子）和 bias，并一次性舍入到 BF16。`nvfp4_mse` 不传入 bias，而是把它加到 BF16 输出上，因为服务中带全局缩放因子的 CUTLASS kernel 没有 bias。

参考实现做精确除法。Wall-X kernel 以 fast-math 编译（近似除法、`rsqrtf`、`__expf`）：当精确的商恰为舍入平局点或与之相差几个 ulp 以内时，它们可能选到相邻的缩放值或元素。

## 后端

PyTorch 提供 MXFP8 和 NVFP4 的参考算子。在 CUDA 上，包含 MXFP8 GEMM 和归一化入口的 xcompute 构建可以加速 Wall-X 2B 的 MXFP8 预设。构建和所选算子的检查方法见 [kernel 指南](kernels.zh-CN.md)。

## 测试

| 测试 | 检查内容 |
|---|---|
| `tests/unit/test_mxfp8.py` | 参考格式、误差界和融合参考实现（CPU）；可融合形式：BF16 与 eager 组合逐位一致，MXFP8 贴近 BF16（CPU） |
| `tests/unit/test_nvfp4.py` | 参考格式对照手算的编码和缩放因子、MSE 缩放值搜索、融合参考实现、GEMM epilogue（CPU）；`nvfp4` 和 `nvfp4_mse` 层及其可融合形式（CPU）；参考 GEMM 对比 `torch._scaled_mm` NVFP4，即 Wall-X 服务在没有 CUTLASS kernel 的形状上所用的 GEMM（GPU，sm100 或更新） |

量化改动按[精度](precision.zh-CN.md)中的方式检查：与参考组合的一致性，以及内部精度门禁。
