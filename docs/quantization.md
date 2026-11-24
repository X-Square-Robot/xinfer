# Quantization

English | [简体中文](quantization.zh-CN.md)

Linear layers get their method from a `QuantConfig` (docs/design.md §12).
Models assign each linear layer a *group*; the config maps groups to
methods, and a group inherits from its dot-separated parents.

```python
from xinfer.models.layers.quantization import QuantConfig

QuantConfig(groups={"vision.blocks": "mxfp8", "vision.merger": "mxfp8", "prefill": "mxfp8"})
```

After the checkpoint is loaded, `process_weights(model)` converts each
layer once, before weights are shared or graphs captured. Models may define
presets (`quantization="<preset>"`), or take a `QuantConfig` or a dict with `groups`, `default` and
`options`; WALL-OSS-1.0-Nano defines `"bf16"` (default), `"mxfp8"` and `"nvfp4"`. Its MXFP8 preset covers the vision blocks and the prompt stream with a fixed 1×128 scale recipe. Its NVFP4 preset is the Wall-X NVFP4 recipe: `nvfp4` for the vision transformer and `nvfp4_mse` for the prompt stream, whose DeltaNet z/b/a projection stays BF16. For individual stages, use `{"groups":{"vision":"mxfp8"}}` or `{"groups":{"prefill":"mxfp8"}}` (or `nvfp4`, `nvfp4_mse`). See the [WALL-OSS-1.0-Nano design](models/wall_oss_1_0_nano_design.md). WALL-OSS-0.5 uses BF16.

## Methods

| Method | Weights | Forward |
|---|---|---|
| `bf16` (default) | loaded dtype | `linear` operator |
| `mxfp8` | E4M3 + E8M0 block scales, quantized from the BF16 weight; BF16 weight dropped | quantize the BF16 input per row, block-scaled GEMM with BF16 output (fused forms below) |
| `nvfp4` | E2M1 + E4M3 block scales from the block amax, quantized from the BF16 weight; BF16 weight dropped | quantize the BF16 input per row, block-scaled GEMM with the bias in the epilogue, BF16 output |
| `nvfp4_mse` | as `nvfp4`, but each block scale is chosen by an MSE search under a per-tensor global scale | as `nvfp4`; the global scale multiplies the FP32 accumulator and the bias is added to the BF16 output |

`mxfp8` options are passed as `QuantConfig(options={"mxfp8": {...}})`:
- `share_scale_128` (default `True`): one scale per 128 input columns, the Wall-X "1x128" recipe.
- `fast_accum`: `"auto"`, `True` or `False`. `"auto"` turns on fast FP8 accumulation for `rows <= 16 or out_features >= 8192`, which is the Wall-X service rule. It applies to the plain GEMM only.

The input width must be a multiple of 32. A config that maps an unsupported
layer to `mxfp8` fails when the model is built.

`nvfp4` and `nvfp4_mse` are the two recipes of the Wall-X
NVFP4 service (`nvfp4_1d1d_16bs` for the ViT, `nvfp4_mse_1d1d_16bs` for
expert 0). They are two methods rather than one method with an option
because options apply per method and that recipe uses both in one model. They
have no options. Activations always use amax scales. Input and output widths
must be multiples of 16.

## Fusable layer forms

Layers request the form of each projection; the method decides what to fuse.
An unquantized method runs the plain composition, which is the eager
numerics, so a BF16 model is unchanged by the request.

| Layer call | Computes | `mxfp8` activation quantization | `mxfp8` GEMM |
|---|---|---|---|
| `linear(x)` | `x W^T + b` | `mxfp8_quantize` | `mxfp8_gemm` (bias in the epilogue) |
| `linear(x, norm=n)` | `n(x) W^T + b` | `mxfp8_rmsnorm_quantize`: RMSNorm in FP32, quantized without a BF16 rounding | as above |
| `linear(gu, input_act="silu_and_mul")` | `(silu(g) * u) W^T + b` | `mxfp8_silu_mul_quantize`: computed in FP32, no BF16 rounding | as above |
| `linear(x, residual=r)` | `r + x W^T + b` | as above | `mxfp8_gemm_residual`: `r` added to the FP32 accumulator, one rounding |
| `QKVLinear(x, rope, norm=n)` | rotated q, k and v | as above | `mxfp8_gemm_rope`: bias and the rotary embedding applied to the FP32 accumulator, with FP32 cos/sin |
| `GatedLinear(x, norm=n)` | `silu(g) * u` of `[g \| u]` | as above | `mxfp8_gemm_silu_mul`: on the FP32 accumulator, one rounding |

- `QKVLinear` and `GatedLinear` weights are stored with their rows in the order the fused epilogues read (`rope_pair_permutation`, `gate_up_interleave`). Calling such a layer through the plain form raises.
- `GatedMLP(act_fused_into="gate_up" | "down")` places the gated activation in the gate/up epilogue or in the down projection's input quantization.
- `RopeTables` (from `MRope.tables`) hold one forward's FP32 cos/sin. They serve both the eager rotation, with cos/sin rounded to the activation dtype, and the fused kernels' packed table, with unrounded values.

These forms match the fusions of the Wall-X 4B MXFP8 service (ViT down via `silu_mul_quantize`, the whole expert-0 layer fused). No model preset uses them now; `tests/unit/test_mxfp8.py` covers them.

`nvfp4` and `nvfp4_mse` fuse only into the activation quantization, as the
Wall-X NVFP4 service did:

| Layer call | `nvfp4` / `nvfp4_mse` activation quantization |
|---|---|
| `linear(x)` | `nvfp4_quantize` |
| `linear(x, norm=n)` | `nvfp4_rmsnorm_quantize`: `n(x)` rounded to BF16 first, as the Wall-X kernel does |
| `linear(gu, input_act="silu_and_mul")` | `nvfp4_silu_mul_quantize`: computed in FP32, no BF16 rounding |

The GEMM is always `nvfp4_gemm` with BF16 output. Everything else runs the
plain composition on that output: the residual add, the eager rotary of
`QKVLinear` and `silu_and_mul` of `GatedLinear`. Weight rows keep their
checkpoint order. The service computed the expert-0 gated activation in the
down projection's input quantization, which is `GatedMLP(act_fused_into="down")`.

The WALL-OSS-1.0-Nano stages call NVFP4 forms of the public GEMM family directly instead of linear layers. They take
`(q, scales)` activations and `(q, scales, global_scale)` weights. The GEMM rounds `g · x̂·Ŵᵀ + bias` once to BF16,
the epilogue works on that BF16 tensor and rounds again, and a quantized output is `nvfp4_quantize` of the final BF16
tensor, so every NVFP4 GEMM input is a BF16 tensor quantized with the activation recipe:

| Operator | Computes |
|---|---|
| `nvfp4_gemm_epilogue` | `bf16(g · x̂·Ŵᵀ + bias)`, then `bf16(act(·))`; `bf16(y + residual)` with a residual, or quantized (`out_quant`) |
| `nvfp4_gemm_glu_quantize` | `bf16(silu(y_gate) · y_up)` of the BF16 GEMM `y`, quantized |
| `nvfp4_qkv_rope_epilogue` | `[Q \| K \| V] = bf16(g · x̂·Ŵᵀ + bias)` with rotate-half RoPE on the BF16 Q and K heads, rounded again |
| `nvfp4_layer_norm_quantize` | the BF16 LayerNorm (`layer_norm_cast`), quantized |
| `nvfp4_add_zero_centered_rms_norm_quantize` | `add_zero_centered_rms_norm`'s BF16 sum, and its BF16 norm quantized |

## MXFP8 format

The format is defined by the torch reference in `xinfer/ops/reference/mxfp8.py`, with the scale layout and weight row orders in `xinfer/ops/layout/mxfp8.py`, and follows xcompute:
- **Elements**: E4M3.
- **Scales**: one E8M0 byte per 32 columns, with `share_scale_128` replicating one scale over 128 columns.
- **Scale value**: the smallest power of two that maps the block's amax into ±448.
- **Rounding**: round to nearest even after clamping.
- **Scale storage**: the SWIZZLE_32_4_4 layout (rows padded to 128, blocks to 4).

The fused references follow the kernels' arithmetic where it is specified:
- `rmsnorm_quantize` sums squares in the kernel's order (128 per-thread partial sums, then a tree) and computes `(x * rsqrt(mean + eps)) * w`.
- The GEMM references multiply dequantized operands in FP32 and apply the epilogue in FP32.

## NVFP4 format

The format is defined by the torch reference in `xinfer/ops/reference/nvfp4.py`, with the scale layout in `xinfer/ops/layout/nvfp4.py`, and follows the Wall-X NVFP4 kernels of
the production service and the GEMMs they feed:
- **Elements**: E2M1 (0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives), two per byte as uint8 `[M, K/2]`: column `2j` in the low nibble, `2j + 1` in the high nibble, bit 3 the sign.
- **Scales**: one FP8 E4M3 scale per 16 columns, stored as `float8_e4m3fn` in the SWIZZLE_32_4_4 layout (rows padded to 128, 16-column blocks to 4, padding zero).
- **Global scale**: `nvfp4_mse` weights carry a per-tensor FP32 scale `g = amax / (448 · 6)` of the whole weight (computed in double precision, rounded to FP32; 1 for an all-zero weight). A value is `e2m1 · scale · g`; activations and `nvfp4` weights have `g = 1`.
- **Scale value (amax)**: `e4m3(clamp(amax / (6g), 2^-6, 448))`, rounded to nearest even. Blocks with amax below `6 · 2^-6` get the floor scale `2^-6`.
- **Scale value (MSE, `nvfp4_mse` weights)**: each of the 126 positive finite E4M3 values `v`, in increasing order, reconstructs the block as `rn(x / (v · g)) · (v · g)`; the squared errors are summed in the kernel's warp-reduction order (column `i` plus `i + 8`, then the halves of those 8 sums, and so on), and the first smallest sum wins.
- **Rounding**: an element is `x / (scale · g)` in FP32, rounded to the nearest E2M1 value with ties to the even code, saturated to ±6; a negative value that rounds to zero keeps its sign bit.

The fused references follow the kernels' arithmetic:
- `rmsnorm_quantize` sums squares in the kernel's order (256 per-thread partial sums, then a tree), computes `x · (rsqrt(mean + eps) · w)` with the norm weight rounded to BF16, and rounds that to BF16 before quantizing.
- `silu_mul_quantize` computes `silu(gate) · up` in FP32 and quantizes it without a BF16 rounding.
- `gemm` multiplies dequantized operands in FP32, then applies `alpha` (the global scale) and the bias to the accumulator and rounds once to BF16. `nvfp4_mse` passes no bias and adds it to the BF16 output, as the service's CUTLASS kernel with the global scale had no bias.

The reference divides exactly. The Wall-X kernels are built with fast-math (approximate division, `rsqrtf`,
`__expf`): where an exact quotient is a rounding tie or within a few ulps of one, they may pick the neighbouring
scale or element.

## Backends

PyTorch provides reference operators for MXFP8 and NVFP4. On CUDA (SM120), the kernel library accelerates the WALL-OSS-1.0-Nano MXFP8 and NVFP4 presets. Builds without NVFP4 support use the reference operators. See the [kernel guide](kernels.md) to build it and check the selected operators.

## Tests

| Test | Checks |
|---|---|
| `tests/unit/test_mxfp8.py` | reference format, error bounds and fused references (CPU); fusable forms: BF16 equals the eager composition bitwise, MXFP8 tracks BF16 (CPU) |
| `tests/unit/test_nvfp4.py` | reference format against hand-computed codes and scales, MSE scale search, fused references, GEMM epilogues (CPU); `nvfp4` and `nvfp4_mse` layers and their fusable forms (CPU); the GEMM-family forms against their definitions (CPU); reference GEMM vs `torch._scaled_mm` NVFP4, the GEMM the Wall-X service ran for the shapes without a CUTLASS kernel (GPU, sm100 or newer); public CUDA NVFP4 operators against the references at WALL-OSS-1.0-Nano's shapes (GPU) |

Quantization changes are checked as in [precision](precision.md): agreement with the reference composition, and the internal precision gate.
