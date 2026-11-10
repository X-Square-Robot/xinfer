# Quantization

English | [简体中文](quantization.zh-CN.md)

Linear layers get their method from a `QuantConfig` (docs/design.md §12).
Models assign each linear layer a *group*; the config maps groups to
methods, and a group inherits from its dot-separated parents.

```python
from xinfer.quantization import QuantConfig

QuantConfig(groups={"vision.blocks": "mxfp8", "vision.merger": "mxfp8", "prefill": "mxfp8"})
```

After the checkpoint is loaded, `process_weights(model)` converts each
layer once, before weights are shared or graphs captured. Models may define
presets (`quantization="<preset>"`), or take a `QuantConfig` or a dict with `groups`, `default` and
`options`; the bundled models define no preset now (Wall-OSS runs BF16 only). No model preset uses `nvfp4` or
`nvfp4_mse` yet.

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

`nvfp4` and `nvfp4_mse` are the two recipes of the embodied-serving Wall-X
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

## MXFP8 format

The format is defined by the torch reference in `xinfer/ops/mxfp8.py`, which follows xcompute:
- **Elements**: E4M3.
- **Scales**: one E8M0 byte per 32 columns, with `share_scale_128` replicating one scale over 128 columns.
- **Scale value**: the smallest power of two that maps the block's amax into ±448.
- **Rounding**: round to nearest even after clamping.
- **Scale storage**: the SWIZZLE_32_4_4 layout (rows padded to 128, blocks to 4).

The fused references follow the kernels' arithmetic where it is specified:
- `rmsnorm_quantize` sums squares in the kernel's order (128 per-thread partial sums, then a tree) and computes `(x * rsqrt(mean + eps)) * w`.
- The GEMM references multiply dequantized operands in FP32 and apply the epilogue in FP32.

## NVFP4 format

The format is defined by the torch reference in `xinfer/ops/nvfp4.py`, which follows the Wall-X NVFP4 kernels of
the embodied-serving `nvfp4_v2.patch` (`wall_x/infer/csrc/nvfp4/ops.cu`) and the GEMMs they feed:
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

Without a backend the reference implementations run, in torch and slowly.
NVFP4 has only the torch reference implementation: no device backend
registers the `nvfp4_*` operators.

| Platform | Backend | Library |
|---|---|---|
| RTX 5090 (sm120) | `xinfer.ops.backends.xcompute`, all seven MXFP8 operators (table below) | a compatible MXFP8 build of xcompute (separate from the BF16 unified build) |
| Thor (sm110) | not registered yet | main-branch xcompute has SM110 MXFP8 kernels; to be validated on Thor |

| Operator | xcompute function | Measured vs reference (RTX 5090) |
|---|---|---|
| `mxfp8_quantize` | `wallx_sm120_quantize_mxfp8` | equal except a neighbouring scale in < 1 % of blocks (fast-math `amax / 448`) |
| `mxfp8_rmsnorm_quantize` | `wallx_sm120_rmsnorm_quant_mxfp8` | bitwise on the test data; contract allows rare one-step differences (fast-math `rsqrtf`); BF16 norm weight only |
| `mxfp8_silu_mul_quantize` | `wallx_sm120_silu_mul_quant_mxfp8` | bitwise on the test data; same contract (`__expf`) |
| `mxfp8_gemm` | `wallx_sm120_mxfp8_gemm_bf16` (cuBLASLt) | within a few BF16 ulps |
| `mxfp8_gemm_rope` | `wallx_sm120_mxfp8_qkv_rope` mode 0 (CUTLASS) | ≥ 99.99 % of elements bitwise, others 1 BF16 ulp |
| `mxfp8_gemm_silu_mul` | `wallx_sm120_mxfp8_qkv_rope` mode 1 (CUTLASS) | same |
| `mxfp8_gemm_residual` | `wallx_sm120_oproj_resid_add_bf16` (CUTLASS) | same; no bias |

All are registered with tolerance contracts.
- **cuBLASLt GEMM**: autotunes each shape once per process on its first eager call. All lanes of a process share the result, but different processes may pick different algorithms. Engines run every new graph key eagerly before capturing it, which satisfies the kernel's warm-before-capture requirement.
- **CUTLASS GEMMs**: fixed 128×128×128 tiles, no split-K and no workspace.

## Tests

| Test | Checks |
|---|---|
| `tests/unit/test_mxfp8.py` | reference format, error bounds and fused references (CPU); fusable forms: BF16 equals the eager composition bitwise, MXFP8 tracks BF16 (CPU); xcompute operators vs the references, graph capture (GPU, `XINFER_XCOMPUTE_LIBRARY`) |
| `tests/unit/test_nvfp4.py` | reference format against hand-computed codes and scales, MSE scale search, fused references, GEMM epilogues (CPU); `nvfp4` and `nvfp4_mse` layers and their fusable forms (CPU); reference GEMM vs `torch._scaled_mm` NVFP4, the GEMM the Wall-X service ran for the shapes without a CUTLASS kernel (GPU, sm100 or newer) |

The precision gate for quantization changes is quick100 ([precision.md](precision.md)).
