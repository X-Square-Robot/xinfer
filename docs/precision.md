# Precision

English | [简体中文](precision.zh-CN.md)

XInfer serves each model in two compositions. The reference composition (`numerics="reference"`) follows the
original model implementation step by step on PyTorch operators. The optimized composition (the default) fuses steps
into xcompute kernels, so it rounds at other points and is not bitwise equal to the reference. XInfer states
precision as the agreement between the two, checked by the tests in this repository; the accuracy of a model itself
is that of its checkpoint.

## How agreement is checked

| Check | Test | What it compares |
|---|---|---|
| Reference composition against the original implementation | `tests/models/test_wall_oss_parity.py` | Wall-OSS's reference composition against the original eager BF16 model on recorded requests: each stage from the original's inputs to it, and the actions |
| Optimized against reference, per stage | `tests/models/test_wall_oss_optimized.py` | each stage of the optimized composition against the reference's result from the same inputs, within a tolerance; CUDA graph replays equal eager runs bit for bit |
| Structure of both compositions | `tests/models/test_wall_oss_tiny.py`, `tests/models/test_wallx2b_tiny.py` | a tiny random checkpoint on the CPU: complete weight mapping, the stages through the engine, pinned RTC actions returned exactly, the optimized composition close to the reference |
| End to end | `tests/models/test_wallx2b_policy.py` | Wall-X 2B through `Policy` with a supplied checkpoint, in both compositions |

The tests with real checkpoints need a GPU and the model's assets (the environment variables each test names); they
skip without them. The tiny-checkpoint tests run in CI. Wall-X 2B's optimized composition was also compared stage by
stage with its checkpoint's BF16 sampler on full checkpoints ([Wall-X 2B validation](models/wallx2b_validation.md#numerical-checks-against-the-bf16-sampler)).

## Known differences

- **Rounding.** The optimized composition applies epilogues (bias, RoPE, activations, residual adds) to FP32
  accumulators and rounds to BF16 once, where the reference rounds after each step.
- **Pinned RTC actions.** The served semantics cut the attention of pinned RTC action tokens to the denoising tokens
  at every step, while the original eager model lets them attend at step 0
  ([Wall-OSS design](models/wall_oss_design.md)). Both compositions serve the cut form.
- **Batching.** Batch size can change floating-point rounding; the per-stage tolerances cover single requests.

## Internal validation

Every change that can alter a model's outputs also passes an internal precision gate on real robot data before it
is released. Its data and tools are not part of this repository.

On Ascend 950DT, the xcompute implementations listed in the [Ascend guide](ascend950dt.md) passed the same internal
gate, with image processing on the CPU and on the device.
