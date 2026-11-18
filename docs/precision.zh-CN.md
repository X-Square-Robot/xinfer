# 精度

[English](precision.md) | 简体中文

XInfer 为每个模型提供两种组合。参考组合（`numerics="reference"`）在 PyTorch 算子上逐步复现原始模型实现。optimized 组合（默认）把多个步骤融合进 xcompute kernel，因此舍入位置不同，与参考组合并非逐位一致。XInfer 用两者之间的一致性来表述精度，由本仓库中的测试检查；模型本身的准确度取决于其 checkpoint。

## 如何检查一致性

| 检查 | 测试 | 比较内容 |
|---|---|---|
| 参考组合对照原始实现 | `tests/models/test_wall_oss_parity.py` | 在录制的请求上，把 Wall-OSS 的参考组合与原始 eager BF16 模型比较：每个阶段从原始模型的输入出发，以及最终动作 |
| optimized 对照参考，逐阶段 | `tests/models/test_wall_oss_optimized.py` | optimized 组合的每个阶段与参考组合在相同输入下的结果比较，在容差内；CUDA graph 回放与 eager 运行逐位一致 |
| 两种组合的结构 | `tests/models/test_wall_oss_tiny.py`、`tests/models/test_wallx2b_tiny.py` | 在 CPU 上用很小的随机 checkpoint：权重映射完整，各阶段经由引擎运行，RTC 固定动作原样返回，optimized 组合接近参考组合 |
| 端到端 | `tests/models/test_wallx2b_policy.py` | 用给定的 checkpoint 经 `Policy` 运行 Wall-X 2B，两种组合都覆盖 |

使用真实 checkpoint 的测试需要 GPU 和模型资产（各测试列出所需的环境变量），缺少时跳过。小随机 checkpoint 的测试在 CI 中运行。Wall-X 2B 的 optimized 组合还在完整 checkpoint 上与其 BF16 采样器逐阶段比较过（[Wall-X 2B 验证](models/wallx2b_validation.zh-CN.md#与-bf16-采样器的数值对齐)）。

## 已知差异

- **舍入。** optimized 组合在 FP32 累加结果上做 epilogue（bias、RoPE、激活、残差加），只舍入一次到 BF16；参考组合在每一步之后舍入。
- **RTC 固定动作。** 服务的语义在每一步都切断 RTC 固定动作 token 对去噪 token 的注意力，而原始 eager 模型在第 0 步允许其注意（[Wall-OSS 设计](models/wall_oss_design.zh-CN.md)）。两种组合都按切断的形式服务。
- **组批。** 批大小会改变浮点舍入；逐阶段的容差针对单请求。

## 内部验证

每个可能改变模型输出的改动，在发布前还要通过基于真实机器人数据的内部精度门禁。它的数据和工具不在本仓库中。
