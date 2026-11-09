# 精度门禁

[English](precision.md) | 简体中文

可能改变 Wall-OSS 输出的改动必须通过 embodied-serving 的 quick100 门禁，即 Wall-X 4B 服务上线时所用的门禁。

**规则。** 每个必选场景相对冻结的原始 BF16 模型，GT-MSE 的增幅不得超过 +1 %：`MSE_candidate / MSE_BF16 − 1 ≤ 0.01`。

**范围**（embodied-serving `docs/PRECISION_REGRESSION.md`）：
- **输入**：5 个真实 episode，100 个窗口，随机种子 42，RTC 延迟 5。
- **指标**：在归一化空间中对 18 个非夹爪维度计算 GT-MSE，不计每个窗口的前 `delay` 步。
- **参考**：原始的全 BF16 Wall-X 模型，B1，已冻结。

XInfer 通过一个对接 embodied-serving 测试框架（harness）的适配器运行该门禁（`scripts/check_precision.py`，配置 `quick100-batch32-v1`，范围 `model_batch`）。

## 工具

| 文件 | 作用 |
|---|---|
| `tools/precision/wall_oss_quick100_adapter.py` | 供测试框架使用的 `PRECISION_ADAPTER`（详见下文） |
| `tools/precision/configs/optimized.example.json` | 候选配置：optimized 组合（默认） |
| `tools/precision/quick100_b1_score.py` | 用 harness 自身的函数为保存的 harness 运行结果计算 B1 场景的分数 |

适配器对每个请求组执行以下操作：
- 对传输格式（wire）的观测运行 XInfer 的输入处理，使用该请求保存的 FP32 噪声（prompt 与服务时一样填充到 `processing.prompt_bucket`，默认 16 token）；在 `processing.image_preprocess_device="gpu"` 时，它与引擎一样在 vision 阶段之前于 GPU 上解码并 resize 各视图（`xinfer.engine.device_images`）；
- 逐请求运行 vision 和 prefill；
- 对整个组调用一次 action 阶段（B1 或该场景的 BN）。不同的 prompt 长度（例如 760 和 761 token）与引擎一样以左填充方式合并（`xinfer.engine.batching`）。optimized 组合每次 action 调用只接受一个请求：遇到组批的请求组时适配器报错，harness 在 B1 之后停止，用 `quick100_b1_score.py` 计算该运行的分数；
- 报告该次调用实际的批和延迟。

## 运行

测试框架、资产和 BF16 参考都不在本仓库中：

| 输入 | 说明 |
|---|---|
| 测试框架 | 一个处于 `fa2e8af` 且无改动的 embodied-serving 检出 |
| 资产 | quick100 导入包：manifest SHA `d2ec98c8…`，checkpoint `5_25000` 权重 SHA `37a4b2ea…` |
| BF16 参考 | 冻结的原始 BF16 输出（导入包中的 `reference1`） |
| 目标（target） | 旧 Wall-X 4B MXFP8 服务的运行结果，embodied-serving `54691d0` |

```bash
PYTHONPATH=/path/to/xinfer/tools/precision python scripts/check_precision.py \
  --assets $ASSETS --reference $ASSETS/reference1 \
  --target $TARGET_RUN --target-ref 54691d0e20b5745eae8ebb6c1f25cdf039067dff \
  --require-scope model_batch --adapter wall_oss_quick100_adapter:Adapter \
  --config optimized.json --device 0 --output /path/to/run
python /path/to/xinfer/tools/precision/quick100_b1_score.py . $ASSETS $ASSETS/reference1 $TARGET_RUN /path/to/run
```

请在测试框架的检出目录中运行。checkpoint 会对照 manifest 中的哈希进行校验，因此要使用原始的 checkpoint 目录，而不是修改过 `config.yml` 的副本。

## Wall-OSS quick100 门禁

结果（2026-11-07，RTX 5090，torch 2.10.0+cu130）：`numerics="optimized"`，使用默认后端（xcompute `651195d`、flash-attn），B1：

| 场景 | 相对 BF16 的 GT-MSE | episode 95 % CI | 相对旧服务（target） |
|---|---|---|---|
| quick100 B1，延迟 5（100 个窗口） | **−6.09 %：PASS** | [−12.07, +0.32] | +0.57 % |

- 按同样方式计算：旧的 MXFP8 服务为 −6.62 %，XInfer 参考组合（使用 eager 模型的 kernel）为 −6.28 %（torch 2.9.1），optimized 组合在 torch 2.9.1 下为 −6.25 %（CI [−12.17, −0.25]；其动作与 2.10 下的并非逐位一致，因为 torch 算子的舍入不同）。
- **原始 BF16 并不是服务的语义。** 服务和 XInfer 在每一步都切断 RTC 固定动作 token 对去噪 token 的注意力，而原始 eager 模型在第 0 步允许其注意（docs/models/wall_oss_design.md）。与原始 BF16 的差距主要来自这一差异。
- 组批场景需要每次调用接受多个请求的 action 阶段；optimized 组合只接受一个。
- GPU 测试把每个阶段与 eager BF16 模型比较，并逐位检查 graph；输出质量只由本门禁判断。

证据：`xinfer-wall-oss-optimized-20261107.tar.gz` 中的运行 `q100-wos210-run`（torch 2.10）和 `q100-wos-run`（2.9.1）（哈希见 docs/models/wall_oss_validation.md）。
