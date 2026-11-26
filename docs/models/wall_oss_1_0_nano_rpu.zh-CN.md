[English](wall_oss_1_0_nano_rpu.md) | 简体中文

# WALL-OSS-1.0-Nano 的 RPU TP8 配置

当设备为 `rpu` 或 `rpu:0` 时，模型 ID `wall-oss-1.0-nano` 会选择 `wall-oss-1.0-nano-fp16-r1` 配置。
其他设备继续使用 GPU WALL-OSS-1.0-Nano 实现及其 checkpoint 格式。RPU 配置没有
CPU、BF16 或 MXFP8 执行回退。

此配置通过 Rhino Launch SDK 使用 Rhino RPU (R1)。SDK 单独安装，原生算子后端与
外部 `.ref` 资产从 `xinfer/ops/backends/rpu/src` 构建（详见[原生 RPU 算子](../rpu_ops.zh-CN.md)）。
实际执行需要匹配的 RPU 算子后端、规格定义、注册表注册、运行时共享库及编译后的 `.ref` 资产。

执行属于实验特性，同时需要显式的候选准入和匹配的 AOT `xinfer_rpu.so` 库及外部算子资产。
主机侧模型计划、处理与生命周期契约已经实现；用于生产前仍需完成板级数值、
ABI 和时延验收。

## 执行配置

| 属性 | 契约 |
| --- | --- |
| 设备 | 一个 RPU，物理核心 0–7；仅接受 `rpu:0` |
| 组批 | B1，一个准入请求，每阶段一个通道（lane） |
| 动作 | FP32 物理 relative26 `[1,16,26]`；8 步去噪，不使用 RTC |
| 图像 | 三个规范相机，使用固定的缩放、patch 打包和 M-RoPE 规则 |
| 执行 | SDK 准备的有序执行（ordered execution），显式等待完成 |
| 中间数据 | 原生常驻缓冲区；最终输出复制到独立拥有的 CPU 存储 |

`torch.device` 不能描述此设备，RPU 路径不使用 CUDA graph、Torch
stream/event 或通用 CUDA `Engine`。执行使用进程内 `Policy` 接口。
下述冻结请求格式用于说明待检查的输入，不是额外的执行命令；此配置不为
`xinfer-serve` 增加 RPU 服务模式。

模型配置提供 `Policy` 用于配置解析和 tuned 表选择的 FP16 默认精度。RPU 配置适配器提供单请求、
单通道默认值；B1 来自模型声明，流优先级支持情况来自平台声明。此引擎不生成
负载剖析所需的准入和阶段 span，因此不支持 `xinfer-profile load`。

## 无板卡的 checkpoint 检查

安装模型处理依赖，例如在通过 `python -m pip install -e ".[models]"`
准备的环境中。预检查不加载张量或原生算子库，即可检查 checkpoint 元数据、
配置和资产哈希：

```python
from xinfer.models.wall_oss_1_0_nano.rpu.config import preflight_checkpoint

manifest = preflight_checkpoint(checkpoint_path)
print(manifest)
```

检查冻结请求时，提供不使用 pickle 的 NPZ，其中 `face_view`、
`left_wrist_view` 和 `right_wrist_view` 是 RGB uint8 HWC 数组，`state`
形状为 `[26]`，`instruction` 是 Unicode 标量，`initial_noise` 是形状
`[1,16,26]` 的 FP32 数组。必须提供实际噪声数组；只有种子不能保证输入字节相同。
请求中的视图、状态和显式初始噪声随后交给与服务路径相同的输入处理器。

视频解码、任务评估和 ground-truth 评分不在此接口的范围内。

## 显式候选准入

创建实际执行上下文前，先针对 Rhino Launch SDK 构建 AOT 算子库。仅含主机
resolver 的库不能执行请求。在启动进程前选择运行时库及单独的算子资产：

```bash
export XINFER_RPU_LIBRARY=/path/to/xinfer_rpu.so
export XINFER_RPU_ASSETS=/path/to/rpu-assets
```

provider 检查 ABI、catalog 指纹及资产 manifest 哈希。先取得部署环境的板卡租约，
并检查设备占用情况。导入 Python 模块不会触发库或设备算子的编译。

执行可通过 `Policy` 获得。其上下文管理器会在推理抛出异常时仍关闭
引擎和输入处理器：

```python
from xinfer import Policy

with Policy("wall-oss-1.0-nano", checkpoint, device="rpu:0", quantization="fp16",
            allow_candidate=True) as policy:
    result = policy.infer(observation)
```

执行不会隐式启用候选能力。如果后端没有实现选定的 family、selector、
集体窗口或融合队列，则准入失败，而不会静默替换为另一种执行计划。

RPU 与 CPU、CUDA、Ascend 使用同一注册表和 `build_table`。匹配的仓内算子包
注册 `prepare_<family>` 调用，例如 `prepare_gemm` 和 `prepare_attention`，
使用常驻缓冲区签名并报告 `numerics="unqualified"`。这些调用没有 Torch
参考实现；固定选择不受支持或设置 `reference_ops=True` 时显式报错。
空的 RPU 选择在绑定时报错；缺少某个调用时，在准备该操作时报错。

工厂向平台上下文注入 `rpu.bind_ops(model.ops)`。模型计划继续调用
`context.prepare(...)`，由表中选中的调用完成原生准备。family selector
原样传给原生 resolver。批执行和完成等待保持不变，注册表选择不构成板级数值验收。

## 模型选项

RPU 模型只接受 `quantization="fp16"` 并要求 `allow_candidate=True`。
这些选项与公开的 `image_preprocess_device` 设置相互独立：RPU 输入处理
始终在 CPU 上执行。

| 选项 | 默认值 | 契约 |
| --- | --- | --- |
| `image_workers` | `6` | 输入处理器的相机池大小 |
| `audit_inputs` | `False` | 保留额外的大输入哈希 |
| `max_cached_signatures` | `2` | 缓存的 prefix 签名数，从 1 到 2 |
| `prefill_collective_window` | `1` | 请求的 AR 集体窗口，从 1 到 15 |
| `action_collective_window` | `1` | 请求的 M16 集体窗口，1 或 4 |
| `action_whole_stage` | `False` | 准备一个整体融合 Action 阶段 |
| `action_fused_steps` | `False` | 把 Action 入口折叠到其阶段准备中 |
| `vision_whole_stage` | `False` | 准备一个整体融合 Vision 阶段 |
| `vision_fused_stage` | `False` | 在不启用 whole-stage 选项时融合 Vision 阶段 |
| `prefill_whole_stage` | `False` | 准备一个整体融合 Prefill 阶段 |

非默认集体和融合模式要求对应的原生能力。互斥的组合会在读取 checkpoint
之前被拒绝。窗口设置跨普通请求和显式 reset 保留，不会静默降低。

例如：

```python
with Policy("wall-oss-1.0-nano", checkpoint, device="rpu:0", quantization="fp16",
            allow_candidate=True, prefill_collective_window=4) as policy:
    result = policy.infer(observation)
```

## 处理与权重

输入处理器保留 checkpoint 的两步图像缩放、相机顺序、patch 布局、状态 token
提示、flow token 和多模态 RoPE 构造。`audit_inputs=True` 记录额外的输入哈希。
通过 `Policy.close()` 关闭处理器以排空相机池；取消在提交相机作业前注册，
相机处理失败会在请求清理前排空已经提交的工作。

`preflight_checkpoint` 检查 checkpoint 的完整配置、资产和张量清单。随后
`WeightPreparer` 生成主机 FP16 权重记录，交由原生打包和不可变设备分配。
在第一个请求之前准备全部权重和常驻计划；计划构造不得执行推进状态的预热。
普通请求绑定自己的显式输入并重放准备好的阶段，不重新准备队列。

阶段输出留在原生所有者中。最终归一化的 Action 结果会物化为独立的 CPU
存储，并在模型的收集回调中反归一化；输出处理器返回这些物理结果。
相对 relative26 和 absolute14 的几何
工具可供评估使用；ground-truth 评分不计入策略计时。

## 上游来源

checkpoint 语义、分带图像流水线和评估几何处理改编自公开可访问的
RhinoForge Wall Qwen3.5 实现：

- [配置](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/python/rpu_backend/adapters/wall_qwen35/profile.py)及 [checkpoint 准入](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/python/rpu_backend/adapters/wall_qwen35/checkpoint.py)。
- [相机预处理](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/python/rpu_backend/adapters/wall_qwen35/preprocessing.py)。
- [评估几何辅助函数](https://github.com/HUIXI-AI/RhinoForge/blob/2535a27756486bdb139b06764975edbe00004d63/examples/wall_qwen35_openloop.py)。

## 失败与生命周期所有权

RPU 引擎一次准入一个请求。工作线程开始前可以取消；设备提交后，关闭和结果
处理都要等待实际完成，因为公开 TP ABI 没有非阻塞完成查询或执行中途取消。

准备、执行或收集错误会阻止后续请求，直到显式 reset。回滚只释放请求绑定并将
计划标记为失效；完整计划仅在显式 reset 时重建。模型计划不会重放部分执行的工作。
只能在引擎空闲且未关闭时调用 reset；它使状态代际失效、重置
原生状态所有者并重建计划，同时保留已打包权重。如果 reset 失败，引擎继续
保持阻塞。恢复及回退到其他运行时都不是自动的。

请求结果统计包含绑定清理。只有输出收集和绑定释放都成功，请求才计为完成。
释放失败会计为一次失败，并要求 reset。

`Policy.close()` 排空未完成的引擎工作，再关闭模型的执行资源和输入处理器。
`RpuEngine.shutdown()` 执行相同的排空及模型/上下文清理；外部提供的执行
上下文保留其原有资源。重复关闭是安全的。清理失败不会把释放失败的原生
所有者标记为可用上下文。Policy 上下文退出时若已有异常，清理操作会保留
原异常。

`max_cached_signatures` 限制 prefix 计划数量。请求拥有的资源在完成后释放，
不可变权重和缓存的准备批次保留到模型关闭。返回的 CPU 数组在后续请求、
reset 和关闭之后仍然有效。

## 验证与限制

`test_rpu_runtime.py` 的主机生命周期契约使用原生 fake，不需要板卡或
原生算子库，覆盖资源所有权、显式等待、请求准入、取消、失败、reset 和释放。
其他模型测试覆盖处理、checkpoint 清单、权重变换和 CPU 数值参考；部分计划测试
需要本仓的 host-only `rpu_tp_resolve` 库导出，缺少该导出时会跳过。

```bash
python -m pytest -q -m "not gpu and not assets" \
  tests/models/test_wall_oss_1_0_nano_action.py \
  tests/models/test_wall_oss_1_0_nano_denoise.py \
  tests/models/test_wall_oss_1_0_nano_family_calls.py \
  tests/models/test_wall_oss_1_0_nano_identity.py \
  tests/models/test_wall_oss_1_0_nano_model.py \
  tests/models/test_wall_oss_1_0_nano_position_reference.py \
  tests/models/test_wall_oss_1_0_nano_prefill.py \
  tests/models/test_wall_oss_1_0_nano_profile.py \
  tests/models/test_wall_oss_1_0_nano_rpu_processing.py \
  tests/models/test_wall_oss_1_0_nano_static_steps.py \
  tests/models/test_wall_oss_1_0_nano_vision.py \
  tests/models/test_wall_oss_1_0_nano_weights.py \
  tests/unit/test_rpu_runtime.py
```

这些检查不能验收板卡实现。完整验证必须采用实际 checkpoint、选定的 SDK/AOT
库和隔离板卡，检查库选择、导出的 ABI 符号、准备操作重放、与参考实现的输出
对齐、故障恢复以及重复关闭/重新打开行为。本次集成未声明任何完整请求的
数值或时延验收结论。
