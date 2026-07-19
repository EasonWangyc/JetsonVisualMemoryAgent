# 项目学习与完成路线

本文面向刚拿到 Jetson Orin Nano、准备继续完成本项目的开发者，目标是给出从理解代码、复现基线，到训练、量化和板端部署的完整路线。

项目当前真实链路是：CLIP 全局图像 embedding、余弦检索、SmolVLM 单图推理和结构化 JSON。视频仅按间隔抽帧，episode replay 只是离线 VLM planner。语义分割、时序视频理解、模型训练、INT8 量化、TensorRT 运行时和真实 VLA 控制均不能表述为已完成能力。

## 总体路线

```text
理解代码与模型
  -> hash/mock 冒烟测试
  -> CLIP/SmolVLM 真实基线
  -> 固定数据集与评测口径
  -> CLIP ONNX/TensorRT FP16
  -> CLIP INT8 PTQ
  -> Edge-LLM VLM FP16
  -> LoRA/SFT（有明确收益时再做）
  -> VLM INT8/INT4
  -> C++ runtime 与最终板端报告
```

按业余时间估算，整个项目约需 2～3 个月；集中开发约需 6～9 周。实际推进以每个阶段的退出条件为准，不以时间代替完成证据。

## 明日学习安排

如果公司电脑不适合安装开发环境或下载模型，只做阅读、画图和知识整理。未经公司允许，不下载模型权重、不运行训练、不安装系统依赖。

### 上午：理解项目链路

建议按以下顺序阅读：

1. [项目 README](../README.md)
2. [系统架构](development/architecture.md)
3. [`apps/agent_cli.py`](../apps/agent_cli.py)
4. [`jetson_visual_memory/encoders.py`](../jetson_visual_memory/encoders.py)
5. [`jetson_visual_memory/memory_index.py`](../jetson_visual_memory/memory_index.py)
6. [`jetson_visual_memory/vlm.py`](../jetson_visual_memory/vlm.py)
7. [`jetson_visual_memory/schema.py`](../jetson_visual_memory/schema.py)
8. [`jetson_visual_memory/metrics.py`](../jetson_visual_memory/metrics.py)
9. [`tools/benchmark.py`](../tools/benchmark.py)
10. [`jetson_visual_memory/backends.py`](../jetson_visual_memory/backends.py)

阅读时回答以下问题：

- 图片在哪里被发现和读取？
- 图片与文字怎样变成 embedding？
- top-k 图片怎样被选出来？
- VLM 如何读取候选图片？
- 模型输出怎样变成合法的 AgentResult JSON？
- 哪些后端是真实模型，哪些只是冒烟测试？

自己画一张数据流图：

```text
用户 query -> CLIP text encoder -> text embedding -----+
                                                        |
图片 -> CLIP image encoder -> image embeddings          |
                                                        v
                                               cosine similarity
                                                        |
                                                        v
                                                   top-k 图片
                                                        |
                                                        v
                                                     VLM
                                                        |
                                                        v
                                               AgentResult JSON
```

需要明确模块职责：

- CLIP 负责从图片集合中检索语义相关候选。
- VLM 负责分析一张候选图片并生成结构化建议。
- Schema 负责约束和校验模型输出。
- Agent 只产生白名单建议，不执行 shell 或机器人动作。

### 下午：掌握核心知识

#### CLIP 与向量检索

需要掌握：

- image encoder 与 text encoder
- embedding 和向量空间
- L2 normalization
- cosine similarity
- top-k 检索
- Recall@1、Recall@5
- 正样本、负样本和 hard negative
- embedding 漂移和排名翻转
- 为什么建库和查询必须使用相同模型

余弦相似度为：

```text
cosine_similarity(a, b) = (a · b) / (||a|| ||b||)
```

本项目会先将 embedding 归一化，因此检索可以使用矩阵乘法：

```text
scores = image_embeddings @ text_embedding
```

CLIP 提供全局语义相关性，不提供像素级分割、目标边界或可靠的抓取位姿。详细设计见 [CLIP / ViT 视觉记忆设计](development/clip_memory_design.md)。

#### VLM 基础

典型 VLM 链路：

```text
图片 -> vision encoder -> multimodal projector
    -> language model -> token generation
```

需要掌握：

- vision tower
- multimodal projector
- tokenizer 和 chat template
- autoregressive generation
- prefill 和 decode
- KV cache 和 context length
- tokens/s
- hallucination
- structured output

SmolVLM 是当前 Transformers 功能基线，不是已经完成的 TensorRT 部署模型。

#### ONNX 与 TensorRT

理解以下转换链：

```text
PyTorch checkpoint
  -> ONNX graph
  -> ONNX Runtime 验证
  -> TensorRT engine
  -> CUDA/TensorRT runtime
```

需要掌握：

- ONNX opset
- 静态 shape 与动态 shape
- TensorRT binding
- host/device memory
- CUDA stream
- engine 与硬件、JetPack、TensorRT 版本的关系
- 预处理一致性

engine 构建成功不等于完成部署。只有真实输入经过 TensorRT runtime，并进入应用主链路，才能表述为部署完成。框架选型见 [推理框架选型说明](development/serving_framework_notes.md)。

#### 量化基础

| 精度 | 特点 | 本项目用途 |
| --- | --- | --- |
| FP32 | 精度基线，内存和计算开销较大 | PyTorch/ONNX 基线 |
| FP16 | 通常精度损失较小 | TensorRT 与 Edge-LLM 第一部署基线 |
| INT8 | 需要代表性校准数据 | CLIP PTQ、VLM SmoothQuant |
| INT4 | 内存占用更低，精度风险更高 | VLM AWQ |

需要掌握：

- PTQ 与 QAT
- calibration set
- activation range
- per-tensor 与 per-channel
- symmetric 与 asymmetric quantization
- SmoothQuant
- AWQ
- 量化误差和离群值

本项目首先对 CLIP image encoder 做 INT8 PTQ，不重新训练 CLIP。

#### LoRA 与 SFT

需要掌握：

- SFT 监督微调
- LoRA 的 rank、alpha、dropout 和 target modules
- vision tower 冻结策略
- multimodal projector
- train/validation/test 切分
- 过拟合和数据泄漏
- adapter merge 与 runtime adapter

训练决策顺序：

```text
确认基础模型的稳定失败模式
  -> 判断 prompt/schema 是否能解决
  -> prompt 无法解决
  -> 构造高质量数据并进行 LoRA/SFT
```

不要为了展示“训练能力”而训练没有明确目标的数据集。

### 明日学习产出

在一天结束前整理以下四份笔记：

1. 一张完整项目数据流图。
2. 一张 FP32、FP16、INT8、INT4 对比表。
3. 一份“当前已实现/尚未实现”清单。
4. 一份按下文 M0～M9 排列的任务 checklist。

## M0：建立开发与硬件基线

预计 1～2 天。

### 任务

- 固定并记录 JetPack、Jetson Linux、CUDA、TensorRT 和 Python 版本。
- 安装与 JetPack 匹配的 NVIDIA PyTorch，禁止默认使用普通 PyPI torch wheel。
- 验证 CUDA、cuDNN 和 PyTorch GPU 可用性。
- 记录 `nvpmodel` 功耗模式、散热条件和存储空间。
- 准备 30～100 张本地测试图片。
- 跑完单元测试和 hash/mock 冒烟测试。

### 需要掌握

- Python 虚拟环境和 system-site-packages
- JetPack、CUDA、cuDNN、TensorRT 的关系
- `nvpmodel`、`tegrastats`
- Linux 内存、swap、进程和文件系统

### 退出条件

- 无硬件单元测试全部通过。
- hash/mock 链路稳定输出合法 JSON。
- 环境报告包含可复现的软件和硬件信息。

环境和依赖操作见 [Jetson 部署与验证](ops/jetson-deployment.md)。

## M1：复现真实模型基线

预计 2～4 天。

### 任务

- 在 PC 和 Jetson 上显式运行 CLIP。
- 使用独立目录构建 CLIP 索引，不能复用 hash 索引。
- 人工检查文本—图片 top-k 检索结果。
- 单独运行 SmolVLM，确认图片确实进入模型。
- 运行 CLIP 检索到 SmolVLM JSON 的完整链路。
- 保存错误检索、视觉误判、幻觉和 JSON 失败样例。

### 需要掌握

- Transformers 模型和 processor 加载
- CPU/GPU device 与 dtype
- Hugging Face 模型缓存
- 图像预处理
- tokenizer 和 generation
- CUDA OOM 基础排查

### 退出条件

- CLIP 与 SmolVLM 均显式使用真实后端。
- 禁止用 `auto` 静默回退结果作为真实模型证据。
- 至少收集一批能够复现的真实失败样例。

## M2：建立正式数据集与评测集

预计 3～5 天。

至少建立三类数据：

```text
retrieval_test
  图片 + 文本 query + relevant image IDs

vlm_test
  图片 + query + 期望结构化字段

calibration
  代表性图片/文本，只用于量化校准
```

数据要求：

- 使用 manifest 固定文件、标签和数据版本。
- 校准集与测试集分离。
- 视频帧按原视频或 episode 分组切分，避免相邻帧泄漏。
- 保留正常、异常、暗光、模糊、遮挡和 hard negative 场景。
- 私有媒体和标注不得提交到仓库。

CLIP 指标：

- Recall@1、Recall@5
- embedding cosine drift
- top-k 排名翻转率
- 编码和检索延迟

VLM 指标：

- JSON schema 有效率
- risk level、action type 和 target object 正确率
- safety constraints 覆盖率
- 幻觉和失败样例数量

系统指标：

- 冷启动
- warm p50/p90/p99
- vision、prefill、decode 分阶段延迟
- tokenizer 真实 tokens/s
- 峰值内存
- 功耗、温度和降频状态

### 退出条件

- 数据集有固定 manifest、版本和切分策略。
- 任意模型和精度都能使用同一测试集进行公平比较。

## M3：修正 benchmark 口径

预计 3～5 天。

当前需要补齐：

- 使用 tokenizer 统计真实生成 token 数。
- 区分当前 RSS 与真正的峰值内存。
- 连续采集并解析 `tegrastats`。
- 增加 warmup，区分冷启动和热运行。
- 在索引中保存 encoder 和 model 身份。
- 拒绝使用不兼容的旧索引。
- 报告模型、精度、输入尺寸、功耗模式和校准集。
- 保存 failed/skipped 的可复现原因。

建议测试口径：

```text
cold start：独立进程运行 5～10 次
warmup：5～10 次
warm benchmark：至少 100 次
```

30 次可用于观察趋势，但不足以稳定估计 p99。测试规则见 [测试指南](development/testing.md)。

### 退出条件

- 指标字段含义与实际采集行为一致。
- 同一命令在相同环境中可以复现近似结果。
- 未执行的后端明确标记为 skipped，而不是 passed。

## M4：CLIP ONNX 与 TensorRT FP16

预计 1 周。

实施顺序：

```text
PyTorch FP32 embedding
  -> 导出 ONNX
  -> ONNX Runtime embedding
  -> 数值一致性测试
  -> TensorRT FP16 engine
  -> TensorRT runtime
  -> 接入 MemoryIndex
```

需要实现：

- 与 Hugging Face CLIP 一致的 resize、crop、RGB、normalize。
- `OnnxClipImageBackend.encode_images()`。
- TensorRT engine 反序列化和 binding 管理。
- host/device buffer 与 CUDA stream。
- TensorRT image backend。
- batch=1 和约定的动态 batch。
- PyTorch、ONNX、TensorRT embedding 一致性测试。
- backend 状态和异常回归测试。

当前优先完成 image encoder。动态自然语言查询仍需要 text encoder；第一版可保留 PyTorch text encoder。若要实现完全无 PyTorch 的 C++ 检索链路，后续还需部署 text encoder，或者把查询限制为预定义文本。

### 退出条件

- ONNX 和 TensorRT 都处理真实图片，而不是只加载 artifact。
- TensorRT embedding 被真实 MemoryIndex 消费。
- FP16 Recall@K 与 FP32 基线差异可量化、可解释。
- 不能仅凭 engine 文件表述为部署完成。

## M5：CLIP INT8 PTQ

预计 1 周。

### 任务

- 固定代表性 calibration manifest。
- 实现 TensorRT INT8 calibrator。
- 保存 calibration cache 和校准配置。
- 构建 INT8 engine。
- 比较 FP32、FP16 和 INT8。

### 比较指标

- Recall@1、Recall@5
- embedding cosine drift
- top-k 排名翻转率
- 冷启动和 warm p50/p90/p99
- 峰值内存
- 平均和峰值功耗
- 温度和降频
- engine 大小及构建时间

### 退出条件

- INT8 Recall@K 的变化在项目约定的质量门槛内。
- embedding 漂移有分布统计，而不是只有平均值。
- INT8 runtime 被真实查询链路消费。

## M6：选择并部署 Edge-LLM VLM FP16

预计 1～2 周。

SmolVLM 保留为 Transformers 基线。Edge-LLM 主线选择其官方支持的轻量 VLM，优先评估 1B 级 InternVL，再根据内存和质量尝试 Qwen3-VL-2B。实际模型和精度组合必须在开始实施时重新核对官方支持列表。

官方资料：

- [TensorRT Edge-LLM 支持模型](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)
- [TensorRT Edge-LLM 安装](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)
- [TensorRT Edge-LLM VLM 示例](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/examples/vlm.html)

推荐工作流：

```text
x86 NVIDIA GPU 主机
  -> checkpoint 准备、量化、ONNX 导出

Jetson
  -> TensorRT engine 构建、C++ runtime 推理
```

### 退出条件

- VLM FP16 runtime 确实读取真实图片。
- C++ runtime 能输出文本结果。
- 输出能适配当前 AgentResult schema。
- 固定任务集上有质量和性能报告。

## M7：LoRA/SFT

预计 1～2 周，只在基础模型存在稳定且可重复的任务缺陷时执行。

建议训练样本：

```json
{
  "image": "relative/path.jpg",
  "task": "risk_assessment",
  "query": "判断是否适合机器人操作",
  "answer": {
    "scene_summary": "...",
    "risk_level": "high",
    "suggested_action": "...",
    "action_type": "suggest_robot_action",
    "preconditions": [],
    "safety_constraints": []
  }
}
```

建议第一轮：

- 使用数百到两千条高质量样本验证方向。
- 冻结 vision tower。
- 优先对语言主干做 LoRA。
- 必要时单独实验 multimodal projector。
- 在 PC/x86 GPU 上训练，不在 Jetson 上训练。
- 比较 base、prompt-only 和 LoRA 三组结果。
- 固定随机种子、数据版本和训练配置。

### 退出条件

- LoRA 在独立测试集上显著优于基础模型。
- JSON 合法率、风险判断或动作字段有可重复提升。
- 没有明显破坏通用视觉理解能力。
- adapter 能合并或被目标 runtime 正确加载。

## M8：VLM INT8/INT4

预计 1 周。

Jetson Orin 当前主线：

```text
FP16
  -> INT8 SmoothQuant
  -> INT4 AWQ
```

精度支持随 Edge-LLM 版本变化，实施前必须检查其安装和量化文档：

- [Edge-LLM 精度与安装要求](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html)
- [Edge-LLM 量化说明](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html)

需要比较：

- JSON schema 有效率
- 固定任务正确率
- vision、prefill、decode 延迟
- 真实 tokens/s
- KV cache 和峰值内存
- 功耗、温度和降频
- FP16/INT8/INT4 失败样例

### 退出条件

- 至少一种精度能在 8GB 板端稳定运行。
- 质量下降和性能收益均有量化数据。
- 完成 checkpoint、导出、engine、runtime 和应用接入，不把单独的量化文件当作部署完成。

## M9：完整 C++ 部署闭环

预计 1 周。

目标结构：

```text
输入图片/视频抽帧
  -> TensorRT CLIP image encoder
  -> MemoryIndex
  -> CLIP text encoder
  -> top-k 图片
  -> Edge-LLM C++ VLM runtime
  -> JSON parser
  -> schema validator
  -> 白名单 action suggestion
```

需要掌握：

- CMake
- TensorRT C++ API
- CUDA memory 生命周期
- JSON 解析
- Python/C++ 边界
- 进程内调用、CLI 或 IPC
- 错误码和 backend status
- artifact 与版本 manifest

### 退出条件

- 不依赖 hash/mock。
- 不依赖手工拼接中间结果。
- 一条命令完成真实图片到结构化 JSON。
- 模型缺失、OOM、runtime 失败和 JSON 错误均有明确状态。
- VLM 输出只作为白名单建议，不直接执行机器人控制。

## 最终报告要求

每个模型和精度组合至少记录：

- JetPack、Jetson Linux、CUDA、TensorRT
- 模型 checkpoint 和版本
- FP32/FP16/INT8/INT4
- 输入尺寸和 batch
- calibration set 和测试集版本
- 功耗模式和散热条件
- 冷启动和 warm p50/p90/p99
- vision、prefill、decode 延迟
- tokenizer 真实 tokens/s
- 峰值内存
- 功耗、温度和降频
- Recall@K 或任务正确率
- JSON 合法率
- 失败样例

最终至少形成三张对照表：

1. CLIP FP32、FP16、INT8。
2. SmolVLM Transformers 与 Edge-LLM VLM。
3. VLM FP16、INT8、INT4。

索引、权重、ONNX、TensorRT engine、私有媒体和生成报告不提交到仓库。需要分享时，只提交脱敏的实验方法和汇总数据。

## 暂不作为主线的内容

以下内容当前不是阻塞项：

- 语义分割和目标检测训练
- 时序视频理解模型
- 强化学习
- 真实 VLA checkpoint 和闭环控制
- vLLM 高并发服务
- TVM/MLIR 编译器
- QAT
- 多机多卡训练

当前核心始终是：

```text
CLIP 语义检索
+ VLM 结构化推理
+ TensorRT/Edge-LLM 部署
+ 可复现 benchmark
```

任何阶段都必须具备明确输入、真实 runtime、量化前后质量对比和可复现报告，不能用“脚本存在”代替“能力完成”。
