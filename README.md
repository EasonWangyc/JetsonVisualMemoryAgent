# Jetson Visual Memory Agent

## 项目定位

这是一个面向 Jetson Orin Nano 的端侧多模态训练、量化与部署实践项目。当前代码使用 CLIP 建立图像级视觉语义记忆，通过文本—图像 embedding 相似度检索候选图片，再由 SmolVLM 分析单张图片并输出结构化 action JSON。项目下一阶段将围绕 LoRA/SFT、INT8/INT4 量化、TensorRT 与 TensorRT Edge-LLM 建立可复现的板端链路和 benchmark。

```text
当前：图片/视频抽帧 -> CLIP embedding -> 余弦检索 -> SmolVLM -> action JSON
目标：训练样本 -> LoRA/SFT -> 量化 -> ONNX/TensorRT engine -> C++ runtime -> Jetson benchmark
```

项目不执行机器人动作。当前没有语义分割、时序视频模型、真实 VLA checkpoint 或闭环控制；episode replay 是 VLA 风格的离线接口实验。完整边界见 [开发架构说明](docs/development/architecture.md)。

## 能力状态

| 模块 | 当前状态 | 下一步 |
| --- | --- | --- |
| 视觉记忆 | CLIP 全局 embedding、NumPy 索引与余弦检索 | CLIP INT8 PTQ 与 TensorRT runtime |
| 多模态推理 | Transformers + `SmolVLM-500M-Instruct` | 选取 Edge-LLM 支持的轻量 VLM |
| 训练 | 尚未实现 | PC 端构造任务数据并进行 LoRA/SFT |
| 量化 | 尚未实现 | VLM INT8 SmoothQuant 与 INT4 AWQ 对照 |
| 部署 | CLIP ONNX 导出、TensorRT engine 构建脚本；运行时未闭环 | Edge-LLM C++ runtime 与真实输入绑定 |
| 视频/VLA | 固定间隔抽帧、离线 episode prompt | 暂不扩展真实控制链路 |

## 快速开始

项目要求 Python 3.10 或更高版本，并以 `uv` + `pyproject.toml` 作为依赖入口。

```text
uv sync --extra dev
uv run python -m unittest discover -s tests
uv run python tools/build_index.py --image-dir data/images --out artifacts/index --encoder hash
uv run python apps/agent_cli.py --query "找出不适合机器人操作的场景" --encoder hash --model mock --top-k 3
```

`hash` 编码器和 `mock` VLM 不读取真实视觉语义，只适合验证流程接线。模型实验与 benchmark 必须显式使用 `clip`/`smolvlm`，避免 `auto` 在依赖缺失时静默回退。真实模型、视频和导出能力按需安装：

| 能力 | 安装命令 |
| --- | --- |
| CLIP / SmolVLM | `uv sync --extra model` |
| 视频抽帧 | `uv sync --extra video` |
| ONNX 导出与验证 | `uv sync --extra deploy` |
| PC 端 PyTorch | `uv sync --extra pc-torch` |

Jetson 上的 PyTorch 通常应使用与 JetPack 匹配的 NVIDIA 构建，不要默认安装 PyPI wheel。完整步骤见 [Jetson 部署与验证](docs/ops/jetson-deployment.md)。

## 周末实施计划

### 第一天：理解并复现基线

1. 按 `apps/ -> encoders/backends -> memory_index -> schema` 顺序阅读源码，画出数据类型和调用链。
2. 先跑 `hash`/`mock` 冒烟测试，再显式运行 CLIP 与 SmolVLM，验证图片确实进入模型。
3. 在 Jetson 上记录环境、模型加载、CLIP 编码、检索和 VLM 生成的分阶段数据。
4. 修正 benchmark 口径：使用 tokenizer 统计生成 token，区分当前 RSS 与峰值内存，并连续采集 `tegrastats`。

### 第二天：打通优化入口

1. 为 CLIP 固定代表性校准集，对比 FP32、FP16 和 INT8 的 embedding 余弦漂移、Recall@1/5、延迟、内存和功耗。
2. 补齐 TensorRT 图像预处理、binding 与推理代码，使 engine 构建后能被真实查询链路消费。
3. 在 JetPack 6.2+ 上验证 TensorRT Edge-LLM 环境，优先跑通官方支持的小型 VLM FP16 示例；训练和 INT8/INT4 作为后续里程碑，不要求两天内全部完成。

Edge-LLM 模型与精度组合以[官方支持列表](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html)和[量化说明](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/features/quantization.html)为准。SmolVLM 继续作为当前 PyTorch/Transformers 基线；Edge-LLM 主线初步评估 Qwen3-VL-2B 或 1B 级 InternVL。

## 常用任务

### 准备数据

将图片放入 `data/images/`。视频可按固定间隔抽帧：

```text
uv run python tools/extract_frames.py --video data/videos/demo.mp4 --every 30 --out-dir data/images
```

离线具身推理样例位于 `data/episodes/sample_episodes.jsonl`。

### 构建和查询视觉记忆

```text
uv run python tools/build_index.py --image-dir data/images --out artifacts/index --encoder clip
uv run python apps/query_memory.py --query "哪些场景适合机器人抓取？" --top-k 5 --encoder clip
```

建库和查询必须使用相同编码器。CLIP/ViT 的设计和局限见 [视觉记忆设计](docs/development/clip_memory_design.md)。

### 运行单图问答与 episode replay

```text
uv run python apps/ask_image.py --image data/images/example.jpg --prompt "描述风险并输出 JSON" --model smolvlm
uv run python apps/replay_episode.py --episodes data/episodes/sample_episodes.jsonl --index artifacts/index --encoder hash --model mock --top-k 3
```

VLM 输出只作为白名单 action 建议，不直接执行 shell 或机器人控制动作。

### 运行性能评测

```text
uv run python tools/benchmark.py --image-dir data/images --encoder hash --clip-only --runs 30
uv run python tools/benchmark_matrix.py --image-dir data/images --encoder hash --model mock --runs 30
```

结果写入 `reports/`。测试分层和验证要求见 [测试指南](docs/development/testing.md)。正式报告至少包含：软硬件版本、模型与精度、输入尺寸、校准集、冷启动、p50/p90/p99、ViT/prefill/decode 延迟、tokens/s、峰值内存、功耗温度、JSON 有效率与失败样例。

量化不能只比较速度。CLIP 需要报告 embedding 漂移和 Recall@K；VLM 需要在固定任务集上比较结果正确率、JSON 有效率及端到端性能。只有在真实 TensorRT/Edge-LLM runtime 消费 engine 后，才可表述为“完成部署”。

## 目录概览

| 路径 | 职责 |
| --- | --- |
| `jetson_visual_memory/` | 编码器、后端、视觉索引、episode、schema 和指标 |
| `apps/` | 查询、问答、回放和 Agent CLI 入口 |
| `tools/` | 建库、抽帧、导出、环境探测和 benchmark 工具 |
| `tests/` | 无硬件单元测试 |
| `data/` | 样例 episode 与本地媒体入口 |
| `artifacts/` | 索引、ONNX 和 TensorRT 等生成物 |
| `reports/` | benchmark 和部署状态报告 |
| `docs/` | 开发与部署文档索引 |

## 文档地图

- [文档中心](docs/README.md)：按任务选择文档的总入口。
- [开发文档](docs/development/README.md)：系统架构、视觉记忆、测试和扩展设计。
- [部署与运维文档](docs/ops/README.md)：Jetson 环境、依赖、ONNX/TensorRT 和故障证据。

新增、移动或删除文档时，请同步更新最近一级索引，避免出现失效链接。
