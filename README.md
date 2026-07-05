# Jetson Visual Memory Agent

端侧多模态 / 具身推理部署 mini-infra，面向 Jetson Orin Nano 远程开发场景。项目不依赖摄像头硬件调试，输入来自本地图像、短视频抽帧或离线 episode，核心目标是展示 `ViT/CLIP + VLM/LLM + VLA-style replay + structured action JSON + edge benchmark/profiling`。

## Why This Project

机器人和智能体系统需要把多模态输入转成稳定、低延迟、可审计的动作建议。本项目围绕这一需求构建端侧 pipeline：先用 ViT/CLIP 提取图像语义 embedding，再结合轻量 VLM/LLM 完成自然语言检索、场景分析、VLA-style episode replay、结构化 action JSON 输出和 Jetson 端侧性能评测。

## Project Layout

```text
apps/                    用户入口：检索、单图问答、Agent CLI
tools/                   建库、benchmark、环境探测、视频抽帧
jetson_visual_memory/    核心库
data/images/             输入图片
data/videos/             可选视频
data/episodes/           VLA-style observation-language-action 样例
artifacts/               embedding index 输出
reports/                 benchmark 和 demo 结果
docs/                    设计与面试材料
tests/                   单元测试
```

## Quick Start

### 1. 使用 uv 创建环境

```bash
uv sync
uv run python -m unittest discover -s tests
```

Jetson 上如果已经通过 NVIDIA/JetPack 安装了 PyTorch，建议创建可见系统包的 venv：

```bash
uv venv --system-site-packages
uv sync
```

PC 上如果需要让 uv 安装 PyTorch，可使用：

```bash
uv sync --extra pc-torch
```

### 2. 安装可选能力

真实 CLIP / SmolVLM 路线：

```bash
uv sync --extra model
```

视频抽帧能力：

```bash
uv sync --extra video
```

完整本地开发环境：

```bash
uv sync --extra model --extra video --extra dev
```

ONNX Runtime 导出/验证能力：

```bash
uv sync --extra deploy
```

### 3. 准备图片

把 20-200 张图片放到：

```text
data/images/
```

如果只有视频：

```bash
uv run python tools/extract_frames.py --video data/videos/demo.mp4 --every 30 --out-dir data/images
```

### 4. 构建视觉记忆库

无模型依赖 smoke test：

```bash
uv run python tools/build_index.py --image-dir data/images --out artifacts/index --encoder hash
```

真实 CLIP/ViT 路线：

```bash
uv run python tools/build_index.py --image-dir data/images --out artifacts/index --encoder clip
```

### 5. 自然语言检索

```bash
uv run python apps/query_memory.py --query "哪些场景适合机器人抓取？" --top-k 5 --encoder hash
```

使用 CLIP 建库时，查询也应使用 CLIP：

```bash
uv run python apps/query_memory.py --query "哪些场景适合机器人抓取？" --top-k 5 --encoder clip
```

### 6. 单图 VLM 问答

无模型依赖 mock pipeline：

```bash
uv run python apps/ask_image.py --image data/images/example.jpg --prompt "描述风险并输出JSON" --model mock
```

SmolVLM 路线：

```bash
uv run python apps/ask_image.py --image data/images/example.jpg --prompt "描述风险并输出JSON" --model smolvlm
```

### 7. Agent CLI

```bash
uv run python apps/agent_cli.py --query "找出不适合机器人操作的场景" --encoder hash --model mock --top-k 3
```

输出为结构化 JSON，字段包括：

```json
{
  "query": "...",
  "matched_images": [],
  "scene_summary": "...",
  "risk_level": "low|medium|high|unknown",
  "suggested_action": "...",
  "failure_modes": [],
  "confidence": 0.0,
  "action_type": "suggest_robot_action",
  "target_object": "...",
  "preconditions": [],
  "safety_constraints": [],
  "reasoning_summary": "..."
}
```

### 8. VLA-Style Episode Replay

离线 episode 使用 observation-language-action 格式：

```bash
uv run python apps/replay_episode.py \
  --episodes data/episodes/sample_episodes.jsonl \
  --index artifacts/index \
  --encoder hash \
  --model mock \
  --top-k 3
```

每条 episode 包含 `image`、`instruction`、`state`、`candidate_actions`、`chosen_action` 和 `safety_flags`。输出是带安全约束的 action JSON，用来模拟具身模型的离线推理链路。

### 9. Benchmark

CLIP-only 或 hash-only 检索评测：

```bash
uv run python tools/benchmark.py --image-dir data/images --encoder hash --clip-only --runs 30
```

VLM 精读评测：

```bash
uv run python tools/benchmark.py --image-dir data/images --encoder clip --model smolvlm --runs 30 --top-k 1
uv run python tools/benchmark.py --image-dir data/images --encoder clip --model smolvlm --runs 30 --top-k 3
```

输出：

```text
reports/benchmark.jsonl
reports/benchmark.md
```

Benchmark matrix 会一次性跑检索、VLM 精读和 episode replay，并报告后端状态：

```bash
uv run python tools/benchmark_matrix.py --image-dir data/images --encoder hash --model mock --runs 30
```

输出：

```text
reports/benchmark_matrix.jsonl
reports/benchmark_matrix.md
```

### 10. ONNX / TensorRT

导出 CLIP image encoder：

```bash
uv run python tools/export_clip_onnx.py --out artifacts/onnx/clip_image_encoder.onnx
```

构建 TensorRT engine：

```bash
uv run python tools/build_trt_engine.py \
  --onnx artifacts/onnx/clip_image_encoder.onnx \
  --engine artifacts/trt/clip_image_encoder.engine \
  --fp16
```

如果缺少模型依赖、ONNX artifact 或 `trtexec`，脚本会写入 `reports/*.json` 并标记为 `skipped`，方便在 Jetson 上保留可复现的失败原因。

## Jetson Notes

已知目标板环境：

- Jetson Orin Nano
- JetPack 6.2.1 / Jetson Linux R36.4.7
- Python 3.10
- TensorRT 10.3
- 约 7.4 GiB RAM
- NVMe 空间充足

建议先运行：

```bash
uv run python tools/probe_env.py
```

## Dependency Policy

本项目以 `uv` + `pyproject.toml` 为主依赖入口：

- 默认依赖只包含 `numpy` 和 `psutil`，保证 smoke test 轻量可跑。
- `model` extra 安装 `transformers/Pillow/accelerate`，用于 CLIP 和 SmolVLM。
- `video` extra 安装 `opencv-python`，用于视频抽帧。
- `deploy` extra 安装 `onnx/onnxruntime`，用于 ONNX 导出后的基线验证。
- `pc-torch` extra 只建议 PC 使用；Jetson 上 PyTorch 通常应使用 NVIDIA/JetPack 匹配版本，避免从 PyPI 拉到不兼容 wheel。

## Model Backends

- `hash` encoder：无依赖，适合 smoke test，不代表真实语义能力。
- `clip` encoder：`openai/clip-vit-base-patch32`，用于 ViT 图文 embedding 检索。
- `mock` VLM：无依赖，适合验证 pipeline 和 JSON schema。
- `smolvlm`：`HuggingFaceTB/SmolVLM-500M-Instruct`，用于真实图文问答。
- `onnx_clip`：CLIP image encoder ONNX artifact 状态检查和后续 ONNX Runtime 对比。
- `tensorrt_clip`：CLIP image encoder TensorRT engine 构建报告，当前优先用于部署证据而不是完整 VLM 转换。

## JD Alignment

这个项目优先服务大模型推理部署、具身智能 Infra、车端/机器人端侧 AI 部署和 VLM/VLA 工程化岗位。核心面试关键词：

- VLM model structure and bottlenecks
- TensorRT / ONNX Runtime deployment
- edge model benchmark
- structured action output
- robotics inference safety boundary
- VLA-style observation-language-action interface

更多见 [docs/jd_alignment.md](docs/jd_alignment.md) 和 [docs/serving_framework_notes.md](docs/serving_framework_notes.md)。

## Interview Pitch

一句话：

> 基于 Jetson Orin Nano 构建端侧多模态 / 具身推理部署 mini-infra，使用 ViT/CLIP 建立视觉语义记忆，结合轻量 VLM 输出机器人可消费的 action JSON，并通过 benchmark matrix 记录端侧延迟、内存、tokens/s 和后端部署状态。

更多见 [docs/interview_pitch.md](docs/interview_pitch.md)。
