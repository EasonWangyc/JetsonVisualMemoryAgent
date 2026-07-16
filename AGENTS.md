# Repository Guidelines

## 项目定位与事实边界

本仓库面向 Jetson Orin Nano 的端侧多模态推理与部署实践。当前真实链路是：CLIP 全局图像 embedding、余弦检索、SmolVLM 单图推理和结构化 JSON；视频仅按间隔抽帧，episode replay 只是离线 VLM planner。`hash`/`mock` 仅用于冒烟测试。不得把语义分割、时序视频理解、模型训练、INT8 量化、TensorRT 运行时或真实 VLA 控制写成已完成能力，除非代码、板端日志和报告均可复现。

## 目录与文档入口

- `jetson_visual_memory/`：编码器、后端、索引、schema、episode 和指标。
- `apps/`：CLI 与流程编排；复用逻辑下沉到核心库。
- `tools/`：建库、抽帧、导出、环境探测与 benchmark。
- `tests/`：无硬件单元测试；文件命名为 `test_<module>.py`。
- `data/episodes/`：可公开样例；`artifacts/`、`reports/` 为本地生成物。
- `docs/development/` 与 `docs/ops/`：设计、测试和 Jetson 部署说明。不要加入 JD 对照、面试话术或个人求职材料。

开始前阅读 `README.md` 和 `docs/README.md`。修改核心链路、测试或部署时，分别查阅 `architecture.md`、`testing.md`、`jetson-deployment.md`，并同步更新相关文档。

## 当前实施优先级

1. 用显式 `clip`/`smolvlm` 后端在 PC 与 Jetson 复现基线，禁止用 `auto` 的静默回退结果充当真实模型数据。
2. 完成 CLIP 图像编码器的 INT8 PTQ：固定校准集，实现 ONNX/TensorRT 推理和预处理，比较 FP32、FP16、INT8 的 Recall@K、embedding 漂移、延迟、内存与功耗。
3. 选择 TensorRT Edge-LLM 官方支持的轻量 VLM，先跑 FP16，再尝试 LoRA/SFT、INT8 SmoothQuant 与 INT4 AWQ；SmolVLM 保留为 Transformers 基线。
4. VLA 只保留离线接口实验，暂不接真实执行器或宣称已部署 VLA 模型。

## 编码与验证规则

使用四空格缩进、类型标注、`pathlib.Path`；遵循 `snake_case`、`PascalCase` 和大写常量命名。优先小模块、dataclass/protocol 与现有接口。修改 schema、后端状态或报告字段时，同步更新调用方和回归测试。

```text
uv run python -m unittest discover -s tests
uv run python tools/build_index.py --image-dir data/images --out artifacts/index --encoder hash
uv run python apps/agent_cli.py --query "查找高风险场景" --encoder hash --model mock --top-k 3
```

板端报告必须记录 JetPack、模型与精度、校准集、输入尺寸、冷启动、p50/p90/p99、真实 tokenizer 吞吐、峰值内存、功耗温度和失败样例。不要提交私有媒体、权重、ONNX、TensorRT engine 或生成报告。未经明确要求，不提交、推送或改写 Git 历史。
