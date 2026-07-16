# Jetson 部署与验证

## 环境基线

当前目标环境：

- Jetson Orin Nano
- JetPack 6.2.1 / Jetson Linux R36.4.7
- Python 3.10
- TensorRT 10.3
- 约 7.4 GiB RAM，使用 NVMe 存储

实际部署前先运行环境探测，不要仅依赖本文中的静态版本：

```text
uv run python tools/probe_env.py
```

## 依赖策略

默认依赖只有 `numpy` 和 `psutil`。按任务安装可选能力：

```text
uv sync --extra model
uv sync --extra video
uv sync --extra deploy
```

Jetson 上如已通过 JetPack 安装 PyTorch，创建可见系统包的环境：

```text
uv venv --system-site-packages
uv sync
```

`pc-torch` 只用于普通 PC。Jetson 不应默认从 PyPI 安装 PyTorch，以免获得与 CUDA、cuDNN 或 JetPack 不兼容的 wheel。

## 后端验证顺序

1. 使用 `hash` + `mock` 跑通数据、schema 和报告链路。
2. 使用 PyTorch CLIP 验证真实 embedding 与检索行为。
3. 导出 CLIP image encoder 到 ONNX，并用 ONNX Runtime 做基线检查。
4. 使用 Jetson 自带 `trtexec` 构建 TensorRT FP16 engine。
5. 运行 benchmark matrix，对比阶段延迟、峰值内存和后端状态。

```text
uv run python tools/export_clip_onnx.py --out artifacts/onnx/clip_image_encoder.onnx
uv run python tools/build_trt_engine.py --onnx artifacts/onnx/clip_image_encoder.onnx --engine artifacts/trt/clip_image_encoder.engine --fp16
uv run python tools/benchmark_matrix.py --image-dir data/images --encoder clip --model smolvlm --runs 30
```

当前优先优化稳定的 CLIP image encoder 子图，不把完整 VLM 转换作为基础部署前提。具体取舍见 [../development/serving_framework_notes.md](../development/serving_framework_notes.md)。

## 输出与故障证据

- 索引、ONNX 和 TensorRT engine 写入 `artifacts/`。
- benchmark 与部署状态写入 `reports/`。
- 缺少模型依赖、artifact、`trtexec` 或 `tegrastats` 时，脚本应输出 `skipped` 和可复现原因。
- 不提交模型权重、私有媒体、ONNX、engine 或运行报告；需要分享时只提交脱敏后的说明或汇总数据。

仓库当前不包含自动部署、发布、回滚或远程服务重启流程。执行目标板安装、远程连接或系统级变更前必须取得用户明确授权。
