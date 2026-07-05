# JD Alignment

## 项目定位

Jetson Visual Memory Agent 面向端侧多模态和具身推理部署：在 Jetson 上完成视觉语义索引、VLM 精读、结构化动作 JSON、VLA-style episode replay 和 benchmark/profiling。

## JD 关键词映射

| JD 方向 | 项目证据 |
| --- | --- |
| 大模型推理部署 | SmolVLM 端侧推理、tokens/s、peak memory、top-1/top-3 精读延迟对比 |
| TensorRT / ONNX Runtime | CLIP image encoder ONNX 导出、TensorRT engine 构建脚本、backend status report |
| 具身智能 Infra | episode replay 使用 observation-language-action 格式，输出 action JSON 和 safety constraints |
| VLM / VLA 工程化 | VLM 负责图像精读，episode replay 把 image、instruction、state、candidate actions 串成推理输入 |
| 机器人安全边界 | LLM/VLM 只输出白名单 action 建议，不执行 shell 或真实控制动作 |
| 性能优化与 profiling | benchmark matrix 输出 p50/p90/p99、tokens/s、memory、tegrastats 状态和失败样例 |
| 边缘部署 | Jetson Orin Nano 上验证 Python/Transformers 主链路，重依赖缺失时输出 skipped 而不是崩溃 |

## 面试叙事

1. 我没有把项目做成单纯模型调用，而是做成端侧推理链路：数据输入、语义索引、VLM 精读、结构化输出、性能报告。
2. 我把 VLA 拆成可落地的接口问题：observation-language-action episode replay，不做真实机器人控制，但保留 action 和 safety schema。
3. 我把 TensorRT 放在最适合落地的位置：先优化 CLIP/ViT image encoder，而不是一开始就转换完整 VLM。
4. 我明确知道 vLLM 更适合服务端高并发 LLM serving，所以在 Jetson 上不把它作为主依赖，而是在文档里作为 serving framework 对照。

## 简历 bullet 示例

- 构建 Jetson 端侧多模态推理系统，基于 CLIP/ViT 建立视觉语义记忆库，结合轻量 VLM 输出机器人可消费的 action JSON。
- 设计 VLA-style episode replay 格式，将 image、language instruction、robot state、candidate actions 和 safety flags 统一到离线评测链路。
- 实现 benchmark matrix，统计视觉编码、检索、VLM 生成、JSON 解析的 p50/p90/p99 延迟、峰值内存和 tokens/s。
- 增加 CLIP image encoder 的 ONNX 导出和 TensorRT engine 构建脚本，用于对比 PyTorch、ONNX Runtime 和 TensorRT 后端状态。
