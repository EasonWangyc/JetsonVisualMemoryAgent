# 推理框架选型说明

## 项目取舍

主链路是 Jetson 端侧 VLM 推理和视觉语义检索：`CLIP/ViT -> visual memory -> VLM -> structured action JSON -> benchmark`。核心目标是让低功耗设备上的多模态推理链路可运行、可测量并可定位瓶颈。

## TensorRT 与 ONNX Runtime

- TensorRT 适合优化稳定的视觉编码器子图，例如 CLIP image encoder。
- ONNX Runtime 作为 PyTorch 和 TensorRT 之间的可移植基线，用于检查导出正确性和延迟。
- 项目优先导出 CLIP image encoder，不强行转换完整 VLM；后者涉及视觉编码、语言模型、processor 和生成逻辑，端侧调试成本更高。

## vLLM

vLLM 主要解决服务端大语言模型的 PagedAttention、Continuous Batching、KV Cache 管理和高吞吐调度。Jetson Orin Nano 的资源约束和本项目的单机低并发场景不适合作为 vLLM 主链路，因此它只作为后端对照，不是项目依赖。

## llama.cpp、MNN 与 TVM

- llama.cpp：适合 CPU/GPU 混合的量化 LLM 推理，可作为轻量文本后端扩展。
- MNN/NCNN：更偏移动端和嵌入式部署，适合迁移到非 NVIDIA 平台。
- TVM/MLIR：更偏编译器和图优化研究，不作为当前版本的阻塞项。

## 当前默认

- VLM：Transformers + SmolVLM，保证功能闭环。
- Vision backend：PyTorch CLIP 为功能基线，ONNX/TensorRT 为加速对比项。
- Benchmark：显式记录阶段延迟、峰值内存、tokens/s、失败样例和 `tegrastats` 状态。
