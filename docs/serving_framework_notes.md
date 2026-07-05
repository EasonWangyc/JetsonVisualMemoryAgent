# Serving Framework Notes

## 项目取舍

本项目的主链路是 Jetson 端侧 VLM 推理和视觉语义检索：`CLIP/ViT -> visual memory -> VLM -> structured action JSON -> benchmark`。核心目标是证明低功耗设备上的多模态推理链路能跑通、能测量、能定位瓶颈。

## TensorRT / ONNX Runtime

- TensorRT 更适合优化稳定的视觉编码器子图，例如 CLIP image encoder。
- ONNX Runtime 适合作为 PyTorch 和 TensorRT 之间的可移植基线，方便检查导出模型的正确性和延迟。
- 本项目优先导出 CLIP image encoder，而不是强行转换完整 VLM，因为 VLM 的视觉编码、语言模型、processor 和生成逻辑更复杂，端侧调试成本更高。

## vLLM

vLLM 的核心价值在服务端大语言模型 serving：PagedAttention、Continuous Batching、KV Cache 管理和高吞吐请求调度。Jetson Orin Nano 的资源约束和本项目的单机低并发场景不适合作为 vLLM 主链路。

本项目保留 vLLM 作为对照知识点：面试时可以说明 vLLM 解决的是高并发 LLM serving 问题，而本项目解决的是端侧多模态推理链路、显存占用、延迟和结构化输出可靠性。

## llama.cpp / MNN / TVM

- llama.cpp：适合 CPU/GPU 混合的量化 LLM 推理，可作为轻量文本模型后端拓展。
- MNN/NCNN：更偏移动端和嵌入式部署，适合后续迁移到非 NVIDIA 平台。
- TVM/MLIR：更偏编译器和图优化研究，适合作为深入方向，不作为当前两周版本阻塞项。

## 当前默认

- VLM：Transformers + SmolVLM，保证功能闭环。
- Vision backend：PyTorch CLIP 为功能基线，ONNX/TensorRT 为加速对比项。
- Benchmark：显式记录阶段延迟、峰值内存、tokens/s、失败样例和 tegrastats 状态。
