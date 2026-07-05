# Interview Pitch

## 项目一句话

基于 Jetson Orin Nano 构建端侧多模态 / 具身推理部署 mini-infra：使用 ViT/CLIP 建立视觉语义记忆，结合轻量 VLM 输出机器人可消费的结构化 action JSON，并通过 benchmark matrix 记录端侧延迟、内存、tokens/s 和后端部署状态。

## 为什么做这个

机器人和智能体系统需要把多模态输入转成稳定、低延迟、可审计的动作建议。这个项目从端侧部署角度实现一条完整链路：图片/抽帧输入、ViT/CLIP 语义索引、VLM 精读、VLA-style episode replay、JSON schema 校验和性能 benchmark。

## 系统架构

- Visual memory：ViT/CLIP 将图像和文本映射到同一语义空间，支持自然语言检索图片。
- VLM reasoning：SmolVLM 对 top-k 图片进行场景解释、风险判断和任务建议。
- VLA-style replay：episode 输入包含 image、instruction、state、candidate actions 和 safety flags。
- Structured action：模型输出被约束为 action JSON，包含 action_type、target_object、preconditions、safety_constraints。
- Edge benchmark：记录视觉编码、检索、VLM 生成、JSON 解析、tokens/s、peak memory 和 tegrastats 状态。

## 性能优化与局限

- TensorRT/ONNX Runtime 优先用于 CLIP image encoder，避免一开始转换完整 VLM 带来的高调试成本。
- vLLM 不作为 Jetson 主依赖；它更适合服务端高并发 LLM serving，本项目重点是低功耗单机端侧多模态链路。
- LLM/VLM 输出只作为白名单 action 建议，不直接执行任意 shell 或真实机器人控制动作。
- 当前不训练大模型，不做真机控制；重点是端侧部署、接口设计、离线评测和失败归因。

## 可以回答的面试问题

- 为什么选择 ViT/CLIP 做视觉语义索引？
- ViT patch 和 CNN feature map 的区别是什么？
- CLIP 为什么能做图文检索？
- embedding 检索和 RAG 的关系是什么？
- VLM 输出 JSON 不稳定怎么办？
- 端侧 VLM 的瓶颈在哪里，如何用 p50/p90/p99 和 tokens/s 描述？
- TensorRT 应该优化哪一段，为什么先选 CLIP image encoder？
- vLLM、TensorRT、ONNX Runtime 的适用场景有什么区别？
- top-1 和 top-3 精读有什么延迟/效果权衡？
- 这个项目如何往 VLA 扩展？

## 局限和下一步

- 接 ROS2 bridge，让机器人系统通过 topic 查询视觉记忆。
- 接真实 LeRobot/OpenVLA 数据格式，把图像、语言、状态、动作串起来。
- 对 CLIP image encoder 做 TensorRT FP16 对比，并补充 Nsight Systems profiling。
