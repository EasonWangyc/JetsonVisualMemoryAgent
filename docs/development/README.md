# 开发文档

本目录面向代码修改、测试和系统扩展。开始开发前，先根据任务选择对应文档：

- [architecture.md](architecture.md)：系统主链路、模块职责、数据流与安全边界，是架构事实来源。
- [testing.md](testing.md)：测试分层、命令、fixture 和完成标准。
- [clip_memory_design.md](clip_memory_design.md)：CLIP/ViT 视觉记忆原理、数据流和局限。
- [serving_framework_notes.md](serving_framework_notes.md)：PyTorch、ONNX Runtime、TensorRT 等后端取舍。
- [ros2_bridge_design.md](ros2_bridge_design.md)：可选 ROS2 bridge 的 topic、节点行为和控制边界。

新增开发文档时更新本索引。实现发生变化时，优先更新最接近代码职责的文档，不要把长篇说明追加到 `AGENTS.md`。
