# 文档中心

这里是仓库文档的统一入口。根 `README.md` 只负责项目简介和快速上手；详细设计、测试与部署说明按任务拆分到子目录。

## 按任务查找

| 任务 | 先读文档 |
| --- | --- |
| 学习项目并规划完整实施路线 | [route.md](route.md) |
| 理解系统边界或修改核心链路 | [development/architecture.md](development/architecture.md) |
| 修改视觉索引、编码器或检索逻辑 | [development/clip_memory_design.md](development/clip_memory_design.md) |
| 新增或调整测试 | [development/testing.md](development/testing.md) |
| 评估 ROS2 接入 | [development/ros2_bridge_design.md](development/ros2_bridge_design.md) |
| 选择推理或导出后端 | [development/serving_framework_notes.md](development/serving_framework_notes.md) |
| 配置 Jetson、ONNX 或 TensorRT | [ops/jetson-deployment.md](ops/jetson-deployment.md) |
| 查阅已确认的重构设计 | [superpowers/README.md](superpowers/README.md) |

## 分类入口

- [开发文档](development/README.md)：代码架构、测试策略和扩展设计。
- [部署与运维文档](ops/README.md)：目标板环境、部署验证和故障证据。
- [设计决策](superpowers/README.md)：经确认、尚未实施的重构设计。

## 维护规则

- `AGENTS.md` 只保留代理每次工作都需要的操作规则和风险边界。
- 新增详细文档时，同步更新本文件和最近一级子目录索引。
- 工程行为以代码、测试、`pyproject.toml` 和开发/部署文档为准；个人求职材料不进入项目文档。
- 删除或移动文档时，使用仓库内搜索检查并修复全部引用。
