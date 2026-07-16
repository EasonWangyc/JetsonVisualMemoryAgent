# 系统架构

本文是 Jetson Visual Memory Agent 的架构事实入口。公共数据结构以 `jetson_visual_memory/schema.py` 和 `jetson_visual_memory/episodes.py` 为准，命令参数以 `apps/` 与 `tools/` 中的实现为准。

## 目标与边界

系统将本地图像、视频抽帧或离线 episode 转换为可检索的视觉记忆，再由 VLM 生成可审计的结构化建议。项目关注端侧部署、接口设计、离线评测和失败归因，不训练大模型，也不直接控制机器人。

## 主链路

```text
data/images 或 data/episodes
  -> Encoder（hash / CLIP）
  -> MemoryIndex（向量归一化、持久化、top-k 检索）
  -> VLM backend（mock / SmolVLM）
  -> AgentResult / action JSON
  -> benchmark 与 backend status report
```

`hash` 与 `mock` 用于无模型 smoke test；`clip` 与 `smolvlm` 用于真实语义检索和图文分析。ONNX Runtime 与 TensorRT 当前主要服务于 CLIP image encoder 的导出和部署验证。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `encoders.py` | 图像与文本编码接口及实现 |
| `memory_index.py` | 图片记录、embedding 归一化、存储和查询 |
| `backends.py` | ONNX/TensorRT 等部署后端状态 |
| `vlm.py` | VLM 分析与结构化输出适配 |
| `schema.py` | Agent 输出契约与校验 |
| `episodes.py` | observation-language-action episode 读取与校验 |
| `metrics.py` | 延迟、分位数、内存和吞吐指标 |

`apps/` 负责用户入口和流程编排；可复用行为应留在核心库。`tools/` 负责离线数据准备、构建、导出和评测，不应成为运行时隐式依赖。

## 数据与生成物

```text
data/images -> tools/build_index.py -> artifacts/index
data/episodes -> apps/replay_episode.py -> 结构化结果
apps/tools -> reports/*.json、reports/*.jsonl、reports/*.md
```

`data/images/` 和 `data/videos/` 是本地输入区，`artifacts/` 与 `reports/` 是可重建输出区。除目录说明和样例外，这些内容不应提交。

## 安全边界

VLM/LLM 只能输出白名单 action 建议和安全约束，不得执行 shell、写入系统配置或直接驱动机器人。未来接入 ROS2 时，下游控制节点仍需校验 action 白名单、前置条件和安全规则；接口草案见 [ros2_bridge_design.md](ros2_bridge_design.md)。
