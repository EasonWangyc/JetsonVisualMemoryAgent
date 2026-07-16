# 可选 ROS2 Bridge 设计

本项目默认不安装 ROS2，也不依赖摄像头。ROS2 bridge 是未来接入机器人系统时的接口草案，不属于当前运行主链路。

## Topic 设计

```text
/visual_memory/query
  type: std_msgs/String
  payload: natural language query

/visual_memory/result_json
  type: std_msgs/String
  payload: AgentResult JSON
```

## 节点行为

```text
query_node
  subscribe /visual_memory/query
  call MemoryIndex.query()
  call VLMAnalyzer.analyze()
  publish /visual_memory/result_json
```

## 安全边界

LLM/VLM 输出只作为 JSON 建议，不直接执行 shell 或机器人动作。下游控制节点必须再次校验白名单 action、前置条件和安全规则。

允许的 action 示例：

- `describe_scene`
- `rank_images`
- `find_risky_cases`
- `suggest_robot_action`
