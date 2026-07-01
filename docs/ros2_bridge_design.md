# Optional ROS2 Bridge Design

本项目默认不安装 ROS2，也不依赖摄像头。ROS2 bridge 只是未来接机器人系统时的接口设计。

## Topics

```text
/visual_memory/query
  type: std_msgs/String
  payload: natural language query

/visual_memory/result_json
  type: std_msgs/String
  payload: AgentResult JSON
```

## Node Behavior

```text
query_node
  subscribe /visual_memory/query
  call MemoryIndex.query()
  call VLMAnalyzer.analyze()
  publish /visual_memory/result_json
```

## Safety Boundary

LLM/VLM 输出只作为 JSON 建议，不直接执行任意 shell 或机器人动作。下游控制节点必须根据白名单 action 和安全规则二次确认。

Allowed action examples:

- `describe_scene`
- `rank_images`
- `find_risky_cases`
- `suggest_robot_action`
