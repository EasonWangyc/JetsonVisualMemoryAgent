# 测试指南

## 测试目标

默认测试必须在没有模型权重、摄像头、CUDA 和 Jetson 硬件的环境中运行。测试使用 `unittest.TestCase`，pytest 仅作为可选运行器。文件命名为 `test_<模块>.py`，测试方法命名为 `test_<行为>`。

## 常用命令

运行单个模块：

```text
uv run python -m unittest tests.test_memory_index
```

运行全量测试：

```text
uv run python -m unittest discover -s tests
uv run pytest
```

运行轻量端到端验证：

```text
uv run python tools/build_index.py --image-dir data/images --out artifacts/index --encoder hash
uv run python apps/agent_cli.py --query "查找高风险场景" --encoder hash --model mock --top-k 3
```

## 分层要求

- `schema.py`、`episodes.py`：覆盖必填字段、默认值、非法输入和 JSON 往返。
- `memory_index.py`：使用 fake encoder 验证归一化、排序、空索引和持久化。
- `backends.py`：验证依赖或 artifact 缺失时返回明确状态，而不是直接崩溃。
- `metrics.py`：覆盖边界值、分位数和序列化结果。
- CLI 或工具变更：至少执行一条 `hash`/`mock` smoke test；不要把网络下载或真实模型作为单元测试前提。

测试产生的临时文件应放在系统临时目录或 `artifacts/` 并在结束后清理。仓库没有覆盖率阈值；功能变更和 bug 修复应为受影响行为补充回归测试。

## 性能与目标板验证

性能或部署变更需额外运行相关 benchmark，并记录命令、硬件环境、后端和输出文件。模型依赖、ONNX artifact、`trtexec` 或 `tegrastats` 缺失时，报告应保留 `skipped` 及原因，不能把“未执行”描述为“通过”。
