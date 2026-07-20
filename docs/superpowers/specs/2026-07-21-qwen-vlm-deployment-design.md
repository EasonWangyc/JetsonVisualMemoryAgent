# Qwen VLM 部署评测设计

- 状态：已确认，尚未实施
- 日期：2026-07-21
- 目标平台：Jetson Orin Nano
- 第一版模型：`Qwen/Qwen3-VL-2B-Instruct`

## 1. 项目目标

将仓库主线重构为一个可复现的 Jetson VLM 推理部署与评测项目。第一版使用 `Qwen/Qwen3-VL-2B-Instruct` 完成同一个单图场景/风险分析任务，对比 Transformers 正确性参考与 TensorRT Edge-LLM FP16 实际运行时，并输出严格 JSON、质量结果和板端性能证据。开始 engine 构建前必须以目标 JetPack 与 TensorRT Edge-LLM 版本重新验证该模型组合的兼容性。

项目要证明的能力是：在真实 Jetson 环境中，将 VLM engine 接入应用调用路径，固定输入和生成参数后，解释正确性、延迟、吞吐、内存、功耗和失败原因的差异。

## 2. 第一版范围

### 必须完成

- 使用固定图片、提示词和 JSON schema 的单图场景/风险分析工作负载。
- 提供 Transformers Qwen3-VL runtime，作为功能正确性参考。
- 提供 TensorRT Edge-LLM FP16 runtime，并由项目 Python 调用实际 engine。
- 在同一任务集上检查 JSON 合法性、字段完整性、任务结果和失败分类。
- 在 Jetson 上记录冷启动、预热后 p50/p90/p99、tokens/s、峰值内存、功耗、温度和失败样例。
- 记录 JetPack、GPU power mode、模型版本、runtime/engine 版本、输入尺寸和 generation 参数。

### 明确不做

- 不把 CLIP 检索、视觉记忆、Agent 编排或 episode/VLA 作为主路径。
- 不做 LoRA/SFT、INT8/INT4、AWQ、SmoothQuant、C++ runtime、批处理、KV Cache 优化或并发调度。
- 不将仅生成 engine、仅有导出脚本或仅有 PC 结果表述为完成部署。
- 不执行机器人动作，也不将 JSON 建议连接到控制器。

## 3. 总体结构

```text
jetson-vlm-deploy/
├─ jetson_vlm_deploy/
│  ├─ contracts/       # 请求、结果、JSON schema、评测与报告记录
│  ├─ runtimes/        # runtime protocol、Transformers 和 Edge-LLM 实现
│  ├─ workload/        # 单图风险分析 prompt、案例和加载逻辑
│  ├─ evaluation/      # JSON 校验、任务判定、失败分类
│  ├─ benchmark/       # 时间、tokens/s、内存、功耗和温度采集
│  └─ board/           # Jetson 环境和依赖探测
├─ apps/               # 单图调用和固定评测集入口
├─ tools/              # 环境探测、engine 构建、正式 benchmark
├─ configs/            # 模型、generation 和 benchmark 配置
├─ data/eval/          # 可公开评测图片、提示词和预期字段
├─ tests/              # 无硬件单测和 runtime contract 测试
├─ reports/            # 本地可重建的板端结果，不提交原始产物
├─ docs/development/   # runtime contract、评测与架构说明
└─ docs/ops/           # Jetson、engine 构建和板端复现说明
```

现有 CLIP、SmolVLM 和 episode 代码将冻结为 `legacy/` 工作负载：不删除，第一版不再扩展，也不与新主路径共用 runtime 接口。

## 4. 模块边界

`workload` 只定义图片、prompt、预期 JSON 字段和案例版本，不依赖模型或推理引擎。

`runtimes` 对外实现统一的 `analyze(image, prompt)` 契约。Transformers 和 Edge-LLM 只可在内部替换，返回相同的结构化结果、阶段时间和原始错误摘要。

`evaluation` 消费统一结果，判定 JSON 合法性、字段完整性、任务结果和失败类别；不得因 backend 不可用而伪造结果。

`benchmark` 只采集事实：冷启动与预热后的延迟、tokens/s、峰值内存、功耗、温度和采样状态；不得将缺失的硬件数据替换为零值后视为成功。

`board` 负责 JetPack、TensorRT Edge-LLM、GPU mode、`tegrastats` 和模型产物可用性探测，返回可审计的 ready/skipped/failed 状态。

## 5. 数据流与报告

```text
RunConfig
  -> board probe
  -> fixed image + prompt + expected JSON fields
  -> runtime.analyze()
  -> schema validation and task evaluation
  -> resource sampling
  -> JSONL raw rows + Markdown summary
```

每条原始记录绑定：git commit、环境版本、模型与 runtime/engine 版本、精度、任务集版本、输入尺寸、generation 参数、运行状态、错误摘要和指标。

首版 JSON 契约固定为：`schema_version`、非空 `scene_summary`、枚举值 `risk_level`（`low`、`medium`、`high`）、字符串数组 `hazards` 和非空 `recommended_action`。评测集的每个案例提供期望 `risk_level` 与至少一个必需 `hazard` 标签；任务结果正确表示风险等级完全匹配，且输出覆盖全部必需风险标签。非 JSON、缺字段、枚举值非法或类型不符均为 schema 失败。

质量评测与性能评测分开运行：

- `run_eval` 只比较 JSON、字段、风险等级、必需风险标签覆盖和失败类型。
- `benchmark_runtime` 只在固定配置下记录冷启动、预热后分位数、tokens/s、内存、功耗和温度。

## 6. 失败分类与测试

失败分类包括：依赖缺失、engine 构建失败、engine 加载失败、OOM、超时、生成非 JSON、schema 不通过、任务结果错误和资源采样不可用。

测试分为三层：

1. 无硬件单元测试：schema、任务集、统计、报告格式和 runtime contract。
2. PC 冒烟：Transformers Qwen3-VL 能处理一张图并返回合法 JSON。
3. Jetson 集成：实际 Edge-LLM runtime、固定评测集和 benchmark；板端报告是部署完成的唯一证据。

## 7. 实施门槛与里程碑

1. 建立核心结构、统一结果契约、固定评测集和报告格式。
2. 完成 Qwen3-VL Transformers 正确性对照；此时不称部署完成。
3. 构建并接入 TensorRT Edge-LLM FP16 engine，确保项目实际调用 engine。
4. 在 Jetson 完成同任务集正确性对照与正式性能报告。

仅在前一里程碑具有可复现代码和报告时，才开始后一里程碑。

## 8. 第二阶段扩展

第二阶段按以下顺序扩展：

```text
FP16 runtime 证据
  -> INT8/INT4 量化与精度对齐
  -> C++ runtime、buffer 管理与计时细化
  -> 批处理、KV Cache 和并发调度
```

训练和微调不属于第一版。只有固定评测集证明原模型在任务能力上存在明确缺口时，才单独启动 LoRA/SFT：构造数据、与原模型对照、再评估 adapter 或合并权重的部署兼容性。训练结果不能替代 runtime 集成或板端性能证据。
