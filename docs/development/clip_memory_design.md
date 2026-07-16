# CLIP / ViT 视觉记忆设计

## 核心思路

本项目把图像编码为语义向量，并在统一语义空间内完成图文检索：

```text
image -> ViT image encoder -> image embedding
text query -> text encoder -> text embedding
cosine similarity -> top-k image retrieval
```

CLIP 将图像和文本 embedding 训练到同一个语义空间，因此可以用自然语言直接检索图片。

## 为什么使用 ViT

ViT 将图像切成 patch，把每个 patch 视为视觉 token，再通过 Transformer 建模全局关系。对本项目而言，ViT 提供全局语义表征，使图片可以被自然语言查询、聚类并交给 VLM 进一步推理。

## 数据流

```text
data/images
  -> tools/build_index.py
  -> artifacts/index/embeddings.npy
  -> artifacts/index/metadata.jsonl
  -> apps/query_memory.py
```

建库和查询必须使用相同编码器。索引存储和检索行为以 `jetson_visual_memory/memory_index.py` 为准。

## 局限

- CLIP 检索偏向全局语义相关性，不提供像素级定位或精细边界。
- 对细粒度空间关系、遮挡和抓取可行性，CLIP 只负责候选筛选，最终需要 VLM 或机器人感知模块验证。
- 本项目不训练模型，重点是端侧部署、系统接口和评测链路。
