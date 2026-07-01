# CLIP / ViT Visual Memory Design

## 核心思路

本项目把图像编码为语义向量，并在统一语义空间内完成图文检索：

```text
image -> ViT image encoder -> image embedding
text query -> text encoder -> text embedding
cosine similarity -> top-k image retrieval
```

CLIP 的价值在于图像和文本 embedding 被训练到同一个语义空间里，因此可以用自然语言直接检索图片。

## 为什么是 ViT

ViT 会把图像切成 patch，把每个 patch 看作视觉 token，再通过 Transformer 建模全局关系。对本项目来说，ViT 的核心价值是提供全局语义表征，让图片能够被自然语言查询、聚类和进一步推理。

## 数据流

```text
data/images
  -> tools/build_index.py
  -> artifacts/index/embeddings.npy
  -> artifacts/index/metadata.jsonl
  -> apps/query_memory.py
```

## 局限

- CLIP 检索偏向全局语义相关性，不提供像素级定位或精细边界。
- 对细粒度空间关系、遮挡、抓取可行性，只能作为候选筛选，最终需要 VLM 精读或机器人感知模块验证。
- 本项目不训练模型，重点是端侧部署和系统接口。
