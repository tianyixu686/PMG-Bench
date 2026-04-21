# userpref_v1 数据集（拟定）

目标：在同一 benchmark 下评测两种算法（PMG vs DreamBooth/LoRA）对“用户偏好对齐”的生成能力。

本数据集由两部分组成：
- 阶段1（history）：每个用户的历史交互图片（以及质量/偏好等评分）。
- 阶段2（queries）：每个用户 2 个任务 query（A/B），每个 query 下有 5 张候选图（来自某次生成）及其人工评分；同时每个候选图有生成时使用的完整 prompt。

## 目录结构（在 PMG-Bench 内的推荐组织）

```
PMG-Bench/
  data/
    userpref_v1/
      sources.example.json          # sources.json 模板（提交到 repo）
      sources.json                  # 你本机/服务器的真实路径配置（已在 .gitignore 中忽略）
      splits.json                   # 按 user 划分 train/val/test（构建生成，忽略提交）
      processed_dataset/
        train.json                  # 构建生成（忽略提交）
        val.json
        test.json
      manifests/
        stage2_queries.jsonl        # 构建生成（忽略提交）
        users.jsonl                 # 构建生成（忽略提交）

      images/                       # 真实图片放这里（不提交到 repo）
        history/                    # 阶段1历史图：{image_id}.jpg
        stage2/{user_id}/           # 阶段2候选图：{image_name}
```

注意：`processed_dataset/*.json` 中的 `image_path` 现在约定为 **相对 data/userpref_v1 的路径**（例如 `images/history/13670.jpg`）。
运行脚本时会按 `data_root`（即 `data/userpref_v1`）解析成绝对路径，因此你把代码搬到服务器后，只要把图片按上述目录放好即可。

## 关键 JSON schema（约定）

### `processed_dataset/*.json`（list[dict]）
每条样本对应一个 `(user_id, query_variant)`：

- `worker_id`: int/str
- `history_items_info`: list[{
  - `item_name`: str（建议用 image_id）
  - `image_path`: str（相对路径，history 图片）
  - `caption`: str（建议用 prompt_simple）
  - `preference_score`: number（可选）
  - `quality_score`: number（可选）
}]
- `target_item_info`: {
  - `item_name`: str（建议用 stage2 的 image_name）
  - `image_path`: str（相对路径，stage2 参考图）
  - `caption`: str（该 query 的 raw_prompt_simple）
  - `query_variant`: "A"|"B"
}
- `stage2_candidates`: list[{
  - `image_name`: str
  - `image_path`: str（相对路径，stage2 候选图）
  - `prompt`: str（生成时完整 prompt，来自 `prompts.txt`）
  - `initial_rank`: int
  - `quality_score`: number
  - `preference_score`: number
  - `task_match_score`: number
}]

其中 `target_item_info` 的参考图默认取该 query 下 `preference_score` 最高的候选图（弱监督）。

### `manifests/stage2_queries.jsonl`
每行一条 `(user_id, query_variant)`：包含 `raw_prompt_simple`（任务）和 5 个候选的评分与 prompt，便于后续做评测（例如用候选集做“相似度加权的期望偏好分”）。

## 切分策略（必须按 user）
- `splits.json` 只按 `user_id` 划分 train/val/test；任何用到阶段2评分做训练/调参的实验，都必须在 `train_users` 内完成。
- `test_users` 的阶段2评分只用于最终汇报。

## 构建脚本
- `tools/build_userpref_v1_dataset.py`：读取 `sources.json` 指定的原始文件，生成上述 `processed_dataset/*.json`、`splits.json`、`manifests/stage2_queries.jsonl`。

### 如何配置 sources

1) 复制模板：`data/userpref_v1/sources.example.json` -> `data/userpref_v1/sources.json`
2) 修改其中 3 个原始数据路径（rating / bench_add / merged_data）为你服务器上的真实位置

> `sources.json` 不会提交到 git（已加入 `.gitignore`）。
