# 示例数据 / Example assets

本目录仅用于快速了解数据格式与可视化，**不能**替代完整 `userpref_v1` 数据集。

| 路径 | 说明 |
|------|------|
| `history/*.jpg` | 阶段1 历史交互图示例（3 张） |
| `stage2/0/*.png` | 阶段2 候选图示例（用户 0，2 张） |
| `../userpref_v1/manifests/stage2_queries.sample.jsonl` | 前 5 条 query 元数据 |
| `test_cases.sample.json` | 2 条与 `processed_dataset` 同 schema 的迷你样本 |

完整数据请按 `data/userpref_v1/README.md` 运行 `tools/build_userpref_v1_dataset.py` 构建。
