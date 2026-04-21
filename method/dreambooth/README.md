# DreamBooth (LoRA) for PMG-Bench

该目录在 PMG-Bench 的数据格式上复现 DreamBooth 的一个常用实现：**DreamBooth + UNet LoRA**，并额外支持训练一个**单 token 的可学习词向量**（用于替换你们 mask 里的 `[V]`）。

> 说明
> - 这里默认训练 **LoRA**（更省显存、收敛更快），不是全量微调 UNet 全参数。
> - 训练/评测都沿用 `method/README.md` 里的路径占位符机制（例如 `{SER_DATASET_BASE_PATH}`、`{SD15_MODEL_PATH}`）。
> - 训练数据字段与 `method/textual_inversion/*_textual_inversion_sd15.py` 对齐：SER 用 `history_items_info`；POG 用 `history_items_info` + captions json；FLICKR-AES 用 `interaction_sequence` + captions json。

## 目录结构
- `train_dreambooth_lora_sd15_diffusers.py`
  - **推荐**：对齐 diffusers 官方范式（`accelerate` + `peft` LoRA）
  - 保存为官方格式权重文件：`pytorch_lora_weights.safetensors`
  - 仍保留 PMG-Bench 适配：从 JSON 采样 history 图像 + masked caption，并将 `[V]` → `instance_token`

- `evaluate_ser_dreambooth.py` / `evaluate_pog_dreambooth.py` / `evaluate_flickr_aes_dreambooth.py`
  - 评测入口：加载 base SD1.5 + LoRA + token embedding
  - prompt 中把 `[V]` 替换成 `instance_token`
  - 指标与 `method/textual_inversion/evaluate_*.py` 保持一致（尽量复用相同实现）。

## 复现步骤（SER 示例）

1) （可选）先做 caption masking（生成包含 `[V]` 的 masked caption）

```powershell
cd E:\proj\PMG-Bench\method
python .\textual_inversion\mask_captions_with_llm.py
```

2) 训练 DreamBooth-LoRA

```powershell
cd E:\proj\PMG-Bench\method\dreambooth

# 官方范式（需要 peft + accelerate）
accelerate launch .\train_dreambooth_lora_sd15_diffusers.py --dataset ser --pretrained_model_name_or_path {SD15_MODEL_PATH}
```

3) 评测

```powershell
python .\evaluate_ser_dreambooth.py \
  --test_json {SER_DATASET_BASE_PATH}/processed_masked/test.json \
  --sd15_path {SD15_MODEL_PATH} \
  --lora_dir {SER_DATASET_BASE_PATH}/dreambooth_lora_sd15 \
  --token_path {SER_DATASET_BASE_PATH}/dreambooth_lora_sd15/learned_token.bin \
  --instance_token sks
```

## 关键约定

- `[V]`：你们 masking 脚本生成的“外观/风格特征占位符”。
- `instance_token`：DreamBooth 学习的“概念 token”（必须是 **单 token**，脚本会自动 `tokenizer.add_tokens()`）。
- 训练 prompt：默认取样本里的 `masked_caption`（或 `caption`），再把其中的 `[V]` → `instance_token`。

## userpref_v1（逐用户 LoRA，评测 test_users）

你的 `userpref_v1` 目标是：对 PMG 的 `test_users`，分别做 **per-user LoRA 训练**（仅用阶段1历史图+caption），再用阶段2的 A/B query prompt 生成图做评测。

该 benchmark 关注的是 **style 个性化**：为了避免 prompt 里直接写出“油画/水彩/光影/笔触/渲染”等风格词导致模型“靠文字对齐风格”，`USERPREF_DB_TRAIN.py` / `USERPREF_DB_INFER.py` 默认会对 prompt 做一层 **风格词掩码（style mask）**。

在 DreamBooth 侧，这层 mask 的输出不是“删除风格词”，而是：
- 将风格/外观类词（风格、笔触、光影、纹理、渲染等）**替换成一个陌生 token**（默认 `sks`）
- **整句必须且只能出现一次** `sks`
- 如果句子里没有检测到风格词，会在句末追加：`in sks style`

> 注意：style mask 仅用于 DreamBooth 基线（方法侧处理），PMG 默认不需要/不启用这层 mask。

本仓库已提供两个封装脚本：
- `USERPREF_DB_TRAIN.py`：从 `data/userpref_v1/splits.json` + `processed_dataset/test.json` 自动为每个 test user 生成一个 SER 兼容的 per-user `train.json`，并调用 `train_dreambooth_lora_sd15_diffusers.py` 训练 LoRA。
- `USERPREF_DB_INFER.py`：加载 base SD1.5 + 每个 user 的 LoRA，按 `test.json` 里的 A/B `target_item_info.caption`（即 raw_prompt_simple）生成图片。

### Style mask 参数

- 默认开启：训练时写入 `masked_caption`（SER 数据集会优先用它）；推理时会基于 `caption` 现算 mask（保证 `sks` 规则）。
- 关闭：传 `--no_style_mask`（`--no_mask` 为兼容旧参数，等价于关闭）。
- 词表：`method/dreambooth/style_mask_terms.json`，可用 `--style_mask_terms path/to/terms.json` 自定义。
- token：用 `--instance_token sks` 指定（训练与推理应一致）。

### Token embedding（可选）

`USERPREF_DB_TRAIN.py` 默认只训练 **LoRA/网络参数**（不训练 token embedding，脚本会传 `--no_train_instance_token_embedding`）。

同时，内部训练器仍会：
- 把 `--instance_token`（如 `sks`）加入 tokenizer（保证它是“单 token”）
- 保存 `learned_token.bin`（即使没训练 token embedding，这个文件也会存在，用于提供确定的初始化向量；推理脚本会优先加载它）

如果你需要做对比实验（同时训练 token embedding），可以加 `--train_instance_token_embedding` 透传给内部训练器。

训练（默认仅打印命令，不真正训练；加 `--run` 才会跑）：

```powershell
cd E:\proj\PMG-Bench\method\dreambooth

python .\USERPREF_DB_TRAIN.py \
  --sd15_path {SD15_MODEL_PATH} \
  --dry_run

# 真跑（示例：只跑两个用户）
python .\USERPREF_DB_TRAIN.py \
  --sd15_path {SD15_MODEL_PATH} \
  --user_ids 0,1 \
  --history_policy topk --history_topk 30 \
  --max_train_steps 800 --rank 4 --mixed_precision fp16 \
  --run
```

推理：

```powershell
cd E:\proj\PMG-Bench\method\dreambooth

python .\USERPREF_DB_INFER.py \
  --sd15_path {SD15_MODEL_PATH} \
  --lora_root E:\proj\PMG-Bench\outputs\userpref_v1_dreambooth_lora \
  --output_dir E:\proj\PMG-Bench\outputs\userpref_v1_dreambooth_infer
```

