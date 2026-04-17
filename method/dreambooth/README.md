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

