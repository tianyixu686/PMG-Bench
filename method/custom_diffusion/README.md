# Custom Diffusion (PMG-Bench 适配版)

该目录提供一个 **Custom Diffusion（diffusers 路线）** 的最小复现与 PMG-Bench 数据适配：

- 训练：只更新 UNet **cross-attention** 中的 **key/value 投影矩阵**（`to_k/to_v`），可选训练一个 **modifier token** 的 embedding。
- 保存：输出 `delta.bin`（仅包含被训练的 UNet 参数子集）与 `learned_token.bin`（modifier token embedding）。
- 评测：加载 base SD1.5 + `delta.bin` + `learned_token.bin`，并将 prompt 中的 `[V]` 替换为 `modifier_token`。

> 说明：官方仓库也提供了 diffusers 训练版本（README 中已提及）。这里按 PMG-Bench 的 SER/POG/FLICKR-AES 数据格式写了数据集与评测脚本。

## 目录结构

- `datasets.py`：SER/POG/FLICKR-AES 的训练数据读取（从 history/sequence 采样 image + masked caption）。
- `delta_utils.py`：选择可训练的 K/V 参数；保存/加载 `delta.bin`。
- `train_custom_diffusion_sd15_diffusers.py`：**对齐官方范式** 的训练入口（Accelerate；支持 `freeze_model=crossattn_kv|crossattn`、mask 加权 loss、可选 prior-preservation、可选 modifier token embedding）。
- `evaluate_*_custom_diffusion.py`：三套评测（生成 + LPIPS/SSIM/CLIP 相似度；Flickr 可选 verifier）。

## 训练

### 推荐（对齐官方范式，Accelerate）

以 SER 为例：

```bash
python train_custom_diffusion_sd15_diffusers.py \
  --dataset ser \
  --train_json "{SER_DATASET_BASE_PATH}/processed_masked/train.json" \
  --sd15_path "{SD15_MODEL_PATH}" \
  --output_dir "{SER_DATASET_BASE_PATH}/custom_diffusion_sd15" \
  --modifier_token "<new1>" \
  --initializer_token "photo" \
  --freeze_model crossattn_kv \
  --train_modifier_token_embedding \
  --mixed_precision fp16
```

产物仍保持与评测脚本兼容：

- `delta.bin`（只包含被训练的 UNet 参数子集）
- `learned_token.bin`（可选：训练了 modifier token 时才会输出）

### SER

```bash
python train_custom_diffusion_sd15_diffusers.py \
  --dataset ser \
  --train_json "{SER_DATASET_BASE_PATH}/processed_masked/train.json" \
  --sd15_path "{SD15_MODEL_PATH}" \
  --output_dir "{SER_DATASET_BASE_PATH}/custom_diffusion_sd15" \
  --modifier_token "<new1>" \
  --train_modifier_token_embedding
```

### POG

```bash
python train_custom_diffusion_sd15_diffusers.py \
  --dataset pog \
  --train_json "{POG_BASE_PATH}/processed_dataset/train.json" \
  --sd15_path "{SD15_MODEL_PATH}" \
  --output_dir "{POG_BASE_PATH}/custom_diffusion_sd15" \
  --modifier_token "<new1>" \
  --train_modifier_token_embedding
```

### FLICKR-AES

```bash
python train_custom_diffusion_sd15_diffusers.py \
  --dataset flickr_aes \
  --train_json "{FLICKR_AES_BASE_PATH}/processed_dataset/train.json" \
  --sd15_path "{SD15_MODEL_PATH}" \
  --output_dir "{FLICKR_AES_BASE_PATH}/custom_diffusion_sd15" \
  --modifier_token "<new1>" \
  --train_modifier_token_embedding
```

训练产物：

- `custom_diffusion_sd15/delta.bin`
- `custom_diffusion_sd15/learned_token.bin`

## 评测

### SER

```bash
python evaluate_ser_custom_diffusion.py \
  --test_json "{SER_DATASET_BASE_PATH}/processed_masked/test.json" \
  --sd15_path "{SD15_MODEL_PATH}" \
  --delta_path "{SER_DATASET_BASE_PATH}/custom_diffusion_sd15/delta.bin" \
  --token_path "{SER_DATASET_BASE_PATH}/custom_diffusion_sd15/learned_token.bin" \
  --modifier_token "<new1>"
```

### POG

```bash
python evaluate_pog_custom_diffusion.py \
  --test_json "{POG_BASE_PATH}/processed_dataset/test.json" \
  --sd15_path "{SD15_MODEL_PATH}" \
  --delta_path "{POG_BASE_PATH}/custom_diffusion_sd15/delta.bin" \
  --token_path "{POG_BASE_PATH}/custom_diffusion_sd15/learned_token.bin" \
  --modifier_token "<new1>"
```

### FLICKR-AES

```bash
python evaluate_flickr_aes_custom_diffusion.py \
  --test_json "{FLICKR_AES_BASE_PATH}/processed_dataset/test.json" \
  --sd15_path "{SD15_MODEL_PATH}" \
  --delta_path "{FLICKR_AES_BASE_PATH}/custom_diffusion_sd15/delta.bin" \
  --token_path "{FLICKR_AES_BASE_PATH}/custom_diffusion_sd15/learned_token.bin" \
  --modifier_token "<new1>"
```

## 依赖

至少需要：`torch`, `diffusers`, `transformers`。
评测脚本另外需要：`lpips`, `torchmetrics`, `clip`。
