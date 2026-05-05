#!/bin/bash
#SBATCH -p debug                     # 分区 (DO NOT CHANGE)
#SBATCH -t 0-12:00:00                # 运行时长
#SBATCH --nodes=1                    # 节点数
#SBATCH --gres=gpu:1                 # GPU数量
#SBATCH --qos=normal                 # 服务质量 [normal, long]
#SBATCH --job-name=up_db_1user        # 作业名
#SBATCH --output=logs/up_db_1user_%j.out  # 输出文件

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"

# 你的 conda 环境名（见 method/environment.yaml: name: combined_environment）
conda activate pmg-bench

REPO_ROOT="/data-nfs/gpu2/u18871384022/PMG-Bench"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ===== 需要你改的两个参数 =====
USER_ID="2"   # 只跑一个 user（test_users 里的 worker_id 字符串）
SD15_PATH="/data-nfs/gpu2/u18871384022/models/stable-diffusion-v1-5"  # 必须指向 diffusers 格式目录（含 tokenizer/text_encoder/unet/vae 等子目录）

# ===== 输出目录（可按需改）=====
LORA_ROOT="${REPO_ROOT}/outputs/userpref_v1_dreambooth_lora"
INFER_OUT="${REPO_ROOT}/outputs/userpref_v1_dreambooth_infer/${USER_ID}"
EVAL_JSON="${REPO_ROOT}/outputs/userpref_v1_dreambooth_eval/${USER_ID}/metrics.json"

mkdir -p "${REPO_ROOT}/logs"

echo "Repo: ${REPO_ROOT}"
echo "User: ${USER_ID}"
echo "SD1.5: ${SD15_PATH}"
echo "LoRA root: ${LORA_ROOT}"
echo "Infer out: ${INFER_OUT}"
echo "Eval json: ${EVAL_JSON}"

python method/dreambooth/tasks/train_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --user_ids "${USER_ID}" \
  --history_policy threshold \
  --history_threshold 4.0 \
  --output_root "${LORA_ROOT}"

python method/dreambooth/tasks/infer_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --user_ids "${USER_ID}" \
  --lora_root "${LORA_ROOT}" \
  --output_dir "${INFER_OUT}" \
  --num_images_per_sample 1

# 评估：目标图像 = stage2_candidates 中 preference_score 最高的那张（代码已按此选择；若候选缺失则 fallback 到 target_item_info.image_path）
python method/dreambooth/tasks/eval_userpref.py \
  --user_ids "${USER_ID}" \
  --infer_output_dir "${INFER_OUT}" \
  --output_json "${EVAL_JSON}"

echo "Done."
