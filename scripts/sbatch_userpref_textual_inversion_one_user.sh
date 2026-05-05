#!/bin/bash
#SBATCH -p debug
#SBATCH -t 0-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=64G
#SBATCH --job-name=up_ti_1user
#SBATCH --output=logs/up_ti_1user_%j.out

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pmg-bench

REPO_ROOT="/data-nfs/gpu2/u18871384022/PMG-Bench"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${REPO_ROOT}/logs"

# 单卡更稳
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ===== 需要你改的两个参数 =====
USER_ID="2"
SD15_PATH="/data-nfs/gpu2/u18871384022/models/stable-diffusion-v1-5"

# ===== 输出 =====
TI_ROOT="${REPO_ROOT}/outputs/userpref_v1_textual_inversion"
INFER_OUT="${REPO_ROOT}/outputs/userpref_v1_textual_inversion_infer/${USER_ID}"

mkdir -p "${TI_ROOT}"
mkdir -p "${INFER_OUT}"

echo "Repo: ${REPO_ROOT}"
echo "User: ${USER_ID}"
echo "SD1.5: ${SD15_PATH}"
echo "TI root: ${TI_ROOT}"
echo "Infer out: ${INFER_OUT}"

python method/textual_inversion/tasks/train_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --output_root "${TI_ROOT}" \
  --user_ids "${USER_ID}" \
  --history_policy threshold \
  --history_threshold 4.0 \
  --target_token "[V]" \
  --initializer_token "style" \
  --num_vectors 8 \
  --train_batch_size 2 \
  --max_train_steps 600 \
  --lr 5e-4 \
  --save_interval 100 \
  --image_size 512 \
  --num_workers 0

python method/textual_inversion/tasks/infer_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --ti_root "${TI_ROOT}" \
  --output_dir "${INFER_OUT}" \
  --user_ids "${USER_ID}" \
  --target_token "[V]" \
  --num_vectors 8 \
  --num_images_per_sample 1 \
  --num_inference_steps 50 \
  --guidance_scale 7.5 \
  --seed 42

echo "Done."

