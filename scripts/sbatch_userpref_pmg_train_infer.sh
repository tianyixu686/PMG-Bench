#!/bin/bash
#SBATCH -p debug
#SBATCH -t 1-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=128G
#SBATCH --job-name=up_pmg_ti
#SBATCH --output=logs/up_pmg_ti_%j.out


set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pmg-bench

REPO_ROOT="/data-nfs/gpu2/u18871384022/PMG-Bench"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
# 强制只用一张 GPU，避免 transformers `device_map="auto"` 跨卡切分导致 device mismatch。
# 如需指定别的卡：提交时带 `CUDA_VISIBLE_DEVICES=3 sbatch ...`
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Reduce fragmentation risk (helps some OOMs)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${REPO_ROOT}/logs"

# ===== 必改：模型路径 =====
LLAMA_PATH="/data-nfs/gpu2/u18871384022/models/Llama-2-7b-hf"   # transformers 格式目录
SD15_PATH="/data-nfs/gpu2/u18871384022/models/stable-diffusion-v1-5"  # diffusers 格式目录

# ===== 可选：数据路径（默认就是 userpref_v1 processed_dataset）=====
TRAIN_JSON="${REPO_ROOT}/data/userpref_v1/processed_dataset/train.json"
VAL_JSON="${REPO_ROOT}/data/userpref_v1/processed_dataset/val.json"
TEST_JSON="${REPO_ROOT}/data/userpref_v1/processed_dataset/test.json"

# ===== 输出目录 =====
# 建议用固定 RUN_NAME，便于被 kill 后重新 sbatch 继续训练
RUN_NAME="${RUN_NAME:-userpref_pmg_exp1}"
TRAIN_OUT_DIR="${REPO_ROOT}/outputs/userpref_v1_pmg/${RUN_NAME}"
INFER_OUT_DIR="${REPO_ROOT}/outputs/userpref_v1_pmg_infer/${RUN_NAME}"

echo "Repo: ${REPO_ROOT}"
echo "Train out: ${TRAIN_OUT_DIR}"
echo "Infer out: ${INFER_OUT_DIR}"
echo "LLaMA: ${LLAMA_PATH}"
echo "SD1.5: ${SD15_PATH}"

mkdir -p "${TRAIN_OUT_DIR}"
mkdir -p "${INFER_OUT_DIR}"

# ===== 自动续训：如果目录下已有 ckpt-step-* / ckpt-epoch*，就用最新的那个 =====
RESUME_FROM=""
if ls -1 "${TRAIN_OUT_DIR}"/ckpt-step-* >/dev/null 2>&1; then
  RESUME_FROM="$(ls -1d "${TRAIN_OUT_DIR}"/ckpt-step-* | sort -V | tail -n 1)"
elif ls -1 "${TRAIN_OUT_DIR}"/ckpt-epoch* >/dev/null 2>&1; then
  RESUME_FROM="$(ls -1d "${TRAIN_OUT_DIR}"/ckpt-epoch* | sort -V | tail -n 1)"
fi
if [[ -n "${RESUME_FROM}" ]]; then
  echo "Auto-resume from: ${RESUME_FROM}"
else
  echo "No prior accelerate checkpoint found; starting fresh."
fi

echo "==== [1/2] Train PMG on userpref ===="
python method/PMG/USERPREF_PMG_TRAIN.py \
  --train_json "${TRAIN_JSON}" \
  --val_json "${VAL_JSON}" \
  --llama_path "${LLAMA_PATH}" \
  --sd_path "${SD15_PATH}" \
  --output_dir "${TRAIN_OUT_DIR}" \
  --model_name "userpref_v1_pmg" \
  --mixed_precision fp16 \
  --weight_dtype fp16 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 12 \
  --num_train_epochs 1 \
  --num_image_prompt 2 \
  --num_prefix_prompt 2 \
  --max_txt_len 512 \
  --validation_steps 200 \
  --save_steps 500 \
  --allow_tf32 \
  ${RESUME_FROM:+--resume_from "${RESUME_FROM}"}

echo "==== Pick latest checkpoint ===="
CKPT=""
if ls -1 "${TRAIN_OUT_DIR}"/model-epoch*.pth >/dev/null 2>&1; then
  CKPT="$(ls -1 "${TRAIN_OUT_DIR}"/model-epoch*.pth | sort -V | tail -n 1)"
elif ls -1 "${TRAIN_OUT_DIR}"/save-steps-*.pth >/dev/null 2>&1; then
  CKPT="$(ls -1 "${TRAIN_OUT_DIR}"/save-steps-*.pth | sort -V | tail -n 1)"
fi

if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] No checkpoint found under ${TRAIN_OUT_DIR}"
  exit 2
fi

echo "Using checkpoint: ${CKPT}"

echo "==== [2/2] Inference with latest checkpoint ===="
python method/PMG/USERPREF_PMG_INFER.py \
  --test_json "${TEST_JSON}" \
  --llama_path "${LLAMA_PATH}" \
  --sd_path "${SD15_PATH}" \
  --checkpoint_path "${CKPT}" \
  --output_dir "${INFER_OUT_DIR}" \
  --num_image_prompt 2 \
  --num_prefix_prompt 2 \
  --max_txt_len 512 \
  --image_size 512 \
  --weight_dtype fp16 \
  --num_images_per_sample 1 \
  --seed 42 \
  --weight_target 1.0 \
  --weight_image 1.0 \
  --weight_preference 0.0

echo "Done."
