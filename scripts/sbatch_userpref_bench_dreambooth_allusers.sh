#!/bin/bash
#SBATCH -p debug
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=64G
#SBATCH --job-name=bench_db_all
#SBATCH --output=logs/bench_db_all_%j.out

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pmg-bench

REPO_ROOT="/data-nfs/gpu2/u18871384022/PMG-Bench"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${REPO_ROOT}/logs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SD15_PATH="${SD15_PATH:-/data-nfs/gpu2/u18871384022/models/stable-diffusion-v1-5}"
RUN_NAME="${RUN_NAME:-bench_fair_latest}"

LORA_ROOT="${REPO_ROOT}/outputs/userpref_v1_dreambooth_lora/${RUN_NAME}"
INFER_OUT="${REPO_ROOT}/outputs/userpref_v1_dreambooth_infer/${RUN_NAME}"
EVAL_JSON="${REPO_ROOT}/outputs/experiments/bench_metrics/${RUN_NAME}/dreambooth_eval.json"

mkdir -p "$(dirname "${EVAL_JSON}")"
mkdir -p "${LORA_ROOT}" "${INFER_OUT}"

echo "RUN_NAME=${RUN_NAME}"
echo "LORA_ROOT=${LORA_ROOT}"
echo "INFER_OUT=${INFER_OUT}"

# DreamBooth LoRA：与 DreamBench 表对齐（eff batch=4）
python method/dreambooth/tasks/train_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --output_root "${LORA_ROOT}" \
  --history_policy threshold \
  --history_threshold 4.0 \
  --max_train_steps 500 \
  --learning_rate 5e-5 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --rank 4

python method/dreambooth/tasks/infer_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --lora_root "${LORA_ROOT}" \
  --output_dir "${INFER_OUT}" \
  --num_images_per_sample 1 \
  --num_inference_steps 100 \
  --guidance_scale 7.5 \
  --seed 42

python experiments/exp3_bench_metrics/run_eval_benchmark.py \
  --test_json "${REPO_ROOT}/data/userpref_v1/processed_dataset/test.json" \
  --input "${INFER_OUT}" \
  --output_json "${EVAL_JSON}" \
  --compute_hps \
  --compute_fid

echo "Done DreamBooth. Eval: ${EVAL_JSON}"
