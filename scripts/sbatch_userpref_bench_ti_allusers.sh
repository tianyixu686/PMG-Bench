#!/bin/bash
#SBATCH -p debug
#SBATCH -t 2-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=64G
#SBATCH --job-name=bench_ti_all
#SBATCH --output=logs/bench_ti_all_%j.out

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
# 与 DreamBooth sbatch 使用同一默认名，避免未 export RUN_NAME 时两套输出目录不一致
RUN_NAME="${RUN_NAME:-bench_fair_latest}"

TI_ROOT="${REPO_ROOT}/outputs/userpref_v1_textual_inversion/${RUN_NAME}"
INFER_OUT="${REPO_ROOT}/outputs/userpref_v1_textual_inversion_infer/${RUN_NAME}"
EVAL_JSON="${REPO_ROOT}/outputs/experiments/bench_metrics/${RUN_NAME}/textual_inversion_eval.json"

mkdir -p "$(dirname "${EVAL_JSON}")"
mkdir -p "${TI_ROOT}" "${INFER_OUT}"

echo "RUN_NAME=${RUN_NAME}"
echo "TI_ROOT=${TI_ROOT}"
echo "INFER_OUT=${INFER_OUT}"

# 训练：全部 test 中出现的用户（不传 --user_ids）
# 公平参数见 scripts/BENCH_FAIR_PARAMS.md
python method/textual_inversion/tasks/train_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --output_root "${TI_ROOT}" \
  --history_policy threshold \
  --history_threshold 4.0 \
  --target_token "[V]" \
  --initializer_token "style" \
  --num_vectors 8 \
  --train_batch_size 4 \
  --max_train_steps 1000 \
  --lr 5e-4 \
  --save_interval 250 \
  --image_size 512 \
  --num_workers 0

# 推理：同一 RUN_NAME；扁平 sample_XXXX（不按 user 分子目录）
python method/textual_inversion/tasks/infer_userpref.py \
  --sd15_path "${SD15_PATH}" \
  --ti_root "${TI_ROOT}" \
  --output_dir "${INFER_OUT}" \
  --target_token "[V]" \
  --num_vectors 8 \
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

echo "Done TI. Eval: ${EVAL_JSON}"
