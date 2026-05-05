#!/bin/bash
#SBATCH -p debug
#SBATCH -t 0-08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=64G
#SBATCH --job-name=up_ipad
#SBATCH --output=logs/up_ipad_%j.out

set -euo pipefail

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pmg-bench

REPO_ROOT="/data-nfs/gpu2/u18871384022/PMG-Bench"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${REPO_ROOT}/logs"

# 强制单卡（避免意外多卡切分/驱动 P2P 问题）
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ===== 必改：模型路径 =====
SD15_PATH="/data-nfs/gpu2/u18871384022/models/stable-diffusion-v1-5"   # diffusers 格式目录

# IP-Adapter 目录：需要满足 diffusers 的 `pipe.load_ip_adapter(ip_adapter_path, subfolder="models", weight_name=...)`
# 也就是 `${IP_ADAPTER_PATH}/models/${IP_ADAPTER_WEIGHT}` 必须存在
IP_ADAPTER_PATH="/data-nfs/gpu2/u18871384022/models/ip-adapter"
IP_ADAPTER_WEIGHT="ip-adapter_sd15.bin"

# ===== 数据/输出 =====
DATA_ROOT="${REPO_ROOT}/data/userpref_v1"
TEST_JSON="${DATA_ROOT}/processed_dataset/test.json"

RUN_NAME="${RUN_NAME:-userpref_ip_adapter_exp1}"
OUT_DIR="${REPO_ROOT}/outputs/userpref_v1_ip_adapter/${RUN_NAME}"

mkdir -p "${OUT_DIR}"

echo "Repo: ${REPO_ROOT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "SD1.5: ${SD15_PATH}"
echo "IP-Adapter: ${IP_ADAPTER_PATH} (weight: ${IP_ADAPTER_WEIGHT})"
echo "Test json: ${TEST_JSON}"
echo "Out dir: ${OUT_DIR}"

python method/ip-adater/personalized_generation_userpref.py \
  --test_json "${TEST_JSON}" \
  --data_root "${DATA_ROOT}" \
  --output_dir "${OUT_DIR}" \
  --sd_model_path "${SD15_PATH}" \
  --ip_adapter_path "${IP_ADAPTER_PATH}" \
  --ip_adapter_weight "${IP_ADAPTER_WEIGHT}" \
  --device cuda \
  --num_inference_steps 50 \
  --guidance_scale 10.0 \
  --ip_adapter_scale 0.3 \
  --min_preference_score 4.0 \
  --max_history_images 10 \
  --seed 42

echo "Done. Results: ${OUT_DIR}/generation_results.json"

