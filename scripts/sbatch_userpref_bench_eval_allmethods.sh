#!/bin/bash
#SBATCH -p debug
#SBATCH -t 0-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --mem=64G
#SBATCH --job-name=bench_eval_viz
#SBATCH --output=logs/bench_eval_viz_%j.out
#SBATCH --error=logs/bench_eval_viz_%j.err

echo "[bench_eval_viz] START $(date -Is) job=${SLURM_JOB_ID:-?} host=$(hostname)"

set -euo pipefail

export PYTHONUNBUFFERED=1

# 统一缓存到数据盘：避免计算节点 $HOME 与登录机不一致时重复下载/写穿 NFS 极慢
export TORCH_HOME="${TORCH_HOME:-/data-nfs/gpu2/u18871384022/.cache/torch}"
export HF_HOME="${HF_HOME:-/data-nfs/gpu2/u18871384022/.cache/huggingface}"

# HuggingFace 国内镜像（https://hf-mirror.com/）；公开权重一般无需 HF_TOKEN
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"

if [[ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/opt/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] No conda.sh found." >&2
  exit 3
fi
conda activate pmg-bench
echo "[conda] HF_ENDPOINT=${HF_ENDPOINT} python=$(command -v python)"
echo "[cache] TORCH_HOME=${TORCH_HOME} HF_HOME=${HF_HOME}"

REPO_ROOT="/data-nfs/gpu2/u18871384022/PMG-Bench"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
mkdir -p "${REPO_ROOT}/logs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# 与训练脚本使用同一 RUN_NAME，或手动 export RUN_NAME=bench_fair_YYYYMMDD
RUN_NAME="${RUN_NAME:-bench_fair_latest}"
MET_DIR="${REPO_ROOT}/outputs/experiments/bench_metrics/${RUN_NAME}"
FIG_DIR="${REPO_ROOT}/outputs/experiments/bench_paper_figs/${RUN_NAME}"
mkdir -p "${MET_DIR}" "${FIG_DIR}"

TEST_JSON="${REPO_ROOT}/data/userpref_v1/processed_dataset/test.json"

# IP-Adapter：已有全量结果（仅重算统一指标）
IP_JSON="${IP_JSON:-${REPO_ROOT}/outputs/userpref_v1_ip_adapter/userpref_ip_adapter_exp1/generation_results.json}"
IP_EVAL="${MET_DIR}/ip_adapter_eval.json"

# PMG：默认使用此前全量 infer 目录（可 export PMG_INFER_DIR=... 覆盖）
PMG_INFER_DIR="${PMG_INFER_DIR:-${REPO_ROOT}/outputs/userpref_v1_pmg_infer/userpref_pmg_exp1}"
PMG_EVAL="${MET_DIR}/pmg_eval.json"

TI_EVAL="${MET_DIR}/textual_inversion_eval.json"
DB_EVAL="${MET_DIR}/dreambooth_eval.json"
TI_INFER_DIR="${REPO_ROOT}/outputs/userpref_v1_textual_inversion_infer/${RUN_NAME}"
DB_INFER_DIR="${REPO_ROOT}/outputs/userpref_v1_dreambooth_infer/${RUN_NAME}"

echo "RUN_NAME=${RUN_NAME}"

if [[ -f "${IP_JSON}" ]]; then
  python -u experiments/exp3_bench_metrics/run_eval_benchmark.py \
    --test_json "${TEST_JSON}" \
    --input "${IP_JSON}" \
    --output_json "${IP_EVAL}" \
    --compute_hps \
    --compute_fid
else
  echo "[WARN] Skip IP-Adapter eval, missing: ${IP_JSON}"
fi

if [[ -d "${PMG_INFER_DIR}" ]]; then
  python -u experiments/exp3_bench_metrics/run_eval_benchmark.py \
    --test_json "${TEST_JSON}" \
    --input "${PMG_INFER_DIR}" \
    --output_json "${PMG_EVAL}" \
    --compute_hps \
    --compute_fid
else
  echo "[WARN] Skip PMG eval, missing dir: ${PMG_INFER_DIR}"
fi

if [[ -d "${TI_INFER_DIR}" ]]; then
  echo "[TI] Recomputing metrics from ${TI_INFER_DIR}"
  python -u experiments/exp3_bench_metrics/run_eval_benchmark.py \
    --test_json "${TEST_JSON}" \
    --input "${TI_INFER_DIR}" \
    --output_json "${TI_EVAL}" \
    --compute_hps \
    --compute_fid
elif [[ -f "${TI_EVAL}" ]]; then
  echo "[TI] Using existing ${TI_EVAL} (no infer dir: ${TI_INFER_DIR})"
else
  echo "[WARN] Missing TI eval and infer dir: ${TI_EVAL} / ${TI_INFER_DIR}"
fi

if [[ -d "${DB_INFER_DIR}" ]]; then
  echo "[DB] Recomputing metrics from ${DB_INFER_DIR}"
  python -u experiments/exp3_bench_metrics/run_eval_benchmark.py \
    --test_json "${TEST_JSON}" \
    --input "${DB_INFER_DIR}" \
    --output_json "${DB_EVAL}" \
    --compute_hps \
    --compute_fid
elif [[ -f "${DB_EVAL}" ]]; then
  echo "[DB] Using existing ${DB_EVAL} (no infer dir: ${DB_INFER_DIR})"
else
  echo "[WARN] Missing DreamBooth eval and infer dir: ${DB_EVAL} / ${DB_INFER_DIR}"
fi

# 论文图：至少要有 IP + PMG；TI/DB 在文件存在时自动加入
SPECS=( "ip_adapter=${IP_EVAL}" "pmg=${PMG_EVAL}" )
[[ -f "${TI_EVAL}" ]] && SPECS+=( "textual_inversion=${TI_EVAL}" )
[[ -f "${DB_EVAL}" ]] && SPECS+=( "dreambooth=${DB_EVAL}" )

python -u experiments/exp3_bench_metrics/plot_paper_figs.py \
  --test_json "${TEST_JSON}" \
  --out_dir "${FIG_DIR}" \
  --eval_specs "${SPECS[@]}" \
  --data_root "${REPO_ROOT}/data/userpref_v1" \
  --case_users "2,11,25" \
  --case_users_history "2,11,25,78,82,100" \
  --case_users_history_moodboard "2,11,25"

echo "Figures under: ${FIG_DIR}"
echo "Metric JSONs under: ${MET_DIR}"
