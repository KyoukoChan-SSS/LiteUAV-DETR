#!/usr/bin/env bash
# LiteUAVDETR HPC wrapper.
# Model implementation adapted from D-FINE: https://github.com/Peterande/D-FINE
#SBATCH --job-name=liteuav_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
MODEL_VARIANT="${MODEL_VARIANT:-fast}"
SEED="${SEED:-0}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/data/VisDrone2019/Image}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/output/liteuavdetr_visdrone}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
MASTER_PORT="${MASTER_PORT:-$((7000 + (${SLURM_JOB_ID:-0} % 2000)))}"

cd "${REPO_DIR}"
export MODEL_VARIANT SEED DATA_ROOT OUTPUT_ROOT LOG_ROOT MASTER_PORT
bash scripts/train_liteuavdetr_visdrone.sh
