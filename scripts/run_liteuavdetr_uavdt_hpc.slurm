#!/usr/bin/env bash
#SBATCH --job-name=liteuav_uavdt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_ROOT="${RUN_ROOT:-${REPO_DIR}/output/liteuavdetr_uavdt_hpc}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the processed UAVDT directory}"
MODEL_VARIANT="${MODEL_VARIANT:-fast}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_ROOT}/checkpoints}"
LOG_ROOT="${LOG_ROOT:-${RUN_ROOT}/logs}"
MASTER_PORT="${MASTER_PORT:-$((7000 + (${SLURM_JOB_ID:-0} % 2000)))}"

export REPO_DIR RUN_ROOT DATA_ROOT MODEL_VARIANT SEED OUTPUT_ROOT LOG_ROOT MASTER_PORT
export NPROC_PER_NODE=1

cd "${REPO_DIR}"
bash scripts/train_liteuavdetr_uavdt.sh
