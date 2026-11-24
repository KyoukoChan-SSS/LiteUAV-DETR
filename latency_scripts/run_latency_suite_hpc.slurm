#!/usr/bin/env bash
#SBATCH --job-name=liteuav_latency
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR}}"
PY="${PY:-python}"
COCO_IMAGES="${COCO_IMAGES:?Set COCO_IMAGES to the VisDrone validation image directory}"
COCO_ANN="${COCO_ANN:?Set COCO_ANN to the VisDrone COCO annotation JSON}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-500}"
NUM_IMAGES="${NUM_IMAGES:-500}"
FOLD="${FOLD:-0}"
IMPLEMENTATION="${IMPLEMENTATION:-dfine}"
ULTRALYTICS_ROOT="${ULTRALYTICS_ROOT:-${REPO_DIR}/ultralytics_liteuavdetr}"
ULTRAFAST_WEIGHTS="${ULTRAFAST_WEIGHTS:-${REPO_DIR}/checkpoints/stripfact_res_n_visdrone_best_stg2.pth}"
FAST_WEIGHTS="${FAST_WEIGHTS:-${REPO_DIR}/checkpoints/stripfact_res_m_visdrone_best_stg2.pth}"

FOLD_ARGS=()
if [[ "${FOLD}" == "1" || "${FOLD}" == "true" ]]; then
  FOLD_ARGS=(--fold)
fi

cd "${REPO_DIR}"
srun bash latency_scripts/run_latency_suite.sh \
  --target hpc \
  --python "${PY}" \
  --implementation "${IMPLEMENTATION}" \
  --ultralytics-root "${ULTRALYTICS_ROOT}" \
  --ultrafast-weights "${ULTRAFAST_WEIGHTS}" \
  --fast-weights "${FAST_WEIGHTS}" \
  --coco-images "${COCO_IMAGES}" \
  --coco-ann "${COCO_ANN}" \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --num-images "${NUM_IMAGES}" \
  --output-root "latency_scripts/results_hpc_${SLURM_JOB_ID}" \
  "${FOLD_ARGS[@]}"
