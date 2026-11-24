#!/usr/bin/env bash
#SBATCH --job-name=liteuav_t4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
RUN_BASE="${RUN_BASE:-${REPO_DIR}/latency_scripts/results_t4}"
PY="${PY:-$(command -v python || command -v python3 || true)}"
COCO_IMAGES="${COCO_IMAGES:?Set COCO_IMAGES to the VisDrone validation image directory}"
COCO_ANN="${COCO_ANN:?Set COCO_ANN to the VisDrone validation COCO JSON}"
ULTRAFAST_WEIGHTS="${ULTRAFAST_WEIGHTS:-${REPO_DIR}/checkpoints/stripfact_res_n_visdrone_best_stg2.pth}"
FAST_WEIGHTS="${FAST_WEIGHTS:-${REPO_DIR}/checkpoints/stripfact_res_m_visdrone_best_stg2.pth}"
IMPLEMENTATION="${IMPLEMENTATION:-dfine}"
FOLD="${FOLD:-0}"
SMOKE="${SMOKE:-0}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-500}"
NUM_IMAGES="${NUM_IMAGES:-500}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RUN_BASE}/job_${SLURM_JOB_ID}}"

FOLD_ARGS=()
if [[ "${FOLD}" == "1" || "${FOLD}" == "true" ]]; then
    FOLD_ARGS=(--fold)
fi
if [[ "${SMOKE}" == "1" || "${SMOKE}" == "true" ]]; then
    WARMUP=2
    ITERS=3
    NUM_IMAGES=5
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

cd "${REPO_DIR}"
echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "repo_commit=$(git rev-parse HEAD)"
echo "implementation=${IMPLEMENTATION}"
echo "warmup=${WARMUP} iters=${ITERS} num_images=${NUM_IMAGES}"
echo "coco_images=${COCO_IMAGES}"
echo "coco_ann=${COCO_ANN}"
"${PY}" --version
"${PY}" -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
sha256sum \
    configs/liteuavdetr/custom/liteuavdetr_ultrafast_visdrone_640.yml \
    configs/liteuavdetr/custom/liteuavdetr_fast_visdrone_640.yml \
    latency_scripts/benchmarkonly.py \
    latency_scripts/run_latency_suite.sh \
    ultralytics_liteuavdetr/ultralytics/nn/tasks.py \
    ultralytics_liteuavdetr/ultralytics/nn/modules/liteuavdetr.py \
    "${ULTRAFAST_WEIGHTS}" \
    "${FAST_WEIGHTS}" \
    "${COCO_ANN}"

srun bash latency_scripts/run_liteuavdetr_latency.sh \
    --target t4 \
    --python "${PY}" \
    --implementation "${IMPLEMENTATION}" \
    --ultrafast-weights "${ULTRAFAST_WEIGHTS}" \
    --fast-weights "${FAST_WEIGHTS}" \
    --coco-images "${COCO_IMAGES}" \
    --coco-ann "${COCO_ANN}" \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --num-images "${NUM_IMAGES}" \
    --output-root "${OUTPUT_ROOT}" \
    "${FOLD_ARGS[@]}"
