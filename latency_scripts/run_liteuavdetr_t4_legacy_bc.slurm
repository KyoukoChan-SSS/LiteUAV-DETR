#!/usr/bin/env bash
#SBATCH --job-name=liteuav_legacy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
PY="${PY:-$(command -v python || command -v python3 || true)}"
IMAGE_FOLDER="${IMAGE_FOLDER:?Set IMAGE_FOLDER to the validation image directory}"
RUN_BASE="${RUN_BASE:-${REPO}/latency_scripts/results_t4_legacy_bc}"
RUN_ROOT="${RUN_BASE}/job_${SLURM_JOB_ID}"
ULTRA_WEIGHT="${ULTRA_WEIGHT:-${REPO}/checkpoints/stripfact_res_n_visdrone_best_stg2.pth}"
FAST_WEIGHT="${FAST_WEIGHT:-${REPO}/checkpoints/stripfact_res_m_visdrone_best_stg2.pth}"
ULTRA_CONFIG="${REPO}/configs/liteuavdetr/custom/liteuavdetr_ultrafast_visdrone_640.yml"
FAST_CONFIG="${REPO}/configs/liteuavdetr/custom/liteuavdetr_fast_visdrone_640.yml"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

mkdir -p "${RUN_ROOT}"
cd "${REPO}"

run_case() {
    local variant="$1"
    local config="$2"
    local weights="$3"
    local mode="$4"
    local output_dir="${RUN_ROOT}/${variant}"
    mkdir -p "${output_dir}"
    "${PY}" latency_scripts/legacy_latency_bc.py \
        --mode "${mode}" \
        --repo "${REPO}" \
        --config "${config}" \
        --weights "${weights}" \
        --image-folder "${IMAGE_FOLDER}" \
        --input-size 640 640 \
        --warmup 50 \
        --expected-images 548 \
        --device cuda:0 \
        --tag "${variant}_legacy_${mode}" \
        --output "${output_dir}/legacy_${mode,,}.json"
}

echo "job_id=${SLURM_JOB_ID}"
echo "node=${SLURMD_NODENAME:-unknown}"
echo "repo_commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
sha256sum "${ULTRA_WEIGHT}" "${FAST_WEIGHT}"

run_case ultrafast "${ULTRA_CONFIG}" "${ULTRA_WEIGHT}" B
run_case ultrafast "${ULTRA_CONFIG}" "${ULTRA_WEIGHT}" C
run_case fast "${FAST_CONFIG}" "${FAST_WEIGHT}" B
run_case fast "${FAST_CONFIG}" "${FAST_WEIGHT}" C

for result in "${RUN_ROOT}"/{ultrafast,fast}/legacy_{b,c}.json; do
    jq -r '"\(.tag): \(.latency.mean_ms) +/- \(.latency.std_ms) ms (n=\(.num_images))"' "${result}"
done
