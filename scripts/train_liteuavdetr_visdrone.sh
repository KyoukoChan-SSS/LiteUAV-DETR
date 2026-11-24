#!/bin/bash
# LiteUAVDETR training entry point.
# Model implementation adapted from D-FINE: https://github.com/Peterande/D-FINE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${REPO_DIR}"

if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun not found. Activate the project environment first." >&2
    exit 127
fi

MODEL_VARIANT="${MODEL_VARIANT:-fast}"
MASTER_PORT="${MASTER_PORT:-7782}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
SEED="${SEED:-0}"

case "${MODEL_VARIANT}" in
    ultrafast)
        MODEL_VARIANT="ultrafast"
        CONFIG_PATH="configs/liteuavdetr/custom/liteuavdetr_ultrafast_visdrone_640.yml"
        ;;
    fast)
        MODEL_VARIANT="fast"
        CONFIG_PATH="configs/liteuavdetr/custom/liteuavdetr_fast_visdrone_640.yml"
        ;;
    *)
        echo "Unsupported MODEL_VARIANT='${MODEL_VARIANT}'. Use ultrafast or fast." >&2
        exit 2
        ;;
esac

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME:-liteuavdetr_${MODEL_VARIANT}_visdrone_640_${TIMESTAMP}_${JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/output/liteuavdetr_visdrone}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/liteuavdetr_visdrone}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_NAME}.log}"

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}" "$(dirname "${LOG_FILE}")"

U=()
if [[ -n "${DATA_ROOT:-}" ]]; then
    U+=(
        "train_dataloader.dataset.img_folder=${DATA_ROOT}/DET-train/images"
        "train_dataloader.dataset.ann_file=${DATA_ROOT}/annotation_external_tool/VisDrone2019-DET_train_coco.json"
        "val_dataloader.dataset.img_folder=${DATA_ROOT}/DET-val/images"
        "val_dataloader.dataset.ann_file=${DATA_ROOT}/annotation_external_tool/VisDrone2019-DET_val_coco.json"
    )
fi

CMD=(
    torchrun
    --master_port="${MASTER_PORT}"
    --nproc_per_node="${NPROC_PER_NODE}"
    train.py
    -c "${CONFIG_PATH}"
    --seed="${SEED}"
    --output-dir "${OUTPUT_DIR}"
    --summary-dir "${TENSORBOARD_DIR}"
)

if (( ${#U[@]} > 0 )); then
    CMD+=(-u "${U[@]}")
fi

CMD+=("$@")

{
    echo "============================================"
    echo "Start Time: $(date '+%F %T %Z')"
    echo "Host: $(hostname)"
    echo "Workdir: ${REPO_DIR}"
    echo "Run: ${RUN_NAME}"
    echo "Config: ${CONFIG_PATH}"
    echo "Output: ${OUTPUT_DIR}"
    echo "TensorBoard: ${TENSORBOARD_DIR}"
    echo "Model variant: ${MODEL_VARIANT}"
    echo "Seed: ${SEED}"
    echo "Data root override: ${DATA_ROOT:-<from config>}"
    echo "Runtime overrides (${#U[@]}):"
    for o in "${U[@]}"; do
        echo "  ${o}"
    done
    echo "============================================"

    "${CMD[@]}"

    echo "============================================"
    echo "End Time: $(date '+%F %T %Z')"
    echo "Output saved to: ${OUTPUT_DIR}"
} 2>&1 | tee "${LOG_FILE}"
