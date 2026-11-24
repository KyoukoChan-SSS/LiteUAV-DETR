#!/usr/bin/env bash
# LiteUAVDETR supervised UAVDT training entry point.
# Model implementation adapted from D-FINE: https://github.com/Peterande/D-FINE

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
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
        CONFIG_PATH="configs/liteuavdetr/custom/liteuavdetr_ultrafast_uavdt_640.yml"
        ;;
    fast)
        CONFIG_PATH="configs/liteuavdetr/custom/liteuavdetr_fast_uavdt_640.yml"
        ;;
    *)
        echo "Unsupported MODEL_VARIANT='${MODEL_VARIANT}'. Use ultrafast or fast." >&2
        exit 2
        ;;
esac

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
JOB_ID="${SLURM_JOB_ID:-local}"
RUN_NAME="${RUN_NAME:-liteuavdetr_${MODEL_VARIANT}_uavdt_640_${TIMESTAMP}_${JOB_ID}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/output/liteuavdetr_uavdt}"
LOG_ROOT="${LOG_ROOT:-${REPO_DIR}/logs/liteuavdetr_uavdt}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
LOG_FILE="${LOG_FILE:-${LOG_ROOT}/${RUN_NAME}.log}"

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}" "$(dirname -- "${LOG_FILE}")"

if [[ -n "${DATA_ROOT:-}" ]]; then
    UAVDT_IMAGES="${UAVDT_IMAGES:-${DATA_ROOT}/UAV-benchmark-M}"
    UAVDT_ANN_DIR="${UAVDT_ANN_DIR:-${DATA_ROOT}/coco_annotations}"
fi

U=()
if [[ -n "${UAVDT_IMAGES:-}" || -n "${UAVDT_ANN_DIR:-}" ]]; then
    : "${UAVDT_IMAGES:?Set UAVDT_IMAGES or DATA_ROOT}"
    : "${UAVDT_ANN_DIR:?Set UAVDT_ANN_DIR or DATA_ROOT}"
    U+=(
        "train_dataloader.dataset.img_folder=${UAVDT_IMAGES}"
        "train_dataloader.dataset.ann_file=${UAVDT_ANN_DIR}/UAVDT-DET_train_coco.json"
        "val_dataloader.dataset.img_folder=${UAVDT_IMAGES}"
        "val_dataloader.dataset.ann_file=${UAVDT_ANN_DIR}/UAVDT-DET_test_coco.json"
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
    echo "Start Time: $(date '+%F %T %Z')"
    echo "Host: $(hostname)"
    echo "Workdir: ${REPO_DIR}"
    echo "Repository commit: $(git rev-parse HEAD)"
    echo "Repository changes (UAVDT release files):"
    git status --short -- \
        "${CONFIG_PATH}" \
        configs/dataset/uavdt_detection.yml \
        scripts/train_liteuavdetr_uavdt.sh \
        tools/dataset/uavdt2coco.py
    echo "Run: ${RUN_NAME}"
    echo "Config: ${CONFIG_PATH}"
    echo "Output: ${OUTPUT_DIR}"
    echo "Model variant: ${MODEL_VARIANT}"
    echo "Seed: ${SEED}"
    echo "UAVDT images: ${UAVDT_IMAGES:-<from config>}"
    echo "UAVDT annotations: ${UAVDT_ANN_DIR:-<from config>}"
    python --version
    python -c 'import torch; print(f"torch={torch.__version__} cuda={torch.version.cuda}")'
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
    sha256sum \
        "${CONFIG_PATH}" \
        configs/dataset/uavdt_detection.yml \
        tools/dataset/uavdt2coco.py
    if [[ -n "${UAVDT_ANN_DIR:-}" ]]; then
        sha256sum \
            "${UAVDT_ANN_DIR}/UAVDT-DET_train_coco.json" \
            "${UAVDT_ANN_DIR}/UAVDT-DET_test_coco.json"
    fi
    "${CMD[@]}"
    echo "End Time: $(date '+%F %T %Z')"
    echo "Output saved to: ${OUTPUT_DIR}"
} 2>&1 | tee "${LOG_FILE}"
