#!/usr/bin/env bash
# LiteUAVDETR supervised UAVDT evaluation entry point.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"
MODEL_VARIANT="${MODEL_VARIANT:-fast}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the matching UAVDT-trained .pth file}"

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

if [[ -n "${DATA_ROOT:-}" ]]; then
    UAVDT_IMAGES="${UAVDT_IMAGES:-${DATA_ROOT}/UAV-benchmark-M}"
    UAVDT_ANN_DIR="${UAVDT_ANN_DIR:-${DATA_ROOT}/coco_annotations}"
fi

U=()
if [[ -n "${UAVDT_IMAGES:-}" || -n "${UAVDT_ANN_DIR:-}" ]]; then
    : "${UAVDT_IMAGES:?Set UAVDT_IMAGES or DATA_ROOT}"
    : "${UAVDT_ANN_DIR:?Set UAVDT_ANN_DIR or DATA_ROOT}"
    U+=(
        "val_dataloader.dataset.img_folder=${UAVDT_IMAGES}"
        "val_dataloader.dataset.ann_file=${UAVDT_ANN_DIR}/UAVDT-DET_test_coco.json"
    )
fi

cd "${REPO_DIR}"
CMD=(
    "${PYTHON}"
    train.py
    -c "${CONFIG_PATH}"
    -r "${CHECKPOINT}"
    --test-only
    -d "${DEVICE}"
)
if (( ${#U[@]} > 0 )); then
    CMD+=(-u "${U[@]}")
fi
CMD+=("$@")

PYTHONDONTWRITEBYTECODE=1 "${CMD[@]}"
