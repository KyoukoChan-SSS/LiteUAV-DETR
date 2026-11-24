#!/usr/bin/env bash
# LiteUAVDETR ONNX export entry point.
# Model implementation adapted from D-FINE: https://github.com/Peterande/D-FINE
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_DIR}"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-exports/onnx}"
IMAGE_SIZE="${IMAGE_SIZE:-640}"
PRECISION="${PRECISION:-fp32}"

mkdir -p "${OUT_DIR}"

export_one() {
  local name="$1"
  local config="$2"
  local weights="$3"
  local output="${OUT_DIR}/${name}_${PRECISION}.onnx"

  "${PYTHON}" tools/deployment/export_liteuavdetr_onnx_precision.py     --config "${config}"     --resume "${weights}"     --output "${output}"     --precision "${PRECISION}"     --image-size "${IMAGE_SIZE}"     --batch-size 1
}

export_one   "liteuavdetr_ultrafast_visdrone"   "configs/liteuavdetr/custom/liteuavdetr_ultrafast_visdrone_640.yml"   "checkpoints/stripfact_res_n_visdrone_best_stg2.pth"

export_one   "liteuavdetr_fast_visdrone"   "configs/liteuavdetr/custom/liteuavdetr_fast_visdrone_640.yml"   "checkpoints/stripfact_res_m_visdrone_best_stg2.pth"

echo "ONNX exports saved under ${OUT_DIR}"
