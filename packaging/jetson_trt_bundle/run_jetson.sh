#!/usr/bin/env bash
# One-shot Jetson run: build engines from ONNX, then time B/C for each, per
# trt_manifest.json. Results in results/<tag>_trt_bc.json.
#
#   export IMAGE_FOLDER=/path/to/images     # or set COCO_IMAGES and COCO_ANN
#   export COCO_IMAGES=/path/VisDrone.../DET-val/images
#   export COCO_ANN=/path/VisDrone.../VisDrone2019-DET_val_coco.json
#   ./run_jetson.sh                         # full: warmup 50 / iters 500 / 500 imgs
#   ./run_jetson.sh --smoke                 # warmup 5 / iters 20 / 20 imgs
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-$(command -v python3 || command -v python)}"
ENGINE_DIR="${ENGINE_DIR:-${DIR}/engines}"
OUT_DIR="${OUT_DIR:-${DIR}/results}"
WARMUP="${WARMUP:-50}"; ITERS="${ITERS:-500}"; NUM_IMAGES="${NUM_IMAGES:-500}"
DEVICE="${DEVICE:-cuda:0}"
if [[ -d "${DIR}/../configs" && -d "${DIR}/../src" ]]; then
  REPO_DEFAULT="$(cd -- "${DIR}/.." && pwd)"
else
  REPO_DEFAULT="$(cd -- "${DIR}/../.." && pwd)"
fi
REPO="${REPO:-${REPO_DEFAULT}}"
EXPORT_ONNX="${EXPORT_ONNX:-auto}"  # auto/1 exports missing ONNX from ckpt; 0 skips
IMAGE_FOLDER="${IMAGE_FOLDER:-}"

if [[ "${1:-}" == "--smoke" ]]; then WARMUP=5; ITERS=20; NUM_IMAGES=20; fi

DATA_ARGS=()
if [[ -n "${COCO_IMAGES:-}" && -n "${COCO_ANN:-}" ]]; then
  DATA_ARGS=(--coco-images "${COCO_IMAGES}" --coco-ann "${COCO_ANN}")
elif [[ -d "${IMAGE_FOLDER}" ]]; then
  DATA_ARGS=(--image-folder "${IMAGE_FOLDER}")
else
  echo "Set COCO_IMAGES/COCO_ANN or IMAGE_FOLDER for C latency." >&2
  echo "Checked IMAGE_FOLDER=${IMAGE_FOLDER}" >&2
  exit 1
fi

if [[ "${EXPORT_ONNX}" != "0" && "${EXPORT_ONNX}" != "false" ]]; then
  "${PY}" "${DIR}/export_onnx_from_manifest.py" --manifest "${DIR}/trt_manifest.json" --repo "${REPO}" --python "${PY}"
fi
bash "${DIR}/build_trt.sh"
mkdir -p "${OUT_DIR}"

mapfile -t ROWS < <("${PY}" - "${DIR}/trt_manifest.json" <<'PY'
import json, sys
for m in json.load(open(sys.argv[1]))["models"]:
    print(f"{m['tag']}\t{m['image_size']}")
PY
)

for row in "${ROWS[@]}"; do
  tag="${row%%$'\t'*}"; img="${row##*$'\t'}"
  engine="${ENGINE_DIR}/${tag}_fp16.engine"
  [[ -f "${engine}" ]] || { echo "[skip] ${tag}: no engine (build failed?)"; continue; }
  echo "== ${tag} =="
  "${PY}" "${DIR}/benchmark_trt_bc.py" \
    --engine "${engine}" --tag "${tag}" --image-size "${img}" \
    --manifest "${DIR}/trt_manifest.json" \
    --device "${DEVICE}" --warmup "${WARMUP}" --iters "${ITERS}" --num-images "${NUM_IMAGES}" \
    "${DATA_ARGS[@]}" \
    --output "${OUT_DIR}/${tag}_trt_bc.json"
done
echo "[done] results in ${OUT_DIR}"
