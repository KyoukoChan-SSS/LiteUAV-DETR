#!/usr/bin/env bash
# LiteUAVDETR UltraFast/Fast latency suite.
# Model implementation adapted from D-FINE: https://github.com/Peterande/D-FINE
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

TARGET="${TARGET:-local}"
PY="${PY:-$(command -v python || command -v python3 || true)}"
DEVICE="${DEVICE:-cuda:0}"
IMPLEMENTATION="${IMPLEMENTATION:-dfine}"
ULTRALYTICS_ROOT="${ULTRALYTICS_ROOT:-${REPO_DIR}/ultralytics_liteuavdetr}"
ULTRAFAST_WEIGHTS="${ULTRAFAST_WEIGHTS:-${REPO_DIR}/checkpoints/stripfact_res_n_visdrone_best_stg2.pth}"
FAST_WEIGHTS="${FAST_WEIGHTS:-${REPO_DIR}/checkpoints/stripfact_res_m_visdrone_best_stg2.pth}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-500}"
NUM_IMAGES="${NUM_IMAGES:-500}"
OUTPUT_ROOT=""
FOLD=0

usage() {
  cat <<'EOF'
Usage: latency_scripts/run_latency_suite.sh [options]

Runs torch fp32 B/C latency for LiteUAVDETR UltraFast and Fast. The D-FINE
implementation remains the default paper/release path.

Options:
  --target NAME         Output label. Default: local.
  --python PATH         Python executable. Default: active python.
  --device DEVICE       Torch device. Default: cuda:0.
  --implementation API  dfine, ultralytics, or both. Default: dfine.
  --ultralytics-root P  Bundled Ultralytics fork. Default: repository copy.
  --ultrafast-weights P UltraFast D-FINE .pth checkpoint.
  --fast-weights P      Fast D-FINE .pth checkpoint.
  --coco-images PATH    VisDrone validation image directory.
  --coco-ann PATH       VisDrone COCO annotation JSON.
  --image-folder PATH   Image-only fallback for end-to-end timing.
  --warmup N            Warmup iterations. Default: 50.
  --iters N             Timed inference iterations. Default: 500.
  --num-images N        End-to-end images. Default: 500.
  --output-root PATH    Result directory.
  --fold                Fold supported Conv+BN layers before timing.
  --smoke               Use warmup=2, iters=3, num-images=5.
  -h, --help            Show this help.
EOF
}

COCO_IMAGES="${COCO_IMAGES:-}"
COCO_ANN="${COCO_ANN:-}"
IMAGE_FOLDER="${IMAGE_FOLDER:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --implementation) IMPLEMENTATION="$2"; shift 2 ;;
    --ultralytics-root) ULTRALYTICS_ROOT="$2"; shift 2 ;;
    --ultrafast-weights) ULTRAFAST_WEIGHTS="$2"; shift 2 ;;
    --fast-weights) FAST_WEIGHTS="$2"; shift 2 ;;
    --coco-images) COCO_IMAGES="$2"; shift 2 ;;
    --coco-ann) COCO_ANN="$2"; shift 2 ;;
    --image-folder) IMAGE_FOLDER="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --iters) ITERS="$2"; shift 2 ;;
    --num-images) NUM_IMAGES="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --fold) FOLD=1; shift ;;
    --smoke) WARMUP=2; ITERS=3; NUM_IMAGES=5; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

case "${IMPLEMENTATION}" in
  dfine|ultralytics|both) ;;
  *) echo "Unsupported implementation: ${IMPLEMENTATION}" >&2; usage; exit 2 ;;
esac

if [[ -z "${PY}" || ! -x "${PY}" ]]; then
  echo "No executable Python found. Activate the project environment or pass --python." >&2
  exit 1
fi

DATA_ARGS=()
if [[ -d "${COCO_IMAGES}" && -f "${COCO_ANN}" ]]; then
  DATA_ARGS=(--dataset-type coco --coco-images "${COCO_IMAGES}" --coco-ann "${COCO_ANN}")
elif [[ -d "${IMAGE_FOLDER}" ]]; then
  DATA_ARGS=(--dataset-type folder --image-folder "${IMAGE_FOLDER}")
else
  echo "Provide COCO_IMAGES/COCO_ANN or IMAGE_FOLDER." >&2
  exit 1
fi

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${SCRIPT_DIR}/results_${TARGET}_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "${OUTPUT_ROOT}"

FOLD_ARGS=()
RESULT_NAME="torch_fp32_bc"
if [[ "${FOLD}" == "1" ]]; then
  FOLD_ARGS=(--fold)
  RESULT_NAME="torch_fp32_fold_bc"
fi

run_one() {
  local implementation="$1"
  local variant="$2"
  local config="$3"
  local weights="$4"
  local tag="liteuavdetr_${variant}"
  local load_args=(
    --model-type liteuavdetr
    --repo "${REPO_DIR}"
    --config "${REPO_DIR}/${config}"
  )
  if [[ "${implementation}" == "ultralytics" ]]; then
    tag="ultralytics_liteuavdetr_${variant}"
    load_args=(
      --model-type ultralytics
      --ultralytics-root "${ULTRALYTICS_ROOT}"
      --ultralytics-api liteuavdetr
    )
  fi
  local out_dir="${OUTPUT_ROOT}/${tag}"
  mkdir -p "${out_dir}"

  "${PY}" "${SCRIPT_DIR}/benchmarkonly.py" \
    "${load_args[@]}" \
    --weights "${weights}" \
    --device "${DEVICE}" \
    --input-size 640 640 \
    --warmup "${WARMUP}" \
    --iters "${ITERS}" \
    --tests B C \
    --num-images "${NUM_IMAGES}" \
    --tag "${tag}" \
    --output "${out_dir}/${RESULT_NAME}.json" \
    --log "${out_dir}/${RESULT_NAME}.txt" \
    "${FOLD_ARGS[@]}" \
    "${DATA_ARGS[@]}"
}

IMPLEMENTATIONS=("${IMPLEMENTATION}")
if [[ "${IMPLEMENTATION}" == "both" ]]; then
  IMPLEMENTATIONS=(dfine ultralytics)
fi

for implementation in "${IMPLEMENTATIONS[@]}"; do
  run_one \
    "${implementation}" \
    ultrafast \
    configs/liteuavdetr/custom/liteuavdetr_ultrafast_visdrone_640.yml \
    "${ULTRAFAST_WEIGHTS}"

  run_one \
    "${implementation}" \
    fast \
    configs/liteuavdetr/custom/liteuavdetr_fast_visdrone_640.yml \
    "${FAST_WEIGHTS}"
done

"${PY}" "${SCRIPT_DIR}/summarize_latency_results.py" \
  --torch-dir "${OUTPUT_ROOT}" \
  --output "${OUTPUT_ROOT}/summary.csv"

echo "Results: ${OUTPUT_ROOT}"
