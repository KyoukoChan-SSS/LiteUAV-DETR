#!/usr/bin/env bash
# Build fp16 TensorRT engines from the bundled ONNX files. Run ON THE JETSON
# (engines are GPU/CUDA/TensorRT-version specific). Reads trt_manifest.json.
#
#   ./build_trt.sh                      # build all, workspace 2048 MB (Jetson-safe)
#   WORKSPACE=1024 ./build_trt.sh       # lower workspace for Orin Nano / low RAM
#   TRTEXEC=/usr/src/tensorrt/bin/trtexec ./build_trt.sh   # explicit trtexec path
set -euo pipefail

DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRTEXEC="${TRTEXEC:-trtexec}"
WORKSPACE="${WORKSPACE:-2048}"          # MB; Jetson-safe default (5090 can use 4096)
ENGINE_DIR="${ENGINE_DIR:-${DIR}/engines}"
MANIFEST="${DIR}/trt_manifest.json"

command -v "${TRTEXEC}" >/dev/null 2>&1 || {
  echo "trtexec not found: ${TRTEXEC}" >&2
  echo "On Jetson it usually lives at /usr/src/tensorrt/bin/trtexec — pass TRTEXEC=..." >&2
  exit 1
}
mkdir -p "${ENGINE_DIR}"

# Parse manifest with python if available, else fall back to a simple grep.
mapfile -t ROWS < <(python3 - "$MANIFEST" <<'PY' 2>/dev/null || true
import json, sys
d = json.load(open(sys.argv[1]))
for m in d["models"]:
    print(f"{m['tag']}\t{m['onnx']}\t{m['image_size']}")
PY
)

if [[ "${#ROWS[@]}" -eq 0 ]]; then
  echo "Could not parse ${MANIFEST} (need python3)." >&2
  exit 1
fi

for row in "${ROWS[@]}"; do
  tag="${row%%$'\t'*}"; rest="${row#*$'\t'}"
  onnx="${DIR}/${rest%%$'\t'*}"; img="${rest##*$'\t'}"
  engine="${ENGINE_DIR}/${tag}_fp16.engine"
  if [[ ! -f "${onnx}" ]]; then
    echo "[skip] ${tag}: ONNX missing (${onnx}) — export it off-device first." >&2
    continue
  fi
  echo "[build] ${tag}  img=${img}  workspace=${WORKSPACE}MB"
  "${TRTEXEC}" \
    "--onnx=${onnx}" \
    "--saveEngine=${engine}" \
    --fp16 \
    "--minShapes=images:1x3x${img}x${img},orig_target_sizes:1x2" \
    "--optShapes=images:1x3x${img}x${img},orig_target_sizes:1x2" \
    "--maxShapes=images:1x3x${img}x${img},orig_target_sizes:1x2" \
    "--memPoolSize=workspace:${WORKSPACE}" \
    > "${ENGINE_DIR}/${tag}_build.log" 2>&1 && echo "  ok -> ${engine}" \
    || { echo "  FAILED (see ${ENGINE_DIR}/${tag}_build.log)"; continue; }
done
echo "engines in ${ENGINE_DIR}"
