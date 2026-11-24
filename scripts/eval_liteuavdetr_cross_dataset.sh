#!/usr/bin/env bash
# VisDrone-trained LiteUAVDETR zero-shot evaluation on the OOD8 manifest.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON="${PYTHON:-python}"

cd "${REPO_DIR}"
exec "${PYTHON}" tools/cross_dataset/run_cross_dataset.py "$@"
