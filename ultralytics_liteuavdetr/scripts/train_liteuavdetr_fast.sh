#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${REPO_ROOT}/.ultralytics-config}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON:-python}" "${SCRIPT_DIR}/train_liteuavdetr.py" "$@" --variant fast
