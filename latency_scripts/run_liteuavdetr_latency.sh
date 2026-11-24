#!/usr/bin/env bash
# LiteUAVDETR latency entry point.
# Model implementation adapted from D-FINE: https://github.com/Peterande/D-FINE
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_latency_suite.sh" "$@"
