#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_ARGS=()
if [[ "${1:-}" == "--include-workspace-runs" ]]; then
  BUILD_ARGS+=("--include-workspace-runs")
  shift
fi

python backend/scripts/build_dashboard_data.py "${BUILD_ARGS[@]}"
python backend/server.py --host 0.0.0.0 --port 80
