#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python backend/scripts/build_dashboard_data.py
python backend/server.py --host 0.0.0.0 --port 80
