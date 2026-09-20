#!/usr/bin/env bash
set -euo pipefail
TASK_UT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$TASK_UT/unittest.py" "$@"
