#!/bin/bash
# Run the immutable op unittest with the CORRECT baseline environment.
# PYTHONPATH=baseline_overlay is LOAD-BEARING: it makes
# sglang.kernels.ops.attention.decode_attention resolve to the ACCEPTED [geak-tune r2] overlay
# (BLOCK_N=32, waves_per_eu=2). Without it the denominator is the stock BLOCK_N=16 kernel and
# every speedup is inflated by the ~1.73x already banked in the tuning round.
# Usage: bash run_unittest.sh <gpu_id>   (single id from the OPTIMIZATION pool, not the serving TP set)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${1:?usage: run_unittest.sh <gpu_id>}"
GEAK="${GEAK_ROOT:-/shared_nfs/hongtaom/qwen3_14B/hl_matrix_0824/deps/kimi-k3/GEAK}"
cd "$HERE"
PYTHONPATH="$HERE/baseline_overlay${PYTHONPATH:+:$PYTHONPATH}" \
  bash "$GEAK/kernel_workflow/scripts/gpu_lock.sh" "$GPU" python3 unittest.py
