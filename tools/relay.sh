#!/usr/bin/env bash
# One relay step: pick a live reservation, clean root residue, launch `pending`.
#
# Spur reservations on this cluster live ~5 minutes while a single correctness leg
# takes tens of minutes, so a detached sweep rarely finishes. Progress accumulates
# in _results/ instead (see tools/summarize_runs.py), and this script is meant to
# be re-run: each invocation resumes exactly where the last one died.
set -uo pipefail
SUITE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JOB="$(timeout 90 nssh -l 2>/dev/null | awk 'NR>1{print $1; exit}')"
NODE="$(timeout 90 nssh -l 2>/dev/null | awk 'NR>1{print $2; exit}')"
[[ -n "$JOB" ]] || { echo "no live reservation"; exit 1; }
echo "relay on job $JOB ($NODE)"

# Root residue from a killed container blocks both clean.sh and build_suite.
timeout 280 nssh "$JOB" "docker run --rm -u 0 -v /shared_nfs:/shared_nfs \
  --entrypoint /bin/bash rocm/primus:v26.6 -c 'bash $SUITE/tools/clean.sh'" >/dev/null 2>&1
bash "$SUITE/tools/clean.sh" >/dev/null 2>&1

timeout 280 "$SUITE/tools/run_on_gpu.sh" "$JOB" pending 2>&1 \
  | grep -E "queued|launched|nothing to do|ERROR"
