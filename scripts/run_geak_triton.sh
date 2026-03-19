#!/usr/bin/env bash
# Run GEAK-v3 Triton benchmark with 2-stream parallelism (GPUs 0-3 and 4-7).
# Everything runs inside the GEAK Docker container.
#
# Usage:
#   ./scripts/run_geak_triton.sh                          # all 8 kernels, heterogeneous, memory OFF
#   ./scripts/run_geak_triton.sh config_geak_triton_2kernel.yaml  # 2 kernels only
#   GEAK_CONFIG_NAME=heterogeneous_memory_on ./scripts/run_geak_triton.sh  # memory ON
#
# Requires: AMD_LLM_API_KEY env var, geak-agent Docker container running
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AKA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_NAME="${1:-config_geak_triton.yaml}"
CONFIG_NAME="${CONFIG_NAME#--config-name=}"
[[ -f "$AKA_ROOT/$CONFIG_NAME" ]] || CONFIG_NAME="config_geak_triton.yaml"

GPU_A="0,1,2,3"
GPU_B="4,5,6,7"

CONTAINER="geak-agent-${USER:-sapmajum}"
export GEAK_CONFIG_NAME="${GEAK_CONFIG_NAME:-heterogeneous_memory_off}"
export GEAK_SRC="${GEAK_SRC:-/home/sapmajum/GEAK-agent-filtering-and-cli-unification/src}"

# Ensure container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
  if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Starting stopped container $CONTAINER..."
    docker start "$CONTAINER"
    sleep 3
  else
    echo "ERROR: Container $CONTAINER not found."
    echo "Create it first: AMD_LLM_API_KEY=<key> /path/to/GEAK/scripts/run-docker.sh -- echo ready"
    exit 1
  fi
fi

echo "============================================================"
echo "  GEAK-v3 Triton Benchmark (AgentKernelArena)"
echo "  Everything runs inside Docker container: $CONTAINER"
echo "============================================================"
echo "  Config:       $CONFIG_NAME"
echo "  Mode:         $GEAK_CONFIG_NAME"
echo "  Stream A:     GPUs $GPU_A"
echo "  Stream B:     GPUs $GPU_B"
echo "  GEAK_SRC:     $GEAK_SRC"
echo "  AKA_ROOT:     $AKA_ROOT"
echo "============================================================"
echo ""

# Generate per-stream configs by splitting tasks
python3 - "$AKA_ROOT/$CONFIG_NAME" "$AKA_ROOT" << 'PYEOF'
import sys, yaml
from pathlib import Path

config_path, aka_root = sys.argv[1], sys.argv[2]
with open(config_path) as f:
    cfg = yaml.safe_load(f)

tasks = cfg.get("tasks", [])
stream_a = tasks[0::2]
stream_b = tasks[1::2]

for suffix, task_list in [("_stream_a", stream_a), ("_stream_b", stream_b)]:
    out = dict(cfg)
    out["tasks"] = task_list
    out_path = Path(aka_root) / f".tmp_config{suffix}.yaml"
    with open(out_path, "w") as f:
        yaml.dump(out, f, default_flow_style=False)
    print(f"  Stream config: {out_path} ({len(task_list)} tasks: {', '.join(task_list)})")
PYEOF

echo ""
echo "[$(date -Iseconds)] Starting Stream A (GPUs $GPU_A)..."
docker exec \
  -e "GEAK_CONFIG_NAME=$GEAK_CONFIG_NAME" \
  -e "GEAK_SRC=$GEAK_SRC" \
  -e "AMD_LLM_API_KEY=${AMD_LLM_API_KEY:-}" \
  -e "GEAK_GPU_IDS=$GPU_A" \
  -w "$AKA_ROOT" "$CONTAINER" \
  python3 main.py --config_name "$AKA_ROOT/.tmp_config_stream_a.yaml" &
PID_A=$!

echo "[$(date -Iseconds)] Starting Stream B (GPUs $GPU_B)..."
docker exec \
  -e "GEAK_CONFIG_NAME=$GEAK_CONFIG_NAME" \
  -e "GEAK_SRC=$GEAK_SRC" \
  -e "AMD_LLM_API_KEY=${AMD_LLM_API_KEY:-}" \
  -e "GEAK_GPU_IDS=$GPU_B" \
  -w "$AKA_ROOT" "$CONTAINER" \
  python3 main.py --config_name "$AKA_ROOT/.tmp_config_stream_b.yaml" &
PID_B=$!

echo ""
echo "Stream A PID: $PID_A (GPUs $GPU_A)"
echo "Stream B PID: $PID_B (GPUs $GPU_B)"
echo "Waiting for both streams..."
echo ""

FAIL=0
wait $PID_A || { echo "[$(date -Iseconds)] Stream A FAILED (exit $?)"; FAIL=1; }
wait $PID_B || { echo "[$(date -Iseconds)] Stream B FAILED (exit $?)"; FAIL=1; }

# Cleanup temp configs
rm -f "$AKA_ROOT/.tmp_config_stream_a.yaml" "$AKA_ROOT/.tmp_config_stream_b.yaml"

echo ""
echo "============================================================"
echo "  Benchmark Complete"
echo "============================================================"
echo ""

# Print results summary
python3 - "$AKA_ROOT" << 'PYEOF'
import sys
from pathlib import Path

aka_root = Path(sys.argv[1])
for ws_dir in sorted(aka_root.glob("workspace_*_geak_v3_triton")):
    for run_dir in sorted(ws_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
            continue
        print(f"Run: {run_dir.name}")
        print(f"{'Task':<60} {'Status':<12} {'Speedup'}")
        print("-" * 85)
        for task_dir in sorted(run_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "reports":
                continue
            if task_dir.name.endswith("_logs"):
                continue
            result_file = task_dir / "task_result.yaml"
            if result_file.exists():
                import yaml
                with open(result_file) as f:
                    r = yaml.safe_load(f) or {}
                speedup = r.get("speedup", "N/A")
                status = r.get("status", "unknown")
                print(f"  {task_dir.name:<58} {status:<12} {speedup}")
            else:
                print(f"  {task_dir.name:<58} {'no result':<12}")
        break
PYEOF

exit $FAIL
