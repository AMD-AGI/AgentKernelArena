#!/bin/bash

# Batch run script for geak_benchmark
# Usage: ./run_parallel.sh <gpu_range> [num_configs]
# Example: ./run_parallel.sh 0-3 4    # Uses GPU 0,1,2,3 for config_0 to config_3
#          ./run_parallel.sh 0-7 4    # 8 GPUs (0-7), 4 configs, 2 GPUs per config

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_range> [num_configs]"
    echo "Example: $0 0-3 4    # GPU 0,1,2,3"
    echo "         $0 0-7 4    # GPU 0,1,2,3,4,5,6,7"
    exit 1
fi

GPU_RANGE="$1"
NUM_CONFIGS=${2:-4}

# Parse GPU range (e.g., "0-3" -> 0 1 2 3)
GPU_START=$(echo "$GPU_RANGE" | cut -d'-' -f1)
GPU_END=$(echo "$GPU_RANGE" | cut -d'-' -f2)

# Build GPU array
GPU_ARRAY=()
for g in $(seq $GPU_START $GPU_END); do
    GPU_ARRAY+=($g)
done
NUM_GPUS=${#GPU_ARRAY[@]}
GPU_IDS_STR=$(IFS=','; echo "${GPU_ARRAY[*]}")

CONFIG_DIR="agents/geak_benchmark/task_configs"
AGENT_CONFIG_DIR="agents/geak_benchmark"
MAIN_DIR="/data/yueliu14/AgentKernelArena"
AGENT_CONFIG_TEMPLATE="${AGENT_CONFIG_DIR}/agent_config.yaml"

cd "$MAIN_DIR"

# Read num_parallel from agent_config.yaml
NUM_PARALLEL=$(grep 'num_parallel:' "$AGENT_CONFIG_TEMPLATE" | awk '{print $2}')
NUM_PARALLEL=${NUM_PARALLEL:-1}

# Calculate GPUs per config
GPUS_PER_CONFIG=$NUM_PARALLEL

echo "=========================================="
echo "Starting batch geak_benchmark runs"
echo "Config directory: $CONFIG_DIR"
echo "Available GPUs: $GPU_IDS_STR (total: $NUM_GPUS)"
echo "Number of configs: $NUM_CONFIGS"
echo "GPUs per config (num_parallel): $GPUS_PER_CONFIG"
echo "=========================================="

# Check if GPU count matches exactly
REQUIRED_GPUS=$((NUM_CONFIGS * GPUS_PER_CONFIG))
if [ $NUM_GPUS -ne $REQUIRED_GPUS ]; then
    echo "[ERROR] GPU count mismatch!"
    echo "  Required: num_configs($NUM_CONFIGS) x num_parallel($GPUS_PER_CONFIG) = $REQUIRED_GPUS GPUs"
    echo "  Provided: $NUM_GPUS GPUs ($GPU_IDS_STR)"
    echo "  Please provide exactly $REQUIRED_GPUS GPUs."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Launch each config with assigned GPUs
GPU_INDEX=0
for i in $(seq 0 $((NUM_CONFIGS - 1))); do
    CONFIG_FILE="${CONFIG_DIR}/config_${i}.yaml"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "[WARN] Config file not found: $CONFIG_FILE, skipping..."
        continue
    fi
    
    # Assign GPUs for this config
    ASSIGNED_GPUS=""
    for j in $(seq 0 $((GPUS_PER_CONFIG - 1))); do
        IDX=$(( (GPU_INDEX + j) % NUM_GPUS ))
        if [ -z "$ASSIGNED_GPUS" ]; then
            ASSIGNED_GPUS="${GPU_ARRAY[$IDX]}"
        else
            ASSIGNED_GPUS="${ASSIGNED_GPUS},${GPU_ARRAY[$IDX]}"
        fi
    done
    GPU_INDEX=$((GPU_INDEX + GPUS_PER_CONFIG))
    
    # Create temporary agent_config for this run
    TEMP_AGENT_CONFIG="${CONFIG_DIR}/agent_config_${i}.yaml"
    
    # Read original configs line and append --gpu-ids
    ORIGINAL_CONFIGS=$(grep "configs:" "$AGENT_CONFIG_TEMPLATE" | sed "s/configs: *'\(.*\)'/\1/" | sed 's/configs: *"\(.*\)"/\1/')
    
    # Generate temporary agent_config.yaml with updated configs
    sed "s|configs:.*|configs: '${ORIGINAL_CONFIGS} --gpu-ids ${ASSIGNED_GPUS}'|" "$AGENT_CONFIG_TEMPLATE" > "$TEMP_AGENT_CONFIG"
    
    echo "[INFO] Launching config_${i}.yaml with GPUs: $ASSIGNED_GPUS"
    echo "[INFO] Agent config: $TEMP_AGENT_CONFIG"
    
    GEAK_AGENT_CONFIG="$MAIN_DIR/$TEMP_AGENT_CONFIG" \
        nohup python main.py --config_name "$CONFIG_FILE" \
        > "logs/geak_benchmark_config_${i}.log" 2>&1 &
    
    echo "[INFO] PID: $! - Log: logs/geak_benchmark_config_${i}.log"
done

echo "=========================================="
echo "All configs launched!"
echo "Use 'ps aux | grep main.py' to check running processes"
echo "Use 'tail -f logs/geak_benchmark_config_*.log' to monitor logs"
echo "=========================================="
