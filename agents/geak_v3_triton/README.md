## `GEAK-V3-Triton`

Triton kernel optimization agent for AgentKernelArena. Uses a two-step pipeline: `geak-preprocess` (profiling, baseline, COMMANDMENT generation) followed by `geak-orchestrate` (multi-round heterogeneous optimization with verified evaluation).

### Setup

```bash
# 1. Clone repos
git clone -b af/heterogeneous-postprocess git@github.com:AMD-AGI/GEAK.git ~/GEAK-agent
git clone -b geak-triton-benchmark git@github.com:AMD-AGI/AgentKernelArena.git ~/AgentKernelArena

# 2. Build Docker
export AMD_LLM_API_KEY="<your-key>"
cd ~/GEAK-agent && scripts/run-docker.sh --rebuild

# 3. Verify
docker exec geak-agent-$USER python3 -c "import minisweagent; print('OK')"
docker exec geak-agent-$USER rocm-smi --showuse | head -10
```

### Running All 16 Triton Kernels (2 Slots)

```bash
# Slot 1: GPUs 0-3 (4 kernels)
docker exec -d \
  -e "GEAK_SRC=$HOME/GEAK-agent/src" \
  -e "GEAK_CONFIG_NAME=heterogeneous_memory_off" \
  -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
  -e "GEAK_GPU_IDS=0,1,2,3" \
  -e "GEAK_MODEL=claude-opus-4-6" \
  -e "GEAK_MODEL_ENSEMBLE=claude-opus-4-6" \
  -e "GEAK_BENCHMARK_ITERATIONS=30" \
  -e "PYTORCH_ROCM_ARCH=gfx950" \
  -w $HOME/AgentKernelArena \
  geak-agent-$USER \
  python3 main.py --config_name config_geak_triton_slot1.yaml

# Slot 2: GPUs 4-7 (4 kernels)
docker exec -d \
  -e "GEAK_SRC=$HOME/GEAK-agent/src" \
  -e "GEAK_CONFIG_NAME=heterogeneous_memory_off" \
  -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
  -e "GEAK_GPU_IDS=4,5,6,7" \
  -e "GEAK_MODEL=claude-opus-4-6" \
  -e "GEAK_MODEL_ENSEMBLE=claude-opus-4-6" \
  -e "GEAK_BENCHMARK_ITERATIONS=30" \
  -e "PYTORCH_ROCM_ARCH=gfx950" \
  -w $HOME/AgentKernelArena \
  geak-agent-$USER \
  python3 main.py --config_name config_geak_triton_slot2.yaml
```

### Config Files

| Config | Kernels |
|--------|---------|
| `config_geak_triton.yaml` | All 8 original kernels |
| `config_geak_triton_slot1.yaml` | Slot 1 subset (customize as needed) |
| `config_geak_triton_slot2.yaml` | Slot 2 subset (customize as needed) |

### Pipeline

The launcher (`launch_agent.py`) calls:
```
Step 1: python3 -m minisweagent.run.preprocess.preprocessor <kernel> --harness <harness> -o <logs_dir>
Step 2: python3 -m minisweagent.run.orchestrator --preprocess-dir <logs_dir> --gpu-ids <gpus> --max-rounds 3 --heterogeneous
```

- **Preprocess**: profiles kernel, measures baseline, generates COMMANDMENT.md
- **Orchestrate**: 3 rounds of heterogeneous LLM-driven optimization with FULL_BENCHMARK verification

### Agent Config

Edit `agents/geak_v3_triton/agent_config.yaml`:
- `orchestrate.max_rounds` — optimization rounds (default: 3)
- `orchestrate.model` — LLM model (default: claude-opus-4.6)
- `geak_env.GEAK_MODEL_ENSEMBLE` — model ensemble for parallel agents
- `configs` — memory on/off, heterogeneous/homogeneous mode

### All 16 Triton Kernels

| # | Kernel | Level | Shapes (B/FB) | Status |
|---|--------|-------|---------------|--------|
| 1 | `llama_ff_triton` | L1 | 3/3 | existing |
| 2 | `moe_routing_sigmoid_top1` | L1 | 25/34 | existing |
| 3 | `fused_append_shared_experts` | L1 | 18/18 | new |
| 4 | `mla_decode` | L1 | ~25/320 | new |
| 5 | `topk` | L2 | 25/80 | existing |
| 6 | `fast_rms_layernorm` | L2 | 1/1 | existing |
| 7 | `lean_atten_paged` | L2 | ~25 | existing |
| 8 | `rope` | L2 | 25/6480 | new |
| 9 | `fused_qkv_rope` | L3 | 25/1200 | existing |
| 10 | `fused_rms_fp8` | L3 | 25/25 | existing |
| 11 | `gemm_a16wfp4` | L3 | 25/57 | existing |
| 12 | `gemm` | L3 | 13/13 | new |
| 13 | `gemm_a16w16_atomic` | L3 | 13/13 | new |
| 14 | `fused_qk_rope_cache_mla` | L3 | 25/128 | new |
| 15 | `fused_mxfp4_quant_moe_sort` | L3 | 24/24 | new |
| 16 | `fused_moe_mxfp4` | L3 | 15/15 | new |

New kernels require `aiter` (AMD inference library). The Docker container has it pre-installed at `/sgl-workspace/aiter/`.

### Monitoring

```bash
# Container health
docker ps --filter name=geak-agent-$USER --format "{{.Status}}"

# Progress
grep -E "Round [0-9]|Task.*completed|FULL_BENCHMARK verified" logs/MI300_geak_v3_triton_*.log | tail -20

# Results
for f in workspace_*/run_*/*/task_result.yaml; do
  [ -f "$f" ] && echo "$(basename $(dirname $f)): $(grep speedup_ratio $f)"
done

# GEAK internal speedups
for d in workspace_*/run_*/*_logs/final_report.json; do
  [ -f "$d" ] && kernel=$(echo "$d"|grep -o 'geak_eval_[^/]*'|sed 's/geak_eval_//;s/_[0-9].*//') \
  && python3 -c "import json;d=json.load(open('$d'));fb=(d.get('round_evaluation') or {}).get('full_benchmark') or {};print(f'  $kernel: verified={fb.get(\"verified_speedup\",\"N/A\")}x')"
done
```

### Mini-SWE Agent (Baseline Comparison)

A simpler single-round agent is also available for comparison:

```bash
docker exec -d \
  -e "GEAK_SRC=$HOME/GEAK-agent/src" \
  -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
  -e "GEAK_GPU_IDS=0,1,2,3" \
  -e "GEAK_MODEL=claude-opus-4-6" \
  -e "GEAK_BENCHMARK_ITERATIONS=30" \
  -e "PYTORCH_ROCM_ARCH=gfx950" \
  -w $HOME/AgentKernelArena \
  geak-agent-$USER \
  python3 main.py --config_name config_mini_swe_triton.yaml
```

Uses 2 parallel agents, 100 step limit, no preprocessing/orchestration. Isolates the value of GEAK's structured pipeline.
