## GEAK-V3-Triton

Triton kernel optimization agent for AgentKernelArena. Uses the unified `geak` CLI
which auto-detects Triton harnesses and runs heterogeneous multi-round optimization
with working memory.

### Setup

```bash
# 1. Clone repos
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK && git checkout main && pip install -e .

git clone https://github.com/AMD-AGI/AgentKernelArena.git
cd AgentKernelArena && git checkout geak-triton-common-benchmark

# 2. Docker (recommended — torch + triton + aiter required)
# Use any container with ROCm 7.0+, torch, triton 3.4+, aiter
docker exec <container> pip install -e /path/to/GEAK
docker exec <container> pip install -r /path/to/AgentKernelArena/requirements.txt

# 3. Checkout aiter to pinned commit (required by L1/L2/L3 kernels)
docker exec <container> bash -c \
  "cd /sgl-workspace/aiter && git fetch && git reset --hard && git clean -fd && \
   git checkout 22122345c03991cb8026947b8df05e02f50d1f88"

# 4. Set API key
export AMD_LLM_API_KEY="your-key"
```

### Defaults

The agent uses these defaults (from `agent_config.yaml` → `geak_env`):

| Setting | Default | Description |
|---------|---------|-------------|
| `GEAK_MAX_ROUNDS` | 5 | Optimization rounds per kernel |
| `GEAK_MODEL` | claude-opus-4.6 | LLM model |
| `GEAK_MODEL_ENSEMBLE` | gpt-5.2,claude-opus-4.6 | Model ensemble for parallel agents |
| `GEAK_BENCHMARK_ITERATIONS` | 30 | Benchmark iterations per shape |
| Heterogeneous | auto (Triton → heterogeneous) | Diverse strategy mode |
| Working Memory | ON by default | Cross-round learning |

Override any setting via environment variables in the docker exec command.

### Pipeline

The launcher (`launch_agent.py`) calls the unified `geak` CLI:

```
geak --kernel-url <kernel.py> --test-command 'python3 <harness.py>' \
     --gpu-ids <ids> --num-parallel <N> --yolo --exit-immediately \
     -t <task_prompt.md> -o <logs_dir>
```

GEAK internally handles:
1. **Preprocessing**: harness validation, profiling, baseline capture, COMMANDMENT generation
2. **Orchestration**: N rounds of heterogeneous LLM-driven optimization with 4 parallel agents
3. **Evaluation**: FULL_BENCHMARK verification + profiling per round
4. **Selection**: best patch across all rounds

AKA reads GEAK's JSON output directly (`final_report.json`, `round_N_evaluation.json`)
instead of re-running benchmarks.

### Running All 18 Triton Kernels (2 Slots, 8 GPUs)

```bash
# Slot 1: GPUs 0-3 (9 kernels)
docker exec -d \
  -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
  -e "GEAK_GPU_IDS=0,1,2,3" \
  -e "PYTORCH_ROCM_ARCH=gfx950" \
  -w /path/to/AgentKernelArena \
  <container> \
  python3 main.py --config_name config_geak_triton_slot1.yaml

# Slot 2: GPUs 4-7 (9 kernels)
docker exec -d \
  -e "AMD_LLM_API_KEY=$AMD_LLM_API_KEY" \
  -e "GEAK_GPU_IDS=4,5,6,7" \
  -e "PYTORCH_ROCM_ARCH=gfx950" \
  -w /path/to/AgentKernelArena \
  <container> \
  python3 main.py --config_name config_geak_triton_slot2.yaml
```

### Config Files

| Config | Kernels |
|--------|---------|
| `config_geak_triton_slot1.yaml` | 9 L1/L2/L3 kernels for GPUs 0-3 |
| `config_geak_triton_slot2.yaml` | 9 L1/L2/L3 kernels for GPUs 4-7 |
| `config_geak_triton_all16.yaml` | All 16 original kernels (single slot) |

### All 18 Triton Kernels

| # | Kernel | Level | Configs | @triton.jit |
|---|--------|-------|---------|-------------|
| 1 | `llama_ff_triton` | L1 | 3 | direct |
| 2 | `fused_append_shared_experts` | L1 | 18 | direct |
| 3 | `moe_routing_sigmoid_top1` | L1 | 34 | direct |
| 4 | `mla_decode` | L1 | 320 | wrapper (aiter) |
| 5 | `ff_backward` | L1 | 4 | direct |
| 6 | `refk_identity` | L1 | self-contained | direct |
| 7 | `refk_fp8_blockwise_mm` | L1 | self-contained | direct |
| 8 | `fast_rms_layernorm` | L2 | 1 | direct |
| 9 | `topk` | L2 | 80 | direct |
| 10 | `lean_atten_paged` | L2 | 7 | direct |
| 11 | `rope` | L2 | 6480 | direct |
| 12 | `gemm_a16w16_atomic` | L3 | 13 | direct |
| 13 | `gemm` | L3 | 13 | wrapper (aiter) |
| 14 | `fused_qkv_rope` | L3 | 1200 | direct |
| 15 | `fused_mxfp4_quant_moe_sort` | L3 | 24 | wrapper (aiter) |
| 16 | `fused_moe_mxfp4` | L3 | 15 | direct |
| 17 | `fused_qk_rope_cache_mla` | L3 | 128 | direct |
| 18 | `fused_rms_fp8` | L3 | 25 | direct |

Wrapper kernels (marked "wrapper") import Triton kernels from aiter submodules.
GEAK detects these via import-following (PR #107) and routes to heterogeneous mode.

### Agent Config

Edit `agents/geak_v3_triton/agent_config.yaml`:

- `geak_env.GEAK_MAX_ROUNDS` — optimization rounds (default: 5)
- `geak_env.GEAK_MODEL` — LLM model (default: claude-opus-4.6)
- `geak_env.GEAK_MODEL_ENSEMBLE` — model ensemble for parallel agents
- `geak_env.GEAK_BENCHMARK_ITERATIONS` — benchmark iterations per shape

### Monitoring

```bash
# Progress
tail -f logs/*.log

# Per-kernel results
for f in workspace_*/run_*/*/task_result.yaml; do
  [ -f "$f" ] && echo "$(basename $(dirname $f)): $(grep speedup_ratio $f)"
done

# GEAK internal results
for d in workspace_*/run_*/*_logs/final_report.json; do
  [ -f "$d" ] && python3 -c "
import json; d=json.load(open('$d'))
fb=(d.get('round_evaluation') or {}).get('full_benchmark') or {}
vs=fb.get('verified_speedup','N/A')
bm=d.get('round_evaluation',{}).get('benchmark_speedup','N/A')
print(f'  verified={vs}x  benchmark={bm}x')
"
done
```
