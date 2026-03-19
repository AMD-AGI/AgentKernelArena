## `GEAK-V3-Triton`

Triton kernel optimization agent for AgentKernelArena. Uses the unified `geak` CLI with the full preprocessing + orchestration pipeline (multi-round, heterogeneous, verified evaluation).

> For complete setup instructions and both HIP + Triton documentation, see [`agents/geak_v3/README.md`](../geak_v3/README.md).

### Quick Start

```bash
cd AgentKernelArena/

docker exec -d \
  -e GEAK_SRC=/path/to/GEAK/src \
  -e AMD_LLM_API_KEY=$AMD_LLM_API_KEY \
  -e GEAK_GPU_IDS=0,1,2,3 \
  -w /path/to/AgentKernelArena \
  geak-agent-sapmajum \
  python3 main.py --config_name config_geak_triton.yaml
```

### How it works

The launcher calls:
```
geak --kernel-url <kernel.py> --eval <test_kernel_harness.py> \
     --gpu-ids 0,1,2,3 --max-rounds 3 --heterogeneous --yolo
```

This runs the full GEAK pipeline:
1. **Preprocess** — profile kernel, measure baseline, generate COMMANDMENT.md
2. **Orchestrate** — multi-round optimization with task generation, agent dispatch, and verified evaluation

### Config

Edit `agents/geak_v3_triton/agent_config.yaml` to change:
- `orchestrate.max_rounds` — number of optimization rounds (default: 3)
- `orchestrate.model` — LLM model (default: claude-opus-4.6)
- `configs` — heterogeneous/homogeneous mode settings

### Triton Kernels (8 total)

| Difficulty | Kernel |
|-----------|--------|
| L1 | `llama_ff_triton`, `moe_routing_sigmoid_top1` |
| L2 | `topk`, `fast_rms_layernorm`, `lean_atten_paged` |
| L3 | `fused_qkv_rope`, `fused_rms_fp8`, `gemm_a16wfp4` |

### Monitoring

```bash
tail -f logs/MI300_geak_v3_triton_*.log
grep -E "Task [0-9]+|Round [0-9]+|best.*speedup|config_mismatch" logs/MI300_geak_v3_triton_*.log
```
