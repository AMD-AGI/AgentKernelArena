## GEAK-V3-Triton

Triton kernel optimization agent for AgentKernelArena. It wraps the unified
`geak` CLI (**GEAK v3.2.2**, built on mini-SWE-agent), which auto-detects the
Triton harness and runs heterogeneous, multi-round optimization with working
memory.

**AgentKernelArena is the single source of truth for scoring.** After GEAK
finishes, AKA re-evaluates the optimized kernel itself (compile → correctness →
performance) and computes `speedup_ratio`. GEAK's own reported numbers are not
used for the final result.

GEAK parallelizes across GPUs using **git worktrees inside the same container**
— it does **not** launch nested Docker, so it does not conflict with AKA's own
docker-first flow. Run everything inside one ROCm container and invoke `geak`
there as a plain CLI.

### Setup

```bash
# 1. Clone GEAK and check out the pinned release
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK && git checkout v3.2.2

# 2. Start the provided sglang ROCm container (ships torch, triton, aiter).
#    gfx942 (MI300/MI308) -> mi30x image;  gfx950 (MI35x) -> mi35x image.
#    Mount both repos and expose the GPUs.
docker run -d --name geak-aka \
  --ipc=host --network=host --privileged \
  --group-add render --group-add video \
  --device=/dev/kfd --device=/dev/dri \
  -e PYTORCH_ROCM_ARCH=gfx942 \
  -v /path/to/AgentKernelArena:/workspace \
  -v /path/to/GEAK:/GEAK -w /workspace \
  lmsysorg/sglang:v0.5.12-rocm720-mi30x sleep infinity

# 3. Install GEAK (provides the `geak` CLI). The pinned image supplies the
# AgentKernelArena runtime dependencies.
docker exec geak-aka pip install -e /GEAK
```

### LLM authentication (AMD gateway or personal)

GEAK's LLM backend works with either the AMD gateway or a personal key. Pass the
credentials as environment variables to the container at run time.

```bash
# Option A — AMD LLM gateway (default base_url https://llm-api.amd.com/Anthropic)
-e AMD_LLM_API_KEY="<gateway-key>"   # sent as Ocp-Apim-Subscription-Key
-e GEAK_USER="<your-ntid>"           # e.g. syounesi (avoids "Invalid user header")

# Option B — personal key: provide your own key (and point the model backend at
# your provider via geak.yaml `model:` / base_url if not using the AMD gateway)
-e AMD_LLM_API_KEY="<your-key>"
```

### Run (docker-based)

Point `config.yaml` (or a custom `--config_name`) at this agent and the Triton
task(s):

```yaml
agent:
  template: geak_v3_triton
tasks:
  - triton2triton/geak_eval/L3/gemm
target_gpu_model: MI300
```

Then run `main.py` inside the container:

```bash
docker exec \
  -e AMD_LLM_API_KEY="<key>" -e GEAK_USER="<ntid>" \
  -e GEAK_GPU_IDS=0 -e GEAK_MODEL=claude-opus-4.6 -e GEAK_RUN_MODE=quick \
  geak-aka bash -lc 'cd /workspace && python3 main.py --config_name config.yaml'
```

To shard across GPUs, run twice with disjoint `GEAK_GPU_IDS` and different
`workspace_directory_prefix` values.

### Defaults (`agents/geak_v3_triton/agent_config.yaml` → `geak_env`)

| Setting | Default | Description |
|---------|---------|-------------|
| `GEAK_MAX_ROUNDS` | 5 | Optimization rounds per kernel |
| `GEAK_MODEL` | claude-opus-4.6 | Gateway model name |
| `GEAK_BENCHMARK_ITERATIONS` | 30 | Benchmark iterations per shape |
| `num_parallel` | 4 | Parallel GEAK sub-agents (git worktrees) |
| `run_mode` | full | `quick` ≈ 1h/kernel, `full` ≈ 2h/kernel (forwarded as `--mode`) |

Override per run via env: `GEAK_GPU_IDS`, `GEAK_NUM_PARALLEL`, `GEAK_MODEL`,
`GEAK_RUN_MODE`, `GEAK_MAX_ROUNDS`.

### Pipeline

1. **AKA baseline** — runs the harness on the original `kernel.py`
   (`--full-benchmark`, which prints `GEAK_RESULT_LATENCY_MS`).
2. **GEAK** — the launcher calls:
   ```
   geak --kernel-url kernel.py \
        --test-command 'python3 test_kernel_harness.py' \
        --gpu-ids <ids> --num-parallel <N> [--mode quick|full] \
        --yolo --exit-immediately -t task_prompt.md -o <logs_dir>
   ```
   GEAK auto-promotes the test command to harness mode (it detects the
   `--correctness`/`--benchmark` argparse modes) and scores on
   `GEAK_RESULT_LATENCY_MS`.
3. **Patch apply** — GEAK auto-applies the winning patch to the workspace and
   git-commits it (the launcher's `_apply_best_patch` is only a fallback).
4. **AKA re-evaluation** — AKA recompiles, re-checks correctness, re-measures
   performance, and writes the authoritative `task_result.yaml`
   (`pass_compilation`, `pass_correctness`, `speedup_ratio`).

### Monitoring

```bash
docker exec geak-aka tail -f /workspace/logs/*.log

# Per-kernel AKA results (authoritative)
for f in workspace_*/run_*/*/task_result.yaml; do
  [ -f "$f" ] && echo "$(basename $(dirname "$f")): $(grep speedup_ratio "$f")"
done
```
