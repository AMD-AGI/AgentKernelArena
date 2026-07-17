## GEAK-V3

Integrates **GEAK v3.2.2** into AgentKernelArena as the optimizing agent for
**HIP** kernels. It drives the unified `geak` CLI (built on mini-SWE-agent) with
`-c geak.yaml`, pointing GEAK at the task's compile/correctness/performance
commands.

**AgentKernelArena is the single source of truth for scoring.** After GEAK
finishes, AKA re-evaluates the optimized kernel itself (compile → correctness →
performance) and computes `speedup_ratio`; GEAK's own numbers are not used for
the final result.

GEAK parallelizes across GPUs using **git worktrees inside the same container**
— no nested Docker — so it does not conflict with AKA's docker-first flow. Run
everything inside one ROCm container and invoke `geak` there as a plain CLI.

### Setup

```bash
# 1. Clone GEAK and check out the pinned release
git clone https://github.com/AMD-AGI/GEAK.git
cd GEAK && git checkout v3.2.2

# 2. Start the provided sglang ROCm container (ships hipcc/ROCm, torch, aiter).
#    gfx942 (MI300/MI308) -> mi30x image;  gfx950 (MI35x) -> mi35x image.
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

Pass credentials to the container at run time.

```bash
# Option A — AMD LLM gateway (default base_url https://llm-api.amd.com/Anthropic)
-e AMD_LLM_API_KEY="<gateway-key>"   # sent as Ocp-Apim-Subscription-Key
-e GEAK_USER="<your-ntid>"           # e.g. syounesi (avoids "Invalid user header")

# Option B — personal key: provide your own key (and point the model backend at
# your provider via geak.yaml `model:` / base_url if not using the AMD gateway)
-e AMD_LLM_API_KEY="<your-key>"
```

The model is set in `agents/geak_v3/geak.yaml` (`model.model_name`, default
`claude-opus-4.5`).

### Configure the GEAK runner

Edit `agents/geak_v3/agent_config.yaml`:

- **`run.configs`** — CLI options, e.g.
  `-c geak.yaml --yolo --num-parallel=2 --gpu-ids=0,1`.
  `-c geak.yaml` resolves to `agents/geak_v3/geak.yaml` (auto-expanded to an
  absolute path). `--num-parallel` / `--gpu-ids` are overridden at launch by
  `GEAK_NUM_PARALLEL` / `num_parallel` and `GEAK_GPU_IDS`.

To use a different `agent_config.yaml` without editing the repo:
`export GEAK_AGENT_CONFIG="/abs/path/to/agent_config.yaml"`.

### Run (docker-based)

Set `config.yaml` to this agent and the HIP task(s):

```yaml
agent:
  template: geak_v3
tasks:
  - hip2hip/others/knn
target_gpu_model: MI300
```

Then run `main.py` inside the container:

```bash
docker exec \
  -e AMD_LLM_API_KEY="<key>" -e GEAK_USER="<ntid>" \
  -e GEAK_GPU_IDS=0 \
  geak-aka bash -lc 'cd /workspace && python3 main.py --config_name config.yaml'
```

Tasks in the `tasks:` list run **sequentially**; increase per-task throughput
with more GPUs via `GEAK_GPU_IDS` / `--num-parallel`.

### Where to find results

- **AKA run log:** `logs/*.log` (path from `log_directory` in `config.yaml`).
- **Per-task result (authoritative):**
  `workspace_<gpu>_geak_v3/run_<timestamp>/<task>_<timestamp>/task_result.yaml`
  (plus `baseline_perf.yaml`, `optimized_perf.yaml`, and
  `build/performance_report.json`).
- **GEAK internals:**
  `workspace_<gpu>_geak_v3/run_<timestamp>/<task>_<timestamp>_logs/`
  (`final_report.json`, `geak_agent.log`, the winning `.diff`).
- **Aggregate:** `workspace_<gpu>_geak_v3/run_<timestamp>/reports/overall_summary.csv`.
