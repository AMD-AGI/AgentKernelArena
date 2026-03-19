## `GEAK-V3`

This agent template integrates **GEAK v3** into AgentKernelArena for optimizing both **HIP** and **Triton** kernels on AMD MI300X GPUs.

Both agents (`geak_v3` for HIP and `geak_v3_triton` for Triton) use the same unified `geak` CLI with the `--eval` flag:
- **Triton**: `geak --kernel-url kernel.py --eval test_kernel_harness.py` (harness file path)
- **HIP**: `geak --kernel-url kernel.hip --eval “compile && correctness && perf”` (shell commands)

The `--eval` flag auto-detects whether the input is a file path (→ harness mode) or a shell command (→ command mode).

### 1) Prerequisites

- **GEAK installed** (`geak` CLI in PATH):
  ```bash
  cd /path/to/GEAK && pip install -e .
  ```
- **AMD LLM API key**:
  ```bash
  export AMD_LLM_API_KEY=”your-key-here”
  ```

### 2) Running HIP kernels (16 kernels)

```bash
cd AgentKernelArena/

# Inside GEAK Docker container:
docker exec -d \
  -e GEAK_SRC=/path/to/GEAK/src \
  -e AMD_LLM_API_KEY=$AMD_LLM_API_KEY \
  -e GEAK_GPU_IDS=4,5 \
  -w /path/to/AgentKernelArena \
  geak-agent-sapmajum \
  python3 main.py --config_name config_geak_hip.yaml

# Or directly (if GEAK installed locally):
python3 main.py --config_name config_geak_hip.yaml
```

### 3) Running Triton kernels (8 kernels)

```bash
cd AgentKernelArena/

# Inside GEAK Docker container:
docker exec -d \
  -e GEAK_SRC=/path/to/GEAK/src \
  -e AMD_LLM_API_KEY=$AMD_LLM_API_KEY \
  -e GEAK_GPU_IDS=0,1,2,3 \
  -w /path/to/AgentKernelArena \
  geak-agent-sapmajum \
  python3 main.py --config_name config_geak_triton.yaml
```

### 4) Monitoring

```bash
# Check progress
grep -E “Task [0-9]+|Score:|Speedup:|Average speedup” logs/MI300_geak_v3*.log

# GPU usage
rocm-smi --showpidgpus

# Container processes
docker exec geak-agent-sapmajum ps aux | grep -E 'main\.py|orchestrat|geak'
```

### 5) Where to find results

- **Run log**: `logs/MI300_geak_v3*.log`
- **Per-task results**: `workspace_*/<task>_<timestamp>/task_result.yaml`
- **Orchestrator reports** (Triton): `*_logs/preprocess/final_report.json`
- **Summary CSV**: `workspace_*/run_*/reports/overall_summary.csv`

### 6) Config files

| Config | Agent | Tasks |
|--------|-------|-------|
| `config_geak_hip.yaml` | `geak_v3` | 16 HIP kernels (12 L1/L2 + 4 L3 rocPRIM) |
| `config_geak_triton.yaml` | `geak_v3_triton` | 8 Triton kernels (L1/L2/L3) |

### Important notes

- Tasks run **sequentially** within each `main.py`. Use `GEAK_GPU_IDS` for parallelism within each task.
- Ensure HIP and Triton runs use **non-overlapping GPUs** (e.g., HIP on 4,5 and Triton on 0-3).
- If the container exits with code 137 (OOM), restart with `docker start geak-agent-sapmajum`.
