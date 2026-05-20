## `GEAK-V3-FlyDSL`

Run **FlyDSL** kernel optimization tasks in AgentKernelArena using **GEAK v3** as the optimizing agent.

### 1) Install GEAK

```bash
cd /path/to/GEAK
pip install -e .
```

### 2) Install FlyDSL

```bash
pip install flydsl
```

Verify: `python3 -c "import flydsl; print(flydsl.__version__)"`

### 3) Install aiter (required for `pa_decode_fp8_kernel`)

```bash
pip install git+https://github.com/ROCm/aiter.git
```

> Other kernels do **not** require aiter.

### 4) Configure AMD LLM environment variables

```bash
export AMD_LLM_API_KEY="your-key-here"
```

### 5) Configure tasks in AgentKernelArena

Edit your config YAML (e.g. `config_geak_flydsl.yaml`):

```yaml
agent:
  template: geak_v3_flydsl

tasks:
  - flydsl2flydsl/softmax_kernel
  - flydsl2flydsl/rmsnorm_kernel
  - flydsl2flydsl/layernorm_kernel
  # add more kernels from tasks/flydsl2flydsl/ as needed
```

See `tasks/flydsl2flydsl/README.md` for difficulty levels and kernel descriptions.

### 6) Run

```bash
cd /path/to/AgentKernelArena
python main.py --config_name config_geak_flydsl.yaml
```

### 7) Where to find results

- **Run log**: `logs/*.log`
- **Workspace root**: `workspace_MI300_geak_v3_flydsl/`
- **Per-task results**: `workspace_.../run_<timestamp>/<task>_<timestamp>/task_result.yaml`
- **GEAK logs**: `workspace_.../<task>_<timestamp>_logs/` (see `final_report.json`, `geak_summary.json`)
- **Aggregate summary**: `workspace_.../task_results_summary.csv`

### Important: tasks run serially

In AgentKernelArena, the `tasks:` list is executed **sequentially (one task at a time)**. If you want overall throughput, add more GPUs to GEAK parallelism inside each task via `GEAK_GPU_IDS` (e.g. `export GEAK_GPU_IDS="0,1"`).

### Shared kernel dependencies

`tasks/flydsl2flydsl/kernels/` contains shared Python modules (`kernels_common.py`, `tensor_shim.py`) that individual kernel sources import. Do not delete this directory.
