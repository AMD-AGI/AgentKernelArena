## `GEAK-V3`

This agent template integrates **GEAK v3** into AgentKernelArena so you can run AgentKernelArena tasks using GEAK-v3 as the optimizing agent.

### 1) Install GEAK

GEAK provides the `geak` CLIs. Install it in your Python environment:

```bash
cd /path/to/GEAK
pip install -e .
```

### 2) Configure AMD LLM environment variables

```bash
export AMD_LLM_API_KEY="your-key-here"
```

### 3) Configure the GEAK runner in geak_v3

Edit `agents/geak_v3/agent_config.yaml`.

Key fields:
- **`run.cmd`**: which executable to run `geak`
- **`run.configs`**: CLI options passed to that executable

Example:

```yaml
run:
  cmd: geak
  configs: "-c geak.yaml --yolo --num-parallel=2 --gpu-ids=0,1"
```

Notes:
- `-c geak.yaml` points to `agents/geak_v3/geak.yaml` (the launcher automatically resolves it to an absolute path).
- `--num-parallel` / `--gpu-ids` controls **parallel sub-agents inside a single task** (multi-GPU). This does *not* change how AgentKernelArena schedules tasks (see the “Tasks run serially” note below).
- If you want to use a different `agent_config.yaml` without editing the repo, set:

```bash
export GEAK_AGENT_CONFIG="/abs/path/to/agent_config.yaml"
```

### 4) Configure tasks in AgentKernelArena

Edit `AgentKernelArena/config.yaml`:

1) Select this agent template:

```yaml
agent:
  template: geak_v3
```

2) Select tasks to run (task names are relative to `tasks/`):

Here are tasks of hip kernels: 
```yaml
tasks:
  - hip2hip/others/
  - repository/rocprim/block_radix_rank
  - repository/rocprim/device_binary_search
  - repository/rocprim/device_search_n
  - repository/rocprim/device_merge_sort
```

### 5) Run

From the `AgentKernelArena/` directory:

```bash
python3 main.py
```

### 6) Where to find results

Quick checklist:

- **AgentKernelArena Run log**: `logs/*.log` (path controlled by `log_directory` in `AgentKernelArena/config.yaml`)
- **Workspace root**: `workspace_<GPU>_geak_v3/` (you can rename it by changing `workspace_directory_prefix` in `AgentKernelArena/config.yaml`)
- **Per-task results**: `workspace_.../<task>_<timestamp>/task_result.yaml` (also `baseline_perf.yaml`, `optimized_perf.yaml`, `build/performance_report.json`)
- **GEAK logs**: `workspace_.../<task>_<timestamp>_logs/` (see `final_report.json` or legacy `best_results.json`, `parallel_*/`)
- **Aggregate summary**: `workspace_.../task_results_summary.csv` (and sometimes `task_results_report.txt`)

### Important: tasks run serially

In AgentKernelArena, the `tasks:` list is executed **sequentially (one task at a time)**. If you want overall throughput, add more GPUs to **GEAK parallelism inside each task** via `--num-parallel` and `--gpu-ids`.
