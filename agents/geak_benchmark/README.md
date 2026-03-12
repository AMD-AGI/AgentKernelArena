# GEAK Benchmark Agent (AgentKernelArena)

This directory contains the **`geak_benchmark`** agent integration for **AgentKernelArena**.

It supports:
- **Serial execution** via `main.py` (one task at a time)
- **Task-level parallel execution** via `run_parallel.py` (multiple tasks concurrently across GPU groups)

---

## What it does

For each task, the pipeline is:
1. Create a task workspace (copy task files and clone repo if needed)
2. Compile baseline kernel and measure baseline performance (when applicable)
3. Launch the agent (typically `mini`) to propose/iterate kernel changes
4. Run centralized evaluation
5. Write `task_result.yaml`
6. (Optional) Run post-processing to generate a report

The parallel runner uses the **same pipeline** as `main.py`, but schedules different tasks onto different GPU groups.

---

## Key files

- `launch_agent.py`: the agent launcher used by `main.py` (single-task entry)
- `run_parallel.py`: task-level parallel runner (multi-task entry)
- `agent_config.yaml`: agent settings (command, num_parallel, options)
- `geak.yaml`: config file consumed by the `mini` agent
- `geak_pre_process.py`: prompt building and workspace prep helpers

---

## Configuration

### 1) Main Arena config (`config.yaml`)

In the AgentKernelArena project root, `config.yaml` selects:
- which **agent template** to use (set to `geak_benchmark`)
- which **tasks** to run
- GPU model, logging, workspace naming

Example:

```yaml
agent:
  template: geak_benchmark

tasks:
  - repository/rocprim/block_radix_rank

target_gpu_model: MI308
log_directory: logs
workspace_directory_prefix: workspace
```

### 2) Agent config (`agents/geak_benchmark/agent_config.yaml`)

This controls how the geak_benchmark agent invokes the underlying command (usually `mini`).

Important fields:
- `run.cmd`: `mini` (or `geak`)
- `run.num_parallel`: **GPUs per task** (used by both `launch_agent.py` and `run_parallel.py`)
- `run.configs`: extra CLI options passed to the agent command

Example:

```yaml
run:
  cmd: mini
  num_parallel: 2
  configs: "-c geak.yaml --yolo"
```

Optional overrides (for the parallel runner only):

```yaml
parallel_tasks:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]  # explicit GPU pool
  max_workers: 4                      # max concurrent tasks (defaults to number of GPU groups)
```

---

## Serial run (recommended for debugging)

From the AgentKernelArena root:

```bash
python3 main.py --config_name config.yaml
```

This runs tasks **one by one** using `launch_agent.py`.

---

## Task-level parallel run (recommended for throughput)

From the AgentKernelArena root:

```bash
python3 -m agents.geak_benchmark.run_parallel --config_name config.yaml
```

### GPU allocation logic

The parallel runner builds a GPU pool, then groups GPUs into chunks of size `run.num_parallel`.

Let:
- `G = number of GPUs in the pool`
- `P = run.num_parallel` (GPUs per task)

Then:
- `max_concurrent_tasks = floor(G / P)`
- each task is assigned one GPU group, e.g. with `G=8`, `P=2`:
  - `[[0,1], [2,3], [4,5], [6,7]]` → **4 tasks in parallel**

### How GPU pool is determined

Priority:
1. CLI override: `--gpu-ids 0,1,2,3,...`
2. `agent_config.yaml`: `parallel_tasks.gpu_ids`
3. Environment variables: `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`
4. `rocm-smi` probe
5. Fallback: `[0,1,2,3,4,5,6,7]`

### Dry run

```bash
python3 -m agents.geak_benchmark.run_parallel --dry-run
```

### Explicit GPU pool examples

Use GPUs 0..7:

```bash
python3 -m agents.geak_benchmark.run_parallel --gpu-ids 0,1,2,3,4,5,6,7
```

Restrict to a subset (still grouped by `run.num_parallel`):

```bash
python3 -m agents.geak_benchmark.run_parallel --gpu-ids 2,3,6,7
```

---

## Outputs & logs

Parallel runner writes:
- `logs/parallel_<timestamp>/parallel_runner.log`
- `logs/parallel_<timestamp>/<task_name>.log`
- `logs/parallel_<timestamp>/summary.json`

Each task still gets its own workspace under:
- `workspace_<GPU_MODEL>_geak_benchmark/<task>_<timestamp>/`

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'yaml'`

Install PyYAML in the Python environment you use to run AgentKernelArena:

```bash
pip install pyyaml
```

### Python version issues

AgentKernelArena uses modern Python syntax in some modules; use **Python 3.9+** if possible.

### `rocm-smi` not found

Either ensure ROCm tools are available in `PATH`, or pass GPU IDs explicitly:

```bash
python3 -m agents.geak_benchmark.run_parallel --gpu-ids 0,1,2,3,4,5,6,7
```

