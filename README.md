# AgentKernelArena

**A benchmark for evaluating AI coding agents on GPU kernel optimization.**

AgentKernelArena measures how well autonomous AI coding agents optimize GPU kernels on AMD GPUs. It provides 196 tasks across three categories, a gated evaluation pipeline (compilation, correctness, performance), and held-out generalization testing to detect overfitting to visible test shapes.

The platform currently ships with cheatsheets for MI300X and MI355X, and is designed to be extended to other GPU architectures.

## Key Features

- **196 tasks** in three categories:
  - **HIP-to-HIP** (24 tasks): optimize existing HIP kernels
  - **Triton-to-Triton** (148 tasks): optimize existing Triton kernels (118 from vLLM, 30 from ROCmBench)
  - **PyTorch-to-HIP** (24 tasks): translate PyTorch modules into HIP kernels
- **Gated evaluation pipeline**: compilation must pass before correctness is checked; correctness must pass before speedup is measured
- **Multi-shape testing**: each task tests multiple input shapes and data types
- **Held-out generalization**: evaluates agent-generated kernels on unseen shapes to detect hardcoded assumptions
- **Agent-agnostic**: supports Cursor Agent, Claude Code, Codex CLI, and custom agents through a simple registration interface
- **Domain-specific cheatsheets**: architecture guides (MI300X, MI355X) and language references (HIP, Triton) provided to agents as context
- **Automated task validation**: a built-in validator agent checks new tasks for correctness and completeness before inclusion
- **Visualization dashboard**: web-based UI for comparing run results across agents and models

## Repository Structure

```
AgentKernelArena/
├── main.py                  # Orchestrator: task discovery, agent dispatch, evaluation
├── config.yaml              # Run configuration (agent, tasks, GPU target)
├── agents/
│   ├── cursor/              # Cursor Agent (CLI-based)
│   ├── claude_code/         # Claude Code (CLI-based)
│   ├── codex/               # OpenAI Codex CLI
│   ├── task_validator/      # Automated task validation agent
│   └── __init__.py          # Agent registry
├── src/
│   ├── prompt_builder.py    # Constructs task prompts from config + source + cheatsheet
│   ├── evaluator.py         # Runs compile / correctness / performance checks
│   ├── score.py             # Scoring logic (20 compile + 100 correctness + speedup×100)
│   ├── preprocessing.py     # Workspace isolation and environment setup
│   ├── postprocessing.py    # Result collection and report generation
│   └── prompts/
│       └── cheatsheet/      # MI300X/MI355X architecture docs, HIP/Triton guides
├── tasks/
│   ├── hip2hip/             # 24 HIP optimization tasks (from GPU Mode)
│   ├── triton2triton/       # 148 Triton optimization tasks (vLLM + ROCmBench)
│   └── torch2hip/           # 24 PyTorch-to-HIP translation tasks (from GPU Mode)
├── held_out/                # Held-out generalization evaluation framework
├── visualization/           # Web dashboard for result comparison
├── Makefile                 # Environment setup helpers
└── requirements.txt
```

## Requirements

- Python 3.12+
- AMD GPU with ROCm 6.4+ (tested with 6.4, 7.0, 7.1)
- PyTorch 2.6+ built for ROCm
- Triton (included with ROCm PyTorch builds)
- `hipcc` compiler (ships with ROCm)
- Node.js 18+ (only needed for Codex CLI)
- At least one agent CLI installed (Cursor, Claude Code, or Codex)

## Setup

### Option A: Docker (recommended)

The easiest way to get a working environment is the official ROCm PyTorch Docker image, which includes ROCm, PyTorch, and Triton pre-installed. All experiments in the paper were run using this image:

```bash
# Clone the repo first, then mount it into the container
git clone <repository-url>

docker run -it --name agentkernalarena --network=host --privileged \
    --device=/dev/kfd --device=/dev/dri \
    --group-add=render --ipc=host \
    -v $(pwd)/AgentKernelArena:/workspace \
    --workdir /workspace \
    rocm/pytorch:rocm7.1.1_ubuntu24.04_py3.12_pytorch_release_2.10.0
```

Inside the container:

```bash
pip install -r requirements.txt
```

The `-v` mount ensures your work persists outside the container. To re-enter later: `docker start -ai agentkernalarena`.

### Option B: Native installation

If you already have ROCm and PyTorch installed:

```bash
git clone <repository-url>
cd AgentKernelArena

# (Optional) Create a virtual environment
make setup-venv
source .venv/bin/activate

# Or install directly
pip install -r requirements.txt
```

Verify your environment:

```bash
python -c "import torch; print(torch.version.hip)"   # should print ROCm version
hipcc --version                                        # should print hipcc version
python -c "import triton; print(triton.__version__)"   # should print triton version
```

### Install Agent CLIs

Install whichever agents you plan to use:

```bash
# Cursor Agent
curl https://cursor.com/install -fsS | bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Claude Code
curl -fsSL https://claude.ai/install.sh | bash

# Codex CLI
npm install -g @openai/codex
```

Set API keys:

```bash
export CURSOR_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

## Running an Evaluation

### 1. Configure

Edit `config.yaml` to select an agent and tasks:

```yaml
agent:
  template: claude_code       # cursor, claude_code, codex

tasks:
  - hip2hip                   # all HIP-to-HIP tasks
  - triton2triton/vllm        # all vLLM Triton tasks
  - torch2hip/gpumode/16636_SiLU  # a single task

target_gpu_model: MI300
```

Pre-built configs are provided for running each task category separately:

```bash
cp config_hip2hip.yaml config.yaml && python main.py        # 24 HIP-to-HIP tasks
cp config_triton2triton.yaml config.yaml && python main.py  # 148 Triton-to-Triton tasks
cp config_torch2hip.yaml config.yaml && python main.py      # 24 PyTorch-to-HIP tasks
```

### 2. Run

```bash
python main.py
```

Each task runs in an isolated timestamped workspace. The agent receives the source code, instructions (compile/test/benchmark commands), and an optional cheatsheet, then iterates autonomously. After the agent finishes, the framework evaluates compilation, correctness, and performance independently.

### 3. View Results

Results are written to `workspace_<gpu>_<agent>/run_<timestamp>/reports/`:
- `overall_report.txt`: summary with compilation/correctness rates and speedup statistics
- `overall_summary.csv`: per-task breakdown
- Per-task `task_result.yaml` files in each task workspace

To use the visualization dashboard:

```bash
cd visualization
python backend/scripts/build_dashboard_data.py
python backend/server.py --port 8080
```

## Held-Out Generalization Testing

The `held_out/` directory contains tooling to evaluate whether agent-optimized kernels generalize to unseen input shapes. See [`held_out/README.md`](held_out/README.md) for details.

```bash
# Generate held-out shapes (uses an LLM to create diverse test inputs)
python held_out/generate_heldout.py --tasks-dir tasks/ --output-dir held_out_tests/

# Evaluate a completed run against held-out shapes
python held_out/run_heldout_eval.py \
    --run-dir workspace_MI300_cursor/run_20260417_142419 \
    --heldout-dir held_out_tests/ \
    --tasks-dir tasks/
```

Each task is classified into a generalization quadrant: `both_pass`, `opt_regression` (optimization broke generalization), `both_fail` (shape exceeds kernel design spec), or `opt_improvement` (agent improved robustness).

## Scoring

| Gate | Points | Condition |
|------|--------|-----------|
| Compilation | 20 | Code compiles without errors |
| Correctness | 100 | Output matches reference within tolerance |
| Speedup | ratio x 100 | Wall-clock speedup over baseline kernel |

A task that compiles, passes correctness, and achieves 2.0x speedup scores 20 + 100 + 200 = 320 points. Tasks that fail compilation score 0. Tasks that compile but fail correctness score 20.

## Adding a New Agent

1. Create `agents/your_agent/launch_agent.py`:

```python
from agents import register_agent

@register_agent("your_agent")
def launch_agent(eval_config, task_config_dir, workspace, config_path=None):
    # Read the prompt, call your agent, write optimized code to workspace
    return agent_output
```

2. Add `your_agent` to the `AgentType` enum in `src/module_registration.py`.

The framework handles all evaluation after your agent writes the optimized file.

## Adding a New Task

1. Create a task directory under `tasks/<category>/<source>/<task_name>/`.
2. Add source files, a `config.yaml`, and evaluation scripts.
3. Validate with the built-in task validator:

```bash
# In config.yaml
agent:
  template: task_validator
tasks:
  - <category>/<task_name>

python main.py
```

See [`agents/task_validator/README.md`](agents/task_validator/README.md) for the 10 automated checks that all tasks must pass.

## Task Sources and Licenses

| Source | Tasks | License |
|--------|-------|---------|
| [GPU Mode](https://github.com/GPUMODE/kernelbot-data) | HIP-to-HIP, PyTorch-to-HIP | CC-BY-4.0 |
| [vLLM](https://github.com/vllm-project/vllm) | Triton-to-Triton | Apache-2.0 |
| [ROCmBench](https://github.com/AMD-AGI/GEAK-eval) | Triton-to-Triton | Cited (no specified license) |

## License

Apache License 2.0. See [LICENSE](LICENSE).
