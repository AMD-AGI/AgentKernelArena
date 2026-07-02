# AgentKernelArena: Competitive Arena for GPU Kernel Optimization Agents

As AI coding agents, like Claude Code and OpenAI Codex, rapidly improve, we need more than cherry-picked demos. Especially in specialized domains like GPU programming. 

AgentKernelArena is a standardized evaluation arena built by AMD to measure how well AI coding agents perform on real GPU kernel optimization tasks. 

## Demo

A live illustrative demo is available at: http://165.245.130.75/

This demo is provided only to illustrate the results and should not be treated as the final benchmark leaderboard.

## Overview & Features

AgentKernelArena provides an end-to-end, siloed benchmarking environment where LLM-powered agents (Cursor Agent, Claude Code, Codex, and custom agents) are evaluated side-by-side on the same kernel tasks using objective and reproducible metrics.

AgentKernelArena enables systematic evaluation of AI agents on GPU kernel optimization tasks by combining:
- **Multi-Agent Arena**: Cursor, Claude Code, Codex, and custom agents
- **Multi-Model Support**: OpenAI (GPT-5), Anthropic Claude (Opus and Sonnet families), and other models via OpenRouter or vLLM
- **Task Categories**: HIP (ROCm examples, rocPRIM, customer HIP), Triton (vLLM-style local harnesses and ROCmBench), and Torch2HIP conversions
- **Real Metrics**: Automated evaluation of compilation success, correctness, and real GPU performance speedups
- **Designed for Fair Comparison**: Standardized tasks, environments, prompts, and scoring for leaderboard-style evaluation
- **A/B Testing for Agent Tools**: Compare whether a new MCP server, skill, prompt, or agent-side tool actually improves outcomes by running the same task set with and without it and comparing standardized scores
- **Workspace Isolation**: Each task runs in a timestamped duplicate workspace for reproducibility
- **Comprehensive Logging**: Detailed logs with timestamps, prompts, outputs, and results for every task execution
- **Flexible Configuration**: YAML-based configuration for tasks, agents, and LLM parameters

### A/B Testing and Ablation Studies

Beyond comparing different agents and models, AgentKernelArena can also be used to evaluate whether new agent-side capabilities actually help. For example, if you introduce a new MCP server, skill, prompt strategy, or tool integration, you can run the same task set twice — once with the capability enabled and once without it — and compare compilation, correctness, performance, and overall scores under the same evaluation conditions.

This makes AgentKernelArena useful not only as a leaderboard-style benchmark, but also as a controlled A/B testing framework for measuring the real impact of agent improvements.


## **Leaderboard Coming: Stay Tuned!**

AgentKernelArena is actively under development. Upcoming releases will publish detailed evaluation results comparing agent performance across multiple task categories, using standardized correctness and performance scores. 


| Model         | Compiled | Correctness | Performance | Score |
|---------------|----------|-------------|-------------|-------|
| Cursor Agent  | xx       | xx          | xx          | xx    |
| Claude Code   | xx       | xx          | xx          | xx    |
| OpenAI Codex  | xx       | xx          | xx          | xx    |


## Architecture

### Core Components

```
AgentKernelArena/
├── main.py                      # Main orchestration entry point
├── config.yaml                  # Global configuration
├── src/
│   ├── module_registration.py  # Dynamic agent/prompt/post-processing loading
│   ├── preprocessing.py         # Workspace setup and environment checks
│   ├── prompt_builder.py        # Task prompt construction
│   ├── postprocessing.py        # Result analysis and report generation
│   ├── score.py                  # Scoring logic for evaluation metrics
│   ├── tasks.py                 # Task discovery and registration
│   └── utils/
│       └── report_generation.py # Aggregate report analysis utilities
├── agents/
│   ├── cursor/                  # Cursor agent integration
│   ├── claude_code/             # Claude Code agent integration
│   ├── codex/                   # Codex CLI agent integration
│   ├── task_validator/          # Task quality validator
│   └── __init__.py              # Agent registry
└── tasks/                       # Task definitions
    ├── rocm-examples/           # ROCm example kernels
    ├── rocprim/                 # rocPRIM kernels
    ├── customer_hip/            # Custom HIP kernels
    ├── triton/                  # Triton benchmark kernels
    └── torch2hip/               # Torch2HIP conversion tasks
```

### Execution Flow

1. **Configuration Loading**: Load `config.yaml` with agent, task, and LLM settings
2. **Agent Registration**: Dynamically load agent launcher, prompt builder, and post-processing handler based on AgentType enum
3. **Task Discovery**: Scan `tasks/` directory for task configurations matching specified categories
4. **Workspace Setup**: Create isolated workspace with timestamp for each task
5. **Prompt Building**: Construct task-specific prompts from config, source code, and instructions/cheatsheets
6. **Agent Execution**: Launch agent in workspace with constructed prompt
7. **Result Collection**: Save agent output, logs, and modified code
8. **Post-Processing**: Run compilation, correctness tests, performance profiling, and scoring
9. **Report Generation**: Generate comprehensive evaluation report with metrics

## Installation

### Prerequisites

- Docker
- The benchmark Docker image for your GPU arch (`gfx942` uses `lmsysorg/sglang:v0.5.12-rocm720-mi30x`; `gfx950` uses `lmsysorg/sglang:v0.5.12-rocm720-mi35x`; Navi/RDNA/Radeon `gfx10*`/`gfx11*`/`gfx12*` uses `rocm/vllm-dev:rocm7.2.1_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.16.0`)
- Git
- Host-installed agent CLIs for the agents you plan to evaluate

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AgentKernelArena

# Install agent CLIs (examples)
# For Claude Code:
npm install -g @anthropic-ai/claude-code

# For Codex CLI: install per the official Codex CLI instructions,
# then ensure `codex` is available in PATH.

# Verify Docker can see the GPU/runtime and reuse agent login state.
make docker-smoke
make docker-check-agents
```

Performance timing helpers are maintained in `src/tools/perf/`; committed task
sources keep stubs that are materialized into run workspaces. See
`src/tools/perf/README.md` for the helper workflow.

## Usage

### Basic Usage

1. **Configure `config.yaml`**:

```yaml
# Select agent type
agent:
  template: claude_code  # Options: cursor, claude_code, codex, task_validator
  max_iterations: 5

# Specify tasks to run
tasks:
  - rocm-examples/bitonic_sort
  - customer_hip/silu
  # - all  # Run ALL tasks

target_gpu_model: MI300
log_directory: logs
workspace_directory_prefix: workspace

```

2. **Run evaluation**:

```bash
make docker-run CONFIG=config.yaml
```


### Advanced Usage

#### Running Specific Task Categories

```yaml
tasks:
  - rocm-examples/*           # All ROCm examples
  - rocprim/*                 # All rocPRIM tasks
  - customer_hip/mmcv/*       # All MMCV HIP kernels
  - triton2triton/vllm/*      # vLLM-style Triton kernel tasks
  - triton2triton/rocmbench/* # ROCmBench Triton tasks
  - instruction2triton/rocmbench/* # Instruction-to-Triton ROCmBench tasks
  - torch2hip/*               # All Torch2HIP conversion tasks
```

## Task Configuration

Each task is defined by a `config.yaml` in its directory:

```yaml
# tasks/rocm-examples/bitonic_sort/config.yaml
source_file_path:
  - main.hip

target_kernel_functions:
  - bitonic_sort_kernel

compile_command:
  - make

correctness_command:
  - ./applications_bitonic_sort -l 15

performance_command:
  - rocprof-compute profile -n kernelgen --path rocprof_compute_profile --no-roof --join-type kernel -b SQ -b TCP -b TCC -- ./applications_bitonic_sort -l 15
  - rocprof-compute analyze --path rocprof_compute_profile -b 2
task_type: hip2hip
prompt:
  source_code: null      # Optional: override default source code inclusion
  instructions: null     # Optional: custom instructions
  cheatsheet: null       # Optional: provide cheatsheet/reference

# Optional: skip tasks that do not apply to this run's target GPU arch.
platform_support:
  required_arch: gfx942   # Compared with target_gpu_model's resolved gfx arch
  status: active          # active | skip
  skip_reason: null       # Recommended when status is skip
```

If `platform_support` is omitted, the task runs as before. If
`platform_support.status: skip`, or if `required_arch` does not match the
current `target_gpu_model`, the framework skips the task before creating a
workspace or launching an agent. Historical fields such as
`runnable_on_gfx942` are documentation only and are not used for scheduling.


## Scoring System

AgentKernelArena uses a cumulative scoring system:

| Metric | Points | Description |
|--------|--------|-------------|
| **Compilation** | 20 | Code compiles successfully without errors |
| **Correctness** | 100 | Code produces correct output (passes tests) |
| **Speedup** | ratio × 100 | Performance improvement over baseline |

**Example**: A submission that compiles (20), passes correctness (100), and achieves 1.5× speedup (150) would score 270 points.

Note: This is not the only way to score. Users could always define their own ways.


## Development

### Adding a New Agent

1. **Create agent directory**: `agents/your_agent/`

2. **Implement launch function**:

```python
# agents/your_agent/launch_agent.py
from agents import register_agent

@register_agent("your_agent")
def launch_agent(prompt: str, log_directory: str, workspace: str) -> str:
    """
    Launch your agent.

    Returns:
        str: Agent output
    """
    # Your agent implementation
    return result
```

3. **Register in module_registration.py**:

```python
# Add to AgentType enum
class AgentType(Enum):
    YOUR_AGENT = "your_agent"

# Add import in load_agent_launcher
if agent_type == AgentType.YOUR_AGENT:
    from agents.your_agent import launch_agent
```

4. **Add prompt builder support** (if needed):

```python
# In load_prompt_builder
if agent_type in [..., AgentType.YOUR_AGENT]:
    return prompt_builder
```

5. **Add post-processing support** (if needed):

```python
# In load_post_processing_handler
if agent_type in [..., AgentType.YOUR_AGENT]:
    return general_post_processing
```

### Adding a New Task

1. **Create task directory**: `tasks/<task_type>/<task_name>/`

2. **Add source files and scripts** following this structure:

```
tasks/<task_type>/<task_name>/
├── config.yaml                  # Task configuration (required)
├── scripts/
│   └── task_runner.py           # Compile/correctness/performance runner (recommended)
└── src/
    └── <kernel files>           # .cu, .hip, .py, etc.
```

3. **Create `config.yaml`** with all required fields as **lists** (not scalar strings):

```yaml
source_file_path:
  - src/my_kernel.hip

target_kernel_functions:
  - my_kernel_function

compile_command:
  - python3 scripts/task_runner.py --mode compile

correctness_command:
  - python3 scripts/task_runner.py --mode correctness

performance_command:
  - python3 scripts/task_runner.py --mode performance

task_type: hip2hip   # one of: hip2hip, cuda2hip, triton2triton, torch2hip, instruction2triton, repository, flydsl2flydsl

prompt:
  source_code: null
  instructions: null
  cheatsheet: null
```

4. **Add baseline performance** (optional): Create `baseline.txt` with expected performance metrics

5. **Run the Task Validator Agent** (required):

All new tasks **must** pass the task validator agent before being merged. The validator runs 10 automated checks covering config schema, source file existence, kernel symbol resolution, compilation, correctness, performance, self-containedness, GPU hang detection, correctness implementation review, and result template compatibility.

```bash
# Configure the validator to target your new task
# In config.yaml at repo root:
agent:
  template: task_validator
tasks:
  - <task_type>/<task_name>

# Run validation
make docker-run CONFIG=config.yaml
```

Review the generated `validation_report.yaml` in the workspace directory. The task must achieve **PASS** overall status (all checks pass). A **WARN** status (no failures but warnings) is acceptable with justification. A **FAIL** status means the task must be fixed before merging.

See [agents/task_validator/README.md](agents/task_validator/README.md) for the full list of validation checks and requirements.


## Next Steps

- Enhance A/B Testing with Better Interactivity and User Experience
- Benchmarking State-of-the-Art Agents for Technical Reporting
- Standardize Holdout Tests with Comprehensive Shape Coverage
- Add Holdout Test Evaluation via Independent Agent
- New Feature: Support Multi Agents in Multi GPUs Server
- New Feature: Resume the Evaluation From Previous Experiment
- Agents Can Hang During Task Execution, Blocking Test Completion
- Expand Pytorch2HIP Task Set to 100+ Tasks
- Expand CUDA2HIP Task Set to 100+ Tasks
- Expand Triton2Triton Task Set to 100+ Tasks
- Expand HIP2HIP Task Set to 100+ Tasks
- Restructure Task Directory by Take Type and Difficulty Level
