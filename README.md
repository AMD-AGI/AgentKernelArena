# AgentKernelArena: Competitive Arena for GPU Kernel Optimization Agents

As AI coding agents, like Claude Code and OpenAI Codex, rapidly improve, we need more than cherry-picked demos. Especially in specialized domains like GPU programming. 

AgentKernelArena is a standardized evaluation arena built by AMD to measure how well AI coding agents perform on real GPU kernel optimization tasks. 

## Overview & Features

AgentKernelArena provides an end-to-end, siloed benchmarking environment where LLM-powered agents (Cursor Agent, Claude Code, Codex, SWE-agent, GEAK, and custom agents) are evaluated side-by-side on the same kernel tasks using objective and reproducible metrics.

AgentKernelArena enables systematic evaluation of AI agents on GPU kernel optimization tasks by combining:
- **Multi-Agent Arena**: Cursor, Claude Code, SWE-agent, OpenEvolve (GEAK), single LLM calls (Codex/others), and custom agents
- **Multi-Model Support**: OpenAI (GPT-5), Anthropic Claude (Opus 4.5, Sonnet 4.5), and other models via OpenRouter or vLLM
- **Task Categories**: HIP (ROCm examples, rocPRIM, customer HIP), Triton (TritonBench, ROCmBench), and Torch2HIP conversions
- **Real Metrics**: Automated evaluation of compilation success, correctness, and real GPU performance speedups
- **Designed for Fair Comparison**: Standardized tasks, environments, prompts, and scoring for leaderboard-style evaluation
- **Workspace Isolation**: Each task runs in a timestamped duplicate workspace for reproducibility
- **Comprehensive Logging**: Detailed logs with timestamps, prompts, outputs, and results for every task execution
- **Flexible Configuration**: YAML-based configuration for tasks, agents, and LLM parameters


## **Leaderboard Coming: Stay Tuned!**

AgentKernelArena is actively under development. Upcoming releases will publish detailed evaluation results comparing agent performance across multiple task categories, using standardized correctness and performance scores. 


| Model         | Compiled | Correctness | Performance | Score |
|---------------|----------|-------------|-------------|-------|
| Cursor Agent  | xx       | xx          | xx          | xx    |
| Claude Code   | xx       | xx          | xx          | xx    |
| OpenAI Codex  | xx       | xx          | xx          | xx    |
| SWE-agent     | xx       | xx          | xx          | xx    |
| GEAK          | xx       | xx          | xx          | xx    |


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
│   ├── SWE_agent/               # SWE-agent integration
│   ├── openevolve/              # OpenEvolve (GEAK) integration
│   ├── geak_optimagentv2/        # GEAK OptimAgent v2 integration
│   ├── geak_hip/                 # GEAK HIP integration
│   ├── geak_ourllm_kernel2kernel/ # GEAK OurLLM kernel-to-kernel integration
│   ├── single_llm_call/         # Single LLM call implementation
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

- Python 3.12+
- ROCm toolkit (for HIP kernels): `hipcc`, `rocprof-compute`
- Triton (for Triton kernels)
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd AgentKernelArena

# Install dependencies
pip install -r requirements.txt

# Set up API keys (choose one or more)
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export OPENROUTER_API_KEY="your_openrouter_key"

# Install agent CLIs (using claude_code as an example)
# For Claude Code:
npm install -g @anthropic-ai/claude-code

```

## Usage

### Basic Usage

1. **Configure `config.yaml`**:

```yaml
# Select agent type
agent:
  template: claude_code  # Options: cursor, claude_code, swe_agent, single_llm_call, openevolve, geak_optimagentv2, geak_hip, geak_ourllm_kernel2kernel
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
python main.py
```


### Advanced Usage

#### Running Specific Task Categories

```yaml
tasks:
  - rocm-examples/*           # All ROCm examples
  - rocprim/*                 # All rocPRIM tasks
  - customer_hip/mmcv/*       # All MMCV HIP kernels
  - triton/tritonbench/*      # All Triton benchmarks
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
```


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

1. **Create task directory**: `tasks/category/task_name/`

2. **Add source files**: `main.hip`, `Makefile`, etc.

3. **Create config.yaml**:

```yaml
source_file_path:
  - main.hip

target_kernel_functions:
  - your_kernel_function

compile_command:
  - make

correctness_command:
  - ./your_executable --test

performance_command:
  - rocprof-compute profile ... -- ./your_executable
  - rocprof-compute analyze ...
task_type: triton2triton

prompt:
  source_code: null
  instructions: null
  cheatsheet: null
```

4. **Add baseline performance** (optional): Create `baseline.txt` with expected performance metrics


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
