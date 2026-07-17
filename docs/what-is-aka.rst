.. meta::
   :description: Learn how AgentKernelArena supports controlled A/B experiments and RL-ready reward signals for GPU kernel agents.
   :keywords: AgentKernelArena, A/B testing, agent reinforcement learning, GPU kernel optimization, AI agents, ROCm, HIP, Triton, AMD

*************************
What is AgentKernelArena?
*************************

AgentKernelArena is a controlled experimentation platform for developing AI
agents on real GPU kernel optimization tasks. It enables reproducible A/B tests
across models, prompts, tools, and agent policies, while providing objective
compilation, correctness, and performance signals that can serve as rewards for
agent reinforcement learning.

The platform supplies isolated environments and reward signals. It does not
currently include an RL trainer, replay buffer, or policy-update loop.

The AgentKernelArena source code is hosted in the
`AMD-AGI/AgentKernelArena <https://github.com/AMD-AGI/AgentKernelArena>`_ GitHub
repository.

What AgentKernelArena does
==========================

AgentKernelArena runs one agent configuration per experiment. To test a change,
run the same task set as a baseline and treatment while holding the hardware,
workspace setup, and evaluation rules constant. Each task executes in an
isolated, timestamped workspace and follows the same outcome pipeline:

* **Compile**: Build the agent's kernel and confirm it compiles cleanly.
* **Validate**: Run the task's correctness check against a reference.
* **Profile**: Measure GPU execution time and compute speedup over a baseline.
* **Reward**: Write structured signals and combine them into a configurable score.

Key features
============

AgentKernelArena includes the following key features.

* **Controlled A/B testing**: Compare a baseline with a treatment that changes a
  model, prompt, MCP server, skill, tool, memory strategy, or agent policy.
* **RL-ready feedback**: Export compilation, correctness, timing, speedup, and
  score fields for use by external reinforcement-learning systems.
* **Multiple agent integrations**: Cursor Agent, Claude Code, Codex, GEAK,
  mini-swe-agent-based flows, and custom agents.
* **Docker-first runtime**: Experiments execute inside pinned ROCm/SGLang
  Docker images selected from the target GPU architecture.
* **Task categories**: HIP (``hip2hip``), CUDA-to-HIP (``cuda2hip``), Triton
  (``triton2triton``, ``instruction2triton``), Torch-to-HIP (``torch2hip``),
  Torch/Triton-to-FlyDSL (``torch2flydsl``, ``triton2flydsl``), and FlyDSL
  (``flydsl2flydsl``), plus repository-level tasks.
* **Objective metrics**: Automated compilation, correctness, and real GPU
  performance speedups.
* **Performance timing provenance**: Timing method metadata is recorded for
  baseline and optimized runs so mixed-method comparisons are visible.
* **Workspace isolation**: Each task runs in its own timestamped workspace for
  reproducibility.
* **Multi-GPU parallel runs**: On multi-GPU servers, start one isolated Docker
  worker per GPU and keep idle GPUs busy with a shared task queue.
* **Task validator**: An agent that runs 10 automated checks on task quality
  before tasks are used in shared experiments.
* **Held-out evaluation**: Re-evaluate optimized kernels on unseen shapes and
  quantify generalization regressions. See :doc:`how-to/held-out-evaluation`.
* **Visualization dashboard**: A static dashboard for comparing run reports.

Use cases
=========

AgentKernelArena supports the following use cases.

* A/B test whether a model, prompt, MCP server, skill, tool, or policy improves
  outcomes under controlled conditions.
* Generate objective rewards for an external agent-RL or policy-search loop.
* Compare agent configurations while keeping tasks and evaluation rules fixed.
* Run large task suites faster by distributing tasks across all GPUs on a
  multi-GPU server.
* Curate and validate a high-quality, self-contained GPU kernel task suite.
