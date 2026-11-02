.. meta::
   :description: Learn how AgentKernelArena supports controlled A/B experiments and RL-ready reward signals for GPU kernel agents.
   :keywords: AgentKernelArena, A/B testing, agent reinforcement learning, GPU kernel optimization, AI agents, ROCm, HIP, Triton, AMD

*************************
What is AgentKernelArena?
*************************

AgentKernelArena provides real GPU optimization tasks and independently
evaluated reward signals for developing and benchmarking kernel agents.
External reinforcement-learning systems can consume the structured task
outcomes and final score. Matched runs compare agents, policy checkpoints,
models, prompts or tools under the same task and evaluation conditions.

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
* **Benchmark**: Measure GPU execution time or fixed-workload model throughput
  against an independently preserved performance baseline.
* **Reward**: Write structured outcomes and a bounded performance score. The
  correctness reference and performance baseline have separate roles.

Key features
============

AgentKernelArena includes the following key features.

* **Controlled A/B testing**: Compare a baseline with a treatment that changes a
  model, prompt, MCP server, skill, tool, memory strategy, or agent policy.
* **RL-ready feedback**: Export compilation, correctness, timing, speedup, and
  score fields for use by external reinforcement-learning systems.
* **Agent integrations**: CLI integrations include Cursor, Claude Code, Codex and
  DeepSeek Harness. GEAK and KernelForge have separate setup requirements. See
  :doc:`the agent guide <how-to/agents>` for selectable templates and support
  boundaries.
* **Docker-first runtime**: Experiments execute inside pinned ROCm/SGLang
  Docker images selected from the target GPU architecture.
* **Operator tasks**: Optimize existing implementations, implement PyTorch
  references in a kernel language, or translate compute to FlyDSL. The bundled
  suites also include production operators under ``Aiter-task`` and
  ``Aiter-task-mi300``, and declared in-image sources under ``image_kernel``.
  Task selectors are actual paths relative to ``tasks/``; the
  :doc:`task contract <how-to/add-task>` defines behavior.
* **End-to-end tasks**: Rewrite kernel implementations and fuse compute across
  a model's declared source trees. Weights, precision, TP/EP/PP, serving flags
  and workload settings remain fixed. The Magpie catalog provides one task
  directory per text, multimodal or AgentX recipe. Runtime binding and GPU
  qualification are required separately; see :doc:`how-to/e2e-kernel-tasks`.
* **Objective metrics**: Automated compilation, correctness, and real GPU
  performance speedups.
* **Performance timing provenance**: Timing method metadata is recorded for
  baseline and optimized runs so mixed-method comparisons are visible.
* **Workspace isolation**: Each task runs in its own timestamped workspace for
  reproducibility.
* **Multi-GPU parallel runs**: Start Docker workers with one GPU or a declared
  GPU group, then distribute tasks through a shared queue.
* **Task validator**: An agent plus deterministic report gate that runs 12 checks
  on task quality, benchmark integrity, and harness integrity
  before tasks are used in shared experiments.
* **Task repair**: The :doc:`quality loop <how-to/quality-loop>` supports bounded
  repairs, fresh validation and independent review. Its repair-only mode
  preserves the scored contract and does not report repairs as speedups.
* **Held-out evaluation**: Re-evaluate supported kernels on unseen shapes and
  quantify generalization regressions. This is a separate post-run workflow;
  see :doc:`how-to/held-out-evaluation`.

The default score lies in ``[0, 1]`` and requires more than ``1.05×`` speedup
from a valid comparison. Compilation and correctness are prerequisites, with
no separate point bonuses. An agent completing its invocation, a candidate
passing correctness and a positive reward are distinct outcomes. See
:doc:`the configuration and scoring reference <reference/api-reference>`.

Task workspaces preserve reproducibility and separate concurrent runs. The
privileged worker containers are not security sandboxes.

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
