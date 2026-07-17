.. meta::
   :description: AgentKernelArena is an A/B testing and RL-ready environment for developing AI agents on real GPU kernel optimization tasks.
   :keywords: AgentKernelArena, A/B testing, agent reinforcement learning, ROCm, GPU, kernel optimization, AI agent, HIP, Triton, AMD

******************************
AgentKernelArena documentation
******************************

AgentKernelArena is a controlled experimentation platform for developing AI
agents on real GPU kernel optimization tasks. It enables reproducible A/B tests
across models, prompts, tools, and agent policies, and produces objective
compilation, correctness, and performance signals that external agent-RL
systems can use as rewards.

The AgentKernelArena source code is hosted in the
`AMD-AGI/AgentKernelArena <https://github.com/AMD-AGI/AgentKernelArena>`_ GitHub
repository.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install AgentKernelArena <install/install>`

   .. grid-item-card:: How to

      * :doc:`Run an experiment <how-to/run-evaluation>`
      * :doc:`Run tasks in parallel across multiple GPUs <how-to/parallel-run>`
      * :doc:`Configure agents and models <how-to/agents>`
      * :doc:`Add a task <how-to/add-task>`
      * :doc:`Validate tasks <how-to/task-validator>`
      * :doc:`Visualize and compare runs <how-to/visualization>`

   .. grid-item-card:: Examples

      * :doc:`Examples <examples/examples>`

   .. grid-item-card:: Reference

      * :doc:`Configuration and API reference <reference/api-reference>`
      * :doc:`Performance measurement methodology <reference/benchmark-methodology>`

To contribute to the documentation, see the
`AgentKernelArena GitHub repository <https://github.com/AMD-AGI/AgentKernelArena>`_.

AgentKernelArena is released under the Apache 2.0 license. For details, see the
:doc:`License <about/license>` page.
