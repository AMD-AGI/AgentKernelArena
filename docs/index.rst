.. meta::
   :description: AgentKernelArena provides real GPU optimization tasks and independently evaluated reward signals for agent reinforcement learning and controlled benchmarking.
   :keywords: AgentKernelArena, A/B testing, agent reinforcement learning, ROCm, GPU, kernel optimization, AI agent, HIP, Triton, AMD

******************************
AgentKernelArena documentation
******************************

AgentKernelArena provides GPU optimization tasks, controlled execution and
independently evaluated reward signals for agent reinforcement learning.
Tasks range from individual operators to end-to-end model workloads. Use the
same tasks, hardware and evaluation policy to compare agents, trained policies,
models, prompts or tools. Training and policy updates belong to the external
training system.

The AgentKernelArena source code is hosted in the
`AMD-AGI/AgentKernelArena <https://github.com/AMD-AGI/AgentKernelArena>`_ GitHub
repository.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Install AgentKernelArena <install/install>`

   .. grid-item-card:: How to

      * :doc:`Run an experiment <how-to/run-evaluation>`
      * :doc:`Check kernels with evaluation tools <how-to/use-evaluation-tools>`
      * :doc:`Run tasks in parallel across multiple GPUs <how-to/parallel-run>`
      * :doc:`Configure agents and models <how-to/agents>`
      * :doc:`Task definition, schema, and authoring <how-to/add-task>`
      * :doc:`Optimize kernels through serving throughput <how-to/e2e-kernel-tasks>`
      * :doc:`Validate tasks <how-to/task-validator>`
      * :doc:`Audit and repair tasks <how-to/quality-loop>`

   .. grid-item-card:: Examples

      * :doc:`Examples <examples/examples>`

   .. grid-item-card:: Reference

      * :doc:`Configuration and API reference <reference/api-reference>`
      * :doc:`Performance measurement methodology <reference/benchmark-methodology>`
      * :doc:`MI355X runtime and compatibility <reference/mi355x-runtime>`

To contribute to the documentation, see the
`AgentKernelArena GitHub repository <https://github.com/AMD-AGI/AgentKernelArena>`_.

AgentKernelArena is released under the Apache 2.0 license. For details, see the
:doc:`License <about/license>` page.
