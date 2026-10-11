# Whole-case GPU groups

Tasks may opt into `trusted_evaluation.case_parallelism.schema_version: 1`.
The task's protected `ut/case_parallel.py` and `ut/gpu_binding.py` must be exact
copies of the canonical helpers in `src/tools/`. Without an explicit plan the
existing single-GPU command and measurement work remain unchanged.

The framework still launches one full-task validator. A phase command starts
fresh worker interpreters and distributes complete original cases over a fixed
physical GPU group. Every case keeps its complete setup, compilation, seeds,
negative controls, warmups, samples, recorded-state controls and paired native
reference. The original `cases.json` is never partitioned or rewritten.

`AKA_CASE_GPU_PLAN` names a read-only `case-gpu-plan-v1` JSON document and
`AKA_CASE_GPU_PLAN_SHA256` binds its exact bytes. A usable plan is a fresh,
explicitly authorized allocation grant; a template is not authorization. It
binds the task path, full manifest digest, image digest, baseline commit, job,
node, allocation receipt digest, restart generation, work cutoff, ordered
physical GPU bindings, and ordered per-worker case assignments. GPU bindings
contain `worker_id`, `render_device`, `drm_minor`, `kfd_node`, `rocr_uuid`, and
`pci_bus_id`. Assignments contain `worker_id` and `case_ids`; their canonical
JSON digest is `assignment_sha256`.

Capacity is explicit. `allocation_capacity` contains positive integer `cpus`,
`memory_gib`, and `gpus`. `resources` contains positive integer
`cpus_per_worker`, `memory_gib_per_worker`, `coordinator_cpus`, and
`coordinator_memory_gib`. Worker sums plus coordinator headroom must fit the
fresh allocation. Optional `total_cpus` and `total_memory_gib` include that
headroom. The helper never reduces task work or silently shrinks resource
requests to make a plan fit.

Each worker receives the identical parent request through `--request` and a
separate fresh `--case-worker` assignment. The assignment binds the parent,
plan, fixed case mapping, worker nonce, source leg, physical GPU, cutoff and
independently computed input-tree digest. A group parent request includes
`gpu_group` and `case_parallelism`, and omits the single-device `gpu` field.
The original producer's package and source identity fields are preserved.

Workers select their stable ROCr UUID before importing Torch or task native
runtime code. They prove one HSA UUID and one HIP PCI identity. Their private
outputs live under `build/case-parallel/<request_id>/worker-<id>/`. A worker
writes only a `whole-case-worker-result-v1` receipt, marked `full_task: false`
and `scoreable: false`; it cannot publish scoreable cases or a validator PASS.

The coordinator observes process exits, checks all receipts, restores original
manifest order and returns the complete raw union. It cancels owned sibling
process groups when a worker fails or the cutoff expires, retaining logs. It
does not retry, rebalance, resume an existing request, or reuse old receipts.
The task then runs its existing full-report finalizer and paired comparison
validator once. The framework independently checks the group receipt chain
before scoring. All original task-validator checks remain required.

CPU tests cover receipt integrity, exact unions, fresh-process dispatch and
failure cancellation. They do not qualify GPU behavior or eight-GPU scaling;
a fresh original full-task GPU validator pass remains required.
