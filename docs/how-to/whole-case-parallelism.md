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

`phase_timeouts` records the task's existing positive integer `compile`,
`correctness`, and `performance` timeouts. Host admission checks them against
the original task config. Each worker deadline is the earlier of that phase
cap and `work_stop_unix`.

Use the existing `make docker-run CONFIG=<one-full-task-validator-config>`
with both plan environment variables set. Host admission verifies the current
Git commit, image, exact task selection, Slurm job state and restart generation,
lease end, local node, allocated CPU/memory/GPU capacity, and every render-node UUID/PCI
binding. It mounts the plan read-only and exposes only `/dev/kfd` and the granted
render devices. The group container omits privileged mode; normal serial Docker
behavior remains unchanged. Group plans are rejected by `docker-parallel-run`.

Native Slurm admission remains the default and requires its `AllocTRES`
allocation counters. A Spur plan must explicitly set `scheduler_provider:
spur`; this invokes only `/usr/local/bin/spur show job <id>`. There is no PATH
lookup for Spur and no automatic fallback between providers. Slash-containing
keys such as `CPUs/Task` are preserved. User command text cannot override the
controller's job fields.

Spur's text display exposes request fields, so they are not relabelled as
`AllocTRES`. The supported Spur mode requires `spur_allocation.schema` to be
`spur-recorded-reservation-v1`, with exact JSON strings and SHA-256 digests for
the root-owned `ready`, `assignment`, `binding`, and `renders` receipts. The
plan's `allocation_receipt_sha256` must match the readiness receipt. This is an
explicit recorded root reservation: the readiness resource profile must agree
with every available controller request field for one node and one task,
including CPU count, memory, GPU count and GPU type. Missing or conflicting
evidence rejects admission.

The same receipts must bind the current job, node, start time and restart
generation, and the complete scheduler-assigned GPU-ID set to the actual render
device set observed inside that job's mount namespace. Physical UUID/BDF/render
identities are checked again on the host. Individual scheduler registry IDs are
never assumed equal to AMD-SMI ordinals. Historical receipts can be used as CPU
test fixtures; a current cancelled job always fails the fresh controller check.

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

The trusted host retest accepts `--case-gpu-plan PATH` and
`--case-gpu-plan-sha256 SHA` instead of `--render-device`. It retains the source,
fixture, image and cache checks of the ordinary trusted path. Each phase uses
one fresh restricted container per binding, then waits for the complete union
before advancing. Eight bindings produce 48 fresh worker containers across the
three phases and both source legs. Every container has its own writable build
mount and keeps the parent request unchanged. The host observes exits and
retains per-worker logs, container state and receipts; failure cancels the
owned sibling containers.

The task exposes a CPU-only `finalize_case_union(manifest, request, union)`
hook shared by its normal parent path and trusted host path. The host invokes
that protected hook in a fresh isolated Python process which cannot import
candidate sources or Torch/Triton. Numerical validation and native paired input
validation still consume the complete original manifest. Trusted measurement
reports record `gpu_group` and the receipt directories; they do not assert a
framework task-validator PASS.
