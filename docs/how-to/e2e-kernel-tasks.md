---
myst:
    html_meta:
        "description": "Define fixed-workload serving tasks in AgentKernelArena, prepare pinned runtimes, budget long searches, and read paired throughput results."
        "keywords": "AgentKernelArena, e2e, serving, SGLang, Magpie, InferenceX, runtime lock, GPU groups, task budget"
---

# End-to-end kernel optimization

Serving tasks measure model throughput while allowing only kernel source changes.
TP/EP/PP, weights, precision, serving flags, request concurrency, prompt lengths,
output lengths and benchmark policy remain fixed before either implementation runs.
The imported Magpie recipes in `tasks/e2e/` are workload data, not automatically
qualified tasks. The first executable adapter targets SGLang and Qwen3-0.6B on
one MI355X; other models/frameworks need their own integration and qualification.

## Run configuration

See `example_configs/e2e_qwen3_codex_mi355x.yaml`. `tasks` selects an executable
workload. `resources.gpu_groups` contains indices into the scheduler-assigned GPU
pool: `[[0, 1], [2, 3]]` creates two workers with two GPUs each. Group size must
match every selected task's runtime lock. Overlap and unallocated devices fail.
The same mapping applies to Docker and Slurm/Spur entrypoints.

`budget.task_wall_time_s` is the per-task wall budget (86400 or 172800 for 24 or
48 hours), including setup and final evaluation. Its absolute deadline is saved
outside the candidate and retained on resume. The agent receives the remaining
search time after reserving final evaluation. An explicit run-level agent timeout
can shorten it; the agent template's default does not override the run budget.
Each task action also retains its own timeout, capped by the remaining wall time.

`final_evaluation_reserve_s` is a floor. Initial baseline action durations provide
a second floor: 1.5 times compile + correctness + all final A/B performance actions.
The runtime lock can specify a further minimum. Search does not begin if that
reserve cannot fit. A timeout or incomplete measurement cannot produce a gain.

Prepare immutable dependencies and weights on the Docker host, then validate:

```bash
python3 -m src.prepare_serving_runtime --config example_configs/e2e_qwen3_validator_mi355x.yaml
make docker-run CONFIG=example_configs/e2e_qwen3_validator_mi355x.yaml
make docker-run CONFIG=example_configs/e2e_qwen3_codex_mi355x.yaml
```

Only asset acquisition runs on the host; GPU experiments use Docker. The asset
cache defaults to `~/.cache/aka-serving`; set `AKA_SERVING_CACHE` on the host for
another prepared location. For Slurm, use the corresponding `make slurm-run`
entrypoint and request enough scheduler wall time for the task's budget.

## Runtime and integrity

The task's `evaluation.measurement.runtime_lock` is authoritative. It fixes an
image digest, model revision, GPU count, workload hash, benchmark revisions,
and the composed Magpie/InferenceX scripts. Model and dependency revisions must
be full commit identifiers; moving branches and tags are rejected. The host verifies cached file hashes
before each action. Conflicting image overrides, mixed images or mixed kernel
and serving tasks in one experiment fail before execution.

Host action artifacts and trusted templates live outside the agent-mounted checkout,
under `~/.local/state/aka-serving` by default (`AKA_SERVING_ARTIFACT_ROOT` overrides it).
The host reconstructs a fresh task from a trusted template for every action and
copies only declared editable kernel files for candidate actions. A separate
container loads these files through protected integration hooks. Every rank must
record actual calls to both target operators with the expected source hash.
Effective server settings and selected runtime environment are compared across
paired measurements. The initial source uses production AITER kernels.

The small task checks kernels against independent FP32 calculations, then checks
fixed prompt and decode log probabilities against a Transformers FP32 reference. Its
protected source policy also rejects environment/filesystem imports and runtime
attribute changes. This is an integrity boundary, not a Python security sandbox.
Model equality is numerical within declared tolerances, not token determinism.

The adapter uses pinned Magpie benchmark scripts composed with pinned InferenceX
sources. It disables GPU autodiscovery, reuse and profiling; sets local execution;
and preserves complete request/output-token counts. Imported YAML stays verbatim;
`benchmark.yaml` is an explicitly documented small-model variant.

The clean container mounts only its task directory and read-only prepared assets,
with no optimization-agent credentials or Docker socket. The host Unix socket
accepts only task-declared actions and source submissions under the experiment
checkout. Source archives are checked for path traversal and escaping links;
scoring runs offline against prepared assets. GPU containers and the agent's
ordinary repository mounts remain outside a hostile-code security boundary.

### Isolation before final measurement

The optimization agent runs in the ordinary worker container, which shares the
GPU group and the host network with the clean evaluation containers. Two checks
keep agent leftovers out of the final A/B:

- Before final evaluation the framework stops every process in the worker
  container outside its own ancestry and records them in `process_sweep.json`
  and in `task_result.yaml` under `process_sweep`. The Docker runner marks
  `run` and parallel worker containers with
  `AGENT_KERNEL_ARENA_DEDICATED_CONTAINER=1`; the sweep refuses to run anywhere
  else. A survivor, or a serving task without a sweepable container, makes the
  candidate not accepted.
- Before each serving performance action, the clean container checks that no
  process already holds the measured devices. Used memory above the lock's
  `idle_vram_limit_mib` fails that action, and the observed values are kept in
  the action metadata as `gpu_memory_used_before_mib`.

### Runtime limits

The lock's optional `limits` mapping replaces adapter constants. Every value is
a positive integer; omitted keys use the defaults in `src/measurement.py`:

| Key | Default | Meaning |
| --- | ---: | --- |
| `server_startup_timeout_s` | 600 | Wait for the framework server to answer its readiness endpoint. |
| `client_timeout_s` | 500 | Wall limit for the pinned benchmark client. |
| `max_candidate_source_bytes` | 16 MiB | Largest editable file the host executor accepts. |
| `idle_vram_limit_mib` | 2048 | Used device memory tolerated before a performance action starts. |
| `minimum_measurement_s` | 30 | Shortest accepted client measurement window; a shorter run fails instead of scoring. |

Agent-invoked checks through `scripts/evaluate.py` use the task's declared
action timeout from `config.yaml`; the host executor caps every request by that
same declaration.

## Measurement and results

The schema-v2 action envelope remains `arena-eval-v1`. Only tasks declaring
`measurement.kind: serving` can report `serving_wall_clock`. `execution_time_ms`
is actual client wall duration; throughput is a separate metric, checked against
completed requests, output tokens and that duration.

Final evaluation alternates baseline/candidate order for at least three fresh
pairs. Each case uses the median paired throughput ratio; multiple cases use a
geometric mean. Raw pair samples and rank evidence are retained. An optional
`max_p99_tpot_ms` rejects candidates exceeding a fixed tail-latency threshold.
The existing score remains 120 plus 100 times the accepted ratio after all gates;
no missing ratio is reconstructed from average durations. A measured gain is
not a statistical-significance claim: reports retain all pairs, indicate
whether every pair improved, and record per-case `uncertainty` with the ratio
range, the repeat-to-repeat throughput variation of each side, and whether the
median ratio lies inside that variation. Kernel and serving summaries are
grouped separately.

### Expected signal

An end-to-end ratio is bounded by the time share of the editable kernels. In
the first task, RMSNorm is a small fraction of Qwen3-0.6B decode at concurrency
8, so even a much faster kernel moves throughput by well under one percent,
which is the same order as the measured repeat variation. That task exercises
the contract; it is not a discriminating benchmark. Tasks meant to rank
kernels should edit an operator that dominates the profile, such as attention
decode or the main GEMM or MoE path, and should keep the client window long
enough (`minimum_measurement_s`) that startup and ramp effects do not dominate.

Qualification evidence for the first task is recorded in
[the e2e qualification record](../reference/e2e-qualification-2026-09-29.md).
Changing `benchmark.yaml`, the lock, or `config.yaml` changes the task contract
and requires a fresh validator PASS before the task is treated as qualified.

Qualification requires a fresh framework-finalized task-validator PASS on the
locked GPU runtime. CPU regression tests alone do not qualify an e2e task.
