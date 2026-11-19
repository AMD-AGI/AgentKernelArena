---
myst:
    html_meta:
        "description": "Reference for AgentKernelArena end-to-end serving task configuration: budget mechanics, integrity boundaries, measurement policy, and model-wide kernel bundles."
        "keywords": "AgentKernelArena, e2e, serving, runtime lock, budget, integrity, measurement, scoring, kernel bundles"
---

# AgentKernelArena end-to-end serving task reference

This reference covers the mechanics behind end-to-end serving tasks:
budget and reserve calculations, the runtime and integrity model, measurement
and scoring policy, and the model-wide kernel bundle contract. For how to run
the maintained tasks, see [Run end-to-end serving tasks](../how-to/e2e-kernel-tasks.md).
For how to qualify a new Magpie catalog task, see
[Qualify a Magpie serving task](../how-to/e2e-magpie-qualification.md).

## Budget mechanics

`budget.task_wall_time_s` is the per-task wall budget (86400 or 172800 for 24
or 48 hours), including setup and final evaluation. Its absolute deadline is
saved outside the candidate and retained on resume. The agent receives the
remaining search time after reserving final evaluation. An explicit run-level
agent timeout can shorten it; the agent template's default does not override
the run budget. Each task action also retains its own timeout, capped by the
remaining wall time.

`final_evaluation_reserve_s` is a floor for the reserve. Two additional floors
are computed automatically: 1.5× (compile + correctness + all final A/B
performance actions), and the independent verifier timeout. The runtime lock
can specify a further minimum. Search does not begin if the reserve cannot fit.
Resume still evaluates retained candidate files within the original deadline
and writes a result even if evaluation exhausts it. A timeout or incomplete
measurement cannot produce a gain.

`agent_execution.candidate_changed` describes the current agent invocation.
`candidate_changed_from_initial` and `optimization_outcome` compare the
retained submission with the original task, including after a resume with no
new search.

`budget.final_evaluation_pairs` overrides the task's declared pair count (3 to
100 pairs; at least the task's declared count). This run-level policy is
persisted with the deadline and cannot change on resume. It controls both
reserve estimation and the final fresh A/B count without editing the task or
its correctness checks. Reports preserve the task's declared count, the
override, and all raw measurements.

## Runtime and integrity

The task's `evaluation.measurement.runtime_lock` is authoritative. It fixes an
image digest, model revision, GPU count, workload hash, benchmark revisions,
and the composed Magpie and InferenceX scripts. Model and dependency revisions
must be full commit identifiers; moving branches and tags are rejected. The
host verifies prepared file hashes and unchanged file identity before each
action. Conflicting image overrides, mixed images, or mixed kernel and serving
tasks in one experiment fail before execution.

Host action artifacts and trusted templates live outside the agent-mounted
checkout, under `~/.local/state/aka-serving` by default (`AKA_SERVING_ARTIFACT_ROOT`
overrides this). The host reconstructs a fresh task from a trusted template for
every action and copies declared editable kernel files and source trees for
candidate actions. A separate container loads these files through protected
integration hooks. Every replacement must execute with the expected bundle hash.
Independent model observers record effective settings and model execution on
every TP rank. Effective server settings and selected runtime environment are
compared across paired measurements.

`AKA_DOCKER_WORKDIR` selects the worker checkout mount (default `/workspace`).
Submitted paths must remain within that checkout and outside its task templates,
without following symlinks. Files are read through open directory descriptors
and bounded before being copied into the trusted action directory. A
disconnected caller cancels its active action; container removal is confirmed
before the next GPU action starts. Cleanup retries transient removal or
inspection failures for up to 90 seconds. A failed removal command alone does
not establish whether a container remains; only a successful exact-name query
showing absence permits further GPU work. Lingering or unknown container state
blocks subsequent actions.

Leftover worker processes invalidate performance for any task. Serving also
requires a successful process sweep; an unavailable or failed sweep blocks
performance. Compilation and correctness evidence remain in the report, but no
speedup or performance credit enters scoring or aggregation.

New files inside a declared editable source tree are copied and hashed. Other
new workspace files are not copied, including import shadows such as
`scripts/yaml.py` or `scripts/usercustomize.py`. Changes to the workspace's
harness or runtime lock cannot replace the trusted template. The general
workspace guard additionally removes newly added files matching protected
harness patterns before scoring, or rejects them in strict mode.

The small task checks kernels against independent FP32 calculations, then
checks fixed prompt and decode log probabilities against a Transformers FP32
reference. Its protected source policy also rejects environment and filesystem
imports and runtime attribute changes. This is an integrity boundary, not a
Python security sandbox. Model equality is numerical within declared tolerances,
not token determinism.

On clusters that prohibit privileged workers, `AKA_DOCKER_PRIVILEGED=0` keeps
Docker's default capabilities. If the default bridge cannot reach the agent's
provider, set `AKA_DOCKER_NETWORK=host` where the cluster permits it. This
changes only the worker's network namespace and does not change clean
evaluation-container settings, model flags, or workload.

### Isolation before final measurement

The optimization agent runs in the ordinary worker container, which shares the
GPU group and the host network with the clean evaluation containers. Two checks
keep agent leftovers out of the final A/B measurement:

- Before final evaluation, the framework stops every process in the worker
  container outside its own ancestry and records them in `process_sweep.json`
  and in `task_result.yaml` under `process_sweep`. The Docker runner marks
  `run` and parallel worker containers with
  `AGENT_KERNEL_ARENA_DEDICATED_CONTAINER=1`; the sweep refuses to run
  anywhere else. A survivor, or a serving task without a sweepable container,
  makes the candidate not accepted.
- Before each serving performance action, the clean container checks that no
  process already holds the measured devices. Used memory above the lock's
  `idle_vram_limit_mib` fails that action.

### Runtime limits

When an action fails before producing its declared cases, the framework rejects
the incomplete evidence and retains the original task failure reason in the
error summary.

Clean actions run as a non-root user. The shared adapter removes inaccessible
or missing directories from the image's executable search path, preserving the
order of usable tools.

The adapter selects the HTTP port and its derived auxiliary port outside the
host's ephemeral TCP range and checks both before launch. A logged bind failure
rejects the action and cleans up the process group. Failed starts remain failed
evidence and do not contribute measurements.

The lock's optional `limits` mapping replaces adapter constants. Every value is
a positive integer; omitted keys use the defaults in `src/measurement.py`:

| Key | Default | Meaning |
| --- | ---: | --- |
| `server_startup_timeout_s` | 600 | Wait for the framework server to answer its readiness endpoint. |
| `client_timeout_s` | 500 | Wall limit for the pinned benchmark client. |
| `max_candidate_source_bytes` | 16 MiB | Maximum submitted source bytes; applies to the whole tree for a kernel bundle. |
| `idle_vram_limit_mib` | 2048 | Used device memory tolerated before a performance action starts. |
| `minimum_measurement_s` | 30 | Shortest accepted client measurement window; a shorter run fails instead of scoring. |

Agent-invoked checks through `scripts/evaluate.py` use the task's declared
action timeout from `config.yaml`; the host executor caps every request by that
same declaration.

## Measurement and scoring

The schema-v2 action envelope remains `arena-eval-v1`. Only tasks declaring
`measurement.kind: serving` can report `serving_wall_clock`. `execution_time_ms`
is actual client wall duration; throughput is a separate metric, checked against
completed requests, output tokens, and that duration.

By default, final evaluation alternates baseline and candidate order for at
least three fresh pairs. Each case uses the median paired throughput ratio;
multiple cases use a geometric mean. Raw pair samples and rank evidence are
retained. An optional `max_p99_tpot_ms` rejects candidates exceeding a fixed
tail-latency threshold.

Every pair must identify the baseline and candidate kernel hashes, and each
role's hash must remain fixed across pairs. Identical hashes cap that case's
scored ratio at 1.0: runtime variation cannot earn kernel-optimization credit
without a kernel change. These tasks set `minimum_speedup_ratio: 1.05`: a
positive median gain below 5% is retained as a raw measurement but earns no
optimization credit (scored ratio 1.0). Regressions remain visible.

Reports preserve `raw_speedup_ratio`, raw `case_ratios` and `paired_ratios`,
alongside `kernel_changed` and `scored_case_ratios`. Repetition spread is
reported without claiming statistical significance. Choose the count before the
run; do not select the best result from extra repetitions.

Serving tasks use the shared [scoring policy](api-reference.md#scoring): a
bounded score in `[0, 1]`, with the default linear curve requiring more than
1.05× speedup for positive reward. All applicable acceptance gates must pass;
no missing ratio is reconstructed from average durations.

### Candidate verification

Before accepting or exporting a serving candidate, the framework starts a
fresh read-only verification agent on frozen original and candidate sources and
the final measurement evidence. Configure it at run level, independently of
the optimizer:

```yaml
candidate_verification:
  backend: claude_code
  timeout_seconds: 600
  # Optional model and effort use the review CLI's supported values.
```

The Docker runner provisions this additional backend. A current CLI supporting
restricted or safe mode and Read, Glob, and Grep-only tools is required.
Missing credentials, unsupported options, timeout, malformed reports, and
inconclusive findings block acceptance. The reviewer has no editing, shell, or
external MCP tools. It checks allowed scope, computation and precision and work
preservation, measurement integrity, and whether computation actually changed.
Wrappers identified as unchanged computation may pass correctness but cannot
earn gain credit. This supplements runtime fingerprints and numerical tests
but cannot override a failed numeric gate or guarantee detection of every
possible malicious program.

### Expected signal and measurement window

An end-to-end ratio is bounded by the time share of the changed computation.
The original Qwen3-0.6B RMSNorm-only qualification showed that gains in a
small operator can be comparable to serving repeat variation. The current
bundle contract permits other kernels and fusion, so that historical result
does not bound the new task's optimization scope. Agents may use profiling
to identify attention, GEMM, MoE, or other useful targets.

The locked client window must stay long enough (`minimum_measurement_s`) that
startup and ramp effects do not dominate. Choose dozens of full pairs when the
budget supports them; each pair starts fresh baseline and candidate servers.
This count is distinct from requests per server and from the canonical kernel
helper's 100 graph timing samples. Do not replace end-to-end wall-clock
throughput with kernel-only timing; the fixed serving runtime already uses its
declared graph configuration.

Serving manifests may declare `input_lens` and `output_lens` arrays instead of
uniform token counts. The trusted adapter derives these arrays from the pinned
client's seed and length distribution. Metrics must match every request and
the aggregate totals; matching only the total token count is insufficient.
Original catalog recipes remain immutable, including their random length ratios.

## Model-wide kernel bundles

Whole-model tasks declare the backend-neutral `candidate.language: gpu`
contract. The executable Qwen3 and Qwen3-Next tasks declare a backend-neutral
`gpu` source tree. Their initial RMSNorm and GDN implementations are seeds, not
a closed list of editable kernels. `source/patches.json` selects callable
compute boundaries in the pinned image.

The protected installer rejects duplicate targets, escaping paths, and targets
outside the declared compute namespaces. It records complete bundle hashes,
actual replacement calls, and model execution in every TP rank. Runtime
identity is computed from fixed per-rank model observations, independent of
patch count. For compiled regions it records `compiled_trace` separately from
direct calls; the model must still execute on every rank. Tracing alone is not
GPU execution evidence.

The clean executor rebuilds the source tree, including additions and deletions,
inside a fresh evaluator copy. It never overlays arbitrary agent directories on
the runtime image. HIP and C++ extensions build from declared source under the
pinned toolchain with local temporary caches. These are integrity controls, not
a Python or native-code security sandbox. Arbitrary native code still requires
review.

The bundle excludes configuration and observation namespaces from replacement
targets, so a candidate cannot declare the evaluator's own configuration
accessor as a kernel. The separate `candidate_verifier` agent reviews the
frozen implementation and execution evidence for preserved semantics,
generality, and reward hacking. It must not demand a hand-written oracle for
each replacement or treat the optimizer's claims as evidence. A failed,
unavailable, or inconclusive required review blocks acceptance; review never
overrides numerical failures.

The independent framework reserve always checks delivered bytes again,
including when search has exhausted its budget. Resuming within that reserve
finalizes the retained candidate without renewing the original deadline.
`optimization_outcome` distinguishes unchanged controls, accepted changes,
rejected changes, and incomplete runs; it does not assert a statistically
significant performance gain.

New model-wide workloads should reuse high-level checks rather than enumerate
the model's current operator set. Every compute kernel used by the model is an
eligible optimization target, including newly introduced kernels and fusion.
Profiling, call and workload analysis, and fusion exploration are optional
recommendations to the agent. There is no mandatory profiling API, report
format, or automatic per-fusion oracle generation.

Changing `benchmark.yaml`, the runtime lock, or `config.yaml` changes the task
contract and requires a fresh validator PASS before the task is treated as
qualified.
