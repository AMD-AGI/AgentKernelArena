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
qualified tasks. Executable adapters target SGLang with Qwen3-0.6B on one MI355X
and Qwen3-Next-80B on four MI355X devices. Each task version requires its own
qualification; other models/frameworks need their own integration.

## Run configuration

See `example_configs/e2e_qwen3_codex_mi355x.yaml`. `tasks` selects an executable
workload. `resources.gpu_groups` contains indices into the scheduler-assigned GPU
pool: `[[0, 1], [2, 3]]` creates two workers with two GPUs each. Group size must
match every selected task's runtime lock. Overlap and unallocated devices fail.
The same mapping applies to Docker and Slurm/Spur entrypoints.

Whole-model tasks declare the backend-neutral `candidate.language: gpu` contract.
Use an integration that accepts it, such as the direct Codex/Claude Code agents
or GEAK. The pinned Forge engine has no generic `gpu` backend and rejects these
tasks explicitly; selecting a narrower language would change the task contract.

`budget.task_wall_time_s` is the per-task wall budget (86400 or 172800 for 24 or
48 hours), including setup and final evaluation. Its absolute deadline is saved
outside the candidate and retained on resume. The agent receives the remaining
search time after reserving final evaluation. An explicit run-level agent timeout
can shorten it; the agent template's default does not override the run budget.
Each task action also retains its own timeout, capped by the remaining wall time.

`final_evaluation_reserve_s` is a floor. Initial baseline action durations provide
a second floor: 1.5 times compile + correctness + all final A/B performance actions.
The independent verifier timeout is added to the measured reserve.
The runtime lock can specify a further minimum. Search does not begin if that
reserve cannot fit. A timeout or incomplete measurement cannot produce a gain.

For a measurement campaign with a preselected repetition count, set `budget.final_evaluation_pairs`
to an integer from 1 to 100. Otherwise the task's declared pair count applies.
This run-level policy is persisted with the deadline and cannot change on resume;
it controls both reserve estimation and the final fresh A/B count without editing
the task or its correctness checks. Reports preserve the task's declared count,
the override, and all raw measurements. One pair reports unavailable noise estimates,
never zero variance or statistical significance. Choose the count before the run;
do not select the best result from extra repetitions.

Prepare immutable dependencies and weights on the Docker host, then validate:

```bash
python3 -m venv ~/.local/share/aka-serving-venv
~/.local/share/aka-serving-venv/bin/python -m pip install PyYAML
export AKA_SERVING_PYTHON=~/.local/share/aka-serving-venv/bin/python
"$AKA_SERVING_PYTHON" -m src.prepare_serving_runtime --config example_configs/e2e_qwen3_validator_mi355x.yaml
make docker-run CONFIG=example_configs/e2e_qwen3_validator_mi355x.yaml
make docker-run CONFIG=example_configs/e2e_qwen3_codex_mi355x.yaml
```

Serving asset preparation and the clean executor require host Python with PyYAML.
`AKA_SERVING_PYTHON` selects that interpreter (default `python3`). Ordinary kernel
runs do not require host PyYAML: metadata parsing falls back to the selected
Docker image, with read-only checkout/config mounts, no GPU devices, credentials
or network access. Parser failures stop launch rather than assuming a kernel task.

Asset acquisition and executor orchestration run on the host; GPU experiments use Docker. The asset
cache defaults to `~/.cache/aka-serving`; set `AKA_SERVING_CACHE` on the host for
another prepared location. For Slurm, use the corresponding `make slurm-run`
entrypoint and request enough scheduler wall time for the task's budget.

## Runtime and integrity

The task's `evaluation.measurement.runtime_lock` is authoritative. It fixes an
image digest, model revision, GPU count, workload hash, benchmark revisions,
and the composed Magpie/InferenceX scripts. Model and dependency revisions must
be full commit identifiers; moving branches and tags are rejected. The host verifies
prepared file hashes and unchanged file identity before each
action. Conflicting image overrides, mixed images or mixed kernel
and serving tasks in one experiment fail before execution.

Host action artifacts and trusted templates live outside the agent-mounted checkout,
under `~/.local/state/aka-serving` by default (`AKA_SERVING_ARTIFACT_ROOT` overrides it).
The host reconstructs a fresh task from a trusted template for every action and
copies declared editable kernel files and source trees for candidate actions. A
separate container loads these files through protected integration hooks. Every
replacement must execute with the expected bundle hash. Independent model
observers record effective settings and model execution on every TP rank.
Effective server settings and selected runtime environment are compared across
paired measurements. Initial sources preserve the pinned production implementation.
`AKA_DOCKER_WORKDIR` selects the worker checkout mount (default `/workspace`);
the trusted launcher passes the same root to the executor. Submitted paths must
remain within that checkout and outside its task templates, including after
symlink resolution.

Leftover worker processes invalidate performance for any task. Serving also
requires a successful process sweep; an unavailable or failed sweep blocks
performance. Compilation and correctness evidence remain in the report, but
no speedup or performance credit enters scoring or aggregation.

New files inside a declared editable source tree are copied and hashed. Other new
workspace files are not copied, including import shadows such as `scripts/yaml.py`
or `scripts/usercustomize.py`.
Changes to the workspace's harness or runtime lock cannot replace the trusted
template. The general workspace guard additionally removes newly added files
matching protected harness patterns before scoring, or rejects them in strict
mode. Its captured `new_file_policy` describes this behavior; unrelated scratch
files are not prohibited by that guard.

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

Direct `codex` runs and `task_validator` backends also save complete sanitized
stdout/stderr events beside the workspace, in a private directory created for
each invocation. The framework log prints its location. Events are flushed as
they arrive, preserving tool output and failure details beyond the shortened
console summary even if the backend times out. These logs are diagnostics, not
grading evidence; keep them private and review them before sharing.

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

When an action fails before producing its declared cases, the framework rejects
the incomplete evidence and retains the original task failure reason in the
error summary. The full command output remains in the action record.

Clean actions run as a non-root user. The shared adapter removes inaccessible or
missing directories from the image's executable search path, preserving the
order of usable tools. This lets optional compiler probes report a missing tool
normally when the image contains a private root-home path. It does not change
serving flags, compiler options, or the pinned image.

The adapter selects the HTTP port and its derived auxiliary port outside the
host's ephemeral TCP range and checks both before launch. Another process can
still race that check; a logged bind failure rejects the action and cleans up
the process group even if the server parent remains alive. Readiness requests
have short timeouts bounded by the original startup deadline. Failed starts
remain failed evidence and do not contribute measurements.

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

## Measurement and results

The schema-v2 action envelope remains `arena-eval-v1`. Only tasks declaring
`measurement.kind: serving` can report `serving_wall_clock`. `execution_time_ms`
is actual client wall duration; throughput is a separate metric, checked against
completed requests, output tokens and that duration.

By default, final evaluation alternates baseline/candidate order for at least
three fresh pairs; the explicit run-level override above can reduce or increase this count.
Each case uses the median paired throughput ratio; multiple cases use a
geometric mean. Raw pair samples and rank evidence are retained. An optional
`max_p99_tpot_ms` rejects candidates exceeding a fixed tail-latency threshold.
Every pair must identify the baseline and candidate kernel hashes, and each
role's hash must remain fixed across pairs. Identical hashes cap that case's
scored ratio at 1.0: runtime variation cannot earn kernel-optimization credit
without a kernel change. Reports retain `raw_speedup_ratio`, raw `case_ratios`
and `paired_ratios`, alongside `kernel_changed` and `scored_case_ratios`.
These tasks set `minimum_speedup_ratio: 1.05`: a positive median gain below 5%
is retained as a raw measurement but earns no optimization credit (scored ratio
1.0). Regressions remain visible. This is an engineering cutoff, not a
significance test. Choose dozens of full pairs when the budget supports them;
each pair starts fresh baseline and candidate servers. This count is distinct
from requests per server and from the canonical kernel helper's 100 Graph timing
samples. Do not replace end-to-end wall-clock throughput with kernel-only timing;
the fixed serving runtime already uses its declared graph configuration.

Before accepting or exporting a serving candidate, the framework starts a fresh
read-only verification agent on frozen original/candidate sources and the final
measurement evidence. Configure it at run level, independently of the optimizer:

```yaml
candidate_verification:
  backend: claude_code
  timeout_seconds: 600
  # Optional model and effort use the review CLI's supported values.
```

The Docker runner provisions this additional backend. A current CLI supporting
restricted/safe mode and Read/Glob/Grep-only tools is required; missing credentials,
unsupported options, timeout, malformed reports and inconclusive findings block
acceptance. The reviewer has no editing, shell or external MCP tools. It checks
allowed scope, computation/precision/work preservation, measurement integrity,
and whether computation actually changed. Pure wrappers may pass but cannot earn
gain credit. Reports bind evidence hashes and source locations; source mutations
invalidate the review. The policy is fixed before optimization and on resume.
This supplements runtime fingerprints and numerical tests. It cannot override a
failed numeric gate or guarantee detection of every possible malicious program.

The existing score remains 120 plus 100 times the accepted ratio after all gates;
no missing ratio is reconstructed from average durations. A measured gain is
not a statistical-significance claim: reports retain all pairs, indicate
whether every pair improved, and record per-case `uncertainty` with the ratio
range, the repeat-to-repeat throughput variation of each side, and whether the
median ratio lies inside that variation. Kernel and serving summaries are
grouped separately.

### Expected signal

An end-to-end ratio is bounded by the time share of the changed computation.
The original Qwen3-0.6B RMSNorm-only qualification showed that gains in a small
operator can be comparable to serving repeat variation. The current bundle
contract permits other kernels and fusion, so that historical result does not
bound the new task's optimization scope. Agents may use profiling to identify
attention, GEMM, MoE or other useful targets. The locked client window must stay
long enough (`minimum_measurement_s`) that startup and ramp effects do not dominate.

Historical qualification evidence for the original task is recorded in
[the e2e qualification record](../reference/e2e-qualification-2026-09-29.md).
Changing `benchmark.yaml`, the lock, or `config.yaml` changes the task contract
and requires a fresh validator PASS before the task is treated as qualified.

Qualification requires a fresh framework-finalized task-validator PASS on the
locked GPU runtime. CPU regression tests alone do not qualify an e2e task.

### Variable request lengths and large models

Serving manifests may declare `input_lens` and `output_lens` arrays instead of
uniform token counts. The trusted adapter derives these arrays from the pinned
client's seed and length distribution. Metrics must match every request and the
aggregate totals; matching only the total token count is insufficient. Original
catalog recipes remain immutable, including their random length ratios.

The `e2e/qwen3_next_80b_sglang` task binds the imported FP8 recipe to a
four-device runtime. Its initial Gated DeltaNet implementation has output and
recurrent-state oracles, including graph replay with changing inputs and state
slots. Other compute regions may also be replaced or fused through the bundle
contract below. Those operator oracles do not cover every possible replacement.

Whole-model acceptance retains a fixed, attributed 64-question GSM8K regression
subset, with a locked absolute accuracy floor and maximum loss against a fresh
uninjected original-image reference. This is not the full Magpie/lm-eval suite.
Short prompt/decode samples and one full concurrent batch using the scored
workload's input/output lengths compare input-prefix logprobs under identical
token prefixes against the locked numerical threshold. Generated continuations are
retained as diagnostics because divergent tokens have different contexts and
cannot be compared pointwise. Baseline calibration must pass before search;
do not convert a failed candidate's required check into a diagnostic afterward.
Both request modes are checked: logprob collection and the no-logprob path used
by scoring. For the latter, every generated token in the small task must be an
FP32 reference argmax within the same locked logprob tolerance. Teacher forcing
checks all generated positions even when a near tie changes the continuation.
The FP8 production-quality task permits normal differences in generated text.
It scores each actual no-logprob continuation by teacher forcing on the separate
original server, and bounds per-case maximum, mean and 99th-percentile greedy
regret. Input-prefix logprob differences have separate per-case bounds on the
same three statistics. These limits live in `correctness.production_tolerances`
in the protected runtime lock and apply equally to original and candidate runs.
They must be calibrated on repeated unmodified runs before search, then checked
with held-out original runs and deliberately incorrect implementations. They are
not an FP32 oracle or proof of equivalence for every possible fusion. Missing,
misaligned or nonfinite probabilities still fail. No case can dilute another
case's error by contributing many easy tokens.

The fixed answer-accuracy subset remains an additional regression gate. Its
minimum accuracy, allowed drop, sample count and answer parser remain task-owned;
normal-run calibration does not authorize an agent to change them. Threshold
changes are task maintenance and require new GPU qualification.

Host-prepared shared model directories are mounted read-only. Every action checks
content hashes as well as the unchanged file inventory, inode, size, modification
time and change time, including the original manifest. Equal metadata does not
skip hashing: rapid writes can share a filesystem timestamp tick. This preparation
cost is outside throughput timing but consumes the action's setup budget. Changes
are rejected; the asset identity never refreshes within an experiment. Hidden model
download metadata is excluded. This assumes a trusted host and filesystem, not
protection against host root. Keep the manifest outside the agent checkout. An absolute
host asset location belongs in private runtime preparation, never task source.
For persistent experiment state, the Slurm wrapper respects `AKA_CONTAINER_HOME`
and `AGENT_HOME_ISOLATION` overrides; the configured home must have a backing mount.
These do not change serving parameters. GPU runtime temporary storage must support
the runtime's file operations; preserve logs and results in the experiment store.

## Model-wide kernel bundles

The executable Qwen3 and Qwen3-Next tasks now declare a backend-neutral `gpu`
source tree. Their initial RMSNorm/GDN implementations are seeds, not a closed
list of editable kernels. `source/patches.json` selects callable compute boundaries
in the pinned image. The direct CLI agents and GEAK accept this backend-neutral
contract. Backend-specific integrations still need a compatible capability;
Forge currently requires a concrete registered backend and rejects `gpu`.
The manifest supports multiple files and model/layer-level fusion in the locked runtime.
The protected installer rejects duplicate targets, escaping paths and targets
outside the declared compute namespaces. It records complete bundle hashes,
actual replacement calls and model execution in every TP rank. Runtime identity
is computed from fixed per-rank model observations, independent of patch count.

The clean executor rebuilds the source tree, including additions and deletions,
inside a fresh evaluator copy. It never overlays arbitrary agent directories on
the runtime image. HIP/C++ extensions build from declared source under the pinned
toolchain with local temporary caches. These are integrity controls, not a Python
or native-code security sandbox. Arbitrary native code still requires review.

Profiling, call/workload analysis and fusion exploration are optional recommendations
to the agent. There is no mandatory profiling API, report format or automatic
per-fusion oracle generation. Existing targeted operator checks do not claim
coverage of every model-wide replacement. Whole-model numerical and quality gates
remain mandatory, with their finite coverage stated in task instructions.

Serving GEAK runs reserve up to 45 seconds of their existing agent budget for
artifact-only delivery. Within the same search invocation, identical source/task/
harness bytes may reuse passed compile and correctness results; performance is
never cached. Timeout recovery uses the last correctness-checked checkpoint when
available and records recovery separately from engine success. The independent
framework reserve always checks delivered bytes again. `optimization_outcome`
distinguishes unchanged controls, accepted changes, rejected changes and incomplete
runs; it does not assert a statistically significant performance gain.

Before optimization, required baseline correctness compares the frozen seed with
an independent unpatched server. For the production FP8 quality task, this is
recorded as baseline calibration with both answer sets and fixed thresholds. A
failing calibration blocks search; do not rerun until a convenient score appears
or adjust tolerances after seeing a candidate. An accepted calibration is a
finite check, not a variance estimate. Candidate quality or numerical failures
retain their case identity and failure category instead of becoming a manifest
mismatch.
