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
qualified tasks. Maintained task packages include SGLang Qwen3-0.6B and Qwen3-Next-80B.
The [Magpie catalog](#magpie-workload-catalog) materializes additional models through
shared SGLang, vLLM and ATOM backend/scenario adapters. Each bound runtime needs
baseline qualification; catalog membership does not establish a GPU pass.

## Run configuration

See `example_configs/e2e_qwen3_codex_mi355x.yaml`. `tasks` selects an executable
workload. `resources.gpu_groups` contains indices into the scheduler-assigned GPU
pool: `[[0, 1], [2, 3]]` creates two workers with two GPUs each. Group size must
match every selected task's runtime lock. Overlap and unallocated devices fail.
The same mapping applies to Docker and Slurm/Spur entrypoints.

Whole-model tasks declare the backend-neutral `candidate.language: gpu` contract.
The supported integrations for this contract are Codex, Cursor, DeepSeek Harness
and Claude Code. GEAK and KernelForge adaptation is separate work; their existing
single-kernel integrations are unchanged by this feature.

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
reserve cannot fit. Resume still evaluates retained candidate files within the
original deadline and writes a result if evaluation exhausts it. A timeout or
incomplete measurement cannot produce a gain.
`agent_execution.candidate_changed` describes the current agent invocation;
`candidate_changed_from_initial` and `optimization_outcome` compare the retained
submission with the original task, including after a resume with no new search.

For a measurement campaign with a preselected repetition count, set `budget.final_evaluation_pairs`
to an integer from 3 to 100, at least the task's declared pair count. Otherwise
the task's declared count applies; a run cannot reduce it.
This run-level policy is persisted with the deadline and cannot change on resume;
it controls both reserve estimation and the final fresh A/B count without editing
the task or its correctness checks. Reports preserve the task's declared count,
the override, and all raw measurements. Repetition spread is reported without
claiming statistical significance. Choose the count before the run;
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
Each clean action explicitly selects the declared executable as the container
entrypoint. A runtime image's default serving CLI cannot reinterpret evaluator
arguments; all remaining action arguments are passed directly, without a shell.

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
remain within that checkout and outside its task templates, without following symlinks. Files are read through open directory descriptors
and bounded before being copied into the trusted action directory. A disconnected
caller cancels its active action; container removal is confirmed before the next
GPU action starts. Cleanup retries transient removal or inspection failures for
up to 90 seconds and records each attempt in the action's `cleanup_evidence.json`.
A failed removal command alone does not establish whether a container remains;
only a successful exact-name query showing absence permits further GPU work.
Lingering or unknown container state blocks subsequent actions. Requests retain
their original deadline while queued.

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

On clusters that prohibit privileged workers, `AKA_DOCKER_PRIVILEGED=0` keeps
Docker's default capabilities. If the default bridge cannot reach the agent's
provider, explicitly select `AKA_DOCKER_NETWORK=host` where the cluster permits
it. This changes only the worker's network namespace; it grants no additional
capabilities and does not change clean evaluation-container settings, model
flags or workload. Unset preserves the runner's existing network choice.

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
three fresh pairs; the explicit run-level override above can only increase this count.
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
and whether computation actually changed. Wrappers identified as unchanged
computation may pass correctness but cannot earn gain credit. Reports bind evidence hashes and source locations; source mutations
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
workload's input/output lengths compare logprobs under identical token prefixes
against the locked numerical thresholds. Independent free-running continuations
can diverge and cannot be compared pointwise. The production numerical and quality
policies instead teacher-force each actual continuation on the uninjected
original server, in both request modes and with the declared batch concurrency.
Baseline calibration must pass before search;
do not convert a failed candidate's required check into a diagnostic afterward.
Both request modes are checked: logprob collection and the no-logprob path used
by scoring. For the latter, every generated token in the small task must be an
FP32 reference argmax within the same locked logprob tolerance. Teacher forcing
checks all generated positions even when a near tie changes the continuation.
The FP8 production-quality task permits normal differences in generated text.
It scores each actual continuation by teacher forcing on the separate original
server, and bounds per-case maximum, mean and 99th-percentile greedy regret in
both request modes. The diagnostic mode also compares every returned decode
logprob with the original score for that exact token and prefix. All logprobs
must be finite and nonpositive. Input-prefix logprob differences have separate per-case bounds on the
same three statistics. These limits live in `correctness.production_tolerances`
in the protected runtime lock and apply equally to original and candidate runs.
They must be calibrated on repeated unmodified runs before search, then checked
with held-out original runs and deliberately incorrect implementations. They are
not an FP32 oracle or proof of equivalence for every possible fusion. Missing,
misaligned or nonfinite probabilities still fail. No case can dilute another
case's error by contributing many easy tokens.

`correctness.decode_logprob_tolerances` may declare independently calibrated
`max_abs`, `mean_abs` and `p99_abs` bounds for the decode logprob comparison.
Without that optional mapping, the existing input-prefix absolute error bounds
apply to decode too; an older profile does not bypass the new check. The initial
unmodified baseline must pass all checks before optimization. Calibrate any
model-specific decode bounds on original-only runs before search and verify
them on a held-out original run. Do not adjust them to accept a candidate.

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
in the pinned image. Codex, Cursor, DeepSeek Harness and Claude Code accept this
backend-neutral contract. GEAK and KernelForge adaptation is deferred to a
separate change; this change preserves their existing single-kernel integration.
The manifest supports multiple files and model/layer-level fusion in the locked runtime.
The protected installer rejects duplicate targets, escaping paths and targets
outside the declared compute namespaces. It records complete bundle hashes,
actual replacement calls and model execution in every TP rank. Runtime identity
is computed from fixed per-rank model observations, independent of patch count.
For compiled regions it records `compiled_trace` separately from direct calls;
the model must still execute on every rank. Tracing alone is not GPU execution
evidence. The vLLM observer sits outside compiled `forward`, so configuration
inspection and audit I/O do not break or disable its compute graph. Source review
and model checks remain necessary; this is not a per-kernel dispatch proof.
The observer retains the configured compiler-cache directory before the first
model call. Only for the native empty/automatic input, runtime identity normalizes
vLLM's generated source-hash directory suffix. The cache root, explicit overrides,
compile mode, backend, graph settings and precision stay compared. Complete
effective paths remain in the raw evidence.

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

New model-wide workloads should reuse high-level checks rather than enumerate
the model's current operator set. Every compute kernel used by the model is an
eligible optimization target, including newly introduced kernels and fusion.
The framework checks fixed workload/runtime identity, model outputs and fresh
paired measurements. The separate `candidate_verifier` agent reviews the frozen
implementation and execution evidence for preserved semantics, generality and
reward hacking. It must not demand a hand-written oracle for each replacement
or treat the optimizer's claims as evidence. A failed, unavailable or inconclusive
required review blocks acceptance; review never overrides numerical failures.

The independent framework reserve always checks delivered bytes again, including
when search has exhausted its budget. Resuming within that reserve finalizes the
retained candidate without renewing the original deadline. `optimization_outcome`
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

## Magpie workload catalog

The maintained `tasks/e2e/magpie/catalog.json` includes all recipes from its
pinned upstream commit. List or refresh it from a clean upstream checkout:

```bash
python3 -m src.magpie_catalog list
python3 -m src.magpie_catalog refresh --checkout ../Magpie
```

The catalog stores original recipe bytes and hashes, not fifty copies of the
model evaluator. Backend adapters cover SGLang, vLLM and ATOM. Scenario adapters
cover fixed-request text, random multimodal inputs and canonical AgentX replay.
Ray deployment, external endpoints and server reuse are normalized to a fresh
evaluator-owned local server inside the pinned container. Profiling examples use
the same workload with profiling outside scoring. TP/EP/PP, precision, engine
options and workload settings remain protected after binding.
The bundle excludes configuration/observation namespaces from replacement targets,
so a candidate cannot declare the evaluator's own configuration accessor as a
kernel. This boundary does not enumerate compute operators. Indirect mutation and
other semantic violations still require the independent candidate review.

### Bind once, then materialize

A recipe becomes a selectable v2 task after the maintainer supplies a runtime
profile. This step occurs **before optimization**. The profile is private run
input; it is not generated by the optimizing agent and is never refreshed during
a run or resume. It declares:

- `image`: immutable registry digest; `gpu_arch`, `gpu_count`, and actual
  `model_architectures` class names from that model's configuration.
- `model`: the recipe's model ID and exact revision. Host-local model examples
  additionally require an explicit `source_model_alias`.
- `correctness`: the shared numerical reference policy and baseline-calibrated
  tolerances, sample counts and generation lengths. `inputs` contains protected
  `correctness.json` prompts; `production_quality` additionally needs its frozen
  `quality.json` and quality policy. Do not reuse another model's calibration
  without checking it. `production_numerical` uses fresh original-server prefix
  comparisons and teacher forcing of actual candidate continuations, without
  imposing one answer dataset on every model. FP32-capable models can instead
  use `transformers_fp32`.
- Archive-backed `client_runtime.review_sources` may map up to 16 relative
  Python/shell source paths inside that immutable archive to SHA-256 digests.
  Validation evidence includes these exact native timer/phase implementations,
  bound to the executed archive, without importing or changing them. The limit
  is 256 KiB per file and 512 KiB combined. This is useful when an AgentX driver
  runs from a prepared interpreter rather than the benchmark shell recipe.
- ATOM images without a full logprob API expose actual generated IDs through
  the reporting adapter, with exact prompt-ID and completion-count checks. It
  never re-tokenizes decoded text. Bind one text policy before optimization:
  `production_tokens` with `token_match: exact` compares every generated token
  against a fresh original server; `transformers_tokens` instead teacher-forces
  every actual continuation on the independent FP32 model and bounds each
  generated token's regret from the FP32 argmax by `model_logprob_atol`. The
  latter permits near-tied greedy choices while rejecting incorrect tokens,
  requires the complete FP32 reference model to fit the evaluation hardware,
  and does not claim engine logprob equivalence. Both retain the protected short,
  long and concurrent probe plan, must pass original baseline qualification,
  and never switch policies after a failure. Existing tasks keep their original
  checks. Ordinary timed/streaming
  responses and computation are unchanged by the reporting adapter.
- Optional `limits` and action/final-reserve budgets appropriate to actual model
  startup and measurement cost. Do not shrink them after search begins.
  The client timeout must exceed the complete fixed AgentX replay duration;
  materialization rejects a shorter or equal budget before starting GPU work.
  Leave additional time for startup and native replay drain.
- Optional `measurement_num_prompts` can **increase** the native text/media
  sample count during baseline preparation. This makes tiny profiling examples
  usable for a stable measurement window. It cannot reduce the original count,
  change per-request lengths/concurrency/engine settings, or change after search
  begins. `provenance.json` records both counts; baseline and candidate use the
  same prepared count. Media request manifests must cover that complete count.
  Canonical trace replay keeps its original duration instead of this override.
- Text workloads require `client_runtime`, prepared before binding. Use
  `{kind: image, executable: /opt/client/bin/python}` when the pinned image
  already supplies the native client's Python and complete dependencies.
  Otherwise use `{kind: archive, asset: client-python.tar.gz,
  executable: python/bin/python3.12}` and declare that relocatable Python/package
  archive in `assets` with its SHA-256 and either a public HTTPS URL or
  `source: provided`. For offline supplied files, pass
  `--asset-file client-python.tar.gz=PATH` to `src.prepare_serving_runtime`.
  The host path stays in preparation input, never the task or runtime lock.
  Every host needs that declared file or an already verified cache; a missing
  file fails before GPU actions. The host acquires and verifies it;
  each clean action extracts it locally before measurement. The archive must
  include the complete environment, not an installer or an unfrozen requirements
  file. The native client receives `MAGPIE_BENCHMARK_PYTHON`; package indexes and
  automatic interpreter downloads are disabled. Missing dependencies fail rather
  than bootstrap during scoring. Full client sources and extracted runtime
  contents are checked before/after execution. This setup changes no requests,
  model computation or serving flags.
  A report-only hook captures the native generator's actual request bytes
  (hashed) and token lengths before its HTTP timer. A separate preparation call
  runs the same generator without sending requests. Both roles must match that
  complete plan, including any native decode/re-encode length variation; nominal
  ISL alone is not an exact realized token count. Missing plans, changed payloads,
  missing requests and incorrect measured lengths fail closed.
- A fixed `server_seed` for SGLang. An explicit upstream seed is preserved;
  otherwise materialization freezes 1729 before either role starts. The shared
  launcher supplies it to both roles, rather than ignoring seed differences in
  runtime fingerprints. A conflicting explicit seed is rejected.
  Fingerprints retain effective compute, precision and parallelism settings.
  Raw rank evidence also retains derived compile times, native source-cache keys
  and ATOM-generated rendezvous addresses/ports; those observations are separated
  from configured switches so a fresh server or changed source can be compared.
- `server_topology` for original client-only recipes; `runner_type` when hardware
  is not declared upstream. Old native scripts that cannot split server/client
  phases require an explicit `server_launch` with literal `argv`, `env` and
  `health_path`, prepared from that native recipe. Their measurement uses the
  common framework client. This adaptation must be reviewed during qualification.

Complete script/source bindings from clean, pinned upstream checkouts, then
materialize a selected workload under the normal task root:

```bash
python3 -m src.magpie_catalog bind \
  --id benchmark_vllm_qwen3_next_80b_fp8 \
  --profile /tmp/runtime-profile.json \
  --magpie ../Magpie --inferencex ../InferenceX \
  --output /tmp/runtime-binding.json
python3 -m src.magpie_catalog materialize \
  --id benchmark_vllm_qwen3_next_80b_fp8 \
  --binding /tmp/runtime-binding.json \
  --output tasks/e2e/magpie/benchmark_vllm_qwen3_next_80b_fp8
```

Both commands refuse to overwrite an existing binding/package. Select the
resulting `e2e/magpie/benchmark_vllm_qwen3_next_80b_fp8` task in a run config,
prepare immutable assets with `src.prepare_serving_runtime`, and use the normal
Docker runner and `task_validator`. A generated package is self-contained after
materialization. Keep experiment-specific bindings/packages untracked unless
intentionally contributing a maintained, GPU-qualified task.

### Media and trace bindings

Multimodal profiles additionally declare a `request_manifest` with immutable
`dataset_sha256` plus complete `input_lens`/`output_lens` from the pinned native
client's unmodified preparation run. The native client saves detailed per-request
lengths; this reporting adaptation does not change serving settings. The protected
`inputs.media_checks.json` contains image-bearing chat requests with inline data
URIs and fixed decode lengths. Set `correctness.media_sha256` to those exact JSON
bytes and use `production_serving`: both numerical/logprob and no-logprob paths
must match fresh original-server continuations. This strict policy must pass
unmodified baseline calibration; media models with divergent normal continuations
need a separately qualified numerical policy, not an answer-only bypass. Text-only
probes do not qualify image kernels.

AgentX profiles contain the resolved canonical `agentx.launch_spec` (literal
`argv`, `env`, `health_path`), its `recipe_fingerprint`, and `agentx.client`
(literal `argv` and `env`) captured from the pinned native replay preparation.
Readiness requires a successful HTTP status at `health_path`; empty or plain-text
health responses are valid. Inference responses still require their complete
structured outputs, numerical checks and runtime identity evidence.
The client must run the `inferencex-agentx-mvp` scenario for the canonical duration,
not fast/diagnostic mode. Use `{model}`, `{client}`, `{root}`, `{port}` and
`{assets}` placeholders for evaluator-owned paths/endpoints. Preinstall client
requirements in the immutable image: scoring does not bootstrap packages or
follow moving dataset revisions. Declare the exact AIPerf source in `dependencies`
and offline trace files in `assets` as relative names with SHA-256 and either a
public HTTPS URL or `source: provided` (host-only `--asset-file`); these
are acquired on the host, content-verified and mounted read-only. The original
Magpie/InferenceX resolved recipe and client configuration remain review evidence.
An optional prepared `client_runtime` also supports `{client_python}` in the
canonical replay argv, for example a pinned runtime running `-m aiperf`.

The AgentX `request_manifest` fixes `dataset_sha256`, `duration_s` and trace turns
(`requests`: unique `id` as `conversation-id:turn-index`, exact `input_tokens` and
`output_tokens` by default). If `input_tokens` instead counts the native loader's
reconstruction budget, bind `input_token_policy: native_reconstruction` before
qualification. The pinned loader decodes synthetic token blocks into chat text;
BPE re-encoding and chat formatting can produce either fewer or more wire tokens.
These are different count domains, not an error tolerance. This explicit policy
requires `--use-server-token-count`, checks native usage against both reported
input/output sequence lengths, and keeps output lengths exact. Results preserve
the nominal reconstruction budgets separately from actual wire input counts.
The protected corpus, native client, configuration, accepted native scenario
verdict and absence of skipped/error work bind request construction; wire counts
alone do not prove that the HTTP payloads match the intended corpus. Independent
review must retain this coverage limit. Never apply this policy to excuse modified
client inputs or truncate requests. Keep the default exact policy when the
manifest describes complete wire requests. Qualify the policy against the fixed
client and tokenizer before search. The policy is protected and included in
the measurement identity. Preserve the complete native conversation ID, including
subagent and flattened-branch suffixes such as `::sa:child` and `::fa:001`;
these identify distinct agents, not interchangeable replay sessions.
The complete turn list stays in the protected runtime lock. Action results bind
its SHA-256 and turn count instead of repeating the whole corpus in every
measurement and review record; exact per-request checks still use that list.
`correctness.trace_probe_indices` selects protected serial trace shapes for
model numerical checks. `correctness.trace_batch_indices` selects one additional
complete batch with the scored session concurrency; each index fixes a trace
turn's input and output lengths. Freeze this selection before baseline
calibration and search. Both logprob and ordinary no-logprob generation exercise
the concurrent batch and compare against a fresh original server. These probes
sample model computation; they do not validate every stateful trajectory or
every output in the timed replay. Independent candidate review must retain that
coverage limit. All replayed profiling records are checked against
the turn manifest, with exact integer token counts and no failed or cancelled
requests. A completed record must explicitly report `was_cancelled: false`;
matching token counts cannot turn a cancelled request into completed work.
Native end-of-window credit cancellations do not become completed requests;
preserve their shutdown logs and reconcile the accepted native export with all
completed rows. Throughput uses the native observation start/end timestamps,
including initial idle time and terminal grace/drain time, rather than only the
first-request/last-response span. Native token totals and throughput must agree
with the independently summed completed records and that observation window.
Native aggregate completion/error counters must agree with those records; missing
rows and context-overflow skips cannot silently change the measured workload.
The native export must also identify the canonical scenario and explicitly mark
the submission valid, without invalid-reason tags; matching counters alone cannot
override a rejected native run.
Warmup records remain diagnostic. Fixed-duration replay intentionally permits
more completed requests when faster; the corpus, replay command, session
concurrency and measurement duration must remain the same. No token total is
inferred from a rounded throughput field. Native request records are retained.

Catalog import, task materialization, baseline calibration, actual candidate
acceptance and performance gain are separate outcomes. Representative GPU
validation of shared paths does not qualify every catalog entry. Any unsupported
new upstream framework/scenario fails visibly during import or binding; do not
silently replace it with a different workload. The separate candidate verifier
remains mandatory for every serving candidate, with failed or inconclusive
review blocking acceptance.
